# Copyright (c) 2026 Rockylix. MIT license; see LICENSE in this directory.
"""RTC fusion primitives adapted from piper-cosmos3-edge; see docs/PORTABLE_REAL_ROBOT_CLIENT.md."""
from __future__ import annotations
import threading
from dataclasses import dataclass
from typing import Any, Protocol, Mapping
import numpy as np

class PolicyClient(Protocol):
    def infer(self, obs: Mapping[str, Any]) -> np.ndarray:
        """Return an action chunk with shape [T, action_dim]."""


class ObservationSource(Protocol):
    def read_observation(self, t: int) -> Mapping[str, Any]:
        """Read one RTC observation for control step ``t``."""


class ActionSink(Protocol):
    def send_action(self, t: int, action: np.ndarray) -> None:
        """Consume one selected action for control step ``t``."""


@dataclass(frozen=True)
class _BufferedActionChunk:
    actions: np.ndarray
    arrival_control_t: int


@dataclass(frozen=True)
class ActionFusionSnapshot:
    """One atomic view of the candidates used for an RTC action."""

    action: np.ndarray | None
    revision: int
    cursors: tuple[int, ...]
    arrival_control_t: tuple[int, ...]
    local_indices: tuple[int, ...]
    ramps: tuple[float, ...]
    weights: tuple[float, ...]
    candidate_actions: np.ndarray


class RealTimeChunkingBuffer:
    """Thread-safe buffer that fuses overlapping action chunks online."""

    def __init__(
        self,
        chunk_size: int,
        exp_weight_factor: float = 0.5,
        arrival_smoothstep_steps: int = 0,
        debug: bool = False,
    ):
        if int(chunk_size) <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if int(arrival_smoothstep_steps) < 0:
            raise ValueError(
                "arrival_smoothstep_steps must be non-negative, got "
                f"{arrival_smoothstep_steps}"
            )
        self.chunk_size = int(chunk_size)
        self.exp_weight_factor = float(exp_weight_factor)
        self.arrival_smoothstep_steps = int(arrival_smoothstep_steps)
        self.debug = bool(debug)
        self.control_t = 0
        self.chunks: dict[int, _BufferedActionChunk] = {}
        self.generation = 0
        self.revision = 0
        self.lock = threading.Lock()

    def clear(self) -> None:
        with self.lock:
            self.control_t = 0
            self.chunks = {}
            self.generation += 1
            self.revision += 1

    def set_control_time(self, control_t: int) -> None:
        with self.lock:
            self.control_t = int(control_t)

    def get_control_time(self) -> int:
        with self.lock:
            return self.control_t

    def get_generation(self) -> int:
        with self.lock:
            return self.generation

    def get_revision(self) -> int:
        with self.lock:
            return self.revision

    def has_chunk(self, cursor: int) -> bool:
        with self.lock:
            return int(cursor) in self.chunks

    def enqueue(self, chunk: np.ndarray, cursor: int, generation: int | None = None) -> bool:
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim != 2:
            raise ValueError(f"Expected action chunk shape [T,D], got {chunk.shape}")
        cursor = int(cursor)
        with self.lock:
            if generation is not None and int(generation) != self.generation:
                if self.debug:
                    print(
                        f"[rtc-buffer] drop stale chunk cursor={cursor} "
                        f"generation={generation} current={self.generation}"
                    )
                return False
            self.chunks[cursor] = _BufferedActionChunk(
                actions=np.ascontiguousarray(chunk),
                arrival_control_t=int(self.control_t),
            )
            self.revision += 1
            return True

    def get_action(self, current_time: int) -> np.ndarray | None:
        action, _, _ = self.get_action_snapshot(current_time)
        return action

    def get_action_snapshot(
        self,
        current_time: int,
        *,
        latest_only: bool = False,
    ) -> tuple[np.ndarray | None, int, tuple[int, ...]]:
        """Return one action plus the enqueue revision and contributing cursors."""
        snapshot = self.get_action_fusion_snapshot(
            current_time,
            latest_only=latest_only,
        )
        return snapshot.action, snapshot.revision, snapshot.cursors

    def get_action_fusion_snapshot(
        self,
        current_time: int,
        *,
        latest_only: bool = False,
    ) -> ActionFusionSnapshot:
        """Return the fused action and the exact candidates/weights behind it."""
        current_time = int(current_time)
        with self.lock:
            relevant: dict[int, tuple[np.ndarray, int]] = {}
            expired: list[int] = []
            before = sorted(self.chunks)
            for cursor, buffered in self.chunks.items():
                chunk = buffered.actions
                end = cursor + len(chunk)
                if (
                    cursor <= current_time < end
                    and buffered.arrival_control_t <= current_time
                ):
                    relevant[cursor] = (
                        chunk[current_time - cursor],
                        buffered.arrival_control_t,
                    )
                elif end <= current_time:
                    expired.append(cursor)
            for cursor in expired:
                del self.chunks[cursor]
            if self.debug:
                print(
                    f"[rtc-buffer] t={current_time} before={before} "
                    f"expired={sorted(expired)} after={sorted(self.chunks)}"
                )
            if not relevant:
                return ActionFusionSnapshot(
                    action=None,
                    revision=self.revision,
                    cursors=(),
                    arrival_control_t=(),
                    local_indices=(),
                    ramps=(),
                    weights=(),
                    candidate_actions=np.zeros((0, 0), dtype=np.float32),
                )
            items = sorted(relevant.items(), key=lambda item: item[0])
            if latest_only:
                items = items[-1:]
            actions = np.asarray(
                [action for _, (action, _) in items],
                dtype=np.float32,
            )
            arrivals = np.asarray(
                [arrival for _, (_, arrival) in items],
                dtype=np.float32,
            )
            cursors = tuple(cursor for cursor, _ in items)
            arrival_control_t = tuple(int(value) for value in arrivals)
            local_indices = tuple(current_time - cursor for cursor in cursors)
            revision = self.revision

        rank_logits = (
            self.exp_weight_factor
            * np.arange(len(actions), dtype=np.float32)
        )
        rank_weights = np.exp(rank_logits - rank_logits.max())
        ramps = np.ones(len(actions), dtype=np.float32)
        if (
            not latest_only
            and len(actions) > 1
            and self.arrival_smoothstep_steps > 0
        ):
            ramp_progress = np.clip(
                (
                    float(current_time)
                    - arrivals
                    + 1.0
                )
                / float(self.arrival_smoothstep_steps),
                0.0,
                1.0,
            )
            ramps = (
                ramp_progress
                * ramp_progress
                * (3.0 - 2.0 * ramp_progress)
            )
        weights = rank_weights * ramps
        weight_sum = float(weights.sum())
        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            raise RuntimeError(
                f"Invalid RTC fusion weights at control step {current_time}"
            )
        weights = (weights / weight_sum)[:, None]
        action = np.ascontiguousarray((actions * weights).sum(axis=0).astype(np.float32))
        if self.debug:
            print(
                f"[rtc-buffer] t={current_time} cursors={list(cursors)} "
                f"arrivals={arrivals.astype(np.int64).tolist()} "
                f"ramps={ramps.tolist()} "
                f"weights={weights[:, 0].tolist()}"
            )
        return ActionFusionSnapshot(
            action=action,
            revision=revision,
            cursors=cursors,
            arrival_control_t=arrival_control_t,
            local_indices=local_indices,
            ramps=tuple(float(value) for value in ramps),
            weights=tuple(float(value) for value in weights[:, 0]),
            candidate_actions=np.ascontiguousarray(actions.copy()),
        )


class RecordingActionSink:
    """Action sink for dry-run RTC tests; it records selected actions only."""

    def __init__(self) -> None:
        self.records: list[tuple[int, np.ndarray]] = []

    def send_action(self, t: int, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float32)
        self.records.append((int(t), np.ascontiguousarray(action.copy())))

    @property
    def actions(self) -> np.ndarray:
        if not self.records:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack([action for _, action in self.records], axis=0).astype(np.float32, copy=False)


def summarize_array(array: np.ndarray) -> dict[str, Any]:
    array = np.asarray(array, dtype=np.float32)
    summary: dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite": bool(np.isfinite(array).all()),
    }
    if array.size:
        summary.update(
            {
                "min": float(array.min()),
                "max": float(array.max()),
                "mean": float(array.mean()),
                "std": float(array.std()),
            }
        )
    return summary


def summarize_numbers(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float32)
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
    }
