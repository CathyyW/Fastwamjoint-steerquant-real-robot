# Copyright (c) 2026 Rockylix. MIT license; see LICENSE in this directory.
# Adapted from piper-cosmos3-edge d984db2; see docs/PORTABLE_REAL_ROBOT_CLIENT.md.
"""Asynchronous RTC runtime for a persistent Piper14 policy server.

The policy RPC remains synchronous.  It runs exclusively in one producer
thread while the caller's thread executes actions from the RTC chunk buffer.
"""

from __future__ import annotations

import json
import os
import select
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .buffer import (
    ActionFusionSnapshot,
    ActionSink,
    ObservationSource,
    PolicyClient,
    RealTimeChunkingBuffer,
    summarize_array,
    summarize_numbers,
)


class RetryableActionSafetyError(RuntimeError):
    """Action-dependent safety rejection that a fresher chunk may correct."""


class _RuntimeStopRequested(Exception):
    """Internal control-flow signal for a requested runtime stop."""


class ContinuousActionGovernor:
    """Shape asynchronous RTC targets without weakening the final safety sink.

    The governor is intentionally optional and is only attached by the
    asynchronous real-robot entrypoint.  It records raw-policy distribution
    excursions for diagnostics, projects every finite target into the robot's
    safe command envelope, and rate-limits each dimension relative to the last
    action that the sink accepted.
    """

    def __init__(
        self,
        *,
        initial_action: np.ndarray,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        max_step_delta: np.ndarray,
        bound_margin: np.ndarray,
        raw_action_lower_bounds: np.ndarray,
        raw_action_upper_bounds: np.ndarray,
        raw_action_bound_tolerance: np.ndarray,
    ) -> None:
        self.previous_action = self._vector(initial_action, "initial_action")
        self.initial_action = self.previous_action.copy()
        self.lower_bounds = self._vector(lower_bounds, "lower_bounds")
        self.upper_bounds = self._vector(upper_bounds, "upper_bounds")
        self.max_step_delta = self._vector(max_step_delta, "max_step_delta")
        self.bound_margin = self._vector(bound_margin, "bound_margin")
        self.raw_action_lower_bounds = self._vector(
            raw_action_lower_bounds,
            "raw_action_lower_bounds",
        )
        self.raw_action_upper_bounds = self._vector(
            raw_action_upper_bounds,
            "raw_action_upper_bounds",
        )
        self.raw_action_bound_tolerance = self._vector(
            raw_action_bound_tolerance,
            "raw_action_bound_tolerance",
        )
        self.action_dim = int(self.previous_action.size)
        for name, value in (
            ("lower_bounds", self.lower_bounds),
            ("upper_bounds", self.upper_bounds),
            ("max_step_delta", self.max_step_delta),
            ("bound_margin", self.bound_margin),
            ("raw_action_lower_bounds", self.raw_action_lower_bounds),
            ("raw_action_upper_bounds", self.raw_action_upper_bounds),
            ("raw_action_bound_tolerance", self.raw_action_bound_tolerance),
        ):
            if value.size != self.action_dim:
                raise ValueError(
                    f"{name} must contain {self.action_dim} values, got {value.size}"
                )
        if np.any(self.lower_bounds >= self.upper_bounds):
            raise ValueError("lower_bounds must be strictly below upper_bounds")
        if np.any(self.max_step_delta <= 0):
            raise ValueError("max_step_delta must be positive")
        if np.any(self.bound_margin < 0):
            raise ValueError("bound_margin must be non-negative")
        if np.any(
            self.raw_action_lower_bounds >= self.raw_action_upper_bounds
        ):
            raise ValueError(
                "raw_action_lower_bounds must be strictly below "
                "raw_action_upper_bounds"
            )
        if np.any(self.raw_action_bound_tolerance < 0):
            raise ValueError("raw_action_bound_tolerance must be non-negative")
        self.raw_action_safe_lower_bounds = (
            self.raw_action_lower_bounds - self.raw_action_bound_tolerance
        )
        self.raw_action_safe_upper_bounds = (
            self.raw_action_upper_bounds + self.raw_action_bound_tolerance
        )
        self.safe_lower_bounds = self.lower_bounds + self.bound_margin
        self.safe_upper_bounds = self.upper_bounds - self.bound_margin
        if np.any(self.safe_lower_bounds > self.safe_upper_bounds):
            raise ValueError("bound_margin leaves an empty action interval")
        if np.any(self.previous_action < self.lower_bounds) or np.any(
            self.previous_action > self.upper_bounds
        ):
            raise ValueError("initial_action must be within the hard action bounds")

        self.reset()

    @staticmethod
    def _vector(value: np.ndarray, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if array.size == 0:
            raise ValueError(f"{name} must not be empty")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain only finite values")
        return np.array(array, dtype=np.float32, copy=True, order="C")

    def shape(self, action: np.ndarray, *, t: int) -> np.ndarray:
        """Return a bounded, rate-limited proposal without changing state."""
        raw_action = self._vector(action, "action")
        if raw_action.size != self.action_dim:
            raise ValueError(
                f"action must contain {self.action_dim} values, got {raw_action.size}"
            )

        bounded_target = np.clip(
            raw_action,
            self.safe_lower_bounds,
            self.safe_upper_bounds,
        )
        target_delta = bounded_target - self.previous_action
        shaped_delta = np.clip(
            target_delta,
            -self.max_step_delta,
            self.max_step_delta,
        )
        shaped_action = self.previous_action + shaped_delta
        # Both endpoints are inside the hardware interval, so this is only a
        # numerical guard.  Raw distribution excursions are diagnostic and
        # cannot bypass this projection or the final action sink.
        shaped_action = np.clip(
            shaped_action,
            self.lower_bounds,
            self.upper_bounds,
        )
        return np.ascontiguousarray(shaped_action.astype(np.float32, copy=False))

    def reset(self) -> None:
        self.previous_action = self.initial_action.copy()
        self.shaped_steps = 0
        self.slew_limited_steps = 0
        self.bound_projected_steps = 0
        self.raw_out_of_distribution_steps = 0
        self.raw_ood_dimension_counts = np.zeros(
            self.action_dim,
            dtype=np.int64,
        )
        self.max_raw_distribution_excess = 0.0
        self.max_raw_delta = 0.0
        self.max_sent_delta = 0.0

    def commit(
        self,
        *,
        raw_action: np.ndarray,
        shaped_action: np.ndarray,
        t: int,
    ) -> None:
        """Advance the governor only after the final sink accepted the action."""
        raw_action = self._vector(raw_action, "raw_action")
        shaped_action = self._vector(shaped_action, "shaped_action")
        bounded_target = np.clip(
            raw_action,
            self.safe_lower_bounds,
            self.safe_upper_bounds,
        )
        raw_delta = float(np.max(np.abs(raw_action - self.previous_action)))
        sent_delta = float(np.max(np.abs(shaped_action - self.previous_action)))
        bound_projected = not np.array_equal(raw_action, bounded_target)
        lower_excess = np.maximum(
            self.raw_action_safe_lower_bounds - raw_action,
            0.0,
        )
        upper_excess = np.maximum(
            raw_action - self.raw_action_safe_upper_bounds,
            0.0,
        )
        distribution_excess = np.maximum(lower_excess, upper_excess)
        raw_ood_mask = distribution_excess > 0.0
        raw_out_of_distribution = bool(np.any(raw_ood_mask))
        slew_limited = not np.allclose(
            shaped_action,
            bounded_target,
            rtol=0.0,
            atol=1e-7,
        )

        self.shaped_steps += 1
        self.bound_projected_steps += int(bound_projected)
        self.raw_out_of_distribution_steps += int(raw_out_of_distribution)
        self.raw_ood_dimension_counts += raw_ood_mask.astype(np.int64)
        self.max_raw_distribution_excess = max(
            self.max_raw_distribution_excess,
            float(distribution_excess.max()),
        )
        self.slew_limited_steps += int(slew_limited)
        self.max_raw_delta = max(self.max_raw_delta, raw_delta)
        self.max_sent_delta = max(self.max_sent_delta, sent_delta)
        self.previous_action = np.ascontiguousarray(shaped_action.copy())

        if bound_projected or slew_limited or raw_out_of_distribution:
            raw_ood_indices = np.flatnonzero(raw_ood_mask).tolist()
            print(
                "[wam-async-governor] "
                f"t={t} bound_projected={bound_projected} "
                f"slew_limited={slew_limited} "
                f"raw_out_of_distribution={raw_out_of_distribution} "
                f"raw_ood_indices={raw_ood_indices} "
                f"raw_linf={raw_delta:.4f} "
                f"sent_linf={sent_delta:.4f}",
                flush=True,
            )

    def report(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "initial_action": self.initial_action.tolist(),
            "lower_bounds": self.lower_bounds.tolist(),
            "upper_bounds": self.upper_bounds.tolist(),
            "max_step_delta": self.max_step_delta.tolist(),
            "bound_margin": self.bound_margin.tolist(),
            "raw_reference_lower_bounds": self.raw_action_lower_bounds.tolist(),
            "raw_reference_upper_bounds": self.raw_action_upper_bounds.tolist(),
            "shaped_steps": int(self.shaped_steps),
            "slew_limited_steps": int(self.slew_limited_steps),
            "bound_projected_steps": int(self.bound_projected_steps),
            "raw_out_of_distribution_steps": int(
                self.raw_out_of_distribution_steps
            ),
            "raw_ood_dimension_counts": (
                self.raw_ood_dimension_counts.astype(int).tolist()
            ),
            "max_raw_distribution_excess": float(
                self.max_raw_distribution_excess
            ),
            "max_raw_delta": float(self.max_raw_delta),
            "max_sent_delta": float(self.max_sent_delta),
        }


@dataclass(frozen=True)
class AsyncJointRTCRuntimeConfig:
    action_dim: int = 14
    chunk_size: int = 32
    control_hz: float = 40.0
    max_steps: int = 100
    exp_weight_factor: float = 0.2
    arrival_smoothstep_steps: int = 5
    prompt: str | None = None
    startup_timeout_s: float = 15.0
    action_wait_timeout_s: float = 0.2
    fresh_chunk_grace_s: float = 0.6
    producer_poll_s: float = 0.002
    producer_join_timeout_s: float = 16.0
    min_latency_headroom_steps: float = 1.0
    sleep: bool = True
    debug: bool = False
    timestamp_debug: bool = False
    stop_on_spacebar: bool = False


@dataclass(frozen=True)
class InferenceChunkRecord:
    inference_id: int
    probe_cursor: int
    cursor: int
    arrival_control_t: int
    observation_started_monotonic_ns: int
    observation_finished_monotonic_ns: int
    inference_started_monotonic_ns: int
    inference_finished_monotonic_ns: int
    inference_started_wall_ns: int
    inference_finished_wall_ns: int
    latency_s: float
    camera_timestamps_s: tuple[float, ...]
    observation_state: np.ndarray
    chunk: np.ndarray

    @property
    def age_steps(self) -> int:
        return max(0, self.arrival_control_t - self.cursor)

    @property
    def observation_cursor_drift_steps(self) -> int:
        return self.cursor - self.probe_cursor


@dataclass(frozen=True)
class ActionTraceRecord:
    control_t: int
    step_started_monotonic_ns: int
    sent_monotonic_ns: int
    sent_wall_ns: int
    actual_dt_s: float
    selection_mode: str
    revision: int
    cursors: tuple[int, ...]
    arrival_control_t: tuple[int, ...]
    counter_local_indices: tuple[int, ...]
    wall_time_local_indices: tuple[int, ...]
    ramps: tuple[float, ...]
    weights: tuple[float, ...]
    candidate_actions: np.ndarray
    fused_raw_action: np.ndarray
    sent_action: np.ndarray


class AsyncJointRTCRuntime:
    """Run policy production and action consumption on separate threads."""

    def __init__(
        self,
        *,
        policy: PolicyClient,
        observation_source: ObservationSource,
        action_sink: ActionSink,
        config: AsyncJointRTCRuntimeConfig | None = None,
        action_governor: ContinuousActionGovernor | None = None,
    ) -> None:
        self.policy = policy
        self.observation_source = observation_source
        self.action_sink = action_sink
        self.config = config or AsyncJointRTCRuntimeConfig()
        self.action_governor = action_governor
        self._validate_config()
        self.buffer = RealTimeChunkingBuffer(
            chunk_size=self.config.chunk_size,
            exp_weight_factor=self.config.exp_weight_factor,
            arrival_smoothstep_steps=self.config.arrival_smoothstep_steps,
            debug=self.config.debug,
        )
        self.stop_event = threading.Event()
        self.first_chunk_event = threading.Event()
        self.chunk_ready_event = threading.Event()
        self.producer_thread: threading.Thread | None = None
        self.keyboard_thread: threading.Thread | None = None
        self.producer_error: BaseException | None = None
        self._stop_lock = threading.Lock()
        self._stop_reason: str | None = None
        self._first_action_started_monotonic_ns: int | None = None
        self._action_execution_ended_monotonic_ns: int | None = None
        self._previous_inference_latency_s: float | None = None
        self._records_lock = threading.Lock()
        self.selected_actions: list[np.ndarray] = []
        self.inference_chunks: list[InferenceChunkRecord] = []
        self.action_trace: list[ActionTraceRecord] = []
        self._inference_records_by_cursor: dict[int, InferenceChunkRecord] = {}
        self._next_inference_id = 0
        self._last_action_sent_monotonic_ns: int | None = None
        self.starved_waits = 0
        self.dropped_stale_chunks = 0
        self.safety_retry_attempts = 0
        self.safety_retry_rejections = 0
        self.safety_retry_successes = 0
        self.safety_retry_timeouts = 0

    def _validate_config(self) -> None:
        if self.config.action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if self.config.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.config.control_hz <= 0:
            raise ValueError("control_hz must be positive")
        if self.config.arrival_smoothstep_steps < 0:
            raise ValueError("arrival_smoothstep_steps must be non-negative")
        if self.config.max_steps < 0:
            raise ValueError("max_steps must be non-negative")
        if self.config.startup_timeout_s <= 0:
            raise ValueError("startup_timeout_s must be positive")
        if self.config.action_wait_timeout_s <= 0:
            raise ValueError("action_wait_timeout_s must be positive")
        if self.config.fresh_chunk_grace_s < 0:
            raise ValueError("fresh_chunk_grace_s must be non-negative")
        if self.config.producer_poll_s < 0:
            raise ValueError("producer_poll_s must be non-negative")
        if self.config.producer_join_timeout_s <= 0:
            raise ValueError("producer_join_timeout_s must be positive")
        if self.config.min_latency_headroom_steps < 0:
            raise ValueError("min_latency_headroom_steps must be non-negative")

    def run(self) -> dict[str, Any]:
        self._reset_run_state()
        if self.config.max_steps == 0:
            return self.report()

        self._start_keyboard_listener()
        self.producer_thread = threading.Thread(
            target=self._producer_loop,
            name="wam-joint-producer",
            daemon=True,
        )
        self.producer_thread.start()
        try:
            if not self.first_chunk_event.wait(self.config.startup_timeout_s):
                raise TimeoutError(
                    "Timed out waiting for the first WAM action chunk after "
                    f"{self.config.startup_timeout_s:.3f}s"
                )
            self._raise_producer_error()
            if self.stop_event.is_set():
                self._join_producer()
                return self.report()
            if self.buffer.get_action(0) is None:
                raise RuntimeError(
                    "Producer signaled readiness without an action at control step 0"
                )
            print(
                "[wam-async-rtc] first action chunk ready; starting control loop",
                flush=True,
            )
            self._control_loop()
            # Freeze production before building the report.  A synchronous
            # policy call may still be in flight when the last control step
            # completes, so join it first instead of allowing records to change
            # after report() has already taken its snapshot.
            self.stop_event.set()
            self.chunk_ready_event.set()
            self._join_producer()
            self._raise_producer_error()
            return self.report()
        finally:
            self.stop_event.set()
            self.chunk_ready_event.set()
            self._join_producer()
            self._join_keyboard_listener()

    def _reset_run_state(self) -> None:
        self.buffer.clear()
        self.stop_event.clear()
        self.first_chunk_event.clear()
        self.chunk_ready_event.clear()
        self.producer_error = None
        self.keyboard_thread = None
        with self._stop_lock:
            self._stop_reason = None
        self._first_action_started_monotonic_ns = None
        self._action_execution_ended_monotonic_ns = None
        self._previous_inference_latency_s = None
        self.selected_actions = []
        with self._records_lock:
            self.inference_chunks = []
            self.action_trace = []
            self._inference_records_by_cursor = {}
        self._next_inference_id = 0
        self._last_action_sent_monotonic_ns = None
        self.starved_waits = 0
        self.dropped_stale_chunks = 0
        self.safety_retry_attempts = 0
        self.safety_retry_rejections = 0
        self.safety_retry_successes = 0
        self.safety_retry_timeouts = 0
        if self.action_governor is not None:
            self.action_governor.reset()

    def _producer_loop(self) -> None:
        try:
            while not self.stop_event.is_set():
                probe_cursor = self.buffer.get_control_time()
                if self.buffer.has_chunk(probe_cursor):
                    self.stop_event.wait(self.config.producer_poll_s)
                    continue

                observation_started_monotonic_ns = time.monotonic_ns()
                observation = dict(
                    self.observation_source.read_observation(probe_cursor)
                )
                observation_finished_monotonic_ns = time.monotonic_ns()
                source_timing = observation.pop("_runtime_timing", {})
                if self.config.prompt is not None:
                    observation["prompt"] = self.config.prompt
                cursor = self.buffer.get_control_time()
                generation = self.buffer.get_generation()
                if self.buffer.has_chunk(cursor):
                    continue

                inference_id = self._next_inference_id
                self._next_inference_id += 1
                inference_started_monotonic_ns = time.monotonic_ns()
                inference_started_wall_ns = time.time_ns()
                observation_state = np.asarray(
                    observation.get(
                        "state",
                        np.zeros(self.config.action_dim, dtype=np.float32),
                    ),
                    dtype=np.float32,
                ).reshape(-1)
                camera_timestamps_s = self._camera_timestamps(source_timing)
                self._print_timestamp_trace(
                    "infer_start",
                    {
                        "infer_id": inference_id,
                        "monotonic_ns": inference_started_monotonic_ns,
                        "wall_ns": inference_started_wall_ns,
                        "probe_cursor": probe_cursor,
                        "cursor": cursor,
                        "observation_cursor_drift_steps": cursor - probe_cursor,
                        "observation_duration_ms": (
                            observation_finished_monotonic_ns
                            - observation_started_monotonic_ns
                        )
                        / 1e6,
                        "camera_timestamps_s": list(camera_timestamps_s),
                        "state": observation_state.tolist(),
                    },
                )
                chunk = np.asarray(self.policy.infer(observation), dtype=np.float32)
                inference_finished_monotonic_ns = time.monotonic_ns()
                inference_finished_wall_ns = time.time_ns()
                latency_s = (
                    inference_finished_monotonic_ns
                    - inference_started_monotonic_ns
                ) / 1e9
                if self.stop_event.is_set():
                    self._print_timestamp_trace(
                        "infer_drop",
                        {
                            "infer_id": inference_id,
                            "cursor": cursor,
                            "latency_ms": latency_s * 1000.0,
                            "reason": "runtime_stopped",
                        },
                    )
                    return
                chunk = self._validate_chunk(chunk, cursor)
                chunk = chunk[: self.config.chunk_size]
                arrival_control_t = self.buffer.get_control_time()

                previous_latency_s = self._previous_inference_latency_s
                if previous_latency_s is None:
                    # The control loop does not start until this first chunk is
                    # ready, so startup/compile latency consumes no buffered
                    # action horizon.  Seed the first online coverage window
                    # with zero rather than the startup wall-clock latency.
                    self._previous_inference_latency_s = 0.0
                else:
                    self._validate_latency_window(
                        previous_latency_s=previous_latency_s,
                        current_latency_s=latency_s,
                    )
                    self._previous_inference_latency_s = float(latency_s)

                if arrival_control_t >= cursor + len(chunk):
                    self.dropped_stale_chunks += 1
                    self._print_timestamp_trace(
                        "infer_drop",
                        {
                            "infer_id": inference_id,
                            "cursor": cursor,
                            "arrival_control_t": arrival_control_t,
                            "age_steps": arrival_control_t - cursor,
                            "chunk_len": len(chunk),
                            "latency_ms": latency_s * 1000.0,
                            "reason": "expired",
                        },
                        force=self.config.debug,
                    )
                    continue

                record = InferenceChunkRecord(
                    inference_id=inference_id,
                    probe_cursor=probe_cursor,
                    cursor=cursor,
                    arrival_control_t=arrival_control_t,
                    observation_started_monotonic_ns=(
                        observation_started_monotonic_ns
                    ),
                    observation_finished_monotonic_ns=(
                        observation_finished_monotonic_ns
                    ),
                    inference_started_monotonic_ns=(
                        inference_started_monotonic_ns
                    ),
                    inference_finished_monotonic_ns=(
                        inference_finished_monotonic_ns
                    ),
                    inference_started_wall_ns=inference_started_wall_ns,
                    inference_finished_wall_ns=inference_finished_wall_ns,
                    latency_s=float(latency_s),
                    camera_timestamps_s=camera_timestamps_s,
                    observation_state=np.ascontiguousarray(
                        observation_state.copy()
                    ),
                    chunk=chunk.copy(),
                )
                # Publish timing metadata before making the chunk visible to
                # the control thread, so an immediately selected action can
                # always resolve its cursor back to this inference.
                with self._records_lock:
                    self._inference_records_by_cursor[cursor] = record
                if self.buffer.enqueue(chunk, cursor=cursor, generation=generation):
                    with self._records_lock:
                        self.inference_chunks.append(record)
                    print(
                        "[wam-async-rtc] enqueue "
                        f"infer_id={inference_id} "
                        f"cursor={cursor} arrival_t={arrival_control_t} "
                        f"age={record.age_steps} latency={latency_s * 1000.0:.1f}ms "
                        f"shape={tuple(chunk.shape)}",
                        flush=True,
                    )
                    aligned_index = record.age_steps
                    self._print_timestamp_trace(
                        "infer_return",
                        {
                            "infer_id": inference_id,
                            "monotonic_ns": inference_finished_monotonic_ns,
                            "wall_ns": inference_finished_wall_ns,
                            "cursor": cursor,
                            "arrival_control_t": arrival_control_t,
                            "age_steps": record.age_steps,
                            "latency_ms": latency_s * 1000.0,
                            "chunk_0": chunk[0].tolist(),
                            "chunk_at_arrival": chunk[aligned_index].tolist(),
                        },
                    )
                    self.chunk_ready_event.set()
                    self.first_chunk_event.set()
                else:
                    with self._records_lock:
                        if (
                            self._inference_records_by_cursor.get(cursor)
                            is record
                        ):
                            del self._inference_records_by_cursor[cursor]
        except Exception as exc:  # noqa: BLE001 - thread boundary propagates failures
            self.producer_error = exc
            self.stop_event.set()
            self.first_chunk_event.set()
            self.chunk_ready_event.set()

    def _control_loop(self) -> None:
        period_s = 1.0 / self.config.control_hz
        try:
            for t in range(self.config.max_steps):
                self._raise_producer_error()
                if self.stop_event.is_set():
                    break
                step_started = time.perf_counter()
                step_started_monotonic_ns = time.monotonic_ns()
                self.buffer.set_control_time(t)
                action = self._select_and_send_action(
                    t,
                    step_started_monotonic_ns=step_started_monotonic_ns,
                )
                self.selected_actions.append(action.copy())
                if self.config.sleep:
                    elapsed_s = time.perf_counter() - step_started
                    self.stop_event.wait(max(0.0, period_s - elapsed_s))
        except _RuntimeStopRequested:
            pass
        finally:
            if self._first_action_started_monotonic_ns is not None:
                self._action_execution_ended_monotonic_ns = time.monotonic_ns()
            with self._stop_lock:
                if self._stop_reason is None:
                    self._stop_reason = (
                        "stop_requested"
                        if self.stop_event.is_set()
                        else "max_steps"
                    )

    def _select_and_send_action(
        self,
        t: int,
        *,
        step_started_monotonic_ns: int | None = None,
    ) -> np.ndarray:
        if step_started_monotonic_ns is None:
            step_started_monotonic_ns = time.monotonic_ns()
        snapshot = self._wait_for_action(t)
        assert snapshot.action is not None
        raw_action = snapshot.action
        try:
            action = self._shape_and_send_action(t, raw_action)
            self._record_action_trace(
                t=t,
                step_started_monotonic_ns=step_started_monotonic_ns,
                snapshot=snapshot,
                sent_action=action,
                selection_mode="fused",
            )
            return action
        except RetryableActionSafetyError as initial_error:
            self.safety_retry_attempts += 1
            grace_s = self.config.fresh_chunk_grace_s
            deadline = time.monotonic() + grace_s
            previous_revision = snapshot.revision
            previous_cursors = set(snapshot.cursors)
            last_error = initial_error
            print(
                "[wam-async-rtc] action safety retry "
                f"t={t} grace={grace_s * 1000.0:.1f}ms "
                f"previous_cursors={list(snapshot.cursors)} reason={initial_error}",
                flush=True,
            )
            while True:
                fresh = self._wait_for_fresh_action(
                    t=t,
                    previous_revision=previous_revision,
                    previous_cursors=tuple(previous_cursors),
                    deadline=deadline,
                )
                if fresh is None:
                    raise last_error
                assert fresh.action is not None
                fresh_action = fresh.action
                previous_revision = fresh.revision
                previous_cursors.update(fresh.cursors)
                try:
                    action = self._shape_and_send_action(t, fresh_action)
                except RetryableActionSafetyError as fresh_error:
                    self.safety_retry_rejections += 1
                    last_error = fresh_error
                    print(
                        "[wam-async-rtc] reject fresh chunk during safety "
                        f"retry t={t} cursors={list(fresh.cursors)} "
                        f"reason={fresh_error}",
                        flush=True,
                    )
                    continue

                self.safety_retry_successes += 1
                print(
                    f"[wam-async-rtc] safety retry succeeded at t={t}",
                    flush=True,
                )
                if self.config.sleep:
                    # A long safety hold must not be followed by an immediate
                    # catch-up command. Re-establish one complete control period.
                    self.stop_event.wait(1.0 / self.config.control_hz)
                self._record_action_trace(
                    t=t,
                    step_started_monotonic_ns=step_started_monotonic_ns,
                    snapshot=fresh,
                    sent_action=action,
                    selection_mode="fresh_latest",
                )
                return action

    def _shape_and_send_action(
        self,
        t: int,
        raw_action: np.ndarray,
    ) -> np.ndarray:
        self._raise_if_stop_requested()
        raw_action = self._validate_action(raw_action, t)
        action = raw_action
        if self.action_governor is not None:
            action = self.action_governor.shape(raw_action, t=t)
        action_started_monotonic_ns = time.monotonic_ns()
        self.action_sink.send_action(t, action)
        action_was_executed = bool(
            getattr(self.action_sink, "execute_actions", True)
        )
        if (
            action_was_executed
            and self._first_action_started_monotonic_ns is None
        ):
            self._first_action_started_monotonic_ns = (
                getattr(
                    self.action_sink,
                    "first_action_execution_started_monotonic_ns",
                    None,
                )
                or action_started_monotonic_ns
            )
        # ``control_t`` is the next action index the consumer still owes, not
        # the index of the most recently sent action.  Publishing t + 1 here
        # gives observations captured during the following sleep period the
        # correct cursor and makes a newly enqueued chunk start at smoothstep
        # phase 1 on its first selectable control step.
        self.buffer.set_control_time(t + 1)
        if self.action_governor is not None:
            self.action_governor.commit(
                raw_action=raw_action,
                shaped_action=action,
                t=t,
            )
        return action

    def _wait_for_action(
        self,
        t: int,
    ) -> ActionFusionSnapshot:
        deadline = time.monotonic() + self.config.action_wait_timeout_s
        waited = False
        while True:
            self._raise_producer_error()
            self._raise_if_stop_requested()
            snapshot = self.buffer.get_action_fusion_snapshot(t)
            if snapshot.action is not None:
                if waited:
                    print(
                        f"[wam-async-rtc] action stream resumed at t={t}",
                        flush=True,
                    )
                return snapshot
            if not waited:
                self.starved_waits += 1
                waited = True
                print(
                    f"[wam-async-rtc] waiting for action at t={t}",
                    flush=True,
                )
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                raise TimeoutError(
                    f"No RTC action available at control step {t} for "
                    f"{self.config.action_wait_timeout_s:.3f}s; stopping safely"
                )
            self.chunk_ready_event.clear()
            if self.buffer.get_action(t) is not None:
                continue
            self.chunk_ready_event.wait(remaining_s)

    def _wait_for_fresh_action(
        self,
        *,
        t: int,
        previous_revision: int,
        previous_cursors: tuple[int, ...],
        deadline: float,
    ) -> ActionFusionSnapshot | None:
        while True:
            self._raise_producer_error()
            self._raise_if_stop_requested()
            snapshot = self.buffer.get_action_fusion_snapshot(
                t,
                latest_only=True,
            )
            if (
                snapshot.action is not None
                and snapshot.revision > previous_revision
                and snapshot.cursors
                and snapshot.cursors[-1] not in previous_cursors
            ):
                print(
                    "[wam-async-rtc] retry with fresh latest chunk "
                    f"t={t} cursor={snapshot.cursors[-1]} "
                    f"revision={snapshot.revision}",
                    flush=True,
                )
                return snapshot

            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                self.safety_retry_timeouts += 1
                print(
                    "[wam-async-rtc] no fresh chunk arrived during safety "
                    f"retry at t={t}; preserving safety stop",
                    flush=True,
                )
                return None
            self.chunk_ready_event.clear()
            snapshot = self.buffer.get_action_fusion_snapshot(
                t,
                latest_only=True,
            )
            if (
                snapshot.action is not None
                and snapshot.revision > previous_revision
                and snapshot.cursors
                and snapshot.cursors[-1] not in previous_cursors
            ):
                continue
            self.chunk_ready_event.wait(remaining_s)

    @staticmethod
    def _camera_timestamps(source_timing: Any) -> tuple[float, ...]:
        if not isinstance(source_timing, dict):
            return ()
        values = source_timing.get("camera_timestamps_s", {})
        if not isinstance(values, dict):
            return ()
        ordered_keys = ("cam_high", "cam_left_wrist", "cam_right_wrist")
        return tuple(float(values.get(key, np.nan)) for key in ordered_keys)

    def _record_action_trace(
        self,
        *,
        t: int,
        step_started_monotonic_ns: int,
        snapshot: ActionFusionSnapshot,
        sent_action: np.ndarray,
        selection_mode: str,
    ) -> None:
        assert snapshot.action is not None
        sent_monotonic_ns = time.monotonic_ns()
        sent_wall_ns = time.time_ns()
        if self._last_action_sent_monotonic_ns is None:
            actual_dt_s = float("nan")
        else:
            actual_dt_s = (
                sent_monotonic_ns - self._last_action_sent_monotonic_ns
            ) / 1e9
        self._last_action_sent_monotonic_ns = sent_monotonic_ns

        wall_time_local_indices: list[int] = []
        with self._records_lock:
            records_by_cursor = dict(self._inference_records_by_cursor)
        for cursor, counter_index in zip(
            snapshot.cursors,
            snapshot.local_indices,
        ):
            inference = records_by_cursor.get(cursor)
            if inference is None:
                wall_time_local_indices.append(-1)
                continue
            if inference.cursor == 0 and inference.arrival_control_t == 0:
                # Startup inference intentionally completes before the control
                # clock begins, so its wall latency must not consume horizon.
                wall_time_local_indices.append(counter_index)
                continue
            elapsed_s = (
                sent_monotonic_ns
                - inference.observation_finished_monotonic_ns
            ) / 1e9
            # ``cursor`` names the next pending action.  If observation capture
            # finishes just after action k, chunk[0] targets k + 1, one control
            # period after that observation.  Remove this one-period phase
            # before comparing the wall clock against t - cursor.
            wall_time_local_indices.append(
                max(
                    0,
                    int(np.rint(elapsed_s * self.config.control_hz)) - 1,
                )
            )

        record = ActionTraceRecord(
            control_t=int(t),
            step_started_monotonic_ns=int(step_started_monotonic_ns),
            sent_monotonic_ns=int(sent_monotonic_ns),
            sent_wall_ns=int(sent_wall_ns),
            actual_dt_s=float(actual_dt_s),
            selection_mode=str(selection_mode),
            revision=int(snapshot.revision),
            cursors=snapshot.cursors,
            arrival_control_t=snapshot.arrival_control_t,
            counter_local_indices=snapshot.local_indices,
            wall_time_local_indices=tuple(wall_time_local_indices),
            ramps=snapshot.ramps,
            weights=snapshot.weights,
            candidate_actions=np.ascontiguousarray(
                snapshot.candidate_actions.copy()
            ),
            fused_raw_action=np.ascontiguousarray(snapshot.action.copy()),
            sent_action=np.ascontiguousarray(
                np.asarray(sent_action, dtype=np.float32).copy()
            ),
        )
        with self._records_lock:
            self.action_trace.append(record)

        self._print_timestamp_trace(
            "action",
            {
                "t": record.control_t,
                "monotonic_ns": record.sent_monotonic_ns,
                "wall_ns": record.sent_wall_ns,
                "actual_dt_ms": (
                    record.actual_dt_s * 1000.0
                    if np.isfinite(record.actual_dt_s)
                    else None
                ),
                "selection_mode": record.selection_mode,
                "revision": record.revision,
                "cursors": list(record.cursors),
                "arrival_control_t": list(record.arrival_control_t),
                "counter_local_indices": list(record.counter_local_indices),
                "wall_time_local_indices": list(
                    record.wall_time_local_indices
                ),
                "index_offsets": [
                    wall_index - counter_index
                    for wall_index, counter_index in zip(
                        record.wall_time_local_indices,
                        record.counter_local_indices,
                    )
                ],
                "ramps": list(record.ramps),
                "weights": list(record.weights),
                "candidate_actions": record.candidate_actions.tolist(),
                "fused_raw_action": record.fused_raw_action.tolist(),
                "sent_action": record.sent_action.tolist(),
            },
        )

    def _print_timestamp_trace(
        self,
        event: str,
        payload: dict[str, Any],
        *,
        force: bool = False,
    ) -> None:
        if not (self.config.timestamp_debug or force):
            return
        json_payload = self._json_safe({"event": event, **payload})
        print(
            "[cosmos-time-trace] "
            + json.dumps(
                json_payload,
                allow_nan=False,
                separators=(",", ":"),
            ),
            flush=True,
        )

    @classmethod
    def _json_safe(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): cls._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._json_safe(item) for item in value]
        if isinstance(value, (float, np.floating)):
            return float(value) if np.isfinite(value) else None
        if isinstance(value, (int, np.integer)):
            return int(value)
        return value

    def _validate_latency_window(
        self,
        *,
        previous_latency_s: float,
        current_latency_s: float,
    ) -> None:
        # Chunk N+1 starts after chunk N returns.  It must return before chunk
        # N's horizon expires, so the actual coverage window is L_n + L_n+1.
        # Doubling only the current sample incorrectly turns one latency spike
        # into two consecutive spikes and can stop a stream that still has
        # sufficient buffered actions.
        required_steps = (
            previous_latency_s + current_latency_s
        ) * self.config.control_hz
        available_steps = (
            self.config.chunk_size - self.config.min_latency_headroom_steps
        )
        if required_steps > available_steps:
            raise RuntimeError(
                "Consecutive inference latencies cannot safely sustain this RTC "
                "schedule: "
                f"({previous_latency_s:.3f}s + {current_latency_s:.3f}s) * "
                f"{self.config.control_hz:.3f}Hz "
                f"= {required_steps:.2f} steps > {available_steps:.2f} available "
                "steps after headroom"
            )

    def _validate_chunk(self, chunk: np.ndarray, cursor: int) -> np.ndarray:
        if chunk.ndim != 2 or chunk.shape[1] != self.config.action_dim:
            raise ValueError(
                f"Expected action chunk [T,{self.config.action_dim}], "
                f"got {chunk.shape} at cursor {cursor}"
            )
        if chunk.shape[0] == 0:
            raise ValueError(f"Expected non-empty action chunk at cursor {cursor}")
        if not np.isfinite(chunk).all():
            raise ValueError(
                f"Policy returned non-finite action chunk at cursor {cursor}"
            )
        return np.ascontiguousarray(chunk.astype(np.float32, copy=False))

    def _validate_action(self, action: np.ndarray, t: int) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.config.action_dim,):
            raise ValueError(
                f"Expected selected action [{self.config.action_dim}], "
                f"got {action.shape} at step {t}"
            )
        if not np.isfinite(action).all():
            raise ValueError(f"RTC selected non-finite action at step {t}")
        return np.ascontiguousarray(action)

    def _raise_producer_error(self) -> None:
        if self.producer_error is not None:
            raise RuntimeError(
                "WAM inference producer failed"
            ) from self.producer_error

    def _raise_if_stop_requested(self) -> None:
        if self.stop_event.is_set():
            raise _RuntimeStopRequested

    def request_stop(self, reason: str = "external") -> None:
        """Request a safe stop from another thread."""
        with self._stop_lock:
            if self._stop_reason is None:
                self._stop_reason = str(reason)
        self.stop_event.set()
        self.first_chunk_event.set()
        self.chunk_ready_event.set()

    def _start_keyboard_listener(self) -> None:
        if not self.config.stop_on_spacebar:
            return
        try:
            stdin_fd = sys.stdin.fileno()
        except (AttributeError, OSError):
            print(
                "[wam-async-rtc] stdin has no file descriptor; "
                "spacebar stop is unavailable.",
                flush=True,
            )
            return
        if not os.isatty(stdin_fd):
            print(
                "[wam-async-rtc] stdin is not a terminal; "
                "spacebar stop is unavailable.",
                flush=True,
            )
            return
        self.keyboard_thread = threading.Thread(
            target=self._spacebar_listener_loop,
            args=(stdin_fd,),
            name="wam-joint-keyboard",
            daemon=True,
        )
        self.keyboard_thread.start()

    def _spacebar_listener_loop(self, stdin_fd: int) -> None:
        import termios
        import tty

        original_terminal = None
        try:
            original_terminal = termios.tcgetattr(stdin_fd)
            tty.setcbreak(stdin_fd)
            print(
                "[wam-async-rtc] Press Space to stop safely.",
                flush=True,
            )
            while not self.stop_event.is_set():
                readable, _, _ = select.select([stdin_fd], [], [], 0.1)
                if not readable:
                    continue
                key = os.read(stdin_fd, 1)
                if not key:
                    return
                if key == b" ":
                    print(
                        "[wam-async-rtc] Space pressed; stopping safely.",
                        flush=True,
                    )
                    self.request_stop("spacebar")
                    return
        except (OSError, termios.error) as exc:
            print(
                f"[wam-async-rtc] spacebar stop unavailable: {exc}",
                flush=True,
            )
        finally:
            if original_terminal is not None:
                try:
                    termios.tcsetattr(
                        stdin_fd,
                        termios.TCSADRAIN,
                        original_terminal,
                    )
                except (OSError, termios.error):
                    pass

    def _join_producer(self) -> None:
        if self.producer_thread is None:
            return
        self.producer_thread.join(timeout=self.config.producer_join_timeout_s)
        if self.producer_thread.is_alive():
            raise RuntimeError(
                "WAM inference producer did not stop within "
                f"{self.config.producer_join_timeout_s:.3f}s"
            )

    def _join_keyboard_listener(self) -> None:
        if self.keyboard_thread is None:
            return
        self.keyboard_thread.join(timeout=1.0)
        if self.keyboard_thread.is_alive():
            print(
                "[wam-async-rtc] keyboard listener did not stop promptly.",
                flush=True,
            )

    def report(self) -> dict[str, Any]:
        actions = (
            np.stack(self.selected_actions, axis=0)
            if self.selected_actions
            else np.zeros((0, self.config.action_dim), dtype=np.float32)
        )
        with self._records_lock:
            records = list(self.inference_chunks)
            action_trace = list(self.action_trace)
        finite_action_dt = [
            record.actual_dt_s
            for record in action_trace
            if np.isfinite(record.actual_dt_s)
        ]
        index_offsets = [
            float(wall_index - counter_index)
            for record in action_trace
            for wall_index, counter_index in zip(
                record.wall_time_local_indices,
                record.counter_local_indices,
            )
            if wall_index >= 0
        ]
        action_execution_elapsed_s = None
        if (
            self._first_action_started_monotonic_ns is not None
            and self._action_execution_ended_monotonic_ns is not None
        ):
            action_execution_elapsed_s = (
                self._action_execution_ended_monotonic_ns
                - self._first_action_started_monotonic_ns
            ) / 1e9
        return {
            "mode": "async_single_producer",
            "steps": len(self.selected_actions),
            "stop_reason": self._stop_reason,
            "action_execution_elapsed_s": action_execution_elapsed_s,
            "num_inferences": len(records),
            "starved_waits": int(self.starved_waits),
            "dropped_stale_chunks": int(self.dropped_stale_chunks),
            "safety_retry_attempts": int(self.safety_retry_attempts),
            "safety_retry_rejections": int(self.safety_retry_rejections),
            "safety_retry_successes": int(self.safety_retry_successes),
            "safety_retry_timeouts": int(self.safety_retry_timeouts),
            "control_hz": float(self.config.control_hz),
            "chunk_size": int(self.config.chunk_size),
            "exp_weight_factor": float(self.config.exp_weight_factor),
            "arrival_smoothstep_steps": int(
                self.config.arrival_smoothstep_steps
            ),
            "timestamp_debug": bool(self.config.timestamp_debug),
            "selected_actions": summarize_array(actions),
            "inference_latency_s": summarize_numbers(
                [record.latency_s for record in records]
            ),
            "chunk_age_steps": summarize_numbers(
                [float(record.age_steps) for record in records]
            ),
            "observation_cursor_drift_steps": summarize_numbers(
                [
                    float(record.observation_cursor_drift_steps)
                    for record in records
                ]
            ),
            "actual_action_period_s": summarize_numbers(finite_action_dt),
            "wall_vs_counter_index_offset_steps": summarize_numbers(
                index_offsets
            ),
            "action_governor": (
                self.action_governor.report()
                if self.action_governor is not None
                else {"enabled": False}
            ),
        }

    def inference_chunk_records(self) -> list[InferenceChunkRecord]:
        with self._records_lock:
            return list(self.inference_chunks)

    def action_trace_records(self) -> list[ActionTraceRecord]:
        with self._records_lock:
            return list(self.action_trace)
