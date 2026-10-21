"""No-hardware matched BF16/W4A8 policy, measurement and observation utilities."""
from __future__ import annotations

import hashlib
from pathlib import Path
import time
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from .adapters.real_robot import verify_file
from .robot_async_server import prepare_adapter, PackedRobotPolicy, CAMERAS
from .rollout_graph import LiveJointDiTGraph
from .topology import enumerate_fastwamjoint_linears


def tensor_hash(tensor):
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy()
    return hashlib.sha256(raw.tobytes()).hexdigest()


def observation(spec, config, index):
    with h5py.File(spec["episode_hdf5"], "r") as f:
        return dict(images={k: f[f"observations/images/{k}"][index] for k in CAMERAS},
                    state=np.asarray(f["observations/qpos"][index], np.float32),
                    prompt=config["wam"]["prompt"], image_channel_order="as_stored",
                    observation_id=f'{spec["episode_hdf5"]}#frame={index}')


class BF16RobotPolicy:
    """Original 15k weights and BF16 dtype; live Graph defaults on for fair comparisons."""
    def __init__(self, config, checkpoint, *, cuda_graph=True):
        self.config = config
        self.cuda_graph = cuda_graph
        self.adapter, self.contract = prepare_adapter(config)
        source = Path(checkpoint).expanduser().resolve(strict=True)
        # Identity comes from content hash + embedded step, never a filename.
        verify_file(source, self.adapter.task["source_checkpoint_sha256"])
        payload = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
        if payload.get("step") != 15000 or str(payload.get("torch_dtype")) not in ("torch.bfloat16", "bfloat16"):
            raise ValueError("Baseline checkpoint step/dtype does not match the original BF16 15k model")
        del payload
        self.source = str(source)
        self.adapter.config["assets"]["source_checkpoint"] = self.source
        train = __import__("yaml").safe_load(Path(config["wam"]["train_config"]).read_text())
        self.adapter.config["assets"]["vae_checkpoint"] = str(Path(train["model"]["model_id"]) / "Wan2.2_VAE.pth")
        model = self.adapter.load_model(device=config["wam"]["device"])
        sites = enumerate_fastwamjoint_linears(model)
        if len(sites) != 600 or any(s.module.weight.dtype != torch.bfloat16 for s in sites):
            raise RuntimeError("The original unquantized target Linears must all be BF16")
        self.policy = SimpleNamespace(model=model)
        if cuda_graph:
            LiveJointDiTGraph(model, num_calls=10).install()
        elif getattr(model, "_rollout_graph", None) is not None:
            raise RuntimeError("BF16 eager model unexpectedly contains a CUDA Graph")
        self.summary = None
        self.observation = None

    @torch.inference_mode()
    def infer(self, observation):
        record = dict(observation, task=self.adapter.task["task"])
        record.setdefault("image_channel_order", "RGB")
        started = time.perf_counter()
        graph = getattr(self.policy.model, "_rollout_graph", None)
        before = graph.replays if graph else 0
        kwargs = self.adapter.infer_kwargs(self.policy.model, record, seed=int(self.config["wam"]["seed"]))
        output = self.policy.model.infer_action(**kwargs)
        self.last_normalized_action = output["action"].detach().float().cpu().numpy()
        action = self.adapter.denormalize_actions(output["action"]).float().numpy()
        if action.shape != (32, 14) or not np.isfinite(action).all():
            raise RuntimeError("Invalid BF16 physical action")
        self.summary = dict(finite=True, action_shape=list(action.shape), denoise_calls=10,
                            graph_replays=graph.replays-before if graph else 0, elapsed_seconds=time.perf_counter()-started)
        return np.ascontiguousarray(action)

    def diagnostics(self):
        return dict(ok=True, backend="bf16", task=self.adapter.task["task"], checkpoint=self.source,
                    source_checkpoint_sha256=self.adapter.task["source_checkpoint_sha256"],
                    action_coordinates="physical", cuda_graph=self.cuda_graph, dtype="bfloat16", sparse_enabled=False,
                    summary=self.summary)


class DenoiseTimer:
    """CUDA events for actual evolving denoise trajectory; no fixed-input replay.

Loop: first _predict_joint_noise entry -> final action scheduler.step return,
including all 10 joint calls, both schedulers and intermediate latent updates.
Core: sum of 10 live-Graph core calls (input copies and output clones included).
Events are enabled ONLY in the separate denoise phase, not model/RPC timings.
"""
    def __init__(self, model):
        self.model = model
        self.active = False
        self.audit = False
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.pairs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(10)]
        self.reset()
        predict, core, step = model._predict_joint_noise, model._joint_denoise_core, model.infer_action_scheduler.step

        def predict_wrapper(*a, **kw):
            if self.predict_calls == 0 and self.active:
                self.start.record()
            self.predict_calls += 1
            return predict(*a, **kw)

        def core_wrapper(*a, **kw):
            i = self.core_calls
            if self.audit and i == 0:
                self.input_hashes = {"video_noise": tensor_hash(kw["latents_video"][:, :, 1:]),
                                     "action_noise": tensor_hash(kw["latents_action"]),
                                     "first_frame": tensor_hash(kw["latents_video"][:, :, :1]),
                                     "context": tensor_hash(kw["context"])}
            if self.active:
                self.pairs[i][0].record()
            result = core(*a, **kw)
            if self.active:
                self.pairs[i][1].record()
            if self.audit and i == 0:
                static = model._rollout_graph.entries[0][2]
                self.input_hashes["graph_copied_current_inputs_exactly"] = all(
                    torch.equal(static[k], kw[k]) for k in ("latents_video", "latents_action", "context"))
            self.core_calls += 1
            return result

        def step_wrapper(*a, **kw):
            result = step(*a, **kw)
            self.scheduler_calls += 1
            if self.active and self.scheduler_calls == 10:
                self.end.record()
            return result

        model._predict_joint_noise = predict_wrapper
        model._joint_denoise_core = core_wrapper
        model.infer_action_scheduler.step = step_wrapper

    def reset(self):
        self.predict_calls = self.core_calls = self.scheduler_calls = 0
        self.input_hashes = None

    def result(self):
        if (self.predict_calls, self.core_calls, self.scheduler_calls) != (10, 10, 10):
            raise RuntimeError("Measured denoise path must have ten calls and scheduler updates")
        if not self.active:
            return {}
        self.end.synchronize()
        return {"denoise_loop_scheduler_gpu_ms": self.start.elapsed_time(self.end),
                "core_live_graph_io_gpu_ms": sum(a.elapsed_time(b) for a, b in self.pairs)}


def audit_live_inputs(policy, records):
    """Before timing: compare two distinct Graph inputs against their own eager outputs."""
    model = policy.policy.model
    graph = model._rollout_graph
    results = []
    graph_actions = []
    for record in records[:2]:
        actual = policy.infer(record)
        normalized = policy.last_normalized_action.copy()
        graph_actions.append(normalized)
        infer, core = model.infer_action, model._joint_denoise_core
        try:
            model.infer_action, model._joint_denoise_core = graph.original_infer, graph.original_core
            expected = policy.infer(record)
            torch.testing.assert_close(torch.from_numpy(policy.last_normalized_action), torch.from_numpy(normalized), rtol=1e-3, atol=1e-3)
            torch.testing.assert_close(torch.from_numpy(expected), torch.from_numpy(actual), rtol=1e-3, atol=1e-3)
        finally:
            model.infer_action, model._joint_denoise_core = infer, core
        results.append(dict(observation_id=record["observation_id"], graph_eager_match=True))
    if np.array_equal(graph_actions[0], graph_actions[1]):
        raise RuntimeError("Two distinct observations yielded identical actions; inspect stale input risk")
    return results


class MeasuredPolicy:
    def __init__(self, policy):
        self.inner = policy
        self.observation = None
        self.summary = None
        self.timer = DenoiseTimer(policy.policy.model)
        self.records = []
        self.normalized = []
        self.physical = []

    def infer(self, observation):
        phase = observation.get("comparison_phase", "unmeasured")
        self.timer.reset()
        self.timer.active, self.timer.audit = phase == "denoise", phase == "action"
        torch.cuda.synchronize()
        started = time.perf_counter()
        action = self.inner.infer(observation)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter()-started)*1000
        metrics = self.timer.result()
        graph = self.inner.policy.model._rollout_graph
        self.summary = {**self.inner.summary, **metrics, "model_wall_ms": elapsed}
        self.records.append(dict(observation_id=observation.get("observation_id"), phase=phase,
                                 seed=int(self.inner.config["wam"]["seed"]),
                                 input_hashes=self.timer.input_hashes, **self.summary))
        self.normalized.append(self.inner.last_normalized_action.copy())
        self.physical.append(action.copy())
        if len(graph.entries) != 10 or self.summary["graph_replays"] != 10:
            raise RuntimeError("Graph capture/replay contract changed")
        return action

    def diagnostics(self):
        return {**self.inner.diagnostics(), "summary": self.summary}
