"""Packed W4A8 service with the existing async50 wire/action contract.

This process imports only the audited model handoff, never robot/ROS code.
The listener is opened only after a recorded-observation native kernel audit
and (when requested) strict live CUDA Graph warmup have succeeded.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import copy
import json
from multiprocessing.connection import Listener
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch
import yaml

from .adapters.real_robot import RealRobotAdapter, verify_file
from .deployment import validate_payload
from .policy import QuantizedPolicy
from .robot_async_client import authkey_from_config
from .runtime import WAMQuantLinear
from .rollout_graph import LiveJointDiTGraph

CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def read_config(path):
    config = yaml.safe_load(Path(path).read_text())
    w, r = config["wam"], config["runtime"]
    if (w["quantization"]["backend"] not in ("cutlass_w4a8", "cutlass_w4a4") or w["quantization"].get("cuda_graph") is not True
            or w.get("version1", {}).get("enabled")
            or w["action_horizon"] != 32 or w["num_inference_steps"] != 10
            or w["num_video_frames"] != 9 or w["mixed_precision"] != "bf16"
            or w["rand_device"] != "cpu" or w["use_text_encoder"]
            or r["rospy_rate"] not in (25, 40, 50) or r["action_chunk_size"] != 32):
        raise ValueError("Configuration differs from the audited dense W4A8/async50 contract")
    return config


def recorded_observation(config, index=0, episode=None):
    import h5py
    with h5py.File(episode or config["offline"]["episode_hdf5"], "r") as f:
        i = min(index, len(f["observations/qpos"]) - 1)
        return {"images": {k: f[f"observations/images/{k}"][i] for k in CAMERAS},
                "state": np.asarray(f["observations/qpos"][i], dtype=np.float32),
                "prompt": config["wam"]["prompt"],
                "image_channel_order": config["offline"]["image_channel_order"]}


def check_online_contract(config, model_config):
    """Check real current configs/assets, not filenames or an architecture guess."""
    w = config["wam"]
    task = model_config["real_robot"]
    train = yaml.safe_load(Path(w["train_config"]).read_text())
    if w["prompt"] != task["prompt"] or w["quantization"]["task"] != task["task"]:
        raise ValueError("Task/prompt differs from deployment")
    for key, expected in task["architecture"].items():
        if train["model"].get(key) != expected:
            raise ValueError(f"Current inference architecture differs: {key}")
    if train["data"]["train"]["processor"] != task["processor"]:
        raise ValueError("Current processor configuration differs from calibration")
    if train["data"]["train"]["context_len"] != 128:
        raise ValueError("Context length differs")
    verify_file(w["dataset_stats"], task["asset_sha256"]["dataset_stats"])
    cache_name = Path(model_config["assets"]["context_cache"]).name
    verify_file(Path(w["context_cache_dir"]) / cache_name, task["asset_sha256"]["context_cache"])
    # Compare the actual pure online functions without importing hardware or
    # importing a second qi namespace into this model process.
    root = Path(config["offline"]["async_workspace"])
    current = ast.parse((root / "src/qi/deployment/wam_policy_client.py").read_text())
    handoff = Path(w["quantization"]["handoff_root"])
    audited = ast.parse((handoff / "src/offline_preprocessing.py").read_text())
    names = ("rgb_to_float_chw", "preprocess_real_images", "normalize_proprio", "denormalize_actions", "format_prompt")
    functions = lambda tree: {n.name: ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, ast.FunctionDef)}
    a, b = functions(current), functions(audited)
    for name in names:
        if a[name] != b[name]:
            raise ValueError(f"Online preprocessing differs from handoff: {name}")
    for relative in ("qi/datasets/lerobot/processors/fastwam_processor.py",
                     "qi/datasets/lerobot/utils/normalizer.py", "qi/datasets/dataset_utils.py",
                     "qi/datasets/lerobot/transforms/action_state_merger.py"):
        if (root / "src" / relative).read_bytes() != (handoff / "src" / relative).read_bytes():
            raise ValueError(f"Online preprocessing dependency differs: {relative}")
    return {"architecture_processor_preprocessing_match": True,
            "stats_text_sha256_match": True, "camera_order": list(CAMERAS),
            "raw_shape": [480, 640, 3], "model_image_shape": [1, 3, 384, 320],
            "source_checkpoint_sha256": task["source_checkpoint_sha256"],
            "source_checkpoint": model_config["assets"]["source_checkpoint"],
            "current_bf16_checkpoint": w["quantization"]["baseline_checkpoint"],
            "vae_checkpoint_sha256": task["vae_checkpoint_sha256"],
            "historical_hdf_color": "unconfirmed; offline as_stored, live RGB unchanged",
            "normalized_action_scale": model_config.get("normalized_action_scale"),
            "calibration_approval": model_config.get("calibration_approval")}


def prepare_adapter(config):
    w, q = config["wam"], config["wam"]["quantization"]
    verify_file(w["ckpt"], q["deployment_sha256"])
    manifest = Path(w["ckpt"]).parent / "SHA256SUMS"
    if manifest.read_text().split() != [q["deployment_sha256"], "deployment.pt"]:
        raise ValueError("Checkpoint directory SHA256SUMS differs from config")
    payload = torch.load(w["ckpt"], map_location="cpu", weights_only=True, mmap=True)
    validate_payload(payload)
    bits, rotation = (4, "rht") if q["backend"] == "cutlass_w4a4" else (8, "none")
    if (payload["activation_bits"] != bits or len(payload["sites"]) != 600
            or payload["num_calls"] != 10 or any(s["rotation"] != rotation for s in payload["sites"])):
        raise ValueError("Deployment bits/rotation/sites/calls differ from configured backend")
    model_config = copy.deepcopy(payload["model_config"])
    del payload
    contract = check_online_contract(config, model_config)
    handoff = Path(q["handoff_root"])
    task = q["task"]
    model_config["handoff_root"] = str(handoff)
    for key, path in {
        "training_config": handoff / f"configs/train_config_joint_{task}3_infer.yaml",
        "dataset_stats": handoff / f"assets/stats/{task}/dataset_stats.json",
        "context_cache": handoff / f"assets/text_embeds/{task}" / Path(model_config["assets"]["context_cache"]).name,
        "handoff_manifest": handoff / "SHA256SUMS",
    }.items():
        model_config["assets"][key] = str(path)
    return RealRobotAdapter(model_config), contract


class PackedRobotPolicy:
    def __init__(self, config):
        self.config = config
        w, q = config["wam"], config["wam"]["quantization"]
        self.adapter, self.contract = prepare_adapter(config)
        os.environ["FASTWAM_W4A8_EXTENSION"] = q["extension"]
        profile = q.get("runtime_profile", str(Path(__file__).resolve().parents[2] /
                                             "configs/real_robot_w4a8_runtime.json"))
        # Profile must configure tile selection BEFORE the extension is loaded.
        self.policy = QuantizedPolicy(w["ckpt"], self.adapter, device=w["device"],
                                      runtime_profile=profile)
        from .kernels.w4a8.loader import load_extension
        self.extension = load_extension()
        if Path(self.extension.__file__).resolve() != Path(q["extension"]).resolve():
            raise RuntimeError("A different kernel extension was already loaded")
        self.quant_layers = {n: m for n, m in self.policy.model.named_modules() if isinstance(m, WAMQuantLinear)}
        if len(self.quant_layers) != 600:
            raise RuntimeError("Expected 600 distinct native quantized layers")
        self.assert_native()
        self.observation = None
        self.summary = None
        self.kernel_audit = {}

    def assert_native(self):
        settings = self.policy.runtime_settings
        if (settings["w4a8_tile"] != "64" or settings["cuda_graph"] is not True
                or settings["fuse_block"] is not False or self.policy.fused_modules != 0):
            raise RuntimeError("W4A8 server requires tile64 + Graph without extra block fusion")
        for name, module in self.quant_layers.items():
            if (module.backend != "cutlass_w4a8" or module.rotation != "none"
                    or module.supports_fused_adaln or module.supports_fused_gate_residual
                    or module.weight.dtype != torch.uint8 or module.weight.device.type != "cuda"
                    or any(getattr(module, k).dtype != torch.float32 for k in
                           ("weight_scales", "input_scale", "gamma_gains", "clipping_ranges"))):
                raise RuntimeError(f"Packed/native/FP32 scale invariant failed: {name}")

    def infer(self, observation):
        self.summary = None
        self.assert_native()
        record = dict(observation)
        record["task"] = self.config["wam"]["quantization"]["task"]
        # Existing LiveImages already emits RGB. Recorded readers explicitly
        # label as_stored; neither path performs a speculative channel swap.
        record.setdefault("image_channel_order", "RGB")
        graph = getattr(self.policy.model, "_rollout_graph", None)
        before = graph.replays if graph else 0
        started = time.perf_counter()
        output = self.policy.infer(record, seed=int(self.config["wam"]["seed"]))
        self.last_normalized_action = output["action"].detach().float().cpu().numpy()
        action = self.adapter.denormalize_actions(output["action"]).float().numpy()
        if action.shape != (32, 14) or not np.isfinite(action).all():
            raise RuntimeError("Nonfinite or invalid physical-coordinate action output")
        self.summary = {"denoise_calls": self.policy.tracker.total_calls,
                        "graph_replays": graph.replays - before if graph else 0,
                        "finite": True, "action_shape": list(action.shape),
                        "elapsed_seconds": time.perf_counter() - started}
        return np.ascontiguousarray(action)

    def warmup(self, observation):
        from .kernels.w4a8 import ops
        # Audit eager native dispatch before capture, then restore the profile's
        # Graph mode. No listener is open during this startup-only audit.
        self.policy.model._rollout_graph.remove()
        counts = Counter()
        native_calls = 0
        handles = []
        original = ops.symmetric_scheduled_stream_linear

        def counted(*args, **kwargs):
            nonlocal native_calls
            result = original(*args, **kwargs)
            native_calls += 1
            return result

        def hook(name):
            def count(module, args, result):
                counts[name] += 1
            return count

        try:
            for name, module in self.quant_layers.items():
                handles.append(module.register_forward_hook(hook(name)))
            ops.symmetric_scheduled_stream_linear = counted
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as profile:
                eager = self.infer(observation)
            kernels = Counter(e.name for e in profile.events() if e.device_type == torch.autograd.DeviceType.CUDA)
            # Exclude BF16 VAE/attention CUTLASS kernels: the W4A8 extension's
            # packed GEMM has this visitor-based epilogue (also in mangled names).
            cutlass = {k: n for k, n in kernels.items() if "GemmWithEpilogueVisitor" in k}
            producer = {k: n for k, n in kernels.items() if "quantize_static_per_tensor_stream_kernel" in k}
            if set(counts) != set(self.quant_layers) or set(counts.values()) != {10} or native_calls != 6000:
                raise RuntimeError("Some target layer did not execute W4A8 on every denoise step")
            if sum(cutlass.values()) != 6000 or sum(producer.values()) != 6000:
                raise RuntimeError("Profiler did not prove 6000 A8 producers and native CUTLASS GEMMs")
            tile64 = {k: n for k, n in cutlass.items()
                      if "GemmShapeILi64ELi128ELi128" in k or "GemmShape<64, 128, 128>" in k}
            if sum(tile64.values()) != 6000:
                raise RuntimeError("Profiler did not prove tile64 for every W4A8 GEMM")
            self.kernel_audit = {"verified_sites": len(counts), "native_calls": native_calls,
                                 "cuda_kernel_verified": True, "producer_kernels": sum(producer.values()),
                                 "cutlass_kernels": sum(cutlass.values()),
                                 "tile64_kernels": sum(tile64.values()),
                                 "extension": self.extension.__file__,
                                 "producer_names": list(producer),
                                 "cutlass_names": [k[:250] for k in cutlass]}
        finally:
            ops.symmetric_scheduled_stream_linear = original
            for handle in handles:
                handle.remove()
        if self.config["wam"]["quantization"]["cuda_graph"]:
            LiveJointDiTGraph(self.policy.model, num_calls=10).install()
            graph_output = self.infer(observation)
            torch.testing.assert_close(torch.from_numpy(graph_output), torch.from_numpy(eager), rtol=1e-3, atol=1e-3)
            self.kernel_audit["graph_eager_output_match"] = True
        self.infer(observation)
        self.observation = None

    def diagnostics(self):
        q = self.config["wam"]["quantization"]
        return {"ok": True, "backend": "cutlass_w4a8", "activation_bits": 8, "rotation": "none",
                "sparse_enabled": False, "task": q["task"], "action_coordinates": "physical",
                "deployment_sha256": q["deployment_sha256"], "checkpoint": self.config["wam"]["ckpt"],
                "cuda_graph": q["cuda_graph"], "runtime_settings": self.policy.runtime_settings,
                "kernel_audit": self.kernel_audit, "summary": self.summary}


def serve_connection(policy, conn):
    """Same RPC operations/envelopes; one model/connection owner, no worker pool."""
    while True:
        try:
            request = conn.recv()
        except EOFError:
            return False
        try:
            if not isinstance(request, dict):
                raise ValueError("Expected request mapping")
            op = request.get("op")
            response = {"ok": True}
            if op in ("update_observation", "infer"):
                policy.observation = request["obs"]
            if op in ("get_action", "infer"):
                if policy.observation is None:
                    raise ValueError("Observation is empty")
                response["action"] = policy.infer(policy.observation)
            elif op == "diagnostics":
                response = policy.diagnostics()
            elif op == "reset":
                policy.observation = None
                policy.summary = None
            elif op == "shutdown":
                conn.send(response)
                return True
            elif op != "update_observation":
                raise ValueError(f"Unknown operation: {op!r}")
            conn.send(response)
        except Exception as exc:
            conn.send({"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--offline-only", action="store_true", help="Audit forward without opening a listener")
    p.add_argument("--report", required=True)
    p.add_argument("--port", type=int)
    args = p.parse_args()
    torch.set_num_threads(4)
    config = read_config(args.config)
    key = None if args.offline_only else authkey_from_config(config)
    policy = PackedRobotPolicy(config)
    policy.warmup(recorded_observation(config))
    report = {"diagnostics": policy.diagnostics(), "contract": policy.contract,
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(), "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({"ready": True, "task": config["wam"]["quantization"]["task"],
                      "summary": policy.summary, "report": str(path)}, ensure_ascii=False), flush=True)
    if args.offline_only:
        return
    server = config["policy_server"]
    with Listener((server["host"], args.port or int(server["port"])), authkey=key) as listener:
        print(f"W4A8 server listening at {listener.address}", flush=True)
        while True:
            with listener.accept() as conn:
                # Do not leak a previous connection's observation/episode state.
                policy.observation = None
                policy.summary = None
                if serve_connection(policy, conn):
                    break


if __name__ == "__main__":
    main()
