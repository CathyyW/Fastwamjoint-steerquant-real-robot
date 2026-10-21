"""CPU-only mode/identity and unchanged runtime deadline checks."""
from pathlib import Path
import sys

import pytest
import yaml

from fastwam_steerquant.robot_profiles import validate_profile, baseline_checkpoint, BF16EagerPolicyClient, W4A4RHTPolicyClient
from fastwam_steerquant.robot_async_server import read_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("task", ["pack", "stack"])
@pytest.mark.parametrize("mode,rate,window", [("bf16", 25, 1.240), ("w4a8", 40, .775)])
def test_profiles_and_original_runtime_budget(task, mode, rate, window):
    config = read_config(ROOT / f"configs/robot_{task}_{mode}_async{rate}.yaml")
    assert validate_profile(config) == (mode, rate)
    original = yaml.safe_load((ROOT / f"configs/robot_{task}_w4a8_async50.yaml").read_text())
    for key, value in original["runtime"].items():
        if key not in ("rospy_rate", "output_dir"):
            assert config["runtime"][key] == value
    workspace = Path(config["offline"]["async_workspace"])
    sys.path.insert(0, str(workspace / "src"))
    from qi.deployment.async_joint_runtime import AsyncJointRTCRuntime, AsyncJointRTCRuntimeConfig
    runtime = AsyncJointRTCRuntime.__new__(AsyncJointRTCRuntime)
    runtime.config = AsyncJointRTCRuntimeConfig(control_hz=rate, chunk_size=32)
    assert (32-runtime.config.min_latency_headroom_steps)/rate == window
    runtime._validate_latency_window(previous_latency_s=window/2-.0001, current_latency_s=window/2)
    with pytest.raises(RuntimeError, match="Consecutive inference latencies"):
        runtime._validate_latency_window(previous_latency_s=window/2+.0001, current_latency_s=window/2)
    config["runtime"]["rospy_rate"] = 50
    with pytest.raises(ValueError):
        validate_profile(config)


def test_missing_pack_does_not_use_legacy_30k(monkeypatch):
    config = read_config(ROOT / "configs/robot_pack_bf16_async25.yaml")
    config["deployment_profile"].pop("baseline_checkpoint", None)
    monkeypatch.delenv("PACK_STEP15K_CKPT", raising=False)
    with pytest.raises(ValueError, match="step_030000"):
        baseline_checkpoint(config)


def test_bf16_rejects_graph_or_wrong_source():
    client = BF16EagerPolicyClient.__new__(BF16EagerPolicyClient)
    client.expected_sha256, client.expected_task = "source15k", "stack"
    info = dict(backend="bf16", dtype="bfloat16", cuda_graph=False, sparse_enabled=False,
                source_checkpoint_sha256="source15k", task="stack", action_coordinates="physical",
                summary=dict(denoise_calls=10, finite=True, graph_replays=0))
    client.validate_diagnostics(info)
    for key, value in (("cuda_graph", True), ("source_checkpoint_sha256", "30k"), ("dtype", "float32")):
        with pytest.raises(RuntimeError):
            client.validate_diagnostics({**info, key: value})


def test_w4a4_rht_rejects_w4a8_or_fallback():
    config = read_config(ROOT / "configs/robot_pack_w4a4_async40.yaml")
    assert validate_profile(config) == ("w4a4", 40)
    assert "runtime_profile" not in config["wam"]["quantization"]
    client = W4A4RHTPolicyClient.__new__(W4A4RHTPolicyClient)
    client.expected_sha256, client.expected_task = "a4", "pack"
    info = dict(backend="cutlass_w4a4", activation_bits=4, rotation="rht", sparse_enabled=False,
                deployment_sha256="a4", task="pack", action_coordinates="physical", cuda_graph=True,
                kernel_audit=dict(verified_sites=600, native_calls=6000, rht_producer_kernels=6000,
                                  cutlass_kernels=6000, cuda_kernel_verified=True),
                runtime_settings=dict(cuda_graph=True, fuse_block=False, fused_modules=0, w4a4_tile="auto"),
                summary=dict(denoise_calls=10, finite=True, graph_replays=10))
    client.validate_diagnostics(info)
    for key, value in (("backend", "cutlass_w4a8"), ("rotation", "none"), ("cuda_graph", False)):
        with pytest.raises(RuntimeError):
            client.validate_diagnostics({**info, key: value})
    info["kernel_audit"]["rht_producer_kernels"] = 0
    with pytest.raises(RuntimeError):
        client.validate_diagnostics(info)
