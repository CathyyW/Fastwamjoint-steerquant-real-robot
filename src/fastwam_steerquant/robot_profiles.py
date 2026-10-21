"""Explicit BF16 eager/25 Hz and packed W4A8 Graph/40 Hz deployment choices."""
import os

from .robot_async_client import W4A8Async50PolicyClient, authkey_from_config


def validate_profile(config):
    profile = config.get("deployment_profile")
    rate = float(config["runtime"]["rospy_rate"])
    if profile is None:
        if rate != 50:
            raise ValueError("Non-50 Hz configurations require an explicit deployment_profile")
        return "w4a8", rate
    mode = profile["mode"]
    expected = {"bf16": (25, False), "w4a8": (40, True), "w4a4": (40, True)}
    if mode not in expected or (rate, profile["cuda_graph"]) != expected[mode]:
        raise ValueError("Required profiles: BF16 eager/25 Hz or W4A8/W4A4 Graph/40 Hz")
    if mode == "w4a4" and config["wam"]["quantization"]["task"] != "pack":
        raise ValueError("Only pack W4A4 has been configured")
    if config["runtime"]["action_chunk_size"] != 32 or config["wam"]["action_horizon"] != 32:
        raise ValueError("Expected 32-step action chunks")
    return mode, rate


def baseline_checkpoint(config):
    profile = config["deployment_profile"]
    value = profile.get("baseline_checkpoint")
    env = profile.get("baseline_checkpoint_env")
    if env:
        value = os.environ.get(env) or value
    if not value:
        raise ValueError(f"Missing original step_015000 checkpoint; set {env or 'baseline_checkpoint'}. "
                         "step_030000 is not an acceptable substitute.")
    return value


class BF16EagerPolicyClient(W4A8Async50PolicyClient):
    def validate_diagnostics(self, info, *, require_summary=True):
        if (info.get("backend") != "bf16" or info.get("dtype") != "bfloat16"
                or info.get("cuda_graph") is not False or info.get("sparse_enabled") is not False
                or info.get("source_checkpoint_sha256") != self.expected_sha256
                or info.get("task") != self.expected_task or info.get("action_coordinates") != "physical"):
            raise RuntimeError("BF16 step15k/eager/physical-coordinate contract mismatch")
        if require_summary:
            summary = info.get("summary") or {}
            if (summary.get("denoise_calls") != 10 or summary.get("finite") is not True
                    or summary.get("graph_replays") != 0):
                raise RuntimeError("BF16 eager inference diagnostics failed")


class W4A4RHTPolicyClient(W4A8Async50PolicyClient):
    def validate_diagnostics(self, info, *, require_summary=True):
        if (info.get("backend") != "cutlass_w4a4" or info.get("activation_bits") != 4
                or info.get("rotation") != "rht" or info.get("sparse_enabled") is not False
                or info.get("deployment_sha256") != self.expected_sha256
                or info.get("task") != self.expected_task or info.get("action_coordinates") != "physical"
                or info.get("cuda_graph") is not True):
            raise RuntimeError("W4A4/RHT deployment identity mismatch")
        proof, settings = info.get("kernel_audit") or {}, info.get("runtime_settings") or {}
        if (proof.get("verified_sites") != 600 or proof.get("native_calls") != 6000
                or proof.get("rht_producer_kernels") != 6000 or proof.get("cutlass_kernels") != 6000
                or proof.get("cuda_kernel_verified") is not True
                or settings.get("cuda_graph") is not True or settings.get("fuse_block") is not False
                or settings.get("fused_modules") != 0 or settings.get("w4a4_tile") != "auto"):
            raise RuntimeError("W4A4 native/Graph/no-extra-fusion audit failed")
        if require_summary:
            summary = info.get("summary") or {}
            if (summary.get("denoise_calls") != 10 or summary.get("finite") is not True
                    or summary.get("graph_replays") != 10):
                raise RuntimeError("W4A4 inference diagnostics failed")


def make_client(config, *, host=None, port=None):
    mode, rate = validate_profile(config)
    server, q = config["policy_server"], config["wam"]["quantization"]
    cls = {"bf16": BF16EagerPolicyClient, "w4a8": W4A8Async50PolicyClient, "w4a4": W4A4RHTPolicyClient}[mode]
    expected = (config["deployment_profile"]["source_checkpoint_sha256"] if mode == "bf16"
                else q["deployment_sha256"])
    client = cls(host or server["host"], port or server["port"], authkey_from_config(config),
                 expected_sha256=expected, expected_task=q["task"])
    if "deployment_profile" in config:
        original = client.validate_diagnostics

        def validate(info, *, require_summary=True):
            original(info, require_summary=require_summary)
            if info.get("control_hz") != rate or info.get("latency_window_ms") != 31000 / rate:
                raise RuntimeError("Server/client action frequency or latency window mismatch")

        client.validate_diagnostics = validate
    return client
