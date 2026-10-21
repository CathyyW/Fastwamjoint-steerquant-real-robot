"""Validate the two published W4A8 experiment identities and robot settings."""
from pathlib import Path

import numpy as np
import yaml

from fastwam_steerquant.robot_async_client import W4A8Async50PolicyClient, authkey_from_config
from .actions import build_action_governor
from .observations import CAMERAS

DEPLOYMENTS = {
    "pack": "4e3b7ff923623117d394d267243e1bb62f3027f5920ad5413754129e8dcc95b8",
    "stack": "833a84d13ca4d80c1c6e6cdd1e7b41ce4b53380aa01c7c8c794c73ba142e1912",
}


def validate_config(config):
    policy = config["policy"]
    if policy["task"] not in DEPLOYMENTS or policy["deployment_sha256"] != DEPLOYMENTS[policy["task"]]:
        raise ValueError("Expected the published pack/stack W4A8 task and deployment SHA256")
    if not isinstance(policy["prompt"], str) or not policy["prompt"].strip():
        raise ValueError("A task prompt is required")
    cfg = config["runtime"]
    if (cfg["control_hz"] != 40 or cfg["chunk_size"] != 32 or cfg["action_dim"] != 14
            or cfg["arrival_smoothstep_steps"] != 5 or cfg["exp_weight_factor"] != .2
            or cfg["min_latency_headroom_steps"] != 1):
        raise ValueError("Expected paper runtime: 40 Hz, [32,14], 5-step smoothing, weight .2, headroom 1")
    allowed = {"control_hz", "chunk_size", "action_dim", "arrival_smoothstep_steps",
               "exp_weight_factor", "min_latency_headroom_steps", "startup_timeout_s",
               "action_wait_timeout_s", "fresh_chunk_grace_s", "producer_poll_s", "producer_join_timeout_s"}
    if set(cfg) != allowed:
        raise ValueError("Specify all runtime timing settings; motion flags belong on the CLI")
    for key, value in cfg.items():
        if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"runtime.{key} must be finite and positive")
    if cfg["producer_join_timeout_s"] <= 24:
        raise ValueError("Producer join timeout must cover two 10-second RPCs and two 2-second observation reads")
    robot = config["robot"]
    for key in ("action_safety_threshold", "max_action_delta", "max_gripper_delta",
                "state_safety_threshold", "state_max_age_s", "max_linear_vel", "max_angular_vel",
                "max_linear_acc", "max_angular_acc", "gripper_force", "initial_period_s"):
        if not np.isfinite(robot[key]) or robot[key] <= 0:
            raise ValueError(f"robot.{key} must be finite and positive")
    for key in ("bitrate", "speed_pct", "initial_steps"):
        if type(robot[key]) is not int or robot[key] <= 0:
            raise ValueError(f"robot.{key} must be a positive integer")
    if robot["initial_steps"] < 2 or robot["speed_pct"] > 100:
        raise ValueError("Invalid initialization steps or speed percentage")
    if (not robot["left_channel"] or not robot["right_channel"]
            or robot["left_channel"] == robot["right_channel"]):
        raise ValueError("Two distinct CAN channels are required")
    initial = []
    for side in ("left", "right"):
        pose = np.asarray(robot[f"{side}_init_position"], dtype=np.float32)
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise ValueError("Each initial pose must contain 7 finite values")
        initial.extend(pose.tolist())
    # Constructor validates all vector sizes, finiteness, bounds and margins.
    if any(np.asarray(config["safety"][key]).shape != (14,) for key in (
            "lower_bounds", "upper_bounds", "raw_action_lower_bounds", "raw_action_upper_bounds")):
        raise ValueError("Safety bounds must each contain 14 values")
    build_action_governor(config, initial)
    cameras = config["cameras"]
    if set(cameras["topics"]) != set(CAMERAS) or not all(cameras["topics"].values()):
        raise ValueError("Specify all three camera topics")
    for key in ("max_age_s", "max_skew_s", "startup_timeout_s"):
        if not np.isfinite(cameras[key]) or cameras[key] <= 0:
            raise ValueError(f"cameras.{key} must be finite and positive")
    server = config["policy_server"]
    if not server["host"] or type(server["port"]) is not int or not 0 < server["port"] < 65536:
        raise ValueError("Invalid model server address")
    if "authkey" in server:
        raise ValueError("Use FASTWAM_AUTHKEY in the environment, never a YAML key")
    return config


def load_config(path):
    return validate_config(yaml.safe_load(Path(path).read_text()))


def validate_server(client, info, *, require_summary=False):
    # Reuse the original protocol's complete W4A8/native/physical-action audit.
    client.validate_diagnostics(info, require_summary=require_summary)
    validate_server_profile(info)


def validate_server_profile(info):
    if (info.get("cuda_graph") is not True or info.get("control_hz") != 40
            or info.get("latency_window_ms") != 775):
        raise RuntimeError("Expected W4A8 Graph server profile at 40 Hz / 775 ms")


def make_client(config, *, host=None, port=None):
    server, policy = config["policy_server"], config["policy"]
    key = authkey_from_config({"policy_server": {"authkey": "${FASTWAM_AUTHKEY}"}})
    client = W4A8Async50PolicyClient(host or server["host"], port or server["port"], key,
        expected_sha256=policy["deployment_sha256"], expected_task=policy["task"])
    original = client.validate_diagnostics

    def validate(info, *, require_summary=True):
        original(info, require_summary=require_summary)
        validate_server_profile(info)

    client.validate_diagnostics = validate
    return client
