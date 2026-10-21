#!/usr/bin/env python3
"""Hardware-free service for BF16 eager/25 Hz or W4A8 Graph/40 Hz clients."""
import argparse
import json
import time
from multiprocessing.connection import Listener
from pathlib import Path

import torch
import numpy as np

from fastwam_steerquant.robot_async_client import authkey_from_config
from fastwam_steerquant.robot_async_server import (
    read_config, recorded_observation, PackedRobotPolicy, serve_connection,
)
from fastwam_steerquant.robot_comparison import BF16RobotPolicy, audit_live_inputs
from fastwam_steerquant.robot_profiles import validate_profile, baseline_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--offline-only", action="store_true")
    parser.add_argument("--port", type=int)
    parser.add_argument("--quick-benchmark", action="store_true", help="12 warm full-policy wall-clock samples before listening")
    args = parser.parse_args()
    config = read_config(args.config)
    mode, rate = validate_profile(config)
    # Reject missing pack 15k before inspecting/loading any model payload.
    checkpoint = baseline_checkpoint(config) if mode == "bf16" else None
    key = None if args.offline_only else authkey_from_config(config)
    torch.set_num_threads(4)
    records = [dict(recorded_observation(config, i), observation_id=f"warmup:{i}") for i in (0, 35)]
    if mode == "bf16":
        policy = BF16RobotPolicy(config, checkpoint, cuda_graph=False)
        if policy.adapter.task["source_checkpoint_sha256"] != config["deployment_profile"]["source_checkpoint_sha256"]:
            raise ValueError("Baseline source identity differs from requested profile")
        for record in records:
            policy.infer(record)
        checks = {"cuda_graph": False, "finite_observations": len(records)}
    else:
        if mode == "w4a4":
            from fastwam_steerquant.robot_w4a4 import PackedW4A4RobotPolicy
            policy = PackedW4A4RobotPolicy(config)
        else:
            policy = PackedRobotPolicy(config)
        policy.warmup(records[0])
        checks = audit_live_inputs(policy, records)
    diagnostics = policy.diagnostics
    policy.diagnostics = lambda: {**diagnostics(), "control_hz": rate, "latency_window_ms": 31000 / rate}
    torch.cuda.synchronize()
    report = {"diagnostics": policy.diagnostics(), "warmup_checks": checks, "contract": policy.contract}
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    if args.quick_benchmark:
        timings, normalized, physical = [], [], []
        for i in range(12):
            torch.cuda.synchronize()
            started = time.perf_counter()
            action = policy.infer(records[i % 2])
            torch.cuda.synchronize()
            timings.append((time.perf_counter()-started)*1000)
            normalized.append(policy.last_normalized_action.copy())
            physical.append(action.copy())
        actions_path = path.with_name(path.stem + "_actions.npz")
        np.savez(actions_path, normalized=np.stack(normalized), physical=np.stack(physical))
        report["quick_benchmark"] = dict(scope="raw recorded observation -> physical action; preprocess/VAE/10-step denoise/denormalize included; RPC excluded",
            seed=config["wam"]["seed"], episode=config["offline"]["episode_hdf5"], frame_indices=[0,35]*6,
            count=12, milliseconds=timings, mean_ms=float(np.mean(timings)),
            p50_ms=float(np.percentile(timings,50)), p95_ms=float(np.percentile(timings,95)),
            consecutive_two_ms=[sum(timings[i:i+2]) for i in range(0,12,2)],
            actions=str(actions_path), action_shape=list(action.shape), finite=bool(np.isfinite(physical).all()),
            gpu=torch.cuda.get_device_name(), torch=str(torch.__version__), cuda=torch.version.cuda)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({"mode": mode, "control_hz": rate, "latency_window_ms": 31000 / rate,
                      "cuda_graph": policy.diagnostics()["cuda_graph"], "report": str(path)}), flush=True)
    if args.offline_only:
        return
    server = config["policy_server"]
    with Listener((server["host"], args.port or int(server["port"])), authkey=key) as listener:
        print(f"PROFILE_READY {listener.address}", flush=True)
        while True:
            with listener.accept() as conn:
                policy.observation = None
                policy.summary = None
                if serve_connection(policy, conn):
                    break


if __name__ == "__main__":
    main()
