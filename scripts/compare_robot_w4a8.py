#!/usr/bin/env python3
"""Serial fresh-process BF16-Graph / W4A8-profile comparison, no hardware."""
import argparse
import builtins
from dataclasses import asdict
import json
from multiprocessing.connection import Client
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time

import h5py
import numpy as np
import yaml

from fastwam_steerquant.robot_async_client import W4A8Async50PolicyClient
from fastwam_steerquant.robot_comparison import observation


class ComparisonClient:
    def __init__(self, port, key, mode, config):
        self.conn = Client(("127.0.0.1", port), authkey=key.encode())
        self.mode = mode
        q = config["wam"]["quantization"]
        self.validator = W4A8Async50PolicyClient.__new__(W4A8Async50PolicyClient)
        self.validator.expected_sha256, self.validator.expected_task = q["deployment_sha256"], q["task"]
        self.records = []

    def _request(self, payload):
        self.conn.send(payload)
        if not self.conn.poll(10):
            raise TimeoutError("Original ten-second RPC timeout exceeded")
        result = self.conn.recv()
        if not isinstance(result, dict) or not result.get("ok"):
            raise RuntimeError(f"RPC failed: {result}")
        return result

    def infer(self, obs):
        start = time.monotonic_ns()
        action = self._request({"op": "infer", "obs": obs})["action"]
        received = time.monotonic_ns()
        info = self._request({"op": "diagnostics"})
        if self.mode == "w4a8":
            self.validator.validate_diagnostics(info)
        elif (info.get("backend") != "bf16" or info.get("task") != self.validator.expected_task
                or info.get("action_coordinates") != "physical"
                or info.get("summary", {}).get("graph_replays") != 10):
            raise RuntimeError("BF16 Graph server identity/Graph check failed")
        if action.shape != (32, 14) or action.dtype != np.float32 or not np.isfinite(action).all():
            raise ValueError("Invalid complete physical action chunk")
        done = time.monotonic_ns()
        self.records.append(dict(observation_id=obs.get("observation_id"), phase=obs.get("comparison_phase", "async"),
                                 submitted_monotonic_ns=start, received_monotonic_ns=received,
                                 diagnostics_finished_monotonic_ns=done,
                                 rpc_action_ms=(received-start)/1e6, rpc_with_diagnostics_ms=(done-start)/1e6,
                                 server_summary=info["summary"]))
        return action


def test_async(client, config, spec, output):
    # Import the exact original runtime without allowing any hardware imports.
    workspace = Path(config["offline"]["async_workspace"])
    sys.path.insert(0, str(workspace / "src"))
    original_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in ("rospy", "sensor_msgs", "pyAgxArm", "piper_sdk"):
            raise RuntimeError("Hardware imports are forbidden in this experiment")
        return original_import(name, *args, **kwargs)
    builtins.__import__ = guarded
    from qi.deployment.async_joint_runtime import AsyncJointRTCRuntime, AsyncJointRTCRuntimeConfig
    from qi.deployment.async25_adapters import ObservationSource, ShadowActionSink, build_action_governor
    class Source(ObservationSource):
        def read_observation(self, t):
            obs = super().read_observation(t)
            obs.update(image_channel_order="as_stored", comparison_phase="async", observation_id=f'async:{t}')
            return obs
    error = None
    runtime = None
    enqueue_trace = []
    try:
        with h5py.File(spec["episode_hdf5"], "r") as f:
            source = Source(config, episode=f)
            governor = build_action_governor(config, source.read_observation(0)["state"])
            r = config["runtime"]
            runtime = AsyncJointRTCRuntime(policy=client, observation_source=source, action_sink=ShadowActionSink(),
                action_governor=governor,
                config=AsyncJointRTCRuntimeConfig(control_hz=50, chunk_size=32, max_steps=128,
                    exp_weight_factor=float(r["exp_weight_factor"]), arrival_smoothstep_steps=int(r["arrival_smoothstep_steps"]),
                    prompt=config["wam"]["prompt"], fresh_chunk_grace_s=float(r["fresh_chunk_grace_s"])))
            enqueue = runtime.buffer.enqueue
            def timed_enqueue(chunk, cursor, generation=None):
                started = time.monotonic_ns()
                accepted = enqueue(chunk, cursor=cursor, generation=generation)
                completed = time.monotonic_ns()
                enqueue_trace.append(dict(cursor=cursor, generation=generation, accepted=accepted,
                                          started_monotonic_ns=started, completed_monotonic_ns=completed))
                return accepted
            runtime.buffer.enqueue = timed_enqueue
            runtime.run()
    except Exception as exc:
        error = repr(exc)
        while exc.__cause__ is not None:
            exc = exc.__cause__
            error += " <- " + repr(exc)
    finally:
        builtins.__import__ = original_import
        result = dict(error=error, hardware_connected=False, robot_actions_executed=False,
                      runtime=runtime.report() if runtime else None, enqueue_trace=enqueue_trace)
        if runtime:
            result["inference_chunks"] = [asdict(x) for x in runtime.inference_chunks]
            result["action_trace"] = [asdict(x) for x in runtime.action_trace]
        (output / "async.json").write_text(json.dumps(result, default=lambda x: x.tolist(), indent=2))


def run_mode(args, mode, root, config, spec, output):
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(root / "src"), FASTWAM_AUTHKEY=secrets.token_hex(32))
    if mode == "bf16":
        env.pop("FASTWAM_RUNTIME_PROFILE", None)
        env.pop("FASTWAM_W4A8_TILE", None)
    command = [sys.executable, "-u", str(root / "scripts/serve_robot_comparison.py"),
               "--config", str(root / f"configs/robot_{args.task}_w4a8_async50.yaml"),
               "--spec", str(Path(args.spec).resolve()), "--mode", mode,
               "--port", str(args.port), "--output-dir", str(output)]
    (output / "gpu_before.txt").write_text(subprocess.check_output(["nvidia-smi"], text=True))
    client = None
    pairs = []
    with (output / "server.log").open("w") as log:
        server = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic()+420
            while "COMPARISON_READY" not in (output / "server.log").read_text():
                if server.poll() is not None:
                    raise RuntimeError(f"Server failed; see {output / 'server.log'}")
                if time.monotonic() > deadline:
                    raise TimeoutError("Server startup timed out")
                time.sleep(.25)
            client = ComparisonClient(args.port, env["FASTWAM_AUTHKEY"], mode, config)
            records = [observation(spec, config, i) for i in spec["frame_indices"]]
            if args.only_async:
                test_async(client, config, spec, output)
                return
            # Two-record smoke/audit was completed server-side before listening.
            for obs in records:
                client.infer(dict(obs, comparison_phase="action"))
            perf = records[::2][:spec["performance_observations"]]
            for phase in ("latency", "denoise"):
                for obs in perf:
                    for _ in range(spec["warmup_per_observation"]):
                        client.infer(dict(obs, comparison_phase="warmup"))
                    for _ in range(spec["repeats"]):
                        client.infer(dict(obs, comparison_phase=phase))
            for i in range(12):
                started = time.perf_counter_ns()
                client.infer(dict(perf[i % len(perf)], comparison_phase="pair"))
                client.infer(dict(perf[(i+1) % len(perf)], comparison_phase="pair"))
                pairs.append(dict(pair=i, consecutive_rpc_with_diagnostics_ms=(time.perf_counter_ns()-started)/1e6))
            test_async(client, config, spec, output)
        finally:
            if client:
                (output / "client.json").write_text(json.dumps(dict(records=client.records, consecutive_pairs=pairs), indent=2))
                try:
                    client._request({"op": "shutdown"})
                finally:
                    client.conn.close()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.terminate()
                server.wait(timeout=10)
            (output / "gpu_after.txt").write_text(subprocess.check_output(["nvidia-smi"], text=True))
    if server.returncode:
        raise RuntimeError(f"Comparison server exited {server.returncode}")
    print(json.dumps(dict(task=args.task, mode=mode, output=str(output), measured=True)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("pack", "stack"))
    parser.add_argument("--spec")
    parser.add_argument("--modes", nargs="+", choices=("bf16", "w4a8"), default=["bf16", "w4a8"])
    parser.add_argument("--port", type=int, default=18766)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--only-async", action="store_true", help="Separate enqueue-timestamp probe, without repeating performance/action sweeps")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.spec = args.spec or str(root / f"configs/comparison_{args.task}_15k.json")
    spec = json.loads(Path(args.spec).read_text())
    if spec["task"] != args.task:
        parser.error("Task differs from sampling spec")
    if "bf16" in args.modes and not spec["baseline_checkpoint"]:
        parser.error("Missing step_015000 baseline; provide its verified path, never use step_030000")
    config = yaml.safe_load((root / f"configs/robot_{args.task}_w4a8_async50.yaml").read_text())
    if spec["seed"] != config["wam"]["seed"]:
        parser.error("Matched seed differs from policy configuration")
    for mode in args.modes:
        run_mode(args, mode, root, config, spec, Path(args.output_dir).resolve() / args.task / mode)


if __name__ == "__main__":
    main()
