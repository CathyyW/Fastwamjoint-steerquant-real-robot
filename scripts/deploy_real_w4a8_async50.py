#!/usr/bin/env python3
"""BF16/W4A8 entry reusing the joint-asi-async50 control implementation.

Derived from that worktree's scripts/deploy_real_async50_asi_wam.py at
836dd4796ae57653c22c48991ec5c9f6fd8b9c88. Model protocol and configured control
frequency are selected locally; motion gates and queue implementation are kept.
This hardware-capable entry is NOT the offline smoke entry; use
scripts/mock_w4a8_async50.py for validation without ROS/CAN/robot objects.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from qi.deployment.async_joint_runtime import AsyncJointRTCRuntime, AsyncJointRTCRuntimeConfig
from qi.deployment.async25_adapters import (
    ExecutingActionSink, LiveImages, ObservationSource, PassivePiperState,
    ShadowActionSink, build_action_governor,
)
from fastwam_steerquant.robot_profiles import make_client, validate_profile, baseline_checkpoint


class QuantizedObservationSource(ObservationSource):
    def read_observation(self, t):
        observation = super().read_observation(t)
        observation["image_channel_order"] = "RGB" if self.images else "as_stored"
        return observation


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.add_argument("--max-steps", type=int, default=128)
    # Explicit opt-in; never inherit motion flags from a YAML file.
    p.add_argument("--execute-actions", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--move-to-initial", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--no-robot", action="store_true", help="Use recorded state; do not open CAN")
    p.add_argument("--live-cameras", action="store_true", help="With --no-robot, use ROS images + recorded state")
    p.add_argument("--episode-hdf5")
    p.add_argument("--output-dir")
    p.add_argument("--yes", action="store_true", help="Skip Enter prompt (shadow only)")
    args = p.parse_args(argv)
    if args.max_steps <= 0:
        p.error("--max-steps must be positive")
    if args.move_to_initial and not args.execute_actions:
        p.error("--move-to-initial requires --execute-actions")
    if args.execute_actions and (args.no_robot or args.yes):
        p.error("--execute-actions cannot be combined with --no-robot or --yes")
    if args.no_robot and not args.episode_hdf5:
        p.error("--no-robot requires --episode-hdf5")
    if args.episode_hdf5 and not args.no_robot:
        p.error("--episode-hdf5 is only used with --no-robot")
    return args


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def save_results(runtime, output, metadata):
    output.mkdir(parents=True, exist_ok=True)
    report = {**runtime.report(), **metadata}
    trace = runtime.action_trace_records()
    report["overlap_steps"] = sum(len(r.cursors) > 1 for r in trace)
    report["ramping_steps"] = sum(any(0 < ramp < 1 for ramp in r.ramps) for r in trace)
    (output / "report.json").write_text(json.dumps(report, default=json_default, indent=2) + "\n")
    np.save(output / "selected_actions.npy", np.asarray(runtime.selected_actions, dtype=np.float32).reshape(-1, 14))
    for name, records in (("action_trace", trace), ("inference_chunks", runtime.inference_chunk_records())):
        with (output / f"{name}.jsonl").open("w") as handle:
            for record in records:
                payload = runtime._json_safe(asdict(record))
                handle.write(json.dumps(payload, default=json_default, allow_nan=False) + "\n")
    print(json.dumps(report, default=json_default, indent=2), flush=True)
    print(f"[asi-async50] results: {output}", flush=True)


def prepare_action_sink(args, config, state, stack, metadata):
    """Initialize before the Enter gate only when movement was requested."""
    def connect():
        sink = ExecutingActionSink(config, state)
        stack.callback(sink.close)
        metadata["robot_enabled_by_client"] = True
        return sink

    sink = ShadowActionSink()
    if args.execute_actions and args.move_to_initial:
        sink = connect()
        print("[asi-async50] moving to configured initial pose before waiting for Enter", flush=True)
        sink.move_initial()
        print("[asi-async50] initial pose commands complete; waiting to start policy execution", flush=True)
    if not args.yes:
        input("Press Enter to start policy inference/execution; Ctrl+C to stop: ")
    if args.execute_actions and not args.move_to_initial:
        sink = connect()
    return sink


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    cfg = config["runtime"]
    mode, control_hz = validate_profile(config)
    if cfg.get("arm_mode", "dual") != "dual":
        raise ValueError("This entrypoint requires dual arms")
    if config["wam"].get("quantization", {}).get("backend") not in ("cutlass_w4a8", "cutlass_w4a4"):
        raise ValueError("This entrypoint requires an audited packed deployment contract")
    if int(cfg["action_chunk_size"]) != 32 or int(config["wam"]["action_horizon"]) != 32:
        raise ValueError("This experiment requires a 32-step action horizon")
    output = Path(args.output_dir or cfg["output_dir"]) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    metadata = {"execute_actions": args.execute_actions, "robot_enabled_by_client": False,
                "state_source": args.episode_hdf5 if args.no_robot else "passive_socketcan",
                "live_cameras": not args.no_robot or args.live_cameras, "config": str(config_path)}
    runtime = None
    with ExitStack() as stack:
        server = config["policy_server"]
        client = stack.enter_context(make_client(config, host=args.host, port=args.port))
        client.reset()
        diagnostics = client._request({"op": "diagnostics"})
        client.validate_diagnostics(diagnostics, require_summary=False)
        expected_checkpoint = Path(baseline_checkpoint(config) if mode == "bf16" else config["wam"]["ckpt"]).expanduser()
        if not expected_checkpoint.is_absolute():
            expected_checkpoint = config_path.parent / expected_checkpoint
        if diagnostics.get("checkpoint") and Path(diagnostics["checkpoint"]).resolve() != expected_checkpoint.resolve():
            raise RuntimeError("Server checkpoint does not match this task's client config")
        metadata["server"] = diagnostics
        state = images = episode = None
        if args.no_robot:
            import h5py
            episode = stack.enter_context(h5py.File(args.episode_hdf5, "r"))
        else:
            state = PassivePiperState(config["robot"])
            stack.callback(state.close)
            print(f"[asi-async50] passive joint state: {state.get().tolist()}", flush=True)
        if metadata["live_cameras"]:
            images = LiveImages(config)
            stack.callback(images.close)
            images.get(timeout_s=10.0)
        source = QuantizedObservationSource(config, state=state, images=images, episode=episode)
        initial_observation = source.read_observation(0)
        # Validate shaping settings and task statistics before any robot enable.
        build_action_governor(config, initial_observation["state"], config_path.parent)
        mode = "EXECUTE robot actions" if args.execute_actions else "SHADOW (no enable, no motion commands)"
        print(f"[asi-async50] {mode}; {control_hz:g} Hz, horizon=32, smoothstep=5", flush=True)
        sink = prepare_action_sink(args, config, state, stack, metadata)
        # Anchor to the last accepted initialization command when executing;
        # shadow mode uses a fresh observed state after the Enter gate.
        initial_action = sink.previous if args.execute_actions else source.read_observation(0)["state"]
        governor = build_action_governor(config, initial_action, config_path.parent)
        metadata["action_shaping_stats"] = config["wam"]["dataset_stats"] if governor else None
        print(f"[asi-async50] continuous action shaping: {'ON' if governor else 'OFF'}", flush=True)
        runtime = AsyncJointRTCRuntime(policy=client, observation_source=source, action_sink=sink,
            action_governor=governor,
            config=AsyncJointRTCRuntimeConfig(control_hz=control_hz, chunk_size=32, max_steps=args.max_steps,
                exp_weight_factor=float(cfg.get("exp_weight_factor", .2)),
                arrival_smoothstep_steps=int(cfg.get("arrival_smoothstep_steps", 5)),
                prompt=config["wam"]["prompt"], stop_on_spacebar=True,
                fresh_chunk_grace_s=float(cfg.get("fresh_chunk_grace_s", .6))))
        try:
            runtime.run()
            metadata["server_last_inference"] = client.last_diagnostics
        except KeyboardInterrupt:
            metadata["interrupted"] = True
        except BaseException as exc:
            metadata["error"] = repr(exc)
            raise
        finally:
            # run() always joins the sole producer before observation/RPC cleanup.
            save_results(runtime, output, metadata)


if __name__ == "__main__":
    main()
