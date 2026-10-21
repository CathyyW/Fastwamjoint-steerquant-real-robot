#!/usr/bin/env python3
"""Recorded-HDF asynchronous consumer (legacy 50 / BF16 25 / W4A8 40 Hz). No hardware options or hardware objects.

Uses the unmodified worktree's runtime, observation reader, governor and shadow
sink. Hardware module imports are actively refused in this mock process.
"""
import argparse
import builtins
from dataclasses import asdict
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import yaml

from fastwam_steerquant.robot_profiles import make_client, validate_profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--port", type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shutdown-server", action="store_true")
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("max-steps must be positive")
    config = yaml.safe_load(Path(args.config).read_text())
    workspace = Path(config["offline"]["async_workspace"])
    sys.path.insert(0, str(workspace / "src"))
    original_import = builtins.__import__

    def guarded_import(name, *a, **kw):
        if name.split(".")[0] in {"rospy", "sensor_msgs", "pyAgxArm", "piper_sdk"}:
            raise RuntimeError(f"Hardware import forbidden in recorded mock: {name}")
        return original_import(name, *a, **kw)

    builtins.__import__ = guarded_import
    from qi.deployment.async_joint_runtime import AsyncJointRTCRuntime, AsyncJointRTCRuntimeConfig
    from qi.deployment.async25_adapters import ObservationSource, ShadowActionSink, build_action_governor

    class RecordedSource(ObservationSource):
        def read_observation(self, t):
            obs = super().read_observation(t)
            obs["image_channel_order"] = config["offline"]["image_channel_order"]
            return obs

    server = config["policy_server"]
    q, r = config["wam"]["quantization"], config["runtime"]
    mode, control_hz = validate_profile(config)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    runtime = None
    error = None
    diagnostics = None
    try:
        with h5py.File(config["offline"]["episode_hdf5"], "r") as episode, make_client(
                config, port=args.port) as client:
            try:
                client.reset()
                diagnostics = client._request({"op": "diagnostics"})
                client.validate_diagnostics(diagnostics, require_summary=False)
                source = RecordedSource(config, episode=episode)
                governor = build_action_governor(config, source.read_observation(0)["state"], Path(args.config).parent)
                runtime = AsyncJointRTCRuntime(policy=client, observation_source=source,
                    action_sink=ShadowActionSink(), action_governor=governor,
                    config=AsyncJointRTCRuntimeConfig(control_hz=control_hz, chunk_size=32, max_steps=args.max_steps,
                        exp_weight_factor=float(r["exp_weight_factor"]),
                        arrival_smoothstep_steps=int(r["arrival_smoothstep_steps"]),
                        prompt=config["wam"]["prompt"], stop_on_spacebar=False,
                        fresh_chunk_grace_s=float(r["fresh_chunk_grace_s"])))
                runtime.run()
                diagnostics = client.last_diagnostics
            finally:
                diagnostics = client.last_diagnostics or diagnostics
                if args.shutdown_server:
                    client._request({"op": "shutdown"})
    except BaseException as exc:
        error = repr(exc)
        cause = exc.__cause__
        while cause is not None:
            error += " <- " + repr(cause)
            cause = cause.__cause__
        raise
    finally:
        report = {"hardware_import_guard": True, "execute_actions": False,
                  "robot_objects_created": False, "error": error,
                  "runtime": runtime.report() if runtime else None, "diagnostics": diagnostics}
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        if runtime:
            np.save(output / "selected_actions.npy", np.asarray(runtime.selected_actions))
            for name in ("inference_chunks", "action_trace"):
                records = [asdict(x) for x in getattr(runtime, name)]
                (output / f"{name}.json").write_text(json.dumps(records, default=lambda v: v.tolist()))


if __name__ == "__main__":
    main()
