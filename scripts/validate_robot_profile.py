#!/usr/bin/env python3
"""Start an isolated local selected-profile server and a hardware-free asynchronous mock.

The per-run authentication key exists only in child process environments.
Only the processes created here are stopped on timeout/failure.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("pack", "stack"), required=True)
    parser.add_argument("--mode", choices=("bf16", "w4a8", "w4a4"), required=True)
    parser.add_argument("--quick-benchmark", action="store_true")
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--max-steps", type=int, default=128)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / "outputs/robot_profile_validation" / (args.task + "_" + args.mode + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True)
    rate = 25 if args.mode == "bf16" else 40
    config = root / f"configs/robot_{args.task}_{args.mode}_async{rate}.yaml"
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(root / "src"),
               FASTWAM_AUTHKEY=secrets.token_hex(32))
    server_command = [sys.executable, "-u", str(root / "scripts/serve_robot_profile.py"),
                      "--config", str(config), "--port", str(args.port), "--report", str(output / "server_audit.json")]
    if args.quick_benchmark:
        server_command.append("--quick-benchmark")
    if args.mode == "w4a4":
        env.pop("FASTWAM_RUNTIME_PROFILE", None)
    mock_command = [sys.executable, "-u", str(root / "scripts/mock_w4a8_async50.py"),
                    "--config", str(config), "--port", str(args.port), "--max-steps", str(args.max_steps),
                    "--output-dir", str(output / "mock"), "--shutdown-server"]
    status = {"task": args.task, "mode": args.mode, "control_hz": rate, "output": str(output), "hardware": False}
    occupied = subprocess.check_output([
        "nvidia-smi", "-i", "0", "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader"], text=True).strip()
    (output / "gpu_before.txt").write_text(occupied + "\n")
    if occupied:
        raise RuntimeError("GPU already has a compute process; stop the existing task/service before a serial benchmark. "
                           "No existing process was stopped. See gpu_before.txt")
    with (output / "server.log").open("w") as log:
        server = subprocess.Popen(server_command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 300
            while "PROFILE_READY" not in (output / "server.log").read_text():
                if server.poll() is not None:
                    raise RuntimeError(f"Server startup failed: see {output / 'server.log'}")
                if time.monotonic() > deadline:
                    raise TimeoutError("Server warmup exceeded 300 seconds")
                time.sleep(.25)
            with (output / "mock.log").open("w") as mock_log:
                result = subprocess.run(mock_command, cwd=root, env=env, stdout=mock_log,
                                        stderr=subprocess.STDOUT, timeout=120)
            status["mock_exit_code"] = result.returncode
            server.wait(timeout=20)
            status["server_exit_code"] = server.returncode
        except BaseException as exc:
            status["error"] = repr(exc)
            raise
        finally:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
            (output / "status.json").write_text(json.dumps(status, indent=2))
            print(json.dumps(status), flush=True)
    return status.get("mock_exit_code", 1)


if __name__ == "__main__":
    sys.exit(main())
