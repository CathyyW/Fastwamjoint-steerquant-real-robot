#!/usr/bin/env python3
"""Start an isolated local W4A8 server and a hardware-free async50 mock.

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
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--max-steps", type=int, default=128)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / "outputs/robot_async50_validation" / (args.task + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True)
    config = root / f"configs/robot_{args.task}_w4a8_async50.yaml"
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(root / "src"),
               FASTWAM_AUTHKEY=secrets.token_hex(32))
    server_command = [sys.executable, "-u", "-m", "fastwam_steerquant.robot_async_server",
                      "--config", str(config), "--port", str(args.port), "--report", str(output / "server_audit.json")]
    mock_command = [sys.executable, "-u", str(root / "scripts/mock_w4a8_async50.py"),
                    "--config", str(config), "--port", str(args.port), "--max-steps", str(args.max_steps),
                    "--output-dir", str(output / "mock"), "--shutdown-server"]
    status = {"task": args.task, "output": str(output), "hardware": False}
    with (output / "server.log").open("w") as log:
        server = subprocess.Popen(server_command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 300
            while "W4A8 server listening" not in (output / "server.log").read_text():
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
