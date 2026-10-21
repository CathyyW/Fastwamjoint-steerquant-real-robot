#!/usr/bin/env python3
"""Offline recorded-observation comparison server; never imports robot code."""
import argparse
import json
from multiprocessing.connection import Listener
from pathlib import Path
import time

import numpy as np
import torch

from fastwam_steerquant.robot_async_server import read_config, PackedRobotPolicy, serve_connection
from fastwam_steerquant.robot_async_client import authkey_from_config
from fastwam_steerquant.robot_comparison import BF16RobotPolicy, MeasuredPolicy, observation, audit_live_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--mode", choices=("bf16", "w4a8"), required=True)
    parser.add_argument("--port", type=int, default=18766)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = read_config(args.config)
    spec = json.loads(Path(args.spec).read_text())
    records = [observation(spec, config, i) for i in spec["frame_indices"]]
    started = time.perf_counter()
    if args.mode == "bf16":
        policy = BF16RobotPolicy(config, spec["baseline_checkpoint"])
    else:
        policy = PackedRobotPolicy(config)
    torch.cuda.synchronize()
    loading_s = time.perf_counter()-started
    started = time.perf_counter()
    if args.mode == "w4a8":
        policy.warmup(records[0])
    else:
        for _ in range(3):
            policy.infer(records[0])
    live_checks = audit_live_inputs(policy, records)
    for _ in range(3):
        policy.infer(records[0])
    torch.cuda.synchronize()
    startup = dict(mode=args.mode, loading_s=loading_s, warmup_and_audit_s=time.perf_counter()-started,
                   graph_live_input_checks=live_checks, contract=policy.contract,
                   diagnostics=policy.diagnostics(), gpu=torch.cuda.get_device_name(),
                   torch=str(torch.__version__), cuda=torch.version.cuda,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved())
    (output / "startup.json").write_text(json.dumps(startup, ensure_ascii=False, indent=2))
    measured = MeasuredPolicy(policy)
    try:
        with Listener(("127.0.0.1", args.port), authkey=authkey_from_config(config)) as listener:
            print("COMPARISON_READY", flush=True)
            while True:
                with listener.accept() as conn:
                    measured.observation = None
                    if serve_connection(measured, conn):
                        break
    finally:
        graph = policy.policy.model._rollout_graph
        report = dict(startup=startup, spec=spec, records=measured.records,
                      graph=dict(entries=len(graph.entries), replays=graph.replays, chunks=graph.chunks,
                                 validation_records=graph.validation_records),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
        (output / "server.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        if measured.records:
            np.savez(output / "actions.npz", normalized=np.stack(measured.normalized), physical=np.stack(measured.physical))


if __name__ == "__main__":
    main()
