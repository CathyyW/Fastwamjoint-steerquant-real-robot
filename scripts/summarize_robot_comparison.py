#!/usr/bin/env python3
"""Aggregate recorded measurements without loading models or touching hardware."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


DIMENSIONS = [f"{side}_{'gripper' if j == 6 else 'joint_'+str(j+1)}"
              for side in ("left", "right") for j in range(7)]
JOINTS = [i for i in range(14) if i not in (6, 13)]


def stats(values):
    a = np.asarray(values, np.float64)
    return dict(n=len(a), mean=float(a.mean()), p50=float(np.median(a)),
                p95=float(np.percentile(a, 95)), min=float(a.min()), max=float(a.max())) if len(a) else None


def error_metrics(baseline, quantized):
    a, b = np.asarray(baseline, np.float64), np.asarray(quantized, np.float64)
    if a.shape != b.shape or a.shape[-1] != 14 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Cannot compare mismatched or nonfinite complete action arrays")
    d = (b-a).reshape(-1, 14)
    return dict(rmse=float(np.sqrt(np.mean(d*d))), mae=float(np.mean(np.abs(d))),
                max_abs=float(np.max(np.abs(d))),
                per_dimension=[dict(name=name, rmse=float(np.sqrt(np.mean(d[:, i]**2))),
                                    mae=float(np.mean(np.abs(d[:, i]))), max_abs=float(np.max(np.abs(d[:, i]))))
                               for i, name in enumerate(DIMENSIONS)])


def physical_metrics(a, b):
    per_dim = error_metrics(a, b)["per_dimension"]
    for i, row in enumerate(per_dim):
        row["unit"] = "m" if i in (6, 13) else "rad"
    d = np.asarray(b, np.float64)-np.asarray(a, np.float64)
    grouped = {}
    for name, inds, scale in (("joints_rad", JOINTS, 1), ("grippers_m", [6, 13], 1), ("grippers_mm", [6, 13], 1000)):
        x = d[..., inds]*scale
        grouped[name] = dict(rmse=float(np.sqrt(np.mean(x*x))), mae=float(np.mean(np.abs(x))), max_abs=float(np.max(np.abs(x))))
    return dict(per_dimension=per_dim, by_unit=grouped)


def summarize_mode(path):
    server = json.loads((path / "server.json").read_text())
    client = json.loads((path / "client.json").read_text())
    actions = np.load(path / "actions.npz", allow_pickle=False)
    raw = server["records"]
    perf = dict(
        denoise_loop_scheduler_gpu_ms=stats([r["denoise_loop_scheduler_gpu_ms"] for r in raw if r["phase"] == "denoise"]),
        core_live_graph_io_gpu_ms=stats([r["core_live_graph_io_gpu_ms"] for r in raw if r["phase"] == "denoise"]),
        model_wall_ms=stats([r["model_wall_ms"] for r in raw if r["phase"] == "latency"]),
        rpc_action_ms=stats([r["rpc_action_ms"] for r in client["records"] if r["phase"] == "latency"]),
        rpc_with_diagnostics_ms=stats([r["rpc_with_diagnostics_ms"] for r in client["records"] if r["phase"] == "latency"]),
        consecutive_rpc_with_diagnostics_ms=stats([r["consecutive_rpc_with_diagnostics_ms"] for r in client["consecutive_pairs"]]))
    async_result = json.loads((path / "async.json").read_text())
    async_rows = [r for r in client["records"] if r["phase"] == "async"]
    # Runtime inference finish precedes publication into the buffer; record exact
    # enqueue visibility separately through the first selectable action trace.
    arrivals = []
    for chunk in async_result.get("inference_chunks", []):
        matches = [r for r in async_rows if r["observation_id"] == f'async:{chunk["probe_cursor"]}']
        traces = [r for r in async_result.get("action_trace", []) if chunk["cursor"] in r["cursors"]]
        if matches and traces:
            request = matches[0]
            arrivals.append(dict(cursor=chunk["cursor"], request_submitted_ns=request["submitted_monotonic_ns"],
                                 action_received_ns=request["received_monotonic_ns"],
                                 inference_finished_ns=chunk["inference_finished_monotonic_ns"],
                                 first_selected_sent_ns=traces[0]["sent_monotonic_ns"],
                                 request_to_first_selected_ms=(traces[0]["sent_monotonic_ns"]-request["submitted_monotonic_ns"])/1e6))
    indices = [i for i, r in enumerate(raw) if r["phase"] == "action"]
    a = {key: actions[key][indices] for key in ("normalized", "physical")}
    rows = [raw[i] for i in indices]
    ranges = {}
    task = server["spec"]["task"]
    root = Path(__file__).resolve().parents[1]
    training = json.loads((root / f"local/robot_handoff/assets/stats/{task}/dataset_stats.json").read_text())["action"]["default"]
    lo, hi = np.asarray(training["global_min"]), np.asarray(training["global_max"])
    for key, data in a.items():
        flat = data.reshape(-1, 14)
        ranges[key] = dict(shape=list(data.shape), nan_count=int(np.isnan(data).sum()), inf_count=int(np.isinf(data).sum()),
                           min_by_dimension=flat.min(axis=0).tolist(), max_by_dimension=flat.max(axis=0).tolist())
    ranges["physical"]["outside_training_extrema_by_dimension"] = ((a["physical"] < lo) | (a["physical"] > hi)).sum(axis=(0, 1)).tolist()
    ranges["normalized"]["abs_gt_5_by_dimension"] = (np.abs(a["normalized"]) > 5).sum(axis=(0, 1)).tolist()
    mode = dict(performance=perf, ranges=ranges, startup=server["startup"], graph=server["graph"],
                peak_allocated_bytes=server["peak_allocated_bytes"], peak_reserved_bytes=server["peak_reserved_bytes"],
                async_error=async_result["error"], async_runtime=async_result["runtime"], async_arrivals=arrivals,
                consecutive_pairs_over_620ms=sum(r["consecutive_rpc_with_diagnostics_ms"] > 620 for r in client["consecutive_pairs"]))
    return mode, a, rows, server, actions, async_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    args = parser.parse_args()
    root = Path(args.input_dir)
    summary, csv_rows, raw_rows, range_rows = {}, [], [], []
    for task in ("pack", "stack"):
        available = {}
        for name in ("bf16", "w4a8"):
            path = root / task / name
            if not (path / "server.json").is_file():
                continue
            available[name] = summarize_mode(path)
            ranges = available[name][0]["ranges"]
            for i, dim in enumerate(DIMENSIONS):
                range_rows.append(dict(task=task, mode=name, dimension=i, name=dim,
                    physical_unit="m" if i in (6, 13) else "rad",
                    normalized_min=ranges["normalized"]["min_by_dimension"][i],
                    normalized_max=ranges["normalized"]["max_by_dimension"][i],
                    physical_min=ranges["physical"]["min_by_dimension"][i],
                    physical_max=ranges["physical"]["max_by_dimension"][i],
                    normalized_abs_gt5=ranges["normalized"]["abs_gt_5_by_dimension"][i],
                    physical_outside_training_extrema=ranges["physical"]["outside_training_extrema_by_dimension"][i]))
            client = json.loads((path / "client.json").read_text())["records"]
            server = available[name][3]["records"]
            if len(client) != len(server):
                raise ValueError("Client/server measurement counts differ")
            for index, (s, c) in enumerate(zip(server, client)):
                if (s["observation_id"], s["phase"]) != (c["observation_id"], c["phase"]):
                    raise ValueError("Client/server request alignment differs")
                raw_rows.append(dict(task=task, mode=name, request_index=index,
                    observation_id=s["observation_id"], phase=s["phase"], seed=s["seed"],
                    model_wall_ms=s["model_wall_ms"], denoise_loop_scheduler_gpu_ms=s.get("denoise_loop_scheduler_gpu_ms"),
                    core_live_graph_io_gpu_ms=s.get("core_live_graph_io_gpu_ms"),
                    rpc_action_ms=c["rpc_action_ms"], rpc_with_diagnostics_ms=c["rpc_with_diagnostics_ms"],
                    submitted_monotonic_ns=c["submitted_monotonic_ns"], received_monotonic_ns=c["received_monotonic_ns"]))
            for boundary, s in available[name][0]["performance"].items():
                if s:
                    csv_rows.append(dict(task=task, mode=name, boundary=boundary, **s))
        result = {name: x[0] for name, x in available.items()}
        if set(available) == {"bf16", "w4a8"}:
            bf, q = available["bf16"], available["w4a8"]
            if [(r["observation_id"], r["seed"]) for r in bf[2]] != [(r["observation_id"], r["seed"]) for r in q[2]]:
                raise ValueError("Mismatched observations/seeds")
            for b, w in zip(bf[2], q[2]):
                for key in ("video_noise", "action_noise", "first_frame", "context"):
                    if b["input_hashes"][key] != w["input_hashes"][key]:
                        raise ValueError(f"Initial input/noise mismatch: {key}")
                if not b["input_hashes"]["graph_copied_current_inputs_exactly"] or not w["input_hashes"]["graph_copied_current_inputs_exactly"]:
                    raise ValueError("Graph stale input detected")
            result["action_error"] = dict(observations=len(bf[2]), full_chunk_shape=list(bf[1]["normalized"].shape),
                initial_noise_and_encoded_condition_hashes_match=True,
                normalized=error_metrics(bf[1]["normalized"], q[1]["normalized"]),
                physical=physical_metrics(bf[1]["physical"], q[1]["physical"]))
            result["speedup"] = {key: {stat: bf[0]["performance"][key][stat]/q[0]["performance"][key][stat]
                                       for stat in ("mean", "p50", "p95")}
                                  for key in bf[0]["performance"]}
            # Align only the SAME recorded input, then evaluate the actual
            # chunk indices that contributed to either runtime's shadow output.
            used = {}
            for item in (bf, q):
                chunks = {c["cursor"]: c for c in item[5].get("inference_chunks", [])}
                for trace in item[5].get("action_trace", []):
                    for cursor, index, weight in zip(trace["cursors"], trace["counter_local_indices"], trace["weights"]):
                        if weight > 0 and cursor in chunks:
                            ident = f'async:{chunks[cursor]["probe_cursor"]}'
                            used.setdefault(ident, set()).add(index)
            consumed = []
            for ident, selected in sorted(used.items()):
                matched = [[i for i, r in enumerate(item[3]["records"]) if r["phase"] == "async" and r["observation_id"] == ident] for item in (bf, q)]
                if not all(matched):
                    continue
                ids = sorted(selected)
                bi, qi = matched[0][0], matched[1][0]
                consumed.append(dict(observation_id=ident, consumed_candidate_indices_union=ids,
                    normalized=error_metrics(bf[4]["normalized"][bi, ids], q[4]["normalized"][qi, ids]),
                    physical=physical_metrics(bf[4]["physical"][bi, ids], q[4]["physical"][qi, ids])))
            result["actual_consumed_candidate_error"] = consumed
        else:
            result["comparison_blocked"] = "Missing verified pack step_015000 baseline; no action error or speedup inferred"
        summary[task] = result
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    if range_rows:
        with (root / "action_ranges.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(range_rows[0]))
            writer.writeheader()
            writer.writerows(range_rows)
    if raw_rows:
        with (root / "raw_timings.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(raw_rows[0]))
            writer.writeheader()
            writer.writerows(raw_rows)
    with (root / "performance.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "mode", "boundary", "n", "mean", "p50", "p95", "min", "max"])
        writer.writeheader()
        writer.writerows(csv_rows)
    error_rows = []
    for task, result in summary.items():
        if "action_error" in result:
            for space in ("normalized", "physical"):
                for row in result["action_error"][space]["per_dimension"]:
                    error_rows.append({"task": task, "space": space, "unit": "normalized", **row})
    with (root / "action_error.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "space", "name", "unit", "rmse", "mae", "max_abs"])
        writer.writeheader()
        writer.writerows(error_rows)
    print(root / "summary.json")


if __name__ == "__main__":
    main()
