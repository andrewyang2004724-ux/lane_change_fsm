# evaluation/benchmark.py
import os
import sys
import json
import csv
import time
import argparse
import subprocess
import statistics as st
from pathlib import Path
from copy import deepcopy


DEFAULT_SCENARIOS = ["FREE_FLOW", "FOLLOW_SLOW_LEAD", "LEFT_BLOCKED", "DENSE_TRAFFIC"]


def parse_seed_list(seed_text: str):
    """
    支持:
    - "1,2,3,4"
    - "1-10"
    """
    seed_text = seed_text.strip()
    if "-" in seed_text and "," not in seed_text:
        a, b = seed_text.split("-")
        a, b = int(a), int(b)
        if a > b:
            a, b = b, a
        return list(range(a, b + 1))
    return [int(x.strip()) for x in seed_text.split(",") if x.strip()]


def flatten_dict(d, prefix=""):
    """
    将嵌套字典扁平化，例如 state_ratio.LANE_CHANGE_LEFT -> state_ratio__LANE_CHANGE_LEFT
    """
    out = {}
    for k, v in d.items():
        nk = f"{prefix}__{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, nk))
        else:
            out[nk] = v
    return out


def is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def aggregate_by_scenario(raw_rows, group_key="scenario"):
    """
    对 numeric 字段做 mean/std/min/max
    """
    groups = {}
    for r in raw_rows:
        g = r.get(group_key, "UNKNOWN")
        groups.setdefault(g, []).append(r)

    agg_rows = []
    for g, rows in groups.items():
        agg = {group_key: g, "n_runs": len(rows)}
        # 收集 numeric keys
        keys = sorted({k for r in rows for k, v in r.items() if is_number(v)})
        for k in keys:
            vals = [r[k] for r in rows if is_number(r.get(k))]
            if not vals:
                continue
            agg[f"{k}__mean"] = sum(vals) / len(vals)
            agg[f"{k}__std"] = st.pstdev(vals) if len(vals) > 1 else 0.0
            agg[f"{k}__min"] = min(vals)
            agg[f"{k}__max"] = max(vals)
        agg_rows.append(agg)

    # 固定排序
    agg_rows.sort(key=lambda x: x.get(group_key, ""))
    return agg_rows


def main():
    parser = argparse.ArgumentParser("Batch benchmark for lane_change_fsm")
    parser.add_argument("--scenarios", type=str, default=",".join(DEFAULT_SCENARIOS),
                        help="逗号分隔，例如 FREE_FLOW,FOLLOW_SLOW_LEAD")
    parser.add_argument("--seeds", type=str, default="1-10",
                        help='例如 "1-10" 或 "1,2,3,4"')
    parser.add_argument("--num_traffic", type=int, default=45)
    parser.add_argument("--max_ticks", type=int, default=4000)
    parser.add_argument("--record_video", type=int, default=0)
    parser.add_argument("--show_top_view", type=int, default=0)
    parser.add_argument("--output_root", type=str, default="outputs/benchmark")
    parser.add_argument("--python_exec", type=str, default=sys.executable)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    ts = time.strftime("%Y%m%d_%H%M%S")
    bench_dir = repo_root / args.output_root / f"bench_{ts}"
    bench_dir.mkdir(parents=True, exist_ok=True)

    scenarios = [x.strip() for x in args.scenarios.split(",") if x.strip()]
    seeds = parse_seed_list(args.seeds)

    # 保存实验配置
    cfg = {
        "timestamp": ts,
        "repo_root": str(repo_root),
        "scenarios": scenarios,
        "seeds": seeds,
        "num_traffic": args.num_traffic,
        "max_ticks": args.max_ticks,
        "record_video": args.record_video,
        "show_top_view": args.show_top_view,
        "python_exec": args.python_exec
    }
    with open(bench_dir / "benchmark_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    raw_rows = []
    failed_rows = []

    total_runs = len(scenarios) * len(seeds)
    run_idx = 0

    for sc in scenarios:
        for sd in seeds:
            run_idx += 1
            out_dir = bench_dir / sc / f"seed_{sd}"
            out_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                args.python_exec, "main.py",
                "--scenario", sc,
                "--seed", str(sd),
                "--num_traffic", str(args.num_traffic),
                "--max_ticks", str(args.max_ticks),
                "--record_video", str(args.record_video),
                "--show_top_view", str(args.show_top_view),
                "--out_dir", str(out_dir)
            ]

            print(f"[{run_idx}/{total_runs}] Running: scenario={sc}, seed={sd}")
            print("  CMD:", " ".join(cmd))

            t0 = time.time()
            ret = subprocess.run(cmd, cwd=repo_root)
            dt = time.time() - t0

            if ret.returncode != 0:
                failed_rows.append({
                    "scenario": sc,
                    "seed": sd,
                    "return_code": ret.returncode,
                    "elapsed_s": round(dt, 3),
                    "out_dir": str(out_dir)
                })
                print(f"  -> FAILED (code={ret.returncode})")
                continue

            # 优先读取 paper_metrics.json
            metrics_path = out_dir / "paper_metrics.json"
            summary_path = out_dir / "summary.json"

            if not metrics_path.exists():
                failed_rows.append({
                    "scenario": sc,
                    "seed": sd,
                    "return_code": 0,
                    "elapsed_s": round(dt, 3),
                    "out_dir": str(out_dir),
                    "reason": "paper_metrics.json not found"
                })
                print("  -> FAILED (paper_metrics.json missing)")
                continue

            row = load_json(metrics_path)
            row = flatten_dict(row)

            # 尝试补充 summary
            if summary_path.exists():
                s = flatten_dict(load_json(summary_path), prefix="summary")
                row.update(s)

            row["scenario"] = sc
            row["seed"] = sd
            row["elapsed_s"] = round(dt, 3)
            row["out_dir"] = str(out_dir)
            raw_rows.append(row)

            print("  -> OK")

    raw_csv = bench_dir / "raw_results.csv"
    grouped_csv = bench_dir / "grouped_results.csv"
    failed_csv = bench_dir / "failed_runs.csv"

    write_csv(raw_csv, raw_rows)
    write_csv(failed_csv, failed_rows)

    grouped_rows = aggregate_by_scenario(raw_rows, group_key="scenario")
    write_csv(grouped_csv, grouped_rows)

    print("\n========== BENCHMARK DONE ==========")
    print(f"Output dir      : {bench_dir}")
    print(f"Raw results     : {raw_csv}")
    print(f"Grouped results : {grouped_csv}")
    print(f"Failed runs     : {failed_csv}")
    print(f"Success runs    : {len(raw_rows)} / {total_runs}")


if __name__ == "__main__":
    main()
