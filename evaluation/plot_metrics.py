# evaluation/plot_metrics.py
import csv
import argparse
from pathlib import Path
import matplotlib.pyplot as plt


def read_csv(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def to_float(x):
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def group_values(rows, scenario_key, metric_key):
    d = {}
    for r in rows:
        sc = r.get(scenario_key, "UNKNOWN")
        v = to_float(r.get(metric_key))
        if v is None:
            continue
        d.setdefault(sc, []).append(v)
    return d


def bar_mean(metric_dict, title, ylabel, out_path):
    scenarios = sorted(metric_dict.keys())
    means = [sum(metric_dict[s]) / len(metric_dict[s]) if metric_dict[s] else 0 for s in scenarios]

    plt.figure(figsize=(8, 4.5))
    plt.bar(scenarios, means)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def boxplot(metric_dict, title, ylabel, out_path):
    scenarios = sorted(metric_dict.keys())
    data = [metric_dict[s] for s in scenarios]

    plt.figure(figsize=(8, 4.5))
    plt.boxplot(data, labels=scenarios, showfliers=True)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser("Plot benchmark metrics")
    parser.add_argument("--bench_dir", type=str, required=True,
                        help="例如 outputs/benchmark/bench_20260101_120000")
    args = parser.parse_args()

    bench_dir = Path(args.bench_dir)
    raw_csv = bench_dir / "raw_results.csv"
    fig_dir = bench_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not raw_csv.exists():
        raise FileNotFoundError(f"Not found: {raw_csv}")

    rows = read_csv(raw_csv)

    # 1) 平均速度（km/h）
    d_speed = group_values(rows, "scenario", "avg_speed_kmh")
    bar_mean(d_speed, "Average Speed by Scenario", "km/h", fig_dir / "avg_speed_bar.png")

    # 2) 最小TTC箱线图
    d_ttc = group_values(rows, "scenario", "min_ttc_s")
    boxplot(d_ttc, "Min TTC Distribution by Scenario", "s", fig_dir / "min_ttc_box.png")

    # 3) 风险事件（TTC<2s）均值
    d_risk = group_values(rows, "scenario", "risk_events_ttc_lt_2s")
    bar_mean(d_risk, "Risk Events (TTC<2s) by Scenario", "count", fig_dir / "risk_events_bar.png")

    # 4) 状态切换次数
    d_switch = group_values(rows, "scenario", "state_switch_count")
    bar_mean(d_switch, "FSM State Switch Count by Scenario", "count", fig_dir / "state_switch_bar.png")

    print(f"[OK] figures saved to: {fig_dir}")


if __name__ == "__main__":
    main()
