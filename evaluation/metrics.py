# evaluation/metrics.py
import json
from collections import Counter

def summarize_logger(logger, collision_count=0, lane_invasion_count=0):
    rows = logger.rows
    if len(rows) == 0:
        return {}

    avg_speed = sum(r["speed"] for r in rows) / len(rows)
    min_front = min(r["curr_front_dist"] for r in rows)
    state_counter = Counter(r["state"] for r in rows)

    summary = {
        "num_ticks": len(rows),
        "avg_speed_mps": avg_speed,
        "avg_speed_kmh": avg_speed * 3.6,
        "min_curr_front_dist": min_front,
        "collision_count": collision_count,
        "lane_invasion_count": lane_invasion_count,
        "state_distribution": dict(state_counter),
    }
    return summary

def save_summary(summary, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
