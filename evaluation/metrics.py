# evaluation/metrics.py
import json
import math
from collections import Counter


def _pick(row, keys, default=None):
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return default


def _safe_vals(values):
    out = []
    for v in values:
        if v is None:
            continue
        try:
            fv = float(v)
            if math.isnan(fv):
                continue
            out.append(fv)
        except Exception:
            continue
    return out


def safe_min(values, default=None):
    vals = _safe_vals(values)
    return min(vals) if vals else default


def safe_mean(values, default=None):
    vals = _safe_vals(values)
    return sum(vals) / len(vals) if vals else default


def percentile(values, p):
    vals = sorted(_safe_vals(values))
    if not vals:
        return None
    k = (len(vals) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(vals) - 1)
    if f == c:
        return vals[f]
    return vals[f] + (vals[c] - vals[f]) * (k - f)


def compute_metrics(records):
    speed = [_pick(r, ["ego_speed_mps", "speed"]) for r in records]
    accel = [_pick(r, ["ego_accel_mps2", "acc"]) for r in records]
    jerk = [_pick(r, ["ego_jerk_mps3", "jerk"]) for r in records]
    ttc = [_pick(r, ["ttc_front_s", "ttc"]) for r in records]
    state = [str(_pick(r, ["state", "fsm_state"], "UNKNOWN")) for r in records]

    lane_change_ticks = sum(1 for s in state if s in ("LANE_CHANGE_LEFT", "LANE_CHANGE_RIGHT"))
    emergency_ticks = sum(1 for s in state if s == "EMERGENCY_BRAKE")

    return {
        "avg_speed_mps": safe_mean(speed),
        "avg_speed_kmh": safe_mean(speed) * 3.6 if safe_mean(speed) is not None else None,
        "max_accel_mps2": max(_safe_vals(accel), default=None),
        "min_accel_mps2": min(_safe_vals(accel), default=None),
        "jerk_p95_mps3": percentile(jerk, 95),
        "ttc_min_s": safe_min(ttc),
        "risk_events_ttc_lt_2s": sum(1 for v in _safe_vals(ttc) if v < 2.0),
        "risk_events_ttc_lt_1p5s": sum(1 for v in _safe_vals(ttc) if v < 1.5),
        "lane_change_ticks": lane_change_ticks,
        "emergency_ticks": emergency_ticks,
        "state_switch_count": sum(1 for i in range(1, len(state)) if state[i] != state[i - 1]),
        "total_ticks": len(records),
    }


def summarize_logger(logger, collision_count=0, lane_invasion_count=0):
    """
    兼容 main.py:
    summary = summarize_logger(logger, collision_count, invasion_count)
    """
    rows = logger.rows if hasattr(logger, "rows") else logger
    metrics = compute_metrics(rows)

    speeds = [_pick(r, ["ego_speed_mps", "speed"]) for r in rows]
    front_dists = [_pick(r, ["curr_front_dist", "front_dist_m"], 999.0) for r in rows]
    states = [str(_pick(r, ["state", "fsm_state"], "UNKNOWN")) for r in rows]

    state_dist = dict(Counter(states))

    summary = {
        "avg_speed_kmh": safe_mean(speeds) * 3.6 if safe_mean(speeds) is not None else 0.0,
        "avg_speed_mps": safe_mean(speeds) if safe_mean(speeds) is not None else 0.0,
        "collision_count": int(collision_count),
        "lane_invasion_count": int(lane_invasion_count),
        "min_curr_front_dist": safe_min(front_dists, default=999.0),
        "num_ticks": len(rows),
        "state_distribution": state_dist,
    }

    # 可选附加
    summary.update({
        "ttc_min_s": metrics.get("ttc_min_s"),
        "risk_events_ttc_lt_2s": metrics.get("risk_events_ttc_lt_2s"),
        "risk_events_ttc_lt_1p5s": metrics.get("risk_events_ttc_lt_1p5s"),
        "state_switch_count": metrics.get("state_switch_count"),
    })
    return summary


def save_summary(summary, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def save_metrics(path, metrics):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
