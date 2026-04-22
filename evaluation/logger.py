# evaluation/logger.py
import csv
import math


def _vec_norm(v):
    return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def _pick(d, keys, default=None):
    if not isinstance(d, dict):
        return default
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _state_name(s):
    if hasattr(s, "name"):
        return s.name
    return str(s) if s is not None else "UNKNOWN"


class DataLogger:
    """
    兼容 main.py 的日志器:
    - logger.log(...)
    - logger.rows
    - logger.save_csv(...)
    """
    def __init__(self, dt=0.05):
        self.dt = dt
        self.rows = []
        self._prev_acc = None

    def log(self, tick, ego_vehicle, state, scene, extra=None):
        extra = extra or {}

        # ego speed
        v = ego_vehicle.get_velocity()
        speed = _vec_norm(v)

        # ego accel
        try:
            a_vec = ego_vehicle.get_acceleration()
            acc = _vec_norm(a_vec)
        except Exception:
            acc = 0.0

        # jerk
        if self._prev_acc is None:
            jerk = 0.0
        else:
            jerk = (acc - self._prev_acc) / max(self.dt, 1e-3)
        self._prev_acc = acc

        # 前车信息（兼容多种字段）
        front_dist = None
        front_speed = None
        front_rel_speed = None

        if isinstance(scene, dict):
            curr_front = scene.get("curr_front", None)
            if isinstance(curr_front, dict):
                front_dist = curr_front.get("dist", None)
                front_speed = curr_front.get("speed", None)

            if front_dist is None:
                front_dist = _pick(scene, ["curr_front_dist", "front_dist", "front_distance_m"], None)

            front_rel_speed = _pick(scene, ["curr_front_rel_speed", "front_rel_speed", "rel_speed_mps"], None)

            if front_rel_speed is None and front_speed is not None and front_speed < 900:
                front_rel_speed = speed - front_speed

        # TTC
        ttc = None
        if front_dist is not None and front_rel_speed is not None and front_rel_speed > 1e-3:
            ttc = front_dist / front_rel_speed

        # lane_id
        lane_id = None
        if isinstance(scene, dict):
            ego_wp = scene.get("ego_wp", None)
            lane_id = getattr(ego_wp, "lane_id", None) if ego_wp is not None else None
            if lane_id is None:
                lane_id = _pick(scene, ["ego_lane_id", "lane_id"], None)

        # min surround dist（可选）
        min_surround = None
        if isinstance(scene, dict):
            min_surround = _pick(scene, ["min_surround_dist", "min_dist"], None)
            if min_surround is None:
                cands = []
                for k in ["curr_front", "left_front", "left_rear", "right_front", "right_rear"]:
                    obj = scene.get(k, None)
                    if isinstance(obj, dict):
                        d = obj.get("dist", None)
                        if d is not None and d < 999:
                            cands.append(d)
                if cands:
                    min_surround = min(cands)

        row = {
            "tick": tick,
            "state": _state_name(state),

            # 兼容 build_paper_metrics
            "speed": speed,
            "acc": acc,
            "jerk": jerk,
            "ttc": ttc,

            # 新字段
            "ego_speed_mps": speed,
            "ego_accel_mps2": acc,
            "ego_jerk_mps3": jerk,
            "curr_front_dist": front_dist if front_dist is not None else 999.0,
            "front_rel_speed_mps": front_rel_speed,
            "ttc_front_s": ttc,
            "lane_id": lane_id,
            "min_surround_dist_m": min_surround,
        }

        row.update(extra)
        self.rows.append(row)

    def save_csv(self, path):
        if not self.rows:
            with open(path, "w", newline="", encoding="utf-8") as f:
                f.write("")
            return

        keys = sorted({k for r in self.rows for k in r.keys()})
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in self.rows:
                w.writerow(r)


class RunLogger(DataLogger):
    """
    兼容你之前新增的 benchmark/metrics 调用风格:
    - logger.records
    - logger.log_tick(...)
    - logger.close()
    """
    def __init__(self, dt=0.05, csv_path=None):
        super().__init__(dt=dt)
        self.csv_path = csv_path

    @property
    def records(self):
        return self.rows

    def log_tick(self, tick, sim_time, ego_vehicle, scene, fsm_state, extra=None):
        extra = extra or {}
        extra.setdefault("sim_time", sim_time)
        self.log(tick, ego_vehicle, fsm_state, scene, extra=extra)

    def close(self):
        if self.csv_path:
            self.save_csv(self.csv_path)
