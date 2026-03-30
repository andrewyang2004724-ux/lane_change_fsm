# evaluation/logger.py
import pandas as pd

class DataLogger:
    def __init__(self):
        self.rows = []

    def log(self, tick, ego, state, scene, extra=None):
        loc = ego.get_location()
        vel = ego.get_velocity()
        speed = (vel.x**2 + vel.y**2 + vel.z**2) ** 0.5

        row = {
            "tick": tick,
            "x": loc.x,
            "y": loc.y,
            "z": loc.z,
            "speed": speed,
            "state": state.name,

            "curr_front_dist": scene["curr_front"]["dist"],
            "curr_front_speed": scene["curr_front"]["speed"],

            "left_front_dist": scene["left_front"]["dist"],
            "left_front_speed": scene["left_front"]["speed"],
            "left_rear_dist": scene["left_rear"]["dist"],
            "left_rear_speed": scene["left_rear"]["speed"],

            "right_front_dist": scene["right_front"]["dist"],
            "right_front_speed": scene["right_front"]["speed"],
            "right_rear_dist": scene["right_rear"]["dist"],
            "right_rear_speed": scene["right_rear"]["speed"],
        }

        if scene["ego_wp"] is not None:
            row["lane_id"] = scene["ego_wp"].lane_id
            row["road_id"] = scene["ego_wp"].road_id
        else:
            row["lane_id"] = None
            row["road_id"] = None

        if extra:
            row.update(extra)

        self.rows.append(row)

    def save_csv(self, path):
        df = pd.DataFrame(self.rows)
        df.to_csv(path, index=False)
