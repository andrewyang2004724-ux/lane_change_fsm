# decision/safety.py
import math
from config import MIN_GAP, TIME_HEADWAY, COMFORT_DEC

def speed_norm(vec):
    return math.sqrt(vec.x ** 2 + vec.y ** 2 + vec.z ** 2)

def required_front_gap(v_ego, v_front):
    closing = max(0.0, v_ego - v_front)
    return MIN_GAP + TIME_HEADWAY * v_ego + (closing ** 2) / (2.0 * COMFORT_DEC)

def required_rear_gap(v_ego, v_rear):
    closing = max(0.0, v_rear - v_ego)
    return MIN_GAP + TIME_HEADWAY * v_rear + (closing ** 2) / (2.0 * COMFORT_DEC)

def compute_ttc(distance, closing_speed):
    if closing_speed <= 1e-6:
        return float("inf")
    return max(0.0, distance / closing_speed)

def is_safe_to_change(v_ego, front_dist, front_speed, rear_dist, rear_speed,
                      ttc_front_min, ttc_rear_min):
    front_req = required_front_gap(v_ego, front_speed)
    rear_req = required_rear_gap(v_ego, rear_speed)

    front_ttc = compute_ttc(front_dist, max(0.0, v_ego - front_speed))
    rear_ttc = compute_ttc(rear_dist, max(0.0, rear_speed - v_ego))

    safe = (
        front_dist >= front_req and
        rear_dist >= rear_req and
        front_ttc >= ttc_front_min and
        rear_ttc >= ttc_rear_min
    )

    return {
        "safe": safe,
        "front_req": front_req,
        "rear_req": rear_req,
        "front_ttc": front_ttc,
        "rear_ttc": rear_ttc
    }
