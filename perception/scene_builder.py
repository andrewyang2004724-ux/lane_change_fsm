# perception/scene_builder.py
from perception.lane_utils import (
    get_current_waypoint,
    get_left_lane_if_possible,
    get_right_lane_if_possible,
    lane_change_left_allowed,
    lane_change_right_allowed,
)
from perception.vehicle_filter import (
    get_lane_vehicles,
    split_front_rear,
    vehicle_speed,
    longitudinal_distance,
)
from config import DETECTION_RADIUS

# =========================================================
# 轻量缓存：降低跨线/压线阶段 waypoint 瞬时抖动
# =========================================================
_LAST_SCENE_CACHE = {
    "ego_wp": None,
    "left_wp": None,
    "right_wp": None,
    "ego_lane_id": None,
    "ego_road_id": None,
    "left_hold_ticks": 0,
    "right_hold_ticks": 0,
}

# 邻道 waypoint 丢失后的短时保持帧数
_NEIGHBOR_HOLD_TICKS = 6

def vehicle_info_relative(ego, veh):
    if veh is None:
        return {
            "vehicle": None,
            "dist": 999.0,
            "speed": 999.0,
        }
    return {
        "vehicle": veh,
        "dist": abs(longitudinal_distance(ego, veh)),
        "speed": vehicle_speed(veh),
    }

def _same_road(a, b):
    if a is None or b is None:
        return False
    return getattr(a, "road_id", None) == getattr(b, "road_id", None)

def _same_section(a, b):
    if a is None or b is None:
        return False
    return (
        getattr(a, "road_id", None) == getattr(b, "road_id", None)
        and getattr(a, "section_id", None) == getattr(b, "section_id", None)
    )

def _lane_gap_ok(curr_wp, candidate, max_gap=2):
    if curr_wp is None or candidate is None:
        return False
    if not hasattr(curr_wp, "lane_id") or not hasattr(candidate, "lane_id"):
        return False
    return abs(candidate.lane_id - curr_wp.lane_id) <= max_gap

def _is_reasonable_neighbor(curr_wp, candidate, side="right"):
    """
    判断 candidate 是否像 curr_wp 的相邻车道 waypoint。
    这里只做工程上的稳健过滤，不追求地图语义绝对严格。
    """
    if curr_wp is None or candidate is None:
        return False

    if not _same_section(curr_wp, candidate):
        return False

    if not _lane_gap_ok(curr_wp, candidate, max_gap=2):
        return False

    if not hasattr(curr_wp, "lane_id") or not hasattr(candidate, "lane_id"):
        return False

    curr_lane = curr_wp.lane_id
    cand_lane = candidate.lane_id

    # CARLA 中 lane_id 正负方向可能和直觉不完全一致，
    # 所以这里不强依赖“左一定增 / 右一定减”，只避免完全同 lane。
    if cand_lane == curr_lane:
        return False

    return True

def _advance_wp(wp, step=3.0):
    if wp is None:
        return None
    try:
        nxts = wp.next(step)
        if nxts and len(nxts) > 0:
            return nxts[0]
    except Exception:
        pass
    return wp

def _recover_neighbor_wp(curr_wp, cached_wp, side="right"):
    """
    邻道 waypoint 瞬时丢失时，尝试从缓存 waypoint 恢复。
    """
    if curr_wp is None or cached_wp is None:
        return None

    if not _same_section(curr_wp, cached_wp):
        return None

    # 优先向前推进一点，减少使用过旧 waypoint
    candidate = _advance_wp(cached_wp, step=3.0)

    if _is_reasonable_neighbor(curr_wp, candidate, side=side):
        return candidate

    # 如果 next 后不合理，退回原 cached_wp 再试一次
    if _is_reasonable_neighbor(curr_wp, cached_wp, side=side):
        return cached_wp

    return None

def _get_lane_vehicles_safe(world, world_map, ego, lane_wp, radius):
    if lane_wp is None:
        return []
    try:
        return get_lane_vehicles(world, world_map, ego, lane_wp, radius)
    except Exception:
        return []

def _split_front_rear_safe(vehicles):
    if not vehicles:
        return None, None
    try:
        return split_front_rear(vehicles)
    except Exception:
        return None, None

def build_scene(world, world_map, ego):
    global _LAST_SCENE_CACHE

    ego_wp = get_current_waypoint(world_map, ego)

    left_wp = get_left_lane_if_possible(ego_wp) if ego_wp is not None else None
    right_wp = get_right_lane_if_possible(ego_wp) if ego_wp is not None else None

    # -----------------------------------------------------
    # 邻道 waypoint 恢复 + 短时 hold
    # -----------------------------------------------------
    if ego_wp is not None:
        cached_left = _LAST_SCENE_CACHE["left_wp"]
        cached_right = _LAST_SCENE_CACHE["right_wp"]

        # 恢复 left
        if left_wp is None and cached_left is not None:
            recovered_left = _recover_neighbor_wp(ego_wp, cached_left, side="left")
            if recovered_left is not None:
                left_wp = recovered_left
                _LAST_SCENE_CACHE["left_hold_ticks"] = _NEIGHBOR_HOLD_TICKS
        elif left_wp is not None:
            _LAST_SCENE_CACHE["left_hold_ticks"] = 0
        else:
            if _LAST_SCENE_CACHE["left_hold_ticks"] > 0 and cached_left is not None:
                recovered_left = _recover_neighbor_wp(ego_wp, cached_left, side="left")
                if recovered_left is not None:
                    left_wp = recovered_left
                _LAST_SCENE_CACHE["left_hold_ticks"] -= 1

        # 恢复 right
        if right_wp is None and cached_right is not None:
            recovered_right = _recover_neighbor_wp(ego_wp, cached_right, side="right")
            if recovered_right is not None:
                right_wp = recovered_right
                _LAST_SCENE_CACHE["right_hold_ticks"] = _NEIGHBOR_HOLD_TICKS
        elif right_wp is not None:
            _LAST_SCENE_CACHE["right_hold_ticks"] = 0
        else:
            if _LAST_SCENE_CACHE["right_hold_ticks"] > 0 and cached_right is not None:
                recovered_right = _recover_neighbor_wp(ego_wp, cached_right, side="right")
                if recovered_right is not None:
                    right_wp = recovered_right
                _LAST_SCENE_CACHE["right_hold_ticks"] -= 1

    # -----------------------------------------------------
    # 当前 / 左 / 右车道车辆
    # -----------------------------------------------------
    curr_vehicles = _get_lane_vehicles_safe(world, world_map, ego, ego_wp, DETECTION_RADIUS)
    curr_front, curr_rear = _split_front_rear_safe(curr_vehicles)

    left_front = left_rear = None
    right_front = right_rear = None

    left_vehicles = _get_lane_vehicles_safe(world, world_map, ego, left_wp, DETECTION_RADIUS)
    if left_vehicles:
        left_front, left_rear = _split_front_rear_safe(left_vehicles)

    right_vehicles = _get_lane_vehicles_safe(world, world_map, ego, right_wp, DETECTION_RADIUS)
    if right_vehicles:
        right_front, right_rear = _split_front_rear_safe(right_vehicles)

    # -----------------------------------------------------
    # allowed 计算
    # 注意：allowed 应基于当前 ego 所在车道，而不是恢复出的邻道
    # -----------------------------------------------------
    left_allowed = lane_change_left_allowed(ego_wp) if ego_wp is not None else False
    right_allowed = lane_change_right_allowed(ego_wp) if ego_wp is not None else False

    scene = {
        "ego_wp": ego_wp,
        "left_wp": left_wp,
        "right_wp": right_wp,
        "left_allowed": left_allowed,
        "right_allowed": right_allowed,

        "curr_front": vehicle_info_relative(ego, curr_front),
        "curr_rear": vehicle_info_relative(ego, curr_rear),

        "left_front": vehicle_info_relative(ego, left_front),
        "left_rear": vehicle_info_relative(ego, left_rear),

        "right_front": vehicle_info_relative(ego, right_front),
        "right_rear": vehicle_info_relative(ego, right_rear),

        # 调试字段，方便你在日志里看 scene 是否在抖
        "debug_scene": {
            "ego_lane_id": getattr(ego_wp, "lane_id", None) if ego_wp is not None else None,
            "ego_road_id": getattr(ego_wp, "road_id", None) if ego_wp is not None else None,
            "ego_section_id": getattr(ego_wp, "section_id", None) if ego_wp is not None else None,

            "left_lane_id": getattr(left_wp, "lane_id", None) if left_wp is not None else None,
            "right_lane_id": getattr(right_wp, "lane_id", None) if right_wp is not None else None,

            "left_hold_ticks": _LAST_SCENE_CACHE["left_hold_ticks"],
            "right_hold_ticks": _LAST_SCENE_CACHE["right_hold_ticks"],
        }
    }

    # -----------------------------------------------------
    # 更新缓存
    # -----------------------------------------------------
    if ego_wp is not None:
        _LAST_SCENE_CACHE["ego_wp"] = ego_wp
        _LAST_SCENE_CACHE["ego_lane_id"] = getattr(ego_wp, "lane_id", None)
        _LAST_SCENE_CACHE["ego_road_id"] = getattr(ego_wp, "road_id", None)

    if left_wp is not None:
        _LAST_SCENE_CACHE["left_wp"] = left_wp

    if right_wp is not None:
        _LAST_SCENE_CACHE["right_wp"] = right_wp

    return scene
