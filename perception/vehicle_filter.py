# perception/vehicle_filter.py
from decision.safety import speed_norm
from perception.lane_utils import get_current_waypoint, waypoint_same_lane

def dot(v1, v2):
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

def longitudinal_distance(ego, other):
    """
    保留原接口：按 ego 当前朝向投影计算纵向距离
    适合简单场景或兼容旧逻辑。
    """
    ego_tf = ego.get_transform()
    ego_loc = ego_tf.location
    ego_fwd = ego_tf.get_forward_vector()
    rel = other.get_location() - ego_loc
    return dot(rel, ego_fwd)

def longitudinal_distance_along_wp(ego, other, ref_wp):
    """
    更稳健版本：沿参考 waypoint 的前向方向计算纵向距离。
    在弯道、变道中比直接使用 ego 朝向更稳定。
    """
    ego_loc = ego.get_location()
    rel = other.get_location() - ego_loc

    if ref_wp is not None:
        ref_fwd = ref_wp.transform.get_forward_vector()
    else:
        ref_fwd = ego.get_transform().get_forward_vector()

    return dot(rel, ref_fwd)

def vehicle_speed(vehicle):
    return speed_norm(vehicle.get_velocity())

def get_lane_vehicles(world, world_map, ego_vehicle, target_wp, radius=100.0):
    """
    获取目标车道内车辆，并返回相对 ego 的纵向投影距离。
    这里使用 target_wp 的前向作为参考，提升前后车判定稳定性。
    """
    if target_wp is None:
        return []

    vehicles = world.get_actors().filter("vehicle.*")
    ego_loc = ego_vehicle.get_location()
    result = []

    for veh in vehicles:
        if veh.id == ego_vehicle.id:
            continue

        if veh.get_location().distance(ego_loc) > radius:
            continue

        wp = get_current_waypoint(world_map, veh)
        if wp is None:
            continue

        if waypoint_same_lane(wp, target_wp):
            s = longitudinal_distance_along_wp(ego_vehicle, veh, target_wp)
            result.append((veh, s))

    return result

def split_front_rear(lane_vehicles):
    """
    从目标车道车辆中找最近前车和最近后车
    lane_vehicles: [(veh, s), ...]
    s >= 0 为前方，s < 0 为后方
    """
    front = None
    rear = None
    min_front = float("inf")
    max_rear = -float("inf")

    for veh, s in lane_vehicles:
        if s >= 0.0:
            if s < min_front:
                min_front = s
                front = veh
        else:
            if s > max_rear:
                max_rear = s
                rear = veh

    return front, rear
