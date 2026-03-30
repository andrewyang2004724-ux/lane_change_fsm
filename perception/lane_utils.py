# perception/lane_utils.py
import carla

def get_current_waypoint(world_map, vehicle):
    loc = vehicle.get_location()
    return world_map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)

def get_left_lane_if_possible(wp):
    if wp is None:
        return None
    left_wp = wp.get_left_lane()
    if left_wp is None:
        return None
    if left_wp.lane_type != carla.LaneType.Driving:
        return None
    return left_wp

def get_right_lane_if_possible(wp):
    if wp is None:
        return None
    right_wp = wp.get_right_lane()
    if right_wp is None:
        return None
    if right_wp.lane_type != carla.LaneType.Driving:
        return None
    return right_wp

def waypoint_same_lane(wp1, wp2):
    if wp1 is None or wp2 is None:
        return False
    return (
        wp1.road_id == wp2.road_id and
        wp1.section_id == wp2.section_id and
        wp1.lane_id == wp2.lane_id
    )

def lane_change_left_allowed(wp):
    if wp is None:
        return False
    return str(wp.lane_change) in ["Left", "Both", "carla.LaneChange.Left", "carla.LaneChange.Both"]

def lane_change_right_allowed(wp):
    if wp is None:
        return False
    return str(wp.lane_change) in ["Right", "Both", "carla.LaneChange.Right", "carla.LaneChange.Both"]
