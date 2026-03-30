# scripts/spawn_traffic.py
import random
import carla
from config import (
    EGO_ROLE_NAME,
    NUM_TRAFFIC,
    TRAFFIC_MANAGER_PORT,
    TRAFFIC_GLOBAL_DISTANCE,
    TRAFFIC_SPEED_DIFF_PERCENT,
)

def get_blueprint(world, ego=False):
    bp_lib = world.get_blueprint_library()

    if ego:
        candidates = bp_lib.filter("vehicle.tesla.model3")
        if len(candidates) == 0:
            candidates = bp_lib.filter("vehicle.*")
        bp = random.choice(candidates)

        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", EGO_ROLE_NAME)

        if bp.has_attribute("color"):
            bp.set_attribute("color", "255,0,0")   # 红色自车

        return bp

    else:
        candidates = bp_lib.filter("vehicle.*")
        bp = random.choice(candidates)
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", "autopilot")
        return bp


def try_spawn_vehicle(world, transform, ego=False):
    bp = get_blueprint(world, ego)
    return world.try_spawn_actor(bp, transform)

def spawn_ego(world):
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    for sp in spawn_points:
        veh = try_spawn_vehicle(world, sp, ego=True)
        if veh is not None:
            return veh
    raise RuntimeError("Failed to spawn ego vehicle.")

def spawn_traffic(world, client, ego, num_traffic):
    tm = client.get_trafficmanager(TRAFFIC_MANAGER_PORT)
    tm.set_global_distance_to_leading_vehicle(TRAFFIC_GLOBAL_DISTANCE)
    tm.global_percentage_speed_difference(TRAFFIC_SPEED_DIFF_PERCENT)

    actors = []
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    used = 0
    for sp in spawn_points:
        if used >= num_traffic:
            break
        if ego.get_location().distance(sp.location) < 20.0:
            continue
        veh = try_spawn_vehicle(world, sp, ego=False)
        if veh is not None:
            veh.set_autopilot(True, TRAFFIC_MANAGER_PORT)
            actors.append(veh)
            used += 1

    return actors, tm
