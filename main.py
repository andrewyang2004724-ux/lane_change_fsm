# main.py
import time
import math
import carla

from config import (
    HOST, PORT, TIMEOUT,
    SYNC_MODE, FIXED_DELTA_SECONDS,
    MAP_NAME, NUM_TRAFFIC,
    MAX_TICKS, LOG_CSV, SUMMARY_JSON
)

from scripts.spawn_traffic import spawn_ego, spawn_traffic
from decision.fsm_decider import LaneChangeDecider
from control.basic_controller import BasicLaneController
from evaluation.logger import DataLogger
from evaluation.metrics import summarize_logger, save_summary

def setup_world(client):
    world = client.get_world()

    current_map_name = world.get_map().name.split("/")[-1]
    if current_map_name != MAP_NAME:
        print(f"Loading map: {MAP_NAME} ...")
        world = client.load_world(MAP_NAME)
        time.sleep(2.0)

    settings = world.get_settings()
    settings.synchronous_mode = SYNC_MODE
    settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
    world.apply_settings(settings)

    return world

def attach_collision_sensor(world, vehicle, collision_counter):
    bp = world.get_blueprint_library().find("sensor.other.collision")
    sensor = world.spawn_actor(bp, carla.Transform(), attach_to=vehicle)

    def callback(event):
        collision_counter["count"] += 1

    sensor.listen(callback)
    return sensor

def attach_lane_invasion_sensor(world, vehicle, invasion_counter):
    bp = world.get_blueprint_library().find("sensor.other.lane_invasion")
    sensor = world.spawn_actor(bp, carla.Transform(), attach_to=vehicle)

    def callback(event):
        invasion_counter["count"] += 1

    sensor.listen(callback)
    return sensor

def print_ego_info(ego):
    tf = ego.get_transform()
    print(
        f"Ego spawned: id={ego.id}, "
        f"location=({tf.location.x:.2f}, {tf.location.y:.2f}, {tf.location.z:.2f}), "
        f"yaw={tf.rotation.yaw:.2f}"
    )

def _clamp_angle_deg(angle):
    while angle > 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle

def follow_ego_view(world, ego, cache):
    """
    更稳定的 chase camera:
    - 相机放在车后上方
    - 相机始终看向车辆前上方一点
    - 不直接生硬复制车辆 yaw，降低大幅扭动
    """
    spectator = world.get_spectator()

    ego_tf = ego.get_transform()
    ego_loc = ego_tf.location
    forward = ego_tf.get_forward_vector()
    right = ego_tf.get_right_vector()

    # 常见 chase camera 偏移：后方 + 上方
    target_cam_loc = carla.Location(
        x=ego_loc.x - 8.0 * forward.x + 0.0 * right.x,
        y=ego_loc.y - 8.0 * forward.y + 0.0 * right.y,
        z=ego_loc.z + 3.5
    )

    # 看向车前上方一点
    look_at = carla.Location(
        x=ego_loc.x + 12.0 * forward.x,
        y=ego_loc.y + 12.0 * forward.y,
        z=ego_loc.z + 1.5
    )

    dx = look_at.x - target_cam_loc.x
    dy = look_at.y - target_cam_loc.y
    dz = look_at.z - target_cam_loc.z

    target_yaw = math.degrees(math.atan2(dy, dx))
    horiz_dist = math.sqrt(dx * dx + dy * dy)
    target_pitch = -math.degrees(math.atan2(dz, horiz_dist))

    alpha_pos = 0.20
    alpha_rot = 0.08

    if cache["transform"] is None:
        new_loc = target_cam_loc
        new_rot = carla.Rotation(pitch=target_pitch, yaw=target_yaw, roll=0.0)
    else:
        old_tf = cache["transform"]
        old_loc = old_tf.location
        old_rot = old_tf.rotation

        new_loc = carla.Location(
            x=(1 - alpha_pos) * old_loc.x + alpha_pos * target_cam_loc.x,
            y=(1 - alpha_pos) * old_loc.y + alpha_pos * target_cam_loc.y,
            z=max(1.5, (1 - alpha_pos) * old_loc.z + alpha_pos * target_cam_loc.z)
        )

        yaw_err = _clamp_angle_deg(target_yaw - old_rot.yaw)
        new_yaw = old_rot.yaw + alpha_rot * yaw_err
        new_pitch = (1 - alpha_rot) * old_rot.pitch + alpha_rot * target_pitch

        new_rot = carla.Rotation(
            pitch=new_pitch,
            yaw=new_yaw,
            roll=0.0
        )

    new_tf = carla.Transform(new_loc, new_rot)
    spectator.set_transform(new_tf)
    cache["transform"] = new_tf

def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(TIMEOUT)

    actor_list = []
    sensor_list = []

    collision_counter = {"count": 0}
    invasion_counter = {"count": 0}
    camera_cache = {"transform": None}

    world = None

    try:
        world = setup_world(client)
        world_map = world.get_map()

        ego = spawn_ego(world)
        actor_list.append(ego)
        print_ego_info(ego)

        traffic_actors, tm = spawn_traffic(world, client, ego, NUM_TRAFFIC)
        actor_list.extend(traffic_actors)
        print(f"Spawned traffic vehicles: {len(traffic_actors)}")

        collision_sensor = attach_collision_sensor(world, ego, collision_counter)
        lane_sensor = attach_lane_invasion_sensor(world, ego, invasion_counter)
        sensor_list.extend([collision_sensor, lane_sensor])

        for _ in range(20):
            world.tick()
            follow_ego_view(world, ego, camera_cache)

        decider = LaneChangeDecider(world, world_map, ego)
        controller = BasicLaneController(world_map, ego)
        logger = DataLogger()

        print("Simulation started.")

        for tick in range(MAX_TICKS):
            world.tick()

            state, scene, extra = decider.update()
            control, ctrl_debug = controller.run_step(scene, state)
            ego.apply_control(control)

            follow_ego_view(world, ego, camera_cache)

            all_debug = {}
            all_debug.update(extra)
            all_debug.update(ctrl_debug)
            logger.log(tick, ego, state, scene, all_debug)

            if tick % 20 == 0:
                lane_id = scene["ego_wp"].lane_id if scene["ego_wp"] is not None else None
                road_id = scene["ego_wp"].road_id if scene["ego_wp"] is not None else None

                print(
                    f"[{tick}] "
                    f"state={state.name}, "
                    f"speed={logger.rows[-1]['speed'] * 3.6:.2f} km/h, "
                    f"road={road_id}, lane={lane_id}, "
                    f"front={scene['curr_front']['dist']:.2f}m/"
                    f"{scene['curr_front']['speed'] * 3.6:.2f}km/h, "
                    f"safe_dist={ctrl_debug.get('safe_dist', -1):.2f}m, "
                    f"desired_speed={ctrl_debug.get('desired_speed', -1) * 3.6:.2f}km/h, "
                    f"mode={ctrl_debug.get('mode', 'na')}, "
                    f"throttle={ctrl_debug.get('throttle', 0):.2f}, "
                    f"brake={ctrl_debug.get('brake', 0):.2f}, "
                    f"left_front={scene['left_front']['dist']:.2f}m, "
                    f"left_rear={scene['left_rear']['dist']:.2f}m, "
                    f"right_front={scene['right_front']['dist']:.2f}m, "
                    f"right_rear={scene['right_rear']['dist']:.2f}m, "
                    f"left_allowed={scene['left_allowed']}, "
                    f"right_allowed={scene['right_allowed']}"
                )

            if collision_counter["count"] > 0:
                print("Collision detected. Stopping simulation early for debugging.")
                break

        logger.save_csv(LOG_CSV)
        summary = summarize_logger(
            logger,
            collision_count=collision_counter["count"],
            lane_invasion_count=invasion_counter["count"]
        )
        save_summary(summary, SUMMARY_JSON)

        print("Simulation finished.")
        print("Summary:", summary)

    finally:
        print("Cleaning actors...")

        for s in sensor_list:
            try:
                if s.is_alive:
                    s.stop()
                    s.destroy()
            except Exception:
                pass

        for a in actor_list:
            try:
                if a.is_alive:
                    a.destroy()
            except Exception:
                pass

        if world is not None:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)

if __name__ == "__main__":
    main()
