# main.py
import os
import time
import math
import carla
import queue
import threading
import cv2
import numpy as np

from config import (
    HOST, PORT, TIMEOUT,
    SYNC_MODE, FIXED_DELTA_SECONDS,
    MAP_NAME, NUM_TRAFFIC,
    MAX_TICKS, LOG_CSV, SUMMARY_JSON,
    UNLOAD_MAP_LAYERS,
    RECORD_VIDEO, VIDEO_OUTPUT_DIR,
    VIDEO_WIDTH, VIDEO_HEIGHT
)

from scripts.spawn_traffic import spawn_ego, spawn_traffic
from decision.fsm_decider import LaneChangeDecider
from control.basic_controller import BasicLaneController
from evaluation.logger import DataLogger
from evaluation.metrics import summarize_logger, save_summary

# ==========================================
# 核心修复 1：CARLA 崩溃急救机制
# 每次启动前强制解除服务器的同步锁定，防止 Timeout
# ==========================================
def rescue_carla_server(client):
    try:
        world = client.get_world()
        settings = world.get_settings()
        if settings.synchronous_mode:
            print("Recovering CARLA server from previous crash (disabling sync mode)...")
            settings.synchronous_mode = False
            world.apply_settings(settings)
    except Exception as e:
        print("Rescue check passed or no connection.")

def setup_world(client):
    current_map = client.get_world().get_map().name.split("/")[-1]
    
    if current_map != MAP_NAME:
        print(f"Loading map: {MAP_NAME} ...")
        world = client.load_world(MAP_NAME)
        time.sleep(2.0) # 给引擎加载物理的时间
    else:
        world = client.get_world()

    # 卸载冗余图层
    if MAP_NAME.endswith("_Opt") and UNLOAD_MAP_LAYERS:
        print("Force unloading map layers...")
        world.unload_map_layer(carla.MapLayer.All)
        time.sleep(1.0) # 【核心修复 2】：给虚幻引擎 1 秒钟来清理数以万计的建筑 Mesh，防止死锁

    settings = world.get_settings()
    settings.synchronous_mode = SYNC_MODE
    settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
    world.apply_settings(settings)
    
    print("Flushing rendering pipeline...")
    for _ in range(10):
        world.tick()
        
    return world

# ==========================================
# 核心修复 3：绝对隔离的视频后台线程
# ==========================================
def video_writer_thread(image_queue, writer, stop_event):
    """纯后台运行，主线程绝对不碰 image_queue，彻底消除死锁"""
    while not stop_event.is_set() or not image_queue.empty():
        try:
            image = image_queue.get(timeout=0.1)
            # CARLA 数据转 OpenCV BGR 格式
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]  
            writer.write(array)
        except queue.Empty:
            continue

def attach_collision_sensor(world, vehicle, collision_counter):
    bp = world.get_blueprint_library().find("sensor.other.collision")
    sensor = world.spawn_actor(bp, carla.Transform(), attach_to=vehicle)
    sensor.listen(lambda event: collision_counter.update({"count": collision_counter["count"] + 1}))
    return sensor

def attach_lane_invasion_sensor(world, vehicle, invasion_counter):
    bp = world.get_blueprint_library().find("sensor.other.lane_invasion")
    sensor = world.spawn_actor(bp, carla.Transform(), attach_to=vehicle)
    sensor.listen(lambda event: invasion_counter.update({"count": invasion_counter["count"] + 1}))
    return sensor

def attach_rgb_camera(world, vehicle, image_queue):
    bp = world.get_blueprint_library().find('sensor.camera.rgb')
    bp.set_attribute('image_size_x', str(VIDEO_WIDTH))
    bp.set_attribute('image_size_y', str(VIDEO_HEIGHT))
    bp.set_attribute('fov', '90')
    
    transform = carla.Transform(carla.Location(x=-8.0, z=3.5), carla.Rotation(pitch=-15.0))
    camera = world.spawn_actor(bp, transform, attach_to=vehicle)
    camera.listen(image_queue.put)
    return camera

def _clamp_angle_deg(angle):
    while angle > 180.0: angle -= 360.0
    while angle < -180.0: angle += 360.0
    return angle

def follow_ego_view(world, ego, cache):
    spectator = world.get_spectator()
    ego_tf = ego.get_transform()
    ego_loc = ego_tf.location
    forward = ego_tf.get_forward_vector()

    target_cam_loc = carla.Location(
        x=ego_loc.x - 8.0 * forward.x,
        y=ego_loc.y - 8.0 * forward.y,
        z=ego_loc.z + 3.5
    )

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

    if cache["transform"] is None:
        new_loc = target_cam_loc
        new_rot = carla.Rotation(pitch=target_pitch, yaw=target_yaw, roll=0.0)
    else:
        old_tf = cache["transform"]
        alpha_pos, alpha_rot = 0.20, 0.08
        new_loc = carla.Location(
            x=(1 - alpha_pos) * old_tf.location.x + alpha_pos * target_cam_loc.x,
            y=(1 - alpha_pos) * old_tf.location.y + alpha_pos * target_cam_loc.y,
            z=max(1.5, (1 - alpha_pos) * old_tf.location.z + alpha_pos * target_cam_loc.z)
        )
        new_yaw = old_tf.rotation.yaw + alpha_rot * _clamp_angle_deg(target_yaw - old_tf.rotation.yaw)
        new_pitch = (1 - alpha_rot) * old_tf.rotation.pitch + alpha_rot * target_pitch
        new_rot = carla.Rotation(pitch=new_pitch, yaw=new_yaw, roll=0.0)

    new_tf = carla.Transform(new_loc, new_rot)
    spectator.set_transform(new_tf)
    cache["transform"] = new_tf


def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(10.0)

    # 1. 启动前先急救 CARLA 服务器，防止上一次崩溃导致的 timeout
    rescue_carla_server(client)

    actor_list = []
    sensor_list = []
    collision_counter = {"count": 0}
    invasion_counter = {"count": 0}
    camera_cache = {"transform": None}

    video_writer = None
    image_queue = None
    writer_thread = None
    stop_writer_event = threading.Event()

    if RECORD_VIDEO:
        if not os.path.exists(VIDEO_OUTPUT_DIR):
            os.makedirs(VIDEO_OUTPUT_DIR)
        rec_name = f"experiment_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        rec_path = os.path.join(os.getcwd(), VIDEO_OUTPUT_DIR, rec_name)
        
        fps = int(1.0 / FIXED_DELTA_SECONDS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(rec_path, fourcc, fps, (VIDEO_WIDTH, VIDEO_HEIGHT))
        image_queue = queue.Queue()
        
        writer_thread = threading.Thread(target=video_writer_thread, args=(image_queue, video_writer, stop_writer_event))
        writer_thread.start()
        print(f"Video recording initialized in background: {rec_path}")

    world = None

    try:
        # 2. 设置世界与地图图层
        world = setup_world(client)
        world_map = world.get_map()

        ego = spawn_ego(world)
        actor_list.append(ego)

        traffic_actors, tm = spawn_traffic(world, client, ego, NUM_TRAFFIC)
        actor_list.extend(traffic_actors)

        # 3. 挂载传感器
        sensor_list.append(attach_collision_sensor(world, ego, collision_counter))
        sensor_list.append(attach_lane_invasion_sensor(world, ego, invasion_counter))
        if RECORD_VIDEO:
            sensor_list.append(attach_rgb_camera(world, ego, image_queue))

        # 4. 预热，丢弃初始不稳定的帧
        for _ in range(20):
            world.tick()
            follow_ego_view(world, ego, camera_cache)

        decider = LaneChangeDecider(world, world_map, ego)
        controller = BasicLaneController(world_map, ego)
        logger = DataLogger()

        print("Simulation started.")

        # 5. 主循环：专注控制逻辑，绝对不碰视频 IO
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
                print(f"[{tick}] state={state.name}, speed={logger.rows[-1]['speed'] * 3.6:.2f} km/h")

            if collision_counter["count"] > 0:
                print("Collision detected. Stopping simulation early for debugging.")
                break

        # 6. 数据保存
        logger.save_csv(LOG_CSV)
        summary = summarize_logger(logger, collision_counter["count"], invasion_counter["count"])
        save_summary(summary, SUMMARY_JSON)
        print("Simulation finished.")

    finally:
        print("Cleaning up resources...")
        
        # 1. 优雅退出后台录制线程
        if RECORD_VIDEO:
            print("Waiting for video writer thread to finish...")
            stop_writer_event.set()
            if writer_thread is not None:
                writer_thread.join()
            if video_writer is not None:
                video_writer.release()
            print("Video saved successfully.")

        # ==========================================
        # 核心修复：防止 CARLA 服务端崩溃闪退
        # 绝对必须在销毁任何 Actor 之前，先恢复异步模式！
        # ==========================================
        if world is not None:
            try:
                settings = world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                world.apply_settings(settings)
                # 等待 0.5 秒让引擎彻底切换回异步状态
                time.sleep(0.5) 
            except Exception as e:
                print(f"Warning: Failed to disable sync mode: {e}")

        # 2. 极其重要：先停止传感器，再销毁
        for s in sensor_list:
            try:
                if s.is_alive:
                    s.stop()
                    s.destroy()
            except Exception:
                pass

        # 3. 最后销毁车辆
        for a in actor_list:
            try:
                if a.is_alive:
                    a.destroy()
            except Exception:
                pass
        
        print("Cleanup complete.")

if __name__ == "__main__":
    main()