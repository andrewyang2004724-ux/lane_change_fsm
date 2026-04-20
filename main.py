# main.py
import os
import time
import math
import sys
import carla
import queue
import threading
import cv2
import numpy as np

from config import (
    HOST, PORT, TIMEOUT,
    SYNC_MODE, FIXED_DELTA_SECONDS,
    MAP_NAME, NUM_TRAFFIC, SCENARIO_MODE,
    MAX_TICKS, LOG_CSV, SUMMARY_JSON,
    RECORD_VIDEO, VIDEO_OUTPUT_DIR,
    VIDEO_WIDTH, VIDEO_HEIGHT
)

from scripts.spawn_traffic import spawn_scenario
from decision.fsm_decider import LaneChangeDecider
from control.basic_controller import BasicLaneController
from evaluation.logger import DataLogger
from evaluation.metrics import summarize_logger, save_summary

# ==========================================
# 1. 核心修复：CARLA 崩溃急救与连接探测
# ==========================================
def check_connection_and_rescue(client):
    """在启动前探测服务端状态，如果处于僵死同步模式则强行唤醒"""
    client.set_timeout(3.0) 
    try:
        world = client.get_world()
        settings = world.get_settings()
        if settings.synchronous_mode:
            print("Recovering CARLA server from previous crash (disabling sync mode)...")
            settings.synchronous_mode = False
            world.apply_settings(settings)
        client.set_timeout(TIMEOUT)
        return True
    except RuntimeError:
        print("\n" + "="*60)
        print("❌ 致命错误：无法连接到 CARLA 服务端！")
        print("👉 请检查：是否已经手动启动了 CarlaUE4.exe？")
        print("="*60 + "\n")
        return False

def setup_world(client):
    """按照用户要求：回退版本，不再执行图层卸载，保持最稳健的地图加载"""
    current_map = client.get_world().get_map().name.split("/")[-1]
    if current_map != MAP_NAME:
        print(f"Loading map: {MAP_NAME} ...")
        world = client.load_world(MAP_NAME)
        time.sleep(2.0) 
    else:
        world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = SYNC_MODE
    settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
    world.apply_settings(settings)
    return world

# ==========================================
# 2. 核心修复：后台视频写入线程 (防止 I/O 阻塞导致掉帧)
# ==========================================
def video_writer_thread(image_queue, writer, stop_event):
    while not stop_event.is_set() or not image_queue.empty():
        try:
            image = image_queue.get(timeout=0.1)
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]  
            writer.write(array)
        except queue.Empty:
            continue

# ==========================================
# 3. 传感器挂载辅助函数
# ==========================================
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
    """录制视角：挂在车后上方"""
    bp = world.get_blueprint_library().find('sensor.camera.rgb')
    bp.set_attribute('image_size_x', str(VIDEO_WIDTH))
    bp.set_attribute('image_size_y', str(VIDEO_HEIGHT))
    bp.set_attribute('fov', '90')
    transform = carla.Transform(carla.Location(x=-8.0, z=3.5), carla.Rotation(pitch=-15.0))
    camera = world.spawn_actor(bp, transform, attach_to=vehicle)
    camera.listen(image_queue.put)
    return camera

def attach_top_down_camera(world, vehicle, image_queue):
    """俯视视角：挂在车正上方 35 米处垂直向下看"""
    bp = world.get_blueprint_library().find('sensor.camera.rgb')
    bp.set_attribute('image_size_x', '640') # 小地图窗口分辨率
    bp.set_attribute('image_size_y', '480')
    bp.set_attribute('fov', '90')
    transform = carla.Transform(carla.Location(x=0, z=35.0), carla.Rotation(pitch=-90.0))
    camera = world.spawn_actor(bp, transform, attach_to=vehicle)
    camera.listen(image_queue.put)
    return camera

# ==========================================
# 4. 视角同步辅助函数
# ==========================================
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

# ==========================================
# 5. 主程序入口
# ==========================================
def main():
    client = carla.Client(HOST, PORT)
    if not check_connection_and_rescue(client):
        sys.exit(1)

    actor_list = []
    sensor_list = []
    collision_counter = {"count": 0}
    invasion_counter = {"count": 0}
    camera_cache = {"transform": None}

    # 视频录制变量初始化
    video_writer = None
    image_queue = None
    writer_thread = None
    stop_writer_event = threading.Event()

    # 俯视视角变量初始化 (修复 Bug 的核心点)
    top_down_queue = queue.Queue()

    if RECORD_VIDEO:
        if not os.path.exists(VIDEO_OUTPUT_DIR):
            os.makedirs(VIDEO_OUTPUT_DIR)
        rec_name = f"exp_{SCENARIO_MODE}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        rec_path = os.path.join(os.getcwd(), VIDEO_OUTPUT_DIR, rec_name)
        
        fps = int(1.0 / FIXED_DELTA_SECONDS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(rec_path, fourcc, fps, (VIDEO_WIDTH, VIDEO_HEIGHT))
        image_queue = queue.Queue()
        
        # 启动后台写入线程，daemon=True 防止主程序退出后残留
        writer_thread = threading.Thread(target=video_writer_thread, args=(image_queue, video_writer, stop_writer_event), daemon=True)
        writer_thread.start()

    world = None

    try:
        world = setup_world(client)
        world_map = world.get_map()

        # 生成选定实验场景
        print(f"Spawning Scenario: {SCENARIO_MODE}")
        ego, traffic_actors, tm = spawn_scenario(world, client)
        
        actor_list.append(ego)
        actor_list.extend(traffic_actors)

        # 挂载传感器
        sensor_list.append(attach_collision_sensor(world, ego, collision_counter))
        sensor_list.append(attach_lane_invasion_sensor(world, ego, invasion_counter))
        
        # 同时挂载两个相机
        sensor_list.append(attach_top_down_camera(world, ego, top_down_queue)) # 用于 OpenCV 弹窗
        if RECORD_VIDEO:
            sensor_list.append(attach_rgb_camera(world, ego, image_queue))     # 用于保存视频

        # 预热 Tick
        for _ in range(10):
            world.tick()

        decider = LaneChangeDecider(world, world_map, ego)
        controller = BasicLaneController(world_map, ego)
        logger = DataLogger()

        print("Simulation started. Press Ctrl+C to stop.")

        for tick in range(MAX_TICKS):
            world.tick()

            # --- 实时渲染俯视窗口 (不阻塞) ---
            try:
                top_image = top_down_queue.get_nowait()
                top_array = np.frombuffer(top_image.raw_data, dtype=np.dtype("uint8"))
                top_array = np.reshape(top_array, (top_image.height, top_image.width, 4))
                cv2.imshow("Real-time Top-Down View", top_array[:, :, :3])
                cv2.waitKey(1)
            except queue.Empty:
                pass

            # --- 逻辑更新 ---
            state, scene, extra = decider.update()
            control, ctrl_debug = controller.run_step(scene, state)
            ego.apply_control(control)

            follow_ego_view(world, ego, camera_cache)

            # --- 数据记录 ---
            all_debug = {}
            all_debug.update(extra)
            all_debug.update(ctrl_debug)
            logger.log(tick, ego, state, scene, all_debug)

            if tick % 20 == 0:
                print(f"[{tick}] state={state.name}, speed={logger.rows[-1]['speed'] * 3.6:.2f} km/h")

            if collision_counter["count"] > 0:
                print("Collision detected! Terminating...")
                break

        # 仿真结束保存数据
        logger.save_csv(LOG_CSV)
        summary = summarize_logger(logger, collision_counter["count"], invasion_counter["count"])
        save_summary(summary, SUMMARY_JSON)
        print("Simulation finished.")

    except KeyboardInterrupt:
        print("\nManually stopped.")

    finally:
        print("Cleaning up resources...")
        cv2.destroyAllWindows()
        
        if RECORD_VIDEO:
            print("Waiting for video writer...")
            stop_writer_event.set()
            if writer_thread is not None and writer_thread.is_alive():
                writer_thread.join(timeout=2.0)
            if video_writer is not None:
                video_writer.release()

        # 核心修复：必须先退出同步模式，再销毁 Actor，防止服务端闪退
        if world is not None:
            try:
                client.set_timeout(2.0)
                settings = world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                world.apply_settings(settings)
                time.sleep(0.5) 
            except Exception:
                pass

        for s in sensor_list:
            if s.is_alive: s.destroy()
        for a in actor_list:
            if a.is_alive: a.destroy()
        
        print("Cleanup complete.")

if __name__ == "__main__":
    main()