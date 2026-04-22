# main.py
import os
import time
import math
import sys
import json
import argparse
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
# 0. CLI 参数
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser("Lane Change FSM Experiment Runner")
    parser.add_argument("--scenario", type=str, default=SCENARIO_MODE,
                        choices=[
                            "Overtake_Left", "Blocked_Wait", "Random",   # 兼容旧模式
                            "FREE_FLOW", "FOLLOW_SLOW_LEAD", "LEFT_BLOCKED", "DENSE_TRAFFIC"  # 新模板
                        ])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_traffic", type=int, default=NUM_TRAFFIC)
    parser.add_argument("--max_ticks", type=int, default=MAX_TICKS)
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--record_video", type=int, default=1 if RECORD_VIDEO else 0)
    parser.add_argument("--show_top_view", type=int, default=1)
    return parser.parse_args()


def _safe_mean(values):
    vals = [v for v in values if v is not None]
    return float(sum(vals) / len(vals)) if vals else None


def _safe_min(values):
    vals = [v for v in values if v is not None]
    return float(min(vals)) if vals else None


def _pick(row, keys):
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return None


def build_paper_metrics(logger_rows, collision_count, invasion_count, dt=0.05):
    """
    从 DataLogger rows 中尽可能提取论文常用指标（字段名做兼容）。
    """
    if not logger_rows:
        return {
            "total_ticks": 0,
            "collisions": collision_count,
            "lane_invasions": invasion_count
        }

    speeds = [_pick(r, ["speed", "ego_speed_mps"]) for r in logger_rows]
    ttcs = [_pick(r, ["ttc", "ttc_front", "ttc_front_s"]) for r in logger_rows]
    accs = [_pick(r, ["acc", "accel", "ego_accel_mps2"]) for r in logger_rows]

    # jerk：若无直接字段则差分加速度
    jerks = []
    direct_jerk = [_pick(r, ["jerk", "ego_jerk_mps3"]) for r in logger_rows]
    if any(v is not None for v in direct_jerk):
        jerks = [v for v in direct_jerk if v is not None]
    else:
        last_a = None
        for a in accs:
            if a is None:
                continue
            if last_a is not None:
                jerks.append((a - last_a) / max(dt, 1e-3))
            last_a = a

    # 状态序列（兼容 enum/string）
    states = []
    for r in logger_rows:
        s = _pick(r, ["state", "fsm_state"])
        if hasattr(s, "name"):
            s = s.name
        elif s is not None:
            s = str(s)
        states.append(s)

    state_switch_count = 0
    lane_change_enter_count = 0
    last = None
    for s in states:
        if last is not None and s != last:
            state_switch_count += 1
        # 统计“进入换道状态”的次数
        if s in ("LANE_CHANGE_LEFT", "LANE_CHANGE_RIGHT") and last != s:
            lane_change_enter_count += 1
        last = s

    # 状态占比
    state_hist = {}
    for s in states:
        key = s if s is not None else "UNKNOWN"
        state_hist[key] = state_hist.get(key, 0) + 1
    total = len(states)
    state_ratio = {k: v / total for k, v in state_hist.items()}

    # 百分位
    jerk_p95 = None
    if jerks:
        jerk_p95 = float(np.percentile(jerks, 95))

    metrics = {
        "total_ticks": len(logger_rows),
        "sim_time_s": len(logger_rows) * dt,

        "avg_speed_mps": _safe_mean(speeds),
        "avg_speed_kmh": (_safe_mean(speeds) * 3.6) if _safe_mean(speeds) is not None else None,
        "min_ttc_s": _safe_min(ttcs),
        "risk_events_ttc_lt_2s": int(sum(1 for x in ttcs if x is not None and x < 2.0)),
        "risk_events_ttc_lt_1p5s": int(sum(1 for x in ttcs if x is not None and x < 1.5)),

        "jerk_p95_mps3": jerk_p95,
        "state_switch_count": state_switch_count,
        "lane_change_enter_count": lane_change_enter_count,
        "state_ratio": state_ratio,

        "collisions": int(collision_count),
        "lane_invasions": int(invasion_count),
    }
    return metrics


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
        print("\n" + "=" * 60)
        print("❌ 致命错误：无法连接到 CARLA 服务端！")
        print("👉 请检查：是否已经手动启动了 CarlaUE4.exe？")
        print("=" * 60 + "\n")
        return False


def setup_world(client):
    """保持稳健地图加载"""
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
# 2. 视频写入线程
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
# 3. 传感器挂载
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
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(VIDEO_WIDTH))
    bp.set_attribute("image_size_y", str(VIDEO_HEIGHT))
    bp.set_attribute("fov", "90")
    transform = carla.Transform(carla.Location(x=-8.0, z=3.5), carla.Rotation(pitch=-15.0))
    camera = world.spawn_actor(bp, transform, attach_to=vehicle)
    camera.listen(image_queue.put)
    return camera


def attach_top_down_camera(world, vehicle, image_queue):
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", "640")
    bp.set_attribute("image_size_y", "480")
    bp.set_attribute("fov", "90")
    transform = carla.Transform(carla.Location(x=0, z=35.0), carla.Rotation(pitch=-90.0))
    camera = world.spawn_actor(bp, transform, attach_to=vehicle)
    camera.listen(image_queue.put)
    return camera


# ==========================================
# 4. 视角同步
# ==========================================
def _clamp_angle_deg(angle):
    while angle > 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
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
    args = parse_args()
    run_ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir if args.out_dir else os.path.join("outputs", f"{args.scenario}_seed{args.seed}_{run_ts}")
    os.makedirs(out_dir, exist_ok=True)

    client = carla.Client(HOST, PORT)
    if not check_connection_and_rescue(client):
        sys.exit(1)

    actor_list = []
    sensor_list = []
    collision_counter = {"count": 0}
    invasion_counter = {"count": 0}
    camera_cache = {"transform": None}

    video_writer = None
    image_queue = None
    writer_thread = None
    stop_writer_event = threading.Event()

    top_down_queue = queue.Queue()
    record_video = bool(args.record_video)

    if record_video:
        if not os.path.exists(VIDEO_OUTPUT_DIR):
            os.makedirs(VIDEO_OUTPUT_DIR)
        rec_name = f"exp_{args.scenario}_seed{args.seed}_{run_ts}.mp4"
        rec_path = os.path.join(os.getcwd(), VIDEO_OUTPUT_DIR, rec_name)

        fps = int(1.0 / FIXED_DELTA_SECONDS)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(rec_path, fourcc, fps, (VIDEO_WIDTH, VIDEO_HEIGHT))
        image_queue = queue.Queue()

        writer_thread = threading.Thread(
            target=video_writer_thread,
            args=(image_queue, video_writer, stop_writer_event),
            daemon=True
        )
        writer_thread.start()

    world = None

    try:
        world = setup_world(client)
        world_map = world.get_map()

        # ==========================================
        # [新增] 冻结全地图红绿灯为绿灯，确保高速畅通
        # ==========================================
        traffic_lights = world.get_actors().filter('*traffic_light*')
        for tl in traffic_lights:
            tl.set_state(carla.TrafficLightState.Green)
            tl.freeze(True)
        print(f"🚦 已将全地图 {len(traffic_lights)} 个红绿灯锁定为绿灯。")
        # ==========================================

        print(f"Spawning Scenario: {args.scenario} (seed={args.seed}, num_traffic={args.num_traffic})")
        spawn_ret = spawn_scenario(
            world=world,
            client=client,
            scenario_mode=args.scenario,
            seed=args.seed,
            num_traffic=args.num_traffic
        )

        # 兼容3返回值/4返回值
        if len(spawn_ret) == 4:
            ego, traffic_actors, tm, run_cfg = spawn_ret
        else:
            ego, traffic_actors, tm = spawn_ret
            run_cfg = {
                "scenario_mode": args.scenario,
                "seed": args.seed,
                "num_traffic": args.num_traffic
            }

        actor_list.append(ego)
        actor_list.extend(traffic_actors)

        sensor_list.append(attach_collision_sensor(world, ego, collision_counter))
        sensor_list.append(attach_lane_invasion_sensor(world, ego, invasion_counter))
        sensor_list.append(attach_top_down_camera(world, ego, top_down_queue))
        if record_video:
            sensor_list.append(attach_rgb_camera(world, ego, image_queue))

        # 保存运行配置
        run_config = {
            "timestamp": run_ts,
            "map": MAP_NAME,
            "sync_mode": SYNC_MODE,
            "fixed_delta_seconds": FIXED_DELTA_SECONDS,
            "max_ticks": args.max_ticks,
            "scenario": args.scenario,
            "seed": args.seed,
            "num_traffic": args.num_traffic,
            "spawn_config": run_cfg
        }
        with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as f:
            json.dump(run_config, f, ensure_ascii=False, indent=2)

        for _ in range(10):
            world.tick()

        decider = LaneChangeDecider(world, world_map, ego)
        controller = BasicLaneController(world_map, ego)
        logger = DataLogger()

        print("Simulation started. Press Ctrl+C to stop.")

        for tick in range(args.max_ticks):
            world.tick()

            if args.show_top_view:
                try:
                    top_image = top_down_queue.get_nowait()
                    top_array = np.frombuffer(top_image.raw_data, dtype=np.dtype("uint8"))
                    top_array = np.reshape(top_array, (top_image.height, top_image.width, 4))
                    cv2.imshow("Real-time Top-Down View", top_array[:, :, :3])
                    cv2.waitKey(1)
                except queue.Empty:
                    pass

            state, scene, extra = decider.update()
            control, ctrl_debug = controller.run_step(scene, state)
            ego.apply_control(control)

            follow_ego_view(world, ego, camera_cache)

            all_debug = {}
            all_debug.update(extra if extra is not None else {})
            all_debug.update(ctrl_debug if ctrl_debug is not None else {})
            logger.log(tick, ego, state, scene, all_debug)

            state_name = state.name if hasattr(state, "name") else str(state)
            if tick % 20 == 0:
                spd = logger.rows[-1].get("speed", 0.0) * 3.6
                front_dist = logger.rows[-1].get("curr_front_dist", 999.0)
                throttle_cmd = logger.rows[-1].get("throttle", 0.0)
                brake_cmd = logger.rows[-1].get("brake", 0.0)
                idm_accel = logger.rows[-1].get("idm_accel", 0.0)
                print(f"[{tick}] state={state_name}, speed={spd:.2f} km/h, front_dist={front_dist:.1f}m, "
                      f"throttle={throttle_cmd:.3f}, brake={brake_cmd:.3f}, accel={idm_accel:.3f}m/s²")
            
            if collision_counter["count"] > 0:
                print("Collision detected! Terminating...")
                break

        # 保存日志
        log_csv_path = os.path.join(out_dir, os.path.basename(LOG_CSV) if LOG_CSV else "tick_log.csv")
        summary_path = os.path.join(out_dir, os.path.basename(SUMMARY_JSON) if SUMMARY_JSON else "summary.json")
        paper_metrics_path = os.path.join(out_dir, "paper_metrics.json")

        logger.save_csv(log_csv_path)

        summary = summarize_logger(logger, collision_counter["count"], invasion_counter["count"])
        summary["scenario"] = args.scenario
        summary["seed"] = args.seed
        summary["num_traffic"] = args.num_traffic
        save_summary(summary, summary_path)

        paper_metrics = build_paper_metrics(
            logger_rows=logger.rows,
            collision_count=collision_counter["count"],
            invasion_count=invasion_counter["count"],
            dt=FIXED_DELTA_SECONDS
        )
        with open(paper_metrics_path, "w", encoding="utf-8") as f:
            json.dump(paper_metrics, f, ensure_ascii=False, indent=2)

        print(f"Simulation finished. Outputs saved to: {out_dir}")

    except KeyboardInterrupt:
        print("\nManually stopped.")

    finally:
        print("Cleaning up resources...")
        cv2.destroyAllWindows()

        if record_video:
            print("Waiting for video writer...")
            stop_writer_event.set()
            if writer_thread is not None and writer_thread.is_alive():
                writer_thread.join(timeout=2.0)
            if video_writer is not None:
                video_writer.release()

        # 先退出同步模式再销毁 Actor
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
            try:
                if s is not None and s.is_alive:
                    s.destroy()
            except Exception:
                pass

        for a in actor_list:
            try:
                if a is not None and a.is_alive:
                    a.destroy()
            except Exception:
                pass

        print("Cleanup complete.")


if __name__ == "__main__":
    main()
