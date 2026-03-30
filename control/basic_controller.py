# control/basic_controller.py
import math
import carla
from config import (
    TARGET_SPEED,
    OVERTAKE_SPEED,
    KP_SPEED,
    MAX_THROTTLE,
    MAX_BRAKE,
    KP_STEER,
    KD_STEER,
    MAX_STEER,
    MIN_GAP,
    TIME_HEADWAY,
)
from decision.state_machine import FSMState

class BasicLaneController:
    """
    改进点：
    1. 换道期间优先跟踪“目标车道中心 + 前瞻点”，减少目标跳变
    2. 增加转向平滑，抑制横向抽动
    3. 纵向控制加入油门/刹车平滑，避免频繁切换
    4. 换道阶段降低对当前车道前车的过度敏感，避免卡住不动
    """

    def __init__(self, world_map, ego):
        self.world_map = world_map
        self.ego = ego

        self.last_steer_error = 0.0
        self.last_steer_cmd = 0.0
        self.last_throttle = 0.0
        self.last_brake = 0.0

        # 换道时记住目标 waypoint，避免瞬时丢失导致目标乱跳
        self.last_target_wp = None
        self.last_target_lane_change_side = None  # "left" / "right" / None

    def _speed(self):
        v = self.ego.get_velocity()
        return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)

    def _normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def _is_left_change_state(self, state):
        return state in [
            FSMState.PREPARE_LANE_CHANGE_LEFT,
            FSMState.LANE_CHANGE_LEFT,
        ]

    def _is_right_change_state(self, state):
        return state in [
            FSMState.LANE_CHANGE_RIGHT,
            FSMState.PREPARE_RETURN_RIGHT,
        ]

    def _select_lane_change_target_wp(self, lane_wp, preview_dist):
        if lane_wp is None:
            return None
        nxt = lane_wp.next(preview_dist)
        if len(nxt) > 0:
            return nxt[0]
        return lane_wp

    def _target_waypoint(self, scene, state):
        ego_wp = scene["ego_wp"]
        if ego_wp is None:
            return None, "ego_wp_missing"

        # ---------- 左变道 ----------
        if self._is_left_change_state(state):
            target_wp = self._select_lane_change_target_wp(scene["left_wp"], 18.0)
            if target_wp is not None:
                self.last_target_wp = target_wp
                self.last_target_lane_change_side = "left"
                return target_wp, "left_changing_wp"

            # 目标车道瞬时丢失时，继续沿用上一帧目标，避免抖动
            if self.last_target_lane_change_side == "left" and self.last_target_wp is not None:
                return self.last_target_wp, "left_changing_wp_reuse"

            nxt = ego_wp.next(14.0)
            if len(nxt) > 0:
                return nxt[0], "left_changing_wp_missing_fallback"
            return ego_wp, "left_changing_wp_missing_fallback"

        # ---------- 右变道 / 右返回 ----------
        if self._is_right_change_state(state):
            target_wp = self._select_lane_change_target_wp(scene["right_wp"], 18.0)
            if target_wp is not None:
                self.last_target_wp = target_wp
                self.last_target_lane_change_side = "right"
                return target_wp, "right_changing_wp"

            # 目标车道瞬时丢失时，继续沿用上一帧目标，避免抖动
            if self.last_target_lane_change_side == "right" and self.last_target_wp is not None:
                return self.last_target_wp, "right_changing_wp_reuse"

            nxt = ego_wp.next(14.0)
            if len(nxt) > 0:
                return nxt[0], "right_changing_wp_missing"
            return ego_wp, "right_changing_wp_missing"

        # ---------- 正常跟车 ----------
        self.last_target_lane_change_side = None

        nxt = ego_wp.next(12.0)
        if len(nxt) > 0:
            self.last_target_wp = nxt[0]
            return nxt[0], "follow_lane_wp"
        self.last_target_wp = ego_wp
        return ego_wp, "follow_lane_wp"

    def _smooth_transition(self, prev, new, rise_rate=0.12, fall_rate=0.20):
        """
        对控制量进行斜率限制：
        - 上升慢一点，避免猛给
        - 下降可稍快，保证安全
        """
        if new > prev:
            return min(new, prev + rise_rate)
        else:
            return max(new, prev - fall_rate)

    def _longitudinal_control(self, scene, state):
        """
        纵向控制：巡航 + 跟车 + 紧急制动 + 换道期抑制抽动
        """
        current_speed = self._speed()

        curr_front_dist = scene["curr_front"]["dist"]
        curr_front_speed = scene["curr_front"]["speed"]

        left_front_dist = scene["left_front"]["dist"]
        left_front_speed = scene["left_front"]["speed"]

        right_front_dist = scene["right_front"]["dist"]
        right_front_speed = scene["right_front"]["speed"]

        # 默认目标速度
        if state in [FSMState.LANE_CHANGE_LEFT]:
            desired_speed = OVERTAKE_SPEED
        elif state in [FSMState.LANE_CHANGE_RIGHT, FSMState.PREPARE_RETURN_RIGHT]:
            desired_speed = min(TARGET_SPEED + 1.0, OVERTAKE_SPEED)
        elif state == FSMState.EMERGENCY_BRAKE:
            desired_speed = 0.0
        else:
            desired_speed = TARGET_SPEED

        safe_dist = MIN_GAP + TIME_HEADWAY * current_speed

        # 选择“当前主要约束前车”
        front_dist = curr_front_dist
        front_speed = curr_front_speed
        front_source = "curr"

        # 左变道时，优先关注左前车；否则当前车道前车会让车在换道时抽动
        if state in [FSMState.PREPARE_LANE_CHANGE_LEFT, FSMState.LANE_CHANGE_LEFT]:
            if left_front_dist < 999.0:
                blended = min(curr_front_dist + 6.0, left_front_dist)
                if left_front_dist <= curr_front_dist + 8.0:
                    front_dist = blended
                    front_speed = min(curr_front_speed, left_front_speed)
                    front_source = "blend_left"
                else:
                    front_dist = left_front_dist
                    front_speed = left_front_speed
                    front_source = "left"

        # 右变道 / 回右时，优先关注右前车
        if state in [FSMState.LANE_CHANGE_RIGHT, FSMState.PREPARE_RETURN_RIGHT]:
            if right_front_dist < 999.0:
                blended = min(curr_front_dist + 6.0, right_front_dist)
                if right_front_dist <= curr_front_dist + 8.0:
                    front_dist = blended
                    front_speed = min(curr_front_speed, right_front_speed)
                    front_source = "blend_right"
                else:
                    front_dist = right_front_dist
                    front_speed = right_front_speed
                    front_source = "right"

        # 紧急制动
        if state == FSMState.EMERGENCY_BRAKE:
            throttle_cmd = 0.0
            brake_cmd = 0.8
            mode = "emergency_brake"
        else:
            # 极近距离，强制刹车
            if front_dist < max(6.0, 0.45 * safe_dist):
                throttle_cmd = 0.0
                brake_cmd = min(MAX_BRAKE, 0.9)
                mode = "hard_brake"
            else:
                # 跟车逻辑
                if front_dist < safe_dist:
                    desired_speed = min(desired_speed, max(0.0, front_speed - 0.5))
                elif front_dist < 1.6 * safe_dist:
                    blend = (front_dist - safe_dist) / max(0.1, 0.6 * safe_dist)
                    blend = max(0.0, min(1.0, blend))
                    front_based_speed = front_speed + 1.0
                    desired_speed = min(
                        desired_speed,
                        blend * desired_speed + (1.0 - blend) * front_based_speed
                    )

                speed_error = desired_speed - current_speed

                if speed_error >= 0.2:
                    throttle_cmd = min(MAX_THROTTLE, KP_SPEED * speed_error)
                    brake_cmd = 0.0
                    mode = "cruise_or_acc"
                elif speed_error <= -0.4:
                    throttle_cmd = 0.0
                    brake_cmd = min(MAX_BRAKE, -0.35 * speed_error)
                    mode = "follow_or_decel"
                else:
                    # 小误差死区，避免油门刹车来回抖
                    throttle_cmd = 0.0
                    brake_cmd = 0.0
                    mode = "hold"

        # 油门 / 刹车互斥 + 平滑
        if brake_cmd > 0.0:
            throttle_cmd = 0.0
        if throttle_cmd > 0.0:
            brake_cmd = 0.0

        throttle = self._smooth_transition(self.last_throttle, throttle_cmd, rise_rate=0.10, fall_rate=0.18)
        brake = self._smooth_transition(self.last_brake, brake_cmd, rise_rate=0.18, fall_rate=0.22)

        # 互斥再保护一次
        if brake > 0.05 and throttle < 0.08:
            throttle = 0.0
        if throttle > 0.05 and brake < 0.05:
            brake = 0.0

        self.last_throttle = float(max(0.0, min(MAX_THROTTLE, throttle)))
        self.last_brake = float(max(0.0, min(MAX_BRAKE, brake)))

        return self.last_throttle, self.last_brake, {
            "mode": mode,
            "safe_dist": safe_dist,
            "desired_speed": desired_speed,
            "front_source": front_source,
            "front_dist_used": front_dist,
            "front_speed_used": front_speed,
        }

    def run_step(self, scene, state):
        target_wp, wp_reason = self._target_waypoint(scene, state)
        if target_wp is None:
            control = carla.VehicleControl(throttle=0.0, brake=0.5, steer=0.0)
            return control, {"reason": "target_wp_none"}

        ego_tf = self.ego.get_transform()
        ego_loc = ego_tf.location
        ego_yaw = math.radians(ego_tf.rotation.yaw)

        tar_loc = target_wp.transform.location
        dx = tar_loc.x - ego_loc.x
        dy = tar_loc.y - ego_loc.y
        target_yaw = math.atan2(dy, dx)

        yaw_error = self._normalize_angle(target_yaw - ego_yaw)
        d_error = yaw_error - self.last_steer_error
        self.last_steer_error = yaw_error

        raw_steer = KP_STEER * yaw_error + KD_STEER * d_error

        # 换道时允许略大一点的转向，但要平滑
        if state in [FSMState.LANE_CHANGE_LEFT, FSMState.LANE_CHANGE_RIGHT]:
            steer_limit = min(MAX_STEER, 0.85)
        else:
            steer_limit = min(MAX_STEER, 0.70)

        raw_steer = max(-steer_limit, min(steer_limit, raw_steer))

        # 转向低通平滑，避免抽动
        steer = 0.75 * self.last_steer_cmd + 0.25 * raw_steer
        steer = max(-steer_limit, min(steer_limit, steer))
        self.last_steer_cmd = steer

        throttle, brake, lon_debug = self._longitudinal_control(scene, state)

        control = carla.VehicleControl(
            throttle=float(throttle),
            brake=float(brake),
            steer=float(steer)
        )

        debug = {
            "steer": float(steer),
            "throttle": float(throttle),
            "brake": float(brake),
            "yaw_error": float(yaw_error),
            "d_error": float(d_error),
            "wp_reason": wp_reason,
            "reason": wp_reason,   # 兼容你现有 logger 的 reason 字段
        }
        debug.update(lon_debug)

        return control, debug
