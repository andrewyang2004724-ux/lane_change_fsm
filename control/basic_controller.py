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
    修复版：
    1. 加入横向误差 cte 控制，减少沿车道线左右摆动
    2. 预瞄距离随速度变化，减小弯道切弯
    3. 弯道根据 yaw_error 主动降速
    4. 高速时收紧 steer_limit，降低摆振和出道风险
    """

    def __init__(self, world_map, ego):
        self.world_map = world_map
        self.ego = ego

        self.last_steer_error = 0.0
        self.last_steer_cmd = 0.0
        self.last_throttle = 0.0
        self.last_brake = 0.0

        self.last_target_wp = None
        self.last_target_lane_change_side = None

        # 新增：横向误差增益
        self.k_cte = 0.75

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

    def _clear_lane_change_cache_if_needed(self, state):
        if (not self._is_left_change_state(state)) and (not self._is_right_change_state(state)):
            self.last_target_lane_change_side = None

    def _preview_distance(self, state, speed):
        """
        预瞄距离随速度变化。
        跟车时不要太远，避免弯道切弯。
        """
        if self._is_left_change_state(state) or self._is_right_change_state(state):
            return max(10.0, min(18.0, 10.0 + 0.35 * speed))
        return max(8.0, min(16.0, 8.0 + 0.30 * speed))

    def _select_lane_change_target_wp(self, lane_wp, preview_dist):
        if lane_wp is None:
            return None
        try:
            nxt = lane_wp.next(preview_dist)
            if len(nxt) > 0:
                return nxt[0]
        except Exception:
            pass
        return lane_wp

    def _target_waypoint(self, scene, state):
        ego_wp = scene["ego_wp"]
        if ego_wp is None:
            return None, "ego_wp_missing"

        self._clear_lane_change_cache_if_needed(state)
        speed = self._speed()
        preview_dist = self._preview_distance(state, speed)

        if self._is_left_change_state(state):
            target_wp = self._select_lane_change_target_wp(scene["left_wp"], preview_dist)
            if target_wp is not None:
                self.last_target_wp = target_wp
                self.last_target_lane_change_side = "left"
                return target_wp, "left_changing_wp"

            if self.last_target_lane_change_side == "left" and self.last_target_wp is not None:
                return self.last_target_wp, "left_changing_wp_reuse"

            nxt = ego_wp.next(max(8.0, preview_dist))
            if len(nxt) > 0:
                return nxt[0], "left_changing_wp_missing_fallback"
            return ego_wp, "left_changing_wp_missing_fallback"

        if self._is_right_change_state(state):
            target_wp = self._select_lane_change_target_wp(scene["right_wp"], preview_dist)
            if target_wp is not None:
                self.last_target_wp = target_wp
                self.last_target_lane_change_side = "right"
                return target_wp, "right_changing_wp"

            if self.last_target_lane_change_side == "right" and self.last_target_wp is not None:
                return self.last_target_wp, "right_changing_wp_reuse"

            nxt = ego_wp.next(max(8.0, preview_dist))
            if len(nxt) > 0:
                return nxt[0], "right_changing_wp_missing"
            return ego_wp, "right_changing_wp_missing"

        try:
            nxt = ego_wp.next(preview_dist)
            if len(nxt) > 0:
                self.last_target_wp = nxt[0]
                return nxt[0], "follow_lane_wp"
        except Exception:
            pass

        self.last_target_wp = ego_wp
        return ego_wp, "follow_lane_wp"

    def _smooth_transition(self, prev, new, rise_rate=0.12, fall_rate=0.20):
        if new > prev:
            return min(new, prev + rise_rate)
        return max(new, prev - fall_rate)

    def _curve_speed_limit(self, yaw_error_abs):
        """
        基于航向误差的最小侵入弯道限速。
        """
        if yaw_error_abs > 0.22:
            return 11.0
        if yaw_error_abs > 0.16:
            return 14.0
        if yaw_error_abs > 0.10:
            return 18.0
        if yaw_error_abs > 0.06:
            return 22.0
        return TARGET_SPEED

    def _longitudinal_control(self, scene, state, yaw_error_abs=0.0):
        current_speed = self._speed()

        curr_front_dist = scene["curr_front"]["dist"]
        curr_front_speed = scene["curr_front"]["speed"]

        left_front_dist = scene["left_front"]["dist"]
        left_front_speed = scene["left_front"]["speed"]

        right_front_dist = scene["right_front"]["dist"]
        right_front_speed = scene["right_front"]["speed"]

        if state in [FSMState.LANE_CHANGE_LEFT]:
            desired_speed = OVERTAKE_SPEED
        elif state in [FSMState.LANE_CHANGE_RIGHT, FSMState.PREPARE_RETURN_RIGHT]:
            desired_speed = min(TARGET_SPEED + 1.0, OVERTAKE_SPEED)
        elif state == FSMState.EMERGENCY_BRAKE:
            desired_speed = 0.0
        else:
            desired_speed = TARGET_SPEED

        # 新增：弯道限速
        desired_speed = min(desired_speed, self._curve_speed_limit(yaw_error_abs))

        safe_dist = MIN_GAP + TIME_HEADWAY * current_speed

        front_dist = curr_front_dist
        front_speed = curr_front_speed
        front_source = "curr"

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

        if state == FSMState.EMERGENCY_BRAKE:
            throttle_cmd = 0.0
            brake_cmd = 0.8
            mode = "emergency_brake"
        else:
            if front_dist < max(6.0, 0.45 * safe_dist):
                throttle_cmd = 0.0
                brake_cmd = min(MAX_BRAKE, 0.9)
                mode = "hard_brake"
            else:
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
                elif speed_error <= -0.3:
                    throttle_cmd = 0.0
                    brake_cmd = min(MAX_BRAKE, -0.40 * speed_error)
                    mode = "follow_or_decel"
                else:
                    throttle_cmd = 0.0
                    brake_cmd = 0.0
                    mode = "hold"

        if brake_cmd > 0.0:
            throttle_cmd = 0.0
        if throttle_cmd > 0.0:
            brake_cmd = 0.0

        throttle = self._smooth_transition(self.last_throttle, throttle_cmd, rise_rate=0.08, fall_rate=0.15)
        brake = self._smooth_transition(self.last_brake, brake_cmd, rise_rate=0.20, fall_rate=0.20)

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

        # 新增：横向误差 cte（在自车坐标系下，目标点的横向偏移）
        local_x = math.cos(ego_yaw) * dx + math.sin(ego_yaw) * dy
        local_y = -math.sin(ego_yaw) * dx + math.cos(ego_yaw) * dy
        cte = local_y

        # 速度越高，cte 权重要适当收敛，防止高速大幅摆动
        speed = self._speed()
        k_cte_eff = self.k_cte / max(1.0, 0.25 * speed)

        raw_steer = KP_STEER * yaw_error + KD_STEER * d_error + k_cte_eff * math.atan2(cte, max(4.0, speed))

        # 高速时降低 steer limit
        if speed > 22.0:
            steer_limit = min(MAX_STEER, 0.38)
        elif speed > 16.0:
            steer_limit = min(MAX_STEER, 0.45)
        elif speed > 10.0:
            steer_limit = min(MAX_STEER, 0.55)
        else:
            steer_limit = min(MAX_STEER, 0.70)

        if state in [FSMState.LANE_CHANGE_LEFT, FSMState.LANE_CHANGE_RIGHT]:
            steer_limit = min(MAX_STEER, max(steer_limit, 0.55))

        raw_steer = max(-steer_limit, min(steer_limit, raw_steer))

        # 转向平滑再加强一点
        steer = 0.82 * self.last_steer_cmd + 0.18 * raw_steer
        steer = max(-steer_limit, min(steer_limit, steer))
        self.last_steer_cmd = steer

        throttle, brake, lon_debug = self._longitudinal_control(scene, state, yaw_error_abs=abs(yaw_error))

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
            "cte": float(cte),
            "local_x": float(local_x),
            "local_y": float(local_y),
            "wp_reason": wp_reason,
            "reason": wp_reason,
            "target_lane_change_side": self.last_target_lane_change_side,
        }
        debug.update(lon_debug)

        return control, debug
