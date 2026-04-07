# control/basic_controller.py
import math
import carla
from config import (
    TARGET_SPEED,
    OVERTAKE_SPEED,
    MAX_THROTTLE,
    MAX_BRAKE,
    MAX_STEER,
    MIN_GAP,
    TIME_HEADWAY,
)
from decision.state_machine import FSMState

class BasicLaneController:
    """
    高阶重构版 (参照 Apollo / Autoware & IDM 理念):
    1. 横向采用前馈加反馈的 Stanley Controller，消除纯跟踪在急弯的“切弯超调”现象。
    2. 纵向采用 Intelligent Driver Model (IDM)，输出连续平滑加速度，实现拟人化跟车。
    """

    def __init__(self, world_map, ego):
        self.world_map = world_map
        self.ego = ego

        self.last_steer_cmd = 0.0
        self.last_throttle = 0.0
        self.last_brake = 0.0

        self.last_target_wp = None
        self.last_target_lane_change_side = None

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
        Stanley 的预瞄距离 (Preview Distance)
        因为车道线是先验且精确的，Stanley主要依赖当前点的法向误差，
        这里的 preview 只是为了补偿 Carla 物理引擎的系统延迟。
        """
        if self._is_left_change_state(state) or self._is_right_change_state(state):
            # 换道时，看远一些，产生柔和的过渡轨迹
            return max(8.0, speed * 0.8)
        
        # 巡航时，距离极短，紧紧咬住当前车道线
        return max(3.0, speed * 0.25)

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

    def _target_waypoint(self, scene, state, preview_dist):
        ego_wp = scene["ego_wp"]
        if ego_wp is None:
            return None, "ego_wp_missing"

        self._clear_lane_change_cache_if_needed(state)

        if self._is_left_change_state(state):
            target_wp = self._select_lane_change_target_wp(scene["left_wp"], preview_dist)
            if target_wp is not None:
                self.last_target_wp = target_wp
                self.last_target_lane_change_side = "left"
                return target_wp, "left_changing_wp"
            if self.last_target_lane_change_side == "left" and self.last_target_wp is not None:
                return self.last_target_wp, "left_changing_wp_reuse"
            nxt = ego_wp.next(max(8.0, preview_dist))
            return nxt[0] if len(nxt) > 0 else ego_wp, "left_changing_wp_missing_fallback"

        if self._is_right_change_state(state):
            target_wp = self._select_lane_change_target_wp(scene["right_wp"], preview_dist)
            if target_wp is not None:
                self.last_target_wp = target_wp
                self.last_target_lane_change_side = "right"
                return target_wp, "right_changing_wp"
            if self.last_target_lane_change_side == "right" and self.last_target_wp is not None:
                return self.last_target_wp, "right_changing_wp_reuse"
            nxt = ego_wp.next(max(8.0, preview_dist))
            return nxt[0] if len(nxt) > 0 else ego_wp, "right_changing_wp_missing"

        try:
            nxt = ego_wp.next(preview_dist)
            if len(nxt) > 0:
                self.last_target_wp = nxt[0]
                return nxt[0], "follow_lane_wp"
        except Exception:
            pass

        self.last_target_wp = ego_wp
        return ego_wp, "follow_lane_wp"

    def _smooth_transition(self, prev, new, rise_rate=0.08, fall_rate=0.15):
        if new > prev:
            return min(new, prev + rise_rate)
        return max(new, prev - fall_rate)

    def _curve_speed_limit(self, yaw_error_abs):
        # 弯道降速逻辑保留，提升安全性
        if yaw_error_abs > 0.22:
            return 11.0
        if yaw_error_abs > 0.16:
            return 15.0
        if yaw_error_abs > 0.10:
            return 19.0
        return TARGET_SPEED

    def _longitudinal_control_idm(self, scene, state, yaw_error_abs):
        """
        引入 Intelligent Driver Model (IDM) 智能驾驶跟驰模型
        彻底解决僵硬的起步刹车，实现如同人类驾驶员般平滑丝滑的过渡。
        """
        current_speed = max(self._speed(), 0.1)

        # 1. 确定期望速度 v0
        if state in [FSMState.LANE_CHANGE_LEFT]:
            v0 = OVERTAKE_SPEED
        elif state in [FSMState.LANE_CHANGE_RIGHT, FSMState.PREPARE_RETURN_RIGHT]:
            v0 = min(TARGET_SPEED + 1.0, OVERTAKE_SPEED)
        elif state == FSMState.EMERGENCY_BRAKE:
            v0 = 0.0
        else:
            v0 = TARGET_SPEED
        v0 = min(v0, self._curve_speed_limit(yaw_error_abs))

        # 2. 识别关键前车
        s = scene["curr_front"]["dist"]
        v_lead = scene["curr_front"]["speed"]

        # 换道时的前车融合博弈逻辑
        if state in [FSMState.PREPARE_LANE_CHANGE_LEFT, FSMState.LANE_CHANGE_LEFT]:
            left_dist = scene["left_front"]["dist"]
            if left_dist < 999.0:
                s = min(s, left_dist)
                v_lead = min(v_lead, scene["left_front"]["speed"])

        if state in [FSMState.LANE_CHANGE_RIGHT, FSMState.PREPARE_RETURN_RIGHT]:
            right_dist = scene["right_front"]["dist"]
            if right_dist < 999.0:
                s = min(s, right_dist)
                v_lead = min(v_lead, scene["right_front"]["speed"])

        # 3. IDM 参数定义
        a_max = 1.5   # 车辆最大舒适加速度 (m/s^2)
        b = 2.0       # 车辆舒适减速度 (m/s^2)
        delta = 4.0   # 加速度指数
        s0 = MIN_GAP  # 停止时的最小安全间距

        delta_v = current_speed - v_lead

        # 4. IDM 核心微分计算
        if s > 150.0:
            # 前方畅通无阻，自由巡航
            s_star = 0.0
            term2 = 0.0
        else:
            # 期望跟车距离计算
            s_star = s0 + current_speed * TIME_HEADWAY + (current_speed * delta_v) / (2.0 * math.sqrt(a_max * b))
            s_star = max(s_star, s0) # 保底安全距离
            term2 = (s_star / max(s, 0.1)) ** 2
            
        term1 = (current_speed / max(v0, 0.1)) ** delta
        
        # 得到连续平滑的期望加速度
        accel = a_max * (1.0 - term1 - term2)

        # 5. 紧急状况超控
        if state == FSMState.EMERGENCY_BRAKE or s < s0 * 0.5:
            throttle_cmd, brake_cmd = 0.0, 0.8
            mode = "emergency_brake"
        else:
            # 将加速度映射为油门和刹车踏板指令，设置小死区防抖
            if accel > 0.15:
                throttle_cmd = min(MAX_THROTTLE, accel / a_max)
                brake_cmd = 0.0
                mode = "idm_accel"
            elif accel < -0.15:
                throttle_cmd = 0.0
                brake_cmd = min(MAX_BRAKE, -accel / b)
                mode = "idm_brake"
            else:
                throttle_cmd, brake_cmd = 0.0, 0.0
                mode = "idm_coast"

        # 踏板平滑滤波模拟人类脚部动作
        throttle = self._smooth_transition(self.last_throttle, throttle_cmd, rise_rate=0.06, fall_rate=0.15)
        brake = self._smooth_transition(self.last_brake, brake_cmd, rise_rate=0.15, fall_rate=0.15)

        # 踏板互斥
        if brake > 0.02: throttle = 0.0
        if throttle > 0.02: brake = 0.0

        self.last_throttle = float(max(0.0, min(MAX_THROTTLE, throttle)))
        self.last_brake = float(max(0.0, min(MAX_BRAKE, brake)))

        return self.last_throttle, self.last_brake, {
            "mode": mode, "idm_accel": accel, "s_star": s_star if s <= 150 else 0.0, "s": s
        }

    def run_step(self, scene, state):
        speed = self._speed()
        preview_dist = self._preview_distance(state, speed)
        
        target_wp, wp_reason = self._target_waypoint(scene, state, preview_dist)
        if target_wp is None:
            return carla.VehicleControl(throttle=0.0, brake=0.5, steer=0.0), {"reason": "no_wp"}

        ego_tf = self.ego.get_transform()
        ego_loc = ego_tf.location
        ego_yaw = math.radians(ego_tf.rotation.yaw)

        tar_loc = target_wp.transform.location
        tar_yaw = math.radians(target_wp.transform.rotation.yaw)
        
        # 1. Stanley: 计算前轴中心点的物理预测位置 (补偿延迟)
        L = 2.8 # 假定车辆轴距
        t_delay = 0.10 # CARLA控制系统延迟
        L_forward = (L / 2.0) + speed * t_delay
        
        front_x = ego_loc.x + L_forward * math.cos(ego_yaw)
        front_y = ego_loc.y + L_forward * math.sin(ego_yaw)
        
        dx = front_x - tar_loc.x
        dy = front_y - tar_loc.y
        
        # 2. 计算 Cross-Track Error (CTE, 横向偏差)
        # 根据CARLA左手坐标系法则精确投影计算
        cte = math.sin(tar_yaw) * dx - math.cos(tar_yaw) * dy
        
        # 3. 计算 Heading Error (航向角偏差)
        yaw_error = self._normalize_angle(tar_yaw - ego_yaw)
        
        # 4. Stanley 控制律核心公式
        k_ce = 1.0  # 横向误差跟踪增益
        k_s = 3.0   # 低速软化常数，防止低速打盘过猛
        
        # 为了让状态机在突然换道(Jump)时平滑，我们在换道态降低增益并限制最大偏差
        if self._is_left_change_state(state) or self._is_right_change_state(state):
            cte = max(-2.5, min(2.5, cte))
            k_ce = 0.6  # 调小增益，让换道像贝塞尔曲线般丝滑融入
            yaw_error = max(-0.3, min(0.3, yaw_error))
            
        cte_steer = math.atan2(k_ce * cte, speed + k_s)
        steer_rad = yaw_error + cte_steer
        
        MAX_STEER_RAD = 1.22
        raw_steer = steer_rad / MAX_STEER_RAD

        # 车辆动力学高低速转向限位保障防翻车
        if speed > 22.0:
            steer_limit = min(MAX_STEER, 0.20)
        elif speed > 15.0:
            steer_limit = min(MAX_STEER, 0.35)
        else:
            steer_limit = min(MAX_STEER, 0.70)

        raw_steer = max(-steer_limit, min(steer_limit, raw_steer))

        # 轻微滞后滤波消除高频抖动
        steer = 0.30 * self.last_steer_cmd + 0.70 * raw_steer
        self.last_steer_cmd = steer

        # 5. 纵向 IDM 跟驰计算
        throttle, brake, lon_debug = self._longitudinal_control_idm(scene, state, yaw_error_abs=abs(yaw_error))

        control = carla.VehicleControl(
            throttle=float(throttle),
            brake=float(brake),
            steer=float(steer)
        )

        debug = {
            "steer": float(steer), "throttle": float(throttle), "brake": float(brake),
            "cte": float(cte), "yaw_error": float(yaw_error), "wp_reason": wp_reason
        }
        debug.update(lon_debug)
        return control, debug