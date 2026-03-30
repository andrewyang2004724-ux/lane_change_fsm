# decision/fsm_decider.py
from decision.state_machine import FSMState
from decision.behavior_rules import (
    should_overtake,
    emergency_needed,
    evaluate_left_change,
    can_return_right,
)
from perception.scene_builder import build_scene
from perception.vehicle_filter import vehicle_speed

class LaneChangeDecider:
    """
    更稳健的有限状态机决策器

    改进点：
    1. FOLLOW -> PREPARE -> EXECUTE 使用连续稳定计数，减少抖动
    2. PREPARE 阶段不是 1 帧跳转，而是再次确认安全后才执行
    3. LANE_CHANGE 阶段增加超时保护，避免一直卡住
    4. OVERTAKE / RETURN 过程增加稳定保持逻辑
    5. EMERGENCY_BRAKE 恢复更温和
    6. 兼容现有 FSMState，不新增 state_machine.py 枚举
    """

    def __init__(
        self,
        world,
        world_map,
        ego,
        stable_ticks=4,
        prepare_confirm_ticks=2,
        lane_change_timeout=45,
        emergency_release_ticks=6,
    ):
        self.world = world
        self.world_map = world_map
        self.ego = ego

        self.state = FSMState.FOLLOW_LANE

        # 跟你原来兼容的计数器
        self.prepare_count = 0
        self.stable_ticks = stable_ticks

        # 新增：prepare 再确认
        self.prepare_confirm_ticks = prepare_confirm_ticks
        self.prepare_confirm_count = 0

        # 新增：变道超时
        self.lane_change_timeout = lane_change_timeout
        self.lane_change_ticks = 0

        # 新增：紧急制动恢复稳定计数
        self.emergency_release_ticks = emergency_release_ticks
        self.emergency_release_count = 0

        # 超车目标
        self.overtake_target_id = None

        # False: 正常“超车完成后回右”
        # True : “左侧不可用时，主动向右绕行”
        self.right_change_is_bypass = False

        # 记录准备方向，避免 prepare 阶段来回跳
        self.pending_direction = None  # "left" / "right" / None

    def _find_actor_by_id(self, actor_id):
        if actor_id is None:
            return None
        actors = self.world.get_actors().filter("vehicle.*")
        for a in actors:
            if a.id == actor_id:
                return a
        return None

    def _get_target_pass_margin(self, target_id):
        """
        计算 ego 相对目标车的纵向领先距离。
        正值通常表示 ego 已经超过目标车。
        """
        if target_id is None:
            return 0.0

        target = self._find_actor_by_id(target_id)
        if target is None:
            return 999.0

        from perception.vehicle_filter import longitudinal_distance
        return longitudinal_distance(self.ego, target)

    def _need_bypass(self, ego_speed, curr_front):
        """
        是否需要绕过前车。
        """
        if curr_front["vehicle"] is None:
            return False

        if should_overtake(ego_speed, curr_front["dist"], curr_front["speed"]):
            return True

        # 补充一个更积极的触发条件，减少长期低速跟车
        rel_speed = ego_speed - curr_front["speed"]
        if curr_front["dist"] < 28.0 and rel_speed > 2.0:
            return True

        return False

    def _evaluate_right_bypass(self, ego_speed, curr_front, right_front, right_rear, scene):
        """
        评估是否可以向右绕行。
        不依赖 behavior_rules 新接口，保持兼容。
        """
        if scene["right_wp"] is None or (not scene["right_allowed"]):
            return {"safe": False, "reason": "right_not_allowed"}, -1.0

        if curr_front["vehicle"] is None:
            return {"safe": False, "reason": "no_front_vehicle"}, -1.0

        min_front_gap = max(12.0, ego_speed * 1.2)
        min_rear_gap = max(10.0, ego_speed * 1.0)

        right_front_ok = right_front["dist"] > min_front_gap
        right_rear_ok = right_rear["dist"] > min_rear_gap

        rear_rel_speed = right_rear["speed"] - ego_speed
        rear_speed_ok = rear_rel_speed < 5.0

        safe = right_front_ok and right_rear_ok and rear_speed_ok

        curr_penalty = 0.0
        if curr_front["dist"] < 35.0:
            curr_penalty += (35.0 - curr_front["dist"]) / 35.0
        if ego_speed > curr_front["speed"]:
            curr_penalty += min(1.0, (ego_speed - curr_front["speed"]) / 8.0)

        right_bonus = min(1.0, right_front["dist"] / 40.0)
        benefit = 0.6 * curr_penalty + 0.4 * right_bonus - 0.2

        reason = "safe" if safe else "unsafe_gap"
        return {
            "safe": safe,
            "reason": reason,
            "min_front_gap": min_front_gap,
            "min_rear_gap": min_rear_gap,
            "rear_rel_speed": rear_rel_speed,
        }, benefit

    def _reset_prepare(self):
        self.prepare_count = 0
        self.prepare_confirm_count = 0
        self.pending_direction = None

    def _start_prepare(self, direction, curr_front):
        """
        direction: "left" / "right"
        """
        self.pending_direction = direction
        self.prepare_count = 0
        self.prepare_confirm_count = 0

        if curr_front["vehicle"] is not None:
            self.overtake_target_id = curr_front["vehicle"].id

    def _enter_lane_change(self, to_right_bypass=False):
        self.lane_change_ticks = 0
        self.right_change_is_bypass = to_right_bypass

        if to_right_bypass:
            self.state = FSMState.LANE_CHANGE_RIGHT
        else:
            self.state = FSMState.LANE_CHANGE_LEFT

    def update(self):
        scene = build_scene(self.world, self.world_map, self.ego)
        ego_speed = vehicle_speed(self.ego)

        curr_front = scene["curr_front"]
        left_front = scene["left_front"]
        left_rear = scene["left_rear"]
        right_front = scene["right_front"]
        right_rear = scene["right_rear"]

        emergency, ttc = emergency_needed(
            curr_front["dist"],
            ego_speed,
            curr_front["speed"]
        )

        extra = {
            "ttc": ttc,
            "prepare_count": self.prepare_count,
            "prepare_confirm_count": self.prepare_confirm_count,
            "lane_change_ticks": self.lane_change_ticks,
            "right_change_is_bypass": self.right_change_is_bypass,
            "overtake_target_id": self.overtake_target_id,
            "pending_direction": self.pending_direction,
        }

        # =============================
        # 紧急制动优先
        # =============================
        if emergency and self.state != FSMState.EMERGENCY_BRAKE:
            self.state = FSMState.EMERGENCY_BRAKE
            self._reset_prepare()
            self.lane_change_ticks = 0
            self.emergency_release_count = 0
            extra["decision"] = "enter_emergency"
            extra["reason"] = "emergency"
            return self.state, scene, extra

        need_bypass = self._need_bypass(ego_speed, curr_front)
        extra["need_bypass"] = need_bypass

        left_candidate = False
        left_safe = False
        left_benefit = -1.0
        left_safe_info = None

        right_candidate = False
        right_safe = False
        right_benefit = -1.0
        right_safe_info = None

        # ---------- 左侧评估 ----------
        if scene["left_wp"] is not None and scene["left_allowed"] and need_bypass:
            left_candidate = True
            safe_info, benefit = evaluate_left_change(
                ego_speed,
                curr_front["dist"], curr_front["speed"],
                left_front["dist"], left_front["speed"],
                left_rear["dist"], left_rear["speed"]
            )
            left_safe = safe_info.get("safe", False)
            left_benefit = benefit
            left_safe_info = safe_info

        # ---------- 右侧绕行评估 ----------
        if scene["right_wp"] is not None and scene["right_allowed"] and need_bypass:
            right_candidate = True
            safe_info_r, benefit_r = self._evaluate_right_bypass(
                ego_speed, curr_front, right_front, right_rear, scene
            )
            right_safe = safe_info_r.get("safe", False)
            right_benefit = benefit_r
            right_safe_info = safe_info_r

        extra["left_candidate"] = left_candidate
        extra["left_safe"] = left_safe
        extra["left_benefit"] = left_benefit
        extra["left_safe_info"] = left_safe_info

        extra["right_candidate"] = right_candidate
        extra["right_safe"] = right_safe
        extra["right_benefit"] = right_benefit
        extra["right_safe_info"] = right_safe_info

        choose_left = left_candidate and left_safe and (left_benefit > 0.02)
        choose_right = right_candidate and right_safe and (right_benefit > 0.02)

        # =============================
        # FOLLOW_LANE
        # =============================
        if self.state == FSMState.FOLLOW_LANE:
            if not need_bypass:
                self._reset_prepare()
                extra["decision"] = "stay_follow_no_need"
                return self.state, scene, extra

            # 优先左侧，其次右绕行
            if choose_left:
                if self.pending_direction not in (None, "left"):
                    self._reset_prepare()

                if self.pending_direction is None:
                    self._start_prepare("left", curr_front)

                self.prepare_count += 1
                extra["decision"] = "prepare_left_counting"

                if self.prepare_count >= self.stable_ticks:
                    self.state = FSMState.PREPARE_LANE_CHANGE_LEFT
                    self.prepare_confirm_count = 0
                    extra["decision"] = "enter_prepare_left"

            elif choose_right:
                if self.pending_direction not in (None, "right"):
                    self._reset_prepare()

                if self.pending_direction is None:
                    self._start_prepare("right", curr_front)

                self.prepare_count += 1
                extra["decision"] = "prepare_right_bypass_counting"

                if self.prepare_count >= self.stable_ticks:
                    self.state = FSMState.PREPARE_RETURN_RIGHT
                    self.prepare_confirm_count = 0
                    self.right_change_is_bypass = True
                    extra["decision"] = "enter_prepare_right_bypass"

            else:
                self._reset_prepare()
                extra["decision"] = "stay_follow_no_safe_gap"

        # =============================
        # PREPARE_LANE_CHANGE_LEFT
        # =============================
        elif self.state == FSMState.PREPARE_LANE_CHANGE_LEFT:
            if not need_bypass:
                self.state = FSMState.FOLLOW_LANE
                self._reset_prepare()
                extra["decision"] = "cancel_prepare_left_no_need"

            elif not choose_left:
                self.state = FSMState.FOLLOW_LANE
                self._reset_prepare()
                extra["decision"] = "cancel_prepare_left_not_safe"

            else:
                self.prepare_confirm_count += 1
                extra["decision"] = "prepare_left_confirming"

                if self.prepare_confirm_count >= self.prepare_confirm_ticks:
                    self._reset_prepare()
                    self._enter_lane_change(to_right_bypass=False)
                    extra["decision"] = "execute_left_change"

        # =============================
        # LANE_CHANGE_LEFT
        # =============================
        elif self.state == FSMState.LANE_CHANGE_LEFT:
            self.lane_change_ticks += 1

            # 注意：这里沿用你原来的 lane_id 判定方式
            if scene["left_wp"] is not None and scene["ego_wp"] is not None:
                if scene["ego_wp"].lane_id == scene["left_wp"].lane_id:
                    self.state = FSMState.OVERTAKE_CRUISE
                    self.lane_change_ticks = 0
                    extra["decision"] = "left_change_done_enter_overtake"
                else:
                    if self.lane_change_ticks > self.lane_change_timeout:
                        # 超时后退回跟车，避免永远卡在变道态
                        self.state = FSMState.FOLLOW_LANE
                        self.lane_change_ticks = 0
                        self._reset_prepare()
                        extra["decision"] = "left_change_timeout_fallback"
                    else:
                        extra["decision"] = "left_changing"
            else:
                if self.lane_change_ticks > self.lane_change_timeout:
                    self.state = FSMState.FOLLOW_LANE
                    self.lane_change_ticks = 0
                    self._reset_prepare()
                    extra["decision"] = "left_change_wp_missing_timeout_fallback"
                else:
                    extra["decision"] = "left_changing_wp_missing"

        # =============================
        # OVERTAKE_CRUISE
        # =============================
        elif self.state == FSMState.OVERTAKE_CRUISE:
            pass_margin = self._get_target_pass_margin(self.overtake_target_id)
            extra["pass_margin"] = pass_margin

            if scene["right_wp"] is not None and scene["right_allowed"]:
                if can_return_right(
                    pass_margin,
                    right_front["dist"], right_front["speed"],
                    right_rear["dist"], right_rear["speed"],
                    ego_speed
                ):
                    self.state = FSMState.PREPARE_RETURN_RIGHT
                    self.prepare_confirm_count = 0
                    self.right_change_is_bypass = False
                    extra["decision"] = "prepare_return_right"
                else:
                    extra["decision"] = "keep_overtake_cruise"
            else:
                extra["decision"] = "keep_overtake_cruise_no_right"

        # =============================
        # PREPARE_RETURN_RIGHT
        # 兼容两种语义：
        # 1) 左超后回右
        # 2) 主动右绕行
        # =============================
        elif self.state == FSMState.PREPARE_RETURN_RIGHT:
            if self.right_change_is_bypass:
                # 右绕行：仍需确认 need_bypass 和右侧安全
                if not need_bypass:
                    self.state = FSMState.FOLLOW_LANE
                    self._reset_prepare()
                    self.right_change_is_bypass = False
                    extra["decision"] = "cancel_prepare_right_bypass_no_need"

                elif not choose_right:
                    self.state = FSMState.FOLLOW_LANE
                    self._reset_prepare()
                    self.right_change_is_bypass = False
                    extra["decision"] = "cancel_prepare_right_bypass_not_safe"

                else:
                    self.prepare_confirm_count += 1
                    extra["decision"] = "prepare_right_bypass_confirming"

                    if self.prepare_confirm_count >= self.prepare_confirm_ticks:
                        self._reset_prepare()
                        self.state = FSMState.LANE_CHANGE_RIGHT
                        self.lane_change_ticks = 0
                        self.right_change_is_bypass = True
                        extra["decision"] = "execute_right_bypass"
            else:
                # 回右：主要看右侧是否允许存在
                if scene["right_wp"] is None or (not scene["right_allowed"]):
                    self.state = FSMState.OVERTAKE_CRUISE
                    self.prepare_confirm_count = 0
                    extra["decision"] = "cancel_return_right_no_right_lane"
                else:
                    self.prepare_confirm_count += 1
                    extra["decision"] = "prepare_return_right_confirming"

                    if self.prepare_confirm_count >= self.prepare_confirm_ticks:
                        self._reset_prepare()
                        self.state = FSMState.LANE_CHANGE_RIGHT
                        self.lane_change_ticks = 0
                        self.right_change_is_bypass = False
                        extra["decision"] = "execute_return_right"

        # =============================
        # LANE_CHANGE_RIGHT
        # =============================
        elif self.state == FSMState.LANE_CHANGE_RIGHT:
            self.lane_change_ticks += 1

            if scene["right_wp"] is not None and scene["ego_wp"] is not None:
                if scene["ego_wp"].lane_id == scene["right_wp"].lane_id:
                    if self.right_change_is_bypass:
                        self.state = FSMState.OVERTAKE_CRUISE
                        extra["decision"] = "right_bypass_done_enter_overtake"
                    else:
                        self.state = FSMState.FOLLOW_LANE
                        self.overtake_target_id = None
                        extra["decision"] = "return_right_done_follow"

                    self.right_change_is_bypass = False
                    self.lane_change_ticks = 0
                else:
                    if self.lane_change_ticks > self.lane_change_timeout:
                        # 右绕行失败 / 回右失败，保守回退
                        if self.right_change_is_bypass:
                            self.state = FSMState.FOLLOW_LANE
                            extra["decision"] = "right_bypass_timeout_fallback"
                        else:
                            self.state = FSMState.OVERTAKE_CRUISE
                            extra["decision"] = "return_right_timeout_back_to_overtake"

                        self.lane_change_ticks = 0
                        self.right_change_is_bypass = False
                    else:
                        if self.right_change_is_bypass:
                            extra["decision"] = "right_bypass_changing"
                        else:
                            extra["decision"] = "return_right_changing"
            else:
                if self.lane_change_ticks > self.lane_change_timeout:
                    if self.right_change_is_bypass:
                        self.state = FSMState.FOLLOW_LANE
                        extra["decision"] = "right_bypass_wp_missing_timeout_fallback"
                    else:
                        self.state = FSMState.OVERTAKE_CRUISE
                        extra["decision"] = "return_right_wp_missing_timeout_back_to_overtake"

                    self.lane_change_ticks = 0
                    self.right_change_is_bypass = False
                else:
                    extra["decision"] = "right_changing_wp_missing"

        # =============================
        # EMERGENCY_BRAKE
        # =============================
        elif self.state == FSMState.EMERGENCY_BRAKE:
            if emergency:
                self.emergency_release_count = 0
                extra["decision"] = "keep_emergency"
            else:
                self.emergency_release_count += 1
                extra["decision"] = "emergency_releasing"

                if self.emergency_release_count >= self.emergency_release_ticks:
                    self.state = FSMState.FOLLOW_LANE
                    self._reset_prepare()
                    self.lane_change_ticks = 0
                    self.emergency_release_count = 0
                    extra["decision"] = "recover_from_emergency"

        return self.state, scene, extra
