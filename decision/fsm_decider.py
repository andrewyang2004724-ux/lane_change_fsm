# decision/fsm_decider.py
from decision.state_machine import FSMState
from decision.behavior_rules import (
    should_overtake,
    emergency_needed,
    evaluate_left_change,
    evaluate_right_bypass,
    can_return_right,
)
from perception.scene_builder import build_scene
from perception.vehicle_filter import vehicle_speed

class LaneChangeDecider:
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
        self.prepare_count = 0
        self.stable_ticks = stable_ticks
        self.prepare_confirm_ticks = prepare_confirm_ticks
        self.prepare_confirm_count = 0
        self.lane_change_timeout = lane_change_timeout
        self.lane_change_ticks = 0
        self.emergency_release_ticks = emergency_release_ticks
        self.emergency_release_count = 0
        self.overtake_target_id = None
        self.right_change_is_bypass = False
        self.pending_direction = None
        self.target_lane_id = None

    def _find_actor_by_id(self, actor_id):
        if actor_id is None:
            return None
        actors = self.world.get_actors().filter("vehicle.*")
        for a in actors:
            if a.id == actor_id:
                return a
        return None

    def _get_target_pass_margin(self, target_id):
        if target_id is None: return 0.0
        target = self._find_actor_by_id(target_id)
        if target is None: return 999.0
        from perception.vehicle_filter import longitudinal_distance
        return longitudinal_distance(self.ego, target)

    def _need_bypass(self, ego_speed, curr_front):
        if curr_front["vehicle"] is None:
            return False
        if should_overtake(ego_speed, curr_front["dist"], curr_front["speed"]):
            return True
        return False

    def _reset_prepare(self):
        self.prepare_count = 0
        self.prepare_confirm_count = 0
        self.pending_direction = None

    def _start_prepare(self, direction, curr_front):
        self.pending_direction = direction
        self.prepare_count = 0
        self.prepare_confirm_count = 0
        if curr_front["vehicle"] is not None:
            self.overtake_target_id = curr_front["vehicle"].id

    def _lock_target_lane_id(self, scene, direction):
        target_wp = scene["left_wp"] if direction == "left" else scene["right_wp"]
        if target_wp is not None:
            self.target_lane_id = getattr(target_wp, "lane_id", None)
        else:
            self.target_lane_id = None

    def _enter_lane_change(self, scene, to_right_bypass=False):
        self.lane_change_ticks = 0
        self.right_change_is_bypass = to_right_bypass
        if to_right_bypass:
            self._lock_target_lane_id(scene, "right")
            self.state = FSMState.LANE_CHANGE_RIGHT
        else:
            self._lock_target_lane_id(scene, "left")
            self.state = FSMState.LANE_CHANGE_LEFT

    def _is_lane_change_finished(self, scene):
        ego_wp = scene.get("ego_wp", None)
        if ego_wp is None: return False
        ego_lane_id = getattr(ego_wp, "lane_id", None)
        if ego_lane_id is None: return False
        if self.target_lane_id is not None:
            return ego_lane_id == self.target_lane_id
        return False

    def update(self):
        scene = build_scene(self.world, self.world_map, self.ego)
        ego_speed = vehicle_speed(self.ego)

        curr_front = scene["curr_front"]
        left_front = scene["left_front"]
        left_rear = scene["left_rear"]
        right_front = scene["right_front"]
        right_rear = scene["right_rear"]

        emergency, ttc = emergency_needed(curr_front["dist"], ego_speed, curr_front["speed"])

        extra = {
            "ttc": ttc,
            "prepare_count": self.prepare_count,
            "prepare_confirm_count": self.prepare_confirm_count,
            "lane_change_ticks": self.lane_change_ticks,
            "right_change_is_bypass": self.right_change_is_bypass,
            "overtake_target_id": self.overtake_target_id,
            "pending_direction": self.pending_direction,
            "target_lane_id": self.target_lane_id,
        }

        if emergency and self.state != FSMState.EMERGENCY_BRAKE:
            self.state = FSMState.EMERGENCY_BRAKE
            self._reset_prepare()
            self.lane_change_ticks = 0
            self.emergency_release_count = 0
            self.target_lane_id = None
            extra["decision"] = "enter_emergency"
            return self.state, scene, extra

        need_bypass = self._need_bypass(ego_speed, curr_front)
        extra["need_bypass"] = need_bypass

        left_candidate, left_safe, left_benefit = False, False, -1.0
        right_candidate, right_safe, right_benefit = False, False, -1.0

        # 左侧统一接入 MOBIL
        if scene["left_wp"] is not None and scene["left_allowed"] and need_bypass:
            left_candidate = True
            safe_info, benefit = evaluate_left_change(
                ego_speed, curr_front["dist"], curr_front["speed"],
                left_front["dist"], left_front["speed"], left_rear["dist"], left_rear["speed"]
            )
            left_safe = safe_info.get("safe", False)
            left_benefit = benefit

        # 右侧统一接入 MOBIL
        if scene["right_wp"] is not None and scene["right_allowed"] and need_bypass:
            right_candidate = True
            safe_info_r, benefit_r = evaluate_right_bypass(
                ego_speed, curr_front["dist"], curr_front["speed"],
                right_front["dist"], right_front["speed"], right_rear["dist"], right_rear["speed"]
            )
            right_safe = safe_info_r.get("safe", False)
            right_benefit = benefit_r

        # MOBIL 分数中已经扣除了阈值，因此只要净收益 > 0 即可变道
        choose_left = left_candidate and left_safe and (left_benefit > 0.0)
        choose_right = right_candidate and right_safe and (right_benefit > 0.0)

        # 状态机流转逻辑
        if self.state == FSMState.FOLLOW_LANE:
            self.target_lane_id = None
            if not need_bypass:
                self._reset_prepare()
                extra["decision"] = "stay_follow_no_need"
                return self.state, scene, extra

            # 倾向于优先超车道(左侧)
            if choose_left:
                if self.pending_direction not in (None, "left"): self._reset_prepare()
                if self.pending_direction is None: self._start_prepare("left", curr_front)
                self.prepare_count += 1
                if self.prepare_count >= self.stable_ticks:
                    self.state = FSMState.PREPARE_LANE_CHANGE_LEFT
                    self.prepare_confirm_count = 0
            elif choose_right:
                if self.pending_direction not in (None, "right"): self._reset_prepare()
                if self.pending_direction is None: self._start_prepare("right", curr_front)
                self.prepare_count += 1
                if self.prepare_count >= self.stable_ticks:
                    self.state = FSMState.PREPARE_RETURN_RIGHT
                    self.prepare_confirm_count = 0
                    self.right_change_is_bypass = True
            else:
                self._reset_prepare()
                extra["decision"] = "stay_follow_no_safe_gap"

        elif self.state == FSMState.PREPARE_LANE_CHANGE_LEFT:
            if not need_bypass or not choose_left:
                self.state = FSMState.FOLLOW_LANE
                self._reset_prepare()
            else:
                self.prepare_confirm_count += 1
                if self.prepare_confirm_count >= self.prepare_confirm_ticks:
                    self._reset_prepare()
                    self._enter_lane_change(scene, to_right_bypass=False)

        elif self.state == FSMState.LANE_CHANGE_LEFT:
            self.lane_change_ticks += 1
            if self._is_lane_change_finished(scene):
                self.state = FSMState.OVERTAKE_CRUISE
                self.lane_change_ticks = 0
                self.target_lane_id = None
            elif self.lane_change_ticks > self.lane_change_timeout:
                self.state = FSMState.FOLLOW_LANE
                self.lane_change_ticks = 0
                self._reset_prepare()
                self.target_lane_id = None

        elif self.state == FSMState.OVERTAKE_CRUISE:
            self.target_lane_id = None
            pass_margin = self._get_target_pass_margin(self.overtake_target_id)
            if scene["right_wp"] is not None and scene["right_allowed"]:
                if can_return_right(pass_margin, right_front["dist"], right_front["speed"], right_rear["dist"], right_rear["speed"], ego_speed):
                    self.state = FSMState.PREPARE_RETURN_RIGHT
                    self.prepare_confirm_count = 0
                    self.right_change_is_bypass = False

        elif self.state == FSMState.PREPARE_RETURN_RIGHT:
            if self.right_change_is_bypass:
                if not need_bypass or not choose_right:
                    self.state = FSMState.FOLLOW_LANE
                    self._reset_prepare()
                    self.right_change_is_bypass = False
                    self.target_lane_id = None
                else:
                    self.prepare_confirm_count += 1
                    if self.prepare_confirm_count >= self.prepare_confirm_ticks:
                        self._reset_prepare()
                        self._lock_target_lane_id(scene, "right")
                        self.state = FSMState.LANE_CHANGE_RIGHT
                        self.lane_change_ticks = 0
                        self.right_change_is_bypass = True
            else:
                if scene["right_wp"] is None or not scene["right_allowed"]:
                    self.state = FSMState.OVERTAKE_CRUISE
                    self.prepare_confirm_count = 0
                    self.target_lane_id = None
                else:
                    self.prepare_confirm_count += 1
                    if self.prepare_confirm_count >= self.prepare_confirm_ticks:
                        self._reset_prepare()
                        self._lock_target_lane_id(scene, "right")
                        self.state = FSMState.LANE_CHANGE_RIGHT
                        self.lane_change_ticks = 0
                        self.right_change_is_bypass = False

        elif self.state == FSMState.LANE_CHANGE_RIGHT:
            self.lane_change_ticks += 1
            if self._is_lane_change_finished(scene):
                if self.right_change_is_bypass:
                    self.state = FSMState.OVERTAKE_CRUISE
                else:
                    self.state = FSMState.FOLLOW_LANE
                    self.overtake_target_id = None
                self.right_change_is_bypass = False
                self.lane_change_ticks = 0
                self.target_lane_id = None
            elif self.lane_change_ticks > self.lane_change_timeout:
                if self.right_change_is_bypass:
                    self.state = FSMState.FOLLOW_LANE
                else:
                    self.state = FSMState.OVERTAKE_CRUISE
                self.lane_change_ticks = 0
                self.right_change_is_bypass = False
                self.target_lane_id = None

        elif self.state == FSMState.EMERGENCY_BRAKE:
            if emergency:
                self.emergency_release_count = 0
            else:
                self.emergency_release_count += 1
                if self.emergency_release_count >= self.emergency_release_ticks:
                    self.state = FSMState.FOLLOW_LANE
                    self._reset_prepare()
                    self.lane_change_ticks = 0
                    self.emergency_release_count = 0
                    self.target_lane_id = None

        return self.state, scene, extra