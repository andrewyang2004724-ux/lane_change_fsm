# decision/behavior_rules.py
import math
from config import TARGET_SPEED, RETURN_OVERTAKE_MARGIN
from decision.safety import is_safe_gap

NO_CAR_DIST = 999.0
NO_CAR_SPEED = 999.0

def get_idm_acceleration(v, v_lead, s, v0=TARGET_SPEED):
    """
    [新增] 基于 IDM 动力学模型的预估加速度计算
    为 MOBIL 效用函数提供精确的微观物理依据，彻底取代线性打分。
    """
    a_max = 3.0   # 车辆最大舒适加速度 (m/s^2) - 提升至更激进的值
    b = 2.0       # 车辆舒适减速度 (m/s^2)
    delta = 4.0   # 速度指数
    s0 = 4.0      # 最小安全间距
    T = 1.0       # 期望车头时距

    if s >= NO_CAR_DIST:
        s_star = 0.0
        term2 = 0.0
    else:
        delta_v = v - v_lead
        s_star = s0 + v * T + (v * delta_v) / (2.0 * math.sqrt(a_max * b))
        s_star = max(s_star, s0)
        term2 = (s_star / max(s, 0.1)) ** 2

    term1 = (v / max(v0, 0.1)) ** delta
    return a_max * (1.0 - term1 - term2)

def mobil_lane_change_benefit(ego_speed, curr_front_dist, curr_front_speed,
                              tgt_front_dist, tgt_front_speed,
                              tgt_rear_dist, tgt_rear_speed):
    """
    [新增] 经典 MOBIL 换道效用评估
    返回换道的净收益 (加速度的绝对提升值 m/s^2)
    """
    curr_front_speed = curr_front_speed if curr_front_speed < NO_CAR_SPEED else TARGET_SPEED
    tgt_front_speed = tgt_front_speed if tgt_front_speed < NO_CAR_SPEED else TARGET_SPEED
    tgt_rear_speed = tgt_rear_speed if tgt_rear_speed < NO_CAR_SPEED else ego_speed

    # 1. 自车预期加速度评估 (利己)
    a_ego_curr = get_idm_acceleration(ego_speed, curr_front_speed, curr_front_dist)
    a_ego_tgt = get_idm_acceleration(ego_speed, tgt_front_speed, tgt_front_dist)

    # 2. 目标车道后车预期加速度评估 (利他博弈)
    # 换道前：目标车道后车原本跟它的前车
    dist_rear_to_tgt_front = tgt_rear_dist + tgt_front_dist
    if dist_rear_to_tgt_front >= NO_CAR_DIST: dist_rear_to_tgt_front = NO_CAR_DIST
    a_rear_curr = get_idm_acceleration(tgt_rear_speed, tgt_front_speed, dist_rear_to_tgt_front)
    
    # 换道后：目标车道后车变成了跟自车
    a_rear_tgt = get_idm_acceleration(tgt_rear_speed, ego_speed, tgt_rear_dist)

    # 3. MOBIL 核心公式
    p = 0.25 # 礼貌因子 (0~1之间。越小越激进加塞，越大越照顾后车感受)
    ego_gain = a_ego_tgt - a_ego_curr
    rear_gain = a_rear_tgt - a_rear_curr

    # 仅当后车受损时才计算惩罚，后车加速不计入额外收益
    if rear_gain > 0: rear_gain = 0.0

    mobil_score = ego_gain + p * rear_gain

    # 4. 换道动作本身施加的摩擦阻力阈值，防止在两车道匀速时左右横跳
    a_th = 0.15
    return float(mobil_score - a_th)

def should_overtake(ego_speed, curr_front_dist, curr_front_speed):
    """
    更新：基于预期减速度来触发换道意图，极其灵敏拟人
    """
    if curr_front_dist >= NO_CAR_DIST:
        return False
    # 如果 IDM 预判我即将被迫踩刹车 (加速度 < -0.2)，且距离适中，立刻萌生换道想法
    a_ego_curr = get_idm_acceleration(ego_speed, curr_front_speed, curr_front_dist)
    return bool(a_ego_curr < -0.2 and curr_front_dist < 45.0)

def emergency_needed(curr_front_dist, ego_speed, curr_front_speed):
    closing_speed = max(0.0, ego_speed - curr_front_speed)
    if closing_speed > 0:
        ttc = curr_front_dist / closing_speed
    else:
        ttc = 999.0
    return (ttc < 2.5), ttc

def evaluate_change(ego_speed, curr_front_dist, curr_front_speed,
                    tgt_front_dist, tgt_front_speed,
                    tgt_rear_dist, tgt_rear_speed):
    """底层通用的换道评估，左右变道逻辑完全收敛统一"""
    front_info = {"dist": tgt_front_dist, "speed": tgt_front_speed} if tgt_front_dist < NO_CAR_DIST else None
    rear_info = {"dist": tgt_rear_dist, "speed": tgt_rear_speed} if tgt_rear_dist < NO_CAR_DIST else None

    # 第一关：绝对物理安全关卡 (调用 safety.py 中的 TTC 判断)
    safe, details = is_safe_gap(ego_speed, front_info, rear_info)

    # 第二关：MOBIL 收益评估
    benefit = mobil_lane_change_benefit(
        ego_speed, curr_front_dist, curr_front_speed,
        tgt_front_dist, tgt_front_speed,
        tgt_rear_dist, tgt_rear_speed
    )
    return {"safe": safe, "details": details}, benefit

def evaluate_left_change(*args):
    return evaluate_change(*args)

def evaluate_right_bypass(*args):
    safe_info, benefit = evaluate_change(*args)
    # 交通规则中通常鼓励左侧超车，因此给右侧绕行施加 0.1 的轻微惩罚
    return safe_info, benefit - 0.10

def can_return_right(pass_margin, right_front_dist, right_front_speed, right_rear_dist, right_rear_speed, ego_speed):
    front_info = {"dist": right_front_dist, "speed": right_front_speed} if right_front_dist < NO_CAR_DIST else None
    rear_info = {"dist": right_rear_dist, "speed": right_rear_speed} if right_rear_dist < NO_CAR_DIST else None

    safe, details = is_safe_gap(ego_speed, front_info, rear_info)
    enough_margin = pass_margin > RETURN_OVERTAKE_MARGIN
    return bool(safe and enough_margin)