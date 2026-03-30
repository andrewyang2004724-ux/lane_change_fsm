# decision/behavior_rules.py
from config import (
    TARGET_SPEED,
    TTC_FRONT_MIN,
    TTC_REAR_MIN,
    TTC_EMERGENCY,
    SPEED_DIFF_THRESHOLD,
    RETURN_OVERTAKE_MARGIN,
)
from decision.safety import is_safe_to_change, compute_ttc

NO_CAR_DIST = 999.0
NO_CAR_SPEED = 999.0

def _norm_dist(d, default=80.0):
    if d is None or d >= NO_CAR_DIST:
        return float(default)
    return float(d)

def _norm_speed(v, default=TARGET_SPEED):
    if v is None or v >= NO_CAR_SPEED:
        return float(default)
    return float(v)

def lane_change_benefit(curr_front_dist, curr_front_speed, tgt_front_dist, tgt_front_speed):
    """
    变道收益评估：
    - 目标车道前车更快：收益增加
    - 目标车道更空：收益增加
    - 当前车道受阻明显：收益增加
    - 固定变道成本：抑制频繁无效变道
    """
    curr_front_dist = _norm_dist(curr_front_dist, 80.0)
    tgt_front_dist = _norm_dist(tgt_front_dist, 80.0)
    curr_front_speed = _norm_speed(curr_front_speed, TARGET_SPEED)
    tgt_front_speed = _norm_speed(tgt_front_speed, TARGET_SPEED)

    speed_gain = tgt_front_speed - curr_front_speed
    dist_gain = tgt_front_dist - curr_front_dist

    score = 0.0

    # 速度收益：保留，但降低线性激进程度
    score += 0.45 * speed_gain

    # 距离收益：目标车道前方更空更好
    score += 0.30 * max(-1.0, min(1.5, dist_gain / 25.0))

    # 当前受阻补偿
    if curr_front_dist < 18.0:
        score += 0.60
    elif curr_front_dist < 28.0:
        score += 0.30
    elif curr_front_dist < 38.0:
        score += 0.10

    # 当前前车明显更慢时，再提高收益
    if curr_front_speed < TARGET_SPEED - 2.0:
        score += 0.20

    # 若目标车道非常空，再略加收益
    if tgt_front_dist > curr_front_dist + 15.0:
        score += 0.25

    # 固定变道成本，避免轻微优势也频繁切
    score -= 0.20

    return float(score)

def should_overtake(ego_speed, curr_front_dist, curr_front_speed):
    """
    是否需要准备超车：
    比原版更关注 ego 与前车的相对关系，而不只是前车相对目标速度。
    """
    if curr_front_dist >= NO_CAR_DIST or curr_front_speed >= NO_CAR_SPEED:
        return False

    rel_speed = ego_speed - curr_front_speed
    target_gap = TARGET_SPEED - curr_front_speed

    blocked = curr_front_dist < 35.0
    front_is_slow = target_gap > max(1.5, SPEED_DIFF_THRESHOLD - 0.5)
    ego_is_catching = rel_speed > 1.0
    ego_not_too_fast = ego_speed <= TARGET_SPEED + 4.0

    # 近距离且前车明显慢，或者 ego 正在明显逼近
    return bool(blocked and ego_not_too_fast and (front_is_slow or ego_is_catching))

def emergency_needed(curr_front_dist, ego_speed, curr_front_speed):
    closing_speed = max(0.0, ego_speed - curr_front_speed)
    ttc = compute_ttc(curr_front_dist, closing_speed)
    return (ttc < TTC_EMERGENCY), ttc

def evaluate_left_change(
    ego_speed,
    curr_front_dist,
    curr_front_speed,
    left_front_dist,
    left_front_speed,
    left_rear_dist,
    left_rear_speed
):
    safe_info = is_safe_to_change(
        ego_speed,
        left_front_dist,
        left_front_speed,
        left_rear_dist,
        left_rear_speed,
        TTC_FRONT_MIN,
        TTC_REAR_MIN
    )

    benefit = lane_change_benefit(
        curr_front_dist,
        curr_front_speed,
        left_front_dist,
        left_front_speed
    )

    # 左车道几乎完全空，适度提高收益
    if left_front_dist >= NO_CAR_DIST and left_rear_dist >= NO_CAR_DIST:
        benefit += 0.35

    return safe_info, float(benefit)

def evaluate_right_change(
    ego_speed,
    right_front_dist,
    right_front_speed,
    right_rear_dist,
    right_rear_speed
):
    safe_info = is_safe_to_change(
        ego_speed,
        right_front_dist,
        right_front_speed,
        right_rear_dist,
        right_rear_speed,
        TTC_FRONT_MIN,
        TTC_REAR_MIN
    )
    return safe_info

def evaluate_right_bypass(
    ego_speed,
    curr_front_dist,
    curr_front_speed,
    right_front_dist,
    right_front_speed,
    right_rear_dist,
    right_rear_speed
):
    """
    右侧绕行评估：
    返回 (safe_info, benefit)
    """
    safe_info = evaluate_right_change(
        ego_speed,
        right_front_dist,
        right_front_speed,
        right_rear_dist,
        right_rear_speed
    )

    benefit = lane_change_benefit(
        curr_front_dist,
        curr_front_speed,
        right_front_dist,
        right_front_speed
    )

    # 右绕行相对左超车略保守一点，给一点额外成本
    benefit -= 0.10

    # 若右后车较快，再削弱收益
    if right_rear_speed < NO_CAR_SPEED:
        rear_rel_speed = right_rear_speed - ego_speed
        if rear_rel_speed > 3.0:
            benefit -= 0.30

    return safe_info, float(benefit)

def can_return_right(
    pass_margin,
    right_front_dist,
    right_front_speed,
    right_rear_dist,
    right_rear_speed,
    ego_speed
):
    """
    回右条件：
    1. 已经超过被超车辆足够距离
    2. 右侧安全
    3. 右前不能太近
    """
    safe_info = evaluate_right_change(
        ego_speed,
        right_front_dist,
        right_front_speed,
        right_rear_dist,
        right_rear_speed
    )

    enough_margin = pass_margin > RETURN_OVERTAKE_MARGIN

    # 比原版略保守，减少刚回去就受限
    right_front_ok = (right_front_dist >= 15.0) or (right_front_dist >= NO_CAR_DIST)

    return bool(enough_margin and safe_info["safe"] and right_front_ok)
