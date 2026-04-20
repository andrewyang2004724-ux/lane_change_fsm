# decision/safety.py
import math

def speed_norm(velocity):
    """
    计算 CARLA 速度向量的标量值 (m/s)
    """
    if velocity is None:
        return 0.0
    return math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

def is_safe_gap(ego_speed, front_info, rear_info):
    """
    核心安全间隙评估：基于动态 TTC (碰撞时间) 与 THW (车头时距)
    彻底取代死板的距离阈值，让换道在低速时果断，在高速时敬畏物理极限。
    """
    # ==========================
    # 核心安全阈值参数配置
    # ==========================
    MIN_SAFE_DIST = 4.0   # 极限物理最小安全缓冲 (米) - 防止低速下强行塞车
    TTC_THRESHOLD = 3.0   # 最小碰撞时间阈值 (秒) - 预判相对速度带来的危险
    TIME_HEADWAY = 1.0    # 期望车头时距 (秒) - 留出人类反应时间的纵向空间

    # 1. 评估目标车道前方车辆 (Front Vehicle)
    front_safe = True
    if front_info and front_info["dist"] < 999.0:
        df = front_info["dist"]
        vf = front_info["speed"]
        
        # a. 绝对底线：无论如何不能撞上
        if df < MIN_SAFE_DIST:
            front_safe = False
        else:
            # b. 基于自车速度的车头时距约束 (随自车速度动态拉长)
            if df < ego_speed * TIME_HEADWAY:
                front_safe = False
                
            # c. TTC 碰撞时间预测 (只在前车比我慢的时候才可能追尾)
            v_rel_front = ego_speed - vf
            if v_rel_front > 0:
                ttc_f = df / v_rel_front
                if ttc_f < TTC_THRESHOLD:
                    front_safe = False

    # 2. 评估目标车道后方车辆 (Rear Vehicle)
    rear_safe = True
    if rear_info and rear_info["dist"] < 999.0:
        dr = rear_info["dist"]
        vr = rear_info["speed"]
        
        # a. 绝对底线
        if dr < MIN_SAFE_DIST:
            rear_safe = False
        else:
            # b. 基于后车速度的车头时距约束 (防止我们强行加塞导致后车急刹)
            if dr < vr * TIME_HEADWAY:
                rear_safe = False
                
            # c. TTC 碰撞时间预测 (只在后车比我快的时候才可能被追尾)
            v_rel_rear = vr - ego_speed
            if v_rel_rear > 0:
                ttc_r = dr / v_rel_rear
                if ttc_r < TTC_THRESHOLD:
                    rear_safe = False

    # 返回综合判定结果与详情（方便你的 logger 和 metrics 记录）
    return front_safe and rear_safe, {
        "front_safe": front_safe,
        "rear_safe": rear_safe
    }

def check_lane_change_safety(ego_speed, scene, side):
    """
    状态机调用入口：根据 scene 中构建好的相对车辆信息，评估指定方向的换道安全性。
    """
    if side == "left":
        # 提取我们在 scene_builder.py 中算好的精准 Bumper-to-Bumper 距离
        front_info = scene.get("left_front")
        rear_info = scene.get("left_rear")
    elif side == "right":
        front_info = scene.get("right_front")
        rear_info = scene.get("right_rear")
    else:
        return False, {"error": "Invalid side"}

    is_safe, details = is_safe_gap(ego_speed, front_info, rear_info)
    return is_safe, details