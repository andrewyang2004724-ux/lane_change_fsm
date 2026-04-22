# scripts/spawn_traffic.py
import carla
import random
from config import (
    EGO_ROLE_NAME, SCENARIO_MODE, NUM_TRAFFIC,
    TRAFFIC_MANAGER_PORT, EGO_SPAWN_POINT_INDEX
)


def _normalize_scenario(mode: str) -> str:
    """
    兼容旧模式命名与新模板命名。
    """
    mapping = {
        "Overtake_Left": "FOLLOW_SLOW_LEAD",
        "Blocked_Wait": "LEFT_BLOCKED",
        "Random": "FREE_FLOW",
    }
    return mapping.get(mode, mode)


def _build_config(scenario_mode: str, seed: int, num_traffic: int):
    """
    统一场景参数（可复现）。
    """
    mode = _normalize_scenario(scenario_mode)

    cfg = {
        "scenario_mode": mode,
        "seed": int(seed),
        "num_traffic": int(num_traffic),

        # 通用参数
        "spawn_front_range_m": 220.0,
        "spawn_back_range_m": 70.0,
        "spawn_step_m": 10.0,
        "spawn_gap_min_m": 16.0,
        "speed_min_kmh": 65.0,
        "speed_max_kmh": 125.0,

        # 车道采样权重（ego/left/right）
        "lane_bias": {"ego": 0.45, "left": 0.35, "right": 0.20},

        # 特定场景参数
        "slow_lead_enable": False,
        "slow_lead_distance_m": 35.0,
        "slow_lead_speed_kmh": 65.0,
        "left_block_enable": False,
        "left_block_offsets_m": [-10.0, 8.0, 28.0, 50.0],
        "density_factor": 0.35,  # 候选点填充率
    }

    if mode == "FREE_FLOW":
        cfg["density_factor"] = 0.30
        cfg["speed_mean_kmh"] = 102.0
        cfg["speed_std_kmh"] = 8.0
    elif mode == "FOLLOW_SLOW_LEAD":
        cfg["density_factor"] = 0.45
        cfg["speed_mean_kmh"] = 96.0
        cfg["speed_std_kmh"] = 7.0
        cfg["slow_lead_enable"] = True
        cfg["slow_lead_distance_m"] = 32.0
        cfg["slow_lead_speed_kmh"] = 62.0
    elif mode == "LEFT_BLOCKED":
        cfg["density_factor"] = 0.55
        cfg["speed_mean_kmh"] = 92.0
        cfg["speed_std_kmh"] = 6.0
        cfg["slow_lead_enable"] = True
        cfg["slow_lead_distance_m"] = 35.0
        cfg["slow_lead_speed_kmh"] = 68.0
        cfg["left_block_enable"] = True
    elif mode == "DENSE_TRAFFIC":
        cfg["density_factor"] = 0.75
        cfg["speed_mean_kmh"] = 88.0
        cfg["speed_std_kmh"] = 6.0
        cfg["num_traffic"] = max(int(num_traffic), 60)
        cfg["lane_bias"] = {"ego": 0.40, "left": 0.35, "right": 0.25}
    else:
        # 未知模式回退
        cfg["scenario_mode"] = "FREE_FLOW"
        cfg["density_factor"] = 0.30
        cfg["speed_mean_kmh"] = 100.0
        cfg["speed_std_kmh"] = 8.0

    return cfg


def _is_valid_driving_lane(wp):
    return (wp is not None) and (wp.lane_type == carla.LaneType.Driving)


def _shift_waypoint(wp, distance_m):
    """
    沿当前车道前后平移 waypoint。distance_m > 0 前进；< 0 后退。
    """
    if wp is None:
        return None
    
    # 处理距离为 0 的特殊情况（CARLA API 要求 distance > 0）
    if abs(distance_m) < 1e-3:
        return wp
    
    if distance_m > 0:
        nxt = wp.next(float(distance_m))
    else:
        nxt = wp.previous(float(-distance_m))
    if not nxt:
        return None
    return nxt[0]


def _collect_lane_refs(base_wp):
    lanes = {"ego": None, "left": None, "right": None}

    if _is_valid_driving_lane(base_wp):
        lanes["ego"] = base_wp

    l = base_wp.get_left_lane() if base_wp is not None else None
    if _is_valid_driving_lane(l):
        lanes["left"] = l

    r = base_wp.get_right_lane() if base_wp is not None else None
    if _is_valid_driving_lane(r):
        lanes["right"] = r

    return lanes


def _collect_candidates(lane_wp, back_range, front_range, step):
    if lane_wp is None:
        return []
    cands = []
    d = -float(back_range)
    while d <= float(front_range):
        wp = _shift_waypoint(lane_wp, d)
        if wp is not None:
            cands.append((d, wp))
        d += float(step)
    return cands


def _dist(a_loc, b_loc):
    return a_loc.distance(b_loc)


def _can_spawn_here(wp, occupied_locs, min_gap_m):
    loc = wp.transform.location
    for p in occupied_locs:
        if _dist(loc, p) < min_gap_m:
            return False
    return True


def _choose_vehicle_bp(bp_lib, rng):
    bps = bp_lib.filter("vehicle.*")
    # 优先四轮车，避免两轮车影响稳定性
    filtered = []
    for bp in bps:
        try:
            if bp.has_attribute("number_of_wheels") and int(bp.get_attribute("number_of_wheels")) == 4:
                filtered.append(bp)
        except Exception:
            continue
    if not filtered:
        filtered = bps
    return rng.choice(filtered)


def _set_vehicle_behavior(tm, actor, target_speed_kmh, allow_lane_change=False):
    actor.set_autopilot(True, tm.get_port())
    
    # 获取车辆所处车道，防止外侧慢车道的车乱变道影响高速秩序
    world = actor.get_world()
    wp = world.get_map().get_waypoint(actor.get_location())
    abs_lane = abs(wp.lane_id) if wp else 2
    
    # 强制让慢车道(3, 4道)的车辆禁止变道
    final_allow_lc = allow_lane_change if abs_lane <= 2 else False
    tm.auto_lane_change(actor, final_allow_lc)

    speed_limit = actor.get_speed_limit()
    # 如果地图没有配置限速，提供一个默认的高速基准限速 (Town04 通常适用)
    if speed_limit <= 1.0:
        speed_limit = 130.0

    # TM百分比计算：正数表示比限速慢，负数表示比限速快(超速)
    perc = 100.0 * (speed_limit - target_speed_kmh) / max(speed_limit, 1e-3)
    # 放开负数下限，确保超车道车辆可以达到极速
    perc = max(-80.0, min(95.0, perc)) 
    tm.vehicle_percentage_speed_difference(actor, perc)


def _sample_speed_kmh(rng, mean, std, vmin, vmax):
    val = rng.gauss(mean, std)
    return max(vmin, min(vmax, val))


def spawn_scenario(world, client, scenario_mode=None, seed=42, num_traffic=None):
    """
    返回: ego, traffic_actors, tm, run_cfg
    """
    scenario_mode = scenario_mode if scenario_mode is not None else SCENARIO_MODE
    num_traffic = NUM_TRAFFIC if num_traffic is None else num_traffic

    cfg = _build_config(scenario_mode=scenario_mode, seed=seed, num_traffic=num_traffic)
    rng = random.Random(cfg["seed"])

    world_map = world.get_map()
    bp_lib = world.get_blueprint_library()
    spawn_points = world_map.get_spawn_points()

    # 1) 自车生成点
    if EGO_SPAWN_POINT_INDEX >= len(spawn_points):
        print(f"[spawn] Warning: Index {EGO_SPAWN_POINT_INDEX} out of range, fallback to 0")
        ego_spawn_pt = spawn_points[0]
    else:
        ego_spawn_pt = spawn_points[EGO_SPAWN_POINT_INDEX]

    loc = ego_spawn_pt.location
    print(f">>> Ego spawning @Index[{EGO_SPAWN_POINT_INDEX}] x={loc.x:.1f}, y={loc.y:.1f}, z={loc.z:.1f}")

    # 2) 生成自车
    ego_bp = bp_lib.find("vehicle.audi.a2")
    ego_bp.set_attribute("role_name", EGO_ROLE_NAME)
    ego = world.spawn_actor(ego_bp, ego_spawn_pt)

    # 3) TM 配置
    tm = client.get_trafficmanager(TRAFFIC_MANAGER_PORT)
    tm.set_synchronous_mode(True)
    try:
        tm.set_global_distance_to_leading_vehicle(2.5)
    except Exception:
        pass

    # 候选车道
    base_wp = world_map.get_waypoint(loc)
    lanes = _collect_lane_refs(base_wp)
    if lanes["ego"] is None:
        # 极端情况 fallback
        lanes["ego"] = base_wp

    traffic_actors = []
    occupied = [ego.get_location()]

    def try_spawn_on_wp(wp, target_speed_kmh, min_gap=None, allow_lane_change=False):
        if wp is None:
            return None
        if min_gap is None:
            min_gap = cfg["spawn_gap_min_m"]
        if not _can_spawn_here(wp, occupied, min_gap):
            return None
        bp = _choose_vehicle_bp(bp_lib, rng)
        try:
            npc = world.spawn_actor(bp, wp.transform)
            _set_vehicle_behavior(tm, npc, target_speed_kmh, allow_lane_change=allow_lane_change)
            traffic_actors.append(npc)
            occupied.append(npc.get_location())
            return npc
        except Exception:
            return None

    # 4) 先放置“触发逻辑车”
    # 4.1 前方慢车（用于触发超车）
    if cfg["slow_lead_enable"] and lanes["ego"] is not None:
        slow_wp = _shift_waypoint(lanes["ego"], cfg["slow_lead_distance_m"])
        if slow_wp is not None:
            try_spawn_on_wp(
                slow_wp,
                target_speed_kmh=cfg["slow_lead_speed_kmh"],
                min_gap=max(12.0, cfg["spawn_gap_min_m"] - 2.0),
                allow_lane_change=False
            )

    # 4.2 左道封堵（用于测试等待窗口）
    if cfg["left_block_enable"] and lanes["left"] is not None:
        for d in cfg["left_block_offsets_m"]:
            blk_wp = _shift_waypoint(lanes["left"], d)
            if blk_wp is not None:
                spd = _sample_speed_kmh(
                    rng,
                    mean=cfg["speed_mean_kmh"] - 8.0,
                    std=max(2.0, cfg["speed_std_kmh"] * 0.7),
                    vmin=cfg["speed_min_kmh"],
                    vmax=cfg["speed_max_kmh"]
                )
                try_spawn_on_wp(blk_wp, target_speed_kmh=spd, min_gap=12.0, allow_lane_change=False)

    # 5) 背景流量：围绕 ego 的局部高速路段生成（避免“全图稀疏”）
    lane_candidates = {
        "ego": _collect_candidates(lanes["ego"], cfg["spawn_back_range_m"], cfg["spawn_front_range_m"], cfg["spawn_step_m"]),
        "left": _collect_candidates(lanes["left"], cfg["spawn_back_range_m"], cfg["spawn_front_range_m"], cfg["spawn_step_m"]) if lanes["left"] else [],
        "right": _collect_candidates(lanes["right"], cfg["spawn_back_range_m"], cfg["spawn_front_range_m"], cfg["spawn_step_m"]) if lanes["right"] else [],
    }

    # 合并候选并去重（按位置粗粒度）
    pool = []
    for lane_name, lst in lane_candidates.items():
        for offset, wp in lst:
            pool.append((lane_name, offset, wp))

    # 目标数量：受 density_factor 与 num_traffic 双约束
    desired = min(int(cfg["num_traffic"]), int(len(pool) * cfg["density_factor"]))
    desired = max(desired, len(traffic_actors))

    # 按车道权重采样
    lane_keys = ["ego", "left", "right"]
    lane_weights = [cfg["lane_bias"].get(k, 0.0) for k in lane_keys]
    lane_buckets = {k: [] for k in lane_keys}
    for lane_name, offset, wp in pool:
        lane_buckets[lane_name].append((offset, wp))

    # 各 lane 内随机
    for k in lane_keys:
        rng.shuffle(lane_buckets[k])

    spawn_trials = 0
    max_trials = max(200, desired * 8)

    while len(traffic_actors) < desired and spawn_trials < max_trials:
        spawn_trials += 1
        lane_name = rng.choices(lane_keys, weights=lane_weights, k=1)[0]
        bucket = lane_buckets.get(lane_name, [])
        if not bucket:
            continue

        offset, wp = bucket.pop()  # 取一个候选
        # 避免在 ego 极近处刷车
        if abs(offset) < 15.0:
            continue

        # ==========================================
        # [修改点] 根据车道动态分配车辆速度，贴合实际中国4车道高速
        # 1道(最左侧) -> 超车道; 4道(最右侧) -> 慢车道
        # ==========================================
        abs_lane = abs(wp.lane_id) if wp is not None else 2
        if abs_lane == 1:
            speed = rng.uniform(105.0, 115.0)  # 超车道: 最快
        elif abs_lane == 2:
            speed = rng.uniform(90.0, 105.0)   # 快车道: 较快
        elif abs_lane == 3:
            speed = rng.uniform(80.0, 95.0)    # 行车道: 中速
        else:
            speed = rng.uniform(65.0, 80.0)    # 慢车/大客车道: 最慢

        # 默认关闭背景车主动变道，让实验更“可控”
        try_spawn_on_wp(wp, target_speed_kmh=speed, min_gap=cfg["spawn_gap_min_m"], allow_lane_change=False)

    print(f"[spawn] mode={cfg['scenario_mode']} seed={cfg['seed']} traffic={len(traffic_actors)}")

    run_cfg = {
        "scenario_mode": cfg["scenario_mode"],
        "seed": cfg["seed"],
        "num_traffic_requested": cfg["num_traffic"],
        "num_traffic_spawned": len(traffic_actors),
        "density_factor": cfg["density_factor"],
        "slow_lead_enable": cfg["slow_lead_enable"],
        "left_block_enable": cfg["left_block_enable"],
        "speed_mean_kmh": cfg["speed_mean_kmh"],
        "speed_std_kmh": cfg["speed_std_kmh"],
        "spawn_gap_min_m": cfg["spawn_gap_min_m"],
        "lane_bias": cfg["lane_bias"],
    }

    return ego, traffic_actors, tm, run_cfg