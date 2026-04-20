# scripts/spawn_traffic.py
import random
import carla
from config import EGO_ROLE_NAME, SCENARIO_MODE, NUM_TRAFFIC, TRAFFIC_MANAGER_PORT

def _find_good_highway_waypoint(world_map):
    """
    智能寻找一个高速公路的中间车道，确保左右都有车道，方便做变道实验。
    """
    spawn_points = world_map.get_spawn_points()
    for sp in spawn_points:
        wp = world_map.get_waypoint(sp.location)
        if wp.lane_type == carla.LaneType.Driving:
            left_wp = wp.get_left_lane()
            right_wp = wp.get_right_lane()
            # 找到一个中间车道（左右都有同向车道）
            if left_wp is not None and right_wp is not None:
                if left_wp.lane_id * wp.lane_id > 0 and right_wp.lane_id * wp.lane_id > 0:
                    return wp
    # 兜底：随便返回一个出生点
    return world_map.get_waypoint(spawn_points[0].location)

def _spawn_actor_at_wp(world, blueprint, wp, z_offset=0.5):
    """在指定 waypoint 处生成车辆"""
    transform = wp.transform
    transform.location.z += z_offset
    return world.spawn_actor(blueprint, transform)

def spawn_scenario(world, client):
    """
    统一的场景生成器。根据 config 中的 SCENARIO_MODE 布置车辆与车流速度。
    """
    world_map = world.get_map()
    bp_lib = world.get_blueprint_library()
    
    ego_bp = bp_lib.find('vehicle.audi.a2')
    ego_bp.set_attribute('role_name', EGO_ROLE_NAME)
    
    npc_bp = bp_lib.filter('vehicle.*')[0] # 选个默认的NPC车型
    
    tm = client.get_trafficmanager(TRAFFIC_MANAGER_PORT)
    tm.set_synchronous_mode(True)
    
    ego = None
    traffic_actors = []
    
    # 获取一个绝佳的起点
    base_wp = _find_good_highway_waypoint(world_map)
    
    if SCENARIO_MODE == "Overtake_Left":
        # ========================================================
        # 场景一：完美的左侧超车
        # 自车正常行驶，前方 40 米有一辆极慢的龟速车，左侧车道空旷
        # ========================================================
        ego = _spawn_actor_at_wp(world, ego_bp, base_wp)
        
        # 前方慢车
        wp_front = base_wp.next(40.0)[0]
        front_car = _spawn_actor_at_wp(world, npc_bp, wp_front)
        traffic_actors.append(front_car)
        
        # 严格限制NPC：禁止变道，强制低速(比限速低 50%)
        front_car.set_autopilot(True, tm.get_port())
        tm.auto_lane_change(front_car, False)
        tm.vehicle_percentage_speed_difference(front_car, 50.0)

    elif SCENARIO_MODE == "Blocked_Wait":
        # ========================================================
        # 场景二：左侧封堵博弈
        # 前方有慢车，你想左变道，但左侧正好有一辆车与你并排甚至更快
        # 预期 FSM 行为：保持跟车(FOLLOW)，直到左侧车开走腾出空间，再执行超车
        # ========================================================
        ego = _spawn_actor_at_wp(world, ego_bp, base_wp)
        
        # 前方慢车 (40m外)
        wp_front = base_wp.next(40.0)[0]
        front_car = _spawn_actor_at_wp(world, npc_bp, wp_front)
        traffic_actors.append(front_car)
        front_car.set_autopilot(True, tm.get_port())
        tm.auto_lane_change(front_car, False)
        tm.vehicle_percentage_speed_difference(front_car, 50.0) # 慢速
        
        # 左侧封堵车 (10m外)
        wp_left = base_wp.get_left_lane().next(10.0)[0]
        block_car = _spawn_actor_at_wp(world, npc_bp, wp_left)
        traffic_actors.append(block_car)
        block_car.set_autopilot(True, tm.get_port())
        tm.auto_lane_change(block_car, False)
        tm.vehicle_percentage_speed_difference(block_car, 10.0) # 速度比自车快一点点

    else:
        # ========================================================
        # 场景三："Random" (随机生成大量背景车，用于压力测试)
        # ========================================================
        ego = _spawn_actor_at_wp(world, ego_bp, base_wp)
        spawn_points = world_map.get_spawn_points()
        random.shuffle(spawn_points)
        
        for sp in spawn_points[:NUM_TRAFFIC]:
            if sp.location.distance(base_wp.transform.location) < 15.0:
                continue # 不要生成在自车头上
            try:
                npc = world.spawn_actor(random.choice(bp_lib.filter('vehicle.*')), sp)
                traffic_actors.append(npc)
                npc.set_autopilot(True, tm.get_port())
                # 关闭它们的变道，让它们充当移动路障，避免干扰你的变道逻辑
                tm.auto_lane_change(npc, False) 
            except Exception:
                pass

    return ego, traffic_actors, tm