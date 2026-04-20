# scripts/spawn_traffic.py
import carla
import random
from config import EGO_ROLE_NAME, SCENARIO_MODE, NUM_TRAFFIC, TRAFFIC_MANAGER_PORT, EGO_SPAWN_POINT_INDEX

def spawn_scenario(world, client):
    """
    根据配置的索引生成自车，并围绕该点布置实验场景。
    """
    world_map = world.get_map()
    bp_lib = world.get_blueprint_library()
    spawn_points = world_map.get_spawn_points()
    
    # 1. 验证索引有效性
    if EGO_SPAWN_POINT_INDEX >= len(spawn_points):
        print(f"Warning: Index {EGO_SPAWN_POINT_INDEX} out of range, using index 0")
        ego_spawn_pt = spawn_points[0]
    else:
        ego_spawn_pt = spawn_points[EGO_SPAWN_POINT_INDEX]
    
    # 打印当前索引的位置，方便你在 Google Map 或本地坐标系中对比
    loc = ego_spawn_pt.location
    print(f">>> Ego spawning at Index [{EGO_SPAWN_POINT_INDEX}]: x={loc.x:.1f}, y={loc.y:.1f}, z={loc.z:.1f}")

    # 2. 生成自车
    ego_bp = bp_lib.find('vehicle.audi.a2')
    ego_bp.set_attribute('role_name', EGO_ROLE_NAME)
    ego = world.spawn_actor(ego_bp, ego_spawn_pt)
    
    traffic_actors = []
    tm = client.get_trafficmanager(TRAFFIC_MANAGER_PORT)
    tm.set_synchronous_mode(True)
    
    # 获取自车起始点的 Waypoint，用于相对定位其他车
    base_wp = world_map.get_waypoint(loc)
    npc_bp = bp_lib.filter('vehicle.*')[0]

    # 3. 根据场景布置车辆（使用直接获取全局信息，舍弃感知误差）
    if SCENARIO_MODE == "Overtake_Left":
        # 前方 40m 慢车
        wp_f = base_wp.next(40.0)[0]
        f_car = world.spawn_actor(npc_bp, wp_f.transform)
        traffic_actors.append(f_car)
        f_car.set_autopilot(True, tm.get_port())
        tm.vehicle_percentage_speed_difference(f_car, 60.0) # 极慢
        tm.auto_lane_change(f_car, False)

    elif SCENARIO_MODE == "Blocked_Wait":
        # 前方慢车 + 左侧平行封堵
        wp_f = base_wp.next(35.0)[0]
        wp_l = base_wp.get_left_lane().next(5.0)[0] # 左侧近距离封堵
        
        f_car = world.spawn_actor(npc_bp, wp_f.transform)
        l_car = world.spawn_actor(npc_bp, wp_l.transform)
        
        for c in [f_car, l_car]:
            traffic_actors.append(c)
            c.set_autopilot(True, tm.get_port())
            tm.auto_lane_change(c, False)
            
        tm.vehicle_percentage_speed_difference(f_car, 50.0)
        tm.vehicle_percentage_speed_difference(l_car, 10.0)
        
    else:
        # ========================================================
        # 场景三："Random" (随机生成大量背景车，用于压力测试)
        # ========================================================
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