# control/path_generator.py
import math

def cubic_lane_change_points(start_wp, target_wp, length=40.0, ds=2.0):
    """
    生成从当前车道到目标车道的平滑参考点
    这里只返回 waypoint 列表附近的空间点，后续可交给控制器跟踪
    """
    points = []

    start_tf = start_wp.transform
    target_tf = target_wp.transform

    x0 = start_tf.location.x
    y0 = start_tf.location.y

    x1 = target_tf.transform.location.x if hasattr(target_tf, 'transform') else target_tf.location.x
    y1 = target_tf.transform.location.y if hasattr(target_tf, 'transform') else target_tf.location.y

    # 实际工程中更推荐 Frenet 生成；这里给毕设做简化
    n = int(length / ds)
    for i in range(n + 1):
        s = i * ds
        u = s / length
        offset = 3*u*u - 2*u*u*u
        ref_wp = start_wp.next(s)[0]
        tar_wp = target_wp.next(s)[0]
        x = ref_wp.transform.location.x * (1 - offset) + tar_wp.transform.location.x * offset
        y = ref_wp.transform.location.y * (1 - offset) + tar_wp.transform.location.y * offset
        z = ref_wp.transform.location.z
        points.append((x, y, z))

    return points
