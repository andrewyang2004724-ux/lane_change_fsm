# utils/mobil_sim.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.patches as patches

# --- IDM 参数 ---
A_MAX, B_SAFE, T_GAP, S0, DELTA = 1.5, 2.0, 1.0, 4.0, 4.0

def get_idm_accel(v, v_lead, s, v0=25.0):
    if s <= 0.1: return -5.0
    dv = v - v_lead
    s_star = S0 + v * T_GAP + (v * dv) / (2.0 * np.sqrt(A_MAX * B_SAFE))
    s_star = max(s_star, S0)
    return A_MAX * (1.0 - (v / v0)**DELTA - (s_star / s)**2)

# ==========================================
# UI 绝对坐标布局设计
# ==========================================
fig = plt.figure(figsize=(12, 9))
fig.canvas.manager.set_window_title("MOBIL Simulator V2 - Fixed Layout")

# 1. 顶部视觉仿真区 (占上方 30% 空间，从 Y=0.65 开始)
ax_vis = plt.axes([0.05, 0.65, 0.9, 0.30])
ax_vis.set_xlim(-50, 150)
ax_vis.set_ylim(-1.5, 1.5)
ax_vis.axis('off')
ax_vis.set_title("Highway Real-time Overview", fontsize=14, fontweight='bold', pad=10)

# 画车道线
ax_vis.axhline(0.5, color='gray', linestyle='--')
ax_vis.axhline(-0.5, color='gray', linestyle='--')
ax_vis.axhline(1.5, color='black', linewidth=2)
ax_vis.axhline(-1.5, color='black', linewidth=2)

# 定义车辆色块
car_width, car_height = 8, 0.6
ego_car = patches.Rectangle((0, -car_height/2 - 0.5), car_width, car_height, color='blue', label='Ego (You)')
curr_front_car = patches.Rectangle((30, -car_height/2 - 0.5), car_width, car_height, color='red', label='Curr Front')
tgt_front_car = patches.Rectangle((60, -car_height/2 + 0.5), car_width, car_height, color='green', label='Target Front')
tgt_rear_car = patches.Rectangle((-20, -car_height/2 + 0.5), car_width, car_height, color='orange', label='Target Rear')

for car in [ego_car, curr_front_car, tgt_front_car, tgt_rear_car]:
    ax_vis.add_patch(car)
ax_vis.legend(loc='upper right', bbox_to_anchor=(1.05, 1.1))

# 动态换道箭头
lane_change_arrow = ax_vis.annotate(
    '', xy=(car_width/2, 0.5), xytext=(car_width/2, -0.5),
    arrowprops=dict(facecolor='magenta', shrink=0.05, width=3, headwidth=10),
    visible=False
)

# 2. 右下方：计算数据展板区
ax_text = plt.axes([0.55, 0.05, 0.40, 0.50])
ax_text.axis('off')
text_info = ax_text.text(0.1, 0.5, "", transform=ax_text.transAxes, va='center', size=13, family='monospace')

# 3. 左下方：控制滑块区
# 将 9 个滑块均匀分布在 Y=0.05 到 Y=0.55 的区间内
y_positions = np.linspace(0.55, 0.05, 9)
configs = [
    ([0.15, y_positions[0], 0.30, 0.03], 'Politeness (p)', 0.0, 1.0, 0.25),
    ([0.15, y_positions[1], 0.30, 0.03], 'Threshold (a_th)', 0.0, 1.0, 0.15),
    ([0.15, y_positions[2], 0.30, 0.03], 'V_ego (m/s)', 10, 35, 25),
    ([0.15, y_positions[3], 0.30, 0.03], 'Dist Curr Front (m)', 5, 120, 30),
    ([0.15, y_positions[4], 0.30, 0.03], 'V Curr Front (m/s)', 0, 35, 15),
    ([0.15, y_positions[5], 0.30, 0.03], 'Dist Tgt Front (m)', 5, 120, 60),
    ([0.15, y_positions[6], 0.30, 0.03], 'V Tgt Front (m/s)', 0, 35, 25),
    ([0.15, y_positions[7], 0.30, 0.03], 'Dist Tgt Rear (m)', 5, 120, 20),
    ([0.15, y_positions[8], 0.30, 0.03], 'V Tgt Rear (m/s)', 10, 40, 28)
]

sliders = []
for pos, label, vmin, vmax, valinit in configs:
    ax = plt.axes(pos)
    sliders.append(Slider(ax, label, vmin, vmax, valinit=valinit))
s_p, s_ath, s_vego, s_dcurr, s_vcurr, s_dtgt_f, s_vtgt_f, s_dtgt_r, s_vtgt_r = sliders

def update(val):
    p, a_th, v_ego = s_p.val, s_ath.val, s_vego.val
    s_curr, v_curr = s_dcurr.val, s_vcurr.val
    s_tgt_f, v_tgt_f = s_dtgt_f.val, s_vtgt_f.val
    s_tgt_r, v_tgt_r = s_dtgt_r.val, s_vtgt_r.val

    # 更新车辆动画位置
    curr_front_car.set_x(s_curr)
    tgt_front_car.set_x(s_tgt_f)
    tgt_rear_car.set_x(-s_tgt_r)  # 后车在负轴

    # 计算 MOBIL
    a_curr = get_idm_accel(v_ego, v_curr, s_curr)
    a_tgt = get_idm_accel(v_ego, v_tgt_f, s_tgt_f)
    ego_gain = a_tgt - a_curr

    a_rear_old = get_idm_accel(v_tgt_r, v_tgt_f, s_tgt_r + s_tgt_f)
    a_rear_new = get_idm_accel(v_tgt_r, v_ego, s_tgt_r)
    rear_loss = min(0, a_rear_new - a_rear_old)

    total_utility = ego_gain + p * rear_loss
    will_change = total_utility > a_th

    # 更新动画箭头
    lane_change_arrow.set_visible(will_change)

    # 更新数据看板
    decision_color = "🟢 GREEN (Go!)" if will_change else "🔴 RED (Wait)"
    text_info.set_text(
        f"--- IDM ACCELERATION ---\n"
        f"Ego Stay:   {a_curr:>6.2f} m/s²\n"
        f"Ego Change: {a_tgt:>6.2f} m/s²\n\n"
        f"--- MOBIL UTILITY ---\n"
        f"Ego Gain:   {ego_gain:>6.2f} m/s²\n"
        f"Rear Loss:  {rear_loss:>6.2f} m/s²\n"
        f"Total Util: {total_utility:>6.2f} m/s²\n\n"
        f"--- DECISION ---\n"
        f"Required: > {a_th:.2f}\n"
        f"Status: {decision_color}"
    )
    fig.canvas.draw_idle()

for s in sliders: s.on_changed(update)
update(None)
plt.show()