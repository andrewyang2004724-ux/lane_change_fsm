# config.py

# =========================
# 仿真基础配置
# =========================
HOST = "127.0.0.1"
PORT = 2000
TIMEOUT = 10.0

SYNC_MODE = True
FIXED_DELTA_SECONDS = 0.05

MAP_NAME = "Town06"   # 高速环路，适合高速公路换道实验

# =========================
# 交通流配置
# =========================
NUM_TRAFFIC = 45
TRAFFIC_MANAGER_PORT = 8000
TRAFFIC_GLOBAL_DISTANCE = 2.5
TRAFFIC_SPEED_DIFF_PERCENT = 35.0   # 背景车低于限速的百分比
TRAFFIC_HYBRID_PHYSICS = False

# =========================
# 自车配置
# =========================
EGO_ROLE_NAME = "hero"
TARGET_SPEED = 90.0 / 3.6          # m/s
OVERTAKE_SPEED = 105.0 / 3.6       # m/s
MAX_SPEED = 120.0 / 3.6            # m/s

# =========================
# 安全评估参数
# =========================
MIN_GAP = 10.0
TIME_HEADWAY = 1.8
COMFORT_DEC = 3.0

TTC_FRONT_MIN = 5.0
TTC_REAR_MIN = 3.0
TTC_EMERGENCY = 2.0

DETECTION_RADIUS = 100.0
PREPARE_STABLE_TICKS = 4
RETURN_OVERTAKE_MARGIN = 15.0
SPEED_DIFF_THRESHOLD = 3.0 / 3.6   # 3km/h

# =========================
# 控制参数
# =========================
KP_STEER = 1.0
KD_STEER = 0.1
MAX_STEER = 0.6

KP_SPEED = 0.8
MAX_THROTTLE = 0.65
MAX_BRAKE = 0.8

# =========================
# 实验时长
# =========================
MAX_TICKS = 4000   # 4000 * 0.05 = 200s

# =========================
# 输出
# =========================
LOG_CSV = "run_log.csv"
SUMMARY_JSON = "summary.json"
