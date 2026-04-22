# scripts/scenario_manager.py
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import random
import json
import os

@dataclass
class ScenarioConfig:
    name: str
    seed: int = 42
    density: float = 0.35          # 0~1
    speed_mean_kmh: float = 95.0
    speed_std_kmh: float = 8.0
    truck_ratio: float = 0.15
    lane_bias: Dict[int, float] = None  # 车道采样权重，如 {-1:0.2, 0:0.4, 1:0.4}
    spawn_gap_min_m: float = 18.0
    total_vehicles: int = 40
    left_lane_block_prob: float = 0.0
    slow_lead_enable: bool = False
    slow_lead_speed_kmh: float = 65.0
    slow_lead_distance_m: float = 35.0

    def to_dict(self):
        d = asdict(self)
        if d["lane_bias"] is None:
            d["lane_bias"] = {-1: 0.2, 0: 0.4, 1: 0.4}
        return d


class ScenarioManager:
    TEMPLATES = {
        "FREE_FLOW": dict(density=0.2, total_vehicles=25, speed_mean_kmh=105, speed_std_kmh=10),
        "FOLLOW_SLOW_LEAD": dict(density=0.35, total_vehicles=35, slow_lead_enable=True,
                                 slow_lead_speed_kmh=65, slow_lead_distance_m=30),
        "LEFT_BLOCKED": dict(density=0.45, total_vehicles=45, left_lane_block_prob=0.8,
                             slow_lead_enable=True, slow_lead_speed_kmh=70, slow_lead_distance_m=35),
        "DENSE_TRAFFIC": dict(density=0.7, total_vehicles=70, speed_mean_kmh=90, speed_std_kmh=6),
    }

    @classmethod
    def build(cls, template_name: str, seed: int = 42, overrides: Optional[Dict[str, Any]] = None) -> ScenarioConfig:
        if template_name not in cls.TEMPLATES:
            raise ValueError(f"Unknown template: {template_name}")
        cfg = dict(name=template_name, seed=seed)
        cfg.update(cls.TEMPLATES[template_name])
        if overrides:
            cfg.update(overrides)
        if "lane_bias" not in cfg:
            cfg["lane_bias"] = {-1: 0.2, 0: 0.4, 1: 0.4}
        return ScenarioConfig(**cfg)

    @staticmethod
    def save_run_config(path: str, cfg: ScenarioConfig):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg.to_dict(), f, ensure_ascii=False, indent=2)

    @staticmethod
    def rng(seed: int):
        r = random.Random(seed)
        return r
