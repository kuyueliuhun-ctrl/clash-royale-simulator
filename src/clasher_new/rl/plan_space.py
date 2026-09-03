"""计划空间：Prophet / BeliefPlanner 输出的 plan token（规划文档 5.2 / 5.5）。

维度口径（P0-5）：`PLAN_DIM = len(PlanToken().to_vector())` 是**唯一**常量源，
任何加载 follower checkpoint 的入口都必须从 checkpoint 元数据或该常量读取，
禁止魔法数字。
"""

import os
import sys

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np

MACRO_INTENTS = [
    "defend_left", "defend_right", "defend_king",
    "push_left", "push_right", "counterpush",
    "spell_value", "cycle_and_wait",
]

FOCUS_REGIONS = [
    "own_left", "own_center", "own_right",
    "bridge_left", "bridge_right",
    "enemy_left", "enemy_center", "enemy_right",
]


@dataclass
class PlanToken:
    macro_intent: str = "cycle_and_wait"
    focus_region: str = "own_center"
    suggested_card: Optional[int] = None       # 槽位 1..4
    bundle_size_hint: int = 1
    combo_hint: int = 0                        # 0 无 / 1 坦克+后排 / 2 法术+单位 / 3 双路
    risk_profile: float = 0.5                  # 0 保守 .. 1 激进
    value_estimate: float = 0.0

    def to_vector(self) -> np.ndarray:
        """离散化为固定长度向量，供 follower 拼接。"""
        intent = np.zeros(len(MACRO_INTENTS), dtype=np.float32)
        if self.macro_intent in MACRO_INTENTS:
            intent[MACRO_INTENTS.index(self.macro_intent)] = 1.0
        else:
            intent[MACRO_INTENTS.index("cycle_and_wait")] = 1.0
        region = np.zeros(len(FOCUS_REGIONS), dtype=np.float32)
        if self.focus_region in FOCUS_REGIONS:
            region[FOCUS_REGIONS.index(self.focus_region)] = 1.0
        else:
            region[FOCUS_REGIONS.index("own_center")] = 1.0
        return np.concatenate([
            intent,
            region,
            np.array([
                float((self.suggested_card or 0)) / 4.0,   # 归一化槽位（P2）
                float(self.bundle_size_hint),
                float(self.combo_hint),
                float(np.clip(self.risk_profile, 0.0, 1.0)),
                float(np.clip(np.tanh(self.value_estimate), -1.0, 1.0)),
            ], dtype=np.float32),
        ])

    @classmethod
    def zeros(cls) -> "PlanToken":
        return cls()


#: 计划向量唯一维度常量（P0-5）
PLAN_DIM = int(len(PlanToken().to_vector()))
