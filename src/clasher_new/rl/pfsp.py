"""PFSP 对手采样（规划文档 7.3）。

P(opponent) ∝ (1 - winrate(main, opponent))^β

修复（P2）：未采样过的对手按 0 胜率（乐观先验）处理，保证新对手也会被提升采样；
校验 beta >= 0。
"""

import os
import sys
import random

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import numpy as np


class PFSP:
    def __init__(self, beta: float = 1.0, seed: int = 0):
        if beta < 0:
            raise ValueError("beta 必须 >= 0")
        self.beta = beta
        self.rng = random.Random(seed)
        # winrate: {(agent_id, opponent_id): float}
        self.winrates = {}

    def update_winrate(self, agent_a, agent_b, score_a: float, alpha: float = 0.05):
        """EMA 更新 (a 对 b) 的胜率。"""
        key = (agent_a, agent_b)
        prev = self.winrates.get(key, 0.5)
        self.winrates[key] = prev * (1 - alpha) + score_a * alpha

    def weights(self, agent_id, opponents) -> np.ndarray:
        ws = []
        for op in opponents:
            # 未采样对手视为 0 胜率（乐观先验）→ 高采样权重（P2）
            w = (1.0 - self.winrates.get((agent_id, op), 0.0)) ** self.beta
            ws.append(max(w, 1e-6))
        return np.array(ws, dtype=np.float64)

    def sample(self, agent_id, opponents) -> str:
        if not opponents:
            raise ValueError("无对手可采样")
        w = self.weights(agent_id, opponents)
        p = w / w.sum()
        return self.rng.choices(list(opponents), weights=p, k=1)[0]
