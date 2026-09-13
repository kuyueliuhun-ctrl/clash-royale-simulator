"""PFSP 对手采样（规划文档 7.3）。

P(opponent) ∝ (1 - winrate(main, opponent))^β

修复（P2）：未采样过的对手按 0 胜率（乐观先验）处理，保证新对手也会被提升采样；
校验 beta >= 0。

D1（2026-09-13，`docs/cycling_league_plan_2026-09-13.md`）新增三个**可选**参数，
默认值 = 旧行为（逐位等价，回归哨兵见 `selftest.test_pfsp_gate_and_mix`）：

- ``alpha``：胜率 EMA 的新息率。旧值 0.05 在 20k run（≈64 局、每 ckpt ≈1.5 局）下
  EMA 几乎不动 ⇒ PFSP 名义存在、实际近似均匀采样。训练侧改用 0.20 让它真有分辨率。
- ``gate_hi`` / ``gate_penalty``：**易胜对手门禁**——EMA 胜率 > gate_hi 的 ckpt
  权重 ×gate_penalty（默认 1.0 ⇒ 关闭）。语义：从"已经能碾压的旧自己"身上学不到
  东西，把采样预算让给有信息量的对手。
"""

import os
import sys
import random

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import numpy as np


class PFSP:
    def __init__(self, beta: float = 1.0, seed: int = 0, alpha: float = 0.05,
                 gate_hi: float = 1.0, gate_penalty: float = 1.0):
        if beta < 0:
            raise ValueError("beta 必须 >= 0")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha 必须落在 (0, 1]")
        if not 0.0 <= gate_penalty <= 1.0:
            raise ValueError("gate_penalty 必须落在 [0, 1]")
        self.beta = beta
        self.rng = random.Random(seed)
        self.alpha = float(alpha)
        self.gate_hi = float(gate_hi)
        self.gate_penalty = float(gate_penalty)
        # winrate: {(agent_id, opponent_id): float}
        self.winrates = {}

    def update_winrate(self, agent_a, agent_b, score_a: float, alpha: float = None):
        """EMA 更新 (a 对 b) 的胜率。``alpha=None`` ⇒ 用实例的 ``self.alpha``。"""
        a = self.alpha if alpha is None else float(alpha)
        key = (agent_a, agent_b)
        prev = self.winrates.get(key, 0.5)
        self.winrates[key] = prev * (1 - a) + score_a * a

    def weights(self, agent_id, opponents) -> np.ndarray:
        ws = []
        for op in opponents:
            key = (agent_id, op)
            seen = key in self.winrates
            # 未采样对手视为 0 胜率（乐观先验）→ 高采样权重（P2）
            w = (1.0 - self.winrates.get(key, 0.0)) ** self.beta
            # D1 门禁：已能碾压（EMA 胜率 > gate_hi）⇒ 降权（默认 gate_hi=1.0 不触发）
            if seen and self.gate_hi < 1.0 and self.winrates[key] > self.gate_hi:
                w *= self.gate_penalty
            ws.append(max(w, 1e-6))
        return np.array(ws, dtype=np.float64)

    def sample(self, agent_id, opponents) -> str:
        if not opponents:
            raise ValueError("无对手可采样")
        w = self.weights(agent_id, opponents)
        p = w / w.sum()
        return self.rng.choices(list(opponents), weights=p, k=1)[0]
