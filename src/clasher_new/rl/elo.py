"""Elo 评分（规划文档 7.4）。只用于联赛调度与分析，不直接作为 RL reward。

修复：
- P1-11：update 支持 n_games 参数，按局数缩放 K（聚合多局成绩不压垮量表）；
- P2：save / load 持久化。
"""

import os
import sys

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)


class Elo:
    def __init__(self, k: float = 32.0, initial: float = 1500.0):
        self.k = k
        self.initial = initial
        self.ratings = {}

    def ensure(self, agent_id) -> float:
        if agent_id not in self.ratings:
            self.ratings[agent_id] = self.initial
        return self.ratings[agent_id]

    @staticmethod
    def expected(r_a: float, r_b: float) -> float:
        return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))

    def update(self, agent_a, agent_b, score_a: float, n_games: int = 1):
        """score_a ∈ [0,1]：agent_a 的胜率（1=胜，0=负，0.5=平）。

        n_games>1 表示本次成绩聚合了 n 局，K 按局数缩放（P1-11）。
        """
        r_a = self.ensure(agent_a)
        r_b = self.ensure(agent_b)
        e_a = self.expected(r_a, r_b)
        k_eff = self.k * max(1, int(n_games))
        self.ratings[agent_a] = r_a + k_eff * (score_a - e_a)
        self.ratings[agent_b] = r_b + k_eff * ((1 - score_a) - (1 - e_a))
        return self.ratings[agent_a], self.ratings[agent_b]

    def table(self):
        return dict(sorted(self.ratings.items(), key=lambda kv: -kv[1]))

    def save(self, path):
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"k": self.k, "initial": self.initial, "ratings": self.ratings}, f)

    @classmethod
    def load(cls, path):
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        e = cls(k=float(data.get("k", 32.0)), initial=float(data.get("initial", 1500.0)))
        e.ratings = {k: float(v) for k, v in data["ratings"].items()}
        return e
