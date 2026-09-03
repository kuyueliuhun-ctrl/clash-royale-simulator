"""对手循环牌序的贝叶斯滤波（粒子滤波实现，规划文档 3.6 规则信念层）。

原理：CR 卡组是固定 8 卡循环，打出某张卡后从手牌移除并追加到队尾。
给定对手已打出的卡序列，用粒子集维护「当前循环排列」的后验分布。

更新规则（与 player.play_card 一致）：
    循环排列 perm，打出 c ∈ perm[:4] → 新排列 = [x for x in perm if x != c] + [c]
"""

import os
import sys
import random

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import numpy as np


class CycleBayesFilter:
    """对对手 8 卡循环排列做贝叶斯粒子滤波。"""

    def __init__(self, deck, n_particles: int = 128, seed: int = 0):
        self.deck = list(deck)
        self.n_particles = n_particles
        self._rng = random.Random(seed)
        self.particles = []
        self.weights = np.ones(n_particles) / n_particles
        self.observed = []
        self._reset()

    def _reset(self):
        self.particles = []
        for _ in range(self.n_particles):
            perm = list(self.deck)
            self._rng.shuffle(perm)
            self.particles.append(perm)
        self.weights = np.ones(self.n_particles) / self.n_particles
        self.observed = []

    def reset(self, deck=None):
        if deck is not None:
            self.deck = list(deck)
        self._reset()

    def update(self, played_card):
        """观测到对手打出了 played_card → 贝叶斯更新。"""
        if played_card is None:
            return
        new_particles, new_weights = [], []
        for w, perm in zip(self.weights, self.particles):
            if played_card in perm[:4]:
                new_perm = [c for c in perm if c != played_card] + [played_card]
                new_particles.append(new_perm)
                new_weights.append(w)
        if not new_particles:
            # 所有粒子与观测不一致（可能误观测），重置为均匀先验
            self._reset()
            self.observed.append(played_card)
            return
        w = np.array(new_weights)
        w = w / w.sum()
        idx = self._rng.choices(range(len(new_particles)), weights=w, k=self.n_particles)
        self.particles = [new_particles[i] for i in idx]
        self.weights = np.ones(self.n_particles) / self.n_particles
        self.observed.append(played_card)

    def hand_probs(self) -> np.ndarray:
        """每张卡在对手手牌（cycle[:4]）中的概率，按 deck 顺序返回。"""
        probs = {c: 0.0 for c in self.deck}
        for perm in self.particles:
            for c in perm[:4]:
                probs[c] += 1.0 / self.n_particles
        return np.array([probs[c] for c in self.deck], dtype=np.float32)

    def next_probs(self) -> np.ndarray:
        """每张卡是对手下一张牌（cycle[4]）的概率，按 deck 顺序返回。"""
        probs = {c: 0.0 for c in self.deck}
        for perm in self.particles:
            probs[perm[4]] += 1.0 / self.n_particles
        return np.array([probs[c] for c in self.deck], dtype=np.float32)

    def entropy(self) -> float:
        p = self.hand_probs()
        p = p[p > 0]
        return float(-(p * np.log(p)).sum()) if p.size else 0.0
