"""对手 8 卡循环牌序信念（规划文档 3.6 规则信念层）。

引擎队列规则（与 player.play_card 一致）：
    循环排列 cycle，手牌 = cycle[:4]；打出 c（须在 cycle[:4]）→
    cycle = [x for x in cycle if x != c] + [c]（打出卡移队尾，队头补进手牌）

v2 结构 —— O(1) 队列锁定，替代 v1 的 40320 全量重建粒子滤波：

数学事实（暴力验证 3000 随机对局 × 59 步零反例）：
    对任意合法出牌流，从第 4 张起当前手牌集合 = 卡组 − 最近 4 张（互异），
    下一张进手 = 第 k−3 张打出的牌 —— 与开局洗牌顺序（40320 排列）无关。
    因此只要 8 张卡内容已知 + 出牌按序全观测，第 4 张起信念即精确 0/1，
    O(1) 滑动窗口即可；开局排列不可唯一复原（6144 兼容排列）不影响任何
    可观测量（手牌集合 / 下一张），不需要排列级粒子/全量重建。

模式：
- 锁定流(stream)：维护规范 cycle（手牌顺序取卡组序，队尾队列 = 最近 4 张
  出牌的时序），每次 update remove+append O(1) 精确推进；hand/next = 0/1。
- 粒子相(particles)：仅用于前 3 张（初始手牌/牌库边界未定：可能手牌集
  35 → 15 → 5 种）与异常观测（锁定流出现手牌外出牌、观测不完整导致的
  状态分歧、卡不在卡组内等）：均匀先验粒子 + 逐观测一致性筛选；连续
  4 张一致合法出牌后按「最近 4 张」重建规范 cycle，回到锁定流（第 4 张起
  与历史无关，重锁必精确）。若观测缺口小到从未造成任何不一致（超出
  “按序全观测”前提），只能维持粒子近似 —— 与 v1 粒子模式能力相当。
- 无伪锁语义不变：锁定只由最近 4 张合法出牌重建触发，数学上不可能锁错；
  4 张窗口含重复/未知卡时保持粒子相不锁。

不引入 hash/系统随机 → 同 seed 对局跨进程一致。
"""

import os
import sys
import random

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import numpy as np


class CycleBayesFilter:
    """对手 8 卡循环队列信念：O(1) 锁定流 + 早期/异常粒子相。"""

    def __init__(self, deck, n_particles: int = 128, seed: int = 0):
        self.deck = list(deck)
        self.n_particles = n_particles
        self._rng = random.Random(seed)
        self.observed = []          # 观测到的出牌序列（含异常观测，调试/重锁窗口用）
        self.particles = []         # 粒子相候选排列（当前 cycle 状态）
        self.weights = np.ones(n_particles, dtype=np.float32) / n_particles
        self._cycle = None          # 规范 cycle（8 卡）；None = 粒子相
        self._run = 0               # 粒子相连续一致步数（≥4 → 可重建锁定流）
        self._resample_uniform()

    @property
    def locked(self) -> bool:
        """是否处于 O(1) 精确锁定流（hand/next 概率 0/1、熵 0）。"""
        return self._cycle is not None

    def reset(self, deck=None):
        if deck is not None:
            self.deck = list(deck)
        self.observed = []
        self._cycle = None
        self._run = 0
        self._resample_uniform()

    # ---- 队列规则 ----

    @staticmethod
    def _play(perm, card):
        """打出 card（须在手牌 perm[:4]）→ 移队尾。与 player.play_card 一致。"""
        return [c for c in perm if c != card] + [card]

    def _lock_from_tail(self, last4):
        """用最近 4 张合法出牌重建规范 cycle（手牌集合 = 卡组 − last4）。

        规范 cycle = [卡组序手牌] + [last4 时序]；手牌内部顺序不可观测且不影响
        后续 remove+append 演化，故取卡组序保证确定性；队尾队列 = 真实进手序。
        """
        queue = list(last4)
        qset = set(queue)
        self._cycle = [c for c in self.deck if c not in qset] + queue
        self.particles = []
        self.weights = np.zeros(0, dtype=np.float32)
        self._run = 0

    def _resample_uniform(self):
        """均匀先验重采样粒子（去重，保证等权；确定性由 seed 的 Random 保证）。"""
        self.particles = []
        seen = set()
        while len(self.particles) < self.n_particles:
            perm = list(self.deck)
            self._rng.shuffle(perm)
            key = tuple(perm)
            if key not in seen:
                seen.add(key)
                self.particles.append(perm)
        self.weights = np.ones(self.n_particles, dtype=np.float32) / self.n_particles

    def _degrade(self):
        """异常/分歧 → 退回粒子相（保留 observed，后续合法 4 张自动重锁）。"""
        self._cycle = None
        self._run = 0
        self._resample_uniform()

    def _consistent(self, card):
        """粒子相一致性筛选：只保留能打出 card 的候选，并推进到打出后状态。"""
        out = {}
        for perm in self.particles:
            if card in perm[:4]:
                out[tuple(self._play(perm, card))] = True
        return [list(k) for k in out]

    def update(self, played_card):
        """观测到对手打出了 played_card → 更新队列信念（O(1) 锁定流或粒子相）。"""
        if played_card is None:
            return
        self.observed.append(played_card)

        # —— 锁定流：O(1) remove+append 精确推进 ——
        if self._cycle is not None:
            if played_card in self._cycle[:4]:
                self._cycle = self._play(self._cycle, played_card)
                return
            # 手牌外出牌 / 状态分歧 → 粒子相（保守重估，不推进）
            self._degrade()
            return

        # —— 粒子相：逐观测一致性 ——
        perms = self._consistent(played_card)
        if not perms:
            # 采样损耗或异常观测 → 均匀重采样（确定性 seed），run 清零
            self._degrade()
            return
        # 输入粒子 ≤ n_particles，一致解必 ≤ n → 全量保留，无抽样灭真解
        self.particles = perms
        self.weights = np.ones(len(perms), dtype=np.float32) / len(perms)
        self._run += 1

        # 连续 4 张一致合法出牌 → 锁定流（窗口 = 最近 4 张，与更早历史无关）
        if self._run >= 4 and len(self.observed) >= 4:
            last4 = self.observed[-4:]
            if (len(set(last4)) == 4
                    and all(c in self.deck for c in last4)):
                self._lock_from_tail(last4)

    # ---- 查询 ----

    def hand_probs(self) -> np.ndarray:
        """每张卡在对手手牌（cycle[:4]）中的概率，按 deck 顺序返回。

        锁定流返回精确 0/1；粒子相返回频率近似（前 3 张/异常期间）。
        """
        probs = {c: 0.0 for c in self.deck}
        if self._cycle is not None:
            for c in self._cycle[:4]:
                probs[c] = 1.0
        else:
            n = max(1, len(self.particles))
            for perm in self.particles:
                for c in perm[:4]:
                    probs[c] += 1.0 / n
        return np.array([probs[c] for c in self.deck], dtype=np.float32)

    def next_probs(self) -> np.ndarray:
        """每张卡是对手下一张牌（cycle[4]）的概率，按 deck 顺序返回。"""
        probs = {c: 0.0 for c in self.deck}
        if self._cycle is not None:
            probs[self._cycle[4]] = 1.0
        else:
            n = max(1, len(self.particles))
            for perm in self.particles:
                probs[perm[4]] += 1.0 / n
        return np.array([probs[c] for c in self.deck], dtype=np.float32)

    def entropy(self) -> float:
        """锁定流熵 = 0（明牌，第 4 张起内容已知+全观测即精确）。"""
        if self._cycle is not None:
            return 0.0
        p = self.hand_probs()
        p = p[p > 0]
        return float(-(p * np.log(p)).sum()) if p.size else 0.0
