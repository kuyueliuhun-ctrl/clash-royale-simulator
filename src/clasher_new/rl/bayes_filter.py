"""对手循环牌序的贝叶斯滤波（粒子滤波实现，规划文档 3.6 规则信念层）。

原理：CR 卡组是固定 8 卡循环，打出某张卡后从手牌移除并追加到队尾。
给定对手已打出的卡序列，用粒子集维护「当前循环排列」的后验分布。

更新规则（与 player.play_card 一致）：
    循环排列 perm，打出 c ∈ perm[:4] → 新排列 = [x for x in perm if x != c] + [c]

v1 硬收敛补缺口（docs/rl_plan_design_v1.md 评审：后期手牌 = 明牌）：
- 粒子模式只做**近似后验**（不锁定）：一致解 ≤ n_particles 时全量保留（防抽样灭真解），
  > n 时才采样；
- **全局重建**：一旦观测序列覆盖全部 8 张卡（对手完整轮转），做一次全量重放校验
  （40320 排列 × observed，每对局一次，成本可控）：
  * 唯一一致解 → 切 ``_exact`` 确定性模式（hand/next 概率精确 0/1）；
  * 有限多解（≤256）→ 转为显式精确集，后续观测增量剔除，收敛到 1 即锁定；
  * 超过 256（信息仍不足）→ 继续粒子近似，等下一轮再重建。
- 确定性模式对手牌外出牌（镜像等特殊机制）→ 保守退回粒子重估。

不引入 hash/系统随机 → 同 seed 对局跨进程粒子路径一致（并行评估逐局等价）。
"""

import os
import sys
import random
from itertools import permutations

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import numpy as np


class CycleBayesFilter:
    """对对手 8 卡循环排列做贝叶斯粒子滤波（完整一轮后全局重建 → 明牌）。"""

    #: 全局重建允许保留的最大显式一致解数（超出视为信息不足，回到粒子近似）
    MAX_EXPLICIT = 256

    def __init__(self, deck, n_particles: int = 128, seed: int = 0):
        self.deck = list(deck)
        self.n_particles = n_particles
        self._rng = random.Random(seed)
        self.particles = []
        self.weights = np.ones(n_particles) / n_particles
        self.observed = []
        self._exact = None        # 确定性排列（None=未锁定）
        self._explicit = False    # particles 是否为全局重建的显式精确解集
        self._rebuild_counter = 0  # full-cycle 后重建节流计数
        self._rebuild_every = 4    # 重建节流基数（信息不足时指数退避）
        self._reset()

    def _reset(self):
        self.particles = []
        for _ in range(self.n_particles):
            perm = list(self.deck)
            self._rng.shuffle(perm)
            self.particles.append(perm)
        self.weights = np.ones(self.n_particles) / self.n_particles
        self.observed = []
        self._exact = None
        self._explicit = False
        self._rebuild_counter = 0
        self._rebuild_every = 4

    def reset(self, deck=None):
        if deck is not None:
            self.deck = list(deck)
        self._reset()

    @staticmethod
    def _play(perm, card):
        return [c for c in perm if c != card] + [card]

    @staticmethod
    def _replay_ok(perm, seq):
        """校验候选排列能否重放整段观测序列（队列规则逐步重放）。"""
        p = list(perm)
        for c in seq:
            if c not in p[:4]:
                return False
            p = CycleBayesFilter._play(p, c)
        return True

    def _rebuild_global(self):
        """全量扫描 deck 排列，收集与全部观测一致的解（最多 MAX_EXPLICIT+1 个）。

        全扫成本实测 ~0.01-0.05s（快速失败），full-cycle 后每 4 步重试一次可接受。
        """
        hits = []
        for perm in permutations(self.deck):
            if self._replay_ok(list(perm), self.observed):
                hits.append(list(perm))
                if len(hits) > self.MAX_EXPLICIT:
                    break
        return hits

    def _apply_exact(self, played_card):
        perm = self._exact
        if played_card in perm[:4]:
            self._exact = self._play(perm, played_card)
            self.observed.append(played_card)
            return True
        return False

    def _try_lock_single(self, perms):
        if len(perms) == 1:
            self._exact = list(perms[0])
            self.particles = []
            self.weights = np.zeros(0, dtype=np.float32)
            self._explicit = False
            return True
        return False

    def update(self, played_card):
        """观测到对手打出了 played_card → 贝叶斯更新（完整一轮后全局重建锁定）。"""
        if played_card is None:
            return
        # —— 确定性模式：精确推进 ——
        if self._exact is not None:
            if not self._apply_exact(played_card):
                # 精确模式下出现手牌外出牌（镜像/特殊机制等）：保守退回粒子重估
                self._reset()
                self.update(played_card)
            return

        self.observed.append(played_card)

        # —— 完整一轮后节流重建（指数退避：信息不足时降低频率）——
        if (self._exact is None and not self._explicit
                and len(set(self.observed)) == len(self.deck)):
            self._rebuild_counter += 1
            if self._rebuild_counter >= self._rebuild_every:
                self._rebuild_counter = 0
                hits = self._rebuild_global()
                if self._try_lock_single(hits):
                    return
                if 1 < len(hits) <= self.MAX_EXPLICIT:
                    # 显式精确集：后续观测增量剔除，收敛到 1 即锁定
                    self._explicit = True
                    self.particles = hits
                    self.weights = np.ones(len(hits), dtype=np.float32) / len(hits)
                    return
                # hits=0（防御）或 >MAX（对手出牌信息不足以唯一化）→ 保持粒子近似，
                # 指数退避重建频率（观测信息通常不随重复轮次增长）
                self._rebuild_every = min(64, self._rebuild_every * 2)

        # —— 显式集增量剔除（全局重建出的精确候选，逐观测收敛到 1）——
        if self._explicit:
            seen = {}
            for p in self.particles:
                if played_card in p[:4]:
                    seen[tuple(self._play(p, played_card))] = True
            keep = [list(k) for k in seen]
            if self._try_lock_single(keep):
                return
            if not keep:
                # 观测与显式集冲突（特殊机制/误观测）→ 回退粒子近似并允许重建
                self._explicit = False
                self.particles = []
                for _ in range(self.n_particles):
                    perm = list(self.deck)
                    self._rng.shuffle(perm)
                    self.particles.append(perm)
                self.weights = np.ones(self.n_particles, dtype=np.float32) / self.n_particles
                self._rebuild_counter = 0
                return
            self.particles = keep
            self.weights = np.ones(len(keep), dtype=np.float32) / len(keep)
            return

        # —— 粒子近似：合并一致排列（保留 distinct，≤n 全量不抽样）——
        consistent = {}
        for w, perm in zip(self.weights, self.particles):
            if played_card in perm[:4]:
                key = tuple(perm)
                consistent[key] = consistent.get(key, 0.0) + w
        if not consistent:
            # 粒子被抽样灭掉/误观测：用现有 rng 重采样全排列先验（保留 observed）
            self.particles = []
            for _ in range(self.n_particles):
                perm = list(self.deck)
                self._rng.shuffle(perm)
                self.particles.append(perm)
            self.weights = np.ones(self.n_particles, dtype=np.float32) / self.n_particles
            return
        perms = [list(k) for k in consistent]
        if len(perms) <= self.n_particles:
            self.particles = perms
            self.weights = np.ones(len(perms), dtype=np.float32) / len(perms)
        else:
            keys = list(consistent)
            ws = np.array([consistent[k] for k in keys], dtype=np.float32)
            ws = ws / ws.sum()
            idx = self._rng.choices(range(len(keys)), weights=ws, k=self.n_particles)
            self.particles = [list(keys[i]) for i in idx]
            self.weights = np.ones(self.n_particles, dtype=np.float32) / self.n_particles

    def hand_probs(self) -> np.ndarray:
        """每张卡在对手手牌（cycle[:4]）中的概率，按 deck 顺序返回。

        确定性模式返回精确 0/1（后期手牌 = 明牌）。
        """
        probs = {c: 0.0 for c in self.deck}
        if self._exact is not None:
            for c in self._exact[:4]:
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
        if self._exact is not None:
            probs[self._exact[4]] = 1.0
        else:
            n = max(1, len(self.particles))
            for perm in self.particles:
                probs[perm[4]] += 1.0 / n
        return np.array([probs[c] for c in self.deck], dtype=np.float32)

    def entropy(self) -> float:
        """确定性模式熵 = 0（明牌）。"""
        if self._exact is not None:
            return 0.0
        p = self.hand_probs()
        p = p[p > 0]
        return float(-(p * np.log(p)).sum()) if p.size else 0.0
