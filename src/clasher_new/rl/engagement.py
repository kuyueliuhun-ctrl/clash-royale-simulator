# -*- coding: utf-8 -*-
"""局面圣水交换 · **在线**监视器（预注册 §6 第 5 项的引擎侧接线件，**默认关**）。

设计原则（逐条对准预注册，并可被 `scripts/offline_engagement_trade.py` **独立复算**）：
  * **同一套口径**：局面 = 索敌关系连通分量；交战对 = 跨边 ∧ 至少一侧非产物（规则①）；
    身份判据 = **交战核心**（跨边上的非塔实体）；结算 = 连续 `SETTLE_TICKS` tick 无交战对
    / 分量清空 / 局末。
  * **净交换走 `phi_mode=global`（预注册 §2.1 的字面读法）**：`Φ_x = 手牌圣水 + 残值 V_x`。
    ★ 这条是关键简化：**potential 形式下"出牌"不改变 Φ**（圣水 −c、残值 +c 相消）⇒
    窗口基线取哪一 tick 都不影响 `ΔΦ_0 − ΔΦ_1`（自然回复对两侧相同、也相消）
    ⇒ **在线实现不需要逐窗口的部署记账**。
  * `τ` = **P4b**（唯一候选代理，预注册 §11.7.1 的逐字公式）：
    `gate × Σ_E cost(E) × min(1, 钉住秒数(E)/T_ref)`；`cost(E)` 取在线 `_v_share`
    （**产物体恒 0** ⇒ 绝不估值，§4）；`gate` = 窗口内我方三塔**零掉血**。
  * **零和**：`trade_opp = -trade_me` 由构造保证（§7-1）；`score = max(0, trade − θ)` 两侧各自铰链。
  * **默认关**：开关关时 `RLEnv` **不构造本对象、不做任何 tick 工作**（【R2】逐位回旧）。

成本：每 tick 一次并查集（E≈15）⇒ 可忽略；且只在开关打开时发生。
"""

from __future__ import annotations


class EngagementTradeMonitor:
    """按 tick 维护局面窗口；结算时产出该局面的 `trade`（**p0 视角**）与 `score`。"""

    #: K = 连续多少个 **engine tick** 无交战对 ⇒ 结算（预注册 §1；30 tick = 1 决策帧）
    SETTLE_TICKS = 30

    def __init__(self, theta=1.0, t_ref=2.0, gate=True, shares=None,
                 tick_seconds=1.0 / 30.0):
        self.theta = float(theta)
        self.t_ref = float(t_ref) if t_ref else 0.0
        self.gate = bool(gate)
        self.dt = float(tick_seconds)
        self.shares = shares if shares is not None else {0: {}, 1: {}}   # 在线 _v_share（按引用）
        self._open = {}
        self._pending = 0.0
        self.n_settled = 0
        self.sum_trade = 0.0
        self.settled = []          # [(trade, score, n_members)]，供测试/取证

    # ---------------- 图 ----------------
    @staticmethod
    def _components(ents):
        parent = {i: i for i in ents}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        for i, e in ents.items():
            t = getattr(e, "target_id", None)
            if t is not None and t in ents:
                ra, rb = find(i), find(t)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
        g = {}
        for i in ents:
            g.setdefault(find(i), set()).add(i)
        return [g[k] for k in sorted(g, key=lambda r: min(g[r]))]

    @staticmethod
    def _is_product(e):
        return getattr(e, "is_product", None) is True

    @classmethod
    def _cross_and_core(cls, comp, ents):
        cross, core = set(), set()
        for i in comp:
            t = getattr(ents[i], "target_id", None)
            if t is None or t not in comp or ents[i].player == ents[t].player:
                continue
            a, b = (i, t) if i < t else (t, i)
            cross.add((a, b))
            if a > 6:
                core.add(a)
            if b > 6:
                core.add(b)
        return cross, core

    @classmethod
    def _engaged(cls, cross, ents):
        """规则①：至少一侧非产物。"""
        for a, b in cross:
            if not (cls._is_product(ents[a]) and cls._is_product(ents[b])):
                return True
        return False

    @staticmethod
    def _tower_hp(battle, pid):
        tot = 0.0
        for i in ((3, 4, 6) if pid == 0 else (1, 2, 5)):
            e = battle.entities.get(i)
            if e is not None and getattr(e, "is_alive", False):
                tot += float(e.hp)
        return tot

    # ---------------- 主循环 ----------------
    def tick(self, battle, v):
        """`v` = `[V_0, V_1]`（= `RLEnv._active_v`）。返回本 tick 结算掉的局面数。"""
        ents = {eid: e for eid, e in battle.entities.items()
                if getattr(e, "is_alive", False)}
        comps = self._components(ents)
        act, cores, crosses = [], [], []
        for c in comps:
            cr, co = self._cross_and_core(c, ents)
            crosses.append(cr)
            cores.append(co)
            act.append(self._engaged(cr, ents))

        # —— 匹配：分量 ↔ 窗口。活跃分量用"交战核心"，不活跃用全成员（去抖窗内可续接）；
        #    窗口侧比较的是它**最后一次活跃的核心** ⇒ 塔单独重叠不算续接。
        hits = {}
        for ci, c in enumerate(comps):
            key = set(cores[ci]) if act[ci] else set(c)
            if not key:
                continue
            for rep, w in self._open.items():
                if w["key"] & key:
                    hits.setdefault(rep, []).append(ci)

        closed, n_settled = [], 0
        for rep, cis in hits.items():
            w = self._open[rep]
            active = [ci for ci in cis if act[ci]]
            for ci in cis:
                w["members"] |= set(comps[ci])
                w["core"] |= set(cores[ci])
            if active:
                best = max(active, key=lambda ci: (len(w["key"] & set(cores[ci])), -ci))
                w["key"] = set(cores[best])
                w["last_active"] = battle.tick
                w["inactive"] = 0
            else:
                w["inactive"] += 1
                if w["inactive"] >= self.SETTLE_TICKS:
                    closed.append(rep)
        for rep in list(self._open):
            if rep in closed or rep not in hits:
                n_settled += self._settle(self._open.pop(rep), battle, v)

        # —— P4b 钉住计数：本 tick 里"被 p0 的非塔实体钉住"的敌方实体 ——
        pinned = set()
        for ci in range(len(comps)):
            if not act[ci]:
                continue
            for a, b in crosses[ci]:
                for x, y in ((a, b), (b, a)):
                    if x <= 6:                     # 必须是**非塔**物体 x 钉住 y
                        continue
                    if ents[y].player == 0:        # y 必须是**敌方**（p0 视角）
                        continue
                    pinned.add(y)
        if pinned:
            for w in self._open.values():
                for eid in (pinned & w["members"]):
                    w["pin"][eid] = w["pin"].get(eid, 0) + 1

        # —— 开新窗口：本 tick 活跃、且没有窗口续接它的核心 ——
        claimed = {ci for cis in hits.values() for ci in cis}
        for ci, c in enumerate(comps):
            if not act[ci] or not cores[ci] or ci in claimed:
                continue
            rep = ("w", battle.tick, ci)
            self._open[rep] = {
                "rep": rep, "members": set(c), "core": set(cores[ci]),
                "key": set(cores[ci]), "inactive": 0, "last_active": battle.tick,
                "phi0": float(battle.players[0].elixir) + float(v[0]),
                "phi1": float(battle.players[1].elixir) + float(v[1]),
                "thp0": self._tower_hp(battle, 0), "pin": {},
            }
        return n_settled

    def _settle(self, w, battle, v):
        p0, p1 = battle.players
        d0 = (float(p0.elixir) + float(v[0])) - w["phi0"]
        d1 = (float(p1.elixir) + float(v[1])) - w["phi1"]
        tau = 0.0
        if self.t_ref and w["pin"]:
            g = 1.0
            if self.gate and self._tower_hp(battle, 0) < w["thp0"] - 1e-6:
                g = 0.0
            for eid, cnt in w["pin"].items():
                owner = 1 if eid in self.shares[1] else 0
                cost = float(self.shares[owner].get(eid, 0.0))   # 产物体恒 0
                tau += g * cost * min(1.0, cnt * self.dt / self.t_ref)
        trade = d0 - d1 + tau
        score = max(0.0, trade - self.theta)
        self._pending += score
        self.n_settled += 1
        self.sum_trade += trade
        self.settled.append((trade, score, len(w["members"])))
        return 1

    # ---------------- 收口 ----------------
    def flush(self, battle, v):
        """局末结算所有仍开着的窗口（与离线仪器的 `episode_end` 对齐）。"""
        n = 0
        for rep in list(self._open):
            n += self._settle(self._open.pop(rep), battle, v)
        return n

    def pop_scores(self):
        """取走"自上次调用以来结算掉的局面"的 `score` 之和（p0 视角）。"""
        s = self._pending
        self._pending = 0.0
        return s
