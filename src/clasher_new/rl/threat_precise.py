# -*- coding: utf-8 -*-
"""精确塔伤（在线可用版）：**触发式**调用引擎推演，**双向一次算完**。

取代对象：`rl/belief_planner.py:159-173` 的 `_enemy_pressure`——它只数单位 + 距离，
不看待打伤害。实测该近似闸门的 balanced accuracy 只有 0.44~0.66，且 4/6 组**劣于
「有敌军就判有威胁」这条零信息基线**（`docs/threat_precise_probe_verdict_2026-09-14.md`）。

设计依据（全部有实测，不是推断）：

| 决定 | 依据 |
|---|---|
| **不能每帧调用** | `estimate_tower_threat` 单次 28~168 ms（随视界线性），预算是 3.85 ms/帧 ⇒ 超 1e4 倍（`docs/engine_tick_cost_2026-09-14.md`） |
| **用触发式而不是周期采样** | 采样保持的滞后误判 7.8~20.8%（`docs/threat_precise_cheapening_verdict_2026-09-14.md` §3）；触发式在触发帧拿到的是**最新值**，无滞后层 |
| **只在上升沿调用一次** | 电平触发率 27.0~29.5% ⇒ 22.4~24.5 ms/帧（超预算 6×）；上升沿 2.8% ⇒ **2.28 ms/帧**（`docs/threat_trigger_verdict_2026-09-14.md` §2） |
| **触发条件 = 半场无兵 ∧ 敌兵到桥头/已过河** | 同上，BA 0.71~0.83；H=20s 精度 **100%**（触发时花的钱从不白花） |
| **双向一次算完** | `estimate_tower_threat(b, 0)` 与 `(b, 1)` 推演的是**同一条轨迹**（双方都不再部署）⇒ 一次 deepcopy+step 就能同时量出两侧塔损，避免触发帧成本翻倍 |
| **默认关闭** | 【红线 R2】：不开启时生产路径必须逐位不变 |

触发（用户提案 + 实测修订；任一条成立即触发，两个方向各判一次）：

1. 存在半场 h：h 内**我方无兵** ∧ h 内敌方有兵且在桥头带（`y ≤ RIVER_Y2 + 1.5`）
2. 存在半场 h：h 内**我方无兵** ∧ h 内敌方有兵**已过河**（`y ≤ RIVER_Y1`）

> ⚠️ 我先前建议的第三条「H 秒内能到桥」**已放弃**：对 H=20 s 它退化为
> 「该半场有敌方单位」（任何单位 20 s 内都能走到桥头）⇒ 等于零信息基线。
> 详见 `docs/threat_trigger_verdict_2026-09-14.md` §5 的更正。

半场口径（用户指定）：**左右半场**，中轴 x=9.0（`arena.py:157-170` 的左 [0,9) / 右 [9,18)），
两半场**都含中轴旁那一格**（cell 9）⇒ 每支部队至少落在一个半场里。

用法::

    from rl.threat_precise import PreciseThreat
    _P = PreciseThreat()
    threat, my_pressure = _P.pressures(battle)     # 与旧 _enemy_pressure 同量纲
"""

from __future__ import annotations

import copy
import math

__all__ = ["PreciseThreat", "combine_both_directions", "HP_PER_PRESSURE",
           "HORIZON_S", "CELL_AXIS", "BRIDGEHEAD_Y", "CROSSED_Y",
           "trigger_directions", "bridge_cols"]

#: 推演视界（秒）。**由成本预算反推**（多 seed 实测，`docs/threat_precise_impl_2026-09-14.md`）：
#:   H=20s ⇒ 摊销 7.0/8.5/10.4 ms/帧（超预算 1.8~2.7×）
#:   H=10s ⇒ 5.9/6.5 ms/帧（超 1.5~1.7×）
#:   H= 5s ⇒ **3.5/2.5 ms/帧（预算 3.85 内）** ← 取此
#: 上升沿触发率与 H 无关（4.0~6.7%），单次成本线性于 H，故 H 是唯一线性杠杆。
HORIZON_S = 5.0

#: 左右半场的中轴（格）。竞技场自身约定：左 x∈[0,9)（cell ≤8）、右 x∈[9,18)（cell ≥9）。
#: 用户口径「半场要包含中轴右侧/左侧一格」⇒ 两半场都含 cell 9（见模块 docstring）。
CELL_AXIS = 9

#: 河：与 `arena.TileGrid.RIVER_Y1/RIVER_Y2` 同源（【红线 R7】）。
RIVER_Y1 = 15.0
RIVER_Y2 = 16.0

#: 桥列**从引擎自身的可走性判定推导**（不是复制字面量）：在河中心那一行上，
#: `TileGrid.is_walkable` 为真的格子就是桥面。这样"哪些格是桥"只有一个来源。
_BRIDGE_COLS = None


def bridge_cols():
    """可走性推导出的桥面格子横坐标集合（惰性缓存）。"""
    global _BRIDGE_COLS
    if _BRIDGE_COLS is None:
        from arena import TileGrid
        from core import Position
        ar = TileGrid()
        y = (RIVER_Y1 + RIVER_Y2) / 2.0
        _BRIDGE_COLS = frozenset(x for x in range(32)
                                 if ar.is_walkable(Position(x + 0.5, y)))
    return _BRIDGE_COLS

#: 桥头带：敌方 y ≤ RIVER_Y2 + 该值 视为"已到桥头"（1.5 ⇒ y ≤ 17.5）
BRIDGEHEAD_Y = RIVER_Y2 + 1.5

#: 已过河：y ≤ RIVER_Y1（`belief_planner.OWN_HALF_EDGE = 15.0` 同口径）
CROSSED_Y = RIVER_Y1

#: 保持帧数：触发后保持该帧数（1 = 只在触发帧用它；随后由 `_exit` 条件决定活性）
HOLD_FRAMES = 1

#: HP → 旧 pressure 刻度的单一折算常量（【R7】单常量源）。
#: 标定规则（**跑之前写死**）：令新标量与旧标量在参考批次上**等均值**
#: （`mean(E)/mean(old_threat)`），取 **3 个 seed 的中位**（不是单次观测，满足【R16】）。
#: H=5s 实测：138 / 205 / 199 ⇒ 取 **200**。
#: 代价（已实测并接受）：闸门阳性率 54.0/60.0/65.3% → 40.0/50.0/64.0%（−1~−14pp）；
#: tanh 饱和 10.7/42.7/53.3% → 32.0/40.7/57.3%（±21pp）。
#: 标定详情：`docs/threat_precise_impl_2026-09-14.md`（脚本 `scripts/probe_precise_threat.py`）。
HP_PER_PRESSURE = 200.0


def _is_tower(e) -> bool:
    return "Tower" in getattr(e, "name", "")


def _mobile(e) -> bool:
    """可移动单位（塔与建筑不算）。"""
    try:
        return float(getattr(e, "speed", 0) or 0) > 0
    except (TypeError, ValueError):
        return False


def _half_of(x: float):
    """返回该 x 落在哪些半场：('L',) / ('R',) / ('L','R')（中轴旁那一格两半都算）。"""
    cx = int(x)
    out = []
    if cx <= CELL_AXIS:
        out.append("L")
    if cx >= CELL_AXIS:
        out.append("R")
    return tuple(out)


def _bridgehead(e) -> bool:
    """敌方单位是否"已到桥头"（在桥头带内且落在桥列上）。"""
    y = float(e.position.y)
    if y > BRIDGEHEAD_Y:
        return False
    return int(float(e.position.x)) in bridge_cols()


def _crossed(e) -> bool:
    """敌方单位是否**已过河**（进入我方半场深处）。"""
    return float(e.position.y) <= CROSSED_Y


def trigger_directions(battle, pusher: int):
    """返回 (triggered, half_set)。

    `pusher` = 进攻方（其单位正在压向 `1 - pusher` 的塔）。
    条件：存在半场 h，使 `1 - pusher`（防守方）在 h 内**没有单位**，
    而 `pusher` 在 h 内有单位已到桥头**或**已过河。
    """
    defenders = set()
    attackers = set()
    for e in battle.entities.values():
        if not e.is_alive or _is_tower(e):
            continue
        p = getattr(e, "player", None)
        if p == 1 - pusher:
            for h in _half_of(float(e.position.x)):
                defenders.add(h)
        elif p == pusher:
            if _bridgehead(e) or _crossed(e):
                for h in _half_of(float(e.position.x)):
                    attackers.add(h)
    hit = attackers - defenders
    return (len(hit) > 0), hit


#: 塔实体 id（与 `threat_calc._TOWER_IDS` 同源约定；由 selftest 对账）
_TOWER_IDS = {0: (3, 4, 6), 1: (1, 2, 5)}


def combine_both_directions(battle, horizon: float = HORIZON_S, dt: float = 1 / 60):
    """**一次推演同时量出两个方向的塔损**。

    语义与 `threat_calc.estimate_tower_threat` 完全一致（"双方不再部署"），
    但对 player 0 与 player 1 各返回一份塔损——因为两侧推演的是**同一条轨迹**，
    调两次工具等于把同一场推演跑两遍。

    返回 `{"to_p0": float, "to_p1": float, "sim_time": float}`。
    由 `rl/selftest.py::test_precise_threat` 对账两个方向都必须与工具单调用值相等。
    """
    from threat_calc import _hostiles_present     # 单一来源：与工具同一守卫

    zero = {"to_p0": 0.0, "to_p1": 0.0, "sim_time": 0.0}
    if battle.game_over:
        return zero
    if not _hostiles_present(battle, 0) and not _hostiles_present(battle, 1):
        return zero

    sim = copy.deepcopy(battle)
    hp = {}
    for pid, tids in _TOWER_IDS.items():
        for tid in tids:
            hp[(pid, tid)] = float(sim.entities[tid].hp)

    t = 0.0
    while t < horizon - 1e-9:
        if sim.game_over:
            break
        sim.step(dt)
        t += dt
        if (not _hostiles_present(sim, 0)) and (not _hostiles_present(sim, 1)):
            break
        if all(not sim.entities[i].is_alive for i in _TOWER_IDS[0]) and \
           all(not sim.entities[i].is_alive for i in _TOWER_IDS[1]):
            break

    out = {"sim_time": round(t, 3)}
    for pid, key in ((0, "to_p0"), (1, "to_p1")):
        tot = 0.0
        for tid in _TOWER_IDS[pid]:
            e = sim.entities[tid]
            tot += max(0.0, hp[(pid, tid)] - float(e.hp))
        out[key] = round(tot, 1)
    return out


class PreciseThreat:
    """触发式精确塔伤提供器（**有状态**：每局必须 `reset()`）。

    `pressures(battle)` 返回 `(threat, my_pressure)`——量纲与旧 `_enemy_pressure` 相同
    （pressure 单位），可直接替换，`PRESSURE_THRESHOLD` 等阈值**无需改动**。

    状态机（每方向独立）：

    - **ARMED**：触发条件不成立 ⇒ 用 `fallback`（默认回退到旧粗糙公式，
      保证"未触发时的行为与今天一致"，而不是凭空给 0）；
    - **HOLD**：触发（上升沿）时算一次精确值并保持；只要该方向仍"有威胁源"
      （pusher 在该半场仍有兵、或防守方该半场仍无兵）就继续持有；
    - 威胁源消失 ⇒ 回到 ARMED。
    """

    def __init__(self, horizon: float = HORIZON_S, fallback_crude: bool = True,
                 enabled: bool = True):
        self.horizon = float(horizon)
        self.fallback_crude = bool(fallback_crude)
        self.enabled = bool(enabled)
        #: 诊断计数（只读用途，不参与决策）——必须在 reset() 之前建好
        self.stats = {"calls": 0, "triggers": 0, "sim_ms": 0.0}
        self._last_time = -1.0
        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        """每局开始必须调用（跨局不得沿用上一局的保持值）。"""
        self._hold = {"A": None, "B": None}      # A: 敌方压我；B: 我压敌方
        self._hold_half = {"A": set(), "B": set()}
        self.stats["calls"] = 0
        self.stats["triggers"] = 0
        self.stats["sim_ms"] = 0.0

    # ------------------------------------------------------------------
    def _exit(self, battle, pusher: int) -> bool:
        """保持状态是否应当退出（威胁源已消失）。"""
        triggered, _hit = trigger_directions(battle, pusher)
        if triggered:
            return False
        # 防守方该半场是否已有兵？进攻方是否还有兵在桥头/过河？
        for e in battle.entities.values():
            if not e.is_alive or _is_tower(e):
                continue
            if getattr(e, "player", None) == pusher and (_bridgehead(e) or _crossed(e)):
                return False
        return True

    # ------------------------------------------------------------------
    def pressures(self, battle):
        """返回 `(threat, my_pressure)`（pressure 量纲，与旧实现可比）。"""
        if not self.enabled:
            raise RuntimeError("PreciseThreat 未启用时不应被调用")
        # 跨局自动复位：新一局的 battle.time 会回退 ⇒ 不得沿用上一局的保持值
        t_now = float(getattr(battle, "time", 0.0))
        if t_now < self._last_time:
            self.reset()
        self._last_time = t_now
        self.stats["calls"] += 1
        trigA, halfA = trigger_directions(battle, 1)     # 敌方压我
        trigB, halfB = trigger_directions(battle, 0)     # 我压敌方

        need_sim = False
        for tag, trig in (("A", trigA), ("B", trigB)):
            if trig and self._hold[tag] is None:
                need_sim = True                          # 上升沿
            elif self._hold[tag] is not None and self._exit(battle, 1 if tag == "A" else 0):
                self._hold[tag] = None                   # 退出保持
        if need_sim:
            import time as _t
            t0 = _t.perf_counter()
            res = combine_both_directions(battle, self.horizon)
            self.stats["sim_ms"] += (_t.perf_counter() - t0) * 1e3
            self.stats["triggers"] += 1
            # 一次推演**本来就同时得到两个方向**的塔损（同一条轨迹）⇒ 两个方向的
            # 保持值一起更新。理由：① 免费；② 避免同一帧里"一个方向 HP 刻度、
            # 另一个方向计数刻度"的量纲混用。
            self._hold["A"] = res["to_p0"]
            self._hold["B"] = res["to_p1"]
            self._hold_half = {"A": halfA, "B": halfB}

        fb_t, fb_m = self._crude(battle) if (
            self._hold["A"] is None or self._hold["B"] is None) else (0.0, 0.0)
        # ⚠️ 只有 HP 来源才除以折算常量；回退值已是 pressure 刻度，**不得再除**
        threat = (self._hold["A"] / HP_PER_PRESSURE
                  if self._hold["A"] is not None else fb_t)
        mine = (self._hold["B"] / HP_PER_PRESSURE
                if self._hold["B"] is not None else fb_m)
        return (threat, mine)

    # ------------------------------------------------------------------
    @staticmethod
    def _crude(battle):
        """回退口径 = 旧 `belief_planner._enemy_pressure`（逐位同公式，便于对账）。"""
        threat = 0.0
        my_pressure = 0.0
        for e in battle.entities.values():
            if not e.is_alive or _is_tower(e):
                continue
            if getattr(e, "player", None) == 1:
                threat += 1.0 + max(0.0, (16 - float(e.position.y)) / 16.0)
            else:
                my_pressure += 1.0 + max(0.0, (float(e.position.y) - 16) / 16.0)
        return threat, my_pressure


def _selftest_symmetry():
    """轻量自检：空场 ⇒ 双向 0；单侧单位只影响对应方向。"""
    import battle as bm
    import player as pm
    from core import Position
    deck = ["Knight", "MiniPekka", "Arrows", "Minions", "Musketeer", "Fireball",
            "Giant", "Archer"]
    bs = bm.BattleState(pm.PlayerState(0, list(deck), 5.0),
                        pm.PlayerState(1, list(deck), 5.0), card_level=11)
    assert combine_both_directions(bs)["to_p0"] == 0.0
    p1 = bs.players[1]
    p1.cycle = ["Giant"] + [c for c in p1.cycle if c != "Giant"]
    p1.elixir = 10.0
    bs.deploy_card(1, "Giant", Position(14.5, 19.0))
    r = combine_both_directions(bs)
    assert r["to_p0"] > 0 and r["to_p1"] == 0.0, r
    assert math.isfinite(r["to_p0"])
    return r
