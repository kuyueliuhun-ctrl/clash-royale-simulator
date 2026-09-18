# -*- coding: utf-8 -*-
"""引擎每 tick 成本的**族归类**剖分（只读仪器）。

回答「能不能靠优化寻路把成本降下来」：把 `BattleState.step` 的耗时按族聚合，
并给出「若某族快 X 倍，整体能省多少」——这是判断优化是否够用的唯一诚实方式。

用法（必须在 src/clasher_new 下运行）：
    cd src/clasher_new && PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe \
        ../../scripts/probe_engine_profile.py
"""
from __future__ import annotations

import cProfile
import copy
import io
import os
import pstats
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src", "clasher_new")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from rl.io_bootstrap import force_utf8_stdout  # noqa: E402  (T1-1: UTF-8 兜底单一来源)
force_utf8_stdout()

import numpy as np  # noqa: E402

#: 族归类：函数名（pstats 的 func 字段）→ 族
FAMILY = {
    # —— 寻路族 ——
    "calculate": "pathfinding",                    # pathfinding_heap.EntityPathfinder.calculate
    "pathfind_ground_walkable": "pathfinding",
    "ground_walkable": "pathfinding",
    "position_to_cell": "pathfinding",
    "cell_to_position": "pathfinding",
    "heuristic": "pathfinding",
    "get_neighboring_points": "pathfinding",
    # —— 索敌族 ——
    "get_nearest_target": "targeting",
    "update_current_target": "targeting",
    "in_sight_range": "targeting",
    "edge_distance_from": "targeting",
    "distance_to": "targeting",
    "dist_to_rect": "targeting",
    "tower_rect_dist": "targeting",
    # —— 碰撞族 ——
    "resolve_collisions": "collision",
    "_push_troop_out": "collision",
    "_push_troop": "collision",
}
FILES = {"pathfinding_heap.py": "pathfinding"}


def build(n_units, deck=None):
    import battle as bm
    import player as pm
    from core import Position
    deck = deck or ["Knight", "MiniPekka", "Arrows", "Minions", "Musketeer",
                    "Fireball", "Giant", "Archer"]
    bs = bm.BattleState(pm.PlayerState(0, list(deck), 5.0),
                        pm.PlayerState(1, list(deck), 5.0), card_level=11)
    cards = ["Knight", "Archer", "Giant", "Musketeer", "Minions", "MiniPekka"]
    for i in range(n_units):
        pl = i % 2
        p = bs.players[pl]
        c = cards[i % len(cards)]
        p.cycle = [c] + [x for x in p.cycle if x != c]
        p.elixir = 10.0
        if pl == 0:
            pos = Position(3.0 + 3 * (i % 6), 11.0 + 3 * (i // 6))
        else:
            pos = Position(15.0 - 3 * (i % 6), 21.0 + 3 * (i // 6))
        bs.deploy_card(pl, c, pos)
    return bs


def profile_board(name, bs, ticks):
    def run():
        s = copy.deepcopy(bs)
        for _ in range(ticks):
            s.step(1 / 60)

    t0 = time.perf_counter()
    run()
    wall = (time.perf_counter() - t0) * 1e3
    ne = sum(1 for e in bs.entities.values() if e.is_alive)
    pr = cProfile.Profile()
    pr.enable(); run(); pr.disable()
    st = pstats.Stats(pr)

    fam_tt = {}
    fam_ct = {}
    fam_calls = {}
    total_ct = 0.0
    for (fn, _ln, func), (cc, nc, tt, ct, _callers) in st.stats.items():
        base = os.path.basename(fn)
        f = FILES.get(base) or FAMILY.get(func)
        if func == "step" and base == "battle.py":
            total_ct = ct
        if f is None:
            continue
        fam_tt[f] = fam_tt.get(f, 0.0) + tt
        fam_ct[f] = fam_ct.get(f, 0.0) + ct
        fam_calls[f] = fam_calls.get(f, 0) + nc
    tot_tt = sum(v[2] for v in st.stats.values())
    print(f"\n  [{name}] 实体={ne} tick={ticks} | 无 profiler 墙钟={wall:.1f} ms "
          f"=> **{wall / ticks:.4f} ms/tick**")
    print(f"      全程序 tottime 合计={tot_tt:.3f}s；"
          f"`BattleState.step` cumtime={total_ct:.3f}s")
    print(f"      {'族':12s} {'tottime(s)':>11s} {'占全程序':>9s} {'cumtime(s)':>11s} "
          f"{'占 step':>8s} {'调用次数':>10s} {'快10×后省':>10s}")
    for f in sorted(fam_tt, key=lambda k: -fam_tt[k]):
        # 保守口径：族内函数可能互相调用（cum 会重复计），故"省多少"用 tottime 估
        save = fam_tt[f] * 0.9 / tot_tt if tot_tt else 0.0
        print(f"      {f:12s} {fam_tt[f]:11.3f} {fam_tt[f] / tot_tt:9.1%} "
              f"{fam_ct[f]:11.3f} "
              f"{(fam_ct[f] / total_ct if total_ct else float('nan')):8.1%} "
              f"{fam_calls[f]:10d} {save:10.1%}")
    return fam_tt, tot_tt, wall / ticks


def main():
    ticks = 200
    print("=" * 78)
    print("§1 引擎 step 的族归类剖分（cProfile；tottime 口径，族内 cum 会重复计）")
    res = {}
    for name, nu in (("稀疏盘面（8 单位）", 8), ("中局盘面（24 单位）", 24)):
        res[name] = profile_board(name, build(nu), ticks)

    print("\n" + "=" * 78)
    print("§2 关键判据：优化寻路**够不够**把成本降到预算内")
    print("    需求（P3-1 实测）：精确塔伤每帧调用 = 1200 tick，"
          "而预算 = 一帧的 10% = 3 tick 的等价成本")
    for name, (_ft, _tt, per_tick) in res.items():
        print(f"    {name}: 单 tick {per_tick:.4f} ms ⇒ 工具 H=20s ≈ "
              f"{1200 * per_tick:.0f} ms = {1200 * per_tick / 38.5:.1f} 个训练帧")
    print("    ⇒ 要让每帧精确推演进预算，需要 tick 成本降 **40 倍以上**；"
          "寻路族只占约 1/3 ⇒ 即使寻路快 ∞ 倍也只降到 ~2/3")


if __name__ == "__main__":
    main()
