# -*- coding: utf-8 -*-
"""出兵环「爆炸」探针（只读诊断，不写任何文件）：SkeletonArmy 部署后逐 tick 的真实位置。

动机：`get_spawn_position`（battle.py:2610）给出半径 `spawn_radius` 的**圆环**，实测
`Card('SkeletonArmy').spawn_radius = 0.55`；但录像第一帧（t=0.1 s）里 15 只已经摊到
半径 ~2 格。本脚本把部署后**每个 tick** 的位置打出来，判定这 4 倍膨胀发生在哪一步、
由什么产生（候选：`resolve_collisions` 把重叠的单位推开）。

跑法（必须用带 torch 的 venv，RLEnv 依赖）：
    .venv/Scripts/python.exe scripts/_skarmy_spawn_probe.py
"""
import os
import sys
import math

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src", "clasher_new")
sys.path.insert(0, SRC)
os.chdir(SRC)
from rl.io_bootstrap import force_utf8_stdout  # noqa: E402  (T1-1: UTF-8 兜底单一来源)
force_utf8_stdout()

from rl.env_wrapper import RLEnv          # noqa: E402
from battle import Position, get_spawn_position  # noqa: E402
from card_utils import Card               # noqa: E402

DECK0 = ["SkeletonArmy", "Knight", "Arrows", "Fireball",
         "Musketeer", "Giant", "Minions", "MiniPekka"]
DECK1 = ["Knight", "Arrows", "Fireball", "Musketeer",
         "Giant", "Minions", "MiniPekka", "Skeletons"]

PLACEMENTS = [("bridge_left", 3.5, 13.5), ("behind_king", 9.5, 0.5)]
DT = 1.0 / 60.0


def skels(bst):
    return [e for e in bst.entities.values()
            if str(getattr(e, "name", "")) .startswith("Skeleton")
            and e.player == 0 and e.is_alive]


def summary(tag, bst):
    es = skels(bst)
    if not es:
        print(f"    {tag}: 无骷髅")
        return
    cx = sum(e.position.x for e in es) / len(es)
    cy = sum(e.position.y for e in es) / len(es)
    rs = sorted(math.hypot(e.position.x - cx, e.position.y - cy) for e in es)
    xs = sorted(e.position.x for e in es)
    ys = sorted(e.position.y for e in es)
    # 重叠对数（判定「是否还挤在一起」）
    n_ov = 0
    for i in range(len(es)):
        for j in range(i + 1, len(es)):
            d = es[i].position.distance_to(es[j].position)
            if d < es[i].data.collision_radius + es[j].data.collision_radius:
                n_ov += 1
    print(f"    {tag}: n={len(es):2d}  x=[{xs[0]:5.2f},{xs[-1]:5.2f}] "
          f"y=[{ys[0]:5.2f},{ys[-1]:5.2f}]  半径 中位={rs[len(rs)//2]:5.2f} 最大={rs[-1]:5.2f}  "
          f"重叠对={n_ov:3d}  质心=({cx:5.2f},{cy:5.2f})")


def main():
    c = Card("SkeletonArmy")
    print(f"[card] N={c.spawn_number}  spawn_radius={c.spawn_radius}  "
          f"collision_radius={c.collision_radius}  summonRadius键="
          f"{'summonRadius' in getattr(c, 'data', {})}")
    for name, cx, cy in PLACEMENTS:
        ring = get_spawn_position(c, Position(cx, cy), 0)
        rr = [math.hypot(p.x - cx, p.y - cy) for p in ring]
        print(f"\n=== {name} 部署点世界({cx},{cy})  "
              f"get_spawn_position 半径={min(rr):.2f}~{max(rr):.2f} ===")
        env = RLEnv(deck0=DECK0, deck1=DECK1, opponent=None)
        env.reset()
        bst = env.battle
        # 把 SkeletonArmy 塞进手牌首位并给足圣水（否则 can_play_card 直接拒）
        p = bst.players[0]
        p.elixir = 10.0
        if "SkeletonArmy" in p.cycle:
            p.cycle.insert(0, p.cycle.pop(p.cycle.index("SkeletonArmy")))
        print(f"  cycle[:4]={p.cycle[:4]}  elixir={p.elixir}  "
              f"can_play={p.can_play_card('SkeletonArmy')}")
        ok = bst._deploy_card_impl(0, "SkeletonArmy", Position(cx, cy))
        print(f"  _deploy_card_impl -> {ok}")
        summary("tick 0 (部署即刻，未推进)", bst)
        for k in range(1, 13):
            bst.step(DT)
            if k in (1, 2, 3, 6, 12) or k > 6:
                summary(f"tick {k} (t={k*DT:.3f}s)", bst)


if __name__ == "__main__":
    main()
