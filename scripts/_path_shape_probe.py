# -*- coding: utf-8 -*-
"""O3 诊断：把我方引擎里骷髅军团某只的 **A\* 路径** 逐点打出来。

问题：实测我方单位在己方半场沿 `x = 6.00` 一条**垂直线**走到 y≈13.4 才横移到桥，
而上游是**持续**向桥心收拢（7.0→5.5→4.8→4.3）。要判定「dogleg 是 A* 路径本身就长这样，
还是跟随路径的方式造成的」——本脚本直接打印路径点、goal、当前路点。

只读诊断：不写任何文件（除 stdout）。必须用带 torch 的 venv：
    .venv/Scripts/python.exe scripts/_path_shape_probe.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src", "clasher_new")
sys.path.insert(0, SRC)
os.chdir(SRC)
from rl.io_bootstrap import force_utf8_stdout  # noqa: E402
force_utf8_stdout()

from rl.env_wrapper import RLEnv                       # noqa: E402
from rl.action_bundle import ActionBundle, SubAction   # noqa: E402
from battle import Position                            # noqa: E402
from pathfinding_heap import EntityPathfinder, position_to_cell  # noqa: E402

DECK0 = ["SkeletonArmy", "Knight", "Arrows", "Fireball",
         "Musketeer", "Giant", "Minions", "MiniPekka"]
DECK1 = ["Knight", "Arrows", "Fireball", "Musketeer",
         "Giant", "Minions", "MiniPekka", "Skeletons"]

#: 打印路径的决策步（每步 6 tick = 0.1 s）
SHOW_AT = {5, 20, 50, 90, 120}


def skels(bst):
    return sorted([e for e in bst.entities.values()
                   if str(getattr(e, "name", "")).startswith("SkeletonArmy")
                   and e.player == 0 and e.is_alive], key=lambda e: e.id)


def show_path(env, tag):
    bst = env.battle
    es = skels(bst)
    if not es:
        print(f"  [{tag}] 无骷髅")
        return
    print(f"  [{tag}] 存活 {len(es)} 只")
    for e in es[:2] + es[-1:]:
        tgt = getattr(e, "current_target", None)
        path = getattr(e, "path", None) or []
        pts = [(round(p.x, 2), round(p.y, 2)) for p in path]
        thin = pts  # 全量打印（≤60 点）
        pf = EntityPathfinder(e, tgt, bst) if tgt is not None else None
        goal = None
        if pf is not None:
            try:
                pf.calculate()
                goal = (round(pf.goal[0] / 2 + 0.25, 2), round(pf.goal[1] / 2 + 0.25, 2)) \
                    if pf.goal else None
            except Exception as exc:  # noqa: BLE001
                goal = f"ERR {exc}"
        print(f"    id={e.id:3d} pos=({e.position.x:5.2f},{e.position.y:5.2f}) "
              f"lane_off={getattr(e, '_lane_offset', None)}")
        print(f"          target={getattr(tgt, 'name', None)} "
              f"tgt_pos=({getattr(getattr(tgt, 'position', None), 'x', -1):5.2f},"
              f"{getattr(getattr(tgt, 'position', None), 'y', -1):5.2f})  "
              f"goal={goal}")
        print(f"          path({len(pts)} 点，抽样 {len(thin)}): {thin}")


def main():
    env = RLEnv(opponent=lambda obs: ActionBundle([]), seed=0, card_level=11,
                decision_frames=6, deck0=list(DECK0), deck1=list(DECK1))
    obs, _ = env.reset(seed=100)
    cyc = list(env.battle.players[0].cycle)
    cyc.remove("SkeletonArmy")
    cyc.insert(0, "SkeletonArmy")
    env.battle.players[0].cycle = cyc

    for step in range(0, 140):
        bundle = (ActionBundle([SubAction(kind="deploy", slot=1, x=9, y=0)])
                  if step == 0 else ActionBundle([]))
        obs, reward, term, trunc, info = env.step(bundle)
        if step in SHOW_AT:
            show_path(env, f"step {step} (t={step*0.1:.1f}s)")
        if term or trunc:
            print(f"  结束于 step {step}")
            break
    # 桥带可走性 + **格字符/代价** 快照：判定 dogleg 是代价表造成的还是别的
    bst = env.battle
    from pathfinding_heap import contents, position_to_cell  # noqa: PLC0415
    print("\n格字符快照  y 自上而下 = 17→8，x 自左而右 = 1.75→7.25（每 0.5 格一列）")
    print("  图例：'.'/'_'=普通地面(代价5)  'W'=800(地面单位)  '#'=A* 不可走")
    for yi in range(34, 15, -1):
        y = (yi + 0.5) / 2
        row = []
        for xi in range(3, 15):
            x = (xi + 0.5) / 2
            if not bst.pathfind_ground_walkable(Position(x, y), 0.5):
                row.append("#")
                continue
            ch = contents[63 - yi][xi]
            row.append(ch if ch in ".W" else "_")
        print(f"  y={y:5.2f} {''.join(row)}")
    print("\n同一区域的 tile_char 与计算代价（只看 x∈[3.75,6.75]）")
    for yi in range(20, 30):
        y = (yi + 0.5) / 2
        cells = []
        for xi in range(7, 14):
            x = (xi + 0.5) / 2
            ch = contents[63 - yi][xi]
            cost = 800 if ch == 'W' else (5 if ch in "._" else 5)
            cells.append(f"x{x:.2f}:{ch}{cost}")
        print(f"  y={y:5.2f}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
