# -*- coding: utf-8 -*-
"""阵型机制探针（2026-09-19）：**改前/改后同脚本对照** + 「其余卡零改动」不变式。

三部分：
  ① 形状模板（纯函数）：偏移、两两最小间距、与「不重叠阈值 2r」之比；
  ② 实跑 A/B：**同一场景跑两遍** —— 一遍用形状表（改后），一遍把形状表**临时清空**
     （= 原均匀圆环，改前），报 tick 0 的重叠对数与两两最小间距。
     ⚠️ 基线**由脚本复算**（【R4】禁手抄）：清空 `formation.SHAPE_BY_CARD` /
     `ARRIVAL_SHAPES` 即回到旧行为，跑完立刻还原。
  ③ **不变式**：未登记形状表的卡，`get_spawn_position` 必须与「本地重算的原圆环」
     **逐值相同**（可执行的「其余卡零改动」证据）。

用法（需要 torch 的 venv）：
    .venv/Scripts/python.exe scripts/probe_formation.py
    .venv/Scripts/python.exe scripts/probe_formation.py --offline   # 只跑 ①③
"""
import argparse
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src", "clasher_new")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
from rl.io_bootstrap import force_utf8_stdout  # noqa: E402
force_utf8_stdout()

import formation  # noqa: E402

DECK0 = ["Goblins", "RoyalHogs", "GoblinBarrel", "Knight",
         "Arrows", "Fireball", "Giant", "Minions"]
DECK1 = ["Knight", "Arrows", "Fireball", "Musketeer",
         "Giant", "Minions", "MiniPekka", "Skeletons"]

CASES = [
    ("Goblins(4) 空地", "Goblins", (9.0, 12.0), {"Goblins"}, 4, 120),
    ("RoyalHogs(4) 空地", "RoyalHogs", (9.0, 12.0), {"RoyalHogs"}, 4, 200),
    ("Barrel → 敌方左公主塔正中", "GoblinBarrel", (3.5, 25.5), {"Goblins"}, 3, 400),
    ("Barrel → 空地", "GoblinBarrel", (9.0, 12.0), {"Goblins"}, 3, 200),
]


def min_pair_dist(pts):
    best = float("inf")
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            best = min(best, math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]))
    return best


def reference_ring(spawn_number, r, position, player, offset_angle=True):
    """原圆环的**本地重算**（逐字照抄 `battle.get_spawn_position` 的旧逻辑）。"""
    if spawn_number == 1:
        return [(position[0], position[1])]
    angle_offset = {2: 0, 3: math.pi / 2, 4: math.pi / 4, 6: 0}
    out = []
    for i in range(spawn_number):
        angle = 2 * math.pi * i / spawn_number
        if offset_angle:
            angle += angle_offset.get(spawn_number, 0)
        if player == 1:
            angle += math.pi
        out.append((position[0] + r * math.cos(angle), position[1] + r * math.sin(angle)))
    return out


# ------------------------------------------------------------------ ①
def part1_shapes():
    print("=" * 78)
    print("### ① 形状模板（偏移 = 格；相对出生点；玩家 0 的 +y = 前进）")
    print(f"    间距系数 SPACING_FACTOR = {formation.SPACING_FACTOR}")
    for shape, r in ((formation.SQUARE, 0.5), (formation.SQUARE, 0.6),
                     (formation.TRIANGLE, 0.5),
                     (formation.BARREL_TOWER, 0.5)):
        offs = formation.template_offsets(shape, r)
        d = min_pair_dist(offs)
        need = 2.0 * r
        print(f"  {shape:14s} r={r:.2f} n={len(offs)} 两两最小间距={d:.4f} "
              f"阈值2r={need:.2f} ⇒ {'OK' if d >= need else '重叠!'}")
        print(f"      偏移: {[(round(a, 3), round(b, 3)) for a, b in offs]}")
    print(f"  ⚠️ `barrel_tower` 用**绝对格**（锚在塔 3×3 足迹）："
          f"|y|={formation.BARREL_BEHIND_Y} ≥ 塔半高+单位半径 2.0 ⇒ 在塔推出区之外。")


# ------------------------------------------------------------------ ②
_SHAPE_LOG = []


def _measure(builder, card, world, expect, want, max_steps):
    """跑一个场景，返回读数。`_SHAPE_LOG` 记录本次实际调用到的形状名。"""
    del _SHAPE_LOG[:]
    from battle import Position  # noqa: PLC0415
    env = builder()
    env.reset()
    bst = env.battle
    p = bst.players[0]
    p.elixir = 10.0
    if card in p.cycle:
        p.cycle.insert(0, p.cycle.pop(p.cycle.index(card)))
    else:
        p.cycle[0] = card
    import formation as _f  # noqa: PLC0415
    _orig_sfc, _orig_af = _f.shape_for_card, _f.arrival_shape

    def _log_sfc(name, count):
        sh = _orig_sfc(name, count)
        _SHAPE_LOG.append(("card", name, count, sh))
        return sh

    def _log_af(name, count, position, battle_state):
        sh = _orig_af(name, count, position, battle_state)
        _SHAPE_LOG.append(("arrival", name, count, sh))
        return sh

    _f.shape_for_card, _f.arrival_shape = _log_sfc, _log_af
    try:
        ok = bst._deploy_card_impl(0, card, Position(*world))
    finally:
        _f.shape_for_card, _f.arrival_shape = _orig_sfc, _orig_af
    if not ok:
        return None
    spawned = []
    for _ in range(max_steps):
        bst.step(1 / 60.0)
        spawned = [e for e in bst.entities.values()
                   if e.player == 0 and str(getattr(e, "name", "")) in expect]
        if len(spawned) >= want:
            break
    if not spawned:
        return None
    pts = [(round(e.position.x, 4), round(e.position.y, 4)) for e in spawned]
    rs = [float(e.data.collision_radius) for e in spawned]
    ov = sum(1 for i in range(len(spawned)) for j in range(i + 1, len(spawned))
             if spawned[i].position.distance_to(spawned[j].position) < rs[i] + rs[j])
    return {"n": len(spawned), "ov": ov, "d_min": min_pair_dist(pts), "pts": pts,
            "r_max": max(rs), "shapes": list(_SHAPE_LOG)}


def part2_live():
    print("=" * 78)
    print("### ② 实跑 A/B：形状表 ON（改后） vs 形状表清空（改前 = 原圆环）")

    def builder():
        from rl.env_wrapper import RLEnv  # noqa: PLC0415
        return RLEnv(opponent=lambda obs: None, seed=0, card_level=11, decision_frames=1,
                     deck0=list(DECK0), deck1=list(DECK1))

    for label, card, world, expect, want, max_steps in CASES:
        saved = (formation.SHAPE_BY_CARD, formation.ARRIVAL_SHAPES)
        # 顺序固定：先「形状表 ON」再「清空形状表」，各自记录实际选中的形状名
        res_on = _measure(builder, card, world, expect, want, max_steps)
        formation.SHAPE_BY_CARD, formation.ARRIVAL_SHAPES = {}, {}
        try:
            res_off = _measure(builder, card, world, expect, want, max_steps)
        finally:
            formation.SHAPE_BY_CARD, formation.ARRIVAL_SHAPES = saved
        print(f"\n  [{label}]")
        for tag, res in (("形状表 ON （改后）", res_on),
                         ("形状表清空（改前=原圆环）", res_off)):
            if res is None:
                print(f"    {tag}: 未观测到出生单位")
                continue
            print(f"    {tag}: n={res['n']}  重叠对={res['ov']}  "
                  f"两两最小间距={res['d_min']:.4f}  2r={2 * res['r_max']:.2f}")
            print(f"        实际选中形状 {res['shapes']}")
            print(f"        坐标 {res['pts']}")
        if res_on and res_off:
            print(f"    ⇒ 重叠对 改前 {res_off['ov']} → 改后 {res_on['ov']}；"
                  f"最小间距 改前 {res_off['d_min']:.4f} → 改后 {res_on['d_min']:.4f}")


# ------------------------------------------------------------------ ③
def part3_invariant():
    print("=" * 78)
    print("### ③ 不变式：未登记形状表的卡，出生几何必须与本地重算的原圆环**逐值相同**")
    import battle                # noqa: PLC0415
    from card_utils import Card  # noqa: PLC0415
    pos = (9.0, 12.0)
    checked = bad = 0
    for name in ("SkeletonArmy", "Minions", "Bats", "Barbarians", "SpearGoblins",
                 "ThreeMusketeers", "Giant", "Knight", "Wallbreakers", "SkeletonDragons",
                 "RoyalRecruits", "MiniSparkys"):
        c = Card(name)
        got = battle.get_spawn_position(c, battle.Position(*pos), 0)
        exp = reference_ring(c.spawn_number, c.spawn_radius, pos, 0)
        same = (len(got) == len(exp) and
                all(abs(g.x - ex) < 1e-12 and abs(g.y - ey) < 1e-12
                    for g, (ex, ey) in zip(got, exp)))
        checked += 1
        if not same:
            bad += 1
            print(f"  [BAD] {name}: n={c.spawn_number} 与参照不同")
    print(f"  ⇒ 未登记卡 {checked} 张："
          f"{'全部逐值相同 ✅' if bad == 0 else f'{bad} 张不一致 ❌'}")
    g = Card("Goblins")
    got = battle.get_spawn_position(g, battle.Position(*pos), 0)
    ring = reference_ring(g.spawn_number, g.spawn_radius, pos, 0)
    changed = any(abs(a.x - ex) > 1e-9 or abs(a.y - ey) > 1e-9
                  for a, (ex, ey) in zip(got, ring))
    print(f"  ⇒ 登记卡 `Goblins`(n={g.spawn_number})："
          f"{'与原圆环不同 ✅（表已生效）' if changed else '与原圆环相同 ❌（表没生效）'}")
    return bad == 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="只跑 ①③")
    args = ap.parse_args(argv)
    part1_shapes()
    ok = part3_invariant()
    if not args.offline:
        part2_live()
    print("=" * 78)
    print("判读：① 的间距必须 ≥ 2r；② 的「重叠对」必须下降且不重叠；③ 必须全部逐值相同。")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
