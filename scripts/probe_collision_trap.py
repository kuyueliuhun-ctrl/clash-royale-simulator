# -*- coding: utf-8 -*-
"""卡位第 6 轮：**更正朝向** —— 女巫 = p0（王塔后 → +y 走），石头人 = p1（放在她**前方**）。

用户更正（2026-09-19 原文）：「不，暗夜女巫的 y 应该比石头人小，部署时是石头人放在暗夜女巫
**前一点**，如果按照圆形建模，卡住女巫的方式是，用**圆与正方形之间的后方空隙**卡住女巫。」

⇒ 我前五轮把朝向搞反了：我让女巫当 **p1**、从 y=31.5 **往下**走，于是她始终在 y **较大**的一侧。
按本轮口径：
  · **女巫 = p0**，部署在**己方王塔后** `(9.5, 0.5)`，朝 **+y** 前进（p0 的前进方向）；
  · 她逼近 **p1 的公主塔**（`(14.5, 25.5)`，矩形半宽/半高 1.5 ⇒ 足迹 `x∈[13,16], y∈[24,27]`）时，
    **石头人 = p1** 放在她**正前方 1.0 格** ⇒ 必然 `y_女巫 < y_石头人` ✓（你的口径）；
  · 同时报**石头人圆 ↔ 塔矩形的最近距离**（「圆与正方形之间的空隙」是否真的形成）。

质量模型同 §4.7：`m = r^k / speed`，k=0 逐值退化 = 现状。k ∈ {0, 2, 3}。

用法：
    .venv/Scripts/python.exe scripts/probe_collision_trap.py
"""
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src", "clasher_new")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
from rl.io_bootstrap import force_utf8_stdout  # noqa: E402
force_utf8_stdout()

DT = 1.0 / 60.0
WITCH_SPAWN = (9.5, 0.5)          # p0 王塔后（本脚本口径）
TRIGGER_Y = 21.0                  # 她逼近 p1 右公主塔（y=25.5）时投放石头人
GOLEM_AHEAD = 1.0
WINDOW_SECONDS = 20.0
CONTACT_SEP = 1.30
MAXT = 9000
KS = (0, 2, 3)


def _mass(e, k):
    r = max(float(e.data.collision_radius), 1e-6)
    s = max(float(e.data.speed), 1e-6)
    return (r ** k) / s


def make_resolver(k, pair_acc):
    """`battle.py:3291-3320` 的逐字拷贝，只把「谁被推多少」换成按质量反比分配。"""
    def resolve_collisions(self):
        from battle import Troop, Building  # noqa: PLC0415
        from itertools import combinations  # noqa: PLC0415
        al = [e for e in self.entities.values()
              if e.is_alive and (isinstance(e, Troop) or isinstance(e, Building))]
        groups = (combinations([e for e in al if not e.data.is_air_unit
                                and getattr(e.entity_holder, '_mk_jump', None) is None], 2),
                  combinations([e for e in al if e.data.is_air_unit], 2))
        for troop in groups:
            for e1, e2 in troop:
                if e1.id <= 6 and getattr(e1, 'persistent', False):
                    self._push_troop_out_of_tower(e2, e1); continue
                if e2.id <= 6 and getattr(e2, 'persistent', False):
                    self._push_troop_out_of_tower(e1, e2); continue
                if e1.position.distance_to(e2.position) < e1.data.collision_radius + e2.data.collision_radius:
                    overlap = e1.data.collision_radius + e2.data.collision_radius - e1.position.distance_to(e2.position)
                    dv = complex(e2.position.x-e1.position.x, e2.position.y-e1.position.y)
                    if abs(dv) == 0: return
                    dv /= abs(dv)
                    if e1.data.speed + e2.data.speed == 0: continue
                    m1, m2 = _mass(e1, k), _mass(e2, k)
                    s2, s1 = m1 / (m1 + m2), m2 / (m1 + m2)
                    p1 = (e1.position.x, e1.position.y); p2 = (e2.position.x, e2.position.y)
                    e2.position.x += dv.real * s2 * overlap
                    e2.position.y += dv.imag * s2 * overlap
                    e1.position.x += -dv.real * s1 * overlap
                    e1.position.y += -dv.imag * s1 * overlap
                    key = (min(e1.id, e2.id), max(e1.id, e2.id))
                    d1 = ((e1.position.x-p1[0])**2 + (e1.position.y-p1[1])**2) ** 0.5
                    d2 = ((e2.position.x-p2[0])**2 + (e2.position.y-p2[1])**2) ** 0.5
                    a = pair_acc.setdefault(key, [0.0, 0.0])
                    if e1.id <= e2.id:
                        a[0] += d1; a[1] += d2
                    else:
                        a[0] += d2; a[1] += d1

    return resolve_collisions


def _in_hand(bst, player, card):
    p = bst.players[player]
    p.elixir = 10.0
    if card in p.cycle:
        p.cycle.insert(0, p.cycle.pop(p.cycle.index(card)))
    else:
        p.cycle[0] = card


def _find(bst, name, player):
    for e in bst.entities.values():
        if str(getattr(e, "name", "")) == name and e.player == player and e.is_alive:
            return e
    return None


def _rect_gap(pos, center, hw, hh):
    dx = max(abs(pos.x - center.x) - hw, 0.0)
    dy = max(abs(pos.y - center.y) - hh, 0.0)
    return (dx * dx + dy * dy) ** 0.5


def run_trap(k):
    from rl.env_wrapper import RLEnv      # noqa: PLC0415
    from battle import Position           # noqa: PLC0415
    deck = ["DarkWitch", "Knight", "Arrows", "Fireball",
            "Golem", "Minions", "Musketeer", "Giant"]
    env = RLEnv(opponent=lambda obs: None, seed=0, card_level=11, decision_frames=1,
                deck0=list(deck), deck1=list(deck))
    env.reset()
    bst = env.battle
    pair_acc = {}
    bst.resolve_collisions = types.MethodType(make_resolver(k, pair_acc), bst)

    _in_hand(bst, 0, "DarkWitch")                 # 女巫 = p0
    if not bst._deploy_card_impl(0, "DarkWitch", Position(*WITCH_SPAWN)):
        return {"error": "女巫部署失败"}

    rows, golem, trigger, wit_id, tower = [], None, None, None, None
    for t in bst.arena.towers:                    # p1 的右公主塔
        if t[3] == 1 and abs(t[0].x - 14.5) < 0.1 and abs(t[0].y - 25.5) < 0.1:
            tower = t
    for tick in range(1, MAXT + 1):
        bst.step(DT)
        w = _find(bst, "DarkWitch", 0)
        if w is None:
            break
        wit_id = w.id
        if trigger is None and w.position.y >= TRIGGER_Y:
            trigger = tick
            gx, gy = w.position.x, w.position.y + GOLEM_AHEAD   # **她前方**（y 更大）
            if not bst._deploy_card_impl(1, "Golem", Position(gx, gy)):
                return {"error": f"石人部署失败 @({gx:.2f},{gy:.2f})"}
            golem = _find(bst, "Golem", 1)
        gy_ = golem.position.y if (golem is not None and golem.is_alive) else None
        rows.append((tick, w.position.y, w.position.x, gy_))
        if trigger is not None and (tick - trigger) > WINDOW_SECONDS * 60:
            break

    if trigger is None:
        return {"error": f"女巫从未到达触发线 y>={TRIGGER_Y}"}
    win = [r for r in rows if r[0] >= trigger]
    if len(win) < 5:
        return {"error": "窗口过短"}

    flips = front = 0
    prev = None
    for r in win:
        if r[3] is None:
            continue
        s = 1 if r[1] < r[3] else -1        # 她的 y 比石头人**小** = 用户口径的正确构型
        if s < 0:
            front += 1
        if prev is not None and s != prev:
            flips += 1
        prev = s
    contact = [r for r in win if r[3] is not None and abs(r[1] - r[3]) <= CONTACT_SEP]
    push_g = push_w = 0.0
    if golem is not None and wit_id is not None:
        key = (min(golem.id, wit_id), max(golem.id, wit_id))
        if key in pair_acc:
            a = pair_acc[key]
            push_g = a[0] if golem.id <= wit_id else a[1]
            push_w = a[1] if golem.id <= wit_id else a[0]
    tinfo = None
    if tower is not None and golem is not None:
        tinfo = round(_rect_gap(golem.position, tower[0], tower[1], tower[2]), 3)
    return {
        "trigger": trigger, "window_s": round(len(win) * DT, 2),
        "contact_ticks": len(contact),
        "min_sep": round(min(abs(r[1]-r[3]) for r in win if r[3] is not None), 3),
        "y_witch_lt_golem_flips": flips, "witch_beyond_golem_ticks": front,
        "advance_window": round(win[-1][1] - win[0][1], 3),     # **正 = 她往 +y 前进**
        "push_golem": round(push_g, 3), "push_witch": round(push_w, 3),
        "golem_at_tower_gap": tinfo,
        "y_series": [(round(r[0]*DT, 1), round(r[1], 1), None if r[3] is None else round(r[3], 1))
                     for r in win[::240]],
    }


def main():
    print("=" * 100)
    print("### 第 6 轮：更正朝向（女巫 = p0 王塔后 → +y；石头人 = p1 放在她前方 1.0 格）")
    print(f"    触发线：她的 y ≥ {TRIGGER_Y}（p1 右公主塔中心 y=25.5，足迹 y∈[24,27]）")
    for k in KS:
        r = run_trap(k)
        print("=" * 100)
        print(f"### k={k}")
        if "error" in r:
            print(f"  ⚠️ {r['error']}")
            continue
        print(f"  接触 {r['contact_ticks']} tick / 窗口 {r['window_s']} s / min_sep={r['min_sep']}")
        print(f"  **她的 y < 石头人的 y**（= 你的口径）：翻转 {r['y_witch_lt_golem_flips']} 次；"
              f"她**跑到石头人外侧**的 tick = {r['witch_beyond_golem_ticks']}")
        print(f"  她的净前进（正 = 往 +y 走） = **{r['advance_window']} 格**；"
              f"push石人={r['push_golem']} push女巫={r['push_witch']}")
        print(f"  石头人圆 ↔ 公主塔矩形 最近距离 = {r['golem_at_tower_gap']} 格"
              f"（0 = 贴着塔；「圆与正方形之间的空隙」需要它足够小）")
        print(f"  逐 4 s (t, y女巫, y石人): {r['y_series']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
