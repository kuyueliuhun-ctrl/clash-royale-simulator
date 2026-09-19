# -*- coding: utf-8 -*-
"""卡位实验 第 5 轮：石头人部署之后，**长时程**里女巫是在**推着它走**还是**被挡住**？
（执行预注册 §4.9）

用户问（原文）：「我们再做一个测试，石头人部署之后，**两种 k 值能否保持暗夜女巫推着石头人
一直走**，即**暗夜女巫一直在石头人之后**？」

**为什么单独一轮**：第 4 轮窗口 3.0 s **完全落在石人的 `deploy_time` 冻结期内** ⇒ 那时石人
**不能自己走**，只测到了「静止的墙」。冻结期一过石人开始自己前进 ⇒ 格局可能完全不同
⇒ 本轮用 **20 s 长窗口**（含冻结期之后）。

判据（§4.9，跑前写死）：
  ① 顺序：`order_flips == 0` ⇒ 「她一直在石头人之后」
  ② 她在推还是被挡：`advance_window ≤ −3.0 格` **且** `push_golem ≥ 0.5 格` ⇒ 「推着石头人走」；
     否则 ⇒ 「被挡住」。
  对照：阳性（contact ≥ 30、min_sep ≤ 1.30）；阴性（不放石人 20 s 内她净前进 ≥ 8.0 格）。

k = 0（现状）・2・3・6（「两种 k 值」= 2 与 3；两端作参照）。质量模型同 §4.7：
`m = r^k / speed`，**k=0 逐值退化为现状**。

用法：
    .venv/Scripts/python.exe scripts/probe_collision_order.py
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
WITCH_SPAWN = (9.5, 31.5)
TRIGGER_Y = 14.0
GOLEM_AHEAD = 1.0
WINDOW_SECONDS = 20.0
CONTACT_SEP = 1.30
MAXT = 7200
CRIT_CONTACT_TICKS = 30
CRIT_PUSH_WALK = -3.0
CRIT_PUSH_GOLEM = 0.5
CRIT_NOGOLEM_ADVANCE = 8.0
KS = (0, 2, 3, 6)


def _mass(e, k):
    r = max(float(e.data.collision_radius), 1e-6)
    s = max(float(e.data.speed), 1e-6)
    return (r ** k) / s


def make_resolver(k, pair_acc):
    """`battle.py:3291-3320` 的逐字拷贝，只把「谁被推多少」换成按质量反比分配（同第 4 轮）。"""
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


def run_long(k, use_golem=True):
    from rl.env_wrapper import RLEnv      # noqa: PLC0415
    from battle import Position           # noqa: PLC0415
    deck = ["DarkWitch", "Knight", "Arrows", "Fireball",
            "Golem", "Minions", "Musketeer", "Giant"]
    env = RLEnv(opponent=lambda obs: None, seed=0, card_level=11, decision_frames=1,
                deck0=list(deck), deck1=list(deck))
    env.reset()
    bst = env.battle
    pair_acc = {}
    if k is not None:
        bst.resolve_collisions = types.MethodType(make_resolver(k, pair_acc), bst)

    _in_hand(bst, 1, "DarkWitch")
    if not bst._deploy_card_impl(1, "DarkWitch", Position(*WITCH_SPAWN)):
        return {"error": "女巫部署失败"}

    rows, golem, trigger, wit_id, first_gy = [], None, None, None, None
    for tick in range(1, MAXT + 1):
        bst.step(DT)
        w = _find(bst, "DarkWitch", 1)
        if w is None:
            break
        wit_id = w.id
        if trigger is None and w.position.y <= TRIGGER_Y:
            trigger = tick
            if use_golem:
                gx, gy = w.position.x, w.position.y - GOLEM_AHEAD
                if not bst._deploy_card_impl(0, "Golem", Position(gx, gy)):
                    return {"error": "石人部署失败"}
                golem = _find(bst, "Golem", 0)
        gy = golem.position.y if (golem is not None and golem.is_alive) else None
        if gy is not None and first_gy is None:
            first_gy = gy
        rows.append((tick, w.position.y, w.position.x, gy))
        if trigger is not None and (tick - trigger) > WINDOW_SECONDS * 60:
            break

    if trigger is None:
        return {"error": f"女巫从未到达触发线 y<={TRIGGER_Y}"}
    win = [r for r in rows if r[0] >= trigger]
    if len(win) < 5:
        return {"error": "窗口过短"}

    flips = front = 0
    prev_sign = None
    for r in win:
        if r[3] is None:
            continue
        s = 1 if r[1] > r[3] else -1
        if s < 0:
            front += 1
        if prev_sign is not None and s != prev_sign:
            flips += 1
        prev_sign = s

    seps = [abs(r[1] - r[3]) for r in win if r[3] is not None]
    contact = [r for r in win if r[3] is not None and abs(r[1] - r[3]) <= CONTACT_SEP]
    push_g = 0.0
    if golem is not None and wit_id is not None:
        key = (min(golem.id, wit_id), max(golem.id, wit_id))
        if key in pair_acc:
            a = pair_acc[key]
            push_g = a[0] if golem.id <= wit_id else a[1]
    last_gy = next((r[3] for r in reversed(win) if r[3] is not None), None)
    return {
        "trigger": trigger, "window_ticks": len(win), "window_s": round(len(win) * DT, 2),
        "contact_ticks": len(contact),
        "min_sep": round(min(seps), 3) if seps else None,
        "order_flips": flips, "witch_front_ticks": front,
        "advance_window": round(win[0][1] - win[-1][1], 3),
        "push_golem": round(push_g, 3),
        "golem_net_dy": (round(last_gy - first_gy, 3) if (last_gy is not None and first_gy is not None) else None),
        "y_series": [(round(r[0] * DT, 1), round(r[1], 1), None if r[3] is None else round(r[3], 1))
                     for r in win[::120]],
    }


def main():
    print("=" * 100)
    print("### 第 5 轮：长时程（20 s，含 deploy_time 冻结期之后）——她在推石头人，还是被挡住？")
    rows = [(f"k={k}", run_long(k)) for k in KS]
    nogo = run_long(0, use_golem=False)
    print(f"\n  阴性对照（不放石人）：20 s 内她净前进 {nogo.get('advance_window')} 格 "
          f"(需 ≥ {CRIT_NOGOLEM_ADVANCE})")
    print(f"\n{'臂':8}{'contact':>8}{'顺序翻转':>9}{'她在前的tick':>13}{'她的净前进':>12}"
          f"{'push石人':>10}{'石人净dy':>10}{'min_sep':>9}")
    for lbl, r in rows:
        if "error" in r:
            print(f"{lbl:8}  ⚠️ {r['error']}")
            continue
        print(f"{lbl:8}{r['contact_ticks']:>8}{r['order_flips']:>9}{r['witch_front_ticks']:>13}"
              f"{r['advance_window']:>12.3f}{r['push_golem']:>10.3f}"
              f"{str(r['golem_net_dy']):>10}{str(r['min_sep']):>9}")
    for lbl, r in rows:
        if "error" not in r:
            print(f"\n  {lbl} 逐秒 (t, y女巫, y石人): {r['y_series']}")

    print("\n### 判据（预注册 §4.9，机械判定）")
    ok_arm = [lbl for lbl, r in rows if "error" not in r and r["contact_ticks"] >= CRIT_CONTACT_TICKS
              and r["min_sep"] is not None and r["min_sep"] <= CONTACT_SEP]
    print(f"  阳性对照（每臂 contact ≥ {CRIT_CONTACT_TICKS} 且 min_sep ≤ {CONTACT_SEP}）：{ok_arm}")
    neg = nogo.get("advance_window", 0) >= CRIT_NOGOLEM_ADVANCE
    print(f"  阴性对照（不放石人净前进 ≥ {CRIT_NOGOLEM_ADVANCE} 格）："
          f"{nogo.get('advance_window')} ⇒ {'通过' if neg else '未通过 ⇒ 作废'}")
    if len(ok_arm) != len(rows) or not neg:
        print("  ⚠️ 对照未全过 ⇒ 按 §4.9 不得下判据结论（读数仍可作描述）")
    print()
    for lbl, r in rows:
        if "error" in r:
            continue
        order = "她一直在他之后 ✅" if r["order_flips"] == 0 else f"顺序翻转 ❌（{r['order_flips']} 次）"
        pushing = (r["advance_window"] <= CRIT_PUSH_WALK and r["push_golem"] >= CRIT_PUSH_GOLEM)
        mode = "**推着石头人走**" if pushing else "被挡住"
        print(f"  {lbl:8} ① {order}   ② {mode}"
              f"   （净前进 {r['advance_window']} 格 / push石人 {r['push_golem']} 格）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
