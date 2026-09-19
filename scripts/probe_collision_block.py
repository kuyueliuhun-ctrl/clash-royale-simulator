# -*- coding: utf-8 -*-
"""碰撞减半（`*0.5`）卡位实验（**第 2 轮**，执行预注册 §4.2）。

第 1 轮按 §2/§3 字面执行后**被自己的阴性对照判定作废**：女巫**自己**会在 y=10.09
停 2.7 s（场上无石人），而我的触发线写的是 `y ≤ 10.0` ⇒ 既不可达、`max_stall` 指标也被污染。
详见 `docs/collision_half_block_test_2026-09-19.md` §4.1。

第 2 轮（判据见该文档 §4.2，**跑前写死**）只改「怎么测」：
  · 触发改 `y ≤ 14.0`（实测此时她仍匀速前进）；
  · 石人投在 `(她的 x, 她的 y − 4.0)`；窗口 = 触发起 **12 s**（或她死亡为止）；
  · 主指标改成**接触期**的 `contact_ticks` / `advance_during_contact` / `golem_displaced`，
    `max_stall` **只报数**，不再当判据。

三臂：`A_full`（e1 全量，现状）・`B_half`（e1 ×0.5，上游口径）・`N_nogolem`（阴性对照）。

单变量：**只**把 e1 的两行位移乘 `half_k`（下面 `make_resolver` 是与
`battle.py:3291-3320` 逐字相同的拷贝，仅那两行加 `*half_k`）；塔特判
（`_push_troop_out_of_tower`）、除零守卫、e2 侧位移**逐字保留**。
引擎源码零改动（猴补丁 `BattleState.resolve_collisions` 的实例属性）。

用法：
    .venv/Scripts/python.exe scripts/probe_collision_block.py
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
TRIGGER_Y = 16.0                 # 第 3 轮：更早触发，使 4 s 窗口**早于**她的自然停车(≈22.0s)
GOLEM_AHEAD = 4.0
WINDOW_SECONDS = 4.0
CONTACT_SEP = 1.30               # 半径和 1.25 + 0.05
STALL_EPS = 0.02
MAXT = 5400
CRIT_CONTACT_TICKS = 30
CRIT_NOGOLEM_ADVANCE = 3.0   # 阴性对照②：无石人时她本来能前进多少


def make_resolver(half):
    """`battle.py:3291-3320` 的逐字拷贝，仅 e1 的两行位移乘 `half_k`。"""
    half_k = 0.5 if half else 1.0

    def resolve_collisions(self):
        from battle import Troop, Building  # noqa: PLC0415
        from itertools import combinations  # noqa: PLC0415
        entities_alive = [each for each in self.entities.values()
                          if each.is_alive and (isinstance(each, Troop) or isinstance(each, Building))]
        ground_troops = combinations([each for each in entities_alive
                                      if not each.data.is_air_unit
                                      and getattr(each.entity_holder, '_mk_jump', None) is None], 2)
        flying_troops = combinations([each for each in entities_alive if each.data.is_air_unit], 2)
        for troop in (ground_troops, flying_troops):
            for e1, e2 in troop:
                if e1.id <= 6 and getattr(e1, 'persistent', False):
                    self._push_troop_out_of_tower(e2, e1); continue
                if e2.id <= 6 and getattr(e2, 'persistent', False):
                    self._push_troop_out_of_tower(e1, e2); continue
                if e1.position.distance_to(e2.position) < e1.data.collision_radius + e2.data.collision_radius:
                    overlap = e1.data.collision_radius + e2.data.collision_radius - e1.position.distance_to(e2.position)
                    direction_vector = complex(e2.position.x-e1.position.x, e2.position.y-e1.position.y)
                    if abs(direction_vector) == 0: return
                    direction_vector /= abs(direction_vector)
                    total_speed = e1.data.speed + e2.data.speed
                    if total_speed == 0: continue
                    movement_ratio = e2.data.speed / total_speed
                    e2.position.x += direction_vector.real*movement_ratio*overlap
                    e2.position.y += direction_vector.imag*movement_ratio*overlap
                    e1.position.x += -direction_vector.real * (1-movement_ratio)*overlap*half_k
                    e1.position.y += -direction_vector.imag * (1-movement_ratio)*overlap*half_k

    return resolve_collisions


def _force_in_hand(bst, player, card):
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


def run_arm(half, use_golem):
    from rl.env_wrapper import RLEnv      # noqa: PLC0415
    from battle import Position           # noqa: PLC0415
    deck = ["DarkWitch", "Knight", "Arrows", "Fireball",
            "Golem", "Minions", "Musketeer", "Giant"]
    env = RLEnv(opponent=lambda obs: None, seed=0, card_level=11, decision_frames=1,
                deck0=list(deck), deck1=list(deck))
    env.reset()
    bst = env.battle
    bst.resolve_collisions = types.MethodType(make_resolver(half), bst)

    _force_in_hand(bst, 1, "DarkWitch")
    if not bst._deploy_card_impl(1, "DarkWitch", Position(*WITCH_SPAWN)):
        return {"error": "女巫部署失败"}

    rows = []            # (tick, witch_y, witch_x, sep|None)
    golem = None
    golem_pt = None
    golem_deploy_ok = None
    golem_disp_max = 0.0
    trigger = None
    for tick in range(1, MAXT + 1):
        bst.step(DT)
        witch = _find(bst, "DarkWitch", 1)
        if witch is None:
            break
        if trigger is None and witch.position.y <= TRIGGER_Y:
            trigger = tick
            if use_golem:
                gx, gy = witch.position.x, witch.position.y - GOLEM_AHEAD
                golem_pt = (round(gx, 3), round(gy, 3))
                golem_deploy_ok = bst._deploy_card_impl(0, "Golem", Position(gx, gy))
                golem = _find(bst, "Golem", 0)
        sep = None
        if golem is not None and golem.is_alive:
            sep = witch.position.distance_to(golem.position)
            golem_disp_max = max(golem_disp_max, ((golem.position.x - golem_pt[0]) ** 2
                                                 + (golem.position.y - golem_pt[1]) ** 2) ** 0.5)
        rows.append((tick, witch.position.y, witch.position.x, sep))
        if trigger is not None and (tick - trigger) > WINDOW_SECONDS * 60:
            break

    if trigger is None:
        return {"error": f"女巫从未到达触发线 y<={TRIGGER_Y}"}
    win = [r for r in rows if r[0] >= trigger]
    if len(win) < 5:
        return {"error": "窗口过短"}
    seps = [(r[0], r[3]) for r in win if r[3] is not None]
    contact = [r for r in win if r[3] is not None and r[3] <= CONTACT_SEP]
    best = cur = 0
    for (_, y0, x0, _), (_, y1, x1, _) in zip(win, win[1:]):
        if ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 < STALL_EPS:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return {
        "trigger": trigger, "window_ticks": len(win),
        "golem_pt": golem_pt, "golem_ok": golem_deploy_ok,
        "min_sep": round(min(s for _, s in seps), 3) if seps else None,
        "contact_ticks": len(contact),
        "advance_during_contact": (round(contact[0][1] - contact[-1][1], 3)
                                   if len(contact) >= 2 else None),
        "golem_displaced": round(golem_disp_max, 3),
        "max_stall": best,
        "y_at_trigger": round(win[0][1], 2), "y_at_end": round(win[-1][1], 2),
        "advance_window": round(win[0][1] - win[-1][1], 3),
        "sample": [(round(r[0] * DT, 1), round(r[1], 1), None if r[3] is None else round(r[3], 2))
                   for r in win[::36]],
    }


def main():
    arms = [("A_full   (e1 全量，现状)", False, True),
            ("B_half   (e1 ×0.5，上游)", True, True),
            ("N_nogolem(阴性对照：不放石人)", False, False)]
    res = {}
    for label, half, use_golem in arms:
        r = run_arm(half, use_golem)
        res[label] = r
        print("=" * 78)
        print(f"### {label}")
        if "error" in r:
            print(f"  ⚠️ {r['error']}   {r}")
            continue
        print(f"  触发 tick={r['trigger']}（她 y={r['y_at_trigger']}）"
              f"  窗口 {r['window_ticks']} tick = {r['window_ticks']/60:.1f} s")
        print(f"  石人投放 {r['golem_pt']} ok={r['golem_ok']}")
        print(f"  **contact_ticks={r['contact_ticks']}**  "
              f"advance_during_contact={r['advance_during_contact']} 格  "
              f"min_sep={r['min_sep']}  石人被推走={r['golem_displaced']} 格")
        print(f"  （只报数）max_stall={r['max_stall']} tick ；y: {r['y_at_trigger']} → {r['y_at_end']}")
        print(f"  t→(y,sep) 每 3.6 s: {r['sample']}")

    a, b, n = (res[k] for k in res)
    print("=" * 78)
    print("### 判据（预注册 §4.2，机械判定）")
    if any("error" in x for x in (a, b, n)):
        print("  ⚠️ 有臂异常 ⇒ 本轮作废"); return 2
    pos = all(x["min_sep"] is not None and x["min_sep"] <= CONTACT_SEP for x in (a, b))
    print(f"  阳性对照（两臂 min_sep ≤ {CONTACT_SEP}）：A={a['min_sep']} B={b['min_sep']} ⇒ "
          f"{'通过' if pos else '**未通过 ⇒ 作废**'}")
    n1 = n["max_stall"] < 18
    n2 = n["advance_window"] >= CRIT_NOGOLEM_ADVANCE
    print(f"  阴性对照①（不放石人 max_stall < 18）：N={n['max_stall']} ⇒ {'通过' if n1 else '**未通过 ⇒ 作废**'}")
    print(f"  阴性对照②（不放石人窗口内前进 ≥ {CRIT_NOGOLEM_ADVANCE} 格）："
          f"N={n['advance_window']} 格 ⇒ {'通过' if n2 else '**未通过 ⇒ 作废**'}")
    if not (pos and n1 and n2):
        return 2
    c1 = b["contact_ticks"] >= CRIT_CONTACT_TICKS
    c2 = (b["advance_during_contact"] is not None and b["advance_during_contact"] <= 0.0)
    print(f"  (a) contact_ticks(B_half) ≥ {CRIT_CONTACT_TICKS}：{b['contact_ticks']} ⇒ {c1}")
    print(f"  (b) advance_during_contact(B_half) ≤ 0（接触期无净前进）："
          f"{b['advance_during_contact']} ⇒ {c2}")
    print(f"  (c) 阴性②成立（她本来能过）：{n2} ⇒ {n2}")
    print(f"  【对照读数，不作判据】A_full: contact={a['contact_ticks']} "
          f"advance_contact={a['advance_during_contact']} 石人位移={a['golem_displaced']}；"
          f"B_half: 石人位移={b['golem_displaced']}")
    ok = c1 and c2 and n2
    print(f"  ⇒ **{'碰撞减半卡得住 ⇒ 做' if ok else '碰撞减半卡不住 ⇒ 不做'}**")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
