# -*- coding: utf-8 -*-
"""卡位实验 第 4 轮：**质量模型剂量-反应**（执行预注册 §4.7）。

用户口径（2026-09-19）：「我们要的当然是**石头人能够站得住**……或许需要使用**移速或体积**
来定义一个质量，真实游戏中，**卡位的石头人是一动不动的**。」

已查明的两条前提（可逐值验）：
  1. **现状已经等价于「质量 ∝ 1/移速」**：`battle.py:3316` 的 `movement_ratio = s2/(s1+s2)`
     与「按质量反比分配」在 `m = 1/s` 下**逐值相同** ⇒ **缺的是体积项**。
  2. 第 2 轮的 `golem_displaced = 2.36 格` 是**净位移**（自己走 + 被推），
     **不是**「被推走」的证据 ⇒ 本轮**把推挤分量单独积分**（`push_golem` / `push_witch`）。

单变量 = 质量指数 `k`：`m = (collision_radius ** k) / speed`，**k=0 逐值退化为现状**
（⇒ k=0 同时是阳性对照）。扫 `k ∈ {0,2,3,4,6}`；另加 `k=0 + half`（上游 `*0.5` 口径）、
**未打补丁**一行作恒等对照、**不放石人**一行作阴性对照。

判据（§4.7，跑前写死）：判「石头人站得住」= `push_golem ≤ 0.10 格`；报出满足它的最小 k，
并**必须同报** `push_witch` 与 `advance_during_contact`（站得住是靠转嫁给女巫换来的）。

前 3 轮的账见 `docs/collision_half_block_test_2026-09-19.md` §4.1–§4.6
（两轮被自己的对照判废、一轮阳性对照未过；净结论 = **两种碰撞口径都能卡住**，
差别只在量级 ⇒ 本轮改问「能不能站得住」）。

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
TRIGGER_Y = 14.0
GOLEM_AHEAD = 1.0   # 紧贴她正前方：石人 deploy_time=3s 冻结，这 3s 它就是「墙」
WINDOW_SECONDS = 3.0
CONTACT_SEP = 1.30
MAXT = 5400
CRIT_PUSH_GOLEM = 0.10
CRIT_CONTACT_TICKS = 30
CRIT_NOGOLEM_ADVANCE = 3.0


def _mass(e, k):
    r = max(float(e.data.collision_radius), 1e-6)
    s = max(float(e.data.speed), 1e-6)
    return (r ** k) / s


def make_resolver(k, half, pair_acc):
    """`battle.py:3291-3320` 的逐字拷贝，只把「谁被推多少」换成按质量反比分配。

    现状（k=0）： `m = 1/speed` ⇒ e2 拿 `m1/(m1+m2) = s2/(s1+s2)` = 原 `movement_ratio`。
    """
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
                    m1, m2 = _mass(e1, k), _mass(e2, k)
                    share_e2 = m1 / (m1 + m2)
                    share_e1 = m2 / (m1 + m2)
                    p1x, p1y = e1.position.x, e1.position.y
                    p2x, p2y = e2.position.x, e2.position.y
                    e2.position.x += direction_vector.real * share_e2 * overlap
                    e2.position.y += direction_vector.imag * share_e2 * overlap
                    e1.position.x += -direction_vector.real * share_e1 * overlap * half_k
                    e1.position.y += -direction_vector.imag * share_e1 * overlap * half_k
                    key = (min(e1.id, e2.id), max(e1.id, e2.id))
                    d1 = ((e1.position.x - p1x) ** 2 + (e1.position.y - p1y) ** 2) ** 0.5
                    d2 = ((e2.position.x - p2x) ** 2 + (e2.position.y - p2y) ** 2) ** 0.5
                    a = pair_acc.setdefault(key, [0.0, 0.0])   # [小 id 被推, 大 id 被推]
                    if e1.id <= e2.id:
                        a[0] += d1; a[1] += d2
                    else:
                        a[0] += d2; a[1] += d1

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


def run_arm(k, half=False, use_golem=True):
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
        bst.resolve_collisions = types.MethodType(make_resolver(k, half, pair_acc), bst)

    _force_in_hand(bst, 1, "DarkWitch")
    if not bst._deploy_card_impl(1, "DarkWitch", Position(*WITCH_SPAWN)):
        return {"error": "女巫部署失败"}

    rows, golem, golem_pt, trigger = [], None, None, None
    wit_id = None
    for tick in range(1, MAXT + 1):
        bst.step(DT)
        witch = _find(bst, "DarkWitch", 1)
        if witch is None:
            break
        wit_id = witch.id
        if trigger is None and witch.position.y <= TRIGGER_Y:
            trigger = tick
            if use_golem:
                gx, gy = witch.position.x, witch.position.y - GOLEM_AHEAD
                golem_pt = (round(gx, 3), round(gy, 3))
                if not bst._deploy_card_impl(0, "Golem", Position(gx, gy)):
                    return {"error": "石人部署失败"}
                golem = _find(bst, "Golem", 0)
        sep = (witch.position.distance_to(golem.position)
               if golem is not None and golem.is_alive else None)
        rows.append((tick, witch.position.y, witch.position.x, sep))
        if trigger is not None and (tick - trigger) > WINDOW_SECONDS * 60:
            break

    if trigger is None:
        return {"error": f"女巫从未到达触发线 y<={TRIGGER_Y}"}
    win = [r for r in rows if r[0] >= trigger]
    if len(win) < 5:
        return {"error": "窗口过短"}
    contact = [r for r in win if r[3] is not None and r[3] <= CONTACT_SEP]

    # 推挤分量：只在「女巫↔石人」这一对上积分（其它对如蝙蝠→石人不计入主指标）
    push_g = push_w = 0.0
    if golem is not None and use_golem and wit_id is not None:
        key = (min(golem.id, wit_id), max(golem.id, wit_id))
        if key in pair_acc:
            a = pair_acc[key]
            push_g = a[0] if golem.id <= wit_id else a[1]
            push_w = a[1] if golem.id <= wit_id else a[0]
    return {
        "trigger": trigger, "window_ticks": len(win), "golem_pt": golem_pt,
        "min_sep": round(min(r[3] for r in win if r[3] is not None), 3) if contact else None,
        "contact_ticks": len(contact),
        "advance_during_contact": (round(contact[0][1] - contact[-1][1], 3)
                                   if len(contact) >= 2 else None),
        "push_golem_pair": round(push_g, 3), "push_witch_pair": round(push_w, 3),
        "advance_window": round(win[0][1] - win[-1][1], 3),
        "end_witch": (round(win[-1][2], 4), round(win[-1][1], 4)),
        "end_golem": ((round(golem.position.x, 4), round(golem.position.y, 4))
                      if (golem is not None and golem.is_alive) else None),
    }


def main():
    print("=" * 96)
    print("### 第 4 轮：质量模型 m = r^k / speed 的剂量-反应（单变量 k）")
    rows = []
    for k in (0, 2, 3, 4, 6):
        rows.append((f"k={k}  (m=r^{k}/speed)", run_arm(k)))
    rows.append(("k=0 + half (上游 *0.5)", run_arm(0, half=True)))
    base = run_arm(None)          # 未打补丁
    nogo = run_arm(0, use_golem=False)
    print(f"\n  阴性对照 N_nogolem（不放石人）：窗口内前进 {nogo.get('advance_window')} 格 "
          f"(需 ≥ {CRIT_NOGOLEM_ADVANCE})")
    print(f"\n{'臂':26s}{'contact':>8}{'push石人':>10}{'push女巫':>10}{'接触期前进':>12}"
          f"{'min_sep':>9}{'窗口前进':>10}")
    for label, r in rows + [("未打补丁(恒等对照)", base)]:
        if "error" in r:
            print(f"{label:26s}  ⚠️ {r['error']}")
            continue
        f = lambda v: "-" if v is None else str(v)
        print(f"{label:26s}{f(r['contact_ticks']):>8}{r['push_golem_pair']:>10.3f}"
              f"{r['push_witch_pair']:>10.3f}{f(r['advance_during_contact']):>12}"
              f"{f(r['min_sep']):>9}{r['advance_window']:>10.3f}")

    print("\n### 判据（预注册 §4.7）")
    c0 = rows[0][1]
    ident = (base.get("end_witch") == c0.get("end_witch")
             and base.get("end_golem") == c0.get("end_golem"))
    print(f"  阳性对照①：k=0 与未打补丁末位置逐值相同 ⇒ {ident}")
    print(f"      k=0 末位置 女巫={c0.get('end_witch')} 石人={c0.get('end_golem')}")
    print(f"      未打补丁   女巫={base.get('end_witch')} 石人={base.get('end_golem')}")
    pos = all(r.get("min_sep") is not None and r["min_sep"] <= CONTACT_SEP
              and r["contact_ticks"] >= CRIT_CONTACT_TICKS for _, r in rows)
    print(f"  阳性对照②：各臂 min_sep ≤ {CONTACT_SEP} 且 contact ≥ {CRIT_CONTACT_TICKS} ⇒ {pos}")
    neg = nogo.get("advance_window", 0) >= CRIT_NOGOLEM_ADVANCE
    print(f"  阴性对照：不放石人窗口前进 ≥ {CRIT_NOGOLEM_ADVANCE} 格 ⇒ {neg}")
    if not (ident and pos and neg):
        print("  ⚠️ 对照未全过 ⇒ 按 §4.7 不得下判据结论（读数仍可作描述）")
    print(f"\n  主判据「石头人站得住」= push_golem ≤ {CRIT_PUSH_GOLEM} 格：")
    ok_k = []
    for lbl, r in rows:
        if "error" in r:
            continue
        good = r["push_golem_pair"] <= CRIT_PUSH_GOLEM
        if good:
            ok_k.append(lbl.split()[0])
        print(f"    {'✅' if good else '  '} {lbl:26s} push石人={r['push_golem_pair']:.3f}  "
              f"push女巫={r['push_witch_pair']:.3f}  接触期前进={r['advance_during_contact']}")
    print(f"  ⇒ 满足的最小 k：{'无（所有 k 都不满足）' if not ok_k else ok_k[0]}")
    print("  ⚠️ 影响面：本改动改变**所有**单位对的推挤分配，不是只改石人；"
          "即使通过也**不是**可直接上线的结论（§4.7 末）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
