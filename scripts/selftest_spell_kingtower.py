# -*- coding: utf-8 -*-
"""**回归测试**（【R8】）：法术「只罩王塔」的落点必须被判非法 + 槽位无落点必须被禁。

被保护的修复（2026-09-22，台账 C20）：
`rl/action_mask.py::_spell_tower_ev_illegal` 原来只判「罩到敌方**公主塔**」的落点，
两座公主塔之间的中路口（只罩王塔、罩不到公主塔）会走到**提前 return** 被放行 ——
实测被 IL 策略当成「开局第一手火球砸敌方王塔」的合法通道（7 次、每次 206 血）。
修法：王塔纳入判定，估值沿用同源闸门 `tower_value_mult(king=True, princesses_alive=…)`。

**同时保护** `rl/env_wrapper.py::get_action_mask_for` 的不变式：合法槽必须至少有一个
合法落点；否则 `act()` 会把 576 格全填 -1e9，`argmax` 落到格 0（角落）⇒ 提交被
`validate_bundle` 整包拒收（白掉一帧 + `invalid_penalty`）。

用法（仓库根）：
    ./.venv/Scripts/python.exe scripts/selftest_spell_kingtower.py     # PASS -> rc=0
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

import numpy as np  # noqa: E402

from battle import BattleState  # noqa: E402
from card_utils import Card  # noqa: E402
from core import Position  # noqa: E402
from player import PlayerState  # noqa: E402
from rl.action_bundle import K_MAX, sub_position  # noqa: E402
from rl.action_mask import (_spell_deals_damage, _spell_radius_m,  # noqa: E402
                            _spell_tower_damage, legal_cells,
                            SPELL_EV_EDW, TOWER_HP_PER_ELIXIR_EARLY)
from rl.env_wrapper import RLEnv  # noqa: E402
from rl.reward import tower_value_mult  # noqa: E402

DECK = ["Fireball", "Knight", "Archer", "Tesla", "Skeletons", "IceWizard", "Xbow", "Log"]
KING_ENEMY = Position(9.0, 29.0)
KING_COL = 1.4
R_FB = _spell_radius_m("Fireball", Card("Fireball"))

FAILS = []
N_ASSERT = [32]   # 23（C20/C21 批）+ 9（2026-09-22 W1 低费空砸豁免）


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


#: 敌方卡组（Log 放第 1 张 ⇒ 一定在**手牌**里；`deploy_card` 只认手牌内的卡）
ENEMY_DECK_LOG_FIRST = ["Log", "Knight", "Archer", "Tesla",
                        "Skeletons", "IceWizard", "Xbow", "Fireball"]


def make(units=None, time=0.0, princess_hp=None, king_hp=None, enemy_deck=None):
    bs = BattleState(PlayerState(0, DECK, 5.0),
                     PlayerState(1, list(enemy_deck or DECK), 5.0), card_level=11)
    bs.time = time
    for c, (x, y) in (units or []):
        bs.deploy_card(1, c, Position(x, y))
    if princess_hp is not None:
        for tid in (1, 2):
            tw = bs.entities.get(tid)
            if tw is not None:
                tw.hp = princess_hp
    if king_hp is not None:
        for tid in (5, 6):
            tw = bs.entities.get(tid)
            if tw is not None:
                tw.hp = king_hp
    return bs


def king_only_cells(bs, pid, card, radius):
    """返回「只罩王塔」格的 (x, y) 列表（罩到王塔、罩不到公主塔、罩不到任何非塔目标）。"""
    out = []
    cells = legal_cells(bs, pid, card)
    opp = 1 - pid
    king = next((e for e in bs.entities.values()
                 if getattr(e, "player", None) == opp and e.is_alive
                 and (getattr(e, "name", "") or "") == "KingTower"), None)
    prs = [e for e in bs.entities.values()
           if getattr(e, "player", None) == opp and e.is_alive
           and "Princess" in (getattr(e, "name", "") or "")]
    for y in range(32):
        for x in range(18):
            pos = sub_position(pid, x, y)
            rk = radius + (getattr(king.data, "collision_radius", 0.0) or 0.0) if king else 0.0
            in_king = king is not None and pos.distance_to(king.position) <= rk + 1e-9
            in_pr = any(pos.distance_to(p.position) <= radius + (getattr(p.data, "collision_radius", 0.0) or 0.0) + 1e-9
                        for p in prs)
            in_unit = any((getattr(e, "player", None) == opp and e.is_alive
                           and e.id not in (1, 2, 3, 4, 5, 6)
                           and pos.distance_to(e.position)
                           <= radius + (getattr(getattr(e, "data", None), "collision_radius", 0.0) or 0.0) + 1e-9)
                          for e in bs.entities.values())
            if in_king and not in_pr and not in_unit:
                out.append((x, y, bool(cells[y, x])))
    return out


def king_cell_legal(bs, pid, card):
    cells = legal_cells(bs, pid, card)
    return bool(cells[29, 9]) if pid == 0 else bool(cells[2, 8])


def main():
    # ① 空场（双公主塔存活）：中路「只罩王塔」的格子全部非法
    bs = make()
    ko = king_only_cells(bs, 0, "Fireball", R_FB)
    legal_ko = [c for c in ko if c[2]]
    check("① 空场：存在「只罩王塔」格（前提）", len(ko) > 0, f"n={len(ko)}")
    check("① 空场：这些格**全部非法**（修复前是全部合法）", len(legal_ko) == 0,
          f"非法 {len(ko) - len(legal_ko)}/{len(ko)}，仍合法 {legal_ko[:4]}")
    check("① 空场：敌方王塔格本身非法", not king_cell_legal(bs, 0, "Fireball"))

    # ② 负对照 a：敌方**部队**在半径内 ⇒ 仍然合法（不许误伤「打躲在王塔后的部队」）
    bs2 = make(units=[("Knight", (9.5, 31.5))])
    check("② 部队在半径内 ⇒ 王塔格仍合法", king_cell_legal(bs2, 0, "Fireball"))

    # ② 负对照 b：敌方**公主塔**格本来就非法（9h 闸门，未被本次修复改变）
    check("② 敌方公主塔格非法（原闸门口径不变）",
          not bool(legal_cells(bs, 0, "Fireball")[25, 14]))

    # ③ 双倍期（t≥120）⇒ 闸门整体放行（既有语义不变）
    bs3 = make(time=130.0)
    check("③ 双倍期王塔格合法（闸门 time 条款不变）", king_cell_legal(bs3, 0, "Fireball"))

    # ④ 「只罩王塔」现在按**王塔自身**估值 ⇒ 与公式逐值一致（任一公主塔被破后的语义）
    bs4 = make(princess_hp=0, king_hp=4824.0 * 0.9)
    dmg = _spell_tower_damage("Fireball")
    mult = tower_value_mult(0.9, king=True, princesses_alive=1)
    want = (dmg * mult) / TOWER_HP_PER_ELIXIR_EARLY < SPELL_EV_EDW * Card("Fireball").elixir - 1e-9
    check("④ 公主塔破 1 座：王塔按 tower_value_mult(king=True, princesses_alive=1) 判",
          king_cell_legal(bs4, 0, "Fireball") == (not want),
          f"dmg={dmg} mult={mult:.4f} 公式判非法={want}")

    # ⑤ 槽位不变式：合法槽必须至少有一个合法落点；否则该槽被禁，且 STOP 仍合法
    env = RLEnv(opponent=None, seed=0, card_level=11, deck0=DECK, deck1=DECK)
    env.reset(seed=0)
    #: reset 会**洗牌**手牌 ⇒ 显式把 Fireball 放到槽 1（掩码指纹含 `p.cycle`，会自然失效重算）
    cyc = list(env.battle.players[0].cycle)
    cyc.remove("Fireball")
    cyc.insert(0, "Fireball")
    env.battle.players[0].cycle = cyc
    m = env.get_action_mask()
    empty = [i for i in range(K_MAX) if m["slots"][i] and not m["cells"][i].any()]
    check("⑤ 不变式：不存在「合法槽但零合法落点」", not empty, f"违例槽={[i + 1 for i in empty]}")
    check("⑤ any_legal 恒真（STOP 兜底）", bool(m["any_legal"]))
    #: ⚠️ 手牌在 reset 时会**洗牌**（`p.cycle` 顺序 ≠ 传入的 deck 顺序）⇒ 必须按 cycle 找 Fireball，
    #: 不能按 `DECK.index()` 假定槽号（第一版就这么写错过，打印出的是别的卡的 224 个落点）。
    cyc = [str(c) for c in env.battle.players[0].cycle[:K_MAX]]
    fb_slot = (cyc.index("Fireball") + 1) if "Fireball" in cyc else None
    if fb_slot is None:
        print(f"[info] 本局开局手牌无 Fireball（cycle[:4]={cyc}）⇒ ⑤b 跳过")
    else:
        n_cells = int(np.asarray(m["cells"][fb_slot - 1]).sum())
        check("⑤b 开局空场：Fireball 槽被**禁**（无合法落点 ⇒ 等价于买不起）",
              not bool(m["slots"][fb_slot - 1]) and n_cells == 0,
              f"cycle={cyc} slot{fb_slot} legal={bool(m['slots'][fb_slot - 1])} cells={n_cells}")
    print(f"[info] 开局 mask：cycle[:4]={cyc} slots={np.asarray(m['slots']).astype(int).tolist()} "
          f"cells_per_slot={[int(np.asarray(m['cells'][i]).sum()) for i in range(K_MAX)]}")

    # ⑤c 【F2】效果载体（法术/弹道/区域效果的载体）**不算**敌方目标
    #     —— 先例：IL 策略在 t=1.0 把 Fireball 丢到敌方王塔格，唯一"理由"是场上有
    #     一枚**对方 Log 的滚动弹**（`LogProjectileRolling`）；它无血、打不掉，是假目标。
    #: ⚠️ 两个坑（都实测踩过）：① `deploy_card` 只认**手牌内**的卡 ⇒ Log 必须在敌方卡组前 4 张；
    #: ② 滚动弹是**下一 tick**才生成的实体（`projectile` 类型）⇒ 部署后必须 `step` 一次。
    bs5 = make(enemy_deck=ENEMY_DECK_LOG_FIRST)
    bs5.deploy_card(1, "Log", Position(8.5, 29.0))
    bs5.step(0.05)                                     # ⇒ 生成 LogProjectileRolling（player=1）
    eff = [e for e in bs5.entities.values()
           if getattr(getattr(e, "data", None), "type", "") in ("projectile", "area_effect", "bomb")]
    check("⑤c 状态里确有「效果载体」（前提）", len(eff) > 0,
          f"n={len(eff)} names={[e.name for e in eff][:3]}")
    check("⑤c 只有效果载体在半径内 ⇒ 王塔格**非法**", not king_cell_legal(bs5, 0, "Fireball"))
    #   负对照：同一状态再放一个**真实部队** ⇒ 恢复合法（不许把"打部队"一起误伤）
    bs6 = make(units=[("Knight", (9.5, 31.5))], enemy_deck=ENEMY_DECK_LOG_FIRST)
    bs6.deploy_card(1, "Log", Position(8.5, 29.0))
    bs6.step(0.05)
    check("⑤c 真实部队 + 效果载体同在 ⇒ 合法", king_cell_legal(bs6, 0, "Fireball"))

    # ⑦ 【2026-09-22 用户指令】`_spell_deals_damage` 与引擎同源 + 滚动类豁免
    #    背景：旧谓词只读 data["projectileData"]["damage"]|data["damage"]，
    #    漏判 Zap/Poison/Earthquake/Tornado/Vines/Freeze/RoyalDelivery/Log/BarbLog
    #    ⇒ 8h 空砸 + 9h 砸塔 EV 闸门对它们**根本没开**。
    from rl.action_mask import _spell_deals_damage, _spell_requires_placement_target
    bs7 = make()
    check("⑦ 谓词修复：Zap/Poison/Vines/Tornado/Earthquake 现在算伤害法术",
          all(_spell_deals_damage(c) for c in ("Zap", "Poison", "Vines", "Tornado", "Earthquake")))
    check("⑦ Heal/Rage/Clone/Mirror **不**算敌方伤害（友方语义，防误伤）",
          not any(_spell_deals_damage(c) for c in ("Heal", "Rage", "Clone", "Mirror")))
    check("⑦ 滚动类（Log/BarbLog）算伤害但**不适用落点几何闸门**",
          _spell_deals_damage("Log") and _spell_deals_damage("BarbLog")
          and not _spell_requires_placement_target("Log")
          and not _spell_requires_placement_target("BarbLog"))
    #: ⚠️ 2026-09-22 W1 起：**Zap/Snowball（2 费）已被低费豁免**，不能再拿它们验"闸门收起"
    #: ⇒ 改用 **Fireball（4 费）**验「空砸闸门确实还在起作用」；Zap 的新契约见 ⑧。
    fb_n = int(legal_cells(bs7, 0, "Fireball").sum())
    check("⑦ Fireball 在「只有塔」的场上被空砸闸门收起（修前 576 全合法）",
          fb_n == 0, f"legal={fb_n}")
    #: ⚠️ 别把单位放在**王塔矩形内**（王塔 4×4、中心 (9,29) ⇒ x∈[7,11], y∈[27,31]）——
    #: `deploy_card` 会静默失败（第一版就踩了：Zap 仍 0 合法，看着像闸门坏了）
    bs7b = make(units=[("Knight", (8.5, 22.0))])
    check("⑦ Zap 对着部队仍然合法（不误伤主用途）", int(legal_cells(bs7b, 0, "Zap").sum()) > 0,
          f"legal={int(legal_cells(bs7b, 0, 'Zap').sum())}")
    #: Log 的落点是"部署点"（伤害在滚动走廊）⇒ 自己半场空放**必须仍然合法**
    #: （人类 99.4% 的 Log 就落在己方半场；这条防的是"修谓词把过牌一起修死"）
    log_own = int(legal_cells(bs7, 0, "Log")[:16, :].sum())
    check("⑦ Log 己方半场空放仍合法（过牌行为不被误伤）", log_own > 0, f"own-half legal={log_own}")
    heal_n = int(legal_cells(bs7, 0, "Heal").sum())
    check("⑦ Heal（友方法术）仍可放（不被当伤害法术禁掉）", heal_n > 0, f"legal={heal_n}")

    # ⑥ 白盒：修复把「王塔」纳入判定这件事本身（防止有人回退成只判公主塔）
    calls = {"n": 0}
    import rl.action_mask as am  # noqa: E402
    src = open(os.path.join(SRC, "rl", "action_mask.py"), encoding="utf-8").read()
    check("⑥ 源码里存在王塔判定（hits_king）", "hits_king" in src)
    check("⑥ 源码里存在效果载体排除（_is_effect_body）", "_is_effect_body" in src)
    check("⑥ 提前 return 已改为 (hits_princess or hits_king)",
          "if not (hits_princess or hits_king):" in src)

    # ⑧ 【2026-09-22 W1】低费法术**空砸豁免**（用户：「空砸这个问题，首先是放开一些低费卡牌的限制」）
    #    口径：费用 ≤ `SPELL_WHIFF_FREE_MAX_COST`(=2) 的伤害型法术不再受 **8h 空砸**闸门约束；
    #    **9h 砸塔 EV 闸门不豁免** ⇒ 空地 Zap 合法，而「只罩敌方王塔」的 Zap/Snowball 仍非法
    #    （用户上一轮要求的 F1 修复**不回退**）。用户同批更正：**Poison 是 4 费，不便宜**。
    from rl.action_mask import (SPELL_WHIFF_FREE_MAX_COST,  # noqa: E402
                                _spell_whiff_gate_applies)
    check("⑧ 阈值 = 2（= 过牌法术集合；Poison 4 费 / Arrows 3 费都不在内）",
          float(SPELL_WHIFF_FREE_MAX_COST) == 2.0, f"={SPELL_WHIFF_FREE_MAX_COST}")
    _tbl = {"Log": False, "BarbLog": False, "Rage": False, "GoblinCurse": False,
            "Zap": False, "Snowball": False,
            "Arrows": True, "Earthquake": True, "Tornado": True, "Vines": True,
            "Fireball": True, "Poison": True, "Rocket": True}
    _bad = sorted(c for c, want in _tbl.items() if _spell_whiff_gate_applies(c) != want)
    check("⑧ 空砸闸门真值表（≤2 费不适用 / ≥3 费适用）", not _bad, f"不符={_bad}")
    _bs8 = make()
    _zap = int(legal_cells(_bs8, 0, "Zap").sum())
    _snow = int(legal_cells(_bs8, 0, "Snowball").sum())
    check("⑧ 空场：Zap 现在**有**合法落点（修前 0 ⇒ 过牌/空放被解锁）", _zap > 0, f"legal={_zap}")
    check("⑧ 空场：Snowball 现在**有**合法落点（修前 0）", _snow > 0, f"legal={_snow}")
    #: 负对照（R13 逐位口径的轻量版）：≥3 费伤害法术**一个都不许**被放松
    _exp = {c: int(legal_cells(_bs8, 0, c).sum())
            for c in ("Arrows", "Tornado", "Fireball", "Poison", "Rocket")}
    check("⑧ 负对照：≥3 费伤害法术空场仍**全 0** 合法落点（未放松贵牌）",
          all(v == 0 for v in _exp.values()), f"{_exp}")
    #: F1 不回归：低费豁免只碰"空砸"，不碰"只罩王塔"
    for _c in ("Zap", "Snowball"):
        _r = _spell_radius_m(_c, Card(_c))
        _ko = king_only_cells(_bs8, 0, _c, _r)
        _lg = [c for c in _ko if c[2]]
        check(f"⑧ F1 不回归：{_c}「只罩敌方王塔」格仍**全部非法**",
              len(_ko) > 0 and len(_lg) == 0, f"n={len(_ko)} 仍合法={_lg[:3]}")
    #: 白盒：**两处**调用点都要换（提交路径 `_position_legal` 与掩码路径 `legal_cells`
    #: 分叉过一次的教训）⇒ 定义 1 次 + 调用 ≥2 次
    _n_call = src.count("_spell_whiff_gate_applies(")
    check("⑧ 白盒：_spell_whiff_gate_applies 出现 ≥3 次（定义 1 + 两处调用）",
          _n_call >= 3, f"count={_n_call}")
    check("⑧ 白盒：9h EV 闸门**未**被加上费用豁免（F1 保护留在原处）",
          "SPELL_WHIFF_FREE_MAX_COST" not in src.split("def _spell_tower_ev_illegal")[1].split("def ")[0])

    print(f"\n=== selftest_spell_kingtower: {len(FAILS)} failed / {N_ASSERT[0]} assertions ===")
    if FAILS:
        for f in FAILS:
            print("  FAIL:", f)
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
