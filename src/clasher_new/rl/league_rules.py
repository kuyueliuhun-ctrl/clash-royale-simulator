# -*- coding: utf-8 -*-
"""联赛的**判定语义**（Tier 2 · T2-3，从 `rl/run_league.py` 搬出）。

**搬了什么**：僵局早停参数、塔血合计、`timeout_winner`（加时/超时判胜）、
`settle_stall*`（僵局结算）、`_stall_probe`（探针）。
**为什么能安全整体搬（实测，不是推测）**：用 AST 逐个函数扫自由名 ⇒ 这一簇对模块内的
**唯一**外部依赖是 `_overtime_timeout_winner`（`rl.overtime`）⇒ `league_rules` **不 import
`run_league`**、**没有循环**；`run_league` 反向 import 本模块并**re-export** 全部 8 个名字
⇒ 外部调用方（`rl/selftest.py` 的 `run_league.timeout_winner` / `settle_stall` /
`STALL_LIMIT` / `_stall_probe`，`rl/evaluate.py` 的 `from rl.run_league import timeout_winner`）
**一行未动**。

**⚠️ 「评估并行」那一半（原计划 L581-759）本次未搬**：它的依赖
（`_play_one_game` / `_spec_to_policy` / `_eval_env` / `_ET_MEASURE`）**都在该边界之前**，
而方案又要求 `_ET_MEASURE` / `_set_et_measure`（S2 接线点）**留在 `run_league.py` 原地**
⇒ 直接搬会形成 `run_league ⇄ league_eval` **循环 import**。见
`docs/structure_tier2_2026-09-19.md` §4。

**逐字保证**：正文由 `run_league.py` 的**原行整段切片**生成，未改一个字符。
"""
from __future__ import annotations

from rl.overtime import timeout_winner as _overtime_timeout_winner  # noqa: F401

#: 评估早停：连续 STALL_LIMIT 次检查（每 STALL_WINDOW 步一次）双方塔血合计零变化
#: → 判僵局为平局、提前结束对局，省掉拖满 max_ep_steps 的无效模拟（费差 shaping 下
#: 双方都龟缩的对局占比不小，单局可从 ~23s 降到 ~4s）。CR 无塔治疗，塔血只降不升，
#: 长时间零塔损是可靠僵局信号。
STALL_WINDOW = 10
STALL_LIMIT = 10   # 10×10=100 步无塔损判平


def towers_hp(env):
    """双方三塔血量合计（僵局检测用；塔血只降不升）。"""
    p0 = env.battle.players[0]
    p1 = env.battle.players[1]
    return (p0.king_tower_hp + p0.left_tower_hp + p0.right_tower_hp
            + p1.king_tower_hp + p1.left_tower_hp + p1.right_tower_hp)


def _min_alive_tower_pct(battle, player_id):
    """该玩家存活塔中最低的血量百分比（真实 CR 加时末裁决口径）。

    实体不可得（纯 mock/假 battle，如 selftest）返回 None，调用方退回平局。
    """
    ents = getattr(battle, 'entities', None)
    if not ents:
        return None
    best = None
    for eid in ((3, 4, 6) if player_id == 0 else (1, 2, 5)):
        e = ents.get(eid)
        if e is None or not getattr(e, 'is_alive', False):
            continue
        max_hp = float(getattr(getattr(e, 'data', None), 'hp', 0) or 0)
        if max_hp <= 0:
            return None
        pct = float(e.hp) / max_hp
        if best is None or pct < best:
            best = pct
    return best


def timeout_winner(battle, hp_tiebreak=None):
    """截断/早停时的到期结算兜底（不动引擎，只在 episode 提前结束时补判）。

    规则（2026-09 定稿）：
      1) 皇冠多者胜：皇冠 = 对方被拆塔数（players[X].get_crown_count()
         是 X 侧被拆塔数 = 对方得分）；
      2) 皇冠相同 → **三塔血量合计**多者胜（2026-09-17 用户指定口径；
         旧口径是"存活塔最低血量百分比"，见 docs/draw_rule_prereg_2026-09-17.md）；
         完全相等 → None（平局）。
         僵局早停/截断等价于把终局提前到这里，不能一律记平局——否则出现
         "1-1、双方塔血差 1000+ HP 却记 D" 的错误平局（回放实证：
         economy league_6000 局3/局11 等）。mock 战场无实体信息时退回平局。

    hp_tiebreak: 兼容旧签名保留，不再作为开关（塔血裁决总是启用，见规则 2）。
    """
    if battle is None:
        return None
    p0, p1 = battle.players
    lost0 = int(p0.get_crown_count())
    lost1 = int(p1.get_crown_count())
    if lost1 > lost0:
        return 0
    if lost0 > lost1:
        return 1
    return _overtime_timeout_winner(battle)      # 单一来源（引擎 BattleState.timeout_winner）


def settle_stall_from_counts(lost0, lost1, min_pct0, min_pct1, margin=0.05):
    """C'（2026-09-12）早停局低置信裁定降噪（纯函数，可单测）。

    与 timeout_winner 同口径，但给"皇冠相同"的塔血%细差加置信门槛：
    - 皇冠不同 → 决定性，照常返回 0/1；
    - 皇冠相同且塔血%差 ≥ margin → 决定性，返回 0/1；
    - 皇冠相同且塔血%差 < margin（掷硬币级裁定）→ None（记平局，调用方按
      平局=失败惩罚），**不再按细差判胜负**。

    依据：docs/critic_probe_experiment_2026-09-12.md 实验 3 —— 早停局占比
    28~40%，其中皇冠相同的塔血%细差裁定标签噪声最大（早停把未定局的胜负
    提前裁定，细差方向近乎随机）。仅用于**训练侧**结算降噪；eval 仍用
    timeout_winner（真实 CR 规则），保证评估口径与历史可对比。
    """
    if lost1 > lost0:
        return 0
    if lost0 > lost1:
        return 1
    if min_pct0 is None or min_pct1 is None:
        return None
    if min_pct0 > min_pct1 + margin:
        return 0
    if min_pct1 > min_pct0 + margin:
        return 1
    return None


def settle_stall(battle, margin=0.0):
    """早停局结算。**默认（margin=0）走 2026-09-17 新口径**：皇冠 → 三塔血量**合计**，
    完全相等才平局（= timeout_winner 同一规则）。

    `margin > 0` 时走 2026-09-12 的 C′ 旧路径（存活塔最低血量百分比 + 边距，
    "低置信细差记平局"）——保留它是为了**逐位复现旧标签**（`--stall-draw-margin 0.05`）。
    为什么改：正常防守 ⇒ 100 步无塔损 ⇒ 僵局早停 ⇒ 细差 < margin ⇒ 记 D（且 D 按失败罚），
    即"正常防守被判为平局"。口径与判据见 docs/draw_rule_prereg_2026-09-17.md。
    """
    if battle is None:
        return None
    if margin and float(margin) > 0.0:
        p0, p1 = battle.players
        return settle_stall_from_counts(int(p0.get_crown_count()), int(p1.get_crown_count()),
                                        _min_alive_tower_pct(battle, 0),
                                        _min_alive_tower_pct(battle, 1), float(margin))
    return timeout_winner(battle)


def _stall_probe(env, last_hp, stall_count):
    """僵局探针：每 STALL_WINDOW 步调用一次。

    返回 (early_stop, last_hp, stall_count)。连续 STALL_LIMIT 次零塔血变化 → early_stop。
    """
    hp = towers_hp(env)
    if last_hp is None:
        return False, hp, 0
    if abs(hp - last_hp) < 1e-9:
        stall_count += 1
        if stall_count >= STALL_LIMIT:
            return True, hp, stall_count
        return False, hp, stall_count
    return False, hp, 0
