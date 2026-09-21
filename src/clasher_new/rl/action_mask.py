"""动作合法性掩码 + 整包校验（规划文档 3.4 / 3.4.1）。

掩码规则尽量与 battle.deploy_card 的真实校验保持一致；即使掩码误判，
执行时仍以 deploy_card 的返回值作为最终依据。

坐标契约（docs/rl_review_fix_plan.md §5）：
- ``SubAction(x, y)`` 一律是**玩家本地坐标**；
- 掩码层与提交层共用 :func:`rl.action_bundle.sub_position` 做唯一换算，
  消除“掩码世界坐标 vs 提交镜像坐标”的分裂（P0-4）。
"""

import os
import sys

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from typing import List, Optional, Tuple

import numpy as np

from core import Position
from card_utils import Card
from rl.action_bundle import ActionBundle, K_MAX, sub_position

GRID_H, GRID_W = 32, 18


def _card_cost(player, card_name: str) -> Optional[float]:
    """实际出牌费用（Mirror 按引擎语义 = 上一张牌费用 + 1；无上一张牌 → None）。"""
    if card_name == "Mirror":
        if not getattr(player, "last_card", None):
            return None
        return Card(player.last_card).elixir + 1
    return Card(card_name).elixir


def _effective_card(player, card_name: str) -> str:
    """引擎实际部署/校验的卡名（Mirror 重放上一张牌）。"""
    if card_name == "Mirror":
        return getattr(player, "last_card", None) or card_name
    return card_name


def _slot_playable(player, card_name: str, elixir: float) -> bool:
    if player.king_tower_hp <= 0:
        return False
    if card_name not in player.cycle[:4]:
        return False
    cost = _card_cost(player, card_name)
    if cost is None:
        return False
    if elixir < cost:
        return False
    return True


def slot_mask(player, elixir_override: float = None, used_slots=None) -> np.ndarray:
    """返回 (K_MAX,) bool 掩码：哪些手牌槽在当前圣水下可出（且未在 bundle 中使用）。"""
    elixir = player.elixir if elixir_override is None else elixir_override
    used = set(used_slots or [])
    mask = np.zeros(K_MAX, dtype=bool)
    for i in range(K_MAX):
        if i in used:
            continue
        mask[i] = _slot_playable(player, player.cycle[i], elixir)
    return mask


#: 已毁敌方塔“本体格”屏蔽半径（本地网格≈1 格=1 单位；邻格中心距 ≥1.0 → 只挡本体格，
#: 不误伤“打塔旁敌军”的合法溅射法术）
DEAD_TOWER_BODY_R = 0.9

#: —— 8h 不空砸：伤害型法术必须能罩到 ≥1 个存活敌方目标（塔/建筑/单位都算）——
#: 引擎数值表半径以千分之一单位存储（Arrows 3500 → 3.5；Fireball 2500 → 2.5）。


def _spell_radius_m(card_name: str, card_info: "Card" = None) -> float:
    """法术溅射半径（世界单位）；无半径数据返回 0.0（=不做闸门，保持旧语义）。
    P0 优化：card_info 已构造时直接复用（省 7µs/次构造，掩码循环热路径）。"""
    data = getattr(card_info if card_info is not None else Card(card_name), "data", None) or {}
    raw = data.get("radius")
    if raw is None:
        raw = ((data.get("projectileData") or {}).get("radius"))
    return (float(raw) if raw else 0.0) / 1000.0


def _spell_deals_damage(card_name: str, card_info: "Card" = None) -> bool:
    """是否输出伤害的法术。伤害型法术受空砸闸门约束；增益/位移/召唤类法术放行。"""
    data = getattr(card_info if card_info is not None else Card(card_name), "data", None) or {}
    pd = data.get("projectileData") or {}
    return (float(pd.get("damage") or 0.0) > 0.0
            or float(data.get("damage") or 0.0) > 0.0)


#: 引擎里"不是可被打的目标"的实体类型（法术/弹道/区域效果的**载体**）：
#: `projectile`（Log/BarbLog 滚动弹、箭矢）、`area_effect`（Poison/Heal 等）、`bomb`。
#: 与 `rl/observation.py::_TYPE_ALIAS` 同一集合、同一理由（它们不是部队/建筑，溅射打不到）。
_EFFECT_BODY_TYPES = frozenset({"projectile", "area_effect", "bomb"})


def _is_effect_body(e) -> bool:
    """该实体是不是**效果载体**（不是可被法术命中的部队/建筑/塔）。"""
    t = getattr(getattr(e, "data", None), "type", "") or ""
    return t in _EFFECT_BODY_TYPES


def _spell_has_enemy_target(battle, player_id: int, pos: Position, radius: float) -> bool:
    """溅射半径内是否有存活敌方目标（塔/建筑/部队）。命中口径与引擎溅射一致：
    距离 ≤ 半径 + 目标碰撞半径。"""
    opp = 1 - player_id
    for e in battle.entities.values():
        if not getattr(e, "is_alive", True):
            continue
        if getattr(e, "player", None) != opp:
            continue
        if _is_effect_body(e):
            #: ★ 2026-09-22：**效果载体不算"敌方目标"**。先例：IL 策略在 t=1.0 把 Fireball 丢到
            #: 敌方王塔格，唯一"理由"是场上有一枚 **对方 Log 的滚动弹**（`LogProjectileRolling`，
            #: player=1、id>6）⇒ 旧谓词判"溅射内有敌方目标"⇒ 放行。滚动弹无血、打不掉，
            #: 是**假目标**。修掉后"只罩塔"的落点才会真正落到 EV 闸门上去判。
            continue
        col = getattr(getattr(e, "data", None), "collision_radius", 0.0) or 0.0
        if pos.distance_to(e.position) <= radius + col + 1e-9:
            return True
    return False


#: —— 9h 前段法术对塔 EV 闸门（2026-09-07 100k 取证定稿，AGENTS.md 优先级1）——
#: 取证：37 次砸塔 100% 选择型（被迫=0）、68% 纯空砸塔，前段砸塔率 16%→54% 不降反升。
#: 软惩罚（edw×费用 ≈ −1.8/次）未被吸收 → 升级为硬约束：
#: 伤害型法术落点若**只**罩到对手皇冠塔（无任何部队/建筑受益）且
#: `对塔伤害折费 < edw×费用`（前段 1 费≈500 塔 HP），则该落点非法；双倍期放行。
#: 数值口径：对塔伤害 = spell_module 引擎标定（首次调用时标定+缓存）；
#: edw 与奖励 config.DEFAULT_REWARD.elixir_diff_weight 同源（消费者注入，缺省 0.5）。
TOWER_HP_PER_ELIXIR_EARLY = 500.0   # 与 spell_module.TOWER_HP_PER_ELIXIR 一致
SPELL_EV_EDW = 0.5                  # 奖励经济前段费差权重（economy 预设）

_spell_tower_dmg_cache = {}


def _spell_tower_damage(card_name: str) -> float:
    """对塔伤害（引擎标定缓存）。标定失败（环境异常）返回 0.0 = 闸门自动放行。"""
    if card_name in _spell_tower_dmg_cache:
        return _spell_tower_dmg_cache[card_name]
    dmg = 0.0
    try:
        from spell_module import get_spell_profile
        prof = get_spell_profile(card_name)
        if prof.get("deals_damage"):
            dmg = float(prof.get("tower_damage") or 0.0)
    except Exception:
        dmg = 0.0
    _spell_tower_dmg_cache[card_name] = dmg
    return dmg


def _spell_covers_non_tower(battle, player_id: int, pos: Position, radius: float) -> bool:
    """溅射半径内是否有对手的**非塔**目标（部队/建筑）。有 → 法术有正事可干，放行。"""
    opp = 1 - player_id
    tower_ids = {1, 2, 3, 4, 5, 6}
    for e in battle.entities.values():
        if not getattr(e, "is_alive", True):
            continue
        if getattr(e, "player", None) != opp:
            continue
        if e.id in tower_ids:
            continue
        if _is_effect_body(e):
            continue          # ★ 2026-09-22：效果载体不是"有正事可干"的目标（同上）
        col = getattr(getattr(e, "data", None), "collision_radius", 0.0) or 0.0
        if pos.distance_to(e.position) <= radius + col + 1e-9:
            return True
    return False


def _spell_tower_ev_illegal(battle, player_id: int, card_name: str, pos: Position,
                            card_info: "Card" = None) -> bool:
    """前段"纯砸塔"落点非法判定（空砸闸门的对塔特化加强版）。

    条件（全部满足才拒）：
    1. 双倍期前（battle.time < 120，双倍期奖励口径本身把砸塔调成近正 EV）；
    2. 伤害型法术、有半径（无半径/无标定数据 → 放行，不误伤）；
    3. 落点罩得到对手存活**塔**（公主塔或王塔）；
       ★ 2026-09-22 修复：原实现只判公主塔，理由写作「王塔在公主塔后面，砸到公主塔必含
       王塔误差」。该理由对**正面贴公主塔**的落点成立，但对**两座公主塔之间的中路口**
       （本地 x≈9、y≈25~29）不成立：那里到公主塔 5.0~6.5（> 半径+1.4）却到王塔 ≤ 3.9
       ⇒ 「只罩王塔」的落点走到本函数的**提前 return**，连条件 4/5 都不跑 ⇒ 合法。
       实测（IL 策略自对弈，`runs/il_readout_mixR00/025/10`）：该通道被当成「开局第一手
       火球砸敌方王塔」的合法路线（7 次、每次恰好 206 血、7/7 在 t ≤ 2.5 s）。
       现在王塔纳入判定；王塔的估值沿用奖励侧同源闸门（`tower_value_mult(king=True)`：
       两座公主塔存活时单位血价值 ×0.05 ≈ 0）⇒ 前段满血王塔的中路空砸被判非法，
       而**残血王塔**（任一公主塔被破后恢复全价）仍会自动合法。
    4. 落点半径内**无**对手非塔目标（部队/建筑）——有即放行；
    5. `对塔伤折费 < edw×费用`：标定对塔伤 / 500 < edw×卡费（前段经济账）。
       **塔血差异化定价（2026-09-10）**：对塔伤按罩到的存活公主塔的残血加权
       （低血塔单位血价值凹形溢价）——残血斩杀落点在更低的血线上自动合法
       （与 belief_planner 的 tower_value 同源；满血塔行为不变）。
    """
    if battle.time >= 120.0:
        return False
    if not _spell_deals_damage(card_name, card_info):
        return False
    radius = _spell_radius_m(card_name, card_info)
    if radius <= 0.0:
        return False
    dmg = _spell_tower_damage(card_name)
    if dmg <= 0.0:
        return False
    # 只对"罩得到存活公主塔"的落点判定（王塔在公主塔后面，砸到公主塔必含王塔误差，
    # 不重复判王塔； princess 全破后纯砸王塔同样按条件 3-5 判）
    opp = 1 - player_id
    hits_princess = False
    opp_alive_princess = 0
    for tid in ((1, 2) if opp == 1 else (3, 4)):
        tw = battle.entities.get(tid)
        if tw is None or not tw.is_alive:
            continue
        opp_alive_princess += 1
        col = getattr(tw.data, "collision_radius", 0.0) or 0.0
        if pos.distance_to(tw.position) <= radius + col + 1e-9:
            hits_princess = True
    #: ★ 2026-09-22：王塔一并判定（见 docstring 条件 3）。塔实体按名字找，不写死 id。
    king_ent = None
    for e in battle.entities.values():
        if (getattr(e, "player", None) == opp and e.is_alive
                and (getattr(e, "name", "") or "") == "KingTower"):
            king_ent = e
            break
    hits_king = False
    if king_ent is not None:
        col_k = getattr(getattr(king_ent, "data", None), "collision_radius", 0.0) or 0.0
        if pos.distance_to(king_ent.position) <= radius + col_k + 1e-9:
            hits_king = True
    if not (hits_princess or hits_king):
        return False
    if _spell_covers_non_tower(battle, player_id, pos, radius):
        return False
    cost = card_info.elixir if card_info is not None else Card(card_name).elixir
    if cost <= 0:
        return False
    # 残血加权：罩到的公主塔取**最残**（残血塔边际价值最高 → mult 最大）；未罩到的
    # 存活公主塔也计入"存活数"（王塔贬值闸门只关心公主塔是否都活着，与落点无关）
    from rl.env_wrapper import tower_value_mult  # 惰性 import，避免循环依赖
    best_mult = 1.0
    if hits_princess:
        for tid in ((1, 2) if opp == 1 else (3, 4)):
            tw = battle.entities.get(tid)
            if tw is None or not tw.is_alive:
                continue
            col = getattr(tw.data, "collision_radius", 0.0) or 0.0
            if pos.distance_to(tw.position) > radius + col + 1e-9:
                continue
            hp = float(tw.hp)
            ratio = hp / 3052.0 if hp > 0.0 else 0.0
            mult = tower_value_mult(ratio, king=False,
                                    princesses_alive=opp_alive_princess)
            best_mult = max(best_mult, mult)   # 最残塔 = mult 最大
    elif king_ent is not None:
        #: 只罩王塔：按**王塔自身**残血 + 公主塔存活数取倍数（与奖励/MCTS/planner 同源）。
        #: 两座公主塔都活着 ⇒ ×0.05 ≈ 0 ⇒ 前段中路满血王塔空砸必判非法；
        #: 任一公主塔被破 ⇒ 恢复全价（残血斩杀落点在该血线上自动合法）。
        hp = float(king_ent.hp)
        max_hp = float(getattr(getattr(king_ent, "data", None), "hp", 0.0) or 0.0) or 4824.0
        ratio = hp / max_hp if hp > 0.0 else 0.0
        best_mult = tower_value_mult(ratio, king=True,
                                     princesses_alive=opp_alive_princess)
    dmg_eff = dmg * best_mult
    return dmg_eff / TOWER_HP_PER_ELIXIR_EARLY < SPELL_EV_EDW * cost - 1e-9


#: —— 8h 不裸下：圣水无优势时禁止“单独放高承诺进攻单位”（用户口径）——
#: 例：单独下 MiniPekka，对方手里有 Archers 可解；只有我方多 3~4 费、能用
#: 法术破防时这波进攻才有意义。不满足 → 模型必须攒费或同刻多卡协同进攻。
#: 例外：对方已压境（防守紧急）或对方出不了手（费不够最低手牌费）时不拦。
SOLO_LEAD_ELIXIR = 3.0
#: 裸下受限的高承诺单位（用户例子 + 坦克裸下送费）；其余便宜卡/后排可单放。
SOLO_COMMIT_CARDS = frozenset({"MiniPekka", "Giant"})


def _opp_min_hand_cost(battle, player_id: int) -> Optional[float]:
    """对手手牌最低可出费用；手牌为空/国王已倒 → None（=对手出不了手）。"""
    opp = battle.players[1 - player_id]
    if opp.king_tower_hp <= 0:
        return None
    costs = [_card_cost(opp, c) for c in opp.cycle[:4]]
    costs = [c for c in costs if c is not None]
    return min(costs) if costs else None


def _enemy_in_my_half(battle, player_id: int) -> bool:
    """对方是否有单位已进入我半场（压境 → 防守优先，裸下闸门放行）。"""
    opp = 1 - player_id
    for e in battle.entities.values():
        if not getattr(e, "is_alive", True):
            continue
        if getattr(e, "player", None) != opp:
            continue
        y = e.position.y
        if (player_id == 0 and y <= 16.0) or (player_id == 1 and y >= 16.0):
            return True
    return False


def solo_commit_blocked(battle, player_id: int, card_name: str,
                        own_elixir: float) -> bool:
    """高承诺单卡裸下是否被禁止：
    - 卡不在受限名单 → 放行；
    - 对方无法出手（国王倒/费不够手牌最低费）→ 放行；
    - 对方压境（防守响应）→ 放行；
    - 己方圣水 - 对方圣水 ≥ 3 → 放行（有费差可用法术/多卡破防）；
    - 否则禁止（需攒费或多卡协同）。
    """
    if card_name not in SOLO_COMMIT_CARDS:
        return False
    min_opp = _opp_min_hand_cost(battle, player_id)
    if min_opp is None:
        return False
    opp = battle.players[1 - player_id]
    if opp.elixir < min_opp - 1e-9:
        return False
    if _enemy_in_my_half(battle, player_id):
        return False
    if own_elixir - opp.elixir >= SOLO_LEAD_ELIXIR - 1e-9:
        return False
    return True


#: —— 8h 坦克后屯兵（用户口径）：远程/普通飞行/高输出后排要放在高血量单位
#: （Giant/Knight）**后面**，间距 ≥ 后排攻击距离；移速比坦克快的还要更靠后，
#: 防止后排超车裸奔/抢仇恨。Giant/Knight 是前排，MiniPekka 高输出按后排保护。
TANK_CARDS = frozenset({"Giant", "Knight"})
BACKLINE_CARDS = frozenset({"Musketeer", "Archer", "Minions", "MiniPekka"})
TANK_LANE_HALF_W = 5.0      # 同路判宽（世界单位）：只约束同一路推进的支援
BACKLINE_SPEED_SCALE = 2.0   # 移速差 → 额外后置距离（世界单位 / 速度差）
TANK_DEFENSE_R = 4.0        # 落点附近有敌军=防守响应 → 几何门放行（防误伤防守）
#: 坦克进入“推进段”才启用几何门（避免在国王塔旁的早期铺场被过度限制）：
#: 推进方向 = 朝河（P0 y 增 / P1 y 减），带内 = 离国王塔已走出的行程。
_TANK_ACTIVE_BAND = {0: (8.0, 25.0), 1: (7.0, 24.0)}


def _active_push_tanks(battle, player_id: int):
    """本方可作推进前排的存活坦克（Giant/Knight），且已离开国王塔进入推进段。"""
    lo, hi = _TANK_ACTIVE_BAND.get(player_id, (0.0, 99.0))
    out = []
    for e in battle.entities.values():
        if not getattr(e, "is_alive", True):
            continue
        if getattr(e, "player", None) != player_id:
            continue
        name = getattr(e, "name", "") or ""
        if name not in TANK_CARDS:
            continue
        y = e.position.y
        if lo <= y <= hi:
            out.append(e)
    return out


#: 后排/坦克 Card 静态解析缓存：(卡名, Card.default_level) → Card。
#: Card 构造 ~7µs 且 backline gap 检查逐坦克调用；数值只随 (名, 级) 变化，
#: Card 实例上可变状态仅 set_level 产物，同 level 重建结果恒定 → 可安全复用。
#: 注意：Card.default_level 被 BattleState 构造时切换，缓存键必须含它。
_card_static_cache = {}


def _card_cached(card_name: str) -> "Card":
    key = (card_name, Card.default_level)
    c = _card_static_cache.get(key)
    if c is None:
        c = Card(card_name)
        _card_static_cache[key] = c
    return c


def _backline_min_gap_m(back_card: str, tank_card: str,
                        back_info: "Card" = None) -> float:
    """后排与坦克的最小纵向间距（世界单位）：
    基础 = 后排攻击距离；后排比坦克快 → 每 1 速度差再多后置 2 单位。"""
    b = back_info if back_info is not None else _card_cached(back_card)
    t = _card_cached(tank_card)
    base = float(getattr(b, "range", 0.0) or 0.0)
    dv = max(0.0, float(getattr(b, "speed", 0.0) or 0.0)
             - float(getattr(t, "speed", 0.0) or 0.0))
    return base + dv * BACKLINE_SPEED_SCALE


def _backline_placement_illegal(battle, player_id: int, card_name: str,
                                pos: Position, card_info: "Card" = None) -> bool:
    """后排落点几何闸门：同路有推进坦克时，落点必须位于某坦克之后且留足
    最小间距（否则会抢仇恨/超车裸奔）。异路防守部署不受影响。"""
    if card_name not in BACKLINE_CARDS:
        return False
    # 防守响应放行：落点附近有敌军（解牌/拦截），几何门不拦
    for e in battle.entities.values():
        if not getattr(e, "is_alive", True):
            continue
        if getattr(e, "player", None) != 1 - player_id:
            continue
        if pos.distance_to(e.position) <= TANK_DEFENSE_R:
            return False
    tanks = [e for e in _active_push_tanks(battle, player_id)
             if abs(e.position.x - pos.x) <= TANK_LANE_HALF_W]
    if not tanks:
        return False
    for tk in tanks:
        if player_id == 0:
            gap_along = tk.position.y - pos.y
        else:
            gap_along = pos.y - tk.position.y
        need = _backline_min_gap_m(card_name, tk.name, card_info)
        if gap_along >= need - 0.5:
            return False  # 能跟在至少一个推进坦克后面 → 合法
    return True


def _hits_dead_enemy_tower(battle, player_id: int, pos: Position) -> bool:
    """法术落点是否贴着已毁敌方塔本体（塔实体 id≤6 且 is_alive=False 永留场）。"""
    opp = 1 - player_id
    for e in battle.entities.values():
        if e.player != opp or e.is_alive:
            continue
        name = getattr(e, "name", "") or ""
        eid = getattr(e, "id", None)
        if "Tower" not in name and not (eid is not None and eid <= 6):
            continue  # 非塔的死亡实体会被 step 清理；防御式双保险
        if pos.distance_to(e.position) <= DEAD_TOWER_BODY_R:
            return True
    return False


def _position_legal(battle, player_id: int, card_name: str, pos: Position,
                    card_info: "Card" = None) -> bool:
    """复刻 battle.deploy_card 中的部署区域合法性（法术额外挡已毁塔本体 + 空砸闸门）。

    7h：法术可打任意格，但不得砸在**已毁敌方塔本体**上——引擎里法术是打坐标，
    已毁塔 is_alive=False 会被溅射跳过，落在那里=纯空砸（也不会转伤国王塔）。
    8h：伤害型法术必须罩到 ≥1 个存活敌方目标（塔/建筑/单位），否则视为纯空砸拒绝。
    P0 优化：card_info 由调用方传入可省逐格 Card 构造（legal_cells 576 格热路径）；
    缺省仍自建（提交路径 validate_bundle 语义不变）。
    """
    if card_info is None:
        card_info = Card(card_name)
    if card_info.type == "spell":
        if _hits_dead_enemy_tower(battle, player_id, pos):
            return False
        if _spell_deals_damage(card_name, card_info):
            radius = _spell_radius_m(card_name, card_info)
            if radius > 0.0 and not _spell_has_enemy_target(battle, player_id, pos, radius):
                return False
            # 9h 前段纯砸塔 EV 闸门：无部队/建筑可溅、账面亏费 → 非法
            if radius > 0.0 and _spell_tower_ev_illegal(battle, player_id, card_name, pos,
                                                        card_info):
                return False
        return True
    if battle.is_position_occupied_by_building(pos, 0.0):
        return False
    # 王塔身后 1 格宽禁建筑（与 battle.deploy_card 同源，塔矩形几何 2026-09-09）
    if card_info.type == "building" and battle.arena.is_behind_king(pos, player_id):
        return False
    if player_id == 0:
        if pos.y <= 1.0 and (pos.x <= 6.0 or pos.x > 12.0):
            return False
        if pos.y >= 21.0:
            return False
        if pos.y >= 15.0:
            if pos.x <= 9:
                if battle.players[1].left_tower_hp > 0:
                    return False
            else:
                if battle.players[1].right_tower_hp > 0:
                    return False
    else:
        if pos.y > 31.0 and (pos.x <= 6.0 or pos.x > 12.0):
            return False
        if pos.y <= 10:
            return False
        if pos.y <= 17.0:
            if pos.x <= 9:
                if battle.players[0].left_tower_hp > 0:
                    return False
            else:
                if battle.players[0].right_tower_hp > 0:
                    return False
    # 8h 坦克后屯兵：后排落点必须待在推进坦克后面并留出攻击距离间距
    if _backline_placement_illegal(battle, player_id, card_name, pos, card_info):
        return False
    return True


def legal_cells(battle, player_id: int, card_name: str) -> np.ndarray:
    """返回 (32,18) bool：该卡在玩家本地网格中可以部署的格子。

    7h：法术任意位置但**排除已毁敌方塔本体格**（砸已炸掉的塔=纯空砸）。
    8h：伤害型法术再加空砸闸门——只有溅射能罩到存活敌方目标的格子才合法；
    其余按规则——本地格子 (x, y) 经 sub_position 换算为世界坐标后
    由 _position_legal 校验（与提交路径完全同源，P0-4）。
    P0 优化（2026-09-08）：Card 构造提出 576 格循环（此前每格重建 1-3 张，
    占本函数耗时 ~92%）；判定逻辑逐位不变。
    """
    cells = np.ones((GRID_H, GRID_W), dtype=bool)
    p = battle.players[player_id]
    eff = _effective_card(p, card_name)
    eff_info = Card(eff)
    is_spell = eff_info.type == "spell"
    if is_spell:
        deals_dmg = _spell_deals_damage(eff, eff_info)
        radius = _spell_radius_m(eff, eff_info) if deals_dmg else 0.0
        ev_gate = radius > 0.0 and deals_dmg
        for y in range(GRID_H):
            for x in range(GRID_W):
                pos = sub_position(player_id, x, y)
                if _hits_dead_enemy_tower(battle, player_id, pos):
                    cells[y, x] = False
                elif radius > 0.0 and not _spell_has_enemy_target(battle, player_id, pos, radius):
                    cells[y, x] = False
                elif ev_gate and _spell_tower_ev_illegal(battle, player_id, eff, pos,
                                                         eff_info):
                    cells[y, x] = False
        return cells
    for y in range(GRID_H):
        for x in range(GRID_W):
            if not _position_legal(battle, player_id, eff, sub_position(player_id, x, y),
                                   eff_info):
                cells[y, x] = False
    return cells


def _ready_ability_cost(battle, player_id: int) -> Optional[float]:
    """返回场上首个就绪英雄技能的耗蓝；无就绪英雄返回 None。

    与引擎 battle.use_ability 的就绪判定保持一致（必要非充分：引擎还会先调
    holder.use_ability()，掩码层只做静态预判）。
    """
    p = battle.players[player_id]
    if p.king_tower_hp <= 0:
        return None
    for e in battle.entities.values():
        if not e.is_alive or e.player != player_id:
            continue
        ability = getattr(e.data, "ability", None)
        if not ability or getattr(e, "ability_cd", 0) > 0:
            continue
        if not hasattr(getattr(e, "entity_holder", None), "use_ability"):
            continue
        return float(ability.get("manaCost", 0))
    return None


def ability_legal(battle, player_id: int, elixir_override: float = None,
                  already_used: bool = False) -> bool:
    """bundle 中是否还能触发英雄技能。

    - already_used：bundle 已含技能 → False；
    - elixir_override：bundle 内模拟扣费后的剩余圣水（P1-6）。
    """
    if already_used:
        return False
    cost = _ready_ability_cost(battle, player_id)
    if cost is None:
        return False
    elixir = battle.players[player_id].elixir if elixir_override is None else elixir_override
    return elixir >= cost


def ability_mana(battle, player_id: int) -> Optional[float]:
    """返回就绪英雄技能的耗蓝；无就绪英雄返回 None（不再用 0 作哨兵，P2）。"""
    return _ready_ability_cost(battle, player_id)


def validate_bundle(battle, player_id: int, bundle: ActionBundle):
    """整包校验（不修改任何状态）：任一子动作非法则拒绝整包。

    返回 (ok, reason, resolved_actions)：
    - resolved_actions: [(card_name, SubAction), ...]，按决策时刻手牌解析，
      避免 commit 时因循环前移导致槽位错位；技能统一为 ("__ability__", sa)。

    v1 限制：
    - 只能用决策开始时已在手牌（cycle[:4]）的牌；
    - 同一槽位不可重复；
    - Mirror 按引擎语义（重放 last_card、费用 +1）校验，允许单卡与多卡 bundle。
    """
    p = battle.players[player_id]
    elixir = p.elixir
    used = set()
    has_ability = False
    resolved = []
    for sa in bundle.sub_actions:
        if sa.kind == "ability":
            if has_ability:
                return False, "bundle 内重复技能", resolved
            cost = ability_mana(battle, player_id)
            if cost is None:
                return False, "无就绪英雄技能", resolved
            if elixir < cost:
                return False, "技能圣水不足", resolved
            elixir -= cost
            has_ability = True
            resolved.append(("__ability__", sa))
            continue
        if sa.slot == 0:
            continue
        if sa.slot < 1 or sa.slot > K_MAX:
            return False, f"slot {sa.slot} 越界", resolved
        if sa.slot in used:
            return False, f"slot {sa.slot} 重复", resolved
        if sa.x < 0 or sa.x >= GRID_W or sa.y < 0 or sa.y >= GRID_H:
            return False, f"坐标越界 ({sa.x},{sa.y})", resolved
        card = p.cycle[sa.slot - 1]
        eff = _effective_card(p, card)
        if not _slot_playable(p, card, elixir):
            return False, f"{card} 不可出（圣水/手牌/塔状态）", resolved
        if not _position_legal(battle, player_id, eff, sa.to_position(player_id)):
            return False, f"{card} 部署位置非法 ({sa.x},{sa.y})", resolved
        elixir -= _card_cost(p, card) or 0.0
        used.add(sa.slot)
        resolved.append((card, sa))
    # 8h 不裸下兜底（mask 只管“空 bundle 首卡”，BC/旧回放里的单卡裸下在这里拒绝）：
    # 整包只放 1 张高承诺单位且无圣水优势 → 拒绝
    deploys = [c for c, _ in resolved if c != "__ability__"]
    if len(deploys) == 1 and solo_commit_blocked(battle, player_id, deploys[0], p.elixir):
        return False, "不裸下: 无圣水优势时禁止单独放高承诺卡", resolved
    return True, "ok", resolved
