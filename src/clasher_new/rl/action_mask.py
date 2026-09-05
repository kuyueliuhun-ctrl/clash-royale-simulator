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


def _spell_radius_m(card_name: str) -> float:
    """法术溅射半径（世界单位）；无半径数据返回 0.0（=不做闸门，保持旧语义）。"""
    data = getattr(Card(card_name), "data", None) or {}
    raw = data.get("radius")
    if raw is None:
        raw = ((data.get("projectileData") or {}).get("radius"))
    return (float(raw) if raw else 0.0) / 1000.0


def _spell_deals_damage(card_name: str) -> bool:
    """是否输出伤害的法术。伤害型法术受空砸闸门约束；增益/位移/召唤类法术放行。"""
    data = getattr(Card(card_name), "data", None) or {}
    pd = data.get("projectileData") or {}
    return (float(pd.get("damage") or 0.0) > 0.0
            or float(data.get("damage") or 0.0) > 0.0)


def _spell_has_enemy_target(battle, player_id: int, pos: Position, radius: float) -> bool:
    """溅射半径内是否有存活敌方目标（塔/建筑/部队）。命中口径与引擎溅射一致：
    距离 ≤ 半径 + 目标碰撞半径。"""
    opp = 1 - player_id
    for e in battle.entities.values():
        if not getattr(e, "is_alive", True):
            continue
        if getattr(e, "player", None) != opp:
            continue
        col = getattr(getattr(e, "data", None), "collision_radius", 0.0) or 0.0
        if pos.distance_to(e.position) <= radius + col + 1e-9:
            return True
    return False


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


def _position_legal(battle, player_id: int, card_name: str, pos: Position) -> bool:
    """复刻 battle.deploy_card 中的部署区域合法性（法术额外挡已毁塔本体 + 空砸闸门）。

    7h：法术可打任意格，但不得砸在**已毁敌方塔本体**上——引擎里法术是打坐标，
    已毁塔 is_alive=False 会被溅射跳过，落在那里=纯空砸（也不会转伤国王塔）。
    8h：伤害型法术必须罩到 ≥1 个存活敌方目标（塔/建筑/单位），否则视为纯空砸拒绝。
    """
    card_info = Card(card_name)
    if card_info.type == "spell":
        if _hits_dead_enemy_tower(battle, player_id, pos):
            return False
        if _spell_deals_damage(card_name):
            radius = _spell_radius_m(card_name)
            if radius > 0.0 and not _spell_has_enemy_target(battle, player_id, pos, radius):
                return False
        return True
    if battle.is_position_occupied_by_building(pos, 0.0):
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
    return True


def legal_cells(battle, player_id: int, card_name: str) -> np.ndarray:
    """返回 (32,18) bool：该卡在玩家本地网格中可以部署的格子。

    7h：法术任意位置但**排除已毁敌方塔本体格**（砸已炸掉的塔=纯空砸）。
    8h：伤害型法术再加空砸闸门——只有溅射能罩到存活敌方目标的格子才合法；
    其余按规则——本地格子 (x, y) 经 sub_position 换算为世界坐标后
    由 _position_legal 校验（与提交路径完全同源，P0-4）。
    """
    cells = np.ones((GRID_H, GRID_W), dtype=bool)
    eff = _effective_card(battle.players[player_id], card_name)
    if Card(eff).type == "spell":
        radius = _spell_radius_m(eff) if _spell_deals_damage(eff) else 0.0
        for y in range(GRID_H):
            for x in range(GRID_W):
                pos = sub_position(player_id, x, y)
                if _hits_dead_enemy_tower(battle, player_id, pos):
                    cells[y, x] = False
                elif radius > 0.0 and not _spell_has_enemy_target(battle, player_id, pos, radius):
                    cells[y, x] = False
        return cells
    for y in range(GRID_H):
        for x in range(GRID_W):
            if not _position_legal(battle, player_id, eff, sub_position(player_id, x, y)):
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
