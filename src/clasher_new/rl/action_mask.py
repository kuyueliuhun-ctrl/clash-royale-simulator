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
    """复刻 battle.deploy_card 中的部署区域合法性（法术额外挡已毁塔本体）。

    7h：法术可打任意格，但不得砸在**已毁敌方塔本体**上——引擎里法术是打坐标，
    已毁塔 is_alive=False 会被溅射跳过，落在那里=纯空砸（也不会转伤国王塔）。
    """
    card_info = Card(card_name)
    if card_info.type == "spell":
        return not _hits_dead_enemy_tower(battle, player_id, pos)
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

    法术任意位置但**排除已毁敌方塔本体格**（7h：砸已炸掉的塔=纯空砸）；
    其余按规则——本地格子 (x, y) 经 sub_position 换算为世界坐标后
    由 _position_legal 校验（与提交路径完全同源，P0-4）。
    """
    cells = np.ones((GRID_H, GRID_W), dtype=bool)
    eff = _effective_card(battle.players[player_id], card_name)
    if Card(eff).type == "spell":
        for y in range(GRID_H):
            for x in range(GRID_W):
                if _hits_dead_enemy_tower(battle, player_id, sub_position(player_id, x, y)):
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
    return True, "ok", resolved
