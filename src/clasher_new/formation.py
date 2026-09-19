# -*- coding: utf-8 -*-
"""阵型（formation）：把「N 个单位出生在什么形状上」从**均匀圆环**改成**可声明的形状**。

## 为什么需要（2026-09-19）

- 两个引擎的出生几何**只有** `get_spawn_position` 的均匀圆环（两仓逐字相同），
  而 `gamedata.json` 里**没有任何阵型/形状字段**（只有 `summonRadius`；`SkeletonArmy`
  连它都没有，走 `card_utils.py` 的硬编码兜底 550/1000）。
- 后果被实测出来（`docs/skarmy_two_engine_observations_2026-09-19.md` §5）：
  `SkeletonArmy` 15 只塞进半径 0.55 的环 ⇒ 间距 0.23 而每只半径 0.5 ⇒
  **105 对里 75 对重叠** ⇒ 屏幕上看到的形状是**碰撞解算在 0.1 s 内炸出来的**，不是阵型。
- 更极端的例子：`RoyalHogs` 的 `summonRadius` 是 **1**（毫格口径下 = 0.001 格，全表唯一），
  4 只猪落在**几乎同一点** ⇒ 6 对全深度重叠。

## 用户口径（2026-09-19 原文）

> 「阵型机制确实需要，但我们先从一些简单的入手，如四只哥布林在空地呈正方形，
> 哥布林飞桶砸向防御塔正中央会散到塔后中央一只，上方左右角对角线各一只，
> 丢到空地则是一个小三角形。」

## ⚠️ 口径纪律（读代码前必看）

1. **形状偏移量是「用户口径 + 待标定假设」，不是从官方数据反推的**——仓内没有任何
   可依据的阵型数据（已取证）。所有自由参数集中在下面 `#: 待标定` 段落，便于按观测校正。
2. **唯一有硬依据的是「不重叠」**：间距由**碰撞半径**推出（`2r × SPACING_FACTOR`），
   这是可自证的（见 `scripts/probe_formation.py` 的 `min_side` 检查）。
3. **未列入形状表的卡一律逐字走原圆环**（`shape_for_card` 返回 `None`）⇒ 其余卡零改动。
   `scripts/selftest_formation.py` 有一条不变式专门守这一点。
"""
from __future__ import annotations

import math

__all__ = [
    "SQUARE", "TRIANGLE", "BARREL_TOWER", "BARREL_OPEN",
    "SHAPE_BY_CARD", "ARRIVAL_SHAPES", "SPACING_FACTOR",
    "BARREL_BEHIND_Y", "BARREL_FLANK_X", "BARREL_FLANK_Y",
    "shape_for_card", "arrival_shape", "shape_positions", "template_offsets",
]

SQUARE = "square"
TRIANGLE = "triangle"
BARREL_TOWER = "barrel_tower"
BARREL_OPEN = "barrel_open"

#: 间距系数：单位间距 = `2 · collision_radius · SPACING_FACTOR` ⇒ 恒 > 半径和 ⇒ 出生不重叠。
#: 1.05 = 留 5% 余量（碰撞解算每 tick 会把重叠推开，出生不重叠才谈得上「形状」）。
SPACING_FACTOR = 1.05

#: —— 待标定（用户口径的几何参数，单位 = 格）——
#: 飞桶落在塔上时：1 只在「塔后中央」，2 只在「上方左右角、沿对角线」。
#: 取 `|y| ≥ 塔半高 + 哥布林半径 = 1.5 + 0.5 = 2.0` ⇒ 落在塔矩形推出区**之外**，
#: 出生后不会被 `_push_troop_out_of_tower` 再挪一次（这一步是当前「左右各一只」
#: 看起来贴在塔腰上的原因，见 docs/formation_2026-09-19.md §1）。
BARREL_BEHIND_Y = 2.05
BARREL_FLANK_X = 2.05
BARREL_FLANK_Y = 2.05

#: 直接投放的卡 → 形状。键 = (卡数据名 `data['name']`, 出兵数)。
#: ⚠️ 只登记**用户点名的那一类**（n=4 → 正方形）；其余卡不在表内 = 行为逐字不变。
SHAPE_BY_CARD = {
    ("Goblins", 4): SQUARE,
    ("RoyalHogs", 4): SQUARE,
}

#: 落地出兵（投射物到达）→ 形状。键 = (出生卡数据名, 出兵数)，
#: 值 = (命中塔时用的形状, 空地时用的形状)。
#: 实测：只有哥布林飞桶会以 `count=3` 落地生 `Goblins`
#: （`projectileData.spawnCharacterCount = 3`，`character_to_card['Goblin'] = 'Goblins'`）。
ARRIVAL_SHAPES = {
    ("Goblins", 3): (BARREL_TOWER, BARREL_OPEN),
}


def _side(collision_radius):
    """不重叠的最小间距：`2r · SPACING_FACTOR`（r 缺省 0.5 = 常见小体型）。"""
    r = float(collision_radius) if collision_radius else 0.5
    return 2.0 * r * SPACING_FACTOR


def _square(s):
    h = s / 2.0
    return [(h, h), (-h, h), (-h, -h), (h, -h)]


def _equilateral(s):
    """正三角形，边长 s，一个顶点指向 +y（玩家 0 的前进方向）。"""
    h = s / (2.0 * math.sqrt(3.0))
    return [(0.0, 2.0 * h), (-s / 2.0, -h), (s / 2.0, -h)]


def template_offsets(shape, collision_radius=None):
    """形状 → 相对出生点的偏移列表（**玩家 0 的"前进 = +y"坐标系**）。未知形状 → None。"""
    if shape == SQUARE:
        return _square(_side(collision_radius))
    if shape == TRIANGLE:
        return _equilateral(_side(collision_radius))
    if shape == BARREL_OPEN:
        return _equilateral(_side(collision_radius))
    if shape == BARREL_TOWER:
        # 绝对格（锚在塔的 3×3 足迹上），**不**随碰撞半径缩放
        return [(0.0, BARREL_BEHIND_Y),
                (-BARREL_FLANK_X, BARREL_FLANK_Y),
                (BARREL_FLANK_X, BARREL_FLANK_Y)]
    return None


def shape_positions(shape, position, player, collision_radius=None):
    """形状 → 世界坐标列表。未知形状返回 `None`（调用方回落到原圆环）。

    镜像：与旧圆环的 `angle += pi` **同构**（180° 旋转，dx/dy 同时取反）。
    """
    offs = template_offsets(shape, collision_radius)
    if not offs:
        return None
    from core import Position
    out = []
    for dx, dy in offs:
        if player == 1:
            dx, dy = -dx, -dy
        out.append(Position(position.x + dx, position.y + dy))
    return out


def shape_for_card(card_data_name, count):
    """直接投放的卡 → 形状名；未登记返回 None（= 走原圆环）。"""
    if card_data_name is None or count is None:
        return None
    return SHAPE_BY_CARD.get((str(card_data_name), int(count)))


def arrival_shape(card_data_name, count, position, battle_state):
    """落地出兵 → 形状名（按是否命中存活塔在两种形状间二选一）；未登记返回 None。"""
    pair = ARRIVAL_SHAPES.get((str(card_data_name), int(count))) if card_data_name else None
    if not pair:
        return None
    on_tower, on_open = pair
    return on_tower if on_tower_rect(position, battle_state) else on_open


def on_tower_rect(position, battle_state):
    """落点是否落在某座塔的**矩形足迹**内（= 用户说的「砸向防御塔正中央」）。

    塔足迹口径与 `arena.towers` 一致：`(中心, 半宽, 半高, player)`，
    半开区间 `|dx| ≤ hw 且 |dy| ≤ hh`（与 `battle.py` 的塔矩形判定同源）。
    """
    try:
        towers = getattr(getattr(battle_state, "arena", None), "towers", None) or []
    except Exception:                                    # noqa: BLE001 — 兜底不得崩
        return False
    for entry in towers:
        try:
            center, hw, hh = entry[0], entry[1], entry[2]
        except Exception:                                # noqa: BLE001
            continue
        if abs(position.x - center.x) <= hw and abs(position.y - center.y) <= hh:
            return True
    return False
