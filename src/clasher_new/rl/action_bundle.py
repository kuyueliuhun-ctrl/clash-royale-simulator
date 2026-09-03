"""同刻多卡动作包 ActionBundle（规划文档 3.4.1）。

核心语义：
- 一个决策步 = 一个 ActionBundle；
- bundle 内所有合法子动作在同一决策 tick 内提交（多次 deploy_card、期间不推进 battle.step）；
- 整包校验、整包提交：默认任一子动作非法即拒绝整包并施加惩罚，避免半执行状态。

坐标契约（跨模块统一，见 docs/rl_review_fix_plan.md §5）：
- ``SubAction(x, y)`` 一律是**玩家本地坐标**（0..17 / 0..31）；
- 世界坐标换算只通过 :func:`sub_position` / :meth:`SubAction.to_position` 一个入口，
  掩码层（legal_cells）与提交层（deploy_card）共用，杜绝镜像分裂。
"""

import os
import sys

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from core import Position
from card_utils import Card

#: 单个决策步最多同时打出的卡数（规划建议 K_max = 4）
K_MAX = 4


def sub_position(player_id: int, x: int, y: int) -> Position:
    """玩家本地网格坐标 (x, y) → 世界坐标（唯一换算入口）。

    - player 0：本地即世界，(x+0.5, y+0.5)；
    - player 1：镜像，(17.5-x, 31.5-y)。
    """
    if player_id == 0:
        return Position(x + 0.5, y + 0.5)
    return Position(17.5 - x, 31.5 - y)


@dataclass
class SubAction:
    """单卡子动作。

    - slot = 1..4 表示打出 cycle[slot-1]；slot = 0 为 no-op；
    - (x, y) 为**玩家本地**网格坐标（见模块 docstring 坐标契约）。

    kind = "deploy"  出牌（slot/x/y 生效）
    kind = "ability" 触发英雄技能（引擎 battle.use_ability 自动选取就绪英雄，slot/x/y 忽略）
    """

    kind: str = "deploy"
    slot: int = 0
    x: int = 0
    y: int = 0

    @classmethod
    def ability(cls) -> "SubAction":
        return cls(kind="ability")

    def card_name(self, player) -> Optional[str]:
        if self.kind != "deploy" or self.slot <= 0 or self.slot > K_MAX:
            return None
        return player.cycle[self.slot - 1]

    def to_position(self, player_id: int = 0) -> Position:
        return sub_position(player_id, self.x, self.y)

    def to_tuple(self) -> Tuple[int, int, int]:
        """旧接口兼容：(slot, y, x)。"""
        return (self.slot, self.y, self.x)


@dataclass
class ActionBundle:
    sub_actions: List[SubAction] = field(default_factory=list)

    def __post_init__(self):
        if len(self.sub_actions) > K_MAX:
            raise ValueError(f"ActionBundle 子动作数超过 K_MAX={K_MAX}")

    def add(self, slot: int, x: int, y: int) -> "ActionBundle":
        self.sub_actions.append(SubAction(kind="deploy", slot=slot, x=x, y=y))
        return self

    def add_ability(self) -> "ActionBundle":
        self.sub_actions.append(SubAction.ability())
        return self

    @property
    def size(self) -> int:
        return len(self.sub_actions)

    def to_tuple(self) -> Tuple[int, int, int]:
        """n<=1 兼容模式：转成旧版 (slot, y, x)。"""
        if self.size == 0:
            return (0, 0, 0)
        return self.sub_actions[0].to_tuple()

    @classmethod
    def from_single(cls, slot: int, x: int, y: int) -> "ActionBundle":
        return cls(sub_actions=[SubAction(slot=slot, x=x, y=y)])

    @classmethod
    def noop(cls) -> "ActionBundle":
        return cls(sub_actions=[])

    def contains_card(self, player, card_name: str) -> bool:
        return any(sa.card_name(player) == card_name for sa in self.sub_actions)
