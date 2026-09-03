"""脚本对手 / 基线模型（random / heuristic / 卡组完全随机）。

- ``ScriptedPolicy``：统一的可作任意一侧（player-0 或 player-1 对手）的脚本策略；
  - mode="random"：从掩码随机选合法出牌；
  - mode="heuristic"：同上（当前启发式对手即基于掩码采样，P0-3/P0-4 修复后已正常）；
  - pool=None：固定默认卡组；pool=list：**每局重新随机采样 8 张卡**（卡组完全随机模型）。

卡池由引擎数据构建（``build_card_pool``）：过滤 0 费/生成物/塔，实测可 deploy 的卡。
"""

import os
import sys
import random

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import numpy as np

from card_utils import Card, card_data
from rl.action_bundle import ActionBundle, K_MAX

DECK_SIZE = 8


def build_card_pool() -> list:
    """引擎可部署的卡池（实测 139 张；Mirror 依赖 last_card，放池内由掩码门控）。"""
    pool = []
    for n in card_data.keys():
        try:
            c = Card(n)
        except Exception:
            continue
        if n.startswith("King_") or "Tower" in n:
            continue
        cost = getattr(c, "elixir", 0)
        if cost is None or cost <= 0:
            continue
        if getattr(c, "type", None) not in ("character", "spell", "building"):
            continue
        pool.append(n)
    return sorted(pool)


def sample_deck(rng, pool) -> list:
    """从卡池随机采样 8 张互不相同的卡（卡组完全随机）。"""
    return list(rng.sample(pool, min(DECK_SIZE, len(pool))))


class ScriptedPolicy:
    """脚本策略：掩码采样合法动作，可作 player-0 / player-1 对手。

    Attributes:
        pool: None → 固定默认卡组；list → 每局 deck() 从卡池重采样随机 8 张。
        deck_pool: list[deck] → 每局 deck() 从该卡组集合里**随机抽一副完整卡组**
            （三分类卡组 / 全 200 卡组模型用）。
        env: 由 play_pair / 训练循环注入，供 player-1 对手调用。
    """

    def __init__(self, mode="random", pool=None, deck_pool=None, seed=0, env=None):
        if mode not in ("random", "heuristic"):
            raise ValueError(f"未知 mode: {mode}")
        self.mode = mode
        self.pool = list(pool) if pool else None
        self.deck_pool = list(deck_pool) if deck_pool else None
        self.seed = seed
        self.rng = random.Random(seed)
        self.env = env

    def deck(self):
        """本局使用的卡组：有 deck_pool 随机抽一副；有 pool 重采样 8 张；否则固定默认。"""
        if self.deck_pool:
            pick = self.rng.choice(self.deck_pool)
            return list(pick["cards"]) if isinstance(pick, dict) else list(pick)
        if self.pool is not None:
            return sample_deck(self.rng, self.pool)
        return None

    def play(self, env, player_id: int) -> ActionBundle:
        """从 player_id 的合法掩码随机选一子动作。"""
        mask = env.get_action_mask_for(player_id)
        slots = np.flatnonzero(mask["slots"])
        if slots.size == 0:
            return ActionBundle.noop()
        slot = int(self.rng.choice(slots))
        cells = np.flatnonzero(mask["cells"][slot])
        if cells.size == 0:
            return ActionBundle.noop()
        cell = int(self.rng.choice(cells))
        return ActionBundle.from_single(slot + 1, int(cell % 18), int(cell // 18))

    def __call__(self, obs):
        """player-1 对手接口（env 由外部注入）。"""
        if self.env is None:
            raise RuntimeError("ScriptedPolicy 需要先注入 env（.env = ...）")
        return self.play(self.env, 1)
