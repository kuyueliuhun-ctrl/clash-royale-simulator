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


#: 四种卡组（2026-09-07 定稿，docs/four_decks_manual.md）：
#: 对齐 FirstLight CR 训练哲学的四个互补 archetype——速攻循环 / 推进 / 自闭攻城 /
#: 双线快攻。全部 8/8 引擎可用（batch smoke 通过），数值口径见手册。
FOUR_DECK_SET = [
    # 速猪 2.6（均费 2.62）：野猪快攻 + Cannon 拉扯 + 极速循环
    ["HogRider", "IceGolemite", "IceSpirits", "Musketeer",
     "Cannon", "Skeletons", "Fireball", "Log"],
    # 皇家巨人（均费 3.12）：桥头平推 + 空中护航
    ["RoyalGiant", "Hunter", "MegaMinion", "Bats",
     "Skeletons", "Cannon", "Log", "Fireball"],
    # X 弩（均费 3.25）：自闭攻城 + 层层保弩
    ["Xbow", "Tesla", "Skeletons", "IceWizard",
     "Archer", "Knight", "Log", "Fireball"],
    # 双线快攻（均费 3.38）：野猪/皇家猪双路分推
    ["HogRider", "RoyalHogs", "IceWizard", "Musketeer",
     "Valkyrie", "Zap", "Fireball", "Skeletons"],
]


def build_card_pool() -> list:
    """引擎可部署的卡池（实测 139 张；Mirror 依赖 last_card，放池内由掩码门控）。"""
    pool = []
    for n in card_data.keys():
        try:
            c = Card(n)
        except Exception:
            continue
        if n.startswith("King_"):
            # 王塔/塔形态卡都有 King_ 前缀（King_PrincessTowers/King_CannonTowers...）。
            # 注意不能用 "Tower" in n / endswith("Tower")：会误伤 BombTower、
            # InfernoTower（名字里带 Tower 的合法建筑卡，registry 均 implemented）。
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


class SelfDefenderPolicy:
    """真防守脚本对手（9j，A 层对手池组件）：把 simulate_exchange.script_defender
    的确定性反制逻辑包装成 env 对手——威胁出现时从手牌选 (DPS+HP/15)/费 最优的
    反制部队、塔前迎击线落点；无威胁时按低频缓出（60% 帧停手，其余从掩码随机，
    模拟"控场但会响应"的人类基线）。

    与 ScriptedPolicy(mode="heuristic")（= mask 随机）的本质区别：会真的把部队
    放在过河敌军的行进路线上 → 单边堆牌/换家策略在这里讨不到便宜。

    纯函数式反制：script_defender(sim, defender_id) 是无副作用纯函数（selftest
    test_simulate_exchange 对账），候选落点由本类经 env.battle 的合法性真实执行。
    """

    def __init__(self, seed=0, env=None, passive_prob: float = 0.6,
                 deck_pool=None):
        self.seed = seed
        self.rng = random.Random(seed)
        self.env = env
        #: 无威胁帧的停手概率（高=更省费、更防守；1.0=纯防守零进攻）
        self.passive_prob = float(passive_prob)
        #: list[deck] → 每局 deck() 从该卡组集合随机抽一副（None=固定默认卡组）
        self.deck_pool = list(deck_pool) if deck_pool else None

    def deck(self):
        """本局卡组：有 deck_pool 抽一副完整卡组（四卡组对手池用），否则固定默认。"""
        if self.deck_pool:
            pick = self.rng.choice(self.deck_pool)
            return list(pick["cards"]) if isinstance(pick, dict) else list(pick)
        return None   # 固定默认卡组（与 ScriptedPolicy 无 pool 时一致）

    def _defend_action(self, player_id: int):
        """script_defender 反制 → (slot, x, y) 或 None。落点必须过引擎合法性。"""
        from simulate_exchange import script_defender
        acts = script_defender(self.env.battle, player_id)
        if not acts:
            return None
        p = self.env.battle.players[player_id]
        for card, pos in acts:
            if not p.can_play_card(card):
                continue
            # 手牌槽位（部署要求卡在手牌前 4）
            if card not in p.cycle[:4]:
                continue
            slot = p.cycle.index(card) + 1
            # script_defender 返回**世界坐标**（喂 sim.deploy_card 的口径）；
            # ActionBundle 是**本地网格**坐标 → 需逆变换（P1 有镜像）。
            wx, wy = float(pos[0]), float(pos[1])
            if player_id == 1:
                gx = int(round(17.0 - wx))
                gy = int(round(31.0 - wy))
            else:
                gx = int(round(wx - 0.5))
                gy = int(round(wy - 0.5))
            return (slot, gx, gy)
        return None

    def _random_action(self, player_id: int):
        mask = self.env.get_action_mask_for(player_id)
        slots = np.flatnonzero(mask["slots"])
        if slots.size == 0:
            return None
        slot = int(self.rng.choice(slots))
        cells = np.flatnonzero(mask["cells"][slot])
        if cells.size == 0:
            return None
        cell = int(self.rng.choice(cells))
        return (slot, int(cell % 18), int(cell // 18))

    def play(self, env, player_id: int = 1) -> ActionBundle:
        self.env = env
        act = None
        if self.rng.random() >= self.passive_prob:
            act = self._random_action(player_id)   # 无威胁帧：低频随机缓出
        if act is None:
            act = self._defend_action(player_id)   # 威胁帧：真防守反制
        if act is None:
            return ActionBundle.noop()
        slot, x, y = act
        return ActionBundle.from_single(slot, int(x), int(y))

    def __call__(self, obs):
        if self.env is None:
            raise RuntimeError("SelfDefenderPolicy 需要先注入 env（.env = ...）")
        return self.play(self.env, 1)
