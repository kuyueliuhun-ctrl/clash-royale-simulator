"""RLEnv：POMDP 训练包装器（规划文档 8.1 / 10.1）。

- step 接收 ActionBundle：同一决策 tick 内整包校验 → 依次 deploy_card → 统一推进决策帧；
- 整包校验通过才提交；非法则拒绝整包并施加惩罚（避免半执行）；
- 只暴露玩家视角观测；特权状态通过 get_hidden_state() / get_prophet_state() 单独提供。

info 契约（P1-5）：``info["opp_played"]`` 为结构化列表
``[{"card": 卡名, "x": 世界x, "y": 世界y}, ...]``（含技能哨兵 ``"__ability__"``，
由信念模块入口过滤），不再是"最后一张卡"字符串。
"""

import os
import sys

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import random
import warnings
from typing import Callable, Optional

import gymnasium as gym
import numpy as np

import battle
import player
from card_utils import Card
from rl.action_bundle import ActionBundle, SubAction, K_MAX, sub_position
from rl.action_mask import (validate_bundle, slot_mask, legal_cells, ability_legal,
                            ability_mana, solo_commit_blocked)
from rl.observation import observe, hidden_labels, GRID_H, GRID_W, GRID_C, ENTITY_NAMES

DEFAULT_DECK = ["Knight", "MiniPekka", "Arrows", "Minions", "Musketeer", "Fireball", "Giant", "Archer"]
DEFAULT_DECK_1 = ["Minions", "Archer", "MiniPekka", "Musketeer", "Giant", "Fireball", "Arrows", "Knight"]

#: 默认奖励权重（与 rl/config.DEFAULT_REWARD 保持一致；勿单独改一处）
#: reward v2（2026 重构）：费差=资源账（手牌圣水+场上部署份额）；价格两段离散
#: （120s 切双倍：tower_dmg_late 塔血贵、elixir_diff_late 费贱）；unit_dmg_k 单位受伤 shaping。
_DEFAULT_REWARD = {
    "crown_weight": 8.0,
    "tower_dmg_opp": 0.001,
    "tower_dmg_self": 0.001,
    "tower_dmg_late": 0.002,     # v2 双倍期塔血系数（t≥120；斩杀/法术砸塔自动变正 EV）
    "win_bonus": 10.0,
    "lose_penalty": 10.0,
    "invalid_penalty": 0.05,
    "elixir_bonus": 0.0,
    "normalize_tower_dmg": True,   # 塔损按塔血%归一化（默认打开=跨等级一致）
    "elixir_diff_weight": 0.5,     # v2 资源账 edw 前段（t<120：费贵，教珍惜圣水）
    "elixir_diff_late": 0.1,       # v2 双倍期 edw（t≥120：费贱，亏费换塔血可接受）
    "unit_dmg_k": 0.0005,          # v2 单位受伤 shaping（客观伤害：敌方单位每掉 1 血 → 我方 +k）
}

#: 塔血参考（真实游戏，lv11）：**国王塔对所有人恒定 4824**；四种公主塔血量各不相同。
#: （lv1 基础值来自 gamedata.json items.spells：Tower Princess 1400 / Cannoneer 1200 /
#:  Dagger Duchess 1270 / Royal Chef 1340；lv11 参考值为用户提供的真实游戏数据。）
#: 注意：引擎目前只模拟标准 Tower Princess（3052/4824，即 PlayerState 硬编码值），
#: 其余塔型仅作奖励归一化的参考表；一旦引擎支持塔型，分母自动按本局真实塔血适配。
TOWER_TROOP_HP_LV11 = {
    "PrincessTower": 3052.0,   # 皇家塔公主
    "DaggerDuchess": 2768.0,   # 飞刀女爵
    "RoyalChef": 2703.0,       # 皇家大厨
    "Cannoneer": 2616.0,       # 炮兵
}
KING_TOWER_HP_LV11 = 4824.0


def tower_total_hp(troop_hp: float, king_hp: float) -> float:
    """一方的总塔血 = 国王塔 + 两座公主塔（塔血归一化的分母；lv11 标准塔 = 10928）。"""
    return 2.0 * troop_hp + king_hp


#: 塔血归一化锚点 = 标准 Tower Princess 的 lv11 总塔血 2×3052 + 4824 = 10928。
#: normalize_tower_dmg 用 ``raw_delta * 锚点 / 本局初始总塔血``，分母取本局**真实**塔血
#: （自动适应不同公主塔型与等级）→ 同一"塔血百分比事件"在任何塔型、任何等级给同一奖励，
#: 且 lv11 标准塔下与旧公式逐位一致（见 selftest.test_reward_economy_level_invariance /
#: test_tower_troop_hp_reference）。
_TOWER_HP_ANCHOR = tower_total_hp(TOWER_TROOP_HP_LV11["PrincessTower"], KING_TOWER_HP_LV11)

#: reward v2 价格分段切换点：引擎圣水回复 120s 切双倍（<120 正常 2.8s/点；>=120 双倍 1.4s/点）。
#: 价格只做**两段离散值**（不线性）：前段费贵塔血便宜（教珍惜圣水），后段费贱塔血贵
#: （双倍期亏费换塔血/法术砸塔自动变正 EV，行为随价格涌现，不手工规定"何时该砸塔"）。
PHASE_SWITCH_S = 120.0


def _phase_weights(rw, battle_time):
    """返回 (tower_coef, edw_coef)：120s 切换的两段离散权重。

    tower_coef 同时作用于打击/自损塔血；edw_coef 作用于资源账 Φ。
    rw 缺 late 键时回退 base（兼容旧 reward 字典）。
    """
    rw = dict(_DEFAULT_REWARD, **(rw or {}))
    late = float(battle_time) >= PHASE_SWITCH_S
    tower = (float(rw.get("tower_dmg_late", rw["tower_dmg_opp"])) if late
             else float(rw["tower_dmg_opp"]))
    edw = (float(rw.get("elixir_diff_late", rw.get("elixir_diff_weight", 0.0))) if late
           else float(rw.get("elixir_diff_weight", 0.0)))
    return tower, edw


def compute_reward(rw, *, blue_hps_old, red_hps_old, blue_hps_new, red_hps_new,
                   blue_left_old, red_left_old, blue_left_new, red_left_new,
                   my_elixir_before, opp_elixir_before, my_elixir_after, opp_elixir_after,
                   my_v_before=0.0, opp_v_before=0.0, my_v_after=0.0, opp_v_after=0.0,
                   winner, invalid_count, blue_hps_max=None, red_hps_max=None,
                   game_over=False, draw_penalty=None):
    """逐决策帧奖励（RLEnv.step 与 selftest 共用）。

    全部新开关关闭（normalize_tower_dmg=False、elixir_diff_weight=0）时与旧公式逐位一致。
    - ``normalize_tower_dmg``：塔损按 本局初始总塔血 归一化到 lv11 锚 → 奖励等级不变，
      修复"等级越高磨血/挨打越值钱、皇冠/胜负不变"的漂移（费差↔塔血校准）；
    - ``elixir_diff_weight``：potential-style shaping，Δ(手牌圣水 + 场上部署费)差 即"资源账"，
      显式给圣水定价，让模型学会"让塔挨打换圣水/费差"这类真实游戏 trade。
      v2 记账式：场上部署费 = 存活部署实体的部署份额（RLEnv 维护，创建即固定、死亡注销，
      不做 HP 折价）。部署帧 E−c 与 V+c 同帧抵消 → 下牌不罚；死亡才注销；法术施放挂账、
      击杀返还对方份额 → 空砸/纯砸塔净亏、解牌赚费差（防双算：spell 产物份额记 0）。
    winner: None=未终局/平局；0=我方胜；其它=负。invalid_count: 非法动作次数。
    game_over: 对局是否已结束（平局判负需要它，避免把进行中的普通步当失败罚）。
    draw_penalty: 平局惩罚（缺省取 rw["draw_penalty"]，再缺省与 lose_penalty 相同）。
    """
    rw = dict(_DEFAULT_REWARD, **(rw or {}))
    if draw_penalty is None:
        draw_penalty = rw.get("draw_penalty", rw["lose_penalty"])
    blue_dmg = blue_hps_old - blue_hps_new
    red_dmg = red_hps_old - red_hps_new
    if rw.get("normalize_tower_dmg"):
        if blue_hps_max:
            blue_dmg = blue_dmg * (_TOWER_HP_ANCHOR / blue_hps_max)
        if red_hps_max:
            red_dmg = red_dmg * (_TOWER_HP_ANCHOR / red_hps_max)
    reward = (
        rw["crown_weight"] * (red_left_old - red_left_new)
        - rw["crown_weight"] * (blue_left_old - blue_left_new)
        + rw["tower_dmg_opp"] * red_dmg
        - rw["tower_dmg_self"] * blue_dmg
        + rw["elixir_bonus"] * my_elixir_after
    )
    edw = rw.get("elixir_diff_weight") or 0.0
    if edw:
        # 资源账（v2 记账式）：potential Φ = (手牌圣水 + 场上部署费) 差；
        # 未传 v_*（纯函数/旧测试）时退化为纯手牌圣水差（V≡0），行为不变。
        me_b = my_elixir_before + float(my_v_before)
        op_b = opp_elixir_before + float(opp_v_before)
        me_a = my_elixir_after + float(my_v_after)
        op_a = opp_elixir_after + float(opp_v_after)
        reward += edw * ((me_a - op_a) - (me_b - op_b))
    if winner == 0:
        reward += rw["win_bonus"]
    elif winner is not None:
        reward -= rw["lose_penalty"]
    elif game_over:
        # 对局结束且无胜者 = 平局 → 按失败惩罚（平局不再免费，"躺平即最优"被消除）
        reward -= float(draw_penalty)
    if invalid_count:
        reward -= rw["invalid_penalty"] * invalid_count
    return reward

_NUM_IDS = len(ENTITY_NAMES) - 1  # 有效卡 id 上界


class ActionBundleSpace(gym.spaces.Space):
    """ActionBundle 的 gym 空间占位（自定义训练用，不参与 SB3 标准优化）。"""

    def __init__(self, k_max: int = K_MAX):
        super().__init__(shape=None, dtype=object)
        self.k_max = k_max

    def sample(self, mask=None, rng=None):
        rng = rng or random
        n = rng.randint(0, self.k_max)
        sub = [SubAction(slot=rng.randint(0, self.k_max), x=rng.randint(0, 17), y=rng.randint(0, 31))
               for _ in range(n)]
        return ActionBundle(sub_actions=sub)

    def contains(self, x):
        return isinstance(x, ActionBundle) and len(x.sub_actions) <= self.k_max

    def __repr__(self):
        return f"ActionBundleSpace(k_max={self.k_max})"


class RLEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        opponent: Optional[Callable] = None,
        deck0: Optional[list] = None,
        deck1: Optional[list] = None,
        deck0_factory: Optional[Callable[[], list]] = None,
        deck1_factory: Optional[Callable[[], list]] = None,
        visualize: bool = False,
        speed: float = 1.0,
        decision_frames: int = 30,
        dt: float = 1 / 60,
        record_hidden: bool = True,
        seed: int = 0,
        reward_weights: Optional[dict] = None,
        card_level: Optional[int] = None,
    ):
        super().__init__()
        if speed <= 0:
            raise ValueError("speed 必须 > 0")
        # 奖惩机制权重（命名配置 rl/config.py 传入；缺省 = 旧公式，行为不变）
        self.reward_weights = dict(_DEFAULT_REWARD)
        if reward_weights:
            self.reward_weights.update(reward_weights)
        self.deck0 = list(deck0) if deck0 else list(DEFAULT_DECK)
        self.deck1 = list(deck1) if deck1 else list(DEFAULT_DECK_1)
        # 每局重采样卡组工厂（卡组完全随机模型用）：reset 时调用，优先于固定 deck
        self.deck0_factory = deck0_factory
        self.deck1_factory = deck1_factory
        self.opponent = opponent  # callable(obs_dict) -> (slot,y,x) | ActionBundle | None=random
        self.visualize = visualize
        self.speed = speed
        self.decision_frames = decision_frames
        self.dt = dt
        self.record_hidden = record_hidden
        self.seed = seed
        self.card_level = card_level  # None=引擎默认（lv11）；11-16 全等级支持

        self.observation_space = gym.spaces.Dict({
            "grid": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(GRID_H, GRID_W, GRID_C), dtype=np.float32),
            "hand": gym.spaces.Box(low=0, high=_NUM_IDS, shape=(5,), dtype=np.int32),
            "elixir": gym.spaces.Box(low=0.0, high=10.0, shape=(1,), dtype=np.float32),
            "next_card": gym.spaces.Box(low=0, high=_NUM_IDS, shape=(1,), dtype=np.int32),
            "time": gym.spaces.Box(low=0.0, high=400.0, shape=(1,), dtype=np.float32),
        })
        self.action_space = ActionBundleSpace()

        self.battle: Optional[battle.BattleState] = None
        self._visualizer = None
        self._rng = random.Random(seed)
        self._mask_fp = None
        self._mask_cells = None
        # —— reward v2 资源账：场上部署份额记账（创建即固定，不做 HP 折价）——
        self._v_share = {0: {}, 1: {}}   # player -> {entity_id: 部署份额}
        self._active_v = [0.0, 0.0]      # 每方当前场上部署价值 Σ份额
        self._seen_max_id = 0            # 实体 id 水位（id 单调递增 → 新实体检测）

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng.seed(seed)
        deck0 = list(self.deck0_factory()) if self.deck0_factory else list(self.deck0)
        deck1 = list(self.deck1_factory()) if self.deck1_factory else list(self.deck1)
        self._rng.shuffle(deck0)
        self._rng.shuffle(deck1)
        # 同步 self.deck0/deck1 到本局实际卡组（工厂重采样时保持下游读取一致）
        self.deck0 = deck0
        self.deck1 = deck1
        self.battle = battle.BattleState(
            player.PlayerState(0, deck0, 5.0),
            player.PlayerState(1, deck1, 5.0),
            card_level=self.card_level,
        )
        # 同步 PlayerState 塔血到真实实体 HP（默认 PlayerState 是硬编码 4824/3052/3052，
        # 与卡牌数据不符；不同步会导致每局第一步出现"塔血暴涨"的假奖励，等级越高越严重）。
        self.battle.update_player_hp()
        # 塔血归一化的本局基准：reset 时三塔全满，记录初始总塔血（economy 机制用）
        self._blue_hps_max = (self.battle.players[0].king_tower_hp
                              + self.battle.players[0].left_tower_hp
                              + self.battle.players[0].right_tower_hp)
        self._red_hps_max = (self.battle.players[1].king_tower_hp
                             + self.battle.players[1].left_tower_hp
                             + self.battle.players[1].right_tower_hp)
        # 清空掩码缓存（P0-3）：新对局的 tick/手牌/建筑都不再匹配旧指纹
        self._mask_fp = None
        self._mask_cells = None
        # 清空 reward v2 资源账（新对局从零记账）
        self._v_share = {0: {}, 1: {}}
        self._active_v = [0.0, 0.0]
        self._seen_max_id = max(self.battle.entities) if self.battle.entities else 0
        if self.visualize:
            from new_visualization import Visualizer
            self._visualizer = Visualizer(self.battle)
        return self.observe(0), {}

    # ---- 观测 ----

    def observe(self, player_id: int = 0) -> dict:
        return observe(self.battle, player_id)

    def get_hidden_state(self) -> dict:
        return hidden_labels(self.battle, 0)

    def get_prophet_state(self) -> dict:
        """特权完整状态摘要（仅先知规划器使用）。"""
        p0, p1 = self.battle.players
        return {
            "time": self.battle.time,
            "my_cycle": list(p0.cycle),
            "opp_cycle": list(p1.cycle),
            "my_elixir": p0.elixir,
            "opp_elixir": p1.elixir,
            "my_towers": [p0.king_tower_hp, p0.left_tower_hp, p0.right_tower_hp],
            "opp_towers": [p1.king_tower_hp, p1.left_tower_hp, p1.right_tower_hp],
            "my_crown": p0.get_crown_count(),
            "opp_crown": p1.get_crown_count(),
            "entities": [
                {"name": e.name, "player": e.player, "pos": (e.position.x, e.position.y), "hp": e.hp}
                for e in self.battle.entities.values() if e.is_alive
            ],
        }

    # ---- 动作掩码 ----

    def get_action_mask(self, partial_bundle: Optional[ActionBundle] = None) -> dict:
        """返回玩家 0（agent）的动态动作掩码。"""
        return self.get_action_mask_for(0, partial_bundle)

    def get_action_mask_for(self, player_id: int, partial_bundle: Optional[ActionBundle] = None) -> dict:
        """返回指定玩家的动态动作掩码，供 autoregressive bundle head 使用。

        指纹包含 player_id 与手牌（P0-3）：同一 tick 下 P0/P1 的手牌与部署规则不同，
        绝不能共用缓存。reset() 显式清缓存。
        """
        assert player_id in (0, 1), f"player_id 必须为 0/1，收到 {player_id}"
        p = self.battle.players[player_id]
        if partial_bundle is None:
            partial_bundle = ActionBundle()
        fp = (
            player_id,
            self.battle.tick,
            tuple(hp for pl in self.battle.players for hp in
                  (pl.king_tower_hp, pl.left_tower_hp, pl.right_tower_hp)),
            tuple(sorted(self.battle.building_positions)),
            tuple(p.cycle),
        )
        if getattr(self, "_mask_fp", None) != fp:
            cells = np.zeros((K_MAX, GRID_H, GRID_W), dtype=bool)
            for i in range(K_MAX):
                cells[i] = legal_cells(self.battle, player_id, p.cycle[i])
            self._mask_cells = cells
            self._mask_fp = fp
        cells = self._mask_cells.copy()

        # 模拟已消耗的圣水/手牌（含技能耗蓝，P1-6）
        used = set()
        elixir = p.elixir
        has_ability = False
        for sa in partial_bundle.sub_actions:
            if sa.kind == "ability":
                cost = ability_mana(self.battle, player_id)
                elixir -= cost if cost is not None else 0.0
                has_ability = True
            elif sa.slot >= 1 and sa.slot <= K_MAX and sa.slot not in used:
                from card_utils import Card
                card = p.cycle[sa.slot - 1]
                cost = Card(card).elixir
                if card == "Mirror" and getattr(p, "last_card", None):
                    cost = Card(p.last_card).elixir + 1
                elixir -= cost
                used.add(sa.slot - 1)
        elixir = max(0.0, elixir)
        slots = slot_mask(p, elixir_override=elixir, used_slots=used)
        for i in used:
            cells[i] = False
        # 8h 不裸下（mask 层）：bundle 还没放牌（首卡决策）时，若禁裸条件成立则整槽禁掉
        # 高承诺单位 → 模型只能 STOP 攒费或先放别的卡凑同刻多卡协同。
        # 防守压境/对手无法出手/己方圣水领先≥3 时会由 solo_commit_blocked 放行。
        if len(partial_bundle.sub_actions) == 0:
            for i in range(K_MAX):
                if slots[i] and solo_commit_blocked(self.battle, player_id, p.cycle[i], elixir):
                    slots[i] = False
                    cells[i] = False
        return {
            "slots": slots,
            "cells": cells,
            "ability_legal": bool(ability_legal(self.battle, player_id,
                                                elixir_override=elixir, already_used=has_ability)),
            "used_slots": np.array(sorted(used), dtype=np.int32),
            "at_cap": len(partial_bundle.sub_actions) >= K_MAX,
            "any_legal": bool(slots.any()) or bool(ability_legal(self.battle, player_id,
                                                                 elixir_override=elixir,
                                                                 already_used=has_ability)),
        }

    # ---- 对手 ----

    def _run_opponent(self) -> list:
        """执行对手动作，返回结构化 played 列表 [{card, x, y}, ...]（P1-5）。"""
        played: list = []
        if self.opponent is None:
            return self._random_opponent()
        obs1 = self.observe(1)
        act = self.opponent(obs1)
        if isinstance(act, ActionBundle):
            bundle = act
        else:
            bundle = legacy_action_to_bundle(act)
        # 先按决策时刻手牌解析卡名/技能，再依次执行（避免循环前移导致槽位错位）
        resolved = []
        for sa in bundle.sub_actions:
            if sa.kind == "ability":
                resolved.append(("__ability__", sa))
            elif sa.slot <= 0 or sa.slot > K_MAX:
                continue
            else:
                card = self.battle.players[1].cycle[sa.slot - 1]
                resolved.append((card, sa))
        for card, sa in resolved:
            if card == "__ability__":
                if self.battle.use_ability(1):
                    played.append({"card": "__ability__", "x": None, "y": None})
                continue
            pos = sa.to_position(player_id=1)  # 本地坐标 → 世界坐标（镜像），P0-4
            elixir_before = self.battle.players[1].elixir
            ok = self.battle.deploy_card(1, card, pos)
            if ok:
                self._deploy_ledger(1, elixir_before, card)
                played.append({"card": card, "x": float(pos.x), "y": float(pos.y)})
        return played

    def _random_opponent(self) -> list:
        """用修复后的掩码采样合法格子（P2：不再六成落禁区）。"""
        mask = self.get_action_mask_for(1)
        slots = np.flatnonzero(mask["slots"])
        if slots.size == 0:
            return []
        slot = int(self._rng.choice(slots))
        cells = np.flatnonzero(mask["cells"][slot])
        if cells.size == 0:
            return []
        cell = int(self._rng.choice(cells))
        x, y = int(cell % GRID_W), int(cell // GRID_W)
        card = self.battle.players[1].cycle[slot]
        pos = sub_position(1, x, y)
        elixir_before = self.battle.players[1].elixir
        ok = self.battle.deploy_card(1, card, pos)
        if ok:
            self._deploy_ledger(1, elixir_before, card)
        return [{"card": card, "x": float(pos.x), "y": float(pos.y)}] if ok else []

    # ---- step ----

    #: 单位受伤 shaping 排除的引擎临时/弹道实体（非部署单位，无受伤语义）
    _NON_UNIT_ENTITY = (battle.Projectile, battle.SpawnProjectile, battle.AreaEffect,
                        battle.GenericBomb, battle.EvoEffectZone, battle.TimedExplosive)

    def _unit_hp_map(self):
        """存活部署单位（非塔、非法术临时实体）hp 快照：{entity_id: (player, hp)}。"""
        hp = {}
        for e in self.battle.entities.values():
            if not e.is_alive or e.id <= 6:
                continue
            if isinstance(e, self._NON_UNIT_ENTITY):
                continue
            hp[e.id] = (e.player, float(e.hp))
        return hp

    def _deploy_ledger(self, pid, elixir_before, card_name):
        """部署记账：把该卡实际花费分摊给本帧新增的部署实体（创建即固定，死亡注销）。

        - 非 spell 卡（或 Mirror 复制单位卡）且有新部署实体 → 每实体 share = cost/n；
        - spell 产物（clone/墓园/GoblinBarrel 等）份额记 0 → 成本已挂 E 账、产物死亡
          不再扣（防双算）；延迟出兵在推进帧出现、无归属 → 同样记 0（免费产物）。
        """
        if Card(card_name).type == "spell" and card_name != "Mirror":
            return  # spell：成本走 E 账，产物免费（防双算）
        cost = elixir_before - self.battle.players[pid].elixir
        if cost <= 1e-9:
            return
        new_ids = [eid for eid in range(self._seen_max_id + 1, self.battle.next_entity_id)
                   if eid in self.battle.entities
                   and isinstance(self.battle.entities[eid], (battle.Troop, battle.Building))]
        if not new_ids:
            return  # 纯法术 / 延迟产出：无本帧部署实体可归属
        share = cost / len(new_ids)
        for eid in new_ids:
            self._v_share[pid][eid] = self._v_share[pid].get(eid, 0.0) + share
        self._active_v[pid] += cost

    def _collect_deaths(self):
        """注销死亡实体的部署份额（推进段死亡在每步末统一清理；份额固定，不做 HP 折价）。"""
        for pid in (0, 1):
            shares = self._v_share[pid]
            dead = [eid for eid, e in shares.items()
                    if eid not in self.battle.entities
                    or not self.battle.entities[eid].is_alive]
            for eid in dead:
                self._active_v[pid] -= shares.pop(eid)

    def step(self, action_bundle: ActionBundle):
        if not isinstance(action_bundle, ActionBundle):
            raise TypeError(f"step 需要 ActionBundle，收到 {type(action_bundle)}")

        p0, p1 = self.battle.players
        blue_hps_old = p0.king_tower_hp + p0.left_tower_hp + p0.right_tower_hp
        red_hps_old = p1.king_tower_hp + p1.left_tower_hp + p1.right_tower_hp
        blue_left_old = 3 - p0.get_crown_count()
        red_left_old = 3 - p1.get_crown_count()
        my_elixir_before = p0.elixir
        opp_elixir_before = p1.elixir
        # reward v2 资源账：本帧起点的 Φ 分量（手牌圣水 + 场上部署份额）
        my_v_before = self._active_v[0]
        opp_v_before = self._active_v[1]
        hp_map_before = self._unit_hp_map()

        # 1) 整包校验（不修改状态）→ 返回按决策时刻手牌解析好的 (card, sub_action)
        ok, reason, resolved = validate_bundle(self.battle, 0, action_bundle)
        invalid_count = 0
        if not ok:
            invalid_count = 1
        else:
            # 2) 整包提交：同一 tick 内依次执行（技能用 use_ability、出牌用 deploy_card，
            #    均按决策时刻解析，避免循环前移错位），期间不推进 battle.step
            for card, sa in resolved:
                if card == "__ability__":
                    if not self.battle.use_ability(0):
                        invalid_count += 1
                    continue
                elixir_before_card = p0.elixir
                succeed = self.battle.deploy_card(0, card, sa.to_position(player_id=0))
                if succeed:
                    # 记账：本卡实际花费（elixir 差自动覆盖 Mirror+1/MergeMaiden 形态费）
                    self._deploy_ledger(0, elixir_before_card, card)
                    self._seen_max_id = max(self._seen_max_id,
                                            self.battle.next_entity_id - 1)
                else:
                    # 引擎级拒绝（掩码误判、同 tick 建筑占位快照滞后等），计为非法
                    # （整包已部分提交，无法回滚，记录之）。P1-20：报警以便发现掩码缺口。
                    invalid_count += 1
                    warnings.warn(
                        f"P1-20: validate 通过但引擎拒绝 {card}@{sa.x},{sa.y} —— 掩码缺口",
                        RuntimeWarning)

        # 3) 对手动作
        opp_played = self._run_opponent()

        # 4) 统一推进决策帧
        frame_steps = max(1, int(round(self.speed)))
        for _ in range(self.decision_frames):
            if self.battle.game_over:
                break
            for _ in range(frame_steps):
                self.battle.step(self.dt)
            if self.visualize and self._visualizer is not None:
                self._visualizer.render_frame()
                import time as _time
                _time.sleep(1 / 60)

        # 推进段产生的死亡统一注销（deploy 段即时入账，死亡段步末统一扣减）
        self._collect_deaths()
        self._seen_max_id = max(self._seen_max_id, self.battle.next_entity_id - 1)

        blue_hps_new = p0.king_tower_hp + p0.left_tower_hp + p0.right_tower_hp
        red_hps_new = p1.king_tower_hp + p1.left_tower_hp + p1.right_tower_hp
        blue_left_new = 3 - p0.get_crown_count()
        red_left_new = 3 - p1.get_crown_count()

        # 5) reward v2：两段离散价格（120s 切双倍）——塔血系数与资源账 edw 同步切换
        tower_coef, edw_coef = _phase_weights(self.reward_weights, self.battle.time)
        rw = dict(self.reward_weights)
        rw["tower_dmg_opp"] = tower_coef
        rw["tower_dmg_self"] = tower_coef
        rw["elixir_diff_weight"] = edw_coef

        reward = compute_reward(
            rw,
            blue_hps_old=blue_hps_old, red_hps_old=red_hps_old,
            blue_hps_new=blue_hps_new, red_hps_new=red_hps_new,
            blue_left_old=blue_left_old, red_left_old=red_left_old,
            blue_left_new=blue_left_new, red_left_new=red_left_new,
            my_elixir_before=my_elixir_before, opp_elixir_before=opp_elixir_before,
            my_elixir_after=p0.elixir, opp_elixir_after=p1.elixir,
            my_v_before=my_v_before, opp_v_before=opp_v_before,
            my_v_after=self._active_v[0], opp_v_after=self._active_v[1],
            winner=self.battle.winner if self.battle.game_over else None,
            invalid_count=invalid_count,
            blue_hps_max=getattr(self, "_blue_hps_max", None),
            red_hps_max=getattr(self, "_red_hps_max", None),
            game_over=self.battle.game_over,
        )

        # 6) 单位受伤 shaping（Phase 3）：每点敌方单位掉血 → 我方 +k（客观伤害事件；
        #    只统计 hp 实际减少的既有实体 → 新 spawn/治疗不误计；死亡帧按掉完计一次）。
        uk = float(rw.get("unit_dmg_k") or 0.0)
        if uk and hp_map_before:
            hurt = [0.0, 0.0]
            for eid, (pid, bhp) in hp_map_before.items():
                e = self.battle.entities.get(eid)
                ahp = float(e.hp) if (e is not None and e.is_alive) else 0.0
                if bhp > ahp:
                    hurt[pid] += bhp - ahp
            reward += uk * (hurt[1] - hurt[0])

        terminated = self.battle.game_over
        info = {
            "bundle_ok": ok,
            "bundle_reason": reason if not ok else "ok",
            "invalid_count": invalid_count,
            "opp_played": opp_played,
            "battle_time": self.battle.time,
            "winner": self.battle.winner,
            "field_v": [float(self._active_v[0]), float(self._active_v[1])],
        }
        if self.record_hidden:
            info["hidden"] = self.get_hidden_state()
        return self.observe(0), reward, terminated, False, info


def legacy_action_to_bundle(act) -> ActionBundle:
    """兼容旧接口 (slot, y, x) / list。"""
    slot, y, x = int(act[0]), int(act[1]), int(act[2])
    if slot == 0:
        return ActionBundle.noop()
    return ActionBundle.from_single(slot, x, y)
