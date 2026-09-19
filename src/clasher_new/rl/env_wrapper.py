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
from rl.action_bundle import ActionBundle, SubAction, K_MAX, sub_position, INTENT_CANCEL
from rl.action_mask import (validate_bundle, slot_mask, legal_cells, ability_legal,
                            ability_mana, solo_commit_blocked, _card_cost)
from rl.observation import observe, hidden_labels, GRID_H, GRID_W, GRID_C, ENTITY_NAMES
from rl.engagement import EngagementTradeMonitor   # S2：局面圣水交换（**默认关**）

DEFAULT_DECK = ["Knight", "MiniPekka", "Arrows", "Minions", "Musketeer", "Fireball", "Giant", "Archer"]
DEFAULT_DECK_1 = ["Minions", "Archer", "MiniPekka", "Musketeer", "Giant", "Fireball", "Arrows", "Knight"]

# T2-7（2026-09-19）：**奖励数学**已拆到 `rl/reward.py`（纯搬运，逐字未改；块 = 原 L38-264）。
# 这里**显式重导出** ⇒ 既有调用方（`rl/flow_league.py` / `rl/mcts.py` / `rl/action_mask.py` /
# `scripts/probe_reward_composition.py` / `scripts/phi_offline_check.py` …）**一行都不用改**。
# ⚠️ 下划线名（`_DEFAULT_REWARD` / `_TOWER_HP_ANCHOR` / `_phase_weights` …）**不会**被 `*` 带过来
# ⇒ 必须逐个列出（漏一个就是 import 期 ImportError，不会静默）。
from rl.reward import (  # noqa: F401
    _DEFAULT_REWARD,
    TOWER_TROOP_HP_LV11, KING_TOWER_HP_LV11, tower_total_hp,
    _TOWER_HP_ANCHOR, PHASE_SWITCH_S, DEFAULT_TOWER_PREMIUM_K, DEFAULT_KING_GATE,
    tower_value_mult, tower_premium_k, _princesses_alive, _per_tower_norm_dmg,
    _phase_weights, compute_reward,
)


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
        intent_save: bool = False,
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
        # —— 攒费意图动作（intent-save；2026-09-19 用户拍板扩参）——
        # 预注册 docs/intent_save_prereg_2026-09-19.md。**默认关**：关时本类不读写任何
        # 意图状态、掩码与观测与旧行为**逐位相同**（【R2】）。
        # 开启后：策略可以对一张**买不起**的手牌下"为它攒费"的意图；意图**跨帧保持**
        # （保持不需要逐帧重抽），策略可随时用 SAVE(其它槽)/CANCEL/任意出牌打断。
        self.intent_save = bool(intent_save)
        self._intent_slot = 0        # 当前 pending 目标槽（1..K_MAX）；0 = 无
        self._intent_age = 0         # 已保持的决策帧数
        self._intent_fired_slot = 0  # 刚"攒够"的目标槽（等策略这一帧把它打出去）
        self.intent_stats = {"set": 0, "held": 0, "ready": 0, "fired": 0,
                             "cancelled": 0, "dropped": 0, "fired_held_max": 0,
                             "fired_held": []}

        self.observation_space = gym.spaces.Dict({
            "grid": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(GRID_H, GRID_W, GRID_C), dtype=np.float32),
            "hand": gym.spaces.Box(low=0, high=_NUM_IDS, shape=(5,), dtype=np.int32),
            "elixir": gym.spaces.Box(low=0.0, high=10.0, shape=(1,), dtype=np.float32),
            "next_card": gym.spaces.Box(low=0, high=_NUM_IDS, shape=(1,), dtype=np.int32),
            "time": gym.spaces.Box(low=0.0, high=400.0, shape=(1,), dtype=np.float32),
        })
        if self.intent_save:
            # 攒费意图观测（**只在开启时登记**；关时 dict 与原样逐位一致）
            self.observation_space.spaces["intent_slot"] = gym.spaces.Box(
                low=0, high=K_MAX, shape=(1,), dtype=np.int32)
            self.observation_space.spaces["intent_age"] = gym.spaces.Box(
                low=0.0, high=1e6, shape=(1,), dtype=np.float32)
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
        # —— S2 §6 第 5 项：局面圣水交换奖励项（**默认关**；关时不做任何额外工作）——
        # 规格 docs/engagement_trade_prereg_2026-09-18.md §2/§11.7.1；
        # 在线实现 rl/engagement.py::EngagementTradeMonitor（口径与离线仪器逐条对齐）。
        _rw = self.reward_weights
        self._et_w = float(_rw.get("engagement_trade", 0.0) or 0.0)
        self._et_theta = float(_rw.get("engagement_trade_theta", 1.0))
        self._et_t_ref = float(_rw.get("engagement_trade_t_ref", 2.0))
        self._et_gate = bool(_rw.get("engagement_trade_gate", 1))
        # **measure-only**（2026-09-18 第四轮）：跑监视器、把在线窗口写进 info，
        # 但**一分奖励都不加** ⇒ 行为与不接线时**逐位相同**。用途：在真实训练/评估局上
        # 量"按在线口径切出来的 Trade"，再与结果做配对相关（这是放行 S3 的前置，见预注册 §11.9.4）。
        self._et_measure_only = bool(_rw.get("engagement_trade_measure_only", 0))
        self._et = None                  # 关时恒为 None ⇒ 零开销、逐位回旧
        self._et_scores = []             # 逐决策帧的 score（p0 视角），供取证/对账

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
        # 与每塔满血数组（塔血差异化定价 per-tower 分母）
        self._blue_hps_max = (self.battle.players[0].king_tower_hp
                              + self.battle.players[0].left_tower_hp
                              + self.battle.players[0].right_tower_hp)
        self._red_hps_max = (self.battle.players[1].king_tower_hp
                             + self.battle.players[1].left_tower_hp
                             + self.battle.players[1].right_tower_hp)
        self._blue_towers_max = self._tower_snapshot(self.battle.players[0])
        self._red_towers_max = self._tower_snapshot(self.battle.players[1])
        # 清空掩码缓存（P0-3）：新对局的 tick/手牌/建筑都不再匹配旧指纹
        self._mask_fp = None
        self._mask_cells = None
        # 清空 reward v2 资源账（新对局从零记账）
        self._v_share = {0: {}, 1: {}}
        self._active_v = [0.0, 0.0]
        self._seen_max_id = max(self.battle.entities) if self.battle.entities else 0
        # S2：局边界完全重建监视器（账本/窗口不得跨局残留；【R8】9j 事故类）
        self._et = (EngagementTradeMonitor(theta=self._et_theta, t_ref=self._et_t_ref,
                                           gate=self._et_gate, shares=self._v_share,
                                           tick_seconds=self.dt)
                    if (self._et_w or self._et_measure_only) else None)
        self._et_scores = []
        # 攒费意图：局边界必须完全清空当前 pending（【R8】跨局残留 = 9j 事故类）；
        # `intent_stats` **跨局保留**（它是 run 级取证量，清掉就无法判 J1）。
        self._intent_slot = 0
        self._intent_age = 0
        self._intent_fired_slot = 0
        if self.visualize:
            from new_visualization import Visualizer
            self._visualizer = Visualizer(self.battle)
        return self.observe(0), {}

    # ---- 观测 ----

    def flush_engagement_trade(self):
        """局末（**含按步数截断**）调用：把仍开着的局面结掉，返回尾部明细。

        ⚠️ 两件事必须分清（2026-09-18 第五轮）：
        * 对**奖励**：这部分 score **进不了任何一帧**（奖励已随最后一帧返回）
          ⇒ 若 `engagement_trade > 0`，尾部信用**丢失**。这是**设计取舍**而非 bug：
          "局已结束"时结算一个未走完的局面本身就有争议。
        * 对**测量**：必须结掉，否则录像里的 `et` 会漏掉尾部窗口，
          使测量口径与"结算式"定义不一致（第四轮的 Δρ 就是在这个口径下量的）。
        返回 `{"score","phi","tau","n"}`（p0 视角；无监视器时全 0）。
        """
        if self._et is None:
            return {"score": 0.0, "phi": 0.0, "tau": 0.0, "n": 0}
        self._et.flush(self.battle, self._active_v)
        det = self._et.pop_detail()
        score = self._et.pop_scores()
        return {"score": float(score),
                "phi": float(sum(d["phi_part"] for d in det)),
                "tau": float(sum(d["tau"] for d in det)),
                "n": int(len(det))}

    def observe(self, player_id: int = 0) -> dict:
        d = observe(self.battle, player_id)
        if self.intent_save and player_id == 0:
            # 攒费意图观测（**只在 agent 侧 p0 注入**；p1 由对手策略驱动，无意图状态）。
            # 默认关时不注入 ⇒ 观测 dict 与旧行为逐位一致。
            d["intent_slot"] = np.array([self._intent_slot], dtype=np.int32)
            d["intent_age"] = np.array([float(self._intent_age)], dtype=np.float32)
        return d

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
        # ⚠️ off-by-one 修复（2026-09-18）：`used` 里存的是 **0-based 槽位下标**（`sa.slot - 1`），
        # 成员测试也必须 0-based。原写法 `sa.slot not in used` 拿 **1-based** 的 `sa.slot` 去测
        # 0-based 集合：partial bundle 里一旦出现「(s, s−1)」这种**相邻降序对**，低槽位 `s−1`
        # 会因 `s−1 ∈ used`（那是高槽位存进去的下标）被**整段跳过** ⇒ 既漏记 `used`
        # （掩码放行**重复槽位** ⇒ `validate_bundle` **整包拒绝** ⇒ 白掉一帧 + 吃 `invalid_penalty`），
        # 又漏扣该卡费用（圣水模拟与校验口径不一致）。
        # 实测（修复前，`runs/et_solo100k/replays/`）：A_et **497 帧**（0.7%）提交非法包被拒，
        # 最长**连续 225 帧**同一非法包卡死；取证 `docs/mask_used_slot_offbyone_fix_2026-09-18.md`。
        used = set()
        elixir = p.elixir
        has_ability = False
        for sa in partial_bundle.sub_actions:
            if sa.kind == "ability":
                cost = ability_mana(self.battle, player_id)
                elixir -= cost if cost is not None else 0.0
                has_ability = True
            elif sa.slot >= 1 and sa.slot <= K_MAX and (sa.slot - 1) not in used:
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
        _ability = bool(ability_legal(self.battle, player_id,
                                      elixir_override=elixir, already_used=has_ability))
        out = {
            "slots": slots,
            "cells": cells,
            "ability_legal": _ability,
            "used_slots": np.array(sorted(used), dtype=np.int32),
            "at_cap": len(partial_bundle.sub_actions) >= K_MAX,
            "any_legal": bool(slots.any()) or _ability,
        }
        # —— 攒费意图（intent-save，默认关）：只在 bundle 首卡决策、且只在 agent 侧 p0 ——
        if self.intent_save and player_id == 0 and len(partial_bundle.sub_actions) == 0:
            _is = [False] * K_MAX
            for i in range(K_MAX):
                if i in used:
                    continue
                _c = _card_cost(p, p.cycle[i])
                # 只有**买不起**才够格下攒费意图：买得起就该直接打（这条把"意图"与"出牌"
                # 的语义分干净，也让 SAVE 永远不会成为"赖着不出牌"的后门）。
                if _c is not None and p.king_tower_hp > 0 and elixir < _c:
                    _is[i] = True
            _cancel = self._intent_slot != 0
            _hold = self._intent_holding(p, elixir)
            if _hold:
                # 【承诺期】已为某张牌攒费且仍买不起 ⇒ 本帧压制一切花费（含技能与落点），
                # 合法集只剩 {保持=STOP, CANCEL, 改攒其它槽}。★ 这就是"选择这项 =
                # 等下一个或多个决策帧"的机械实现：**保持不需要逐帧重抽**，
                # 否则 k 帧合取 p^k 会把轨迹压到采样不到（预注册 §1/§5 R-1）。
                slots = np.zeros(K_MAX, dtype=bool)
                cells = np.zeros((K_MAX, GRID_H, GRID_W), dtype=bool)
                _ability = False
                out["slots"], out["cells"], out["ability_legal"] = slots, cells, False
                out["any_legal"] = True         # 保持/取消恒合法
            out["intent_slots"] = _is
            out["intent_cancel"] = _cancel
            out["intent_hold"] = _hold
        return out

    # ---- 对手 ----

    def _intent_holding(self, p, elixir) -> bool:
        """pending 目标**仍买不起** ⇒ True（=「承诺期」，本帧压制一切花费）。

        目标无效（离手 / 王塔亡 / 费用不可算）返回 False —— 清理由 `_apply_intent` 负责，
        本函数**必须无副作用**（它在掩码热路径里被调用）。
        """
        if not self.intent_save or not self._intent_slot:
            return False
        i = int(self._intent_slot)
        if not (1 <= i <= K_MAX) or p.king_tower_hp <= 0 or i - 1 >= len(p.cycle):
            return False
        c = _card_cost(p, p.cycle[i - 1])
        if c is None:
            return False
        return float(elixir) < float(c)

    def _apply_intent(self, intent: int, p0, bundle: ActionBundle) -> dict:
        """攒费意图状态机（**只在 `intent_save=True` 时被调用**）。

        语义（预注册 `docs/intent_save_prereg_2026-09-19.md` §2）：
        - `intent == 0` 且本帧**没出牌**（纯 STOP）⇒ 意图**原样保留**
          —— ★ 这是全部机制价值所在：**保持不需要逐帧重抽**。
        - `intent == 0` 且本帧出了牌 ⇒ 意图终止（花掉的钱与"为某张牌攒"不相容）。
        - `1..K_MAX` ⇒ 设置/替换 pending 目标（`set`）。
        - `INTENT_CANCEL` ⇒ 显式撤销（`cancelled`）—— 用户要的"中途改变想法"。
        - 目标攒够 ⇒ 标记 `ready`，等策略把这张牌打出去（`fired` 记录**当时已保持帧数**）。
        - 目标失效 ⇒ 静默清（`dropped`，**不得**产生任何非法动作）。
        """
        st = self.intent_stats
        ev: dict = {}
        if bundle.sub_actions:
            if self._intent_fired_slot:
                _deploy_slots = [sa.slot for sa in bundle.sub_actions if sa.kind == "deploy"]
                if self._intent_fired_slot in _deploy_slots:
                    st["fired"] += 1
                    st["fired_held"] = (st["fired_held"] + [int(self._intent_age)])[-256:]
                    st["fired_held_max"] = max(st["fired_held_max"], int(self._intent_age))
                    ev["intent_fired"] = 1
                    ev["intent_fired_held"] = int(self._intent_age)
                else:
                    st["dropped"] += 1
                    ev["intent_dropped"] = 1
                self._intent_fired_slot = 0
            elif self._intent_slot:
                st["dropped"] += 1
                ev["intent_dropped"] = 1
            self._intent_slot = 0
            self._intent_age = 0
            self._intent_fired_slot = 0
            return ev
        if intent == INTENT_CANCEL:
            if self._intent_slot:
                st["cancelled"] += 1
                ev["intent_cancelled"] = 1
            self._intent_slot = 0
            self._intent_age = 0
            self._intent_fired_slot = 0
            return ev
        if 1 <= intent <= K_MAX:
            self._intent_slot = int(intent)
            self._intent_age = 0
            self._intent_fired_slot = 0
            st["set"] += 1
            ev["intent_set"] = 1
            return ev
        # intent == 0 且无子动作 = 保持 / 纯 STOP：意图原样保留
        if not self._intent_slot:
            return ev
        i = int(self._intent_slot)
        _valid = (1 <= i <= K_MAX and p0.king_tower_hp > 0 and i - 1 < len(p0.cycle)
                  and _card_cost(p0, p0.cycle[i - 1]) is not None)
        if not _valid:
            self._intent_slot = 0
            self._intent_age = 0
            self._intent_fired_slot = 0
            st["dropped"] += 1
            ev["intent_dropped"] = 1
            return ev
        if self._intent_holding(p0, p0.elixir):
            self._intent_age += 1
            st["held"] += 1
            ev["intent_held"] = 1
        elif not self._intent_fired_slot:
            # 已攒够：标记 ready，等策略把这张牌打出去（下一帧起它就是普通可出牌）
            self._intent_fired_slot = i
            st["ready"] += 1
            ev["intent_ready"] = 1
        return ev

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

    @staticmethod
    def _tower_snapshot(p) -> list:
        """三塔血量快照 [king, left, right]（塔血差异化定价的 per-tower 输入）。"""
        return [p.king_tower_hp, p.left_tower_hp, p.right_tower_hp]

    def step(self, action_bundle: ActionBundle):
        if not isinstance(action_bundle, ActionBundle):
            raise TypeError(f"step 需要 ActionBundle，收到 {type(action_bundle)}")

        p0, p1 = self.battle.players
        # —— 攒费意图状态机（**默认关时整段跳过 ⇒ 与旧行为逐位相同**）——
        # 位置在"提交之前、决策时刻的圣水"：与掩码 `get_action_mask_for` 同一时刻取值，
        # 保证「掩码说买不起 ⇒ 记为保持」与「掩码说买得起 ⇒ 记为 ready」永不打架。
        _intent_evt = (self._apply_intent(int(getattr(action_bundle, "intent", 0) or 0),
                                          p0, action_bundle)
                       if self.intent_save else {})
        blue_hps_old = p0.king_tower_hp + p0.left_tower_hp + p0.right_tower_hp
        red_hps_old = p1.king_tower_hp + p1.left_tower_hp + p1.right_tower_hp
        blue_towers_old = self._tower_snapshot(p0)
        red_towers_old = self._tower_snapshot(p1)
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
                if self._et is not None:            # S2：只在开关打开时发生
                    self._et.tick(self.battle, self._active_v)
            if self.visualize and self._visualizer is not None:
                self._visualizer.render_frame()
                import time as _time
                _time.sleep(1 / 60)

        # 推进段产生的死亡统一注销（deploy 段即时入账，死亡段步末统一扣减）
        self._collect_deaths()
        self._seen_max_id = max(self._seen_max_id, self.battle.next_entity_id - 1)

        blue_hps_new = p0.king_tower_hp + p0.left_tower_hp + p0.right_tower_hp
        red_hps_new = p1.king_tower_hp + p1.left_tower_hp + p1.right_tower_hp
        blue_towers_new = self._tower_snapshot(p0)
        red_towers_new = self._tower_snapshot(p1)
        blue_left_new = 3 - p0.get_crown_count()
        red_left_new = 3 - p1.get_crown_count()

        # 5) reward v2：两段离散价格（120s 切双倍）——塔血系数与资源账 edw 同步切换
        #    （塔血打击/自损不对称：crown_lose 与 tower_dmg_self 均高于进攻侧）
        tower_opp, tower_self, edw_coef = _phase_weights(self.reward_weights, self.battle.time)
        rw = dict(self.reward_weights)
        rw["tower_dmg_opp"] = tower_opp
        rw["tower_dmg_self"] = tower_self
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
            blue_towers_old=blue_towers_old, red_towers_old=red_towers_old,
            blue_towers_new=blue_towers_new, red_towers_new=red_towers_new,
            blue_towers_max=getattr(self, "_blue_towers_max", None),
            red_towers_max=getattr(self, "_red_towers_max", None),
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

        # —— S2 §6 第 5 项：局面圣水交换（默认关 ⇒ 一个浮点运算都不做）——
        et_score = 0.0
        et_phi = et_tau = et_tau0 = et_tau1 = 0.0
        et_n = 0
        if self._et is not None:
            if self.battle.game_over:
                self._et.flush(self.battle, self._active_v)   # 局末把开着的窗口结掉
            et_score = self._et.pop_scores()
            _det = self._et.pop_detail()
            et_n = len(_det)
            et_phi = sum(d["phi_part"] for d in _det)
            et_tau = sum(d["tau"] for d in _det)
            et_tau0 = sum(d["tau0"] for d in _det)
            et_tau1 = sum(d["tau1"] for d in _det)
            self._et_scores.append(float(et_score))
            if not self._et_measure_only:
                reward += self._et_w * et_score      # ← 只有这里会动奖励
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
        if self.intent_save:
            # 攒费意图取证字段（**只读不参与奖励**）：本帧事件 + 状态 + 累计计数。
            # 判据 J1（预注册 §4）用的"意图保持 ≥34 帧后真打出"从 `fired_held` 读。
            info["intent"] = {
                "slot": int(self._intent_slot),
                "age": int(self._intent_age),
                "hold": bool(self._intent_holding(p0, p0.elixir)),
                "event": dict(_intent_evt),
                "stats": dict(self.intent_stats, fired_held=list(self.intent_stats["fired_held"])),
            }
        if self._et is not None:
            # 取证字段（不影响任何行为）：本决策帧结算掉的局面明细 + 累计
            info["engagement_trade"] = float(et_score)
            info["engagement_trade_cum"] = float(sum(self._et_scores))
            info["engagement_trade_n"] = int(self._et.n_settled)
            # 逐帧明细：phi 部分 / 对称化后的 τ / 本帧 score / 本帧结算的窗口数 /
            # 以及 τ 的**两侧原始值**（供离线核对称性与做 p1 镜像的符号还原）
            info["engagement_trade_detail"] = [float(et_phi), float(et_tau),
                                               float(et_score), int(et_n),
                                               float(et_tau0), float(et_tau1)]
            info["engagement_trade_measure_only"] = bool(self._et_measure_only)
        if self.record_hidden:
            info["hidden"] = self.get_hidden_state()
        return self.observe(0), reward, terminated, False, info


def legacy_action_to_bundle(act) -> ActionBundle:
    """兼容旧接口 (slot, y, x) / list。"""
    slot, y, x = int(act[0]), int(act[1]), int(act[2])
    if slot == 0:
        return ActionBundle.noop()
    return ActionBundle.from_single(slot, x, y)
