"""计划空间：Prophet / BeliefPlanner 输出的 plan token（规划文档 5.2 / 5.5）。

维度口径（P0-5）：`PLAN_DIM = len(PlanToken().to_vector())` 是**唯一**常量源，
任何加载 follower checkpoint 的入口都必须从 checkpoint 元数据或该常量读取，
禁止魔法数字。

v1 扩展（Phase 2 结构先行）：**全部尾部追加，旧 21 维布局逐位不动**——
- 前 21 维 = 旧布局：intent 8（旧意图 one-hot，索引 0-7）+ region 8 + 旧标量 5；
- 新意图**不占用**旧 intent 位：旧意图帧 → 旧组 one-hot + 新组全 0；
  新意图帧 → 旧组全 0（=无旧意图）+ 尾部新意图组（索引 21-33）one-hot；
- 其后追加 v1 新字段：target_kind(5) / placement_hint(7) / opp_spell_threat(6) /
  elixir_budget(1) / hold_mask(4)。PLAN_DIM = 57。
旧 checkpoint（plan_dim=21）经 ``rl.follower.load_checkpoint`` 前 21 列拷贝、尾部补零加载。

设计文档：docs/rl_plan_design_v1.md（全量意图 17 / pull 距离制胜 / anti_spell / save_ace hold_mask）。
"""

import os
import sys

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np

#: 宏观意图：**前 8 个是旧布局（顺序不可变，兼容旧 ckpt/旧向量语义）**，其后为 v1 新意图。
MACRO_INTENTS = [
    # —— 旧 8 意图（保持索引 0-7）——
    "defend_left", "defend_right", "defend_king",
    "push_left", "push_right", "counterpush",
    "spell_value", "cycle_and_wait",
    # —— v1 追加（索引 8-20）——
    "soft_control",     # 对威胁单位放冰冻/藤蔓
    "spell_trade",      # 法术解后排/关键单位
    "punish",           # 对面沉底/低圣水 → 另一路进攻
    "setup_wait",       # 沉底蓄力（有目的等待/憋 combo）
    "push_commit",      # 坦克推进中 → 部署到能走到坦克后方的区域
    "pre_defend",       # 预判防守（对面即将进攻前占位）
    "spell_finish",     # 后期法术持续磨塔血/压血线
    "king_activate",    # 激活国王塔
    "cycle_small",      # 小费过牌保手牌质量
    "pull",             # 拉扯：改道/横穿/拉远（距离制胜）
    "protect_backline", # 保后排（挤开/吸仇恨/重置目标）
    "anti_spell",       # 防法术：对手有什么/藏什么 → 法术收益最小化
    "save_ace",         # 藏终结卡 + 留费（hold_mask 指名别出的槽）
]
_OLD_INTENT_COUNT = 8  # 前 8 位旧语义（兼容锚）

FOCUS_REGIONS = [
    "own_left", "own_center", "own_right",
    "bridge_left", "bridge_right",
    "enemy_left", "enemy_center", "enemy_right",
]

#: v1：行动目标类型（"对谁行动"）
TARGET_KINDS = [
    "none", "unit", "building", "tower", "my_backline",
]

#: v1：放位策略类型（不是静态标准格，语义见 docs/rl_plan_design_v1.md §2.1）
PLACEMENT_HINTS = [
    "none",
    "pull_across",       # 拉扯：横穿换路（可左→右跨全图）
    "pull_aggro",        # 拉扯：拉到己方输出集中点/转锁建筑
    "support_zone",      # 推进：部署到能走到坦克后方的区域
    "anti_spell_zone",   # 防法术：单位轨迹离开塔溅射区的落点
    "bridge_front",      # 桥头拦截/预判防守位
    "king_front",        # 国王塔仇恨位（激活）
]

#: v1：对手手牌/牌序里的法术威胁（anti_spell 核心信息）
OPP_SPELL_THREATS = [
    "none", "fireball", "poison", "lightning", "freeze", "big_unknown",
]

#: save_ace 固定 ace 名单（设计定稿：固定，不按卡组动态；只列"可能一波终结"的法术，
#: 常规解牌/磨塔法术如 Fireball/Poison/Earthquake 不算 ace——它们该正常用）
ACE_CARDS = ["Lightning", "Vines", "Freeze", "Rocket"]

#: 常量（bundle 组合）语义：0 无 / 1 坦克+后排 / 2 法术+单位 / 3 双路（沿用旧）
COMBO_NONE, COMBO_TANK_SUPPORT, COMBO_SPELL_UNIT, COMBO_SPLIT_PUSH = 0, 1, 2, 3


def _one_hot(name, choices, default_idx=0, dtype=np.float32) -> np.ndarray:
    vec = np.zeros(len(choices), dtype=dtype)
    if name in choices:
        vec[choices.index(name)] = 1.0
    else:
        vec[default_idx] = 1.0
    return vec


@dataclass
class PlanToken:
    macro_intent: str = "cycle_and_wait"
    focus_region: str = "own_center"
    suggested_card: Optional[int] = None       # 槽位 1..4（旧）
    bundle_size_hint: int = 1                  # 旧
    combo_hint: int = 0                        # 旧：0 无 / 1 坦克+后排 / 2 法术+单位 / 3 双路
    risk_profile: float = 0.5                  # 旧：0 保守 .. 1 激进
    value_estimate: float = 0.0                # 旧
    # —— v1 追加字段（全部尾部）——
    target_kind: str = "none"                  # TARGET_KINDS
    placement_hint: str = "none"               # PLACEMENT_HINTS
    opp_spell_threat: str = "none"             # OPP_SPELL_THREATS
    elixir_budget: float = 1.0                 # 0..1：本帧允许投入的圣水比例（1 = 不限）
    hold_mask: int = 0                         # 4 bit：bit(slot-1)=1 → 本帧别出该槽（save_ace）

    # ---- 便捷构造 ----

    @classmethod
    def intent(cls, name, region="own_center", **kw) -> "PlanToken":
        return cls(macro_intent=name, focus_region=region, **kw)

    def hold_slots(self) -> List[int]:
        """hold_mask 命中的槽位列表（1..4）。"""
        return [i + 1 for i in range(4) if (self.hold_mask >> i) & 1]

    def to_vector(self) -> np.ndarray:
        """离散化为固定长度向量，供 follower 拼接。

        布局（兼容锚）：**前 21 维 = 旧布局**（intent 8 + region 8 + 旧标量 5），
        其后全部为 v1 尾部追加。新意图不占用旧 intent 位：
        - macro_intent ∈ 旧 8 → 旧组 one-hot、新组全 0（与旧向量前 21 位完全一致）；
        - macro_intent ∈ 新意图 → 旧组全 0（=无旧意图）、新意图组 one-hot；
        - 未知 → fallback cycle_and_wait（旧组）。
        """
        intent_old = np.zeros(_OLD_INTENT_COUNT, dtype=np.float32)
        intent_new = np.zeros(len(MACRO_INTENTS) - _OLD_INTENT_COUNT, dtype=np.float32)
        if self.macro_intent in MACRO_INTENTS[:_OLD_INTENT_COUNT]:
            intent_old[MACRO_INTENTS.index(self.macro_intent)] = 1.0
        elif self.macro_intent in MACRO_INTENTS:
            intent_new[MACRO_INTENTS.index(self.macro_intent) - _OLD_INTENT_COUNT] = 1.0
        else:
            intent_old[MACRO_INTENTS.index("cycle_and_wait")] = 1.0
        region = _one_hot(self.focus_region, FOCUS_REGIONS,
                          default_idx=FOCUS_REGIONS.index("own_center"))
        old_scalars = np.array([
            float((self.suggested_card or 0)) / 4.0,   # 归一化槽位（旧 P2）
            float(self.bundle_size_hint),
            float(self.combo_hint),
            float(np.clip(self.risk_profile, 0.0, 1.0)),
            float(np.clip(np.tanh(self.value_estimate), -1.0, 1.0)),
        ], dtype=np.float32)
        target = _one_hot(self.target_kind, TARGET_KINDS)
        hint = _one_hot(self.placement_hint, PLACEMENT_HINTS)
        threat = _one_hot(self.opp_spell_threat, OPP_SPELL_THREATS)
        hold = np.array([float((self.hold_mask >> i) & 1) for i in range(4)],
                        dtype=np.float32)
        return np.concatenate([
            intent_old, region, old_scalars,          # 旧 21 维（兼容锚）
            intent_new, target, hint, threat,          # v1 意图组 + 目标 + 位置 + 法术威胁
            np.array([float(np.clip(self.elixir_budget, 0.0, 1.0))], dtype=np.float32),
            hold,
        ])

    @classmethod
    def from_old_layout(cls, old_vec: np.ndarray) -> "PlanToken":
        """把旧 21 维向量解析回 token（新字段取默认）——兼容旧 replay/数据。

        旧布局 = intent 8 + region 8 + 标量 5。
        """
        intent_idx = int(np.argmax(old_vec[0:8])) if old_vec.size >= 8 else 0
        region_idx = int(np.argmax(old_vec[8:16])) if old_vec.size >= 16 else 0
        scal = old_vec[16:21] if old_vec.size >= 21 else np.zeros(5)
        return cls(
            macro_intent=MACRO_INTENTS[min(intent_idx, len(MACRO_INTENTS) - 1)],
            focus_region=FOCUS_REGIONS[min(region_idx, len(FOCUS_REGIONS) - 1)],
            suggested_card=int(round(float(scal[0]) * 4.0)) if scal[0] > 0 else None,
            bundle_size_hint=int(scal[1]),
            combo_hint=int(scal[2]),
            risk_profile=float(scal[3]),
            value_estimate=float(np.arctanh(np.clip(scal[4], -0.999, 0.999))),
        )

    @classmethod
    def zeros(cls) -> "PlanToken":
        return cls()


#: 计划向量唯一维度常量（P0-5）：21 旧维 + v1 追加 = 57
PLAN_DIM = int(len(PlanToken().to_vector()))
_OLD_PLAN_DIM = 21
