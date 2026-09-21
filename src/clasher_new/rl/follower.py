"""跟随者策略 FollowerPolicy（规划文档 6.1 / 6.2 / 6.3.2）。

- 输入：可见观测 + belief_token + plan_token + 动态动作掩码；
- 输出：autoregressive ActionBundle（出牌 slot → 位置 | 英雄技能 ABILITY | STOP）；
- 出牌与开技能可在同一 bundle 内组合（同 tick 提交）；
- 支持 belief/plan 消融（token 置零即可）。
- PPO 重放使用 rollout 时记录的掩码序列与进入步的隐状态，保证 logprob 一致（P0-1）。
- evaluate 额外返回真实熵（对所有 decoder 步求和），修复熵项方向（P0-2）。
- checkpoint 统一携带元数据 {state_dict, plan_dim, belief_dim, hidden_dim}（P0-5）。
"""

import os
import sys

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.action_bundle import ActionBundle, SubAction, K_MAX, INTENT_CANCEL
from rl.observation import GRID_H, GRID_W, GRID_C, ENTITY_NAMES
from rl.plan_space import PLAN_DIM, FOCUS_REGIONS
from rl.belief import belief_token_dim

NUM_ENTITY = len(ENTITY_NAMES)

#: 无神经编码时的默认信念维度（8 卡卡组 → 23）
DEFAULT_OPP_DECK = ["Minions", "Archer", "MiniPekka", "Musketeer",
                    "Giant", "Fireball", "Arrows", "Knight"]
BELIEF_DIM = belief_token_dim(DEFAULT_OPP_DECK)

# bundle head 决策空间：0..K_MAX-1 = 出牌槽位；K_MAX = 英雄技能；K_MAX+1 = STOP
ABILITY_IDX = K_MAX
STOP_IDX = K_MAX + 1
NUM_SLOT_OPTIONS = K_MAX + 2

#: —— 独立 act 头（2026-09-22，C23 下游；预注册 `docs/fl_il_il2_prereg_2026-09-22.md` §7）——
#: 把「出不出牌」从 6 类共享头里拆出来：`act_head`(2 类) **先决**、`slot_head`(`K_MAX+1`=5 类) **后决**。
#: 动机（C23 实测）：零成本组合臂把 `playable_rate` 从 0.1037 抬到 0.8900（**8.6×**）⇒ 动作头面对
#: **训练时从未见过**的输入分布 ⇒ 「门 + 动作头」的离线拼接**原理上无法隔离**「act/play 共用一个
#: 6 类头」这**一个**变量 ⇒ 必须**联合训练**（这才是本改动的存在理由）。
#: 默认关（`decoupled_act=False`）⇒ 架构与旧 ckpt 逐位一致、旧路径**一行未改**。
ACT_STOP_IDX = 0          #: act 头：不出牌（= 旧 STOP_IDX 的语义）
ACT_PLAY_IDX = 1          #: act 头：出牌 / 放技能
NUM_ACT_OPTIONS = 2

#: ★ **GRU 的 `sub_emb` one-hot 空间**（与「头出维」**解耦**）：拆头后 `slot_head` 出维 6→5，
#: 但 `sub_emb` 仍须是 `K_MAX+2=6`（历史编号空间：槽 0..3 + ABILITY 4 + STOP 5）——
#: 否则 `sub_emb`(128,8)→(128,7) 会与 `enc_fc`(128,2731→2730) 一起走
#: `load_checkpoint:152` 的**静默重置**分支（不报错、权重随机）。
#: 付出这个常量，形状破坏面就收缩到只剩 `slot_head.*` + 新增 `act_head.*`。
SUB_SPACE_DIM = NUM_SLOT_OPTIONS

#: 拆头后 `slot_head` 的出维：4 槽 + ABILITY（**无 STOP**）。
#: ⚠️ `ABILITY_IDX = K_MAX = 4` **数值不变** ⇒ 该处索引语义零迁移。
NUM_SLOT_OPTIONS_DECOUPLED = K_MAX + 1

#: —— 攒费意图动作（intent-save；2026-09-19 用户拍板扩参）——
#: 预注册 `docs/intent_save_prereg_2026-09-19.md`。**尾部追加**（0..STOP_IDX 的索引与
#: 语义逐位不变 ⇒ 与仓内「尾部追加 + 尾零兼容」同一纪律）：
#:   INTENT_SAVE_BASE + i (i=0..K_MAX-1) = SAVE(i+1)：为槽 i+1 攒费
#:   CANCEL_IDX                          = 显式撤销 pending
#: 仅在 `FollowerPolicy(intent_options=True)` 下进入动作空间；默认关时
#: `num_slot_options == NUM_SLOT_OPTIONS` ⇒ 架构与旧 ckpt 逐位一致。
INTENT_SAVE_BASE = NUM_SLOT_OPTIONS                    # = 6
CANCEL_IDX = INTENT_SAVE_BASE + K_MAX                  # = 10
INTENT_OPTION_COUNT = K_MAX + 1                        # = 5（SAVE×4 + CANCEL）
NUM_SLOT_OPTIONS_INTENT = NUM_SLOT_OPTIONS + INTENT_OPTION_COUNT   # = 11

#: 意图观测维度：pending 目标 one-hot（none + 4 槽）+ 已等帧数 → 6。
#: 「我在为谁等、等了多久」必须可见，否则等待帧依然匿名、信用分配无从下手。
INTENT_OBS_DIM = K_MAX + 2                             # = 6

#: —— 7h plan 结构软偏置（软生效：只加 logit bias，不硬禁，防 BP 判断错误锁死探索）——
PLAN_CARD_BIAS = 0.8      # plan.suggested_card 槽位 logit 加成
PLAN_HOLD_BIAS = 2.5      # plan.hold_mask 命中槽位 logit 扣减（攒费/藏 ace 用）
PLAN_REGION_BIAS = 0.8    # plan.focus_region 中心附近落点 logit 加成
PLAN_REGION_R = 2         # region 加成半径（本地网格曼哈顿距离）

#: focus_region → 本地网格中心（与 train_bc.REGION_CENTERS 同口径）
_REGION_CENTERS = {
    "own_left": (4, 20), "own_center": (9, 20), "own_right": (14, 20),
    "bridge_left": (4, 16), "bridge_right": (14, 16),
    "enemy_left": (4, 12), "enemy_center": (9, 12), "enemy_right": (14, 12),
}


def save_checkpoint(policy, path):
    """保存带元数据的 checkpoint（P0-5）。"""
    torch.save({
        "state_dict": policy.state_dict(),
        "plan_dim": int(policy.plan_dim),
        "belief_dim": int(policy.belief_dim),
        "hidden_dim": int(policy.hidden_dim),
        "value_bypass": bool(getattr(policy, "value_bypass", False)),
        "value_independent": bool(getattr(policy, "value_independent", False)),
        # 攒费意图（intent-save）：改 `slot_head`/`sub_emb`/`enc_fc` 形状 ⇒ 必须记元数据，
        # 否则"旧 ckpt 加载进 intent 架构"会静默丢 `enc_fc`（见 load_checkpoint 告警）。
        "intent_options": bool(getattr(policy, "intent_options", False)),
        # ★ 独立 act 头（预注册 §7）：同纪律 —— 改 `slot_head` 出维并新增 `act_head`
        "decoupled_act": bool(getattr(policy, "decoupled_act", False)),
    }, path)


def load_checkpoint(path, hidden_dim=None, plan_dim=None, belief_dim=None,
                    value_bypass=None, value_independent=None, intent_options=None,
                    decoupled_act=None):
    """加载 checkpoint；优先读取元数据，旧格式（裸 state_dict）回退到显式/常量维度。

    value_bypass（2026-09-12，实验 B′）/ value_independent（E′）：元数据携带架构标志；
    显式传入（训练侧 cfg）且与元数据不一致时告警——旧 ckpt 加载进新架构 = value
    语义错位，须 --fresh（同 enc_ln/grid_ln 纪律）。

    v1 兼容扩展（Phase 2 结构先行）：旧 checkpoint 的 plan_dim（如 21）< 请求维度（如 57）时，
    plan_mlp 首层权重**前 pd_old 列**拷贝、尾部补零 —— 旧权重对前 21 维语义不变，
    尾部新字段从零开始学（与 PlanToken 尾部追加布局一致）。
    9k（B 层事件通道）：belief_mlp.0.weight 同样处理 —— 旧 belief token（23 维，
    无事件通道）加载后前 23 维语义不变，尾部事件维度从零学。
    词表 v2（2026-09-09）：ENTITY_NAMES 13→177，entity_emb 行数变化 —— 旧行
    （前 13 位序冻结）逐行拷贝、新行从零学；obs 的 hand/next_card 编码同序扩展，
    旧卡 id 语义不变。
    """
    data = torch.load(path, map_location="cpu")
    if isinstance(data, dict) and "state_dict" in data:
        md = data
    else:
        md = {"state_dict": data, "plan_dim": None, "belief_dim": None, "hidden_dim": None}
    sd_src = md["state_dict"]
    pd = int(plan_dim or md.get("plan_dim") or PLAN_DIM)
    bd = int(belief_dim or md.get("belief_dim") or BELIEF_DIM)
    hd = int(hidden_dim or md.get("hidden_dim") or 128)
    vb_meta = bool(md.get("value_bypass", False))
    if value_bypass is not None and bool(value_bypass) != vb_meta:
        from rl.diagnostics import print_safe   # 惰性 import，避免模块级依赖
        print_safe(f"[follower] ⚠️ {path} value_bypass 元数据={vb_meta} 与请求 "
                   f"{bool(value_bypass)} 不一致：value 通路语义错位，续训须 --fresh")
    vb = bool(value_bypass) if value_bypass is not None else vb_meta
    vi_meta = bool(md.get("value_independent", False))
    if value_independent is not None and bool(value_independent) != vi_meta:
        from rl.diagnostics import print_safe
        print_safe(f"[follower] ⚠️ {path} value_independent 元数据={vi_meta} 与请求 "
                   f"{bool(value_independent)} 不一致：价值编码器结构不同，续训须 --fresh")
    vi = bool(value_independent) if value_independent is not None else vi_meta
    # intent_options：显式传入优先；否则回落到 ckpt 元数据（旧 ckpt 无该键 => False）
    _io = bool(md.get("intent_options", False)) if intent_options is None else bool(intent_options)
    #: ★ 独立 act 头（预注册 §7）：同 intent_options 纪律 —— 元数据携带 + 显式不一致告警
    _da = bool(md.get("decoupled_act", False)) if decoupled_act is None else bool(decoupled_act)
    policy = FollowerPolicy(hidden=hd, plan_dim=pd, belief_dim=bd,
                            value_bypass=vb, value_independent=vi, intent_options=_io,
                            decoupled_act=_da)
    target = policy.state_dict()
    for k, v in sd_src.items():
        if k not in target:
            continue
        tv = target[k]
        if tv.shape == v.shape:
            tv.copy_(v)
        elif k == "plan_mlp.0.weight" and v.dim() == 2 and v.shape[1] <= tv.shape[1]:
            # 尾部追加兼容：整行先清零再拷贝前 ckpt_pd 列（尾部必须从零学，
            # 不能残留新网络的随机初始化——否则旧 ckpt 加载即注入噪声）
            tv.zero_()
            tv[:, :v.shape[1]].copy_(v)
        elif k == "belief_mlp.0.weight" and v.dim() == 2 and v.shape[1] <= tv.shape[1]:
            # 9k：事件通道尾部追加（同 plan_mlp 模式）
            tv.zero_()
            tv[:, :v.shape[1]].copy_(v)
        elif k == "entity_emb.weight" and v.dim() == 2 and v.shape[0] <= tv.shape[0] \
                and v.shape[1] == tv.shape[1]:
            # 词表 v2：embedding 行追加（前 13 位序冻结 → 旧行语义不变；新行从零学，
            # 不残留随机初始化——与 plan/belief 尾零兼容同一纪律）
            tv.zero_()
            tv[:v.shape[0], :].copy_(v)
        elif k in ("plan_mlp.0.bias", "belief_mlp.0.bias"):
            tv.copy_(v)
        # 其余形状不匹配（结构大改）→ 保持新初始化，不静默崩
    policy.load_state_dict(target)
    policy.eval()
    # v3 P0-A 护栏（2026-09-11）：`enc_ln` 是 v3 新增键。旧 ckpt 缺该键时，
    # `load_state_dict(target)` 用新网络的 LayerNorm 默认仿射 (weight=1,bias=0) ——
    # 不是 no-op，而是"正常归一化但仿射从未训练"，而 trunk 权重仍是**饱和态**下学的
    # ⇒ 该 ckpt 不能用于续训（v3 §1 要求 `--fresh`），只能当对照基线/对手池。
    # 原先这条路径完全静默（target 由新网络构造，形状恒匹配，不报错）——见 v3 §3.5 注 5。
    # v3 归一化键护栏（2026-09-11）：`enc_ln`（P0-A）与 `grid_ln`（P0-A 备选）都是
    # 新增键。旧 ckpt 缺该键时 `load_state_dict(target)` 用新网络的 LayerNorm 默认
    # 仿射 (weight=1,bias=0) —— 不是 no-op，而是"正常归一化但仿射从未训练"，而 trunk
    # 权重仍是**旧量级**下学的 ⇒ 该 ckpt 不能用于续训（要求 `--fresh`），只能当对照
    # 基线/对手池。原先这条路径完全静默（target 由新网络构造，形状恒匹配，不报错）。
    _missing_ln = [k for k in ("enc_ln", "grid_ln") if not any(
        _k.startswith(k + ".") for _k in sd_src)]
    if _missing_ln:
        from rl.diagnostics import print_safe   # 惰性 import，避免模块级依赖
        print_safe(f"[follower] ⚠️ {path} 缺 {'/'.join(_missing_ln)}.*（旧架构）："
                   "LayerNorm 为默认新初始化，该 ckpt 不可续训（要求 --fresh），"
                   "只能当对照基线/对手池")
    # 攒费意图（intent-save）护栏：新 option 改 `slot_head`(6→11) / `sub_emb`(8→13) /
    # `enc_fc`(输入 +6) 形状。**`enc_fc` 没有"尾部追加"兼容分支**（标量在 fused 中段，
    # 不是尾部）⇒ 一旦架构不一致，`enc_fc` 会被**整体重置**为随机初始化（静默！），
    # 该 ckpt 行为与训练时完全不同 ⇒ 必须显式告警（同 enc_ln/grid_ln 纪律）。
    _ck_intent = bool(md.get("intent_options", False))
    _want_intent = bool(getattr(policy, "intent_options", False))
    if _ck_intent != _want_intent:
        from rl.diagnostics import print_safe   # 惰性 import，避免模块级依赖
        print_safe(f"[follower] ⚠️ {path} 的 intent_options={_ck_intent}，"
                   f"而目标网络 intent_options={_want_intent}：slot_head/sub_emb/enc_fc "
                   "形状不一致 ⇒ enc_fc 被整体重置（**静默**）⇒ 该 ckpt 不可续训，"
                   "要求 --fresh；只能当对照基线/对手池")
    # ★ 独立 act 头（预注册 §7）：`slot_head` 出维 6↔5 且 `act_head` 存在与否 ⇒ 走
    # `load_checkpoint` 的形状不匹配分支（`:152`）⇒ **双双静默随机**。这条告警是唯一防线。
    _ck_dec = bool(md.get("decoupled_act", False))
    _want_dec = bool(getattr(policy, "decoupled_act", False))
    if _ck_dec != _want_dec:
        from rl.diagnostics import print_safe   # 惰性 import，避免模块级依赖
        print_safe(f"[follower] ⚠️ {path} 的 decoupled_act={_ck_dec}，"
                   f"而目标网络 decoupled_act={_want_dec}：slot_head 出维 6↔5 且 act_head "
                   "缺失/多余 ⇒ 形状不匹配 ⇒ **两者双双保持随机初始化（静默！）**"
                   "（load_checkpoint 的形状不匹配分支不报错）⇒ 该 ckpt 不可续训，"
                   "要求 --fresh；只能当对照基线/对手池")
    return policy


class FollowerPolicy(nn.Module):
    def __init__(self, hidden=256, plan_dim=None, belief_dim=None, num_entity=NUM_ENTITY,
                 stop_logit_bias=-1.0, value_bypass=False, value_independent=False,
                 intent_options=False, decoupled_act=False):
        """stop_logit_bias：新初始化时给 STOP logit 的偏置（负数=初始更愿意出牌）。

        intent_options（2026-09-19，用户拍板扩参；预注册
        `docs/intent_save_prereg_2026-09-19.md`）：True 时在 slot 头**尾部追加**
        `SAVE(1..4)` + `CANCEL` 共 5 个 option（`INTENT_SAVE_BASE..CANCEL_IDX`），
        并在观测标量尾部追加 `INTENT_OBS_DIM=6` 维（pending 目标 one-hot + 已等帧数）。
        **架构变更**：`slot_head` 出维 6→11、`sub_emb` 入维 8→13、`enc_fc` 入维 +6
        ⇒ 与旧 ckpt **不兼容**（须 `--fresh`）。默认 False ⇒ `num_slot_options ==
        NUM_SLOT_OPTIONS` 且 `scalar_dim == 3` ⇒ **与旧架构逐位一致、旧 ckpt 可加载**。

        decoupled_act（2026-09-22，C23 下游；预注册 `docs/fl_il_il2_prereg_2026-09-22.md` §7）：
        True 时把「出不出牌」从 6 类共享头里**拆出来** —— `act_head`(2 类: STOP/ACT) 先决，
        `slot_head` 出维降为 `K_MAX+1`=5（4 槽 + ABILITY，**无 STOP**）。
        理由：C23 实测「门 + 独立动作头」的**离线拼接**会把 `playable_rate` 抬 8.6×（状态分布
        巨变）⇒ 动作头面对训练时从未见过的输入分布 ⇒ **无法隔离**「共用一个 6 类头」这一个变量。
        ⇒ 只有**联合训练**有效，故本开关是给 BC 训练用的（`act_parallel`/`evaluate_batch`
        这两个 PPO 批量路径**尚未实现**，开启时会显式 `NotImplementedError`，不静默错）。
        **架构变更** ⇒ 须 `--fresh`；旧 ckpt 加载时 `load_checkpoint` 会告警（两处静默风险）。
        默认 False ⇒ `num_slot_options == NUM_SLOT_OPTIONS` 且 `act_head` 不存在 ⇒ **旧路径零改动**。

        value_bypass（2026-09-12，实验 B′ 落地）：True 时 value 头直连 post-LN enc
        （`value_head(enc)`），**跳过 GRU**（策略头 slot/cell 仍走 GRU 隐状态）。
        依据：`docs/critic_probe_experiment_2026-09-12.md` 实验 B′ —— 同监督配方下
        无 GRU 通路 test EV +0.294 vs 带 GRU +0.217，GRU 净损耗 ~0.08 EV。
        仅影响 value 计算；GRU 隐状态仍由 act/rollout 推进（策略状态依赖不变）。
        旧 ckpt（value_bypass=False 训练）加载进 True 网络 = 语义错位，须 --fresh；
        load_checkpoint 读元数据并在显式覆盖不一致时告警。

        value_independent（2026-09-12，实验 E′）：**独立价值编码器 + 非线性价值头**。
        `value = value_head_mlp(value_enc_ln(relu(value_enc_fc(fused))))`——价值通路有
        自己的融合层/归一化/头，**只吃 value 梯度，不与策略共享参数**（感知前端
        CNN/entity_emb 仍共享）。依据（三条）：
          ① MLP 探针 `post-LN enc → return` R²=+0.42 vs 线性 +0.135
             ⇒ enc 里的价值信息**主要是非线性的**，单个 `nn.Linear` 头有先天上限；
          ② 监督实验（同网络 value-only 目标）可达 EV +0.22~0.29，而 on-policy 联合
             训练 EV≈0（`byp_cprime_20k`）⇒ 差距在"价值吸收"而不是表征；
          ③ bypass（只换接线、仍共享/线性）无效 ⇒ 需要给价值通路自己的容量与梯度。
        **优先级 independent > bypass > shared**（independent 时 value 不走 GRU）。
        架构变更 ⇒ 须 `--fresh`；ckpt 元数据记录该标志。

        纯 RL 冷启动修复：随机初始化下模型天然容易吸附 STOP（手牌合法项少时
        STOP 几乎恒合法）；把 STOP logit 压低让初始 P(出牌)≈0.7-0.8，
        打破“开局双双 STOP → 对局无事件 → 无梯度”的自锁。加载旧 ckpt 会覆盖该偏置。
        """
        if plan_dim is None or belief_dim is None:
            raise ValueError("FollowerPolicy 需要显式 plan_dim/belief_dim（禁止魔法默认值，P0-5）")
        super().__init__()
        self.hidden_dim = hidden
        self.plan_dim = plan_dim
        self.belief_dim = belief_dim
        self.value_bypass = bool(value_bypass)
        self.value_independent = bool(value_independent)

        self.entity_emb = nn.Embedding(num_entity, 8)
        cnn_in = (GRID_C - 1) + 8 + 4  # 14 rest + 8 embed + card_type onehot(4)
        self.cnn = nn.Sequential(
            nn.Conv2d(cnn_in, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, cnn_in, GRID_H, GRID_W)
            cnn_out = self.cnn(dummy).shape[1]
        self.cnn_out = cnn_out

        hand_dim = 5 * 8
        scalar_dim = 3  # elixir + time + next_card（归一化）
        # 攒费意图（intent-save）：仅打开时追加 INTENT_OBS_DIM 维（默认关 = 3，旧架构不变）
        self.intent_options = bool(intent_options)
        self.decoupled_act = bool(decoupled_act)
        if self.decoupled_act and self.intent_options:
            # 预注册 §7.9：两者都动「option 空间尾部」，联合未定义 ⇒ 本批互斥（显式报错）。
            raise ValueError("decoupled_act 与 intent_options 本批互斥（预注册 §7.9）："
                             "SAVE/CANCEL 挂在 6 类空间尾部，与拆头后的 2+5 空间未定义")
        #: GRU 的 one-hot 空间（**与头出维解耦**，见 `SUB_SPACE_DIM` 注释）：
        #: 拆头后仍取 6（或 intent 时的 11）⇒ `sub_emb`/`_sub_vec` 形状不变 ⇒ 旧 ckpt 可加载。
        self.sub_space_dim = (NUM_SLOT_OPTIONS_INTENT if self.intent_options
                              else SUB_SPACE_DIM)
        self.num_slot_options = (NUM_SLOT_OPTIONS_INTENT if self.intent_options
                                 else (NUM_SLOT_OPTIONS_DECOUPLED if self.decoupled_act
                                       else NUM_SLOT_OPTIONS))
        self.scalar_dim = scalar_dim + (INTENT_OBS_DIM if self.intent_options else 0)
        scalar_dim = self.scalar_dim
        self.plan_mlp = nn.Sequential(nn.Linear(plan_dim, 64), nn.ReLU())
        self.belief_mlp = nn.Sequential(nn.Linear(belief_dim, 64), nn.ReLU())
        enc_dim = cnn_out + hand_dim + scalar_dim + 64 + 64
        self.enc_fc = nn.Linear(enc_dim, hidden)
        # v3 P0-A（2026-09-11）：enc 后归一化。旧版 enc 无归一化，L2 范数 ≈533
        # （CNN 输出的常数分量 ≈468、跨帧 std 仅 1.06）→ GRUCell 的 tanh 候选饱和
        # （|n|≈0.994）→ 隐状态 h 跨帧 std ≈2.6e-5（数值恒定）→ value_head 恒输出
        # 常数、EV 恒负，且 slot_head/cell_head 同样失去状态依赖（策略网络基本开环）。
        # 取证见 docs/value_channel_saturation_diagnosis_2026-09-11.md。
        # ⚠️ 架构变更：旧 ckpt（无 enc_ln 键）可加载（权重保持 LN 默认 1/0）但
        # **不可续训**，重训须 --fresh。
        self.enc_ln = nn.LayerNorm(hidden)
        # v3 P0-A 备选（2026-09-11，第二轮）：**CNN 输出独立归一化**。
        # `enc_ln` 只切断了"量级压死 tanh"这条通路（n(abs) 0.994→0.46/0.55 已验证），
        # 但 20k 跑实测暴露出**融合层尺度失衡**（`runs/fix_gru_ln_20k/`，
        # 取证 `docs/rl_training_fix_plan_v3.md` §3.7）：
        #   grid_feat ‖·‖=101.0 跨帧 std=0.426（占 fused 102.99 的 98%，几乎恒定）
        #   vs scalar 17.6/std 6.97、hand 6.4/0.50、plan 1.65/0.199、belief 0.65/0.104
        # ⇒ 用某一维归一化把各分量的"每元素尺度"拉到同一量级，模型才不必先用
        #   若干千步把 grid 权重压下去才能听见 scalar/hand/plan/belief。
        # 归一化**不能凭空创造信息**（grid 的跨帧相对变化 0.4% 不变），它修的是
        # **条件数 + 直流分量主导**——是否够用由下一次 20k 的 `value_std_ratio` 判定。
        # 只上 grid_feat（计划 §1 原文"给 CNN 输出加独立归一化"）：它是被测到的元凶
        # （98% 范数占比）。**不动 scalar**（只有 3 个语义通道 elixir/time/next_card，
        # LN 会强加 sum≈0 的跨通道约束、把不同的物理量耦合起来），也不动
        # plan_f/belief_f（已被各自的 MLP+ReLU 产出，量级小但不是"压死"方）。
        # ⚠️ 架构变更：本次之前的 ckpt（无 `grid_ln.*`）不可续训，重训须 --fresh。
        self.grid_ln = nn.LayerNorm(cnn_out)
        self.gru_cell = nn.GRUCell(hidden, hidden)

        self.slot_head = nn.Linear(hidden, self.num_slot_options)     # 出牌槽位 + ABILITY (+ STOP)
        #: ★ 独立 act 头（仅 decoupled_act=True 时存在）：0=STOP / 1=ACT
        if self.decoupled_act:
            self.act_head = nn.Linear(hidden, NUM_ACT_OPTIONS)
        self.cell_head = nn.Linear(hidden, GRID_H * GRID_W)
        self.value_head = nn.Linear(hidden, 1)
        # E'（2026-09-12）：独立价值编码器 + 非线性头（详见 __init__ docstring 的三条依据）。
        # 只吃 value 梯度：`value_enc_fc` / `value_enc_ln` / `value_head_mlp` 不在策略
        # 前向里出现；感知前端（cnn / entity_emb / plan_mlp / belief_mlp）仍共享，
        # 若要进一步隔离可再拆 CNN（本轮不做，保持单变量 = "价值通路有自己的参数与头"）。
        if self.value_independent:
            self.value_enc_fc = nn.Linear(enc_dim, hidden)
            self.value_enc_ln = nn.LayerNorm(hidden)
            _vm = max(32, hidden // 2)
            self.value_head_mlp = nn.Sequential(
                nn.Linear(hidden, _vm), nn.ReLU(), nn.Linear(_vm, 1))
        self.sub_emb = nn.Linear(self.sub_space_dim + 2, hidden)      # option onehot + (x/18, y/32)

        # 纯 RL 冷启动：初始压低 STOP logit（新随机初始化生效；load_checkpoint 会覆盖）
        if stop_logit_bias:
            with torch.no_grad():
                if self.decoupled_act:
                    self.act_head.bias[ACT_STOP_IDX] += stop_logit_bias
                else:
                    self.slot_head.bias[STOP_IDX] += stop_logit_bias

        self.device = "cpu"
        #: 7h2：plan 软偏置总开关。默认开（player0 侧主流程）。
        #: FollowerOpponent 打 player1 时 BP 仍是 player0 视角 → 置 False，避免
        #: hold_mask/建议卡/区域偏置对 p1 错位（红方冻结根因之一）。
        self.plan_biases_enabled = True

    def to_device(self, device):
        self.device = device
        self.to(device)
        return self

    def _encode_parts(self, obs, belief_token, plan_token):
        """返回 `(fused, enc)`：fused 是融合特征（E′ 独立价值编码器的输入），
        enc 是共享编码器输出（策略 GRU / value 的既有输入）。"""
        grid = torch.as_tensor(obs["grid"], dtype=torch.float32).unsqueeze(0).to(self.device)
        hand = torch.as_tensor(obs["hand"], dtype=torch.long).unsqueeze(0).to(self.device)
        elixir = torch.as_tensor(obs["elixir"], dtype=torch.float32).unsqueeze(0).to(self.device)
        time = torch.as_tensor(obs["time"], dtype=torch.float32).unsqueeze(0).to(self.device)
        next_card = torch.as_tensor(obs["next_card"], dtype=torch.float32).unsqueeze(0).to(self.device) / 12.0

        card_ids = grid[..., 0].long()
        card_vecs = self.entity_emb(card_ids)                       # (1,32,18,8)
        rest = grid[..., 1:]                                        # (1,32,18,14)
        card_type = rest[..., 2].long()                             # 真实卡类型通道（P2 4.4）
        card_type_oh = F.one_hot(card_type, num_classes=4).float()
        x = torch.cat([rest, card_vecs, card_type_oh], dim=-1)      # (1,32,18,C)
        x = x.permute(0, 3, 1, 2)
        grid_feat = self.cnn(x)                                     # (1,cnn_out)
        grid_feat = self.grid_ln(grid_feat)                         # v3 P0-A 备选：分量归一化

        hand_feat = self.entity_emb(hand).reshape(1, -1)            # (1,40)
        _scalar_parts = [elixir, time, next_card]
        if self.intent_options:
            # 攒费意图观测（尾部追加；默认关时不进 enc_fc ⇒ 旧架构/旧 ckpt 不变）
            _scalar_parts.append(self._intent_obs_vec(obs))
        scalar = torch.cat(_scalar_parts, dim=1)                    # (1,self.scalar_dim)

        plan_v = torch.as_tensor(plan_token, dtype=torch.float32).unsqueeze(0).to(self.device)
        belief_v = torch.as_tensor(belief_token, dtype=torch.float32).unsqueeze(0).to(self.device)
        plan_f = self.plan_mlp(plan_v)
        belief_f = self.belief_mlp(belief_v)

        fused = torch.cat([grid_feat, hand_feat, scalar, plan_f, belief_f], dim=1)
        # v3 P0-A：enc 后 LayerNorm（防 GRU tanh 候选饱和，见 __init__ 注释）
        enc = self.enc_ln(torch.relu(self.enc_fc(fused)))            # (1,hidden)
        return fused, enc

    def _encode(self, obs, belief_token, plan_token):
        return self._encode_parts(obs, belief_token, plan_token)[1]

    def _value_from(self, enc, h, fused):
        """按当前 value 架构算 value（**唯一实现**，act/evaluate/diagnostics 共用）。

        优先级：independent（自己的编码器+MLP 头）> bypass（线性头吃 enc）> shared
        （线性头吃 GRU 隐状态 h）。返回 tensor（单条 (1,1) / 批量 (N,1) 皆可）。
        """
        if self.value_independent:
            v = self.value_enc_ln(torch.relu(self.value_enc_fc(fused)))
            return self.value_head_mlp(v)
        if self.value_bypass:
            return self.value_head(enc)
        return self.value_head(h)

    def _slot_mask_tensor(self, mask):
        """把 mask 转成 (self.num_slot_options,) 的合法选项掩码。

        - 已用槽位由 env 在 mask["slots"]/mask["cells"] 中体现（P1-6）；
        - bundle 已达 K_MAX（at_cap）时只放行 STOP（P1-18）；
        - 攒费意图（intent-save）：`mask["intent_slots"]` / `mask["intent_cancel"]`
          **缺省即全非法** ⇒ 默认关（或旧 mask dict）时这 5 个 option 概率恒 0
          ⇒ 与旧行为**逐位相同**（关键：不能沿用 `torch.ones` 的默认"合法"）。
        """
        sm = torch.zeros(self.num_slot_options, device=self.device)
        sm[:K_MAX] = torch.as_tensor(mask["slots"], dtype=torch.float32, device=self.device)
        sm[ABILITY_IDX] = 1.0 if mask.get("ability_legal") else 0.0
        sm[STOP_IDX] = 1.0
        if self.intent_options:
            _is = mask.get("intent_slots")
            if _is is not None:
                sm[INTENT_SAVE_BASE:INTENT_SAVE_BASE + K_MAX] = torch.as_tensor(
                    _is, dtype=torch.float32, device=self.device)
            if mask.get("intent_cancel"):
                sm[CANCEL_IDX] = 1.0
        if mask.get("at_cap"):
            sm[:K_MAX] = 0.0
            sm[ABILITY_IDX] = 0.0
            sm[STOP_IDX] = 1.0
            if self.intent_options:
                sm[INTENT_SAVE_BASE:] = 0.0
        return sm

    def _act_slot_masks(self, mask):
        """★ 拆头版掩码：返回 `(act_mask(2,), slot_mask(num_slot_options,))`（预注册 §7.2）。

        - `slot_mask`：`0..K_MAX-1` 槽 + `ABILITY_IDX`；`at_cap` ⇒ 全 0（**无 STOP 档**）。
        - `act_mask`：`STOP` **恒合法**；`ACT` 合法 ⟺ 任一槽合法 ∨ `ability_legal`。
          ⇒ 「无牌可出」时 ACT 被掩掉 ⇒ 不会出现全 `-1e9` 的 NaN（证伪了一次真实风险）。
        """
        slot = torch.zeros(self.num_slot_options, device=self.device)
        slot[:K_MAX] = torch.as_tensor(mask["slots"], dtype=torch.float32, device=self.device)
        slot[ABILITY_IDX] = 1.0 if mask.get("ability_legal") else 0.0
        if mask.get("at_cap"):
            slot = torch.zeros(self.num_slot_options, device=self.device)
        act = torch.zeros(NUM_ACT_OPTIONS, device=self.device)
        act[ACT_STOP_IDX] = 1.0
        act[ACT_PLAY_IDX] = 1.0 if float(slot.sum()) > 0.0 else 0.0
        return act, slot

    def _plan_biases(self, plan_token):
        """从 plan 向量解析软偏置：(slot_bias(NUM_SLOT_OPTIONS,), cell_bias(H,W))。

        7h：BP 建议只作为 logit 软偏置——建议卡槽 +bias、hold_mask 命中槽 -bias、
        focus_region 中心附近落点 +bias。rollout 与 PPO 重放共用本函数，保证
        采样与 logprob 同分布（不改 env action_mask，不污染 BC/mask 契约）。
        """
        slot_bias = torch.zeros(self.num_slot_options, device=self.device)
        cell_bias = torch.zeros((GRID_H, GRID_W), device=self.device)
        if not getattr(self, "plan_biases_enabled", True):
            # 7h2：player1（FollowerOpponent）不消费 player0 视角 plan 的软偏置
            return slot_bias, cell_bias
        if plan_token is None:
            return slot_bias, cell_bias
        if torch.is_tensor(plan_token):
            arr = plan_token.detach().cpu().numpy()
        else:
            arr = np.asarray(plan_token)
        v = np.asarray(arr, dtype=np.float64).reshape(-1)
        if v.shape[0] < 21:
            return slot_bias, cell_bias
        if float(np.abs(v).sum()) < 1e-9:
            # 全零 = plan 置零（消融 plan-off / plan_dropout）→ 不给任何偏置
            return slot_bias, cell_bias
        # —— 槽位软偏置 ——
        sug = int(round(float(v[16]) * 4.0)) if v[16] > 0.0 else None
        if sug is not None and 1 <= sug <= K_MAX:
            slot_bias[sug - 1] += PLAN_CARD_BIAS
        if v.shape[0] >= PLAN_DIM:
            for i in range(min(4, K_MAX)):
                if v[PLAN_DIM - 4 + i] > 0.5:
                    slot_bias[i] -= PLAN_HOLD_BIAS
        # —— 落点软偏置（focus_region 中心附近）——
        rseg = v[8:16]
        ridx = int(np.argmax(rseg)) if rseg.size and float(rseg.max()) > 0 else \
            FOCUS_REGIONS.index("own_center")
        if 0 <= ridx < len(FOCUS_REGIONS):
            cx, cy = _REGION_CENTERS.get(FOCUS_REGIONS[ridx], (9, 20))
            yy = np.arange(GRID_H, dtype=np.int64)
            xx = np.arange(GRID_W, dtype=np.int64)
            dist = np.abs(yy[:, None] - cy) + np.abs(xx[None, :] - cx)
            cell_bias += torch.as_tensor(
                (dist <= PLAN_REGION_R).astype(np.float32) * PLAN_REGION_BIAS,
                device=self.device)
        return slot_bias, cell_bias

    @staticmethod
    def _mask_or_fallback(masks, j):
        """取第 j 个掩码；缺省用全合法回退（与 evaluate 单条路径一致）。"""
        if j < len(masks):
            return masks[j]
        return {"slots": np.ones(K_MAX, dtype=bool),
                "cells": np.ones((K_MAX, GRID_H, GRID_W), dtype=bool),
                "ability_legal": False}

    def _sub_vec(self, option_idx, x=0.0, y=0.0):
        #: ⚠️ 用 `sub_space_dim`（**不是** `num_slot_options`）：拆头后头出维变了，
        #: 但 GRU 的 one-hot 编号空间必须保持 6 ⇒ `sub_emb` 形状不变、旧 ckpt 可加载。
        sub = torch.zeros(1, self.sub_space_dim + 2, device=self.device)
        sub[0, option_idx] = 1.0
        sub[0, self.sub_space_dim] = x / GRID_W
        sub[0, self.sub_space_dim + 1] = y / GRID_H
        return sub

    def _intent_obs_vec(self, obs):
        """攒费意图观测（INTENT_OBS_DIM=6 维）：pending 目标 one-hot(none+4 槽) + 已等帧数。

        - `obs["intent_slot"]`：当前 pending 的目标槽（1..K_MAX），0 = 无意图；
        - `obs["intent_age"]`：该意图已保持的决策帧数（本帧尚未计）。
        **默认关时返回全零**（且 `scalar_dim` 不含该块 ⇒ 不参与 `enc_fc`）。
        """
        v = torch.zeros(1, INTENT_OBS_DIM, device=self.device)
        if not self.intent_options:
            return v
        try:
            slot = int(np.asarray(obs.get("intent_slot", 0)).reshape(-1)[0])
        except (TypeError, ValueError, IndexError):
            slot = 0
        try:
            age = float(np.asarray(obs.get("intent_age", 0.0)).reshape(-1)[0])
        except (TypeError, ValueError, IndexError):
            age = 0.0
        v[0, slot - 1 if 1 <= slot <= K_MAX else K_MAX] = 1.0   # index K_MAX = "无意图"档
        v[0, K_MAX + 1] = min(max(age, 0.0), 64.0) / 64.0
        return v

    @staticmethod
    def terminal_option(bundle):
        """bundle 的**终止 option**：普通 STOP，或意图动作 SAVE(i)/CANCEL。

        rollout（`act` / `act_parallel`）与 PPO 重放（`evaluate` / 批量重放）**必须
        用同一个函数**，否则终止步的 logprob 会与采样分布不一致。
        """
        it = int(getattr(bundle, "intent", 0) or 0)
        if it == 0:
            return STOP_IDX
        if it <= K_MAX:
            return INTENT_SAVE_BASE + it - 1
        return CANCEL_IDX

    def _sub_update(self, h, option_idx, x=0.0, y=0.0):
        return self.gru_cell(self.sub_emb(self._sub_vec(option_idx, x, y)), h)

    def _encode_batch_parts(self, obs_list, belief_list, plan_list):
        """批量编码：返回 `(fused, enc)`（N 个观测一次前向）。"""
        N = len(obs_list)
        grid = torch.stack([torch.as_tensor(o["grid"], dtype=torch.float32) for o in obs_list]).to(self.device)
        hand = torch.stack([torch.as_tensor(o["hand"], dtype=torch.long) for o in obs_list]).to(self.device)
        elixir = torch.stack([torch.as_tensor(o["elixir"], dtype=torch.float32) for o in obs_list]).to(self.device)
        time_ = torch.stack([torch.as_tensor(o["time"], dtype=torch.float32) for o in obs_list]).to(self.device)
        next_card = torch.stack(
            [torch.as_tensor(o["next_card"], dtype=torch.float32) for o in obs_list]).to(self.device) / 12.0

        card_ids = grid[..., 0].long()
        card_vecs = self.entity_emb(card_ids)                       # (N,32,18,8)
        rest = grid[..., 1:]                                        # (N,32,18,14)
        card_type = rest[..., 2].long()
        card_type_oh = F.one_hot(card_type, num_classes=4).float()
        x = torch.cat([rest, card_vecs, card_type_oh], dim=-1)      # (N,32,18,C)
        x = x.permute(0, 3, 1, 2)
        grid_feat = self.cnn(x)                                     # (N,cnn_out)
        grid_feat = self.grid_ln(grid_feat)                         # v3 P0-A 备选：分量归一化

        hand_feat = self.entity_emb(hand).reshape(N, -1)            # (N,40)
        _scalar_parts = [elixir, time_, next_card]
        if self.intent_options:
            _scalar_parts.append(torch.cat([self._intent_obs_vec(o) for o in obs_list],
                                           dim=0))                  # (N,INTENT_OBS_DIM)
        scalar = torch.cat(_scalar_parts, dim=1)                    # (N,self.scalar_dim)

        plan_v = torch.stack([torch.as_tensor(p, dtype=torch.float32) for p in plan_list]).to(self.device)
        belief_v = torch.stack([torch.as_tensor(b, dtype=torch.float32) for b in belief_list]).to(self.device)
        plan_f = self.plan_mlp(plan_v)
        belief_f = self.belief_mlp(belief_v)

        fused = torch.cat([grid_feat, hand_feat, scalar, plan_f, belief_f], dim=1)
        # v3 P0-A：enc 后 LayerNorm（与 _encode 同口径，保证单条/批量数值一致）
        enc = self.enc_ln(torch.relu(self.enc_fc(fused)))            # (N,hidden)
        return fused, enc

    def _encode_batch(self, obs_list, belief_list, plan_list):
        return self._encode_batch_parts(obs_list, belief_list, plan_list)[1]

    def act(self, obs, belief_token, plan_token, get_mask, hidden=None, deterministic=False):
        """在线动作生成：返回 (ActionBundle, logprob, value, hidden, masks)。

        - masks 为 rollout 过程中每个 decoder 步使用的动作掩码序列，供 PPO 重放；
        - 全程 no_grad、返回的 hidden 已 detach（P1-24），不跨决策步构建计算图。
        """
        if self.decoupled_act:
            # ★ 预注册 §7：拆头版走**独立实现**（旧路径一行未改 ⇒ R2 逐位不变更易证）
            return self._act_decoupled(obs, belief_token, plan_token, get_mask, hidden,
                                       deterministic)
        with torch.no_grad():
            fused, enc = self._encode_parts(obs, belief_token, plan_token)
            if hidden is None:
                hidden = torch.zeros(1, self.hidden_dim, device=self.device)
            h = self.gru_cell(enc, hidden.detach())
            value = float(self._value_from(enc, h, fused).item())

            bundle = ActionBundle()
            logprob = 0.0
            masks = []
            slot_bias, cell_bias = self._plan_biases(plan_token)   # 7h：BP 软结构偏置
            for step in range(K_MAX + 2):
                mask = get_mask(bundle)
                masks.append(mask)
                # 【掩码不变式·1】partial bundle 里**已经用掉的槽位**必须已被掩码置非法。
                # 这是「掩码 ↔ 整包校验」一致性的最小可检形式，且**零成本**（掩码本来就要取）。
                # 先例（2026-09-18）：`env_wrapper.get_action_mask_for` 的 `used` 集合
                # 0/1-based off-by-one，让「(s, s−1)」降序对里的低槽位仍被放行 ⇒
                # 采样器照掩码行事 ⇒ 提交**重复槽位**的包 ⇒ `validate_bundle` **整包拒绝**
                # ⇒ 白掉一帧 + 吃 `invalid_penalty` 罚（实测两臂录像 497 帧，最长连续 225 帧）。
                # ⚠️ 注意：这类错误**只能**由本不变式（或整包校验）抓到，
                # 下面「被掩掉的槽位不得被选中」那条抓不到它（掩码自己错了，采样器没违规）。
                for _sa in bundle.sub_actions:
                    if (_sa.kind == "deploy" and 1 <= _sa.slot <= K_MAX
                            and bool(mask["slots"][_sa.slot - 1])):
                        raise RuntimeError(
                            f"[mask 不变式·1] partial 里的槽位 slot={_sa.slot} 在掩码里仍合法"
                            f"（mask['slots']={np.asarray(mask['slots']).astype(int).tolist()}，"
                            f"used_slots={np.asarray(mask.get('used_slots', [])).tolist()}）"
                            "——掩码与整包校验不一致，按引擎级问题处理")
                slot_mask = self._slot_mask_tensor(mask)
                slot_logits = self.slot_head(h) + slot_bias
                slot_logits = slot_logits.masked_fill(slot_mask == 0, -1e9)
                slot_dist = torch.distributions.Categorical(
                    logits=F.log_softmax(slot_logits, dim=-1))
                if deterministic:
                    option = int(torch.argmax(slot_logits, dim=-1).item())
                else:
                    option = int(slot_dist.sample().item())
                # 【掩码不变式·2】被置 -1e9 的槽位**不可能**被选中（STOP 恒有有限 logit）⇒
                # 一旦发生就是「采样器 ↔ 掩码」不一致（采样器违规）。它**补不上**·1 那一类
                # （掩码自己错时采样器并不违规），两条都要有。
                if option < K_MAX and not bool(mask["slots"][option]):
                    raise RuntimeError(
                        f"[mask 不变式] 采样器选中了被掩掉的槽位 slot={option + 1}"
                        f"（mask['slots']={np.asarray(mask['slots']).astype(int).tolist()}，"
                        f"已用={np.asarray(mask.get('used_slots', [])).tolist()}）"
                        "——掩码与采样不一致，按引擎级问题处理")
                logprob += float(slot_dist.log_prob(torch.tensor([option], device=self.device)).item())

                if option == STOP_IDX:
                    break
                if option >= INTENT_SAVE_BASE:
                    # 攒费意图动作（intent-save）：只设意图、**不落子**，且与 STOP 同形地
                    # **终止本 bundle**（不推 `_sub_update`，与 evaluate 的终止步一致）。
                    # 默认关时 `num_slot_options == INTENT_SAVE_BASE` ⇒ 本分支不可达。
                    bundle.intent = (INTENT_CANCEL if option == CANCEL_IDX
                                     else option - INTENT_SAVE_BASE + 1)
                    break
                if option == ABILITY_IDX:
                    bundle.add_ability()
                    h = self._sub_update(h, ABILITY_IDX)
                    continue

                cells = torch.as_tensor(mask["cells"][option], dtype=torch.float32, device=self.device)
                cell_logits = self.cell_head(h).view(1, GRID_H, GRID_W) + cell_bias
                cell_logits = cell_logits.masked_fill(cells == 0, -1e9)
                flat = cell_logits.reshape(1, -1)
                cell_dist = torch.distributions.Categorical(logits=F.log_softmax(flat, dim=-1))
                if deterministic:
                    cell = int(torch.argmax(flat, dim=-1).item())
                else:
                    cell = int(cell_dist.sample().item())
                logprob += float(cell_dist.log_prob(torch.tensor([cell], device=self.device)).item())
                x, y = int(cell % GRID_W), int(cell // GRID_W)
                bundle.add(option + 1, x, y)
                h = self._sub_update(h, option, x, y)
        return bundle, logprob, value, h.detach(), masks

    def _require_shared_head(self, where):
        """★ 预注册 §7：PPO 批量路径（`act_parallel` / `evaluate_batch`）**本批未改写**。

        宁可**显式报错**，也不许静默用共享头口径算 logprob —— 那会让 PPO 的 `ratio ≠ 1`
        （本仓已有先例：多步帧 `lp_roll` vs `lp_batch` 差 1.2e-07 就已经让 ratio 不精确为 1）。
        """
        if getattr(self, "decoupled_act", False):
            raise NotImplementedError(
                f"{where} 尚未实现 decoupled_act 路径（预注册 §7）：本批只覆盖 BC 训练+留出读数+"
                "部署（`evaluate` / `act`）。PPO 需要这三者与批量路径**同源**，未改写前"
                "不许用它算 logprob。")

    def _act_decoupled(self, obs, belief_token, plan_token, get_mask, hidden, deterministic):
        """★ 独立 act 头版在线动作生成（预注册 §7；**与 `act()` 分离实现** ⇒ 旧路径零改动）。

        两级决策：`act_head` 先判「出不出」，再由 `slot_head` 判「出哪张 / 放技能」。
        返回契约与 `act()` 逐字相同：`(bundle, logprob, value, hidden, masks)`。
        """
        with torch.no_grad():
            fused, enc = self._encode_parts(obs, belief_token, plan_token)
            if hidden is None:
                hidden = torch.zeros(1, self.hidden_dim, device=self.device)
            h = self.gru_cell(enc, hidden.detach())
            value = float(self._value_from(enc, h, fused).item())

            bundle = ActionBundle()
            logprob = 0.0
            masks = []
            slot_bias, cell_bias = self._plan_biases(plan_token)
            for step in range(K_MAX + 2):
                mask = get_mask(bundle)
                masks.append(mask)
                # 【掩码不变式·1】与 `act()` 同款（partial 里已用槽位必须已被掩码置非法）
                for _sa in bundle.sub_actions:
                    if (_sa.kind == "deploy" and 1 <= _sa.slot <= K_MAX
                            and bool(mask["slots"][_sa.slot - 1])):
                        raise RuntimeError(
                            f"[mask 不变式·1] partial 里的槽位 slot={_sa.slot} 在掩码里仍合法"
                            "——掩码与整包校验不一致，按引擎级问题处理")
                act_mask, slot_mask = self._act_slot_masks(mask)
                # —— 第一级：act（出不出）——
                act_logits = self.act_head(h).masked_fill(act_mask == 0, -1e9)
                act_dist = torch.distributions.Categorical(
                    logits=F.log_softmax(act_logits, dim=-1))
                if deterministic:
                    a = int(torch.argmax(act_logits, dim=-1).item())
                else:
                    a = int(act_dist.sample().item())
                logprob += float(act_dist.log_prob(
                    torch.tensor([a], device=self.device)).item())
                if a == ACT_STOP_IDX:
                    break
                # —— 第二级：slot（出哪张 / 放技能）——
                slot_logits = self.slot_head(h) + slot_bias
                slot_logits = slot_logits.masked_fill(slot_mask == 0, -1e9)
                slot_dist = torch.distributions.Categorical(
                    logits=F.log_softmax(slot_logits, dim=-1))
                if deterministic:
                    option = int(torch.argmax(slot_logits, dim=-1).item())
                else:
                    option = int(slot_dist.sample().item())
                # 【掩码不变式·2】被置 -1e9 的槽位不可能被选中
                if option < K_MAX and not bool(mask["slots"][option]):
                    raise RuntimeError(
                        f"[mask 不变式] 采样器选中了被掩掉的槽位 slot={option + 1}"
                        "——掩码与采样不一致，按引擎级问题处理")
                logprob += float(slot_dist.log_prob(
                    torch.tensor([option], device=self.device)).item())
                if option == ABILITY_IDX:
                    bundle.add_ability()
                    h = self._sub_update(h, ABILITY_IDX)
                    continue
                cells = torch.as_tensor(mask["cells"][option], dtype=torch.float32,
                                        device=self.device)
                cell_logits = self.cell_head(h).view(1, GRID_H, GRID_W) + cell_bias
                cell_logits = cell_logits.masked_fill(cells == 0, -1e9)
                flat = cell_logits.reshape(1, -1)
                cell_dist = torch.distributions.Categorical(logits=F.log_softmax(flat, dim=-1))
                if deterministic:
                    cell = int(torch.argmax(flat, dim=-1).item())
                else:
                    cell = int(cell_dist.sample().item())
                logprob += float(cell_dist.log_prob(
                    torch.tensor([cell], device=self.device)).item())
                x, y = int(cell % GRID_W), int(cell // GRID_W)
                bundle.add(option + 1, x, y)
                h = self._sub_update(h, option, x, y)
        return bundle, logprob, value, h.detach(), masks

    def masks_for(self, obs, belief_token, plan_token, bundle, get_mask):
        """为给定 bundle 重建 rollout 时的掩码序列（BC/离线监督用，不采样）。

        与 act() 中 autoregressive 掩码生成完全一致。
        """
        masks = []
        partial = ActionBundle()
        n = len(bundle.sub_actions)
        for i in range(n + 1):
            masks.append(get_mask(partial))
            if i < n:
                sa = bundle.sub_actions[i]
                if sa.kind == "ability":
                    partial.add_ability()
                else:
                    partial.add(sa.slot, sa.x, sa.y)
        return masks

    def act_parallel(self, obs_list, belief_list, plan_list, get_mask_list,
                     hidden_list=None, deterministic=False, get_masks_batch=None):
        """批量 act：N 个 env 一次前向（并行多环境训练用）。

        与 act() 逐位一致：同一输入下对每个 env 产生完全相同的 bundle/logprob/
        value/hidden/masks。返回 (bundles, logprobs, values, hidden_list, masks_list)。
        hidden_list 输入为 list（元素可 None），输出为 detach 后的 list。

        get_masks_batch: 可选，跨进程场景下用于把每个 decoder 步的 N 个掩码请求
        **一次性并发发出**再统一回收（避免 worker 逐个串行计算 legal_cells 拖垮并行度）。
        签名：get_masks_batch(list_of_partials) -> list_of_masks（顺序对应）。
        """
        self._require_shared_head("act_parallel")
        N = len(obs_list)
        with torch.no_grad():
            fused, enc = self._encode_batch_parts(obs_list, belief_list, plan_list)   # (N,hidden)
            if hidden_list is None:
                h = self.gru_cell(enc, torch.zeros(N, self.hidden_dim, device=self.device))
            else:
                h0 = torch.stack([
                    (hid.reshape(-1) if hid is not None
                     else torch.zeros(self.hidden_dim, device=self.device))
                    for hid in hidden_list])
                h = self.gru_cell(enc, h0.detach())
            values = self._value_from(enc, h, fused)[:, 0].tolist()

            bundles = [ActionBundle() for _ in range(N)]
            partials = [ActionBundle() for _ in range(N)]
            logprobs = [0.0] * N
            masks_list = [[] for _ in range(N)]
            # 7h：每个 env 的 plan 软偏置（本帧内不变）
            bias_pairs = [self._plan_biases(plan_list[i]) for i in range(N)]
            for step in range(K_MAX + 2):
                if get_masks_batch is not None:
                    masks_step = get_masks_batch(partials)
                    for i in range(N):
                        masks_list[i].append(masks_step[i])
                else:
                    for i in range(N):
                        masks_list[i].append(get_mask_list[i](partials[i]))
                _cur_masks = [self._mask_or_fallback(masks_list[i], step) for i in range(N)]
                # 【掩码不变式·1（批路径）】同单条 act()：partial 里已用掉的槽位必须已被置非法
                for i in range(N):
                    for _sa in partials[i].sub_actions:
                        if (_sa.kind == "deploy" and 1 <= _sa.slot <= K_MAX
                                and bool(_cur_masks[i]["slots"][_sa.slot - 1])):
                            raise RuntimeError(
                                f"[mask 不变式·1] act_parallel 行{i}：partial 里的槽位 "
                                f"slot={_sa.slot} 在掩码里仍合法——掩码与整包校验不一致")
                slot_masks = torch.stack(
                    [self._slot_mask_tensor(m) for m in _cur_masks])
                slot_logits = self.slot_head(h) + torch.stack(
                    [bias_pairs[i][0] for i in range(N)])
                slot_logits = slot_logits.masked_fill(slot_masks == 0, -1e9)
                slot_dist = torch.distributions.Categorical(
                    logits=F.log_softmax(slot_logits, dim=-1))
                if deterministic:
                    options = torch.argmax(slot_logits, dim=-1)
                else:
                    options = slot_dist.sample()
                # 【掩码不变式】与单条 act() 同一口径：被掩掉的槽位不得被选中。
                _gathered = slot_masks.gather(1, options.view(-1, 1)).view(-1)
                if bool((_gathered == 0).any()):
                    _bad = [int(i) for i in torch.nonzero(_gathered == 0).view(-1).tolist()]
                    raise RuntimeError(
                        f"[mask 不变式] act_parallel 选中了被掩掉的槽位（行 {_bad}）"
                        "——掩码与采样不一致，按引擎级问题处理")
                opts_list = options.tolist()               # 一次批量同步
                lp_all = slot_dist.log_prob(options)       # (N,)
                lp_list = lp_all.tolist()                  # 一次批量同步
                for i in range(N):
                    logprobs[i] += lp_list[i]
                active = []
                for i in range(N):
                    _o = opts_list[i]
                    if _o >= INTENT_SAVE_BASE:
                        # 攒费意图动作：只设意图、不落子、**结束本 bundle**
                        # （默认关时 `num_slot_options == INTENT_SAVE_BASE` ⇒ 不可达）
                        bundles[i].intent = (INTENT_CANCEL if _o == CANCEL_IDX
                                             else _o - INTENT_SAVE_BASE + 1)
                    elif _o != STOP_IDX:
                        active.append(i)
                if not active:
                    break
                # 出牌（deploy）子集：批量 cell head + 批量采样
                dep_envs = [i for i in active if opts_list[i] < K_MAX]
                cell_lp_list = None
                if dep_envs:
                    cell_masks = torch.stack([
                        torch.as_tensor(self._mask_or_fallback(masks_list[i], step)
                                        ["cells"][opts_list[i]],
                                        dtype=torch.float32, device=self.device)
                        for i in dep_envs])
                    hdep = h[dep_envs]
                    cell_logits = self.cell_head(hdep).view(len(dep_envs), GRID_H, GRID_W) \
                        + torch.stack([bias_pairs[i][1] for i in dep_envs])
                    cell_logits = cell_logits.masked_fill(cell_masks == 0, -1e9)
                    flat = cell_logits.reshape(len(dep_envs), -1)
                    cell_dist = torch.distributions.Categorical(
                        logits=F.log_softmax(flat, dim=-1))
                    if deterministic:
                        cells = torch.argmax(flat, dim=-1)
                    else:
                        cells = cell_dist.sample()
                    cells_list = cells.tolist()            # 一次批量同步
                    cell_lp_list = cell_dist.log_prob(cells).tolist()
                    dep_xy = {}
                    for m, i in enumerate(dep_envs):
                        x, y = int(cells_list[m] % GRID_W), int(cells_list[m] // GRID_W)
                        dep_xy[i] = (x, y)
                        logprobs[i] += cell_lp_list[m]
                        bundles[i].add(opts_list[i] + 1, x, y)
                        partials[i].add(opts_list[i] + 1, x, y)
                # 技能（ability）
                for i in active:
                    if opts_list[i] == ABILITY_IDX:
                        bundles[i].add_ability()
                        partials[i].add_ability()
                # 批量 GRU：一个 GRUCell 调用更新所有 active 行
                sub_in = torch.zeros(len(active), self.num_slot_options + 2, device=self.device)
                for m, i in enumerate(active):
                    opt = opts_list[i]
                    sub_in[m, opt] = 1.0
                    if opt < K_MAX:
                        x, y = dep_xy[i]
                        sub_in[m, self.num_slot_options] = x / GRID_W
                        sub_in[m, self.num_slot_options + 1] = y / GRID_H
                h[active] = self.gru_cell(self.sub_emb(sub_in), h[active])
            return (bundles, logprobs, values,
                    [h[i:i + 1].detach() for i in range(N)], masks_list)

    def evaluate_batch(self, obs_list, belief_list, plan_list, bundle_list,
                       masks_list, hidden_list=None):
        """可微批量重放（并行 PPO 更新用）。

        与 evaluate() 逐位一致：同一组 transition 下 logprob / value / entropy 与逐条
        计算完全相等（数值/梯度方向一致），只是把 CNN 编码与每个 decoder 步批量化。
        返回 (logprobs (B,), values (B,1), entropies (B,))。
        """
        self._require_shared_head("evaluate_batch")
        B = len(obs_list)
        fused, enc = self._encode_batch_parts(obs_list, belief_list, plan_list)   # (B,hidden)
        h_rows = []
        for i in range(B):
            base = (hidden_list[i] if hidden_list is not None and hidden_list[i] is not None
                    else torch.zeros(1, self.hidden_dim, device=self.device))
            h_rows.append(self.gru_cell(enc[i:i + 1], base.detach()))
        value = self._value_from(enc, torch.cat(h_rows, dim=0), fused)   # (B,1)

        lengths = [len(b.sub_actions) for b in bundle_list]
        max_len = max(lengths) if lengths else 0
        logprob = [torch.zeros((), device=self.device) for _ in range(B)]
        entropy = [torch.zeros((), device=self.device) for _ in range(B)]
        # 7h：plan 软偏置与 rollout（act/act_parallel）完全一致
        bias_pairs = [self._plan_biases(plan_list[i]) for i in range(B)]

        for j in range(max_len + 1):
            idx = [i for i in range(B) if lengths[i] >= j]
            if not idx:
                break
            hh = torch.cat([h_rows[i] for i in idx], dim=0)
            slot_masks = torch.stack(
                [self._slot_mask_tensor(self._mask_or_fallback(masks_list[i], j))
                 for i in idx])
            slot_logits = self.slot_head(hh) + torch.stack(
                [bias_pairs[i][0] for i in idx])
            slot_logits = slot_logits.masked_fill(slot_masks == 0, -1e9)
            slot_dist = torch.distributions.Categorical(
                logits=F.log_softmax(slot_logits, dim=-1))
            ent = slot_dist.entropy()                       # (|idx|,)
            opts = []
            for i in idx:
                if lengths[i] > j:
                    sa = bundle_list[i].sub_actions[j]
                    opts.append(ABILITY_IDX if sa.kind == "ability" else sa.slot - 1)
                else:
                    # 终止步：普通 STOP，或攒费意图 SAVE(i)/CANCEL（与 rollout 同源）
                    opts.append(self.terminal_option(bundle_list[i]))
            lp_contrib = slot_dist.log_prob(
                torch.tensor(opts, device=self.device))     # 一次批量 op
            for k, i in enumerate(idx):
                entropy[i] = entropy[i] + ent[k]
                logprob[i] = logprob[i] + lp_contrib[k]
            # 出牌格子的 cell head（批量）
            dep = [(k, i) for k, i in enumerate(idx)
                   if lengths[i] > j and bundle_list[i].sub_actions[j].kind == "deploy"]
            if dep:
                ids = [i for _, i in dep]
                hdep = torch.cat([h_rows[i] for i in ids], dim=0)
                cell_masks = torch.stack([
                    torch.as_tensor(self._mask_or_fallback(masks_list[i], j)
                                    ["cells"][bundle_list[i].sub_actions[j].slot - 1],
                                    dtype=torch.float32, device=self.device)
                    for i in ids])
                cell_logits = self.cell_head(hdep).view(len(dep), GRID_H, GRID_W) \
                    + torch.stack([bias_pairs[i][1] for i in ids])
                cell_logits = cell_logits.masked_fill(cell_masks == 0, -1e9)
                flat = cell_logits.reshape(len(dep), -1)
                cell_dist = torch.distributions.Categorical(
                    logits=F.log_softmax(flat, dim=-1))
                cent = cell_dist.entropy()
                cell_idx_list = [
                    bundle_list[i].sub_actions[j].y * GRID_W + bundle_list[i].sub_actions[j].x
                    for i in ids]
                clp = cell_dist.log_prob(
                    torch.tensor(cell_idx_list, device=self.device))   # 一次批量 op
                for m, i in enumerate(ids):
                    entropy[i] = entropy[i] + cent[m]
                    logprob[i] = logprob[i] + clp[m]
            # GRU 更新（批量：一个 GRUCell 处理所有仍活跃的 transition）
            grp = [i for i in idx if lengths[i] > j]
            if grp:
                sub_in = torch.zeros(len(grp), self.num_slot_options + 2, device=self.device)
                for m, i in enumerate(grp):
                    sa = bundle_list[i].sub_actions[j]
                    opt = ABILITY_IDX if sa.kind == "ability" else sa.slot - 1
                    sub_in[m, opt] = 1.0
                    if sa.kind == "deploy":
                        sub_in[m, self.num_slot_options] = sa.x / GRID_W
                        sub_in[m, self.num_slot_options + 1] = sa.y / GRID_H
                hg = torch.cat([h_rows[i] for i in grp], dim=0)
                h_new = self.gru_cell(self.sub_emb(sub_in), hg)
                for m, i in enumerate(grp):
                    h_rows[i] = h_new[m:m + 1]

        return torch.stack(logprob), value, torch.stack(entropy)

    def value(self, obs, belief_token, plan_token, hidden=None) -> float:
        """只算当前状态的 value（不采样动作），用于截断 episode 的 GAE bootstrap（P1-7）。"""
        with torch.no_grad():
            fused, enc = self._encode_parts(obs, belief_token, plan_token)
            if hidden is None:
                hidden = torch.zeros(1, self.hidden_dim, device=self.device)
            h = self.gru_cell(enc, hidden.detach())
            return float(self._value_from(enc, h, fused).item())

    def _evaluate_decoupled(self, obs, belief_token, plan_token, bundle, masks, hidden=None):
        """★ 独立 act 头版可微重放（预注册 §7；**与 `evaluate()` 分离实现** ⇒ 旧路径零改动）。

        损失形态（预注册 §7.2 写死）：
          - **stop 帧**（空 bundle）：`lp = log p₂(STOP)`（1 次抽样）
          - **play 帧**：`lp = log p₂(ACT) + log p₅(槽|ACT) + log p(cell|槽) + log p₂(终止步 STOP)`

        ⚠️ **与共享头口径不可直接比 NLL**：`p₂(STOP)` 是 `p₆(STOP)` 的**边缘** ⇒ stop 帧新 NLL
        **恒 ≤ 旧**（构造性），play 帧多一项 `-log p₂(ACT) ≥ 0` ⇒ **恒更差**。
        可比口径 = **嵌入 6 类分布** `p₆'(i)=p₂(ACT)·p₅(i)`、`p₆'(STOP)=p₂(STOP)`
        （此时 play 帧 `lp` 与解耦和**逐值相等**，证明见预注册 §7.3）⇒ 报告须用该口径算 NLL/top1。
        """
        fused, enc = self._encode_parts(obs, belief_token, plan_token)
        if hidden is None:
            hidden = torch.zeros(1, self.hidden_dim, device=self.device)
        h = self.gru_cell(enc, hidden.detach())
        value = self._value_from(enc, h, fused)

        logprob = 0.0
        entropy = 0.0
        slot_bias, cell_bias = self._plan_biases(plan_token)
        for i, sa in enumerate(bundle.sub_actions):
            mask = masks[i]
            act_mask, slot_mask = self._act_slot_masks(mask)
            # —— 第一级：act（出不出）——本帧人类出牌 ⇒ 标签恒为 ACT
            act_logits = self.act_head(h).masked_fill(act_mask == 0, -1e9)
            act_dist = torch.distributions.Categorical(logits=F.log_softmax(act_logits, dim=-1))
            logprob = logprob + act_dist.log_prob(
                torch.tensor([ACT_PLAY_IDX], device=self.device))
            entropy = entropy + act_dist.entropy()
            # —— 第二级：slot（出哪张 / 放技能）——
            slot_logits = self.slot_head(h) + slot_bias
            slot_logits = slot_logits.masked_fill(slot_mask == 0, -1e9)
            slot_dist = torch.distributions.Categorical(logits=F.log_softmax(slot_logits, dim=-1))
            entropy = entropy + slot_dist.entropy()

            if sa.kind == "ability":
                logprob = logprob + slot_dist.log_prob(
                    torch.tensor([ABILITY_IDX], device=self.device))
                h = self._sub_update(h, ABILITY_IDX)
                continue

            option = sa.slot - 1
            logprob = logprob + slot_dist.log_prob(
                torch.tensor([option], device=self.device))
            cells = torch.as_tensor(mask["cells"][option], dtype=torch.float32, device=self.device)
            cell_logits = self.cell_head(h).view(1, GRID_H, GRID_W) + cell_bias
            cell_logits = cell_logits.masked_fill(cells == 0, -1e9)
            flat = cell_logits.reshape(1, -1)
            cell_dist = torch.distributions.Categorical(logits=F.log_softmax(flat, dim=-1))
            entropy = entropy + cell_dist.entropy()
            cell_idx = sa.y * GRID_W + sa.x
            logprob = logprob + cell_dist.log_prob(
                torch.tensor([cell_idx], device=self.device))
            h = self._sub_update(h, option, sa.x, sa.y)

        # —— 终止步：act 头选 STOP（对应共享头口径里的「终止 STOP 抽样」）——
        mask = masks[len(bundle.sub_actions)] if len(masks) > len(bundle.sub_actions) else {
            "slots": np.ones(K_MAX, dtype=bool), "cells": np.ones((K_MAX, GRID_H, GRID_W), dtype=bool),
            "ability_legal": False,
        }
        act_mask, _slot_mask = self._act_slot_masks(mask)
        act_logits = self.act_head(h).masked_fill(act_mask == 0, -1e9)
        act_dist = torch.distributions.Categorical(logits=F.log_softmax(act_logits, dim=-1))
        entropy = entropy + act_dist.entropy()
        logprob = logprob + act_dist.log_prob(
            torch.tensor([ACT_STOP_IDX], device=self.device))
        return logprob, value, h, entropy

    def evaluate(self, obs, belief_token, plan_token, bundle, masks, hidden=None):
        """可微重放给定 bundle（使用 rollout 时记录的掩码/隐状态）。

        返回 (logprob, value, hidden, entropy)：
        - entropy 为所有 decoder 步分布熵之和（真实熵，非 -lp，P0-2）。
        """
        if self.decoupled_act:
            # ★ 预注册 §7：拆头版走**独立实现**（旧路径一行未改）
            return self._evaluate_decoupled(obs, belief_token, plan_token, bundle, masks, hidden)
        fused, enc = self._encode_parts(obs, belief_token, plan_token)
        if hidden is None:
            hidden = torch.zeros(1, self.hidden_dim, device=self.device)
        h = self.gru_cell(enc, hidden.detach())
        # B'/E'：value 走统一入口（independent 用自己的编码器；bypass 直连 enc；否则 h）
        value = self._value_from(enc, h, fused)

        logprob = 0.0
        entropy = 0.0
        slot_bias, cell_bias = self._plan_biases(plan_token)   # 7h：与 act() 同偏置
        for i, sa in enumerate(bundle.sub_actions):
            mask = masks[i]
            slot_mask = self._slot_mask_tensor(mask)
            slot_logits = self.slot_head(h) + slot_bias
            slot_logits = slot_logits.masked_fill(slot_mask == 0, -1e9)
            slot_dist = torch.distributions.Categorical(logits=F.log_softmax(slot_logits, dim=-1))
            entropy = entropy + slot_dist.entropy()

            if sa.kind == "ability":
                option = ABILITY_IDX
                logprob = logprob + slot_dist.log_prob(torch.tensor([option], device=self.device))
                h = self._sub_update(h, ABILITY_IDX)
                continue

            option = sa.slot - 1
            logprob = logprob + slot_dist.log_prob(torch.tensor([option], device=self.device))
            cells = torch.as_tensor(mask["cells"][option], dtype=torch.float32, device=self.device)
            cell_logits = self.cell_head(h).view(1, GRID_H, GRID_W) + cell_bias
            cell_logits = cell_logits.masked_fill(cells == 0, -1e9)
            flat = cell_logits.reshape(1, -1)
            cell_dist = torch.distributions.Categorical(logits=F.log_softmax(flat, dim=-1))
            entropy = entropy + cell_dist.entropy()
            cell_idx = sa.y * GRID_W + sa.x
            logprob = logprob + cell_dist.log_prob(torch.tensor([cell_idx], device=self.device))
            h = self._sub_update(h, option, sa.x, sa.y)

        # STOP（使用下一个掩码；若已耗尽则全合法）
        mask = masks[len(bundle.sub_actions)] if len(masks) > len(bundle.sub_actions) else {
            "slots": np.ones(K_MAX, dtype=bool), "cells": np.ones((K_MAX, GRID_H, GRID_W), dtype=bool),
            "ability_legal": False,
        }
        slot_mask = self._slot_mask_tensor(mask)
        slot_logits = self.slot_head(h) + slot_bias
        slot_logits = slot_logits.masked_fill(slot_mask == 0, -1e9)
        slot_dist = torch.distributions.Categorical(logits=F.log_softmax(slot_logits, dim=-1))
        entropy = entropy + slot_dist.entropy()
        # 终止 option：普通 STOP，或攒费意图 SAVE(i)/CANCEL（与 `act()` 同源）
        _term = self.terminal_option(bundle)
        logprob = logprob + slot_dist.log_prob(torch.tensor([_term], device=self.device))
        return logprob, value, h, entropy
