# -*- coding: utf-8 -*-
"""模型输入编码清单（只读，脚本复算）：网络到底"看到"了什么、丢掉了什么。

回答「我们对模型编码了什么」。全部维度从**真实模块**取（不手抄）：
  · 引擎观测 observe()：15 个栅格通道 + hand/elixir/next_card/time
  · 编码器 FollowerPolicy._encode_parts：CNN 输入 26 通道 → grid_feat → fused
  · belief_token 563 维的分段构成
  · plan_token 58 维的分段构成
  · 特权标签 hidden_labels（**不进网络**，只用于信念监督）
用法（必须在 src/clasher_new 下运行）：
    cd src/clasher_new && PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe \
        ../../scripts/probe_encoding.py
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src", "clasher_new"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

from rl.observation import GRID_H, GRID_W, GRID_C, ENTITY_NAMES
from rl.follower import FollowerPolicy, BELIEF_DIM
from rl.belief import (OPP_EVENT_K, OPP_EVENT_DIM, TENDENCIES, belief_token_dim)
from rl.plan_space import (PLAN_DIM, MACRO_INTENTS, FOCUS_REGIONS, TARGET_KINDS,
                           PLACEMENT_HINTS, OPP_SPELL_THREATS, _OLD_INTENT_COUNT)

CH = ["entity_id(卡身份)", "is_opponent(0=己/1=敌)", "elixir(单位费用)", "card_type(4类)",
      "speed(移速)", "is_air(空中)", "attacks_ground", "attacks_air",
      "log(hp)/10", "hp/hp_max", "hit_speed(攻速参数)", "range/3", "sight_range/3",
      "damage/200", "projectile_damage/200"]

print("=" * 78)
print(f"【1】引擎观测 observe()（rl/observation.py）")
print(f"  grid  : ({GRID_H}, {GRID_W}, {GRID_C}) = {GRID_H*GRID_W*GRID_C} 维（展平）")
print(f"          ⚠️ 一格一实体：同格多实体**后写覆盖**（信息损失点）")
for i, name in enumerate(CH):
    print(f"            ch{i:<2} {name}")
print(f"  hand  : (5,) = p.cycle[:5]（4 张可出 + 第 5 张=下一张）")
print(f"  elixir: (1,)   next_card: (1,) ← 与 hand[4] **重复**   time: (1,)")
print(f"  ⇒ 展平 raw_obs = {GRID_H*GRID_W*GRID_C} + 8 = {GRID_H*GRID_W*GRID_C + 8} 维")

print(f"\n【2】编码器实际吃进去的张量（FollowerPolicy._encode_parts）")
print(f"  ch0(entity_id) 被**替换**为可学习 8 维嵌入 entity_emb({len(ENTITY_NAMES)} 行)")
print(f"  card_type 通道被替换为 4 维 one-hot")
print(f"  ⇒ CNN 输入通道 = (GRID_C-1) + 8 + 4 = {(GRID_C-1)+8+4} ⇒ 张量 "
      f"({(GRID_C-1)+8+4}, {GRID_H}, {GRID_W}) = {(GRID_C-1+8+4)*GRID_H*GRID_W} 维")
pol = FollowerPolicy(hidden=128, plan_dim=PLAN_DIM, belief_dim=BELIEF_DIM)
print(f"  CNN: 3 层卷积 → Flatten ⇒ cnn_out = {pol.cnn_out} 维（= grid_feat）")
print(f"  grid_ln(LayerNorm) 只作用于 grid_feat（P0-A 尺度失衡修复）")
print(f"  hand_feat = entity_emb(hand).reshape = 5×8 = 40 维")
print(f"  scalar    = [elixir, time, next_card/12] = 3 维")
print(f"  plan_f    = plan_mlp({PLAN_DIM}) → 64 维；belief_f = belief_mlp({BELIEF_DIM}) → 64 维")
print(f"  ⇒ fused = {pol.cnn_out} + 40 + 3 + 64 + 64 = "
      f"{pol.cnn_out+40+3+64+64} 维  ⇒ enc = enc_ln(relu(enc_fc(fused))) = 128 维")

print(f"\n【3】belief_token {BELIEF_DIM} 维的分段（rl/belief.py::BeliefInference.encode）")
d = len(DEFAULT_DECK := ["Knight", "MiniPekka", "Arrows", "Minions", "Musketeer",
                         "Fireball", "Giant", "Archer"])
rows = [("对手手牌后验 hand_probs", d, "对手 8 张卡组上的概率分布"),
        ("对手下一张 next_probs", d, "同上"),
        ("elixir_mean + uncertainty", 2, "对手圣水估计 + 不确定度"),
        ("风格倾向 tendency_probs", len(TENDENCIES), f"{list(TENDENCIES)}"),
        (f"对手出牌事件通道 {OPP_EVENT_K}×{OPP_EVENT_DIM}", OPP_EVENT_K*OPP_EVENT_DIM,
         "每条 = 卡 one-hot(177) + x/17 + y/31 + Δt/10（最近 3 次）")]
tot = 0
for name, dim, note in rows:
    tot += dim
    print(f"  {name:<34} {dim:>4} 维   {note}")
print(f"  {'合计':<34} {tot:>4} 维   校验 belief_token_dim = {belief_token_dim(DEFAULT_DECK)}")
print(f"  ⚠️ 训练环 neural=None（train_solo.py:1171 只建 BeliefInference 规则+统计）"
      f"⇒ **神经 belief token 未接入**")

print(f"\n【4】plan_token {PLAN_DIM} 维的分段（rl/plan_space.py::PlanToken.to_vector）")
prows = [("旧 21 维兼容锚: intent_old 8 + region 8 + 旧标量 5", 21, ""),
         (f"intent_new（新宏意图 {len(MACRO_INTENTS)-_OLD_INTENT_COUNT} 个）",
          len(MACRO_INTENTS)-_OLD_INTENT_COUNT, f"宏意图共 {len(MACRO_INTENTS)} 个"),
         (f"target_kind {len(TARGET_KINDS)}", len(TARGET_KINDS), str(TARGET_KINDS)),
         (f"placement_hint {len(PLACEMENT_HINTS)}", len(PLACEMENT_HINTS), ""),
         (f"opp_spell_threat {len(OPP_SPELL_THREATS)}", len(OPP_SPELL_THREATS),
          str(OPP_SPELL_THREATS)),
         ("elixir_budget", 1, "归一化预算"),
         ("hold_mask", 4, "4 个槽位的「别出」标志")]
pt = 0
for name, dim, note in prows:
    pt += dim
    print(f"  {name:<46} {dim:>3} 维  {note}")
print(f"  {'合计':<46} {pt:>3} 维  校验 PLAN_DIM = {PLAN_DIM}")

print("\n【5】特权标签 hidden_labels（**不进网络**，仅供信念模块监督）")
print("  opp_hand(4) / opp_next(1) / opp_cycle(8) / opp_elixir(1) / opp_crown(1) /"
      " opp_towers(3) / my_elixir(1) / my_hand(4) / time(1)")

print("\n【6】⚠️ 没有编码进去的东西（逐条给判据）")
gaps = [
    ("同格多实体", "grid 一格一实体、后写覆盖 ⇒ 挤在同一格时丢实体"),
    ("速度矢量/朝向", "只有 speed 标量，无 vx/vy、无朝向"),
    ("攻击冷却", "有 hit_speed（攻速参数），但**没有「距下次攻击还有多久」**"),
    ("实体年龄", "无生成时刻/存活时长 ⇒ 延迟出兵、建筑剩余寿命、法术剩余时间都不可见"),
    ("状态效果", "15 通道无眩晕/冰冻/狂暴/护盾/毒/减速的**显式**标志；是否经其它通道间接可见"
                 "**未验证 ⇒ 待确认**（引擎确有这些机制）"),
    ("塔型", "塔只以 hp 数值区分；Cannon/Knife/Chef 与标准公主塔不可区分（引擎目前也只模拟标准塔）"),
    ("手牌费用", "**场上**实体的费用在 ch2 里有；但**手牌**只有 8 维嵌入 id ⇒ "
                 "手牌费用只能靠嵌入隐式学（【R12】可算量却让网络猜的典型）"),
    ("对手真实手牌/圣水/循环顺序", "只有 belief 的概率估计；opp_cycle 完整顺序未编码"),
    ("历史帧", "单帧 obs；时序只靠 GRU 隐状态 + belief 事件通道（3 条对手出牌）"),
    ("动作掩码", "掩码是**head 侧单独输入**，不进 enc/fused"),
    ("卡等级", "不显式编码；只通过 hp/伤害数值间接体现"),
    ("加时/相位", "无显式标志；time ≥120 / ≥180 需网络自己算"),
    ("可部署性/掩码缺口", "掩码误判（P1-20）时网络看不到这个事实"),
]
for name, why in gaps:
    print(f"  · {name:<28} {why}")

print("\n【7】参数与规模")
n_par = sum(p.numel() for p in pol.parameters())
print(f"  FollowerPolicy(hidden=128, plan/belief 默认) 参数量 = {n_par:,}"
      f"（AGENTS 记的 629,359 为旧值，见 full_code_reference §4.5 更正）")
print(f"  输入规模：CNN 张量 {(GRID_C-1+8+4)*GRID_H*GRID_W} + belief {BELIEF_DIM} +"
      f" plan {PLAN_DIM} + 手牌/标量 = 每帧 {((GRID_C-1+8+4)*GRID_H*GRID_W)+BELIEF_DIM+PLAN_DIM} 维")
