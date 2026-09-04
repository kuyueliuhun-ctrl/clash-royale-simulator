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

from rl.action_bundle import ActionBundle, SubAction, K_MAX
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
    }, path)


def load_checkpoint(path, hidden_dim=None, plan_dim=None, belief_dim=None):
    """加载 checkpoint；优先读取元数据，旧格式（裸 state_dict）回退到显式/常量维度。

    v1 兼容扩展（Phase 2 结构先行）：旧 checkpoint 的 plan_dim（如 21）< 请求维度（如 57）时，
    plan_mlp 首层权重**前 pd_old 列**拷贝、尾部补零 —— 旧权重对前 21 维语义不变，
    尾部新字段从零开始学（与 PlanToken 尾部追加布局一致）。
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
    policy = FollowerPolicy(hidden=hd, plan_dim=pd, belief_dim=bd)
    target = policy.state_dict()
    for k, v in sd_src.items():
        if k not in target:
            continue
        tv = target[k]
        if tv.shape == v.shape:
            tv.copy_(v)
        elif k == "plan_mlp.0.weight" and v.dim() == 2 and v.shape[1] <= tv.shape[1]:
            # 尾部追加兼容：前 ckpt_pd 列原样拷贝，其余列保持 0（新字段从零学）
            tv[:, :v.shape[1]].copy_(v)
        elif k == "plan_mlp.0.bias":
            tv.copy_(v)
        # 其余形状不匹配（结构大改）→ 保持新初始化，不静默崩
    policy.load_state_dict(target)
    policy.eval()
    return policy


class FollowerPolicy(nn.Module):
    def __init__(self, hidden=256, plan_dim=None, belief_dim=None, num_entity=NUM_ENTITY,
                 stop_logit_bias=-1.0):
        """stop_logit_bias：新初始化时给 STOP logit 的偏置（负数=初始更愿意出牌）。

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
        self.plan_mlp = nn.Sequential(nn.Linear(plan_dim, 64), nn.ReLU())
        self.belief_mlp = nn.Sequential(nn.Linear(belief_dim, 64), nn.ReLU())
        enc_dim = cnn_out + hand_dim + scalar_dim + 64 + 64
        self.enc_fc = nn.Linear(enc_dim, hidden)
        self.gru_cell = nn.GRUCell(hidden, hidden)

        self.slot_head = nn.Linear(hidden, NUM_SLOT_OPTIONS)     # 出牌槽位 + ABILITY + STOP
        self.cell_head = nn.Linear(hidden, GRID_H * GRID_W)
        self.value_head = nn.Linear(hidden, 1)
        self.sub_emb = nn.Linear(NUM_SLOT_OPTIONS + 2, hidden)   # option onehot + (x/18, y/32)

        # 纯 RL 冷启动：初始压低 STOP logit（新随机初始化生效；load_checkpoint 会覆盖）
        if stop_logit_bias:
            with torch.no_grad():
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

    def _encode(self, obs, belief_token, plan_token):
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

        hand_feat = self.entity_emb(hand).reshape(1, -1)            # (1,40)
        scalar = torch.cat([elixir, time, next_card], dim=1)        # (1,3)

        plan_v = torch.as_tensor(plan_token, dtype=torch.float32).unsqueeze(0).to(self.device)
        belief_v = torch.as_tensor(belief_token, dtype=torch.float32).unsqueeze(0).to(self.device)
        plan_f = self.plan_mlp(plan_v)
        belief_f = self.belief_mlp(belief_v)

        fused = torch.cat([grid_feat, hand_feat, scalar, plan_f, belief_f], dim=1)
        return torch.relu(self.enc_fc(fused))                        # (1,hidden)

    def _slot_mask_tensor(self, mask):
        """把 mask 转成 (NUM_SLOT_OPTIONS,) 的合法选项掩码。

        - 已用槽位由 env 在 mask["slots"]/mask["cells"] 中体现（P1-6）；
        - bundle 已达 K_MAX（at_cap）时只放行 STOP（P1-18）。
        """
        sm = torch.ones(NUM_SLOT_OPTIONS, device=self.device)
        sm[:K_MAX] = torch.as_tensor(mask["slots"], dtype=torch.float32, device=self.device)
        sm[ABILITY_IDX] = 1.0 if mask.get("ability_legal") else 0.0
        sm[STOP_IDX] = 1.0
        if mask.get("at_cap"):
            sm[:K_MAX] = 0.0
            sm[ABILITY_IDX] = 0.0
            sm[STOP_IDX] = 1.0
        return sm

    def _plan_biases(self, plan_token):
        """从 plan 向量解析软偏置：(slot_bias(NUM_SLOT_OPTIONS,), cell_bias(H,W))。

        7h：BP 建议只作为 logit 软偏置——建议卡槽 +bias、hold_mask 命中槽 -bias、
        focus_region 中心附近落点 +bias。rollout 与 PPO 重放共用本函数，保证
        采样与 logprob 同分布（不改 env action_mask，不污染 BC/mask 契约）。
        """
        slot_bias = torch.zeros(NUM_SLOT_OPTIONS, device=self.device)
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
        sub = torch.zeros(1, NUM_SLOT_OPTIONS + 2, device=self.device)
        sub[0, option_idx] = 1.0
        sub[0, NUM_SLOT_OPTIONS] = x / GRID_W
        sub[0, NUM_SLOT_OPTIONS + 1] = y / GRID_H
        return sub

    def _sub_update(self, h, option_idx, x=0.0, y=0.0):
        return self.gru_cell(self.sub_emb(self._sub_vec(option_idx, x, y)), h)

    def _encode_batch(self, obs_list, belief_list, plan_list):
        """批量编码：N 个观测一次前向（CNN / embedding / MLP 全部批量化）。

        与 _encode 数值逐位一致（batch 维 = N），供 act_parallel / evaluate_batch 使用。
        """
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

        hand_feat = self.entity_emb(hand).reshape(N, -1)            # (N,40)
        scalar = torch.cat([elixir, time_, next_card], dim=1)       # (N,3)

        plan_v = torch.stack([torch.as_tensor(p, dtype=torch.float32) for p in plan_list]).to(self.device)
        belief_v = torch.stack([torch.as_tensor(b, dtype=torch.float32) for b in belief_list]).to(self.device)
        plan_f = self.plan_mlp(plan_v)
        belief_f = self.belief_mlp(belief_v)

        fused = torch.cat([grid_feat, hand_feat, scalar, plan_f, belief_f], dim=1)
        return torch.relu(self.enc_fc(fused))                        # (N,hidden)

    def act(self, obs, belief_token, plan_token, get_mask, hidden=None, deterministic=False):
        """在线动作生成：返回 (ActionBundle, logprob, value, hidden, masks)。

        - masks 为 rollout 过程中每个 decoder 步使用的动作掩码序列，供 PPO 重放；
        - 全程 no_grad、返回的 hidden 已 detach（P1-24），不跨决策步构建计算图。
        """
        with torch.no_grad():
            enc = self._encode(obs, belief_token, plan_token)
            if hidden is None:
                hidden = torch.zeros(1, self.hidden_dim, device=self.device)
            h = self.gru_cell(enc, hidden.detach())
            value = float(self.value_head(h).item())

            bundle = ActionBundle()
            logprob = 0.0
            masks = []
            slot_bias, cell_bias = self._plan_biases(plan_token)   # 7h：BP 软结构偏置
            for step in range(K_MAX + 2):
                mask = get_mask(bundle)
                masks.append(mask)
                slot_mask = self._slot_mask_tensor(mask)
                slot_logits = self.slot_head(h) + slot_bias
                slot_logits = slot_logits.masked_fill(slot_mask == 0, -1e9)
                slot_dist = torch.distributions.Categorical(
                    logits=F.log_softmax(slot_logits, dim=-1))
                if deterministic:
                    option = int(torch.argmax(slot_logits, dim=-1).item())
                else:
                    option = int(slot_dist.sample().item())
                logprob += float(slot_dist.log_prob(torch.tensor([option], device=self.device)).item())

                if option == STOP_IDX:
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
        N = len(obs_list)
        with torch.no_grad():
            enc = self._encode_batch(obs_list, belief_list, plan_list)   # (N,hidden)
            if hidden_list is None:
                h = self.gru_cell(enc, torch.zeros(N, self.hidden_dim, device=self.device))
            else:
                h0 = torch.stack([
                    (hid.reshape(-1) if hid is not None
                     else torch.zeros(self.hidden_dim, device=self.device))
                    for hid in hidden_list])
                h = self.gru_cell(enc, h0.detach())
            values = self.value_head(h)[:, 0].tolist()

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
                slot_masks = torch.stack(
                    [self._slot_mask_tensor(self._mask_or_fallback(masks_list[i], step))
                     for i in range(N)])
                slot_logits = self.slot_head(h) + torch.stack(
                    [bias_pairs[i][0] for i in range(N)])
                slot_logits = slot_logits.masked_fill(slot_masks == 0, -1e9)
                slot_dist = torch.distributions.Categorical(
                    logits=F.log_softmax(slot_logits, dim=-1))
                if deterministic:
                    options = torch.argmax(slot_logits, dim=-1)
                else:
                    options = slot_dist.sample()
                opts_list = options.tolist()               # 一次批量同步
                lp_all = slot_dist.log_prob(options)       # (N,)
                lp_list = lp_all.tolist()                  # 一次批量同步
                for i in range(N):
                    logprobs[i] += lp_list[i]
                active = [i for i in range(N) if opts_list[i] != STOP_IDX]
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
                sub_in = torch.zeros(len(active), NUM_SLOT_OPTIONS + 2, device=self.device)
                for m, i in enumerate(active):
                    opt = opts_list[i]
                    sub_in[m, opt] = 1.0
                    if opt < K_MAX:
                        x, y = dep_xy[i]
                        sub_in[m, NUM_SLOT_OPTIONS] = x / GRID_W
                        sub_in[m, NUM_SLOT_OPTIONS + 1] = y / GRID_H
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
        B = len(obs_list)
        enc = self._encode_batch(obs_list, belief_list, plan_list)     # (B,hidden)
        h_rows = []
        for i in range(B):
            base = (hidden_list[i] if hidden_list is not None and hidden_list[i] is not None
                    else torch.zeros(1, self.hidden_dim, device=self.device))
            h_rows.append(self.gru_cell(enc[i:i + 1], base.detach()))
        value = self.value_head(torch.cat(h_rows, dim=0))               # (B,1)

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
                    opts.append(STOP_IDX)
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
                sub_in = torch.zeros(len(grp), NUM_SLOT_OPTIONS + 2, device=self.device)
                for m, i in enumerate(grp):
                    sa = bundle_list[i].sub_actions[j]
                    opt = ABILITY_IDX if sa.kind == "ability" else sa.slot - 1
                    sub_in[m, opt] = 1.0
                    if sa.kind == "deploy":
                        sub_in[m, NUM_SLOT_OPTIONS] = sa.x / GRID_W
                        sub_in[m, NUM_SLOT_OPTIONS + 1] = sa.y / GRID_H
                hg = torch.cat([h_rows[i] for i in grp], dim=0)
                h_new = self.gru_cell(self.sub_emb(sub_in), hg)
                for m, i in enumerate(grp):
                    h_rows[i] = h_new[m:m + 1]

        return torch.stack(logprob), value, torch.stack(entropy)

    def value(self, obs, belief_token, plan_token, hidden=None) -> float:
        """只算当前状态的 value（不采样动作），用于截断 episode 的 GAE bootstrap（P1-7）。"""
        with torch.no_grad():
            enc = self._encode(obs, belief_token, plan_token)
            if hidden is None:
                hidden = torch.zeros(1, self.hidden_dim, device=self.device)
            h = self.gru_cell(enc, hidden.detach())
            return float(self.value_head(h).item())

    def evaluate(self, obs, belief_token, plan_token, bundle, masks, hidden=None):
        """可微重放给定 bundle（使用 rollout 时记录的掩码/隐状态）。

        返回 (logprob, value, hidden, entropy)：
        - entropy 为所有 decoder 步分布熵之和（真实熵，非 -lp，P0-2）。
        """
        enc = self._encode(obs, belief_token, plan_token)
        if hidden is None:
            hidden = torch.zeros(1, self.hidden_dim, device=self.device)
        h = self.gru_cell(enc, hidden.detach())
        value = self.value_head(h)

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
        logprob = logprob + slot_dist.log_prob(torch.tensor([STOP_IDX], device=self.device))
        return logprob, value, h, entropy
