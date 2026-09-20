# -*- coding: utf-8 -*-
"""把 IL 的 NLL **拆成三项**：选牌 / 落点 / 终止 —— 「算力买到的是什么」的定量仪器。

背景（预注册 `docs/fl_il_scaling_prereg_2026-09-20.md` §4 J-M5）：
`FollowerPolicy.evaluate` 返回的是**一个和**（`follower.py:887-928`）：
    logprob = log p(slot) + log p(cell | slot) + log p(STOP)
本仓数据每帧恰好 1 个子动作（人类单事件帧）⇒ 三项分别 = **选牌 / 落点 / 终止**。
上游判读只说「top-1 40.42% 只比 trivial 高 0.58 pp」，**没有**说清 NLL 那 3 个 nat 的下降发生在哪一项。
本脚本补齐：把每一项的 NLL 与**同口径参考值**（见下）并排放，于是「预算买到的是落点还是选牌」成为读数问题。

⚠️ **不改 `follower.py`**：本脚本**复刻** `evaluate` 的前向（同 `_encode_parts` / `gru_cell` / `_plan_biases` /
`_slot_mask_tensor` / `slot_head` / `cell_head` / `_sub_update` / `terminal_option`），并**自带一致性闸门**：
逐帧 `|复刻三项之和 − policy.evaluate 的 lp|` 必须 ≤ **1e-5**，否则**直接报错退出**（复刻漂了 ⇒ 读数不可用）。

参考值（同一次遍历、同掩码，脚本复算，禁手抄【R4】）：
* 选牌项：`uniform6 = ln 6`；`constant_slot`（恒选留出集最高频槽）= `-ln(该槽留出占比)`；`label_entropy` = 槽标签经验熵（**先验匹配**下界）。
* 落点项：`uniform_legal` = 逐帧 `ln(人类所选槽的合法格数)` 的均值（均匀撒在该槽合法格上的 NLL）。
* 终止项：`-ln p(STOP)`（掩码通常把 STOP 压成唯一合法 ⇒ ≈0）。

用法（Windows venv，从仓库根跑）:
    ./.venv/Scripts/python.exe scripts/il_nll_decompose.py \
        --ckpt runs/_fl_il_bc/sweep/bc_fl_e10_lr0.001.pt \
        --data-dir runs/_fl_il_bc/holdout --out docs/fl_il_2026-09-20/sweep/nll_decomp_m1.json
"""
import argparse
import collections
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import il_eval_holdout as ie  # noqa: E402  （复用 _abs / load_dir；导入时 chdir 到 SRC）

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from rl.follower import ABILITY_IDX, GRID_H, GRID_W, load_checkpoint  # noqa: E402


def evaluate_parts(policy, obs, tok, plan, bundle, masks, hidden=None):
    """复刻 `FollowerPolicy.evaluate`（follower.py:874-929），返回三项 logprob 与总和。

    ⚠️ 逐行对应；**不**做任何「等价改写」，以便一致性闸门能真正抓住漂移。
    """
    fused, enc = policy._encode_parts(obs, tok, plan)
    if hidden is None:
        hidden = torch.zeros(1, policy.hidden_dim, device=policy.device)
    h = policy.gru_cell(enc, hidden.detach())
    lp_slot = lp_cell = 0.0
    slot_bias, cell_bias = policy._plan_biases(plan)
    for i, sa in enumerate(bundle.sub_actions):
        mask = masks[i]
        slot_mask = policy._slot_mask_tensor(mask)
        slot_logits = policy.slot_head(h) + slot_bias
        slot_logits = slot_logits.masked_fill(slot_mask == 0, -1e9)
        slot_dist = torch.distributions.Categorical(logits=F.log_softmax(slot_logits, dim=-1))
        if sa.kind == "ability":
            option = ABILITY_IDX
            lp_slot += float(slot_dist.log_prob(torch.tensor([option], device=policy.device)).item())
            h = policy._sub_update(h, ABILITY_IDX)
            continue
        option = sa.slot - 1
        lp_slot += float(slot_dist.log_prob(torch.tensor([option], device=policy.device)).item())
        cells = torch.as_tensor(mask["cells"][option], dtype=torch.float32, device=policy.device)
        cell_logits = policy.cell_head(h).view(1, GRID_H, GRID_W) + cell_bias
        cell_logits = cell_logits.masked_fill(cells == 0, -1e9)
        flat = cell_logits.reshape(1, -1)
        cell_dist = torch.distributions.Categorical(logits=F.log_softmax(flat, dim=-1))
        cell_idx = sa.y * GRID_W + sa.x
        lp_cell += float(cell_dist.log_prob(torch.tensor([cell_idx], device=policy.device)).item())
        h = policy._sub_update(h, option, sa.x, sa.y)

    mask = masks[len(bundle.sub_actions)] if len(masks) > len(bundle.sub_actions) else {
        "slots": np.ones(4, dtype=bool), "cells": np.ones((4, GRID_H, GRID_W), dtype=bool),
        "ability_legal": False,
    }
    slot_mask = policy._slot_mask_tensor(mask)
    slot_logits = policy.slot_head(h) + slot_bias
    slot_logits = slot_logits.masked_fill(slot_mask == 0, -1e9)
    slot_dist = torch.distributions.Categorical(logits=F.log_softmax(slot_logits, dim=-1))
    _term = policy.terminal_option(bundle)
    lp_stop = float(slot_dist.log_prob(torch.tensor([_term], device=policy.device)).item())
    return lp_slot, lp_cell, lp_stop


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--random-init", action="store_true")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if not args.ckpt and not args.random_init:
        ap.error("需要 --ckpt 或 --random-init")
    for name in ("ckpt", "data_dir", "out"):
        v = getattr(args, name, None)
        if v:
            setattr(args, name, ie._abs(v))

    holdout = ie.load_dir(args.data_dir, args.limit)
    if not holdout:
        print(f"[decomp] {args.data_dir} 下没有 bc_*.pkl")
        return 2
    if args.random_init:
        torch.manual_seed(0)
        from rl.follower import FollowerPolicy
        policy = FollowerPolicy(hidden=128, plan_dim=len(holdout[0][2]), belief_dim=len(holdout[0][1]))
        policy.eval()
        tag = "random-init|torch.manual_seed(0)"
    else:
        policy = load_checkpoint(args.ckpt)
        tag = os.path.basename(args.ckpt)

    slots = collections.Counter()
    lp_s = lp_c = lp_t = lp_tot = 0.0
    unif_legal = 0.0
    worst = 0.0
    n = 0
    for obs, tok, plan, bundle, masks in holdout:
        sa = bundle.sub_actions[0]
        slots[int(sa.slot) - 1] += 1
        a, b, c = evaluate_parts(policy, obs, tok, plan, bundle, masks, hidden=None)
        with torch.no_grad():
            lp, _v, _h, _e = policy.evaluate(obs, tok, plan, bundle, masks, hidden=None)
        worst = max(worst, abs((a + b + c) - float(lp)))
        lp_s += a; lp_c += b; lp_t += c; lp_tot += float(lp)
        unif_legal += math.log(max(1, int(np.asarray(masks[0]["cells"][int(sa.slot) - 1]).sum())))
        n += 1

    #: **一致性闸门**：复刻漂了就报错，不产出读数（预注册 §4 J-M5）
    if worst > 1e-5:
        print(f"[decomp] ✗ 复刻与 evaluate 不一致：max|Δ|={worst:.3e} > 1e-5 —— 读数不可用")
        return 3
    print(f"[decomp] ✓ 复刻一致性 max|Δ|={worst:.2e}（{n} 帧，阈值 1e-5）")

    maj_slot, maj_cnt = slots.most_common(1)[0]
    ent = -sum((c / n) * math.log(c / n) for c in slots.values())
    res = {
        "ckpt": tag, "holdout_samples": n,
        "nll_total": lp_tot / n,
        "nll_slot": lp_s / n,          # 选牌项
        "nll_cell": lp_c / n,          # 落点项
        "nll_stop": lp_t / n,          # 终止项
        "refs": {
            "uniform6_slot": math.log(6),
            "constant_majority_slot": -math.log(maj_cnt / n),
            "majority_slot_index": maj_slot,
            "label_entropy_slot": ent,
            "uniform_legal_cell": unif_legal / n,
        },
        "replica_max_abs_diff": worst,
        "note": "nll_total = nll_slot + nll_cell + nll_stop（每帧恰 1 子动作）；复刻 evaluate 自带 1e-5 闸门",
    }
    print(f"[decomp] {tag}")
    print(f"  total={res['nll_total']:.4f}  slot(选牌)={res['nll_slot']:.4f}  "
          f"cell(落点)={res['nll_cell']:.4f}  stop={res['nll_stop']:.4f}")
    print(f"  参考: uniform6={res['refs']['uniform6_slot']:.4f}  "
          f"恒选槽{maj_slot + 1}={res['refs']['constant_majority_slot']:.4f}  "
          f"标签熵={ent:.4f}  均匀合法格={res['refs']['uniform_legal_cell']:.4f}")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        print(f"[decomp] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
