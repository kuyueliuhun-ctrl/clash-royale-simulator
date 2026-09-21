# -*- coding: utf-8 -*-
"""**回归测试**（【R8】）：独立 act 头（预注册 `docs/fl_il_il2_prereg_2026-09-22.md` §7）。

被保护的改动（2026-09-22，C23 下游）：把「出不出牌」从 6 类共享头里拆成
`act_head`(2 类: STOP/ACT) + `slot_head`(5 类: 4 槽 + ABILITY)。

本测试的**核心断言是 P3**：§7.3 的「嵌入 6 类分布」可比性证明
    `p₆'(i) = p₂(ACT)·p₅(i)`（i ≤ K_MAX）、`p₆'(STOP) = p₂(STOP)`
⇒ 嵌入式 play 帧 `lp` 必须与解耦损失**逐值相等**。
**这条不是形式主义**：拆头后 stop 帧的 NLL `-log p₂(STOP)` 是共享头口径 `-log p₆(STOP)` 的
**边缘**（恒 ≤），play 帧又多一项 `-log p₂(ACT) ≥ 0`（恒更差）⇒ **两侧都不能直接比 NLL**。
没有这条对账，J8.2（top1/NLL）会得出**构造性的假结论**。

用法（仓库根）：
    ./.venv/Scripts/python.exe scripts/selftest_decoupled_act.py     # PASS -> rc=0
"""
import glob
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src", "clasher_new")
sys.path.insert(0, SRC)
os.chdir(SRC)

from rl.io_bootstrap import force_utf8_stdout  # noqa: E402

force_utf8_stdout()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rl.follower import (ABILITY_IDX, ACT_PLAY_IDX, ACT_STOP_IDX,  # noqa: E402
                         BELIEF_DIM, K_MAX, PLAN_DIM, STOP_IDX, FollowerPolicy,
                         load_checkpoint, save_checkpoint)

FAILS = []
N_ASSERT = [0]


def check(name, cond, detail=""):
    N_ASSERT[0] += 1
    print(f"[{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


def load_n(d, n):
    """取前 n 条 BC 样本（平铺 bc_*.pkl；与 human_play.load_bc_samples 同结构）。"""
    out = []
    for fn in sorted(glob.glob(os.path.join(d, "bc_*.pkl"))):
        with open(fn, "rb") as f:
            out.extend(pickle.load(f))
        if len(out) >= n:
            break
    return out[:n]


def _p2(pol, h, mask):
    am, sm = pol._act_slot_masks(mask)
    al = pol.act_head(h).masked_fill(am == 0, -1e9)
    return torch.softmax(al, dim=-1)[0], sm


def _p5(pol, h, sm, slot_bias):
    sl = (pol.slot_head(h) + slot_bias).masked_fill(sm == 0, -1e9)
    return torch.softmax(sl, dim=-1)[0]


def embedded_lp(pol, obs, tok, plan, bundle, masks):
    """用**嵌入 6 类口径** `p₆'` 重算 lp（§7.3）。独立于 `_evaluate_decoupled` 实现。"""
    from rl.observation import GRID_H, GRID_W
    fused, enc = pol._encode_parts(obs, tok, plan)
    h = pol.gru_cell(enc, torch.zeros(1, pol.hidden_dim, device=pol.device))
    slot_bias, cell_bias = pol._plan_biases(plan)
    lp = 0.0
    for i, sa in enumerate(bundle.sub_actions):
        mask = masks[i]
        p2, sm = _p2(pol, h, mask)
        p5 = _p5(pol, h, sm, slot_bias)
        p6 = torch.zeros(K_MAX + 2, device=pol.device)
        p6[:K_MAX + 1] = p2[ACT_PLAY_IDX] * p5          # 嵌入：ACT 边缘 × 槽条件
        p6[STOP_IDX] = p2[ACT_STOP_IDX]
        if sa.kind == "ability":
            lp += float(torch.log(p6[ABILITY_IDX]))
            h = pol._sub_update(h, ABILITY_IDX)
            continue
        option = sa.slot - 1
        lp += float(torch.log(p6[option]))
        cells = torch.as_tensor(mask["cells"][option], dtype=torch.float32, device=pol.device)
        cl = pol.cell_head(h).view(1, GRID_H, GRID_W) + cell_bias
        cl = cl.masked_fill(cells == 0, -1e9)
        cd = torch.distributions.Categorical(logits=torch.log_softmax(cl.reshape(1, -1), dim=-1))
        lp += float(cd.log_prob(torch.tensor([sa.y * GRID_W + sa.x], device=pol.device)))
        h = pol._sub_update(h, option, sa.x, sa.y)
    mask = masks[len(bundle.sub_actions)] if len(masks) > len(bundle.sub_actions) else {
        "slots": np.ones(K_MAX, dtype=bool),
        "cells": np.ones((K_MAX, GRID_H, GRID_W), dtype=bool), "ability_legal": False}
    p2, _sm = _p2(pol, h, mask)
    lp += float(torch.log(p2[ACT_STOP_IDX]))            # 终止步 = act 选 STOP
    return lp


def main():
    dev = "cpu"
    # ── P1 结构（两种架构） ──────────────────────────────────────────────
    p_def = FollowerPolicy(hidden=128, plan_dim=PLAN_DIM, belief_dim=BELIEF_DIM)
    p_dec = FollowerPolicy(hidden=128, plan_dim=PLAN_DIM, belief_dim=BELIEF_DIM,
                           decoupled_act=True)
    check("P1a 默认架构：slot_head=(6,128)、无 act_head、sub_space_dim=6",
          tuple(p_def.slot_head.weight.shape) == (6, 128)
          and not hasattr(p_def, "act_head") and p_def.sub_space_dim == 6)
    check("P1b 拆头架构：act_head=(2,128) ∧ slot_head=(5,128)",
          tuple(p_dec.act_head.weight.shape) == (2, 128)
          and tuple(p_dec.slot_head.weight.shape) == (5, 128))
    check("P1c ★ sub_emb 与 enc_fc **不变**（形状破坏面未扩散）",
          tuple(p_dec.sub_emb.weight.shape) == (128, 8)
          and tuple(p_dec.enc_fc.weight.shape) == (128, 2731),
          f"sub_emb={tuple(p_dec.sub_emb.weight.shape)} enc_fc={tuple(p_dec.enc_fc.weight.shape)}")

    # ── P2（快）默认架构与旧 ckpt 逐张量同形 ──────────────────────────────
    old_p = os.path.join(ROOT, "runs", "_fl_il_bc", "bc_fl.pt")
    if os.path.isfile(old_p):
        old = torch.load(old_p, map_location="cpu")["state_dict"]
        sd = p_def.state_dict()
        diff = [k for k in old if k not in sd or tuple(sd[k].shape) != tuple(old[k].shape)]
        check("P2 默认架构 state_dict 与旧 ckpt 逐张量同形（29/29）",
              not diff and len(sd) == len(old), f"n={len(sd)}/{len(old)} diff={diff[:4]}")
    else:
        check("P2 旧 ckpt 存在（前提）", False, old_p)

    # ── 真实样本 ────────────────────────────────────────────────────────
    tr = os.path.join(ROOT, "runs", "_fl_il_bc_save", "train")
    if not os.path.isdir(tr):
        tr = os.path.join(ROOT, "runs", "_fl_il_bc", "train")
    samples = load_n(tr, 60)
    plays = [s for s in samples if len(s[3].sub_actions) > 0]
    stops = [s for s in samples if len(s[3].sub_actions) == 0]
    check("数据集有 play/stop 两类样本（前提）",
          len(plays) > 0 and len(stops) > 0, f"play={len(plays)} stop={len(stops)}")

    # ── P3 ★★ 嵌入口径 == 解耦损失（§7.3 证明的可执行版） ────────────────
    maxd = 0.0
    for obs, tok, plan, bundle, masks in plays[:8]:
        with torch.no_grad():
            lp_dec = float(p_dec._evaluate_decoupled(obs, tok, plan, bundle, masks, None)[0])
        lp_emb = embedded_lp(p_dec, obs, tok, plan, bundle, masks)
        maxd = max(maxd, abs(lp_dec - lp_emb))
    check("P3 ★ play 帧：嵌入 6 类口径 lp == 解耦 lp（max|Δ| ≤ 1e-5）",
          maxd <= 1e-5, f"max|Δ|={maxd:.3e} (n={min(8, len(plays))})")

    # ── P4 拆头下**不存在**「静默拿到错掩码」的路径（loud failure 不变式） ────
    #: `_slot_mask_tensor` 产的是 6 类掩码（含 STOP_IDX=5），而拆头 `slot_head` 只有 5 维
    #: ⇒ 在拆头策略上调用它**必须炸**（IndexError），不许静默产出错掩码。
    #: 这是本批"不静默错"纪律的一部分（同 P6）。
    try:
        p_dec._slot_mask_tensor(masks[0])
        check("P4 拆头下 `_slot_mask_tensor`（6 类）**不可用** ⇒ 必须炸而非静默", False,
              "未抛异常")
    except IndexError:
        check("P4 拆头下 `_slot_mask_tensor`（6 类）**不可用** ⇒ 必须炸而非静默", True)
    #: 另证：拆头用的是 `_act_slot_masks`，其 slot 掩码维 = 5（与 slot_head 对齐）
    _am, _sm = p_dec._act_slot_masks(masks[0])
    check("P4 act_mask 维=2 ∧ slot_mask 维=5（与两个头对齐）",
          tuple(_am.shape) == (2,) and tuple(_sm.shape) == (5,),
          f"act={tuple(_am.shape)} slot={tuple(_sm.shape)}")

    # ── P5/P6 互斥与显式拒绝 ────────────────────────────────────────────
    try:
        FollowerPolicy(hidden=128, plan_dim=PLAN_DIM, belief_dim=BELIEF_DIM,
                       decoupled_act=True, intent_options=True)
        check("P5 decoupled_act ∧ intent_options ⇒ 显式 ValueError", False, "未报错")
    except ValueError:
        check("P5 decoupled_act ∧ intent_options ⇒ 显式 ValueError", True)
    raised = {}
    #: ⚠️ 必须给**全部位置参数**：守卫在函数体内，实参绑定失败会先抛 TypeError（不是我们要证的）
    for nm, fn in (("act_parallel", lambda: p_dec.act_parallel([], [], [], [])),
                   ("evaluate_batch", lambda: p_dec.evaluate_batch([], [], [], [], []))):
        try:
            fn()
            raised[nm] = False
        except NotImplementedError:
            raised[nm] = True
        except Exception as e:  # noqa: BLE001 —— 其它异常不算通过
            raised[nm] = f"{type(e).__name__}"
    check("P6 PPO 批量路径在拆头下**显式 NotImplementedError**（不静默错）",
          raised.get("act_parallel") is True and raised.get("evaluate_batch") is True,
          str(raised))

    # ── P7 act 掩码：无合法槽 ⇒ ACT 非法、STOP 恒合法 ────────────────────
    m_none = {"slots": np.zeros(K_MAX, dtype=bool),
              "cells": np.zeros((K_MAX, 32, 18), dtype=bool), "ability_legal": False}
    m_cap = {"slots": np.ones(K_MAX, dtype=bool),
             "cells": np.ones((K_MAX, 32, 18), dtype=bool), "ability_legal": True,
             "at_cap": True}
    a_none, s_none = p_dec._act_slot_masks(m_none)
    a_cap, s_cap = p_dec._act_slot_masks(m_cap)
    check("P7 无合法槽 ⇒ ACT 掩掉、STOP 合法（不会全 -1e9 ⇒ 无 NaN）",
          float(a_none[ACT_PLAY_IDX]) == 0.0 and float(a_none[ACT_STOP_IDX]) == 1.0
          and float(s_none.sum()) == 0.0)
    check("P7 at_cap ⇒ slot 全掩、ACT 掩掉、STOP 合法",
          float(a_cap[ACT_PLAY_IDX]) == 0.0 and float(s_cap.sum()) == 0.0)

    # ── P8 ckpt meta 往返 ───────────────────────────────────────────────
    tmp = os.path.join(ROOT, "runs", "_selftest_decoupled_act.pt")
    save_checkpoint(p_dec, tmp)
    md = torch.load(tmp, map_location="cpu")
    check("P8a save_checkpoint 写了 decoupled_act=True",
          bool(md.get("decoupled_act", False)) is True)
    p_back = load_checkpoint(tmp)
    check("P8b load_checkpoint 还原 decoupled_act=True 且 act_head 存在",
          bool(getattr(p_back, "decoupled_act", False)) and hasattr(p_back, "act_head"))
    # 旧 ckpt（decoupled=False）加载进拆头请求 ⇒ 应告警且不静默当成功
    if os.path.isfile(old_p):
        p_warn = load_checkpoint(old_p, decoupled_act=True)
        check("P8c 旧 ckpt + 请求拆头 ⇒ 仍能构造（告警路径），不抛异常",
              bool(getattr(p_warn, "decoupled_act", False)))
    try:
        os.remove(tmp)
    except OSError:
        pass

    print(f"\n=== selftest_decoupled_act: {len(FAILS)} failed / {N_ASSERT[0]} assertions ===")
    if FAILS:
        for f in FAILS:
            print("  FAIL:", f)
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
