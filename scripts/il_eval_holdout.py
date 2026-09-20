# -*- coding: utf-8 -*-
"""IL（行为克隆）**留出集读数**仪器 —— 预注册 J3 的判据执行者。

背景：本仓的 BC 通路（`rl/human_play.py::train_bc_from_human`）**只在训练集自己身上**打印
`mean_logprob`，**没有**任何留出、准确率、top-k 或基线对照（`docs/agents/env.md` §2.4 仪器表缺口）。
本脚本补这一段：

* `top1_option`  —— 策略**确定性 argmax** 选的第一步 option 是否等于人类标签的 option
  （6 类空间：4 个出牌槽 + ABILITY + STOP；本实验的样本全是出牌帧）。
* `cell_exact`   —— 落点 576 格里 flat index 是否完全一致（含 `option` 一致与全体两个口径）。
* `nll`          —— 留出集 `-logprob` 均值（`policy.evaluate(..., hidden=None)`，与 BC 训练同口径）。
* **基线（脚本复算，禁手抄）**：uniform-6 = 1/6、uniform-4 = 1/4、
  **majority-card**（用训练集标签分布的最高频卡，映射到该帧手牌槽位；不在手牌则记 miss）。
* **健康度**：`logprob < -1e8` 的样本数（= bundle 不在 masks 支撑集内，会被静默掩掉）。

⚠️ 口径（与 BC 训练一致）：每条样本都当**首帧**（`hidden=None`）；`act`/`evaluate` 用同一实现。
⚠️ `get_mask` 重建：样本里存的 `masks` 就是 `masks_for` 的产物（`masks[i]` = 恰好 i 个子动作时的掩码）
   ⇒ `get_mask(partial) = masks[min(len(partial.sub_actions), len(masks)-1)]`。
   超出部分回落到最后一个掩码（只影响第 2 步之后，不影响本脚本要测的第一步）。

用法（Windows venv）:
    .venv/Scripts/python.exe scripts/il_eval_holdout.py \
        --ckpt runs/_fl_il_bc/bc_fl.pt --data-dir runs/_fl_il_bc/holdout \
        --train-dir runs/_fl_il_bc/train --out docs/fl_il_2026-09-20/holdout_metrics.json
"""
import argparse
import collections
import glob
import json
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src", "clasher_new")
ORIG_CWD = os.getcwd()
sys.path.insert(0, SRC)
os.chdir(SRC)

from rl.io_bootstrap import force_utf8_stdout  # noqa: E402

force_utf8_stdout()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from rl.follower import load_checkpoint  # noqa: E402
from rl.observation import ENTITY_NAMES  # noqa: E402


def _abs(p):
    """相对路径按 ORIG_CWD 解析；WSL 的 `/mnt/e/...` 译成 `E:\\...`（本仓 Windows 侧踩过）。"""
    m = p.replace("\\", "/")
    if m.startswith("/mnt/") and len(m) > 6:
        head, rest = m.split("/", 3)[2:]
        return head.upper() + ":\\" + rest.replace("/", "\\")
    return os.path.normpath(os.path.join(ORIG_CWD, p))


def load_dir(d, limit=0):
    """与 `human_play.load_bc_samples` 同口径：平铺 `bc_*.pkl`，内容 = list[5 元组]。"""
    out = []
    for fn in sorted(os.listdir(d)):
        if fn.startswith("bc_") and fn.endswith(".pkl"):
            with open(os.path.join(d, fn), "rb") as f:
                out.extend(pickle.load(f))
            if limit and len(out) >= limit:
                return out[:limit]
    return out


def majority_card(samples):
    c = collections.Counter()
    for s in samples:
        for sa in s[3].sub_actions:
            if sa.kind == "deploy":
                c[int(s[0]["hand"][sa.slot - 1])] += 1        # hand 存的是 ENTITY_NAMES 下标
    return c


def make_get_mask(masks):
    """由**本帧全掩码** `masks[0]` 重建 `get_mask(partial)`。

    ⚠️ 不能直接按 `len(partial.sub_actions)` 去索引样本里存的 `masks` 序列：那串掩码是按**人类那次
    出牌**的 partial 建的，`used_slots` 是人类的槽位；策略一旦选了**别的**槽，`follower.act` 的
    「掩码不变式·1」会立刻判不一致并抛 `RuntimeError`（实测踩过）。
    正确做法 = 取本帧全掩码，按**策略自己**的 partial 扣掉已用槽位（与
    `env_wrapper.get_action_mask_for(pid, partial)` 同语义：本帧状态未推进，只有 used 集合在变）。
    """
    m0 = masks[0]

    def get_mask(partial):
        used = sorted({int(sa.slot) - 1 for sa in partial.sub_actions if sa.kind == "deploy"})
        m = dict(m0)
        slots = np.array(m0["slots"], dtype=bool, copy=True)
        cells = np.array(m0["cells"], dtype=bool, copy=True)
        for u in used:
            if 0 <= u < slots.shape[0]:
                slots[u] = False
                cells[u] = False
        m["slots"], m["cells"] = slots, cells
        m["used_slots"] = np.array(used, dtype=np.int32)
        m["any_legal"] = bool(slots.any())
        return m

    return get_mask


def evaluate(ckpt, holdout, train, limit, random_init=False):
    if random_init:
        #: 未训练对照：维度从**第一条样本**反推（与 `train_bc_from_human:214-215` 同口径）。
        #: ⚠️ **必须播种**：`FollowerPolicy` 的初始化走 torch 全局 RNG，不播种的话每次运行
        #: 抽到不同权重 ⇒ 对照读数不可复现（实测两次 `top1` 0.3287 vs 0.3261、`nll` 8.72 vs 8.80）。
        torch.manual_seed(0)
        from rl.follower import FollowerPolicy
        policy = FollowerPolicy(hidden=128, plan_dim=len(holdout[0][2]),
                                belief_dim=len(holdout[0][1]))
        policy.eval()
    else:
        policy = load_checkpoint(ckpt)      # follower.py:184 `return policy`（已 .eval()）
    if limit:
        holdout = holdout[:limit]

    maj = majority_card(train)
    #: **trivial 基线的真正上限**：常数预测训练集最高频的**槽位**。
    #: （实测：随机初始化策略在 12 条 smoke 留出上就拿到 41.7%，恰好等于该集内的 majority-slot
    #:  ⇒ 只报 uniform(1/6) 会把「有偏但没学到东西」误判成 J3 PASS【R16】）
    slot_dist = collections.Counter(int(s[3].sub_actions[0].slot) - 1 for s in train) or \
        collections.Counter({0: 1})
    maj_slot = slot_dist.most_common(1)[0][0]
    maj_slot_hit = 0
    conf = collections.Counter()      # (label_opt, pred_opt) 列联表
    maj_ids = [i for i, _n in maj.most_common(8)]
    counts = collections.Counter()
    n_opt = n_cell = n_opt_and_cell = 0
    n_opt_cond_cell = 0
    lps, health = [], 0
    opt_hist = collections.Counter()
    maj_hit = maj_den = 0
    for obs, tok, plan, bundle, masks in holdout:
        lab = bundle.sub_actions[0]
        lab_opt, lab_x, lab_y = int(lab.slot) - 1, int(lab.x), int(lab.y)

        get_mask = make_get_mask(masks)

        pred, _lp_act, _v, _h, _mk = policy.act(obs, tok, plan, get_mask,
                                                hidden=None, deterministic=True)
        #: NLL 用 `evaluate`（与 BC 训练**同一函数、同一 hidden=None 口径**）；
        #: `act` 返回的是「确定性 argmax 那条路的 logprob」，不是标签的似然，不能当 NLL。
        with torch.no_grad():
            lp, _val, _hh, _ent = policy.evaluate(obs, tok, plan, bundle, masks, hidden=None)
        lps.append(float(lp))
        if float(lp) < -1e8:
            health += 1
        if pred.sub_actions:
            p = pred.sub_actions[0]
            p_opt, px, py = int(p.slot) - 1, int(p.x), int(p.y)
        else:
            p_opt, px, py = -1, -1, -1
        ok_opt = (p_opt == lab_opt)
        ok_cell = ok_opt and (px == lab_x and py == lab_y)
        n_opt += int(ok_opt)
        n_cell += int(px == lab_x and py == lab_y)
        n_opt_and_cell += int(ok_cell)
        if ok_opt:
            n_opt_cond_cell += int(px == lab_x and py == lab_y)
        opt_hist[p_opt] += 1
        counts["n"] += 1
        maj_slot_hit += int(lab_opt == maj_slot)
        conf[(lab_opt, p_opt)] += 1
        #: majority-card 基线：把训练集最高频卡映射到本帧手牌（不在手牌 ⇒ miss）
        if maj_ids:
            hand = [int(v) for v in obs["hand"][:4]]
            hit = maj_ids[0] in hand
            maj_den += 1
            maj_hit += int(hit and (hand.index(maj_ids[0]) == lab_opt))

    n = counts["n"]
    #: **对「常数预测」免疫的指标**：各槽位 recall 的宏平均。
    #: 常数预测器（如恒选槽 4）宏平均恒 = 1/4 = 25%，而普通 top-1 会被槽位分布偏斜抬高
    #: （实测 majority_slot = 39.8% > 预注册 40% 闸门的邻域 ⇒ 单看 top1 不能判「学到了」）。
    recalls = {}
    for j in range(4):
        den = sum(v for (lab, _p), v in conf.items() if lab == j)
        if den:
            recalls[j] = conf[(j, j)] / den
    macro = (sum(recalls.values()) / len(recalls)) if recalls else None
    return {
        "option_confusion_label_x_pred": {f"{a}->{b}": v for (a, b), v in sorted(conf.items())},
        "macro_recall_option": macro,
        "per_slot_recall": recalls,
        "ckpt": ("<random-init|torch.manual_seed(0)>" if random_init else os.path.basename(ckpt)),
        "holdout_samples": n,
        "train_samples": len(train),
        "J3": {
            "top1_option": n_opt / n if n else None,
            "cell_exact": n_cell / n if n else None,
            "cell_exact_given_option": (n_opt_cond_cell / n_opt) if n_opt else None,
            "nll_mean": (-sum(lps) / len(lps)) if lps else None,
            "logprob_lt_minus_1e8": health,
            "gate_top1_ge_0.40": "PASS" if n and n_opt / n >= 0.40 else
                                 ("WEAK" if n and n_opt / n >= 0.25 else "FAIL"),
        },
        "baselines": {
            "uniform_option6": 1.0 / 6.0,
            "uniform_option4": 0.25,
            "majority_card_top1_option": (maj_hit / maj_den) if maj_den else None,
            "majority_slot_top1_option": (maj_slot_hit / n) if n else None,
            "majority_slot_from_train": int(maj_slot),
            "majority_card_declared": [ENTITY_NAMES[i] for i in maj_ids[:5]],
        },
        "label_distribution_top_slots": dict(
            sorted(collections.Counter(
                int(s[3].sub_actions[0].slot) - 1 for s in holdout).items())),
        "pred_option_histogram": dict(sorted(opt_hist.items())),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--random-init", action="store_true", help="对照：未训练（随机初始化）策略")
    ap.add_argument("--data-dir", required=True, help="留出集目录（平铺 bc_*.pkl）")
    ap.add_argument("--train-dir", default=None, help="训练集目录（只用于 majority 基线）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if not args.ckpt and not args.random_init:
        ap.error("需要 --ckpt 或 --random-init")
    for name in ("ckpt", "data_dir", "train_dir", "out"):
        v = getattr(args, name, None)
        if v:
            setattr(args, name, _abs(v))

    holdout = load_dir(args.data_dir, args.limit)
    train = load_dir(args.train_dir) if args.train_dir else []
    if not holdout:
        print(f"[eval] {args.data_dir} 下没有 bc_*.pkl")
        return 2
    res = evaluate(args.ckpt, holdout, train, args.limit, args.random_init)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    j, b = res["J3"], res["baselines"]
    print(f"[eval] holdout={res['holdout_samples']}  "
          f"top1_option={j['top1_option']}  cell_exact={j['cell_exact']}  "
          f"nll={j['nll_mean']}")
    print(f"  baselines: uniform6={b['uniform_option6']:.4f} uniform4={b['uniform_option4']} "
          f"majority_card={b['majority_card_top1_option']} "
          f"majority_slot={b['majority_slot_top1_option']} (slot {b['majority_slot_from_train']})")
    print(f"  macro_recall_option={res['macro_recall_option']}  "
          f"per_slot_recall={res['per_slot_recall']}")
    print(f"  verdict J3 = {j['gate_top1_ge_0.40']}  "
          f"health(logprob<-1e8)={j['logprob_lt_minus_1e8']}")
    if args.out:
        print(f"[eval] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
