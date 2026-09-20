# -*- coding: utf-8 -*-
"""**配对**判别仪器：同一留出集上比较两个 IL ckpt（`A` = 对照，`B` = 处理）。

为什么需要它（预注册 `docs/fl_il_scaling_prereg_2026-09-20.md` §4 J-M1/J-M2）：
`scripts/il_eval_holdout.py` 只出**每个 ckpt 各自的**聚合读数；【R16】要求「阈值余量 ≫ run 间散布」时
改用**配对**判据。本脚本在**一次**遍历里同时跑 A/B 两个策略，于是能给出：

* **NLL 差**：逐样本 `nll_B - nll_A` 的均值 / 中位数 / 符号检验（A 赢 vs B 赢的计数 + 精确二项 p）；
* **top-1 option 的 McNemar 表**：`b01`（A 错 B 对）/ `b10`（A 对 B 错）/ 精确 p；
* **逐槽 recall 的配对变化**（A vs B，同一标签子集）——比「B 的 recall 是否超随机基线」更贴近病灶；
* **`cell_exact_given_option`** 两口径 + 配对（落点是否退步）。

口径与 `il_eval_holdout.py` **完全相同**（直接复用它导出的 `make_get_mask` / `load_dir`）：
`hidden=None`、NLL 走 `policy.evaluate`、top-1 走 `policy.act(deterministic=True)`。
⚠️ 顺序敏感：`b01`/`b10` 只在「同一批样本、同一口径」下有意义（本脚本保证）。

用法（Windows venv，从仓库根跑）:
    ./.venv/Scripts/python.exe scripts/il_paired_compare.py \
        --ckpt-a runs/_fl_il_bc/sweep/bc_fl_e3_lr0.001.pt \
        --ckpt-b runs/_fl_il_bc/sweep/bc_fl_e10_lr0.001.pt \
        --data-dir runs/_fl_il_bc/holdout --out docs/fl_il_2026-09-20/sweep/paired_m1_vs_m0.json
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import il_eval_holdout as ie  # noqa: E402  （复用其 _abs / load_dir / make_get_mask；导入时会 chdir 到 SRC）

import numpy as np  # noqa: E402
import torch  # noqa: E402
from rl.follower import load_checkpoint  # noqa: E402


def binom_two_sided(k, n, p=0.5):
    """精确二项检验（双侧，= 2×P(X ≤ min(k, n-k))，截断到 1）。R16 要求的「不看单点观测」。"""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def mcnemar_exact(b01, b10):
    """McNemar 精确检验（双侧）：只看两张表不一致的格子。"""
    return binom_two_sided(b01, b01 + b10)


def run_policy(policy, holdout):
    """一次遍历：逐样本 (nll, pred_opt, pred_cell) + 标签。"""
    nlls, preds = [], []
    labels = []
    health = 0
    for obs, tok, plan, bundle, masks in holdout:
        lab = bundle.sub_actions[0]
        lab_opt, lab_x, lab_y = int(lab.slot) - 1, int(lab.x), int(lab.y)
        labels.append((lab_opt, lab_x, lab_y))
        pred, _lp, _v, _h, _mk = policy.act(obs, tok, plan, ie.make_get_mask(masks),
                                            hidden=None, deterministic=True)
        if pred.sub_actions:
            p = pred.sub_actions[0]
            preds.append((int(p.slot) - 1, int(p.x), int(p.y)))
        else:
            preds.append((-1, -1, -1))
        with torch.no_grad():
            lp, _val, _hh, _ent = policy.evaluate(obs, tok, plan, bundle, masks, hidden=None)
        nlls.append(-float(lp))
        if float(lp) < -1e8:
            health += 1
    return {"nll": np.asarray(nlls, dtype=np.float64),
            "pred": preds, "label": labels, "health": health}


def summarize(res, labels):
    """与 il_eval_holdout 同口径的聚合（用于与单 ckpt 读数对账）。"""
    n = len(labels)
    top1 = sum(int(p[0] == l[0]) for p, l in zip(res["pred"], labels))
    cell_any = sum(int(p[0] == l[0] and p[1] == l[1] and p[2] == l[2])
                   for p, l in zip(res["pred"], labels))
    cond_den = sum(int(p[0] == l[0]) for p, l in zip(res["pred"], labels))
    cell_cond = sum(int(p[0] == l[0] and p[1] == l[1] and p[2] == l[2])
                    for p, l in zip(res["pred"], labels))
    recalls = {}
    for j in range(4):
        den = sum(int(l[0] == j) for l in labels)
        hit = sum(int(l[0] == j and p[0] == j) for p, l in zip(res["pred"], labels))
        recalls[str(j)] = (hit / den) if den else None
    macro = sum(v for v in recalls.values() if v is not None) / max(1, sum(v is not None for v in recalls.values()))
    return {"n": n, "top1_option": top1 / n, "cell_exact": cell_any / n,
            "cell_exact_given_option": (cell_cond / cond_den) if cond_den else None,
            "nll_mean": float(res["nll"].mean()), "per_slot_recall": recalls,
            "macro_recall_option": macro, "logprob_lt_minus_1e8": res["health"]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-a", required=True, help="对照（基准）")
    ap.add_argument("--ckpt-b", required=True, help="处理（要判的）")
    ap.add_argument("--data-dir", required=True, help="留出集目录（平铺 bc_*.pkl）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    for name in ("ckpt_a", "ckpt_b", "data_dir", "out"):
        v = getattr(args, name, None)
        if v:
            setattr(args, name, ie._abs(v))

    holdout = ie.load_dir(args.data_dir, args.limit)
    if not holdout:
        print(f"[paired] {args.data_dir} 下没有 bc_*.pkl")
        return 2
    print(f"[paired] holdout={len(holdout)}  A={os.path.basename(args.ckpt_a)}  "
          f"B={os.path.basename(args.ckpt_b)}", flush=True)

    pa = load_checkpoint(args.ckpt_a)
    pb = load_checkpoint(args.ckpt_b)
    ra = run_policy(pa, holdout)
    rb = run_policy(pb, holdout)
    labels = ra["label"]
    sa, sb = summarize(ra, labels), summarize(rb, labels)

    d = rb["nll"] - ra["nll"]                      # >0 = B 更差
    n = len(d)
    n_b_better = int((d < 0).sum())
    n_a_better = int((d > 0).sum())
    ties = int((d == 0).sum())
    b01 = sum(int(a[0] != l[0] and b[0] == l[0]) for a, b, l in zip(ra["pred"], rb["pred"], labels))
    b10 = sum(int(a[0] == l[0] and b[0] != l[0]) for a, b, l in zip(ra["pred"], rb["pred"], labels))
    per_slot_paired = {}
    for j in range(4):
        idx = [i for i, l in enumerate(labels) if l[0] == j]
        per_slot_paired[str(j)] = {
            "n": len(idx),
            "A_hit": sum(int(ra["pred"][i][0] == j) for i in idx),
            "B_hit": sum(int(rb["pred"][i][0] == j) for i in idx),
            "B_only": sum(int(ra["pred"][i][0] != j and rb["pred"][i][0] == j) for i in idx),
            "A_only": sum(int(ra["pred"][i][0] == j and rb["pred"][i][0] != j) for i in idx),
        }

    res = {
        "ckpt_a": os.path.basename(args.ckpt_a), "ckpt_b": os.path.basename(args.ckpt_b),
        "holdout_samples": n,
        "A": sa, "B": sb,
        "nll_paired": {"mean_diff_B_minus_A": float(d.mean()),
                       "median_diff_B_minus_A": float(np.median(d)),
                       "B_better": n_b_better, "A_better": n_a_better, "ties": ties,
                       "sign_test_p": binom_two_sided(n_b_better, n_b_better + n_a_better)},
        "top1_mcnemar": {"A_wrong_B_right": b01, "A_right_B_wrong": b10,
                         "B_right_A_right": sum(int(a[0] == l[0] and b[0] == l[0])
                                                for a, b, l in zip(ra["pred"], rb["pred"], labels)),
                         "both_wrong": sum(int(a[0] != l[0] and b[0] != l[0])
                                           for a, b, l in zip(ra["pred"], rb["pred"], labels)),
                         "exact_p": mcnemar_exact(b01, b10)},
        "per_slot_paired": per_slot_paired,
        "note": "口径与 il_eval_holdout.py 相同（hidden=None；NLL 走 evaluate；top-1 走 act(deterministic)）",
    }
    print(f"[paired] A top1={sa['top1_option']:.4f} nll={sa['nll_mean']:.4f}  "
          f"slot_recall={[round(v, 4) if v else v for v in sa['per_slot_recall'].values()]}")
    print(f"[paired] B top1={sb['top1_option']:.4f} nll={sb['nll_mean']:.4f}  "
          f"slot_recall={[round(v, 4) if v else v for v in sb['per_slot_recall'].values()]}")
    print(f"[paired] NLL Δ(B-A) mean={d.mean():+.4f} median={np.median(d):+.4f}  "
          f"B better {n_b_better} / A better {n_a_better} / tie {ties}  p={res['nll_paired']['sign_test_p']:.3g}")
    print(f"[paired] top1 McNemar: A→B fixed={b01}  B broke={b10}  p={res['top1_mcnemar']['exact_p']:.3g}")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        print(f"[paired] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
