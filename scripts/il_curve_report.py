# -*- coding: utf-8 -*-
"""把预算扫参的读数**汇成一张表**（禁手抄基线【R4】）。

读三类产物（全部由脚本生成，本脚本只聚合、不重算）：
* `curve_<tag>.json`      ← `scripts/il_eval_holdout.py`（留出读数 + trivial 基线，脚本复算）
* `nll_decomp_<tag>.json` ← `scripts/il_nll_decompose.py`（NLL 三项分解 + 参考值）
* `*.log`                 ← `scripts/il_bc_sweep.py` 的逐 epoch `mean_logprob`（训练集曲线）

用法（WSL python3 即可，只读 JSON）:
    python3 scripts/il_curve_report.py --dir docs/fl_il_2026-09-20/sweep \
        --rows "ep01:1:1e-3,ep02:2:1e-3,m0_3ep_1e-3:3:1e-3,ep05:5:1e-3,m1_10ep_1e-3:10:1e-3"
"""
import argparse
import glob
import json
import os
import re


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def train_curve(d):
    """从 sweep 日志里抓逐 epoch 训练集 mean_logprob：{epochs: {ep: lp}}。"""
    out = {}
    for fn in glob.glob(os.path.join(d, "*.log")):
        with open(fn, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = re.search(r"\[sweep\] ep=(\d+) lr=(\S+)\s+epoch (\d+)/\d+ mean_logprob=(-?[\d.]+)", line)
                if m:
                    e, lr, k, lp = int(m.group(1)), m.group(2), int(m.group(3)), float(m.group(4))
                    out.setdefault((e, lr), {})[k] = lp
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="docs/fl_il_2026-09-20/sweep")
    ap.add_argument("--rows", required=True, help="tag:epochs:lr,逗号分隔（顺序即表序）")
    args = ap.parse_args()
    tc = train_curve(args.dir)

    rows = []
    for spec in args.rows.split(","):
        tag, e, lr = spec.strip().split(":")
        e = int(e)
        cv = load(os.path.join(args.dir, f"curve_{tag}.json"))
        dc = load(os.path.join(args.dir, f"nll_decomp_{tag}.json"))
        try:
            lr_key = f"{float(lr):g}"          # 日志里 `_lr_tag` 用 %g（1e-3 → 0.001），两种写法都要能对上
        except ValueError:
            lr_key = lr
        #: 训练集曲线按 (总 epoch 数, lr) 分文件；同一 epoch 可能出现在多个 run 里（同 seed ⇒ 应当同值）
        cands = [v[e] for (E, L), v in tc.items() if L == lr_key and e in v]
        lp = None
        if cands:
            lp = cands[0]
            if max(cands) - min(cands) > 1e-9:
                print(f"⚠️ epoch {e} lr {lr_key} 在不同 run 里训练读数不一致：{cands}")
        rows.append((tag, e, lr, lp, cv, dc))

    print("| 格 | epochs | lr | 训练末 mean_lp | 留出 NLL | 分解 slot/落点/终止 | top1 | 落点(条件) | 槽1 | 槽2 | 槽3 | 槽4 | macro |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for tag, e, lr, lp, cv, dc in rows:
        if not cv:
            print(f"| {tag} | {e} | {lr} | {lp if lp is None else f'{lp:.3f}'} | *未产出* | | | | | | | | |")
            continue
        j = cv["J3"]
        ps = cv["per_slot_recall"]
        dec = (f"{dc['nll_slot']:.4f}/{dc['nll_cell']:.4f}/{dc['nll_stop']:.4f}" if dc else "—")
        print(f"| {tag} | {e} | {lr} | {lp if lp is None else f'{lp:.3f}'} | {j['nll_mean']:.4f} | {dec} | "
              f"{j['top1_option']:.4f} | {j['cell_exact_given_option']:.4f} | "
              + " | ".join(f"{ps[str(i)]:.4f}" for i in range(4))
              + f" | {cv['macro_recall_option']:.4f} |")
    if rows:
        cv0 = rows[-1][4] or rows[0][4]
        if cv0 and "baselines" in cv0:
            b = cv0["baselines"]
            print(f"\ntrivial 基线（脚本复算，同留出集 8,877）：uniform6={b['uniform_option6']:.4f} "
                  f"uniform4={b['uniform_option4']:.4f} majority_card={b['majority_card_top1_option']:.4f} "
                  f"majority_slot={b['majority_slot_top1_option']:.4f}(槽 {b['majority_slot_from_train'] + 1})")
        if rows[0][5] or any(r[5] for r in rows):
            dcx = next((r[5] for r in rows if r[5]), None)
            rf = dcx["refs"]
            print(f"NLL 参考：uniform6={rf['uniform6_slot']:.4f} 先验下界 H(槽)={rf['prior_only_floor_entropy_slot']:.4f} "
                  f"均匀合法格={rf['uniform_legal_cell']:.4f}")


if __name__ == "__main__":
    main()
