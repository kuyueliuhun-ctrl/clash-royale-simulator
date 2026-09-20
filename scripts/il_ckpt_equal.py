# -*- coding: utf-8 -*-
"""**跨进程确定性闸门**（预注册 `docs/fl_il_scaling_prereg_2026-09-20.md` §4 J-M6）。

各扫参格同 seed（`torch.manual_seed(0)` + `np.random.seed(0)`，模型构造在 permutation 之前）⇒
「实验格」与「长跑格在同一 epoch 的快照」在数学上应当是**同一组参数**。本脚本逐张量核对：

* 逐张量 `torch.equal`（要求真）＋ `max|Δ|`；
* 汇总 PASS/FAIL 与退出码（FAIL = 2）。

用途：证明扫参**真的**是单变量（不是「两次同配置其实抽了不同的签」），
并让「长跑格」的中间快照可以**直接替代**独立的短格（省一整次 10-epoch 训练）。
⚠️ 语义：`torch.equal` 为真 = 逐位相同；数值几乎相同但不 equal ⇒ 仍报 FAIL（存在未控随机源/线程数差异）。

用法（Windows venv，从仓库根跑）:
    ./.venv/Scripts/python.exe scripts/il_ckpt_equal.py \
        --a runs/_fl_il_bc/bc_fl.pt --b runs/_fl_il_bc/sweep/bc_fl_e3_lr0.001.pt
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src", "clasher_new")
ORIG_CWD = os.getcwd()
sys.path.insert(0, SRC)
os.chdir(SRC)

from rl.io_bootstrap import force_utf8_stdout  # noqa: E402

force_utf8_stdout()

import torch  # noqa: E402


def _abs(p):
    m = p.replace("\\", "/")
    if m.startswith("/mnt/") and len(m) > 6:
        head, rest = m.split("/", 3)[2:]
        return head.upper() + ":\\" + rest.replace("/", "\\")
    return os.path.normpath(os.path.join(ORIG_CWD, p))


def state_dict(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    return d["state_dict"] if isinstance(d, dict) and "state_dict" in d else d


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    args = ap.parse_args(argv)
    a, b = state_dict(_abs(args.a)), state_dict(_abs(args.b))
    ka, kb = set(a.keys()), set(b.keys())
    if ka != kb:
        print(f"[ckpt-eq] ✗ 键集合不同：only-A={sorted(ka - kb)} only-B={sorted(kb - ka)}")
        return 2
    bad = []
    worst = 0.0
    for k in sorted(ka):
        ta, tb = a[k], b[k]
        if ta.shape != tb.shape:
            bad.append((k, "shape"))
            continue
        if not torch.equal(ta, tb):
            d = float((ta.float() - tb.float()).abs().max().item())
            worst = max(worst, d)
            bad.append((k, f"max|Δ|={d:.3e}"))
    print(f"[ckpt-eq] {os.path.basename(args.a)} vs {os.path.basename(args.b)}："
          f"{len(ka)} 张量，{len(ka) - len(bad)} 逐位相同")
    for k, why in bad[:10]:
        print(f"  ✗ {k}: {why}")
    if bad:
        print(f"[ckpt-eq] FAIL（最差 max|Δ|={worst:.3e}）")
        return 2
    print("[ckpt-eq] PASS（逐张量 torch.equal）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
