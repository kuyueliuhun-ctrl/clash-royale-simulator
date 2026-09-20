# -*- coding: utf-8 -*-
"""BC（IL）**单步耗时标定**：`batch=1` 训练到底卡在哪 —— 线程数还是开销。

用途（预注册 `docs/fl_il_scaling_prereg_2026-09-20.md` §6 的跑前环境标定）：
本仓 BC 训练逐样本 `evaluate → -logprob → backward → Adam step`（`human_play.py:222-228`），
**batch=1**。标定得到的事实决定了扫参怎么并行：

* 实测（本机 16 核，417 样本 × 300 步，两种线程数各两次）：
  `threads=1` → 11.55 / 11.73 ms/step；`threads=8` → 11.68 / 11.37 ms/step
  ⇒ **线程数完全无差别 = 开销瓶颈**（每步 Python/调度开销主导，矩阵乘法占比小）。
  ⇒ 提吞吐的唯一正确方式是**多进程 × 单线程**（每进程 RSS 2.49 GB），不是加线程。

用法（Windows venv，从仓库根跑）:
    ./.venv/Scripts/python.exe scripts/il_bc_bench.py --threads 1 --files 20 --steps 300
"""
import argparse
import glob
import os
import pickle
import sys
import time

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
from rl.follower import FollowerPolicy  # noqa: E402


def _abs(p):
    m = p.replace("\\", "/")
    if m.startswith("/mnt/") and len(m) > 6:
        head, rest = m.split("/", 3)[2:]
        return head.upper() + ":\\" + rest.replace("/", "\\")
    return os.path.normpath(os.path.join(ORIG_CWD, p))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="runs/_fl_il_bc/train")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--files", type=int, default=20, help="只读前 N 个 bc_*.pkl（每文件 ~22 样本）")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--hidden", type=int, default=128)
    args = ap.parse_args(argv)
    args.data_dir = _abs(args.data_dir)
    torch.set_num_threads(args.threads)

    files = sorted(glob.glob(os.path.join(args.data_dir, "bc_*.pkl")))[:args.files]
    S = []
    for fn in files:
        with open(fn, "rb") as f:
            S.extend(pickle.load(f))
    if not S:
        print(f"[bench] {args.data_dir} 下没有 bc_*.pkl")
        return 2

    torch.manual_seed(0)
    np.random.seed(0)
    policy = FollowerPolicy(hidden=args.hidden, plan_dim=len(S[0][2]), belief_dim=len(S[0][1]))
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    perm = np.random.permutation(len(S))
    t0 = time.time()
    for i in range(args.steps):
        obs, tok, plan, bundle, masks = S[perm[i % len(S)]]
        lp, _, _, _ = policy.evaluate(obs, tok, plan, bundle, masks, hidden=None)
        loss = -lp
        opt.zero_grad()
        loss.backward()
        opt.step()
    dt = time.time() - t0
    print(f"[bench] threads={args.threads} samples={len(S)} steps={args.steps} "
          f"{dt:.2f}s → {dt / args.steps * 1000:.2f} ms/step", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
