# -*- coding: utf-8 -*-
"""诊断：**「打一两轮就挂机到结束」是 IL 数据/管线的问题，还是「隐状态携带口径」的问题？**

用户 2026-09-20 追问（看 8702 = `--hidden carry` 臂）：「总是在 1 轮对牌后就开始挂机到结束」。
本仪器把该现象**从 env / dashboard / 我的对局 runner 里彻底剥离**再复现：

* 用**留出集里真实对局的帧序列**（`runs/_fl_il_bc/holdout/bc_XXXX.pkl`，每文件 = 一局，帧按时间序）；
* 喂给策略的是样本里**存好的** `belief_tok` / `plan_vec`（= BC 训练时用的同一批输入）；
* **只改一件事**：隐状态是否跨帧携带。
  - `none`      = 逐帧（= `human_play.py:224` 的训练口径）
  - `carry`     = 跨帧携带（= 生产约定，`FollowerOpponent` 的做法）
  - `carry_r4`  = 携带但每 4 帧清零（剂量对照：吸引子是否有记忆）

判据（工具自带，不预设结论）：
* 若 `none` 正常出牌而 `carry` 塌成 STOP ⇒ 崩溃**只**由携带隐状态引起，与数据/引擎/runner 无关；
* 若 `none` 也塌 ⇒ 问题在数据/标签侧。

另附**独立复算**：直接从 arm C 自己的录像（schema 5，`frames[].bundle`）数 p0 出牌次数与
「每局最后一次出牌」的相对位置（**不复用 runner 的计数器**）。

用法（Windows venv，从仓库根跑）:
    ./.venv/Scripts/python.exe scripts/il_carry_diag.py --ckpt runs/_fl_il_bc/bc_fl.pt --games 30
"""
import collections
import glob
import os
import pickle
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src", "clasher_new")
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(SRC)

from rl.io_bootstrap import force_utf8_stdout  # noqa: E402

force_utf8_stdout()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import il_eval_holdout as ie  # noqa: E402
from rl.follower import load_checkpoint  # noqa: E402

import argparse  # noqa: E402


def opt_of(bundle):
    """返回 'STOP' / 'ABILITY' / 0..3（槽位，0-based）。"""
    if not bundle.sub_actions:
        return "STOP"
    sa = bundle.sub_actions[0]
    if getattr(sa, "kind", "deploy") == "ability":
        return "ABILITY"
    return int(sa.slot) - 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "runs/_fl_il_bc/bc_fl.pt"))
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--arm-c", default=os.path.join(ROOT, "runs/il_readout_carry/replays"))
    args = ap.parse_args(argv)
    CKPT, N_GAMES = args.ckpt, args.games
    files = sorted(glob.glob(os.path.join(ROOT, "runs/_fl_il_bc/holdout/bc_*.pkl")))[:N_GAMES]
    policy = load_checkpoint(CKPT)
    stats = {m: collections.Counter() for m in ("none", "carry", "carry_r4")}
    first_stop = {m: [] for m in stats}
    hnorm = []
    for fn in files:
        S = ie.load_dir(os.path.dirname(fn)) if False else None
        import pickle
        with open(fn, "rb") as f:
            S = pickle.load(f)
        for mode in stats:
            h = None
            fstop = None
            for i, (obs, tok, plan, bundle, masks) in enumerate(S):
                gm = ie.make_get_mask(masks)
                b, _lp, _v, h_out, _mk = policy.act(obs, tok, plan, gm, hidden=h, deterministic=True)
                o = opt_of(b)
                stats[mode][o] += 1
                if o == "STOP" and fstop is None:
                    fstop = i
                if mode == "carry":
                    hnorm.append(float(h_out.float().norm().item()))
                h = None if mode == "none" else h_out
                if mode == "carry_r4" and (i + 1) % 4 == 0:
                    h = None
            if len(S):
                first_stop[mode].append((fstop if fstop is not None else len(S)) / len(S))
    print(f"[diag] {len(files)} 局 / 每局帧数≈{stats['none'].total() // max(1, len(files))}；ckpt={os.path.basename(CKPT)}")
    for mode in ("none", "carry", "carry_r4"):
        c = stats[mode]
        n = c.total()
        plays = sum(v for k, v in c.items() if isinstance(k, int))
        print(f"  hidden={mode:<10} 帧={n}  出牌={plays} ({plays/n:.4f}/帧)  "
              f"STOP={c['STOP']} ({c['STOP']/n:.4f})  槽分布="
              + " ".join(f"槽{k+1}:{c[k]}" for k in range(4)))
    #: 每局「第一次 STOP」的相对位置（占该局帧数比例，**逐局**统计）
    for mode in ("none", "carry"):
        v = np.array(first_stop[mode], dtype=float)
        print(f"  hidden={mode:<6} 每局「第一次 STOP」相对位置：中位 {np.median(v):.3f} "
              f"（n={len(v)} 局；1.000 = 全程没 STOP）")
    if hnorm:
        hn = np.array(hnorm)
        print(f"  carry 下 ‖h‖：中位 {np.median(hn):.3f}  最小 {hn.min():.3f}  最大 {hn.max():.3f}")

    # ---- 独立复算：arm C 录像里 p0 出牌次数与最后一次出牌帧号 ----
    #: ⚠️ runner 按 `--block` 逐块落盘，且**后一块的文件是累计的** ⇒ 直接遍历所有文件会**重复计数**。
    #: 只取「局数最多」的那一个文件（并列时取最大者），并在输出里标明用的是哪个。
    _all = sorted(glob.glob(os.path.join(args.arm_c, "*.pkl")))
    reps = []
    if _all:
        def _ng(p):
            with open(p, "rb") as f:
                return len(pickle.load(f).get("games", []))
        reps = [max(_all, key=lambda p: (_ng(p), os.path.getsize(p)))]
    if reps:
        with open(reps[0], "rb") as f:
            r0 = pickle.load(f)
        print(f"[diag] arm C 录像（只取累计最全的 1 个文件，避免重复计数）："
              f"{os.path.basename(reps[0])}（{len(r0.get('games', []))} 局）")
        total_play, total_frames, last_play, games_n = 0, 0, [], 0
        for p in reps:
            with open(p, "rb") as f:
                dd = pickle.load(f)
            for g in dd.get("games", []):
                frames = g.get("frames") or []
                if not frames:
                    continue
                games_n += 1
                total_frames += len(frames)
                idx = [i for i, fr in enumerate(frames) if (fr.get("bundle") or [])]
                total_play += len(idx)
                last_play.append((len(frames), idx[-1] if idx else 0))
        print(f"[diag] 独立复算（{os.path.basename(args.arm_c)}）：{games_n} 局 / {total_frames} 帧、"
              f"p0 出牌 {total_play} 次（{total_play/max(1,total_frames):.4f}/帧、局均 {total_play/max(1,games_n):.2f} 次）")
        if last_play:
            fr = np.array([a for a, _ in last_play], dtype=float)
            lp = np.array([b for _, b in last_play], dtype=float)
            print(f"[diag] 每局「最后一次出牌」的相对位置：中位 {np.median(lp/fr):.3f}（0 = 开局即停）")


if __name__ == "__main__":
    main()
