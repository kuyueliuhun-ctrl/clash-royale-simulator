# -*- coding: utf-8 -*-
"""**描述性归因仪器**（预注册 `docs/il_whiff_handscore_prereg_2026-09-22.md` §6.8）：
训练后的策略**到底有没有在用**那 17 维手牌/信息分？

为什么需要（主判据 J-W2.3 只给**总效应**，不给**归因**）：
若 J-W2.3 判无收益，有两种**完全不同**的解释，出路也完全不同：
  (a) 特征里没有增量信息（belief token 的 `hand_probs` + scalar 的 `elixir` 已经把信息给全了）；
  (b) 信息在，但**没被学进去**（训练强度/优化问题）⇒ 该加轮数或改注入位置。
本仪器把两者分开：**在训练好的网上扰动那 17 列**，看 NLL 动不动。

三种干预（同一批帧、逐帧配对）：
  * `base`   ：原样；
  * `zero`   ：把 17 列置零（**保留**边际形状、破坏全部信息）；
  * `perm`   ：把 17 列**整块跨帧打乱** n 次（**保留边际分布**、只破坏"帧 ↔ 特征"的对应）
               ⇒ 这是"特征值本身有用吗"的**干净**检验（`zero` 会同时改输入尺度，可能被
               `plan_mlp` 的偏置吸收，故单独看 `zero` 会低估使用度）。

判读（**描述性，非门禁**；R10）：
  ΔNLL(perm) ≈ 0 ⇒ 模型**没用**这些列（解释 (b) 或该列冗余）；
  ΔNLL(perm) 明显 > 逐帧配对的噪声尺度 ⇒ 模型**在用**。

用法（仓库根）：
    ./.venv/Scripts/python.exe scripts/il_hs_feature_use.py \
        --ckpt runs/_fl_il_bc_hs/sweep_mixHS00/s0/bc_fl_e3_lr0.001.pt \
        --data-dir runs/_fl_il_bc_hs/holdout --limit 4000 \
        --out docs/fl_il_2026-09-21/hs_feature_use_HS00_s0.json
"""
import argparse
import glob
import json
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src", "clasher_new")
sys.path.insert(0, SRC)
#: ⚠️ 必须在 `chdir` **之前**记下调用者的 cwd：`_abs()` 用它解析相对路径。
#: 首版在 `os.chdir(SRC)` 之后才 `os.getcwd()` ⇒ 相对 ckpt 路径被拼成
#: `<repo>/src/clasher_new/runs/...` ⇒ `FileNotFoundError`（本批实测踩到）。
ORIG_CWD = os.getcwd()
os.chdir(SRC)

from rl.io_bootstrap import force_utf8_stdout  # noqa: E402

force_utf8_stdout()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rl.follower import load_checkpoint  # noqa: E402
from rl.hand_score import PLAN_EXTRA_DIM  # noqa: E402
from rl.plan_space import PLAN_BASE_DIM  # noqa: E402


def _abs(p):
    m = p.replace("\\", "/")
    if os.path.isabs(m) and not m.startswith("/mnt/"):
        return m                       # 已是绝对路径（Windows 风格），原样用
    if m.startswith("/mnt/") and len(m) > 6:
        head, rest = m.split("/", 3)[2:]
        return head.upper() + ":\\" + rest.replace("/", "\\")
    return os.path.normpath(os.path.join(ORIG_CWD, p))


def load_samples(data_dir, limit=0):
    out = []
    for fn in sorted(glob.glob(os.path.join(data_dir, "bc_*.pkl"))):
        with open(fn, "rb") as f:
            out.extend(pickle.load(f))
        if limit and len(out) >= limit:
            break
    return out[:limit] if limit else out


def mean_lp(pol, samples, extra_plans=None):
    """返回 (逐帧 lp 列表, 是否为 play 帧的布尔列表)。"""
    lps, is_play = [], []
    for i, s in enumerate(samples):
        obs, tok, plan, bundle, masks = s
        p = plan if extra_plans is None else extra_plans[i]
        lp, _v, _h, _m = pol.evaluate(obs, tok, p, bundle, masks, hidden=None)
        lps.append(float(lp.item() if hasattr(lp, "item") else lp))
        is_play.append(bool(bundle.sub_actions))
    return np.asarray(lps), np.asarray(is_play)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--limit", type=int, default=4000)
    ap.add_argument("--n-perm", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true",
                    help="即使 ckpt 是抹零臂也照跑 —— **负对照**：`plan_extras_zero>0` 的臂"
                         "前向已把尾列置零 ⇒ `ΔNLL(perm)`/`ΔNLL(zero)` 必须**恰好为 0**。"
                         "这条对照用来证明仪器本身不是在测「任意扰动都会变差」")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    data_dir = _abs(args.data_dir)
    samples = load_samples(data_dir, args.limit)
    pol = load_checkpoint(_abs(args.ckpt))
    print(f"[use] {len(samples)} 帧；ckpt plan_dim={pol.plan_dim} "
          f"plan_extras_zero={pol.plan_extras_zero}")
    if int(pol.plan_dim) != PLAN_BASE_DIM + PLAN_EXTRA_DIM:
        print(f"[use] ⚠️ plan_dim={pol.plan_dim} 不是 {PLAN_BASE_DIM}+{PLAN_EXTRA_DIM} "
              "⇒ 本仪器不适用（该 ckpt 没有追加特征）")
        return 2
    if int(pol.plan_extras_zero) != 0 and not args.force:
        print(f"[use] ⚠️ 该 ckpt 的 plan_extras_zero={pol.plan_extras_zero}"
              "（前向已把尾列抹零）⇒ 扰动无意义，本仪器只对**全活臂**适用"
              "（要跑负对照加 --force）")
        return 2
    if int(pol.plan_extras_zero) != 0:
        print(f"[use] ★ 负对照模式：plan_extras_zero={pol.plan_extras_zero} ⇒ "
              "ΔNLL(perm)/ΔNLL(zero) 必须**恰好 0**")

    plans = [np.asarray(s[2], dtype=np.float32) for s in samples]
    base_lp, is_play = mean_lp(pol, samples)
    zero_plans = []
    for p in plans:
        q = p.copy()
        q[PLAN_BASE_DIM:] = 0.0
        zero_plans.append(q)
    zero_lp, _ = mean_lp(pol, samples, zero_plans)

    rng = np.random.RandomState(args.seed)
    perm_lps = []
    for t in range(max(1, args.n_perm)):
        order = rng.permutation(len(plans))
        perm_plans = []
        for i, p in enumerate(plans):
            q = p.copy()
            q[PLAN_BASE_DIM:] = plans[order[i]][PLAN_BASE_DIM:]   # 整块跨帧打乱
            perm_plans.append(q)
        lp_t, _ = mean_lp(pol, samples, perm_plans)
        perm_lps.append(lp_t)
        print(f"[use] perm {t + 1}/{args.n_perm}  mean_lp={lp_t.mean():.6f}")
    perm_lp = np.mean(perm_lps, axis=0)

    def nll(x):
        return -float(np.mean(x))

    def nll_sub(x, m):
        return -float(np.mean(x[m])) if m.any() else None

    res = {
        "ckpt": args.ckpt, "n_frames": int(len(samples)),
        "n_play": int(is_play.sum()), "n_stop": int((~is_play).sum()),
        "n_perm": int(args.n_perm),
        "nll_base": nll(base_lp), "nll_zero": nll(zero_lp), "nll_perm": nll(perm_lp),
        "d_nll_zero": nll(zero_lp) - nll(base_lp),
        "d_nll_perm": nll(perm_lp) - nll(base_lp),
        "d_nll_perm_play": (nll_sub(perm_lp, is_play) - nll_sub(base_lp, is_play)
                            if is_play.any() else None),
        "d_nll_perm_stop": (nll_sub(perm_lp, ~is_play) - nll_sub(base_lp, ~is_play)
                            if (~is_play).any() else None),
        #: 逐帧配对的**噪声尺度**：perm 各次之间的逐帧标准差 / √n（同一批帧的重复扰动）
        "perm_lp_std_across_shuffles": float(np.mean(np.std(np.asarray(perm_lps), axis=0))),
        "note": ("描述性（非门禁）。ΔNLL(perm) ≈ 0 ⇒ 模型没用这 17 列；"
                 "ΔNLL(perm) 明显大于 perm 抖动尺度 ⇒ 在用。"),
    }
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if args.out:
        os.makedirs(os.path.dirname(_abs(args.out)), exist_ok=True)
        with open(_abs(args.out), "w", encoding="utf-8", newline="\n") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        print(f"[use] → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
