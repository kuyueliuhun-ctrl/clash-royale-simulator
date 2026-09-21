# -*- coding: utf-8 -*-
"""IL / BC **预算剂量-反应**仪器：固定数据与口径，只动 `epochs` × `lr`。

背景（预注册：`docs/fl_il_scaling_prereg_2026-09-20.md`）：
上游 3 epoch / 1e-3 的留出读数是「落点学到了、出牌选择没学到」（top-1 40.42% vs trivial 恒选槽 4 = 39.84%，
且槽 1/2/3 的 recall **低于随机初始化**）。用户的判断是「**IL 还不够，要更多轮的训练或者调高学习率**」。
本脚本把这个判断做成**单变量**实验：同一批样本、同一初值、同一 permutation 顺序，只改训练强度。

⚠️ **复刻纪律**：训练循环**逐行复刻** `rl/human_play.py::train_bc_from_human`（第 205-233 行）：
    np.random.seed(seed) → FollowerPolicy(...) → Adam(lr) → 每 epoch `np.random.permutation(len(samples))`
    → 逐样本 `policy.evaluate(obs, tok, plan, bundle, masks, hidden=None)` → `loss = -lp` → `backward/step`
唯一差别：样本**只加载一次**在内存里复用（原函数每调用一次重读 1.6 GB），且把逐 epoch 曲线与耗时落盘。
**不改** `human_play.py` / `follower.py` / 掩码任何一行。

等价性自检（预注册 §4 F0）：跑 `--configs 3:1e-3` 必须复现已提交 ckpt 的留出读数
（`nll == 5.596222589020527`、`top1 == 0.40419060493409936`）——不满足则复刻有语义差，结论作废。

用法（Windows venv；从仓库根跑）:
    ./.venv/Scripts/python.exe scripts/il_bc_sweep.py \
        --data-dir runs/_fl_il_bc/train --out-dir runs/_fl_il_bc/sweep \
        --configs 3:1e-3,10:1e-3,30:1e-3,10:3e-3,10:1e-2
"""
import argparse
import json
import os
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
from rl.follower import FollowerPolicy, save_checkpoint  # noqa: E402
from rl.human_play import load_bc_samples  # noqa: E402


def _abs(p):
    """相对路径按 ORIG_CWD 解析；WSL 的 `/mnt/e/...` 译成 `E:\\...`（本仓 Windows 侧踩过）。"""
    m = p.replace("\\", "/")
    if m.startswith("/mnt/") and len(m) > 6:
        head, rest = m.split("/", 3)[2:]
        return head.upper() + ":\\" + rest.replace("/", "\\")
    return os.path.normpath(os.path.join(ORIG_CWD, p))


def _lr_tag(lr):
    """1e-3 → '1e-3'（用于文件名，避免 '0.001' 与 '1e-3' 混写）。"""
    return f"{lr:g}"


def parse_configs(s):
    """'3:1e-3,10:3e-3' → [(3, 0.001), (10, 0.003)]"""
    out = []
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        e, lr = item.split(":")
        out.append((int(e), float(lr)))
    return out


def train_one(samples, epochs, lr, seed, out_path, tag_prefix=None, save_every=0,
              mix_ratio=None, decoupled_act=False):
    """复刻 human_play.py::train_bc_from_human 的训练循环（见模块 docstring 的复刻纪律）。

    `save_every=N>0` 时每 N 个 epoch 额外落一份快照 `<out_path 去后缀>_ep{k}.pt`：
    **不消耗任何 RNG**（只序列化参数）⇒ 不影响训练轨迹，却把「更多轮」做成**曲线**而不是单个终点。
    """
    torch.manual_seed(seed)
    np.random.seed(seed)                     # 与训练循环同序：在构造 policy 之前
    #: ★ 2026-09-21 **混比（mix-ratio）**：`mix_ratio >= 0` 时每个 epoch 只抽
    #: `round(mix_ratio × n_play)` 条 STOP/save 帧（与全部 play 帧拼成一个 epoch）。
    #: ⚠️ **不做逐样本 loss 加权**：本训练循环是 Adam + batch=1 ⇒ 更新 ≈ `lr·sign(g)`（尺度无关），
    #: 乘一个标量权重**不改变任何更新**（实测级结论，见 `docs/fl_il_il2_prereg_2026-09-22.md` §0）
    #: ⇒ 唯一有效的杠杆是「每类样本**出现的频率**」。默认 `-1` = 关 ⇒ 旧行为**逐位不变**。
    _play_idx = np.array([i for i, s in enumerate(samples) if s[3].sub_actions], dtype=np.int64)
    _stop_idx = np.array([i for i, s in enumerate(samples) if not s[3].sub_actions], dtype=np.int64)
    belief_dim = len(samples[0][1])
    plan_dim = len(samples[0][2])
    policy = FollowerPolicy(hidden=128, plan_dim=plan_dim, belief_dim=belief_dim,
                            decoupled_act=bool(decoupled_act))
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    n = len(samples)
    #: 每个 epoch 的索引池（混比只改这个池；`--mix-ratio` 关 = 全部样本）
    if mix_ratio is not None and mix_ratio >= 0 and len(_play_idx) and len(_stop_idx):
        k_stop = int(round(float(mix_ratio) * len(_play_idx)))
        k_stop = max(0, min(k_stop, len(_stop_idx)))
        pool = np.concatenate([_play_idx, _stop_idx[:k_stop]]) if k_stop else _play_idx.copy()
        pool = np.sort(pool)
    else:
        pool = np.arange(n, dtype=np.int64)
    n_ep = len(pool)
    curve = []
    snaps = []
    for ep in range(epochs):
        t0 = time.time()
        perm = pool[np.random.permutation(n_ep)]
        tot = 0.0
        for i in perm:
            obs, tok, plan, bundle, masks = samples[i]
            lp, _, _, _ = policy.evaluate(obs, tok, plan, bundle, masks, hidden=None)
            loss = -lp
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(lp.item())
        mean_lp = tot / max(1, n_ep)
        dt = time.time() - t0
        curve.append({"epoch": ep + 1, "mean_logprob": mean_lp, "sec": dt})
        print(f"[sweep] ep={epochs} lr={_lr_tag(lr)}  "
              f"epoch {ep + 1}/{epochs} mean_logprob={mean_lp:.3f}  {dt:.1f}s", flush=True)
        if save_every > 0 and (ep + 1) % save_every == 0:
            sp = f"{os.path.splitext(out_path)[0]}_ep{ep + 1}.pt"
            save_checkpoint(policy, sp)
            snaps.append({"epoch": ep + 1, "ckpt": os.path.basename(sp)})
            print(f"[sweep] snapshot {sp}", flush=True)
    save_checkpoint(policy, out_path)
    print(f"[sweep] saved {out_path}", flush=True)
    return {"epochs": epochs, "lr": lr, "seed": seed, "samples": n,
            "mix_ratio": (None if mix_ratio is None else float(mix_ratio)),
            "epoch_pool": int(n_ep),
            "n_play": int(len(_play_idx)), "n_stop": int(len(_stop_idx)),
            "ckpt": os.path.basename(out_path), "curve": curve, "snapshots": snaps,
            "total_sec": sum(c["sec"] for c in curve)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="训练集目录（平铺 bc_*.pkl）")
    ap.add_argument("--out-dir", required=True, help="ckpt + manifest 落盘目录")
    ap.add_argument("--configs", default="3:1e-3,10:1e-3,30:1e-3,10:3e-3,10:1e-2",
                    help="epochs:lr 逗号分隔；第一格应当是 3:1e-3（F0 等价性自检）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0, help="torch 线程数（0 = 不动，默认单线程口径）")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--mix-ratio", type=float, default=None,
                    help="每个 epoch 抽 round(R × n_play) 条 STOP/save 帧（R = play:save 的 save 侧倍数）；"
                         "缺省 = 关 = 旧行为逐位不变。见 docs/fl_il_il2_prereg_2026-09-22.md §0/§5")
    ap.add_argument("--save-every", type=int, default=0,
                    help="每 N 个 epoch 额外落一份快照（不消耗 RNG，不影响训练轨迹）")
    #: ★ 独立 act 头（预注册 `docs/fl_il_il2_prereg_2026-09-22.md` §7）：把「出不出牌」从
    #: 6 类共享头拆成 `act_head`(2) + `slot_head`(5)，**联合训练**（离线拼接已被 C23 证否）。
    #: 缺省 = 关 = 旧行为**逐位不变**。架构变更 ⇒ 该臂必须是 `--fresh`（本来就是新建网络）。
    ap.add_argument("--decoupled-act", action="store_true",
                    help="启用独立 act 头（§7）；需配 --out-dir 形如 sweep_mixR<tag>")
    args = ap.parse_args(argv)

    args.data_dir = _abs(args.data_dir)
    args.out_dir = _abs(args.out_dir)
    if args.manifest:
        args.manifest = _abs(args.manifest)
    os.makedirs(args.out_dir, exist_ok=True)
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    t_load = time.time()
    samples = load_bc_samples(args.data_dir)
    print(f"[sweep] {len(samples)} 样本，加载 {time.time() - t_load:.1f}s；"
          f"threads={torch.get_num_threads()}", flush=True)
    if not samples:
        print(f"[sweep] {args.data_dir} 下没有 bc_*.pkl")
        return 2

    cfgs = parse_configs(args.configs)
    if cfgs and cfgs[0] != (3, 1e-3):
        print("[sweep] ⚠️ 第一格不是 3:1e-3 —— 预注册 §4 的 F0 等价性自检将无法执行", flush=True)

    runs = []
    for epochs, lr in cfgs:
        out_path = os.path.join(args.out_dir, f"bc_fl_e{epochs}_lr{_lr_tag(lr)}.pt")
        print(f"[sweep] === epochs={epochs} lr={_lr_tag(lr)} → {out_path} ===", flush=True)
        runs.append(train_one(samples, epochs, lr, args.seed, out_path,
                              save_every=args.save_every,
                              mix_ratio=args.mix_ratio,
                              decoupled_act=args.decoupled_act))

    manifest = {"data_dir": args.data_dir, "seed": args.seed, "runs": runs,
                "decoupled_act": bool(args.decoupled_act),
                "note": "训练循环逐行复刻 rl/human_play.py::train_bc_from_human；"
                        "每格同 seed ⇒ 同初值 + 同 permutation，唯一变量 = epochs/lr（或 mix_ratio）"}
    mpath = args.manifest or os.path.join(args.out_dir, "sweep_manifest.json")
    with open(mpath, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"[sweep] manifest → {mpath}", flush=True)
    for r in runs:
        print(f"[sweep] e={r['epochs']:>3} lr={_lr_tag(r['lr']):>5}  "
              f"last mean_logprob={r['curve'][-1]['mean_logprob']:.3f}  "
              f"{r['total_sec']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
