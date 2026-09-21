# -*- coding: utf-8 -*-
"""**Stage 0 上界诊断**：act 决策的「信息增益上界」——H-A（优化/混比）vs H-B（表示不足）的判决者。

预注册：`docs/fl_il_il2_prereg_2026-09-22.md` §1–§2（判据跑前写死）。

问题：IL 策略塌成「永不出牌」（部署 0.3 次/局）。两种竞争机制：

* **H-A 优化/混比**：save 帧占 77.1%，共享 `slot_head` 的 STOP logit 被多数类推高 ⇒ 修混比。
* **H-B 表示不足**：act 决策所需信息（对手真值手牌/循环/圣水、双方「距下 1 点圣水还有几帧」）不在观测里
  ⇒「何时出牌」原则上不可判 ⇒ 最优预测器就是多数类 ⇒ 补输入。

做法：**同一批帧**、两组特征、各训一个小分类器（**不碰策略网络**）预测 `act ∈ {0,1}`，
比较 **ΔAUC = AUC(F_oracle) − AUC(F_obs)**：

* `F_obs`：人类回放里**可观测/可重建**的量（我圣水、时间、next_card、距下点帧数、信念后验、最近 3 手对手事件）
* `F_oracle`：`F_obs` + **真值**（对手圣水、对手手牌 4、循环 8、下一张、距下点帧数）
* `F_oracle_shuffled`：**同维度的打乱对照** ⇒ 排除「只是特征更多所以 AUC 更高」

纪律（预注册 §1.2）：**按局分组 5 折**（【R9】）+ **按局 bootstrap CI**；两族模型（线性 / MLP）**同向才算数**；
基线 = **只用「我圣水」的单变量 AUC**（【R15】本实验自己的量纲）。

用法（Windows venv）:
    .venv/Scripts/python.exe scripts/il_act_upper_bound.py \
        --frames-dir runs/_fl_il_frames --out docs/fl_il_2026-09-21/act_upper_bound.json
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src", "clasher_new")
ORIG_CWD = os.getcwd()
sys.path.insert(0, SRC)

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

import numpy as np  # noqa: E402
import torch  # noqa: E402

from card_utils import Card  # noqa: E402
from rl.observation import ENTITY_NAMES  # noqa: E402

torch.set_num_threads(4)
N_FOLD = 5
N_BOOT = 400


def _abs(p):
    m = p.replace("\\", "/")
    if m.startswith("/mnt/") and len(m) > 6:
        head, rest = m.split("/", 3)[2:]
        return head.upper() + ":\\" + rest.replace("/", "\\")
    return os.path.normpath(os.path.join(ORIG_CWD, p))


_COST = {}


def cost_of(idx):
    """卡下标 → 费用（未登记 / 空槽 → 0）。卡表只读一次。"""
    if idx is None or int(idx) < 0 or int(idx) >= len(ENTITY_NAMES):
        return 0.0
    key = int(idx)
    if key not in _COST:
        try:
            _COST[key] = float(Card(ENTITY_NAMES[key]).elixir)
        except Exception:  # noqa: BLE001
            _COST[key] = 0.0
    return _COST[key]


def load(frames_dir):
    files = sorted(glob.glob(os.path.join(frames_dir, "feat_*.npz")))
    if not files:
        raise SystemExit(f"[upper-bound] {frames_dir} 下没有 feat_*.npz")
    rows, game, tag = [], [], []
    for gi, f in enumerate(files):
        d = np.load(f, allow_pickle=False)
        n = len(d["frame"])
        rows.append(d)
        game.append(np.full(n, gi, dtype=np.int32))
        tag.append(str(d["tag"]))
    return files, rows, np.concatenate(game), tag


def pop_mask(rows):
    """预注册 §1.2 的帧总体：掩码放行 ≥1 个有合法落点槽的帧 ∪ 人类出牌的帧。"""
    out = []
    for d in rows:
        out.append(((d["opt_cell"] == 1) | (d["act"] == 1)).astype(bool))
    return out


def build_features(rows, keep):
    """→ (F_obs, F_oracle_extra, F_grid)，均为 (N, d) float32；`F_grid` 可能为 None。"""
    obs_parts, orc_parts, grid_parts = [], [], []
    for d, m in zip(rows, keep):
        ev_c, ev_x, ev_y, ev_dt = d["ev_c"][m], d["ev_x"][m], d["ev_y"][m], d["ev_dt"][m]
        ev_cost = np.vectorize(cost_of, otypes=[np.float32])(ev_c)
        b_hand, b_next = d["b_hand"][m], d["b_next"][m]
        obs = np.column_stack([
            d["my_elixir"][m], d["time"][m], d["next_card"][m], d["next_card_cost"][m],
            d["f2n_my"][m], d["b_elixir"][m], d["b_unc"][m],
            b_hand, b_next,
            ev_cost, ev_x, ev_y, ev_dt, ev_c.astype(np.float32),
        ]).astype(np.float32)
        oh, oc = d["opp_hand"][m], d["opp_cycle"][m]
        oh_cost = np.vectorize(cost_of, otypes=[np.float32])(oh)
        oc_cost = np.vectorize(cost_of, otypes=[np.float32])(oc)
        extra = np.column_stack([
            d["opp_elixir"][m], d["f2n_opp"][m],
            oc[:, 4].astype(np.float32), np.vectorize(cost_of, otypes=[np.float32])(oc[:, 4]),
            oh.astype(np.float32), oh_cost,
            oc.astype(np.float32), oc_cost,
        ]).astype(np.float32)
        obs_parts.append(obs)
        orc_parts.append(extra)
        if "grid_c" in d:
            grid_parts.append(np.asarray(d["grid_c"][m], dtype=np.float32))
    grids = np.vstack(grid_parts) if grid_parts else None
    return np.vstack(obs_parts), np.vstack(orc_parts), grids


def auc(y, s):
    """秩和（Mann-Whitney）AUC；y ∈ {0,1}，s 为分数（越大越倾向 1）。"""
    y = np.asarray(y)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return None
    r = np.empty(len(s), dtype=np.float64)
    order = np.argsort(s, kind="mergesort")
    sr = np.asarray(s, dtype=np.float64)[order]
    i = 0
    ranks = np.empty(len(s), dtype=np.float64)
    while i < len(sr):
        j = i
        while j + 1 < len(sr) and sr[j + 1] == sr[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def standardize(Xtr, Xte, min_std=0.0):
    """按**训练折**做标准化；`min_std>0` 时**丢掉近常量列**。

    ⚠️ 为什么需要这个开关：粗网格（4×4 均值池化）里大量「该区块无实体」的通道是**近常量**，
    标准差 ~1e-4 时会被标准化**放大 10^4 倍** ⇒ 噪声淹没信号（实测 MLP 反而掉 AUC）。
    默认 `min_std=0` ⇒ **不丢列**，已有读数（Stage 0 主表）逐位不变。
    """
    mu, sd = Xtr.mean(0), Xtr.std(0)
    keep = sd >= max(min_std, 1e-6)
    if keep.sum() == 0:
        keep = np.ones_like(sd, dtype=bool)
    mu, sd = mu[keep], sd[keep]
    return (Xtr[:, keep] - mu) / sd, (Xte[:, keep] - mu) / sd


def fit_torch(Xtr, ytr, kind, seed=0):
    """两族小模型：linear（logistic）/ mlp（64-隐层）。返回打分函数。"""
    torch.manual_seed(seed)
    d = Xtr.shape[1]
    if kind == "linear":
        net = torch.nn.Linear(d, 1)
        epochs, lr = 300, 0.05
    else:
        net = torch.nn.Sequential(torch.nn.Linear(d, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
        epochs, lr = 200, 0.01
    X = torch.as_tensor(Xtr, dtype=torch.float32)
    y = torch.as_tensor(ytr, dtype=torch.float32).view(-1, 1)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    lossf = torch.nn.BCEWithLogitsLoss()
    net.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(net(X), y)
        loss.backward()
        opt.step()
    net.eval()

    def score(Xte):
        with torch.no_grad():
            return net(torch.as_tensor(Xte, dtype=torch.float32)).view(-1).numpy()
    return score


def oof_scores(X, y, game, kind, seed=0, min_std=0.0):
    """按局分组 5 折 → 折外打分（OOF）。标准化在训练折上拟合（防泄漏）。"""
    gids = np.unique(game)
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(gids))
    fold_of = {int(g): i % N_FOLD for i, g in enumerate(gids[order])}
    fold = np.array([fold_of[int(g)] for g in game])
    oof = np.zeros(len(y), dtype=np.float64)
    for k in range(N_FOLD):
        te = fold == k
        tr = ~te
        if te.sum() == 0 or tr.sum() == 0:
            continue
        Xtr, Xte = standardize(X[tr], X[te], min_std=min_std)
        sc = fit_torch(Xtr, y[tr], kind, seed=seed + k)
        oof[te] = sc(Xte)
    return oof


def boot_ci(y, sa, sb, game, n=N_BOOT, seed=0):
    """按局 bootstrap：ΔAUC = AUC(a) − AUC(b) 的 CI。"""
    rng = np.random.RandomState(seed)
    gids = np.unique(game)
    deltas = []
    for _ in range(n):
        pick = rng.choice(gids, size=len(gids), replace=True)
        idx = np.concatenate([np.where(game == g)[0] for g in pick])
        aa, ab = auc(y[idx], sa[idx]), auc(y[idx], sb[idx])
        if aa is not None and ab is not None:
            deltas.append(aa - ab)
    if not deltas:
        return None, None
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def readout_curve(path):
    """从对局 runner 的录像取**模型侧**的「按圣水分箱出牌率」（J7.4 的对照曲线）。

    ⚠️ 只取**局数最多的那个文件**（runner 按 block 落盘且后一块**累计**——上一轮踩过重复计数）。
    """
    import pickle
    if os.path.isdir(path):
        cand = sorted(glob.glob(os.path.join(path, "league_*.pkl")),
                      key=lambda f: os.path.getsize(f))
        if not cand:
            return None
        path = cand[-1]
    with open(path, "rb") as f:
        d = pickle.load(f)
    bins = {}
    elix = []
    n_play = 0
    for g in d.get("games", []):
        for fr in g.get("frames", []):
            e = fr.get("elixir0")
            if e is None:
                continue
            elix.append(float(e))
            b = int(float(e))
            cell = bins.setdefault(b, [0, 0])
            cell[0] += 1
            if fr.get("bundle"):
                cell[1] += 1
                n_play += 1
    return {"replay": os.path.basename(path),
            "frames": int(sum(v[0] for v in bins.values())), "plays": int(n_play),
            "elixir_median": (float(np.median(elix)) if elix else None),
            "elixir_ge9_share": (float(np.mean(np.asarray(elix) >= 9)) if elix else None),
            "by_bin": {str(k): {"n": v[0], "plays": v[1], "act_rate": v[1] / v[0]}
                       for k, v in sorted(bins.items())}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-std", type=float, default=0.0,
                    help="标准化时丢掉训练折 std < 该值的近常量列（0 = 不丢；粗网格探针建议 1e-3）")
    ap.add_argument("--readout-replay", default=None,
                    help="对局 runner 的 run 目录（或 league_*.pkl）⇒ 附上**模型侧**分箱出牌率曲线（J7.4 对照）")
    args = ap.parse_args(argv)
    args.frames_dir = _abs(args.frames_dir)
    if args.out:
        args.out = _abs(args.out)

    files, rows, game, tag = load(args.frames_dir)
    keep = pop_mask(rows)
    keep_all = np.concatenate(keep)
    game = game[keep_all]          # ⚠️ 必须与帧总体同步过滤（第一版漏了 ⇒ 布尔索引长度不符）
    y = np.concatenate([d["act"][m] for d, m in zip(rows, keep)]).astype(np.int64)
    n_rows = int(len(y))
    base = float(y.mean())
    obs, extra, grid = build_features(rows, keep)
    rng = np.random.RandomState(args.seed)
    extra_shuf = extra[rng.permutation(len(extra))]

    elixir_only = obs[:, 0]                       # 单变量基线：只用「我圣水」
    sets = {
        "elixir_only": elixir_only,
        "F_obs": obs,
        "F_oracle": np.hstack([obs, extra]),
        "F_oracle_shuffled": np.hstack([obs, extra_shuf]),
    }
    if grid is not None:
        #: ★ Stage 0 尾项：粗网格（4×4 均值 → 8×5×15 = 600 维）⇒ 回答「act 能否从**棋盘**读出来」
        sets["F_obs+grid"] = np.hstack([obs, grid])
    res = {
        "frames_dir": os.path.basename(args.frames_dir), "games": len(files),
        "frame_rows_total": int(sum(len(d["frame"]) for d in rows)),
        "population_rows": n_rows, "base_rate_act": base,
        "n_fold": N_FOLD, "n_boot": N_BOOT, "min_std": args.min_std,
        "population_rule": "掩码放行 ≥1 个有合法落点的出牌槽 ∪ 人类出牌帧（预注册 §1.2）",
        "auc": {},
    }
    store = {}          # OOF 打分数组**不进 JSON**（ndarray 不可序列化；只报 AUC）
    for name, X in sets.items():
        X = np.asarray(X, dtype=np.float32)
        res["auc"][name] = {"elixir_only_single": None}
        if name == "elixir_only":
            res["auc"][name] = {"raw_single_feature": auc(y, X)}
            continue
        for kind in ("linear", "mlp"):
            sc = oof_scores(X, y, game, kind, seed=args.seed, min_std=args.min_std)
            res["auc"][name][kind] = auc(y, sc)
            store.setdefault(name, {})[kind] = sc

    # ΔAUC（配对：同折外样本、同 bootstrap 局）
    delta = {}
    for kind in ("linear", "mlp"):
        a = store["F_oracle"][kind]
        b = store["F_obs"][kind]
        lo, hi = boot_ci(y, a, b, game, seed=args.seed)
        delta[kind] = {"dAUC": float(auc(y, a) - auc(y, b)), "ci95": [lo, hi]}
    da = np.mean([delta["linear"]["dAUC"], delta["mlp"]["dAUC"]])
    verdict = ("H-B（表示不足 ⇒ 先做输入阶梯）" if da >= 0.05 else
               "H-A（优化/混比 ⇒ 先做混比）" if da < 0.03 else
               "边界带 ⇒ 预注册规定扩到 600 局复判")
    res["J7_1_delta_auc"] = {"per_family": delta, "mean_dAUC": float(da),
                             "gate": ">=0.05 ⇒ H-B ; <0.03 ⇒ H-A ; [0.03,0.07) ⇒ 扩样",
                             "verdict": verdict}

    if grid is not None:
        dg = {}
        for kind in ("linear", "mlp"):
            a = store["F_obs+grid"][kind]
            b = store["F_obs"][kind]
            lo, hi = boot_ci(y, a, b, game, seed=args.seed)
            dg[kind] = {"dAUC": float(auc(y, a) - auc(y, b)), "ci95": [lo, hi]}
        res["J7_1b_delta_auc_grid"] = {
            "grid_dim": int(grid.shape[1]), "per_family": dg,
            "mean_dAUC": float(np.mean([dg["linear"]["dAUC"], dg["mlp"]["dAUC"]])),
            "note": "粗网格（4×4 均值池化）增量 ⇒ 0.68 是「下界」还是「天花板」的判据",
        }

    # J7.4 分箱：人类出牌率 vs 我圣水（整数箱），只在「有得选」的帧上
    bins = {}
    for b in range(0, 12):
        sel = (obs[:, 0] >= b) & (obs[:, 0] < b + 1)
        n = int(sel.sum())
        if n:
            bins[str(b)] = {"n": n, "act_rate": float(y[sel].mean())}
    res["J7_4_human_act_rate_by_elixir"] = bins
    if args.readout_replay:
        res["J7_4_model_act_rate_by_elixir"] = readout_curve(_abs(args.readout_replay))
    res["monotone_note"] = ("人类出牌率 vs 我圣水：若低圣水区高、高圣水区也高而中段低 ⇒ 与「攒费等待」一致；"
                            "该表是模型侧曲线的靶子（H2 自强化陷阱的判定用）")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"[upper-bound] 局={len(files)} 总体帧={n_rows} act 基率={base:.4f}")
    print(f"  AUC elixir_only={auc(y, elixir_only):.4f}")
    for name in ("F_obs", "F_oracle", "F_oracle_shuffled"):
        print(f"  AUC {name:18s} linear={res['auc'][name]['linear']:.4f}  mlp={res['auc'][name]['mlp']:.4f}")
    for kind in ("linear", "mlp"):
        d = delta[kind]
        print(f"  ΔAUC(oracle−obs) {kind:6s} = {d['dAUC']:+.4f}  CI95={d['ci95']}")
    print(f"  J7.1 平均 ΔAUC = {da:+.4f} ⇒ {verdict}")
    if "J7_1b_delta_auc_grid" in res:
        g = res["J7_1b_delta_auc_grid"]
        print(f"  J7.1b 加粗网格（{g['grid_dim']} 维）ΔAUC：线性 {g['per_family']['linear']['dAUC']:+.4f} "
              f"MLP {g['per_family']['mlp']['dAUC']:+.4f} ⇒ 平均 {g['mean_dAUC']:+.4f}")
        print(f"    F_obs+grid AUC：线性 {res['auc']['F_obs+grid']['linear']:.4f} "
              f"MLP {res['auc']['F_obs+grid']['mlp']:.4f}")
    if res.get("J7_4_model_act_rate_by_elixir"):
        m = res["J7_4_model_act_rate_by_elixir"]
        top = {k: v for k, v in list(m["by_bin"].items())[-3:]}
        print(f"  J7.4 模型侧（{m['replay']}）：帧={m['frames']} 出牌={m['plays']} "
              f"圣水中位={m['elixir_median']} 圣水>=9 占比={m['elixir_ge9_share']:.3f} "
              f"| 高箱 {[(k, round(v['act_rate'], 4)) for k, v in top.items()]}")
    print("  J7.4 人类出牌率（按我圣水整数箱）: " +
          ", ".join(f"{k}:{v['act_rate']:.3f}(n={v['n']})" for k, v in list(bins.items())[:6]))
    if args.out:
        print(f"[upper-bound] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
