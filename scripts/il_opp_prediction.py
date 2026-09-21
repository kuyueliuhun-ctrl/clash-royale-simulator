# -*- coding: utf-8 -*-
"""**Stage 0 对手预判仪器**：现有（规则）对手模型到底准不准 —— 学习化对手模型的判据基线。

预注册：`docs/fl_il_il2_prereg_2026-09-22.md` §2 J7.3（门禁）与 §3 J9.1。
★ 2026-09-21 口径更正（第一版把 `next_probs` 拿去比「实际打出的牌」，那**不是**它的目标）：
`belief.next_probs` 的语义是「每张卡是**对手下一张进手牌（cycle[4]）**的概率」
（`rl/bayes_filter.py:167-175`，锁定流下是 **0/1 的 one-hot**）；
而「**对手下一次会打出哪张**」要从**手牌后验** `hand_probs`（cycle[:4]，和为 4）里读。

四组读数（真值来自回放里对手的**实际出牌**与**实际循环**，不是信念自己的输出）：

1. **下一张进手牌**：`next_probs` vs `opp_cycle[:,4]` ⇒ top-1 / NLL / 「锁定率」（p=1 的占比）。
2. **下一次打出的牌**：`hand_probs`（归一化后）vs 实际打出的牌 ⇒ top-1 / NLL；基线 = 均匀 1/8。
3. **手牌后验校准**：`hand_probs` 对「该牌此刻是否在对手手牌（`opp_cycle[:4]`）」的 Brier / ECE。
4. **对手 k 帧内会不会出牌**：二分类 AUC（k = 4/8/17），特征 = 信念可用量 ⇒ 「预判时机」是否可学。

⚠️ 对手出牌时刻由 `opp_play_count` 的**增量**给出（dump 在本帧 `belief.update` **之前**）
⇒ 帧 `i` 打出的牌 = 帧 `i+1` 的最新事件卡（见 `fl_il_to_bc.py` 循环末端）。

用法（Windows venv；需含 `opp_deck` 的 `--dump-frames` 产物）:
    .venv/Scripts/python.exe scripts/il_opp_prediction.py \
        --frames-dir runs/_fl_il_frames --out docs/fl_il_2026-09-21/opp_prediction.json
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

torch.set_num_threads(4)
KS = (4, 8, 17)
N_FOLD = 5


def _abs(p):
    m = p.replace("\\", "/")
    if m.startswith("/mnt/") and len(m) > 6:
        head, rest = m.split("/", 3)[2:]
        return head.upper() + ":\\" + rest.replace("/", "\\")
    return os.path.normpath(os.path.join(ORIG_CWD, p))


def auc(y, s):
    y = np.asarray(y)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return None
    order = np.argsort(np.asarray(s, dtype=np.float64), kind="mergesort")
    sr = np.asarray(s, dtype=np.float64)[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(sr):
        j = i
        while j + 1 < len(sr) and sr[j + 1] == sr[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def play_timeline(d):
    """→ (帧号数组, 该帧打出的卡下标数组)。用 `opp_play_count` 增量 + 下一帧的最新事件卡。"""
    cnt = d["opp_play_count"]
    inc = np.where(np.diff(cnt) > 0)[0] + 1
    frames, cards = [], []
    for i in inc:
        if i + 1 < len(cnt):
            cards.append(int(d["ev_c"][i + 1][-1]))
            frames.append(int(i))
    return np.asarray(frames, dtype=np.int64), np.asarray(cards, dtype=np.int64)


def fit_logit(Xtr, ytr, seed=0, epochs=250, lr=0.05):
    torch.manual_seed(seed)
    net = torch.nn.Linear(Xtr.shape[1], 1)
    X = torch.as_tensor(Xtr, dtype=torch.float32)
    y = torch.as_tensor(ytr, dtype=torch.float32).view(-1, 1)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    lf = torch.nn.BCEWithLogitsLoss()
    net.train()
    for _ in range(epochs):
        opt.zero_grad()
        lf(net(X), y).backward()
        opt.step()
    net.eval()
    return lambda Z: net(torch.as_tensor(Z, dtype=torch.float32)).view(-1).detach().numpy()


def oof(X, y, game, seed=0):
    gids = np.unique(game)
    rng = np.random.RandomState(seed)
    fold_of = {int(g): i % N_FOLD for i, g in enumerate(gids[rng.permutation(len(gids))])}
    fold = np.array([fold_of[int(g)] for g in game])
    out = np.zeros(len(y))
    for k in range(N_FOLD):
        te, tr = fold == k, fold != k
        if te.sum() == 0 or tr.sum() == 0:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd = np.where(sd < 1e-6, 1.0, sd)
        sc = fit_logit((X[tr] - mu) / sd, y[tr], seed=seed + k)
        out[te] = sc((X[te] - mu) / sd)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    args.frames_dir = _abs(args.frames_dir)
    if args.out:
        args.out = _abs(args.out)

    files = sorted(glob.glob(os.path.join(args.frames_dir, "feat_*.npz")))
    if not files:
        raise SystemExit(f"[opp-pred] {args.frames_dir} 下没有 feat_*.npz")
    if "opp_deck" not in np.load(files[0], allow_pickle=False):
        raise SystemExit("[opp-pred] npz 缺 `opp_deck`：请用带该字段的 `--dump-frames` 重跑")

    n1a = h1a = lock1a = 0
    nll1a = []
    n1b = h1b = 0
    nll1b = []
    brier = []
    cal = {}
    freq = {}
    haz = {k: {"X": [], "y": [], "g": []} for k in KS}
    n_games = 0
    for gi, f in enumerate(files):
        d = np.load(f, allow_pickle=False)
        n_games += 1
        deck = d["opp_deck"]
        pos_of = {int(c): k for k, c in enumerate(deck)}
        hand, nxt = d["b_hand"], d["b_next"]
        frames, cards = play_timeline(d)
        for c in cards:
            freq[int(c)] = freq.get(int(c), 0) + 1

        # 1) 下一张**进手**牌 = cycle[4]（`next_probs` 的设计目标）
        for i in range(len(d["frame"])):
            k = pos_of.get(int(d["opp_cycle"][i][4]))
            if k is None:
                continue
            p = nxt[i]
            n1a += 1
            h1a += int(int(np.argmax(p)) == k)
            lock1a += int(p[k] >= 0.999)
            nll1a.append(-float(np.log(max(p[k], 1e-9))))

        # 2) 下一次**打出**的牌 —— 从手牌后验读（归一化）
        for fi, ci in zip(frames, cards):
            k = pos_of.get(int(ci))
            if k is None:
                continue
            p = hand[max(0, fi - 1)].astype(np.float64)
            n1b += 1
            h1b += int(int(np.argmax(p)) == k)
            s = p.sum()
            nll1b.append(-float(np.log(max(p[k] / s, 1e-9))) if s > 0 else None)

        # 3) 手牌后验校准（对「是否在手牌」）
        inhand = np.zeros((len(d["frame"]), len(deck)), dtype=np.float32)
        for i in range(len(d["frame"])):
            inhand[i] = np.isin(deck, d["opp_cycle"][i][:4]).astype(np.float32)
        brier.append(float(np.mean((hand - inhand) ** 2)))
        #: ⚠️ 必须含 p=1.0 箱（`np.arange(0,1.0,0.1)` 会漏掉锁定流的 one-hot 正例——第一版漏了）
        for lo in np.arange(0, 1.01, 0.1):
            hi = lo + 0.1
            sel = (hand >= lo) & (hand < hi) if hi < 1.0 else (hand >= lo - 1e-6)
            if sel.sum() >= 200:
                cal.setdefault(round(float(lo), 1), []).append(
                    (int(sel.sum()), float(inhand[sel].mean()), float(hand[sel].mean())))

        # 4) k 帧内会不会出牌
        for k in KS:
            lab = np.zeros(len(d["frame"]), dtype=np.float32)
            for fi_ in frames:
                lab[max(0, fi_ - k):fi_] = 1.0
            X = np.column_stack([nxt, hand, d["b_elixir"], d["b_unc"], d["time"]] +
                                [d["ev_dt"][:, i] for i in range(d["ev_dt"].shape[1])])
            haz[k]["X"].append(X)
            haz[k]["y"].append(lab)
            haz[k]["g"].append(np.full(len(lab), gi, dtype=np.int64))

    maj = max(freq, key=freq.get) if freq else -1
    nll1b_v = [v for v in nll1b if v is not None]
    res = {
        "frames_dir": os.path.basename(args.frames_dir), "games": n_games,
        "J7_3a_next_into_hand": {
            "n": n1a, "top1": (h1a / n1a) if n1a else None,
            "nll": (float(np.mean(nll1a)) if nll1a else None),
            "locked_share_p_eq_1": (lock1a / n1a) if n1a else None,
            "target": "opp_cycle[:,4]（下一张进手牌）—— `next_probs` 的设计目标",
            "gate": "规则 filter 必须 > 均匀 1/8 = 0.125；学习化对手头（J9.1）必须超过规则 filter",
        },
        "J7_3b_next_played": {
            "n": n1b, "top1": (h1b / n1b) if n1b else None,
            "nll": (float(np.mean(nll1b_v)) if nll1b_v else None),
            "uniform_top1": 0.125, "uniform_nll": float(np.log(8.0)),
            "target": "对手**实际打出的下一张牌**（来自手牌后验 hand_probs）",
            "majority_card_global": {"card_idx": int(maj), "n_plays": int(freq.get(maj, 0))},
        },
        "J9_1_hand_posterior": {
            "brier_mean": (float(np.mean(brier)) if brier else None),
            "calibration_bins": {str(k): {"n": sum(x[0] for x in v),
                                          "emp_rate": float(np.mean([x[1] for x in v])),
                                          "pred_mean": float(np.mean([x[2] for x in v]))}
                                 for k, v in cal.items()},
            "brier_constant_half": 0.25,
            "note": "「是否在手牌」的先验 = 4/8 = 0.5 ⇒ 常数 0.5 预测的 Brier = 0.25；emp_rate 与 pred_mean 越近越校准",
        },
        "J9_1_hazard": {},
    }
    for k in KS:
        X = np.vstack(haz[k]["X"])
        y = np.concatenate(haz[k]["y"])
        g = np.concatenate(haz[k]["g"])
        if y.min() == y.max():
            res["J9_1_hazard"][str(k)] = {"base_rate": float(y.mean()), "auc_logit": None}
            continue
        sc = oof(X, y, g)
        res["J9_1_hazard"][str(k)] = {"n": int(len(y)), "base_rate": float(y.mean()),
                                      "auc_logit": auc(y, sc), "auc_chance": 0.5}
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    a, b = res["J7_3a_next_into_hand"], res["J7_3b_next_played"]
    print(f"[opp-pred] 局={n_games}")
    print(f"  1) 下一张**进手**牌(cycle[4]) top-1={a['top1']} NLL={a['nll']} 锁定率={a['locked_share_p_eq_1']} (n={a['n']})")
    print(f"  2) 下一次**打出**的牌 top-1={b['top1']} (均匀 0.125) NLL={b['nll']} (n={b['n']})")
    print(f"  3) 手牌后验 Brier={res['J9_1_hand_posterior']['brier_mean']}")
    for k in KS:
        h = res["J9_1_hazard"][str(k)]
        print(f"  4) {k} 帧内出牌 AUC={h.get('auc_logit')} (基率 {h['base_rate']:.3f})")
    if args.out:
        print(f"[opp-pred] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
