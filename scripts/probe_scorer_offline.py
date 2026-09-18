# -*- coding: utf-8 -*-
""""训练一个打分器给每帧打分"的离线可行性探针（只读）。

问题（用户 2026-09-14）：能不能换一种给每帧分的方式——**训练一个打分器**，按局面
（场上卡牌/手内圣水/手内卡牌/敌方已知卡牌）给每个决策帧打分？

本探针只回答其中**可证伪**的那一半：**"打分器"在可在线口径下能不能学到东西**。
（另一半是等价性论证：potential 形式的打分器 = 基线层，见 docs 说明。）

协议（与 probe_credit_baseline 的"因果滚动"同族，但用前向链式分块 CV + 早停）：
  把 138 个局按时间切成 F 块；第 i 折只用**它之前的块**训练、预测**当前块**——
  这是训练循环唯一可能的姿势（看不见未来）。
对照：
  · 线性岭回归（同折同协议）—— 已知读数
  · 常数基线（EV_within 按定义 = 0.0000 = "什么都不做"）
  · 阳性对照：同块内拟合（in-sample，应显著 >0）
  · 阴性对照：标签打散（应 ≤0）

用法（仓库根，CPU，~2-4 min）：
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/probe_scorer_offline.py \
        --npz src/clasher_new/runs/_probe_cache_v2_stoch.npz
"""
from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src", "clasher_new"))
from rl.io_bootstrap import force_utf8_stdout  # noqa: E402  (T1-1: UTF-8 兜底单一来源)
force_utf8_stdout()

import numpy as np
import torch
import torch.nn as nn


def ev_within(pred, y, ep):
    p = np.asarray(pred, dtype=np.float64).copy()
    yy = np.asarray(y, dtype=np.float64).copy()
    for e in np.unique(ep):
        m = (ep == e)
        p[m] -= p[m].mean()
        yy[m] -= yy[m].mean()
    var = float(np.var(yy))
    if var <= 1e-12:
        return None
    return 1.0 - float(np.mean((p - yy) ** 2)) / var


def ridge(Xtr, ytr, Xte, alpha, sd):
    xm, ym = Xtr.mean(0), float(ytr.mean())
    Xc = (Xtr - xm) / sd
    d = Xc.shape[1]
    A = Xc.T @ Xc + alpha * np.eye(d)
    w = np.linalg.solve(A, Xc.T @ (ytr - ym))
    return ((Xte - xm) / sd) @ w + ym


def mlp_fit_predict(Xtr, ytr, Xte, *, hidden=64, epochs=60, lr=1e-3, seed=0, sd=None,
                    val_frac=0.15):
    torch.manual_seed(seed)
    xm, ym = Xtr.mean(0), float(ytr.mean())
    ys = float(ytr.std()) or 1.0
    Xn = torch.tensor((Xtr - xm) / sd, dtype=torch.float32)
    yn = torch.tensor((ytr - ym) / ys, dtype=torch.float32).unsqueeze(1)
    n = len(Xn)
    nv = max(1, int(n * val_frac))
    Xa, Ya, Xv, Yv = Xn[:-nv], yn[:-nv], Xn[-nv:], yn[-nv:]
    net = nn.Sequential(nn.Linear(Xn.shape[1], hidden), nn.ReLU(),
                        nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    best, best_state = float("inf"), None
    for _ in range(epochs):
        net.train()
        opt.zero_grad()
        loss = nn.functional.mse_loss(net(Xa), Ya)
        loss.backward()
        opt.step()
        net.eval()
        with torch.no_grad():
            v = float(nn.functional.mse_loss(net(Xv), Yv))
        if v < best:
            best = v
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        p = net(torch.tensor((Xte - xm) / sd, dtype=torch.float32)).squeeze(1).numpy()
    return p * ys + ym


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="src/clasher_new/runs/_probe_cache_v2_stoch.npz")
    ap.add_argument("--layer", choices=["fused", "raw_obs"], default="fused")
    ap.add_argument("--proj", type=int, default=512)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--alpha", type=float, default=1e4)
    a = ap.parse_args()

    z = np.load(a.npz)
    y = z["y"].astype(np.float64)
    v = z["v"].astype(np.float64)
    ep = z["ep"].astype(np.int64)
    key = "X_fused" if a.layer == "fused" else "X_raw_obs"
    X = z[key].astype(np.float64)
    if a.proj and X.shape[1] > a.proj:
        rng = np.random.default_rng(0)
        X = X @ (rng.normal(size=(X.shape[1], a.proj)) / np.sqrt(a.proj))
    eps = np.unique(ep)
    print(f"[load] layer={a.layer} proj={a.proj} X={X.shape} frames={len(y)} episodes={len(eps)}")
    print(f"[baseline] 网络 critic EV_within={ev_within(v, y, ep):+.4f} | "
          f"常数基线=+0.0000（=什么都不做）")

    cuts = np.linspace(0, len(eps), a.folds + 1).astype(int)
    pl = np.full(len(y), np.nan)
    pm = np.full(len(y), np.nan)
    ps = np.full(len(y), np.nan)
    for i in range(1, a.folds):
        tr_eps, te_eps = eps[:cuts[i]], eps[cuts[i]:cuts[i + 1]]
        tr = np.isin(ep, tr_eps)
        te = np.isin(ep, te_eps)
        if tr.sum() < 200 or te.sum() < 50:
            continue
        sd = X[tr].std(0)
        sd[sd < 1e-8] = 1.0
        pl[te] = ridge(X[tr], y[tr], X[te], a.alpha, sd)
        pm[te] = mlp_fit_predict(X[tr], y[tr], X[te], hidden=a.hidden,
                                 epochs=a.epochs, sd=sd)
        ps[te] = mlp_fit_predict(X[tr], np.random.default_rng(1).permutation(y[tr]),
                                 X[te], hidden=a.hidden, epochs=a.epochs, sd=sd)
        print(f"  [fold {i}] train_ep={len(tr_eps)} test_ep={len(te_eps)} "
              f"train_n={int(tr.sum())} test_n={int(te.sum())}", flush=True)

    m = ~np.isnan(pl)
    print(f"\n=== 前向链式 CV（只用过去的局训练、预测未来的局；n={int(m.sum())} 帧）===")
    print(f"  常数基线（什么都不做）        EV_within = +0.0000")
    print(f"  线性岭回归（α={a.alpha:g}）        EV_within = {ev_within(pl[m], y[m], ep[m]):+.4f}")
    print(f"  非线性打分器 MLP({a.hidden})        EV_within = {ev_within(pm[m], y[m], ep[m]):+.4f}")
    print(f"  对照：标签打散 MLP            EV_within = {ev_within(ps[m], y[m], ep[m]):+.4f}")

    # 阳性对照：同块内拟合（in-sample，理论上"能拟合"时必然显著为正）
    tr_eps, te_eps = eps[:cuts[1]], eps[cuts[1]:cuts[2]]
    tr, te = np.isin(ep, tr_eps), np.isin(ep, te_eps)
    sd = X[tr].std(0); sd[sd < 1e-8] = 1.0
    ins = mlp_fit_predict(X[te], y[te], X[te], hidden=a.hidden, epochs=a.epochs, sd=sd,
                          val_frac=0.15)
    print(f"\n[阳性对照] 同块内拟合（in-sample）EV_within = "
          f"{ev_within(ins, y[te], ep[te]):+.4f}（应显著 >0 ⇒ 打分器有容量拟合该目标）")


if __name__ == "__main__":
    main()
