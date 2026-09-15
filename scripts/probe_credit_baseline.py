# -*- coding: utf-8 -*-
"""结算基线可行性探针（只读、离线）：闭式"状态基线"能不能替死掉的 critic 干活？

背景（【O2】【C3】【C11】）：critic 实测恒为常数（局内 std=0）⇒ 优势里的状态可解释
分量一点没被减掉，`EV_within(v) ≈ 0`；而信息侧已判 **P3/H-LEARN**：局内回报变化在
现有输入里**线性可读**（Ridge +0.30）。⇒ 候选方案：用一个**闭式、状态only、可复算**
的线性基线替 critic 做"减水平"这件事（不改奖励、不改价值目标、不依赖网络健康）。

本探针不改任何仓库代码，只回答三件事：
  A) 复核 `EV_within(v)` ≈ 0（网络 critic 的局内解释力）；
  B) 同一批数据上，**状态 only 线性基线**能到多少（OOF=乐观上限 / 因果滚动=可实现）；
  C) 控制组：局内打散特征（应塌到 ≤0）、oracle（应 =1.00）、常数基线（应 ≈0）。

用法（仓库根）：
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/probe_credit_baseline.py \
        --npz src/clasher_new/runs/_probe_cache_v2_stoch.npz
"""
from __future__ import annotations

import argparse
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

GRID_FLAT = 8640          # 18*32*15（observation.observe 的 grid 展平）
NONGRID = 8               # hand(5) + elixir + next_card + time
CH = 15                   # 每格 15 通道
WIN = 128                 # update_interval：一次 PPO 更新的帧数（= 结算窗口）


def ev_within(pred, y, ep):
    """局内中心化 EV（与 scripts/pomdp_ceiling_probe.py:202 同口径，逐位一致实现）。"""
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


def state_features(raw):
    """从 raw_obs(8640+8) 提**动作无关**的状态特征（全部可由引擎数值直接算出）。

    通道序（rl/observation.py:110-114）：0 entity_id / 1 is_opponent / 2 elixir /
    3 card_type / 4 speed / 5 is_air / 6 att_ground / 7 att_air / 8 log(hp)/10 /
    9 hp/hp_max / 10 hit_speed / 11 range/3 / 12 sight/3 / 13 damage/200 /
    14 projectile_damage/200。栅格是**一格一实体**（后写覆盖），故这里是"格级聚合"。
    """
    g = raw[:, :GRID_FLAT].reshape(len(raw), 576, CH)
    hand = raw[:, GRID_FLAT:GRID_FLAT + 5]
    elixir = raw[:, GRID_FLAT + 5]
    next_card = raw[:, GRID_FLAT + 6]
    t = raw[:, GRID_FLAT + 7]

    ent_id = g[:, :, 0]
    is_opp = g[:, :, 1] > 0.5
    occ = ent_id > 0
    hp_pct = g[:, :, 9]
    hp_log = g[:, :, 8]
    dmg = g[:, :, 13]
    tower = occ & (ent_id <= 6.0)                    # 塔 id 1..6
    my_t = tower & (~is_opp)
    op_t = tower & is_opp
    my_u = occ & (~is_opp) & (~tower)
    op_u = occ & is_opp & (~tower)

    def masked_sum(mask, val):
        return (np.where(mask, val, 0.0)).sum(axis=1)

    feats = np.column_stack([
        np.ones(len(raw)),                            # 0 截距
        t / 180.0,                                    # 1 时间
        (t >= 120.0).astype(np.float64),              # 2 相位（双倍期）
        elixir,                                       # 3 圣水
        (elixir >= 6.0).astype(np.float64),           # 4 攒够 6 费
        (elixir >= 4.0).astype(np.float64),           # 5 攒够 4 费
        hand.sum(axis=1) / 1000.0,                    # 6 手牌 id 和（廉价代理）
        next_card / 200.0,                            # 7 下一张 id
        masked_sum(my_t, hp_pct),                     # 8 我方塔血%
        masked_sum(op_t, hp_pct),                     # 9 敌方塔血%
        masked_sum(my_t, hp_log),                     # 10
        masked_sum(op_t, hp_log),                     # 11
        masked_sum(my_u, hp_pct),                     # 12 我方单位血%
        masked_sum(op_u, hp_pct),                     # 13 敌方单位血%
        masked_sum(my_u, hp_log),                     # 14
        masked_sum(op_u, hp_log),                     # 15
        masked_sum(op_u, dmg),                        # 16 敌方威胁
        masked_sum(occ, hp_pct),                      # 17 总血%
        (my_u.sum(axis=1) - op_u.sum(axis=1)),        # 18 单位数差
    ])
    return feats


def _std_fit(X):
    """只做**尺度**归一（不中心化：中心化交给 ridge_fit 的截距处理）。"""
    sd = X.std(axis=0)
    sd[sd < 1e-8] = 1.0
    return sd


def load_layer(npz, layer, proj, seed=0):
    """取某一层特征：small=19 维手写状态特征 / fused(2731) / raw_obs(8648)。

    proj>0 时先做**随机投影**（JL）降维，使闭式岭回归在滚动拟合下的成本可控。
    """
    z = np.load(npz)
    y = z["y"].astype(np.float64)
    v = z["v"].astype(np.float64)
    ep = z["ep"].astype(np.int64)
    if layer == "small":
        X = state_features(z["X_raw_obs"].astype(np.float64))
    elif layer == "fused":
        X = z["X_fused"].astype(np.float64)
    elif layer == "raw_obs":
        X = z["X_raw_obs"].astype(np.float64)
    else:
        raise ValueError(layer)
    if proj and X.shape[1] > proj:
        rng = np.random.default_rng(seed)
        R = rng.normal(size=(X.shape[1], proj)) / np.sqrt(proj)
        X = X @ R
    return X, y, v, ep


def ridge_fit(X, y, alpha):
    """闭式岭回归，**截距不受罚**（对 X、y 双侧中心化后再解）。返回 (w, xm, ym)。"""
    xm = X.mean(axis=0)
    ym = float(y.mean())
    Xc = X - xm
    d = X.shape[1]
    A = Xc.T @ Xc + alpha * np.eye(d)
    w = np.linalg.solve(A, Xc.T @ (y - ym))
    return w, xm, ym


def ridge_pred(X, model):
    w, xm, ym = model
    return (X - xm) @ w + ym


def oof_pred(X, y, ep, alpha):
    """留一局（leave-one-episode-out）：每折用其它局拟合 → 预测该局（含折内标准化）。"""
    pred = np.zeros(len(y))
    for e in np.unique(ep):
        te = (ep == e); tr = ~te
        if tr.sum() < 5:
            pred[te] = 0.0; continue
        sd = _std_fit(X[tr])
        m = ridge_fit(X[tr] / sd, y[tr], alpha)
        pred[te] = ridge_pred(X[te] / sd, m)
    return pred


def causal_pred(X, y, ep, alpha, refit_every=10):
    """因果滚动（可上线口径）：只用**之前的局**拟合，预测当前局；每 refit_every 局重拟合。"""
    pred = np.zeros(len(y))
    eps = np.unique(ep)
    m = None; sd = None
    for i, e in enumerate(eps):
        te = (ep == e); tr = ep < e
        if m is None or (i % refit_every == 0 and tr.sum() >= 5):
            if tr.sum() >= 5:
                sd = _std_fit(X[tr])
                m = ridge_fit(X[tr] / sd, y[tr], alpha)
        if m is None:
            pred[te] = 0.0
        else:
            pred[te] = ridge_pred(X[te] / sd, m)
    return pred


def window_pred(X, y, ep, alpha, win=WIN, trail=8):
    """真实机制口径：每个 win 帧窗口用**其前面 trail*win 帧**拟合（含标准化）。"""
    n = len(y)
    pred = np.zeros(n)
    for s in range(0, n, win):
        e = min(n, s + win); t0 = max(0, s - trail * win)
        if s - t0 < 30:
            pred[s:e] = 0.0; continue
        sd = _std_fit(X[t0:s])
        m = ridge_fit(X[t0:s] / sd, y[t0:s], alpha)
        pred[s:e] = ridge_pred(X[s:e] / sd, m)
    return pred


def a_structure(y, v, ep, win=WIN):
    """优势 A = y − v 的方差剖析：多少来自"尖峰"、多少来自"慢漂移/窗口间"。

    用途：决定结算改哪一层——尖峰主导 ⇒ 尾部收缩类手段有机制依据；
    窗口间漂移主导 ⇒ 只有基线（学到的或闭式的）能动它，而"闭式基线"已被本探针否证。
    """
    A = y - v
    print(f"\n=== 优势结构 A = y − v | frames={len(A)} episodes={int(ep.max())+1} "
          f"| v 唯一值={len(np.unique(np.round(v, 6)))}（常数 ⇒ A = y − 常数）===")
    print(f"[A] mean={A.mean():+.4f} std={A.std():.4f} min={A.min():+.2f} max={A.max():+.2f} "
          f"kurtosis={float(((A - A.mean()) ** 4).mean() / A.var() ** 2):.2f}")
    for p in (0.9, 0.99, 0.999):
        thr = np.quantile(np.abs(A), p)
        m = np.abs(A) > thr
        print(f"  顶部 {100 * (1 - p):>5.1f}% 帧(|A|>{thr:7.2f}, n={int(m.sum()):>4d}) "
              f"占 ΣA² 的 {100 * (A[m] ** 2).sum() / (A ** 2).sum():5.1f}%"
              f"  占 Σ|A| 的 {100 * np.abs(A[m]).sum() / np.abs(A).sum():5.1f}%")
    acs = []
    for e in np.unique(ep):
        a = A[ep == e]
        if len(a) > 10:
            acs.append(float(np.corrcoef(a[:-1], a[1:])[0, 1]))
    print(f"[局内 lag-1 自相关] mean={np.mean(acs):+.3f} median={np.median(acs):+.3f} "
          f"n_ep={len(acs)}（≈1 ⇒ 优势在帧间几乎不变 ⇒ 逐帧结算的'信息'很少）")
    wth, btw = [], []
    for e in np.unique(ep):
        a = A[ep == e]
        for s in range(0, len(a) - 1, win):
            w = a[s:s + win]
            if len(w) < 8:
                continue
            wth.append(w.var())
            btw.append(w.mean())
    wth, btw = np.array(wth), np.array(btw)
    tot = wth.mean() + btw.var()
    print(f"[{win} 帧窗口分解] 窗口内 Var 均值={wth.mean():.3f} | 窗口均值间 Var={btw.var():.3f}"
          f" ⇒ 窗口内 {100 * wth.mean() / tot:.1f}% / 窗口间 {100 * btw.var() / tot:.1f}%")
    AA = np.concatenate([A[ep == e] - A[ep == e].mean() for e in np.unique(ep)])
    print(f"[局内中心化] std(A_within)={AA.std():.4f}（critic 完全失效时优势的实际起伏）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--layer", choices=["small", "fused", "raw_obs"], default="small")
    ap.add_argument("--proj", type=int, default=512, help="随机投影维度（0=不投影）")
    ap.add_argument("--alphas", default="0.01,1,100,10000")
    ap.add_argument("--refit-every", type=int, default=10)
    ap.add_argument("--skip-oof", action="store_true", help="高维层 OOF 太慢时跳过")
    ap.add_argument("--a-structure", action="store_true",
                    help="只做优势 A=y-v 的结构剖析（尖峰 vs 慢漂移），不拟合")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    X, y, v, ep = load_layer(a.npz, a.layer, a.proj)
    if a.a_structure:
        a_structure(y, v, ep)
        return
    n_ep = int(ep.max()) + 1
    alphas = [float(x) for x in a.alphas.split(",") if x.strip()]
    print(f"[load] layer={a.layer} proj={a.proj} X={X.shape} frames={len(y)} "
          f"episodes={n_ep} ep_len_mean={len(y)/n_ep:.1f}")
    print(f"[y] mean={y.mean():+.4f} std={y.std():.4f} | [v] mean={v.mean():+.4f} "
          f"std={v.std():.6f} 唯一值={len(np.unique(np.round(v, 6)))}")
    yw = np.concatenate([y[ep == e] - y[ep == e].mean() for e in np.unique(ep)])
    vw = np.concatenate([v[ep == e] - v[ep == e].mean() for e in np.unique(ep)])
    print(f"[within] std(y_within)={yw.std():.4f} std(v_within)={vw.std():.6f} "
          f"⇒ v 局内活动量/ y = {vw.std()/max(1e-12, yw.std()):.2e}")

    print(f"\n=== EV_within（局内中心化）| 网络 critic = {ev_within(v, y, ep):+.4f} | "
          f"常数基线 = {ev_within(np.array([y[ep==e].mean() for e in ep]), y, ep):+.4f} ===")
    hdr = f"{'alpha':>10} {'OOF(乐观上限)':>15} {'因果滚动(可实现)':>18} "           f"{'窗口128x8(机制口径)':>20} {'std(pred)/std(y)':>18}"
    print(hdr)
    best = None
    for al in alphas:
        p_oof = None if a.skip_oof else oof_pred(X, y, ep, al)
        p_ca = causal_pred(X, y, ep, al, a.refit_every)
        p_wi = window_pred(X, y, ep, al)
        e_oof = None if p_oof is None else ev_within(p_oof, y, ep)
        e_ca, e_wi = ev_within(p_ca, y, ep), ev_within(p_wi, y, ep)
        pw = np.concatenate([p_ca[ep == e] - p_ca[ep == e].mean() for e in np.unique(ep)])
        so = "     (skip)" if e_oof is None else f"{e_oof:>+15.4f}"
        print(f"{al:>10.4g} {so} {e_ca:>+18.4f} {e_wi:>+20.4f} "
              f"{pw.std()/yw.std():>18.3f}")
        if best is None or (e_ca or -9) > best[1]:
            best = (al, e_ca)
    print(f"\n[控制组] oracle=y: {ev_within(y, y, ep):+.4f}（应 1.00）")
    rng = np.random.default_rng(1)
    Xn = rng.normal(size=X.shape)
    print(f"[控制组] 纯噪声特征 OOF: {ev_within(oof_pred(Xn, y, ep, 1.0), y, ep):+.4f}")
    if a.out:
        import json
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"npz": a.npz, "layer": a.layer, "proj": a.proj,
                       "frames": int(len(y)), "episodes": n_ep,
                       "ev_within_v": ev_within(v, y, ep),
                       "within_std_y": float(yw.std()), "within_std_v": float(vw.std()),
                       "alphas": alphas}, f, ensure_ascii=False, indent=1)
        print(f"[out] {a.out}")


if __name__ == "__main__":
    main()
