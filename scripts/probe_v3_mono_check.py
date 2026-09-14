"""只读诊断（**跑后追加，非预注册**）：v3 阶梯的 `α` 选择是否让层间比较失去意义？

起因：v3 预注册的 **G-MONO 在 3/3 条 seed 上 FAIL**——`EV_within(fused) < EV_within(grid_ln_out)`
而 `fused` 的前 2560 列**逐位就是** `grid_ln_out`（`fused = cat[grid_ln_out, hand_f, scalar, plan_f, belief_f]`）。
这说明**不是信息丢了，而是估计器选出了不同的 `α`**（fused 选的 α 比 grid_ln_out 大 10~1000 倍）。

本脚本做三件事（全部只读）：
  §1 `X_fused[:, :2560]` 与 `X_grid_ln_out` 的**逐位相等**验证（超集关系是否成立）；
  §2 每层的**完整 α 曲线**（test/val 的 `EV_within`）——看清"单层 argmax"与"共同 α"两种读法；
  §3 **共同 α 阶梯**：把每层放在同一批 α 上比较 ⇒ 这才是可比的层间比较。

用法（仓库根）：
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/probe_v3_mono_check.py \
      --npz runs/_probe_v3/seed7.npz --seed 7 --exclude grid_x
"""

import argparse
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src", "clasher_new")
for p in (_SRC, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

from pomdp_ceiling_probe import ev_within, standardize  # noqa: E402
from probe_value_ln import ALPHAS_V2, GRID_FLAT  # noqa: E402

#: 共同 α 阶梯要打印的点（跨层同一列可比）
COMMON_ALPHAS = (1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6)
LAYERS = ("raw_obs", "plan_f", "cnn_pre_ln", "grid_ln_out", "fused", "enc", "pre_ln")


def split_masks(ep, seed, test_frac=0.25, val_frac=0.15):
    rng = np.random.default_rng(seed)
    eps = np.unique(ep)
    rng.shuffle(eps)
    n_test = max(1, int(round(len(eps) * test_frac)))
    n_val = max(1, int(round(len(eps) * val_frac)))
    test_eps = set(eps[:n_test].tolist())
    val_eps = set(eps[n_test:n_test + n_val].tolist())
    m_te = np.array([e in test_eps for e in ep])
    m_va = np.array([e in val_eps for e in ep])
    m_tr = ~(m_te | m_va)
    return m_tr, m_va, m_te


def alpha_curve(Xl, y, ep, m_tr, m_va, m_te, ep_va, alphas=ALPHAS_V2):
    """返回 {α: (test_EV_within, val_EV_within, dim)}——与 `fit_ridge` 同一数学，只多报曲线。"""
    Xl = np.asarray(Xl, dtype=np.float64)
    sd_tr = Xl[m_tr].std(axis=0)
    keep = sd_tr > 1e-8
    if int(keep.sum()) == 0:
        return {al: (0.0, 0.0) for al in alphas}, 0
    Xk = Xl[:, keep]
    Xtr, Xva, Xte = standardize(Xk[m_tr], [Xk[m_tr], Xk[m_va], Xk[m_te]])
    lo, hi = float(y[m_tr].min()), float(y[m_tr].max())
    mu = float(y[m_tr].mean())
    XtX = Xtr.T @ Xtr
    Xty = Xtr.T @ (y[m_tr] - mu)
    I = np.eye(Xtr.shape[1])
    out = {}
    for al in alphas:
        w = np.linalg.solve(XtX + al * I, Xty)
        pv = np.clip(Xva @ w + mu, lo, hi)
        pt = np.clip(Xte @ w + mu, lo, hi)
        out[al] = (ev_within(pt, y[m_te], ep[m_te]),
                   ev_within(pv, y[m_va], ep_va))
    return out, int(keep.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--exclude", default="grid_x")
    ap.add_argument("--layers", nargs="*", default=list(LAYERS))
    ap.add_argument("--raw-obs-layer", default="raw_obs",
                    help="共同 α 的参考层（其 val 最优 α 作为『参考 α』）")
    a = ap.parse_args()

    _ex = {x.strip() for x in (a.exclude or "").split(",") if x.strip()}
    z = np.load(a.npz)
    X = {}
    for k in z.files:
        if not k.startswith("X_"):
            continue
        key = k.split("_", 1)[1]
        if key in _ex:
            continue
        arr = z[k]
        X[key] = arr if arr.dtype == np.float32 else arr.astype(np.float32)
    y = z["y"].astype(np.float64)
    ep = z["ep"]
    print(f"[npz] {a.npz}: frames={len(y)} episodes={int(ep.max()) + 1} 层={list(X)}",
          flush=True)

    m_tr, m_va, m_te = split_masks(ep, a.seed)
    ep_va = ep[m_va]
    y = y.copy()
    for e in np.unique(ep):
        m = (ep == e)
        y[m] -= y[m].mean()
    print(f"[split] train={m_tr.sum()} val={m_va.sum()} test={m_te.sum()} | "
          f"Var(R)_within={np.var(y):.4f}", flush=True)

    # ---- §1 超集关系逐位验证 ----
    print("\n--- §1 超集关系验证（逐位）---", flush=True)
    if "fused" in X and "grid_ln_out" in X:
        g = X["grid_ln_out"]
        f0 = X["fused"][:, :g.shape[1]]
        eq = bool(np.array_equal(f0, g))
        dmax = float(np.abs(f0.astype(np.float64) - g.astype(np.float64)).max())
        print(f"  X_fused[:, :{g.shape[1]}] == X_grid_ln_out ? {eq}  (max|Δ|={dmax:.3e})",
              flush=True)
        print("  ⇒ " + ("超集关系成立：fused 的前 2560 列**就是** grid_ln_out"
                        if eq else "⚠️ 超集关系不成立（仪器有问题）"), flush=True)
    else:
        print("  [skip] 缺 fused 或 grid_ln_out", flush=True)
        eq = None

    # ---- §2 每层完整 α 曲线 ----
    curves, dims = {}, {}
    print("\n--- §2 每层 α 曲线（test EV_within / val EV_within）---", flush=True)
    for key in a.layers:
        if key not in X:
            print(f"  [skip] {key} 不在 npz 中", flush=True)
            continue
        c, d = alpha_curve(X[key], y, ep, m_tr, m_va, m_te, ep_va)
        curves[key], dims[key] = c, d
        best_a = max(c, key=lambda al: (c[al][1] if c[al][1] is not None else -9e9))
        print(f"\n  {key} (dim={d}/{X[key].shape[1]}): 选中的 α={best_a:g}", flush=True)
        for al in ALPHAS_V2:
            t, v = c[al]
            mark = "  ← val 最优" if al == best_a else ""
            print(f"      α={al:<9g} test={0.0 if t is None else t:+.5f}  "
                  f"val={0.0 if v is None else v:+.5f}{mark}", flush=True)

    # ---- §3 共同 α 阶梯 ----
    ref = a.raw_obs_layer
    ref_alpha = None
    if ref in curves:
        ref_alpha = max(curves[ref],
                        key=lambda al: (curves[ref][al][1]
                                        if curves[ref][al][1] is not None else -9e9))
    print(f"\n--- §3 共同 α 阶梯（test EV_within；同一列 = 同一 α ⇒ 可比）---", flush=True)
    print(f"  参考层 {ref} 的 val 最优 α* = {ref_alpha:g}", flush=True)
    names = [k for k in a.layers if k in curves]
    print(f"{'α':>9} " + " ".join(f"{k:>12}" for k in names), flush=True)
    for al in COMMON_ALPHAS:
        cells = []
        for k in names:
            if al in curves[k]:
                t, _ = curves[k][al]
                cells.append(f"{0.0 if t is None else t:>+12.5f}")
            else:
                cells.append(f"{'—':>12}")
        print(f"{al:>9g} " + " ".join(cells), flush=True)

    print(f"\n--- §3b 每层自己的 val 最优 α（= 现有的『阶梯』读法）---", flush=True)
    for k in names:
        c = curves[k]
        best_a = max(c, key=lambda al: (c[al][1] if c[al][1] is not None else -9e9))
        t, v = c[best_a]
        R0 = None
        if ref in curves:
            ra = max(curves[ref],
                     key=lambda al: (curves[ref][al][1]
                                     if curves[ref][al][1] is not None else -9e9))
            R0 = curves[ref][ra][0]
        rho = (t / R0) if (R0 not in (None, 0) and R0 > 0 and t is not None) else None
        print(f"  {k:>12}  α*={best_a:<9g} test={0.0 if t is None else t:+.5f}  "
              f"ρ={('—' if rho is None else f'{rho:.4f}')}", flush=True)

    # ---- §4 结论判定 ----
    print("\n--- §4 判定（脚本复算，不手抄）---", flush=True)
    if "fused" in curves and "grid_ln_out" in curves and ref_alpha is not None:
        tf = curves["fused"][ref_alpha][0]
        tg = curves["grid_ln_out"][ref_alpha][0]
        print(f"  在**同一 α*({ref_alpha:g})** 下：fused={tf:+.5f} vs "
              f"grid_ln_out={tg:+.5f} ⇒ 差 {tf - tg:+.5f}", flush=True)
        print("  ⇒ " + ("超集不劣（差值 ≥ −1e-6）成立 ⇒ **层间落差是 α 选择造成的仪器假象**"
                        if (tf - tg) >= -1e-6 else
                        "⚠️ 同一 α 下 fused 仍劣于其子块 ⇒ 非单纯 α 选择问题"), flush=True)
    return 0


if __name__ == "__main__":
    main()
