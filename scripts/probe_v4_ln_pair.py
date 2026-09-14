"""只读：价值通路探针 v4 —— **真·同一张量**的「投影 / 归一化」分段对账。

预注册：`docs/value_ln_probe4_prereg_2026-09-14.md`（判据/闸门/分支 **跑前写死**，【红线 R3】）。

不跑 rollout：直接读 v3 的 npz（`X_fused` 等）**离线精确重算**中间张量
（`z_enc = enc_fc(fused)` → `r_enc = relu(...)` → `enc_off = enc_ln(...)`），
并用 v3 抓下来的 `X_enc` 做 **G-OFFLINE 对账**——重算对不对不靠信任，靠逐位比。

用法（在 src/clasher_new 下）：
  PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe ../../scripts/probe_v4_ln_pair.py \
      --npz ../../runs/_probe_v3/seed7.npz --ckpt runs/critic_inert_probe_20k/solo_main_20000.pt \
      --seed 7 --exclude grid_x --out ../../runs/_probe_v4/seed7.json
"""

import argparse
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src", "clasher_new")
for _p in (_SRC, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

from rl.config import TrainConfig  # noqa: E402
from rl.belief import BeliefInference  # noqa: E402
from rl.plan_space import PLAN_DIM  # noqa: E402
from rl.follower import load_checkpoint  # noqa: E402
from rl.train_solo import solo_env  # noqa: E402

from pomdp_ceiling_probe import ev_within, standardize  # noqa: E402
from probe_value_ln import ALPHAS_V2  # noqa: E402

#: 判据参照集（预注册 §5）
COMMON_ALPHAS = (1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6)
#: G-REPRO2 参考值（= v3 seed 7 的权威读数，`runs/_probe_v3/seed7.json`）
REPRO2 = {"EV_raw": 0.12145137497621039, "EV_fused": 0.04312000460119425}
RAND_SEED = 20260914          # P-RANDOM 的固定种子（预注册 §7 要求报出）


def split_masks(ep, seed, test_frac=0.25, val_frac=0.15):
    rng = np.random.default_rng(seed)
    eps = np.unique(ep)
    rng.shuffle(eps)
    n_test = max(1, int(round(len(eps) * test_frac)))
    n_val = max(1, int(round(len(eps) * val_frac)))
    te = set(eps[:n_test].tolist())
    va = set(eps[n_test:n_test + n_val].tolist())
    m_te = np.array([e in te for e in ep])
    m_va = np.array([e in va for e in ep])
    return ~(m_te | m_va), m_va, m_te


def alpha_curve(Xl, y, ep, m_tr, m_va, m_te, ep_va, alphas=ALPHAS_V2):
    Xl = np.asarray(Xl, dtype=np.float64)
    sd = Xl[m_tr].std(axis=0)
    keep = sd > 1e-8
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
        pt = np.clip(Xte @ w + mu, lo, hi)
        pv = np.clip(Xva @ w + mu, lo, hi)
        out[al] = (ev_within(pt, y[m_te], ep[m_te]), ev_within(pv, y[m_va], ep_va))
    return out, int(keep.sum())


def layer_norm(X, w, b, eps=1e-5):
    """复刻 torch.nn.LayerNorm（对最后一维）。"""
    mu = X.mean(axis=-1, keepdims=True)
    var = X.var(axis=-1, keepdims=True)
    return (X - mu) / np.sqrt(var + eps) * w + b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--exclude", default="grid_x")
    ap.add_argument("--out", default=None)
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
    print(f"[npz] {a.npz}: frames={len(y)} episodes={int(ep.max()) + 1}", flush=True)

    # ---- 载 ckpt：拿 enc_fc / enc_ln 的权重做离线重算 ----
    cfg = TrainConfig.resolve("economy")
    run_dir = os.path.dirname(os.path.abspath(a.ckpt))
    cp = os.path.join(run_dir, "config.json")
    if os.path.exists(cp):
        d = json.load(open(cp, encoding="utf-8"))
        for k, vv in d.items():
            if hasattr(cfg, k):
                try:
                    setattr(cfg, k, vv)
                except Exception:
                    pass
    _e = solo_env(cfg, a.seed)
    _bd = len(BeliefInference(opp_deck=_e.deck1, n_particles=128,
                              seed=a.seed).encode(None, None))
    pol = load_checkpoint(a.ckpt, hidden_dim=cfg.hidden_dim, plan_dim=PLAN_DIM,
                          belief_dim=_bd)
    pol.eval()

    def _np(t):
        return t.detach().float().cpu().numpy()

    W_fc, b_fc = _np(pol.enc_fc.weight), _np(pol.enc_fc.bias)          # (128,2731),(128,)
    g_ln, be_ln = _np(pol.enc_ln.weight), _np(pol.enc_ln.bias)
    fused = X["fused"].astype(np.float32)
    print(f"[offline] enc_fc: {W_fc.shape} ‖W‖={np.linalg.norm(W_fc):.4f} "
          f"‖b‖={np.linalg.norm(b_fc):.4f} | enc_ln γ mean={g_ln.mean():+.5f}", flush=True)
    z_enc = fused @ W_fc.T + b_fc
    r_enc = np.maximum(z_enc, 0.0)
    enc_off = layer_norm(r_enc.astype(np.float64), g_ln.astype(np.float64),
                         be_ln.astype(np.float64)).astype(np.float32)

    # ---- G-OFFLINE：离线重算 vs v3 抓下来的 X_enc ----
    print("\n--- G-OFFLINE：离线重算 vs 网络实抓 ---", flush=True)
    off_ok, off_dmax = None, None
    if "enc" in X:
        off_dmax = float(np.abs(enc_off.astype(np.float64)
                                - X["enc"].astype(np.float64)).max())
        off_ok = bool(off_dmax <= 1e-4)
        print(f"  max|enc_off − X_enc| = {off_dmax:.3e}  ≤1e-4 ? {off_ok}", flush=True)
    else:
        print("  ⚠️ npz 里没有 X_enc ⇒ 无法对账（G-OFFLINE FAIL）", flush=True)
    X["z_enc"] = z_enc
    X["r_enc"] = r_enc
    X["enc_off"] = enc_off

    # ---- z_rand：随机投影（容量对照，预注册 §2/§5）----
    rng = np.random.default_rng(RAND_SEED)
    R = rng.standard_normal((128, fused.shape[1]))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    X["z_rand"] = (fused @ R.T.astype(np.float32)).astype(np.float32)
    print(f"  [rand] z_rand = R·fused，R 为按行归一化高斯，seed={RAND_SEED}", flush=True)
    # 描述性扩展（**不参与判据**，预注册 §7 只要求单一固定种子）：再补 2 个随机投影，
    # 用来判断"训练出来的投影 ≈ 随机投影"这条描述是否只是那一个随机种子的运气。
    for j, sd in enumerate((RAND_SEED + 1, RAND_SEED + 2), start=2):
        r2 = np.random.default_rng(sd)
        R2 = r2.standard_normal((128, fused.shape[1]))
        R2 /= np.linalg.norm(R2, axis=1, keepdims=True)
        X[f"z_rand{j}"] = (fused @ R2.T.astype(np.float32)).astype(np.float32)
    print(f"  [rand] 描述性扩展 z_rand2/z_rand3（种子 {RAND_SEED + 1}/{RAND_SEED + 2}，"
          f"不参与 P-RANDOM）", flush=True)

    # ---- 切分 / 目标（与 v3 逐位同一构造）----
    m_tr, m_va, m_te = split_masks(ep, a.seed)
    ep_va = ep[m_va]
    y = y.copy()
    for e in np.unique(ep):
        m = (ep == e)
        y[m] -= y[m].mean()
    print(f"[split] train={m_tr.sum()} val={m_va.sum()} test={m_te.sum()} | "
          f"Var(R)_within={np.var(y):.4f}", flush=True)

    LAYERS = ["raw_obs", "grid_ln_out", "fused",
              "z_enc", "r_enc", "enc", "enc_off", "z_rand", "z_rand2", "z_rand3",
              "pre_ln", "relu_ln", "post_ln"]
    curves, dims = {}, {}
    print("\n--- α 曲线（test EV_within）---", flush=True)
    for key in LAYERS:
        if key not in X:
            print(f"  [skip] {key}", flush=True)
            continue
        c, d = alpha_curve(X[key], y, ep, m_tr, m_va, m_te, ep_va)
        curves[key], dims[key] = c, d
        best = max(c, key=lambda al: (c[al][1] if c[al][1] is not None else -9e9))
        line = " ".join(f"{al:g}:{0.0 if c[al][0] is None else c[al][0]:+.4f}"
                        for al in COMMON_ALPHAS)
        print(f"  {key:12s} D={d:5d}/{X[key].shape[1]:<5d} α*={best:<9g} "
              f"test@α*={0.0 if c[best][0] is None else c[best][0]:+.5f}", flush=True)
        print(f"      {line}", flush=True)

    # ---- G-REPRO2 ----
    print("\n--- 闸门 ---", flush=True)
    gates = {}
    R0 = curves["raw_obs"][max(curves["raw_obs"],
                               key=lambda al: curves["raw_obs"][al][1])][0]
    print(f"  EV(raw_obs)={R0:+.8f} (v3 参考 {REPRO2['EV_raw']:+.8f})", flush=True)
    if a.seed == 7:
        d_raw = abs(R0 - REPRO2["EV_raw"])
        # fused 的 α* 读数（与 v3 同一读法）
        f_a = max(curves["fused"], key=lambda al: curves["fused"][al][1])
        d_fu = abs(curves["fused"][f_a][0] - REPRO2["EV_fused"])
        gates["G-REPRO2"] = bool(d_raw <= 0.002 and d_fu <= 0.002)
        print(f"  G-REPRO2 ΔEV(raw)={d_raw:.3e}≤0.002 ? {d_raw <= 0.002} | "
              f"ΔEV(fused)={d_fu:.3e}≤0.002 ? {d_fu <= 0.002} ⇒ {gates['G-REPRO2']}",
              flush=True)
    else:
        gates["G-REPRO2"] = None
        print(f"  G-REPRO2 N/A（仅 seed 7）", flush=True)

    # G-OFFLINE 的 EV 侧
    if off_ok is not None and "enc" in curves and "enc_off" in curves:
        e_a = max(curves["enc"], key=lambda al: curves["enc"][al][1])
        o_a = max(curves["enc_off"], key=lambda al: curves["enc_off"][al][1])
        d_ev = abs(curves["enc_off"][o_a][0] - curves["enc"][e_a][0])
        off_ok = bool(off_ok and d_ev <= 0.002)
        print(f"  G-OFFLINE EV 侧：ΔEV(enc_off vs enc)={d_ev:.3e}≤0.002 ? {d_ev <= 0.002}",
              flush=True)
    gates["G-OFFLINE"] = off_ok

    # 退化护栏 + 估计器电池（与 v3 同构，简化重算）
    yscr = y.copy()
    k_in = np.zeros(len(ep))
    for e in np.unique(ep):
        idx = np.where(ep == e)[0]
        k_in[idx] = np.arange(len(idx))
    cclk, _ = alpha_curve(k_in.reshape(-1, 1), y, ep, m_tr, m_va, m_te, ep_va)
    clk_a = max(cclk, key=lambda al: cclk[al][1])
    gates["G-CLK"] = bool(cclk[clk_a][0] is not None and cclk[clk_a][0] <= 0.05)
    gates["G-VAR"] = bool(float(np.var(y)) > 1.0)
    ypc, _ = None, None
    try:
        from pomdp_ceiling_probe import make_positive_control, scramble_rows_by_clock
        ypc, _ = make_positive_control(X["fused"], ep, seed=a.seed,
                                       within_var=58.0, between_var=166.0)
        for e in np.unique(ep):
            m = (ep == e)
            ypc[m] -= ypc[m].mean()
        cpc, _ = alpha_curve(X["fused"], ypc, ep, m_tr, m_va, m_te, ep_va)
        pc_a = max(cpc, key=lambda al: cpc[al][1])
        gates["G-PC"] = bool(cpc[pc_a][0] is not None and cpc[pc_a][0] >= 0.15)
        scr = {}
        for key in ("raw_obs", "fused", "grid_ln_out"):
            if key not in X:
                continue
            Xs = scramble_rows_by_clock(X[key], ep, seed=a.seed)
            cs, _ = alpha_curve(Xs, y, ep, m_tr, m_va, m_te, ep_va)
            sa = max(cs, key=lambda al: cs[al][1])
            scr[key] = cs[sa][0]
        gates["G-SCR"] = bool(all(v is not None and v <= 0.05 for v in scr.values()))
        scr_max = max([v for v in scr.values() if v is not None] or [0.0])
    except Exception as exc:
        print(f"  ⚠️ 阳性/打散对照构造失败: {type(exc).__name__}", flush=True)
        gates["G-PC"] = gates["G-SCR"] = None
        scr_max = 0.0
    gates["G-ANCHOR"] = bool(R0 is not None and R0 >= 0.05 and R0 >= 5.0 * scr_max)
    print("  " + " ".join(f"{k}={'N/A' if v is None else ('PASS' if v else 'FAIL')}"
                          for k, v in gates.items()), flush=True)
    gates_ok = all(v for v in gates.values() if v is not None)

    # ---- 判据（预注册 §5；同 α 配对，全 α 同向才算命中）----
    print("\n--- 判据（预注册 §5；同 α 配对，要求每个 α 同向）---", flush=True)

    def S(L, al):
        c = curves.get(L)
        return None if c is None else c[al][0]

    def all_alpha(pred):
        vals = []
        for al in COMMON_ALPHAS:
            try:
                vals.append(bool(pred(al)))
            except Exception:
                vals.append(False)
        return bool(all(vals)), vals

    br = {}
    br["P-PROJ"], v_proj = all_alpha(
        lambda al: (S("z_enc", al) is not None and S("fused", al) is not None
                    and S("z_enc", al) <= 0.5 * S("fused", al)))
    br["P-LN"], v_ln = all_alpha(
        lambda al: (S("z_enc", al) is not None and S("fused", al) is not None
                    and S("enc", al) is not None
                    and S("z_enc", al) >= 0.5 * S("fused", al)
                    and S("enc", al) <= 0.5 * S("z_enc", al)))
    br["P-BOTH"], v_both = all_alpha(
        lambda al: (S("z_enc", al) is not None and S("fused", al) is not None
                    and S("enc", al) is not None
                    and S("z_enc", al) <= 0.5 * S("fused", al)
                    and S("enc", al) <= 0.5 * S("z_enc", al)))
    br["P-RANDOM"], v_rnd = all_alpha(
        lambda al: (S("z_enc", al) is not None and S("z_rand", al) is not None
                    and S("z_enc", al) <= 1.5 * S("z_rand", al)))
    br["P-VLN"], v_vln = all_alpha(
        lambda al: (S("relu_ln", al) is not None and S("pre_ln", al) is not None
                    and S("post_ln", al) is not None
                    and S("relu_ln", al) >= 0.5 * S("pre_ln", al)
                    and S("post_ln", al) <= 0.5 * S("relu_ln", al)))
    br["P-VLN-NEUTRAL"], v_vn = all_alpha(
        lambda al: (S("post_ln", al) is not None and S("relu_ln", al) is not None
                    and S("post_ln", al) >= 0.9 * S("relu_ln", al)))

    order = ["P-PROJ", "P-LN", "P-BOTH", "P-RANDOM", "P-VLN", "P-VLN-NEUTRAL"]
    print(f"{'α':>9} {'fused':>9} {'z_enc':>9} {'enc':>9} {'z_rand':>9} "
          f"{'pre_ln':>9} {'relu_ln':>9} {'post_ln':>9}", flush=True)
    for al in COMMON_ALPHAS:
        def f(L):
            v = S(L, al)
            return f"{v:>+9.5f}" if v is not None else f"{'—':>9}"
        print(f"{al:>9g} {f('fused')} {f('z_enc')} {f('enc')} {f('z_rand')} "
              f"{f('pre_ln')} {f('relu_ln')} {f('post_ln')}", flush=True)
    print("", flush=True)
    for b in order:
        print(f"  {b:16s} {br[b]}   per-α: {[int(x) for x in {'P-PROJ': v_proj, 'P-LN': v_ln, 'P-BOTH': v_both, 'P-RANDOM': v_rnd, 'P-VLN': v_vln, 'P-VLN-NEUTRAL': v_vn}[b]]}", flush=True)
    hits = [b for b in order if br[b]]
    if not gates_ok:
        verdict = "V4_INVALID"
    elif not hits:
        verdict = "NOT_PREREGISTERED"
    else:
        verdict = "+".join(hits)
    print(f"\n  gates_ok={gates_ok}  hits={hits}  VERDICT(v4) = {verdict}", flush=True)

    res = {"npz": a.npz, "ckpt": a.ckpt, "seed": a.seed, "frames": int(len(y)),
           "episodes": int(ep.max()) + 1, "gates": gates, "gates_ok": bool(gates_ok),
           "offline_max_abs_diff": off_dmax,
           "curves": {k: {str(al): v for al, v in c.items()} for k, c in curves.items()},
           "dims": dims, "branches": br, "hits": hits, "verdict": verdict,
           "per_alpha_flags": {"P-PROJ": v_proj, "P-LN": v_ln, "P-BOTH": v_both,
                               "P-RANDOM": v_rnd, "P-VLN": v_vln,
                               "P-VLN-NEUTRAL": v_vn},
           "rand_seed": RAND_SEED,
           "n_prereg": "docs/value_ln_probe4_prereg_2026-09-14.md"}
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        print(f"[out] {a.out}", flush=True)
    return res


if __name__ == "__main__":
    main()
