"""只读：价值通路「逐层线性可读性阶梯」。

预注册：`docs/value_ln_probe_prereg_2026-09-14.md`（判据/闸门/分支在跑之前写死，【红线 R3】）。

问题：从 `fused` 到 `value` 的逐层阶梯上，局内可预测性（`EV_within`）在第几层掉到 ≈0？
方法：单 ckpt / 单 rollout 逐帧抓 6 层激活 → 每层用第二轮主估计器（Ridge + alpha 网格 +
      输出裁剪 + `select=within`）拟**同一个目标**（GAE 回报，按局中心化），**同一局分组切分**
      ⇒ 层间是配对比较，落差可信。

只有前向、不写 ckpt、不改任何训练状态。

用法（在 src/clasher_new 下）：
  PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe \
      ../../scripts/probe_value_ln.py --ckpt runs/critic_inert_probe_20k/solo_main_20000.pt \
      --frames 30000 --seed 7 --device cuda --tag primary \
      --out ../../docs/value_ln_probe_primary.json
"""

import argparse
import json
import os
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src", "clasher_new")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rl.config import TrainConfig  # noqa: E402
from rl.belief import BeliefInference  # noqa: E402
from rl.belief_planner import BeliefPlanner  # noqa: E402
from rl.plan_space import PLAN_DIM  # noqa: E402
from rl.follower import FollowerPolicy, load_checkpoint  # noqa: E402
from rl.train_follower import FollowerOpponent  # noqa: E402
from rl.train_solo import solo_env, resolve_deck_set  # noqa: E402
from rl.ppo import PPOTrainer  # noqa: E402

# 复用第二轮主估计器与对照构造（逐位一致，避免"两套估计器"）
from pomdp_ceiling_probe import (  # noqa: E402
    ev, ev_within, ev_within_raw, standardize, fit_ridge, feat_obs,
    make_positive_control, scramble_rows_by_clock,
)

#: α 网格。v1 = 上一轮用的（下界 1e-1）；v2 = 本轮主网格（下界 1e-6，修 ⑩ 类"边界"缺陷）。
ALPHAS_V1 = (1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8)
ALPHAS_V2 = (1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8)
GRIDS = {"v1": ALPHAS_V1, "v2": ALPHAS_V2}

#: 阶梯：(标签, 缓冲区键)。顺序 = 前向顺序，判据 §3.2 的 `k*` 就是这个次序的下标。
LADDER = [
    ("L0_fused", "fused"),
    ("L1_pre_ln", "pre_ln"),
    ("L2_relu_ln", "relu_ln"),
    ("L3_post_ln", "post_ln"),
    ("L4_mlp0_post", "mlp0_post"),
    ("L5_value", "value"),
]


def _force_utf8_stdout():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# --------------------------------------------------------------------------
# rollout（逐帧抓 6 层）
# --------------------------------------------------------------------------
def rollout_layers(a, cfg, device):
    env = solo_env(cfg, a.seed)
    bdim = len(BeliefInference(opp_deck=env.deck1, n_particles=128,
                               seed=a.seed).encode(None, None))
    pol = load_checkpoint(a.ckpt, hidden_dim=cfg.hidden_dim, plan_dim=PLAN_DIM,
                          belief_dim=bdim)
    pol.to_device(device).eval()
    print(f"=== value-LN probe | ckpt={a.ckpt} mode={a.mode} frames<={a.frames} "
          f"device={device} ===", flush=True)
    print(f"    independent={pol.value_independent} bypass={pol.value_bypass} "
          f"γ={cfg.gamma} λ={cfg.gae_lambda}", flush=True)
    if not pol.value_independent:
        raise SystemExit("[abort] 该 ckpt value_independent=False ⇒ 无 value_enc_ln 层，"
                         "本预注册的问题不适用（先确认 ckpt 元数据）")

    for nm, p in pol.named_parameters():
        if "value" in nm:
            print(f"  [param] {nm:26s} {tuple(p.shape)} "
                  f"‖·‖={p.detach().float().norm().item():.8f}", flush=True)

    opp = FollowerPolicy(hidden=cfg.hidden_dim, plan_dim=PLAN_DIM, belief_dim=bdim,
                         value_bypass=bool(pol.value_bypass),
                         value_independent=bool(pol.value_independent))
    opp.to_device(device)
    opp.load_state_dict(pol.state_dict())
    det = (a.mode == "det")
    env.opponent = FollowerOpponent(
        opp, env, belief=BeliefInference(opp_deck=env.deck1, n_particles=128,
                                         seed=a.seed + 1), deterministic=det)
    _, _ = resolve_deck_set(getattr(cfg, "deck_set", None) or "default")

    bp = BeliefPlanner()
    belief = BeliefInference(opp_deck=env.deck1, n_particles=128, seed=a.seed)
    obs, _ = env.reset(seed=a.seed)
    belief.reset(env.deck1)
    hidden = None

    keys = [k for _, k in LADDER] + (["raw_obs"] if a.raw_obs else [])
    acc = {k: [] for k in keys}
    Y, V, EP = [], [], []
    ep_rew, ep_val, ep_term, ep_trunc = [], [], [], []
    ep_acc = {k: [] for k in keys}
    ep_idx, ep_lens, n_frame = 0, [], 0
    t0 = time.time()
    step = 0
    with torch.no_grad():
        while n_frame < a.frames:
            plan = bp.plan(env.battle, belief.state(), obs)
            tok = belief.encode(obs, None)
            pv = plan.to_vector()
            bundle, _, val, hidden, _ = pol.act(obs, tok, pv, env.get_action_mask,
                                                hidden=hidden, deterministic=det)
            fused, _enc = pol._encode_parts(obs, tok, pv)
            pre_ln = pol.value_enc_fc(fused)
            relu_ln = torch.relu(pre_ln)
            post_ln = pol.value_enc_ln(relu_ln)
            m0 = pol.value_head_mlp[0]
            mlp0_post = torch.relu(m0(post_ln))
            val_t = pol.value_head_mlp[2](mlp0_post)
            ep_acc["fused"].append(fused.detach().cpu().numpy().ravel())
            ep_acc["pre_ln"].append(pre_ln.detach().cpu().numpy().ravel())
            ep_acc["relu_ln"].append(relu_ln.detach().cpu().numpy().ravel())
            ep_acc["post_ln"].append(post_ln.detach().cpu().numpy().ravel())
            ep_acc["mlp0_post"].append(mlp0_post.detach().cpu().numpy().ravel())
            ep_acc["value"].append(np.asarray([float(val_t.item())], dtype=np.float32))
            if a.raw_obs:
                # 参考集：与第二轮 A 集同一构造（原始 obs 展平）⇒ 同源配对比较
                ep_acc["raw_obs"].append(feat_obs(obs))
            ep_val.append(float(np.asarray(val, dtype=np.float64).ravel()[0]))
            obs, r, term, trunc, info = env.step(bundle)
            ep_rew.append(float(r))
            ep_term.append(bool(term))
            ep_trunc.append(bool(trunc))
            belief.update(obs, info.get("opp_played"))
            n_frame += 1
            step += 1
            if term or trunc:
                last_val = 0.0
                if not term:
                    nb = bp.plan(env.battle, belief.state(), obs)
                    last_val = float(np.asarray(
                        pol.value(obs, belief.encode(obs, None), nb.to_vector(),
                                  hidden), dtype=np.float64).ravel()[0])
                _, ret = PPOTrainer.compute_gae(ep_rew, ep_val, ep_term, cfg.gamma,
                                                cfg.gae_lambda, truncated=ep_trunc,
                                                last_value=last_val)
                for k in keys:
                    acc[k].extend(ep_acc[k])
                Y.extend(np.asarray(ret, dtype=np.float64).tolist())
                V.extend(ep_val)
                EP.extend([ep_idx] * len(ep_rew))
                ep_lens.append(len(ep_rew))
                ep_idx += 1
                ep_rew, ep_val, ep_term, ep_trunc = [], [], [], []
                ep_acc = {k: [] for k in keys}
                obs, _ = env.reset(seed=a.seed + step)
                belief.reset(env.deck1)
                hidden = None
                if a.verbose and ep_idx % 20 == 0:
                    print(f"  [rollout] ep={ep_idx} frames={n_frame} "
                          f"{time.time() - t0:.1f}s", flush=True)
    dt = time.time() - t0
    X = {k: np.asarray(acc[k], dtype=np.float32) for k in keys}
    y = np.asarray(Y, dtype=np.float64)
    v = np.asarray(V, dtype=np.float64)
    ep = np.asarray(EP, dtype=np.int64)
    print(f"[rollout] mode={a.mode} frames={len(y)} episodes={ep_idx} "
          f"ep_len mean={np.mean(ep_lens):.1f} wall={dt:.1f}s "
          f"({len(y) / max(1e-9, dt):.1f} frames/s)", flush=True)
    if a.save_npz:
        z = {f"X_{k}": X[k] for k in keys}
        z.update({"y": y, "v": v, "ep": ep})
        np.savez_compressed(a.save_npz, **z)
        print(f"[npz] {a.save_npz}", flush=True)
    return X, y, v, ep, pol


# --------------------------------------------------------------------------
# 每层拟合（与第二轮主估计器逐位一致）
# --------------------------------------------------------------------------
def fit_layer(Xl, y, ep, m_tr, m_va, m_te, ep_va, select="within", repeats=1,
              alphas=None):
    """返回 dict：dim / EV_within(test) / val_EV_within / EV_pooled / alpha / degenerate。

    `alphas`：α 网格（默认 v2）。选中值落在网格**边界**时打 `alpha_at_boundary`
    （台账 §2 病理 ⑩：落在边界就是网格不够宽 ⇒ 该层不得用来下"无信号"结论）。
    """
    alphas = tuple(alphas) if alphas is not None else ALPHAS_V2
    Xl = np.asarray(Xl, dtype=np.float64)
    sd_tr = Xl[m_tr].std(axis=0)
    keep = sd_tr > 1e-8
    deg = (int(keep.sum()) == 0)
    out = {"dim_raw": int(Xl.shape[1]), "dim": int(keep.sum()), "degenerate": bool(deg)}
    if deg:
        # 训练集全零方差 ⇒ 最优预测 = 训练均值 = 常数 ⇒ 局内中心化后 EV_within 恒为 0
        out.update({"EV_within": 0.0, "EV_pooled": 0.0, "EV_within_raw": 0.0,
                    "val_EV_within": 0.0, "alpha": None, "note": "全零方差（平凡预测）"})
        return out
    Xk = Xl[:, keep]
    Xtr, Xva, Xte = standardize(Xk[m_tr], [Xk[m_tr], Xk[m_va], Xk[m_te]])
    ytr, yva, yte = y[m_tr], y[m_va], y[m_te]
    ew_tr = ep[m_tr]
    p_te, alpha, best_va, p_va = fit_ridge(Xtr, ytr, Xva, yva, Xte, ep_va=ep_va,
                                           select=select, alphas=alphas)
    out.update({
        "EV_within": ev_within(p_te, yte, ep[m_te]),
        "EV_pooled": ev(p_te, yte),
        "EV_within_raw": ev_within_raw(p_te, yte, ep[m_te]),
        "val_EV_within": best_va,
        "alpha": float(alpha),
        "alpha_at_boundary": bool(alpha == min(alphas)),
        "n_tr": int(m_tr.sum()), "n_va": int(m_va.sum()), "n_te": int(m_te.sum()),
    })
    return out


def layer_magstats(name, X, ep):
    """幅度阶梯（描述量，不判决）：恒定分量 / 跨帧变化 / 零变化维占比。"""
    X = np.asarray(X, dtype=np.float64)
    dstd = X.std(axis=0)
    fixed = float(np.linalg.norm(X.mean(axis=0)))
    varying = float(np.linalg.norm(dstd))
    return {"name": name, "D": int(X.shape[1]), "const_norm": fixed,
            "varying_norm": varying,
            "rel": varying / max(1e-12, fixed),
            "zero_var_dim_frac": float((dstd < 1e-9).mean()),
            "dim_std_mean": float(dstd.mean())}


def within_std_vec(X, ep):
    """‖每维局内 std 向量‖₂ —— 该表示"局内变化的总幅度"。"""
    X = np.asarray(X, dtype=np.float64)
    s = np.zeros(X.shape[1])
    for e in np.unique(ep):
        m = (ep == e)
        if m.sum() < 2:
            continue
        s += X[m].var(axis=0) * (m.sum() - 1)
    s /= max(1, len(ep) - len(np.unique(ep)))
    return float(np.sqrt(max(0.0, s.sum())))


def within_std_scalar(y, ep):
    yy = np.asarray(y, dtype=np.float64).copy()
    for e in np.unique(ep):
        m = (ep == e)
        yy[m] -= yy[m].mean()
    return float(yy.std())


# --------------------------------------------------------------------------
def main():
    _force_utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--mode", choices=["stoch", "det"], default="stoch")
    ap.add_argument("--frames", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--npz", default=None, help="从落盘 npz 离线重拟合（免重跑 rollout）")
    ap.add_argument("--save-npz", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=None)
    ap.add_argument("--verbose", action="store_true")
    # ---- v2（2026-09-14 第二轮预注册）：加原始 obs 参考集 + 扩 α 网格下界 ----
    ap.add_argument("--raw-obs", action="store_true",
                    help="额外抓原始 obs（参考集，与第二轮 A 集同构造）")
    ap.add_argument("--alpha-grid", choices=["v1", "v2"], default="v2",
                    help="主 α 网格：v1=上一轮(下界1e-1)；v2=下界1e-6")
    a = ap.parse_args()

    cfg = TrainConfig.resolve("economy")
    run_dir = a.run_dir or os.path.dirname(os.path.abspath(a.ckpt))
    cp = os.path.join(run_dir, "config.json")
    if os.path.exists(cp):
        d = json.load(open(cp, encoding="utf-8"))
        for k, vv in d.items():
            if hasattr(cfg, k):
                try:
                    setattr(cfg, k, vv)
                except Exception:
                    pass

    if a.npz:
        z = np.load(a.npz)
        X = {k.split("_", 1)[1]: z[k].astype(np.float32) for k in z.files
             if k.startswith("X_")}
        y = z["y"].astype(np.float64)
        v = z["v"].astype(np.float64)
        ep = z["ep"]
        print(f"[npz] 载入 {a.npz}: frames={len(y)} episodes={int(ep.max()) + 1}",
              flush=True)
        # 离线重拟合时仍加载 ckpt，只为描述量（max_gain_head / LN γ / 塌缩指纹）
        pol = None
        if a.ckpt:
            try:
                _e = solo_env(cfg, a.seed)
                _bd = len(BeliefInference(opp_deck=_e.deck1, n_particles=128,
                                          seed=a.seed).encode(None, None))
                pol = load_checkpoint(a.ckpt, hidden_dim=cfg.hidden_dim,
                                      plan_dim=PLAN_DIM, belief_dim=_bd)
                pol.eval()
            except Exception as exc:      # 只影响描述量，不影响判决
                print(f"[npz] ⚠️ ckpt 加载失败（描述量将缺失）: {type(exc).__name__}",
                      flush=True)
                pol = None
    else:
        device = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"
        X, y, v, ep, pol = rollout_layers(a, cfg, device)

    # ---- 局分组切分（与第二轮逐位同一算法，R9④）----
    n_ep = int(ep.max()) + 1
    rng = np.random.default_rng(a.seed)
    eps = np.unique(ep)
    rng.shuffle(eps)
    n_test = max(1, int(round(len(eps) * a.test_frac)))
    n_val = max(1, int(round(len(eps) * a.val_frac)))
    test_eps = set(eps[:n_test].tolist())
    val_eps = set(eps[n_test:n_test + n_val].tolist())
    m_te = np.array([e in test_eps for e in ep])
    m_va = np.array([e in val_eps for e in ep])
    m_tr = ~(m_te | m_va)
    ep_va = ep[m_va]
    print(f"[split] train={m_tr.sum()} val={m_va.sum()} test={m_te.sum()} "
          f"(episodes {len(eps) - n_test - n_val}/{n_val}/{n_test})", flush=True)

    # ---- 目标：按局中心化（与第二轮主跑一致）----
    y = y.copy()
    for e in np.unique(ep):
        m = (ep == e)
        y[m] -= y[m].mean()
    print(f"[target] 目标=GAE 回报按局中心化；Var(R)_within={np.var(y):.4f} "
          f"std={np.std(y):.4f}", flush=True)

    # ---- 闸门 ----
    gates = {}
    print("\n--- 闸门（不过 ⇒ L0_INVALID）---", flush=True)
    Xa = X["fused"]
    # G-UP + 主阶梯
    rows = {}
    _AG = GRIDS[a.alpha_grid]
    for label, key in LADDER:
        r = fit_layer(X[key], y, ep, m_tr, m_va, m_te, ep_va, select="within",
                      alphas=_AG)
        rows[key] = r
        rows[key]["label"] = label
        print(f"[fit] {label:14s} dim={r['dim']:5d}(raw {r['dim_raw']}) "
              f"val_EV_within={r['val_EV_within']:+.4f} | "
              f"EV_within={r['EV_within']:+.4f} EV_pooled={r['EV_pooled']:+.4f} "
              f"a={r['alpha']}" + ("  [退化]" if r["degenerate"] else ""), flush=True)

    ev_fused = rows["fused"]["EV_within"]
    gates["G-UP"] = bool(ev_fused is not None and ev_fused >= 0.15)
    print(f"  G-UP    EV_within(fused)={ev_fused:+.4f} ≥0.15 ? {gates['G-UP']}",
          flush=True)

    # G-PC 阳性对照（在 fused 上，与第二轮同构构造）
    ypc, _ = make_positive_control(Xa, ep, seed=a.seed, within_var=58.0,
                                   between_var=166.0)
    for e in np.unique(ep):
        m = (ep == e)
        ypc[m] -= ypc[m].mean()
    rpc = fit_layer(Xa, ypc, ep, m_tr, m_va, m_te, ep_va, select="within", alphas=_AG)
    gates["G-PC"] = bool(rpc["EV_within"] is not None and rpc["EV_within"] >= 0.15)
    print(f"  G-PC    EV_within(阳性对照)={rpc['EV_within']:+.4f} ≥0.15 ? "
          f"{gates['G-PC']}", flush=True)

    # G-CLK 时钟基线（单特征：局内第几帧）
    k_in_ep = np.zeros(len(ep), dtype=np.float64)
    for e in np.unique(ep):
        idx = np.where(ep == e)[0]
        k_in_ep[idx] = np.arange(len(idx))
    Xclk = k_in_ep.reshape(-1, 1).astype(np.float32)
    rclk = fit_layer(Xclk, y, ep, m_tr, m_va, m_te, ep_va, select="within", alphas=_AG)
    gates["G-CLK"] = bool(rclk["EV_within"] is not None and rclk["EV_within"] <= 0.05)
    print(f"  G-CLK   EV_within(时钟基线)={rclk['EV_within']:+.4f} ≤0.05 ? "
          f"{gates['G-CLK']}", flush=True)

    # G-SCR 状态打散（fused 与 post_ln）
    scr = {}
    for key in [k for k in ("raw_obs", "fused", "post_ln") if k in X]:
        Xs = scramble_rows_by_clock(X[key], ep, seed=a.seed)
        rs = fit_layer(Xs, y, ep, m_tr, m_va, m_te, ep_va, select="within",
                       alphas=_AG)
        scr[key] = rs["EV_within"]
        print(f"  G-SCR[{key:8s}] EV_within(打散)={rs['EV_within']:+.4f}", flush=True)
    gates["G-SCR"] = bool(all(x is not None and x <= 0.05 for x in scr.values()))
    print(f"  G-SCR   两者 ≤0.05 ? {gates['G-SCR']}", flush=True)

    gates_ok = all(gates.values())

    # ---- v2 模式（预注册 docs/value_ln_probe2_prereg_2026-09-14.md §3/§4）----
    v2_res = None
    if a.alpha_grid == "v2":
        v2_res = {}
        if "raw_obs" not in X:
            raise SystemExit("[abort] --alpha-grid v2 需要 --raw-obs（参考集）")
        print("\n--- v2：参考集与诊断重拟合 ---", flush=True)
        r_raw = fit_layer(X["raw_obs"], y, ep, m_tr, m_va, m_te, ep_va,
                          select="within", alphas=ALPHAS_V2)
        rows["raw_obs"] = dict(r_raw, label="RAW_obs")
        print(f"[fit] {'RAW_obs':14s} dim={r_raw['dim']:5d}(raw {r_raw['dim_raw']}) "
              f"val_EV_within={r_raw['val_EV_within']:+.4f} | "
              f"EV_within={r_raw['EV_within']:+.4f} a={r_raw['alpha']}"
              + ("  [边界α]" if r_raw["alpha_at_boundary"] else ""), flush=True)
        r_raw_v1 = fit_layer(X["raw_obs"], y, ep, m_tr, m_va, m_te, ep_va,
                             select="within", alphas=ALPHAS_V1)
        r_post_v1 = fit_layer(X["post_ln"], y, ep, m_tr, m_va, m_te, ep_va,
                              select="within", alphas=ALPHAS_V1)
        print(f"[diag] raw_obs  v1 网格 EV_within={r_raw_v1['EV_within']:+.4f} "
              f"(a={r_raw_v1['alpha']})  ← 与第二轮 +0.2999 的可比性检查",
              flush=True)
        print(f"[diag] post_ln  v1 网格 EV_within={r_post_v1['EV_within']:+.4f} "
              f"(a={r_post_v1['alpha']})  v2={rows['post_ln']['EV_within']:+.4f} "
              f"Δ={rows['post_ln']['EV_within'] - r_post_v1['EV_within']:+.4f}",
              flush=True)
        # ⚠️ 修正（2026-09-14 自披露）：`gates` 里仍留着 v1 的 G-UP（fused≥0.15），
        #    但 **v2 预注册 §3 的闸门集不含 G-UP**（它已被 G-RAW 取代，见【红线 R15】）。
        #    首次 v2 跑因把 G-UP 算进 gates_ok 而误判 V2_INVALID ⇒ 此处按预注册重算。
        GATE_SET_V2 = ("G-RAW", "G-PC", "G-CLK", "G-SCR", "G-VAR")
        print(f"  [gate-set] v2 权威闸门集 = {GATE_SET_V2}（G-UP 仅作 v1 遗留诊断，不计入）",
              flush=True)
        gates["G-VAR"] = bool(float(np.var(y)) > 1.0)
        gates["G-RAW"] = bool(r_raw["EV_within"] is not None
                              and r_raw["EV_within"] >= 0.15)
        print(f"  G-RAW   EV_within(raw_obs)={r_raw['EV_within']:+.4f} ≥0.15 ? "
              f"{gates['G-RAW']}", flush=True)
        print(f"  G-VAR   Var(R)_within={float(np.var(y)):.4f} >1 ? "
              f"{gates['G-VAR']}", flush=True)
        gates_ok = all(gates[k] for k in GATE_SET_V2)
        R0 = r_raw["EV_within"]
        E1, E2, E3 = (rows["pre_ln"]["EV_within"], rows["relu_ln"]["EV_within"],
                      rows["post_ln"]["EV_within"])
        E0 = rows["fused"]["EV_within"]
        hits = []
        if R0 is not None and E0 is not None and R0 >= 2.0 * E0:
            hits.append("V-LOSS-FRONT")
        if E1 is not None and E2 is not None and E1 >= 0.05 and E2 < 0.05:
            hits.append("V-LOSS-RELU")
        if (E2 is not None and E3 is not None and E2 >= 0.05 and E3 < 0.05
                and not rows["post_ln"].get("alpha_at_boundary")):
            hits.append("V-LOSS-LN")
        if E3 is not None and E3 >= 0.05:
            hits.append("V-NO-LOSS-IN-BRANCH")
        if (E3 is not None and E3 >= 0.05
                and not (R0 is not None and E0 is not None and R0 >= 2.0 * E0)):
            hits.append("V-NO-LOSS-AT-ALL")
        if not gates_ok:
            v2_verdict = "V2_INVALID"
        elif not hits:
            v2_verdict = "NOT_PREREGISTERED"
        else:
            v2_verdict = "+".join(hits)
        print("\n=== v2 分支（预注册 §4，逐条报命中）===", flush=True)
        print(f"  R0=EV_within(raw_obs)={R0:+.4f}  E0(fused)={E0:+.4f}  "
              f"E1(pre_ln)={E1:+.4f}  E2(relu_ln)={E2:+.4f}  E3(post_ln)={E3:+.4f}",
              flush=True)
        print(f"  V-LOSS-FRONT: R0 ≥ 2×E0 ? {R0 >= 2.0 * E0} "
              f"({R0:.4f} vs {2.0 * E0:.4f})", flush=True)
        print(f"  V-LOSS-RELU : E1≥0.05 且 E2<0.05 ? "
              f"{E1 >= 0.05 and E2 < 0.05}", flush=True)
        print(f"  V-LOSS-LN   : E2≥0.05 且 E3<0.05 ? "
              f"{E2 >= 0.05 and E3 < 0.05}"
              + ("  [作废：post_ln α 在边界]" if rows["post_ln"].get(
                  "alpha_at_boundary") else ""), flush=True)
        print(f"  V-NO-LOSS-IN-BRANCH: E3≥0.05 ? {E3 >= 0.05}", flush=True)
        print(f"  VERDICT(v2) = {v2_verdict}", flush=True)
        v2_res.update({"R0_raw": R0, "E0_fused": E0, "E1_pre": E1,
                       "E2_relu": E2, "E3_post": E3, "hits": hits,
                       "verdict": v2_verdict,
                       "raw_v1_grid_EV_within": r_raw_v1["EV_within"],
                       "raw_v1_grid_alpha": r_raw_v1["alpha"],
                       "post_v1_grid_EV_within": r_post_v1["EV_within"],
                       "post_v1_grid_alpha": r_post_v1["alpha"]})

    # ---- 判决：首个 EV_within < 0.05 的层 ----
    evs = [rows[k]["EV_within"] for _, k in LADDER]
    kstar = None
    for i, e in enumerate(evs):
        if e is not None and e < 0.05:
            kstar = i
            break
    if v2_res is not None:
        verdict = v2_res["verdict"]
    elif not gates_ok:
        verdict = "L0_INVALID"
    elif kstar == 0:
        verdict = "L4_INVALID"
    elif kstar == 1:
        verdict = "L5_LINEAR_PROJ"
    elif kstar == 2:
        verdict = "L3_RELU"
    elif kstar == 3:
        verdict = "L1_LAYERNORM"
    elif kstar == 4:
        verdict = "L2b_MLP0_DEAD"
    elif kstar == 5:
        verdict = "L2_OUTPUT"
    else:
        verdict = ("L2_NOT_LN" if (evs[3] is not None and evs[3] >= 0.15)
                   else "L1p_WEAK")

    print("\n=== 判据（照预注册 §3 读）===", flush=True)
    print("  阶梯 EV_within: " + "  ".join(
        f"{lab}={e if e is None else round(e, 4)}" for (lab, _), e in zip(LADDER, evs)),
        flush=True)
    print(f"  闸门: " + " ".join(f"{k}={'PASS' if x else 'FAIL'}"
                                 for k, x in gates.items())
          + f"  ⇒ all={gates_ok}", flush=True)
    print(f"  首个 EV_within<0.05 的层 k*={kstar}"
          + (f" ({LADDER[kstar][0]})" if kstar is not None else " (不存在)"), flush=True)
    print(f"  VERDICT = {verdict}", flush=True)

    # ---- 描述量（不判决）----
    print("\n--- 描述量（不参与判决，见预注册 §4）---", flush=True)
    ws_R = within_std_scalar(y, ep)
    desc = {"within_std_R": ws_R}
    for label, key in LADDER:
        st = layer_magstats(label, X[key], ep)
        wv = within_std_vec(X[key], ep)
        gn = ws_R / max(1e-12, wv)
        st.update({"within_std_vec": wv, "gain_needed": gn})
        desc[key] = st
        print(f"  {label:14s} D={st['D']:5d} ‖恒定‖={st['const_norm']:10.4f} "
              f"‖跨帧变化‖={st['varying_norm']:9.5f} 相对={st['rel']:.5f} "
              f"零变化维={st['zero_var_dim_frac']:.3f} "
              f"within_std_vec={wv:9.5f} gain_needed={gn:12.2f}", flush=True)
    if pol is not None:
        import torch as _t
        W1 = pol.value_head_mlp[0].weight.detach().float().cpu().numpy()
        W2 = pol.value_head_mlp[2].weight.detach().float().cpu().numpy()
        s1 = float(np.linalg.svd(W1, compute_uv=False)[0])
        s2 = float(np.linalg.svd(W2, compute_uv=False)[0])
        mg = s1 * s2
        gn_post = desc["post_ln"]["gain_needed"]
        print(f"  max_gain_head = σmax(W1)*σmax(W2) = {s1:.4f}*{s2:.4f} = {mg:.4f}",
              flush=True)
        print(f"  post_ln gain_needed={gn_post:.2f} vs max_gain_head={mg:.4f} "
              f"⇒ 缺口 {gn_post / max(1e-12, mg):.1f}×", flush=True)
        desc["max_gain_head"] = mg
        desc["gain_shortfall_post_ln"] = gn_post / max(1e-12, mg)
        desc["ln_weight_mean"] = float(pol.value_enc_ln.weight.detach().float()
                                       .cpu().mean())
        desc["ln_weight_norm"] = float(pol.value_enc_ln.weight.detach().float()
                                       .cpu().norm())
        desc["ln_bias_norm"] = float(pol.value_enc_ln.bias.detach().float()
                                     .cpu().norm())
        rl = X["relu_ln"]
        m0p = X["mlp0_post"]
        print(f"  LN γ: mean={desc['ln_weight_mean']:+.5f} "
              f"‖γ‖={desc['ln_weight_norm']:.5f} ‖β‖={desc['ln_bias_norm']:.5f}",
              flush=True)
        print(f"  relu_ln 逐帧存活率={(rl > 0).mean():.4f}  "
              f"存活维占比={(rl > 0).any(axis=0).mean():.4f}", flush=True)
        print(f"  mlp0_post 逐帧存活率={(m0p > 0).mean():.6f}  "
              f"全零帧占比={float((m0p == 0).all(axis=1).mean()):.4f}", flush=True)
        V = X["value"].ravel()
        print(f"  value 唯一值={len(np.unique(V))}/{len(V)} std={V.std():.3e} "
              f"EV_within={rows['value']['EV_within']:+.6f} "
              f"std(V)/std(R)={V.std() / max(1e-12, np.std(y)):.6f}", flush=True)
        desc["value_unique"] = int(len(np.unique(V)))
        desc["value_std"] = float(V.std())

    res = {"ckpt": a.ckpt, "tag": a.tag, "mode": a.mode, "seed": a.seed,
           "frames": int(len(y)), "episodes": int(n_ep),
           "test_frac": a.test_frac, "val_frac": a.val_frac,
           "gates": gates, "gates_ok": bool(gates_ok),
           "ev_within_ladder": {lab: e for (lab, _), e in zip(LADDER, evs)},
           "kstar": kstar, "verdict": verdict,
           "rows": rows, "controls": {"poscontrol": rpc, "clockbase": rclk,
                                      "scramble": scr},
           "descriptive": desc,
           "critic": {"EV_within": rows["value"]["EV_within"],
                      "EV_pooled": rows["value"]["EV_pooled"],
                      "std_ratio": float(X["value"].std() / max(1e-12, np.std(y)))},
           "alpha_grid": a.alpha_grid, "raw_obs": bool(a.raw_obs),
           "v2": v2_res,
           "gate_set_v2": list(GATE_SET_V2) if v2_res is not None else None,
           "note_gup_legacy": ("G-UP 是 v1 遗留，v2 判决不计入（预注册 §3）"
                               if v2_res is not None else None),
           "n_prereg": ("docs/value_ln_probe2_prereg_2026-09-14.md"
                        if a.alpha_grid == "v2"
                        else "docs/value_ln_probe_prereg_2026-09-14.md")}
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        print(f"[out] {a.out}", flush=True)
    return res


if __name__ == "__main__":
    main()
