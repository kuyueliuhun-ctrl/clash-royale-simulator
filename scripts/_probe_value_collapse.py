"""只读取证：value 通路逐层方差解剖（不写盘、不改训练）。

回答「value 输出方差塌到 rstd 的 0.1%」发生在哪一层：
对 ckpt 在真实 rollout 帧上逐帧测
  fused(各分量) → value_enc_fc(relu) → value_enc_ln → ReLU → value_head_mlp
以及同局 GAE 回报 R 的跨帧 std / 每局内部 std，算同帧 EV。

用法：
  PYTHONIOENCODING=utf-8 python scripts/_probe_value_collapse.py \
      --ckpt src/clasher_new/runs/d1_long_100k/solo_main_100000.pt --frames 900
"""

import argparse
import json
import os
import sys

# T1-1b 补齐（2026-09-19）：原先这里是**手写**的 UTF-8 兜底块（只处理 stdout、且不处理 stderr
# ⇒ traceback 在 GBK 下仍是乱码）。现收敛到 T1-1 的**单一实现**（两路 + errors='replace'）。
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                                 "src", "clasher_new"))
from rl.io_bootstrap import force_utf8_stdout  # noqa: E402
force_utf8_stdout()

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src", "clasher_new")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

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



def measurement_line(pol, L):
    """MLP 第 0 层 pre-activation 的跨帧统计（在 python 侧用 L 的 post 重算）。"""
    import numpy as _np
    W = pol.value_head_mlp[0].weight.detach().cpu().numpy()
    b = pol.value_head_mlp[0].bias.detach().cpu().numpy()
    PRE = L["v_enc_post"] @ W.T + b
    frac_pos = float((PRE > 0).mean())
    print(f"    pre 跨帧 每维std mean={PRE.std(axis=0).mean():.6f} |pre|均值={_np.abs(PRE).mean():.6f} "
          f"pos占比={frac_pos:.4f}")
    print(f"    LayerNorm 前(pre-LN) 跨帧 每维std mean={L['v_enc_pre'].std(axis=0).mean():.6f} "
          f"|值|均值={_np.abs(L['v_enc_pre']).mean():.6f}")
    dead = int((L["v_hidden"] <= 0).all(axis=0).sum())
    print(f"    ReLU 全零维数 = {dead}/{L['v_hidden'].shape[1]}   "
          f"hidden 逐帧 L1 均值={_np.abs(L['v_hidden']).mean():.6f}")
    return ""


def _std(x):
    return float(np.asarray(x, dtype=np.float64).std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--frames", type=int, default=900)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    cfg = TrainConfig.resolve("economy")
    run_dir = a.run_dir or os.path.dirname(os.path.abspath(a.ckpt))
    cp = os.path.join(run_dir, "config.json")
    if os.path.exists(cp):
        d = json.load(open(cp, encoding="utf-8"))
        for k, v in d.items():
            if hasattr(cfg, k):
                try:
                    setattr(cfg, k, v)
                except Exception:
                    pass
    device = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"

    env = solo_env(cfg, a.seed)
    bdim = len(BeliefInference(opp_deck=env.deck1, n_particles=128, seed=a.seed).encode(None, None))
    pol = load_checkpoint(a.ckpt, hidden_dim=cfg.hidden_dim, plan_dim=PLAN_DIM,
                          belief_dim=bdim)
    pol.to_device(device).eval()
    print(f"[probe] ckpt={os.path.basename(a.ckpt)} value_independent="
          f"{getattr(pol, 'value_independent', False)} value_bypass="
          f"{getattr(pol, 'value_bypass', False)} gamma={cfg.gamma} lam={cfg.gae_lambda}")

    mirror_deck, _ = resolve_deck_set(getattr(cfg, "deck_set", None) or "default")
    bp = BeliefPlanner()
    opp = FollowerPolicy(hidden=cfg.hidden_dim, plan_dim=PLAN_DIM, belief_dim=bdim,
                         value_bypass=bool(getattr(pol, "value_bypass", False)),
                         value_independent=bool(getattr(pol, "value_independent", False)))
    opp.to_device(device)
    opp.load_state_dict(pol.state_dict())
    env.opponent = FollowerOpponent(opp, env, belief=BeliefInference(
        opp_deck=env.deck1, n_particles=128, seed=a.seed + 1), deterministic=True)

    belief = BeliefInference(opp_deck=env.deck1, n_particles=128, seed=a.seed)
    obs, _ = env.reset(seed=a.seed)
    belief.reset(env.deck1)
    hidden = None

    layers = {"fused": [], "v_enc_pre": [], "v_enc_post": [], "v_hidden": [], "value": [],
              "enc": [], "h": []}
    per_ep = []          # 每局 (R 序列, value 序列)
    ep_obs, ep_bel, ep_plan, ep_rew, ep_val, ep_term, ep_trunc = [], [], [], [], [], [], []
    with torch.no_grad():
        for t in range(a.frames):
            plan = bp.plan(env.battle, belief.state(), obs)
            tok = belief.encode(obs, None)
            pv = plan.to_vector()
            fused, enc = pol._encode_parts(obs, tok, pv)
            h = pol.gru_cell(enc, (torch.zeros(1, cfg.hidden_dim, device=device)
                                   if hidden is None else hidden.detach()))
            hidden = h
            val = float(pol._value_from(enc, h, fused).item())
            # 逐层
            pre = pol.value_enc_fc(fused)
            post = pol.value_enc_ln(torch.relu(pre))
            hid = torch.relu(pol.value_head_mlp[0](post))   # MLP 隐层 (.,64)
            layers["fused"].append(fused.detach().cpu().numpy().ravel())
            layers["v_enc_pre"].append(pre.detach().cpu().numpy().ravel())
            layers["v_enc_post"].append(post.detach().cpu().numpy().ravel())
            layers["v_hidden"].append(hid.detach().cpu().numpy().ravel())
            layers["value"].append(val)
            layers["enc"].append(enc.detach().cpu().numpy().ravel())
            layers["h"].append(h.detach().cpu().numpy().ravel())

            ep_obs.append(obs); ep_bel.append(tok); ep_plan.append(pv)
            ep_rew.append(0.0); ep_val.append(val)
            ep_term.append(False); ep_trunc.append(False)

            bundle, _, _, _, _ = pol.act(obs, tok, pv, env.get_action_mask,
                                         hidden=hidden, deterministic=True)
            obs, r, term, trunc, info = env.step(bundle)
            ep_rew[-1] = float(r)
            ep_term[-1] = bool(term)
            ep_trunc[-1] = bool(trunc)
            belief.update(obs, info.get("opp_played"))
            if term or trunc:
                last_val = 0.0
                if not term:
                    last_val = pol.value(obs, belief.encode(obs, None),
                                         bp.plan(env.battle, belief.state(), obs).to_vector(),
                                         hidden)
                adv, ret = PPOTrainer.compute_gae(
                    ep_rew, ep_val, ep_term, cfg.gamma, cfg.gae_lambda,
                    truncated=ep_trunc, last_value=last_val)
                per_ep.append({"R": np.asarray(ret, dtype=np.float64),
                               "V": np.asarray(ep_val, dtype=np.float64),
                               "len": len(ep_rew), "rew_last": ep_rew[-1]})
                obs, _ = env.reset(seed=a.seed + t)
                belief.reset(env.deck1); hidden = None
                ep_obs, ep_bel, ep_plan, ep_rew, ep_val, ep_term, ep_trunc = \
                    [], [], [], [], [], [], []

    L = {k: np.asarray(v) for k, v in layers.items()}
    V = L["value"]
    print(f"\n=== 跨帧逐层（{len(V)} 帧，{len(per_ep)} 局）===")
    for k in ("fused", "v_enc_pre", "v_enc_post", "v_hidden", "enc", "h", "value"):
        X = L[k]
        if X.ndim == 1:
            X = X[:, None]
        dm = X.std(axis=0)
        nrm = np.linalg.norm(X, axis=1)
        print(f"  {k:12s} dim={X.shape[1]:5d} ||x|| mean={nrm.mean():9.4f} "
              f"std={nrm.std():7.4f} | 每维跨帧std mean={dm.mean():.6f} "
              f"max={dm.max():.6f} | 相对变化={dm.mean()/max(1e-12, np.abs(X).mean()):.5f}")
    print(f"  value       mean={V.mean():+.5f} std={V.std():.6f} "
          f"min={V.min():+.5f} max={V.max():+.5f}")
    print(f"  value 唯一值数 = {len(np.unique(V))} / {len(V)}  (逐帧|Δ|max={np.abs(np.diff(V)).max():.3e})")
    print(f"  value 末层 bias = {float(pol.value_head_mlp[2].bias.item()):+.8f}"
          f"  → {'= bias（hidden 逐帧恒定）' if len(np.unique(V))==1 else '有逐帧变化'}")
    # —— MLP 第 0 层的 pre-activation（判断 ReLU 是否死）——
    for _c in range(len(L["v_hidden"])):
        pass
    print("\n  [MLP-0 pre-activation] " + measurement_line(pol, L))
    # 直接复算「末层权重 · 隐层」的跨帧变化，定位方差死在 ReLU 前还是后
    wn = pol.value_head_mlp[2].weight.detach().cpu().numpy().ravel()   # (64,)
    hv = L["v_hidden"] - L["v_hidden"].mean(axis=0, keepdims=True)
    print(f"  [dbg] v_hidden {L['v_hidden'].shape} wn {wn.shape}")
    proj = hv @ wn
    print(f"  Σ w·(hidden−mean) 跨帧 std = {proj.std():.3e}   (末层 ||w||={np.linalg.norm(wn):.5f})")
    hid_fixed = np.linalg.norm(L["v_hidden"].mean(axis=0))
    hid_var = np.linalg.norm(hv.std(axis=0))
    print(f"  hidden 恒定分量 ||mean||={hid_fixed:.4f} vs 跨帧变化 ||std||={hid_var:.4f} "
          f"→ 信噪比={hid_var / max(1e-12, hid_fixed):.5f}")
    cw = hv.std(axis=0)
    top = np.argsort(-cw)[:5]
    print("  hidden 跨帧std top5: " + " ".join(
        f"d{int(i)}(std={cw[i]:.4f},w={wn[i]:+.4f})" for i in top))
    print(f"  Σ_k |std_k·w_k| 上界 = {np.abs(cw * wn).sum():.3e} vs 实测 proj.std={proj.std():.3e}")
    U, S, Vt = np.linalg.svd(hv, full_matrices=False)
    v1 = Vt[0]
    wdotv1 = float(wn @ v1)
    print(f"  SVD(hv) 前3奇异值 = {S[0]:.4f}/{S[1]:.4f}/{S[2]:.4f} (Σ={S.sum():.4f})  "
          f"|cos(w,PC1)|={abs(wdotv1) / max(1e-12, np.linalg.norm(wn)):.5f}")
    hmean_dir = L["v_hidden"].mean(axis=0) / max(1e-12, hid_fixed)
    print(f"  |cos(w, hidden 均值方向)|={abs(float(wn @ hmean_dir)) / max(1e-12, np.linalg.norm(wn)):.5f}")
    # 对照：同帧的其余 value 通路（不参与前向，仅看它们"看得见"多少状态变化）
    with torch.no_grad():
        h_t = torch.as_tensor(L["h"], dtype=torch.float32).to(device)
        alt_shared = pol.value_head(h_t).squeeze(-1).double().cpu().numpy()
    print(f"  [对照] 未启用的 value_head(h)  std={alt_shared.std():.3e} "
          f"||w||={pol.value_head.weight.norm().item():.4f}（该头在 100k 内逐位未变，见 ckpt diff）")
    print(f"  [对照] enc 跨帧 每维std mean={L['enc'].std(axis=0).mean():.6f} "
          f"h 跨帧 每维std mean={L['h'].std(axis=0).mean():.6f}")
    # LayerNorm 前/后：共同模（跨维同向）占比
    for nm in ("v_enc_pre", "v_enc_post"):
        X = L[nm]
        common = X.mean(axis=1)                      # 每帧的跨维均值 = 共同模
        resid = X - common[:, None]
        print(f"  {nm:11s} 跨帧 std(跨维均值)={common.std():.6e} "
              f"std(去共同模残差,每维均值)={resid.std(axis=0).mean():.6e}")

    R_all = np.concatenate([e["R"] for e in per_ep])
    V_all = np.concatenate([e["V"] for e in per_ep])
    n = min(len(R_all), len(V_all))
    R_all, V_all = R_all[:n], V_all[:n]
    print(f"\n=== 回报（同帧 GAE，γ={cfg.gamma} λ={cfg.gae_lambda}）===")
    print(f"  R 全网 std={R_all.std():.5f}   mean={R_all.mean():+.4f}  "
          f"min={R_all.min():+.3f} max={R_all.max():+.3f}")
    ep_means = np.asarray([e["R"].mean() for e in per_ep])
    within = np.concatenate([e["R"] - e["R"].mean() for e in per_ep])
    print(f"  局间 std(每局均值)={ep_means.std():.5f}   局内 std(R-局均值)={within.std():.5f}")
    print(f"  → 回报方差分解：局间 {ep_means.std()**2:.3f} vs 局内 {within.std()**2:.4f}")
    mse = float(((V_all - R_all) ** 2).mean())
    var = float(R_all.var())
    # 每局「常数预测局均值」的 MSE
    mse_const = float(np.mean([((e["R"].mean() - e["R"]) ** 2).mean() for e in per_ep]))
    print(f"  EV(同帧池化) = {1 - mse / max(1e-12, var):+.4f}   (mse={mse:.4f} var={var:.4f})")
    print(f"  EV(局均值常数预测) = {1 - mse_const / max(1e-12, var):+.4f} (mse={mse_const:.4f})")
    print(f"  value 输出 std / 全网 R std = {V_all.std() / max(1e-12, R_all.std()):.6f}")
    print(f"  value 输出 std / 局内 R std = {V_all.std() / max(1e-12, within.std()):.6f}")
    print(f"  相关性 corr(V,R) = {np.corrcoef(V_all, R_all)[0,1]:+.4f}")

    # 逐局：value 是否只是在预测局均值
    if per_ep:
        cs = [np.corrcoef(e["V"], e["R"])[0, 1] for e in per_ep
              if e["len"] > 3 and e["R"].std() > 1e-9 and e["V"].std() > 1e-9]
        print(f"  逐局 corr(V,R)（std>0 的局，n={len(cs)}）: mean="
              f"{np.mean(cs) if cs else float('nan'):+.4f}")
        print("  抽样 6 局: " + " ".join(
            f"[len={e['len']} Rmean={e['R'].mean():+.2f} Rstd={e['R'].std():.2f} "
            f"Vstd={e['V'].std():.5f}]" for e in per_ep[:6]))
    print("\n[注] 局内 std 才是 critic 真正要解释的信号；vstd/rstd 的门槛分母用的是"
          "含局间差异的池化 rstd，量纲不可比。")


if __name__ == "__main__":
    main()
