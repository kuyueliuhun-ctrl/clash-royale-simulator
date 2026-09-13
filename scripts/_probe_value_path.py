"""只读取证：独立价值通路逐层方差解剖（不写盘、不改训练）。

在真实 rollout 帧上逐帧测 value 通路的每一层：
  fused → value_enc_fc → ReLU → value_enc_ln → [MLP0 → ReLU] → MLP2 → value
报每层「跨帧逐维 std」「恒定分量」「ReLU 存活率」，并同帧算 GAE 回报 R 的
局内/局间方差，给出 EV 分解。

用法（在 src/clasher_new 下）：
  PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe \
      ../../scripts/_probe_value_path.py --ckpt runs/d1_long_100k/solo_main_100000.pt --frames 900
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


def layer_stats(name, X):
    """X: (T, D)。返回一行描述。"""
    X = np.asarray(X, dtype=np.float64)
    dstd = X.std(axis=0)
    fixed = np.linalg.norm(X.mean(axis=0))
    varying = np.linalg.norm(dstd)
    rel = varying / max(1e-12, fixed)
    frac_dead = float((dstd < 1e-9).mean())
    return (f"  {name:14s} D={X.shape[1]:5d} |均值|={np.abs(X).mean():9.5f} "
            f"‖恒定‖={fixed:9.4f} ‖跨帧变化‖={varying:9.5f} 相对变化={rel:.5f} "
            f"逐维std mean={dstd.mean():.6f} 零变化维占比={frac_dead:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--frames", type=int, default=900)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tag", default="")
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
    bdim = len(BeliefInference(opp_deck=env.deck1, n_particles=128,
                               seed=a.seed).encode(None, None))
    pol = load_checkpoint(a.ckpt, hidden_dim=cfg.hidden_dim, plan_dim=PLAN_DIM,
                          belief_dim=bdim)
    pol.to_device(device).eval()
    print(f"=== {a.tag or os.path.basename(a.ckpt)}  "
          f"independent={pol.value_independent} bypass={pol.value_bypass} "
          f"γ={cfg.gamma} λ={cfg.gae_lambda} ===")
    # 参数表
    for nm, p in pol.named_parameters():
        if "value" in nm:
            print(f"  [param] {nm:26s} {tuple(p.shape)} ‖·‖={p.detach().float().norm().item():.6f}"
                  + (f" mean={p.detach().float().mean().item():+.6f}"
                     if p.dim() == 1 else ""))

    _, _ = resolve_deck_set(getattr(cfg, "deck_set", None) or "default")
    bp = BeliefPlanner()
    opp = FollowerPolicy(hidden=cfg.hidden_dim, plan_dim=PLAN_DIM, belief_dim=bdim,
                         value_bypass=bool(pol.value_bypass),
                         value_independent=bool(pol.value_independent))
    opp.to_device(device)
    opp.load_state_dict(pol.state_dict())
    env.opponent = FollowerOpponent(opp, env, belief=BeliefInference(
        opp_deck=env.deck1, n_particles=128, seed=a.seed + 1), deterministic=True)

    belief = BeliefInference(opp_deck=env.deck1, n_particles=128, seed=a.seed)
    obs, _ = env.reset(seed=a.seed)
    belief.reset(env.deck1)
    hidden = None

    buf = {k: [] for k in ("fused", "pre_ln", "relu_ln", "post_ln", "mlp0_pre",
                           "mlp0_post", "mlp2_pre", "value", "enc", "h", "gru_pre")}
    per_ep = []
    ep_rew, ep_val, ep_term, ep_trunc = [], [], [], []
    with torch.no_grad():
        for t in range(a.frames):
            plan = bp.plan(env.battle, belief.state(), obs)
            tok = belief.encode(obs, None)
            pv = plan.to_vector()
            fused, enc = pol._encode_parts(obs, tok, pv)
            h = pol.gru_cell(enc, (torch.zeros(1, cfg.hidden_dim, device=device)
                                   if hidden is None else hidden.detach()))
            hidden = h
            # GRU 候选 pre-activation
            gc = pol.gru_cell
            gi = enc @ gc.weight_ih.t() + gc.bias_ih
            gh = h @ gc.weight_hh.t() + gc.bias_hh
            Hd = gc.hidden_size
            n_pre = gi[:, 2 * Hd:] + torch.sigmoid(
                gi[:, :Hd] + gh[:, :Hd]) * gh[:, 2 * Hd:]

            pre_ln = pol.value_enc_fc(fused)
            relu_ln = torch.relu(pre_ln)
            post_ln = pol.value_enc_ln(relu_ln)
            if pol.value_independent:
                m0 = pol.value_head_mlp[0]
                mlp0_pre = m0(post_ln)
                mlp0_post = torch.relu(mlp0_pre)
                val_t = pol.value_head_mlp[2](mlp0_post)
                buf["mlp2_pre"].append(mlp0_post.detach().cpu().numpy().ravel())
            else:
                mlp0_pre = mlp0_post = post_ln
                val_t = pol._value_from(enc, h, fused)
            buf["fused"].append(fused.detach().cpu().numpy().ravel())
            buf["pre_ln"].append(pre_ln.detach().cpu().numpy().ravel())
            buf["relu_ln"].append(relu_ln.detach().cpu().numpy().ravel())
            buf["post_ln"].append(post_ln.detach().cpu().numpy().ravel())
            buf["mlp0_pre"].append(mlp0_pre.detach().cpu().numpy().ravel())
            buf["mlp0_post"].append(mlp0_post.detach().cpu().numpy().ravel())
            buf["value"].append(float(val_t.item()))
            buf["enc"].append(enc.detach().cpu().numpy().ravel())
            buf["h"].append(h.detach().cpu().numpy().ravel())
            buf["gru_pre"].append(n_pre.detach().cpu().numpy().ravel())

            ep_rew.append(0.0)
            ep_val.append(float(val_t.item()))
            ep_term.append(False)
            ep_trunc.append(False)
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
                                         bp.plan(env.battle, belief.state(),
                                                 obs).to_vector(), hidden)
                _, ret = PPOTrainer.compute_gae(ep_rew, ep_val, ep_term, cfg.gamma,
                                                cfg.gae_lambda, truncated=ep_trunc,
                                                last_value=last_val)
                per_ep.append({"R": np.asarray(ret, dtype=np.float64),
                               "V": np.asarray(ep_val, dtype=np.float64)})
                obs, _ = env.reset(seed=a.seed + t)
                belief.reset(env.deck1)
                hidden = None
                ep_rew, ep_val, ep_term, ep_trunc = [], [], [], []

    print("\n--- 逐层（跨帧，%d 帧 / %d 局）---" % (len(buf["value"]), len(per_ep)))
    for k in ("fused", "pre_ln", "relu_ln", "post_ln", "mlp0_pre", "mlp0_post",
              "enc", "h", "gru_pre"):
        print(layer_stats(k, buf[k]))
    V = np.asarray(buf["value"], dtype=np.float64)
    print(f"  {'value':14s} mean={V.mean():+.6f} std={V.std():.3e} "
          f"min={V.min():+.6f} max={V.max():+.6f} 唯一值={len(np.unique(V))}/{len(V)}")
    if pol.value_independent:
        print(f"  {'MLP2 bias':14s} = {pol.value_head_mlp[2].bias.item():+.8f}  "
              f"(value 与 bias 相等 ⇒ MLP0 全零)")
    refl = np.asarray(buf["relu_ln"])
    print(f"  ReLU(value_enc) 存活维占比 = {(refl > 0).any(axis=0).mean():.3f}  "
          f"逐帧存活率 mean={(refl > 0).mean():.4f}")
    m0p = np.asarray(buf["mlp0_pre"])
    m0o = np.asarray(buf["mlp0_post"])
    print(f"  MLP0 ReLU 逐帧存活率 mean={(m0o > 0).mean():.6f}  "
          f"全零帧占比={float((m0o == 0).all(axis=1).mean()):.3f}  "
          f"pre 跨帧每维std mean={m0p.std(axis=0).mean():.6e} "
          f"pre |值|均值={np.abs(m0p).mean():.6e}")

    R_all = np.concatenate([e["R"] for e in per_ep])
    V_all = np.concatenate([e["V"] for e in per_ep])
    n = min(len(R_all), len(V_all))
    R_all, V_all = R_all[:n], V_all[:n]
    ep_means = np.asarray([e["R"].mean() for e in per_ep])
    within = np.concatenate([e["R"] - e["R"].mean() for e in per_ep])
    mse = float(((V_all - R_all) ** 2).mean())
    var = float(R_all.var())
    mse_ep = float(np.mean([((e["R"].mean() - e["R"]) ** 2).mean() for e in per_ep]))
    print("\n--- 回报 / EV ---")
    print(f"  R 全网 std={R_all.std():.5f} mean={R_all.mean():+.4f}")
    print(f"  局间 std(局均值)={ep_means.std():.5f}  局内 std(R−局均值)={within.std():.5f}")
    print(f"  MSE(v,R)={mse:.4f}  Var(R)={var:.4f}  EV(池化)={1 - mse / max(1e-12, var):+.4f}")
    print(f"  MSE(局均值常数预测)={mse_ep:.4f}  EV(局均值预测)={1 - mse_ep / max(1e-12, var):+.4f}")
    print(f"  vstd/rstd(池化) = {V_all.std() / max(1e-12, R_all.std()):.6f}")
    print(f"  vstd/局内rstd   = {V_all.std() / max(1e-12, within.std()):.6f}")
    if V_all.std() > 1e-12:
        print(f"  corr(V,R) = {np.corrcoef(V_all, R_all)[0, 1]:+.4f}")
    print(f"  局内 MSE(v,R−局均值) = {float(np.mean([((e['V'] - e['R']) ** 2).mean() for e in per_ep])):.4f}")


if __name__ == "__main__":
    main()
