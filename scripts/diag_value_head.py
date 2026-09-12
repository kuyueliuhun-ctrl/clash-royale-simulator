"""价值头解剖（2026-09-11 §4 诊断第二步）。

回答：critic 输出近乎常数（743 帧 std=0.028），是
  (a) value_head 权重塌成 ~0（死头），还是
  (b) 输入 h（GRU 隐状态）本身跨状态近乎恒定，或
  (c) 两者皆有。

做法：加载 ckpt，打印各头参数量级；rollout 若干帧收集 h 与 value_head(h)，
报每维 h 的跨状态标准差、value_head 输出分布、权重范数与偏置。
"""

import argparse
import os
import sys

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--run-dir", default="runs/prod_200k_valnorm_ev")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    cfg = TrainConfig.resolve("economy")
    import json
    p = os.path.join(a.run_dir, "config.json")
    if os.path.exists(p):
        d = json.load(open(p, encoding="utf-8"))
        for k, v in d.items():
            if hasattr(cfg, k):
                try:
                    setattr(cfg, k, v)
                except Exception:
                    pass

    device = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"
    env = solo_env(cfg, 0)
    bdim = len(BeliefInference(opp_deck=env.deck1, n_particles=128, seed=0).encode(None, None))
    pol = load_checkpoint(a.ckpt, hidden_dim=cfg.hidden_dim, plan_dim=PLAN_DIM,
                          belief_dim=bdim)
    pol.to_device(device)

    print("=== 参数范数（按模块）===")
    groups = {}
    for name, prm in pol.named_parameters():
        mod = name.split(".")[0]
        groups.setdefault(mod, [0.0, 0, 0.0])
        g = groups[mod]
        g[0] += float((prm.detach() ** 2).sum().item())   # sum of squares
        g[1] += prm.numel()
        g[2] = max(g[2], float(prm.detach().abs().max().item()))
    for mod, (ss, n, mx) in sorted(groups.items(), key=lambda x: -x[1][1]):
        print(f"  {mod:14s} numel={n:8d}  ||W||={np.sqrt(ss):10.4f}  RMS={np.sqrt(ss/max(n,1)):.5f}  max|W|={mx:.4f}")

    # value_head 细节
    vw = pol.value_head.weight.detach().cpu().numpy().ravel()
    vb = pol.value_head.bias.detach().cpu().numpy().ravel()
    print(f"\n=== value_head ===\n  weight: dim={vw.size} ||w||={np.linalg.norm(vw):.6f} "
          f"rms={np.sqrt((vw**2).mean()):.6f} max|w|={np.abs(vw).max():.6f}")
    print(f"  bias={vb.item():.6f}")

    # rollout 收集 h 与 value
    mirror_deck, _ = resolve_deck_set(getattr(cfg, "deck_set", None) or "default")
    bp = BeliefPlanner()
    # B'/E'：镜像对手必须用与 ckpt 相同的 value 架构（否则键集不匹配）
    opp = FollowerPolicy(hidden=cfg.hidden_dim, plan_dim=PLAN_DIM, belief_dim=bdim,
                         value_bypass=bool(getattr(pol, "value_bypass", False)),
                         value_independent=bool(getattr(pol, "value_independent", False)))
    opp.to_device(device); opp.load_state_dict(pol.state_dict())
    opp_side = FollowerOpponent(opp, env, belief=BeliefInference(opp_deck=env.deck1,
                                n_particles=128, seed=7), deterministic=True)
    env.opponent = opp_side
    belief = BeliefInference(opp_deck=env.deck1, n_particles=128, seed=7)
    obs, _ = env.reset(seed=7)
    belief.reset(env.deck1)
    hidden = None
    H, VV, ENC = [], [], []
    with torch.no_grad():
        for t in range(a.frames):
            plan = bp.plan(env.battle, belief.state(), obs)
            tok = belief.encode(obs, None)
            fused, enc = pol._encode_parts(obs, tok, plan.to_vector())
            h = pol.gru_cell(enc, (torch.zeros(1, cfg.hidden_dim, device=device)
                                   if hidden is None else hidden.detach()))
            hidden = h
            # B'/E'（2026-09-12）：value 走策略真实通路（统一入口 _value_from）
            val = float(pol._value_from(enc, h, fused).item())
            H.append(h.detach().cpu().numpy().ravel())
            VV.append(val)
            ENC.append(enc.detach().cpu().numpy().ravel())
            bundle, _, _, _, _ = pol.act(obs, tok, plan.to_vector(), env.get_action_mask,
                                         hidden=hidden, deterministic=False)
            obs, r, term, trunc, info = env.step(bundle)
            belief.update(obs, info.get("opp_played"))
            if term or trunc:
                obs, _ = env.reset(seed=7 + t)
                belief.reset(env.deck1); hidden = None
    H = np.asarray(H); VV = np.asarray(VV); ENC = np.asarray(ENC)
    h_std_per_dim = H.std(axis=0)
    e_std_per_dim = ENC.std(axis=0)
    print(f"\n=== 隐状态 h / 编码 enc / 价值输出（{len(VV)} 帧）===")
    print(f"  enc 跨帧 每维std: mean={e_std_per_dim.mean():.5f} "
          f"median={np.median(e_std_per_dim):.5f} max={e_std_per_dim.max():.5f}")
    print(f"  enc ||x|| : mean={np.linalg.norm(ENC,axis=1).mean():.3f} "
          f"min={np.linalg.norm(ENC,axis=1).min():.3f} max={np.linalg.norm(ENC,axis=1).max():.3f}")
    print(f"  h   跨帧 每维std: mean={h_std_per_dim.mean():.6f} "
          f"median={np.median(h_std_per_dim):.6f} max={h_std_per_dim.max():.6f}")
    print(f"  ||h||: mean={np.linalg.norm(H,axis=1).mean():.4f} "
          f"min={np.linalg.norm(H,axis=1).min():.4f} max={np.linalg.norm(H,axis=1).max():.4f}")
    _vlabel = ("value_head_mlp(value_enc)[E' independent]"
               if getattr(pol, "value_independent", False)
               else "value_head(enc)[bypass]" if getattr(pol, "value_bypass", False)
               else "value_head(h)")
    print(f"  {_vlabel}: mean={VV.mean():.4f} std={VV.std():.6f} "
          f"min={VV.min():.4f} max={VV.max():.4f}")
    hw = H @ vw
    print(f"  h·w（去 bias）: std={hw.std():.6f}  （value 中随状态变化的部分）")

    # —— GRU 门饱和诊断 ——
    gc = pol.gru_cell
    def _g(which, x):
        return getattr(gc, which)
    try:
        Wih = gc.weight_ih.detach(); Whh = gc.weight_hh.detach()
        bih = gc.bias_ih.detach(); bhh = gc.bias_hh.detach()
        x = torch.as_tensor(ENC[[0, len(ENC)//2, -1]], dtype=torch.float32, device=device)
        hprev = torch.as_tensor(H[[0, len(H)//2, -1]], dtype=torch.float32, device=device)
        gi = x @ Wih.t() + bih
        gh = hprev @ Whh.t() + bhh
        Hd = gc.hidden_size
        r = torch.sigmoid(gi[:, :Hd] + gh[:, :Hd])
        z = torch.sigmoid(gi[:, Hd:2*Hd] + gh[:, Hd:2*Hd])
        n = torch.tanh(gi[:, 2*Hd:] + r * gh[:, 2*Hd:])
        print(f"\n=== GRU 门（3 个抽样帧）===")
        print(f"  z(更新门) mean={z.mean().item():.4f}  （→1 则 h 冻结不更新）")
        print(f"  r(重置门) mean={r.mean().item():.4f}")
        print(f"  n(候选)   mean={n.mean().item():.4f} abs_mean={n.abs().mean().item():.4f}")
        print(f"  |z-1| mean={ (1-z).abs().mean().item():.6f}")
    except Exception as e:
        print(f"  GRU 门诊断失败: {e!r}")

    print(f"\n[结论] h 跨帧 std≈0 而 enc 跨帧 std 明显>0 → GRU 饱和/冻死；"
          f"若 enc 也≈0 → 编码器或输入恒定")


if __name__ == "__main__":
    main()
