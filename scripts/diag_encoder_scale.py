"""encoder 输入尺度定位（2026-09-11 §4 诊断第三步）。

enc = relu(enc_fc(fused)) 的 ||enc||≈535（GRU 因此饱和）。
本脚本拆解 fused 各分量的尺度，找出"把 encoder 顶到 535 的元凶"。
"""

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src", "clasher_new")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

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
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    cfg = TrainConfig.resolve("economy")
    p = os.path.join(a.run_dir, "config.json")
    if os.path.exists(p):
        for k, v in json.load(open(p, encoding="utf-8")).items():
            if hasattr(cfg, k):
                try:
                    setattr(cfg, k, v)
                except Exception:
                    pass
    device = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"
    env = solo_env(cfg, 0)
    bdim = len(BeliefInference(opp_deck=env.deck1, n_particles=128, seed=0).encode(None, None))
    pol = load_checkpoint(a.ckpt, hidden_dim=cfg.hidden_dim, plan_dim=PLAN_DIM, belief_dim=bdim)
    pol.to_device(device)

    mirror_deck, _ = resolve_deck_set(getattr(cfg, "deck_set", None) or "default")
    bp = BeliefPlanner()
    # B'/E'：镜像对手必须用与 ckpt 相同的 value 架构（否则键集不匹配）
    opp = FollowerPolicy(hidden=cfg.hidden_dim, plan_dim=PLAN_DIM, belief_dim=bdim,
                         value_bypass=bool(getattr(pol, "value_bypass", False)),
                         value_independent=bool(getattr(pol, "value_independent", False)))
    opp.to_device(device); opp.load_state_dict(pol.state_dict())
    env.opponent = FollowerOpponent(opp, env, belief=BeliefInference(opp_deck=env.deck1,
                                   n_particles=128, seed=7), deterministic=True)
    belief = BeliefInference(opp_deck=env.deck1, n_particles=128, seed=7)
    obs, _ = env.reset(seed=7)
    belief.reset(env.deck1)
    hidden = None

    # 输入原始量级
    print("=== 原始 obs 量级 ===")
    print(f"  obs['time']     = {np.asarray(obs['time']).ravel()[:4]}")
    print(f"  obs['elixir']   = {np.asarray(obs['elixir']).ravel()[:4]}")
    print(f"  obs['grid']     min={obs['grid'].min():.3f} max={obs['grid'].max():.3f} "
          f"mean={obs['grid'].mean():.3f}")
    print(f"  obs['grid'] 各通道 max（前 8）={np.asarray(obs['grid']).reshape(-1, obs['grid'].shape[-1]).max(axis=0)[:8] if obs['grid'].ndim==3 else 'n/a'}")

    parts = {"grid_feat": [], "hand_feat": [], "scalar": [], "plan_f": [], "belief_f": [],
             "fused": [], "enc": [], "enc_pre": []}
    with torch.no_grad():
        for t in range(a.frames):
            plan = bp.plan(env.battle, belief.state(), obs)
            tok = belief.encode(obs, None)
            grid = torch.as_tensor(obs["grid"], dtype=torch.float32).unsqueeze(0).to(device)
            hand = torch.as_tensor(obs["hand"], dtype=torch.long).unsqueeze(0).to(device)
            elixir = torch.as_tensor(obs["elixir"], dtype=torch.float32).unsqueeze(0).to(device)
            time_ = torch.as_tensor(obs["time"], dtype=torch.float32).unsqueeze(0).to(device)
            nxt = torch.as_tensor(obs["next_card"], dtype=torch.float32).unsqueeze(0).to(device) / 12.0
            card_ids = grid[..., 0].long()
            card_vecs = pol.entity_emb(card_ids)
            rest = grid[..., 1:]
            ct = rest[..., 2].long()
            ct_oh = F.one_hot(ct, num_classes=4).float()
            x = torch.cat([rest, card_vecs, ct_oh], dim=-1).permute(0, 3, 1, 2)
            gf = pol.cnn(x)
            hf = pol.entity_emb(hand).reshape(1, -1)
            sc = torch.cat([elixir, time_, nxt], dim=1)
            pf = pol.plan_mlp(torch.as_tensor(plan.to_vector(), dtype=torch.float32).unsqueeze(0).to(device))
            bf = pol.belief_mlp(torch.as_tensor(tok, dtype=torch.float32).unsqueeze(0).to(device))
            fused = torch.cat([gf, hf, sc, pf, bf], dim=1)
            pre = pol.enc_fc(fused)
            enc = torch.relu(pre)
            for k, v in (("grid_feat", gf), ("hand_feat", hf), ("scalar", sc),
                         ("plan_f", pf), ("belief_f", bf), ("fused", fused),
                         ("enc_pre", pre), ("enc", enc)):
                parts[k].append(np.linalg.norm(v.detach().cpu().numpy().ravel()))
            hidden = pol.gru_cell(enc, (torch.zeros(1, cfg.hidden_dim, device=device)
                                        if hidden is None else hidden.detach()))
            bundle, _, _, _, _ = pol.act(obs, tok, plan.to_vector(), env.get_action_mask,
                                         hidden=hidden, deterministic=False)
            obs, r, term, trunc, info = env.step(bundle)
            belief.update(obs, info.get("opp_played"))
            if term or trunc:
                obs, _ = env.reset(seed=7 + t)
                belief.reset(env.deck1); hidden = None

    print(f"\n=== fused 各分量 L2 范数（{a.frames} 帧均值 ± 跨帧std）===")
    for k in ("grid_feat", "hand_feat", "scalar", "plan_f", "belief_f", "fused",
              "enc_pre", "enc"):
        arr = np.asarray(parts[k])
        print(f"  {k:10s} dim={int(pol.hidden_dim) if k=='enc' else '':>0} "
              f"||·|| mean={arr.mean():10.3f}  std={arr.std():8.3f}  "
              f"min={arr.min():10.3f} max={arr.max():10.3f}")


if __name__ == "__main__":
    main()
