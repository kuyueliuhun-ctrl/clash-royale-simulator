"""GRU 饱和归因实验（2026-09-11 §4 诊断第四步）。

已知：enc=relu(enc_fc(fused)) 的 ||enc||≈533（其中 grid_feat 常数分量 ~468），
GRUCell 的 tanh 候选饱和（abs_mean 0.994）→ h 冻成常数 → critic 恒为常数 → EV<0。

本实验做三组对照，证明"信息是被量级压死的"而非"输入里没有信息"：
  A. raw enc（现状）
  B. enc - running_mean（仅去常量偏置）
  C. enc 标准化（去均值除标准差）
对每组：① h 跨帧 std（衡量状态信息是否进入 h）；② 线性探针 H→R 的 EV
（h 里"线性可读出"的回报信息）；③ value_head(h) 的 EV。

若 B/C 下 h-std 与探针 EV 大幅回升 → 根因确认为 encoder 输出量级（GRU 饱和），
修复方向 = 在 enc 后加归一化（LayerNorm / 标准化），而非改奖励或换 critic。
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

from rl.config import TrainConfig, reward_to_env  # noqa: E402
from rl.belief import BeliefInference  # noqa: E402
from rl.belief_planner import BeliefPlanner  # noqa: E402
from rl.plan_space import PLAN_DIM  # noqa: E402
from rl.follower import FollowerPolicy, load_checkpoint  # noqa: E402
from rl.ppo import PPOTrainer  # noqa: E402
from rl.train_follower import FollowerOpponent  # noqa: E402
from rl.run_league import timeout_winner, overtime_open, _stall_probe, STALL_WINDOW  # noqa: E402
from rl.train_solo import solo_env, _draw_penalty, resolve_deck_set  # noqa: E402


def ev(v, r):
    v = np.asarray(v, np.float64); r = np.asarray(r, np.float64)
    var = float(r.var())
    if var <= 1e-12:
        return float("nan")
    return float(1.0 - ((v - r) ** 2).mean() / var)


def ridge_ev(H, R, EP, lam=1e-2):
    """线性探针（**留出法**：按局 70/30 切分，只在测试集算 EV）。

    避免"训练=测试"造成的过拟合假高分；A_raw 的 h 近乎常数时，
    若不过拟合，测试 EV 应≈0（甚至为负），这正是要对比的。
    """
    H = np.asarray(H, np.float64); R = np.asarray(R, np.float64); EP = np.asarray(EP)
    gs = np.unique(EP)
    rs = np.random.RandomState(0)
    rs.shuffle(gs)
    cut = max(1, int(len(gs) * 0.7))
    tr_g, te_g = set(gs[:cut].tolist()), set(gs[cut:].tolist())
    m_tr = np.isin(EP, list(tr_g))
    m_te = np.isin(EP, list(te_g))
    mu = H[m_tr].mean(0); sd = H[m_tr].std(0) + 1e-6
    Xtr = np.hstack([(H[m_tr] - mu) / sd, np.ones((m_tr.sum(), 1))])
    Xte = np.hstack([(H[m_te] - mu) / sd, np.ones((m_te.sum(), 1))])
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
    w = np.linalg.solve(A, Xtr.T @ R[m_tr])
    return ev(Xte @ w, R[m_te])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--run-dir", default="runs/prod_200k_valnorm_ev")
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--seed", type=int, default=999)
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
    env0 = solo_env(cfg, 0)
    bdim = len(BeliefInference(opp_deck=env0.deck1, n_particles=128, seed=0).encode(None, None))
    pol = load_checkpoint(a.ckpt, hidden_dim=cfg.hidden_dim, plan_dim=PLAN_DIM, belief_dim=bdim)
    pol.to_device(device)

    mirror_deck, _ = resolve_deck_set(getattr(cfg, "deck_set", None) or "default")
    env = solo_env(cfg, a.seed, deck0=mirror_deck, deck1=mirror_deck)
    bp = BeliefPlanner()
    # B'/E'：镜像对手必须用与 ckpt 相同的 value 架构（否则键集不匹配）
    opp = FollowerPolicy(hidden=cfg.hidden_dim, plan_dim=PLAN_DIM, belief_dim=bdim,
                         value_bypass=bool(getattr(pol, "value_bypass", False)),
                         value_independent=bool(getattr(pol, "value_independent", False)))
    opp.to_device(device); opp.load_state_dict(pol.state_dict())

    ENC, R_list, EP = [], [], []
    with torch.no_grad():
        for g in range(a.games):
            env.opponent = FollowerOpponent(opp, env, belief=BeliefInference(
                opp_deck=env.deck1, n_particles=128, seed=a.seed + g), deterministic=True)
            belief = BeliefInference(opp_deck=env.deck1, n_particles=128, seed=a.seed + 1000 + g)
            obs, _ = env.reset(seed=a.seed + 2000 + g)
            belief.reset(env.deck1)
            hidden = None
            last_hp, stall = None, 0
            ep_enc, ep_val, ep_rew, ep_term, ep_trunc = [], [], [], [], []
            steps, done = 0, False
            while not done and (steps < cfg.max_ep_steps or overtime_open(env.battle)):
                if cfg.train_stall_stop and steps and steps % STALL_WINDOW == 0:
                    early, last_hp, stall = _stall_probe(env, last_hp, stall)
                    if early:
                        break
                plan = bp.plan(env.battle, belief.state(), obs)
                tok = belief.encode(obs, None)
                fused, enc = pol._encode_parts(obs, tok, plan.to_vector())
                h = pol.gru_cell(enc, (torch.zeros(1, cfg.hidden_dim, device=device)
                                       if hidden is None else hidden.detach()))
                hidden = h
                ep_enc.append(enc.detach().cpu().numpy().ravel())
                # B'/E'：走策略真实 value 通路（勿硬编码 value_head(h)）
                ep_val.append(float(pol._value_from(enc, h, fused).item()))
                bundle, _, _, _, _ = pol.act(obs, tok, plan.to_vector(),
                                             env.get_action_mask, hidden=hidden,
                                             deterministic=False)
                obs, r, term, trunc, info = env.step(bundle)
                ep_rew.append(float(r)); ep_term.append(bool(term)); ep_trunc.append(bool(trunc))
                belief.update(obs, info.get("opp_played"))
                done = term or trunc
                steps += 1
            if env.battle.winner is None and not env.battle.game_over and ep_rew:
                virt = timeout_winner(env.battle)
                rw = reward_to_env(cfg)
                if virt == 0:
                    ep_rew[-1] += float(rw["win_bonus"])
                elif virt == 1:
                    ep_rew[-1] -= float(rw["lose_penalty"])
                else:
                    ep_rew[-1] -= _draw_penalty(cfg)
            adv, ret = PPOTrainer.compute_gae(ep_rew, ep_val, ep_term, cfg.gamma,
                                              cfg.gae_lambda, truncated=ep_trunc,
                                              last_value=0.0)
            for i in range(len(ep_enc)):
                ENC.append(ep_enc[i]); R_list.append(float(ret[i])); EP.append(g)

    ENC = np.asarray(ENC); R = np.asarray(R_list); EP = np.asarray(EP)
    n = len(R)
    print(f"[ablation] 帧数={n} 局数={a.games}  Var(R)={R.var():.3f} E[R]={R.mean():.3f}")

    mu = ENC.mean(0)
    sd = ENC.std(0) + 1e-6
    variants = {
        "A_raw": ENC,
        "B_center": ENC - mu,
        "C_standardized": (ENC - mu) / sd,
    }
    W = pol.gru_cell.weight_hh.detach().cpu().numpy()
    print(f"\n{'variant':16s} {'||enc||':>9s} {'h跨帧std':>10s} {'h_norm':>8s} "
          f"{'probe_EV(h→R)':>15s} {'vhead_EV':>9s}")
    results = {}
    for name, Xi in variants.items():
        Xt = torch.as_tensor(Xi, dtype=torch.float32, device=device)
        H = []
        with torch.no_grad():
            for i in range(n):
                h = pol.gru_cell(Xt[i:i + 1], torch.zeros(1, cfg.hidden_dim, device=device))
                H.append(h.cpu().numpy().ravel())
        H = np.asarray(H)
        vh = (H @ pol.value_head.weight.detach().cpu().numpy().ravel()
              + float(pol.value_head.bias.detach().cpu().numpy().ravel()[0]))
        results[name] = (H, vh)
        print(f"{name:16s} {np.linalg.norm(Xi[0]):9.2f} {H.std(0).mean():10.6f} "
              f"{np.linalg.norm(H, axis=1).std():8.5f} "
              f"{ridge_ev(H, R, EP):15.4f} {ev(vh, R):9.4f}")

    # 逐局看 h 是否随局变化（B/C 下应显著）
    print("\n=== 各 variant 的 h 跨局标准差（每局均值再跨局求 std）===")
    for name, (H, _) in results.items():
        gmean = np.asarray([H[EP == g].mean(0) for g in np.unique(EP)])
        print(f"  {name:16s} 跨局 per-dim std mean={gmean.std(0).mean():.6f} "
              f"||局均差|| std={np.linalg.norm(gmean - gmean.mean(0), axis=1).std():.5f}")


if __name__ == "__main__":
    main()
