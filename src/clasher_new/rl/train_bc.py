"""跟随者行为克隆（BC）预训练（规划文档 8.5 / 评审 P1-23）。

在 PPO 之前先用规则专家（BeliefPlanner 建议卡 + 就近合法格）的行为克隆初始化
follower，给 RL 一个非零起点。产物可直接喂给 train_follower --init-from。

流程：
1. 专家策略：belief plan → suggested_card → 在 focus_region 中心附近的合法格部署；
2. 收集 (obs, belief_tok, plan_vec, bundle, masks)；
3. 最大化 follower 对专家 bundle 的 logprob（行为克隆损失）；
4. save_checkpoint 保存（带元数据）。
"""

import os
import sys
import random
import argparse

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import numpy as np
import torch

from rl.env_wrapper import RLEnv
from rl.action_bundle import ActionBundle, K_MAX
from rl.belief import BeliefInference
from rl.belief_planner import BeliefPlanner
from rl.follower import FollowerPolicy, save_checkpoint
from rl.plan_space import PLAN_DIM

#: focus_region → 目标网格（本地坐标）
REGION_CENTERS = {
    "own_left": (4, 20), "own_center": (9, 20), "own_right": (14, 20),
    "bridge_left": (4, 16), "bridge_right": (14, 16),
    "enemy_left": (4, 12), "enemy_center": (9, 12), "enemy_right": (14, 12),
}


def expert_bundle(env, belief, bp, obs, rng):
    """规则专家：plan → suggested_card → focus_region 中心最近合法格。"""
    plan = bp.plan(env.battle, belief.state(), obs)
    plan_vec = plan.to_vector()
    slot = plan.suggested_card
    if slot is None or slot < 1 or slot > K_MAX:
        return ActionBundle.noop(), plan_vec
    mask = env.get_action_mask_for(0)
    slot0 = slot - 1
    if not mask["slots"][slot0]:
        return ActionBundle.noop(), plan_vec
    cells = np.flatnonzero(mask["cells"][slot0])
    if cells.size == 0:
        return ActionBundle.noop(), plan_vec
    cx, cy = REGION_CENTERS.get(plan.focus_region, (9, 16))
    best = min(cells, key=lambda c: (abs(c % 18 - cx) + abs(c // 18 - cy)))
    x, y = int(best % 18), int(best // 18)
    return ActionBundle.from_single(slot, x, y), plan_vec


def collect(n_games, seed, policy, max_steps=600):
    env = RLEnv(opponent=None, seed=seed)
    rng = random.Random(seed)
    belief = BeliefInference(opp_deck=env.deck1, n_particles=128, seed=seed)
    bp = BeliefPlanner()
    samples = []
    for g in range(n_games):
        obs, _ = env.reset()
        belief.reset(env.deck1)
        done = False
        steps = 0
        while not done and steps < max_steps:
            bundle, plan_vec = expert_bundle(env, belief, bp, obs, rng)
            belief_tok = belief.encode(obs, None)
            # 掩码必须在采集时刻生成（battle 状态随后会变）
            masks = policy.masks_for(obs, belief_tok, plan_vec, bundle, env.get_action_mask)
            samples.append((obs, belief_tok, plan_vec, bundle, masks))
            obs, _, term, trunc, info = env.step(bundle)
            belief.update(obs, info.get("opp_played"))
            done = term or trunc
            steps += 1
        if (g + 1) % 10 == 0:
            print(f"[collect] game {g+1}/{n_games} samples={len(samples)}", flush=True)
    return samples


def train_bc(n_games=50, epochs=3, lr=1e-3, hidden_dim=128, seed=0,
             out="follower_bc.pt", max_steps=600):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = RLEnv(opponent=None, seed=seed)
    belief = BeliefInference(opp_deck=env.deck1, n_particles=128, seed=seed)
    belief_dim = len(belief.encode(None, None))
    policy = FollowerPolicy(hidden=hidden_dim, plan_dim=PLAN_DIM, belief_dim=belief_dim)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)

    samples = collect(n_games, seed, policy, max_steps=max_steps)
    if not samples:
        raise RuntimeError("BC 未采到任何样本")
    print(f"[bc] {len(samples)} samples, dims plan={PLAN_DIM} belief={belief_dim}")

    for ep in range(epochs):
        perm = np.random.permutation(len(samples))
        tot = 0.0
        for i in perm:
            obs, tok, plan, bundle, masks = samples[i]
            lp, _, _, _ = policy.evaluate(obs, tok, plan, bundle, masks, hidden=None)
            loss = -lp
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(lp.item())
        print(f"[bc] epoch {ep+1}/{epochs} mean_logprob={tot/len(samples):.3f}", flush=True)

    save_checkpoint(policy, out)
    print(f"[save] {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-games", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="follower_bc.pt")
    ap.add_argument("--max-steps", type=int, default=600)
    args = ap.parse_args()
    train_bc(n_games=args.n_games, epochs=args.epochs, lr=args.lr,
             hidden_dim=args.hidden_dim, seed=args.seed, out=args.out,
             max_steps=args.max_steps)
