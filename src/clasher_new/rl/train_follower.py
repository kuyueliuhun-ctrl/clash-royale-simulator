"""跟随者训练：PPO + 同刻多卡 ActionBundle + 信念/先知 plan（规划文档 8.5）。

- 信念：BeliefInference（规则+统计+可选神经）；
- plan：BeliefPlanner 为主，按概率注入 ProphetPlanner（蒸馏），支持 plan/belief dropout；
- 对手：随机 / 脚本启发式 / 固定 FollowerPolicy（exploiter 场景）。
- PPO 重放记录 init_hidden（P0-1）；截断 episode 用 last_value bootstrap（P1-7）。
"""

import os
import sys
import random
import argparse
import time

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import numpy as np
import torch

from rl.env_wrapper import RLEnv
from rl.action_bundle import ActionBundle, K_MAX
from rl.belief import BeliefInference
from rl.belief_planner import BeliefPlanner
from rl.prophet import ProphetPlanner
from rl.plan_space import PlanToken, PLAN_DIM
from rl.follower import FollowerPolicy, save_checkpoint, load_checkpoint
from rl.ppo import PPOTrainer
from rl.observation import ENTITY_NAMES


def heuristic_opponent(env, rng=None):
    """脚本对手：从修复后的 player-1 掩码采样一张可出的牌放到合法格子。"""
    rng = rng or random

    def _opp(obs):
        mask = env.get_action_mask_for(1)
        slots = np.flatnonzero(mask["slots"])
        if slots.size == 0:
            return ActionBundle.noop()
        slot = int(rng.choice(slots))
        cells = np.flatnonzero(mask["cells"][slot])
        if cells.size == 0:
            return ActionBundle.noop()
        cell = int(rng.choice(cells))
        x, y = cell % 18, cell // 18
        return ActionBundle.from_single(slot + 1, x, y)
    return _opp


class FollowerOpponent:
    """把 FollowerPolicy 包装成 env 对手（player-1 视角，完整信念/plan 链路，P1-8）。

    - 独立 BeliefInference（追踪 player-0 的出牌）+ BeliefPlanner；
    - mask 用 env.get_action_mask_for(1)（依赖 P0-3/P0-4 修复）；
    - hidden 在 reset()/episode 结束时清空。
    """

    def __init__(self, policy, env, belief=None, planner=None, deterministic=True):
        self.policy = policy
        self.env = env
        self.belief = belief or BeliefInference(opp_deck=env.deck0, n_particles=128, seed=0)
        self.planner = planner or BeliefPlanner()
        self.hidden = None
        self.deterministic = deterministic
        #: 最近一步的完整轨迹（flow 联赛双侧收集用）；每步 __call__ 刷新。
        self._last_step = None

    def observe_opponent_played(self, played_cards):
        """用 player-0（agent）本 tick 打出的卡更新对手信念。"""
        for c in played_cards:
            self.belief.update(None, c)

    def reset(self):
        self.hidden = None
        self.belief.reset()
        self._last_step = None

    def __call__(self, obs):
        belief_tok = self.belief.encode(obs, None)
        plan = self.planner.plan(self.env.battle, self.belief.state(), obs)
        plan_tok = plan.to_vector()
        if plan_tok.shape[0] > self.policy.plan_dim:
            plan_tok = plan_tok[:self.policy.plan_dim]
        elif plan_tok.shape[0] < self.policy.plan_dim:
            plan_tok = np.pad(plan_tok, (0, self.policy.plan_dim - plan_tok.shape[0]))

        def mask_fn(partial=None):
            return self.env.get_action_mask_for(1, partial)

        bundle, lp, val, self.hidden, masks = self.policy.act(
            obs, belief_tok, plan_tok.astype(np.float32), mask_fn,
            hidden=self.hidden, deterministic=self.deterministic)
        # 记录本步完整轨迹，供 flow 联赛收集 player-1 侧 transition（P-flow）。
        self._last_step = {
            "obs": obs,
            "belief": belief_tok,
            "plan": plan_tok.astype(np.float32),
            "bundle": bundle,
            "lp": float(lp),
            "val": float(val),
            "hidden": self.hidden,
            "masks": masks,
        }
        return bundle

    def take_last_step(self):
        """返回最近一步 player-1 侧轨迹 dict（flow 双侧收集用）；无记录返回 None。"""
        return self._last_step


def run_training(total_steps, n_envs=1, batch_size=128, update_interval=128,
                 lr=3e-4, gamma=0.99, gae_lambda=0.95, clip=0.2,
                 plan_prophet_prob=0.3, plan_dropout=0.1, belief_dropout=0.1,
                 seed=0, opponent="random", main_policy_path=None,
                 hidden_dim=128, save="follower.pt", eval_every=2000, max_ep_steps=600,
                 prophet_model=None, init_from=None):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    env = RLEnv(opponent=None, seed=seed)
    rng = random.Random(seed)
    opp_agent = None
    if opponent == "heuristic":
        env.opponent = heuristic_opponent(env, rng)
    elif opponent == "main_policy":
        mp = load_checkpoint(main_policy_path, hidden_dim=hidden_dim)
        mp.eval()
        opp_agent = FollowerOpponent(mp, env)
        env.opponent = opp_agent

    belief = BeliefInference(opp_deck=env.deck1, n_particles=128, seed=seed)
    belief_planner = BeliefPlanner()
    prophet = ProphetPlanner()
    policy = FollowerPolicy(hidden=hidden_dim, plan_dim=PLAN_DIM,
                            belief_dim=len(belief.encode(None, None)))
    if init_from:
        bc = load_checkpoint(init_from, hidden_dim=hidden_dim)
        assert bc.plan_dim == policy.plan_dim and bc.belief_dim == policy.belief_dim, \
            f"BC checkpoint 维度不符: {bc.plan_dim}/{bc.belief_dim} vs {policy.plan_dim}/{policy.belief_dim}"
        policy.load_state_dict(bc.state_dict())
        print(f"[init] 从 BC checkpoint 初始化: {init_from}")
    ppo = PPOTrainer(policy, lr=lr, gamma=gamma, gae_lambda=gae_lambda, clip=clip)

    def make_plan(obs, use_prophet):
        if use_prophet and prophet_model is not None:
            from rl.train_prophet import prophet_policy_to_plan
            return prophet_policy_to_plan(prophet_model, obs, env)
        if use_prophet:
            return prophet.plan(env.get_prophet_state())
        return belief_planner.plan(env.battle, belief.state(), obs)

    obs, _ = env.reset()
    belief.reset(env.deck1)
    hidden = None
    ep_obs, ep_belief, ep_plan, ep_bundle, ep_lp, ep_val, ep_rew = [], [], [], [], [], [], []
    ep_term, ep_trunc, ep_masks, ep_init = [], [], [], []
    transitions = []

    t0 = time.time()
    for step in range(1, total_steps + 1):
        use_prophet = rng.random() < plan_prophet_prob
        plan = make_plan(obs, use_prophet)
        plan_vec = plan.to_vector()
        if rng.random() < plan_dropout:
            plan_vec = np.zeros_like(plan_vec)
        belief_tok = belief.encode(obs, None)
        if rng.random() < belief_dropout:
            belief_tok = np.zeros_like(belief_tok)

        # 记录进入该步的隐状态（P0-1）
        init_hidden = hidden
        bundle, lp, val, hidden, masks = policy.act(
            obs, belief_tok, plan_vec, env.get_action_mask, hidden=hidden, deterministic=False)

        # 解析 agent 本 tick 实际打的卡（用于 FollowerOpponent 信念更新）
        agent_played = []
        for sa in bundle.sub_actions:
            if sa.kind == "deploy" and 1 <= sa.slot <= K_MAX:
                cid = int(obs["hand"][sa.slot - 1])
                if 0 <= cid < len(ENTITY_NAMES):
                    agent_played.append(ENTITY_NAMES[cid])

        obs2, reward, term, trunc, info = env.step(bundle)
        done = term or trunc

        if opp_agent is not None:
            opp_agent.observe_opponent_played(agent_played)

        ep_obs.append(obs); ep_belief.append(belief_tok); ep_plan.append(plan_vec)
        ep_bundle.append(bundle); ep_lp.append(lp); ep_val.append(val)
        ep_rew.append(reward); ep_term.append(term); ep_trunc.append(trunc)
        ep_masks.append(masks); ep_init.append(init_hidden)

        belief.update(obs2, info.get("opp_played"))
        obs = obs2

        if done or len(ep_rew) >= max_ep_steps:
            truncated = (not term) and (len(ep_rew) >= max_ep_steps)
            last_val = 0.0
            if truncated:
                # 截断时用截断后状态的 value bootstrap（P1-7）
                last_val = policy.value(obs, belief_tok, plan_vec, hidden)
            adv, ret = PPOTrainer.compute_gae(ep_rew, ep_val, ep_term, gamma, gae_lambda,
                                              truncated=ep_trunc, last_value=last_val)
            for i in range(len(ep_rew)):
                transitions.append({
                    "obs": ep_obs[i], "belief": ep_belief[i], "plan": ep_plan[i],
                    "bundle": ep_bundle[i], "old_logprob": ep_lp[i],
                    "adv": float(adv[i]), "returns": float(ret[i]),
                    "masks": ep_masks[i], "init_hidden": ep_init[i],
                })
            obs, _ = env.reset()
            belief.reset(env.deck1)
            hidden = None
            if opp_agent is not None:
                opp_agent.reset()
            ep_obs, ep_belief, ep_plan, ep_bundle, ep_lp, ep_val, ep_rew = [], [], [], [], [], []
            ep_term, ep_trunc, ep_masks, ep_init = [], [], [], []

        if len(transitions) >= update_interval:
            stats = ppo.update(transitions[:batch_size] if len(transitions) > batch_size else transitions)
            transitions = transitions[batch_size:] if len(transitions) > batch_size else []
            print(f"[step {step}] policy={stats['policy_loss']:.4f} value={stats['value_loss']:.4f} "
                  f"entropy={stats['entropy']:.4f} elapsed={time.time()-t0:.1f}s", flush=True)

        if eval_every and step % eval_every == 0:
            wr, mrew = evaluate(policy, belief, belief_planner, n_games=5, seed=step)
            print(f"[eval@{step}] vs random winrate={wr:.2f} mean_reward={mrew:.2f}", flush=True)

    save_checkpoint(policy, save)
    print(f"[save] {save}")


def evaluate(policy, belief, belief_planner, n_games=5, seed=0):
    env = RLEnv(opponent=None, seed=seed)
    policy.eval()
    wins, rew_sum = 0, 0.0
    for _ in range(n_games):
        obs, _ = env.reset()
        belief.reset(env.deck1)
        hidden = None
        done = False
        while not done:
            plan = belief_planner.plan(env.battle, belief.state(), obs)
            tok = belief.encode(obs, None)
            bundle, _, _, hidden, _ = policy.act(obs, tok, plan.to_vector(),
                                              env.get_action_mask, hidden=hidden, deterministic=True)
            obs, r, term, trunc, info = env.step(bundle)
            belief.update(obs, info.get("opp_played"))
            done = term or trunc
            rew_sum += r
        if env.battle.winner == 0:
            wins += 1
    return wins / n_games, rew_sum / n_games


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-steps", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--update-interval", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--plan-prophet-prob", type=float, default=0.3)
    ap.add_argument("--plan-dropout", type=float, default=0.1)
    ap.add_argument("--belief-dropout", type=float, default=0.1)
    ap.add_argument("--opponent", type=str, default="random", choices=["random", "heuristic", "main_policy"])
    ap.add_argument("--main-policy-path", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", type=str, default="follower.pt")
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--max-ep-steps", type=int, default=600)
    ap.add_argument("--init-from", type=str, default=None, help="BC 预训练 checkpoint 初始化")
    args = ap.parse_args()
    run_training(**vars(args))
