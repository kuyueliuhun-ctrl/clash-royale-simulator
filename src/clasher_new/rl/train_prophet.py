"""先知规划器 Route A：特权观测 PPO（规划文档 5.3 路线 A）。

privileged obs = 标准可见观测 + 对手手牌/牌序/圣水/塔血 向量（仅训练期存在）。
输出仍是 (slot, y, x) 单卡动作（先知的粗糙建议），由 follower 蒸馏使用。

修复：
- P1-13：提供 :func:`prophet_policy_to_plan` 适配器，把 SB3 产物转成 PlanToken，
  可直接接入 train_follower 的 make_plan 蒸馏链路（不再是孤儿实验）；
- P2：priv 归一化（cycle/13、elixir/10、tower hp/5000），复用 observation 常量。
"""

import os
import sys
import argparse

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from rl.env_wrapper import RLEnv, legacy_action_to_bundle
from rl.observation import ENTITY_NAMES, GRID_H, GRID_W, GRID_C
from rl.plan_space import PlanToken

PRIV_DIM = 8 + 1 + 3  # opp cycle(8 ids) + opp elixir + opp towers(3)


def _build_priv(env, obs):
    """归一化特权向量（P2：塔血/圣水/牌序都压到 0..1）。"""
    hid = env.get_hidden_state()
    priv = np.concatenate([
        hid["opp_cycle"].astype(np.float32) / max(1, len(ENTITY_NAMES)),
        hid["opp_elixir"].astype(np.float32) / 10.0,
        hid["opp_towers"].astype(np.float32) / 5000.0,
    ]).astype(np.float32)
    return priv


class ProphetEnv(gym.Env):
    """给先知用的特权环境：观测 = 标准 obs + priv 向量，动作 = 单卡。"""

    def __init__(self, **env_kwargs):
        self.env = RLEnv(**env_kwargs)
        grid_shape = self.env.observation_space["grid"].shape
        self.observation_space = gym.spaces.Dict({
            "grid": self.env.observation_space["grid"],
            "hand": self.env.observation_space["hand"],
            "elixir": self.env.observation_space["elixir"],
            "next_card": self.env.observation_space["next_card"],
            "time": self.env.observation_space["time"],
            "priv": gym.spaces.Box(low=0.0, high=1.0, shape=(PRIV_DIM,), dtype=np.float32),
        })
        self.action_space = gym.spaces.MultiDiscrete([5, GRID_H, GRID_W])

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._augment(obs), info

    def _augment(self, obs):
        return {**obs, "priv": _build_priv(self.env, obs)}

    def step(self, action):
        bundle = legacy_action_to_bundle(action)
        obs, r, term, trunc, info = self.env.step(bundle)
        return self._augment(obs), r, term, trunc, info

    def __getattr__(self, name):
        return getattr(self.env, name)


class ProphetExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        self.entity_emb = nn.Embedding(len(ENTITY_NAMES), 8)
        in_ch = (GRID_C - 1) + 8 + 4
        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, GRID_H, GRID_W)
            cnn_out = self.cnn(dummy).shape[1]
        self.fc = nn.Linear(cnn_out + 5 * 8 + 3 + PRIV_DIM, features_dim)

    def forward(self, observation):
        grid = observation["grid"]
        hand = observation["hand"].long()
        elixir = observation["elixir"]
        priv = observation["priv"]
        extra = torch.cat([elixir, observation["next_card"].float(),
                           observation["time"].float()], dim=1)
        card_ids = grid[..., 0].long()
        card_vecs = self.entity_emb(card_ids)
        rest = grid[..., 1:]
        ct = rest[..., 2].long()  # 真实卡类型通道（P2 4.4）
        ct_oh = torch.nn.functional.one_hot(ct, num_classes=4).float()
        x = torch.cat([rest, card_vecs, ct_oh], dim=-1).permute(0, 3, 1, 2).float()
        grid_feat = self.cnn(x)
        hand_feat = self.entity_emb(hand).flatten(1)
        return torch.relu(self.fc(torch.cat([grid_feat, hand_feat, extra, priv], dim=1)))


def prophet_policy_to_plan(model, obs, env) -> PlanToken:
    """把 RL 先知（SB3 PPO）输出适配成 PlanToken，供 follower 蒸馏（P1-13）。

    - 由落点 (y) 推导 macro_intent / focus_region；
    - suggested_card 取动作槽位。
    """
    priv = _build_priv(env, obs)
    aug = {**obs, "priv": priv}
    action, _ = model.predict(aug, deterministic=True)
    slot, y, x = int(action[0]), int(action[1]), int(action[2])
    if slot == 0:
        return PlanToken.zeros()
    if y < 15:
        intent = "push_left" if x < 9 else "push_right"
        region = "enemy_left" if x < 9 else "enemy_right"
    else:
        intent = "defend_left" if x < 9 else "defend_right"
        region = "own_left" if x < 9 else "own_right"
    return PlanToken(
        macro_intent=intent,
        focus_region=region,
        suggested_card=slot,
        bundle_size_hint=1,
        combo_hint=0,
        risk_profile=0.5,
        value_estimate=0.0,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-timesteps", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", type=str, default="prophet_ppo")
    args = ap.parse_args()
    env = ProphetEnv(opponent=None, seed=args.seed)
    model = PPO("MultiInputPolicy", env,
                policy_kwargs={"features_extractor_class": ProphetExtractor},
                n_steps=2048, batch_size=256, learning_rate=1e-4, n_epochs=4,
                target_kl=0.03, seed=args.seed, verbose=1)
    model.learn(total_timesteps=args.total_timesteps)
    model.save(args.save)
    print(f"[save] {args.save}.zip")


if __name__ == "__main__":
    main()
