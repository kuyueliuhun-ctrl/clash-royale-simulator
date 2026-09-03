"""单卡 PPO baseline（规划文档 8.2）：SB3 PPO + ActionBundle 的 n<=1 兼容模式。

作为对照 baseline；正式版用 train_follower.py 的同刻多卡 bundle head。
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
from rl.action_bundle import ActionBundle


class SingleCardAdapter(gym.Env):
    """把 RLEnv 的 ActionBundle 动作转成旧式 (slot, y, x) 单卡动作，供 SB3 使用。"""

    def __init__(self, **env_kwargs):
        self.env = RLEnv(**env_kwargs)
        self.observation_space = self.env.observation_space
        self.action_space = gym.spaces.MultiDiscrete([5, 32, 18])

    def reset(self, *, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        bundle = legacy_action_to_bundle(action)
        return self.env.step(bundle)

    def __getattr__(self, name):
        return getattr(self.env, name)


class CRFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        grid_shape = observation_space["grid"].shape  # (32,18,15)
        self.entity_emb = nn.Embedding(13, 8)
        in_ch = (grid_shape[-1] - 1) + 8 + 4
        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, 32, 18)
            cnn_out = self.cnn(dummy).shape[1]
        self.fc = nn.Linear(cnn_out + 5 * 8 + 3, features_dim)

    def forward(self, observation):
        grid = observation["grid"]
        hand = observation["hand"].long()
        elixir = observation["elixir"]
        extra = torch.cat([elixir, observation["next_card"].float(),
                           observation["time"].float()], dim=1)

        card_ids = grid[..., 0].long()
        card_vecs = self.entity_emb(card_ids)
        rest = grid[..., 1:]
        card_type = rest[..., 2].long()  # 真实卡类型通道（P2 4.4）
        card_type_oh = torch.nn.functional.one_hot(card_type, num_classes=4).float()
        x = torch.cat([rest, card_vecs, card_type_oh], dim=-1).permute(0, 3, 1, 2).float()
        grid_feat = self.cnn(x)
        hand_feat = self.entity_emb(hand).flatten(1)
        return torch.relu(self.fc(torch.cat([grid_feat, hand_feat, extra], dim=1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-timesteps", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", type=str, default="baseline_ppo")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--n-envs", type=int, default=1)
    args = ap.parse_args()

    env = SingleCardAdapter(opponent=None, seed=args.seed)
    model = PPO(
        "MultiInputPolicy", env,
        policy_kwargs={"features_extractor_class": CRFeatureExtractor},
        n_steps=2048, batch_size=256, learning_rate=args.lr, n_epochs=4,
        target_kl=0.03, seed=args.seed, verbose=1,
    )
    model.learn(total_timesteps=args.total_timesteps)
    model.save(args.save)
    print(f"[save] {args.save}.zip")


if __name__ == "__main__":
    main()
