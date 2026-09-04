"""轻量 PPO 更新器（规划文档 8.5 跟随者 RL 微调）。

针对 follower 的 autoregressive bundle head 设计：逐条 transition 重放计算
当前策略 logprob，支持 GAE 返回值与 clip 目标。

修复要点：
- P0-1：重放时使用 rollout 记录的隐状态（init_hidden），保证 ratio 是有效 IS 比；
- P0-2：熵正则使用 evaluate 返回的真实分布熵（非 -logprob）；
- P1-7：截断（truncated）episode 用 last_value bootstrap，不再把截断当终止。
"""

import os
import sys

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import numpy as np
import torch
import torch.nn.functional as F


class PPOTrainer:
    def __init__(self, policy, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip=0.2,
                 vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, adv_norm="batch"):
        self.policy = policy
        self.opt = torch.optim.Adam(policy.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip = clip
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.adv_norm = adv_norm    # batch=整批中心化(旧) / scale=只除std / none=原始
        self.updates = 0            # 累计 update 次数（flow 联赛 / 测试校验用）

    @staticmethod
    def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95,
                    truncated=None, last_value=0.0):
        """values: 每步 value。返回 advantages 与 returns。

        - dones[t]=True（真实终止）→ next_val=0；
        - truncated[t]=True 且非终止（max_ep_steps 截断）→ next_val=last_value（bootstrap）。
        """
        T = len(rewards)
        if truncated is None:
            truncated = [False] * T
        adv = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                if truncated[t] and not dones[t]:
                    next_val = last_value
                else:
                    next_val = 0.0
            else:
                next_val = 0.0 if dones[t] else values[t + 1]
            delta = rewards[t] + gamma * next_val - values[t]
            gae = delta + gamma * lam * (0.0 if (t == T - 1 or dones[t]) else gae)
            adv[t] = gae
        returns = adv + np.asarray(values, dtype=np.float32)
        return adv, returns

    def update(self, transitions, ent_coef=None, adv_norm=None):
        """transitions: list of dicts {obs, belief, plan, bundle, old_logprob,
        adv, returns, masks, init_hidden(可选)}。

        并行版：用 evaluate_batch 一次前向+反向算完所有 transition（CNN 编码只做一次，
        每个 decoder 步按存活子集批量），梯度与逐条累加完全等价（loss 用 sum 保留量纲）。

        ent_coef / adv_norm：单次更新覆盖（None = 用构造参数）。
        adv_norm: batch=整批中心化（旧默认）/ scale=只除以批 std（不中心化，
        避免“躺平零优势帧”被批均值抬成伪正优势）/ none=不归一化。
        返回 stats 附带 ratio/clip/raw-adv/梯度范数诊断。
        """
        self.policy.train()
        if not transitions:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                    "ratio_mean": 1.0, "clip_frac": 0.0,
                    "adv_mean": 0.0, "adv_std": 0.0, "grad_norm": 0.0, "n": 0}
        self.updates += 1
        adv_raw = np.array([t["adv"] for t in transitions], dtype=np.float32)
        advs = adv_raw.copy()
        mode = self.adv_norm if adv_norm is None else adv_norm
        std = float(advs.std())
        if mode == "scale":
            if std > 1e-6:
                advs = advs / (std + 1e-8)
        elif mode == "none":
            pass
        elif mode == "batch":
            if std > 1e-6:
                advs = (advs - advs.mean()) / (std + 1e-8)
        else:
            raise ValueError(f"未知 adv_norm='{mode}'（batch/scale/none）")
        coef = self.ent_coef if ent_coef is None else ent_coef

        dev = self.policy.device
        lp_new, value, ent = self.policy.evaluate_batch(
            [t["obs"] for t in transitions],
            [t["belief"] for t in transitions],
            [t["plan"] for t in transitions],
            [t["bundle"] for t in transitions],
            [t["masks"] for t in transitions],
            [t.get("init_hidden") for t in transitions])
        old = torch.tensor([t["old_logprob"] for t in transitions],
                           dtype=torch.float32, device=dev)
        ratio = torch.exp(lp_new - old)
        adv_t = torch.tensor(advs, dtype=torch.float32, device=dev)
        surr1 = ratio * adv_t
        surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv_t
        p_loss = -torch.min(surr1, surr2).sum()
        rets = torch.tensor([t["returns"] for t in transitions],
                            dtype=torch.float32, device=dev)
        v_loss = F.mse_loss(value.squeeze(-1), rets, reduction="sum")
        loss = p_loss + self.vf_coef * v_loss - coef * ent.sum()

        self.opt.zero_grad()
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.max_grad_norm))
        self.opt.step()
        n = len(transitions)
        with torch.no_grad():
            clip_frac = float((((ratio < 1 - self.clip) | (ratio > 1 + self.clip))
                               .float().mean().item()))
            ratio_mean = float(ratio.mean().item())
        return {"policy_loss": float(p_loss.item()) / n,
                "value_loss": float(v_loss.item()) / n,
                "entropy": float(ent.mean().item()),
                "ratio_mean": ratio_mean,
                "clip_frac": clip_frac,
                "adv_mean": float(adv_raw.mean()),
                "adv_std": float(adv_raw.std()),
                "grad_norm": grad_norm,
                "n": n}
