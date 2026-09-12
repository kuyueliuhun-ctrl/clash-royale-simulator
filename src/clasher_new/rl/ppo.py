"""轻量 PPO 更新器（规划文档 8.5 跟随者 RL 微调）。

针对 follower 的 autoregressive bundle head 设计：逐条 transition 重放计算
当前策略 logprob，支持 GAE 返回值与 clip 目标。

修复要点：
- P0-1：重放时使用 rollout 记录的隐状态（init_hidden），保证 ratio 是有效 IS 比；
- P0-2：熵正则使用 evaluate 返回的真实分布熵（非 -logprob）；
- P1-7：截断（truncated）episode 用 last_value bootstrap，不再把截断当终止。
- P0-1b（2026-09-11 审计整改）：价值通道量纲修复——`value_norm` + 梯度成分诊断。
- P0-1c（2026-09-11 判读整改）：`explained_variance` 诊断——`value_loss` 被 `s²`
  除过、跨版本不可比，改用无量纲的 EV 判读"critic 到底有没有在学"（实测 EV≈0）。

关于 `value_norm`（重要，勿凭直觉改回"归一化 critic 输出"的写法）：
本实现的 rollout 与更新共用同一份权重，单次 update 内 `evaluate_batch` 重算的
logprob 与 rollout 记录的 `old_logprob` **恒等** → `ratio≡1.000`、`clip_frac≡0`
是**结构性的**（单轮 on-policy、n_epochs=1），不是"策略没在动"的证据。
真正的病是 **`value_loss` 量纲压倒策略项**（实测 gnorm 中位 6.8e3 vs 阈值 0.5，
value_loss 中位 ~48）。因此这里采取**保留 critic 原始输出域、只缩放价值损失**的做法：
    v_loss = MSE(v, R) / s²，  s = 回报的运行标准差（Welford）
这与"把 critic 输出重参数化到归一化域再训练"在梯度上只差一个常数因子 σ，
而 Adam 对逐参数梯度尺度近似不变 → 两种写法实际等价；但本写法**不需要**在
rollout 的每个 `value()` 消费点反归一化，因而不触碰 act/value 的语义，
`value_norm="none"` 时逐位等于旧实现。
"""

import math
import os
import sys

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import numpy as np
import torch
import torch.nn.functional as F


class ReturnScaler:
    """回报尺度的运行统计（Welford 在线均值/方差），用于价值损失的量纲对齐。

    只做标量统计，不改变任何返回值域；`scale()` 在样本不足或方差退化时返回 0，
    调用方据此回退"不缩放"（= 旧行为）。
    """

    def __init__(self, eps=1e-6):
        self.eps = float(eps)
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, values):
        """批量喂入一批回报（list/np.ndarray）。"""
        for v in np.asarray(values, dtype=np.float64).ravel():
            self.count += 1
            d = float(v) - self.mean
            self.mean += d / self.count
            self.m2 += d * (float(v) - self.mean)

    def var(self):
        return self.m2 / (self.count - 1) if self.count > 1 else 0.0

    def std(self):
        return float(math.sqrt(max(0.0, self.var())))

    def scale(self):
        """价值损失的除数 s（v_loss /= s²）。统计未建立/方差过小 → 0.0 表示不缩放。"""
        s = self.std()
        return s if s > self.eps else 0.0

    def to_dict(self):
        return {"count": self.count, "mean": self.mean, "m2": self.m2}

    @classmethod
    def from_dict(cls, d):
        s = cls()
        d = d or {}
        s.count = int(d.get("count", 0))
        s.mean = float(d.get("mean", 0.0))
        s.m2 = float(d.get("m2", 0.0))
        return s


class PPOTrainer:
    def __init__(self, policy, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip=0.2,
                 vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, adv_norm="batch",
                 value_norm="none", diagnose_every=0, ret_scaler=None):
        """value_norm: none=旧行为（价值损失不缩放）/ running=按回报运行 std 缩放。
        diagnose_every: >0 时每 N 次 update 额外算一次 p_gnorm/v_gnorm 梯度分解
        （多两次 backward，仅诊断用；0 = 关闭，旧行为）。
        ret_scaler: 断点续训时传入已恢复的 ReturnScaler（None = 新建）。
        """
        self.policy = policy
        self.opt = torch.optim.Adam(policy.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip = clip
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.adv_norm = adv_norm    # batch=整批中心化(旧) / scale=只除std / none=原始
        self.value_norm = value_norm
        self.diagnose_every = int(diagnose_every or 0)
        self.ret_scaler = ret_scaler if ret_scaler is not None else ReturnScaler()
        self.updates = 0            # 累计 update 次数（flow 联赛 / 测试校验用）
        #: v3 P0-B：最近一次 update 的逐帧 `(value_pred, return)`。
        #: 训练环把它累积到评估窗口，评估时**对全体帧算一次 EV**（池化口径）。
        #: 旧口径"各更新批 EV 取均值"因批=128 连续帧、相邻帧 corr(R)≈0.99，
        #: 批内 Var(R) 仅为全局 0.32 倍 → 把 EV 放大约 3 倍（−0.58 vs −1.74）。
        self.last_ev_pairs = None

    @staticmethod
    def explained_variance(values, returns):
        """`EV = 1 − MSE(v, R) / Var(R)`：critic 对回报的解释力。

        - `EV = 1`：完美预测；`EV = 0`：等价于"恒定预测批均值"（= 没学到东西）；
        - `EV < 0`：比常数预测还差。

        **为什么不用 `value_loss` 判读 critic**：`value_loss` 会被 `value_norm`
        除以 `s²`，量纲随回报尺度漂移，跨版本不可比；EV 是**无量纲**的，是
        "critic 到底有没有在学"的唯一干净答案（2026-09-11 20k 验证跑发现
        `value_loss=1.59` 看似达标，但 EV≈0，即价值网络对回报的解释力等于常数）。
        `Var(R)≈0`（单批回报退化）→ 返回 0.0（不可判读，不抛错）。
        """
        v = np.asarray(values, dtype=np.float64).ravel()
        r = np.asarray(returns, dtype=np.float64).ravel()
        if r.size < 2 or v.size != r.size:
            return 0.0
        var = float(r.var())
        if var <= 1e-12:
            return 0.0
        mse = float(((v - r) ** 2).mean())
        return float(1.0 - mse / var)

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
            return {"policy_loss": 0.0, "value_loss": 0.0, "value_loss_raw": 0.0,
                    "entropy": 0.0, "ratio_mean": 1.0, "clip_frac": 0.0,
                    "adv_mean": 0.0, "adv_std": 0.0, "grad_norm": 0.0,
                    "value_scale": 1.0, "explained_variance": 0.0, "n": 0}
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
        rets_np = np.array([t["returns"] for t in transitions], dtype=np.float32)
        rets = torch.tensor(rets_np, dtype=torch.float32, device=dev)
        v_mse_sum = F.mse_loss(value.squeeze(-1), rets, reduction="sum")
        # —— 价值通道量纲对齐（P0-1b）——
        # 先更新运行统计再取 scale：本批即参与统计（均值/方差对单批不敏感）。
        if self.value_norm == "running":
            self.ret_scaler.update(rets_np)
            s = self.ret_scaler.scale()
            v_scale = s if s > 0 else 1.0
        elif self.value_norm in ("none", "", None):
            v_scale = 1.0
        else:
            raise ValueError(f"未知 value_norm='{self.value_norm}'（none/running）")
        v_loss = v_mse_sum / (v_scale * v_scale)
        loss = p_loss + self.vf_coef * v_loss - coef * ent.sum()

        # —— 梯度成分诊断（P0-1a，可证伪"评论家主导"）——
        # p_loss / vf_coef*v_loss 分别求梯度范数后再走原有合并 backward（多两次
        # 反传，仅诊断时启用）。两者共享参数，范数之比即"谁在推动更新"。
        p_gnorm = v_gnorm = None
        if self.diagnose_every and (self.updates % self.diagnose_every == 0):
            params = [p for p in self.policy.parameters() if p.requires_grad]
            gv_loss = self.vf_coef * v_loss
            try:
                gp = torch.autograd.grad(p_loss, params, retain_graph=True,
                                         allow_unused=True)
                gv = torch.autograd.grad(gv_loss, params, retain_graph=True,
                                         allow_unused=True)

                def _norm(gs):
                    tot = 0.0
                    for g in gs:
                        if g is not None:
                            tot += float((g.detach() ** 2).sum().item())
                    return math.sqrt(tot)
                p_gnorm, v_gnorm = _norm(gp), _norm(gv)
            except RuntimeError:
                p_gnorm = v_gnorm = None
            finally:
                self.opt.zero_grad()
                loss.backward()
        else:
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
            # v3 P0-B：存下逐帧 (value, return)，供训练环做池化 EV
            self.last_ev_pairs = (value.squeeze(-1).detach().cpu().numpy(), rets_np)
        out = {"policy_loss": float(p_loss.item()) / n,
               "value_loss": float(v_loss.item()) / n,
               "value_loss_raw": float(v_mse_sum.item()) / n,
               "entropy": float(ent.mean().item()),
               "ratio_mean": ratio_mean,
               "clip_frac": clip_frac,
               "adv_mean": float(adv_raw.mean()),
               "adv_std": float(adv_raw.std()),
               "grad_norm": grad_norm,
               "value_scale": float(v_scale),
               # EV 用未缩放的原始 MSE 算（与 value_norm 无关，跨版本可比）
               "explained_variance": self.explained_variance(
                   value.squeeze(-1).detach().cpu().numpy(), rets_np),
               "n": n}
        # 仅在真的跑了分解时才带这两项——保持 stats 全为有限浮点，
        # 旧测试的 all(np.isfinite(v) for v in stats.values()) 不受影响。
        if p_gnorm is not None:
            out["p_gnorm"] = float(p_gnorm)
            out["v_gnorm"] = float(v_gnorm)
        return out
