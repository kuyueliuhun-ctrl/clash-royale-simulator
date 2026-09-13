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
import random
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
                 value_norm="none", diagnose_every=0, ret_scaler=None,
                 n_epochs=1, minibatch_size=0, shuffle=False, seed=12345):
        """value_norm: none=旧行为（价值损失不缩放）/ running=按回报运行 std 缩放。
        diagnose_every: >0 时每 N 次 update 额外算一次 p_gnorm/v_gnorm 梯度分解
        （多两次 backward，仅诊断用；0 = 关闭，旧行为）。
        ret_scaler: 断点续训时传入已恢复的 ReturnScaler（None = 新建）。

        **F′（2026-09-12）：真正的 PPO 更新预算。**
        `n_epochs`：同一份 rollout 上重复几轮；`minibatch_size`：每轮切成多大的
        小批（0 或 ≥n = 不切）；`shuffle`：每轮是否打乱样本顺序。

        为什么需要：旧实现 = **1 次 forward / 1 次 backward / 1 次 `opt.step`**，
        而 `train_solo` 喂进来的批是**同一局连续 128 帧**（相邻帧 `corr(R_t,R_{t+1})≈0.99`）
        ⇒ 20k 步只有 156 次梯度步、且每次梯度目标几乎相同，价值头（无论什么架构）
        都学不到东西（实测四种价值架构 EV 全部 ≤0，而表征的 MLP 探针 R² 有 0.24~0.42）。

        兼容性红线：`n_epochs=1, minibatch_size=0, shuffle=False`（默认）时走
        **原单一 pass 分支**，loss 仍是 `sum` 口径、逐位等于旧实现；只有显式开启
        才切到"多轮 × 打乱 × 小批 + `mean` 口径"的新分支（`run_league`/
        `flow_league`/`train_follower`/`train_prophet` 共用本类，默认行为不得变）。
        开启后 `ratio` 会离开 1.000、`clip_frac` 会 >0 —— 这是**正常且期望**的，
        AGENTS 里"ratio≡1.000 是结构性的"只对 `n_epochs=1` 成立。
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
        # —— F′：更新预算（见 __init__ docstring；默认值 = 旧行为）——
        self.n_epochs = max(1, int(n_epochs or 1))
        self.minibatch_size = max(0, int(minibatch_size or 0))
        self.shuffle = bool(shuffle)
        self.rng = random.Random(int(seed))
        #: 累计 `opt.step()` 次数——判读"梯度步够不够"的硬指标
        #: （旧实现恒等于 `updates`，这也是 critic 学不动的直接原因）。
        self.grad_steps = 0
        #: v3 P0-B：最近一次 update 的逐帧 `(value_pred, return)`。
        #: 训练环把它累积到评估窗口，评估时**对全体帧算一次 EV**（池化口径）。
        #: 旧口径"各更新批 EV 取均值"因批=128 连续帧、相邻帧 corr(R)≈0.99，
        #: 批内 Var(R) 仅为全局 0.32 倍 → 把 EV 放大约 3 倍（−0.58 vs −1.74）。
        self.last_ev_pairs = None
        #: F'：**更新前**整批预测算出的 EV（对外报出的口径，与历史 run 可比）。
        self.last_ev_pre = None
        #: 2026-09-13 惰性检验：替代优势（V≡常数）的统计与梯度余弦，见 update() 的 adv_alt。
        self._adv_inert_stats = {}
        self.last_adv_inert_grad = None

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

    def update(self, transitions, ent_coef=None, adv_norm=None, adv_alt=None):
        """transitions: list of dicts {obs, belief, plan, bundle, old_logprob,
        adv, returns, masks, init_hidden(可选)}。

        并行版：用 evaluate_batch 一次前向+反向算完所有 transition（CNN 编码只做一次，
        每个 decoder 步按存活子集批量），梯度与逐条累加完全等价（loss 用 sum 保留量纲）。

        ent_coef / adv_norm：单次更新覆盖（None = 用构造参数）。
        adv_norm: batch=整批中心化（旧默认）/ scale=只除以批 std（不中心化，
        避免“躺平零优势帧”被批均值抬成伪正优势）/ none=不归一化。
        返回 stats 附带 ratio/clip/raw-adv/梯度范数诊断。

        adv_alt（2026-09-13，critic 惰性检验；**默认 None = 逐位旧行为**）：
        与 transitions 等长的**替代优势**（同口径：由调用方用 V≡常数 重算 GAE）。
        给定后**不参与任何被写入梯度的量**，只在诊断命中的更新上多算 1 次前向 + 1 次反传，
        返回 `adv_inert_*` 统计与 `last_adv_inert_grad`（策略损失对替代优势的梯度余弦）。
        见 docs/critic_inertia_prereg_2026-09-13.md。
        """
        self.policy.train()
        if not transitions:
            return {"policy_loss": 0.0, "value_loss": 0.0, "value_loss_raw": 0.0,
                    "entropy": 0.0, "ratio_mean": 1.0, "clip_frac": 0.0,
                    "adv_mean": 0.0, "adv_std": 0.0, "grad_norm": 0.0,
                    "value_scale": 1.0, "explained_variance": 0.0, "n": 0,
                    "ppo_epochs": self.n_epochs, "grad_steps": 0}
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
        n = len(transitions)
        rets_np = np.array([t["returns"] for t in transitions], dtype=np.float32)
        # —— 价值通道量纲对齐（P0-1b）——
        # 先更新运行统计再取 scale：本批即参与统计（均值/方差对单批不敏感）。
        # F′：统计**永远按整份 rollout 更新一次**（不按 minibatch 更新），
        # 否则多轮小批会让 s 随小批抖动，跨 epoch 的目标尺度不再一致。
        if self.value_norm == "running":
            self.ret_scaler.update(rets_np)
            s = self.ret_scaler.scale()
            v_scale = s if s > 0 else 1.0
        elif self.value_norm in ("none", "", None):
            v_scale = 1.0
        else:
            raise ValueError(f"未知 value_norm='{self.value_norm}'（none/running）")

        # 旧分支判定（见 __init__ docstring 的兼容性红线）
        legacy = (self.n_epochs <= 1 and not self.shuffle
                  and (self.minibatch_size <= 0 or self.minibatch_size >= n))
        diag_on = bool(self.diagnose_every) and (self.updates % self.diagnose_every == 0)
        # —— 惰性检验：替代优势的**同一尺度化口径**（与 advs 逐条对应；不改 advs 本身）——
        advs_alt = None
        if adv_alt is not None:
            alt_raw = np.asarray(adv_alt, dtype=np.float32)
            if alt_raw.shape != adv_raw.shape:
                raise ValueError(f"adv_alt 形状 {alt_raw.shape} 与批 {adv_raw.shape} 不一致")
            advs_alt = alt_raw.copy()
            _std_a = float(advs_alt.std())
            if mode == "scale":
                if _std_a > 1e-6:
                    advs_alt = advs_alt / (_std_a + 1e-8)
            elif mode == "batch":
                if _std_a > 1e-6:
                    advs_alt = (advs_alt - advs_alt.mean()) / (_std_a + 1e-8)
            # none：原样（与 advs 同分支）
            _sr, _sa = float(adv_raw.std()), float(alt_raw.std())
            if _sr > 1e-12 and _sa > 1e-12:
                _cc = float(np.corrcoef(adv_raw, alt_raw)[0, 1])
            else:
                _cc = 1.0
            # 关键定义（2026-09-13，跑前修订 1）：真正进损失的是**归一化后**的优势
            # （adv_norm="scale" ⇒ A/σ）。优势整体缩放对梯度是恒等变换，所以
            # 「critic 的贡献」必须量在归一化向量上，否则会把纯缩放当成贡献。
            # resid_frac（raw）只作描述：它同时含"形状差"与"尺度差"。
            _srn = float(advs.std())
            _san = float(advs_alt.std())
            if _san > 1e-12:
                _rn = advs - advs_alt
                _resid_norm = float(_rn.std() / _san)
                _level_shift = float(abs(_rn.mean()) / _san)
            else:
                _resid_norm = _level_shift = None
            self._adv_inert_stats = {
                "adv_inert_corr": _cc,
                "adv_inert_resid_frac": float(np.std(adv_raw - alt_raw) / max(1e-12, _sa)),
                "adv_inert_resid_frac_norm": _resid_norm,
                "adv_inert_level_shift_norm": _level_shift,
                "adv_inert_std_ratio": _sr / max(1e-12, _sa),
                "adv_inert_norm_std_ratio": (_srn / _san if _san > 1e-12 else None),
                "adv_inert_mean_real": float(adv_raw.mean()),
                "adv_inert_mean_alt": float(alt_raw.mean()),
            }
        else:
            self._adv_inert_stats = {}
        self.last_adv_inert_grad = None
        if legacy:
            pack = self._loss_pass(transitions, np.arange(n), advs, rets_np,
                                   coef, v_scale, reduction="sum")
            pack_alt = None
            if diag_on and advs_alt is not None:
                pack_alt = self._loss_pass(transitions, np.arange(n), advs_alt, rets_np,
                                           coef, v_scale, reduction="sum")
            p_gnorm, v_gnorm = self._apply_grad(pack, diag_on, pack_alt=pack_alt)
            self.grad_steps += 1
            out = pack["stats"]
            out.update({"ratio_mean": pack["ratio_mean"],
                        "clip_frac": pack["clip_frac"],
                        "grad_norm": pack["grad_norm"],
                        "explained_variance": pack["ev"],
                        "grad_steps": 1,
                        "ppo_epochs": 1,
                        "ppo_minibatch": n})
            self.last_ev_pairs = pack["ev_pairs"]
        else:
            # —— EV 口径（2026-09-12，F' 首跑发现的口径陷阱，重要）——
            # 必须用**更新前**的预测算 EV：旧实现天然如此（单次 forward 之后才 backward）。
            # F' 若沿用"末轮预测"，读到的是**刚在这 128 帧上训过 12 步**的 in-sample 值
            # ——实测 fprime_20k 末点训练 EV=+0.41，而同权重在 50 局独立 rollout 上
            # EV_global=−0.0255；受控实验同批 EV：更新前 +0.011 / 更新后 +0.042 /
            # **更新后换一条独立 rollout −0.384**。跨版本比较会因此系统性虚高，
            # 并把"记住这 128 帧"误读成"critic 学会了"。
            # 代价：每次 update 多一次整批 no_grad 前向（≈1/16 的更新开销）。
            with torch.no_grad():
                v_pre, _, _ = self.policy.evaluate_batch(
                    [t["obs"] for t in transitions],
                    [t["belief"] for t in transitions],
                    [t["plan"] for t in transitions],
                    [t["bundle"] for t in transitions],
                    [t["masks"] for t in transitions],
                    [t.get("init_hidden") for t in transitions])
                self.last_ev_pairs = (v_pre.squeeze(-1).cpu().numpy(), rets_np)
                self.last_ev_pre = self.explained_variance(
                    v_pre.squeeze(-1).cpu().numpy(), rets_np)
            p_gnorm, v_gnorm, out = self._update_epochs(
                transitions, advs, rets_np, coef, v_scale, diag_on,
                advs_alt=advs_alt)
            # 对外只报更新前口径（与全部历史 run 可比）；末轮 in-sample 值另存对照
            out["explained_variance_insample"] = out["explained_variance"]
            out["explained_variance"] = self.last_ev_pre
        out.update({"adv_mean": float(adv_raw.mean()),
                    "adv_std": float(adv_raw.std()),
                    "value_scale": float(v_scale),
                    "n": n})
        # 惰性检验统计（adv_alt=None 时为空 dict ⇒ 旧 stats 键集不变）
        if self._adv_inert_stats:
            out.update(self._adv_inert_stats)
            out.update(self.last_adv_inert_grad or {})
        # 仅在真的跑了分解时才带这两项——保持 stats 全为有限浮点，
        # 旧测试的 all(np.isfinite(v) for v in stats.values()) 不受影响。
        if p_gnorm is not None:
            out["p_gnorm"] = float(p_gnorm)
            out["v_gnorm"] = float(v_gnorm)
        return out

    # ------------------------------------------------------------------
    # F′：多轮 × 打乱 × 小批的实现（旧单 pass 分支共用同一个 loss 计算）
    # ------------------------------------------------------------------
    def _loss_pass(self, transitions, idx, advs, rets_np, coef, v_scale,
                   reduction="mean"):
        """在 `idx` 指定的样本子集上算一次 loss 及各诊断量。

        `reduction="sum"` = 旧实现的量纲（整批求和，梯度与逐条累加等价）；
        `"mean"` = 标准 PPO 小批口径（`run_league` 等默认路径不走这里）。
        """
        dev = self.policy.device
        sub = [transitions[i] for i in idx]
        lp_new, value, ent = self.policy.evaluate_batch(
            [t["obs"] for t in sub],
            [t["belief"] for t in sub],
            [t["plan"] for t in sub],
            [t["bundle"] for t in sub],
            [t["masks"] for t in sub],
            [t.get("init_hidden") for t in sub])
        old = torch.tensor([t["old_logprob"] for t in sub],
                           dtype=torch.float32, device=dev)
        ratio = torch.exp(lp_new - old)
        adv_t = torch.tensor(advs[idx], dtype=torch.float32, device=dev)
        surr1 = ratio * adv_t
        surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv_t
        rets = torch.tensor(rets_np[idx], dtype=torch.float32, device=dev)
        if reduction == "sum":
            p_loss = -torch.min(surr1, surr2).sum()
            v_mse = F.mse_loss(value.squeeze(-1), rets, reduction="sum")
            ent_term = ent.sum()
        else:
            p_loss = -torch.min(surr1, surr2).mean()
            v_mse = F.mse_loss(value.squeeze(-1), rets, reduction="mean")
            ent_term = ent.mean()
        v_loss = v_mse / (v_scale * v_scale)
        loss = p_loss + self.vf_coef * v_loss - coef * ent_term
        m = max(1, len(idx))
        with torch.no_grad():
            ratio_mean = float(ratio.mean().item())
            clip_frac = float((((ratio < 1 - self.clip) | (ratio > 1 + self.clip))
                               .float().mean().item()))
            v_np = value.squeeze(-1).detach().cpu().numpy()
        return {"loss": loss, "p_loss": p_loss, "v_loss": v_loss,
                "v_mse": v_mse, "ratio_mean": ratio_mean, "clip_frac": clip_frac,
                "ev": self.explained_variance(v_np, rets_np[idx]),
                "ev_pairs": (v_np, rets_np[idx]),
                "stats": {"policy_loss": float(p_loss.item()) / m,
                          "value_loss": float(v_loss.item()) / m,
                          "value_loss_raw": float(v_mse.item()) / m,
                          "entropy": float(ent.mean().item())}}

    def _apply_grad(self, pack, diag_on=False, pack_alt=None):
        """backward + 梯度诊断 + clip + `opt.step()`（一次梯度步）。

        `pack_alt`（可选）：替代优势（V≡常数）在同一 minibatch 上的 loss pack。
        给定且 diag_on 时，额外算策略损失对它的梯度，写入 `self.last_adv_inert_grad`
        （`grad_cos` / `grad_norm_ratio`）——**不改变被写入梯度的任何量**。
        """
        p_gnorm = v_gnorm = None
        # 注意：这里**不重置** last_adv_inert_grad —— 诊断只在 (ep==0, bi==0) 那一次
        # 调用上命中，而之后同一次 update 还有 15 次 _apply_grad 调用；早期版本在这里
        # 每调用一次清一次，把刚算出的 grad_cos 擦掉（端到端 smoke 抓到：日志里
        # adv_inert 统计正常但 n_grad_cos=0）。重置职责归 update() 开头。
        # —— 梯度成分诊断（P0-1a，可证伪"评论家主导"）——
        # p_loss / vf_coef*v_loss 分别求梯度范数后再走原有合并 backward（多两次
        # 反传，仅诊断时启用）。两者共享参数，范数之比即"谁在推动更新"。
        if diag_on:
            params = [p for p in self.policy.parameters() if p.requires_grad]
            gv_loss = self.vf_coef * pack["v_loss"]
            try:
                gp = torch.autograd.grad(pack["p_loss"], params, retain_graph=True,
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
                if pack_alt is not None:
                    ga = torch.autograd.grad(pack_alt["p_loss"], params,
                                             retain_graph=True, allow_unused=True)
                    num = na = 0.0
                    for g1, g2 in zip(gp, ga):
                        if g1 is None or g2 is None:
                            continue
                        num += float((g1.detach() * g2.detach()).sum().item())
                        na += float((g2.detach() ** 2).sum().item())
                    na = math.sqrt(na)
                    if p_gnorm > 0.0 and na > 0.0:
                        self.last_adv_inert_grad = {
                            "grad_cos": num / (p_gnorm * na),
                            "grad_norm_ratio": p_gnorm / na}
            except RuntimeError:
                p_gnorm = v_gnorm = None
            finally:
                self.opt.zero_grad()
                pack["loss"].backward()
        else:
            self.opt.zero_grad()
            pack["loss"].backward()
        pack["grad_norm"] = float(torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.max_grad_norm))
        self.opt.step()
        return p_gnorm, v_gnorm

    def _plan_batches(self, n):
        """一轮内的样本顺序切分（`shuffle` 时打乱；小批不足则退化为整批）。"""
        order = list(range(n))
        if self.shuffle:
            self.rng.shuffle(order)
        if self.minibatch_size <= 0 or self.minibatch_size >= n:
            return [np.array(order, dtype=np.int64)]
        return [np.array(order[i:i + self.minibatch_size], dtype=np.int64)
                for i in range(0, n, self.minibatch_size)]

    def _update_epochs(self, transitions, advs, rets_np, coef, v_scale, diag_on,
                       advs_alt=None):
        """F′ 主循环：`n_epochs` 轮 × 小批 ×（可选）打乱，每小批一次 `opt.step()`。

        统计口径（判读时注意）：
        - `policy_loss`/`value_loss`/`entropy`：按小批样本数加权平均（跨全部轮次）；
        - `ratio_mean`/`clip_frac`/`grad_norm`：**只取最后一轮**（标准 PPO 报告口径——
          第一轮 ratio 仍≈1、clip≈0，混着平均会把"策略确实动了"稀释掉）；
        - `explained_variance` 与 `last_ev_pairs`：**最后一轮**的逐帧 `(v,R)` 池化
          （每帧只被预测一次，与旧口径同尺度，可直接跨版本比较）。
        """
        n = len(transitions)
        agg = {"policy_loss": 0.0, "value_loss": 0.0, "value_loss_raw": 0.0,
               "entropy": 0.0}
        w_tot = 0
        p_gnorm = v_gnorm = None
        n_batches = 1
        ratio_last = 1.0
        clip_last = 0.0
        gnorm_sum, gnorm_cnt = 0.0, 0
        ev_last = 0.0
        for ep in range(self.n_epochs):
            batches = self._plan_batches(n)
            n_batches = len(batches)
            ev_v, ev_r = [], []
            r_acc, c_acc = 0.0, 0.0
            w_ep = 0          # 本轮样本数（ratio/clip 必须除本轮，不能除全部轮次）
            for bi, idx in enumerate(batches):
                pack = self._loss_pass(transitions, idx, advs, rets_np,
                                       coef, v_scale, reduction="mean")
                # 梯度分解只在**第一轮的第一个小批**做（否则每步 3 次反传）
                _diag_here = diag_on and ep == 0 and bi == 0
                pack_alt = None
                if _diag_here and advs_alt is not None:
                    # 惰性检验：同一 minibatch 上、同一尺度化口径下的替代优势
                    pack_alt = self._loss_pass(transitions, idx, advs_alt, rets_np,
                                               coef, v_scale, reduction="mean")
                pg, vg = self._apply_grad(pack, _diag_here, pack_alt=pack_alt)
                if pg is not None:
                    p_gnorm, v_gnorm = pg, vg
                self.grad_steps += 1
                m = len(idx)
                for k in agg:
                    agg[k] += pack["stats"][k] * m
                w_tot += m
                r_acc += pack["ratio_mean"] * m
                c_acc += pack["clip_frac"] * m
                w_ep += m
                if ep == self.n_epochs - 1:
                    gnorm_sum += pack["grad_norm"]
                    gnorm_cnt += 1
                    ev_v.append(pack["ev_pairs"][0])
                    ev_r.append(pack["ev_pairs"][1])
            if ep == self.n_epochs - 1:
                ev_v = np.concatenate(ev_v) if ev_v else np.zeros(0, dtype=np.float32)
                ev_r = np.concatenate(ev_r) if ev_r else np.zeros(0, dtype=np.float32)
                self.last_ev_pairs = (ev_v, ev_r)
                ev_last = self.explained_variance(ev_v, ev_r)
                ratio_last = r_acc / max(1, w_ep)
                clip_last = c_acc / max(1, w_ep)
        w = max(1, w_tot)
        out = {k: v / w for k, v in agg.items()}
        out["ratio_mean"] = ratio_last
        out["clip_frac"] = clip_last
        out["grad_norm"] = gnorm_sum / max(1, gnorm_cnt)
        out["explained_variance"] = ev_last
        out["ppo_epochs"] = self.n_epochs
        out["ppo_minibatch"] = (self.minibatch_size if self.minibatch_size > 0
                                else n)
        out["grad_steps"] = self.n_epochs * n_batches
        return p_gnorm, v_gnorm, out
