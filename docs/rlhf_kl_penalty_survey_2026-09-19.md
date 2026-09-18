# RLHF 家族的奖惩机制调研 —— 重点：「KL penalty to reference policy」这个惩罚项

> **取数日期**：2026-09-19（CST；UTC 2026-09-18 19:33 起）
> **调研对象**：InstructGPT / ChatGPT 系（OpenAI）、Anthropic HH-RLHF、开源实现 TRL / DeepSpeed-Chat
> **任务**：作为另一个项目（本仓 `clash-royale-simulator`）的**借鉴来源**
> **纪律声明**：本文**区分「论文原文事实」与「二手解读」**；每个关键结论附 URL；查不到的一律标 `[NOT-FOUND]` / 「未验证」；**不做跨项目数字对拍**。
> **⚠️ 对任务书的三处前提更正**（均有取证）：① InstructGPT v1 只有 **2 个编号公式**（Eq.1 RM 损失、Eq.2 RL 目标），**没有 Eq.3/4/5**；② Anthropic HH 论文**没有**写出 pairwise sigmoid 损失（那个公式在 Askell et al. 2021，arXiv:2112.00861）；③ TRL 的 `PPOTrainer` **已于 2026-09-04 从 `main` 移除**（commit `700b845c`，「Remove PPOTrainer (#7020)」）——下文引的是**存档 tag `v0.11.2`**。另：任务书若引用 `arXiv:2402.14740` 作为「Secrets of RLHF Part I」是**错的**，该文是 "Back to Basics"（REINFORCE 风格）；Secrets of RLHF Part I 是 **arXiv:2307.04964**。

---

## §1 标识与链接

| 对象 | 标识 | URL | 备注 |
|---|---|---|---|
| **InstructGPT（= ChatGPT 的方法学祖本）** | **arXiv:2203.02155v1**，"Training language models to follow instructions with human feedback"，Ouyang et al., OpenAI，2022-03-04 提交 | 摘要 <https://arxiv.org/abs/2203.02155>　HTML <https://arxiv.org/html/2203.02155v1>　PDF <https://arxiv.org/pdf/2203.02155v1> | ✅ 取数验证。全文（含附录 C.4 / E.7）经 v1 HTML + v1 TeX 源码 tarball 逐字核对 |
| **Anthropic HH-RLHF** | **arXiv:2204.05862v1**，"Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback"，Bai, Jones, Ndousse, Askell, Chen, ... Kaplan（31 作者），2022-04-12 提交 | 摘要 <https://arxiv.org/abs/2204.05862>　HTML <https://arxiv.org/html/2204.05862v1>　PDF <https://arxiv.org/pdf/2204.05862v1>　数据 <https://github.com/anthropics/hh-rlhf> | ✅ 取数验证（PDF 74 页 + HTML 全文） |
| **HH 的 PM 损失出处（必须另引）** | **arXiv:2112.00861v3**，"A General Language Assistant as a Laboratory for Alignment"，Askell et al.，§3.1 Eq.(3.1) | <https://arxiv.org/html/2112.00861v3> | HH §3.1 自称 PM 设置与本文 "identical" |
| **TRL（HuggingFace）** | 经典 `PPOTrainer` **在 tag `v0.11.2`**；`main` 上已删除 | 代码 <https://github.com/huggingface/trl/blob/v0.11.2/trl/trainer/ppo_trainer.py>　配置 <https://github.com/huggingface/trl/blob/v0.11.2/trl/trainer/ppo_config.py>　控制器 <https://github.com/huggingface/trl/blob/v0.11.2/trl/trainer/utils.py>　文档源 <https://github.com/huggingface/trl/blob/v0.15.2/docs/source/ppo_trainer.md> | ⚠️ `https://huggingface.co/docs/trl/main/en/ppo_trainer` **本环境无法访问**（host 不可达，重试 3 次失败），且 `main` 上该文档文件已删。上文文档源 URL 是**验证过存在并读到内容**的替代品 |
| **DeepSpeed-Chat（Microsoft/DeepSpeed）** | `applications/DeepSpeed-Chat`，仓库 `microsoft/DeepSpeedExamples` 现重定向到 **`deepspeedai/DeepSpeedExamples`** | README <https://github.com/deepspeedai/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/README.md>　PPO <https://github.com/deepspeedai/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/dschat/rlhf/ppo_trainer.py> | ⚠️ 任务书里的旧路径 `training/step3_rlhf_finetuning/ppo_trainer.py` **在 master 上已不存在**：commit `e86d0c6d`（"Refactor deepspeed-chat into a python package (#731)"，2023-11-06）把它 **rename** 到 `dschat/rlhf/ppo_trainer.py` |

**辅助一手源（本报告 §4 用）**

| 对象 | 标识 | URL |
|---|---|---|
| Stiennon et al. 2020（RLHF 三阶段祖本 + KL 作用原文） | arXiv:2009.01325 | <https://ar5iv.labs.arxiv.org/html/2009.01325> |
| Gao, Schulman, Hilton 2022（reward model 过优化标度律） | arXiv:2210.10760v1 | <https://arxiv.org/abs/2210.10760> / <https://arxiv.org/html/2210.10760v1> |
| LIMA（IL 数据质量决定上限） | arXiv:2305.11206v1 | <https://arxiv.org/abs/2305.11206> / <https://arxiv.org/html/2305.11206v1> |
| Secrets of RLHF in LLMs Part I: PPO | **arXiv:2307.04964** | <https://arxiv.org/abs/2307.04964> / <https://ar5iv.labs.arxiv.org/html/2307.04964> |

---

## §2 三阶段结构与损失（逐项原文引用 + URL）

> 说明：**InstructGPT v1 全文只有两个编号公式**。其「三阶段」是 **SFT → RM → PPO**（Table/Fig.2 与 §3.5 的小标题明确），但**只有 RM 与 PPO 两步给了公式**。

### 2.1 三阶段总览

| | **① IL / SFT** | **② Reward Model（偏好数据）** | **③ PPO（奖励 = RM 分数 − β·KL）** |
|---|---|---|---|
| **InstructGPT** | 人类示范 → 监督微调；**论文未给损失公式** | `(K choose 2)` pairwise 排序损失（Eq.1） | Eq.2：`E[r_θ] − β·log(π_RL/π_SFT)`（+ γ 预训练项） |
| **Anthropic HH** | **无**显式 SFT/IL 阶段（PM 先在 LM 语料上预训练） | PM 训练；**该文未写损失公式**（引 Askell 2021 Eq.3.1） | Eq.(4.1)：`r_total = r_PM − λ_KL·D_KL(policy‖policy_0)` |
| **TRL / DeepSpeed-Chat** | Step 1 SFT | Step 2 RM（pairwise） | Step 3：逐 token `−kl_coef·(logp − ref_logp)` + 末 token 加 RM 分数 |

### 2.2 ① SFT / IL：损失是什么？

**InstructGPT** —— **论文（v1）没有给出 SFT 损失方程**，只有一句「用监督学习微调」：

> "We fine-tune GPT-3 on our labeler demonstrations using supervised learning. We trained for 16 epochs, using a cosine learning rate decay, and residual dropout of 0.2."
> — InstructGPT §3.5 Supervised fine-tuning (SFT)，<https://arxiv.org/html/2203.02155v1>

全 v1 源码 grep `per-token cross entropy` / SFT 损失式 = **`[NOT-FOUND]`**（唯一的 "cross-entropy" 命中出现在 RM 段里转述 Stiennon et al.）。§C.1 只给超参：
> "For our 1.3B and 6B models, we use an LR of 9.65e-6 and a batch size of 32. For 175B, we use a LR of 5.03e-6 and a batch size of 8." — §C.1，同上 URL

**「IL 损失 = 对示范的最大似然（token 级交叉熵）」这句话的可靠出处是 Stiennon et al. 2020**（InstructGPT 明确自称 follow 该文）：

> "When applying them to a specific task, they are usually fine-tuned using supervised learning, often to maximize the log probability of a set of human demonstrations."
> — Stiennon et al. 2020 §1，<https://ar5iv.labs.arxiv.org/html/2009.01325>

> "as well as the maximum likelihood objective has no distinction between important errors (e.g. making up facts) and unimportant errors (e.g. selecting the precise word from a set of synonyms); models are incentivized to place probability mass on all human demonstrations, including those that are low-quality"
> — 同上，§1（这句话同时是「IL 阶段固有缺陷」的原文依据）

### 2.3 ② RM：偏好数据 + 损失

**InstructGPT Eq.(1)**（逐字，含 `(K choose 2)` 归一）：

```
loss(θ) = −(1 / (K choose 2)) · E_{(x, y_w, y_l) ~ D} [ log( σ( r_θ(x, y_w) − r_θ(x, y_l) ) ) ]
```

> "where r_θ(x, y) is the scalar output of the reward model for prompt x and completion y with parameters θ, y_w is the preferred completion out of the pair of y_w and y_l, and D is the dataset of human comparisons."
> — InstructGPT §3.5 Reward modeling (RM)，Eq.(1)，<https://arxiv.org/html/2203.02155v1>

配套原文（同一节）：

> "They use a cross-entropy loss, with the comparisons as labels—the difference in rewards represents the log odds that one response will be preferred to the other by a human labeler."

> "Instead, we train on all (K choose 2) comparisons from each prompt as a single batch element."

> "Finally, since the RM loss is invariant to shifts in reward, we normalize the reward model using a bias so that the labeler demonstrations achieve a mean score of 0 before doing RL."

§C.2 细节：`"We trained a single 6B reward model which we used for all PPO models of all sizes."` / `"We trained for a single epoch ... at a learning rate of lr = 9e-6 ... batch size of 64."` / `"Therefore, a single batch could contain up to 64 × (K choose 2) ≤ 2,304 comparisons."` / `"Ties were dropped."`

**RM 数据规模**（§3.2）：
> "The SFT dataset contains about 13k training prompts (from the API and labeler-written), the RM dataset has 33k training prompts (from the API and labeler-written), and the PPO dataset has 31k training prompts (only from the API)."

**Anthropic HH** —— **该文没有写 PM 损失公式**（PDF 74 页 + HTML 全文 grep `sigmoid` / `logistic` / `Bradley` / `loss` 对损失函数 = 0 命中）。它只给「分数差 → 偏好概率」的校准式 **Eq.(4.2)**：
`P(A>B) = 1 / (1 + e^{r_PM(B) − r_PM(A)})` — HH §4.1，<https://arxiv.org/html/2204.05862v1>

其 PM 损失必须引 **Askell et al. 2021 §3.1 Eq.(3.1)**：
`L_PM = log(1 + e^{r_bad − r_good})`（等价于 `−log σ(r_good − r_bad)`），"and for batched sample pairs we take the mean over all pairs."
— <https://arxiv.org/html/2112.00861v3>；HH §3.1 原话：`"Our preference model training setup is also identical to that in [Askell et al., 2021] ..."`

### 2.4 ③ PPO：奖励 = RM 分数 − β·KL(π‖π_SFT)

**InstructGPT Eq.(2)**（逐字；注意**印刷式里没有可见的 Σ**，逐 token 是散文说明的）：

```
objective(φ) = E_{(x,y) ~ D_{π_φ^RL}} [ r_θ(x,y) − β · log( π_φ^RL(y | x) / π^SFT(y | x) ) ]
             + γ · E_{x ~ D_pretrain} [ log( π_φ^RL(x) ) ]
```

> "In addition, we add a per-token KL penalty from the SFT model at each token to mitigate over-optimization of the reward model."
> — InstructGPT §3.5 Reinforcement learning (RL)，<https://arxiv.org/html/2203.02155v1>　★ 这是全篇**唯一**解释 KL 项存在理由的句子

> "The KL reward coefficient, β, and the pretraining loss coefficient, γ, control the strength of the KL penalty and pretraining gradients respectively. For 'PPO' models, γ is set to 0."

**Anthropic HH Eq.(4.1)**：

> "To stabilize RL training, we use Proximal Policy Optimization (PPO) [Schulman et al., 2017]. We also follow other work [Stiennon et al., 2020] and apply an empirically-estimated KL penalty term in the reward, with the total reward given by"
> `r_total = r_PM − λ_KL · D_KL(policy ‖ policy_0)`　**Eq.(4.1)**
> "where λ_KL ≥ 0 is a hyperparameter."
> — HH §4.1，<https://arxiv.org/html/2204.05862v1>

> "Note that here the 'KL' is more precisely D_KL(π‖π_0), where π denotes the policy distribution (and π_0 the initial policy), as evaluated empirically on the samples drawn from the policy during training."
> — HH §4.3，同上 URL

**⚠️ 关键事实区分（对 §5 至关重要）**：HH 论文**把 KL 写成序列级**（Eq.4.1 不带 token 求和），且全文 grep `per-token` / `per token` / `at each token` = **0 命中** ⇒ **HH 的 KL 是 per-token 还是序列级：`[NOT-FOUND]`（该文根本没写）**。而且 HH 自己就把这个项**降级**了：

> "In practice we use a very small value of λ_KL = 0.001, which likely has a very minor impact during most of RL training (as D_KL < 100 typically), and might actually be wholly unnecessary."
> — HH §4.1，同上 URL

**Stiennon et al. 2020（KL 作用的最完整原文）**：

> "This KL term serves two purposes. First, it acts as an entropy bonus, encouraging the policy to explore and deterring it from collapsing to a single mode."
> "Second, it ensures the policy doesn't learn to produce outputs that are too different from those that the reward model has seen during training."
> — Stiennon et al. 2020 §3.4（Human feedback policies），<https://ar5iv.labs.arxiv.org/html/2009.01325>

★ **这两句是「KL 为什么必须有」的最强一手依据，而且它们把 KL 的作用拆成了两个独立理由**——理由①（熵/防塌缩）**不依赖 RM**；理由②（不越出 RM 训练分布）**完全依赖 RM 是学出来的**。这个拆分是 §5 结论的支点。

---

## §3 KL 惩罚项：作用 / 实现 / 取值

### 3.1 为什么要有它 —— 分三条，**逐条标依赖**

| # | 理由（原文） | 出处 | 依赖「学出来的 RM」吗？ |
|---|---|---|---|
| ① | "it acts as an entropy bonus, encouraging the policy to explore and deterring it from collapsing to a single mode" | Stiennon §3.4 | **不依赖** |
| ② | "it ensures the policy doesn't learn to produce outputs that are too different from those that the reward model has seen during training" | Stiennon §3.4 | **完全依赖**（RM 有训练分布） |
| ③ | "a per-token KL penalty from the SFT model at each token **to mitigate over-optimization of the reward model**" | InstructGPT §3.5 | **完全依赖**（RM 是 imperfect proxy） |

**⚠️ 反面一手证据（必须并列，否则结论偏）**：Gao et al. 2022（OpenAI）在**受控合成实验**里测 KL 惩罚，结论与「KL 防 reward hacking」的流行读法**相冲突**：

> "The KL penalty only causes the gold RM score to converge earlier, but does not affect the KL_RL-gold reward frontier, and so the effect of the penalty on the gold score is akin to early stopping."
> — arXiv:2210.10760v1 §3.6 Effect of KL Penalty，<https://arxiv.org/html/2210.10760v1>

> "Because we observe that using KL penalty has a strictly larger proxy-gold gap, we set KL penalty to 0 for all other RL experiments in this paper."
> — 同上 §3.6（**即：显式 KL 惩罚让「代理分数−金标分数」的缺口更大**）

该文并给出受控标度律（RL 侧）：
`R_RL(d) = d·(α_RL − β_RL·log d)`，其中 `d := sqrt(D_KL(π‖π_init))` — 同上 §1、§3.3

⇒ **可核查的结论**：在受控条件下，显式 KL 惩罚**不改善**真奖励前沿，只改变**优化速度**（等价早停）。原文亦承认：「we have seen some evidence that this result could be particularly sensitive to hyperparameters.」以及「We do not know why this indirect effect appears to lead to less overoptimization than an explicit KL penalty.」（同 §3.6）

### 3.2 per-token 还是序列级？

| 对象 | 结论 | 证据 |
|---|---|---|
| InstructGPT | **per-token**（散文），公式无 Σ | "per-token KL penalty from the SFT model **at each token**" — §3.5 |
| Anthropic HH | **`[NOT-FOUND]`** —— 该文未说明；Eq.4.1 写成序列级 | grep `per-token`/`at each token` = 0 命中 |
| TRL v0.11.2 | **per-token（逐 token 张量）**，RM 分数**只加在最后一个非 mask token** | 见下方代码 |
| DeepSpeed-Chat | **per-token（不跨 token 求和）**，RM 分数**只加在最后一个 response token** | 见下方代码 |
| Secrets of RLHF Part I | per-token | Eq.(19) 以 `π^RL_θ(y_i\|x)` 逐 token 表述，<https://ar5iv.labs.arxiv.org/html/2307.04964> |

### 3.3 β / λ_KL / kl_coef 的典型取值（全部为**验证到的原文/代码值**）

| 来源 | 符号 | 值 | 出处 |
|---|---|---|---|
| InstructGPT | β | **0.02** | §C.4："These models are also used to compute the KL reward, in the same way as Stiennon et al., 2020, with **β=0.02** (see Equation 2)." |
| InstructGPT（E.7 调参） | β | **最优 ≈ 0.01 ~ 0.02**；0 与 2 都差 | §E.7："Both 0 and 2 for KL reward coefficient result in poor performance. The optimal value is around 0.01 and 0.02." |
| InstructGPT（E.6 反证） | β | **加大到 2.0（默认的 100 倍）仍修不好回退** | §E.6/Fig.34："We find that even by increasing the KL reward coefficient to 2.0, which is 100 times of the default value, the regressions still cannot be fixed." |
| InstructGPT（γ，**不是** KL） | γ | **27.8** | §C.4："We multiply the pretraining gradients by a coefficient, γ=27.8 (see Equation 2)" |
| Anthropic HH | λ_KL | **0.001** | §B.1："...a KL reward coefficient of **λ_KL = 0.001** (4.1), PPO clipping ϵ = 0.2, discount factor γ = 1, and no entropy bonus."（"We performed a variety of hyperparameter scans..."） |
| TRL v0.11.2 | `init_kl_coef` | **0.2** | `trl/trainer/ppo_config.py:161` |
| TRL v0.11.2 | `target`（自适应目标） | **6.0** | 同上（docstring："Target KL value for adaptive KL control."） |
| TRL v0.11.2 | `horizon` | **10000.0** | 同上 |
| TRL v0.11.2 | `target_kl`（**另一个参**，早停门限） | **1.0** | `ppo_config.py:181`，docstring："Stop early if we exceed this value by over 50%" |
| TRL（移除前 `trl/experimental/ppo`） | `kl_coef` | **0.05** | commit `56f6675…` 的 `ppo_config.py` |
| DeepSpeed-Chat | `kl_ctl` | **0.1**（硬编码，注释 `# Those value can be changed`） | `dschat/rlhf/ppo_trainer.py:65-71` |

⚠️ **更正**：任务书猜的 `kl_coef * (1 + horizon*(target-current))` 或「TRL `target_kl` 默认 6 / 0.01」**都不是 TRL 的实现**。TRL 的自适应控制器用的是**比例误差 + ±0.2 裁剪**（见下）。

### 3.4 开源实现的具体代码位置（可点击 + 行号）

**（A）TRL v0.11.2 — `trl/trainer/ppo_trainer.py`**

KL 映射（四种口径，逐 token）— `_kl_penalty`，**lines 1150–1162**：
```python
    def _kl_penalty(self, logprob, ref_logprob) -> torch.FloatTensor:
        if self.config.kl_penalty == "kl":
            return logprob - ref_logprob
        if self.config.kl_penalty == "abs":
            return (logprob - ref_logprob).abs()
        if self.config.kl_penalty == "mse":
            return 0.5 * (logprob - ref_logprob).square()
        if self.config.kl_penalty == "full":
            return F.kl_div(ref_logprob, logprob, log_target=True, reduction="none").sum(-1)
```
— <https://github.com/huggingface/trl/blob/v0.11.2/trl/trainer/ppo_trainer.py>

KL 进入奖励的位置 — `compute_rewards`，**lines 1135–1146**：
```python
            kl = self._kl_penalty(logprob, ref_logprob)
            non_score_reward = -self.kl_ctl.value * kl
            reward = non_score_reward.clone()
            last_non_masked_index = mask.nonzero()[-1]
            # reward is preference model score + KL penalty
            reward[last_non_masked_index] += score
```
⇒ **逐 token 扣 KL，RM 分数只在最后一个有效 token 加一次**（与 DeepSpeed-Chat 同构）。

自适应控制器 — `trl/trainer/utils.py` **lines 54–69**（`class AdaptiveKLController`，docstring 指向 HF 论文 1909.08593）：
```python
    def update(self, current, n_steps):
        target = self.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult
```
更新调用：`self.kl_ctl.update(stats["objective/kl"], self.config.batch_size * self.accelerator.num_processes)`（`ppo_trainer.py` lines 881–884）。
`FixedKLController.update` = **空操作**（同文件 lines 72–79）；选择逻辑 `ppo_trainer.py` lines 319–322。

参考模型来源：`ref_logprobs` 来自 `self.ref_model`（非 PEFT 且未传入时由 `create_reference_model(self.model, ...)` 构建），每次 `with torch.no_grad()` 前向 — `ppo_trainer.py` lines 745–763。

TRL 自己的退化诊断（**可执行的安全阀**）— `ppo_trainer.py` **lines 1311–1317**：
```python
        if mean_kl.item() < -1.0:
            warnings.warn(f"KL divergence is starting to become negative: {mean_kl.item():.2f} - this might be a precursor for failed training.")
```

TRL 文档对 `objective/non_score_reward` 的定义—`docs/source/ppo_trainer.md` @ v0.15.2：
> "* `objective/non_score_reward`: The mean reward from non-score-related sources, basically **`beta * kl.sum(1)`**, where `beta` is the KL penalty coefficient and `kl` is the **per-token KL divergence**."
> "* `objective/rlhf_reward`: The mean RLHF reward, which is `score - non_score_reward`."
— <https://github.com/huggingface/trl/blob/v0.15.2/docs/source/ppo_trainer.md>（⚠️ 文档符号约定与代码相反，代码里逐 token 是 **减** KL）

**（B）DeepSpeed-Chat — `applications/DeepSpeed-Chat/dschat/rlhf/ppo_trainer.py`（master）**

默认值 **lines 65–71**：
```python
        # Those value can be changed
        self.kl_ctl = 0.1
        self.clip_reward_value = 5
        self.cliprange = 0.2
        self.cliprange_value = 0.2
        self.gamma = 1.0
        self.lam = 0.95
```
KL 计算 **lines 181–194**：
```python
    def compute_rewards(self, prompts, log_probs, ref_log_probs, reward_score, action_mask):
        kl_divergence_estimate = -self.kl_ctl * (log_probs - ref_log_probs)
        rewards = kl_divergence_estimate
        start = prompts.shape[1] - 1
        ends = start + action_mask[:, start:].sum(1) + 1
        reward_clip = torch.clamp(reward_score, -self.clip_reward_value, self.clip_reward_value)
        for j in range(batch_size):
            rewards[j, start:ends[j]][-1] += reward_clip[j]
        return rewards
```
⇒ **逐 token KL（未求和）+ RM 分数裁剪到 ±5 只加末 token**。
— <https://github.com/deepspeedai/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/dschat/rlhf/ppo_trainer.py>

参考模型：`self.ref_model = self.rlhf_engine.ref`；`'ref_logprobs': gather_log_probs(logits_ref[:, :-1, :], seq[:, 1:])`（同文件 lines 154–174）。

三阶段自述（README，为「开源文档链接」这条要求）：
> "It can automatically take your favorite pre-trained large language models through an OpenAI InstructGPT style **three stages** to produce your very own high-quality ChatGPT-style model."
> "#### 🕐 Step 1 - [Supervised Fine-Tuning] / 🕑 Step 2 - [Reward Model] / 🕒 Step 3 - [Reinforcement Learning with Human Feedback]"
— <https://github.com/deepspeedai/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/README.md>

---

## §4 实践中的坑（**只收带数字或明确结论的**）

### 4.1 Reward hacking / RM 过优化

| 证据 | 数字/结论 | 来源 | 性质 |
|---|---|---|---|
| 过优化存在且可标度 | 金标分数随 `d=sqrt(D_KL)` 先升后降；RL 拟合式 `R_RL(d)=d(α_RL−β_RL log d)` | <https://arxiv.org/html/2210.10760v1> | **论文原文** |
| RM 数据量硬门槛 | "for amounts of data less than around **2,000 comparisons**, there is very little improvement over near-chance loss" | 同上 §3.3 | **论文原文** |
| 该实验规模 | "We generate **100,000** synthetic comparisons and reserve **10%** as held out"；RM 训练用 **90,000** | 同上 §2.1/§3.2 | **论文原文** |
| ⚠️ 任务书传言的「峰值在 ~20k 样本」 | **该文不存在此数字**（grep `20,000`/`20k` = 0 命中）；峰值以 KL/`d` 报，不以样本数报 | 同上 | **更正（防止臆造）** |
| 显式 KL 惩罚不减缺口 | "using KL penalty has a **strictly larger** proxy-gold gap"；故其余实验 **KL 惩罚设为 0** | 同上 §3.6 | **论文原文** |
| 早期一手观察（摘要级已验） | "we show that ... its coefficients scale smoothly with the number of reward model parameters"；KL 惩罚只等价早停 | 同上 | **论文原文** |
| 学出来的奖励会与真人偏好**反相关** | "as we optimize further, true preferences fall off compared to the prediction, and **eventually the reward model becomes anti-correlated with human preferences**. ... Similar behavior has been observed in learned reward functions in the robotics domain" | Stiennon et al. §4.3，<https://ar5iv.labs.arxiv.org/html/2009.01325> | **论文原文** |
| PM 在更高分处失校准 | "the train and test PM's disagree, with the train PM assigning a higher mean reward"（**约 150k 训练样本**之后） | HH 论文 Fig.4 caption，<https://arxiv.org/html/2204.05862v1> | **论文原文** |
| PM 过优化 = 真偏好变差 | "The divergence is likely an indication that the preference model is less robust and more easily exploited at higher rewards. That is, the policy has been **over-optimized on the train PM**" | HH §4.2，同上 | **论文原文** |
| PM 高分 ≠ 真人高评 | "PMs also become less calibrated at higher scores, so higher rewards do not necessarily imply better performance."；§B.4 "the naive PM predictions significantly **overestimate** the empirical Elos" | HH §4.1 / §B.4 | **论文原文** |
| RM 规模/数据标度 | 数据翻倍 → RM 验证精度 **+~1.1%**；模型翻倍 → **+~1.8%** | Stiennon §4.3，同上 | **论文原文** |
| **具体数值的 reward hacking（工程侧）** | "The unscorable completion (idx 7) gets a **+0.264 advantage** without the defense — a positive policy gradient despite carrying no learning signal." | TRL PR #6429，<https://github.com/huggingface/trl/pull/6429> | **[二手：issue/PR 报告]** |

### 4.2 KL 系数调参失败模式

| 现象 | 数字/原文 | 来源 | 性质 |
|---|---|---|---|
| 最优区间窄 | InstructGPT：β=0 与 β=2 **都差**，最优 **0.01~0.02** | arXiv:2203.02155 §E.7 | 论文原文 |
| 加大 KL 救不了 alignment tax | "even by increasing the KL reward coefficient to **2.0**, which is **100 times** of the default value, the regressions still cannot be fixed" | 同上 §E.6 | 论文原文 |
| KL 太大反而掉分 | "As expected, too large KL reward coefficient causes a significant drop in the validation reward." | 同上 §E.6 | 论文原文 |
| **负 KL 爆炸是失败前兆** | 维护者（lvwerra）亲述："A common failure mode of the training is that the generation kwargs are not correctly set which leads to **negative KL-divergence which is almost always a precursor for failed training**." | TRL issue **#235**，<https://github.com/huggingface/trl/issues/235> | **[二手：维护者 issue]** |
| 实测失败率 | "I noticed **3 out 10 experiments** would experience some amount of **negative KL explosion**, though one of them recovered" | TRL issue **#417**，<https://github.com/huggingface/trl/issues/417> | **[二手：issue 报告]** |
| KL 压倒奖励（目标失衡） | "after step12 it looks like the model **cares more about lowering the kl and stops boosting the reward**"（配置 `init_kl_coef: 0.2, target: 6, target_kl: 1`） | TRL issue **#1178**，<https://github.com/huggingface/trl/issues/1178> | **[二手：issue 报告]** |
| 校准失败的自承认 | "we have seen some evidence that this result could be particularly sensitive to hyperparameters." | arXiv:2210.10760 §3.6 | 论文原文 |

### 4.3 「IL 阶段数据质量决定上限」的证据

| 证据 | 数字/结论 | 来源 | 性质 |
|---|---|---|---|
| 1000 条精选 > 5 万条未筛选（且**不做 RLHF**） | "fine-tuned with the standard supervised loss on **only 1,000 carefully curated** prompts and responses, **without any reinforcement learning or human preference modeling**" | LIMA 摘要，<https://arxiv.org/abs/2305.11206> | 论文原文 |
| 数据质量消融（**带数字**） | "there is a significant **0.5 point difference** between models trained on the **filtered and unfiltered** data sources." | <https://arxiv.org/html/2305.11206v1> | 论文原文 |
| 数据**数量**无用 | "surprisingly, **doubling the training set does not improve response quality**." | 同上 | 论文原文 |
| 未做 RLHF 仍胜出做过 RLHF 的模型 | "as high as ... **65% versus DaVinci003, which was trained with human feedback**"；"what is striking about this result is the fact that DaVinci003 was trained with RLHF" | 同上 | 论文原文 |
| 理论假设 | "Superficial Alignment Hypothesis: A model's knowledge and capabilities are learnt almost entirely **during pretraining**, while alignment teaches it **which subdistribution of formats** should be used" | 同上 | 论文原文 |
| RM 质量决定策略上限 | "We find that the quality of the reward model **directly determines the upper bound of the policy model**" | arXiv:2307.04964，<https://ar5iv.labs.arxiv.org/html/2307.04964> | 论文原文 |
| ⚠️ 未找到 | **没有任何论文**写「弱 SFT 训出的 RM 无法超过该 SFT 的质量」——不要把这句当引用 | — | **`[NOT-FOUND]`** |
| ⚠️ 反例提醒 | InstructGPT **自己**报告过 SFT 验证损失 1 epoch 后就过拟合、但多训几个 epoch 仍**同时**改善 RM 分与人类偏好 ⇒ 「早停看验证损失」不是普适判据 | InstructGPT §3.5 | 论文原文 |

### 4.4 交叉验证过的「RLHF 三阶段」原文（用于 §2 表）

> "Step 1: Collect demonstration data, and train a supervised policy. / Step 2: Collect comparison data, and train a reward model. / Step 3: Optimize a policy against the reward model using PPO."
> — InstructGPT §3.1，<https://arxiv.org/html/2203.02155v1>

> "All PMs go through three phases of training: (1) language model (LM) pre-training on a large language corpus, (2) preference model pretraining (PMP), and (3) finetuning on human feedback."
> — **HH §A.2**（⚠️ 注意：这是**PM 自己的**三阶段，不是助手的三阶段；HH 的助手流程在 §4.1 只写成**两步** PM→RLHF）

---

## §5 与我们项目的类比边界 —— **KL 惩罚在我们没有人类数据时还有意义吗？**

### 5.0 先摆本项目的事实（已逐行核实）

| 项 | 事实 | 出处（本仓） |
|---|---|---|
| 奖励 | **22 键手工表**（不是学出来的 RM），默认值硬编码 | `src/clasher_new/rl/config.py:37-68`（`sed -n '37,68p' \| grep -c '":'` = **22**，2026-09-19 复算） |
| PPO | **手写** `PPOTrainer`，目标 = `min(ratio·A, clamp(ratio,1±0.2)·A)` + 熵 + 价值；**无任何 KL 项** | `src/clasher_new/rl/ppo.py:379-392`（`ratio`、`surr1/surr2`、`v_loss`）；`clip=0.2`、`ent_coef=0.01`、`n_epochs=1`（`:87`） |
| KL 安全阀 | **零命中**：`target_kl` / `approx_kl` / `kl_` 在 `rl/ppo.py`、`rl/train_solo.py`、`rl/run_league.py` 全无（唯一 `early_stop` 是「僵局早停」，与 KL 无关） | `docs/cmp_train_ours_2026-09-19.md:153`、`docs/cmp_hyperparams_2026-09-19.md:51` |
| 参考策略 | **没有 reference policy**。最接近的是行为克隆预热 `train_bc.py`（专家 = **规则**，非人类）与冻结副本/frozen 对手（是对手，不是参考策略） | `src/clasher_new/rl/train_bc.py`（131 行）；`docs/cmp_train_ours_2026-09-19.md:23` |
| BC 损失 | `loss = −logprob(expert_bundle)`，**无 entropy/KL 正则** | `train_bc.py:102-113`；`docs/cmp_train_ours_2026-09-19.md:33` |
| 人类偏好数据 | **没有**（对照 FirstLight_CR 的 25.2 万真人回放） | `docs/cmp_manual_2026-09-19.md:280` |
| 唯一的真实锚 | 我们**自己的引擎**（规则可判定的胜负/塔血/费差）；另有诊断用 KL 探针（`docs/tool_usage_audit_2026-09-14.md:50` 的 `KL(A‖·)`）**不是训练目标** | 本仓 |

### 5.1 结论：**KL 惩罚在「没有人类数据」时，其「主干理由」失效；剩下的部分要么已被 PPO clip 覆盖，要么与本项目的瓶颈反向。**

分三层说明。

**第一层：三条理由里，两条直接不成立。**

- 理由②「不让策略跑到 RM 训练分布之外」（Stiennon §3.4）与理由③「mitigate over-optimization of the reward model」（InstructGPT §3.5）**都以「奖励是学出来的代理」为前提**。我们没有 RM → 没有「RM 的训练分布」这个概念 → 这两条**无对象**。
- 而且这两条要防的是 **Goodhart 式代理错配**（Gao et al.: "the reward model is an **imperfect proxy**"）。我们的 22 键奖励是**对同一台引擎状态的确定性函数**（塔血差、费差、胜负、非法动作计数），**不是对某个真目标的抽样估计** ⇒ 不存在「代理 vs 金标」的分裂。⚠️ 本仓已有实证支持这一点：`docs/reward_composition_verdict_2026-09-14.md` 把逐帧奖励精确拆成五项，**没有主导项**（crown 30.95% / edw 29.84% / tower 21.47% / terminal 10.11% / unit 7.63%）⇒ 「奖励被某项带偏」不成立。
- ★ 但**必须留一个例外**：我们**确实**有另一种「代理」，只是它不在奖励里 —— **价值函数 V 是估计量**（本仓实测 `V` 重建误差：严格相等仅 **44.9%**、p90 **3.0**、max **7.13** 圣水，见 `docs/engagement_trade_online_2026-09-18.md`）。KL 惩罚**不修 V 的误差**；把 KL 当成「压住 V 误差」的手段是**范畴错误**。这一点在文献里也拿不到支持：Gao §3.6 表明 KL 惩罚**只改变优化速度、不改善真奖励前沿**。

**第二层：剩下的理由①（熵/防塌缩）在我们这里已被**其他机制**覆盖，且加 KL 会与本项目唯一的已知瓶颈冲突。**

- TRL 的实践形态（`−kl_coef·KL_t` 逐 token）**在数学上就是逐token熵正则的一个加权变体**：当 `π_θ ≈ π_ref` 时 `log(π_θ/π_ref) ≈ 0`，该项把策略**拉向**参考分布。我们的 `ent_coef=0.01`（`ppo.py:87`）已经在做「防塌缩」；再加一个朝 π_ref 的拉项，是**第二个反向拉力**，且**没有本项目的任何测量支持**。
- **更关键的是方向**：本项目已确认的**唯一**病理是**探索不足**——`docs/agents/ledger.md` **C14**：十万步里「放弃一张买得起的牌」发生 **0 次**（0/7,238 段），A 段最长 **22 帧 < 34 帧**（攒到 Xbow 所需）⇒ **0 样本 ⇒ 0 梯度**；同一册 **O7**：机制自锁（`save_ace`/`setup_wait` 在 Xbow 卡组 **0 帧触发**）。**KL 惩罚是一个「把策略拉回参考分布」的项** ⇒ 在我们这里它只会**进一步压制**那唯一被证明有效的方向。按目录纪律（【R11】不因「文献里有」就改奖励），**这构成「不加」的正面理由，而不只是「没理由加」**。
- 补充：`docs/exploration_pressure_gate_2026-09-18.md` 已实测「门控偏置」是**目前唯一同时抬覆盖率且不崩回报**的形态（`gate@6:d`：≥6 费帧 **6.765%** vs 基线 **0.000%**，回报 **+1.68** vs 基线 **+1.63**）⇒ 我们的探索问题有**已验证的杠杆**，不需要 KL 这个反向项。

**第三层：如果将来真要「锚定」，正确的先例不是 RLHF 的 KL，而是另一个技术家族。**

- 文献里「KL 朝参考策略」在 RLHF 语境下**始终**伴随一个**学出来的奖励**（InstructGPT β=0.02 / HH λ_KL=0.001 / TRL `init_kl_coef`=0.2 / DS-Chat `kl_ctl`=0.1，全部见 §3.3）——**没有一篇**把它当作「无 RM 场景下的安全装置」来论证。⚠️ 反过来，**HH 自己**说这个项 "might actually be wholly unnecessary"。
- 因此：**KL 惩罚在无人类数据时「可迁移的只有它的数学形式，不是它的理由」。** 若要用，必须改用**它所属的正确族**：把**规则专家**当 π_ref 做**正则化**（离线 RL 的 BC-正则 / KL-约束策略族，或 Stiennon 理由①的「熵/防塌缩」读法）。这在文献上是**有先例的**，但**先例来自 offline RL / KL-regularized RL，而不是 RLHF 的 KL 到 RLHF 之前的 SFT 模型**——**不要**把两者当成同一件事引用。

### 5.2 §5 一句话回答（给决策用）

> **KL penalty to reference policy 在我们这里「意义有限且当前不建议加」**：它的两条主干理由（防 RM 过优化、别跑出 RM 训练分布）**以「奖励是学出来的代理」为前提，我们不成立**；剩下的熵/防塌缩理由已被 `ent_coef` 覆盖；**且「无人类数据下 KL 仍有用」这一命题在本次核查的 RLHF 文献中查无依据（`[NOT-FOUND]`，见 U4/U5/U6/U16）**；同时 Gao et al. 的受控证据表明**显式 KL 惩罚不改善真奖励前沿，只等价早停**；本项目已确证的瓶颈是**探索不足**（C14/O7），KL 的作用方向与之**相反**。

---

## §6 可迁移点与警示

### 6.1 有先例的（可迁移，但要引对文献）

| 想法 | 先例状态 | 该引什么（**别引错**） |
|---|---|---|
| **「用规则专家打分」替代学出来的 RM** | ⚠️ **半有先例**。Stiennon §4.3 已经把「优化一个自动指标（ROUGE）」与「优化 RM」并列比较，并发现 ROUGE **更早 peak、更差**；Gao §4.2.2 也把「代理只看部分特征」当一类失败。但**没有**找到「用手写规则函数替代 RM 做 RLHF」的 RLHF 论文 | **不要**说「RLHF 文献支持规则专家当 RM」。可说的是：「自动指标当奖励是**已知会被优化的**，Stiennon §4.3 / Gao §4.2.2 有先例；我们与它们的差别是**我们的奖励不是真目标的代理，而是引擎状态的确定性函数**」——这是我们**自己的**论证，须自证，不能借文献 |
| **用 KL 锚定一个 BC/规则策略** | ✅ **有先例，但不在 RLHF 族里**。KL 正则化策略族 / 离线 RL 的 BC 正则（「在参考策略附近最大化奖励」）是有理论家底的；Stiennon §3.4 的理由①（"acts as an entropy bonus, encouraging the policy to explore and deterring it from collapsing to a single mode"）是**与 RM 无关**的那一半 | 引 **Stiennon §3.4 理由①**（这一句是可用的）；**不要**引 InstructGPT/HH 的 β/λ 作为「无 RM 也适用」的依据 |
| **把 KL 当「优化速度调节器」而非「防 hacking 装置」** | ✅ **有直接先例**（且是反直觉的那条） | 引 Gao et al. §3.6：「akin to **early stopping**」+「does not affect the ... frontier」。这条对我们最有用：它把 KL 从「安全机制」降级为「**速率旋钮**」 |
| **PPO 的 clip 本身 = 隐式信任域** | ✅ **有直接先例**（同一篇） | 引 Gao §3.6："PPO's surrogate objective incorporates an **implicit penalty on D_KL(π_old‖π)** ... This penalty is used to control how fast the policy changes" ⇒ **我们已经有这个**（`ppo.py:379-382` 的 `clamp(ratio,1±0.2)`），而且 **TRL v0.11.2 / DS-Chat 也都有 `cliprange=0.2`**（§3.3）⇒ 「我们没有 KL」不等于「我们没有信任域」 |

### 6.2 全新且**无先例**的（要标「本方案原创，风险自负」）

| 想法 | 状态 |
|---|---|
| **用 22 键可判定奖励 + 手工机制的**、**完全不含人类偏好数据**的对战环境 RL | 🔶 **在本报告核查范围内无先例**。RLHF 全族的 KL 项都绑定在「学出来的奖励」上；纯规则奖励 + 无参考策略的方案，本报告在 InstructGPT / HH / TRL / DS-Chat / Stiennon / Gao / LIMA / Secrets-of-RLHF 中**未找到**可对标的做法。⇒ **不能声称「文献支持」**，只能作为自证方案 |
| **「奖励项开了但没有可学样本」这一失效模式**（`economy_et` 100k 两臂实测：全帧圣水≥6 **0.52% vs 0.05%**，Xbow **5/8769 vs 1/8370**，与手写专家的 **4.69%** 差 ~2 个数量级） | 🔶 **文献里没有对应物**：RLHF 的失败模式是「奖励被利用」，我们的是「**奖励没有样本可作用**」。这是**反向**的病。⇒ 任何从 RLHF 借来的「防奖励被利用」手段（含 KL）**都不对准**我们的病根 |
| **用 `invalid_penalty`（每帧 0.05）当「按次惩罚」** | 🔶 无 RLHF 先例可对（RLHF 的动作空间无「非法动作」概念）。本仓实测教训：单局最长连续 **225 帧**被整包拒绝 ⇒ 累计 **≈ −11.25 > 胜利奖励 +10**（`docs/agents/ledger.md` O9）。⇒ 提醒：**按次惩罚的累计量级会盖过终局奖励**，这是本项目自己的量纲问题，无外部先例可抄 |

### 6.3 警示（本次调研中**最容易被误用**的三条）

1. **不要写「KL 惩罚防止 reward hacking」而不加限定。** InstructGPT 确实这么写了（"to mitigate over-optimization of the reward model"），**但**同一研究组的受控实验（Gao et al. §3.6）发现显式 KL 惩罚**让代理-金标缺口更大**、**不改善真奖励前沿**。两者并列才算诚实。
2. **不要跨论文搬 β。** β=0.02（InstructGPT）、λ_KL=0.001（HH）、`init_kl_coef`=0.2（TRL）、`kl_ctl`=0.1（DS-Chat）量纲与奖励尺度、token 数、优化器都不同（HH 自己都说 0.001 "might actually be wholly unnecessary"）。本仓【R15】已明文禁止跨实验照抄阈值。
3. **不要把「HH 有 KL」读成「KL 必需」。** HH 的 Eq.4.1 是**序列级**写法、全文未提 per-token，且作者自评可能完全没必要。**「HH 用了」≠「HH 证明它有用」**。

---

## §7 来源清单（URL + 取数时间）

**取数时间**：全部为 **2026-09-19**（CST，会话 `date -u` = 2026-09-18 19:33 UTC 起）。本仓文件核对时间同。

| # | 来源 | URL | 取数结果 |
|---|---|---|---|
| 1 | InstructGPT 摘要页 | <https://arxiv.org/abs/2203.02155> | ✅ 内容 |
| 2 | InstructGPT v1 全文 HTML（Eq.1/Eq.2/§3.5/§E.6/§E.7/§C.1–C.4） | <https://arxiv.org/html/2203.02155v1> | ✅ 内容（附录经 v1 HTML + **v1 TeX 源码 tarball** 逐字核对） |
| 3 | InstructGPT v1 TeX 源码 | <https://arxiv.org/e-print/2203.02155v1> | ✅ 内容（子智能体用于附录复核） |
| 4 | Anthropic HH 摘要页 | <https://arxiv.org/abs/2204.05862> | ✅ 内容 |
| 5 | Anthropic HH v1 全文（Eq.4.1/4.2、§4.1–4.3、§B.1、§B.4、Fig.4） | <https://arxiv.org/html/2204.05862v1> + <https://arxiv.org/pdf/2204.05862v1>（74 页） | ✅ 内容 |
| 6 | Anthropic 官方公告（元数据/日期 Apr 12, 2022） | <https://www.anthropic.com/news/training-a-helpful-and-harmless-assistant-with-reinforcement-learning-from-human-feedback> | ⚠️ 仅元数据，无正文 |
| 7 | Askell et al. 2021 §3.1 Eq.(3.1) PM 损失 | <https://arxiv.org/html/2112.00861v3> | ✅ 内容 |
| 8 | Stiennon et al. 2020（KL 两个 purposes 原文、RM 损失、§4.3 过优化） | <https://ar5iv.labs.arxiv.org/html/2009.01325> | ✅ 内容（705,275 B） |
| 9 | Gao/Schulman/Hilton 2022 过优化标度律 | <https://arxiv.org/abs/2210.10760> + <https://arxiv.org/html/2210.10760v1> + <https://ar5iv.labs.arxiv.org/html/2210.10760> | ✅ 内容 |
| 10 | LIMA | <https://arxiv.org/abs/2305.11206> + <https://arxiv.org/html/2305.11206v1> | ✅ 内容 |
| 11 | Secrets of RLHF in LLMs Part I: PPO（**= 2307.04964**） | <https://arxiv.org/abs/2307.04964> + <https://ar5iv.labs.arxiv.org/html/2307.04964> | ✅ 内容 |
| 12 | TRL v0.11.2 `ppo_trainer.py`（含 `_kl_penalty` 1150–1162、`compute_rewards` 1119–1148、KL 负值告警 1311–1317） | <https://github.com/huggingface/trl/blob/v0.11.2/trl/trainer/ppo_trainer.py> | ✅（Git blob SHA 逐字节核对 `42f916dd…`） |
| 13 | TRL v0.11.2 `ppo_config.py`（默认值 161–181） | <https://github.com/huggingface/trl/blob/v0.11.2/trl/trainer/ppo_config.py> | ✅ |
| 14 | TRL v0.11.2 `utils.py`（`AdaptiveKLController` 54–69、`FixedKLController` 72–79） | <https://github.com/huggingface/trl/blob/v0.11.2/trl/trainer/utils.py> | ✅（SHA `b6fe2cd4…`） |
| 15 | TRL 文档源 `docs/source/ppo_trainer.md` @ v0.15.2 | <https://github.com/huggingface/trl/blob/v0.15.2/docs/source/ppo_trainer.md> | ✅ |
| 16 | TRL `PPOTrainer` 移除 commit | `700b845c5d4b1cfd292cf9c067626d8a19fc0fe1`「Remove PPOTrainer (#7020)」，2026-09-04 | ✅（经 api.github.com） |
| 17 | TRL issue #235（KL 负值 = 失败前兆） | <https://github.com/huggingface/trl/issues/235> | ✅ **[二手]** |
| 18 | TRL issue #417（3/10 负 KL 爆炸） | <https://github.com/huggingface/trl/issues/417> | ✅ **[二手]** |
| 19 | TRL issue #1178（KL 压倒奖励） | <https://github.com/huggingface/trl/issues/1178> | ✅ **[二手]** |
| 20 | TRL PR #6429（+0.264 advantage 的 reward hacking） | <https://github.com/huggingface/trl/pull/6429> | ✅ **[二手]** |
| 21 | DeepSpeed-Chat README（三阶段自述） | <https://github.com/deepspeedai/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/README.md> | ✅ |
| 22 | DeepSpeed-Chat `dschat/rlhf/ppo_trainer.py`（`kl_ctl=0.1` 65–71、`compute_rewards` 181–194） | <https://github.com/deepspeedai/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/dschat/rlhf/ppo_trainer.py> | ✅ |
| 23 | DeepSpeed-Chat 重命名 commit | `e86d0c6d38d65158836c275e2d7b6a8f8e92f026`（"Refactor deepspeed-chat into a python package (#731)"，2023-11-06） | ✅（patch 显示 `"status":"renamed"`） |

**本仓（本项目）被引用的文件与行号**（全部 2026-09-19 核对）
`src/clasher_new/rl/config.py:37-68`（22 键）｜`src/clasher_new/rl/ppo.py:87,379-392`（手写 PPO，无 KL）｜`src/clasher_new/rl/train_bc.py`（131 行，BC 无 KL 正则）｜`docs/cmp_train_ours_2026-09-19.md:23,33,153,280,811`｜`docs/cmp_manual_2026-09-19.md:15-26,280,1049,1068,1115,1329-1330,1378`｜`docs/cmp_hyperparams_2026-09-19.md:51`｜`docs/reward_composition_verdict_2026-09-14.md`（五项拆解）｜`docs/agents/ledger.md` C14/O7/O8/O9｜`docs/exploration_pressure_gate_2026-09-18.md`（`gate@6:d` 6.765%/+1.68）｜`docs/engagement_trade_online_2026-09-18.md`（V 重建误差 p90 3.0）｜`docs/tool_usage_audit_2026-09-14.md:50`（KL 探针，非训练目标）

---

## §8 未验证 / 未能核实项

| # | 项 | 状态与原因 |
|---|---|---|
| U1 | **TRL 官方文档页** `huggingface.co/docs/trl/main/en/ppo_trainer` | ⚠️ **本环境无法访问**（host 不可达，`web_fetch` 与 `curl` 均失败，重试 3 次 rc=000）。且该文档文件已从 `main` 移除 ⇒ 用 **repo 文档源 v0.15.2** 替代（#15）。**任务书给的这个 URL 在本次环境未能验证可读** |
| U2 | **InstructGPT 的 SFT 损失公式** | `[NOT-FOUND]`。v1 全文（HTML + TeX 源码）无 SFT 损失方程，也无 "per-token cross entropy" 措辞 ⇒ §2.2 只能引 Stiennon et al. §1 的「maximize the log probability of demonstrations」并**显式标注这是转引** |
| U3 | **InstructGPT Eq.(2) 是否显式对 token 求和** | **公式里没有可见的 Σ**；"per-token ... at each token" 只见于散文（§3.5）⇒ 「公式未写、散文写了」这一区分已如实标注 |
| U4 | **Anthropic HH 的 KL 是 per-token 还是序列级** | `[NOT-FOUND]`。全文 grep `per-token` / `per token` / `at each token` / `sum over tokens` = **0 命中**。**不做推断**（不因为 "D_KL < 100 typically" 去反推，该文未把该数字绑定到任何口径） |
| U5 | **Anthropic HH 的 PM 损失方程** | `[NOT-FOUND]`（该文确实没写）。§3.1 自称与 Askell et al. 2021 "identical" ⇒ 引 **arXiv:2112.00861 §3.1 Eq.(3.1)** |
| U6 | **Anthropic HH 的 KL 系数消融** | `[NOT-FOUND]`。全文无 `ablation` 一词，无 β/λ 扫描报告；只有 §B.1「We performed a variety of hyperparameter scans…」。⚠️ **勿与** §5.1.2 的 `L_Total = L_Helpfulness + λ·L_Harmlessness`（λ∈{1,2,3,4,10}）混淆——那是**损失混合权重**，不是 KL |
| U7 | **Gao et al.「金标峰值在 ~20k 样本」** | **不存在**。全文 grep `20,000`/`20000`/`20k` = **0 命中**；峰值以 KL/`d` 报，非样本数。（任务书若含此数字，应视为**误传**） |
| U8 | **Gao et al. 的图 9（KL 扫描）具体 β 取值表** | 未逐值提取（只验证到「β 变化 → 金标只更早收敛」的结论句与「其余实验 KL=0」）。图的横轴具体数值**未验证** |
| U9 | **Stiennon et al. Fig.5 里 β 的具体取值序列** | 未提取（只验证到 §4.3 的结论文字与 KL 两个 purposes 的原文）。图 5 的 β 数值表**未验证** |
| U10 | **DeepSpeed-Chat 的「reward running average」** | `[NOT-FOUND]`：`ppo_trainer.py` 全文 grep `mean|std|moving|average|ema` **无命中**。仅见 Step-3 `main.py` import 了 `moving_average`/`ExponentialMovingAverage`（来自 `dschat.utils.utils`），但**取不到该文件**（GitHub API 限流 + raw 不可达）⇒ **不断言**它是否对奖励做滑动平均 |
| U11 | **DeepSpeed-Chat 的 per-stage 详细文档** | `applications/DeepSpeed-Chat/training/README.md` **未能取到**（限流）⇒ 其 Step-3 双损失的官方描述**未验证**（代码层的 `actor_loss_fn`/`critic_loss_fn` 已读到，但其中一行 `torch.clamp` 是**两次独立转写共识**而非逐字节验证，已在 §3.4 标注） |
| U12 | **TRL `main` 上 PPOTrainer 移除后的替代路径** | 移除前的实验版是 `trl/experimental/ppo`（`kl_coef=0.05`、`kl_estimator k1/k3`，**整文件未做 SHA 校验**，仅短片段解码）。移除后 `main` 用什么替代**未调查** |
| U13 | **TRL `trl/core.py` 是否有 `compute_kl`** | `[NOT-FOUND]`（未逐字节校验整个 `core.py` 以证明「不存在」；仅确认 `PPOTrainer` 的 import 列表里没有该名字，KL 映射实现在 `PPOTrainer._kl_penalty`） |
| U14 | **star / fork / 最近发布等仓库热度指标** | **本报告不提供**：`web_search` 工具本次会话**不可用**（HTTP 402 "Insufficient Balance"），且 `api.github.com` 匿名限流（60 req/h）已耗尽 ⇒ 无法可靠取数。**为遵守「禁止臆造数据」，此处留空** |
| U15 | **TRL 最新版本号** | 仅观察到 `api.github.com/repos/huggingface/trl/tags` 返回的最新 tag = **v1.13.0**（子智能体取数，2026-09-19）。该 tag 的**发布日期未验证** |
| U16 | **「弱 SFT → RM 上限」类命题** | `[NOT-FOUND]`：**未找到**任何论文写「弱 SFT 训出的 RM 无法超过该 SFT 的质量」。最接近的一手命题是 Secrets of RLHF Part I 的「RM 质量决定策略上限」与 LIMA 的「SFT 胜过 RLHF」。**不要把这句当引用** |
| U17 | **reward hacking 在「手工规则奖励」下的实例** | `[NOT-FOUND]`：本报告未找到「手工/规则奖励 + RL 产生可复现 exploit」的**受控论文证据**（CoastRunners 类案例本轮**未验证**：`blog.openai.com` DNS 失败、`openai.com/index/faulty-reward-functions/` 返回 403）。⇒ **不要**在 §4 里引用未验证的 CoastRunners |
| U18 | **「IL 数据质量决定上限」是否适用于本项目的 BC 预热** | **未验证**。LIMA 的证据是「1000 条**人类精选** > 5 万条未筛选」，而**我们**的 BC 专家是**规则**且自动产 50 局（`train_bc.py:121`）⇒ 两者的「数据质量」不是同一个量。本报告**不做**跨场景外推 |
