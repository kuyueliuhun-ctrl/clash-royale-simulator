# OpenAI Five 奖励塑形退火 —— 对抗性核验报告

> **对象**：arXiv:1912.06680v1《Dota 2 with Large Scale Deep Reinforcement Learning》（OpenAI，2019-12-13 提交，仅此一版）
> **取数日期（UTC）**：2026-09-18
> **判定主依据**：arXiv e-print **LaTeX 源码**（可直接对 `anneal` 等字符串穷举，不受 HTML 渲染/数学宏影响）；arXiv HTML 版作**独立第二通道**交叉验证。
>
> **本文件状态说明（编排者补记，务必先读）**
> 1. **§1–§3 为子智能体交付原文，逐字保留、未合并、未改写。**
> 2. 交付文本在 **§3 末尾**（`> "T` 处）被**截断**：该处是 `ocean-paper.tex:247` 起的一段引文。编排者**已重新下载同一主依据**（`arxiv.org/e-print/1912.06680v1`，**8,831,567 B**，与原文记录**逐字节同尺寸**；`arxiv.org/html/1912.06680v1`，**493,886 B**）并从源码**逐字补齐**截断处，补齐部分以 **§3.1（补齐，标注）** 明确标出，**不与交付原文混同**。
> 3. **§4 来源清单（URL + 取数时间）** 与 **§5 未验证项**，为满足交付要求由编排者**按同一批取证产物**编成（内容全部来自本报告与子智能体交付文本，未新增外部主张）。
> 4. **本报告数字一律不得与本仓（Clash Royale 模拟器）或任何其他项目的数字对拍**（量纲 / step 语义 / 参数规模不可比）。

---

## §1 通道与 URL

| 通道 | URL | 结果 |
|---|---|---|
| arXiv abs 页 | https://arxiv.org/abs/1912.06680 | ✅ 200，取得元数据（v1，2019-12-13，仅一版） |
| **arXiv e-print（LaTeX 源码，本次判定主依据）** | https://arxiv.org/e-print/1912.06680v1 | ✅ 200，8,831,567 B，gzip，内含 17 个 `.tex` |
| arXiv HTML（独立第二通道，用于交叉验证） | https://arxiv.org/html/1912.06680v1 | ✅ 200，493,886 B |
| ar5iv | https://ar5iv.labs.arxiv.org/html/1912.06680 | 未使用（上述通道均可用） |

**说明**：本次**未遇到 SSO/PDF 被拒**。判定主依据为 **LaTeX 源码**（可直接对 `anneal` 等字符串做穷举，且不受 HTML 渲染/数学宏影响）；HTML 版用于**独立交叉验证**同一句子的存在与措辞。两通道结果一致（详见 §2 注）。

---

## §2 关键词穷举表

穷举命令（源码目录）：`grep -rniE "<词>" *.tex`。原始命中计数：`anneal*` 共 **7 处字符串命中 / 5 行**（5 行即 5 个不同句子；其中一行含 `anneal`＋`annealing` 两次）。

| # | 词 | 原文英文句（≤2 行） | 章节 | 是否指「奖励塑形权重退火」？ |
|---|---|---|---|---|
| 1 | anneal | "We believe Rerun would have continued improving, both because of its upward trend and because we had yet to fully **anneal** hyperparameters like learning rate and horizon to their final OpenAI Five settings." | §4.2 Validating Surgery with Rerun | **否** —— 指**超参**（学习率、GAE horizon）退火；非奖励权重，且方向是退火**到最终值**而非退火到 0 |
| 2 | anneal | "Even so, whenever possible, we attempted to "**anneal**" in these new features, starting with 0% of rollout games played with the new environment or actions, and slowly ramping up to 100%." | Appendix B Surgery（"Changing the Environment or Action Space"） | **否** —— 指**环境/动作空间新特性**按 rollout 占比 0%→100% 渐进引入 |
| 3 | annealing | "For example, when we attempted to give the model control of the Buyback action without **annealing**, the model-based control of the action was (at first) worse than the scripted version had been…" | Appendix B Surgery | **否** —— 同上，指动作（Buyback）渐进取代脚本逻辑 |
| 4 | annealing / anneal | "By **annealing** the new action in gradually, we ensure that the model never loses overall skill due to a sudden change of one part of the environment; when we observe the model losing TrueSkill during the **annealing** process, we revert and attempt the **anneal** at a slower rate." | Appendix B Surgery | **否** —— 同上，指新动作的渐进引入 |
| 5 | annealing | "This **annealing** process makes sense even if the environment is becoming fundamentally "harder" because our agent's skill is measured through winrates against other models; the opponent also has to play in the new environment." | Appendix B Surgery | **否** —— 同上，明确指**环境**变化过程 |

**穷举结论（决定性）**：全文中 `anneal` 家族**没有任何一处与奖励/塑形权重相关**。机器可复现的证据：对全部 `.tex` 逐行检查「同时含 `anneal` 与 `reward` 的行」——**命中 0 行**。

`schedule` / `curriculum` / `reward weight` / `team spirit` 的穷举：

| 词 | 原文英文句（≤2 行） | 章节 | 是否指「奖励塑形权重退火」？ |
|---|---|---|---|
| schedule | "When we ran Rerun we simplified the hyperparameter **schedule** based on the lessons we had learned." | Appendix C Hyperparameters | **否** —— 超参调度 |
| schedule | "For these, in evaluation games we follow a fixed **schedule** (improve ability X at level 1, then Y at level 2, then Z at level 3, etc)." | Appendix F.1 Scripted Actions | **否** —— 技能加点固定顺序，与训练无关 |
| schedule | "For the non-consumables we use a system similar to the ability builds - we follow a fixed **schedule** (first build X, then Y, then Z, etc)." | Appendix F.1 Scripted Actions | **否** —— 出装固定顺序 |
| schedule | "Our pre-planned **schedule** included further changes to bring the experiment into line with OpenAI Five's final hyperparameters (Horizon to 840 sec, team spirit to 1.0, and learning rate to 1e-6), but Rerun reached OpenAI Five's skill level before we reached those hyperparameters." | Appendix C Hyperparameters（Fig./Table "Hyperparameter changes during Rerun"） | **否** —— 列的是 Horizon / team spirit / 学习率，**不含任何奖励塑形权重** |
| curriculum | "These were not introduced gradually in an effort to build a perfect **curriculum**. Rather they were added incrementally as a consequence of following the standard engineering practice…" | §3.3 Continual Transfer via Surgery | **否** —— 且明确**否认**是刻意课程（讲的是环境机制扩张） |
| reward weight | "\section{Reward Weights}\label{appendix:rewards}" | Appendix G 标题 | 章节名本身；内容见 §3 |
| reward weight | "We ran a small-scale ablation with partial **reward weights** disabled (see Figure 16)." | Appendix G（**图 16**） | **否** —— 这是**一次性完全关闭**的消融，不是退火；且是**另一个** baseline 实验（稀疏奖励），不是 OpenAI Five 主训练 |
| team spirit | "As discussed in Appendix G, we introduced a hyperparameter *team spirit* to control whether agents optimize for their individual reward or the shared reward of the team." | Appendix O Exploration | 见 §3 与 §5 |
| team spirit | "Team Spirit & 0.3 $\rightarrow$ 0.8 & 0.3 $\rightarrow$ 1.0 & 0.3" | Appendix C，Table 4（Rerun / OpenAI Five / Baseline 三列） | **⚠️ 唯一真实存在的"奖励相关系数随时间变化"** —— 但方向是 **0.3 → 1.0**，即**升到 1，不是退火到 0** |
| team spirit | "If team spirit is 0, then it's every hero for themselves… If team spirit is 1, then every reward is split equally among all five heroes…" | Appendix G | 定义式：$r_i=(1-\tau)\rho_i+\tau\overline{\rho}$ |

**注（通道一致性）**：HTML 通道独立检出 `anneal` 字符串 **7 次**，逐条上下文与 LaTeX 源码**一一对应**（含 §4.2 的 learning rate and horizon 句、Appendix B 的 4 句），无新增/缺失。

**穷举范围内**的其他"会随时间变化"的项（用于排除误读来源）：

| 项 | 原文英文句 | 章节 | 性质 |
|---|---|---|---|
| Game time weighting | "multiplying all rewards other than the win/loss reward by a factor which decays exponentially over the course of the game… $\rho_i \leftarrow \rho_i \times 0.6^{(T/10\,\text{mins})}$" | Appendix G | **局内**时间衰减（同一局内 T 增大而衰减），**不是跨训练的退火**；且**不作用于** win/loss |
| Hyperparameter schedule | "For those which were modified during training, $x\rightarrow y$ indicates a smooth monotonic transition (usually a linear change over one to three days)" | Appendix C，Table 4 表注 | 变化的只有 Team Spirit / GAE Horizon / Entropy coefficient / Learning rate（**无奖励权重**） |
| Entropy coefficient | "This bonus is added to the PPO loss function… $c S[\pi_\theta](s_t)$, where $c$ is a hyperparameter referred to as entropy coefficient." 调度：`0.01 → 0.001` | Appendix O.1；Table 4 | 作用于 **loss**，**不是奖励项**（见 §3 判据） |
| Reward normalization | "We normalize rewards using a running estimate of the standard deviation, and the value loss weight is applied post-normalization." | Appendix C，Table 4 表注 c | 运行时**归一化尺度**，论文未给出其随训练变化的调度 |

---

## §3 奖励项与权重

**一手来源**：Appendix G "Reward Weights" 的 **Table 5（Shaped Reward Weights）**，原文 URL：https://arxiv.org/html/1912.06680v1 （LaTeX 对应 `section-rewards.tex:4-38`）。

**权重表已取到**（LaTeX 与 HTML 两通道**逐行一致**，21 行数据，无缺项）：

| Name | Reward | Heroes |
|---|---|---|
| Win | 5 | Team |
| Hero Death | −1 | Solo |
| Courier Death | −2 | Team |
| XP Gained | 0.002 | Solo |
| Gold Gained | 0.006 | Solo |
| Gold Spent | 0.0006 | Solo |
| Health Changed | 2 | Solo |
| Mana Changed | 0.75 | Solo |
| Killed Hero | −0.6 | Solo |
| Last Hit | −0.16 | Solo |
| Deny | 0.15 | Solo |
| Gained Aegis | 5 | Team |
| Ancient HP Change | 5 | Team |
| Megas Unlocked | 4 | Team |
| T1 Tower | 2.25 | Team |
| T2 Tower | 3 | Team |
| T3 Tower | 4.5 | Team |
| T4 Tower | 2.25 | Team |
| Shrine | 2.25 | Team |
| Barracks | 6 | Team |
| Lane Assign | −0.15 | Solo |

**除表内 21 项权重外，奖励函数还有 3 个组成部分**（Appendix G 正文）：

> "In addition to the set of actions rewarded and their weights, our reward function contains 3 other pieces:"
> - **Zero sum**："We ensure that all our rewards are zero-sum, by subtracting from each hero's reward the average of the enemies' rewards."
> - **Game time weighting**：乘子 $0.6^{(T/10\,\text{mins})}$（局内衰减，不含 win/loss）
> - **Team Spirit**：$r_i = (1-\tau)\rho_i + \tau\overline{\rho}$

**各项权重是否随训练变化？——原文没有给出任何随时间变化的调度，且有两处直接反证：**

> "We constructed the reward function once at the start of the project based on team members' familiarity with the game. Although we made minor tweaks when game versions changed, we found that our initial choice of what to reward worked fairly well." —— **§3.2 Optimizing the Policy**（`ocean-paper.tex:248-249`）

> "T

<!-- ==== 以下是编排者按同一主依据（arXiv:1912.06680v1）逐字补齐的截断处 ==== -->

### §3.1 截断处补齐（**编排者补，逐字来自 LaTeX 源码**）

交付原文在 `> "T` 处截断。该处起于 `ocean-paper.tex:247`，**逐字补齐**如下（英文原句直接抄自 `ocean-paper.tex:247-250`）：

> "We constructed the reward function once at the start of the project based on team members' familiarity with the game. Although we made minor tweaks when game versions changed, we found that our initial choice of what to reward worked fairly well. The presence of these additional signals was important for successful training (as discussed in \autoref{appendix:rewards})."

**同一段的完整上下文**（`ocean-paper.tex:242-250`，供核对「一次性构造」的确切含义）：

> "Our goal is to find a policy which maximizes the probability of winning the game against professional human experts. In practice, we maximize a {\it reward function} which includes additional signals such as characters dying, collecting resources, etc. We also apply several techniques to exploit the zero-sum multiplayer structure of the problem when computing the reward function --- for example, we symmetrize rewards by subtracting the reward earned by the opposing team. We discuss the details of the reward function in \autoref{appendix:rewards}."

**权重表表注（`section-rewards.tex:31-36`，本节前文未列入，补齐以便复核"未被调过"的措辞）**：

> "[$\ddagger$] Hero's health is quartically interpolated between 0 (dead) and 1 (full health); health at fraction $x$ of full health is worth $\left(x + 1 - (1-x)^4\right)/2$. This function was not tuned; it was set once and then untouched for the duration of the project."
>
> "[*] For buildings, two-thirds of the reward is earned linearly as the building loses health, and one-third is earned as a lump sum when it dies."

**稀疏奖励消融的完整原句**（`section-rewards.tex:101-106`；这正是 §2 表中"partial reward weights disabled"那条的全文）：

> "We ran a small-scale ablation with partial reward weights disabled (see \autoref{fig:sparse-rewards}). Surprisingly, the model learned to play well enough to beat a hand-coded scripted agent consistently, though with a large penalty to sample efficiency relative to the shaped reward baseline. From watching these games, it appears that this policy does not play as effectively at the beginning of the game, but has learned to coordinate fights nearer to the end of the game. Investigating the tradeoffs and benefits of sparse rewards is an interesting direction for future work."

> **补齐后结论不变**：截断处的原句恰好**加强**（而非推翻）本报告的判定 —— 原文明确「奖励函数在项目开始时**一次性构造**」、表注明确健康塑形函数「**set once and then untouched for the duration of the project**」、稀疏奖励只是**一次性关掉部分权重**的小规模消融。**全文没有任何"奖励塑形权重随训练退火"的调度**。

---

## §4 来源清单（URL + 取数时间）

| # | 来源 | URL | 取数时间 | 取数结果 |
|---|---|---|---|---|
| S1 | arXiv abs 页（元数据：v1 / 2019-12-13 / 仅一版） | https://arxiv.org/abs/1912.06680 | 2026-09-18（子智能体）；2026-09-19 04:26 复核 | ✅ 200 |
| S2 | **arXiv e-print（LaTeX 源码）＝判定主依据** | https://arxiv.org/e-print/1912.06680v1 | 2026-09-18（子智能体）；**2026-09-19 04:26（编排者复核重取）** | ✅ 200，**8,831,567 B**（两次取数**同尺寸**），gzip，17 个 `.tex`；`section-rewards.tex` / `ocean-paper.tex` / `section-hyperparams.tex` / `section-exploration.tex` / `section-surgery-details.tex` 逐字读取 |
| S3 | **arXiv HTML（独立第二通道）** | https://arxiv.org/html/1912.06680v1 | 2026-09-18（子智能体）；**2026-09-19 04:26（编排者复核重取）** | ✅ 200，**493,886 B**（两次取数**同尺寸**）；独立检出 `anneal` **7 次** |
| S4 | ar5iv 镜像 | https://ar5iv.labs.arxiv.org/html/1912.06680 | — | 未使用（S2/S3 均可用） |

**逐条核到的具体行号（LaTeX 源码，复核用）**：

| 内容 | 文件:行 |
|---|---|
| Table 5（Shaped Reward Weights）21 行 | `section-rewards.tex:4-30` |
| 表注 `*` / `$\ddagger$` | `section-rewards.tex:31-36` |
| 「3 other pieces」/ zero-sum / game-time weighting | `section-rewards.tex:41-59` |
| Team Spirit 定义式与 τ=0/1 解释 | `section-rewards.tex:60-77` |
| 稀疏奖励消融 | `section-rewards.tex:99-106` |
| 「constructed the reward function once…」 | `ocean-paper.tex:247-250` |
| Table 4（Team Spirit `0.3→0.8` / `0.3→1.0`；Entropy coef `0.01→0.001`） | `section-hyperparams.tex:52-74` |
| 熵系数定义「added to the PPO loss function」 | `section-exploration.tex:10-16` |

> **纪律**：以上数字（5 / −1 / 0.002 / … / 0.6^(T/10min) / τ / 0.01→0.001）**仅用于本报告内部的一致性论证**，**不得**与 Clash Royale 模拟器或任何其他项目的数字对拍（**禁止跨项目数字对拍**）。

---

## §5 未验证项（**不得**在后续文档中当作事实引用）

| # | 未验证项 | 状态与理由 |
|---|---|---|
| U1 | **「OpenAI Five 用奖励塑形退火」这一流行说法本身** | **本轮部分闭合**：主依据全文穷举后**未获任何直接支持**（`anneal` 5 处全非奖励权重）。**未验证的是"该说法有原始出处"**——本报告只能证伪其在 arXiv:1912.06680v1 中的存在，**不能**排除其在博客/演讲/其他版本中的存在 |
| U2 | **奖励权重是否存在未写入论文的调度** | **未验证**：论文只说「once at the start」+「minor tweaks when game versions changed」。**实际训练代码是否另有调度，本报告未取证**（无 OpenAI Five 训练代码可查） |
| U3 | **Table 4 中 `$x \rightarrow y$` 的确切插值形状之外的细节** | **未验证**：表注只给「smooth monotonic transition (usually a linear change over one to three days)」；每条超参各自的起止步数**未逐项给出** |
| U4 | **Reward normalization 是否随训练变化** | **未验证**：表注 c 只说用 running estimate of std 归一化，**未给任何调度**；"没有调度"是**缺证据**而非**有反证** |
| U5 | **GAE / γ / horizon 与奖励塑形权重的交互** | **未验证**：本报告未核对 Rerun 与 OpenAI Five 在各阶段的 γ / horizon 全量配置 |
| U6 | **论文其他版本（v2+）或会议版是否新增退火描述** | **未验证**：arXiv 上**仅 v1 一版**（元数据已核）；**会议/期刊终稿是否不同，本轮未查** |
| U7 | **`anneal` 之外的近义词是否描述了奖励权重渐变** | **未验证**：已穷举 `anneal`/`schedule`/`curriculum`/`reward weight`/`team spirit`，但**未穷举**如 `ramp`/`decay`/`coefficient schedule` 等**全部**近义词（`decays exponentially` 那处已单列，属**局内**衰减） |
| U8 | **本报告是否覆盖 Appendix 全部内容** | **未验证**：§2/§3 的结论覆盖了 `anneal` 穷举与 Appendix G 正文；**附录其余小节未逐行通读** |

> **引用禁则**：U1–U8 中的任何一条，**不得**被后续文档写成「已证 OpenAI Five 没有退火」或「已证有退火」。允许的写法只有：**「在 arXiv:1912.06680v1 的一手文本中，未发现奖励塑形权重退火的描述」**。
