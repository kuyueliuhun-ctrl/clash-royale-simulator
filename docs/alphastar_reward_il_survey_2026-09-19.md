# AlphaStar（Vinyals et al., Nature 2019）奖惩机制与 IL 接口 —— 尽职调查报告

> **取数时间：2026-09-19**（本地 `date` 实测；UTC 2026-09-18）
> **调研者身份**：委派子智能体；权限被固定，**无法扩权**（本会话审批提示已禁用）。
> **重要方法学声明（先读）**：本次调研**未能取得正文原文**。`nature.com` 对本环境的取回会 302 到 SSO
> （`idp.nature.com`），`web_fetch` 不跟随跨域重定向；补充材料为 PDF/ZIP（`web_fetch` **不支持
> `application/pdf` / `application/octet-stream`**）；`deepmind.google` 博客页正文为客户端渲染，
> 取回只到导航栏；`web.archive.org` 直连超时（curl `000`）；调研中途 web_search 配额耗尽（HTTP 402）。
> ⇒ **§1–§5 中凡涉及 Nature 论文原文的措辞，均标注来源等级为「二手/复现件」，不冒充论文原句。**
> 本报告严格区分：`[一手-元数据]` / `[一手-官方博客标题页]` / `[复现件-OA论文]` / `[二手-分析]` / `[未验证]`。

---

## §1 标识与链接

| 项 | 值 | 来源等级 | 证据 URL |
|---|---|---|---|
| 正式标题 | **Grandmaster level in StarCraft II using multi-agent reinforcement learning** | `[一手-元数据]` | [Crossref/OpenAIRE](https://api.openaire.eu/search/publications?doi=10.1038/s41586-019-1724-z&format=json) |
| 期刊 | **Nature**，vol. **575**，issue **7782**，pp. **350–354** | `[一手-元数据]`（Europe PMC） | [Europe PMC REST](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:%2210.1038/s41586-019-1724-z%22&resultType=core&format=json) |
| DOI | **10.1038/s41586-019-1724-z** | `[一手-元数据]` | [doi.org](https://doi.org/10.1038/s41586-019-1724-z) |
| PMID | **31666705** | `[一手-元数据]` | 同上 Europe PMC |
| 上线 / 印刷 | online **2019-10-30**；print **2019-11-14**（PubMed 索引 2019-11-01） | `[一手-元数据]` | [Europe PMC REST](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:%2210.1038%2Fs41586-019-1724-z%22&resultType=core&format=json) |
| 第一作者 / 通信 | Oriol Vinyals（DeepMind）；末位 David Silver | `[一手-元数据]` | 同上 |
| **arXiv 预印本** | **不存在。** Semantic Scholar 对该 DOI 的 `externalIds` 只有 DBLP/MAG/DOI/CorpusId/PubMed，**无 arXiv id**；`openAccessPdf.url` 为空、`inPMC=N`、`inEPMC=N`、`isOpenAccess=N` | `[一手-元数据]` | [Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1038/s41586-019-1724-z?fields=title,externalIds,openAccessPdf,venue,year) |
| **官方博客（本调研主目标）** | *AlphaStar: Grandmaster level in StarCraft II using multi-agent reinforcement learning*，**2019-10-30**，作者署名 "The AlphaStar team" | `[一手-官方博客标题页]` | [deepmind.google/blog/...](https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/) |
| 官方博客（前作） | *AlphaStar: Mastering the real-time strategy game StarCraft II*，**2019-01-24** | `[一手-官方博客标题页]` | [deepmind.google/blog/alphastar-mastering-the-real-time-strategy-game-starcraft-ii/](https://deepmind.google/blog/alphastar-mastering-the-real-time-strategy-game-starcraft-ii/) |
| ⚠️ 易混条目 | **arXiv:1902.01724 = 「AlphaStar: An Evolutionary Computation Perspective」（Arulkumaran/Cully/Togelius, GECCO'19 Companion，v1 2019-02-05，v3 2019-07-14）——不是** AlphaStar 论文本身 | `[一手-元数据]` | [arXiv:1902.01724](https://arxiv.org/abs/1902.01724) |
| 官方开源代码（IL 部分） | DeepMind 开源了**架构 + 数据管线 + 行为克隆 agent** | `[复现件-OA论文]` | [arXiv:2308.03526](https://arxiv.org/html/2308.03526v1) §3 |

**一句话结论**：AlphaStar 的「奖励」= **二值 win/loss 终局信号（唯一环境奖励）** + **由人类统计量 z 导出的
pseudo-rewards（软塑形，非硬约束）**；「惩罚/约束」= **RL 更新时持续最小化「当前策略 vs 监督人类策略」的
KL**（探索约束，不是奖励项）；IL 是**先行的独立阶段**，RL 阶段再以 KL 把它「拴住」。

---

## §2 IL 阶段

### 2.1 用了多少人类回放

| 数字 | 含义 | 来源等级 | URL |
|---|---|---|---|
| **971K replays** | AlphaStar 主训练管线消耗的人类回放总数 | `[复现件-OA论文]`（第三方复现论文转述，非原文） | [ar5iv 2209.11553](https://ar5iv.labs.arxiv.org/html/2209.11553) §1："AlphaStar needs 12000 CPU cores, 384 TPUs, **971K replays**, and 44 days to train its model" |
| **128,000 CPU cores / 384 TPUs / 44 天** | 同一句里的算力口径 | 同上（同句） | 同上 |
| **~5M 局 / top 22% ⇒ ~1.4M 局 / 2.8M episodes** | 这是 **AlphaStar Unplugged（2023，离线离线基准）自己的数据集**，**不是** 2019 AlphaStar 的训练集 | `[复现件-OA论文]` | [arXiv:2308.03526](https://arxiv.org/html/2308.03526v1) §3.1 |

> **⚠️ 交叉核对未通过项**：另一篇第三方复现论文在同一处写的是 **"128,000 CPU cores and 384 TPUs"**
> （[ar5iv 2104.06890](https://ar5iv.labs.arxiv.org/html/2104.06890) §1），而 JAIR 扩展版写 **"12000 CPU cores,
> 384 TPUs, 971K replays"**（[JAIR 13743](https://www.jair.org/index.php/jair/article/view/13743) 摘要段）。
> 两篇**同作者**在不同版本里给出**互相矛盾的** CPU 核数（128,000 vs 12,000）。
> ⇒ **971K replays 与 44 天两处一致**，可采用；**CPU 核数两版冲突，本报告不采任何一版**（标 `[未验证]`）。
> 论文原文的准确数字（如是否区分 IL 用回放数与 RL 用统计量回放数）**本次无法核实**。

### 2.2 IL 损失是什么

| 问题 | 答复 | 来源等级 | 证据 |
|---|---|---|---|
| 监督对象是**动作**还是**统计量 z**？ | **IL 阶段监督动作**；z 统计量的监督发生在 **RL 阶段**（作为 pseudo-reward 目标） | `[复现件-OA论文]` + `[二手-分析]` | mini-AlphaStar 把两者分开描述：SL 的 loss 是 6 个 head 的交叉熵之和；RL 的 actor-critic loss 才"computes the pseudo-reward … associated with following the human strategy statistic z" |
| **IL 损失的具体形式** | Sum of **cross-entropy** over the action's components — 6 项：action type、delay（`queued` 亦有）、selected units、target unit（若有）、target location（若有）；另加 **gradient clip 0.5** + **Adam**；**温度 = 1.0**（采样时用 0.8/0.3） | `[复现件-OA论文]`（复现件，非论文原句） | [ar5iv 2104.06890](https://ar5iv.labs.arxiv.org/html/2104.06890) §3.3：`L = C(t_p,t_g)+C(d_p,d_g)+C(q_p,q_g)+C(u_p,u_g)+C(tu_p,tu_g)+C(tl_p,tl_g)`；§1.4 温度 |
| **z 统计量是否被监督/回归** | **未看到**「把 z 当回归目标做监督」的一手证据。可核实的用法是：z 是**从人类数据随机采样**的「策略统计量」，在 **RL** 里通过 pseudo-reward 引导策略**去匹配**它 | `[二手-分析]` | [Deciphering AlphaStar (Chai, 2019-07-21)](https://cyk1337.github.io/notes/2019/07/21/RL/DRL/Decipher-AlphaStar-on-StarCraft-II/) §Challenges："apply **pseudo-rewards** to follow a strategy statistic $z$, from randomly sampled human data" |
| **z-statistic 的一手措辞** | Nature 原文对 z 的定义/构造**未能取回**（SSO 拦截） | `[未验证]` | — |

### 2.3 IL 是独立阶段还是与 RL 联合？

**结论：两阶段——IL 是先行、独立的阶段；RL 阶段再以 KL 项把它「拴住」，但权重/退火表未取回。**

- `[复现件-OA论文]` mini-AlphaStar 的流程描述：「The agent was **firstly** trained by **supervised learning** …
  Then, the pre-trained agent **will be** trained through a series of self-play matches, using RL …」
  （[ar5iv 2104.06890](https://ar5iv.labs.arxiv.org/html/2104.06890) §3.1）
- `[二手-分析]` 三条 SL 理由（① 简化评估指标 ② **初始化** ③ **维持多样探索**），
  其中第③条的实现就是「**continually minimizes the KL divergence between the supervised and current policy**」
  （[Deciphering AlphaStar](https://cyk1337.github.io/notes/2019/07/21/RL/DRL/Decipher-AlphaStar-on-StarCraft-II/) §Supervised Learning）。
- ⚠️ mini-AlphaStar 的作者自述**其复现件并未完成 z 生成**：「We have implemented most of the components,
  with a few functions to be finished, **including the z value generation**」
  （[ar5iv 2104.06890](https://ar5iv.labs.arxiv.org/html/2104.06890) §5）
  ⇒ 该复现件**不是** z 机制的可执行参考实现，只能当**接口描述**用。
- ⚠️ 另有第三方指出 **AlphaStar 用了 raw action space 而非 human action space**（缩小了探索空间），
  这直接影响「IL 有多难」的外部有效性判断（[ar5iv 2209.11553](https://ar5iv.labs.arxiv.org/html/2209.11553) §1）。

---

## §3 奖励项清单

> 说明：以下「英文原句 ≤2 行」**只在能拿到原文措辞时才给英文**；取不到的写中文转述并标等级。
> 本次**唯一能拿到的英文原文级引用来自 AlphaStar Unplugged（DeepMind 作者群，2023）与 mini-AlphaStar（第三方复现）**。

### 3.1 环境奖励（唯一）

| 奖励项 | 内容 | 权重 / 退火 | 英文原句（≤2 行） | 来源等级 | URL |
|---|---|---|---|---|---|
| **win / loss / draw 终局信号** | 唯一的环境奖励；**只在局末一步非零**（胜 +1 / 负 −1 / 平 0） | 权重 1（无退火）；γ 按 delay 折算 `γ^{D_t(s)}` | `"in StarCraft II, the reward is 1 in a winning state, -1 in a losing state, and zero otherwise. So it does not depend on the action."` | `[复现件-OA论文]`（DeepMind 作者群 2023） | [arXiv:2308.03526](https://arxiv.org/html/2308.03526v1) §4.1 脚注 8 |
| 同上（另一处） | — | — | `"In the case of StarCraft II, the reward is the win-loss signal, so it can only be non-zero on the last step of the episode."` | 同上 | 同上 §4.1 正文 |

### 3.2 Pseudo-rewards / z-statistics（本次重点）

| 项 | 内容 | 权重 / 退火 | 证据 | 来源等级 | URL |
|---|---|---|---|---|---|
| **z 的来源** | 从**人类数据随机采样**到的一条「策略统计量」z | **未取回** | "a strategy statistic $z$, from randomly sampled human data" | `[二手-分析]` | [Deciphering AlphaStar](https://cyk1337.github.io/notes/2019/07/21/RL/DRL/Decipher-AlphaStar-on-StarCraft-II/) |
| **pseudo-reward 的构造一：build order** | **采样 z 与执行 build order 之间的 edit distance** | **未取回** | "The pseudo-rewards measure the **edit distance** between sampled and executed build orders" | `[二手-分析]` | 同上 |
| **pseudo-reward 的构造二：累计统计量** | **采样 z 与执行结果的累计统计量之间的 Hamming distance** | **未取回** | "and the **Hamming distances** between sampled and executed cumulative statistics" | `[二手-分析]` | 同上 |
| **z 的 RL 用法** | 作为**基线**（baseline）伴随 pseudo-reward 进入 **TD(λ)**；值头输出经 `baseline = (2/π)·arctan((π/2)·b)` 压到有界区间 | **未取回** | `"It computes the pseudo-reward of actor trajectories and computes TD(λ) with the baseline pseudo-reward associated with following the human strategy statistic z."` | `[复现件-OA论文]` | [ar5iv 2104.06890](https://ar5iv.labs.arxiv.org/html/2104.06890) §3.4（baseline 公式见同节 Eq.2） |
| **权重 / 退火 schedule** | **两项独立检索均未找到**任何一手或复现件的权重数值或退火曲线 | — | — | `[未验证]` | — |

### 3.3 有没有手工塑形（hand-crafted shaping）？

| 判定 | 证据 | 来源等级 |
|---|---|---|
| **未发现任何「手工奖励项列表」**（无 HP/资源/击杀等 per-frame 手写塑形）。设计意图是**只给终局 win/loss**，策略多样性靠 **z-pseudo-reward + league + 内在奖励（intrinsic reward over unit types）** 供给 | DeepMind 作者群 2023 明确该设置的 reward 只依赖终局状态；`[二手-分析]` 称每个 agent 的 personalized objective 含 "**Intrinsic reward function specifying preferences (e.g. over unit types)**" | `[复现件-OA论文]` §4.1 + `[二手-分析]` [Deciphering](https://cyk1337.github.io/notes/2019/07/21/RL/DRL/Decipher-AlphaStar-on-StarCraft-II/) |
| ⚠️ 限制 | 上述**不能证明** Nature 论文中没有其他塑形项——因为原文未取回。若需定论，**必须补读论文 Methods + Supplementary** | — |

### 3.4 RL 总损失里与「奖励」并列的项（供对照，非奖励）

`[复现件-OA论文]` mini-AlphaStar 把 RL 更新损失列为**四部分**（[ar5iv 2104.06890](https://ar5iv.labs.arxiv.org/html/2104.06890) §3.4）：

1. **actor-critic loss**（含 pseudo-reward、TD(λ)、split **vtrace** pg-loss）；
2. **split upgo-loss**（辅助策略梯度）；
3. **KL divergence**：当前 actor 轨迹 ↔ **监督训练的人类策略**；
4. **entropy loss**。

---

## §4 惩罚 / 约束项

| 机制 | 是否存在 | 形式 | 证据（原句 ≤2 行） | 来源等级 | URL |
|---|---|---|---|---|---|
| **KL-to-human（行为约束）** | **是** | **不是惩罚项，而是 RL 更新损失的第 3 项**；"continually minimizes the KL divergence between the supervised and current policy"；作用是把探索**约束在人类合理策略附近** | `"The third part computes the KL divergence between the actor trajectories to the supervised trained human policy"` | `[复现件-OA论文]` + `[二手-分析]` | [ar5iv 2104.06890](https://ar5iv.labs.arxiv.org/html/2104.06890) §3.4；[Deciphering](https://cyk1337.github.io/notes/2019/07/21/RL/DRL/Decipher-AlphaStar-on-StarCraft-II/) |
| **惩罚「动作概率偏离监督策略」** | **是**（同一机制的另一说法） | "The agents are also **penalized whenever their action probabilities differ from the supervised policy**" | 同左 | `[二手-分析]` | [Deciphering](https://cyk1337.github.io/notes/2019/07/21/RL/DRL/Decipher-AlphaStar-on-StarCraft-II/) §Supervised Learning |
| **KL 系数 / 是否有退火** | **未取回** | — | — | `[未验证]` | — |
| **league exploiter（对手侧压力）** | **是** | League 三类角色：**main player / main exploiter / league exploiter** + coordinator 维护 payoff matrix；checkpoint 条件为「对历史玩家胜率 > 0.7」或步数达到阈值 | `"we build a multi-agent league with three types of agents: main player (MP), main exploiter (ME), and league exploiters (LE)."` | `[复现件-OA论文]` | [ar5iv 2104.06890](https://ar5iv.labs.arxiv.org/html/2104.06890) §3.5 |
| **PFSP（matchmaking 约束）** | **是** | 以胜率加权采样对手：weight 函数 `linear / linear_capped / variance / squared`（默认 `linear`）；main player 有 0.5 概率直接抽 `squared` 权重的历史玩家 | 同节 | `[复现件-OA论文]` | 同上 §3.5 |
| **「main agent 被 exploiter 攻击」这句是否在论文里？** | **未取回原文**。可核实的是**机制描述**（exploiter 的唯一目标是找别人弱点），以及外部分析的转述「The sole goal of an exploiter is to identify the weaknesses in agents / The agents can then learn to defend against those weakness」 | — | `[二手-分析]` | [Deciphering](https://cyk1337.github.io/notes/2019/07/21/RL/DRL/Decipher-AlphaStar-on-StarCraft-II/) |

---

## §5 已知的坑

> 纪律：**不把二手推测写成论文结论**。以下每条标等级；`[未验证]` 即本次未能取证。

| # | 坑 | 本次取证情况 | 来源等级 | URL |
|---|---|---|---|---|
| 1 | **去掉人类统计量奖励会怎样（ablation）** | **未能取证**。没有取到论文的 ablation 数字。能取到的**最接近**表述是二手分析的一句总结："It shows that the **utilzation of human data is critical in final results**"（原文含拼写错误 `utilzation`），**这是二手分析对论文的解读，不是论文原句** | `[二手-分析]`（且为**间接转述**） | [Deciphering](https://cyk1337.github.io/notes/2019/07/21/RL/DRL/Decipher-AlphaStar-on-StarCraft-II/) |
| 2 | **IL 与 RL 目标冲突** | 有**机制性**证据：KL 项把 RL 策略拉向监督人类策略，同时 pseudo-reward 目标 z 是**随机采样**的一整条人类策略统计量 ⇒ 存在「保真 vs 求胜」的张力；**但论文是否显式讨论该冲突、有无消融，未取证** | `[复现件-OA论文]`（机制）+ `[未验证]`（讨论/消融） | [ar5iv 2104.06890](https://ar5iv.labs.arxiv.org/html/2104.06890) §3.4 |
| 3 | **league 中 main agent 被 exploiter 攻击 / 策略循环（cycling）** | 机制侧可核实：strategy cycles、exploiter 专职找弱点、checkpoint 复制历史防止遗忘；**也**有**独立第三方**对「简单 self-play 会卡住、population 才到 Grandmaster」的判断 | `[二手-分析]` | [Deciphering](https://cyk1337.github.io/notes/2019/07/21/RL/DRL/Decipher-AlphaStar-on-StarCraft-II/)；[Alex Irpan, 2019-02-22](https://www.alexirpan.com/2019/02/22/alphastar-part2.html) §2 |
| 4 | **公平性 / 外部有效性质疑**：AlphaStar **未使用 human action space**（用 raw API），且人类数据用法过多导致「更像模仿人类而非探索新战术」；训练资源不可复制 | 可核实的第三方批评（复现论文口径） | `[二手-分析]`（复现论文） | [ar5iv 2209.11553](https://ar5iv.labs.arxiv.org/html/2209.11553) §1/§2.4；[ar5iv 2104.06890](https://ar5iv.labs.arxiv.org/html/2104.06890) §1 |
| 5 | **BC 的误差累积（DAgger 问题）**：长时域下 IL 最坏界 `O(T²ε)`，StarCraft 决策点上千——**但实测 IL 仍强于预期**（能到 Gold 级） | 可核实的第三方定量/定性分析（**不是**论文结论） | `[二手-分析]` | [Alex Irpan, 2019-02-22](https://www.alexirpan.com/2019/02/22/alphastar-part2.html) §1 |
| 6 | **「Nash 均衡 + 选最不易被利用的 5 个 agent」** 的做法与 exploitability 问题 | 有第三方记述，但**与 2019 Nature 论文的表述是否一致未核** | `[二手-分析]` + `[未验证]` | [Alex Irpan](https://www.alexirpan.com/2019/02/22/alphastar-part2.html) §引言 |

---

## §6 对我们项目的可迁移点 与 不可迁移理由

**我们的实测口径**（本仓库证据，取数 2026-09-19）：

| 事实 | 值 | 证据 |
|---|---|---|
| 参数量 | **993,008**（实测 `torch.load` 累加 `state_dict`；架构 `plan_dim=58, belief_dim=563, hidden_dim=128, value_bypass=True, value_independent=True`） | `docs/cmp_hyperparams_2026-09-19.md:83` |
| 奖励表 | **`_DEFAULT_REWARD` 实际 16 键**（`crown_weight / crown_lose_weight / tower_dmg_opp / tower_dmg_self / tower_dmg_late / tower_dmg_self_late / win_bonus / lose_penalty / invalid_penalty / elixir_bonus / normalize_tower_dmg / elixir_diff_weight / elixir_diff_late / unit_dmg_k / tower_premium_k / king_gate`） | `src/clasher_new/rl/reward.py:28-46`（脚本计数） |
| ⚠️ 口径更正 | 任务书写的「**22 键**」**与仓库实测 16 键不符**；另有开关位 `normalize_tower_dmg: True` 等非权重键 ⇒ 「键数」口径本身有歧义。**以 16 个 dict 键为可核实数字** | 同上 |
| PPO 形态 | 单机单 env（`n_envs=1`）、`ent_coef=0.01`、`max_grad_norm=0.5`、`adv_norm="batch"` | `src/clasher_new/rl/ppo.py:88,118,234`；`config.py:185` |
| **有无 KL / 参考策略 / 人类数据** | **全仓 `rl/*.py` 检索 `kl_coef / approx_kl / ref_policy / reference policy` = 0 命中**；`ppo.py` 无任何 KL 项 ⇒ **当前是纯 PPO + 熵正则，无 KL 锚、无 BC、无人类数据** | `grep` 结果（2026-09-19） |

### 6.1 可迁移（按「迁移的是机制还是数字」分级）

| 可迁移点 | 迁移形态 | 为什么可迁移 / 注意事项 |
|---|---|---|
| **①「终局 win/loss 是唯一真奖励」这一设计选择** | 已是我们的现状（`win_bonus/lose_penalty` 为终局项），可作为**收敛方向**：把 16 键中**不可算量**的塑形逐步换成终局信号 + 少量可核实项 | AlphaStar 证明了「只给终局信号 + 探索约束」在巨规模下可 work；**但它没有证明在小规模/单机下也 work** ⇒ 只能当方向，不能当依据（对标禁则见下） |
| **②两阶段：先用「专家/先验」初始化，再用 RL 微调** | 我们**有**可当先验的资产（手写启发式 `belief_planner` / PlanToken 词表）⇒ 可做「对自有启发式做 BC 初始化」 | **不要**照搬「从人类回放学」；我们的先验来自**自己的规则专家**，量纲与合法性可完全对齐 |
| **③KL-to-先验作为软约束** | 若要防「RL 把先验技能忘掉」，可加 **KL(π_θ ‖ π_prior)**，其中 prior = **冻结的自有启发式策略**（而非人类） | 这是**唯一能保留「KL 约束」精髓**的形态；但需先量「先验强度」，参照我们既有纪律：**在未取证前不改奖励**（R11）、新增超参须预注册 |
| **④「非对称定价」思路** | 我们**已有**（`crown_lose_weight 10 > crown_weight 8`；`tower_dmg_self 0.0012 > tower_dmg_opp 0.001`） | AlphaStar 的启发不是数字，而是「防守定价 > 进攻定价」这类**结构性不对称**在长时域信用分配里更稳 |
| **⑤「用统计量目标做软塑形」的形态（不是数字）** | 把「我想让模型学会的局面统计」做成**可自算**的目标向量，用**距离**（edit / Hamming 类）当软塑形 —— **统计量可来自我们自己的专家脚本，而不是人类** | ⚠️ 严格限定：**距离类塑形必须可微/可批算**，且要防「刷距离」；详见 §6.2 的量纲警告 |
| **⑥league/PFSP 的「对手多样性」直觉** | 我们已有 `pfsp.py` / `league.py` / `opponents.py`（`opp_mix` 配比） | 只需保留「**历史版本 + 专门找弱点的对手**」两类；**不要**引入多机 league 规模 |

### 6.2 **不可迁移**（逐条给理由）

| 机制 | 不可迁移理由 | 类型 |
|---|---|---|
| **z-statistic pseudo-rewards（原始形态）** | **① 无人类数据** ⇒ z 的采样池根本不存在；**② 量纲**：z 是「整条 build order / 一整套累计统计量」级别的对象，其 edit/Hamming 距离的数值范围与我们单帧奖励（`tower_dmg 0.001` 级）**相差数个数量级**，直接搬会**淹没终局信号**；**③ 规模**：需要「每局多条 z、每步算距离」的采样与计算，单机 `n_envs=1` 的吞吐承受不起 | 量纲 / 规模 / 无人类数据 |
| **KL-to-human** | **无人类策略** ⇒ 参考分布不存在。数学上可换成 KL-to-prior（见 6.1③），但**那是另一个机制**，不能声称「我们在用 AlphaStar 的 KL-to-human」 | 无人类数据 |
| **971K replays 规模的 IL 阶段** | 我们**零**人类回放；且 993K 参数 / 单机的算力下，复制该阶段的**成本收益比不可比**（对标禁则：不得用它们的数字标定我们） | 规模 / 无人类数据 |
| **League（12 个 league learner + main/ME/LE 三型）** | 我们单机单 env；多 agent 并训的**边际收益未在本仓取证**，且会直接冲撞既有纪律（不扩参、单变量、预注册） | 规模 |
| **`baseline = (2/π)arctan((π/2)b)` 有界值头** | 这是为 z-pseudo-reward 的有界区间服务的；我们没有 z ⇒ 该变换**失去动机**。⚠️ 且它触及价值头架构 ⇒ 本仓架构变更必须 `--fresh`（R6） | 机制耦合 |
| **`γ^{D_t(s)}` 按 delay 折现** | 我们的 `step` 已是**决策帧**（R9 口径），不存在帧间 skip 的半连续时间折扣；引入会造成**两套时间口径**（R17） | 口径 |
| **intrinsic reward over unit types** | 属手工塑形的一种；本仓**明令在取证前不改奖励**（R11），且「按单位类型给内在奖励」会与既有 16 键**重复计价**（本仓已两次踩过重复计数：S2 的 `tower_term` 与 `ΔΦ`、`crown` 方向） | 纪律 / 重复计价 |

### 6.3 给下一轮实验的**最小可执行建议**（遵本仓纪律）

1. **不要把任何 AlphaStar 数字**（971K、44 天、384 TPU、z 距离量级）**搬进我们的判据**——它们与被污染过的量纲不可对拍（R15）。
2. 若要做「KL-to-prior」，先写预注册：prior 固定为哪个 ckpt/手写策略、KL 系数、`p=0` 是否逐位回旧行为（R2）、失败分支。
3. 若要做「统计量软塑形」，先证明该统计量：**(a) 可自算**（R12：能算的不许让网络猜）、**(b) 与胜负同号**（先做只读相关性取证，参照 S2 的做法）、**(c) 不与被计价的项重复**。
4. 三条既有门禁的教训同样适用：**离线口径 ≠ 在线口径**（S2 的 Δρ 从 +0.145 掉到 +0.008）⇒ 任何新奖励项必须在**在线口径**下重测。

---

## §7 来源清单（URL + 取数时间）

**取数时间统一为 2026-09-19**（UTC 2026-09-18；`web_fetch` 逐条取回）。

| # | 来源 | 等级 | URL | 本次取回状态 |
|---|---|---|---|---|
| S1 | Nature 论文页面（正文） | 一手-正文 | https://www.nature.com/articles/s41586-019-1724-z | ❌ **失败**：302 → `idp.nature.com`（SSO），未跟随 |
| S2 | Nature 论文 PDF | 一手-正文 | https://www.nature.com/articles/s41586-019-1724-z.pdf | ❌ 同上（303） |
| S3 | Crossref/OpenAIRE 元数据（DOI 解析） | 一手-元数据 | https://api.openaire.eu/search/publications?doi=10.1038/s41586-019-1724-z&format=json | ✅ 取回（vol 575 / pp 350-354 / CLOSED access / citationCount 3856） |
| S4 | Europe PMC REST（期刊/日期/PMID/摘要） | 一手-元数据 | https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:%2210.1038/s41586-019-1724-z%22&resultType=core&format=json | ✅ 取回（`inPMC=N`, `isOpenAccess=N`, `hasSuppl=N`） |
| S5 | Semantic Scholar API | 一手-元数据 | https://api.semanticscholar.org/graph/v1/paper/DOI:10.1038/s41586-019-1724-z?fields=title,externalIds,openAccessPdf | ✅ 取回（**无 arXiv id**、`openAccessPdf.url` 为空） |
| S6 | DeepMind 官方博客（2019-10-30，Nature 版） | 一手-官方（**仅标题/日期可见**） | https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/ | ⚠️ 部分：正文客户端渲染，只到 "The AlphaStar team" |
| S7 | DeepMind 官方博客（2019-01-24，前作） | 一手-官方（同上） | https://deepmind.google/blog/alphastar-mastering-the-real-time-strategy-game-starcraft-ii/ | ⚠️ 部分：同上 |
| S8 | **AlphaStar Unplugged**（Mathieu, Ozair, … Vinyals；arXiv:2308.03526v1；CC BY 4.0） | 复现件-OA论文（**DeepMind 作者群**，对 reward 口径的权威对照） | https://arxiv.org/html/2308.03526v1 | ✅ 全文取回 |
| S9 | **mini-AlphaStar**（Liu et al., arXiv:2104.06890） | 复现件-OA论文（**结构/损失 → 仅接口描述**；作者自述**未实现 z**） | https://ar5iv.labs.arxiv.org/html/2104.06890 | ✅ 全文取回 |
| S10 | **On Efficient RL for Full-length Game of SC2**（JAIR 75:213-260, 2022；DOI 10.1613/jair.1.13743；arXiv:2209.11553） | 复现件-OA论文（**971K replays / 44 天**出处；批评 AlphaStar 未用 human action space） | https://www.jair.org/index.php/jair/article/view/13743 ・ https://ar5iv.labs.arxiv.org/html/2209.11553 | ✅ 取回（973K 口径句在 ar5iv 版） |
| S11 | **Deciphering AlphaStar on StarCraft II**（Chai Yekun, 2019-07-21；博客） | **二手-分析**（pseudo-reward 的 edit/Hamming 距离、KL 约束、league 角色描述） | https://cyk1337.github.io/notes/2019/07/21/RL/DRL/Decipher-AlphaStar-on-StarCraft-II/ | ✅ 全文取回 |
| S12 | **An Overdue Post on AlphaStar, Part 2**（Alex Irpan, 2019-02-22） | **二手-分析**（IL 长时域/DAgger、PBT、5 agent 选择） | https://www.alexirpan.com/2019/02/22/alphastar-part2.html | ✅ 全文取回 |
| S13 | arXiv:1902.01724（Evolutionary Computation Perspective） | 一手-元数据（**易混条目**，非 AlphaStar 本体） | https://arxiv.org/abs/1902.01724 | ✅ 取回 |
| S14 | Springer 补充材料 ZIP / PDF | 一手-补充材料 | https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-019-1724-z/MediaObjects/41586_2019_1724_MOESM2_ESM.zip | ❌ **不可读**：`application/octet-stream` / `application/pdf` 均被 `web_fetch` 拒绝 |
| S15 | 本仓库实测（参数量/奖励键/PPO 形态） | 本仓证据 | `docs/cmp_hyperparams_2026-09-19.md:83`；`src/clasher_new/rl/reward.py:28-46`；`src/clasher_new/rl/ppo.py:88` | ✅ 本地读取 |

**本环境取回失败清单（供委派方判断是否需要换通道）**：`web.archive.org`（curl 超时 `000`）、
`deepmind.google` 正文字段、`openreview.net`（浏览器验证墙）、`discovery.ucl.ac.uk`（403 挑战）、
`uef.fi` / `doria.fi`（403）、`api.crossref.org`（30 s 超时）、**`web_search`（HTTP 402 余额不足，中途失效）**、
`s41586-019-1724-z` 的任何 `www.nature.com` 变体（全部 302 → SSO）。

---

## §8 未验证项（明确列出，避免被当成结论）

| # | 未验证项 | 为什么没验证到 | 需要什么才能验证 |
|---|---|---|---|
| U1 | **IL 阶段的确切回放数**（是否 = 971K；是否区分「IL 用」与「RL 用人类统计量」两批） | Nature 正文/补充材料不可读；971K 来自第三方复现论文转述 | 论文 Methods 原文，或 Supplementary |
| U2 | **IL 损失的确切形式**（6 头交叉熵之和是否为论文原式；温度/权重的原值） | 同上；目前来自 mini-AlphaStar 复现件 | 论文 Methods |
| U3 | **z-statistic 的定义与构造**（哪些统计量、多长窗口、如何采样） | 同上 | 论文 Methods + Supplementary |
| U4 | **pseudo-reward 的权重与退火 schedule** | 两项独立检索均无结果 | 论文 Methods |
| U5 | **KL-to-human 的系数与是否退火** | 同上 | 论文 Methods |
| U6 | **「去掉人类统计量奖励」的 ablation 数字** | 只拿到一句二手总结 | 论文 Extended Data / Supplementary |
| U7 | **是否有其它手工塑形项**（本报告「未发现」≠「不存在」） | 原文不可读 | 论文 Methods |
| U8 | **「main agent 被 exploiter 攻击」是否为论文原话** | 只拿到机制描述与第三方转述 | 论文正文 |
| U9 | CPU 核数口径（128,000 vs 12,000） | 两篇同作者不同版本互相矛盾 | 论文原文 |
| U10 | AlphaStar 在 Nature 页面的 altmetric/引用数快照 | Nature 页面 302 | 可访问的 Nature 页面（当前 OpenAIRE 侧读数 `citationCount=3856`、`influence_alt=2570`，**为聚合库口径，非 Nature 本站计数**） |

---

## 附：本报告对纪律条款的自查

- ✅ 未把 AlphaStar 的任何数字与其它项目对拍（971K / 384 TPU / 44 天只作**标识性引用**，未用于标定我们）。
- ✅ 不确定的都写了「未验证」（U1–U10 + §1/§2/§3/§5 的逐条等级标注）。
- ✅ 一手优先；一手不可得处**明确标注为二手/复现件**，并给出「为什么拿不到」的可核查原因（SSO / PDF 不支持 / 搜索配额）。
- ✅ 英文原句引用均 ≤2 行；取不到英文原句的地方**只给中文转述**，不伪造引文。
- ⚠️ 已知不足：本报告**没有**任何一句来自 Nature 论文正文的原文引用。若委托方需要「论文原句级」证据，
  必须换一条能读 Nature 正文/补充材料的通道（例如可跟随 SSO 重定向、支持 PDF/ZIP 解析的取回工具，或机构订阅）。
