# 腾讯绝悟 / JueWu 奖惩机制与 IL 接口 · 尽职调查报告

**取数时间：2026-09-19 03:30–03:45 CST（UTC+8）｜观测者：调研执行者**
**方法：以 arXiv 原文（PDF 全文抽取 + arXiv HTML）、腾讯官方一手页面、NeurIPS 官方论文集为唯一事实来源；二手解读一律标注并降级。**

---

## §1 一句话结论

绝悟系列的奖惩机制在**公开一手来源中可核实的核心**是：**同一套「按游戏功能分组的密集+稀疏奖励表」+ 动作合法性掩码（action mask / legal_action）接口 + 分阶段课程自对弈**；**奖励是人工设计并手工调参的（腾讯官方口径：「人工智能的训练，本质上就是奖励设计的过程」）**，而**「奖励权重退火（annealing）」在已核实的一手文献中没有任何直接证据**——不能断言其使用，也不能断言其未使用。

---

## §2 标识与链接

### 2.1 论文（arXiv API 复算，2026-09-19）

| # | 标题 | arXiv | 首次提交（UTC） | 版本 | 发表venue |
|---|---|---|---|---|---|
| P1 | **Mastering Complex Control in MOBA Games with Deep Reinforcement Learning**（**1v1 / Tencent Solo**） | [1912.09729](https://arxiv.org/abs/1912.09729) | **2019-12-20 09:56** | v1 2019-12-20 / v2 2020-01-03 / v3 2020-12-15 | AAAI-2020（[OJS](https://ojs.aaai.org/index.php/AAAI/article/view/6144)） |
| P2 | **Supervised Learning Achieves Human-Level Performance in MOBA Games: A Case Study of Honor of Kings**（**JueWu-SL**） | [2011.12582](https://arxiv.org/abs/2011.12582) | **2020-11-25 08:45** | v1 | IEEE TNNLS，DOI 10.1109/TNNLS.2020.3029475 |
| P3 | **Towards Playing Full MOBA Games with Deep Reinforcement Learning**（**JueWu-RL 5v5**） | [2011.12692](https://arxiv.org/abs/2011.12692) | **2020-11-25 12:52** | v1 / v4 2020-12-31 | **NeurIPS 2020**（[论文集 PDF](https://proceedings.neurips.cc/paper/2020/file/06d5ae105ea1bea4d800bc96491876e9-Paper.pdf)） |
| P4 | **Which Heroes to Pick? Learning to Draft in MOBA Games with Neural Networks and Tree Search**（**JueWuDraft**） | [2012.10171](https://arxiv.org/abs/2012.10171) | 2020-12-18 11:19 | v4 2021-08-05 | IEEE Transactions on Games |
| P5 | **Learning Diverse Policies in MOBA Games via Macro-Goals**（MGG） | [2110.14221](https://arxiv.org/abs/2110.14221) | 2021-10-27 07:15 | v1 | **NeurIPS 2021** |
| P6 | **Honor of Kings Arena: an Environment for Generalization in Competitive Reinforcement Learning**（**开悟 / hok_env 官方环境论文**） | [2209.08483](https://arxiv.org/abs/2209.08483) | 2022-09-18 06:29 | v3 2022-10-18 | **NeurIPS 2022**，35:11881-11892 |
| P7 | Hierarchical Macro Strategy Model for MOBA Game AI（HMS，前身） | [1812.07887](https://arxiv.org/abs/1812.07887) | —（本次仅取到 PDF 正文，未复核提交时间 ⇒ **未验证**） | — | AAAI-2019 |

> 上表 arXiv id / 提交时间由 <https://export.arxiv.org/api/query?id_list=2011.12582,2011.12692,2012.10171> 于 2026-09-19 03:31 直接返回并逐字核对。

### 2.2 官方项目页 / 博客

| 对象 | URL | 内容要点 |
|---|---|---|
| 腾讯开悟（AI 竞技平台）「关于我们」 | <https://tencentarena.com/aiarena/zh/about> | 官方自我定位、七大板块（强化训练/数据抽取/监督训练/推理服务/对战管理/录像管理/入门指引） |
| 腾讯开悟「开放服务」 | <https://tencentarena.com/aiarena/zh/open-gamecore> | 指向 **官方开源环境 <https://github.com/tencent-ailab/hok_env>** 与论文 P6；「仅支持 Windows 系统」；需签承诺函申请许可 |
| 腾讯开悟「平台动态」 | <https://tencentarena.com/aiarena/zh/trends> | 有据可查的动态：`2020-05-07 AI+游戏：通用人工智能的试金石`、`2022-11-20 腾讯开悟「王者荣耀AI开放研究环境」启动申请`、`2024-05-20 / 2025-05-11 全球公开赛启动` |
| 中文深度报道（非腾讯 AI Lab 本人执笔，但含**腾讯王者 AI 团队受访原话**） | <https://cloud.tencent.com.cn/developer/article/2160236>（浅黑科技《王者荣耀的B面…》，腾讯云社区，原始发表 2022-08-19） | 「所有的『奖励机制』都需要人类从头设计」；**「如果罚得太多，英雄就会不敢出战，各种逃窜，甚至掉一丝血就想回城补血；如果罚得太少，英雄又会傻冲，容易被团灭」**；对杀敌奖励为负的解释 |
| 事件性中文报道（转载自雷锋网，非腾讯一手） | <https://www.openi.org.cn/腾讯-ai-「绝悟」制胜王者荣耀，升级至电竞职业水平/> （2019-08-07） | 2019-08-02 世界冠军杯半决赛特设环节 5v5 战胜职业选手赛区联队；同日 ChinaJoy 1v1 首次公开体验，首日 504 场胜率 99.8%；研发始于 2017-12 |
| 英文旁证（同 P1 数据） | <https://www.zhuanzhi.ai/document/3e8f55cf22065d92bbf6d2fc6c75d034>（标题：「2100场王者荣耀，1v1胜率99.8%，腾讯绝悟 AI 技术解读 \| AAAI 2020」） | **该域名 DNS 解析失败（getaddrinfo ENOTFOUND），正文未取到 ⇒ 仅记录标题，内容未验证** |

---

## §3 IL 阶段（模仿学习）

### 3.1 数据从哪来、多少量（**P2 原文，唯一权威口径**）

| 项 | 原文事实 | 出处 |
|---|---|---|
| 玩家来源 | **"the top 1% human players"**（前 1% 人类玩家），**非职业选手** | P2 §V-A-1） |
| 局数 | **"100 million samples from about 0.12 million games are extracted for one hero"**（**每个英雄**：约 **1 亿条样本**、约 **12 万局**） | P2 §V-A-1） |
| 划份 | 每英雄数据集随机切分：训练约 **9000 万**样本、测试约 **1000 万** | P2 §V-A-1） |
| 筛选标准 | 过滤「表现分」差的局：**"empirically set as the overall score that exceeds 90% of players using the same hero"**（例：貂蝉 ≥ 10.0） | P2 §V-A-1） |
| 采样率 | 预处理后 **"only one-twentieth of the frames will be kept on average"**（平均只保留 **1/20** 帧） | P2 §V-A-1） |
| 训练成本 | **每英雄 16 张 Nvidia P40、约 36 小时**；Adam lr=1e-4；batch=256/卡 | P2 §V-A-2） |

> ⚠️ 数字口径提醒（**禁止跨项目对拍**）：「1 亿样本 / 12 万局」是**单英雄**的，而 SL 是**一英雄一模型**（原文：`We train one model for one AI hero`）。它**不是**整个 5v5 系统的数据量。

**职业选手数据的真实用途（易被混淆）**：
- P2 的 IL **不用**职业选手数据做训练；职业选手是**评测对手**（KPL 半职业队 5 场 test、AI 先赢 5 场，P2 §IV）。
- **用职业选手录像做训练**的是后来的 **MGG（P5）**：`Stage 1: Extract states D_t and its corresponding macro-goal g^D from top e-sports demonstrations.`
- 「请游戏策划把 KPL 录像里的优秀操作人工标注出来」的说法**只出现在中文二手报道**（[cloud.tencent.com.cn/developer/article/2160236](https://cloud.tencent.com.cn/developer/article/2160236)），**在 P2 原文中未找到对应表述 ⇒ 未验证**。

### 3.2 IL 的目标是什么（**P2 原文**）

不是「监督单个动作」，而是**分层多任务多类分类**：

1. **动作标签（micromanagement）是两级的**：`level-1`（做什么：move / normal attack / skill1-3 / summons / return base …）+ `level-2`（怎么做：移动方向、攻击目标、技能的方向/目标/位置——取决于技能类型）。
2. **意图标签（macro-strategy）是多视图的**：
   - **global intent**：把地图划成 **24×24 = 576 类**，以「持续攻击发生的区域」为 ground truth 区域；
   - **local intent**：以自身为中心划 **12×12**，从两次攻击事件之间抽取玩家的中间位置（王者的平均 **6.5 s** 计算一次中间位置）。
3. **损失函数 = 四项交叉熵的加权和 + L2 正则**：
   `ℓ(p,y) = w_a0·ℓ_CE(p0,y0_a) + w_a1·ℓ_CE(p_{y0_a}, y1_a) + w_bg·ℓ_CE(p_{m+1}, y^g_b) + w_bl·ℓ_CE(p_{m+2}, y^l_b)`，目标 `min_θ Σℓ + λ‖θ‖²`。
   原文：`the weights of the four losses, i.e., w_a0, w_a1, w_bg, and w_bl, are of the same scale ... thereby tuned to 1 empirically`，`λ = 1`。
   ⇒ **四个权重都设为 1，λ=1**；**没有 KL 项、没有熵项**。
4. 输出在**离散化的类别**上做交叉熵（`ℓ_CE denotes the cross entropy loss, as we discretize intent and action labels`）⇒ 是**预测人类动作/意图的类别分布**，不是回归、不是 GAIL 式对抗。

### 3.3 IL 与 RL 是两段式还是联合？

**就 P2 本身：SL 是独立的、端到端训到「High King 水平」的完整方案，不以 RL 为目标。**

P2 结论节原文（关键）：
> `As an ongoing step, we are combining SL and RL to further improve our AI's ability.`
> `our network structure can be used as the policy network in RL. Also, our model can be easily adapted to provide suitable initializations in an RL setting and can also serve as the opponent for guiding RL's policy training.`

⇒ 官方表述是**「正在结合」**——即**两段式的接口被明确指出（SL 权重作 RL 初始化 / SL 作对手）**，但 P2 论文**没有**给出「SL→RL 联合训练」的完整实验。
**⚠️ 未验证项**：「绝悟 5v5 的最终版本实际就是用 SL 初始化 RL 的」——**P3（5v5 RL）通篇未提及以 JueWu-SL 初始化**，P3 的 Phase 1/2/3 全部是自我对弈 + 多教师蒸馏。

**P3/P5 的「模仿」是另一条线**：Phase 2 的 **distillation（蒸馏）是监督式的**，但教师是 **RL 自对弈训出来的固定阵容模型**，不是人类：

| 阶段 | 做什么 | 数据/信号来源 |
|---|---|---|
| P3 Phase 1 | **固定阵容自我对弈**（40 英雄分 4 组，每组 10 个；组内 5v5 胜率 ≈ 50% 才成组，**该胜率来自海量人类玩家数据**）；教师模型参数量约为最终模型一半（9M vs 17M） | 自对弈 RL |
| P3 Phase 2 | **多教师策略蒸馏**：`L_distil(θ) = Σ_teacher_i E[Σ_t H^×(π_i(s_t)‖π_θ(s_t)) + Σ_head_k (V̂_i^k − V̂_θ^k)²]` ——**策略交叉熵 + 价值 MSE**，同时蒸馏策略与价值 | 教师模型的预测 |
| P3 Phase 3 | 用 Phase 2 蒸馏模型**初始化**，在英雄池内**随机阵容持续训练** | 自对弈 RL |
| 晋升规则 | `The rule of advancing to the next phase in CSPL is based on the convergence of Elo scores.` | Elo 收敛 |

---

## §4 奖励项清单（逐项、带来源）

### 4.1 1v1（P1，AAAI-2020，Table 6）

> 原文：`All the trained 1v1 heroes use the same reward, shown in Table 6. The reward design is inspired by OpenAI Five's Dota reward. The reward is zero-sum, i.e., one's mean reward is subtracted from that of the opponent.`

| 奖励项 | 权重 | 类型 | 描述（原文） |
|---|---|---|---|
| hp point | **2.0** | dense | the health point of hero |
| tower hp point | **10.0** | sparse | the health point of turrets and base |
| money (gold) | **0.008** | dense | the gold gained |
| **ep rate** | **0.8** | dense | the rate of mana |
| **death** | **−1.0** | sparse | being killed |
| **kill** | **−0.5** | sparse | kill an enemy hero |
| exp | **0.008** | dense | the experience gained |
| last hit | **0.5** | sparse | last hitting to enemy units |

- **零和**是显式设计（`zero-sum`），且框架支持**训练中在线观察奖励**（`Our framework allows on-the-fly reward analysis during training`，Fig. 7 给出貂蝉的训练中奖励曲线 x=训练小时）。
- ⚠️ **`kill = −0.5` 是 PDF 表格的字面值**（Python 抽取文本 `kill -0.5 sparse kill an enemy hero.`）。中文报道给出了与之一致的解释（「**注意看，这里杀敌的奖励反而是负的，因为这个过程会导致其他奖励都在增加，如果此处再给奖励会让AI过于执着于击杀敌人**」，<https://cloud.tencent.com.cn/developer/article/2160236>）。**但该解释并非腾讯官方论文原文**，且**后两版（HoK Arena）把 kill 改成 −0.6 也仍是负值** ⇒ 本报告按「字面负值 + 存在外部解释」记录，**机制归因标未验证**。

### 4.2 5v5（P3，NeurIPS 2020，Supplementary Table 4）— **含奖励分解**

P3 把奖励**分解成 5 个类别，作为 5 个价值头（multi-head value, MHV）**：

> `we introduce multi-head value (MHV) into MOBA by decomposing the reward, which is inspired by the hybrid reward architecture (HRA) used on the Atari game Ms. Pac-Man. Specifically, we design five reward categories as the five value heads ... based on game expert's knowledge and the accumulative value loss in each head.`
> `L_value(θ) = Ê_t[Σ_head_k (R_t^k − V̂_t^k)²],  V̂_t = Σ_head_k w_k V̂_t^k`

| 价值头 | 奖励项 | 权重 | 类型 | 描述（原文） |
|---|---|---|---|---|
| **Farming Related** | Gold | **0.005** | Dense | The gold gained. |
| | Experience | **0.001** | Dense | The experience gained. |
| | Mana | **0.05** | Dense | The rate of mana (**to the fourth power**). |
| | **No-op** | **−0.00001** | Dense | **Stop and do nothing.** |
| | Attack monster | 0.1 | Sparse | Attack monster. |
| **KDA Related** | Kill | 1 | Sparse | Kill a enemy hero. |
| | **Death** | **−1** | Sparse | Being killed. |
| | Assist | 1 | Sparse | Assists. |
| | Tyrant buff | 1 | Sparse | Get buff of killing tyrant, dark tyrant, storm tyrant. |
| | Overlord buff | 1.5 | Sparse | Get buff of killing the overlord. |
| | Expose invisible enemy | 0.3 | Sparse | Get visions of enemy heroes. |
| | Last hit | 0.2 | Sparse | Last hitting an enemy minion. |
| **Damage Related** | Health point | **3** | Dense | The health point of the hero (**to the fourth power**). |
| | Hurt to hero | 0.3 | Sparse | Attack enemy heroes. |
| **Pushing Related** | Attack turrets | 1 | Sparse | Attack turrets. |
| | Attack crystal | 1 | Sparse | Attack enemy home base. |
| **Win/Lose Related** | Destroy home base | **2.5** | Sparse | Destroy enemy home base. |

**有奖励分解吗？→ 有，而且是双重的**：① 训练目标把奖励按 5 类拆成 5 个价值头、**每头单独回归**（`Σ_head_k (R_t^k − V̂_t^k)²`），② 总价值是各头的**加权和** `V̂_t = Σ w_k V̂_t^k`。
⚠️ 原文**没有给出 `w_k` 的具体数值**（只给出每项的奖励权重）⇒ **`w_k` 数值：未验证**。

### 4.3 官方开源环境（P6，NeurIPS 2022，Table 5）— **对外可复现的那张表**

> 原文：`Users can customize and redefine their own rewards (including the termination rewards) from the returned 'info' from 'env.step()', while we provide a basic set of rewards here.`

| 奖励项 | 权重 | 类型 | 描述（原文） |
|---|---|---|---|
| hp_point | **2.0** | dense | the rate of health point of hero |
| tower_hp_point | **10.0** | dense | the rate of health point of tower |
| money (gold) | **0.006** | dense | the total gold gained |
| ep_rate | **0.75** | dense | the rate of mana point |
| **death** | **−1.0** | sparse | being killed |
| **kill** | **−0.6** | sparse | killing an enemy hero |
| exp | **0.006** | dense | the experience gained |

五类划分（P6 §3.2 / Appendix F 原文）：
> `1) Farming related: the amount of gold and experience, and the penalty of not acting, which are dense reward signals; 2) KDA related: the number of kill, death and assist, and the last hit to enemy units, which are sparse reward signals; 3) Damage related: a dense reward - the number of health point, and a sparse reward - the amount of attack to enemy hero; 4) Pushing related: the amount of attack to enemy turrets and crystal, which are dense rewards; 5) Win/lose related: destroy the enemy home base, which are sparse reward signals received at the end of the game.`

**关于「时间/节奏」项**：三篇论文的奖励表**均无显式的「时间惩罚/节奏项」**。与时间相关的只有：
- **折扣因子**（承担远视界信用分配）：**1v1 线**（P1）为 `discount factor = 0.997`（原文：`for the case of Honor of Kings, this discount is valuing future rewards with a half-life of about 46 seconds`）；**开悟**（P6）PPO 用 **0.997**、DQN 目标网络用 **0.98**；而**5v5**（P3）写 `The discount factor is set as 0.998`。⚠️ **P3（5v5）与 P1（1v1）/P6（开悟）的折扣因子不一致（0.998 vs 0.997），并列不调和，不做调和**——注意 P1 与 P6 之间是**一致**的。
- P3 的**推塔/水晶/胜负**项本身即承担节奏（`high-level turret pushing without minions` 被列为 AI 表现）。

### 4.4 课程学习（curriculum）—— **有，且是 lineup 级的**

| 层次 | 设计 | 来源 |
|---|---|---|
| 阵容课程 | **CSPL 三阶段**：固定阵容 → 多教师蒸馏 → 随机阵容持续训练；`start small` | P3 §3.3，Fig. 2 |
| 阶段晋升 | **按 Elo 收敛**触发 | P3 §3.3 |
| 分组依据 | 40 英雄分 4 组 × 10；**组内 5v5 胜率 ≈50%**（该胜率用人类玩家数据算出） | P3 §3.3 |
| 模型容量课程 | 教师 **9M** 参数 → 最终 **17M** 参数 | P3 §4.1 |
| 环境课程 | **P6 的 20 英雄 → 20×20 = 400 个 task**，用来暴露泛化缺口 | P6 §3.1 |
| 分段（非课程但有「时间分段」） | 观测里有 `VecCampsWholeInfo → Current Period: Divide game time into 5 periods`（把一局按时间划成 5 段） | P6 Appendix D |

### 4.5 「奖励塑形 + 退火 / 多阶段权重」—— **本报告最重要的一条否定性发现**

| 子问题 | 结论 | 证据 |
|---|---|---|
| 有**奖励塑形**吗？ | **有（密集项本身就是塑形）** | P1/P3/P6 的 dense 项：hp、mana^4、gold、exp、tower_hp 等。P6 明说 `Honor of Kings has both sparse and dense reward configurations` |
| 有**奖励分解**吗？ | **有** | P3 的 5 个价值头 + 加权和（§4.2） |
| 有**课程**吗？ | **有** | P3 CSPL 三阶段（§4.4） |
| 有**奖励项权重的退火 / 多阶段调度**吗？ | **在一手可核实文献中找不到任何直接证据** | 已逐字检索 P1/P2/P3/P5/P6/P7 全文的关键词（`anneal` / `schedule` / `reward weight` / `weight decay schedule` / `curriculum weight`），**无一处描述奖励权重随时间/阶段变化**；P1/P3/P6 的奖励表都是**固定权重** |
| 有**多阶段的价值头权重 `w_k` 调度**吗？ | **未验证** | P3 只给 `V̂_t = Σ w_k V̂_t^k`，**未给 `w_k` 数值**，更未提调度 |
| 有**基于累计价值损失的奖励项选择**吗？ | **有（这是唯一可核实的「按损失调奖励」机制）** | P3 原文：5 个价值头是 `based on game expert's knowledge and **the accumulative value loss in each head**` |

> **纪律声明**：`anneal` 类做法在 OpenAI Five（Dota 2）里是知名技术，但**本项目禁止跨项目数字对拍**，且 P1 只说 `reward design is inspired by OpenAI Five's Dota reward`（**只提奖励设计灵感，未提退火**）⇒ **不得**由 OpenAI Five 反推绝悟用了退火。

---

## §5 惩罚 / 约束

### 5.1 显式惩罚项（可核实）

| 惩罚项 | 权重 | 力度与形态 | 来源 |
|---|---|---|---|
| **No-op（不做动作）** | **−0.00001**（**仅 5v5 有**） | 密集惩罚，量级极小（约为 kill=1 的 **1/100000**），作用是**打破平局/给出方向而非强约束** | P3 Table 4 |
| Death（被击杀） | **−1.0**（1v1、开悟）/ **−1**（5v5） | 稀疏惩罚 | P1 Table 6 / P3 Table 4 / P6 Table 5 |
| Kill（击杀） | **−0.5**（1v1）/ **−0.6**（开悟）/ **+1**（5v5） | 1v1 与开悟为**负**，5v5 为**正**；**不一致，并列不调和** | P1 / P6 / P3 |

**关于「送死 / 消极 / 无效动作」**：
- **「消极」有针对性设计**：`No-op (Stop and do nothing) = −0.00001`。**但 1v1 与开悟的奖励表里没有 no-op 项**（P1 Table 6、P6 Table 5 均无）⇒ **1v1 系统里没有显式的「不作为」惩罚**。
- **惩罚力度是真实调参痛点（中文一手受访原话）**：`如果罚得太多，英雄就会不敢出战，各种逃窜，甚至掉一丝血就想回城补血；如果罚得太少，英雄又会傻冲，容易被团灭。`（[article/2160236](https://cloud.tencent.com.cn/developer/article/2160236)）
- **没有找到**任何「无效动作 / 越界动作」被 F 扣分的描述（见 §5.2：非法动作走的是**掩码剔除**而非惩罚）。

### 5.2 行为约束 / KL 锚定 / 规则门禁

| 候选机制 | 是否存在于绝悟 | 证据（原文） |
|---|---|---|
| **动作合法性掩码（action mask）** | **✅ 是核心设计，三代论文均强调** | P1：`a game-knowledge-based pruning method, called action mask, is developed to guide explorations during the reinforcement process`；`the action mask helps eliminate several unreasonable aspects` |
| **trust region / PPO ratio clip** | **✅ 有，但是标准 PPO 的 ratio clip** | P1：`the standard PPO algorithm involves a ratio clip ... to penalize extreme changes to the policy` |
| **Dual-clip PPO（比标准 PPO 更强的不对称裁剪）** | **✅ 有，且是绝悟系列标志性算法** | P1 提出；P3/P5 沿用。动机：`when A_t < 0 ... the ratio will introduce a big and unbounded variance since r_t(θ)·Â_t ≪ 0`，用第二个下界 `c·Â_t`（`c=3`, `ϵ=0.2`）兜住 |
| **KL 锚定（对旧策略/人类策略的显式 KL 惩罚项）** | **❌ 未找到** | P1/P2/P3/P5/P6 全文均无「KL penalty / KL regularization / KL anchor」项。P3 的蒸馏用的是**交叉熵 `H^×(π_i‖π_θ)`**（教师→学生），不是 KL 锚定 |
| **规则门禁（reward gate / 行为门禁）** | **❌ 未找到** | 无任何「满足条件才发奖」的门控项 |
| **观测层遮蔽** | **✅ 有** | P6 Appendix D：敌人不可见单位的状态**也在观测里**（`the states of enemy units, even if they're invisible to the ego camp, are also available`），但**`should be masked out during policy execution`**；P3 明确该全套信息**只喂价值网络**（`this is performed only during training, as we only use the policy network during evaluation`） |

### 5.3 非法动作（不合法技能 / 越界）在系统里怎么处理 —— **剔除，不是惩罚**

**P1（1v1）原文给出的四类被掩码剔除的「不合理动作」**（逐字）：
> `1) physically forbidden areas on map, e.g., suppose the predicted action is to move towards a direction, which cannot be performed as that direction is occupied by obstacles in the map; 2) skill or attack availability, e.g., the predicted action to release a skill within Cool Down time shall be eliminated; 3) being controlled by enemy hero skill or equipment effects; 4) hero-/item-specific restrictions.`

**P6（开悟官方环境）把这件事做成了 API 契约**：
> `legal_action describes current legal sub-actions with 1 NumPy array. The legal sub-actions incorporates prior knowledge of experienced human players and helps eliminate several unreasonable aspects: 1) skill or attack availability ...; 2) being controlled by enemy hero skill or equipment effects; 3) hero-/item-specific restrictions.`
> `sub_action_mask is a NumPy array describing **dependencies of different Button actions**.`
> 动作空间本身也带游戏知识：`Note that different heroes have different **prohibited skill offsets** since they have different skills.`

**P6 的消融实验（关键量化证据）**：
> `H.2 Legal action mask ... it provides legal action information to help eliminate unreasonable actions for reference, where researchers can design their own legal actions based on their knowledge. We can use the legal action information to mask unreasonable actions out during the training process.`
> `As expected, **without legal action, the agent quickly converges to a local optimum** under the large action space, and **it is critical to use action mask** for better performances.`（Fig. 13：reward / hurt per frame / win rate 三项）

⇒ **结论**：绝悟体系对非法动作的处理方式是 **「策略输出层屏蔽（mask）+ `sub_action_mask` 表达动作间的依赖**」；**公开文献中不存在「非法动作扣分」的惩罚项**。这正是与我们项目最大的接口差异点（见 §7.1 A1）。

---

## §6 已知的坑 / 失败模式

> 说明：**以下全部为论文作者自己写下的失败/局限**。**没有任何一手来源提到绝悟出现「reward hacking」或「AI 学会卡 bug」**——这两点在本报告中标 **未验证**。

### 6.1 「学习崩溃」（learning collapse）—— 扩大英雄池时的真实失败

> P3 §1 原文：`existing methods by randomly presenting these disordered hero combinations to a learning system can lead to "learning collapse", which has been observed from both **OpenAI Five** and **our experiments**. For instance, OpenAI attempted to expand the hero pool up to 25 heroes, resulting in unacceptably slow training and degraded AI performance, even with thousands of GPUs.`

### 6.2 泛化崩溃：胜率**恒为 0**（最硬的失败读数）

> P6 §6 原文：`we also noticed that for certain tasks like opponent heroes changed to Peiqinhu/Shangguanwaner in Figure 6, or target heroes changed to Peiqinhu/Shangguanwaner in Figure 7, the winning rate for the Diaochan model is **constantly zero**. This makes it hard to evaluate the performance of different techniques for generalization.`
> 以及：`the performance of the same trained policy **drops dramatically**, as the change of opponent hero differs the testing setting from the training setting, indicating the **lack of transferability** of the policy learned by existing methods.`
> 缓解手段（不是根治）：**multi-task training** 与 **student-driven policy distillation**（`improves the performance in all the test tasks`）。

### 6.3 没有 action mask 时**收敛到局部最优**

> P6 H.2：`without legal action, the agent quickly converges to a local optimum under the large action space`（§5.3 已引）

### 6.4 奖励函数的**策略偏置**：一套通用奖励 → 所有阵容都打同一种打法

> P5 §4.4 原文：`In Honor of Kings, the most common strategy for human players is three-lane-strategy ... Therefore, **the general reward function [Ye et al. 2020a] is designed following the common strategy, so agents will perform similar strategies in different lineups and thus cannot perform the most suitable strategy in some special lineups.**`
> P5 §1：`... the same hand-crafted reward signal and then present similar strategies even in different matches`
> 量化：Table 2 `macro-state entropy` — Built-in bots **0.000** / RL-baseline **0.014** / MGG **0.408**（同阵容不同局里策略多样性）

### 6.5 IL 数据分布问题（P2 自己承认并处理的三个坑）

| 坑 | 原文 | 处理 |
|---|---|---|
| **场景不平衡** | `due to the defined scenes types will appear imbalanced`（推塔 vs 团战频率差很多，中文二手转述）；原文为 `Scene Identification` + 欠采样 | 按场景欠采样/平衡 |
| **无效移动噪声** | `in other scenes, it can be a coarse-grained step, since **the player usually executes meaningless move actions**` | 移动方向标签改为**未来 N 帧位置差**：Combat **N=5（0.33 s）**，其他 **N=15（1 s）** |
| **目标选择样本不平衡 → 模型偏向错误目标** | `the examples of target selection for attacking are imbalanced between low-damage high-health (LDHH) heroes and high-damage low-health (HDLH) heroes ... **Without downsampling, the model will prefer to attack the LDHH hero. However, the prior target normally is an HDLH hero, which is the key to win a local battle.**` | **attack sample normalization**（对同一整个攻击过程采样相同数量样本）；`dramatically improves the quality of target selection` |
| **职业选手操作也有噪声**（若直接 IL 人类） | P5 §3：`top players' matches are observed to have high-quality and diverse macro-strategies but **noisy micro-operations**. Thus, **completely imitating human demonstrations, including micro-operations, will worsen the performance of the policy.**` | MGG：**只模仿 macro-goal**，微操交给 RL（`avoid the decline of micro-operation ability caused by completely imitation learning`） |
| **SL 的固有代价** | P2 §V 讨论：`one drawback of SL is the requirement of **high-quality human data**. Furthermore, **SL's upper limit is theoretically lower than the RL counterpart**` | 定位为 tradeoff |

### 6.6 稀疏奖励的固有难度（作者主动承认）

> P6 Appendix F：`It is possible that only using the dense reward version of the Honor of Kings will likely resemble the sparsity, which is often seen in previously sparse rewarding benchmarks. Given the sparse-reward nature of this task, **we encourage researchers to develop novel intrinsic reward-based systems, such as curiosity, empowerment, or other signals to augment the external reward signal** provided by the environment.`

---

## §7 对我们项目（Clash Royale 自对弈 PPO）的可迁移点 与 不可迁移理由

**我们的基线（来自本项目 `AGENTS.md` 与 `docs/`）：** 993K 参数、单机、**无人类数据**、**手工 22 键奖励表**、**无 MCTS**、**单 seed/臂 时胜率分辨率 ±0.31**（`docs/et_solo100k_judgment_2026-09-18.md` §11.13.11）、已有 `invalid_penalty=0.05` 与四层动作合法性（`docs/illegal_action_layers_2026-09-18.md`）。

### 7.1 可迁移（按性价比排序）

| # | 迁移点 | 绝悟证据 | 为什么我们能用 | 落地形态建议 |
|---|---|---|---|---|
| **A1** | **把「非法动作 = 接口契约」而非「奖励惩罚」** | P6 的 `legal_action` / `sub_action_mask` 是 **env 返回的 API 字段**，不是 reward 项；我们的 `invalid_penalty` 是 reward 项（`env_wrapper.py:50`）| 我们已查实**四层并列且互不推导**（掩码过严 C1/C6、掩码缺口 C2、作用域空隙 C3、掩码专有约束 C4），根因是**没有单一权威的「合法动作」语义源** | 照着 `legal_action` + `sub_action_mask`（**动作间依赖**）建**一层可复算的合法性产物 + 位数对账**，让 22 键奖励表**不再承担「纠错」职能**（避免奖励项互相抵消） |
| **A2** | **多价值头 + 奖励分解** | P3 `V̂_t = Σ_head_k w_k V̂_t^k`，5 头（Farming/KDA/Damage/Pushing/Win-Lose），损失是**每头单独 MSE** | 我们**已经有 22 键手工奖励表**——这正是「奖励分解」的原料；把 22 键按功能归并成 4–6 头，比重新设计奖励**风险低得多**（不改奖励语义 ⇒ 不违【R2】） | 先做**价值头分组**（不改奖励），再单独预注册「头权重 `w_k`」实验；**学绝悟：`w_k` 不要和奖励项权重混为一谈**（P3 两者是分开的两套数） |
| **A3** | **场景/条件分层采样（scene-based sampling）** | P2 的 Scene Identification（推塔/团战/清线/打野…按轨迹分段）+ 欠采样 + **attack sample normalization** | 我们的病根之一是**罕见状态从未被采样**（「放弃一张买得起的牌」75,891 帧 **0 次**，`docs/elixir_saving_audit_2026-09-18.md`）| 学 P2 的**按场景分段 + 每类等量采样**（而不是全局均匀），做**离线数据集层面的再平衡**（不碰在线探索，零 off-policy 风险）|
| **A4** | **只模仿「高层意图」而非底层动作** | P5：人类微操有噪声，**完全模仿会掉性能**；只模仿 macro-goal + 内蕴奖励 `R_goal = ‖f(s_t)−ĝ_t‖ − ‖f(s_{t+1})−ĝ_t‖` | 我们**没有人类数据**，但完全可以把这条**反向用**：把「专家录像/规则建议」降级为**高层意图信号**（我们已有 `belief_planner` 的 `PlanToken`、`save_ace` 等） | 任何手写专家规则，**只作为 macro-goal 的先验/辅助信号**，绝不用它直接替代策略动作（避免 S1 门禁里「手写专家打得出、模型学不到」的错配） |
| **A5** | **零和 / 对称化奖励结构** | P1：`The reward is zero-sum, i.e., one's mean reward is subtracted from that of the opponent` | 我们已有「按局面结算的圣水交换」提案（`docs/frame_credit_proposal_review_2026-09-18.md`），其**对称性**要求与 zero-sum 同源 | 把**新奖励项一律写成两视角反对称**，并用脚本对账（我们已有 `analyze_online_trade.py` 这层仪器） |
| **A6** | **用「累计价值损失」挑选奖励项** | P3：5 个头的划分依据 = `game expert's knowledge` **+ `the accumulative value loss in each head`** | 我们**本来就在记录**每项的贡献路径（`docs/item_health.json`、`readout_et_solo100k/item_health.json`） | 把它形式化成**奖励项体检指标**：某项的价值头损失长期不降 ⇒ 该奖励项是噪声或已被重复计数（我们已实测过 `τ == kill_credit` 的**重复计数**缺陷，`docs/s2_instrument_2026-09-18.md` §7） |
| **A7** | **渐进式课程（阵容/对手池从小到多）** | P3 CSPL：固定阵容 → 蒸馏 → 随机阵容；晋升按 Elo 收敛 | 我们已有 `solo` / `run` / `flow` 三种模式与 `FOUR_DECK_SET` / 139 卡池（`run_league.py:795-812`） | 方向与我们的「先 `solo`、后扩对手」一致，**但它不是新信息**⇒ 优先级低于 A1–A3 |

### 7.2 **不可迁移**（含明确理由）

| # | 不可迁移项 | 绝悟的规模/前提 | 为什么我们不适用 |
|---|---|---|---|
| **N1** | **宏观算力与分布式 actor-learner** | P1：**600,000 CPU 核 + 1,064 GPU**（单英雄 48×P40 + 18,000 核）；P3：**320 GPU + 35,000 CPU**（一个 resource unit），集群同时跑 6–7 个实验；P1 自述「每天每英雄 ≈ **500 年人类数据**」 | 我们**单机**。绝悟的 off-policy 适应、Dual-clip、蒸馏**都是为「跨众多策略源的巨量 off-policy batch」设计的**；在单机上直接照搬会把「大 batch 稳定性问题」引入而没有对应收益 |
| **N2** | **单英雄一模型 / 17M 参数** | P2：`We train one model for one AI hero`，每英雄 16×P40×36h；P3：教师 9M / 最终 17M，LSTM 512→1024，LSTM 步长 16 | 我们 **993K 参数、一个模型管全局**。**绝不能跨项目对拍参数量再按比例调超参**（P2 的「一英雄一模型」与我们的「一模型全牌」不是同一个学习问题） |
| **N3** | **MCTS / 树搜索** | P4 JueWuDraft（MCTS+PUCT）、P3 drafting 用 UCT + 胜率预测器（**3000 万样本**训胜率预测器、**1 亿样本**训价值网络）；P3 明确 `value network` 用 MCTS 自对弈数据训练 | 我们**无 MCTS**，且【R11】明确「不上训练时 MCTS」。绝悟的 MCTS 还依赖**廉价可复现的 gamecore 仿真**——我们没有等价物 |
| **N4** | **人类数据驱动的 IL / 教师蒸馏** | P2 每英雄 **12 万局 × 前 1% 玩家**；P5 用**顶级电竞录像**抽 macro-goal；P3 教师是**固定阵容 RL 模型**（也要自对弈算力） | 我们**无人类数据**。⇒ IL 类迁移全部**降级**为「手写规则/启发式做高层意图」（A4），**不做**任何形式的 BC 训练（无数据源，且会引入我们无法验证的分布外动作） |
| **N5** | **奖励手工调参的「千次迭代」工作流** | 中文一手口径：`往往奖励值变化一点点，就会导致训练出来的英雄性格迥异。所以，需要技术宅们蹲在电脑前反复调整实验` | 我们的**统计分辨率**（【R5】/【R16】：20k 单跑只看大效应；**同配置重跑单点差 0.60**）**不支持**「小步改奖励、快跑对比」的调参循环 ⇒ 我们必须**先解决可分辨性**（多 seed / 配对判据），再谈奖励项增删 |
| **N6** | **「模仿人类实现温启动 SL→RL」这条具体路径** | P2 明说 SL 可作 RL 初始化，但**P3 的 5v5 RL 论文并未采用**（见 §3.3 的未验证项） | 即使想模仿也**没有可复现的官方配方**；照抄文档不存在的流程 = 凭空造假设 |
| **N7** | **英雄/技能专属的硬编码约束** | P1：`hero-/item-specific restrictions`；P6：`different heroes have different prohibited skill offsets` | 我们的卡牌机制**本身就需要**类似东西（我们已查出 C1 MergeMaiden 静态 6 vs 引擎动态 3/6、C6 Miner 全区豁免、F1 `_effective_card` 只处理 Mirror）⇒ **可借用「必须有单一权威源」的原则**，但**不能借用他们的具体规则表**（游戏不同） |
| **N8** | **「无 no-op 惩罚也不消极」的乐观假设** | P1/P6 的奖励表**都没有 no-op 项**，但它们有**塔血密集项**（tower_hp_point 10.0）天然推动进攻；且 1v1 地图小、目标单一 | 我们的实测是：**「放弃一张买得起的牌」在 75,891 帧里发生 0 次**、**88–91% 的帧买不起任何手牌**（`docs/pass_prob_2026-09-18.md`）⇒ 我们的「不作为」是**机制/圣水经济**造成的，**不是奖励缺项**；照抄「加一个 no-op 惩罚」会**误治病灶**（我们已实测：抬 stop 概率能把圣水≥6 的帧从 0% 抬到 48%，但**每局回报从 −4.46 掉到 −41.5**，`docs/exploration_bias_pass_2026-09-18.md`） |

---

## §8 来源清单（URL + 取数时间）

**全部取数时间：2026-09-19 03:30–03:45 CST（UTC+8）**

| 类别 | 内容 | URL |
|---|---|---|
| arXiv API（标识/时间复算） | 2011.12582 / 2011.12692 / 2012.10171 的标题、提交时间、venue | <https://export.arxiv.org/api/query?id_list=2011.12582,2011.12692,2012.10171> |
| 论文 P1（1v1，含 Table 6 奖励表、action mask 四类、Dual-clip PPO、系统配置） | 全文 PDF 抽取（8 页 / 41,205 字符，sha256 前缀 `5f1a28430e9160eb`） | <https://arxiv.org/pdf/1912.09729v3>；摘要页 <https://arxiv.org/abs/1912.09729v2>；AAAI 版 <https://ojs.aaai.org/index.php/AAAI/article/view/6144> |
| 论文 P2（JueWu-SL，含数据 12 万局/1 亿样本、四项 CE 损失、场景采样、attack sample normalization） | 全文 PDF 抽取（11 页 / 61,785 字符，sha256 前缀 `d50a988b5a1b31a8`） | <https://arxiv.org/pdf/2011.12582v1>（⚠️ arxiv.org 无该文 HTML；`https://arxiv.org/html/2011.12582v1` 返回 **404 No HTML**） |
| 论文 P3（5v5 RL，含 Table 4 完整奖励表、MHV、CSPL 三阶段、32 万 GPU/3.5 万 CPU） | 全文 PDF 抽取（15 页 / 57,832 字符，sha256 前缀 `ceaf3530a1bfa107`）；HTML 版 | <https://arxiv.org/pdf/2011.12692v4>；<https://arxiv.org/html/2011.12692v4>；NeurIPS 论文集 <https://proceedings.neurips.cc/paper/2020/file/06d5ae105ea1bea4d800bc96491876e9-Paper.pdf> |
| 论文 P6（开悟官方环境，含 Table 5 奖励表、`legal_action`/`sub_action_mask`、H.2 legal action mask 消融、§6 泛化失败「胜率恒为 0」） | 全文 PDF 抽取（20 页 / 55,706 字符，sha256 前缀 `f5044fb32ebbb20b`）；HTML 版 | <https://arxiv.org/pdf/2209.08483v3>；<https://arxiv.org/html/2209.08483v3> |
| 论文 P5（MGG，含「人类微操有噪声、完全模仿会掉性能」、内蕴奖励 `R_goal`、macro-state entropy 表） | 全文 PDF 抽取（12 页 / 43,726 字符，sha256 前缀 `9a1dc08505a4132c`） | <https://arxiv.org/pdf/2110.14221.pdf>；<https://arxiv.org/abs/2110.14221> |
| 论文 P7（HMS，前身） | 全文 PDF 抽取（8 页 / 35,804 字符，sha256 前缀 `6e08537820d2357c`） | <https://arxiv.org/pdf/1812.07887.pdf> |
| 论文 P4（JueWuDraft） | 摘要页 + HTML 全文（本次仅用于标识核对） | <https://arxiv.org/abs/2012.10171>；<https://arxiv.org/html/2012.10171v4> |
| 官方项目页 | 腾讯开悟「关于我们」 | <https://tencentarena.com/aiarena/zh/about> |
| 官方开放服务页 | 指向官方开源环境 hok_env + P6 论文；Windows-only；需承诺函 | <https://tencentarena.com/aiarena/zh/open-gamecore> |
| 官方平台动态 | 2020-05-07 起的时间线；2022-11-20「王者荣耀AI开放研究环境」启动申请 | <https://tencentarena.com/aiarena/zh/trends> |
| 官方开源仓库 | `tencent-ailab/hok_env`（Honor of Kings AI Open Environment of Tencent） | <https://github.com/tencent-ailab/hok_env>（⚠️ **本会话内 `raw.githubusercontent.com` SSL 失败、`github.com` git clone 133 s 超时**，仅 github.com 网页经 web_fetch 首页可读；**仓库源码文件未能取到**） |
| 中文一手受访报道（非腾讯官方执笔，但含王者 AI 团队原话） | 《王者荣耀的B面：人类在此喧闹，AI却在他们脚下悟道》 | <https://cloud.tencent.com.cn/developer/article/2160236> |
| 中文二手技术转述（非官方） | 《王者荣耀：在绝悟上进行监督学习》（作者自述「腾讯后台策略工程师」，2022-04-10 原发 / 2022-11-04 发布） | <https://cloud.tencent.com/developer/article/2150992> |
| 事件性中文报道（转载雷锋网） | 2019-08-02 世界冠军杯 + ChinaJoy 1v1（首日 504 场 99.8%） | <https://www.openi.org.cn/腾讯-ai-「绝悟」制胜王者荣耀，升级至电竞职业水平/> |

**留档**：六篇论文的全文抽取文本已复制到本仓库 `docs/juewu_research_2026-09-19/`（`juewu_1v1.txt` / `juewu_sl.txt` / `juewu_rl5v5.txt` / `hok_arena.txt` / `macrog.txt` / `hms.txt`），sha256 见 §8 表内前缀。

---

## §9 未验证 / 未能核实项（**禁止当作事实引用**）

| # | 未验证项 | 状态与原因 |
|---|---|---|
| **U1** | **「奖励塑形 + 退火 / 多阶段权重」是否存在** | **未验证（倾向「无直接证据」）**：P1/P2/P3/P5/P6/P7 全文检索 `anneal` / `schedule` / `reward weight` 均无「权重随时间变化」的描述；三张奖励表都是固定权重。**不能据此断言绝悟没用过**——官方未公开训练细节，且这是通常不写进论文的工程细节 |
| **U2** | **P3 的 5 个价值头权重 `w_k` 的数值** | **未验证**：P3 只给公式 `V̂_t = Σ w_k V̂_t^k`，未给数值，也未提是否调度 |
| **U3** | **`kill = −0.5` / `−0.6` 的机制归因** | **字面值已验证**（PDF 表格逐字），**「为什么为负」的解释未验证**（中文报道的解释非官方论文原文） |
| **U4** | **P1 vs P6 奖励权重差异的原因**（money 0.008→0.006、ep_rate 0.8→0.75、kill −0.5→−0.6） | **未验证**：无任何来源说明是版本迭代、场景适配还是口误/排版 |
| **U5** | **折扣因子不一致**（P3 5v5 `0.998` vs P1 1v1 / P6 开悟 `0.997`） | **并列不调和**：均为论文原文。**补充实测**：P1 与 P6 **一致**（都是 0.997）；只有 P3 不同 ⇒ 差异口径是「5v5 vs 1v1/开悟」。不做归一化，不推断哪个是「实际用的」 |
| **U6** | **绝悟是否出现 reward hacking / 卡 bug / 利用地图漏洞** | **未验证 / 未找到**：六篇论文与官方页面**均无此类记载**。「AI 学会消极」的唯一一手相关材料是 **no-op 惩罚项**（P3，5v5）与中文受访的**惩罚力度两难**表述；**1v1 与开悟的奖励表均无 no-op 项** |
| **U7** | **「绝悟 5v5 最终版用 JueWu-SL 初始化 RL」** | **未验证**：P2 只说 `As an ongoing step, we are combining SL and RL`；**P3 全文未提 SL 初始化** |
| **U8** | **hok_env 源码中的奖励/合法性实现细节**（我们项目的 `invalid_penalty` 对应物是否存在） | **未验证 / 未能取到**：本会话 `raw.githubusercontent.com` SSL 失败、`github.com` `git clone` 133 s 超时。**仅能从 P6 的 API 文档与论文描述确认接口契约**，未读代码 |
| **U9** | **「游戏策划人工标注 KPL 优秀操作」** | **未验证**：只见于中文二手报道 <https://cloud.tencent.com.cn/developer/article/2160236>，**P2 原文无对应表述**；P2 的数据来源是**前 1% 玩家的对局 + 表现分筛选**（不是职业录像人工标注） |
| **U10** | **开悟平台的「奖励信号接口」是否允许选手自定义奖励** | **部分验证**：P6 原文 `Users can customize and redefine their own rewards ... from the returned 'info' from 'env.step()'` ⇒ **允许自定义**；但**平台层面（非论文）的具体表单/字段未验证**（未取到平台文档站 <https://aiarena.tencent.com/hok/doc/>，该域名属 P6 引用的旧文档地址） |
| **U11** | **绝悟 1v1 AI 击败职业选手的确切日期/比分** | **部分验证**：P1 Table 3 给出 BO5 **3:0 / 3:0 / 3:0 / 3:1 / 3:0**（5 个英雄，对手为 KPL 现役顶尖选手，Table 2 列出 eStarPro/QGhappy/TS/WE）与 Table 4 公开赛 **2100 场 99.81%**（2019-08-02 至 08-05，ChinaJoy）；**但具体对局日期在 P1 原文中未逐场给出** |
| **U12** | **`sub_action_mask` 的确切维度/语义** | **未验证**：P6 只给一句话描述（`describing dependencies of different Button actions`），**未给维度与取值语义**；本文档中的 `Table 4 Action Space` 给了各维度（Button 11 类、Move X/Z 各 16、Skill X/Z 各 16、Target 8 类），但**两者的对应关系未在论文中说明** |
