# 绝悟 / JueWu（王者荣耀 MOBA）RL 阶段奖惩设计 · 统一对比表

**取数时间：2026-09-19 04:35–04:50 CST（UTC+8）｜观测者：调研执行者（本会话）**
**方法：只读 `docs/juewu_research_2026-09-19.md` + `docs/juewu_research_2026-09-19/*.txt` 六份论文全文抽取，逐字 grep 复核数字。**
**本会话独立复核的抽取文件 sha256 前缀（16 位，与主报告 §8 逐字一致）：**
`juewu_1v1.txt 5f1a28430e9160eb` / `juewu_sl.txt d50a988b5a1b31a8` / `juewu_rl5v5.txt ceaf3530a1bfa107` / `hok_arena.txt f5044fb32ebbb20b` / `macrog.txt 9a1dc08505a4132c` / `hms.txt 6e08537820d2357c`

> **纪律**：① 本文所有数字均带行号出处（`<文件>:<行>`），可原样 grep 复现；② `w_k` 未给就写「未给」；③ **禁止跨项目数字对拍**；④ 口径冲突**并列不调和**。
>
> **★ 统一 schema（五份对象报告共用：字段名 / 顺序 / 单位口径一致 — 2026-09-19 编排者补，仅加声明、不改正文）**
> **字段集 ①–⑧**：① 奖励项清单（项名 / 权重 / 量纲 / 作用域 / 是否零和）・② 惩罚项（显式 / 隐式 / 用约束或掩码替代）・③ 塑形与调度（是否 PBRS？是否退火？局内时间权重？）・④ 归一化三问（奖励 / 价值 / 优势）・⑤ value head 设计・⑥ 已知坑（带原文句；二手必须标注）・⑦ 一手出处（URL / `file:line` + 取数时间）・⑧ 阶段归属（IL / RL / 跨阶段接口）
> **本报告落点**：①=§1.1（三表合并） ・ ②=§3.1（另见 §3.2 掩码契约） ・ ③=§5（另见 §1.3） ・ ④=§4 ・ ⑤=§2 ・ ⑥=§6 ・ ⑦=§8 ・ ⑧=**不适用**（纯 RL 自对弈，无 IL 阶段；蒸馏见 §3.3 / §7.3-N5）
> **统一单位口径**：权重一律**按来源原样记录**并显式标注量纲（分母单位，见 §1.1「量纲（Description）」列）；无单位者写「无单位」；**同名项若量纲不同则并列不调和**，不折算、不换算、不对拍。
> **证据等级图例**（沿用盘上约定，不新建体系）：`[一手-论文原文]`・`[一手-官方代码]`・`[复现件-OA论文]`・`[二手-分析]`・`[未验证]`・`[本地-实测]`
> **三条硬纪律**：① **禁止跨项目数字对拍**（一切数字只用于标识与各自内部一致性论证）；② **标「未验证」的项一律保持未验证**，不因抽成表而升级证据等级；③ **不得把 IL 阶段的事写成 RL 阶段**，跨阶段项必须落在 ⑧（本对象 ⑧ 不适用）。

---

## §0 一句话结论

绝悟系列的 RL 奖惩在公开一手来源中可核实的形态是：**三张「按游戏功能分组 + 权重手工固定」的奖励表 + 动作合法性掩码（`legal_action` / `sub_action_mask`）接口 + 阵容级三阶段课程自对弈（Elo 晋升）**；惩罚侧**只有 3 类**（Death / Kill / No-op，且 Kill 符号三张表互相矛盾），**非法动作走掩码剔除而非扣分**；**奖励权重退火：未找到**；**5 个价值头的权重 `w_k` 数值：论文未给**；**奖励/价值/优势的归一化：三篇均未描述**。

---

## §1 三张奖励表合并对比

三张表分别是 P1（1v1，AAAI-2020，**Table 6**）、P3（JueWu-RL 5v5，NeurIPS 2020，**Table 4**，在 Section 8.5 Reward Design 之下）、P6（开悟官方环境 HoK Arena，NeurIPS 2022，**Table 5**，位于 Appendix F）。

### 1.1 逐项合并对比表（按功能分组）

| 功能组 | 项名 | 1v1 (P1 T6) 权重 | 5v5 (P3 T4) 权重 | 开悟 (P6 T5) 权重 | 量纲（原文 Description） | 稀疏/密集 | 作用范围 |
|---|---|---|---|---|---|---|---|
| — | hp point / hp_point | **2.0** | — | **2.0** | 1v1：`the health point of hero`；开悟：`the rate of health point of hero` | dense（1v1 表标 dense；开悟表标 dense） | 双方英雄（P1 零和） |
| — | Health point（Damage 头） | — | **3** | — | `The health point of the hero (**to the fourth power**)` | Dense | 双方英雄 |
| — | tower hp point / tower_hp_point | **10.0** | — | **10.0** | 1v1：`the health point of turrets and base`；开悟：`the rate of health point of tower` | **1v1 表标 sparse，开悟表标 dense（两表标注冲突，并列保留）** | 塔/基地 |
| Farming | money (gold) | **0.008** | **0.005** | **0.006** | 1v1：`the gold gained`；5v5：`The gold gained.`；开悟：`the total gold gained` | dense | 己方经济 |
| Farming | exp | **0.008** | **0.001** | **0.006** | `the experience gained` | dense | 己方经济 |
| Farming | ep rate / Mana | **0.8** | **0.05** | **0.75** | 1v1：`the rate of mana`；5v5：`The rate of mana (**to the fourth power**)` | dense | 己方资源 |
| Farming | **No-op** | **不存在** | **−0.00001** | **不存在** | `Stop and do nothing.` | Dense | **仅 5v5 有**（见 §3.1） |
| Farming | Attack monster | — | **0.1** | — | `Attack monster.` | Sparse | 中立野怪 |
| KDA | death / Death | **−1.0** | **−1** | **−1.0** | `being killed` | sparse | 己方英雄 |
| KDA | kill / Kill | **−0.5** | **1** | **−0.6** | 1v1：`kill an enemy hero`；5v5：`Kill a enemy hero.`；开悟：`killing an enemy hero` | sparse | 敌方英雄 |
| KDA | Assist | — | **1** | — | `Assists.` | Sparse | 队伍 |
| KDA | Tyrant buff | — | **1** | — | `Get buff of killing tyrant, dark tyrant, storm tyrant.` | Sparse | 队伍 |
| KDA | Overlord buff | — | **1.5** | — | `Get buff of killing the overlord.` | Sparse | 队伍 |
| KDA | Expose invisible enemy | — | **0.3** | — | `Get visions of enemy heroes.` | Sparse | 情报 |
| KDA | Last hit | **0.5** | **0.2** | — | 1v1：`last hitting to enemy units`；5v5：`Last hitting an enemy minion.` | sparse | 兵线 |
| Damage | Hurt to hero | — | **0.3** | — | `Attack enemy heroes.` | Sparse | 敌方英雄 |
| Pushing | Attack turrets | — | **1** | — | `Attack turrets.` | **Sparse** | 塔 |
| Pushing | Attack crystal | — | **1** | — | `Attack enemy home base.` | **Sparse** | 基地 |
| Win/Lose | Destroy home base | — | **2.5** | — | `Destroy enemy home base.` | Sparse | 终局 |

**逐项出处（本会话 grep 复核）**：
- 1v1 Table 6：`docs/juewu_research_2026-09-19/juewu_1v1.txt:775–783`（`Table 6: Reward Design` 后 8 行逐字）。
- 5v5 Table 4：`juewu_rl5v5.txt:898–918`（`Table 4: Reward design details.`（:898）+ 表头 `Head Reward Item Weight Type Description`（:899）+ 5 个 Head 与 17 个奖励项，末行 `Win/Lose Related Destroy home base 2.5 Sparse Destroy enemy home base.`（:918））。
- 开悟 Table 5：`hok_arena.txt:679–687`（`Table 5: Reward Design` 后 7 行）。
- 开悟五类划分文字：`hok_arena.txt:639–646`。

### 1.2 合并表暴露的三处**表间冲突**（并列不调和）

| # | 冲突 | 三张表读数 | 本报告处理 |
|---|---|---|---|
| **C-1** | `kill` 的**符号** | 1v1 **−0.5** / 5v5 **+1** / 开悟 **−0.6** | **并列保留**。不可推断「哪个是实际用的」；主报告 §9-U3 已把「为何为负」的机制归因标**未验证**（中文报道的解释非论文原文） |
| **C-2** | `tower hp point` 的**稀疏/密集标注** | 1v1 表标 **sparse**；开悟表标 **dense**；5v5 表无此项（其 Pushing 组 `Attack turrets/crystal` 标 Sparse，而 `hok_arena.txt:645` 的文字又把这组描述为 `which are **dense** rewards`） | **并列保留**；开悟表「Type 列」与「Appendix F 文字」**自相矛盾**（文字说 Pushing 是 dense，表里 `hp_point`/`tower_hp_point` 标 dense 但无 attack turrets 行） |
| **C-3** | 同一语义项的**权重漂移** | money 0.008→0.005→0.006；ep_rate 0.8→0.05→0.75；exp 0.008→0.001→0.006；last hit 0.5→0.2 | **并列保留**；原因**未验证**（无任何来源说明是版本迭代、场景适配还是排版差异，对应主报告 U4） |

> ⚠️ **量纲提示**：三张表的 weight 都是**无单位标量**，但「量纲（Description）」一列揭示**同一名字在不同论文里指不同的量**：1v1 的 `hp point` 是 `health point`（绝对值），5v5 的 `Health point` 是 `health point **to the fourth power**`（四次幂），开悟的 `hp_point` 是 `the **rate** of health point`。**三者不可直接数值对拍**（【纪律】禁跨项目对拍；此处甚至是跨版本）。
> **四阶幂**只在 5v5（P3）出现，两处：`Mana 0.05 Dense The rate of mana (**to the fourth power**)`（`juewu_rl5v5.txt:903`）、`Health point 3 Dense The health point of the hero (**to the fourth power**)`（`:914`）。1v1（`hp point`/`ep rate`）与开悟（`hp_point`/`ep_rate`）的表**都没有**四次幂。

### 1.3 时间/节奏项：三张表**均无**

- 三张表**都没有**显式的「时间惩罚 / 节奏惩罚 / 拖时间罚」项（逐字看表：1v1 8 项、5v5 17 项、开悟 7 项，无一项的描述含时间）。
- 唯一与「时间分段」相关的机制是**观测侧**：开悟把一局按时间划成 **5 段**（`VecCampsWholeInfo | Current Period | Divide game time into 5 periods | 5`，`hok_arena.txt:632`）——这是**特征**，**不是奖励项**。
- 承担远视界信用分配的只有**折扣因子**（见 §4）。

---

## §2 奖励分解与价值头（`V = Σ w_k V_k`）

### 2.1 原文（P3，NeurIPS 2020）——**逐字，本会话复核于 `juewu_rl5v5.txt:231–257`**

> `In order to estimate the value of the ever-changing game state more accurately, we introduce multi-head value (MHV) into MOBA by **decomposing the reward**, which is inspired by the hybrid reward architecture (HRA) used on the Atari game Ms. Pac-Man [32]. Specifically, we design **five reward categories as the five value heads**, as shown in Fig. 1, **based on game expert's knowledge and the accumulative value loss in each head**. These value heads and the reward items contained in each head are:`
> `1) Farming related: gold, experience, mana, attack monster, no-op (not acting); 2) KDA related: kill, death, assist, tyrant buff, overlord buff, expose invisible enemy, last hit; 3) Damage related: health point, hurt to hero; 4) Pushing related: attack turrets, attack enemy home base; 5) Win/lose related: destroy enemy home base.`
> `L_value(θ) = Ê_t[ Σ_head_k (R_t^k − V̂_t^k)² ],   V̂_t = Σ_head_k w_k V̂_t^k,   (2)`
> `where R_t^k and V̂_t^k are **the discounted reward sum and value estimation of the kth head**, respectively. Then, the total value estimation is **the weighted sum of the head value estimates**.`

### 2.2 5 个头 ↔ 具体项映射（**论文给的，不是推断**）

| # | 价值头 | 包含的奖励项（P3 Table 4 分组，与 `:240–241` 文字逐字一致） |
|---|---|---|
| 1 | **Farming Related** | Gold 0.005 / Experience 0.001 / Mana 0.05 / **No-op −0.00001** / Attack monster 0.1 |
| 2 | **KDA Related** | Kill 1 / Death −1 / Assist 1 / Tyrant buff 1 / Overlord buff 1.5 / Expose invisible enemy 0.3 / Last hit 0.2 |
| 3 | **Damage Related** | Health point 3 / Hurt to hero 0.3 |
| 4 | **Pushing Related** | Attack turrets 1 / Attack crystal 1 |
| 5 | **Win/Lose Related** | Destroy home base 2.5 |

出处：`juewu_rl5v5.txt:898–918`（Table 4 的 `Head` 列直接按上述分组印刷）。

### 2.3 `w_k` 数值：**论文未给**（★ 必答项）

- **结论：未给。** P3 只给公式 `V̂_t = Σ_head_k w_k V̂_t^k`（`:252`）与文字 `the total value estimation is the weighted sum of the head value estimates`（`:257`），**通篇没有 `w_k` 的任何数值**。
- 本会话的**证伪式检索**：对全部 6 份抽取 grep `headk` / `head_k` / `wk ` / `weighted sum`，命中 **7 处，全部在 `juewu_rl5v5.txt`（:246/:251/:252/:257/:310/:319 + Table 4 印刷）**——`wk` 的**唯一**一次出现就是公式里的符号本身（`:252`），**没有任何赋值**。
- ⇒ **`w_k` 数值 = 未给 / 未验证**（与主报告 §9-U2 一致）。**禁止**用 Table 4 的**奖励项权重**代替 `w_k`——P3 里这是**两套独立的数**（论文从未说它们相等）。

### 2.4 奖励分解的**第二层**：每头单独回归

`L_value(θ) = Ê_t[Σ_head_k (R_t^k − V̂_t^k)²]` ⇒ **每头一个 MSE 目标、各自回归**，再对头估计做加权和得总价值。即「分解」体现在**损失层（5 个独立回归目标）+ 输出层（加权合成）**两个位置。

### 2.5 5 个头的**选取依据**（唯一可核实的「按损失调奖励」机制）

`based on game expert's knowledge and **the accumulative value loss in each head**`（`:234–235`）——头的划分依据 = **专家知识 + 各头累计价值损失**。论文**没有**给这个过程的形式化算法、阈值或迭代次数。

---

## §3 惩罚与掩码契约

### 3.1 显式惩罚项清单（三张表并集，只有 3 类）

| 惩罚项 | 1v1 (P1 T6) | 5v5 (P3 T4) | 开悟 (P6 T5) | 形态与量级 |
|---|---|---|---|---|
| **Death**（被击杀） | **−1.0** sparse | **−1** Sparse | **−1.0** sparse | 三表一致（−1），**稀疏** |
| **Kill**（击杀） | **−0.5** sparse | **+1** Sparse | **−0.6** sparse | **符号冲突（C-1）**，1v1/开悟为负；**稀疏** |
| **No-op**（不做动作） | **无此项** | **−0.00001** Dense | **无此项** | **仅 5v5 有**；密集；量级 ≈ kill(=1) 的 **1/100000** |

**`No-op −0.00001` 的精确归属（★ 必答）**：
- **只出现在 P3（5v5，NeurIPS 2020）的 Table 4**，位于 **Farming Related 头**内。
- 逐字：`No-op | -0.00001 | Dense | Stop and do nothing.`（`juewu_rl5v5.txt:904`），并在正文五类枚举里被复述：`1) Farming related: gold, experience, mana, attack monster, **no-op (not acting)**`（`:240`）。
- **1v1（P1 Table 6）与开悟（P6 Table 5）都完全没有 no-op 项** —— 本会话对全部抽取 grep `no-op|noop|no op` 共 4 处命中：`juewu_rl5v5.txt:240, :585(Table 另一处), :904` 与 `macrog.txt:454`（后者是 **P5 的基线对比表**列名 `No-op | Built-in bots | Human player | SL | RL-baseline | MGG`，即 P5 把「No-op 行为占比」当成**行为指标**，**不是奖励项**）。
- **为何 1v1/开悟没有**：**公开一手来源没有给出理由**。可核实的只是两个**并存的事实**，不是因果：
  ① 两者都有**塔血密集项 `tower hp point = 10.0`**（1v1/开悟表最高权重）作为天然进攻推力；
  ② 1v1 是**小地图单一目标**、开悟环境论文的实验也以 1v1 为主（`hok_arena.txt:709`（Fig. 12 标题）`the Honor of Kings Arena 1v1 mode`）。
  ⇒ **本报告不写「因为 X 所以没加 no-op」**（主报告 §7.2 N8 把这条列为**不可迁移的乐观假设**）。

### 3.2 「非法动作」的处理：**剔除，不是扣分**（★ 必答）

**三篇论文都找不到「非法动作扣分」的奖励项。** 逐字证据：

**P1（1v1）——掩码的四类**（`juewu_1v1.txt:449–461`）：
> `To improve the training efficiency, an **action mask** is proposed to incorporate the correlations between action elements at the final output layers of the policy based on prior knowledge of experienced human player, which helps prune the exploration of RL. Specifically, our action mask helps eliminate several unreasonable aspects: 1) **physically forbidden areas on map**, e.g., suppose the predicted action is to move towards a direction, which cannot be performed as that direction is occupied by obstacles in the map; 2) **skill or attack availability**, e.g., the predicted action to release a skill within Cool Down time shall be eliminated; 3) **being controlled by enemy hero skill or equipment effects**; 4) **hero-/item-specific restrictions**.`

**P6（开悟）——把这件事做成 API 契约**（`hok_arena.txt:192–197`）：
> `• **legal_action** describes current legal sub-actions with 1 NumPy array. The legal sub-actions incorporates prior knowledge of experienced human players and helps eliminate several unreasonable aspects: 1) skill or attack availability, e.g., the predicted action to release a skill within Cool Down time shall be eliminated; 2) being controlled by enemy hero skill or equipment effects; 3) hero-/item-specific restrictions.`
> `• **sub_action_mask** is a NumPy array describing **dependencies of different Button actions**.`

**P6 的消融结论（★ 必答原文，`hok_arena.txt:711–718`）**：
> `H.2 Legal action mask — ... In Honor of Kings Arena, it provides legal action information to help eliminate unreasonable actions for reference, where researchers can design their own legal actions based on their knowledge. We can use the legal action information to **mask unreasonable actions out during the training process**. The experiment results is shown in Figure 13.`
> `As expected, **without legal action, the agent quickly converges to a local optimum under the large action space, and it is critical to use action mask for better performances.**`
> （Fig. 13 三幅子图：`(a) Reward (b) Hurt per frame (c) Win rate`，`Error bars represent standard deviation`，横轴 `the number of training samples`。）

**P1 的消融（与 P6 的**结论方向一致但收益口径不同**，并列保留）**（`juewu_1v1.txt:700–708`）：
> `We first analyze three components from the model network, including the action mask (AM), target attention (TA) and LSTM we developed. In Table 5, we show the results of a tournament between different "DiRenJie" AI versions using the same amount of training resources. ... We see that using action mask can **largely reduce the training time**, while achieving the same AI ability as the Base (**win rate 50.5%**).`
- ⚠️ **P1 的收益是「省训练时间」，P6 的收益是「避免局部最优 / 性能更好」** —— 两篇的**评价口径不同**（P1 = 同资源下的胜率 50.5% 打平、时间更短；P6 = reward/hurt-per-frame/win-rate 三项曲线）。**并列不调和**。

**`sub_action_mask` 语义的已知边界**：论文只给一句话 `describing dependencies of different Button actions`，**未给维度、取值语义、与 `legal_action` 的对应关系**（对应主报告 §9-U12）。开悟 Table 4 给了动作空间各维度（Button 类、Move X/Z 各 16、Skill X/Z 各 16、Target 类，`hok_arena.txt:648–677`），但**两者的对应关系论文没说明**。

### 3.3 其他约束机制（是否为「惩罚」）

| 候选 | 存在于绝悟？ | 原文证据 | 是奖励项吗 |
|---|---|---|---|
| PPO ratio clip | ✅ | P1 `:464–468`：`the standard PPO algorithm involves a ratio clip ... to penalize extreme changes to the policy` | ❌ 目标函数项，不是 reward |
| **Dual-clip PPO**（绝悟标志算法） | ✅ | P1 `:469–476` 提出；动机 `When Ât < 0 ... the ratio will introduce a big and unbounded variance`；`c > 1` 是下界；**ϵ=0.2, c=3**（P1 `:553–554`、P3 `:389–390`、P6 `:691–692` **三处一致**） | ❌ 目标函数项 |
| KL 锚定（对旧策略/人类策略的显式 KL 惩罚） | ❌ **未找到** | P1/P2/P3/P5/P6 全文无 `KL penalty / KL regularization / KL anchor`；P3 的蒸馏用**交叉熵** `H^×(π_i‖π_θ)`（`juewu_rl5v5.txt:307–324` 区域） | — |
| 规则门禁（满足条件才发奖） | ❌ **未找到** | 无任何此类项 | — |
| 观测层遮蔽 | ✅ | P3 `:232–234`：不可见对手信息 `only applied to the value network during training`；P6 Appendix D 同类 | ❌ 观测层 |

---

## §4 归一化与折扣因子口径（★ 口径冲突并列不调和）

### 4.1 归一化：**三篇均未描述「奖励 / 价值 / 优势」的归一化**

本会话对 6 份抽取逐字 grep `normaliz`，**全部命中仅 8 处，无一处是奖励/价值/优势**：

| 命中位置 | 内容 | 属于 |
|---|---|---|
| `juewu_rl5v5.txt:891–893` | `The way of **feature** normalization is as follows. For continuous features, we use their maximum and minimum values to normalize them into the interval of [0,1], such as health point (HP), mana, speed, etc.` | **观测特征**归一化 |
| `juewu_sl.txt:600, 611, 617` | `Attack Sample Normalization` —— 对同一整个攻击过程**采样相同数量样本** | **IL 数据采样**再平衡 |

⇒ **结论：奖励归一化 / 价值归一化 / 优势归一化（advantage normalization / reward scaling / value scaling / running-mean whitening）= 三篇论文均未描述**。同一次检索也未命中 `advantage normali` / `reward scaling` / `value normali` / `running mean` / `whiten`（0 命中）。
- 唯一与「价值尺度」相邻的可核实事实是 **P3 蒸馏损失里的价值 MSE** `Σ_head_k (V̂_i^k − V̂_θ^k)²`（`juewu_rl5v5.txt:307–324` 区域）——**没有**任何缩放系数被提及。
- ⚠️ **不得**据此反推「绝悟没用归一化」——官方未公开训练细节；只能写「**论文未描述**」。

### 4.2 折扣因子：**0.998 vs 0.997 冲突（并列不调和）**

| 论文 | 折扣因子 | 逐字出处 | 同段其他超参 |
|---|---|---|---|
| **P1（1v1）** | **0.997** | `juewu_1v1.txt:556`：`The discount factor is set as 0.997. For the case of Honor of Kings, this discount is valuing future rewards with a **half-life of about 46 seconds**.` | `ϵ=0.2, c=3`（`:553–554`）；`λ = 0.95 in GAE`（`:559`） |
| **P6（开悟）** | **0.997**（PPO）；**DQN 目标网络 0.98** | `hok_arena.txt:692`：`The discount factor is set as 0.997. ... half-life of about 46 seconds.`；`:695`：`For DQN, the discount factor of target Q-network is set as 0.98.` | `ϵ=0.2, c=3`（`:691`）；`λ = 0.95 in GAE`（`:693`） |
| **P3（5v5）** | **0.998** | `juewu_rl5v5.txt:390`：`The discount factor is set as 0.998.` | `ϵ=0.2, c=3`；`λ = 0.95`（`:390–391`） |

**并列不调和（不做任何归并、不推断「哪个是实际用的」）**：
- **观测到的口径分层**：**1v1 与开悟一致 = 0.997**；**只有 5v5 = 0.998**。⇒ 差异的**划分线**是「5v5 vs 1v1/开悟」。
- ⚠️ **P3 的 0.998 段落里没有 half-life 换算**（P1/P6 才有 `half-life of about 46 seconds`）⇒ 无法用半衰期交叉验证。
- ⚠️ **不得**用 `γ=0.997` 与 `γ=0.998` 的差去对拍任何其他项目（【纪律】禁跨项目对拍）。

### 4.3 GAE λ：三篇**一致**（0.95）

P1 `:559` / P6 `:693` / P3 `:390–391` 均为 `λ = 0.95 to reduce the variance caused by delayed effects`。**这是唯一三篇完全一致的 RL 超参。**

### 4.4 其他跨篇一致/冲突的超参（供对照组参考）

| 超参 | P1 1v1 | P3 5v5 | P6 开悟 | 一致？ |
|---|---|---|---|---|
| ϵ（PPO clip） | 0.2 | 0.2 | 0.2 | ✅ |
| c（Dual-clip 下界） | 3 | 3 | 3 | ✅ |
| GAE λ | 0.95 | 0.95 | 0.95 | ✅ |
| Adam 初始 lr | 0.0001（`juewu_1v1.txt:552`） | 0.0001（`juewu_rl5v5.txt:388–389`） | 0.0001（`hok_arena.txt:691`） | ✅ |
| 折扣因子 | **0.997** | **0.998** | **0.997**（PPO）/ 0.98（DQN 目标网） | ❌ **冲突** |

---

## §5 课程与调度

### 5.1 CSPL 三阶段 + Elo 晋升（P3，`juewu_rl5v5.txt:293–330`）

> `we propose **curriculum self-play learning (CSPL)** to guide MOBA AI learning. CSPL includes three phases, shown in Fig. 2, described as follows. **The rule of advancing to the next phase in CSPL is based on the convergence of Elo scores.**`（`:295–297`）

| 阶段 | 做什么 | 关键数字 | 出处 |
|---|---|---|---|
| **Phase 1** | `start with easy tasks by training **fixed lineups**`（固定阵容自对弈） | 40 英雄分 4 组 ×10；组内需满足 5v5 胜率条件才成组（主报告 §3.3 记「≈50%」，来源 P3 §3.3）；教师模型**参数量约为最终模型的一半** | `:298–305`；Fig. 2（`:283–286`：`Small task with small model`） |
| **Phase 2** | `**multi-teacher policy distillation**`，Phase 1 模型作教师 `π_i` 蒸馏进单一学生 `π_θ`；**监督式**；`student-driven policy distillation` | 损失 = 策略交叉熵 + 价值 MSE（`Σ_i E[Σ_t H^×(π_i‖π_θ) + Σ_head_k (V̂_i^k − V̂_θ^k)²]`） | `:305–326` |
| **Phase 3** | `continued training by **randomly picking lineups** in the hero pool, using the distilled model from Phase 2 for model initialization` | 最终模型 17M vs 教师 9M（主报告 §4.4 记，P3 §4.1） | `:329–330` |
| **晋升判据** | Elo 收敛（**无数值阈值**） | — | `:296–297` |

### 5.2 开悟的课程（另一条线，`hok_arena.txt`）

- **环境/任务课程**：20 英雄 → **20×20 = 400 个 task**（用于暴露泛化缺口，主报告 §4.4 记自 P6 §3.1）。
- **Elo 评估**：`H.4 Evaluation under Elo`（`:728`）—— 论文**建议**用 Elo 评估不同 agent，并给了一个**非传递性**实验：`a first model A was trained against BT. Then, another agent B was trained against a frozen version of agent A ... While B managed to beat A consistently, its performance against built-in AI was not as good as Model A.`（`:728–736`）
- **时间分段**：观测特征把一局按时间分成 **5 段**（`:632`，观测而非奖励）。

### 5.3 **奖励权重退火（annealing）：未找到**（★ 必答，主报告最重要的一条否定性发现）

**本会话独立复核**：对全部 6 份抽取 grep `anneal` / `reward weight` / `weight schedule` / `curriculum weight` / `schedule`，命中**仅 2 处**：

```
hok_arena.txt:680:Reward Weight Type Description     ← 表头
juewu_1v1.txt:776:Reward Weight Type Description     ← 表头
```

⇒ **0 处描述奖励权重随时间/阶段变化**。三张奖励表的权重**都是固定标量**（不随训练阶段、不随 Elo、不随路线）。

- ⇒ **奖励权重退火的一手证据：未找到。**
- ⚠️ 同一条纪律：`anneal` 类做法在 **OpenAI Five** 里是知名技术，但 P1 只说 `The reward design is **inspired** by OpenAI Five's Dota reward`（`juewu_1v1.txt:793–794`）——**只提奖励设计灵感，未提退火**。**不得**由其他项目反推绝悟用了退火（【纪律】禁跨项目对拍）。
- ⚠️ **可核实的三条「多阶段」但都不是奖励权重调度**：
  ① **奖励分解本身**（5 个价值头，P3，§2）；
  ② **头的划分依据含「累计价值损失」**（`based on game expert's knowledge and the accumulative value loss in each head`，`:234–235`）——这是**一次性设计依据**，论文未说是**在线调度**；
  ③ **课程**（CSPL 三阶段，§5.1）——**数据/阵容/模型容量**的调度，**不是奖励权重的调度**。
- **`w_k` 是否有调度：未验证**（`w_k` 数值本身就没给，见 §2.3）。

---

## §6 已知坑 / 失败模式

> 以下**全部为论文作者自己写下的**失败/局限。**没有任何一手来源提到绝悟出现「reward hacking」或「AI 学会卡 bug」**（主报告 §9-U6）。

| # | 坑 | 原文（逐字） | 出处 | 本会话复核 |
|---|---|---|---|---|
| **K-1** | **learning collapse**（英雄池扩张时） | `existing methods by randomly presenting these disordered hero combinations to a learning system can lead to "**learning collapse**" [1], which has been observed from both **OpenAI Five** [2] and **our experiments**. For instance, OpenAI attempted to expand the hero pool up to 25 heroes, resulting in unacceptably slow training and degraded AI performance, even with thousands of GPUs.` | P3 §1 | `juewu_rl5v5.txt:62–67` ✅ |
| **K-2** | **泛化崩溃：胜率恒 0** | `we also noticed that for certain tasks like opponent heroes changed to Peiqinhu/Shangguanwaner in Figure 6, or target heroes changed to Peiqinhu/Shangguanwaner in Figure 7, the winning rate for the Diaochan model is **constantly zero**. This makes it hard to evaluate the performance of different techniques for generalization.` | P6 §6 | `hok_arena.txt:387–390` ✅ |
| **K-3** | **同一策略换对手性能骤降（缺乏迁移性）** | `the performance of the same trained policy **drops dramatically**, as the change of opponent hero differs the testing setting from the training setting, indicating the **lack of transferability** of the policy learned by existing methods.` | P6 §6 | `hok_arena.txt:339–341`（另 `:373–375` 对 target hero 的同类表述） |
| **K-4** | **无 action mask ⇒ 收敛到局部最优** | `without legal action, the agent quickly converges to a **local optimum** under the large action space, and it is critical to use action mask for better performances.` | P6 H.2 | `hok_arena.txt:717–718` ✅ |
| **K-5** | **通用奖励 ⇒ 所有阵容打法趋同** | `In Honor of Kings, the most common strategy for human players is three-lane-strategy ... Therefore, **the general reward function [Ye et al. 2020a] is designed following the common strategy, so agents will perform similar strategies in different lineups and thus cannot perform the most suitable strategy in some special lineups.**` | P5 §4.4 | 见 `macrog.txt`（对应主报告 §6.4） |
| | 同上（另一处表述） | `... the same hand-crafted reward signal and then present similar strategies even in different matches` | P5 §1 | `macrog.txt:39` ✅ |
| | 量化读数 | Table 2 `macro-state entropy`：Built-in bots **0.000** / RL-baseline **0.014** / MGG **0.408** | P5 | 见 `macrog.txt`（`No-op` 基线对比表在 `:454`） |
| **K-6** | **人类微操有噪声 ⇒ 完全模仿会掉性能** | `Intuitively, top players' matches are observed to have high-quality and diverse macro-strategies but **noisy micro-operations**. Thus, **completely imitating human demonstrations, including micro-operations, will worsen the performance of the policy.** However, the macro-strategies from top players' matches can be used to imitate, which can reduce the cost of exploration in training.` | P5 §1（`macrog.txt:189–196`） | ✅ 逐字 |
| **K-7** | **IL 数据分布三个坑** | ① 场景不平衡（Scene Identification + 欠采样）；② **无效移动噪声**：`the player usually executes meaningless move actions`（移动标签改为未来 N 帧位置差，Combat N=5/0.33 s、其他 N=15/1 s）；③ **目标选择样本不平衡**：`Without downsampling, the model will prefer to attack the LDHH hero. However, the prior target normally is an HDLH hero, which is the key to win a local battle.` → `attack sample normalization` | P2 §V-A | `juewu_sl.txt:600/611/617` ✅（三个坑对应主报告 §6.5） |
| **K-8** | **SL 的固有代价** | `one drawback of SL is the requirement of **high-quality human data**. Furthermore, **SL's upper limit is theoretically lower than the RL counterpart**` | P2 §V | 见 `juewu_sl.txt`（对应主报告 §6.5） |
| **K-9** | **稀疏奖励的固有难度（作者主动承认，并邀请外部内在奖励）** | `It is possible that only using the dense reward version of the Honor of Kings will likely resemble the sparsity, which is often seen in previously sparse rewarding benchmarks. Given the sparse-reward nature of this task, **we encourage researchers to develop novel intrinsic reward-based systems, such as curiosity, empowerment, or other signals to augment the external reward signal** provided by the environment.` | P6 Appendix F | `hok_arena.txt:647–651` ✅ |

**补充：惩罚力度的两难（中文受访原话，非论文）**
> `如果罚得太多，英雄就会不敢出战，各种逃窜，甚至掉一丝血就想回城补血；如果罚得太少，英雄又会傻冲，容易被团灭。`
> 来源：<https://cloud.tencent.com.cn/developer/article/2160236>（含王者 AI 团队受访原话；**非腾讯 AI Lab 执笔**）→ **本报告按「二手但含一手原话」标注，不作为论文事实**。

**P5 的缓解手段**（不是根治）：`multi-task training` 与 `student-driven policy distillation`（`improves the performance in all the test tasks`）。

---

## §7 设计意图（≤5 行）+ 迁移性

### 7.1 设计意图（5 行）

1. **奖励是人工设计并手工调参的**，腾讯官方口径为「人工智能的训练，本质上就是奖励设计的过程」；三张奖励表都是**固定权重的功能分组表**，不是学出来的、也没有退火。
2. **核心手法 = 「密集项给方向 + 稀疏项给目标」**：把经济/血量/资源做成 dense 塑形，把塔、击杀、基地做成 sparse 里程碑；4 处**四次幂**（hp、mana）说明**整形是为了拉大梯度差、不是照抄原始量纲**。
3. **把「不该做的事」从奖励里搬出去**：非法/不合理动作不扣分，而是 `legal_action` + `sub_action_mask` **在策略输出层掩码剔除** ⇒ 奖励表**不承担纠错职能**，从而不产生「奖励项互相抵消」。
4. **用「分解」换价值估计的稳定性**：5 个价值头**各自回归**（`Σ(R_t^k − V̂_t^k)²`）再**加权合成**，头按**专家知识 + 累计价值损失**划；这是为了在巨大状态空间里把信用分配**局部化**。
5. **用「课程 + Elo」换可扩展性**：固定阵容 → 蒸馏 → 随机阵容，晋升按 **Elo 收敛**；因为无序英雄组合会导致 **learning collapse**。整条线的**惩罚项只有 3 类**（Death / Kill / No-op），且 **No-op 只有 5v5 有、量级 1e−5 只用于打破平局**。

### 7.2 对我们（Clash Royale 自对弈 PPO、手工 22 键奖励表、critic 已确证沉睡 C3/O2）**可迁**

| # | 可迁项 | 绝悟证据 | 我们能用在哪 | 落地形态（最小风险） |
|---|---|---|---|---|
| **A1** | **「非法动作 = 接口契约」而非奖励惩罚** | P6 `legal_action` + `sub_action_mask` 是 **env 返回的 API 字段**（`hok_arena.txt:192–197`）；P6 H.2 消融：无掩码 ⇒ 局部最优（`:717–718`） | 我们的 `invalid_penalty=0.05`（实测 `src/clasher_new/rl/config.py:58`）**是奖励项**；且已查实四层合法性**互不推导**（C1/C2/C3/C4，见 `docs/illegal_action_layers_2026-09-18.md`）——**根因是没有单一权威的「合法动作」语义源** | 建一层**可复算的合法性产物 + 位数对账**（照 `sub_action_mask` 表达**动作间依赖**），让 22 键奖励表**不再承担纠错职能** |
| **A2** | **多价值头 + 奖励分解（把总价值拆成加权和，损失逐头 MSE）** | P3 `:231–257`：5 头、`Σ_head_k (R_t^k − V̂_t^k)²`、`V̂_t = Σ w_k V̂_t^k` | 我们**已有 22 键手工奖励表** = 现成的「奖励分解」原料；**归并成 4–6 头不改奖励语义 ⇒ 不违【R2】** | **先只做「价值头分组」**（不改奖励、不改权重），再**单独预注册**「头权重 `w_k`」实验；**学绝悟：`w_k` 与奖励项权重是两套数**（P3 明确分开） |
| **A3** | **用「累计价值损失」挑选/体检奖励项** | P3 `:234–235`：5 头划分依据 = `game expert's knowledge` **+ `the accumulative value loss in each head`** | 我们**本来就在记录**逐项贡献（`docs/readout_et_solo100k/item_health.json`）；且已实测过 `τ == kill_credit` 的**重复计数**缺陷（`docs/s2_instrument_2026-09-18.md` §7） | 形式化成**奖励项体检指标**：某价值头损失**长期不降 + 该头参数不动** ⇒ 该奖励项是噪声或重复计数。**⚠️ 这条正好对齐我们的 C3/O2（critic 沉睡 4 个头的参数）** —— 门禁用【R14】的 `‖ΔW‖/‖W‖`，不用 EV |

> **另**：P1 的**零和**结构（`one's mean reward is subtracted from that of the opponent`，`juewu_1v1.txt:794–795`）与我们的对称化要求同源（主报告 A5）；P2 的**场景分段 + 每类等量采样**（`juewu_sl.txt:600–617`）对我们「罕见状态 0 采样」的病根**方向吻合**（主报告 A3）。两条优先级低于 A1–A3，此处不展开。

### 7.3 **不可迁**

| # | 不可迁项 | 绝悟的前提 | 为什么不适用 |
|---|---|---|---|
| **N1** | **`No-op` 惩罚（照抄 −0.00001）** | **只有 5v5（P3）有**；而同表 5v5 **本身没有塔血密集项**（`tower_hp_point = 10.0` 只存在于 1v1 与开悟表），它的推进信号只有 `Attack turrets 1` / `Attack crystal 1` / `Destroy home base 2.5` 三类**稀疏**项（`juewu_rl5v5.txt:916–918`） | 我们实测：**「放弃一张买得起的牌」在 75,891 帧里发生 0 次**、**88–91% 的帧买不起任何手牌**（`docs/pass_prob_2026-09-18.md`）⇒ 我们的「不作为」是**圣水经济/机制**造成的，**不是奖励缺项**；且已实测把 stop 概率抬高能把「圣水≥6 帧」从 0% 抬到 48%，但**每局回报从 −4.46 掉到 −41.5**（`docs/exploration_bias_pass_2026-09-18.md`）⇒ **照抄 = 误治病灶** |
| **N2** | **一次幂/四次幂的奖励量纲照搬** | 三张表**同名的量纲都不同**（§1.2 量纲提示）：`hp point` 在 1v1 是绝对值、5v5 是**四次幂**、开悟是**rate** | 我们**已有 22 键 + 汇率常量单一源**（【R7】）。跨项目/跨版本照抄权重 = **用别的游戏的状态尺度标定我们自己的奖励**，且违【R15】（阈值须在本实验量纲上标定） |
| **N3** | **手工调奖励的「千次迭代」工作流** | 中文一手口径：`往往奖励值变化一点点，就会导致训练出来的英雄性格迥异。所以，需要技术宅们蹲在电脑前反复调整实验` | 我们的分辨率不支持「小步改奖励、快跑对比」：20k 单跑只看大效应（【R5】）、**同配置重跑单点差 0.60**（`docs/et_solo100k_judgment_2026-09-18.md` §11.13.11）、判据阈值不许跨实验照抄（【R15】）/ 不许单次观测标定（【R16】） |
| **N4** | **`w_k` 的数值照搬** | **论文根本没给 `w_k`**（§2.3，本会话证伪式检索 0 命中） | 没有数字可搬；且我们连「头权重」这个概念都还没落地（A2 只是候选） |
| **N5** | **KL 锚定 / 人类策略锚定** | 绝悟**没有** KL 惩罚项（§3.3）；P3 的蒸馏是**交叉熵**（教师=RL 自对弈模型，**不是人类**） | 我们**无人类数据**、无 reward model；照搬「锚定」**方向与病根相反**（详见 `docs/il_reward_reference_analysis_2026-09-19.md` 的 I8 判定） |
| **N6** | **规模前提（分布式 actor-learner / 600k CPU + 1064 GPU / 单英雄一模型 / MCTS）** | P1：**600,000 CPU 核 + 1,064 GPU**（单英雄 48×P40 + 18,000 核，`juewu_1v1.txt:527–533`）；P3：**32 万 GPU/3.5 万 CPU**（主报告 §N1）；P2：**一英雄一模型**；P3 的 drafting 用 MCTS | 我们**单机 993K 参数、一模型全牌、无 MCTS**（【R11】明确不上训练时 MCTS）。绝悟的 off-policy 适应 / Dual-clip / 蒸馏**是为巨量 off-policy batch 设计的** |

---

## §8 来源清单（URL + 章节/表号 + 取数时间）

**全部取数时间：2026-09-19 04:35–04:50 CST（UTC+8）**；**一级证据 = 本仓库内六份论文全文抽取（只读 grep）**，URL 为对应一手论文。

| # | 一手来源 | arXiv / 出版 | 本地抽取（行号区间 = 本会话复核位置） | 被本文引用的章节/表号 |
|---|---|---|---|---|
| **P1** | Mastering Complex Control in MOBA Games with Deep RL（1v1） | <https://arxiv.org/pdf/1912.09729v3>；AAAI-2020 <https://ojs.aaai.org/index.php/AAAI/article/view/6144> | `docs/juewu_research_2026-09-19/juewu_1v1.txt`（sha256 前缀 `5f1a28430e9160eb`）<br>`:775–783`（Table 6）/ `:449–461`（action mask 四类）/ `:700–708`（AM/TA/LSTM 消融）/ `:466–475`（Dual-clip）/ `:548–559`（系统+超参） | **Table 6**（奖励）、Table 5（消融）、Fig. 3（Dual-clip）、System Setup |
| **P2** | JueWu-SL（监督学习） | <https://arxiv.org/pdf/2011.12582v1> | `juewu_sl.txt`（`d50a988b5a1b31a8`）`:600/611/617` | §V-A（数据）、Attack Sample Normalization |
| **P3** | Towards Playing Full MOBA Games with Deep RL（5v5） | <https://arxiv.org/pdf/2011.12692v4>；NeurIPS 2020 <https://proceedings.neurips.cc/paper/2020/file/06d5ae105ea1bea4d800bc96491876e9-Paper.pdf> | `juewu_rl5v5.txt`（`ceaf3530a1bfa107`）<br>**:898–918（Table 4）/ :231–257（MHV+Eq.2）/ :293–330（CSPL）/ :62–67（collapse）/ :388–391（超参）/ :891–893（特征归一化）** | **Table 4**（奖励+5 头）、**Eq. 2**（`V̂_t = Σ w_k V̂_t^k`）、Fig. 2（CSPL） |
| **P5** | Learning Diverse Policies in MOBA Games via Macro-Goals | <https://arxiv.org/pdf/2110.14221.pdf>；NeurIPS 2021 | `macrog.txt`（`9a1dc08505a4132c`）`:39`、`:145–170`、`:188–196`、`:454` | §1 / §4.4（通用奖励偏置）、Table 2（macro-state entropy）、人类微操噪声 |
| **P6** | Honor of Kings Arena（开悟官方环境） | <https://arxiv.org/pdf/2209.08483v3>；NeurIPS 2022 35:11881-11892 | `hok_arena.txt`（`f5044fb32ebbb20b`）<br>**:679–687（Table 5）/ :192–197（legal_action/sub_action_mask）/ :711–718（H.2 消融）/ :639–651（Appendix F 五类+稀疏邀请）/ :388–390（胜率恒 0）/ :691–695（超参）** | **Table 5**（奖励）、Table 4（动作空间）、**H.2**、H.4（Elo）、Appendix F |
| **P7** | Hierarchical Macro Strategy Model（HMS，前身） | <https://arxiv.org/pdf/1812.07887.pdf> | `hms.txt`（`6e08537820d2357c`） | 仅说明「稀疏+延迟奖励」动机（未给出奖励表） |

**二手来源（本文仅用于标注「非论文原话」的行）**

| 内容 | URL | 取数时间 | 用途限定 |
|---|---|---|---|
| 中文深度报道（含**腾讯王者 AI 团队受访原话**，非腾讯 AI Lab 执笔） | <https://cloud.tencent.com.cn/developer/article/2160236> | 主报告取数 2026-09-19 03:30–03:45；**本会话未重新联网抓取，仅引用主报告转录** | 仅用于「惩罚力度两难」引文与「kill 为负」的**外部解释**；**不当作论文事实** |

**本项目（Clash Royale）参照物**（仅用于 §7 的迁移性判断，**不参与绝悟数字对拍**）

| 对象 | 路径 | 本会话复核 |
|---|---|---|
| 奖励项体检读数 | `docs/readout_et_solo100k/item_health.json` | 未读（引用主报告/台账） |
| `invalid_penalty=0.05` | `src/clasher_new/rl/config.py:58` | ✅ 已 grep 确认存在 |
| 22 键奖励表 | `src/clasher_new/rl/config.py:37`（`DEFAULT_REWARD = {`） | ✅ 已 grep 确认存在 |

---

## §9 未验证 / 未能核实项（**禁止当作事实引用**）

| # | 未验证项 | 状态与原因 |
|---|---|---|
| **U1** | **`w_k`（5 个价值头权重）的数值** | **论文未给**。P3 只有符号（`:252`）；本会话证伪式检索 `headk`/`wk ` **7 处全部无赋值** ⇒ 主报告 U2 复核通过 |
| **U2** | **奖励权重退火是否存在** | **未找到直接证据**（本会话 grep `anneal`/`schedule`/`reward weight` 命中**仅 2 处表头**）⇒ 不能断言用了，也不能断言没用 |
| **U3** | **`w_k` 是否有阶段调度** | **未验证**（`w_k` 数值本身缺失） |
| **U4** | **`kill = −0.5 / −0.6` 为负的机制归因** | **字面值已验证**（`juewu_1v1.txt:782`、`hok_arena.txt:686` 逐字）；**原因为负 = 未验证**（只存在于中文报道） |
| **U5** | **三张表权重差异的原因**（money 0.008→0.005→0.006 / ep_rate 0.8→0.05→0.75 / exp 0.008→0.001→0.006） | **未验证**：无任何来源说明是版本迭代、场景适配还是排版差异 |
| **U6** | **`γ = 0.998`（P3）vs `0.997`（P1/P6）的成因** | **并列不调和**：三处均为论文原文逐字。**不做归并、不推断哪个是「实际用的」** |
| **U7** | **奖励 / 价值 / 优势的归一化** | **论文均未描述**（grep `normaliz` 8 处全部是**观测特征**或 **IL 采样**）⇒ 只能写「论文未描述」，**不得**写「绝悟不用归一化」 |
| **U8** | **`sub_action_mask` 的确切维度与取值语义** | **未验证**：P6 只有一句话 `describing dependencies of different Button actions`（`:197`）；与 `legal_action` 的对应关系论文未说明 |
| **U9** | **三张表 `tower hp point` 的 sparse/dense 标注冲突成因** | **未验证**（1v1 表标 sparse、开悟表标 dense；开悟 Appendix F 文字又把 Pushing 组描述为 dense，与自家表冲突） |
| **U10** | **`hok_env` 源码中奖惩/合法性的实现细节** | **未能取到**（主报告记：本会话 `raw.githubusercontent.com` SSL 失败、`git clone` 133 s 超时）⇒ 只能从 P6 的 API 文档与论文文字确认接口契约，**未读代码** |
| **U11** | **CSPL「Elo 收敛」的数值阈值 / 判据** | **未验证**：P3 只写 `based on the convergence of Elo scores`（`:296–297`），**无阈值、无窗口、无判据** |
| **U12** | **`No-op −0.00001` 为何只在 5v5 有** | **未验证**：可核实的只有「1v1/开悟表无 no-op」「两者都有塔血密集项 10.0」「1v1 地图小目标单一」这三个**并存事实**；**因果未证实** |
| **U13** | **绝悟是否出现 reward hacking / 卡 bug** | **未找到**：六篇论文与官方页面均无此类记载 |
| **U14** | **P1 消融「win rate 50.5%」是指「AM 版 vs Base 版打平」还是「AM 版对 Base 版胜率 50.5%」** | **部分验证**：原文 `using action mask can largely reduce the training time, while achieving the same AI ability as the Base (win rate 50.5%)`（`:706–708`）⇒ 字面是「与 Base 同能力 = 50.5% 胜率」；**具体对局配置（局数、是否同资源、置信区间）论文未给** |
| **U15** | **`docs/juewu_research_2026-09-19.md` 的 sha256 前缀标注格式** | **观察到的差异（非数字冲突）**：主报告 §8 写 16 位、§8 表内「留档」行写「sha256 见 §8 表内前缀」；本会话复算 6 个抽取文件的前 16 位**与主报告 §8 逐字一致** ⇒ ✅ 通过 |
| **U16** | **中文受访报道的联网复核** | **本会话未重新抓取**（引用主报告转录）；且主报告本身把该域名标为「非腾讯 AI Lab 执笔」⇒ **不当作论文事实** |

---

## 附：本会话可复现命令（供交叉验证）

```bash
cd /mnt/e/clash-royale-simulator-main/docs/juewu_research_2026-09-19
sha256sum *.txt | cut -c1-16                      # 6 个哈希，与主报告 §8 一致
grep -an -i "no-op\|noop\|no op" *.txt            # 4 处：rl5v5:240/585/904 + macrog:454
sed -n '898,915p' juewu_rl5v5.txt                 # P3 Table 4（5 头 + 17 项）
sed -n '775,783p' juewu_1v1.txt                   # P1 Table 6（8 项）
sed -n '679,687p' hok_arena.txt                   # P6 Table 5（7 项）
sed -n '231,257p' juewu_rl5v5.txt                 # MHV + Eq.2（w_k 只有符号、无数值）
grep -an -i "anneal\|reward weight\|schedule" *.txt   # 仅 2 处：表头
grep -an "0\.998\|0\.997" *.txt                   # γ 冲突：rl5v5 0.998 vs 1v1/hok 0.997
grep -an -i "normaliz" *.txt                      # 8 处：全是特征/IL 采样，无奖励/价值/优势
sed -n '711,718p' hok_arena.txt                   # legal action mask 消融原文
```
