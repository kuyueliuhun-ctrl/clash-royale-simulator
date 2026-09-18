# OpenAI Five vs AlphaStar —— RL 阶段奖惩设计统一抽取表

> **取数时间**：2026-09-19（UTC 2026-09-18；本地 `date -u` = `2026-09-18T20:39Z`）
> **对象 A**：OpenAI Five（Dota 2）—— arXiv:1912.06680v1《Dota 2 with Large Scale Deep Reinforcement Learning》，2019-12-13，**仅此一版**
> **对象 B**：AlphaStar（StarCraft II）—— Vinyals et al., *Nature* **575**, 350–354 (2019)，DOI `10.1038/s41586-019-1724-z`
> **任务**：把两者的 RL 阶段奖惩设计抽成同一张字段表，供跨项目横向对比。
>
> **本文件的三条硬纪律（写入正文，不只在脚注）**
> 1. **禁止跨项目数字对拍。** 本文件任何数字**只用于标识与内部一致性论证**，不得用于标定第三项目（含本仓）。
> 2. **盘上报告标「未验证」的项，本文件一律保持未验证。** 不因「抽成表」而升级其证据等级。
> 3. **不得把 IL 阶段的事写成 RL 阶段。** AlphaStar 的 IL 与 RL 是**两个阶段**；涉及跨阶段的项，本文件设独立字段 **⑧ 阶段归属** 显式标注，不混入 ①–⑦。
>
> **证据等级标注图例**（沿用盘上两份报告的约定，不新建体系）
> `[一手-论文原文]` 论文正文/附录源码逐字 ・ `[一手-官方代码]` 官方开源代码 ・
> `[复现件-OA论文]` 第三方复现/同作者群后续论文 ・ `[二手-分析]` 博客等分析 ・ `[未验证]` 本次未取证 ・
> `[本地-A]` 本仓实测（我们项目）
>
> **★ 统一 schema（五份对象报告共用：字段名 / 顺序 / 单位口径一致 — 2026-09-19 编排者补，仅加声明、不改正文）**
> **字段集 ①–⑧**：① 奖励项清单（项名 / 权重 / 量纲 / 作用域 / 是否零和）・② 惩罚项（显式 / 隐式 / 用约束或掩码替代）・③ 塑形与调度（是否 PBRS？是否退火？局内时间权重？）・④ 归一化三问（奖励 / 价值 / 优势）・⑤ value head 设计・⑥ 已知坑（带原文句；二手必须标注）・⑦ 一手出处（URL / `file:line` + 取数时间）・⑧ 阶段归属（IL / RL / 跨阶段接口）
> **本报告落点**：①=§1.1①/§2.1① ・ ②=§1.1②/§2.1② ・ ③=§1.1③/§2.1③ ・ ④=§1.1④/§2.1④ ・ ⑤=§1.1⑤/§2.1⑤ ・ ⑥=§1.1⑥/§2.1⑥ ・ ⑦=§1.1⑦/§2.1⑦ ・ ⑧=§2.1⑧（本文件为 AlphaStar 新设字段）
> **统一单位口径**：权重一律**按来源原样记录**并显式标注量纲（分母单位）；无单位者写「无单位」；**同名项若量纲不同则并列不调和**，不折算、不换算、不对拍。
> **证据等级图例**：见上（本文件已在顶部写入，未新建体系）。
> **三条硬纪律**：见上 1./2./3.（本文件顶部已写入正文，此处不重复）。

---

## §0 一句话结论

**OpenAI Five 走「密集手写塑形 + 零和化 + 队内共享系数」路线（21 项权重 + 3 个结构性组件，奖励函数一次成型、从不退火）；AlphaStar 走「终局二值 win/loss 为唯一环境奖励 + 由人类统计量 z 导出的软塑形 + KL 约束」路线。两者的「惩罚」都不以显式惩罚项为主：Five 的负号项是「计价」（与正项同一套数值体系），AlphaStar 则用 KL 与掩码替代惩罚。对我们（993K 参数、单机、22/16 键手工表、无人类数据、决策帧 step）而言，可迁的是 Five 的「结构性不变量」（零和、量纲、局内时间权重、多成分单头 value、奖励标准化），不可迁的是两者的「规模型机制」（60k–196k 样本批、league 规模、人类数据锚、高维距离型塑形）。**

> ⚠️ 上述结论中，AlphaStar 一侧的措辞**受限于 Nature 正文未取回**（见 §2 与 §4）；凡标 `[未验证]` 者不得被后续文档当作论文事实引用。

---

## §1 OpenAI Five（Dota 2）

### 1.1 统一字段表

#### ① 奖励项清单（项名 / 权重 / 量纲 / team|solo / 是否零和）

一手来源：**Appendix G「Reward Weights」的 Table 5（Shaped Reward Weights），21 行数据**。
- 论文原文通道：https://arxiv.org/html/1912.06680v1 （2026-09-19 复核取回 493,886 B）
- **LaTeX 源码通道（本次判定主依据）**：https://arxiv.org/e-print/1912.06680v1 （2026-09-19 复核重取，**8,831,567 B**，与盘上审计记录同尺寸）→ `section-rewards.tex:7-28`（表头 `:7`，数据行 `:8-28`，表注 `:30-34`，表标题 `:36`）

| # | 项名 | 权重 | 量纲（论文明确者标出） | team \| solo | 是否零和 |
|---|---|---|---|---|---|
| 1 | Win | 5 | 终局 1 次 | Team | 是（构造为全局零和）|
| 2 | Hero Death | −1 | 每次死亡 | Solo | 是（同上）|
| 3 | Courier Death | −2 | 每次 | Team | 是 |
| 4 | XP Gained | 0.002 | 每单位经验 | Solo | 是 |
| 5 | Gold Gained | 0.006 | 每单位金币（花掉/失去**不**回收）| Solo | 是 |
| 6 | Gold Spent | 0.0006 | 每单位金币（不通过信使买装备）| Solo | 是 |
| 7 | Health Changed | 2 | **占英雄最大生命的比例** | Solo | 是 |
| 8 | Mana Changed | 0.75 | **占英雄最大法力的比例** | Solo | 是 |
| 9 | Killed Hero | −0.6 | 每次击杀敌方英雄（**抵减**金/经验奖励）| Solo | 是 |
| 10 | Last Hit | −0.16 | 每次（使 last hit 合计 ≈0.4）| Solo | 是 |
| 11 | Deny | 0.15 | 每次 | Solo | 是 |
| 12 | Gained Aegis | 5 | 每次 | Team | 是 |
| 13 | Ancient HP Change | 5 | **占远古最大生命的比例** | Team | 是 |
| 14 | Megas Unlocked | 4 | 一次性 | Team | 是 |
| 15 | T1 Tower | 2.25 | 见注 `*` | Team | 是 |
| 16 | T2 Tower | 3 | 见注 `*` | Team | 是 |
| 17 | T3 Tower | 4.5 | 见注 `*` | Team | 是 |
| 18 | T4 Tower | 2.25 | 见注 `*` | Team | 是 |
| 19 | Shrine | 2.25 | 见注 `*` | Team | 是 |
| 20 | Barracks | 6 | 见注 `*` | Team | 是 |
| 21 | Lane Assign | −0.15 | **每秒**在错误分路 | Solo | 是 |

**表注（`section-rewards.tex:31-34`，逐字）**
- `[*]` 建筑：**2/3 的奖励随建筑掉血线性累计，1/3 在建筑死亡时一次性给付**。
- `[$\ddagger$]` 英雄血量按四次插值在 0（死）与 1（满血）之间：满血占比 $x$ 时价值 $(x + 1 - (1-x)^4)/2$。
  原文追加：**"This function was not tuned; it was set once and then untouched for the duration of the project."**
- `[$\dagger$]` Lane Assign 见 lane assignment 小节。

> **本文件对盘上审计的一处措辞修正（不改其数字结论）**：盘上报告把 Table 5 记为三列「Name | Reward | Heroes」；
> LaTeX 源码表头实为**四列** `Name & Reward & Heroes & Description`（`section-rewards.tex:7`）。
> 21 行数据与全部权重值**逐行一致**，故该修正**只涉及列数描述，不涉及任何数值**。

**除表内 21 项权重外，奖励函数还有 3 个组成部分**（`section-rewards.tex:48-76`，逐字见 1.1 ③）。

**规模性事实（标识用，不作对拍）**：combined policy + value network **158,502,815 参数**（`section-network-architecture.tex:31`，一手）。

#### ② 惩罚项（显式 / 隐式 / 用约束或掩码替代）

| 类别 | 内容 | 证据 |
|---|---|---|
| **显式惩罚（负号权重项）** | Hero Death −1（Solo）、Courier Death −2（Team）、Killed Hero −0.6（Solo，明确语义是**抵减**金/经验奖励）、Last Hit −0.16（Solo，明确语义是把 last hit 总奖励**压到 ≈0.4**）、Lane Assign −0.15（Solo，每秒错误分路）| `section-rewards.tex:9,10,16,17,28` |
| **隐式惩罚（不产生负奖励，但动作被吞掉）** | **非法参数组合被判为 no-op**，论文用「no-op」而非惩罚 | `section-action-space.tex:41`：*"some parameter combinations are invalid and those actions are treated as no-ops."*；另 `:26`：*"attempting an action with an invalid target results in a noop."* |
| **用掩码替代惩罚** | **梯度侧掩码**（不是奖励侧）：被忽略的参数输出在优化时被 mask 掉 | `section-action-space.tex:10`：*"...some of them are read and others ignored (when optimizing, we mask out the ignored ones since their gradients would be pure noise)."* |
| **用固定脚本序列替代惩罚** | 技能加点与出装走**固定 schedule**，不交给模型 → 无须为「乱加点」设惩罚 | `section-hyperparams` 之外见 Appendix F.1：*"in evaluation games we follow a fixed schedule (improve ability X at level 1, then Y at level 2, then Z at level 3, etc)"*；*"we follow a fixed schedule (first build X, then Y, then Z, etc)"*（盘上审计 §2 已记录，本次未重取） |
| **⚠️ 未找到项** | 全文无 `illegal` 一词命中；**未发现「非法动作惩罚」这一独立机制** | 本次源码穷举 `illegal` = 0 命中 |

> **判定**：Five 的「惩罚」**不是约束体系**，而是**同一套零和计价里的负号项**。真正起约束作用的是**掩码（梯度侧）+ 固定脚本序列**，而非奖励惩罚。

#### ③ 塑形与调度（是否 PBRS？是否退火？局内时间权重？）

| 问题 | 结论 | 原文 / 出处 |
|---|---|---|
| **是否 PBRS？** | **不是严格 PBRS**，但论文**自称「松散地仿照」势函数塑形**，并**明确声明其理论保证不适用** | `section-rewards.tex:42`（逐字）：*"Our shaped reward is modeled loosely after potential-based shaping functions [Ng99policyinvariance], though the guarantees therein do not apply here."* |
| **是否退火（奖励权重随训练变化）？** | **没有。** 奖励函数在项目开始时**一次性构造**；健康塑形函数「set once and untouched for the duration of the project」 | `ocean-paper.tex:247-249`（逐字）：*"We constructed the reward function once at the start of the project... Although we made minor tweaks when game versions changed, we found that our initial choice of what to reward worked fairly well."*；表注见 `section-rewards.tex:33` |
| **⚠️「奖励塑形退火」传说的处置** | 该说法在 **arXiv:1912.06680v1 一手文本中未获任何支持**（`anneal` 家族 5 处全部指超参/环境/动作的渐进引入）。**本文件不据此断言「Five 没有退火」**——只允许写「在该版一手文本中未发现奖励权重退火」 | 盘上审计 §2 穷举表；本次**近义词扩检**（见下）|
| **本次新增：近义词穷举（闭合盘上审计 U7 的一部分）** | 对 `ramp` / `decay` / `coefficient schedule` / `warm-up` / `curriculum` / `schedul` / `linear change` / `gradually` 全源码穷举，**与「reward weight 调度」同时命中的行 = 1 处且是局内衰减** | `section-rewards.tex:55`（局内指数衰减）；另 `section-hyperparams.tex:5` 是**注释行**（`%We modify ... the reward decay factor`，未生效），`section-substudies.tex:309` 亦是注释行 |
| **局内时间权重？** | **有。** 所有奖励**除 win/loss 外**乘指数衰减因子，游戏时间 $T$：$\rho_i \leftarrow \rho_i \times 0.6^{(T/10\,\text{min})}$。论文给的理由是「**后期天然产出更多奖励**，不修正则学习只关注后期、忽略前期」 | `section-rewards.tex:52-58`（逐字）|
| **队内共享系数（team spirit）调度** | **有，且方向是「升到 1」而非退火到 0**：OpenAI Five 列 `0.3 → 1.0`（Rerun 列 `0.3 → 0.8`，Baseline 列 `0.3`）。表注：变化被逐步施加于 1–2 天，对应数千次迭代 | `section-hyperparams.tex:69`（Table「Hyperparameters」）、`:40`（表注）|
| **队内共享的定义式** | $r_i = (1-\tau)\rho_i + \tau\overline{\rho}$，$\overline{\rho}$ 为 $\rho$ 的均值；$\tau=0$ 各自为战，$\tau=1$ 五人平分 | `section-rewards.tex:65-72`（逐字）|
| **零和构造** | 从每个英雄奖励中**减去敌方奖励的均值** | `section-rewards.tex:51`（逐字）|

**team spirit 的设计动机（逐字，`section-rewards.tex:74`）**：*"Ultimately we care about optimizing for team spirit $\tau=1$... However we find that lower team spirit reduces gradient variance in early training, ensuring that agents receive clearer reward for advancing their mechanical and tactical ability to participate in fights individually."*

**调度的其他项（非奖励项，列出以免混淆）**：Entropy coefficient `0.01 → 0.001`（作用于 **PPO loss**，`section-exploration.tex:12` 定义为 $cS[\pi_\theta](s_t)$），GAE Horizon `180s → 360s`，Learning Rate `5e-5 → 5e-6`（`section-hyperparams.tex:32,30,34`）。

#### ④ 归一化三问

| 问 | 结论 | 证据 |
|---|---|---|
| **奖励归一化** | **有**：用**运行标准差的估计**归一化奖励；**价值损失权重在归一化之后施加** | `section-hyperparams.tex:84`（表注 c，逐字）：*"We normalize rewards using a running estimate of the standard deviation, and the value loss weight is applied post-normalization."* |
| **价值归一化** | **未发现**论文给出「value/return 单独归一化」的机制。可核实的只有**价值损失权重** `1.0`（Rerun）/ `0.25 ↔ 1.0`（OpenAI Five）/ `1.0`（Baseline），施加于归一化后 | `section-hyperparams.tex:73,84` |
| **优势归一化** | **未验证**：论文只说用 GAE（$\lambda=0.95$）作方差缩减，**未描述 advantage 归一化步骤** | `ocean-paper.tex:259`、`section-hyperparams.tex:71`（$\lambda$）；无 advantage normalization 原文 |
| **观测归一化（旁证，非奖励）** | 所有浮点观测归一化：维护历史均值/标准差，减去均值除以标准差，结果 clip 到 $(-5,5)$ | `section-observation-space.tex:210`（逐字）|

#### ⑤ value head 设计

| 问 | 结论 | 证据 |
|---|---|---|
| 单头 / 多头 / 分解？ | **单头（single, non-decomposed）**，且**与策略共享 LSTM 与梯度** | `section-network-architecture.tex:48-49`（逐字）：*"In addition to the action logits, the value function is computed as another linear projection of the LSTM state. Thus our value function and action policy share a network and share gradients."* |
| 结构 | 中央共享 **单层大 LSTM** → 分别经**独立全连接层**产出策略与价值 | `ocean-paper.tex:260`（逐字）：*"We train a network with a central, shared LSTM block, that feeds into separate fully connected layers producing policy and value function outputs."* |
| LSTM 规模 | **4096**（OpenAI Five 列 `2048 → 4096`；Rerun/Baseline 列 4096）| `section-hyperparams.tex:67` |
| 价值损失权重 | `1.0`（Rerun）/ `0.25 ↔ 1.0`（OpenAI Five）/ `1.0`（Baseline）| `section-hyperparams.tex:73` |
| **反例（不是架构，但是关键设计选择）** | **无反向价值分解**：无 multi-head value / 无 reward decomposition（与绝悟 P3 的 5 价值头是相反设计；此处仅作方向对照，**不比较数字**）| 源码全文无 value head 分解痕迹 |

#### ⑥ 已知坑（带原文句）

| # | 坑 | 原文句（≤2 行，逐字） | 出处 |
|---|---|---|---|
| 1 | **塑形奖励与胜率脱钩（reward–objective 不一致）**：一个 128 参数的嵌入被置零后**胜率显著提高（约 +55% 胜率）**，而**塑形奖励几乎没变** ⇒ 优化器找不到该方向 | *"It turned out that replacing a certain set of 128 learned parameters in the model with zero increased the model's performance significantly (about 55\% winrate after versus before)."* / *"We believe that the optimizers were unable to find this direction for improvement because although the win rate was higher, the shaped reward ... was approximately the same."* | `section-bloopers.tex:25-26` |
| 2 | **奖励方差过大 → 价值函数学不可靠**（Divine Rapier 负反馈循环）| *"it is possible that the reward collected in a game increases in variance, thereby preventing OpenAI Five from learning a reliable value function."*（并观察到 *"a decrease in episodic reward and TrueSkill"*）| `section-bloopers.tex:55-57` |
| 3 | **探索的时序脆弱性**（长而特定的动作序列越难探索）| *"If a long and very specific series of actions is necessary ... and any deviation from that sequence will result in negative advantage, then the longer this series, the less likely is agent to explore this skill"* | `section-exploration.tex:94` |
| 4 | **稀疏奖励（关掉部分权重）样本效率大幅下降**（一次性消融，**不是退火**）| *"the model learned to play well enough to beat a hand-coded scripted agent consistently, though with a large penalty to sample efficiency relative to the shaped reward baseline."* | `section-rewards.tex:88` |
| 5 | **手调超参不可靠**（人类直觉不是设超参的好方式）| *"we ultimately believe that human intuition, especially under time pressure, is not the best way to set hyperparameters."* | `section-bloopers.tex:8` |
| 6 | **零和构造的代价**：team spirit 的存在正是为了处理「部分奖励可能给不同英雄的好动作**加方差**」 | *"they may backfire and in fact add more variance if an agent receives reward when a \emph{different} agent takes a good action."* | `section-rewards.tex:62` |
| 7 | **被外部利用**（OpenAI Five Arena，2019-04-18~21）：有一支队伍靠利用「对隐身英雄理解不足」连胜 10 场 | *"One team won 10 games in a row, by exploiting a weakness in OpenAI Five's understanding of invisible heroes."* | 论文 **Appendix: Human Evaluation**（HTML 通道 https://arxiv.org/html/1912.06680v1 ）。⚠️ **本次复核发现该句在 LaTeX 源码中处于注释状态**（`section-human-evaluation.tex:68-79` 全段以 `%` 开头），而盘上审计已证 **HTML 通道含 LaTeX 源码中被注释的材料**（如 `anneal`）⇒ 故本行以 **HTML 通道**为出处，**不**引用 LaTeX 行号。另有**未被注释**的对应句：*"To explore whether \openaifive could be consistently exploited by creative or out-of-distribution play, we ran {\it OpenAI Five Arena}, in which we opened \openaifive to the public for competitive online games from April 18-21, 2019."*（`ocean-paper.tex:355`）|

> **注（坑 1 的量级）**：该 55% 是**论文自己给出的观测**（一手），但它是**一次特定手术失败场景**的读数，**不得**当作「塑形奖励普遍无用」的通用结论，也**不得**与任何其他项目的胜率对拍。

#### ⑦ 一手出处

| 内容 | 出处 |
|---|---|
| Table 5（21 行权重）+ 表注 | 论文 **Appendix G**「Reward Weights」 / LaTeX `section-rewards.tex:4-38` |
| 3 个结构组件（零和 / 局内时间权重 / team spirit） | 论文 **Appendix G** / `section-rewards.tex:48-76` |
| PBRS 声明 | 论文 **Appendix G** / `section-rewards.tex:42` |
| 「奖励函数一次构造」 | 论文 **§3.2 Optimizing the Policy** / `ocean-paper.tex:247-249` |
| 超参表（team spirit / entropy / horizon / lr / value loss weight / 表注 c）| 论文 **Appendix C** / `section-hyperparams.tex:52-91` |
| value 头与共享梯度 | 论文 **Appendix: Neural Network Architecture** / `section-network-architecture.tex:30-49` |
| 优化算法与共享 LSTM | 论文 **§3.2** / `ocean-paper.tex:258-261` |
| 观测归一化 | 论文 **Appendix: Observation Space** / `section-observation-space.tex:210` |
| 坑 1 | 论文 **Appendix: Bloopers** §「Zero Team Spirit Embedding」 / `section-bloopers.tex:23-51` |
| 坑 2 | 论文 **Appendix: Bloopers** §「Learning path dependency」 / `section-bloopers.tex:53-57` |
| 坑 7 | 论文 **Appendix: Human Evaluation**（HTML 通道）；⚠️ 该段在 LaTeX 源码中为**注释状态**，故以 HTML 通道为出处 |
| 21 项权重的一手核验台账（URL + 逐条行号 + 取数时间） | 本地：`docs/openai_five_reward_anneal_audit_2026-09-19.md` §3 / §4（2026-09-18 取数，2026-09-19 04:26 编排者复核重取） |

### 1.2 设计意图（≤5 行）

1. **用「人类共识的中间量」拆解极长的信用分配链**（原文：*"In order to simplify the credit assignment problem ... we use a more detailed reward function."*，`section-rewards.tex:41`）。
2. **让奖励可加、可比、可平均** ⇒ 零和化 + 队内共享系数 $\tau$，把多智能体信用分配问题降维成单标量（`section-rewards.tex:51,65`）。
3. **修正「后期天然奖励更多」的尺度失衡** ⇒ 局内指数时间权重，且**刻意不作用于 win/loss**（`section-rewards.tex:52-58`）。
4. **先各自变强、再学会协作** ⇒ $\tau$ 从 0.3 升到 1.0（降早期梯度方差），**不是**为了让奖励塑形逐渐失效（`section-rewards.tex:74` / `section-hyperparams.tex:69`）。
5. **奖励函数一次成型即冻结**：把「调奖励」换成「调环境与动作空间的渐进引入」，以 TrueSkill 为唯一真判据（`ocean-paper.tex:247-249`）。

### 1.3 迁移性（我们 = Clash Royale 自对弈 PPO / 993K 参数 / 单机 / 手工奖励表 / 无人类数据 / 决策帧 step）

> **我们侧可核实事实（`[本地-A]`，2026-09-19 实测）**：参数量 **993,008**（`docs/cmp_hyperparams_2026-09-19.md:83`）；
> 奖励表**存在两处常量源且键数不同** —— `rl/reward.py:28-46` 的 `_DEFAULT_REWARD` = **16 键**，`rl/config.py:37-` 的 `DEFAULT_REWARD` = **22 键**，
> 差集 = **6**（`engagement_trade`、`_theta`、`_t_ref`、`_gate`、`_measure_only`、`draw_penalty`），且代码注释自称「保持一致；勿单独改一处」；
> PPO 形态 `n_envs=1`、`vf_coef=0.5`、`ent_coef=0.01`、`clip=0.2`、`max_grad_norm=0.5`（`rl/ppo.py:87-88`、`rl/config.py:183-193`）；
> `adv_norm="scale"`、`value_norm="none"`（默认）（`rl/config.py:195,200`）；`rl/*.py` 检索 KL / ref_policy = **0 命中**，无人类数据。

#### ✅ 可迁（机制层面，且多为「我们已有，可对账」）

| # | 可迁项 | 我们这边的现状与注意 |
|---|---|---|
| A1 | **零和构造**（每个体的奖励减去对手平均）| 我们**奖励本身就是单一 team 标量**（`compute_reward`），不存在「每个体视角」⇒ **无法照搬**；可迁的只是**「双向定价必须对称」**这一不变量（我们已用 `crown_lose_weight 10 > crown_weight 8` 体现） |
| A2 | **防守定价 > 进攻定价的非对称** | 我们已有（`tower_dmg_self 0.0012 > tower_dmg_opp 0.001`；双倍期 `0.0022 > 0.002`）——与 Five 的「Lane Assign 是唯一显式负项」同源思路 |
| A3 | **局内时间权重**（后期天然奖励更多 ⇒ 需修正）| **强候选**。我们已有反向版本：`tower_dmg_late`（双倍期塔血更贵）、`elixir_diff_late`（双倍期费更贱）⇒ 本质就是**两段式时间权重的粗粒度版**。⚠️ 但 Five 的因子作用在**除 win/loss 外的全部项**；我们若照做必须逐项声明、且**不得**把 Five 的 $0.6$ 数值搬来（纪律 1） |
| A4 | **奖励运行标准差归一化** | 我们**完全没有**：奖励项自带量纲、不做 running 归一化（`docs/cmp_hyperparams_2026-09-19.md`）。**这是 Five 与我们差异最大、也最可能低成本获益的一项**；对应我们的 `value_norm="running"`（已实现但默认 `none`）⇒ 属**已有开关**，不是新代码 |
| A5 | **「价值损失权重施加在归一化之后」的顺序** | 属实现顺序约定，可直接对账（我们 `v_loss = MSE/(v_scale²)`，`rl/ppo.py:241-248,392`）|
| A6 | **多成分监督信号的重要性**（原文：*"The presence of these additional signals was important for successful training."*）| 支持我们**保留**手工奖励表，而不是「尽快退化成纯终局信号」——与我们现有方向一致 |
| A7 | **价值通路的独立性与容量** | Five 是共享 LSTM + 线性投影单头；我们**已有更强的形态**：`value_bypass`（跳过 GRU）与 `value_independent`（独立编码器+非线性头），且有探针依据（`rl/follower.py:156-180`）⇒ 我们在这一点上**不需要向 Five 学习** |
| A8 | **奖励项体检的思路**：Five 的「奖励几乎没变但胜率 +55%」是**必须同时看 winrate 与 reward** 的经典案例 | 与我们既有的「不看 EV、要看参数有没有在动」（`【R14】`）同源；可直接把我们已有的 `item_health` 类读数形式化为「奖励项 vs 胜负符号一致性」检查 |

#### ❌ 不可迁（逐条给理由）

| 机制 | 不可迁理由 |
|---|---|
| **21 项手写权重表本身** | 量纲与游戏语义完全绑定（`Ancient HP Change` 对远古血量占比、`Lane Assign` 按秒）⇒ 无一项可平移；且数字被纪律 1 禁止对拍 |
| **team spirit 系数 $\tau$ 及其调度** | 我们是**单策略单主体**（`n_envs=1`，奖励为单一 team 标量）⇒ 不存在「五个体奖励取平均」的对象；$\tau$ 的语义在我们这里**无定义** |
| **零和化的「减去敌方奖励均值」实现** | 同上：需要 per-agent 奖励向量；我们是单标量奖励 |
| **固定脚本序列替代惩罚**（技能加点/出装 schedule）| 那是把**动作子空间**从模型手里拿走。我们已有同类物（`belief_planner` 的手写 PlanToken 规则、掩码）⇒ 若要扩张，撞**动作语义冻结 `K_MAX=4`** |
| **158M 参数 / 60k–196k 样本批 / 512–1536 优化器 GPU** | 规模差数个数量级；在任何判据里引入这些数字即违纪律 1，且我们有 `【R5】`（n=1 分辨率不足）与 `【R11】`（不扩参） |
| **「奖励函数一次构造永不改」的实践** | 方向可敬，但**与我们当前状态不符**：我们正在做 A′ 类取证，且有 `【R11】`「不在 A′ 类取证之前改奖励」⇒ 我们的等价物是**先取证再改**，而不是「假设已定稿」 |

**⚠️ 一条必须点名的风险（跨项目同型）**：Five 的坑 1 与我们的 **C14**（「放弃一张买得起的牌」在 75,891 帧里发生 0 次 ⇒ 0 样本 ⇒ 0 梯度）**是同一类失败的一个实例与一个变体**：都是**奖励信号无法指引优化器到真正提高胜率的区域**。⇒ 我们**不得**因为 Five 也踩过就降低对该病根的重视；恰相反，Five 的例子说明**这类问题能在大规模下长期潜伏**。

---

## §2 AlphaStar（StarCraft II）

> **⚠️ 本节的证据等级天花板（先读）**：Nature 正文（`nature.com/articles/s41586-019-1724-z`）及其 PDF/补充材料**本次均不可读**
> （302 → `idp.nature.com` SSO；PDF/ZIP 被 `web_fetch` 拒绝）。因此**本节没有任何一句来自 Nature 正文的原文引用**。
> 涉及论文原文的措辞一律标 `[未验证]`，或标为 `[复现件-OA论文]` / `[二手-分析]`。详见 §4。

### 2.1 统一字段表

#### ① 奖励项清单（项名 / 权重 / 量纲 / team|solo / 是否零和）

| 项名 | 权重 | 量纲 | team \| solo | 是否零和 | 证据等级 | 出处 |
|---|---|---|---|---|---|---|
| **win / loss / draw 终局信号**（唯一环境奖励）| **1**（胜 +1 / 负 −1 / 平 0）| 仅在局末一步非零 | team（阵营级）| 是（二值、双方互斥）| `[复现件-OA论文]`（DeepMind 作者群 2023，描述的是其离线基准的 MDP 定义） | arXiv:2308.03526v1；本次独立复核：LaTeX `main.tex:270` 脚注逐字：*"in StarCraft II, the reward is 1 in a winning state, -1 in a losing state, and zero otherwise. So it does not depend on the action."* |
| 同上（正文另述） | — | *"the reward is the win-loss signal, so it can only be non-zero on the last step of the episode."* | — | — | `[复现件-OA论文]` | 同上 §4.1 正文 |
| **pseudo-rewards / z-statistic（软塑形）** | **未验证**（权重未取回）| **未验证** | 策略级 | **未验证** | `[二手-分析]`（机制描述）| 见下 |

> **⚠️ 关键限定（不得省略）**：上表第一行的原文来自 **AlphaStar Unplugged（2023）**，它是**离线 RL 基准**，
> 其 MDP 定义**不代表 2019 AlphaStar 的在线训练奖励构成**。把这句话当作「2019 AlphaStar RL 阶段唯一奖励」的**直接证据是不成立的**；
> 它只能作为 **DeepMind 作者群对「StarCraft II 环境奖励是什么」的口径确认**。

**pseudo-rewards（z-statistics）——三项机制，权重与调度均未取回**

| 项 | 内容 | 权重 / 退火 | 证据等级 | 出处 |
|---|---|---|---|---|
| z 的来源 | 从**人类数据随机采样**得到的「策略统计量」$z$ | **未验证** | `[二手-分析]` | [Deciphering AlphaStar](https://cyk1337.github.io/notes/2019/07/21/RL/DRL/Decipher-AlphaStar-on-StarCraft-II/) |
| 构造一 | **采样 $z$ 与执行 build order 之间的 edit distance** | **未验证** | `[二手-分析]` | 同上 |
| 构造二 | **采样 $z$ 与执行结果的累计统计量之间的 Hamming distance** | **未验证** | `[二手-分析]` | 同上 |
| 用法 | pseudo-reward 作为 **baseline** 进入 **TD(λ)**；值输出经 `baseline = (2/π)·arctan((π/2)·b)` 压到有界区间 | **未验证** | `[复现件-OA论文]` | arXiv:2104.06890 §3.4 |
| **内在奖励（intrinsic reward over unit types）** | 每个 agent 的个性化目标含「对单位类型的偏好」 | **未验证** | `[二手-分析]` | Deciphering AlphaStar |

**是否存在手工 per-frame 塑形列表？** → **在本次取证范围内未发现任何「手工奖励项列表」**（无 HP / 资源 / 击杀等 per-frame 手写塑形）。
⚠️ **「未发现」≠「不存在」**：因 Nature 正文未取回（`[未验证]`，见 §4 U7）。

#### ② 惩罚项（显式 / 隐式 / 用约束或掩码替代）

| 类别 | 内容 | 证据等级 | 出处 |
|---|---|---|---|
| **显式惩罚项** | **未发现**（本次取证范围内）。源码/取回文本中 `penalt` 在 AlphaStar Unplugged 源码 **0 命中** | `[一手-源码]`（就 AU 而言）| 本次独立复核：`grep -rniE "penalt" /tmp/asu/*.tex` = 0 命中 |
| **隐式惩罚：动作被掩码 / 被约束** | 观测中就带掩码（**看不见的对手单位其信息被 mask**）；`Unit tags` 是**对己方单位的掩码**，决定哪些单位执行动作 | `[一手-源码]` | arXiv:2308.03526 源码 `appendix.tex:84`、`:113` |
| **用约束替代惩罚（核）** | **KL 约束**：RL 更新时持续最小化「当前策略 ↔ 监督人类策略」的 KL，把探索**约束在人类合理策略附近**。**它不是惩罚项，而是 RL 更新损失的组成部分**（第 3 部分）| `[复现件-OA论文]` + `[二手-分析]` | arXiv:2104.06890 §3.4；本次独立复核（e-print `report.tex:141`）：*"The first part is actor-critic loss... It also computes the split vtrace pg-loss (policy-gradient), by calculating vtrace pg-loss respectively for ``action type'', ``delay'' and ``arguments''"* |
| **用掩码替代惩罚（已取回具体实现）** | mini-AlphaStar 在**每个动作头内部**用掩码，而不是给非法动作任何惩罚：① `queue head`：排队值在加入自回归嵌入前**先过掩码**（检查该动作型是否可排队）；② `selected-units head`：计算「哪些单位可被选」的掩码，并对注意力 logits 施加掩码；③ `location head`：施加**由动作型决定的掩码以排除非法位置**，logits 与 location-out 各自再过一次「该动作型是否涉及位置定位」的掩码 | `[复现件-OA论文]` | arXiv:2104.06890 §3.3；本次独立复核 e-print `report.tex:110,112,116`（逐字）|
| **⚠️ 由此得出的判定（就复现件而言）** | mini-AlphaStar 全文源码检索 **`penalt` = 0 命中**（本次独立复核）⇒ **在「复现件层」可见的实现中，确实没有惩罚项**；非法性一律走掩码。**Nature 原文是否如此，仍为 `[未验证]`** | `[一手-源码]`（就 mini-AlphaStar 而言）| 本次 `grep -rniE "penalt" report.tex` = 0 命中 |
| **「动作概率偏离监督策略」被罚** | 同一机制的另一说法：*"The agents are also penalized whenever their action probabilities differ from the supervised policy"* | `[二手-分析]` | Deciphering AlphaStar §Supervised Learning |
| **KL 系数 / 是否退火** | **未验证** | `[未验证]` | — |
| **对手侧压力（league exploiter）** | League 三类角色：main player / main exploiter / league exploiter + coordinator 维护 payoff matrix | `[复现件-OA论文]` | arXiv:2104.06890 §3.5 |
| **matchmaking 约束（PFSP）** | 以胜率加权采样对手（`linear/linear_capped/variance/squared`，默认 `linear`）| `[复现件-OA论文]` | 同上 |

> **判定**：AlphaStar 的「惩罚」**几乎不以奖励负项形式存在**。它用**两条替代路径**：① **KL 约束**（把策略拴在人类先验附近）；② **掩码**（不可观测/不可用的东西直接不出现在动作空间里）。这与 Five 的「负号计价项」是**两种不同哲学**。

#### ③ 塑形与调度（是否 PBRS？是否退火？局内时间权重？）

| 问题 | 结论 | 证据等级 |
|---|---|---|
| **是否 PBRS？** | **未验证**。本次未取回 Nature 正文，**无任何** PBRS / potential-based shaping 的原文证据 | `[未验证]` |
| **是否退火？** | **未验证**（pseudo-reward 的权重、KL 系数的调度均未取回）。**已取回的只有机制**：两阶段（先 SL 再 RL），RL 阶段带 KL 项 | `[未验证]` + `[复现件-OA论文]` |
| **局内时间权重？** | **未验证**。有一项**相关但不同**的机制被取回：**按 delay 折算的折扣** $\gamma^{D_t(s)}$（因为 StarCraft 动作有游戏内 delay）——这是**折扣折算**，**不是**「局内时间奖励权重」 | `[复现件-OA论文]`（mini-AlphaStar §3.4 一带）|
| **总损失构成（对照用，非奖励）** | 四部分：① actor-critic loss（含 pseudo-reward、TD(λ)、split vtrace pg-loss）② split upgo-loss ③ **KL divergence（当前 actor ↔ 监督人类策略）** ④ entropy loss | `[复现件-OA论文]` arXiv:2104.06890 §3.4 |

#### ④ 归一化三问

| 问 | 结论 | 证据等级 |
|---|---|---|
| **奖励归一化** | **未验证**。本次未取回任何「reward normalization / scaling / clipping」的一手或复现件证据。**补充负证据**：mini-AlphaStar 的 reward 侧描述里**完全未出现**归一化措辞（该 e-print 全文 `normaliz*` 只命中 PFSP 的「normalized probability」与架构的 layer normalization，**均与奖励无关**）| `[未验证]`（就 Nature 而言）+ `[一手-源码]`（就 mini-AlphaStar 的「未描述」而言）|
| **价值归一化** | **未验证（就 Nature 而言）**。复现件层可核实的是 **有界 baseline 变换**：`baseline = (2/π)·arctan((π/2)·b)`，其中 $b$ 由一个 **ReLU + ResBlocks(带 layer norm) + ReLU + 单隐单元线性层** 产出。⚠️ **该变换的作用对象是 baseline（用于 TD(λ)），不是「reward/value 的尺度归一化」**⇒ **不得**把它写成「AlphaStar 做了价值归一化」| `[复现件-OA论文]` arXiv:2104.06890 §3.4（公式）+ §3.2（$b$ 的构造）；本次独立复核 e-print `report.tex:143-144`（逐字）|
| **优势归一化** | **未验证**。取回的只是估计器族（TD(λ) / split vtrace / split upgo），**未见** advantage normalization 的描述。复现件对 upgo 只描述「importance weight × 优势」的加权，**不是**优势归一化 | `[未验证]` |
| **观测/动作侧归一化（旁证，非奖励）** | 复现件里可核实的是 **layer normalization**（架构内）与**掩码**（动作头内），以及 PFSP 的**归一化采样概率**。三项**均不属于奖励/价值/优势归一化** | `[一手-源码]`（mini-AlphaStar）|

#### ⑤ value head 设计

| 问 | 结论 | 证据等级 |
|---|---|---|
| 单头 / 多头 / 分解？ | **未验证（就 Nature 而言）**。复现件层可核实的是**单标量价值通路**：观测与 LSTM 输出经「线性(original_256) + ReLU → ResBlocks(original_256, layer norm) → ReLU → **线性到 1 个隐单元**」得到标量 $b$，再经有界变换得 baseline。**未出现 multi-head value / value decomposition 的描述** | `[复现件-OA论文]` arXiv:2104.06890 §3.2/§3.4；本次独立复核 e-print `report.tex:143-144` |
| ⚠️ 「split」一词的**正确解读**（防止误读成多头价值） | mini-AlphaStar 的两个 `split` 指的是 **pg-loss 的拆分**（按 `action type` / `delay` / `arguments` 三组 logits 分别算 vtrace pg-loss 再相加）与 **split upgo-loss**，**与价值头分解无关** | `[复现件-OA论文]` 同上 §3.4；本次独立复核 e-print `report.tex:141,146,152` |
| 与 pseudo-reward 的关系 | 可核实的只是：pseudo-reward 作为 **baseline** 参与 TD(λ)，且 baseline 经有界变换 | `[复现件-OA论文]` |
| **对照（不比较数字）** | 与本仓/绝悟不同，AlphaStar 这一侧**没有**可核实的多价值头奖励分解证据（无论原文还是复现件）。绝悟 P3 的「5 价值头 + $V=\sum w_kV_k$」是**另一个项目**的机制（见 `docs/juewu_research_2026-09-19.md:119-146`），**不得**当 AlphaStar 的证据 | — |

#### ⑥ 已知坑（带原文句 / 或明确标注为二手）

| # | 坑 | 取证情况 | 证据等级 | 出处 |
|---|---|---|---|---|
| 1 | **去掉人类统计量奖励的 ablation** | **未验证**。最接近的是二手总结 *"It shows that the utilization of human data is critical in final results"*（原文含拼写错误 `utilzation`），**不是论文原句** | `[二手-分析]`（间接转述） | Deciphering AlphaStar |
| 2 | **IL 与 RL 目标冲突**（保真 vs 求胜）| **机制性证据有**：KL 项把 RL 策略拉向监督人类策略，同时 pseudo-reward 目标 $z$ 是**随机采样**的一整条人类策略统计量 ⇒ 张力存在。**论文是否显式讨论/消融：未验证** | `[复现件-OA论文]`（机制）+ `[未验证]`（讨论） | arXiv:2104.06890 §3.4 |
| 3 | **league 中策略循环（cycling）/ 被 exploiter 攻击** | 机制侧可核实（strategy cycles、exploiter 专职找弱点、checkpoint 复制历史防遗忘）；另有独立第三方判断「简单 self-play 会卡住、population 才到 Grandmaster」 | `[二手-分析]` | Deciphering AlphaStar；[Alex Irpan 2019-02-22](https://www.alexirpan.com/2019/02/22/alphastar-part2.html) §2 |
| 4 | **公平性 / 外部有效性质疑**：**未使用 human action space**（用 raw API）；人类数据用法过多 ⇒ 更像模仿而非探索新战术；训练资源不可复制 | 可核实的第三方批评 | `[二手-分析]` | arXiv:2209.11553 §1/§2.4；arXiv:2104.06890 §1 |
| 5 | **BC 的误差累积（DAgger 问题）**：长时域 IL 最坏界 $O(T^2\varepsilon)$，StarCraft 决策点上千——但实测 IL 仍强于预期（能到 Gold 级） | 第三方定量/定性分析，**不是**论文结论 | `[二手-分析]` | Alex Irpan §1 |

#### ⑦ 一手出处

| 内容 | 出处（URL + 章节/行号）|
|---|---|
| 论文标识（期刊卷期页 / DOI / PMID / 日期）| Europe PMC REST / OpenAIRE（`[一手-元数据]`）|
| **环境奖励 = 二值 win/loss 的定义句** | arXiv:2308.03526v1 源码 `main.tex:270` 脚注（`[复现件-OA论文]`，本次独立复核取回）；HTML 版 §4.1 |
| RL 损失四部分（含 KL）| arXiv:2104.06890 §3.4（`[复现件-OA论文]`）|
| 有界 baseline 变换与单标量价值通路 | arXiv:2104.06890 §3.2/§3.4；本次独立复核 e-print `report.tex:143-144` |
| 三个动作头内部的掩码（queue / selected-units / location）| arXiv:2104.06890 §3.3；本次独立复核 e-print `report.tex:110,112,116` |
| pseudo-reward = edit distance / Hamming distance | Deciphering AlphaStar（`[二手-分析]`）|
| league 三角色 / PFSP | arXiv:2104.06890 §3.5（`[复现件-OA论文]`）|
| 观测掩码 / Unit tags 掩码 | arXiv:2308.03526 源码 `appendix.tex:84,113`（`[一手-源码]`）|
| **Nature 正文** | ❌ 本次**不可读**（302 → SSO）；PDF 亦不可读 |
| 全量来源清单与失败通道 | 本地：`docs/alphastar_reward_il_survey_2026-09-19.md` §7（2026-09-19 取数）|

#### ⑧ 阶段归属（本文件新增字段，用于执行「不得把 IL 阶段的事写成 RL 阶段」）

| 机制 | 阶段归属 | 说明 |
|---|---|---|
| 人类回放 IL / 行为克隆（6 头交叉熵）| **IL 阶段** | `[复现件-OA论文]`。**不得**写成 RL 阶段奖励设计 |
| z-statistic 的采样与「作为软塑形目标」 | **RL 阶段**（消费方）| z 本身来自人类数据（IL 的产物），但在 RL 阶段作为 pseudo-reward 目标 |
| KL(当前策略 ‖ 监督人类策略) | **RL 阶段的损失项**，但其**参考分布来自 IL 阶段** | ⚠️ **跨阶段接口**：既不是纯 IL，也不是奖励项。写作时必须显式说明「这是 RL 更新损失的第 3 项，参考分布是 IL 阶段的产物」 |
| league / PFSP / exploiter | **RL 阶段** | 对手侧机制，非奖励侧 |
| 掩码（观测与动作） | **RL 阶段**（数据管线）| — |

### 2.2 设计意图（≤5 行）

1. **把探索空间压到「人类合理策略附近」**，用一个信息量极高、但极其昂贵的先验（971K 人类回放级的 IL）换掉大部分探索成本。
2. **奖励侧刻意保持极简**：只给终局 win/loss，让多样性由 **z-pseudo-reward + league + 内在奖励** 供给，而不是由 per-frame 手写塑形供给。
3. **用 KL 约束而非惩罚**：不惩罚「错」，而是把策略拴在人类分布上——这是「约束替代惩罚」的最彻底形态。
4. **对手多样性是机制的一部分**：league（main / main exploiter / league exploiter）+ PFSP 用来对抗 self-play 的策略循环与遗忘。
5. **诚实边界**：上述 1–4 中，**第 2、3 条的确切权重、调度、消融均未验证**（Nature 正文未取回）⇒ 只能当**方向性假设**，不能当可复现配方。

### 2.3 迁移性（我们 = Clash Royale 自对弈 PPO / 993K 参数 / 单机 / 手工奖励表 / 无人类数据 / 决策帧 step）

#### ✅ 可迁（迁移的是**机制形态**，不是数字）

| # | 可迁项 | 注意 |
|---|---|---|
| B1 | **「终局 win/loss 是唯一真奖励」这一设计方向** | 我们已有终局项（`win_bonus` / `lose_penalty`）。可作为**收敛方向**：把不可自算的塑形项逐步换成终局信号 + 少量可核实项。⚠️ AlphaStar **没有证明**它在小规模/单机下也 work ⇒ 只能当方向 |
| B2 | **两阶段：先「专家/先验」初始化，再 RL 微调** | 我们**有**可用先验（手写 `belief_planner` / PlanToken）。⇒ 可做「对**自有规则专家**做 BC 初始化」——**不要**照搬「从人类回放学」 |
| B3 | **KL-to-prior 作为软约束**（唯一能保留 KL 精髓的形态）| 若要防「RL 忘掉先验技能」，可加 $\mathrm{KL}(\pi_\theta \| \pi_{\text{prior}})$，prior = **冻结的自有启发式策略**（而非人类）。⚠️ 这是**另一个机制**，**不得**声称「在用 AlphaStar 的 KL-to-human」。⚠️ 新增超参须预注册（`【R3】`）|
| B4 | **「用统计量目标做软塑形」的形态**：把想要的局面统计做成**可自算**的目标向量，用**距离**当软塑形 —— 统计量可来自**我们自己的专家脚本** | ⚠️ 严格限定：距离类塑形必须**可微/可批算**，且要防「刷距离」（reward hacking）。参见 `【R12】`（能算的不许让网络猜）|
| B5 | **「防守定价 > 进攻定价」的结构性不对称** | 我们**已有**（`crown_lose_weight 10 > crown_weight 8`；`tower_dmg_self 0.0012 > tower_dmg_opp 0.001`）|
| B6 | **对手多样性的直觉**（与 Five 的 20% 历史对手同类）| 我们已有 `pfsp.py` / `league.py` / `opponents.py`。只需保留「历史版本 + 专门找弱点的对手」两类；**不要**引入多机 league 规模 |

#### ❌ 不可迁（逐条给理由）

| 机制 | 不可迁理由 |
|---|---|
| **z-statistic pseudo-rewards（原始形态）** | ① **无人类数据** ⇒ z 的采样池根本不存在；② **量纲**：z 是「整条 build order / 一整套累计统计量」级别的对象，其 edit/Hamming 距离的数值范围与我们单帧奖励（`tower_dmg 0.001` 级）**相差数个数量级** ⇒ 直接搬会**淹没终局信号**；③ **规模**：需「每局多条 z、每步算距离」的采样与计算，单机 `n_envs=1` 吞吐承受不起 |
| **KL-to-human** | **无人类策略** ⇒ 参考分布不存在。数学上可换成 KL-to-prior（B3），但**那是另一个机制** |
| **971K replays 规模的 IL 阶段** | 我们**零**人类回放；且 993K 参数 / 单机的成本收益比不可比（违纪律 1）|
| **League（12 个 league learner + 三型角色）** | 单机单 env；多 agent 并训的边际收益未在本仓取证，且冲撞「不扩参 / 单变量 / 预注册」 |
| **有界 baseline 变换 `(2/π)arctan((π/2)b)`** | 它是为 z-pseudo-reward 的有界区间服务的；我们无 z ⇒ **失去动机**。且它触及价值头架构 ⇒ `【R6】`（架构变更须 `--fresh`），而 `【R11】` 明令不在 A′ 类取证之前拆价值头 |
| **$\gamma^{D_t(s)}$ 按 delay 折现** | 我们的 `step` 已是**决策帧**（`【R9】` 口径），不存在帧间 skip 的半连续时间折扣；引入会造成**两套时间口径**（`【R17】`）|
| **intrinsic reward over unit types** | 属手工塑形；`【R11】` 明令取证前不改奖励；且会与既有奖励表**重复计价**（本仓已两次踩过：S2 的 `tower_term` 与 $\Delta\Phi$、`crown` 方向）|
| **Action masking 的全部做法** | 我们**已有掩码**，且已知它是一处**真实缺陷源**（`used` 集合 0/1-based off-by-one 造成 497 帧整包被拒）。⇒ 不能把「加掩码」当免费的改进；任何掩码改动须走 `【R13】`（位图对账）|

**⚠️ 一条反直觉的告诫**：AlphaStar 路线（极简奖励 + 极强先验）**看起来**很诱人，但它的成功**建立在人类数据 + 巨算力**之上；
我们两者都没有，且本仓已实测「**约束在探索/机制侧**」（S1 门禁未通过：全帧圣水≥6 = 0/8191、Xbow = 0/929）。
⇒ 对我们的病根，**机制侧（O7）比奖励侧更贴**；把奖励改得更稀疏**不会**凭空生出探索能力。

---

## §3 来源清单（URL / 文件 + 取数时间）

**统一取数时间：2026-09-19**（UTC 2026-09-18；本地 `date -u` = `2026-09-18T20:39Z`）

### 3.1 外部来源

| # | 来源 | 等级 | URL | 本次取回状态 |
|---|---|---|---|---|
| E1 | **OpenAI Five 论文 arXiv e-print（LaTeX 源码）＝ §1 判定主依据** | 一手-论文原文 | https://arxiv.org/e-print/1912.06680v1 | ✅ **8,831,567 B**（与盘上审计两次取数同尺寸）；解出 18 个 `.tex`；`section-rewards.tex` / `ocean-paper.tex` / `section-hyperparams.tex` / `section-network-architecture.tex` / `section-bloopers.tex` / `section-observation-space.tex` / `section-exploration.tex` / `section-action-space.tex` 逐段读取 |
| E2 | OpenAI Five 论文 arXiv HTML（独立第二通道）| 一手-论文原文 | https://arxiv.org/html/1912.06680v1 | 盘上审计复核取回 493,886 B（本次未重取）|
| E3 | OpenAI Five arXiv abs（元数据：v1 / 2019-12-13 / 仅一版）| 一手-元数据 | https://arxiv.org/abs/1912.06680 | 盘上审计已取回（本次未重取）|
| E4 | **AlphaStar Unplugged（DeepMind 作者群，2023）** | 复现件-OA论文 | https://arxiv.org/html/2308.03526v1 | ✅ 全文（盘上报告）；**本次另独立取回 e-print** https://arxiv.org/e-print/2308.03526 （1,800,543 B，`main.tex`/`appendix.tex`）并逐字复核奖励定义句 |
| E5 | mini-AlphaStar（arXiv:2104.06890）| 复现件-OA论文 | https://ar5iv.labs.arxiv.org/html/2104.06890 | ✅ 全文（盘上报告）；**本次另独立取回 e-print** https://arxiv.org/e-print/2104.06890 （1,594,549 B，单文件 `report.tex`）并逐字复核 §3.2/§3.3/§3.4 的价值通路、掩码与损失拆分 |
| E5b | mini-AlphaStar 开源代码仓库 | 一手-官方代码（复现件作者）| https://github.com/liuruoze/mini-AlphaStar | ✅ 可访问（本次 `curl` HTTP **200**）；⚠️ **本次仅验可达性，未逐行读代码** ⇒ 本文件不引其代码行号 |
| E6 | On Efficient RL for Full-length Game of SC2（JAIR 75:213-260 / arXiv:2209.11553）| 复现件-OA论文 | https://www.jair.org/index.php/jair/article/view/13743 ・ https://ar5iv.labs.arxiv.org/html/2209.11553 | ✅ 取回（盘上报告）|
| E7 | Deciphering AlphaStar on StarCraft II（Chai Yekun, 2019-07-21）| 二手-分析 | https://cyk1337.github.io/notes/2019/07/21/RL/DRL/Decipher-AlphaStar-on-StarCraft-II/ | ✅ 全文（盘上报告）|
| E8 | An Overdue Post on AlphaStar, Part 2（Alex Irpan, 2019-02-22）| 二手-分析 | https://www.alexirpan.com/2019/02/22/alphastar-part2.html | ✅ 全文（盘上报告）|
| E9 | Nature 论文页面 / PDF / 补充材料 | 一手-正文 | https://www.nature.com/articles/s41586-019-1724-z (+`.pdf`) ・ Springer ESM ZIP | ❌ **全部不可读**：302 → `idp.nature.com` SSO；PDF/ZIP 被 `web_fetch` 拒绝 |
| E10 | 元数据源（OpenAIRE / Europe PMC / Semantic Scholar）| 一手-元数据 | `api.openaire.eu` ・ `ebi.ac.uk/europepmc` ・ `api.semanticscholar.org` | ✅ 取回（盘上报告）；确认**无 arXiv 预印本**、`isOpenAccess=N` |
| E11 | DeepMind 官方博客（2019-10-30 / 2019-01-24）| 一手-官方（仅标题/日期）| `deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/` | ⚠️ 部分：正文客户端渲染 |

**本次环境的通道可用性实测**：`web_search` = ❌ **HTTP 402（余额不足）**，全程不可用；`curl` 直连 `arxiv.org/e-print/*` = ✅ 可用（本文件两条 e-print 均由此取得）；`web_fetch` 对 arXiv HTML = ✅ 可用（盘上报告已证）。

### 3.2 本地来源（`file:line`，2026-09-19 读取）

| # | 文件 | 用途 |
|---|---|---|
| L1 | `docs/openai_five_reward_anneal_audit_2026-09-19.md` §3 / §4 / §5 | 21 项权重一手核验台账、来源清单、未验证项 U1–U8 |
| L2 | `docs/alphastar_reward_il_survey_2026-09-19.md` §1–§8 | AlphaStar 全部证据等级标注、来源清单、未验证项 U1–U10 |
| L3 | `docs/cmp_hyperparams_2026-09-19.md:83`（参数量）、`:37-` 段（PPO 超参 / adv_norm / value_norm）| 我们侧基线事实 |
| L4 | `src/clasher_new/rl/reward.py:28-46` | `_DEFAULT_REWARD` = **16 键**（脚本计数）|
| L5 | `src/clasher_new/rl/config.py:37-` （`DEFAULT_REWARD`）| = **22 键**；与 L4 差集 **6**（`engagement_trade`、`_theta`、`_t_ref`、`_gate`、`_measure_only`、`draw_penalty`）|
| L6 | `src/clasher_new/rl/ppo.py:87-88`（`vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, adv_norm="batch"` 形参默认）、`rl/config.py:183-193`（生效默认）| PPO 形态与归一化三问的我们侧答案 |
| L7 | `src/clasher_new/rl/config.py:195`（`adv_norm="scale"`）、`:200`（`value_norm="none"`）| 同上 |
| L8 | `src/clasher_new/rl/follower.py:156-180`、`:239-248`、`:308-314` | 价值头三形态（shared / bypass / independent）|
| L9 | `src/clasher_new/rl/env_wrapper.py:41-48`、`:99`、`:516` | `_DEFAULT_REWARD` 为默认奖励表（L4 的 16 键为默认生效值），而非 `config.DEFAULT_REWARD` |
| L10 | `docs/juewu_research_2026-09-19.md:119-146` | 绝悟 P3 的 5 价值头分解 —— **仅作方向对照，不作 AlphaStar 或 Five 的证据** |
| L11 | `docs/train_health_metrics_2026-09-18.md` | 我们 `v_scale` 实测轨迹（1.75 → 24.87）；GBK 陷阱 |

> **纪律复述**：本文件所有数字（Five 的 5 / −1 / 0.002 / $0.6^{T/10\min}$ / $\tau$ / 158,502,815；AlphaStar 的 +1/−1/0；我们的 993,008 / 16 / 22）
> **仅用于各自内部的一致性论证与标识**，**一律不得跨项目对拍**，也不得用于标定任何其他项目（含本仓的新实验判据）。

---

## §3.5 横向对比（同一字段、并排；**只比结构，不比数字**）

> 本表的用法：**读「定性」列与「证据强度」列**，**不要**把两边的数值放在一起看。
> 单元格内不出现任何跨项目算术（无差值、无倍数、无排名）。

| 字段 | OpenAI Five | AlphaStar | 我们（本仓基线）|
|---|---|---|---|
| **① 奖励形态** | **密集手写塑形**：21 项权重 + 3 个结构组件（零和 / 局内时间权重 / team spirit）| **极简**：二值终局 win/loss 为唯一环境奖励 + 由人类统计量 z 导出的 pseudo-reward（软塑形）；无 per-frame 手写项（**就本次取证范围**）| **密集手写塑形**：默认 16 键（`_DEFAULT_REWARD`）/ 配置层 22 键（`DEFAULT_REWARD`）|
| **① 量纲是否显式** | **是**（论文给「占最大生命比例」「每秒」「每单位金币」等）| **未验证**（pseudo-reward 的量纲未取回）| **是**（但**两处常量源键数不同**，见 L4/L5）|
| **① 是否零和** | **是**，且给了实现（减去敌方奖励均值）| **未验证**（终局信号天然互斥；pseudo-reward 侧未取回）| 奖励为**单一 team 标量** ⇒ 无「每主体视角」，零和无定义 |
| **② 惩罚哲学** | **负号计价项**（5 个负权重），非法动作走 no-op 而非惩罚 | **约束替代惩罚**：KL 约束 + 动作头内掩码；复现件 `penalt` = 0 命中 | **显式惩罚项**（`invalid_penalty` 0.05 / `lose_penalty` 10 / `draw_penalty`）+ 掩码并存 ⚠️ 掩码曾出真实缺陷（`used` 0/1-based off-by-one，497 帧）|
| **③ PBRS** | **自称「松散仿照」势函数塑形**，并声明其保证不适用 | **未验证** | 我们**没有** PBRS 声明；`_phase_weights` 类时间权重是分段常数 |
| **③ 退火** | **奖励权重无调度**（一次构造）；**team spirit 是升到 1 的 heat-up**，**不是**退火 | **未验证** | **无奖励权重调度**；`ent_coef` 固定 0.01（无退火）|
| **③ 局内时间权重** | **有**：除 win/loss 外乘 $0.6^{(T/10\min)}$ | **未验证**（只有按 delay 的折扣折算，是另一回事）| **有（粗粒度两段式）**：`tower_dmg_late` / `elixir_diff_late` 在双倍期切换 |
| **④ 奖励归一化** | **有**：运行标准差归一化；价值损失权重施加在归一化之后 | **未验证** | **无**（奖励项自带量纲）|
| **④ 价值归一化** | **未发现**（论文只给价值损失权重 0.25↔1.0）| **未验证**；复现件层是**有界 baseline 变换**（作用对象 = baseline，**非**尺度归一化）| **有开关**：`value_norm="running"` ⇒ $v_{loss}=\text{MSE}/v_{scale}^2$，$v_{scale}$ = 回报运行 std（**默认 `none`**）|
| **④ 优势归一化** | **未验证**（只描述 GAE $\lambda=0.95$）| **未验证** | **有**：`adv_norm`，默认 **`scale`**（只除批 std、不中心化）|
| **⑤ value head** | **单头、线性投影、与策略共享 LSTM 与梯度** | **未验证（Nature）**；复现件层 = **单标量价值通路、无多价值头分解**（`split` 指 pg-loss 拆分）| **三形态可选**：shared（`value_head` 线性）/ `value_bypass`（跳 GRU）/ `value_independent`（独立编码器 + 非线性头）|
| **⑥ 已知坑（各选最相关一条）** | **塑形奖励与胜率脱钩**：128 参数置零使胜率显著提高，而塑形奖励几乎不变 ⇒ 优化器找不到该方向 | **IL↔RL 张力**（机制侧成立；论文是否讨论**未验证**）| **reward hacking 已实测**：C14「放弃一张买得起的牌」在 75,891 帧发生 **0 次** ⇒ 0 样本 ⇒ 0 梯度 |
| **⑦ 证据强度（本文件内）** | **高**：arXiv e-print LaTeX 源码逐字（含本次复核重取，尺寸与盘上一致）| **低**：**Nature 正文一句未得**；全部为 `[复现件-OA论文]` / `[二手-分析]` / `[未验证]` | **高**：本仓源码 + 实测（`file:line` 齐）|
| **⑧ 阶段归属风险** | 无 IL 阶段（纯 RL 自对弈）| **有**：IL 先行、RL 再以 KL 拴住 ⇒ 必须区分「IL 的产物」与「RL 的损失项」| 无 IL 阶段（无 BC 锚、无人类数据）|

### 3.5.1 三条可复查的「同型风险」清单（跨项目同型 ≠ 数字可比）

| 风险型 | Five | AlphaStar | 我们 |
|---|---|---|---|
| **奖励信号与真实目标脱钩** | 有实例（置零 +55% 胜率而奖励不变）| 机制上已知（KL 保真 vs pseudo-reward 求胜）| 有实测（C14：0 样本 ⇒ 0 梯度）|
| **奖励方差压垮价值函数** | 有实例（Divine Rapier ⇒ 奖励方差上升 ⇒ 价值函数不可靠）| **未验证** | 已知（`vraw` 逐 update 无分辨力，MAD 与中位同量级）|
| **约束/掩码自身出错** | 非法组合 → no-op（设计如此，未见缺陷）| 掩码遍布三个动作头（复现件层）| **已出真实缺陷**（`used` off-by-one，497 帧整包被拒）⇒ 掩码改动必须位图对账（`【R13】`）|

---

## §4 未验证项

### 4.1 OpenAI Five（承接盘上审计 U1–U8，**状态原样保留**）

| # | 未验证项 | 状态 |
|---|---|---|
| U1 | **「OpenAI Five 用奖励塑形退火」这一流行说法本身** | **部分闭合**：`anneal` 家族 5 处全非奖励权重（见 §1.1 ③）。**未验证的是「该说法有原始出处」**——只能证伪其在 arXiv:1912.06680v1 中的存在，**不能**排除其在博客/演讲/其他版本中的存在 |
| U2 | **奖励权重是否存在未写入论文的调度** | **未验证**：无 OpenAI Five 训练代码可查 |
| U3 | **Table 4 中 $x \to y$ 的确切插值形状之外的细节** | **未验证**：只给「smooth monotonic transition (usually a linear change over one to three days)」；每条超参各自的起止步数**未逐项给出** |
| U4 | **Reward normalization 是否随训练变化** | **未验证**：表注 c 只给 running std，**未给任何调度** ⇒「没有调度」是**缺证据**而非**有反证** |
| U5 | **GAE / γ / horizon 与奖励塑形权重的交互** | **未验证** |
| U6 | **论文其他版本（v2+）或会议版是否新增退火描述** | **未验证**：arXiv 上**仅 v1 一版**（元数据已核）；会议/期刊终稿**未查** |
| U7 | **`anneal` 之外的近义词是否描述了奖励权重渐变** | **本轮进一步闭合（仍非全闭合）**：本次扩检 `ramp`/`decay`/`coefficient schedule`/`warm-up`/`curriculum`/`schedul`/`linear change`/`gradually`，**与 reward weight 调度同时命中的有效行 = 0**（唯一命中是局内衰减 `section-rewards.tex:55`；另两处是**注释行** `section-hyperparams.tex:5`、`section-substudies.tex:309`）。**仍未穷举**「全部」近义词，故保持未验证 |
| U8 | **本报告是否覆盖 Appendix 全部内容** | **未验证**：覆盖了 `anneal` 穷举与 Appendix G 正文；**附录其余小节未逐行通读** |

**本节新增的确定性缺口（不是「未验证」，而是可证的结构性问题）**
| # | 缺口 | 证据 |
|---|---|---|
| G1 | **一手的「优势归一化」与「价值归一化」机制在论文中不存在**（不是「未找到」，而是穷举后无该概念）| 本次源码穷举 `advantage` / `normaliz` / `value head` / `value loss` 全部命中已逐条列出（§1.1 ④⑤）|
| G2 | **Table 5 列数：盘上审计写三列，源码表头为四列** | `section-rewards.tex:7`：`Name & Reward & Heroes & Description`。**21 行数据与权重逐行一致** |

> **引用禁则（沿用盘上审计）**：U1–U8 中任何一条，**不得**被后续文档写成「已证 OpenAI Five 没有退火」或「已证有退火」。
> 允许的写法只有：**「在 arXiv:1912.06680v1 的一手文本中，未发现奖励塑形权重退火的描述」**。

### 4.2 AlphaStar（承接盘上调查 U1–U10，**状态原样保留** + 本文件新增）

| # | 未验证项 | 为什么没验证到 |
|---|---|---|
| U1 | **IL 阶段的确切回放数**（是否 = 971K；是否区分「IL 用」与「RL 用人类统计量」两批）| Nature 正文/补充材料不可读 |
| U2 | **IL 损失的确切形式**（6 头交叉熵之和是否为论文原式；温度/权重的原值）| 同上；目前来自 mini-AlphaStar 复现件 |
| U3 | **z-statistic 的定义与构造**（哪些统计量、多长窗口、如何采样）| 同上 |
| U4 | **pseudo-reward 的权重与退火 schedule** | 两项独立检索均无结果 |
| U5 | **KL-to-human 的系数与是否退火** | 同上 |
| U6 | **「去掉人类统计量奖励」的 ablation 数字** | 只拿到一句二手总结 |
| U7 | **是否有其它手工塑形项**（「未发现」≠「不存在」）| 原文不可读 |
| U8 | **「main agent 被 exploiter 攻击」是否为论文原话** | 只拿到机制描述与第三方转述 |
| U9 | **CPU 核数口径**（128,000 vs 12,000）| 两篇同作者不同版本互相矛盾 ⇒ 本文件**不采任何一版** |
| U10 | **Nature 页面 altmetric/引用数快照** | Nature 页面 302（当前 OpenAIRE 侧 `citationCount=3856`、`influence_alt=2570`，**为聚合库口径，非 Nature 本站计数**）|

**§2 侧由此产生的字段级缺口（本文件因此「留空」而非猜测的字段）**

> **本次新增的口径澄清（重要）**：下表「未验证」一律指 **就 Nature 论文而言**。其中 ④⑤ 两项与 ② 的**掩码部分**，
> 已在 **复现件层**取得可核实的负证据 / 结构描述（见 §2.1 对应表内的 `[复现件-OA论文]` 行）。
> ⇒ **写作规则**：可以说「复现件 mini-AlphaStar 如此实现」，**不得**说「AlphaStar 论文如此规定」。

| 字段 | 状态 |
|---|---|
| ① pseudo-reward 的**权重 / 量纲 / 是否零和** | **未验证**（留空）|
| ② 显式惩罚项清单 | **未验证（就 Nature 而言）**；复现件层负证据：mini-AlphaStar `penalt` = 0 命中，非法性一律走掩码 |
| ③ 是否 PBRS / 是否退火 / 局内时间权重 | **未验证**（留空）|
| ④ 奖励 / 优势归一化 | **未验证（就 Nature 而言）**；复现件层负证据：未描述奖励或优势归一化 |
| ④ 价值归一化 | **未验证（就 Nature 而言）**；复现件层可核实的是**有界 baseline 变换**（作用对象 = baseline，**不是**尺度归一化）|
| ⑤ value head 单头/多头/分解 | **未验证（就 Nature 而言）**；复现件层可核实：**单标量价值通路**、**无多价值头分解**；两个 `split` 是 **pg-loss 拆分**与 **split upgo-loss** |
| ⑧ KL 项的阶段归属 | **已可写**：RL 阶段的损失项、参考分布来自 IL 阶段（`[复现件-OA论文]`）⇒ 这是本文件 AlphaStar 侧**少数可以下结论的**结构事实 |

> **引用禁则（本文件新增）**：§2 中一切 `[未验证]`，**不得**在后续文档中被写成论文事实。
> 特别地：**不得**把 AlphaStar Unplugged（离线 RL 基准）的 MDP 奖励定义，当成 2019 AlphaStar 在线 RL 阶段奖励构成的直接证据。

### 4.3 本文件自身的已知不足

| # | 不足 |
|---|---|
| N1 | 本文件**未取得** AlphaStar Nature 正文的任何一句原文 ⇒ §2 的证据等级**整体低于** §1 |
| N2 | OpenAI Five 侧未取回 HTML 通道（仅用 LaTeX 源码；盘上审计已证两通道一致，故风险低）|
| N3 | Five 的 §1.1 ②「固定脚本序列」一行（Appendix F.1）为**盘上审计转录**，本次**未从源码复核** ⇒ 该行标为转引。另：**本次复核发现 `section-human-evaluation.tex` 整段在 LaTeX 源码中被注释**（`%` 开头），而盘上审计已证 **HTML 通道含 LaTeX 源码中被注释的材料**（`anneal` 即一例）⇒ 凡引用 Human Evaluation 附录，本文件一律以 **HTML 通道**为出处、**不引 LaTeX 行号**。**未对该注释差异做穷举普查**（可能还有其他段落存在同类差异）⇒ 本项保持开放 |
| N4 | 本文件**未**对 AlphaStar 的 971K / 44 天 / 384 TPU 作独立复核（承接盘上报告，且按纪律 1 不作对拍）|
| N5 | 本文件**不涉及**任何第三项目（绝悟 / VPT / FirstLight 等）的证据；`L10` 的绝悟引用**仅为方向对照**，未用于任何 AlphaStar/Five 结论 |
| N6 | 本文件曾委派一个子智能体专查 AlphaStar 的 ⑤/④ 字段（value head / 归一化）；**该子智能体在交付前被主动停止**（同一批事实已由本文件作者用 `arxiv.org/e-print/*` 通道独立取得并逐字复核，见 §2.1 ④⑤）。⇒ **没有**「未取回的中间结论」被并入本文件；`§4.2` 中仍标 `[未验证]` 的项，就是**本次确实没拿到**的项 |
