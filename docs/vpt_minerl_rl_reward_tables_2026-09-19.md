# VPT × MineRL/BASALT —— RL 阶段奖惩设计统一对照表

- **取数时间**：2026-09-19（UTC 2026-09-18T20:39Z 起；各条 URL 的抓取见 §4，均落在 20:39–20:47Z 窗口）
- **方法**：`curl` 直取一手文本后本地落地 grep 定位——VPT 全文 HTML（`/tmp/vpt/vpt.txt`，1262 行，由 arXiv HTML 转纯文本）、MineRL 三篇论文 HTML（`/tmp/mr/*.txt`）、`minerllabs/minerl` 三个 ref 的全仓 tarball（`@dev`=v1.0.2 / tag `v0.4.4` / tag `v0.3.7`，本地 `/tmp/mr/`）、`openai/Video-Pre-Training@main` 全仓 tarball（`/tmp/vptrepo/Video-Pre-Training-main/`）、**官方 `.model` 参数文件**（`pickle` 反序列化，`/tmp/vptrepo/models/`）、**BASALT 2022 BC 基线全仓 tarball**（`/tmp/mr/baserepo/`）。`web_search` 本会话 402（不可用）。
- **纪律**：① **「60× 奖励缩放」不得引用**——本报告对 VPT 全文两次独立 grep（`scale` / `scaling` / `60` / `reward`）未核到任何奖励缩放倍数，任务书该数字**在原文不存在**，列 §5-U1。② **禁止跨项目数字对拍**（VPT/MineRL = Minecraft 20Hz 键鼠/20 tick；本地 = 卡牌决策帧）。③ 本地数字只作**量纲锚**引用，不与上游对拍。④ 未核到的一律标 `未验证`。
- **与既有报告的关系**：本报告是 [`docs/vpt_minerl_dd_2026-09-19.md`](vpt_minerl_dd_2026-09-19.md) 的**补充与四组更正**（见 §6）。既有报告结论**未被推翻**；四组更正：① `reward_shaping`「未验证」→ **已核到该 API 符号在 5 个 ref 全不存在、并核到成就 delta 的真实现法**（§2.2-C2–C4）；② BASALT 2022 与 MineRL 2019 **是两条不同的竞赛线**（归类更正）；③ 「未核到禁止 dense 塑形条文」**不成立**——2019/2020 提案明文禁止人工硬编码塑形（§2.1-M20/M21）；④ **MineRL env 代际差异**（`v0.3.7`/`v0.4.4` 的 `dense` ≠ `v1.0.2` 的 `dense`，竞赛环境实际全用 sparse；§2.1-M17）。
- **★ 本轮两项超出既有报告的新一手证据**：**官方 `.model` 参数文件里的完整 RL 奖励表 + monitor 权重**（§1.1.0/§1.1.1）——把「奖励表」从论文单源升级为**论文 + 代码双源**，并第三次否证「×60」；以及 **BASALT 2022 BC 基线里的 KL 锚定项（含「VPT 没用过」的自注）**（§2.5）——直接改变 §3.4 的答案结构（由二分法变三分法）。
>
> **★ 统一 schema（五份对象报告共用：字段名 / 顺序 / 单位口径一致 — 2026-09-19 编排者补，仅加声明、不改正文）**
> **字段集 ①–⑧**：① 奖励项清单（项名 / 权重 / 量纲 / 作用域 / 是否零和）・② 惩罚项（显式 / 隐式 / 用约束或掩码替代）・③ 塑形与调度（是否 PBRS？是否退火？局内时间权重？）・④ 归一化三问（奖励 / 价值 / 优势）・⑤ value head 设计・⑥ 已知坑（带原文句；二手必须标注）・⑦ 一手出处（URL / `file:line` + 取数时间）・⑧ 阶段归属（IL / RL / 跨阶段接口）
> **本报告落点**：①=§1.1 / §2.1 ・ ②=§1.2 / §2.3 ・ ③=§1.3 / §2.3 ・ ④=§1.4 / §2.3 ・ ⑤=§1.5 / §2.3 ・ ⑥=§1.6（KL 消融）/ §2.5（BASALT 锚定项自注） ・ ⑦=§4 ・ ⑧=§3.4（IL→RL 锚定；另见 §1.2、§2.5）
> **统一单位口径**：权重一律**按来源原样记录**并显式标注量纲（分母单位 / 时间基：VPT = Minecraft 20Hz 键鼠、MineRL = 20 tick、本仓 = 卡牌决策帧）；无单位者写「无单位」；**同名项/不同代际若量纲不同则并列不调和**，不折算、不换算、不对拍。
> **证据等级图例**（沿用盘上约定，不新建体系）：`[一手-论文原文]`・`[一手-官方代码]`・`[复现件-OA论文]`・`[二手-分析]`・`[未验证]`・`[本地-实测]`
> **三条硬纪律**：① **禁止跨项目数字对拍**（一切数字只用于标识与各自内部一致性论证）；② **标「未验证」的项一律保持未验证**，不因抽成表而升级证据等级；③ **不得把 IL 阶段的事写成 RL 阶段**，跨阶段项必须落在 ⑧。

---

## 一句话结论

VPT 的 RL 阶段是「**稀疏 item 成就奖励（4 档 1/2/4/8 ÷ 该 item 总奖励数；实现形态 = 5 个具名 monitor 的加权和，只有 `order_invariant_curriculum` 权重为 1）+ 一个锚在冻结起点策略上的 KL 正则项（论文 ρ=0.2、×0.9995/iter，替代熵正则）+ 只对值函数目标做 EWMA 归一化 + 单标量值头**」，**全程没有任何惩罚项进入奖励**；MineRL/BASALT 线则是「**成就/里程碑奖励（2019 竞赛 = 里程碑链式翻倍 ×2、终局 ×4；2020 竞赛 = Table 1 里程碑 1→1024；env 代码 = 同一梯队 + Navigate 的 dense 距离项）+ 零惩罚 + 明文禁用人工硬编码塑形（但允许 IRL）+ BASALT 干脆不给奖励函数、改用人工成对比较 + TrueSkill**」；两家都**没有**把「KL 到先验」当作惩罚型 reward（VPT 是**辅助损失项/正则**）——**关于「把 π_pt 换成早期自对弈 ckpt」：`π_pt` 确实就是「RL 微调起点 ckpt」（主实验 = early-game ckpt、对照实验两臂各锚各自起点），所以「锚在冻结的早期 ckpt」在原文有直接先例；但「换成自对弈产物」原文完全未做、未讨论、未消融 ⇒ 未验证类推；另有一处同生态第二实例（BASALT 2022 BC 基线，KL 权重恒 1.0、不退火、且自注「VPT 没用过」）只能当「有实现先例」、不能当效力证据（§3.4 给三分法结论）**。

---

## §1 VPT 表（RL 微调阶段：奖励 / 惩罚 / 塑形 / 归一化 / value head / 坑）

### 1.1 奖励项清单（★ 逐 item 已核到 Table 7；★★ 本轮还从**官方 `.model` 参数文件**核到同一张表的实现形态）

#### 1.1.0 ★★ 新发现：VPT 的 RL 奖励实现 = **可插拔 monitor 加权和**（论文没写，代码侧一手）

官方公开 blob（**未在 README 链接**）的 `.model` 参数文件被 `pickle` 反序列化后，**逐字**带着 RL 工厂路径与奖励配置：

| 事实 | 逐字值 |
|---|---|
| RL 工厂路径（3 个 RL 模型） | `forc.model.rl.ppg.ppg_policy:create`（**PPG**，与论文 Table 6 的 PPG 一致）；foundation 模型是 `forc.model.rl.ppo.ppo_policy:create` |
| 网络工厂 | `ypt.model.policy:MinecraftPolicy`；`pi_head_opts = {'temperature': 2}`（foundation 为 1） |
| **奖励 = 5 个 monitor 的加权和** | `craft_stats` **weight 0** / `mine_stats` **weight 0** / **`order_invariant_curriculum` weight 1** / `pickup_stats` **weight 0** / `variety` **weight 0** |
| ⇒ 有效奖励 | **只有 `order_invariant_curriculum` 一项生效**，其余 4 个 monitor 在发布配置里**权重为 0**（未启用），但它们的 items 清单仍在文件里 |
| ★★ 与本地接口的关系 | 这正是「**奖励 = 多键加权和、单项可置 0 关闭**」的一手先例，与我们 `rl/reward.py` 的 `_DEFAULT_REWARD`（16 键，`engagement_trade` 默认 0.0）**接口同构**（**只对结构，不对数值**） |
| 复现 | `https://openaipublic.blob.core.windows.net/minecraft-rl/models/rl-from-early-game-2x.model`（HTTP 200，**4,369 B**）、`rl-from-house-2x.model`（**3,773 B**）、`foundation-model-1x.model`（1,861 B）；md5：`2x.model` == `rl-from-early-game-2x.model` == `1290cde10558881d83814a42ef2bc102` |

#### 1.1.1 论文侧（Table 7）与代码侧（`.model`）逐项对照

**`order_invariant_curriculum.args.curriculum` 逐字（early-game / foundation；格式 = `item: [数量上限, 每件奖励]`）**

| item | 代码 `[数量, 奖励]` | 化简 | 论文 Table 7 | 一致？ |
|---|---|---|---|---|
| log | [8, 0.125] | 1/8 | 8 / 1/8 | ✅ |
| planks | [20, 0.05] | 1/20 | 20 / 1/20 | ✅ |
| stick | [16, 0.0625] | 1/16 | 16 / 1/16 | ✅（**house 版为 [12, 1/12]**） |
| crafting_table | [1, 1] | 1 | 1 / 1 | ✅ |
| wooden_pickaxe | [1, 1] | 1 | 1 / 1 | ✅ |
| cobblestone | [11, 0.09090909090909091] | 1/11 | 11 / 1/11 | ✅ |
| stone_pickaxe | [1, 1] | 1 | 1 / 1 | ✅ |
| furnace | [1, 1] | 1 | 1 / 1 | ✅ |
| coal | [5, 0.4] | 2/5 | 5 / 2/5 | ✅（**house 版为 [5, 0.2] = 1/5**） |
| torch | [16, 0.125] | **1/8** | 16 / **1/8** | ✅（**与 §1.1 里那条「数量 16 却给 1/8」的观察完全一致 ⇒ 说明这不是笔误，代码里就是这个值**） |
| iron_ore | [3, 1.3333333333333333] | 4/3 | 3 / 4/3 | ✅ |
| iron_ingot | [3, 1.3333333333333333] | 4/3 | 3 / 4/3 | ✅（**house 版为 [3, 1/3]**） |
| iron_pickaxe | [1, 4] | 4 | 1 / 4 | ✅（**house 版为 [1, 1]**） |
| diamond | [10000, 2.6666666666666665] | 8/3 | **inf** / 8/3 | ⚠️ **语义等价、数值不同**（论文写 inf、代码写 10000） |
| diamond_pickaxe | [10000, 8] | 8 | **inf** / 8 | ⚠️ 同上（**house 版为 [1, 1]**） |
| **obsidian** | **[10000, 16]** | **16** | **论文 Table 7 无此项** | ❌ **代码配置比论文表多一项**（来源未验证） |

⇒ **三条硬结论**：① 论文 Table 7 与官方配置**逐项一致**（含 torch 的 1/8）；② **无任何额外奖励缩放系数**（最小正奖励 = `1/20 = 0.05`）⇒ **再次否证「×60」**；③ house 变体的「每件奖励一律 = 1/quantity」与论文 §G.2 原句**逐字吻合**（coal 1/5、torch 1/16、iron 1/3、diamond 1/3、stick 1/12）。

#### 1.1.2 论文侧奖励项清单

| # | 奖励项 | 数值 | 出处 |
|---|---|---|---|
| R0 | **奖励形态** = 「按 item 成就给奖励」，不是终端胜负奖励；靠近序列末端的 item 给更高奖励、需大量采集的给低奖励 | 原句 "Agents are rewarded for each item obtained in the sequence, with lower rewards for items that have to be collected in bulk and higher rewards for items near the end of the sequence." | VPT §4.4（`vpt.txt:217-218`） |
| R1 | **4 档基础奖励** | tier1（木/石）= **1**；tier2（需煤）= **2**；tier3（需铁）= **4**；tier4（钻石）= **8** | 附录 G.1（`vpt.txt:1147-1149`） |
| R2 | **★ 除以该 item 的总奖励数量**（防偏科） | 每 item 奖励 = 基础奖励 ÷ 该 item 总奖励数。反推法构造奖励表：从钻石镐沿科技树反推 → 再加煤/火把 → 最后额外奖励原木（需求 5、奖励到 **8**） | 附录 G.1（`vpt.txt:1150`、`1092-1097`） |
| R3 | **无上限项（inf）** | 钻石总量记 **3**、钻石镐记 **1** 用于除数量，但**实际不限量**（鼓励反复挖/合）；代码侧写作 `10000`（§1.1.1） | 附录 G.1（`vpt.txt:1097`、`1150`） |
| R4 | **稀疏性量级** | 即使前置全齐，也可能要**上千动作**才见下一个奖励（"human players often take more than 10,000 actions to find a diamond after crafting an iron pickaxe"） | 附录 G.1（`vpt.txt:1152`） |
| R5 | **欺骗性（deceptive）** | 拿 cobblestone 最快 = 造完木镐立刻下挖、把工作台留在原地 ⇒ 后续做石镐更难 | 附录 G.1（`vpt.txt:1153-1157`） |

**★ Table 7 论文原文（附录 G.1 / `vpt.txt:1100-1145`，实为「奖励数量 / 每 item 奖励」两列，无 `tier` 列）**

| item | 奖励数量（quantity rewarded） | 每 item 奖励 |
|---|---|---|
| Log | 8 | 1/8 |
| Planks | 20 | 1/20 |
| Stick | 16 | 1/16 |
| Crafting table | 1 | 1 |
| Wooden pickaxe | 1 | 1 |
| Cobblestone | 11 | 1/11 |
| Stone pickaxe | 1 | 1 |
| Furnace | 1 | 1 |
| Coal | 5 | 2/5 |
| Torch | 16 | **1/8** |
| Iron ore | 3 | 4/3 |
| Iron ingot | 3 | 4/3 |
| Iron pickaxe | 1 | 4 |
| Diamond | **inf** | 8/3 |
| Diamond pickaxe | **inf** | 8 |

> ✅ **Torch 行已由官方 `.model` 配置消解**（§1.1.1）：代码里逐字就是 `torch: [16, 0.125]` ⇒ **16 × 1/8 = 2 = tier2 基础值**，与「基础奖励 ÷ 数量」自洽，**不是笔误**。
> ⚠️ **版本差异（§1.2-G.2）**：house 变体的「每 item 奖励」全部等于 `1/quantity`，故其 coal=1/5、torch=1/16、iron=1/3、diamond=1/3、stick=1/12 —— **论文 Table 7 只给主实验版本**（§1.1.1 已并列）。

### 1.2 惩罚项：KL 到冻结先验 = **正则项（辅助损失）**，不是奖励侧惩罚

| # | 问题 | 一手答案 | 出处 |
|---|---|---|---|
| P1 | **形式** | `L_klpt = ρ · KL(π_pt, π_θ)`（式 2）；`π_pt` = frozen pretrained policy，`π_θ` = 被训练策略，`ρ` = 权重 | 附录 G.1 式(2)（`vpt.txt:1080-1083`） |
| P2 | **算惩罚项还是正则项？** | **正则项/辅助损失项**：英文用 "auxiliary … loss"（§4.4 与 G.1 两处），式(2) 与「值函数损失、辅助值函数损失」并列由 Table 6 给系数；它**不进环境奖励**，agent 的 reward 全程只有 §1.1 的 item 成就 | VPT §4.4（`vpt.txt:221-222`）；附录 G.1（`vpt.txt:1036`、`1079-1083`） |
| P3 | **它替代了什么？** | **替代熵正则**：明文 "this KL divergence loss **replaces** the common entropy maximization loss"；理由：熵最大化在「状态/动作空间大 + 奖励稀疏」时不可行，改为「在等价值状态下模仿先验动作分布」 | 附录 G.1（`vpt.txt:1084-1088`） |
| P4 | **随机初始化跑 RL 时呢？** | **无 KL、无 KL decay**，改为 entropy bonus **0.01** | Table 6 注释（`vpt.txt:1078`） |
| P5 | **是否还有其他惩罚项？** | **没有**：全文 grep（`penalt` / `negative reward` / `punish` / `bonus` 四词）只命中 ① 上述 "entropy bonus" 与 ② 参考文献/公式排版噪声；环境侧 `Bonuses` 与 `Penalties` 均为 0（`agent_start` 一律 `reward=0, penalty=0`，`agent_handlers.py:13-30`） | `vpt.txt` grep（本报告 §4 方法）；`white_gravel.py:13-30` |
| P6 | **死亡是否终止/惩罚？** | **不终止、无惩罚**：原文 "We do not terminate the episode on agent 'death'. Instead … the agent drops all its items when it dies and respawns"；策略隐状态**不被 mask** | 附录 C.1（`vpt.txt:910`） |
| P7 | **奖励是否被调度/退火？** | **没有**：奖励表构造一次即固定（原文 "the reward function was based on human expectations…"）。**衰减的只有 ρ，不是奖励**；`.model` 配置里奖励表是**静态字典**（§1.1.0） | 附录 G.1（`vpt.txt:1092-1097`、`1089-1091`）；`.model` 反序列化 |
| P8 | ★ **ρ 的出处层级** | **ρ 只能引论文 Table 6**（0.2 / ×0.9995；§G.2 对照为 0.4）。**官方代码与 `.model` 里 ρ 完全不可见**（全仓 `rho` / `kl_prior` / `prior` 均 0 命中；`.model` 无 KL 键）⇒ **ρ 无法与代码交叉验证** | `vpt.txt:1074-1077`、`:1163`；官方仓 grep（§4-S2）；`.model` 反序列化（§1.1.0） |

### 1.3 塑形与调度

| # | 项 | 值 / 事实 | 出处 |
|---|---|---|---|
| S1 | **ρ 初值** | **0.2** | Table 6（`vpt.txt:1074-1075`） |
| S2 | **ρ 衰减率** | **0.9995**，**每次 iteration** 乘一次（decay，不是固定 β） | Table 6（`vpt.txt:1076-1077`） |
| S3 | **ρ 的两难（原文）** | "a **high** coefficient ρ … would prevent the agent from properly optimizing the reward function while a **low** coefficient ρ was ineffective at protecting the learned skills" ⇒ 故「先高后衰减」 | 附录 G.1（`vpt.txt:1089-1091`） |
| S4 | **是否替代熵正则** | **是**（见 P3）；RL-from-scratch 才用 0.01 熵 bonus | 附录 G.1（`vpt.txt:1084-1088`） |
| S5 | **无 KL 消融臂的超参** | KL 设 0 **且 lr 从 2e-5 降到 3e-6**（5 个 lr 扫描里的最佳）；作者注释：KL 本身限制了单步策略变化 ⇒ 有 KL 才能用高 lr | Table 6 注释（`vpt.txt:1078`） |
| S6 | **附录 G.2 对照实验** | 两臂 KL 系数设 **0.4**、lr **6e-5**、每 item 奖励统一 `1/quantity`（取消 tier 递增） | 附录 G.2（`vpt.txt:1163`） |
| S7 | **★ VPT 是否奖励归一化？** | **奖励本身不归一化**；只对**值函数目标**做归一化（见 N1）。这是两件事，不得混写 | 附录 G.1（`vpt.txt:1041`） |
| S8 | **迭代结构** | 1 个 iteration = **1 个 wake cycle + 2 个 sleep cycle**；PPG sleep 的 KL 系数 1.0（这是**睡眠期锁策略**的 KL，与 ρ 项不同名不同值） | 附录 G.1（`vpt.txt:1037`）；Table 6（`vpt.txt:1070-1071`） |

### 1.4 归一化三问（逐问回答）

| 问 | 答 | 证据 |
|---|---|---|
| **N1 奖励归一化？** | **奖励值本身不归一化**。归一化的是**值函数的目标**："we normalize the **value-function target** by subtracting the mean and dividing by the standard deviation, which are estimated through an **exponentially weighted moving average**"（目的：让值函数损失的梯度不强烈依赖奖励量级） | 附录 G.1（`vpt.txt:1041`） |
| **N2 优势归一化？** | **论文未提**。全文无 `advantage normalization` / `normalize advantage` 表述（grep 命中 `advantage` 仅 GAE 叙述与参考文献）。**未在论文核到** | 附录 G.1（`vpt.txt:1026-1029`）；§5-U2 |
| **N3 观测/输入归一化？** | 有，但属**网络输入**层：`lib/policy.py` 的 `_normalize_image`（图 22-25）；这属 IL/RL 共用主干，**非奖励侧** | 官方仓 `lib/policy.py:22-25` |

### 1.5 value head 设计（★ 论文 + 官方仓双向一致）

| # | 事实 | 证据（论文 / 代码） |
|---|---|---|
| V1 | **单标量线性头**：每个值函数 = 预训练模型最后一个 residual transformer block 之上的**单个全连接层** | 论文附录 G.1（`vpt.txt:1039`）；代码 `lib/policy.py:237-238` `make_value_head → ScaledMSEHead(v_out_size, **1**)` |
| V1b | ★ **无离散化/categorical 头**（本轮 grep 复核） | `two_hot` / `hl_gauss` / categorical-value 在官方仓 **0 命中**；`lib/scaled_mse_head.py:24` 就是 `nn.Linear(input_size, output_size)`，`output_size=1` |
| V2 | **不是**离散化/categorical/two-hot/HL-Gauss —— 输出维度 = **1**，损失 = **MSE**（在归一化空间算） | 代码 `lib/scaled_mse_head.py:43` `F.mse_loss(prediction, self.normalizer(target))` |
| V3 | **EWMA 归一化头**：`ScaledMSEHead` 自带 `NormalizeEwma`，"scales itself so that targets are always normalized to N(0, 1)"；推理时 `denormalize` | 代码 `lib/scaled_mse_head.py:11-13, 45-50`；`lib/policy.py:305, 323` |
| V3b | ★ EWMA 的 `beta` 值（**注意别与 ρ decay 混淆**） | `lib/normalize_ewma.py:9` 逐字 `norm_axes=2, beta=0.99999, per_element_update=False, epsilon=1e-5`；forward 内先 `detach` 再按 `weight=self.beta` 更新 running mean/var，并有 `debiasing_term` 去偏。⚠️ **`0.99999`（EWMA）≠ 论文的 `0.9995`（ρ decay）**，论文未给 EWMA beta |
| V4 | **两个值头（PPG）**：regular value head（**零初始化**）+ auxiliary value head（**随机初始化**，其输出训练中不使用，只喂共享表征）；三者损失系数在 Table 6 | 论文附录 G.1（`vpt.txt:1032-1036, 1039-1040`）；Table 6（`vpt.txt:1066-1069`） |
| V5 | **零初始化的理由**（原文） | "which appeared to prevent destructive updates early in training"（`vpt.txt:1040`） |
| V6 | **值头是否吃 KL 项？** | 论文未写；代码侧 `get_kl_of_action_dists` 是 `pi_head.kl_divergence` 的封装（**VPT 仓内无人调用**、**官方 .model 里也无 KL 键**）；同一封装被 **BASALT BC 基线当作 KL 锚调用**（§2.5-K1）。⇒ **VPT 的 KL 训练回路未公开**；出处：`lib/policy.py:281-285`（+ `lib/action_head.py:211-220`） |

### 1.6 有 KL vs 无 KL（消融结果，作机制证据，不作数值对拍）

| # | 事实 | 出处 |
|---|---|---|
| A1 | **无 KL 臂**：只学会早期 4 个 item（logs/planks/sticks/crafting table），**不再有新 item**；进度在 **100,000 episodes** 后**停滞**（"progress stalls after 100,000 episodes, suggesting that the skills necessary to make further progress have been catastrophically forgotten"） | §4.4（`vpt.txt:231`）；附录 G.2 Fig.16（`vpt.txt:1165`） |
| A2 | **有 KL（主实验）**：iron pickaxe **>80%**、钻石 **~20%**、钻石镐 **2.5%**；人类对照 57%/15%/12% | §4.4（`vpt.txt:234-238`） |
| A3 | **回报量级**（仅作量级参照）：RL-from-scratch ≈ 0、RL from foundation ≈ **13**、RL from early-game ≈ **25** | §4.4 Fig.7a（`vpt.txt:230`） |
| A4 | **从零 RL 的失败**：随机初始化策略 "fails to achieve almost any reward"、"never learns to reliably collect logs"（偶因打树叶拿到 stick） | §4.4（`vpt.txt:222`、`230`） |
| A5 | **规模/成本**：PPG、~248M 参数、~1.3M 集（≈1.4×10¹⁰ 帧）、每集 10 分钟、~6 天 / 80 GPU + 56,719 CPU、~4,000 iter | §4.4（`vpt.txt:218`）；附录 G.2（`vpt.txt:1158`） |

---

## §2 MineRL / BASALT 表

> **★ 归类纪律（本次更正）**：`Navigate/Treechop/Obtain*` 与 `ObtainDiamond` **不是同一套奖励**，必须按**来源**分列——
> ① **数据集论文（1907.13440）= 各任务的环境奖励定义**；
> ② **2019 竞赛（2003.05012）= ObtainDiamond 的「里程碑链式翻倍」奖励**；
> ③ **2020 竞赛（2101.11071）= ObtainDiamond 的「Table 1 里程碑 1→1024」**；
> ④ **env 代码（minerl@dev）= 实际实现值**；
> ⑤ **BASALT（2107.01969）= 不给奖励函数**。
> 既有报告把 ②③ 与 ① 混在一节叙述，易被误读为同一环境的不同说法——**本次并列不调和**。

### 2.1 奖励项清单（逐任务 / 逐来源）

| # | 来源 | 任务 | 数值 / 奖励项 | 证据 | 出处 |
|---|---|---|---|---|---|
| M1 | 数据集论文 1907.13440 | **Navigate** | **sparse**：到达目标 | **+1**，到达即终止 | `minerl_ds.txt:174` |
| M2 | 同上 | **Navigate (Dense)** | **dense**：与「朝目标移动的距离」成正比 | 比例奖励（论文未给系数；代码 `reward_per_block=1.0`，`density=PER_TICK`） | `minerl_ds.txt:174`；代码 `navigate_specs.py:46-48` |
| M3 | 同上 | **Treechop** | 每获得 **1 单位木头** | **+1 / 单位**，拿满 **64** 单位终止 | `minerl_ds.txt:179` |
| M4 | 同上 | **Obtain\***（IronPickaxe / Diamond / CookedMeat / Bed） | 获得目标物品 | **+1**，获得即终止 | `minerl_ds.txt:187` |
| M5 | 同上 | **Survival** | 论文明确**「没有已知奖励函数」**，只按游玩时长给奖励以免强加人为奖励 | — | `minerl_ds.txt:134-135` |
| M6 | 2019 竞赛 2003.05012 | **ObtainDiamond** | **里程碑链式**：首次达成每个前置子任务都给奖励，**每级 = 前一级 ×2，起点 1**；**例外**：最终拿到钻石 = 前一级的 **×4** | 原句 "An agent receives twice the reward as received for accomplishing the previous subtask (starting from a reward of 1). The exception … obtaining a diamond: … worth four times as much" | `minerl2019.txt:132-138` |
| M7 | 2020 竞赛 2101.11071 | **ObtainDiamond** | **Table 1 里程碑表**（首次获得即触发） | log 1 / planks 2 / stick 4 / crafting_table 4 / wooden_pickaxe 8 / stone 16 / furnace 32 / stone_pickaxe 32 / iron_ore 64 / iron_ingot 128 / iron_pickaxe 256 / **diamond 1024** | `minerl2020.txt:380-410`（Table 1 逐行） |
| M8 | 同上 | ObtainDiamond 的**形态描述** | 「拿到钻石给高奖励 + 前置 item 给较小的辅助奖励」 | 原句 "The agent receives a high reward for obtaining a diamond and smaller, auxiliary rewards for obtaining prerequisite items." | `minerl2020.txt:261` |
| M9 | 同上 | **Navigate**（当届辅助环境） | **sparse：+100**，到达即终止 | 原句 "The agent is given a sparse reward (+100 upon reaching the goal, at which point the episode terminates)." | `minerl2020.txt:273` |
| M10 | 同上 | **Navigate (dense)** | 每 tick 给「与目标距离的变化量」 | 原句 "receives reward every tick corresponding to the change in distance" | `minerl2020.txt:274` |
| M11 | 同上 | **Treechop** | 每单位木头 +1，拿满 64 或步限终止 | — | `minerl2020.txt:277` |
| M12 | env 代码 minerl@dev | **ObtainDiamondShovel-v0**（v1.0.2 全仓仅存此 Obtain） | 里程碑奖励表**硬编码** | 与 M7 同：`["oak_log"…],1` / planks 2 / stick 4 / crafting_table 4 / wooden_pickaxe 8 / cobblestone 16 / furnace 32 / stone_pickaxe 32 / iron_ore 64 / iron_ingot 128 / iron_pickaxe 256 / diamond 1024 / **diamond_shovel 2048**（最后一项触发 done） | `obtain_specs.py:12-26`、`:38-55` |
| M13 | env 代码 | **Navigate-v0 / NavigateDense-v0** | sparse = `RewardForTouchingBlockType(diamond_block, reward=100.0, onceOnly)`；dense 追加 `RewardForDistanceTraveledToCompassTarget(reward_per_block=1.0)` | 与 M9/M10 逐项一致 | `navigate_specs.py:40-48`；`envs.py:22-25` |
| M14 | env 代码 | **Treechop-v0** | **每单位木头 +1**（可重复计） | `RewardForCollectingItems([{type:"log", amount:1, reward:1.0}])`（**非** `…ItemsOnce`，即**可重复**按量计）；`max_episode_steps=8000`、`reward_threshold=64.0` | `@dev` `treechop_specs.py:46-51`；v0.4.4 `treechop_specs.py:130-135`、`L37/L127` |
| M15 | env 代码 | **BASALT 四任务** | **无奖励函数**（环境结构上不提供） | **`create_rewardables()` 返回空元组 `()`** ⇒ 环境无奖励；docstring 明写 "Basalt environment have no rewards, so this is always False." | `basalt_specs.py:214-222`；四个 spec 见 `envs.py:32-35` |
| M16 | BASALT 论文 2107.01969 | 四任务 | **不提供奖励函数**（任务定义 = 一段自然语言） | **明确不提供奖励函数**：与 MineRL Diamond 的差异条文写 "The MineRL Diamond challenge specifies tasks that include reward functions for all tasks, whereas **our tasks are explicitly designed to not include them**."；任务定义 = 一段自然语言 | `basalt.txt:116`；`basalt.txt:77`、`85`（"without any associated reward function"） |
| M17 | **env 代码（★ 代际差异，本次新增）** | `MineRLObtainDiamond-v0` 的 **sparse/dense 语义** | **sparse/dense 语义**（代际差异） | **2019/2020 竞赛版（`v0.3.7`/`v0.4.4`）**：`dense` 与 `sparse` **共用同一张 schedule**，区别只在「**每次获得都算**」(`RewardForCollectingItems`) vs 「**每种物品只算第一次**」(`RewardForCollectingItemsOnce` + `seen_dict` 去重）；原句 `self.reward_text = "every time it obtains an item"` / `"only once per item the first time it obtains that item"`。**竞赛环境 `comp_envs` 全部取 `dense=False`（= sparse）**，dense 变体另有注册但不在竞赛列表内。**2021 后（`v1.0.2`/`@dev`）**：Obtain 只剩 `ObtainDiamondShovel-v0`，其 `dense` 是**构造函数参数**（默认 sparse），奖励表增加 `diamond_shovel 2048` 并在该项 `done=True` | v0.4.4 `obtain_specs.py:36-39`（原句）、`:95-102`（handler 选择）、`envs.py:22-42`（`comp_envs` 四环境全 `dense=False`）；v0.3.7 `obtain_specs.py:37-44`、`envs.py:16-45`（同上）；`@dev` `obtain_specs.py:12-26, 38-55` |
| M18 | env 代码（★ 本次新增） | **竞赛评测的成功判定（代码侧）** | **竞赛评测的成功判定（代码侧）** | 2019/2020 版 `determine_success_from_rewards` 用 `allow_missing_ratio = 0.1`；Navigate 的 `reward_threshold = 100.0`（dense 时 `+60`） | v0.4.4 `obtain_specs.py:143-155`、`navigate_specs.py:113-117` |
| M19 | env 代码（★ 本次新增） | **唯一可为负的奖励项** | **唯一可为负的奖励项** | 只有 **dense Navigate** 的距离项：`RewardForDistanceTraveledToCompassTarget.from_universal` 返回 `self._prev_delta − delta`（靠近为正、**远离为负**）；docstring 逐字 "…how much closer (**or negative reward for farther**) the agent gets to the target" | v0.4.4 `hero/handlers/agent/reward.py:274-287`、`navigate_specs.py:142`；`@dev` 同构 `reward.py:244-258` |
| M20 | **2020 竞赛论文 §2.2 Rules（★ 本次新增）** | **明文禁止人工硬编码 shaping，但允许 IRL / curiosity** | **规则条文（逐字）** | 逐字（父条目 "The submission must train a machine learning model without relying on human domain knowledge." 下第一子条）："The reward function may not be changed (shaped) based on **manually engineered, hard-coded functions of the state**. For example, additional rewards for approaching tree-like objects are not permitted, but rewards for encountering novel states ("**curiosity rewards**") are permitted."；§2.2 开篇另写 "we discourage the use of environment-specific, hand-engineered features" | `minerl2020.txt:579-581`；同篇 §1.4.1 `:371` 明确把 "reward shaping **via inverse reinforcement learning**" 列为解法之一 |
| M21 | **2019 竞赛提案（arXiv:1904.10079v3）§2.2 Rules（★ 本次新增）** | **同一句禁令在 2019 竞赛提案中已存在** | **规则条文（逐字）** | 逐字同上；该届 §2.2 另有 "A manually specified policy may not be used as a component of this model." | `minerl2019v3.txt:579-581`、`:502` |

### 2.2 「`reward_shaping` 成就 delta」究竟存不存在？（★ 本次首次取到一手代码证据）

| # | 问题 | 答案 | 证据 |
|---|---|---|---|
| C1 | MineRL 论文里 grep `reward_shaping` | **0 命中**（三篇论文全文）；仅 2020 报告出现过一次泛述 "reward shaping"（指「需 IRL 塑形」的方向性论断，不是 API） | `minerl2020.txt:371`（泛述）；本报告 grep |
| C2 | **代码里 grep `reward_shaping`** | **`minerllabs/minerl@dev` 全仓 `grep -rn "reward_shaping" --include=*.py` = 0 命中**（本报告独立复核） | 全仓 tarball `/tmp/mr/src/minerl-dev/` |
| C3 | 那么「成就 delta」在哪实现？ | 在 **Malmo reward handler** 层，且**靠 XML 模板 + `diff.changes` 的 `quantity_change`**：`RewardForCollectingItems`（`sparse=False`，按量累加奖励）与 `RewardForCollectingItemsOnce`（`sparse=True`，用 `seen_dict` 保证**每个 item 只给一次**） | `hero/handlers/agent/reward.py:95-146`（逐字见 §4.1） |
| C4 | **delta 的定义** | `total_reward += change_json['quantity_change'] * reward_dict[item]['reward']`，**只在 `quantity_change > 0` 时**累加 ⇒ 拿到物品为正、**丢弃/销毁不产生负奖励** | `reward.py:110-118` |
| C5 | 有 `+1/-1` 形式的奖惩吗？ | **没有**：reward handler 只有 5 个类（CollectingItems / CollectingItemsOnce / MissionEnd / TouchingBlockType / DistanceTraveledToCompassTarget），**无任何负值常量**；唯一可为负的是 dense Navigate 的距离项（§2.1-M19）。`RewardForMissionEnd.from_universal` 直接 `return 0`；全仓 `AgentQuitFromDying` = **0 命中**；`AgentQuitFromPossessingItem` / `AgentQuitFromCraftingItem` 只做**终止**不给奖励 | `reward.py:73-260`（v0.4.4 行号 `:185/225/269`）；全仓 grep（5 个 ref：`dev`/`master`/`v0.4.4`/`v0.3.7`/`v1.0.2`） |
| C5b | ★ **代际差异**：`master`/`v0.3.7` 用另一个文件 | 老 API 在 `minerl/herobraine/hero/handlers/rewardables.py`，`RewardForCollectingItems.__init__(self, item_dict)` **无** `sparse`/`exclude_loops` 参数 | v0.3.7 `rewardables.py`（本报告独立复核） |
| C6 | 奖测量最小正差分 | Navigate dense 的 `RewardForDistanceTraveledToCompassTarget.from_universal` 返回 `self._prev_delta - delta`（**相对上一 tick 的距离差**，`PER_TICK`）；靠近 ⇒ `delta` 减小 ⇒ 返回**正**，远离 ⇒ 返回**负** | `reward.py:244-258`（`@dev`）；v0.4.4 `:274-287`。⚠️ **符号方向为代码读法，未跑环境验证**（§5-U5） |

### 2.3 惩罚项 / 塑形与调度 / 归一化 / value head

| # | 项 | MineRL / BASALT | 证据 |
|---|---|---|---|
| Q1 | **有惩罚项吗？** | **没有**。env 代码层面：奖励全为**非负常量**，无死亡/生存/时限负奖励；2019/2020 竞赛报告亦无负奖励条文。唯一「负向」机制是**终止**（到达即 done、拿满 64 即 done、拿钻石即 done） | §2.2-C5；`minerl_ds.txt:174-187` |
| Q2 | **奖励是否被调度/退火？** | 未核到任何退火/课程式奖励调度；里程碑表**常量**、`1/2/4/8 …` 表**常量** | `minerl2020.txt:380-410`；`obtain_specs.py:12-26` |
| Q3 | **是否有 dense/sparse 之争？** | **有，而且是一手明文的「禁 hard-code 塑形、允许 IRL 塑形」**（★ 见 Q3b）。两层区分很重要：① **Navigate** 的 sparse（+1 论文版 / +100 竞赛版）与 dense（Δ距离）**都是官方正式提供的两变体**；② **Obtain 系列**的 `dense` **不是 per-tick 塑形**，只是「每次获得都算 vs 每种物品只算一次」 | `minerl_ds.txt:174`；`minerl2020.txt:273-274`；§2.1-M17 |
| Q3b | ★ **竞赛是否禁止 dense/shaping？** | **禁止「人工硬编码」塑形，但不禁 IRL/curiosity**（2019 与 2020 竞赛提案同一句条文，逐字见 §2.1-M20/M21）；**评测侧只用 sparse**——`v0.3.7` 与 `v0.4.4` 的 `comp_envs` 四个环境**全部 `dense=False`** | `minerl2020.txt:579-581`；`minerl2019v3.txt:579-581`；v0.3.7 `envs.py:16/23/26/45`、v0.4.4 `envs.py:20-42` |
| Q4 | **竞赛是否限制「从模拟器提额外信息」？** | 有：「Models may only be trained against the competition environments（…`VectorOb(f)`）」；2020 主赛道样本预算 **8,000,000** 次交互（外加提供的数据集）；预训练模型不许用过 MineRL/Minecraft 数据 | `minerl2020.txt:589`、`:593`、`:605` |
| Q4b | **评测局数** | **2019 报告：Round 2 = 每提交跑 100 局取平均**；**2020 竞赛：500 局取平均**，里程碑取「首次获得」，平局按达标局数分解 | `minerl2019.txt:146`；`minerl2020.txt:410-413` |
| Q5 | **BASALT 的惩罚/奖励** | **全都没有**：无奖励函数、无惩罚；评测 = **人工成对比较 + TrueSkill**，跨 4 任务把 TrueSkill 分数**任务内标准化到 mean 0 / sd 1** 后取平均 | `basalt.txt:137`、`141-142`（Fig.2 + 评测流程）；`basalt.txt:116` |
| Q6 | **归一化** | ① 竞赛评分层面：**TrueSkill 跨任务标准化**（唯一归一化，属**评测**不是训练）；② 训练层面：论文/报告**未描述**奖励或优势归一化（标 §5-U7） | `basalt.txt:142` |
| Q7 | **value head 设计** | MineRL/BASALT 论文**未给** RL 基线的网络细节；官方 BC 基线是 **VPT 1x 权重微调**（其 `openai_vpt/` 内 vendored VPT 代码含 `value_head`），但**基线是纯 BC，不训 critic、不使用 reward**（`run_agent.py:27` 把 reward 直接丢弃）⇒ **该基线不存在「RL 的 value head 设计」可谈** | `basalt_readme.md` 首段（"fine-tunes the 'width-x1' models of OpenAI VPT"）；§2.5-K5；§5-U8 |

### 2.4 规模与「人类数据」接口（与本报告 §3 相关）

| # | 事实 | 证据 |
|---|---|---|
| H1 | MineRL-v0 数据集：**>60M** 自动标注 state-action 对、**500+ 小时**人类示范、**六个**任务、每版本 **130 GB**、20 Hz 逐 tick | `minerl_ds.txt` 摘要段 + §（既有报告已核，本次未重取，沿用） |
| H2 | 2019 竞赛：**纯人类示范、零环境交互**的层次化策略拿总榜**第 2**（Round 2 均分 42.41，榜首 CDS 61.61） | `minerl2019.txt`（既有报告 Table 1）；本次复核到 "trained their hierarchical policies entirely from human demonstrations with no environment interactions"（`minerl2019.txt:257`） |
| H3 | 无队伍拿到钻石 ⇒ 2020 沿用同一任务 | `minerl2019.txt:337`（"Because no team obtained a diamond, we plan to focus on the same task, ObtainDiamond"） |
| H4 | BASALT 2022 BC 基线：微调 VPT 1x；数据量 BuildVillageHouse 146G/1399 视频、CreateVillageAnimalPen 165G/2833、FindCave 165G/5466、MakeWaterfall 175G/4230 | `basalt_readme.md`（本次直取，逐字一致） |
| H5 | BASALT 环境之所以能「无奖励」：论文理由是**手写奖励函数本身困难/可被钻空子**，并引 CoastRunners 刷分不完成比赛案例 | `basalt.txt:87` |

### 2.5 ★ 旁证：BASALT 2022 官方 **BC 基线**里存在一个「KL 到冻结起点 ckpt」锚定项（本次新核，对 §3.4 有直接影响）

| # | 事实 | 逐字证据 | 出处（file:line） |
|---|---|---|---|
| K1 | 该基线的训练损失 = **BC 的 NLL + 一个 KL 到「原模型」的正则项** | `behavioural_cloning.py:152` 逐字：`loss = (-log_prob + KL_LOSS_WEIGHT * kl_div) / BATCH_SIZE`；`:43` `KL_LOSS_WEIGHT = 1.0` | `behavioural_cloning.py`:152 / :43 |
| K2 | 「原模型」= **同权重的冻结副本**（锚点是**起点 ckpt 本身**） | `:63-70` 逐字：`original_agent = MineRLAgent(...)` + `original_agent.load_weights(in_weights)`（与 `agent` 同一份 `in_weights`，即 VPT `foundation-model-1x.weights`）；`:142` `kl_div = policy.get_kl_of_action_dists(pi_distribution, original_pi_distribution)` | `behavioural_cloning.py`:63-70 / :142 |
| K3 | **★ 该 KL 项不是 VPT 原创，是 BASALT 基线自己加的**（代码注释原话） | `:42-43` 逐字：`# KL loss to the original model was not used in OpenAI VPT` / `KL_LOSS_WEIGHT = 1.0` | `behavioural_cloning.py`:42-43 |
| K4 | 与 VPT 的两处机制差异 | ① **权重恒定 1.0，不退火**（VPT：ρ=0.2 ×0.9995/iter）；② 它是 **BC 损失的一部分**（VPT 是 RL 微调的辅助损失、替代熵正则）。训练侧**只解冻 `policy.net.lastlayer` + `policy.pi_head`**（`:73-80`） | `behavioural_cloning.py`:73-80；论文 Table 6 |
| K5 | 该基线**不接触任何奖励** | 全仓 `reward_shaping` = **0 命中**、`dagger` = **0 命中**；`run_agent.py:27` 逐字 `obs, _, done, _ = env.step(action)`（**reward 被丢弃**）；`openai_vpt/lib/policy.py` 里 VPT 遗留的 `active_reward_monitors` 是**死参数**（基线从不传入）。且它**不做 KL 消融**（论文 §1.6 未给该基线的 KL 项任何消融） | 同一仓库 grep；`run_agent.py`:27 |
> ★ 出处：全仓 tarball `https://codeload.github.com/minerllabs/basalt-2022-behavioural-cloning-baseline/tar.gz/refs/heads/main`（46,740 B），本地 `/tmp/mr/baserepo/`（本报告独立复核 K1/K2/K3 三行逐字）

> **对 §3.4 的意义**：「冻结起点 ckpt + KL 锚定」这套结构在**同一生态里还有第二个（且更贴我们的）实例**——BASALT 的 BC 基线。但它**同时**给出了反向证据：**这个 KL 项被明确标注为「VPT 没用过」**，说明它在该项目里是**工程选择**而非被消融证明有效的机制；且它**没有退火、没有消融**。⇒ 我们要引用时，既不能说「VPT 的消融支持它」，也不能说「生态惯例如此」，只能说「有一处同生态实现，效力未验证」。

---

## §3 设计意图差异 + 对我们的迁移性

### 3.1 一段话（≤5 行）：两家的设计意图差异

> **VPT** 的意图是「**别把先验弄丢**」：它的瓶颈是「有先验但会被 RL 洗掉」，所以奖励**只描述目标（item 成就）+ 一个把策略抽在冻结起点附近的 KL 正则（替代熵正则、ρ 单调衰减）**，靠先验提供探索、靠 KL 保证先验不被遗忘。
> **MineRL/BASALT** 的意图是「**先验够不够、以及奖励根本写不出来时怎么办**」：难度被显式归因于**稀疏 + 长时域**（2020 报告原文），所以 ① 竞赛沿科技树给**阶梯/倍增的里程碑奖励**（2019 链式 ×2、×4 终止；2020 表 1 1→1024）把稀疏变密，② 同时**明文禁止人工硬编码塑形、但允许 IRL 塑形**（2019/2020 同一句），③ BASALT 干脆**取消奖励函数**、改为**人工成对比较 + TrueSkill**（**不用 KL**）。
> 差别一句话：**VPT 用 KL 抽住先验、奖励是稀疏成就（且用可插拔 monitor 加权和实现）；MineRL 用密集变体/里程碑把稀疏放松，竞赛侧**没有任何 KL 锚定**；但 BASALT 2022 的 **BC 基线**自己加了一个位置相同的 KL 锚（并注明 VPT 没用过）。两家奖励侧的惩罚项都为空，VPT 的 KL 是正则项。**

### 3.2 对我们（手工 22 键奖励表、无人类数据、无 reward model、PPO 无 KL 机制）——**可迁**

| # | 可迁项 | 理由（上游证据 + 本地落点） | 落地形态与前置 |
|---|---|---|---|
| T1 | **「按结构沿链反推 + 分档 + 除以该 item 总奖励数」的奖励表设计法** | VPT 表**构造过程**是通用的（从终局目标沿需求图反推、按阶段分档 1/2/4/8、再除以数量防偏科），与「有没有人类数据」无关（§1.1-R2） | 本地 `DEFAULT_REWARD` 22 键已同构（`crown_weight=8` vs `tower_dmg_*≈0.001`）；可据此复核「是否被大 item 主导」，**只迁设计法，禁止迁数值**（跨项目对拍红线） |
| T2 | **「值函数目标做 EWMA 均值/标准差归一化」而非奖励归一化** | VPT 明确只归值目标、理由是防止值损失梯度依赖奖励量级（§1.4-N1）；官方仓实现为 `ScaledMSEHead`（在归一化空间算 MSE、推理时反归一化） | 本地已有 `v_scale` 与「`value=` 是 ÷`v_scale²`、只看它会读到假下降」的教训（`docs/agents` 台账）；可对照检查我们的 value 目标是否等价处理 |
| T3 | **「两个值头：一个零初始化主头 + 一个从不使用的 aux 头只喂共享表征」** | VPT 附录 G.1 给机制与理由（零初始化防早期破坏性更新）；属**纯网络结构**改动，不需要人类数据 | 本地价值通路已多次改造（`value_independent` / `value_bypass` / `value_head_mlp`，`config.py:300-307`），可作**候选结构**；但按【R6】须 `--fresh` + 单变量预注册 |
| T4 | **「把 KL 当正则项并替代熵正则」这一「二选一」的取舍逻辑** | VPT 一手消融：无 KL ⇒ 早期技能被忘、100k 集停滞；且明文 *replaces* entropy（§1.2-P3、§1.6-A1）。**这条可迁的是「替代关系」而不是「必须加 KL」** | 本地 PPO **只有熵项、无 KL**（`rl/ppo.py:393` + `grep -c kl = 0`）⇒ 若将来做锚定，正确先例是 **VPT**（有消融），不是 RLHF 的 KL 防 RM 钻空子 |
| T5 | **「里程碑/分档密集化」的思路（MineRL 侧）** | MineRL 2019 把 ObtainDiamond 沿科技树倍增（1,2,4,…×2，终局 ×4）以缓解稀疏（§2.1-M6）；这与我们「破塔里程碑 `crown_weight=8`」同构 | 可直接用作我们**奖励表分档**的对照基准（**只对结构、不对数值**） |
| T6 | ★ **「奖励 = 多个具名 monitor 的加权和，单项可置 0 关闭」这一接口形态**（本轮新发现） | VPT 官方 `.model` 里逐字是 5 个 monitor + 各自 weight，发布配置**只开 1 个**、其余 4 个为 0（§1.1.0）⇒ 与我们 `_DEFAULT_REWARD` 多键 + `engagement_trade` 默认 0.0 的写法**同构**，且提供「**留好开关、默认关、单变量开关**」的一手先例 | 与本地【R2】（新能力只经 `TrainConfig` 显式开）**方向一致**，可作为该纪律的外部实现参照（**不引入任何新数值**） |

### 3.3 对我们——**不可迁**

| # | 不可迁项 | 理由 |
|---|---|---|
| X1 | **IDM 给海量无标签视频打伪标签 → BC 这条整管线** | 前提是「~270k h 原始 / ~70k h 干净无标签视频 + 少量标注（100 h 即可用）」。我们**两边都没有**（无人类演示、无视频池）⇒ VPT 阶段一在我们场景**不存在** |
| X2 | **「锚住人类先验」的语义** | VPT 的 `π_pt` = 人类视频 BC 出来的行为先验，其可保护的对象是**人类技能链**（smelt iron 的全序列）；我们无人类数据 ⇒ 无论锚什么，都不存在「人类技能」这个被保护物 |
| X3 | **VPT item-tier 的数值本身（1/2/4/8、4/3）** | 是 Minecraft 科技树量纲的产物；与卡牌/圣水/塔血量纲不同源。⇒ 只迁设计法，禁迁数值 |
| X4 | **BASALT「不做奖励函数、改人工成对比较」** | 前提是**有真人标注预算**（要求 <30 min 指令、还有 compute/人力的参赛上限）。我们无人类数据 ⇒ 只能做代理评判，而代理=奖励表本身时构成循环（本地已多次实测该循环的假信号） |
| X5 | **PPG + 248M + 720×V100/9 天、batch 48×40、context 128、1.4×10¹⁰ 帧** | 算力差 2–3 个数量级；【R15】禁止跨实验照抄阈值 |

### 3.4 ⚠️ 特别回答：把 VPT 的「KL 到冻结先验 + ρ 衰减」中的先验换成**早期自对弈 ckpt**，原文有无证据支持？

**结论（明确）：分三半答，不能一句话答「有」或「没有」。**

0. **先消歧：「换先验」这个词有歧义，两种读法结论相反。**
   - 读法 A = 「**锚在冻结的『RL 起点』ckpt**（该 ckpt 可以是任一更早/更弱的策略）」 → **有直接证据**（见 1）。
   - 读法 B = 「**锚在『自对弈产物』ckpt**（用 RL 自己生成的数据训出来的策略当锚）」 → **无证据**（见 2）。
   - 读法 C = 「**把 ρ·KL 这套锚定机制搬到 BC/我们的 IL 侧**」 → **生态里有第二例（BASALT 2022 BC 基线），但它明文标注『VPT 没用过』且无消融**（见 3）。

1. **读法 A：`π_pt` = 冻结的「RL 微调起点」ckpt —— 有直接证据（且原文主实验就是这么做的）。**
   - 原文定义只有一句：`π_pt` = "the **frozen pretrained policy**"（附录 G.1 式 2 说明，`vpt.txt:1083`）。
   - **主实验的起点是 `early-game model`**（VPT 基础模型再 BC 微调到 earlygame_keyword 数据集得到的 ckpt），据 §4.4 与附录 G.2：RL 从 early-game 模型出发效果最好 ⇒ 主实验的 `π_pt` **就是这个 early-game ckpt**（`vpt.txt:235`、`1162-1166`）。
   - 附录 G.2 的对照实验里，**两臂各自把 KL 锚到各自的起点模型**（house-building model / early-game model，KL 系数 0.4）（`vpt.txt:1162-1166`）⇒ 同一套 KL 机制在**两个不同的冻结起点 ckpt** 上都被使用过。
   - ★ **本轮新增的第三条证据（官方配置侧）**：`.model` 里 RL 奖励是**可插拔 monitor 加权和**，且同一套奖励表在 early-game 与 house **两个起点模型上都被使用**（§1.1.0/§1.1.1）⇒ 「换起点 ckpt 不换机制」在**代码侧**也成立。
   - ⇒ 「锚在冻结的早期 ckpt」**结构上是原文原样**。**既有报告的表述「VPT 的 `π_pt` = 人类视频 BC 出来的基础模型」不够准确**（主实验锚的是 BC **微调后**的 early-game ckpt），本次更正（§6-E4）。

2. **读法 B：换成自对弈（RL 自生成产物）ckpt —— 原文完全没有证据。**
   - 原文中所有 ckpt 的来源都是**人类视频的 BC**（foundation / house / early-game 三类，均由 IDM 伪标签 + 承包商数据训出），**不存在**任何自对弈/自我生成数据产生的锚；
   - 全文与官方仓/`.model` 中均未见把锚设为自产策略的讨论；也**没有**「不同锚来源对比」的消融（house vs early-game 的对比是描述性的，且结论是**两者接近**，见 `vpt.txt:1166`）；
   - 机制前提也不成立：KL 项的作用对象是「**已经存在于先验中、但零样本概率低、因而看不到奖励、会被遗忘的技能**」（Fig.16 说明逐字，`vpt.txt:1165`）。**无人类数据时锚对象是早期自对弈策略，它没有「人类技能」可保护**；能保护的是「我们自己早期的行为分布」，而该分布是否有值得保护的长程技能，原文**无从回答**。
   - ⇒ **「先验换成早期自对弈 ckpt」= 未验证类推**（原文既不支持也不反对，因为**根本没做**）。若要做，必须**单独预注册**，且**不能**引用 VPT 的消融效力作为依据。

3. **读法 C：同一生态里已有第二个实例 —— 但它给出的信号是「这机制在本生态内被当工程选择，不是被证明的机制」。**
   - **BASALT 2022 官方 BC 基线**逐字：`loss = (-log_prob + KL_LOSS_WEIGHT * kl_div) / BATCH_SIZE`，`KL_LOSS_WEIGHT = 1.0`，锚点是**同权重的冻结副本**（`original_agent.load_weights(in_weights)`）（§2.5-K1/K2）。
   - **★ 决定性反证**：该文件里紧挨着的注释逐字写 **"# KL loss to the original model was not used in OpenAI VPT"**（§2.5-K3）⇒ 这是**基线作者自己加的锚**，不是 VPT 机制的复用；且它**不退火、无消融**、只解冻 `lastlayer + pi_head`（§2.5-K4/K5）。
   - ⇒ 对读法 C 的结论：**「KL 锚到冻结起点」在生态里有第二例可援引，但两例的效力证据互不通用**；我们若做，仍须自己的预注册与判据（【R3】【R16】）。

4. **附带一条本次新发现（对读法 A 的限定）**：VPT 的 ρ 衰减**没有独立消融**——原文只给「高 ρ 坏 / 低 ρ 坏 ⇒ 故衰减」的**经验陈述**（`vpt.txt:1089-1091`）；且 **ρ 在官方代码/`.model` 里完全不可见**（§1.2-P8）。⇒ 「先高后衰减」这一**调度本身**在原文属**未做消融的经验选择**，不是被消融证明的结论（既有报告未标这一层，本次补记）。

---

## §4 来源清单（URL + 章节 / 表号 + 取数时间）

全部取数时间 **2026-09-19 04:39–04:47 CST**（UTC 2026-09-18T20:39–20:47Z）。

| # | 来源 | URL | 用途 / 落点 |
|---|---|---|---|
| S1 | VPT 全文 HTML v1（arXiv:2206.11795v1） | https://arxiv.org/html/2206.11795v1 | §1 全部；本地转纯文本 `/tmp/vpt/vpt.txt`（1262 行），行号即 `vpt.txt` 行号 |
| S2 | VPT 官方仓 `openai/Video-Pre-Training@main` | https://github.com/openai/Video-Pre-Training ；tarball `https://codeload.github.com/openai/Video-Pre-Training/tar.gz/refs/heads/main`（52,135 B） | §1.4-N3、§1.5-V1–V6、§5-U4；文件 `lib/policy.py`、`lib/action_head.py`、`lib/scaled_mse_head.py`、`lib/normalize_ewma.py`、`behavioural_cloning.py`、`README.md` |
| S2b | ★ **VPT 官方 `.model` 参数文件**（未在 README 链接；`pickle` 反序列化） | `https://openaipublic.blob.core.windows.net/minecraft-rl/models/rl-from-early-game-2x.model`（4,369 B）、`rl-from-house-2x.model`（3,773 B）、`foundation-model-1x.model`（1,861 B）、`2x.model`（md5 同 early-game） | **§1.1.0 / §1.1.1 的 RL 奖励表与 monitor 权重**；本地 `/tmp/vptrepo/models/` |
| S3 | OpenAI VPT 博客 | https://openai.com/index/vpt/ | ❌ **403 / 9,667 B 挑战页，正文未取到**（§5-U3） |
| S4 | MineRL 数据集论文全文 v1（arXiv:1907.13440v1） | https://arxiv.org/html/1907.13440v1 | §2.1-M1–M5；本地 `/tmp/mr/minerl_ds.txt` |
| S5 | MineRL 2020 竞赛**提案**全文 v1（arXiv:2101.11071v1） | https://arxiv.org/html/2101.11071v1 | §2.1-M7–M11/M20、§2.3-Q3/Q3b/Q4/Q4b；本地 `/tmp/mr/minerl2020.txt` |
| S6 | MineRL 2019 竞赛**结果报告**全文 v4（arXiv:2003.05012v4） | https://arxiv.org/html/2003.05012v4 | §2.1-M6、§2.3-Q4b、§2.4-H2/H3；本地 `/tmp/mr/minerl2019.txt` |
| S7 | BASALT 论文全文 v1（arXiv:2107.01969v1） | https://arxiv.org/html/2107.01969v1 | §2.1-M16、§2.3-Q5/Q6、§2.4-H5；本地 `/tmp/mr/basalt.txt` |
| S8 | **`minerllabs/minerl@dev` 全仓快照** | https://github.com/minerllabs/minerl （默认分支 = **dev**，`setup.py` version **1.0.2**，文件 mtime 2025-01-23） | §2.2-C2–C6、§2.1-M12–M15；本地 `/tmp/mr/src/minerl-dev/`。关键文件：`minerl/herobraine/hero/handlers/agent/reward.py`、`minerl/herobraine/env_specs/{obtain,navigate,treechop,basalt}_specs.py`、`minerl/herobraine/envs.py` |
| S8b | **`minerllabs/minerl` tag `v0.4.4` 全仓快照**（= 2020 竞赛版） | tarball `https://codeload.github.com/minerllabs/minerl/tar.gz/refs/tags/v0.4.4`（**123,831,521 B**） | §2.1-M17/M18/M19、§2.2-C5b、§2.3-Q3b；本地 `/tmp/mr/s044/minerl-0.4.4/` |
| S8c | **`minerllabs/minerl` tag `v0.3.7` 全仓快照**（= 2019 竞赛版） | tarball `https://codeload.github.com/minerllabs/minerl/tar.gz/refs/tags/v0.3.7`（**64,767,192 B**） | §2.1-M17、§2.3-Q3b；本地 `/tmp/mr/s037/minerl-0.3.7/` |
| S8d | 2019 竞赛**提案** arXiv:1904.10079v3（规则条文来源；≠ 结果报告 2003.05012） | https://arxiv.org/html/1904.10079v3 | §2.1-M21、§2.3-Q3b；本地 `/tmp/mr/minerl2019v3.txt` |
| S9 | BASALT 2022 BC 基线 README | https://cdn.jsdelivr.net/gh/minerllabs/basalt-2022-behavioural-cloning-baseline@main/README.md | §2.3-Q7、§2.4-H4 |
| S10 | 本地既有报告（必读素材） | `docs/vpt_minerl_dd_2026-09-19.md`（file:line 见 §6 逐条） | §6 更正与对照 |
| S12 | **BASALT 2022 BC 基线全仓快照**（§2.5 的 KL 锚一手证据） | tarball `https://codeload.github.com/minerllabs/basalt-2022-behavioural-cloning-baseline/tar.gz/refs/heads/main`（46,740 B） | §2.5-K1–K5、§3.4-读法 C；本地 `/tmp/mr/baserepo/basalt-2022-behavioural-cloning-baseline-main/behavioural_cloning.py` |
| S11 | 本地代码（只作量纲锚，非上游源） | `src/clasher_new/rl/config.py:37`（`DEFAULT_REWARD`，**22 键**）、`src/clasher_new/rl/reward.py:28`（`_DEFAULT_REWARD`，**16 键**）、`src/clasher_new/rl/ppo.py:88/120/393`、`src/clasher_new/rl/follower.py:239` | §3.2/§3.3 |

### 4.1 逐字代码证据（§2.2-C3 的 `reward.py` 关键行，`minerl/herobraine/hero/handlers/agent/reward.py`）

```
 95  class RewardForCollectingItems(_RewardForPosessingItemBase):
104      super().__init__(sparse=False, exclude_loops=True, item_rewards=item_rewards)
106      def from_universal(self, x):
107          total_reward = 0
108          if 'diff' in x and 'changes' in x['diff']:
109              for change_json in x['diff']['changes']:
110                  item_name = strip_item_prefix(change_json['item'])
...
113                  if item_name in self.reward_dict and 'quantity_change' in change_json:
114                      if change_json['quantity_change'] > 0:
115                          total_reward += change_json['quantity_change'] * self.reward_dict[item_name]['reward']
116          return total_reward

120  class RewardForCollectingItemsOnce(_RewardForPosessingItemBase):
130      super().__init__(sparse=True, exclude_loops=True, item_rewards=item_rewards)
131      self.seen_dict = dict()
...
137                  if item_name in self.reward_dict and 'quantity_change' in change_json and item_name not in self.seen_dict:
138                      if change_json['quantity_change'] > 0:
139                          total_reward += self.reward_dict[item_name]['reward']
140                          self.seen_dict[item_name] = True
```

---

## §5 未验证项（**不得**当作事实引用）

| # | 未验证项 | 本轮尝试 | 结论 |
|---|---|---|---|
| U1 | **「奖励缩放 60×」/ 任何奖励缩放倍数** | VPT 全文 grep（`scale` / `scaling` / `60` / `reward`）两次独立复核：`scale` 命中均为模型/data scaling 与图像增强；`60` 命中为 "60 consecutive attack actions"、60 分钟、`60` 引用编号；**无任何 `reward × k` 形式** | ❌ **未验证 —— 该数字在原文不存在，禁止引用**（与既有报告一致） |
| U2 | **优势归一化 / advantage normalization** | 论文全文无该表述；官方仓全仓 `.py` 里 `advantage` / `gae` / `discount` / `gamma` / `ppo` / `ppg` / `normalize_advantage` / `ValueNorm` / `whiten` / `reward_norm` / `reward_scale` **一律 0 命中**（`clip` 的命中全是 `clip_grad_norm_` / `np.clip`，**不是 PPO ratio clip**）⇒ 代码里**根本没有** RL 训练回路可查 | ❌ **论文侧未提；代码侧确认不存在实现**（不是「没找到」而是「公开物中不存在」） |
| U3 | **OpenAI 官方 VPT 博客正文** | `https://openai.com/index/vpt/` → **HTTP 403 / 9,667 B 挑战页** | ❌ **未验证** |
| U4 | **官方 VPT 仓是否有 RL 微调代码** | 全仓 **25 个文件**（两个分支同一份清单，53 条提交史）**无 RL 训练脚本**；README 明写 "See the paper … for … the exact reward schedule"，`behavioural_cloning.py` 自查 "This code is a **_not_ the original code used for VPT**"；`lib/policy.py:109` 注释 "Unused argument assumed by **forc**" ⇒ 公开代码是内部 `forc` 框架**裁掉 RL 后的残件** | ✅ **已核到「公开仓不含 RL 训练代码」**；`lib/policy.py:281-285` 的 `get_kl_of_action_dists` + `lib/action_head.py:211-220` 的 `CategoricalActionHead.kl_divergence` 是**可直接复用的 KL 工具**，但**无训练回路**（故 §1.2-P1 的 ρ 只能引论文） |
| U4b | ★ **公开 `.model` 参数文件是 RL 配置的替代来源（本次新发现）** | 6 个官方 `.model` 已下载并反序列化成功；`rl-from-early-game-2x.model`（4,369 B）、`rl-from-house-2x.model`（3,773 B）、`foundation-model-1x.model`（1,861 B） | ✅ **已核**（§1.1.0）：奖励表、monitor 权重、PPG 工厂路径、`pi_head temperature`、动作掩码观测维度均可读；**但 ρ / 优化器超参 / aux head 实现不在其中** |
| U4c | **`.model` 里的 `obsidian: [10000, 16]` 为何不在论文 Table 7** | 论文全文无 obsidian 奖励项；代码两个 RL 配置都有 | ⚠️ **未验证**（不解释为"论文漏写"也不解释为"额外项"） |
| U5 | **Navigate dense 的距离奖励符号方向** | 代码读法为 `prev_delta − delta` ⇒ 靠近为正、远离为负，与论文「proportional to distance moved towards the goal」一致；**但未跑环境验证** | ⚠️ **代码层面已核，行为未验证** |
| U6 | ~~2020 竞赛「禁止/允许奖励整形」的条号级引用~~ → **本轮已解决** | 已取到逐字条文（2019 与 2020 提案同一句，§2.1-M20/M21） | ✅ **已核**（原「未见禁止条文」的表述**已作废**，见 §6-E3） |
| U7 | **MineRL/BASALT 训练侧归一化（奖励/优势）** | 论文与报告未描述；官方 BC 基线是 BC 不训 critic | ❌ **未验证** |
| U8 | **2019 竞赛实际使用的 minerl 版本与代码 tag 的对应关系** | 本轮核了 `v0.3.7`（声称竞赛版）、`v0.4.4`（2020 竞赛版）、`v1.0.2`/`@dev`（当前）；**tag 与「哪一届竞赛实际跑的是哪个 tag」的对应关系未从一手材料核到** | ⚠️ **未验证**（本报告的代际比较按 tag 名与推出时间排布，**不等于**已证实某届用了某 tag） |
| U9 | **VPT Table 7 的 Torch 行（数量 16 / 奖励 1/8）是否为笔误** | ✅ **本轮已消解**：官方 `.model` 逐字为 `torch: [16, 0.125]` ⇒ 与论文一致，**不是笔误**（原「未验证」条目作废，见 §1.1.1） | ✅ **已核** |
| U10 | **VPT ρ 衰减的独立消融** | 原文只有经验陈述（高 ρ 坏 / 低 ρ 坏），**未见「有衰减 vs 无衰减」的对照臂** | ⚠️ **未验证 —— 「先高后衰减」在原文属经验选择，未被消融证明** |
| U11 | **本地 993K 参数量** | 本环境未复算（`import torch` 依既有报告失败） | ⚠️ **沿用给定值，不背书** |
| U12 | **`docs/il_reward_reference_analysis_2026-09-19.md:61` 的 I8 引用「`rl/ppo.py:379-392`」** | 本次 grep：`src/clasher_new/rl/ppo.py` `grep -c -i kl` = **0**；379-392 行是策略损失/值损失/熵项代码 | ⚠️ **该 file:line 引用有误**（KL 在本地未实现；同一文档 §1.2 也只说 FL 有、本地无）⇒ 引用时**只写「本地 PPO 无 KL 机制（`ppo.py:393` 的 loss 式含 `−coef·ent_term`、`grep -c kl = 0`）」** |
| U13 | **ObtainCookedMeat / ObtainBed 的代码级奖励值** | `v0.4.4` 的 `envs.py:33-38` 里这两个只有**注释掉的原型**（`# MINERL_OBTAIN_MEAT_V0 = Obtain(...)`、`# MINERL_OBTAIN_BED_V0 = Obtain(target_item='bed', …, reward_schedule=None)`）；未注册 | ❌ **未验证**（只有数据集论文的 `+1` 字面） |
| U14 | **「死亡终止/死亡罚分」** | 2020 论文称 ObtainDiamond 可因死亡结束，但 `v0.4.4` 全仓 `AgentQuitFromDying` = **0 命中**、`minerl/env/*.py` 无 death 逻辑；Malmo `MissionHandlers.xsd` 有 `rewardForDeath` **默认 0.0** 而 minerl 侧从未设置 | ⚠️ **「死亡不罚分」是据 schema 默认值推断，非显式设置**（死亡终止可能在 Java 侧，未取证） |
| U15 | **`RewardForMissionEnd` 在新 herobraine 路径下是否被注入** | Python env_specs 从未实例化它；仅旧回归基准 XML（`env_specs/test/obtainDiamond.xml:68-70`）有 `<Reward description="out_of_time" reward="0"/>`；且其 `from_universal` 直接 `return 0` | ⚠️ **超时给 0 分未直接验证** |

---

## §6 对既有报告的四组更正（只增不改历史结论）

| # | 既有报告位置 | 原表述 | 本次更正 |
|---|---|---|---|
| E1 | `docs/vpt_minerl_dd_2026-09-19.md:100`（§3.2） | 「`reward_shaping` 作为代码符号（成就 delta）❌ **未验证** … 尝试直取两个官方仓库的具体文件路径全部 404/被中断」 | **已取到一手代码证据**：该符号在 `minerllabs/minerl` 的 **5 个 ref（`dev`/`master`/`v0.4.4`/`v0.3.7`/`v1.0.2`）全仓 grep = 0 命中**（本报告独立复核 `v0.4.4` 与 `@dev`）；成就 delta 由 `RewardForCollectingItems{Once}` 的 `quantity_change × reward` 实现（§2.2-C3/C4）。**「该 API 符号不存在」现在是已核事实，而非未验证**；也**不存在** `minerl/env/reward.py`（真文件是 `minerl/herobraine/hero/handlers/agent/reward.py`） |
| E2 | 同上 §3 标题与叙述（`…:81-111`） | 把 BASALT 2022、MineRL 2019 BC 结论、MineRL 各任务奖励混在「§3 MineRL / BASALT」一节 | **归类更正**：BASALT 2022（BC 基线复用 VPT 权重、无奖励函数）与 MineRL 2019（Diamond 竞赛、里程碑奖励）是**两条不同的竞赛线**；奖励形态也不同（2019 链式 ×2/×4 终止 vs 2020 表 1 1→1024 vs 数据集论文 单任务 +1 / +100）。**并列不调和**：§2.1 按来源分列 |
| E3 | 同上 §3.2 末条（`…:101`） | 「**未核到**任何条文禁止 dense 塑形（2020 报告反而把「用 IRL 做 reward shaping」列为解法之一）」 | **该表述不成立**：竞赛提案 §2.2 Rules **明文禁止**人工硬编码塑形——"The reward function **may not be changed (shaped) based on manually engineered, hard-coded functions of the state** … but rewards for encountering novel states (curiosity rewards) are permitted"（2019 提案与 2020 提案同一句，§2.1-M20/M21）；同篇把 **IRL 塑形**列为允许解法（两处**并列不调和**：禁的是人工硬编码、允的是学出来的）。另：**评测侧只用 sparse**（`v0.3.7`/`v0.4.4` 的 `comp_envs` 四环境全 `dense=False`） |
| E4 | 同上 §2.2（`…:60`） | 「`L_klpt = ρ·KL(π_pt, π_θ)`，`π_pt` = 冻结的预训练策略」未说明主实验的 `π_pt` 具体是哪个 ckpt | **补记**：主实验的起点是 **`early-game model`**（VPT 基础模型再 BC 微调），故主实验的 `π_pt` **就是该 early-game ckpt**，而非基础模型本身；附录 G.2 的对照实验**两臂各自锚到各自起点**（house / early-game，KL 系数 0.4）。⇒ **「锚在冻结的早期 ckpt」在原文有直接先例**；「换成**自对弈**产物」仍**无证据**（§3.4） |
| E5 | 同上 §2.2 奖励表一节（`…:47-49`） | 只给了 4 档与「除数量」规则，**逐 item 表（U9）标为未验证** | **补齐并新增实现侧证据**：论文 Table 7 逐项已核（§1.1.2）；并从**官方 `.model` 参数文件**核到同一张表的实现形态 `{item: [数量上限, 每件奖励]}`、**奖励是 5 个 monitor 的加权和**（只有 `order_invariant_curriculum` 权重 1）、house 变体为 `1/quantity`、**torch 的 1/8 不是笔误**；同时**再次否证「×60」**（最小正奖励 1/20，无任何 scale 常数）（§1.1.0/§1.1.1） |

> 另：既有报告 §5.1 把「KL-to-prior 惩罚项 + β 衰减」列为可迁时写「**`π_pt` 完全可以是我们自己的早期 ckpt，不需要人类数据**」——**本报告 §3.4 把该句拆成两半并明确标注**：「锚在冻结的早期 ckpt」有原文先例；「锚在自对弈产物」**无原文证据**。
