# VPT（Video PreTraining）与 MineRL / BASALT 奖惩机制与 IL 接口 · 尽职调查

- **取数时间**：2026-09-19 04:09 CST（UTC 2026-09-18T20:09Z）
- **方法**：直接抓取 arXiv abs/HTML 全文（`arxiv.org/html/...`）与 jsDelivr CDN 上的官方仓库文本；`web_search` 全程 **402 Insufficient Balance**（不可用），故按纪律改用 web_fetch/curl 直取原文。
- **纪律声明**：本节所有英文原句均从一手 HTML 全文抓取后 grep 定位；凡本轮**未**在原文核到的条目一律标 `未验证`，不凭记忆补数字。**禁止跨项目数字对拍**（VPT = Minecraft 20Hz 键鼠；我们 = 卡牌决策帧，两者量纲不可互换）。

---

## §1 标识与链接

| 对象 | 标识 | URL | 核到 |
|---|---|---|---|
| VPT 论文 | arXiv:2206.11795（v1，2022-06-23 提交，cs.LG） | https://arxiv.org/abs/2206.11795 ；全文 https://arxiv.org/html/2206.11795v1 | ✅ abs + HTML 全文 |
| MineRL 数据集论文 | arXiv:1907.13440（IJCAI 2019，7 pages，6 figures） | https://arxiv.org/abs/1907.13440 ；全文 https://arxiv.org/html/1907.13440v1 | ✅ abs + HTML 全文 |
| MineRL 2019 竞赛（结果报告） | arXiv:2003.05012（*Retrospective Analysis of the 2019 MineRL Competition on Sample Efficient RL*） | https://arxiv.org/abs/2003.05012 | ✅ HTML v4 全文 |
| MineRL 2020 竞赛（方案/提案，**非结果报告**） | arXiv:2101.11071（*The MineRL 2020 Competition on Sample Efficient RL using Human Priors*） | https://arxiv.org/abs/2101.11071 | ✅ HTML v1 全文 |
| **BASALT 论文** | arXiv:2107.01969（*The MineRL BASALT Competition on Learning from Human Feedback*） | https://arxiv.org/abs/2107.01969 ；全文 https://arxiv.org/html/2107.01969v1 | ✅ abs + HTML 全文 |
| BASALT 2022 BC 基线官方仓库 | `minerllabs/basalt-2022-behavioural-cloning-baseline` | https://github.com/minerllabs/basalt-2022-behavioural-cloning-baseline | ✅ README（jsDelivr CDN 镜像） |
| OpenAI 官方 VPT 博客 | https://openai.com/research/vpt | — | ❌ **未验证**：本机取回 9,673 B 为 JS/挑战页（无正文），疑被 bot 防护拦截 |
| OpenAI VPT 开源仓库 | `openai/Video-Pre-Training` | https://github.com/openai/Video-Pre-Training | ❌ **未验证**（GitHub API 403 限流、raw 域被中断；BASALT README 中引用了它，但本轮未直取） |

> 注：任务书给的两个备用 ID（2004.03767、2109.13398）经核对**不是**这两篇（分别是光子纠缠与机器遗忘论文），已弃用。

---

## §2 VPT 两阶段与奖励/惩罚

### 2.1 阶段一：IDM → 伪标签 → BC（半监督 IL）

| 事实 | 数字 | 出处 |
|---|---|---|
| IDM 用承包商**标注**数据训练，再给海量**无标签**视频打伪标签 | 最佳 IDM 用 **1962 小时** 承包商数据训练，达 **90.6%** 按键准确率、鼠标 **R²=0.97** | VPT §4.1 / Fig. 3（左） |
| 无标签视频源 | 搜索得 **~270k 小时**，过滤为「clean」子集 **~70k 小时** = `web_clean` | VPT §4.2 |
| IDM 比 BC 数据高效 | **两个数量级**（"two orders of magnitude more data efficient"） | VPT §4.1；§3 |
| 至少需要多少标注数据 | **100 小时**即可训出「fairly accurate」IDM；≥10 h 才有任何 crafting，50 h 起 crafting table，**100 h 后收益平台** | VPT §3 / §4.6 |
| BC 目标 | 标准 NLL：`min_θ Σ_t −log π_θ(a_t\|o_1..o_t)`，其中 `a_t ~ p_IDM(a_t\|o_1..o_T)` | VPT §3 式(1) |
| 基础模型规模 / 成本 | **0.5B** 参数、**30 epochs**、`~9 天 / 720 张 V100` | VPT §4.2 |
| 承包商数据成本 | 全部结论基于 **~4,500 小时**、约 **$90,000**；项目总承包商支出约 **$160k**；IDM 那 ~2000 h ≈ **$40,000**；`contractor_house` ≈ **$8,000**；作者称 **$2,000** 的数据即可拿到大部分结论 | VPT 附录 B.2（"The contractors were paid $20 per hour"） |

### 2.2 阶段二：RL 微调 — 奖励是什么、有没有 KL 惩罚、β 取值

**奖励 = 稀疏成就（item 获得）+ 手工分层**，**不是**终端胜负奖励：

| 事实 | 原文依据（章节） |
|---|---|
| 奖励形态 | "Agents are rewarded for each item obtained in the sequence, with lower rewards for items that have to be collected in bulk and higher rewards for items near the end of the sequence." | §4.4 |
| 4 个 tier 的基础奖励 | tier 1（木/石）= **1**；tier 2（需煤）= **2**；tier 3（需铁）= **4**；tier 4（钻石）= **8** | 附录 G.1（Table 7 说明段） |
| 除数量（防偏科） | 每 item 奖励 = 基础奖励 ÷ 该 item 的**总奖励数量**。例：iron ore 基础 4、上限 3 ⇒ **4/3 per block**。钻石总量记 3、钻石镐记 1（但实际不限量） | 附录 G.1 |
| 奖励表构造方式 | 从钻石镐**沿科技树反推**加需求（先钻石镐 → 3 钻石 + 2 木棍 → 1 铁镐 → …），再加煤/火把，最后额外奖励原木（需求 5、奖励到 **8**） | 附录 G.1 |
| 稀疏性规模 | 即使拿到全部前置，也可能要**上千动作**才见下一个奖励（"human players often take more than 10,000 actions to find a diamond after crafting an iron pickaxe"） | 附录 G.1 |

**是否有 KL 到 BC 先验的惩罚项 → 有，且是论文的正式消融对象。** 英文原句（≤2 行）：

> "To combat the catastrophic forgetting of latent skills such that they can continually improve exploration throughout RL fine-tuning, we add an auxiliary Kullback-Leibler (KL) divergence loss between the RL model and the frozen pretrained policy."
> —— **VPT §4.4 正文**（HTML 行 221）
>
> "To prevent catastrophically forgetting the skills of the pretrained network when RL fine-tuning, we apply an auxiliary KL divergence loss between the RL model and the frozen pretrained policy."
> —— **VPT 附录 G.1**（HTML 行 1076）

形式与系数（**附录 G.1，式(2)**）：`L_klpt = ρ · KL(π_pt, π_θ)`，`π_pt` = 冻结的预训练策略，`π_θ` = 被训练策略，`ρ` = 权重。

| β（论文写作 **ρ**）项 | 取值 | 出处 |
|---|---|---|
| **ρ 初值** | **0.2** | Table 6 |
| **ρ 衰减** | **0.9995** / 每次 iteration（**decay**，不是固定 β） | Table 6 |
| 消融臂 β | **0**（no KL loss），且该臂 lr 从 `2×10⁻⁵` 降到 **`3×10⁻⁶`**（5 个 lr 扫描中的最佳） | Table 6 注释 |
| 附录 G.2 对照实验 | 两臂均把 KL 系数设为 **0.4**、lr **`6×10⁻⁵`**、每 item 奖励统一 `1/quantity` | 附录 G.2 |
| 从随机初始化跑 RL | **无** KL、无 KL decay，改为 entropy bonus **0.01** | Table 6 注释 |
| KL 是否替代熵正则 | 是："this KL divergence loss **replaces** the common entropy maximization loss" | 附录 G.1 |
| β 的两难（原话） | "a high coefficient ρ … would prevent the agent from properly optimizing the reward function while a low coefficient ρ was ineffective at protecting the learned skills" ⇒ 故「先高后衰减」 | 附录 G.1 |

**奖励缩放倍数**：
- **未验证**：**没有**在 VPT 原文核到 `scale`/`60` 形式的奖励缩放倍数。文中与数值缩放有关的三处是：(a) 值函数目标用 EWMA 均值/标准差归一化（附录 G.1，"we normalize the value-function target by subtracting the mean and dividing by the standard deviation"）；(b) 论文报告的**总回报量级**为 RL-from-scratch ≈ 0、VPT 微调 ≈ **13**、early-game 微调 ≈ **25**（§4.4 Fig. 7a）；(c) 步数常数 60（"60 consecutive attack actions"）与缩放假说无关。**任务书提到的「缩放 60」在本轮未获一手证据支撑，标 未验证**。

其它 RL 超参（Table 6，便于对照我们自己的 PPO）：lr `2×10⁻⁵`、weight decay **0.04**、batch size **40**、batches/iter **48**、context **128**、γ **0.999**、GAE λ **0.95**、PPO clip **0.2**、max grad norm **5**、PPG sleep cycles 2、PPG sleep KL 系数 1.0、PPG sleep max sample reuse 6。算法 = **PPG**（phasic policy gradient），规模 = **~248M** 参数 VPT、**~1.3M 集**（≈`1.4×10¹⁰` 帧）、每集 **10 分钟**（BC 评估为 60 分钟）。

**阶段二结果（供量级参照，禁止与本地数字对拍）**：无 KL ⇒ **只**学会 4 个早期 item（logs/planks/sticks/crafting table）且**不再有**新 item；有 KL ⇒ iron pickaxe **>80%**、钻石 **~20%**、钻石镐 **2.5%**（人类 57%/15%/12%）。出处：§4.4 Fig. 7 + 附录 G.2。

---

## §3 MineRL / BASALT

### 3.1 人类示范规模

| 事实 | 数字 | 出处 |
|---|---|---|
| MineRL 数据集总量 | **> 60 million** 自动标注 state-action 对 | arXiv:1907.13440 摘要 |
| 录制时长 | **500+ 小时** 人类示范，跨 **六个** 任务 | 同上 §（"The MineRL-v0 dataset consists of 500+ hours…"） |
| 体积 | 每版本 **130 GB** | 同上 |
| 采样频率 | 每个 Minecraft game tick（**20 ticks/秒**） | 同上下文；2020 报告 §1.3.2 |
| BASALT 每任务示范量 | BuildVillageHouse **1399** 视频/146G；CreateVillageAnimalPen **2833**/165G；FindCave **5466**/165G；MakeWaterfall **4230**/175G | BASALT 2022 BC 基线 README（注意：一个 demonstration 可能被切成多个 video） |

### 3.2 `reward_shaping`（成就 delta）具体怎么算、为什么需要

- **论文一侧（已核到）**：MineRL 环境奖励是**稀疏 + 少量 dense 变体**，**不是**成就 delta 塑形：
  - Navigate：两变体 —— sparse（**+1** 到达目标即终止）/ dense（**reward proportional to distance moved towards the goal**）— arXiv:1907.13440。
  - Treechop：**每获得 1 单位木头 +1**，拿满 **64** 单位终止 — 同上。
  - Obtain*（IronPickaxe/Diamond/CookedMeat/Bed）：**获得目标物品 +1** 并终止 — 同上。
  - MineRL 2020 报告：主任务 ObtainDiamond 是 sparse，"**+100 upon reaching the goal**, at which point the episode terminates"；并明确指出稀疏性/长时域"necessitates the use of efficient exploration techniques, human priors for policy bootstrapping, or **reward shaping via inverse reinforcement learning**" — arXiv:2101.11071。
- **`reward_shaping` 作为代码符号（成就 delta）**：❌ **未验证**。本轮在 MineRL / BASALT 论文全文 grep `reward_shaping` = **0 命中**（2020 报告仅出现 "reward shaping" 一词的泛述）；尝试直取两个官方仓库的具体文件路径（`minerl/env/reward.py`、`basalt_utils/reward_shaping.py` 等）**全部 404/被中断**（GitHub raw 域被 reset、GitHub API 限流）。⇒ **成就 delta 塑形公式（含 delta 定义与系数）本轮未取到一手证据，按纪律记 `未验证`，不凭记忆补写。**
- **为什么需要（已核到的一手理由）**：MineRL 2020 报告 §（任务挑战）写明稀疏奖励 + 长时域是两大核心难点，仅靠稀疏目标奖励需要高效探索或人类先验（引文同上）。

### 3.3「BC 单独就够」的实证结论

- **哪一届**：**NeurIPS 2019 MineRL（Diamond）竞赛**，第二届（**mc_rl** 队）。
- **结论（组织方报告原话）**："The second-place team, mc_rl, trained their hierarchical policies **entirely from human demonstrations with no environment interactions**." — arXiv:2003.05012 §4.2。
- **数字（Round 2 = 留出测试环境，100 局）**：Top-9 榜 **CDS 61.61 → mc_rl 42.41 → i4DS 40.80 → CraftRL 23.81 → UEFDRL 17.90 → TD240 15.19 → LAIR 14.73 → Elytra 8.25 → karolisram 7.87**（Table 1）。即**纯 demo、零环境交互的层次化 BC 拿到总榜第 2**。
- **附加事实**：该届设两个"对立范式"研究奖 —— 纯 IL 奖给 **mc_rl**，纯 RL（不用人类先验）奖给 karolisram（第 9 名，7.87，PPO baseline 改造）。— 同上 §4.3。
- **限定**：mc_rl 走的是**层次化策略**（非单帧 BC），且任务含动作缩减（全部 9 队都做了 action reduction）。Top-4 全部使用某种层次化 RL，Top-9 中 **7/9（78%）** 使用人类数据、**9/9（100%）** 做动作缩减。— 同上 §4.2 / Table 2。
- **BASALT 侧（人工判定、无奖励函数）**：评测 = 人工成对比较 + **TrueSkill**，跨 4 任务归一化后取平均分；4 任务为 FindCave / MakeWaterfall / CreateVillageAnimalPen / BuildVillageHouse，**明确不提供奖励函数**（"our tasks are explicitly designed to not include them"），允许任何人类反馈形式（+ 可选 <30 分钟附加指令）。— arXiv:2107.01969 §1.2/§1.4/§1.5。
- **接口层面的迁移证据**：BASALT 2022 官方 BC 基线的 README 首句 = "This solution **fine-tunes the "width-x1" models of OpenAI VPT** for more sample-efficient training." ⇒ VPT 权重被当作 IL 接口复用；数据集索引托管在 OpenAI VPT 仓库（`openai/Video-Pre-Training#basalt-2022-dataset`）。— 该仓库 README（本轮未逐行核基线代码）。

---

## §4 已知的坑（逐条带出处；能核到的核，核不到的标未验证）

| # | 坑 | 一手证据 | 状态 |
|---|---|---|---|
| 1 | **RL 微调把 BC 学到的行为洗掉（灾难性遗忘）** | 无 KL 臂："progress **stalls after 100,000 episodes**, suggesting that the skills necessary to make further progress have been **catastrophically forgotten**"（§4.4）；附录 G.2："The model learns to obtain all items that the early-game model can craft zero-shot … In contrast to the treatment with a KL-penalty, it **does not learn any items beyond these initial four**"（Fig. 16 说明） | ✅ 已核 |
| 2 | **β 的两难（高 β 不学奖励、低 β 护不住技能）** | 附录 G.1 原句（见 §2.2） | ✅ 已核 |
| 3 | **稀疏奖励不 work（从零 RL）** | "Training from a randomly initialized policy **fails to achieve almost any reward**"；"The model **never learns to reliably collect logs**"（§4.4 Fig. 7a/7b）。同向：MineRL 2020 明确 sparse + long horizon 需人类先验/塑形（arXiv:2101.11071） | ✅ 已核 |
| 4 | **奖励稀疏且具有欺骗性（deceptive）** | "the reward function is still **sparse** and, in some cases, even **deceptive** … the most efficient method for getting one item can make it far more difficult to get the next item"；具体例：拿 cobblestone 最快 = 造完木镐立刻下挖、**把工作台留在原地**，但这让后续更难（附录 G.1） | ✅ 已核（论文用词是 "deceptive"，非 "reward hacking"） |
| 5 | **IL 数据稀缺 / 标注昂贵** | 全结论基于 ~4,500 h、≈$90,000；总承包商支出 ≈$160k；IDM 那 ~2000 h ≈ $40,000（附录 B.2）；"Collecting contractor data can be difficult and expensive"（§4.3）；IDM 的存在意义即 **两个数量级** 数据效率（§4.1） | ✅ 已核 |
| 6 | **reward hacking / 利用奖励漏洞的具体案例** | ❌ **未验证**：VPT/MineRL/BASALT 论文中**未**出现 "reward hacking" 或等价的具体案例（仅见 #4 的 "deceptive" 表述与「agent 很少真的收集额外原木/放火把/用煤当燃料」这一偏置观察）。**不凭记忆补写**。 | ❌ 未验证 |
| 7 | **loss 与下游表现不相关（调参困难）** | "loss was **not consistently correlated** with downstream evaluation metrics … which often made progress slow and hard-won"（VPT §5）；另 §4.2：承包商数据 loss 在 7 epoch 后**上升**但下游游戏统计不下滑 | ✅ 已核 |
| 8 | IL 的**动作缩减/时间抽象**几乎是必需品 | MineRL 2019：Top-9 **100%** 做 action reduction，Top-4 **全部**用层次化结构（arXiv:2003.05012 §4.2/Table 2） | ✅ 已核 |

---

## §5 可迁移 / 不可迁移（对照我们的场景）

**我们的场景（本地已核，非引用）**：Clash Royale **自对弈 PPO**；**无人类数据**、无无标签视频池；手工奖励表 `rl/config.py::DEFAULT_REWARD` = **本轮实测 22 个键**（`engagement_trade` 及其 4 个子键、`crown_weight`、`crown_lose_weight`、`tower_dmg_opp/self/late`、`tower_dmg_self_late`、`win_bonus`、`lose_penalty`、`draw_penalty`、`invalid_penalty`、`elixir_bonus`、`normalize_tower_dmg`、`elixir_diff_weight/late`、`unit_dmg_k`、`tower_premium_k`、`king_gate`）；`step` = 决策帧（【R9】）；IL 侧仅 `src/clasher_new/rl/train_bc.py`（**本轮实读**：第 107 行 `policy.evaluate(obs, tok, plan, bundle, masks, hidden=None)` ⇒ **逐帧无时序**）。参数量 **993K**：❌ 本轮**未复算**（`import torch` 在本环境失败：`libtorch_global_deps.so` 缺失）⇒ 沿用给定值但不背书。

### 5.1 可迁移（机制层，与「有没有人类数据」无关）

| 机制 | 迁移理由 | 对应本地落点 |
|---|---|---|
| **KL-to-prior 惩罚项 + β 衰减**，用它**替代**熵正则 | VPT 是一手消融证据：无惩罚 ⇒ 早期技能被洗掉、进度在 100k 集后停滞；有惩罚 ⇒ 直达终局 item。**关键点：`π_pt` 完全可以是我们自己的早期 ckpt**，不需要人类数据 | PPO 里加 `ρ·KL(π_pt‖π_θ)`，ρ 起 0.2、×0.9995/iter；与 `ent_coef` 二选一（VPT 明确是 *replaces*） |
| **奖励分层 + 除数量（防偏科）** | 「按科技树反推 item 清单、tier 基础分、再除以总奖励数量」是一套**可复用的奖励表设计法**，与数据来源无关 | 现表 22 键已同构（`crown_weight=8` vs `tower_dmg_*`≈0.001 就是分层+除数量）；可据此复核权重比是否被大 item 主导 |
| **β 的「先高后衰减」+ 值函数目标 EWMA 归一化** | VPT 报告了两难与解法；EWMA 归一化是「奖励量纲大 ⇒ 值函数梯度炸」的标准处理 | 与我们 `value=`/`vraw` 口径、`v_scale` 的既有教训同向（本地已确证：只看 `value=` 会读到假的下降） |
| **动作/时间抽象几乎必需** | MineRL 2019 的 100% 动作缩减、78% 用人类数据、Top-4 全层次化 —— 说明**原始动作空间直接 IL/RL 都难** | 支持我们既有的「option / 带时长动作」结论（本地 `hold d~U{1..40}` 对照） |
| **人工判定 + TrueSkill 式比较**（BASALT） | 当奖励函数本身不可信（我们对奖励表已多次实测其口径缺陷）时，**成对人工/代理评判**是独立的评价通道 | 作为评估协议的备选思路（不解决训练侧问题） |

### 5.2 **不可迁移**（因为我们没有人类数据 / 没有无标签视频池）

| VPT/MineRL 机制 | 为什么不可迁移 |
|---|---|
| **① IDM 给海量无标签视频打伪标签 → BC** | 该阶段的前提是「存在 ~270k h 原始 / ~70k h 干净无标签视频」，且 IDM 只需 100–1962 h **标注**数据即可用（VPT §4.1/§4.2）。我们**两边都没有**：无人类演示、无视频池。⇒ **整个 §2.1 阶段一在我们的场景中不存在**，`train_bc.py` 不是 VPT 意义上的 foundation-model BC。 |
| **② 「承包商数据 × 两个数量级」的数据效率论证** | 该论证是「把 $2k 标注换成 70k h 无标签」的杠杆（VPT §4.1/附录 B.2）。**在无无标签池时杠杆不存在**。 |
| **③「预训练行为先验当探索先验」的整体叙事** | VPT §5 的结论句是 "using these learned behavioral priors as **extremely effective exploration priors** for RL" —— 先验来自人类视频分布；自对弈场景里**唯一的先验只能来自我们自己的历史 ckpt**（≈ 机制层的 KL-to-prior 仍可用，但**不是**人类先验）。 |
| **④ VPT 的 item-tier 奖励数值本身** | 1/2/4/8 的档位与「铁矿石 4/3」是 Minecraft 科技树量纲的产物，**与卡牌/圣水/塔血量纲不同源**。⇒ 只迁移「设计法」，**禁止**迁移数值（跨项目对拍红线）。 |
| **⑤ 20Hz 键鼠接口 / 原生动作空间探索难度论证** | 「随机策略拿到一根原木概率 = (1/2)^60」（VPT §1）是连续 60 步按键的合取概率，**不能**搬到决策帧上（我们的帧语义与回费结构不同，本地已有自己的合取分析）。 |
| **⑥ BASALT「不做奖励函数、改用人工成对比较」** | 前提是**有真人标注预算**（并要求 <30 min 指令）。我们无人类数据 ⇒ 只能做**代理评判**（且代理=奖励表本身时构成循环，本地已多次实测该循环的假信号）。 |
| **⑦ PPG + 248M 参数 + 720×V100/9 天 的算力配置** | 单机 + 993K 参数（给定值，未复算）差 2–3 个数量级 ⇒ VPT 的 batch 48×40、context 128、1.4×10¹⁰ 帧等超参**无从照搬**（【R15】禁止跨实验照抄阈值）。 |

### 5.3 我们这里的**最大缺口**（与 VPT 的差距，非「VPT 有我们没有」而是「接口错配」）

1. **KL 惩罚项在我们这儿缺 `π_pt` 的产生方式**：VPT 的 `π_pt` = 人类视频 BC 出来的基础模型；我们只能用**早期自对弈 ckpt** 或**规则/手写专家**当锚。⇒ 可用，但需**单独预注册**（VPT 的 §4.4 消融只证明「锚住人类先验有用」，**未**证明「锚住早期自对弈 ckpt 有用」——这一条属**未验证的类推**）。
2. **`train_bc.py` 逐帧 `hidden=None`** ⇒ 无时序 ⇒ 即便有锚，锚定的也只是**单帧动作分布**，无法表达 VPT 里那种「跨上千动作的技能」被遗忘的语义。VPT 的遗忘对象是**技能链**（smelt iron 的前置全序列），不是单帧动作。⇒ 若要照搬该论证，先要有时序接口。

---

## §6 来源清单（URL + 取数时间）

全部取数时间 **2026-09-19 04:04–04:09 CST**（UTC 2026-09-18T20:04–20:09Z）。

| # | 来源 | URL | 用途 |
|---|---|---|---|
| S1 | VPT abs | https://arxiv.org/abs/2206.11795 | 标识、摘要 |
| S2 | VPT 全文 HTML v1 | https://arxiv.org/html/2206.11795v1 | §1–§5、附录 A/B/D/F/G/H/I（本文所有 VPT 引文） |
| S3 | MineRL 数据集 abs | https://arxiv.org/abs/1907.13440 | 标识 |
| S4 | MineRL 数据集全文 HTML v1 | https://arxiv.org/html/1907.13440v1 | 60M/500+h/130GB、各任务奖励定义 |
| S5 | MineRL 2019 竞赛报告全文 v4 | https://arxiv.org/html/2003.05012v4 | Top-9 分数、mc_rl 纯 IL 结论、Table 1/2 |
| S6 | MineRL 2020 竞赛报告全文 v1 | https://arxiv.org/html/2101.11071v1 | +100 sparse、需塑形/人类先验的论断 |
| S7 | BASALT abs | https://arxiv.org/abs/2107.01969 | 标识 |
| S8 | BASALT 全文 HTML v1 | https://arxiv.org/html/2107.01969v1 | 4 任务、无奖励函数、TrueSkill 评测 |
| S9 | BASALT 2022 BC 基线 README | https://github.com/minerllabs/basalt-2022-behavioural-cloning-baseline （正文经 https://cdn.jsdelivr.net/gh/minerllabs/basalt-2022-behavioural-cloning-baseline@main/README.md 取回） | VPT 1x 微调、BASALT 示范量 |
| S10 | OpenAI VPT 博客 | https://openai.com/research/vpt | ❌ 未取到正文（bot 挑战页） |
| S11 | OpenAI VPT 仓库 | https://github.com/openai/Video-Pre-Training | ❌ 未取到（API 限流 / raw 域被中断） |

> 方法论附注：`web_search` 在本会话返回 **HTTP 402 Insufficient Balance**（非本次调研内容所致），已按任务纪律改用 `web_fetch` + `curl` 直取 arXiv HTML；`raw.githubusercontent.com`、`api.github.com`、`githack` 在本机网络下被中断/限流，`cdn.jsdelivr.net` 可用（但该 CDN 对部分路径返回 false-404，故 S9 只核到 README）。

---

## §7 未验证项（**不得**在后续文档中当作事实引用）

| # | 未验证项 | 本轮尝试 | 结论 |
|---|---|---|---|
| U1 | **奖励缩放倍数 `scale` / `60`** | VPT 全文 grep `scale`/`60`：命中 60 仅为「60 consecutive attack actions」；数值缩放只见 EWMA 值函数归一化（附录 G.1）与回报量级 13/25（§4.4） | ❌ **未验证** —— 任务书所述「缩放 60 倍」未获一手证据 |
| U2 | **`reward_shaping` 成就 delta 的具体公式与系数** | MineRL/BASALT 论文 grep `reward_shaping` = 0 命中；`minerl`/`basalt` 仓库具体文件路径全部 404/被中断 | ❌ **未验证**（论文一侧只核到 sparse/dense 奖励定义） |
| U3 | **reward hacking 的具体案例（VPT/MineRL 线）** | 三篇论文全文均无 "reward hacking" 字样；仅有 "deceptive" 表述 | ❌ **未验证**（不得把 §4#4 改写成「已证实 reward hacking」） |
| U4 | **OpenAI 官方 VPT 博客内容**（含其两阶段措辞） | 取回 9,673 B JS/挑战页 | ❌ **未验证** |
| U5 | **`openai/Video-Pre-Training` 仓库内容**（含 BASALT 数据集索引） | API 限流 + raw 域中断；仅从 S9 间接得知其存在 | ❌ **未验证** |
| U6 | **mc_rl 的方法细节**（层次结构、单帧 BC 还是选项） | 2019 报告仅一句概括；无链接到的论文/代码（报告只给 YouTube 视频链接） | ⚠️ 部分：**「纯 demo、零环境交互、总榜第 2（42.41）」已核**；「具体算法细节」**未验证** |
| U7 | **MineRL 2020 竞赛的最终结果**（本篇 arXiv:2101.11071 是赛前方案，非结果报告） | 本会话未找到 2020/2021 结果报告（候选 ID 试取均非目标） | ❌ **未验证** —— 故 §3.3 的「BC 单独够」结论**只**引 2019 届 |
| U8 | **本地 993K 参数量** | `import torch` 在本环境失败（`libtorch_global_deps.so` 缺失） | ⚠️ **未复算**，沿用给定值 |
| U9 | **VPT 奖励表 Table 7 的逐 item 数值**（20 planks / 1 crafting table / 3 diamonds 等） | 正文段落已核到 tier 与「除数量」规则、以及 4/3 的例子；**逐行表格未逐项抓取** | ⚠️ 部分：规则已核，**逐 item 清单未验证** |
| U10 | **BASALT 2022 竞赛的最终结果 / BC 基线得分** | 只核到 README 与数据集统计 | ❌ **未验证** |

---

## 一句话结论

VPT 与 MineRL/BASALT 线的核心可迁移物**不是**其 IL 管线（IDM 打伪标签的前提是海量无标签视频 + 少量承包商标注，我们两样都没有），而是它**一手消融过的「KL 到冻结先验 + β 衰减、替代熵正则」机制**（`L_klpt = ρ·KL(π_pt,π_θ)`，ρ 起 **0.2**、每 iter ×**0.9995**，VPT §4.4/附录 G.1/Table 6）——其证据是**无该惩罚项时早期技能被灾难性遗忘、进度在 10 万集后停滞**；同时 MineRL 2019 提供「纯人类示范、零环境交互的层次化 BC 拿总榜第 2（42.41）」这一 IL 有效性实证。但**该惩罚项在我们场景需把 `π_pt` 换成早期自对弈 ckpt，这一替换在原文中无证据支持**（属未验证类推）；且任务书提到的「奖励缩放 60 倍」与「`reward_shaping` 成就 delta 公式」本轮**均未取得一手证据**，标 `未验证`。
