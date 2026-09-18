# 「其他优秀项目（用了 IL）的奖惩机制」对我们的参考

## —— 兼《三轴对照 我们 vs FirstLight》的独立复核

> **取数时间**：2026-09-19（本地 CST；外部一手文献多为 2026-09-18T19:33Z–20:30Z 取回）。
> **分析对象**：
> ① 用户指定的对照文档 `三轴对照_奖惩_训练_非法动作_我们vsFirstLight_2026-09-19.md`（**仓库根目录**，295 行，非 `docs/`）；
> ② **其他**使用 IL（模仿学习）的优秀项目的奖惩机制。
> **我们项目** = `/mnt/e/clash-royale-simulator-main`（下略 `src/clasher_new/`）；
> **FL** = `/mnt/e/FirstLight_CR`（HEAD `28d66cc0a5d65888515e22fdf22f11d783b65efb`，单 commit 仓库）。
> **手段**：本地只读 grep/read + 外部一手来源（arXiv e-print/HTML、HF mirror API、GitHub API）直取。
> **本轮未跑训练、未跑全量 selftest**（【R19】）；**未改任何训练/奖励代码**。
> **纪律**：无 `file:line` 或 URL 支撑的陈述一律标「未验证」；**禁止跨项目数字对拍**（量纲 / `step` 语义 / 参数规模不可比）；
> 描述性观察不得升格为判据（【R3】）；口径冲突**并列不调和**（【R17】）。

---

## 0. 一页结论

| # | 结论 | 证据强度 |
|---|---|---|
| **1** | **「IL 项目的奖惩机制」这个提法本身要先拆**：IL 阶段**结构性没有奖励**——FL 的 IL 损失 6 项全是**掩码 CE + Huber**（`native_runner/training/v4/imitation.py:283-290`），`penalty` 在 7 个 IL 文件里 **0 命中**；唯一沾奖励的是第 6 项 `value`（target = **引擎奖励的折扣回报**）。**「IL 项目有真实惩罚项」这个直觉在 FL 上不成立。** | 本地只读，编排者独立复核 |
| **2** | 真正有「**IL→RL 锚定惩罚**」的先例不是 FL（其 `ppo_expert_bc` **默认关闭**），而是 **OpenAI VPT**：`L_klpt = ρ·KL(π_pt, π_θ)`（Eq. 2），`ρ=0.2`、`×0.9995`/iter，**替代熵正则**；消融「无 KL ⇒ 10 万局后停滞、只剩前 4 个 item」。 | 一手 arXiv HTML，编排者独立复核 |
| **3** | **对我们的参考不是「照搬某个奖惩项」**，而是三条**不碰奖励函数**的机制：**I2 时序化 BC（最高性价比）**、**I3 掩码契约（先决条件）**、**I1 BC/NLL（仪器）**。**没有任何 IL 手段能绕过 C14**——IL 的数据侧与奖励侧**共用同一个采样器**。 | 本地只读 + 一手文献交叉 |
| **4** | 三轴 doc 的结论**基本成立**，但复核出 **1 处硬错误、1 处引用错位、1 处强度不一致**；其 **H5（掩码对账无门禁）我已执行并留证**（见 §8）。 | 见 §5 |
| **5** | **本轮最有价值的新发现**：**公开人类 CR 数据是存在的** —— `VanguardX101/IL_Replay`（**252,238 局 / 17,836,160 动作**，公开、非 gated、**许可未声明**），但**只有动作侧**（`card / native_x / native_y / replay_tick_20hz`），**没有逐帧圣水 / 单位 HP / 位置轨迹** ⇒ 状态必须**靠引擎重放反推**。这把「我们无人类数据」从「**无解**」改成「**有输入、缺管道**」（许可 + 可重放性两项未验证）。 | HF API 编排者独立复核 + 子智能体实读 parquet |

---

## 1. 范围、手段与通道限制

| 项 | 内容 |
|---|---|
| 文档对象 | 根目录三轴对照 doc（295 行 / 28,290 B），逐节通读 |
| 本地只读对象 | 我们项目 + FL（`native_runner/training/v4/`）+ 我们自己的 IL/BC 现状 |
| 外部对象（按对象逐个取证） | AlphaStar、绝悟/JueWu、RLHF（InstructGPT/Anthropic HH/TRL/DS-Chat）、离线 IL+RL 微调族（AWAC/IQL/DT/TD3+BC/DAPG/RLPD/Cal-QL/robomimic）、**VPT/BEHAVIOR**、**MineRL/BASALT**、**OpenAI Five**（非 IL，对照）、**Clash Royale 领域公开人类数据与 IL 项目** |
| 通道限制（影响结论强度，如实登记） | ★ 本环境 **`web_search` 全程 HTTP 402（余额不足）** ⇒ 外部取证**全部改为 `web_fetch`/`curl` 直取** arXiv abs/HTML/e-print、ar5iv、`hf-mirror.com`、`api.github.com`。**不可达**：`huggingface.co`、`raw.githubusercontent.com`、`web.archive.org`、`nature.com`（SSO 302）、`developer.clashroyale.com` 正文（登录墙）。⇒ **AlphaStar Nature 正文本轮仍未取回**（无预印本，Semantic Scholar `externalIds` 无 arXiv id） |
| 未做 | 未跑训练/selftest；未加载 FL 的 `.pt`（本机 `libtorch_global_deps.so` 缺失）；未执行 FL 任何测试（依赖 APK/arm64 设备通道） |

---

## 2. 第一问的明确答复：IL 家族能提供什么「不碰奖励」的手段

**背景（我们已确证的病理，不是推测）**：
- **C14**：「放弃一张买得起的牌」在 **75,891 帧**里发生 **0 次**（7,238 段主动不出牌**0 段**在圣水 ≥4.0 时仍不出；A 段最长 **22 帧 < 34 帧**）⇒ **0 样本 ⇒ 0 梯度**，且 `step 0` 与 `100k` **同结构**（`docs/elixir_saving_audit_2026-09-18.md`；台账 C14）。
- **O7**：机制自锁——Xbow **不在** `ACE_CARDS` / `SINK_TANK_CARDS`，`_pick_suggested_card` **只推付得起的牌**（`docs/s1_gate_2026-09-18.md`；台账 O7）。
- **X-19**：奖励侧方案**已完整取证、判不接线**（离线 +0.145 → 在线 +0.008 → 补尾部 flush 归零；`docs/online_measure_2026-09-18.md`；台账 X-19）。
- **O10**：探索侧字面随机被**定量否证**；有效轴是**时长**（option）；且**采样到了 ≠ 会学**（`docs/exploration_pressure_gate_2026-09-18.md`）。

**答复（逐条判定）**：

| # | 机制 | 在我们仓内的落点 | 逐位可验？ | 成本 | 判定 |
|---|---|---|---|---|---|
| **I1** | BC / NLL 示范损失（**IL 阶段无奖励的证明**，不是修复） | `rl/train_bc.py:107-111`（`lp = policy.evaluate(obs,tok,plan,bundle,masks,hidden=None)`，`loss=-lp`）；`rl/human_play.py:205-233` | ✅ 单帧 NLL 可由脚本用同一 ckpt 复算 | 极低 | **可做（作仪器）**；⚠️ 它会把「便宜牌 95.47%」学得更牢，**不针对病根** |
| **I2** | **时序 stateful：RNN + teacher forcing + TBPTT** | `rl/train_bc.py:107`（**每帧 `hidden=None`**）+ `rl/follower.py:753-806`（`evaluate`） | ✅ 同帧 A/B（`hidden` 只影响帧间耦合）+ 梯度链长度可断言 | 低 | **可做（最高性价比，须预注册）** |
| **I3** | **掩码契约：采样即校验** | `rl/follower.py:433`（`act`）/ `:529`（`act_parallel`）+ `rl/action_mask.py:521-573`（`validate_bundle`） | ✅ 【R13】128 张位图 + `validate_bundle` 断言（**门禁本已由本轮补齐，见 §8**） | 低 | **可做（先决条件，与奖励无关）** |
| I4 | 数据侧质量过滤 | `rl/export_replay.py` 等 | ❌ 减法无逐位意义 | 低 | **不做**（过滤只丢样本，不能造样本） |
| I5 | 样本/优势加权（AWAC·AWR·IQL 形态） | `rl/train_bc.py:107-111` 改 `-(w*lp)`；`w` 有两套约定（`exp(A/λ)` 论文形式 vs **批内 softmax = 两个参考实现的默认**） | ✅ 权重张量可逐元素对账；⚠️ 【R17】须报两套约定 | 中 | **待预注册**；⚠️ **零样本区同样退化**（样本数 0 ⇒ 无 Q/V 可依） |
| I6 | return-conditioning（DT / RvS） | 需注入 `R̂` 进 `plan_vec`（【R6】`PLAN_DIM`）或新增输入头（**必须 `--fresh`**） | ✅ 输入可逐元素对账；❗ 性能判据不可判（单点差 0.60） | 中高 | **待预注册**；⚠️ 一手反证：DT/RvS 在**随机环境**下「fail dramatically」且「**not due to a lack of data**」（arXiv:2205.15967） |
| I7 | DAgger / expert iteration | 在线专家 = `rl/belief_planner.py`（`plan()` 只返回 PlanToken，动作仍由 RL 出） | ✅ 纠正帧可复算 | 高 | **待预注册（优先级低于 I2/I3）**；⚠️ **致命前提**：专家自身在 Xbow 卡组 **0 帧**产攒费动作（O7）⇒ 会放大同一局部最优 |
| **I8** | **IL→RL 锚定：KL-to-reference / BC 正则** | `rl/ppo.py:379-392`；参考 = 冻结 BC ckpt | ✅ `p=0` 时逐位回旧行为 | 中 | **不做（方向与病根相反）**：KL 把策略**拉回参考分布**，而瓶颈是**探索不足**；且 RLHF 的 KL 两条主干理由都以「奖励是学出来的代理」为前提，**我们没有 RM**（`docs/rlhf_kl_penalty_survey_2026-09-19.md` §5）；也触发【R11】 |
| I9 | 奖励塑形 + 退火（OpenAI Five 类） | — | — | — | **不做（非 IL 证据）**；且「OpenAI Five 靠奖励退火」是**误读**（见 §4） |

### 2.1 我这一轮补上的两条机制解释（子报告未覆盖）

1. **策略侧没有时域表达，BC 侧也没有** —— 这不是两件事，是**同一处结构性缺失**：
   - `rl/observation.py` 检索 `history|stack|prev_|frame_stack` = **0 命中**（编排者复验）⇒ 观测**没有历史栈**；
   - `rl/train_bc.py:107` 每帧 `hidden=None`（编排者复验）⇒ BC 帧间**无状态、无梯度**。
   ⇒ 与 O10 的结论（**只有 option（带时长的动作）有时域能力**）**同构**：要表达「连续 34 帧不花钱」，**策略侧缺 option、BC 侧缺时序**。⇒ I2 是**唯一**能让未来采样到的长时域片段**可学**的通路；但它**不增加覆盖率**（那是 O7/O10 的活）。
2. **我们仓内的 IL 链路「写过、但从未跑过」**：磁盘上 `follower_bc.pt` / `follower_human.pt` / `bc_*.pkl` / `human_data/` **全部不存在**；`rl/train_bc.py` 全文**无 `pickle.dump` / `json.dump`**（不落盘）⇒ I1/I2 是**现成代码 + 缺一次运行**，不是"要新建一套"。
   ⚠️ 唯一有 `(obs, action, masks)` 落盘通道的是 `rl/human_play.py`（需人类或外部驱动）；**S1 手写专家本身没有**该通道（它只写 schema-4 联赛录像，**不含 32×18 观测网格**）。

---

## 3. 「IL 项目的奖惩机制」先要澄清：五类，互不混用

| 类 | 「奖惩」实际是什么 | 两个实例（URL + 取数时间 2026-09-19） |
|---|---|---|
| **① IL 本身没有奖励** | 只有示范损失（BC / NLL / CE）。「奖惩」在 IL 阶段**结构性缺席** | FL：`imitation.py:283-290` 六项全掩码 CE/Huber，`penalty` **0 命中**；绝悟 IL：双层动作标签 + 多视图意图标签**四项 CE 加权和**（λ 全 1，无 KL 无熵） |
| **② 隐式奖惩落在数据侧** | 示范质量过滤 / 样本加权 / DAgger 纠正 / expert iteration。**是「丢样本」或「重采样」，不是「扣分」** | FL：`dataset.py:22-41` 预检拒绝、`cache.py:87-124` 旧缓存非法标签清洗、`producer.py:170-209` 不可表示窗口降级 WAIT；VPT：IDM 给海量无标签视频打伪标签 |
| **③ 把奖励塞进 IL 损失** | 优势加权（AWAC/AWR/IQL）/ return-conditioning（DT/RvS）/ BC 辅助损失。**奖励以「样本权重」或「条件 token」形式出现** | AWAC `θ←argmax E[logπ_θ·exp(A/λ)]`（arXiv:2006.09359，λ=0.3 manipulation / 1.0 MuJoCo；⚠️ 两个参考实现默认改用**批内 softmax**）；DT `R̂_t=Σ_{t'≥t} r_{t'}` 作**输入 token**（arXiv:2106.01345） |
| **④ IL→RL 微调的锚定惩罚**（**IL 项目里最真实的「惩罚项」**） | KL-to-reference / BC regularizer / trust region | **VPT**：`L_klpt=ρ·KL(π_pt,π_θ)`，ρ=0.2、decay 0.9995，**替代熵正则**（arXiv:2206.11795 Table 6）；RLHF：InstructGPT Eq.(2) `r_θ − β·log(π_RL/π_SFT)`，β=0.02 |
| **⑤ 纯 RL + 奖励塑形并退火**（**非 IL**，对照，**不可当 IL 证据**） | 手写奖励项及其调度 | **OpenAI Five**：Appendix G Table 5 共 21 项权重，**奖励函数「constructed once at the start of the project」**；`team spirit` 是 **0.3→1.0（升，不是退火到 0）**；全文 `anneal` 7 处命中**没有一处**指奖励权重 |

> **这一节的意义**：用户问「IL 项目的奖惩机制」，若不做这层拆分，就会把 ⑤（OpenAI Five 的奖励塑形）或 ④（RLHF 的 KL）当成「IL 的奖惩」，从而推出**方向相反**的结论（我们的瓶颈是探索不足，KL 恰恰是把分布**收窄**）。

---

## 4. 逐项目事实表

| 项目 | 范式 | IL 数据 | 奖励项 | 惩罚/锚定项 | 塑形退火 | 已知坑（作者原文） | 取证等级 |
|---|---|---|---|---|---|---|---|
| **FL** `28d66cc` | **IL → 多机同步 PPO** 两段 | **真人天梯回放**（RoyaleAPI 采集）；声明 **252,238 局 / 17,836,160 动作**；`cache_builder --source-count` 默认 **100** | 引擎层 2 项 + 训练侧 6 项 | **PPO 损失内 0 个 KL/BC 正则**（穷举 0 命中）；`ppo_expert_bc` 是**每次 PPO update 之后的独立 BC 步**，**默认关闭**（`--expert-bc-manifest` 默认 `None`，launch 脚本不传） | 无 | IL 用**人类动作**但 `_sanitize_legacy_illegal_expert_targets` 存在 ⇒ 人类标签有非法项；`decoding.py:102-127` 是**独立复刻**的包内规则（残余同构风险） | 本地只读，编排者独立复核 |
| **AlphaStar**（Nature 575:350–354） | **IL → league 自对弈** | **971K replays / 44 天**（第三方复现件转述，**非原文**） | **二值 win/loss + 人类统计量 z 导出的 pseudo-rewards** | RL 更新里**持续最小化「当前策略 vs 监督人类策略」的 KL** | ★ **两路独立检索均未找到权重/退火** ⇒ `未验证` | — | ⚠️ **Nature 正文本轮仍未取回**（无预印本）；上表多项为**复现件/二手** |
| **绝悟 / JueWu** | **IL → RL + 三阶段课程** | **top 1% 人类玩家**，每英雄约 **12 万局 / 1 亿样本**，预处理后**只留 1/20 帧** | 按游戏功能分组的**密集+稀疏奖励表**（1v1 Table 6 / 5v5 Table 4 / 开悟 Table 5）；**5 个价值头 = 奖励分解**（`V̂=Σw_k V̂^k`，**`w_k` 数值论文没给**） | 唯一显式消极惩罚 = 5v5 `No-op −0.00001`；**找不到 KL 锚定、找不到规则门禁**；非法动作走**掩码剔除**（`legal_action` + `sub_action_mask` 是 **API 契约**） | **「奖励权重退火」在一手文献中找不到任何直接证据**（既不能断言用、也不能断言没用） | learning collapse；泛化崩溃到**胜率恒 0**；**通用奖励函数导致所有阵容打同一种打法**；「人类微操有噪声，完全模仿会掉性能」 | 一手（6 篇论文全文留证 `docs/juewu_research_2026-09-19/`） |
| **RLHF**（InstructGPT / Anthropic HH / TRL / DS-Chat） | **SFT(IL) → RM → PPO** | 人类示范（InstructGPT **13k** SFT 提示） | **学出来的 RM 分数**（不是手写奖励） | ★ **KL penalty to reference policy**：InstructGPT β=**0.02**；HH λ_KL=**0.001**；TRL `init_kl_coef`=0.2（自适应 `target`=6.0）；DS-Chat `kl_ctl`=0.1（**逐 token 扣，RM 分数只在最后一个有效 token 加一次**） | 无 | ★ **反直觉一手证据**：「KL penalty … the effect … is **akin to early stopping**」（arXiv:2210.10760 §3.6），该文其余实验**把 KL 设为 0**；LIMA 用 **1000 条精选**、不做 RLHF，65% 胜过做过 RLHF 的 DaVinci003 | 一手（论文/代码逐行，含 3 处纠正） |
| **离线 IL+微调族** | 离线 IL / offline RL | 人类或机器示范 | 奖励**先经 critic 变标量优势 A(s,a)**，再作模仿**样本权重**（AWAC/IQL）或**条件 token**（DT） | IQL 用**非对称 expectile + 优势加权回归**，刻意**不查询数据集外动作** | — | ★ **BC vs 离线 RL 证据分裂且取决于数据源**：人类示范上 BC-RNN ≫ 离线 RL（Square-MH **78.0 vs BCQ 14.0 / CQL 0.7**）；合成/低数据量上离线 RL 反超（D4RL locomotion BC **466.7** vs IQL **692.4**）；DT ≈ **掐尖 %BC**（平均 **56.1 vs 56.7**）；robomimic：**时序建模（BC-RNN/HBC/IRIS）在人类数据上强** | 一手（6 篇 e-print 逐字 grep） |
| **★ VPT / BEHAVIOR** | **IDM 伪标签 → BC → RL 微调** | 承包商数据 + **海量无标签视频**（IDM 打伪动作标签） | **稀疏 item 成就 + 手工 4 档**（1/2/4/8，再**除以该 item 的总奖励数**） | ★★ **`L_klpt = ρ·KL(π_pt, π_θ)`（Eq. 2）**，ρ=**0.2**、**×0.9995**/iter，**替代熵正则**（Table 6） | 无（ρ 单调衰减） | ★ **无 KL 的消融**：「progress stalls after **100,000** episodes … **catastrophically forgotten**」，只学会 4 个早期 item；「high ρ 会阻碍优化奖励，low ρ 无法防止遗忘」⇒ **两端都坏，必须衰减** | 一手 arXiv HTML，**编排者独立复核** |
| **MineRL / BASALT** | 人类示范 + RL 竞赛 | 人类示范（**小时数/条数未取到**） | 稀疏/密集变体：Navigate +1；Treechop +1/木、64 终止；Obtain* +1 终止；2020 报告 ObtainDiamond **+100** | — | — | ★ **2019 届 mc_rl 队「entirely from human demonstrations with no environment interactions」取得 Round-2 榜第 2（42.41，第 1 为 CDS 61.61）** ⇒ **IL 单独就够**的实证先例；⚠️ **`reward_shaping` 成就 delta 公式三篇论文 grep 0 命中**、官方仓库路径 404 ⇒ `未验证`；2020/2021 结果报告未找到 | 一手（论文）；⚠️ 多项未验证 |
| **OpenAI Five**（**非 IL，对照**） | 纯自对弈 RL（**不用 IL/人类数据**） | 无 | ★ **Appendix G Table 5：21 项权重已逐行取回**（Win 5 / Hero Death −1 / XP 0.002 / Gold 0.006 / Killed Hero −0.6 / Last Hit −0.16 / T1 2.25 / Barracks 6 …）+ **3 个附加件**：零和化、局内时间权重 `0.6^(T/10min)`（**不作用于 win/loss**）、team spirit `r_i=(1−τ)ρ_i+τρ̄` | — | ★ **「奖励塑形退火」在 v1 文本中不存在**：奖励函数「constructed **once at the start of the project**」；`schedule` 命中的全是**超参/出装顺序**；team spirit 是 **0.3→1.0**；稀疏奖励消融是**一次性关闭**而非退火 | — | 一手（LaTeX e-print + HTML 双通道交叉，**编排者复核结论对**） |
| **★ CR 领域公开数据/项目** | — | ★ **`VanguardX101/IL_Replay`**：**252,238 局 / 17,836,160 动作**，**公开、非 gated**、**许可未声明**（`croissant.license=None`）、109 文件、downloads 570、lastModified **2026-09-06**。另有 `Cochon123/clash-royale-replays`（1,592 局）、4 个 Kaggle 数据集（**均无逐帧**） | — | — | — | ★ **公开的「逐帧完整对局状态 + 人类动作」配对数据集 = 不存在**（检索范围：HF datasets API 6 组 / Kaggle 3 组 / GitHub search 14 组 / arXiv 全库 4 条命中）。两个最大公开集的**动作侧**字段为 `card_key / native_x / native_y / replay_tick_20hz / time_seconds / side`；**塔血只有终局、圣水只有全场聚合、`"hp"` 出现 0 次** | HF API 编排者独立复核 + 子智能体实读 parquet |

### 4.1 两条必须一起读的推断（**标注为推断，非确证**）

1. **`VanguardX101/IL_Replay` 很可能是 FL 自己的数据集**：FL HEAD commit 时间 `2026-09-06T22:17:43-04:00` 与该数据集 `lastModified 2026-09-06T00:25:12Z` **同日**，且动作数与 FL 声明的 17,836,160 **逐字相同**。⇒ **FL 的人类数据是公开可下载的**（推断；未经 FL 侧确认）。
2. **FL 必然做了「状态重建」**：既然该数据只有动作侧，FL 的 `cache_builder.py` 的做法是「**起真引擎重建对局**」（`producer.py` 逐帧产生 IL 序列）⇒ 「动作 → 引擎重放 → 状态」这条管道**在 FL 侧是成立的**。我们**有完整引擎**（`battle.py`），⇒ 技术上**同构可行**，但两项**未验证**：① 我们的引擎对**真人卡组/卡等/20Hz tick 的出牌序列**的可重放性（FL 用的是它自己的 native 引擎）；② **许可未声明**（`license=None`）⇒ 在我们仓的纪律下**不得直接用于训练**，须用户拍板。

---

## 5. 三轴对照 doc 的独立复核

> 复核方式：不采信转述，逐条在**当前树**（`ae09a6b` / `mask-diff-gate`）上重跑 grep/read；外部条目另取一手原文。

### 5.1 复核成立（编排者实测）

| 锚点 | doc 说法 | 我的复核 | 结论 |
|---|---|---|---|
| `_make_trainer` 只传 8 参 | `rl/run_league.py:840-843` | ✅ **事实成立**；⚠️ **行号已漂移** → **`rl/run_league.py:722-725`**（doc 基于 `267455b`，此后 T2-3 等重构移动了行） | 成立 |
| `PPOTrainer.__init__` 默认 | `n_epochs=1, minibatch_size=0, shuffle=False, value_norm="none"` | ✅ `rl/ppo.py:87-89` 逐字一致 | 成立 |
| `train_solo` 传全参 | `:1141-1143` | ✅ 实际为 `:1141-1146`，传 `value_norm/diagnose_every/n_epochs/minibatch_size/shuffle/seed` | 成立（行号 +3） |
| `_DEFAULT_REWARD` 只有 16 键 | `env_wrapper.py` | ✅ 现位于 **`rl/reward.py:28-46`**（T2-7 搬运）；`config.DEFAULT_REWARD` = **22 键**；**差集恰为 6** | 成立 |
| FL IL 阶段 6 项损失 | `imitation.py:283-290` | ✅ 逐字一致（gate/candidate/target/delay/continue/value） | 成立 |
| FL PPO 无 KL 锚定 | — | ✅ 锚定词表（`kl_to/kl_ref/ref_policy/reference_policy/kl_coef/kl_penalty/bc_regular/anchor_loss`）在 `training/v4/**.py` **0 命中** | 成立 |
| `_mask_diff_snapshot.py` 无 assert、恒 exit 0 | H5 | ✅ 旧版 **92 行、`assert`/`sys.exit` 0 命中**；**且我另查出一处 doc 未提的洞**（见 §8） | 成立且更严重 |
| FL 训练侧 6 项精细奖罚系数全 0.0 | `launch_ppo_multinode_training.sh:61-77` | ✅ **实质成立、引用错位**：该脚本 **77 行**里 `:64-77` 是 `exec torch.distributed.run` 的参数表，**没有任何奖励系数**；真正的默认值在 **`train_ppo_self_play_cluster.py:937-975`**（`--personal-elixir-overflow-penalty` / `--unilateral-elixir-overflow-penalty` / `--hog-deploy-reward` / `--fireball-king-activation-penalty` / `--first-hog-timing-reward-max` / `--first-hog-missed-deadline-penalty` / `--first-hog-deferral-penalty` **全为 0.0**），launch 脚本**不传这些 flag** | **实质成立，引用需改** |

### 5.2 需要修正的一处硬错误 + 两处需加限定

1. ★ **硬错误（doc §1.3-7 的成因未交代，且我查出的精确形态更严重）**：doc 说「`_DEFAULT_REWARD` 只有 16 键（缺 `draw_penalty` + 5 个 `engagement_trade*`）… 评估 env 靠 `lose_penalty` 回退，**数值巧合相同**」——**成立**，但**成因与风险未写全**：
   - 差集 **6 个键**里，**5 个 `engagement_trade*` 是 config-only（设计如此）**，**只有 `draw_penalty` 是真实缺口**（`rl/reward.py:199` `draw_penalty = rw.get("draw_penalty", rw["lose_penalty"])` ⇒ 裸 `RLEnv()` 走回退，恰好等于 `lose_penalty=10.0`）；
   - ⚠️ **且 `rl/reward.py:25` 的注释写着「与 `rl/config.DEFAULT_REWARD` 保持一致；勿单独改一处」——该注释与代码不符**（6 个键不一致）。这直接关系【R7】（尺度改动必须单常量源 + 对账 selftest）：**注释声称单一来源，实际不是**。⇒ 记为**候选下一步（低成本、可对账）**：要么改注释说清 5+1 的差别，要么加一条断言白名单。
2. **§0 总账「FL 6 项奖罚在发布入口系数全 0.0 **空转**」应加限定**：这是**发布脚本**的事实（默认 0.0 + 未传 flag），**不等于** FL 实际训练时它们是 0 —— doc 自己的 **H2**（出货 ckpt 内实测 `gae_lambda=0.99` vs 服务端默认 `0.98`）已证**存在未记录的覆盖**，而 FL 无 `runs/`、无 `*.log` ⇒ **实际运行值不可由仓库反推**。建议统一写法为「**发布入口全 0；实际训练值不可由仓库反推（H2/H3）**」。
3. **§3.1「掩码×引擎关系：结构上不可漂移」应加限定**：doc 自己的 **F-2** 说 `decoding.py:102-127` 的包内规则是**独立复刻、未见对账测试** ⇒ 正确写法是「**主路径**（`arena.py:705 → :752 → :616`）结构上不可漂移 + **一条残余同构路径**」，而**不是整条轴**不可漂移。doc 的分节已并列，但 **§0 总账只取了强的一半** ⇒ 总账与分节**强度不一致**（本次复核发现的唯一「内部不一致」）。
4. **§1.2「塔血定价 我们更强」宜限定为「可调性/表达力更强」**：FL 的 Φ 是**严格 PBRS**（理论正确性），我们是三层串联的**可调结构**（归一化 → `tower_premium_k` 凹形溢价 → `king_gate`）。「更强」在不同维度上指向不同一侧，同 doc §0 已并列「奖惩：我们（可调性）/ FL（理论正确性）」。

### 5.3 doc 未覆盖、但对「奖惩轴」最关键的三点

1. **FL 的 IL 阶段零奖惩（见 §0 结论 1）** —— doc 把 FL 的奖惩只当「引擎 2 项 + 训练侧 6 项」，**漏了 IL 阶段**这一整块。而这恰恰是用户这次问的入口。
2. **FL 的 IL→PPO 锚定项默认关闭（见 §0 结论 2）** —— doc §1.2 表里**没有这一行**。补齐后，「FL 有 KL 锚定」这个说法**不成立**；真正有锚定的是 VPT。**这条会直接改变「我们该不该加锚定」的推理前提**。
3. **CR 人类数据的真实可得性（见 §0 结论 5）** —— doc 的**不可比项 10** 写「IL 数据来源：FL 25.2 万人类回放 vs **我们无人类数据**」，把差距写成**静态事实**；本次复核把它改成**可操作的**：数据**公开在 HF**、**只有动作侧**、**状态靠引擎重建**、**许可未声明**。

### 5.4 影响面排序（我的判断，按「若成立会改变多少既有解读」）

| 排位 | 事项 | 影响面 | 现状 |
|---|---|---|---|
| **1** | **H1**：`run` 模式 5 个开关代码上不生效 | 影响**所有 `run` 模式历史 run**（`run100k`、`long1m`）的口径解读与成本标定 | 代码事实**成立**；与文档/成本标定的**矛盾成因未定**（【R10】）。**低成本定论路径**：给 `run` 日志补 `gs=` 字段（当前 `[step N]` 只有 `policy/value/entropy` 三字段，**无法从既有日志反证**） |
| **2** | **H5**：掩码对账无门禁 | 【R13】这条红线**此前没有脚本保证** | ✅ **本轮已执行**（§8） |
| **3** | **16 vs 22 键 + 注释与代码不符** | 「单一常量源」的声称与实际不符（【R7】） | 已定位到**精确的 5+1 形态**；修复**未做**（须用户/后续拍板） |
| **4** | **BarbLog 352 格缺口**（`P1-20` 日志 1,644 行 **100% 是 BarbLog**） | solo **实时暴露**（BarbLog 在 `FOUR_DECK_SET`） | **未修**（须走【R13】位图 + 单一权威源，见 §6 的 I3） |

---

## 6. 对我们的参考：判定表

| 判定 | 项 | 依据 / 前提 |
|---|---|---|
| ✅ **可做（本轮已做）** | **H5 掩码对账门禁化** | 见 §8（已提交、已推分支、判别力已证） |
| ✅ **可做（最高性价比，须预注册）** | **I2 时序化 BC**（stateful + teacher forcing + TBPTT） | 唯一能给 BC 注入「跨帧不花钱」能力的通路；与 robomimic「时序建模在人类数据上强」独立同向；**不增加覆盖率** |
| ✅ **可做（先决条件，与奖励无关）** | **I3 掩码契约**（采样即校验 + 合法性单一权威源） | 绝悟 A1（`legal_action`/`sub_action_mask` 是 API 契约）+ FL 的「掩码委托引擎校验」；走【R13】位图 |
| ✅ **可做（仪器）** | **I1 BC/NLL 对账** | 现成代码；不针对病根 |
| 🔶 **待预注册（新，本轮提出）** | **CR 人类数据可行性最小验证**：能否用 `IL_Replay` 的动作序列驱动**我们的引擎**重放并复现终局塔血？ | **前置两项**：① 许可未声明 ⇒ 须用户拍板；② 引擎可重放性未验证。**验证成本低**（单文件样本 + 现有引擎） |
| 🔶 **待预注册** | I5 优势加权 / I6 return-conditioning / I7 DAgger | 效力**全部依赖「先能采到目标局面」**（= O7）；I6 另有一手反证（随机环境失效） |
| 🔶 **待预注册（须 ≥2 seed/臂）** | 任何 I2/I5 的**疗效**判据 | 【R5】【R16】：n=1 胜率层**不可归因**（同配置重跑单点差 **0.60**） |
| ❌ **不做** | **I8 KL/BC 锚定** | 方向与病根相反（把分布**收窄**）；RLHF 的 KL 以「学出来的 RM」为前提，我们**没有 RM**；触发【R11】。✅ **但先例要引对**：若将来做锚定，正确先例是 **VPT 的 KL-to-frozen-prior**（有消融），**不是 RLHF 的 KL 去 reward hacking**（一手证据说它 ≈ early stopping） |
| ❌ **不做** | **「奖励塑形 + 退火」这个 borrow** | 是对 **OpenAI Five 的误读**（`anneal` 无一处指奖励权重；奖励函数「constructed once」） |
| ❌ **不做** | 照搬绝悟 **`No-op` 惩罚** | 我们的「不作为」是**圣水经济**造成的（实测：抬 stop 概率把圣水 ≥6 帧从 0% 抬到 **48%**，代价是每局回报 **−4.46 → −41.5**）⇒ 会**误治病灶** |
| ❌ **不做** | 把外部项目的奖励权重/阈值当我们的基线 | 【R15】【R17】；**量纲不可比**（±10 级 vs ±1 级 + β=0.05；决策帧 0.5 s vs 5 tick 0.25 s） |

---

## 7. 反面清单（明确**不建议借鉴**）

1. **「学 OpenAI Five 那样把奖励塑形退火掉」** —— 该说法在 v1 文本中**不存在**。真被退火的是**超参**（lr / GAE horizon）与**新环境/动作特性的 0%→100% 引入**。
2. **「IL 项目都有 KL 锚定，所以我们也该有」** —— FL **没有**（默认关），绝悟 **找不到**，AlphaStar 有但**参考分布是人类策略**（我们**无人类数据** ⇒ 参考分布不存在）。
3. **「用 RLHF 的 KL 惩罚防 reward hacking」** —— 我们的奖励是**手工 22 键表**，不是学出来的 RM；且一手证据（arXiv:2210.10760 §3.6）说 KL 对金标分数的作用「**akin to early stopping**」，该文其余实验**把 KL 设为 0**。
4. **「绝悟的 5 个价值头 = 奖励分解，我们也照做」** —— 我们的价值侧**已确证沉睡**（**C3**：EV 11 点里 9 点 ≈0 或负；**O2**：价值头隐藏层对几乎全部帧**输出恒零**；**X6** 已闭合 critic 侧）。**在 critic 活着之前**，多价值头只会把「常数」复制 5 份；且【R11】禁止在 A′ 类取证前拆价值头。
5. **「MineRL 2019 有 IL 单独夺冠，所以我们也该走 IL」** —— 那条证据只说明**IL 不被环境交互卡死**，不说明**我们的专家能产出目标轨迹**；S1 已证我们的规则专家在 Xbow 卡组 **0 帧**触发 `save_ace`/`setup_wait`（O7）。
6. **「把 DAgger/expert iteration 先跑起来」** —— 在专家本身不产目标轨迹时，它**放大同一个局部最优**；且【R11】禁止专家迭代（≈120× 每帧经验成本）。

---

## 8. 本轮执行的一件小事：把【R13】的 128 张对账从「流程纪律」变成「脚本门禁」

**触发**：三轴 doc 的 **H5**（`scripts/_mask_diff_snapshot.py` 无 assert、恒 `exit 0`）。复核成立，**且我另查出一处 doc 未提的洞**：

| 洞 | 旧版行为 | 后果 |
|---|---|---|
| ① 无比较 | 只 `np.savez_compressed` + print | 「128 张逐位全等」**靠人工纪律**，脚本不保证 |
| ② **异常被吞**（doc 未提） | `except Exception: cells = np.zeros((32,18))` | `legal_cells` 抛异常 ⇒ **全零位图**，两侧同时坏仍「逐位全等」= **假绿** |

**已落地的改动**（`scripts/_mask_diff_snapshot.py`）：
- 新增 **`--compare A.npz B.npz`**：rc=0 **仅当** 键集相同 + 每张位图 `np.array_equal` + 两侧构造异常数均为 0；
- 新增 **`--selftest`**：**5/5 判别力断言**（identical 必 PASS；**翻 1 位** / 键集不同 / 单侧构造异常 / 形状不同 **必 FAIL**）；
- 构造异常**计数写进快照**（`__meta_errors__`），快照自身带异常时 **rc=2**（不再静默）；
- stdout 保持 **ASCII-only**（本仓 GBK 陷阱）；保留原快照调用形式不变。
- 文档同步（【R18】）：`docs/agents/redlines.md` R13 全文 + `AGENTS.md` R13 行。

**实测留证（编排者亲跑）**：

| 用例 | 命令 | 读数 |
|---|---|---|
| 判别力自检 | `--selftest` | **[selftest] 5/5 assertions PASS**，rc=0 |
| 真实树快照 ×2 | `_mask_diff_snapshot.py /tmp/md_a.npz`（×2） | `snapshot: 128 maps`，两侧 rc=0（**128 = 8 状态 × 双方 × 8 手牌**，算术与实数一致） |
| **正向门禁** | `--compare /tmp/md_a.npz /tmp/md_b.npz` | **`[OK] 128/128 maps bitwise identical; build errors a=0 b=0`**，rc=0 |
| **负向（真实数据翻 1 位）** | 副本翻 `level14|p0|Giant[5,5]` 后 `--compare` | **`[FAIL] 1/128 maps differ`**，**rc=1** |
| 旧快照（新 vs 旧） | 去掉 `__meta_errors__` | `[FAIL] key sets differ`，rc=1 |
| 旧快照（旧 vs 旧） | 两侧都无 meta | `[FAIL] bitmaps identical but snapshots had build errors: a=-1 b=-1`，rc=1 |

⚠️ **行为变更须知**：**改动前 dump 的旧快照会被拒判**（无法排除假绿）⇒ 需要重新 dump 一次（一条命令）。

---

## 9. 未验证 / 未闭合 / 需外部信息

| # | 项 | 性质 |
|---|---|---|
| U1 | **AlphaStar Nature 正文**（`nature.com` SSO 302；无 arXiv 预印本） | ⇒ 报告里**没有一句论文原文引用**；IL 回放数、pseudo-reward 构造均为**复现件/二手**；**权重/退火「未找到」≠「不存在」** |
| U2 | FL 的 **`cache_builder --source-count` 默认 100** 与「252,238 局」的关系 | 需要 FL 的 run 级 manifest（仓库未提交） |
| U3 | `VanguardX101/IL_Replay` 的**许可**（`license=None`）与**版权归属** | 决定能否用于训练；**须用户拍板** |
| U4 | 我们引擎对**真人出牌序列**的可重放性（卡等/卡序/20Hz tick 对齐） | 未做最小验证 |
| U5 | MineRL **`reward_shaping` 成就 delta 公式**（三篇论文 grep 0 命中，官方仓库 404） | 标 `未验证`，不得引用为事实 |
| U6 | MineRL **2020/2021 结果报告** | 未找到 ⇒ §4 的「BC 单独够」**只引 2019 届** |
| U7 | **VPT 的「60× 奖励缩放」** | ★ **该数字在 VPT 全文不存在**（我独立复核：文中只有 EWMA 值函数归一化与回报量级 **13 / 25**）⇒ **不得引用** |
| U8 | 绝悟「奖励权重退火」、5v5 价值头权重 `w_k` 数值、kill 为负的机制归因 | 一手文献**未给**；⛔ 不得臆造 |
| U9 | **RLHF 的一手细节**：HH 的 pairwise 损失（在 Askell 2021 §3.1，**不在 HH 本文**）、InstructGPT 无 SFT 损失公式、TRL `PPOTrainer` 已于 **2026-09-04 从 main 移除** | 已登记，防止后续引用出错 |
| U10 | 三轴 doc 的 **H1/H2/H3/H4** | H1 代码事实成立、矛盾成因未定；H2/H3/H4 均需**外部或额外测量** |
| U11 | 我们项目 `rl/reward.py:25` 注释与代码不符的**修复** | 本轮只定位，**未改**（改动须走【R7】对账） |

---

## 10. 来源清单

### 10.1 本地（`file:line` 可直接查；取数时间 2026-09-19）

| 主题 | 位置 |
|---|---|
| H1 锚点 | `rl/run_league.py:722-725`、`rl/ppo.py:87-89`、`rl/train_solo.py:1141-1146` |
| 16 vs 22 键 | `rl/config.py:37-68`、`rl/reward.py:25/28-46/199`、`rl/config.py:509`、`rl/env_wrapper.py:99` |
| IL/BC 现状 | `rl/train_bc.py:40/60/107`、`rl/human_play.py:205-233`、`rl/observation.py`（`history|stack|prev_` 0 命中） |
| 掩码门禁 | `scripts/_mask_diff_snapshot.py`（本轮改）、`docs/agents/redlines.md` R13 |
| FL IL 损失 | `native_runner/training/v4/imitation.py:283-290`；IL 7 文件 `penalty` 0 命中 |
| FL 六项系数默认 | `native_runner/training/v4/train_ppo_self_play_cluster.py:937-975` |
| FL launch 脚本 | `native_runner/training/v4/launch_ppo_multinode_training.sh:64-77`（77 行全文核） |
| 我们的台账 | `docs/agents/ledger.md`（C14/O7/X-19/O8/O9/O10/C3/O2/X6）、`docs/s1_gate_2026-09-18.md`、`docs/elixir_saving_audit_2026-09-18.md`、`docs/online_measure_2026-09-18.md`、`docs/illegal_action_layers_2026-09-18.md` |
| 本轮分报告 | `docs/il_reward_mechanism_synthesis_2026-09-19.md`、`docs/alphastar_reward_il_survey_2026-09-19.md`、`docs/juewu_research_2026-09-19.md`、`docs/rlhf_kl_penalty_survey_2026-09-19.md`、`docs/vpt_minerl_dd_2026-09-19.md`、`docs/il_human_data_survey_2026-09-19.md`、`docs/openai_five_reward_anneal_audit_2026-09-19.md` |

### 10.2 外部（URL + 取数时间 2026-09-19；★ = 编排者独立复核过）

| 项目 | URL | 取到的关键事实 |
|---|---|---|
| ★ VPT | https://arxiv.org/html/2206.11795v1 | Eq.2 `L_klpt=ρ·KL(π_pt,π_θ)`；Table 6 ρ=0.2 / decay 0.9995；「replaces the … entropy maximization loss」；无 KL 消融（100k 局停滞 / catastrophic forgetting）；**无「60×」** |
| ★ OpenAI Five | https://arxiv.org/e-print/1912.06680v1 ・ https://arxiv.org/html/1912.06680v1 | Appendix G Table 5（21 项权重）；「constructed once at the start of the project」；`anneal` 7 处**均非**奖励权重；team spirit 0.3→1.0 |
| ★ CR 数据 | https://hf-mirror.com/datasets/VanguardX101/IL_Replay（API 复核：`private=False`、`license=None`、109 文件、lastModified 2026-09-06） | 252,238 局 / 17,836,160 动作；动作侧 schema；`"hp"` 0 次 |
| MineRL 2019 结果报告 | https://arxiv.org/abs/2003.05012 | mc_rl 纯示范取得 Round-2 第 2（42.41 vs 61.61） |
| BASALT | https://arxiv.org/abs/2107.01969 | 官方 BC 基线 fine-tune VPT width-x1 |
| AWAC | https://arxiv.org/abs/2006.09359 ・ https://arxiv.org/e-print/2006.09359 | 闭式解 + 加权 MLE；λ=0.3/1.0；丢弃 Z(s) 的消融（door 0%→95%） |
| Decision Transformer | https://arxiv.org/abs/2106.01345 | return-to-go 作条件 token；DT ≈ 掐尖 %BC（56.1 vs 56.7） |
| DT/RvS 反证 | https://arxiv.org/abs/2205.15967 | 随机环境下「fail dramatically … not due to a lack of data」 |
| IQL | https://arxiv.org/abs/2110.06169 | expectile + 优势加权回归；D4RL locomotion 692.4（BC 466.7） |
| RLHF（KL 作用） | https://arxiv.org/abs/2210.10760 §3.6 | 「the effect of the penalty … is **akin to early stopping**」 |
| InstructGPT | https://arxiv.org/abs/2203.02155 | Eq.(2) `r_θ − β·log(π_RL/π_SFT)`；β=0.02 |
| Anthropic HH | https://arxiv.org/abs/2112.00861 | λ_KL=0.001；「might actually be wholly unnecessary」 |
| robomimic | https://arxiv.org/abs/2108.03298 | 时序建模在**人类数据**上强（BC-RNN/HBC/IRIS） |
| DAPG | https://arxiv.org/abs/1709.10087 | `w=λ₀λ₁^k max A^π`，λ₀=0.1、λ₁=0.95（**arXiv id 已纠正**：`1910.10314` 是物理论文） |
| AlphaStar | https://doi.org/10.1038/s41586-019-1724-z（**正文未取回**） | Nature 575:350–354 元数据；无 arXiv 预印本 |
| 绝悟 / JueWu | arXiv `1912.09729` / `2011.12582` / `2011.12692` / `2209.08483` / `2110.14221` | 三张奖励表 + 掩码契约 + 课程；全文留证 `docs/juewu_research_2026-09-19/` |

---

## 附：本报告对纪律条款的自查

| 条款 | 执行情况 |
|---|---|
| 【R3】描述性 ≠ 判据 | 全文读数只作**描述**；§6 的判定均为「可做/待预注册/不做」，未把任何观察升格为判据 |
| 【R5】【R16】分辨率 | I2/I5 的**疗效**一律标「须 ≥2 seed/臂」；引用 `et_solo100k` §11.13.11 的单点差 **0.60** 作为「n=1 不可归因」的依据 |
| 【R7】【R11】【R13】【R17】【R19】 | §6 判定表逐条映射；【R13】本轮**补了门禁**；口径冲突（16 vs 22 键、FL 六项「空转」的强度、AlphaStar 是否用手写塑形）**并列不调和** |
| 【R18】改代码同步文档 | `_mask_diff_snapshot.py` 的行为变更**同一步**写进 `docs/agents/redlines.md` R13 与 `AGENTS.md` R13 行 |
| 【R10】不确定就写不确定 | §9 共 11 条未闭合；外部报告中的「未验证」项**原样保留**（未升格为事实） |
| 未做声明 | **未跑训练**；**未跑全量 selftest**（【R19】）；**未改任何训练/奖励代码**；本轮唯一代码改动 = 掩码**只读对账脚本的门禁化** |
