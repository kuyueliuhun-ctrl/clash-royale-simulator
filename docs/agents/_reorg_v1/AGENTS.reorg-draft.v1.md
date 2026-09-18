# AGENTS — 红线速查 + 决策索引（跨会话必读）

> **⚠️ 这是整理草稿（2026-09-19），不是生效文件**：与 `AGENTS.md` 的差异**只有 B 区第 77–86 行**（10 行巨型行 → 一行式索引 + `docs/agents/index_notes.md` 锚点），其余逐字相同。正文全文已**逐字**搬到 [`docs/agents/index_notes.md`](docs/agents/index_notes.md)。
> **本文件 = 红线（每条一行）+ 索引。** 全文/证据/口径/台账在 [`docs/agents/`](docs/agents/)，**按需打开**。
> **2026-09-18 拆分**：单文件曾涨到 **65,196 B**，而实测注入上限 **65,244 B** ⇒ **余量仅 48 B**，
> 任何一次编辑都可能让**末尾被截断**（= 2026-09-13 那次失效模式的复发：旧文件末尾 35 KB 对会话不可见）。
> 现拆为 6 个文件，`AGENTS.md` 只留**红线一行式摘要 + 索引 + 当前活跃工作**。
>
> **⚠️ 节号保持不变**：分册里保留原 `## N.` / `### N.M` 编号 ⇒ 历史文档里的「`AGENTS.md` §2.1」
> 这类引用**经本文索引一跳可达**，不必逐份改写。
> **纪律不变**：① 新决策**先写 `docs/`** → 再在本文件加**一行指针**；② **只增不改历史结论**
> （被推翻时保留原条目并标注「已被 X 推翻」）；③ 条目编号可引用：`【红线 R7】` `【确证 C5】` `【否证 X2】` `【未决 O3】`。

---

## A. 红线速查（19 条；**全文 + 事故证据** → [`docs/agents/redlines.md`](docs/agents/redlines.md)）

| # | 红线 | 必记要点 |
|---|---|---|
| **R1** | **训练期"性能异常"先归因外部，暂停留证等人类** | 症状先归因外部、**不自动降级**；长跑前 `scripts/check_commit.py`；`--eval-workers 12` |
| **R2** | **训练语义兼容红线** | `ppo.py` 默认参数 = 旧行为；新能力只经 `TrainConfig` 显式开 |
| **R3** | **单变量 + 预注册** | **判据跑前写死**（含失败分支） |
| **R4** | **阈值判据的基线列必须脚本复算，禁止手抄** | 基线用 `scripts/judge_anchor_blocks.py --groups` **复算**，禁手抄 |
| **R5** | **n=1/臂 分辨率不足** | 20k 单跑只能看**大效应**；跨 run 数字不可直接比 |
| **R6** | **架构变更必须 `--fresh`** | 热启动须显式 `plan_dim=58` / `belief_dim=563`；缺 `enc_ln`/`grid_ln` 会静默错 |
| **R7** | **尺度改动必须单常量源 + 对账 selftest** | 汇率 / 值函数 / 闸门 `edw×卡费` / MCTS **同源同步改**；**2026-09-19 起奖励表一致性有可执行对账** `scripts/selftest_reward_tables.py`（6/6；22 键 vs 16 键差集恰为 6、`draw_penalty` 靠 `.get` 回退；`rl/reward.py:25` 注释已按事实改写） |
| **R8** | **每个修复配回归测试；selftest 必须跨局边界** | 回归测试必须**跨局边界**；9j 指纹看日志头 30 行 |
| **R9** | **测量口径纪律** | `step`=决策帧；EV 用**更新前池化**；探针**按局分组留出** |
| **R10** | **不确定就写不确定** | 写「成因未定」，不写自信的错答案 |
| **R11** | **明确不做（未获用户另行拍板前）** | **不在 A′ 类取证之前改奖励**；不上训练时 MCTS；不扩参 |
| **R12** | **EBK：能算的不许让网络猜** | 确定性可算量 → **特征注入**，不让网络猜 |
| **R13** | **改判定逻辑必须位图对账** | 掩码/合法格改动跑 `scripts/_mask_diff_snapshot.py`（128 张**逐位全等**）；**2026-09-19 起该脚本自带门禁** `--compare A B` + `--selftest`（旧版无比较/无退出码、异常吞成全零位图 ⇒ 假绿；旧快照现被拒判，需重 dump） |
| **R14** | **判 critic 好坏不许只看 EV，必须看「参数有没有在动」** | 逐窗 `‖ΔW‖/‖W‖`；解读前先确认**前向路径** |
| **R15** | **判据阈值不许跨实验照抄** | 阈值在**本实验自己的量纲**上标定 |
| **R16** | **判据阈值不许用单次观测标定** | 阈值余量 **≫ run 间散布**；否则改用**比值/配对**判据 |
| **R17** | **判据的分子/分母必须来自同一超参约定**；超参跨特征不可共用时必须报「两套约定」并只采信方向一致的部分；凡比较**包含关系**（`A ⊇ B`）的两层，**必须逐位验证包含**（`np.array_equal`）**并在同一超参下对账** | 分子/分母**同超参**；包含关系**逐位验证** |
| **R18** | **改代码必须同步维护既有文档**（2026-09-14 用户拍板） | 改代码**同一步**改文档；收尾报告列出改了哪些 |
| **R19** | **默认不跑全量 selftest**（2026-09-14 用户拍板） | 默认只跑相关子集：`run_selftests.py test_<名>` |

---

## B. 决策索引（我要找什么 → 读哪个文件）

| 我要找什么 | 去哪 |
|---|---|
| **红线全文 + 事故证据（R1–R19）** | [`docs/agents/redlines.md`](docs/agents/redlines.md) §1 |
| 解释器 / venv / 统一运行姿势 / GBK 陷阱 | [`docs/agents/env.md`](docs/agents/env.md) §2 |
| 三种训练模式（`solo` / `run` / `flow` 语义与选择） | env.md §2.1 |
| **标准 20k 协议**（与历史 run 可比） | env.md §2.2 |
| 100k 长跑 + 评估节奏 C（`--anchor-every`） | env.md §2.3 |
| 判读 / 诊断工具表（含 forensics / probe / 子集 selftest） | env.md §2.4 |
| 仪表盘（**端口 8700**；8090 已被系统保留，废） | env.md §2.5 |
| 自检命令（【R19】只跑相关子集） | env.md §2.6 |
| 训练口径：`step` 语义 / 评估口径 / 噪声地板 / **判读禁则** / 平局与早停 / run 成本 | [`docs/agents/metrics.md`](docs/agents/metrics.md) §3 |
| **关键常量表**（改代码前先对照） | metrics.md §3.1 |
| **已确证 C1–C14**（可作前提引用） | [`docs/agents/ledger.md`](docs/agents/ledger.md) §4 |
| **已否证 / 已关闭 X1–X19**（**勿重走**） | ledger.md §5 |
| **未决 O1–O10 + 候选下一步** | ledger.md §6 |
| 计划台账（哪条路走通 / 被否 / 未决） | [`docs/agents/plans_runs_docs.md`](docs/agents/plans_runs_docs.md) §7 |
| **Run 台账**（每个 run 一句话结论 + 证据路径） | plans_runs_docs.md §8 |
| **文档地图**（想做什么 → 读哪份 `docs/`） | plans_runs_docs.md §9 |
| 历史决策全文（旧 `AGENTS.md` 逐字冻结） | [`docs/agents_archive_2026-09.md`](docs/agents_archive_2026-09.md) |
| **`long1m` 提前终止的账**（12.8% 的实际状态） | [`docs/long1m_stopped_2026-09-18.md`](docs/long1m_stopped_2026-09-18.md) |
| **当前新方案**：按局面结算的圣水交换信用分配（§7–§9 规格） | [`docs/frame_credit_proposal_review_2026-09-18.md`](docs/frame_credit_proposal_review_2026-09-18.md) |
| **新方案预注册**（判据 / 失败分支 / 不变量 / S1–S3；**实现前必读**） | [`docs/engagement_trade_prereg_2026-09-18.md`](docs/engagement_trade_prereg_2026-09-18.md) |
| **S2 实施进度**（schema 3→4 实体带 `id`+`target_id`；§11.5 = 仪器 + 门禁判读） | 同上预注册 §11 |
| **★ S2 实测与门禁判读（第二轮已重算）**（P3 推翻 / P1P2 否决 / **P4b 配对 10/10** / 6 个口径缺陷 / 门禁兑现率） | [`docs/s2_instrument_2026-09-18.md`](docs/s2_instrument_2026-09-18.md) |
| **★ S1 零成本门禁**（手写专家；判「约束在机制/探索侧」+ 冰人白嫖证实） | [`docs/s1_gate_2026-09-18.md`](docs/s1_gate_2026-09-18.md) |
| **引擎侧出牌溯源通道 + 受控落地**（**已落真树**；三步验证 4/4） | [`docs/root_cast_channel_2026-09-18.md`](docs/root_cast_channel_2026-09-18.md) |
| **★ 在线奖励项（默认关）+ 开/关对账 + 两处实质不确定 + CRLF 事故** | [`docs/engagement_trade_online_2026-09-18.md`](docs/engagement_trade_online_2026-09-18.md) |
| **★★ 在线口径实测：`p4b` 优势不成立（Δρ 从 +0.145 掉到 +0.008；250 局）** | [`docs/online_measure_2026-09-18.md`](docs/online_measure_2026-09-18.md) |
| 长跑工程（两级评估 + 评估并行分片） | [`docs/long1m_prereg_2026-09-18.md`](docs/long1m_prereg_2026-09-18.md) |
| **`run100k` 启动记录（正在跑，100k 两级评估）** | [`docs/run100k_2026-09-18.md`](docs/run100k_2026-09-18.md) |
| 仪表盘对接长跑（进度条 / 大点 / 逐对手胜率） | [`docs/dashboard_long1m_2026-09-18.md`](docs/dashboard_long1m_2026-09-18.md) |
| **★★ `et_solo100k` 两臂干预长跑：已跑完 + 判读**（含**三次发射** + **六处测量侧更正** §11.13.6/7/9/10/12/13 + **★分辨率实测标定 §11.13.11**；**结论 = 没有任何一项能用现有仪器分辨出疗效，不构成判决**） | 判读 [`docs/et_solo100k_judgment_2026-09-18.md`](docs/et_solo100k_judgment_2026-09-18.md)、读数 [`docs/readout_et_solo100k.md`](docs/readout_et_solo100k.md)、run 记录 [`docs/et_solo100k_2026-09-18.md`](docs/et_solo100k_2026-09-18.md)、台账 **O8** |
| **★ 训练健康指标**（价值损失 / 策略熵 = **本来就有、只是 UI 看不见**；**能不能判「步数瓶颈」**；A_et 实测熵平台读数；GBK 陷阱新证据） | [`docs/train_health_metrics_2026-09-18.md`](docs/train_health_metrics_2026-09-18.md) |
| **★ 「不出牌」概率（P(STOP)）实测**（用户提问）：**开局帧** A_et **0.052** / B_ctrl 0.034 / 未训练 0.081；**有牌可出时的选择概率** 0.058~0.09（未训练 0.15~0.20）；**全程 ≈0.90 是"没得选"**——**88%~91% 的帧买不起任何手牌**（圣水中位 1.50），「不裸下」门禁 **0** 帧；⇔ 纠正 `P(STOP)=0.945/0.929` 的老读法 | 读数 [`docs/pass_prob_2026-09-18.md`](docs/pass_prob_2026-09-18.md)、仪器 `scripts/probe_pass_prob.py` |
| **★ 攒费穷举实测：「学不会攒费」成立且更狠** | 「放弃一张买得起的牌」**75,891 帧 0 次**；7,238 段主动不出牌**0 段**在圣水 ≥4 时仍不出（y≥4 必有可出）；A 段最长 **22 < 34 帧** ⇒ **这条轨迹从未被采样**（0 样本 ⇒ 0 梯度，step 0 与 100k 同结构）。★ 附带查出 **B 类病理**（非法卡包整包被拒 = 白掉一帧：497 帧/0.7%、39/280 局、最长 225 帧 ≈112 s、跨 9 评估点），污染了「圣水≥6 = 0.52%」读数（**96% 是它**），5 次 Xbow 全在 B 解锁帧 ｜ 审计 [`docs/elixir_saving_audit_2026-09-18.md`](docs/elixir_saving_audit_2026-09-18.md)、仪器 `scripts/pass_streak_audit.py`、台账 **C14 / O9**（O3 已补记）；长条目全文 → `docs/agents/index_notes.md` **§1** |
| **★★ 修复：掩码 `used` 0/1-based off-by-one（B 类根因）** | `used` 存 **0-based** 却用 **1-based** `sa.slot` 做成员测试 ⇒ partial 现「(s, s−1)」时低槽位被整段跳过 ⇒ **掩码放行已用槽位** ⇒ 提交重复槽位包 ⇒ `validate_bundle` **整包拒绝**（真实 100k：497 帧/0.7%、39/280 局、最长 225 帧）。**一行修复 + 两条零成本掩码不变式 + 回归测试** `test_mask_partial_bundle_invariants`（判别力已证）；同种子 **44/256→0/256**、拒绝帧 **474/1707→0/1609**、端到端 **B = 0/3413**。⚠️ 与攒费无关（`A ∩ 圣水≥6` 仍 = 0，C14 不变） ｜ [`docs/mask_used_slot_offbyone_fix_2026-09-18.md`](docs/mask_used_slot_offbyone_fix_2026-09-18.md)、留证 `docs/mask_used_slot_offbyone_fix_2026-09-18/`、台账 **O9**；长条目全文 → `index_notes.md` **§2** |
| **★ 探索随机（第一问）：字面「随机×随机+位置随机+正弦衰减」做不到事** | 攒到 Xbow 是**合取**（连续 34 帧不花钱、其中 ~17 帧须逐帧抽中「不出」）⇒ 成功率 `p^k`（**指数在帧数**）。实测 `uniform`（随机开到最满）**2,084 帧圣水一次没到 6**（峰值 5.36，≥25/≥34 帧段均 0）；解析 **4.5×10⁻¹⁰ /100k 帧**（即便放宽到 `p=1/2` 也只有 5.8×10⁻²）。**决定成败的是「时长」不是「随机」**：`hold d~U{1..40}` ⇒ ≥6 帧占 **28.96%/38.81%**、≥34 帧段 **978 次/100k 帧**（差 ~12 个数量级）。落点那半 = 均匀化（实测高度集中 84/90 格、top10 占 73%/78%）；正弦+衰减是**训练步**上的调度 ⇒ 对局内合取**零作用**且裁梯度；三新超参在 n=1 下不可判。§7 三写法对照（推荐**混合即策略**、记 `log π_train`、`p=0` 逐位回旧、【R2】） ｜ [`docs/exploration_randomization_analysis_2026-09-18.md`](docs/exploration_randomization_analysis_2026-09-18.md)、仪器 `scripts/probe_explore_randomization.py`、留证 `docs/probe_explore_randomization/`、台账 **O10**；长条目全文 → `index_notes.md` **§3** |
| **★ 偏置随机（第二问）：抬「不出牌」概率是唯一正确杠杆** | 逐帧 `1/6→q` 把成功率抬成 `q^k`（每帧 5 倍 ⇒ 12 个数量级）；解析 `q=0.9` = **1,264 事件/100k 帧**；实测 `uniform` 0.000% → `bias_0.9` **18.9%**（峰值 10.00）。**可零 off-policy 实现**：给 **STOP logit 加正偏置 β** ⇒ `ratio≡1`，剂量 β=0/4/5/6/7 ⇒ **0.000→0.178→12.70→51.27→82.98%**；仓内已有 `stop_logit_bias=-1.0`（`follower.py:156/253-255`，**仅初始化生效**）。「抬大费牌概率」方向是**反的**（α=1.5 使 ≥6 占比 **−36%**）。★★ **但覆盖率 ≠ 会学**：每局回报 **−4.46 → −41.5/−54.1** ⇒ 采样到了 PPO 也会学「别这么干」（γ 折扣已排除） ｜ [`docs/exploration_bias_pass_2026-09-18.md`](docs/exploration_bias_pass_2026-09-18.md)、仪器同前、留证 `docs/probe_explore_randomization/bias_*`、台账 **O10**；长条目全文 → `index_notes.md` **§4** |
| **★ 状态条件随机（第三问）：低压时才随机 —— 方向对，买回了回报** | 前提成立：门 `d`（最近 15 帧没掉塔血）开 **60.9%** 帧、最长 544 帧、**838 窗口/100k**。同 ckpt / 同 8 局：`sample@0` 0.000% / **+1.63**；`sample@6`（未门控）48.209% / **−56.13**；**`gate@6:d` 6.765% / +1.68**（三轮里**第一个同时成立**）；`gate@6:h` 0.385% / −13.21。代价 **÷7**（离线预测 ÷1.6 是**乐观上界**）；用码内 `PRESSURE_THRESHOLD=2.0` 只开 **4.7%** ⇒ 等于没有。限定：n=8 且基线回报跨 run 摆动（【O5】）⇒ 回报差异**不可当判据**；**「省下的费没花出去」仍未解决**（攒到 6 费 133 帧、Xbow 0 次）⇒ **仍要 O7** ｜ [`docs/exploration_pressure_gate_2026-09-18.md`](docs/exploration_pressure_gate_2026-09-18.md)、仪器同前（`--mode pressure` / `--gate-mode`）、留证 `docs/probe_explore_randomization/{pressure,gated_*,pressure_gate}.*`、台账 **O10**；长条目全文 → `index_notes.md` **§5** |
| **★ 「初始模型 + 全过程随机」100 局评估 + dashboard 观测** | p0 = 全过程均匀随机 actor、p1 = `solo_main_0.pt` 确定性 argmax（同 `--only-vs-main` 约定）；**100 局 / 31,491 帧 / 1,367.9 s**（schema 5、生产同一写入路径）。读数 **58W/42L/0D = 58.0%±4.9**、局均 **314.9 帧**、空 bundle **90.4%**。独立复算：**A 90.4% / B 0.0%（掩码修复的更大样本复核，31,491 帧）/ C 9.6%**；`A ∩ 圣水≥6 = 0`；**圣水峰值 5.357 = 开局 5.0 + 2 帧回费、只出现在每局第 3 帧 ⇒ 100 局里圣水从未超过开局水平**；A 段最长 24 < 34；**Xbow 0 次**（3,062 次部署）；落点 484/576 格、熵 8.47 bits。dashboard **http://127.0.0.1:8701**。⚠️ p0 是**随机 actor 不是策略**（不能用于训练/BC）；对手与 p0 同源 ⇒ 胜率**不含绝对强度** ｜ [`docs/rand100_eval_2026-09-18.md`](docs/rand100_eval_2026-09-18.md)、runner `scripts/random_eval_100.py`、独立复算 `docs/rand100_eval_2026-09-18/`、台账 **O10**；长条目全文 → `index_notes.md` **§6** |
| **★ 「非法动作」四层取证（只读取证，未改代码）** | 四层**并列且互不推导**：**L1 掩码**（只产 `bool`、**不产 reason**）→ **L2 整包校验**（`validate_bundle` **9 条中文 reason**，任一非法**整包拒收**）→ **L3 惩罚/统计**（`invalid_penalty=0.05`（defensive 0.1）× `invalid_count`；两来源：整包拒 =1、引擎拒逐卡 +1）→ **L4 引擎物理**（失败一律**静默 `return False`**）。**10 条冲突 C1–C10 并列不调和**；**F1** MergeMaiden 掩码层「费用过严 + 位置过松」（根因 `_effective_card` 只处理 Mirror）；**F2** 三冲突卡均不在 `DEFAULT_SOLO_DECK`，BarbLog 在 `FOUR_DECK_SET` ⇒ **solo 实时暴露**，`run` 模式 p1 侧非法**不被计数**。真实日志：`P1-20` 共 **1,644 行、BarbLog 100.0%**；**未定 U8–U12** ｜ [`docs/illegal_action_layers_2026-09-18.md`](docs/illegal_action_layers_2026-09-18.md)（§1 分层 / §6 C1–C10 / §10 复核 + F1/F2）、台账 **O9**；长条目全文 → `index_notes.md` **§7** |
| **★★ 全项目结构优化（盘点 + Tier 0/1/2/3）** | **已拆到分册** → [`docs/agents/structure.md`](docs/agents/structure.md)。盘点：**199 `.py` / 62,402 行**、9 个 God file、**引擎→`rl/` 反向边 = 0**、`card_utils` cwd 隐性契约。**Tier 0/1 完成**（死件清理 31、只读核对器 `scripts/_structure_check.py` 自检 13/13、UTF-8 兜底单一实现、硬编码绝对路径 9→0）；**Tier 2**：`rl/dashboard.py` **3,171→566**、`league_rules.py` **8/8 逐字相同**、`card_mechanics.py` **1,896→861**（65 名逐名相同）、T2-7 奖励簇 → `rl/reward.py`（`env_wrapper.py` **812→598**，逐位对账 4,000 例 0 不一致）、`selftest.py` **6,023→172**（100/100 逐字相同、全量 **100 通过 / 0 失败**）；**Tier 3**：T3-2 **取消 cwd 契约**、T3-4 交付核对器 **⑪**「path 引导须先于产品 import」。**方案 §6 的 21 项：完全完成 15 / 部分完成 3 / 判定不做 3**（每条不做带证据） ｜ [`docs/agents/structure.md`](docs/agents/structure.md)、[`docs/structure_remaining_2026-09-19.md`](docs/structure_remaining_2026-09-19.md)、`docs/agents/scripts_inventory.md`；长条目全文 → `index_notes.md` **§8** |
| **★ 外部调研类台账（IL 奖惩参考 / 人类回放 / 录像格式对照）** | 全文**已拆到分册** → [`docs/agents/reward_surveys.md`](docs/agents/reward_surveys.md)（两条长条目**逐字搬入**，第三条随源文档新建）。四条一句话结论：① **IL 阶段结构性没有奖惩**（FL 六项损失全掩码 CE/Huber、`penalty` 0 命中），**IL→PPO 锚定项默认关闭**，有一手 KL 锚定先例的是 **VPT**；② **「奖励塑形退火」无一手实例**（Five 的 `anneal` 是超参/环境特性，VPT 退的是 KL 系数 ρ）；③ **人类回放**：卡键 100% 可映射、卡组 8/8、手牌循环费 6/6 精确相等、坐标**无需翻转**，但直接重放只走通 **35–62%**（主缺口 = **逐事件圣水不可观测**）⇒ 可用于 IL 但**必须先状态重建**；④ 两格式互补（我方 schema 5 无卡组元数据 / 无 seed，`logprob`/`masks` 在 `run_league.py:190` 被丢弃）；★ 卡覆盖 **177/178，唯一缺 `void`**（≈3% 卡组） ｜ [`docs/agents/reward_surveys.md`](docs/agents/reward_surveys.md)、源分册 `docs/il_reward_reference_analysis_2026-09-19.md` / `docs/il_replay_feasibility_2026-09-19.md` / `docs/replay_format_comparison_2026-09-19.md`；长条目全文 → `index_notes.md` **§9** |
| **★ 上游引擎增量核查（原作者 Jason-XII `f616f19` vs 我们）** | 分叉点 **`f20fa4d`**（两仓 `merge-base --is-ancestor` 都 rc=0；我方分叉后 **242** 提交 / 上游 **24**）。★ **退出码陷阱**：`merge-base --is-ancestor f616f19 …` = **rc=128 + `fatal: Not a valid object name`**（上游对象不在本机），**不是**「不是祖先」（那才是 rc=1）⇒ 不得据此断言「两仓无共同祖先」（与 `docs/upstream_claim_audit_2026-09-19.md` 并列不调和）。上游 24 提交里 **6 个动引擎**（4 文件，净 **+52/−33**），**引擎数据与基座零改动**。逐条：① 建筑掉血 `372ed0e` **我们已独立修好**；② 碰撞 `*0.5` **未跟进**（建议先测）；③ 20 Hz **等价**（`env_wrapper.py:88-89`）；④ 河面 A\* **不照搬**（上游最终版是死代码）；⑤ `1e9716b` 仅注释。★ **顺带查出我们自己的四处**（**本轮一行未改，待拍板**）：**P1** `battle.py:3292` 应为 `continue`；**P2** 桥面 16 半格是 `W` ⇒ 名义 3 格实 2 格；**P3（高）** 过河 jump **无界闩锁**（贴脸 18 s、承伤 0、被误判空军）；**P4** `arena.walkable_cache` 模块级全局 + `int()` 截断 ⇒ **一次越界查询永久改图** ｜ [`docs/upstream_engine_delta_2026-09-19.md`](docs/upstream_engine_delta_2026-09-19.md)、合并件 `docs/upstream_four_sections_merge_2026-09-19.md`；长条目全文 → `index_notes.md` **§10** |

---

## C. 当前活跃工作（2026-09-18）

0. **`run100k` 已跑完**（`run` 模式 **100k 步**、**两级评估** 14 点 / 800 局、`--only-vs-main`）
   ⇒ `runs/run100k/`：**14 个 schema-4 录像**（`league_{0,8000,…,100000}.pkl`）+ 14 ckpt；
   启动前【R1】实测 **14 个孤儿 spawn worker 占 7.7 GB** ⇒ 清掉后可用提交 7.51→**23.24 GB**、
   档位判回 12 OK（工具 `scripts/kill_orphan_workers.ps1`）。
   **它的 14 个录像就是 S2 第一/二轮全部读数的数据源**（schema 4 ⇒ 带 §11.9.3 的重建误差）。
   记录：[`docs/run100k_2026-09-18.md`](docs/run100k_2026-09-18.md)
   ⚠️ 新源：`runs/run_schema5/`（落地验证用的短 smoke，**schema 5**，60 局）—— 精确口径的对照数据。

1. **主线变更**：`long1m`（1M 步 `run` 模式）**已按用户指令提前终止**在 **128,000 步（12.8%）**——
   18 个评估点（含插入的大点 100k）、**0 次降级**、**无判读产出且 J1 按定义不成立**。
   账见 [`docs/long1m_stopped_2026-09-18.md`](docs/long1m_stopped_2026-09-18.md)。
   ⚠️ `runs/long1m/`（393 MB，18 ckpt + 18 录像）**保留** —— 它是新方案的现成回放素材。
2. **新方案（实现中）**：按**局面**结算的**圣水交换**信用分配。
   - 局面 = 索敌关系连通分量；活跃 = 分量内有**交战对**；结算 = 连续 K tick 无交战对（去抖，K=30 tick）
   - 净交换 ≈ 窗口内 `−ΔΦ` + **塔血保全**（后者是唯一真·新信息）；阈值用**平滑 hinge**
   - **免费产物只走溯源路由（`root_cast`）、绝不估值**（估值 = 货币泵，见评审 §9.1）
   - ⚠️ 三条红线级约束：① 绝不估值；② **实验必须挂 `solo`**（`run` 对手是 mask-随机，无拉扯样本）；
     ③ 判据须带**优势兑现率**门禁（圣水上限 10 ⇒ 只守不推可攒交易打平）
   - 规格与判据：[`docs/frame_credit_proposal_review_2026-09-18.md`](docs/frame_credit_proposal_review_2026-09-18.md) §7–§9
   - **预注册（判据/失败分支/不变量/S1–S3）**：[`docs/engagement_trade_prereg_2026-09-18.md`](docs/engagement_trade_prereg_2026-09-18.md)
   - **已落地第一步**：录像 schema 3→4（实体 `id` + `target_id`，末尾追加、向后兼容）⇒ 离线可重建索敌关系图；见预注册 §11
   - **已落地第二步（S2 仪器 + 门禁判读；2026-09-18 第二轮已按"修好缺陷 6 之后"的读数重算）**：
     `scripts/offline_engagement_trade.py`；不变量 **8/8 PASS**（`scripts/selftest_offline_engagement_trade.py`，
     合成帧、零引擎依赖）。主口径 = **`--phi-mode global`**（预注册 §2.1 字面读法），`component` 作保守对照。
     ★ **预注册首选 P3 被推翻**（`τ ≤ kill_credit` 恒成立、`τ==kill_credit` 占 75%–89% ⇒ 只是已有 Φ 击杀项的重新加权）；
     **P1/P2 否决**（P1 `τ` 均值 388–431 = `ΔΦ` 量级的 ~200 倍、ρ(胜) **5/5 批为负**、兑现率只有 80.7–88.6%）；
     **唯一候选 = 新构造 P4b**（塔血零掉血门控 × 钉住时长）：**同批配对 Δρ(塔血) 5/5 正、均值 +0.145，
     Δρ(胜) 5/5 正、均值 +0.108；换 `component` 口径同样 5/5 正（合计 10/10）**。
     缺口实测：**20.6%** 的窗口「钉住但没打死」，其 `Trade` 中位 0、均值 **−0.36**（`p25 = −1.00`）⇒「正解被记为负」成立。
     ⇒ **S2 的相关性门禁算过了**（配对 5/5）；但 §5.1 意义上的**阈值标定只能在 S3 的 run 内做**，
     且 **run 模式胜负方差≈0（96/100）⇒ 必须换 `solo`**。全文：[`docs/s2_instrument_2026-09-18.md`](docs/s2_instrument_2026-09-18.md)；预注册 §11.5
   - **★ S2 仪器共修掉 6 个口径缺陷**（判读文档 §7）：身份判据（全成员 ⇒ span=247 帧）/ `K` 死代码且**不可辨识** /
     **零和只算 p0 视角**（另一半局被系统性扣分 ⇒ 相关性被压成 0）/ **`crown` 方向**（`get_crown_count` 返回"让出的皇冠"，
     用错让 ρ 从 +0.92 变 −0.92）/ `tower_term` 与 `ΔΦ` 的重复计数（P3 信号 ~3/4 来自重复计数）/
     ★★ **组件口径把别路的下牌花费算进本窗口**（第一版读数的致命伤 ⇒ 改走净支出形式 + 回归测试）。
     ⚠️ **第一版读数已作废**（判读文档 §7.7 保留原文不改）。
   - **已跑 S1 零成本门禁（零训练成本，手写专家）**：**未通过（不显著 ≫ 基线）**
     ⇒ 按预注册 §5 表**第二行**：**约束主要在探索/机制侧**（`token_strict` 全帧圣水≥6 = **0/8191**、Xbow = **0/929**；
     机制自锁：Xbow 不在 `ACE_CARDS`/`SINK_TANK_CARDS` ⇒ `save_ace`/`setup_wait` 在 Xbow 卡组 0 帧触发、
     `_pick_suggested_card` 只推荐付得起的牌）。**能力上限已交代**：只多加一条不在 PlanToken 语义里的
     手写攒费→Xbow 规则就打出了 **156× 基线**（47/1003，部署前圣水中位 6.25）⇒ **动作空间/掩码被证伪为约束点**，
     **不得**写成「手写专家做不到」。冰人白嫖推论 **证实**（`ΔΦ = −2.000000` 零散布，`edw=0.5` ⇒ **−1.0**；双倍期 −0.2）。
     全文：[`docs/s1_gate_2026-09-18.md`](docs/s1_gate_2026-09-18.md)
   - **★ 第五轮：尾部窗口 flush（已修）+ 把结论冻进台账**
     ① `RLEnv.flush_engagement_trade()`：局末（**含按步数截断**）把仍开着的局面结掉；
     `run_league` 两条对局路径都在局末调用，并把尾部明细**补进最后一帧**的 `et`。
     **实测尾部占窗口总数 8.0%（296/3679，250 局）**，个别局尾部 φ 达 **−129** ⇒
     第四轮的在线测量**少算了这部分**（奖励侧本来也拿不到，见 flush 的 docstring）。
     **补测（250 局）**：Δρ(塔血) **+0.000**（3/5 正）、Δρ(胜) **+0.001**（**2/5 正**）
     ⇒ **幅度归零、方向也不再稳定** ⇒ X-19 由"证据不足"升级为「**已否证**」。
     ③ **solo 口径复核（预注册强制口径；160 局，目标导向 hist 池）**：`τ` **自身** ρ(τ,塔血)/ρ(τ,胜)
     **4/4 批为正**（+0.250 / +0.257）且与 `φ` 大体不冗余（+0.106）——**全调查唯一 4/4 一致的正信号**；
     但**按本方案的定义**加进 `Trade` 的增量 Δρ(λ=1) 只有 **+0.025/+0.022**，λ 放大到 10/30 均值升到 +0.08
     却**出现批内反号**（−0.125/−0.162），配对 bootstrap 95% 区间**全跨 0**。
     ⇒ 精修判决：**τ 有信号，但本方案的写法（加进 `Trade`）增量低于一切可用仪器的分辨率**
     ⇒ 按 §5.3 第 2 条**不启动 S3**（20k 单跑按【R5】只能看大效应）。
     出路：**O7 修机制**，或把 `τ` 做成**独立**奖励项/价值头输入（**新方案，须另行预注册**）。
     ② 结论已冻进台账：**X-19**（本方案已完整取证、判**不接线**；含 P3 推翻/P1·P2 否决/
     离线 vs 在线口径的 ÷18·÷7.7/`τ` 稀疏/门禁无分辨力）与 **O7**（机制自锁 + 候选下一步，
     须单独预注册）。⇒ 索引：[`docs/agents/ledger.md`](docs/agents/ledger.md) §5/§6
   - **★★ 第四轮：用正确口径重测 ⇒ `p4b` 的优势不成立（最终判读）**：
     做法 = 新预设 `economy_etm`（**measure-only**：跑在线监视器、把每个决策帧结算掉的局面写进录像
     `et` 键、**一分奖励都不加**）+ **250 局**（5 评估点 × 50 局）+ `scripts/analyze_online_trade.py`
     的同批配对 ρ。结果：**Δρ(塔血) +0.008 / Δρ(胜) +0.014**（对照离线口径的 +0.145/+0.108
     ⇒ **÷18 / ÷7.7**）；基准也翻号（`ρ(none,塔血)` 在线 **+0.084** vs 离线 −0.198）。
     ⇒ **§11.8 的"配对 10/10 正、均值 +0.145"是口径产物**；**S2 门禁在正确口径下不通过**。
     连带：`τ` 太稀疏（`Στ/Σφ ≈ 1/12`、只有 ~38% 的对局 `τ≠0`）⇒ 换代理要换**口径**，不是调权重。
     全文：[`docs/online_measure_2026-09-18.md`](docs/online_measure_2026-09-18.md)；预注册 §11.10
   - **★ 三条门禁的合并结论（【R11】，2026-09-18 第二轮更新，已被第四轮取代）**：
     **S1 未通过**（约束在探索/机制侧）；**S2 第二轮曾判"相关性过了"**，但第三轮查出**两处实质不确定**：
     ① **在线与离线的局面切分不是同一个量**（同一局：离线 `identity=core` **67 个窗口** vs
     在线 K=30 tick **2 个**；离线 `settle_frames ∈ {1..20}` 结果**一字不变**）；
     ② **离线重建 `V` 的误差首次量出**：严格相等仅 **44.9%**、中位 0.23、**p90 3.0**、max **7.13** 圣水
     （而窗口 `ΔΦ` 量级只有 ±2）⇒ **schema-4 的全部读数都带这个误差**。
     ⇒ **S3 仍不开**；放行条件（写死）见预注册 §11.9.4：先做 **measure-only 口径重测 ρ**（在线口径、
     权重 0、行为逐位不变）→ 补尾部 flush → 再谈 S1 的机制前置（须单独预注册）→ 阳性对照通过。
     全文：[`docs/engagement_trade_online_2026-09-18.md`](docs/engagement_trade_online_2026-09-18.md)
   - **★ 预注册 §7 的 7 条不变量全部有测试**：§7-1…§7-6 见离线套件（**8/8 PASS**），
     §7-7 `test_engagement_trade_default_off` 见 `scripts/selftest_engagement_trade_online.py`（**1/1 PASS**，
     引擎侧，2026-09-18 第三轮落地）。**顺带把"尾部窗口"缺口实测出来**：60 帧的局 `n_settled = 0`
     （分数只在局末 flush ⇒ 永远进不了任何一帧），拉到 200 帧后窗口在局内结算、digest 才变
     ⇒ **上游按 `max_ep_steps` 截断时，局内未结算窗口的 score 全部丢失**（可复现，非理论顾虑）。
   - **★ 在线奖励项（默认关）+ 引擎侧记录通道 + schema 5 已落真树**：
     `rl/engagement.py::EngagementTradeMonitor` + `RLEnv` 接线 + `DEFAULT_REWARD` 四开关
     （`engagement_trade` **默认 0.0**）；**开/关逐位对账 PASS**（未打补丁 vs 开关关：逐帧奖励 + 全实体状态
     SHA-256 逐字相同 `4d8b27b8…eca0c`）。
     **落地三步验证**：中立性 DIGEST 逐字一致 ✅ / 【R19】selftest 子集 **4/4** ✅ /
     短 smoke 产出 **schema 5**（元组全 15 长、`v0` 覆盖 100%、`is_product` 9.7%、非塔实体 `root_cast` 97.9%）✅。
     ⚠️ 一次 **CRLF 假阴性**事故（Windows python stdout 为 CRLF ⇒ `awk` 取出的 digest 带 `\r` ⇒
     逐字节比较失败 ⇒ 把**成功的**补丁回滚了）；修法 `| tr -d '\r'`，已写进脚本注释。
     记录：[`docs/engagement_trade_online_2026-09-18.md`](docs/engagement_trade_online_2026-09-18.md)
   - **（历史）sandbox 阶段的实现记录**：行为中立性逐位对账 PASS
     （1200 tick 全实体 SHA-256 两棵树逐字相同），schema 5（`root_cast`/`is_product`/`share` + 帧内 `v0/v1`）
     亦已在 sandbox 端到端验证（`exact=True`）。**已 armed 受控落地**：
     `scripts/_apply_s2_channel_when_idle.sh`（等 `run100k` 终局评估点 + 120 s → 干跑预检 → 应用 →
     **中立性 DIGEST 必须等于基线** `9adc2aaf…5b55` → 跑 【R19】selftest 子集 → 短 smoke 产出真 schema-5 录像；
     任一步失败**自动回滚**）。
     三个逐字 diff：[`root_cast_channel_2026-09-18.diff`](docs/root_cast_channel_2026-09-18.diff)、
     [`schema5_replay_2026-09-18.diff`](docs/schema5_replay_2026-09-18.diff)、
     [`schema5_run_league_2026-09-18.diff`](docs/schema5_run_league_2026-09-18.diff)；
     记录与落地步骤：[`docs/root_cast_channel_2026-09-18.md`](docs/root_cast_channel_2026-09-18.md)
3. **★ 第八轮：用户直接要求的 `economy_et` 100k 干预长跑（两臂，★ 已跑完并出判读）**
   - 臂：**`A_et`**（`--config economy_et`，`engagement_trade=0.5` 与 `edw` 同汇率）+ **阴性对照 `B_ctrl`**
     （`--config economy_etm`，见下 §11.13.7），逐字同参同 seed，**顺序跑**。模式 = **`solo`**（预注册 §8 强制）。
   - 判据在开跑前写死：预注册 [`§11.13`](docs/engagement_trade_prereg_2026-09-18.md)（机制/行为/门禁/项体检 + 4 条失败分支）。
   - ⚠️ **本轮不构成判决**：缺阳性对照（§5.2 无法判 `VALID`）+ n=1 seed/臂（【R5】）⇒
     **不得**写成「修好了」或「确认无效」，只能写「观察到什么 / 未达行为可见阈值 / 不可分辨」。
   - ★ **发射遗漏与更正（§11.13.6）**：首次发射**漏了 `--adv-inert-probe`** ⇒ 机制层主判据的
     `resid_norm`/`grad_cos` 两项**无法读**（日志 0 行）。该开关经代码核对为**纯测量**
     （替代优势只喂 `autograd.grad(retain_graph=True)`，`.backward()` 路径逐字不变；网络无 dropout
     ⇒ 不耗 RNG；回归测试 1/1 PASS）⇒ **两臂对称补上后重启**，判据一字未改。
     第一次发射在 **step 7547** 主动终止、进度作废（留证 `runs/_discarded_et_solo100k_noprobe/`
     与 `docs/train_et_solo100k_noprobe_discarded.log`）。
   - ★ **第二处发射问题（§11.13.7）**：`_set_et_measure` 只在 `measure_only` 或 `engagement_trade>0`
     时才让**评估局**记 `et` ⇒ `B_ctrl = --config economy` **两个都不满足**、评估录像**没有 `et`**
     ⇒ §11.13.2 的**门禁对照列是空的**。改用 `economy_etm`（**唯一**有效差异 = `measure_only 0→1`，
     【R4】脚本复算；`test_measure_only_is_behavior_neutral` 逐位中性，复跑 **3/3 PASS**）；
     `A_et` **不中断**，由监视脚本在 `B_ctrl` 刚起来时换掉（损失 < 1 分钟）。
     ⚠️ **该"换臂脚本"已被 §11.13.9 取代**：重启后 `B_ctrl` **直接以 `economy_etm` 起跑**，不再需要它。
   - ★ **第三处问题（§11.13.9）**：§11.13.2 **自身内部不一致** —— 它同时写死「仪器 = `diagnose_every 10`
     日志」与「对照 = 前 **100** 次诊断更新」，而 `advinert` 的**打印频率由 `diagnose_every` 决定**：
     实测 `de=10` 只有 **12 点 / 15895 步**（≈1 点/1325 步）⇒ 100k 仅 ~76 点 ⇒ **`n_later = 0`，主判据不可执行**。
     ⚠️ 我先前写的「145 点/20k 步 ⇒ 100k ~725 点」**是错的**：那个 145/20k 来自
     `train_critic_inert_probe_20k.log`，而它是 **`diagnose_every=1`**（头部自证）⇒ **密度随
     `diagnose_every` 反比变化，跨配置搬数字是错的**（【R10】）。**不改 N**（改成 ~40 就是
     【R16】禁止的**单次观测标定**），改为**让采集满足预先写死的 N**：两臂统一
     **`--diagnose-every 1`**（~781 点/100k ⇒「前 100 点」≈ 前 12.8% 步，与 §11.13.2 原意一致；
     测量侧、判据一字未改）。第 2 次发射在 **step 15895** 终止作废（留证
     `runs/_discarded_et_solo100k_de10/`）；**`B_ctrl` 因此直接以 `economy_etm` 起跑**，
     §11.13.7 那个「杀错臂再换」的监视脚本**已不再需要**（少一个会失败的环节）。
   - ★ **第四处问题（§11.13.10，仅命名）**：门禁行把**指标**写成「优势兑现率」、**仪器**写成
     `analyze_online_trade.py`；但**兑现率只在 `offline_engagement_trade.py` 实现**
     （`REALIZE_N_FRAMES=40` = 20 s 与之相符），前者 `grep realiz|兑现` = **0 命中**（它算配对 Δρ）。
     ⇒ 照字面只跑一个仪器会**整个漏掉预注册的门禁主指标**。更正 = **两者都算、并列标注口径**：
     层 3a 兑现率（**离线口径**）、层 3b 配对 Δρ（**在线口径**）；【R17】不得互相顶替。
   - ★★ **分辨率实测标定（§11.13.11，最重要的一条）**：第 2/3 次发射**同 seed 同配置**，仅
     `--diagnose-every` 10 vs 1 ⇒ `eval@0` **逐字相同（0.650）**，但 **`eval@8000` = 0.800 vs 0.200（差 0.60）**。
     代码层可排除优化器路径（`diag_on` 只 gate 额外梯度+日志，`ppo.py:253/303/306/508`），
     但**成因未定**（【R10】）：CUDA 反向原子累加不可逐位复现 + 自对弈混沌放大；n=1/配置 ⇒
     **不得**归因于 `diagnose_every`，更**不得**写成「de=1 把训练跑坏了」。
     标定（【R4】）：既有**同配置三连跑** `d1_league_20k{, _r2, _r3}` 末点胜率 **0.675/0.475/0.362**（极差 0.313；
     ⚠️ 那是 `run` 模式，本实验 `solo` ⇒ 仅量级参考）。⇒ **胜率/奖励层的 A vs B 差异在 n=1/臂 下一律不可归因**；
     失败分支 1/2 只能读**方向**；落在 §11.13.4-3 ⇒ 记「本分辨率下无结论」。
   - ★ **收尾已自动化 + 接手路径**：`scripts/et_solo100k_readout.py` 一条命令出 §11.13.2 四层读数，`--judgment` 直接按 §11.13.8 结构产出判读骨架（分支 3 机械判定、**1/2/4 标「须人工判读」**）；后台收尾任务等两臂结束→出产物→新建分支 `et-solo100k-readout` 推送。**若需接手**：run 记录 [`docs/et_solo100k_2026-09-18.md`](docs/et_solo100k_2026-09-18.md) **§10（自动收尾）/ §11（接手须知：两种情形 + 两臂训练命令原文 + 幂等重跑 + 写作禁则）/ §12（机制层中途快照，PRELIMINARY）**。
   - 工具（【R1】归因用）：`scripts/ps_list_python.ps1` 采样两次看**累计 CPU**即可区分「在算」与「阻塞」
     —— 本轮据此确认 `eval@8000` 后主进程约 5 min 空转属**评测 worker 收尾等待**，**未降档**。
   - 仪器：`scripts/judge_critic_inertia.py` 新增 `--baseline within`（本 run 前 100 诊断点中位 ± 3×MAD，
     **原始 MAD** 不乘 1.4826；【R4】脚本复算；`--selftest` 4/4 PASS）；
     一键判读器 `scripts/et_solo100k_readout.py`（层 1 机制 + **层 1b EVb/池化 EV** + 层 2 行为 +
     **层 3a 兑现率** + 层 3b 配对 Δρ + 层 4 项体检，含按批次配对 A−B；缺数据大声失败；
     满量程彩排 5 批次/250 局/臂 = **1 min 13 s**，含逐批兑现率；真实 14 批次按外推约 2–3 min）；
     判读文档必写条目见预注册 **§11.13.8**；`--judgment` 可**按该结构直接产出骨架**（静态条目写全 + 四层读数 + 失败分支输入表，分支 3 机械判定、1·2·4 标「须人工判读」）。
   - ★★ **收尾结果（14:44:59 两臂跑完 → 14:47:30 出产物，全部落盘）**：
     `A_et` 训练循环 **8075.6 s**、`B_ctrl` **7338.9 s**，各 **14/14** 评估点与回放、**0 降级**、
     `[gate] WARN` A **9** / B **6**（只报警不中断）。
     - **读数**：`docs/readout_et_solo100k.md`；**原始仪器输出（留证）** `docs/readout_et_solo100k/`
       （`forensics_*` / `gate_*` / `realization_*` / `realization_perbatch/*` / `item_health.json`）。
     - **判读**：`docs/et_solo100k_judgment_2026-09-18.md`（含**人工判读**的 §3.1/§3.2）。
     - **结论（三种允许写法）**：① **观察到**两臂都没把「攒费→打 Xbow」做出来（全帧圣水≥6
       **0.52% vs 0.05%**、Xbow **5/8769 vs 1/8370**、部署前圣水中位 **3.250 vs 3.286**）；
       ② **不可分辨**：机制层 **4/6**（`level_shift`/`level_gap`/`gnorm_ratio` + `EVb`），
       余 3 项越界但绝对量极小（`grad_cos` 0.99730 vs 0.99800 等）⇒ **不写"改善/恶化"**；
       ③ 门禁兑现率 **98.5% vs 98.2%**、逐批方向 **+6/−7/0** ⇒ **饱和、无分辨力**
       （但 A 的"打出塔伤"**低 4.1 pp**，是该病理的弱形式 ⇒ 记为候选下一步）。
       **失败分支 1/2/4 全部"未触发"**（1 前件不成立、2 A 不低于 B、4 无降级）；
       分支 3 **未触发**（并非全部指标都不可分辨）。**不构成判决**（缺阳性对照 + n=1/臂）。
     - ★ **收尾门禁换法（同类错误第二次）**：旧收尾脚本判「**已无 `run_league` 进程**」——
       而两臂是**顺序**跑，A 退出到 B 启动之间有**数秒空窗** ⇒ 会**在 B 还没跑时**就出产物
       （= 用缺席当完成证据）。已改为**阳性门禁** `scripts/finalize_et_solo100k.sh`：每臂都要有
       `[done] solo` + `eval@100000` + **≥14 回放**，任一臂不满足或读数 rc≠0 一律 **ABORT**。
       实测：12:37 armed 时正确报「A_et 缺完成行 / B_ctrl 日志不存在」，14:45:48 才两臂阳性命中。
     - ★ **第五/第六处更正（§11.13.12/§11.13.13，判读阶段查出，测量侧）**：① 在线仪器对
       **零方差批次**（`A_et` 的 `eval@32000` = **20W/0L** ⇒ `ρ(win)` 无定义）打 `n/a`，
       而读数脚本正则只认数字 ⇒ **静默丢一批**（层 3b 的 A 表 14→**13**、`status` 仍 `OK`）；
       ② 行为层第 3 项 `xbow_play_rate` 写成 `0 if never else None` 而"查表"从未实现 ⇒ **恒 `None`**。
       两者均已修 + 扩展 `--selftest`（**7/7 PASS**，含"旧正则会丢该行"的显式断言）。
4. **★ 训练健康指标（价值损失 / 策略熵）—— 用户直接要求（2026-09-18）**
   - 结论先行：**这两个量本来就在算、也在逐 update 打日志**（`ppo.py::_loss_pass` → `[solo step N]` /
     `[step N]`，100k ≈ **781 点**）；`solo_state.json` 的 `history[]` 是逐评估点（14 点）且 **32 键里没有**
     ⇒ **UI 一直看不见**。缺的不是计算，是「取出来 + 怎么判」。
   - 新增：`rl/train_health.py`（共用实现）、`scripts/health_curve.py`（CLI，`--selftest` **5/5 PASS**）、
     dashboard **`/api/health` + 「训练健康」指标组**（`--train-log`，缺省从 `--solo` 目录名推
     `<repo>/docs/train_<run>.log`；推不到就显示"不可用"，**不拿别的 run 顶上**）。
   - ★★ **口径红线**：① 熵 = **掩码后、一次决策内各 decoder 步熵之「和」**（nat）⇒ 与 bundle 长度有混杂，
     且**局面分布由策略决定**（自对弈）⇒ 不是纯锐度；② **价值损失要用 `vraw=`（原始 MSE）**，
     日志 `value=` 是 **÷`v_scale²`**（A_et 实测 `1.75 → 24.87`）⇒ 只看它会读到"损失在降"的**假象**
     （实测恒等式 `0.46/24.87² = 0.00074`）。
   - **A_et 实测（100k，779 点）**：熵 **0.596 → 0.459 → 0.322（前 ~20–30k 步完成全部下降）**，
     其后 70k 步在 0.26–0.36 震荡；局部平台读数 `diff=-0.0568`、`阈值 3×MAD=0.1988`、`snr=0.86` ⇒ **平台**；
     后半斜率 `-0.0077/10k`（< 1 MAD）。`vraw` 每段 MAD(1.6–3.2) 与段中位(1.9–3.8) **同量级**
     ⇒ **该量逐 update 无分辨力，不得当判据**。
   - ⚠️ **不得**写成「熵已平台 ⇒ 已到步数瓶颈」：`ent_coef=0.01` **固定**时熵本就趋于平衡点（正则混杂），
     且**胜率层在 n=1 下无分辨力**（§11.13.11 标定：同配置重跑单点差 **0.60**）⇒ 只能说
     「观察到熵已平台，**性能层在本分辨率下不可判**」。
   - ⚠️ 本读数**一律描述性、非预注册判据**，**不得**并入 §11.13 判据集（【R3】）；真要当判据须先
     按 §5 写死窗口/`k`、并用 **≥3 seed** 标定 `k×MAD ≫ run 间散布`（【R15】【R16】）。
   - 顺带两条工程账（均已落文档）：**GBK 陷阱新证据** —— `PYTHONIOENCODING=utf-8` 在本环境
     **实测不生效**（stdlib 自报 `stdout.encoding=gbk`），指纹 = 同一次输出里 `import judge_critic_inertia`
     （其模块顶层 reconfigure）**之前**的行是 GBK 字节、**之后**是 UTF-8 字节；修法一律
     `reconfigure(encoding="utf-8", errors="replace")`。**我自己犯的 bug**：`dashboard._PARENT` 是
     `src/clasher_new` 而非 `src` ⇒ 我写的 `dirname(_PARENT)` 让日志路径**一个也推不出**、面板永远"没有日志"
     —— 抓到它的是**新加的 health 正面路径回归**（只测"没数据不抛"是抓不到的）。
   - 全文（口径 / 读数 / 判据设计 / 三条工程坑）：[`docs/train_health_metrics_2026-09-18.md`](docs/train_health_metrics_2026-09-18.md)
5. **★ 五份对象报告（2026-09-19）：跨项目 RL 阶段奖惩设计的统一 schema 抽取** —— 对象 ①OpenAI Five + AlphaStar ②绝悟 / JueWu ③VPT + MineRL/BASALT ④RLHF + 离线 IL 族 ⑤我们 vs FirstLight_CR。**五份各自独立、不合并**；共用字段集 ①–⑧ + 统一单位口径；**禁止跨项目数字对拍**；`未验证` 一律原样保留、不得升格为事实。
   - 落盘：[`docs/rl_reward_crossproject_tables_2026-09-19.md`](docs/rl_reward_crossproject_tables_2026-09-19.md)（①）・[`docs/juewu_reward_table_unified_2026-09-19.md`](docs/juewu_reward_table_unified_2026-09-19.md)（②）・[`docs/vpt_minerl_rl_reward_tables_2026-09-19.md`](docs/vpt_minerl_rl_reward_tables_2026-09-19.md)（③）・[`docs/rlhf_offline_il_reward_audit_2026-09-19.md`](docs/rlhf_offline_il_reward_audit_2026-09-19.md)（④，**已补齐字段级正文至 460 行**）・[`docs/ours_vs_firstlight_reward_tables_2026-09-19.md`](docs/ours_vs_firstlight_reward_tables_2026-09-19.md)（⑤，**返回文本在 §1.D 截断**，见其 §G）。
