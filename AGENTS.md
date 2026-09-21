# AGENTS — 红线速查 + 决策索引（跨会话必读）

> **本文件 = 红线（每条一行）+ 索引。** 全文/证据/口径/台账在 [`docs/agents/`](docs/agents/)，**按需打开**。
>
> **⚠️ 节号保持不变**：分册里保留原 `## N.` / `### N.M` 编号 ⇒ 历史文档里的「`AGENTS.md` §2.1」
> 这类引用**经本文索引一跳可达**，不必逐份改写。
> **纪律不变**：① 新决策**先写 `docs/`** → 再在本文件加**一行指针**；② **只增不改历史结论**
> （被推翻时保留原条目并标注「已被 X 推翻」）；③ 条目编号可引用：`【红线 R7】` `【确证 C5】` `【否证 X2】` `【未决 O3】`。

> **（2026-09-19 新增）文件形态纪律**：① 三段在场必要性 —— **A 红线全量在场**（必须遵守，不可拆）／
> **B 索引一行式**（标题 + 一句话结论 + 指针，正文在一手文档）／**C 状态速览**（只留在做什么、状态、未决）；
> ② 总量闸门 **≤ 40 KB**、单行 **≤ 800 B**（避开注入截断与 `read` 工具单行 2000 字符截断）⇒ **可执行**：
> `python scripts/_agents_split.py --check` 的 `[6]`（超标 = FAIL）；③ **逐字下沉 + 压缩**：超长行与被压缩的行
> **逐字**搬进 [`docs/agents/agents_long_entries.md`](docs/agents/agents_long_entries.md)（新起一节，标「出处行号 + 逐字 +
> 压缩前字节数」），行内只留**结论 + 关键读数 + 指针**（`全文 → 分册 §N`）—— **不丢字**：事实**换位置**，不是被删掉。
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
| **R11** | **明确不做（未获用户另行拍板前）** | **不在 A′ 类取证之前改奖励**；不上训练时 MCTS；**扩参不再预设禁止**（★2026-09-21 用户拍板删掉硬卡：只考虑性价比）⇒ 改为**逐项预注册**：参数增量须绑定跑前写死的判据（ΔAUC / ΔNLL / 行为双侧带）并与收益**同批配对**，无收益回滚（全文 → 分册 §1） |
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
| 仪表盘（**端口 8700**） | env.md §2.5 |
| 自检命令（【R19】只跑相关子集） | env.md §2.6 |
| 训练口径：`step` 语义 / 评估口径 / 噪声地板 / **判读禁则** / 平局与早停 / run 成本 | [`docs/agents/metrics.md`](docs/agents/metrics.md) §3 |
| **关键常量表**（改代码前先对照） | metrics.md §3.1 |
| **已确证 C1–C16**（可作前提引用） | [`docs/agents/ledger.md`](docs/agents/ledger.md) §4 |
| **已否证 / 已关闭 X1–X19**（**勿重走**） | ledger.md §5 |
| 计划台账（哪条路走通 / 被否 / 未决） | [`docs/agents/plans_runs_docs.md`](docs/agents/plans_runs_docs.md) §7 |
| **Run 台账**（每个 run 一句话结论 + 证据路径） | plans_runs_docs.md §8 |
| **文档地图**（想做什么 → 读哪份 `docs/`） | plans_runs_docs.md §9 |
| 历史决策全文（旧 `AGENTS.md` 逐字冻结；含 2026-09-19 v2 上线前原文 [`docs/agents/_reorg_v1/AGENTS.pre-v2.md`](docs/agents/_reorg_v1/AGENTS.pre-v2.md)、整理报告 [`REPORT_v2.md`](docs/agents/_reorg_v1/REPORT_v2.md)） | [`docs/_archive/agents_archive_2026-09.md`](docs/_archive/agents_archive_2026-09.md) |
| **`long1m` 提前终止的账**（12.8% 的实际状态） | [`docs/_archive/long1m_stopped_2026-09-18.md`](docs/_archive/long1m_stopped_2026-09-18.md) |
| **新方案预注册**（判据 / 失败分支 / 不变量 / S1–S3；**实现前必读**） | [`docs/_archive/engagement_trade_prereg_2026-09-18.md`](docs/_archive/engagement_trade_prereg_2026-09-18.md) |
| **S2 实施进度**（schema 3→4 实体带 `id`+`target_id`；§11.5 = 仪器 + 门禁判读） | 同上预注册 §11 |
| **★ S2 实测与门禁判读（第二轮已重算）**（P3 推翻 / P1P2 否决 / **P4b 配对 10/10** / 6 个口径缺陷 / 门禁兑现率） | [`docs/_archive/s2_instrument_2026-09-18.md`](docs/_archive/s2_instrument_2026-09-18.md) |
| **★ S1 零成本门禁**（手写专家；判「约束在机制/探索侧」+ 冰人白嫖证实） | [`docs/_archive/s1_gate_2026-09-18.md`](docs/_archive/s1_gate_2026-09-18.md) |
| **引擎侧出牌溯源通道 + 受控落地**（**已落真树**；三步验证 4/4） | [`docs/_archive/root_cast_channel_2026-09-18.md`](docs/_archive/root_cast_channel_2026-09-18.md) |
| **★ 在线奖励项（默认关）+ 开/关对账 + 两处实质不确定 + CRLF 事故** | [`docs/_archive/engagement_trade_online_2026-09-18.md`](docs/_archive/engagement_trade_online_2026-09-18.md) |
| **★★ 在线口径实测：`p4b` 优势不成立（Δρ 从 +0.145 掉到 +0.008；250 局）** | [`docs/_archive/online_measure_2026-09-18.md`](docs/_archive/online_measure_2026-09-18.md) |
| 长跑工程（两级评估 + 评估并行分片） | [`docs/_archive/long1m_prereg_2026-09-18.md`](docs/_archive/long1m_prereg_2026-09-18.md) |
| 仪表盘功能（进度条 / 大点 / 逐对手胜率；端口见 env.md §2.5）；`long1m` 对接记录（**该 run 已终止**，功能仍在用） | [`docs/_archive/dashboard_long1m_2026-09-18.md`](docs/_archive/dashboard_long1m_2026-09-18.md)；压缩前逐字 → 分册 **§20** |
| **★★ `et_solo100k` 两臂干预长跑**（**已跑完 + 已判读**）：**结论 = 没有任何一项能用现有仪器分辨出疗效，不构成判决**；含**三次发射** + **六处测量侧更正** §11.13.6/7/9/10/12/13 + **★分辨率实测标定 §11.13.11**（要点与读数见 **C3** 行） | 判读 [`docs/et_solo100k_judgment_2026-09-18.md`](docs/et_solo100k_judgment_2026-09-18.md)、读数 [`docs/readout_et_solo100k.md`](docs/readout_et_solo100k.md)、run 记录 [`docs/_archive/et_solo100k_2026-09-18.md`](docs/_archive/et_solo100k_2026-09-18.md)、台账 **O8**；全文 → 分册 **§14** |
| **★ 训练健康指标**（价值损失 / 策略熵 = **本来就有、只是 UI 看不见**；GBK 陷阱新证据） | **已交付**（**描述性、非判据**；口径红线与读数见 **C4** 行）｜ [`docs/_archive/train_health_metrics_2026-09-18.md`](docs/_archive/train_health_metrics_2026-09-18.md)；全文 → 分册 **§15** |
| **★ 「不出牌」概率（P(STOP)）实测**（用户提问；**已交付**）：**开局帧** A_et **0.052** / B_ctrl 0.034 / 未训练 0.081；**有牌可出时** 0.058~0.09；**全程 ≈0.90 是「没得选」**（88%~91% 的帧买不起任何手牌，圣水中位 1.50，「不裸下」门禁 **0** 帧）⇔ 纠正 `P(STOP)=0.945/0.929` 的老读法 | 读数 [`docs/_archive/pass_prob_2026-09-18.md`](docs/_archive/pass_prob_2026-09-18.md)、仪器 `scripts/probe_pass_prob.py`；压缩前逐字 → 分册 **§20** |
| **★ 攒费穷举实测：「学不会攒费」成立且更狠**（**已闭合 ⇒ C14**）：「放弃一张买得起的牌」**75,891 帧 0 次**、7,238 段主动不出牌**0 段**在圣水≥4 仍不出、A 段最长 **22 < 34 帧** ⇒ **轨迹从未被采样**（0 样本 ⇒ 0 梯度）；★ 附带查出的 **B 类病理**已修（见下行） | 审计 [`docs/_archive/elixir_saving_audit_2026-09-18.md`](docs/_archive/elixir_saving_audit_2026-09-18.md)、仪器 `scripts/pass_streak_audit.py`、台账 **C14 / O9**、压缩前逐字 → 分册 **§20**；全文 → 分册 **§1** |
| **★★ 修复：掩码 `used` 0/1-based off-by-one（B 类根因；已修）** | `used` 存 **0-based** 却用 **1-based** `sa.slot` 测试 ⇒ **掩码放行已用槽位** ⇒ 重复槽位包 ⇒ `validate_bundle` **整包拒绝**（真实 100k：497 帧 / 0.7%）。**一行修复 + 两条零成本掩码不变式 + 回归测试**：同种子 **44/256→0/256**、拒绝帧 **474/1707→0/1609**、端到端 **B = 0/3413**；⚠️ 与攒费无关（C14 不变） | [`docs/_archive/mask_used_slot_offbyone_fix_2026-09-18.md`](docs/_archive/mask_used_slot_offbyone_fix_2026-09-18.md)、留证目录同名、台账 **O9**、压缩前逐字 → 分册 **§20**；全文 → 分册 **§2** |
| **★ 探索随机（第一问）：「随机×随机+位置随机+正弦衰减」做不到事**（**已否证**） | 攒到 Xbow 是**合取** ⇒ 成功率 `p^k`（解析 **4.5×10⁻¹⁰ /100k 帧**）；**决定成败的是「时长」不是「随机」**（`hold d~U{1..40}` ⇒ ≥6 帧 **28.96%/38.81%**，差 ~12 个数量级）；正弦/衰减是训练步调度 ⇒ 对局内合取**零作用**且裁梯度 | [`docs/exploration_randomization_analysis_2026-09-18.md`](docs/exploration_randomization_analysis_2026-09-18.md)、仪器 `scripts/probe_explore_randomization.py`、留证 `docs/probe_explore_randomization/`、台账 **O10**；全文 → 分册 **§3** |
| **★ 偏置随机（第二问）：抬「不出牌」概率是唯一正确杠杆** | 逐帧 `1/6→q` ⇒ 成功率 `q^k`（每帧 5 倍 = 12 个数量级）；**可零 off-policy 实现**（STOP logit 加正偏置 β，`ratio≡1`），剂量 β=0/4/5/6/7 ⇒ **0.000 → 0.178 → 12.70 → 51.27 → 82.98%**；「抬大费牌概率」方向是**反的**；★ **覆盖率 ≠ 会学**：每局回报 **−4.46 → −41.5/−54.1** ⇒ PPO 会学到「别这么干」 | [`docs/_archive/exploration_bias_pass_2026-09-18.md`](docs/_archive/exploration_bias_pass_2026-09-18.md)、仪器同前、留证 `docs/probe_explore_randomization/bias_*`、台账 **O10**；全文 → 分册 **§4** |
| **★ 状态条件随机（第三问）：低压时才随机 —— 方向对，买回了回报** | 门 `d`（最近 15 帧没掉塔血）开 **60.9%** 帧；`sample@6`（未门控）**48.209% / −56.13** → **`gate@6:d` 6.765% / +1.68**（三轮里**第一个同时成立**）；代价 **÷7**；码内 `PRESSURE_THRESHOLD=2.0` 只开 **4.7%** ⇒ 等于没有；n=8 且基线回报跨 run 摆动 ⇒ **回报差异不可当判据**；**「省下的费没花出去」未解决**（133 帧 / 0 次 Xbow）⇒ **仍要 O7** | [`docs/_archive/exploration_pressure_gate_2026-09-18.md`](docs/_archive/exploration_pressure_gate_2026-09-18.md)、仪器同前、留证 `docs/probe_explore_randomization/{pressure,gated_*,pressure_gate}.*`、台账 **O10**；全文 → 分册 **§5** |
| **★ 「初始模型 + 全过程随机」100 局评估 + dashboard 观测** | p0 = 全过程均匀随机 actor、p1 = `solo_main_0.pt` 确定性 argmax；**100 局 / 31,491 帧**（schema 5）⇒ **58.0%±4.9**、局均 **314.9 帧**、空 bundle **90.4%**；独立复算 **A 90.4% / B 0.0% / C 9.6%**；**圣水峰值 5.357 = 开局 5.0 + 2 帧回费 ⇒ 100 局里从未超过开局水平**；**Xbow 0 次**（3,062 部署）。⚠️ p0 **不能用于训练/BC**；对手同源 ⇒ 胜率不含绝对强度 | [`docs/_archive/rand100_eval_2026-09-18.md`](docs/_archive/rand100_eval_2026-09-18.md)、runner `scripts/random_eval_100.py`、复算 `docs/rand100_eval_2026-09-18/`、台账 **O10**；全文 → 分册 **§6** |
| **★ 「非法动作」四层取证（只读取证，未改代码）** | 四层**并列且互不推导**：**L1 掩码**（只产 `bool`）→ **L2 整包校验**（**9 条 reason**，任一非法**整包拒收**）→ **L3 惩罚/统计**（`invalid_penalty=0.05` × `invalid_count`；两来源）→ **L4 引擎物理**（失败一律**静默 `return False`**）；**10 条冲突 C1–C10 并列不调和**；**F1** MergeMaiden「费用过严 + 位置过松」；**F2** BarbLog 在 `FOUR_DECK_SET` ⇒ **solo 实时暴露**、`run` 模式 p1 侧非法**不被计数**；真实日志 P1-20 = **1,644 行全 BarbLog**；未定 U8–U12 ｜ [`docs/_archive/illegal_action_layers_2026-09-18.md`](docs/_archive/illegal_action_layers_2026-09-18.md)、台账 **O9**；全文 → 分册 **§7** |
| **★ 攒费意图（intent-save）：已跑完 + 判读（★2026-09-20 更正）** | 100k 两臂 A/B ⇒ **J1 FAIL（0 条，两 seed）**。★**判读仪器跑错卡组**（不传 deck ⇒ 原版 8 卡**无 ≥6 费牌**）：**空判据 + OOD** ⇒ 旧读数**作废**；修好后 **`fired=0`** ⇒ 预注册 **F1 触发**（「缺口」论撤回）。★ 归因：`held` 帧 **61.1% 在 Xbow(6 费)**、Skeletons **0.2%** ⇒ **不是「为最便宜的卡等待」**；病灶 = **目标抖动**（74.8% 换目标、段均 1.84 帧、**攒够 457/打出 0**）。⚠️ 同锚点 A **0.500** vs B **0.100**（5σ）。下一刀 = **承诺锁定 / `ready` 松手** → **M2 时长** ｜ [勘误](docs/intent_target_attribution_2026-09-20.md)；**O11**；全文 → 分册 **§19** |
| **★★ 全项目结构优化（Tier 0/1/2/3）**（**已收尾**：21 项 = 完全完成 **15** / 部分 **3** / 不做 **3**） | 盘点 **199 `.py` / 62,402 行**、引擎→`rl/` 反向边 = 0；死件 31、UTF-8 单一实现、绝对路径 9→0；`dashboard` **3,171→566**、`card_mechanics` **1,896→861**、`selftest` **6,023→172**（全量 **100 通过 / 0 失败**）；Tier 3 = 取消 cwd 契约 + 核对器 **⑪** | [`docs/agents/structure.md`](docs/agents/structure.md)、[`docs/_archive/structure_remaining_2026-09-19.md`](docs/_archive/structure_remaining_2026-09-19.md)、`docs/agents/scripts_inventory.md`；全文 → 分册 **§8** |
| **★ 外部调研类台账（IL 奖惩 / 人类回放 / 录像格式）**（**已交付**） | 全文 = [`docs/agents/reward_surveys.md`](docs/agents/reward_surveys.md)：IL 阶段**结构性没有奖惩**、**IL→PPO 锚定项默认关闭**（一手 KL 先例 = VPT）；人类回放：卡键/卡组/费用**可映射**、坐标**无需翻转**，但直接重放只走通 **35–62%**（主缺口 = **圣水不可观测**）⇒ **必须先状态重建**；★ 卡覆盖 **177/178**（唯一缺 `void`） | 分册 [`docs/agents/reward_surveys.md`](docs/agents/reward_surveys.md)（+ 三份源分册）；全文 → 分册 **§9** |
| **★★ 用 FL 的 IL_Replay 做模仿学习（IL）** | **★09-21 二期 Stage 0：「对手信息不足」被否证**——喂入对手**真值**手牌/循环/圣水/双方距下点帧数的 ΔAUC **+0.0006/−0.010** ⇒ 判 **H-A（优化/混比）**；act 上界 **AUC≈0.68**。对手模型首测：下张进手牌 top1 **0.723**、下张打出牌 **0.249**（均匀 0.125）、Brier 0.072。人类 8/9/10 费出牌率 **0.107/0.186/0.094**，塌掉臂 **≥9 费占 96%、10 局出 3 次** ⇒ **自强化陷阱**。★ C17（只取 6.47% 决策帧 ⇒ 塌成永不出牌）见 §23。**下一刀＝混比剂量-反应（已开工）**｜ [判读21](docs/fl_il_2026-09-21.md)、[二期](docs/fl_il_il2_stage0_2026-09-22.md)；**C15–C18/O12/O13**；全文 → **§22–§24** |
| **★ 骷髅军团两引擎观测四问复核** | **O3 仅我方**（我方桥面 A\* 代价 **5** vs 上游 **8**、河格硬阻挡 vs 可走）；**O4 两引擎同值**（王塔后 **6/9**、桥头 **15/0**，期望少侧 5/6）——根因 = **没有阵型**：15 只塞进半径 **0.55** 环、**75 对重叠**，形状是碰撞 **0.1 s** 内炸出来的；**消融**（预注册）`_lane_offset`→0 ⇒ 过桥后横向跨度 **1.3→0.2**（唯一已证因果），越出桥廊 **2→7**、O3/O4 不变 ｜ [`docs/_archive/skarmy_two_engine_observations_2026-09-19.md`](docs/_archive/skarmy_two_engine_observations_2026-09-19.md) |
| **★ O3 修复：横向行程改回上游口径** | 成因 = 代价表**抹掉价差**：车道走廊 `1`/`2`=5 而 `.`=5 ⇒「提前横移」与「到河再横移」**代价严格相等**（都 `3×70+26×50`）⇒ A\* 只出 tie-break 产物（实测我方沿 x=6.00 直走到 y≈13.4）。单变量 `.` 5→**8** ⇒ y=10 横向完成度 **0.35→0.804**（上游 0.833）、y12−y4 的 \|Δ桥心\| **+0.04→−1.56**（判据跑前写死，**H1 成立**）；**分道 6/9、15/0 与桥廊 max 3/7/7 逐值不变**，越出桥廊 5 帧→**0**；代价面 = 过桥慢 0.3 s ｜ [`docs/_archive/o3_lateral_convergence_fix_2026-09-19.md`](docs/_archive/o3_lateral_convergence_fix_2026-09-19.md) |
| **★ 阵型机制（第一批）+ 碰撞减半卡位预注册** | 出生几何原**只有均匀圆环**、gamedata **无阵型字段** ⇒ 新增 `formation.py` 形状表（**表外逐值不变**，12 张未登记卡复算逐值相同）。tick0 重叠对 `Goblins(4)` **2→0**、`RoyalHogs(4)` **4→0**。★ 卡位三轮：**两种口径都能卡住**（无石人则径直通过）⇒ 「减半是卡位的前提」**未证明**（卡位实为两段） ｜ [`formation`](docs/_archive/formation_2026-09-19.md)、[`collision`](docs/_archive/collision_half_block_test_2026-09-19.md)；**行全文 → 分册 §17** |
| **★ 卡位第 4 轮：质量模型剂量-反应** | 现状碰撞**已等价于「质量 ∝ 1/移速」** ⇒ 缺的是**体积项**；3 s 窗口全落在石人 `deploy_time` 冻结期 ⇒ 位移 = **纯推挤**：push石人 **0.107(k=0) → 0.062(k=2，达标) → 0.015(k=6)**，位移转嫁给女巫（0.143→0.235）而**卡位在所有臂都成立**；★ 上游 `*0.5` **反而最差（0.150）** | [`docs/_archive/collision_half_block_test_2026-09-19.md`](docs/_archive/collision_half_block_test_2026-09-19.md)；压缩前逐字 → 分册 **§20** |
| **★ 卡位第 5 轮：长时程（20s，含冻结期之后）** | 四个 k 下她**从未绕到石人前面**（`order_flips=0`、`witch_front_ticks=0`）⇒「她一直在它之后」**成立且与 k 无关**；但方向与「她推着它走」**相反** —— **石人把她往回推 6.3(k=0)→9.0(k=6) 格**、石人自己走 +5.1→+8.1 格 ⇒「纹丝不动」与「别把她一路顶回桥那边」是**同一旋钮两端** | [`docs/_archive/collision_half_block_test_2026-09-19.md`](docs/_archive/collision_half_block_test_2026-09-19.md)；压缩前逐字 → 分册 **§20** |
| **★ 卡位 k 值：用户拍板不动手（2026-09-19）** | **决定：不采用质量项**（k 不动手）；上游 `*0.5` **亦无采用依据**——第 4 轮实测它在「石头人站得住」上**最差**（push石人 +40%），第 5 轮又证两口径都能卡住。⇒ 五轮净结果 = **本仓碰撞口径保持现状**，且五轮**全是猴补丁副本、`battle.py` 一行未改**（无需回滚）。真正决定「像不像真实游戏」的是**几何**（圆↔正方形凹角），**尚未验证** ｜ [`docs/_archive/collision_half_block_test_2026-09-19.md`](docs/_archive/collision_half_block_test_2026-09-19.md) |
| **★ 上游引擎增量核查（`f616f19` vs 我们）** | 分叉点 **`f20fa4d`**（我方 **242** 提交 / 上游 **24**）；★ **退出码陷阱**：`--is-ancestor f616f19` = **rc=128**（对象不在本机）≠「不是祖先」（rc=1）⇒ 不得据此断言无共同祖先；上游 24 提交里 **6 个动引擎**：① 建筑掉血**已独立修好** ② 碰撞 `*0.5` **未跟** ③ 20 Hz **等价** ④ 河面 A\* **不照搬**。★ **顺带查出我们自己的四处**：**P1** `battle.py:3292` 应为 `continue`；**P2** 桥面 16 半格 `W`（3 格实 2 格）；**P3（高）** 过河 jump **无界闩锁**；**P4** `walkable_cache` 全局污染 ｜ [`docs/upstream_engine_delta_2026-09-19.md`](docs/upstream_engine_delta_2026-09-19.md)；全文 → 分册 **§10** |
| **★ 可借鉴什么：上游新版训练方法** | 上游 `7fc64f2` 把**落点熵**从熵奖励里整块删掉（只留卡熵 + `eligible` 门控）；「攒费/解场」做在**对手课程**（`elixir<8` 不出、聚团法术）。★**实测我们熵奖励 85.5%–90.0% 付给落点头**（576 类）⇒ **B1 删落点熵**（唯一动 learner 损失处）、**B2 把已有 `defend`(0.2) 变强**（前文 §3.5③「补一个对手」**前提错**）。**不借**：STOP 偏置改中性（≡β=1，实测落死区 β=0/4≈0%）、8 帧堆叠（R11+R6）、576 落点头（**本来就是**）。上游自 **24.69M 步**热启动 ⇒ 与我 128k 差 **193×** | [`docs/upstream_borrow_2026-09-19.md`](docs/upstream_borrow_2026-09-19.md)、仪器 `scripts/probe_entropy_decomp.py` |
| **★ dashboard 塔尺寸 = 引擎几何（2026-09-19）** | 前端 JS 写死 `half=0.5 格`（公主塔 1 格，**自项目上传起就是**），引擎自 2026-09-09 起是矩形（公主塔 3×3 / 王塔 4×4）⇒ **同一份几何前后端各写一份**。修法 = 引擎写进录像**局级** `meta["tower_geom"]`（**不升 schema**；逐帧记 **+24%~28%** 体积 ⇒ 只记一次），前端优先它、老录像回落同值常量。闸门 `scripts/check_dashboard_js.py --tower-only`（对账 7 + **真绘制实宽** 7 PASS；负对照 4 FAIL）★ 改前端**必须重启 dashboard** ｜ [`docs/_archive/dashboard_tower_geom_2026-09-19.md`](docs/_archive/dashboard_tower_geom_2026-09-19.md) |
| **★ 模型向量清单（有多少向量 / 来源 / 维度）** | 单帧前向 **5 组**：grid **576×15**（→CNN **26ch**）、hand **5×8**、scalar **3**、plan **58**、belief **563** ⇒ `fused` **2731** ⇒ `enc` **128**；输出 `slot_head` **6** ×≤5 步 + `cell_head` **576** ×≤4 步；唯一 `nn.Embedding` = `entity_emb` **177×8**。参数侧 **37 张量 / 993 008 标量 ⇔ 2 265 个 2-D 行向量**（口径 a/b/c 见文档）。★ **本仓无 transformer**（0 层、无 Q/K/V）；`plan_space` 注释 **57** 为陈旧值（实 **58**） ｜ [`docs/model_vector_inventory_2026-09-20.md`](docs/model_vector_inventory_2026-09-20.md) |
| **★ 观测够不够（表示瓶颈判定）** | `grid 576×15` 是**有损投影**：同格**赋值覆盖**吞 **1.3316%** 写入实体（**13.4438%** 帧至少丢 1）、词表外**静默丢弃 3.9430%**、坐标 **`int()` 量化整格**、塔 3×3/4×4 塌 1 格、`Entity.__init__` **58 字段只读 6**（**52 不观测**）、15 通道 **22/26 卡名纯函数**。★ **无损重建=不能 / 人类水平=无法判 / 是否主因=未定**（已确证证据全指向机制·探索·优化侧，**无一条**指向观测缺字段）**/ 只该改 G15+G12 零参数修复**；扩通道被 **R11** 挡、「换通道」**不规避 R6** | [`docs/observation_gap_verdict_2026-09-20.md`](docs/observation_gap_verdict_2026-09-20.md)；**全文 → 分册 §21** |
| **★ 模型如何决策（决策路径）** | 每帧 `fused 2731→enc 128` → 帧级 `GRUCell` 一次得 `h` → **循环 ≤5 步**（`range(K_MAX+2)` 只是保守上界，`K_MAX=4`）：用 `h` 在 **6 类**（4 槽+ABILITY+STOP）选 1 → 出牌则用**同一 `h`** 在 **576 格**选落点 → `h` 更新进下一轮；掩码**每轮重取**，`plan` 只是软偏置。**实测**：多步帧 `lp_roll` vs `lp_batch` 差 **1.2e-07**（double vs float32）⇒ PPO `ratio` 不精确为 1。★ 6 条限制：`K_MAX=4`、576 类吃 **85.5–90%** 熵奖励、plan 不改类数、belief **非学习**（`neural=` 全仓零调用）、STOP 与卡牌共用一头、掩码常压到只剩 STOP ｜ [`docs/model_decision_path_2026-09-20.md`](docs/model_decision_path_2026-09-20.md) |

---

## C. 当前活跃工作

### C-速览（状态 + 必须在场的要点 + 全文锚点）

| # | 项目 | 状态 | 必须在场的要点 | 全文 / 指针 |
|---|---|---|---|---|
| **C0** | `run100k` | ✅ **已跑完**（`run` 模式 100k 步） | 14 点两级评估 / 800 局（`--only-vs-main`）；**`runs/run100k/`：14 个 schema-4 录像**（`league_{0,8000,…,100000}.pkl`）**+ 14 ckpt**，**它就是 S2 全部读数的数据源**（schema 4 ⇒ 带 §11.9.3 重建误差）；新源 `runs/run_schema5/`（schema 5、60 局） | 记录 [`docs/_archive/run100k_2026-09-18.md`](docs/_archive/run100k_2026-09-18.md)（⚠️ 仍是启动记录，终局未回写）；全文 → 分册 **§11** |
| **C1** | `long1m` | ⛔ **已按用户指令提前终止**（128,000 / 1,000,000 = **12.8%**） | 18 个评估点、**0 次降级**、**无判读产出且 J1 按定义不成立**；`runs/long1m/`（393 MB、18 ckpt + 18 录像）**保留** = 新方案现成回放素材 | [`docs/_archive/long1m_stopped_2026-09-18.md`](docs/_archive/long1m_stopped_2026-09-18.md)；全文 → 分册 **§12** |
| **C2** | 新方案（圣水交换信用分配） | ⏸ **已完整取证，判「不接线」**（**不启动 S3**） | 三门禁：**S1 未通过**（约束在探索/机制侧）、**S2 在正确口径下不通过**（Δρ **+0.008/+0.014**，离线 +0.145/+0.108 ⇒ **÷18 / ÷7.7**）、`τ` 有信号但增量低于一切仪器分辨率；**放行条件写死在预注册 §11.9.4**；出路 = **O7 修机制**（须单独预注册） | 预注册 §11.5/§11.9/§11.10、[`docs/_archive/s2_instrument_2026-09-18.md`](docs/_archive/s2_instrument_2026-09-18.md)、[`docs/_archive/s1_gate_2026-09-18.md`](docs/_archive/s1_gate_2026-09-18.md)、[`docs/_archive/online_measure_2026-09-18.md`](docs/_archive/online_measure_2026-09-18.md)、台账 **X19 / O7**；全文 → 分册 **§13** |
| **C3** | `et_solo100k` 两臂 | ✅ **已跑完 + 已判读**（⚠️ **不构成判决**） | 缺阳性对照 + n=1/臂 ⇒ **不得**写成「修好了 / 确认无效」；三种允许写法：**观察到**（两臂都没做出「攒费→打 Xbow」：全帧圣水≥6 **0.52% vs 0.05%**、Xbow **5/8769 vs 1/8370**）、**不可分辨**（机制层 4/6）、**门禁饱和无分辨力**（兑现率 **98.5% vs 98.2%**）；★ 分辨率标定：同 seed 同配置仅 `--diagnose-every` 10 vs 1 ⇒ `eval@8000` **0.800 vs 0.200（差 0.60）** | 判读 [`docs/et_solo100k_judgment_2026-09-18.md`](docs/et_solo100k_judgment_2026-09-18.md)、读数 [`docs/readout_et_solo100k.md`](docs/readout_et_solo100k.md)、run 记录 §10/§11/§12、台账 **O8**；全文 → 分册 **§14** |
| **C4** | 训练健康指标 | ✅ **已交付**（**描述性、非判据**） | 价值损失 / 策略熵**本来就逐 update 在算、只是 UI 看不见**；口径红线：熵 = 掩码后各 decoder 步熵之**和**（nat）、价值损失必须用 **`vraw=`**（`value=` 是 ÷`v_scale²`：A_et 实测 1.75→24.87）；A_et 熵 0.596→0.459→0.322 后 70k 在 0.26–0.36（**平台**）；⚠️ **不得**写成「熵平台 ⇒ 步数瓶颈」 | [`docs/_archive/train_health_metrics_2026-09-18.md`](docs/_archive/train_health_metrics_2026-09-18.md)；全文 → 分册 **§15** |
| **C5** | 五份对象报告 | ✅ **已落盘**（**各自独立、不合并**） | 对象 ①Five+AlphaStar ②绝悟/JueWu ③VPT+MineRL ④RLHF+离线 IL ⑤我们 vs FirstLight_CR；共用字段集 ①–⑧ + 统一单位口径；**禁止跨项目数字对拍**、`未验证` **不得升格**；⚠️ ⑤ 返回文本在 **§1.D 截断**（见其 §G） | 五份：`docs/_archive/rl_reward_crossproject_tables_2026-09-19.md`、`docs/_archive/juewu_reward_table_unified_2026-09-19.md`、`docs/_archive/vpt_minerl_rl_reward_tables_2026-09-19.md`、`docs/_archive/rlhf_offline_il_reward_audit_2026-09-19.md`、`docs/_archive/ours_vs_firstlight_reward_tables_2026-09-19.md`；全文 → 分册 **§16** |

### C-未决 / 待拍板（在场的一行指针）

- **上游核查顺带查出的 P1–P4**（`battle.py:3292` / 桥面 16 半格 / 过河 jump 无界闩锁 / `walkable_cache` 全局污染）——**一行未改，待拍板** ⇒ 见 B 区该行与 [`docs/upstream_engine_delta_2026-09-19.md`](docs/upstream_engine_delta_2026-09-19.md)。
