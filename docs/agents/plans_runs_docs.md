# AGENTS 分册 · 计划 · Run · 文档地图

> **来源**：原 `AGENTS.md` §7 / §8 / §9（2026-09-18 拆分，**逐字未改**）。
> **节号保持不变** ⇒ 历史文档里的「`AGENTS.md` §N.M」引用经索引一跳可达。
> 返回索引 → [`../../AGENTS.md`](../../AGENTS.md)

> 本册内容：计划台账 / run 台账 / 文档地图。

---

## 7. 计划台账（所有计划的定位与状态）

> 完整整合视图（含每项计划的改动清单、v1→v2 修订点、红线全文）见 **[`docs/plan_master.md`](docs/plan_master.md)**。

| 计划文档 | 定位 | 状态 |
|---|---|---|
| `docs/rl_review_fix_plan.md` | 评审版修复手册：6 Critical + ~20 Major + ~40 Minor，逐条配回归测试 | **已落地**（6/6 Critical 有回归测试；§6 的 11 个测试名 2026-09-13 核查全在 `rl/selftest.py`） |
| `docs/training_audit_2026-09-11.md` | 训练体系审计（四问：训练问题/指标/步数预算/参数量） | 已定稿，是 v1 的依据 |
| `docs/rl_training_fix_plan_v1.md` | 整改计划 v1（P0 通道+指标 → P1 结构 → 上量；含 G1/G2/G3 门禁、单变量协议） | 部分被 v2 取代（判据/门禁/预算口径三处已改） |
| `docs/rl_training_fix_plan_v2.md` | v1 修订版：G1 判据 `ratio`→**EV**、门禁改**相对首点**、预算改**局数**、对手池 `0.7/0.3/0.2`→`0.5/0.3/0.2`、**先评估后同步** | §1-3、§5、§6 已落地；**§4 病因判断被 v3 推翻**（见【否证 X2/X3】上下文） |
| `docs/rl_training_fix_plan_v3.md` | **当前生效**：GRU 饱和病因链 + P0-A/B/C + 各验证跑判读（5k/20k/grid_ln/A′/E1/E2）+ 门槛与分支 | P0-A/B 已落地；P0-B-2 第 3 项曾因**分母不存在**不可判读（已补）；P0-C 仅静态护栏；A′/E1 已落地，E2 被证伪 |
| `docs/value_channel_saturation_diagnosis_2026-09-11.md` | 负 EV 真根因诊断（GRU 输入饱和） | 已确证（C1） |
| `docs/critic_probe_experiment_2026-09-12.md` | critic 可预测性探针（含时间泄漏发现） | 已确证（X2） |
| `docs/rl_reward_plan_v2.md` | 奖励与计划通道重构 v2（逐帧账本、两段相位、闸门） | 前置依赖**已落地**；Phase 2 由 plan v1 承接；僵局早停保留未动 |
| `docs/rl_plan_design_v1.md` | PlanToken 战术意图 v1（结构先行 + bp/pp 组意图 + 消融） | **已落地**（`PLAN_DIM=58`，文头"57 维"已过期）；`bait` 留 backlog（无可靠触发） |
| `docs/ai_training_plan.md` | RL 算法闭环总纲（先知/信念/跟随者/POMDP/联赛/同刻多卡） | 代码按要求已落地；§8 阶段验收与 §11.3 N=200/500 协议**无落地记录** |
| `docs/P0-mechanics-plan.md` | 游戏机制补全（族 1-8） | 引擎机制**全部完成**（test_m2 72/72、batch_smoke 156/156）；瓶颈在数据层 |
| `docs/mcts_design.md` | 推理时浅 MCTS（零训练风险，量"搜索赚多少 Elo"） | `rl/mcts.py` + 2 个 selftest 已落地；**评估接入（`evaluate.py --mcts`）未落地**（全仓 `RLMCTS` 仅 selftest 调用） |
| `docs/four_decks_manual.md` | 四卡组逐卡手册（速猪/石头人/X弩/巨骷髅攻城槌） | 已定稿，供 `--deck-set four` |
| `docs/cycling_league_plan_2026-09-13.md` / `docs/d1_long_100k_prereg_2026-09-13.md` | D1 与 100k 的**预注册** | 均已执行并判读（C5/C6） |
| `docs/eval_cadence_c_2026-09-13.md` | 评估节奏 C 方案（密锚点 + 稀全块） | 已落地（`--anchor-every`，默认关） |
| **[`docs/lessons_ledger.md`](docs/lessons_ledger.md)** | **经验台账（冻结版）**：18 条被推翻的结论 + 12 类读数病理 + 12 道检测闸门 + R14 参数位移指纹（论文 Table 1 + 附录底稿）；**不引入新数字**，数字全部指向证据文件 | **已冻结（2026-09-13）**；新增经验只追加、被推翻的保留标注 |
| **[`docs/pomdp_ceiling_v2_verdict_2026-09-13.md`](docs/pomdp_ceiling_v2_verdict_2026-09-13.md)** | **信息侧天花板（第二轮，已判决）**：P3/H-LEARN —— 信息在现有输入里（A/C/B = +0.182/+0.186/+0.197，Ridge +0.30），oracle 只加 1.1pp、**belief token 只加 0.4pp**；四道对照电池全过 | **已判决**；预注册 `docs/pomdp_ceiling2_prereg_2026-09-13.md` |
| **[`docs/paper_outline_2026-09-13.md`](docs/paper_outline_2026-09-13.md)** | **论文大纲（案例 + 方法论合并）**：K1–K6 贡献、章节骨架、证据映射、实验 E1–E6、**停止规则（预注册）**、相关工作待检索清单 | 大纲已定；**缺口 = E4（对抗性基线）/ E5（外部有效性，最高优先）/ E6（用指纹重判四代架构）**；**不要求"强度提升"才动笔** |

---

## 8. Run 台账（结论 + 证据路径）

| run / 实验 | 结论一句话 | 证据 |
|---|---|---|
| `ab_valnorm_20k` | `step`=决策帧、`ratio` 判据作废、critic EV≈0 首次发现、门禁阈值口径错位 | `docs/ab_valnorm_20k_verdict_2026-09-11.md` |
| `prod_200k_valnorm_ev` | 跑到 ~54k 停：EV 全负；**成为饱和态基线**（不可用 `main_init` 续训） | `docs/value_channel_saturation_diagnosis_2026-09-11.md` |
| `fix_gru_ln_5k` / `_20k` | 饱和修复生效（n_abs 0.994→0.46~0.53）；critic 仍常数；查出**融合层尺度失衡**（grid 占 fused 98%） | `docs/train_fix_gru_ln_*.log` |
| `fix_gru_ln_norm_20k`（`grid_ln`） | 尺度修复达成（101→5.87）只修"对中"不修拟合；`n_abs` 回升 0.745（耐久性存疑）；`vs baseline0` 崩塌 = cycling 红旗 | `docs/train_fix_gru_ln_norm_20k.log` |
| `byp_cprime_20k` / `eind_20k` / `fprime_20k` / `fprime_ev_20k` | B′/E′/F′/G′-fix 四代架构干预：EV 全部 ≤0；F′ 真预算保留但未过零 | `docs/train_{byp_cprime,eind,fprime,fprime_ev}_20k.log` |
| `e2_rand_anchor_20k` | 训练侧弱锚点 10% **未破 cycling** | `docs/train_e2_rand_anchor_20k.log` |
| `d1_league_20k{,_r2,_r3}` | **防崩有效**（最差锚点区间不相交）；上限未证明；两处预注册标定错误自披露 | `docs/d1_league_20k_verdict_2026-09-13.md` |
| `d1_long_100k` | **P1 PASS**（防崩在 4× 量级成立）、与 20k 无统计差别、上限仍未解决、加量救不了 critic | `docs/d1_long_100k_verdict_2026-09-13.md` + `docs/diag_d1_long_100k.md` |
| `nostall20k` | **停用早停的端到端验证**：J1–J3 全过；早停**未**致"不爱下牌"；墙钟 47min | `docs/nostall20k_verdict_2026-09-17.md` |
| `long1m`（run 模式，1M 步） | **两级评估 + 评估并行分片的落地载体**：小点 8000 步×10 局/对、大点 100k 步×20 局/对 ⇒ 131 点 / 7100 局；判据 J1（跑完/无静默降级）、J3（对固定脚本池的**非自引用** Elo 曲线，门槛 71 Elo）**已在开跑前写死** | 预注册 `docs/long1m_prereg_2026-09-18.md`（判读 `docs/long1m_verdict_2026-09-18.md` 待出） |
| `critic_inert_probe_20k`（惰性检验 Layer 1） | **P1 成立 ⇒ critic 惰性**：`grad_cos` 中位 **0.9973**、`resid_norm` 中位 **0.0415**（145 点）；**该 run 的 critic 没塌**（EV +0.30、`vstd/rstd` 0.53）⇒ 惰性不是塌缩的副产品；分布尾巴 34% 点 <0.99、3% 反向。顺带查出 2 处文档数字错误（20k `vstd/rstd` 基线、塌缩双稳态）**。⚠️ 已被判读 §0.1 ④ 推翻**：该 run 末点 critic **就是常数**（`value` 唯一值 **1/600** = `value_head_mlp[2].bias`、`MLP0` ReLU 逐帧存活率 **0**）⇒ "没塌"不成立；且 **V≡常数时 `resid≈0`/`grad_cos≈1` 是恒等式**，"P1 成立"**不得**读成"拟合良好的 critic 也只有 4%"，据此关闭 critic 线的处置作废 | `docs/critic_inertia_verdict_2026-09-13.md`、预注册 `docs/critic_inertia_prereg_2026-09-13.md`、判读脚本 `scripts/judge_critic_inertia.py`、日志 `docs/train_critic_inert_probe_20k.log` |

---

## 9. 文档地图

| 想做什么 | 去哪 |
|---|---|
| 项目介绍 / 快速开始 | [`README.md`](README.md)（中文）、[`readme_en.md`](readme_en.md) |
| **所有计划的整合视图**（主线 / 已确证 / 已否证 / 未决 / 优先级） | [`docs/plan_master.md`](docs/plan_master.md) |
| **当前问题的归因**（critic 为何塌成常数 / 上限为何不动 / 判别性实验） | [`docs/d1_long_100k_cause_analysis_2026-09-13.md`](docs/d1_long_100k_cause_analysis_2026-09-13.md) |
| 该归因的结构化取证全文（6 路并行 + 评审，含 3 处已更正的口径错误） | [`docs/report_d1_long_100k_structured_2026-09-13.md`](docs/report_d1_long_100k_structured_2026-09-13.md) |
| **critic 惰性检验**（预注册 / 判读 / 判读脚本） | [`docs/critic_inertia_prereg_2026-09-13.md`](docs/critic_inertia_prereg_2026-09-13.md)、[`docs/critic_inertia_verdict_2026-09-13.md`](docs/critic_inertia_verdict_2026-09-13.md)、`scripts/judge_critic_inertia.py` |
| **经验台账 / 论文计划**（18 次结论推翻、12 类读数病理、12 道检测闸门、R14 指纹、大纲与停止规则） | [`docs/lessons_ledger.md`](docs/lessons_ledger.md)、[`docs/paper_outline_2026-09-13.md`](docs/paper_outline_2026-09-13.md) |
| **信息侧天花板检验 · 第一轮**（预注册 + 修订 3/4/5 → 判读：**形式判决 P0**，三道对照支持 V3 = P3） | [`docs/pomdp_ceiling_prereg_2026-09-13.md`](docs/pomdp_ceiling_prereg_2026-09-13.md)、[`docs/pomdp_ceiling_verdict_2026-09-13.md`](docs/pomdp_ceiling_verdict_2026-09-13.md) |
| **信息侧天花板检验 · 第二轮（结论）**（估计器跑前写死 + 四道对照电池 → **P3/H-LEARN**；belief 通道增量仅 +0.4pp） | [`docs/pomdp_ceiling2_prereg_2026-09-13.md`](docs/pomdp_ceiling2_prereg_2026-09-13.md)、[`docs/pomdp_ceiling_v2_verdict_2026-09-13.md`](docs/pomdp_ceiling_v2_verdict_2026-09-13.md) |
| **外部独立调研的核验**（~44 条断言逐条复算；含文档未有的问题层次"**模型连局部最优都做不到**"） | [`docs/external_review_verification_2026-09-13.md`](docs/external_review_verification_2026-09-13.md) |
| **回放行为取证（卡牌使用 / 圣水 / 合法性）**（仪器 = `scripts/forensics_card_usage.py`；首次把"策略在做什么"变成可复算数字，并推翻「幽灵动作」旧口径） | [`docs/replay_behavior_forensics_2026-09-14.md`](docs/replay_behavior_forensics_2026-09-14.md)、`docs/forensics_card_usage_d1_long_100k.{log,json}` |
| **价值通路逐层线性可读性阶梯**（预注册 + **判读：形式判决 `L0_INVALID`**——我自己的上游闸门阈值跨实验照抄，见【红线 R15】；**描述上 LN 不是掉点层**，且价值头当前权重的放大能力**差 9.7×**） | [`docs/value_ln_probe_prereg_2026-09-14.md`](docs/value_ln_probe_prereg_2026-09-14.md)、[`docs/value_ln_probe_verdict_2026-09-14.md`](docs/value_ln_probe_verdict_2026-09-14.md)、`scripts/probe_value_ln.py` |
| **价值阶梯 · 第四轮（v4，已判读）★**：**真·同张量**对账 ⇒ 断崖是 **`enc_fc`（2731→128 共享投影）**，`enc_ln`/`value_enc_ln` **两处 LN 全被洗清**（P-PROJ 26/27、P-LN 1/27）；附加发现"**训练投影 ≈ 随机投影**"（与 R14 互证）⇒【确证 C11/C12/C13】 | [`docs/value_ln_probe4_prereg_2026-09-14.md`](docs/value_ln_probe4_prereg_2026-09-14.md)、[`docs/value_ln_probe4_verdict_2026-09-14.md`](docs/value_ln_probe4_verdict_2026-09-14.md)、`docs/value_ln_probe4_summary.log`、`runs/_probe_v4/summary.json` |
| **价值阶梯 · 第三轮（v3，已判读）★**：形式判决 `V3_INVALID`+`RATIO_WEAK`，但**推翻了"前端丢 2/3"（X-17）与"比值判据可复现"（X-18）**，并给出**唯一三重受控**的读数（2731→128 共享编码器瓶颈）；立 **R17** + 闸门 11/12 | [`docs/value_ln_probe3_prereg_2026-09-14.md`](docs/value_ln_probe3_prereg_2026-09-14.md)、[`docs/value_ln_probe3_verdict_2026-09-14.md`](docs/value_ln_probe3_verdict_2026-09-14.md)、`docs/value_ln_probe3_mono_seed{7,11,13}.log`、`docs/value_ln_probe3_mono_summary.json` |
| **价值阶梯 · 第二轮（v2，已判读）**：判决 `V2_INVALID`（**G-RAW 阈值用单次观测标定** ⇒【R16】）；本轮两条主结论（"前端丢 2/3 线性可读信号"、"`value_enc_ln` 不是掉点层"）**已被【X-17】/【X-18】推翻**（见上一行）；探针**复现闸 PASS**（播种后同 ckpt/seed 唯一轨迹） | [`docs/value_ln_probe2_prereg_2026-09-14.md`](docs/value_ln_probe2_prereg_2026-09-14.md)、[`docs/value_ln_probe2_verdict_2026-09-14.md`](docs/value_ln_probe2_verdict_2026-09-14.md)、`docs/value_ln_probe2_rngchk_{a,b}.log` |
| **奖励分量分解**（预注册 + 判读：逐帧奖励精确拆五项 → **W4 无主导项**；排除"奖励被资源账带偏"；验证双倍期切价；附带发现塑形缺 `γ`） | [`docs/reward_composition_prereg_2026-09-14.md`](docs/reward_composition_prereg_2026-09-14.md)、[`docs/reward_composition_verdict_2026-09-14.md`](docs/reward_composition_verdict_2026-09-14.md)、`scripts/probe_reward_composition.py` |
| **精确塔伤 · P3（近似闸门否证 → 触发式实现 → 归因与寿命修复）** ★：`threat_precise_prereg` / `threat_precise_probe_verdict`（近似 BA 0.44~0.66）/ `threat_trigger_verdict`（上升沿 vs 电平）/ `threat_precise_cheapening_verdict`（剪枝）/ **`threat_precise_impl_2026-09-14.md`（交付 + 3 seed 验收）** / **`hold_recompute_verdict_2026-09-14.md`（盘面变了会不会重算：不会；寿命/陈旧度/BA 年龄曲线）** / `threat_hold_life_prereg_2026-09-14.md`（寿命上限预注册） | 全部 `docs/threat_*_2026-09-14.md` + `docs/hold_recompute_verdict_2026-09-14.md`（配对 A/B 判读）；仪器 `scripts/probe_precise_threat.py`、`scripts/probe_hold_recompute.py`、`scripts/probe_hold_life_ab.py`（同轨迹配对） |
| **2026-09-17 三项（平局/run 主线/dashboard）** ★：平局 → 皇冠后比**三塔血量合计**、完全相等才平局；run 模式（多卡组）实测修掉 **4 个阻塞级引擎 bug**（各配回归测试）；solo 曲线改**指标多选器**、卡牌统计新增「**按卡组**」矩阵 | [`docs/draw_rule_prereg_2026-09-17.md`](docs/draw_rule_prereg_2026-09-17.md)、[`docs/draw_rule_verdict_2026-09-17.md`](docs/draw_rule_verdict_2026-09-17.md)、仪器 `scripts/analyze_draw_anatomy.py`、`scripts/check_dashboard_js.py`；**`docs/run_mode_multideck_2026-09-17.md`** |
| **2026-09-18 长跑工程（两级评估 / 评估并行 / 1M 步 run 模式）** ★：`_EvalScheduler`（小点保分辨率 + 稀疏大点降噪，两网格取并集）、run 模式评估分片并行（`--eval-workers`，默认仍串行、失败降级）、两个新回归测试 | 预注册 + 判据 [`docs/long1m_prereg_2026-09-18.md`](docs/long1m_prereg_2026-09-18.md)；**dashboard 对接**（目录入口/进度条/两级评估点/逐对手胜率）[`docs/dashboard_long1m_2026-09-18.md`](docs/dashboard_long1m_2026-09-18.md) |
| **★ S1 零成本门禁**（2026-09-18）：`docs/s1_gate_2026-09-18.md` —— `token_strict` 全帧圣水≥6 **0/8191**、Xbow **0/929** ⇒ **未通过 ⇒ 约束在探索/机制侧**（Xbow 不在 `ACE_CARDS`/`SINK_TANK_CARDS` ⇒ 攒费 token 0 帧触发；「只推荐付得起的牌」自锁）；`token_xbow` 加一条手写规则即 **156×** ⇒ 动作空间/掩码被证伪为约束点；冰人白嫖 **证实**（ΔΦ = −2.000000 零散布 ⇒ `edw=0.5` 下 −1.0，双倍期 −0.2）| `scripts/s1_plan_gate.py`、`scripts/phi_offline_check.py` |
| **引擎侧出牌溯源通道**（sandbox 已验、未落真树）：`docs/root_cast_channel_2026-09-18.md` + `docs/root_cast_channel_2026-09-18.diff` —— 行为中立性逐位对账 PASS（1200 tick 全实体 SHA-256 两棵树逐字相同）；落地步骤与回滚写在该文件 §6 | `scripts/_patch_battle_root_cast.py`、`scripts/s2_neutrality_probe.py` |
| **★ S2 仪器实测与门禁判读（★ 第二轮已按修好 6 个口径缺陷后的读数重算）**：`docs/s2_instrument_2026-09-18.md` —— 不变量 **8/8 PASS**；主口径 = `--phi-mode global`（预注册 §2.1 字面读法）；**P3 推翻**（τ==kill_credit 占 75–89% ⇒ 只是已有 Φ 击杀项的重新加权）；**P1/P2 否决**（P1 τ 均值 388–431、ρ(胜) 5/5 批负、兑现率 80.7–88.6%）；**唯一候选 P4b**：**同批配对 Δρ(塔血) 5/5 正（均值 +0.145）、Δρ(胜) 5/5 正（+0.108），换口径同样 5/5（合计 10/10）** ⇒ **S2 相关性门禁算过了**；缺口实测 **20.6%** 窗口「钉住但没打死」（Trade 中位 0 / 均值 −0.36 / p25 −1.00）；σ(Trade)=1.84 主口径；**6 个口径缺陷**（含组件口径把别路下牌算进本窗口 ⇒ 第一版读数作废，§7.7 保留原文）；门禁兑现率 run 模式 99.0–99.7% ⇒ 无分辨力，须换 solo | 仪器 `scripts/offline_engagement_trade.py`、测试 `scripts/selftest_offline_engagement_trade.py` |
| **奖励结算 v3 / 逐帧打分两提案的既有判决 + 逐帧归因提案评审** ★（**原先未被本文件索引**）：E-A/E-B ≈恒等（≤1.5%·std A）、在线状态基线 ≤**+0.0344**、段级无机制（lag-1 .987）、「训打分器」=**不行**；新提案**初判不接受 → §7 澄清后改判**（缺口真实：被阻止的伤害/拉扯现在记 0 甚至负；净交换≈重开窗 Φ + 塔血保全；须**挂 solo**、run 对手是随机脚本无拉扯样本；**免费产物只走溯源路由、绝不估值**=货币泵）、**性能陷阱** = `source=` **是语义参数**（Ronin 会开始吞法术） | [`docs/frame_credit_proposal_review_2026-09-18.md`](docs/frame_credit_proposal_review_2026-09-18.md)（+ `reward_settlement_{prereg,v3_design}_2026-09-14.md`） |
| R1 事件留证（`cudaErrorUnknown` / 宿主提交压力 / 孤儿 worker） | [`docs/r1_incident_2026-09-13_layer1_cuda_unknown.md`](docs/r1_incident_2026-09-13_layer1_cuda_unknown.md) |
| 本文件的过程细节与历史推理链 | [`docs/agents_archive_2026-09.md`](docs/agents_archive_2026-09.md) |
| 文档 ↔ 源码完整索引 | [`docs/README.md`](docs/README.md) |
| RL 代码导读与训练入口 | [`src/clasher_new/rl/README.md`](src/clasher_new/rl/README.md) |
| 自检 / 回归 | **`scripts/run_selftests.py`（按名跑子集，【R19】默认用法）**、`src/clasher_new/rl/selftest.py`（全量，仅 R19 例外情形）、`scripts/test_m*.py`、`scripts/batch_smoke.py` |
| 判读工具 | `scripts/judge_anchor_blocks.py`、`scripts/summarize_solo_run.py`、`scripts/diag_*.py` |
| **代码摸底产物**（逐文件函数全解 / 训练方法 / 游戏引擎；后两者另有 .docx）★ | [`docs/full_code_reference.md`](docs/full_code_reference.md)、[`docs/training_method.md`](docs/training_method.md)、[`docs/game_engine.md`](docs/game_engine.md)；工具链 `scripts/_survey_*.py`（完整保留）；素材 `docs/_survey/` —— ⚠️ **parts/drafts 已于 2026-09-14 清理**，coverage/audit/reverse_check 报告保留（它们是「147 个 `.py`／1479 符号 100% 覆盖」这条结论的证据）；见 `docs/docs_cleanup_2026-09-14.md` |
