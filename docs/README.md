# 文档索引

> 本文件是 `docs/` 与全仓库文档 ↔ 源码文件的导航地图。改文档前先看这里，避免重复/失联。
> 维护约定：新增 curated 文档必须在本文登记；原始采集数据（`_` 前缀文件）只被引用、不手改。

## 0. 快速入口

| 想做什么 | 去哪 |
|---|---|
| 理解模拟器/跑起来 | 根目录 `README.md`（中文）/ `readme_en.md` |
| 启动训练 | `start_rl.bat`（solo/run/flow 三模式 + 仪表盘）、`start_training.bat` |
| 训练算法代码导读 | `src/clasher_new/rl/README.md`（模块表 + 训练入口命令） |
| **所有计划的整合视图**（哪条路走通了/被否证/未决 + 优先级与判据设计） | `plan_master.md` |
| **当前问题的归因**（critic 为何塌成常数 / 上限为何不动 / 判别性实验与预注册要素） | `d1_long_100k_cause_analysis_2026-09-13.md` |
| **critic 惰性检验**（预注册 + 判读） | `critic_inertia_prereg_2026-09-13.md`、`critic_inertia_verdict_2026-09-13.md`（判读脚本 `../scripts/judge_critic_inertia.py`） |
| R1 事件留证（`cudaErrorUnknown` / 宿主提交压力 / 孤儿 worker） | `r1_incident_2026-09-13_layer1_cuda_unknown.md` |
| **逐文件函数全解**（每个代码文件的每个函数：作用/参数/实现/行号；含覆盖对账与编造体检） | `full_code_reference.md`（170 个代码文件 / 1554 条索引） |
| **训练方法文档**（训练过程、被调函数、函数参数、函数索引；docx 同名） | `training_method.md` / `training_method.docx` |
| **游戏引擎文档**（地图 / 寻路 / 索敌 / 战斗主循环；docx 同名） | `game_engine.md` / `game_engine.docx` |
| 跨会话红线与结论索引（先读这个再动方案） | 根目录 `AGENTS.md`（编号可引用：R/C/X/O） |
| 历史决策全文（过程细节与推理链） | `agents_archive_2026-09.md` |
| 自检/回归 | `src/clasher_new/rl/selftest.py`、`scripts/test_m1..m6*.py`、`scripts/batch_smoke.py` |

## 1. Curated 文档（策划/规格/报告，已定稿可引用）

| 文档 | 内容 | 关联源文件/数据 |
|---|---|---|
| `full_code_reference.md` | **项目内容全解（2026-09-14）**：对 `src/` `scripts/` 及仓库根目录共 **170 个代码文件**做逐文件函数级摸底——每个类/函数/方法给出类型、签名（含默认值）、作用、参数、返回、实现步骤、调用关系、置信度，全部带源码行号；含代码库总览、非代码文件清单、**覆盖对账**（147 个 `.py` / 1479 个符号 100% 覆盖）、**签名·行号·参数抽检**（对账 1479 符号：0 类型不符、0 参数缺失；共 170 文件 / 46917 行 / 1554 条索引）与**全局符号索引**（1528 条） | 全仓库代码；生成链 `scripts/_survey_{inventory,groups,brief,verify,audit,merge}.py`（完整保留）；素材 `docs/_survey/`（⚠️ 2026-09-14 已清理 `parts/` 51 份稿件碎片 —— 内容已全文并入本文件；`inventory/groups/coverage/audit/reverse_check` 报告保留） |
| `training_method.md` / `.docx` | **训练方法文档（2026-09-14）**：三种训练模式、一次训练的数据流（观测→前向→动作→奖励→GAE→PPO）、网络逐层结构、**TrainConfig 全 47 字段**、`run_league.py` 全 54 个 CLI 参数、检查点与恢复、评估诊断口径、辅助训练脚本、**rl 全包函数索引**（含字母序快速索引） | `src/clasher_new/rl/*.py`、`battle.py`；素材 `docs/_survey/drafts/training_*.md`（⚠️ 2026-09-14 已清理，内容已并入本文件） |
| `game_engine.md` / `.docx` | **游戏引擎文档（2026-09-14）**：地图与坐标系统、实体体系、`step(dt)` 逐阶段战斗主循环、伤害与命中、皇家塔与胜负判定、**寻路**（两套实现与 A* 代价）、**索敌与目标选择**、卡牌机制分类、法术落点与命中公式、部署合法性 | `battle.py` `arena.py` `core.py` `player.py` `pathfinding*.py` `card_mechanics.py` `spell_module.py` `threat_calc.py` `rl/action_mask.py` |
| `../AGENTS.md` | **红线与结论索引**（2026-09-13 重构）：12 条红线（R）+ 10 条已确证（C）+ 9 条已否证（X）+ 5 条未决（O）+ 计划台账 + run 台账；条目编号可引用 | 全部 `rl/` 与 `scripts/`；过程细节见 `agents_archive_2026-09.md` |
| `plan_master.md` | **计划总纲（所有计划的整合视图）**：决策地图（8 个方向的通/否/未决）、阶段结论、13 份计划的谱系台账、**当前优先级 P0-P4（含判据设计要点与成本）**、纪律清单 | `rl_training_fix_plan_v1/v2/v3.md`、`rl_review_fix_plan.md`、`rl_reward_plan_v2.md`、`rl_plan_design_v1.md`、`ai_training_plan.md`、`P0-mechanics-plan.md`、`mcts_design.md` |
| `d1_long_100k_cause_analysis_2026-09-13.md` | **当前问题归因（2026-09-13）**：价值分支参数级指纹（35k→100k 的 22~24/27 窗口位移**恰为 0** ⇒ 价值函数=常数）、第一性原因（末端 LayerNorm ⇒ 可表达方差 0.14 vs 目标 std 12.11 ⇒ 自举失败 ⇒ ReLU 死亡锁死）、上限不动的策略侧成因（相对目标 + RPS）、**新否证 X10**（锚点每点重抽 40 局）、**新红线 R14**、3 处对上游报告的口径更正、4 条判别性实验 | `runs/d1_long_100k/`（41 ckpt + replays）、`rl/follower.py`、`rl/ppo.py`、`rl/env_wrapper.py`、`rl/train_solo.py` |
| `report_d1_long_100k_structured_2026-09-13.md` | 上述归因的**结构化取证全文**（6 路并行子任务 + 评审归并，41 KB）：训练信号预算基数、critic 目标与归一化、对手漂移时间尺度、行为病理、奖励分解、仪器偏差；含 4 处子任务截断披露 | 只读证据（原始行号与数字） |
| `agents_archive_2026-09.md` | **历史决策全文存档**（旧 `AGENTS.md` 逐字冻结，100,068 B，sha1 见文头）：含全部过程叙述、证据链、逐点判读表与已过期中间结论 | 只读；新结论写 `../AGENTS.md` |
| `P0-mechanics-plan.md` | P0 游戏机制全面补全计划（族1-8）与进度 | `battle.py` `card_mechanics.py`；回归 `scripts/test_m2.py` `scripts/batch_smoke.py` |
| `card_coverage.md` | 卡牌覆盖矩阵（148 条快照、75 张 implemented 口径） | `card_utils.py` `gamedata.json` `cards_stats_*.json`；`scripts/coverage.py`；`card_registry.json` `batch_smoke_report.json` |
| `review_needed.md` | 人工评审队列（L1 结论与剩余未知） | 同上 |
| `数值规则查证汇总.md` | 觉醒周期表/英雄能力/法术锚点三线查证（gamedata=1级基准 vs wiki=11级基准，换算 ≈1.1^10） | `gamedata.json` `cards_stats_spell.json` `evolutions.py` |
| `errata_batch2_13_implementation.md` | 勘误批 2-13 实施汇总（用户三原则 + 回归红线） | `battle.py` `card_utils.py` `core.py` 等（批量）；回归 `test_m2..m6` |
| `data_integration_notes.md` | M7 三卡（Ronin/Vines/Spirit Empress）引擎钩子决策 | `card_mechanics.py` `battle.py`；证据 `docs/_fp_*.txt` |
| `evo_2025_new7.md` | 2025 新觉醒 7 张实现规格（快照无 evolvedSpellsData → 独立数据层） | `evo_2025_data.py` `evolutions.py`；`scripts/test_m3_evo.py` `test_m4_evo7.py`；证据 `docs/_evo_*.txt` `_*_evo.txt` |
| `evo_hook_gap_matrix.md` | 觉醒机制钩子缺口矩阵（42 张闭环审计） | `evolutions.py` `battle.py`（evolved 分支） |
| `elite17_spec.md` | Elite17 精英/Hero 卡规格（内存=官方 15.535 权威，Fandom 双记冲突） | `elite17_data.py` `battle.py`（Hero overlay）；`scripts/test_m6_elite.py`；证据 `docs/_elite_*.txt`；附录 `re/official/extracted/elite17/`（不入库） |
| `_elite_fandom_report.md` | Elite 抓取过程报告（工具/落盘约定） | `scripts/cdp_extract.js`；原始落盘 `docs/_elite_*.txt` |
| `ai_training_plan.md` | RL 训练算法闭环规划（先知/信念/跟随者/联赛） | `rl/` 全包（导读见 `../src/clasher_new/rl/README.md`） |
| `rl_review_fix_plan.md` | RL 链路修复手册（6 条 Critical 已修） | `rl/selftest.py`（回归） |
| `rl_reward_plan_v2.md` | 奖励与计划通道重构 v2（奖励经济预设、法术空砸/裸下/坦克几何闸门） | `rl/config.py` `rl/env_wrapper.py` `rl/action_mask.py` |
| `rl_plan_design_v1.md` | PlanToken 战术意图扩展设计 v1（57 维/21 意图） | `rl/plan_space.py` `rl/belief_planner.py` `rl/prophet.py` `rl/follower.py` |
| `mcts_design.md` | 推理时浅 MCTS 设计（UCT+引擎叶推演，预算/对手口径/评估接入） | `rl/mcts.py` `rl/selftest.py`（test_mcts_*） |
| `question_bank_feasibility.md` | 题库预训练可行性评估（POC 实测：引擎判卷 3 题+checkpoint 答题 EV gap；判卷口径两缺口；三步走路线） | `../scripts/question_bank_poc.py` `simulate_exchange.py` `threat_calc.py` `rl/follower.py` |
| `training_audit_2026-09-11.md` | **训练体系审计（四问：训练问题/评判指标/步数预算/参数量）**：评论家损失主导更新（gnorm 中位 6.8k vs 阈值 0.5、clip_frac 恒 0%）、指标分辨率与锚点缺失、三档步数建议、629,359 参数实测 | `rl/ppo.py` `rl/config.py` `rl/train_solo.py` `rl/evaluate.py` `rl/follower.py`；证据 `docs/train_economy_*.log` `runs/*/solo_state.json` |
| `rl_training_fix_plan_v1.md` | **训练整改计划 v1**（P0 价值通道/指标可信化 → P1 结构性机制 → P2 工程；含 G1/G2/G3 门禁、单变量 A/B 协议、回归测试清单、风险登记册、明确不做项） | `rl/ppo.py` `rl/config.py` `rl/train_solo.py` `rl/follower.py`；依据 `training_audit_2026-09-11.md` |
| `ab_valnorm_20k_verdict_2026-09-11.md` | **`ab_valnorm_20k` 20k 步验证跑 G1 判读**：`step`=决策帧（20k 步≈55~80 局，非"训练很多"）、`v/p` 2.43→0.95 达成、`ratio≡1.000` 判据作废、**critic 解释方差≈0（新发现）**、三曲线净变化为零、门禁阈值口径错位（9.5 vs 实测 28~38） | `rl/ppo.py` `rl/config.py` `rl/train_solo.py`；证据 `../docs/train_ab_valnorm_20k.log` `runs/ab_valnorm_20k/` |
| `rl_training_fix_plan_v2.md` | **训练整改计划 v2**（v1 的修订版）：G1 判据 `ratio`→**EV**、门禁改**相对首个评估点**、预算改**局数口径**、对手池 `0.7/0.2/0.1`→`0.5/0.3/0.2`、main 曲线**先评估后同步**；含 200k 长跑协议与 G1/G2/G3 分支 | `rl/ppo.py` `rl/config.py` `rl/train_solo.py` `rl/dashboard.py`；依据 `ab_valnorm_20k_verdict_2026-09-11.md` |
| `value_channel_saturation_diagnosis_2026-09-11.md` | **价值通道失效根因诊断（GRU 输入饱和）**：`enc` 量级 533（CNN 输出 468）→ GRU tanh 候选饱和(abs 0.994) → 隐状态 h 冻成常数(跨帧 std 2.6e-5) → critic 恒为常数，EV 精确等于 −bias²/Var(R)；另发现"128 连续帧批"把 EV 放大约 3 倍；病理跨全部历史 run | `rl/follower.py` `rl/ppo.py` `rl/train_solo.py`；证据 `scripts/diag_critic_ev.py` `diag_value_head.py` `diag_encoder_scale.py` `diag_gru_ablation.py`；数据 `runs/prod_200k_valnorm_ev/` |
| `rl_training_fix_plan_v3.md` | **训练整改计划 v3**：纠正 v2 §4 病因误判（非奖励/价值头，而是 GRU 饱和）；P0-A enc 后加 LayerNorm、P0-B EV 改池化口径 + 加 GRU 活力指标、P0-C 诊断常态化 + **启动前架构护栏**；§3.6 = **5k 验证跑判读**（n_abs 0.994→0.46/0.55、EV −0.58→−0.06 单调上升未过零、**不执行 v2 §4 高侵入项的理由**、补齐"值/R std"缺口、20k 协议与判读门槛/分支）；§3.7 = **enc_ln 20k 判读**（饱和修复成立、critic 仍常数、查出融合层尺度失衡 grid 占 fused 98%）；§3.8 = **`grid_ln`（P0-A 备选）落地 + 20k 判读**（尺度修复达成：grid 101→5.87；EV_global −0.134→−0.005 只修对中未修拟合；`n_abs` 回升 0.745；**`vs baseline0` 崩塌 0.125 = cycling 红旗**）；§3.8.4 = **A′ 取证**（RPS 三角确凿、绝对强度≈随机、早停裁定混淆）；§3.8.5 = **E1 固定随机锚点落地**（每评估点 `vs baseline_rand` 对照 + 报警线；短验证 0.15/0.15 确定）；§3.9 = **E2 训练侧锚点对手改造**（`rand_anchor` 进训练对手池第 4 槽 frozen 0.4/hist 0.3/defend 0.2/rand 0.1，锚点与 E1 同权重、附带修无 hist 归一化缺陷）；§3.9.3 = **E2 20k 判读**（干预无效：vs baseline_rand 末点 0.050、vs baseline0 0.075，绝对强度崩溃依旧、RPS 循环仍在转、critic 依旧常数；10% 弱锚点压不住自对弈漂移） | `rl/follower.py` `rl/train_solo.py` `rl/ppo.py` `rl/diagnostics.py` `rl/dashboard.py`；依据 `value_channel_saturation_diagnosis_2026-09-11.md`；数据 `runs/fix_gru_ln_5k/` `runs/fix_gru_ln_20k/` `runs/fix_gru_ln_norm_20k/` `runs/e2_rand_anchor_20k/` |
| `fprime_rerun_20k_verdict_2026-09-13.md` | **F′ 修正口径复跑判读（critic 侧收尾）**：与 `fprime_20k` 训练语义等价、同 seed/协议，唯一差异 = 无塔血通道 ⇒ G'-fix 的单变量对照；EV(更新前池化) 均值 **+0.031**（首跑 +0.198 是 in-sample 假象）；同窗口 `vstd/rstd` 8.5× 于 legacy 但 EV 不涨（"活起来的方差不可预测"）；离线 `EV_global −0.159` 且查出 EV 为负的**第二个来源 = 量纲失配**（回归斜率 `R~v` 10~525）；无塔血通道架构上 `enc→塔血差` 分组 R² −0.17（反证 G'-fix 的 +0.69 是通道真接上）；末点崩塌 = cycling 相位（非 G'-fix 特有）；**方法学：同 seed 同代码两次 run 前 435 步逐位一致、之后 1e-4 漂移、`eval@800` 已不同（`PYTHONHASHSEED=0`+单线程 BLAS 无效，机制未定）⇒ 20k 单跑 A/B(n=1/臂) 分辨率不足**；判定 = critic 停、转策略侧 | `rl/train_solo.py`（分支 C 仅度量修复）；证据 `../docs/train_fprime_ev_20k.log` `diag_predict_fprime_ev.log` `diag_vh_fprime_ev_20k.log` `_repro_{a,b,c,d}.log`；数据 `runs/fprime_ev_20k/` |
| `cycling_league_plan_2026-09-13.md` | **D1 预注册**（判据先钉死再改代码）：对手分布去镜像化（frozen 0.4→0.1 / hist 0.3→0.6）+ 动态历史自身联赛（`refresh_hist`）+ PFSP 门禁（alpha 0.05→0.20、易胜对手 ×0.2）；含 P1/P2 判据、判定分支、协议与可观测指纹 | `rl/train_solo.py` `rl/pfsp.py` `rl/config.py`；产物 `runs/d1_league_20k/` |
| `d1_long_100k_verdict_2026-09-13.md` | **D1 100k 长跑判读（已完成）**：**P1 通过**（块 worst 0.388/0.475/0.275/0.237/0.525 → mean 0.380、min 0.237，与无变化组 [0.000,0.025] 不重叠）⇒ **防崩在 4× 训练量（≈374 局）下成立**；min 0.237 落在 D1 20k 区间 [0.125,0.250] 内 ⇒ 该判据上与 20k 无统计差别；**上限仍未解决**（次判据口径敏感 +0.125/+0.062、cycling 未消除）；**加量不能救 critic**（EV 仅前两点为正，`vstd/rstd` 30k 后 ≈0.001，8 次 vitality 告警），但 GRU 活力 100k 全程健康（v3 修复耐久）；含**第三次同类口径自纠**（预注册写 9,8,8,8,8，脚本按序号实切 9,9,9,9,5，两口径都算且均 PASS） | 工具 `scripts/judge_anchor_blocks.py` `scripts/summarize_solo_run.py`；证据 `../docs/train_d1_long_100k.log` `diag_d1_long_100k.md`；数据 `runs/d1_long_100k/` |
| `eval_cadence_c_2026-09-13.md` | **评估节奏 C 方案（密锚点 + 稀全块）**：实测 20k 协议里评估占 **73% 墙钟**（纯训练 26 步/s、全点 210s、轻点 53s）⇒ 100k 从 3.46h 降到 **2.15h**；**为什么不能均匀放宽**（真实序列粗采样回放：D1 三跑锚点谷底只有 1 个点宽、5000 即开始漏、10000 把最差点 0.192→0.462；病态组持续塌陷留得住、健康组周期性瞬态留不住）；实现 `--anchor-every`（默认 0=旧行为）+ 轻点同密度落快照/Adam/run_state；**C ⊇ A 主判据**（锚点读数逐点相同，确定性论证）；两跑冒烟验收 | `rl/train_solo.py` `rl/config.py` `rl/run_league.py` `rl/selftest.py`；证据 `../docs/smoke_anchor_c{,2}.log` |
| `d1_long_100k_prereg_2026-09-13.md` | **D1 100k 长跑预注册**（跑前写死）：唯一变量=训练量；主判据 **C1-块口径**（41 锚点分 5 块、每块 9 点 worst，与 D1 20k 基线同构）；次判据仅描述（cycling 未消除 ⇒ 天花板不设判决线）；跑前披露 5 条偏差（补种 ckpt 挤出、n=1/臂、锚点数 41 vs 9、resume 粒度、运行期冻代码）；失败分支事先写死 | 代码 = `main`（D1）；协议命令与预算见文内 |
| `d1_league_20k_verdict_2026-09-13.md` | **D1 判读（三跑）**：C1「单跑最差锚点」D1 0.125/0.200/0.250 vs 无干预 0.000/0.000/0.025 **区间不相交** ⇒ **消除"整段输给固定随机策略"的相位（防崩有效）**；用户指定的"末 4 点滚动均值"两组重叠（相位主导）⇒ **未证明提高上限**；含**两处预注册标定错误的自披露**（基线列手抄算弱）与新纪律 | `rl/train_solo.py` `rl/pfsp.py`；证据 `../docs/train_d1_league_20k{,_r2,_r3}.log`；数据 `runs/d1_league_20k{,_r2,_r3}/` |
| `../scripts/forensics_response.py` | 9j 响应率取证脚本（防守响应占比/延迟/落点距离，v2 修正坐标口径） | `AGENTS.md` 9j 节（结论存档） |

## 2. 外置工具（引擎侧确定性服务，2026-09 起步；设计与优先级见 `../AGENTS.md`）

| 工具 | 文件 | 语义 | 对账 selftest |
|---|---|---|---|
| ① 塔伤威胁计算器 | `../src/clasher_new/threat_calc.py` | "双方不再部署"下敌方现存部队 20s 内对我方各塔伤害 | `test_tower_threat_calc` |
| ② 交换模拟器 | `../src/clasher_new/simulate_exchange.py` | "我方现在打出这张牌"的反事实推演（none/script/fn 三档对手） | `test_simulate_exchange` |
| ③ 法术知识模块 | `../src/clasher_new/spell_module.py` | 法术引擎标定档案 + 落点估值 + 引擎对账口径 | `test_spell_module` |
| ④ 循环规划器 | 未实现 | 牌序 depth/过牌 ETA 特征（私有信息+确定性队列） | — |

## 3. 原始采集数据（`_` 前缀 / `.cdp*` / `.wikitext`，只读证据）

> **⚠️ 2026-09-14 已清理（用户指令「清除编写文档的中间文件」）**：本节列出的原始抓取文件
> （`_page_*` / `_elite_*` / `_fp_*` / `_evo_*` / `_fandom_*` / `_hero_*` / `_spell_*` / `_*_evo.txt` / 页面级 `.cdp*.js`，
> 共 **190 个 / 6.88 MB**）**已删除**。下表保留**命名约定与引用方**，供将来重新抓取。
> 重新抓取：`node scripts/cdp_extract.js`（通用页）、`node scripts/cdp_evo.js`（觉醒页）。
> 清单与保留项：`docs/docs_cleanup_2026-09-14.md`。

由 CDP 脚本从 Fandom Wiki 抓取，被上表 curated 文档按 `[Fandom]` 标注引用。**禁止手改**；
数值疑义以 `gamedata.json`（游戏快照）为权威，Fandom 只作机制参考（用户三原则）。

| 命名 | 内容 | 引用方 |
|---|---|---|
| `_page_*.txt`（+`_page_*_wiki.txt`/`_clean`） | Fandom 卡牌/机制页整页 | `evo_2025_new7.md`、勘误批报告 |
| `_fp_<卡名>.txt` | 卡牌策略页（"From Players"） | `data_integration_notes.md`、法术病理取证 |
| `_elite_<卡名>.txt` / `_elite_*.txt` | 精英卡逐卡页 + Hero 总览 | `elite17_spec.md`、`_elite_fandom_report.md` |
| `_evo_*.txt` / `_<卡名>_evo.txt` | 觉醒卡页 | `evo_2025_new7.md`、`evo_hook_gap_matrix.md` |
| `_fandom_b06_*` | 批次6 原始页 + wikitext | 勘误批报告 |
| `.cdp*.js` | CDP 抓取/解析脚本（页面级临时件） | `_elite_fandom_report.md` |
| `_digest_v3_for_master_plan.md`、`_digest_three_plans.md`、`_digest_five_plans_2026-09-13.md`、`_five_reports_merged_2026-09-13.md` | 2026-09-13 计划整合时**并行子智能体产出的原始 digest**（各自读一组计划文档后的结构化压缩），只读证据 | `plan_master.md` §6（出处） |
| `re/fandom_check/`、`re/fandom_stats/`（在 `.gitignore` 的 `re/` 内，不入库） | 批次校验报告与结构化 JSON | 勘误批/觉醒批 |

## 4. 数据文件（docs/ 下，curated 可引用）

| 文件 | 内容 | 消费方 |
|---|---|---|
| `card_registry.json` | 卡牌注册/覆盖状态 | `card_coverage.md`、`scripts/coverage.py` |
| `batch_smoke_report.json` | 全卡冒烟结果 | `scripts/batch_smoke.py` |
| `evolution_cycles.json` | 觉醒周期表 | `evolutions.py` |
| `leaderboard_decks.json` / `leaderboard_decks_classified.json` | 天梯 200 副卡组（推进 60/防反 120/自闭 20） | `rl/decks.py`、`rl/run_league.py`、`rl/opponents.py` |
