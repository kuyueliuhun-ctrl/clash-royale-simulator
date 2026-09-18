# `scripts/` 全量登记表（2026-09-19 生成，Tier 0 · T0-3）

> **为什么有这份文件**：[`env.md`](env.md) §2.4 的「判读 / 诊断工具」表只登记了 **40** 个 `.py` 名，
> 而 `scripts/` 顶层实有 **91** 个 `.py` ⇒ **57 个未登记**。把 57 行塞进 `env.md` 会让那张表失去「一眼可读」，
> 故按本仓既有做法（索引 + 分册）建此全量表，`env.md` §2.4 只留**一行指针**。

> **生成方式**：脚本抽取每个文件的模块 docstring 首行 + 是否含 `--selftest` + `assert` 计数 + 是否已被 `env.md` 引用。
> **⚠️ 口径**：本表统计**顶层 `scripts/*.py`（91 个）**；若把 `.sh/.ps1/.js` 与 `scripts/rl/*.py`(11) 算进来则是 **105** 个文件 ——
> `docs/structure_optimization_plan_2026-09-19.md` §5 写的「**69 个未登记**」用的是后一套口径，**两套并列，不调和**。

**合计**：顶层 `scripts/*.py` **91** 个；其中已在 `env.md` 登记 **34** 个、**未登记 57** 个。


## 内部工具（`_` 前缀）（21 个）

| 脚本 | env.md 已登记 | 一行职能（docstring 首行） | `--selftest` | assert |
|---|---|---|---|---|
| `_agents_split.py` | ✅ | AGENTS.md 拆分 + 不丢内容校验（一次性迁移工具，保留作证据）。 |  | 2 |
| `_forensics_cycling.py` | ✅ | cycling 取证（2026-09-12，计划 v3 §3.8.3 A′）。 |  |  |
| `_mask_ab_prefix.py` | ❌ | 临时 A/B（下划线前缀 = 不进正式仪器表）：`used` off-by-one 修复前后的对照。 |  |  |
| `_mask_diff_snapshot.py` | ❌ | P0 掩码优化逐位对账脚本。 |  |  |
| `_mask_vs_engine_reconcile.py` | ❌ | 掩码 ↔ 引擎 部署合法性对账（R13 类取证；**只读**，不改任何判定逻辑）。 |  |  |
| `_patch_battle_root_cast.py` | ✅ | 把 S2 的「出牌溯源纯记录通道」打进 battle.py（sandbox 里先做，验证后再进真树）。 |  | 6 |
| `_probe_eval_frame_ab.py` | ❌ | 评估侧局长 A/B：同一批权重（eval@0 = 全新初始化，seed 0）下，早停开关对局长/结果的影响。 |  |  |
| `_probe_value_collapse.py` | ❌ | 只读取证：value 通路逐层方差解剖（不写盘、不改训练）。 |  |  |
| `_probe_value_path.py` | ❌ | 只读取证：独立价值通路逐层方差解剖（不写盘、不改训练）。 |  |  |
| `_schema5_probe.py` | ❌ | sandbox 验证：schema 5（root_cast / is_product / share + 帧内 v0/v1）是否端到端可用。 |  | 5 |
| `_structure_check.py` | ❌ | 只读结构核对器（Tier 0 · T0-2）。 | ✅ |  |
| `_survey_audit.py` | ❌ | 签名/行号一致性抽检（防编造的第二道闸）。 |  |  |
| `_survey_brief.py` | ❌ | 打印某一组的任务简报（供子代理读取：文件清单 + 必须覆盖的符号清单）。 |  |  |
| `_survey_groups.py` | ❌ | 把 inventory.json 的文件切成「子代理任务组」，保证**全量覆盖、无文件遗漏**。 |  | 1 |
| `_survey_inventory.py` | ❌ | AST 静态清单生成器（文档摸底用，只读）。 |  |  |
| `_survey_md_to_docx.py` | ❌ | Markdown → DOCX 转换器（离线、无外部依赖，仅用 python-docx）。 |  |  |
| `_survey_merge.py` | ❌ | 把 48+ 份逐文件分析素材（docs/_survey/parts/*.md）机械合并成 |  |  |
| `_survey_merge_docs.py` | ❌ | 把 A/B 两部分草稿合并成《训练方法文档》/《游戏引擎文档》（Markdown）。 |  |  |
| `_survey_reverse_check.py` | ❌ | 抽样反查 / 防编造体检（**只读**，可复算）。 |  |  |
| `_survey_verify.py` | ❌ | 覆盖对账（防漏文件 / 防漏符号 / 防凭空造符号）。 |  |  |
| `_verify_fix_replays.py` | ❌ | 临时验证（下划线前缀 = 不进正式仪器表）：用**修复后**的代码跑评估路径并落**真录像**。 |  |  |

## 测试（8 个）

| 脚本 | env.md 已登记 | 一行职能（docstring 首行） | `--selftest` | assert |
|---|---|---|---|---|
| `selftest_engagement_trade_online.py` | ❌ | 预注册 §7-7 `test_engagement_trade_default_off`（**引擎侧**）：默认关 ⇒ 逐位回旧。 |  | 15 |
| `selftest_offline_engagement_trade.py` | ✅ | 预注册 §7 的 7 条不变量 —— 前 6 条是**离线仪器**的测试（零训练成本、零引擎依赖）。 |  | 66 |
| `test_m1.py` | ❌ | M1 机制族测试：递增伤害 + 弹道生成链。直接 python3 scripts/test_m1.py 运行。 |  |  |
| `test_m2.py` | ❌ | M2/M3/M4 机制族验收：族3 最小射程 / 族4 buff 槽 / 族5 拉拽 / 族6 英雄能力 / 族7 觉醒。 |  |  |
| `test_m3_evo.py` | ❌ | M5 觉醒补全验收：动作组解释器 / Pekka 临时复活 / MegaKnight 冲锋 / ElectroDragon 链电 / |  |  |
| `test_m4_evo7.py` | ❌ | M6 — 2025 新觉醒 7 张验收（快照无 evolvedSpellsData, 数据层 evo_2025_data.py + 引擎钩子）。 |  |  |
| `test_m5_data.py` | ❌ | M7 — 数据接入三卡引擎机制验收（Ronin 格挡反击 / Vines 藤蔓束缚 / Spirit Empress 双费用形态）。 |  |  |
| `test_m6_elite.py` | ❌ | M8 — Elite17 精英卡（Hero 化）引擎机制验收（17 卡 × ≥2 行为断言）。 |  | 1 |

## 测试/带自检（9 个）

| 脚本 | env.md 已登记 | 一行职能（docstring 首行） | `--selftest` | assert |
|---|---|---|---|---|
| `assassin_left_bridge_test.py` | ❌ | 用户指定测试（2026-09-09 修订版）：完美解 = 蓝方塔不掉血。 |  | 1 |
| `assassin_vs_megaknight.py` | ❌ | 「刺客解超级骑士」对卡实验（2026-09-09，用户技巧口径）。 |  | 1 |
| `assassin_vs_sparky.py` | ❌ | 「刺客完美解满蓄力电磁炮」搜索脚本 v2（2026-09-10，含防御塔参与）。 |  | 1 |
| `et_solo100k_readout.py` | ✅ | `et_solo100k` 判读执行器：一条命令跑完 §11.13.2 的四层。 | ✅ |  |
| `health_curve.py` | ✅ | 训练健康曲线读数：**价值损失**与**策略熵**（稠密，逐 update）+ 平台（步数瓶颈）描述性读数。 | ✅ |  |
| `judge_critic_inertia.py` | ✅ | critic 惰性检验判读器（预注册 docs/critic_inertia_prereg_2026-09-13.md）。 | ✅ | 7 |
| `probe_explore_randomization.py` | ✅ | 「随机出牌 + 随机位置 + 随机概率正弦/衰减」方案的**可行性取证**仪器（只读）。 |  | 2 |
| `question_bank_poc.py` | ❌ | 题库预训练可行性 POC（2026-09-09）：不写死答案，引擎判卷。 |  | 1 |
| `s1_plan_gate.py` | ✅ | S1 门禁：PlanToken 贪心执行器 vs 同一批局的基线（只读引擎、纯 CPU、零训练成本）。 |  | 1 |

## 一次性取证 / 仪器（53 个）

| 脚本 | env.md 已登记 | 一行职能（docstring 首行） | `--selftest` | assert |
|---|---|---|---|---|
| `analyze_draw_anatomy.py` | ❌ | 平局解剖 + 新旧裁决口径**配对**对照（只读，数据来自已落盘回放）。 |  |  |
| `analyze_online_trade.py` | ❌ | 用**在线口径**量「局面圣水交换」（S2 第四轮；预注册 §11.9.4 的放行前置）。 |  |  |
| `batch_smoke.py` | ❌ | 批量卡牌冒烟测试：全 gamedata 卡 构造→部署→30s 战斗，检查「有行为」（生成单位或造成伤害）。 |  |  |
| `bench_train_speed.py` | ❌ | 训练速度基准（2026-09-12）：跑一个短 solo run，实时给日志行打时间戳， |  |  |
| `cdp_forward.py` | ❌ | CDP 端口转发器：WSL 127.0.0.1:9222 → Windows 主机 <网关IP>:9222 |  |  |
| `check_commit.py` | ✅ | 长跑开跑前的宿主**提交内存**检查（【红线 R1】）。 |  |  |
| `check_dashboard_js.py` | ✅ | 前端冒烟回归：dashboard.py 内嵌 JS 的**真实运行级**检查（不只是语法）。 |  |  |
| `coverage.py` | ❌ | 卡牌内容覆盖矩阵生成器（P0-1 基本信息录入）v2 —— 证据驱动分类 |  |  |
| `diag_critic_ev.py` | ✅ | Critic EV 取证脚本（2026-09-11，§4 诊断）。 |  |  |
| `diag_encoder_scale.py` | ✅ | encoder 输入尺度定位（2026-09-11 §4 诊断第三步）。 |  |  |
| `diag_gru_ablation.py` | ✅ | GRU 饱和归因实验（2026-09-11 §4 诊断第四步）。 |  |  |
| `diag_value_head.py` | ✅ | 价值头解剖（2026-09-11 §4 诊断第二步）。 |  |  |
| `duel_search.py` | ❌ | 泛化对抗搜索器：敌方行进 0.1 格评估 × 我方整格部署，求最小损失。 |  |  |
| `extend_level16.py` | ❌ | 数据文件 16 级支持：把 cards_stats_*.json 中所有 per_level 数值数组延伸到「稀有度规范长度」。 |  |  |
| `forensics_card_usage.py` | ✅ | 只读取证：联赛回放的**卡牌使用 / 圣水 / 部署节奏 / 动作合法性**（只读、离线）。 |  |  |
| `forensics_response.py` | ✅ | 取证 v2：单边堆牌 vs 对牌响应率（2026-09-08，坐标系已修正）。 |  |  |
| `judge_anchor_blocks.py` | ✅ | C1 判据判读器：锚点序列 → 分块 worst/median → 预注册判据（含基线组复算）。 |  |  |
| `judge_probe_v3.py` | ✅ | 只读：v3 探针的**跨 seed 归约与判读**（对应预注册 §5/§6/§7）。 |  |  |
| `judge_probe_v4.py` | ✅ | 只读：v4（真·同一张量的投影/归一化对账）跨 seed 归约 —— 判据全部脚本复算（R4）。 |  |  |
| `offline_engagement_trade.py` | ✅ | 离线「局面圣水交换」仪器（S2，零训练成本，只读回放）。 |  |  |
| `pass_streak_audit.py` | ✅ | 攒费样本审计 v2：把「不出牌」拆成三类，并**自己独立验**非法性（只读录像）。 |  |  |
| `phi_offline_check.py` | ✅ | 离线逐帧复算 Phi（资源账势函数）+ 与回放 reward 对账（只读、纯 CPU、零训练成本）。 |  |  |
| `pomdp_ceiling_probe.py` | ❌ | 只读探针：POMDP 信息天花板检验。 |  |  |
| `probe_belief_accuracy.py` | ❌ | 信念输入准确度实测（只读）：belief_token 里那几项**到底准不准**。 |  |  |
| `probe_channel_gradient.py` | ❌ | plan / belief 辅助通道的「梯度死活」判决（P1；只读）。 |  |  |
| `probe_credit_baseline.py` | ❌ | 结算基线可行性探针（只读、离线）：闭式"状态基线"能不能替死掉的 critic 干活？ |  |  |
| `probe_encoding.py` | ❌ | 模型输入编码清单（只读，脚本复算）：网络到底"看到"了什么、丢掉了什么。 |  |  |
| `probe_engine_profile.py` | ❌ | 引擎每 tick 成本的**族归类**剖分（只读仪器）。 |  |  |
| `probe_gae_kernel.py` | ❌ | GAE 核函数体检（只读）：GAE 会不会把逐帧奖励"平均化/抹平"？ |  |  |
| `probe_hold_life_ab.py` | ❌ | 保持寿命上限（MAX_HOLD_S）的**同轨迹配对 A/B**（只读）。 |  |  |
| `probe_hold_recompute.py` | ❌ | 保持式精确塔伤的状态机取证：**盘面变了会不会重算**（只读探针）。 |  |  |
| `probe_pass_prob.py` | ✅ | 「不出牌」（空 bundle / noop）概率实测（只读仪器）。 |  |  |
| `probe_pathfix.py` | ❌ | 评估「条件满足时算一次路径并固定跟随」能省多少（只读仪器）。 |  |  |
| `probe_precise_threat.py` | ❌ | `rl/threat_precise.py` 的对账 + 标定 + 验收（只读）。 |  |  |
| `probe_reward_composition.py` | ✅ | 只读：100k 策略的**逐帧奖励分量分解**（引擎真值，不做重建）。 |  |  |
| `probe_reward_semantics.py` | ❌ | 奖励语义单元验证（只读）：逐帧分数是「对比上一帧的差分」还是「当前帧的绝对局面」？ |  |  |
| `probe_scorer_offline.py` | ❌ | "训练一个打分器给每帧打分"的离线可行性探针（只读）。 |  |  |
| `probe_search_cost.py` | ❌ | 搜索/推演成本实测（只读）：确认"搜索+推演"路线的成本账是否仍成立。 |  |  |
| `probe_settlement_units.py` | ❌ | 结算单位代价探针（只读、离线）：把"按帧计分"换成事件/局末结算，代价是多少？ |  |  |
| `probe_threat_approx.py` | ❌ | 近似威胁估计 vs 引擎精确塔伤：逐帧对照 + 成本测量（P3-1；只读）。 |  |  |
| `probe_threat_prune.py` | ❌ | 精确塔伤的廉价化可行性：地理剪枝的剪枝率 + **可靠性**（P3-2 阶段 A；只读）。 |  |  |
| `probe_threat_trigger.py` | ❌ | 评估「己方半场无兵 且 对方半场的兵已到桥头 ⇒ 调一次精确塔伤」这个触发条件（只读）。 |  |  |
| `probe_tool_usage.py` | ❌ | 外置工具接线核查 + 模型敏感度实测（只读）： |  |  |
| `probe_v3_mono_check.py` | ✅ | 只读诊断（**跑后追加，非预注册**）：v3 阶梯的 `α` 选择是否让层间比较失去意义？ |  |  |
| `probe_v4_ln_pair.py` | ✅ | 只读：价值通路探针 v4 —— **真·同一张量**的「投影 / 归一化」分段对账。 |  |  |
| `probe_value_ln.py` | ✅ | 只读：价值通路「逐层线性可读性阶梯」。 |  |  |
| `random_eval_100.py` | ✅ | 「初始模型 + 全过程随机」100 局评估 runner（只读，产出 dashboard 可读的录像与状态）。 |  |  |
| `run_selftests.py` | ✅ | 按名运行 `rl/selftest.py` 里的**单个/指定**测试（配合【红线 R19】：不跑全量）。 |  |  |
| `s2_neutrality_probe.py` | ✅ | 行为中立性探针：在**两棵树**上跑同一段确定性推演，打印可对账摘要。 |  |  |
| `s2_trade_probe.py` | ❌ | 开/关逐位对账 + 在线↔离线交叉复算（S2 §6 第 5 项）。 |  |  |
| `summarize_probe_v3_mono.py` | ✅ | 只读：把 v3 的 `mono_check` 日志解析成**脚本复算**的汇总（【红线 R4】禁止手抄）。 |  |  |
| `summarize_solo_run.py` | ✅ | 长 run 诊断汇总器：日志 + solo_state.json + gates.json → 一段 markdown（判读用）。 |  |  |
| `value_displacement_scan.py` | ✅ | 只读：参数位移指纹扫描（R14）。 |  |  |
