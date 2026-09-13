# 五份文档 digest（仅压缩，不评价；取数 2026-09-13）

`rl/selftest.py` 89 个 test。

**1 rl_plan_design_v1.md**
- 定位：PlanToken 战术意图 v1；"前置：reward v2 Phase 1+3 已落地（0c0fbb1），本设计是 Phase 2"。
- 已做：文头记"结构先行(57 维)✅ / bp 组 12 意图 ✅（70% 帧示范，与 pp 同链同序）/ pp 组 ✅（punish、spell_finish、anti_spell、save_ace 特权精确版 + king_activate、protect_backline 先行示范）/ 消融 ✅（plan 注入 vs 置零 + 逐意图采纳探针 region/hold）"；test `test_plan_v1_layout`。
- 未做（照抄）："bait 留 backlog"（触发者"无可靠触发"、依赖"—（暂不实现）"）；"phase 不加"。
- 与主线：前置依赖、已落地。

**2 rl_reward_plan_v2.md**
- 定位："solo 自对弈（economy 默认卡组）与 run/flow 联赛共用一套 reward 语义"。
- 已做："✅ Phase 0（0515b22）：gamma 0.997 / max_ep_steps 360（终端现值修复）/ crown_weight 8"；"✅ Phase 1 + 3（本次提交）：reward v2 落地"；代码 `config.py:38-53`、`env_wrapper.py:164/181/561`，test `test_reward_v2_ledger`。
- 未做（照抄）："⏳ 僵局早停：保留未动（独立性能优化，与 reward 无耦合）；用户可选单独提交移除"；"⏳ Phase 2（plan 通道扩展）：另行详细讨论后实施"；§3.6"单位受伤 shaping（v3 备选，非首期）"、"仅当 1-3 跑完仍学不会再启用"。
- 明确不做（照抄）："❌ 给软控手工事件归因 shaping"、"❌ HP 比例折价 / '单位还能打多少血'式估值"、"❌ 单靠终端加码（不改 discount）解决躺平"、"❌ 删除全部资源 shaping 退回纯目标"、"❌ plan 塞精确动作答案"。
- 与主线：前置依赖、已落地。

**3 ai_training_plan.md**
- 定位：先知+信念+跟随者+POMDP+联赛+同刻多卡总纲；"本阶段不再继续扩展游戏引擎机制，也不再继续补卡牌内容"。
- 已做（照抄）："本文档对应的代码已按计划落地于 `src/clasher_new/rl/`"。
- 未做/未核实：§8 阶段验收（vs Random>90%、Brier/log-loss、信念校准、plan/belief dropout）与 §11.3 N=200/500 协议无落地记录。
- 与主线：前置依赖、总纲；`train_solo.py` 头注"无联赛：不建 League、不写 Elo/PFSP/league_state.json"⇒联赛/Elo 未启用，仅 hist 槽复用 `pfsp.PFSP`。

**4 P0-mechanics-plan.md**
- 定位：补全 `CREnv(agent_deck, opp_deck)` 任意合法卡组所需机制；卡组设计不由本计划负责。
- 已做（照抄）："引擎机制全部完成（族1-8 ✅，test_m2 72/72，batch_smoke 156/156）"；"当前唯一瓶颈 = 数据层（15 张 STATS_UNRESOLVED + Ronin/Vines，见 §8 数据攻坚）"。
- 未做（照抄）："族 8：待 L4：ElectroDragon_EV1 的 doAttackAction 动作组解释器（dump 内 ActionRunActionOnResolvedGameObjects 等 ClassType 已定位）；觉醒熔炉等快照外新卡待数据接入后启用"；§8"key→value 关联结构重建中 → re/official/extracted/"、"Fandom 采集（17 张缺卡 per-level）"——工作区无 `re/`。
- 与主线：前置依赖（环境侧）；§7"P3（SampleFactory APPO + v-trace）：与 P0 正交"。

**5 mcts_design.md**
- 定位："推理时浅 MCTS（RL-MCTS v1）— 方案定稿（2026-09-07）"；"零训练风险，只挂评估/推理路径"。
- 已做：`rl/mcts.py`（`RLMCTS`/`search`）；"候选动作直接复用 `legal_cells`（含 8h 空砸门、8g 不裸下门、9i EV 闸门）"；test `test_mcts_basic`、`test_mcts_defense_and_wait`。
- 未做：66 行"`rl/evaluate.py` 增加 `--mcts` 开关…输出胜率差 + 平均决策耗时…结果写 `docs/` 下评估报告"未落地（`evaluate.py` 内 mcts 引用数 0）。
- 边界（照抄）："v1 不做信念采样（POMCP 留待 v2）"；"训练不可用（符合定位）"。
- 与主线：并行支线（未接入 solo 训练/评估协议）。

**指标**：`card_registry.json` 150 条/implemented 122、STATS_UNRESOLVED 实体 2（Mirror、MergeMaiden）与 §4 文内"15 张"口径不同；PLAN_DIM 58。
