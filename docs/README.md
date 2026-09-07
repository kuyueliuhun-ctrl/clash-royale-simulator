# 文档索引

> 本文件是 `docs/` 与全仓库文档 ↔ 源码文件的导航地图。改文档前先看这里，避免重复/失联。
> 维护约定：新增 curated 文档必须在本文登记；原始采集数据（`_` 前缀文件）只被引用、不手改。

## 0. 快速入口

| 想做什么 | 去哪 |
|---|---|
| 理解模拟器/跑起来 | 根目录 `README.md`（中文）/ `readme_en.md` |
| 启动训练 | `start_rl.bat`（solo/run/flow 三模式 + 仪表盘）、`start_training.bat` |
| 训练算法代码导读 | `src/clasher_new/rl/README.md`（模块表 + 训练入口命令） |
| 跨会话方案决策（先读这个再动方案） | 根目录 `AGENTS.md` |
| 自检/回归 | `src/clasher_new/rl/selftest.py`、`scripts/test_m1..m6*.py`、`scripts/batch_smoke.py` |

## 1. Curated 文档（策划/规格/报告，已定稿可引用）

| 文档 | 内容 | 关联源文件/数据 |
|---|---|---|
| `../AGENTS.md` | **项目决策存档**：AI 工具谱系、LLM 顾问机制、MCTS 路线、外置工具①②③④、条件威胁评估、行为病理诊断 | `threat_calc.py` `simulate_exchange.py` `spell_module.py` `rl/plan_space.py` `rl/pfsp.py` |
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
| `../scripts/forensics_response.py` | 9j 响应率取证脚本（防守响应占比/延迟/落点距离，v2 修正坐标口径） | `AGENTS.md` 9j 节（结论存档） |

## 2. 外置工具（引擎侧确定性服务，2026-09 起步；设计与优先级见 `../AGENTS.md`）

| 工具 | 文件 | 语义 | 对账 selftest |
|---|---|---|---|
| ① 塔伤威胁计算器 | `../src/clasher_new/threat_calc.py` | "双方不再部署"下敌方现存部队 20s 内对我方各塔伤害 | `test_tower_threat_calc` |
| ② 交换模拟器 | `../src/clasher_new/simulate_exchange.py` | "我方现在打出这张牌"的反事实推演（none/script/fn 三档对手） | `test_simulate_exchange` |
| ③ 法术知识模块 | `../src/clasher_new/spell_module.py` | 法术引擎标定档案 + 落点估值 + 引擎对账口径 | `test_spell_module` |
| ④ 循环规划器 | 未实现 | 牌序 depth/过牌 ETA 特征（私有信息+确定性队列） | — |

## 3. 原始采集数据（`_` 前缀 / `.cdp*` / `.wikitext`，只读证据）

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
| `re/fandom_check/`、`re/fandom_stats/`（在 `.gitignore` 的 `re/` 内，不入库） | 批次校验报告与结构化 JSON | 勘误批/觉醒批 |

## 4. 数据文件（docs/ 下，curated 可引用）

| 文件 | 内容 | 消费方 |
|---|---|---|
| `card_registry.json` | 卡牌注册/覆盖状态 | `card_coverage.md`、`scripts/coverage.py` |
| `batch_smoke_report.json` | 全卡冒烟结果 | `scripts/batch_smoke.py` |
| `evolution_cycles.json` | 觉醒周期表 | `evolutions.py` |
| `leaderboard_decks.json` / `leaderboard_decks_classified.json` | 天梯 200 副卡组（推进 60/防反 120/自闭 20） | `rl/decks.py`、`rl/run_league.py`、`rl/opponents.py` |
