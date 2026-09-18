# 全项目结构优化方案（2026-09-19）

> **本文是 5 份并行盘点的合成件**（① 文件清单与规模 ② God file 内部结构 ③ 重复/死代码 ④ 分层耦合 ⑤ 测试与工具布局）。
> **本文只做"结构与文档"层面的结论，不含任何代码改动**；执行者是后续会话。
> **本文不含"重写"类建议**（用户明令，§7 第 1 条）。
> **口径**：所有数字都标注来源与复算命令；不同盘点之间**冲突处并列不调和**（§8）。

**快照**：`git d467ef8`，采集时刻 **2026-09-18T17:04:36Z**（子任务）与 **2026-09-19 复核**（本文）。

---

## 0. 一句话结论

- **规模**：199 个 `.py` / **62,402 行**（其中 `scripts/` 102 文件 / 26,494 行，`src/clasher_new/` 91 文件 / 35,565 行，`runs/` 两处产物区另计）。
- **结构病只有 4 个**：① `rl/selftest.py` 5909 行单文件承载 100 个测试 + 手工登记 `main()`；② `dashboard.py` 3173 行里 **1879 行是模块级 `_HTML` 常量字符串**；③ `battle.py`/`card_mechanics.py` 的巨型类/超多同类；④ `scripts/` 里**一次性取证脚本占 64%**（67/105）而没有目录边界。
- **语法分层是干净的**：**引擎 → `rl/` 反向 import = 0 条**（AST 全仓 + 字符串双查）。
- **真实耦合不在 import 图里，在 cwd**：`card_utils.py:4-17` 用**裸相对路径** `open('gamedata.json')` 读 5 个数据文件，而它是**全仓 importer 最多的引擎模块（20 个）** ⇒ **任何 import 都隐含要求 cwd = `src/clasher_new`**，被 wrapper 的 `os.chdir(_SRC)` 藏起来了。**这是本项目最贵的一条隐性契约。**
- **Tier 0 可以立刻做且零行为变化**：删确认死件（`re_lib.py`、`pathfinding.py`、`tmp_*.py`×5、`runs/_tmp_*`×18、`gamedata.json.bak_*`）+ 加只读结构核对器 + 落文档。**不碰** `battle.py` / `rl/config.py` / `rl/engagement.py`（并发会话占着）。

---

## 1. 现状规模表（文件 / LOC / 职责）

### 1.1 分区

| 区域 | 文件数 | LOC | 职责 | 复算命令 |
|---|---:|---:|---|---|
| `src/clasher_new/`（递归） | **91** | **35,565** | 引擎本体 + RL | `find src/clasher_new -name '*.py'` |
| ├ 顶层（引擎本体） | 34 | 9,997 | `battle`/`card_mechanics`/`card_utils`/`player`/`arena`/`core`/`spell_module`/`threat_calc`/`pathfinding*`… | `find -maxdepth 1` |
| ├ `rl/` | 41 | 23,869 | 训练/评估/对手池/信念/计划/仪表盘/selftest | `find src/clasher_new/rl -name '*.py'` |
| ├ `client_side/` | 5 | 581 | 逆向客户端侧资源（47 `.png`+6 `.json`+5 `.py`，9.9 M） | |
| └ `runs/`（gitignored 产物区） | 11 | 1,118 | 一次性探针 + 缓存 `.npz`（1,622 文件 / 6.7 G） | |
| `scripts/` | **102** | **26,494** | 仪器/探针/判读/测试 wrapper | `find scripts -name '*.py'` |
| ├ 顶层 | 91 | 26,407 | | |
| └ `rl/`（wrapper） | 11 | 87 | `chdir(_SRC)` + `runpy` / `argparse` 转发 | |
| 仓库根 `runs/`（第二产物区） | 3 | 279 | `forensics_100k.py`/`watch_100k.py`/`analyze_curve.py`（2.1 G） | |
| 仓库根 `.py` | 1 | 0 | `re_lib.py`（**空文件，已 tracked**） | |
| `ideas/` + `.tmp/` | 2 | 64 | | |
| **合计**（排除 `.venv`/`clash-royale-simulator-main.venv`/`.git`/`site-packages`/`__pycache__`） | **199** | **62,402** | | `wc -l` 全量 |

> **不含 `runs/*.py` 时为 196 文件 / 62,123 行**（`runs/` 3 文件 279 行）。子任务③用过 **196**（scope）与 **202**（with-runs）两个口径，④ 用过 **196**；差异只来自 gitignored 临时件 —— 见 §8-C1。

### 1.2 Top 40（LOC 降序，2026-09-19 实测）

| # | 路径 | LOC | 层 | # | 路径 | LOC | 层 |
|---:|---|---:|---|---:|---|---:|---|
| 1 | `src/clasher_new/rl/selftest.py` | 5,909 | 测试 | 21 | `scripts/_survey_merge.py` | 566 | 工具(`_`) |
| 2 | `src/clasher_new/battle.py` | 3,457 | 引擎 | 22 | `src/clasher_new/rl/evaluate.py` | 557 | rl |
| 3 | `src/clasher_new/rl/dashboard.py` | 3,173 | rl | 23 | `scripts/selftest_offline_engagement_trade.py` | 552 | 测试 |
| 4 | `src/clasher_new/card_mechanics.py` | 1,895 | 引擎 | 24 | `src/clasher_new/rl/ppo.py` | 547 | rl⚠️R2 |
| 5 | `src/clasher_new/rl/run_league.py` | 1,787 | rl | 25 | `src/clasher_new/card_aliases.py` | 540 | 引擎 |
| 6 | `src/clasher_new/rl/train_solo.py` | 1,748 | rl | 26 | `scripts/probe_channel_gradient.py` | 533 | 工具 |
| 7 | `scripts/offline_engagement_trade.py` | 1,252 | 工具 | 27 | `src/clasher_new/card_utils.py` | 522 | 引擎⚠️cwd |
| 8 | `scripts/probe_explore_randomization.py` | 982 | 工具 | 28 | `scripts/test_m6_elite.py` | 521 | 测试 |
| 9 | `scripts/et_solo100k_readout.py` | 948 | 工具✅ | 29 | `src/clasher_new/rl/config.py` | 520 | rl⚠️R7 |
| 10 | `scripts/probe_value_ln.py` | 918 | 工具 | 30 | `src/clasher_new/rl/prophet.py` | 516 | rl |
| 11 | `src/clasher_new/rl/belief_planner.py` | 891 | rl | 31 | `scripts/test_m3_evo.py` | 488 | 测试 |
| 12 | `scripts/diag_critic_ev.py` | 854 | 工具 | 32 | `scripts/test_m2.py` | 488 | 测试 |
| 13 | `src/clasher_new/rl/env_wrapper.py` | 811 | rl⚠️R13 | 33 | `src/clasher_new/rl/mcts.py` | 480 | rl |
| 14 | `src/clasher_new/rl/follower.py` | 806 | rl⚠️R2 | 34 | `scripts/forensics_card_usage.py` | 479 | 工具 |
| 15 | `scripts/s1_plan_gate.py` | 793 | 工具 | 35 | `scripts/probe_hold_recompute.py` | 429 | 工具 |
| 16 | `scripts/pomdp_ceiling_probe.py` | 726 | 工具 | 36 | `scripts/probe_pass_prob.py` | 408 | 工具 |
| 17 | `scripts/duel_search.py` | 636 | 工具 | 37 | `scripts/check_dashboard_js.py` | 404 | 工具✅ |
| 18 | `src/clasher_new/rl/flow_league.py` | 613 | rl | 38 | `scripts/probe_pathfix.py` | 392 | 工具 |
| 19 | `scripts/phi_offline_check.py` | 584 | 工具 | 39 | `src/clasher_new/rl/belief.py` | 362 | rl |
| 20 | `src/clasher_new/rl/action_mask.py` | 573 | rl⚠️R13 | 40 | `scripts/coverage.py` | 360 | 工具✅ |

✅ = 自带 `--selftest`；⚠️R2/R7/R13 = 红线保护区（§7）。

### 1.3 非源码产物混杂（`src/clasher_new/` 下）

| 条目 | 数量 | 体积 | 处置 |
|---|---:|---:|---|
| `runs/` | 1,622 文件（782 `.pt` / 587 `.pkl` / 232 `.json` / 11 `.py`） | **6.7 G** | **保留**（读数数据源，§7 拒绝项 5） |
| 顶层散落 `*.pt` | **16** | 37 M | 候选归拢（Tier 3） |
| 顶层 `*.json` | **7** | 2.3 M | `card_utils` **裸相对路径**消费 ⇒ 不能随便动（Tier 3） |
| `gamedata.json.bak_ronin_vines` | 1 | 341.6 KB | **删**（Tier 0，gitignored） |
| `client_side/` | 58 | 9.9 M | 保留 |
| `data_official/` | 7 | 3.4 M | 保留 |
| `__pycache__/` | — | 1.1 M | gitignored |
| `hook_raw_capture.js` / `tilemap_lane_grid.txt` | 2 | 12 K | 保留（逆向参考） |

---

## 2. God file 清单 + 内部簇（可拆边界）

### 2.0 方法与"两个家族"

**行数一律用 AST `end_lineno` 精算**，不用"扫到下一个 `^def`/`^class` 为止"的粗算法 —— 后者会把**模块级常量**和**并列语句**并进上一个函数（§8-C2 的冲突来源）。

| 家族 | 特征 | 文件 | 拆分性质 |
|---|---|---|---|
| **A 多职责混杂** | 训练/评估/UI/数据/CLI 并存，簇边界 ≈ 职责边界 | `dashboard` / `run_league` / `train_solo` | **可按职责切** |
| **B 单一巨型类/超多同类** | 职责单一但粒度过大 | `battle` / `selftest` / `card_mechanics` / `belief_planner` / `env_wrapper` / `follower` | **只能按"类/族"切，语义面宽** |

**模块级 vs 可调用体占比**（AST）：

| 文件 | 总行 | def/class 内 | 不在 def/class 内 | 最大单块 |
|---|---:|---:|---:|---|
| `rl/dashboard.py` | 3173 | 1088 (34.3%) | **2086 (65.7%)** | `_HTML` L865-2743 = **1879** |
| `rl/selftest.py` | 5909 | 5657 (95.7%) | 253 | 100 个 `test_*` |
| `card_mechanics.py` | 1895 | 1731 (91.3%) | 165 | `MegaKnight` L509-636 = 128 |
| `battle.py` | 3457 | 3374 (97.6%) | 84 | `BattleState` L2624-3455 = **832** |
| `rl/run_league.py` | 1787 | 1579 (88.3%) | 209 | `main` L1554-1783 = **230** |
| `rl/train_solo.py` | 1748 | 1609 (92.0%) | 140 | `run_solo` L1042-1743 = **702** |

### 2.1 完整清单（`> 700` 行，共 **9 个**，递归确认无第 10 个）

| # | 文件 | 行数 | 顶层 def | 类 | 最大实体 | 家族 |
|---|---|---:|---:|---:|---|---|
| 1 | `rl/selftest.py` | **5,909** | 105 | 1 | 100 × `test_*` | B |
| 2 | `battle.py` | **3,457** | 10 | 18 | `BattleState` 832 | B |
| 3 | `rl/dashboard.py` | **3,173** | 29 | 1 | `_HTML` 1,879 | **A** |
| 4 | `card_mechanics.py` | **1,895** | 4 | **58** | `MegaKnight` 128 | B |
| 5 | `rl/run_league.py` | **1,787** | 43 | 2 | `main` 230 | **A** |
| 6 | `rl/train_solo.py` | **1,748** | 17 | 1 | `run_solo` **702** | **A** |
| 7 | `rl/belief_planner.py` | 891 | 22 | 1 | `BeliefPlanner` 509 | B |
| 8 | `rl/env_wrapper.py` | 811 | 8 | 2 | `RLEnv` 514 | B |
| 9 | `rl/follower.py` | 806 | 2 | 1 | `FollowerPolicy` 653 | B |

### 2.2 `rl/selftest.py`（5909）— 20 簇连续覆盖 L1-5909

**头部事实（决定可拆性的关键）**：模块级 import **仅 7 行**（L23-27 stdlib + L33 `numpy` + L35 `from card_utils import Card`）；**函数内局部 import 471 行**；**全文件唯一模块级全局 = `_PARENT`（L29）**。⇒ 刻意不在模块级拖 torch/引擎，这是 `scripts/run_selftests.py:42` 的 `import rl.selftest as st` 能秒级完成的前提。

| 行号 | 簇 | 职能 | 共享依赖 |
|---|---|---|---|
| L1-37 | 文件头/入口契约 | docstring 回归索引 + 7 行 module import + `_PARENT` | 唯一全局 `_PARENT` |
| L38-130 | ActionBundle 原子/技能 + 贝叶斯基线 | | **定义 `_make_policy_and_tokens`(L123)** |
| L131-328 | P0/P1 评审回归团 | replay 一致/熵符号/掩码双边/启发式对手/exploiter ckpt | 用 `_make_policy_and_tokens`@139,162,254,313；`_mk_env`@253,312 |
| L329-398 | bundle 上限 / replay 往返 / prophet | | **定义 `_mk_env`(L394)** |
| L399-640 | 卡池与联赛基础 | 随机卡组/Elo/PFSP 独立/噪声地板 | 嵌套 `ref_ema`(476)/`noise_prob`(542) |
| L641-956 | 奖励配置与 economy 标定 | 权重/覆盖/等级不变性/圣水差/塔兵参考 | ⚠️ **写跨模块全局 `Card.default_level`@L894** |
| L957-1249 | 联赛持久化 + dashboard 数据面 | resume/replays/deck_pool/payload | **调 `_FakeCfg`@1214（定义 L1250，晚绑定）** |
| L1250-1508 | dashboard 卡牌统计 + 训练基础设施 | clone/CUDA/belief+follower+ppo+league/并行等价 | **`class _FakeCfg`(L1250)；调 `_intents`@1424（定义 L1509）** |
| L1509-1678 | flow 联赛 | smoke/消融落盘/sweep/resume | **定义 `_intents`(L1509)** |
| L1679-1880 | solo/human/stall 通路 | smoke/human_play/resume/stall/play_pair/早停 | `make_fake`(1784) |
| L1881-2013 | 平局惩罚 + reward v2 账本 | | ⚠️ **写 cwd 相对 `runs/_tmp_drawtest`@1926，自清@1942** |
| L2014-2288 | 法术 EV 闸门 + 不裸下 + 坦克后排几何 | | 局部 `mini_mask_slot`/`mk_battle`/`spawn`/`cell_near` |
| L2289-2674 | 计划空间 + Phase2 新意图规则 | | 局部 `new_battle`/`place`/`set_hand`/`belief`/`pstate`；**含最长 2 个测试** |
| L2675-2949 | 贝叶斯队列锁 + 评估并行 + 加时/平局 | | `run_league.overtime_open`/`timeout_winner`/`settle_stall` |
| L2950-3440 | 引擎侧分析工具对账 | threat_calc/simulate_exchange/spell_module/MCTS×2 | `fresh`/`opp_fn`/`snap`/`bkey`/`plan_of` |
| L3441-3820 | 战斗机制回归 | 死亡伤害缩放/藤蔓/滚木方向/MK 落地伤 | `train_solo.behavioral_metrics` |
| L3821-4411 | 对手池 / PFSP / 价值通道 | | **定义 `_tiny_rollout_transitions`(L3972)**；⚠️ **读仓库磁盘 `glob("runs/economy/solo_main_*.pt")`@3933 + `glob("../../runs/archive/*/…")`@3934，无产物 `[SKIP]`@3936/@4266** |
| L4412-4873 | 网络结构回归 | enc_ln 解冻/bypass/stall/独立 value encoder/PPO 多 epoch | **调 `_tiny_rollout_transitions`@4788** |
| L4874-5310 | solo 锚点 / 评估调度 / adv 惰性探针 | | ⚠️ **`inspect.getsource(run_league.main)`@4950,@5096；`eval_round_robin`@5188** |
| L5311-5909 | 精确塔伤对账 + 数据/签名不变量 + `main()` | | ⚠️ **`ast.parse` 扫 `battle.py`+`rl/*.py`@5508，root 由 `__file__` 推@5507/@5568**；**`main()` L5801-5905** |

**最长 5 个函数**：`test_bp_new_intent_rules` L2362-2537 (**176**) / `test_precise_threat` L5311-5484 (174) / `test_pp_new_intent_rules` L2540-2672 (133) / `test_enc_layernorm_gru_vitality` L4412-4542 (131) / `test_eval_scheduler_two_tier` L4997-5121 (125)。

#### ★ 四条硬约束（拆分的真实成本）

| # | 约束 | 精确位置 | 要求 |
|---|---|---|---|
| 1 | 按名 + `dir()` 反射 | `scripts/run_selftests.py:36`（`_names()`）+ **`:42` `import rl.selftest as st`** + `:60` `getattr(st,n)()` | 所有 `test_*` 必须是**模块级函数**、**import 时无重副作用** |
| 2 | 整文件 runpy | `scripts/rl/selftest.py:5`（`os.chdir(_SRC)`）+ **`:7` `runpy.run_path(<SRC>/rl/selftest.py, run_name="__main__")`** | 必须保留 `main()` + 末尾 `if __name__` |
| 3 | `main()` 100 行显式清单 | `selftest.py:5805-5904`（`print("ALL SELFTESTS PASSED")`@5904） | 与 100 个 `def test_*` **双向差集为空** |
| 4 | **外部按名硬编码** | **`scripts/_apply_s2_channel_when_idle.sh:70-71`**：`test_noncombat_entity_contract test_replay_roundtrip test_league_replays test_dashboard_replays` | 这 4 个名字是 S2 落地脚本的验证 2/3 步 |

> 生产代码**零 import** selftest（grep 只命中 `run_selftests.py:42`；`src/` 下全是 docstring/注释）。**无 pytest/CI**（`conftest.py`/`pytest.ini`/`setup.cfg`/`tox.ini`/`pyproject.toml` 实测全不存在）。

#### 可拆边界（判据：不依赖 4 个共享 helper、不做 `__file__` 根推导、不写仓库路径）

| 簇 | 行号 | 可独立抽离？ | 阻塞点 |
|---|---|---|---|
| 法术 EV 闸门 + 坦克后排几何 | L2014-2288 | ✅ | 无 |
| 计划空间 + Phase2 新意图 | L2289-2674 | ✅ | 无 |
| 贝叶斯队列锁 + 评估并行 + 加时平局 | L2675-2949 | ✅ | 无 |
| 引擎侧分析工具对账 | L2950-3440 | ✅ | 无 |
| 战斗机制回归 | L3441-3820 | ✅ | 依赖 `train_solo.behavioral_metrics`（模块级 import 即可） |
| 网络结构回归 | L4412-4873 | ⚠️ | 需连同 `_tiny_rollout_transitions`(L3972) 一起搬 |
| 对手池 / PFSP / 价值通道 | L3821-4411 | ⚠️ | 读仓库磁盘 glob ⇒ 搬走后相对路径基准变 |
| P0/P1 评审回归团 | L131-328 | ⚠️ | 需连同 `_make_policy_and_tokens`(L123)+`_mk_env`(L394) 一起搬 |
| 联赛持久化 ↔ dashboard 统计 | L957-1249 ↔ L1250-1508 | ❌ 必须成对 | `_FakeCfg` 晚绑定（L1214 调用 / L1250 定义） |
| 精确塔伤对账 + 签名不变量 | L5311-5909 | ❌ | `__file__` 推 root（L5507/5568）+ `main()` 义务 |

> ⇒ 本文实测 **5 簇无阻塞、2 簇需带 helper、1 簇需带 2 个 helper**；子任务②原文写"**8 簇**"（其判定表在传递中被截断）。**两说并列，执行前必须重数**（§8-C3）。

### 2.3 `rl/dashboard.py`（3173）— 家族 A，UI 是内联常量

**关键更正**：`make_demo_replays` = **50 行**（L809-858），**不是** 1937 行；L865-2743 是模块级常量 `_HTML = r"""…"""`（**1879 行 / 80,974 字符，无模板引擎**）。

| 行号 | 簇 | 职能 |
|---|---|---|
| L1-34 | docstring | |
| L47-91 | `_PARENT` / `_REPO_ROOT` / `MODEL_COLORS` / `MODEL_LABELS` / 缓存 | |
| L94-262 | state / league 元信息 / 胜率曲线 / `build_payload` | 联赛数据面 |
| L273-326 | sweep 扫描 + payload | |
| L336-362 | **`build_health_payload`**（训练健康，读 `--train-log`） | |
| L369-515 | solo payload / replay 扫描 / `load_replay_payload` | 回放数据面 |
| L532-806 | 卡牌统计（`_stat_file_cards` 89 + `build_card_stats_payload` 108） | |
| L809-858 | `make_demo_replays`（--demo 合成数据） | |
| **L865-2743** | **`_HTML` 常量（1879 行，内联 HTML+JS+CSS）** | ★ 最大单块 |
| L2746-2901 | `class Handler`（156 行，`/api/*` **10 条**） | HTTP |
| L2904-3046 | `make_demo_state/sweep/solo` + `_force_utf8_stdout`(L3033) | |
| L3049-3169 | `main`（121 行，argparse/端口） | CLI |

**可拆边界**：① `_HTML` → 独立模块（**逐字节常量搬移**，Tier 2）；② 数据面函数（L94-806，共 ~20 个纯函数）→ `rl/dashboard_data.py`；③ `Handler` + `main` 留原地。**测试**：selftest 里 5 个（`test_dashboard_replays` L1020 / `test_dashboard_league_payload` L1138 / `test_dashboard_card_stats` L1257 …）+ 独立前端回归 `scripts/check_dashboard_js.py`。

### 2.4 `rl/run_league.py`（1787）— 家族 A

| 行号 | 簇 | 职能 |
|---|---|---|
| L1-155 | docstring / `_PARENT` / `_cuda_hint` / `resolve_device` / **`STALL_WINDOW`/`STALL_LIMIT`(L154-155)** | ⚠️R7 常量 |
| L109-274 | `LeagueGameRecorder` / 塔血度量 / `timeout_winner` / `settle_stall` / `_stall_probe` | 判定语义（⚠️R13 邻域） |
| L277-435 | 单局执行（side0 / scripted / bundle / deck factory / opp / seed offset） | |
| L438-578 | `_play_one_game` / `_set_et_measure`(L478) / `_eval_env` / `_play_pair_games` / `play_pair` / policy spec | |
| L581-759 | 并行评估（`_eval_pair_worker_main` / `_run_eval_pairs_parallel` / `_round_estimates` / **`eval_round_robin`**） | |
| L766-921 | 联赛构建 / 五 agent / 状态存取 | |
| L924-1056 | **`class _EvalScheduler`(L924-1009)** / 对手采样 / `_build_league` | 两级评估调度 |
| L1063-1311 | `_run_single`(107) / `_run_vec`(136) | 训练主循环 |
| L1318-1519 | **`_run_mp`(202)** | 多进程 |
| L1522-1551 | `run_league`(14) / **`_force_utf8_stdout`(L1538)** | |
| L1554-1783 | **`main`(230)** | CLI |

**可拆边界**：① 判定语义（L109-274）→ `rl/league_rules.py`；② 评估并行（L581-759）→ `rl/league_eval.py`；③ 训练循环三条（`_run_single`/`_run_vec`/`_run_mp`）→ `rl/league_train.py`。**`main` 只做 argparse + 分派**。

### 2.5 `rl/train_solo.py`（1748）— 家族 A

**关键更正**：`_new_episode_reset` = **27 行**（L1516-1542），**不是** 234 行；L1544 的 `for step` 是并列语句；`run_solo` 到 L1743 结束（**702 行**）。

| 行号 | 簇 | 职能 |
|---|---|---|
| L1-155 | docstring / `_print_safe` / `DEFAULT_SOLO_DECK`(L56-57) / 锚点常量(L67-69) / `_SOLO_PROPHET_PROB`(L129) / **`_OPP_MIX`(L145)** / `_HIST_POOL_MAX` / **`_PFSP_ALPHA`/`_PFSP_GATE_HI`/`_PFSP_GATE_PENALTY`(L152-154)** | ⚠️R7 常量 |
| L72-126 | `_rand_anchor_warns` / `_make_rand_anchor` / `resolve_deck_set` | |
| L157-317 | `solo_env` / `_draw_penalty` / frozen 同步 / hist ckpt 收集 / 去重 / **`_check_gates`(73)** / 报告 | 门禁 |
| L321-510 | **`class _OpponentPool`(190)** | 对手池 |
| L513-710 | `write_solo_state` / **`behavioral_metrics`(161)** | 行为指标 |
| L713-1039 | `eval_solo` / `_eval_worker_main`(113) / `_collect_worker_results` / `eval_solo_parallel`(97) | 评估 |
| L1042-1743 | **`run_solo`(702)** | 训练主循环 |

**可拆边界**：① 门禁（L157-317）→ `rl/solo_gates.py`；② 对手池 + 锚点（L72-126 + L321-510）→ `rl/opponent_pool.py`；③ 评估（L513-1039）→ `rl/solo_eval.py`；④ `run_solo` 内部按"收集→更新→评估→写状态"抽 4 个函数（**禁止改默认参数与调用顺序**，【R2】）。

### 2.6 `battle.py`（3457）— 家族 B，18 个类

| 行号 | 实体 | 行数 | 行号 | 实体 | 行数 |
|---|---|---:|---|---|---:|
| L43-819 | **`Entity`** | **777** | L2021-2267 | `EvoEffectZone`/`HealAuraZone`/`VinesSnareZone` | 62/54/95 |
| L822-1293 | **`Troop`** | **472** | L2270-2486 | `DeathSlowZone`/`EvoZapZone`/`TimedExplosive` | 37/47/24 |
| L1297-1479 | `Building` | 183 | L2495-2607 | `apply_hero_overlay`/`_barb_log_reroll_effect`/`IceGolemiteSnowZone` | 29/42/38 |
| L1481-1721 | `Projectile` + `_ProjectileShim` | 224+15 | L2610-2621 | `get_spawn_position` | 12 |
| L1724-1960 | `SpawnProjectile`/`AreaEffect`/`GenericBomb` + shims | 47/154/51 | **L2624-3455** | **`BattleState`** | **832** |

**可拆边界（全部高风险）**：`Entity`/`Troop`/`Building`/`Projectile`/`AreaEffect` 五簇 + `BattleState`。⚠️ **【R13】掩码/合法格判定依赖 `BattleState` 的部署语义**（`battle.py:2702` 部署入口、`ai_*`/合法格查询），`card_mechanics.py` 又反向 `import battle` ⇒ 拆 `BattleState` 会同时动 `Entity` 的注册表与卡牌技能挂钩。**见 Tier 3 + §7。**

### 2.7 `card_mechanics.py`（1895）— 家族 B，**58 个类**

| 族 | 类（行号） |
|---|---|
| 基础兵种 | `Ghost`(L8) / `Witch`(66) / `Balloon`(89) / `Golem`(95) / `LavaHound`(105) / `Prince`(113) / `DarkPrince`(155) / `BattleRam`(158) / `GiantSkeleton`(166) |
| 法术/辅助 | `IceWizard`(175) / `Miner`(202) / `Rage`(218) / `RageBarbarian`(261) / `Fisherman`(269) |
| `_HeroBase` 系 | `SkeletonKing`(319) / `ArcherQueen`(355) / `GoldenKnight`(375) / `Monk`(416) / `MightyMiner`(435) / `LittlePrince`(461) / `BossBandit`(477) / **`MegaKnight`(509-636, 最大 128)** |
| 精英/新卡 | `Musketeer`(639) / `Ronin`(669) / `Assassin`(715) / `BattleHealer`(810) |
| Hero 变体（~25 个） | `HeroKnight`(900) … `HeroIceGolemite`(1542) |
| 2025 新族 | `ElectroGiant`(1584) / `_AttackStunMixin`(1603) / `ElectroWizard`(1613) / `MiniSparkys`(1626) / `ElectroSpirit`(1630) / `RamRider`(1662) / `MovingCannon`(1672) / `Phoenix`(1692) / `PhoenixEgg`(1715) / `ThreeMusketeers`(1738) / `GoblinGiant`(1761) |
| 王塔 | `King_KnifeTowers`(1807) / `King_ChefTowers`(1842) |

**可拆边界**：按上表 7 族拆为 `card_mechanics/` 包 + `__init__.py` re-export；**必须保持"类名字符串查表"不变**（`battle.py` 按 `Card(...)` 名动态取类）。**测试**：`scripts/test_m3_evo.py` / `scripts/test_m6_elite.py` / `scripts/coverage.py`（卡牌覆盖率）。

### 2.8 三个 500-650 行单类文件（家族 B，不另展开）

| 文件 | 行数 | 支配实体 | 可拆边界 |
|---|---:|---|---|
| `rl/belief_planner.py` | 891 | `BeliefPlanner` L383-891 (**509**) + 22 个模块级 helper（L118-380） | helper 已是"预拆"形态；`PRESSURE_THRESHOLD`(L51) ⚠️R7 |
| `rl/env_wrapper.py` | 811 | `RLEnv` L290-803 (**514**) + 奖励函数（`compute_reward` L182-264 + `_per_tower_norm_dmg` L132-162） | 奖励簇可独立成 `rl/reward.py`；⚠️R13（掩码）/R7（奖励常量） |
| `rl/follower.py` | 806 | `FollowerPolicy` L154-806 (**653**) + `save/load_checkpoint`(L55-151) | ⚠️R2（策略默认/`stop_logit_bias`）；ckpt 存取可独立 |

---

## 3. 重复 / 死代码清单

> **口径**：`S` = `--include=*.py`，排除 `.git`/`.venv`/`clash-royale-simulator-main.venv`/`site-packages`/`node_modules`/`__pycache__`/`runs`。
> **scope** = 排除 `runs/` 与 gitignored 临时件；**with-runs** = 含之。两者差异只来自 gitignored 件（§8-C4）。

### 3.1 重复清单（逐条：文件:行号 + 次数 + 建议 + 风险）

#### D1 `reconfigure(encoding=…)` 样板 —— **scope 66 / with-runs 77**

| 变体 | 次数 | 代表位置 |
|---|---:|---|
| `reconfigure(encoding="utf-8")`（无 `errors=`） | **41** | `scripts/check_commit.py:20` |
| `reconfigure(encoding="utf-8", errors="replace")` | **23** | `scripts/run_selftests.py:30` |
| 单引号 `encoding='utf-8'` | 2 | `scripts/probe_value_ln.py:33,118` |
| **无 `try` 裸调用** | **11** | — |
| 唯一非代码行（docstring 示例） | 1 | `scripts/probe_explore_randomization.py:43` |

**同一语义三套写法**：

| 写法 | 定义点 | 次数 |
|---|---|---:|
| ① 具名 helper `def _force_utf8_stdout()` | `rl/run_league.py:1538` / `rl/dashboard.py:3033` / `scripts/probe_value_ln.py:116` | **3 次定义（函数体逐字相同）** |
| ② `for _s in ("stdout","stderr"): getattr(sys,_s).reconfigure(...)` | `scripts/judge_anchor_blocks.py:35-38` / `scripts/summarize_solo_run.py:29-33` | 2 |
| ③ `for _s in (sys.stdout, sys.stderr): _s.reconfigure(...)` | `scripts/health_curve.py:46-50` | 1 |
| ④ 内联 `try/except` | 其余 ~59 处 | ~59 |

**消费方 4 处**：`run_league.py:1555` / `dashboard.py:3050` / `scripts/run_selftests.py:40` / `selftest.py:5803`。

- **建议**：新增 `rl/io_bootstrap.py::force_utf8_stdout()`，3 处具名 helper 改为 `from rl.io_bootstrap import force_utf8_stdout as _force_utf8_stdout`（**保留旧名 re-export**），随后按批把 ④ 内联收敛到同一函数。
- **风险**：中低。`run_selftests.py:40` 从 `rl.run_league` 取该名 ⇒ **shim 必须保留**；`src/clasher_new/rl/selftest.py` 在**模块顶层之外**调用（L5803，函数内）⇒ 无 import 副作用。**GBK 陷阱是既存实测坑**（`PYTHONIOENCODING=utf-8` 在本环境实测不生效，见 `docs/train_health_metrics_2026-09-18.md`）⇒ 收敛后必须留这条兜底。

#### D2 `sys.path.insert` 样板 —— **122 行 / 116 文件（with-runs 124）**；`sys.path.append` **0 次**

| 形态 | 数量 | 说明 |
|---|---:|---|
| ① 四行守卫 `if <VAR> not in sys.path: sys.path.insert(0, <VAR>)` | **69 文件**（行级 74） | `<VAR>` ∈ `{_SRC,_PARENT,_HERE,TREE,SCRIPTS,SRC}` |
| ② 无守卫 `sys.path.insert(0, _X)` | 19 | `scripts/_mask_ab_prefix.py:88` / `scripts/_mask_diff_snapshot.py:14` / `scripts/_schema5_probe.py:11` / `scripts/rl/*.py:6`(11 个) / `scripts/s2_neutrality_probe.py:13` / `scripts/s2_trade_probe.py:20,23` / `scripts/summarize_solo_run.py:37` / `scripts/test_m1.py:8` |
| ③ **硬编码绝对 Windows 路径** | **5** | `scripts/assassin_left_bridge_test.py:20` / `assassin_vs_megaknight.py:20` / `assassin_vs_sparky.py:25` / `duel_search.py:35` / `question_bank_poc.py:27` —— 全部 `E:/clash-royale-simulator-main/src/clasher_new` |
| ④ `os.getcwd()` | 2 | `scripts/analyze_online_trade.py:33` / `scripts/_mask_vs_engine_reconcile.py:26` |
| ⑤ 内联 `os.path.join(...)` | 27 | 其余 |

- **建议**：③ 改为 `__file__` 推导（**唯一在换机时会真失效的一类**）；①②④⑤ 收敛到 `scripts/_bootstrap.py`，按批进行。
- **风险**：中。③ 在本机（Linux 容器）**无法端到端验证**，只能在 Windows 侧跑。①② 面太大（116 文件），一次性改会产生巨大 diff ⇒ **必须按目录分批**。

#### D3 `_survey_*.py` 同族脚本 **9 个**（`scripts/_survey_{audit,brief,groups,inventory,md_to_docx,merge,merge_docs,reverse_check,verify}.py`）

- **建议**：全部迁入 `scripts/survey/`，保留 `docs/full_code_reference.md` 的生成入口 `scripts/survey/_survey_merge.py --out docs/full_code_reference.md`。
- **风险**：低-中。`_survey_merge.py:4,15,127` 与 `_survey_merge_docs.py:49,72,254,290` 互相写路径；`_survey_reverse_check.py` 把 `full_code_reference.md` 当抽检基准 ⇒ **移动脚本要同步改文档里的命令串（【R18】）**。

#### D4 `dashboard.py` 的 `_HTML`（1879 行内联 HTML/JS/CSS）

- **不是"重复"，是"内联"** —— 但它是**同一份 UI 的唯一副本**，任何 `dashboard` 改动都与 1879 行字符串耦合。
- **建议**：抽成 `rl/dashboard_html.py::HTML`；**逐字节常量搬移**。
- **风险**：中。字符串被 `Handler` 使用；必须 sha256 比对 + `scripts/check_dashboard_js.py` 回归。

#### D5 selftest 内 **471 行函数内 import**（`import battle as battle_mod` 出现 ~40 处等）

- **建议**：**不动**。这是刻意的"零模块级重依赖"设计（保证 `import rl.selftest` 秒级），改成顶层 import 会让 `scripts/run_selftests.py:42` 变慢并可能引入副作用。

### 3.2 死代码 / 残留清单（逐条：路径 + 依据 + 建议 + 风险）

| # | 路径 | 大小 | 判定依据（实测） | 建议 | 风险 |
|---|---|---:|---|---|---|
| X1 | `re_lib.py`（仓库根） | **0 B** | `.py` 全仓 `re_lib` **0 命中**（src+scripts） | **删** | 极低 |
| X2 | `src/clasher_new/pathfinding.py` | — | `import pathfinding\b` / `from pathfinding import` **全仓 0 命中**（与 `pathfinding_heap.py` 是不同模块） | **删**（先 `git log --follow` 确认非对照参考实现） | 低 |
| X3 | `src/clasher_new/tmp_*.py` ×5 | 11.6 KB | `.gitignore:53`=`tmp_*.py`（`git check-ignore` 命中）；全仓 import **0 命中**；唯一提及是自动生成的 `docs/full_code_reference.md:161` | **删** | 低（人类可能仍要看） |
| X4 | `src/clasher_new/runs/_tmp_*` ×18（+`.out`） | ~340 KB | `runs/` 被 `.gitignore:22` 整目录忽略；全树 0 import；时间戳 Sep 14 一次性 value-path 草稿 | **删/归档 `runs/_scratch/`** | 极低 |
| X5 | `src/clasher_new/gamedata.json.bak_ronin_vines` | **341.6 KB** | `.gitignore:54`=`*.bak_*`（命中）；引擎只读 `gamedata.json` | **删** | 极低 |
| X6 | 仓库根 `_tmp_fl_meta/params/params2.py` | 72/81/83 行 | **已被并发会话删除**（17:03Z→17:04:36Z）；`docs/cmp_manual_2026-09-19.md:1240` 原文明写"仓库根，**可删**" | **无需动作**（已完成） | — |
| X7 | 仓库根 `1`（0 B） / `.p6.png`（456 K） / `0631.pdf`（608 K） | — | 名字像误产生的占位/粘贴产物；`.gitignore` 里有对应条目（`1` / `.p6.png` / `0631.pdf`） | 确认后删 | 低 |
| X8 | `src/clasher_new/runs/_tmp_probe_path.out` | 3,081 B | 配对的 `.py` 已不在 | 删 | 极低 |
| X9 | `hero_ability_extract.json` | 4,925 B | **全仓 Python 0 处消费**（只有 `scripts/coverage.py:240` 的字符串 `'hero_ability: '`） | 标注为"未被代码消费的数据"；**不删**（可能是待接入） | — |
| X10 | `scripts/stop_solo_training_2200.py` + `scripts/stop_solo_2200.log` | 12 KB | 一次性停跑脚本 + 日志（`.log` 已被 gitignore） | 归档或删 | 低 |
| X11 | `checkpoint`/`*_copy*` 类 | **0 命中** | 全树 `find` | 无动作 | — |

---

## 4. 分层耦合问题（谁越层 import 谁）

> 边定义：`A -> B` = A 中至少一条 `import` 引用 B；同名多次 import 记 1 条边。**AST 静态提取**，未运行训练。

### 4.1 一句话：语法层干净，隐蔽层很脏

- **引擎 → `rl/` 反向 import = 0 条**（AST 全仓 + 字符串双查；`card_aliases.py:34` 的 `clasher_new` 只是注释）。
- `src/clasher_new/__init__.py` 是空文件；`rl/__init__.py` 只做 `sys.path` 注入 + docstring。
- **但 `rl/` → 引擎是"平铺 + 直插深层内部"**：28 个有本地 import 的 rl 模块中 **14 个直接 import 引擎**；无中间门面。
- **`rl/` 内 429/572 条 import 写在函数体内（75%）**（排除 `selftest.py` 后仍有 58 条），其中至少 3 处是**显式避免循环依赖的惰性 import**。

### 4.2 `rl/` 直插引擎"深层内部"清单（文件:行号 + 原样语句）

**`battle`（4 个 rl 模块；3 处为模块顶层）**

| 位置 | 原样语句 | 性质 |
|---|---|---|
| `rl/env_wrapper.py:26` | `import battle` | **模块顶层** |
| `rl/replay.py:117` | `from battle import Building, Projectile, SpawnProjectile, AreaEffect, TimedExplosive` | 函数体内 |
| `rl/threat_precise.py:341` | `import battle as bm` | 函数体内 |
| `rl/selftest.py`（~40 处：102/101/2017/2364/3097/5314/5622/5694…） | `import battle as battle_mod` / `from battle import Troop` / `from battle import is_death_bomb` / `from battle import normalize_dir` / `from battle import VinesSnareZone` … | 全部函数体内 |

**`spell_module`（4 个 rl 模块，全部函数体内 —— 已是惰性隔离）**：`rl/action_mask.py:131` / `rl/belief_planner.py:310` / `rl/mcts.py:262` / `rl/selftest.py:3097,3100`。

**`simulate_exchange`（2 处，均函数体内）**：`rl/opponents.py:157` / `rl/selftest.py:3010`。
**`threat_calc`（2 处）**：`rl/threat_precise.py:178`（`from threat_calc import _hostiles_present  # 单一来源：与工具同一守卫`）、`rl/selftest.py:5318`。

**其他顶层深插**：

| 位置 | 原样语句 |
|---|---|
| `rl/env_wrapper.py:27` | `import player` |
| `rl/env_wrapper.py:28` | `from card_utils import Card` |
| `rl/mcts.py:30/31/32` | `from core import Position` / `from card_utils import Card` / `from player import PlayerState` |
| `rl/action_mask.py:23/24` | `from core import Position` / `from card_utils import Card` |
| `rl/belief_planner.py:45` | `from card_utils import Card, card_data` |
| `rl/decks.py:25` | `from card_utils import card_data` |
| `rl/threat_precise.py:71/72` | `from arena import TileGrid` / `from core import Position` |
| `rl/selftest.py:35` | `from card_utils import Card`（另 5317 `from arena import TileGrid`） |

**传递代价（AST 传递闭包）**：`rl/env_wrapper` 一个顶层 `import battle` 就把 **11 个引擎模块**拉进运行时：`arena, battle, card_mechanics, card_utils, core, elite17_data, evo_2025_data, evolutions, new_visualization, pathfinding_heap, player`。

### 4.3 全仓被依赖最多的模块

| # | 模块 | 引擎 importer | **rl importer** | 合计 | import 语句 |
|---:|---|---:|---:|---:|---:|
| 1 | `card_utils` | 8 | **12** | **20** | 44 |
| 2 | `battle` | 15 | 4 | **19** | **91** |
| 3 | `core` | 14 | 5 | 19 | 44 |
| 4 | `player` | 15 | 4 | 19 | 38 |
| 5 | `arena` | 5 | 2 | 7 | 9 |
| 6 | `environment` | 7 | 0 | 7 | 7 |
| 7 | `spell_module` | 0 | 4 | 4 | 4 |
| 8 | `elite17_data` | 3 | 0 | 3 | 7 |
| 9 | `new_visualization` | 2 | 1 | 3 | 3 |
| 10 | `simulate_exchange` | 0 | 2 | 2 | 2 |
| 11 | `threat_calc` | 0 | 2 | 2 | 4 |

**叶子**：`core` / `elite17_data` / `evo_2025_data` / `run_raw_capture`。
**孤立**：`card_aliases`（只被 `rl.decks` 用）、**`pathfinding`（0 importer，死代码 X2）**。

### 4.4 ★ 最贵的问题：cwd 隐性契约

```python
# src/clasher_new/card_utils.py:4-17  （模块顶层，import 时即执行）
with open('gamedata.json', encoding='utf-8') as f: ...
with open('cards_stats_characters.json', encoding='utf-8') as f: ...
with open('cards_stats_spell.json', encoding='utf-8') as f: ...
with open('cards_stats_building.json', encoding='utf-8') as f: ...
with open('cards_stats_projectile.json', encoding='utf-8') as f: ...
```

`card_utils` 是 **importer 最多的引擎模块（20 个，其中 12 个 rl）** ⇒ **任何 import 都隐含要求 cwd = `src/clasher_new`**。仓库用 wrapper 的 `os.chdir(_SRC)` 把这条约束藏起来（`scripts/rl/*.py:5-7` 共 11 个 wrapper + `scripts/run_selftests.py` docstring + `_mask_diff_snapshot.py:13`）。

**后果**：① 在别处 import 会 `FileNotFoundError`；② 两个 pool 测试（`selftest.py:3897/4190`）依赖 cwd 相对 glob `runs/...`/`../../runs/...`，在别处跑会**静默 `[SKIP]`（假绿）**；③ 数据文件挪位 = 全仓 20 个 importer 同时失效（§7 拒绝项 4）。

### 4.5 建议（不重写，只加门面）

| 建议 | 做法 | 风险 |
|---|---|---|
| **加薄门面** | 新增 `rl/engine_api.py`，re-export `battle`/`player`/`card_utils`/`core`/`threat_calc`/`spell_module` 的既有符号；rl 侧新代码只 import 门面 | 低（纯新增，不动既有 import） |
| **cwd 契约显式化** | 在 `card_utils.py` 顶部加**显式断言/报错信息**（找不到数据文件时打印期望 cwd），**不改成自动找路径** | 低（错误信息变，行为不变） |
| **禁止新增引擎 → rl 反向边** | 加一条只读检查进 `scripts/_structure_check.py` | 零 |

---

## 5. 测试与工具布局问题

### 5.1 速览

| 问题 | 结论 |
|---|---|
| `rl/selftest.py` | **5909 行 / 100 个 `test_*`**，`main()` **手工逐条登记**（100↔100 双向无差），**无自动发现** |
| 自动发现 | 只在 `scripts/run_selftests.py --list`（`dir()` 反射，`sorted()` 字典序） |
| 可单独跑 | **100/100 可**；但依赖 cwd=`src/clasher_new`；2 个测试缺真实 ckpt 会**静默 SKIP** |
| 测试互相依赖 | **无 test→test 调用、无全局 env/build 单例**；真实耦合是**仓库文件系统**（2 处 ckpt glob + 1 处写 `runs/_tmp_drawtest`） |
| 归属 | `rl/selftest.py` **100** vs `scripts/selftest_*.py` **11** vs `scripts/test_m*.py` **98** |
| `scripts/` 边界 | 105 个文件中：**测试 16 / 一次性取证 67 / 可复用仪器 22** ⇒ **一次性取证占 64%** |
| `env.md §2.4` | 表 **32 行 / 36 个 `scripts/*` 文件**；实际 **105** ⇒ **69 个未登记** |

### 5.2 `rl/selftest.py` 的 4 条硬约束与 4 个真问题

**硬约束**见 §2.2（`dir()` 反射 / 整文件 runpy / `main()` 100 行清单 / `_apply_s2_channel_when_idle.sh:70-71` 硬编码 4 名）。

**真问题**：

| # | 问题 | 证据 | 后果 |
|---|---|---|---|
| P1 | `main()` **手工登记、失败即中止、无 per-test try/except、无 x/y 计数** | L5801-5909；只有末尾一句 `ALL SELFTESTS PASSED` | 首个失败后**剩余测试全不跑**，且不知道进度 |
| P2 | `--list` 顺序（字母序）≠ `main()` 顺序（手工序） | `run_selftests.py:36-37` | 子集跑的先后与全量不一致 |
| P3 | 2 个测试**静默假绿** | `test_opponent_pool_mix` L3897 / `test_opponent_pool_rand_anchor` L4190，缺 ckpt 时 `[SKIP]` / `if` 整段跳过 | 换机/清 `runs/` 后**看起来还是绿的** |
| P4 | 1 个测试**写仓库工作区**且清理不保证 | `test_draw_penalty_as_loss` L1926 `out_dir="runs/_tmp_drawtest"`，L1941 `shutil.rmtree` | 断言中途失败 ⇒ 残留目录 |

（无 `time.sleep` 0 处、无 `subprocess`/`multiprocessing` 0 处；多进程经 `run_league` 间接起 worker。）

### 5.3 测试归属混乱（三处并存）

| 位置 | 文件 | `test_*` 数 | 行数 |
|---|---|---:|---:|
| **A** `src/clasher_new/rl/selftest.py` | 1 | **100** | 5,909 |
| **B** `scripts/selftest_*.py` | 2 | **11** | 816 |
| **C** `scripts/test_m*.py` | 6 | **98** | 2,300（m1/m2/m3/m4/m5/m6） |

> **A vs B = 100 : 11**；把 C 算上 `scripts/` 侧共 **109**，**超过 A 侧**。三处无统一发现入口 —— `scripts/test_m*.py` 与 `scripts/selftest_*.py` **不在** `run_selftests.py` 的可见范围内（它只反射 `rl.selftest`）。见 §8-C5。

### 5.4 `scripts/` 无目录边界

| 类 | 数量 | 例子 |
|---|---:|---|
| **可复用仪器**（应长期保留） | 22 | `check_commit.py` / `check_dashboard_js.py` / `coverage.py` / `health_curve.py` / `judge_anchor_blocks.py` / `judge_critic_inertia.py` / `et_solo100k_readout.py` / `offline_engagement_trade.py` / `analyze_online_trade.py` / `pass_streak_audit.py` / `_mask_diff_snapshot.py` / `_mask_vs_engine_reconcile.py` / `random_eval_100.py` / `run_selftests.py` / `s1_plan_gate.py` / `_schema5_probe.py` / `s2_neutrality_probe.py` / `s2_trade_probe.py` / `_verify_fix_replays.py` / `phi_offline_check.py` / `kill_orphan_workers.ps1` / `ps_list_python.ps1` |
| **一次性取证**（应归档） | 67 | `probe_*.py`(30) / `_probe_*.py`(3) / `diag_*.py`(4) / `_survey_*.py`(9) / `judge_probe_*.py`(2) / `summarize_*.py`(2) / 其余 |
| **测试** | 16 | `selftest_*.py`(2) + `test_m*.py`(6) + `assassin_*_test.py` 等 |
| **`scripts/rl/`（wrapper）** | 11 | 全部 87 行：`os.chdir(_SRC)` + `runpy`/转发 |

### 5.5 建议

1. **建立三分类目录**：`scripts/tools/`（22 仪器）/ `scripts/probes/`（67 取证，加 ARCHIVE 标记）/ `scripts/tests/`（16）。**先改文档再改路径（【R18】）**，否则 `env.md §2.4`/AGENTS.md 的 32 行引用与 `finalize_et_solo100k.sh` 的相对路径会断。→ Tier 2。
2. **补 `env.md §2.4` 的 69 个未登记项**（纯文档）→ Tier 0。
3. **selftest 加 `--list` 自动发现 + per-test `try/except` + x/y 计数**（**不改 `main()` 的手工清单顺序**）→ Tier 1。
4. **把 P3 的两个假绿改成"显式失败或显式 SKIP 且计入总数"** → Tier 1（行为变化仅限"报告"，但 `[SKIP]` 改 FAIL 会**让当前绿的测试变红** ⇒ 必须先确认 ckpt 存在；谨慎）。
5. **P4 的临时目录改到 `tempfile.mkdtemp()`** → Tier 1（低风险，`test_draw_penalty_as_loss` 单测）。

---

## 6. 分级方案 Tier 0 / 1 / 2 / 3

> **规则**：Tier 0 = **零行为变化 + 可独立验证**；Tier 越高风险越高。
> **每级格式**：做什么 / 涉及文件 / 行为中性理由（或风险）/ 验证手段（**既有脚本名**）/ 预计冲突（与并发会话）。
> **并发会话现状（2026-09-19 实测）**：未跟踪新文档 15 份 = `docs/cmp_*_2026-09-19.md`(8) + `docs/reward_mechanism_*`(4) + `docs/review_reward_redlines_*`(1) + `docs/audit_clashaiaa_*`(1) + `docs/clashaiaa_reward_ddq_*`(1) ⇒ 正在做**奖励机制对比/FL 对照**。因此**高冲突文件** = `rl/config.py`（`DEFAULT_REWARD`）、`rl/engagement.py`、`battle.py`（root_cast 通道）、`scripts/offline_engagement_trade.py`、`scripts/analyze_online_trade.py`、`docs/agents/*`。
> **运行中进程**：`rl/dashboard.py` 两个常驻面板（**8700** = `runs/et_ctrl100k`，**8701** = `runs/rand100_eval`）—— **不得 kill、不得改 `dashboard.py` 的 `/api/*` 契约**。

### Tier 0 — 零行为变化、可独立验证（可立刻执行）

#### T0-1 删除确认死件
- **做什么**：删 `X1 re_lib.py` / `X2 pathfinding.py` / `X3 tmp_*.py`×5 / `X4 runs/_tmp_*`×18(+`.out`) / `X5 gamedata.json.bak_ronin_vines` / `X7 1`、`.p6.png`、`0631.pdf` / `X8 _tmp_probe_path.out` / `X10 stop_solo_*`。`X9 hero_ability_extract.json` **保留**（只标注）。
- **涉及文件**：§3.2 表列出的路径（`git rm` 仅 `re_lib.py` 与 `pathfinding.py` 是 tracked）。
- **为什么行为中性**：全部 **0 importer**（grep 复算）+ gitignored / tracked-dead；不在任何 import 图或运行路径上。
- **验证手段**：`grep` 复算 0 命中；`git status --porcelain` 只减不增；`scripts/run_selftests.py --list` 计数不变（100）；【R19】子集 `scripts/run_selftests.py test_precise_threat test_noncombat_entity_contract test_replay_roundtrip`；`scripts/check_dashboard_js.py`。
- **预计冲突**：**无**。`_tmp_fl_*` 已被并发会话删掉（X6），**不要重复处理**。`docs/` 侧完全不碰。

#### T0-2 新增只读结构核对器 `scripts/_structure_check.py`（新文件）
- **做什么**：复算 ① 分区文件/LOC；② 3 份 `_force_utf8_stdout` 是否仍逐字相同；③ 引擎→`rl/` 反向边是否为 0；④ 死件是否复活；⑤ 是否新增硬编码绝对路径。输出 JSON。
- **涉及文件**：仅新增 `scripts/_structure_check.py`。
- **为什么行为中性**：纯新增只读脚本，不改任何既有文件、不 import 产品代码（用 AST 文本解析）。
- **验证手段**：脚本自身 `--selftest`（新写）；手动跑一次与 §1.1/§3 数字对齐；子任务③的复算命令逐条重跑。
- **预计冲突**：无。

#### T0-3 文档补齐（纯文档，零代码）
- **做什么**：① 本方案落 `docs/structure_optimization_plan_2026-09-19.md`；② 补 `docs/agents/env.md §2.4` 的 **69 个未登记 scripts**；③ 在 `AGENTS.md` 加一行指针（**只增不改历史**）。
- **涉及文件**：`docs/structure_optimization_plan_2026-09-19.md`（新）/ `docs/agents/env.md` / `AGENTS.md`。
- **为什么行为中性**：文档。
- **验证手段**：`scripts/_survey_reverse_check.py`（`full_code_reference.md` 符号抽检）；`scripts/run_selftests.py --list` 数不变。
- **预计冲突**：**中**。`AGENTS.md` 与 `docs/agents/ledger.md` 正被并发会话追加 2026-09-19 奖励条目 ⇒ **只追加自己的段落，不重排**。

#### T0-4 给 `card_utils` 加"数据缺失即明确报错"
- **做什么**：`card_utils.py:4-17` 的 5 个 `open()` 包一层，失败时打印**期望 cwd**；**不改成自动找路径**。
- **涉及文件**：`src/clasher_new/card_utils.py`。
- **为什么行为中性**：成功路径逐字不变（同一 `open`、同一文件名）。
- **验证手段**：`scripts/run_selftests.py test_random_deck_model test_classified_decks`；`scripts/_mask_diff_snapshot.py /tmp/m.npy`（前后逐位）。
- **预计冲突**：低（`card_utils.py` 不在奖励工作集）。**但**它是全仓 importer 最多的模块 ⇒ 改动必须最小。

### Tier 1 — 低风险等价重构（需人工确认，验证可脚本化）

#### T1-1 收敛 `_force_utf8_stdout` 到单一实现
- **做什么**：新增 `rl/io_bootstrap.py::force_utf8_stdout()`；`run_league.py:1538` / `dashboard.py:3033` / `probe_value_ln.py:116` 改为 re-export（**保留旧名**）；随后按批收敛 ④ 内联。
- **涉及文件**：`src/clasher_new/rl/io_bootstrap.py`(新) / `rl/run_league.py` / `rl/dashboard.py` / `scripts/probe_value_ln.py` / 其余 ~59 处（分批）。
- **为什么行为中性**：3 处函数体**已逐字相同**（实测）；re-export 保 API 名。
- **验证手段**：`scripts/run_selftests.py test_dashboard_league_payload test_dashboard_replays test_dashboard_card_stats`；`scripts/check_dashboard_js.py`；`scripts/health_curve.py --selftest`。
- **预计冲突**：**低**。`rl/run_league.py` 的 L1538-1551 与 `dashboard.py` 的 L3033-3046 不在奖励工作集；但**必须等 T0-3 的 AGENTS 追加落盘后再动 `dashboard.py`**（8700/8701 正在跑它）。

#### T1-2 修 5 处硬编码绝对路径
- **做什么**：`assassin_left_bridge_test.py:20` / `assassin_vs_megaknight.py:20` / `assassin_vs_sparky.py:25` / `duel_search.py:35` / `question_bank_poc.py:27` 的 `E:/clash-royale-simulator-main/src/clasher_new` → `__file__` 推导。
- **涉及文件**：5 个 scripts。
- **风险（非中性但有界）**：在本机解析到**同一目录** ⇒ 行为等价；但**这 5 个是 Windows 侧探针，Linux 容器里无法端到端跑** ⇒ 验证只能在 Windows 执行。若无法验证，降级为"加 TODO 注释"。
- **验证手段**：Windows 侧 `python scripts/duel_search.py --help`（或 smoke）；`grep` 复算 5→0。
- **预计冲突**：无。

#### T1-3 selftest 加自动发现 + 计数（**不改手工清单顺序**）
- **做什么**：`selftest.py` 内部加 `_discover()`（`globals()` 里 `test_*`）；`main()` 仍按**现有手工顺序**跑，只额外打印 `x/y` 与每个测试耗时；`scripts/run_selftests.py` 的 `--list` 增加"定义序 = 手工序"的差集报告。
- **涉及文件**：`src/clasher_new/rl/selftest.py` / `scripts/run_selftests.py`。
- **为什么行为中性**：`main()` 的调用序列、断言、`ALL SELFTESTS PASSED` 位置**逐字不变**；只加打印。
- **验证手段**：`scripts/run_selftests.py --list | wc -l` = 100；`diff` 新老 `main()` 调用清单；`scripts/rl/selftest.py`（全量，**可选**，见【R19】）。
- **预计冲突**：**高**。`selftest.py` 是并发会话的 S2 取证入口 ⇒ **只加不删、只改 `main()` 尾部**，避免与 S2 测试增删撞行。若并发会话正在改该文件，**推迟**。

#### T1-4 P4 临时目录改 `tempfile`
- **做什么**：`selftest.py:1926` 的 `out_dir="runs/_tmp_drawtest"` → `tempfile.mkdtemp()`，清理进 `finally`。
- **涉及文件**：`src/clasher_new/rl/selftest.py`。
- **为什么行为中性**：测试的断言不变，只是临时目录位置变；**唯一可观测差异 = 不再在 `runs/` 留目录**（这正是修复目标）。
- **验证手段**：`scripts/run_selftests.py test_draw_penalty_as_loss`；跑后 `ls src/clasher_new/runs/_tmp_drawtest` 不存在。
- **预计冲突**：中（同 T1-3 的文件）。

#### T1-5 显式化 P3 假绿
- **做什么**：`test_opponent_pool_mix`(L3897) / `test_opponent_pool_rand_anchor`(L4190) 在无 ckpt 时打**醒目** `[SKIP-NO-CKPT]` 并**累计到 `main()` 的 skip 计数**（仍不 FAIL，避免让现有的绿变红）。
- **涉及文件**：`src/clasher_new/rl/selftest.py`。
- **风险**：若改成 FAIL ⇒ 当前环境**两侧各 51 个 ckpt 存在**，暂时仍绿；但换机会红。**选 SKIP-NO-CKPT 更安全**。
- **验证手段**：临时改名 `runs/economy` 跑一次看是否打 `[SKIP-NO-CKPT]`（**验证后立即改回**）；再正常跑一次。
- **预计冲突**：中。

#### T1-6 `scripts/_structure_check.py` 进 `run_selftests` 前置
- **做什么**：把 T0-2 的核对器挂在 `scripts/run_selftests.py` 开头（可选 `--check-structure`），默认关。
- **涉及文件**：`scripts/run_selftests.py` / `scripts/_structure_check.py`。
- **为什么行为中性**：默认关；开启时只读。
- **验证手段**：`scripts/run_selftests.py --check-structure test_precise_threat`。
- **预计冲突**：无。

### Tier 2 — 结构性拆分（有明确风险，必须对账）

#### T2-1 `dashboard.py` 抽出 `_HTML` 常量
- **做什么**：L865-2743（1879 行）→ `rl/dashboard_html.py::HTML`；`dashboard.py` 改 `from rl.dashboard_html import HTML as _HTML`。
- **涉及文件**：`rl/dashboard.py` / `rl/dashboard_html.py`(新)。
- **风险**：字符串是**逐字节**依赖；抽错一个转义就白屏。且 **8700/8701 正在跑 `dashboard.py`** ⇒ 抽完需重启面板验证（**需用户同意**）。
- **验证手段**：`sha256sum` 抽出前后常量；`scripts/check_dashboard_js.py`；`scripts/run_selftests.py test_dashboard_replays test_dashboard_league_payload test_dashboard_card_stats`；重启 8701 面板目视（**先问用户**）。
- **预计冲突**：低（文件本体不在奖励工作集），但**运行中进程**是约束。

#### T2-2 `dashboard.py` 抽出数据面（L94-806，~20 个纯函数）
- **做什么**：→ `rl/dashboard_data.py`，留 re-export。
- **风险**：`Handler` 与 `main` 调这些函数；`_REPLAY_META_CACHE`/`_HEALTH_CACHE`/`_CARD_STATS_CACHE`/`_DECK_INDEX` 是**模块级可变缓存**，跟着搬会改变缓存身份（跨模块共享 vs 各自一份）⇒ **缓存对象必须跟着函数一起搬**。
- **验证手段**：`scripts/run_selftests.py test_dashboard_*`；`scripts/check_dashboard_js.py`；`scripts/health_curve.py --selftest`。
- **预计冲突**：低。

#### T2-3 `run_league.py` 抽出判定语义 + 评估并行
- **做什么**：L109-274 → `rl/league_rules.py`；L581-759 → `rl/league_eval.py`。
- **风险**：**中高**。`settle_stall`/`timeout_winner` 被 `selftest.py` 直接调；`_eval_pair_worker_main` 是**多进程入口**（`_run_eval_pairs_parallel`），移动会改 pickle 的模块路径 ⇒ worker 启动语义可能变。
- **验证手段**：`scripts/run_selftests.py test_league_elo_history test_winrate_streams_independent test_parallel_batch_equivalence test_mp_training_loop`；`scripts/s2_neutrality_probe.py`；`scripts/_schema5_probe.py`。
- **预计冲突**：中。`run_league.py` 的 `_set_et_measure`(L478) / `_ET_MEASURE`(L468-475) 是 S2 接线点 ⇒ **保留原地不动**。

#### T2-4 `train_solo.py` 内部函数抽取（**只抽函数，不改默认参数/顺序**）
- **做什么**：`run_solo`(702) 按"收集→更新→评估→写状态"抽 4 个内部函数；门禁 → `rl/solo_gates.py`；对手池 → `rl/opponent_pool.py`。
- **涉及文件**：`rl/train_solo.py` + 2 个新文件。
- **风险**：**高** —— 【R2】训练语义红线。**允许**：把已有代码块搬进函数（行为逐位同）。**禁止**：改 `DEFAULT_SOLO_DECK`、`_OPP_MIX`、`_PFSP_*`、`_HIST_POOL_MAX`、调用顺序、RNG 调用次序。
- **验证手段**：`scripts/check_commit.py`（长跑前必跑）；【R19】子集 `scripts/run_selftests.py test_league_training_loop test_parallel_training_loop test_mp_training_loop test_belief_follower_ppo_league`；⚠️ **"同 seed 逐位对账"在 CUDA 上不可复现**（`docs/et_solo100k_judgment_2026-09-18.md §11.13.11` 实测同配置重跑单点差 0.60）⇒ 只能靠"导入等价 + 子集 selftest"，不能靠逐位。
- **预计冲突**：**高**。并发会话在跑奖励机制对比 ⇒ 可能改 `rl/train_solo.py` 的评估/门禁路径。

#### T2-5 `card_mechanics.py` 按族拆包
- **做什么**：7 族（§2.7）→ `card_mechanics/` 包 + `__init__.py` re-export 全部 58 类。
- **风险**：中。类名是**字符串查表键**（`battle.py` 动态取类）⇒ 名字一个不能改；`evolutions.py` 反向 `import card_utils`，`card_mechanics` 顶层 `import battle` ⇒ 循环 import 顺序敏感。
- **验证手段**：`scripts/test_m3_evo.py` / `scripts/test_m6_elite.py` / `scripts/test_m2.py` / `scripts/test_m4_evo7.py` / `scripts/test_m5_data.py`；`scripts/coverage.py`；`scripts/run_selftests.py test_rlenv_card_level test_tower_troop_hp_reference`。
- **预计冲突**：低。

#### T2-6 `scripts/` 分目录（tools / probes / tests）
- **做什么**：§5.5 建议 1；wrapper `scripts/rl/*` 保持原位（`start_rl.bat` 依赖）。
- **风险**：**高** —— `env.md §2.4`（32 行）、`AGENTS.md` 索引、`finalize_et_solo100k.sh`、`_apply_s2_channel_when_idle.sh`、`run_probe_v3.sh` 的相对路径全断。
- **验证手段**：移动后逐条复算文档里的命令；`scripts/coverage.py`；`scripts/run_probe_v3.sh`（dry）；**先做 T0-3 的文档补齐**。
- **预计冲突**：中（`docs/agents/*` 并发追加）。

#### T2-7 `env_wrapper.py` 抽出奖励簇
- **做什么**：`compute_reward`(L182-264) + `_per_tower_norm_dmg`(L132-162) + `_phase_weights` + `tower_*` + `_princesses_alive`（L75-264）→ `rl/reward.py`。
- **风险**：**高** —— ⚠️【R7】（常量同源）+ 奖励语义在【R2】边界；且 `_per_tower_norm_dmg` 是 S2 读数的历史接口。**并发会话正在改奖励** ⇒ **本项必须排到并发会话收尾之后**。
- **验证手段**：`scripts/selftest_offline_engagement_trade.py`（8/8）；`scripts/selftest_engagement_trade_online.py`（1/1）；`scripts/run_selftests.py test_config_reward_weights test_model_reward_overrides test_reward_economy_* test_tower_troop_hp_reference`；**开关关逐位对账**（`4d8b27b8…eca0c` 模式）。
- **预计冲突**：**极高**（直接撞 `rl/config.py`+`rl/engagement.py` 工作集）⇒ **禁止并发期间执行**。

#### T2-8 selftest 拆分（`rl/selftests/` 包，旧文件只留 re-export + `main()`）
- **做什么**：§2.2 的 5 簇无阻塞先拆；`rl/selftest.py` 保 `_PARENT`、4 个共享 helper、`main()`、末尾 `if __name__`。
- **涉及文件**：`rl/selftest.py` + `rl/selftests/*.py`。
- **风险**：**高**。四条硬约束（§2.2）+ 3 处"晚绑定"（`_FakeCfg`@1250、`_intents`@1509、`_tiny_rollout_transitions`@3972）+ 1 处 `__file__` 根推导（L5507/5568）会因搬移而变。
- **验证手段**：`scripts/run_selftests.py --list | wc -l` = 100 且**逐名 diff**；`scripts/rl/selftest.py`（全量）；`scripts/_mask_diff_snapshot.py`；`bash -n scripts/_apply_s2_channel_when_idle.sh`（确认 4 个硬编码名仍在）。
- **预计冲突**：**极高**（`selftest.py` 是并发会话 S2 取证入口）⇒ **禁止并发期间执行**。

### Tier 3 — 高风险 / 触红线 / 需用户拍板

#### T3-1 拆 `battle.py`
- **做什么**：`BattleState`(832) / `Entity`(777) / `Troop`(472) / `Projectile` / `AreaEffect` / 各类 zone 分文件。
- **风险**：**极高**。【R13】掩码/合法格判定依赖 `BattleState` 的部署语义（`battle.py:2702`/`2996-3090`）；`card_mechanics.py` 反向 `import battle`；`battle` 全仓 **91 条 import 语句 / 19 importer**。位图对账只能证明 **128 张快照**不变，**不能证明全局**。
- **验证手段**：`scripts/_mask_diff_snapshot.py <out.npy>`（**128 张逐位全等**，【R13】强制）；`scripts/_mask_vs_engine_reconcile.py`；`scripts/probe_precise_threat.py`；全量 `scripts/rl/selftest.py`；`scripts/check_dashboard_js.py`。
- **预计冲突**：**极高** —— 并发会话正在 `battle.py` 上做 `root_cast` 出处追踪通道。⇒ **并发期间明确禁止**。

#### T3-2 数据文件归拢（`*.json` + 16 个散落 `.pt`）
- **做什么**：`src/clasher_new/*.json` → `assets/`；`.pt` → `runs/`。
- **风险**：**极高** —— `card_utils.py:4-17` 用**裸相对路径** `open('gamedata.json')`，移动即让 **20 个 importer** 同时失效（§4.4）。除非同时改 `card_utils` 并**立 cwd/根路径契约**，否则不可做。
- **验证手段**：`scripts/_mask_diff_snapshot.py`；`scripts/run_selftests.py --list` 后跑全部 `test_*` 中涉及卡牌的；`scripts/coverage.py`。
- **预计冲突**：低（文件不在奖励工作集），但**语义面极大**。

#### T3-3 `rl/selftest.py` 迁 `tests/` + 引入 pytest
- **做什么**：目录迁移 + 测试框架。
- **风险**：**极高**。触及【R19】（默认只跑子集）与 2 个**硬编码调用方**（`scripts/run_selftests.py:36-42`、`scripts/_apply_s2_channel_when_idle.sh:70-71`）；"全量路径逐位不变"的保证会丢。
- **验证手段**：迁移后必须让 `scripts/run_selftests.py` 与 `_apply_s2_channel_when_idle.sh` 仍能按名调用；`scripts/rl/selftest.py` wrapper 重写。
- **预计冲突**：极高。⇒ **需用户拍板**（§7 拒绝项 2 给出"不做"的默认立场）。

#### T3-4 统一 `sys.path` 引导（116 文件）
- **做什么**：全部收敛到 `scripts/_bootstrap.py`。
- **风险**：高（面大、`sys.path` 顺序影响 import 解析；`os.chdir` 与 `sys.path` 顺序耦合）。
- **验证手段**：每批跑 `scripts/run_selftests.py --list` + 该批脚本的 `--help`；`scripts/coverage.py`。
- **预计冲突**：中（跨 `scripts/` 全目录）。

---

## 7. 碰不得的部分 + 拒绝做的重构

### 7.1 明确"碰不得"（改动必须先立预注册 + 对账）

| 红线 | 具体文件:行号 / 实体 | 约束 |
|---|---|---|
| **【R2】训练语义** | `rl/ppo.py` 全文件（547 行，默认参数 = 旧行为）；`rl/config.py:37-68` `DEFAULT_REWARD` 默认值；`rl/train_solo.py:56-57,67-69,129,145,148,152-154`（`DEFAULT_SOLO_DECK`/锚点/`_SOLO_PROPHET_PROB`/`_OPP_MIX`/`_HIST_POOL_MAX`/`_PFSP_*`）；`rl/follower.py`（`stop_logit_bias` L156/253-255）；`run_solo` 的调用顺序与 RNG 调用次序 | 新能力**只经 `TrainConfig` 显式开**；默认路径逐位回旧 |
| **【R7】单常量源** | `config.py:61,63`（`elixir_diff_weight`/`elixir_diff_late`）；`config.py:53-60,64-68`（`tower_dmg_*`/`win_bonus`/`lose_penalty`/`draw_penalty`/`invalid_penalty`/`unit_dmg_k`/`tower_premium_k`/`king_gate`）；`config.py:73-80` `MODEL_REWARD_OVERRIDES`；`run_league.py:154-155`（`STALL_WINDOW`/`STALL_LIMIT`）；`run_league.py:468-475`（`_ET_MEASURE*`）；`belief_planner.py:51`（`PRESSURE_THRESHOLD=2.0`）；`env_wrapper.py:50`（`invalid_penalty`）| 汇率 / 值函数 / 闸门 `edw×卡费` / MCTS **必须同源同步改** + 对账 selftest |
| **【R13】位图对账** | `rl/action_mask.py` 全文件（573 行，9 条中文 reason，`validate_bundle` L521-573）；`rl/env_wrapper.py:473-536` `get_action_mask_for`（`used` **0-based vs 1-based** off-by-one 历史事故）；`rl/follower.py` `masked_fill(-1e9)`；`battle.py:2702`/`2996-3090` 部署合法性 | 任何掩码/合法格改动**先跑 `scripts/_mask_diff_snapshot.py`（128 张逐位全等）** |
| **已冻结 schema** | `rl/replay.py:82` **`LEAGUE_REPLAY_SCHEMA = 5`**；`rl/replay.py:18` `SCHEMA_VERSION = 2` | **只许末尾追加**新键/新元组位；**禁止重排/删除/改语义**（schema 3→4→5 的既有约定）。`et` 键、`root_cast`/`is_product`/`share`、实体元组长度 15 全部冻结 |
| **运行中契约** | `rl/dashboard.py` 的 `/api/*` **10 条**（8700/8701 两个常驻面板在跑）；`.venv` 解释器约定；**cwd = `src/clasher_new`** | 不得 kill 面板；不得改 API 形状；不得在未立契约前改 cwd 依赖 |
| **既有文档** | `AGENTS.md` + `docs/agents/*`（红线/台账/口径） | **只增不改历史结论**；被推翻时保留原条目并标注（【R18】同一步改文档） |

### 7.2 拒绝做的重构（含理由）

| # | 拒绝项 | 理由 |
|---|---|---|
| 1 | **任何"重写"** | 用户明令禁止；且 `battle.py`/`ppo.py` 的行为已被 100+ selftest 与 6 个口径缺陷的修复史锚定，重写必然丢【R2】【R13】。 |
| 2 | **把 `rl/selftest.py` 改成 pytest 框架** | 触及【R19】+ 2 个外部硬编码调用方（`run_selftests.py:36-42`、`_apply_s2_channel_when_idle.sh:70-71`）；"`main()` 手工清单与全量路径逐位不变"的保证会丢。⇒ Tier 3，需用户拍板；**默认不做**。 |
| 3 | **为"可读性"拆 `battle.py`** | 【R13】语义面太宽：`_mask_diff_snapshot.py` 只覆盖 128 张代表性快照，拆 `BattleState` 无法用位图对账证明全局等价。**收益（可读性）远小于风险（掩码/部署语义漂移）。** |
| 4 | **给 `card_utils.open('gamedata.json')` 加"自动找路径"魔法** | 会同时破坏 ① cwd 约定（11 个 wrapper + 全部文档的"必须在 `src/clasher_new` 下跑"）与 ② 20 个 importer 的既有假设。**要改必须先立 cwd/根路径契约并全量验证**（Tier 3），不接受"顺手加个 `os.path.dirname(__file__)`"。 |
| 5 | **删 `src/clasher_new/runs/` 整目录（6.7 G）** | 它是**全部 S2/长跑读数的数据源**（schema-4/5 录像 + ckpt），台账【C14/O8/O9/O10】直接引用。只删其中的 `_tmp_*`（Tier 0）。 |
| 6 | **合并 `runs/` 与 `src/clasher_new/runs/` 两个产物区** | 合并会让 `selftest.py` 的 `glob("runs/economy/solo_main_*.pt")`(L3933) 与 `glob("../../runs/archive/*/…")`(L3934) **改变命中集** ⇒ 破坏 2 个 pool 测试的语义（当前两侧各 51 ckpt，实测）。 |
| 7 | **删 `docs/full_code_reference.md`（2.4 MB 自动生成物）** | 被 `scripts/_survey_reverse_check.py` 当**符号抽检基准**（每 60 条取 1 条）；删了会丢一条反向核对手段。要删必须先替换核对基准。 |
| 8 | **并发期间动 `battle.py` / `rl/config.py` / `rl/engagement.py` / `scripts/offline_engagement_trade.py` / `scripts/analyze_online_trade.py`** | 并发会话正在做奖励机制对比（15 份未跟踪 `docs/cmp_*`/`reward_mechanism_*`），`battle.py` 上还有 `root_cast` 通道落地。**冲突处并列不调和**：即使本方案认为该拆，也不在此期间拆。 |
| 9 | **把 `selftest.py` 的 471 行函数内 import 提到模块顶层** | 那 471 行是**刻意的零模块级重依赖设计**，保证 `import rl.selftest` 秒级（`run_selftests.py:42` 依赖它）。提到顶层会让子集测试启动变慢并可能引入 torch/引擎副作用。 |
| 10 | **改 `X9 hero_ability_extract.json` 的归属或删除** | 它是**唯一未被代码消费的数据文件**（只被 `coverage.py:240` 的字符串提及）；可能是**待接入**而非死件。标注，不删。 |

---

## 8. 冲突并列（不调和）

> 以下是 5 份盘点之间**互相矛盾或口径不同**之处。**并列保留，不取平均、不裁决**；执行者遇到时**必须重数**。

### C1 文件总数：202 / 199 / 196

| 来源 | 口径 | 值 |
|---|---|---|
| 子任务①（17:03Z） | 全仓 `.py`（排除 venv/git） | **202 文件 / 62,638 行** |
| 子任务①（17:04:36Z，原子快照） | 同上 | **199 文件 / 62,402 行** |
| 子任务③ | `scope`（排除 `runs/` 与 gitignored 临时件） | **196 文件** |
| 子任务③ | `with-runs`（含 gitignored） | **202 文件** |
| 本文（2026-09-19 复核） | 排除 `runs/*.py` | **196 文件 / 62,123 行** |
| 本文 | 含 `runs/*.py`（3 文件 279 行） | **199 文件 / 62,402 行** |

**差异来源**：17:03Z→17:04:36Z 并发会话删除了根目录 `_tmp_fl_meta.py`(72) / `_tmp_fl_params.py`(81) / `_tmp_fl_params2.py`(83) = **3 文件 / 236 行**。**196 vs 199 vs 202 是三个不同的排除规则，不是矛盾，但混用会算错。**

### C2 God file 行数：粗算法 vs AST 精算（**真矛盾**）

| 实体 | 粗算法（扫到下一个 `^def`/`^class`） | **AST `end_lineno`** | 差因 |
|---|---:|---:|---|
| `dashboard.py::make_demo_replays` | **1937 行**（L809-2745） | **50 行**（L809-858） | 粗算法把 L865-2743 的模块级常量 `_HTML` 并进来了 |
| `train_solo.py::_new_episode_reset` | **234 行**（L1516-1749） | **27 行**（L1516-1542） | 粗算法把 L1544 的并列 `for step` 并进来了 |

**并列保留**：本文§2 用 AST 值；**若执行者按粗算法值做拆分预算，会严重高估**。子任务②的原文明确"两处同源：粗算法会把模块级常量/并列语句并进上一个函数"。

### C3 selftest 可抽离簇数：8 vs 5(+2+1)

| 来源 | 结论 |
|---|---|
| 子任务② | **8 簇**低耦合可抽离（判据：只依赖 import 的模块 + 无共享可变模块级状态 + 调用点仅同文件内 `main()`） |
| 本文 §2.2 实测 | **5 簇无阻塞** + **2 簇需带 helper**（`_tiny_rollout_transitions` / `_make_policy_and_tokens`+`_mk_env`） + **1 簇必须成对**（`_FakeCfg` 晚绑定） |

**差异原因**：子任务②的 8 簇判定表在传递中被截断（原文止于"调用点清单（全部"）。⇒ **执行前必须重数**，不要照搬任一数字。

### C4 重复样板计数：scope vs with-runs（口径并存）

| 项 | `scope` | `with-runs` |
|---|---:|---:|
| `reconfigure(encoding` | **66** | **77** |
| `sys.stdout.reconfigure` | 66 | 77 |
| `sys.path.insert` | **122** | **124** |

**并列保留**：报数时必须带口径标签，否则两个数字都对、放一起就错。

### C5 测试归属：100 / 11 / 109（三种算法）

| 算法 | 结果 |
|---|---|
| A vs B（子任务⑤主口径） | **100 : 11** |
| A + B + C（把 `scripts/test_m*.py` 算上） | A **100** vs `scripts/` **109** |

**并列保留**：§5.3 明写"只算 A vs B = 100:11；把 C 算上则 `scripts/` 侧共 109，超过 A 侧"。**两种都是事实，用途不同。**

### C6 两个 `runs/` 产物区（并存，勿合并）

| 区 | 体积 | 内容 |
|---|---:|---|
| `src/clasher_new/runs/` | **6.7 G** | 训练产物主区（782 `.pt` / 587 `.pkl` / 232 `.json`），所有 ckpt/录像 |
| 仓库根 `runs/` | **2.1 G** | `dashboard_10e.log`(1.5 MB) / `forensics_100k.py` / `watch_100k.py` / `analyze_curve.py` / `archive/` |

**`selftest.py:3933-3934` 同时 glob 两处**（`runs/economy/solo_main_*.pt` + `../../runs/archive/*/…`）⇒ **合并会改命中集**（§7 拒绝项 6）。

### C7 selftest 的归属：测试 vs 仪器

- 子任务⑤ 用「测试」口径把它算进 A 侧 100；
- 但 §2.2 实测它**同时是仪器**：`test_*` 里有读仓库磁盘做**不变量核对**的（L5508 `ast.parse` 扫 `battle.py`+`rl/*.py`）、有 `inspect.getsource(run_league.main)`（L4950/5096）、有读真实 ckpt 的（L3897/4190）。
- **并列保留**：它是"测试 + 仪器 + 回归"的混合体，**任何按"纯测试"设计的拆分方案都会漏掉仪器语义**（例：`__file__` 根推导 L5507/5568 搬移即失效）。

---

## 附：执行清单速查（按 Tier 顺序）

| 序 | 项 | Tier | 首要验证脚本 |
|---|---|---|---|
| 1 | T0-1 删死件 | 0 | `scripts/run_selftests.py --list` + 子集 |
| 2 | T0-2 结构核对器 | 0 | 自身 `--selftest` |
| 3 | T0-3 文档补齐 | 0 | `scripts/_survey_reverse_check.py` |
| 4 | T0-4 `card_utils` 报错 | 0 | `scripts/_mask_diff_snapshot.py` |
| 5 | T1-1 `_force_utf8_stdout` 收敛 | 1 | `scripts/check_dashboard_js.py` |
| 6 | T1-2 绝对路径 | 1 | `grep` 5→0 |
| 7 | T1-3 selftest 自动发现 | 1 | `scripts/run_selftests.py --list \| wc -l` |
| 8 | T1-4 临时目录 | 1 | `scripts/run_selftests.py test_draw_penalty_as_loss` |
| 9 | T1-5 假绿显式化 | 1 | 同 7 |
| 10 | T2-1 `_HTML` 抽出 | 2 | `sha256sum` + `scripts/check_dashboard_js.py` |
| 11 | T2-2 dashboard 数据面 | 2 | `scripts/run_selftests.py test_dashboard_*` |
| 12 | T2-3 run_league 拆分 | 2 | `scripts/s2_neutrality_probe.py` |
| 13 | T2-4 train_solo 抽函数 | 2 | `scripts/check_commit.py` |
| 14 | T2-5 card_mechanics 拆包 | 2 | `scripts/test_m3_evo.py` |
| 15 | T2-6 scripts 分目录 | 2 | `scripts/coverage.py` |
| 16 | T2-7 reward 抽出 | 2 | `scripts/selftest_offline_engagement_trade.py` |
| 17 | T2-8 selftest 拆包 | 2 | `scripts/rl/selftest.py`（全量） |
| 18 | T3-1 `battle.py` 拆分 | 3 | `scripts/_mask_diff_snapshot.py`（128 逐位） |
| 19 | T3-2 数据归拢 | 3 | 同 18 |
| 20 | T3-3 pytest 迁移 | 3 | 需用户拍板 |
| 21 | T3-4 sys.path 统一 | 3 | 分批 |

---

*本文只读产出：未修改、创建、删除任何仓库文件（除本文件自身）。所有数字均为 2026-09-19 实测或注明快照来源；冲突处已并列保留。*
