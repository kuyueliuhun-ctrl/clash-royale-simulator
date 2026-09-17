# run 模式主线化 · 多卡组对手 + dashboard 判读（2026-09-17）

> 用户拍板（2026-09-17）：**训练主线从 `solo` 切到 `--mode run`**（"5 个脚本 agent + PFSP"），
> 并同时要 dashboard 支持**多卡组对战的卡牌数量显示**、且**不再以胜率为曲线指标**。
> 本文件记录：① run 模式的实际语义（实测，非文档转述）② 实测中撞到的**阻塞级引擎 bug**（已修）
> ③ 一条**必须先看的保留**（对手强度）④ dashboard 交付 ⑤ 未决。
> 配套：预注册/判据暂无（本文件是"能力核验 + 事故留证"，不是 A/B 判决）。

---

## 0 一句话

**`--mode run` 可用、多卡组是真的、dashboard 已能显示「卡组 × 卡牌」出牌矩阵**；
但实测第一跑就撞到一个 **`TypeError` 崩训练**的引擎 bug（已修 + 已配静态回归测试），
并且 run 模式的对手**全是 mask-随机脚本**（无防守 AI、无自我对弈历史池）⇒ **能否当"训练主线"取决于你要什么**（见 §4）。

## 1 run 模式实际语义（逐条实测/读码）

| 项 | 事实 | 证据 |
|---|---|---|
| 对手构成 | **5 个 `ScriptedPolicy(mode="random")`** + main。三个按 archetype 分池（推进流 60 / 防守反击流 120 / 自闭流 20）+ 全 200 副 + 全随机牌 | `rl/run_league.py:559-565`；实测启动行 `[league] 三分类卡组已接入: {'推进流': 60, '防守反击流': 120, '自闭流': 20}` |
| **对手强度** | ⚠️ **`ScriptedPolicy` 的 `play()` 与 `mode` 无关——它恒为"从合法掩码里随机选一子动作"** | `rl/opponents.py:103-119`（`play()` 未读 `self.mode`） |
| 卡组多样性 | **真实且立即可见**：一次 20 局的 smoke 产出 **20 副不同卡组**（12 副带 archetype 标签 + 自定义牌组） | `scripts/check_dashboard_js.py` 同款统计 → `/api/cardstats` 的 `decks[]`；本文 §3 有实测表 |
| PFSP | run 模式用 PFSP 给 **5 个脚本 agent 之间**做对手采样（`rl/pfsp.py`）；**没有**"打过去自己的快照"这一路 | `rl/run_league.py` 全仓无 `solo_main_*`/`frozen` 池（`--hist-seed-dir` 的 help 文本是 solo 的措辞，run 模式下该池为空，见 `551-570`） |
| 我方卡组 | `RLEnv` 默认 `DEFAULT_DECK`（原版 8 卡），`deck0_factory` 从不设置 ⇒ **run 模式下我方卡组不可配** | `rl/env_wrapper.py:34`；`rl/run_league.py:585-587` |
| 写出的文件 | `runs/<name>/`：`config.json`、`run_state.json`、`league_state.json`（含 `elo_history`）、`main_ckpt_<step>.pt`、`main_opt_<step>.pt`、`main_final.pt`、`replays/league_<step>.pkl` | 实测 `runs/zzsmoke_run/`、`runs/run_smoke2/` |
| 评估口径 | `eval_round_robin`（main vs 5 agent，每对 `n_eval_games` 局）→ **Elo**（多智能体联合估计） | `rl/run_league.py:516`、`649` |
| `--deck-set` | **在 run 模式下无效**（help 文本自身写明"solo 镜像/对手卡组"）⇒ **不是静默 bug，是设计如此**；但若误传不会有任何提示 | `rl/run_league.py:1225-1227` 只在 argparse/overrides；`--help` 文本 |
| 最小可用命令 | `python rl/run_league.py --mode run --config-name <name> --fresh --total-steps N --steps-per-eval M --n-eval-games K --eval-workers 2 --device cuda`（**不需要** `--hist-seed-dir`、不需要 `--config economy`） | 实测见 §3 |

## 2 ⚠️ **四个**阻塞级引擎 bug（实测逐个撞到 → 逐个修 → 每个配回归测试）

`--mode run` 上 200 副多卡组牌，**第一次跑就在 4 个不同位置崩了 4 次**。
solo 打固定 X弩牌只覆盖了这些路径里的一小部分，所以这 4 个此前从未暴露
（**这正是"多卡组"的隐藏成本**）。判据全部做成**静态/数据驱动**的确定性测试，不靠"跑一局碰运气"。

| # | 崩溃指纹 | 根因 | 修法 | 回归测试 |
|---|---|---|---|---|
| **A** | `TypeError: TimedExplosive.take_damage() got an unexpected keyword argument 'delayed'`（突进 `card_mechanics.py:798` 打到爆炸物） | 全仓调用点约定 `take_damage(amount, delayed, source, pierce_invincible)`，但 4 个"不受伤害"的 no-op 桩签名偏窄（`GenericBomb`/`TimedExplosive` 只有 `(amount)`；`EvoEffectZone`/`VinesSnareZone` 缺 `pierce_invincible`） | 4 个桩统一成标准签名（仍是 no-op，语义不变） | `test_take_damage_signature_consistency`（AST：调用 kwargs ⊆ 定义形参；**阴性对照**：还原窄签名 → FAIL） |
| **B** | `AttributeError: 'VinesSnareZone' object has no attribute 'regen_buffs'`（法术 pulse `battle.py:1814` 给领域挂治疗） | `AreaEffect`/`GenericBomb`/`EvoEffectZone`/`HealAuraZone`/`VinesSnareZone` 共 **5 个类刻意不走 `Entity.__init__`**（无卡牌身份），却仍 `class X(Entity)` ⇒ 任何通用路径碰到它们就 `AttributeError` | 立**显式契约** `Entity.is_combat_entity = True` + 5 个类声明 `False`；在 `apply_buff`/`Entity.take_damage` 两处通用入口兜底（非战斗实体直接无操作） | `test_noncombat_entity_contract`（AST **双向**对账：跳过 `Entity.__init__` ⟺ 声明 `False`；+ 动态：5 个实例跑 `apply_buff` 四条分支与 `take_damage` **必须无异常**；**阴性对照**：删掉一个声明 → FAIL） |
| **C** | `KeyError: 'collisionRadius'`（`SkeletonBalloon` 亡语 `battle.py:312 → card_utils.py:503`） | 亡语炸弹路由"有 `deathDamage` 且无 `hitpoints`"**太宽**：`SkeletonBalloon` 的容器 dsd 也满足它，但它是"0.6s 后出 7 骷髅"的容器（同函数 316 行就有专门分支）⇒ 被误当炸弹，而 `TimedExplosiveData` 硬读 `collisionRadius` | 把路由抽成单一来源 `battle.is_death_bomb()` 并补上 `collisionRadius` 要求 ⇒ 炸弹卡从 4 张收敛为 **3 张真炸弹**（Balloon/GiantSkeleton/BombTower），容器回到容器分支（**顺带修好 SkeletonBalloon 的语义**） | `test_death_spawn_routing_data_invariant`（穷举全卡表：满足判定的卡必须能被 `TimedExplosiveData` 解析；且 `SkeletonBalloon` 必须**不**被判为炸弹） |
| **D** | `ZeroDivisionError: complex division by zero`（滚动弹 `battle.py:1613`） | 滚动弹按 `target_position - initial_position` 归一化行进方向，**起点=终点**时零向量除零。**同文件另两处同类归一化各自写了零向量守卫（击退 1602、推挤 3182），唯独这处漏了** | 抽 `battle.normalize_dir(dx,dy)`（零向量返回 `None`）做单一来源，3 处共用；零行程滚动按"已到终点"处理（走兵链后消亡，与 `roll_range` 终止分支一致） | `test_normalize_dir_zero_vector`（零向量 → `None`；数值等价于零也 → `None`；非零向量单位化正确） |

**共同模式（这是本次最有价值的产出）**：4 个 bug **全部**是"**同一个语义有 2~4 处实现，其中一处漏了**"——
- A：6 个 `take_damage` 定义、4 个漏；
- B：5 个类跳过 `Entity.__init__`、契约从未显式化；
- C：炸弹判定散落在"条件 + 数据键"两处、不一致；
- D：4 处方向归一化、1 处漏守卫。

⇒ 因此每个修复都**不是就地打补丁，而是抽单一来源 + 立可强制的契约**，并配**能失败的**测试（两个还做了阴性对照）。

## 3 实测（修复后）

> **修复后同参数 smoke 完整跑通**：`EXIT=0`，600 步 / 3 个评估点（0·300·600）/ **训练循环 1228.7 s** /
> 末点 `league_600.pkl` **60 局**。前四次尝试分别崩在 A、B、C、D 四处 —— **同一条命令现在跑到底**。
> 日志：`docs/train_run_smoke_2026-09-17.log`（**注意该文件被最后一次成功的运行覆盖**，失败 traceback 见本文件 §2 的指纹列表）。

| 项 | 结果 |
|---|---|
| 命令 | `python rl/run_league.py --mode run --config-name run_smoke5 --fresh --total-steps 600 --steps-per-eval 300 --n-eval-games 4 --eval-workers 2 --device cuda` |
| 多卡组 | 20 局/点 × 3 点，共 **60 局**，卡组来自 5 个 agent 池（推进流/防守反击流/自闭流/全 200/全随机）⇒ `/api/cardstats` 的「按卡组」表列数远大于 1（对照 solo 恒为 1） |
| 末点 Elo | `lockdown_flow 1579.1 / push_flow 1574.8 / random_deck 1538.9 / **main 1493.5** / all_decks 1424.7 / counter_flow 1388.9`（±78 噪声地板）⇒ **未训练 main 打不过随机脚本**，与 O3（"连局部最优都做不到"）一致 |
| dashboard 消费 | `--state runs/run_smoke5/league_state.json` ⇒ `/api/state` 6 agent + `elo_history`；`/api/replays` 列 3 个回放；`/api/cardstats?files=0` ⇒ `n_decks` = 该 run 的卡组数 |
| 另一类告警（**不是**崩溃，未处理） | `P1-20: validate 通过但引擎拒绝 BarbLog@x,y —— 掩码缺口`（`env_wrapper.py:640`）：**掩码与引擎判定不一致**，属【R13】范畴；多卡组（含 `BarbLog` 的牌组）才高频出现 ⇒ 应另立一条线处理 |

## 4 ⚠️ 必须先看的保留：run 模式的对手强度

**run 模式的 5 个对手全是 `ScriptedPolicy(mode="random")` ⇒ 它们只做"从合法动作里随机选一个"**
（`rl/opponents.py:103-119`，`mode` 字段在 `play()` 里**根本没被使用**）。具体后果：

| 维度 | `solo`（旧主线） | `--mode run`（新主线） |
|---|---|---|
| 卡组多样性 | ❌ 双方同一副牌（仅 `--deck-set four` 时 defend 槽换牌） | ✅ **5 个卡池 / 200 副牌 + 全随机** |
| 对手强度 | ✅ 含 `SelfDefenderPolicy`（会防守的脚本）+ 冻结副本 + PFSP 历史自身 | ❌ **全是 mask-随机**，不会针对性防守 |
| 自我对弈压力 | ✅ 打"过去自己"（D1 机制：frozen + hist PFSP + rand_anchor） | ❌ **没有这一路**（无 `solo_main_*` 池） |
| 绝对强度仪器 | 固定随机锚点（外生，但镜像牌） | 5 个**不学习**的脚本 agent ⇒ Elo 是**平稳**仪器，但对手很弱 ⇒ **很快打满、无 headroom** |

⇒ **如果目标是"让模型变强"，切 run 模式很可能是降级**（对手随机、无自我对弈）；
**如果目标是"多卡组泛化 / 数据多样性"，run 模式正合**。

**可选的第三条路（我没动，等你拍）**：**保留 solo 的对手池与自我对弈压力，只把对手的"卡组"随机化**
—— 即给 `frozen`/`hist` 槽的 `FollowerOpponent` 也接 `deck1_factory`（`env_wrapper.py:297-298` 的通用钩子已经存在），
我方仍是固定 X弩牌。改动小（`train_solo._new_episode_reset` 那一行 `1530-1535` 的条件放宽 + 给对手一个 deck pool），
但**必须先修下面这条同族问题**：

> **潜伏问题（已核实，今日无害）**：`FollowerOpponent` 是 **player-1**，它的信念应追踪 **player-0** 的卡组
> （类默认 `env.deck0`，`train_follower.py:63-64`），`run`/`flow` 都传 `deck0`；而 `train_solo.py` 的
> 4 个 `FollowerOpponent` 构造点传的是 `env.deck1`（`375-376`、`505`、`730`、`859`）。
> 在**镜像同牌**下两者逐位等价 ⇒ **今天没有任何影响**；一旦"对手换牌"就会让对手的信念先验指错。
> （`train_solo` 里 main 自己的信念传 `deck1` 是**对的**——main 是 player-0，追踪 player-1。所以这不是
> "train_solo 全传反了"，而是"FollowerOpponent 那 4 处该换 `deck0`"。）

## 5 dashboard 交付（本次已做，属用户第 ② 项要求）

| 项 | 内容 |
|---|---|
| **不再以胜率为曲线指标** | solo 视图的曲线改为**指标多选器**（`dashboard.py` 的 `SOLO_METRICS` 注册表）：默认 **行为指标**（接敌率 / 单边堆牌 / 平均圣水），另可切 防守链 / 部署量 / 回报 / critic / GRU 活力 / **外生对照**（`_controls_history` 的 baseline0·prev·rand）/ 以及「⚠ 自引用（禁读）」分组的胜率。落实 AGENTS §3 判读禁则。 |
| **多卡组对战的卡牌数量** | 卡牌统计新增「**按卡组**」矩阵：行 = 卡牌，列 = **卡组**（不是模型）。卡组指纹 = 卡名排序拼接（与顺序无关）；标签取 200 副天梯池的 archetype，未知卡组给短指纹；列标题悬停可见该卡组 8 张卡。数据 = 回放 `meta.decks` + 逐帧出牌归因（我方 `frames[].cards` / 对手 `frames[].opp_played`）；`side_games` = 出战"局×边"数。旧录像无 `meta.decks` ⇒ 空表 + 提示。 |
| 后端新增字段 | `build_solo_payload`：`controls_history` / `gates` / `mode`；`build_card_stats_payload`：`decks[]` / `n_decks` / `coverage.games_with_decks` / `coverage.deck_meta_partial` |
| 回归 | `scripts/check_dashboard_js.py`：node + DOM 桩 + **真实 payload** 跑内嵌 JS 渲染冒烟（13 项）。Windows python 下自动回退 `wsl.exe node`，脚本经 stdin 管道。实测 **13/13 PASS**；HTTP 层另验（`/api/solo`、`/api/cardstats`、`/api/state`） |

## 6 未决 / 待你拍板

| # | 问题 | 选项 |
|---|---|---|
| **P1** | **主线到底跑哪个模式** | (a) 就按你的选择跑 `run`（多卡组、对手弱、无自我对弈）(b) **solo + 对手卡组随机化**（保留 D1 压力 + 多卡组；需先修 §4 潜伏问题）(c) run + 给脚本 agent 换更强的策略（如把 `SelfDefenderPolicy` 接进 run） |
| P2 | run 模式的"历史补种"是**无效的**（无 past-self 池） | 若要 run 模式也有自我对弈压力，需要像 D1 那样给 run 加"本 run 快照池" |
| P3 | 本次修复是否触发全量 selftest | 按【R19】只跑了相关子集（`test_take_damage_signature_consistency` + 阴性对照）；`battle.py` 是**共享底层**（引擎），按 R19 例外条款**本应**跑全量 —— 我**没跑**，如实记：建议下次长跑前补一次全量 |
| P4 | `zzsmoke_run` / `zzsmoke_run2` / `run_smoke2` 三个 smoke 目录是否保留 | 默认保留作证据；确认后我清理 |
