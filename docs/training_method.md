# 训练方法文档

本文档说明**这个项目里的模型是怎么被训练出来的**：训练入口、每一个训练环节调用了哪个函数、
这些函数的参数是什么、超参数在哪里定义、评估与诊断怎么算、以及全部相关函数的索引。

**资料来源**：`src/clasher_new/rl/` 下的源码 + 逐文件函数级分析素材（`docs/_survey/parts/*.md`）。
**不引用**任何设计文档、计划文件或外部资料。

## 目录

- [0. 文档说明与方法](#0-文档说明与方法)
- [训练方法文档 · 第 A 部分：训练模式、数据流、网络与超参](#训练方法文档--第-a-部分训练模式数据流网络与超参)
  - [A.1 训练体系概览](#a1-训练体系概览)
  - [A.2 一次训练的数据流](#a2-一次训练的数据流)
  - [A.3 网络结构](#a3-网络结构)
  - [A.4 超参数全表](#a4-超参数全表)
  - [A.5 命令行与运行方式](#a5-命令行与运行方式)
  - [A.6 检查点 / 目录结构 / 恢复训练](#a6-检查点--目录结构--恢复训练)
  - [A.7 本部分待确认清单](#a7-本部分待确认清单)
- [训练方法文档 · 第 B 部分：训练循环、评估诊断与函数索引](#训练方法文档--第-b-部分训练循环评估诊断与函数索引)
  - [B.1 单次 PPO 更新的逐步过程](#b1-单次-ppo-更新的逐步过程)
  - [B.2 优势 / 回报计算（`compute_gae`）](#b2-优势--回报计算compute_gae)
  - [B.3 PPO 损失与优化（`update()` 内部逐步骤）](#b3-ppo-损失与优化update-内部逐步骤)
  - [B.4 对手与联赛](#b4-对手与联赛)
  - [B.5 评估与诊断口径](#b5-评估与诊断口径)
  - [B.6 辅助训练脚本](#b6-辅助训练脚本)
  - [B.7 推理期附加件](#b7-推理期附加件)
  - [B.8 全量函数索引](#b8-全量函数索引)
  - [B.9 本部分待确认清单](#b9-本部分待确认清单)
- [附录 A：待确认事项汇总](#附录-a待确认事项汇总)
- [附录 B：生成方式与可复现性](#附录-b生成方式与可复现性)

---

## 0. 文档说明与方法

### 0.1 覆盖范围

| 项 | 内容 |
|---|---|
| 训练相关源码 | `src/clasher_new/rl/` 全部 33 个 `.py` 文件（PPO、环境封装、策略网络、对手池、评估、诊断、辅助训练脚本） |
| 引擎接口 | `src/clasher_new/rl/env_wrapper.py` 调用的 `battle.py` / `action_mask.py` 接口（仅涉及训练接口部分） |
| 函数索引 | §B.8（本文件内，含「函数名 → 文件:行号」字母序快速索引） |
| 配套文档 | 《项目内容全解文档》`docs/full_code_reference.md`（逐文件全量函数说明）、《游戏引擎文档》`docs/game_engine.md` |

### 0.2 资料来源与「不编造」的保证

1. 全部内容来自**源码本身**：分析过程由若干子代理执行，每个子代理只读它被指派的源码文件，
   **禁止**阅读设计文档 / 计划文件 / README / `AGENTS.md`，**禁止**上网。
2. 每条机制描述与数值都带**行内来源标注**，格式 `（文件名:行号）`，可直接回源核对。
3. 读不懂、依赖模块未读到、行为只能推测的地方，一律显式写 **「待确认：…（原因：…）」**，并汇总到附录 A——**宁可留白，不许猜测**。
4. 函数清单是先用 Python `ast` 机械抽取（文件/函数名/签名/行号），再逐个补写作用与实现，
   因此**不会漏函数**；覆盖对账见《项目内容全解文档》§4。

### 0.3 阅读约定与索引方式

| 标记 | 含义 |
|---|---|
| `（xxx.py:123）` | 该结论的源码出处：文件与行号 |
| **待确认** | 未能从源码确认，不得当作事实引用 |
| `A.x` / `B.x` | 章节号；A 与 B 是本文档的两个部分（编号保留 A/B 前缀，以保证文中相互引用不会错位）|

**怎么找函数**：按 `Ctrl+F` 搜函数名即可；先用 §B.8 的字母序快速索引定位 `文件:行号`。

---

## 训练方法文档 · 第 A 部分：训练模式、数据流、网络与超参

> **本文的全部事实性陈述均来自以下源码**（未引用任何设计文档、README 或联网内容）：
> `src/clasher_new/rl/config.py`、`run_league.py`、`train_solo.py`、`ppo.py`、`follower.py`、
> `env_wrapper.py`、`observation.py`，以及入口脚本 `start_rl.bat`、`start_training.bat`、
> `scripts/rl/run_league.py`。
> 行内标注格式 `（rl/xxx.py:NNN）` 指该文件的第 NNN 行（`rl/` = `src/clasher_new/rl/`）。
> 无法从源码确认的内容一律标「**待确认**」并说明原因。
> 参数量、`cnn_out`、`PLAN_DIM`、`belief_dim` 等数值由**实例化 `FollowerPolicy` 实算**得到
> （方法见 A.3.4），非抄自文档。

---

### A.1 训练体系概览

#### A.1.1 三种模式

| 模式 | CLI 取值 | 语义（源码原义） | 入口函数 | 源码位置 |
|---|---|---|---|--- |
| **solo** | `--mode solo` | 单人自对弈：单模型 `main`（`FollowerPolicy`+`PPOTrainer`），双方同副固定卡组镜像；对手 = main 的周期冻结副本 + 对手池；无联赛、不写 Elo/PFSP/`league_state.json`；周期评估写 `solo_state.json` | `run_solo(cfg, resume=..., record_replays=...)` | `rl/train_solo.py:1039`；分派 `rl/run_league.py:1365-1369` |
| **run** | `--mode run`（默认） | 联赛主循环：同时维护 5 个模型槽位（`main` + 三分类×3/all_decks + 全随机），PPO 训练 main → PFSP 采对手 → 周期全轮转评估 → 逐局 Elo → 状态持久化 | `run_league(cfg, resume=..., record_replays=...)` | `rl/run_league.py:1143`；分派 `rl/run_league.py:1371` |
| **flow** | `--mode flow` | 全配对分流派联赛（一次 148,800 局） | `rl.flow_league.run_flow(cfg, resume=..., n_random_decks=...)` | 调用点 `rl/run_league.py:1351-1354`；模块 docstring 亦声明 `rl/run_league.py:10` |
| flow 扫描 | `--mode flow-sweep-stream` / `flow-sweep-games5` | 缩小 10× 池的数据效率 A/B | `rl.flow_league.run_flow_sweep(cfg, strategy=..., n_runs=..., pool_scale=..., eval_games=...)` | `rl/run_league.py:1356-1363` |
| eval（非训练） | `--mode eval` | 保存策略轮转对战（换边 + 三态 + 逐局 Elo） | `evaluate_league(policies, kinds, n_games, seed, hidden_dim, max_steps, device)` | `rl/run_league.py:510`；分派 `1298-1304` |

**run 模式内部的三种执行体**（由 `cfg.n_envs` 与 `cfg.parallel` 选择，`rl/run_league.py:1143-1149`）：

| 条件 | 执行体 | 源码 |
|---|---|--- |
| `n_envs <= 1` | `_run_single`（单 env 主循环，旧行为） | `rl/run_league.py:697` |
| `n_envs > 1` 且 `parallel == "proc"` | `_run_vec`（单进程批量化：`act_parallel` + batch PPO） | `rl/run_league.py:803` |
| `n_envs > 1` 且 `parallel != "proc"`（默认 `"mp"`） | `_run_mp`（跨进程 worker，主进程批量 GPU 推理） | `rl/run_league.py:942` |

`n_envs>1` 时 `_run_mp` 若 worker 启动抛 `OSError`，**降级回 `_run_single`**（`rl/run_league.py:988-994`）。

#### A.1.2 模式之间共享的代码

| 被共用的文件 / 对象 | 共享方式 | 证据 |
|---|---|--- |
| `rl/ppo.py`（`PPOTrainer`、`compute_gae`、`ReturnScaler`） | `run_league` / `flow_league` / `train_follower` / `train_prophet` / `train_solo` 共用；类 docstring 明确写"默认行为不得变" | `rl/ppo.py:105-108`；solo 侧 import `rl/train_solo.py:37`；run 侧 import `rl/run_league.py:59` |
| `rl/env_wrapper.py`（`RLEnv`） | run 与 solo 共用；差异只在构造参数（卡组/奖励/等级） | `rl/run_league.py:579-581`（run：`RLEnv(opponent=None, seed, reward_weights, card_level)`）；`rl/train_solo.py:157-161`（solo：额外传 `deck0/deck1` 同副镜像） |
| `rl/follower.py`（`FollowerPolicy`/`save_checkpoint`/`load_checkpoint`） | 三种模式共用 | `rl/run_league.py:55`、`rl/train_solo.py:36` |
| `rl/config.py`（`TrainConfig`/`reward_to_env`） | 三种模式共用同一配置对象与同一奖励换算 | `rl/run_league.py:65`、`rl/train_solo.py:39` |
| `rl/observation.py`（`observe`/`hidden_labels`） | `RLEnv.observe` / `get_hidden_state` 唯一实现 | `rl/env_wrapper.py:32,390-394` |
| `rl/belief.py`、`rl/belief_planner.py`、`rl/prophet.py`、`rl/plan_space.py` | 三种模式的观测/信念/计划通道共用 | `rl/run_league.py:53-56`、`rl/train_solo.py:32-35` |
| `rl/run_league.py` 的若干工具 | **solo 反向 import run_league**：`resolve_device, _bundle_cards, LeagueGameRecorder, _stall_probe, STALL_WINDOW, _load_run_state, timeout_winner, overtime_open, settle_stall` | `rl/train_solo.py:42-44` |
| `rl/replay.py`（`save_league_replays`） | run 与 solo 写回放共用 | `rl/run_league.py:67`、`rl/train_solo.py:45` |
| `rl/pfsp.py` | solo 对手池用 PFSP；run 联赛用 `league.sample_opponent` | `rl/train_solo.py:41,360-362`；`rl/run_league.py:657` |

> **待确认**：flow 模式内部的数据流、模型槽位与评估口径（原因：`rl/flow_league.py` 不在本部分的允许阅读清单内，仅能从 `run_league.py` 看到其入口签名与参数）。需要确认：读 `rl/flow_league.py` 的 `run_flow` / `run_flow_sweep` 全文。

---

### A.2 一次训练的数据流

以 **solo 主循环**为主干（solo 是当前唯一带完整「观测→信念→计划→策略→奖励→GAE→PPO」链路的模式），
凡 run 模式有差异处单列。全部步骤都给出函数名 + 行号 + 张量形状。

#### A.2.0 逐步总览

| # | 阶段 | 函数 | 源码位置 | 输出（形状/类型） |
|---|---|---|---|--- |
| 1 | 重置环境 | `RLEnv.reset(seed=...)` / `self.observe(0)` | `env_wrapper.py:348-386`、`390-391` | `obs: dict`、`info: {}` |
| 2 | 观测构造 | `observation.observe(battle, player_id)` | `observation.py:87` | `grid (32,18,15) f32`、`hand (5,) i32`、`elixir (1,) f32`、`next_card (1,) i32`、`time (1,) f32` |
| 3 | 计划生成 | `ProphetPlanner.plan` 或 `BeliefPlanner.plan` → `plan.to_vector()` | `train_solo.py:1578-1581` | `plan_vec (PLAN_DIM,)`，`PLAN_DIM=58`（实算，`follower.py:26` 导入） |
| 4 | 信念编码 | `BeliefInference.encode(obs, None)` | `train_solo.py:1582` | `belief_tok (563,)`（实算，见 A.3.4） |
| 5 | 策略前向 + 采样 | `FollowerPolicy.act(obs, belief_tok, plan_vec, get_mask, hidden, deterministic=False)` | `follower.py:433-484`；调用 `train_solo.py:1584-1586` | `bundle: ActionBundle`、`lp: float`、`val: float`、`hidden (1,128)`、`masks: list[dict]` |
| 6 | 环境推进 | `RLEnv.step(bundle)` | `env_wrapper.py:598-723` | `obs2 dict`、`reward float`、`term bool`、`trunc bool`（恒 False）、`info dict` |
| 7 | 奖励计算 | `compute_reward(...)`（含 `_phase_weights`、`_per_tower_norm_dmg`、`tower_value_mult`） | `env_wrapper.py:181-263`、`164-178`、`131-161`、`101-118` | `reward: float` |
| 8 | 轨迹缓冲 | `ep_obs/ep_belief/... append` | `train_solo.py:1589-1595` | 每个 episode 一个 `list`，长度 = 决策帧数 |
| 9 | 终端/截断补结算 | `timeout_winner` / `settle_stall` + `win_bonus/lose_penalty/draw_penalty` | `train_solo.py:1599-1613`、`1545-1562`；`run_league.py:179,240` | 修改 `ep_rew[-1]`；置 `ep_trunc[-1]=True` |
| 10 | GAE | `PPOTrainer.compute_gae(rew, val, term, gamma, lam, truncated, last_value)`（solo 经 `_gae_pair` 包装） | `ppo.py:167-192`；`train_solo.py:1221-1238` | `adv (T,) f32`、`returns (T,) f32` |
| 11 | 入 transition 池 | `transitions.append({...})` | `train_solo.py:1625-1633` | `list[dict]`，键见 A.2.6 |
| 12 | PPO 更新 | `PPOTrainer.update(batch, adv_alt=...)` | `ppo.py:194-356`；调用 `train_solo.py:1653` | `stats: dict` |
| 13 | 优势归一化 | `update` 内联 | `ppo.py:220-233` | `advs (n,) f32` |
| 14 | 价值损失量纲 | `ReturnScaler.update/scale` + `v_loss = v_mse / v_scale²` | `ppo.py:41-83`、`241-248`、`392` | 标量 `v_scale` |
| 15 | 损失与反传 | `_loss_pass` → `_apply_grad` → `_update_epochs` | `ppo.py:361-407`、`409-465`、`477-547` | `opt.step()` 每次小批一次 |
| 16 | 评估 / 落盘 | `eval_and_write` / `anchor_point` / `_persist` | `train_solo.py:1268-1408`、`1454-1479`、`1410-1452` | `solo_state.json`、`solo_main.pt`、`league_<step>.pkl` |

#### A.2.1 `RLEnv.reset()` 与观测构造

- `reset()`：按 `deck0_factory`/`deck1_factory` 采样卡组 → `self._rng.shuffle(deck0)` / `shuffle(deck1)`（**原地链式洗牌**：`deck0 = list(self.deck0)` → `shuffle` → `self.deck0 = deck0`）→ 建 `battle.BattleState` → `update_player_hp()` 同步塔血 → 记录本局塔血基准（`_blue_hps_max/_red_hps_max/_blue_towers_max/_red_towers_max`）→ 清掩码缓存 `_mask_fp/_mask_cells` → 清资源账 `_v_share/_active_v` → 设 `_seen_max_id` → 返回 `self.observe(0), {}`（`env_wrapper.py:348-386`）。
- `observe(player_id)` 直接转发 `observation.observe(self.battle, player_id)`（`env_wrapper.py:390-391`）。

**`observation.observe` 的输出字段与维度**（`observation.py:87-144`；常量 `GRID_H, GRID_W, GRID_C = 32, 18, 15` 在 `observation.py:83-84`）：

| 字段 | 形状 | dtype | 构造方式 | 行号 |
|---|---|---|---|--- |
| `grid` | `(32, 18, 15)` | float32 | 全零网格；对 `battle.entities` 中存活且名字在 `ENTITY_NAMES` 里的实体，`obs[y][x] = 15 维属性向量`（y 为第一轴） | `observation.py:89-90,124-130` |
| `hand` | `(5,)` | int32 | `p.cycle[:5]` 的 `ENTITY_NAMES.index`（未登记名 → 0） | `observation.py:133-136` |
| `elixir` | `(1,)` | float32 | `p.elixir` | `observation.py:141` |
| `next_card` | `(1,)` | int32 | `ENTITY_NAMES.index(p.cycle[4])`（未登记 → 0） | `observation.py:137,142` |
| `time` | `(1,)` | float32 | `battle.time` | `observation.py:143` |

**grid 的 15 个通道**（逐元素顺序，`observation.py:125-129`）：

| 通道 idx | 名称 | 计算 | 行号 |
|---|---|---|--- |
| 0 | `entity_id` | `ENTITY_NAMES.index(each.name)` | `95,126` |
| 1 | `is_opponent` | `each.player != player_id`（己方恒 0） | `100,126` |
| 2 | `elixir` | `getattr(d,"elixir",0) or 0` | `105,126` |
| 3 | `card_type` | `CARD_TYPES.index(type)`，`_TYPE_ALIAS` 把 `area_effect/projectile/bomb` 折叠为 `spell`，未知折叠为 `character` | `75,81,96-99,126` |
| 4 | `speed` | `getattr(d,"speed",0) or 0` | `109,126` |
| 5 | `is_air` | `int(bool(d.is_air_unit))` | `106,126` |
| 6 | `attacks_ground` | `int(bool(d.attack_ground))` | `107,126` |
| 7 | `attacks_air` | `int(bool(d.attack_air))` | `108,126` |
| 8 | `hp_left` | `log(each.hp)/10`，hp==0 → 0 | `110,126` |
| 9 | `hp_percentage` | `each.hp / d.hp`，max_hp==0 → 0 | `111-112,126` |
| 10 | `hit_speed` | `getattr(d,"hit_speed",0) or 0` | `113,126` |
| 11 | `attack_range` | `d.range / 3` | `114,126` |
| 12 | `sight_range` | `d.sight_range / 3` | `115,126` |
| 13 | `damage` | `d.damage / 200` | `116,126` |
| 14 | `projectile_damage` | `projectile_data.damage / 200`（无 → 0） | `117-118,126` |

- **镜像**：`player_id == 1` 时 `x = 17 - x`、`y = 31 - y`（`observation.py:121-123`）。
- **越界丢弃**：仅当 `0 <= x < 18 and 0 <= y < 32` 才写入（`observation.py:124`）。
- `hand` 用**卡名**空间、`grid[...,0]` 用**实体名**空间，两套词表同属 `ENTITY_NAMES`（177 项；`observation.py:25-73`，长度由 `follower.py:29` 的 `NUM_ENTITY` 间接使用）。
- 观测空间声明：`gym.spaces.Dict`，`hand/next_card` 上界 `_NUM_IDS = len(ENTITY_NAMES) - 1`（`env_wrapper.py:265,329-335`）。
- 特权标签另走 `hidden_labels`（`observation.py:147-172`），只经 `info["hidden"]` 暴露（`env_wrapper.py:721-722`）。

**run 模式的差异**：`_run_single` 在同一函数里直接 `obs, _ = env.reset()` 后 `belief.reset(env.deck1)`（`rl/run_league.py:721-722`）；`_run_mp` 模式下 `reset` 发生在 worker 进程内（`rl/run_league.py:1033`）。

#### A.2.2 计划与信念 token

| 步骤 | 函数 | 行号 | 说明 |
|---|---|---|--- |
| 计划二选一 | `use_prophet = rng.random() < 0.3` | `train_solo.py:1578`（常量 `_SOLO_PROPHET_PROB = 0.3` 在 `train_solo.py:129`；run 模式同 0.3 在 `run_league.py:732`） | 30% 用 `ProphetPlanner.plan(env.get_prophet_state())`，70% 用 `BeliefPlanner.plan(env.battle, belief.state(), obs)` |
| 计划向量化 | `plan.to_vector()` | `train_solo.py:1581` | 维度 `PLAN_DIM`（`follower.py:26` 导入；实算 58） |
| 信念编码 | `belief.encode(obs, None)` | `train_solo.py:1582` | 一维向量，长度 = `belief_dim`（solo 下 563，见 A.3.4） |
| 信念更新 | `belief.update(obs2, info.get("opp_played"))` | `train_solo.py:1596` | 对手出牌来自 `info["opp_played"]`（结构化列表，`env_wrapper.py:490-522,716`） |
| 信念重置 | `belief.reset(env.deck1)` | `train_solo.py:1534` | 每局重置，先验卡组 = 对手本局卡组 |

#### A.2.3 策略前向：`FollowerPolicy.act` 的每一步

`act()` 全程 `torch.no_grad()`（`follower.py:439`）。单条路径 `_encode_parts`（`follower.py:268-298`）：

| 步 | 操作 | 张量形状 | 行号 |
|---|---|---|--- |
| 1 | `grid` → tensor + `unsqueeze(0)` | `(1,32,18,15)` | `271` |
| 2 | `hand` → long tensor | `(1,5)` | `272` |
| 3 | `elixir` / `time` / `next_card`，`next_card` 除以 12.0 | 各 `(1,1)` | `273-275` |
| 4 | `card_ids = grid[...,0]` → `entity_emb` | `card_ids (1,32,18)` → `card_vecs (1,32,18,8)` | `277-278`；embedding 维度 8 见 `192` |
| 5 | `rest = grid[...,1:]` | `(1,32,18,14)` | `279` |
| 6 | `card_type = rest[...,2]` → `one_hot(num_classes=4)` | `(1,32,18,4)` | `280-281` |
| 7 | `cat([rest, card_vecs, card_type_oh], dim=-1)` | `(1,32,18,26)`（`cnn_in = 14+8+4`） | `282`；`cnn_in` 定义在 `193` |
| 8 | `permute(0,3,1,2)` | `(1,26,32,18)` | `283` |
| 9 | `self.cnn(x)`：`Conv2d(26→32,k3,p1)+ReLU` → `Conv2d(32→64,k3,s2,p1)+ReLU` → `Conv2d(64→64,k3,s2,p1)+ReLU` → `Flatten` | `grid_feat (1,2560)` | `194-199`（结构）、`284`（调用）；`cnn_out` 实算 2560，亦由 `dummy` 前向探测于 `200-203` |
| 10 | `self.grid_ln(grid_feat)` | `(1,2560)` | `234,285` |
| 11 | `hand_feat = entity_emb(hand).reshape(1,-1)` | `(1,40)`（`hand_dim = 5*8`） | `205,287` |
| 12 | `scalar = cat([elixir, time, next_card])` | `(1,3)`（`scalar_dim = 3`） | `206,288` |
| 13 | `plan_v` → `plan_mlp`（`Linear(plan_dim→64)+ReLU`） | `(1,58)` → `plan_f (1,64)` | `207,290,292` |
| 14 | `belief_v` → `belief_mlp`（`Linear(belief_dim→64)+ReLU`） | `(1,563)` → `belief_f (1,64)` | `208,291,293` |
| 15 | `fused = cat([grid_feat, hand_feat, scalar, plan_f, belief_f])` | `(1,2731)`（`enc_dim = 2560+40+3+64+64`） | `209,295` |
| 16 | `enc = enc_ln(relu(enc_fc(fused)))` | `(1,128)` | `210,218,297` |
| 17 | `h = gru_cell(enc, hidden)`（hidden 缺省为零向量 `(1,128)`，传入时先 `.detach()`） | `(1,128)` | `235,441-443` |
| 18 | `value = self._value_from(enc, h, fused)` → `float(...)` | 标量 | `303-314,444` |

批量路径 `_encode_batch_parts` 与单条逐位同口径（`follower.py:397-428`），张量为 `(N,...)`；
`act_parallel` 走批量并在每个 decoder 步对活跃子集批量前向（`follower.py:504-608`）。
`evaluate_batch`（PPO 重放）与 `evaluate`（单条重放）同口径（`follower.py:610-702`、`713-766`）。

#### A.2.4 动作解码（autoregressive bundle head）

决策空间常量：`K_MAX = 4`（`follower.py:24` 自 `rl.action_bundle` 导入）、
`ABILITY_IDX = K_MAX = 4`、`STOP_IDX = K_MAX+1 = 5`、`NUM_SLOT_OPTIONS = K_MAX+2 = 6`（`follower.py:37-39`；实算一致）。

`act()` 的解码循环（`follower.py:450-483`，最多 `K_MAX+2 = 6` 步）：

| 步 | 操作 | 形状 | 行号 |
|---|---|---|--- |
| 1 | `mask = get_mask(bundle)`，压入 `masks` 序列（供 PPO 重放） | `dict`（`slots (4,)`、`cells (4,32,18)`、`ability_legal`、`used_slots`、`at_cap`、`any_legal`） | `451-452`；掩码契约 `env_wrapper.py:476-486` |
| 2 | `_slot_mask_tensor(mask)`：`sm[:4]=mask["slots"]`、`sm[4]=ability_legal`、`sm[5]=1`；`at_cap` 时只放行 STOP | `(6,)` | `316-330` |
| 3 | `slot_logits = slot_head(h) + slot_bias`，掩码位置填 `-1e9` | `(1,6)`；`slot_head: Linear(128→6)` | `454-455`；头定义 `237` |
| 4 | `Categorical(logits=log_softmax(slot_logits))`；`deterministic` → `argmax`，否则 `sample` | 标量 `option` | `456-461` |
| 5 | 累加 `logprob += slot_dist.log_prob(option)` | 标量 | `462` |
| 6 | `option == STOP_IDX` → `break` | — | `464-465` |
| 7 | `option == ABILITY_IDX` → `bundle.add_ability()`，`h = _sub_update(h, ABILITY_IDX)`，`continue` | `h (1,128)` | `466-469`；`_sub_update` 在 `394-395` |
| 8 | 否则：`cell_logits = cell_head(h).view(1,32,18) + cell_bias`，掩码填 `-1e9`，`Categorical` 采样 | `(1,576)` → `(1,32,18)` → 平坦 `(1,576)` | `471-479`；`cell_head: Linear(128→576)` 定义在 `238` |
| 9 | `x = cell % 18`、`y = cell // 18`；`bundle.add(option+1, x, y)`；`h = _sub_update(h, option, x, y)` | — | `481-483` |
| 10 | 返回 `(bundle, logprob, value, h.detach(), masks)` | `hidden (1,128)` | `484` |

**plan 软偏置**（只加 logit，不硬禁；rollout 与 PPO 重放共用同一函数，`follower.py:332-376`）：
`PLAN_CARD_BIAS = 0.8`、`PLAN_HOLD_BIAS = 2.5`、`PLAN_REGION_BIAS = 0.8`、`PLAN_REGION_R = 2`（`follower.py:42-45`）。
`player1` 侧（`FollowerOpponent`）通过 `plan_biases_enabled=False` 关闭（`follower.py:258-261,341-343`）。

**子动作特征**：`sub_emb: Linear(NUM_SLOT_OPTIONS+2 = 8 → 128)`，输入为 option one-hot + `(x/18, y/32)`（`follower.py:250,387-392`）。

#### A.2.5 `env.step()` 与奖励

`RLEnv.step(action_bundle)`（`env_wrapper.py:598-723`）的固定顺序：

1. **取帧前快照**：三塔合计血、per-tower 血量 `[king, left, right]`、剩余皇冠数 `3 - crown_count`、双方圣水、资源账 `_active_v`、单位 hp 图（`602-614`）。
2. **整包校验**：`validate_bundle(battle, 0, bundle)` → `(ok, reason, resolved)`；不 ok 则 `invalid_count = 1` 且**整包不提交**（`617-621`）。
3. **整包提交**：按决策时刻手牌解析后依次 `use_ability` / `deploy_card`；引擎级拒绝也计入 `invalid_count` 并发 `RuntimeWarning`（`622-642`）。
4. **对手动作**：`_run_opponent()` → 返回 `played` 列表 `[{"card","x","y"}]`（`644-645`；实现 `490-522`）。
5. **统一推进决策帧**：`for _ in range(self.decision_frames): battle.step(dt)`（`648-657`；`decision_frames` 默认 30、`dt = 1/60`，构造签名 `301-302`）。
6. **死亡注销**：`_collect_deaths()` 清掉死亡实体的部署份额（`660`；实现 `583-591`）。
7. **两段价格**：`_phase_weights(reward_weights, battle.time)` → `(tower_opp, tower_self, edw_coef)`，切点在 `PHASE_SWITCH_S = 120.0`（`672-676`；实现 `164-178`、`89`）。
8. **奖励**：`compute_reward(...)`（`678-697`；实现 `181-263`）。
9. **单位受伤 shaping**：`unit_dmg_k * (hurt[1] - hurt[0])`（`699-709`）。
10. **返回**：`(self.observe(0), reward, terminated, False, info)`（`723`）——**`truncated` 恒为 `False`**，截断由训练环自行标记（`train_solo.py:1614-1619` 明确写"env 恒返回 trunc=False"）。

**`compute_reward` 的组成**（`env_wrapper.py:209-263`）：

| 项 | 公式 | 行号 |
|---|---|--- |
| 皇冠差 | `crown_weight * (red_left_old - red_left_new) - crown_lose_weight * (blue_left_old - blue_left_new)` | `239-240` |
| 塔伤（敌） | `+ tower_dmg_opp * red_dmg` | `241` |
| 塔伤（己） | `- tower_dmg_self * blue_dmg` | `242` |
| 圣水 shaping | `+ elixir_bonus * my_elixir_after` | `243` |
| 资源账 Δ费差 | `+ edw * ((me_a - op_a) - (me_b - op_b))`，其中 `me/op = 手牌圣水 + 场上部署份额` | `245-253` |
| 胜负 | `winner==0` → `+win_bonus`；`winner is not None` → `-lose_penalty`；`game_over` 且无胜者 → `-draw_penalty` | `254-260` |
| 非法动作 | `- invalid_penalty * invalid_count` | `261-262` |

- **塔血归一化与差异化定价**：`normalize_tower_dmg` 为真时，per-tower 参数齐备 → `_per_tower_norm_dmg`（`131-161`），否则退化为聚合口径 `dmg * (_TOWER_HP_ANCHOR / hps_max)`（`226-237`）；锚点 `_TOWER_HP_ANCHOR = 2*3052 + 4824 = 10928`（`65-84`）；凹形溢价 `tower_value_mult = 1 + k*(1-ratio)²`，王塔两公主塔存活时再 `* king_gate`（`101-118`）。
- 部署份额记账：`_deploy_ledger`（`561-581`），spell 产物份额记 0 防双算（`568-569`）。

#### A.2.6 轨迹缓冲与终端结算

**solo 的缓冲结构**（每局一组 list，`train_solo.py:1506-1507`）：
`ep_obs / ep_belief / ep_plan / ep_bundle / ep_lp / ep_val / ep_rew / ep_term / ep_trunc / ep_masks / ep_init`。

- 逐帧写入：`train_solo.py:1589-1595`（`ep_init` 保存**进入本步时的隐状态** `init_hidden = hidden`，`1583`；PPO 重放用它保证 `ratio` 是有效 IS 比，`ppo.py:7`）。
- **终端补结算**（截断/僵局早停时）：`timeout_winner(battle)`（皇冠多者胜；皇冠相同则存活塔最低血量%更低者输；`rl/run_league.py:179-210`）→ `win_bonus` / `lose_penalty`；无胜者 → `-_draw_penalty(cfg)`（`train_solo.py:1604-1613`）。
- **训练环僵局早停**：`cfg.train_stall_stop` 且每 `STALL_WINDOW = 10` 步探针一次，连续 `STALL_LIMIT = 10` 次零塔血变化 → 判平（`train_solo.py:1545-1547`；常量与探针 `rl/run_league.py:145-146,252-265`）；早停结算用 `settle_stall(battle, cfg.stall_draw_margin)`（`train_solo.py:1552`；实现 `rl/run_league.py:213-249`）。
- **截断标记**：仅当 `truncated and virt is None` 才置 `ep_trunc[-1] = True` 并算 `last_val = main.value(...)`，否则 `last_val = 0.0`（`train_solo.py:1618-1622`）。
- **入池**：`transitions.append({obs, belief, plan, bundle, old_logprob, adv, returns, masks, init_hidden[, adv_const, adv_gap]})`（`train_solo.py:1625-1633`）；solo 额外带 `adv_const`/`adv_gap`（critic 惰性检验用，`train_solo.py:1570-1572`）。
- **局间重置**：`_new_episode_reset(winner)` → `_probe["games"] += 1` → `opp_pool.record(winner)` → `opp_pool.sample()` → `env.opponent = side` → `env.reset()` → `belief.reset(...)` → 清空全部 `ep_*`（`train_solo.py:1513-1539`；`nonlocal` 显式列出全部缓冲，注释指出漏 `nonlocal` 会导致缓冲泄漏事故，`1519-1524`）。

**run 模式的缓冲**：`_run_single` 用同构的 `ep_obs/...`（`rl/run_league.py:725-726`）；`_run_vec`/`_run_mp` 用 **per-env dict** `{"obs","belief","plan","bundle","lp","val","rew","term","trunc","masks","init"(+"winner")}`（`rl/run_league.py:837-841,1027-1029`）。
**run 模式无训练环僵局早停**：`grep -rn train_stall_stop rl/` 的唯一消费点是 `train_solo.py:1545`；
但**评估侧**早停两种模式都有（`_run_side0` 用 `_stall_probe`/`STALL_WINDOW`，`rl/run_league.py:288-292`；
`eval_solo` 同，`train_solo.py:745-749`）。

#### A.2.7 GAE

`PPOTrainer.compute_gae(rewards, values, dones, gamma, lam, truncated=None, last_value=0.0)`（`ppo.py:167-192`）：

- `delta = rewards[t] + gamma * next_val - values[t]`（`188`）；`gae = delta + gamma * lam * (0 if 末步或 done else gae)`（`189`）；`adv[t] = gae`（`190`）。
- `next_val`：末步且 `truncated[t] and not dones[t]` → `last_value`（bootstrap）；否则末步 → 0；非末步 `dones[t]` → 0，否则 `values[t+1]`（`181-187`）。
- 返回 `adv (T,) f32` 与 `returns = adv + values`（`191-192`）。
- 调用点：solo `train_solo.py:1228-1230`（`_gae_pair`）；run `rl/run_league.py:767-768`（single）、`898-900`（vec）、`1091-1093`（mp）。
- solo 额外算 `adv_const`（把 V 换成标量 `c = ppo.ret_scaler.mean` 后重算 GAE）与 `level_gap = |mean(V) - c|`（`train_solo.py:1231-1238`）。

#### A.2.8 PPO 更新

`PPOTrainer.update(transitions, ent_coef=None, adv_norm=None, adv_alt=None)`（`ppo.py:194`）：

1. `adv_raw` → 按 `adv_norm` 归一化：`scale` 只除以批 std（不中心化）、`none` 原样、`batch` 中心化并除以 std；未知值抛 `ValueError`（`220-233`）。
2. `value_norm == "running"` → `ret_scaler.update(rets)` 后 `v_scale = s`；`none/""/None` → `1.0`；否则抛错（`241-248`）。
3. **旧分支判定**：`legacy = (n_epochs <= 1 and not shuffle and (minibatch_size <= 0 or minibatch_size >= n))`（`251-252`）。
   - `legacy` → `_loss_pass(..., reduction="sum")` + `_apply_grad`，**一次 `opt.step()`**（`299-316`），`ppo_epochs=1`、`grad_steps=1`。
   - 非 legacy → 先 `no_grad` 前向算**更新前 EV**（`326-336`），再 `_update_epochs`（`337-339`），并把 in-sample EV 另存 `explained_variance_insample`（`341-342`）。
4. 损失（`_loss_pass`，`361-407`）：`ratio = exp(lp_new - old)`；`surr = min(ratio*A, clip(ratio,1±clip)*A)`；`p_loss = -mean/sum(surr)`；`v_mse = mse(value, rets)`；`v_loss = v_mse / v_scale²`；`loss = p_loss + vf_coef*v_loss - coef*ent`（`379-393`）。
5. 每次 `opt.step()` 前 `clip_grad_norm_(params, max_grad_norm)`（`_apply_grad`，`462-464`）。
6. `_update_epochs` 的多轮/小批/打乱实现：`_plan_batches` 每轮生成顺序（`467-475`），统计口径为"`policy_loss/value_loss/entropy` 按样本数加权跨轮平均；`ratio_mean/clip_frac/grad_norm` 只取最后一轮；EV 取最后一轮逐帧池化"（`477-547`）。
7. 入池裁剪：solo 取 `transitions[:cfg.batch_size]` 后 `transitions = transitions[cfg.batch_size:]`（`train_solo.py:1640-1641,1677`）；run 三种执行体同形（`rl/run_league.py:782-784,916-918,1108-1110`）。
8. `_probe["ev_pairs"]` 累积每次 update 的**更新前** `(value, return)`，评估时**池化**算一次 EV（`train_solo.py:1679-1686`，含 20 万帧内存上界 `1685-1686`）。

**EV 定义**：`EV = 1 - MSE(v,R)/Var(R)`；`Var(R) <= 1e-12` 或样本不足返回 `0.0`（`ppo.py:144-165`）。

---

### A.3 网络结构

唯一策略网络 = `FollowerPolicy`（`follower.py:154`）。构造参数：`hidden=256`（默认）、`plan_dim`、`belief_dim`、`num_entity`、`stop_logit_bias=-1.0`、`value_bypass=False`、`value_independent=False`（`follower.py:155-156`）。

> ⚠️ `plan_dim`/`belief_dim` **无默认值**，两者为 `None` 直接抛 `ValueError`（`follower.py:183-184`）。

#### A.3.1 逐层表（`hidden=128`，即训练实际取值）

| # | 层名 | 类型 | 输入维度 | 输出维度 | 激活 | 是否共享 | 源码 |
|---|---|---|---|---|---|---|--- |
| 1 | `entity_emb` | `nn.Embedding(177, 8)` | 标量 id | 8 | 无 | 共享（grid id 通道 + hand） | `follower.py:192` |
| 2 | `cnn.0` | `Conv2d(26,32,3,padding=1)` | `(N,26,32,18)` | `(N,32,32,18)` | ReLU | 共享 | `194-195` |
| 3 | `cnn.2` | `Conv2d(32,64,3,stride=2,padding=1)` | `(N,32,32,18)` | `(N,64,16,9)` | ReLU | 共享 | `196` |
| 4 | `cnn.4` | `Conv2d(64,64,3,stride=2,padding=1)` | `(N,64,16,9)` | `(N,64,8,4)` | ReLU | 共享 | `197` |
| 5 | `cnn.5` | `Flatten` | `(N,64,8,4)` | `(N,2560)` | — | 共享 | `198`；`cnn_out` 探测于 `200-203` |
| 6 | `grid_ln` | `LayerNorm(2560)` | `(N,2560)` | `(N,2560)` | — | 共享 | `234` |
| 7 | `plan_mlp.0` | `Linear(plan_dim→64)` | `(N,58)` | `(N,64)` | ReLU | 共享 | `207` |
| 8 | `belief_mlp.0` | `Linear(belief_dim→64)` | `(N,563)` | `(N,64)` | ReLU | 共享 | `208` |
| 9 | `enc_fc` | `Linear(2731→hidden)` | `(N,2731)` | `(N,128)` | ReLU（在 `297` 显式调用） | **共享（策略+价值）** | `210` |
| 10 | `enc_ln` | `LayerNorm(hidden)` | `(N,128)` | `(N,128)` | — | 共享 | `218` |
| 11 | `gru_cell` | `nn.GRUCell(hidden, hidden)` | `(N,128)` + `(N,128)` | `(N,128)` | 内部 tanh/sigmoid | **仅策略**（value 在 independent/bypass 下不走它） | `235` |
| 12 | `slot_head` | `Linear(hidden→NUM_SLOT_OPTIONS=6)` | `(N,128)` | `(N,6)` | softmax（采样时） | 策略支路 | `237` |
| 13 | `cell_head` | `Linear(hidden→32*18=576)` | `(N,128)` | `(N,576)`→reshape `(N,32,18)` | softmax（采样时） | 策略支路 | `238` |
| 14 | `sub_emb` | `Linear(NUM_SLOT_OPTIONS+2=8→hidden)` | `(N,8)` | `(N,128)` | 无 | 策略支路（子动作 → GRU） | `250` |
| 15 | `value_head` | `Linear(hidden→1)` | `(N,128)` | `(N,1)` | 无 | 价值支路 | `239` |
| 16 | `value_enc_fc` | `Linear(2731→hidden)`（**仅 `value_independent=True`**） | `(N,2731)` | `(N,128)` | ReLU（`310`） | 价值支路独占 | `245` |
| 17 | `value_enc_ln` | `LayerNorm(hidden)`（仅 `value_independent=True`） | `(N,128)` | `(N,128)` | — | 价值支路独占 | `246` |
| 18 | `value_head_mlp.0` | `Linear(hidden→max(32,hidden//2))`（仅 independent） | `(N,128)` | `(N,64)` | ReLU | 价值支路独占 | `247-249` |
| 19 | `value_head_mlp.2` | `Linear(64→1)`（仅 independent） | `(N,64)` | `(N,1)` | 无 | 价值支路独占 | `249` |

**其它结构常量**：`cnn_in = (GRID_C-1)+8+4 = 26`（`follower.py:193`）；`hand_dim = 5*8 = 40`（`205`）；`scalar_dim = 3`（`206`）；
`enc_dim = cnn_out + 40 + 3 + 64 + 64 = 2731`（`209`）；`ABILITY_IDX=4`/`STOP_IDX=5`/`NUM_SLOT_OPTIONS=6`（`37-39`）；
初始 STOP 偏置 `stop_logit_bias=-1.0` 加到 `slot_head.bias[STOP_IDX]`（`252-255`）。

#### A.3.2 价值支路 vs 策略支路

**价值支路三变体**（唯一实现 `_value_from`，`follower.py:303-314`；优先级 `independent > bypass > shared`，注释见 `236`）：

| 变体 | 触发条件 | 计算公式 | 价值支路吃到的张量 | 行号 |
|---|---|---|---|--- |
| `independent` | `value_independent=True`（无论 `value_bypass`） | `value_head_mlp(value_enc_ln(relu(value_enc_fc(fused))))` | `fused (N,2731)`，**不经过 GRU** | `309-311` |
| `bypass` | `value_independent=False` 且 `value_bypass=True` | `value_head(enc)` | `enc (N,128)`（post-LN 共享编码），跳过 GRU | `312-313` |
| `shared`（默认） | 两者皆 False | `value_head(h)` | GRU 隐状态 `h (N,128)` | `314` |

**策略支路**：`enc → gru_cell → h`，再由 `slot_head(h)` 与 `cell_head(h)` 出分布；每个已选子动作把
`(option one-hot, x/18, y/32)` 经 `sub_emb` 送回 `gru_cell` 更新 `h`（`follower.py:443,454,472,483`）。

**`value_head` 在 `value_independent=True` 下不参与前向**（`_value_from` 分支只走到 `value_head_mlp`，`309-311`）——即权重仍存在、仍会被保存/加载，但不被使用。

#### A.3.3 checkpoint 里的架构元数据

`save_checkpoint` 写 `{state_dict, plan_dim, belief_dim, hidden_dim, value_bypass, value_independent}`（`follower.py:55-64`）。
`load_checkpoint`：优先用元数据决定维度；显式传入与元数据不一致时**只告警不报错**（`value_bypass` `94-98`、`value_independent` `100-104`）；
缺 `enc_ln.*` / `grid_ln.*` 的旧 ckpt 显式告警"不可续训（要求 `--fresh`），只能当对照基线/对手池"（`144-150`）；
尾零兼容分支：`plan_mlp.0.weight`（`114-118`）、`belief_mlp.0.weight`（`119-122`）、`entity_emb.weight`（`123-128`）、`plan_mlp.0.bias`/`belief_mlp.0.bias`（`129-130`）；其余形状不匹配保持新初始化不崩（`131`）。

#### A.3.4 总参数量（实算）

| 配置 | 总参数量 | 说明 |
|---|---|--- |
| 共享 value 通路（`value_bypass=False, value_independent=False`），`hidden=128, plan_dim=58, belief_dim=563` | **634,735** | 含 `enc_fc.weight (128,2731)` 349,568、`cell_head.weight (576,128)` 73,728、`gru_cell.*` 98,304、`belief_mlp.0.weight (64,563)` 36,032 等 |
| `value_bypass=True`（默认 `value_independent=False`） | **634,735** | 只换接线，不新增参数（复用 `value_head`） |
| `value_independent=True` | **993,008** | 额外 `value_enc_fc.weight (128,2731)` 349,568 + `value_enc_ln` 256 + `value_head_mlp.0 (64,128)` 8,192 + `value_head_mlp.2 (1,64)` 64 等，净增 358,273 |

- **实算方法**：`FollowerPolicy(hidden=128, plan_dim=58, belief_dim=563, ...)` 实例化后 `sum(p.numel() for p in model.parameters())`。
- `PLAN_DIM = 58` 来自 `rl/plan_space`（导入于 `follower.py:26`）；`belief_dim = 563` 由 `len(BeliefInference(opp_deck=DEFAULT_SOLO_DECK, n_particles=128, seed=0).encode(None, None))` 实算，与 `train_solo.py:1066-1067` 的口径一致；`NUM_ENTITY = len(ENTITY_NAMES) = 177`（`follower.py:29`，词表 `observation.py:25-73`）。
- **与既有文档数字不一致**：仓库 `AGENTS.md` 中写"629,359"，本文实算为 **634,735 / 993,008**。差异原因**未在源码中找到依据**（可能是不同 `belief_dim`/`num_entity` 下的历史值）。⇒ **以源码实算数为准，差异原因待确认**。

---

### A.4 超参数全表

#### A.4.1 `TrainConfig` 全字段（47 个，每个字段都列出）

来源：`rl/config.py:97-241`（`@dataclass class TrainConfig`）；默认值由 `dataclasses.fields` 实读并逐条核对源码行。
「类型」为源码类型注解（Python 不做运行时强制）。

| # | 参数名 | 类型 | 默认值 | 含义 | 来源行号 |
|---|---|---|---|---|--- |
| 1 | `name` | str | `"standard"` | 配置名 / 输出子目录名 | `rl/config.py:99` |
| 2 | `description` | str | `""` | 人类可读描述 | `rl/config.py:100` |
| 3 | `total_steps` | int | `20000` | 训练总步数（1 步 = 1 决策帧） | `rl/config.py:102` |
| 4 | `steps_per_eval` | int | `4000` | 每 N 步训满再评估（含"全点"评估：main vs 冻结副本 + baseline0 + baseline_prev + baseline_rand） | `rl/config.py:103` |
| 5 | `anchor_every` | int | `0` | C 方案轻量评估点间隔：只跑固定随机锚点 + 落快照，不跑 main/对照两块；`0` = 关闭（旧行为逐位不变） | `rl/config.py:104-110` |
| 6 | `batch_size` | int | `128` | PPO 每次更新从 transition 池取的样本数 | `rl/config.py:111` |
| 7 | `update_interval` | int | `128` | 每积累多少决策帧触发一次 PPO 更新 | `rl/config.py:112` |
| 8 | `lr` | float | `3e-4` | Adam 学习率 | `rl/config.py:113` |
| 9 | `hidden_dim` | int | `128` | GRU / 编码器隐藏维度 | `rl/config.py:114` |
| 10 | `seed` | int | `0` | 全局种子（`torch/random/np.manual_seed`，`train_solo.py:1057-1059`） | `rl/config.py:115` |
| 11 | `n_eval_games` | int | `40` | 每对评估局数（统计契约注释：轮内聚合 SE≈347.5/√(5N)） | `rl/config.py:116-119` |
| 12 | `max_ep_steps` | int | `360` | 单局截断上限（决策步）；加时窗口可延长到 600 步 | `rl/config.py:120-122` |
| 13 | `n_envs` | int | `1` | 并行多环境数（>1 用 `act_parallel` 批量推理） | `rl/config.py:123` |
| 14 | `parallel` | str | `"mp"` | `n_envs>1` 时的并行方式：`mp`=跨进程 worker / `proc`=单进程批量化 | `rl/config.py:124` |
| 15 | `card_level` | int | `11` | 本局全部卡牌等级（11–16） | `rl/config.py:125` |
| 16 | `eval_at_start` | bool | `True` | 训练开始先跑一次评估/快照（WebUI 立刻有真实数据） | `rl/config.py:126` |
| 17 | `gamma` | float | `0.997` | 折扣因子（注释：0.997³⁶⁰≈0.34） | `rl/config.py:127` |
| 18 | `gae_lambda` | float | `0.95` | GAE λ（economy 预设覆盖为 0.99） | `rl/config.py:128` |
| 19 | `ent_coef` | float | `0.01` | 熵正则系数 | `rl/config.py:129` |
| 20 | `vf_coef` | float | `0.5` | 价值损失系数 | `rl/config.py:130` |
| 21 | `clip` | float | `0.2` | PPO 裁剪范围 | `rl/config.py:131` |
| 22 | `max_grad_norm` | float | `0.5` | 梯度裁剪阈值 | `rl/config.py:132` |
| 23 | `adv_norm` | str | `"scale"` | 优势归一化：`batch`=整批中心化（旧）/`scale`=只除批 std/`none`=原始 | `rl/config.py:139`（语义 `133-138`） |
| 24 | `value_norm` | str | `"none"` | 价值损失量纲：`none`=不缩放（旧）/`running`=按回报运行 std 缩放（`v_loss /= s²`） | `rl/config.py:144`（语义 `140-143`） |
| 25 | `diagnose_every` | int | `10` | 每 N 次 update 额外打印 `p_gnorm/v_gnorm` 梯度分解（`0`=关） | `rl/config.py:147`（语义 `145-146`） |
| 26 | `train_stall_stop` | bool | `True` | solo 训练环僵局早停判平（连续 100 步零塔血变化）；`False`=旧行为拖满 `max_ep_steps` | `rl/config.py:150`（语义 `148-149`） |
| 27 | `adv_inert_probe` | bool | `False` | critic 惰性检验**纯测量**开关：额外算 V≡常数优势并报告 corr/resid/grad_cos；不改写入梯度的量 | `rl/config.py:156`（语义 `151-155`） |
| 28 | `critic_baseline` | str | `"value"` | critic 惰性检验**干预**开关：`value`=用网络 V/`const`=把优势里的 V 换成标量 `c` | `rl/config.py:160`（语义 `157-159`） |
| 29 | `stall_draw_margin` | float | `0.05` | C'：早停局皇冠相同时，塔血%最低差 < 该阈值 → 记平局（去掷硬币级标签）；`0`=退化为旧行为 | `rl/config.py:168`（语义 `161-167`） |
| 30 | `ppo_epochs` | int | `1` | 同一 rollout 重复几轮更新（`1`=旧行为，只有 1 次 `opt.step()`） | `rl/config.py:178`（语义 `169-177`） |
| 31 | `ppo_minibatch` | int | `0` | 每轮切分的小批大小；`0`=整批（不切） | `rl/config.py:179` |
| 32 | `ppo_shuffle` | bool | `False` | 每轮是否打乱样本顺序 | `rl/config.py:180` |
| 33 | `decks_path` | str | `None` | 三分类卡组 JSON 路径（缺省自动探测） | `rl/config.py:182` |
| 34 | `deck_set` | str | `"default"` | solo 镜像/对手卡组：`default`=原版 8 卡 / `four`=四卡组对手池 / `list:Card1,...`=显式 8 卡镜像 | `rl/config.py:186`（语义 `183-185`；解析 `train_solo.py:107-126`） |
| 35 | `main_init` | str | `None` | BC 预训练 / 旧检查点热启动路径 | `rl/config.py:187` |
| 36 | `hist_seed_dirs` | list | `None` | 热启动 run 的对手池补种目录（可多个），本目录 `solo_main_*.pt` 不足时补齐 hist 槽 | `rl/config.py:192`（语义 `188-191`） |
| 37 | `device` | str | `"auto"` | `cpu`/`cuda`/`auto`（auto = cuda 可用则 cuda） | `rl/config.py:193`（解析 `run_league.py:94-105`） |
| 38 | `only_vs_main` | bool | `False` | 联赛模式评估只测 main（15 对→5 对） | `rl/config.py:194` |
| 39 | `keep_snapshot` | bool | `False` | 是否维护联赛 `main_ckpt` 快照槽位 | `rl/config.py:195` |
| 40 | `out_dir` | str | `"runs"` | 输出根目录（产物落在 `out_dir/<name>/`） | `rl/config.py:196` |
| 41 | `solo_copy_every` | int | `2000` | 冻结副本同步间隔（步）：每 N 步把 main 权重拷给对手，并触发 `refresh_hist` | `rl/config.py:198`；`train_solo.py:1726-1733` |
| 42 | `eval_workers` | int | `min(16, os.cpu_count() or 1)`（`default_factory`） | 评估并行进程数；`0`/`1` = 串行。注释记录 worker=16 的间歇性失败与"经验安全档位 12" | `rl/config.py:209`（语义 `199-208`） |
| 43 | `gates` | dict | `{"engagement_rate": {"rel": ">=", "frac": 0.5}, "ghost_rate": {"rel": "<=", "frac": 2.0}}` | 行为指标门禁：数字=绝对阈值（`*_max`/`ghost_rate` 用 `<=`，其余 `>=`）；`{"rel","frac"}`=相对首评估点比值门禁；**只报警不阻断** | `rl/config.py:220-223`；实现 `train_solo.py:234-306` |
| 44 | `opp_mix` | dict | `dict(DEFAULT_OPP_MIX)` = `{"frozen":0.1,"hist":0.6,"defend":0.2,"rand_anchor":0.1}` | solo 训练对手池配比 | `rl/config.py:225`；`DEFAULT_OPP_MIX` 在 `rl/config.py:94` |
| 45 | `value_bypass` | bool | `False` | B'：价值头直连 post-LN `enc`（跳过 GRU，策略头仍走 GRU） | `rl/config.py:231`（语义 `226-230`） |
| 46 | `reward` | dict | `dict(DEFAULT_REWARD)`（17 键，见 A.4.5） | 奖励权重（每配置一套） | `rl/config.py:233`；`DEFAULT_REWARD` 在 `rl/config.py:37-57` |
| 47 | `value_independent` | bool | `False` | E'：独立价值编码器 + 非线性价值头（优先级 independent > bypass > shared） | `rl/config.py:241`（语义 `234-240`） |

**路径与序列化方法**（非 dataclass 字段，但同为配置契约）：`folder` `244-245`、`state_path`（`league_state.json`）`247-248`、
`solo_state_path` `250-251`、`solo_main_path` `253-254`、`solo_ckpt_path` `256-258`、`solo_opt_path` `260-262`、
`run_state_path` `264-265`、`gates_path` `267-269`、`config_path` `271-272`、`replays_dir` `274-275`、
`main_final_path` `277-278`、`ckpt_path` `280-281`、`opt_path` `283-284`、`ensure_dirs` `286-288`、
`to_dict` `291-292`、`save` `294-298`、`from_dict` `300-308`、`load` `310-313`、`presets` `316-369`、`resolve` `371-396`。

> `from_dict` 会把 `reward` 与 `DEFAULT_REWARD` 合并（`rl/config.py:305-307`），未知字段被丢弃（`303`）；
> `resolve` 只覆盖合法字段，未知关键字抛 `ValueError`（`389-395`）。

#### A.4.2 预设（`TrainConfig.presets()`）对默认值的覆盖

来源 `rl/config.py:316-369`。下表只列**与 dataclass 默认值不同**的字段。

| 预设 | `reward` 覆盖 | 其它字段覆盖 | 行号 |
|---|---|---|--- |
| `standard` | 无（全默认） | 无 | `319` |
| `aggressive` | `crown_weight=8.0, win_bonus=15.0, lose_penalty=10.0, invalid_penalty=0.05, elixir_bonus=0.0, elixir_diff_weight=0.7` | 无 | `320-325` |
| `defensive` | `crown_weight=3.0, win_bonus=10.0, lose_penalty=10.0, invalid_penalty=0.1, elixir_bonus=0.0, elixir_diff_weight=0.3` | 无 | `326-331` |
| `lockdown` | `crown_weight=8.0, win_bonus=10.0, lose_penalty=10.0, invalid_penalty=0.05, elixir_bonus=0.0, elixir_diff_weight=0.05` | 无 | `332-337` |
| `elixir` | `crown_weight=8.0, win_bonus=10.0, lose_penalty=10.0, invalid_penalty=0.05, elixir_bonus=0.01, elixir_diff_weight=0.5` | 无 | `338-343` |
| `economy` | `crown_weight=8.0, win_bonus=10.0, lose_penalty=10.0, invalid_penalty=0.05, elixir_bonus=0.0, normalize_tower_dmg=True, elixir_diff_weight=0.5` | `gae_lambda=0.99`、`steps_per_eval=8000`、`value_norm="running"`、`value_bypass=True`、`value_independent=True`、`only_vs_main=True` | `344-364` |
| `fast` | 无 | `total_steps=2000`、`steps_per_eval=500`、`n_eval_games=2`、`max_ep_steps=300` | `365-368` |

`MODEL_REWARD_OVERRIDES`（flow 联赛按模型 id 覆盖）：`main/all_decks/random_deck` = `{}`；
`push_flow.elixir_diff_weight = 0.7`；`counter_flow = 0.3`；`lockdown_flow = 0.05`（`rl/config.py:62-69`）。
奖励换算函数：`reward_to_env(cfg) = dict(DEFAULT_REWARD, **cfg.reward)`（`rl/config.py:399-401`）；
`model_reward_weights(model_id, cfg)` 再叠 `MODEL_REWARD_OVERRIDES`（`rl/config.py:404-412`）。

> **注意**：`economy` 预设**没有**设置 `eval_workers`（`rl/config.py:344-364` 无该键），
> 因此走 `default_factory = min(16, os.cpu_count())`；命令行 `--eval-workers` 未给时不会被覆盖。

#### A.4.3 PPO 相关超参：`PPOTrainer.__init__` 的函数签名默认值

来源 `rl/ppo.py:87-90`。**这些默认值与 `TrainConfig` 的同名默认值并不相同**（见"差异"列）。

| 参数 | 签名默认值 | `TrainConfig` 同名默认 | 差异 | 行号 |
|---|---|---|---|--- |
| `lr` | `3e-4` | `3e-4` | 一致 | `ppo.py:87` |
| `gamma` | `0.99` | `0.997` | **不一致** | `ppo.py:87` / `config.py:127` |
| `gae_lambda` | `0.95` | `0.95` | 一致 | `ppo.py:87` / `config.py:128` |
| `clip` | `0.2` | `0.2` | 一致 | `ppo.py:87` |
| `vf_coef` | `0.5` | `0.5` | 一致 | `ppo.py:88` |
| `ent_coef` | `0.01` | `0.01` | 一致 | `ppo.py:88` |
| `max_grad_norm` | `0.5` | `0.5` | 一致 | `ppo.py:88` |
| `adv_norm` | `"batch"` | `"scale"` | **不一致**（函数默认=旧行为，注释 `config.py:137-138`、`ppo.py:105-108` 明确说明） | `ppo.py:88` / `config.py:139` |
| `value_norm` | `"none"` | `"none"` | 一致 | `ppo.py:89` / `config.py:144` |
| `diagnose_every` | `0` | `10` | **不一致** | `ppo.py:89` / `config.py:147` |
| `ret_scaler` | `None`（新建 `ReturnScaler(eps=1e-6)`） | — | 仅 `PPOTrainer` 有此参数，断点续训传入已恢复实例 | `ppo.py:89,123`；`ReturnScaler.__init__` `41:48-52` |
| `n_epochs` | `1` | `ppo_epochs=1` | 一致 | `ppo.py:90` / `config.py:178` |
| `minibatch_size` | `0` | `ppo_minibatch=0` | 一致 | `ppo.py:90` / `config.py:179` |
| `shuffle` | `False` | `ppo_shuffle=False` | 一致 | `ppo.py:90` / `config.py:180` |
| `seed` | `12345` | — | 仅用于小批打乱 RNG（`self.rng`） | `ppo.py:90,129` |

**其它 PPO 内部常量 / 派生量**：`grad_steps` 累计 `opt.step()` 次数（`ppo.py:132`）；`updates` 累计 `update()` 次数（`124`）；
`legacy` 判定式（`251-252`）；`clip` 用在 `surr2` 与 `clip_frac`（`382,397-398`）；梯度裁剪 `clip_grad_norm_`（`462-463`）。

**构建点差异（重要）**：
- solo：`PPOTrainer(main, lr, gamma, gae_lambda, clip, vf_coef, ent_coef, max_grad_norm, adv_norm, value_norm, diagnose_every, n_epochs=cfg.ppo_epochs, minibatch_size=cfg.ppo_minibatch, shuffle=cfg.ppo_shuffle, seed=cfg.seed)` — **`TrainConfig` 的 PPO 字段全部透传**（`rl/train_solo.py:1138-1143`）。
- run：`_make_trainer(main, cfg)` 只传 `lr, gamma, gae_lambda, clip, vf_coef, ent_coef, max_grad_norm, adv_norm`（`rl/run_league.py:584-587`）⇒ **run 模式下 `value_norm` / `diagnose_every` / `ppo_epochs` / `ppo_minibatch` / `ppo_shuffle` 全部退回函数默认值**（即 `none` / `0` / `1` / `0` / `False`，等价于 legacy 单次更新）。`_restore` 也走同一 `_make_trainer`（`rl/run_league.py:602,611`）。

#### A.4.4 枚举 / 字典型字段的全部取值

| 字段 | 全部取值 | 含义 | 来源 |
|---|---|---|--- |
| `adv_norm` | `batch` / `scale` / `none` | 整批中心化+除 std（旧） / 只除批 std / 不归一化；其它值抛 `ValueError` | `rl/config.py:139`；`rl/ppo.py:224-233` |
| `value_norm` | `none`（含 `""`、`None`）/ `running` | 价值损失不缩放 / `v_loss = MSE/s²`；其它值抛 `ValueError` | `rl/config.py:144`；`rl/ppo.py:245-248` |
| `critic_baseline` | `value` / `const` | 优势用网络 V / 优势里 V 整体替换为标量 `c = ppo.ret_scaler.mean` | `rl/config.py:160`；`rl/train_solo.py:1214,1647-1650` |
| `parallel` | `mp` / `proc` | 跨进程 worker / 单进程批量化 | `rl/config.py:124`；`rl/run_league.py:1145-1148` |
| `device` | `cpu` / `cuda` / `auto` | 其它值抛 `ValueError`；`cuda` 不可用则回退 `cpu` 并打印诊断 | `rl/config.py:193`；`rl/run_league.py:94-105` |
| `deck_set` | `default` / `four` / `list:Card1,...` | 原版 8 卡镜像 / 四卡组对手池 / 显式 8 卡镜像（非 8 张或含不可部署卡抛错） | `rl/config.py:186`；`rl/train_solo.py:107-126` |
| `opp_mix` 键 | `frozen` / `hist` / `defend` / `rand_anchor` | 冻结副本 / 历史 ckpt（PFSP）/ 真防守脚本 / 固定随机锚点；缺 `rand_anchor` 键时概率为 0 | `rl/config.py:225,83-84,94`；`rl/train_solo.py:448-477` |
| `gates` 值形态 | 数字 / `{"rel": ">="\|"<=", "frac": float}` | 绝对阈值（op 由名字推断）/ 相对本 run 首个评估点的比值门禁 | `rl/config.py:210-223`；`rl/train_solo.py:279-297` |
| `mode`（CLI） | `eval` / `run` / `flow` / `solo` / `flow-sweep-stream` / `flow-sweep-games5` | 见 A.1.1 | `rl/run_league.py:1171-1173` |
| `reward` 键 | 见 A.4.5（17 键） | 奖励权重 | `rl/config.py:37-57` |

#### A.4.5 奖励权重全表（`DEFAULT_REWARD`，17 键）

来源 `rl/config.py:37-57`；`env_wrapper._DEFAULT_REWARD` 是同值的副本（`rl/env_wrapper.py:40-58`，注释要求"勿单独改一处"）。

| 键 | 默认值 | 含义 | `config.py` 行号 | `env_wrapper.py` 行号 |
|---|---|---|---|--- |
| `crown_weight` | `8.0` | 皇冠差系数（破敌塔每座 +8） | `38` | `41` |
| `crown_lose_weight` | `10.0` | 被破塔惩罚（> `crown_weight`，丢塔比破塔更痛） | `39` | `42` |
| `tower_dmg_opp` | `0.001` | 敌方塔损 → 正奖励（前段 t<120） | `40` | `43` |
| `tower_dmg_self` | `0.0012` | 我方塔损 → 负奖励（不对称） | `41` | `44` |
| `tower_dmg_late` | `0.002` | 双倍期（t≥120）塔血系数 | `42` | `45` |
| `tower_dmg_self_late` | `0.0022` | 双倍期我方塔损系数 | `43` | `46` |
| `win_bonus` | `10.0` | 获胜加成 | `44` | `47` |
| `lose_penalty` | `10.0` | 失败惩罚 | `45` | `48` |
| `draw_penalty` | `10.0` | 平局惩罚（=失败；`0` 即旧行为"免费平局"） | `46` | —（在 `config.py` 有，`env_wrapper` 侧取 `rw.get("draw_penalty", lose_penalty)`，`211`） |
| `invalid_penalty` | `0.05` | 每次非法动作惩罚 | `47` | `49` |
| `elixir_bonus` | `0.0` | 每步按我方剩余圣水的正向 shaping | `48` | `50` |
| `normalize_tower_dmg` | `True` | 塔损按塔血%归一化到 lv11 锚（跨等级一致） | `49` | `51` |
| `elixir_diff_weight` | `0.5` | 资源账 edw 前段（t<120；lv11 下 1 圣水 ≈ 500 塔血） | `50-51` | `52` |
| `elixir_diff_late` | `0.1` | 双倍期 edw（t≥120） | `52` | `53` |
| `unit_dmg_k` | `0.0005` | 单位受伤 shaping（敌方单位每掉 1 血 → 我方 +k） | `53` | `54` |
| `tower_premium_k` | `2.0` | 塔血差异化定价凹形溢价强度（残血塔最多 ×3 单位血价值） | `54-55` | `55-56`（同值常量 `DEFAULT_TOWER_PREMIUM_K` 在 `97`） |
| `king_gate` | `0.05` | 两公主塔存活时王塔单位血价值系数 | `56` | `57`（同值常量 `DEFAULT_KING_GATE` 在 `98`） |

---

### A.5 命令行与运行方式

#### A.5.1 `run_league.py` 完整 CLI 参数表（54 项）

来源：`argparse.ArgumentParser` 定义 `rl/run_league.py:1170-1293`；分派逻辑 `1298-1371`；
`overrides` 组装 `1307-1344`；`resume = not args.fresh`（`1296`）。
「取值」列中 `flag` 表示 `action="store_true"`（出现即 True，无值）。

| # | 参数名 | 默认值 | 取值 | 含义 | 行号 |
|---|---|---|---|---|--- |
| 1 | `--mode` | `"run"` | `eval`/`run`/`flow`/`solo`/`flow-sweep-stream`/`flow-sweep-games5` | 运行模式选择 | `1171-1173` |
| 2 | `--policies` | `None` | 路径列表（`nargs="+"`） | eval 模式的策略检查点列表 | `1175` |
| 3 | `--kinds` | `None` | 字符串列表（`nargs="+"`） | eval 模式的 agent kind | `1176` |
| 4 | `--n-games` | `20` | int | eval 模式每对局数 | `1177` |
| 5 | `--max-steps` | `600` | int | eval 模式单局最大步数 | `1178` |
| 6 | `--config` | `"standard"` | `standard`/`aggressive`/`defensive`/`lockdown`/`elixir`/`economy`/`fast` 或 `--load-config` 的 JSON | 命名配置预设 | `1180-1182` |
| 7 | `--config-name` | `None` | str | 覆盖配置名（= 输出文件夹名），默认用预设名 | `1183-1184` |
| 8 | `--load-config` | `None` | JSON 路径 | 从 JSON 载入自定义配置（含奖励权重） | `1185-1186` |
| 9 | `--save-config` | `None` | JSON 路径 | 把解析后的配置导出为 JSON | `1187-1188` |
| 10 | `--out-dir` | `None` | 目录 | 输出根目录（缺省用 `config.out_dir="runs"`） | `1189-1190` |
| 11 | `--fresh` | `False` | flag | 忽略断点强制从头训练（默认有断点就自动续训） | `1193-1194` |
| 12 | `--resume` | `False` | flag（`help=SUPPRESS`） | 兼容旧脚本的保留参数；现已默认自动续训 | `1195-1196` |
| 13 | `--no-replays` | `False` | flag | 不保存每评估周期的联赛录像（默认保存） | `1197-1198` |
| 14 | `--no-eval-start` | `False` | flag | 训练开始不先跑一次评估（默认跑） | `1199-1200` |
| 15 | `--device` | `None` | `cpu`/`cuda`/`auto` | 计算设备（缺省 auto） | `1201-1202` |
| 16 | `--total-steps` | `None` | int | 覆盖 `total_steps` | `1204` |
| 17 | `--steps-per-eval` | `None` | int | 覆盖 `steps_per_eval` | `1205` |
| 18 | `--anchor-every` | `None` | int | C 方案轻量锚点评估点间隔（只跑 baseline_rand + 落快照）；`0`/缺省 = 关闭 | `1206-1208` |
| 19 | `--n-envs` | `None` | int | 并行多环境数（>1 用批量推理/更新） | `1209-1210` |
| 20 | `--parallel` | `None` | `mp`/`proc` | `n_envs>1` 时的并行方式 | `1211-1213` |
| 21 | `--card-level` | `None` | int（11–16） | 本局全部卡牌等级 | `1214-1215` |
| 22 | `--main-init` | `None` | 检查点路径 | 热启动检查点 | `1216` |
| 23 | `--decks-path` | `None` | JSON 路径 | 三分类卡组 JSON（缺省自动探测） | `1217-1218` |
| 24 | `--deck-set` | `None` | `default`/`four`/`list:卡1,卡2,...` | solo 镜像/对手卡组选择 | `1219-1221` |
| 25 | `--keep-snapshot` | `False` | flag | 同时维护 `main_ckpt` 快照槽位 | `1222-1223` |
| 26 | `--batch-size` | `None` | int | 覆盖 `batch_size` | `1224` |
| 27 | `--update-interval` | `None` | int | 覆盖 `update_interval` | `1225` |
| 28 | `--lr` | `None` | float | 覆盖学习率 | `1226` |
| 29 | `--n-eval-games` | `None` | int | 覆盖每对评估局数 | `1227` |
| 30 | `--max-ep-steps` | `None` | int | 覆盖单局截断步数 | `1228` |
| 31 | `--only-vs-main` | `False` | flag | 评估只打 main vs 其它（默认全轮转） | `1229-1230` |
| 32 | `--seed` | `None` | int | 覆盖种子 | `1231` |
| 33 | `--hidden-dim` | `None` | int | 覆盖隐藏维度 | `1232` |
| 34 | `--n-random-decks` | `30` | int | flow 模式每次训练生成的完全随机卡组套数 | `1234-1235` |
| 35 | `--sweep-runs` | `None` | int | flow-sweep 训练轮数（stream 默认 20 / games5 默认 4） | `1237-1238` |
| 36 | `--sweep-scale` | `0.1` | float | flow-sweep 卡组池缩小比例 | `1239-1240` |
| 37 | `--sweep-eval-games` | `None` | int | flow-sweep 每对评估局数（默认 10） | `1241-1242` |
| 38 | `--solo-copy-every` | `None` | int | solo 冻结副本同步间隔（默认 2000） | `1244-1245` |
| 39 | `--eval-workers` | `None` | int | 评估并行进程数（>1 spawn 并行；`0`=串行） | `1246-1248` |
| 40 | `--gae-lambda` | `None` | float | 覆盖 GAE λ | `1250-1251` |
| 41 | `--ent-coef` | `None` | float | 覆盖熵系数 | `1252-1253` |
| 42 | `--adv-norm` | `None` | `batch`/`scale`/`none` | 优势归一化 | `1254-1255` |
| 43 | `--value-norm` | `None` | `none`/`running` | 价值通道量纲 | `1256-1258` |
| 44 | `--diagnose-every` | `None` | int | 梯度成分诊断采样间隔（`0`=关） | `1259-1261` |
| 45 | `--hist-seed-dir` | `None` | 目录（`action="append"`，可多次） | 对手池 hist 槽补种目录 | `1262-1264` |
| 46 | `--no-train-stall-stop` | `False` | flag | solo 关闭训练环僵局早停（关=旧行为拖满 `max_ep_steps`） | `1265-1266` |
| 47 | `--adv-inert-probe` | `False` | flag | critic 惰性检验纯测量开关 | `1267-1270` |
| 48 | `--critic-baseline` | `None` | `value`/`const` | critic 惰性检验干预开关（`const` 非推荐） | `1271-1273` |
| 49 | `--stall-draw-margin` | `None` | float | C' 早停低置信裁定降噪阈值（`0`=旧行为） | `1274-1276` |
| 50 | `--no-value-bypass` | `False` | flag | 关闭 B'（回旧 GRU value 通路） | `1277-1279` |
| 51 | `--no-value-independent` | `False` | flag | 关闭 E'（回共享 trunk value 通路） | `1280-1282` |
| 52 | `--ppo-epochs` | `None` | int | F'：同一 rollout 重复轮数（默认 1 = 旧行为） | `1284-1287` |
| 53 | `--ppo-minibatch` | `None` | int | F'：小批大小（`0`=整批/旧行为） | `1288-1291` |
| 54 | `--ppo-shuffle` | `False` | flag | F'：每轮打乱样本顺序 | `1292-1293` |

**`--no-*` 型 flag 的 override 语义**：`--no-value-bypass` / `--no-value-independent` / `--no-train-stall-stop`
显式写入 `False`（`1307-1330`）；`--adv-inert-probe` / `--keep-snapshot` / `--only-vs-main` / `--ppo-shuffle` 写 `True`（`1319,1331-1338`）。
`overrides` 白名单字段共 24 个（`1308-1315`）——**只有名单里的 `--xxx` 会覆盖 `TrainConfig`**；
`--hist-seed-dir`、`--stall-draw-margin`、`--critic-baseline`、`--out-dir`、`--config-name`、`--no-eval-start` 单独处理（`1321-1344`）。

**参数生效范围提示（源码 + 全目录 grep 可确认）**：
- `--policies/--kinds/--n-games/--max-steps` 只用于 `--mode eval`（`1298-1304`）。
- `--n-random-decks` 只用于 `--mode flow`（`1351-1354`）；`--sweep-*` 只用于 `flow-sweep-*`（`1356-1363`）。
- `--ppo-*`、`--value-norm`、`--diagnose-every` 在 **run 模式不生效**（`_make_trainer` 不透传，见 A.4.3）。
- `--anchor-every` 会被写入 `TrainConfig`（`1308`），但**只有 solo 主循环消费它**
  （`train_solo.py:1720-1723`）；`grep -rn anchor_every rl/` 在 run 侧只命中 `run_league.py:1308` 的白名单，
  `_run_single`/`_run_vec`/`_run_mp` 均无 `anchor_every` 分支 ⇒ **run 模式下该参数被静默忽略**。
- `--no-train-stall-stop` 同理：`grep -rn train_stall_stop rl/` 的唯一消费点是 `train_solo.py:1545`
  （外加 `run_league.py:1329-1330` 的 override 组装）⇒ 只影响 solo。

#### A.5.2 命令示例

**(a) 逐字抄自源码 docstring 的命令**（`rl/train_solo.py:12-13` 的"用法（run_league 入口）"）：

```bat
python rl/run_league.py --mode solo --config economy --device cuda
```

**(b) 由 argparse 默认值推导的合法命令（本命令由 argparse 默认值推导，非抄自任何脚本）**：

```bat
:: 等价于不带任何参数运行 = run 模式 + standard 预设（默认值均来自 argparse）
python rl/run_league.py
```
推导：`--mode` 默认 `"run"`（`rl/run_league.py:1173`）、`--config` 默认 `"standard"`（`1180`）、
`--device` 默认 `None` ⇒ 不覆盖 `TrainConfig.device="auto"`（`config.py:193`）、
`--n-envs` 默认 `None` ⇒ `n_envs=1` ⇒ 走 `_run_single`（`1145-1149`）。

**(c) start_rl.bat 实际拼出的命令行**（模板 `start_rl.bat:244,258`；默认值 `start_rl.bat:35-46`）：

```bat
:: start_rl.bat 无参数时的等价命令（默认 MODE=solo CONFIG=economy OUT_DIR=runs DEVICE=auto
:: N_ENVS=1 TOTAL_STEPS=20000 STEPS_PER_EVAL=2000 N_EVAL_GAMES=16 MAX_EP_STEPS=360
:: EVAL_WORKERS=16 SOLO_COPY_EVERY=2000；--opt=value 形式见 start_rl.bat:244）
python scripts\rl\run_league.py --mode=solo --config=economy --out-dir=runs --device=auto ^
  --n-envs=1 --total-steps=20000 --steps-per-eval=2000 --n-eval-games=16 --max-ep-steps=360 ^
  --eval-workers=16 --solo-copy-every=2000
```

**(d) start_training.bat 实际拼出的命令行**（模板 `start_training.bat:175`；默认值 `27-36`）：

```bat
:: start_training.bat 无参数时的等价命令（MODE 固定 run；CONFIG=standard；
:: N_EVAL_GAMES=4；MAX_EP_STEPS=600；其余同上）
python scripts\rl\run_league.py --mode run --config=standard --out-dir=runs --device=auto ^
  --n-envs=1 --total-steps=20000 --steps-per-eval=2000 --n-eval-games=4 --max-ep-steps=600
```

> ⚠️ 两个 `.bat` 的默认值与 `argparse`/`TrainConfig` 默认值**不同**：
> `--n-eval-games`（16 / 4 vs 40，`config.py:116`）、`--steps-per-eval`（2000 vs 4000，`config.py:103`）、
> `--eval-workers`（16 vs `min(16, cpu_count)`，`config.py:209`）、`--max-ep-steps`（600 vs 360，`config.py:120`）。

**(e) 入口脚本层级**：`start_rl.bat` 实际调用的是 `scripts\rl\run_league.py`（`start_rl.bat:276`），
该文件是 wrapper：切 cwd 到 `src/clasher_new` 后用 `runpy.run_path` 执行 `rl/run_league.py`（`scripts/rl/run_league.py:8-17`）。

**(f) 并行 / 实验性开关的示例写法（参数名取自 A.5.1 表，取值合法）**：

```bat
python rl/run_league.py --mode solo --config economy --config-name my_run --fresh ^
  --total-steps 20000 --steps-per-eval 8000 --anchor-every 2500 --n-eval-games 40 ^
  --eval-workers 12 --device cuda --value-norm running --adv-norm scale ^
  --ppo-epochs 4 --ppo-minibatch 32 --ppo-shuffle ^
  --hist-seed-dir runs/economy_9k_ft --hist-seed-dir runs/economy_9j
```
（本命令为按 A.5.1 的合法取值组合而成，**不是**抄自任何脚本或文档。）

---

### A.6 检查点 / 目录结构 / 恢复训练

#### A.6.1 目录结构与文件清单

所有产物落在 `out_dir/<name>/`（`folder()`，`rl/config.py:244-245`）；`ensure_dirs()` 创建
主目录与 `replays/`（`rl/config.py:286-288`）。

| 文件 | 写入函数 | 内容 / 字段 | 来源 |
|---|---|---|--- |
| `config.json` | `cfg.save()`（`TrainConfig.save`） | `asdict(self)` 全字段，`ensure_ascii=False, indent=2` | `rl/config.py:294-298`；调用点 `train_solo.py:1049`、`run_league.py:700,805,953`；`--save-config` 另存 `1347-1349` |
| `solo_main.pt` | `save_checkpoint(main, cfg.solo_main_path())` | `{state_dict, plan_dim, belief_dim, hidden_dim, value_bypass, value_independent}` | `train_solo.py:1417,1736`；`follower.py:55-64` |
| `solo_main_<step>.pt` | `save_checkpoint(main, cfg.solo_ckpt_path(step))` | 同上；每次评估/锚点保留一份（回溯 + 作为 hist 池成员） | `train_solo.py:1418`；路径 `config.py:256-258` |
| `solo_opt.pt` | `torch.save(ppo.opt.state_dict(), cfg.solo_opt_path())` | Adam 优化器状态 | `train_solo.py:1419`；路径 `config.py:260-262` |
| `run_state.json`（solo） | `_persist` 内 `json.dump({...})` | `step`、`solo_ckpt`、`solo_opt`、`config`、`device`、`value_norm`、`adv_norm`、`ppo_epochs`、`ppo_minibatch`、`ppo_shuffle`、`adv_inert`、`ret_scaler` | `train_solo.py:1440-1452` |
| `run_state.json`（run） | `_save_snapshot` 内 `json.dump(run_state)` | `step`、`total_steps`、`main_ckpt`、`opt_ckpt`、`config`、`device` | `run_league.py:635-639` |
| `solo_state.json` | `write_solo_state(...)` | `mode="solo"`、`agents`、`history`（每评估点一条 `{step,wins,losses,draws,games,winrate,winrate_se,mean_reward,...}`）、`total_steps`、`target_steps`、`deck`、`opponent="self-play-frozen-copy"`、`copy_every`、`status`、`demo`、`_controls_history` | `train_solo.py:513-547`；调用 `1382-1384,1475-1476` |
| `gates.json` | `_write_gate_report`（经 `_check_gates`） | `step`、`ok`、`baseline`（首评估点基准）、`checks`（`{name,value,op,threshold,baseline,basis,ok}`）；首个评估点只写 `note="基线点（建立 baseline，不判定）"` | `train_solo.py:234-317`；调用 `1386` |
| `main_ckpt_<step>.pt` | `save_checkpoint(main, cfg.ckpt_path(step))` | 同 ckpt 元数据 | `run_league.py:628,630`；路径 `config.py:280-281` |
| `main_opt_<step>.pt` | `torch.save(ppo.opt.state_dict(), cfg.opt_path(step))` | Adam 状态 | `run_league.py:629,631`；路径 `config.py:283-284` |
| `main_final.pt` | `save_checkpoint(main, cfg.main_final_path())` | 训练结束时的 main 权重 | `run_league.py:793,932,1137`；路径 `config.py:277-278` |
| `league_state.json` | `league.save_state(cfg.state_path())`（在 `_save_snapshot` 内） | 联赛状态（Elo/PFSP/agents），由 `rl/league.py` 实现 | `run_league.py:634`；路径 `config.py:247-248` |
| `replays/league_<step>.pkl` | `save_league_replays(replays, ...)` | 逐局录像（`LeagueGameRecorder` 逐帧压缩：`meta{pair,side0,max_steps,steps,decks}`、`winner`、`frames`）；solo 主评估与锚点写，对照组 `save_replays=False` 不写 | `run_league.py:645-648`；`train_solo.py:796-798,1033-1035`；recorder `run_league.py:108-138`；`rl/replay.py` |

**solo 的落盘时机**：
- `_persist(step)`：`solo_main.pt` + `solo_main_<step>.pt` + `solo_opt.pt` + `run_state.json`（`train_solo.py:1410-1452`），
  被 `eval_and_write`（`1408`）与 `anchor_point`（`1474`）共用 ⇒ `anchor_every` 生效时崩溃续训粒度是 `anchor_every` 而非 `steps_per_eval`（注释 `1413-1415`）。
- 训练循环结束后再 `save_checkpoint(main, cfg.solo_main_path())`，若末步不是评估点则补一次 `eval_and_write(total_steps)`（`train_solo.py:1736-1738`）。

**run 的落盘时机**：`_eval_and_snapshot` → `eval_round_robin` + 写回放 + `_save_snapshot`（`run_league.py:642-652`）；
训练结束 `save_checkpoint(main, cfg.main_final_path())`，末步非评估点再补一次（`793-795`）。

#### A.6.2 恢复训练（`--resume` / 自动续训 / `--fresh`）

**(1) 默认自动续训**：`resume = not args.fresh`（`run_league.py:1296`）——不加参数即有断点就接着训，`--fresh` 才从头。
`--resume` 已改为 `help=argparse.SUPPRESS` 的兼容参数（`1195-1196`）。

**(2) run 模式恢复**：`_restore(league, cfg, main, device, resume)`（`run_league.py:600-624`）：
读 `run_state.json`（`_load_run_state`，`590-597`）→ 若 `main_ckpt` 存在则
`load_checkpoint(ckpt, hidden_dim=cfg.hidden_dim)` → `_make_trainer` 重建 PPO → 载入 `opt_ckpt` → `league.load_state(cfg.state_path(), policies={"main": main})`
→ `start_step = rs["step"]`（`607-618`）；`main_ckpt` 缺失时打印"检查点缺失，从头开始"（`619`）。
`resume=False` 时若有 `league_state.json` 仍会 `load_state`（`622-623`）。

**(3) solo 模式恢复**（`train_solo.py:1069-1161`）：
- `rs = _load_run_state(cfg)`（`1073`，函数体同 run 侧：读 `run_state.json`）；
  仅当 `rs["solo_ckpt"]` 存在才认领断点，并用 `solo_state.json` 的 `history` 恢复曲线（`1074-1081`）；否则打印"resume 检查点缺失，从头开始"（`1083`）。
- `main` 构造优先级：**断点权重 > `--main-init` > 随机初始化**
  （`main_init` 分支 `1085-1093`；断点覆盖 `1099-1104`；随机初始化 `1094-1097`）。
- PPO 重建时 `TrainConfig` 的 PPO 字段全部透传（`1138-1143`）；
  `ret_scaler` 从 `run_state["ret_scaler"]` 恢复（`1152-1156`）；Adam 状态恢复失败只打印提示不回滚（`1157-1161`）。
- 冻结副本开局 = main（即断点权重），`frozen_step = start_step`（`1132-1137`）。
- `eval_at_start` 仅在 `start_step == 0` 时跑（`1482`），避免 resume 重复起步评估。

**(4) 注意点（源码可确认的脚枪）**：

| # | 注意点 | 证据 |
|---|---|--- |
| 1 | **`config.json` 不参与 resume 解析**：续训必须重传 `--ppo-epochs/--ppo-minibatch/--ppo-shuffle`，否则静默退回"1 次梯度步/更新"；solo 会在预算不一致时打印告警 | `train_solo.py:1117-1131`（比较 `rs` 与 `cfg` 的预算） |
| 2 | **`--fresh` 挡不住 `--main-init`**：`--fresh` 只把 `resume` 置 False（`rs=None`），`main_init` 分支条件仍成立并加载权重 | `run_league.py:1296`；`train_solo.py:1085-1093` |
| 3 | **solo 热启动显式传 `plan_dim=PLAN_DIM, belief_dim=belief_dim`**，避免旧 ckpt 元数据（旧维度）把 main 建成旧维度后 `_sync_frozen_copy` shape 失配 | `train_solo.py:1085-1093`（注释 `1087-1089`） |
| 4 | **run 模式热启动只传 `hidden_dim`**，未显式传 `plan_dim/belief_dim` ⇒ 维度由旧 ckpt 元数据决定 | `run_league.py:679-681` |
| 5 | 架构标志不一致（`value_bypass`/`value_independent`）只在 `load_checkpoint` 里**告警**，不阻断；solo 另在启动后兜底比对一次 | `follower.py:94-104`；`train_solo.py:1108-1112` |
| 6 | 缺 `enc_ln.*`/`grid_ln.*` 的旧 ckpt 会显式告警"不可续训（要求 `--fresh`），只能当对照基线/对手池" | `follower.py:144-150` |
| 7 | 续训时 `main_init` 的 network 维度若与新网络不同，`load_checkpoint` 走尾零兼容（`plan_mlp`/`belief_mlp`/`entity_emb`），其余形状不匹配保持新初始化、不崩 | `follower.py:114-131` |
| 8 | solo 的 `steps_per_eval` 与 `solo_copy_every` 整除时，"先评估后同步"的顺序保证评估对手不是刚同步的 main | `train_solo.py:1711-1728`（注释说明旧顺序问题） |
| 9 | 恢复后 `start_step` 已被训练过的部分不会重跑：solo 主循环从 `start_step + 1` 开始 | `train_solo.py:1541`；run `run_league.py:731`（single）/ `844`（vec）/ `1039`（mp，三处均以 `start_step` 初始化 `step`） |
| 10 | 相同 `step` 的历史点会被去重（避免 `x=0` 两个点） | `train_solo.py:224-231,1284` |

---

### A.7 本部分待确认清单

| # | 条目 | 无法确认的原因 | 要确认需要什么 |
|---|---|---|--- |
| 1 | flow 模式（`--mode flow`）的完整数据流、模型槽位、评估口径、`cfg` 字段使用范围 | `rl/flow_league.py` 不在本部分允许阅读的源码清单内，只能从 `run_league.py:1351-1363` 看到入口签名与两个参数 | 读 `rl/flow_league.py` 的 `run_flow` / `run_flow_sweep` 全文，以及 `League` 在 flow 下的装配方式 |
| 2 | `flow-sweep-*` 两个策略（`stream` / `games5`）的具体判别与产出一致性 | 同上 | 同上 |
| 3 | 总参数量与既有文档"629,359"不一致的成因 | 源码中没有硬编码参数量；本文两个数（634,735 / 993,008）是实算值，差异只能来自历史 `belief_dim`/`num_entity`/`hidden` 或旧架构 | 找到写 629,359 的文档所依据的版本/配置，或在那个版本上重算 |
| 4 | `PLAN_DIM=58` 与 `belief_dim=563` 的内部构成（各字段偏移、事件 one-hot 宽度） | 定义在 `rl/plan_space.py` 与 `rl/belief.py`，不在允许阅读清单内；本文只用了模块导出的常量与实算长度 | 读 `rl/plan_space.py`（`PlanToken.to_vector`/`PLAN_DIM`）与 `rl/belief.py`（`belief_token_dim`/`encode`） |
| 5 | run 模式下 `--anchor-every` 被接受但不生效，这是设计意图还是遗漏 | 事实层面已由 grep 确认（唯一消费点在 `train_solo.py:1720`；run 侧只有 `run_league.py:1308` 的白名单），但**意图**无法从源码判定（无相关 TODO/注释） | 查 `docs/eval_cadence_c_2026-09-13.md` 声明的适用范围，或向作者确认 run 模式是否需要同节奏 |
| 6 | run 模式评估侧早停与低置信降噪的配合关系 | `_run_side0` 早停后直接调 `timeout_winner`（`run_league.py:307-309`），而 solo 训练环早停用 `settle_stall(stall_draw_margin)`（`train_solo.py:1552`）；run 模式是否应改用 `settle_stall` 属设计选择，源码无说明 | 明确 run 模式评估/训练各自的结算口径设计意图 |
| 7 | `league_state.json` 的具体字段与恢复语义 | 由 `rl/league.py` 的 `save_state` / `load_state` / `sample_opponent` / `record_match` 定义，该文件不在允许阅读清单内 | 读 `rl/league.py` |
| 8 | `info["opp_played"]` 到信念模块的过滤细节（技能哨兵 `"__ability__"` 如何被消费） | 过滤逻辑在 `rl/belief.py` 的 `update` 内，不在允许阅读清单内 | 读 `rl/belief.py:update` |
| 9 | `ProphetPlanner` / `BeliefPlanner` 的输出维度语义（58 维中哪一段对应什么） | 同上第 4 条；正文只用到 `to_vector()` 的返回值与 `PLAN_DIM` | 读 `rl/plan_space.py`、`rl/prophet.py`、`rl/belief_planner.py` |
| 10 | `eval_workers` 默认值在目标机器上的实际数值 | `default_factory = min(16, os.cpu_count() or 1)`（`config.py:209`）依赖运行机器；本文不能给一个确定的数字 | 在目标机器上打印 `TrainConfig().eval_workers` |
| 11 | `--mode eval` 与 `--mode flow` 是否也支持 `--load-config` / `--save-config` 的全部组合 | 分派顺序上 `cfg = TrainConfig.resolve(...)` 在 mode 分派之前（`1346-1353`），但 flow 侧如何使用 `cfg` 未在允许清单内 | 同上第 1 条 |
| 12 | `kinds` 的合法取值集合（`--kinds`） | `evaluate_league` 直接把 `kinds[i]` 透传给 `league.add_agent(..., kind=...)`（`run_league.py:514-518`），未做校验；合法值的定义在 `rl/league.py` | 读 `rl/league.py`（`add_agent`/`Agent`） |

---

---

## 训练方法文档 · 第 B 部分：训练循环、评估诊断与函数索引

> **范围与口径声明**：本文只描述 `src/clasher_new/rl/` 下源码**实际写的**内容。所有函数名、参数、默认值、阈值、行号均来自源码阅读并在行内标注（`rl/xxx.py:NNN`）。凡源码中无法确认的，一律写「待确认：…（原因：…）」，不做推断性补全。默认值的"真源"分两处：`rl/config.py` 的 `TrainConfig`（训练入口经它取值）与各模块的函数签名默认（被其它入口共用时的旧行为锚）。

---

### B.1 单次 PPO 更新的逐步过程

**入口与主循环**：`run_solo(cfg, resume=False, record_replays=True)`（rl/train_solo.py:1039）是 solo 模式唯一主循环；主循环体是 `for step in range(start_step + 1, cfg.total_steps + 1)`（rl/train_solo.py:1541）。**1 `step` = 1 决策帧**（不是一次更新）。单次迭代按固定顺序做下面 7 件事。

#### B.1.1 ① 训练环僵局早停（条件触发）

- 触发条件：`cfg.train_stall_stop and len(ep_rew) and len(ep_rew) % STALL_WINDOW == 0`（rl/train_solo.py:1545）。
- 探针：`_stall_probe(env, last_hp, stall_count)`（rl/run_league.py:252）——每 `STALL_WINDOW=10` 步调用一次（rl/run_league.py:145），连续 `STALL_LIMIT=10` 次塔血零变化（`abs(hp-last_hp) < 1e-9`，rl/run_league.py:260）⇒ 早停（等价 100 步零塔损，rl/run_league.py:146）。
- 早停结算：`settle_stall(env.battle, cfg.stall_draw_margin)`（rl/train_solo.py:1552；`stall_draw_margin` 默认 `0.05`，rl/config.py:168），其纯函数内核是 `settle_stall_from_counts(lost0, lost1, min_pct0, min_pct1, margin=0.05)`（rl/run_league.py:213）：皇冠不同 → 决定性判胜负；皇冠相同且最低塔血百分比差 `< margin` → 返回 `None`（记平局=失败）。
- 奖惩注入：`win_bonus` / `-lose_penalty` / `-_draw_penalty(cfg)` 写进 `ep_rew[-1]`（rl/train_solo.py:1554-1559）；平局惩罚取 `rw.get("draw_penalty", rw.get("lose_penalty", 10.0))`（rl/train_solo.py:164-167）。
- 收尾：算 GAE → 把本局全部 transition 追加进 `transitions`（rl/train_solo.py:1563-1572）→ `_new_episode_reset(winner)` → `continue`（跳过本 step 的采样）。

#### B.1.2 ② 采样一步（rollout 决策）

| 步骤 | 函数 / 表达式 | 行号 | 说明 |
|---|---|---|--- |
| plan 选择 | `rng.random() < _SOLO_PROPHET_PROB`（=0.3，rl/train_solo.py:129）→ `prophet.plan(env.get_prophet_state())`，否则 `bp.plan(env.battle, belief.state(), obs)` | rl/train_solo.py:1578-1580 | 先知（特权状态）以 0.3 概率注入 |
| plan 向量化 | `plan.to_vector()` | rl/train_solo.py:1581 | 维度 `PLAN_DIM`（rl/plan_space.py:187） |
| 信念 token | `belief.encode(obs, None)` | rl/train_solo.py:1582 | 粒子数 128（rl/train_solo.py:1171） |
| 记录上一隐状态 | `init_hidden = hidden` | rl/train_solo.py:1583 | PPO 重放用（保证 ratio 是有效 IS 比） |
| 前向出动作 | `main.act(obs, belief_tok, plan_vec, env.get_action_mask, hidden=hidden, deterministic=False)` → `(bundle, lp, val, hidden, masks)` | rl/train_solo.py:1584-1586 | `masks` 是 decoder 各步掩码序列，供重放 |
| 推进一步 | `env.step(bundle)` → `obs2, reward, term, trunc, info` | rl/train_solo.py:1587 | |
| 缓冲 | `ep_obs/ep_belief/ep_plan/ep_bundle/ep_lp/ep_val/ep_rew/ep_term/ep_trunc/ep_masks/ep_init` 各 append | rl/train_solo.py:1589-1595 | 局内缓冲，局末清算 |
| GRU 活力帧 | `_probe["frames"].append((obs, belief_tok, plan_vec))`；`len > 96` 丢最老 | rl/train_solo.py:1591-1593 | 只前向、不推进 env |
| 信念更新 | `belief.update(obs2, info.get("opp_played"))` | rl/train_solo.py:1596 | |

#### B.1.3 ③ 局结束清算（条件触发）

- 触发：`done or (len(ep_rew) >= cfg.max_ep_steps and not overtime_open(env.battle))`（rl/train_solo.py:1599）；`max_ep_steps` 默认 360（rl/config.py:120）；加时窗口见 `overtime_open`（rl/overtime.py:30，`NORMAL_TIME_S=180.0`/`OVERTIME_END_S=300.0`）。
- 截断标记：`truncated = (not term) and len(ep_rew) >= max_ep_steps and not overtime_open(...)`（rl/train_solo.py:1600-1601）。
- 无胜者时补到期结算：`timeout_winner(env.battle)`（rl/train_solo.py:1606；实现 rl/run_league.py:179 —— 先比皇冠，皇冠相同再比存活塔最低血百分比，完全相等才 `None`）。
- **截断 bootstrap**（P1-7）：`truncated and virt is None` ⇒ `ep_trunc[-1] = True`，`last_val = main.value(obs, belief_tok, plan_vec, hidden)`；否则 `last_val = 0.0`（rl/train_solo.py:1618-1622）。
- GAE 与入队：`_gae_pair(ep_rew, ep_val, ep_term, ep_trunc, last_val)` → 写 `transitions`（键：`obs/belief/plan/bundle/old_logprob/adv/returns/masks/init_hidden/adv_const/adv_gap`，rl/train_solo.py:1623-1633）。
- 局间重置与对手采样：`_new_episode_reset(winner)`（rl/train_solo.py:1513-1539）：`_probe["games"] += 1` → `opp_pool.record(winner)`（PFSP 回填）→ `opp_pool.sample()` → `env.opponent = side` → `deck1_factory` 设置（仅 `kind=="defend"` 且有 `deck_pool`）→ `env.reset()` → `belief.reset` → **全部 `ep_*` 与 `obs/hidden/last_hp/stall_count` 经 `nonlocal` 清空**（rl/train_solo.py:1522-1539；注释明确指出漏 `nonlocal` 会造成"adv=-671、value loss 43 万"的事故形态）。
- 对手状态重置：`isinstance(env.opponent, FollowerOpponent)` ⇒ `env.opponent.reset()`（rl/train_solo.py:1636-1637）。

#### B.1.4 ④ PPO 更新（条件触发）

- 触发：`len(transitions) >= cfg.update_interval`（默认 128，rl/config.py:112）⇒ rl/train_solo.py:1639。
- 批：`batch = transitions[:cfg.batch_size] if len(transitions) > cfg.batch_size else transitions`（`batch_size` 默认 128，rl/config.py:111）⇒ rl/train_solo.py:1640-1641。
- 日志用计数：`n_play`（含 deploy 的样本数）、`avg_size`（平均子动作数）（rl/train_solo.py:1642-1644）。
- 调用：`stats = ppo.update(batch, adv_alt=_adv_alt)`（rl/train_solo.py:1653；`adv_alt` 仅惰性检验测量臂非 None）。
- 缓冲推进：`transitions = transitions[cfg.batch_size:] if len(transitions) > cfg.batch_size else []`（rl/train_solo.py:1677）。
- EV 原料累积：`ppo.last_ev_pairs` 追加进 `_probe["ev_pairs"]`，总帧数超 **200000** 就丢最老的更新批（rl/train_solo.py:1679-1686）。
- 单步日志：`[solo step N] policy=... value=... vraw=... EVb=... EVin=... entropy=... deploy=...% bundle=... ratio=... clip=...% gs=... adv=...±... gnorm=... n=...`（rl/train_solo.py:1701-1709）。**口径提示**：单步 `EVb` 是批内（128 连续帧）口径，判读用评估行的池化 EV（代码注释 rl/train_solo.py:1694-1698）。

#### B.1.5 ⑤ 评估（条件触发，**先评估后同步**）

```text
if cfg.steps_per_eval and step % cfg.steps_per_eval == 0:        # 全点
    eval_and_write(step); last_eval_step = step
elif cfg.anchor_every and step % cfg.anchor_every == 0 and step != cfg.total_steps:
    anchor_point(step)                                            # C 方案轻量锚点
```
（rl/train_solo.py:1717-1723）

- 顺序红线：评估**先于**冻结副本同步（rl/train_solo.py:1711-1716 注释：旧顺序在 `copy_every | steps_per_eval` 时使评估对手恒为"刚同步的 main 自己" ⇒ 镜像局、曲线结构性无信息量）。
- `eval_and_write(step)`（rl/train_solo.py:1268-1408）内部顺序：`_sync_controls_once()` → 主评估（`eval_solo_parallel` 或 `eval_solo`）→ `history[:] = _dedup_history(history, step)`（同 step 去重，rl/train_solo.py:1284 + :224）→ 池化 EV 与配套统计 → GRU 活力探针 → `cum_games/stall_*` → 三组对照 → `write_solo_state` → `_check_gates` → 同步 `baseline_prev` → 打印 → 绝对强度报警 → `_persist(step)`。
- `anchor_point(step)`（rl/train_solo.py:1454-1479）只跑 `baseline_rand` 一块 + `_persist` + `write_solo_state`（省掉一个全点 3/4 成本，注释 rl/train_solo.py:1456-1458）。
- 起始评估：`cfg.eval_at_start and start_step == 0` ⇒ `eval_and_write(0)`（rl/train_solo.py:1482-1484）；`eval_at_start` 默认 True（rl/config.py:126）。
- 收尾：主循环结束后 `save_checkpoint(main, cfg.solo_main_path())`；若 `last_eval_step != cfg.total_steps` 再 `eval_and_write(cfg.total_steps)`（rl/train_solo.py:1736-1738）。

#### B.1.6 ⑥ 冻结副本同步（条件触发）

- 触发：`cfg.solo_copy_every and step % cfg.solo_copy_every == 0`（`solo_copy_every` 默认 2000，rl/config.py:198）⇒ rl/train_solo.py:1726。
- 动作：`_sync_frozen_copy(main, opp)`（rl/train_solo.py:170-172，实现 `opp.load_state_dict(main.state_dict())`）→ `frozen_step = step` → `opp_pool.refresh_hist(step)`（rl/train_solo.py:1727-1733）。

#### B.1.7 ⑦ 快照落盘（`_persist`）

`_persist(step)`（rl/train_solo.py:1410-1452）：
- `save_checkpoint(main, cfg.solo_main_path())`（`solo_main.pt`）；
- `save_checkpoint(main, cfg.solo_ckpt_path(step))`（`solo_main_<step>.pt`，进 hist 池的原料）；
- `torch.save(ppo.opt.state_dict(), cfg.solo_opt_path())`（`solo_opt.pt`）；
- `run_state.json`：`step / solo_ckpt / solo_opt / config / device / value_norm / adv_norm / ppo_epochs / ppo_minibatch / ppo_shuffle / adv_inert / ret_scaler`（rl/train_solo.py:1440-1452）。`ppo_epochs/minibatch/shuffle` 写盘的原因写在 rl/train_solo.py:1445-1447：config.json 不参与 resume 解析，续训忘传 `--ppo-*` 会静默退回 1 次梯度步/更新。

#### B.1.8 触发条件常量总表（默认值）

| 常量 | 默认值 | 位置 | 语义 |
|---|---|---|--- |
| `update_interval` | 128 | rl/config.py:112 | 每 128 决策帧一次 PPO 更新 |
| `batch_size` | 128 | rl/config.py:111 | 每次更新取多少 transition |
| `solo_copy_every` | 2000 | rl/config.py:198 | 冻结副本同步间隔（并触发 `refresh_hist`） |
| `steps_per_eval` | 4000 | rl/config.py:103 | 全评估点间隔；`economy` 预设覆盖为 8000（rl/config.py:353） |
| `anchor_every` | 0（关闭） | rl/config.py:104 | 轻量锚点间隔，0=旧行为逐位不变 |
| `n_eval_games` | 40 | rl/config.py:116 | 每对评估局数 |
| `max_ep_steps` | 360 | rl/config.py:120 | 常规时间截断上限（`overtime_open` 可延长） |
| `eval_at_start` | True | rl/config.py:126 | 开局先评估一次 |
| `diagnose_every` | 10 | rl/config.py:147 | 每 N 次 update 做梯度成分分解 |
| `train_stall_stop` | True | rl/config.py:150 | 训练环僵局早停 |
| `stall_draw_margin` | 0.05 | rl/config.py:168 | 早停局低置信裁定降噪阈值 |
| `ppo_epochs / ppo_minibatch / ppo_shuffle` | 1 / 0 / False | rl/config.py:178-180 | "真正的 PPO 更新预算"，默认=旧行为 |
| `gamma / gae_lambda` | 0.997 / 0.95 | rl/config.py:127-128 | `economy` 预设 `gae_lambda`=0.99（rl/config.py:351） |
| `adv_norm / value_norm` | `scale` / `none` | rl/config.py:139、:144 | `economy` 预设 `value_norm="running"`（rl/config.py:356） |
| `eval_workers` | `min(16, os.cpu_count() or 1)` | rl/config.py:209 | 见 B.5.5 与 B.9 的口径不一致条目 |
| `RAND_ANCHOR_SEED / RAND_ANCHOR_EVAL_SEED / RAND_ANCHOR_WARN_FLOOR` | 99999 / 90000 / 0.35 | rl/train_solo.py:67-69 | 锚点权重种子 / 锚点评估种子 / 报警线（只报警不阻断） |
| `_OPP_MIX`（兜底） | frozen 0.1 / hist 0.6 / defend 0.2 / rand_anchor 0.1 | rl/train_solo.py:145 | 与 `config.DEFAULT_OPP_MIX`（rl/config.py:94）须同步 |
| `_HIST_POOL_MAX` | 12 | rl/train_solo.py:148 | hist 池上限 |
| `_PFSP_ALPHA / _PFSP_GATE_HI / _PFSP_GATE_PENALTY` | 0.20 / 0.85 / 0.2 | rl/train_solo.py:152-154 | 训练侧显式传入的 PFSP 参数 |
| `_SOLO_PROPHET_PROB` | 0.3 | rl/train_solo.py:129 | 训练期先知 plan 注入概率 |

---

### B.2 优势 / 回报计算（`compute_gae`）

**签名**：`PPOTrainer.compute_gae(rewards, values, dones, gamma=0.99, lam=0.95, truncated=None, last_value=0.0)`（rl/ppo.py:167-169），返回 `(adv, returns)`，均为 `np.float32` 数组。

**逐步公式**（反向循环 `for t in reversed(range(T))`，rl/ppo.py:180-190）：

1. `next_val`：
   - `t == T-1` 且 `truncated[t] and not dones[t]` ⇒ `next_val = last_value`（截断 bootstrap）；
   - 否则 `t == T-1` ⇒ `0.0`；
   - `t < T-1` ⇒ `0.0 if dones[t] else values[t+1]`（rl/ppo.py:181-187）。
2. `delta = rewards[t] + gamma * next_val - values[t]`（rl/ppo.py:188）。
3. `gae = delta + gamma * lam * (0.0 if (t == T-1 or dones[t]) else gae)`（rl/ppo.py:189）。
4. `adv[t] = gae`；结束后 `returns = adv + values`（rl/ppo.py:191）。

**默认值与调用方**：
- 函数默认 `gamma=0.99, lam=0.95`（rl/ppo.py:168）——这是给 `run_league`/`flow_league`/`train_follower`/`train_prophet` 共用时的旧行为锚。
- solo 训练实际传 `cfg.gamma`（默认 **0.997**，rl/config.py:127）与 `cfg.gae_lambda`（默认 **0.95**，`economy` 预设 **0.99**，rl/config.py:128、:351）：`_gae_pair` 内部 `PPOTrainer.compute_gae(rew, val, term, cfg.gamma, cfg.gae_lambda, truncated=trunc, last_value=last_val)`（rl/train_solo.py:1228-1230）。
- `truncated=None` 时按全 False 处理（rl/ppo.py:176-177）。

**归一化选项**（都在 `PPOTrainer.update` 内，作用于优势；rl/ppo.py:220-233）：
- `adv_norm="scale"`：`advs = advs / (std + 1e-8)`，仅当 `std > 1e-6`（rl/ppo.py:224-226）——不中心化；
- `adv_norm="none"`：原样（rl/ppo.py:227-228）；
- `adv_norm="batch"`：`(advs - mean) / (std + 1e-8)`（rl/ppo.py:229-231）；
- 其它值抛 `ValueError`（rl/ppo.py:232-233）。
- 生效优先级：`update(..., adv_norm=...)` 单次覆盖 > `self.adv_norm`（rl/ppo.py:222），而 `self.adv_norm` 来自构造参数（默认 `"batch"`，rl/ppo.py:88）；训练入口传 `cfg.adv_norm`（默认 `"scale"`，rl/config.py:139；构造点 rl/train_solo.py:1140）。

**回报尺度归一化**（`value_norm`，只缩放价值损失，不改返回值域）：
- `"running"`：`ret_scaler.update(rets_np)` 后 `s = ret_scaler.scale()`，`v_scale = s if s > 0 else 1.0`（rl/ppo.py:241-244）；统计**每份 rollout 更新一次**（不按 minibatch，rl/ppo.py:239-240 注释）。
- `"none"`/`""`/None：`v_scale = 1.0`（rl/ppo.py:245-246）；其它值抛 `ValueError`（rl/ppo.py:247-248）。
- `ReturnScaler` 是 Welford 在线均值/方差（rl/ppo.py:41-83）：`var = m2/(count-1)`（:63）、`std = sqrt(max(0,var))`（:66）、`scale()` 在 `std <= eps(1e-6)` 时返回 `0.0` 表示"不缩放"（:68-71）；`to_dict/from_dict` 供 run_state 往返（:73-83）。

**EV（explained_variance）定义**：`EV = 1 − MSE(v,R) / Var(R)`；`r.size < 2 or v.size != r.size` 返回 0.0；`Var(R) <= 1e-12` 返回 0.0（rl/ppo.py:144-165）。语义见 rl/ppo.py:146-155：`EV=0` 等价"恒定预测批均值"，`EV<0` 比常数预测还差；之所以不用 `value_loss` 判读，是因为它被 `s²` 除、量纲随回报尺度漂移。

---

### B.3 PPO 损失与优化（`update()` 内部逐步骤）

**构造**：`PPOTrainer(policy, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip=0.2, vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, adv_norm="batch", value_norm="none", diagnose_every=0, ret_scaler=None, n_epochs=1, minibatch_size=0, shuffle=False, seed=12345)`（rl/ppo.py:87-90）；优化器 `torch.optim.Adam(policy.parameters(), lr=lr)`（rl/ppo.py:113）。训练入口的实参见 rl/train_solo.py:1138-1143（`lr/gamma/gae_lambda/clip/vf_coef/ent_coef/max_grad_norm/adv_norm/value_norm/diagnose_every/n_epochs/minibatch_size/shuffle/seed` 全部来自 `cfg`）。

**逐步骤**（rl/ppo.py:194-356）：

1. `self.policy.train()`；空 `transitions` 直接返回零 stats 字典（rl/ppo.py:212-218）。
2. `adv_raw = [t["adv"]]`，按 `adv_norm` 归一化得到 `advs`（rl/ppo.py:220-233，见 B.2）。
3. `rets_np = [t["returns"]]`（rl/ppo.py:236）。
4. 价值尺度：`value_norm=="running"` ⇒ 先 `ret_scaler.update(rets_np)` 再取 `v_scale`（rl/ppo.py:241-244）。**整份 rollout 统计一次**，理由见 rl/ppo.py:239-240（按 minibatch 更新会让 `s` 抖动、跨 epoch 目标尺度不一致）。
5. 旧分支判定：`legacy = (n_epochs <= 1 and not shuffle and (minibatch_size <= 0 or minibatch_size >= n))`（rl/ppo.py:251-252）。
6. 诊断开关：`diag_on = bool(diagnose_every) and (updates % diagnose_every == 0)`（rl/ppo.py:253）。
7. **(A) legacy 分支**（rl/ppo.py:299-316）：整批一次 `_loss_pass(..., reduction="sum")` → `_apply_grad` → `grad_steps += 1`；stats 里 `grad_steps=1`、`ppo_epochs=1`、`ppo_minibatch=n`；`last_ev_pairs = pack["ev_pairs"]`。
8. **(B) 多轮分支**（rl/ppo.py:317-342）：先用 `torch.no_grad()` + `policy.evaluate_batch(...)` 算**更新前**整批预测 `v_pre`（rl/ppo.py:326-336），`last_ev_pairs = (v_pre, rets_np)`、`last_ev_pre = explained_variance(...)`；再 `_update_epochs(...)`；最后把末轮 in-sample EV 改名：`out["explained_variance_insample"] = out["explained_variance"]`，`out["explained_variance"] = last_ev_pre`（rl/ppo.py:340-342）。理由见 rl/ppo.py:318-325（沿用末轮预测会把"记住这 128 帧"误读成"critic 学会了"）。
9. 汇总 `adv_mean/adv_std`（原始未归一化优势）、`value_scale`、`n`（rl/ppo.py:343-346）；惰性检验统计仅在 `adv_alt` 给定时并入（rl/ppo.py:347-350）；`p_gnorm/v_gnorm` 仅在真的做了分解时带出（rl/ppo.py:351-355）。

**损失组成**（`_loss_pass`，rl/ppo.py:361-407）：
```text
ratio  = exp(lp_new - old_logprob)
surr1  = ratio * adv
surr2  = clamp(ratio, 1-clip, 1+clip) * adv
p_loss = -min(surr1, surr2)                      # reduction="sum" 为旧量纲；否则 .mean()
v_mse  = MSE(value, returns)                     # 同上 sum / mean
v_loss = v_mse / (v_scale * v_scale)
loss   = p_loss + vf_coef * v_loss - coef * ent_term
```
（rl/ppo.py:379-393）系数默认 `clip=0.2`、`vf_coef=0.5`、`ent_coef=0.01`（rl/ppo.py:87-88）；`ent_coef` 可在 `update(ent_coef=...)` 单次覆盖（rl/ppo.py:234）。诊断量在此同步产出：`ratio_mean`、`clip_frac`、`ev`、`ev_pairs`、`policy_loss/value_loss/value_loss_raw/entropy`（rl/ppo.py:395-407）。

**minibatch 划分**（`_plan_batches`，rl/ppo.py:467-475）：`order=range(n)`；`shuffle=True` 时用 `self.rng.shuffle(order)`（RNG 种子 = 构造参数 `seed`，rl/ppo.py:129）；`minibatch_size <= 0 or >= n` ⇒ 单一整批；否则按 `order[i:i+minibatch_size]` 切片。

**epoch 循环与梯度步**（`_update_epochs`，rl/ppo.py:477-547）：
- 外层 `for ep in range(self.n_epochs)`（:498），内层逐 minibatch：`_loss_pass(..., reduction="mean")`（:505-506）→ 诊断只在 **(ep==0, bi==0)** 做（:507-513）→ `_apply_grad(...)`（:514）→ `self.grad_steps += 1`（:517）。
- 统计口径（docstring rl/ppo.py:481-486 与实现一致）：`policy_loss/value_loss/entropy` 按样本数加权平均、跨全部轮次（:519-521、:537-538）；`ratio_mean/clip_frac` **只取最后一轮**（:535-536）；`grad_norm` 取最后一轮各 minibatch 均值（:526、:541）；`explained_variance` 与 `last_ev_pairs` 取**最后一轮池化**（:528-534、:542）。
- 输出 `ppo_epochs`、`ppo_minibatch`、`grad_steps = n_epochs * n_batches`（:543-546）。

**梯度裁剪与步进**（`_apply_grad`，rl/ppo.py:409-465）：诊断关闭时 `opt.zero_grad()` → `pack["loss"].backward()`（:460-461）；诊断开启时先分别对 `pack["p_loss"]` 与 `vf_coef*pack["v_loss"]` 求梯度范数 `p_gnorm/v_gnorm`（多两次反传，:424-439），再走原合并 backward（:457-458）；随后
`pack["grad_norm"] = float(torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm))`（:462-463）→ `self.opt.step()`（:464）。`max_grad_norm` 默认 0.5（rl/ppo.py:88）。
- 惰性检验附加：若同时给了 `pack_alt`（替代优势的 loss pack），在 `(ep==0,bi==0)` 上额外用 `torch.autograd.grad` 算策略损失对替代优势的梯度，写 `grad_cos = num/(p_gnorm*na)` 与 `grad_norm_ratio = p_gnorm/na`（rl/ppo.py:440-453）；**不改变被写入梯度的量**。
- 注释警示（rl/ppo.py:417-420）：`last_adv_inert_grad` **不在此处重置**（早期版本每次调用都清，把刚算出的 `grad_cos` 擦掉，出现 `n_grad_cos=0`），重置职责在 `update()` 开头（:298）。

**训练侧预算打印**（rl/train_solo.py:1144-1151）：
```text
_n_mb = ppo_minibatch if >0 else max(1, update_interval)
每 update_interval 帧的梯度步数 = ppo_epochs * ceil(update_interval / _n_mb)
```

---

### B.4 对手与联赛

#### B.4.1 solo 对手池（`_OpponentPool`，rl/train_solo.py:321-511）

- **构成与配比**：`self.mix = dict(cfg.opp_mix or _OPP_MIX)`（rl/train_solo.py:352）；`cfg.opp_mix` 默认 `DEFAULT_OPP_MIX = {"frozen":0.1,"hist":0.6,"defend":0.2,"rand_anchor":0.1}`（rl/config.py:94、:225）；`_OPP_MIX` 是 cfg 缺键时的兜底且值相同（rl/train_solo.py:145）。
- **采样**：`sample()`（:436-446）记录 `kind_counts` 后调 `_sample_kind()`（:448-477）：
  - 有 hist 池时：`r < mix["hist"]` ⇒ PFSP 采样 hist；否则 `r -= hist`；`r < mix["defend"]` ⇒ defend；`r < mix["defend"]+rand_anchor` 且锚点存在 ⇒ rand_anchor；否则 frozen（:452-467）。
  - 无 hist 池时按 `denom = 1 - mix["hist"]` 重新归一化（:468-477），注释（:439-442）说明这同时修掉了旧实现把 hist 概率空间误分给 defend 的缺陷。
- **PFSP 回填**：`record(winner)`：仅 `_last_kind == "hist"` 时 `score = {0:1.0, 1:0.0}.get(winner, 0.5)`，`_pfsp.update_winrate("main", hist_id, score)`（:479-485）。
- **hist 池的收集**：`_collect_hist_ckpts(folder, max_n=_HIST_POOL_MAX=12, extra_dirs=None)`（:175-221）——扫 `solo_main_<step>.pt`，`len(steps) > budget` 时用 `np.linspace(0, len(steps)-1, budget).astype(int)` 均匀抽（:202-203）；本目录优先，不足才从 `extra_dirs` 按传入顺序补（:205-221）。动态刷新：`refresh_hist(step)`（:414-434）重扫并在新增时打印"本目录 N / 新增 M / 累计对手局"；调用点是冻结副本同步处（:1732-1733）。
- **hist 稳定 id**：`_reindex_hist()` 用"父目录名 + 文件名步号"作 id（:403-412），注释解释旧实现用下标作 id 会在池增长时张冠李戴。
- **hist 载入**：`_ensure_hist(path)` 换目标才重载，且**直接用 `load_checkpoint` 返回的策略**（按 ckpt 元数据构造架构），注释（:490-492）标注"默认架构建网再 load_state_dict"是同类 bug 第三次。
- **defend 对手**：`SelfDefenderPolicy(seed=cfg.seed + 7, env=env, deck_pool=defender_deck_pool)`（:346-347）；其实现 rl/opponents.py:125-208：`passive_prob` 默认 0.6（rl/opponents.py:138），威胁帧调 `simulate_exchange.script_defender(battle, player_id)` 取反制，并把**世界坐标逆变换**到本地网格（player 1：`gx=round(17-wx)`, `gy=round(31-wy)`；player 0：`gx=round(wx-0.5)`, `gy=round(wy-0.5)`，rl/opponents.py:172-177）。
- **rand_anchor**：`_make_rand_anchor(cfg, belief_dim, device)`（:80-104）用 `RAND_ANCHOR_SEED=99999` 建 `FollowerPolicy`，构造前后保存/恢复 torch CPU RNG（:95-101）；belief_dim 由 `len(BeliefInference(opp_deck=env.deck1, n_particles=128, seed=0).encode(None, None))` 量取（:375-376）；包装为 `FollowerOpponent(..., deterministic=True)`（:378-382）。注释（:87-91）记载踩坑：锚点架构标志必须随 cfg，否则 worker `load_state_dict` 键集不匹配 → 每周期 worker 启动失败 → 静默降级串行。

#### B.4.2 PFSP（`rl/pfsp.py`）

- 公式：`P(opponent) ∝ (1 - winrate(main, opponent))^β`（模块 docstring rl/pfsp.py:3；实现 `weights()`，rl/pfsp.py:53-64）。
- 构造默认：`PFSP(beta=1.0, seed=0, alpha=0.05, gate_hi=1.0, gate_penalty=1.0)`（rl/pfsp.py:30-31）；校验 `beta >= 0`、`0 < alpha <= 1`、`0 <= gate_penalty <= 1`（:32-37）。
- 未采样对手按 **0 胜率（乐观先验）** 处理 ⇒ 权重高（:58-59）；门禁：`seen and gate_hi < 1.0 and winrates[key] > gate_hi` ⇒ `w *= gate_penalty`（:60-62）；权重下限 `max(w, 1e-6)`（:63）；`sample()` 归一化后 `rng.choices`（:66-71）。
- EMA 更新：`update_winrate(a, b, score_a, alpha=None)`：`prev*(1-a) + score_a*a`，`prev` 缺省 0.5（:46-51）。
- 训练侧显式参数：`self._pfsp = _PFSP(beta=1.0, seed=cfg.seed + 11, alpha=_PFSP_ALPHA=0.20, gate_hi=_PFSP_GATE_HI=0.85, gate_penalty=_PFSP_GATE_PENALTY=0.2)`（rl/train_solo.py:360-362，常量定义 :152-154）；且可被 `cfg.pfsp_alpha/pfsp_gate_hi/pfsp_gate_penalty` 覆盖（:356-359）。`_train_solo.py:398-401` 会打印这三个参数与重扫间隔。

#### B.4.3 联赛（`rl/league.py` / `rl/elo.py` / `rl/run_league.py`）

- `LeagueAgent`（dataclass）：`agent_id / kind="main" / policy=None / path=None / elo=1500.0`（rl/league.py:23-29）。
- `League(pfsp_beta=1.0, elo_k=32.0, seed=0)`（rl/league.py:33）：成员表 + `PFSP` + `Elo` + `history` + `elo_history` + `round_stats`（:34-42）。
- 采样与记账：`sample_opponent`（:52-57）排除自身后 `pfsp.sample`；`record_match`（:59-64）同时更新 Elo、双向 PFSP 胜率与 history。
- 冻结副本/快照：`register_checkpoint`（:66-84）用 `FollowerPolicy` 复制权重、`eval()` 且 `requires_grad_(False)`，新 id 为 `f"{agent_id}_ckpt{n}"`；`refresh_snapshot`（:86-102）刷新固定槽位 `f"{agent_id}_ckpt"`（`replace=True`，不重置 Elo）。
- 持久化：`save_state/load_state`（:146-181）存 `ratings / winrates / agents / history / exploiter_counter / ckpt_counter / elo_history / round_stats / total_steps`。
- Elo：`Elo(k=32.0, initial=1500.0)`（rl/elo.py:17-20）；`expected(r_a,r_b)=1/(1+10^((r_b-r_a)/400))`（:27-29）；`update(..., n_games=1)` 把 K 按局数缩放 `k_eff = k * max(1, n_games)`（:31-42）。
- 轮内聚合估计（评估曲线可信度上限）：`_round_estimates(pair_results)`（rl/run_league.py:448-473）：
  `D̂_ab = 400·log10((w+0.5)/(n−w+0.5))`（Laplace 平滑，:464）、`est = 1500 + mean_pairs(D̂)`、`SE ≈ 347.5/√games`（:469-472）。docstring（:449-455）说明逐局运行 Elo 是"有限记忆跟踪器"，单轮噪声 1σ≈±40 饱和，加局数不会收紧该曲线，要降噪只能做轮内聚合。
- 模式分派：`main()` 的 `--mode` 取值 `eval / run / flow / solo / flow-sweep-stream / flow-sweep-games5`（rl/run_league.py:1171-1173）；`run_league(cfg, ...)` 按 `n_envs` 与 `parallel` 分派 `_run_vec` / `_run_mp` / `_run_single`（:1143-1149）。`eval` → `evaluate_league(policies, kinds, n_games, seed, hidden_dim, max_steps=600, device="auto")`（:510-532）。
- **本部分未逐行读**：`_run_single`（:697-802）、`_run_vec`（:803-941）、`_run_mp`（:942-1142）内部循环细节 ⇒ 见 B.9。

#### B.4.4 flow 赛制（`rl/flow_league.py`）

- 6 个可训练模型 id：`FLOW_MODEL_IDS = ["push_flow","counter_flow","lockdown_flow","all_decks","random_deck","main"]`（rl/flow_league.py:56-57）。
- 卡组池：`build_flow_pools(cfg, n_random_decks=30)`（:67-87）返回 `(label, decks)` 的 OrderedDict——推进流/防守反击流/自闭流来自 `load_classified_decks` 按 archetype 分组，`all_decks` 与 `main` 用全量 200 套，`random_deck` 用 `sample_deck` 生成 `n_random_decks`（默认 30）套。
- 规模：`flow_pair_games(pools)` = `C(6,2)=15` 对、每对 `Σ|pool_i|×|pool_j|`（:90-97）；`scale_pools(pools, factor)` 按固定种子 `random.Random(20240903)` 每池截断到 `max(1, round(len×factor))`，docstring 记载 `factor=0.1` ⇒ 一次训练 1,488 局（:100-117）。
- 按模型的奖惩：`model_reward_weights(model_id, cfg)` = `DEFAULT_REWARD + cfg.reward + MODEL_REWARD_OVERRIDES[model_id]`（rl/config.py:404-412）；覆盖表 `MODEL_REWARD_OVERRIDES`：`push_flow` `elixir_diff_weight=0.7`、`counter_flow` 0.3、`lockdown_flow` 0.05，`main/all_decks/random_deck` 空（rl/config.py:62-69）。
- 主循环：`run_flow(cfg, resume=False, n_random_decks=30, pools=None, max_pairs=None, games_per_deck_pair=1, save=True, seed=None, quiet=False)`（:364-463）——双重循环 `for i in range(len(ids)): for j in range(i+1, len(ids))`（:422-423）逐对跑 `len(pool_a)*len(pool_b)*games_per_deck_pair` 局（:428），每局 `_play_one(...)`，缓冲达 `cfg.update_interval` 就 `_drain(buf, trainers[id], cfg.batch_size)`（:446-449），对结束再 drain 并落盘（:450-459）。返回 `(total_games, models, trainers)`。
- 断点：`_flow_progress_path(cfg)`（:153）与 `_save_flow_progress` / `_load_flow_resume`（:157-205）。
- sweep（数据效率 A/B）：`SWEEP_STRATEGIES`（:470）、`_sweep_trend`（:478）、`_write_sweep`（:491）、`run_flow_sweep`（:528）——**未逐行读**，见 B.9。

#### B.4.5 冻结副本同步与对照组

- 同步实现：`_sync_frozen_copy(main, opp)` = `opp.load_state_dict(main.state_dict())`（rl/train_solo.py:170-172），触发点 B.1.6。
- 三组对照（**只做对比、绝不进训练/迭代**，rl/train_solo.py:1179-1184 注释）：
  | 对照 | 权重来源 | 评估种子 | 行号 |
  |---|---|---|--- |
  | `baseline0` | 训练起点（`_sync_controls_once` 首次同步） | `cfg.seed + 50000 + step` | rl/train_solo.py:1240-1244、:1372-1373 |
  | `baseline_prev` | 上一评估点权重（每轮末刷新） | `cfg.seed + 60000 + step` | rl/train_solo.py:1374-1375、:1388 |
  | `baseline_rand` | 固定随机锚点（`RAND_ANCHOR_SEED`） | `RAND_ANCHOR_EVAL_SEED = 90000` | rl/train_solo.py:1196、:1378-1379 |
- 对照评估走 `eval_control(step, label, opp_model, seed)`（:1246-1266）：`eval_workers > 1` 用 `eval_solo_parallel`，否则 `eval_solo`；两者都 `record_replays=True` 但 `save_replays=False`（只算不落盘）；结果加 `stats["vs"] = label`。
- 绝对强度报警：`_rand_anchor_warns(winrate, floor=RAND_ANCHOR_WARN_FLOOR=0.35)`（:72-77），仅在 `vs == "baseline_rand"` 时检查（:1403-1407、:1472-1473），**只报警不阻断**。

---

### B.5 评估与诊断口径

#### B.5.1 solo 评估：`eval_solo` 与 `eval_solo_parallel`

`eval_solo(env, main, opp, n_games, max_steps, seed, cfg, record_replays=False, replays_dir=None, step=None, frozen_step=None, save_replays=True)`（rl/train_solo.py:713-799）：
- 每局：`FollowerOpponent(opp, env, belief=BeliefInference(opp_deck=env.deck1, n_particles=128, seed=seed+g), deterministic=True)` ⇒ `env.reset(seed=seed+2000+g)`、信念 `seed+1000+g`（:728-735）；主循环用 `main.act(..., deterministic=True)` 且 `steps < max_steps or overtime_open(env.battle)`（:745-763）；每 `STALL_WINDOW` 步做僵局探针（:746-749）。
- 无胜者时与训练同口径补结算：`timeout_winner`，`virt is None` 则 `ep_rew -= _draw_penalty(cfg)`（:764-773）。
- 统计：`winrate = (wins + 0.5*draws)/n`；`se = sqrt(max(0, winrate*(1-winrate))/n)`，`n<=1` 时取 0.5（:783-785）；stats 键 `step/wins/losses/draws/games/winrate/winrate_se/mean_reward`（:786-788）。
- 有回放时 `stats.update(behavioral_metrics(replays))`（:793），失败只打印不影响评估（:794-795）；`replays_dir` 与 `step` 齐备且 `save_replays` ⇒ `save_league_replays(replays, replays_dir/league_<step>.pkl)`（:796-798）。

`eval_solo_parallel(env, main, opp, n_games, max_steps, seed, cfg, n_workers=8, record_replays=False, replays_dir=None, step=None, frozen_step=None, save_replays=True)`（:941-1036）：
- `n_workers = min(n_workers, n_games)`；`< 2` 直接降级串行 `eval_solo`（:955-960）。
- `mp.get_context("spawn")`（:961）；`main_sd/opp_sd` 转 CPU state_dict（:963-964）；`chunks = [games[i::n_workers] for i in range(n_workers)]`（:966）；`env_kwargs` 含 `reward_weights/card_level/deck0/deck1/hidden_dim/n_total/eval_step/frozen_step/value_bypass/value_independent`（:967-972）。
- 启动前把 `CUDA_VISIBLE_DEVICES=""` 屏蔽子进程 CUDA，结束后恢复（:976-977、:1004-1008）。
- worker 目标 `_eval_worker_main(worker_id, main_sd, opp_sd, games, env_kwargs, seed_base, max_steps, n_particles, record, out_q)`（:802-913）：`torch.set_num_threads(1)` + `np.set_num_threads(1)`（:823-829）；每 worker 自建 env/策略/信念，**把 `0..n_total-1` 的 reset 链全部走一遍只打分配到的局**（:853-865，理由：串行复用单 env 时 `reset()` 基于上一局牌序继续洗牌，信念先验相同性靠这条链还原）；异常经 `out_q.put(("error", ...))` 回传（:908-913）。
- 收结果：`_collect_worker_results(procs, out_q, expected)`（:916-938）——`out_q.get(timeout=20)`，超时且无存活进程则返回失败信息（专门覆盖"worker 启动即崩、队列永不来消息"的 WinError 1455 形态）。
- 失败处理：任一 worker 失败 ⇒ `terminate()` 其余进程 → 打印 `[eval] 并行评估 worker 失败，降级串行: ...` → 返回串行 `eval_solo`（同种子结果等价，:993-1000）。
- 汇总口径与串行完全一致（`n = max(1,int(n_games))`、同一 winrate/SE 公式，:1009-1035）；worker 里 `n_particles=128` 硬编码传入（:984）。

#### B.5.2 指标定义

**行为指标** `behavioral_metrics(games)`（rl/train_solo.py:550-710），内部阈值：

| 指标 | 定义 | 阈值/常数 | 行号 |
|---|---|---|--- |
| `defense_invest_rate` | 敌过河帧中有我方 deploy 的比例 | 敌过河 = P1 troop `y < 16.0`（`RIVER`） | :573、:606、:699 |
| `engagement_rate` | 防守部署后 8s 内 5 格内敌我 troop 同框的比例 | 窗口 `8.0/0.5` 帧、曼哈顿距离 `< 5.0` | :644-655、:700 |
| `intercept_rate` | 落点在"敌→我方塔"直线路径 4 格内的比例 | 目标点 `(tx,ty)=(8.5,6.0)`、阈值 `<= 4.0` | :656-667、:701 |
| `response_latency_med` | 威胁开始→首次响应延迟中位数（秒） | — | :678-693、:702 |
| `unilateral_rate` | 非防守/非响应窗口的 deploy 占比 | 响应窗口 `RESP_WINDOW = 5.0` s | :574、:668-671、:703 |
| `tower_diff_avg` | 我方总塔血 − 对手总塔血（均值） | 缺省塔血 `[4824,3052,3052]` | :621-624、:704 |
| `bundle_multi_rate` | 同帧 ≥2 张 deploy 的帧占比 | — | :633-634、:705 |
| `deploy_per_game` | 每局平均 deploy 次数 | — | :706 |
| `elixir_avg` | 整局平均圣水（读 `fr["elixir0"]`） | — | :618-619、:707 |
| `ghost_rate` | 落点 `y >= 20` 的 deploy 占比（脏回放/越界金丝雀） | `y >= 20` | :630-632、:640、:708 |

同帧重复 deploy 去重（`(slot,x,y)` 唯一，:608-617），理由注释：`league_40000` 曾出现每帧 4× 重复 deploy 的脏帧。

**`evaluate.py` 的口径**（离线评测入口，与训练内评估是两套代码）：
- `run_eval(policy_path, n_games=50, opponent="random", seed=0, hidden_dim=None, max_steps=300, opponent_policy=None, ablation=None)`（rl/evaluate.py:168-301）。
- 指标：`WinRate = wins/max(1,n_games)`、`Mean Reward`、`Bundle 合法率 = 1 - bundle_illegal/bundle_total`、`Elixir Efficiency = rew/elixir_spent`（:276-278）、`Next-Card Acc`、`Next-Card Brier`、`Belief ECE`、`Hand Top4 Overlap`、`Mean Crown Diff / Mean Tower HP Diff`（:279-291）。
- ECE：`_ece(conf, acc, n_bins=10)`，等宽 10 分箱 `Σ (n_i/N)·|acc_i − conf_i|`（:41-49）。
- 消融：`ABLATION_VARIANTS = full / plan-off / belief-off / both-off`（:308-313）；`run_ablation` 计算 `Δ = full − variant`、`delta_se = hypot(se_full, se_variant)`、`z = Δ/se`，判定 `|z| >= 2` 为"贡献/损害显著"，否则"无显著差异（噪声内）"（:339-349）；落盘 JSON + CSV（:435-469）。
- plan 采纳探针：`_plan_region_hit` 用 `y < 15.5`（own）/ `y > 16.5`（enemy）/ `13.0 <= y <= 19.0`（bridge）、`x < 9.0` 或 `>= 9.0` 分边（:52-79）；`_record_adoption` 统计 `region/card/hold/budget` 四类采纳（:111-154）；`budget_ok` 判据 `spent <= elixir_budget × 当帧圣水 + 0.01`（:140-148）。
- 独立信念协议：`evaluate_belief(policy_path, n_games=500, seed=0, hidden_dim=None, max_steps=300)`（:477-523）。
- 返回：`winrate / mean_reward / wins / losses / draws / n_games / adoption / winrate_se`，其中 `winrate_se = sqrt(p(1-p)/N)`（:293-301）。
- CLI：`--policy/--n-games=50/--opponent=random/--opponent-policy/--seed=0/--hidden-dim/--max-steps=300/--belief-only/--ablation/--ablation-out`（:528-542）——**没有 `--mcts`**。

**EV 口径**：`PPOTrainer.explained_variance(values, returns)`（rl/ppo.py:144-165，公式见 B.2）。训练落盘的 EV 分三层：
- 单步日志 `EVb` = 批内（128 连续帧）更新前口径（rl/ppo.py:334-336、rl/train_solo.py:1694-1698 注释）；
- 评估行 `explained_variance` = **更新前池化**：把本评估窗口 `_probe["ev_pairs"]` 全部帧 `np.concatenate` 后算一次（rl/train_solo.py:1289-1294）；
- `explained_variance_batched` = 对同一批帧做 `np.random.default_rng(cfg.seed).permutation` 后按 `batch_size` 切块再取均值，用于量化批切片残余影响（:1295-1302）。
- `EVin`（`explained_variance_insample`）= 多轮分支末轮 in-sample 读数，仅用于量化过拟合（rl/ppo.py:341-342、rl/train_solo.py:1699-1700）。

**GRU 活力 / 价值通道统计（写进 history）**（rl/train_solo.py:1303-1362）：

| 键 | 定义 | 阈值（rl/diagnostics.py:56） |
|---|---|--- |
| `h_std` | 隐状态跨帧 std（逐维 std 后取均值） | `> 0.05`，`<= 0.05` 报警 |
| `gru_n_abs` | GRU 候选 `tanh(...)` 的 \|·\| 均值 | `< 0.9`，`>= 0.9` 报警 |
| `value_std` | value_head 输出跨帧 std（最近 96 帧探针） | 无独立报警 |
| `r_std` / `r_std_batch` | 评估窗口回报 std（池化 / 每 128 帧批内再平均） | 分母 |
| `value_std_ev` | **同窗口** `ev_pairs` 的 value std | 门槛分母口径修正的关键（:1311-1321） |
| `value_std_ratio` | `value_std_ev / (r_std + 1e-12)` | `> 0.3`，`<= 0.3` 报警 |
| `value_std_ratio_probe` | 跨窗口旧口径（`value_std / r_std_batch`），仅留档对照、**不可判读** | — |
| `vitality_warns` | `_diag.check_vitality(...)` 的告警列表 | 见下 |
| `cum_games` / `stall_games` / `stall_close_draws` | 累计局数 / 早停局数 / 早停且低置信记平的局数 | — |

`rl/diagnostics.py` 的判定实现：`gru_vitality(policy, frames, max_frames=96)`（:79-121，纯前向不推进 env；`h_std = H.std(axis=0).mean()` :117、`n_abs = mean(ns)` :118、`value_std = V.std()` :119）；`check_vitality(vit)`（:124-140）；`check_policy_architecture(policy)`（:143-163）在启动时静态检查 `enc_ln`/`grid_ln` 是否存在或是否被换成 `nn.Identity`（调用点 rl/train_solo.py:1115-1116）；`_gru_gate_stats`（:59-76）手动复现 GRUCell 内部以取候选饱和判据。

#### B.5.3 行为门禁（gates）

- 默认阈值：`cfg.gates = {"engagement_rate": {"rel": ">=", "frac": 0.5}, "ghost_rate": {"rel": "<=", "frac": 2.0}}`（rl/config.py:220-223）。
- `_check_gates(stats, cfg, path=None)`（rl/train_solo.py:234-306）与 `_write_gate_report(path, report)`（:309-317）：
  - 读已有 `gates.json` 取 `baseline`；**首个评估点只建基线不判定**（:262-271）；
  - 相对门禁 `{"rel","frac"}` ⇒ `thr = frac * baseline[name]`（:279-286）；
  - 数字门禁 ⇒ 绝对阈值，op 推断 `"<=" if name.endswith("_max") or name == "ghost_rate" else ">="`（:287-293）；
  - 不满足只 `print("[gate] WARN ...（只报警，不中断训练）")`（:301-306）。
- 调用点 rl/train_solo.py:1386。

#### B.5.4 断点续训与 `run_state` 往返（评估相关的状态面）

- `_load_run_state(cfg)`（rl/run_league.py:590）读 `run_state.json`；solo 恢复：`start_step`、`history`（由 `solo_state.json`）（rl/train_solo.py:1072-1083）。
- 权重优先级：`resume` 且 `rs["solo_ckpt"]` 存在 ⇒ **断点权重覆盖 `--main-init`**（:1099-1105）；`main_init` 仅在无断点时使用，且必须显式传 `plan_dim=PLAN_DIM`/`belief_dim`（:1085-1093，注释说明否则被旧 ckpt 元数据建成旧维度、`_sync_frozen_copy` 会 shape 失配）。
- 一致性护栏：`value_bypass`/`value_independent` 与 cfg 不一致时打印"必须 `--fresh` 重训"（:1108-1112）；PPO 预算与断点记录不一致时打印告警（:1117-1131）；`ret_scaler` 从 run_state 恢复（:1152-1156）；Adam 状态恢复失败则"从头但仍续训"（:1157-1161）。

#### B.5.5 评估 worker 并行机制与另一套 worker（`rl/workers.py`）

- **评估并行**：见 B.5.1 的 `eval_solo_parallel`（`train_solo.py` 内实现，针对评估局）。`eval_workers` 默认 `min(16, os.cpu_count() or 1)`（rl/config.py:209），CLI `--eval-workers`（rl/run_league.py:1246-1248）；**注意**该 CLI help 文本写"默认 0=串行"，与 dataclass 默认不一致 ⇒ B.9 第 1 条。
- **训练并行**（`n_envs > 1` 才用）：`rl/workers.py` 的 `worker_main(worker_id, seed, reward_weights, in_q, out_q, card_level=None)`（rl/workers.py:71-107）是跨进程 env 推演循环，消息协议写在模块 docstring（:7-20）：`("mask", partial_bundle)→("mask", mask)`、`("step", bundle)→("step", payload, reward, term, trunc, opp_played, winner, overtime_open)`、`("reset", spec)→("ready", payload)`、`None→("closed", None)`；`payload = (obs, plan_vec, belief_tok)` 由 worker 本地算（`_payload`，:45-51，先知注入概率硬编码 `rng.random() < 0.3`）；`_apply_opponent(env, spec)` 只支持 `{"type":"scripted", ...}` 与 `{"type":"none"}`（:54-68），模块注释明确"若联赛采样到学习型对手，worker 回退内置随机"。该文件的调度方是 `run_league._run_mp`（rl/run_league.py:942）。

---

### B.6 辅助训练脚本

> 来源核对方式：六个脚本全文阅读；行号用 `grep -n "def \|^class \|add_argument"` 复核。六个脚本**均未** import `rl/config.py` / `rl/workers.py`（grep 无命中）⇒ 这两个文件不在其调用链内。各脚本的函数清单见 B.8 对应小节。

#### B.6.1 train_belief.py（信念编码器监督训练）

- **训练什么**：监督训练 `NeuralBeliefEncoder`，两个头——下一张牌分类 `CrossEntropyLoss`（标签 `hidden["opp_next"]`，rl/train_belief.py:174、:204）与对手手牌多标签 `BCEWithLogitsLoss`（标签来自 `hidden["opp_hand"]`，rl/train_belief.py:175、:204）；总损失 `ce + bce`（rl/train_belief.py:204），按整局序列前向（`(1,T,D)`，rl/train_belief.py:198-202）。
- **数据来源**：① 现采 `collect_replays()`（`RLEnv` + 掩码随机采样 `sample_bundle`，rl/train_belief.py:33、:49、:55、:63）；② 离线 `--replays-path` 读 pickle（rl/train_belief.py:150-153）。
- **入口**：`if __name__ == "__main__"`（rl/train_belief.py:247）→ `train(...)`（:259-260）。
- **CLI**：`--epochs`=10（:249）、`--n-games`=50（:250）、`--seed`=0（:251）、`--out`=`belief_encoder.pt`（:252）、`--replays-path`=None（:253）、`--opponent`=None（:255）、`--max-steps`=600（:257）。**CLI 未暴露**的内部默认：`lr=1e-3`、`batch_size=64`、`n_games=50`、`val_frac=0.2`、`max_steps=600`（rl/train_belief.py:145-146）。
- **超参**：`hidden=64`、`num_classes=13`（:171）、温度候选网格 `[0.5,0.8,1.0,1.3,1.6,2.0,2.5,3.0,4.0,5.0]`（:138）、ECE `n_bins=10`（:117）；`Adam` 优化 `gru/next_head/hand_head/belief_proj`（:172-173）。`batch_size` 形参在函数体内无其他引用（grep 仅命中 :145）⇒ 该形参在当前实现中未被使用。
- **产物**：`torch.save({...}, out)`（:233-243），键 `gru/next_head/hand_head/belief_proj/in_dim/hidden/num_classes/hand_dim/temperature`（:234-242），默认 `belief_encoder.pt`。
- **调用链**：`RLEnv`(:24、:50) → `EpisodeReplay`(:25、:55) → `NeuralBeliefEncoder`/`build_feature`(:26、:81) → `ENTITY_NAMES`(:27、:84)。
- **与主训练管线的关系**：本文件内未见调用点；全仓未见 `import train_belief`（仅 `scripts/rl/train_belief.py:7` 的 `runpy` 转发 wrapper）。**待确认**是否有外部 Python 调用方（原因：本任务只允许从源码判读，未做全仓调用图）。

#### B.6.2 train_prophet.py（先知 Route A：特权观测 SB3-PPO）

- **训练什么**：在特权观测环境 `ProphetEnv` 上用 SB3 `PPO` 在线 RL。观测 = 标准 obs dict + `priv`（12 维：`opp_cycle`(8)/`len(ENTITY_NAMES)`、`opp_elixir/10.0`、`opp_towers/5000.0`，rl/train_prophet.py:32、:38-42）；动作 = 单卡 `MultiDiscrete([5, GRID_H, GRID_W])`（:60），经 `legacy_action_to_bundle` 回填（:70）；奖励来自 `env.step`（:71）。
- **入口**：`if __name__ == "__main__"`（:157）→ `main()`（:158）。
- **CLI**：`--total-timesteps`=100_000（:143）、`--seed`=0（:144）、`--save`=`prophet_ppo`（:145）。
- **超参**：PPO `n_steps=2048`、`batch_size=256`、`learning_rate=1e-4`、`n_epochs=4`、`target_kl=0.03`（:148-151）；网络 `features_dim=256`（:79）、`Embedding(len(ENTITY_NAMES),8)`（:81）、CNN `32→64(s2)→64(s2)`（:84-86）、融合 `Linear(cnn_out+5*8+3+PRIV_DIM, 256)`（:92）。
- **产物**：`model.save(args.save)`（:153）⇒ 默认 `prophet_ppo.zip`。
- **与主训练管线的关系**：`rl/train_follower.py:153-154` 在 `make_plan()` 内延迟 import `prophet_policy_to_plan` 并调用，仅当 `prophet_model is not None`（train_follower.py:152）；而 `run_training(prophet_model=None)`（train_follower.py:122）且 CLI 无对应参数（`**vars(args)`，train_follower.py:283）⇒ 走 CLI 时该蒸馏分支**不可达**（全仓 `prophet_model` 仅命中 train_follower.py:122/152/154）。**待确认**是否有外部 Python 调用方传入。

#### B.6.3 train_follower.py（跟随者 PPO 主训练 + 对手包装）

- **训练什么**：`PPOTrainer` 训练同刻多卡 `FollowerPolicy`。在线采集：`policy.act` → `env.step`（:179-190）；plan 由 `BeliefPlanner` 或 `ProphetPlanner` 产生（`make_plan`，:151-157），按 `plan_prophet_prob` 切换、按 `plan_dropout`/`belief_dropout` 置零（:168-175）；episode 末尾自算 GAE（`PPOTrainer.compute_gae`，:211-212），截断步显式标记 + `last_value` bootstrap（:204-210），transition 含 `init_hidden`（:178、:218）。
- **入口**：`if __name__ == "__main__"`（:265）→ `run_training(**vars(args))`（:283）。
- **CLI**：`--total-steps`=5000、`--batch-size`=128、`--update-interval`=128、`--lr`=3e-4、`--plan-prophet-prob`=0.3、`--plan-dropout`=0.1、`--belief-dropout`=0.1、`--opponent`=`random`、`--main-policy-path`=None、`--seed`=0、`--save`=`follower.pt`、`--eval-every`=2000、`--hidden-dim`=128、`--max-ep-steps`=600、`--init-from`=None（:267-281）。
- **超参**：`run_training` 默认 `n_envs=1`、`gamma=0.99`、`gae_lambda=0.95`、`clip=0.2`、`prophet_model=None`（:117-122）；`BeliefInference(..., n_particles=128)`（:138）；PPOTrainer 只显式传 5 个参数（:149），其余取 ppo.py 默认 `vf_coef=0.5/ent_coef=0.01/max_grad_norm=0.5/adv_norm="batch"/value_norm="none"/diagnose_every=0/n_epochs=1/minibatch_size=0/shuffle=False/seed=12345`（rl/ppo.py:87-89）；`ppo.update(transitions)` 只传 1 个实参（:229）⇒ `ent_coef/adv_norm` 回落实例默认（rl/ppo.py:222、:234）。
- **产物**：`save_checkpoint(policy, save)`（:238），默认 `follower.pt`。
- **与主训练管线的关系**：`run_training()` 的唯一外部调用点是 `rl/train_exploiter.py:105`；主管线只 import 本文件的 `FollowerOpponent`/`heuristic_opponent`（`rl/run_league.py:58`、`rl/train_solo.py:40`、`rl/flow_league.py:45`、`rl/evaluate.py:29` 等）。**注意**：`heuristic_opponent` 的实现是**掩码随机采样**（:33-48），不是手写策略。

#### B.6.4 train_bc.py（行为克隆预训练）

- **训练什么**：BC 预训练 `FollowerPolicy`。专家 = `BeliefPlanner.plan()` 的 `suggested_card`（:42-44）落到 `focus_region` 中心最近的**合法格**（`env.get_action_mask_for(0)`，:47-56）；损失 = 专家 bundle 的负对数似然 `-lp`，由 `policy.evaluate(obs, tok, plan, bundle, masks, hidden=None)` 给出（:107-108）；样本 = `(obs, belief_tok, plan_vec, bundle, masks)`（:74-76），在线采集 `collect()`（:97），对手固定 `opponent=None`（:61）。
- **入口**：`if __name__ == "__main__"`（:119）→ `train_bc(...)`（:129-131）。
- **CLI**：`--n-games`=50（:121）、`--epochs`=3（:122）、`--lr`=1e-3（:123）、`--hidden-dim`=128（:124）、`--seed`=0（:125）、`--out`=`follower_bc.pt`（:126）、`--max-steps`=600（:127）。
- **超参**：`BeliefInference(n_particles=128)`（:63、:92）、`Adam(lr=lr)`（:95）、每 epoch `np.random.permutation` 打乱（:103）、逐样本 `hidden=None`（:107）；`REGION_CENTERS` 8 区域（:33-37），未命中回落 `(9,16)`（:54），距离 `|dx|+|dy|`（:55）。
- **产物**：`save_checkpoint(policy, out)`（:115），默认 `follower_bc.pt`。
- **与主训练管线的关系**：产物喂 `train_follower --init-from`（`train_follower.py:281` 定义、:143-147 加载并断言 `plan_dim`/`belief_dim` 后 `load_state_dict`）。本文件内未见调用点（全仓无 `import train_bc`）。

#### B.6.5 train_exploiter.py（针对固定 main 的克制策略）

- **训练什么**：复用 `train_follower.run_training`，以 `opponent="main_policy"` + `main_policy_path` 针对固定 Main 训练（:105-115）；训练后加载双方权重**换边**评估（:117-119），胜率 ≥ 阈值则注册进联赛（:122-131）。胜负取 `env.battle.winner`，平局（None）记 0.5（:53、:68-71、:81-84）。
- **入口**：`if __name__ == "__main__"`（:136）→ `main()`（:137）。
- **CLI**：`--main-policy-path`（required，:91）、`--total-steps`=20000（:92）、`--batch-size`=128（:93）、`--update-interval`=128（:94）、`--lr`=3e-4（:95）、`--seed`=0（:96）、`--save`=None（None → `exploiter_{main_name}.pt`，:97、:104）、`--n-eval-games`=10（:98）、`--winrate-threshold`=0.55（:99）、`--league-state`=None（:100）。
- **超参**：训练侧全部透传（:106-114），其中 `eval_every=0`（关闭训练中评估，:114）；评估侧 `max_steps=300`（:27、:66、:80）、`n_particles=128`（:64、:78）、换边 seed 偏移 `+1000`（:75）、胜率分母 = `2×n_games`（:74、:85-86）。
- **产物**：训练权重经 `run_training(save=save)` 落盘（:113）；达标时 `lg.save_state(args.league_state)`（:130），未达标只打印（:132-133）。
- **与主训练管线的关系**：本文件内未见被调用点（全仓无 `import train_exploiter`；仅 `scripts/rl/train_exploiter.py:7` 的 `runpy` wrapper）。

#### B.6.6 train_baseline.py（单卡 PPO 对照 baseline）

- **训练什么**：SB3 `PPO` 单卡对照 baseline。`SingleCardAdapter` 把 bundle 动作退化为 `MultiDiscrete([5, 32, 18])`（:32），经 `legacy_action_to_bundle` 回填（:38）；在线 rollout（`model.learn`，:96），奖励来自 `env.step`（:39）。
- **入口**：`if __name__ == "__main__"`（:101）→ `main()`（:102）。
- **CLI**：`--total-timesteps`=100_000（:82）、`--seed`=0（:83）、`--save`=`baseline_ppo`（:84）、`--lr`=1e-4（:85）、`--n-envs`=1（:86）。**注**：`n_envs` 在 `main()` 内无其他引用（grep 仅命中 :86）⇒ 恒为单环境（:89）。
- **超参**：PPO `n_steps=2048`、`batch_size=256`、`n_epochs=4`、`target_kl=0.03`（:90-95）；`features_dim=256`（:46）、`Embedding(13,8)`（:49）、CNN `32→64(s2)→64(s2)`（:52-55）、`in_ch=(grid_shape[-1]-1)+8+4`（:50）、融合 `Linear(cnn_out+5*8+3,256)`（:60）。
- **产物**：`model.save(args.save)`（:97）⇒ 默认 `baseline_ppo.zip`。
- **与主训练管线的关系**：未见调用点（全仓无 `import train_baseline`；仅 `scripts/rl/train_baseline.py:7` 的 `runpy` wrapper）。

### B.7 推理期附加件

> 口径说明：「训练期」= `train_solo.py` / `run_league.py` 的训练循环；「推理/评估期」= `evaluate.py` 与 `run_league.py` 的联赛对局函数（`_run_side0` / `play_pair` / `eval_round_robin`）。`ppo.py` 对七个模块**零 import**（其导入清单 `ppo.py:27-38` 只含 math/os/random/sys/numpy/torch），只在 `update(transitions)` 里消费 `t["belief"]`/`t["plan"]`/`t["bundle"]` 三个**张量/对象字段**（`ppo.py:195`、`ppo.py:329-331`、`ppo.py:372-374`），不引用任何符号名。

#### B.7.1 mcts.py

**1) 训练期角色：训练期未见调用点。**
证据：在四个入口文件上 `grep -ni "RLMCTS\|enumerate_bundles\|node_value\|leaf_value\|MCTSConfig\|mcts"` → **0 命中**（exit 1）。`mcts.py` 自身也不 import 这四个文件（`mcts.py:21-35` 只 import core/card_utils/player/rl.action_bundle/rl.action_mask/rl.env_wrapper）。

**2) 推理/评估期角色：未接入。**
`evaluate.py` 的 argparse 全量参数为 `--policy/--n-games/--opponent/--opponent-policy/--seed/--hidden-dim/--max-steps/--belief-only/--ablation/--ablation-out`（`evaluate.py:528-542`），**没有 `--mcts`**；全文件无 `RLMCTS` 引用。`RLMCTS` 的 docstring 自述定位「推理时浅 MCTS …… 零训练风险的推理侧增强」（`mcts.py:1-3`、`mcts.py:331`）。
「待确认：`rl/selftest.py` 或其他脚本是否直接调用 `RLMCTS`（原因：可读清单不含这些文件，我未读）。」

**3) 核心数据结构/维度**

| 项 | 值 | 行号 |
|---|---|--- |
| `MCTSConfig.n_simulations` | `24` | mcts.py:44 |
| `max_depth` | `3` | mcts.py:45 |
| `leaf_horizon_s` | `8.0` | mcts.py:46 |
| `prior_top_k` | `8` | mcts.py:47 |
| `c_uct` | `1.4` | mcts.py:48 |
| `decision_frames` | `30` | mcts.py:49 |
| `dt` | `1/60` | mcts.py:50 |
| `max_bundles_per_node` | `64` | mcts.py:51 |
| `reward` dict | `crown_weight=8.0, crown_lose_weight=10.0, tower_dmg_opp=0.001, tower_dmg_self=0.0012, tower_dmg_late=0.002, tower_dmg_self_late=0.0022, elixir_diff_weight=0.5, elixir_diff_late=0.1` | mcts.py:52-57 |
| `_TOWER_HP_ANCHORS` | `[4824.0, 3052.0, 3052.0]` | mcts.py:65 |
| `_Node.__slots__` | `("battle","to_act","parent","action","children","untried","n_visits","sum_value")` | mcts.py:306-307 |
| `priors` dict 契约 | `{"slot": np.ndarray(K_MAX+2), "cell": np.ndarray(GRID_H*GRID_W)}`（docstring 口径） | mcts.py:147 |
| 候选元素 | `(bundle, slot_idx, cell_flat)` | mcts.py:149 |
| `OpponentFn` | `Callable[[object,int], Optional[ActionBundle]]` | mcts.py:327 |

「待确认：`GRID_H`/`GRID_W`/`action_mask.K_MAX` 的具体数值与 `reward` 中 `crown_lose_weight` 等键被谁读取（原因：`rl/action_mask.py`、`rl/env_wrapper.py` 不在可读清单；`node_value` 只直接读 `rw["crown_weight"]`（mcts.py:123），`tower_dmg_*`/`elixir_diff_*` 经 `_phase_weights(rw, battle.time)`（mcts.py:114）间接消费）。」

**4) 主要算法流程**

- `search()`（mcts.py:341-387）：`root = _Node(copy.deepcopy(battle), player_id)`（348）→ `_expand_actions`（349）→ 循环 `cfg.n_simulations` 次：① **选择**：`while node.untried is not None and not node.untried and node.children and depth < cfg.max_depth: node = self._uct_select(node)`（355-357）；② **扩展**：`node.untried is None` 时惰性枚举，`pop(0)` → `_apply` → `append(child)`（359-365）；③ **推演估值**：`value = leaf_value(node.battle, player_id, cfg)`（367）；④ **回传**：沿 `parent` 链累加 `n_visits`/`sum_value`，**不做符号翻转**（369-375）；终局 `best = max(root.children, key=lambda c: (c.n_visits, c.q()))`（381），`info = {"n_sims","elapsed_s","root_value","visits","wait"}`（382-386）。无子节点时返回空 `ActionBundle()` + `"wait": True`（377-380）。
- **UCT 选择**（`_uct_select`，471-479）：`exploit = c.q()`，`explore = cfg.c_uct * sqrt(log(max(1,node.n_visits)) / max(1,c.n_visits))`，取 `exploit+explore` 最大。**先验不进 UCT 项**——先验只用于候选排序/截断（191-197），故实现是 UCT 而非 PUCT。
- **叶估值**（`leaf_value`，128-136）：`deepcopy` 后推 `int(cfg.leaf_horizon_s / cfg.dt)` 步 `sim.step(cfg.dt)`（`game_over` 提前 break），返回 `node_value(sim, player_id, cfg)`。
- **值函数**（`node_value`，99-125）：`tw_opp, tw_self, edw = _phase_weights(rw, battle.time)`（114）→ `val = tw_opp*v_op - tw_self*v_me`（120）`+ rw["crown_weight"]*(op.get_crown_count() - me.get_crown_count())`（123）`+ edw*(me.elixir - op.elixir)`（124）。
- **塔损差异化定价**（`_tower_premium_loss`，68-96）：逐塔 `dmg_i = m_i - hps[i]`，`D = dmg_i/m_i`，`premium_loss = D + k*(D - D² + D³/3)`，`total += dmg_i * premium_loss * gate`；王塔且两公主塔存活时 `gate = DEFAULT_KING_GATE`（90-92）。
- **候选枚举**（`enumerate_bundles`，143-198）：每槽上限 `max_per_slot = cfg.max_bundles_per_node`（159、189-190）；无先验且为法术卡时按 `_cells_by_spell_value` 排序（170-173），否则 `_cells_by_threat`（174-177）；有先验时 `score = priors["slot"][si] + priors["cell"][ci]*1e-3` 降序截断到 `prior_top_k`（191-197）；每个候选过 `validate_bundle`（185-187）。
- **状态转移**（`_apply`，435-469）：先 `validate_bundle(sim, node.to_act, bundle)`，`ok` 则 `sim.use_ability` / `sim.deploy_card`（442-448）；本方节点再注入对手响应（450-460）；推进 `cfg.decision_frames` 步（462-465）；**子节点 `to_act` 恒被强制回根行动方**（466-468，注释：v1 简化 = 我方出手+对手同帧响应）。

**5) 符号清单**

| 符号 | 文件:行 | 作用 | 关键参数（名字=默认值） | 返回 |
|---|---|---|---|--- |
| `MCTSConfig` | mcts.py:43 | MCTS 配置 dataclass | `n_simulations=24, max_depth=3, leaf_horizon_s=8.0, prior_top_k=8, c_uct=1.4, decision_frames=30, dt=1/60, max_bundles_per_node=64, reward=default_factory(dict)` | 实例 |
| `_tower_premium_loss` | mcts.py:68 | 逐塔累计塔损（凹形溢价+王塔闸门） | `player` | `float` |
| `node_value` | mcts.py:99 | 塔血差+皇冠+资源账的值函数 | `battle, player_id, cfg` | `float` |
| `leaf_value` | mcts.py:128 | 叶推演后计 `node_value` | `battle, player_id, cfg` | `float` |
| `enumerate_bundles` | mcts.py:143 | 枚举单卡 bundle 并全量校验 | `battle, player_id, cfg, priors=None` | `list[(bundle, slot, cell)]` |
| `score` | mcts.py:193 | 先验排序打分闭包 | `item` | `float` |
| `validate_slots` | mcts.py:201 | 手牌槽圣水掩码 | `p` | `np.ndarray` |
| `_cells_by_threat` | mcts.py:207 | 缺省格序（近敌+低血塔加权） | `battle, player_id` | `np.ndarray` |
| `_foe_priority` | mcts.py:238 | 敌军优先级（低血塔更高） | `e` | `float` |
| `_is_spell_card` | mcts.py:248 | 卡名是否 spell | `card_name` | `bool` |
| `_cells_by_spell_value` | mcts.py:253 | 法术格序（覆盖价值降序） | `battle, player_id, card_name` | `np.ndarray` |
| `_spell_target_value` | mcts.py:286 | 法术目标价值（含击杀加成） | `e, spell_damage` | `float` |
| `_Node` | mcts.py:305 | 树节点 | — | 实例 |
| `_Node.__init__` | mcts.py:309 | 初始化节点字段 | `battle, to_act, parent=None, action=None` | `None` |
| `_Node.q` | mcts.py:319 | 平均价值 | — | `float` |
| `RLMCTS` | mcts.py:330 | 浅 MCTS 搜索器 | — | 实例 |
| `RLMCTS.__init__` | mcts.py:333 | 存 policy/opponent_fn/cfg | `policy=None, opponent_fn=None, cfg=None` | `None` |
| `RLMCTS.search` | mcts.py:341 | 主循环 select→expand→rollout→backup | `battle, player_id, obs=None` | `(ActionBundle, info)` |
| `RLMCTS._expand_actions` | mcts.py:391 | 本节点候选（等待+单卡/对手） | `node` | `list` |
| `RLMCTS._root_player` | mcts.py:408 | 根行动方 | — | `int` |
| `RLMCTS._priors` | mcts.py:411 | 策略网络 (slot,cell) logits 先验 | `node` | `Optional[dict]` |
| `RLMCTS._opponent_bundle` | mcts.py:427 | 调 `opponent_fn`（异常→None） | `battle, opp_id` | `Optional[ActionBundle]` |
| `RLMCTS._apply` | mcts.py:435 | 部署双方+推进一决策帧 | `node, action` | `_Node` |
| `RLMCTS._uct_select` | mcts.py:471 | Q+U 选子节点 | `node` | `_Node` |

---

#### B.7.2 plan_space.py

**1) 训练期角色：** 被训练循环调用。`PLAN_DIM` 在 `train_solo.py:35` 导入并用于建网络/加载 checkpoint（`train_solo.py:94`、`97`、`1091`、`1095-1096`、`1185-1190`）；训练循环每帧调 `bp.plan(...).to_vector()` / `prophet.plan(...).to_vector()`（`train_solo.py:1579-1581`）。`run_league.py` 同样（`run_league.py:56`、`682`、`733-734`）。

**2) 推理/评估期角色：已接入。** `evaluate.py:28` `from rl.plan_space import PlanToken`，主循环 `plan = bp.plan(env.battle, belief.state(), obs)`（`evaluate.py:196`）→ `plan_vec = plan.to_vector()`（`evaluate.py:198`）→ 送入 `policy.act(obs, tok, plan_vec, ...)`（`evaluate.py:203-204`）；信念协议同样（`evaluate.py:494-497`）。注意 `PlanToken` 这个名字在 evaluate.py 中**只出现于第 28 行的 import**，函数体内用的是 `bp.plan()` 的返回值（未出现 `PlanToken(...)` 构造）。

**3) 维度/布局**

- `PLAN_DIM = int(len(PlanToken().to_vector()))`（plan_space.py:187），**唯一常量源**（模块 docstring 自述，plan_space.py:3-5）。
- 布局（`to_vector`，124-160）：`intent_old(8) + region(8) + old_scalars(5)` = **旧 21 维前段**（155-156）→ `intent_new(len(MACRO_INTENTS)-8) + target(5) + hint(8) + threat(6)`（157）→ `elixir_budget(1)`（158）→ `hold(4)`（159）。
- 由源码列表实算：`MACRO_INTENTS`=旧 8（31-35）+ 新 13（36-49）= **21**；`FOCUS_REGIONS`=**8**（53-57）；`TARGET_KINDS`=**5**（60-62）；`PLACEMENT_HINTS`=**8**（65-74，逐项 none/pull_across/pull_aggro/support_zone/anti_spell_zone/bridge_front/king_front/intercept_mid）；`OPP_SPELL_THREATS`=**6**（77-79）；`hold`=4；⇒ `PLAN_DIM = 21 + 13 + 5 + 8 + 6 + 1 + 4 = 58`。
- 兼容锚：`_OLD_INTENT_COUNT = 8`（51）、`_OLD_PLAN_DIM = 21`（188）；旧意图不占新位（133-140）。
- 旧标量 5 维（143-149）：`suggested_card/4`、`bundle_size_hint`、`combo_hint`、`clip(risk_profile,0,1)`、`clip(tanh(value_estimate),-1,1)`；`COMBO_NONE/COMBO_TANK_SUPPORT/COMBO_SPELL_UNIT/COMBO_SPLIT_PUSH = 0,1,2,3`（86）。
- **口径漂移（源码内部不一致）**：模块 docstring 第 12 行写 `PLAN_DIM = 57`、第 11 行写 `placement_hint(7)`，与实算 58 / 8 不符（`PLACEMENT_HINTS` 实为 8 项，plan_space.py:65-74）。

**4) 主要算法流程**：`PlanToken.to_vector` 是纯离散化——旧 21 维逐位兼容（旧意图命中写旧组、新意图写 tail 组、未知回退 `cycle_and_wait`，133-140），随后追加 one-hot 组与 1 个标量、4 个 bit（150-160）。`from_old_layout` 用 `argmax` 反解旧 21 维（168-178），`value_estimate` 反解走 `arctanh`（178）。无搜索/学习成分。

**5) 符号清单**

| 符号 | 文件:行 | 作用 | 关键参数（名字=默认值） | 返回 |
|---|---|---|---|--- |
| `_one_hot` | plan_space.py:89 | 名字→one-hot，未知取默认位 | `name, choices, default_idx=0, dtype=np.float32` | `np.ndarray` |
| `PlanToken` | plan_space.py:99 | 战术意图 token dataclass | `macro_intent="cycle_and_wait", focus_region="own_center", suggested_card=None, bundle_size_hint=1, combo_hint=0, risk_profile=0.5, value_estimate=0.0, target_kind="none", placement_hint="none", opp_spell_threat="none", elixir_budget=1.0, hold_mask=0` | 实例 |
| `PlanToken.intent` | plan_space.py:117 | 便捷构造 | `cls, name, region="own_center", **kw` | `PlanToken` |
| `PlanToken.hold_slots` | plan_space.py:120 | hold_mask 命中槽位列表 | `self` | `List[int]` |
| `PlanToken.to_vector` | plan_space.py:124 | 离散化为定长向量 | `self` | `np.ndarray` |
| `PlanToken.from_old_layout` | plan_space.py:163 | 旧 21 维反解回 token | `cls, old_vec` | `PlanToken` |
| `PlanToken.zeros` | plan_space.py:182 | 全默认 token | `cls` | `PlanToken` |

---

#### B.7.3 action_bundle.py

**1) 训练期角色：** 结构被训练循环间接使用（由 `FollowerPolicy.act` 产出、`env.step` 消费），但训练代码里**不直接 import 它**——`train_solo.py` 无 `action_bundle` / `ActionBundle` / `SubAction` import；`train_solo.py:1643-1644` 只读 `t["bundle"].sub_actions`。`run_league.py:61` `from rl.action_bundle import ActionBundle, K_MAX`，其中 `ActionBundle` 在全文件中**只见于第 61 行的 import**，实际用的是 `K_MAX`（`run_league.py:346`）。

**2) 推理/评估期角色：已接入（部分）。** `evaluate.py:31` `from rl.action_bundle import K_MAX`，用 `1 <= sa.slot <= K_MAX` 过滤部署动作（`evaluate.py:123`、`evaluate.py:212`）与校验 `plan.suggested_card`（`evaluate.py:132`）；`bundle.size` 用于统计（`evaluate.py:206`）。`ActionBundle` / `SubAction` 类名在 evaluate.py 中未出现（用 `bundle.sub_actions` 属性访问）。

**3) 数据结构/布局**

- `K_MAX = 4`（action_bundle.py:28）——「单个决策步最多同时打出的卡数」。
- `SubAction`（43-56）：`kind: str = "deploy"`、`slot: int = 0`、`x: int = 0`、`y: int = 0`；坐标是**玩家本地坐标**（0..17 / 0..31），`slot=1..4` 表示 `player.cycle[slot-1]`，`slot=0` 为 no-op；`kind="ability"` 触发英雄技能（slot/x/y 忽略）。
- `ActionBundle`（76-77）：`sub_actions: List[SubAction] = field(default_factory=list)`；`__post_init__` 超 `K_MAX` 抛 `ValueError`（79-81）。
- 坐标换算唯一点：`sub_position`（31-39），P0 → `Position(x+0.5, y+0.5)`，P1 镜像 → `Position(17.5-x, 31.5-y)`。
- **注意 `K_MAX` 双源**：`mcts.py:34` 从 `rl.action_mask` 导入 `K_MAX`（与 action_bundle.py:28 的 `K_MAX` 是两个不同模块的常量）；「待确认：`action_mask.K_MAX` 是否等于 4（原因：`rl/action_mask.py` 不在可读清单）。」

**4) 主要算法流程**：无算法，纯数据契约——整包校验/整包提交语义写在模块 docstring（3-6）：「默认任一子动作非法即拒绝整包并施加惩罚，避免半执行状态」。实现侧只提供 `add`/`add_ability` 追加、`from_single`/`noop` 构造、`to_tuple` 旧接口降级（`size==0 → (0,0,0)`，97-99）。

**5) 符号清单**

| 符号 | 文件:行 | 作用 | 关键参数（名字=默认值） | 返回 |
|---|---|---|---|--- |
| `sub_position` | action_bundle.py:31 | 本地网格→世界坐标唯一换算 | `player_id, x, y` | `Position` |
| `SubAction` | action_bundle.py:43 | 单卡子动作 dataclass | `kind="deploy", slot=0, x=0, y=0` | 实例 |
| `SubAction.ability` | action_bundle.py:59 | 构造技能子动作 | `cls` | `SubAction` |
| `SubAction.card_name` | action_bundle.py:62 | slot→卡名（非法 None） | `self, player` | `Optional[str]` |
| `SubAction.to_position` | action_bundle.py:67 | 本地→世界坐标 | `self, player_id=0` | `Position` |
| `SubAction.to_tuple` | action_bundle.py:70 | 旧接口 `(slot,y,x)` | `self` | `Tuple[int,int,int]` |
| `ActionBundle` | action_bundle.py:76 | 同刻多卡动作包 | `sub_actions=list` | 实例 |
| `ActionBundle.__post_init__` | action_bundle.py:79 | 校验子动作数 ≤ `K_MAX` | `self` | `None`（越界抛 `ValueError`） |
| `ActionBundle.add` | action_bundle.py:83 | 追加 deploy 子动作 | `self, slot, x, y` | `ActionBundle` |
| `ActionBundle.add_ability` | action_bundle.py:87 | 追加 ability 子动作 | `self` | `ActionBundle` |
| `ActionBundle.size` | action_bundle.py:92 | 子动作数（property） | `self` | `int` |
| `ActionBundle.to_tuple` | action_bundle.py:95 | n≤1 降级旧 tuple | `self` | `Tuple[int,int,int]` |
| `ActionBundle.from_single` | action_bundle.py:102 | 单卡构造 | `cls, slot, x, y` | `ActionBundle` |
| `ActionBundle.noop` | action_bundle.py:106 | 空 bundle（等待） | `cls` | `ActionBundle` |
| `ActionBundle.contains_card` | action_bundle.py:109 | 是否含指定卡 | `self, player, card_name` | `bool` |

---

#### B.7.4 belief.py / bayes_filter.py

**1) 训练期角色：两者都在训练期活跃，但 `bayes_filter.py` 只被 `belief.py` 间接调用。**
- `belief.py`：`train_solo.py:32` 导入；训练循环 `belief = BeliefInference(opp_deck=env.deck1, n_particles=128, seed=cfg.seed)`（`train_solo.py:1171`）、逐帧 `belief.encode(obs, None)`（`train_solo.py:1582`）、`belief.update(obs2, info.get("opp_played"))`（`train_solo.py:1596`）；`belief_dim` 由 `len(BeliefInference(...).encode(None, None))` 动态量出（`train_solo.py:1066-1067`、`train_solo.py:375-376`、`train_solo.py:835-836`）。`run_league.py:53`、`678`、`735`、`744` 同构。
- `bayes_filter.py`：在 `train_solo.py`/`evaluate.py`/`run_league.py`/`ppo.py` 上 `grep -n "CycleBayesFilter\|bayes_filter"` → **0 命中**（exit 1）。唯一调用点是 `belief.py:27`（import）、`belief.py:265`（构造）、`belief.py:310`（`self.rule.update(card)`）、`belief.py:321-323`（`hand_probs`/`next_probs`/`entropy`）。

**2) 推理/评估期角色：已接入。** `evaluate.py:25` 导入 `BeliefInference`；`evaluate.py:175` 构造，主循环 `evaluate.py:199`（encode）、`evaluate.py:219`（update）、`evaluate.py:222-236`（`belief.state()` 的 next-card acc / Brier / ECE / hand top4 命中）；独立信念协议 `evaluate_belief`（`evaluate.py:477-505`，默认 `n_games=500`，`evaluate.py:477`）。

**3) 数据结构/维度**

- `NUM_CARDS = len(ENTITY_NAMES)`（belief.py:31）；`TENDENCIES = ["aggressive","defensive","cycle","spell_heavy","balanced"]`（32，长度 5）。
- `belief_token_dim(deck) = 2*len(deck) + 2 + len(TENDENCIES) + OPP_EVENT_K*OPP_EVENT_DIM`（belief.py:36-38）。
- `OPP_EVENT_K = 3`（47）；`OPP_EVENT_DIM = len(ENTITY_NAMES) + 3`（49）；单条事件行 = card one-hot + `x/17` + `y/31` + `min(max(Δt,0),10)/10`（52-60，注释称 16 维是旧口径）。
- 实际 `encode()` 拼接顺序（344-362）：`hand_probs(len(deck))` → `next_probs(len(deck))` → `[elixir_mean, uncertainty](2)` → `tendency_probs(5)` → `opp_event_token(3*OPP_EVENT_DIM)` → 可选神经 token(`hidden`)。
- `BeliefState`（108-116）：`deck=[]`、`hand_probs=None`、`next_probs=None`、`elixir_mean=5.0`、`elixir_std=1.0`、`intent_probs=None`、`tendency_probs=None`、`uncertainty=1.0`。
- `BeliefInference` 事件史 `deque(maxlen=16)`（belief.py:272）；`NeuralBeliefEncoder` 默认 `hidden=64, hand_dim=8, max_len=32`（belief.py:173-174）。
- `CycleBayesFilter`：`n_particles=128`、`seed=0`（bayes_filter.py:45）；`weights` 初值 `np.ones(n_particles)/n_particles`（51）。
- 「待确认：`ENTITY_NAMES` 的长度（进而 `belief_token_dim` 的数值刻度，如 177/563 之类）（原因：`ENTITY_NAMES` 定义于 `rl/observation.py`，不在可读清单；代码实际用 `len(BeliefInference(...).encode(None,None))` 运行时量取，见 `train_solo.py:1066-1067`）。」

**4) 主要算法流程**

*Bayes filter（`CycleBayesFilter`）——不是数值贝叶斯更新，而是「O(1) 队列确定性推进 + 粒子一致性筛选」：*
- **锁定流更新**（`update`，116-129）：若 `played_card in self._cycle[:4]` → `self._cycle = self._play(self._cycle, played_card)`（125），其中 `_play(perm, card) = [c for c in perm if c != card] + [card]`（72-74，与 `player.play_card` 队列规则一致）；否则 `_degrade()` 退回粒子相且**不推进**（127-129）。
- **粒子相更新**（131-147）：`perms = self._consistent(played_card)`，`_consistent` 只保留 `card in perm[:4]` 的粒子并 `_play` 推进后按 `tuple` 去重（108-114）；空解 → `_degrade()`（133-136）；否则等权重 `1/len(perms)`（139），`_run += 1`（140）。
- **重锁**：`_run >= 4 and len(observed) >= 4` 且最近 4 张互异且全在卡组内 → `_lock_from_tail(last4)`，规范 cycle = `[卡组序手牌] + last4`（143-147、76-87）。
- **查询**：锁定流 `hand_probs` 返回精确 0/1（157-159），`next_probs` 返回 `cycle[4]` 的 0/1（170-171），`entropy()` 恒 `0.0`（180-181）；粒子相返回频率（161-164、172-175）+ 手牌后验熵（182-184）。
- 数学依据写在模块 docstring（9-14）：从第 4 张起「当前手牌集合 = 卡组 − 最近 4 张（互异）」「下一张进手 = 第 k−3 张打出的牌」，与 40320 开局排列无关。

*Belief 组合层（`BeliefInference`）：*
- `_tick_elixir`（288-298）：按 `obs["time"]` 的 `dt` 以 `+dt/2.8` 回升，`min(10.0, ...)`。
- `update`（300-316）：`normalize_played` 过滤哨兵（101-103：`None`/`"None"`/`__ability__`/非 ENTITY_NAMES 全部丢弃）→ 事件入 `event_history`（306-307）→ `stat.update(card, x=..., y=...)`（312）→ `self._elixir_est -= Card(card).elixir`（314）→ `clip(0,10)`（315）。
- `state`（318-328）：`uncertainty = clip(rule.entropy()/log(8), 0, 1)`（323）、`elixir_std = max(0.5, |est-5|/5)`（327），最后 `normalize()` 补均匀分布（118-127）。
- `NeuralBeliefEncoder.encode`（204-213）：GRU 前向取 `h[-1]`，`belief_proj` 投影 `(1,hidden)` 后 `squeeze(0)` 返回 `(hidden,)`。

**5) 符号清单**

| 符号 | 文件:行 | 作用 | 关键参数（名字=默认值） | 返回 |
|---|---|---|---|--- |
| `belief_token_dim` | belief.py:36 | 无神经编码时 token 维度公式 | `deck` | `int` |
| `_event_row` | belief.py:52 | 单条事件→行向量 | `card, x, y, dt` | `np.ndarray` |
| `opp_event_token` | belief.py:63 | 最近 k 条事件展平（含 Δt） | `history, now=None, k=OPP_EVENT_K` | `np.ndarray` |
| `normalize_played` | belief.py:82 | 规范化出牌列表并过滤哨兵 | `opp_played` | `list[(card,x,y)]` |
| `BeliefState` | belief.py:108 | 信念状态 dataclass | `deck=list, hand_probs=None, next_probs=None, elixir_mean=5.0, elixir_std=1.0, intent_probs=None, tendency_probs=None, uncertainty=1.0` | 实例 |
| `BeliefState.normalize` | belief.py:118 | 缺失字段补均匀分布 | `self` | `BeliefState` |
| `StatisticalBelief` | belief.py:130 | 风格/路线倾向计数信念 | — | 实例 |
| `StatisticalBelief.__init__` | belief.py:133 | 初始化计数容器 | `self` | `None` |
| `StatisticalBelief.update` | belief.py:140 | 按卡型/落点半场累计 | `self, card, x=None, y=None` | `None` |
| `StatisticalBelief.probs` | belief.py:160 | 拉普拉斯平滑倾向概率 | `self` | `np.ndarray` |
| `NeuralBeliefEncoder` | belief.py:167 | GRU 压缩历史→token | — | 实例 |
| `NeuralBeliefEncoder.__init__` | belief.py:173 | 建 GRU/三个头与队列 | `self, in_dim, hidden=64, num_classes=NUM_CARDS, hand_dim=8, max_len=32` | `None` |
| `NeuralBeliefEncoder.reset_history` | belief.py:189 | 清空历史队列 | `self` | `None` |
| `NeuralBeliefEncoder.push_frame` | belief.py:192 | 压入一帧特征 | `self, feat` | `None` |
| `NeuralBeliefEncoder._history_tensor` | belief.py:195 | 队列→`(1,T,D)` 张量 | `self` | `torch.Tensor` |
| `NeuralBeliefEncoder.encode` | belief.py:204 | GRU 前向→token | `self, feat` | `np.ndarray` |
| `NeuralBeliefEncoder.load` | belief.py:216 | 从 ckpt 重建编码器 | `cls, path, in_dim=None` | `NeuralBeliefEncoder` |
| `build_feature` | belief.py:232 | 一帧观测→编码器输入 | `obs, opp_played` | `np.ndarray` |
| `BeliefInference` | belief.py:258 | 规则+统计+神经组合层 | — | 实例 |
| `BeliefInference.__init__` | belief.py:261 | 建 rule/stat/事件史 | `self, opp_deck, use_rule=True, use_stat=True, neural=None, n_particles=128, seed=0` | `None` |
| `BeliefInference.reset` | belief.py:274 | 重置全部子信念 | `self, opp_deck=None` | `None` |
| `BeliefInference._tick_elixir` | belief.py:288 | 按时间推进圣水估计 | `self, obs` | `None` |
| `BeliefInference.update` | belief.py:300 | 用出牌更新信念/圣水/事件史 | `self, obs, opp_played, opp_x=None, opp_card_type=None, elixir_est=None` | `BeliefState` |
| `BeliefInference.state` | belief.py:318 | 汇总为 `BeliefState` | `self` | `BeliefState` |
| `BeliefInference._now` | belief.py:330 | 当前决策时刻（缓存回退） | `self, obs` | `float` |
| `BeliefInference.encode` | belief.py:344 | 拼 belief_token | `self, obs=None, opp_played=None` | `np.ndarray` |
| `CycleBayesFilter` | bayes_filter.py:42 | 8 卡循环队列信念 | — | 实例 |
| `CycleBayesFilter.__init__` | bayes_filter.py:45 | 存卡组/RNG，均匀先验 | `self, deck, n_particles=128, seed=0` | `None` |
| `CycleBayesFilter.locked` | bayes_filter.py:57 | 是否精确锁定流（property） | `self` | `bool` |
| `CycleBayesFilter.reset` | bayes_filter.py:61 | 清观测与锁定态 | `self, deck=None` | `None` |
| `CycleBayesFilter._play` | bayes_filter.py:72 | 出牌移队尾（staticmethod） | `perm, card` | `list` |
| `CycleBayesFilter._lock_from_tail` | bayes_filter.py:76 | 用最近 4 张重建 cycle | `self, last4` | `None` |
| `CycleBayesFilter._resample_uniform` | bayes_filter.py:89 | 均匀去重重采样粒子 | `self` | `None` |
| `CycleBayesFilter._degrade` | bayes_filter.py:102 | 退回粒子相 | `self` | `None` |
| `CycleBayesFilter._consistent` | bayes_filter.py:108 | 粒子一致性筛选+推进 | `self, card` | `list[list]` |
| `CycleBayesFilter.update` | bayes_filter.py:116 | 出牌→锁定推进/粒子筛选/重锁 | `self, played_card` | `None` |
| `CycleBayesFilter.hand_probs` | bayes_filter.py:151 | 每卡在手牌概率（锁定 0/1） | `self` | `np.ndarray` |
| `CycleBayesFilter.next_probs` | bayes_filter.py:167 | 每卡为下一张概率 | `self` | `np.ndarray` |
| `CycleBayesFilter.entropy` | bayes_filter.py:178 | 手牌后验熵（锁定=0） | `self` | `float` |

---

#### B.7.5 prophet.py

**1) 训练期角色：** 被训练循环调用，作为「教师/监督信号」。`train_solo.py:34` 导入，`prophet = ProphetPlanner()`（`train_solo.py:1163`），逐帧 `use_prophet = rng.random() < _SOLO_PROPHET_PROB`（`train_solo.py:1578`，`_SOLO_PROPHET_PROB = 0.3` 定义于 `train_solo.py:129`），命中则 `plan = prophet.plan(env.get_prophet_state())`（`train_solo.py:1579`）。`run_league.py:60` 导入，`709`/`732-733`（solo 主循环）与 `816`/`853-854`（`_run_vec`，概率写死 `0.3`）。

**2) 推理/评估期角色：未接入。** 在 `evaluate.py` 上 `grep -n "Prophet\|prophet"` → **0 命中**（exit 1）；`evaluate.py` 的规划器只有 `BeliefPlanner`（`evaluate.py:176`、`482`）。即评估只跑 bp 路径（30% prophet 帧只存在于训练与联赛）。

**3) 核心数据结构/维度**：无自主张量；输入是 `env.get_prophet_state()` 返回的特权 `full_state` dict，被读取的键（源码可见）：`entities`（`63`、`81`）、`my_cycle`（`142`、`173`）、`my_elixir`（`143`）、`my_towers`（`366`）、`opp_cycle`（`234`、`396`、`429`、`504`）、`opp_elixir`（`281`、`428`、`489`、`494`、`498`）、`opp_towers`（`329-332`）、`opp_crown`/`my_crown`（`485`）、`time`（`326`、`487`）；输出 `PlanToken`（`464`）。「待确认：`get_prophet_state()` 的键定义与张量形状（原因：定义在 `rl/env_wrapper.py`，不在可读清单）。」跨模块常量全部从 `belief_planner` 导入（`prophet.py:37-43`）：`PRESSURE_THRESHOLD`、`KING_ACTIVATE_PRINCESS_HP`、`LANE_SPLIT_X`、`BRIDGE_Y`、`OWN_HALF_EDGE`、`LATE_S`、`SOFT_CONTROL_CARDS`、`TRADE_SPELL_CARDS`、`FINISH_SPELL_CARDS`、`TANK_CARDS`、`PULL_TARGET_CARDS`、`BACKLINE_CARDS`、`BACKLINE_HARASSER_CARDS`、`ACE_CARDS`、`_SPELL_THREAT_KIND`。

**4) 主要算法流程**
- `plan(full_state)`（464-516）：先 `_threat_and_my_pressure`（466），然后**按固定优先链**依次调用 12 个检测器，第一个非 `None` 即返回（469-476）：`_soft_control → _spell_trade → _protect_backline → _pull → _punish → _push_commit → _spell_finish → _setup_wait → _king_activate → _anti_spell → _save_ace → _cycle_small`；全部未命中则走**旧 8 意图回退**（478-516）。
- **回退逻辑分支**（480-490）：`threat >= PRESSURE_THRESHOLD and threat >= my_pressure*0.8` → `defend_left/right`；`my_pressure >= PRESSURE_THRESHOLD` → `push_left/right`；`opp_crown > my_crown` → push；`time > 180 and my_elixir >= opp_elixir+1` → `counterpush`；`opp_elixir < 2.0` → push。
- **特权字段消费（P1-16）**：`bundle_hint = 2` 当 `my_elixir >= 6` 且双向压力均超阈（500-501），但若 `opp_cycle[:4]` 含 spell 则收敛为 1（503-505）；`risk` 按圣水差调（494-499）。
- **术语/阈值继承**（值定义在 belief_planner）：`PRESSURE_THRESHOLD = 2.0`（belief_planner.py:51）、`BRIDGE_Y = 16.0`（54）、`OWN_HALF_EDGE = 15.0`（55）、`LANE_SPLIT_X = 9.0`（56）、`LATE_S = 120.0`（57）、`KING_ACTIVATE_PRINCESS_HP = 800.0`（104）。

**5) 符号清单**

| 符号 | 文件:行 | 作用 | 关键参数（名字=默认值） | 返回 |
|---|---|---|---|--- |
| `_is_tower` | prophet.py:45 | 卡名是否含 Tower | `name` | `bool` |
| `_region_from_intent` | prophet.py:49 | 旧 8 意图→region 映射 | `intent` | `str` |
| `_units` | prophet.py:63 | 特权状态非塔且属卡表实体 | `fs, player_id` | `list` |
| `_mean_x` | prophet.py:73 | 单位平均 x（空=9.0） | `units` | `float` |
| `_threat_and_my_pressure` | prophet.py:78 | 特权威胁/我方压力统计 | `fs` | `(float,float)` |
| `_closest_enemy` | prophet.py:92 | 最接近我方塔的敌单位 | `fs` | `dict\|None` |
| `_pressing_enemy` | prophet.py:102 | 有敌单位进半场/桥头 | `fs` | `bool` |
| `_side` | prophet.py:107 | x→left/right | `x` | `str` |
| `_own_region` | prophet.py:111 | x→own_left/right | `x` | `str` |
| `_enemy_region` | prophet.py:115 | x→enemy_left/right | `x` | `str` |
| `_opposite_enemy_region` | prophet.py:119 | 重心反侧进攻路 | `x` | `str` |
| `_hand_slot` | prophet.py:124 | 卡在手牌槽位 1..4 | `cycle, card_name` | `int\|None` |
| `_spell_threat_in` | prophet.py:128 | 前 depth 张首张强法术 | `cycle, depth=6` | `str\|None` |
| `_pick_suggested` | prophet.py:140 | 按 intent 启发式选槽 | `fs, intent` | `int\|None` |
| `ProphetPlanner` | prophet.py:163 | 特权状态启发式先知 | — | 实例 |
| `ProphetPlanner._soft_control` | prophet.py:168 | 压境威胁→软控法术 | `fs` | `PlanToken\|None` |
| `ProphetPlanner._spell_trade` | prophet.py:183 | 高费后排→伤害法术 | `fs` | `PlanToken\|None` |
| `ProphetPlanner._protect_backline` | prophet.py:206 | 保后排（含手牌预判） | `fs` | `PlanToken\|None` |
| `ProphetPlanner._pull` | prophet.py:254 | 血牛桥头带→拉扯 | `fs` | `PlanToken\|None` |
| `ProphetPlanner._punish` | prophet.py:279 | 对手低圣水→另一路 | `fs` | `PlanToken\|None` |
| `ProphetPlanner._push_commit` | prophet.py:302 | 坦克推进→跟输出 | `fs` | `PlanToken\|None` |
| `ProphetPlanner._spell_finish` | prophet.py:325 | 后期低血塔→磨塔法术 | `fs` | `PlanToken\|None` |
| `ProphetPlanner._setup_wait` | prophet.py:348 | 无压力+有坦克→蓄力 | `fs` | `PlanToken\|None` |
| `ProphetPlanner._king_activate` | prophet.py:364 | 公主塔残血→激活王塔 | `fs` | `PlanToken\|None` |
| `ProphetPlanner._anti_spell` | prophet.py:392 | 直读牌序法术→防溅射 | `fs` | `PlanToken\|None` |
| `ProphetPlanner._save_ace` | prophet.py:410 | 藏终结卡 hold_mask | `fs` | `PlanToken\|None` |
| `ProphetPlanner._cycle_small` | prophet.py:447 | 无压力+小费牌→过牌 | `fs` | `PlanToken\|None` |
| `ProphetPlanner.plan` | prophet.py:464 | 优先链+旧 8 意图回退 | `full_state` | `PlanToken` |

---

#### B.7.6 belief_planner.py

**1) 训练期角色：** 训练循环的一等公民。`train_solo.py:33` 导入，`bp = BeliefPlanner()`（`train_solo.py:723`、`train_solo.py:1162`），主循环 `plan = bp.plan(env.battle, belief.state(), obs)`（`train_solo.py:1579-1580`，未命中 prophet 的 ~70% 帧）；评估 worker 内也建 `bp = BeliefPlanner()`（`train_solo.py:849`、`train_solo.py:882`）。`run_league.py:54` 导入，`708`/`733`（solo）与 `815`/`855`（`_run_vec`）；联赛对局函数 `_run_side0` 每帧 `plan = bp.plan(env.battle, belief.state(), obs)`（`run_league.py:293`）；`play_pair` 每局新建 `BeliefPlanner()`（`run_league.py:424`、`433`）。

**2) 推理/评估期角色：已接入（评估的唯一规划器）。** `evaluate.py:26` 导入，`evaluate.py:176` / `482` 构造，`evaluate.py:196` / `494` 逐帧调用；其输出既作网络输入（`plan_vec`）又作**采纳率探针**标签（`evaluate.py:111` `_record_adoption`，读 `plan.macro_intent`/`focus_region`/`suggested_card`/`elixir_budget`，`evaluate.py:120-141`）。

**3) 核心数据结构/维度**：无张量；输入 `(battle, BeliefState, obs)`，输出 `PlanToken`。模块级常量（全部本文件定义）：

| 常量 | 值 | 行号 |
|---|---|--- |
| `PRESSURE_THRESHOLD` | `2.0` | 51 |
| `BRIDGE_Y` / `OWN_HALF_EDGE` / `LANE_SPLIT_X` / `LATE_S` | `16.0` / `15.0` / `9.0` / `120.0` | 54/55/56/57 |
| `SOFT_CONTROL_CARDS` | `("Freeze","Vines","Tornado")` | 60 |
| `TRADE_SPELL_CARDS` | 11 张（Fireball…Tornado） | 62-63 |
| `FINISH_SPELL_CARDS` | 6 张 | 65 |
| `TANK_CARDS` / `PULL_TARGET_CARDS` | 10 / 9 个卡名 | 67-68 / 71-72 |
| `TANKY_HP` / `MELEE_RANGE` / `PULL_CHEAP_COST` | `1600.0` / `2.0` / `3.0` | 78/80/82 |
| `SAVE_BEHIND_ELIXIR` / `SAVE_FULL_ELIXIR` / `SAVE_HOLD_ALL` | `2.0` / `9.95` / `0b1111` | 85/87/89 |
| `SINK_TANK_CARDS` / `FRONT_TANK_CARDS` | 9 / 13 个卡名 | 91-92 / 94 |
| `BACKLINE_CARDS` / `BACKLINE_HARASSER_CARDS` | 14 / 8 个卡名 | 97-99 / 101-102 |
| `KING_ACTIVATE_PRINCESS_HP` | `800.0` | 104 |
| `TOWER_HP_PER_ELIXIR_EARLY` | `500.0` | 107 |
| `_SPELL_THREAT_KIND` | 10 条卡名→威胁枚举 | 110-114 |
| `_HAND_PROB_THRESHOLD` | `0.55` | 115 |
| `TOWER_HP_PER_ELIXIR_LATE`（局部） | `50.0` | 601 |

**4) 主要算法流程**
- `plan(battle, belief, obs=None)`（772-849）：`threat, my_pressure = _enemy_pressure(battle)`（773）、`p = battle.players[0]`（774，**写死 player0 视角**），然后同序优先链（777-784）逐检测器返回首个非 None：`_soft_control → _spell_trade → _protect_backline → _pull → _punish → _push_commit → _spell_finish → _setup_wait → _king_activate → _anti_spell → _save_ace → _cycle_small`。
- **回退 + 9j 过河触发**（792-804）：`tu0 = _closest_threat(battle)`，若 `tu0.position.y < BRIDGE_Y` 则 `crossed = tu0` 且 intent 直接取 `defend_left/right`（797-798）；否则按 `threat >= 2.0 && threat >= my_pressure*0.8` → defend、`my_pressure >= 2.0` → push、`threat > 0.0` → `defend_king`（799-804）。
- **risk 由信念不确定性驱动**（806-812）：`risk = clip(0.75 - 0.5*belief.uncertainty, 0.1, 0.9)`；若 `belief.next_probs.max() > 0.6` 再 `+0.15`（上限 0.95）。
- **拦截几何 hint 分派**（828-838）：`crossed` 非空时，`fy = crossed.position.y`；`fy < 9.0` → 高血坦克(`hitpoints>=2500` 或 `tidTarget=="TID_TARGETS_BUILDINGS"`) 取 `pull_aggro`，否则 `king_front`；`fy >= 9.0` → `intercept_mid`。
- **belief 驱动型意图**：`_punish` 读 `belief.elixir_mean > 2.5` 即放弃（512）；`_anti_spell` 经 `_opp_spell_threat_of` 用 `_HAND_PROB_THRESHOLD = 0.55` 作**严格大于**阈值筛（326-338）；`_protect_backline` 预判版同样用 `> _HAND_PROB_THRESHOLD` 查 `BACKLINE_HARASSER_CARDS`（423-433）；`_setup_wait` 读 `belief.elixir_mean > p.elixir + SAVE_BEHIND_ELIXIR` 则放弃（630-633），攒满 `SAVE_FULL_ELIXIR = 9.95` 沉底、否则在手牌含后排时返回 `hold_mask=SAVE_HOLD_ALL (=0b1111)`（650-665）。
- **法术经济账**：`_spell_cast_value`（253-306）调 `spell_module.best_cast`，返回 `(score, tower_value, pure_tower)`；`tower_value = Σ dmg×mult / TOWER_HP_PER_ELIXIR`，`mult` 由 `tower_value_mult(ratio, king, princesses_alive)` 给出、塔锚 `4824.0/3052.0`（291-299）；异常一律吞掉返回 `(0.0, 0.0, False)`（305-306）。`_spell_finish` 用 `tower_value_late = score - tv_early + tv_early*(500/50)` 做双倍期折算，`pure_tower and tower_value_late < 卡费` 则跳过（605-609）。

**5) 符号清单**

| 符号 | 文件:行 | 作用 | 关键参数（名字=默认值） | 返回 |
|---|---|---|---|--- |
| `_is_tower` | belief_planner.py:118 | 卡名是否含 Tower | `name` | `bool` |
| `_deployable_entity` | belief_planner.py:122 | 是否可部署单位/建筑 | `e` | `bool` |
| `_unit_card` | belief_planner.py:133 | 卡名→Card（非卡名 None） | `name` | `Card\|None` |
| `_tanky_or_melee` | belief_planner.py:141 | 拉扯对象口径（建筑/高血/近战） | `c` | `bool` |
| `_is_front_tank_name` | belief_planner.py:151 | 前排主体判定 | `name` | `bool` |
| `_enemy_pressure` | belief_planner.py:159 | 双方半场推进压力 | `battle` | `(float,float)` |
| `_enemy_main_x` | belief_planner.py:176 | 敌方非塔重心 x | `battle` | `float` |
| `_my_main_x` | belief_planner.py:182 | 我方非塔重心 x | `battle` | `float` |
| `_region_from_intent` | belief_planner.py:188 | 旧 8 意图→region | `intent` | `str` |
| `_side` | belief_planner.py:202 | x→left/right | `x` | `str` |
| `_own_region` | belief_planner.py:206 | x→own_left/right | `x` | `str` |
| `_enemy_region` | belief_planner.py:211 | x→enemy_left/right | `x` | `str` |
| `_opposite_enemy_region` | belief_planner.py:216 | 重心反侧进攻路 | `x` | `str` |
| `_pick_suggested_card` | belief_planner.py:221 | 按 intent 启发式选卡 | `battle, player_id, belief, intent` | `int\|None` |
| `_hand_slot` | belief_planner.py:249 | 卡在手牌槽位 1..4 | `p, card_name` | `int\|None` |
| `_spell_cast_value` | belief_planner.py:253 | 法术落点估值（工具③） | `battle, player_id, card_name` | `(float,float,bool)` |
| `_closest_threat` | belief_planner.py:309 | 最近我方塔的敌单位 | `battle` | 实体`\|None` |
| `_threat_unit_is_pressing` | belief_planner.py:321 | 威胁是否已过河/桥头 | `e` | `bool` |
| `_opp_spell_threat_of` | belief_planner.py:326 | 手牌后验→法术威胁 | `belief` | `str\|None` |
| `BeliefPlanner` | belief_planner.py:341 | 基于信念的规划器 | — | 实例 |
| `BeliefPlanner.__init__` | belief_planner.py:344 | 存后验采样开关 | `self, use_posterior_sampling=False, n_samples=8` | `None` |
| `BeliefPlanner._soft_control` | belief_planner.py:350 | 压境威胁→软控法术 | `battle, p, threat, belief` | `PlanToken\|None` |
| `BeliefPlanner._spell_trade` | belief_planner.py:366 | 远程脆皮进半场→法术 | `battle, p, threat, belief` | `PlanToken\|None` |
| `BeliefPlanner._protect_backline` | belief_planner.py:399 | 保后排（含信念预判） | `battle, p, threat, belief` | `PlanToken\|None` |
| `BeliefPlanner._pull` | belief_planner.py:449 | 拉扯/拦路（7g 口径） | `battle, p, threat, belief` | `PlanToken\|None` |
| `BeliefPlanner._punish` | belief_planner.py:510 | 读 belief 圣水→压反侧 | `battle, p, threat, belief` | `PlanToken\|None` |
| `BeliefPlanner._push_commit` | belief_planner.py:534 | 前排推进→优先真后排 | `battle, p, threat, belief` | `PlanToken\|None` |
| `_ok` | belief_planner.py:555 | 跟牌候选判定闭包 | `card` | `bool` |
| `BeliefPlanner._spell_finish` | belief_planner.py:582 | 后期低血塔→磨塔法术 | `battle, p, threat, belief` | `PlanToken\|None` |
| `BeliefPlanner._setup_wait` | belief_planner.py:617 | 攒费窗口 hold_mask=1111 | `battle, p, threat, belief` | `PlanToken\|None` |
| `BeliefPlanner._king_activate` | belief_planner.py:668 | 公主塔残血→激活王塔 | `battle, p, threat, belief` | `PlanToken\|None` |
| `BeliefPlanner._save_ace` | belief_planner.py:698 | 藏 ace hold_mask | `battle, p, threat, belief` | `PlanToken\|None` |
| `BeliefPlanner._anti_spell` | belief_planner.py:734 | 信念法术威胁→防溅射 | `battle, p, threat, belief` | `PlanToken\|None` |
| `BeliefPlanner._cycle_small` | belief_planner.py:753 | 无压力+小费牌→过牌 | `battle, p, threat, belief` | `PlanToken\|None` |
| `BeliefPlanner.plan` | belief_planner.py:772 | 优先链+旧回退+拦截 hint | `battle, belief, obs=None` | `PlanToken` |

---

**跨模块要点（一句话式）**：`mcts.py` 是唯一完全未接入的模块（训练/评估双侧 0 命中，`--mcts` flag 不存在）；`bayes_filter.py` 只作为 `belief.py` 的内部规则层存在；`prophet.py` 只在训练/联赛侧以 0.3 概率注入（`train_solo.py:129`、`train_solo.py:1578`、`run_league.py:732`、`run_league.py:853`），评估侧未接入；`belief_planner.py` 是训练与评估共用的唯一规划器，且 `plan()` 的视角**写死为 `battle.players[0]`**（belief_planner.py:774）。

### B.8 全量函数索引

**范围**：`src/clasher_new/rl/` 下**全部 38 个 `.py` 文件**中的每一个类、函数、方法（含嵌套闭包，均在下表中列出并标注所属外层函数）。**行号口径**：`def` / `class` 语句所在行（由 AST 解析源码得到，可用 `grep -n "^\s*\(def\|class\) "` 复核）。**列口径**：`关键参数` 直接取函数签名的参数名与默认值（默认值缺失即无默认值）；`返回` 取返回注解，无注解记 `—`；`一句话作用` 优先取源码 docstring 首行，无 docstring 的取已核对的描述映射。共 **711** 个符号。

#### B.8.0 按字母排序的函数名 → 文件:行号 快速索引

> 同名符号在同一行内以逗号分隔列出全部位置；共 **595** 个唯一名字。

- `__call__` → rl/opponents.py:118, rl/opponents.py:205, rl/train_follower.py:84
- `__getattr__` → rl/train_baseline.py:41, rl/train_prophet.py:74
- `__init__` → rl/bayes_filter.py:45, rl/belief.py:133, rl/belief.py:173, rl/belief.py:261, rl/belief_planner.py:344, rl/elo.py:17, rl/env_wrapper.py:271, rl/env_wrapper.py:292, rl/follower.py:155, rl/human_play.py:66, rl/league.py:33, rl/mcts.py:309, rl/mcts.py:333, rl/opponents.py:86, rl/opponents.py:138, rl/pfsp.py:30, rl/ppo.py:48, rl/ppo.py:87, rl/run_league.py:111, rl/selftest.py:1666, rl/selftest.py:2666, rl/selftest.py:2673, rl/selftest.py:2679, rl/selftest.py:2686, rl/train_baseline.py:29, rl/train_baseline.py:46, rl/train_follower.py:60, rl/train_prophet.py:49, rl/train_prophet.py:79, rl/train_solo.py:336
- `__post_init__` → rl/action_bundle.py:79
- `__repr__` → rl/env_wrapper.py:285
- `_active_push_tanks` → rl/action_mask.py:292
- `_adopt_agg` → rl/evaluate.py:393
- `_adoption_seed` → rl/evaluate.py:82
- `_adoption_summary` → rl/evaluate.py:89
- `_agent` → rl/dashboard.py:377
- `_agent_label` → rl/dashboard.py:330
- `_anti_spell` → rl/belief_planner.py:734, rl/prophet.py:392
- `_apply` → rl/mcts.py:435
- `_apply_grad` → rl/ppo.py:409
- `_apply_opponent` → rl/workers.py:54
- `_augment` → rl/train_prophet.py:66
- `_B` → rl/selftest.py:1670, rl/selftest.py:2672
- `_backline_min_gap_m` → rl/action_mask.py:326
- `_backline_placement_illegal` → rl/action_mask.py:338
- `_BT` → rl/selftest.py:2684
- `_build_league` → rl/run_league.py:672
- `_build_priv` → rl/train_prophet.py:35
- `_bundle_cards` → rl/run_league.py:343
- `_card_cached` → rl/action_mask.py:317
- `_card_cost` → rl/action_mask.py:30
- `_cells_by_spell_value` → rl/mcts.py:253
- `_cells_by_threat` → rl/mcts.py:207
- `_check_gates` → rl/train_solo.py:234
- `_closest_enemy` → rl/prophet.py:92
- `_closest_threat` → rl/belief_planner.py:309
- `_collect_deaths` → rl/env_wrapper.py:583
- `_collect_hist_ckpts` → rl/train_solo.py:175
- `_collect_worker_results` → rl/train_solo.py:916
- `_comp_rms` → rl/selftest.py:4273
- `_consistent` → rl/bayes_filter.py:108
- `_cuda_hint` → rl/run_league.py:70
- `_cycle_small` → rl/belief_planner.py:753, rl/prophet.py:447
- `_deck_factory_of` → rl/run_league.py:353
- `_dedup_history` → rl/train_solo.py:224
- `_default_paths` → rl/decks.py:92
- `_defend_action` → rl/opponents.py:155
- `_degrade` → rl/bayes_filter.py:102
- `_deploy_ledger` → rl/env_wrapper.py:561
- `_deployable_entity` → rl/belief_planner.py:122
- `_drain` → rl/flow_league.py:242
- `_draw_penalty` → rl/train_solo.py:164
- `_E` → rl/selftest.py:1672
- `_ece` → rl/evaluate.py:41
- `_effective_card` → rl/action_mask.py:39
- `_enc_norm` → rl/selftest.py:4232
- `_encode` → rl/follower.py:300
- `_encode_batch` → rl/follower.py:430
- `_encode_batch_parts` → rl/follower.py:397
- `_encode_parts` → rl/follower.py:268
- `_enemy_in_my_half` → rl/action_mask.py:241
- `_enemy_main_x` → rl/belief_planner.py:176
- `_enemy_pressure` → rl/belief_planner.py:159
- `_enemy_region` → rl/belief_planner.py:211, rl/prophet.py:115
- `_engine_lookup` → rl/decks.py:42
- `_ensure_hist` → rl/train_solo.py:487
- `_ensure_play` → rl/dashboard.py:2008
- `_episode_arrays` → rl/train_belief.py:72
- `_eval_and_snapshot` → rl/run_league.py:642
- `_eval_worker_main` → rl/train_solo.py:802
- `_event_row` → rl/belief.py:52
- `_expand_actions` → rl/mcts.py:391
- `_flow_progress_path` → rl/flow_league.py:153
- `_flush_episode` → rl/flow_league.py:223
- `_foe_priority` → rl/mcts.py:238
- `_force_utf8_stdout` → rl/run_league.py:1152
- `_gae_pair` → rl/train_solo.py:1221
- `_gru_gate_stats` → rl/diagnostics.py:59
- `_hand_slot` → rl/belief_planner.py:249, rl/prophet.py:124
- `_history_tensor` → rl/belief.py:195
- `_hits_dead_enemy_tower` → rl/action_mask.py:367
- `_hk` → rl/selftest.py:4277
- `_hp_state` → rl/flow_league.py:206
- `_ids` → rl/observation.py:156
- `_intents` → rl/selftest.py:1389
- `_is_front_tank_name` → rl/belief_planner.py:151
- `_is_spell_card` → rl/mcts.py:248
- `_is_tower` → rl/belief_planner.py:118, rl/prophet.py:45
- `_kind` → rl/replay.py:94
- `_king_activate` → rl/belief_planner.py:668, rl/prophet.py:364
- `_load_flow_resume` → rl/flow_league.py:166
- `_load_run_state` → rl/run_league.py:590
- `_lock_from_tail` → rl/bayes_filter.py:76
- `_loss_pass` → rl/ppo.py:361
- `_make_env` → rl/human_play.py:52, rl/run_league.py:579
- `_make_opp` → rl/run_league.py:366
- `_make_opponent` → rl/evaluate.py:157
- `_make_policy_and_tokens` → rl/selftest.py:122
- `_make_rand_anchor` → rl/train_solo.py:80
- `_make_trainer` → rl/run_league.py:584
- `_mask_or_fallback` → rl/follower.py:379
- `_mean_x` → rl/prophet.py:73
- `_med` → rl/train_solo.py:1423
- `_min_alive_tower_pct` → rl/run_league.py:157
- `_mk_env` → rl/selftest.py:393
- `_my_main_x` → rl/belief_planner.py:182
- `_new_agent` → rl/dashboard.py:350
- `_new_episode_reset` → rl/train_solo.py:1513
- `_Node` → rl/mcts.py:305
- `_norm` → rl/ppo.py:433
- `_now` → rl/belief.py:330
- `_ok` → rl/belief_planner.py:555
- `_one_hot` → rl/plan_space.py:89
- `_opp` → rl/train_follower.py:37
- `_opp_min_hand_cost` → rl/action_mask.py:231
- `_opp_spell_threat_of` → rl/belief_planner.py:326
- `_opponent_bundle` → rl/mcts.py:427
- `_OpponentPool` → rl/train_solo.py:321
- `_opposite_enemy_region` → rl/belief_planner.py:216, rl/prophet.py:119
- `_other_side_id` → rl/dashboard.py:337
- `_overtime_open` → rl/evaluate.py:35
- `_own_region` → rl/belief_planner.py:206, rl/prophet.py:111
- `_P` → rl/selftest.py:1665, rl/selftest.py:2665
- `_pair_seed_offset` → rl/run_league.py:393
- `_params` → rl/selftest.py:4807
- `_parse_replay_step` → rl/dashboard.py:210
- `_payload` → rl/workers.py:45
- `_pct` → rl/train_solo.py:695
- `_per_tower_norm_dmg` → rl/env_wrapper.py:131
- `_persist` → rl/train_solo.py:1410
- `_phase_weights` → rl/env_wrapper.py:164
- `_pick_suggested` → rl/prophet.py:140
- `_pick_suggested_card` → rl/belief_planner.py:221
- `_plan_batches` → rl/ppo.py:467
- `_plan_biases` → rl/follower.py:332
- `_plan_region_hit` → rl/evaluate.py:52
- `_play` → rl/bayes_filter.py:72
- `_play_one` → rl/flow_league.py:252
- `_play_side0` → rl/train_exploiter.py:27
- `_position_legal` → rl/action_mask.py:382
- `_prepare_env` → rl/run_league.py:380
- `_pressing_enemy` → rl/prophet.py:102
- `_princesses_alive` → rl/env_wrapper.py:126
- `_priors` → rl/mcts.py:411
- `_protect_backline` → rl/belief_planner.py:399, rl/prophet.py:206
- `_pull` → rl/belief_planner.py:449, rl/prophet.py:254
- `_punish` → rl/belief_planner.py:510, rl/prophet.py:279
- `_push_commit` → rl/belief_planner.py:534, rl/prophet.py:302
- `_rand_anchor_warns` → rl/train_solo.py:72
- `_random_action` → rl/opponents.py:181
- `_random_opponent` → rl/env_wrapper.py:524
- `_read_body` → rl/dashboard.py:2026
- `_ready_ability_cost` → rl/action_mask.py:479
- `_record_adoption` → rl/evaluate.py:111
- `_region_from_intent` → rl/belief_planner.py:188, rl/prophet.py:49
- `_reindex_hist` → rl/train_solo.py:403
- `_replay_n_games` → rl/dashboard.py:215
- `_resample_uniform` → rl/bayes_filter.py:89
- `_restore` → rl/run_league.py:600
- `_root_player` → rl/mcts.py:408
- `_round_estimates` → rl/run_league.py:448
- `_run_mp` → rl/run_league.py:942
- `_run_opponent` → rl/env_wrapper.py:490
- `_run_side0` → rl/run_league.py:268
- `_run_side0_scripted` → rl/run_league.py:313
- `_run_single` → rl/run_league.py:697
- `_run_vec` → rl/run_league.py:803
- `_sample_kind` → rl/train_solo.py:448
- `_sample_opponent_for` → rl/run_league.py:655
- `_save_ace` → rl/belief_planner.py:698, rl/prophet.py:410
- `_save_flow_progress` → rl/flow_league.py:157
- `_save_snapshot` → rl/run_league.py:627
- `_scan` → rl/train_solo.py:187
- `_send` → rl/dashboard.py:2038
- `_setup_wait` → rl/belief_planner.py:617, rl/prophet.py:348
- `_side` → rl/belief_planner.py:202, rl/prophet.py:107
- `_slot_mask_tensor` → rl/follower.py:316
- `_slot_playable` → rl/action_mask.py:46
- `_soft_control` → rl/belief_planner.py:350, rl/prophet.py:168
- `_spell_cast_value` → rl/belief_planner.py:253
- `_spell_covers_non_tower` → rl/action_mask.py:141
- `_spell_deals_damage` → rl/action_mask.py:89
- `_spell_finish` → rl/belief_planner.py:582, rl/prophet.py:325
- `_spell_has_enemy_target` → rl/action_mask.py:97
- `_spell_radius_m` → rl/action_mask.py:79
- `_spell_target_value` → rl/mcts.py:286
- `_spell_threat_in` → rl/prophet.py:128
- `_spell_tower_damage` → rl/action_mask.py:125
- `_spell_tower_ev_illegal` → rl/action_mask.py:158
- `_spell_trade` → rl/belief_planner.py:366, rl/prophet.py:183
- `_stall_probe` → rl/run_league.py:252
- `_stat_file_cards` → rl/dashboard.py:355
- `_sub_update` → rl/follower.py:394
- `_sub_vec` → rl/follower.py:387
- `_sweep_trend` → rl/flow_league.py:478
- `_sync_controls_once` → rl/train_solo.py:1240
- `_sync_frozen_copy` → rl/train_solo.py:170
- `_T` → rl/selftest.py:2678
- `_tanky_or_melee` → rl/belief_planner.py:141
- `_threat_and_my_pressure` → rl/prophet.py:78
- `_threat_unit_is_pressing` → rl/belief_planner.py:321
- `_tick_elixir` → rl/belief.py:288
- `_tiny_rollout_transitions` → rl/selftest.py:3746
- `_tower_premium_loss` → rl/mcts.py:68
- `_tower_snapshot` → rl/env_wrapper.py:594
- `_tower_state` → rl/flow_league.py:213
- `_uct_select` → rl/mcts.py:471
- `_unit_card` → rl/belief_planner.py:133
- `_unit_hp_map` → rl/env_wrapper.py:550
- `_units` → rl/prophet.py:63
- `_update_epochs` → rl/ppo.py:477
- `_value_from` → rl/follower.py:303
- `_write_gate_report` → rl/train_solo.py:309
- `_write_sweep` → rl/flow_league.py:491
- `ability` → rl/action_bundle.py:59
- `ability_legal` → rl/action_mask.py:500
- `ability_mana` → rl/action_mask.py:516
- `absr` → rl/selftest.py:938
- `act` → rl/follower.py:433, rl/human_play.py:124, rl/selftest.py:1782
- `act_parallel` → rl/follower.py:504
- `ActionBundle` → rl/action_bundle.py:76
- `ActionBundleSpace` → rl/env_wrapper.py:268
- `add` → rl/action_bundle.py:83
- `add_ability` → rl/action_bundle.py:87
- `add_agent` → rl/league.py:44
- `add_exploiter` → rl/league.py:125
- `anchor_point` → rl/train_solo.py:1454
- `ask` → rl/launcher_menu.py:60
- `ask_yesno` → rl/launcher_menu.py:81
- `battle_moved` → rl/selftest.py:3167
- `battle_snapshot` → rl/replay.py:85
- `behavioral_metrics` → rl/train_solo.py:550
- `belief` → rl/selftest.py:2319
- `belief_token_dim` → rl/belief.py:36
- `BeliefInference` → rl/belief.py:258
- `BeliefPlanner` → rl/belief_planner.py:341
- `BeliefState` → rl/belief.py:108
- `bkey` → rl/selftest.py:3014
- `brier_of` → rl/train_belief.py:103
- `build_card_pool` → rl/opponents.py:49
- `build_card_stats_payload` → rl/dashboard.py:430
- `build_cmd` → rl/launcher_menu.py:200
- `build_episodes` → rl/train_belief.py:90
- `build_feature` → rl/belief.py:232
- `build_five_agents` → rl/run_league.py:539
- `build_flow_models` → rl/flow_league.py:124
- `build_flow_pools` → rl/flow_league.py:67
- `build_payload` → rl/dashboard.py:78
- `build_replays_payload` → rl/dashboard.py:258
- `build_solo_payload` → rl/dashboard.py:182
- `build_sweep_payload` → rl/dashboard.py:143
- `card_name` → rl/action_bundle.py:62
- `cell_near` → rl/selftest.py:2089
- `check_policy_architecture` → rl/diagnostics.py:143
- `check_vitality` → rl/diagnostics.py:124
- `ckpt_path` → rl/config.py:280
- `classify_stats` → rl/decks.py:133
- `collect` → rl/train_bc.py:60
- `collect_params` → rl/launcher_menu.py:134
- `collect_replays` → rl/train_belief.py:49
- `compute_gae` → rl/ppo.py:168
- `compute_reward` → rl/env_wrapper.py:181
- `config_path` → rl/config.py:271
- `contains` → rl/env_wrapper.py:282
- `contains_card` → rl/action_bundle.py:109
- `CRFeatureExtractor` → rl/train_baseline.py:45
- `CycleBayesFilter` → rl/bayes_filter.py:42
- `deck` → rl/opponents.py:96, rl/opponents.py:148
- `decks_by_archetype` → rl/decks.py:125
- `delta_vs` → rl/evaluate.py:339
- `do_GET` → rl/dashboard.py:2048
- `do_POST` → rl/dashboard.py:2102
- `done` → rl/run_league.py:136
- `drive_games` → rl/human_play.py:259
- `ece_of` → rl/train_belief.py:117
- `Elo` → rl/elo.py:16
- `elo_table` → rl/league.py:141
- `encode` → rl/belief.py:204, rl/belief.py:344
- `end` → rl/replay.py:47
- `enemy_in_radius` → rl/selftest.py:1895
- `ensure` → rl/elo.py:22
- `ensure_dirs` → rl/config.py:286
- `entropy` → rl/bayes_filter.py:178
- `enumerate_bundles` → rl/mcts.py:143
- `env_total_loss_with` → rl/selftest.py:3609
- `EpisodeReplay` → rl/replay.py:22
- `eval_and_snapshot` → rl/run_league.py:714, rl/run_league.py:819, rl/run_league.py:963
- `eval_and_write` → rl/train_solo.py:1268
- `eval_control` → rl/train_solo.py:1246
- `eval_metrics` → rl/train_belief.py:177
- `eval_round_robin` → rl/run_league.py:476
- `eval_solo` → rl/train_solo.py:713
- `eval_solo_parallel` → rl/train_solo.py:941
- `evaluate` → rl/follower.py:713, rl/train_follower.py:242
- `evaluate_batch` → rl/follower.py:610
- `evaluate_belief` → rl/evaluate.py:477
- `evaluate_league` → rl/run_league.py:510
- `evaluate_winrate` → rl/train_exploiter.py:56
- `expected` → rl/elo.py:28
- `expert_bundle` → rl/train_bc.py:40
- `explained_variance` → rl/ppo.py:145
- `export_data` → rl/human_play.py:236
- `first_cell` → rl/selftest.py:1825
- `fit_temperature` → rl/train_belief.py:132
- `flow_pair_games` → rl/flow_league.py:90
- `folder` → rl/config.py:244
- `FollowerOpponent` → rl/train_follower.py:52
- `FollowerPolicy` → rl/follower.py:154
- `forward` → rl/train_baseline.py:62, rl/train_prophet.py:94
- `frame` → rl/selftest.py:1025, rl/selftest.py:1143, rl/selftest.py:3383
- `fresh` → rl/selftest.py:2736, rl/selftest.py:2788, rl/selftest.py:2879, rl/selftest.py:2990
- `from_dict` → rl/config.py:301, rl/ppo.py:77
- `from_old_layout` → rl/plan_space.py:163
- `from_single` → rl/action_bundle.py:102
- `gates_path` → rl/config.py:267
- `get_action_mask` → rl/env_wrapper.py:417
- `get_action_mask_for` → rl/env_wrapper.py:421
- `get_crown_count` → rl/selftest.py:2669
- `get_hidden_state` → rl/env_wrapper.py:393
- `get_masks_batch` → rl/run_league.py:1018
- `get_prophet_state` → rl/env_wrapper.py:396
- `gru_vitality` → rl/diagnostics.py:80
- `hand_probs` → rl/bayes_filter.py:151
- `Handler` → rl/dashboard.py:1995
- `heuristic_opponent` → rl/train_follower.py:33
- `hidden_labels` → rl/observation.py:147
- `hold_slots` → rl/plan_space.py:120
- `HumanPlaySession` → rl/human_play.py:63
- `Idle` → rl/selftest.py:1727
- `intent` → rl/plan_space.py:117
- `launch` → rl/launcher_menu.py:315
- `leaf_value` → rl/mcts.py:128
- `League` → rl/league.py:32
- `LeagueAgent` → rl/league.py:24
- `LeagueGameRecorder` → rl/run_league.py:108
- `legacy_action_to_bundle` → rl/env_wrapper.py:726
- `legal_cells` → rl/action_mask.py:441
- `load` → rl/belief.py:216, rl/config.py:311, rl/elo.py:53, rl/replay.py:57
- `load_bc_samples` → rl/human_play.py:194
- `load_checkpoint` → rl/follower.py:67
- `load_classified_decks` → rl/decks.py:104
- `load_league_replays` → rl/replay.py:136
- `load_policy` → rl/evaluate.py:473, rl/human_play.py:58
- `load_replay_payload` → rl/dashboard.py:270
- `load_state` → rl/dashboard.py:71, rl/league.py:162
- `locked` → rl/bayes_filter.py:57
- `log_message` → rl/dashboard.py:2099
- `main` → rl/dashboard.py:2270, rl/export_replay.py:23, rl/human_play.py:291, rl/launcher_menu.py:239, rl/run_league.py:1168, rl/selftest.py:4885, rl/train_baseline.py:80, rl/train_exploiter.py:89, rl/train_prophet.py:141
- `main_final_path` → rl/config.py:277
- `make` → rl/selftest.py:4668
- `make_demo_replays` → rl/dashboard.py:502
- `make_demo_solo` → rl/dashboard.py:2239
- `make_demo_state` → rl/dashboard.py:2141
- `make_demo_sweep` → rl/dashboard.py:2192
- `make_fake` → rl/selftest.py:1664
- `make_plan` → rl/train_follower.py:151
- `map_deck_cards` → rl/decks.py:74
- `mask_fn` → rl/run_league.py:1014, rl/train_follower.py:93
- `masks_for` → rl/follower.py:486
- `MCTSConfig` → rl/mcts.py:43
- `mini_mask_slot` → rl/selftest.py:2014
- `mk_battle` → rl/selftest.py:2078
- `mk_bs` → rl/selftest.py:3647
- `mk_decks` → rl/selftest.py:1405, rl/selftest.py:1489, rl/selftest.py:1529
- `model_reward_weights` → rl/config.py:404
- `NeuralBeliefEncoder` → rl/belief.py:167
- `new_battle` → rl/selftest.py:2233, rl/selftest.py:2413
- `new_buf` → rl/run_league.py:837, rl/run_league.py:1027
- `new_ep_buf` → rl/flow_league.py:218
- `next_probs` → rl/bayes_filter.py:167
- `next_spec` → rl/run_league.py:978
- `nll_of` → rl/train_belief.py:110
- `node_value` → rl/mcts.py:99
- `noise_prob` → rl/selftest.py:541
- `Noop` → rl/selftest.py:1780
- `noop` → rl/action_bundle.py:106
- `normalize` → rl/belief.py:118
- `normalize_card` → rl/decks.py:53
- `normalize_played` → rl/belief.py:82
- `observe` → rl/env_wrapper.py:390, rl/observation.py:87
- `observe_opponent_played` → rl/train_follower.py:74
- `opp_event_token` → rl/belief.py:63
- `opp_fn` → rl/selftest.py:2817
- `opt_path` → rl/config.py:283
- `overtime_open` → rl/overtime.py:30
- `PFSP` → rl/pfsp.py:29
- `pick` → rl/launcher_menu.py:92
- `pick_config` → rl/launcher_menu.py:120
- `place` → rl/selftest.py:2237, rl/selftest.py:2417
- `plan` → rl/belief_planner.py:772, rl/prophet.py:464
- `plan_of` → rl/selftest.py:3178
- `PlanToken` → rl/plan_space.py:99
- `play` → rl/opponents.py:105, rl/opponents.py:193, rl/selftest.py:1728
- `play_pair` → rl/run_league.py:406
- `PPOTrainer` → rl/ppo.py:86
- `presets` → rl/config.py:317
- `print_safe` → rl/diagnostics.py:37
- `probs` → rl/belief.py:160
- `prophet_policy_to_plan` → rl/train_prophet.py:112
- `ProphetEnv` → rl/train_prophet.py:46
- `ProphetExtractor` → rl/train_prophet.py:78
- `ProphetPlanner` → rl/prophet.py:163
- `pstate` → rl/selftest.py:2431
- `push_frame` → rl/belief.py:192
- `q` → rl/mcts.py:319
- `r` → rl/selftest.py:758, rl/selftest.py:835, rl/selftest.py:917
- `r_hit_blue` → rl/selftest.py:3554
- `r_hit_king` → rl/selftest.py:3576
- `r_hit_red` → rl/selftest.py:3545
- `rd` → rl/selftest.py:4734
- `record` → rl/run_league.py:128, rl/train_solo.py:479
- `record_elo_history` → rl/league.py:104
- `record_match` → rl/league.py:59
- `record_round_stats` → rl/league.py:111
- `record_step` → rl/replay.py:32
- `recv` → rl/run_league.py:999
- `ref_ema` → rl/selftest.py:475
- `refresh_hist` → rl/train_solo.py:414
- `refresh_snapshot` → rl/league.py:86
- `register_checkpoint` → rl/league.py:66
- `remove_agent` → rl/league.py:131
- `replays_dir` → rl/config.py:274
- `reset` → rl/bayes_filter.py:61, rl/belief.py:274, rl/env_wrapper.py:348, rl/train_baseline.py:34, rl/train_follower.py:79, rl/train_prophet.py:62
- `reset_history` → rl/belief.py:189
- `resolve` → rl/config.py:372
- `resolve_deck_set` → rl/train_solo.py:107
- `resolve_device` → rl/run_league.py:94
- `ReturnScaler` → rl/ppo.py:41
- `reward_to_env` → rl/config.py:399
- `RLEnv` → rl/env_wrapper.py:289
- `RLMCTS` → rl/mcts.py:330
- `run_ablation` → rl/evaluate.py:316
- `run_eval` → rl/evaluate.py:168
- `run_flow` → rl/flow_league.py:364
- `run_flow_sweep` → rl/flow_league.py:528
- `run_league` → rl/run_league.py:1143
- `run_solo` → rl/train_solo.py:1039
- `run_state_path` → rl/config.py:264
- `run_training` → rl/train_follower.py:117
- `sample` → rl/env_wrapper.py:275, rl/pfsp.py:66, rl/train_solo.py:436
- `sample_bundle` → rl/train_belief.py:33
- `sample_deck` → rl/opponents.py:71
- `sample_opponent` → rl/league.py:52
- `sample_training_opponent` → rl/run_league.py:711
- `save` → rl/config.py:294, rl/elo.py:47, rl/human_play.py:164, rl/replay.py:51
- `save_checkpoint` → rl/follower.py:55
- `save_flow_models` → rl/flow_league.py:147
- `save_league_replays` → rl/replay.py:128
- `save_state` → rl/league.py:146
- `scale` → rl/ppo.py:68
- `scale_pools` → rl/flow_league.py:100
- `scan_replays` → rl/dashboard.py:233
- `scan_sweep_dirs` → rl/dashboard.py:122
- `score` → rl/mcts.py:193
- `ScriptedPolicy` → rl/opponents.py:76
- `search` → rl/mcts.py:341
- `SelfDefenderPolicy` → rl/opponents.py:125
- `set_decks` → rl/run_league.py:124
- `set_hand` → rl/selftest.py:2247, rl/selftest.py:2427
- `settle_stall` → rl/run_league.py:240
- `settle_stall_from_counts` → rl/run_league.py:213
- `SingleCardAdapter` → rl/train_baseline.py:26
- `size` → rl/action_bundle.py:92
- `slot_mask` → rl/action_mask.py:59
- `snap` → rl/selftest.py:2998
- `solo_ckpt_path` → rl/config.py:256
- `solo_commit_blocked` → rl/action_mask.py:255
- `solo_env` → rl/train_solo.py:157
- `solo_main_path` → rl/config.py:253
- `solo_opt_path` → rl/config.py:260
- `solo_state_path` → rl/config.py:250
- `spawn` → rl/selftest.py:2083
- `spec_for` → rl/run_league.py:969
- `start` → rl/human_play.py:94, rl/replay.py:28
- `state` → rl/belief.py:318, rl/human_play.py:101
- `state_arg` → rl/launcher_menu.py:229
- `state_path` → rl/config.py:247
- `StatisticalBelief` → rl/belief.py:130
- `std` → rl/ppo.py:65
- `step` → rl/env_wrapper.py:598, rl/train_baseline.py:37, rl/train_prophet.py:69
- `sub_position` → rl/action_bundle.py:31
- `SubAction` → rl/action_bundle.py:43
- `table` → rl/elo.py:44
- `take_last_step` → rl/train_follower.py:112
- `test_ablation_recorded` → rl/selftest.py:1447
- `test_action_bundle_ability` → rl/selftest.py:66
- `test_action_bundle_same_tick` → rl/selftest.py:37
- `test_adv_inert_probe_and_const_baseline` → rl/selftest.py:4771
- `test_anchor_light_point_state` → rl/selftest.py:4695
- `test_battle_clone_fix` → rl/selftest.py:1203
- `test_bayes_filter` → rl/selftest.py:98
- `test_bayes_queue_lock` → rl/selftest.py:2536
- `test_behavioral_metrics` → rl/selftest.py:3370
- `test_belief_follower_ppo_league` → rl/selftest.py:1258
- `test_belief_multi_card_update` → rl/selftest.py:289
- `test_belief_survives_ability` → rl/selftest.py:267
- `test_bp_new_intent_rules` → rl/selftest.py:2223
- `test_bundle_cap_no_crash` → rl/selftest.py:328
- `test_classified_decks` → rl/selftest.py:598
- `test_config_reward_weights` → rl/selftest.py:640
- `test_crossed_river_defend_plan` → rl/selftest.py:3151
- `test_cuda_device_support` → rl/selftest.py:1223
- `test_dashboard_card_stats` → rl/selftest.py:1137
- `test_dashboard_replays` → rl/selftest.py:1019
- `test_death_damage_scaling` → rl/selftest.py:3215
- `test_deck_pool_factory` → rl/selftest.py:1099
- `test_draw_penalty_as_loss` → rl/selftest.py:1743
- `test_elo_eval_granularity` → rl/selftest.py:524
- `test_enc_layernorm_gru_vitality` → rl/selftest.py:4186
- `test_entropy_positive_and_sign` → rl/selftest.py:153
- `test_eval_solo_parallel` → rl/selftest.py:2616
- `test_eval_stall_early_stop` → rl/selftest.py:1717
- `test_exploiter_loads_main_checkpoint` → rl/selftest.py:245
- `test_flow_league_smoke` → rl/selftest.py:1394
- `test_flow_resume` → rl/selftest.py:1523
- `test_flow_sweep_smoke` → rl/selftest.py:1483
- `test_heuristic_opponent_actually_plays` → rl/selftest.py:223
- `test_hidden_replay_consistency` → rl/selftest.py:130
- `test_history_dedup_and_gates` → rl/selftest.py:3857
- `test_human_play_session` → rl/selftest.py:1599
- `test_league_elo_history` → rl/selftest.py:426
- `test_league_replays` → rl/selftest.py:983
- `test_league_resume` → rl/selftest.py:956
- `test_league_training_loop` → rl/selftest.py:623
- `test_log_rolling_direction` → rl/selftest.py:3301
- `test_mask_validate_invariant_both_sides` → rl/selftest.py:180
- `test_mcts_basic` → rl/selftest.py:2970
- `test_mcts_defense_and_wait` → rl/selftest.py:3068
- `test_mk_spawn_damage_and_iw_slow_fl` → rl/selftest.py:3428
- `test_model_reward_overrides` → rl/selftest.py:682
- `test_mp_training_loop` → rl/selftest.py:1372
- `test_no_solo_commit_without_lead` → rl/selftest.py:1993
- `test_opp_event_token` → rl/selftest.py:3098
- `test_opponent_pool_mix` → rl/selftest.py:3671
- `test_opponent_pool_mix_multi_dir` → rl/selftest.py:3905
- `test_opponent_pool_rand_anchor` → rl/selftest.py:3964
- `test_overtime_window` → rl/selftest.py:2653
- `test_parallel_batch_equivalence` → rl/selftest.py:1315
- `test_parallel_training_loop` → rl/selftest.py:1355
- `test_pfsp_gate_and_dynamic_hist` → rl/selftest.py:4086
- `test_plan_v1_layout` → rl/selftest.py:2150
- `test_play_pair_env_reuse` → rl/selftest.py:1691
- `test_pp_new_intent_rules` → rl/selftest.py:2401
- `test_ppo_multi_epoch_minibatch` → rl/selftest.py:4530
- `test_prophet_empty_board_not_defend` → rl/selftest.py:381
- `test_random_deck_model` → rl/selftest.py:398
- `test_register_checkpoint_isolated` → rl/selftest.py:304
- `test_replay_roundtrip` → rl/selftest.py:356
- `test_reward_economy_elixir_diff` → rl/selftest.py:782
- `test_reward_economy_level_invariance` → rl/selftest.py:745
- `test_reward_economy_preset` → rl/selftest.py:712
- `test_reward_economy_trade_pricing` → rl/selftest.py:820
- `test_reward_tower_premium` → rl/selftest.py:3522
- `test_reward_tower_premium_rlenv_flow` → rl/selftest.py:3595
- `test_reward_v2_ledger` → rl/selftest.py:1806
- `test_rlenv_card_level` → rl/selftest.py:872
- `test_simulate_exchange` → rl/selftest.py:2776
- `test_solo_mode_smoke` → rl/selftest.py:1559
- `test_solo_rand_anchor` → rl/selftest.py:4648
- `test_solo_resume` → rl/selftest.py:1632
- `test_spell_empty_value_gate` → rl/selftest.py:1875
- `test_spell_module` → rl/selftest.py:2866
- `test_spell_tower_ev_gate` → rl/selftest.py:1934
- `test_stall_probe` → rl/selftest.py:1660
- `test_stall_settlement_margin` → rl/selftest.py:4404
- `test_tank_backline_geometry` → rl/selftest.py:2064
- `test_tower_threat_calc` → rl/selftest.py:2724
- `test_tower_troop_hp_reference` → rl/selftest.py:897
- `test_tower_value_mult` → rl/selftest.py:3490
- `test_value_bypass` → rl/selftest.py:4319
- `test_value_channel_norm_and_gnorm_split` → rl/selftest.py:3766
- `test_value_independent_encoder` → rl/selftest.py:4437
- `test_vines_snare_fl_duration` → rl/selftest.py:3270
- `test_winrate_streams_independent` → rl/selftest.py:452
- `timeout_winner` → rl/run_league.py:179
- `to_belief_dataset` → rl/replay.py:69
- `to_device` → rl/follower.py:263
- `to_dict` → rl/config.py:291, rl/ppo.py:73
- `to_position` → rl/action_bundle.py:67
- `to_tuple` → rl/action_bundle.py:70, rl/action_bundle.py:95
- `to_vector` → rl/plan_space.py:124
- `tower_premium_k` → rl/env_wrapper.py:121
- `tower_total_hp` → rl/env_wrapper.py:74
- `tower_value_mult` → rl/env_wrapper.py:101
- `towers_hp` → rl/run_league.py:149
- `trace` → rl/selftest.py:3315
- `train` → rl/train_belief.py:145
- `train_bc` → rl/train_bc.py:86
- `train_bc_from_human` → rl/human_play.py:205
- `TrainConfig` → rl/config.py:98
- `troop` → rl/selftest.py:3392
- `update` → rl/bayes_filter.py:116, rl/belief.py:140, rl/belief.py:300, rl/elo.py:31, rl/ppo.py:54, rl/ppo.py:194
- `update_winrate` → rl/pfsp.py:46
- `validate_bundle` → rl/action_mask.py:521
- `validate_slots` → rl/mcts.py:201
- `value` → rl/follower.py:704
- `var` → rl/ppo.py:62
- `weights` → rl/pfsp.py:53
- `worker_main` → rl/workers.py:71
- `write_solo_state` → rl/train_solo.py:513
- `zeros` → rl/plan_space.py:182

#### B.8.1 ~ B.8.38 逐文件符号表

##### B.8.1 `rl/__init__.py`（0 个符号）

（本文件无类/函数定义。）

##### B.8.2 `rl/action_bundle.py`（15 个符号）

| 序号 | 函数/方法 | 文件:行号 | 一句话作用 | 关键参数（名字=默认值） | 返回 |
|---|---|---|---|---|--- |
| 1 | `sub_position` | `rl/action_bundle.py:31` | 本地网格坐标→世界坐标唯一换算（P1 镜像） | `player_id: int, x: int, y: int` | `Position` |
| 2 | `SubAction` | `rl/action_bundle.py:43` | 单卡子动作：kind/slot/本地 x,y | `—` | `—` |
| 3 | `SubAction.ability`<br>（内嵌于 `SubAction`） | `rl/action_bundle.py:59` | 构造英雄技能子动作（kind=ability） | `cls` | `'SubAction'` |
| 4 | `SubAction.card_name`<br>（内嵌于 `SubAction`） | `rl/action_bundle.py:62` | slot→卡名，非法槽位返回 None | `self, player` | `Optional[str]` |
| 5 | `SubAction.to_position`<br>（内嵌于 `SubAction`） | `rl/action_bundle.py:67` | 本地坐标经 sub_position 转世界坐标 | `self, player_id: int=0` | `Position` |
| 6 | `SubAction.to_tuple`<br>（内嵌于 `SubAction`） | `rl/action_bundle.py:70` | 旧接口兼容：转 (slot,y,x) | `self` | `Tuple[int, int, int]` |
| 7 | `ActionBundle` | `rl/action_bundle.py:76` | 同刻多卡动作包（子动作列表，≤K_MAX） | `—` | `—` |
| 8 | `ActionBundle.__post_init__`<br>（内嵌于 `ActionBundle`） | `rl/action_bundle.py:79` | 校验子动作数不超过 K_MAX，越界抛错 | `self` | `—` |
| 9 | `ActionBundle.add`<br>（内嵌于 `ActionBundle`） | `rl/action_bundle.py:83` | 追加 deploy 子动作并返回 self | `self, slot: int, x: int, y: int` | `'ActionBundle'` |
| 10 | `ActionBundle.add_ability`<br>（内嵌于 `ActionBundle`） | `rl/action_bundle.py:87` | 追加 ability 子动作并返回 self | `self` | `'ActionBundle'` |
| 11 | `ActionBundle.size`<br>（内嵌于 `ActionBundle`） | `rl/action_bundle.py:92` | 子动作数量（property） | `self` | `int` |
| 12 | `ActionBundle.to_tuple`<br>（内嵌于 `ActionBundle`） | `rl/action_bundle.py:95` | n≤1 兼容模式转旧 (slot,y,x) | `self` | `Tuple[int, int, int]` |
| 13 | `ActionBundle.from_single`<br>（内嵌于 `ActionBundle`） | `rl/action_bundle.py:102` | 单卡构造 ActionBundle | `cls, slot: int, x: int, y: int` | `'ActionBundle'` |
| 14 | `ActionBundle.noop`<br>（内嵌于 `ActionBundle`） | `rl/action_bundle.py:106` | 空 bundle：本帧等待（合法且可能最优） | `cls` | `'ActionBundle'` |
| 15 | `ActionBundle.contains_card`<br>（内嵌于 `ActionBundle`） | `rl/action_bundle.py:109` | 是否含指定卡名的子动作 | `self, player, card_name: str` | `bool` |

##### B.8.3 `rl/action_mask.py`（24 个符号）

| 序号 | 函数/方法 | 文件:行号 | 一句话作用 | 关键参数（名字=默认值） | 返回 |
|---|---|---|---|---|--- |
| 1 | `_card_cost` | `rl/action_mask.py:30` | 实际出牌费用（Mirror 按引擎语义 = 上一张牌费用 + 1；无上一张牌 → None）。 | `player, card_name: str` | `Optional[float]` |
| 2 | `_effective_card` | `rl/action_mask.py:39` | 引擎实际部署/校验的卡名（Mirror 重放上一张牌）。 | `player, card_name: str` | `str` |
| 3 | `_slot_playable` | `rl/action_mask.py:46` | 单槽可出判定：王塔存活+在手牌前 4+费用足够 | `player, card_name: str, elixir: float` | `bool` |
| 4 | `slot_mask` | `rl/action_mask.py:59` | 返回 (K_MAX,) bool 掩码：哪些手牌槽在当前圣水下可出（且未在 bundle 中使用）。 | `player, elixir_override: float=None, used_slots=None` | `np.ndarray` |
| 5 | `_spell_radius_m` | `rl/action_mask.py:79` | 法术溅射半径（世界单位）；无半径数据返回 0.0（=不做闸门，保持旧语义）。 | `card_name: str, card_info: 'Card'=None` | `float` |
| 6 | `_spell_deals_damage` | `rl/action_mask.py:89` | 是否输出伤害的法术。伤害型法术受空砸闸门约束；增益/位移/召唤类法术放行。 | `card_name: str, card_info: 'Card'=None` | `bool` |
| 7 | `_spell_has_enemy_target` | `rl/action_mask.py:97` | 溅射半径内是否有存活敌方目标（塔/建筑/部队）。命中口径与引擎溅射一致： | `battle, player_id: int, pos: Position, radius: float` | `bool` |
| 8 | `_spell_tower_damage` | `rl/action_mask.py:125` | 对塔伤害（引擎标定缓存）。标定失败（环境异常）返回 0.0 = 闸门自动放行。 | `card_name: str` | `float` |
| 9 | `_spell_covers_non_tower` | `rl/action_mask.py:141` | 溅射半径内是否有对手的**非塔**目标（部队/建筑）。有 → 法术有正事可干，放行。 | `battle, player_id: int, pos: Position, radius: float` | `bool` |
| 10 | `_spell_tower_ev_illegal` | `rl/action_mask.py:158` | 前段"纯砸塔"落点非法判定（空砸闸门的对塔特化加强版）。 | `battle, player_id: int, card_name: str, pos: Position, card_info: 'Card'=None` | `bool` |
| 11 | `_opp_min_hand_cost` | `rl/action_mask.py:231` | 对手手牌最低可出费用；手牌为空/国王已倒 → None（=对手出不了手）。 | `battle, player_id: int` | `Optional[float]` |
| 12 | `_enemy_in_my_half` | `rl/action_mask.py:241` | 对方是否有单位已进入我半场（压境 → 防守优先，裸下闸门放行）。 | `battle, player_id: int` | `bool` |
| 13 | `solo_commit_blocked` | `rl/action_mask.py:255` | 高承诺单卡裸下是否被禁止： | `battle, player_id: int, card_name: str, own_elixir: float` | `bool` |
| 14 | `_active_push_tanks` | `rl/action_mask.py:292` | 本方可作推进前排的存活坦克（Giant/Knight），且已离开国王塔进入推进段。 | `battle, player_id: int` | `—` |
| 15 | `_card_cached` | `rl/action_mask.py:317` | 带缓存的 Card 构造（按卡名+等级） | `card_name: str` | `'Card'` |
| 16 | `_backline_min_gap_m` | `rl/action_mask.py:326` | 后排与坦克的最小纵向间距（世界单位）： | `back_card: str, tank_card: str, back_info: 'Card'=None` | `float` |
| 17 | `_backline_placement_illegal` | `rl/action_mask.py:338` | 后排落点几何闸门：同路有推进坦克时，落点必须位于某坦克之后且留足 | `battle, player_id: int, card_name: str, pos: Position, card_info: 'Card'=None` | `bool` |
| 18 | `_hits_dead_enemy_tower` | `rl/action_mask.py:367` | 法术落点是否贴着已毁敌方塔本体（塔实体 id≤6 且 is_alive=False 永留场）。 | `battle, player_id: int, pos: Position` | `bool` |
| 19 | `_position_legal` | `rl/action_mask.py:382` | 复刻 battle.deploy_card 中的部署区域合法性（法术额外挡已毁塔本体 + 空砸闸门）。 | `battle, player_id: int, card_name: str, pos: Position, card_info: 'Card'=None` | `bool` |
| 20 | `legal_cells` | `rl/action_mask.py:441` | 返回 (32,18) bool：该卡在玩家本地网格中可以部署的格子。 | `battle, player_id: int, card_name: str` | `np.ndarray` |
| 21 | `_ready_ability_cost` | `rl/action_mask.py:479` | 返回场上首个就绪英雄技能的耗蓝；无就绪英雄返回 None。 | `battle, player_id: int` | `Optional[float]` |
| 22 | `ability_legal` | `rl/action_mask.py:500` | bundle 中是否还能触发英雄技能。 | `battle, player_id: int, elixir_override: float=None, already_used: bool=False` | `bool` |
| 23 | `ability_mana` | `rl/action_mask.py:516` | 返回就绪英雄技能的耗蓝；无就绪英雄返回 None（不再用 0 作哨兵，P2）。 | `battle, player_id: int` | `Optional[float]` |
| 24 | `validate_bundle` | `rl/action_mask.py:521` | 整包校验（不修改任何状态）：任一子动作非法则拒绝整包。 | `battle, player_id: int, bundle: ActionBundle` | `—` |

##### B.8.4 `rl/bayes_filter.py`（13 个符号）

| 序号 | 函数/方法 | 文件:行号 | 一句话作用 | 关键参数（名字=默认值） | 返回 |
|---|---|---|---|---|--- |
| 1 | `CycleBayesFilter` | `rl/bayes_filter.py:42` | 8 卡循环队列信念：O(1) 锁定流+粒子相 | `—` | `—` |
| 2 | `CycleBayesFilter.__init__`<br>（内嵌于 `CycleBayesFilter`） | `rl/bayes_filter.py:45` | 存卡组/粒子数/RNG，均匀先验重采样 | `self, deck, n_particles: int=128, seed: int=0` | `—` |
| 3 | `CycleBayesFilter.locked`<br>（内嵌于 `CycleBayesFilter`） | `rl/bayes_filter.py:57` | 是否处于精确锁定流（property） | `self` | `bool` |
| 4 | `CycleBayesFilter.reset`<br>（内嵌于 `CycleBayesFilter`） | `rl/bayes_filter.py:61` | 清观测与锁定态并重采样粒子 | `self, deck=None` | `—` |
| 5 | `CycleBayesFilter._play`<br>（内嵌于 `CycleBayesFilter`） | `rl/bayes_filter.py:72` | 打出手牌卡后移队尾（与引擎一致） | `perm, card` | `—` |
| 6 | `CycleBayesFilter._lock_from_tail`<br>（内嵌于 `CycleBayesFilter`） | `rl/bayes_filter.py:76` | 用最近 4 张出牌重建规范 cycle | `self, last4` | `—` |
| 7 | `CycleBayesFilter._resample_uniform`<br>（内嵌于 `CycleBayesFilter`） | `rl/bayes_filter.py:89` | 均匀先验去重重采样粒子 | `self` | `—` |
| 8 | `CycleBayesFilter._degrade`<br>（内嵌于 `CycleBayesFilter`） | `rl/bayes_filter.py:102` | 异常/分歧退回粒子相并重采样 | `self` | `—` |
| 9 | `CycleBayesFilter._consistent`<br>（内嵌于 `CycleBayesFilter`） | `rl/bayes_filter.py:108` | 粒子相一致性筛选并推进到打出后状态 | `self, card` | `—` |
| 10 | `CycleBayesFilter.update`<br>（内嵌于 `CycleBayesFilter`） | `rl/bayes_filter.py:116` | 观测出牌→O(1) 锁定推进或粒子筛选/重锁 | `self, played_card` | `—` |
| 11 | `CycleBayesFilter.hand_probs`<br>（内嵌于 `CycleBayesFilter`） | `rl/bayes_filter.py:151` | 每张卡在手牌的概率（锁定流为 0/1） | `self` | `np.ndarray` |
| 12 | `CycleBayesFilter.next_probs`<br>（内嵌于 `CycleBayesFilter`） | `rl/bayes_filter.py:167` | 每张卡是下一张牌的概率（cycle[4]） | `self` | `np.ndarray` |
| 13 | `CycleBayesFilter.entropy`<br>（内嵌于 `CycleBayesFilter`） | `rl/bayes_filter.py:178` | 手牌后验熵（锁定流恒 0） | `self` | `float` |

##### B.8.5 `rl/belief.py`（26 个符号）

| 序号 | 函数/方法 | 文件:行号 | 一句话作用 | 关键参数（名字=默认值） | 返回 |
|---|---|---|---|---|--- |
| 1 | `belief_token_dim` | `rl/belief.py:36` | 无神经编码时 belief_token 维度公式 | `deck` | `int` |
| 2 | `_event_row` | `rl/belief.py:52` | 单条对手出牌事件→定长行向量 | `card, x, y, dt` | `np.ndarray` |
| 3 | `opp_event_token` | `rl/belief.py:63` | 最近 k 条事件展平（含 Δt 陈旧度），前补零 | `history, now=None, k: int=OPP_EVENT_K` | `np.ndarray` |
| 4 | `normalize_played` | `rl/belief.py:82` | opp_played 规范化为 [(card,x,y)]，过滤哨兵 | `opp_played` | `—` |
| 5 | `BeliefState` | `rl/belief.py:108` | 信念状态：手牌/下一张/圣水/意图/倾向 | `—` | `—` |
| 6 | `BeliefState.normalize`<br>（内嵌于 `BeliefState`） | `rl/belief.py:118` | 缺失字段补均匀分布并返回 self | `self` | `—` |
| 7 | `StatisticalBelief` | `rl/belief.py:130` | 对手风格/路线倾向的计数统计信念 | `—` | `—` |
| 8 | `StatisticalBelief.__init__`<br>（内嵌于 `StatisticalBelief`） | `rl/belief.py:133` | 初始化倾向/路线计数与总计 | `self` | `—` |
| 9 | `StatisticalBelief.update`<br>（内嵌于 `StatisticalBelief`） | `rl/belief.py:140` | 按卡类型与落点半场累计倾向计数 | `self, card: str, x=None, y=None` | `—` |
| 10 | `StatisticalBelief.probs`<br>（内嵌于 `StatisticalBelief`） | `rl/belief.py:160` | 拉普拉斯平滑后的倾向概率 | `self` | `np.ndarray` |
| 11 | `NeuralBeliefEncoder` | `rl/belief.py:167` | GRU 压缩历史观测→belief_token+两个预测头 | `—` | `—` |
| 12 | `NeuralBeliefEncoder.__init__`<br>（内嵌于 `NeuralBeliefEncoder`） | `rl/belief.py:173` | 建 GRU/next_head/hand_head/belief_proj 与队列 | `self, in_dim: int, hidden: int=64, num_classes: int=NUM_CARDS, hand_dim: int=8, max_len: int=32` | `—` |
| 13 | `NeuralBeliefEncoder.reset_history`<br>（内嵌于 `NeuralBeliefEncoder`） | `rl/belief.py:189` | 清空历史帧队列 | `self` | `—` |
| 14 | `NeuralBeliefEncoder.push_frame`<br>（内嵌于 `NeuralBeliefEncoder`） | `rl/belief.py:192` | 压入一帧特征到历史队列 | `self, feat: np.ndarray` | `—` |
| 15 | `NeuralBeliefEncoder._history_tensor`<br>（内嵌于 `NeuralBeliefEncoder`） | `rl/belief.py:195` | 历史队列→(1,T,D) 张量（空则零帧） | `self` | `—` |
| 16 | `NeuralBeliefEncoder.encode`<br>（内嵌于 `NeuralBeliefEncoder`） | `rl/belief.py:204` | 前向 GRU 取末隐状态投影为 token | `self, feat: np.ndarray` | `np.ndarray` |
| 17 | `NeuralBeliefEncoder.load`<br>（内嵌于 `NeuralBeliefEncoder`） | `rl/belief.py:216` | 从 train_belief 的 ckpt 重建编码器 | `cls, path, in_dim=None` | `—` |
| 18 | `build_feature` | `rl/belief.py:232` | 把一帧观测压成神经编码器输入向量 | `obs: dict, opp_played` | `np.ndarray` |
| 19 | `BeliefInference` | `rl/belief.py:258` | 组合信念模块：规则+统计+神经 | `—` | `—` |
| 20 | `BeliefInference.__init__`<br>（内嵌于 `BeliefInference`） | `rl/belief.py:261` | 建 CycleBayesFilter/StatisticalBelief/事件史 | `self, opp_deck, use_rule=True, use_stat=True, neural=None, n_particles=128, seed=0` | `—` |
| 21 | `BeliefInference.reset`<br>（内嵌于 `BeliefInference`） | `rl/belief.py:274` | 重置全部子信念、圣水估计与事件史 | `self, opp_deck=None` | `—` |
| 22 | `BeliefInference._tick_elixir`<br>（内嵌于 `BeliefInference`） | `rl/belief.py:288` | 按 obs.time 推进圣水估计（2.8s/点） | `self, obs` | `—` |
| 23 | `BeliefInference.update`<br>（内嵌于 `BeliefInference`） | `rl/belief.py:300` | 用本 tick 出牌更新规则/统计/圣水/事件史 | `self, obs, opp_played, opp_x=None, opp_card_type=None, elixir_est=None` | `—` |
| 24 | `BeliefInference.state`<br>（内嵌于 `BeliefInference`） | `rl/belief.py:318` | 汇总为 BeliefState（含不确定性归一） | `self` | `BeliefState` |
| 25 | `BeliefInference._now`<br>（内嵌于 `BeliefInference`） | `rl/belief.py:330` | 当前决策时刻（obs.time 优先，缓存回退） | `self, obs` | `—` |
| 26 | `BeliefInference.encode`<br>（内嵌于 `BeliefInference`） | `rl/belief.py:344` | 拼 belief_token：规则+统计+事件(+神经) | `self, obs=None, opp_played=None` | `np.ndarray` |

##### B.8.6 `rl/belief_planner.py`（35 个符号）

| 序号 | 函数/方法 | 文件:行号 | 一句话作用 | 关键参数（名字=默认值） | 返回 |
|---|---|---|---|---|--- |
| 1 | `_is_tower` | `rl/belief_planner.py:118` | 卡名是否含 Tower | `name: str` | `bool` |
| 2 | `_deployable_entity` | `rl/belief_planner.py:122` | 是否可部署单位/建筑（排除塔与弹道） | `e` | `bool` |
| 3 | `_unit_card` | `rl/belief_planner.py:133` | 卡名→Card（KeyError 返回 None） | `name` | `—` |
| 4 | `_tanky_or_melee` | `rl/belief_planner.py:141` | 拉扯对象口径：只打建筑/高血/近战 | `c` | `bool` |
| 5 | `_is_front_tank_name` | `rl/belief_planner.py:151` | 前排主体判定（沉底名单/高血近战） | `name` | `bool` |
| 6 | `_enemy_pressure` | `rl/belief_planner.py:159` | 敌我双方半场推进压力（排除静态塔） | `battle` | `—` |
| 7 | `_enemy_main_x` | `rl/belief_planner.py:176` | 敌方非塔单位重心 x（空则 9.0） | `battle` | `—` |
| 8 | `_my_main_x` | `rl/belief_planner.py:182` | 我方非塔单位重心 x（空则 9.0） | `battle` | `—` |
| 9 | `_region_from_intent` | `rl/belief_planner.py:188` | 旧 8 意图→focus_region 映射 | `intent: str` | `str` |
| 10 | `_side` | `rl/belief_planner.py:202` | x<LANE_SPLIT_X 为 left，否则 right | `x: float` | `str` |
| 11 | `_own_region` | `rl/belief_planner.py:206` | x→own_left/own_right | `x: float` | `str` |
| 12 | `_enemy_region` | `rl/belief_planner.py:211` | x→enemy_left/enemy_right | `x: float` | `str` |
| 13 | `_opposite_enemy_region` | `rl/belief_planner.py:216` | 敌方重心在 x→建议进攻另一路 | `x: float` | `str` |
| 14 | `_pick_suggested_card` | `rl/belief_planner.py:221` | 按 intent 从可出手牌启发式选一张 | `battle, player_id, belief: Optional[BeliefState], intent: str` | `—` |
| 15 | `_hand_slot` | `rl/belief_planner.py:249` | 卡在 p.cycle[:4] 的槽位 1..4 | `p, card_name` | `—` |
| 16 | `_spell_cast_value` | `rl/belief_planner.py:253` | 法术落点估值（引擎标定+残血塔差异化加权） | `battle, player_id, card_name` | `—` |
| 17 | `_closest_threat` | `rl/belief_planner.py:309` | 最接近我方塔的敌方可部署单位 | `battle` | `—` |
| 18 | `_threat_unit_is_pressing` | `rl/belief_planner.py:321` | 威胁单位是否已进入我方半场/桥头 | `e` | `bool` |
| 19 | `_opp_spell_threat_of` | `rl/belief_planner.py:326` | 由手牌后验概率阈值推强法术威胁 | `belief: Optional[BeliefState]` | `—` |
| 20 | `BeliefPlanner` | `rl/belief_planner.py:341` | 基于信念状态的规划器（12 新意图+旧回退） | `—` | `—` |
| 21 | `BeliefPlanner.__init__`<br>（内嵌于 `BeliefPlanner`） | `rl/belief_planner.py:344` | 存后验采样开关与样本数 | `self, use_posterior_sampling: bool=False, n_samples: int=8` | `—` |
| 22 | `BeliefPlanner._soft_control`<br>（内嵌于 `BeliefPlanner`） | `rl/belief_planner.py:350` | 软控：压境威胁→冰冻/藤蔓/龙卷 | `self, battle, p, threat, belief` | `—` |
| 23 | `BeliefPlanner._spell_trade`<br>（内嵌于 `BeliefPlanner`） | `rl/belief_planner.py:366` | 解牌：远程脆皮进半场→伤害法术 | `self, battle, p, threat, belief` | `—` |
| 24 | `BeliefPlanner._protect_backline`<br>（内嵌于 `BeliefPlanner`） | `rl/belief_planner.py:399` | 保后排：反应+信念预判（概率>0.55） | `self, battle, p, threat, belief` | `—` |
| 25 | `BeliefPlanner._pull`<br>（内嵌于 `BeliefPlanner`） | `rl/belief_planner.py:449` | 拉扯/拦路（7g 高血/近战/建筑口径） | `self, battle, p, threat, belief` | `—` |
| 26 | `BeliefPlanner._punish`<br>（内嵌于 `BeliefPlanner`） | `rl/belief_planner.py:510` | 读 belief 圣水低→压敌方重心反侧 | `self, battle, p, threat, belief` | `—` |
| 27 | `BeliefPlanner._push_commit`<br>（内嵌于 `BeliefPlanner`） | `rl/belief_planner.py:534` | 前排推进中→优先真后排跟输出 | `self, battle, p, threat, belief` | `—` |
| 28 | `BeliefPlanner._push_commit._ok`<br>（内嵌于 `BeliefPlanner._push_commit`） | `rl/belief_planner.py:555` | 跟牌候选判定闭包（3-6 费非坦克角色） | `card` | `—` |
| 29 | `BeliefPlanner._spell_finish`<br>（内嵌于 `BeliefPlanner`） | `rl/belief_planner.py:582` | 后期低血塔→磨塔法术（双倍期经济账） | `self, battle, p, threat, belief` | `—` |
| 30 | `BeliefPlanner._setup_wait`<br>（内嵌于 `BeliefPlanner`） | `rl/belief_planner.py:617` | 攒费窗口：hold_mask=1111 等满费沉底 | `self, battle, p, threat, belief` | `—` |
| 31 | `BeliefPlanner._king_activate`<br>（内嵌于 `BeliefPlanner`） | `rl/belief_planner.py:668` | 公主塔残血+重单位→激活国王塔 | `self, battle, p, threat, belief` | `—` |
| 32 | `BeliefPlanner._save_ace`<br>（内嵌于 `BeliefPlanner`） | `rl/belief_planner.py:698` | 藏 ace：hold_mask 指名+建议普通牌 | `self, battle, p, threat, belief` | `—` |
| 33 | `BeliefPlanner._anti_spell`<br>（内嵌于 `BeliefPlanner`） | `rl/belief_planner.py:734` | 信念高概率有强法术→防溅射站位 | `self, battle, p, threat, belief` | `—` |
| 34 | `BeliefPlanner._cycle_small`<br>（内嵌于 `BeliefPlanner`） | `rl/belief_planner.py:753` | 无压力+1-2 费小牌+费足→过牌 | `self, battle, p, threat, belief` | `—` |
| 35 | `BeliefPlanner.plan`<br>（内嵌于 `BeliefPlanner`） | `rl/belief_planner.py:772` | 主入口：优先链+旧 8 意图回退+拦截几何 hint | `self, battle, belief: BeliefState, obs=None` | `PlanToken` |

##### B.8.7 `rl/config.py`（23 个符号）

| 序号 | 函数/方法 | 文件:行号 | 一句话作用 | 关键参数（名字=默认值） | 返回 |
|---|---|---|---|---|--- |
| 1 | `TrainConfig` | `rl/config.py:98` | 训练配置数据类：超参、奖励权重、架构开关与路径方法 | `—` | `—` |
| 2 | `TrainConfig.folder`<br>（内嵌于 `TrainConfig`） | `rl/config.py:244` | 返回 out_dir/<name> 输出目录 | `self` | `—` |
| 3 | `TrainConfig.state_path`<br>（内嵌于 `TrainConfig`） | `rl/config.py:247` | 返回联赛状态 league_state.json 路径 | `self` | `—` |
| 4 | `TrainConfig.solo_state_path`<br>（内嵌于 `TrainConfig`） | `rl/config.py:250` | 返回 solo 状态 solo_state.json 路径 | `self` | `—` |
| 5 | `TrainConfig.solo_main_path`<br>（内嵌于 `TrainConfig`） | `rl/config.py:253` | 返回 solo main 权重 solo_main.pt 路径 | `self` | `—` |
| 6 | `TrainConfig.solo_ckpt_path`<br>（内嵌于 `TrainConfig`） | `rl/config.py:256` | 返回带步号的 solo 历史检查点路径 | `self, step` | `—` |
| 7 | `TrainConfig.solo_opt_path`<br>（内嵌于 `TrainConfig`） | `rl/config.py:260` | 返回 solo 优化器状态 solo_opt.pt 路径 | `self` | `—` |
| 8 | `TrainConfig.run_state_path`<br>（内嵌于 `TrainConfig`） | `rl/config.py:264` | 返回 run_state.json 路径 | `self` | `—` |
| 9 | `TrainConfig.gates_path`<br>（内嵌于 `TrainConfig`） | `rl/config.py:267` | 返回行为门禁报告 gates.json 路径 | `self` | `—` |
| 10 | `TrainConfig.config_path`<br>（内嵌于 `TrainConfig`） | `rl/config.py:271` | 返回配置存档 config.json 路径 | `self` | `—` |
| 11 | `TrainConfig.replays_dir`<br>（内嵌于 `TrainConfig`） | `rl/config.py:274` | 返回录像目录 replays 路径 | `self` | `—` |
| 12 | `TrainConfig.main_final_path`<br>（内嵌于 `TrainConfig`） | `rl/config.py:277` | 返回训练结束的 main_final.pt 路径 | `self` | `—` |
| 13 | `TrainConfig.ckpt_path`<br>（内嵌于 `TrainConfig`） | `rl/config.py:280` | 返回带步号的 main 检查点路径 | `self, step` | `—` |
| 14 | `TrainConfig.opt_path`<br>（内嵌于 `TrainConfig`） | `rl/config.py:283` | 返回带步号的 main 优化器状态路径 | `self, step` | `—` |
| 15 | `TrainConfig.ensure_dirs`<br>（内嵌于 `TrainConfig`） | `rl/config.py:286` | 创建输出目录与录像目录 | `self` | `—` |
| 16 | `TrainConfig.to_dict`<br>（内嵌于 `TrainConfig`） | `rl/config.py:291` | 用 asdict 把配置转成字典 | `self` | `—` |
| 17 | `TrainConfig.save`<br>（内嵌于 `TrainConfig`） | `rl/config.py:294` | 把配置序列化为 JSON 并返回路径 | `self, path=None` | `—` |
| 18 | `TrainConfig.from_dict`<br>（内嵌于 `TrainConfig`） | `rl/config.py:301` | 由字典构造配置，滤未知键并按默认补奖励权重 | `cls, d` | `—` |
| 19 | `TrainConfig.load`<br>（内嵌于 `TrainConfig`） | `rl/config.py:311` | 从 JSON 文件读取配置 | `cls, path` | `—` |
| 20 | `TrainConfig.presets`<br>（内嵌于 `TrainConfig`） | `rl/config.py:317` | 返回七个命名预设（各含奖励权重与超参） | `cls` | `—` |
| 21 | `TrainConfig.resolve`<br>（内嵌于 `TrainConfig`） | `rl/config.py:372` | 按预设名或 JSON 解析配置并应用命令行覆盖（深拷贝） | `cls, preset, load_config=None, **overrides` | `—` |
| 22 | `reward_to_env` | `rl/config.py:399` | 把配置奖励权重合成 RLEnv 用的 reward_weights 字典 | `cfg: TrainConfig` | `dict` |
| 23 | `model_reward_weights` | `rl/config.py:404` | 按流派模型 id 在所选预设之上叠加奖惩覆盖 | `model_id: str, cfg: TrainConfig` | `dict` |

##### B.8.8 `rl/dashboard.py`（28 个符号）

| 序号 | 函数/方法 | 文件:行号 | 一句话作用 | 关键参数（名字=默认值） | 返回 |
|---|---|---|---|---|--- |
| 1 | `load_state` | `rl/dashboard.py:71` | 读取 JSON 状态文件，不存在返回 None | `path` | `—` |
| 2 | `build_payload` | `rl/dashboard.py:78` | 组装联赛面板数据（agents/elo_history/round_stats） | `path` | `—` |
| 3 | `scan_sweep_dirs` | `rl/dashboard.py:122` | 扫描 sweep 根目录得到策略目录列表 | `root` | `—` |
| 4 | `build_sweep_payload` | `rl/dashboard.py:143` | 读取各 flow_sweep summary.json 组装进度与曲线数据 | `sweep_root` | `—` |
| 5 | `build_solo_payload` | `rl/dashboard.py:182` | 读取 solo_state.json 组装胜率曲线与进度数据 | `path` | `—` |
| 6 | `_parse_replay_step` | `rl/dashboard.py:210` | 从 league_<step>.pkl 文件名解析训练步数 | `fn` | `—` |
| 7 | `_replay_n_games` | `rl/dashboard.py:215` | 读回放文件对局数（带 mtime/size 缓存） | `path, mtime_ts, size` | `—` |
| 8 | `scan_replays` | `rl/dashboard.py:233` | 扫描回放目录返回最近 limit 个文件元数据 | `replays_dir, limit=30` | `—` |
| 9 | `build_replays_payload` | `rl/dashboard.py:258` | 组装回放列表接口数据 | `replays_dir` | `—` |
| 10 | `load_replay_payload` | `rl/dashboard.py:270` | 加载单回放：无 game_idx 返列表，否则返帧 | `replays_dir, filename, game_idx=None` | `—` |
| 11 | `_agent_label` | `rl/dashboard.py:330` | 模型 id 转友好名（frozen_copy 视为 main 冻结副本） | `mid` | `—` |
| 12 | `_other_side_id` | `rl/dashboard.py:337` | 由 pair/side0 推出 player-1 侧模型 id | `meta` | `—` |
| 13 | `_new_agent` | `rl/dashboard.py:350` | 新建空的模型卡牌统计累计器 | `mid` | `—` |
| 14 | `_stat_file_cards` | `rl/dashboard.py:355` | 统计单回放文件中各模型出牌与卡组构成（带缓存） | `path` | `—` |
| 15 | `_stat_file_cards._agent`<br>（内嵌于 `_stat_file_cards`） | `rl/dashboard.py:377` | 取或新建该模型的统计条目 | `mid` | `—` |
| 16 | `build_card_stats_payload` | `rl/dashboard.py:430` | 汇总单文件或最近 N 个回放的卡牌使用统计 | `replays_dir, filename=None, n_files=3` | `—` |
| 17 | `make_demo_replays` | `rl/dashboard.py:502` | 生成演示用合成联赛录像 | `replays_dir, n_games=2, n_frames=40` | `—` |
| 18 | `Handler` | `rl/dashboard.py:1995` | 仪表盘 HTTP 请求处理器（持有面板路径与对局 session） | `—` | `—` |
| 19 | `Handler._ensure_play`<br>（内嵌于 `Handler`） | `rl/dashboard.py:2008` | 懒加载人机对战 session，失败记录可见错误 | `self` | `—` |
| 20 | `Handler._read_body`<br>（内嵌于 `Handler`） | `rl/dashboard.py:2026` | 解析 POST body 为 JSON，失败返回空 dict | `self` | `—` |
| 21 | `Handler._send`<br>（内嵌于 `Handler`） | `rl/dashboard.py:2038` | 发送响应（含 no-cache 头与 Content-Length） | `self, code, body, ctype` | `—` |
| 22 | `Handler.do_GET`<br>（内嵌于 `Handler`） | `rl/dashboard.py:2048` | 路由各 GET 接口：页面/state/sweep/solo/play/replays | `self` | `—` |
| 23 | `Handler.log_message`<br>（内嵌于 `Handler`） | `rl/dashboard.py:2099` | 把访问日志写到 stderr | `self, fmt, *args` | `—` |
| 24 | `Handler.do_POST`<br>（内嵌于 `Handler`） | `rl/dashboard.py:2102` | 处理人机对战出牌与新一局请求 | `self` | `—` |
| 25 | `make_demo_state` | `rl/dashboard.py:2141` | 生成演示用联赛状态 JSON（含合成 Elo 历史） | `path, n_points=10, seed=0` | `—` |
| 26 | `make_demo_sweep` | `rl/dashboard.py:2192` | 生成两份演示 flow-sweep summary | `root, seed=1` | `—` |
| 27 | `make_demo_solo` | `rl/dashboard.py:2239` | 生成演示用 solo_state.json（胜率上升曲线） | `path, n_points=10, seed=3` | `—` |
| 28 | `main` | `rl/dashboard.py:2270` | 解析命令行参数、准备演示数据与路径并启动 HTTP 服务 | `—` | `—` |

##### B.8.9 `rl/decks.py`（7 个符号）

| 序号 | 函数/方法 | 文件:行号 | 一句话作用 | 关键参数（名字=默认值） | 返回 |
|---|---|---|---|---|--- |
| 1 | `_engine_lookup` | `rl/decks.py:42` | 建"归一化卡名→引擎卡名"查询表（惰性单次） | `—` | `—` |
| 2 | `normalize_card` | `rl/decks.py:53` | 数据集卡名 → 引擎卡名；无法映射返回 None。 | `name: str` | `—` |
| 3 | `map_deck_cards` | `rl/decks.py:74` | 映射一副 8 卡卡组；未命中的卡槽用引擎卡池补位（确定性）。 | `cards, seed=0` | `—` |
| 4 | `_default_paths` | `rl/decks.py:92` | 返回三分类卡组 JSON 的候选探测路径列表 | `—` | `—` |
| 5 | `load_classified_decks` | `rl/decks.py:104` | 加载三分类卡组 → list[{"archetype", "cards"(8 张引擎卡), "missing"}] | `path=None` | `—` |
| 6 | `decks_by_archetype` | `rl/decks.py:125` | 按 archetype 分组：dict[str, list[deck]]。 | `decks` | `—` |
| 7 | `classify_stats` | `rl/decks.py:133` | 统计各分类卡组数 / 平均补位卡数。 | `decks` | `—` |

##### B.8.10 `rl/diagnostics.py`（5 个符号）

| 序号 | 函数/方法 | 文件:行号 | 一句话作用 | 关键参数（名字=默认值） | 返回 |
|---|---|---|---|---|--- |
| 1 | `print_safe` | `rl/diagnostics.py:37` | print 兜底：编码不支持的字符降级为 '?'，而不是抛 UnicodeEncodeError。 | `msg` | `—` |
| 2 | `_gru_gate_stats` | `rl/diagnostics.py:59` | 手动复现 GRUCell 内部，返回候选 n 的 \|·\| 均值（饱和判据）。 | `policy, enc, h_prev` | `—` |
| 3 | `gru_vitality` | `rl/diagnostics.py:80` | 测 GRU / value_head 的"活力"。 | `policy, frames, max_frames=96` | `—` |
| 4 | `check_vitality` | `rl/diagnostics.py:124` | 按 THRESHOLDS 返回告警列表（空 = 通过）。P0-C 用。 | `vit` | `—` |
| 5 | `check_policy_architecture` | `rl/diagnostics.py:143` | **启动前检查**（v3 P0-C）：静态可判定的饱和病因护栏，返回告警列表。 | `policy` | `—` |

##### B.8.11 `rl/elo.py`（8 个符号）

| 序号 | 函数/方法 | 文件:行号 | 一句话作用 | 关键参数（名字=默认值） | 返回 |
|---|---|---|---|---|--- |
| 1 | `Elo` | `rl/elo.py:16` | Elo 评分系统：更新、评分表与 JSON 持久化 | `—` | `—` |
| 2 | `Elo.__init__`<br>（内嵌于 `Elo`） | `rl/elo.py:17` | 初始化 K 系数、初始分与评分表 | `self, k: float=32.0, initial: float=1500.0` | `—` |
| 3 | `Elo.ensure`<br>（内嵌于 `Elo`） | `rl/elo.py:22` | 确保成员有初始分并返回其评分 | `self, agent_id` | `float` |
| 4 | `Elo.expected`<br>（内嵌于 `Elo`） | `rl/elo.py:28` | 算 r_a 对 r_b 的期望胜率 | `r_a: float, r_b: float` | `float` |
| 5 | `Elo.update`<br>（内嵌于 `Elo`） | `rl/elo.py:31` | 按成绩与局数缩放的 K 更新双方评分 | `self, agent_a, agent_b, score_a: float, n_games: int=1` | `—` |
| 6 | `Elo.table`<br>（内嵌于 `Elo`） | `rl/elo.py:44` | 返回按评分降序排列的评分表 | `self` | `—` |
| 7 | `Elo.save`<br>（内嵌于 `Elo`） | `rl/elo.py:47` | 把 K/初始分/评分表写入 JSON | `self, path` | `—` |
| 8 | `Elo.load`<br>（内嵌于 `Elo`） | `rl/elo.py:53` | 从 JSON 读回评分表构造 Elo 实例 | `cls, path` | `—` |

##### B.8.12 `rl/env_wrapper.py`（28 个符号）

| 序号 | 函数/方法 | 文件:行号 | 一句话作用 | 关键参数（名字=默认值） | 返回 |
|---|---|---|---|---|--- |
| 1 | `tower_total_hp` | `rl/env_wrapper.py:74` | 一方的总塔血 = 国王塔 + 两座公主塔（塔血归一化的分母；lv11 标准塔 = 10928）。 | `troop_hp: float, king_hp: float` | `float` |
| 2 | `tower_value_mult` | `rl/env_wrapper.py:101` | 单位血价值倍数：凹形溢价 + 王塔贬值闸门。 | `hp_ratio: float, *, king: bool, princesses_alive: int, k: float=DEFAULT_TOWER_PREMIUM_K, king_gate: float=DEFAULT_KING_GATE` | `float` |
| 3 | `tower_premium_k` | `rl/env_wrapper.py:121` | 从奖励字典取溢价强度（缺省 2.0；经 reward_to_env 透传可调）。 | `rw` | `float` |
| 4 | `_princesses_alive` | `rl/env_wrapper.py:126` | 存活公主塔数（血比例 >0 = 未破）。 | `hp_ratio_king, hp_ratio_left, hp_ratio_right` | `int` |
| 5 | `_per_tower_norm_dmg` | `rl/env_wrapper.py:131` | per-tower 塔血差 → 差异化后按 lv11 锚归一化的加权伤害（正=塔掉血）。 | `dmg_list, old_list, new_list, max_list, *, king, k, king_gate` | `—` |
| 6 | `_phase_weights` | `rl/env_wrapper.py:164` | 返回 (tower_opp, tower_self, edw_coef)：120s 切换的两段离散权重。 | `rw, battle_time` | `—` |
| 7 | `compute_reward` | `rl/env_wrapper.py:181` | 逐决策帧奖励（RLEnv.step 与 selftest 共用）。 | `rw, *, blue_hps_old, red_hps_old, blue_hps_new, red_hps_new, blue_left_old, red_left_old, blue_left_new, red_left_new, my_elixir_before, opp_elixir_before, my_elixir_after, opp_elixir_after, my_v_before=0.0, opp_v_before=0.0, my_v_after=0.0, opp_v_after=0.0, winner, invalid_count, blue_hps_max=No…` | `—` |
| 8 | `ActionBundleSpace` | `rl/env_wrapper.py:268` | ActionBundle 的 gym 空间占位（自定义训练用，不参与 SB3 标准优化）。 | `—` | `—` |
| 9 | `ActionBundleSpace.__init__`<br>（内嵌于 `ActionBundleSpace`） | `rl/env_wrapper.py:271` | 记录 k_max 并初始化 gym 空间 | `self, k_max: int=K_MAX` | `—` |
| 10 | `ActionBundleSpace.sample`<br>（内嵌于 `ActionBundleSpace`） | `rl/env_wrapper.py:275` | 随机采 0..k_max 个子动作组成 bundle | `self, mask=None, rng=None` | `—` |
| 11 | `ActionBundleSpace.contains`<br>（内嵌于 `ActionBundleSpace`） | `rl/env_wrapper.py:282` | 判定是否为不超 k_max 的 ActionBundle | `self, x` | `—` |
| 12 | `ActionBundleSpace.__repr__`<br>（内嵌于 `ActionBundleSpace`） | `rl/env_wrapper.py:285` | 空间字符串表示 | `self` | `—` |
| 13 | `RLEnv` | `rl/env_wrapper.py:289` | gym 环境：包装 BattleState、观测与奖励 | `—` | `—` |
| 14 | `RLEnv.__init__`<br>（内嵌于 `RLEnv`） | `rl/env_wrapper.py:292` | 初始化卡组/奖励权重/掩码缓存与资源账 | `self, opponent: Optional[Callable]=None, deck0: Optional[list]=None, deck1: Optional[list]=None, deck0_factory: Optional[Callable[[], list]]=None, deck1_factory: Optional[Callable[[], list]]=None, visualize: bool=False, speed: float=1.0, decision_frames: int=30, dt: float=1 / 60, record_hidden: b…` | `—` |
| 15 | `RLEnv.reset`<br>（内嵌于 `RLEnv`） | `rl/env_wrapper.py:348` | 重洗双方卡组、重建战场并同步塔血 | `self, *, seed=None, options=None` | `—` |
| 16 | `RLEnv.observe`<br>（内嵌于 `RLEnv`） | `rl/env_wrapper.py:390` | 返回该玩家的观测字典 | `self, player_id: int=0` | `dict` |
| 17 | `RLEnv.get_hidden_state`<br>（内嵌于 `RLEnv`） | `rl/env_wrapper.py:393` | 返回特权隐藏标签字典 | `self` | `dict` |
| 18 | `RLEnv.get_prophet_state`<br>（内嵌于 `RLEnv`） | `rl/env_wrapper.py:396` | 特权完整状态摘要（仅先知规划器使用）。 | `self` | `dict` |
| 19 | `RLEnv.get_action_mask`<br>（内嵌于 `RLEnv`） | `rl/env_wrapper.py:417` | 返回玩家 0（agent）的动态动作掩码。 | `self, partial_bundle: Optional[ActionBundle]=None` | `dict` |
| 20 | `RLEnv.get_action_mask_for`<br>（内嵌于 `RLEnv`） | `rl/env_wrapper.py:421` | 返回指定玩家的动态动作掩码，供 autoregressive bundle head 使用。 | `self, player_id: int, partial_bundle: Optional[ActionBundle]=None` | `dict` |
| 21 | `RLEnv._run_opponent`<br>（内嵌于 `RLEnv`） | `rl/env_wrapper.py:490` | 执行对手动作，返回结构化 played 列表 [{card, x, y}, ...]（P1-5）。 | `self` | `list` |
| 22 | `RLEnv._random_opponent`<br>（内嵌于 `RLEnv`） | `rl/env_wrapper.py:524` | 用修复后的掩码采样合法格子（P2：不再六成落禁区）。 | `self` | `list` |
| 23 | `RLEnv._unit_hp_map`<br>（内嵌于 `RLEnv`） | `rl/env_wrapper.py:550` | 存活部署单位（非塔、非法术临时实体）hp 快照：{entity_id: (player, hp)}。 | `self` | `—` |
| 24 | `RLEnv._deploy_ledger`<br>（内嵌于 `RLEnv`） | `rl/env_wrapper.py:561` | 部署记账：把该卡实际花费分摊给本帧新增的部署实体（创建即固定，死亡注销）。 | `self, pid, elixir_before, card_name` | `—` |
| 25 | `RLEnv._collect_deaths`<br>（内嵌于 `RLEnv`） | `rl/env_wrapper.py:583` | 注销死亡实体的部署份额（推进段死亡在每步末统一清理；份额固定，不做 HP 折价）。 | `self` | `—` |
| 26 | `RLEnv._tower_snapshot`<br>（内嵌于 `RLEnv`） | `rl/env_wrapper.py:594` | 三塔血量快照 [king, left, right]（塔血差异化定价的 per-tower 输入）。 | `p` | `list` |
| 27 | `RLEnv.step`<br>（内嵌于 `RLEnv`） | `rl/env_wrapper.py:598` | 推进一个决策帧并返回 obs/reward/term/trunc/info | `self, action_bundle: ActionBundle` | `—` |
| 28 | `legacy_action_to_bundle` | `rl/env_wrapper.py:726` | 兼容旧接口 (slot, y, x) / list。 | `act` | `ActionBundle` |

##### B.8.13 `rl/evaluate.py`（13 个符号）

| 序号 | 函数/方法 | 文件:行号 | 一句话作用 | 关键参数（名字=默认值） | 返回 |
|---|---|---|---|---|--- |
| 1 | `_overtime_open` | `rl/evaluate.py:35` | 惰性 import 判加时窗口，避免循环依赖 | `battle` | `—` |
| 2 | `_ece` | `rl/evaluate.py:41` | 按分箱计算置信度校准误差 ECE | `conf, acc, n_bins=10` | `—` |
| 3 | `_plan_region_hit` | `rl/evaluate.py:52` | 粗判首部署格与 plan.focus_region 的半场/分边吻合 | `region: str, x: float, y: float` | `—` |
| 4 | `_adoption_seed` | `rl/evaluate.py:82` | 返回逐意图采纳计数的零值模板 | `—` | `—` |
| 5 | `_adoption_summary` | `rl/evaluate.py:89` | 把采纳原始计数转成可落盘的率值（空样本为 None） | `adoption` | `—` |
| 6 | `_record_adoption` | `rl/evaluate.py:111` | 按 plan 意图累计本帧 region/对牌/预算/持牌采纳计数 | `stats, plan, bundle, obs=None` | `—` |
| 7 | `_make_opponent` | `rl/evaluate.py:157` | 按名字构造评测对手（heuristic/checkpoint/随机） | `env, opponent, opponent_policy=None, rng=None` | `—` |
| 8 | `run_eval` | `rl/evaluate.py:168` | 评测主入口：跑 N 局统计胜率/奖励/信念/采纳指标 | `policy_path, n_games=50, opponent='random', seed=0, hidden_dim=None, max_steps=300, opponent_policy=None, ablation=None` | `—` |
| 9 | `run_ablation` | `rl/evaluate.py:316` | 跑 belief/plan 置零消融并落盘 JSON+CSV 与显著性判定 | `policy_path, n_games=200, opponent='random', seed=0, hidden_dim=None, max_steps=300, opponent_policy=None, out_path=None` | `—` |
| 10 | `run_ablation.delta_vs`<br>（内嵌于 `run_ablation`） | `rl/evaluate.py:339` | 内层闭包：算变体相对 full 的 ΔWinRate、SE 与 z 判定 | `name` | `—` |
| 11 | `run_ablation._adopt_agg`<br>（内嵌于 `run_ablation`） | `rl/evaluate.py:393` | 内层闭包：跨意图合计对牌采纳率与预算服从率 | `a` | `—` |
| 12 | `load_policy` | `rl/evaluate.py:473` | 从检查点加载策略（薄包装 load_checkpoint） | `path, hidden_dim=None` | `—` |
| 13 | `evaluate_belief` | `rl/evaluate.py:477` | 独立信念协议：只统计下一张预测精度/Brier/ECE | `policy_path, n_games=500, seed=0, hidden_dim=None, max_steps=300` | `—` |

##### B.8.14 `rl/export_replay.py`（1 个符号）

| 序号 | 函数/方法 | 文件:行号 | 一句话作用 | 关键参数（名字=默认值） | 返回 |
|---|---|---|---|---|--- |
| 1 | `main` | `rl/export_replay.py:23` | 跑对局并导出 EpisodeReplay pickle | `—` | `—` |

##### B.8.15 `rl/flow_league.py`（18 个符号）

| 序号 | 函数/方法 | 文件:行号 | 一句话作用 | 关键参数（名字=默认值） | 返回 |
|---|---|---|---|---|--- |
| 1 | `build_flow_pools` | `rl/flow_league.py:67` | 构建 6 个卡组池（三分类+全量+随机+main 全量） | `cfg, n_random_decks=30` | `—` |
| 2 | `flow_pair_games` | `rl/flow_league.py:90` | 按全配对规则统计一次训练的总对局数 | `pools` | `—` |
| 3 | `scale_pools` | `rl/flow_league.py:100` | 按固定种子把各卡组池缩小 factor 倍 | `pools, factor` | `—` |
| 4 | `build_flow_models` | `rl/flow_league.py:124` | 构建 6 个 FollowerPolicy 与对应 PPOTrainer | `cfg, device, belief_dim` | `—` |
| 5 | `save_flow_models` | `rl/flow_league.py:147` | 把 6 个模型保存为 flow_<id>.pt | `cfg, models` | `—` |
| 6 | `_flow_progress_path` | `rl/flow_league.py:153` | 返回 flow 断点文件 flow_run_state.json 路径 | `cfg` | `—` |
| 7 | `_save_flow_progress` | `rl/flow_league.py:157` | 落盘对进度与 6 个优化器状态 | `cfg, pair_ix, game_ix, total_games, trainers` | `—` |
| 8 | `_load_flow_resume` | `rl/flow_league.py:166` | 从磁盘恢复 6 模型/优化器与对进度，失败返回 None | `cfg, device` | `—` |
| 9 | `_hp_state` | `rl/flow_league.py:206` | 取 (总塔血, 剩余皇冠数, 圣水) 供双视角镜像算奖励 | `p` | `—` |
| 10 | `_tower_state` | `rl/flow_league.py:213` | 取三塔血量列表供 per-tower 镜像定价 | `p` | `—` |
| 11 | `new_ep_buf` | `rl/flow_league.py:218` | 新建空的对局轨迹缓冲字典 | `—` | `—` |
| 12 | `_flush_episode` | `rl/flow_league.py:223` | 整局算 GAE 后把转移追加进缓冲（含截断 bootstrap） | `buf, ep, policy, last_obs, last_belief, last_plan, last_hidden, truncated, cfg` | `—` |
| 13 | `_drain` | `rl/flow_league.py:242` | 按 batch_size 流式消费缓冲并训练后释放 | `buf, trainer, batch_size` | `—` |
| 14 | `_play_one` | `rl/flow_league.py:252` | 两模型按指定卡组打 1 局不换边，双侧采轨迹并各自算奖励 | `env, pol_a, pol_b, deckA, deckB, cfg, seed, max_steps, buf_a, buf_b, bp, prophet, rng, rw_a, rw_b` | `—` |
| 15 | `run_flow` | `rl/flow_league.py:364` | 全配对分流派联赛主循环（15 对，支持断点与注入池） | `cfg, resume=False, n_random_decks=30, pools=None, max_pairs=None, games_per_deck_pair=1, save=True, seed=None, quiet=False` | `—` |
| 16 | `_sweep_trend` | `rl/flow_league.py:478` | 按首末 main 轮内估计判定曲线涨跌（\|Δ\|/SE≥2σ） | `rows` | `—` |
| 17 | `_write_sweep` | `rl/flow_league.py:491` | 把 sweep 累计结果落盘 summary.json/csv（含进度） | `out_dir, spec, strategy, pool_scale, sizes, n_runs, games_per_pair, per_run, total_budget, eval_games, device, rows, status, elapsed_s` | `—` |
| 18 | `run_flow_sweep` | `rl/flow_league.py:528` | 缩小 10× 池的 flow 数据效率 A/B 对比 | `cfg, strategy='stream', n_runs=None, games_per_pair=None, pool_scale=0.1, n_random_decks=30, pools=None, eval_games=None` | `—` |

##### B.8.16 `rl/follower.py`（21 个符号）

| 序号 | 函数/方法 | 文件:行号 | 一句话作用 | 关键参数（名字=默认值） | 返回 |
|---|---|---|---|---|--- |
| 1 | `save_checkpoint` | `rl/follower.py:55` | 保存带元数据的 checkpoint（P0-5）。 | `policy, path` | `—` |
| 2 | `load_checkpoint` | `rl/follower.py:67` | 加载 checkpoint；优先读取元数据，旧格式（裸 state_dict）回退到显式/常量维度。 | `path, hidden_dim=None, plan_dim=None, belief_dim=None, value_bypass=None, value_independent=None` | `—` |
| 3 | `FollowerPolicy` | `rl/follower.py:154` | 跟随者策略网络：CNN+实体嵌入+GRU+多决策头 | `—` | `—` |
| 4 | `FollowerPolicy.__init__`<br>（内嵌于 `FollowerPolicy`） | `rl/follower.py:155` | stop_logit_bias：新初始化时给 STOP logit 的偏置（负数=初始更愿意出牌）。 | `self, hidden=256, plan_dim=None, belief_dim=None, num_entity=NUM_ENTITY, stop_logit_bias=-1.0, value_bypass=False, value_independent=False` | `—` |
| 5 | `FollowerPolicy.to_device`<br>（内嵌于 `FollowerPolicy`） | `rl/follower.py:263` | 记录 device 并把模块搬到该设备 | `self, device` | `—` |
| 6 | `FollowerPolicy._encode_parts`<br>（内嵌于 `FollowerPolicy`） | `rl/follower.py:268` | 返回 `(fused, enc)`：fused 是融合特征（E′ 独立价值编码器的输入）， | `self, obs, belief_token, plan_token` | `—` |
| 7 | `FollowerPolicy._encode`<br>（内嵌于 `FollowerPolicy`） | `rl/follower.py:300` | 单条编码，返回共享编码器输出 enc | `self, obs, belief_token, plan_token` | `—` |
| 8 | `FollowerPolicy._value_from`<br>（内嵌于 `FollowerPolicy`） | `rl/follower.py:303` | 按当前 value 架构算 value（**唯一实现**，act/evaluate/diagnostics 共用）。 | `self, enc, h, fused` | `—` |
| 9 | `FollowerPolicy._slot_mask_tensor`<br>（内嵌于 `FollowerPolicy`） | `rl/follower.py:316` | 把 mask 转成 (NUM_SLOT_OPTIONS,) 的合法选项掩码。 | `self, mask` | `—` |
| 10 | `FollowerPolicy._plan_biases`<br>（内嵌于 `FollowerPolicy`） | `rl/follower.py:332` | 从 plan 向量解析软偏置：(slot_bias(NUM_SLOT_OPTIONS,), cell_bias(H,W))。 | `self, plan_token` | `—` |
| 11 | `FollowerPolicy._mask_or_fallback`<br>（内嵌于 `FollowerPolicy`） | `rl/follower.py:379` | 取第 j 个掩码；缺省用全合法回退（与 evaluate 单条路径一致）。 | `masks, j` | `—` |
| 12 | `FollowerPolicy._sub_vec`<br>（内嵌于 `FollowerPolicy`） | `rl/follower.py:387` | 构造子动作 onehot+坐标向量 | `self, option_idx, x=0.0, y=0.0` | `—` |
| 13 | `FollowerPolicy._sub_update`<br>（内嵌于 `FollowerPolicy`） | `rl/follower.py:394` | 用 sub_emb 推进 GRU 一步 | `self, h, option_idx, x=0.0, y=0.0` | `—` |
| 14 | `FollowerPolicy._encode_batch_parts`<br>（内嵌于 `FollowerPolicy`） | `rl/follower.py:397` | 批量编码：返回 `(fused, enc)`（N 个观测一次前向）。 | `self, obs_list, belief_list, plan_list` | `—` |
| 15 | `FollowerPolicy._encode_batch`<br>（内嵌于 `FollowerPolicy`） | `rl/follower.py:430` | 批量编码，返回 enc | `self, obs_list, belief_list, plan_list` | `—` |
| 16 | `FollowerPolicy.act`<br>（内嵌于 `FollowerPolicy`） | `rl/follower.py:433` | 在线动作生成：返回 (ActionBundle, logprob, value, hidden, masks)。 | `self, obs, belief_token, plan_token, get_mask, hidden=None, deterministic=False` | `—` |
| 17 | `FollowerPolicy.masks_for`<br>（内嵌于 `FollowerPolicy`） | `rl/follower.py:486` | 为给定 bundle 重建 rollout 时的掩码序列（BC/离线监督用，不采样）。 | `self, obs, belief_token, plan_token, bundle, get_mask` | `—` |
| 18 | `FollowerPolicy.act_parallel`<br>（内嵌于 `FollowerPolicy`） | `rl/follower.py:504` | 批量 act：N 个 env 一次前向（并行多环境训练用）。 | `self, obs_list, belief_list, plan_list, get_mask_list, hidden_list=None, deterministic=False, get_masks_batch=None` | `—` |
| 19 | `FollowerPolicy.evaluate_batch`<br>（内嵌于 `FollowerPolicy`） | `rl/follower.py:610` | 可微批量重放（并行 PPO 更新用）。 | `self, obs_list, belief_list, plan_list, bundle_list, masks_list, hidden_list=None` | `—` |
| 20 | `FollowerPolicy.value`<br>（内嵌于 `FollowerPolicy`） | `rl/follower.py:704` | 只算当前状态的 value（不采样动作），用于截断 episode 的 GAE bootstrap（P1-7）。 | `self, obs, belief_token, plan_token, hidden=None` | `float` |
| 21 | `FollowerPolicy.evaluate`<br>（内嵌于 `FollowerPolicy`） | `rl/follower.py:713` | 可微重放给定 bundle（使用 rollout 时记录的掩码/隐状态）。 | `self, obs, belief_token, plan_token, bundle, masks, hidden=None` | `—` |

##### B.8.17 `rl/human_play.py`（13 个符号）

| 序号 | 函数/方法 | 文件:行号 | 一句话作用 | 关键参数（名字=默认值） | 返回 |
|---|---|---|---|---|--- |
| 1 | `_make_env` | `rl/human_play.py:52` | 建人机对战用 RLEnv（同副卡组且记 hidden） | `cfg, deck, seed` | `—` |
| 2 | `load_policy` | `rl/human_play.py:58` | 加载 FollowerPolicy checkpoint（带元数据）。 | `path, hidden_dim=128` | `—` |
| 3 | `HumanPlaySession` | `rl/human_play.py:63` | 一局人机对战：人出 ActionBundle，模型对手每 tick 应对；全程记录训练数据。 | `—` | `—` |
| 4 | `HumanPlaySession.__init__`<br>（内嵌于 `HumanPlaySession`） | `rl/human_play.py:66` | 初始化会话：策略对手/信念/录像/session_id | `self, policy, cfg=None, deck=None, seed=0, max_steps=600, out_dir=None, session_id=None` | `—` |
| 5 | `HumanPlaySession.start`<br>（内嵌于 `HumanPlaySession`） | `rl/human_play.py:94` | 重置环境与录像并返回初始状态 | `self` | `—` |
| 6 | `HumanPlaySession.state`<br>（内嵌于 `HumanPlaySession`） | `rl/human_play.py:101` | 返回给前端的战场帧 + 手牌 + 合法落点（battle_snapshot 帧可直接复用渲染器）。 | `self` | `—` |
| 7 | `HumanPlaySession.act`<br>（内嵌于 `HumanPlaySession`） | `rl/human_play.py:124` | 人类出牌：(slot 1..K_MAX, x, y 本地坐标)；slot=0 = 空过等待（noop 推进 0.5s 回圣水）。 | `self, slot, x, y` | `—` |
| 8 | `HumanPlaySession.save`<br>（内嵌于 `HumanPlaySession`） | `rl/human_play.py:164` | 把本局 EpisodeReplay + BC 样本落盘；返回 (ep_path, bc_path, meta_path)。 | `self, out_dir=None, keep_bc=True` | `—` |
| 9 | `load_bc_samples` | `rl/human_play.py:194` | 读取 data_dir 下所有 bc_*.pkl 的 BC 样本并合并。 | `data_dir` | `—` |
| 10 | `train_bc_from_human` | `rl/human_play.py:205` | 在人类 BC 样本上做行为克隆（模仿学习预训练），产物可 --init-from 给 PPO。 | `data_dir, out='follower_human.pt', epochs=5, lr=0.001, hidden_dim=128, seed=0` | `—` |
| 11 | `export_data` | `rl/human_play.py:236` | 把 human_data 目录导出成可直接训练的文件： | `data_dir, out_belief=None, out_bc=None` | `—` |
| 12 | `drive_games` | `rl/human_play.py:259` | 无 UI 冒烟：用随机合法动作驱动 n 局，验证数据采集链路（headless 测试）。 | `policy, n_games, seed=0, max_steps=600, out_dir=None, cfg=None, deck=None` | `—` |
| 13 | `main` | `rl/human_play.py:291` | 人机对战命令行入口 | `—` | `—` |

##### B.8.18 `rl/launcher_menu.py`（9 个符号）

| 序号 | 函数/方法 | 文件:行号 | 一句话作用 | 关键参数（名字=默认值） | 返回 |
|---|---|---|---|---|--- |
| 1 | `ask` | `rl/launcher_menu.py:60` | 逐项问答：回车=默认值；输入 b/B=返回上一步；q/Q=退出。 | `prompt, default, cast=str, allow_back=True` | `—` |
| 2 | `ask_yesno` | `rl/launcher_menu.py:81` | 是/否询问（支持 .. 返回上级） | `prompt, default=True` | `—` |
| 3 | `pick` | `rl/launcher_menu.py:92` | 打印选项列表并让用户选序号；回车选 default_idx。 | `label, options, default_idx=0, extra=None` | `—` |
| 4 | `pick_config` | `rl/launcher_menu.py:120` | 交互选择配置预设或自定义 JSON | `—` | `—` |
| 5 | `collect_params` | `rl/launcher_menu.py:134` | 第三层：逐项参数（显示预设值，回车不改）。返回参数 dict + 标志（..=返回）。 | `mode` | `—` |
| 6 | `build_cmd` | `rl/launcher_menu.py:200` | 由菜单选择拼装 run_league 命令行 | `mode, config, load_config, p` | `—` |
| 7 | `state_arg` | `rl/launcher_menu.py:229` | 按模式返回对应断点状态路径参数 | `mode, p` | `—` |
| 8 | `main` | `rl/launcher_menu.py:239` | 向导入口：逐层选择并启动训练 | `—` | `—` |
| 9 | `launch` | `rl/launcher_menu.py:315` | 新开独立控制台窗口（Windows），其它平台前台子进程。 | `cmd` | `—` |

##### B.8.19 `rl/league.py`（15 个符号）

| 序号 | 函数/方法 | 文件:行号 | 一句话作用 | 关键参数（名字=默认值） | 返回 |
|---|---|---|---|---|--- |
| 1 | `LeagueAgent` | `rl/league.py:24` | 联赛成员数据类：id/类型/策略/检查点/ELO | `—` | `—` |
| 2 | `League` | `rl/league.py:32` | 联赛容器：成员表 + PFSP + Elo + 历史与统计持久化 | `—` | `—` |
| 3 | `League.__init__`<br>（内嵌于 `League`） | `rl/league.py:33` | 初始化成员表、PFSP、Elo、历史与计数器 | `self, pfsp_beta: float=1.0, elo_k: float=32.0, seed: int=0` | `—` |
| 4 | `League.add_agent`<br>（内嵌于 `League`） | `rl/league.py:44` | 注册成员（同 id 需 replace=True）并确保其 Elo | `self, agent_id, kind='main', policy=None, path=None, replace=False` | `—` |
| 5 | `League.sample_opponent`<br>（内嵌于 `League`） | `rl/league.py:52` | 按 PFSP 从其余成员中采样一个对手 | `self, agent_id` | `LeagueAgent` |
| 6 | `League.record_match`<br>（内嵌于 `League`） | `rl/league.py:59` | 累计一局成绩：更新 Elo、双向 PFSP 胜率与对局历史 | `self, agent_a, agent_b, score_a: float, n_games: int=1` | `—` |
| 7 | `League.register_checkpoint`<br>（内嵌于 `League`） | `rl/league.py:66` | 把当前策略的权重副本注册为新的 historical 条目 | `self, agent_id, policy, path=None, metadata=None` | `—` |
| 8 | `League.refresh_snapshot`<br>（内嵌于 `League`） | `rl/league.py:86` | 把 main 权重副本刷到固定槽位 _ckpt，不重置 Elo | `self, agent_id, policy, path=None` | `—` |
| 9 | `League.record_elo_history`<br>（内嵌于 `League`） | `rl/league.py:104` | 记录当前各成员 (step, elo) 供 UI 画曲线 | `self, step: int=None` | `—` |
| 10 | `League.record_round_stats`<br>（内嵌于 `League`） | `rl/league.py:111` | 记录一轮评估的聚合估计与局数（供画误差棒） | `self, step, stats` | `—` |
| 11 | `League.add_exploiter`<br>（内嵌于 `League`） | `rl/league.py:125` | 用递增计数器新增 exploiter 条目并返回其 id | `self, policy, path=None` | `—` |
| 12 | `League.remove_agent`<br>（内嵌于 `League`） | `rl/league.py:131` | 删除成员及其 Elo/PFSP 键，返回是否删除 | `self, agent_id` | `—` |
| 13 | `League.elo_table`<br>（内嵌于 `League`） | `rl/league.py:141` | 返回按 Elo 降序的成员评分表 | `self` | `—` |
| 14 | `League.save_state`<br>（内嵌于 `League`） | `rl/league.py:146` | 把 Elo/PFSP/成员/历史/统计落盘为 JSON | `self, path` | `—` |
| 15 | `League.load_state`<br>（内嵌于 `League`） | `rl/league.py:162` | 从 JSON 恢复联赛状态并可选重新挂载策略本体 | `self, path, policies=None` | `—` |

##### B.8.20 `rl/mcts.py`（24 个符号）

| 序号 | 函数/方法 | 文件:行号 | 一句话作用 | 关键参数（名字=默认值） | 返回 |
|---|---|---|---|---|--- |
| 1 | `MCTSConfig` | `rl/mcts.py:43` | MCTS 配置：模拟次数/深度/叶推演/UCT 系数 | `—` | `—` |
| 2 | `_tower_premium_loss` | `rl/mcts.py:68` | 逐塔累计塔损：凹形溢价+王塔存活闸门 | `player: PlayerState` | `float` |
| 3 | `node_value` | `rl/mcts.py:99` | 值函数：塔血差+皇冠差+资源账（双倍期切权重） | `battle, player_id: int, cfg: MCTSConfig` | `float` |
| 4 | `leaf_value` | `rl/mcts.py:128` | 叶估值：深拷贝确定性推演后计 node_value | `battle, player_id: int, cfg: MCTSConfig` | `float` |
| 5 | `enumerate_bundles` | `rl/mcts.py:143` | 枚举单卡 bundle 并经 validate_bundle 全量校验 | `battle, player_id: int, cfg: MCTSConfig, priors: Optional[dict]=None` | `list` |
| 6 | `enumerate_bundles.score`<br>（内嵌于 `enumerate_bundles`） | `rl/mcts.py:193` | 先验排序打分闭包（slot+1e-3·cell）降序截断 | `item` | `—` |
| 7 | `validate_slots` | `rl/mcts.py:201` | 手牌槽圣水掩码（转发 action_mask.slot_mask） | `p: PlayerState` | `np.ndarray` |
| 8 | `_cells_by_threat` | `rl/mcts.py:207` | 缺省格序：近敌军+低血塔加权，确定性双键排序 | `battle, player_id: int` | `np.ndarray` |
| 9 | `_foe_priority` | `rl/mcts.py:238` | 敌军目标优先级：残血塔权重高（最高 51×） | `e` | `float` |
| 10 | `_is_spell_card` | `rl/mcts.py:248` | 卡名是否 spell 类型 | `card_name: str` | `bool` |
| 11 | `_cells_by_spell_value` | `rl/mcts.py:253` | 法术格序：范围内敌方覆盖价值降序 | `battle, player_id: int, card_name: str` | `np.ndarray` |
| 12 | `_spell_target_value` | `rl/mcts.py:286` | 法术目标价值：min(hp,法术伤)+可击杀加成 | `e, spell_damage: float` | `float` |
| 13 | `_Node` | `rl/mcts.py:305` | MCTS 树节点（惰性枚举 untried，含访问统计） | `—` | `—` |
| 14 | `_Node.__init__`<br>（内嵌于 `_Node`） | `rl/mcts.py:309` | 初始化节点字段与父子/动作指针 | `self, battle, to_act, parent=None, action=None` | `—` |
| 15 | `_Node.q`<br>（内嵌于 `_Node`） | `rl/mcts.py:319` | 节点平均价值 sum_value/n_visits | `self` | `float` |
| 16 | `RLMCTS` | `rl/mcts.py:330` | 推理时浅 MCTS 搜索器（一次决策帧一次 search） | `—` | `—` |
| 17 | `RLMCTS.__init__`<br>（内嵌于 `RLMCTS`） | `rl/mcts.py:333` | 存策略先验来源/对手回调/配置 | `self, policy=None, opponent_fn: Optional[OpponentFn]=None, cfg: Optional[MCTSConfig]=None` | `—` |
| 18 | `RLMCTS.search`<br>（内嵌于 `RLMCTS`） | `rl/mcts.py:341` | 主循环：选择→扩展→叶推演→回传→选根最优 | `self, battle, player_id: int, obs=None` | `tuple` |
| 19 | `RLMCTS._expand_actions`<br>（内嵌于 `RLMCTS`） | `rl/mcts.py:391` | 本节点候选：(等待)+单卡 bundle 或对手单动作 | `self, node: _Node` | `list` |
| 20 | `RLMCTS._root_player`<br>（内嵌于 `RLMCTS`） | `rl/mcts.py:408` | 返回根行动方（_player_id，缺省 0） | `self` | `int` |
| 21 | `RLMCTS._priors`<br>（内嵌于 `RLMCTS`） | `rl/mcts.py:411` | 从策略网络取 (slot,cell) logits 先验，失败返回 None | `self, node: _Node` | `Optional[dict]` |
| 22 | `RLMCTS._opponent_bundle`<br>（内嵌于 `RLMCTS`） | `rl/mcts.py:427` | 调 opponent_fn 取对手 bundle，异常吞掉返回 None | `self, battle, opp_id: int` | `Optional[ActionBundle]` |
| 23 | `RLMCTS._apply`<br>（内嵌于 `RLMCTS`） | `rl/mcts.py:435` | 部署双方 bundle 并推进一步决策帧生成子节点 | `self, node: _Node, action` | `_Node` |
| 24 | `RLMCTS._uct_select`<br>（内嵌于 `RLMCTS`） | `rl/mcts.py:471` | UCT 选择：Q+c_uct·sqrt(logN/n) 最大子节点 | `self, node: _Node` | `_Node` |

##### B.8.21 `rl/observation.py`（3 个符号）

| 序号 | 函数/方法 | 文件:行号 | 一句话作用 | 关键参数（名字=默认值） | 返回 |
|---|---|---|---|---|--- |
| 1 | `observe` | `rl/observation.py:87` | 返回玩家 player_id 的可见观测字典。 | `battle, player_id: int=0` | `dict` |
| 2 | `hidden_labels` | `rl/observation.py:147` | 特权隐藏状态标签（只允许训练期使用，绝不进跟随者观测）。 | `battle, player_id: int=0` | `dict` |
| 3 | `hidden_labels._ids`<br>（内嵌于 `hidden_labels`） | `rl/observation.py:156` | 卡名列表转实体 id 数组（未知名记 0） | `cards` | `—` |

##### B.8.22 `rl/opponents.py`（14 个符号）

| 序号 | 函数/方法 | 文件:行号 | 一句话作用 | 关键参数（名字=默认值） | 返回 |
|---|---|---|---|---|--- |
| 1 | `build_card_pool` | `rl/opponents.py:49` | 过滤出引擎可部署的卡池（去王塔/0 费/非法类型） | `—` | `list` |
| 2 | `sample_deck` | `rl/opponents.py:71` | 从卡池随机抽 8 张不重复卡组成卡组 | `rng, pool` | `list` |
| 3 | `ScriptedPolicy` | `rl/opponents.py:76` | 脚本策略：掩码随机合法出牌，可作任意一侧对手 | `—` | `—` |
| 4 | `ScriptedPolicy.__init__`<br>（内嵌于 `ScriptedPolicy`） | `rl/opponents.py:86` | 校验 mode 并保存卡池/卡组池/RNG/env | `self, mode='random', pool=None, deck_pool=None, seed=0, env=None` | `—` |
| 5 | `ScriptedPolicy.deck`<br>（内嵌于 `ScriptedPolicy`） | `rl/opponents.py:96` | 返回本局卡组：deck_pool 抽一副/pool 重采/默认 | `self` | `—` |
| 6 | `ScriptedPolicy.play`<br>（内嵌于 `ScriptedPolicy`） | `rl/opponents.py:105` | 从该玩家合法掩码随机选一子动作返回 bundle | `self, env, player_id: int` | `ActionBundle` |
| 7 | `ScriptedPolicy.__call__`<br>（内嵌于 `ScriptedPolicy`） | `rl/opponents.py:118` | player-1 对手接口：用注入的 env 出牌 | `self, obs` | `—` |
| 8 | `SelfDefenderPolicy` | `rl/opponents.py:125` | 真防守脚本对手：威胁时按最优反制落点迎击 | `—` | `—` |
| 9 | `SelfDefenderPolicy.__init__`<br>（内嵌于 `SelfDefenderPolicy`） | `rl/opponents.py:138` | 保存种子/RNG/env/停手概率/卡组池 | `self, seed=0, env=None, passive_prob: float=0.6, deck_pool=None` | `—` |
| 10 | `SelfDefenderPolicy.deck`<br>（内嵌于 `SelfDefenderPolicy`） | `rl/opponents.py:148` | 有卡组池则随机抽一副，否则返回 None 用固定卡组 | `self` | `—` |
| 11 | `SelfDefenderPolicy._defend_action`<br>（内嵌于 `SelfDefenderPolicy`） | `rl/opponents.py:155` | 调 script_defender 取反制并把世界坐标转本地格 | `self, player_id: int` | `—` |
| 12 | `SelfDefenderPolicy._random_action`<br>（内嵌于 `SelfDefenderPolicy`） | `rl/opponents.py:181` | 无威胁帧从合法掩码随机取 (slot,x,y) | `self, player_id: int` | `—` |
| 13 | `SelfDefenderPolicy.play`<br>（内嵌于 `SelfDefenderPolicy`） | `rl/opponents.py:193` | 按概率随机缓出，否则防守反制，返回 bundle | `self, env, player_id: int=1` | `ActionBundle` |
| 14 | `SelfDefenderPolicy.__call__`<br>（内嵌于 `SelfDefenderPolicy`） | `rl/opponents.py:205` | player-1 对手接口：用注入的 env 出牌 | `self, obs` | `—` |

##### B.8.23 `rl/overtime.py`（1 个符号）

| 序号 | 函数/方法 | 文件:行号 | 一句话作用 | 关键参数（名字=默认值） | 返回 |
|---|---|---|---|---|--- |
| 1 | `overtime_open` | `rl/overtime.py:30` | 加时（突然死亡）窗口是否仍应继续，绕过 max_ep_steps 在常规时间末的截断。 | `battle` | `—` |

##### B.8.24 `rl/pfsp.py`（5 个符号）

| 序号 | 函数/方法 | 文件:行号 | 一句话作用 | 关键参数（名字=默认值） | 返回 |
|---|---|---|---|---|--- |
| 1 | `PFSP` | `rl/pfsp.py:29` | PFSP 对手采样：按 (1−胜率)^β 加权随机抽对手 | `—` | `—` |
| 2 | `PFSP.__init__`<br>（内嵌于 `PFSP`） | `rl/pfsp.py:30` | 校验并保存 beta/alpha/门禁参数，初始化 RNG 与胜率表 | `self, beta: float=1.0, seed: int=0, alpha: float=0.05, gate_hi: float=1.0, gate_penalty: float=1.0` | `—` |
| 3 | `PFSP.update_winrate`<br>（内嵌于 `PFSP`） | `rl/pfsp.py:46` | 用 EMA 更新 (a 对 b) 的胜率 | `self, agent_a, agent_b, score_a: float, alpha: float=None` | `—` |
| 4 | `PFSP.weights`<br>（内嵌于 `PFSP`） | `rl/pfsp.py:53` | 算各对手采样权重（未采样按 0 胜率，易胜对手降权） | `self, agent_id, opponents` | `np.ndarray` |
| 5 | `PFSP.sample`<br>（内嵌于 `PFSP`） | `rl/pfsp.py:66` | 按归一化权重随机抽一个对手 id | `self, agent_id, opponents` | `str` |

##### B.8.25 `rl/plan_space.py`（7 个符号）

| 序号 | 函数/方法 | 文件:行号 | 一句话作用 | 关键参数（名字=默认值） | 返回 |
|---|---|---|---|---|--- |
| 1 | `_one_hot` | `rl/plan_space.py:89` | 名字→one-hot 向量，未命中取 default_idx | `name, choices, default_idx=0, dtype=np.float32` | `np.ndarray` |
| 2 | `PlanToken` | `rl/plan_space.py:99` | 战术意图 token：旧 21 维字段+v1 追加字段 | `—` | `—` |
| 3 | `PlanToken.intent`<br>（内嵌于 `PlanToken`） | `rl/plan_space.py:117` | 便捷构造：指定 macro_intent 与 region | `cls, name, region='own_center', **kw` | `'PlanToken'` |
| 4 | `PlanToken.hold_slots`<br>（内嵌于 `PlanToken`） | `rl/plan_space.py:120` | hold_mask 命中的槽位列表（1..4） | `self` | `List[int]` |
| 5 | `PlanToken.to_vector`<br>（内嵌于 `PlanToken`） | `rl/plan_space.py:124` | 离散化为定长向量，前 21 维兼容旧布局 | `self` | `np.ndarray` |
| 6 | `PlanToken.from_old_layout`<br>（内嵌于 `PlanToken`） | `rl/plan_space.py:163` | 旧 21 维向量反解回 token（新字段取默认） | `cls, old_vec: np.ndarray` | `'PlanToken'` |
| 7 | `PlanToken.zeros`<br>（内嵌于 `PlanToken`） | `rl/plan_space.py:182` | 返回全默认 PlanToken（zeros） | `cls` | `'PlanToken'` |

##### B.8.26 `rl/ppo.py`（18 个符号）

| 序号 | 函数/方法 | 文件:行号 | 一句话作用 | 关键参数（名字=默认值） | 返回 |
|---|---|---|---|---|--- |
| 1 | `ReturnScaler` | `rl/ppo.py:41` | 回报尺度的运行统计（Welford 在线均值/方差），用于价值损失的量纲对齐。 | `—` | `—` |
| 2 | `ReturnScaler.__init__`<br>（内嵌于 `ReturnScaler`） | `rl/ppo.py:48` | 记录 eps 并初始化 Welford 均值/方差统计 | `self, eps=1e-06` | `—` |
| 3 | `ReturnScaler.update`<br>（内嵌于 `ReturnScaler`） | `rl/ppo.py:54` | 批量喂入一批回报（list/np.ndarray）。 | `self, values` | `—` |
| 4 | `ReturnScaler.var`<br>（内嵌于 `ReturnScaler`） | `rl/ppo.py:62` | 样本方差 m2/(count-1) | `self` | `—` |
| 5 | `ReturnScaler.std`<br>（内嵌于 `ReturnScaler`） | `rl/ppo.py:65` | 方差开方（负值截零） | `self` | `—` |
| 6 | `ReturnScaler.scale`<br>（内嵌于 `ReturnScaler`） | `rl/ppo.py:68` | 价值损失的除数 s（v_loss /= s²）。统计未建立/方差过小 → 0.0 表示不缩放。 | `self` | `—` |
| 7 | `ReturnScaler.to_dict`<br>（内嵌于 `ReturnScaler`） | `rl/ppo.py:73` | 导出 count/mean/m2 供 run_state 落盘 | `self` | `—` |
| 8 | `ReturnScaler.from_dict`<br>（内嵌于 `ReturnScaler`） | `rl/ppo.py:77` | 由字典恢复统计（缺键取默认） | `cls, d` | `—` |
| 9 | `PPOTrainer` | `rl/ppo.py:86` | PPO 更新器：GAE/裁剪/更新预算/EV 与梯度诊断 | `—` | `—` |
| 10 | `PPOTrainer.__init__`<br>（内嵌于 `PPOTrainer`） | `rl/ppo.py:87` | value_norm: none=旧行为（价值损失不缩放）/ running=按回报运行 std 缩放。 | `self, policy, lr=0.0003, gamma=0.99, gae_lambda=0.95, clip=0.2, vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, adv_norm='batch', value_norm='none', diagnose_every=0, ret_scaler=None, n_epochs=1, minibatch_size=0, shuffle=False, seed=12345` | `—` |
| 11 | `PPOTrainer.explained_variance`<br>（内嵌于 `PPOTrainer`） | `rl/ppo.py:145` | `EV = 1 − MSE(v, R) / Var(R)`：critic 对回报的解释力。 | `values, returns` | `—` |
| 12 | `PPOTrainer.compute_gae`<br>（内嵌于 `PPOTrainer`） | `rl/ppo.py:168` | values: 每步 value。返回 advantages 与 returns。 | `rewards, values, dones, gamma=0.99, lam=0.95, truncated=None, last_value=0.0` | `—` |
| 13 | `PPOTrainer.update`<br>（内嵌于 `PPOTrainer`） | `rl/ppo.py:194` | transitions: list of dicts {obs, belief, plan, bundle, old_logprob, | `self, transitions, ent_coef=None, adv_norm=None, adv_alt=None` | `—` |
| 14 | `PPOTrainer._loss_pass`<br>（内嵌于 `PPOTrainer`） | `rl/ppo.py:361` | 在 `idx` 指定的样本子集上算一次 loss 及各诊断量。 | `self, transitions, idx, advs, rets_np, coef, v_scale, reduction='mean'` | `—` |
| 15 | `PPOTrainer._apply_grad`<br>（内嵌于 `PPOTrainer`） | `rl/ppo.py:409` | backward + 梯度诊断 + clip + `opt.step()`（一次梯度步）。 | `self, pack, diag_on=False, pack_alt=None` | `—` |
| 16 | `PPOTrainer._apply_grad._norm`<br>（内嵌于 `PPOTrainer._apply_grad`） | `rl/ppo.py:433` | 对梯度列表求总 L2 范数 | `gs` | `—` |
| 17 | `PPOTrainer._plan_batches`<br>（内嵌于 `PPOTrainer`） | `rl/ppo.py:467` | 一轮内的样本顺序切分（`shuffle` 时打乱；小批不足则退化为整批）。 | `self, n` | `—` |
| 18 | `PPOTrainer._update_epochs`<br>（内嵌于 `PPOTrainer`） | `rl/ppo.py:477` | F′ 主循环：`n_epochs` 轮 × 小批 ×（可选）打乱，每小批一次 `opt.step()`。 | `self, transitions, advs, rets_np, coef, v_scale, diag_on, advs_alt=None` | `—` |

##### B.8.27 `rl/prophet.py`（28 个符号）

| 序号 | 函数/方法 | 文件:行号 | 一句话作用 | 关键参数（名字=默认值） | 返回 |
|---|---|---|---|---|--- |
| 1 | `_is_tower` | `rl/prophet.py:45` | 卡名是否含 Tower | `name: str` | `bool` |
| 2 | `_region_from_intent` | `rl/prophet.py:49` | 旧 8 意图→focus_region 映射 | `intent: str` | `str` |
| 3 | `_units` | `rl/prophet.py:63` | 特权状态里的非塔且属卡表部署实体 | `fs, player_id: int` | `—` |
| 4 | `_mean_x` | `rl/prophet.py:73` | 单位平均 x（空则 9.0） | `units` | `float` |
| 5 | `_threat_and_my_pressure` | `rl/prophet.py:78` | 特权状态威胁/我方压力统计（滤静态塔） | `fs` | `—` |
| 6 | `_closest_enemy` | `rl/prophet.py:92` | 最接近我方塔的敌方部署单位 | `fs` | `—` |
| 7 | `_pressing_enemy` | `rl/prophet.py:102` | 是否有敌单位进入我方半场/桥头 | `fs` | `bool` |
| 8 | `_side` | `rl/prophet.py:107` | x<LANE_SPLIT_X 为 left，否则 right | `x: float` | `str` |
| 9 | `_own_region` | `rl/prophet.py:111` | x→own_left/own_right | `x: float` | `str` |
| 10 | `_enemy_region` | `rl/prophet.py:115` | x→enemy_left/enemy_right | `x: float` | `str` |
| 11 | `_opposite_enemy_region` | `rl/prophet.py:119` | 敌方重心在 x→建议进攻另一路 | `x: float` | `str` |
| 12 | `_hand_slot` | `rl/prophet.py:124` | 卡在 cycle[:4] 的槽位 1..4，无则 None | `cycle, card_name` | `—` |
| 13 | `_spell_threat_in` | `rl/prophet.py:128` | cycle 前 depth 张里首张强法术→威胁枚举 | `cycle, depth: int=6` | `—` |
| 14 | `_pick_suggested` | `rl/prophet.py:140` | 按 intent 从可出手牌启发式选槽位 | `fs, intent: str` | `—` |
| 15 | `ProphetPlanner` | `rl/prophet.py:163` | 特权状态启发式先知（输出 PlanToken） | `—` | `—` |
| 16 | `ProphetPlanner._soft_control`<br>（内嵌于 `ProphetPlanner`） | `rl/prophet.py:168` | 软控：压境威胁→冰冻/藤蔓/龙卷 | `self, fs` | `—` |
| 17 | `ProphetPlanner._spell_trade`<br>（内嵌于 `ProphetPlanner`） | `rl/prophet.py:183` | 法术解高费后排（血牛放行给拉扯） | `self, fs` | `—` |
| 18 | `ProphetPlanner._protect_backline`<br>（内嵌于 `ProphetPlanner`） | `rl/prophet.py:206` | 保后排：吸仇恨+对手手牌预判版 | `self, fs` | `—` |
| 19 | `ProphetPlanner._pull`<br>（内嵌于 `ProphetPlanner`） | `rl/prophet.py:254` | 拉扯：血牛在桥头带→便宜单位/建筑 | `self, fs` | `—` |
| 20 | `ProphetPlanner._punish`<br>（内嵌于 `ProphetPlanner`） | `rl/prophet.py:279` | 趁虚另一路（直读特权对手圣水） | `self, fs` | `—` |
| 21 | `ProphetPlanner._push_commit`<br>（内嵌于 `ProphetPlanner`） | `rl/prophet.py:302` | 坦克推进中→跟 3-6 费角色 | `self, fs` | `—` |
| 22 | `ProphetPlanner._spell_finish`<br>（内嵌于 `ProphetPlanner`） | `rl/prophet.py:325` | 后期低血公主塔→磨塔法术 | `self, fs` | `—` |
| 23 | `ProphetPlanner._setup_wait`<br>（内嵌于 `ProphetPlanner`） | `rl/prophet.py:348` | 无压力+手牌有坦克→沉底蓄力 | `self, fs` | `—` |
| 24 | `ProphetPlanner._king_activate`<br>（内嵌于 `ProphetPlanner`） | `rl/prophet.py:364` | 公主塔残血+重单位→激活国王塔 | `self, fs` | `—` |
| 25 | `ProphetPlanner._anti_spell`<br>（内嵌于 `ProphetPlanner`） | `rl/prophet.py:392` | 直读对手牌序法术→防溅射站位 | `self, fs` | `—` |
| 26 | `ProphetPlanner._save_ace`<br>（内嵌于 `ProphetPlanner`） | `rl/prophet.py:410` | 藏终结卡 hold_mask（最强一波解除） | `self, fs` | `—` |
| 27 | `ProphetPlanner._cycle_small`<br>（内嵌于 `ProphetPlanner`） | `rl/prophet.py:447` | 无压力+1-2 费小牌→小费过牌 | `self, fs` | `—` |
| 28 | `ProphetPlanner.plan`<br>（内嵌于 `ProphetPlanner`） | `rl/prophet.py:464` | 主入口：优先链依次检测，未命中回退旧 8 意图 | `self, full_state: dict` | `PlanToken` |

##### B.8.28 `rl/replay.py`（11 个符号）

| 序号 | 函数/方法 | 文件:行号 | 一句话作用 | 关键参数（名字=默认值） | 返回 |
|---|---|---|---|---|--- |
| 1 | `EpisodeReplay` | `rl/replay.py:22` | 单局 replay 容器：逐步记录含隐藏标签的监督数据 | `—` | `—` |
| 2 | `EpisodeReplay.start`<br>（内嵌于 `EpisodeReplay`） | `rl/replay.py:28` | 清空步列表并标记为采集中 | `self` | `—` |
| 3 | `EpisodeReplay.record_step`<br>（内嵌于 `EpisodeReplay`） | `rl/replay.py:32` | 记录一步（obs/动作/奖励/对手出牌/可选 hidden） | `self, obs, bundle, reward, info, hidden=None` | `—` |
| 4 | `EpisodeReplay.end`<br>（内嵌于 `EpisodeReplay`） | `rl/replay.py:47` | 结束采集并返回统一 schema 的字典 | `self` | `dict` |
| 5 | `EpisodeReplay.save`<br>（内嵌于 `EpisodeReplay`） | `rl/replay.py:51` | 把 end() 结果 pickle 落盘 | `self, path` | `—` |
| 6 | `EpisodeReplay.load`<br>（内嵌于 `EpisodeReplay`） | `rl/replay.py:57` | 从 pickle 读取 replay，兼容 v1 裸容器 | `cls, path` | `'EpisodeReplay'` |
| 7 | `EpisodeReplay.to_belief_dataset`<br>（内嵌于 `EpisodeReplay`） | `rl/replay.py:69` | 转成信念监督样本 [(obs, 对手出牌, hidden)] | `self` | `—` |
| 8 | `battle_snapshot` | `rl/replay.py:85` | 把一步压成轻量帧（时间/动作/奖励/塔血/实体列表） | `battle, bundle, reward, info` | `—` |
| 9 | `battle_snapshot._kind`<br>（内嵌于 `battle_snapshot`） | `rl/replay.py:94` | 内层辅助：按实体类型返回建筑/法术/投射物/部队 | `e` | `—` |
| 10 | `save_league_replays` | `rl/replay.py:128` | 把联赛录像集合 pickle 落盘（带 schema 版本） | `games, path` | `—` |
| 11 | `load_league_replays` | `rl/replay.py:136` | 读取联赛录像集合，兼容旧容器 | `path` | `—` |

##### B.8.29 `rl/run_league.py`（50 个符号）

| 序号 | 函数/方法 | 文件:行号 | 一句话作用 | 关键参数（名字=默认值） | 返回 |
|---|---|---|---|---|--- |
| 1 | `_cuda_hint` | `rl/run_league.py:70` | CUDA 不可用时返回诊断提示，区分 CPU 构建与驱动问题 | `—` | `str` |
| 2 | `resolve_device` | `rl/run_league.py:94` | 解析 --device 为实际设备（auto 按 CUDA 可用性） | `device: str` | `str` |
| 3 | `LeagueGameRecorder` | `rl/run_league.py:108` | 逐局录像采集器：把每个决策步压成轻量帧供联赛回放 | `—` | `—` |
| 4 | `LeagueGameRecorder.__init__`<br>（内嵌于 `LeagueGameRecorder`） | `rl/run_league.py:111` | 初始化录像元信息（对阵双方/先手/步数/卡组）与帧缓冲 | `self, a_id, b_id, side0, max_steps, steps=None, decks=None` | `—` |
| 5 | `LeagueGameRecorder.set_decks`<br>（内嵌于 `LeagueGameRecorder`） | `rl/run_league.py:124` | 记录本局双方实际卡组（reset 后才可知） | `self, deck0, deck1` | `—` |
| 6 | `LeagueGameRecorder.record`<br>（内嵌于 `LeagueGameRecorder`） | `rl/run_league.py:128` | 把单步战斗快照与信息追加为一帧录像 | `self, env, bundle, reward, info, cards=None` | `—` |
| 7 | `LeagueGameRecorder.done`<br>（内嵌于 `LeagueGameRecorder`） | `rl/run_league.py:136` | 结束录像并返回 meta/winner/frames 结构 | `self, winner` | `—` |
| 8 | `towers_hp` | `rl/run_league.py:149` | 双方三塔血量合计（僵局检测用，塔血只降不升） | `env` | `—` |
| 9 | `_min_alive_tower_pct` | `rl/run_league.py:157` | 该玩家存活塔中最低血量百分比，取不到实体返回 None | `battle, player_id` | `—` |
| 10 | `timeout_winner` | `rl/run_league.py:179` | 截断/早停时的结算兜底：先比皇冠再比最低塔血百分比 | `battle, hp_tiebreak=None` | `—` |
| 11 | `settle_stall_from_counts` | `rl/run_league.py:213` | 早停局低置信裁定降噪：塔血差小于 margin 记平局 | `lost0, lost1, min_pct0, min_pct1, margin=0.05` | `—` |
| 12 | `settle_stall` | `rl/run_league.py:240` | 从 battle 取皇冠与最低塔血后调用裁定纯函数 | `battle, margin=0.05` | `—` |
| 13 | `_stall_probe` | `rl/run_league.py:252` | 僵局探针：连续 STALL_LIMIT 次零塔血变化则返回早停 | `env, last_hp, stall_count` | `—` |
| 14 | `_run_side0` | `rl/run_league.py:268` | 让 policy 以 player-0 打完整对局并返回胜者 | `env, policy, belief, bp, max_steps=300, recorder=None, reset_seed=None` | `—` |
| 15 | `_run_side0_scripted` | `rl/run_league.py:313` | 脚本策略版对局循环：直接 play 出牌，含僵局早停与录像 | `env, policy, max_steps=300, recorder=None, reset_seed=None` | `—` |
| 16 | `_bundle_cards` | `rl/run_league.py:343` | 从 bundle 的部署子动作反查本次打出的卡名列表 | `bundle, obs` | `—` |
| 17 | `_deck_factory_of` | `rl/run_league.py:353` | 取脚本策略的每局换卡组工厂，其他策略返回 None | `policy` | `—` |
| 18 | `_make_opp` | `rl/run_league.py:366` | 把对手策略包成 player-1 对手（脚本或学习型） | `policy, env, deck` | `—` |
| 19 | `_prepare_env` | `rl/run_league.py:380` | reset 前装配 env 的卡组工厂与对手，防跨 pair 泄漏 | `env, side0_pol, side1_pol, deck0_prior=None` | `—` |
| 20 | `_pair_seed_offset` | `rl/run_league.py:393` | 按 pair 索引与双方 id 哈希生成独立种子偏移 | `idx, a, b` | `—` |
| 21 | `play_pair` | `rl/run_league.py:406` | a/b 换边打 n 局，逐局更新 Elo 与 PFSP，可选采集录像 | `league, a_id, a_pol, b_id, b_pol, n_games, max_steps, seed, record=False` | `—` |
| 22 | `_round_estimates` | `rl/run_league.py:448` | 轮内 BT-lite 聚合 Elo 估计（Laplace 平滑）与 SE | `pair_results` | `—` |
| 23 | `eval_round_robin` | `rl/run_league.py:476` | 全轮转评估：带策略 agent 两两换边对战并记录 Elo 曲线 | `league, n_games, max_steps, seed, step, only_vs_main=False, record=False` | `—` |
| 24 | `evaluate_league` | `rl/run_league.py:510` | eval 模式入口：加载多个 ckpt 轮转对战并打印 Elo 表 | `policies, kinds, n_games, seed, hidden_dim, max_steps=600, device='auto'` | `—` |
| 25 | `build_five_agents` | `rl/run_league.py:539` | 注册 5 个卡组模型，无三分类数据集时退回旧五槽位 | `league, main, seed, decks_path=None` | `—` |
| 26 | `_make_env` | `rl/run_league.py:579` | 按配置新建带奖励权重与卡等的 RLEnv | `cfg, seed` | `—` |
| 27 | `_make_trainer` | `rl/run_league.py:584` | 按配置用 main 新建 PPOTrainer | `main, cfg` | `—` |
| 28 | `_load_run_state` | `rl/run_league.py:590` | 读取 run_state.json，缺失或解析失败返回 None | `cfg` | `—` |
| 29 | `_restore` | `rl/run_league.py:600` | 恢复联赛与训练进度，返回 (start_step, ppo, main) | `league, cfg, main, device, resume` | `—` |
| 30 | `_save_snapshot` | `rl/run_league.py:627` | 保存 main/优化器检查点、联赛状态与 run_state | `league, main, ppo, cfg, step, device` | `—` |
| 31 | `_eval_and_snapshot` | `rl/run_league.py:642` | 跑一轮评估、保存联赛录像与快照并打印 Elo 表 | `league, main, ppo, cfg, step, device, record_replays` | `—` |
| 32 | `_sample_opponent_for` | `rl/run_league.py:655` | 从联赛 PFSP 采样对手并装配到 env 供训练收集 | `league, env, seed` | `—` |
| 33 | `_build_league` | `rl/run_league.py:672` | 公共前缀：建 env/main/league 等并恢复进度 | `cfg, device, resume` | `—` |
| 34 | `_run_single` | `rl/run_league.py:697` | 单 env 主循环（旧行为，默认与兜底路径） | `cfg: TrainConfig, resume=False, record_replays=True` | `—` |
| 35 | `_run_single.sample_training_opponent`<br>（内嵌于 `_run_single`） | `rl/run_league.py:711` | 内层闭包：为训练 env 重采一个 PFSP 对手 | `—` | `—` |
| 36 | `_run_single.eval_and_snapshot`<br>（内嵌于 `_run_single`） | `rl/run_league.py:714` | 内层闭包：零参包装 _eval_and_snapshot 触发评估 | `step` | `—` |
| 37 | `_run_vec` | `rl/run_league.py:803` | 单进程多环境主循环（批量推理 + 批量 PPO 更新） | `cfg: TrainConfig, resume=False, record_replays=True` | `—` |
| 38 | `_run_vec.eval_and_snapshot`<br>（内嵌于 `_run_vec`） | `rl/run_league.py:819` | 内层闭包：触发一轮评估与快照 | `step` | `—` |
| 39 | `_run_vec.new_buf`<br>（内嵌于 `_run_vec`） | `rl/run_league.py:837` | 内层闭包：新建每环境轨迹缓冲字典 | `—` | `—` |
| 40 | `_run_mp` | `rl/run_league.py:942` | 跨进程并行主循环：worker 推演、主进程批量 GPU 推理 | `cfg: TrainConfig, resume=False, record_replays=True` | `—` |
| 41 | `_run_mp.eval_and_snapshot`<br>（内嵌于 `_run_mp`） | `rl/run_league.py:963` | 内层闭包：触发一轮评估与快照 | `step` | `—` |
| 42 | `_run_mp.spec_for`<br>（内嵌于 `_run_mp`） | `rl/run_league.py:969` | 内层闭包：把对手策略转成 worker 可用的 spec 字典 | `pol` | `—` |
| 43 | `_run_mp.next_spec`<br>（内嵌于 `_run_mp`） | `rl/run_league.py:978` | 内层闭包：采样对手并取其 worker spec | `—` | `—` |
| 44 | `_run_mp.recv`<br>（内嵌于 `_run_mp`） | `rl/run_league.py:999` | 内层闭包：带超时收 worker 消息并校验类型 | `i, expect` | `—` |
| 45 | `_run_mp.mask_fn`<br>（内嵌于 `_run_mp`） | `rl/run_league.py:1014` | 内层闭包：向 worker 请求动作掩码并回收 | `i, partial` | `—` |
| 46 | `_run_mp.get_masks_batch`<br>（内嵌于 `_run_mp`） | `rl/run_league.py:1018` | 内层闭包：并发发出全部掩码请求后统一回收 | `partials` | `—` |
| 47 | `_run_mp.new_buf`<br>（内嵌于 `_run_mp`） | `rl/run_league.py:1027` | 内层闭包：新建含 winner 字段的每环境轨迹缓冲 | `—` | `—` |
| 48 | `run_league` | `rl/run_league.py:1143` | 联赛主循环入口：按 n_envs 与 parallel 选择三条训练路径 | `cfg: TrainConfig, resume=False, record_replays=True` | `—` |
| 49 | `_force_utf8_stdout` | `rl/run_league.py:1152` | 把 stdout/stderr 切到 UTF-8，防中文日志编码崩溃 | `—` | `—` |
| 50 | `main` | `rl/run_league.py:1168` | 命令行入口：解析参数、组装配置并分发各训练/评估模式 | `—` | `—` |

##### B.8.30 `rl/selftest.py`（161 个符号）

| 序号 | 函数/方法 | 文件:行号 | 一句话作用 | 关键参数（名字=默认值） | 返回 |
|---|---|---|---|---|--- |
| 1 | `test_action_bundle_same_tick` | `rl/selftest.py:37` | 自检：同刻多卡正确打出并原子拒绝非法包 | `—` | `—` |
| 2 | `test_action_bundle_ability` | `rl/selftest.py:66` | 自检：出牌与英雄技能同 tick 生效、无就绪英雄整包拒绝 | `—` | `—` |
| 3 | `test_bayes_filter` | `rl/selftest.py:98` | 自检：贝叶斯粒子滤波后验收敛、下一张牌进 top3 | `—` | `—` |
| 4 | `_make_policy_and_tokens` | `rl/selftest.py:122` | 辅助：构造信念对象、token、零 plan 与 PLAN_DIM | `env, seed=0` | `—` |
| 5 | `test_hidden_replay_consistency` | `rl/selftest.py:130` | 自检 P0-1：用记录 hidden 重放，新 logprob≈旧 | `—` | `—` |
| 6 | `test_entropy_positive_and_sign` | `rl/selftest.py:153` | 自检 P0-2：熵非负且正优势下已选动作概率上升 | `—` | `—` |
| 7 | `test_mask_validate_invariant_both_sides` | `rl/selftest.py:180` | 自检 P0-3/P0-4：掩码合法格必过 validate、双侧缓存隔离 | `—` | `—` |
| 8 | `test_heuristic_opponent_actually_plays` | `rl/selftest.py:223` | 自检 P0-3/P0-4：heuristic 对手出牌率不再极低 | `—` | `—` |
| 9 | `test_exploiter_loads_main_checkpoint` | `rl/selftest.py:245` | 自检 P0-5：checkpoint 元数据与旧格式回退均可加载 | `—` | `—` |
| 10 | `test_belief_survives_ability` | `rl/selftest.py:267` | 自检 P0-6：技能哨兵不进信念、不崩溃不重置先验 | `—` | `—` |
| 11 | `test_belief_multi_card_update` | `rl/selftest.py:289` | 自检 P1-5：同 tick 出两张牌时信念对两张都做排除 | `—` | `—` |
| 12 | `test_register_checkpoint_isolated` | `rl/selftest.py:304` | 自检 P1-9：快照注册后 historical 参数与 main 解耦 | `—` | `—` |
| 13 | `test_bundle_cap_no_crash` | `rl/selftest.py:328` | 自检 P1-18：低费局面稳定产出不超 K_MAX 的 bundle | `—` | `—` |
| 14 | `test_replay_roundtrip` | `rl/selftest.py:356` | 自检 P1-21：录像 save→load→监督数据集非空且 hidden 对齐 | `—` | `—` |
| 15 | `test_prophet_empty_board_not_defend` | `rl/selftest.py:381` | 自检 P1-4：先知空场开局意图不是 defend_* | `—` | `—` |
| 16 | `_mk_env` | `rl/selftest.py:393` | 辅助：创建无对手的 RLEnv（seed=0） | `—` | `—` |
| 17 | `test_random_deck_model` | `rl/selftest.py:398` | 自检：随机卡组每局重采样 8 卡且脚本动作合法 | `—` | `—` |
| 18 | `test_league_elo_history` | `rl/selftest.py:426` | 自检：联赛 Elo 历史记录与 save/load 往返 | `—` | `—` |
| 19 | `test_winrate_streams_independent` | `rl/selftest.py:452` | 自检：不同 pair 的 PFSP 胜率 EMA 流独立且双向互补 | `—` | `—` |
| 20 | `test_winrate_streams_independent.ref_ema`<br>（内嵌于 `test_winrate_streams_independent`） | `rl/selftest.py:475` | 辅助：按 α=0.05 独立重算某比分序列的 EMA 参考值 | `seq` | `—` |
| 21 | `test_elo_eval_granularity` | `rl/selftest.py:524` | 自检：评估噪声地板 SE、轮内聚合估计与误差棒全链路 | `—` | `—` |
| 22 | `test_elo_eval_granularity.noise_prob`<br>（内嵌于 `test_elo_eval_granularity`） | `rl/selftest.py:541` | 辅助：纯噪声下两轮 Elo 差超 delta 的正态近似概率 | `se_per_round, delta=100.0` | `—` |
| 23 | `test_classified_decks` | `rl/selftest.py:598` | 自检：三分类 200 副卡组的数量、卡名归一化与随机抽取 | `—` | `—` |
| 24 | `test_league_training_loop` | `rl/selftest.py:623` | 自检：联赛主循环跑完对局并正常重置（防解包 bug） | `—` | `—` |
| 25 | `test_config_reward_weights` | `rl/selftest.py:640` | 自检：命名配置解析互不污染、塔血不对称与权重注入 | `—` | `—` |
| 26 | `test_model_reward_overrides` | `rl/selftest.py:682` | 自检：按流派覆盖费差权重，main/all/random 同基线 | `—` | `—` |
| 27 | `test_reward_economy_preset` | `rl/selftest.py:712` | 自检：economy/standard 默认开归一化与费差，JSON 往返 | `—` | `—` |
| 28 | `test_reward_economy_level_invariance` | `rl/selftest.py:745` | 自检：塔损按塔血百分比归一化后跨等级奖励一致 | `—` | `—` |
| 29 | `test_reward_economy_level_invariance.r`<br>（内嵌于 `test_reward_economy_level_invariance`） | `rl/selftest.py:758` | 辅助：给定权重与总塔血上限，复算固定比例塔损奖励 | `weights, total_max, event_frac=0.05` | `—` |
| 30 | `test_reward_economy_elixir_diff` | `rl/selftest.py:782` | 自检：费差项显式给圣水定价且闭环累计归零 | `—` | `—` |
| 31 | `test_reward_economy_trade_pricing` | `rl/selftest.py:820` | 自检：费差与塔血的真实 trade 定价（1 圣水≈500 血） | `—` | `—` |
| 32 | `test_reward_economy_trade_pricing.r`<br>（内嵌于 `test_reward_economy_trade_pricing`） | `rl/selftest.py:835` | 辅助：以固定 base 参数复算带额外覆盖项的奖励 | `weights, **kw` | `—` |
| 33 | `test_rlenv_card_level` | `rl/selftest.py:872` | 自检：RLEnv 支持 11–16 级卡牌，reset 同步真实塔血 | `—` | `—` |
| 34 | `test_tower_troop_hp_reference` | `rl/selftest.py:897` | 自检：塔血参考表与归一化对塔型/等级不变 | `—` | `—` |
| 35 | `test_tower_troop_hp_reference.r`<br>（内嵌于 `test_tower_troop_hp_reference`） | `rl/selftest.py:917` | 辅助：按某塔型真实总塔血复算固定比例塔损奖励 | `troop_hp, event_frac=0.05` | `—` |
| 36 | `test_tower_troop_hp_reference.absr`<br>（内嵌于 `test_tower_troop_hp_reference`） | `rl/selftest.py:938` | 辅助：按某塔型总血复算固定绝对伤害的奖励 | `troop_hp` | `—` |
| 37 | `test_league_resume` | `rl/selftest.py:956` | 自检：联赛断点续训从旧 step 续跑并刷新快照 | `—` | `—` |
| 38 | `test_league_replays` | `rl/selftest.py:983` | 自检：每评估周期录像采集、保存与回读 | `—` | `—` |
| 39 | `test_dashboard_replays` | `rl/selftest.py:1019` | 自检：仪表盘回放列表、单局帧、非法名防护与 demo 生成 | `—` | `—` |
| 40 | `test_dashboard_replays.frame`<br>（内嵌于 `test_dashboard_replays`） | `rl/selftest.py:1025` | 辅助：构造仪表盘测试用的单帧字典 | `t, t0, t1, entities, **kw` | `—` |
| 41 | `test_deck_pool_factory` | `rl/selftest.py:1099` | 自检：deck_pool 工厂逐局生效且跨 pair 清空 | `—` | `—` |
| 42 | `test_dashboard_card_stats` | `rl/selftest.py:1137` | 自检：卡牌统计的双侧归属、旧录像降级与汇总防护 | `—` | `—` |
| 43 | `test_dashboard_card_stats.frame`<br>（内嵌于 `test_dashboard_card_stats`） | `rl/selftest.py:1143` | 辅助：构造卡牌统计测试用的默认帧字典 | `**kw` | `—` |
| 44 | `test_battle_clone_fix` | `rl/selftest.py:1203` | 自检：克隆法术克隆冰法不再触发 battle_state 崩溃 | `—` | `—` |
| 45 | `test_cuda_device_support` | `rl/selftest.py:1223` | 自检：cpu 必跑、cuda 可用时额外验证 act/evaluate/PPO | `—` | `—` |
| 46 | `test_belief_follower_ppo_league` | `rl/selftest.py:1258` | 自检：跟随者动作合法、PPO 收敛、两个规划器与联赛 | `—` | `—` |
| 47 | `test_parallel_batch_equivalence` | `rl/selftest.py:1315` | 自检：批量 act/evaluate 与单条路径逐位等价 | `—` | `—` |
| 48 | `test_parallel_training_loop` | `rl/selftest.py:1355` | 自检：n_envs=2 parallel=proc 训练主循环跑通并落盘 | `—` | `—` |
| 49 | `test_mp_training_loop` | `rl/selftest.py:1372` | 自检：n_envs=2 parallel=mp 跨进程训练主循环跑通并落盘 | `—` | `—` |
| 50 | `_intents` | `rl/selftest.py:1389` | 辅助：返回 plan_space.MACRO_INTENTS 合法意图表 | `—` | `—` |
| 51 | `test_flow_league_smoke` | `rl/selftest.py:1394` | 自检：flow 全配对规模、mini 池双侧训练与 6 模型落盘 | `—` | `—` |
| 52 | `test_flow_league_smoke.mk_decks`<br>（内嵌于 `test_flow_league_smoke`） | `rl/selftest.py:1405` | 辅助：造 n 副同卡表的假卡组 | `n, arch` | `—` |
| 53 | `test_ablation_recorded` | `rl/selftest.py:1447` | 自检：belief/plan 四变体消融 + delta/z 判定 + JSON/CSV 落盘 | `—` | `—` |
| 54 | `test_flow_sweep_smoke` | `rl/selftest.py:1483` | 自检：flow-sweep 缩小池通路与 summary.json/csv 落盘 | `—` | `—` |
| 55 | `test_flow_sweep_smoke.mk_decks`<br>（内嵌于 `test_flow_sweep_smoke`） | `rl/selftest.py:1489` | 辅助：造 n 副同卡表的假卡组 | `n, arch` | `—` |
| 56 | `test_flow_resume` | `rl/selftest.py:1523` | 自检：flow 断点续练跳过已完成对、总局数等于全量 | `—` | `—` |
| 57 | `test_flow_resume.mk_decks`<br>（内嵌于 `test_flow_resume`） | `rl/selftest.py:1529` | 辅助：造 n 副同卡表的假卡组 | `n, arch` | `—` |
| 58 | `test_solo_mode_smoke` | `rl/selftest.py:1559` | 自检：solo 落盘且 PPO 批不超环境步（防缓冲区泄漏） | `—` | `—` |
| 59 | `test_human_play_session` | `rl/selftest.py:1599` | 自检：人机对战落盘 EpisodeReplay/BC 并可导出训练 | `—` | `—` |
| 60 | `test_solo_resume` | `rl/selftest.py:1632` | 自检：solo 断点续练恢复权重/优化器/曲线且不重复评估 | `—` | `—` |
| 61 | `test_stall_probe` | `rl/selftest.py:1660` | 自检：连续零塔损触发僵局早停、塔损重置计数 | `—` | `—` |
| 62 | `test_stall_probe.make_fake`<br>（内嵌于 `test_stall_probe`） | `rl/selftest.py:1664` | 辅助：造只含六项塔血的最小假 env | `hps` | `—` |
| 63 | `test_stall_probe.make_fake._P`<br>（内嵌于 `test_stall_probe.make_fake`） | `rl/selftest.py:1665` | 假玩家：仅持有王塔与左右塔血量 | `—` | `—` |
| 64 | `test_stall_probe.make_fake._P.__init__`<br>（内嵌于 `test_stall_probe.make_fake._P`） | `rl/selftest.py:1666` | 假玩家初始化：记录三项塔血 | `self, k, l, r` | `—` |
| 65 | `test_stall_probe.make_fake._B`<br>（内嵌于 `test_stall_probe.make_fake`） | `rl/selftest.py:1670` | 假战斗：以两个假玩家承载塔血 | `—` | `—` |
| 66 | `test_stall_probe.make_fake._E`<br>（内嵌于 `test_stall_probe.make_fake`） | `rl/selftest.py:1672` | 假 env：暴露 battle 属性供探针读取 | `—` | `—` |
| 67 | `test_play_pair_env_reuse` | `rl/selftest.py:1691` | 自检：play_pair 复用单 env 换边多局并更新 Elo/PFSP | `—` | `—` |
| 68 | `test_eval_stall_early_stop` | `rl/selftest.py:1717` | 自检：双方都不部署时僵局早停远早于打满步数 | `—` | `—` |
| 69 | `test_eval_stall_early_stop.Idle`<br>（内嵌于 `test_eval_stall_early_stop`） | `rl/selftest.py:1727` | 假脚本对手：恒返回 noop 且不部署 | `—` | `—` |
| 70 | `test_eval_stall_early_stop.Idle.play`<br>（内嵌于 `test_eval_stall_early_stop.Idle`） | `rl/selftest.py:1728` | 恒返回 ActionBundle.noop() | `self, env, player_id` | `—` |
| 71 | `test_draw_penalty_as_loss` | `rl/selftest.py:1743` | 自检：引擎终局平局与僵局平局都按失败惩罚 | `—` | `—` |
| 72 | `test_draw_penalty_as_loss.Noop`<br>（内嵌于 `test_draw_penalty_as_loss`） | `rl/selftest.py:1780` | 恒 noop 的跟随者策略子类（用于确定性平局断言） | `—` | `—` |
| 73 | `test_draw_penalty_as_loss.Noop.act`<br>（内嵌于 `test_draw_penalty_as_loss.Noop`） | `rl/selftest.py:1782` | 恒返回 noop bundle 与零 logprob/value | `self, obs, belief_token, plan_token, get_mask, hidden=None, deterministic=False` | `—` |
| 74 | `test_reward_v2_ledger` | `rl/selftest.py:1806` | 自检：reward v2 资源账（部署不罚/份额注销/法术花费/双倍期） | `—` | `—` |
| 75 | `test_reward_v2_ledger.first_cell`<br>（内嵌于 `test_reward_v2_ledger`） | `rl/selftest.py:1825` | 辅助：取某槽位掩码中第一个合法格的整数坐标 | `slot` | `—` |
| 76 | `test_spell_empty_value_gate` | `rl/selftest.py:1875` | 自检 8h 空砸闸门：伤害法术无目标格 mask+validate 双拒 | `—` | `—` |
| 77 | `test_spell_empty_value_gate.enemy_in_radius`<br>（内嵌于 `test_spell_empty_value_gate`） | `rl/selftest.py:1895` | 辅助：判断某世界坐标溅射半径内是否有存活敌军 | `pos` | `—` |
| 78 | `test_spell_tower_ev_gate` | `rl/selftest.py:1934` | 自检 9h：前段纯砸公主塔 EV 不足被双拒、双倍期放行 | `—` | `—` |
| 79 | `test_no_solo_commit_without_lead` | `rl/selftest.py:1993` | 自检 8h 不裸下：无费差单卡高承诺单位双拒 | `—` | `—` |
| 80 | `test_no_solo_commit_without_lead.mini_mask_slot`<br>（内嵌于 `test_no_solo_commit_without_lead`） | `rl/selftest.py:2014` | 辅助：读 MiniPekka 槽位掩码是否可用 | `—` | `—` |
| 81 | `test_tank_backline_geometry` | `rl/selftest.py:2064` | 自检 8h 坦克后屯兵：后排须留攻击距离、坦克前非法 | `—` | `—` |
| 82 | `test_tank_backline_geometry.mk_battle`<br>（内嵌于 `test_tank_backline_geometry`） | `rl/selftest.py:2078` | 辅助：造固定卡组的双人战斗状态 | `—` | `—` |
| 83 | `test_tank_backline_geometry.spawn`<br>（内嵌于 `test_tank_backline_geometry`） | `rl/selftest.py:2083` | 辅助：在指定位置放置指定血量的部队 | `bs, pid, name, x, y, hp` | `—` |
| 84 | `test_tank_backline_geometry.cell_near`<br>（内嵌于 `test_tank_backline_geometry`） | `rl/selftest.py:2089` | 辅助：找最接近给定世界坐标的本地网格格点 | `bs, pid, wx, wy, tol=0.9` | `—` |
| 85 | `test_plan_v1_layout` | `rl/selftest.py:2150` | 自检：PlanToken v1 布局兼容、字段落位与旧 ckpt 补零加载 | `—` | `—` |
| 86 | `test_bp_new_intent_rules` | `rl/selftest.py:2223` | 自检：BeliefPlanner v1 十余种意图规则与守卫回退 | `—` | `—` |
| 87 | `test_bp_new_intent_rules.new_battle`<br>（内嵌于 `test_bp_new_intent_rules`） | `rl/selftest.py:2233` | 辅助：造固定卡组双方 10 费的战斗状态 | `—` | `—` |
| 88 | `test_bp_new_intent_rules.place`<br>（内嵌于 `test_bp_new_intent_rules`） | `rl/selftest.py:2237` | 辅助：部署卡牌并搬移到指定坐标 | `bs, pid, card, x, y` | `—` |
| 89 | `test_bp_new_intent_rules.set_hand`<br>（内嵌于 `test_bp_new_intent_rules`） | `rl/selftest.py:2247` | 辅助：为 P0 设定手牌与圣水 | `bs, cards` | `—` |
| 90 | `test_bp_new_intent_rules.belief`<br>（内嵌于 `test_bp_new_intent_rules`） | `rl/selftest.py:2319` | 辅助：构造指定圣水/手牌概率的 BeliefState | `elixir=5.0, probs=None` | `—` |
| 91 | `test_pp_new_intent_rules` | `rl/selftest.py:2401` | 自检：ProphetPlanner 特权意图规则与 bp 同链标签一致 | `—` | `—` |
| 92 | `test_pp_new_intent_rules.new_battle`<br>（内嵌于 `test_pp_new_intent_rules`） | `rl/selftest.py:2413` | 辅助：造固定卡组双方 10 费的战斗状态 | `—` | `—` |
| 93 | `test_pp_new_intent_rules.place`<br>（内嵌于 `test_pp_new_intent_rules`） | `rl/selftest.py:2417` | 辅助：部署卡牌并搬移到指定坐标 | `bs, pid, card, x, y` | `—` |
| 94 | `test_pp_new_intent_rules.set_hand`<br>（内嵌于 `test_pp_new_intent_rules`） | `rl/selftest.py:2427` | 辅助：为指定玩家设定手牌与圣水 | `bs, cards, pid=0` | `—` |
| 95 | `test_pp_new_intent_rules.pstate`<br>（内嵌于 `test_pp_new_intent_rules`） | `rl/selftest.py:2431` | 辅助：构造与 get_prophet_state 同构的特权摘要 | `bs` | `—` |
| 96 | `test_bayes_queue_lock` | `rl/selftest.py:2536` | 自检：信念 O(1) 队列锁定定理、异常重锁与确定性 | `—` | `—` |
| 97 | `test_eval_solo_parallel` | `rl/selftest.py:2616` | 自检：并行 eval_solo 与串行同种子结果一致 | `—` | `—` |
| 98 | `test_overtime_window` | `rl/selftest.py:2653` | 自检：180s 皇冠平进加时，到顶按最低塔血百分比裁决 | `—` | `—` |
| 99 | `test_overtime_window._P`<br>（内嵌于 `test_overtime_window`） | `rl/selftest.py:2665` | 假玩家：按整数皇冠数返回皇冠计数 | `—` | `—` |
| 100 | `test_overtime_window._P.__init__`<br>（内嵌于 `test_overtime_window._P`） | `rl/selftest.py:2666` | 假玩家初始化：记录皇冠数 | `self, crowns` | `—` |
| 101 | `test_overtime_window._P.get_crown_count`<br>（内嵌于 `test_overtime_window._P`） | `rl/selftest.py:2669` | 返回假玩家的皇冠数 | `self` | `—` |
| 102 | `test_overtime_window._B`<br>（内嵌于 `test_overtime_window`） | `rl/selftest.py:2672` | 假战场：持有时间、双方假玩家与终局标志 | `—` | `—` |
| 103 | `test_overtime_window._B.__init__`<br>（内嵌于 `test_overtime_window._B`） | `rl/selftest.py:2673` | 假战场初始化：时间/双方皇冠/终局标志 | `self, t, c0, c1, over=False` | `—` |
| 104 | `test_overtime_window._T`<br>（内嵌于 `test_overtime_window`） | `rl/selftest.py:2678` | 假塔：记录当前血、最大血与存活 | `—` | `—` |
| 105 | `test_overtime_window._T.__init__`<br>（内嵌于 `test_overtime_window._T`） | `rl/selftest.py:2679` | 假塔初始化：按 hp>0 判存活并造 data.max_hp | `self, hp, max_hp` | `—` |
| 106 | `test_overtime_window._BT`<br>（内嵌于 `test_overtime_window`） | `rl/selftest.py:2684` | 带塔实体的假战场（p0 ids 3/4/6，p1 1/2/5） | `—` | `—` |
| 107 | `test_overtime_window._BT.__init__`<br>（内嵌于 `test_overtime_window._BT`） | `rl/selftest.py:2686` | 假战场初始化：按 id 挂载双方假塔 | `self, t, c0, c1, towers0, towers1, over=False` | `—` |
| 108 | `test_tower_threat_calc` | `rl/selftest.py:2724` | 自检：塔伤威胁计算器空场为 0、无污染、确定且双向可用 | `—` | `—` |
| 109 | `test_tower_threat_calc.fresh`<br>（内嵌于 `test_tower_threat_calc`） | `rl/selftest.py:2736` | 辅助：造固定卡组 lv11 的双人战斗状态 | `—` | `—` |
| 110 | `test_simulate_exchange` | `rl/selftest.py:2776` | 自检：交换模拟器 none/script/fn 防守、非法部署与无污染 | `—` | `—` |
| 111 | `test_simulate_exchange.fresh`<br>（内嵌于 `test_simulate_exchange`） | `rl/selftest.py:2788` | 辅助：造固定卡组、指定圣水的战斗状态 | `elixir=10.0` | `—` |
| 112 | `test_simulate_exchange.opp_fn`<br>（内嵌于 `test_simulate_exchange`） | `rl/selftest.py:2817` | 注入式对手回调：首 tick 下 Knight 解场 | `sim, defender_id` | `—` |
| 113 | `test_spell_module` | `rl/selftest.py:2866` | 自检：法术档案、预测与引擎实测对账、best_cast 与限制 | `—` | `—` |
| 114 | `test_spell_module.fresh`<br>（内嵌于 `test_spell_module`） | `rl/selftest.py:2879` | 辅助：造固定卡组 lv11、双方 10 费的战斗状态 | `—` | `—` |
| 115 | `test_mcts_basic` | `rl/selftest.py:2970` | 自检：浅 MCTS 合法/无污染/确定、值函数量纲与预算控制 | `—` | `—` |
| 116 | `test_mcts_basic.fresh`<br>（内嵌于 `test_mcts_basic`） | `rl/selftest.py:2990` | 辅助：造指定手牌与圣水的战斗状态并同步塔血 | `elixir=5.0, hand=None` | `—` |
| 117 | `test_mcts_basic.snap`<br>（内嵌于 `test_mcts_basic`） | `rl/selftest.py:2998` | 辅助：取局面指纹（时间/圣水/实体数/手牌） | `bs` | `—` |
| 118 | `test_mcts_basic.bkey`<br>（内嵌于 `test_mcts_basic`） | `rl/selftest.py:3014` | 辅助：把 bundle 子动作序列转成可比较的 key | `b` | `—` |
| 119 | `test_mcts_defense_and_wait` | `rl/selftest.py:3068` | 自检：浅 MCTS 防守场景决策合法且耗时可控 | `—` | `—` |
| 120 | `test_opp_event_token` | `rl/selftest.py:3098` | 自检 9j B 层：对手事件通道维度、陈旧度与零拷贝兼容 | `—` | `—` |
| 121 | `test_crossed_river_defend_plan` | `rl/selftest.py:3151` | 自检：敌军过河触发 defend_* 与拦截落点提示 | `—` | `—` |
| 122 | `test_crossed_river_defend_plan.battle_moved`<br>（内嵌于 `test_crossed_river_defend_plan`） | `rl/selftest.py:3167` | 辅助：部署后把该单位搬到指定坐标 | `card, side_x, y_target` | `—` |
| 123 | `test_crossed_river_defend_plan.plan_of`<br>（内嵌于 `test_crossed_river_defend_plan`） | `rl/selftest.py:3178` | 辅助：用观测与信念状态调用 BeliefPlanner 出计划 | `b` | `—` |
| 124 | `test_death_damage_scaling` | `rl/selftest.py:3215` | 自检：亡语伤害等级缩放与冰人死亡减速圈 | `—` | `—` |
| 125 | `test_vines_snare_fl_duration` | `rl/selftest.py:3270` | 自检：Vines 束缚 2.0s 与首跳伤害即时结算 | `—` | `—` |
| 126 | `test_log_rolling_direction` | `rl/selftest.py:3301` | 自检：滚木/滚筒凭空出现且纯纵向滚动，Firecracker 不误伤 | `—` | `—` |
| 127 | `test_log_rolling_direction.trace`<br>（内嵌于 `test_log_rolling_direction`） | `rl/selftest.py:3315` | 辅助：部署后逐帧记录滚动弹坐标轨迹 | `name, pid, pos, nsteps=300` | `—` |
| 128 | `test_behavioral_metrics` | `rl/selftest.py:3370` | 自检：合成回放验证六类行为指标判定 | `—` | `—` |
| 129 | `test_behavioral_metrics.frame`<br>（内嵌于 `test_behavioral_metrics`） | `rl/selftest.py:3383` | 辅助：构造行为指标测试用帧 | `t, bundle, entities, opp_played=None, towers0=None, towers1=None, elixir0=5.0` | `—` |
| 130 | `test_behavioral_metrics.troop`<br>（内嵌于 `test_behavioral_metrics`） | `rl/selftest.py:3392` | 辅助：构造一条 troop 实体列表 | `x, y, pl` | `—` |
| 131 | `test_mk_spawn_damage_and_iw_slow_fl` | `rl/selftest.py:3428` | 自检：MK 落地溅射与 IceWizard 落地冰雾口径 | `—` | `—` |
| 132 | `test_tower_value_mult` | `rl/selftest.py:3490` | 自检：塔血凹形溢价曲线与王塔贬值闸门 | `—` | `—` |
| 133 | `test_reward_tower_premium` | `rl/selftest.py:3522` | 自检：塔血溢价接入 compute_reward 且双向对称 | `—` | `—` |
| 134 | `test_reward_tower_premium.r_hit_red`<br>（内嵌于 `test_reward_tower_premium`） | `rl/selftest.py:3545` | 辅助：复算打敌方某残血塔的奖励 | `dmg, start_frac` | `—` |
| 135 | `test_reward_tower_premium.r_hit_blue`<br>（内嵌于 `test_reward_tower_premium`） | `rl/selftest.py:3554` | 辅助：复算我方某残血塔挨打的奖励 | `dmg, start_frac` | `—` |
| 136 | `test_reward_tower_premium.r_hit_king`<br>（内嵌于 `test_reward_tower_premium`） | `rl/selftest.py:3576` | 辅助：复算打敌方王塔（公主塔存活可变）的奖励 | `dmg, alive_left, alive_right` | `—` |
| 137 | `test_reward_tower_premium_rlenv_flow` | `rl/selftest.py:3595` | 自检：塔血溢价经 RLEnv.step 生效且 MCTS 值同源 | `—` | `—` |
| 138 | `test_reward_tower_premium_rlenv_flow.env_total_loss_with`<br>（内嵌于 `test_reward_tower_premium_rlenv_flow`） | `rl/selftest.py:3609` | 辅助：造 env 并把敌方右塔置残后累计几帧奖励 | `start_frac` | `—` |
| 139 | `test_reward_tower_premium_rlenv_flow.mk_bs`<br>（内嵌于 `test_reward_tower_premium_rlenv_flow`） | `rl/selftest.py:3647` | 辅助：造战斗状态并按参数覆盖 P1 塔血 | `**tower_overrides` | `—` |
| 140 | `test_opponent_pool_mix` | `rl/selftest.py:3671` | 自检 9j A 层：对手池混合分布与 SelfDefender 反制 | `—` | `—` |
| 141 | `_tiny_rollout_transitions` | `rl/selftest.py:3746` | 辅助：构造 n 条优势正负交替的最小 transition | `pol, env, belief, tok, plan, n=4` | `—` |
| 142 | `test_value_channel_norm_and_gnorm_split` | `rl/selftest.py:3766` | 自检：价值通道量纲、原始 MSE 对账与梯度诊断 | `—` | `—` |
| 143 | `test_history_dedup_and_gates` | `rl/selftest.py:3857` | 自检 P0-2：history 按 step 去重与门禁相对基线 | `—` | `—` |
| 144 | `test_opponent_pool_mix_multi_dir` | `rl/selftest.py:3905` | 自检 P1-1：hist-seed-dir 补种与本目录优先 | `—` | `—` |
| 145 | `test_opponent_pool_rand_anchor` | `rl/selftest.py:3964` | 自检 E2：训练侧随机锚点槽与无 hist 退化归一化 | `—` | `—` |
| 146 | `test_pfsp_gate_and_dynamic_hist` | `rl/selftest.py:4086` | 自检 D1：PFSP 默认兼容、门禁降权与动态 hist 刷新 | `—` | `—` |
| 147 | `test_enc_layernorm_gru_vitality` | `rl/selftest.py:4186` | 自检：enc_ln/grid_ln 吸收量级漂移、GRU 保持活力 | `—` | `—` |
| 148 | `test_enc_layernorm_gru_vitality._enc_norm`<br>（内嵌于 `test_enc_layernorm_gru_vitality`） | `rl/selftest.py:4232` | 辅助：取策略对固定帧的 enc 输出范数 | `—` | `—` |
| 149 | `test_enc_layernorm_gru_vitality._comp_rms`<br>（内嵌于 `test_enc_layernorm_gru_vitality`） | `rl/selftest.py:4273` | 辅助：hook 抓 fused 并按分量算每元素 RMS | `p` | `—` |
| 150 | `test_enc_layernorm_gru_vitality._comp_rms._hk`<br>（内嵌于 `test_enc_layernorm_gru_vitality._comp_rms`） | `rl/selftest.py:4277` | forward hook：缓存 enc_fc 的输入张量 | `_m, inp, _o` | `—` |
| 151 | `test_value_bypass` | `rl/selftest.py:4319` | 自检 B'：value_bypass 通路语义、元数据往返与诊断口径 | `—` | `—` |
| 152 | `test_stall_settlement_margin` | `rl/selftest.py:4404` | 自检 C'：早停低置信裁定按 margin 降噪 | `—` | `—` |
| 153 | `test_value_independent_encoder` | `rl/selftest.py:4437` | 自检 E'：独立价值编码器加 MLP 头通路与元数据 | `—` | `—` |
| 154 | `test_ppo_multi_epoch_minibatch` | `rl/selftest.py:4530` | 自检 F'：多轮小批 PPO 预算、划分可复现与 EV 口径 | `—` | `—` |
| 155 | `test_solo_rand_anchor` | `rl/selftest.py:4648` | 自检 E1：固定随机锚点的可复现性与报警线 | `—` | `—` |
| 156 | `test_solo_rand_anchor.make`<br>（内嵌于 `test_solo_rand_anchor`） | `rl/selftest.py:4668` | 辅助：临时设种子构造策略后恢复调用方 RNG | `seed` | `—` |
| 157 | `test_anchor_light_point_state` | `rl/selftest.py:4695` | 自检 C 方案：轻量锚点与全点对照共存且可复原 | `—` | `—` |
| 158 | `test_anchor_light_point_state.rd`<br>（内嵌于 `test_anchor_light_point_state`） | `rl/selftest.py:4734` | 辅助：读取 solo_state.json | `—` | `—` |
| 159 | `test_adv_inert_probe_and_const_baseline` | `rl/selftest.py:4771` | 自检：adv_alt 惰性探针零副作用与常数基线 GAE 恒等式 | `—` | `—` |
| 160 | `test_adv_inert_probe_and_const_baseline._params`<br>（内嵌于 `test_adv_inert_probe_and_const_baseline`） | `rl/selftest.py:4807` | 辅助：克隆策略全部参数用于逐位比较 | `p` | `—` |
| 161 | `main` | `rl/selftest.py:4885` | 入口：强制 UTF-8 stdout 并按序运行全部自检 | `—` | `—` |

##### B.8.31 `rl/train_baseline.py`（9 个符号）

| 序号 | 函数/方法 | 文件:行号 | 一句话作用 | 关键参数（名字=默认值） | 返回 |
|---|---|---|---|---|--- |
| 1 | `SingleCardAdapter` | `rl/train_baseline.py:26` | 把 bundle 动作适配成旧式单卡动作 | `—` | `—` |
| 2 | `SingleCardAdapter.__init__`<br>（内嵌于 `SingleCardAdapter`） | `rl/train_baseline.py:29` | 包装 RLEnv 并设 MultiDiscrete 动作空间 | `self, **env_kwargs` | `—` |
| 3 | `SingleCardAdapter.reset`<br>（内嵌于 `SingleCardAdapter`） | `rl/train_baseline.py:34` | 直接转发内部环境 reset | `self, *, seed=None, options=None` | `—` |
| 4 | `SingleCardAdapter.step`<br>（内嵌于 `SingleCardAdapter`） | `rl/train_baseline.py:37` | 单卡动作转 bundle 后推进环境 | `self, action` | `—` |
| 5 | `SingleCardAdapter.__getattr__`<br>（内嵌于 `SingleCardAdapter`） | `rl/train_baseline.py:41` | 未定义属性转发给内部 RLEnv | `self, name` | `—` |
| 6 | `CRFeatureExtractor` | `rl/train_baseline.py:45` | SB3 特征提取器（实体嵌入+CNN） | `—` | `—` |
| 7 | `CRFeatureExtractor.__init__`<br>（内嵌于 `CRFeatureExtractor`） | `rl/train_baseline.py:46` | 构建 CNN 与融合全连接层 | `self, observation_space: gym.spaces.Dict, features_dim: int=256` | `—` |
| 8 | `CRFeatureExtractor.forward`<br>（内嵌于 `CRFeatureExtractor`） | `rl/train_baseline.py:62` | 拼接网格手牌额外特征输出 | `self, observation` | `—` |
| 9 | `main` | `rl/train_baseline.py:80` | 解析参数、训练 SB3 PPO 并保存 | `—` | `—` |

##### B.8.32 `rl/train_bc.py`（3 个符号）

| 序号 | 函数/方法 | 文件:行号 | 一句话作用 | 关键参数（名字=默认值） | 返回 |
|---|---|---|---|---|--- |
| 1 | `expert_bundle` | `rl/train_bc.py:40` | 规则专家：建议卡落到区域中心最近合法格 | `env, belief, bp, obs, rng` | `—` |
| 2 | `collect` | `rl/train_bc.py:60` | 用专家策略采集 BC 样本 | `n_games, seed, policy, max_steps=600` | `—` |
| 3 | `train_bc` | `rl/train_bc.py:86` | 做行为克隆并保存 checkpoint | `n_games=50, epochs=3, lr=0.001, hidden_dim=128, seed=0, out='follower_bc.pt', max_steps=600` | `—` |

##### B.8.33 `rl/train_belief.py`（10 个符号）

| 序号 | 函数/方法 | 文件:行号 | 一句话作用 | 关键参数（名字=默认值） | 返回 |
|---|---|---|---|---|--- |
| 1 | `sample_bundle` | `rl/train_belief.py:33` | 从掩码随机采样一张卡与落点 | `mask, rng` | `—` |
| 2 | `collect_replays` | `rl/train_belief.py:49` | 随机策略跑多局录带 hidden 标签回放 | `n_games, seed, opponent=None, max_steps=600` | `—` |
| 3 | `_episode_arrays` | `rl/train_belief.py:72` | 单局回放转特征与两类标签数组 | `ep` | `—` |
| 4 | `build_episodes` | `rl/train_belief.py:90` | 回放列表转 episode 序列并取卡组 | `replays` | `—` |
| 5 | `brier_of` | `rl/train_belief.py:103` | 计算下一张牌预测平均 Brier 分数 | `probs, y` | `—` |
| 6 | `nll_of` | `rl/train_belief.py:110` | 计算交叉熵 NLL | `logits, y` | `—` |
| 7 | `ece_of` | `rl/train_belief.py:117` | 按置信度分箱算期望校准误差 | `probs, y, n_bins=10` | `—` |
| 8 | `fit_temperature` | `rl/train_belief.py:132` | 候选温度网格最小化 NLL 选 T | `logits, y` | `—` |
| 9 | `train` | `rl/train_belief.py:145` | 训练信念编码器并保存权重与温度 | `epochs=10, lr=0.001, batch_size=64, n_games=50, seed=0, out='belief_encoder.pt', replays_path=None, opponent=None, val_frac=0.2, max_steps=600` | `—` |
| 10 | `train.eval_metrics`<br>（内嵌于 `train`） | `rl/train_belief.py:177` | 前向评估 next-acc/Brier/NLL/ECE | `episodes_, temperature=1.0` | `—` |

##### B.8.34 `rl/train_exploiter.py`（3 个符号）

| 序号 | 函数/方法 | 文件:行号 | 一句话作用 | 关键参数（名字=默认值） | 返回 |
|---|---|---|---|---|--- |
| 1 | `_play_side0` | `rl/train_exploiter.py:27` | policy 以 p0 身份打整局返回赢家 | `env, policy, belief, bp, max_steps=300` | `—` |
| 2 | `evaluate_winrate` | `rl/train_exploiter.py:56` | 换边评估 exploiter 对 main 胜率 | `exploiter, main, n_games=10, seed=0, max_steps=300` | `—` |
| 3 | `main` | `rl/train_exploiter.py:89` | 训练 exploiter 评估并按阈值入联赛 | `—` | `—` |

##### B.8.35 `rl/train_follower.py`（12 个符号）

| 序号 | 函数/方法 | 文件:行号 | 一句话作用 | 关键参数（名字=默认值） | 返回 |
|---|---|---|---|---|--- |
| 1 | `heuristic_opponent` | `rl/train_follower.py:33` | 返回脚本对手（p1 掩码随机出牌） | `env, rng=None` | `—` |
| 2 | `heuristic_opponent._opp`<br>（内嵌于 `heuristic_opponent`） | `rl/train_follower.py:37` | 对手闭包：采样合法槽位与格子 | `obs` | `—` |
| 3 | `FollowerOpponent` | `rl/train_follower.py:52` | 把 FollowerPolicy 包装成 p1 对手 | `—` | `—` |
| 4 | `FollowerOpponent.__init__`<br>（内嵌于 `FollowerOpponent`） | `rl/train_follower.py:60` | 初始化对手信念规划器与 plan 偏置开关 | `self, policy, env, belief=None, planner=None, deterministic=True, use_plan_biases=False` | `—` |
| 5 | `FollowerOpponent.observe_opponent_played`<br>（内嵌于 `FollowerOpponent`） | `rl/train_follower.py:74` | 用 p0 出牌更新对手信念 | `self, played_cards` | `—` |
| 6 | `FollowerOpponent.reset`<br>（内嵌于 `FollowerOpponent`） | `rl/train_follower.py:79` | 清空 hidden 与信念状态 | `self` | `—` |
| 7 | `FollowerOpponent.__call__`<br>（内嵌于 `FollowerOpponent`） | `rl/train_follower.py:84` | 算信念与 plan 后出一步动作 | `self, obs` | `—` |
| 8 | `FollowerOpponent.__call__.mask_fn`<br>（内嵌于 `FollowerOpponent.__call__`） | `rl/train_follower.py:93` | 取 player-1 的合法动作掩码（partial 为已选子动作） | `partial=None` | `—` |
| 9 | `FollowerOpponent.take_last_step`<br>（内嵌于 `FollowerOpponent`） | `rl/train_follower.py:112` | 返回最近一步 p1 轨迹字典 | `self` | `—` |
| 10 | `run_training` | `rl/train_follower.py:117` | 在线采 rollout 并用 PPOTrainer 训练 follower | `total_steps, n_envs=1, batch_size=128, update_interval=128, lr=0.0003, gamma=0.99, gae_lambda=0.95, clip=0.2, plan_prophet_prob=0.3, plan_dropout=0.1, belief_dropout=0.1, seed=0, opponent='random', main_policy_path=None, hidden_dim=128, save='follower.pt', eval_every=2000, max_ep_steps=600, proph…` | `—` |
| 11 | `run_training.make_plan`<br>（内嵌于 `run_training`） | `rl/train_follower.py:151` | 按开关选先知或信念规划器产出 plan | `obs, use_prophet` | `—` |
| 12 | `evaluate` | `rl/train_follower.py:242` | 对随机对手评估胜率与平均回报 | `policy, belief, belief_planner, n_games=5, seed=0` | `—` |

##### B.8.36 `rl/train_prophet.py`（12 个符号）

| 序号 | 函数/方法 | 文件:行号 | 一句话作用 | 关键参数（名字=默认值） | 返回 |
|---|---|---|---|---|--- |
| 1 | `_build_priv` | `rl/train_prophet.py:35` | 构造归一化对手特权向量（12 维） | `env, obs` | `—` |
| 2 | `ProphetEnv` | `rl/train_prophet.py:46` | 特权观测 gym 环境（obs+priv，单卡动作） | `—` | `—` |
| 3 | `ProphetEnv.__init__`<br>（内嵌于 `ProphetEnv`） | `rl/train_prophet.py:49` | 包装 RLEnv 并扩展 priv 观测空间 | `self, **env_kwargs` | `—` |
| 4 | `ProphetEnv.reset`<br>（内嵌于 `ProphetEnv`） | `rl/train_prophet.py:62` | 重置环境并给观测附加 priv | `self, *, seed=None, options=None` | `—` |
| 5 | `ProphetEnv._augment`<br>（内嵌于 `ProphetEnv`） | `rl/train_prophet.py:66` | 给观测字典追加 priv 字段 | `self, obs` | `—` |
| 6 | `ProphetEnv.step`<br>（内嵌于 `ProphetEnv`） | `rl/train_prophet.py:69` | 单卡动作转 bundle 后推进环境 | `self, action` | `—` |
| 7 | `ProphetEnv.__getattr__`<br>（内嵌于 `ProphetEnv`） | `rl/train_prophet.py:74` | 未定义属性转发给内部 RLEnv | `self, name` | `—` |
| 8 | `ProphetExtractor` | `rl/train_prophet.py:78` | SB3 特征提取器（CNN+实体嵌入+priv） | `—` | `—` |
| 9 | `ProphetExtractor.__init__`<br>（内嵌于 `ProphetExtractor`） | `rl/train_prophet.py:79` | 构建 CNN、实体嵌入与融合全连接层 | `self, observation_space: gym.spaces.Dict, features_dim: int=256` | `—` |
| 10 | `ProphetExtractor.forward`<br>（内嵌于 `ProphetExtractor`） | `rl/train_prophet.py:94` | 拼接网格手牌额外特权特征输出 | `self, observation` | `—` |
| 11 | `prophet_policy_to_plan` | `rl/train_prophet.py:112` | 把先知动作适配成 PlanToken | `model, obs, env` | `PlanToken` |
| 12 | `main` | `rl/train_prophet.py:141` | 解析参数、训练 SB3 PPO 并保存 | `—` | `—` |

##### B.8.37 `rl/train_solo.py`（35 个符号）

| 序号 | 函数/方法 | 文件:行号 | 一句话作用 | 关键参数（名字=默认值） | 返回 |
|---|---|---|---|---|--- |
| 1 | `_rand_anchor_warns` | `rl/train_solo.py:72` | 绝对强度警报（E1）：低于阈值返回告警列表（空=通过）。阈值未标定，先只报警。 | `winrate, floor=RAND_ANCHOR_WARN_FLOOR` | `—` |
| 2 | `_make_rand_anchor` | `rl/train_solo.py:80` | 构造固定随机锚点策略（E1/E2 共用，权重种子 RAND_ANCHOR_SEED）。 | `cfg, belief_dim, device=None` | `—` |
| 3 | `resolve_deck_set` | `rl/train_solo.py:107` | cfg.deck_set → (mirror_deck, deck_pool_for_defender)。 | `deck_set: str` | `—` |
| 4 | `solo_env` | `rl/train_solo.py:157` | 双方固定卡组的镜像 RLEnv（deck_set=list:/default 时同副镜像）。 | `cfg, seed, deck0=None, deck1=None` | `—` |
| 5 | `_draw_penalty` | `rl/train_solo.py:164` | 平局惩罚（= 失败：平局不再免费）。缺省与 lose_penalty 相同。 | `cfg` | `float` |
| 6 | `_sync_frozen_copy` | `rl/train_solo.py:170` | 把 main 当前权重同步给冻结副本（周期执行）。 | `main, opp` | `—` |
| 7 | `_collect_hist_ckpts` | `rl/train_solo.py:175` | 收集 solo 输出目录的历史 checkpoint（solo_main_<step>.pt）。 | `folder, max_n=_HIST_POOL_MAX, extra_dirs=None` | `—` |
| 8 | `_collect_hist_ckpts._scan`<br>（内嵌于 `_collect_hist_ckpts`） | `rl/train_solo.py:187` | 单目录扫描 solo_main_*.pt 并按步数均匀抽 | `d, budget` | `—` |
| 9 | `_dedup_history` | `rl/train_solo.py:224` | 按 step 去重（同 step 只留一条）。 | `history, step` | `—` |
| 10 | `_check_gates` | `rl/train_solo.py:234` | 行为指标门禁（P0-2b，2026-09-11 判读整改）：**先只报警不阻断**。 | `stats, cfg, path=None` | `—` |
| 11 | `_write_gate_report` | `rl/train_solo.py:309` | 把门禁报告写 gates.json（失败只打印不中断） | `path, report` | `—` |
| 12 | `_OpponentPool` | `rl/train_solo.py:321` | 训练对手选择器（9j + E2）：frozen / hist / defend / rand_anchor 四类按 mix 采样。 | `—` | `—` |
| 13 | `_OpponentPool.__init__`<br>（内嵌于 `_OpponentPool`） | `rl/train_solo.py:336` | 建四类对手（frozen/hist/defend/rand_anchor）与 PFSP | `self, cfg, env, frozen_side, rng, device, defender_deck_pool=None, hist_seed_dirs=None` | `—` |
| 14 | `_OpponentPool._reindex_hist`<br>（内嵌于 `_OpponentPool`） | `rl/train_solo.py:403` | D1：把 hist 池映射到**稳定 id**（父目录名 + 文件名），供 PFSP 统计跨刷新保留。 | `self` | `—` |
| 15 | `_OpponentPool.refresh_hist`<br>（内嵌于 `_OpponentPool`） | `rl/train_solo.py:414` | D1：重扫磁盘，把本 run 自己的新快照纳入 hist 池（原来只在 __init__ 扫一次， | `self, step=None` | `—` |
| 16 | `_OpponentPool.sample`<br>（内嵌于 `_OpponentPool`） | `rl/train_solo.py:436` | 为本局选对手：返回 (kind, opponent, hist_id_or_None)。 | `self` | `—` |
| 17 | `_OpponentPool._sample_kind`<br>（内嵌于 `_OpponentPool`） | `rl/train_solo.py:448` | 按 mix 抽样，返回 (kind, side, hist_id) | `self` | `—` |
| 18 | `_OpponentPool.record`<br>（内嵌于 `_OpponentPool`） | `rl/train_solo.py:479` | 上一局结束回填 PFSP 胜率（frozen/defend 局无操作）。 | `self, winner` | `—` |
| 19 | `_OpponentPool._ensure_hist`<br>（内嵌于 `_OpponentPool`） | `rl/train_solo.py:487` | 载入 hist ckpt（换目标才重载；belief_dim 尾部零拷贝兼容旧 23 维）。 | `self, path` | `—` |
| 20 | `write_solo_state` | `rl/train_solo.py:513` | 把 solo 训练状态落盘（增量写；dashboard --solo 实时读取）。 | `path, cfg, history, step, status='running', deck=None, copy_every=None, target_steps=None, controls=None` | `—` |
| 21 | `behavioral_metrics` | `rl/train_solo.py:550` | 从回放 games 算行为指标（2026-09-10 用户：胜率在镜像自对弈下自我对冲≈0.5 恒定， | `games` | `—` |
| 22 | `behavioral_metrics._pct`<br>（内嵌于 `behavioral_metrics`） | `rl/train_solo.py:695` | 百分比工具（分母 0 返回 0.0） | `a, b` | `—` |
| 23 | `eval_solo` | `rl/train_solo.py:713` | main（deterministic）vs 冻结副本（deterministic）打 n_games。 | `env, main, opp, n_games, max_steps, seed, cfg, record_replays=False, replays_dir=None, step=None, frozen_step=None, save_replays=True` | `—` |
| 24 | `_eval_worker_main` | `rl/train_solo.py:802` | 并行评估 worker：独立进程打 games（全局游戏索引列表）里每局。 | `worker_id, main_sd, opp_sd, games, env_kwargs, seed_base, max_steps, n_particles, record, out_q` | `—` |
| 25 | `_collect_worker_results` | `rl/train_solo.py:916` | 收齐 expected 份 worker 消息，返回 (results, failure_or_None)。 | `procs, out_q, expected` | `—` |
| 26 | `eval_solo_parallel` | `rl/train_solo.py:941` | eval_solo 的进程池并行版：n_games 局均分到 n_workers 个 spawn 进程打。 | `env, main, opp, n_games, max_steps, seed, cfg, n_workers=8, record_replays=False, replays_dir=None, step=None, frozen_step=None, save_replays=True` | `—` |
| 27 | `run_solo` | `rl/train_solo.py:1039` | 单人自对弈主循环（无联赛；写 solo_state.json + solo_main.pt）。 | `cfg, resume=False, record_replays=True` | `—` |
| 28 | `run_solo._gae_pair`<br>（内嵌于 `run_solo`） | `rl/train_solo.py:1221` | 返回 (adv_real, ret_real, adv_const, level_gap)。 | `rew, val, term, trunc, last_val` | `—` |
| 29 | `run_solo._sync_controls_once`<br>（内嵌于 `run_solo`） | `rl/train_solo.py:1240` | 首次把 main 权重同步给 baseline0/baseline_prev | `—` | `—` |
| 30 | `run_solo.eval_control`<br>（内嵌于 `run_solo`） | `rl/train_solo.py:1246` | main vs 对照对手打 n_eval_games 局（确定性），返回 stats dict（不落盘、不迭代）。 | `step, label, opp_model, seed` | `—` |
| 31 | `run_solo.eval_and_write`<br>（内嵌于 `run_solo`） | `rl/train_solo.py:1268` | 主评估点：评估+池化 EV+活力探针+三组对照+落盘 | `step` | `—` |
| 32 | `run_solo._persist`<br>（内嵌于 `run_solo`） | `rl/train_solo.py:1410` | 落盘训练态：主快照 + 步进快照（D1 自身联赛成员）+ Adam + run_state。 | `step` | `—` |
| 33 | `run_solo._persist._med`<br>（内嵌于 `run_solo._persist`） | `rl/train_solo.py:1423` | 取诊断记录中某键的中位数 | `key` | `—` |
| 34 | `run_solo.anchor_point`<br>（内嵌于 `run_solo`） | `rl/train_solo.py:1454` | C 方案（2026-09-13）：轻量评估点 = 只跑固定随机锚点（C1 判据原料）+ 落盘。 | `step` | `—` |
| 35 | `run_solo._new_episode_reset`<br>（内嵌于 `run_solo`） | `rl/train_solo.py:1513` | 局间重置 + A 层对手池采样。 | `winner` | `—` |

##### B.8.38 `rl/workers.py`（3 个符号）

| 序号 | 函数/方法 | 文件:行号 | 一句话作用 | 关键参数（名字=默认值） | 返回 |
|---|---|---|---|---|--- |
| 1 | `_payload` | `rl/workers.py:45` | 计算下一个决策步的 (obs, belief_tok, plan_vec)。 | `env, belief, bp, prophet, obs, rng` | `—` |
| 2 | `_apply_opponent` | `rl/workers.py:54` | 按 spec 装配对手（脚本策略在 worker 本地重建并绑定 env）。 | `env, spec` | `—` |
| 3 | `worker_main` | `rl/workers.py:71` | 跨进程 env 推演循环（独立进程入口，绕开 GIL 实现多核并行）。 | `worker_id, seed, reward_weights, in_q, out_q, card_level=None` | `—` |

---

### B.9 本部分待确认清单

| # | 条目 | 无法确认的原因 | 要确认需要什么 |
|---|---|---|--- |
| 1 | `eval_workers` 的生效默认值 | `rl/config.py:209` 的 dataclass 默认是 `min(16, os.cpu_count() or 1)`，而 CLI `--eval-workers` 的 help 文本写"默认 0=串行"（rl/run_league.py:1246-1248），两者矛盾；`economy` 预设未设置该字段（rl/config.py:344-364）⇒ 实际继承 dataclass 默认 | 跑一次 `--help` 与一次短 run，打印 `cfg.eval_workers`（或看启动日志"eval@0 … (并行worker=N)"，rl/train_solo.py:1489-1490） |
| 2 | `PLAN_DIM` 到底是 57 还是 58 | `rl/plan_space.py:11-12` 的模块 docstring 写 "PLAN_DIM = 57"、"placement_hint(7)"，而 `PLACEMENT_HINTS` 实为 8 项（:65-74）、`PLAN_DIM = len(PlanToken().to_vector())`（:187）与 `to_vector` 的拼接（:155-160）在只读源码下算得 58 | 直接执行 `python -c "from rl.plan_space import PLAN_DIM; print(PLAN_DIM)"`（本文档不执行代码） |
| 3 | `resolve_deck_set` docstring 与 `FOUR_DECK_SET` 不一致 | docstring 写 `"four"` = "速猪/皇家巨人/X弩/双线快攻"（rl/train_solo.py:111-113），而 `FOUR_DECK_SET` 实际是速猪 / 石头人 / X 弩 / 巨骷髅攻城槌（rl/opponents.py:30-46） | 以代码为准；docstring 待维护者更新，或确认是否存在历史别名表 |
| 4 | `mcts.py` 是否有训练/评估之外的调用点 | `evaluate.py` 的 argparse 无 `--mcts`（rl/evaluate.py:528-542），`train_solo.py`/`run_league.py`/`ppo.py` 无 `RLMCTS` 命中；仅见 `rl/selftest.py:2983/3005/3017/3062/3075/3088` 调用 | 全仓（含 `scripts/`）grep `RLMCTS`；本任务限定文件范围内无法判定 |
| 5 | belief token 的实际维度（563？） | 维度公式在 `rl/belief.py:36-38` 依赖 `len(ENTITY_NAMES)`，而 `ENTITY_NAMES` 定义在 `rl/observation.py`（本文档未获准阅读）⇒ 无法手算 | 跑 `python -c "from rl.belief import BeliefInference; ..."` 或读 `rl/observation.py` 数出 `ENTITY_NAMES` 长度 |
| 6 | `run_league` 三条主循环（`_run_single`/`_run_vec`/`_run_mp`）内部顺序 | 本部分只读了 `run_league` 的模式分派（:1143-1149）与公共前缀/评估函数，未逐行读 :697-1142 的三个循环体 | 逐行读 `rl/run_league.py:697-1142`（或另开一节 C 部分） |
| 7 | flow-sweep 策略细节（`SWEEP_STRATEGIES`/`_sweep_trend`/`_write_sweep`/`run_flow_sweep`） | 只读了 `run_flow` 主循环与池/模型构造（:56-463），:470-613 未读 | 读 `rl/flow_league.py:470-613` |
| 8 | `RLEnv.step` 的奖励合成细节与 `compute_reward` 分量 | 本部分只需 `env.step` 的调用位置，未读 `rl/env_wrapper.py:598+` 的实现 | 读 `rl/env_wrapper.py`（奖励账本在 `compute_reward`） |
| 9 | `train_belief.train()` 的 `batch_size=64` 形参是否真未使用 | 该名在函数体内只出现在签名行（grep 证据），但可能存在**动态关键字**调用方（本任务未做全仓调用图） | 全仓 grep `train_belief` 的调用点与 `**kwargs` 传参 |
| 10 | `train_follower` 的先知蒸馏分支是否有 Python 调用方 | `prophet_model=None` 且 CLI 无对应参数（rl/train_follower.py:122、:281-283），全仓 `prophet_model` 只命中该文件三处 | 全仓（含 `scripts/`、notebook）grep `prophet_model=` |
| 11 | hist 池中"本目录 ckpt 挤出外部补种"的时间点 | `_collect_hist_ckpts` 本目录优先（rl/train_solo.py:205-207），但挤出的具体步数取决于每评估点是否落 `solo_main_<step>.pt`（`_persist` 只在评估点调用，:1408、:1474） | 实跑一个 20k run 并解析 `solo_state.json` 的 `_controls_history`/日志中 `对手池刷新` 行（rl/train_solo.py:431-433） |
| 12 | `_probe["adv_inert"]` 之外的诊断字段是否有下游消费者 | `run_state.json` 的 `adv_inert` 汇总（:1420-1439）由脚本消费，本文档未读 `scripts/` | 读 `scripts/` 下判读脚本（本任务范围外） |

---

## 附录 A：待确认事项汇总

下列条目是各部分**显式标注为无法从源码确认**的内容，集中列在这里以防被误当作事实。

> **重要提示**：不少条目的「无法确认的原因」并不是「源码里查不到」，而是**该子代理的阅读清单里没有那个文件**。
> 这些文件在本项目的《项目内容全解文档》[`full_code_reference.md`](full_code_reference.md) 里已被逐函数说明
> （例如 `pathfinding_heap.py`、`rl/action_mask.py`、`spell_module.py`、`rl/env_wrapper.py`、`new_visualization.py`）——
> 因此请**先到该文档对应章节取行号，再回源码核验**；本附录只负责标明「哪些结论还没被独立确认过」。

### 来源：A.7 本部分待确认清单

| # | 条目 | 无法确认的原因 | 要确认需要什么 |
|---|---|---|---|
| 1 | flow 模式（`--mode flow`）的完整数据流、模型槽位、评估口径、`cfg` 字段使用范围 | `rl/flow_league.py` 不在本部分允许阅读的源码清单内，只能从 `run_league.py:1351-1363` 看到入口签名与两个参数 | 读 `rl/flow_league.py` 的 `run_flow` / `run_flow_sweep` 全文，以及 `League` 在 flow 下的装配方式 |
| 2 | `flow-sweep-*` 两个策略（`stream` / `games5`）的具体判别与产出一致性 | 同上 | 同上 |
| 3 | 总参数量与既有文档"629,359"不一致的成因 | 源码中没有硬编码参数量；本文两个数（634,735 / 993,008）是实算值，差异只能来自历史 `belief_dim`/`num_entity`/`hidden` 或旧架构 | 找到写 629,359 的文档所依据的版本/配置，或在那个版本上重算 |
| 4 | `PLAN_DIM=58` 与 `belief_dim=563` 的内部构成（各字段偏移、事件 one-hot 宽度） | 定义在 `rl/plan_space.py` 与 `rl/belief.py`，不在允许阅读清单内；本文只用了模块导出的常量与实算长度 | 读 `rl/plan_space.py`（`PlanToken.to_vector`/`PLAN_DIM`）与 `rl/belief.py`（`belief_token_dim`/`encode`） |
| 5 | run 模式下 `--anchor-every` 被接受但不生效，这是设计意图还是遗漏 | 事实层面已由 grep 确认（唯一消费点在 `train_solo.py:1720`；run 侧只有 `run_league.py:1308` 的白名单），但**意图**无法从源码判定（无相关 TODO/注释） | 查 `docs/eval_cadence_c_2026-09-13.md` 声明的适用范围，或向作者确认 run 模式是否需要同节奏 |
| 6 | run 模式评估侧早停与低置信降噪的配合关系 | `_run_side0` 早停后直接调 `timeout_winner`（`run_league.py:307-309`），而 solo 训练环早停用 `settle_stall(stall_draw_margin)`（`train_solo.py:1552`）；run 模式是否应改用 `settle_stall` 属设计选择，源码无说明 | 明确 run 模式评估/训练各自的结算口径设计意图 |
| 7 | `league_state.json` 的具体字段与恢复语义 | 由 `rl/league.py` 的 `save_state` / `load_state` / `sample_opponent` / `record_match` 定义，该文件不在允许阅读清单内 | 读 `rl/league.py` |
| 8 | `info["opp_played"]` 到信念模块的过滤细节（技能哨兵 `"__ability__"` 如何被消费） | 过滤逻辑在 `rl/belief.py` 的 `update` 内，不在允许阅读清单内 | 读 `rl/belief.py:update` |
| 9 | `ProphetPlanner` / `BeliefPlanner` 的输出维度语义（58 维中哪一段对应什么） | 同上第 4 条；正文只用到 `to_vector()` 的返回值与 `PLAN_DIM` | 读 `rl/plan_space.py`、`rl/prophet.py`、`rl/belief_planner.py` |
| 10 | `eval_workers` 默认值在目标机器上的实际数值 | `default_factory = min(16, os.cpu_count() or 1)`（`config.py:209`）依赖运行机器；本文不能给一个确定的数字 | 在目标机器上打印 `TrainConfig().eval_workers` |
| 11 | `--mode eval` 与 `--mode flow` 是否也支持 `--load-config` / `--save-config` 的全部组合 | 分派顺序上 `cfg = TrainConfig.resolve(...)` 在 mode 分派之前（`1346-1353`），但 flow 侧如何使用 `cfg` 未在允许清单内 | 同上第 1 条 |
| 12 | `kinds` 的合法取值集合（`--kinds`） | `evaluate_league` 直接把 `kinds[i]` 透传给 `league.add_agent(..., kind=...)`（`run_league.py:514-518`），未做校验；合法值的定义在 `rl/league.py` | 读 `rl/league.py`（`add_agent`/`Agent`） |

### 来源：B.9 本部分待确认清单

| # | 条目 | 无法确认的原因 | 要确认需要什么 |
|---|---|---|---|
| 1 | `eval_workers` 的生效默认值 | `rl/config.py:209` 的 dataclass 默认是 `min(16, os.cpu_count() or 1)`，而 CLI `--eval-workers` 的 help 文本写"默认 0=串行"（rl/run_league.py:1246-1248），两者矛盾；`economy` 预设未设置该字段（rl/config.py:344-364）⇒ 实际继承 dataclass 默认 | 跑一次 `--help` 与一次短 run，打印 `cfg.eval_workers`（或看启动日志"eval@0 … (并行worker=N)"，rl/train_solo.py:1489-1490） |
| 2 | `PLAN_DIM` 到底是 57 还是 58 | `rl/plan_space.py:11-12` 的模块 docstring 写 "PLAN_DIM = 57"、"placement_hint(7)"，而 `PLACEMENT_HINTS` 实为 8 项（:65-74）、`PLAN_DIM = len(PlanToken().to_vector())`（:187）与 `to_vector` 的拼接（:155-160）在只读源码下算得 58 | 直接执行 `python -c "from rl.plan_space import PLAN_DIM; print(PLAN_DIM)"`（本文档不执行代码） |
| 3 | `resolve_deck_set` docstring 与 `FOUR_DECK_SET` 不一致 | docstring 写 `"four"` = "速猪/皇家巨人/X弩/双线快攻"（rl/train_solo.py:111-113），而 `FOUR_DECK_SET` 实际是速猪 / 石头人 / X 弩 / 巨骷髅攻城槌（rl/opponents.py:30-46） | 以代码为准；docstring 待维护者更新，或确认是否存在历史别名表 |
| 4 | `mcts.py` 是否有训练/评估之外的调用点 | `evaluate.py` 的 argparse 无 `--mcts`（rl/evaluate.py:528-542），`train_solo.py`/`run_league.py`/`ppo.py` 无 `RLMCTS` 命中；仅见 `rl/selftest.py:2983/3005/3017/3062/3075/3088` 调用 | 全仓（含 `scripts/`）grep `RLMCTS`；本任务限定文件范围内无法判定 |
| 5 | belief token 的实际维度（563？） | 维度公式在 `rl/belief.py:36-38` 依赖 `len(ENTITY_NAMES)`，而 `ENTITY_NAMES` 定义在 `rl/observation.py`（本文档未获准阅读）⇒ 无法手算 | 跑 `python -c "from rl.belief import BeliefInference; ..."` 或读 `rl/observation.py` 数出 `ENTITY_NAMES` 长度 |
| 6 | `run_league` 三条主循环（`_run_single`/`_run_vec`/`_run_mp`）内部顺序 | 本部分只读了 `run_league` 的模式分派（:1143-1149）与公共前缀/评估函数，未逐行读 :697-1142 的三个循环体 | 逐行读 `rl/run_league.py:697-1142`（或另开一节 C 部分） |
| 7 | flow-sweep 策略细节（`SWEEP_STRATEGIES`/`_sweep_trend`/`_write_sweep`/`run_flow_sweep`） | 只读了 `run_flow` 主循环与池/模型构造（:56-463），:470-613 未读 | 读 `rl/flow_league.py:470-613` |
| 8 | `RLEnv.step` 的奖励合成细节与 `compute_reward` 分量 | 本部分只需 `env.step` 的调用位置，未读 `rl/env_wrapper.py:598+` 的实现 | 读 `rl/env_wrapper.py`（奖励账本在 `compute_reward`） |
| 9 | `train_belief.train()` 的 `batch_size=64` 形参是否真未使用 | 该名在函数体内只出现在签名行（grep 证据），但可能存在**动态关键字**调用方（本任务未做全仓调用图） | 全仓 grep `train_belief` 的调用点与 `**kwargs` 传参 |
| 10 | `train_follower` 的先知蒸馏分支是否有 Python 调用方 | `prophet_model=None` 且 CLI 无对应参数（rl/train_follower.py:122、:281-283），全仓 `prophet_model` 只命中该文件三处 | 全仓（含 `scripts/`、notebook）grep `prophet_model=` |
| 11 | hist 池中"本目录 ckpt 挤出外部补种"的时间点 | `_collect_hist_ckpts` 本目录优先（rl/train_solo.py:205-207），但挤出的具体步数取决于每评估点是否落 `solo_main_<step>.pt`（`_persist` 只在评估点调用，:1408、:1474） | 实跑一个 20k run 并解析 `solo_state.json` 的 `_controls_history`/日志中 `对手池刷新` 行（rl/train_solo.py:431-433） |
| 12 | `_probe["adv_inert"]` 之外的诊断字段是否有下游消费者 | `run_state.json` 的 `adv_inert` 汇总（:1420-1439）由脚本消费，本文档未读 `scripts/` | 读 `scripts/` 下判读脚本（本任务范围外） |

---

## 附录 B：生成方式与可复现性

```bash
cd <仓库根目录>
PY=.venv/Scripts/python.exe
# 1) 逐文件摸底（AST 清单 + 分组 + 子代理分析 + 覆盖对账）
PYTHONIOENCODING=utf-8 $PY scripts/_survey_inventory.py --out docs/_survey/inventory.json
PYTHONIOENCODING=utf-8 $PY scripts/_survey_groups.py --inventory docs/_survey/inventory.json --out docs/_survey/groups.json
PYTHONIOENCODING=utf-8 $PY scripts/_survey_verify.py
# 2) 合并三份交付物
PYTHONIOENCODING=utf-8 $PY scripts/_survey_merge.py         # -> docs/full_code_reference.md
PYTHONIOENCODING=utf-8 $PY scripts/_survey_merge_docs.py --kind training --out docs/training_method.md
PYTHONIOENCODING=utf-8 $PY scripts/_survey_merge_docs.py --kind engine   --out docs/game_engine.md
# 3) 生成 DOCX
PYTHONIOENCODING=utf-8 $PY scripts/_survey_md_to_docx.py --md docs/training_method.md --docx docs/training_method.docx --title 训练方法文档
PYTHONIOENCODING=utf-8 $PY scripts/_survey_md_to_docx.py --md docs/game_engine.md     --docx docs/game_engine.docx     --title 游戏引擎文档
```

| 源材料 | 说明 |
|---|---|
| `docs/_survey/inventory.json` | AST 先验清单（每个 `.py` 的全部符号/签名/行号）|
| `docs/_survey/parts/*.md` | 48 份逐文件分析素材（本文件正文的来源）|
| `docs/_survey/drafts/` | 本文档 A/B 两部分草稿 |
| `docs/_survey/coverage_report.json` | 覆盖对账结果 |

