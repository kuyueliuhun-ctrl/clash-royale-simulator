# 训练方法文档 · 第 A 部分：训练模式、数据流、网络与超参

> **本文的全部事实性陈述均来自以下源码**（未引用任何设计文档、README 或联网内容）：
> `src/clasher_new/rl/config.py`、`run_league.py`、`train_solo.py`、`ppo.py`、`follower.py`、
> `env_wrapper.py`、`observation.py`，以及入口脚本 `start_rl.bat`、`start_training.bat`、
> `scripts/rl/run_league.py`。
> 行内标注格式 `（rl/xxx.py:NNN）` 指该文件的第 NNN 行（`rl/` = `src/clasher_new/rl/`）。
> 无法从源码确认的内容一律标「**待确认**」并说明原因。
> 参数量、`cnn_out`、`PLAN_DIM`、`belief_dim` 等数值由**实例化 `FollowerPolicy` 实算**得到
> （方法见 A.3.4），非抄自文档。

---

## A.1 训练体系概览

### A.1.1 三种模式

| 模式 | CLI 取值 | 语义（源码原义） | 入口函数 | 源码位置 |
|---|---|---|---|---|
| **solo** | `--mode solo` | 单人自对弈：单模型 `main`（`FollowerPolicy`+`PPOTrainer`），双方同副固定卡组镜像；对手 = main 的周期冻结副本 + 对手池；无联赛、不写 Elo/PFSP/`league_state.json`；周期评估写 `solo_state.json` | `run_solo(cfg, resume=..., record_replays=...)` | `rl/train_solo.py:1039`；分派 `rl/run_league.py:1365-1369` |
| **run** | `--mode run`（默认） | 联赛主循环：同时维护 5 个模型槽位（`main` + 三分类×3/all_decks + 全随机），PPO 训练 main → PFSP 采对手 → 周期全轮转评估 → 逐局 Elo → 状态持久化 | `run_league(cfg, resume=..., record_replays=...)` | `rl/run_league.py:1143`；分派 `rl/run_league.py:1371` |
| **flow** | `--mode flow` | 全配对分流派联赛（一次 148,800 局） | `rl.flow_league.run_flow(cfg, resume=..., n_random_decks=...)` | 调用点 `rl/run_league.py:1351-1354`；模块 docstring 亦声明 `rl/run_league.py:10` |
| flow 扫描 | `--mode flow-sweep-stream` / `flow-sweep-games5` | 缩小 10× 池的数据效率 A/B | `rl.flow_league.run_flow_sweep(cfg, strategy=..., n_runs=..., pool_scale=..., eval_games=...)` | `rl/run_league.py:1356-1363` |
| eval（非训练） | `--mode eval` | 保存策略轮转对战（换边 + 三态 + 逐局 Elo） | `evaluate_league(policies, kinds, n_games, seed, hidden_dim, max_steps, device)` | `rl/run_league.py:510`；分派 `1298-1304` |

**run 模式内部的三种执行体**（由 `cfg.n_envs` 与 `cfg.parallel` 选择，`rl/run_league.py:1143-1149`）：

| 条件 | 执行体 | 源码 |
|---|---|---|
| `n_envs <= 1` | `_run_single`（单 env 主循环，旧行为） | `rl/run_league.py:697` |
| `n_envs > 1` 且 `parallel == "proc"` | `_run_vec`（单进程批量化：`act_parallel` + batch PPO） | `rl/run_league.py:803` |
| `n_envs > 1` 且 `parallel != "proc"`（默认 `"mp"`） | `_run_mp`（跨进程 worker，主进程批量 GPU 推理） | `rl/run_league.py:942` |

`n_envs>1` 时 `_run_mp` 若 worker 启动抛 `OSError`，**降级回 `_run_single`**（`rl/run_league.py:988-994`）。

### A.1.2 模式之间共享的代码

| 被共用的文件 / 对象 | 共享方式 | 证据 |
|---|---|---|
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

## A.2 一次训练的数据流

以 **solo 主循环**为主干（solo 是当前唯一带完整「观测→信念→计划→策略→奖励→GAE→PPO」链路的模式），
凡 run 模式有差异处单列。全部步骤都给出函数名 + 行号 + 张量形状。

### A.2.0 逐步总览

| # | 阶段 | 函数 | 源码位置 | 输出（形状/类型） |
|---|---|---|---|---|
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

### A.2.1 `RLEnv.reset()` 与观测构造

- `reset()`：按 `deck0_factory`/`deck1_factory` 采样卡组 → `self._rng.shuffle(deck0)` / `shuffle(deck1)`（**原地链式洗牌**：`deck0 = list(self.deck0)` → `shuffle` → `self.deck0 = deck0`）→ 建 `battle.BattleState` → `update_player_hp()` 同步塔血 → 记录本局塔血基准（`_blue_hps_max/_red_hps_max/_blue_towers_max/_red_towers_max`）→ 清掩码缓存 `_mask_fp/_mask_cells` → 清资源账 `_v_share/_active_v` → 设 `_seen_max_id` → 返回 `self.observe(0), {}`（`env_wrapper.py:348-386`）。
- `observe(player_id)` 直接转发 `observation.observe(self.battle, player_id)`（`env_wrapper.py:390-391`）。

**`observation.observe` 的输出字段与维度**（`observation.py:87-144`；常量 `GRID_H, GRID_W, GRID_C = 32, 18, 15` 在 `observation.py:83-84`）：

| 字段 | 形状 | dtype | 构造方式 | 行号 |
|---|---|---|---|---|
| `grid` | `(32, 18, 15)` | float32 | 全零网格；对 `battle.entities` 中存活且名字在 `ENTITY_NAMES` 里的实体，`obs[y][x] = 15 维属性向量`（y 为第一轴） | `observation.py:89-90,124-130` |
| `hand` | `(5,)` | int32 | `p.cycle[:5]` 的 `ENTITY_NAMES.index`（未登记名 → 0） | `observation.py:133-136` |
| `elixir` | `(1,)` | float32 | `p.elixir` | `observation.py:141` |
| `next_card` | `(1,)` | int32 | `ENTITY_NAMES.index(p.cycle[4])`（未登记 → 0） | `observation.py:137,142` |
| `time` | `(1,)` | float32 | `battle.time` | `observation.py:143` |

**grid 的 15 个通道**（逐元素顺序，`observation.py:125-129`）：

| 通道 idx | 名称 | 计算 | 行号 |
|---|---|---|---|
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

### A.2.2 计划与信念 token

| 步骤 | 函数 | 行号 | 说明 |
|---|---|---|---|
| 计划二选一 | `use_prophet = rng.random() < 0.3` | `train_solo.py:1578`（常量 `_SOLO_PROPHET_PROB = 0.3` 在 `train_solo.py:129`；run 模式同 0.3 在 `run_league.py:732`） | 30% 用 `ProphetPlanner.plan(env.get_prophet_state())`，70% 用 `BeliefPlanner.plan(env.battle, belief.state(), obs)` |
| 计划向量化 | `plan.to_vector()` | `train_solo.py:1581` | 维度 `PLAN_DIM`（`follower.py:26` 导入；实算 58） |
| 信念编码 | `belief.encode(obs, None)` | `train_solo.py:1582` | 一维向量，长度 = `belief_dim`（solo 下 563，见 A.3.4） |
| 信念更新 | `belief.update(obs2, info.get("opp_played"))` | `train_solo.py:1596` | 对手出牌来自 `info["opp_played"]`（结构化列表，`env_wrapper.py:490-522,716`） |
| 信念重置 | `belief.reset(env.deck1)` | `train_solo.py:1534` | 每局重置，先验卡组 = 对手本局卡组 |

### A.2.3 策略前向：`FollowerPolicy.act` 的每一步

`act()` 全程 `torch.no_grad()`（`follower.py:439`）。单条路径 `_encode_parts`（`follower.py:268-298`）：

| 步 | 操作 | 张量形状 | 行号 |
|---|---|---|---|
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

### A.2.4 动作解码（autoregressive bundle head）

决策空间常量：`K_MAX = 4`（`follower.py:24` 自 `rl.action_bundle` 导入）、
`ABILITY_IDX = K_MAX = 4`、`STOP_IDX = K_MAX+1 = 5`、`NUM_SLOT_OPTIONS = K_MAX+2 = 6`（`follower.py:37-39`；实算一致）。

`act()` 的解码循环（`follower.py:450-483`，最多 `K_MAX+2 = 6` 步）：

| 步 | 操作 | 形状 | 行号 |
|---|---|---|---|
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

### A.2.5 `env.step()` 与奖励

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
|---|---|---|
| 皇冠差 | `crown_weight * (red_left_old - red_left_new) - crown_lose_weight * (blue_left_old - blue_left_new)` | `239-240` |
| 塔伤（敌） | `+ tower_dmg_opp * red_dmg` | `241` |
| 塔伤（己） | `- tower_dmg_self * blue_dmg` | `242` |
| 圣水 shaping | `+ elixir_bonus * my_elixir_after` | `243` |
| 资源账 Δ费差 | `+ edw * ((me_a - op_a) - (me_b - op_b))`，其中 `me/op = 手牌圣水 + 场上部署份额` | `245-253` |
| 胜负 | `winner==0` → `+win_bonus`；`winner is not None` → `-lose_penalty`；`game_over` 且无胜者 → `-draw_penalty` | `254-260` |
| 非法动作 | `- invalid_penalty * invalid_count` | `261-262` |

- **塔血归一化与差异化定价**：`normalize_tower_dmg` 为真时，per-tower 参数齐备 → `_per_tower_norm_dmg`（`131-161`），否则退化为聚合口径 `dmg * (_TOWER_HP_ANCHOR / hps_max)`（`226-237`）；锚点 `_TOWER_HP_ANCHOR = 2*3052 + 4824 = 10928`（`65-84`）；凹形溢价 `tower_value_mult = 1 + k*(1-ratio)²`，王塔两公主塔存活时再 `* king_gate`（`101-118`）。
- 部署份额记账：`_deploy_ledger`（`561-581`），spell 产物份额记 0 防双算（`568-569`）。

### A.2.6 轨迹缓冲与终端结算

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

### A.2.7 GAE

`PPOTrainer.compute_gae(rewards, values, dones, gamma, lam, truncated=None, last_value=0.0)`（`ppo.py:167-192`）：

- `delta = rewards[t] + gamma * next_val - values[t]`（`188`）；`gae = delta + gamma * lam * (0 if 末步或 done else gae)`（`189`）；`adv[t] = gae`（`190`）。
- `next_val`：末步且 `truncated[t] and not dones[t]` → `last_value`（bootstrap）；否则末步 → 0；非末步 `dones[t]` → 0，否则 `values[t+1]`（`181-187`）。
- 返回 `adv (T,) f32` 与 `returns = adv + values`（`191-192`）。
- 调用点：solo `train_solo.py:1228-1230`（`_gae_pair`）；run `rl/run_league.py:767-768`（single）、`898-900`（vec）、`1091-1093`（mp）。
- solo 额外算 `adv_const`（把 V 换成标量 `c = ppo.ret_scaler.mean` 后重算 GAE）与 `level_gap = |mean(V) - c|`（`train_solo.py:1231-1238`）。

### A.2.8 PPO 更新

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

## A.3 网络结构

唯一策略网络 = `FollowerPolicy`（`follower.py:154`）。构造参数：`hidden=256`（默认）、`plan_dim`、`belief_dim`、`num_entity`、`stop_logit_bias=-1.0`、`value_bypass=False`、`value_independent=False`（`follower.py:155-156`）。

> ⚠️ `plan_dim`/`belief_dim` **无默认值**，两者为 `None` 直接抛 `ValueError`（`follower.py:183-184`）。

### A.3.1 逐层表（`hidden=128`，即训练实际取值）

| # | 层名 | 类型 | 输入维度 | 输出维度 | 激活 | 是否共享 | 源码 |
|---|---|---|---|---|---|---|---|
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

### A.3.2 价值支路 vs 策略支路

**价值支路三变体**（唯一实现 `_value_from`，`follower.py:303-314`；优先级 `independent > bypass > shared`，注释见 `236`）：

| 变体 | 触发条件 | 计算公式 | 价值支路吃到的张量 | 行号 |
|---|---|---|---|---|
| `independent` | `value_independent=True`（无论 `value_bypass`） | `value_head_mlp(value_enc_ln(relu(value_enc_fc(fused))))` | `fused (N,2731)`，**不经过 GRU** | `309-311` |
| `bypass` | `value_independent=False` 且 `value_bypass=True` | `value_head(enc)` | `enc (N,128)`（post-LN 共享编码），跳过 GRU | `312-313` |
| `shared`（默认） | 两者皆 False | `value_head(h)` | GRU 隐状态 `h (N,128)` | `314` |

**策略支路**：`enc → gru_cell → h`，再由 `slot_head(h)` 与 `cell_head(h)` 出分布；每个已选子动作把
`(option one-hot, x/18, y/32)` 经 `sub_emb` 送回 `gru_cell` 更新 `h`（`follower.py:443,454,472,483`）。

**`value_head` 在 `value_independent=True` 下不参与前向**（`_value_from` 分支只走到 `value_head_mlp`，`309-311`）——即权重仍存在、仍会被保存/加载，但不被使用。

### A.3.3 checkpoint 里的架构元数据

`save_checkpoint` 写 `{state_dict, plan_dim, belief_dim, hidden_dim, value_bypass, value_independent}`（`follower.py:55-64`）。
`load_checkpoint`：优先用元数据决定维度；显式传入与元数据不一致时**只告警不报错**（`value_bypass` `94-98`、`value_independent` `100-104`）；
缺 `enc_ln.*` / `grid_ln.*` 的旧 ckpt 显式告警"不可续训（要求 `--fresh`），只能当对照基线/对手池"（`144-150`）；
尾零兼容分支：`plan_mlp.0.weight`（`114-118`）、`belief_mlp.0.weight`（`119-122`）、`entity_emb.weight`（`123-128`）、`plan_mlp.0.bias`/`belief_mlp.0.bias`（`129-130`）；其余形状不匹配保持新初始化不崩（`131`）。

### A.3.4 总参数量（实算）

| 配置 | 总参数量 | 说明 |
|---|---|---|
| 共享 value 通路（`value_bypass=False, value_independent=False`），`hidden=128, plan_dim=58, belief_dim=563` | **634,735** | 含 `enc_fc.weight (128,2731)` 349,568、`cell_head.weight (576,128)` 73,728、`gru_cell.*` 98,304、`belief_mlp.0.weight (64,563)` 36,032 等 |
| `value_bypass=True`（默认 `value_independent=False`） | **634,735** | 只换接线，不新增参数（复用 `value_head`） |
| `value_independent=True` | **993,008** | 额外 `value_enc_fc.weight (128,2731)` 349,568 + `value_enc_ln` 256 + `value_head_mlp.0 (64,128)` 8,192 + `value_head_mlp.2 (1,64)` 64 等，净增 358,273 |

- **实算方法**：`FollowerPolicy(hidden=128, plan_dim=58, belief_dim=563, ...)` 实例化后 `sum(p.numel() for p in model.parameters())`。
- `PLAN_DIM = 58` 来自 `rl/plan_space`（导入于 `follower.py:26`）；`belief_dim = 563` 由 `len(BeliefInference(opp_deck=DEFAULT_SOLO_DECK, n_particles=128, seed=0).encode(None, None))` 实算，与 `train_solo.py:1066-1067` 的口径一致；`NUM_ENTITY = len(ENTITY_NAMES) = 177`（`follower.py:29`，词表 `observation.py:25-73`）。
- **与既有文档数字不一致**：仓库 `AGENTS.md` 中写"629,359"，本文实算为 **634,735 / 993,008**。差异原因**未在源码中找到依据**（可能是不同 `belief_dim`/`num_entity` 下的历史值）。⇒ **以源码实算数为准，差异原因待确认**。

---

## A.4 超参数全表

### A.4.1 `TrainConfig` 全字段（47 个，每个字段都列出）

来源：`rl/config.py:97-241`（`@dataclass class TrainConfig`）；默认值由 `dataclasses.fields` 实读并逐条核对源码行。
「类型」为源码类型注解（Python 不做运行时强制）。

| # | 参数名 | 类型 | 默认值 | 含义 | 来源行号 |
|---|---|---|---|---|---|
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

### A.4.2 预设（`TrainConfig.presets()`）对默认值的覆盖

来源 `rl/config.py:316-369`。下表只列**与 dataclass 默认值不同**的字段。

| 预设 | `reward` 覆盖 | 其它字段覆盖 | 行号 |
|---|---|---|---|
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

### A.4.3 PPO 相关超参：`PPOTrainer.__init__` 的函数签名默认值

来源 `rl/ppo.py:87-90`。**这些默认值与 `TrainConfig` 的同名默认值并不相同**（见"差异"列）。

| 参数 | 签名默认值 | `TrainConfig` 同名默认 | 差异 | 行号 |
|---|---|---|---|---|
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

### A.4.4 枚举 / 字典型字段的全部取值

| 字段 | 全部取值 | 含义 | 来源 |
|---|---|---|---|
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

### A.4.5 奖励权重全表（`DEFAULT_REWARD`，17 键）

来源 `rl/config.py:37-57`；`env_wrapper._DEFAULT_REWARD` 是同值的副本（`rl/env_wrapper.py:40-58`，注释要求"勿单独改一处"）。

| 键 | 默认值 | 含义 | `config.py` 行号 | `env_wrapper.py` 行号 |
|---|---|---|---|---|
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

## A.5 命令行与运行方式

### A.5.1 `run_league.py` 完整 CLI 参数表（54 项）

来源：`argparse.ArgumentParser` 定义 `rl/run_league.py:1170-1293`；分派逻辑 `1298-1371`；
`overrides` 组装 `1307-1344`；`resume = not args.fresh`（`1296`）。
「取值」列中 `flag` 表示 `action="store_true"`（出现即 True，无值）。

| # | 参数名 | 默认值 | 取值 | 含义 | 行号 |
|---|---|---|---|---|---|
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

### A.5.2 命令示例

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

## A.6 检查点 / 目录结构 / 恢复训练

### A.6.1 目录结构与文件清单

所有产物落在 `out_dir/<name>/`（`folder()`，`rl/config.py:244-245`）；`ensure_dirs()` 创建
主目录与 `replays/`（`rl/config.py:286-288`）。

| 文件 | 写入函数 | 内容 / 字段 | 来源 |
|---|---|---|---|
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

### A.6.2 恢复训练（`--resume` / 自动续训 / `--fresh`）

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
|---|---|---|
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

## A.7 本部分待确认清单

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

---

TRAIN_A_OK sections=31 citations=310 trainconfig_fields=47 cli_params=54
