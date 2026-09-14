# 独立校验报告（对三份文档的抽样反查）

> 校验员立场：**唱反调**。目标不是确认文档"大致没错"，而是找出**编造、夸大、与源码不符**之处。
> 未查到的地方一律写"无法判定"，绝不把"我没查"写成"一致"。

---

## 0. 方法与抽样口径

### 0.1 快照（重要：文档在校验期间正在被重写）

校验开始（2026-09-14 22:38）时 `docs/project_full_reference.md` **正被后台进程重写**：连续观测到
`22:38:22`（2,505,771 B，条目 937）与 `22:38:49`（2,506,939 B，条目 940）两个修订，条目数在两次
读取之间变化。为让行号可复算，我在 `22:38:49` 稳定后**冻结快照**并全程只读快照：

```
mkdir -p /tmp/audit_snapshot && cp -p docs/project_full_reference.md docs/training_method.md docs/game_engine.md /tmp/audit_snapshot/
md5sum /tmp/audit_snapshot/*
  13604ba13307fd4ba3a8fe58e7ee7e20  project_full_reference.md   (2,506,939 B, 26,358 行)
  e182b7f4da812c58e34350821e5bff54  training_method.md          (337,971 B, 3,132 行)
  be1ad9c2f60804e65c21bf2c9078b29f  game_engine.md              (128,449 B, 1,450 行)
```

会话中途父代理通知：该文件已**重命名为** `docs/full_code_reference.md`（内容一字未改）。已核对
`md5sum docs/full_code_reference.md` = `13604ba1…`，与本快照**逐位相同** ⇒ 本报告的行号/判定对新路径同样有效。
源码树在校验期间稳定（`src/clasher_new` 无 22:00 后的写入；最新改动是生成文档用的 `scripts/_survey_*.py` 本身）。

### 0.2 第一轮：函数全解文档抽样（40 条）

抽取正则（文档条目形如 `#### 2.13.1 _analyze_pkl [L43-66]`）：

```python
pat = re.compile(r'^#### (2\.\d+\.\d+) (.+?) \[L(\d+)-(\d+)\]')
entries = [m for l in lines if (m := pat.match(l))]
```

**口径更正（必须先说）**：任务书写"每 60 条取 1 条…得到约 40 条"。实测该文档符合此正则的条目共
**937 条**（另 568 条是 `##### 2.x.y.z` 子符号，53 条是 `<顶层执行段>`/`[L-]` 等非标准条目），
`entries[::60]` 只能得到 **16 条**，不是约 40 条（任务书按"约 2400 条"估的）。为达成"抽 40 条"的
实际检验量，我改用**确定性步长 24**（`entries[::24]`，即第 1、25、49… 条），得 **40 条**，
仍然是"按文档顺序等距、与内容无关"的确定性抽样，可复算。

映射章节：章节标题 `^### (2\.\d+) \`(路径)\`` → 条目号 `2.x.y` 归属 `2.x` 的路径；40 条全部成功映射（0 条未映射）。

每题判定依据 = 打开该路径源码、逐行读 `[L起-止]`（用带真实行号的 `awk 'NR>=s&&NR<=e{printf "%d: %s\n",NR,$0}'`），
并对条目内引用的**跨文件行号**（如 `battle.py:1162-1165`）逐条回源码核对 —— 共回核 60+ 处跨文件引用。

### 0.3 第二轮：两份专题文档抽样（15 + 11 条）

按任务指定章节取正文行（非空行），**每 40 行取 1 行**：

- `training_method.md`：域 = §A.3(L330-399)、§A.4(400-557)、§A.5(558-695)、§B.1(792-893)、
  §B.2(894-928)、§B.3(929-976)、§B.4(977-1036)、§B.5(1037-1129) ⇒ 非空 574 行 ⇒ **15 条**。
- `game_engine.md`：域 = §A.2(130-253)、§A.4(380-422)、§B.3(741-807)、§B.4(808-852)、
  §B.5(853-975)、§B.8(1243-1370) ⇒ 非空 414 行 ⇒ **11 条**。

（目标写"各 12 条"，但"每 40 行取 1 行"在两个域上分别给出 15 与 11；我保留该确定性口径，实检 26 条 > 24 条。）
抽样命中纯标题/表格分隔线时（如 `### A.3 网络结构`、`|---|---|`），按**同行域内下一条非空正文行**替换，
替换规则固定、不挑内容；本轮共替换 3 处（TRAIN-01、ENGINE-01、ENGINE-04）。

### 0.4 第三轮：专项攻击

- **4.1 行内数值引用**：扫描三份文档全部 `（xxx.py:NNN）` 形式引用且同行含数字的行，共 **337 条**候选，
  确定性步长 23（`rows[::23]`）取 **15 条**，逐条打开被引源码行核对"数值是否真在那一行"。
- **4.2 "待确认"条目**：从"待确认清单"章节与 `- 置信度: …无法确认…` 行中筛出 **56 条**真实未确认断言，
  确定性步长取 **3 条**，逐条尝试**用源码把它确认掉**。
- **4.3 跨文档一致性**：对 16 个共有关键字（`K_MAX`/`NUM_SLOT_OPTIONS`/`PLAN_DIM`/`belief_dim`/
  `hidden_dim`/`ENTITY_NAMES`/`update_interval`/`solo_copy_every`/`max_ep_steps`/`eval_workers`/
  `max_grad_norm`/`_HIST_POOL_MAX`/`stall_draw_margin`/`n_eval_games`/`steps_per_eval`/`train_stall_stop`）
  程序化抽取三份文档各自出现的数值集合做对照；对高价值常量另用**实跑**取真值
  （`TrainConfig` 字段数、`_DEFAULT_REWARD` 键数、`PLAN_DIM`、`presets()[*].reward` 键数、
  `add_argument` 计数、`action_mask._spell_deals_damage` / `spell_module._deals_damage`）。

### 0.5 判定符号

`一致` / `部分不符(说明哪里)` / `不符(给出源码证据)` / `无法判定(说明为什么)`。

---

## 1. 函数全解文档（40 条，步长 24）

| # | 文档条目 | 源码位置 | 判定 | 证据 / 问题说明 |
|---|---|---|---|---|
| 1 | 2.13.1 `_analyze_pkl [L43-66]` | `scripts/_forensics_cycling.py:43-66` | 一致 | 签名、6 个统计量、边界（`n==0` 除零）逐条对上；`max_steps` 的 `or 360` 兜底在 L51 ✓ |
| 2 | 2.21.6 `_walk [L120-126]` | `scripts/_survey_inventory.py:120-126` | 一致 | 只递归类（L125-126）、不收集嵌套函数 ✓ |
| 3 | 2.26.1 `run_case [L33-82]` | `scripts/assassin_left_bridge_test.py:33-82` | 一致 | 9 段实现描述逐条对上；"`a_alive` 缺 None 保护"（L74）与"907 硬编码"（L69）如实标注 ✓ |
| 4 | 2.37.2 `extractPage.send [L15-19]` | `scripts/cdp_spell.js:15-19` | 一致 | `params={}` 默认值、无 reject 路径 ✓ |
| 5 | 2.39.13 `main [L667-850]` | `scripts/diag_critic_ev.py:667-850` | 一致 | argparse 全部默认值、EV 4 口径、方差结构、探针/换标签/`--save-npz` 逐段对上；"`--out` 全文未使用"经 `grep args.out` 复核为真 ✓ |
| 6 | 2.44.2 `rarity_of [L36-38]` | `scripts/extend_level16.py:36-38` | 一致 | 恒返回 None ✓；`CANON.get(rarity_of(arr),19)`（L31）恒取 19 ✓；`main` 用 `r.get('rarity')`（L49-50）确为正确路径 ✓ |
| 7 | 2.50.1 `S [L24-29]` | `scripts/judge_probe_v4.py:24-29` | 一致 | 双键查找（`str(float(al))` / `f"{al:g}"`）、`v[0]`；调用点 80/84/97/98/109/111 全对 ✓ |
| 8 | 2.53.3 `main [L91-307]` | `scripts/probe_reward_composition.py:91-307` | **部分不符** | 条目主体正确（argparse、5 项占比、判据分支、`res.json` 全对）。但脚注称"`EW.compute_reward` 的**实际**调用点在 `rl/env_wrapper.py:684`"——实际调用表达式起于 **`env_wrapper.py:678`**（`reward = compute_reward(`），L684 只是 `my_elixir_before=…` 这个关键字实参行。**引用的行号不是调用点。** |
| 9 | 2.69.2 `fit [L37-46]` | `scripts/run_probe_v3.sh:37-46` | 一致 | `$LOGABS`（L16，绝对）/`$RELOUT`（L17，相对 `src/clasher_new`）关系正确；调用点 L57/L65 ✓ |
| 10 | 2.74.1 `check [L15-18]` | `scripts/test_m2.py:15-18` | **部分不符** | 函数体逐字正确；但"被本文件全部 28 个测试函数调用（**共约 60 处**）"——实测 `grep -cE '^\s*check\(' = 74` 处（28 个测试函数数正确、异常兜底 L486 正确）。**调用点计数少报 ~19%（74 vs 约 60）。** |
| 11 | 2.74.25 `test_evo_wizard_shield [L291-295]` | `scripts/test_m2.py:291-295` | **部分不符** | 断言式、`75*1.1**10`、`< 2`、`Wizard_EV1` 的 `shieldHitpoints=75`（`gamedata.json:1175-1178`）全对。但描述"`Entity.take_damage` 的**先扣护盾再扣血**（battle.py L540-542）"与实现不符：源码是 **二选一、无溢出** —— `if not self.shield_health: self.hp -= amount` **`else: self.shield_health = max(0, shield - amount)`**（L541-542），护盾期伤害**完全不溢出到血**。措辞会让读者以为"护盾吸收后余量继续扣血"。 |
| 12 | 2.75.15 `test_megaknight_uppercut [L170-180]` | `scripts/test_m3_evo.py:170-180` | 一致 | `__main__` 元组 L473 确有该测试 ✓ |
| 13 | 2.76.7 `test_princess_slow_shot [L56-76]` | `scripts/test_m4_evo7.py:56-76` | 一致 | `_evo_attack_count` 见 `battle.py:963-965` ✓、`Princess_EV1_DeathZone` 见 `battle.py:1044` ✓、`speed_buff/speed_debuff` 见 `battle.py:129-134` ✓ |
| 14 | 2.77.16 `test_mergemaiden_mirror [L266-280]` | `scripts/test_m5_data.py:266-280` | 一致 | Mirror 分支 `battle.py:2806-2823` ✓、`_from_mirror` 形态 `battle.py:2835-2837` ✓（唯一瑕疵：统一扣费在 L2824，恰在所引区间外一行，可忽略） |
| 15 | 2.78.24 `test_icegolemite [L465-484]` | `scripts/test_m6_elite.py:465-484` | 一致 | `HeroIceGolemite` `card_mechanics.py:1542-1560` ✓（`label` 在 L1556，落在所引区间内）；`__main__` L518 ✓ |
| 16 | 2.83.16 `spawn_vines_zone [L2223-2247]` | `src/clasher_new/battle.py:2223-2247` | 一致 | 全部缺省值（2500→2.5、2000ms、ticks 2、crown 25%、`max_targets` 3、`snare_duration=2.0` 硬编码）逐条对上；调用点 `Projectile._on_arrive` 在 L1445 定义、L1450 调用 ✓ |
| 17 | 2.86.6 `Prince [L113-153]` | `src/clasher_new/card_mechanics.py:113-153` | 一致 | 继承行号 `DarkPrince:155`/`BattleRam:158`/`DarkPrinceHeroRhino:1367`/`RamRider:1662` 全对；速度 ×2（L125）、冷却恒 0（L136）✓ |
| 18 | 2.86.32 `open_goblin_window [L877-883]` | `src/clasher_new/card_mechanics.py:877-883` | 一致 | 窗口字典 5 键逐字对上；消费点 `battle.py:3209-3221` ✓；调用点 `card_mechanics.py:1230`（同组最后一只、无 pending）✓ |
| 19 | 2.86.56 `ElectroSpirit [L1630-1659]` | `src/clasher_new/card_mechanics.py:1630-1659` | 一致 | docstring 转述准确；`count=int(chainedHitCount or 9)`、`radius=(chainedHitRadius or 4000)/1000`、`e.is_alive=False` 自毁 ✓ |
| 20 | 2.90.2 `ip_input_screen [L95-109]` | `src/clasher_new/client_side/client.py:95-109` | 一致 | 矩形 (150,310,400,40)、回车/退格/`isprintable` 三分支 ✓；`if not DEBUG` 门控（L117-118）✓ |
| 21 | 2.98.3 `derive_evolved_stats [L98-133]` | `src/clasher_new/evolutions.py:98-133` | **部分不符** | 参数/返回/分支/全部跨文件行号（`card_utils.py:10,16,182,210`、`battle.py:8,71,786,792,797-802,804,1252,1489`）**全部正确**。唯一错误是计数："L117-123 逐字段透传 **18 个**特殊机制字段"——源码元组（L117-121）实为 **17 个**（doc 自己随后也只列出 17 个名字）。**数目与自身列举不一致。** |
| 22 | 2.104.1 `player.PlayerState [L5-58]` | `src/clasher_new/player.py:5-58` | 一致 | 构造默认 `(4824,3052,3052)` ✓、8 个方法（L17/21/27/32/36/41/50/54）✓、调用点 `battle.py:2543/2643-2650`、`environment.py:57-58`、`threat_calc.py:100-101`、`evaluate.py:22-23` 全对 ✓ |
| 23 | 2.107.21 `_ready_ability_cost [L479-497]` | `src/clasher_new/rl/action_mask.py:479-497` | 一致 | 11 步判定逐行对上；`ability_legal` L509、`ability_mana` L518 ✓ |
| 24 | 2.110.11 `_own_region [L206-208]` | `src/clasher_new/rl/belief_planner.py:206-208` | 一致 | 调用点 360/393/442/502/567/576 恰为 **6 处** ✓ |
| 25 | 2.112.12 `_other_side_id [L337-347]` | `src/clasher_new/rl/dashboard.py:337-347` | 一致 | 兜底返回 `pair[1]`、`<2` 返回 None ✓；调用点 L384 ✓ |
| 26 | 2.116.2 `tower_value_mult [L101-118]` | `src/clasher_new/rl/env_wrapper.py:101-118` | 一致 | `DEFAULT_TOWER_PREMIUM_K=2.0`（L97）、`DEFAULT_KING_GATE=0.05`（L98）✓；三个调用点 env_wrapper:157 / action_mask:204,215 / belief_planner:269,299 ✓ |
| 27 | 2.119.4 `build_flow_models [L124-144]` | `src/clasher_new/rl/flow_league.py:124-144` | 一致 | `FLOW_MODEL_IDS` 6 项（L56-57）✓；调用点 L412 ✓ |
| 28 | 2.121.7 `drive_games [L259-288]` | `src/clasher_new/rl/human_play.py:259-288` | 一致 | 默认 `seed=0/max_steps=600`、`slot+1`、`rng.Random(seed+g)`、返回 4 键 ✓；调用点 `human_play.py:330`、`selftest.py:1615` ✓ |
| 29 | 2.124.12 `_Node [L305-320]` | `src/clasher_new/rl/mcts.py:305-320` | 一致 | `__slots__` 恰 8 项、`q()` 除零保护 ✓；实例化 L348/L466 ✓ |
| 30 | 2.131.11 `_opposite_enemy_region [L119-121]` | `src/clasher_new/rl/prophet.py:119-121` | 一致 | 与 `_enemy_region`（L115-116）方向相反 ✓；`LANE_SPLIT_X=9.0`（`belief_planner.py:56`）✓；调用点 L287 ✓ |
| 31 | 2.133.16 `_pair_seed_offset [L393-403]` | `src/clasher_new/rl/run_league.py:393-403` | 一致 | `sha1(...).hexdigest()[:7]`、16 进制 ✓；调用点 L488/L522 ✓；docstring 历史值 `0.5×0.95⁴` 转述准确 ✓ |
| 32 | 2.134.5 `test_hidden_replay_consistency [L130-150]` | `src/clasher_new/rl/selftest.py:130-150` | 一致 | 两步重放、`< 1e-3` ✓；`main()` 调用行 L4893 ✓ |
| 33 | 2.134.29 `test_rlenv_card_level [L872-894]` | `src/clasher_new/rl/selftest.py:872-894` | 一致 | lv16 `(5726,9816)`、lv11 `(3052,4824)`、L889-890 注释、L893 复位全对 ✓；调用行 L4926 ✓ |
| 34 | 2.134.53 `test_draw_penalty_as_loss [L1743-1803]` | `src/clasher_new/rl/selftest.py:1743-1803` | 一致 | ①②③④ 断言与 `Noop` 覆写、`eval_solo(...,2,600,5,...)`、清理 L1802-1803 全对 ✓；调用行 L4917 ✓ |
| 35 | 2.134.77 `test_tower_value_mult [L3490-3519]` | `src/clasher_new/rl/selftest.py:3490-3519` | 一致 | `1+2×0.5²=1.5`、`1+2×0.9025=2.805`、王塔 0.05/0.075、`k=1` 中点 1.25 全部与源码断言式一致 ✓；调用行 L4963 ✓ |
| 36 | 2.136.3 `train_bc [L86-116]` | `src/clasher_new/rl/train_bc.py:86-116` | 一致 | 7 个默认值全对；`__main__` L119-131 的 7 个 argparse 默认值与文档逐一相同 ✓；`follower.py:55/155/713` 引用全对 ✓ |
| 37 | 2.141.3 `resolve_deck_set [L107-126]` | `src/clasher_new/rl/train_solo.py:107-126` | 一致 | 三分支与 `ValueError`（L121-122）✓；调用点 L1062 ✓ |
| 38 | 2.144.4 `pct [L74-75]` | `src/clasher_new/runs/_tmp_behavior_recount.py:74-75` | 一致 | `s[min(len(s)-1,int(q*len(s)))]`、空表 IndexError ✓；调用点 L101（6 分位）/L104（p90,p99）/L182（p10,p90）全对 ✓ |
| 39 | 2.156.1 `_tower_report [L50-56]` | `src/clasher_new/simulate_exchange.py:50-56` | 一致 | 函数体、`_TOWER_IDS`(L45)/`_TOWER_NAMES`(L46)、"本文件内无调用点"（grep 仅命中定义）、`_report` 闭包 L261 ✓ |
| 40 | 2.166.2 `WeightsCopyingCallback [L61-68]` | `src/clasher_new/train.py:61-68` | 一致 | 裸名 `opponent`（L67）⇒ 到 50000 步 NameError 推断正确；callback 列表只含 `CheckpointCallback`（L131-133）✓；全 .py 中该类仅 L61 定义处出现 ✓ |

**第一轮小结：36 条 `一致`，4 条 `部分不符`（#8/#10/#11/#21），0 条 `不符`，0 条 `无法判定`。**

---

## 2. 训练方法文档（15 条，每 40 行 1 条）

| # | 文档条目 | 源码位置 | 判定 | 证据 / 问题说明 |
|---|---|---|---|---|
| T1 | A.3 首句：`FollowerPolicy` 构造参数 `hidden=256`(默认)、`stop_logit_bias=-1.0`、`value_bypass/value_independent=False` | `rl/follower.py:154-156` | 一致 | 签名逐字相同 ✓（`plan_dim/belief_dim` 无默认值也正确，L183-184 抛 ValueError） |
| T2 | `load_checkpoint` 元数据不一致**只告警不报错**；`value_bypass` 94-98、`value_independent` 100-104；缺 `enc_ln/grid_ln` 告警 144-150；尾零兼容 114-118/119-122/123-128/129-130/131 | `rl/follower.py:86-152` | 一致 | 5 个行号区间与 5 条分支**逐行对上**；确为 `print_safe` 无 raise ✓ |
| T3 | `max_grad_norm` float 默认 `0.5` | `rl/config.py:132` | 一致 | `max_grad_norm: float = 0.5` ✓ |
| T4 | `lockdown` 预设 6 个奖励覆盖值 `8.0/10.0/10.0/0.05/0.0/0.05` | `rl/config.py:332-337` | 一致 | L335-337 逐值与文档相同 ✓ |
| T5 | `parallel` 取 `mp`/`proc` | `rl/config.py:124`；`rl/run_league.py:1145-1148` | 一致 | 默认 `"mp"`、注释语义相同 ✓；分派 `if n_envs>1: proc→_run_vec / else→_run_mp / else→_run_single`（L1145-1149）✓ |
| T6 | CLI `--max-steps` 默认 `600` | `rl/run_league.py:1178` | 一致 | `ap.add_argument("--max-steps", type=int, default=600)` ✓ |
| T7 | CLI `--hist-seed-dir` 默认 `None`、`action="append"` | `rl/run_league.py:1262-1264` | 一致 | 逐字相同 ✓ |
| T8 | `start_rl.bat` 默认 `EVAL_WORKERS=16`、`SOLO_COPY_EVERY=2000`；`--opt=value` 形式见 `start_rl.bat:244` | `start_rl.bat:45,46,244` | 一致 | `set "EVAL_WORKERS=16"`(L45)、`set "SOLO_COPY_EVERY=2000"`(L46)；L243 注释 + L244 正是 `--mode=… --opt=value` 拼接行 ✓ |
| T9 | `belief.encode(obs, None)` 见 `train_solo.py:1582`；粒子数 128 见 `:1171` | `rl/train_solo.py:1582,1171` | 一致 | L1582 `belief_tok = belief.encode(obs, None)`；L1171 `n_particles=128` ✓ |
| T10 | `_persist(step)` 位于 `rl/train_solo.py:1410-1452` | `rl/train_solo.py:1410-1452` | 一致 | L1410 `def _persist(step):`，L1452 为该函数最后一行（L1454 已是下一函数 `anchor_point`）✓ |
| T11 | `compute_gae` 函数默认 `gamma=0.99, lam=0.95` | `rl/ppo.py:168` | 一致 | `def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95, …)` ✓ |
| T12 | `_update_epochs`：外层 `:498`、`_loss_pass` `:505-506`、诊断仅 `(ep==0,bi==0)` `:507-513`、`_apply_grad` `:514`、`grad_steps` `:517` | `rl/ppo.py:477-547` | 一致 | 5 处行号**逐行命中**，含 L508 `_diag_here = diag_on and ep == 0 and bi == 0` ✓ |
| T13 | `--mode` 6 个取值（1171-1173）；`run_league` 分派（1143-1149）；`evaluate_league(..., max_steps=600, device="auto")` 定义于 510-532 | `rl/run_league.py:1171-1173,1143-1149,510` | 一致 | L1171-1173 六个取值逐字相同；L510 签名逐字相同 ✓ |
| T14 | `defense_invest_rate`：敌过河 = P1 troop `y < 16.0`（`RIVER`），行号 `:573/:606/:699` | `rl/train_solo.py:573,606,699` | 一致 | L573 `RIVER = 16.0`；L606 `foes_crossed = any(y < RIVER …)`；L699 `"defense_invest_rate": _pct(def_deploy_frames, def_frames)` ✓（顺带发现源码自身矛盾：L601 注释写 `RIVER=15`，与 L573 的 16.0 不符——这是**源码注释**的错，文档此处反倒是对的） |
| T15 | `_check_gates` 在 `train_solo.py:234-306`、`_write_gate_report` 在 `:309-317` | `rl/train_solo.py:234-306,309-317` | 一致 | L234 `def _check_gates`，L306 `return report`；L309 `def _write_gate_report`，L317 收尾 ✓ |

**第二轮（训练）小结：15 条全部 `一致`。**

---

## 3. 游戏引擎文档（11 条，每 40 行 1 条）

| # | 文档条目 | 源码位置 | 判定 | 证据 / 问题说明 |
|---|---|---|---|---|
| E1 | A.2 首表：`TileGrid.width, height = 18, 32`（arena.py:9）、`tile_size=100.0`（:10）、合法域 `0<=x<18 / 0<=y<32`（:100-101）、塔几何 3×3 / 4×4（:37-38） | `arena.py:9,10,100-101,37-38` | 一致 | 逐项对上 ✓ |
| E2 | 红右公主塔 `(14.5,25.5)`、半宽高 `1.5/1.5`、`player_id=1`、依据 `arena.py:44` | `arena.py:16,44` | 一致 | `RED_RIGHT_TOWER = Position(14.5,25.5)`（:16）；`towers[4] = (RED_RIGHT_TOWER, 1.5, 1.5, 1)`（:44）；列名确认是 **`player_id`** 而非实体 id ✓ |
| E3 | 观测张量 `(32,18,15)`（environment.py:42,126）、`x,y=int(position.x),int(position.y)`（:146,:152）、`obs[y][x]`、网格下标=向下取整 | `environment.py:42,126,146,152` | 一致 | 四处全中；镜像分支在 L147-149（下一句才讲，不影响）✓ |
| E4 | `step` 的 `dt` 唯一来源 `self.battle.step(1/60)`（environment.py:102）；每决策最多 `range(30)` 且先判 `game_over` break（:98-100）；30×1/60=0.5 s/决策（:80-81） | `environment.py:80-81,98-102` | 一致 | L102 `self.battle.step(1/60)` 位于 `for j in range(int(self.speed))` 内（speed 缩放次数，dt 值不变）✓；L98-100 与 L80 docstring 均对上 ✓ |
| E5 | 空 goals 兜底（扫描区内最近可达格，全无则目标格）见 `pathfinding_heap.py:85-101` | `pathfinding_heap.py:85-101` | 一致 | L85 `if not self.goals:` … L101 `self.goals.add(target_cell)` ✓ |
| E6 | `.` 面代价：heap 版 **5**（`:140`）vs 旧版 **8**（`pathfinding.py:109-110`） | `pathfinding_heap.py:140`；`pathfinding.py:109-110` | 一致 | L135 `elif tile_char == '.':` → L140 `tile_cost = 5` ✓；`pathfinding.py:110 tile_cost = 8` ✓ |
| E7 | 塔兵名 `King_Cannon/Knife/ChefTowers` **不含** `'PrincessTower'` ⇒ 不吃 `bonus=0.5`，`update_current_target` 的脱距保留分支（`battle.py:684`）也不生效 | `battle.py:588-591,604,684`；`battle.py:2554-2559` | 一致 | L588 `if 'PrincessTower' in target.name: bonus=0.5`；L684 同款子串判定；`King_CannonTowers` 三变体确实都不含 `'PrincessTower'`（名源 `player.py:23`），也不含 `'KingTower'` ✓ |
| E8 | `HeroKnight.use_ability` 把 6.5 格内敌军 `_taunt_until = time + tauntDuration` 并清 `path`（`card_mechanics.py:908-922`）；覆盖点在 `battle.py:729-736` | `card_mechanics.py:908-918`；`elite17_data.py:73`；`battle.py:729-736` | 一致 | L915 `tauntRadius + collision_radius`；L916-918 赋值/清 path；`tauntRadius=6.5`、`tauntDuration=5.0` ✓；`_hero_taunt_override` 在 729-736 ✓ |
| E9 | `eff = _effective_card(p, card_name)`（Mirror→`last_card`）、`eff_info = Card(eff)`（`action_mask.py:452-455`） | `action_mask.py:452-455` | 一致 | L452-455 逐行命中 ✓ |
| E10 | 手牌/圣水/王塔：引擎 `can_play_card`（`player.py:36-39`）vs 掩码 `_slot_playable`（`action_mask.py:46-56`），"一致（除 Mirror 动态费）" | `player.py:36-39`；`action_mask.py:46-56` | 一致 | `can_play_card`：`cycle[:4]` + `elixir >= Card.elixir` + `king_tower_hp>0` ✓；`_slot_playable` 同三条件 + `_card_cost`（Mirror 走动态费）✓ |
| E11 | `_spell_deals_damage` vs `spell_module._deals_damage` 的差异表：`Zap/Poison/Tornado/Earthquake/Freeze/Heal` = **False / True**；`Arrows/Fireball` = True/True；`Lightning/Rage/BarbLog/Log/GoblinBarrel/Mirror` = False/False | `rl/action_mask.py:89-94`；`spell_module.py:54-61` | 一致 | **实跑两个函数**（见 §0.4）：对 14 张卡逐张打印，返回值与表**完全一致**（`action_mask` 只看 `Card.data` 的 `projectileData.damage`/顶层 `damage`，故 6 张区域/增益法术为 False；`spell_module` 另查 spells 行 `damage`/`buff_data.damage_per_second`，故 6 张为 True）✓ |

**第三轮（引擎）小结：11 条全部 `一致`。**

---

## 4. 专项攻击结果

### 4.1 行内引用的数值（15 条）

抽样：三份文档中形如 `（xxx.py:NNN）` 且同行含数字的候选 **337 条**，确定性步长 23 取 15 条，
逐条打开被引源码行核对。

| # | 文档出处（文档行） | 被引位置 | 断言中的数值/事实 | 源码核对结果 | 判定 |
|---|---|---|---|---|---|
| N1 | `training_method.md:794` | `rl/train_solo.py:1039,1541` | `run_solo(cfg,resume,record_replays)`；主循环 `range(start_step+1,total_steps+1)` | L1039 签名逐字 ✓；L1541 循环逐字 ✓ | 一致 |
| N2 | `training_method.md:851` | `rl/config.py:126`；`rl/train_solo.py:1482-1484` | `eval_at_start` 默认 True；起始评估 `eval_and_write(0)` | L126 `eval_at_start: bool = True` ✓；L1482-1484 `if cfg.eval_at_start and start_step==0: …eval_and_write(0)` ✓ | 一致 |
| N3 | `training_method.md:922` | `rl/ppo.py:245-248` | `none/""/None` ⇒ `v_scale=1.0`，其它抛 `ValueError` | L245-246 `v_scale = 1.0`；L247-248 `raise ValueError` ✓ | 一致 |
| N4 | `training_method.md:966` | `rl/ppo.py:440-453` | `grad_cos = num/(p_gnorm*na)`、`grad_norm_ratio = p_gnorm/na` | L451-453 逐字 ✓ | 一致 |
| N5 | `training_method.md:1119` | `rl/run_league.py:590`；`rl/train_solo.py:1072-1083` | `_load_run_state`；solo 恢复 `start_step`/`history` | L590 `def _load_run_state(cfg):` ✓；L1072 `if resume:`、L1073 `rs=_load_run_state(cfg)` ✓ | 一致 |
| N6 | `training_method.md:3053` | `rl/config.py:209`；`rl/run_league.py:1246-1248`；`rl/config.py:344-364` | dataclass 默认 `min(16,os.cpu_count() or 1)`，而 CLI help 写"默认 0=串行"，`economy` 未设该字段 | L209 逐字 ✓；L1248 help 确写"默认 0=串行"（**两处确实自相矛盾，文档如实披露**）✓；`economy` 预设 L344-364 无 `eval_workers` ✓ | 一致 |
| N7 | `game_engine.md:142` | `arena.py:150,152`（另 :11,:14） | 注释称玩家 0 为 "bottom half"，但区间是 `y=1..14` | L150 `if player_id == 0:  # Player 0 (bottom half)` ✓；L152 `zones.append((0,1,width,RIVER_Y1))  # y=1 to y=14` ✓ | 一致 |
| N8 | `game_engine.md:212` | `arena.py:92-94,96-98` | 王塔后禁放带 `(7.0,0.0,11.0,1.0)` / `(7.0,31.0,11.0,32.0)`；判定半开区间 | L93-94 两个元组逐字 ✓；L96-98 `x1<=pos.x<x2 and y1<=pos.y<y2` ✓ | 一致 |
| N9 | `game_engine.md:336` | `battle.py:738-746`；`battle.py:2770-2773` | `Entity.create_projectile` 与 `spawn_projectile_chain` 是绕过 `ensure_walkability` 的两条旁路 | L738-746 直接 `entities[id]=projectile` + `next_entity_id+=1` ✓；L2770-2773 同款 ✓ | 一致 |
| N10 | `game_engine.md:347` | `battle.py:276-323,298-302,303` | `_generic_death_spawn`；炸弹型（有 `deathDamage` 无 `hitpoints`）→ `TimedExplosive`；SkeletonBalloon/Container 走 7 个延迟 Skeleton | L276 定义 ✓；L298-302 条件与构造 ✓；L303-304 `_cnt = int(deathSpawnCount or 7)` ✓ | 一致 |
| N11 | `game_engine.md:373` | `card_utils.py:226,227,239,240` | `hp`、`elixir=manaCost`、`collision_radius/1000`、`hit_speed/1000` | 四行逐字命中 ✓ | 一致 |
| N12 | `game_engine.md:478` | `core.py:53-56` | 目标名含 `'King'`/`'PrincessTower'` ⇒ `damage*tower_damage_mult`，否则原伤害 | L53-56 逐字 ✓ | 一致 |
| N13 | `game_engine.md:556` | `arena.py:154-170`；`battle.py:2872-2887` | 塔破后追加敌半场 4 格回推区；`deploy_card` 另有一份塔血判定 | L156-160 / L166-170 追加区 ✓；L2872-2887 `left/right_tower_hp>0` 判定 ✓ | 一致 |
| N14 | `game_engine.md:603` | `player.py:6,39`；`battle.py:2654` | 默认塔血 `(4824,3052,3052)` 是 lv11 值；首次 `step` 覆盖前 `can_play_card` 读默认值 | L6 默认元组 ✓；L39 `king_tower_hp > 0` ✓；L2654 `self.update_player_hp()` 是 `step` 内第一动作 ✓ | 一致 |
| N15 | `full_code_reference.md:8655`（原 `project_full_reference.md`） | `card_mechanics.py:1562-1579` | `HERO_CLASSES` 共 **16 项** | L1562-1579 恰 **16** 个映射（Knight…IceGolemite）✓ | 一致 |

**4.1 小结：15/15 一致。**这一项文档表现很好——行号级引用没有抓到错位。

### 4.2 过度保守的"待确认"（3 条）

抽样：从"待确认清单"章节与 `- 置信度: …无法确认…` 行筛出 56 条，取 3 条尝试确认。

**① ★实锤：`core.BlankEntity` 的"全仓无使用点"是错的（不是过度保守，是事实错误）**

- 文档（`full_code_reference.md:11194-11195`）：
  > `- 调用: 仓库内 grep BlankEntity 只命中本文件 ⇒ **未发现在别处被实例化或引用**（无法确认它的实际消费方）。`
  > `- 置信度: 已确认（定义与无引用的事实）；其设计意图的消费方无法确认（原因：全仓无使用点）`
- 源码（我实跑 `grep -rn BlankEntity --include=*.py .`）：
  ```
  src/clasher_new/battle.py:1:from core import BlankEntity          ← 导入
  src/clasher_new/battle.py:1670:        self.target = BlankEntity(target_position)   ← 实例化
  src/clasher_new/battle.py:2964:            target = BlankEntity(position)           ← 实例化
  src/clasher_new/core.py:11:class BlankEntity:
  ```
- 判定：**`不符`**。文档声称的"grep 只命中本文件""全仓无使用点"与事实矛盾；它把一条**可确认**的事实写成了"无法确认"，而且给出的原因（无使用点）是**编造/错误的**。这是本次校验发现的**最严重一条**：不是"漏读"，是"断言了一个不存在的 grep 结果"。

**② 过度保守：`scripts/test_m3_evo.py` 的 `spawn_building` 第 5 个位置参数**

- 文档（`full_code_reference.md:5587`）：`置信度: 已确认（第 5 个位置参数 False 的具体语义无法确认，原因：未读取 Building.__init__ 定义）`
- 源码：`scripts/test_m3_evo.py:41` `Building(bs.next_entity_id, Position(x, y), player, card, False, evolved=evolved)`；
  `battle.py:1234` `def __init__(self, id, position, player, card_name, persistent=False, evolved=False)`
  ⇒ 第 5 个实参 `False` = **`persistent=False`**（非持久塔标记，L1240 `self.persistent = persistent`）。
- 判定：**过度保守**。只差打开一个已知的 `__init__` 定义即可确认，文档却以"未读取"留白。

**③ 过度保守/自相矛盾：`game_engine.md` A.7 第 4 项 vs 同一文档 §B.8**

- A.7 第 4 项（`game_engine.md:596`）：
  > `部署合法性由哪一层兜底 | battle.deploy_card 的部队/建筑路径不检查 BLOCKED_TILES/河道…只有 arena.can_deploy_at 检查…；RL 掩码层是否补齐无法确认 | 读 RL 侧掩码实现…，不在本轮清单`
- 但**同一份文档的 §B.8.2（L1252-1288）与 §B.8.4（L1300-1358）** 正是逐格列出
  `legal_cells` / `_position_legal` 的判定流程、并逐项对照引擎 `deploy_card`（含"掩码更严：无 Miner 例外"）。
- 判定：**过度保守 + 文档内自相矛盾**。A.7 把 §B.8 已经回答的事实标为"无法确认"。

### 4.3 跨文档一致性

**方法（可复算）**：
1. 对 16 个共有关键常量，程序化抽取三份文档各自出现的数值集合，逐 key 对照（脚本见 §0.4 描述）；
2. 对可执行的高价值量用**实跑源码**取真值：`TrainConfig` 字段数、`_DEFAULT_REWARD` 键数（两份）、
   `PLAN_DIM`、`presets()[*].reward` 键数、`run_league.py` 的 `add_argument` 计数；
3. 逐条比对各文档对同一常量的**文字断言**。

**结论 A：关键常量在三份文档之间没有互相打架。** 以实跑真值对齐后，三份文档一致的有：
`max_grad_norm=0.5`（config.py:132）、`update_interval=128`、`solo_copy_every=2000`、`max_ep_steps=360`、
`RAND_ANCHOR_EVAL_SEED=90000`、`_HIST_POOL_MAX=12`、`PLAN_DIM=58`（实跑确认 58；三份文档都把源码
docstring 里的"57"如实标注为**源码内部口径漂移**，非文档错误）、`belief_dim=563`、`hidden_dim=128`
（config 默认；`FollowerPolicy` 自身默认 256 也被正确区分）、`ENTITY_NAMES/NUM_ENTITY=177`、
`HERO_CLASSES=16 项`、`eval_workers=min(16,os.cpu_count() or 1)`、`TrainConfig` **47 字段**
（实跑 `len(dataclasses.fields(TrainConfig))==47`）、`run_league.py` **54 个 `add_argument`**
（实跑 `grep -c add_argument = 54`，与 training_method「54 项」一致）。

**结论 B：抓到 2 处"同一事实数值不一致"（均为文档内部/文档↔源码，非两文档间）：**

| # | 位置 | 文档说法 | 源码真值 | 判定 |
|---|---|---|---|---|
| B1 | `training_method.md:534` | "`env_wrapper._DEFAULT_REWARD` 是**同值的副本**" | `rl/config.py:37-57 DEFAULT_REWARD` = **17 键**（含 `draw_penalty`）；`rl/env_wrapper.py:40-58 _DEFAULT_REWARD` = **16 键**（**无** `draw_penalty`）；实跑 `config==env` 为 `False`，差集恰为 `{'draw_penalty'}` | **不符**。且**同一文档自己的表格**（L546）在 `draw_penalty` 行的 env_wrapper 列写"—（在 config.py 有…）"，与该句"同值副本"直接矛盾。full_ref（L13960）反而正确地只把 17 键归给 config，并如实转录 config 里"与 RLEnv 保持一致"的源码注释 |
| B2 | `full_code_reference.md:14031` | 同一行内先写"实测 `aggressive.reward` 仅 **7** 键…见下 `presets` 条目实测：`aggressive` 仅 **6** 键" | 实跑：`aggressive.reward` = **6** 键（crown_weight/win_bonus/lose_penalty/invalid_penalty/elixir_bonus/elixir_diff_weight）；7 键的其实是 `economy` | **部分不符**：一行内 7/6 自相矛盾，正确值是 6 |

**附带（未计入缺陷，但读者须知）**：源码自身有两处内部矛盾，文档均已**如实披露**而非掩盖——
(a) `rl/run_league.py:1248` 的 `--eval-workers` help 写"默认 0=串行"，与 `rl/config.py:209` 的
`min(16, os.cpu_count() or 1)` 冲突；(b) `rl/train_solo.py:601` 注释写 `RIVER=15`，而 `:573` 定义为 `16.0`。

---

## 5. 结论

### 5.1 发现的不符项：9 条（其中 4 条为第一轮判定的"部分不符"，1 条严重）

**最严重 3 条（按严重度排序）：**

1. **★ `core.BlankEntity` 的"全仓无使用点"是伪造的 grep 结论（§4.2 ①）**
   文档宣称"仓库内 grep BlankEntity 只命中本文件 ⇒ 未发现在别处被实例化或引用"，并据此把一条
   **可确认**事实标为"无法确认"。实际 `battle.py:1` 导入它、`:1670` 与 `:2964` 两次实例化它。
   → 这是**唯一一条"编造证据"级**的问题：不是漏读，而是写了一个不存在的检索结果。

2. **`training_method.md:534` 的"`env_wrapper._DEFAULT_REWARD` 是同值的副本"（§4.3 B1）**
   两份奖励表**不等值**（17 键 vs 16 键，差 `draw_penalty`），且被同一文档 L546 的表格自我否定。
   这是最容易被下游照抄的一类错误（改奖励常量时按"同值副本"处理会漏改）。

3. **`battle.py:540-542` 的 `take_damage` 被描述成"先扣护盾再扣血"（§1 #11）**
   源码是 **"有盾只扣盾、无盾才扣血，伤害不溢出"**（`if not self.shield_health: self.hp -= amount`
   `else: self.shield_health = max(0, shield - amount)`）。"先扣护盾再扣血"这个措辞会让读者以为
   护盾吸收后余量会继续扣血——对一条被三份文档引用的伤害入口，这是一个会误导实现的语义错误。
   （同一档次的还有 §4.3 B2：`full_code_reference.md:14031` 同一行内 7 键/6 键自相矛盾，正确值是 6。）

**其余 5 条（轻微，均为计数/行号层面的不精确）：**
- `_analyze_pkl` 条目所属文档把 `env_wrapper.py:684` 称作 `compute_reward` 的"实际调用点"——实为 **L678**（§1 #8）；
- `test_m2.py` 的 `check()` 调用点"约 60 处"——实测 **74 处**（§1 #10）；
- `derive_evolved_stats` 特殊字段"**18 个**"——源码元组实为 **17 个**，文档自己随后也只列 17 个（§1 #21）；
- `spawn_building` 第 5 实参语义被标"无法确认"——读 `battle.py:1234` 即为 `persistent`（§4.2 ②）；
- `game_engine.md` A.7 #4 把同文档 §B.8 已回答的"掩码层是否补齐"标为"无法确认"（§4.2 ③）。

### 5.2 是否有编造证据：**有（1 条，性质明确）**

`full_code_reference.md:11194-11195` 的 `BlankEntity` 条目：
文档原文 —— "仓库内 `grep BlankEntity` 只命中本文件 ⇒ **未发现在别处被实例化或引用**"；
源码原文 —— `battle.py:1 from core import BlankEntity` / `battle.py:1670 self.target = BlankEntity(target_position)` /
`battle.py:2964 target = BlankEntity(position)`。
**该 grep 结论与仓库事实相反**，属编造/臆测而非"保守留白"。

除这一条外，未发现其他"说了源码里没有的行为""把 if 分支说反""编造常量数值"的情况；
第一轮 40 条里 4 条是计数与措辞不精确，无一是凭空捏造的实现逻辑。

### 5.3 对文档质量的总体判读（不客气版）

- **亮点（客观成立，不是恭维）**：**跨文件行号引用极准**。§4.1 的 15/15、第一轮 60+ 处跨文件引用、
  第二/三轮 26 条里大量 `xxx.py:NNNN-NNNN` 区间，**没有抓到一处系统性错位**（唯一一处 #8 是 6 行偏差）。
  这在 2.5 MB 逐符号文档里属于少见的高水位。
- **风险集中在两处**：(a) **"grep 已核"类断言**（越界的元断言——`BlankEntity` 一条已经证伪，
  说明这类断言没有逐条复算）；(b) **计数与"同值/副本"类等价性断言**（18 vs 17、60 vs 74、17 vs 16 键，
  都是"看一眼就算数"的地方）。建议对全部"共 N 项/共 M 处/同值副本/全仓无引用"类句子做一次脚本复算。
- **两份专题文档明显更干净**：训练 15/15、引擎 11/11 全对，包括最容易错的"数值常量 / 坐标区间 /
  阶段顺序 / 函数名"四类；`_spell_deals_damage` 对照表这种"两面函数逐一枚举"的断言**实跑也对**。

### 5.4 我对自己这份校验的置信度与局限

**置信度：高（对已抽到的 66 条 + 15 条数值引用 + 3 条待确认）。**
所有判定都基于**亲读的源码行号**；关键的"可执行断言"（`_spell_deals_damage`/`_deals_damage`、
`PLAN_DIM`、两份 `DEFAULT_REWARD` 键集、`TrainConfig` 字段数、`presets()[*].reward`）是**实跑取值**，
不依赖文档自述。报告给出可复算脚本与 md5，任何人都能回到同一快照重放。

**局限（必须明说）：**
1. **不是全量**：937 条符号只抽了 40 条（≈4.3%）；专题文档分别抽 15/11 条（§A.3/A.4/A.5/B.1–B.5 与
   §A.2/A.4/B.3–B.5/B.8 的正文行），**没有覆盖** §A.1/A.2/A.6/A.7、B.6–B.9、以及 `#####` 子符号
   （568 条）。因此"未发现更多不符"**不能**外推为"全文正确"。
2. **文档在被重写**：我只对 md5 `13604ba1…` 的那个修订负责；父代理随后又把文件重命名并可能继续重建
   （快照前 40 秒内它变了两次）。若内容再变，行号需重新对齐。
3. **§4.2 的 3 条是抽样**，56 条真实"待确认"里还有大量条目未验证；已发现的这类问题可能只是冰山一角。
4. **不执行训练/不跑引擎**：我只做静态读取 + 少量纯计算函数的实跑（`_spell_deals_damage` 等），
   **没有**启动训练、渲染或对局，因此"实现描述与运行期行为是否一致"（如掩码与引擎位图对账）**未验证**。
5. **`full_code_reference.md` 里大量"（未读 / 不在允许清单内）"的留白**（如 flow 模式、`rl/league.py`
   正文）我**没有**逐条去补读确认，只抽了 3 条；`training_method` 的自述范围限制我照单接受，未越界判其错。
