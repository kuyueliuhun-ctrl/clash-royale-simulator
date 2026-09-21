# 预注册：**把觉醒/进化（evolution）接进 RL/IL 通路 + 放进观测**（2026-09-22，用户拍板「是时候去做了」）

> 用户指令（2026-09-22）：「对于觉醒，我认为是时候去做了。」
> 纪律：判据**跑前写死**（R3）、观测/架构变更必须 `--fresh`（R6）、扩参与新输入按**逐项性价比**（R11）、
> 不动奖励表（R7）、掩码改动走位图对账（R13）、每个修复配回归测试（R8）。
> 本文件先落**判据与失败分支**，再实现；实现进度在各阶段小节里回写。

## §0 一句话

引擎**已经有**觉醒（`evolutions.py` + `battle.py` 触发），但**RL/IL 通路从来没有声明过觉醒位**
（`rl/` 对 `evo_slots/evo_plays/hero_slots/evolution_state` 的引用数 = **0**）⇒ 我们所有对局里
**觉醒永不发生**；而人类回放的牌组**带着** `-ev1/-ev2/-ev3/-hero` 后缀 ⇒ **重建战局与人类真实战局不同**
（数据保真度缺口），且 `observe()` 只有 `grid/hand/elixir/next_card/time` 五键、
**没有任何觉醒状态**（= 一旦觉醒生效，网络会缺输入）。

## §1 已取证事实（来源见括号；不确定的一律标未定）

| # | 事实 | 证据 |
|---|---|---|
| 1 | 引擎**有**觉醒：`EVOLUTION_CYCLES` 42 张（cycle=1/2）、`evolution_state(plays, card) = plays % (cycle+1) == cycle`、`derive_evolved_stats` | `src/clasher_new/evolutions.py:11-30, 89-95, 98-133` |
| 2 | 触发条件：`card_info.evo_raw and card_name in _p.evo_slots and card_name not in _p.hero_slots and evolution_state(_p.evo_plays[card], card)`；计数在 `battle.py:2934-2936`；挂载 `_apply_evolution` | `src/clasher_new/battle.py:3068-3070, 840-841, 846-899` |
| 3 | 槽位 API：`PlayerState.evo_slots / evo_plays / hero_slots` + `set_evolution_slots` | `src/clasher_new/player.py:12-19` |
| 4 | 数据源：`card_utils.py:356 evo_raw = data.get('evolvedSpellsData')`；`gamedata.json` 里 `evolvedSpellsData` 出现 **34** 次；另有 `evo_2025_data.py` 自建 7 张 | `card_utils.py:356` + 计数 |
| 5 | **RL 侧零引用**：`rl/` 目录对 `evo_slots/evo_plays/hero_slots/evolution_state/evolvedSpellsData` 的引用数 = **0** ⇒ 本仓对局里觉醒永不触发 | 全目录 grep |
| 6 | **观测无觉醒字段**：`observe()` 只返回 `grid / hand / elixir / next_card / time` | `src/clasher_new/rl/observation.py:138-144` |
| 7 | 人类回放牌组键**带**变体后缀（`elite-barbarians-ev1` 1007、`zap-ev1` 589、`berserker-hero` 1274 …），但**事件流不标注**哪一次是觉醒形态（`form_at_play` 全 `"unknown"`）⇒ 只能用 `card_play_number` + `EVOLUTION_CYCLES` **推导** | FL jsonl 键树 + 抽样 200 局 15,900 条事件 |
| 8 | IL 数据管线 `fl_il_to_bc.py` **没有** `set_evolution_slots` ⇒ 重建战局按基础卡跑（与人类真实战局不一致） | 代码 + 本文件 §2 S1 |
| 9 | 未定：**本仓 `gamedata` 的 34 张觉醒形态是否覆盖回放里出现的全部后缀**（`-ev1/-ev2/-ev3/-hero`）；S1 第一步就是测这个覆盖率 | 待测 |

## §2 阶段与判据（每阶段独立可验；判据写死）

### S1（数据保真）—— `fl_il_to_bc.py` 解析回放牌组后缀并在重建战局里声明觉醒位

* 做法：从回放 `decks[side][i][0]`（`kebab-case` + 可选 `-ev1/-ev2/-ev3/-hero`）解析出
  (本仓卡名, 觉醒档/英雄位)，对双方 `PlayerState` 调 `set_evolution_slots(...)`；
  **仅当卡名与档位都能映射**时才声明，映射不了的走**原路径**（不静默改卡）。
* 判据（跑前写死）：
  1. **覆盖率**：回放里出现的 `-ev*/-hero` 后缀，本仓觉醒表**能映射的比例 ≥ 95%**（未映射的逐条列出）；
  2. **不变量**：同一 200 局重跑，`labels` 与旧读数之差 **≤ 5%**（= 掩码/跨进程漂移量级），`errors = 0`；
  3. **生效性**：同 200 局里**觉醒触发次数 > 0**（当前必然为 0），并给出「声明觉醒位的局数 / 总局数」；
  4. 失败分支：若覆盖率 < 95% ⇒ 先补映射表（不动机制）；若 `labels` 漂移 > 5% ⇒ 判**重建语义变了**，
     停手并回到本文件讨论（不许把数据管线改动混进模型结论）。

### S2（观测加字段）—— `observe()` 增加觉醒状态（**R6：观察维变 ⇒ 必须 `--fresh`**）

* 建议字段（最小集，维度显式）：
  * `evo_slots_mask`：本局我方声明了觉醒位的卡（8 维 0/1，或 1 维「本卡是否带觉醒档」并入现有 hand 通道）；
  * `hand_evo_form`：当前手牌 4 张**本手是否为觉醒形态**（4 维 0/1，由 `evolution_state` 决定）；
  * `evo_counter`：每张带觉醒档的卡「还差几次触发」（4 维归一化）。
* 判据：① 字段与 `battle.py` 的触发判定**逐局对账一致**（同一 tick 同值，抽 20 局全等）；
  ② 维度变化**写进 `plan_dim`/`belief_dim` 之外的 scalar 块**，并断言 `--fresh` 训练的
  `enc_fc` 输入维 = 旧 + Δ（R6 的静默错防护）；③ 回归测试覆盖「觉醒当且仅当计数器命中」。

### S3（RL 侧声明）—— 训练/评估 env 给双方分配觉醒位

* 做法：卡组 → 觉醒位映射表（**沿用人类回放的分布**或固定 1 槽，A/B 两臂单变量）；
* 判据：① 同 seed 20k 对照，觉醒触发率 > 0 且**行为双侧带 [18,28.5] 不变**；
  ② 不引入新的非法包（`invalid` 计数 = 0）；③ 若触发率仍为 0 ⇒ 查 §1#4 的数据是否齐全（失败分支）。

### S4（验收，含成本闸门 R11）

* 判据（跑前写死，三项**同批配对**）：
  1. **触发率对齐**：本仓对局里觉醒使用率 vs 人类回放里可推导的使用率（差 ≤ ±20% 相对）；
  2. **不劣化**：IL 留出 `NLL` 不升（ΔNLL ≤ 0）或 act-AUC 不降（ΔAUC ≥ 0）；
  3. **行为**：10 局双侧出牌带不变、`Xbow`/C14 结论不被推翻。
* **无收益回滚**（R11）：若 S4 三项全不达标 ⇒ 观测字段与觉醒位声明**回滚**（保留 S1 数据保真）。

## §3 不变量（任一被破坏即停手）

1. 不改奖励表（R7：`tower_value_mult` / `edw` / `汇率` 同源）；
2. 不改掩码判定（R13：本轮觉醒工作**不碰** `action_mask.py`）；
3. 观测维变化 ⇒ 必须 `--fresh`（R6），旧 ckpt 不许热启动；
4. 任何「觉醒生效」的结论都要给出**触发计数**，不接受「应该会觉醒」；
5. 人类回放侧只做**推导**（`card_play_number` + 周期表），不得把「推导出的觉醒」当作回放事实写死。

## §4 已知风险

* **回放不标注觉醒形态**（§1#7）⇒ 人类使用率只能推导，误差未定；
* 觉醒形态的**数值**在本仓是 `derive_evolved_stats` 推导的，未必等于真实游戏（未验证）；
* 若 S1 覆盖率低（§1#9 未定），S2/S3 都会建立在**不完整**的觉醒集合上 ⇒ 先测覆盖率；
* 成本：S2/S3 需要**新 ckpt**（R6），按 20k 标准协议估；是否值得按 S4 的性价比条款判。
'''

## §5 首测（2026-09-22，S1 判据① 覆盖率）

口径：扫描 FL 回放前 **3,000 局**的 `decks[side][i][0]`，用 `fl_il_to_bc.py::split_variant()` 拆后缀、
`map_key()` 映射到本仓卡名，再查 `EVOLUTION_CYCLES` / `Card(name).evo_raw`。

| 变体 | 槽数 | 本仓可覆盖 | 覆盖率 | 结论 |
|---|---|---|---|---|
| `-ev1` | **11,293** | 11,293 | **100.0%** | ✅ **S1 判据① PASS（≥95%）** |
| `-ev2` / `-ev3` | 0 / 0 | — | — | 该数据分片里没有 |
| `-hero` | **4,550** | **0** | **0.0%** | ❌ **不是觉醒**：Hero 是另一套机制（`set_hero_slots` / `hero_raw`），本仓 Hero 表**没有**这批卡（Berserker 1265 / Valkyrie 513 / Bowler 371 / Knight 273 …）⇒ **Hero 也是一条完全未接线的机制**（单列，不在本预注册范围内） |

⇒ S1 可以继续（觉醒侧覆盖 100%）；**Hero 侧 4,550 槽 0% 覆盖**必须另开一项，否则「人类对战的一半特殊形态」在我们引擎里根本不存在。


---

## §6 S1 实现（2026-09-22，跑前回写；`src/` 零改动）

**改的东西**（`battle.py` / `action_mask.py` / `player.py` / `evolutions.py` **一行未改**）：

| 件 | 位置 | 作用 | 关掉时（缺省） |
|---|---|---|---|
| `evo_slot_cards(vs)` | `scripts/fl_il_to_bc.py` | 牌组变体 → 可声明的觉醒卡；**只接 `-ev*`**，`-hero` 显式跳过（另一套机制，§5：覆盖 0%） | 不调用 |
| `install_evo_counters(battle, st)` | 同上 | 给重建战局装**两个只读计数器**（见下） | 不调用 |
| `--evo-slots` | 同上 | 开关（缺省**关**） | **旧路径逐位不变**（【R2】） |
| `r["_evo"]` | `run_samples` | 把**被剥掉的变体后缀**带进 `_convert_one`（旧版只带 `names` ⇒ 后缀丢失，这正是 §1#8 的成因） | 只多一个键，不被读 |
| `S1_evo` 读数块 | `run_samples` 汇总 | 判据①③的全部分子/分母 | **不出现**（E1 在产物层面验证） |
| `scripts/selftest_evo_slots.py` | 新 | 【R8】回归：**20 断言** | — |
| `scripts/probe_evo_pair.py` | 新 | **同进程配对**探针（判据②的强形式，见 §7） | — |
| `scripts/il_evo_probe_report.py` | 新 | 判读 + schema 不变量 + 跨进程噪声标定 | — |
| `scripts/_run_evo_probe.sh` | 新 | 200 局 off/on 一条链（可续跑） | — |

**可声明条件（必须两处齐备）**：变体是 `ev*` **且** `name ∈ EVOLUTION_CYCLES`（周期表）
**且** `Card(name).evo_raw` 为真（觉醒数值快照）。缺任一条 ⇒ **声明了也不触发**（`battle.py:3068-3070`
的四个条件之一必然为假）= **死代码** ⇒ 计入「未映射」，**不静默声明**。

**两个计数器（【R13】位图对账的同一种思路：两路读数互校）**：
* `finish`（**主口径**）= 包 `BattleState._finish_deploy`，在**调用前**按 `battle.py:3068-3070` 的同一组
  输入读一次（`evo_plays` 的自增在**函数体内** ⇒ 调用前 = 引擎自己读到的值）⇒ 口径 =「觉醒**出牌**次数」，
  覆盖部队/建筑/**法术**；
* `wrap`（**交叉核对**）= 包 `BattleState._wrap`，数 `len(entity_data)==6 and entity_data[5]`——
  那是引擎从出生队列里带过来的**觉醒标记位**（`:2758-2761` 读、`:3215` 写）⇒ **不重算谓词，直接读引擎的位**
  ⇒ 口径 =「觉醒**实体**个数」，只覆盖部队/建筑（法术觉醒不走 `_wrap`）。
  ⚠️ **两口径量纲不同**：单兵卡 `wrap == finish`、n 兵卡 `wrap ≈ n × finish`、法术只进 `finish`
  （预注册初稿曾写成「`finish ≥ wrap`」——**不准确**，已在实现里改正并写进 JSON 的 `cross_check_note`）。

---

## §7 S1 门禁（全部脚本复算）

1. **`scripts/selftest_evo_slots.py`：20/20 PASS**（【R19】只跑本文件）
   T1 映射与拒收（`-hero` **不进未映射**、不在周期表/缺 `evo_raw` 都记未映射）；
   T2 **引擎真跑**：Knight(cycle=2) 连出 6 次 ⇒ `finish == 2` 且**单兵卡 `wrap == finish`**；
   T4 触发节律累积计数 `[0,0,1,1,1,2]`；T7 与 `evolution_state` 解析式一致；
   T3 **负对照**（同样出 6 次但**不声明**）⇒ 两计数器 **都 = 0**；
   T5 **Hero 互斥**（同卡同时声明 evo+hero 槽）⇒ 两计数器 **都 = 0**（证明我的谓词与引擎的
   `card_name not in hero_slots` **同口径**，不是自己另写一套）；
   T6 法术（Zap）只进 `finish`、`wrap == 0`。
2. **同进程配对探针**（判据②的**强形式**；因为【C17】已证**跨进程不可复现**，两个独立进程的 200 局
   只能证明「没有大改动」）：
   P1 = **构造对照**（把牌组变体清空 ⇒ 一局都不声明）上 off/on **全部公共字段逐值相等**；
   P2 = 真实回放上至少一局触发 > 0；P3 = 崩溃计数（照实报）。
   ⚠️ **第一版设计错误（已修）**：真实回放里 **95%+ 的局都会声明** ⇒ 自然样本对 P1 **零覆盖**，
   而初版照样打印 `PASS (0 局中 0 局有差异)` —— 这是一个**空洞 PASS**（与 C26 的 J-W3.2「判据饱和」
   同型的病）⇒ 改为**显式构造对照**（`--p1-control N`），否则 P1 **不许**报 PASS。
3. **产物 schema 不变量（E1）**：`--evo-slots` 关的产物里**不得**出现 `S1_evo` 块或任何逐局 `evo_*` 字段。

---

## §8 S1 结果（200 局，`--stop-mode save --stop-stride 4 --plan-extras`，唯一变量 = `--evo-slots`）

### §8.1 判据

| 判据 | 读数 | 判定 |
|---|---|---|
| ① 覆盖率 ≥ 0.95 | **能声明率 = 659 / 716 = 0.9204** | ❌ **FAIL** |
| ② `labels` 漂移 ≤ 5% ∧ `errors = 0` | labels **4469 → 4182（−6.42%）**、`errors` **0 → 9** | ❌ **FAIL** |
| ③ 触发次数 > 0 | **445 次**（我方 248 / 对手 197）、`wrap` 1239、声明 **191/200 局** | ✅ **PASS** |

**① 的口径差（本批第一处预注册缺口）**：判据①字面说的是「后缀**能映射**」（§5 实测 **100%**），
而实现报的是更严的「**能声明**」= 能映射 ∧ 周期表 ∧ `evo_raw`。两者**不是同一个量**，差集**恰好一张卡**：

| 键 | 槽数 | 解析到 | `in_cycles` | `has_evo_raw` |
|---|---|---|---|---|
| `elite-barbarians-ev1` | **57** | `AngryBarbarians` | ✅ | ❌ **无快照** |

⇒ 这不是「映射表缺一条」（判据④的失败分支猜的那种），而是**本仓根本没有这张卡的觉醒数值快照**
⇒ **补映射表无效**，只能补数据（新开放项 **O-E1**）。

**③ 是实打实的阳性**：`rl/` 从来没有声明过觉醒位（C21），所以**旧路径触发必然 = 0**；
本批第一次让觉醒在重建战局里真的发生（445 次），且**100% 的可成功转换局都声明了觉醒位**
（191/200，差的 9 局全是崩溃，见 §8.2）。

### §8.2 ② 的失败**归因**：全部来自**引擎侧两个真崩溃**，不是「重建语义变了」

9 局（4.5%）在 on 臂抛异常（`_worker` 吞掉 ⇒ `errors=9`、这些局所有计数为 0）。逐 tag 剔除后
（**191 局可比**）：

| 字段 | off（191 局） | on（191 局） | 相对差 |
|---|---|---|---|
| `labels` | 4252 | 4182 | **−1.65%** |
| `frames` | 62793 | 60874 | −3.06% |
| `team_single` | 4455 | 4380 | −1.68% |
| `mask_reject` | 203 | 198 | −2.46% |
| `stop_save_cand` | 54574 | 52732 | −3.38% |

⇒ **剔除崩溃后 ② 会 PASS（−1.65% ≤ 5%）**。其余字段的负移（frames/stop_save_cand）是**觉醒真的改变了
战局**（数值不同 ⇒ 局长度不同）的**预期结果**，不是管线缺陷。

★ **崩溃根因（只读取证，一行未改）**：`evolutions.py:70-86 collect_evo_mechanics()` 走**全树扫描 +
同名键首见优先**，而 `evolvedSpellsData.statsTags` 是一张 **「字段名 → 标签串」的映射表**，
且在 JSON 键序里排**第 4 位**（`baseData` 排第 30 位）⇒ 标签串**遮蔽**真数值：

| 卡 | 被遮蔽字段 | 泄漏值 | 真值 | 崩点 |
|---|---|---|---|---|
| `Witch_EV1` | `spawnPauseTime` | `'interval_of_spawns'` | `7000`（`baseData`） | `card_mechanics.py:87` `TypeError: str / float` |
| ~~`GoblinDrill_EV1`~~ → **`GoblinCage_EV1_TEMPNAME`** | `deathSpawnCount` | `'spawn_count'` | **`1`** | `battle.py:1383` `ValueError: int('spawn_count')` |

> ★ **勘误（2026-09-22 同批，见 §11.5）**：本表初版把第二张卡写成 `GoblinDrill_EV1`，**是错的** —— 真正被遮蔽的是 **`GoblinCage_EV1_TEMPNAME`**（真值 `1`）；`GoblinDrill_EV1` 的字符串值 `spawnCharacterOnHide='Goblin'` 是**合法的名字字段**。§11.2 的 A/B 仪器逐键复算给出了确证。

影响面（全量枚举 34 张 evo）：mechanics dict 里含字符串值的有 **6 张**，其中只有上述 **2 张**是
**数值字段**被污染（其余 4 张的字符串是合法名字：`onKilledDoneAction='PekkaEV1_Heal'`、
`nextAction.spawnData=...`、`spawnCharacterOnHide='Goblin'`）⇒ **觉醒必崩**只在 2 张卡上。

**为什么从来没人踩到**：`rl/` 从没声明过觉醒位（C21）⇒ 这条代码路径在本仓对局里**从未被执行过**。
S1 的副产物就是**把一条从未跑过的路径跑起来了**，并且**立刻崩了 4.5% 的局**。

### §8.3 判据④的执行与**第二处预注册缺口**

判据④原文：「若 `labels` 漂移 > 5% ⇒ 判**重建语义变了**，停手并回到本文件讨论」。
实际：漂移 > 5% **成立**，但**成因不是**它唯一设想的那种（语义变了），而是**引擎崩溃**。

⇒ **处置（按纪律）**：
1. **停手**：**不启动 S2/S3**（观测加字段 / RL 侧声明），在本文件登记；
2. **不改判据文本**（与 `il_whiff_handscore_prereg` §6.8 / `fl_il_il2_prereg` §6.8 同款纪律：
   跑后**不许**改判据），把「判据④只设想了『映射缺表』与『语义变了』两种成因」登记为**预注册缺口**；
3. 代码**不回滚**（`--evo-slots` **默认关**、`src/` 零改动 ⇒ 不影响任何既有读数），
   在崩溃修好并重新预注册之前**不用于生成训练语料**。

### §8.4 附带读数

* **E1（schema 不变量）PASS**：off 产物逐局 **无** `evo_*` 字段、无 `S1_evo` 块；on 侧有 8 个。
  ⚠️ 报告脚本初版把 `source.evo_slots`（**配置回显**）当成「off 侧出现了 evo 字段」⇒ **假 FAIL**，已修。
* **E2 双口径**：`wrap / finish = 1239 / 445 = 2.784`（>1 合理：多兵卡每次觉醒出多个实体）；
  逐卡 top：Cannon 62、Princess 44、Firecracker 42、SkeletonArmy 40、MinionHorde 33、GoblinBarrel 31。
* **E3 跨进程噪声地板**（**非判据**）：与 2200 局旧语料（`samples_stop_save_hs.json`）里**同一批
  200 个 tag** 比，`labels` **4468 vs 4469 = +0.022%** ⇒ 5% 阈值对**聚合 labels** 来说**充分宽**。
* **同进程配对**：P1（8 局构造对照）**0 字段差异**、P2（13 局声明 / 18 触发）PASS、P3 **1/14 崩溃**
  （Witch_EV1，与 §8.2 同源）；**逐局语义差是双向的**（`labels` 13→13 / 18→21 / 38→22 / 25→29）
  ⇒ 觉醒对单局的影响量级可达 **±40%**，但**聚合后**（剔除崩溃）只有 −1.65%。

---

## §9 本批查出的新开放项（**全部一行未修**）

| # | 项 | 证据 | 影响 |
|---|---|---|---|
| **O-E1** | 本仓**缺 `AngryBarbarians`（`elite-barbarians`）的觉醒数值快照**（`evo_raw` 为空，但周期表里有 ⇒ `in_cycles=True ∧ has_evo_raw=False`） | §8.1 | 该卡**永远不可能觉醒**（声明了也是死代码）；57/716 个回放觉醒槽落空 |
| **O-E2** | `collect_evo_mechanics` 的 **`statsTags` 字符串遮蔽**（上面两张卡） | §8.2 | 觉醒**必崩**：Witch_EV1 `TypeError`、GoblinDrill_EV1 `ValueError`；4.5% 的局 |
| **O-E3** | `fl_il_to_bc.py` 的 `[J6.1]` 打印在 `save_frame_share_of_off is None`（= `team_off == 0`，全崩时）抛 `TypeError`，**把 `errors=N` 的真失败盖掉** | 本批实测（第一轮 on 臂就是这样被盖住的） | 已修：None 安全 + 新增 `[errors]` 行打印首个 worker 异常与 tb |
| **O-E4** | `arena.can_deploy_at(..., is_spell=True)` 是**死路**：`arena.py:181` 调 `TileGrid._is_rolling_projectile_spell()`，**该方法在 `TileGrid` 上不存在** ⇒ 必 `AttributeError` | `selftest_evo_slots.py` T6 实测踩到；生产代码**无一处**用 `is_spell=True` | 潜伏；本轮**不修**（不在 S1 范围） |
| **O-E5** | `probe_evo_pair.py` 的 `--out` 用 `os.path.abspath` 解析，而脚本**已经 `chdir(SRC)`** ⇒ 产物**静默写到 `src/clasher_new/docs/fl_il_2026-09-21/`**（`docs/` 下什么都没生成；本仓**第三次**踩同一类路径坑） | **已修**（改用本脚本自己的 `ORIG_CWD` 解析）；用 `--games 1` 冒烟复验落盘位置正确 |

---

## §10 下一刀（**预注册，尚未执行**）：修 O-E2 的两个引擎崩溃，然后重跑 S1

**做法**（单变量，最小改动）：`collect_evo_mechanics()` 扫描时**跳过 `statsTags` 子树**
（它是「字段名 → 标签串」的**元数据**，不是机制参数）——一行 `if k == 'statsTags': continue`；
**不**改任何触发/数值逻辑，**不**动 `battle.py` / `action_mask.py`。

**判据（跑前写死）**：
1. **数值不变量**：对全 34 张 `evolvedSpellsData` 跑 `collect_evo_mechanics`，新旧结果**逐键对比**：
   变化集合**必须**恰好 = {`Witch_EV1.spawnPauseTime`, `GoblinDrill_EV1.deathSpawnCount`}，
   且新值 = 真数值（`7000` / `4`）；**其余 32 张逐键逐值相等**（`==`，非近似）。
2. **回归**：`selftest_evo_slots.py` 20/20 仍 PASS，**新增** T8 = Witch 连出 3 次不抛异常
   （= 本崩溃的**跨局边界**回归，【R8】）；`run_selftests.py` 相关子集无新增失败（【R19】）。
3. **端到端**：S1 200 局重跑 ⇒ `errors = 0` 且 `labels` 相对 off 漂移 **≤ 5%**（预计 −1.6%~−2%）。
4. **失败分支**：若判据 1 出现**其它**卡的变化 ⇒ 说明 `statsTags` 之外还有第二个遮蔽源，
   **停手**、逐条列差异、回到本文件；若判据 3 仍 `errors > 0` ⇒ 说明还有**第三条**崩溃路径，
   同样停手并登记（**不许**为了「让 errors=0」而吞异常）。

---

## §11 §10 的执行与判读（2026-09-22 同批续；**判据在 §10 跑前已写死**）

### §11.1 若何（单变量最小改动）

`evolutions.py::collect_evo_mechanics()` 的 `_walk` 加一行 `if k == 'statsTags': continue`
（`statsTags` = 「字段名 → 标签串」的**元数据表**，不是机制参数）。**不动**触发/数值逻辑、
**不动** `battle.py` / `action_mask.py` / `player.py`。

### §11.2 判据①（数据层 A/B，同进程、同数据、真单变量）

仪器：`scripts/probe_evo_mechanics_ab.py`（**修复前逻辑在本文件内逐字复制**，不 import、不靠 git 回滚）
⇒ `docs/fl_il_2026-09-21/evo_mechanics_ab.json`。全 **34** 张 `evolvedSpellsData`：

| 判据 | 结果 |
|---|---|
| **A1**（预注册字面）变化集合 == {`GoblinDrill_EV1.deathSpawnCount`, `Witch_EV1.spawnPauseTime`} | ❌ **FAIL** |
| **A1'**（**跑后修正口径**）变化集合 = 恰好这 2 个 `(卡,键)` 且新值 == 该卡 evo 树里的真值 | ✅ **PASS** |
| **A2** 其余 **32 张**逐键逐值相等（`==`） | ✅ **PASS**（`unchanged=32`） |
| **A3**（预注册字面）新输出**无任何**字符串 | ❌ **FAIL**（残留 4 个） |
| **A3'**（**跑后修正口径**）**数值语义**字段无字符串（判据由数据给出：残留字符串须能在同卡 evo 树里找到同键的**字符串**真值） | ✅ **PASS**（合法名字型残留 4 / 非法残留 **0**） |

**实际变化**（逐字）：

```
Witch_EV1               : spawnPauseTime     'interval_of_spawns' → 7000   (真值 7000 ✓)
GoblinCage_EV1_TEMPNAME : deathSpawnCount    'spawn_count'        → 1      (真值 1    ✓)
```

### §11.3 判据②（回归）

`scripts/selftest_evo_slots.py` **28/28 PASS**（原 20 + **新增 T8** 6 条 + **T9** 3 条；原 T1–T7 全未变）：
* **T8（O-E2 的跨 tick 回归，【R8】）**：Witch（cycle=1）出 3 次 ⇒ 场上存在**觉醒** Witch 且
  `evo['spawnPauseTime'] == 7000`；**推进 8 tick（4 s）不抛异常**（旧实现**必崩**）；并**钉住真值被使用**：
  出兵那一 tick 的 `next_spawn_remaining` 轨迹 = `[1.0, 1.0, 0.5, 0.0, **7.0**, 6.5, 6.0, 5.5]`
  ⇒ 最大值必须**恰为 7.0**（= `7000/1000`，不是默认 `1.0`、也不是字符串路径）。
  ⚠️ 第一版把断言写成「跑完 8 tick 后读值 == 7.0」——**错**（该值每 tick 递减，末尾是 5.5）
  ⇒ 改成**逐 tick 记录取最大值**。
* **T9（量纲构造性演示）**：Skeletons（cycle=2、`spawn_number=3`）出 3 次 ⇒ `finish == 1`、
  `wrap == 3 == spawn_number × finish` ⇒ 把「两口径量纲不同」写成可执行断言。

### §11.4 判据③（端到端，200 局重跑）

| 判据 | 修前 | **修后** |
|---|---|---|
| ① 覆盖率 ≥ 0.95 | 0.9204 | **0.9239**（692/749）❌ **仍 FAIL** |
| ② `labels` 漂移 ≤ 5% ∧ `errors = 0` | labels −6.42%、errors **9** | labels **4465 → 4407（−1.30%）**、errors **0** ✅ **PASS** |
| ③ 触发次数 > 0 | 445 | **482**（我方 267 / 对手 215）、`wrap` **1311** ✅ PASS |

★ **决定性证据**：修后触发逐卡表里**首次**出现 `Witch: 5` 与 `GoblinCage: 5` —— 正是修前**必崩**的那两张卡
⇒ 修复把「崩掉 4.5% 的局」变成「正常触发 10 次」。**声明觉醒位 200/200 局**（修前 191/200，差的 9 局全是崩溃）。

**★ 同进程配对探针（同 14 局 tag）**：修前 P3 = **CRASHES 1/14**（Witch_EV1）⇒ 修后 **CLEAN 0/14**；P1（8 局构造对照）**逐值相等**不变、P2 14/14 局声明 / 21 触发。

### §11.5 两处**跑后修正**（登记为预注册缺口，**不改** §10 判据文本）

3. **§10 判据①把卡名写错了**：写的是 `GoblinDrill_EV1`，实际被 `statsTags` 遮蔽的是
   **`GoblinCage_EV1_TEMPNAME`**。`GoblinDrill_EV1` 的字符串值是**合法的名字**字段
   （`spawnCharacterOnHide='Goblin'`）。⇒ 变化集合的**实质**与预注册完全一致（恰 2 张卡 × 各 1 个数值键），
   只有**卡名**写错 ⇒ 字面判 FAIL、实质判 PASS，两行都保留在读数 JSON 里。
4. **§10 判据③的 A3 过严**：写的是「新输出不得有任何字符串」，但 4 个残留全是**合法的名字/动作引用**
   （`onKilledDoneAction='PekkaEV1_Heal'`、`nextAction.spawnData` ×2、`spawnCharacterOnHide='Goblin'`）
   —— 它们在**原数据里本来就是字符串**。⇒ 改为数据驱动的口径 A3'（残留字符串必须在同卡 evo 树里能找到
   同键的字符串真值），字面 FAIL / 修正 PASS 两行都保留。
5. **（仪器侧）** `probe_evo_pair.py` 的 `--out` 曾用 `abspath` 解析，而脚本已 `chdir(SRC)`
   ⇒ 产物**静默写到** `src/clasher_new/docs/…`（`docs/` 下什么都没有）⇒ **已修**并冒烟复验（= **O-E5**）。

### §11.6 S1 **总判更新**与下一步

**S1 现在 = ① FAIL / ② PASS / ③ PASS**。②③ 已兑现；**① 的失败原因单一且明确**：
`elite-barbarians-ev1`（→ `AngryBarbarians`）**本仓没有觉醒数值快照** ⇒ 57 / 749 个回放觉醒槽
**不可声明**（覆盖率 0.9239）。这不是「映射表缺一条」⇒ 判据④的第一条失败分支（补映射表）**不适用**。

⇒ **处置**：
1. `--evo-slots` **仍保持缺省关**（`src/` 只多了一处 `statsTags` 跳过；对不声明觉醒位的旧路径**零影响**，
   E1 已在产物层面验证）；
2. **仍不启动 S2/S3**：① 未达标，而 O-E1 的修法是**补游戏数据**（不是改代码）⇒ 需要
   **另行预注册 + 外部数值来源**，本轮**不做**（【R10】不猜数值）；
3. **O-E1 留作下一刀候选**：可用 `evo_2025_data.py` 已有的「自建 7 张」先例（`Furnace`/`BabyDragon`/
   `SkeletonArmy`/`Ghost`/`RoyalHogs`/`MinionHorde`/`Princess`）+ 一张外部来源表；**判据须跑前写死**，
   且必须给出**数值来源 URL 与取数时间**（不许把推导值当官值写死）。
