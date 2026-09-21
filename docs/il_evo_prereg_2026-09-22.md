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

