# 两个录像格式逐字段对照：**人类回放** vs **我们自己的联赛录像**

> **触发**：用户要求「专门拉出几个回放看格式」。
> **对象**：① `VanguardX101/IL_Replay`（HF 公开，取数 2026-09-19）；② 我们 `src/clasher_new` 的联赛录像（schema 5）。
> **手段**：新增只读仪器 `scripts/_il_replay_format_scan.py`（`--schema` / `--stats` / `--check-cycle-elixir`）
> + 对**真实文件**直接 `pickle.load` / `pyarrow` 读取；源码为权威（`rl/replay.py`）。
> **纪律**：只写文件内容或实测支持的结论；标「未验证」的项不升格（【R10】）；
> 两边口径不同**并列不调和**（【R17】）。**未跑训练、未改任何训练/奖励代码。**

---

## 0. 一句话结论

**两个格式是互补的，不是"一个是另一个的子集"** ——
人类回放给的是「**动作时间线 + 卡组元数据**」（每卡等级、觉醒/英雄变体、第 9 张塔兵），**没有状态**；
我们自己的录像给的是「**逐帧状态流 + 奖励/价值**」（圣水、三塔血、实体 15 元组、`reward`、`v0/v1`），**没有卡组元数据、没有 seed**。
⇒ **两边都不能直接当 IL/BC 样本**：人类回放要**重建状态**，我们自己的录像要**重建观测**（且因为没有 seed，**无法确定性重放**）。

**并且**：这套人类回放**就是 FirstLight 自己的数据集** —— 其 README 自述名 `Firstlight anonymized battle replays`，
并写明用 `CR_REPLAY_DATASET_ROOT` / `--dataset-root` 喂它的训练缓存生成（**我上一轮的推断被一手证据确认**）。

---

## 1. 数据面与取数

| 项 | 读数 |
|---|---|
| 分片 | **104** 个 parquet（`replays/` 52 + `actions/` 52） |
| 行数 | replays **252,238** + actions **17,836,160** = **18,088,398**（与 manifest 逐项吻合） |
| 体积 | **1,888,381,137 B**（1.76 GiB） |
| 本轮实读分片 | `part-000000`（actions 384,275 行 / replays 5,000 行）、`part-000025`、`part-000051`（**末段仅 852 局**） |
| 抽取/校验 | 108 个文件带 `sha256`（manifest）；`verification`：`full_nested_privacy_scan=passed`、`full_write_readback_equality=passed`、`all_source_file_hashes_unchanged=true`、`duplicate_source_replay_ids=0` |
| `compatibility.json` | `preflight_equal=156`、`native_replay_conversion_equal=156`、`browser_loads=10`、`training_stream_rows=256`、`existing_rejections`（空） |

**脱敏范围（manifest 逐字，8 删 / 7 留）**：

| removed（8） | preserved（7） |
|---|---|
| 对局日期与采集时间戳 / 玩家名与 ID / 部落名与 ID / 奖杯与奖杯变化 / 原 replay ID 与 URL / 采集账号与代理 ID / **自由文本采集告警** / 玩家索引与采集日志 | **全部对局行与动作** / **相对动作 tick 与秒** / **坐标与 side** / **卡组、形态与等级** / **塔兵与塔血** / **皇冠、结果与 game_mode** / **时长与数值校验计数器** |

⇒ 关键：`average_elixir` / `elixir_leaked` / `four_card_cycle_elixir` / `tower_card` / `final_tower_hitpoints`
**都在 preserved 里**（属"全部对局行"与"塔兵与塔血"）。

---

## 2. 人类回放格式（Firstlight anonymized battle replays）

### 2.1 两张表 + **三套互不相同的版本号**

| 位置 | 字段 | 实测值 |
|---|---|---|
| replays 表 | `schema_version` | `royaleapi-battle-actions.v1` |
| payload 顶层 | `collection_schema_version` | `royaleapi-replay-collection.v1` |
| actions 表 | `schema_version` | `royaleapi-replay-parquet.v1` |

⚠️ **三个版本号三条命名线**，改格式时**必须三个一起看**（只改一个的迁移会漏）。
两张表的列：replays **11 列**、actions **21 列**（清单见 §2.5）。

### 2.2 payload 顶层 7 键

`battle` / `collection_schema_version` / `events` / `replay` / `schema_version` / `source` / `validation`

### 2.3 ★ 结构是**严格固定**的（1000 局实测）

| 读数 | 值 |
|---|---|
| 1000 局里出现过的路径总数 | **370** |
| **恒存在（1000/1000）** | **136 条** |
| **非 `card_counts` 的可选路径** | **0 条** |
| 唯一的"可变"部分 | `replay.card_counts.<side>.<card_key>` —— **按卡名动态键**（`card_counts.team.skeletons` 只在 346/1000 出现，因为只有这些局用了它） |

⇒ 「抽 3 局 162 路径 / 抽 2 局 167 路径」的差异**全部**来自 `card_counts` 的卡名不同，**不是 schema 变化**。
⇒ 对读方意味着：**除了 `card_counts`，其余 136 条路径可以按定长表读，不需要容错**（但按 §2.6 仍需容忍 None 值）。

### 2.4 `events[]` 逐字段（play_card 用 21 列 / activate_ability 用其中一部分）

| 字段 | 类型 | 实测语义 |
|---|---|---|
| `kind` | str | **只有 2 值**：`play_card` / `activate_ability` |
| `side` | str | **只有 2 值**：`team` / `opponent` |
| `card_key` | str | RoyaleAPI kebab（`hog-rider`、`the-log`）；分片0 前 200k 行 **122 个不同值** |
| `card_play_number` | int ∈ [1,17] | **同一 (局,侧,卡) 内 1..k 连续、无重复**（抽样 3000 行 **0 例外**）⇒ 语义 = 该侧该卡在本局的第几次出牌 |
| `replay_tick_20hz` | int ∈ [108,5979] | 20 Hz tick |
| `time_seconds` | float ∈ [5.4,298.95] | **= tick/20**（`216 → 10.8` 逐项吻合） |
| `native_x` / `native_y` | int | **世界坐标**（`x∈[499,17501]`、`y∈[500,31500]`） |
| `grid_x` / `grid_y` | int | **18×32** 网格（`grid_x` 18 值、`grid_y` 32 值）—— **与我们引擎同构** |
| `arena_column` / `arena_row` | str / int | 人类可读标签：**`column == chr('A'+grid_x)`、`row == grid_y+1`（抽样 0 例外）** |
| `form_at_play` | str | ★ **全分片恒为 `'unknown'`**（play_card）/ `None`（ability） |
| `ability_source_authoritative` | bool | ★ **全分片恒为 `None` / `False`** |
| `deck_card_key_candidates` | list | ★ **长度恒为 1**（play_card）/ 0（ability） |
| `ability_source_candidates` | list | ★ **唯一有信息的"候选"字段**：长度 0(play_card) / 1(**15,396**) / 2(**3,517**) |
| `source_fields` | dict | 原始采集字段（`data_c/data_i/data_s/data_t/data_x/data_y/timeline_card_key`）= **溯源** |
| `event_json`（actions 表） | str | 事件的 JSON 原样（199,795 个不同值） |
| `coordinates`（payload） | dict | **四套坐标 + 3 个布尔/偏移**（见下） |

### 2.5 ★ 四套坐标与实测变换（4001 个事件）

样例：`native_world_units={x:16501,y:30499}`、`display_royaleapi_units={x:16501,y:1501}`、
`raw_royaleapi_units={x:1499,y:30499}`、`grid_cell_floor={x:16,y:30}`、`arena_label_from_blue={column:'Q',row:31}`、
`inversion_applied=true`、`inside_18x32_arena=true`、`subcell_offset_from_floor_center={x:0.001,y:-0.001}`。

| 检验 | 成立数 / 总数 |
|---|---|
| `native.x == display.x` | **4001 / 4001** ✅ |
| `native.y == 32000 − display.y` | **4001 / 4001** ✅ |
| `native.y == raw.y` | **2041 / 4001**（其余不成立 ⇒ 随事件变） |
| `native.x == 32000 − raw.x` | **0 / 4001**（该猜想被否证） |

⇒ **`native_world_units` 是权威世界坐标**（`x = display.x`、`y = 32000 − display.y`），
**`raw_royaleapi_units` 是采集端原值**（raw→native 的完整规则**未定**，需 FL 侧代码确认）。
⇒ **用 actions 表的 `native_x/native_y` 是正确的**（它 = `native_world_units`），
这也解释了上一轮可行性探针的坐标 A/B 为什么判定 **`raw`（不翻转）**。

### 2.6 ★ 五个"看起来很有信息、实际是常量"的字段

| 字段 | 实测 | 含义 | **FL 读不读**（一手取证） |
|---|---|---|---|
| `form_at_play` | `'unknown'` × 365,362 / `None` × 18,913 | **当前不携带信息** | ❌ **不读**（形态改从 `deck[].card_key` 的 `-ev1`/`-hero` **后缀**推断） |
| `ability_source_authoritative` | `None` / `False` × 18,913 | **恒假** | ✅ **读**（`True` 才采信 candidates，否则回退后缀/champion 绑定表） |
| `deck_card_key_candidates` | 长度恒 1 / 0 | **不携带歧义信息** | ❌ **不读** |
| `ability_source_candidates` | 0 / 1 / 2 | ★ **唯一真正携带信息的候选字段** | ✅ **读**（能力事件归因；>1 时清空名字提示，交给引擎「唯一 Ready 实例」查找） |
| （对照）`source_fields` | 全部有值 | **溯源**，不是载荷 | ⚠️ **只读 `data_i` 一个子键**（决定 owner + 坐标翻转）；其余 6 个子键 **0 命中** |

⇒ 这一节的意义：**payload 看起来"字段很富"，但有 3 个字段在当前版本里是 schema 占位符**，
且**其中 2 个连 FL 也不读**。按"字段数"估数据价值会高估。

### 2.7 质量计数器（1000 局实测）

| 字段 | 分布 |
|---|---|
| `replays.warning_count` | **全 0**（1000/1000）—— ⚠️ 但 manifest 说**自由文本告警被脱敏删除** ⇒ 现在只剩计数 |
| `validation.unmatched_timeline_event_count` | **全 0** ⇒ 时间线映射完整 |
| `validation.{ability,coordinate,map_marker,normalized,timeline}_event_count` | 6 键**全 1000/1000 存在**；ability p50 **3** / max 18；coordinate p50 **70** / max 176 |
| `battle.crowns`（team） | 0:264 / 1:558 / 2:65 / 3:113（分片0 1000 局）|

### 2.8 抽样读数（1000 局 × 2 个分片，**供口径参考，不作判据**）

| 项 | 分片0 | 分片25 |
|---|---|---|
| `game_mode` | Ranked 904 / 1v1 Battle 46 / Ladder 28 / Grand Challenge 14 / 1v1 4 / Classic Challenge 3 / **1v1 Showdown 1** | Ranked 816 / 1v1 Battle 55 / **Normal Battle 47** / Grand Challenge 39 / Ladder 22 / **1v1 Showdown 12** / 1v1 4 |
| `tower_card` | princess 1920 / royal-chef 29 / dagger-duchess 28 / cannoneer 23 | princess 1904 / **dagger-duchess 62** / royal-chef 30 / cannoneer 4 |
| 变体后缀 | `-ev1` 3766 / `-hero` 1441 | `-ev1` 3757 / `-hero` 1735 |
| `duration.timeline_seconds` | min **43.05** / p50 228.55 / max **314.05** | min 45.05 / p50 206.55 / max 313.05 |
| deck 长度 | **8/8**（2000 侧全同） | 同 |

⚠️ **两个实施注意事项**：
1. **`game_mode` 必须过滤**（`1v1 Showdown` / `Normal Battle` 与我们引擎建模的 1v1 标准模式不是同一件事）；
2. **时长尾部 > 300 s**（max 313–314 s）⇒ **超出我们引擎 300 s 硬上限**（上一轮可行性探针已实测：我们引擎在 180–223.5 s 就按皇冠规则提前终局）。

### 2.9 `tower_card` ↔ 我们引擎的**现成映射**

回放 4 个取值 `tower-princess` / `dagger-duchess` / `cannoneer` / `royal-chef`
**恰好对应** `PlayerState.set_tower_troop` 的合法值域
（`None`(=Princess) / `King_KnifeTowers` / `King_CannonTowers` / `King_ChefTowers`，`player.py:21-25`）。
⇒ **第 9 张塔兵不需要新建机制**。

### 2.10 ★ 语义验证：`four_card_cycle_elixir` = 「最便宜 4 张的费用和」（**1941/1941**）

用我们自己的卡表复算 1000 局 × 双侧：**0 例 `declared < ours`、0 例 `declared > ours`、1941 例精确相等**；
59 侧因卡键 `void` 无法映射而跳过（见 §2.11）。
⇒ 这条同时证明两件事：**该字段的语义** 与 **我们的费用表与线上一致**。
⚠️ **但必须补一句限定**：**FL 自己并不读这个字段**（§2.12 ⑤，`grep` 0 命中）——
它宁可起真引擎重算圣水。所以这条验证的价值是"**格式语义与我们的卡表对齐**"，**不是**"这个字段有用"。

### 2.11 ★ 卡覆盖：**178 个卡键里 1 个我们认不出 —— `void`**

| 检查 | 读数 |
|---|---|
| actions 表全部 `card_key`（384,275 行） | **122** 个不同值 → 可用 **121** / 不可用 **1（`void`）** |
| 1000 局卡组 `deck[].card_key` | **178** 个不同值 → 可用 **177** / 不可用 **1（`void`）** |
| `void` 影响面 | **59 / 1941 侧（≈3.0%）** 的卡组含它 |
| 我们引擎 | `Card('Void')` → **KeyError**；`resolve_card('void')` → **None**；引擎 `card_data`（**212 名**）**不含 'oid'** |
| ⚠️ 但 | `src/clasher_new/cards.json`（123 条官方卡）**含 `"name":"Void", id 28000023, elixirCost 3`** ⇒ **仓内有更新的卡表数据，引擎吃的却是另一份** |

⇒ 对「用真人数据做 IL」这是一个**硬缺口**（3% 的卡组无法表示），同时也是**一条线索**（仓内已有 Void 的数据行）。

---

### 2.12 ★★ FL 自己怎么读这套格式（一手取证，`E:\FirstLight_CR` HEAD `28d66cc`）

> 这一节是**权威答案**：数据集是 FL 的，所以"哪些字段是载荷"由**它的读取代码**定义。
> FL 官方文档亦自述（`docs/training.md:29`）：「The cache builder reads `replays/part-*.parquet`… decks, forms, tower troops, 20 Hz action times, sides, coordinates, **source-side markers in `source_fields.data_i`**, and the replay end time.」

**① 三个读取入口，IL 生产只有 1 个**

| 入口 | 位置 | 读的列 / 用途 |
|---|---|---|
| **A（IL 主路径）** | `native_runner/training/v4/dataset.py:107-117` `ParquetReplayStreamV4.__iter__` | **只投影 `["replay_tag","payload_json"]`**，且**只取值 `payload_json`**（`replay_tag` 列**值被丢弃**，真 tag 从 payload 内 `source.replay_tag` 取） |
| B | `build_ppo_deck_pool.py:19-32`（`--dataset-root` 在 `:55`） | 只看 deck/tower，不看 events |
| C | `royaleapi_replay.py` 控制台/列表侧 | 额外读 `requested_player_tag` / `played_at_utc` / `team_tags`（**不进训练**） |

⚠️ **`actions/` 子目录：FL 生产代码里没有任何 reader 触碰它**
（`grep -arn "'actions'\|\"actions\"\|actions/" --include=*.py native_runner/` 排除 tests → 命中全是缓存/张量里的 `actions` 字段，**无一处目录 glob**；所有 parquet 读取都锚在 `replays/`）。
⇒ **§2.4 那张 21 列 actions 表，FL 的 IL 管线根本不读** —— 它服务的是别的消费者（浏览器/分析）。
⇒ 这也**纠正了我上一轮探针的隐含口径**：我用的是 actions 表的 `native_x/native_y`，而 FL 用的是 payload 里的 `grid_cell_floor`。两条路**都能到位**（见 ③），但不是同一条。

**② FL 的调用链（IL 缓存生产）**
```
cache_builder.main → _select_partition → dataset.ParquetReplayStreamV4.__iter__   （只产 payload）
→ prepare_collected_replay(payload)        royaleapi_replay.py:1783   ← 唯一字段消费点
→ calibrate_collected_replay_deal(...)     royaleapi_replay.py:596    ← 起真 native 引擎探测发牌
→ require_accepted_timeline(...)           expert.py:135
→ produce_il_replay_batch(...)             producer.py:435            ← 起真引擎逐步重演，产出观测
```
`prepare_collected_replay` 之后，**下游再也不看 payload**。

**③ 落点：FL 读的是 `grid_cell_floor` + `subcell_offset_from_floor_center`，并自己做翻转**

```python
royaleapi_replay.py:1928   grid = coordinates.get("grid_cell_floor")
                    :1929   offset = coordinates.get("subcell_offset_from_floor_center")
                    :1932   target_grid, subcell_offset = _oriented_grid_target(grid, offset,
                                flip_horizontal=flip_horizontal, flip_vertical=flip_vertical)
```
翻转由 `source_fields.data_i` 决定（`:1801-1817`）：
`data_i == 0 → bottom_side="team", flip_horizontal=False, flip_vertical=True`；
否则 `bottom_side="opponent", flip_horizontal=True, flip_vertical=False`。
格→世界单位由 FL 自己算（`replay_viewer.py:346-349` → `arena.py:158-166` 的 `x*1000+500`），再加亚格偏移。
- ★ **`native_world_units` / `raw_royaleapi_units` / `display_royaleapi_units` / `inversion_applied` / `arena_label_from_blue` 全部 `grep` 0 命中 ⇒ FL 不读。**
- ★ **`inside_18x32_arena` 是硬门禁**：非真直接 raise（`:1875-1876`）。
- ⇒ 上一轮的 U1（"raw→native 完整变换规则未定"）**对 FL 不重要**（它不读那几个字段）；对我们**已由 A/B 判定用 `native_*`**（§2.5，4001/4001 的 `native=(display.x, 32000−display.y)`）。

**④ 时间：三步整数换算**
`native_tick = source_tick + 1`（`:54`，命令边界 → 引擎可观测 tick）；
决策帧 = `90 + floor((tick−90)/5)*5`（`expert.py:26-27, :90`，`FIRST_POLICY_DECISION_TICK=90`、`POLICY_DECISION_TICKS=5`）；
**首决策帧 = tick 90 = 4.5 s，每决策帧 5 tick = 250 ms**（`producer.py:143-151` 先空跑到 90，不一致直接 `RuntimeError`）。

**⑤ ★★ 逐事件圣水/塔血：不是读来的，是"起真引擎重演"出来的**
- `average_elixir` / `elixir_leaked` / `four_card_cycle_elixir` / `replay.aggregate_stats.*`：**FL 全部不读**（`grep` 0 命中）。
- 圣水唯一来源是**引擎观测**：`producer.py:159-166` 取 `observation.players[].elixir_exact`；每帧写进张量（`tensorizer.py:2058`）。
- 而且**初手与循环牌序是"编造 + 校准"**：
  ① 用 `sha256(f"{replay_tag}|owner={owner}|{salt}")` 播种随机，生成一组**与公开出牌序列相容**的初手 4 + 循环 4（`royaleapi_replay.py:1395-1469`）；
  ② **起暂停的真 native match** 读出真实发牌，把编造牌序**重映射**到引擎实际 `deckSlot`（`calibrate_collected_replay_deal:596-714`）；
  ③ 再把公开动作**按 tick 调度进真引擎**（`producer.py:107/157`），**观测全部来自引擎**。
⇒ 这三条**独立证据**证实我上一轮的推断：**"动作 → 真引擎重建"是这条路线的唯一实现方式**，而且比我想的更绕（要先猜一手相容的牌序）。

**⑥ 质量门禁：只有 2 个真门禁，其余都是标签/读数**

| 机制 | 是否真门禁 | 证据 |
|---|---|---|
| **`require_accepted_timeline`** | ✅ **真门禁**：`overflow_window_count == 0 and early_action_count == 0`，否则 raise（`expert.py:135-145`，两处调用 `cache_builder.py:204` / `producer.py:106`） | 直接 raise |
| **终局保真** | ✅ **真判据**：引擎 `winner`/`crowns` 与 `battle.result`/`crowns` 不符 ⇒ `completed=False` + `first_untrusted_tick`（`producer.py:284-292`）。⚠️ 用的是 **result + crowns**，**不是**塔血 | `:276-288` |
| 预检 `ReplayPreflightV4.accepted` | ❌ **只产标签**：`_select_partition` **不按它过滤**，被拒行**仍进引擎**（`cache_builder.py:451-457`；有测试固化 `tests/test_training_v4_cache.py:49-75`） | 无过滤分支 |
| `validation.*` / `warning_count` | ❌ **不读**（0 命中） | — |
| **失败时的处理** | ⭐ **丢标签、保帧**：`gate_loss=False` + `trusted_decision_mask` 保留可信前缀，**不丢整局**（`producer.py:190-203`、`_finalize_il_lane:383-390`） | — |

**⑦ 最小字段集（FL 自己的权威样例）**
`tools/make_demo_data.py:34-39`（被 `docs/training.md:31` 称为"完整最小示例"）只有
`schema_version` / `source.replay_tag` / `battle` / `events` / `replay.duration.timeline_seconds` 五项。
**必须有**：`source.replay_tag`(兼作发牌种子)、`battle.<side>.players`（**恰好 1 人**）、`deck[8].card_key`（含 `-ev1`/`-hero` 后缀）、`events[]` 的 `kind/side/replay_tick_20hz/card_key`、**`source_fields.data_i`（唯一被读的 source_fields 子键，缺则 raise）**、`coordinates.inside_18x32_arena`、`coordinates.grid_cell_floor`、`replay.duration`。
**可忽略**：所有圣水/塔血聚合量、`final_tower_hitpoints`、`validation`、`warning_count`、`card_play_number`、`card_counts`、`native_world_units` 等。
⚠️ **两个例外别误删**：`ability_source_candidates`/`ability_source_authoritative`（能力事件归因）与 `deck[].level`/`tower_card.level`（等级众数 → `level_cap`）。

---

### 2.13 ★ 三个直接问答：终局结果 / 塔血 / 塔血是否逐次记录

> 复跑：`/usr/bin/python3 scripts/_il_replay_format_scan.py --check-terminal-state --actions /tmp/il_actions.parquet --replays /tmp/il_replays.parquet --out docs/il_replay_probe_2026-09-19/format_terminal_state.json`
> 覆盖：分片0 的 **5,000 局 / 384,275 事件**；另在分片 25/51 **逐事件穷举复核**，合计 **10,852 局 / 802,854 事件**。

#### Q1 「包括最终对局结果吗？」—— ✅ **有，而且有两份**

| 位置 | 字段 | 实测 |
|---|---|---|
| payload | `battle.result` ∈ {`victory`,`defeat`,`draw`} + `battle.<side>.crowns` | 5,000 局：**victory 3349 / defeat 1617 / draw 34** |
| parquet 列 | `result` / `team_crowns` / `opponent_crowns` | 与 payload **5,000/5,000 完全一致**（无口径分歧） |
| FL 读哪份 | **读 payload 那份** | `_winner`（`royaleapi_replay.py:1721-1727`）+ 终局保真校验（`producer.py:276-288`）；**parquet 列 FL 不读** |

#### Q2 「包括塔血吗？」—— ✅ **有，但只有「终局」一份**

`battle.<side>.players[0].final_tower_hitpoints = {king, princess_left, princess_right, total}`

| 检查 | 读数 |
|---|---|
| 字段缺失 | **0**（5,000 局 × 双侧） |
| 恒等式 `total == king + princess_left + princess_right` | **10,000 / 10,000 成立**（0 反例） |
| 取值形态 | **绝对值、连续**（king **2,210** 个不同取值 / princess **4,062** 个）⇒ 不是分档近似 |
| 满塔血（该等级未受损） | king **7,728**（4,305 次）/ princess **4,858**（1,413 次） |
| `0` 的含义 | 塔被摧毁（king **775** 次 / princess **6,854** 次） |
| 等级相关 | `tower_card.level` 分布 = **16**(9004) / 11(938) / 0(42) / 15(15) / 12(1) |
| ⚠️ FL 的 IL 主路径 | **不读**（§2.12 ⑤）；只有**旁路**实验工具 `build_hog_expert_manifest.py:57-64` 用它挑「小分差负局」做专家样本 |

#### Q3 「塔血在每次变化时有记录吗？」—— ❌ **没有。一条都没有。**

三条独立证据（**跨 10,852 局 / 802,854 事件穷举**）：

| # | 证据 | 读数 |
|---|---|---|
| ① | **事件键并集**（遍历每局的**全部**事件，不是抽样） | **恰好 13 个键，且 100% 恒存在**；其中**没有任何** `hp` / `health` / `tower` / `damage` 字段（`per_event_tower_fields = []`） |
| ② | **`kind` 只有 2 种** | `play_card` / `activate_ability` ⇒ **不存在「塔被破坏」或「塔掉血」这一类事件** |
| ③ | **整份 payload 的关键词路径只有 4 类** | `battle.<side>.crowns`（终局皇冠）、`...final_tower_hitpoints`（**终局**塔血）、`...tower_card`（塔兵**类型**，不是塔血）、`replay.card_counts.*.{bomb,inferno}-tower`（**那是卡名**） |
| ④ | `actions` 表的 `event_json` 列 | 解析出的键 = 与 `events[]` **完全相同**的 13 个 ⇒ **没有未展开的隐藏字段** |

⇒ **`final_*` 里的 "final" 是字面意思**：塔血只有**对局结束时的一份快照**，
**既没有逐次变化记录，也无法从事件流反推「何时掉的塔」**（事件里既无塔的血量、也无单位状态、也无伤害事件）。
⇒ 想拿「塔血随时间」，**唯一路径 = 起引擎重演**（这正是 FL 的做法，§2.12 ⑤）。

#### ★ 附带查出：我们引擎的**塔血等级阶梯与线上不一致**（lv11 一致，lv16 不一致）

| 塔等级 | 线上（数据集实测满塔血） | 我们引擎（**实体** HP） | 一致？ |
|---|---|---|---|
| **lv11** | king **4824** / princess **3052** | **4824 / 3052** | ✅ |
| lv12 | —（该分片只 1 侧） | 6744 / 3934 | 无法比 |
| lv13 | —（0 侧） | 7416 / 4326 | 无法比 |
| lv14 | —（0 侧） | 8136 / 4746 | 无法比 |
| lv15 | —（15 侧，未取到满血值） | 8928 / 5208 | 无法比 |
| **lv16**（数据集 **9,004/10,000 侧**） | king **7728** / princess **4858** | king **9816** / princess **5726** | ❌ **不一致** |

- 数据集**全局最大塔血 = 7728**；我们 lv16 的 **9816 在数据集里一次都没出现**；
- ⇒ **至少 lv16 的塔血表与线上不符**（lv11 相符）。**成因未定**（【R10】）：可能是平衡性改动、
  等级口径不同（`tower_card.level` 是**塔兵卡**等级，是否等同王塔等级**未验证**），或我们的表有误。
- ⚠️ **影响面**：`_TOWER_HP_ANCHOR = 10928` 恰等于 **我们 lv11 的满塔血**（4824 + 3052×2）。
  若在 lv16 用我们的表重放真人局，单侧塔血尺度是 **21,268 vs 线上 17,444（高 ≈22%）**
  ⇒ 会直接改变 `tower_dmg_*` / `tower_premium_k` 的归一化口径。
- ⚠️ 另一处**易踩的坑**（本轮实测）：`PlayerState.king_tower_hp` 等**字段在构造时恒为 lv11 值（4824/3052）**、
  **不随 `card_level` 变**；真正的等级缩放在**实体**上，字段要经 `BattleState.update_player_hp()`
  （`battle.py:2796`；`step()` 在 `game_over` 早退之后**第一件事**就是它）才同步。
  ⇒ 直接读 `p.king_tower_hp` 拿等级化塔血**会读到错值**；而 `rl/replay.py:134-135` 写的 `towers0/towers1`
  **正是这两个字段** ⇒ 只有在"至少推进过一帧"的帧上才是真值。

---

## 3. 我们自己的联赛录像格式（schema 5，源码为权威）

### 3.1 顶层与 meta

```
{"schema": 5, "games": [{"meta": {...}, "winner": int, "frames": [...]}, ...]}
```
（写入点 `rl/replay.py:168-173` `save_league_replays`）

| `meta` 键（实测并集，20 局） | 含义 |
|---|---|
| `pair` | 双方策略名（如 `['main','push_flow']`） |
| `side0` | 哪一侧是 player 0（主 policy） |
| `max_steps` | 本局步数上限（如 360） |
| `decks` | 双方卡组（8 卡名 ×2） |

⚠️ ★ **`meta` 没有 `seed`**（实测 20 局键并集只有这 4 个）⇒ **录像不能确定性重放**。

### 3.2 ★ 帧 **16 键**（`rl/replay.py:129-165` + `rl/run_league.py:130-144`）

| 键 | 类型 | 含义 / 来源 |
|---|---|---|
| `t` | float | 局内秒（`battle.time`，2 位小数） |
| `bundle` | list | **我方动作包**：`[(kind, slot, x, y), ...]`（`sa.kind/slot/x/y`） |
| `reward` | float | 本帧奖励 |
| `opp_played` | list/None | 对手出的牌 `{card, x, y}` |
| `towers0` / `towers1` | list[3] | 双方 `[王塔, 左公主, 右公主]` 血 |
| **`elixir0` / `elixir1`** | float | ★ **双方逐帧圣水** —— 正是人类回放**缺**的东西 |
| `et` | list/None | S2 局面交换逐帧明细（**可选**） |
| `et_src` | str/None | `"online"` / `None`（**可选**） |
| `crown0` / `crown1` | int | 双方已丢皇冠数 |
| `v0` / `v1` | float/None | 在线**残值真值**（`RLEnv._active_v`；缺省 None ⇒ 读方回落离线重建） |
| `entities` | list | 存活实体，**15 元组**（见下） |
| `cards` | list | **可选**：本步我方打出的卡名（`run_league.py:139-143`；旧录像无此键）。实测 **`cards` 键恰在全部非空 `bundle` 帧出现**（231/231） |
| `et_tail` | dict | **可选，只写进最后一帧**：局末 flush 的尾部窗口明细（`run_league.py:203-209` / `:240-249`）。⚠️ 抽 5 个 schema-5 文件**0 帧命中** ⇒ 触发条件**未验证** |

### 3.2.1 ★ 三个**版本号/口径**必须区分

| 常量 | 值 | 位置 | 用途 |
|---|---|---|---|
| `LEAGUE_REPLAY_SCHEMA` | **5** | `replay.py:82` | **联赛录像**（本节的对象） |
| `SCHEMA_VERSION` | **2** | `replay.py:18` | **`EpisodeReplay`**（含 `obs` 的信念监督录像，**另一套**，不要混用） |

**实体元组长度按 schema 分档（实测）**：schema **3 → 10**、schema **4 → 12**、schema **5 → 15**。
⇒ 读方**必须**按 `schema >= N` 判定（`replay.py:89/96-98`），否则下标错位。
⚠️ **但公共读函数 `load_league_replays()` 把版本号丢了**（`replay.py:176-183` 只返回 `games`）
⇒ 用它读的消费者**看不到 schema 号**，只能自己 `pickle.load` 或按实体元组长度反推。这是**读侧的结构性缺口**。

### 3.2.2 ★ 最有行动价值的一条：**logprob / value / masks 算了但被丢掉**

策略采样点返回 5 个东西：`Follower.act(...) -> (ActionBundle, logprob, value, hidden, masks)`（`follower.py:433-437`），
而调用点是：

```python
bundle, _, _, hidden, _ = policy.act(...)      # rl/run_league.py:190
```

⇒ **`logprob` / `value` / `masks` 三个都写成了 `_` 丢掉**。其中：
- **`masks`**（32×18 合法格 + 合法项）**是 IL 训练的必需项**，采样时**现成算好**；
- **`logprob`** 是重要性比（`ratio`）的必需项 ⇒ 没有它，录像**不能用于 off-policy 训练**。

⇒ 结论：我们缺的不是"算不出来的东西"，而是**三个已经算出来但没落盘的量**。这与人类回放那边"状态根本没记录"是**完全不同性质**的缺口。

### 3.3 ★ 实体 **15 元组**（`rl/replay.py:106-162`，逐位）

`[name, x, y, hp, player, kind, max_hp, shield, shield_max, radius, id, target_id, root_cast, is_product, share]`

- 下标 **0–9** = 旧格式（`kind ∈ troop/building/projectile/effect`）；
- 下标 **10–11** = **schema 4 追加**（`id` / `target_id`）⇒ 离线可重建索敌关系图；
- 下标 **12–14** = **schema 5 追加**（`root_cast` / `is_product` / `share`）；
- **向后兼容规则**：一律**追加在末尾**，读方按 `schema >= N` 判定（`replay.py:83-89`）。

实测：`['King_PrincessTowers', 3.5, 25.5, 3052.0, 1, 'building', 3052.0, 0.0, 0.0, 1.0, 1, 3, None, False, 0.0]`，**长度 15** ✅

### 3.4 ★ 两条**不同的**"录像"链路（容易混淆）

| | 联赛录像（落盘） | Episode 录像（内存） |
|---|---|---|
| 写入器 | `rl/replay.py::battle_snapshot` + `save_league_replays` | `rl/replay.py::EpisodeReplay.record_step(obs, ...)`（`:32-54`） |
| **是否含 `obs`** | ❌ **不含** | ✅ **含 `"obs": obs`**（`:37`） |
| 用途 | 联赛回放 / dashboard / 离线仪器 | `to_belief_dataset()`（`:70`）信念监督 |
| 结论 | 落盘录像**没有观测网格** | 内存里有，但**不落盘** |

⇒ **"我们有两套录像"这件事本身是格式理解的关键**；`runs/*/replays/*.pkl` 是前者。

### 3.5 同一 schema 号下的**可选键差异**（读方必须容忍）

| 文件 | 时间 | `et`/`et_src` | `cards` |
|---|---|---|---|
| `runs/et_solo100k/replays/league_100000.pkl` | 2026-09-18 晚 | ✅ 有（实测 `[0.0,0.0,0.0,0,0.0,0.0]`） | ✅ 有 |
| `runs/run_schema5/replays/league_0.pkl` | 2026-09-18 05:52 | ❌ **无** | ✅ 有 |

⚠️ 且**发现一处注释与实现不一致**：`rl/replay.py:139` 写 `engagement_trade_detail` 是 **4 元组**
`[phi_part, tau, score, n_windows]`，而**实测 `et` 是 6 元组** `[0.0,0.0,0.0,0,0.0,0.0]`。
⇒ 记为**待核**（未定性：可能后来扩到 6 项而注释未改）。

---

## 4. ★ 逐字段对照：IL / 训练到底需要什么，谁给得起

| 需要的东西 | 人类回放（IL_Replay） | 我们 schema 5 | 备注 |
|---|---|---|---|
| **动作（卡 + 落点 + 时间）** | ✅ `card_key` + `native_x/y` + `tick@20Hz` | ✅ `bundle(kind,slot,x,y)` + `t` | 两边都够 |
| **对手动作** | ✅ 同表（`side`） | ✅ `opp_played` | 两边都够 |
| **逐帧圣水** | ❌ 只有 `average_elixir` / `elixir_leaked` / `four_card_cycle_elixir`（**聚合**） | ✅ `elixir0/1` | ★ **我们的强项** |
| **塔血随时间** | ❌ 只有 `final_tower_hitpoints`（**终局**） | ✅ `towers0/1`（逐帧） | ★ 我们的强项 |
| **实体状态（位置/HP/索敌）** | ❌ 完全没有（`"hp"` `grep` = 0） | ✅ `entities` 15 元组含 `id`/`target_id` | ★ 我们的强项 |
| **奖励** | ❌ 无 | ✅ `reward` | |
| **价值估计** | ❌ 无 | ✅ `v0/v1` | |
| **32×18 观测网格** | ❌ | ❌（`replay.py:102` 明说"不含 32×18 观测网格，体积可控"） | **两边都缺**，但**性质不同**：我们是从实体快照重建，人类回放是**根本不记录状态** |
| **掩码 mask** | ❌ | ❌ 落盘 —— ★ **但采样时现成**（`follower.py:433` 第 5 个返回值），被 `run_league.py:190` 的 `_` **丢弃** | ★ 我们是"没落盘"，人类回放是"不存在" |
| **logprob** | ❌ | ❌ 落盘 —— ★ 同样**算了但被丢**（`run_league.py:190`） | 没有它 ⇒ **录像不能用于 off-policy/重要性比** |
| **critic value** | ❌ | ❌（`v0/v1` 是 `RLEnv._active_v` = **场上部署价值 Σ份额**，`env_wrapper.py:132`，**不是** value head 输出） | ⚠️ 易误读：字段名 `v0/v1` 看起来像 critic |
| **seed（确定性重放）** | —（不需要） | ❌ **meta 无 seed** | ★ 我方缺口 |
| **卡等** | ✅ 每卡 `level`（实测全 16） | ❌（全局 `card_level`） | ★ 对方强项 |
| **觉醒 / 英雄变体** | ✅ `-ev1` / `-hero` 后缀 | ❌（录像里未见） | ★ 对方强项 |
| **第 9 张塔兵** | ✅ `tower_card`（4 值） | ❌ | ★ 对方强项 |
| **卡组** | ✅ `deck[]`（8 张，含名字） | ✅ `meta.decks`（名字） | |
| **game_mode / 结果 / 皇冠** | ✅ | ✅ `winner` + `crown0/1` | |
| **时长** | ✅ `duration` | 末帧 `t` | |

---

## 5. 结论

1. **格式互补**：人类回放 = **动作 + 卡组元数据**；我们 = **状态流 + 奖励/价值**。
   **没有一方是另一方的超集** ⇒ 「拿真人数据直接换掉我们的录像」或反之都**不成立**。
2. **两边都要"重建"**：
   - 人类回放 → 必须**重建状态**（圣水/单位实况）；这正是 FirstLight 写 `cache_builder`（起真引擎重建对局）的原因；
   - 我们自己的录像 → 要训练仍需**重建观测网格**（录像只存实体/圣水/塔血，不存 32×18 网格），
     而**meta 没有 seed** ⇒ **不能确定性重放** ⇒ 重建只能靠"从实体快照重算 obs"（可行性**未验证**）。
3. **对接成本主要在三处**：① `void` 这张卡我们引擎没有（3% 卡组）；② `game_mode` 需要过滤（Showdown/Normal Battle）；
   ③ 时长尾部 > 300 s 与我们 300 s 上限冲突。
4. **可直接复用**：坐标（`native_x/native_y` **就是**我们的世界坐标，A/B 已判）、网格 18×32、`tower_card` ↔ `set_tower_troop`、
   费用表（`four_card_cycle_elixir` 1941/1941 相等）、`card_aliases.resolve_card`（177/178）。

---

## 6. 本轮查出的问题（含**我自己的**两处错）

| # | 问题 | 性质 |
|---|---|---|
| 1 | **`void` 卡缺失**（引擎 212 名里没有；`cards.json` 123 条里**有**） | 数据/卡表缺口，影响 3% 卡组 |
| 2 | `replay.py:139` 注释写 `et` 是 **4 元组**，实测 **6 元组** `[phi_part,tau,score,n_windows,tau0,tau1]`（构造点 `env_wrapper.py:583-585`） | ✅ **已定性**：docstring 陈旧（非实现问题） |
| 3 | `meta` **无 seed** / 无时间戳 / 无 git hash | 格式缺口（录像**不可确定性重放**） |
| 4 | **`load_league_replays()` 把 schema 号丢掉**（`replay.py:176-183` 只返回 `games`） | 读侧结构性缺口：用公共读函数者**看不到版本**，会被迫按元组长度猜 |
| 5 | **`logprob`/`value`/`masks` 在 `run_league.py:190` 被 `_` 丢弃** | ★ 最有行动价值：三者**都已算好**，落盘即可 ⇒ 缺的不是能力而是**存盘决定** |
| 6 | `et_tail`（只写进最后一帧的可选键）在抽样的 5 个 schema-5 文件里 **0 帧命中** | 触发条件**未验证**（`run_league.py:203-209`/`:240-249`） |
| 7 | **我自己错**：把 `runs/run_schema5/replays/league_0.pkl` 先当成"我们的录像格式"，实际它**缺 `et`/`et_src`** ⇒ 同 schema 号下有版本内差异；权威口径必须回到 `rl/replay.py` 源码 + `et_solo100k` 的真实文件 | 已按事实改正（本文件 §3.2/§3.5） |
| 8 | **我自己错（上一轮）**：曾推断人类回放需要 `32−y` 翻转；本轮由 `native = (display.x, 32000−display.y)` **4001/4001** 的实测变换**否证**，并用 A/B 确认 `native_x/native_y` 直用 | 已被仪器否证 |
| 9 | `form_at_play` / `ability_source_authoritative` / `deck_card_key_candidates` **三个字段是常量** | 按字段数估值会高估数据（§2.6） |
| 10 | `v0/v1` 字段名像 critic 的 value，实际是"场上部署价值 Σ份额" | ⚠️ 误读风险（写进 §4 备忘） |

---

## 7. 复现命令（原样可跑）

```bash
R=/mnt/e/clash-royale-simulator-main

# 0) 拉分片（示例：0 / 25 / 51）
for n in 000000 000025 000051; do
  curl -sL -o /tmp/il_actions_$n.parquet "https://hf-mirror.com/datasets/VanguardX101/IL_Replay/resolve/main/actions/part-$n.parquet"
  curl -sL -o /tmp/il_replays_$n.parquet "https://hf-mirror.com/datasets/VanguardX101/IL_Replay/resolve/main/replays/part-$n.parquet"
done

# 1) 元数据（格式的权威说明：脱敏范围 / 文件 sha256 / 兼容性抽查）
curl -sL -o /tmp/il_readme.md       "https://hf-mirror.com/datasets/VanguardX101/IL_Replay/raw/main/README.md"
curl -sL -o /tmp/il_manifest.json   "https://hf-mirror.com/datasets/VanguardX101/IL_Replay/raw/main/manifest.json"
curl -sL -o /tmp/il_compatibility.json "https://hf-mirror.com/datasets/VanguardX101/IL_Replay/raw/main/compatibility.json"

# 2) 结构（抽 N 局，逐路径类型/长度/样例）
/usr/bin/python3 $R/scripts/_il_replay_format_scan.py --schema \
    --actions /tmp/il_actions_000000.parquet --replays /tmp/il_replays_000000.parquet --n 3 \
    --out $R/docs/il_replay_probe_2026-09-19/format_schema_s0.json

# 3) 字段出现率与取值分布（1000 局）
/usr/bin/python3 $R/scripts/_il_replay_format_scan.py --stats --limit 1000 \
    --actions /tmp/il_actions_000000.parquet --replays /tmp/il_replays_000000.parquet \
    --out $R/docs/il_replay_probe_2026-09-19/format_stats_s0.json

# 4) four_card_cycle_elixir 语义验证（对 1000 局 × 双侧复算最便宜 4 张）
/usr/bin/python3 $R/scripts/_il_replay_format_scan.py --check-cycle-elixir --limit 1000 \
    --actions /tmp/il_actions_000000.parquet --replays /tmp/il_replays_000000.parquet \
    --out $R/docs/il_replay_probe_2026-09-19/format_cycle_elixir.json

# 5) 我们自己的录像格式（源码为权威 + 真实文件对照）
sed -n '100,175p' $R/src/clasher_new/rl/replay.py
$R/.venv/Scripts/python.exe -c "import pickle;d=pickle.load(open('runs/et_solo100k/replays/league_100000.pkl','rb'));print(d['schema'],sorted(d['games'][0]['frames'][0].keys()))"
```

---

## 8. 未验证 / 待外部信息

| # | 项 | 状态 |
|---|---|---|
| U1 | `raw_royaleapi_units` → `native_world_units` 的完整变换规则 | **对我们已够用**：`native = (display.x, 32000−display.y)` **4001/4001** 成立（§2.5），且 A/B 判定用 `native_*`；**FL 侧不读这三个字段**（§2.12 ③）⇒ raw→native 的完整规则**仍无需求**（从"必须查"降级为"可查可不查"） |
| U7 | FL 侧 consumer 的字段清单（哪个字段真正被读） | ✅ **已闭合并写进 §2.12**（一手取证：IL 主路径只读 `payload_json`，落点用 `grid_cell_floor`+`data_i`，圣水靠引擎重演，`validation`/`warning_count` 不是门禁） |
| U2 | `et` 到底是 4 元组还是 6 元组（注释 vs 实测） | ✅ **已定性**：实测 **6 元组**，构造点 `env_wrapper.py:583-585` ⇒ `replay.py:139` 的 docstring **陈旧**（见 §6 第 2 条） |
| U2b | `et_tail` 的触发条件（写点存在但抽样 0 命中） | **未验证** |
| U2c | `v0/v1`（`_active_v`）是否数值上等于某次 forward 的 critic value | **未验证**（已确认二者**不同源**） |
| U3 | 我们的录像缺哪些卡组元数据（觉醒/英雄变体是否在别处） | 本轮**未逐源确认**（只说"录像里未见"） |
| U4 | 从 `entities` 快照重算 32×18 观测网格的可行性 | **未验证**（且 **无 seed** ⇒ 不可确定性重放） |
| U5 | `game_mode` 各取值与我们引擎建模的对应关系 | 只列了分布，**未逐模式判定** |
| U6 | 其余 50 个分片是否与本轮 3 个分片同构 | 只抽 3/104 分片（`card_counts` 之外的路径恒存在这一点在 2 个分片上成立） |

