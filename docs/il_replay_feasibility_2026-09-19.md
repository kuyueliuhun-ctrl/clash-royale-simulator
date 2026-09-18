# 公开人类回放 → 我们的引擎：**重放可行性最小验证**（2026-09-19）

> **问题**：`VanguardX101/IL_Replay`（公开人类 CR 回放，**动作侧**）能不能驱动 `src/clasher_new` 的引擎？
> 这决定「我们无人类数据」能否从 **无解** 变成「**有输入、缺管道**」。
> **数据**：`hf-mirror.com` 的 `VanguardX101/IL_Replay`，`actions/part-000000.parquet`（**22,697,255 B**）
> + `replays/part-000000.parquet`（**17,432,059 B**）；取数 **2026-09-19**；
> 该数据集公开、非 gated、**许可未声明**（`license=None`，HF API 复核）。
> **工具**（本轮新增，均只读）：
> `scripts/_il_replay_extract.py`（pyarrow，用 **WSL `/usr/bin/python3`** —— Windows venv 无 pyarrow）
> + `scripts/replay_feasibility_probe.py`（引擎侧重放，用 `.venv/Scripts/python.exe`）。
> **纪律**：这是**可计算性**检查，**不是等价性判据**；所有读数带分子/分母，不升格为判据（【R3】）。

---

## 0. 一句话结论

**数据侧与我们模型的下游几乎逐项对齐（卡键 100% 可映射、卡组 8/8 可重建、手牌循环费用 6/6 精确相等、
坐标无需变换），但「直接重放」只能走通 35%–62% 的动作** ——
缺口**不在数据格式**，而在**逐事件状态不可观测**（圣水、手牌顺序、卡牌实况）。
⇒ **能用于 IL，但必须先做「状态重建」（起真引擎 + 约束求解），不能当作"录像回放"直接用** ——
这正是 FirstLight 为什么要写 `cache_builder`（「起真引擎重建对局」）而不是简单回放。

---

## 1. 数据规模（实测）

| 项 | 读数 |
|---|---|
| actions 分片行数 | **384,275**（`play_card` **365,362** + `activate_ability` **18,913**） |
| 该分片内不同对局数 | **4,997** |
| replays 分片行数 | **5,000** |
| 抽检 3 局的时长 | **311.55 s / 300.55 s / 309.05 s**（⚠️ **全部超过我们引擎 300 s 硬上限**） |
| 全局声明（HF README） | 252,238 局 / 17,836,160 动作（**未逐分片核对**，本验证只覆盖 1/52 分片） |

---

## 2. 六个读数（抽检 3 局；写死分子/分母）

### R1 事件卡键可映射率 —— **100%**

| 局 | 可映射 / 出现 | 重命名映射（RoyaleAPI key → 我们卡名） |
|---|---|---|
| #1 | **14/14** | `fire-spirit→FireSpirits`、`giant-snowball→Snowball`、`magic-archer→**EliteArcher**`、`the-log→Log` |
| #2 | **15/15** | 同上 + `ice-spirit→IceSpirits` |
| #3 | **13/13** | `fire-spirit→FireSpirits`、`ice-spirit→IceSpirits`、`the-log→Log` |

⚠️ **两条必须记录的细节**：
1. 我们仓内**本来就有** RoyaleAPI key 的别名表（`card_aliases.py::resolve_card`），映射是**现成的**；
2. `resolve_card` 对未知键**返回 `None` 而不是抛异常**（实测）—— 任何对账脚本必须显式判 `None`，
   否则会把"没映射上"记成"映射上了"。
3. 部分映射是**跨版本语义漂移**：`magic-archer → EliteArcher`、`guards → SkeletonWarriors`
   ⇒ 「映射成功」不等于「同一张卡」，须逐卡复核（**未做**）。

### R2 卡组可重建率 —— **8/8（两侧），含觉醒/英雄变体**

| 项 | 读数 |
|---|---|
| 卡组长度 | team **8** / opponent **8**（3 局全同） |
| 卡等 | **{16}**（单一值；我们探针取 `--level 16`） |
| 变体后缀 | `-ev1`（觉醒）×2–3 + `-hero`（英雄）×1 **每侧** ⇒ 走我们**已有**的 `PlayerState.set_evolution_slots()` / `set_hero_slots()` |
| 第 9 张「塔兵」 | `tower-princess` ⇒ 对应我们的**默认** Princess ⇒ **不需要** `set_tower_troop()` |
| 未映射 | **0** |

### R5 手牌循环费用一致性 —— **6/6 精确相等**（不需要引擎的自检）

回放给了 `four_card_cycle_elixir`；我们对**同一卡组**算「最便宜 4 张的费用和」：

| 局 | 侧 | 回放声明 | 我们复算 | 相等？ |
|---|---|---|---|---|
| #1 | team / opponent | 6.0 / 6.0 | 6.0 / 6.0 | ✅ / ✅ |
| #2 | team / opponent | 6.0 / 6.0 | 6.0 / 6.0 | ✅ / ✅ |
| #3 | team / opponent | 4.0 / 7.0 | 4.0 / 7.0 | ✅ / ✅ |

⇒ **卡组 / 卡等 / 费用表三者对齐**（这是本轮**最干净**的一致性证据，且完全可复算）。

### R3 引擎接受率 —— 坐标约定 A/B（**判定：不需要翻转**）

做法：按 `replay_tick_20hz` 升序，**先按 `DT=0.05 s` 推进引擎**（`BattleState.step`，让圣水回费与单位行为发生）
再 `deploy_card(pid, card, Position(native_x/1000, native_y/1000))`；`--level 16`。

| 局 | `raw` 接受率 | `flip_y`（`y:=32−y`）接受率 | 引擎终局时刻 |
|---|---|---|---|
| #1 | **70/173 = 40.5%** | 4/173 = 2.3% | 180.0 s（`game_over=True`） |
| #2 | **79/176 = 44.9%** | 4/176 = 2.3% | 205.6 s（`game_over=True`） |
| #3 | **102/166 = 61.5%** | 0/166 = 0.0% | 223.5 s（`game_over=True`） |

- **A/B 判定 `raw`**：`flip_y` 下失败**全部**进 `other`（位置非法）且 `unaffordable=0`
  ⇒ 翻转是错的。**这是我先写错的一条假设**：我原先按 `_mask_diff_snapshot.py` 的用例命名
  推断「pid0 拥有 y>16 一侧」，从而推出需要 `32−y`；**A/B 把它否证了**。
- 正向证据：回放 `grid_cell_floor=(8,0)` 与 `native_x/1000=8.5` 的关系，与
  `rl/action_bundle.py::sub_position(0, x, y) = Position(x+0.5, y+0.5)` **逐位吻合** ⇒
  **回放的 `native_x/native_y` 就是我们的世界坐标**。

### R4 失败归因（`raw` + level 16）

| 局 | `unaffordable` | `not_in_cycle` | `other` | 说明 |
|---|---|---|---|---|
| #1 | **86** | 0 | 17 | 缺 **逐事件圣水** ⇒ 我们只能从 5.0 起按 `2.8 s/圣水` 推 |
| #2 | **69** | 0 | 28 | 同上；`other` 主要是位置/规则 |
| #3 | **62** | 0 | **2** | 同上 |

⇒ **主缺口是"圣水不可观测"**（不是格式、不是卡表）。
且**被拒绝的部署不扣费** ⇒ 我们的圣水只多不少，漂移随时间累积（**方向已知，幅度未标定**）。

### R6 终局塔血 —— **明确记为「不可比」**

| 局 | 我们（`raw`，lv16） | 回放（终局） |
|---|---|---|
| #1 | team `king 9816 / princess [5726, 5726]`；opp `king 9556 / princess [−63, 5726]` | team `king 5980 / p 3110, 0`；opp `king 5980 / p 2462, 488` |
| #2 | team `9816 / [5151.8, 5634]`；opp `9816 / [−191, 4702]` | team `7728 / 1336, 4714`；opp `7728 / 4294, 0` |
| #3 | team `3564 / [−241, −88]`；opp `9560 / [−410, 3122]` | team `7504 / 3827, 1030`；opp `6832 / 4522, 0` |

- 我们 lv16 的**满塔血 = king 9816 / princess 5726**；回放的**终局**值（含满血侧如 #2 的 4714）
  **不在同一标尺** ⇒ **卡等 ↔ 塔血等级的映射与我们不同构**（我们的 `card_level` 同时驱动卡牌与塔血）。
- **四项不可比理由**（写死）：① 手牌循环**顺序未知**；② **逐事件圣水缺失**；③ 卡等/塔血等级未对齐；
  ④ 回放时长 **300.55–311.55 s** 而**我们引擎 300 s 硬上限**
  （且实测我们引擎在 **180–223.5 s** 就因**皇冠差规则**提前终局 ⇒ 重建已经发散）。
- ⇒ **R6 只作"我们能把它算出来"的可计算性读数，不得当等价性证据**（【R3】【R10】）。

---

## 3. 结论与边界

| 判定 | 内容 |
|---|---|
| ✅ **成立** | **数据侧与我们模型的下游高度对齐**：卡键（100%）/ 卡组（8/8 + 变体 + 等级）/ 费用语义（6/6 精确）/ 坐标（无需变换，A/B 判定）/ 第 9 张塔兵（= 默认） |
| ✅ **成立** | **引擎可以被人来动作驱动**：`BattleState.step` + `deploy_card` 在 `raw` 坐标下**接受 35%–62%** 的人类出牌，且失败**主要可归因** |
| ❌ **不成立** | **「录像回放 = 等价对局」**：逐事件状态不可观测（圣水/手牌顺序）+ 时长超我们 300 s 上限 + 皇冠规则提前终局 ⇒ **重建必然发散** |
| 🔶 **前置（未做）** | ① **许可**：数据集 `license=None` ⇒ **在你的拍板前不得用于训练**；② 我们此前未知该数据集是否与 FL 同源（时间戳与动作数吻合，**推断**） |
| 🔶 **下一步的最小件** | **状态重建**：用回放动作 + 我们的引擎做**约束求解/搜索**（例如以"每个出牌时刻必须付得起"为约束反推圣水轨迹），或直接照 FL 的 `cache_builder` 思路；**成本与可行性未评估** |
| ⚠️ **未做** | 未跑训练；未用 IL 数据训练；未核对全部分片（只抽 1/52 分片、3 局）；未做逐卡语义复核（`magic-archer→EliteArcher` 等） |

---

## 4. 复现命令（原样可跑）

```bash
# 0) 下载（一个分片；~40 MB）
curl -sL -o /tmp/il_actions.parquet  "https://hf-mirror.com/datasets/VanguardX101/IL_Replay/resolve/main/actions/part-000000.parquet"
curl -sL -o /tmp/il_replays.parquet  "https://hf-mirror.com/datasets/VanguardX101/IL_Replay/resolve/main/replays/part-000000.parquet"

# 1) 抽取（WSL python3：pyarrow 25.0.1；Windows venv 无 pyarrow）
/usr/bin/python3 scripts/_il_replay_extract.py \
    --actions /tmp/il_actions.parquet --replays /tmp/il_replays.parquet --n 3 \
    --out docs/il_replay_probe_2026-09-19/one_shard.json

# 2) 引擎侧重放（Windows venv python；含坐标 A/B 与失败归因）
.venv/Scripts/python.exe scripts/replay_feasibility_probe.py \
    --events-json docs/il_replay_probe_2026-09-19/one_shard.json --level 16 \
    --out docs/il_replay_probe_2026-09-19/probe_result_lv16.json

# 3) 留证（随本文件一并提交：docs/il_replay_probe_2026-09-19/）
ls -la docs/il_replay_probe_2026-09-19/
```
