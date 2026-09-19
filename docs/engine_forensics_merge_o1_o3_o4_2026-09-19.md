# 引擎取证合并报告：O1（过桥后行为）/ O3（己方半场横向对齐）/ O4（出兵圆环分道）

> **性质**：这是三份**互相独立**的只读取证的**合并**，不是第四份取证。
> **取数时间**：三份取证约 2026-09-19 07:45–08:00 CST；本合并 2026-09-19 08:0x CST。
> **仓库 pin**：**A（我方）** = `/mnt/e/clash-royale-simulator-main` @ `be0b47b`；**B（上游 jason）** = `/mnt/e/clash-royale-simulator-main-by-jason` @ `f616f19`。共同祖先（分叉点）= `f20fa4d`。
> **未改任何仓库源文件**；本合并只新增本文档与 `/tmp` 下的复跑脚本。

---

## §0 合成口径（必读；决定本文能说什么、不能说什么）

### 0.1 证据分级标记

| 标记 | 含义 |
|---|---|
| **【原文】** | 逐字来自三份取证的交付文本（含其表格、代码块、引文） |
| **【复核】** | 本次合成**只读核对**该引用的 `file:line` 处文本是否与描述一致 |
| **【复跑】** | 本次合成**重跑**了留存在 `/tmp` 的原始探针脚本（只读、内存内） |
| **【缺口】** | 三份取证的交付文本在该处被截断，或缺原始留证，**不可得** |

### 0.2 合成侧独立核验（覆盖范围必须明写）

对三份取证中出现的关键 `file:line` 引用，本次合成派了两个**只读**核验子任务逐条比对源文件：

- **A 侧**：**27/27 OK，0 MISMATCH**。
- **B 侧**：**22/22 OK，0 MISMATCH**。

**核验覆盖的是**：① 该行号处**确实存在**所描述的代码/文本；② 关键引文**逐字一致**；③ 若干 grep 的命中集合。
**核验没有覆盖**：运行期行为、任何数值口径、任何跨引擎的可比性。**"OK" 只等于"引用位置正确"，不等于"结论成立"。**

**核验发现的 3 处引用精度问题**（结论不变，表述需收窄）：

| # | 原文表述 | 核验所见 |
|---|---|---|
| P-a | O3：「全仓 grep `jumping_across_river` **只剩** `battle.py:33` 与 `248`、`334/337`、`562`」（指 B） | 注释行 `284/285/305/307` 同样命中；"只剩"不精确。"无置 True 处 ⇒ 恒 False"**仍成立** |
| P-b | O3：「全仓 grep `LEFT_BRIDGE\|RIGHT_BRIDGE` 在 `arena.py` 之外**零命中**（A、B 皆然）」 | **B 成立**；**A 不成立**：`scripts/probe_threat_trigger.py:50-51` 有同名标识符（探针自定义 x 区间元组，非该常量）。"**arena 常量无任何引用**"**仍成立** |
| P-c | O3：A「`battle.py:1-7` 的 import 列表里没有 `strategies`」 | 成立，且更强：**A 仓根本没有 `strategies.py` 这个文件** |

### 0.3 ⚠ 证据完整度声明（【R10】：不确定就写不确定）

**三份取证的交付文本在传入本合成会话时，全部被截断**；其原始报告**未落盘**（检索 `docs/`、`/tmp` 无同文）。截断点：

| 报告 | 截断位置 | 该报告缺失的内容（**不可得**） |
|---|---|---|
| **O1** | §3a，正在引 `A/battle.py:1188` 时中断 | §3a 后半（闩锁复位全文）、§3 的 3b、以及 §4 之后的全部章节（含其自带的"未验证"清单） |
| **O3** | §4，正在引探针输出 `deploy (9.0,13.5` 时中断 | §4 后半的探针输出、以及之后的全部章节（含其自带的"未验证"清单） |
| **O4** | §2.3 表格第 6 行（`i=6`）中断 | `bridge_right` 表 `i=7..14`、§3–§7（含其自带的"副作用与恢复"与"未验证"清单） |

**处理规则（本文严格执行）**：对缺口**不做概述、不做补齐、不做推断**；缺口逐条列入 §C。凡本文给出缺口处的内容，一律来自 **【复跑】** 并单独标注，**不冒充**原报告文本。

### 0.4 行文禁则（用户要求 ⑤）

本文**只写结构差异与可观测后果**。**禁止**任何"我方/上游引擎更好/更差/更拟合/更正确"的表述。
凡涉及方向性判断的，一律标注为**某份取证的原话**或**代码注释的自述**，不作为本文的结论。
用户前提中的措辞（如"过桥后走位混乱"）以「用户前提」形式引用，**不是**本文的评价。

### 0.5 交接摘要（便于快速定位）

- 三份取证的**全部关键 `file:line` 引用已被逐条核验命中**（A 27/27、B 22/22），并纠出 3 处**表述精度**问题（§0.2 P-a/P-b/P-c）。
- 三份取证的**交付文本均被截断**（O1/O3/O4 各断一处），原报告未落盘 ⇒ 缺口已逐条列入 §C，**未做任何补齐或推断**；§O3.4 / §O4.2 中标注为【复跑】的内容来自本次重跑的原始探针。
- **三方冲突 4 条**（§A-1 `_lane_offset` 净效应；§A-2 桥向是否部署时定死；§A-3 用户前提 vs O1 更正；§A-4 同构的层级），另 1 条为合成侧复跑与结构能力的层级差异（§A-5）。**全部并列，未调和。**
- 对**下一步口径选择**的影响：§C-7 表明 O1/O3/O4 的读数**不在同一时间口径上**；任何跨节或跨引擎的数字比较，须先固定 tick 口径与统计口径（【R17】式同口径要求）。

---

## §O1 — 过桥后行为（两引擎）

> 来源：子任务 O1（completed）。本节保留其原文引文与 `file:line`。

### O1.0 报告自身的两条头条结论【原文】

1. **"河"在两侧根本不是同一个东西**。A 把河带设成**硬可行走走廊**（`A/arena.py:113-116`），B 在 `2daab60` 里**把河的阻挡整体删掉**（commit message 原文：*"I removed the river tiles from blocked_tiles."*）⇒ B 的河是**普通可走地面，只是 A\* 代价高**。这决定了"过桥"在 B 里不是一个状态，在 A 里是一个会**把人弹回岸上**的约束。
2. **"过桥后走位混乱"的主因是我方独有的 `_lane_offset` 队形车道**，不是探针 commit 里猜的"卡死自救丢路点"。实测把 `_lane_offset` 归零后，跨河后横向散布 std **1.81 → 0.86**、**单只自身的横向抖动 std 0.39 → 0.06（÷6.5）**；而把卡死自救关掉，读数**逐位不变**。

> ⚠️ **对用户前提的一条更正**【原文】：`SkeletonArmy` 的 `jump_speed = 0.0`（实测 `Card('SkeletonArmy').jump_speed == 0.0`）⇒ `has_jump_ability` 恒假 ⇒ **骷髅军团永远不置位 `jumping_across_river`**。第 3 问的 jump 闩锁是 HogRider/MegaKnight 一类的事，与本次 15 只骷髅的现象**无因果**。下面仍按要求给出该闩锁的置位/复位条件。

**【复核】O1 头条① 的 commit 依据**（本次合成实查 B 仓）：

```
2daab60dae3c19ad80f406947eb22b00ed90a867
jiahaoxiang    Thu Sep 17 18:06:00 2026 +0800
BUGFIX: previous code settings are not compatible with the new A* pathfinding algorithm,
especially the part that handles river jumps. I removed the river tiles from blocked_tiles.
--- 改动统计 ---
 src/clasher_new/arena.py             |  8 --------
 src/clasher_new/battle.py            | 22 ++++++++++------------
 src/clasher_new/new_visualization.py |  6 ++----
 src/clasher_new/pathfinding_heap.py  |  5 ++++-
```

### O1.1 证据对照表【原文】

| 项目 | 代码路径 | file:line | 作用 |
|---|---|---|---|
| A | `battle.py` | `1191` | `current_target = self.update_current_target()` — **每 tick 重索敌**（无 modulo 门） |
| A | `battle.py` | `1245` | `elif self.in_sight_range(current_target) and self.battle_state.tick % 10 == 0:` — 路径**每 10 tick** 重算 |
| A | `battle.py` | `1243-1244` | `if not self.path:` → 仅路径空时立即重算（如目标失效被清空） |
| A | `battle.py` | `1253-1263` | **卡死自救**：每 0.5 s 查位移 `<0.1` → 丢弃最近路点 / 路点丢光则全程重算 |
| A | `battle.py` | `1266-1287` | 选路点（dot 法）+ **`_lane_offset` 横向平移路点** |
| A | `battle.py` | `833` / `3193-3195` | `_lane_offset` 初始化 / 部署时打标（`±0.8` 截断） |
| A | `battle.py` | `755-781` | 无目标时回退到最近敌塔，**且限制不跨中轴**（`best_same_side`） |
| A | `battle.py` | `641-659` / `202-218` | 射程 = **到矩形边缘距离** `edge_distance_from` |
| A | `battle.py` | `2670-2692` | `ensure_walkability`：不可走 → **只改 y 把人弹出河带**（就近岸） |
| A | `battle.py` | `2881-2884` | step 顺序：`update` → `ensure_walkability` → `resolve_collisions` |
| A | `battle.py` | `3274-3303` | 碰撞分离：e1 位移**没有 `*0.5`**（全量） |
| A | `battle.py` | `3305-3336` | `_push_troop_out_of_tower`：塔矩形推挤（A 独有） |
| A | `battle.py` | `827` | `path_blocked_counter = 0` — **仅赋值，全仓无读取（死代码）** |
| A | `arena.py` | `21-34` / `106-119` | `BLOCKED_TILES` **含河**；`is_walkable` 河带只放行 `2.0<=x<5.0` 与 `13.0<=x<16.0`（各 3 格） |
| A | `pathfinding_heap.py` | `74` / `80` / `83` / `106` | goal 集合 = **边缘距离** `< range+0.375`；goal = `min(edge_dist + dist_to_start)` |
| A | `pathfinding_heap.py` | `132-142` | 格子代价：`'W'` → 800(地面)/7(空中)；`'.'`（桥面）→ **5** |
| B | `battle.py` | `288-297` | 目标缓存；`elif self.battle_state.tick % 2 == 0:` — **每 2 tick** 重索敌 |
| B | `battle.py` | `313-316` | `elif self.in_sight_range(current_target) and self.battle_state.tick % 3 == 0:` — 路径**每 3 tick** |
| B | `battle.py` | `284-287` / `304-309` | **置位 `jumping_across_river` 的代码被整段注释掉** |
| B | `battle.py` | `331-342` | 路点；`if self.jumping_across_river:` 只清不置；`elif waypoint_in_river: pass`（**空操作**） |
| B | `battle.py` | `208-219` | 回退到**任意**最近敌塔（**无同侧限制** → 可跨中轴吸塔） |
| B | `battle.py` | `117-130` | 射程 = `distance_to(center) <= range + target.collision_radius + bonus` |
| B | `battle.py` | `561-576` | `ensure_walkability`（**河内永不触发**，因 `is_walkable` 恒真） |
| B | `battle.py` | `654-657` | step 顺序：同 A |
| B | `battle.py` | `741-757` | 碰撞分离：e1 位移**带 `*0.5`** |
| B | `battle.py` | `247` | `path_blocked_counter = 0` — **死代码** |
| B | `arena.py` | `21-30` / `46-55` | `BLOCKED_TILES` **无河**；`is_walkable` **无河分支** |
| B | `pathfinding_heap.py` | `55` / `62` / `64` / `67` | goal 集合 = **中心距** `< collision_radius+range+0.375`；goal = `min(中心距 + dist_to_start)` |
| B | `pathfinding_heap.py` | `93-102` | 格子代价：`'W'` → 50(地面)/7(空中或 `jump_speed`)；`'.'`（桥面）→ **8** |
| B | `pathfinding.py` | 全文（130 行） | **未被 import 的死文件**（`battle.py:3` 只 import `pathfinding_heap`） |

**【复核】上表 A 侧 27 条、B 侧 22 条全部命中**（见 §0.2）。补充核到的原行：

```
A/battle.py:3302-3303
3302:                    e1.position.x += -direction_vector.real * (1-movement_ratio)*overlap
3303:                    e1.position.y += -direction_vector.imag * (1-movement_ratio)*overlap
B/battle.py:756-757
756:                    e1.position.x += -direction_vector.real * (1-movement_ratio)*overlap*0.5
757:                    e1.position.y += -direction_vector.imag * (1-movement_ratio)*overlap*0.5
```

### O1.2 §1 goal 是什么、哪一行算出来的【原文】

**goal 不是"继续走向敌方塔中心"，而是"塔的射程环上、离自己最近的一格"**，但两边的"环"用**不同几何**定义。

**A（我方）— `A/pathfinding_heap.py:74`、`:80`、`:83`、`:106`**

```python
74:         edge_radius = self.entity.data.range
80:                 edge_dist = self.target.edge_distance_from(pos)
83:                 if edge_dist < edge_radius + 0.375 and self.battle.pathfind_ground_walkable(...):
...
106:        self.goal = min(self.goals, key=lambda c: self.target.edge_distance_from(cell_to_position(c)) + cell_to_position(c).distance_to(self.start_position))
```

`edge_distance_from` 对塔是**到矩形最近点**（`A/battle.py:202-218`）：

```python
215:            dx = max(abs(px - cx) - hw, 0.0)
216:            dy = max(abs(py - cy) - hh, 0.0)
217:            return math.hypot(dx, dy)
```

**B（上游）— `B/pathfinding_heap.py:55`、`:62`、`:64`、`:67`**

```python
55:         radius = self.target.data.collision_radius + self.entity.data.range
62:                 distance = cell_to_position((x, y)).distance_to(self.target_position)
64:                 if distance < radius+0.375 and self.battle.pathfind_ground_walkable(...):
...
67:        self.goal = min(self.goals, key=lambda c: cell_to_position(c).distance_to(self.target_position)+cell_to_position(c).distance_to(self.start_position))
```

**实测（同一投放点 `3.5,13.5`，同一只骷髅 `id=8`）【原文】**：

| | goal 世界坐标 | 候选 goal 格数 | 行为 |
|---|---|---|---|
| A | `(4.75, 23.25)` → 后段跳到 `(5.25, 23.25)` | **32** | goal **随自身位置横移** |
| B | `(3.75, 23.75)` 全程不动 | **12** | goal **锚定** |

⇒ 回答"走到塔的攻击距离外？"：**不是"塔的攻击距离外"，是"塔边缘外 0.75~0.875 格"**（两边都如此）。差别在 **A 的环更宽（32 格候选）且用边缘距离 ⇒ 环上的点边缘距近乎相等，`min` 的决胜项退化为 `dist_to_start`，于是 goal 跟着人走、横向滑动**；B 的环只有 12 格、用中心距 ⇒ 更锚定。这是"过桥后走位混乱"的**第一层**结构差。

还要注意回退目标规则不同（也是 goal 的一部分）：

- A `battle.py:768-777`【原文】：
  ```
  768:                 # 索敌范围不跨中轴：永远不把对侧公主塔作为回退目标。王塔(位于中轴)
  ...
  774:                 if (possible_princess_tower.position.x - _cx) * (self.position.x - _cx) >= 0:
  ```
  （这里 `_cx = self.battle_state.arena.width / 2`，`battle.py:773`）
- B `battle.py:208-219`【原文】：无此限制，`for i in range(1, 7): ... if distance < min_distance: self.target_id = i` ⇒ 破边塔后可能横穿去对侧车道。

**【复核】A 侧 `battle.py:758-759, 773-774, 778` 与 B 侧 `208-219` 逐行命中**；B 侧另核到：**敌我过滤存在**（`if possible_princess_tower.player == self.player: continue`），且 `self.target_id = 1` 在 `210` 有一处硬预置。

### O1.3 §2 重索敌 / 重算路径的频率【原文】

| | 重索敌 | 重算路径 | 额外重算触发 |
|---|---|---|---|
| **A** | **每 tick**（`battle.py:1191`，无门控；内部 `get_nearest_target` 全实体扫描） | **每 10 tick**（`battle.py:1245`：`self.battle_state.tick % 10 == 0`） | `not self.path`（`1243`）；目标失效清空（`735`/`740`）；卡死自救（`1261`） |
| **B** | **每 2 tick**（`battle.py:294`：`if self.battle_state.tick % 2 == 0`），缓存于 `target`（`288-297`） | **每 3 tick**（`battle.py:315`：`self.battle_state.tick % 3 == 0`） | `not self.path`（`313`）；目标失效清空（`188`/`193`） |

时钟不同（`A/environment.py:102` = `battle.step(1/60)`；`B/environment.py:52` = `self.fps = 20` ⇒ `1/20`）【复核：A `environment.py:102` 命中；B `environment.py:52 self.fps = 20`、`108 self.battle.step(1/self.fps)` 命中】，换算成墙钟：

- A：索敌 **1/60 s ≈ 16.7 ms**；路径 **10/60 = 0.167 s**
- B：索敌 **2/20 = 0.1 s**；路径 **3/20 = 0.15 s**

⇒ **路径重算频率两边几乎一样（0.167 vs 0.15 s）**；**真正的差别是索敌：A 每帧、B 每 0.1 s 且结果缓存**。A 每帧重扫实体表，且目标一旦脱视距就 `self.path = []`（`740`）⇒ A 的路径被清空/重建的次数更多 —— 但实测总重算次数接近（181 vs 186 / 420 帧×15 只）。

**`path_blocked_counter` 在两侧都是死代码**【原文 + 复核】：A 只有 `battle.py:827` 的赋值，B 只有 `battle.py:247` 的赋值，全仓无读取。`stuck` 只有 A 有（`battle.py:828-830`、`1253-1263`），B 全仓 `grep -n "stuck\|rescue\|lane_offset"` **零命中**。

### O1.4 §3a `jumping_across_river` 闩锁（**只对 `jump_speed>0` 的兵种；骷髅不适用**）【原文】

**置位（唯一一处）— `A/battle.py:1233-1239`**

```python
1233:        if (not self.in_attack_range(current_target)) or self.jumping_across_river:
1234:            has_jump_ability = self.data.jump_speed and self.on_both_sides_of_river(current_target) and self.near_river() and self.in_sight_range(current_target)
1235:            if not self.jumping_across_river and has_jump_ability:
1236:                self.start_jumping_position = Position(self.position.x, self.position.y)
1237:                self.jumping_across_river = True
1238:                self.data.is_air_unit = True
1239:                self.speed = self.data.jump_speed
```

**复位（唯一一处）— `A/battle.py:1187-1190`**

```python
1187:        if self.jumping_across_river and self.on_both_sides_of_river(self.start_jumping_position):
1188:            self.jumping_across_river = ...   ← 【缺口】原文在此被截断
```

**【复核】`A/battle.py:1187-1190` 全文（补齐缺口处）：**

```python
1187:        if self.jumping_across_river and self.on_both_sides_of_river(self.start_jumping_position):
1188:            self.jumping_across_river = False
1189:            self.data.is_air_unit = Card(self.name).is_air_unit
1190:            self.speed = self.data.speed
```

> 说明：`1188-1190` 三行来自 **【复核】**（本次合成对该 `file:line` 的只读核对），**不是** O1 原文——O1 的交付文本恰在此处中断。

### O1.5 【缺口】O1 文本缺失部分

O1 交付文本在 §3a 的 `1188` 行中断。**以下内容不可得**，本文不补齐：

- §3a 后半：复位条件的行为后果、实测读数；
- §3b 及其后（若原报告有）；
- O1 自带的"未验证/未能确定"清单；
- O1 头条② 的**消融实验仪器**：`/tmp` 下无对应脚本，`scripts/skarmy_probe.py` 也无消融开关（`--out/--max-steps/--decision-frames/--seed/--only`）。⇒ 该消融**不可从留证复现**，详见 §C-2。

---

## §O3 — 己方半场不主动靠桥（两引擎）

> 来源：子任务 O3（completed）。本节保留其原文引文与 `file:line`。
> 原文声明：**所有行号取自当前工作树；探针脚本都放在 `/tmp`，会话内未写入任何仓库源文件。**

### O3.0 证据对照表【原文】

| 项目 | 代码路径 | file:line | 作用 |
|---|---|---|---|
| A 我方 | `src/clasher_new/arena.py` | `arena.py:106-119`（关键 113-116） | `is_walkable`：**河道是硬障碍**，只有两条桥带 `2.0<=x<5.0` / `13.0<=x<16.0` 可走 |
| A | `src/clasher_new/arena.py` | `arena.py:17-20` | 定义 `LEFT_BRIDGE=Position(3.5,16.0)` / `RIGHT_BRIDGE=Position(14.5,16.0)`、`RIVER_Y1=15.0/RIVER_Y2=16.0` |
| A | `src/clasher_new/battle.py` | `battle.py:3243-3246` | `pathfind_ground_walkable`：A* 的格可走性 = `is_walkable` 且 `building_cache > mover_radius` |
| A | `src/clasher_new/battle.py` | `battle.py:2670-2690` | `ensure_walkability`：出生点修正 + 河岸 **y 夹取**（15±0.5r / 17±0.5r） |
| A | `src/clasher_new/pathfinding_heap.py` | `pathfinding_heap.py:65-106` | `EntityPathfinder.calculate`：生成 goal 候选集、择优出唯一 `goal`、再跑 A* |
| A | `src/clasher_new/pathfinding_heap.py` | `pathfinding_heap.py:132-142` | 格费用表：`'W'=800`（地面）／桥面 `'.'=5`／地面 `5` |
| A | `src/clasher_new/battle.py` | `battle.py:1233-1246` | `Troop.update` 移动分支：跳河触发条件 + 路径首次计算/每 10 tick 重算 |
| A | `src/clasher_new/battle.py` | `battle.py:1265-1287` | 每 tick 选路点 + `_lane_offset` 法线平移 + `move_towards` |
| A | `src/clasher_new/battle.py` | `battle.py:811-819` | `on_both_sides_of_river` / `near_river` |
| A | `src/clasher_new/battle.py` | `battle.py:833`、`3193-3195` | `_lane_offset` 的来源（部署环偏移，±0.8 截断，红方取反） |
| B 上游 | `src/clasher_new/arena.py` | `arena.py:46-55` | `is_walkable`：**完全不区分河道**（河格也是 True） |
| B | `src/clasher_new/battle.py` | `battle.py:724-727` | `pathfind_ground_walkable`（同名，同逻辑，但河格判 True） |
| B | `src/clasher_new/battle.py` | `battle.py:561-578` | `ensure_walkability` |
| B | `src/clasher_new/pathfinding_heap.py` | `pathfinding_heap.py:52-67` | goal 候选集 + 择优（到目标中心距离 + 到起点距离） |
| B | `src/clasher_new/pathfinding_heap.py` | `pathfinding_heap.py:93-102` | 格费用表：`'W'=50`（地面）／桥面 `'.'=8`／地面 `5` |
| B | `src/clasher_new/battle.py` | `battle.py:284-287`、`304-309` | 跳河逻辑**整段被注释**（`jumping_across_river` 永不为 True） |
| B | `src/clasher_new/battle.py` | `battle.py:313-342` | 路径计算 + 路点选取 + `waypoint_in_river` |
| B | `src/clasher_new/battle.py` | `battle.py:517-530` | `get_spawn_position`：只有环上坐标，**没有任何车道/偏移打标** |
| B | `src/clasher_new/battle.py` | `battle.py:231-239` | `on_both_sides_of_river` / `near_river`（与 A 逐字相同） |
| B | `src/clasher_new/battle.py` | `battle.py:650-653` | `building_positions` / `building_cache` 刷新 |

**【复核】A 侧 `arena.py:17-20 / 21-34 / 106-119(`113-116`)`、`battle.py:3243-3246 / 2670-2690 / 1233-1246 / 1265-1287 / 811-819 / 833 / 3193-3195`、`pathfinding_heap.py:65-106 / 132-142` 全部命中**；B 侧对应 20 条全部命中。

### O3.1 §1 每 tick 的"目标点"怎么算出来的【原文】

分三层，三层都在下面给出函数名 + 行号。

**第一层：目标实体**（"打谁"）——`Entity.update_current_target()`：A `battle.py:727`，B `battle.py:180`；兜底选最近敌方公主塔的是 A `battle.py`（`if self.target_id is None:` 分支，输出 `current_target = self.battle_state.entities[self.target_id]`）。

**第二层：goal（路径终点，A* 的目标格）**——`EntityPathfinder.calculate()`：
- A `pathfinding_heap.py:65` `def calculate(self):`，候选集按「到目标**边缘**距离 < range」筛：
  `pathfinding_heap.py:83` `if edge_dist < edge_radius + 0.375 and self.battle.pathfind_ground_walkable(pos, self.entity.data.collision_radius):`
- B 同函数：`pathfinding_heap.py:64` `if distance < radius+0.375 and self.battle.pathfind_ground_walkable(cell_to_position((x, y)), ...)`（**到目标中心**距离，非边缘）。
- 择优成唯一 goal：A `pathfinding_heap.py:106`、B `pathfinding_heap.py:67`（引文见第 2 条）。

**第三层：每 tick 的路点（waypoint）**——`Troop.update`：
- A `battle.py:1266` `min_point = min(self.path, key=lambda pos: pos.distance_to(self.position))`，`battle.py:1271-1273` 过点判定 `if dot >= 0: index += 1`，`battle.py:1274-1275` 走到尽头则 `self.move_towards(current_target.position, dt, True)`，否则 `battle.py:1287` `self.move_towards(_wp, dt, True)`。
- B `battle.py:319-342`：同一套 `min_point`/`dot` 逻辑，`battle.py:331-332` `waypoint = self.path[index]` / `waypoint_in_river = 17 >= waypoint.y >= 15`，`battle.py:342` `self.move_towards(waypoint, dt, True)`。

**路径何时算/重算**：A `battle.py:1243-1246`（`if not self.path:` 首算；`elif self.in_sight_range(current_target) and self.battle_state.tick % 10 == 0:` 每 10 tick 重算）；B `battle.py:313-316`（重算周期是 `tick % 3 == 0`）。

### O3.2 §2 goal 的 x 分量依赖什么【原文】

**依赖目标、射程、可走性，以及实体自己的出发点** —— 最后一项让 goal 不会主动收拢到某条"全局中心线"。

- A `pathfinding_heap.py:106`：
  `self.goal = min(self.goals, key=lambda c: self.target.edge_distance_from(cell_to_position(c)) + cell_to_position(c).distance_to(self.start_position))`
- B `pathfinding_heap.py:67`：
  `self.goal = min(self.goals, key=lambda c: cell_to_position(c).distance_to(self.target_position)+cell_to_position(c).distance_to(self.start_position))`

即两家的择优项都含 `distance_to(self.start_position)`，而 `start_position` 在 `EntityPathfinder.__init__` 里取自实体当前位置（A `pathfinding_heap.py:38` = `Position(entity.position.x, entity.position.y)`）。代码里还有 A 的显式设计注释 `pathfinding_heap.py:102-105`：

> `# 单位从自己一侧接近目标边缘 → 车道保持（左桥来的靠左、右桥来的靠右），而不是全部吸向目标中心正对的全局最短路径。`

**它不依赖任何桥坐标**：全仓 `grep -rn "LEFT_BRIDGE\|RIGHT_BRIDGE" --include=*.py` 在 `arena.py` 之外**零命中**（A、B 皆然）⇒ 桥中心常量是死数据，没有任何移动/寻路代码引用桥中心线。

**【复核】**：`pathfinding_heap.py:38 / 102-105 / 106` 命中；B `pathfinding_heap.py:67` 命中。
**引用精度更正**：见 §0.2 **P-b** —— 对 **A** 而言"arena.py 之外零命中"**不成立**（`scripts/probe_threat_trigger.py:50-51` 有同名标识符，系探针自定义 x 区间）；但"**arena 常量无任何引用**"成立。

### O3.3 §3 有没有「车道分配」？`_lane_offset` 到底做什么【原文】

**没有指向桥的车道分配。** A 的 `_lane_offset` 是**队形横向偏移**（保留部署环上的相对位置），不是"把单位分配到某条桥车道"：

- 打标（部署时，一次性）：A `battle.py:3193-3195`
  ```
  for pos in positions:
      _dx = max(-0.8, min(0.8, pos.x - position.x))
      pos._lane_offset = (1.0 if player_id == 1 else -1.0) * _dx
  ```
  注释 `battle.py:3191-3193`：「记录每个出兵位置相对部署中心的横向偏移（蓝方 +y 推进向的法线 = x 轴；红方镜像取反）…幅度截断 ±0.8 格防窄桥上车道越界。」
- 取用：A `battle.py:833` `self._lane_offset = getattr(position, '_lane_offset', 0.0)`。
- 生效时机与方向：**只在 AI 沿路点行进的分支**里，每 tick 对当前路点生效；方向是"本单位→路点"方向的**法线**，纵向推进时就是 x 平移；幅度 = `_lane_offset`（≤0.8 格）；走到最后一段（`index == len(self.path)`，`battle.py:1274-1275`）不加偏移：
  ```
  1280: _wp = self.path[index]
  1281: _off = self._lane_offset
  1286: _wp = Position(_wp.x - _dyw / _d * _off, _wp.y + _dxw / _d * _off)
  1287: self.move_towards(_wp, dt, True)
  ```
  空军/跳河分支（`battle.py:1240-1241`）直接 `move_towards(current_target.position, ...)`，**不经** `_lane_offset`。
- 实测（A）：`SkeletonArmy` 15 个单位的 `_lane_offset` = −0.550 … +0.538（由出兵环坐标推出）；`Archer` 2 个单位 = ±0.55；两个弓箭手反而**分别走左右两座桥**（`9.55→10.08→14.01` / `8.45→7.93→4.05`），即偏移是"强化既有分边"，不是"收拢到中心线"。
- **上游对应物：没有。** B 的 `Troop.__init__` 无该字段（`battle.py:243-249` 只设 `jumping_across_river`/`start_jumping_position`/`spawned`），`get_spawn_position`（`battle.py:517-530`）只算环坐标、无任何打标；探针对 B 打印 `_lane_offset = NONE`。上游唯一的"车道"字眼在 **agent 层**：`strategies.py:53` `lane_x = 3 if target is None or target["x"] < 9 else 14`（`battle.py:1-7` 的 import 列表里没有 `strategies`，与引擎无关）。
- 另有一条**费用层面的车道政策**：A `pathfinding_heap.py:135-140` 把桥面 `'.'` 的费用从上游的 8 改成 5，注释写明理由：
  ```
  137: # 代价相同，单位保持部署时的横向车道（2026-09-10 用户机制：
  139: # 旧值 8 让 A* 把单位吸向更便宜的桥中央地面格，抹平车道。
  ```
  即**我方 fork 是主动削弱了上游那种"吸向桥中央"的倾向**。

**【复核】A 侧 `battle.py:3193-3195 / 833 / 1280-1287 / 1240-1241 / 243-249(对应 B)` 与 B 侧 `243-249 / 517-530` 命中。**
**【复跑】O3 §3 的三项实测数字被本次复跑逐字复现**（脚本 `/tmp/probe2.py`，A/B 各跑一次）：

```
A: deploy Archer @(9.0,9.5)         -> [(-0.5500000000000007, 9.55, 9.5)]
A: deploy SkeletonArmy @(9.0,9.5)   -> (-0.5500000000000007 ... +0.5379811804035928) 共 15 个
A: dynamic Archer(x2): t=2.10 -> [(10.08,10.58,-0.55), (7.93,10.4,0.55)]
                       t=7.45 -> [(14.01,15.19,-0.55), (4.05,15.02,0.55)]
B: deploy Archer @(9.0,9.5)         -> [('NONE', 9.55, 9.5)]
B: deploy SkeletonArmy @(9.0,9.5)   -> 15 个全部 ('NONE', ...)
```

**【复核】"上游 agent 层 `strategies.py` 与引擎无关"**（本次实查 B 仓）：

```
B: 仅 evaluate.py:11 / train.py:2 / train_autoregressive.py:14 导入 strategies；
   battle.py 不 import strategies。
A: ls src/clasher_new/strategies.py -> No such file or directory   ← A 仓没有该文件
```

### O3.4 §4「改用桥」的触发条件是什么？离河道多远触发？【原文（截断前）】

**两家都没有"到某个 y 才切换到桥"的判定。** 转折是 `EntityPathfinder.calculate()` 在**部署那一刻**就算进整条路径里的（A `battle.py:1243-1244`、B `battle.py:313-314`），由 A* 的格费用决定，非 y 阈值。

存在的 y 相关判定只有这些：
- **跳河触发**（A 独有，真正带 y 阈值）：`battle.py:1234`
  `has_jump_ability = self.data.jump_speed and self.on_both_sides_of_river(current_target) and self.near_river() and self.in_sight_range(current_target)`
  `near_river` A `battle.py:818-819`：`return abs(self.position.y-15.0)<self.data.collision_radius or abs(self.position.y-17.0)<self.data.collision_radius` ⇒ 触发窗 = 距 y=15（或 y=17）**小于自身碰撞半径**。
  实测 HogRider（`collision_radius=0.6`）：`jumping_across_river -> True at t,x,y = (3.9, 5.06, 14.55)` ⇒ **离河岸线 0.45 格**触发。
- **B 的同段逻辑被注释**：`battle.py:304-309`（`# has_jump_ability = ...` / `# self.jumping_across_river = True`）与 `battle.py:284-287` ⇒ 上游 `jumping_across_river` 恒 False。实测 B 的 HogRider：`no jump in run; last pos (3.75, 23.26)`（跳河不是被推迟，是**永不发生**）。
- **河岸 y 夹取**（A `battle.py:2689-2690`）：`if 15-push_ratio*r < y < 17+push_ratio*r and not entity.data.is_air_unit:` → 把 y 夹到 `15-0.5r` 或 `17+0.5r`（只夹 y，不改 x）。
- B 的 `waypoint_in_river = 17 >= waypoint.y >= 15`（`battle.py:332`）只用于跳河状态的收尾判断，且该状态恒 False。

横向对齐的**开始位置**（不是"触发条件"，是 A* 结果）：见下。近岸部署时整段横移会被压成**贴着河岸的一格水平滑步**（探针，A 与 B 输出逐字相同）：

```
deploy (9.0,13.5   ← 【缺口】O3 原文在此处被截断
```

**【复核】A 侧 `battle.py:1234 / 818-819 / 2689-2690 / 1243-1244` 与 B 侧 `304-309 / 284-287 / 332 / 313-314` 命中**；A `pathfinding_heap.py:134` 另核到：`tile_cost = 800 if not self.entity.data.is_air_unit else 7`（**A 的空中判定只看 `is_air_unit`**），而 B `pathfinding_heap.py:95` 是 `if self.entity.data.is_air_unit or self.entity.data.jump_speed:`（**B 多一个 `jump_speed` 项**）。

**【复跑】补全 O3 截断处的探针输出**（脚本 `/tmp/probe4.py`；A 与 B 输出**逐字相同**）：

```
deploy (9.0,13.5): own-half waypoints = [(9.25,13.75),(8.75,13.75),(8.25,13.75),(7.75,13.75),
    (7.25,13.75),(6.75,13.75),(6.25,13.75),(5.75,13.75),(5.25,14.25),(4.75,14.75)]
deploy (9.0,14.0): own-half waypoints = [(9.25,14.25),(8.75,14.25),(8.25,14.25),(7.75,14.25),
    (7.25,14.25),(6.75,14.25),(6.25,14.25),(5.75,14.25),(5.25,14.25),(4.75,14.75)]
deploy (9.0,14.5): own-half waypoints = [(9.25,14.75),(8.75,14.75),(8.25,14.75),(7.75,14.75),
    (7.25,14.75),(6.75,14.75),(6.25,14.75),(5.75,14.75),(5.25,14.75),(4.75,14.75)]
（A 与 B 两次运行 diff = 无差异）
```

**【复跑】另补 O3 未及给出的己方半场横向对齐读数**（脚本 `/tmp/probe3.py`；这是 O3 标题问题"己方半场不主动靠桥"的直接读数）。

`probe3.py`：Knight 部署于 `(dx, 9.5)`，目标 = 红左公主塔 `entities[1]`；`x@start-cell` = 路径首个己方半场路点的 x；`first y with x-change` = 首次出现 x 变化的 y；`x@last own-half` = 己方半场最后一个路点。

```
--- A ---
 dx  | x@start-cell | first y with x-change | x@last own-half (y) | dist to nearest bridge centre
  2.5 |         2.75 | None                  | (2.75, 14.75) | 0.75
  4.0 |         4.25 | None                  | (4.25, 14.75) | 0.75
  6.0 |         6.25 | 13.75                 | (4.75, 14.75) | 1.25
  9.0 |         9.25 | 10.75                 | (4.75, 14.75) | 1.25
 12.0 |        12.25 | 14.25                 | (13.25, 14.75) | 1.25
 14.5 |        14.75 | None                  | (14.75, 14.75) | 0.25
 16.5 |        16.75 | 14.25                 | (15.75, 14.75) | 1.25
--- B ---
 dx  | x@start-cell | first y with x-change | x@last own-half (y) | dist to nearest bridge centre
  2.5 |         2.75 | None                  | (2.75, 14.75) | 0.75
  4.0 |         4.25 | None                  | (4.25, 14.75) | 0.75
  6.0 |         6.25 | 10.25                 | (4.25, 14.75) | 0.75
  9.0 |         9.25 | 10.25                 | (4.25, 14.75) | 0.75
 12.0 |        12.25 |  9.75                 | (4.75, 14.75) | 1.25
 14.5 |        14.75 | None                  | (14.75, 14.75) | 0.25
 16.5 |        16.75 | 10.25                 | (15.25, 14.75) | 0.75
```

**只陈述结构差异与观测值**：`dx=2.5 / 4.0 / 14.5` 三档两仓一致（本就在桥带内，无 x 变化）；`dx=6.0 / 9.0 / 12.0 / 16.5` 四档，**A 的首次 x 变化出现在 y=10.75~14.25，B 出现在 y=9.75~10.25**；`dx=12.0` 一档，**A 的己方半场末路点 x=13.25（右桥带），B 的为 x=4.75（左桥带）**。

### O3.5 【缺口】O3 文本缺失部分

O3 交付文本在 §4 的探针输出处中断。**以下内容不可得**：§4 后半、§5 及其后（若原报告有）、O3 自带的"未验证/未能确定"清单。

---

## §O4 — 分道数量：出兵圆环几何（两引擎）

> 来源：子任务 O4（completed）。本节保留其原文引文、命令与输出。

### O4.0 §0 结论速览【原文】

| # | 问题 | 结论 |
|---|---|---|
| 1 | N / spawn_radius | **两仓完全一致**：`N=15`（来自数据 `summonNumber`）、`spawn_radius=0.55` tile、`spawn_delay=0.0`、`type=character`、`elixir=3`、`summonRadius` **键不存在** ⇒ **0.55 是 `card_utils.py` 的硬编码兜底默认 550/1000，不是卡数据** |
| 2 | 15 点坐标 | 见 §2 三张表 |
| 3 | 纯几何分道 | `behind_king` **跨中线 2 左 / 13 右**；`bridge_left` **15 左 / 0**；`bridge_right` **0 / 15 右** |
| 4 | 其它分道代码 | **两仓引擎都没有任何"把一次部署的 N 只按左右桥分配"的代码**。选桥是**逐单位索敌 + A\* 过河可行格**的涌现结果。两处**相邻**机制：A 独有 `_lane_offset` 队形偏移 + **索敌不跨中轴**钳制；B 独有 **策略层**（`strategies.py`，非引擎）选部署车道 |
| 5 | 出兵几何是否逐字相同 | **逐字相同**（正文 byte-identical，仅行尾 A=LF / B=CRLF 之差）。`card_utils.py` 中唯一功能性差异是 `nested_idx` 的健壮性写法，**不碰几何** |

### O4.1 §1 Q1 — 实际命令与原文输出【原文】

**1.1 我方 A（用户给定命令，直接可跑）**

```bash
cd /mnt/e/clash-royale-simulator-main/src/clasher_new && python3 -c "import sys;sys.stdout.reconfigure(encoding='utf-8',errors='replace');from card_utils import Card;c=Card('SkeletonArmy');print(c.spawn_number,c.spawn_radius,c.spawn_delay,c.type,c.elixir, getattr(c,'data',{}).get('summonRadius'))"
```

原文输出（stdout）：`15 0.55 0.0 character 3 None`　`[exit code: 0]`

**1.2 上游 B（用户给定命令）— 失败**

```bash
cd /mnt/e/clash-royale-simulator-main-by-jason/src/clasher_new && python3 -c "...same..."
```

原文输出（stderr）：

```
Traceback (most recent call last):
  File "<string>", line 1, in <module>
    import sys;sys.stdout.reconfigure(...);from card_utils import Card;c=Card('SkeletonArmy');print(...)
  File "/mnt/e/clash-royale-simulator-main-by-jason/src/clasher_new/card_utils.py", line 2, in <module>
    from fastcore.all import nested_idx
ModuleNotFoundError: No module named 'fastcore'
```

`[exit code: 1]`

**原因**：上游 `card_utils.py:2` 仍 `from fastcore.all import nested_idx`；我方 `card_utils.py:1-3` 已把该依赖内联消除（注释原文：「原依赖 fastcore.nested_idx，仅一处嵌套取值，已内联等价实现——消除非必要依赖」）。

**1.3 上游 B（补齐 `fastcore` 后）— 成功**

```bash
cd /mnt/e/clash-royale-simulator-main-by-jason/src/clasher_new && PYTHONPATH=/tmp/deps_upstream python3 -c "...same..."
```

原文输出：`15 0.55 0.0 character 3 None`　`[exit code: 0]` — **与我方逐字一致**。

**1.4 原始数据交叉验核（两仓 `gamedata.json` 各自实读）**

```bash
cd <repo>/src/clasher_new && python3 -c "import json,sys;sys.stdout.reconfigure(encoding='utf-8',errors='replace');d=json.load(open('gamedata.json'));o=d['items']['spells'][12];print(o['name'],o.get('summonNumber'),repr(o.get('summonRadius')),repr(o.get('summonDeployDelay')),o.get('manaCost'),o.get('tidType'))"
```

两仓原文输出（**逐字相同**）：`SkeletonArmy 15 None None 3 TID_CARD_TYPE_CHARACTER`
以及 `'summonRadius' in c.data` 实测：`data has summonRadius key: False` / `data has summonNumber key: True 15`

⇒ **关键事实**：`N=15` 是**真实卡数据**；而 `spawn_radius=0.55` 走的是 `data.get('summonRadius', 550)/1000` 的**兜底分支**——`gamedata.json` 里 `SkeletonArmy` 没有 `summonRadius` 字段。也就是说，**这份圆环几何的半径并非由卡数据驱动，而是 `card_utils.py` 里的硬编码 550/1000**。两仓此分支完全相同。

**1.5 ⚠️ 一处自我更正（取证纪律）【原文】**

我最初用一个"忠实 fastcore 语义"的内存 shim（裸链式下标 `c = c[i]`）跑上游，得到 `KeyError: 'projectileData'`（上游 `card_utils.py:68`）。**这个中间结论是错的，是我 shim 的假象，不是上游行为**，特此声明并撤回。

我随后把真库取下来读源码定性（`pip download fastcore` 到 `/tmp`，读 whl 内 `fastcore/basics.py`）：fastcore 2.2.28 的 `nested_idx` 是**容错的**——

```python
def _nested_idx(coll, *idxs):
    *idxs,last_idx = idxs
    for idx in idxs:
        if isinstance(idx,str) and hasattr(coll, idx): coll = getattr(coll, idx)
        else:
            if isinstance(coll,str) or not isinstance(coll, typing.Collection): return None,None
            coll = coll.get(idx, None) if hasattr(coll, 'get') else coll[idx] if idx<len(coll) else None
    return coll,last_idx

def nested_idx(coll, *idxs):
    "Index into nested collections, dicts, etc, with `idxs`"
    if not coll or not idxs: return coll
    coll,idx = _nested_idx(coll, *idxs)
    if not coll or not idxs: return coll
    return _access(coll, idx)
```

即 `.get(idx, None)` 链，缺键返回 `None` 而非抛异常 ⇒ 上游 `Card('SkeletonArmy')` 正常构造。§1.3 的结果是拿**真库**跑出来的（不是我手写的 shim）。fastcore 侧依据：[fastcore 文档 basics](https://fastcore.fast.ai/basics.html)（`nested_idx` 定义），实际语义以 `fastcore 2.2.28` wheel 内 `fastcore/basics.py` 为准。

**【复核】O4 的 A 侧机制依据**：A `card_utils.py:1-3` = `import json` / `import os` / 内联注释，**全文件无 `fastcore`**；A `card_utils.py:34` 已是 `with open(_resolve_data('gamedata.json'), encoding='utf-8') as f:`，`_resolve_data`（def 在 `:6`）先试 `__file__` 同目录、再退回裸相对名。B `card_utils.py:2` 仍是 `from fastcore.all import nested_idx`。

### O4.2 §2 Q2 — 三个投放点的全部 15 个出兵世界坐标【原文口径 + 复跑数值】

**口径**（与用户给定一致，并已在代码中复核）【原文】：
- 世界坐标 = 本地 + 0.5 —— 实证 `environment.py:94`：`self.battle.deploy_card(0, card_name, Position(x+0.5, y+0.5))`（红方镜像在 `environment.py:72`）。`deploy_card` 收到的 `position` 即世界坐标，再传给 `get_spawn_position`（A:`battle.py:3186`）。
- `get_spawn_position` 取 `offset_angle=True`（default，deploy 路径未传 False），但 `angle_offset = {2:0, 3:π/2, 4:π/4, 6:0}` 中**无键 15** ⇒ `.get(15, 0) = 0` ⇒ **角度无偏移**；`player == 1` 才加 π，本次为 player 0 ⇒ 不加。
- 故 `angle_i = 2πi/15 = 24°·i`，`r = 0.55`，点 = 投放点 + (r·cos, r·sin)。
- 数值由**真函数**产出（`from battle import get_spawn_position` + `Card('SkeletonArmy')`，`offset_angle` 走默认 `True`），非手算。

**【复核 + 复跑】函数体逐字相同**：`/tmp/gspA.txt` 与 `/tmp/gspB.txt` 为两仓 `get_spawn_position` 全文，**12 行逐字节相同**：

```python
def get_spawn_position(card_info, position, player, offset_angle=True):
    spawn_number, spawn_delay, r = card_info.spawn_number, card_info.spawn_delay, card_info.spawn_radius
    if spawn_number == 1: return [Position(position.x, position.y)]
    positions = []
    angle_offset = {2: 0, 3: math.pi/2, 4: math.pi/4, 6: 0}
    for i in range(spawn_number):
        angle = 2*math.pi*i/spawn_number
        if offset_angle: angle += angle_offset.get(spawn_number, 0)
        if player == 1: angle += math.pi
        dx, dy = r*math.cos(angle), r*math.sin(angle)
        positions.append(Position(position.x+dx, position.y+dy))
    return positions
```

**【复跑】以真函数重跑两仓，输出 `diff` = `IDENTICAL`（排除 REPO 头行后逐字节相同）**。下面三张表因此**同时代表 A 与 B**（脚本 `/tmp/gsp_full.py`）。

**2.1 `behind_king` — 世界中心 (9.5, 0.5)**

| i | 角度 | x | y | 侧 |
|---|---|---|---|---|
| 0 | 0° | 10.0500000000 | 0.5000000000 | 右 |
| 1 | 24° | 10.0024500017 | 0.7237051537 | 右 |
| 2 | 48° | 9.8680218335 | 0.9087296540 | 右 |
| 3 | 72° | 9.6699593469 | 1.0230810840 | 右 |
| 4 | 96° | 9.4425093452 | 1.0469870425 | 右 |
| 5 | 120° | 9.2250000000 | 0.9763139721 | 右 |
| 6 | 144° | 9.0550406531 | 0.8232818888 | 右 |
| **7** | **168°** | **8.9620188196** | 0.6143514299 | **左** |
| **8** | **192°** | **8.9620188196** | 0.3856485701 | **左** |
| 9 | 216° | 9.0550406531 | 0.1767181112 | 右 |
| 10 | 240° | 9.2250000000 | 0.0236860279 | 右 |
| 11 | 264° | 9.4425093452 | **−0.0469870425** | 右 |
| 12 | 288° | 9.6699593469 | −0.0230810840 | 右 |
| 13 | 312° | 9.8680218335 | 0.0912703460 | 右 |
| 14 | 336° | 10.0024500017 | 0.2762948463 | 右 |

`x_min = 8.9620188196`，`x_max = 10.0500000000`；`x<9: 2`，`x>9: 13`，`x==9: 0`。

**2.2 `bridge_left` — 世界中心 (3.5, 13.5)**

| i | 角度 | x | y | 侧 |
|---|---|---|---|---|
| 0 | 0° | 4.0500000000 | 13.5000000000 | 左 |
| 1 | 24° | 4.0024500017 | 13.7237051537 | 左 |
| 2 | 48° | 3.8680218335 | 13.9087296540 | 左 |
| 3 | 72° | 3.6699593469 | 14.0230810840 | 左 |
| 4 | 96° | 3.4425093452 | 14.0469870425 | 左 |
| 5 | 120° | 3.2250000000 | 13.9763139721 | 左 |
| 6 | 144° | 3.0550406531 | 13.8232818888 | 左 |
| 7 | 168° | 2.9620188196 | 13.6143514299 | 左 |
| 8 | 192° | 2.9620188196 | 13.3856485701 | 左 |
| 9 | 216° | 3.0550406531 | 13.1767181112 | 左 |
| 10 | 240° | 3.2250000000 | 13.0236860279 | 左 |
| 11 | 264° | 3.4425093452 | 12.9530129575 | 左 |
| 12 | 288° | 3.6699593469 | 12.9769189160 | 左 |
| 13 | 312° | 3.8680218335 | 13.0912703460 | 左 |
| 14 | 336° | 4.0024500017 | 13.2762948463 | 左 |

`x ∈ [2.9620188196, 4.0500000000]`；`x<9: 15`，`x>9: 0`。

**2.3 `bridge_right` — 世界中心 (14.5, 13.5)**　← **O4 原文在第 6 行（`i=6`）截断；下表 `i=7..14` 为【复跑】补齐**

| i | 角度 | x | y | 侧 |
|---|---|---|---|---|
| 0 | 0° | 15.0500000000 | 13.5000000000 | 右 |
| 1 | 24° | 15.0024500017 | 13.7237051537 | 右 |
| 2 | 48° | 14.8680218335 | 13.9087296540 | 右 |
| 3 | 72° | 14.6699593469 | 14.0230810840 | 右 |
| 4 | 96° | 14.4425093452 | 14.0469870425 | 右 |
| 5 | 120° | 14.2250000000 | 13.9763139721 | 右 |
| 6 | 144° | 14.0550406531 | 13.8232818888 | 右 |
| 7 | 168° | 13.9620188196 | 13.6143514299 | 右 |
| 8 | 192° | 13.9620188196 | 13.3856485701 | 右 |
| 9 | 216° | 14.0550406531 | 13.1767181112 | 右 |
| 10 | 240° | 14.2250000000 | 13.0236860279 | 右 |
| 11 | 264° | 14.4425093452 | 12.9530129575 | 右 |
| 12 | 288° | 14.6699593469 | 12.9769189160 | 右 |
| 13 | 312° | 14.8680218335 | 13.0912703460 | 右 |
| 14 | 336° | 15.0024500017 | 13.2762948463 | 右 |

`x_min = 13.9620188196`，`x_max = 15.0500000000`；`x<9: 0`，`x>9: 15`。

**补注（【复跑】新读出、O4 原文未给）**：`behind_king` 表有 **1 个点的 y 为负**（`i=11`，`y = −0.0469870425`），即出兵环**越出己方场地 y=0 边界**。O4 原文只给出坐标与 x 侧计数，**未对 y 越界作任何说明**；此处仅记录该观测值。

### O4.3 Q3 / Q4 / Q5（结论见 §O4.0 速览；支撑见上）

【原文】Q3 纯几何分道：`behind_king` 跨中线 **2 左 / 13 右**；`bridge_left` **15 左 / 0**；`bridge_right` **0 / 15 右**。
【原文】Q4：**两仓引擎都没有任何"把一次部署的 N 只按左右桥分配"的代码**。选桥是逐单位索敌 + A* 过河可行格的涌现结果。
【原文】Q5：出兵几何**逐字相同**（正文 byte-identical，仅行尾 A=LF / B=CRLF 之差）。

**【复核】Q4 的两处"相邻机制"**：A 侧 `_lane_offset`（`battle.py:833/3193-3195/1280-1287`）与索敌同侧钳制（`battle.py:758-759/773-774/778`）**命中**；B 侧 `strategies.py:53` 存在但**仅被 `evaluate.py:11` / `train.py:2` / `train_autoregressive.py:14` 导入，`battle.py` 不导入**；A 仓无 `strategies.py`。

### O4.4 【缺口】O4 文本缺失部分

O4 交付文本在 §2.3 表第 6 行中断。**以下内容不可得**：`bridge_right` 表 `i=7..14`（已由【复跑】补齐）、**§3–§7 全部**——含 O4 原文 §7「副作用与恢复」（原文 §0 曾以"副作用与恢复见 §7"作指针）与 O4 自带的"未验证/未能确定"清单。⇒ 因此**无法确认 O4 探针是否在仓库里留下了任何副作用**，见 §C-9。

---

## §A 三方结论的冲突并列（**并列不调和**）

> 规则（用户要求 ②）：三方结论互相独立；冲突处**并列**，**不做调和**，只指出冲突点。
> 以下每条冲突**不判断谁对**。

### A-1 `_lane_offset` 的净效应：横向散布的**增加量** vs 车道的**保持量**

| 方 | 说法 |
|---|---|
| **O1** | `_lane_offset` 是"过桥后走位混乱"的**主因**；归零后跨河横向散布 std **1.81→0.86**、单只横向抖动 std **0.39→0.06** |
| **O3** | `_lane_offset` 是**队形横向偏移**，"偏移是**强化既有分边**，不是收拢到中心线"；并引 A 代码注释：桥面代价 8→5 是为了"单位**保持**部署时的横向车道" |
| **O4** | 两仓都**没有**按桥分配车道的代码；选桥是"逐单位索敌 + A\* 过河可行格的**涌现结果**" |

**冲突点**：同一设施的可观测后果被描述为方向相反的两种（**增大横向散布** vs **保持横向车道**）。
**并列的依据差异（不是调和）**：O1 的度量是**散布 std**；O3 的度量是**车道归属 / 路点序列**。两者**不是同一个量**，因此两个说法可以同时为真 —— 但**三份文本都没有做这个限定**，故按原文并列，不合并。

### A-2 桥的转向是否在**部署那一刻**定死

| 方 | 说法 |
|---|---|
| **O3** | 「两家都没有"到某个 y 才切换到桥"的判定。转折是 `EntityPathfinder.calculate()` 在**部署那一刻**就算进整条路径里的」 |
| **O1** | A 的 goal **随自身位置横移**（`(4.75,23.25)` → `(5.25,23.25)`）；且 A 每 10 tick、B 每 3 tick **重算**路径 |

**冲突点**：O3 的"部署时定死"与 O1 的"周期性重算 + goal 随位置移动"**不可同时成立**（除非限定为"桥的转向在部署时定、goal 的横向分量在重算时滑动"——**两份文本都没有这样限定**）。
**并列不调和。**

### A-3 用户前提 vs O1 的更正（jump 闩锁 / 卡死自救 与骷髅走位的因果）

| 方 | 说法 |
|---|---|
| **用户前提**（由探针 commit 猜测表述） | 过桥后走位混乱 ≈ 卡死自救丢路点；jump 闩锁与本次 15 只骷髅有关 |
| **O1** | `SkeletonArmy` `jump_speed = 0.0` ⇒ `has_jump_ability` 恒假 ⇒ 骷髅永不置位 `jumping_across_river`，与 15 只骷髅的现象**无因果**；把卡死自救关掉后读数**逐位不变** |
| **O3** | 把 jump 闩锁作为 **A 独有活代码**描述，给出触发窗「离岸 < 碰撞半径」，实测 HogRider 在**离岸 0.45 格**触发（`(3.9,5.06,14.55)`） |
| **O4** | 给出骷髅出兵几何（与 jump 无关）。**O4 未涉及 `jump_speed`**：其 §1.1 输出 `15 0.55 0.0 character 3 None` 的第三字段是 **`spawn_delay=0.0`**，不是 `jump_speed`。`jump_speed == 0.0` 由 **O1** 实测 |

**冲突点**：**因果归属**。前提把闩锁/自救与骷髅走位相连；O1 **断开**该连接。O3/O4 的文本本身**不含**该因果主张（O3 是机制描述、O4 是几何描述）⇒ **冲突发生在"用户前提"与 O1 之间，不在三份取证之间**。
**并列不调和**：保留前提、保留 O1 的更正。

### A-4 「两方是否同构」的**层级**冲突

| 方 | 说法 | 层级 |
|---|---|---|
| **O4** | 出兵几何**逐字相同**；15 点坐标逐位相同 | **静态几何层** |
| **O3** | A 独有 `_lane_offset`；B 探针打印 `NONE` | **运行期机制层** |
| **O1** | 运行期横向行为不同（消融读数 + goal 横移） | **运行期行为层** |

**冲突点**：同一命题（"两引擎在分道/横向行为上是否相同"）在静态层为**同构**、在运行期层为**不同构**，且两者都被各自证据支持。三份文本**都没有声明自己所在层级**。
**并列不调和**；本文只标层级，不合并。

### A-5 【复跑，非三方原始结论】河的"可走性"：结构能力 vs 探针是否触发

| 来源 | 说法 |
|---|---|
| **O1**（结构能力） | B 的河是**普通可走地面**（代价高）；A 的河是**硬阻挡** |
| **O3**（结构能力） | A 的河是**硬障碍**，只有两条桥带可走 |
| **【复跑】** `/tmp/probe2.py`（12 个投放点：`dy∈{9.5,13.5}` × `dx∈{2.5,6,9,12,14.5,16.5}`） | **两仓的"非桥面河带路点"均为空列表**（B 亦为 `[]`，12/12） |

**冲突点**：O1 的"B 可涉水"是**能力层**结论；复跑显示在**探针覆盖的 12 个局面**里该能力**未产生任何非桥面河带路点**。⇒ **能力存在 ≠ 已观测到被使用**。
**并列不调和**（注：本行证据来自**合成侧复跑**，不是三方文本，故单列）。

---

## §B 代码块清单（用户要求 ③）

> 三份取证的交叉产物。每行都给出**双方行号**与**可观测后果**；不含优劣判断。

### B-1 只有一方有（A 有，B 无对位）

| # | 代码块 | A file:line | B 对位 | 可观测后果（结构差异） |
|---|---|---|---|---|
| 1 | `_lane_offset` 字段与取用 | `battle.py:833` | **无**：B 仓 `grep _lane_offset` = 0 命中；`Troop.__init__`（B `243-249`）无此字段 | A 的路点会被法线平移；B 不会 |
| 2 | `_lane_offset` 部署期打标（±0.8 截断、红方取反） | `battle.py:3193-3195` | **无** | 仅 A 有该字段的来源 |
| 3 | 路点法线平移本体 | `battle.py:1280-1287` | **无**对应分支 | 仅 A 在 `_off != 0` 时改路点 |
| 4 | 卡死自救 `_stuck_check_pos/_time` + 丢最近路点 | `battle.py:828-830`、`1253-1263` | **无**：B 仓 `grep "stuck\|rescue"` = 0 命中 | A 在位移 <0.1 格时丢路点 |
| 5 | 跳河闩锁**活代码**（置位 / 复位） | 置位 `1233-1239`；复位 `1187-1190` | B 同段被**整段注释**（`284-287`、`304-309`） | A 有可置 `True` 的路径；B 无 |
| 6 | 索敌**同侧钳制**（不跨中轴） | `758-759`、`773-774`、`778`（`best_same_side or best_any`） | B `208-219`：无侧向限制 | 破边塔后 A 不把对侧公主塔作为回退目标；B 可以 |
| 7 | 塔矩形推挤 `_push_troop_out_of_tower` | `3305-3336` | **无** | 仅 A 有塔-部队 AABB 推出 |
| 8 | 命中几何 = **到矩形边缘** | `202-218`（矩形最近点）、`641-659`（`in_attack_range`） | B `117-130`：`distance_to(center) <= range + collision_radius + bonus` | 同一 `range` 在塔前停靠点不同 |
| 9 | goal = **边缘距离** + **三级兜底** | `pathfinding_heap.py:74/80/83/101/106` | B `55/62/64/67`：中心距、**无兜底** | 同一局面 goal 可不同；B 的 `goals` 为空时 `min(())` 抛 `ValueError` |
| 10 | 河面**硬阻挡**（`BLOCKED_TILES` 含河 + 桥面分支） | `arena.py:21-34`、`106-119`（关键 `113-116`） | B `21-30` 河面条目已删、`46-55` 无河分支 | A 只能走桥；B 可走（代价高） |
| 11 | 空中判定**不含** `jump_speed` 的 `'W'` 计价项 | `pathfinding_heap.py:134` `800 if not is_air_unit else 7` | B `93-102`：`is_air_unit or jump_speed` → 7 | B 的 `'W'` 计价多一个 `jump_speed` 分支 |

### B-2 只有一方有（B 有，A 无对位）

| # | 代码块 | B file:line | A 对位 | 备注 |
|---|---|---|---|---|
| 1 | 跳河**置位/复位整段注释文本** | `284-287`（复位/落地块）、`304-309`（置位/起飞块） | A 为**活代码**（见 B-1 #5） | B 的注释块本身是 B 独有文本 |
| 2 | `pathfinding.py`（非堆版 A\*，130 行） | `pathfinding.py` 全文 | **A 无此文件**：`ls` → `No such file or directory` | 见 B-3 #2（B 侧死文件） |
| 3 | agent 层 `strategies.py`（`lane_x = 3 if ... else 14`） | `strategies.py:53` | **A 无此文件** | 与引擎无关：B 仅 `evaluate.py:11`/`train.py:2`/`train_autoregressive.py:14` 导入 |
| 4 | 河面**可走**（地面涉水价 50） | `arena.py:21-30`（无河条目）、`46-55`（无河分支）、`pathfinding_heap.py:98`（`tile_cost = 50`） | A 为硬阻挡 + `'W'=800` | 能力层差异（触发情况见 §A-5） |

### B-3 死代码 / 死数据

| # | 代码块 | 位置 | 判定依据 |
|---|---|---|---|
| 1 | **B `pathfinding.py` 全文（130 行，非堆版 A\*）** | B `src/clasher_new/pathfinding.py` | **死文件**：B `battle.py:3` 只 `from pathfinding_heap import ...`；全仓 grep `pathfinding\b` 排除 `_heap` = **0 命中**。**A 侧该文件已不存在**（已删） |
| 2 | `path_blocked_counter` | A `battle.py:827` ／ B `battle.py:247` | **两方同构的死代码**：两仓各自全仓**仅此 1 处赋值**，无读取、无自增 |
| 3 | B 跳河闩锁（置位块 + 复位块 + 读取分支） | B `battle.py:304-309`、`284-287`（注释）；读取 `334-341`、`332` | **B 侧死分支**：`jumping_across_river` **无置 `True` 处** ⇒ 恒 `False`；`334-341` 的 jump 分支只清不置，`332` 的 `waypoint_in_river` 为空转 |
| 4 | `LEFT_BRIDGE` / `RIGHT_BRIDGE` 常量 | A `arena.py:17-18` ／ B `arena.py:17-18` | **两方同构的死数据**：两仓各自的 `arena.py` 之外**无任何引用**。（A 侧唯一同名标识符在 `scripts/probe_threat_trigger.py:50-51`，是探针自定义的 x 区间元组，非该常量——见 §0.2 P-b） |
| 5 | B `strategies.py` 的 `lane_x` | B `strategies.py:53` | **对引擎是死件**：`battle.py` 不导入；仅 agent/训练脚本使用。A 仓无该文件 |

### B-4 两方同构（相同）

| # | 代码块 | A file:line | B file:line | 同构程度 |
|---|---|---|---|---|
| 1 | `get_spawn_position` 全文 | A `battle.py`（函数） | B `battle.py:517-530` | **逐字节相同**（`/tmp/gspA.txt` vs `/tmp/gspB.txt`，12 行）；复跑 15 点坐标两仓 `diff` = IDENTICAL |
| 2 | `on_both_sides_of_river` / `near_river` | `811-819` | `231-239` | **逐字相同**（含 `15.0/17.0` 阈值与 `<collision_radius` 窗） |
| 3 | step 顺序 `update → ensure_walkability → resolve_collisions` | `2881-2884` | `654-657` | **相同顺序** |
| 4 | 路点选取骨架（`min_point` + `start/close` 向量点积 + `dot >= 0 → index += 1`） | `1266-1271`、`1274-1275` | `319-331` | **同构**（A 额外叠 `_lane_offset`，见 B-1 #3） |
| 5 | 网格/邻居/启发式/A\* 主循环/路径重建（`pathfinding_heap.py` 共同血统） | `pathfinding_heap.py` | `pathfinding_heap.py` | **【复核】** 两仓同名同结构函数、同一 import 语句（A `battle.py:4` / B `battle.py:3`）。**注：三份取证未逐字节 diff 主循环**，故此项标"结构同构（未逐字节核）"，见 §C-11 |
| 6 | `path_blocked_counter` 死代码 | `827` | `247` | **同构的死代码**（见 B-3 #2） |

### B-5 同一处、两种实现（**逐项配对差异**；不属于"有/无"）

| 项 | A | B | 可观测后果 |
|---|---|---|---|
| 命中几何 | 矩形边缘 `edge_distance_from`（`202-218`/`641-659`） | 圆心距（`117-130`） | 塔前停靠点 / 开火时机不同 |
| goal 择优基准 | `edge_distance_from(c) + dist_to_start`（`pathfinding_heap.py:106`） | `distance_to(target_position) + dist_to_start`（`67`） | goal 可不同 |
| goal 扫描半径 | `_target_footprint_radius()` 基准 + `+2`（`pathfinding_heap.py:65-106`） | `ceil(radius*2)+1`（`52-67`） | 候选集大小不同（O1 实测 32 vs 12） |
| goal 空集处理 | 三级兜底（`101`） | 无 → `ValueError` | B 在"目标周围全堵死"时抛异常（未实测触发） |
| 河面 | 墙 + 桥（`arena.py:21-34`/`106-119`） | 贵路（`arena.py:21-30`/`46-55`） | 通行能力不同 |
| 桥面代价 `'.'` | **5**（`pathfinding_heap.py:140`，注释 `136-139`） | **8**（`93-102`） | 桥面/地面相对价格不同 |
| `'W'` 地面代价 | **800**（`134`） | **50**（`98`） | 水面的相对价格不同 |
| `'W'` 空中条件 | `not is_air_unit`（`134`） | `is_air_unit or jump_speed`（`95`） | 多一个 `jump_speed` 判定项 |
| 目标重索敌节流 | **每 tick**（`1191`，无门） | `tick % 2`（`294`），缓存于 `288-297` | 索敌频率不同 |
| 路径重算节流 | `tick % 10` @60 Hz（`1245`） | `tick % 3` @20 Hz（`315`） | 墙钟 ≈0.167 s vs ≈0.15 s |
| 碰撞分离 e1 位移 | `(1-movement_ratio)*overlap`（`3302-3303`） | `(1-movement_ratio)*overlap*0.5`（`756-757`） | e1 被推开量不同（B 为 A 的一半） |
| 引擎时钟 | `step(1/60)`（`environment.py:102`） | `fps = 20`；`step(1/self.fps)`（`environment.py:52`/`108`） | tick 粒度不同 |

---

## §C 未验证 / 未能确定（【R10】：不确定就写不确定）

> 按用户要求 ④，本节列**本次合成结束时仍不确定**的全部条目。**不升级为结论。**

**C-1【缺口】三份取证的交付文本全部被截断，且原始报告未落盘。**
截断点：O1 §3a（`battle.py:1188` 引文中）、O3 §4（探针输出 `deploy (9.0,13.5`）、O4 §2.3（`i=6`）。
缺失内容：O1 §3a 后半与 §4 之后全部；O3 §4 后半及其后全部；O4 的 `bridge_right` 表 `i=7..14` 与 §3–§7 全部。
⇒ **三份取证自带的"未验证"清单全部不可得**；本节是**合成侧重建**的清单，**不等于**原报告的清单。

**C-2 O1 头条② 的消融读数量化不可复现。**
数值：横向散布 std **1.81→0.86**、单只横向抖动 std **0.39→0.06**、"关掉卡死自救读数逐位不变"。
**未能确定**：产生这些数的脚本不存在（`/tmp` 无对应文件；`scripts/skarmy_probe.py` 无消融开关，其参数仅 `--out/--max-steps/--decision-frames/--seed/--only`）；**std 的口径未定义**（时间窗、样本集合、"跨河后"的判定、单只抖动如何聚合）。
⇒ 本次合成**未复跑**；这些数字**不可与任何其他数字对拍**（口径未知）。

**C-3 O1 的 goal 实测不可复现。**
数值：A goal `(4.75,23.25)→(5.25,23.25)`、候选 **32** 格；B goal `(3.75,23.75)`、候选 **12** 格（投放点 `3.5,13.5`，`id=8`）。
**未能确定**：该探针脚本未留证；"后段跳到"的触发 tick 未给出。⇒ 未复跑。

**C-4 O1 的"总重算次数 181 vs 186 / 420 帧×15 只"口径不明。**
**未能确定**：计数脚本、采样窗口、"一次重算"的判定（`calculate()` 调用次数？路径被替换次数？）。⇒ 不可复算。

**C-5 O3 的 grep 表述不精确（结论不变）。**
`jumping_across_river` 的"只剩 33/248/334/337/562"遗漏注释行 `284/285/305/307`（B 侧实测命中）。"恒 False"结论**成立**。

**C-6 O3 的桥常量"零命中"表述对 A 不成立（结论不变）。**
A 侧 `scripts/probe_threat_trigger.py:50-51` 有同名标识符；"arena 常量无引用"**成立**。

**C-7 三份取证的时间/tick 口径不统一，跨节数字不可对拍。**
O1/O3 以 `step(1/60)` 喂两仓（为隔离 tick 粒度）；B 生产是 20 Hz（`environment.py:52/108`）⇒ **20 Hz 的影响未隔离**。O4 走卡数据/几何函数，不经 tick。
⇒ **跨节、跨引擎的数字不可直接比较**。

**C-8 【复跑】B 的河面可走能力在探针覆盖的 12 个局面上未产生非桥面河带路点。**
**未能确定**：该能力在什么局面下会被触发（桥被堵？目标在正对岸？）；**未做**针对性构造。

**C-9 O4 的"§7 副作用与恢复"不可得。**
O4 §0 曾以"副作用与恢复见 §7"作指针，但 §7 在截断区。
**未能确定**：O4 的三个探针（`Card` 构造、`get_spawn_position` 直调、`gamedata.json` 读取）是否在仓库或运行目录留下了任何写入。
**本次合成侧的核对**：本次复跑全部走 `python3 -B` + `PYTHONDONTWRITEBYTECODE=1`，只读源文件、只写 `/tmp`；**未改任何仓库文件**。但**这不能替代**对 O4 原始运行的确认。

**C-10 `/tmp/griddiff.txt` 为 0 字节。**
该留证文件存在但**无任何内容** ⇒ 无法确定它原本要验证什么，也**无法确定**它是否曾预期产出。
（同一批留证中 `gspA.txt`/`gspB.txt`/`probe2-4.py`/`probe_path.py` 均有内容。）

**C-11 A\* 主循环 / 邻居 / 启发式未做逐字节 diff。**
B-4 #5 只标"结构同构"。【复核】只覆盖行号处的文本，**未覆盖**两仓 `pathfinding_heap.py` 的逐字节等价性。
**未能确定**：除已列的配对差异（goal / 代价表）外，A\* 主循环是否还有未列出的差异。

**C-12 B 的 `goals` 空集 `ValueError` 未实测触发。**
O1/O3 均给出"无兜底 ⇒ `min(())` 抛 `ValueError`"的**代码阅读**结论。**未能确定**：是否存在可达局面真的触发它。⇒ 未构造、未实测。

**C-13 上游 `2daab60` 的动机与本文的结构差异之间的关系未确定。**
commit message 自述 "previous code settings are not compatible with the new A* pathfinding algorithm, especially the part that handles river jumps"（【复核】已实查该 commit：作者 `jiahaoxiang`，2026-09-17 18:06，`arena.py` −8 行）。
**未能确定**：该 commit 同时改了 `battle.py`（22 行）与 `pathfinding_heap.py`（5 行），**其内部改动与本次三份取证列出的差异如何对应，未逐项归属**。

**C-14 A 侧 `jumping_across_river` 的多处站点未逐一分析。**
【复核】A `battle.py` 实测命中：`83`、`410`、`834`、`1187/1188`、`1233/1235/1237`、`1749`、`1809`、`1983`、`2048`、`2145`、`2204`、`2677`（含多个 per-card 类的 `= False` 重置）。
**未能确定**：三份取证只分析了 `1187-1190` 与 `1233-1239`；其余站点（尤其 `410` 的 `if not self.data.is_air_unit and not self.jumping_across_river:`）对走位的影响**未验证**。

**C-15 `behind_king` 出兵环越出场地边界。**
`i=11` 的 `y = −0.0469870425 < 0`。该**数值本身已在 O4 原文的表中给出**（非新数据）；**未确定**的是：O4 原文**未对该 y 越界作任何说明**，且该点是否被 `ensure_walkability`（A `battle.py:2670-2692` 的 `if y < push_ratio*r: y = push_ratio*r`）修正、修正发生在哪一 tick —— 本次只读出兵坐标，**未追踪后续位置**。
