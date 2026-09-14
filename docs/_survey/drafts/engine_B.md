# 游戏引擎文档 · 第 B 部分：寻路、索敌、卡牌机制与部署合法性

> **取证口径**：本部分只依据源码与本次会话的实测。每条结论后跟行内来源 `（文件.py:行号）`。
> 无法从源码确认的一律写「待确认」。数值原样抄自源码，不改写。
>
> 三个此前会话产出的中间素材（`docs/_survey/parts/G002|G004|G015|G019|G035|G036|G038.md`）仅用于交叉检索关键字，
> 正文结论全部回到源码复核；本部分的实测数字来自会话内临时脚本 `.tmp/mask_vs_engine.py`（`git check-ignore` 确认
> `.tmp/` 被忽略，不入库）。

---

## B.1 寻路总览

### B.1.1 谁需要寻路

| 主体 | 是否走 A* | 依据 |
|---|---|---|
| 地面 `Troop` | **是**，`Troop.update` 里唯一 A* 调用方 | `battle.py:1176-1197` |
| 空中单位（`data.is_air_unit=True`） | **否**，直接 `move_towards(current_target.position)` | `battle.py:1176-1177` |
| 跳河状态（`jumping_across_river`） | **否**，仍走 `move_towards`；跳河时临时置 `data.is_air_unit=True` | `battle.py:1123-1126, 1169-1177` |
| `Building` / `Projectile` | **否**（`Building.update` 与 `Projectile.update` 无 A*） | `battle.py:1273-1330, 1548-1594` |
| 机制类直线位移（Assassin 突进 / MegaKnight 冲刺跳 / HeroValkyrie 冲刺 / HeroGiant 投掷 / Skeletrooper 伞降） | **否**，直接改 `position`，并在位移后清空 `path` | `card_mechanics.py:774-807, 552-628, 1007-1021, 1181-1187, 1437-1438` |

### B.1.2 输入与输出

**输入**（`EntityPathfinder.__init__`，`pathfinding_heap.py:37-45`）：

| 参数 | 用途 |
|---|---|
| `entity` | 取 `entity.position`（起点）、`entity.data.collision_radius`、`entity.data.range`、`entity.data.is_air_unit` |
| `target` | 取 `target.position`、`target.edge_distance_from(pos)`（目标边缘距离）、`target._tower_rect`（塔矩形） |
| `battle_state` | 只用来调 `battle_state.pathfind_ground_walkable(pos, radius)` |

**输出**：`calculate()` 返回 `list[Position]`，元素是**半格中心**世界坐标，起点在 `path[0]`（`pathfinding_heap.py:154-160`）。
路径长度不设上限；无路可走时返回的路径由 A* 实际展开的 `parent` 链决定（见 B.3.5 边界条件）。

### B.1.3 调用点（全部）

全仓只有 `battle.py` 导入并调用：

| 位置 | 内容 |
|---|---|
| `battle.py:4` | `from pathfinding_heap import EntityPathfinder, position_to_cell, cell_to_position` |
| `battle.py:1180` | `if not self.path:` → 重算全程路径 |
| `battle.py:1182` | `elif self.in_sight_range(current_target) and self.battle_state.tick % 10 == 0:` → 目标在视距内时每 10 tick 刷新 |
| `battle.py:1197` | 卡死自救：每 0.5s 检查，位移 `< 0.1` 则删掉当前路点，路点删空后重算 |

`battle.py:1180/1182/1197` 是**仅有的三处** A* 构造点（全仓 grep `EntityPathfinder(` 结果一致）。

### B.1.4 路径消费与失效

消费在 `battle.py:1201-1223`：取 `min(self.path, key=distance_to(self.position))` 作为最近路点，用
`start_vector`（起点→自身）与 `close_vector`（自身→路点）的点积 `dot >= 0` 决定是否切到下一个路点；
最后一格直接 `move_towards(current_target.position)`；中间路点再按 `_lane_offset` 沿法线平移（队形车道）。

`path = []`（强制重算）出现在：目标失效 / 目标脱离视距（`battle.py:680, 685`）、嘲讽锁定（`battle.py:918`）、
眩晕重索敌（`battle.py:123-125`）、机制类位移后（`card_mechanics.py:1020, 1068, 1079, 1124, 1187, 1271, 1522, 1758`）、
非卡牌实体构造（`battle.py:1682, 1736, 1908, 1969, 2062, 2118`）。

---

## B.2 网格与通行性

### B.2.1 两套坐标系

| 层 | 尺寸 | 定义 |
|---|---|---|
| 世界/竞技场 | 18 × 32（`width, height = 18, 32`） | `arena.py:9`；`is_valid_position` 判 `0 <= x < 18 and 0 <= y < 32`（`arena.py:100-101`） |
| 寻路网格 | 36 × 64 **半格** | `position_to_cell = (floor(2x), floor(2y))`（`pathfinding_heap.py:13-15`）；反向 `cell_to_position = ((x+0.5)/2, (y+0.5)/2)`，带 `cell_cache`（`pathfinding_heap.py:17-21`） |

**1 寻路格 = 0.5 世界单位**；`TileGrid.tile_size = 100.0` 声明存在但寻路路径与 `is_walkable` 均未使用它（`arena.py:10`）。

### B.2.2 地形代价表 `tilemap_lane_grid.txt`

- 文件 64 行 × 36 列（**行数 = 64 = 寻路网格高度，列数 = 36 = 宽度**，与半格网格一一对应）。
- 字符分布（会话内统计）：`.` 1500、`1` 346、`2` 346、`W` 112。
- 读取方式：`contents = [list(each) for each in f.read().splitlines()]`（`pathfinding_heap.py:6-8`；`pathfinding.py:6-8` 同）。
- **行索引翻转**：查表用 `contents[63 - ny][nx]`（`pathfinding_heap.py:132`，`pathfinding.py:106` 同），即文件第 0 行对应 `ny=63`。

代价映射（`pathfinding_heap.py:132-147`）：

| 字符 | 地面 `tile_cost` | 空中 `tile_cost` |
|---|---|---|
| `W` | 800 | 7 |
| `.` | 5（heap 版）／8（`pathfinding.py`，`pathfinding.py:109-110`） | 同左 |
| 其它（含 `1`/`2`） | 5 | 5 |

对角线 `geo_cost = 14`，直线 `geo_cost = 10`，`step_cost = tile_cost * geo_cost`（`pathfinding_heap.py:143-147`）。

> **`W` 的 800 代价实际几乎不可达**：A* 的邻居判定先过 `arena.is_walkable`（`battle.py:3053-3054`），
> 而 `is_walkable` 把整条河带（`y ∈ [15.0, 16.0]`）判为不可走，只放行桥面（见 B.2.3）。
> 会话内穷举 36 × 64 格实测：**可走的 `W` 格只有 16 个** ——
> `nx ∈ {4, 9, 26, 31}` × `ny ∈ {30, 31, 32, 33}`（即桥两侧最外一条半格）。
> 因此地面单位过桥时，桥面 6 条半格（`nx 4..9` / `26..31`）里最外两条要付 800 代价、
> 中间四条付 5，A* 会向中间收敛 —— 源码注释声称的「桥面代价一致以保持车道」
> （`pathfinding_heap.py:136-140`）与实测地形表并不完全一致。

### B.2.3 `arena.TileGrid` 的通行性

| 判定 | 规则 | 行号 |
|---|---|---|
| 边界 | `0 <= x < 18 and 0 <= y < 32` | `arena.py:100-101` |
| `BLOCKED_TILES` | 河岸两端 `(0,15)(0,16)(1,15)(1,16)`、`(16,15)(16,16)(17,15)(17,16)`；河中央 `x∈[5,13) × y∈[15,17)`；上下底边各 6 格围栏 `y=0/y=31` 且 `x∈[0,6)∪[12,18)` | `arena.py:21-34` |
| 河流/桥 | `RIVER_Y1 = 15.0`、`RIVER_Y2 = 16.0`；`y ∈ [15.0, 16.0]` 时仅 `2.0 <= x < 5.0`（左桥）或 `13.0 <= x < 16.0`（右桥）可走 | `arena.py:113-116`；桥常量 `LEFT_BRIDGE=(3.5,16.0)`、`RIGHT_BRIDGE=(14.5,16.0)`（`arena.py:17-18`） |
| 缓存 | `walkable_cache[(int(x), int(y))]`（**按整格缓存**） | `arena.py:5, 108-110` |

> `is_walkable` 的缓存键是 `(int(x), int(y))`，因此同一 1×1 世界格内不同子坐标会命中同一结果；
> 这与寻路的 0.5 单位半格网格不同粒度，**是否为有意设计：待确认**（见 B.9 #4）。

塔是**矩形**（2026-09-09 定稿）：公主塔 3×3（half 1.5）、国王塔 4×4（half 2.0），列表见 `arena.py:36-46`，
点到矩形距离 `dist_to_rect`（`arena.py:48-53`），存活塔最近距离 `tower_rect_dist`（`arena.py:65-84`，靠每帧刷新的
`_tower_alive` 缓存 `arena.py:55-63`）。

### B.2.4 两套 walkability API（容易混淆，务必区分）

| 函数 | 判定 | 行号 | 谁用 |
|---|---|---|---|
| `pathfind_ground_walkable(pos, mover_radius)` | `arena.is_walkable(pos)` **且** `building_cache[x][y] > mover_radius` | `battle.py:3053-3056` | **只被 A* 用**（`pathfinding_heap.py:83, 93, 128`） |
| `ground_walkable(pos, mover_radius)` | `arena.is_walkable(pos)` **且** 非建筑/塔占位（活体列表 + 塔矩形） | `battle.py:3058-3060` | 非寻路代码：投射物击退与滚动命中（`battle.py:1578, 1929`）、机制类落点（`card_mechanics.py:1018, 1183, 1519`）、`ensure_walkability`（`battle.py:2590`） |

`building_cache` 是「到最近障碍的距离场」：先把 6 座塔矩形距离写进去，再对 `building_positions`
（`id > 6` 的建筑）逐格取 `min(中心距 − 建筑半径)`（`battle.py:3031-3052`；`building_positions`
在 `step` 每帧重建，`battle.py:2695`）。缓存只在 `cache_fresh == False` 时重算
（构造置 False `battle.py:2570`、新建筑/新延时体置 False `battle.py:2624, 3155`、`step` 内重算 `battle.py:2698-2700`）。

`ground_walkable` 的塔占位走 `_tower_footprint_blocks`（`battle.py:3062-3070`）：`mover_radius <= 0`
时点在矩形内即阻挡，否则到矩形距离 `< mover_radius` 即阻挡；非塔建筑走圆形列表
`(x-position.x)² + (y-position.y)² < (r + mover_radius)²`（`battle.py:3078-3081`）。

---

## B.3 `pathfinding_heap.py`（引擎实际使用的那一套）

**结论先行**：引擎实际使用的是 `pathfinding_heap.py`，证据是唯一的导入点
`battle.py:4` 以及三处调用点 `battle.py:1180, 1182, 1197`（全仓 grep 无其它导入方）。

### B.3.1 清单

| 名字 | 行号 | 说明 |
|---|---|---|
| 模块级 `contents` | 6-8 | 读入 `tilemap_lane_grid.txt`（64 行 × 36 列） |
| `cell_cache` / `neighbor_cache` | 10-11 | 世界坐标缓存 / 邻居表缓存 |
| `position_to_cell(position)` | 13-15 | 世界 → 半格 |
| `cell_to_position(cell)` | 17-21 | 半格 → 世界（带缓存） |
| `get_neighboring_points(x, y)` | 23-33 | 8 邻域；越界 `nx<0 or ny<0 or nx>=36 or ny>=64` 丢弃；结果缓存 |
| `class EntityPathfinder` | 36 | 见下 |
| `.__init__(entity, target, battle_state)` | 37-45 | 状态初始化 |
| `.heuristic(cell)` | 47-50 | `10 * max(|Δx|, |Δy|)`（Chebyshev × 10） |
| `._target_footprint_radius()` | 52-63 | 塔取 `_tower_rect` 半轴较大者，否则 `data.collision_radius` |
| `.calculate()` | 65-160 | 目标格生成 + A* |
| `__main__` 演示 | 162-172 | 直接构造 `BattleState` 冒烟 |

### B.3.2 `g` / `h` 定义

- `g[start_cell] = 0`（`pathfinding_heap.py:112`）。
- 转移代价：`step_cost = tile_cost * geo_cost`（`pathfinding_heap.py:147`），具体数值见 B.2.2 表：
  - 地面：`W=800→8000`，`.`/`1`/`2`/其它`=5→50`（直线）/`70`（对角）
  - 空中：`W=7→70`（直线）/`98`（对角），其余 `5→50/70`
- 启发式：`h(cell) = 10 * max(|x-gx|, |y-gy|)`，`(gx,gy) = self.goal`（`pathfinding_heap.py:47-50`）。
  最小真实步代价是 50，`h` 每格只计 10 ⇒ `h` 是**下界**（admissible & consistent），A* 在
  「目标格=单点」语义下给出最小代价路径。
- `f = g + h`（`pathfinding_heap.py:113, 152`）。

### B.3.3 目标格（goals）的生成与择优

1. `edge_radius = entity.data.range`（`pathfinding_heap.py:74`）。
2. 扫描窗：`scan_radius = ceil((目标占用半径 + edge_radius) * 2) + 2`，中心为目标所在半格
   （`pathfinding_heap.py:76-78`）。
3. 判定：`target.edge_distance_from(pos) < edge_radius + 0.375` **且** `pathfind_ground_walkable(...)`
   （`pathfinding_heap.py:80-84`）。0.375 是让短射程部队够得着塔的裕量（源码注释 `pathfinding_heap.py:81-82`）。
4. 空集兜底：退化为「扫描区内到目标边缘最近的可走格」；仍无则直接以目标格为 goal
   （`pathfinding_heap.py:85-101`）。
5. 择优：`self.goal = min(goals, key=edge_distance_from(pos) + distance_to(start_position))`
   （`pathfinding_heap.py:106`）—— 边缘距离基本相等，起主导的是「离起点最近」，即车道保持。

### B.3.4 堆与松弛

- `open_heap = [(f[start], start)]`，`heapq.heappop` 取最小；用 `closed_set` + 「`current_f > f[current]` 跳过」
  实现**惰性删除**（`pathfinding_heap.py:114-124`）。
- 每个邻居：`closed_set` 跳过；不可走跳过；按 B.2.2 计算 `step_cost`；`tentative_g < g[neighbor]`
  或邻居未访问则更新 `g/parent/f` 并 `heappush`（`pathfinding_heap.py:125-153`）。
- 终止条件：弹出 `current == self.goal` 即 `break`（`pathfinding_heap.py:122-123`）。
- 目标是**单点**（`self.goal`），不是 goals 集合 —— 这是与 `pathfinding.py` 的主要差异之一。

### B.3.5 返回路径与边界条件

- `path = [current]; while path[-1] != start_cell: path.append(parent[path[-1]]); path.reverse()`
  （`pathfinding_heap.py:154-157`）。
- 若堆先耗尽而从未到达 goal（例如目标格被建筑完全封死且兜底也未命中），`current` 是**最后一个弹出的节点**，
  返回的是「到该点的路径」而不是到目标的路径；调用方不支持「无路」信号
  （`battle.py:1202-1211` 只按最近路点走）。
- 若 `start_cell == goal`，`path = [start]`，只返回 1 个点。
- 若目标格在扫描窗内但起点被围死，A* 会正常耗尽 `open_heap`（不抛异常）。
- `_target_footprint_radius` 会惰性调用 `target._bind_tower_rect()`（`pathfinding_heap.py:57-62`；
  `_bind_tower_rect` 定义在 `battle.py:170-183`）。

---

## B.4 `pathfinding.py`（未使用的旧实现，与 B.3 的关系）

### B.4.1 死代码判定

**全仓未见调用点**。依据：

```
grep -rn --include=*.py -e "import pathfinding" -e "from pathfinding" -e "pathfinding\." . \
  --exclude-dir=.venv --exclude-dir=.git --exclude-dir=__pycache__
→ 唯一命中：./src/clasher_new/battle.py:4: from pathfinding_heap import EntityPathfinder, ...
```

`pathfinding.py` 内对 `EntityPathfinder` 的唯一构造在它自己的 `__main__` 演示（`pathfinding.py:141`）。
即：本模块在仓库内**没有任何 import 方**（`battle.py:4` 导入的是 `pathfinding_heap`）。

### B.4.2 与 B.3 的逐项差异

| 维度 | `pathfinding_heap.py`（在用） | `pathfinding.py`（未在用） |
|---|---|---|
| 开放集 | `heapq` 二叉堆 + 惰性删除（`:114-124`） | `set` + 每次 `min(open_set, key=f)` 线性扫描（`pathfinding.py:88, 93-97`） |
| 邻居 | `neighbor_cache` 缓存（`:11, 23-33`） | 每次 `yield` 生成器（`pathfinding.py:25-31`） |
| 启发式 | 单目标 `10*max(Δ)`（`:47-50`） | **goals 集合上的 min**：`min(10*max(...) for gx,gy in self.goals)`（`pathfinding.py:44-49`） |
| `goals` 生成半径 | `entity.range`（边缘口径，`edge_distance_from`）（`:74-83`） | `target.data.collision_radius + entity.range`（**中心距**口径，`pathfinding.py:54, 78`） |
| 塔矩形 | 统一走 `target.edge_distance_from`（`:80`） | 单独分支：`target.persistent and target.id <= 6` 时用 `TileGrid.dist_to_rect`（`pathfinding.py:60-77`） |
| 目标格择优 | 边缘距离 + 到起点距离最小（`:106`） | 只留「到目标中心距离最近」的 **1 个**（`pathfinding.py:83`） |
| `.` 面代价 | 5（`:140`） | 8（`pathfinding.py:109-110`） |
| 空 goals 兜底 | 有（`:85-101`） | **无** —— 若 goals 为空，`min(self.goals, ...)` 会抛 `ValueError`（`pathfinding.py:83`） |
| walkability | `pathfind_ground_walkable`（建筑距离场）（`:83, 93, 128`） | `ground_walkable`（活体建筑列表）（`pathfinding.py:75, 80, 102`） |
| 空中单位 | 仅影响 `W` 代价（`:134`） | 同（`pathfinding.py:108`） |

### B.4.3 `pathfinding.py` 的一处静态错误

`pathfinding.py` 的 `__init__` 只赋值了 `self.battle`（`pathfinding.py:41`），**从未赋值 `self.battle_state`**，
但塔分支引用了它：

```python
if getattr(self.target, 'persistent', False) and getattr(self.target, 'id', 99) <= 6 \
        and self.battle_state is not None:      # pathfinding.py:61-62
```

⇒ 一旦进入该分支（目标是 `persistent` 且 `id <= 6`，如公主塔/国王塔），会因 `AttributeError` 崩溃。
由于该模块无调用方（B.4.1），**实际不可达**；此处只作为「若把它改回调用方会踩的坑」记录。

---

## B.5 索敌与目标选择

### B.5.1 `sight_range` / `range` 字段来源

| 字段 | 来源 | 行号 |
|---|---|---|
| `Card.sight_range` | `summonCharacterData.sightRange / 1000` | `card_utils.py:248` |
| `Card.range` | `summonCharacterData.range / 1000` | `card_utils.py:247` |
| `Card.min_range` | 数值表行 `minimum_range / 1000`（Mortar/GoblinCannon/BarbarianLauncher=3.5） | `card_utils.py:314-315` |
| `Card.is_air_unit` | `characters_data[].flying_height != 0` 名单 或 角色名命中 | `card_utils.py:9, 244` |
| `Card.attack_air` / `attack_ground` | `tidTarget` 含 `AIR` / 含 `GROUND`（`target_only_buildings` 视为可打地） | `card_utils.py:245-246` |
| `Card.target_only_buildings` | `tidTarget == "TID_TARGETS_BUILDINGS"` | `card_utils.py:243` |

**统一射程口径 = 边缘距离** `Entity.edge_distance_from(pos)`（`battle.py:152-168`）：
塔（`_tower_rect` 已绑定）→ 到矩形最近点距离（矩形内 0）；其余实体 → `中心距 − collision_radius`。
`_bind_tower_rect` 在首次索敌时惰性绑定，用 `0` 哨兵区分「未绑定」与「非塔」（`battle.py:158-161, 170-183`）。

### B.5.2 在射程 / 在视距

`in_attack_range(target)`（`battle.py:586-604`）判定顺序：

1. `target is None → False`；
2. `'PrincessTower' in target.name` → `bonus = 0.5`，否则 0（`battle.py:588-591`）；
3. `dist = target.edge_distance_from(self.position)`（`battle.py:593`）；
4. 有 `_range_override`（三枪近战形态）→ 直接用 `dist <= _ro + bonus`（`battle.py:595-597`）；
5. 有 `data.min_range` 且 `dist < min_range` → `False`（`battle.py:598-600`）；
6. 有 `_snipe_range_active`（觉醒火枪手狙击弹）且 `dist <= _snipe` → `True`（`battle.py:601-603`）；
7. 否则 `dist <= data.range + bonus`。

`in_sight_range(target)`（`battle.py:605-616`）：同样的 `bonus` 与 `edge_distance_from`；
仍先看 `_snipe_range_active`，最后 `dist <= data.sight_range + bonus`。

> 塔名的子串约定：引擎里公主塔卡名是 `King_PrincessTowers`（`battle.py:2554-2559`），同时含
> `'King'` 与 `'PrincessTower'`；塔兵变体 `King_CannonTowers / King_KnifeTowers / King_ChefTowers`
> **不含** `'PrincessTower'`（会话内实测 `Card(...).name`）⇒ 塔兵不吃 `bonus = 0.5`，
> 且 `update_current_target` 的「目标脱离视距也保留」分支（`battle.py:684`）对它们也不生效。

### B.5.3 目标选择：`get_nearest_target`

流程（`battle.py:618-649`）：

1. 遍历全部实体，只要 `Troop`/`Building`、`is_alive`、`player != self.player`、`targetable`（`battle.py:623-626`）；
2. 距离用**中心距** `position.distance_to`（`battle.py:627`）——与射程判定口径不同；
3. `min_range` 内目标排除（Mortar 贴脸不打）（`battle.py:628-629`）；
4. 飞行/地面限制：`entity.data.is_air_unit and not self.data.attack_air` → 跳过；地面且
   `not self.data.attack_ground` → 跳过（`battle.py:630-631`）；
5. 必须在视距内 `in_sight_range(entity)`（`battle.py:632`）；
6. 建筑与部队分别入表；`target_only_buildings` 时部队不入表（`battle.py:633-636`）；
7. **优先级**：`target_only_buildings` → 只看建筑；否则若最近建筑**或**最近部队在攻击范围内 →
   `troops + buildings` 一起按距离排序（即**部队优先**）；都不在攻击范围内 → 只有部队，无部队才看建筑
   （`battle.py:640-645`）；
8. 按距离升序取第 1 个（`battle.py:647-649`）。

⇒ 目标选择的排序键是**距离**，**不是**血量最低、也不是建筑硬优先；「建筑优先」只体现为
`target_only_buildings` 卡的硬约束，以及「无部队可打时退化为建筑」。

### B.5.4 换目标条件：`_should_switch_target`

（`battle.py:651-670`）

1. `target_only_buildings` 且新目标不是建筑 → 不换（`battle.py:654`）；
2. `not new_target` → 换（`battle.py:655-656`）；
3. **防御建筑优先转火**：`self` 是 `Building`、新目标是 `Troop`、当前目标不是 `Troop`，
   且新目标在攻击范围内 → 换（`battle.py:657-660`）；
4. 当前目标仍在攻击范围内 → **锁定不换**（`battle.py:661-662`）；
5. 新目标是部队、当前是建筑 → 换（`battle.py:663-667`）；
6. 否则比较与新/旧目标的**中心距**，新目标更近才换（`battle.py:668-669`）。

### B.5.5 重选时机：`update_current_target`

（`battle.py:672-727`）

1. 目标 id 缺失 / 不在实体表 / 已死 → 清 `target_id` 与 `path`（`battle.py:675-680`）；
2. 目标脱离视距 → 非塔目标清空 `path` 并释放目标（塔例外，保留）（`battle.py:683-687`）；
3. 取 `get_nearest_target()`；已有目标则按 `_should_switch_target` 决定是否换，无目标则直接采用
   （`battle.py:689-696`）；
4. **兜底**：仍无目标（如后排刚落地）→ 在敌方 6 座塔里选最近的公主塔，且**不跨中轴**
   （`(塔.x − 9) * (自己.x − 9) >= 0`）；同侧无存活塔才放宽到任意塔；国王塔在中轴不受限
   （`battle.py:698-727`）。

### B.5.6 仇恨 / aggro

- **引擎侧没有通用 aggro/仇恨值机制**：`grep -rn "aggro|仇恨"` 仅命中 `rl/action_mask.py:281, 341` 与
  `rl/belief_planner.py` 的**注释/启发式文案**，`battle.py`、`card_mechanics.py` 无命中。
- 唯一的强制锁定是 **Knight Hero 嘲讽**：`_hero_taunt_override` 在 `update_current_target` 之后覆盖
  普通索敌（`battle.py:729-736`），嘲讽窗内强制锁 `_taunt_target_id`；
  `HeroKnight.use_ability` 把 6.5 格内敌军 `_taunt_until = time + tauntDuration` 并清 `path`
  （`card_mechanics.py:908-922`）。
- 眩晕可附带**重索敌**：`apply_buff(stun=..., retarget=True)` → `target_id = None; path = []`
  （`battle.py:123-125`）；调用点：ZapFreeze 的 `AreaEffect._pulse`（`battle.py:1799`）与 Lightning
  （`battle.py:2804`）。

### B.5.7 法术索敌 vs 单位索敌

| 维度 | 单位/建筑 | 法术 |
|---|---|---|
| 目标是什么 | 实体 id（`target_id`），持续跟踪 | **坐标**（落点），无 `target_id` 语义 |
| 命中判定 | `edge_distance_from ≤ range`（边缘） | 罩圈判定：`中心距 ≤ radius + 目标碰撞半径`（`battle.py:1785-1786`） |
| 视距过滤 | 有（`sight_range`） | **无**（`AreaEffect._pulse` 遍历全实体，`battle.py:1788-1831`） |
| 敌我过滤 | `player != self.player` | `only_enemies` / `only_own_troops` 行字段（`battle.py:1748-1749, 1792-1793`） |
| 建筑 | `target_only_buildings` | `ignore_buildings` 行字段（`battle.py:1750, 1794`） |
| 选目标的方式 | 最近目标（B.5.3） | 半径内全体（AreaEffect）/ 半径内 HP 最高的至多 3 个（Lightning，`battle.py:2793-2804`） |
| 弹道法术 | 弹丸 `homing=True` 追实体位置（`battle.py:1588`） | 落点溅射在 `target_position`（`battle.py:1605-1612`） |

### B.5.8 `threat_calc.py` 的威胁口径

- 语义：**双方都不再部署**的前提下，敌方现存兵力在未来 `horizon` 秒能打掉我方多少塔血
  （模块 docstring `threat_calc.py:1-21`）。
- 实现：`copy.deepcopy(battle)` 后用**引擎自身** `sim.step(dt)` 确定性推演（`threat_calc.py:64-77`），
  `dt = 1/60`（`threat_calc.py:52`），`THREAT_HORIZON_S = 20.0`（`threat_calc.py:30`）。
- **无解析公式**：索敌/攻速/位移/塔兵反击/国王塔激活/飞行物全部由引擎结算（模块 docstring `threat_calc.py:3-6`）。
- 塔 id 约定：`{0: (3,4,6), 1: (1,2,5)}`，名字映射 `{3:'left',4:'right',6:'king',1:'left',2:'right',5:'king'}`
  （`threat_calc.py:33-34`）。
- 输出：`{left, right, king, total, towers_lost, sim_time}`，各值 `round(..., 1)`
  （`threat_calc.py:79-90`）；`towers_lost` 记录推演期内 `not is_alive` 的塔名（`threat_calc.py:86-87`）。
- 早停：`game_over`、敌方威胁源清空（`_hostiles_present` 为假，`threat_calc.py:37-48, 74-75`）、
  我方三塔全破（`threat_calc.py:76-77`）；无敌方非塔存活实体时直接返回全 0（`threat_calc.py:61-62`）。
- **调用点**：生产路径（训练/评估）无调用；全仓仅 `rl/selftest.py:2732`（测试）与本模块 `__main__`
  演示（`threat_calc.py:93-106`）。模块 docstring 声明的「belief_planner/prophet 消费」在代码里**尚未接线**
  （grep `estimate_tower_threat` 无其它命中）。

---

## B.6 卡牌机制分类（`card_mechanics.py`）

### B.6.1 组织方式与装配链

- 每个机制类**必须与卡名同名**，因为 holder 是 `eval` 出来的：
  `self.entity_holder = BasicCharacter(self)`；`if self.card_name in globals() and not isinstance(self, Projectile): self.entity_holder = eval(f"{self.card_name}(self)")`
  （`battle.py:50-52`）。`card_mechanics` 通过 `from card_mechanics import *` 注入 `battle.py` 命名空间（`battle.py:5`）。
- 基类 `BasicCharacter`（`core.py:16-28`）：持 `entity` / `battle_state` / `data`，默认钩子
  `on_spawn`（空）、`on_tick`（只同步 battle_state）、`on_death`（空）、`on_attack`（`core.py:30-68`）。
- 精英卡（Hero）另有一条装配链：`HERO_CLASSES` 表（`card_mechanics.py:1562-1579`）+
  `apply_hero_overlay(entity, bs)` 换 holder 并覆写数值（`battle.py:2409-2436`）。
- 特殊实体**刻意绕过** `Entity.__init__`（否则同名机制类会在 `battle_state` 挂载前跑钩子）：
  `AreaEffect`（`battle.py:1727`）、`EvoEffectZone`（`battle.py:1951`）。

### B.6.2 触发点（钩子调度位置）

| 钩子 | 调度位置 | 说明 |
|---|---|---|
| `on_spawn()` | `battle.py:53`（`Entity.__init__` 末尾） | 部署即触发 |
| `on_tick(dt)` | `Entity.update → self.entity_holder.on_tick(dt)`（`battle.py:362`） | 每帧；`Troop.update` 在冰冻期会提前 `return`，因此 `freeze_timer > 0` 时 `on_tick` 不被调用（`battle.py:1101-1104`） |
| `on_attack(current_target)` | `Troop.update` 攻击分支（`battle.py:1226-1227`） | 冷却到 0 时由 holder 结算 |
| `on_take_damage(amount, source)` | `Entity.take_damage`（`battle.py:518-524`），返回 `True` = 短路本次伤害 | 目前唯一使用者 Ronin 格挡（`card_mechanics.py:695-711`）、HeroBerserker 血量下限（`card_mechanics.py:1334-1340`） |
| `on_damaged(amount, source)` | `Entity.take_damage` 末尾（`battle.py:582-584`） | 不短路伤害；ElectroGiant 反射（`card_mechanics.py:1584-1600`） |
| `on_death()` | `Entity.die`（`battle.py:203`） | 死亡主钩子 |
| `_evo_on_death()` | `Entity.die`（`battle.py:204-205`，仅 Troop/Building） | 觉醒亡语 |
| `_generic_death_spawn()` | `Entity.die`（`battle.py:206-207`） | 通用 `deathSpawnCharacterData` 亡语 |
| `_evo_on_attack(target)` | `BasicCharacter.on_attack` 末段（`core.py:61-63`） | 觉醒攻击后钩子 |
| `_on_attack_done(target)` | `core.py:64-66` | 攻击序列推进 / 多段命中 |
| `dash_tick(dt)` | `Troop.update` 直调（`battle.py:1153-1155`） | Assassin 突进，bypass 冰冻/常规索敌 |

### B.6.3 机制类全名单（含行号、钩子、关键数值）

**基础机制族**

| 类 | 行号 | 钩子 | 机制 / 关键数值 |
|---|---|---|---|
| `Ghost` | 8 | on_tick, on_attack | 隐身：`targetable=False`；显形后 `EVOLVED_INVIS_DELAY = 2.0`，基础取 `buffWhenNotAttackingTime`(缺省 1800ms)（`card_mechanics.py:15, 43-45`） |
| `Witch` | 66 | on_tick | 每 `spawnPauseTime`（缺省 7000ms）出 4 只 Skeletons，首波 1.0s（`card_mechanics.py:69, 78-87`） |
| `Balloon` | 89 | on_death | 死亡生成 `TimedExplosive`（`card_mechanics.py:90-93`） |
| `Golem` | 95 | on_death | 死亡在 `x±0.5` 出 2 只 Golemite（`card_mechanics.py:100-103`） |
| `LavaHound` | 105 | on_death | 死亡出 LavaPups（`card_mechanics.py:109-111`） |
| `Prince` | 113 | on_tick, on_attack | 冲锋：位移 > `charge_range` → `speed *= 2`、`attack_cooldown = 0`；冲锋伤害 `charge_damage`（`card_mechanics.py:122-150`） |
| `DarkPrince` | 155 | — | `pass`（继承 Prince） |
| `BattleRam` | 158 | on_death | 死亡出 Barbarian（`card_mechanics.py:161-163`） |
| `GiantSkeleton` | 166 | on_death | 死亡生成 `TimedExplosive`（`card_mechanics.py:169-173`） |
| `IceWizard` | 175 | on_spawn | 落地冰雾：`damage×level_scale`、半径 `spawn_data['radius']/1000`、对塔 0%（`crownTowerDamagePercent=-100`）、减速/攻速减速取 `buffData.speedMultiplier/hitSpeedMultiplier`、时长缺省 2500ms（`card_mechanics.py:186-200`） |
| `Miner` | 202 | on_tick | 钻地：`freeze_time = 到敌方国王塔距离 / (650/60)`，期间 `targetable=False, invincible=True`（`card_mechanics.py:205-216`） |
| `Rage` | 218 | on_tick | 光环：`speed_multiplier = buffData.hitSpeedMultiplier/100`（注意取的是 **hitSpeedMultiplier** 字段）、半径/寿命/攻速均来自 `death_area_effect`；施法即时 `deal_area_damage`（`card_mechanics.py:218-259`） |
| `RageBarbarian` | 261 | on_death | 死亡生成 `Rage` 实体（`card_mechanics.py:262-264`） |
| `Fisherman` | 269 | on_tick, on_attack | 钩子：普攻无伤；目标 3.5~7 格（`special_min_range`/`special_range` 缺省）触发，拉速 `pull_speed` 缺省 8.5，冷却 `special_load_time` 缺省 1.3s，钩伤 `special_damage` 缺省 `damage`（`card_mechanics.py:277-310`） |
| `MegaKnight` | 509 | on_spawn, on_tick, on_attack | 落地溅射（`MegaKnightAppear`，伤害取 `damage_per_level`，半径缺省 2200/1000，击退缺省 1000/1000）；冲刺跳 3.5~5.0 格（`dashMinRange/dashMaxRange` 缺省 3500/5000），预备 0.5s + 空中 1.2s（合计 1.7s）；上勾拳 `pushBackStrength/1000` 格（`card_mechanics.py:533-636`） |
| `Musketeer` | 639 | on_tick, on_attack | 觉醒狙击弹：按 `attackSequenceList` 逐发交替，`customRange`(缺省 0) `/1000` 临时扩展射程，攻击后归零（`card_mechanics.py:648-664`） |
| `Ronin` | 669 | on_tick, on_take_damage | 格挡：`parryReflectPercent`(缺省 200)/100 反弹、`parryCooldownMs`(缺省 3500)/1000 冷却；`MELEE_RANGE_THRESHOLD = 1.5`，空中近战免疫（`card_mechanics.py:679-711`） |
| `Assassin` | 715 | on_tick, dash_tick | 突进窗 `DASH_MIN = 3.5` / `DASH_MAX = 6.0`，无敌 0.8s，抵达伤害 `dashDamage×level_scale`，移速 `jump_speed`（`card_mechanics.py:725-807`） |
| `BattleHealer` | 810 | on_spawn, on_tick, on_attack | 治疗光环：部署 2.5 格 / 攻击 3.0 格，`TICKS=4`、`INTERVAL=0.25`、`DEPLOY_HEAL_LV11 = 50`（按 `1.1^(level−11)`），攻击光环 `ceil(deploy/2)`（`card_mechanics.py:818-853`） |

**冠军能力族（`_HeroBase`，`card_mechanics.py:315`）**

| 类 | 行号 | 机制 / 关键数值 |
|---|---|---|
| `SkeletonKing` | 319 | 灵魂召唤：`min(6 + souls, 16)`，前摇 0.9s，每 0.25s 放 1 只，半径取 `OFFICIAL_OVERRIDES['SkeletonKing']['spawn_radius']`(缺省 3.5)，角度步进 2.399963（`card_mechanics.py:320-352`） |
| `ArcherQueen` | 355 | 隐身斗篷 3.5s，`hit_speed_mult=2.8`，`speed_mult=0.75`（`card_mechanics.py:356-372`） |
| `GoldenKnight` | 375 | 连环突进：`dash_remaining = 10`，每段 5.5 格内最近未突进目标，伤害 `131 × level_scale`，命中公主塔即停（`card_mechanics.py:375-413`） |
| `Monk` | 416 | 禅定 4s，`damage_reduction` 缺省 0.65（`card_mechanics.py:416-432`） |
| `MightyMiner` | 435 | 每局限 2 次；钻地 0.6s；原地炸弹 `130×level_scale`、半径 2.0、延迟 1.0、击退 1.8；瞬移 `x → 18.0 − x`（`card_mechanics.py:435-458`） |
| `LittlePrince` | 461 | 召唤 `ChampionGuard`，落地 `90×level_scale`、半径 1.5、击退 2.0 格（`card_mechanics.py:461-474`） |
| `BossBandit` | 477 | 手雷：每局限 2 次、隐身 1.0s、向身后传送 6 格（clamp 到 `[0.5, 31.5]`）；攻击侧被动冲刺 3.5~6.0 格双倍伤害（`card_mechanics.py:477-504`） |

**M8 Elite17（Hero）族**（`HERO_CLASSES` 表见 `card_mechanics.py:1562-1579`）

| 类 | 行号 | 关键数值 / 要点 |
|---|---|---|
| `HeroKnight` | 900 | 嘲讽 `tauntRadius` 内敌军 + 护盾 `shieldValue`（`elite17_data` 表） |
| `HeroMusketeer` | 932 | 前方 `frontOffset` 放炮塔，落地 `spawnDamage`、`spawnRadius` |
| `HeroMiniPekka` | 948 | 煎饼进度 `meterSeconds`/`onHitProgress`、`maxMeter`；吃饼等级 `levelsByMeter`，`mult = 1.1**steps`，回复 `healPct` |
| `HeroValkyrie` | 989 | 旋风 `whirlDuration`、`tick`、`radius`、`tickDamage`、`crownMult`；结束冲刺 `dashRange`，禁攻 `forbidAttack` |
| `HeroWizard` | 1051 | 延迟 `flyDelay` 后升空 `flyDuration`、`speedMult`；火球命中处生成 `tornadoRadius/tornadoDuration` 旋风，`tornadoDps` |
| `HeroBowler` | 1107 | 蓄力 `chargeTime` → 迫击炮 `siegeRange/siegeHitSpeed/siegeShots/siegeDuration/siegeDamage/crownMult`，退出还原 `_orig` |
| `HeroGiant` | 1158 | 抓 2 格内 HP 最高敌部队，水平扔 `throwRange` 格，落地 `impactDamage/impactRadius/stun` |
| `HeroGoblins` | 1197 | 2s 内同批算同组；最后一只死亡开 `window` 窗；`Brigade` 增援 `brigadeCount` |
| `HeroMegaMinion` | 1233 | 标记最低 HP 敌人；瞬移 + `warpDamage/warpRadius`；`tower_damage_mult = crownMult`（永久） |
| `HeroTombstone` | 1278 | 预付能力；墓碑破碎 `on_death` 生出 `queenCard`，寿命 `queenLifetime`；构造时 `pop('spawnCharacterData')` |
| `HeroBerserker` | 1305 | 熊灵 `duration` 内 `hitSpeed/speed(÷50)/damage`，受击 HP 下限 clone 到 1 |
| `HeroDarkPrince` | 1343 | 下马：本体溅射 `landingRadius`；`charge_range = 0`；生 `mountCard` 犀牛（`DarkPrinceHeroRhino`, `card_mechanics.py:1367`） |
| `HeroBalloon` | 1373 | 6 格内最近地面敌人投放骷髅伞兵（`Skeletrooper` 类见 `card_mechanics.py:1398`） |
| `HeroIceWizard` | 1447 | 冰封自身 `cubeDuration` → 破碎 `EvoEffectZone(freezeRadius, slowMult, freeze)` |
| `HeroEliteArcher` | 1494 | 瞬移 `warpRange` + 三连射 `tripleShots/tripleHitSpeed` + 假人 `dummyCard/dummyLifetime` |
| `HeroIceGolemite` | 1542 | `IceGolemiteSnowZone` 冰环：`radius/pulses/interval/pulseDamage/slowMult/slowDuration/smallFreeze` |

**勘误/专项族**

| 类 | 行号 | 钩子 | 机制 / 关键数值 |
|---|---|---|---|
| `ElectroGiant` | 1584 | on_damaged | 反射 `reflectedAttackDamage`(缺省 75)×`level_scale` + 眩晕 `reflectedAttackBuffDuration`(缺省 500ms)；半径 `reflectedAttackRadius`(缺省 2000)/1000；冰冻期不反射 |
| `_AttackStunMixin` | 1603 | on_attack | 命中部队附 `_stun_time = 0.5` 眩晕 |
| `ElectroWizard` | 1613 | on_spawn | 落地 3.0 格 / `75×level_scale` / 0.5s 眩晕 |
| `MiniSparkys` | 1626 | — | 仅继承攻击眩晕 |
| `ElectroSpirit` | 1630 | on_attack | 链电 `chainedHitCount`(缺省 9)、`chainedHitRadius`(缺省 4000)/1000；投射物伤害 `×1.1^(level−1)`；命中后自毁 |
| `RamRider` | 1662 | on_attack | 继承 Prince 冲锋 + 命中部队 `speed_mult=0.30, duration=2.0` |
| `MovingCannon` | 1672 | on_tick | HP ≤ 50% → 变身 `BrokenCannon`（保留 HP、`deploy_delay_remaining=0`、非 `die()`） |
| `Phoenix` | 1692 | on_death | 亡语火球 `damage`(缺省 64)×scale、半径 `radius`(缺省 2500)/1000、击退 `pushback`(缺省 2000)/1000 + 产蛋 |
| `PhoenixEgg` | 1715 | on_tick | `hatch = 4.3` 秒后出满血 Phoenix（`_is_rebirth=True`） |
| `ThreeMusketeers` | 1738 | on_tick | 3 格内有敌部队 → `_range_override = 1.2`，否则 6.0；切换时清 `target_id/path` |
| `GoblinGiant` | 1761 | on_tick, on_death | 骑手投掷射程 5.5、攻速 `hitSpeed`(缺省 1700ms)；投射物伤害 `×1.1^(level−1)`；死亡落地 SpearGoblin |
| `King_KnifeTowers` | 1807 | on_attack, on_tick | 飞刀 `MAX_KNIVES = 8`、`RECHARGE = 0.9`、有刀时攻速 `0.5s`；耗尽期间 `attack_cooldown = RECHARGE` |
| `King_ChefTowers` | 1842 | on_attack, on_tick | `BASE_COOK = 23.0`、`FIRST_COOK = 7.0`、`MAX_COOK = 38.0`；攻击延长 `Δ=(38−23)×hitSpeed/38`；一塔失速率 ×2、双塔失停止；只喂 `hp > 33%` 的友军部队，`+10% max hp`（可超上限）与 `_damage_mult ×1.1` |

模块级函数：`open_goblin_window`（`card_mechanics.py:877`）、`_goblin_brigade_effect`（`card_mechanics.py:886`）。

> 源码本身对多处数值标了【假设】/【待对拍】/【Fandom】（如 `card_mechanics.py:571, 815-816, 951, 1219, 1309`），
> 这些是**引擎已实现但官方口径未对拍**的项，不属本部分「从源码确认」的范围（见 B.9 #11）。

---

## B.7 法术与区域效果

### B.7.1 引擎侧「落点解析」链路（`deploy_card` 法术分支）

`deploy_card` 先做**类型无关**检查，再按法术子类分流（`battle.py:2857-2994`）：

| 分支 | 条件 | 行号 | 产物 |
|---|---|---|---|
| BarbLog 专属 | `card_name == 'BarbLog' and not _from_mirror` → `arena.can_deploy_at(..., is_spell=False)` | 2854-2856 | 非法直接 `return False`（**唯一有部署区限制的法术**） |
| Lightning | 硬编码分支 | 2892-2895 | `_cast_lightning(player, position)`（`battle.py:2783-2804`） |
| 区域持续出兵（墓园类） | `srow['spawn_character']` | 2896-2911 | `radius` 缺省 3000/1000、`duration` 缺省 5000/1000、`interval` 缺省 500/1000，`count = int((duration−initial)/interval)`，黄金角 2.399963 散布，`delayed_spawn` |
| 克隆 | `srow['clone']` | 2912-2924 | 半径内友军 `Troop` 克隆，`clone.hp = 1.0`、`_soul_excluded = True` |
| 瞬发区域法术 | `(buff or controls_buff) and not projectile` | 2928-2939 | `AreaEffect`（觉醒 Zap 走 `EvoZapZone`） |
| 滚动类 | `card_name in ('Log','BarbLog')` | 2945-2961 | 直接从落点生成 `LogProjectileRolling` / `BarbLogProjectileRolling`，方向强制 `(0,±1)` |
| 弹道波次 | `card_info.projectiles` | 2941-2994 | 从己方国王塔 `delayed_spawn` 出弹，`wave_interval` 间隔；觉醒 GoblinBarrel 追加诱饵 |

**法术没有边界/阻挡校验**：`card_info.type != 'spell'` 的部署区判定整段被跳过（`battle.py:2865`），
BarbLog 以外的法术不调 `can_deploy_at`。⇒ 直接调用 `deploy_card` 时越界坐标不会被拒（见 B.9 #6）。

### B.7.2 `AreaEffect`（Zap/Freeze/Heal/Rage/Tornado/Earthquake/Poison）

构造（`battle.py:1728-1783`）：

| 参数 | 取值/缺省 | 行号 |
|---|---|---|
| `radius` | `spells[card].radius`，缺省 3000 → `/1000` | 1744 |
| `lifetime` | `OFFICIAL_OVERRIDES.duration` 或 `life_duration/1000` | 1745 |
| `tick` | `OFFICIAL_OVERRIDES.tick` 或 `hit_speed/1000`，`or 0.5` | 1746 |
| `damage_per_tick` | ① `damage_per_tick_lv11 × 1.1^(lv−11)`；② `damage_lv11 × 1.1^(lv−11)`；③ `damage_per_level` 按稀有度轴；④ 行 `damage`；⑤ **DOT**：`buff_data.damage_per_second × tick × level_scale(lv)` | 1759-1769 |
| `crown_pct` | `OFFICIAL_OVERRIDES.crown_tower_percent` 或 `(行 crown_tower_damage_percent 或 buff_data 同名字段) + 100`，再 `/100`；无字段则 `None` | 1770-1774 |
| `building_mult` | `1 + buff_data.building_damage_percent/100`（Earthquake → ×4.5） | 1776 |
| `speed_mult` | `(100 + buff_data.speed_multiplier)/100` | 1778-1779 |
| `heal_per_tick` | `heal_per_tick_lv11 × 1.1^(lv−11)` | 1781 |

命中判定（**罩圈公式**）：

```
_in_radius(e) := e.position.distance_to(self.position) <= self.radius + e.data.collision_radius
```
（`battle.py:1785-1786`）

`_pulse()`（`battle.py:1788-1831`）逐实体过滤顺序：排除 `Projectile/SpawnProjectile/AreaEffect`；
`only_enemies` / `only_own_troops` / `ignore_buildings`；`_in_radius`。然后按 `buff_name` 分发：

| `buff_name` | 效果 | 行号 |
|---|---|---|
| `ZapFreeze` | `apply_buff(stun=buff_time, retarget=True)` | 1798-1799 |
| `Freeze` | 整场只施加一次（`stun_applied` 闸门） | 1800-1804 |
| `Earthquake` / `Poison` | 减速 `speed_mult`，时长 `buff_time or 1.0` | 1805-1808 |
| `Rage` | `speed_mult` 缺省 1.30、残留 `residue` 缺省 1.0s | 1809-1811 |
| `Heal` | `apply_buff(heal={'hps': heal_per_tick/tick, 'time': tick})` | 1812-1814 |

伤害结算（`battle.py:1817-1828`）：**塔**（`isinstance(e, Building)` 且 `crown_pct is not None` 且
名字含 `King` 或 `PrincessTower`）→ `damage_per_tick × crown_pct`；普通建筑且 `building_mult != 1` →
× `building_mult`；否则原伤害（觉醒骷髅军团亡影 `pierce_invincible=True`）。
Freeze 首次 pulse 后把 `damage_per_tick` 置 0（施法单次伤害）（`battle.py:1829-1831`）。

`update`（`battle.py:1833-1868`）：
- Tornado 拉拽（`controls`）：只有 `Troop` 受影响，落点方向匀速收敛、`is_walkable` 才落位（`battle.py:1837-1847`）；
- Rage 施法一次性伤害：`179 × 1.1^(Card.default_level−11)`，建筑只吃 `×0.3`（`battle.py:1848-1860`）；
- pulse 先于寿命判定（保证 `life_duration = 1ms` 的瞬发法术也打满一次）（`battle.py:1861-1865`）；
- `lifetime <= 0` → `is_alive = False`（`battle.py:1866-1868`）。

> **没有「停表」机制**：引擎不冻结全局时钟，`Freeze/Zap` 的「停」= 给实体加 `freeze_timer`，
> 使 `Troop.update` 提前 `return`（停移停攻、冷却暂停）（`battle.py:1101-1104`）；对建筑由
> `Building.update` 自行处理。若指「游戏时间停止」，源码中不存在该机制。

### B.7.3 弹道法术的命中与塔伤

`Projectile._deal_splash_damage`（`battle.py:1596-1628`）：

1. **Monk 禅定反弹**：命中点半径内有 `deflect_active` 的敌方实体 → 整个法术 `reflect_to_tower(实体, 伤害)` 并 `return`（`battle.py:1598-1604`）；
2. 过滤：`invincible` 跳过、同阵营跳过、`hits_air`/`hits_ground`（`battle.py:1606-1609`）；
3. 命中：`entity.position.distance_to(target_position) <= proj.radius + entity.data.collision_radius`（`battle.py:1611-1612`）；
4. 伤害：`base = _damage()`（可被 `damage_override` 覆盖）；**名字含 `King` → `round(base × crown_tower_percent)`**（`battle.py:1613-1615`）；
5. 击退 `pushback`（ms→格）与 `buff_time` 减速 / 女巫诅咒 5s（`battle.py:1616-1628`）。

> 塔名约定：引擎 6 座塔的 `Card(...).name` 全部以 `King` 开头（会话内实测：
> `King_PrincessTowers` / `KingTower` / `King_CannonTowers` / `King_KnifeTowers` / `King_ChefTowers`），
> 因此 `"King" not in entity.name` 这个看似只判国王塔的条件，**实际覆盖全部塔**（公主塔与塔兵都吃降伤）。
> 公主塔卡名同时含 `PrincessTower`（`battle.py:2554-2559`），所以 `in_attack_range` 的 `bonus=0.5` 对公主塔生效。

滚动弹（Log/BarbLog）走另一条分支：沿 `projectileRange` 滚动，逐个判定
`中心距 < 目标碰撞半径 + proj.radius`、**只打地面**（`battle.py:1560-1567`），击退后位移（`battle.py:1569-1585`），
终点也走出兵链（`battle.py:1554-1558`）。

### B.7.4 Lightning

`_cast_lightning`（`battle.py:2783-2804`）：半径 `spells['Lightning'].radius`（3500 → 3.5）；
候选 = 半径内非 `Projectile/SpawnProjectile/AreaEffect` 敌方实体；按 `data.hp` 降序取前 3；
**塔伤 ×0.65**（`battle.py:2802`）；部队附 0.5s 眩晕 + 重索敌（`battle.py:2803-2804`）。
伤害取 `projectiles['LighningSpell']`（注意源码里该键名拼写为 `LighningSpell`，`battle.py:2788`）。

### B.7.5 `TimedExplosive`（Balloon / GiantSkeleton 亡语炸弹）

`dsd.range = 3.0` 硬编码、`crown_tower_damage_percent` 来自 `deathSpawnCharacterData.crownTowerDamagePercent/100`
（`card_utils.py:497-505`）；命中判定为 `中心距 − 碰撞半径 < range`（`battle.py:2391`）；
降伤条件写成 `entity.name in ('King_PrincessTowers', 'KingTower')`（`battle.py:2392`）——
即**塔兵变体（`King_CannonTowers` 等）不在元组里，不吃降伤**，与 B.7.3 的 `"King" in name` 口径不一致。

### B.7.6 `spell_module.py`：法术知识查询服务（三层 API）

| 层 | 函数 | 行号 | 口径 |
|---|---|---|---|
| 静态档案 | `get_spell_profile(card, level)` | 204-210 | 按 `(卡名, 等级)` 惰性缓存 `_PROFILE_CACHE` |
| 落点估值 | `evaluate_cast(battle, pid, card, pos)` | 235-296 | **零推演**静态预测；命中式 `中心距 > radius + col` 即否（`spell_module.py:273`） |
| 引擎实测 | `engine_resolution(battle, pid, card, pos)` | 95-139 | `deepcopy` → `deploy_card` → `step(dt)` 到施法实体消亡（`max_t=15.0`, `dt=1/60`），逐实体 hp 差 |
| 最优落点 | `best_cast(battle, pid, card, grid=1.0)` | 311-338 | 在敌方实体包围盒外扩半径上粗网格扫描 |

- 半径口径 `_radius_m`（`spell_module.py:64-75`）：`spells[card].radius` → `Card.data.radius` →
  `data.projectileData.radius`，最后 `/1000`。实测：`Arrows 3.5 / Fireball 2.5 / Zap 2.5 / Poison 3.5 /
  Tornado 5.5 / Lightning 3.5 / Earthquake 3.5 / BarbLog 1.3 / Log 1.95 / GoblinBarrel 1.5 / Heal 4.0 /
  Freeze 0.0 / Rage 3.0`（会话内调用 `_radius_m` 实测）。
- 伤害型判据 `_deals_damage`（`spell_module.py:54-61`）：`projectileData.damage` **或** `data.damage`
  **或** spells 行 `damage` **或** `buff_data.damage_per_second`。
- 空中/地面过滤 `_hits_filters`（`spell_module.py:221-232`）：仅当卡真有 `projectileData` 时按
  `hits_air/hits_ground` 过滤，否则全命中。
- 标定靶（`spell_module.py:43-51`）：部队靶 `Giant`、建筑靶 `Cannon`、塔靶取 `entities[1]` 位置；
  部队靶候选位 `[(9,19),(9,12)]`；塔 id 表 `{0:(3,4,6), 1:(1,2,5)}`。
- 评分（`spell_module.py:299-308`）：`elixir_killed_value + tower_damage_total/500 + troop_damage_total/650`，
  常量 `TOWER_HP_PER_ELIXIR = 500.0`、`TROOP_HP_PER_ELIXIR = 650.0`。
- 击杀折费：`(data.elixir or 0) / max(data.spawn_number, 1)`（`spell_module.py:285`）。
- DOT 语义：`dot = lifetime >= 2500`（`spell_module.py:185`），部队伤害按「目标满驻留」上界（docstring `spell_module.py:25-27`）。

### B.7.7 塔伤修正汇总

| 位置 | 修正 | 行号 |
|---|---|---|
| `AreaEffect._pulse` | 名字含 `King`/`PrincessTower` → `damage_per_tick × crown_pct` | `battle.py:1819-1822` |
| `AreaEffect._pulse` | 普通建筑 → `× building_mult`（Earthquake `building_damage_percent=350` → ×4.5） | `battle.py:1823-1824`（说明见 `battle.py:1775-1776`） |
| `AreaEffect.update` | Rage 对建筑 `×0.3` | `battle.py:1857-1860` |
| `Projectile._deal_splash_damage` | 名字含 `King` → `× crown_tower_percent` | `battle.py:1614` |
| `Projectile`（Arrows 特例） | `ArrowsSpell` → `crown_tower_percent = 25/122` | `card_utils.py:461-465` |
| `Lightning` | 塔 `×0.65` | `battle.py:2802` |
| `TimedExplosive` | 仅 `('King_PrincessTowers','KingTower')` 吃降伤 | `battle.py:2392-2395` |
| `BasicCharacter.on_attack` | 普攻对塔 `× data.tower_damage_mult`（名字含 `King`/`PrincessTower`） | `core.py:47-56` |
| `deal_area_damage` | 名字含 `King` → `× crown_tower_damage_percent` | `battle.py:3257` |

### B.7.8 命中判定公式对照

| 实现 | 公式 | 行号 |
|---|---|---|
| `AreaEffect._in_radius` | `中心距 ≤ radius + 目标碰撞半径` | `battle.py:1785-1786` |
| `Projectile._deal_splash_damage` | 同上 | `battle.py:1611-1612` |
| `spell_module.evaluate_cast` | 同上 | `spell_module.py:273` |
| `Lightning` | 同上 | `battle.py:2797` |
| `GenericBomb` | 同上（另加 `hits_air/hits_ground`） | `battle.py:1920-1923` |
| `TimedExplosive` | `中心距 − 目标碰撞半径 < range(3.0)` | `battle.py:2391` |
| `deal_area_damage` | `edge_distance_from(position) < range`（塔=矩形边缘，普通=中心距−半径） | `battle.py:3250-3265` |
| 单位普攻射程 | `edge_distance_from ≤ range (+0.5 公主塔)` | `battle.py:586-604` |

---

## B.8 部署合法性

### B.8.1 坐标契约

`SubAction(x, y)` 一律是**玩家本地网格坐标**，唯一换算入口 `sub_position`：
玩家 0 → `(x+0.5, y+0.5)`；玩家 1 → `(17.5−x, 31.5−y)`（`rl/action_bundle.py:31-39`）。
`legal_cells` 与 `validate_bundle` 走同一入口，避免「掩码世界坐标 vs 提交镜像坐标」分裂
（`rl/action_mask.py:6-9, 462, 473, 563`）。

### B.8.2 `legal_cells` 判定流程（逐格）

（`rl/action_mask.py:441-476`）

1. 建 `cells = np.ones((32,18), bool)`（行=y、列=x）；
2. `eff = _effective_card(p, card_name)`（Mirror → `last_card`），`eff_info = Card(eff)`（`action_mask.py:452-455`）；
3. **法术支**（`eff_info.type == 'spell'`，`action_mask.py:456-470`）：
   - `deals_dmg = _spell_deals_damage(eff, eff_info)`；仅伤害型才取半径，否则 `0.0`；
   - `ev_gate = radius > 0.0 and deals_dmg`；
   - 每格：贴**已毁敌方塔本体**（`DEAD_TOWER_BODY_R = 0.9`）→ False；
     半径 > 0 且罩不到任何存活敌方目标 → False；`ev_gate` 且纯砸塔亏费 → False；
4. **非法术支**（`action_mask.py:471-476`）：逐格 `_position_legal(...)`。

`_position_legal` 步骤（`rl/action_mask.py:382-438`）：

| 序 | 条件 | 结果 | 行号 |
|---|---|---|---|
| 1 | 卡是法术 | 进法术子流程（下同） | 394 |
| 1a | 贴已毁敌方塔本体 | False | 395-396 |
| 1b | 伤害型且半径 > 0 且罩不到存活敌方目标 | False | 397-400 |
| 1c | 半径 > 0 且 `_spell_tower_ev_illegal` | False | 402-404 |
| 1d | 否则 | **True**（法术不做部署区/占位校验） | 405 |
| 2 | `is_position_occupied_by_building(pos, 0.0)` | False | 406-407 |
| 3 | `type == 'building'` 且 `arena.is_behind_king(pos, pid)` | False | 409-410 |
| 4 | 玩家 0：`pos.y <= 1.0` 且（`x <= 6.0` 或 `x > 12.0`） | False | 411-413 |
| 5 | 玩家 0：`pos.y >= 21.0` | False | 414-415 |
| 6 | 玩家 0：`pos.y >= 15.0` → 按 `x <= 9` 查左塔、否则右塔；对应敌方塔存活 | False | 416-422 |
| 7 | 玩家 1 镜像：`pos.y > 31.0` 越界带 / `pos.y <= 10` / `pos.y <= 17` 塔存活 | False | 423-434 |
| 8 | `_backline_placement_illegal`（8h 坦克后屯兵几何门） | False | 435-437 |

**注意：`_position_legal` 没有 Miner 例外**（引擎 `deploy_card` 有，见 B.8.4）。

`_spell_tower_ev_illegal`（`action_mask.py:158-219`）条件全部满足才拒：
`battle.time < 120.0`（双倍期前）；伤害型且有半径；对塔标定伤害 > 0；罩得到对手存活公主塔；
半径内无对手非塔目标；且 `dmg_eff / 500 < edw(0.5) × 卡费`，
其中 `dmg_eff = dmg × best_mult`，`best_mult` 由 `rl.env_wrapper.tower_value_mult(hp/3052, king=False, princesses_alive)` 取最残塔最大值
（`action_mask.py:172-219`；常量 `TOWER_HP_PER_ELIXIR_EARLY = 500.0`、`SPELL_EV_EDW = 0.5`，`action_mask.py:119-120`）。

### B.8.3 `slot_mask` / `validate_bundle`

- `_slot_playable`：`king_tower_hp > 0`、卡在 `cycle[:4]`、费用可算、`elixir >= cost`
  （`action_mask.py:46-56`）；`_card_cost`：Mirror = `Card(last_card).elixir + 1`，无 `last_card` → `None`
  （`action_mask.py:30-36`）。
- `validate_bundle`（`action_mask.py:521-573`）：逐子动作——技能查 `ability_mana` 与圣水；slot 0 跳过；
  越界/重复拒绝；`_position_legal` 拒绝；模拟扣费；最后 `solo_commit_blocked` 兜底
  （单张高承诺卡无圣水优势 → 拒绝整包，`action_mask.py:568-572`）。
- `slot_mask` 返回 `(K_MAX=4,)` 布尔（`action_mask.py:59-68`；`K_MAX = 4` 见 `rl/action_bundle.py:28`）。

### B.8.4 与引擎 `deploy_card` 的差异

引擎侧（`battle.py:2806-2887`）与掩码逐条对照：

| 校验项 | 引擎 | 掩码 | 是否一致 |
|---|---|---|---|
| 手牌/圣水/王塔 | `can_play_card`（`player.py:36-39`：`cycle[:4]` + `elixir >= Card.elixir` + `king_tower_hp > 0`） | `_slot_playable`（`action_mask.py:46-56`） | 一致（除 Mirror 动态费） |
| 建筑/塔占位 | `is_position_occupied_by_building(position, 0)`（`battle.py:2867`） | `(pos, 0.0)`（`action_mask.py:406`） | 一致 |
| 王塔后禁建筑 | `card_info.type == 'building'`（`battle.py:2869`） | `card_info.type == "building"`（`action_mask.py:409`） | 一致 |
| **Miner 全场可部署** | `if card_name != 'Miner' and player_id == 0:`（`battle.py:2872, 2880`） | **无 Miner 例外**（`action_mask.py:411-434`） | **不一致：掩码更严** |
| 部署区坐标边界 | `y <= 1.0 / y >= 21.0 / y >= 15.0`（P0）等（`battle.py:2873-2887`） | 同式（`action_mask.py:411-434`） | 一致 |
| **BarbLog 部署区** | `arena.can_deploy_at(..., is_spell=False)`（`battle.py:2854-2856`） | 法术支第 1d 步直接 `True`（`action_mask.py:405`） | **不一致：掩码更松** |
| **Mirror** | 重放 `last_card` → 走**那张卡**的全部部署校验（`battle.py:2808-2825`） | 法术支按 `Mirror` 自身 `type == 'spell'` 返回 True（`action_mask.py:394-405`） | **不一致：掩码更松**（实测见下） |
| 法术空砸/已毁塔 | 无此校验 | 有（7h/8h） | 掩码更严（有意） |
| 后排几何门 | 无 | 有（8h） | 掩码更严（有意） |
| 不裸下 | 无（只在 `_finish_deploy` 之后无校验） | `solo_commit_blocked`（`action_mask.py:255-276`） | 掩码更严（有意） |

**会话内实测对账**（脚本 `.tmp/mask_vs_engine.py`：对每张卡 × 双方 × 576 格，
比较 `legal_cells[y,x]` 与 `deepcopy` 后 `deploy_card` 的返回值）：

| 卡 | mask=True 格数 | 「掩码 False / 引擎 True」 | 「掩码 True / 引擎 False」 |
|---|---|---|---|
| `Knight` | 224 | 0 | 0 |
| `Giant` | 224 | 0 | 0 |
| `Cannon`（建筑） | 220 | 0 | 0 |
| `Miner` | 224 | **284** | 0 |
| `Arrows` | 36 | 540 | 0 |
| `Fireball` | 44 | 532 | 0 |
| `Zap` / `Lightning` / `Poison` / `Tornado` / `Earthquake` / `Freeze` / `Heal` / `Log` / `GoblinBarrel` / `Rage` | 576 | 0 | 0 |
| `BarbLog` | 576 | 0 | **352** |
| `Mirror`（`last_card='Knight'`） | 576 | 0 | **352** |

（双方一致；表中为 player 0 的数值，player 1 对称同数。）

**读法**：
- `Miner`：掩码把敌半场 284 格判非法，引擎接受 → **合法动作被掩码屏蔽**（能力被阉割，非安全风险）。
- `Arrows/Fireball`：掩码额外挡掉 540/532 格（空砸闸门 + 已毁塔本体）→ 设计如此；
- `BarbLog`、`Mirror`：掩码放行 352 格而引擎拒绝 → **掩码缺口**。此路径已被引擎侧护栏接住：
  `env_wrapper` 在 `validate_bundle` 通过后逐卡提交，遇到 `deploy_card` 返回 False 时
  `invalid_count += 1` 并发 `RuntimeWarning("P1-20: validate 通过但引擎拒绝 ... —— 掩码缺口")`，
  **整包已部分提交、无法回滚**（`rl/env_wrapper.py:618-642`），且 `invalid_count` 会扣奖励
  （`rl/env_wrapper.py:261-262`）。
- `Zap/Poison/...` 全 576 格一致，是因为掩码自己的 `_spell_deals_damage` 对它们返回 `False`
  ⇒ 半径闸门整体跳过；而引擎对法术也没有位置校验 ⇒ 双方都「全放行」。

**另一个双口径事实**：`action_mask._spell_deals_damage`（`action_mask.py:89-94`，只看
`data.projectileData.damage` 与 `data.damage`）与 `spell_module._deals_damage`
（`spell_module.py:54-61`，额外看 spells 行 `damage` 与 `buff_data.damage_per_second`）
对同一张卡结论不同（会话内实测）：

| 卡 | `action_mask._spell_deals_damage` | `spell_module._deals_damage` |
|---|---|---|
| `Zap` / `Poison` / `Tornado` / `Earthquake` / `Freeze` / `Heal` | False | True |
| `Arrows` / `Fireball` | True | True |
| `Lightning` / `Rage` / `BarbLog` / `Log` / `GoblinBarrel` / `Mirror` | False | False |

⇒ 「伤害型法术才受空砸闸门约束」这条规则在掩码层对 `Zap/Poison/Tornado/Earthquake` 实际**未生效**。

### B.8.5 禁放区差异（按卡类）

| 卡类 | 禁放区 |
|---|---|
| 部队（非 Miner） | 己方半场（P0：`y ∈ (1,21)` 内、且 `y ≥ 15` 时要求对应敌方公主塔已破）；底边 `y ≤ 1` 仅 `6 < x ≤ 12`；引擎与掩码同式（`battle.py:2872-2879`；`action_mask.py:411-422`） |
| Miner | **全场**（含敌半场），仅受塔/建筑占位与边界限制（`battle.py:2871-2872`）；掩码未实现该例外 |
| 建筑 | 部队规则 + 王塔身后 1 格宽禁区 `(7,0)-(11,1)`（P0）/ `(7,31)-(11,32)`（P1）（`arena.py:86-98`；`battle.py:2869`；`action_mask.py:409`） |
| 普通法术 | 无部署区限制、无占位限制；掩码额外挡「已毁敌方塔本体（半径 0.9）」与空砸/纯砸塔 |
| BarbLog | 引擎=部队部署区（`is_spell=False`）；掩码**无此限制** |
| 法术（墓园/克隆/滚木/弹道波次） | 引擎按分支处理，无位置校验 |

---

## B.9 本部分待确认清单

| # | 条目 | 无法确认的原因 | 要确认需要什么 |
|---|---|---|---|
| 1 | `pathfinding.py` 是否存在**仓库外**调用方 | 只能对本仓 grep；仓外/部署脚本不可见（本仓内已确认无 import 方：`battle.py:4` 导入的是 `pathfinding_heap`） | 部署清单/外仓引用检索；或直接删除前的 owner 确认 |
| 2 | `pathfinding.py:61-62` 的 `self.battle_state` 未赋值错误是否曾被触发/修复过 | 该分支在现有代码里不可达（模块无调用方），无运行期证据 | `git log -p src/clasher_new/pathfinding.py`（本部分未允许运行历史挖掘）或恢复调用后复现 |
| 3 | `tilemap_lane_grid.txt` 里 `W` 代价 800（地面）是否真的只对桥最外 16 个可走半格生效 | 会话内只实测了「可走 W 格 = 16 个」（`nx∈{4,9,26,31} × ny∈{30..33}`），未跑路径对比 | 端到端路径取证：同一单位在桥面不同 x 起点的 A* 路径对比 |
| 4 | `arena.is_walkable` 用 `(int(x), int(y))` 整格缓存、而寻路是半格网格，是否有意 | 源码无注释说明 | 设计文档/对拍；或改键为半格后跑掩码与路径回归 |
| 5 | 寻路「车道保持」（`_lane_offset` + goal 择优）在实践中是否真的成立 | 本部分只读代码，未做回放/轨迹取证 | 回放逐帧取路径与横向偏移统计 |
| 6 | `deploy_card` 对法术**无边界校验**是否会被越界坐标触发 | 本部分只审了 `env_wrapper` 提交路径；`battle.py`/`server.py`/评估脚本的其它 `deploy_card` 调用点未全审 | 全仓 `deploy_card(` 调用点审计 + 逐点坐标范围检查 |
| 7 | `BarbLog`/`Mirror` 掩码缺口在训练中的实际触发频率与代价 | 需要跑训练/评估并统计 `P1-20` 告警与 `invalid_count` | 跑一段训练或对回放统计 `RuntimeWarning`/`invalid_count` |
| 8 | `Miner` 掩码过严造成的策略影响量级 | 需要行为取证（该卡是否本就在卡池里、使用率如何） | `scripts/forensics_card_usage.py` 类回放统计 |
| 9 | 引擎是否存在**通用 aggro/仇恨**机制 | grep 仅命中注释文案，未见引擎实现；但可能藏在未读模块（如 `server.py`/`client_side`） | 全仓 grep `aggro`/`taunt`/`threat` 并逐点确认（本部分只覆盖 `battle.py`/`card_mechanics.py`/`rl`） |
| 10 | `threat_calc` 是否被间接调用（经 `agent_pool`/`belief_planner` 动态 import） | 静态 grep 无命中，但动态 import/字符串调用无法排除 | 运行时插桩统计 `estimate_tower_threat` 调用次数 |
| 11 | `card_mechanics.py` 中源码自标【假设】/【待对拍】/【Fandom】的数值是否符合官方口径 | 本部分被禁止上网，也无法从仓库内数据文件交叉验证全部条目 | 官方对拍（Fandom/内存快照）逐条核验，或在 `docs/数值规则查证汇总.md` 里核对（本部分禁止读 docs） |
| 12 | 塔名用 `'King'` / `'PrincessTower'` 子串做类型判定是否是稳定契约 | 现有 5 种塔名恰好都含 `King`（会话内实测）；若新增不以 `King_` 开头的塔兵会失配 | 设计约束确认；或加显式的塔 id/类型判据并跑掩码回归 |
| 13 | 「停表」语义（本部分结论：引擎无全局停表，只有 `freeze_timer` 定身） | 只能证伪「存在」，不能确认官方是否要求别的形式 | 官方机制对拍 |
| 14 | `Lightning` 投射物键名拼写 `LighningSpell`（`battle.py:2788`）是数据层既有拼写还是笔误 | 需查 `cards_stats_projectile.json` 的键名（本部分未读该文件全量） | 读该 JSON 校验键名；若为笔误，改名会改变行为 |

---

PART_B_OK sections=9 citations=310
