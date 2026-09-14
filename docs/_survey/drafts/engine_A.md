# 游戏引擎文档 · 第 A 部分：地图、实体与战斗主循环

> **依据范围**：`src/clasher_new/battle.py`（3267 行）、`arena.py`、`core.py`、`player.py`、`environment.py`、`card_utils.py`（仅地图/坐标/塔相关部分）。
> 行内标注格式为 `（文件名:行号）`；所有数值原样抄自源码，未做四舍五入或换算（源码自身的 `/1000`、`/50` 等换算照抄）。
> 无法从上述源码确认的项一律集中在 **A.7**，正文不猜测。
> 计数口径：文末 `citations` = 全文「文件名.py:行号」形式的来源引用总数（含各表格的「依据」列；同一括号内并列两条引用时计 2 条）。

---

## A.1 引擎概览

### A.1.1 `battle.py` 的位置

`battle.py` 是**战斗模拟的唯一实现文件**：它同时定义实体类族（`Entity` 及其子类）、机制钩子载体（`BasicCharacter`，实现在 `core.py`）、以及战斗状态机 `BattleState`。地图几何在 `arena.py`，坐标原语在 `core.py`，玩家/圣水/卡序在 `player.py`，Gym 包装在 `environment.py`，卡牌数值解析在 `card_utils.py`。

### A.1.2 对外入口

| 入口 | 位置 | 说明 |
|---|---|---|
| `BattleState.__init__(player_0, player_1, card_level=None)` | battle.py:2539 | 构造即铺设 6 座塔（battle.py:2556-2561） |
| `BattleState.step(dt)` | battle.py:2652 | 单帧推进（A.4） |
| `BattleState.deploy_card(player_id, card_name, position, _from_mirror=False)` | battle.py:2806 | 出牌/施法（返回 `False` = 未生效） |
| `BattleState.use_ability(player_id)` | battle.py:3208 | 英雄能力/条件窗按钮 |
| `BattleState.deal_area_damage(...)` | battle.py:3250 | 区域伤害唯一公共结算函数 |
| `BattleState.update_player_hp()` | battle.py:2643 | 实体 HP → `PlayerState` 三塔 HP |
| `BattleState.winner` / `.game_over` | battle.py:2547-2548 | 终局字段（置位点见 A.6） |
| `environment.CREnv` | environment.py:33 | Gym 环境；持有 `self.battle: battle.BattleState`（environment.py:38） |

`CREnv.reset()` 每次新建一个 `BattleState`，双方初始圣水写死 **5.0**（environment.py:57-58）；`CREnv` 的 `speed` 默认 1.0、`card_level` 默认 11（environment.py:34）。

### A.1.3 依赖方向

| 模块 | 依赖 | 依据 |
|---|---|---|
| `battle.py` | `core.BlankEntity`、`player.PlayerState`、`arena.TileGrid`、`pathfinding_heap.{EntityPathfinder, position_to_cell, cell_to_position}`、`card_mechanics.*`、`card_utils.*`、`evolutions.*` | battle.py:1-8 |
| `arena.py` | `core.Position`（arena.py:2）、`math` | arena.py:1-3 |
| `core.py` | 仅 `dataclasses`/`math`（core.py:1-2） | — |
| `player.py` | `card_utils.Card`（player.py:1） | — |
| `environment.py` | `battle`、`player`、`new_visualization.Visualizer`、`core.Position` | environment.py:1-3 |
| `card_utils.py` | `gamedata.json` + `cards_stats_{characters,spell,building,projectile}.json`（card_utils.py:4-18）、`elite17_data.apply`（card_utils.py:179-180） | — |

⇒ 依赖是单向的：`environment → battle → {arena, core, player, card_utils}`；`arena`/`core`/`player` **不反向依赖** `battle`。

### A.1.4 一处全局命名空间耦合（影响实体构造）

`battle.py:5` 用 `from card_mechanics import *` 把机制类名导入本模块全局命名空间；`Entity.__init__` 随后用 **卡名当类名字符串 eval** 绑定机制钩子载体：

```python
self.entity_holder = BasicCharacter(self)
if self.card_name in globals() and not isinstance(self, Projectile):
    self.entity_holder = eval(f"{self.card_name}(self)")   # battle.py:51-52
```

⇒ 「卡名 ↔ 机制类名」必须同名同命名空间，否则静默回退到默认 `BasicCharacter`。`card_mechanics` 不在本轮允许阅读的源码清单内，本文不描述其内部实现。

---

## A.2 地图与坐标系统

### A.2.1 场地尺寸与坐标范围

| 常量 | 值 | 依据 |
|---|---|---|
| `TileGrid.width, height` | `18, 32`（格） | arena.py:9 |
| `TileGrid.tile_size` | `100.0` | arena.py:10 |
| 合法世界坐标 | `0 <= x < 18`、`0 <= y < 32` | arena.py:100-101 |

**世界坐标 1 单位 = 1 格**：塔几何按「公主塔 3×3（half 1.5/1.5）、国王塔 4×4（half 2.0/2.0）」定义（arena.py:37-38），半径 `2.0` 的圈被注释称为「半径 2 格」（battle.py:567-574），卡牌数值表的毫米量按 `/1000` 归一（如 `range=7500 → 7.0`，card_utils.py:247；`collisionRadius=1000 → 1.0`，card_utils.py:239）。

**坐标系方向**：蓝方（`player_id=0`）王塔在 `y=3.0`、红方（`player_id=1`）王塔在 `y=29.0`（arena.py:11、arena.py:14）⇒ **小 y 侧是蓝方，大 y 侧是红方**；`x` 方向左右对称（蓝左塔 `x=3.5`、蓝右塔 `x=14.5`，arena.py:12-13）。`arena.get_deploy_zones` 的注释把玩家 0 称作 "bottom half"（arena.py:150），但代码给的区间是 `y=1..14`（arena.py:152）——注释方向与数值方向的对应关系需渲染层才能解释（见 A.7 第 5 项）。

### A.2.2 关键坐标常量（原样抄录）

| 常量 | 值 `(x, y)` | 依据 |
|---|---|---|
| `BLUE_KING_TOWER` | `Position(9.0, 3.0)` | arena.py:11 |
| `BLUE_LEFT_TOWER` | `Position(3.5, 6.5)` | arena.py:12 |
| `BLUE_RIGHT_TOWER` | `Position(14.5, 6.5)` | arena.py:13 |
| `RED_KING_TOWER` | `Position(9.0, 29.0)` | arena.py:14 |
| `RED_LEFT_TOWER` | `Position(3.5, 25.5)` | arena.py:15 |
| `RED_RIGHT_TOWER` | `Position(14.5, 25.5)` | arena.py:16 |
| `LEFT_BRIDGE` | `Position(3.5, 16.0)` | arena.py:17 |
| `RIGHT_BRIDGE` | `Position(14.5, 16.0)` | arena.py:18 |
| `RIVER_Y1` | `15.0` | arena.py:19 |
| `RIVER_Y2` | `16.0` | arena.py:20 |

`Position` 是 `core.py` 的 `@dataclass`，仅 `x: float`、`y: float` 两个字段与 `distance_to(other)`（`math.hypot`）（core.py:4-9）。注意 `arena.py` 的塔常量定义在 `@dataclass class TileGrid` 体内——它们没有类型注解，因此是**类属性**而非常规 dataclass 字段；`tile_size: float = 100.0` 才带注解（arena.py:8-10）。

### A.2.3 河、桥与不可通行格

- 河道带：`RIVER_Y1=15.0` 与 `RIVER_Y2=16.0`（arena.py:19-20）；`is_walkable` 判定 `self.RIVER_Y1 <= pos.y <= self.RIVER_Y2` 时**只有桥面可走**：左桥 `2.0 <= x < 5.0`、右桥 `13.0 <= x < 16.0`（arena.py:113-116）。
- `BLOCKED_TILES`（arena.py:21-34），逐项为：
  - 河岸边缘格：`(0,15) (0,16) (1,15) (1,16)`，`x∈[5,12] × y∈{15,16}`，`(16,15) (16,16) (17,15) (17,16)`（arena.py:23-25）；
  - `y=0` 与 `y=31` 两行：`x∈{0..5}` 与 `x∈{12..17}`（每行各 12 格）（arena.py:28-33）。
  ⇒ 桥面整数格为 `x∈{2,3,4}`（左）与 `x∈{13,14,15}`（右），与 `is_walkable` 的半开区间一致。
- `is_walkable` 带全局缓存 `walkable_cache`（按 `(int(x), int(y))` 键）（arena.py:5、arena.py:108-119）。
- `BattleState.in_river(position)` 用另一份同样的河格集合以格号判定（battle.py:2573-2577）——两处**各自硬编码**，不是同一常量源。
- 边界/河道夹取 `ensure_walkability(entity)`（battle.py:2579-2601）：不可走时把坐标夹回，`push_ratio = 0.5`；`y` 下界 `0.5*r`、上界 `32-0.5*r`；`x` 下界写作 `x = r`（注意：**不是** `0.5*r`）、上界 `18-0.5*r`；河道夹取 `y = 15-0.5*r` 或 `17+0.5*r`（取较近者），仅对非空中单位生效（battle.py:2592-2601）。`Building`、`Projectile`、`SpawnProjectile`、`AreaEffect`、`GenericBomb`、以及超骑跳跃空中态直接跳过该修正（battle.py:2583-2588）。

### A.2.4 塔占位（矩形几何）

塔以**轴对齐矩形**参与占位与射程，忽略 `collisionRadius`：

| 塔 | 中心 | 半宽 / 半高 | `player_id` | 依据 |
|---|---|---|---|---|
| 蓝左公主塔 | `(3.5, 6.5)` | `1.5 / 1.5` | 0 | arena.py:40 |
| 蓝右公主塔 | `(14.5, 6.5)` | `1.5 / 1.5` | 0 | arena.py:41 |
| 蓝国王塔 | `(9.0, 3.0)` | `2.0 / 2.0` | 0 | arena.py:42 |
| 红左公主塔 | `(3.5, 25.5)` | `1.5 / 1.5` | 1 | arena.py:43 |
| 红右公主塔 | `(14.5, 25.5)` | `1.5 / 1.5` | 1 | arena.py:44 |
| 红国王塔 | `(9.0, 29.0)` | `2.0 / 2.0` | 1 | arena.py:45 |

约定：矩形 = 中心 ± half，边界属塔内，部署/占位判定用脚点（arena.py:39）。相关函数：

- `dist_to_rect(px, py, cx, cy, hw, hh)`：矩形内 0，矩形外到最近边欧氏距离（arena.py:48-53）。
- `tower_rect_dist(pos, battle_state, player_filter)`：到**存活**塔矩形最近距离，无存活/全被过滤返回 `inf`（arena.py:65-84）。
- `refresh_tower_alive_cache(battle_state)`：每帧刷新 6 个布尔位（arena.py:55-63）；`step` 每帧调用一次（battle.py:2697）。
- `is_tower_tile(pos, battle_state)`：脚点落在任一**存活**塔矩形内（含边界 `<=`）（arena.py:131-143）。
- `get_tower_blocked_x_ranges(y, battle_state)`：给定 y 上的塔遮挡 x 区间（arena.py:193-206）。
- 塔存活判定 `_is_tower_alive` 按 **位置精确相等 + 归属方**在实体表里查（arena.py:121-129）；查不到 ⇒ 视为已死（arena.py:129）。
- 实体侧：`Entity.edge_distance_from(pos)` 对「`persistent=True` 且 `id<=6`」的实体返回矩形最近点距离，其余实体返回 `中心距 − collision_radius`；矩形惰性绑定一次（battle.py:152-183）。

### A.2.5 部署区 / 禁放区

**（a）`arena.get_deploy_zones` 的区间**（半开 `[x1,x2) × [y1,y2)`，判定见 arena.py:185）：

| 玩家 | 区间 | 展开后实际格 | 开启条件 | 依据 |
|---|---|---|---|---|
| 0 | `(0, 1, 18, 15)` | `x 0..17, y 1..14` | 恒有 | arena.py:152 |
| 0 | `(6, 0, 12, 6)` | `x 6..11, y 0..5` | 恒有（「己方王塔后 6 格，含边缘行」） | arena.py:153 |
| 0 | `(0, 17, 9, 20)` | `x 0..8, y 17..19` | `players[1].left_tower_hp <= 0` | arena.py:157 |
| 0 | `(9, 17, 18, 20)` | `x 9..17, y 17..19` | `players[1].right_tower_hp <= 0` | arena.py:160 |
| 1 | `(0, 17, 18, 31)` | `x 0..17, y 17..30` | 恒有 | arena.py:162 |
| 1 | `(6, 26, 12, 32)` | `x 6..11, y 26..31` | 恒有 | arena.py:163 |
| 1 | `(0, 11, 9, 15)` | `x 0..8, y 11..14` | `players[0].left_tower_hp <= 0` | arena.py:167 |
| 1 | `(9, 11, 18, 15)` | `x 9..17, y 11..14` | `players[0].right_tower_hp <= 0` | arena.py:170 |

后 4 行（破塔回推区）仅在 `battle_state` 参数被传入且非空时才会追加（arena.py:154、arena.py:164）；`arena.py:145` 的签名允许 `battle_state=None`，此时只返回前两行。

**（b）王塔身后 1 格宽建筑禁放带** `behind_king_zone`：蓝方 `(7.0, 0.0, 11.0, 1.0)`、红方 `(7.0, 31.0, 11.0, 32.0)`（arena.py:92-94）；判定 `x1 <= x < x2 and y1 <= y < y2`（arena.py:96-98）。**只禁建筑，不禁部队**（arena.py:180 由 `is_building` 参数控制；battle.py:2869 同一口径）。

**（c）`can_deploy_at` 的判定顺序**（arena.py:173-191）：越界或 `BLOCKED_TILES` → `False`（arena.py:176）→ 非法术且落在塔矩形内 → `False`（arena.py:177）→ 建筑且在王塔身后带 → `False`（arena.py:180）→ **法术且「非滚动法术」直接 `True`**（arena.py:181）→ `y==0 或 y==31` 且 `x∉[6,11]` → `False`（arena.py:182）→ 命中任一部署区 → `True`（arena.py:183-186）→ 末尾再对 `(player 0, y==0, 6<=x<=11)` 与 `(player 1, y==31, 6<=x<=11)` 补判（arena.py:187-190）。

> 事实：arena.py:181 调用的 `self._is_rolling_projectile_spell(...)` **在本文件内没有定义**（`arena.py` 共 209 行，全文无该方法）。当前唯一的 `can_deploy_at` 调用方传的是 `is_spell=False`（battle.py:2855），因此该分支未被触发；否则会抛 `AttributeError`。列入 A.7。

**（d）`battle.deploy_card` 内的另一套合法性判定**（battle.py:2865-2887，仅对 `card_info.type != 'spell'` 且卡名不是 `Miner` 时生效）：

| 玩家 | 规则（原样） | 依据 |
|---|---|---|
| 0 | `y <= 1.0` 且（`x <= 6.0` 或 `x > 12.0`）⇒ `False` | battle.py:2873 |
| 0 | `y >= 21.0` ⇒ `False` | battle.py:2874 |
| 0 | `15.0 <= y < 21.0`：`x <= 9` 时红左塔存活 ⇒ `False`；否则红右塔存活 ⇒ `False` | battle.py:2875-2879 |
| 1 | `y > 31.0` 且（`x <= 6.0` 或 `x > 12.0`）⇒ `False` | battle.py:2881 |
| 1 | `y <= 10` ⇒ `False` | battle.py:2882 |
| 1 | `10 < y <= 17.0`：`x <= 9` 时蓝左塔存活 ⇒ `False`；否则蓝右塔存活 ⇒ `False` | battle.py:2883-2887 |
| 双方 | 另有 `is_position_occupied_by_building(position, 0)`（塔矩形脚点 + 非塔建筑圆形） | battle.py:2867、battle.py:3072-3082 |
| 双方 | 建筑卡再受王塔身后带限制 | battle.py:2869-2870 |
| 仅 `BarbLog` | 额外要求 `arena.can_deploy_at(...is_spell=False)` 通过（即 BarbLog 只能用部队部署区） | battle.py:2854-2856 |

⇒ **两套判定并存**：`battle.deploy_card` 的部队/建筑路径**不检查** `BLOCKED_TILES`/河道可走性（只查建筑占位与上述 y/x 阈值）；`arena.can_deploy_at` 才查 `BLOCKED_TILES`。哪些调用方走哪条路径属 RL/掩码层问题，见 A.7。

### A.2.6 世界坐标 ↔ 本地网格（观测坐标）

观测张量形状 `(32, 18, 15)`（environment.py:42、environment.py:126），写入方式为 `obs[y][x] = obs_arr`，其中 `x, y = int(position.x), int(position.y)`（environment.py:146、environment.py:152）⇒ **网格下标 = 世界坐标向下取整**，格 `(x, y)` 覆盖 `[x, x+1) × [y, y+1)`。

镜像/翻转**只发生在观测编码层，且只对观测视角玩家 ≠ 0 时发生**：

| 变换 | 方向 | 公式 | 依据 |
|---|---|---|---|
| 世界 → 本地网格（视角 0） | 不翻转 | `x = int(px)`、`y = int(py)` | environment.py:146 |
| 世界 → 本地网格（视角 1，`observe(1)`） | 180° 点对称 | `x = 17 - int(px)`、`y = 31 - int(py)` | environment.py:147-149 |
| 本地网格 → 世界（玩家 0 出牌） | 格心 | `Position(x + 0.5, y + 0.5)` | environment.py:94 |
| 本地网格 → 世界（对手 = 玩家 1 出牌） | 格心 + 180° 反变换 | `Position(18 - (x + 0.5), 32 - (y + 0.5))` | environment.py:72 |

- 「谁对谁做」：`observe(1)` 把**全场实体**（含双方）坐标翻到玩家 1 的本地系，并把归属标签写成 `each.player != player_id_observe`（己方恒 0、敌方恒 1）（environment.py:132、environment.py:147-149）；`opponent_action` 把玩家 1 视角的 `(y, x)` 反变换回世界坐标再 `deploy_card(1, ...)`（environment.py:66-72）。
- **实体真实坐标永不被翻转**：翻转只发生在生成观测数组的那几行。
- 手牌用循环队列前 5 张（environment.py:154），而可出牌判据是前 4 张（player.py:37）。
- 寻路层另有一套**半格网格**（36×64，即每格 2×2 个寻路单元）：`calculate_building_cache` 以 `(x_cell in 0..35, y_cell in 0..63)` 建距离场（battle.py:3033-3036），并通过 `cell_to_position`（battle.py:4 导入）取格心。该变换的定义在 `pathfinding_heap.py`，不在本轮允许阅读的源码清单内 ⇒ 列入 A.7。

---

## A.3 实体体系

### A.3.1 类职责一览

| 类 | 基类 | 职责 | 依据 |
|---|---|---|---|
| `Entity` | — | 最基类：公共属性、索敌/换目标、伤害结算、buff 槽、投射物创建 | battle.py:13 |
| `BasicCharacter`（`core.py`） | — | 默认钩子载体：`on_spawn` 空、`on_tick` 刷新 `battle_state`、`on_death` 空、`on_attack` 结算攻击 | core.py:16-69 |
| `Troop` | `Entity` | 可移动实体：部署延迟、A* 行军、跳河、攻击、卡死自救、队形车道 | battle.py:759 |
| `Building` | `Entity` | 静态实体：不移动、可 `persistent=True`（6 座塔）、可 `lifetime` 自然衰减 | battle.py:1233 |
| `Projectile` | `Entity` | 弹道：追踪/直线、滚动、到达结算、溅射、生成链、觉醒命中钩子 | battle.py:1417 |
| `SpawnProjectile` | `Projectile` | 二段弹（数值表行直接驱动，**跳过 `Entity.__init__`**，恒 `invincible=True`） | battle.py:1656-1700 |
| `AreaEffect` | `Entity` | 瞬发/持续区域法术（Zap/Freeze/Heal/Rage/Tornado/Earthquake/Poison） | battle.py:1723 |
| `EvoZapZone` | `AreaEffect` | 觉醒 Zap 领域（进入即眩晕 + 次级 AOE） | battle.py:2328 |
| `EvoEffectZone` | `Entity` | 觉醒效果领域（DPS 脉冲 / 减速 / 吸引 / 友方增益） | battle.py:1946 |
| `DeathSlowZone` | `EvoEffectZone` | 亡语减速圈（单次瞬时） | battle.py:2184 |
| `IceGolemiteSnowZone` | `EvoEffectZone` | Hero 冰雪光环（按 `collision_radius <= 0.5` 分档冻结/减速） | battle.py:2484-2521 |
| `HealAuraZone` | `Entity` | 治疗光环（脉冲计数制，仅部队） | battle.py:2038 |
| `VinesSnareZone` | `Entity` | Vines 束缚领域（锁定 HP 最高 N 个 + 跳伤害 + 拽落） | battle.py:2091 |
| `GenericBomb` | `Entity` | 可编程定时炸弹（延迟后 AoE + 可选击退） | battle.py:1892 |
| `TimedExplosive` | `Entity` | 引信爆炸（读卡牌 `death_spawn_data`） | battle.py:2377 |
| `_ProjectileShim` / `_EffectShim` / `_BombShim` | — | 非卡牌实体的 `data` 兼容垫片（提供下游访问的最小属性集） | battle.py:1639 / 1707 / 1876 |

### A.3.2 公共属性（`Entity.__init__`，battle.py:14-99）

| 属性 | 初值 | 依据 |
|---|---|---|
| `id, position, player, card_name, battle_state` | 构造参数 | battle.py:16 |
| `data` | `Card(card_name)` | battle.py:17 |
| `level` | `self.data.level`（11-16） | battle.py:18 |
| `name` | `self.data.name` | battle.py:19 |
| `is_alive` | `True` | battle.py:22 |
| `attack_cooldown` | `hit_speed - load_time` | battle.py:23 |
| `speed` | `data.speed` | battle.py:24 |
| `hp` | `data.hp` | battle.py:25 |
| `shield_health` | `data.shield_health` | battle.py:26 |
| `target_id` | `None` | battle.py:27 |
| `targetable` / `invincible` | `True` / `False` | battle.py:32-33 |
| `jumping_across_river` | `False` | battle.py:37 |
| `speed_buff` / `speed_debuff` | `1.0` / `1.0` | battle.py:40-41 |
| `buff_time_remaining` / `debuff_time_remaining` | `0.0` / `0.0` | battle.py:42-43 |
| `hit_speed_mult` | `1.0` | battle.py:44 |
| `entity_holder` | `BasicCharacter(self)` 或 `eval("卡名")(self)` | battle.py:50-52 |
| `path` / `pending_damage` | `[]` / `[]` | battle.py:55、battle.py:57 |
| `ramp_target_id` / `ramp_timer` / `ramp_stage` | `None` / `0.0` / `0` | battle.py:60-62 |
| `freeze_timer` / `damage_reduction` / `damage_reduction_timer` | `0.0` 三者 | battle.py:65-67 |
| `regen_buffs` | `[]`（元素 `{'hps','time'}`） | battle.py:68 |
| `hook_pull` | `None` | battle.py:69 |
| `evo` / `_fortify_dr` / `last_attack_time` | `None` / `0.0` / `-999.0` | battle.py:71-73 |
| `ability_cd` / `ability_uses` | `0.0` / `0` | battle.py:74-75 |
| `hero_mode` | `False` | battle.py:77 |
| `attack_seq` / `attack_seq_mode` / `attack_seq_damages` / `attack_seq_stage` / `attack_seq_pending` / `attack_seq_hit_timer` | `None`/`None`/`[]`/`0`/`0`/`0.0` | battle.py:83-88 |

子类追加：

| 类 | 追加属性 | 依据 |
|---|---|---|
| `Troop` | `deploy_delay_remaining = data.deploy_time`、`path_blocked_counter`、`_stuck_check_pos/_time`、`_lane_offset`（由 `position._lane_offset` 带入）、`jumping_across_river`、`start_jumping_position`、`spawned`、`evo_hits`、`evo_extra_spawned` | battle.py:762-775 |
| `Building` | `deploy_delay_remaining = data.deploy_time`、`lifetime_elapsed`、`target_id`、`tower_active=False`、`persistent`、`_tower_rect=0`（哨兵） | battle.py:1236-1244 |
| `Projectile` | `target_position`、`initial_position`、`proj = data.projectile_data`、`rolling = bool(proj.roll_range)`、`homing`、`target`、`source`、`collision_radius`（法术取 `proj.radius`，否则 `0.3`）、`damage_dealt`、`damage_override` | battle.py:1420-1434 |

注：`Troop.path_blocked_counter`、`Troop.spawned`、`Building.lifetime_elapsed`、`BattleState.regen`（battle.py:2550）在 `battle.py` 内**只被赋值、未见读取**（本文件内 grep 结果）。

### A.3.3 实体注册表与 id 分配

- 注册表：`BattleState.entities: dict[int, Entity]`（battle.py:2542），键 = `int` id。
- id 计数器：`next_entity_id` 初值 `1`（battle.py:2549）。
- 6 座塔在 `__init__` 内按 `_spawn_entity` 顺序占用 id 1..6（battle.py:2556-2561）；因为 `_spawn_entity` 用计数器**覆盖**传入 id（battle.py:2606），实际映射为：

| id | 实体 | 归属 | 依据 |
|---|---|---|---|
| 1 | 红左公主塔 | 1 | battle.py:2556 |
| 2 | 红右公主塔 | 1 | battle.py:2557 |
| 3 | 蓝左公主塔 | 0 | battle.py:2558 |
| 4 | 蓝右公主塔 | 0 | battle.py:2559 |
| 5 | 红国王塔 | 1 | battle.py:2560 |
| 6 | 蓝国王塔 | 0 | battle.py:2561 |

- `_spawn_entity(entity)`（battle.py:2603-2608）：① `ensure_walkability(entity)` → ② `entity.battle_state = self` → ③ `entity.id = self.next_entity_id`（**覆盖**）→ ④ 入表 → ⑤ `next_entity_id += 1`。
- `_wrap(entity_data)`（battle.py:2610-2635）：按卡名分派构造 `Projectile`（长度 7）/`Entity`（法术）/`Building`（`buildings` 表，`persistent=False`）/`Troop`；并在此处给 Wild slot 的 Hero 卡套 `apply_hero_overlay`（battle.py:2633-2634）。
- `delayed_spawn(entity, delay)`（battle.py:2637-2641）：`delay` 为真则入 `schedule`（到点 `self.time+delay`），否则立即 `_wrap`+`_spawn_entity`。
- **id 消耗**：`_wrap` 先 `entity_data[0] = next_entity_id` 且自增（battle.py:2613-2614），随后 `_spawn_entity` 再覆盖并自增（battle.py:2606-2608）⇒ 每个经 `delayed_spawn` 的实体消耗 **2 个** id。
- 直接入表的两条旁路（不判 `ensure_walkability`、不覆写 id）：`Entity.create_projectile`（battle.py:738-746）与 `spawn_projectile_chain`（battle.py:2770-2773）。
- 移除：`step` 开头过滤 `value.is_alive or key <= 6`（battle.py:2691）⇒ 死亡的非塔实体在**下一帧开头**从表里消失；**6 座塔即使 `is_alive=False` 也永久保留在表内**（供按 id 取 HP/存活判定）。
- 其它侧表：`schedule`（延迟出兵队列，battle.py:2563）、`resurrect_queue`（觉醒临时复活队列，battle.py:2564）、`hero_windows`（Hero 条件窗，battle.py:2565）、`souls = [0, 0]`（battle.py:2571）、`building_positions`（非塔建筑占位列表，battle.py:2695）。

### A.3.4 出生（spawn）流程

1. **出牌入口** `deploy_card`（battle.py:2806）：镜像分支（battle.py:2808-2825）→ `MergeMaiden` 双费用形态分支（battle.py:2833-2849）→ 手牌/圣水/王塔存活性校验（player.py:36-39）→ 觉醒形态判定（battle.py:2861-2863）→ 位置合法性（A.2.5）→ 按 `card_info.type` 分派：
   - 法术：`Lightning` 特判（battle.py:2892-2895）、区域持续出兵（battle.py:2896-2911）、克隆（battle.py:2912-2924）、瞬发区域法术 `AreaEffect`/`EvoZapZone`（battle.py:2928-2939）、有弹道的法术（battle.py:2941-2994）；
   - 部队/建筑：`get_spawn_position` 得到一组落点（battle.py:2996）→ 打车道偏移标记（battle.py:3003-3005）→ 逐个 `delayed_spawn`（battle.py:3007-3009）→ 可选第二部队（battle.py:3013-3025）→ `_finish_deploy` 收尾（battle.py:2728-2749，扣费 + 卡序推进 + 觉醒出牌计数）。
2. **多单位落点** `get_spawn_position(card_info, position, player, offset_angle=True)`（battle.py:2524-2535）：`spawn_number == 1` 时返回原位置；否则按下标 `i` 取角 `2πi/spawn_number`，`player == 1` 时整组加 `π`，半径 `spawn_radius`（`summonRadius/1000`，card_utils.py:234）；`spawn_number == 2/3/4/6` 时按 `{2:0, 3:π/2, 4:π/4, 6:0}` 加固定偏角（battle.py:2528）。
3. **延迟出兵** 在 `step` 的固定阶段结算（battle.py:2709-2711）。
4. **亡语出兵** `_generic_death_spawn`（battle.py:276-323）：读 `death_spawn_data`；炸弹型（有 `deathDamage` 无 `hitpoints`）改生成 `TimedExplosive`（battle.py:298-302）；`SkeletonBalloon`/`SkeletonContainer` 走 7 个延迟 `Skeleton`（battle.py:303-310）；其余按 `deathSpawnCount` 出兵（battle.py:311-323）。有专属 `on_death` 的机制类与觉醒同名亡语会跳过（battle.py:292-295）。

### A.3.5 死亡（kill）流程

`take_damage` 内 `hp <= 0 and is_alive` 时调用 `die()`（battle.py:558-559）。`Entity.die()`（battle.py:200-229）依次：

1. `is_alive = False`（battle.py:202）；
2. `entity_holder.on_death()`（battle.py:203）；
3. `Troop`/`Building` 追加 `_evo_on_death()`（battle.py:204-205）；
4. `_generic_death_spawn()`（battle.py:207）；
5. `voodoo_curse` 存在时生成 `VoodooHog`（battle.py:209-218）；
6. 击杀者为本方觉醒 Pekka 且带 `onKilledDoneAction` 时按 `resurrectParameters[2]` 回血（battle.py:219-228）；
7. `battle_state.on_death(self)`（battle.py:229）。

`BattleState.on_death(entity)`（battle.py:3148-3167）：

- 实体 `name == 'King_PrincessTowers'` ⇒ 把**同方** `KingTower` 置 `tower_active = True`（battle.py:3149-3154）；
- 死的是 `Building` ⇒ `cache_fresh = False`（建筑距离场下一帧重算）（battle.py:3155）；
- 死的是 General Gerry ⇒ 其名下亡影立即 `die()`（battle.py:3158-3161）；
- 死的是 `Troop` 且非 `SkeletonKingSkeleton` 且未标 `_soul_excluded` ⇒ 双方灵魂计数各 `+1`（上限 10）（battle.py:3163-3167）。

死亡后**不立即从 `entities` 删除**，由下一帧 `step` 开头清理（battle.py:2691）；本体死亡时的亡语伤害/减速圈在 `take_damage` 内就地结算（battle.py:560-581）。

### A.3.6 等级与数值缩放（与实体构造相关）

- `Card.default_level` 类的全局默认等级 = `11`（card_utils.py:220）；`BattleState.__init__` 把它设为本次战斗的 `card_level`（battle.py:2540-2541）——源码注释声明该机制基于**单战斗串行**假设（card_utils.py:218-219），多战斗并行需显式传 `level`。
- `Card.__init__` 从数值表读取并换算：`hp`（card_utils.py:226）、`elixir = manaCost`（card_utils.py:227）、`collision_radius = collisionRadius/1000`（card_utils.py:239）、`hit_speed = hitSpeed/1000`（card_utils.py:240）、`load_time = loadTime/1000`（card_utils.py:241）、`speed = speed/50`（card_utils.py:242）、`range/sight_range = /1000`（card_utils.py:247-248）、`deploy_time = deployTime/1000`（card_utils.py:249）、`spawn_number/spawn_delay/spawn_radius`（card_utils.py:232-234）、`tower_damage_mult = 1 + crownTowerDamagePercent/100`（card_utils.py:291）。
- 逐级取值 `_value_at_level(arr, rarity, level, base)`：索引在数组内直接取值，越界按 `arr[-1] * 1.1**(li - len(arr) + 1)` 延续，数组为空按 `base * 1.1**(level-1)`（card_utils.py:196-207）；稀有度轴 `_rarity_level_index`：Common `level-1`、Rare `level-3`、Epic `level-6`、Legendary `level-9`、Champion `level-11`，未知按 Common（card_utils.py:182-193）。`level_scale(level) = 1.1 ** (level - 1)`（card_utils.py:210-214）。
- 幻影等级窗口：`step` 内镜像出兵时临时 `Card.default_level += 1`，处理完 `schedule` 后还原（battle.py:2706-2714）。
- 战斗中**可变**的全局态：`Card.default_level` 会被 `BattleState.__init__`（battle.py:2541）与镜像窗口（battle.py:2708/2713）改写。

---

## A.4 战斗主循环

### A.4.1 `step(dt)` 逐阶段顺序（严格按源码顺序）

| # | 阶段 | 行号 | 内容 |
|---|---|---|---|
| 1 | 终局短路 | battle.py:2653 | `if self.game_over: return`——终局后不再推进任何状态 |
| 2 | 塔血同步 | battle.py:2654 → battle.py:2643-2650 | 实体 HP 写回 `PlayerState`：蓝 `king←entities[6]`、`left←[3]`、`right←[4]`；红 `king←[5]`、`left←[1]`、`right←[2]` |
| 3 | 皇冠读取 | battle.py:2655-2656 | `p0 = players[0].get_crown_count()`、`p1 = players[1].get_crown_count()`（口径见 A.6） |
| 4 | 胜负判定（4 个分支） | battle.py:2659-2688 | 见 A.6；命中即 `game_over = True` 并 `return`（不加时、不推进 `time`） |
| 5 | 圣水回复 | battle.py:2689-2690 | `regenerate_elixir(dt, 2.8 if time<120 else 1.4 if time<240 else 2.8/3)`；上限 10（player.py:34），`elixir_per_second = 1/base_regen_time`（player.py:33） |
| 6 | 死亡实体清理 | battle.py:2691 | 保留 `is_alive` 或 `key <= 6` |
| 7 | 建筑占位表 + 塔存活缓存 | battle.py:2695-2697 | `building_positions` 仅收 `Building` 且 `id > 6`（塔不进圆形列表）；`arena.refresh_tower_alive_cache(self)` |
| 8 | 建筑距离场重算 | battle.py:2698-2700 | 仅当 `cache_fresh == False`；实现见 battle.py:3031-3052 |
| 9 | 实体更新（含索敌/移动/攻击/伤害） | battle.py:2701-2703 | 对 `list(self.entities.values())` 逐个 `entity.update(dt)` + `ensure_walkability(entity)` |
| 10 | 碰撞解算 | battle.py:2704 → battle.py:3084-3146 | 地面/空中两组两两推挤（圆形），塔-部队按矩形推出并全位移给部队 |
| 11 | 镜像等级窗口 | battle.py:2706-2708 | `_mirror_level_bonus` 存在时临时抬高 `Card.default_level` |
| 12 | 延迟出兵结算 | battle.py:2709-2711 | 到点项 `_spawn_entity(_wrap(entity))`，随后过滤未到点项 |
| 13 | 镜像等级还原 | battle.py:2712-2714 | 还原并清零 `_mirror_level_bonus` |
| 14 | 觉醒临时复活 | battle.py:2717-2724 | `resurrect_queue` 到点→`Troop(..., evolved=True)` 并写 `hp`/`_evo_temp_lifetime` |
| 15 | 时间推进 | battle.py:2725-2726 | `self.time += dt`；`self.tick += 1` |

**说明**：

- **引擎内没有独立的「AI/决策」阶段**。决策在 `CREnv.step` 中于推进物理**之前**完成：玩家 0 的 `deploy_card(0, card, Position(x+0.5, y+0.5))`（environment.py:91-94），随后 `opponent_action()` 让对手策略出一次牌（environment.py:96、environment.py:65-73）。
- **也没有独立的「攻击/伤害结算」阶段**：一次攻击与伤害全部发生在阶段 9 的 `entity.update(dt)` 内部（`Troop.update` battle.py:1225-1229；`Building.update` battle.py:1302-1308；`BasicCharacter.on_attack` core.py:30-69）。
- **「计时」阶段**：`time`/`tick` 在阶段 15 一次性推进；其余计时（冷却、buff、寿命、引信）都在各自实体的 `update` 内以 `dt` 递减。死亡自身不消耗额外时钟（`die()` 同步执行）。
- `tick` 的用途：地面单位索敌目标在视野内时，`self.battle_state.tick % 10 == 0` 触发路径重算（battle.py:1181）。
- 阶段 2 在阶段 6/9 之前 ⇒ 阶段 4 的胜负判定读到的是**上一帧结束时的 HP**（本帧的伤害尚未结算）。

### A.4.2 `dt` 取值来源与固定步长

| 事实 | 值 | 依据 |
|---|---|---|
| `step` 的 `dt` 唯一来源 | `self.battle.step(1/60)` | environment.py:102 |
| 每次决策最多推进 | `for i in range(30)`，且每帧先判 `game_over` 提前 `break` | environment.py:98-100 |
| 固定步长含义 | 30 × 1/60 = **0.5 秒/决策** | environment.py:80-81、environment.py:98 |
| 子步乘数 | `for j in range(int(self.speed))`（`speed` 默认 1.0；`speed<1` 时 `int()` 截断为 0） | environment.py:101、environment.py:34 |

⇒ 引擎侧是**固定步长 1/60 s** 的离散模拟；`dt` 不是自适应值。

---

## A.5 伤害与命中

### A.5.1 伤害唯一入口 `Entity.take_damage(amount, delayed=False, source=None, pierce_invincible=False)`（battle.py:507）

按代码顺序：

| # | 判定 | 行为 | 依据 |
|---|---|---|---|
| 1 | 首击面纱 | `_evo_veil_time > 0` 且未用过 ⇒ 记 `_evo_veil_used=True`、`_evo_veil_timer=_evo_veil_time`、`invincible=True`，**本次伤害整段丢弃并 return** | battle.py:511-515 |
| 2 | 无敌 | `invincible and not pierce_invincible` ⇒ return | battle.py:516 |
| 3 | 伤害来源记录 | `source is not None` ⇒ `last_hit_by = source` | battle.py:517 |
| 4 | 受击钩子 | `entity_holder.on_take_damage(amount, source)` 返回真值 ⇒ 短路（格挡/免伤） | battle.py:523-525 |
| 5 | 减伤 | `dr = max(damage_reduction if damage_reduction_timer>0 else 0.0, _fortify_dr)`；`amount *= (1.0 - dr)` | battle.py:526-529 |
| 6 | 落地受击（RoyalHogs） | `_evo_landing` 存在 ⇒ 取消落地态、`is_air_unit=False`、就地 AoE（`land['radius']`, `land['damage']`, 空地双打） | battle.py:530-536 |
| 7 | 延迟结算 | `delayed=True` ⇒ 仅 `pending_damage.append(amount)` 后 return | battle.py:537-539 |
| 8 | 护盾优先 | `shield_health` 真值 ⇒ 只扣护盾 `max(0, shield - amount)`；否则 `hp -= amount` | battle.py:540-542 |
| 9 | 破盾副作用 | 破盾瞬间重置蓄能（`ramp_stage/timer/target_id`）并按 `evo.shieldLostActionData` 结算破盾 AoE（默认半径 `3000/1000`） | battle.py:544-556 |
| 10 | 死亡 | `hp <= 0 and is_alive` ⇒ `die()`；`data.death_damage` 存在 ⇒ 以 `death_damage_radius`（缺省 `2.0`）结算空地双打亡语 AoE；`death_area_effect` 存在 ⇒ 生成 `DeathSlowZone` | battle.py:558-581 |
| 11 | 受击后钩子 | 仍存活且 `entity_holder.on_damaged` 存在 ⇒ 调用（不短路伤害） | battle.py:583-584 |

**延迟伤害的重放时机**：`Entity.update` 里 `for pending_damage in self.pending_damage: self.take_damage(pending_damage, delayed=False)`，随后清空（battle.py:405-407）。注意 `Projectile.update`（battle.py:1548-1594）**不调用** `Entity.update`，因此弹道实体自身不重放 `pending_damage`、也不衰减 buff 计时。

### A.5.2 护盾 / 减伤 / 治疗

| 机制 | 规则 | 依据 |
|---|---|---|
| 护盾 | `shield_health` 优先吸收；`data.shield_health` 初值来自 `shieldHitpoints` | battle.py:26、battle.py:540-542、card_utils.py:272 |
| 减伤（buff 槽） | `apply_buff(damage_reduction=...)` 取 `max` 并给 `damage_reduction_timer` | battle.py:146-148 |
| 减伤（fortify 独立槽） | 觉醒 `buffWhenNotAttackingData.damageReduction/100`；「脱战」= 距 `last_attack_time` 超过 `1.0` 秒 | battle.py:397-403 |
| 减伤生效 | 两槽取 `max` 后 `amount *= (1-dr)` | battle.py:526-529 |
| 治疗（导槽） | `regen_buffs` 每项 `{'hps','time'}`，逐帧 `hp = min(_cap, hp + hps*dt)`；`_cap = data.hp`，觉醒 `allowedOverHealPerc` 存在时 `_cap = data.hp*(1+perc/100)` | battle.py:374-388 |
| 治疗（光环） | `HealAuraZone`：`ticks=4`、`interval=0.25`、只疗 `Troop`、排除 `exclude_id`、上限 `data.hp` | battle.py:2043-2055、battle.py:2078-2088 |
| 治疗（区域法术） | `AreaEffect` `buff_name == 'Heal'`：`heal={'hps': heal_per_tick/tick, 'time': tick}` | battle.py:1812-1814 |
| 冻结/眩晕 | `apply_buff(stun=s)`：突进态免控（battle.py:106-109）；`freeze_timer = max(...)`、`attack_cooldown = max(attack_cooldown, hit_speed)`、重置蓄能与冲锋、可选重索敌 | battle.py:106-126 |

### A.5.3 攻击间隔与首次攻击

| 项 | 规则 | 依据 |
|---|---|---|
| 初始冷却（首次攻击延迟） | `attack_cooldown = data.hit_speed - data.load_time` | battle.py:23（`load_time = loadTime/1000`，card_utils.py:241） |
| 一次攻击后的冷却 | `attack_cooldown = data.hit_speed`（`on_attack` 末尾） | core.py:60 |
| 冷却递减（移动中） | `attack_cooldown = max(hit_speed - load_time, attack_cooldown - dt*speed_buff*speed_debuff*hit_speed_mult*hit_speed_debuff)` | battle.py:1224 |
| 冷却递减（站定待攻） | `attack_cooldown -= dt * speed_buff * speed_debuff * hit_speed_mult * hit_speed_debuff` | battle.py:1229 |
| 建筑冷却递减 | 同乘子口径 | battle.py:1297-1298 |
| 可攻击条件（部队） | 目标在攻击范围内且 `attack_cooldown <= 0` ⇒ `entity_holder.on_attack(current_target)` | battle.py:1225-1227 |
| 可攻击条件（建筑） | `target and in_attack_range(target) and attack_cooldown <= 0` | battle.py:1302 |
| 部署延迟（首次可行动时刻） | `deploy_delay_remaining = data.deploy_time`；`Troop.update` 在其 >0 时递减并 `return`（`Miner` 例外：先 `super().update(dt)` 再判延迟） | battle.py:762、battle.py:1113-1115、battle.py:1111-1121 |
| 部署延迟（建筑） | 同字段；`Building.update` 在 >0 时递减并 `return` | battle.py:1236、battle.py:1283-1285 |
| 建筑寿命衰减 | `lifetime > 0 且 not persistent` ⇒ `take_damage(hp/lifetime*dt)` | battle.py:1293-1296 |
| 攻速/移速 buff 影响 | `speed_buff/speed_debuff` 与 `hit_speed_mult/hit_speed_debuff` 都乘进冷却递减 | battle.py:1224、battle.py:1229 |
| 攻击结算主体 | `BasicCharacter.on_attack`：`last_attack_time = battle_state.time`；有 `damage` 或 `attack_seq` ⇒ 按 `area_damage_radius` 分派溅射/单体；`damage==0` 且有弹道 ⇒ `create_projectile` | core.py:30-60 |

**对塔伤害口径（`BasicCharacter.on_attack`）**：

- 溅射分支：以**自身**为中心 `deal_area_damage`；若主目标不在溅射圈内，额外补一记单体主伤，塔类再乘 `tower_damage_mult`（core.py:36-51）；圈内判定 `target.edge_distance_from(center) < radius`（core.py:25-28）。
- 单体分支：目标名含 `'King'` 或 `'PrincessTower'` ⇒ `take_damage(damage * tower_damage_mult)`，否则原伤害（core.py:53-56）。
- 弹道分支：`damage==0` 且 `data.projectiles` ⇒ `create_projectile(current_target)`（core.py:57-59）；投射物伤害另有 `Projectile._deal_splash_damage` 的 `"King" in name` 折扣（battle.py:1614）。
- `kamikaze` ⇒ 攻击后自毁（core.py:68-69）。

### A.5.4 投射物飞行与命中

| 项 | 规则 | 依据 |
|---|---|---|
| 追踪目标点 | `homing=True` 时逐帧追 `self.target.position`，否则飞向发射瞬间的 `target_position` | battle.py:1588 |
| 到达判定 | 到目标点距离 `<= proj.speed * dt` ⇒ `_on_arrive()` 且 `is_alive = False`；否则按 `speed*dt` 直线前进 | battle.py:1590-1594、battle.py:1630-1636 |
| 滚动弹 | `rolling = bool(proj.roll_range)`；滚出 `roll_range` 即 `_chain` + 消亡；滚动途中对 `type(each).__name__` 不在排除集、非空中、非同方且 `中心距 < each.collision_radius + proj.radius` 的实体逐个伤害并推挤（`pushback`），同一目标只打一次（`damage_dealt`） | battle.py:1423、battle.py:1551-1586 |
| 点到命中（无 radius） | 对 `target.take_damage(self._damage(), source=self.source)`；有 `buff_time` 时给 `speed_debuff`，可挂 `voodoo_curse` | battle.py:1453-1473 |
| 溅射命中（有 radius） | 见下 | battle.py:1474-1475、battle.py:1596-1628 |
| 到达后生成链 | `_chain(impact)`：`proj.spawn_projectile` ⇒ `spawn_projectile_chain`，`proj.spawn_characters` ⇒ `spawn_arrival_troops` | battle.py:1476-1477、battle.py:1534-1541 |
| 觉醒命中钩子 | `_evo_impact`：链电 / 二段爆炸 / 效果领域 | battle.py:1478、battle.py:1480-1532 |

`Projectile._deal_splash_damage`（battle.py:1596-1628）的顺序：

1. 若半径内存在敌方 `deflect_active`（Monk 禅定）⇒ 整个法术反弹至最近敌方公主塔并 `return`（battle.py:1599-1604）；
2. 逐实体过滤：`invincible` 跳过、同方跳过、死亡跳过、按 `proj.hits_air/hits_ground` 过滤（battle.py:1605-1609）；
3. 命中判据 `entity.position.distance_to(self.target_position) <= proj.radius + entity.data.collision_radius`（battle.py:1612）；
4. 伤害 `base if "King" not in entity.name else round(base * proj.crown_tower_percent)`（battle.py:1614）；
5. `proj.pushback` 存在且目标是 `Troop` ⇒ 沿背离落点方向位移 `min(pushback, 距离)`（battle.py:1617-1622）；
6. `proj.buff_time` ⇒ 施加 `speed_debuff`（`min(1 + speedMultiplier/100, 现值)`）与 `debuff_time_remaining`，可挂 5 秒 `voodoo_curse`（battle.py:1623-1628）。

`crown_tower_percent` 定义：`(crownTowerDamagePercent + 100)/100`，`TowerPrincessProjectile` 特判为 `25/122`（card_utils.py:460-465）。

### A.5.5 索敌与射程几何

| 函数 | 规则 | 依据 |
|---|---|---|
| `in_attack_range(target)` | `dist = target.edge_distance_from(self.position)`；`'PrincessTower' in target.name` 时 `bonus = 0.5`（battle.py:588-591）；`_range_override` 存在则 `dist <= 覆盖值 + bonus`（battle.py:595-597）；`min_range` 存在且 `dist < min_range` ⇒ `False`（battle.py:599-600）；狙击临时射程命中 ⇒ `True`（battle.py:602-603）；否则 `dist <= data.range + bonus`（battle.py:604） |
| `in_sight_range(target)` | 同 `bonus` 口径；狙击射程同步扩展；否则 `dist <= data.sight_range + bonus` | battle.py:605-616 |
| `get_nearest_target` | 遍历 `Troop`/`Building`、存活、非同方、`targetable`、`min_range` 过滤、空地能力过滤、`in_sight_range` 过滤；建筑与部队分桶，`target_only_buildings` / 已在攻击范围 / 否则只看部队，最后按距离排序取最近 | battle.py:618-649 |
| 无目标兜底 | 从 id 1..6 里挑**同侧**最近的敌方公主塔（`(tower.x - width/2) * (self.x - width/2) >= 0`）；同侧无存活则取 `best_any`；距离用 `edge_distance_from` | battle.py:700-727 |
| `_should_switch_target` | `target_only_buildings` 时拒绝非建筑；建筑遇进入攻击范围的部队立即转火；当前目标已在攻击范围则不换；否则比距离 | battle.py:651-670 |
| `update_current_target` | 目标死亡/出视野即清空（塔目标不清路径）；再取最近目标并决定是否切换 | battle.py:672-727 |
| Hero 嘲讽覆盖 | `_taunt_until > battle_state.time` 时强制锁定嘲讽者 | battle.py:729-736 |
| `deal_area_damage` | 逐实体：死亡/同方/`invincible` 跳过；`"King" in name` ⇒ `amount * crown_tower_damage_percent`；按 `attack_air`/`attack_ground` 与实体 `is_air_unit` 过滤；`dist = entity.edge_distance_from(position)`，**严格 `dist < range`** 才命中 | battle.py:3250-3265 |
| 塔的 edge 距离 | `persistent and id<=6` ⇒ 矩形最近点距离；否则 `中心距 - collision_radius` | battle.py:152-168 |

---

## A.6 皇家塔与胜负

### A.6.1 塔的 id / 数量 / 血量

- 数量：每方 3 座（2 公主塔 + 1 国王塔），全场 6 座，id 1..6（battle.py:2556-2561，映射见 A.3.3）。
- 塔是 `Building(persistent=True)` 实例——第 5 个位置参数为 `True`（battle.py:2556-2561，形参见 battle.py:1234）。
- 公主塔卡名由 `PlayerState.tower_troop` 决定：默认 `'King_PrincessTowers'`（battle.py:2554-2555，`tower_troop` 默认 `None`，player.py:15）；合法值 `King_CannonTowers / King_KnifeTowers / King_ChefTowers`（player.py:24）。
- 国王塔卡名固定 `'KingTower'`（battle.py:2560-2561）。

**血量**（本轮可确证的来源）：

| 塔 | 数值 | 依据 |
|---|---|---|
| 公主塔 / 国王塔 `PlayerState` 默认三塔血 | `(4824, 3052, 3052)`（顺序：king, left, right） | player.py:6、player.py:10 |
| 国王塔基值 | `hitpoints = 2100`（硬编码表） | card_utils.py:40 |
| 国王塔其它硬编码数值 | `hitSpeed 1000`、`damage 109`、`sightRange 7000`、`range 7000`、`collisionRadius 1400`、`deployTime 3300`、`loadTime 700`、弹道 `KingProjectile speed 600 / damage 109` | card_utils.py:41-53 |
| 逐级取值路径 | `Building`/`troop` 分支经 `_value_at_level(row['hitpoints_per_level'], rarity, level, base)` 取 HP | card_utils.py:366-367、card_utils.py:390-397 |
| lv11 解析结果 | `KingTower hp=4824 / dmg=109 / hit_speed=1.0 / deploy_time=3.3 / range=7.0 / sight=7.0 / collision=1.4`；`King_PrincessTowers hp=3052 / dmg=0 / hit_speed=0.8 / deploy_time=0.0 / range=7.5 / sight=7.5 / collision=1.0` | 由 card_utils.py:222-345 管线在 `level=11` 下解析（与 player.py:6 的硬编码默认值一致） |

⇒ `PlayerState` 的三塔血字段在**首次 `step`**（阶段 2，battle.py:2654）之前是硬编码默认值，之后每个 `step` 都被实体真实 HP 覆盖（battle.py:2643-2650）。

### A.6.2 国王塔激活（`tower_active`）

| 触发 | 条件 | 依据 |
|---|---|---|
| 自身受击 | `Building.take_damage` 内 `data.name == 'KingTower'` 且未激活 ⇒ 置 `True` | battle.py:1266-1271 |
| 同方公主塔阵亡 | `on_death` 内 `entity.name == 'King_PrincessTowers'` ⇒ 把同方 `KingTower` 置 `True` | battle.py:3148-3154 |

未激活的国王塔在 `Building.update` 开头直接 `return`（不索敌、不攻击）（battle.py:1282）。

> 源码事实：第二个触发条件比对的是实体 `name == 'King_PrincessTowers'`（battle.py:3149），而塔兵为 `King_CannonTowers/King_KnifeTowers/King_ChefTowers` 时实体 name 是卡名本身（`Card.name = data['name']`，card_utils.py:228；塔兵卡名与 `statCharacterData` 挂载见 card_utils.py:57-61）⇒ 这两种情形下公主塔阵亡**不会**经该路径激活国王塔。是否为设计意图无法从源码判定，列入 A.7。

### A.6.3 塔被摧毁后的效果

1. **皇冠/胜负计数**：`get_crown_count` 把该玩家**自己被摧毁的塔**计入（own HP ≤ 0）；国王塔 ≤ 0 直接返回 3（player.py:54-58）。
2. **解除部署区封锁**：`arena.get_deploy_zones` 依据 `battle_state.players[对方].left/right_tower_hp <= 0` 追加敌半场 4 格回推区（arena.py:154-170）；`battle.deploy_card` 另有一份 `left_tower_hp/right_tower_hp > 0` 判定（battle.py:2872-2887）。
3. **退出占位/索敌**：每帧 `refresh_tower_alive_cache` 后，`tower_rect_dist` 跳过死塔（arena.py:79-80、battle.py:2697）；`is_tower_tile` 与 `get_tower_blocked_x_ranges` 同样跳过死塔（arena.py:135-138、arena.py:198-201）；`_push_troop_out_of_tower` 对死塔不再推出（battle.py:3119）。
4. **激活己方国王塔**：见 A.6.2。
5. **建筑距离场重算**：`on_death` 对 `Building` 置 `cache_fresh = False`（battle.py:3155）。
6. 塔死后的**实体仍留在 `entities`**（id ≤ 6），因此读 `entities[1..6]` 恒定安全（battle.py:2691）。

### A.6.4 `game_over` / `winner` 的全部置位点

`winner` 初值 `None`、`game_over` 初值 `False`（battle.py:2547-2548）；**全部置位都只发生在 `step` 的阶段 4**（本文件内 grep `game_over|winner` 无其它写入点）：

| # | 条件 | 结果 | 依据 |
|---|---|---|---|
| 1 | `p0 == 3`（玩家 0 三塔全失） | `game_over=True`，`winner=1`，`return` | battle.py:2659-2662 |
| 2 | `p1 == 3` | `game_over=True`，`winner=0`，`return` | battle.py:2663-2666 |
| 3 | `300 > time >= 180` 且 `p0 > p1` | `game_over=True`，`winner=1`，`return` | battle.py:2667-2671 |
| 4 | `300 > time >= 180` 且 `p0 < p1` | `game_over=True`，`winner=0`，`return` | battle.py:2672-2675 |
| 5 | `time >= 300` 且 `_m0 > _m1` | `winner=0` | battle.py:2685-2686 |
| 6 | `time >= 300` 且 `_m1 > _m0` | `winner=1` | battle.py:2687-2688 |
| — | `time >= 300` 且 `_m0 == _m1` | 只 `game_over=True`，`winner` **保持 `None`（平局）** | battle.py:2680-2688 |

其中：

- `p0/p1` 是**自己被摧毁的塔数**（A.6.3 第 1 条）⇒ `p0==3` 表示玩家 0 全失、玩家 1 获胜（battle.py:2659-2662），`p0 > p1` 同理归给玩家 1（battle.py:2668-2670）。
- `time >= 300` 分支（加时硬顶，battle.py:2676-2688）：`_m0 = min(entities[i].hp / entities[i].data.hp for i in (3,4,6) if is_alive)`（玩家 0 存活塔的最低**血量百分比**）、`_m1` 对应 `(1,2,5)`（battle.py:2681-2684）；注释说明「双方存活塔中血量百分比最低者输，完全相等才平局」（battle.py:2677-2679）。
- **加时/平局**：`180 <= time < 300` 时若皇冠相等则**不结束**（无 else 分支）；只有到 `time >= 300` 才可能以平局收场（`winner` 保持 `None`）。
- `CREnv.step` 侧的后果：`winner == 0` 加 `+10`，否则（含 `winner is None` 的平局）减 `10`（environment.py:112-118）；返回 `terminated = truncated = battle.game_over`（environment.py:121）。

### A.6.5 皇冠计数来源

`PlayerState.get_crown_count()`（player.py:54-58）读的是 `PlayerState` 的三个 HP 字段：`king_tower_hp <= 0 ⇒ 3`，否则 `int(left_tower_hp <= 0) + int(right_tower_hp <= 0)`。这些字段由 `BattleState.update_player_hp()` 在每个 `step` 开头从实体同步（battle.py:2643-2650）；`PlayerState` 自身的初值是构造参数（environment.py:57-58 用默认 `tower_hps`，player.py:6）。

---

## A.7 本部分待确认清单

| # | 条目 | 无法确认的原因 | 要确认需要什么 |
|---|---|---|---|
| 1 | 半格寻路网格的坐标变换（`position_to_cell` / `cell_to_position` 的具体公式） | 定义在 `pathfinding_heap.py`，不在本轮允许阅读的源码清单内；`battle.py` 只显示网格为 36×64（battle.py:3033-3036），`battle.py:4` 仅导入函数名 | 允许阅读 `pathfinding_heap.py`（或让作者给出该文件的行号引用）；可先看 `docs/_survey/parts/G037.md` 的记载再回源码核验 |
| 2 | `TileGrid.tile_size = 100.0` 的用途 | 该常量在 `arena.py`/`battle.py` 内定义后无任何读取点（本目录 `*.py` grep 仅命中 arena.py:10） | 全仓（含渲染层 `new_visualization.py`）搜 `tile_size` |
| 3 | `arena.can_deploy_at(is_spell=True, spell_obj=...)` 的行为 | arena.py:181 调用的 `self._is_rolling_projectile_spell(...)` 在 `arena.py`（209 行）内未定义，全仓亦未找到 Python 定义；当前唯一调用方传 `is_spell=False`（battle.py:2855），该分支从未执行 | 查 `spell_module.py:261` 的调用参数（不在本轮清单）；或让作者确认该方法是否被删除 |
| 4 | 部署合法性由哪一层兜底 | `battle.deploy_card` 的部队/建筑路径不检查 `BLOCKED_TILES`/河道（battle.py:2865-2887），只有 `arena.can_deploy_at` 检查（arena.py:176）；RL 掩码层是否补齐无法确认 | 读 RL 侧掩码实现（`rl/action_mask.py` / `rl/env_wrapper.py` 一类），不在本轮清单 |
| 5 | 「bottom half / top half」注释与 y 数值方向的对应 | arena.py:150/arena.py:161 注释称玩家 0 为 bottom、玩家 1 为 top，而数值上玩家 0 的区间是 `y=1..14`（小 y 侧，arena.py:152）；坐标是否在渲染层翻转需 `new_visualization.py` | 允许阅读 `new_visualization.py`（或给出其坐标变换行号） |
| 6 | 塔兵为 Cannon/Knife/Chef 时国王塔激活语义 | 只有 `name == 'King_PrincessTowers'` 的公主塔阵亡才激活同方国王塔（battle.py:3149-3154），塔兵换卡后 name 变为该卡卡名（card_utils.py:57-61、card_utils.py:228）⇒ 该路径不触发 | 官方机制口径（源码内无注释说明是否为有意），或补一条针对塔兵卡名的分支/测试 |
| 7 | 180 秒皇冠判定的规则来源 | 源码只有数值 `300 > self.time >= 180`（battle.py:2667）与其后注释（battle.py:2677-2679），未标注对应真实 CR 规则或数据出处 | 作者/设计文档口径，或对局录像对拍 |
| 8 | 对手出牌失败是否被吞掉 | `opponent_action` 丢弃 `deploy_card` 返回值（environment.py:72），失败（费用不足/位置非法）静默忽略 | 确认 RL/脚本对手层是否有独立合法性检查（不在本轮清单） |
| 9 | 国王塔逐级 HP 数组与「4824」的确切出处 | 本轮只能确证：`king_tower_stats.hitpoints = 2100`（card_utils.py:40）+ 经 `buildings` 表按稀有度轴取值（card_utils.py:366-367）+ 与 `player.py:6` 硬编码 `(4824, 3052, 3052)` 一致；`cards_stats_building.json` 是数据文件，非 `.py` 源码 | 允许直接读 `gamedata.json` / `cards_stats_building.json` 的 `KingTower`、`PrincessTower` 行，或给出引用该表的源码行 |
| 10 | 引擎是否被 RL 侧以外的方式驱动（决策帧、动作槽位语义） | 本文只覆盖 `CREnv`（5 槽手牌、格心动作，environment.py:94、environment.py:154）；训练用的 `rl/` 环境不在本轮清单 | 读 `rl/env_wrapper.py`、`rl/action_mask.py`（不在本轮清单） |
| 11 | `PlayerState` 默认塔血与 `card_level` 不一致的窗口 | `tower_hps` 默认 `(4824, 3052, 3052)` 是 lv11 值（player.py:6），而 `BattleState(card_level=...)` 可为 11-16；在首次 `step` 覆盖之前（battle.py:2654），`can_play_card` 读的是默认值（player.py:39） | 确认是否存在「首次 step 前读 `*_tower_hp`」的调用路径（`rl/` 不在本轮清单） |
| 12 | 多战斗并行的等级全局态安全 | `Card.default_level` 是类变量，由 `BattleState.__init__` 与镜像窗口改写（card_utils.py:220、battle.py:2541、battle.py:2706-2714）；源码注释声明基于单战斗串行假设（card_utils.py:218-219），并行场景的后果未在源码中给出 | 官方并行/多进程方案的说明，或一条并发回归测试 |

---

PART_A_OK sections=7 citations=451
