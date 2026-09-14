# 游戏引擎文档

本文档说明**这个项目的游戏引擎是怎么实现的**：地图与坐标、实体与战斗主循环、皇家塔与胜负判定、
寻路、索敌、卡牌机制与法术、以及部署合法性判定。

**资料来源**：`src/clasher_new/` 下的引擎源码（`battle.py`、`arena.py`、`core.py`、`player.py`、
`pathfinding*.py`、`card_mechanics.py`、`spell_module.py`、`threat_calc.py`、`card_utils.py`、
`rl/action_mask.py` 等）+ 逐文件函数级分析素材。**不引用**任何设计文档、计划文件或外部资料。

## 目录

- [0. 文档说明与方法](#0-文档说明与方法)
- [游戏引擎文档 · 第 A 部分：地图、实体与战斗主循环](#游戏引擎文档--第-a-部分地图实体与战斗主循环)
  - [A.1 引擎概览](#a1-引擎概览)
  - [A.2 地图与坐标系统](#a2-地图与坐标系统)
  - [A.3 实体体系](#a3-实体体系)
  - [A.4 战斗主循环](#a4-战斗主循环)
  - [A.5 伤害与命中](#a5-伤害与命中)
  - [A.6 皇家塔与胜负](#a6-皇家塔与胜负)
  - [A.7 本部分待确认清单](#a7-本部分待确认清单)
- [游戏引擎文档 · 第 B 部分：寻路、索敌、卡牌机制与部署合法性](#游戏引擎文档--第-b-部分寻路索敌卡牌机制与部署合法性)
  - [B.1 寻路总览](#b1-寻路总览)
  - [B.2 网格与通行性](#b2-网格与通行性)
  - [B.3 `pathfinding_heap.py`（引擎实际使用的那一套）](#b3-pathfinding_heappy引擎实际使用的那一套)
  - [B.4 `pathfinding.py`（未使用的旧实现，与 B.3 的关系）](#b4-pathfindingpy未使用的旧实现与-b3-的关系)
  - [B.5 索敌与目标选择](#b5-索敌与目标选择)
  - [B.6 卡牌机制分类（`card_mechanics.py`）](#b6-卡牌机制分类card_mechanicspy)
  - [B.7 法术与区域效果](#b7-法术与区域效果)
  - [B.8 部署合法性](#b8-部署合法性)
  - [B.9 本部分待确认清单](#b9-本部分待确认清单)
- [附录 A：待确认事项汇总](#附录-a待确认事项汇总)
- [附录 B：生成方式与可复现性](#附录-b生成方式与可复现性)

---

## 0. 文档说明与方法

### 0.1 覆盖范围

| 项 | 内容 |
|---|---|
| 地图 | §A.2（场地尺寸、坐标系统、桥/河/塔位、部署区） |
| 寻路 | §B.1–B.4（两套寻路实现、通行性网格、A* 代价与启发式） |
| 索敌 | §B.5（视野/射程、目标选择、锁定与重选、威胁评估） |
| 战斗与实体 | §A.3–A.5（实体体系、`step(dt)` 逐阶段、伤害与命中） |
| 胜负判定 | §A.6（塔血量、摧毁效果、`winner`/`game_over` 全部置位点） |
| 卡牌与法术 | §B.6–B.7（机制类组织、法术落点与命中公式） |
| 部署合法性 | §B.8（`legal_cells`/`_position_legal` 与引擎侧校验的一致性） |
| 配套文档 | 《项目内容全解文档》`docs/full_code_reference.md`、《训练方法文档》`docs/training_method.md` |
| 一致性取证 | `docs/mask_vs_engine_reconcile_2026-09-14.log` —— 掩码 `legal_cells` 与引擎 `deploy_card` 的**逐格双向对账**（18×32×2 方 × 24 张卡），脚本 `scripts/_mask_vs_engine_reconcile.py`（只读） |

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

**怎么找函数**：按 `Ctrl+F` 搜函数名或常量名；引擎常量（格子尺寸、塔血量、代价表）都在 §A.2/§B.3/§B.5 的表格里。

---

## 游戏引擎文档 · 第 A 部分：地图、实体与战斗主循环

> **依据范围**：`src/clasher_new/battle.py`（3267 行）、`arena.py`、`core.py`、`player.py`、`environment.py`、`card_utils.py`（仅地图/坐标/塔相关部分）。
> 行内标注格式为 `（文件名:行号）`；所有数值原样抄自源码，未做四舍五入或换算（源码自身的 `/1000`、`/50` 等换算照抄）。
> 无法从上述源码确认的项一律集中在 **A.7**，正文不猜测。
> 计数口径：文末 `citations` = 全文「文件名.py:行号」形式的来源引用总数（含各表格的「依据」列；同一括号内并列两条引用时计 2 条）。

---

### A.1 引擎概览

#### A.1.1 `battle.py` 的位置

`battle.py` 是**战斗模拟的唯一实现文件**：它同时定义实体类族（`Entity` 及其子类）、机制钩子载体（`BasicCharacter`，实现在 `core.py`）、以及战斗状态机 `BattleState`。地图几何在 `arena.py`，坐标原语在 `core.py`，玩家/圣水/卡序在 `player.py`，Gym 包装在 `environment.py`，卡牌数值解析在 `card_utils.py`。

#### A.1.2 对外入口

| 入口 | 位置 | 说明 |
|---|---|--- |
| `BattleState.__init__(player_0, player_1, card_level=None)` | battle.py:2539 | 构造即铺设 6 座塔（battle.py:2556-2561） |
| `BattleState.step(dt)` | battle.py:2652 | 单帧推进（A.4） |
| `BattleState.deploy_card(player_id, card_name, position, _from_mirror=False)` | battle.py:2806 | 出牌/施法（返回 `False` = 未生效） |
| `BattleState.use_ability(player_id)` | battle.py:3208 | 英雄能力/条件窗按钮 |
| `BattleState.deal_area_damage(...)` | battle.py:3250 | 区域伤害唯一公共结算函数 |
| `BattleState.update_player_hp()` | battle.py:2643 | 实体 HP → `PlayerState` 三塔 HP |
| `BattleState.winner` / `.game_over` | battle.py:2547-2548 | 终局字段（置位点见 A.6） |
| `environment.CREnv` | environment.py:33 | Gym 环境；持有 `self.battle: battle.BattleState`（environment.py:38） |

`CREnv.reset()` 每次新建一个 `BattleState`，双方初始圣水写死 **5.0**（environment.py:57-58）；`CREnv` 的 `speed` 默认 1.0、`card_level` 默认 11（environment.py:34）。

#### A.1.3 依赖方向

| 模块 | 依赖 | 依据 |
|---|---|--- |
| `battle.py` | `core.BlankEntity`、`player.PlayerState`、`arena.TileGrid`、`pathfinding_heap.{EntityPathfinder, position_to_cell, cell_to_position}`、`card_mechanics.*`、`card_utils.*`、`evolutions.*` | battle.py:1-8 |
| `arena.py` | `core.Position`（arena.py:2）、`math` | arena.py:1-3 |
| `core.py` | 仅 `dataclasses`/`math`（core.py:1-2） | — |
| `player.py` | `card_utils.Card`（player.py:1） | — |
| `environment.py` | `battle`、`player`、`new_visualization.Visualizer`、`core.Position` | environment.py:1-3 |
| `card_utils.py` | `gamedata.json` + `cards_stats_{characters,spell,building,projectile}.json`（card_utils.py:4-18）、`elite17_data.apply`（card_utils.py:179-180） | — |

⇒ 依赖是单向的：`environment → battle → {arena, core, player, card_utils}`；`arena`/`core`/`player` **不反向依赖** `battle`。

#### A.1.4 一处全局命名空间耦合（影响实体构造）

`battle.py:5` 用 `from card_mechanics import *` 把机制类名导入本模块全局命名空间；`Entity.__init__` 随后用 **卡名当类名字符串 eval** 绑定机制钩子载体：

```python
self.entity_holder = BasicCharacter(self)
if self.card_name in globals() and not isinstance(self, Projectile):
    self.entity_holder = eval(f"{self.card_name}(self)")   # battle.py:51-52
```

⇒ 「卡名 ↔ 机制类名」必须同名同命名空间，否则静默回退到默认 `BasicCharacter`。`card_mechanics` 不在本轮允许阅读的源码清单内，本文不描述其内部实现。

---

### A.2 地图与坐标系统

#### A.2.1 场地尺寸与坐标范围

| 常量 | 值 | 依据 |
|---|---|--- |
| `TileGrid.width, height` | `18, 32`（格） | arena.py:9 |
| `TileGrid.tile_size` | `100.0` | arena.py:10 |
| 合法世界坐标 | `0 <= x < 18`、`0 <= y < 32` | arena.py:100-101 |

**世界坐标 1 单位 = 1 格**：塔几何按「公主塔 3×3（half 1.5/1.5）、国王塔 4×4（half 2.0/2.0）」定义（arena.py:37-38），半径 `2.0` 的圈被注释称为「半径 2 格」（battle.py:567-574），卡牌数值表的毫米量按 `/1000` 归一（如 `range=7500 → 7.0`，card_utils.py:247；`collisionRadius=1000 → 1.0`，card_utils.py:239）。

**坐标系方向**：蓝方（`player_id=0`）王塔在 `y=3.0`、红方（`player_id=1`）王塔在 `y=29.0`（arena.py:11、arena.py:14）⇒ **小 y 侧是蓝方，大 y 侧是红方**；`x` 方向左右对称（蓝左塔 `x=3.5`、蓝右塔 `x=14.5`，arena.py:12-13）。`arena.get_deploy_zones` 的注释把玩家 0 称作 "bottom half"（arena.py:150），但代码给的区间是 `y=1..14`（arena.py:152）——注释方向与数值方向的对应关系需渲染层才能解释（见 A.7 第 5 项）。

#### A.2.2 关键坐标常量（原样抄录）

| 常量 | 值 `(x, y)` | 依据 |
|---|---|--- |
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

#### A.2.3 河、桥与不可通行格

- 河道带：`RIVER_Y1=15.0` 与 `RIVER_Y2=16.0`（arena.py:19-20）；`is_walkable` 判定 `self.RIVER_Y1 <= pos.y <= self.RIVER_Y2` 时**只有桥面可走**：左桥 `2.0 <= x < 5.0`、右桥 `13.0 <= x < 16.0`（arena.py:113-116）。
- `BLOCKED_TILES`（arena.py:21-34），逐项为：
  - 河岸边缘格：`(0,15) (0,16) (1,15) (1,16)`，`x∈[5,12] × y∈{15,16}`，`(16,15) (16,16) (17,15) (17,16)`（arena.py:23-25）；
  - `y=0` 与 `y=31` 两行：`x∈{0..5}` 与 `x∈{12..17}`（每行各 12 格）（arena.py:28-33）。
  ⇒ 桥面整数格为 `x∈{2,3,4}`（左）与 `x∈{13,14,15}`（右），与 `is_walkable` 的半开区间一致。
- `is_walkable` 带全局缓存 `walkable_cache`（按 `(int(x), int(y))` 键）（arena.py:5、arena.py:108-119）。
- `BattleState.in_river(position)` 用另一份同样的河格集合以格号判定（battle.py:2573-2577）——两处**各自硬编码**，不是同一常量源。
- 边界/河道夹取 `ensure_walkability(entity)`（battle.py:2579-2601）：不可走时把坐标夹回，`push_ratio = 0.5`；`y` 下界 `0.5*r`、上界 `32-0.5*r`；`x` 下界写作 `x = r`（注意：**不是** `0.5*r`）、上界 `18-0.5*r`；河道夹取 `y = 15-0.5*r` 或 `17+0.5*r`（取较近者），仅对非空中单位生效（battle.py:2592-2601）。`Building`、`Projectile`、`SpawnProjectile`、`AreaEffect`、`GenericBomb`、以及超骑跳跃空中态直接跳过该修正（battle.py:2583-2588）。

#### A.2.4 塔占位（矩形几何）

塔以**轴对齐矩形**参与占位与射程，忽略 `collisionRadius`：

| 塔 | 中心 | 半宽 / 半高 | `player_id` | 依据 |
|---|---|---|---|--- |
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

#### A.2.5 部署区 / 禁放区

**（a）`arena.get_deploy_zones` 的区间**（半开 `[x1,x2) × [y1,y2)`，判定见 arena.py:185）：

| 玩家 | 区间 | 展开后实际格 | 开启条件 | 依据 |
|---|---|---|---|--- |
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
|---|---|--- |
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

#### A.2.6 世界坐标 ↔ 本地网格（观测坐标）

观测张量形状 `(32, 18, 15)`（environment.py:42、environment.py:126），写入方式为 `obs[y][x] = obs_arr`，其中 `x, y = int(position.x), int(position.y)`（environment.py:146、environment.py:152）⇒ **网格下标 = 世界坐标向下取整**，格 `(x, y)` 覆盖 `[x, x+1) × [y, y+1)`。

镜像/翻转**只发生在观测编码层，且只对观测视角玩家 ≠ 0 时发生**：

| 变换 | 方向 | 公式 | 依据 |
|---|---|---|--- |
| 世界 → 本地网格（视角 0） | 不翻转 | `x = int(px)`、`y = int(py)` | environment.py:146 |
| 世界 → 本地网格（视角 1，`observe(1)`） | 180° 点对称 | `x = 17 - int(px)`、`y = 31 - int(py)` | environment.py:147-149 |
| 本地网格 → 世界（玩家 0 出牌） | 格心 | `Position(x + 0.5, y + 0.5)` | environment.py:94 |
| 本地网格 → 世界（对手 = 玩家 1 出牌） | 格心 + 180° 反变换 | `Position(18 - (x + 0.5), 32 - (y + 0.5))` | environment.py:72 |

- 「谁对谁做」：`observe(1)` 把**全场实体**（含双方）坐标翻到玩家 1 的本地系，并把归属标签写成 `each.player != player_id_observe`（己方恒 0、敌方恒 1）（environment.py:132、environment.py:147-149）；`opponent_action` 把玩家 1 视角的 `(y, x)` 反变换回世界坐标再 `deploy_card(1, ...)`（environment.py:66-72）。
- **实体真实坐标永不被翻转**：翻转只发生在生成观测数组的那几行。
- 手牌用循环队列前 5 张（environment.py:154），而可出牌判据是前 4 张（player.py:37）。
- 寻路层另有一套**半格网格**（36×64，即每格 2×2 个寻路单元）：`calculate_building_cache` 以 `(x_cell in 0..35, y_cell in 0..63)` 建距离场（battle.py:3033-3036），并通过 `cell_to_position`（battle.py:4 导入）取格心。该变换的定义在 `pathfinding_heap.py`，不在本轮允许阅读的源码清单内 ⇒ 列入 A.7。

---

### A.3 实体体系

#### A.3.1 类职责一览

| 类 | 基类 | 职责 | 依据 |
|---|---|---|--- |
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

#### A.3.2 公共属性（`Entity.__init__`，battle.py:14-99）

| 属性 | 初值 | 依据 |
|---|---|--- |
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
|---|---|--- |
| `Troop` | `deploy_delay_remaining = data.deploy_time`、`path_blocked_counter`、`_stuck_check_pos/_time`、`_lane_offset`（由 `position._lane_offset` 带入）、`jumping_across_river`、`start_jumping_position`、`spawned`、`evo_hits`、`evo_extra_spawned` | battle.py:762-775 |
| `Building` | `deploy_delay_remaining = data.deploy_time`、`lifetime_elapsed`、`target_id`、`tower_active=False`、`persistent`、`_tower_rect=0`（哨兵） | battle.py:1236-1244 |
| `Projectile` | `target_position`、`initial_position`、`proj = data.projectile_data`、`rolling = bool(proj.roll_range)`、`homing`、`target`、`source`、`collision_radius`（法术取 `proj.radius`，否则 `0.3`）、`damage_dealt`、`damage_override` | battle.py:1420-1434 |

注：`Troop.path_blocked_counter`、`Troop.spawned`、`Building.lifetime_elapsed`、`BattleState.regen`（battle.py:2550）在 `battle.py` 内**只被赋值、未见读取**（本文件内 grep 结果）。

#### A.3.3 实体注册表与 id 分配

- 注册表：`BattleState.entities: dict[int, Entity]`（battle.py:2542），键 = `int` id。
- id 计数器：`next_entity_id` 初值 `1`（battle.py:2549）。
- 6 座塔在 `__init__` 内按 `_spawn_entity` 顺序占用 id 1..6（battle.py:2556-2561）；因为 `_spawn_entity` 用计数器**覆盖**传入 id（battle.py:2606），实际映射为：

| id | 实体 | 归属 | 依据 |
|---|---|---|--- |
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

#### A.3.4 出生（spawn）流程

1. **出牌入口** `deploy_card`（battle.py:2806）：镜像分支（battle.py:2808-2825）→ `MergeMaiden` 双费用形态分支（battle.py:2833-2849）→ 手牌/圣水/王塔存活性校验（player.py:36-39）→ 觉醒形态判定（battle.py:2861-2863）→ 位置合法性（A.2.5）→ 按 `card_info.type` 分派：
   - 法术：`Lightning` 特判（battle.py:2892-2895）、区域持续出兵（battle.py:2896-2911）、克隆（battle.py:2912-2924）、瞬发区域法术 `AreaEffect`/`EvoZapZone`（battle.py:2928-2939）、有弹道的法术（battle.py:2941-2994）；
   - 部队/建筑：`get_spawn_position` 得到一组落点（battle.py:2996）→ 打车道偏移标记（battle.py:3003-3005）→ 逐个 `delayed_spawn`（battle.py:3007-3009）→ 可选第二部队（battle.py:3013-3025）→ `_finish_deploy` 收尾（battle.py:2728-2749，扣费 + 卡序推进 + 觉醒出牌计数）。
2. **多单位落点** `get_spawn_position(card_info, position, player, offset_angle=True)`（battle.py:2524-2535）：`spawn_number == 1` 时返回原位置；否则按下标 `i` 取角 `2πi/spawn_number`，`player == 1` 时整组加 `π`，半径 `spawn_radius`（`summonRadius/1000`，card_utils.py:234）；`spawn_number == 2/3/4/6` 时按 `{2:0, 3:π/2, 4:π/4, 6:0}` 加固定偏角（battle.py:2528）。
3. **延迟出兵** 在 `step` 的固定阶段结算（battle.py:2709-2711）。
4. **亡语出兵** `_generic_death_spawn`（battle.py:276-323）：读 `death_spawn_data`；炸弹型（有 `deathDamage` 无 `hitpoints`）改生成 `TimedExplosive`（battle.py:298-302）；`SkeletonBalloon`/`SkeletonContainer` 走 7 个延迟 `Skeleton`（battle.py:303-310）；其余按 `deathSpawnCount` 出兵（battle.py:311-323）。有专属 `on_death` 的机制类与觉醒同名亡语会跳过（battle.py:292-295）。

#### A.3.5 死亡（kill）流程

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

#### A.3.6 等级与数值缩放（与实体构造相关）

- `Card.default_level` 类的全局默认等级 = `11`（card_utils.py:220）；`BattleState.__init__` 把它设为本次战斗的 `card_level`（battle.py:2540-2541）——源码注释声明该机制基于**单战斗串行**假设（card_utils.py:218-219），多战斗并行需显式传 `level`。
- `Card.__init__` 从数值表读取并换算：`hp`（card_utils.py:226）、`elixir = manaCost`（card_utils.py:227）、`collision_radius = collisionRadius/1000`（card_utils.py:239）、`hit_speed = hitSpeed/1000`（card_utils.py:240）、`load_time = loadTime/1000`（card_utils.py:241）、`speed = speed/50`（card_utils.py:242）、`range/sight_range = /1000`（card_utils.py:247-248）、`deploy_time = deployTime/1000`（card_utils.py:249）、`spawn_number/spawn_delay/spawn_radius`（card_utils.py:232-234）、`tower_damage_mult = 1 + crownTowerDamagePercent/100`（card_utils.py:291）。
- 逐级取值 `_value_at_level(arr, rarity, level, base)`：索引在数组内直接取值，越界按 `arr[-1] * 1.1**(li - len(arr) + 1)` 延续，数组为空按 `base * 1.1**(level-1)`（card_utils.py:196-207）；稀有度轴 `_rarity_level_index`：Common `level-1`、Rare `level-3`、Epic `level-6`、Legendary `level-9`、Champion `level-11`，未知按 Common（card_utils.py:182-193）。`level_scale(level) = 1.1 ** (level - 1)`（card_utils.py:210-214）。
- 幻影等级窗口：`step` 内镜像出兵时临时 `Card.default_level += 1`，处理完 `schedule` 后还原（battle.py:2706-2714）。
- 战斗中**可变**的全局态：`Card.default_level` 会被 `BattleState.__init__`（battle.py:2541）与镜像窗口（battle.py:2708/2713）改写。

---

### A.4 战斗主循环

#### A.4.1 `step(dt)` 逐阶段顺序（严格按源码顺序）

| # | 阶段 | 行号 | 内容 |
|---|---|---|--- |
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

#### A.4.2 `dt` 取值来源与固定步长

| 事实 | 值 | 依据 |
|---|---|--- |
| `step` 的 `dt` 唯一来源 | `self.battle.step(1/60)` | environment.py:102 |
| 每次决策最多推进 | `for i in range(30)`，且每帧先判 `game_over` 提前 `break` | environment.py:98-100 |
| 固定步长含义 | 30 × 1/60 = **0.5 秒/决策** | environment.py:80-81、environment.py:98 |
| 子步乘数 | `for j in range(int(self.speed))`（`speed` 默认 1.0；`speed<1` 时 `int()` 截断为 0） | environment.py:101、environment.py:34 |

⇒ 引擎侧是**固定步长 1/60 s** 的离散模拟；`dt` 不是自适应值。

---

### A.5 伤害与命中

#### A.5.1 伤害唯一入口 `Entity.take_damage(amount, delayed=False, source=None, pierce_invincible=False)`（battle.py:507）

按代码顺序：

| # | 判定 | 行为 | 依据 |
|---|---|---|--- |
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

#### A.5.2 护盾 / 减伤 / 治疗

| 机制 | 规则 | 依据 |
|---|---|--- |
| 护盾 | `shield_health` 优先吸收；`data.shield_health` 初值来自 `shieldHitpoints` | battle.py:26、battle.py:540-542、card_utils.py:272 |
| 减伤（buff 槽） | `apply_buff(damage_reduction=...)` 取 `max` 并给 `damage_reduction_timer` | battle.py:146-148 |
| 减伤（fortify 独立槽） | 觉醒 `buffWhenNotAttackingData.damageReduction/100`；「脱战」= 距 `last_attack_time` 超过 `1.0` 秒 | battle.py:397-403 |
| 减伤生效 | 两槽取 `max` 后 `amount *= (1-dr)` | battle.py:526-529 |
| 治疗（导槽） | `regen_buffs` 每项 `{'hps','time'}`，逐帧 `hp = min(_cap, hp + hps*dt)`；`_cap = data.hp`，觉醒 `allowedOverHealPerc` 存在时 `_cap = data.hp*(1+perc/100)` | battle.py:374-388 |
| 治疗（光环） | `HealAuraZone`：`ticks=4`、`interval=0.25`、只疗 `Troop`、排除 `exclude_id`、上限 `data.hp` | battle.py:2043-2055、battle.py:2078-2088 |
| 治疗（区域法术） | `AreaEffect` `buff_name == 'Heal'`：`heal={'hps': heal_per_tick/tick, 'time': tick}` | battle.py:1812-1814 |
| 冻结/眩晕 | `apply_buff(stun=s)`：突进态免控（battle.py:106-109）；`freeze_timer = max(...)`、`attack_cooldown = max(attack_cooldown, hit_speed)`、重置蓄能与冲锋、可选重索敌 | battle.py:106-126 |

#### A.5.3 攻击间隔与首次攻击

| 项 | 规则 | 依据 |
|---|---|--- |
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

#### A.5.4 投射物飞行与命中

| 项 | 规则 | 依据 |
|---|---|--- |
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

#### A.5.5 索敌与射程几何

| 函数 | 规则 | 依据 |
|---|---|--- |
| `in_attack_range(target)` | `dist = target.edge_distance_from(self.position)`；`'PrincessTower' in target.name` 时 `bonus = 0.5`（battle.py:588-591）；`_range_override` 存在则 `dist <= 覆盖值 + bonus`（battle.py:595-597）；`min_range` 存在且 `dist < min_range` ⇒ `False`（battle.py:599-600）；狙击临时射程命中 ⇒ `True`（battle.py:602-603）；否则 `dist <= data.range + bonus`（battle.py:604） | |
| `in_sight_range(target)` | 同 `bonus` 口径；狙击射程同步扩展；否则 `dist <= data.sight_range + bonus` | battle.py:605-616 |
| `get_nearest_target` | 遍历 `Troop`/`Building`、存活、非同方、`targetable`、`min_range` 过滤、空地能力过滤、`in_sight_range` 过滤；建筑与部队分桶，`target_only_buildings` / 已在攻击范围 / 否则只看部队，最后按距离排序取最近 | battle.py:618-649 |
| 无目标兜底 | 从 id 1..6 里挑**同侧**最近的敌方公主塔（`(tower.x - width/2) * (self.x - width/2) >= 0`）；同侧无存活则取 `best_any`；距离用 `edge_distance_from` | battle.py:700-727 |
| `_should_switch_target` | `target_only_buildings` 时拒绝非建筑；建筑遇进入攻击范围的部队立即转火；当前目标已在攻击范围则不换；否则比距离 | battle.py:651-670 |
| `update_current_target` | 目标死亡/出视野即清空（塔目标不清路径）；再取最近目标并决定是否切换 | battle.py:672-727 |
| Hero 嘲讽覆盖 | `_taunt_until > battle_state.time` 时强制锁定嘲讽者 | battle.py:729-736 |
| `deal_area_damage` | 逐实体：死亡/同方/`invincible` 跳过；`"King" in name` ⇒ `amount * crown_tower_damage_percent`；按 `attack_air`/`attack_ground` 与实体 `is_air_unit` 过滤；`dist = entity.edge_distance_from(position)`，**严格 `dist < range`** 才命中 | battle.py:3250-3265 |
| 塔的 edge 距离 | `persistent and id<=6` ⇒ 矩形最近点距离；否则 `中心距 - collision_radius` | battle.py:152-168 |

---

### A.6 皇家塔与胜负

#### A.6.1 塔的 id / 数量 / 血量

- 数量：每方 3 座（2 公主塔 + 1 国王塔），全场 6 座，id 1..6（battle.py:2556-2561，映射见 A.3.3）。
- 塔是 `Building(persistent=True)` 实例——第 5 个位置参数为 `True`（battle.py:2556-2561，形参见 battle.py:1234）。
- 公主塔卡名由 `PlayerState.tower_troop` 决定：默认 `'King_PrincessTowers'`（battle.py:2554-2555，`tower_troop` 默认 `None`，player.py:15）；合法值 `King_CannonTowers / King_KnifeTowers / King_ChefTowers`（player.py:24）。
- 国王塔卡名固定 `'KingTower'`（battle.py:2560-2561）。

**血量**（本轮可确证的来源）：

| 塔 | 数值 | 依据 |
|---|---|--- |
| 公主塔 / 国王塔 `PlayerState` 默认三塔血 | `(4824, 3052, 3052)`（顺序：king, left, right） | player.py:6、player.py:10 |
| 国王塔基值 | `hitpoints = 2100`（硬编码表） | card_utils.py:40 |
| 国王塔其它硬编码数值 | `hitSpeed 1000`、`damage 109`、`sightRange 7000`、`range 7000`、`collisionRadius 1400`、`deployTime 3300`、`loadTime 700`、弹道 `KingProjectile speed 600 / damage 109` | card_utils.py:41-53 |
| 逐级取值路径 | `Building`/`troop` 分支经 `_value_at_level(row['hitpoints_per_level'], rarity, level, base)` 取 HP | card_utils.py:366-367、card_utils.py:390-397 |
| lv11 解析结果 | `KingTower hp=4824 / dmg=109 / hit_speed=1.0 / deploy_time=3.3 / range=7.0 / sight=7.0 / collision=1.4`；`King_PrincessTowers hp=3052 / dmg=0 / hit_speed=0.8 / deploy_time=0.0 / range=7.5 / sight=7.5 / collision=1.0` | 由 card_utils.py:222-345 管线在 `level=11` 下解析（与 player.py:6 的硬编码默认值一致） |

⇒ `PlayerState` 的三塔血字段在**首次 `step`**（阶段 2，battle.py:2654）之前是硬编码默认值，之后每个 `step` 都被实体真实 HP 覆盖（battle.py:2643-2650）。

#### A.6.2 国王塔激活（`tower_active`）

| 触发 | 条件 | 依据 |
|---|---|--- |
| 自身受击 | `Building.take_damage` 内 `data.name == 'KingTower'` 且未激活 ⇒ 置 `True` | battle.py:1266-1271 |
| 同方公主塔阵亡 | `on_death` 内 `entity.name == 'King_PrincessTowers'` ⇒ 把同方 `KingTower` 置 `True` | battle.py:3148-3154 |

未激活的国王塔在 `Building.update` 开头直接 `return`（不索敌、不攻击）（battle.py:1282）。

> 源码事实：第二个触发条件比对的是实体 `name == 'King_PrincessTowers'`（battle.py:3149），而塔兵为 `King_CannonTowers/King_KnifeTowers/King_ChefTowers` 时实体 name 是卡名本身（`Card.name = data['name']`，card_utils.py:228；塔兵卡名与 `statCharacterData` 挂载见 card_utils.py:57-61）⇒ 这两种情形下公主塔阵亡**不会**经该路径激活国王塔。是否为设计意图无法从源码判定，列入 A.7。

#### A.6.3 塔被摧毁后的效果

1. **皇冠/胜负计数**：`get_crown_count` 把该玩家**自己被摧毁的塔**计入（own HP ≤ 0）；国王塔 ≤ 0 直接返回 3（player.py:54-58）。
2. **解除部署区封锁**：`arena.get_deploy_zones` 依据 `battle_state.players[对方].left/right_tower_hp <= 0` 追加敌半场 4 格回推区（arena.py:154-170）；`battle.deploy_card` 另有一份 `left_tower_hp/right_tower_hp > 0` 判定（battle.py:2872-2887）。
3. **退出占位/索敌**：每帧 `refresh_tower_alive_cache` 后，`tower_rect_dist` 跳过死塔（arena.py:79-80、battle.py:2697）；`is_tower_tile` 与 `get_tower_blocked_x_ranges` 同样跳过死塔（arena.py:135-138、arena.py:198-201）；`_push_troop_out_of_tower` 对死塔不再推出（battle.py:3119）。
4. **激活己方国王塔**：见 A.6.2。
5. **建筑距离场重算**：`on_death` 对 `Building` 置 `cache_fresh = False`（battle.py:3155）。
6. 塔死后的**实体仍留在 `entities`**（id ≤ 6），因此读 `entities[1..6]` 恒定安全（battle.py:2691）。

#### A.6.4 `game_over` / `winner` 的全部置位点

`winner` 初值 `None`、`game_over` 初值 `False`（battle.py:2547-2548）；**全部置位都只发生在 `step` 的阶段 4**（本文件内 grep `game_over|winner` 无其它写入点）：

| # | 条件 | 结果 | 依据 |
|---|---|---|--- |
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

#### A.6.5 皇冠计数来源

`PlayerState.get_crown_count()`（player.py:54-58）读的是 `PlayerState` 的三个 HP 字段：`king_tower_hp <= 0 ⇒ 3`，否则 `int(left_tower_hp <= 0) + int(right_tower_hp <= 0)`。这些字段由 `BattleState.update_player_hp()` 在每个 `step` 开头从实体同步（battle.py:2643-2650）；`PlayerState` 自身的初值是构造参数（environment.py:57-58 用默认 `tower_hps`，player.py:6）。

---

### A.7 本部分待确认清单

| # | 条目 | 无法确认的原因 | 要确认需要什么 |
|---|---|---|--- |
| 1 | 半格寻路网格的坐标变换（`position_to_cell` / `cell_to_position` 的具体公式） | 定义在 `pathfinding_heap.py`，不在本轮允许阅读的源码清单内；`battle.py` 只显示网格为 36×64（battle.py:3033-3036），`battle.py:4` 仅导入函数名 | 允许阅读 `pathfinding_heap.py`（或让作者给出该文件的行号引用）；可先看 `docs/_survey/parts/G037.md` 的记载再回源码核验 |
| 2 | `TileGrid.tile_size = 100.0` 的用途 | 该常量在 `arena.py`/`battle.py` 内定义后无任何读取点（本目录 `*.py` grep 仅命中 arena.py:10） | 全仓（含渲染层 `new_visualization.py`）搜 `tile_size` |
| 3 | `arena.can_deploy_at(is_spell=True, spell_obj=...)` 的行为 | arena.py:181 调用的 `self._is_rolling_projectile_spell(...)` 在 `arena.py`（209 行）内未定义，全仓亦未找到 Python 定义；当前唯一调用方传 `is_spell=False`（battle.py:2855），该分支从未执行 | 查 `spell_module.py:261` 的调用参数（不在本轮清单）；或让作者确认该方法是否被删除 |
| 4 | ~~部署合法性由哪一层兜底~~ **（已由本文档 §B.8 回答，2026-09-14 独立校验后更正）** | `battle.deploy_card` 的部队/建筑路径**不**检查 `BLOCKED_TILES`/河道（battle.py:2865-2887），只有 `arena.can_deploy_at` 检查（arena.py:176）；**RL 掩码层补齐情况见 §B.8**（`legal_cells`/`_position_legal` 逐格流程 + 与引擎 `deploy_card` 的逐项对照，含"掩码更严：无 Miner 例外"）；逐格双向对账见 `docs/mask_vs_engine_reconcile_2026-09-14.log` | 已完成（§B.8） |
| 5 | 「bottom half / top half」注释与 y 数值方向的对应 | arena.py:150/arena.py:161 注释称玩家 0 为 bottom、玩家 1 为 top，而数值上玩家 0 的区间是 `y=1..14`（小 y 侧，arena.py:152）；坐标是否在渲染层翻转需 `new_visualization.py` | 允许阅读 `new_visualization.py`（或给出其坐标变换行号） |
| 6 | 塔兵为 Cannon/Knife/Chef 时国王塔激活语义 | 只有 `name == 'King_PrincessTowers'` 的公主塔阵亡才激活同方国王塔（battle.py:3149-3154），塔兵换卡后 name 变为该卡卡名（card_utils.py:57-61、card_utils.py:228）⇒ 该路径不触发 | 官方机制口径（源码内无注释说明是否为有意），或补一条针对塔兵卡名的分支/测试 |
| 7 | 180 秒皇冠判定的规则来源 | 源码只有数值 `300 > self.time >= 180`（battle.py:2667）与其后注释（battle.py:2677-2679），未标注对应真实 CR 规则或数据出处 | 作者/设计文档口径，或对局录像对拍 |
| 8 | 对手出牌失败是否被吞掉 | `opponent_action` 丢弃 `deploy_card` 返回值（environment.py:72），失败（费用不足/位置非法）静默忽略 | 确认 RL/脚本对手层是否有独立合法性检查（不在本轮清单） |
| 9 | 国王塔逐级 HP 数组与「4824」的确切出处 | 本轮只能确证：`king_tower_stats.hitpoints = 2100`（card_utils.py:40）+ 经 `buildings` 表按稀有度轴取值（card_utils.py:366-367）+ 与 `player.py:6` 硬编码 `(4824, 3052, 3052)` 一致；`cards_stats_building.json` 是数据文件，非 `.py` 源码 | 允许直接读 `gamedata.json` / `cards_stats_building.json` 的 `KingTower`、`PrincessTower` 行，或给出引用该表的源码行 |
| 10 | 引擎是否被 RL 侧以外的方式驱动（决策帧、动作槽位语义） | 本文只覆盖 `CREnv`（5 槽手牌、格心动作，environment.py:94、environment.py:154）；训练用的 `rl/` 环境不在本轮清单 | 读 `rl/env_wrapper.py`、`rl/action_mask.py`（不在本轮清单） |
| 11 | `PlayerState` 默认塔血与 `card_level` 不一致的窗口 | `tower_hps` 默认 `(4824, 3052, 3052)` 是 lv11 值（player.py:6），而 `BattleState(card_level=...)` 可为 11-16；在首次 `step` 覆盖之前（battle.py:2654），`can_play_card` 读的是默认值（player.py:39） | 确认是否存在「首次 step 前读 `*_tower_hp`」的调用路径（`rl/` 不在本轮清单） |
| 12 | 多战斗并行的等级全局态安全 | `Card.default_level` 是类变量，由 `BattleState.__init__` 与镜像窗口改写（card_utils.py:220、battle.py:2541、battle.py:2706-2714）；源码注释声明基于单战斗串行假设（card_utils.py:218-219），并行场景的后果未在源码中给出 | 官方并行/多进程方案的说明，或一条并发回归测试 |

---

---

## 游戏引擎文档 · 第 B 部分：寻路、索敌、卡牌机制与部署合法性

> **取证口径**：本部分只依据源码与本次会话的实测。每条结论后跟行内来源 `（文件.py:行号）`。
> 无法从源码确认的一律写「待确认」。数值原样抄自源码，不改写。
>
> 三个此前会话产出的中间素材（`docs/_survey/parts/G002|G004|G015|G019|G035|G036|G038.md`）仅用于交叉检索关键字，
> 正文结论全部回到源码复核；本部分的实测数字来自会话内临时脚本 `.tmp/mask_vs_engine.py`（`git check-ignore` 确认
> `.tmp/` 被忽略，不入库）。

---

### B.1 寻路总览

#### B.1.1 谁需要寻路

| 主体 | 是否走 A* | 依据 |
|---|---|--- |
| 地面 `Troop` | **是**，`Troop.update` 里唯一 A* 调用方 | `battle.py:1176-1197` |
| 空中单位（`data.is_air_unit=True`） | **否**，直接 `move_towards(current_target.position)` | `battle.py:1176-1177` |
| 跳河状态（`jumping_across_river`） | **否**，仍走 `move_towards`；跳河时临时置 `data.is_air_unit=True` | `battle.py:1123-1126, 1169-1177` |
| `Building` / `Projectile` | **否**（`Building.update` 与 `Projectile.update` 无 A*） | `battle.py:1273-1330, 1548-1594` |
| 机制类直线位移（Assassin 突进 / MegaKnight 冲刺跳 / HeroValkyrie 冲刺 / HeroGiant 投掷 / Skeletrooper 伞降） | **否**，直接改 `position`，并在位移后清空 `path` | `card_mechanics.py:774-807, 552-628, 1007-1021, 1181-1187, 1437-1438` |

#### B.1.2 输入与输出

**输入**（`EntityPathfinder.__init__`，`pathfinding_heap.py:37-45`）：

| 参数 | 用途 |
|---|--- |
| `entity` | 取 `entity.position`（起点）、`entity.data.collision_radius`、`entity.data.range`、`entity.data.is_air_unit` |
| `target` | 取 `target.position`、`target.edge_distance_from(pos)`（目标边缘距离）、`target._tower_rect`（塔矩形） |
| `battle_state` | 只用来调 `battle_state.pathfind_ground_walkable(pos, radius)` |

**输出**：`calculate()` 返回 `list[Position]`，元素是**半格中心**世界坐标，起点在 `path[0]`（`pathfinding_heap.py:154-160`）。
路径长度不设上限；无路可走时返回的路径由 A* 实际展开的 `parent` 链决定（见 B.3.5 边界条件）。

#### B.1.3 调用点（全部）

全仓只有 `battle.py` 导入并调用：

| 位置 | 内容 |
|---|--- |
| `battle.py:4` | `from pathfinding_heap import EntityPathfinder, position_to_cell, cell_to_position` |
| `battle.py:1180` | `if not self.path:` → 重算全程路径 |
| `battle.py:1182` | `elif self.in_sight_range(current_target) and self.battle_state.tick % 10 == 0:` → 目标在视距内时每 10 tick 刷新 |
| `battle.py:1197` | 卡死自救：每 0.5s 检查，位移 `< 0.1` 则删掉当前路点，路点删空后重算 |

`battle.py:1180/1182/1197` 是**仅有的三处** A* 构造点（全仓 grep `EntityPathfinder(` 结果一致）。

#### B.1.4 路径消费与失效

消费在 `battle.py:1201-1223`：取 `min(self.path, key=distance_to(self.position))` 作为最近路点，用
`start_vector`（起点→自身）与 `close_vector`（自身→路点）的点积 `dot >= 0` 决定是否切到下一个路点；
最后一格直接 `move_towards(current_target.position)`；中间路点再按 `_lane_offset` 沿法线平移（队形车道）。

`path = []`（强制重算）出现在：目标失效 / 目标脱离视距（`battle.py:680, 685`）、嘲讽锁定（`battle.py:918`）、
眩晕重索敌（`battle.py:123-125`）、机制类位移后（`card_mechanics.py:1020, 1068, 1079, 1124, 1187, 1271, 1522, 1758`）、
非卡牌实体构造（`battle.py:1682, 1736, 1908, 1969, 2062, 2118`）。

---

### B.2 网格与通行性

#### B.2.1 两套坐标系

| 层 | 尺寸 | 定义 |
|---|---|--- |
| 世界/竞技场 | 18 × 32（`width, height = 18, 32`） | `arena.py:9`；`is_valid_position` 判 `0 <= x < 18 and 0 <= y < 32`（`arena.py:100-101`） |
| 寻路网格 | 36 × 64 **半格** | `position_to_cell = (floor(2x), floor(2y))`（`pathfinding_heap.py:13-15`）；反向 `cell_to_position = ((x+0.5)/2, (y+0.5)/2)`，带 `cell_cache`（`pathfinding_heap.py:17-21`） |

**1 寻路格 = 0.5 世界单位**；`TileGrid.tile_size = 100.0` 声明存在但寻路路径与 `is_walkable` 均未使用它（`arena.py:10`）。

#### B.2.2 地形代价表 `tilemap_lane_grid.txt`

- 文件 64 行 × 36 列（**行数 = 64 = 寻路网格高度，列数 = 36 = 宽度**，与半格网格一一对应）。
- 字符分布（会话内统计）：`.` 1500、`1` 346、`2` 346、`W` 112。
- 读取方式：`contents = [list(each) for each in f.read().splitlines()]`（`pathfinding_heap.py:6-8`；`pathfinding.py:6-8` 同）。
- **行索引翻转**：查表用 `contents[63 - ny][nx]`（`pathfinding_heap.py:132`，`pathfinding.py:106` 同），即文件第 0 行对应 `ny=63`。

代价映射（`pathfinding_heap.py:132-147`）：

| 字符 | 地面 `tile_cost` | 空中 `tile_cost` |
|---|---|--- |
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

#### B.2.3 `arena.TileGrid` 的通行性

| 判定 | 规则 | 行号 |
|---|---|--- |
| 边界 | `0 <= x < 18 and 0 <= y < 32` | `arena.py:100-101` |
| `BLOCKED_TILES` | 河岸两端 `(0,15)(0,16)(1,15)(1,16)`、`(16,15)(16,16)(17,15)(17,16)`；河中央 `x∈[5,13) × y∈[15,17)`；上下底边各 6 格围栏 `y=0/y=31` 且 `x∈[0,6)∪[12,18)` | `arena.py:21-34` |
| 河流/桥 | `RIVER_Y1 = 15.0`、`RIVER_Y2 = 16.0`；`y ∈ [15.0, 16.0]` 时仅 `2.0 <= x < 5.0`（左桥）或 `13.0 <= x < 16.0`（右桥）可走 | `arena.py:113-116`；桥常量 `LEFT_BRIDGE=(3.5,16.0)`、`RIGHT_BRIDGE=(14.5,16.0)`（`arena.py:17-18`） |
| 缓存 | `walkable_cache[(int(x), int(y))]`（**按整格缓存**） | `arena.py:5, 108-110` |

> `is_walkable` 的缓存键是 `(int(x), int(y))`，因此同一 1×1 世界格内不同子坐标会命中同一结果；
> 这与寻路的 0.5 单位半格网格不同粒度，**是否为有意设计：待确认**（见 B.9 #4）。

塔是**矩形**（2026-09-09 定稿）：公主塔 3×3（half 1.5）、国王塔 4×4（half 2.0），列表见 `arena.py:36-46`，
点到矩形距离 `dist_to_rect`（`arena.py:48-53`），存活塔最近距离 `tower_rect_dist`（`arena.py:65-84`，靠每帧刷新的
`_tower_alive` 缓存 `arena.py:55-63`）。

#### B.2.4 两套 walkability API（容易混淆，务必区分）

| 函数 | 判定 | 行号 | 谁用 |
|---|---|---|--- |
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

### B.3 `pathfinding_heap.py`（引擎实际使用的那一套）

**结论先行**：引擎实际使用的是 `pathfinding_heap.py`，证据是唯一的导入点
`battle.py:4` 以及三处调用点 `battle.py:1180, 1182, 1197`（全仓 grep 无其它导入方）。

#### B.3.1 清单

| 名字 | 行号 | 说明 |
|---|---|--- |
| 模块级 `contents` | 6-8 | 读入 `tilemap_lane_grid.txt`（64 行 × 36 列） |
| `cell_cache` / `neighbor_cache` | 10-11 | 世界坐标缓存 / 邻居表缓存 |
| `position_to_cell(position)` | 13-15 | 世界 → 半格 |
| `cell_to_position(cell)` | 17-21 | 半格 → 世界（带缓存） |
| `get_neighboring_points(x, y)` | 23-33 | 8 邻域；越界 `nx<0 or ny<0 or nx>=36 or ny>=64` 丢弃；结果缓存 |
| `class EntityPathfinder` | 36 | 见下 |
| `.__init__(entity, target, battle_state)` | 37-45 | 状态初始化 |
| `.heuristic(cell)` | 47-50 | `10 * max(\|Δx\|, \|Δy\|)`（Chebyshev × 10） |
| `._target_footprint_radius()` | 52-63 | 塔取 `_tower_rect` 半轴较大者，否则 `data.collision_radius` |
| `.calculate()` | 65-160 | 目标格生成 + A* |
| `__main__` 演示 | 162-172 | 直接构造 `BattleState` 冒烟 |

#### B.3.2 `g` / `h` 定义

- `g[start_cell] = 0`（`pathfinding_heap.py:112`）。
- 转移代价：`step_cost = tile_cost * geo_cost`（`pathfinding_heap.py:147`），具体数值见 B.2.2 表：
  - 地面：`W=800→8000`，`.`/`1`/`2`/其它`=5→50`（直线）/`70`（对角）
  - 空中：`W=7→70`（直线）/`98`（对角），其余 `5→50/70`
- 启发式：`h(cell) = 10 * max(|x-gx|, |y-gy|)`，`(gx,gy) = self.goal`（`pathfinding_heap.py:47-50`）。
  最小真实步代价是 50，`h` 每格只计 10 ⇒ `h` 是**下界**（admissible & consistent），A* 在
  「目标格=单点」语义下给出最小代价路径。
- `f = g + h`（`pathfinding_heap.py:113, 152`）。

#### B.3.3 目标格（goals）的生成与择优

1. `edge_radius = entity.data.range`（`pathfinding_heap.py:74`）。
2. 扫描窗：`scan_radius = ceil((目标占用半径 + edge_radius) * 2) + 2`，中心为目标所在半格
   （`pathfinding_heap.py:76-78`）。
3. 判定：`target.edge_distance_from(pos) < edge_radius + 0.375` **且** `pathfind_ground_walkable(...)`
   （`pathfinding_heap.py:80-84`）。0.375 是让短射程部队够得着塔的裕量（源码注释 `pathfinding_heap.py:81-82`）。
4. 空集兜底：退化为「扫描区内到目标边缘最近的可走格」；仍无则直接以目标格为 goal
   （`pathfinding_heap.py:85-101`）。
5. 择优：`self.goal = min(goals, key=edge_distance_from(pos) + distance_to(start_position))`
   （`pathfinding_heap.py:106`）—— 边缘距离基本相等，起主导的是「离起点最近」，即车道保持。

#### B.3.4 堆与松弛

- `open_heap = [(f[start], start)]`，`heapq.heappop` 取最小；用 `closed_set` + 「`current_f > f[current]` 跳过」
  实现**惰性删除**（`pathfinding_heap.py:114-124`）。
- 每个邻居：`closed_set` 跳过；不可走跳过；按 B.2.2 计算 `step_cost`；`tentative_g < g[neighbor]`
  或邻居未访问则更新 `g/parent/f` 并 `heappush`（`pathfinding_heap.py:125-153`）。
- 终止条件：弹出 `current == self.goal` 即 `break`（`pathfinding_heap.py:122-123`）。
- 目标是**单点**（`self.goal`），不是 goals 集合 —— 这是与 `pathfinding.py` 的主要差异之一。

#### B.3.5 返回路径与边界条件

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

### B.4 `pathfinding.py`（未使用的旧实现，与 B.3 的关系）

#### B.4.1 死代码判定

**全仓未见调用点**。依据：

```
grep -rn --include=*.py -e "import pathfinding" -e "from pathfinding" -e "pathfinding\." . \
  --exclude-dir=.venv --exclude-dir=.git --exclude-dir=__pycache__
→ 唯一命中：./src/clasher_new/battle.py:4: from pathfinding_heap import EntityPathfinder, ...
```

`pathfinding.py` 内对 `EntityPathfinder` 的唯一构造在它自己的 `__main__` 演示（`pathfinding.py:141`）。
即：本模块在仓库内**没有任何 import 方**（`battle.py:4` 导入的是 `pathfinding_heap`）。

#### B.4.2 与 B.3 的逐项差异

| 维度 | `pathfinding_heap.py`（在用） | `pathfinding.py`（未在用） |
|---|---|--- |
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

#### B.4.3 `pathfinding.py` 的一处静态错误

`pathfinding.py` 的 `__init__` 只赋值了 `self.battle`（`pathfinding.py:41`），**从未赋值 `self.battle_state`**，
但塔分支引用了它：

```python
if getattr(self.target, 'persistent', False) and getattr(self.target, 'id', 99) <= 6 \
        and self.battle_state is not None:      # pathfinding.py:61-62
```

⇒ 一旦进入该分支（目标是 `persistent` 且 `id <= 6`，如公主塔/国王塔），会因 `AttributeError` 崩溃。
由于该模块无调用方（B.4.1），**实际不可达**；此处只作为「若把它改回调用方会踩的坑」记录。

---

### B.5 索敌与目标选择

#### B.5.1 `sight_range` / `range` 字段来源

| 字段 | 来源 | 行号 |
|---|---|--- |
| `Card.sight_range` | `summonCharacterData.sightRange / 1000` | `card_utils.py:248` |
| `Card.range` | `summonCharacterData.range / 1000` | `card_utils.py:247` |
| `Card.min_range` | 数值表行 `minimum_range / 1000`（Mortar/GoblinCannon/BarbarianLauncher=3.5） | `card_utils.py:314-315` |
| `Card.is_air_unit` | `characters_data[].flying_height != 0` 名单 或 角色名命中 | `card_utils.py:9, 244` |
| `Card.attack_air` / `attack_ground` | `tidTarget` 含 `AIR` / 含 `GROUND`（`target_only_buildings` 视为可打地） | `card_utils.py:245-246` |
| `Card.target_only_buildings` | `tidTarget == "TID_TARGETS_BUILDINGS"` | `card_utils.py:243` |

**统一射程口径 = 边缘距离** `Entity.edge_distance_from(pos)`（`battle.py:152-168`）：
塔（`_tower_rect` 已绑定）→ 到矩形最近点距离（矩形内 0）；其余实体 → `中心距 − collision_radius`。
`_bind_tower_rect` 在首次索敌时惰性绑定，用 `0` 哨兵区分「未绑定」与「非塔」（`battle.py:158-161, 170-183`）。

#### B.5.2 在射程 / 在视距

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

#### B.5.3 目标选择：`get_nearest_target`

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

#### B.5.4 换目标条件：`_should_switch_target`

（`battle.py:651-670`）

1. `target_only_buildings` 且新目标不是建筑 → 不换（`battle.py:654`）；
2. `not new_target` → 换（`battle.py:655-656`）；
3. **防御建筑优先转火**：`self` 是 `Building`、新目标是 `Troop`、当前目标不是 `Troop`，
   且新目标在攻击范围内 → 换（`battle.py:657-660`）；
4. 当前目标仍在攻击范围内 → **锁定不换**（`battle.py:661-662`）；
5. 新目标是部队、当前是建筑 → 换（`battle.py:663-667`）；
6. 否则比较与新/旧目标的**中心距**，新目标更近才换（`battle.py:668-669`）。

#### B.5.5 重选时机：`update_current_target`

（`battle.py:672-727`）

1. 目标 id 缺失 / 不在实体表 / 已死 → 清 `target_id` 与 `path`（`battle.py:675-680`）；
2. 目标脱离视距 → 非塔目标清空 `path` 并释放目标（塔例外，保留）（`battle.py:683-687`）；
3. 取 `get_nearest_target()`；已有目标则按 `_should_switch_target` 决定是否换，无目标则直接采用
   （`battle.py:689-696`）；
4. **兜底**：仍无目标（如后排刚落地）→ 在敌方 6 座塔里选最近的公主塔，且**不跨中轴**
   （`(塔.x − 9) * (自己.x − 9) >= 0`）；同侧无存活塔才放宽到任意塔；国王塔在中轴不受限
   （`battle.py:698-727`）。

#### B.5.6 仇恨 / aggro

- **引擎侧没有通用 aggro/仇恨值机制**：`grep -rn "aggro|仇恨"` 仅命中 `rl/action_mask.py:281, 341` 与
  `rl/belief_planner.py` 的**注释/启发式文案**，`battle.py`、`card_mechanics.py` 无命中。
- 唯一的强制锁定是 **Knight Hero 嘲讽**：`_hero_taunt_override` 在 `update_current_target` 之后覆盖
  普通索敌（`battle.py:729-736`），嘲讽窗内强制锁 `_taunt_target_id`；
  `HeroKnight.use_ability` 把 6.5 格内敌军 `_taunt_until = time + tauntDuration` 并清 `path`
  （`card_mechanics.py:908-922`）。
- 眩晕可附带**重索敌**：`apply_buff(stun=..., retarget=True)` → `target_id = None; path = []`
  （`battle.py:123-125`）；调用点：ZapFreeze 的 `AreaEffect._pulse`（`battle.py:1799`）与 Lightning
  （`battle.py:2804`）。

#### B.5.7 法术索敌 vs 单位索敌

| 维度 | 单位/建筑 | 法术 |
|---|---|--- |
| 目标是什么 | 实体 id（`target_id`），持续跟踪 | **坐标**（落点），无 `target_id` 语义 |
| 命中判定 | `edge_distance_from ≤ range`（边缘） | 罩圈判定：`中心距 ≤ radius + 目标碰撞半径`（`battle.py:1785-1786`） |
| 视距过滤 | 有（`sight_range`） | **无**（`AreaEffect._pulse` 遍历全实体，`battle.py:1788-1831`） |
| 敌我过滤 | `player != self.player` | `only_enemies` / `only_own_troops` 行字段（`battle.py:1748-1749, 1792-1793`） |
| 建筑 | `target_only_buildings` | `ignore_buildings` 行字段（`battle.py:1750, 1794`） |
| 选目标的方式 | 最近目标（B.5.3） | 半径内全体（AreaEffect）/ 半径内 HP 最高的至多 3 个（Lightning，`battle.py:2793-2804`） |
| 弹道法术 | 弹丸 `homing=True` 追实体位置（`battle.py:1588`） | 落点溅射在 `target_position`（`battle.py:1605-1612`） |

#### B.5.8 `threat_calc.py` 的威胁口径

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

### B.6 卡牌机制分类（`card_mechanics.py`）

#### B.6.1 组织方式与装配链

- 每个机制类**必须与卡名同名**，因为 holder 是 `eval` 出来的：
  `self.entity_holder = BasicCharacter(self)`；`if self.card_name in globals() and not isinstance(self, Projectile): self.entity_holder = eval(f"{self.card_name}(self)")`
  （`battle.py:50-52`）。`card_mechanics` 通过 `from card_mechanics import *` 注入 `battle.py` 命名空间（`battle.py:5`）。
- 基类 `BasicCharacter`（`core.py:16-28`）：持 `entity` / `battle_state` / `data`，默认钩子
  `on_spawn`（空）、`on_tick`（只同步 battle_state）、`on_death`（空）、`on_attack`（`core.py:30-68`）。
- 精英卡（Hero）另有一条装配链：`HERO_CLASSES` 表（`card_mechanics.py:1562-1579`）+
  `apply_hero_overlay(entity, bs)` 换 holder 并覆写数值（`battle.py:2409-2436`）。
- 特殊实体**刻意绕过** `Entity.__init__`（否则同名机制类会在 `battle_state` 挂载前跑钩子）：
  `AreaEffect`（`battle.py:1727`）、`EvoEffectZone`（`battle.py:1951`）。

#### B.6.2 触发点（钩子调度位置）

| 钩子 | 调度位置 | 说明 |
|---|---|--- |
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

#### B.6.3 机制类全名单（含行号、钩子、关键数值）

**基础机制族**

| 类 | 行号 | 钩子 | 机制 / 关键数值 |
|---|---|---|--- |
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
|---|---|--- |
| `SkeletonKing` | 319 | 灵魂召唤：`min(6 + souls, 16)`，前摇 0.9s，每 0.25s 放 1 只，半径取 `OFFICIAL_OVERRIDES['SkeletonKing']['spawn_radius']`(缺省 3.5)，角度步进 2.399963（`card_mechanics.py:320-352`） |
| `ArcherQueen` | 355 | 隐身斗篷 3.5s，`hit_speed_mult=2.8`，`speed_mult=0.75`（`card_mechanics.py:356-372`） |
| `GoldenKnight` | 375 | 连环突进：`dash_remaining = 10`，每段 5.5 格内最近未突进目标，伤害 `131 × level_scale`，命中公主塔即停（`card_mechanics.py:375-413`） |
| `Monk` | 416 | 禅定 4s，`damage_reduction` 缺省 0.65（`card_mechanics.py:416-432`） |
| `MightyMiner` | 435 | 每局限 2 次；钻地 0.6s；原地炸弹 `130×level_scale`、半径 2.0、延迟 1.0、击退 1.8；瞬移 `x → 18.0 − x`（`card_mechanics.py:435-458`） |
| `LittlePrince` | 461 | 召唤 `ChampionGuard`，落地 `90×level_scale`、半径 1.5、击退 2.0 格（`card_mechanics.py:461-474`） |
| `BossBandit` | 477 | 手雷：每局限 2 次、隐身 1.0s、向身后传送 6 格（clamp 到 `[0.5, 31.5]`）；攻击侧被动冲刺 3.5~6.0 格双倍伤害（`card_mechanics.py:477-504`） |

**M8 Elite17（Hero）族**（`HERO_CLASSES` 表见 `card_mechanics.py:1562-1579`）

| 类 | 行号 | 关键数值 / 要点 |
|---|---|--- |
| `HeroKnight` | 900 | 嘲讽 `tauntRadius` 内敌军 + 护盾 `shieldValue`（`elite17_data` 表） |
| `HeroMusketeer` | 932 | 前方 `frontOffset` 放炮塔，落地 `spawnDamage`、`spawnRadius` |
| `HeroMiniPekka` | 948 | 煎饼进度 `meterSeconds`/`onHitProgress`、`maxMeter`；吃饼等级 `levelsByMeter`，`mult = 1.1**steps`，回复 `healPct` |
| `HeroValkyrie` | 989 | 旋风 `whirlDuration`、`tick`、`radius`、`tickDamage`、`crownMult`；结束冲刺 `dashRange`，禁攻 `forbidAttack` |
| `HeroWizard` | 1051 | 延迟 `flyDelay` 后升空 `flyDuration`、`speedMult`；火球命中处生成 `tornadoRadius/tornadoDuration` 旋风，`tornadoDps` |
| `HeroBowler` | 1102 | 蓄力 `chargeTime` → 迫击炮 `siegeRange/siegeHitSpeed/siegeShots/siegeDuration/siegeDamage/crownMult`，退出还原 `_orig` |
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
|---|---|---|--- |
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

### B.7 法术与区域效果

#### B.7.1 引擎侧「落点解析」链路（`deploy_card` 法术分支）

`deploy_card` 先做**类型无关**检查，再按法术子类分流（`battle.py:2857-2994`）：

| 分支 | 条件 | 行号 | 产物 |
|---|---|---|--- |
| BarbLog 专属 | `card_name == 'BarbLog' and not _from_mirror` → `arena.can_deploy_at(..., is_spell=False)` | 2854-2856 | 非法直接 `return False`（**唯一有部署区限制的法术**） |
| Lightning | 硬编码分支 | 2892-2895 | `_cast_lightning(player, position)`（`battle.py:2783-2804`） |
| 区域持续出兵（墓园类） | `srow['spawn_character']` | 2896-2911 | `radius` 缺省 3000/1000、`duration` 缺省 5000/1000、`interval` 缺省 500/1000，`count = int((duration−initial)/interval)`，黄金角 2.399963 散布，`delayed_spawn` |
| 克隆 | `srow['clone']` | 2912-2924 | 半径内友军 `Troop` 克隆，`clone.hp = 1.0`、`_soul_excluded = True` |
| 瞬发区域法术 | `(buff or controls_buff) and not projectile` | 2928-2939 | `AreaEffect`（觉醒 Zap 走 `EvoZapZone`） |
| 滚动类 | `card_name in ('Log','BarbLog')` | 2945-2961 | 直接从落点生成 `LogProjectileRolling` / `BarbLogProjectileRolling`，方向强制 `(0,±1)` |
| 弹道波次 | `card_info.projectiles` | 2941-2994 | 从己方国王塔 `delayed_spawn` 出弹，`wave_interval` 间隔；觉醒 GoblinBarrel 追加诱饵 |

**法术没有边界/阻挡校验**：`card_info.type != 'spell'` 的部署区判定整段被跳过（`battle.py:2865`），
BarbLog 以外的法术不调 `can_deploy_at`。⇒ 直接调用 `deploy_card` 时越界坐标不会被拒（见 B.9 #6）。

#### B.7.2 `AreaEffect`（Zap/Freeze/Heal/Rage/Tornado/Earthquake/Poison）

构造（`battle.py:1728-1783`）：

| 参数 | 取值/缺省 | 行号 |
|---|---|--- |
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
|---|---|--- |
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

#### B.7.3 弹道法术的命中与塔伤

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

#### B.7.4 Lightning

`_cast_lightning`（`battle.py:2783-2804`）：半径 `spells['Lightning'].radius`（3500 → 3.5）；
候选 = 半径内非 `Projectile/SpawnProjectile/AreaEffect` 敌方实体；按 `data.hp` 降序取前 3；
**塔伤 ×0.65**（`battle.py:2802`）；部队附 0.5s 眩晕 + 重索敌（`battle.py:2803-2804`）。
伤害取 `projectiles['LighningSpell']`（注意源码里该键名拼写为 `LighningSpell`，`battle.py:2788`）。

#### B.7.5 `TimedExplosive`（Balloon / GiantSkeleton 亡语炸弹）

`dsd.range = 3.0` 硬编码、`crown_tower_damage_percent` 来自 `deathSpawnCharacterData.crownTowerDamagePercent/100`
（`card_utils.py:497-505`）；命中判定为 `中心距 − 碰撞半径 < range`（`battle.py:2391`）；
降伤条件写成 `entity.name in ('King_PrincessTowers', 'KingTower')`（`battle.py:2392`）——
即**塔兵变体（`King_CannonTowers` 等）不在元组里，不吃降伤**，与 B.7.3 的 `"King" in name` 口径不一致。

#### B.7.6 `spell_module.py`：法术知识查询服务（三层 API）

| 层 | 函数 | 行号 | 口径 |
|---|---|---|--- |
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

#### B.7.7 塔伤修正汇总

| 位置 | 修正 | 行号 |
|---|---|--- |
| `AreaEffect._pulse` | 名字含 `King`/`PrincessTower` → `damage_per_tick × crown_pct` | `battle.py:1819-1822` |
| `AreaEffect._pulse` | 普通建筑 → `× building_mult`（Earthquake `building_damage_percent=350` → ×4.5） | `battle.py:1823-1824`（说明见 `battle.py:1775-1776`） |
| `AreaEffect.update` | Rage 对建筑 `×0.3` | `battle.py:1857-1860` |
| `Projectile._deal_splash_damage` | 名字含 `King` → `× crown_tower_percent` | `battle.py:1614` |
| `Projectile`（Arrows 特例） | `ArrowsSpell` → `crown_tower_percent = 25/122` | `card_utils.py:461-465` |
| `Lightning` | 塔 `×0.65` | `battle.py:2802` |
| `TimedExplosive` | 仅 `('King_PrincessTowers','KingTower')` 吃降伤 | `battle.py:2392-2395` |
| `BasicCharacter.on_attack` | 普攻对塔 `× data.tower_damage_mult`（名字含 `King`/`PrincessTower`） | `core.py:47-56` |
| `deal_area_damage` | 名字含 `King` → `× crown_tower_damage_percent` | `battle.py:3257` |

#### B.7.8 命中判定公式对照

| 实现 | 公式 | 行号 |
|---|---|--- |
| `AreaEffect._in_radius` | `中心距 ≤ radius + 目标碰撞半径` | `battle.py:1785-1786` |
| `Projectile._deal_splash_damage` | 同上 | `battle.py:1611-1612` |
| `spell_module.evaluate_cast` | 同上 | `spell_module.py:273` |
| `Lightning` | 同上 | `battle.py:2797` |
| `GenericBomb` | 同上（另加 `hits_air/hits_ground`） | `battle.py:1920-1923` |
| `TimedExplosive` | `中心距 − 目标碰撞半径 < range(3.0)` | `battle.py:2391` |
| `deal_area_damage` | `edge_distance_from(position) < range`（塔=矩形边缘，普通=中心距−半径） | `battle.py:3250-3265` |
| 单位普攻射程 | `edge_distance_from ≤ range (+0.5 公主塔)` | `battle.py:586-604` |

---

### B.8 部署合法性

#### B.8.1 坐标契约

`SubAction(x, y)` 一律是**玩家本地网格坐标**，唯一换算入口 `sub_position`：
玩家 0 → `(x+0.5, y+0.5)`；玩家 1 → `(17.5−x, 31.5−y)`（`rl/action_bundle.py:31-39`）。
`legal_cells` 与 `validate_bundle` 走同一入口，避免「掩码世界坐标 vs 提交镜像坐标」分裂
（`rl/action_mask.py:6-9, 462, 473, 563`）。

#### B.8.2 `legal_cells` 判定流程（逐格）

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
|---|---|---|--- |
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

#### B.8.3 `slot_mask` / `validate_bundle`

- `_slot_playable`：`king_tower_hp > 0`、卡在 `cycle[:4]`、费用可算、`elixir >= cost`
  （`action_mask.py:46-56`）；`_card_cost`：Mirror = `Card(last_card).elixir + 1`，无 `last_card` → `None`
  （`action_mask.py:30-36`）。
- `validate_bundle`（`action_mask.py:521-573`）：逐子动作——技能查 `ability_mana` 与圣水；slot 0 跳过；
  越界/重复拒绝；`_position_legal` 拒绝；模拟扣费；最后 `solo_commit_blocked` 兜底
  （单张高承诺卡无圣水优势 → 拒绝整包，`action_mask.py:568-572`）。
- `slot_mask` 返回 `(K_MAX=4,)` 布尔（`action_mask.py:59-68`；`K_MAX = 4` 见 `rl/action_bundle.py:28`）。

#### B.8.4 与引擎 `deploy_card` 的差异

引擎侧（`battle.py:2806-2887`）与掩码逐条对照：

| 校验项 | 引擎 | 掩码 | 是否一致 |
|---|---|---|--- |
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
|---|---|---|--- |
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
|---|---|--- |
| `Zap` / `Poison` / `Tornado` / `Earthquake` / `Freeze` / `Heal` | False | True |
| `Arrows` / `Fireball` | True | True |
| `Lightning` / `Rage` / `BarbLog` / `Log` / `GoblinBarrel` / `Mirror` | False | False |

⇒ 「伤害型法术才受空砸闸门约束」这条规则在掩码层对 `Zap/Poison/Tornado/Earthquake` 实际**未生效**。

#### B.8.5 禁放区差异（按卡类）

| 卡类 | 禁放区 |
|---|--- |
| 部队（非 Miner） | 己方半场（P0：`y ∈ (1,21)` 内、且 `y ≥ 15` 时要求对应敌方公主塔已破）；底边 `y ≤ 1` 仅 `6 < x ≤ 12`；引擎与掩码同式（`battle.py:2872-2879`；`action_mask.py:411-422`） |
| Miner | **全场**（含敌半场），仅受塔/建筑占位与边界限制（`battle.py:2871-2872`）；掩码未实现该例外 |
| 建筑 | 部队规则 + 王塔身后 1 格宽禁区 `(7,0)-(11,1)`（P0）/ `(7,31)-(11,32)`（P1）（`arena.py:86-98`；`battle.py:2869`；`action_mask.py:409`） |
| 普通法术 | 无部署区限制、无占位限制；掩码额外挡「已毁敌方塔本体（半径 0.9）」与空砸/纯砸塔 |
| BarbLog | 引擎=部队部署区（`is_spell=False`）；掩码**无此限制** |
| 法术（墓园/克隆/滚木/弹道波次） | 引擎按分支处理，无位置校验 |

---

### B.9 本部分待确认清单

| # | 条目 | 无法确认的原因 | 要确认需要什么 |
|---|---|---|--- |
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
| 1 | 半格寻路网格的坐标变换（`position_to_cell` / `cell_to_position` 的具体公式） | 定义在 `pathfinding_heap.py`，不在本轮允许阅读的源码清单内；`battle.py` 只显示网格为 36×64（battle.py:3033-3036），`battle.py:4` 仅导入函数名 | 允许阅读 `pathfinding_heap.py`（或让作者给出该文件的行号引用）；可先看 `docs/_survey/parts/G037.md` 的记载再回源码核验 |
| 2 | `TileGrid.tile_size = 100.0` 的用途 | 该常量在 `arena.py`/`battle.py` 内定义后无任何读取点（本目录 `*.py` grep 仅命中 arena.py:10） | 全仓（含渲染层 `new_visualization.py`）搜 `tile_size` |
| 3 | `arena.can_deploy_at(is_spell=True, spell_obj=...)` 的行为 | arena.py:181 调用的 `self._is_rolling_projectile_spell(...)` 在 `arena.py`（209 行）内未定义，全仓亦未找到 Python 定义；当前唯一调用方传 `is_spell=False`（battle.py:2855），该分支从未执行 | 查 `spell_module.py:261` 的调用参数（不在本轮清单）；或让作者确认该方法是否被删除 |
| 4 | ~~部署合法性由哪一层兜底~~ **（已由本文档 §B.8 回答，2026-09-14 独立校验后更正）** | `battle.deploy_card` 的部队/建筑路径**不**检查 `BLOCKED_TILES`/河道（battle.py:2865-2887），只有 `arena.can_deploy_at` 检查（arena.py:176）；**RL 掩码层补齐情况见 §B.8**（`legal_cells`/`_position_legal` 逐格流程 + 与引擎 `deploy_card` 的逐项对照，含"掩码更严：无 Miner 例外"）；逐格双向对账见 `docs/mask_vs_engine_reconcile_2026-09-14.log` | 已完成（§B.8） |
| 5 | 「bottom half / top half」注释与 y 数值方向的对应 | arena.py:150/arena.py:161 注释称玩家 0 为 bottom、玩家 1 为 top，而数值上玩家 0 的区间是 `y=1..14`（小 y 侧，arena.py:152）；坐标是否在渲染层翻转需 `new_visualization.py` | 允许阅读 `new_visualization.py`（或给出其坐标变换行号） |
| 6 | 塔兵为 Cannon/Knife/Chef 时国王塔激活语义 | 只有 `name == 'King_PrincessTowers'` 的公主塔阵亡才激活同方国王塔（battle.py:3149-3154），塔兵换卡后 name 变为该卡卡名（card_utils.py:57-61、card_utils.py:228）⇒ 该路径不触发 | 官方机制口径（源码内无注释说明是否为有意），或补一条针对塔兵卡名的分支/测试 |
| 7 | 180 秒皇冠判定的规则来源 | 源码只有数值 `300 > self.time >= 180`（battle.py:2667）与其后注释（battle.py:2677-2679），未标注对应真实 CR 规则或数据出处 | 作者/设计文档口径，或对局录像对拍 |
| 8 | 对手出牌失败是否被吞掉 | `opponent_action` 丢弃 `deploy_card` 返回值（environment.py:72），失败（费用不足/位置非法）静默忽略 | 确认 RL/脚本对手层是否有独立合法性检查（不在本轮清单） |
| 9 | 国王塔逐级 HP 数组与「4824」的确切出处 | 本轮只能确证：`king_tower_stats.hitpoints = 2100`（card_utils.py:40）+ 经 `buildings` 表按稀有度轴取值（card_utils.py:366-367）+ 与 `player.py:6` 硬编码 `(4824, 3052, 3052)` 一致；`cards_stats_building.json` 是数据文件，非 `.py` 源码 | 允许直接读 `gamedata.json` / `cards_stats_building.json` 的 `KingTower`、`PrincessTower` 行，或给出引用该表的源码行 |
| 10 | 引擎是否被 RL 侧以外的方式驱动（决策帧、动作槽位语义） | 本文只覆盖 `CREnv`（5 槽手牌、格心动作，environment.py:94、environment.py:154）；训练用的 `rl/` 环境不在本轮清单 | 读 `rl/env_wrapper.py`、`rl/action_mask.py`（不在本轮清单） |
| 11 | `PlayerState` 默认塔血与 `card_level` 不一致的窗口 | `tower_hps` 默认 `(4824, 3052, 3052)` 是 lv11 值（player.py:6），而 `BattleState(card_level=...)` 可为 11-16；在首次 `step` 覆盖之前（battle.py:2654），`can_play_card` 读的是默认值（player.py:39） | 确认是否存在「首次 step 前读 `*_tower_hp`」的调用路径（`rl/` 不在本轮清单） |
| 12 | 多战斗并行的等级全局态安全 | `Card.default_level` 是类变量，由 `BattleState.__init__` 与镜像窗口改写（card_utils.py:220、battle.py:2541、battle.py:2706-2714）；源码注释声明基于单战斗串行假设（card_utils.py:218-219），并行场景的后果未在源码中给出 | 官方并行/多进程方案的说明，或一条并发回归测试 |

### 来源：B.9 本部分待确认清单

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

