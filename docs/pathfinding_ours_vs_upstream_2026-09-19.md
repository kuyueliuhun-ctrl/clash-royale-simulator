# 寻路算法对照：原作者最新版 vs 我们（+ FL 回放拟合度实验的前置核实）

> **取数时间**：2026-09-19 07:30–08:20 CST，**只读核查**（读文件 + 只读 `python3 -B` 探针；未改任何仓库文件、未跑训练、无 git 写操作）。
> **A（我方）** = `/mnt/e/clash-royale-simulator-main` @ `8d2552a`；**B（上游最新）** = `/mnt/e/clash-royale-simulator-main-by-jason` @ `f616f19`。
> 共同祖先（分叉点）= `f20fa4d`。本文是 [`upstream_engine_delta_2026-09-19.md`](upstream_engine_delta_2026-09-19.md) 的**专题下钻**（那份给"上游改了什么/我们什么状态"，这份给"两套寻路到底怎么不一样"）。
> **探针原文留证**：`/tmp/pf_ours.txt`、`/tmp/pf_up.txt`（本机实跑，脚本 `/tmp/pfprobe.py`，全文贴在 §4）。

---

## §0 结论先行

**两套寻路的「骨架」完全相同（同一份 `pathfinding_heap.py` 血统），差异集中在 4 处：目标格生成、代价表、河面可走性、以及 battle.py 里的路径跟随/节流。**

| # | 层 | 上游最新 | 我们 | 观测到的行为后果 |
|---|---|---|---|---|
| **D1** | **目标格生成（goal）** | `radius = target.data.collision_radius + entity.range`，用**中心距** `distance_to(target_position) < radius + 0.375` 筛格；`scan_radius = ceil(radius*2)+1`；**无兜底**（`goals` 空 ⇒ `min(())` **ValueError**） | 用 `target.edge_distance_from(pos)`（**塔=矩形边缘**、其余=圆边缘），`scan_radius` 由 `_target_footprint_radius()`（塔取矩形半轴）算；**三级兜底**（无解时退"最近可达格"、再退"目标格"） | **同一场景可以给出不同的停靠点**：§4 的 B 场景 A\* 路径 **上游 24 点 / 我们 34 点**、goal 分别是 `(18,42)` 与 `(18,52)`；A 场景（近战打公主塔）**逐字节相同** |
| **D2** | **代价表** | `W: 7(空中或有 jump_speed) / 50(地面)`；**`'.'`（桥面）= 8**；其它 = 5 | `W: 7(空中) / 800(地面)`；**`'.'` = 5**（= 半场地面价，注释写明"旧值 8 会把单位吸向桥心、抹平车道"） | **桥面/河面语义完全不同**（见 D3）；我们桥面与地面同价（三车道），上游桥面比地面**贵 60%** |
| **D3** | **河面可走性** | `arena.BLOCKED_TILES` **已删**河面条目、`is_walkable` **已删**桥面分支 ⇒ **河面全部可走**（地面涉水价 50，桥面 8） | 与分叉点**逐字相同**：河面 32 格全封、只有 `x∈[2,5)∪[13,16)` 的桥面可走（`is_walkable` 的桥面分支 + `BLOCKED_TILES` 双重把关） | 上游地面单位**可以涉水**（贵但能走）；我们**只能走桥** |
| **D4** | **路径跟随 / 节流（battle.py）** | 目标扫描 `tick%2` 复用；寻路重算 `tick%3`（20 Hz ⇒ ≈0.15 s）；**无卡死自救**；HEAD 的过河跳跃**整段注释（死代码）** | **每帧都重扫目标**；寻路重算 `tick%10`（60 Hz ⇒ ≈0.167 s，墙钟与上游基本一致）；**有卡死自救**（`battle.py:1248-1263`，"桥头楔死 60 s"实证）+ **队形车道偏移 `_lane_offset`**；过河跳跃是**活代码**（并带 §5-P3 的闩锁缺陷） | 上游若卡在桥头/墙角**没有自救**；我们有；我们每帧重扫目标 ⇒ 目标切换更灵敏（也**更贵 6 倍**） |
| **D5** | **命中几何（`in_attack_range`）** | `center_dist ≤ range + target.collision_radius + bonus`（**塔用圆形碰撞半径**：公主 1.0 / 王塔 1.4） | `target.edge_distance_from(pos) ≤ range + bonus`（**塔用矩形**：公主 3×3 half 1.5 / 王塔 4×4 half 2.0） | **同一射程在塔前差 0.5~1.0 格** ⇒ 停靠点与开火时机不同 ⇒ **终局塔血必然不同**（正是 FL 实验要量化的那类差异） |
| **D6** | **文件面** | `pathfinding.py`（旧版，130 行）**仍在**，但 `battle.py` 只 import `pathfinding_heap` ⇒ **死件** | `pathfinding.py` **已删**（Tier 0 死件清理，`git diff --numstat f20fa4d..8d2552a` = `0 130`） | 无功能影响（两边都是死件/已删） |
| **D7** | **`walkable_cache`（两个引擎都有）** | `arena.py:5` 模块级全局 + `int(x)` 向零截断 + **先查缓存后判越界** ⇒ 越界查询会污染相邻合法格 | **完全相同**（`arena.py:5/106-119`，逐字同构） | 见 §5-P4：**双方共有的同源缺陷**，与寻路算法选择无关 |

**一句话**：**骨架同源、几何口径不同**。真正会造成"路径/停靠点不一样"的是 **D1（目标格）** 与 **D5（命中几何）**；造成"能不能过去"的是 **D3**；**D2/D4** 是代价与调度的口味差异。

---

## §1 文件与血缘

| 文件 | 我方 | 上游 | 分叉点 |
|---|---|---|---|
| `pathfinding_heap.py` | **8,320 B / 178 行** | 5,505 B / ~105 行 | 两边同源；`git log --diff-filter=A` 溯源到共享提交 `114939f`「Use heap to manage open set nodes…」 |
| `pathfinding.py` | **已删**（`git diff --numstat f20fa4d..8d2552a` = `0 130`） | 5,120 B（**死件**：`battle.py` 只 import `pathfinding_heap`） | 分叉点两边都有 |
| 分叉后我方改动 | `pathfinding_heap.py` **+54 / −11** | — | — |
| 分叉后上游改动 | — | `pathfinding_heap.py` **+4 / −1**（只有 `2daab60` 的 `W` 代价那 5 行） | — |
| 调用方 | `battle.py:4` `from pathfinding_heap import EntityPathfinder, position_to_cell, cell_to_position` | `battle.py:3` 同一行 | 一致 |

⇒ **骨架没有分家**：网格、邻居、启发式、A\* 主循环、路径重建两边**逐字相同**（我已逐段比对）。差异全在 §0 的 D1–D5。

---

## §2 逐层对照（代码级）

### 2.1 网格与邻居（**完全相同**）

```python
# 两边逐字相同
def position_to_cell(position): return math.floor(2*x), math.floor(2*y)   # 半格分辨率 36×64
def cell_to_position(cell):     return Position((x+0.5)/2, (y+0.5)/2)
def get_neighboring_points(x,y): 8 邻域, 边界 36/64
def heuristic(self, cell):      return 10 * max(abs(x-gx), abs(y-gy))     # 切比雪夫 ×10
```

### 2.2 目标格生成（**D1，差异最大**）

```python
# ---- 我们（pathfinding_heap.py:65-106）----
scan_radius = math.ceil((self._target_footprint_radius() + edge_radius) * 2) + 2
edge_radius = self.entity.data.range
    edge_dist = self.target.edge_distance_from(pos)          # 塔=矩形边缘；其余=圆心距−r
    if edge_dist < edge_radius + 0.375 and self.battle.pathfind_ground_walkable(...): goals.add(cell)
if not self.goals:  # 三级兜底
    ... 最近可达格 ... else: goals.add(target_cell)
self.goal = min(self.goals, key=lambda c: target.edge_distance_from(c) + dist_to_start(c))

# ---- 上游（pathfinding_heap.py:63-80）----
radius = self.target.data.collision_radius + self.entity.data.range
scan_radius = math.ceil(radius*2) + 1
    distance = cell_to_position((x,y)).distance_to(self.target_position)   # 用【目标中心】
    if distance < radius + 0.375 and self.battle.pathfind_ground_walkable(...): self.goals.add((x,y))
self.goal = min(self.goals, key=lambda c: dist_to_target_center(c) + dist_to_start(c))   # 无兜底
```

三点实质差别：

1. **`0.375` 裕量的基准**：上游加在 `collision_radius + range` 上（圆口径）；我们加在 `range` 上（**已扣掉目标半径**）⇒ 对**普通单位**两者等价（`d ≤ r_t + range + 0.375` ⟺ `d − r_t ≤ range + 0.375`），
   **但对塔不等价**：塔在我们这里是**矩形**（half 1.5 / 2.0），上游是**圆**（1.0 / 1.4）⇒ 我们的"可达环"离塔心**远 0.5~0.6 格**（公主）/ **0.6 格**（王塔）。
2. **`scan_radius` 的基准**：我们取 `footprint_radius`（塔取矩形半轴 `max(hw,hh)` = 1.5/2.0），上游取 `collision_radius`（1.0/1.4）⇒ 我们扫描窗更大（≥2 格），**且多出 `+2`**。
3. **兜底**：上游 `goals` 为空时 `min(())` **抛 `ValueError`**（我们实测过 `min` 空集的行为）；我们有三段兜底 ⇒ **上游在"目标周围全被堵死"的近战场景会直接崩**（未实测触发，见 §6）。

### 2.3 代价表（**D2**）

| `tile_char` | 我们 | 上游 | 说明 |
|---|---|---|---|
| `'W'`（水） | 空中 **7** / 地面 **800** | 空中**或 `jump_speed`** **7** / 地面 **50** | 我们河面不可走 ⇒ 800 基本是死配置（**除 §5-P2 的 16 个桥面 sliver 半格**）；上游河面可走 ⇒ 50 是"涉水税" |
| `'.'` | **5**（注释：旧值 8 会把单位吸向桥心、抹平车道，2026-09-10 用户机制） | **8** | 上游桥面比地面贵 |
| 其它 | **5** | **5** | — |
| 对角 `geo_cost` | 14 / 直行 10 | **相同** | — |

### 2.4 河面可走性（**D3**）

```python
# 我们 arena.py:21-34 + 106-119（＝分叉点原样）
BLOCKED_TILES = [(0,15),(0,16),(1,15),(1,16), *[(i,j) for i in range(5,13) for j in range(15,17)],
                 (16,15),(16,16),(17,15),(17,16), ...]
elif self.RIVER_Y1 <= pos.y <= self.RIVER_Y2:
    on_left_bridge  = 2.0  <= pos.x < 5.0
    on_right_bridge = 13.0 <= pos.x < 16.0
    walkable_cache[int_pos] = on_left_bridge or on_right_bridge

# 上游 arena.py（2daab60 之后）
BLOCKED_TILES = [ ... 只剩上下两行的栅栏 ... ]          # 河面条目 8 行已删
def is_walkable(...):                                    # 无河面分支
    if not valid or blocked: False
    else: True
```

⇒ **上游把"河"从一条墙改成了一条贵路**（地面涉水 `50×10=500` 每格 vs 桥面 `8×10=80`），
我们用"墙 + 桥"。
**两条路线的可分辨后果**：当**桥被堵**（塔/建筑/单位）时，上游单位会**绕行或涉水**，我们单位会**卡住并触发自救**（我们把自救写在 `Troop.update`，见 D4）。

### 2.5 A\* 主循环与路径重建（**完全相同**）

`g/f/parent/closed_set + heapq`、`if current_f > f[current]: continue`、`if current == self.goal: break`、
`path = [current]` 回溯到 `start_cell` —— 两边逐字相同。**（注：`current` 在 `break` 时就是 goal；若堆先空则不 break，`current` 是最后弹出的节点，回溯链仍完整。）**

### 2.6 路径跟随（`Troop.update`，**D4**，差异在 battle.py 不在 pathfinding）

| 项 | 我们 | 上游 HEAD |
|---|---|---|
| 目标扫描 | **每帧** `self.update_current_target()`（`battle.py:1191`）+ 我们的 `_hero_taunt_override`/机制钩子 | `tick%2` 复用（`:283-292`），注释写明"为了省时间，最多 0.1 s 一次" |
| 寻路重算 | `in_sight_range` 且 `tick%10==0`（`:1245`） | `in_sight_range` 且 `tick%3==0`（`:303`） |
| 路点选取 | `min_point`（离自己最近的路点）+ 起点向量点积判"是否已越过" ⇒ `index+1`；**再叠 `_lane_offset` 法线偏移**（`:1280-1287`） | 同样的 `min_point`+点积 `index+1`；**无车道偏移** |
| 卡死自救 | **有**：每 0.5 s 检查位移 <0.1 ⇒ 丢弃当前路点、必要时整条重算（`:1248-1263`，注释记"一对弓箭手楔死在桥头 60 s"） | **无**（`grep stuck` = 0 命中） |
| 过河跳跃 | **活代码**（`:1187-1190 / 1233-1239`） | **整段注释**（触发块 + 复位块），只剩 waypoint 里的死分支（`jumping_across_river` 再无置 True 处） |
| 空中单位 | `if self.data.is_air_unit: move_towards(target)`（相同） | 相同 |

### 2.7 命中几何（`in_attack_range`，**D5**）

```python
# 我们 battle.py:641-653
dist = target.edge_distance_from(self.position)      # 塔=到矩形最近点；其余=圆心距−r
return dist <= range(+override) + bonus(公主塔 +0.5)

# 上游 battle.py:117-123
return self.position.distance_to(target.position) <= self.data.range + target.data.collision_radius + bonus
```

对普通圆形单位两者**代数等价**；**对塔不等价**（矩形 half 1.5/2.0 vs 圆 1.0/1.4）。
⇒ **贴塔的停靠点不同**，进而**开火时机不同**，进而**终局塔血不同** —— 这正是 FL 回放实验的判别对象。

---

## §3 我方独有的两级"抗卡死/车道"设施（上游没有）

这两条写在 `battle.py`，**不是路径算法本身**，但会显著改变"最终位置/塔血"：

1. **卡死自救**（`battle.py:1248-1263`）：每 0.5 s，若 `path` 存在且位移 < 0.1 格 ⇒ 删掉最近路点；路点删光则整条重算。注释留证："一对弓箭手楔死在桥头，60 s 纹丝不动"。
2. **队形车道偏移**（`battle.py:1277-1287`）：把路点沿"本单位→路点"的法线平移 `_lane_offset`，让同卡多单位保持部署时的横向车道并排推进。

上游两者皆无 ⇒ **上游在多单位挤窄桥/墙角时可能永久停住**；我们在同一局面会绕开。
⇒ **FL 实验里若某局出现"挤在桥头"的形态，两引擎的差异会非常大**（上游停住 / 我们继续），这类局**不能算作对某一方"更拟合"的证据**，要单列。

---

## §4 实测对照（同一输入、两台引擎、只读探针）

**探针脚本**：`/tmp/pfprobe.py`（85 行，ASCII 输出；同一份脚本两边跑）。
**上游运行姿势（本次新查明）**：

```bash
# 上游引擎需要 fastcore（两个解释器都没有）⇒ 装到隔离目录，不动系统环境
/usr/bin/python3 -m pip install --quiet --target /tmp/deps_upstream fastcore   # 实测成功 fastcore 2.2.28
cd /mnt/e/clash-royale-simulator-main-by-jason/src/clasher_new   # ← 上游 card_utils 用相对路径 open('gamedata.json')，cwd 契约仍在
PYTHONPATH=/tmp/deps_upstream PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B /tmp/pfprobe.py upstream
# 我方
cd /mnt/e/clash-royale-simulator-main/src/clasher_new
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B /tmp/pfprobe.py ours
```

### 4.1 场景 A（蓝 Knight 于 `(3.5,9.5)` → 红左公主塔 `(3.5,25.5)`）：**逐字节相同**

```
【两边输出完全一致】
A goal=(7, 46) len=28
A path=[(3.75,9.75), (3.75,10.25), ..., (3.75,23.25)]      # 28 点，全程 x=3.75 直上（左桥）
MARCH t=0.5..8.0   x=3.500→3.750  y=9.500→17.859  hp=1766  （16 个采样点全部一致）
```

### 4.2 场景 B（蓝 Musketeer 于 `(9.5,9.5)` → 红 KingTower `(9.0,29.0)`）：**分叉**

```
上游: B goal=(18, 42) len=24
      B path=[(9.75,9.75), (10.25,10.25), ..., (13.25,13.25), (13.75,13.75), (13.75,14.25),
              (13.75,14.75), (13.75,15.25), (14.25,15.75), (14.25,16.25), (13.75,16.75),
              (13.25,17.25), ..., (9.25,21.25)]
我们: B goal=(18, 52) len=34
      B path=[(9.75,9.75), (9.75,10.25), (9.75,10.75), (9.75,11.25), (10.25,11.75), ...,
              (13.75,15.25), (13.75,15.75), (13.75,16.25), (13.75,16.75), ..., (9.75,25.75), (9.25,26.25)]
```

- **goal 不同**：上游 `(18,42)` = `Position(9.25,21.25)`（离王塔约 7.75 格）；我们 `(18,52)` = `Position(9.25,26.25)`（离王塔约 **2.75** 格）⇒ 我们停在**更靠近塔**的位置，上游在**更远处**就认为"到位"。
- **路径形态不同**：上游先斜穿到 `x=13.25`（右车道）再折回 `x=9.25`；我们先直上再在 `y≈24` 才折回。
- 两者都走了**右桥**（`x∈[13.5,15.5]` 的 deck 半格），说明 D3 的河面差异在本场景**没有触发**（桥没被堵）。

### 4.3 由实测得到的 3 条**可预期**结论（待 FL 实验验证）

1. **近战打塔**这类"目标就在正前方、路径是直线"的场面，两引擎**高度一致**（A 场景逐字节相同）⇒ **FL 实验若只看这类局，会判"没差别"**。
2. **中远程/需要绕行**这类场面，**停靠点会差 1~4 格**（B 场景），开火时机随之不同 ⇒ **终局塔血会分开**。
3. 差异**不是"谁更像真机"**，要由 FL 数据裁决；而且 **D5（命中几何）与 D1（目标格）会耦合**：即使 A\* 路径相同，**开火开始的时间点**也可能因为 `in_attack_range` 口径不同而不同。

---

## §5 FL 回放拟合度实验：前置核实（本次已实测的部分）

用户提议的方法：**FL 人类回放里已有"最终数据"，把同一局分别灌进两台引擎，看谁的终局数据更接近回放记录**。
这一节只做**前置核实**，把"能不能跑、跑得动多少、会混淆什么"钉死；**判据见 §6（先写死再跑）**。

### 5.1 已核实「能跑」（本次实测）

| 项 | 结论 | 依据 |
|---|---|---|
| 上游引擎可执行 | **可以**：装 `fastcore` 到隔离目录即可（`pip install --target`），无其他缺件；`BattleState` 构造 / `deploy_card` / `step(dt)` 全部跑通（600 tick 无异常） | §4 的运行姿势 + 探针实测 |
| 上游 cwd 契约 | **仍需要** cwd = `src/clasher_new`（`card_utils.py` 用 `open('gamedata.json')` 相对路径）；我方已取消该契约（可在仓库根跑） | 上游 `card_utils.py:5`；我方 Tier 3-2 |
| 卡表覆盖 | **上游 150 key ⊆ 我们 212 key**（多出的 62 个是 `*_EV1` 觉醒体 / Hero / 派生体）⇒ **同一副卡组若含觉醒或英雄卡，上游根本没这张牌** | `card_utils.card_data` 两侧计数（150 / 212），`comm -23` = 62 |
| 两引擎 API 同构度 | `BattleState(PlayerState(0,deck,elixir), PlayerState(1,deck,elixir))` + `deploy_card(side, card, Position(x,y))` + `step(dt)` **两边同签名**（上游无 `card_level`） | 探针同一份脚本两边跑通 |
| FL 回放提供什么 | 逐事件 `(tick, side, card, position)` + 牌组 + **终局结果**（victory/defeat/draw）+ **`final_tower_hitpoints{king,left,right,total}`** | 见 [`replay_format_comparison_2026-09-19.md`](replay_format_comparison_2026-09-19.md) §2.13（10,852 局穷举） |

### 5.2 已核实的**混淆与硬约束**（必须在判据里交代）

1. **逐事件圣水不可观测**（已确证缺口）：FL 回放不带每步圣水，直接重放只走通 **35–62%**，主因 `unaffordable`。
   ⇒ 若照"合法性"跑，**"接受率"本身**会变成一个指标，而它与寻路无关（会污染"谁更拟合"）。
   ⇒ 三种做法要**先选一种并写死**：(a) **强制注入**同一动作表（绕过圣水/手牌，只保留落点合法性）；(b) 只统计**两边都接受**的事件子集；(c) FL 式状态重建（编造相容牌序 → 与真引擎发牌校准）——**成本最高**，且会把"发牌模型"的误差混进来。
2. **时长口径**：FL 回放时长 **300.55–311.55 s**，**超过我们 300 s 的步数上限**；我们的引擎常在 180–223.5 s 就因皇冠规则结束。⇒ **"终局"的定义两边不同**，必须先固定成"回放记录的终局时刻"或"第一个结束者"。
3. **卡表差**：含觉醒/英雄的牌组**上游无法构造**（§5.1）⇒ 采样时必须**报出"只跑两边都支持的牌组"的占比**，否则分数的可比性不成立（【R17】）。
4. **上游 20 Hz vs 我们 60 Hz**：上游 `environment.py` 用 `10×1/20`，我们是 `30×1/60`（决策时长都是 0.5 s），但**引擎内部 tick 粒度不同** ⇒ 位移/冷却的离散化误差不同。这**不是寻路差异**，要**先声明它算"引擎整体差异"还是"要排除的混淆"**。
5. **D7（`walkable_cache` 污染）两边都有** ⇒ 它不会偏向任何一方，但它会让**同一进程里后面的局**受前面的局影响 ⇒ **必须每局新进程，或每局清 `arena.walkable_cache`**（否则分数不可复现）。
6. **我们的自救 / 车道设施（§3）上游没有** ⇒ "挤在桥头"的局两边差异巨大；这类局要**单列**，不能算作对某方"更拟合"的证据。

---

## §6 预注册（**判据先写死，再跑**；【R3】）

> 本节只是把判据写下来；**尚未启动实验**。判据分三档，失败分支也写死。

**主口径（必须三选一，先定）**：`REPLAY_MODE = force | both-accepted | reconstruct`（§5.2-1）。
**样本**：FL 回放中"**两侧引擎都支持其牌组**"的局（觉醒/英雄牌组剔除，占比要报）。
**仪器隔离**：每局一个**新进程**（或每局清 `arena.walkable_cache`），同一台机器、同一解释器（WSL `python3 3.14.4`）。

| 指标 | 定义 | 判据（写死） | 失败分支 |
|---|---|---|---|
| **M1 终局塔血误差** | 对每局算 `Δ = Σ|engine_final_tower_hp − replay_final_tower_hp|`（`king+left+right` 三个分量的绝对差之和，单位 HP） | **配对比较**两引擎的 `Δ`：用 **Wilcoxon 符号秩**（同一局的配对），报中位数差与 95% 区间；**若区间跨 0 ⇒ 判「本样本下不可分辨」** | 若某引擎的裁决局数 < 总样本 60%（大量局在开始时就被拒）⇒ **整批作废**，先修 replay 驱动，不得用剩余局下结论 |
| **M2 胜负一致率** | `engine_result == replay_result`（victory/defeat/draw 三分） | 报两引擎的一致率与**配对 McNemar**；差异 < 5 pp ⇒ 判「不可分辨」 | 若两引擎的 `draw` 占比都 > 30%（与回放不符）⇒ **判「步数/裁决口径没对齐」**，先修口径 |
| **M3 时长误差** | `|engine_end_time − replay_end_time|` | 仅作**描述性**，**不作判据**（§5.2-2 的口径差异未解决前） | — |
| **M4 位置轨迹（可选、只读）** | 用回放的 `(tick, side, card, position)` 与引擎内该实体的落点逐事件比对 | 只报**分布**，**不作判据**（FL 落点是真机坐标，我们的坐标系换算已在别处验证过 4001/4001） | — |

**阳性对照（必须先过）**：**同一台引擎跑两遍**（同 seed 同输入）⇒ M1/M2 **必须逐位相同**（若不同 ⇒ 驱动里有非确定性，先修驱动）。
**阴性对照**：把**上游引擎的 D1/D5 关掉**（改成与我们同口径）⇒ M1 应该**变差或不变**；若"关掉差异后反而更拟合"，说明差异方向与真机**相反**（这本身就是有价值的结论）。

**写作禁则**（沿用本仓纪律）：样本量不足 ⇒ 只写"观察到 X / 不可分辨"；**不得**写"我们的寻路更好"；跨引擎数字**不得**与历史 run 的数字对拍（【R17】）。

---

## §7 未验证 / 未做

1. **未启动 FL 拟合实验**：本文只做前置核实与判据预注册，**一个回放都还没跑**（§5 的"能跑"是**引擎冒烟**，不是回放）。
2. **上游 `goals` 空集的崩溃**只做了**代码阅读**（`min(())` ⇒ `ValueError`），**未构造出触发局面**。
3. **D1/D5 的相对贡献未分离**：§4.2 的路径不同**同时**有 D1（goal 不同）与 D5（命中几何不同）的份，本文**没有做单变量拆解**（要拆需要在探针里分别替换两处口径，属改代码，未做）。
4. **上游 20 Hz 的影响未隔离**（§5.2-4）：探针统一用 `step(1/60)` 喂两台引擎，**上游生产是 1/20** ⇒ 本文的"march track 相同"只对"同 dt 喂法"成立。
5. `fastcore` 装在**临时目录**（`/tmp/deps_upstream`）⇒ 重启机器后需重装（命令已写在 §4）。

---

## §8 一句话给决策者

- **寻路骨架同源**，差异就是 **D1 目标格 / D2 代价 / D3 河面可走性 / D5 命中几何**（外加 D4 跟随与节流）；实测**近战场面两引擎逐字节相同、中远程场面停靠点差 1–4 格**。
- **FL 拟合度实验跑得起来**：上游引擎只缺 `fastcore`（隔离装即可）、cwd 契约仍要守、卡表 **150 ⊂ 212**（觉醒/英雄牌组跑不了）——这些都已实测。
- **判据已先写死**（§6）：配对 Wilcoxon + McNemar + 阳性/阴性对照 + 三条失败分支；**是否开跑、以及用三种 replay 口径里的哪一种，等你一句话**（口径直接决定分数的含义，【R17】）。
