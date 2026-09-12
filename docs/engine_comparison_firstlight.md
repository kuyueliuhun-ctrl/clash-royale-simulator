# 引擎对比：clash-royale-simulator vs FirstLight CR

> 对比对象：本项目 `E:\clash-royale-simulator-main`（自研纯 Python 引擎）
> vs `E:\FirstLight_CR`（FirstLight CR，借壳原生引擎）
> 轴线：寻路机制 / 单位表示 / 特殊机制实现
> 生成日期：2026-09-11

---

## 0. 一句话结论

**两者不是"两套实现"，而是两种引擎哲学。**

| | 本项目 | FirstLight CR |
|---|---|---|
| 引擎本体 | 自研纯 Python 引擎，一行行复刻游戏规则 | **无自研引擎**，运行 Null's Royale 原生 `libg.so`（ARM64，MuMu 安卓 VM） |
| Python 角色 | **就是引擎** | **只是外壳**：发指令、读遥测、校验契约 |
| 寻路 | 自己写的 A* | Python 侧零实现，全交原生 |
| 单位 | 自己写的 `Entity` 类 + 行为方法 | 原生内存快照（裸数据）+ 严格校验 |
| 卡牌机制 | 自己写的 `card_mechanics.py`（1895 行） | 原生自带；Python 只做**描述 + 身份 join + 就绪门控** |

FirstLight 官方自述即为定论（`docs/architecture.md:5`）：
> "The native engine executes battles, the Python environment builds observations and submits actions."

`cr_native_env.py:2-8` 进一步说明它"**without screenshot interpretation or ADB touch injection**"——靠原生命令/状态 API，不是截图识别。

**所以本对比的实质是：一个自研复刻引擎，和一个 hook 官方引擎的黑盒外壳之间的差异。** 下面三条轴线的所有具体差异都由这一点派生。

---

## 1. 寻路机制（Pathfinding）

### 1.1 本项目：自研 8 邻接 A\*

实际使用 `pathfinding_heap.py`（`battle.py:4` 明确导入 `from pathfinding_heap import EntityPathfinder, ...`）。

- **算法**：8 邻接 A\*，open set 用 `heapq` 最小堆（`pathfinding_heap.py:116-123`）。
- **启发式**：`10 * max(|Δx|, |Δy|)`，即切比雪夫距离缩放（`pathfinding_heap.py:47-50`）。对 octile 网格可采纳（对角真代价 14 > 10，不高估）。
- **几何步代价**：正交 10 / 对角 14（`pathfinding_heap.py:143-146`）。
- **地形成本**（`pathfinding_heap.py:132-142`）：`W`(河) 地面单位 800、飞行单位 7；`.`(桥面) 5；普通地面 5。
- **网格**：36 × 64 格；世界坐标 `[0,18]×[0,32]`，`floor(2x), floor(2y)` 映射，**每世界单位 = 2 格**（`pathfinding.py:12-14`）。
- **可通行权威**：不在 `tilemap_lane_grid.txt`，而在 `arena.py:106-119` 的 `is_walkable`（河道仅左桥 `2≤x<5` / 右桥 `13≤x<16` 可走）+ `BLOCKED_TILES`；塔矩形并入 36×64 的 `building_cache` 距离场（`battle.py:3031-3056`）。
- **移动方式**：**路点跟随 + 连续浮点插值**，不是格子跳跃。每 10 tick 在视距内重算路径（`battle.py:1168-1223`）；用点积判断是否已越过当前路点，再递增 index。
- **速度单位**：`Card.speed = data['speed']/50`，即世界单位/秒（`card_utils.py:242`）。
- **特殊处理**：
  - 空中单位直接跳过 A\* 直线飞（`battle.py:1176-1177`）；
  - `jump_speed` 单位可跳河（`battle.py:1169-1175`）；
  - 塔矩形碰撞用最近点法线推出（`battle.py:3115-3146`）；
  - **卡死自愈**：每 0.5s 检查位移≈0 则丢弃路点、重算全程（`battle.py:1189-1199`），注释记录了"两个弓箭手楔死在桥头 60s 纹丝不动"的实证。
- **两个寻路文件的区别**：
  - `pathfinding.py`：早期版，线性扫描 `min()`（`:94`）、桥成本 8、塔圆形目标；
  - `pathfinding_heap.py`：**当前在用**，堆实现 + edge-based 目标选择 + 桥成本改 5 + 车道保持。
  - 桥成本从 8 降到 5 的原因写在注释里：旧值会把单位"吸向更便宜的桥中央地面格，抹平车道"（`pathfinding_heap.py:136-139`）。
  - 目标格选择改为 **edge-based**（`edge_distance_from`，`:68-106`）：取"到目标边缘最近 + 到起点最近"，实现左桥靠左、右桥靠右的车道保持（2026-09-10 机制定稿）。

### 1.2 FirstLight CR：Python 侧零寻路实现

**负面证据（最确凿）**：全项目搜 `A*` / `astar` / `navmesh` / `waypoint` / `pathfind` / `dijkstra`，**无任何算法实现**。命中的全是原生路径的**字段名或引用**：

- `building_placement_profiles.py:13,44,63-64`：`FULL_ARENA_PATHFIND_MORPH`、`"static-root:GoblinDrillDig:SpawnPathfindMorph=GoblinDrill"`——哥布林钻机的**原生**生成路径变形行为。
- `arena.py:98`："pathfinding deploys intentionally remain absent because their live roots do..."（Python 侧有意不处理）。
- `resource_compiler/mechanics.py:261,1361,1395`：`SpawnPathfindMorph` 是从原生卡牌数据**提取出的字段标签**，不是路径算法。

**Python 侧只有落点换算，不是寻路**：
- `arena.py:158-177`：`cell_to_world` / `world_to_cell`，18×32 粗格 ↔ 原生世界坐标（`CELL_UNITS=1000`，世界 18000×32000）。
- `battle_env.py:2281-2283`：`_world_target` 把 `action.target_grid` 转世界坐标，**仅用于提交部署落点**。
- `arena.py:180-189`：`_bottom_walkable` 注释写明 "Native-probed coarse-cell terrain for rows 0..16 of tilemap.csv"——**地形也是原生探针读的**，Python 不计算。

**如何触发/读取移动**：
- **触发**：Python **不注入移动**，只注入"在哪下牌"。通过 `probe/native_touch_interceptor.h:32-35` 拦截 `GameApp.nOnTouchEvent`，以及 `cr_native_env.py:2861-2865` 的 `inject_command`——"Queue one exact native logic command **without advancing the engine**"。下牌之后所有移动全归原生引擎。
- **读取**：probe 在原生移动函数挂内联 hook 观测（黄金骑士跳 `libg+f49d24`、warp `libg+e82a34`，见 `probe/action_movement_runtime_layout.h:24-25`；冲刺执行 `libg+f609dc`，见 `probe/special_movement_runtime_layout.h:20-21`），直接读对象位置字段 `kObjectPositionXOffset = 0x7c`。
- **无位置注入 / 无 movement override**：`probe/remaining_runtime_arm64.S:57-61` 注释明确 "The observer only records child-pointer provenance; **the relocated original remains solely responsible for native execution**."
- Python 侧的 `action_movement_runtime.py` / `special_movement_runtime.py` / `character_state_runtime.py` 全是 **"strict typed contract" 校验器**，按硬编码 hook 偏移校验原生遥测，不是移动逻辑。

### 1.3 寻路轴线小结

| 维度 | 本项目 | FirstLight CR |
|---|---|---|
| 算法归属 | 自研 A\*（`pathfinding_heap.py`） | 原生引擎内部（未公开） |
| Python 参与 | 完整实现 | 仅 `cell↔world` 换算（部署落点用） |
| 网格 | 36×64，2 格/世界单位 | 18×32 粗格，`CELL_UNITS=1000` |
| 可通行来源 | 自研 `arena.is_walkable` + 塔距离场 | 原生探针读 tilemap.csv |
| 车道保持 | 自研 edge-based 目标 + 桥成本调参 | 原生行为，Python 不感知 |
| 卡死处理 | 自研 0.5s 卡死自愈 | 原生内部 |

---

## 2. 单位表示（Unit representation）

### 2.1 本项目：实体类 + 行为方法

- **数据结构**：普通 `class`（非 dataclass / dict / numpy）。`Entity`（`battle.py:13-98`）、`Troop`（`:759`）、`Building`（`:1233`）、`Projectile`（`:1417`）、`AreaEffect`、`EvoEffectZone` 等一整套子类。
- **核心字段**：`position`（`Position` dataclass，`core.py:4-9`）、`target_id`、`is_alive`、`hp`、`shield_health`；状态标志 `targetable`、`invincible`、`jumping_across_river`、`speed_buff/debuff/hit_speed_mult`、`freeze_timer`、`damage_reduction`、`regen_buffs`、`hook_pull`；机制槽 `ramp_*`（递增伤害）、`evo`、`attack_seq*`（动作链）、`evo_chain`、`_snipe_range_active`、`_evo_landing`。
- **机制代理对象**：`self.entity_holder = eval(f"{self.card_name}(self)")`（`battle.py:50-53`），把卡牌逻辑从 Entity 中拆出去。
- **坐标系**：世界 `x∈[0,18], y∈[0,32]`；`Position.distance_to` 用欧氏 `math.hypot`（`core.py:8-9`）。y 由蓝方(底)向红方(顶)递增（`BLUE_KING_TOWER=(9,3)`, `RED_KING_TOWER=(9,29)`，`arena.py:11,14`）。
- **状态机**：**没有枚举/位标志/转移表**，完全由 `Troop.update` 的过程式条件驱动（`battle.py:1099-1229`），状态散落为隐式布尔/数值标志。
- **数据来源**：运行期真正的数据源是 `gamedata.json` + `cards_stats_{characters,spell,building,projectile}.json`（`card_utils.py:4-18`）。
  - ⚠️ **`cards.json` 在引擎核心中并未加载**，它只被 `client_side/download_images.py:21` 和 `minimal_visualizer.py:63` 使用（RoyaleAPI 英文名/图标元数据）。
  - `card_aliases.py` 仅被 `rl/decks.py` 用于中文俗名→卡名映射，不参与单位构建。
  - `Card` 从 `summonCharacterData` 读 hp/伤害/碰撞半径/速度/射程/攻速/飞行标记等（`card_utils.py:217-345`）；`set_level` 按 per-level 数组或 `1.1^(level-1)` 曲线放大到 11-16 级（`:347-443`）。
- **相互作用**：`target_id` 是整数索引 `battle_state.entities`（`battle.py:27`）；索敌在 `update_current_target`（`:672-727`）/`get_nearest_target`（`:618-649`），含"建筑优先""不跨中轴"规则。
- **碰撞**：`resolve_collisions`（`:3084-3113`）对地面部队两两、空中两两做**圆形推开**；塔用矩形推出。**无空间哈希/网格加速**，O(n²) 成对检测（注释直言 "gets called several millions times per game"）。

### 2.2 FirstLight CR：原生内存快照 + 契约校验

- **内存读取在 C++ probe，不在 Python**：
  - `probe/cr_replay_probe.cpp:421,858,925`：`g_libg_base` / `find_libg_base_from_proc_maps()` 从 `/proc/pid/maps` 定位 `libg.so` 基址。
  - `:785-786`：`memcpy` 读游戏状态上下文根。
  - `:1700-1707`：`create_state_snapshot` 通过 AArch64 桥（`native_call_arm64.S:24-39`）**直接调用原生函数**序列化整局状态。
  - `:2277`：`read_object_field<std::int32_t>(object, 0x7c)` 读位置。
- **Python 侧不读原始内存**：全项目搜 `struct.unpack` / `mmap` / `from_buffer`，只有 `native_overlay.py:251-308` 用 `ctypes` 操作 **Windows 窗口句柄**（纯 UI）。Python 收到的是 probe 输出的 **JSON 遥测**。
- **实体结构（来自原生 wire 契约）**：`runtime_schema.py:26-35,96-130` 校验 `nativeObjectId / entityKey / owner / objectIndex / secondaryIndex / cardId / objectKind / position / visibilityValidated / invisibleCount`；`position` 是原生世界坐标 `tuple[int,int]`。
- **无 Python 实体行为类**：全是 `dataclass` 契约/快照（`contracts.py:2119` 的 `EntityStateV1`）。观测栅格只是把**原生已供给的实体**栅格化，不仿真（`contracts.py:2473-2476`：`RasterV2` "built only from the already filtered entities/towers supplied"）。
- **卡牌/单位数据是描述性数据表**：
  - `card_names.py:1-9`："Display names from the bundled game CN/EN text tables ... **UI labels only**"。
  - `resource_compiler/__main__.py:1,111`：从本地提供的游戏资源**复现**固定数据集，并强调 "Conditions and paths are categorical feature labels; **the loader never executes them**"。
  - `card_specs.py` / `card_logic.py` 是静态规格图（`card_logic.py:1-5`："The graph is **evidence data**"）。

### 2.3 单位表示轴线小结

| 维度 | 本项目 | FirstLight CR |
|---|---|---|
| 容器 | Python class `Entity`/`Troop` + holder 代理 | 原生内存结构体 + JSON 快照 |
| 行为归属 | 在 Python 类方法里 | 在原生引擎里，Python 无行为 |
| 读取方式 | 直接访问对象 | C++ probe `memcpy` + 原生函数调用 |
| 状态机 | 过程式条件 + 散落布尔标志 | 原生内部（Python 只读 `kCharacterStateOffset = 0x11c`） |
| 数据来源 | `gamedata.json` + `cards_stats_*.json` | 原生资源编译出的静态规格图（仅描述） |
| 坐标 | 世界 0-18 / 0-32 | 原生 0-18000 / 0-32000 + Python 粗格换算 |
| 碰撞 | 自研 O(n²) 圆形推开 + 塔矩形推出 | 原生内部 |
| Python 内存操作 | 无（纯对象） | 仅 Windows 窗口句柄（UI 覆盖层） |

---

## 3. 特殊机制实现（Special mechanics）

### 3.1 本项目：命名约定式注册表，逐卡实现

**分发机制**（`battle.py:50-53`）——这是最独特的架构点：

```python
self.entity_holder = BasicCharacter(self)
if self.card_name in globals() and not isinstance(self, Projectile):
    self.entity_holder = eval(f"{self.card_name}(self)")
self.entity_holder.on_spawn()
```

即 **"卡名即类名"**：每张有特殊逻辑的卡 = 一个与卡名同名的 `BasicCharacter` 子类（如 `class Witch(BasicCharacter)`）。`card_mechanics.py` 末尾 `from card_mechanics import *` 把全部类名注入 `battle.py` 全局命名空间，引擎用 `eval` 按卡名查找。**无注册表 dict、无 if-else 链、无工厂函数、无事件总线。**

代表性机制（代码实证）：

- **弹道 / 溅射 / 链**（`battle.py:1417-1636`）：homing 追踪；Log 类沿方向滚动持续伤害+击退；`_deal_splash_damage` 圆形判定、对塔按 `crown_tower_percent` 减伤；`_chain` 触发二段弹/落地出兵；Monk 禅定反弹（`:1455-1465,1598-1604`）。
- **冲刺 / Charge / Dash**：Prince 冲锋（`card_mechanics.py:113-153`，超 `charge_range` 后 `speed*=2`）；刺客突进（`:715-807`，**不可打断**，0.8s 全程无敌，`Troop.update` 直调 `dash_tick` 绕过冰冻/眩晕，`battle.py:1153-1155`）；超骑三阶段冲刺跳（`:509-637`）。
- **召唤 / 生成**：`_troop_spawner_tick`（`battle.py:247-276`）+ `_generic_death_spawn`（`:276-325`）；Witch 周期产骷髅（`card_mechanics.py:66-89`）；凤凰死亡→火球+产蛋孵化（`:1692-1713`）。
- **进化卡**：`evolution_state(plays, card_name)` 按 `EVOLUTION_CYCLES` 周期表判定本手是否觉醒（`evolutions.py:89-95`）；`derive_evolved_stats` 推导数值（`:98-133`）；`Troop._apply_evolution` 装配（`battle.py:782-848`）；`_evo_on_attack` / `_evo_on_death` 按字段族分发（`:890-1056`）。
- **英雄技能**：基类 `_HeroBase.use_ability`（`card_mechanics.py:315-316`），各 Hero 覆写；入口 `BattleState.use_ability`（`battle.py:3208-3248`，含条件窗 + 单次使用 + 圣水预检/未生效返还）；能力表在 `elite17_data.HERO_ABILITIES`。
- **递增伤害**：`has_ramp` + `ramped_damage`（`battle.py:442-456`），按 `ramp_stage_times` 切 `ramp_stage_damages`。
- **反射 / 护盾**：Monk 反弹；电磁王 `on_damaged` 反射 75×等级+0.5s 眩晕（`card_mechanics.py:1584-1600`）；`take_damage` 先扣 `shield_health`（`battle.py:540-542`）。

**钩子接线方式**：**固定方法调用链，非事件回调**。tick 顺序硬编码在 `BattleState.step`（`battle.py:2652-2726`）：

```
step():
  update_player_hp / 加时裁决 / regenerate_elixir
  prune dead; refresh_tower_alive_cache
  calculate_building_cache（按需）
  for entity: entity.update(dt); ensure_walkability(entity)
  resolve_collisions()
  schedule 延迟生成; resurrect_queue 觉醒复活
  time += dt; tick += 1
```

单位内顺序：`Troop.update` → `freeze`/`deploy_delay` 早退 → `Entity.update`（刷 buff/ramp/攻击序列，调 `entity_holder.on_tick`）→ 索敌/嘲讽/各 `tick` 钩子 → 移动或 `on_attack`。生命周期钩子：`on_spawn` / `on_tick` / `on_attack` / `on_death` / `on_take_damage` / `on_damaged` / `use_ability`。

**已知占位/未实现（代码中有明确标注）**：

- 链电半径无字段 → 默认 3.0，"待 L4"（`battle.py:807`）
- 二段爆炸半径无字段 → 默认 1.0，"待 L4"（`battle.py:1513`）
- 墓园本体死亡即停止（官方：墓园继续刷完，待 L4）（`card_mechanics.py:338`）
- Musketeer 狙击弹伤害数据缺失 → 沿用普攻，"低置信度"（`:642`）
- Ronin 当前按纯被动实现（`:677-678`）
- 大量 Hero 能力时长/位移为"【假设-暂借】"（`:949,983,992,1345,1360,1407,1449,1496-1497`）
- ElectroWizard 部署 Zap 半径/伤害"快照缺字段，Fandom，待对拍"（`:1614`）

### 3.2 FirstLight CR：机制全部原生自带，Python 只描述

**三个 catalog 都是描述/分类，不是实现**：

- `card_logic.py:1-5`：加载固定静态玩法图，"**The graph is evidence data**"。
- `effect_catalog.py:1`：`resolve_runtime_effect` 把原生 `activeEffect`（`provenance: "authoritative_native_type3_component"`）**join** 到静态语义，"without inventing semantics"。
- `projectile_catalog.py:1,206`：把原生 `LogicProjectileData+0x40` 的 global ID join 到静态语义；并明示 `runtime_active_homing_available: False`——**Python 不实现主动追踪**。
- `resource_compiler/mechanics.py:1-6`："**The native logic graph remains the source of truth.** This module only projects explicit graph structure, typed native actions, Buff definitions ... it **does not infer mechanics from card names**."

**`phase_runtime.py` 是"拼接"而非"执行"**（`:1-8`）：
> "The native layer deliberately emits raw facts. This module performs the small amount of stateful joining needed for attack phases, Inferno damage ramp, classic movement charge ... Every resolver **fails closed** when its required native hook or static AttackSequence evidence is absent."

且 `phase_runtime.py:21` 的 `EXACT_LIBG_SHA256` 与 `supported_engine.json:8` 的 `libg_sha256` 完全一致，**精确绑定到某个原生二进制哈希**。

**特殊移动是遥测契约校验**：`action_movement_runtime.py` / `special_movement_runtime.py` 识别原生动作的 hook 偏移与卡 ID（黄金骑士、Boss 女盗、精英弓手 warp、普攻 dash），**不实现这些动作**。

**resource_compiler 做什么**：从用户提供的游戏资源**复现**固定数据集（card_logic / effects / projectiles / model_catalogs），并断言生成摘要与发行版 manifest 的 SHA-256 一致（`__main__.py:47-48`）。是"提取数据 + 验证"，不是"编译出机制"。

**fail-closed 门控与覆盖范围**：
- `semantic_subset.py:384-393`：`NormalModePolicyReadinessV1` 含 `ready_card_ids` / `excluded_card_ids`。
- `:440-448`：卡被排除当且仅当 `NotVisible` / `NotInUse`，或 `REASON_MISSING_MECHANICS_READINESS`（**缺机制就绪证据**）。
- `:457`：每张允许卡都带 `mechanics_readiness_sha256`。
- `:543,666`：`"fail_closed_before_native_mutation": True`——**任何未覆盖机制在触碰原生前就被 fail-closed**。
- `:218-221,288-294`：进化 / 英雄形态受 `allowed_form_availability` 位掩码控制，必须有对应 `"evolution:"` / `"hero:"` 证据。
- 规模：源卡 152 张，竞技模式可玩 **122 张**（`supported_engine.json` / `runtime_scope.py:16-18`）。

### 3.3 特殊机制轴线小结

| 维度 | 本项目 | FirstLight CR |
|---|---|---|
| 实现归属 | 自研 `card_mechanics.py`（1895 行） | 原生引擎自带 |
| 分发机制 | `eval(f"{card_name}(self)")` 命名约定 | catalog **join**（描述层，非实现层） |
| Python 的角色 | 实现者 | 描述者 / 校验者 / 门控者 |
| 钩子接线 | 固定方法链（on_spawn/tick/attack/death…） | 内联 hook 观测原生函数 + 契约校验 |
| 事件系统 | 无（硬编码方法分发） | 无（原生遥测 + fail-closed） |
| 未覆盖机制 | 代码内标注【假设】【待 L4】占位 | `excluded_card_ids` + fail-closed 显式排除 |
| 数值可信度 | 部分为 Fandom/推算，需 L4 对拍 | 与原生二进制哈希绑定（`EXACT_LIBG_SHA256`） |
| 版本耦合 | 无（自研，数据驱动） | **强耦合**：绑死 libg SHA256 + 资源版本 15.535.86 |

---

## 4. 本质差异总结

### 4.1 三条轴线的对比矩阵

| 轴线 | 本项目 | FirstLight CR | 根本原因 |
|---|---|---|---|
| 寻路 | 自研 A\*（36×64 格、路点跟随、卡死自愈、edge-based 车道保持） | 原生引擎内部；Python 仅 cell↔world 换算 | 自研 vs 借壳 |
| 单位表示 | Python class + 行为方法 + 隐式状态机 | 原生内存快照 + JSON 遥测 + 契约校验 | 自研 vs 借壳 |
| 特殊机制 | `card_mechanics.py` 逐卡实现，eval 分发 | 原生自带，Python 只 catalog join + fail-closed | 自研 vs 借壳 |

### 4.2 两种架构的取舍

**本项目（自研）的优势**
- **完全可控**：任何机制都能改，可做消融实验、可注入自定义奖励/约束（这也是本项目 RL 研究的前提）。
- **速度与并行**：纯 Python 单进程，无 VM、无 socket、无 APK；实测量级 ~19k battle-steps/s（lv11 交火），一局几十毫秒级。
- **无版本耦合**：数据驱动，不受某个原生二进制哈希限制。
- **可解释**：所有状态在内存里可见，取证/调试直接。

**本项目（自研）的代价**
- **数值保真度**：需要靠 Fandom / 推算补缺字段，代码里大量【假设】【待 L4】标注（链电半径、二段爆炸半径、狙击弹伤害、墓园刷完、Hero 时长等）。
- **长尾机制成本**：152 张卡 × 每张特殊机制，逐卡实现的工作量与维护成本高。
- **行为差异**：寻路/索敌/碰撞是自研近似（edge-based 启发式、桥成本调参、0.5s 卡死自愈都是经验修正），与真实引擎未必逐帧一致。

**FirstLight（借壳）的优势**
- **行为 100% 真实**：寻路、机制、数值就是官方引擎本身，不存在"近似"。
- **机制成本为零**：新卡/新机制由游戏版本更新带来，无需实现。
- **数据权威**：`EXACT_LIBG_SHA256` 绑定，可复现。

**FirstLight（借壳）的代价**
- **强环境依赖**：需要 MuMu 安卓 VM + root + 特定 APK（Null's Royale 15.535.13 / 内容 15.535.86）+ JDK17/NDK，装机门槛高。
- **强版本耦合**：`libg.so` 换版本就要重做 hook 偏移与资源编译。
- **吞吐瓶颈**：多进程原生实例（`:engine0..:engine23`），远低于纯 Python 单进程；RL 训练要靠集群与 IL 预热（252K 回放 / 17.8M 动作）弥补采样效率。
- **不可修改**：无法改动引擎规则本身（无法注入自定义奖励/约束/消融）。

### 4.3 对本项目的启示

1. **保真度是本项目唯一的结构性短板**。FirstLight 用"借壳"换来了零保真度损失，本项目用"自研"换来了完全可控。既然可控是 RL 研究的刚需，**保真度只能靠对拍来补**——建议把代码里的【待 L4 对拍】清单整理成一张可执行的验证表。
2. **FirstLight 的 `semantic_subset` / fail-closed 思路值得借鉴**：它对"未覆盖机制"是**显式排除**，而本项目目前是"标了假设继续跑"。可以为本项目建立一份"机制可信度分级"清单（已验证 / 推算 / 假设），让训练与评估时能区分对待。
3. **FirstLight 的观测层设计（event token、catalog join）与本项目 `rl/belief.py` 的思路同源**，可对比其"原生事件 → 结构化 token"的字段设计。
4. **不要试图照搬 FirstLight 的架构**：本项目的价值恰恰在于"引擎是自己的"，这是做 RL 消融与机制定制的唯一前提。

---

## 附录：关键源码索引

**本项目**
| 主题 | 文件:行 |
|---|---|
| 寻路入口（heap 版） | `src/clasher_new/battle.py:4` |
| A\* 核心 / 启发式 | `src/clasher_new/pathfinding_heap.py:47-160` |
| 网格映射 | `src/clasher_new/pathfinding.py:12-33` |
| 可通行权威 | `src/clasher_new/arena.py:106-119` |
| 单位移动 | `src/clasher_new/battle.py:1087-1223` |
| 碰撞 | `src/clasher_new/battle.py:3084-3146` |
| 机制分发 | `src/clasher_new/battle.py:50-53` |
| 机制库 | `src/clasher_new/card_mechanics.py`（全 1895 行） |
| tick 主循环 | `src/clasher_new/battle.py:2652-2726` |
| 卡数据加载 | `src/clasher_new/card_utils.py:4-18, 217-443` |
| 进化 | `src/clasher_new/evolutions.py:11-133` |

**FirstLight CR**
| 主题 | 文件:行 |
|---|---|
| 架构声明 | `docs/architecture.md:5` |
| 无截图/无 ADB 声明 | `native_runner/cr_native_env.py:2-8` |
| 坐标换算 | `native_runner/arena.py:158-189` |
| 原生地形探针 | `native_runner/arena.py:180-189` |
| 内存读取 / 基址 | `native_runner/probe/cr_replay_probe.cpp:421,858,925,785-786,1700-1707,2277` |
| hook 仅观测 | `native_runner/probe/remaining_runtime_arm64.S:57-61` |
| 移动遥测契约 | `native_runner/action_movement_runtime.py:20-152` |
| 冲刺遥测契约 | `native_runner/special_movement_runtime.py` |
| 机制图（描述） | `native_runner/card_logic.py:1-5` |
| 效果 join | `native_runner/effect_catalog.py:1,283-351` |
| 弹道 join | `native_runner/projectile_catalog.py:1,206,210-247` |
| 静态机制投影 | `native_runner/resource_compiler/mechanics.py:1-6` |
| phase 拼接 | `native_runner/phase_runtime.py:1-21` |
| fail-closed 门控 | `native_runner/semantic_subset.py:384-457,543,666` |
| 引擎版本绑定 | `native_runner/supported_engine.json:8` |
