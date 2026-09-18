# 上游对照 · 按对象四节合并（2026-09-19）

> **本文件是四份子任务稿的「按对象合并件」，不是新一轮取证。**
> 合并规则：**逐字保留原稿的 file:line、命令、diff 与判定**；原稿未见的部分**一律不代写**，以【截断】标注；互相矛盾之处**并列不调和**。
> 唯一例外 = 明确标注的 **【合并者复算】** 段：为补齐源稿被截断处，本合并会话在 A/B 两仓上重跑的**只读** git 命令（命令与原始输出逐字附上）。
> 四节 = §1 上游提交清点 / §2 过河与 A\* / §3 碰撞 / §4 非引擎改动。

---

## §0 合并前置：坐标、源稿、截断清单、标记约定

### 0.1 坐标与取数

| 项 | 值 | 出处 |
|---|---|---|
| A（我方） | `/mnt/e/clash-royale-simulator-main`，HEAD `8d2552a0eba5eb601e5966a24e601891cd060e83`，remote `kuyueliuhun-ctrl/clash-royale-simulator` | 稿 1 |
| B（上游） | `/mnt/e/clash-royale-simulator-main-by-jason`，HEAD `f616f190f5c84b28c6e48c48392cf73fc895a477`，remote `https://github.com/Jason-XII/clash-royale-simulator.git` | 稿 1 |
| 分叉点（两仓共同祖先，稿 1 认定） | `f20fa4d535b355e1d5727becbd6568bae21ff5b9`（2026-08-27 23:07:59 +0800，`Add entropy coefficient to encourage more diverse actions`） | 稿 1 |
| 稿 1 观测时间 | 2026-09-18 22:47 UTC（= 2026-09-19 06:47 +0800），取数机 = 本地 WSL | 稿 1 |
| 稿 4 观测时间 | 2026-09-18 22:50 UTC | 稿 4 |
| 稿 4 记录的上游 HEAD 时间 | 2026-09-18 23:24:18 +0800（`f616f19`） | 稿 4 |
| 稿 3 取证范围 | 我方 `src/clasher_new/battle.py`、`arena.py`、`rl/replay.py`、`rl/env_wrapper.py`、`rl/run_league.py`、`docs/`、`scripts/`；上游 `by-jason`（`git log/show` 只读） | 稿 3 |
| 全程纪律 | **只读**：`log/rev-list/show/merge-base/cat-file/ls-remote/grep/read`；无写、无 `checkout`、未跑训练、未跑 git 写操作 | 稿 1 / 稿 3 / 稿 4 |
| 稿 2 运行姿势（其自述限定） | 仓库 `.venv` 是 Windows venv（`Scripts/python.exe`），WSL 下不可执行 ⇒ 用系统 `python3 3.14.4`，经 **stdin** 传代码（**未落盘脚本**，禁写 `.pyc`）；稿 2 自述「**这不算按 `docs/agents/env.md` 的正式运行姿势**」 | 稿 2 |

### 0.2 源稿截断清单（四稿传入本合并会话时**均被截断**）

| 源稿 | 对象 | 截断处最后可见文字 | 未见内容（**不代写**） |
|---|---|---|---|
| 稿 1（researcher） | 上游引擎提交独立清点 | §3.2 `372ed0e` 的 diff 块内 `\`\`\`diff` + ` class` | `372ed0e` 其余 diff；`fcb1d24` / `1e9716b` / `2daab60` / `574e946` 的逐条摘要；§4 存在性核对等 |
| 稿 2（reviewer） | 过河与 A\* | §1.6 末「⇒ 所以 `tile_cost = 800`（我们的地面 W 代价）**恰恰只在过」 | 该句余下；§0 提到的「我方另有 3 个真实问题（1 个高严重度，见 §4）」全文 |
| 稿 3（reviewer） | 碰撞 | §3 建议仪器命令块内 `d = pickle.load(open("runs/arch` | 命令块余下；其后各节 |
| 稿 4（researcher） | 非引擎改动 | §4.1 判定表第 7 行（`defensive_strategy.py` 行） | 第 8 行起的判定；§4.2 起的各节 |

### 0.3 标记约定

- **【原文】** = 子任务稿内容（默认，不逐句加标）。
- **【合并者复算】** = 本合并会话在 A/B 上重跑的只读命令 + 原始输出。
- **【截断】** = 源稿在此处被切断；**未见的结论不补写**。
- **【冲突并列】** = 两说并列，**不调和、不裁决**。

---

# §1 上游提交清点（B 仓库相对分叉点的独占提交 + 引擎改动）

> 本节主体 = 稿 1（标题《B 仓库相对分叉点的独占提交与「引擎改动」独立复算报告》）。
> **截断说明**：稿 1 在 §3.2 `372ed0e` 的 diff 内被截断（见 §0.2）。§1.4.3–§1.4.6 为 **【合并者复算】** 补齐，格式与稿 1 §3 一致，命令与原始输出逐字附上。

## 1.1 分叉点与所用命令（稿 1 §1 逐字）

| 项 | 值 | 命令 | 结果 |
|---|---|---|---|
| 共同祖先（分叉点） | `f20fa4d535b355e1d5727becbd6568bae21ff5b9` | `git -C A log -1 f20fa4d` | `f20fa4d…bae21ff5b9｜2026-08-27 23:07:59 +0800｜Add entropy coefficient to encourage more diverse actions` |
| 分叉点是 B HEAD 的祖先？（**必须在 B 里判，A、B 的 object store 不共享**） | 是 | `git -C B merge-base --is-ancestor f20fa4d HEAD` (rc=0) | YES |
| 分叉点是 A HEAD 的祖先？ | 是 | `git -C A merge-base --is-ancestor f20fa4d HEAD` (rc=0) | YES |
| B 侧独占集合 | **24 个提交** | `git -C B rev-list HEAD --not f20fa4d \| wc -l` | `24` |
| 同一集合（另一种写法，交叉验证） | 24 | `git -C B rev-list --count f20fa4d..HEAD` | `24` |
| 历史是否线性（→ 无 merge 使集合语义歧义） | 线性，0 merge | `git -C B log --merges --oneline f20fa4d..HEAD \| wc -l` = `0`；`git -C B log --format='%h %p' f20fa4d..HEAD` 每行恰 1 个父 | — |
| A 侧分叉后提交数（参照） | **242** | `git -C A rev-list --count f20fa4d..HEAD` | `242` |
| 独占集合中是否有也存在于 A 的提交 | 否（抽样 `2aa7b5e`） | `git -C A cat-file -t 2aa7b5e` | `absent-in-A` |
| B 本地是否＝上游 GitHub 最新 | 是（截至取数时刻） | `git -C B ls-remote origin refs/heads/main` | `f616f190f5c84b28c6e48c48392cf73fc895a477 refs/heads/main` = 本地 HEAD |
| B 分支跟踪 | `* main f616f19 [origin/main]` | `git -C B branch -vv` | 与 origin/main 齐平 |

**稿 1 记录的踩坑（逐字保留）**：最初在 A 里跑 `git merge-base HEAD /mnt/e/…-by-jason/HEAD` → `fatal: Not a valid object name`。原因：跨仓库路径语法不成立，且 **A 的 object store 里根本没有 B 的独占提交**（`git -C A cat-file -t 2aa7b5e` 失败）。所以「B 的独占集合」必须在 **B 里**算；A 只能提供 `f20fa4d` 这一侧的坐标。

## 1.2 B 独占提交全表（24 个，按时间正序；稿 1 §2 逐字）

生成命令：`git -C B log --reverse --format='%h | %ad | %s' --date=short f20fa4d..HEAD`

| # | 短 hash | 日期 | 提交标题 | 文件分类（判据见 §1.3） |
|---|---|---|---|---|
| 1 | `2aa7b5e` | 2026-09-07 | BUGFIX: fix wrong transform of observation. | RL‑训练侧（environment.py, temp_test.py）＋**删除** watch_random_models.py |
| 2 | `87c5a90` | 2026-09-07 | ENHANCE: remove the illegal move penalty, planning to add action masking later | RL‑训练侧 |
| 3 | `b652f0c` | 2026-09-07 | ENHANCE: Add time and battle phase to environment observation, may improve training | RL‑训练侧 |
| 4 | `00c2b4f` | 2026-09-08 | ENHANCE: phase and battle clock are now considered by the model | RL‑训练侧 |
| 5 | `a316cae` | 2026-09-09 | BUGFIX: finally made sense of tensor shapes, training code is working now. Add debugging parameter for easier debugging | RL‑训练侧 |
| 6 | `23e5b8c` | 2026-09-14 | Big update: add several strategies that crushes the random strategy. Planning to train AI using these strategies as the opponent | RL‑训练侧（新增 strategies.py） |
| 7 | `964af5d` | 2026-09-14 | Prepare training script; fix ent_coef issue; add missing file; doesn't import pygame when there's no visualization | RL‑训练侧（**新增 defensive_strategy.py**） |
| 8 | **`d1e0a16`** | 2026-09-14 | BUGFIX: fix troop collision issue mentioned by @billy948787 | **引擎 `battle.py`** ＋可视化 |
| 9 | `8903460` | 2026-09-15 | Improve evaluation script | RL‑训练侧 |
| 10 | `2750ebd` | 2026-09-15 | Architecture change: now the observation uses stacked frames | RL‑训练侧 |
| 11 | `31c8232` | 2026-09-16 | Architecture change: Use autoregressive actions, choose cards first, then place them. | RL‑训练侧 |
| 12 | **`372ed0e`** | 2026-09-16 | BUGFIX: fix buildings hp do not decrease, thanks to @longlingking | **引擎 `battle.py` + `card_utils.py`** ＋可视化 |
| 13 | `e7dcad5` | 2026-09-16 | Update benchmark script | RL‑训练侧 |
| 14 | **`fcb1d24`** | 2026-09-16 | BREAKING: change simulator to be 20Hz instead of 60Hz, which significantly improves speed. | **引擎 `battle.py`** ＋RL 侧＋可视化 |
| 15 | `a5e4ab1` | 2026-09-16 | Update readme | 文档 |
| 16 | **`1e9716b`** | 2026-09-17 | Add some comments in `battle.py`; Update evaluation strategy to check for invalid moves; Change architecture to mask actions and choose actions based on content scoring. | **引擎 `battle.py`（仅注释）** ＋RL 侧 |
| 17 | `8744065` | 2026-09-17 | Enhance: upgrade defensive strategy to be significantly stronger | RL‑训练侧＋可视化 |
| 18 | `e049d55` | 2026-09-17 | Enhance: strengthen all existing strategies | RL‑训练侧 |
| 19 | **`2daab60`** | 2026-09-17 | BUGFIX: previous code settings are not compatible with the new A* pathfinding algorithm, especially the part that handles river jumps. I removed the river tiles from blocked_tiles. | **引擎 `arena.py` + `battle.py` + `pathfinding_heap.py`** ＋可视化 |
| 20 | **`574e946`** | 2026-09-18 | bugfix: partly implement river jumping logic. enhance: update visualizer script to correctly give observations to the new architecture. | **引擎 `battle.py`** ＋可视化 |
| 21 | `d428b79` | 2026-09-18 | Increase opponent randomness to make them harder to exploit | RL‑训练侧 |
| 22 | `1a86041` | 2026-09-18 | Update evaluation to add entropy diagnosis; Increase ent_coef | RL‑训练侧 |
| 23 | `7fc64f2` | 2026-09-18 | Update entropy calculation method | RL‑训练侧 |
| 24 | `f616f19` | 2026-09-18 | Use new placement head hoping to generalize better | RL‑训练侧 |

## 1.3 文件枚举完整性与分类口径（稿 1 §2 附带段，逐字）

**文件枚举完整性核对**：对 24 个提交逐个 `git show --name-status` 汇总，全部条目只有
`M` × 47、`A` × 3（`defensive_strategy.py`、`strategies.py`、`temp_test.py`）、`D` × 1（`watch_random_models.py`，在 `2aa7b5e` 删除，`git -C B log --diff-filter=D -- …/watch_random_models.py` 唯一命中该提交）。**无 R/C（改名/复制）**。
⇒ 不存在「用改名把引擎文件藏起来」的情形。

**分类口径**（稿 1 自定，逐字对齐任务书）：
- **引擎文件**：`battle.py, card_utils.py, arena.py, pathfinding_heap.py, pathfinding.py, core.py, player.py, timing.py`，且 `f20fa4d:<path>` 与 `HEAD:<path>` **两处都存在**（存在性核对见稿 1 §4 第 6 问）。
- **引擎数据**：`cards*.json / gamedata.json / tilemap_lane_grid.txt`。
- **RL‑训练侧**：`environment.py, evaluate.py, strategies.py, defensive_strategy.py, agent_pool.py, train*.py, benchmark_speed.py, special_eval.py, server.py, temp_test.py, watch_random_models.py`。
- 其余 `.py` = 可视化或工具；非 `.py` = 文档/其他。

**引擎命中汇总（稿 1 重点）**：24 个提交里 **6 个** 动过引擎文件 → 涉及文件仅 4 个：
`battle.py`（6 次）、`arena.py`（1 次）、`pathfinding_heap.py`（1 次）、`card_utils.py`（1 次）。
分叉点→B HEAD 的引擎净改动量：`git -C B diff --stat f20fa4d..HEAD -- <4 文件>`
→ `arena.py -8 / battle.py +70/‑… / card_utils.py 1 / pathfinding_heap.py +5`，合计 `4 files changed, 52 insertions(+), 33 deletions(-)`。

## 1.4 引擎改动逐条「行为级」摘要

### 1.4.1 `d1e0a16` — BUGFIX: fix troop collision issue mentioned by @billy948787（稿 1 §3.1，逐字）

- 文件：`src/clasher_new/battle.py`（`2 +/-`）；同提交可视化文件 `minimal_visualizer.py`、`new_visualization.py`（非引擎）。
- **类型：功能性修复**（重叠分离量被减半，改变单位物理位置，非注释非纯 perf）。
- 改了什么：`BattleState.resolve_collisions()` 中把 e1（被撞方）被推开的重叠量乘 **0.5**——即碰撞解算不再把 e1 按全量推开。

```diff
                     movement_ratio = e2.data.speed / (e1.data.speed+e2.data.speed)
                     e2.position.x += direction_vector.real*movement_ratio*overlap
                     e2.position.y += direction_vector.imag*movement_ratio*overlap
-                    e1.position.x += -direction_vector.real * (1-movement_ratio)*overlap
-                    e1.position.y += -direction_vector.imag * (1-movement_ratio)*overlap
+                    e1.position.x += -direction_vector.real * (1-movement_ratio)*overlap*0.5
+                    e1.position.y += -direction_vector.imag * (1-movement_ratio)*overlap*0.5
```
B HEAD 位置：`src/clasher_new/battle.py:756-757`（同文件 `:741 def resolve_collisions`）。

**【合并者复算】`git -C B show d1e0a16 -- src/clasher_new/battle.py`（原始输出）：**

```
commit d1e0a162d432b6ff90d9ed3ddc96ae02ef298beb
    BUGFIX: fix troop collision issue mentioned by @billy948787

@@ -729,8 +729,8 @@ class BattleState:
                     movement_ratio = e2.data.speed / (e1.data.speed+e2.data.speed)
                     e2.position.x += direction_vector.real*movement_ratio*overlap
                     e2.position.y += direction_vector.imag*movement_ratio*overlap
-                    e1.position.x += -direction_vector.real * (1-movement_ratio)*overlap
-                    e1.position.y += -direction_vector.imag * (1-movement_ratio)*overlap
+                    e1.position.x += -direction_vector.real * (1-movement_ratio)*overlap*0.5
+                    e1.position.y += -direction_vector.imag * (1-movement_ratio)*overlap*0.5
```
（diff 头 `index 730dcd9..e67f57a 100644`；作者 `jiahaoxiang <jiahaoxiang@mail.ustc.edu.cn>`，2026-09-14 19:39:53 +0800。）
⇒ 复算与稿 1 一致：该提交的引擎面**只有这 2 行**（`2 +/-` 指 2 行改动）。

### 1.4.2 `372ed0e` — BUGFIX: fix buildings hp do not decrease（@longlingking）（稿 1 §3.2，逐字 + 截断）

- 文件：`src/clasher_new/battle.py`（`3 +/-`）、`src/clasher_new/card_utils.py`（`1 +/-`）；另 `new_visualization.py`（非引擎）。
- **类型：功能性修复**（两处协同：生命周期单位 + 建筑掉血）。**这是本次唯一改动 `card_utils.py` 的提交。**
- 改了什么（三件事）：
  1. `Building.__init__` 插入 `battle_state` 形参（`persistent` 顺位后移）；
  2. 掉血衰减块：去掉 `self.data.lifetime > 0 and` 前置条件、去掉**重复的第二次** `self.take_damage(decay)`；
  3. `card_utils.Card.lifetime` 由 **毫秒改为秒**（`/1000`）。

```diff
 class
```
**【截断】稿 1 在此处被切断**（§0.2）。以下为 **【合并者复算】** 的同提交引擎面完整 diff：
`git -C B show 372ed0e -- src/clasher_new/battle.py src/clasher_new/card_utils.py`

```diff
commit 372ed0e70f2b983146e9521ad0b25d05d91c1f0b
    BUGFIX: fix buildings hp do not decrease, thanks to @longlingking

diff --git a/src/clasher_new/battle.py b/src/clasher_new/battle.py
index e67f57a..8121437 100644
--- a/src/clasher_new/battle.py
+++ b/src/clasher_new/battle.py
@@ -326,12 +326,13 @@ class Troop(Entity):
 class Building(Entity):
-    def __init__(self, id, position, player, card_name, persistent=False):
+    def __init__(self, id, position, player, card_name, battle_state=None, persistent=False):
         super().__init__(id, position, player, card_name)
         self.deploy_delay_remaining = self.data.deploy_time
         self.lifetime_elapsed = 0.0
         self.target_id = None
         self.tower_active = False
+        self.battle_state = battle_state
         self.persistent = persistent
         self.name = self.data.name
@@ -353,10 +354,9 @@ class Building(Entity):
             self.deploy_delay_remaining = max(0.0, self.deploy_delay_remaining - dt)
             return
         super().update(dt)
-        if self.data.lifetime > 0 and not self.persistent:
+        if not self.persistent:
             decay = (self.data.hp / float(self.data.lifetime)) * dt
             self.take_damage(decay)
-            self.take_damage(decay)
         if self.attack_cooldown > 0:
             self.attack_cooldown = max(0, self.attack_cooldown-dt*self.speed_buff*self.speed_debuff)
         target = self.update_current_target()

diff --git a/src/clasher_new/card_utils.py b/src/clasher_new/card_utils.py
index a11931d..e13c80b 100644
--- a/src/clasher_new/card_utils.py
+++ b/src/clasher_new/card_utils.py
@@ -87,7 +87,7 @@ class Card:
         self.charge_damage = self.data['summonCharacterData'].get('damageSpecial', 0)
         self.shield_health = self.data['summonCharacterData'].get('shieldHitpoints', 0)
 
-        self.lifetime = self.data['summonCharacterData'].get('lifeTime', float('inf'))
+        self.lifetime = self.data['summonCharacterData'].get('lifeTime', float('inf')) / 1000
 
         self.death_spawn_data = self.data['summonCharacterData'].get('deathSpawnCharacterData', {})
         self.death_area_effect = self.data['summonCharacterData'].get('deathAreaEffectData', {})
```

### 1.4.3 `fcb1d24` — BREAKING: 20Hz instead of 60Hz（【合并者复算】）

命令：`git -C B show fcb1d24 -- src/clasher_new/battle.py` / `git -C B show --stat fcb1d24`
统计：`battle.py 11 +++++++++--`、`environment.py 7 ++++---`、`new_visualization.py 6 +++---`（3 files changed, 16 insertions(+), 8 deletions(-)）

```diff
commit fcb1d24f63e99b2ef3e2f81bbaf5287fdcca47c5
    BREAKING: change simulator to be 20Hz instead of 60Hz, which significantly improves speed.

diff --git a/src/clasher_new/battle.py b/src/clasher_new/battle.py
@@ -282,7 +282,14 @@ class Troop(Entity):
             self.jumping_across_river = False
             self.data.is_air_unit = Card(self.name).is_air_unit
             self.speed = self.data.speed
-        current_target = self.update_current_target()
+        target = self.battle_state.entities.get(self.target_id)
+        if target is None or not target.is_alive or not target.targetable:
+            # scan immediately
+            current_target = self.update_current_target()
+        elif self.battle_state.tick % 2 == 0:
+            current_target = self.update_current_target()
+        else:
+            current_target = target
@@ -300,7 +307,7 @@ class Troop(Entity):
             else:
                 if not self.path:
                     self.path = EntityPathfinder(self, current_target, self.battle_state).calculate()
-                elif self.in_sight_range(current_target) and self.battle_state.tick % 10 == 0:
+                elif self.in_sight_range(current_target) and self.battle_state.tick % 3 == 0:
                     self.path = EntityPathfinder(self, current_target, self.battle_state).calculate()
```

### 1.4.4 `1e9716b` — battle.py 仅注释（【合并者复算】）

命令：`git -C B show 1e9716b -- src/clasher_new/battle.py`
统计（该提交全量）：`battle.py 10 +-`、`evaluate.py 153 ++++…`、`train_autoregressive.py 168 ++++…`（3 files changed, 314 insertions(+), 17 deletions(-)）
⇒ 引擎面**仅注释与空行**，无行为改动（与稿 1 §2 第 16 行的「仅注释」分类一致）。

```diff
commit 1e9716b506a743536104fe04b53013a8b656d83e
    Add some comments in `battle.py`; Update evaluation strategy …

@@ -41,14 +41,16 @@ class Entity:
         # If a card doesn't have special logic like the knight and mini-pekka, then only `BasicCharacter` will be
-        # used.
+        # used. Projectiles currently don't have EntityHolders, but we may need a special one for firecracker.
         self.entity_holder = BasicCharacter(self)
         if self.card_name in globals() and not isinstance(self, Projectile):
             self.entity_holder = eval(f"{self.card_name}(self)")
+
+        # Note that `on_spawn` is directly called when this object is initialized, so we only create entity objects directly
+        # before we spawn them.
         self.entity_holder.on_spawn()
 
         self.path = []
-
         self.pending_damage = []
@@ -70,6 +72,7 @@ class Entity:
         """Automatically call entity holder's on_death to prevent bugs"""
         self.is_alive = False
         self.entity_holder.on_death()
+        # We also call `battle_state.on_death()` because the death of a princess tower should activate the king tower.
         self.battle_state.on_death(self)
@@ -93,7 +96,6 @@ class Entity:
             self.take_damage(pending_damage, delayed=False)
         self.pending_damage = []
 
-
     def take_damage(self, amount: float, delayed=False):
@@ -283,6 +285,8 @@ class Troop(Entity):
             self.speed = self.data.speed
         target = self.battle_state.entities.get(self.target_id)
+
+        # I used this weird target selection logic to save time, so target are calculated every 0.1s max instead of every frame
         if target is None or not target.is_alive or not target.targetable:
             # scan immediately
             current_target = self.update_current_target()
```

### 1.4.5 `2daab60` — A\* 与河道阻挡不自洽的修复（【合并者复算】；**本节与 §2 同一对象**）

命令：`git -C B show 2daab60 -- src/clasher_new/arena.py src/clasher_new/battle.py src/clasher_new/pathfinding_heap.py`
统计：`arena.py 8 --------`、`battle.py 22 ++++++++++------------`、`new_visualization.py 6 ++----`、`pathfinding_heap.py 5 ++++-`（4 files changed, 16 insertions(+), 25 deletions(-)）

```diff
commit 2daab60dae3c19ad80f406947eb22b00ed90a867
    BUGFIX: previous code settings are not compatible with the new A* pathfinding algorithm,
    especially the part that handles river jumps. I removed the river tiles from blocked_tiles.

diff --git a/src/clasher_new/arena.py b/src/clasher_new/arena.py
@@ -19,10 +19,6 @@ class TileGrid:
     RIVER_Y1 = 15.0
     RIVER_Y2 = 16.0
     BLOCKED_TILES = [
-        # Edge tiles next to river
-        (0, 15), (0, 16), (1, 15), (1, 16),
-        *[(i, j) for i in range(5, 13) for j in range(15, 17)], # (5, 15) to (12, 16)
-        (16, 15), (16, 16), (17, 15), (17, 16),
         
@@ -54,10 +50,6 @@ class TileGrid:
             return walkable_cache[int_pos]
         if not self.is_valid_position(pos) or self.is_blocked_tile(int(pos.x), int(pos.y)):
             walkable_cache[int_pos] = False
-        elif self.RIVER_Y1 <= pos.y <= self.RIVER_Y2:
-            on_left_bridge = 2.0 <= pos.x < 5.0
-            on_right_bridge = 13.0 <= pos.x < 16.0
-            walkable_cache[int_pos] = on_left_bridge or on_right_bridge
         else:
             walkable_cache[int_pos] = True
         return walkable_cache[int_pos]

diff --git a/src/clasher_new/battle.py b/src/clasher_new/battle.py
@@ -272,7 +272,7 @@ class Troop(Entity):
         if self.deploy_delay_remaining > 0:
             self.deploy_delay_remaining = max(0.0, self.deploy_delay_remaining - dt)
-            return # Haven't finished deploying yet
+            return
@@ -280,6 +280,7 @@ class Troop(Entity):
         if self.name != 'Miner':
             super().update(dt)
         # The miner needs to update before deployment.
+
         if self.jumping_across_river and self.on_both_sides_of_river(self.start_jumping_position):
@@ -299,13 +300,13 @@ class Troop(Entity):
         # Move towards target if out of attack range
-        if (not self.in_attack_range(current_target)) or self.jumping_across_river:
-            has_jump_ability = self.data.jump_speed and self.on_both_sides_of_river(current_target) and self.near_river() and self.in_sight_range(current_target)
-            if not self.jumping_across_river and has_jump_ability:
-                self.start_jumping_position = Position(self.position.x, self.position.y)
-                self.jumping_across_river = True
-                self.data.is_air_unit = True
-                self.speed = self.data.jump_speed
+        if not self.in_attack_range(current_target):
+            # has_jump_ability = self.data.jump_speed and self.on_both_sides_of_river(current_target) and self.near_river() and self.in_sight_range(current_target)
+            # if not self.jumping_across_river and has_jump_ability:
+            #     self.start_jumping_position = Position(self.position.x, self.position.y)
+            #     self.jumping_across_river = True
+            #     self.data.is_air_unit = True
+            #     self.speed = self.data.jump_speed
@@ -323,10 +324,7 @@ class Troop(Entity):
                 if dot >= 0:
                     # move towards next waypoint
                     index += 1
-                if index == len(self.path):
-                    self.move_towards(current_target.position, dt, True)
-                else:
-                    self.move_towards(self.path[index], dt, True)
+                self.move_towards(self.path[index], dt, True)

diff --git a/src/clasher_new/pathfinding_heap.py b/src/clasher_new/pathfinding_heap.py
@@ -92,7 +92,10 @@ class EntityPathfinder:
                 px, py = current
                 tile_char = contents[63-ny][nx]
                 if tile_char == 'W':
-                    tile_cost = 800 if not self.entity.data.is_air_unit else 7
+                    if self.entity.data.is_air_unit or self.entity.data.jump_speed:
+                        tile_cost = 7
+                    else:
+                        tile_cost = 50
                 elif tile_char == '.':
                     tile_cost = 8
                 else:
```

### 1.4.6 `574e946` — 部分实现过河跳跃（【合并者复算】；**本节与 §2 同一对象**）

命令：`git -C B show 574e946 -- src/clasher_new/battle.py`
统计：`battle.py 25 +++++++++++++++----`、`minimal_visualizer.py 49 ++++…`（2 files changed, 56 insertions(+), 18 deletions(-)）

```diff
commit 574e946db4f714237708a331cdfc3169a45a8437
    bugfix: partly implement river jumping logic.
    enhance: update visualizer script to correctly give observations to the new architecture.

diff --git a/src/clasher_new/battle.py b/src/clasher_new/battle.py
@@ -281,10 +281,10 @@ class Troop(Entity):
         # The miner needs to update before deployment.
 
-        if self.jumping_across_river and self.on_both_sides_of_river(self.start_jumping_position):
-            self.jumping_across_river = False
-            self.data.is_air_unit = Card(self.name).is_air_unit
-            self.speed = self.data.speed
+        # if self.jumping_across_river and self.on_both_sides_of_river(self.start_jumping_position):
+        #     self.jumping_across_river = False
+        #     self.data.is_air_unit = Card(self.name).is_air_unit
+        #     self.speed = self.data.speed
         target = self.battle_state.entities.get(self.target_id)
@@ -324,7 +324,22 @@ class Troop(Entity):
                 if dot >= 0:
                     # move towards next waypoint
                     index += 1
-                self.move_towards(self.path[index], dt, True)
+
+                if index == len(self.path):
+                    self.move_towards(current_target.position, dt, True)
+                else:
+                    waypoint = self.path[index]
+                    waypoint_in_river = 17 >= waypoint.y >= 15
+                    distance = waypoint.distance_to(self.position)
+                    if self.jumping_across_river:
+                        if not waypoint_in_river:
+                            if distance < self.speed * dt:
+                                self.jumping_across_river = False
+                                self.data.is_air_unit = False
+                                self.speed = self.data.speed
+                    elif waypoint_in_river:
+                        pass
+                    self.move_towards(waypoint, dt, True)
```

## 1.5 净改动量（【合并者复算】交叉核对，与稿 1 一致）

```
$ git -C B diff --stat f20fa4d..HEAD -- src/clasher_new/battle.py src/clasher_new/arena.py src/clasher_new/pathfinding_heap.py src/clasher_new/card_utils.py
 src/clasher_new/arena.py            |  8 -----
 src/clasher_new/battle.py           | 70 +++++++++++++++++++++++++------------
 src/clasher_new/card_utils.py       |  2 +-
 src/clasher_new/pathfinding_heap.py |  5 ++-
 4 files changed, 52 insertions(+), 33 deletions(-)
```
（与稿 1 §2 的「4 files changed, 52 insertions(+), 33 deletions(-)」**逐字相同**。）

**分叉点→B HEAD 的全仓净改动（17 个文件；稿 1 未列，【合并者复算】补）：**

```
$ git -C B diff --stat f20fa4d..HEAD
 README.md                               |   2 +-
 src/clasher_new/agent_pool.py           |   2 +-
 src/clasher_new/arena.py                |   8 -
 src/clasher_new/battle.py               |  70 ++++---
 src/clasher_new/benchmark_speed.py      | 148 +++-----------
 src/clasher_new/card_utils.py           |   2 +-
 src/clasher_new/defensive_strategy.py   | 161 +++++++++++++++
 src/clasher_new/environment.py          |  67 +++++--
 src/clasher_new/evaluate.py             | 319 +++++++++++++++++++++++++----
 src/clasher_new/minimal_visualizer.py   |  50 +++--
 src/clasher_new/new_visualization.py    |   9 +-
 src/clasher_new/pathfinding_heap.py     |   5 +-
 src/clasher_new/strategies.py           | 334 +++++++++++++++++++++++++++++++
 src/clasher_new/temp_test.py            |  29 +++
 src/clasher_new/train.py                |  87 ++++----
 src/clasher_new/train_autoregressive.py | 344 ++++++++++++++++++++++++++++++--
 src/clasher_new/watch_random_models.py  |  17 --
 17 files changed, 1336 insertions(+), 318 deletions(-)
```

## 1.6 【冲突并列 §1-C1】共同祖先 / 是否独立历史线 —— 两说并列，不调和

**说一（稿 1，§1 表）**：`f20fa4d` 是 A、B 两仓 HEAD 的**共同祖先 / 分叉点**。
- `git -C B merge-base --is-ancestor f20fa4d HEAD` → **rc=0（YES）**
- `git -C A merge-base --is-ancestor f20fa4d HEAD` → **rc=0（YES）**
- 稿 1 的踩坑记录：`git -C A cat-file -t 2aa7b5e`（B 独占提交）**失败**、`git merge-base HEAD /mnt/e/…-by-jason/HEAD` → `fatal: Not a valid object name`；稿 1 的处置是「B 的独占集合必须在 B 里算，A 只提供 `f20fa4d` 这一侧的坐标」。

**说二（`docs/upstream_claim_audit_2026-09-19.md:231-232`，逐字）**：
> 「本机 clone 的 HEAD 是 `8d2552a`，与上游 f616f19 **无共同祖先**（`git merge-base --is-ancestor` 返回 **NO**，`f616f190…` 在本地都不是合法 commit 名）⇒ 本机项目是**独立历史线**（最多算概念/代码层面的衍生），**不能**用「本地 HEAD」当作上游 HEAD 的证据。」

**【合并者复算】只列原始输出（不裁决）：**

```
$ git -C A merge-base --is-ancestor f20fa4d HEAD ; echo "rc=$?"
rc=0
$ git -C A cat-file -t f616f19 ; echo "rc=$?"
fatal: Not a valid object name f616f19
rc=128
$ git -C B merge-base --is-ancestor f20fa4d HEAD ; echo "rc=$?"
rc=0
```

> 并列要点（事实，不调和）：说一与说二**用了不同的判据对象**——说二判的是 `f616f190…`（上游 HEAD）在 **A 的 object store** 里是否存在/是否为祖先，说一判的是 `f20fa4d`（分叉点）在 A、B 两侧的祖先关系。**本合并不判定哪一说成立**；若后续要裁决，需另立取证。

## 1.7 【截断】稿 1 未覆盖部分

稿 1 在 §3.2 后截断（§0.2）。**其 §4「存在性核对（第 6 问）」等未见，不代写**；本节 1.4.3–1.4.6 与 1.5 的复算**只补 diff 层面证据，不代替稿 1 的判定**。

---

# §2 过河与 A\*（我们 vs 上游 2daab60 / 574e946）

> 本节主体 = 稿 2（标题《过河/寻路：我们 vs 上游 2daab60 / 574e946 — 只读取证报告》），**逐字保留**。
> **截断说明**：稿 2 在 §1.6 末尾被截断（见 §0.2）；其 §0 提到的「我方另有 3 个真实问题（1 个高严重度，见 §4）」**未见**，不代写。

## 2.1 取证环境与手段（稿 2 开头，逐字）

**取证环境**：`/mnt/e/clash-royale-simulator-main`（HEAD `8d2552a`）、上游 `/mnt/e/clash-royale-simulator-main-by-jason`（HEAD `f616f19`）。未修改/创建任何文件，未跑训练，无 git 写操作。
**取证手段**：文件读取 + `git show` 三方对账 + `python3 -B`（**禁止写 `.pyc`**）经 **stdin** 传入代码运行真实引擎（未落盘脚本）。
⚠️ 仓库 `.venv` 是 Windows venv（`Scripts/python.exe`），WSL 下不可执行，故用系统 `python3 3.14.4`；`import battle` 成功。**这不算"按 `docs/agents/env.md` 的正式运行姿势"**，但本轮全部结论都有代码位置或受控实验支撑。

## 2.2 一句话结论（稿 2 §0，逐字）

**我们是"第三种实现"——但它不是重写，而是"上游 2daab60 之前的 arena/jump + 上游 2daab60 之后的另一条支线"的混合体**：`arena.py` 的 `BLOCKED_TILES` 与 `is_walkable` **逐字节等于分叉点**（保留河面阻挡 + 桥面区间），`battle.py` 的 jump 触发块与 waypoint 分支**逐字节等于 2daab60 之前**，而 `pathfinding_heap.py` 的 W 代价卡在**最老的 800**（既非 before 也非 after 的 50）。

**上游所修的那个缺陷（河面阻挡与 A\* 不自洽）在我们这套实现里不存在** —— 我们的 `BLOCKED_TILES` 与 `is_walkable` 桥面区间自洽，且已实测地面单位可正常过桥、塔毁后解锁部署区可达。**但我们在过河/寻路上另有 3 个真实问题**（1 个高严重度，见 §4）。→ **【截断】§4 全文未见。**

## 2.3 逐行定位（稿 2 §1；file:line + 代码引用逐字）

### 2.3.1 `src/clasher_new/arena.py` — `BLOCKED_TILES`

```python
# arena.py:19-34
19:     RIVER_Y1 = 15.0
20:     RIVER_Y2 = 16.0
21:     BLOCKED_TILES = [
22:         # Edge tiles next to river
23:         (0, 15), (0, 16), (1, 15), (1, 16),
24:         *[(i, j) for i in range(5, 13) for j in range(15, 17)], # (5, 15) to (12, 16)
25:         (16, 15), (16, 16), (17, 15), (17, 16),
26:         
28:         (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0),
29:         (12, 0), (13, 0), (14, 0), (15, 0), (16, 0), (17, 0),
32:         (0, 31), ... (5, 31),
33:         (12, 31), ... (17, 31),
34:     ]
```
河面条目 = **全部 32 格**：左侧 `(0,15)(0,16)(1,15)(1,16)`、中间 `(5..12, 15..16)` 共 16 格、右侧 `(16,15)(16,16)(17,15)(17,16)`。**注意 `(2..4, 15/16)` 与 `(13..15, 15/16)` 不在表内** —— 那 12 格正是桥面。

### 2.3.2 `arena.py` — `is_walkable`

```python
# arena.py:106-119
106:    def is_walkable(self, pos: Position) -> bool:
107:        x, y = pos.x, pos.y
108:        int_pos = (int(x), int(y))
109:        if int_pos in walkable_cache:
110:            return walkable_cache[int_pos]
111:        if not self.is_valid_position(pos) or self.is_blocked_tile(int(pos.x), int(pos.y)):
112:            walkable_cache[int_pos] = False
113:        elif self.RIVER_Y1 <= pos.y <= self.RIVER_Y2:      # ← 上游 2daab60 删掉了这一段
114:            on_left_bridge = 2.0 <= pos.x < 5.0
115:            on_right_bridge = 13.0 <= pos.x < 16.0
116:            walkable_cache[int_pos] = on_left_bridge or on_right_bridge
117:        else:
118:            walkable_cache[int_pos] = True
119:        return walkable_cache[int_pos]
```
`walkable_cache` 是**模块级全局字典**（`arena.py:5`），key = `(int(x), int(y))` —— 即"格坐标"，值是该格的可行走性（与查询点的亚格精度无关）。

### 2.3.3 `battle.py` — `pathfind_ground_walkable` / `ground_walkable`

```python
# battle.py:3243-3250
3243:    def pathfind_ground_walkable(self, position, mover_radius):
3244:        if not self.arena.is_walkable(position): return False
3245:        x, y = position_to_cell(position)
3246:        return self.building_cache[x][y] > mover_radius
3247:
3248:    def ground_walkable(self, position, mover_radius):
3249:        if not self.arena.is_walkable(position): return False
3250:        return not self.is_position_occupied_by_building(position, mover_radius)
```
- `pathfind_ground_walkable` = `is_walkable` **∧** 静态距离场 `building_cache`（A\* 用，**含塔**）。
- `ground_walkable` = `is_walkable` **∧** 动态占位（**含活塔矩形**，`battle.py:3252-3272`；`_tower_footprint_blocks` 走 `arena.tower_rect_dist(..., self)`，**按 `_tower_alive` 过滤活塔**，`arena.py:79`）。
- `building_cache` 构造于 `battle.py:3221-3242`：**先把 6 个塔矩形无条件铺进距离场**（`:3227-3234`），再叠建筑面积距离（`:3235-3242`）。

### 2.3.4 `battle.py` — `resolve_collisions`

```python
# battle.py:3274-3303（关键差异行）
3279:        ground_troops = combinations([each for each in entities_alive
3280:                                      if not each.data.is_air_unit
3281:                                      and getattr(each.entity_holder, '_mk_jump', None) is None], 2)
3282:        flying_troops = combinations([each for each in entities_alive if each.data.is_air_unit], 2)
...
3287:                if e1.id <= 6 and getattr(e1, 'persistent', False):
3288:                    self._push_troop_out_of_tower(e2, e1); continue
3298:                    if total_speed == 0: continue
```
推挤**本身不做可行走性判定**，也不把单位推回岸上；越界/河岸修正是 `ensure_walkability` 的职责，而它**不推挤**。上游 HEAD 的 `resolve_collisions`（`by-jason src/clasher_new/battle.py:741-758`）无 `_mk_jump` 排除、无塔矩形分支、无除零保护 —— 我们这边是独立增强。

### 2.3.5 `battle.py` — `Troop.update` 过河跳跃块

```python
# battle.py:1187-1190  ← 跳跃复位（上游 574e946 把它注释掉了）
1187:        if self.jumping_across_river and self.on_both_sides_of_river(self.start_jumping_position):
1188:            self.jumping_across_river = False
1189:            self.data.is_air_unit = Card(self.name).is_air_unit
1190:            self.speed = self.data.speed

# battle.py:1232-1247  ← 跳跃触发（上游 2daab60 把它整段注释掉了）
1232:        # Move towards target if out of attack range
1233:        if (not self.in_attack_range(current_target)) or self.jumping_across_river:
1234:            has_jump_ability = self.data.jump_speed and self.on_both_sides_of_river(current_target) and self.near_river() and self.in_sight_range(current_target)
1235:            if not self.jumping_across_river and has_jump_ability:
1236:                self.start_jumping_position = Position(self.position.x, self.position.y)
1237:                self.jumping_across_river = True          # ← 全场唯一置 True 处
1238:                self.data.is_air_unit = True
1239:                self.speed = self.data.jump_speed
1240:            if self.data.is_air_unit:
1241:                self.move_towards(current_target.position, dt, True)
1242:            else:
...
1265:                # determine the next waypoint and move towards that waypoint
1266:                min_point = min(self.path, key=lambda pos: pos.distance_to(self.position))
...
1274:                if index == len(self.path):
1275:                    self.move_towards(current_target.position, dt, True)
1276:                else:
1280:                    _wp = self.path[index]
1281:                    _off = self._lane_offset
1287:                    self.move_towards(_wp, dt, True)
```

**配套四个辅助（稿 2 表，逐字）：**

| 符号 | 位置 | 代码 |
|---|---|---|
| `jumping_across_river` 初始化 | `battle.py:834-835` | `self.jumping_across_river = False` / `self.start_jumping_position = None` |
| `on_both_sides_of_river` | `battle.py:811-816` | `if y < 15.0: return self.position.y > 17.0` `else: return self.position.y < 15.0` |
| `near_river` | `battle.py:818-819` | `return abs(self.position.y-15.0)<self.data.collision_radius or abs(self.position.y-17.0)<self.data.collision_radius` |
| 跳跃期岸上豁免 | `battle.py:2677` | `if entity.jumping_across_river and self.in_river(entity.position): return` |
| `in_river` | `battle.py:2664-2668` | 硬编码 32 格河面表（与 `BLOCKED_TILES` 河面部**逐项相同**） |
| `jump_speed` 数据源 | `card_utils.py:315-316` | `self.jump_speed = self.data['summonCharacterData'].get('jumpSpeed', 0) / 60` |
| `jump_speed` 被清零 | `battle.py:896-897` | RoyalHogs 飞行期 `self.data.jump_speed = 0` |

### 2.3.6 `pathfinding_heap.py` — `tile_char == 'W'` 代价

```python
# pathfinding_heap.py:132-142
132:                tile_char = contents[63-ny][nx]
133:                if tile_char == 'W':
134:                    tile_cost = 800 if not self.entity.data.is_air_unit else 7   # ← 我们
135:                elif tile_char == '.':
140:                    tile_cost = 5      # 桥面（我们；上游 = 8）
141:                else:
142:                    tile_cost = 5
143:                if nx != px and ny != py: geo_cost = 14
145:                else:                     geo_cost = 10
147:                step_cost = tile_cost * geo_cost
```
**关键事实（稿 2 自述「新证据，之前没人查过」）**：`tilemap_lane_grid.txt` 的实际字符分布

```
char histogram: {'.': 1500, '1': 346, '2': 346, 'W': 112}
engine y=15: WWW11WWWWWWWWW22WW
engine y=16: WWW1.WWWWWWWWW2.WW
```
⇒ **`'W'` 在整张图上只出现 112 次，全部落在 y=15/16 两行**；桥面格（`x=2,3,4` 与 `13,14,15`）是 `'.'`。
⇒ 所以 `tile_cost = 800`（我们的地面 W 代价）**恰恰只在过**  ← **【截断】稿 2 在此处被切断**（§0.2），该句余下与 §4 全文未见。

## 2.4 【合并者复算 · 交叉引用 §1.4.5 / §1.4.6】上游侧的对应改动（原文 diff）

稿 2 引用上游时给的是 `by-jason` 行号（`battle.py:741-758`）；上游完整的对应 diff 已在 **§1.4.5（2daab60）/ §1.4.6（574e946）** 逐字给出，本节不重复。**两侧可直接对账的锚点（原文并列，不调和）：**

| 锚点 | 我方 A（`8d2552a`） | 上游 B（`f616f19`） |
|---|---|---|
| 河面阻挡 / 桥面区间 | `arena.py:21-34` 河面 32 格 + `arena.py:113-116` 桥面半开区间（= 分叉点原样） | `2daab60` 删除 `arena.py` 河面条目 8 行与 `is_walkable` 桥面分支 4 行 |
| W 代价（地面非跳单位） | `pathfinding_heap.py:134` `tile_cost = 800` | `2daab60` 改为 `50`（空中或 `jump_speed` 为 `7`） |
| `.`（桥面/地面）代价 | `pathfinding_heap.py:140/142` `tile_cost = 5` | 该提交处 `tile_cost = 8`（未随 2daab60 改动） |
| jump 触发块 | `battle.py:1232-1247` 完整保留（含唯一置 `True` 处） | `2daab60` 整段注释掉 |
| jump 复位块 | `battle.py:1187-1190` 保留 | `574e946` 注释掉；改由 waypoint 分支内的 `waypoint_in_river` 判定复位（`574e946` 新增） |
| waypoint `index == len(path)` 分支 | `battle.py:1274-1276` 保留（越界时直指 `current_target.position`） | `2daab60` 删除该分支、直接 `self.move_towards(self.path[index], …)`；`574e946` 又恢复 |

## 2.5 【冲突并列 §2-C1】同一符号在仓内文档中的行号 —— 两说并列，不调和

- **稿 2（本批，HEAD `8d2552a`）**：`pathfind_ground_walkable` = `battle.py:3243-3250`；`ground_walkable` = `battle.py:3248-3250`；`is_walkable` 调用点 `battle.py:3244/3249`。（【合并者复算】抽查确认 `battle.py:3274 def resolve_collisions`、`3288/3290 _push_troop_out_of_tower`、`3295/3298/3299-3303` 与稿 3 引用一致。）
- **`docs/game_engine.md`（2026-09-14 快照）**：`pathfind_ground_walkable(pos, mover_radius)` = `battle.py:3053-3056`；`ground_walkable(pos, mover_radius)` = `battle.py:3058-3060`；`ensure_walkability` 调用点 `battle.py:2590`；同文档另注 `is_walkable` 桥面 = `arena.py:113-116`、`BLOCKED_TILES` = `arena.py:21-34`（与稿 2/本合并一致）。
- 两说并列（行号不同、`arena.py` 部分相同），**不调和**。文件时间戳：`docs/game_engine.md` 早于 A HEAD `8d2552a`；本合并只记录该事实，不据此裁决哪一组行号「对」。

---

# §3 碰撞（上游 `d1e0a16` 的 `*0.5` 是否应移植）

> 本节主体 = 稿 3（标题《审查报告：上游 `d1e0a16`（troop collision `*0.5`）是否应移植》），**逐字保留**。
> **截断说明**：稿 3 在 §3 的仪器命令块内被截断（见 §0.2）。

## 3.0 结论表（稿 3 §0，逐字）

**取证范围（全部只读）**：我方 `src/clasher_new/battle.py`、`arena.py`、`rl/replay.py`、`rl/env_wrapper.py`、`rl/run_league.py`、`docs/`、`scripts/`；上游 `/mnt/e/clash-royale-simulator-main-by-jason`（`git log/show` 只读）。
**未做**：任何写操作、任何训练、任何 git 写。
**时间**：本轮会话读取的工作树状态。

| # | 问题 | 严重程度 | 位置 | 原因 / 结论 |
|---|---|---|---|---|
| C1 | 上游 `*0.5` **未移植**到我们引擎 | —（事实） | `src/clasher_new/battle.py:3302-3303` | 上游 `d1e0a16` 是 `f20fa4d` 之后**唯一**的引擎碰撞修复（`git log --oneline f20fa4d..HEAD`），其 battle.py 部分未进入我方；我方该两行无 `*0.5` |
| C2 | `*0.5` 使**单帧永不完成分离**（f∈[0.5,1.0]） | 低-中（视觉欠分离） | `battle.py:3299-3303` | 只有 `e2` 一侧吃满位移；`e1` 只吃 `(1−ratio)` 的 50% ⇒ 残余 `0.5·overlap·(1−ratio)` 每帧几何衰减（比例 0.75），不是"穿模"，也不是"卡住" |
| C3 | **不负责**消除症状：双建筑重叠永久穿透 | 低 | `battle.py:3298`（M1 `continue`） | 两侧 `speed==0` ⇒ `total_speed==0` ⇒ 直接 `continue`，`*0.5` 与 M1 正交；两个 0 速建筑若出生即重叠，**永远**保持重叠（只靠 `ensure_walkability` 的矩形/边界修正，见 C4） |
| C4 | e1 的最终位置**不止**由本处决定；本处之后**无夹取** | 低（既有，非新） | `battle.py:2881-2884`、`:2670-2692`、`:2728` | tick 内顺序 = `entity.update(dt)`(移动) → `ensure_walkability`(夹取/矩形推出) → `resolve_collisions`(对推) → 出兵队列 → `time+=dt`。碰撞写在夹取**之后**，所以碰撞写入的位置**到下一 tick 才被修正**（1 帧窗口）。这是上游同构问题（上游 `step()` 同样是 `update→ensure_walkability→resolve_collisions`） |
| C5 | `resolve_collisions` 内的**顺序耦合**使塔-部队推出可被随后的一对推回塔里 | 低-中（既有） | `battle.py:3287-3303` | 塔 id≤6 恒排在 `entities` 序前列 ⇒ `(塔,X)` 对**先**执行；`(W,X)`（W<X）**后**执行，可把 X 推回塔矩形内 1 帧。`*0.5` 既不加剧也不缓解（塔路径是 `_push_troop_out_of_tower`，不走 `*0.5`） |
| C6 | `if abs(direction_vector)==0: return` 是**整个函数早退**（不是跳过该对） | 低（潜伏） | `battle.py:3295` | 与上游逐字相同；同心重叠时会中断剩余所有对的解算。上游 `git show` 确认未改 |
| C7 | 建筑-塔对会把**建筑**当"部队"推出（违反注释声明的"位移全给部队"） | 低（既有） | `battle.py:3287-3290` + `:3305-3309` | `_push_troop_out_of_tower` 只在 `arena.towers` 里按坐标匹配塔、**不校验 `troop` 类型**；`(塔, 建筑)` 对会命中。上游无此分支 |
| C8 | `rel_ratio=1` 分支（e2 为 0 速建筑）与 `*0.5` 无关：现在就是**满额**推开 | 信息 | `battle.py:3302-3303` 算式 | 当 0 速建筑落在 `e2` 位置（`combinations` 顺序决定），e1 位移 = `1×overlap`（不乘 0.5）。所以 `*0.5` 的效果**取决于迭代顺序**，不是一个均匀的"阻尼系数" |
| C9 | 碰撞族**无任何现有测试**（含 `*0.5` 无回归） | 中 | `src/clasher_new/rl/selftests/*.py` grep `resolve_collisions` = 0 命中；无 `tests/` 目录 | 上游那个 BUGFIX 提交本身也**没带测试**（`git show d1e0a16 --stat`：battle.py + 两个 visualizer） |
| C10 | 上游修复的**症状描述无法核实** | 中（未验证） | web 检索 | 检索未命中 `billy948787` 的 issue 原文（见 §6）；因此"症状是否仍存在于我方"**只能推理 + 需要实测**，不能断言 |
| C11 | 吞吐不成问题：碰撞只占单 tick 的 **2.3–2.6%** | 信息 | `docs/engine_tick_cost_2026-09-14.md:133-136` | 所以"多迭代几次分离"的替代方案代价可忽略 |

## 3.1 事实核对（稿 3 §1，逐字）

| 项 | 我方（重写实现） | 上游（jason 仓库） |
|---|---|---|
| 分叉点后的碰撞修复 | 无 | `d1e0a16`；`git log --oneline f20fa4d..HEAD` 显示它之后**只有**它一个 BUGFIX（其余为 visualizer/entropy/placement head） |
| e1 位移 | `battle.py:3302-3303` 无系数 | `battle.py:756-757` `...*overlap*0.5` |
| e2 位移 | `battle.py:3300-3301` 不含 0.5 | `battle.py:754-755` 同样不含 0.5 |
| 除零保护 | `battle.py:3298` `if total_speed == 0: continue`（M1） | 无（`754` 行直接除） |
| 塔 | 矩形推出 `_push_troop_out_of_tower` `battle.py:3305-3336` | 无：塔走圆形 ovelap 分支（`git grep _push_troop` 上游 = 0 命中） |
| 空中/跳 | `_mk_jump` 排除 + `is_air_unit` 分组 `battle.py:3279-3282` | 仅 `is_air_unit` 分组 |
| 数据源 | `card_utils.py:269/533` `collisionRadius/1000`（0.3–1.4 格） | 同源 |
| 生产 tick | `RLEnv` 默认 `dt = 1/60`（`env_wrapper.py:89`），生产 `_make_env` 未覆盖 dt（`run_league.py:717-719`）⇒ 2.4 ms/帧 | 20 Hz（上游 `fcb1d24 BREAKING: 20Hz`） |

## 3.2 数学分析（稿 3 §2，逐字）

记 `r = movement_ratio = s2/(s1+s2) ∈ [0,1]`，`Δ = overlap`，`f` 为施加在 `e1` 那一侧的系数（我方 `f=1`，上游 `f=0.5`）。

**单帧、成对、孤立观看**（`battle.py:3291-3303`）：
```
e2 位移 =  Δ·r              （两版相同，未乘系数）
e1 位移 =  Δ·(1−r)·f        （我方 f=1，上游 f=0.5）
net 分离量 = Δ·( r + f·(1−r) ) = Δ·( f + (1−f)·r )
相对欠分离 = Δ − net = Δ·(1−f)·(1−r)
```
代入 `f=0.5`：
```
net = Δ·(0.5 + 0.5·r)          ← 题目给的公式，正确
欠分离 = 0.5·Δ·(1−r)           ← 正确
残余比例 (1−net/Δ) = 0.5·(1−r)
```
**逐帧几何衰减**：分离是"乘系数"而非"置位"，所以下一帧按比例缩小：
```
f_eff := net/Δ = 0.5 + 0.5r ∈ [0.5, 1]
第 n 帧残余 = 0.5^n·(1−r)·Δ     （r 恒定、无新注入时）
```
⇒ 4 帧后只剩 `0.0625·(1−r)·Δ`。**所以"重叠永不消失"不成立**（除 C3 的 0 速对）。

**关键：0.75 是"乘性阻尼倍数"，不是 0.5**。逐对来看 `f_eff ∈ [0.5,1]`（均值随 r 变），残差按 `(1−f_eff) ∈ [0,0.5]` 衰减 —— 这就是"抖动阻尼"的机制：多单位堆叠时，过冲（overshoot）按 `(1−f_eff)` 被砍半，收敛由"可能发散/振荡"变为"单调但更慢"。

**各 ratio 端点后果**

| 场景 | r | f_eff | 单帧 net | 一次性推开的概率 |
|---|---|---|---|---|
| e2 是 0 速建筑/器械，快单位是 e1 | 0 | 0.5 | `0.5Δ` | 需 2 帧（残余 `0.25Δ`→`0.125Δ`→…） |
| e2 是快单位、e1 是 0 速建筑 | 1 | 1.0 | `Δ` | **1 帧满额**（`*0.5` 被 `(1−r)=0` 吃掉） |
| 等速对撞 | 0.5 | 0.75 | `0.75Δ` | 2 帧（`0.25Δ`→`0.0625Δ`） |
| 一方速度为 0 且是 `e2`（建筑/塔的**另一侧**顺序） | 0 | 0.5 | 缓慢但收敛 | 收敛 |
| **双方速度都是 0**（双建筑、或塔-建筑） | — | **不进入公式** | — | `battle.py:3298` `continue` ⇒ **永久重叠**（M1 既有；`*0.5` 完全不涉） |

**叠塔/穿模风险评估**：
1. **不会新增"穿模/隧穿"**：推进步长是 `speed·dt`（`battle.py:1155-1161`），生产 `dt=1/60`、`speed≈1.2 格/s`（`card_utils.py:272` 的 `speed/50`）⇒ 单帧位移 ≈ `0.02` 格，远小于任何 `collision_radius`（0.3–1.4）。碰撞是离散逐帧判定，**几何上不可能跨过一整个碰撞体**——位置只是停在"重叠"状态，视觉上是"叠了一点"，不是"穿过去"。
2. **会新增可视欠分离**：稳态近似（每帧注入 `v_rel·dt` 的新重叠、每帧按 `f_eff` 衰减）：
   `r* ≈ (1−f_eff)/f_eff · v_rel·dt ≈ 0.5·v_rel·dt`（f=0.5 时），对比 `f=1` 的 `r* ≈ 0.06·v_rel·dt`。取等速对撞 `v_rel ≈ 0.04 格/帧` ⇒ **`*0.5` 的稳态重叠 ≈ 0.02 格，原式 ≈ 0.002 格**。相对 `collision_radius=0.5` 是 **4% vs 0.4%** —— 肉眼可辨的"贴脸略叠"，不是"穿模"。（**该稳态式为解析近似，未实测**；r 恒定、忽略多体耦合与顺序效应。）
3. **不会把单位挤进塔/建筑**：这是**顺序**问题，不是系数问题（C5）。塔-部队走 `_push_troop_out_of_tower`（`battle.py:3305-3336`），**满额**推出且与 `*0.5` 无关；被"挤回去"的那 1 帧来自后续 `(W,X)` 对的**满额** `e2` 位移——这一项在两版里完全相同。
4. **对博弈量的影响**：位置不参与 `game_over`/加时裁决（`battle.py:2840-2868` 只看塔血/皇冠），所以 `*0.5` 不是"胜负语义"改动；但它会轻微改变 `center-distance` 类判定（溅射 `battle.py:1670/1680`、法术罩圈 `:1857/1996`、`edge_distance_from`）的边缘命中，即**逐帧位置 → 伤害边界**的耦合。

**判定**：`*0.5` = **欠分离阻尼**，不是引入穿模；也不是"消除抖动"的完整解（它同时把分离速度砍半）。它属于"用一点稳态重叠换一点振荡抑制"的纯旋钮。

## 3.3 上游针对的症状在我方是否仍存在？（稿 3 §3，逐字）

**症状原文 = 未验证**：本轮 web 检索未取得 `billy948787` 报告的原文（检索结果只有无关仓库/帖子，见 §6）。因此不能把"症状"写成某个具体现象。

**可确定的推理**：
- 该提交**只动 2 行、无测试**（`git show d1e0a16 --stat`：`battle.py` 4 行 + 两个 visualizer），说明它是**经验性抑制**（jitter/overshoot 类观感），不是一条被复现的物理断言。
- 我方已有的**两层机制覆盖情况**：
  - **M1 除零保护**（`battle.py:3298`）：覆盖的是"双 0 速重叠 → ZeroDivisionError 崩溃"，与上游症状**无关**（上游根本没处理，会在双建筑重叠时直接崩）。⇒ **不是对同一症状的覆盖**。
  - **塔矩形推出**（`battle.py:3287-3290/3305-3336`）：上游**没有**这一层（塔走圆）。若上游症状包含"单位卡进塔/被塔挤"，我方是**单帧满额推出**，比上游的 `Δ·(0.5+0.5r)` 圆推更彻底 ⇒ **该子症状在我方大概率不存在或显著减弱**。
  - **`_mk_jump` 排除 + 空地分组**（`battle.py:3279-3282`）：与上游分组同义，不构成差异。
- **若症状是"部队间叠着抖动"**：我方保真度上**更可能残留轻微抖动**，因为产生 `*0.5` 的那个环境（无塔矩形、无 jump 排除）与我方不同构，**不能假设"上游需要 0.5 ⇒ 我们也需要"**。反过来也不能假设"我们更干净所以不需要"。
- **若症状是观感型**（视觉叠在一起）而非物理/语义型：则应**先测**再决定，因为没有训练/胜率侧的因果通道。

**结论**：**未验证**。不能断言症状存在或不存在。需要下述观测才能判定。

### 需要什么观测（稿 3 建议的只读仪器，**未执行**）

判定阈值必须**先写死**（【R3】）；这里只给观测定义：

| 观测量 | 定义 | 说明 |
|---|---|---|
| `n_pairs_overlap` | 每帧 `{(i,j): dist < ri+rj}` 的**对数** | 采样 1/10 帧即可 |
| `resid_max / p95` | 每帧 `max / p95 (ri+rj−dist)`（格） | 与 `collision_radius` 归一化后可比"4% vs 0.4%" |
| `"closing" 判定` | 该对**当前帧**仍在互相接近（`d(dist)/dt < 0`） | 只统计 closing 对，才是"被运动注入的稳态重叠"；其余属衰减残余 |
| `tower_pen` | 落在 `arena.towers` 矩形内的非塔实体数、每实体侵入深度 | 直接检验 C5 的 1 帧窗口 |
| `jitter` | **无目标**（`target_id is None`）或 `\|Δpos\| < 0.25·speed·dt` 的实体，其逐帧 `Δpos` 的中位/极差 | 只对"应该站着不动"的实体测抖动，避免把正常行军算成抖动 |
| `zero_speed_overlap` | 双方 `radius>0` 且 `\|Δpos\|≈0` 的持续重叠对 | 直接量化 C3 |

**通道可用性（稿 3 已核实）**：联赛录像 schema 5 的实体元组带 `[name, x, y, hp, player, kind, max_hp, shield, shield_max, radius, id, target_id, root_cast, is_product, share]`（`rl/replay.py:106`、`:149-164`），且**记录点在 `battle.step()` 之后**（`run_league.py:193-195`）⇒ 录像里的位置就是 `resolve_collisions` 写完之后的**帧末位置**。现成读取范本：`scripts/offline_engagement_trade.py:112-143`（`load_replay` + `entity_table`，含全部字段下标）。
⚠️ **量化限制（必须写进任何报告的限定段）**：`replay.py:150` 把 `x/y` **round 到 0.1 格**，而 `radius` 是全精度（`:156`）⇒ 距离误差可达 **≈0.07 格**，正好与 `*0.5` 的稳态重叠（≈0.02 格）**同量级甚至更大** ⇒ **录像只够判"是否大量重叠/是否夹在塔里"，不足以分辨 `*0.5` 与无 `*0.5` 的差异**。要分辨那一档，必须用**引擎内浮点探针**（下述 B）。

**命令（只读；稿 3 自述「此处只给思路，未执行、未创建任何文件」）**

```
# A. 录像侧（粗判：重叠对/塔内侵入/零速重叠），现成读取范本可复制字段下标
cd /mnt/e/clash-royale-simulator-main
.venv/Scripts/python.exe - <<'PY'   # 需先按 offline_engagement_trade.py:112-143 定义 entity_table
import pickle, itertools, math
d = pickle.load(open("runs/arch
```
**【截断】稿 3 在此处被切断**（§0.2）：命令块余下、以及其后各节未见，**不代写**。

### 【合并者复算 · 交叉引用 §1.4.1】

上游 `d1e0a16` 的完整 diff 与作者/时间已在 **§1.4.1** 逐字给出（`index 730dcd9..e67f57a`；2026-09-14 19:39:53 +0800）；本合并复算确认该提交的**引擎面只有那 2 行**，与稿 3 的 C1 事实（「我方该两行无 `*0.5`」）互为对账面。**两稿对同一对象无冲突。**

---

# §4 非引擎改动（上游 24 个独占提交中的非引擎面）

> 本节主体 = 稿 4（标题《上游 24 个独占提交中的**非引擎**改动 —— 逐项清点与相关性判定》），**逐字保留**。
> **截断说明**：稿 4 在 §4.1 判定表第 7 行后被截断（见 §0.2）；第 8 行起的判定与 §4.2 起各节**未见，不代写**。

## 4.0 取证模式与坐标（稿 4 开头，逐字）

**取证模式**：只读（`git log/diff/show`、`grep`、`read`）。未创建/修改任何文件，未跑训练，未做任何 git 写操作。
**取数时间**：2026-09-18 22:50 UTC（系统 `date -u`）。
**上游**：`/mnt/e/clash-royale-simulator-main-by-jason` @ `f616f19`（2026-09-18 23:24:18 +0800）
**我方**：`/mnt/e/clash-royale-simulator-main` @ `8d2552a`（2026-09-19 06:31:13 +0800）

## 4.1 一句话结论（稿 4 §1，逐字）

**上游这 24 个提交的非引擎改动，没有一项会实质影响我方训练/对局路径。** 因为我方真正在跑的 `src/clasher_new/rl/` 对 `environment` 的引用数为 **0**（62 个 `.py` 逐文件 `grep -c` 全为 0），被改的那些文件全部属于一条**我方已冻结在分叉点的旧训练路径**；唯一值得单独看的两件事——「上游手写对手（`strategies.py`/`defensive_strategy.py`）」与「obs 加阶段/堆叠帧」——我方**已有功能等价或更强的替代物**（`rl/opponents.py::SelfDefenderPolicy`、`rl/observation.py` 的绝对 `time` + `rl/reward.py::_phase_weights`），只是形态不同、**不可直接移植**。

## 4.2 每个文件是否被改、改了多少行（稿 4 §2，逐字）

命令（上游仓库内）：
```bash
cd /mnt/e/clash-royale-simulator-main-by-jason
git rev-list --count f20fa4d..HEAD -- <file>      # 提交数
git diff f20fa4d..HEAD --stat -- <file>           # 行数
```

事实：分叉点 `f20fa4d` 是两仓 `merge-base`，上游独占提交 **24 个**（`git rev-list --count f20fa4d..HEAD` → `24`），与我方 README 声明一致。

| 文件 | 上游提交数 | 改动行数 | 我方自 `f20fa4d` 起改动 | 备注 |
|---|---|---|---|---|
| `environment.py` | 9 | +67/-24（含新增 12 行） | **+5/-1**（我方自行加的 `card_level`） | 旧路径 |
| `evaluate.py` | 5 | +319/-?（净 +319 区块） | 0 | 旧路径 |
| **`strategies.py`** | 3 | **+334（新文件）** | **不存在** | 上游新增 |
| **`defensive_strategy.py`** | 3 | **+161（新文件）** | **不存在** | 上游新增 |
| `agent_pool.py` | 1 | +1/-1 | 0 | 旧路径 |
| `train.py` | 4 | +87 区块（净改） | 0 | 旧路径 |
| `train_autoregressive.py` | 6 | +344 | 0 | 旧路径 |
| `benchmark_speed.py` | 1 | +44/-130（**大幅缩水**） | 0 | 旧路径 |
| `minimal_visualizer.py` | 2 | +50 区块 | **+217/-214**（我方自行重写） | 我方自维护 |
| `new_visualization.py` | 5 | +9 区块（4+/5-） | 0 | 我方 `rl/env_wrapper.py:191` 会 import |
| **`temp_test.py`** | 1 | **+29（新文件）** | **不存在** | 上游新增调试脚本 |
| **`watch_random_models.py`** | 1 | **-17（被删除）** | **存在（16 行）** | 上游删除，我方保留 |
| `special_eval.py` | **0** | 0 | 0 | 双方均未动 |
| `server.py` | **0** | 0 | 0 | 双方均未动 |
| `README.md` | 1 | +1/-1 | **+187/-104**（我方自行重写） | 见 §5-17（**该节未见**） |
| `readme_en.md` | **0** | 0 | 0 | 双方均未动 |

`git diff f20fa4d..HEAD --stat`（我方，同批文件）只有 3 行输出：
```
 README.md                             | 291 +++++---
 src/clasher_new/environment.py        |   6 +-
 src/clasher_new/minimal_visualizer.py | 431 ++++----
```
⇒ **除这 3 个文件外，我方这批非引擎文件逐字等于分叉点。**

## 4.3 核实「`rl/` 不使用 `environment.py`」（稿 4 步骤 3，逐字）

### 4.3.1 `rl/` 对 `environment` 的引用 = 0
```bash
cd /mnt/e/clash-royale-simulator-main
grep -rn "environment" src/clasher_new/rl/          # → 无输出
grep -rc "environment" src/clasher_new/rl/          # → 62 个 .py 全为 0（含 __pycache__ 全为 0）
```
逐文件输出（节选，全部为 0）：`env_wrapper.py:0`、`run_league.py:0`、`opponents.py:0`、`ppo.py:0`、`follower.py:0`、`evaluate.py:0`、`observation.py:0`、`config.py:0`、`train_solo.py:0`、`action_mask.py:0` …（共 62 个 `.py`）。

### 4.3.2 全仓 `environment` 的 import 面（只有 7 个文件）
```bash
grep -rn "environment" --include=*.py src/ scripts/ tests/ | grep -i import
```
```
src/clasher_new/agent_pool.py:1:        from environment import CREnv, random_strategy
src/clasher_new/benchmark_speed.py:24:  from environment import CREnv, legal_random_strategy
src/clasher_new/evaluate.py:4:          from environment import CREnv, random_strategy, player_0_deck, shuffle, Position
src/clasher_new/minimal_visualizer.py:7:from environment import entity_names, card_types
src/clasher_new/train.py:1:            from environment import CREnv, random_strategy, entity_names
src/clasher_new/train_autoregressive.py:1:from environment import CREnv, random_strategy
src/clasher_new/watch_random_models.py:1:  from environment import CREnv, random_strategy
```
⇒ **7 个引用方全部是任务书点名的旧路径文件本体**，`scripts/` 与 `src/clasher_new/rl/` 一个都没引用（`grep -rn "import train\b\|from train import\|benchmark_speed\|agent_pool\|special_eval" --include=*.py scripts/ src/clasher_new/rl/` → 唯一命中是 `scripts/_structure_check.py:556` 把 `special_eval` 当作**字符串常量**列在清单里，不是 import）。
**结论：`rl/` 是独立路径，`environment.py`（含上游全部 9 个提交）我方训练/对局不可达。**

### 4.3.3 我方 `environment.py` 确实停留在旧状态（`1/60`）
```bash
cd /mnt/e/clash-royale-simulator-main
grep -n "fps\|1/60\|1/20\|shape=(8\|phase\|history" src/clasher_new/environment.py
```
```
102:                self.battle.step(1/60)
105:                time.sleep(1/60)
```
- 无 `self.fps`、无 `phase`、无 `history`、`observation_space["grid"].shape == (32,18,15)`（`environment.py:41`，diff 未触及该行）。
- 我方对该文件的唯一改动是 `card_level`（`environment.py:34` 增参数、`:58` 增 `BattleState(..., card_level=...)`），属我方引擎侧（不是我方任务范围）。

对照上游 `fcb1d24`（20Hz）：
```python
-        for i in range(30):
+        for i in range(self.fps//2)          # fps=20 → 10 帧
-                self.battle.step(1/60)
+                self.battle.step(1/self.fps)  # 1/20
```

### 4.3.4 我方 `rl/` 自己的时间口径（dt = 1/60）
```
src/clasher_new/rl/env_wrapper.py:88:  decision_frames: int = 30,
src/clasher_new/rl/env_wrapper.py:89:  dt: float = 1 / 60,
src/clasher_new/rl/env_wrapper.py:485: for _ in range(self.decision_frames):
src/clasher_new/rl/env_wrapper.py:489:     self.battle.step(self.dt)
```
且 `grep -n "dt=" src/clasher_new/rl/run_league.py` → **无命中**（无调用方覆盖）⇒ 我方 = 30 × 1/60 = **0.5 s/决策**；上游 = 10 × 1/20 = **0.5 s/决策**。**每次决策的仿真时长相同，差异只在引擎 tick 粒度（1/60 vs 1/20）——那是引擎侧，已由另一路覆盖。**

## 4.4 逐条行为级摘要 + 相关性判定（稿 4 §4.1，**在第 7 行后被截断**）

| # | 上游改动 | 提交 | 行为级要点 | 我方现有等价物（file:line） | 判定 |
|---|---|---|---|---|---|
| 1 | `environment.py`: obs 加 `phase` + `time_till_next_phase` | `b652f0c`, `00c2b4f` | `observation_space` 增 `Discrete(4)` + `Box(1,)`；`observe()` 按 120/180/240 秒切 4 段，`phase-1` 与 `time_left/120.0` 入 obs | `rl/observation.py:143` `"time": np.array([battle.time])`（绝对秒）；`rl/reward.py:161 _phase_weights(rw, battle_time)` 已按 `PHASE_SWITCH_S` 分段调奖励 | **不需要**（我方已有绝对 `time`，阶段可推出；且奖励侧已阶段感知。无独立 `phase` one-hot / 无归一化 `time_left` 特征 —— 见 §4.3 说明） |
| 2 | `environment.py`: obs 改堆叠 8 帧 `(8,32,18,15)` | `2750ebd` | `history` 队列 8 帧 → `np.stack`；仅 player 0 堆叠 | 我方不用堆叠帧：走 sequence/信念通道（`rl/belief.py`、`rl/follower.py`、`GRU` 路径存在） | **不需要**（架构不同，详见 §4.4 说明） |
| 3 | `environment.py`: 60Hz→20Hz | `fcb1d24` | `self.fps=20`；10 帧 × 1/20 = 0.5 s | `rl/env_wrapper.py:88/89`：30 帧 × 1/60 = 0.5 s | **不需要**（决策时长等价；引擎 tick 差异属引擎侧） |
| 4 | `environment.py`: 删非法出牌惩罚 `-0.05` | `87c5a90` | 删 `if slot != 0 and not succeed: reward -= 0.05`（声称"计划改成 action masking"） | 我方 = 掩码 + 惩罚**并存**：`rl/config.py:58 "invalid_penalty": 0.05`、`rl/env_wrapper.py:262-263` | **不需要**（我方早已上掩码且**刻意保留**惩罚项；上游那一步是它自己的过渡态） |
| 5 | `environment.py`: obs 通道顺序修正 + 坐标 clip + `opponent.reset()` | `2aa7b5e`, `d428b79` | `[entity_id, player_id, elixir, ...]` → `[entity_id, card_type, player_id, elixir, ...]`；`x=clip(0,17)/y=clip(0,31)`；无 `reset()` 的对手不报错 | 我方 obs 通道序为 `[entity_id, is_opponent, elixir, card_type, ...]`（`rl/observation.py:125-130`），坐标系为**本地镜像**（`rl/observation.py:121-123`，player 1 做 `17-x/31-y`） | **不需要**（不同 obs schema；我方坐标镜像契约见 `docs/full_code_reference.md` 引用的 `observation.py:121-123`） |
| 6 | **`strategies.py` 新增 334 行**：5 个具名策略 + `DiverseOpponent` 4 风格 | `23e5b8c`, `e049d55`, `d428b79` | `bridge_pressure_strategy` / `split_lane_strategy` / `counterpush_strategy` / `punish_strategy` / `defensive_strategy`；`DiverseOpponent`（deep_defense/counterpush/opposite_lane/spell_control，每局重抽超参）；`make_opponent_pool()` 供 train/train_autoregressive/evaluate 共用 | **部分有**：`rl/opponents.py:125 SelfDefenderPolicy`（真防守脚本，`simulate_exchange.script_defender` 反制；进 solo 对手池 `"defend"` 槽 `train_solo.py:338/346`，配比 0.2 `train_solo.py:148`）；`scripts/s1_plan_gate.py:241 PlanTokenExecutor`（手写规划转录器） | **部分不需要 / 部分未覆盖**（防守与手写专家侧有等价物；`bridge_pressure`/`split_lane`/`counterpush`/`punish`/4 随机风格**我方无对应**，但对我方路径零影响 —— 见 §4.2） |
| 7 | **`defensive_strategy.py` 新增 161 行** | `964af5d`, `8744065`, `d428b79` | 独立规则对手：手牌/圣水/最近敌群 → 3 群聚放法术 → 迎击最深入侵者 → 保 Giant → 圣水≥8 推弱塔路；自带 13 名 `ENTITY_NAMES` 与 8 卡 `ELIXIR_COST` | **有等价物**：`rl/opponents.py:125 SelfDefenderPolicy`（选 (DPS+HP/15)/费 最优反制、塔前线落点、过河威胁触发；无威胁帧 `passive_prob=0.6` 缓出） | **已有**（**仅此判"已有"**：有直接证据。但"能碾压随机策略"**未验证**，见 §6） |
| 8… | **【截断】稿 4 在此处被切断**（§0.2） | | | | |

**【截断】稿 4 的第 8 行起判定、§4.2 起的各节（含行 7 引用的「见 §6」）全部未见，不代写。**

## 4.5 【冲突并列 §4-C1】"改动行数"口径 —— 稿 4 表 vs【合并者复算】`--numstat`

**说一（稿 4 §2 表，"改动行数"列）**：`environment.py +67/-24`、`evaluate.py +319/-?`、`train.py +87`、`train_autoregressive.py +344`、`benchmark_speed.py +44/-130`、`new_visualization.py +9（4+/5-）`、`README.md +1/-1`、`strategies.py +334`、`defensive_strategy.py +161`、`temp_test.py +29`、`watch_random_models.py -17`。

**说二（【合并者复算】）**：`git -C B diff --numstat f20fa4d..HEAD`（插入/删除）：
```
1	1	README.md
1	1	src/clasher_new/agent_pool.py
0	8	src/clasher_new/arena.py
47	23	src/clasher_new/battle.py
31	117	src/clasher_new/benchmark_speed.py
1	1	src/clasher_new/card_utils.py
161	0	src/clasher_new/defensive_strategy.py
50	17	src/clasher_new/environment.py
277	42	src/clasher_new/evaluate.py
38	12	src/clasher_new/minimal_visualizer.py
4	5	src/clasher_new/new_visualization.py
4	1	src/clasher_new/pathfinding_heap.py
334	0	src/clasher_new/strategies.py
29	0	src/clasher_new/temp_test.py
34	53	src/clasher_new/train.py
324	20	src/clasher_new/train_autoregressive.py
0	17	src/clasher_new/watch_random_models.py
```
逐项并列（**不调和**）：

| 文件 | 稿 4 "改动行数" | 复算 numstat（+/-） | 备注 |
|---|---|---|---|
| `environment.py` | +67/-24 | `50 17` | 复算 `50+17=67` 恰等于稿 4 的 `+67`；但稿 4 的 `/-24` 无 numstat 对应 |
| `evaluate.py` | +319/-? | `277 42` | 复算 `277+42=319` 恰等于稿 4 的 `+319` |
| `train.py` | +87 区块 | `34 53` | 复算 `34+53=87` 恰等于 `+87` |
| `train_autoregressive.py` | +344 | `324 20` | 复算 `324+20=344` 恰等于 `+344` |
| `benchmark_speed.py` | +44/-130 | `31 117` | **`44/130` 与 `31/117` 不相等，且和为 174 vs 148** |
| `new_visualization.py` | +9 区块（4+/5-） | `4 5` | 一致 |
| `README.md` | +1/-1 | `1 1` | 一致 |
| `strategies.py` | +334（新文件） | `334 0` | 一致 |
| `defensive_strategy.py` | +161（新文件） | `161 0` | 一致 |
| `temp_test.py` | +29（新文件） | `29 0` | 一致 |
| `watch_random_models.py` | -17（被删除） | `0 17` | 一致 |
| `agent_pool.py` | +1/-1 | `1 1` | 一致 |
| `arena.py` / `battle.py` / `card_utils.py` / `pathfinding_heap.py` | （稿 4 未在"非引擎"表内列行数） | `0/8`、`47/23`、`1/1`、`4/1` | 属 §1 引擎表 |

> 并列要点（事实，不调和）：多数行的稿 4 数字 = 复算的**插入+删除之和**（`--stat` 口径），但 `environment.py` 的 `/-24` 与 `benchmark_speed.py` 的 `+44/-130` 在两口径下都对不上。**本合并不裁决哪一口径正确**；两条原始读数并列如上。

## 4.6 【冲突并列 §4-C2】共同祖先 / 独立历史线

**与 §1.6 是同一冲突**（稿 1/稿 4 以 `f20fa4d` 为分叉点统计「上游 24 个独占提交」「我方自 `f20fa4d` 起改动」；`docs/upstream_claim_audit_2026-09-19.md:231-232` 主张「无共同祖先 / 独立历史线」）。**证据、原始输出与并列说明见 §1.6，本节不重复、不调和。**

---

## 附：本合并件的可复算命令清单（全部只读）

```bash
A=/mnt/e/clash-royale-simulator-main
B=/mnt/e/clash-royale-simulator-main-by-jason

git -C "$A" log -1 --format='%H|%ad|%s' --date=iso f20fa4d
git -C "$A" merge-base --is-ancestor f20fa4d HEAD ; echo rc=$?      # → 0
git -C "$A" cat-file -t f616f19 ; echo rc=$?                        # → fatal / 128
git -C "$B" merge-base --is-ancestor f20fa4d HEAD ; echo rc=$?      # → 0
git -C "$B" rev-list --count f20fa4d..HEAD                          # → 24
git -C "$B" ls-remote origin refs/heads/main                        # → f616f19… refs/heads/main
git -C "$B" diff --stat f20fa4d..HEAD -- src/clasher_new/battle.py src/clasher_new/arena.py \
        src/clasher_new/pathfinding_heap.py src/clasher_new/card_utils.py   # → 52 insertions(+), 33 deletions(-)
git -C "$B" diff --stat  f20fa4d..HEAD        # → 17 files changed, 1336 insertions(+), 318 deletions(-)
git -C "$B" diff --numstat f20fa4d..HEAD      # → §4.5 表
git -C "$B" show d1e0a16 -- src/clasher_new/battle.py
git -C "$B" show 372ed0e -- src/clasher_new/battle.py src/clasher_new/card_utils.py
git -C "$B" show fcb1d24 -- src/clasher_new/battle.py
git -C "$B" show 1e9716b -- src/clasher_new/battle.py
git -C "$B" show 2daab60 -- src/clasher_new/arena.py src/clasher_new/battle.py src/clasher_new/pathfinding_heap.py
git -C "$B" show 574e946 -- src/clasher_new/battle.py
# 我方行号抽查（§2.5 / §3 的 file:line）
grep -n "RIVER_Y1\|BLOCKED_TILES = \|def is_walkable\|on_left_bridge" "$A/src/clasher_new/arena.py"
grep -n "tile_cost" "$A/src/clasher_new/pathfinding_heap.py"
grep -n "def resolve_collisions\|total_speed == 0\|_push_troop_out_of_tower\|jumping_across_river = True" \
        "$A/src/clasher_new/battle.py"
```
