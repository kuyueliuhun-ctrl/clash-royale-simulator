# 上游引擎增量核查：原作者最新版 vs 我们（4 个引擎文件 / 6 个提交 / 逐条状态）

> **取数时间**：2026-09-19 06:35–07:20 CST，取数机 = 本地 WSL，**全程只读**
> （`git log/rev-list/show/diff/merge-base/cat-file/ls-remote`、`read`、`grep`、只读 `python3 -B`；未改任何文件、未跑训练、无 git 写操作）。
> **A（我方）** = `/mnt/e/clash-royale-simulator-main`，核查时 HEAD `8d2552a`（分支 `replay-format-terminal-state`），remote `kuyueliuhun-ctrl/clash-royale-simulator`。
> **B（上游原作者最新版）** = `/mnt/e/clash-royale-simulator-main-by-jason`，HEAD `f616f19`，remote `https://github.com/Jason-XII/clash-royale-simulator.git`；
> `git -C B ls-remote origin refs/heads/main` = `f616f190…a477` = 本地 HEAD ⇒ **该目录就是上游 main 的最新状态**（截至取数时刻）。
> **原始稿**（本轮编排的四路子任务逐字稿 + 合并件）→ [`upstream_four_sections_merge_2026-09-19.md`](upstream_four_sections_merge_2026-09-19.md)（**其中四份源稿有 4 处截断，已在该文件 §0.2 逐条标注**）。

---

## §0 结论先行（一句话回答）

**是，原作者在共同祖先之后确实更新了引擎**：24 个独占提交里 **6 个**动过引擎文件，涉及 **4 个文件**（`battle.py` / `arena.py` / `pathfinding_heap.py` / `card_utils.py`），净 **+52 / −33 行**。
**引擎数据与时间基座零改动**（`cards*.json`、`gamedata.json`、`tilemap_lane_grid.txt`、`timing.py`、`core.py`、`player.py` 一个都没被碰）。

逐条状态（详 §3）：

| # | 上游引擎改动 | 类型 | 我方状态 | 需不需要动作 |
|---|---|---|---|---|
| E1 | `d1e0a16` 碰撞推挤给 `e1` 乘 **0.5**（1 处 2 行） | 行为修复（阻尼） | **未跟进**（我们同处无 `*0.5`），但我们有两层上游没有的替代机制 | 建议**先测再定**（§4.1） |
| E2 | `372ed0e` 建筑掉血 **两处**：`lifeTime/1000` + 去掉重复 `take_damage` | 行为修复 | **我们已在分叉点后独立修好，同因同果** | 无需动作（§3.2） |
| E3 | `fcb1d24` **60 Hz → 20 Hz**（BREAKING，含目标扫描 `%2`、寻路重算 `%10→%3`） | **性能**（换 3× 吞吐）+ 行为 | **未跟进**（我们仍是 `30×1/60 = 0.5 s/决策`，决策时长与上游等价） | 建议**不跟**（§4.2） |
| E4 | `1e9716b` `battle.py` **仅加注释**（+7/−3） | 注释 | 无所谓（我们两处注释语义已自行覆盖） | 无需动作（§3.4） |
| E5 | `2daab60` 删 `BLOCKED_TILES` 河面条目 + 删 `is_walkable` 桥面分支 + `W` 代价 `800→50` + 注释掉 jump 触发 | 结构/行为（**被作者自己的下一提交推翻一半**） | **未跟进**（我们 `arena.py` 这两段与分叉点**逐字相同**，`W` 仍是 800/7） | 建议**不照搬**（§4.3） |
| E6 | `574e946` 过河跳跃「半成品」：注释掉 jump 复位 + 新增 waypoint 出河复位 | **WIP（上游最终版跳跃是死代码）** | **未跟进**（我们的跳跃是**活代码**） | 建议**不照搬**（§4.4） |

**上游 24 个提交里另外 18 个只动非引擎**（RL 训练/评估/手写策略/可视化/文档/benchmark）。这些**对我们零影响**：我方主训练与对局路径 `src/clasher_new/rl/` 对 `environment.py` 的引用数 = **0**（62 个 `.py` 全为 0 命中），而被上游改动的 `environment.py`/`train*.py`/`evaluate.py`/`agent_pool.py`/`benchmark_speed.py` 全部属于**我方已冻结在分叉点的旧训练路径**（§6）。

⚠️ **顺带查出两处「我们自己的」引擎潜在问题**（不是上游引入的；上游一个同源、一个在它那边不可达）——见 §5，都带可复现计数：

- **P1** `battle.py:3292` `if abs(direction_vector) == 0: return` 用的是 `return` 而不是 `continue` ⇒ **同心重叠会把本 tick 剩余的碰撞对全部跳过**（上游同一行**逐字相同**，作者未修）。
- **P2** 桥面最外侧各半格在 `tilemap_lane_grid.txt` 里是 `W`，而 `arena.is_walkable` 判它们**可走** ⇒ A\* 对它们计价 **800**（×10/14） ⇒ **名义 3 格宽的桥，对行军实际只有 2 格宽**（16 个格，左右桥各 8）。

---

## §1 两仓关系与分叉点（含一个**退出码陷阱**）

| 项 | 值 | 命令 / 依据 | 原始结果 |
|---|---|---|---|
| 共同祖先（分叉点） | `f20fa4d535b355e1d5727becbd6568bae21ff5b9`（2026-08-27 23:07:59 +0800，`Add entropy coefficient to encourage more diverse actions`） | `git -C A log -1 f20fa4d` | 该提交在两仓都存在 |
| 分叉点是我方 HEAD 的祖先？ | **是** | `git -C A merge-base --is-ancestor f20fa4d 8d2552a; echo $?` | `rc=0` |
| 分叉点是上游 HEAD 的祖先？ | **是** | `git -C B merge-base --is-ancestor f20fa4d f616f19; echo $?` | `rc=0` |
| 分叉点确实是最新的共同点 | `git -C A merge-base f20fa4d 8d2552a` → `f20fa4d…` | — | 同 hash |
| 上游独占提交数 | **24** | `git -C B rev-list --count f20fa4d..HEAD` | `24` |
| 我方分叉后提交数 | **242** | `git -C A rev-list --count f20fa4d..8d2552a` | `242` |
| 上游历史是否线性 | 线性、**0 个 merge** | `git -C B log --merges --oneline f20fa4d..HEAD \| wc -l` | `0` |
| 我方 object store 是否含上游独占提交 | **不含**（抽样 `2aa7b5e`） | `git -C A cat-file -t 2aa7b5e` | `fatal: Not a valid object name` |
| 上游本地是否 = GitHub 最新 | 是 | `git -C B ls-remote origin refs/heads/main` | `f616f19…` |

### 1.1 ⚠️ 退出码陷阱（会直接被读成「无共同祖先」）

```
$ git -C A merge-base --is-ancestor f616f19 8d2552a ; echo rc=$?
fatal: Not a valid object name f616f19
rc=128
```

`f616f19` 是**上游独占提交**，本机 A 的 object store 里**根本没有这个对象** ⇒ git 直接**报错**而不是回答「不是祖先」。
「不是祖先」的正确答案是 **`rc=1` 且无输出**；**`rc=128` 是「查询失败」**，不能当否证使用。
⇒ 判「两仓是否是同一历史线」必须用**共同祖先方向**问（`merge-base --is-ancestor <已知的共同提交> HEAD`，上表 rc=0），
或直接 `git -C A cat-file -t f616f19` 看对象在不在，**不能**用「问一个本机没有的 hash」的返回值下结论。

### 1.2 与另一份会话文档的口径冲突（**并列不调和**，只登记事实）

`docs/upstream_claim_audit_2026-09-19.md:231-232`（**另一会话的未跟踪文档，本文件不改动它**）写：
>「本机 clone 的 HEAD 是 `8d2552a`，与上游 f616f19 **无共同祖先**（`git merge-base --is-ancestor` 返回 NO，`f616f190…` 在本地都不是合法 commit 名）⇒ 本机项目是**独立历史线**」

两份文档的**原始观测本身不冲突**（它自己就记下了「在本地不是合法 commit 名」），冲突只在**解读**：
- 本文件 §1 的读数：共同祖先 = `f20fa4d`（**两仓两侧都 rc=0**）；我方与上游差 24 个上游独占提交。
- 该文档的读数：本机 **没有** `f616f19` 这个对象 ⇒ 「本地 HEAD 不能当上游 HEAD 的证据」（**这一句本文件完全同意**）。
- 需要并列的分歧点只有一处：**「无共同祖先 / 独立历史线」 vs 「有共同祖先 f20fa4d，只是上游独占提交不在本机」**。
  判别依据已在上表逐条给出（`rc=128` vs `rc=1` 的区别，§1.1）。**另一会话文档未经其作者确认前不代为改写**；
  本文件只登记两说与各自的原始命令，**不裁决**。

---

## §2 上游 24 个独占提交全表（按时间正序）

生成命令：`git -C B log --reverse --format='%h | %ad | %s' --date=short f20fa4d..HEAD`

| # | hash | 日期 | 标题 | 是否动引擎 |
|---|---|---|---|---|
| 1 | `2aa7b5e` | 09-07 | BUGFIX: fix wrong transform of observation. | |
| 2 | `87c5a90` | 09-07 | ENHANCE: remove the illegal move penalty, planning to add action masking later | |
| 3 | `b652f0c` | 09-07 | ENHANCE: Add time and battle phase to environment observation | |
| 4 | `00c2b4f` | 09-08 | ENHANCE: phase and battle clock are now considered by the model | |
| 5 | `a316cae` | 09-09 | BUGFIX: finally made sense of tensor shapes… | |
| 6 | `23e5b8c` | 09-14 | Big update: add several strategies that crushes the random strategy | |
| 7 | `964af5d` | 09-14 | Prepare training script; fix ent_coef issue…（新增 `defensive_strategy.py`） | |
| **8** | **`d1e0a16`** | 09-14 | **BUGFIX: fix troop collision issue mentioned by @billy948787** | ✅ `battle.py` |
| 9 | `8903460` | 09-15 | Improve evaluation script | |
| 10 | `2750ebd` | 09-15 | Architecture change: now the observation uses stacked frames | |
| 11 | `31c8232` | 09-16 | Architecture change: Use autoregressive actions… | |
| **12** | **`372ed0e`** | 09-16 | **BUGFIX: fix buildings hp do not decrease, thanks to @longlingking** | ✅ `battle.py`+`card_utils.py` |
| 13 | `e7dcad5` | 09-16 | Update benchmark script | |
| **14** | **`fcb1d24`** | 09-16 | **BREAKING: change simulator to be 20Hz instead of 60Hz** | ✅ `battle.py`（+`environment.py`） |
| 15 | `a5e4ab1` | 09-16 | Update readme | |
| **16** | **`1e9716b`** | 09-17 | **Add some comments in `battle.py`**; Update evaluation… | ✅ `battle.py`（仅注释） |
| 17 | `8744065` | 09-17 | Enhance: upgrade defensive strategy to be significantly stronger | |
| 18 | `e049d55` | 09-17 | Enhance: strengthen all existing strategies | |
| **19** | **`2daab60`** | 09-17 | **BUGFIX: previous code settings are not compatible with the new A\* pathfinding…removed the river tiles from blocked_tiles** | ✅ `arena.py`+`battle.py`+`pathfinding_heap.py` |
| **20** | **`574e946`** | 09-18 | **bugfix: partly implement river jumping logic** | ✅ `battle.py` |
| 21 | `d428b79` | 09-18 | Increase opponent randomness to make them harder to exploit | |
| 22 | `1a86041` | 09-18 | Update evaluation to add entropy diagnosis; Increase ent_coef | |
| 23 | `7fc64f2` | 09-18 | Update entropy calculation method | |
| 24 | `f616f19` | 09-18 | Use new placement head hoping to generalize better | |

**文件枚举完整性**：24 个提交逐个 `git show --name-status` 汇总 = `M`×47 / `A`×3（`defensive_strategy.py`、`strategies.py`、`temp_test.py`）/ `D`×1（`watch_random_models.py`，在 `2aa7b5e` 删除）；**无 R/C（改名/复制）** ⇒ 不存在「用改名把引擎文件藏起来」的情形。

**引擎侧净改动量**（`git -C B diff --numstat f20fa4d..HEAD -- <4 文件>`）：

```
arena.py              0     8
battle.py            47    23
card_utils.py         1     1
pathfinding_heap.py   4     1
```

---

## §3 引擎 6 条改动逐条 × 我方状态

> 判定规则（本轮自定，写在前面以免事后追认）：**「已有」= 我方在同一条逻辑上有等价实现**，须给 file:line；
> **「未跟进」= 我方停留在该改动之前的状态或第三种实现**；**「不适用」= 该改动对我方路径不可达**。

### 3.1 E1 `d1e0a16` 碰撞推挤 `*0.5` —— **未跟进（行为差异）**

上游只改了 2 行（`battle.py` HEAD 位于 `:756-757`）：

```diff
-                    e1.position.x += -direction_vector.real * (1-movement_ratio)*overlap
-                    e1.position.y += -direction_vector.imag * (1-movement_ratio)*overlap
+                    e1.position.x += -direction_vector.real * (1-movement_ratio)*overlap*0.5
+                    e1.position.y += -direction_vector.imag * (1-movement_ratio)*overlap*0.5
```

我方同处（`src/clasher_new/battle.py:3274-3303`）**无 `*0.5`**，且我方在该函数上已有**两层上游没有的**机制：

| 机制 | 我方 | 上游 |
|---|---|---|
| 除零保护 | `battle.py:3298` `if total_speed == 0: continue`（注释记为 M1） | **无**（直接除 `e1.speed+e2.speed`，双 0 速重叠会 `ZeroDivisionError`） |
| 塔的占位方式 | 矩形推出 `_push_troop_out_of_tower`（`battle.py:3287-3290 / 3305-3336`） | 塔走圆形 overlap 分支（`grep _push_troop` 上游 = 0 命中） |
| 跳动/空中单位 | `_mk_jump is None` 排除 + `is_air_unit` 分组（`:3279-3282`） | 仅 `is_air_unit` 分组 |

**语义**：记 `r = movement_ratio`、`Δ = overlap`，`e1` 侧系数 `f`（我方 1 / 上游 0.5）⇒
单帧净分离 `= Δ·(f + (1−f)·r)`，**欠分离** `= Δ·(1−f)·(1−r)`。上游 `f=0.5` 时：
`f_eff = net/Δ = 0.5 + 0.5r ∈ [0.5, 1]`，逐帧按 `0.5^n·(1−r)·Δ` 几何衰减 ⇒
**这**不是「穿模」，是「用一点稳态欠分离换一点振荡抑制」的**阻尼旋钮**；且它的效果**取决于 `combinations` 的遍历顺序**
（当 0 速建筑恰落在 `e2` 位，`r=1` ⇒ `*0.5` 被 `(1−r)=0` 吃掉，仍是满额推开）。

**建议**：**先测再定**，理由是上游那个提交**无测试、无复现**，且我们不是同构实现 —— 见 §4.1。

### 3.2 E2 `372ed0e` 建筑掉血（两半） —— **✅ 我们已在分叉点后独立修好（同因同果）**

上游的三件事：① `Building.__init__` 插入 `battle_state` 形参；② 衰减块去掉 `self.data.lifetime > 0 and` 前置 + **删掉重复的第二次** `self.take_damage(decay)`；③ `card_utils.Card.lifetime` 由**毫秒改秒**（`/1000`）。

我方对应（**两条都在，且我方注释明确记着「此前是错的」**）：

| 上游修的那一半 | 我方 | 证据 |
|---|---|---|
| `lifeTime` 毫秒→秒 | **已有** | `src/clasher_new/card_utils.py:304-306`：`# M2 修复：lifeTime 单位为毫秒，此前未除 1000（Cannon 30s 寿命被当成 30000s → 永不衰减）` / `_lt = self.data['summonCharacterData'].get('lifeTime')` / `self.lifetime = _lt / 1000 if _lt else float('inf')` |
| 同一帧 `take_damage` 两次（衰减 ×2） | **已有** | `src/clasher_new/battle.py:1357-1360`：`if self.data.lifetime > 0 and not self.persistent:` / `# M2 修复：此前同一帧 take_damage 两次（衰减速率 ×2）` / `decay = (self.data.hp / float(self.data.lifetime)) * dt` / `self.take_damage(decay)` |
| `Building.__init__` 加 `battle_state` 形参 | 形式不同 | 我方 `battle.py:1298` 签名是 `def __init__(self, id, position, player, card_name, persistent=False, evolved=False)`；`battle_state` 由 `Entity.__init__` 统一接收（`battle.py:53-56`）⇒ **同一需求已由结构解决**，不是缺口 |

⇒ **结论：这一条不需要任何动作**。两份独立修复（上游与我们在分叉点后各自发现）指向同一个根因，**互为交叉验证**。

### 3.3 E3 `fcb1d24` 60 Hz → 20 Hz（**BREAKING**）—— **未跟进，建议不跟**

上游改动：
```python
# environment.py
+        self.fps = 20
-        for i in range(30):
+        for i in range(self.fps//2)      # = 10
-                self.battle.step(1/60)
+                self.battle.step(1/self.fps)   # = 1/20
# battle.py：目标复用 + 节流
-        current_target = self.update_current_target()
+        target = self.battle_state.entities.get(self.target_id)
+        if target is None or not target.is_alive or not target.targetable:
+            current_target = self.update_current_target()      # 立刻重扫
+        elif self.battle_state.tick % 2 == 0:
+            current_target = self.update_current_target()
+        else:
+            current_target = target                            # 复用（注释：每 0.1s 最多一次）
...
-                elif self.in_sight_range(current_target) and self.battle_state.tick % 10 == 0:
+                elif self.in_sight_range(current_target) and self.battle_state.tick % 3 == 0:
```

我方现状：

| 项 | 我方 | 上游 |
|---|---|---|
| 决策步长 | `rl/env_wrapper.py:88-89`：`decision_frames=30`、`dt=1/60` ⇒ **30×1/60 = 0.5 s/决策** | `10 × 1/20 = 0.5 s/决策`（**等价**） |
| 引擎 tick 粒度 | **1/60 s** | 1/20 s |
| 旧路径 `environment.py` | `environment.py:102/105` 仍是 `step(1/60)` / `sleep(1/60)`（我方对分叉点只改了 `card_level` 一个参数，`git diff f20fa4d..8d2552a -- environment.py` = +4/−2） | 20 Hz |
| 目标扫描节流 | **每帧都扫**（`battle.py:1191` `current_target = self.update_current_target()`，无缓存分支） | `tick%2` 复用 |
| 寻路重算节流 | `battle.py:1245` `tick % 10 == 0`（@60 Hz ⇒ **0.167 s**） | `tick%3`（@20 Hz ⇒ **0.15 s**）⇒ **墙钟口径几乎相同**，只是各自对着自己的 tick 基 |

**判断**：上游这一条是**用精度换 3× 吞吐**的工程权衡（它自己的标题写了 `BREAKING`）。
我方把同样的目标/寻路节流换算成 60 Hz 的等价 tick 数即可（`%2`→`%6`、`%3`→`%10`），而**我们当前是「每帧扫目标 + `%10` 重算路径」**：
目标扫描比我方等价口径**多 6 倍** —— 这是**唯一值得单独看的性能项**，但它属于我们自己的性能议题，**不是「上游修了我们没修」**。
⇒ 建议：**不跟 20 Hz**（会动全部卡牌时序/录像/训练语义），若关心吞吐，另开性能项按「等价墙钟节流」单独立项、按【R2】默认参数不动、按【R7】对账。

### 3.4 E4 `1e9716b` `battle.py` 仅注释 —— **不适用**

该提交对 `battle.py` 只有 +7/−3：加了三段注释（`Projectile` 的 entity_holder 说明、`on_spawn` 调用时机、`battle_state.on_death()` 会触发王塔激活）、删了两处空行。
**零功能改动**。（同提交的 `evaluate.py`/`train_autoregressive.py` 属非引擎，见 §6。）

### 3.5 E5 `2daab60` 河面阻挡 / `W` 代价 / jump 注释 —— **未跟进（且上游是半成品）**

上游这一提交做四件事：
1. `arena.py BLOCKED_TILES` **删掉** 8 行河面条目（`(0,15),(0,16),(1,15),(1,16)`、`(5..12,15/16)`、`(16,15),(16,16),(17,15),(17,16)`）；
2. `arena.py is_walkable` **删掉**桥面分支（`elif RIVER_Y1 <= y <= RIVER_Y2: on_left_bridge = 2.0<=x<5.0; on_right_bridge = 13.0<=x<16.0`）；
3. `pathfinding_heap.py`：`W` 代价 `800 if not is_air_unit else 7` → **`7 if (is_air_unit or jump_speed) else 50`**；
4. `battle.py`：把 **jump 触发块整段注释掉**，并删掉 `if index == len(self.path): move_towards(target)` 分支。

我方现状（**逐字核对，不是印象**）：

| 项 | 我方 | 判定 |
|---|---|---|
| `arena.py BLOCKED_TILES` | `arena.py:21-34`，**与分叉点逐字相同**（程序化抽取比对：`identical: True`，14 行河面条目全在） | 未跟进 |
| `arena.py is_walkable` 桥面分支 | `arena.py:106-119`，**与分叉点逐字相同**（`identical: True`，`2.0<=x<5.0` / `13.0<=x<16.0` 都在） | 未跟进 |
| `pathfinding_heap.py` `W` 代价 | `pathfinding_heap.py:132-134`，**与分叉点逐字相同**（`800 / 7`）；桥面 `'.'` 我方是 **5**（上游 8，我们按「桥面与半场地面同价」改过） | 未跟进 |
| jump 触发块 | `battle.py:1233-1239` **活的**（`if not self.jumping_across_river and has_jump_ability: … self.jumping_across_river = True`） | 未跟进（我方是活代码） |

**为什么建议不照搬（三条理由，都可复核）**：

1. **上游自己把它改成了半成品**：`2daab60` 注释掉 jump 触发，`574e946`（次日）在 waypoint 里新增了「出河复位」逻辑，但**触发端没恢复** ⇒
   `grep -n "jumping_across_river" B/src/clasher_new/battle.py` 只有 `init False`、被注释的复位、被注释的触发、waypoint 里的**只读判断**与 `in_river` 豁免 —— **上游 HEAD 再无任何 `jumping_across_river = True`**。也就是说那条新代码在它自己的仓库里**不可达**。
2. **它修的是一个「与它自己新 A\* 不自洽」的问题，而那个不自洽在我方不存在**：我方 `pathfind_ground_walkable`（`battle.py:3243-3246`）= `arena.is_walkable` ∧ 静态障碍距离场；`is_walkable` 的桥面区间与 `BLOCKED_TILES` 的河面条目**自洽**（河面 32 格：`x∈{0,1}∪[5,13)∪{16,17}` 全被 `BLOCKED_TILES` 封住、桥面 `x∈[2,5)∪[13,16)` 全开）⇒ 地面单位**只能走桥**，与真机规则一致。
3. **它做的是「让地面单位改从河面涉水」**（`W` 代价 50 而非 800），这是**机制口径的改动**，不是修 bug；我们若跟进，会**同时改变过河规则与桥的意义**，而我们的过河/桥面已经有独立的实证修正（`battle.py:1248-1263` 的「卡死自救」，注释写着「实证：一对弓箭手楔死在桥头，60s 纹丝不动」）。

⇒ 建议：**不照搬**。但**上游这一条暴露出来的真正问题在我们这边也存在另一形式**（桥面最外半格 `W` 计价 800），见 §5-P2，**那条值得单独立项修**。

### 3.6 E6 `574e946` 过河跳跃「半成品」 —— **未跟进，建议不照搬**

上游把 jump **复位块**（`if self.jumping_across_river and self.on_both_sides_of_river(self.start_jumping_position): …`）注释掉，改为在 waypoint 移动里做：

```python
waypoint_in_river = 17 >= waypoint.y >= 15
distance = waypoint.distance_to(self.position)
if self.jumping_across_river:
    if not waypoint_in_river:
        if distance < self.speed * dt:
            self.jumping_across_river = False
            self.data.is_air_unit = False
            self.speed = self.data.speed
elif waypoint_in_river:
    pass
```

我方：`battle.py:1187-1190` 的复位块是**活的**（`self.data.is_air_unit = Card(self.name).is_air_unit`，用**重建 Card 取真值**，比上游 `= False` 更严）；
触发在 `battle.py:1233-1239`，并有上游没有的配套：`_mk_jump` 在空中/预备阶段**直接 return 跳过常规寻路**（`battle.py:1224-1230`，注释记录了「2026-09-10 修正：`_mk_jump` 挂在 entity_holder 上而非实体，旧写法永远取不到 → 双重位移」）。

⇒ 结论：**我们的跳跃是活代码且有配套修正；上游的是死代码**。不照搬（若将来想重构跳跃，可借鉴它的 `waypoint_in_river` 判据形式，但**不是**当作 bugfix 移植）。

---

## §4 该不该移植（逐条结论 + 判据 + 失败分支）

> 纪律：【R3】判据跑前写死（含失败分支）；【R5】n=1 分辨率不足；【R11】未获用户拍板前不改奖励/不扩参。

| 条目 | 建议 | 若做，判据（**先写死**） | 失败分支（做到就停，不叠加解释） |
|---|---|---|---|
| **E1 碰撞 `*0.5`** | **先测再定** | 只读仪器统计「每帧仍重叠的对数 / `max、p95` 欠分离量」在**同一批录像**上的分布；若 `*0.5` 前就已经 `p95 ≈ 0`（无可见欠分离），则**裁决「上游症状在我方不存在」** | 若读数落在录像量化误差内（录像 `x/y` 四舍五入到 0.1 格 ⇒ 距离误差可达 ≈0.07 格，而 `*0.5` 的稳态重叠 ≈0.02 格）⇒ **判「现有仪器分辨不出」，不改代码** |
| **E2 建筑掉血** | **不做** | 已有等价实现（§3.2） | — |
| **E3 20 Hz** | **不跟**（若要吞吐，另立性能项） | 任何做法都必须【R2】：默认参数逐位不变，新行为只经显式开关 | 若某项读数在 20 Hz 下变化超出同配置重跑散布 ⇒ 判「不可归因」。⚠️ 我方已有同配置重跑单点差 **0.60** 的标定（`docs/agents/redlines.md` R16 附近），n=1 下不得归因 |
| **E4 注释** | **不做** | — | — |
| **E5 河面 A\*** | **不照搬**；但把 §5-P2（桥面 sliver 计价 800）**单独立项** | P2 若修：判据 = 修前后**同一批录像**里「实际走桥的路径点集合」宽度分布，以及桥头「近似零位移」事件的计数（`battle.py:1253-1263` 已埋点） | 若宽度分布不变 ⇒ 判「A\* 本来就没走进 sliver，P2 是纯一致性问题」，按低优先级记档 |
| **E6 river jump** | **不照搬**（上游为死代码） | — | — |

**移植成本口径（供拍板用）**：引擎侧任何改动都要过【R13】（改判定逻辑必须位图对账）与【R18】（改代码同步改文档）；
E1/E3 都会改变逐帧位置 ⇒ **现有录像不能作为判据基线**（旧录像与新引擎不可逐位对拍），必须走「改动前后各自重跑同一批 seed」的路子。

---

## §5 我们自己的两处引擎潜在问题（本轮自主查出，均带可复现计数）

> 这两条**不是**上游「有更新我们没跟」，而是核查过程中顺带查出的**我方**问题；登记在此是为了不丢掉证据。

### P1 `resolve_collisions` 里 `return` 应为 `continue`（**同心重叠会中断本 tick 剩余所有碰撞对**）

`src/clasher_new/battle.py:3292`：

```python
direction_vector = complex(e2.position.x-e1.position.x, e2.position.y-e1.position.y)
if abs(direction_vector) == 0: return          # ← 在二重循环内 return
```

`return` 位于 `for troop in (ground_troops, flying_troops): for e1, e2 in troop:` 的**内层循环体**里 ⇒
一旦出现**圆心完全重合**的一对（`distance == 0`），**本 tick 之后的所有碰撞对都不再解算**。

- **上游同源性**：上游 `by-jason/src/clasher_new/battle.py:748` 是**逐字相同**的一行 ⇒ **不是我们引入的**，且作者**未修**（`d1e0a16` 只动了 `*0.5` 那两行）。
- **可达性**：需 `e2.position - e1.position == (0,0)` **精确**为零（浮点）。两个单位同点部署、或推挤后恰好落到同点时可出现；**未实测发生率**。
- **影响**：发生的那一 tick 里，其余所有重叠对**不推开**（下一 tick 会补上，因为重叠仍在）⇒ 更像「偶发一帧漏推」而非不可恢复；但若某对**长期**保持同心，它会**持续**吃掉本 tick 的所有后续解算 ⇒ 需要实测才能定量。
- **建议**：改成 `continue` 是**一行**、语义可逐位论证（该对无方向、本来就跳过），且属于【R8】「每个修复配回归测试」的典型对象。**未改，等拍板**（改判定逻辑要走【R13】位图对账）。

### P2 桥面最外侧各半格在代价图里是 `W`（800）而在合法性判定里可走 ⇒ **桥对行军实际只有 2 格宽**

**事实（程序化计数，可复跑）**：

- `tilemap_lane_grid.txt` 是 64×36 半格图，`'W'` **只出现在 4 行**（`ny = 30..33` ↔ 引擎 `y ∈ [15.0, 17.0]`），全图 `'W'` 共 **112** 个；
- 这 4 行内 `arena.is_walkable` 判为**可走**的格共 **32** 个（= 左右桥各 16 个）；
- 两者**相交 = 16 个格**（左桥 `x=2.25`、`4.75`，右桥 `x=13.25`、`15.75`，各 4 行）。
  ⇒ 这 16 格**合法可走（`is_walkable` True）但 A\* 计价 800**（`pathfinding_heap.py:133-134`，`800×geo`）。

复算脚本（只读，输出已核对）：

```
cd /mnt/e/clash-royale-simulator-main/src/clasher_new
/usr/bin/python3 - <<'PY'
rows=[list(l) for l in open('tilemap_lane_grid.txt',encoding='utf-8').read().splitlines()]
BLOCKED=[(0,15),(0,16),(1,15),(1,16)]+[(i,j) for i in range(5,13) for j in range(15,17)]+[(16,15),(16,16),(17,15),(17,16)]
def walk(x,y):
    if not (0<=x<18 and 0<=y<32): return False
    if (int(x),int(y)) in BLOCKED: return False
    if 15.0<=y<=16.0: return (2.0<=x<5.0) or (13.0<=x<16.0)
    return True
mm=[]
for ny in range(30,34):
    for nx in range(36):
        if rows[63-ny][nx]=='W' and walk((nx+0.5)/2,(ny+0.5)/2): mm.append((round((nx+0.5)/2,2),round((ny+0.5)/2,2)))
print(len(mm), sorted(set(x for x,_ in mm)))
PY
# ⇒ 16 [2.25, 4.75, 13.25, 15.75]
```

**后果**：桥的名义宽度是 **3 格**（`x∈[2,5)` / `[13,16)`，`arena.py:113-116` + `BLOCKED_TILES` 只封河面），
但 A\* 进入最外半格要付 `800×10=8000`（对比桥面 `'.'`/`'1'`/`'2'` 的 `5×10=50`，相差 **160×**）
⇒ **行军路径实际被压到中间 2 格宽**。这与我们自己在 `pathfinding_heap.py:136-139` 追求的「桥面与半场地面同价、三车道过桥、车道保持」**口径不符**：
代价表只给了 2 条便宜车道，第三条（最外 λ=0.25 格）实质是禁止的。

**上游对照**：上游把 `W` 地面价从 800 降到 50（`2daab60`），在他们「河面可走」的设定下，这三个格子直接变成「比桥面贵 6 倍但能走」——
**同一处不一致，上游用降代价绕过，根因（tilemap 的 `W` 边界与 arena 的桥面边界差半格）两边都没修**。

**建议**：单独立项（【R3】预注册 + 【R13】位图对账），两个候选做法（**未拍板**）：
① 把 `tilemap_lane_grid.txt` 的桥面 `'1'/'2'` 扩到与 `is_walkable` 的 `[2,5)`/`[13,16)` 对齐；
② 或让 `pathfinding_heap` 的 `W` 判定改用 `arena.is_walkable`（等价于「合法即可走，河面本就到不了」）。
**不选哪个先不定** —— 需要先有一份「单位是否真的走不进这 16 格」的只读实测（录像里按格统计路径点）。

---

## §6 非引擎部分：**对我们零影响**

- **隔离证据**：`grep -rn "environment" src/clasher_new/rl/ --include=*.py | wc -l` = **0**（62 个 `.py` 逐文件为 0）。
- **引用 `environment.py` 的只有 7 个文件**，全部是旧路径本体：`agent_pool.py:1`、`benchmark_speed.py:24`、`evaluate.py:4`、`minimal_visualizer.py:7`、`train.py:1`、`train_autoregressive.py:1`、`watch_random_models.py:1`。
- 我方 `environment.py` 相对分叉点只改了 `card_level` 一个参数（`git diff --numstat f20fa4d..8d2552a -- src/clasher_new/environment.py` = `4 2`）⇒ 上游对它的 9 个提交**一个都没进我们树**，**且不会进**。
- 上游新增的两份「能碾压随机策略的手写对手」（`strategies.py` +334 行、`defensive_strategy.py` +161 行）属于**旧训练路径的对手池**；我方 `rl/` 有自己的对手体系（`rl/opponents.py`）。
  **注**：本轮**未**逐条比对两者的强度/行为差异（我只核到「路径不可达」这一层），**不要把上游那两份策略当成我方对手池的替代或补充**（未验证）。

---

## §7 未验证 / 未做（**不得升级为事实**）

1. **上游 `d1e0a16` 的症状原文**：`@billy948787` 报告的 issue 原文**未取得**（web 检索无命中）⇒ §3.1 的「症状是否在我方存在」**只能推理，不能断言**；E1 的判据因此写成「先测」。
2. **P1 / P2 的发生率未实测**：两条都只有**静态可达性分析 + 计数**，**没有**跑录像统计发生率（本轮纪律 = 只读核查上游增量，不做新的引擎实验）。
3. **我方视角的 river/jump 深层缺陷**：本轮由子任务做过一轮（含用系统 `python3` 经 stdin 跑真实引擎的受控实验），**其结论稿在传回时被截断**，未验证部分已在该稿中标注；本轮**不代为补写**（见 [`upstream_four_sections_merge_2026-09-19.md`](upstream_four_sections_merge_2026-09-19.md) §0.2 稿 2 行）。
4. **`*0.5` 的稳态重叠解析式**（≈0.02 格 vs 原式 ≈0.002 格）是**解析近似**（r 恒定、忽略多体耦合与遍历顺序），**未实测**。
5. **两仓关系之争未裁决**：§1.2 只并列两说与原始命令，**不判定另一会话文档对错**（该文件属另一会话，本文件不改动它）。
6. **未评估「移植 E1/E3/E5 是否影响训练读数」**：那需要改动引擎 + 重跑，超出本轮「只读核查」的范围。

---

## §8 复算命令清单（逐条可复跑）

```bash
B=/mnt/e/clash-royale-simulator-main-by-jason
A=/mnt/e/clash-royale-simulator-main

# 1) 分叉点与两侧计数
git -C A log -1 f20fa4d
git -C A merge-base --is-ancestor f20fa4d 8d2552a ; echo "A rc=$?"
git -C B merge-base --is-ancestor f20fa4d HEAD    ; echo "B rc=$?"
git -C B rev-list --count f20fa4d..HEAD           # 24
git -C A rev-list --count f20fa4d..8d2552a        # 242

# 2) 退出码陷阱
git -C A merge-base --is-ancestor f616f19 8d2552a ; echo "rc=$?"   # fatal + rc=128（不是 rc=1）
git -C A cat-file -t f616f19                                        # fatal: Not a valid object name

# 3) 上游独占提交 + 引擎文件命中
git -C B log --reverse --format='%h | %ad | %s' --date=short f20fa4d..HEAD
git -C B log --oneline f20fa4d..HEAD -- src/clasher_new/battle.py src/clasher_new/arena.py \
    src/clasher_new/pathfinding_heap.py src/clasher_new/card_utils.py
git -C B diff --numstat f20fa4d..HEAD -- src/clasher_new/battle.py src/clasher_new/arena.py \
    src/clasher_new/pathfinding_heap.py src/clasher_new/card_utils.py

# 4) 引擎数据/基座零改动
git -C B log --oneline f20fa4d..HEAD -- src/clasher_new/cards.json src/clasher_new/gamedata.json \
    src/clasher_new/tilemap_lane_grid.txt src/clasher_new/timing.py src/clasher_new/core.py src/clasher_new/player.py
# （空输出 = 零改动）

# 5) 逐条引擎 diff
git -C B show d1e0a16 -- src/clasher_new/battle.py
git -C B show 372ed0e -- src/clasher_new/battle.py src/clasher_new/card_utils.py
git -C B show fcb1d24 -- src/clasher_new/battle.py
git -C B show 2daab60 -- src/clasher_new/arena.py src/clasher_new/battle.py src/clasher_new/pathfinding_heap.py
git -C B show 574e946 -- src/clasher_new/battle.py
git -C B show 1e9716b -- src/clasher_new/battle.py

# 6) 我方状态（逐条）
grep -n "movement_ratio" $A/src/clasher_new/battle.py            # 3299-3303：无 *0.5
grep -n "abs(direction_vector) == 0" $A/src/clasher_new/battle.py # 3292：return
grep -n "lifeTime" $A/src/clasher_new/card_utils.py               # 304-306：/1000
grep -n "decay" $A/src/clasher_new/battle.py                      # 1357-1360：只 take_damage 一次
grep -n "jumping_across_river" $A/src/clasher_new/battle.py       # 1187/1233/1237：活代码
grep -n "jumping_across_river" $B/src/clasher_new/battle.py       # 上游：无置 True 处
grep -n "tile_char == 'W'" -A1 $A/src/clasher_new/pathfinding_heap.py   # 800 / 7
grep -rn "environment" $A/src/clasher_new/rl/ --include=*.py | wc -l    # 0
```

---

## §9 一句话给决策者

- **上游引擎增量 = 4 个文件 / 6 个提交**，其中 **2 条我们自己已经修好、1 条与我们无关、1 条是注释**；
- 真正「上游改了、我们没改」的只有 **3 条**：碰撞 `*0.5`（建议先测）、20 Hz（建议不跟）、河面 A\*（建议不照搬，上游自己是半成品）；
- 本轮更值得动手的其实是**我们自己的两处**：`return`→`continue`（P1）与桥面 sliver 计价 800（P2）——**都待你拍板，本轮一行代码都没改**。
