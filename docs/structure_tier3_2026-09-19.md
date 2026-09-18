# Tier 3 执行记录（2026-09-19）

> 依据 [`structure_optimization_plan_2026-09-19.md`](structure_optimization_plan_2026-09-19.md) §6 的 Tier 3（4 项，方案标注为「高风险 / 触红线」）。
> **本轮完成 T3-2 的「Stage A」**（**消除 cwd 契约**）；Stage B（搬文件）与 T3-1 / T3-3 / T3-4 见 §5（**未做，如实标注**）。
> 前序：Tier 0 [`structure_cleanup_2026-09-19.md`](structure_cleanup_2026-09-19.md)、Tier 1 [`structure_tier1_2026-09-19.md`](structure_tier1_2026-09-19.md)、Tier 2 [`structure_tier2_2026-09-19.md`](structure_tier2_2026-09-19.md)。

| 项 | 状态 | 关键验证 |
|---|---|---|
| **T3-2 Stage A**：`card_utils` 用 `__file__` 定位数据文件 ⇒ **cwd 契约取消** | ✅ | A/B **7/7 逐字节 SAME**；**从仓库根**跑通 `import battle` / `duel_search --help` / `assassin_left_bridge_test`（改前**全部失败**） |
| T3-2 Stage B：把数据文件搬进 `assets/`、16 个 `.pt` 搬进 `runs/` | ❌ **不做**（理由见 §3） | 引用面实测：`.pt` 被 **34 个文件**引用；`gamedata.json` 被 **~20 份文档**引用 |
| T3-1 拆 `battle.py` / T3-3 迁 `tests/` / T3-4 统一 `sys.path` | ❌ 未做 | 见 §5 |

---

## 1. 为什么先做 T3-2（而不是按方案的 Tier 2 剩余项顺序）

方案把 T3-2 排在最后、并标注「**极高风险**」，因为它写的是「数据文件归拢 + 静态 `.pt` 归档」。
但本轮先做它的**前半**，理由是**它是全项目最贵的一个结构缺陷**，且**前半是低风险的**：

- 盘点（`docs/structure_optimization_plan_2026-09-19.md` §4.4）实测：`card_utils` 是**全仓被依赖最多的模块**
  （**20 个 importer**，其中 12 个是 rl 模块），而它在模块顶层用**裸相对路径**读 5 个数据文件
  ⇒ **import 任何 rl/引擎模块都隐含要求 `cwd = src/clasher_new`**。整个 Tier 0/1/2 期间它反复咬人：
  - Tier 1：`duel_search.py --help` 在仓库根**必然失败**（且 traceback 在 GBK 下是**乱码**）；
  - Tier 2：`build_payload` / `build_solo_payload` 的 A/B **拿不到样本**（`PermissionError` 之外还叠着路径问题）；
  - 仓内用 `scripts/rl/*.py` 包装脚本的 `os.chdir(_SRC)` 把这个约束**藏了起来**。

**关键的情报**：`card_aliases.py:432` **一直是正确写法** ——
`open(os.path.join(_HERE, "gamedata.json"))`。也就是说 `card_utils.py` 不是"历史遗留的必然"，
而是全仓的**异类**；`pathfinding_heap.py` 也用 `__file__`。⇒ 修它不是发明新约定，而是**向已有约定看齐**。

---

## 2. 改了什么（3 处，`__file__` 优先）

| 文件 | 改动 |
|---|---|
| `src/clasher_new/card_utils.py` | `_resolve_data(name)`：候选顺序 = ① `__file__` 同目录 → ② 原样的裸相对路径（兼容旧 cwd 用法）；都找不到才报错，且报错里**同时列出两个试过的路径** |
| `src/clasher_new/minimal_visualizer.py` | `open('cards.json')` → `open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cards.json'))`（补 `import os`） |
| （`card_aliases.py` / `pathfinding_heap.py`） | **本来就对，不动** |

**T0-4 的报错信息保留并升级**：原来说「cwd 必须是 `src/clasher_new`」，现在说「已试过这两个路径」——
因为**约束本身已经不存在了**。

### 2.1 行为等价性怎么证的（不是"我觉得一样"）

1. **派生数据逐字节 A/B**：把 `git show HEAD:src/clasher_new/card_utils.py` 取成同目录临时模块，
   两边各自 `exec_module`，对 7 个派生结构做 `pickle` 逐字节指纹：

| 结构 | HEAD | 新 | 结论 |
|---|---|---|---|
| `data` | `82a560cc7f93ac28` | 同 | **SAME** |
| `card_data` | `3bdeda0e191ecbb3` | 同 | **SAME** |
| `characters` | `54016f80721a9d47` | 同 | **SAME** |
| `spells` | `34a68443c98c2f04` | 同 | **SAME** |
| `buildings` | `69a6b3e8e9743eb9` | 同 | **SAME** |
| `projectiles` | `7cf23481feb91c7e` | 同 | **SAME** |
| `air_units` | `c7cf6e54dee2b14e` | 同 | **SAME** |

⇒ **7/7 SAME**（临时模块用完即删，`ls` 正向确认不存在）。

2. **「旧行为没有退化」+「新能力真的到手」两条都测**：

| 场景 | 改前 | 改后 |
|---|---|---|
| **文档口径**（`cwd = src/clasher_new`）下 import 引擎 | ✅ | ✅（**同一条路径**：`__file__` 同目录与 cwd 指向同一个文件） |
| **从仓库根** `import battle` / `player` / `card_utils`，读 `Card('Xbow').elixir` | ❌ `FileNotFoundError: gamedata.json` | ✅ `= 6`（cwd = `E:\clash-royale-simulator-main`） |
| **从仓库根** `scripts/duel_search.py --help` | ❌ 失败（traceback 还是 GBK 乱码） | ✅ **rc=0**，正常 usage |
| **从仓库根** `scripts/assassin_left_bridge_test.py`（真跑引擎） | ❌ 失败 | ✅ 跑完并打出 `第5格… 完美解 ✓` |

> ⚠️ **口径限定**：这三条新能力**只覆盖引擎顶层的数据文件**。`client_side/*.py`（逆向辅助工具的**另一份**
> `card_utils` 副本）仍用裸相对路径 —— 它是独立子目录的脚本集合，**本轮有意不动**（见 §5）。

---

## 3. 为什么**不做** Stage B（搬文件）—— 实测数字，不是偏好

| 若搬 | 代价（实测） |
|---|---|
| 5 个 `*.json`（`gamedata.json` / `cards_stats_*.json`）→ `assets/` | `gamedata.json` / `cards_stats_*` 被 **~20 份文档**引用（`docs/README.md`、`docs/game_engine.md`、`docs/full_code_reference.md`、`docs/data_integration_notes.md` …）；还牵连 `client_side/download_images.py`、`minimal_visualizer.py` |
| 16 个散落 `.pt`（37 MB）→ `runs/` | 被 **34 个文件**引用（脚本 / 文档 / `.bat`） |

**而收益是什么？** 契约修好后，"文件在 `src/clasher_new/` 还是 `assets/`"对**功能零影响**，只剩目录美观。
⇒ **花 34 + 20 处引用去换美观，且每一处都可能是"改漏了就静默失败"** —— 不对称，**不做**。
方案把 T3-2 标为「极高风险」的判断是对的；**风险全在 Stage B**，而 Stage B 恰恰是不必要的那一半。

---

## 4. 回归

| 检查 | 结果 |
|---|---|
| `py_compile` `card_utils.py` / `minimal_visualizer.py` | OK |
| selftest 子集（卡牌 / 血量 / 卡组 / 录像 / 精确塔伤 / 非战斗实体 / 亡语路由 / 启发式对手 ×10） | **10/10 PASS** |
| `scripts/_structure_check.py` 硬门禁 ③④⑦ | 全过（引擎→rl 反向边 **0**、死件复活 **0**、selftest 登记 **100/100**） |

---

## 5. 未做（**如实标注**）

| # | 项 | 为什么没做 |
|---|---|---|
| 5.1 | **T3-1** 拆 `battle.py` | 触【R13】（掩码/合法格依赖 `BattleState` 的部署语义，位图对账只能证 **128 张快照**、不能证全局）；且**并发会话的工作集包含 `battle.py`** ⇒ 方案自己标注「并发期间禁止」 |
| 5.2 | **T3-2 Stage B** 数据文件归拢 | 见 §3：**34 + 20 处引用**去换目录美观，不对称 ⇒ 不做 |
| 5.3 | **T3-3** 迁 `rl/selftest.py` → `tests/` + pytest | 触【R19】与 2 个**硬编码调用方**（`scripts/run_selftests.py:36-42`、`scripts/_apply_s2_channel_when_idle.sh:70-71`）；「全量路径逐位不变」的保证会丢 |
| 5.4 | **T3-4** 统一 `sys.path` 引导（116 文件） | 面大、且 `sys.path` 顺序与 `os.chdir` 耦合；**T3-2 Stage A 已消除了其中最主要的一部分必要性**（数据文件不再依赖 cwd）⇒ 优先级下降 |
| 5.5 | `client_side/*.py` 的裸相对路径 | 逆向辅助工具的独立副本；不在引擎/rl 依赖链上 |

---

## 6. T3-4「统一 `sys.path` 引导」—— 做的是**顺序保证**，不是**参数统一**

### 6.1 先量：135 条引导语句的**骨架**（实测，不是估）

| 骨架（把变量名抹成 X） | 条数 |
|---|---:|
| `X.X.X(0, X)` | 94 |
| `X.X.X(0, X.X.X(X.X.X(X.X.X(X.X.X(X))),`（跨行 join） | 33 |
| `X.X.X(0, X.X.X(X, "X", "X"))` | 7 |
| `X.X.X(0, X.X.X(X.X.X(X), '..', 'X', 'X'))` | 6 |
| 其它（`X.X()` / 单行 join / 带尾注释） | ~8 |

**★ 结论：参数表达式有 ~10 种** —— `_SRC`（= `src/clasher_new`）、`_HERE`（= `scripts/`）、`os.getcwd()`、
`TREE`（来自 `argv`）、`os.path.join(_ROOT, "src", "clasher_new")` …… 也就是说
**这不是「同一个东西抄了 135 遍」，而是 135 个文件各算各需要的路径** ⇒ **参数层面无可统一**
（把参数统一掉 = 改行为）。

另外：**guarded 72 / unguarded 63**。把 unguarded 改成 guarded **会改顺序语义**
（`insert(0, X)` 无条件把 X 顶到最前；带 guard 则保留 X 原有位置）—— 这正是方案 §7 警告的
「`sys.path` 顺序影响 import 解析」⇒ **不动**。

### 6.2 真正可统一、且有价值的是**顺序保证** ⇒ 新增门禁检查 ⑪

**两次真实事故**（都靠日志/仪器发现，不是靠复读）已经证明这个 bug 类是真的：

| 事故 | 症状 |
|---|---|
| `scripts/_schema5_probe.py`（T1-1b） | 引导被插在 `os.chdir(TREE)` 与 `sys.path.insert` **之间** ⇒ `ModuleNotFoundError: rl.io_bootstrap` |
| `rl/selftest.py`（T2-8 拆分后） | 聚合文件的 `from rl.selftest_common import ...` 排在引导**之前** ⇒ **全量套件根本起不来** |

**⑪ 的判据（刻意保守，sound）**：只查**入口文件**（含 `if __name__ == "__main__"`），
要求「第一个产品 import 之前必须出现过 `__file__`」。

- **为什么只查入口文件**：**包内成员**（如 `rl/follower.py`）被 import 时 `rl` 已在 `sys.path` 上 ⇒
  它**不需要**自带引导，查它就是**假阳性**；
- 而入口文件被 `python <file>` 直接跑时 `sys.path[0]` 是**脚本所在目录**：
  `scripts/x.py` → `scripts/`、`rl/x.py` → `rl/` ⇒ 这两类**必须**先插路径；
- **引擎顶层不算**（`src/clasher_new/arena.py` 的 `sys.path[0]` 就是 `src/clasher_new`）。

**结果**：候选集内 **102 个入口文件 / 99 个自带 `__file__` 依据**；**⑪ = 0**（进硬门禁）。
**判别力已断言**：合成树里造一个「先 `import battle`、后 `sys.path.insert(0, __file__)`」的入口文件 ⇒ **必须被抓**；
改成正确顺序 ⇒ 必须回 OK。

**⑨ 与 ⑪ 的分工**：⑨ 只看 `rl.io_bootstrap` 这一个 import（**只报告**）；⑪ 看**全部产品 import**（**门禁**）。

### 6.3 ★ 我自己在写 ⑪ 时的两个坑（留档）

| # | 坑 | 现象 | 修法 |
|---|---|---|---|
| 1 | **判据假阳性**：第一版把**引擎顶层**也纳入候选 | 报 **10 个「违规」**（`arena.py`/`environment.py`/`pathfinding_heap.py` …），**实测全是假阳性** —— `python src/clasher_new/arena.py` 时 `sys.path[0]` 就是 `src/clasher_new` | 候选集收紧为 `scripts/` + `rl/`；注释写清「为什么引擎顶层不算」 |
| 2 | **行级替换打错位置 + 把注释插进续行中间** | 文件里有**两处**同样的 `cands = ...`（⑨ 与 ⑪）；我的 `next()` 只取第一处 ⇒ **改了 ⑨、留下 ⑪**；且把 4 行注释插在**反斜杠续行**的中间 ⇒ `SyntaxError`（连续两次） | ① 语句改成**不用续行**的形式（`+ [` 后换行）；② 注释一律放在**语句之后**；③ 改用 **read + edit 工具**精确替换，不再做索引算术 |

> **教训**：**绝不在反斜杠续行语句内部插东西**；同一文件里出现多次同形语句时，索引算术必然出错 ⇒ 用工具级精确替换。

### 6.4 交付物

| 物 | 内容 |
|---|---|
| 核对器 **⑪** | 「入口文件的 path 引导先于产品 import」（**门禁**，含判别力断言） |
| 核对器 **⑨**（收紧） | 覆盖面从「`scripts/` + `rl/` + 引擎顶层」收紧到「`scripts/` + `rl/`」（去掉假阳性来源） |
| 留档 | 本节（含 135 条骨架分布、guard 语义风险、两次真实事故、我的两个坑） |
