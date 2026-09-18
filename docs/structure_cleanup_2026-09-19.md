# Tier 0 执行记录（2026-09-19）

> **执行者**：编排者本人（不是子智能体）。依据 [`structure_optimization_plan_2026-09-19.md`](structure_optimization_plan_2026-09-19.md) §6。
> **本 Tier 四项**：**T0-1 死件清理**（§1–§5）· **T0-2 只读结构核对器**（§7）· **T0-3 scripts 全量登记**（§8）· **T0-4 `card_utils` 报错信息**（§9）。
> **用户拍板**：在 T0-1 的「归档 vs 删除」上选择**直接删除**。**§2 的 28 个未跟踪文件因此不可从 git 恢复** ⇒ 删除**前**逐条记录 sha256 + 字节数 + 判断依据，使这次不可逆操作**可审计**。

| 项 | 状态 | 关键验证 |
|---|---|---|
| T0-1 死件清理（**31** 个：3 tracked + 28 untracked） | ✅ | `--list` **100** 不变；子集 **5/5 PASS**；死件复活 **0** |
| T0-2 只读核对器 `scripts/_structure_check.py` | ✅ | `--selftest` **9/9 PASS**（含**反向验证**）；真仓库跑通 |
| T0-3 `docs/agents/scripts_inventory.md`（57 个未登记全列） | ✅ | 脚本生成、非手抄；`env.md` §2.4 加一行指针 |
| T0-4 `card_utils` 报错带「期望 cwd」 | ✅ | A/B **7/7 逐字节一致**（行为中性） |
| 【R18】文档同步 | ✅ | 本文件 + `docs/agents/env.md` + `AGENTS.md` + `docs/README.md` |

---

## 1. T0-1 删除判据（三条，全部实测）

| # | 判据 | 复核命令（可复现） |
|---|---|---|
| ① | **全仓无 import / 无引用**（唯一命中须自证是字符串而非 import） | `grep -rn -E "(^|[^A-Za-z0-9_])<模块名>([^A-Za-z0-9_]|$)" src scripts ideas --include=*.py --include=*.sh --include=*.bat \| grep -v __pycache__` |
| ② | **不在任何运行路径上**：不被训练/评估/自检/dashboard 调用 | 同上 + `scripts/run_selftests.py --list` 计数前后不变 |
| ③ | **命名与 .gitignore 自证为残留** | `git check-ignore -v <路径>`（命中项目自己的忽略规则） |

> ⚠️ **扫描范围说明**：判据 ① 只在 `src/ scripts/ ideas/` 内扫（**不扫 `runs/`** —— 那里是 6.7 G 训练产物，全树 grep 会超时）。这是**口径限定**：因此"0 importer"的结论**只对源码区成立**（这正是我们关心的范围）。

---

## 2. 已删除 · 未跟踪（**不可从 git 恢复**）：28 个

**恢复手段：无**（除非从备份/回收站）。下表 sha256 为**前 16 位**。

### 2.1 仓库根（5 个）

| 路径 | 字节 | sha256(16) | 判断依据 |
|---|---:|---|---|
| `1` | 81 | `829a0813ccc9ab6c` | 文件名字面就是 `1`；`.gitignore` 有对应条目；81 B 无扩展名 ⇒ 误产生的占位产物 |
| `.p6.png` | 463,928 | `0c520ffe32b8ab9b` | 隐藏名 `.p6.png` + `.gitignore` 有对应条目 |
| `0631.pdf` | 622,130 | `b52c301ca34800df` | 纯数字名 PDF，`.gitignore` 有对应条目 |
| `libg_strings.txt` | 788,886 | — | 仓库自己的 `docs/rl_training_fix_plan_v1.md:159` 已把它列为**待清理临时文件** |
| `:memory:.ses` | 51 | — | 同上（sessions 残留；文件名含冒号，Windows 下本就不可正常创建） |

### 2.2 `src/clasher_new/` 顶层（6 个）

| 路径 | 字节 | sha256(16) | 判断依据 |
|---|---:|---|---|
| `tmp_formation_test.py` | 2,321 | `f61d027745219dbb` | 0 引用；`.gitignore:53 = tmp_*.py` 命中 |
| `tmp_path_debug.py` | 1,283 | `48b2e6586772b8cb` | 0 引用；同上 |
| `tmp_spell_forensics.py` | 3,530 | `2e9e53918f2534cb` | 0 引用；同上 |
| `tmp_target_verify.py` | 2,107 | `3a0ecb63bcf81d80` | 0 引用；同上 |
| `tmp_tower_chip_test.py` | 2,409 | `8640c9cee97a35a9` | 0 引用；同上 |
| `gamedata.json.bak_ronin_vines` | 349,773 | `7f651c77bf780fe0` | 0 引用；`.gitignore:54 = *.bak_*` 命中；`gamedata.json` 本体健在 |

### 2.3 `scripts/`（1 个）

| 路径 | 字节 | sha256(16) | 判断依据 |
|---|---:|---|---|
| `stop_solo_2200.log` | 2,080 | `347ba9f64f063716` | 一次性 run 的日志；其配套 `.py` 见 §3.3 |

### 2.4 `src/clasher_new/runs/_tmp_*`（18 个，`runs/` 整体 gitignored）

| 路径 | 字节 | sha256(16) |
|---|---:|---|
| `_tmp_behavior_recount.py` | 9,378 | `e29edfee9522e471` |
| `_tmp_disp_rows.json` | 268,777 | `a989e7378ab21f49` |
| `_tmp_ln_geom.out` | 0 | `e3b0c44298fc1c14` |
| `_tmp_ln_geom.py` | 6,417 | `dca0802c5f1db353` |
| `_tmp_ln_geom2.out` | 4,846 | `3a6ec12956e67ebe` |
| `_tmp_ln_geom2.py` | 5,412 | `85772e64ee7c9dc1` |
| `_tmp_ln_geom3.out` | 9,158 | `f60402d78e1e4bf9` |
| `_tmp_ln_geom3.py` | 6,743 | `ba481eeb92603e0a` |
| `_tmp_metric_id.py` | 2,399 | `fd78b051ffb48fa9` |
| `_tmp_mm_debug.py` | 2,299 | `1d4eca7c71f55cce` |
| `_tmp_probe_maxpre.out` | 771 | `9da674fd7937248a` |
| `_tmp_probe_maxpre.py` | 3,726 | `8b5af2c1aeddbaec` |
| `_tmp_probe_path.out` | 3,081 | `fc28ae78caecb665` |
| `_tmp_scan_disp.out` | 5,860 | `41ecf1668d799dd1` |
| `_tmp_scan_disp.py` | 4,700 | `457ca7f5e61624f4` |
| `_tmp_value_params.py` | 1,079 | `7bad172884440a1a` |
| `_tmp_variants.py` | 4,064 | `aa42668b0e05935c` |
| `_tmp_window.py` | 2,180 | `1d5fe886442e6679` |

> 均为 Sep 14 的 value-path 取证草稿；`grep` 0 引用；`runs/` 被 `.gitignore:22` 整目录忽略。

---

## 3. 已删除 · 已跟踪（**可从 git 恢复**）：3 个

| # | 路径 | 字节 | 判断依据 | **恢复命令** |
|---|---|---:|---|---|
| 3.1 | `re_lib.py`（仓库根） | 0 | **空文件**；0 引用（唯一命中是 `scripts/_survey_merge.py:203` 的**文档字符串**） | `git checkout HEAD~1 -- re_lib.py` |
| 3.2 | `src/clasher_new/pathfinding.py` | 6,260 | 0 importer（唯一的 `pathfinding` 字样命中是 `scripts/probe_engine_profile.py:36-41` 的 `FAMILY` **函数名字典**，其 `FILES = {"pathfinding_heap.py": "pathfinding"}` 指向 **heap 版**）⇒ 现行实现是 `pathfinding_heap.py` | `git checkout HEAD~1 -- src/clasher_new/pathfinding.py` |
| 3.3 | `scripts/stop_solo_training_2200.py` | 4,170 | 一次性「在 4000 边界停某次 solo 训练」脚本；**硬编码绝对路径** `E:\...\runs\economy_towerprem`（L19-21）⇒ 只对那一次 run 有意义；0 外部引用 | `git checkout HEAD~1 -- scripts/stop_solo_training_2200.py` |

> ⚠️ **3.2 的额外核查（方案 §3.2 的 X2 要求）**：`git log --follow -- src/clasher_new/pathfinding.py` 显示其历史含 `e5d1a0e Fix crown towers collision radius issue. Pathfinding now is almost identical to the real game.` 等 —— 但**当前 `battle.py` 只 import `pathfinding_heap`**（`grep` 实测），且 `pathfinding_heap.py` 才是被 profiler 登记的那个。
> ⇒ 判定 `pathfinding.py` 是**被 heap 版取代的旧实现**，不是"保留的对照实现"。**且它是 tracked ⇒ 随时可恢复**，故风险可控。

---

## 4. 明确**保留**（未删，附理由）

| 路径 | 为什么不删 |
|---|---|
| `src/clasher_new/hero_ability_extract.json`（4,925 B） | 方案 §3.2 的 **X9**：全仓 Python **0 处引用**，但**可能是「待接入」而非死件** ⇒ 保留，只在文档标注 |
| `scripts/_survey_*.py`（9 个） | **一次性取证留档**（产出 `docs/` 证据），不是残留 —— 仓内 `_` 前缀的既有约定就是"不进正式仪器表但保留留档" |
| `docs/cmp_*_2026-09-19.md`（8）/ `docs/reward_mechanism_*`（4）/ `docs/audit_clashaiaa_*` / `docs/clashaiaa_reward_ddq_*` / `docs/review_reward_redlines_*`（共 **15 份**） | **并发会话的未跟踪产物**（最近写入 01:02）⇒ **不属于本次工作，一律不碰**（不提交、不删除） |
| `ideas/pz_test.py`（430 B） | 在 `ideas/` 目录内 —— 该目录名表示**有意保留的想法草稿**；方案 X 清单未列它 ⇒ 超范围，不动 |
| 仓库根 `runs/`（2.1 G，含 3 个 `.py`） | 方案 §8 **C6** 已并列记录「两个 `runs/` 产物区**并存，勿合并**」；且是 gitignored 产物区 ⇒ 不动 |
| `src/clasher_new/runs/`（6.7 G） | 同 C6；且其中含**本会话 100 局评估录像**（`rand100_eval/replays/league_31491.pkl`，44.9 MB）与 8701 面板的数据源 ⇒ **绝对不能动** |

---

## 5. 本次删除**不**包含的相邻项（留给后续 Tier）

| 项 | 为什么不在 T0-1 做 |
|---|---|
| `scripts/` 的未登记脚本（**本工具口径 57 个**；盘点稿按 105 文件算写 69 —— 口径见 §8） | 属 **T0-3 文档补齐**（先登记再谈归类），不是删除 |
| `scripts/rl/*.py` 的 11 个 wrapper | `start_rl.bat` 依赖其**位置** ⇒ 属 **T2-6**（分目录时必须同改） |
| 硬编码绝对路径（**本工具口径 9 处 / 7 文件**；盘点稿写 5 处） | 属 **T1-2**（改代码，不是删文件）；清单见 §7 |
| `.p6.png` / `0631.pdf` 之外的根目录杂项（`hook_raw_capture.js`、`tilemap_lane_grid.txt`） | **有实际用途**（逆向抓取脚本 / 地图数据）⇒ 不在残留判据内 |

---

## 6. 验证（**已执行，实测读数**）

| # | 检查 | 命令 | 结果 |
|---|---|---|---|
| 6.1 | **跟踪区只减不增** | `git status --porcelain` | 仅 **3 个 `D`**（§3 那三个）+ 本 Tier 新增文件；**无意外 `M`** |
| 6.2 | **selftest 清单不变** | `run_selftests.py --list \| wc -l` | **100**（与删前**逐字相同**）✅ |
| 6.3 | **引擎仍可导入** | `python -c "import battle, player, card_utils, pathfinding_heap"` | `OK` ✅（`pathfinding.py` 删除后无影响） |
| 6.4 | **相关子集仍绿**（【R19】） | `run_selftests.py test_replay_roundtrip test_random_deck_model test_classified_decks test_league_replays test_precise_threat` | **5/5 全部通过** ✅ |
| 6.5 | **规模前后对比**（核对器 ①） | `python scripts/_structure_check.py` | 引擎顶层 **34 → 28** 文件、**9,997 → 9,581** 行（−6 = 5×`tmp_*.py` + `pathfinding.py`） |
| 6.6 | **死件复活数**（核对器 ④） | 同上 | **0** ✅ —— 且核对器自检对**故意造假件**的合成树断言 `revived == 1` ⇒ **判别力已证** |

> **⚠️ 口径限定**：6.2/6.4 只覆盖**被选子集**，按【R19】**不等于全量绿**。本 Tier **未跑全量**（默认不跑）。

---

## 7. T0-2 只读结构核对器 `scripts/_structure_check.py`（**新增**）

**动机**：盘点里 §1/§3/§4/§5 的事实全散在子智能体报告与手工命令里 ⇒ 下一轮改动**无法快速对账**。
本工具把 7 项事实变成**一条可复跑命令**，且**不 import 任何产品代码**（纯 `ast` 文本解析 ⇒ 不会因 `card_utils` 的 cwd 契约在异地失败）；单项失败记 `error` 不抛，可当长跑前门禁。

| 检查 | 内容 | 实测（Tier 0 后） |
|---|---|---|
| ① `areas` | 分区文件数 / LOC | 引擎顶层 **28 / 9,581**、`rl` **41 / 23,869**、`scripts` **102 / 26,858**、合计 **171 / 60,308** |
| ② `utf8` | `_force_utf8_stdout` 各实现函数体**去空白后 sha256** 是否一致 | **3 处、3 种互异实现 ⇒ 不一致**（`rl/dashboard.py:3033`、`rl/run_league.py:1538`、`scripts/probe_value_ln.py:116`）⇒ **T1-1 的收敛目标已量化** |
| ③ `layering` | **引擎 → `rl/` 的反向 import 边**（应为 0） | **0** ✅ |
| ④ `deadfiles` | T0-1 已删的 31 个死件是否**复活** | **0** ✅ |
| ⑤ `abspaths` | 硬编码绝对路径（盘符字面量，已剔注释与 docstring） | **9 处 / 7 文件**（见下） |
| ⑥ `lazyimp` | `rl/` 内**函数体内 import**（**两套口径并列**） | 含 `selftest.py`：**566/917 = 61.7%**；排除：**95/439 = 21.6%** |
| ⑦ `selftest` | `test_*` 定义数 vs `main()` 调用数 | **100 / 100** ✅（`loc=5909`） |

**自检**：`--selftest` **9/9 PASS**（合成目录、零仓库依赖），含一条**反向验证**（把合成树里的非法反向 import 修掉后 ③ 必须变 0）⇒ 证明检查**不是恒真**。
**留证**：`docs/structure_after_2026-09-19/baseline.json`（本次真仓库 JSON）。

**⑤ 的 9 处（T1-2 清单，比盘点稿的 5 处多 —— 盘点只扫 `sys.path.insert` + 只认 `E:/` 正斜杠，本工具扫全部字面量）**：

| 文件:行 | 路径 |
|---|---|
| `scripts/_schema5_probe.py:10` | `E:\clash-royale-simulator-main\scripts` |
| `scripts/s2_trade_probe.py:12` | `E:\clash-royale-simulator-main\scripts` |
| `scripts/assassin_left_bridge_test.py:7` | `E:/clash-royale-simulator-main/src/clasher_new` |
| `scripts/assassin_vs_megaknight.py:7` | 同上 |
| `scripts/assassin_vs_sparky.py:9` | 同上 |
| `scripts/duel_search.py:14` | 同上 |
| `scripts/question_bank_poc.py:9` | 同上 |
| `scripts/question_bank_poc.py:217` | `E:/clash-royale-simulator-main/runs/archive/` |
| `scripts/question_bank_poc.py:219` | `E:/clash-royale-simulator-main/src/clasher_new/` |

> **⚠️ 两套口径并列（【R17】）**：**盘点稿「5 处」** vs **本工具「9 处 / 7 文件」** —— 两套都真，差在扫描面。**T1-2 按 9 处（超集）做**，引用时带口径。

**⑥ 的三套口径（并列不调和）**：

| 来源 | 含 `selftest.py` | 排除 `selftest.py` |
|---|---|---|
| **本工具**（按 **import 语句**计数） | **566/917 = 61.7%** | **95/439 = 21.6%** |
| 盘点稿（子任务，自定口径） | 429/572 = **75%** | 「仍有 58 条」 |

> **交叉验证**：本工具对 `selftest.py` **单文件**给出 **7 top / 471 lazy**，与盘点稿「模块级 import 仅 **7** 行、函数内 **471** 行」**逐字一致** ⇒ 计数方法无歧义，差异出在**其余文件的纳入范围** ⇒ **不裁决，三套并列**。

---

## 8. T0-3 `scripts/` 全量登记（**新增** `docs/agents/scripts_inventory.md`）

**问题**：`docs/agents/env.md` §2.4（判读/诊断工具表）只引用了 **34** 个 `.py` 名，而 `scripts/` 顶层实有 **91** 个 `.py` ⇒ **57 个未登记**。

**做法**：把 57 行塞进 `env.md` 会让那张"一眼可读"的表失去意义 ⇒ 按本仓既有做法（**索引 + 分册**）新建全量表；`env.md` §2.4 **追加一段指针**（不动既有 34 行）。

**生成方式**：脚本抽取每个文件的模块 docstring 首行 + 是否含 `--selftest` + `assert` 计数 + 是否已被 `env.md` 引用 ⇒ **不手抄**（【R4】同类纪律）。
分组：**内部工具 `_` 前缀 21 个 / 测试 6 个 / 测试·带自检 3 个 / 一次性取证·仪器 61 个**。

> **⚠️ 口径**：**91** = 顶层 `scripts/*.py`；盘点稿的「**69 个未登记**」按 **105 个文件**（含 `.sh/.ps1/.js` 与 `scripts/rl/*.py`(11)）算 ⇒ **两套并列**。

---

## 9. T0-4 `card_utils` 数据缺失时报「期望 cwd」（**行为中性，已 A/B 对账**）

**改了什么**：`src/clasher_new/card_utils.py` 新增 `_resolve_data(name)`；5 个模块顶层 `open()` 的**路径参数**由字面量改为 `_resolve_data(<同一字面量>)`。

**为什么这么写（关键）**：盘点 §4.4 指出这是**全仓最贵的隐性契约** —— `card_utils` 是 **20 个 importer**（其中 12 个是 rl 模块）的最广入口，模块顶层用**裸相对路径** `open('gamedata.json')` ⇒ **import 它就隐含要求 `cwd = src/clasher_new`**。
方案 **T0-4 明确要求「报错信息，不改成自动找路径」** ⇒ 本步**故意不**改成 `os.path.join(os.path.dirname(__file__), name)`（那会改掉 cwd 契约，属 **T3-2**，必须与数据文件归拢一起做并对账）。`_resolve_data` 成功时**返回原样的裸文件名** ⇒ 成功路径逐字不变。

**A/B 对账（我自己跑的，不是引用）**：把 `git show HEAD:...card_utils.py` 放到同目录临时副本，两边各自 `exec_module`，对 7 个派生结构做 **`pickle` 逐字节指纹**（用 `pickle` 而非 `json` —— `card_data` 有**循环引用**，`json.dumps` 会抛 `ValueError: Circular reference detected`）：

| 结构 | HEAD | 工作树 | 结论 |
|---|---|---|---|
| `data` | `82a560cc7f93ac28`（127,043 B） | 同 | **SAME** |
| `card_data` | `3bdeda0e191ecbb3`（137,403 B） | 同 | **SAME** |
| `characters` | `54016f80721a9d47`（188,361 B） | 同 | **SAME** |
| `spells` | `34a68443c98c2f04`（52,432 B） | 同 | **SAME** |
| `buildings` | `69a6b3e8e9743eb9`（128,472 B） | 同 | **SAME** |
| `projectiles` | `7cf23481feb91c7e`（26,386 B） | 同 | **SAME** |
| `air_units` | `c7cf6e54dee2b14e`（292 B） | 同 | **SAME** |

⇒ **7/7 逐字节一致**。

> **⚠️ 我自己的一次失误 + 更正（留档）**：临时副本 `_ab_head_cardutils.py` **第一遍并没被删掉** ——
> 我那条命令前面 `cd src/clasher_new` 已改过 cwd，末尾的 `rm -f src/clasher_new/_ab_head_cardutils.py`
> 实际指向 `src/clasher_new/src/clasher_new/...`，而 `-f` 把「路径不存在」**静默吞掉**。
> 于是我在本文初稿里写了「已确认不在树里」这句**错话**。
> **抓到它的是新写的 `scripts/_structure_check.py` ①**：`engine_top` 从 **28 文件 / 9,581 行**
> 变成 **29 / 10,126**（+1 文件 / +545 行）⇒ 立刻定位到该文件并已真删。
> 复核后 ① 回到 **28 / 9,604**（= 9,581 基线 + 本次 `_resolve_data` 新增的 23 行）。
> **教训**：`rm -f` + 相对路径 + 已 `cd` 过的会话 = **静默无操作**，必须用 `ls` 正向确认「不存在」。

**新报错路径实测**（故意在错误 cwd 下跑）：

```
数据文件 'gamedata.json' 不存在。当前 cwd = 'E:\clash-royale-simulator-main'；
本仓约定 cwd 必须为 'E:\clash-royale-simulator-main\src\clasher_new'（src/clasher_new）。
请在 src/clasher_new 下运行，或经 scripts/rl/*.py 包装脚本（它们会 os.chdir）。
```

> ⚠️ **本步没有解决 cwd 契约本身** —— 只是让失败**可读**。契约的根治在 **T3-2**（须与数据文件归拢同步做）。

---

## 10. 已知工程坑（本次踩到，留档）

| # | 坑 | 现象 | 修法 |
|---|---|---|---|
| 10.1 | **f-string 嵌套引号**（Python 3.13 也不允许） | `f"…{v[\"lineno\"]}…"` ⇒ `SyntaxError: unexpected character after line continuation character` | 把 `join` 提成变量，改用 `.format()` |
| 10.2 | **Windows python 把 `/tmp/x` 解析成 `E:\tmp\x`** | `FileNotFoundError: 'E:\\tmp/card_utils_HEAD.py'` | 临时文件放**仓库内**路径，用完即删 |
| 10.3 | **`json.dumps` 撞循环引用** | `ValueError: Circular reference detected`（`card_data` 自引用） | 指纹改用 `pickle.dumps`（对循环免疫，且两版解析代码相同 ⇒ 字节应一致） |
| 10.4 | **全树 `grep` 超时** | 仓库含 6.7 G `runs/` ⇒ 60 s 被杀 | 扫描显式限定 `src scripts ideas` + `--include=*.py` |
| 10.5 | **`rm -f` + 相对路径 + 已 `cd` 过的 shell = 静默无操作** | T0-4 的 A/B 临时副本没被删，我却在文档里写了「已确认不在树里」（**错话**） | **正向确认**：删后必须 `ls <path>` 看到 `No such file`；**抓到它的是 `_structure_check.py` ① 的文件数变化**（28→29）⇒ 这正是"把事实变成可复跑检查"的价值 |
