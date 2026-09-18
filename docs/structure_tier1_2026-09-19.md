# Tier 1 执行记录（2026-09-19）

> 依据 [`structure_optimization_plan_2026-09-19.md`](structure_optimization_plan_2026-09-19.md) §6（Tier 1 = 低风险等价重构）。
> **执行者**：编排者本人。**逐项验证**；**偏离方案处显式说明理由**；**不确定处写不确定**。
> Tier 0（死件清理 / 只读核对器 / scripts 登记 / `card_utils` 报错）见 [`structure_cleanup_2026-09-19.md`](structure_cleanup_2026-09-19.md)。

| 项 | 状态 | 关键验证 |
|---|---|---|
| **T1-1** UTF-8 兜底收敛到单一实现 | ✅（含 **T1-1b 部分收敛**，见 §1.4） | 核对器 ②：**定义 1 处 / 别名 3 处**；新增回归 **12/12 PASS**；`main()` 与仪器 5/5 |
| **T1-2** 修 9 处硬编码绝对路径 | ✅ | 核对器 ⑤：**9 → 0**；7/7 编译；`duel_search --help` rc=0 |
| **T1-3** selftest 加发现 + x/y 计数 | ✅ | `main()` 100 行调用序列**与 HEAD 逐字相同**；`--order-check` 出三种顺序差集 |
| **T1-4** `runs/_tmp_drawtest` → `tempfile` | ✅ | 跑后 `runs/_tmp_drawtest` **不存在** |
| **T1-5** 显式化 2 个假绿 SKIP | ✅ | 打桩空 ckpt 强跑 ⇒ **2 条 `[SKIP-NO-CKPT]` 都登记并汇总** |
| **Tier 1 合并回归** | ✅ | selftest 子集 **10/10**；`check_dashboard_js.py` **FAILURES=0**；[`scripts/_structure_check.py`](../scripts/_structure_check.py) `--selftest` **13/13** |

---

## 1. T1-1 UTF-8 兜底收敛

### 1.1 先纠一处**盘点稿的不准确**（我实测出来的）

方案 §6 T1-1 写「3 处函数体**已逐字相同**」。**实测不是**（`scripts/_structure_check.py` ②）：

| 位置 | 函数体 | 判定 |
|---|---|---|
| `rl/run_league.py:1538` | `for _stream in (sys.stdout, sys.stderr)` + `errors="replace"` | **强版** |
| `rl/dashboard.py:3033` | **与上一行逐字相同**（仅 docstring 不同） | **强版** |
| `scripts/probe_value_ln.py:116` | 只 `sys.stdout`、**无 `errors="replace"`**、**不管 stderr** | **弱版** |

> ⚠️ **我自己的仪器 bug 留档**：核对器 ② 最初把 **docstring 也算进指纹**，于是给出「**3 种互异**」这个
> **误导性**结论（看起来像三份都不同）。已改为**两条指纹分开**：`body_digest`（去 docstring，语义）
> 与 `full_digest`（含 docstring），判据取 **body**；并加了回归断言
> 「a/b 函数体判为同一实现、含 docstring 才分得开」。**这正是"仪器先于结论"的价值**。

### 1.2 做了什么

1. **新增单一实现** [`src/clasher_new/rl/io_bootstrap.py`](../src/clasher_new/rl/io_bootstrap.py)：
   `force_utf8_stdout(streams=None) -> int` —— 两路 + `errors="replace"`、**幂等**、**吞异常**（兜底本身不能崩）、
   **零依赖**（只 `sys`/`typing` ⇒ 可在任何脚本的最早期调用）。docstring 里写清了两起事故（2026-09-11 崩训练 / 2026-09-18 dashboard 起不来）。
2. **3 处 definer 改为别名**（**调用点一行未动**）：
   | 文件 | 形态 |
   |---|---|
   | `rl/dashboard.py:3043` | `_force_utf8_stdout = force_utf8_stdout`（assign） |
   | `rl/run_league.py:1547` | 同上 |
   | `scripts/probe_value_ln.py:51` | `from rl.io_bootstrap import force_utf8_stdout as _force_utf8_stdout`（import-as） |
   两处原 docstring 的历史背景**改写为注释保留**（不丢信息）。
3. **行为变更声明（只增不减）**：`probe_value_ln.py` 由弱版升为强版 ⇒ **现在也处理 stderr 且带 `errors="replace"`**。
   那两处强版**逐字等价** ⇒ 行为不变。
4. **新增回归测试** [`scripts/selftest_io_bootstrap.py`](../scripts/selftest_io_bootstrap.py)（**12/12 PASS**，【R8】）：
   行为 9 项（两路 / 参数逐字 / 返回计数 / 异常流不抛 / 幂等 / 真实流切到 utf-8 / **反向验证：中文+emoji 打得出来**）
   + **单一来源 3 项**（复用核对器 `check_utf8`，**不造第二套口径**，【R17】）：
   断言「全仓恰 1 处强实现」「无弱化实现」「别名 ≥3」。
5. **核对器 ② 升级为三阶段可读**（否则收敛后它会**变成空检查**）：收敛前比体指纹；收敛中 `def` + 别名同数；
   收敛后判据 = **恰 1 处强定义 + 0 处弱定义 + ≥2 处别名**。别名同时识别 `X = f` 与 `from m import f as X`。
   `--selftest` 现在 **13 项**，含一条**反向验证**（未收敛时 `single_impl` 必须为 `False`）。

### 1.3 收敛结果（核对器 ② 实测）

```
== ② UTF-8 兜底收敛：定义 1 处 / 别名 3 处 ⇒ 函数体 1 种（含 docstring 1 种）**已收敛为单一实现**
     （定义：src/clasher_new/rl/io_bootstrap.py:44）
     别名 src/clasher_new/rl/dashboard.py:3043  _force_utf8_stdout = force_utf8_stdout（assign）
     别名 src/clasher_new/rl/run_league.py:1547  _force_utf8_stdout = force_utf8_stdout（assign）
     别名 scripts/probe_value_ln.py:51  _force_utf8_stdout = force_utf8_stdout（import-as）
```

### 1.4 ★ T1-1b：61 个**内联调用点**的部分收敛（含**不做的部分**与理由）

方案说「随后按批收敛 ④ 内联」。**先量化**（实测，不是估）：

| 分类 | 个数 | 处置 |
|---|---:|---|
| 含 `sys.stdout.reconfigure(` 的文件 | **61** | — |
| · reconfigure **之前已有** `sys.path.insert` | **29** | ✅ **已收敛**（codemod） |
| · reconfigure 之前**没有** path guard | **17** | ❌ **本步不做**（理由见下） |
| · 形态不匹配（`if hasattr(...)` 变体 / 缩进不同） | **15** | ❌ 本步不做 |

**为什么只做 29 个**：对**没有 path guard** 的文件，收敛就必须**新增一条 `sys.path.insert(0, ...)`** ——
那会**改变这些脚本的 import 解析顺序**（它们可能刻意依赖 cwd 相对导入，例如 `import offline_engagement_trade`）。
收益只是 **DRY**（这些块是**调用点**，不是第二份**实现**），风险不对称 ⇒ **不做**，把清单留档。
**codemod 本身留档**：[`scripts/_converge_utf8_bootstrap.py`](../scripts/_converge_utf8_bootstrap.py)（`--dry-run` 可复现）。

**收敛后验证**：29/29 编译；`run_selftests.py --list` = **100**；抽 3 个仪器端到端
（`offline_engagement_trade.py --help`、`pass_streak_audit.py --help`、`selftest_io_bootstrap.py` 全通过）。

### 1.5 ★ 顺带量到的**真实洞**：stderr 覆盖

61 个调用点里，**绝大多数只处理 stdout**。实测后果：`scripts/duel_search.py` 在 cwd 不对时抛异常，
其 traceback 走 **stderr** ⇒ 在 GBK 下输出成**乱码**（我实测到的原文就是 `�����ļ� 'gamedata.json' ...`）。
⇒ 这不是理论顾虑。**已收敛的 29 个**现在两路都覆盖；**未收敛的 32 个**仍是 stdout-only，列为**未决**（§6）。

---

## 2. T1-2 修 9 处硬编码绝对路径 → **0**

| 文件 | 原样 | 改后 |
|---|---|---|
| `scripts/_schema5_probe.py:12-13` | `"/mnt/e/.../scripts" if os.name != "nt" else r"E:\...\scripts"` | `os.path.dirname(os.path.abspath(__file__))` |
| `scripts/s2_trade_probe.py:21-22` | 同上（`SCRIPTS` 常量） | 同上 |
| `scripts/assassin_left_bridge_test.py:20` | `sys.path.insert(0, r"E:/.../src/clasher_new")` | `<file>/../.. + src/clasher_new`（并补 `import os`） |
| `scripts/assassin_vs_megaknight.py:20` | 同上 | 同上 |
| `scripts/assassin_vs_sparky.py:25` | 同上 | 同上 |
| `scripts/duel_search.py:35` | 同上 | 同上 |
| `scripts/question_bank_poc.py:27` | 同上 | 同上（并补 `import os`，另立 `_ROOT`） |
| `scripts/question_bank_poc.py:235,237` | 两处 `E:/.../runs/archive/...` 与 `E:/.../src/clasher_new/main_ckpt_32000.pt` | 由 `_ROOT` 派生 |

**验证**：核对器 ⑤ **9 处 / 7 文件 → 0 处 / 0 文件**；7/7 编译；`duel_search.py --help` 在文档口径的
cwd（`src/clasher_new`）下 **rc=0**、输出正常 usage。

> **⚠️ 一条重要限定（不是回归）**：`duel_search.py --help` 在**仓库根** cwd 下**仍然失败** ——
> 但它卡在 `import battle` → `card_utils` 的 **cwd 隐性契约**（T0 盘点 §4.4），**与本次改动无关**。
> **取证**：把 `git show HEAD:scripts/duel_search.py` 的**修改前副本**放回同一 cwd 跑 ⇒ **报同一个错**。
> ⚠️ 且这次报错**能看懂原因**，正是 **T0-4** 的收益；改前只有一句裸 `FileNotFoundError: gamedata.json`。
> 契约的根治在 **T3-2**。

---

## 3. T1-3 selftest 可见性（x/y 计数 + 逐测试耗时）

**硬约束**：`main()` 里那 **100 行手工调用**的顺序与内容**不得改动** —— 它是全量路径"逐位不变"的保证，
也是 `scripts/_apply_s2_channel_when_idle.sh:70-71` 等外部调用方的稳定接口。

**做法**：在 `main()` **入口**加一行 `_instrument_tests()` —— 它把 `globals()` 里的每个 `test_*`
换成「计时 + 计数」包装。因为 `main()` 的函数体在**调用时**才按名从 `globals()` 解析，
**那 100 行一行都不用动**。另在末尾（`print("ALL SELFTESTS PASSED")` 之前）加一行 `_report_tests()`。

| 新增 | 作用 |
|---|---|
| `_instrument_tests()` / `_make_wrapper()` | 幂等包装；保留 `__name__`/`__doc__`；失败时也记账后 `raise`（**仍是 fail-fast**，本步**不**改这一点） |
| `discover_tests()` | 全量 `test_*` 按**定义序**（`co_firstlineno`）—— 供顺序对账 |
| `_report_tests()` | `共 N 个测试：x 通过 / y 失败；跳过 k 个` + **跳过清单重复打印** + 最慢 5 项 + 累计耗时 |
| `scripts/run_selftests.py --order-check` | **定义序 / `main()` 序 / `dir()` 字母序**的差集报告 |

**验证 1（最要紧的一条）**：用 AST 抽出 `main()` 的调用序列，与 `git show HEAD:` 版本对比 ⇒
**100 / 100 且序列 `True`（逐字相同）**。
**验证 2**：`discover_tests()` = **100**；包装后 `--list` 仍 **100**；直接跑 2 个测试 ⇒ 记录
`[('test_replay_roundtrip', 0.17, True), ('test_bundle_cap_no_crash', 2.75, True)]`，汇总打印正常。

**`--order-check` 实测**：
```
[order] 三个集合的大小：定义 100 / main() 调用 100 / dir() 列出 100
[order] ① 定义集合 == main() 调用集合 ⇒ OK   （必须成立：有测试没被登记 = 它从不跑）
[order] ② main() 顺序 == 定义序 ⇒ False      （不要求成立，本仓 main() 是手工排序）
[order] ③ dir() 字母序 == main() 顺序 ⇒ False （不要求成立；但不同意味着 --list 挑的子集跑的先后 ≠ 全量）
```

> **偏离方案处（显式说明）**：方案原写「**`--list` 增加**差集报告」。我改成**独立开关 `--order-check`** ——
> 因为 `--list` 有外部消费者（按行数/内容解析），改它的输出契约**风险不对称**：
> 收益只是省一个参数，代价可能是静默打断别人的脚本。

---

## 4. T1-4 临时目录改 `tempfile`（+ `finally` 清理）

`test_draw_penalty_as_loss` 原先 `out_dir="runs/_tmp_drawtest"`，清理语句放在**函数末尾** ⇒
两个毛病：① 断言一旦中途失败，`rmtree` **不执行** ⇒ **在仓库工作区留残留目录**；
② **相对**路径 ⇒ 只有 cwd=`src/clasher_new` 时才落在预期位置。
改为 `tempfile.mkdtemp(prefix="selftest_draw_penalty_")` + `try/finally` ⇒ 无论成败都不在仓库留痕、且与 cwd 无关。

**验证**：跑 `test_draw_penalty_as_loss` **前后** `ls -d runs/_tmp_drawtest` 均 `No such file or directory`；
该测试在 10 项子集里 **PASS**。

---

## 5. T1-5 两个「假绿」SKIP 显式化

| 测试 | 原行为 | 现行为 |
|---|---|---|
| `test_opponent_pool_mix` | 缺 ckpt 时 `print("[SKIP] ...")` + `return` | `_mark_skip(...)` ⇒ 醒目标签 + **进计数** |
| `test_opponent_pool_rand_anchor` | 缺 ckpt 时 `else: print("[SKIP] ...")`，②③④**静默整段跳过** | `_mark_skip(...)`，并在理由里写明**哪几条未验证** |

`_mark_skip()` 用调用栈找出所属 `test_*`，打印 `[SKIP-NO-CKPT] <名>：<理由>`；`_report_tests()`
**再汇总重复打一遍**（混在几千行输出里的一行 `[SKIP]` 是看不见的）。

**验证（真跑跳过分支，不去改 `runs/`）**：打桩 `glob.glob = lambda *a, **k: []` 强跑两个测试 ⇒
**2 条 SKIP 都被登记**、`_report_tests()` 打出「跳过 2 个」与清单；
且第二个测试的**其余分支（①④⑤⑥）照跑并 PASS** —— 与理由里写的「只跑了其余分支」一致。

> ⚠️ **本步不改"绿/红"语义**：缺 ckpt **仍然不算 FAIL**（改成 FAIL 会让换机时整体变红）。
> 它做的是**把"没测"与"测过且通过"分开**。

---

## 6. 未决 / 未做（**不补全**）

| # | 项 | 状态 |
|---|---|---|
| 6.1 | **32 个调用点仍是 stdout-only**（17 无 guard + 15 形态不匹配） | 未收敛；理由见 §1.4。**stderr 乱码这个洞在这 32 个文件里仍在** |
| 6.2 | `probe_value_ln.py` 之外的**弱化实现**是否还有别的形态 | 核对器只认名为 `_force_utf8_stdout`/`force_utf8_stdout` 的 def ⇒ **不覆盖**「换个名字的内联块」；已由 §1.4 的计数间接量到（61 个） |
| 6.3 | `scripts/_schema5_probe.py` / `s2_trade_probe.py` 改后**未做端到端跑** | 它们需要真录像与 `sys.argv[1]`；只做了编译 + 路径推导自证（`os.path.isdir` = True） |
| 6.4 | `test_opponent_pool_*` 的**真实**（非打桩）跳过路径 | 本机两侧各有 51 个 ckpt ⇒ 走不到跳过分支；只能用打桩验证（已做） |
| 6.5 | 全量 selftest 是否仍绿 | **未跑全量**（【R19】默认只跑子集）。**残留风险**：T1-3 的包装机制在**全量 `main()`** 下未经端到端验证（只验证了机制 + 100 行未变 + 子集） |

---

## 7. 本次踩到的工程坑（留档）

| # | 坑 | 现象 | 修法 |
|---|---|---|---|
| 7.1 | **f-string 里嵌套引号 + 反斜杠转义** | Python 3.13 也 `SyntaxError: unexpected character after line continuation character` | 把 `join` 提成变量、改用 `.format()` |
| 7.2 | **我的断言写错**（`f1.calls` 累积了两次调用却只比一次） | 打印 `两路都改 = False`，看起来像**代码**坏了 | 打印实际 `f1.calls` 才看出是**测试**写错。**教训**：`False` 先怀疑断言 |
| 7.3 | 同一命令里 `cd` 过之后再用**相对路径**删文件 | `rm -f` 静默无操作 ⇒ 临时文件残留（Tier 0 §9 已记） | 删后**正向** `ls` 确认 |
