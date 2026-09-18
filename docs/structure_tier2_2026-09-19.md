# Tier 2 执行记录（2026-09-19）

> 依据 [`structure_optimization_plan_2026-09-19.md`](structure_optimization_plan_2026-09-19.md) §6 的 Tier 2（结构性拆分，8 项）。
> **本文件覆盖本轮已完成的 T2-1 / T2-2**；T2-3…T2-8 的状态见 §4（**未完成项如实标注，不写成已做**）。
> Tier 0 / Tier 1 见 [`structure_cleanup_2026-09-19.md`](structure_cleanup_2026-09-19.md) / [`structure_tier1_2026-09-19.md`](structure_tier1_2026-09-19.md)。

| 项 | 状态 | 关键验证 |
|---|---|---|
| **T2-1** `dashboard.py` 抽 `_HTML`（1879 行 / 80,974 字符）→ `rl/dashboard_html.py` | ✅ | 值 **sha256 逐字相同**（且是**同一对象**）；`dashboard.py` **3,171 → 1,296 行** |
| **T2-2** `dashboard.py` 抽数据面（常量 + 4 缓存 + 24 函数）→ `rl/dashboard_data.py` | ✅ | **A/B 4/4 SAME**（旧代码 vs 新代码，剥掉 `updated_at`）；`dashboard.py` **1,296 → 566 行** |
| 合并回归 | ✅ | selftest 子集 **4/4**；`check_dashboard_js.py` **FAILURES=0**；核对器 ① ② 读数正常 |

---

## 1. T2-1 抽 `_HTML`

### 1.1 做了什么

`dashboard.py` 里 `_HTML = r"""<!DOCTYPE html>…"""` 是**单条语句占 1,879 行**（值 80,974 字符），
与后端逻辑零耦合 ⇒ 抽到 `rl/dashboard_html.py`，`dashboard.py` 侧用
`from rl.dashboard_html import HTML as _HTML` **别名导入** ⇒ **本文件所有引用点一行未动**。

**逐字保证的做法**：不是手抄，而是 `ast.get_source_segment` 取**精确源片段**，只把左侧名字
`_HTML` 改成 `HTML`，右侧字符串字面量**一个字符未改**。

### 1.2 验证

| 检查 | 结果 |
|---|---|
| `_HTML` 值 sha256（抽取**前**实测） | `7dcbde466de4bd9df231fff60372d72a26c604c31b70499bc4e9956e428e723b` |
| `rl.dashboard_html.HTML` | **同上**（长度 80,974） |
| `rl.dashboard._HTML`（别名） | **同上**，且 `HTML is dashboard._HTML` = **True**（同一对象） |
| 外部引用不破 | `rl/selftest.py:1187/1413` 的 `dash._HTML` 仍可用（别名） |
| `dashboard.py` 行数 | **3,171 → 1,296** |

### 1.3 ★ 我的改动打破了一台既有仪器，随即被它自己抓到（留档）

抽出后 `scripts/check_dashboard_js.py` **立刻失败**：

```
[FAIL] dashboard.py 里找不到 <script> 块
```

原因：该仪器原先**只读 `dashboard.py` 的源码文本**、正则找 `<script>…</script>`，而页面已搬到别的文件。
**修法**（三层回退，向后兼容）：① 先读 `rl/dashboard_html.py`；② 再读 `rl/dashboard.py`（抽走前的历史位置）；
③ 都读不到时退化为 `import rl.dashboard._HTML`（最权威，但要拉起 torch）。
修后 **FAILURES=0**。

> ⚠️ **两层教训**：① 仪器**按源码文本**定位而不是**按被测试对象**（这里应该 import 常量）本身就是脆弱点；
> ② 但也正因为有这台仪器，这个破坏在**同一轮内**就被发现 ⇒ 比"等面板白屏了才发现"好得多。

---

## 2. T2-2 抽数据面

### 2.1 先做**搬运前的可行性取证**（三条，都是实测）

| # | 问题 | 实测结论 |
|---|---|---|
| ① | 搬走缓存会不会让 `dashboard.py` 读到**旧对象**？ | `dashboard.py` **L811 之后的代码没有任何一处直接读这些缓存**（`Handler` 只调 `build_*` 函数，`main` 只调 `resolve_state_path`/`make_demo_*`）⇒ 不会 |
| ② | `global` 重绑定怎么办？ | 全文件只有**一处** `global _DECK_INDEX`（`_deck_index()` 内）⇒ **函数与缓存必须同模块**，本文件正是如此（同行搬到 `dashboard_data.py`） |
| ③ | 外部调用方会破吗？ | `rl/selftest.py` 用 `dash.<名>`、`scripts/check_dashboard_js.py` 用 `from rl import dashboard as D` ⇒ 在 `dashboard.py` 做**显式 re-export**（32 个名字）即可，**调用点一行未动** |

### 2.2 搬了什么 / 没搬什么

- **搬**：`MODEL_COLORS` / `MODEL_LABELS` / `CKPT_COLOR` / `FALLBACK_COLORS`（L66-90）、
  4 个缓存（`_REPLAY_META_CACHE` / `_SWEEP_PRIORITY` / `_HEALTH_CACHE` / `_CARD_STATS_CACHE` / `_DECK_INDEX`）、
  以及 24 个数据面函数（`load_state` → `build_card_stats_payload`）；正文由 `dashboard.py` 的**原行整段切片**生成。
- **没搬**：`Handler`、`main()`、`make_demo_*`、内嵌页面（后者已在 `dashboard_html.py`）。
- `dashboard.py` 保留 `_PARENT` / `_REPO_ROOT`（`_REPO_ROOT` 仍被 `check_dashboard_js` 使用）；
  `dashboard_data.py` 自带同名 `_REPO_ROOT`（由 `__file__` 推 ⇒ 与前者**同值**，因为两文件同目录）。

### 2.3 ★ A/B 对账（**这一步差点被我误判成回归**）

第一版对账用「payload JSON 的 sha256」直接比，结果 **4 个 DIFF** —— 看起来像搬坏了。**但我没有据此下结论**，
而是先做**易变性取证**：同一份新代码连跑两遍，逐键比 ⇒ **唯一差异键是 `updated_at`**
（`'2026-09-19T02:20:49' != '…:51'`，payload 里那句 `datetime.datetime.now().isoformat()`）。

随后做**真正的 A/B**：把 `git show HEAD:src/clasher_new/rl/dashboard.py` 取出为临时模块，
**两边都剥掉 `updated_at`** 再比指纹：

| 用例 | 旧（HEAD） | 新 | 结论 |
|---|---|---|---|
| `build_replays_payload(runs/rand100_eval/replays)` | `8bc282ee7cd083f68f25` | 同 | **SAME** |
| `build_card_stats_payload(runs/rand100_eval)` | `3f31e93077a1bb7fa3ad` | 同 | **SAME** |
| `build_replays_payload(runs/et_ctrl100k/replays)` | `3309b549df8e9bff7e41` | 同 | **SAME** |
| `build_card_stats_payload(runs/et_ctrl100k)` | `3f31e93077a1bb7fa3ad` | 同 | **SAME** |

⇒ **4/4 SAME**，行为中性**已证**。临时模块用完即删（`ls` 正向确认不存在）。

> **口径留档**：`build_payload` / `build_solo_payload` 在本轮**无法入 A/B** —— 它们抛
> `PermissionError`，原因是**运行中的 8700/8701 面板占着日志/状态文件**（实测前后一致报同一个错）。
> 这不是本次改动引入的（改动前后**同样**报错），但也意味着**这两个函数的搬运没有 A/B 覆盖** ⇒ 见 §4 未决。

### 2.4 回归

| 检查 | 结果 |
|---|---|
| `py_compile` 两个文件 | OK |
| selftest 子集 `test_dashboard_replays` / `test_dashboard_league_payload` / `test_dashboard_card_stats` / `test_replay_roundtrip` | **4/4 PASS** |
| `scripts/check_dashboard_js.py` | **FAILURES=0** |
| re-export 可用性（12 个常用名 + `_HTML` + `_REPO_ROOT`） | 全部 `hasattr` = True |
| `scripts/_structure_check.py` ② | 收敛状态未变（别名行号随搬移更新为 `dashboard.py:438`） |

---

## 3. 规模变化（核对器 ① 实测）

| 区域 | Tier 1 后 | Tier 2 后 | 说明 |
|---|---|---|---|
| `engine_top` | 28 / 9,604 | 28 / 9,604 | 未动 |
| `rl` | 41 / 23,869 | **44 / 24,115** | +3 文件（`io_bootstrap` / `dashboard_html` / `dashboard_data`）；LOC +246 = 新文件的**头部 docstring**（记录为什么不这么做会出错） |
| `scripts` | 103 / 27,223 | 104 / 27,223 | +1（`selftest_io_bootstrap`） |
| **`rl/dashboard.py`** | 3,171 | **566** | **−82%**（其中 1,879 行是内联页面、~700 行是数据面） |

---

## 4. 未完成 / 未决（**如实标注，不写成已做**）

| # | 项 | 状态 | 备注 |
|---|---|---|---|
| 4.1 | **T2-3** `run_league.py` 抽判定语义（L109-274）+ 评估并行（L581-759） | **未做** | 风险中高：`_eval_pair_worker_main` 是**多进程入口**，移动会改 pickle 的模块路径 |
| 4.2 | **T2-4** `train_solo.py` 抽函数 | **未做** | 高危【R2】；且**CUDA 上无逐位对账**（§11.13.11 实测同配置重跑单点差 **0.60**）⇒ 只能靠导入等价 + 子集 selftest |
| 4.3 | **T2-5** `card_mechanics.py` 按族拆包（58 类） | **未做** | 类名是**字符串查表键** ⇒ 名字一个不能改 |
| 4.4 | **T2-6** `scripts/` 分目录 | **未做** | 会让 `env.md §2.4`、`AGENTS.md`、`finalize_et_solo100k.sh`、`_apply_s2_channel_when_idle.sh`、`run_probe_v3.sh` 的相对路径**全断** |
| 4.5 | **T2-7** 抽奖励簇 → `rl/reward.py` | **未做（并发期禁止）** | 直接撞并发会话的 `rl/config.py` + `rl/engagement.py` 工作集 |
| 4.6 | **T2-8** 拆 `selftest.py` → `rl/selftests/` | **未做（并发期禁止）** | `selftest.py` 是并发会话 S2 取证入口；且 4 条硬约束 + 3 处晚绑定 |
| 4.7 | **T2-2 的 `build_payload` / `build_solo_payload` 无 A/B 覆盖** | **未决** | 因运行中面板占文件而 `PermissionError`；要覆盖需**先停面板**（需用户同意）或换 `--state` 指向副本 |
| 4.8 | 8700/8701 面板**仍在跑旧代码** | **未决** | 抽包不改变已加载进程的行为；新进程已用新代码验证（`--help` rc=0 + 前端回归）。**是否重启面板验证由用户定** |
| 4.9 | Tier 3（T3-1…T3-4） | **未做** | T3-1 拆 `battle.py` 触【R13】且**并发会话正在改 `battle.py`**；T3-2 数据归拢会同时打断 **20 个 importer** |

---

## 5. 本轮新踩到的环境坑（留档）

| # | 坑 | 现象 | 判据/修法 |
|---|---|---|---|
| 5.1 | **WSL 的 `127.0.0.1` ≠ Windows 面板的 `127.0.0.1`** | 从 bash 工具 `curl http://127.0.0.1:8701/api/solo` 得 **HTTP 000**（无响应），**看上去像面板死了** | 本机是 **WSL2**（`Linux 6.18…-microsoft-standard-WSL2`，eth0 `172.28.144.0/20`）⇒ **两个网络命名空间**。真判据：① `ps -ef` 里 4 个 dashboard 进程仍在；② 面板**日志尾部**有 `[dashboard] "GET /api/solo HTTP/1.1" 200`（**服务端自证**）。⚠️ 面板绑的是 `--host 127.0.0.1` ⇒ 从 WSL 侧**根本连不上**，不是配置问题 |
| 5.2 | **`build_payload` / `build_solo_payload` 的 `PermissionError`** | T2-2 的 A/B 拿不到这两个用例 | 成因（本条**未定**）：怀疑是**运行中的面板占着 `solo_state.json` / 训练日志**（Windows 文件锁）。**但改动前后同样报错** ⇒ 不是本次引入 |
| 5.3 | **一度把"挂钟"当成"回归"** | 直接比 payload sha256 得 **4 个 DIFF** | 先做**易变性取证**：同码连跑两遍、逐键比 ⇒ 唯一差异键 `updated_at`（`datetime.now().isoformat()`）⇒ 再做剥离后的真 A/B（**4/4 SAME**）。**教训**：payload 类对象的指纹必须先剥时间戳 |

---

## 6. T2-3（判定语义半簇）与 T2-5（`card_mechanics` 拆 M8 段）

### 6.1 T2-3 · 判定语义 → `rl/league_rules.py` ✅（**评估并行半簇未搬**）

**搬了**：`STALL_WINDOW`/`STALL_LIMIT`、`towers_hp`、`_min_alive_tower_pct`、`timeout_winner`、
`settle_stall_from_counts`、`settle_stall`、`_stall_probe`（原 `L151-275`，`run_league.py` **1,784 → 1,665 行**）。
`run_league.py` 侧**显式 re-export** 8 个名字 ⇒ 既有调用点（`rl/selftest.py` 的 `run_league.timeout_winner` /
`settle_stall` / `STALL_LIMIT` / `_stall_probe`；`rl/evaluate.py` 的 `from rl.run_league import timeout_winner`）**一行未动**。

**无环是怎么保证的（实测，不是推测）**：用 AST 逐个函数扫**自由名** ⇒ 该簇对模块内的**唯一**外部依赖是
`_overtime_timeout_winner`（来自 `rl.overtime`）⇒ `league_rules` **不 import `run_league`**。

| 对账 | 结果 |
|---|---|
| 8 个定义的**函数体**（去空白后 sha256）HEAD vs 新模块 | **8/8 SAME** |
| re-export 可用性 + 与 `league_rules` 同名对象 | True |
| `rl.evaluate` / `rl.flow_league` / `rl.human_play` 导入 | OK |
| selftest 子集（加时/平局/僵局/联赛/并行评估 ×10，**含 spawn 多进程**） | **10/10 PASS** |

**★ 为什么「评估并行」半簇（原计划 L581-759）没搬**：它的依赖
`_play_one_game` / `_spec_to_policy` / `_eval_env` / `_ET_MEASURE` **全部位于该边界之前**，
而方案又要求 `_ET_MEASURE` / `_set_et_measure`（**S2 接线点**）留在 `run_league.py` 原地
⇒ 直接搬会形成 `run_league ⇄ league_eval` **循环 import**。破环有两条路，**都不做**：
① 把 `_ET_MEASURE` 一起搬（违反方案的「保留原地」）；
② 给 `_run_eval_pairs_parallel` 加 `et_measure` 形参（**改签名**，不再是"纯搬运"）。
⇒ 记录为未做（§4 已列）。

> **附带查到一条硬约束**：`rl/selftest.py:5298` 断言
> `run_league._eval_pair_worker_main.__module__ == "rl.run_league"` —— 即**该 worker 必须留在
> `rl.run_league`**（spawn 的 pickle 路径契约）。这独立佐证了上一条决定。

### 6.2 T2-5 · `card_mechanics.py` 拆 M8 段 → `card_mechanics_elite17.py` ✅（**不是**方案写的「包」）

**方案原写**：「7 族 → `card_mechanics/` 包 + `__init__.py` re-export 全部 58 类」。
**我改成了两文件切分**，理由是一个**具体的危险**：`battle.py:5` 是 `from card_mechanics import *` ——
`import *` 的语义是「导出该模块**公开命名空间**」，而拆分前的 `card_mechanics` 会把**它自己的 import**
（`math` / `BasicCharacter` / `Position` / `Card` / `TileGrid` / `OFFICIAL_OVERRIDES` / `level_scale`）一并**泄漏**给 `battle`。
改成包、由 `__init__.py` 重新聚合，**未必**能复现这套泄漏 ⇒ 有静默改变 `battle` 命名空间的风险。
**两文件方案把风险降到 0**：`battle.py:5` **一字不动**，`card_mechanics` 仍是那个聚合模块。

| 项 | 值 |
|---|---|
| 切点 | 原 `L856` 的 `# ===== M8：Elite17 …` 横幅；M8 段占 **1,040 行 / 35 个定义 + `HERO_CLASSES`** |
| 依赖方向（AST 实测） | **基准段引用的 M8 名字 = 0 个** ⇒ 单向、干净 |
| 结果 | `card_mechanics.py` **1,896 → 861 行**；新增 `card_mechanics_elite17.py` 1,071 行 |
| 反向需要 | M8 段**精确**需要 5 个基准名：`Balloon` / `Prince` / `DarkPrince` / `IceWizard` / `_HeroBase`（逐函数扫自由名得到，**不是猜的**） |

**★ 「后半部分不能独立导入」这一失败模式，我做了守卫**：若先 `import card_mechanics_elite17`，
则 `card_mechanics` 末尾的 `import *` 会在该模块**尚未完成**时执行 ⇒ 公共命名空间会**静默少掉 35 个定义**。
故 `card_mechanics_elite17.py` 顶部加显式检查并**大声抛 `ImportError`**（附正确用法），而不是留个坏模块。
实测：直接导入 ⇒ **被拒**；`import card_mechanics` / `import battle` ⇒ 正常。

**逐名对账（本步最硬的一条证据）**：

| 检查 | 结果 |
|---|---|
| 公开命名空间（`vars(mod)` 里不以 `_` 开头的）拆前 vs 拆后 | **65 名 == 65 名，缺失 0 / 新增 0** |
| 泄漏名仍指向同一对象（`Position`/`BasicCharacter`/`Card`/`TileGrid`/`OFFICIAL_OVERRIDES`/`math`） | **True** |
| `HERO_CLASSES` 条目数 | **16**（与拆前同） |
| `import battle` + `battle.HeroKnight` 解析 | OK（`__module__` 现为 `card_mechanics_elite17`） |
| **`__module__` 依赖排查** | 全仓**无**对 `card_mechanics` 类 `__module__` 的依赖（唯一命中是 `selftest.py:5298` 对 `_eval_pair_worker_main` 的断言，**与本次无关**） |
| `scripts/test_m6_elite.py`（M8 机制真实跑 battle） | **通过 84 / 失败 0** |

> ⚠️ **限定**：把类挪到新模块会改 `__module__`。若**将来**有人对这些类做 pickle 或按 `__module__` 查表，
> 就会踩到（当前**实测无**）。这一条写进 §4 未决。

---

## 7. T2-5 验证过程中的**意外收获**：13 个入口脚本缺 UTF-8 兜底 ⇒ 引擎测试在**假失败**

### 7.1 怎么发现的

T2-5 后按计划跑引擎测试，得到 `scripts/test_m3_evo.py` **通过 39 / 失败 12**、`test_m5_data.py` **39 / 1**。
**我没有先假设是自己的回归**，而是先做 A/B：`git stash` 掉本次 tracked 改动（`card_mechanics.py` + `run_league.py`）
再跑同一批 ⇒ **HEAD 上同样是 39/12 与 39/1** ⇒ **既有失败，与本次拆分无关**。

随后把失败条目解出来（`iconv -f gbk`）：

```
FAIL test_musketeer_snipe 异常  ['gbk' codec can't encode character '\u246a' in position 4: illegal multibyte seq]
FAIL test_valkyrie_tornado 异常  [ … '\u246b' … ]      ← 同因
… 共 10 条同因
FAIL 冲刺后贴脸（<1.5 格）  [d=4.00]                      ← 真实行为失败
FAIL 落地冲刺伤害 ≈545  [dmg=0]                          ← 真实行为失败
```

**根因**：该文件的测试名用了 ⑪⑫⑬…（`U+246A` 起）—— **GBK 编不出**；而文件没有 UTF-8 兜底
⇒ `print(测试名)` 抛 `UnicodeEncodeError`，被本文件的 `try/except` 记成「异常」⇒ **10 个假失败**。

| 加兜底前 | 加兜底后 |
|---|---|
| 通过 **39** / 失败 **12** | 通过 **60** / 失败 **2** |

⇒ 不只 10 个假失败消失，**另有 21 个测试现在才真正跑起来**（原先一抛异常就跳过该子测试的断言）。
**剩下 2 个是真实行为失败**（冲刺贴脸距离、落地冲刺伤害），**本次不动**（属引擎语义，超范围）。

### 7.2 修了什么（**入口脚本 13 个**）

判据：**含 GBK 编不出的字符** ∧ **无任何 UTF-8 兜底**。分两类处置：

| 类 | 个数 | 处置 |
|---|---:|---|
| 已有 `sys.path.insert`（⇒ `rl` 可导入） | **10** | 在 path 引导之后 `from rl.io_bootstrap import force_utf8_stdout; force_utf8_stdout()` |
| **完全不 import `rl`**（纯文本/AST/子进程工具：`_agents_split` / `_patch_battle_root_cast` / `bench_train_speed`） | **3** | 补一段**自足** path 引导（与仓内 69 个脚本同形态）再 import 单一实现 |

**刻意不用「复制一份手写 `reconfigure` 块」**——那会造出 T1-1 刚消灭的**第二份实现**。

**⚠️ 只加在入口脚本上，`rl/` 库模块一律不加**：库在 import 时改宿主的 stdout/stderr 是错的
（会污染调用方）。⇒ 扫描里 25 个 `rl/*.py` 模块**有意不动**。

### 7.3 顺带发现：仓库实际有**四种** UTF-8 引导形态

| # | 形态 | 实测数量 | 缺陷 |
|---|---|---:|---|
| ① | `try: sys.stdout.reconfigure(encoding="utf-8") except: pass` | 见 T1-1b | 不管 stderr、无 `errors="replace"` |
| ② | 同 ① 但带 `errors="replace"` | 见 T1-1b | 不管 stderr |
| ③ | `sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")` | **6** | 不管 stderr、无 `errors="replace"`、且**替换** stdout 对象（调用方若持有旧引用会失效） |
| ④ | `from rl.io_bootstrap import force_utf8_stdout`（T1-1 的单一实现） | **3 + 13** | — |

### 7.4 ★ 我自己在改这 13 个文件时踩的坑（留档）

**第一次插入把 8 个文件改成了语法错误。** 原因与修法：

| 步 | 我做了什么 | 结果 |
|---|---|---|
| 1 | 用**正则** `^\s*sys\.path\.insert\(` 找插入点，插在匹配行**之后** | ❌ `assassin_*` / `duel_search` 的 `sys.path.insert(0, os.path.join(` 是**跨行调用** ⇒ 代码被插进**括号中间** ⇒ `SyntaxError`（8 个文件） |
| 2 | 改用 **AST** 取语句 `end_lineno` | ❌ 我的「点号名提取」只处理 **2 级**属性（`os.chdir`），而 `sys.path.insert` 是 **3 级** ⇒ 一个都没匹配上 ⇒ 断言失败（**这次失败救了场：断言在写入前触发，8 个文件未被再次破坏**） |
| 3 | 改为**通用属性链**还原（`while isinstance(node, ast.Attribute)` 逐级收集） | ✅ 3 个文件通过；`diag_critic_ev.py` 仍失败 |
| 4 | 发现 `diag_critic_ev.py` 的 insert **嵌在 `if _SRC not in sys.path:` 里** ⇒ 只看 `tree.body` 漏掉 | 改为 `ast.walk` 全树 + 「必须落在首个产品 import 之前」约束 ⇒ ✅ |

**收尾**：把 8 个被改坏的文件 `git checkout HEAD --` 回退（它们的 T1-2 / T1-1b 改动**已在 HEAD 里**，不会丢），
再用第 3/4 步的正确方法重做 ⇒ **10/10 编译通过**，并逐个运行时冒烟：
`test_m3_evo`（**60/2**）、`assassin_left_bridge_test`（真跑引擎，能打出 `✓`）、`duel_search --help`（rc=0）、
`health_curve --selftest`（**5/5**）、`diag_critic_ev --help`、`_agents_split --check`（**PASS**）、
`_patch_battle_root_cast --help`、`bench_train_speed --help`。

> **教训（写进 §4 未决）**：**跨行调用**上做文本/正则插入是陷阱；插代码必须用 AST 的 `end_lineno`，
> 且点号名要**通用**还原（别假设属性级数）。另：一次插入失败后**别在原状上继续改**，
> 先 `git checkout` 回基线再重做 —— 我这次正是靠这个 + 断言前置于写入，才没有留下半坏的树。

---

## 9. T2-8 拆 `rl/selftest.py`（**6,023 行 → 172 行**）

### 9.1 拆成什么

| 文件 | 行数 | 内容 |
|---|---:|---|
| `rl/selftest.py` | **172** | 原 docstring（回归索引）+ **顶部 path 引导** + 聚合重导出 + **`main()`** + `if __name__` |
| `rl/selftest_common.py` | 202 | 模块头导入 + `_PARENT` + **4 个模块级状态**（`_RUN_STATS`/`_SKIPS`/`_ORIG_FUNCS`/`_INSTRUMENTED`）+ **10 个 helper** |
| `rl/selftests/part{1..5}.py` | 各 ~20 测试 | 100 个测试按**定义序**每 20 个切一片（`part1` = `test_action_bundle_same_tick` … ；`part5` 末 = `test_mask_partial_bundle_invariants`） |
| `rl/selftests/__init__.py` | — | 说明「这 5 个 part **不是**独立可跑模块」 |

### 9.2 拆之前先做的**可行性审计**（方案要求的前置，实测）

| 审计项 | 结果 |
|---|---|
| `__file__` 出现处 | **仅 3 处**：`_PARENT`(L29) + **两个测试内**（L5615/L5676 自算 root 去 AST 扫 `battle.py`/`rl/*.py`） |
| 测试依赖的**模块级名字** | **仅 6 个**：`_make_policy_and_tokens`(5 个测试用) / `_tiny_rollout_transitions`(3) / `_mk_env`(2) / `_mark_skip`(2) / `_FakeCfg`(1) / `_intents`(1) |
| test → test 调用 | **0**（【T0 盘点】已证） |
| 四条硬约束可否满足 | ✅ 全部可用 re-export 满足（见 9.3） |

### 9.3 四条硬约束**一条都没破**

| 约束 | 怎么满足 |
|---|---|
| ① `scripts/run_selftests.py:36-42` 用 `dir()` 反射按名调用 | 聚合模块 `from rl.selftests.partN import *` ⇒ 100 个 `test_*` 都是 `rl.selftest` 的属性（`--list` 实测 **100**） |
| ② `scripts/rl/selftest.py:5-7` 用 `runpy.run_path(<SRC>/rl/selftest.py)` | `main()` 与末尾 `if __name__` **留在聚合文件** |
| ③ `main()` 的 100 行手工调用清单逐字不变 | **AST 对账：HEAD 100 个 / 现在 100 个，序列 `True`** |
| ④ `scripts/_apply_s2_channel_when_idle.sh:70-71` 外部硬编码 4 个测试名 | 实测 4 个名字都 `callable`（`bash -n` 也 OK） |

### 9.4 拆分**必须**配套的 6 处代码改动（缺一个就静默坏或直接崩）

| # | 改动 | 不做会怎样 |
|---|---|---|
| 1 | 聚合文件**顶部加 path 引导**（`_PARENT` + `sys.path.insert`），且**必须在任何 `from rl.` 之前** | ✗✗ **实测到过**：`python rl/selftest.py` 时 `sys.path[0]` 是 `rl/` ⇒ `ModuleNotFoundError: No module named 'rl'`（**全量套件根本没跑起来**）。旧文件本就是靠这段，拆分后这段**只能留在聚合文件** |
| 2 | `_instrument_tests(namespace=None)`：聚合文件显式传 `globals()` | 包装器默认只看 `selftest_common` 的 globals（**一个 `test_*` 都没有**）⇒ **静默**一个测试都不计时、不计数 |
| 3 | `register_namespace(globals())` + **导入期**就 `_instrument_tests(globals())` | `discover_tests()` / `--order-check` 原先靠 `main()` 的副作用填充 ⇒ 不跑 `main()` 时返回**空** |
| 4 | `discover_tests()` 的排序键改成 **`(模块名, 局部行号)`** | 各 part 的**局部行号会交错**（part1 第 30 行 vs part2 第 30 行）⇒ 「定义序」失真 |
| 5 | 两个自算 root 的测试改用 `_PARENT` | 测试搬到 `rl/selftests/` 后 `dirname(dirname(__file__))` 少一层 ⇒ AST 扫不到 `battle.py` |
| 6 | `_make_wrapper` 用 **`functools.wraps`** | 包装后所有测试的 `__module__` 都变成 `rl.selftest_common` ⇒ 排序键(4)失效、且调试时看不出测试在哪片 |

### 9.5 验证（**全量套件真跑过**）

| 检查 | 结果 |
|---|---|
| **`python rl/selftest.py` 全量** | **`[selftest] 共 100 个测试：100 通过 / 0 失败；跳过 0 个`**、**`ALL SELFTESTS PASSED`**、EXIT=0；累计 **218.5 s**；最慢 5 项 `test_solo_resume 42.7s` / `test_solo_mode_smoke 31.9s` / `test_opponent_pool_rand_anchor 13.1s` / `test_league_resume 12.0s` / `test_eval_solo_parallel 11.3s` |
| `main()` 调用序列 vs HEAD | **100/100 逐字相同** |
| `run_selftests.py --list` | **100** |
| `run_selftests.py --order-check` | ① 定义集合 == main() 调用集合 **OK** |
| **跨 5 个 part 的子集**（每片 2 个，10 个） | **10/10 PASS**（这证明每片的**显式导入完整** —— 缺一个私有 helper 就是 NameError） |
| `discover_tests()` / `dir()` 里的 `test_*` | **100 / 100**，且分片计数 **20+20+20+20+20** |
| `scripts/_structure_check.py` ⑦ | **定义 100 / main() 调用 100** |
| 全量日志 | `docs/selftest_full_after_t2_8.log`（2,564 行；`docs/*.log` 已 gitignore ⇒ 关键行已抄进本表） |

### 9.6 ★ 我在这次拆分里**漏了 3 次**，全部由仪器而不是靠复读发现

| # | 我漏了什么 | 谁抓到的 | 修法 |
|---|---|---|---|
| 1 | 模块级**状态变量**没搬过去（`_RUN_STATS`/`_SKIPS`/`_ORIG_FUNCS` 是 `AnnAssign`，我的"零丢失"对账第一版**只查 `Assign`** ⇒ 第一次只补回 `_INSTRUMENTED`） | `--order-check` 报 `NameError: _INSTRUMENTED` | 对账改成**含注解赋值**的完整绑定集 ⇒ 丢失归 0 |
| 2 | 核对器 **⑦ 变成假失败**（它只 AST 解析聚合文件 ⇒ 「定义 0 个」而 main() 有 100） | 核对器自己 `rc=1` | ⑦ 改为**分片感知**（定义数从聚合文件 + 全部分片一起数） |
| 3 | 聚合文件**没有 path 引导** ⇒ 全量套件**根本起不来** | 后台全量日志第一行 `ModuleNotFoundError: No module named 'rl'` | 顶部补 `_PARENT` + `sys.path.insert`（并写进注释说明"必须在任何 `from rl.` 之前"） |

| 4 | 核对器 **① 的 `rl` 区域是非递归扫描** ⇒ 新子包 `rl/selftests/`（6,164 行）**没被计入**，`rl` 读数从 24,145 **掉到 18,495** —— 看上去像"代码少了 5,650 行" | 我自己核对 ① 的读数时觉得不对 | 改成 `recursive=True`；并把这条写进代码注释（"这类假读数最危险：没人会质疑行数变少"） |

> **第 3 条与 Tier 1 的 T1-1b 是同一类失败模式**（`import` 排在 path 引导之前）的**第二次发生** ⇒ 我已把核对器 **⑨** 的覆盖面从 `scripts/` **扩到 `rl/` 与引擎顶层**（`__init__.py` 除外），扩展后 ⑨ 仍为 **0**。

---

## 10. T2-6（`scripts/` 分目录）—— **实测代价后判定不做**，改用自动索引

### 10.1 为什么先量代价

T2-6 的风险**不在功能面而在文档面**，所以先按家族把「引用面」量出来（实测，不是估）：

| 家族 | 文件数 | 全仓文档提及 | 其中**活跃索引** | 涉及**历史留证**文档数 |
|---|---:|---:|---:|---:|
| `test_m*.py` | 6 | 193 | 4 | 13 |
| `_survey_*.py` | 9 | **128** | 0 | 8 |
| 其它 `_*.py` | 13 | 138 | 15 | 36 |
| `probe_/diag_/forensics_*` | 30 | 264 | 30 | 59 |
| 其它 | 35 | 552 | 75 | 72 |
| **合计** | **93** | **~1,275** | ~124 | — |

外加：**全仓 `scripts/<名>.py` 被提及 1,293 次 / 113 份文档**（按扩展名：`.md` 1,293、`.py` 217、`.sh` 5、`.ps1` 1、`.bat` 0）。

### 10.2 判定 = **不做**（三条理由，全部基于上面的数字）

1. **文档面代价压倒一切**：`~1,167 / 1,293` 次提及落在 `docs/*.md`（98 份）里，而这些**绝大多数是历史留证/判读文档**。
   本仓纪律是「新决策先写 docs（可改）」+「**只增不改历史结论**」 ⇒ 搬迁要么**漏改**（文档开始说谎），
   要么**改写历史**（违反纪律）。**即使只搬最小的家族（`_survey_*` 9 个）也要碰 8 份历史文档**（128 处提及）。
2. **功能面其实很小**：`.sh` 只有 **5 处**、`.ps1` 1 处（且是散文）、`.bat` 只引用 `scripts/rl/*`（方案自己就要求保持原位）
   ⇒ 「怕改漏导致脚本跑不起来」这个担心**方向是错的** —— 真风险全在文档。
3. **T2-6 想要的「可寻性」已经给到**：**T0-3** 的 `docs/agents/scripts_inventory.md`（91 个脚本 + docstring 首行 +
   `--selftest`/assert 数 + 是否已登记）+ 本轮的 `scripts/README.md`（分类 + 漂移检查）。
   ⇒ **不搬文件也能一眼分类**。

### 10.3 替代物（做了）：`scripts/README.md` + 检查 **⑩**

| 项 | 内容 |
|---|---|
| 生成物 | `scripts/README.md` —— **自动生成**（勿手改）：4 类（测试 / 一次性取证 `_` / 探针诊断仪器 / 其它）+ 每类的判据 + 文件清单 + 合计 |
| 生成/校验器 | `scripts/_structure_check.py`：`--write-scripts-readme` 重写；**检查 ⑩** 校验一致性（**进硬门禁**） |
| 分类依据 | **纯文件名前缀**（确定性、可复现）⇒ 只影响"读到哪一行"，**不影响任何运行行为** |
| 判别力 | 合成树：生成后 OK → 加一个脚本 **必须报漂移** → 删掉 **必须回 OK**；真仓库同样实测：临时加 `scripts/zz_drift_probe.py` ⇒ **FAIL/DRIFT（94 个）+ rc=1**，删掉 ⇒ OK + rc=0 |
| 附带 | `scripts/README.md` 里**逐字写明**「为什么没做 T2-6」+ 上表 + 「若将来要搬必须同批改哪些」⇒ 决策与数字**跟着目录走**，不会只躺在 `docs/` 里 |

> **未来若真要搬**（写进 README）：必须**同批**改 `env.md §2.4` / `AGENTS.md` / `docs/README.md` /
> `docs/agents/scripts_inventory.md` / 3 个 `.sh`，并在 98 份历史文档**顶部加一行「路径已迁移」**（而不是改写正文）。
