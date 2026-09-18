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
