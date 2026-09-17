# dashboard 对接新训练（run 模式 · `long1m` 1M 步长跑）—— 2026-09-18

> 关联：预注册 [`docs/long1m_prereg_2026-09-18.md`](long1m_prereg_2026-09-18.md)、
> run 模式语义 [`docs/run_mode_multideck_2026-09-17.md`](run_mode_multideck_2026-09-17.md)。
> 【红线 R18】本文件与代码改动同批提交（`rl/dashboard.py`、`rl/config.py`、`rl/run_league.py`、
> `rl/selftest.py`、`scripts/check_dashboard_js.py`）。

## 1. 为什么要"对接"

原来联赛面板（`--state <league_state.json>`）已经能画 Elo 曲线 + 1σ 误差棒 + 3 s 轮询，
但对 **1M 步 / 131 个评估点 / 约 16 h** 的长跑有四处不够用：

| # | 缺口 | 后果 |
|---|---|---|
| ① | `--state` **只认 JSON 文件**（`--solo`/`--play` 早就支持目录） | 盯着 `runs/long1m/` 还得写全文件名；给目录会 500 |
| ② | 只有 `league_state.total_steps`（= **最近一个评估点**的步数） | 页面上写"总训练步数：24000"，跑到 2.4% 时看起来像**已经跑完** |
| ③ | 没有大/小评估点的区分 | 小点（10 局/对，SE≈49 Elo）与大点（20 局/对，SE≈35）在图上都是"一个点 + 一根棒"，看不出哪个点可信 |
| ④ | 只有 Elo 一条口径 | run 模式的对手是**固定脚本**，逐对手胜率是**非自引用**读数（与 solo 的"打冻结副本"完全不同），但 `league_state.winrates` 只是标量 EMA，没有曲线 |

## 2. 改了什么

| 改动 | 位置 | 说明 |
|---|---|---|
| **目录入口** | `dashboard.resolve_state_path()` | `--state runs/long1m` 自动找 `league_state.json`；目录里没有该文件时给明确提示（"run 模式还没写出第一个评估点？"）；非目录路径原样返回（错误信息指到你查的地方） |
| **长跑进度条** | `dashboard._league_run_meta()` + 前端 `renderRunMeta()` | 读同目录的 `run_state.json`（当前 step）+ `config.json`（两级评估超参）⇒ `24,000 / 1,000,000 步（2.4%）· 评估点 4/131（大 11 / 小 120）· 已跑 3.5 h · 粗估剩余 …`；进度条颜色在 ≥99.5% 时转绿 |
| **卡死检测** | 同上 | `state_age_s`（状态文件多久没写入）+ `stale`（> 15 min）⇒ 前端把时间与提示**变红**（小点节奏 ≈5.5 min，15 min 无写入基本就是卡死/退出了） |
| **大/小评估点标注** | `round_stats[i].kind` / `games_per_pair` + 图上竖虚线 | `kind` 来自 `eval_schedule`（见 §3），前端对 `kind=="big"` 的点画浅色竖虚线 |
| **逐点对手胜率曲线** | `dashboard._winrate_curves()` | 从 `history` + `round_stats[i].games` 的累计局数**切分复原**（见 §4），前端新增指标下拉框：`轮内估计 Elo`（默认）/ `对每个对手的胜率`；胜率图带 `±SE=√(p(1-p)/n)` 误差棒与 0.50 基准线 |
| **排名表新增胜率列** | 前端 `renderTable()` | main 行显示"对手均值"（明细在 tooltip），其余行显示自己的胜率（= 1 − main 对自己的胜率） |
| **横轴改按总计划步数** | 前端 `drawEloChart()` | 有 `run_meta.total_steps` 时用它当横轴上限 ⇒ 曲线不再被最近几个点挤满，"才跑了 2.4%"一眼可见 |

## 3. 防"计划 vs 实际"漂移：单一来源 `eval_schedule`

两级评估的**并集网格**（小点 `steps_per_eval`、大点 `big_eval_every`，不整除时大点被**插入**；
同一步命中两网格只评一次按大预算）现在只有**一份实现**：

- `rl/config.py::eval_schedule(...)` —— 纯函数，无 torch 依赖，返回 `[(step, kind, n_games), ...]`；
- `rl/run_league.py::_EvalScheduler.expected_points()/expected_plan()` **委托**它（实际触发仍走
  `due()`，语义未变；`test_eval_scheduler_two_tier` 断言两者逐点一致）；
- `rl/dashboard.py::_league_run_meta()` 用它算 `plan_points / plan_big / plan_small` 与逐点 `kind`。

不这么做的具体危害：dashboard 若自己数一遍网格，改超参后会出现"页面说 131 点、实际只跑了 128 点"，
而这类漂移在长跑里**看起来就像训练异常**（会浪费一整轮归因）。

## 4. 逐点胜率是怎么复原的（含口径限制）

`league_state.json` 里 `winrates` 只有**标量 EMA**（如 `"main|push_flow": 0.857`），**没有时间维度**；
但另有两样东西可以复原曲线：

- `history`：逐局 `[a, b, score_a]` 的**追加流**（**不含 step**）；
- `round_stats[i].games[aid]`：第 i 个评估点每方的局数 ⇒ `sum(games.values()) // 2` = 该点总对局数。

于是按评估点顺序**切分** history（第 i 段 = 第 i 个点的全部对局），段内按 `(a,b)` 聚合
`wr = (胜 + 0.5·平) / 局数` ⇒ `winrate_curves["main|push_flow"] = [[0,0.95],[8000,0.90],…]`。

⚠️ **限制（判读时注意）**：

1. 切分依赖 `history` 的顺序与 `round_stats` 的局数**严格一致**（两者都由同一进程按同一顺序写）；
   若某点还在写（history 长度 < 累计局数），该点**整点丢弃**（宁可缺点，也不要半截读数造成假跳变）。
2. 这条曲线用的是**点内全部局**（含 main 先手/后手换边），与 `round_stats.est` 的
   `D̂ = 400·log10((w+0.5)/(n−w+0.5))` 口径不同 ⇒ **ElO 与胜率两条曲线不可互相换算**，
   判决仍以预注册的 Elo 判据为准（见 prereg §4 J3）。
3. run 模式的胜率曲线**没有** solo 那类"自引用禁读"问题（对手是固定脚本、不学习）；
   但它对**对手池**敏感：本 run 的对手随每局重抽卡组 ⇒ 曲线含卡组方差，单点 50 局的 ±SE 只是二项噪声。

## 5. 验证（三层，全部可复跑）

1. **后端单测**：`test_dashboard_league_payload`（`scripts/run_selftests.py` 按名跑）
   —— 手算 fixture：小点 10 步×2 局、大点 15 步×3 局、总 30 步 ⇒ 计划 `[0 大,10 小,15 大,20 小,30 大]`；
   断言 目录入口 / 进度来自 `run_state`（20/30 = 66.67%）/ 计划点数与 `_EvalScheduler` **逐点相同** /
   round_stats 的 `kind` 标注 / 胜率切分复原 `[[0,0.5],[10,0.75]]` / 20 min 未写入 ⇒ `stale` /
   缺 `run_state`+`config` 时**降级不炸**。
2. **前端渲染冒烟**：`scripts/check_dashboard_js.py --league-run runs/long1m`
   —— node + DOM 桩跑真实 payload：排名表（7 列 × N 行）、进度条（含千分位步数/评估点进度）、
   缺 `run_state` 时留空、Elo 曲线（含大点竖虚线）、胜率曲线与指标切换、空 payload 走空分支。
   另附 **数据级不变量**：已评估点里的"大点"个数 == 计划中**已到达**的大点数（漏读 config ⇒ 立刻 FAIL）。
3. **在线探针**（Windows 侧，因 WSL 访问不到 Windows loopback）：
   `http://127.0.0.1:8700/` 200 且含 `leagueMetric`/`leagueProgress`；
   `/api/state` 200（`run_meta` 24,000/1,000,000、4/131 点、`stale=false`、5 条 `main|*` 胜率曲线）；
   `/api/replays` 200（4 个 `league_*.pkl`）；
   `/api/cardstats?files=2` 200 且 `n_games=100`、`n_decks=6`、`coverage.games_with_decks=100`
   （说明新 run 的录像**带 `meta.decks`**，按卡组矩阵可用）。

## 6. 启动姿势（照抄）

```bash
# 在 src/clasher_new 下；端口 8090 被系统保留段占用（见 AGENTS.md §2.5），用 8700
PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe -u rl/dashboard.py \
  --state runs/long1m --replays runs/long1m/replays --host 127.0.0.1 --port 8700
```
> `--replays` 可省（默认取状态文件同目录的 `replays/`）。想同时看 solo 面板再加
> `--solo runs/<solo_run>`，两个面板会在同一页并列显示。
> 页面每 3 s 轮询 `/api/state`、每 5 s 轮询 `/api/replays` ⇒ 训练写入即刷新，无需重启 dashboard。

## 7. 顺带看到的行为（**不是结论**，仅作为"仪表盘能看见了"的例子）

`/api/cardstats?files=2`（前 2 个评估点、100 局）里 main 的出牌分布：
`Arrows 253 / Minions 246 / Archer 245 / Knight 238 / Fireball 80 / Musketeer 78 / MiniPekka 65 /
Giant 30`（累计 1235 次）——即**便宜卡几乎均匀、唯一赢牌手段 Giant 只占 2.4%**，
与【未决 O3】在 solo 侧的取证（"手里有什么便宜的就打什么"、Xbow 出手 0.03%）**方向一致**。
n=1 个 run、2 个评估点、无对照 ⇒ 按【红线 R5】**只能当作描述**，不得据此下判决。
