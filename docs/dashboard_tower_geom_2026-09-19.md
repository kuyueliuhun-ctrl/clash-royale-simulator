# dashboard 回放的塔尺寸 = 引擎几何（2026-09-19，用户直接要求）

> 用户原话：「**现在我们引擎已经正常实现塔了，但是 dashboard 回放的塔还是 1 格，把它恢复到正常尺寸**」。
> 本文记录：现象、根因（含**历史**，不写成"我们改坏了"）、改法、**验证证据**、**未做与边界**、复现命令。

## §0 结论（一句话）

前端画塔的尺寸原先**写死**在 `dashboard_html.py` 的 JS 里（`half = 0.5 格` ⇒ 公主塔画成 **1 格见方**、
王塔 **1.24 格**），而引擎自 **2026-09-09** 起塔是**矩形**（公主塔 **3×3**、王塔 **4×4**，
`arena.TileGrid.towers`）⇒ **同一份几何前后端各写一份**。
改法 = **几何由引擎侧写进录像**（局级可选键 `meta["tower_geom"]`，**不升 schema**），前端优先用它、
老录像回落"与 `arena.TileGrid.towers` 同值"的常量；并用**可执行对账**把这条边钉死
（`scripts/check_dashboard_js.py --tower-only`：Python 三方对账 7 PASS + node 真绘制路径 7 PASS）。

## §1 现象与根因

### §1.1 现象（可量化的那份）

`drawInterpOn()` 里那三行：

```js
const s = (tw.king ? 0.62 : 0.50) * scale;   // ← 半边长（格）
const bx = X(tw.x) - s, by = Y(tw.y) - s, bs = s * 2;
ctx.strokeRect(bx, by, bs, bs);
```

`scale = min(W/18, H/32)` = 每格像素；页面里 `<canvas id="arena" width="450" height="800">` ⇒ `scale = 25 px/格`。
所以修前：公主塔 **1.00 格 = 25 px** 见方、王塔 **1.24 格 = 31 px** 见方。
修后（本引擎几何）：公主塔 **3.00 格 = 75 px**、王塔 **4.00 格 = 100 px**。

### §1.2 根因 = 两套几何，且前端那份**不随引擎更新**

| 时间 | 事件 | 塔的几何 |
|---|---|---|
| 2026-09-03 `b58db45`（项目上传） | dashboard 内联页首次入库，塔尺寸 = `0.62/0.50` 格 | 前端写死 **1 格**（**与当时引擎也不一致**） |
| 2026-09-09 `f5e7c9c` | **引擎**塔几何：公主塔 3×3 / 王塔 4×4，告别"全员圆形" | 引擎 = 1.5/2.0（半宽） |
| 2026-09-19 `e376e72` | Tier 2 把内联页**逐字**搬到 `rl/dashboard_html.py` | 前端仍 **1 格**（搬家不改值） |
| 2026-09-19（本次） | 用户观测到"回放里的塔还是 1 格" | 二者对齐 |

**⚠️ 必须如实写（【R10】）**：这不是"我们这次改坏的"，前端那个常数从**项目上传那天**就是 1 格；
引擎 2026-09-09 改矩形后**没有任何东西**把前端拉回去（前端是纯字符串里的常量，引擎侧无从知道它）。
也没有"曾经正确过"的历史 —— 所以"恢复到正常尺寸"实际含义 = **回到与引擎一致**。

### §1.3 为什么原先没人抓到

`scripts/check_dashboard_js.py` 当时只做**语法 + 渲染冒烟 + 分支覆盖**：它证明"不抛异常"，
不证明"画得对"。塔尺寸写错不会抛任何异常 ⇒ 三层检查全绿。**本次补的正是这条空缺**（§4.1）。

## §2 为什么记在 `meta`（局级）而不是逐帧

用**真实录像**量的体积代价（`pickle` 默认协议，逐帧新建对象、不吃 `pickle` 的 memo 红利）：

| 样本 | 帧数 | 原大小 | 每帧加 `tower_geom`（6 塔 × 5 数） | 增幅 |
|---|---|---|---|---|
| `runs/skarmy_probe_cost8/replays/league_1800.pkl` | 1,800 | 1.67 MB | 2.14 MB | **+27.73%** |
| `runs/run100k/replays/league_100000.pkl` | 19,893（100 局） | 21.9 MB | 27.1 MB | **+23.39%** |

而塔几何是**每局常量** ⇒ 记在**局级 `meta["tower_geom"]`**（一局一次），体积代价 ≈ 0。
（若把同一个 list 对象复用给所有帧，`pickle` 的 memo 会让增幅看起来只有 0.45% —— 那是**假读数**，已排除。）

## §3 改法（4 处代码 + 1 条对账）

| 位置 | 改动 |
|---|---|
| `src/clasher_new/rl/replay.py` | 新增 `tower_geometry(battle)` → `[[cx, cy, hw, hh, player], ...]`；**只读** `battle.arena.towers`，**两种表形状都认**：4 元组（本仓矩形口径）与 3 元组（上游圆形口径 `hw=hh=r`）；形状不认识/空 ⇒ `None`（前端回落，绝不抛）。文档写进 `LEAGUE_REPLAY_SCHEMA` 注释块：**schema 5.x 可选局级键，不升 schema** |
| `src/clasher_new/rl/run_league.py` | `LeagueGameRecorder.record()` 的**首帧**把 `tower_geometry(env.battle)` 写进 `self.meta["tower_geom"]`（幂等：`"tower_geom" not in self.meta`）⇒ 所有走 recorder 的生产路径（训练联赛 / 骷髅探针 / 随机评估）自动带上 |
| `src/clasher_new/rl/dashboard_html.py` | 新增常量 `TOWER_GEOM`（princess 1.5/1.5、king 2.0/2.0，注释里写明**唯一来源 = `arena.py`** 且被对账测试守着）；`computeTowerMax()` 里把 `curGame.meta.tower_geom`（或帧字段）解析成 `"x,y"(1 位小数) → {hw,hh}`；绘制改**矩形**（`bw = hw*2*scale`、`bh = hh*2*scale`，血条/灰×同步），回落 `TOWER_GEOM[kind]` |
| `scripts/skarmy_probe_upstream.py` | 上游探针把**上游自己的**几何（3 元组 r=1.0/1.4）经同一个 `tower_geometry()` 归一后写进该局 `meta` ⇒ dashboard 不会拿本仓的 3×3/4×4 **冒充**上游 |
| `scripts/check_dashboard_js.py` | 新增第 4 层"塔几何"：Python **三方对账** + node **真绘制路径实宽**；新增 `--tower-only`（不需要 run 目录） |

**优先级**（前端）：`meta.tower_geom` → 帧字段 `tower_geom` → `TOWER_GEOM` 常量。
命中方式 = 按 **(x, y)** 查表（与录像里 `round(x,1)` 同精度）⇒ **与塔顺序无关**，
几何里多了/少了塔、或位置对不上（陈旧 meta）都不会画错（回落常量）。

## §4 验证（全部可复跑）

### §4.1 闸门本身

```
cd src/clasher_new && python3 ../../scripts/check_dashboard_js.py --tower-only
```
实测（**Python 侧 7 PASS + node 侧 7 PASS，`FAILURES=0`，rc=0**）：

```
PASS 引擎塔几何 = 公主塔 3×3 / 国王塔 4×4
PASS 前端 TOWER_GEOM == 引擎 arena.TileGrid.towers
PASS 前端竞技场尺寸 == 引擎 TileGrid.width/height      （18×32 也是前端写死的，一并钉住）
PASS 前端塔位缺省 == 引擎塔中心                        （towerPos 缺省位同样写死，一并钉住）
PASS tower_geometry(真 TileGrid) = 6 座且半宽同引擎
PASS tower_geometry 认上游 3 元组（圆形 r ⇒ hw=hh=r）
PASS tower_geometry 形状不认识/空 ⇒ None（不抛）
PASS ① 老录像（无 meta.tower_geom）⇒ 回落引擎几何：公主塔 3 格 / 王塔 4 格
PASS ② meta.tower_geom（本仓 4 元组）⇒ 逐塔按引擎几何
PASS ③ meta.tower_geom（上游 3 元组 r=1.0/1.4）⇒ 按上游尺寸 2 格 / 2.8 格
PASS ④ 几何按位置命中（顺序打乱无影响）
PASS ⑤ 几何与本局塔位不匹配（陈旧/异图 meta）⇒ 该塔回落常量，不崩
PASS ⑥ 残缺几何项（长度 < 4）被忽略而不是画错
PASS ⑦ 塔心不动（放大后仍以引擎塔位为中心）
```

node 侧量的是 **`strokeRect` 的实参像素**（走 `computeTowerMax()` → `drawInterpOn()` 真路径），
不是读常量 ⇒ "常量改对了但绘制还在用旧变量"这类错也会被抓到。

**既有三层未回归**（同一脚本全量）：
`python ../../scripts/check_dashboard_js.py --run runs/nostall20k --league-run runs/run100k` ⇒ **rc=0**
（solo 15 PASS / 联赛 5 PASS / 塔几何 14 PASS）。

### §4.2 负对照（**闸门真的会红**）

把常量临时改回旧值 `princess: {hw: 0.5, hh: 0.5}`（= 本次要修的 bug）再跑：

```
[dashboard-js] FAIL 前端 TOWER_GEOM == 引擎 arena.TileGrid.towers :: js={...,'princess': (0.5, 0.5)...}
FAIL ① ... 公主塔应有 4 座 75px，实测 {"25x25":4,"100x100":2}      ← 25px = 1 格，正是用户看到的
FAIL ⑤ ... 应全部回落常量: {"25x25":4,"100x100":2}
FAIL ⑥ ... {"25x25":1,"75x75":3,"100x100":2}
FAILURES=3      rc=1
```
随后**已还原**（`git diff` 只留本次改动）。

### §4.3 端到端：本引擎**生产路径**真的写出几何

临时 run 目录跑真探针（40 步、3 局）：
`python scripts/skarmy_probe.py --out <tmp> --max-steps 40` ⇒ 产物每局
`meta.tower_geom = [[3.5,6.5,1.5,1.5,0], [14.5,6.5,1.5,1.5,0], [9,3,2,2,0], [3.5,25.5,1.5,1.5,1], [14.5,25.5,1.5,1.5,1], [9,29,2,2,1]]`
（3/3 局都有；探针 `--out` 用相对路径会落在 `src/clasher_new/src/...`（既有 cwd 契约坑），已清理）。

### §4.4 上游录像：**帧逐值相同**，只多一个键

重跑上游探针（`/usr/bin/python3` + `PYTHONPATH=/tmp/deps_upstream`，3 局 / 1,800 帧）
→ `scripts/convert_upstream_frames.py` → 与线上那份逐项对比：

| 对比项 | 结果 |
|---|---|
| 帧（全部 1,800 帧逐值） | **相同 `True`** |
| `winner`（3 局） | 相同 |
| `solo_state.json` | 相同 |
| `meta` 键 | 旧 4 键 → 新 5 键（**只多** `tower_geom`） |
| `meta.tower_geom` | 旧 `None` → 新 `[[3.5,6.5,1.0,1.0,0], …, [9,29,1.4,1.4,1]]` |

⇒ 覆盖线上文件**不引入任何新事实**，只补上"上游塔是圆形 r=1.0/1.4"这一个**已知几何**。
备份：`/tmp/up_8701_replay_backup.pkl`、`/tmp/up_8701_solo_backup.json`。

### §4.5 dashboard 已重启并核验（**必须重启，否则看不到**）

`_HTML` 是**模块级常量**，运行中的进程**不会**因为文件改动而重新读它 ⇒ 改完必须重启
（旧进程 = 旧 HTML）。本次：停掉 8700/8701/8702/8703 共 4 个进程（8 个 PID），按**原参数**重启
（后台任务 `bash-104`…`bash-107`）。核验（Windows python 经 127.0.0.1 实取）：

| 端口 | run 目录 | HTML 含 `TOWER_GEOM` | 旧 `0.62 : 0.50` | `/api/replay` 的 `meta.tower_geom` |
|---|---|---|---|---|
| 8700 | `skarmy_probe`（O3 前） | ✓ | 已消失 | `None` ⇒ **回落 3×3/4×4** |
| 8701 | `skarmy_probe_upstream` | ✓ | 已消失 | **上游 2×2 / 2.8×2.8** |
| 8702 | `skarmy_probe_nolane`（消融） | ✓ | 已消失 | `None` ⇒ 回落 3×3/4×4 |
| 8703 | `skarmy_probe_cost8`（O3 后） | ✓ | 已消失 | `None` ⇒ 回落 3×3/4×4 |

### §4.6 selftest（**不新增测试**，只在既有测试里加强断言）

`run_selftests.py test_dashboard_replays test_league_replays` ⇒ **2/2 通过**。
`test_dashboard_replays` 内新增断言：4 元组/3 元组/异常形状/空/`None` 五种输入；
**真引擎几何** 6 座且 4 座 (1.5,1.5) + 2 座 (2.0,2.0) 且中心与 `TileGrid.towers` 逐值一致；
`meta.tower_geom` 经 `save/load` **原样透传**给前端。

> 为什么不加进 `rl/selftest.py`：`main()` 那 100 行调用清单是**外部契约**（`_apply_s2_channel_when_idle.sh`
> 等按名调用），为一条渲染回归去动它、并让"全量 100 通过"的既有文档全部漂移，**不划算**。
> 前端回归的正确归宿就是前端自己的闸门 `scripts/check_dashboard_js.py`（已有 `--tower-only` 入口）。

## §5 未做与边界（【R10】：不确定/未做的都写下来）

1. **我方既有录像（8700/8702/8703）没有 `tower_geom` 键** ⇒ 它们走**回落常量**。
   对本引擎**结果正确**（常量 = 引擎当前几何），但**语义不同**：回放渲染的是"今天的引擎几何"，
   而不是"录那段像时的引擎几何"。⇒ **若引擎塔尺寸再次变化，老录像会跟着变**。已声明，未修。
   （不重录的理由：8700 是 **O3 修复前**的产物、8702 是 **`_lane_offset=0` 消融**，
   重录会破坏它们作为历史对照的意义；8703 重录也只是多一个键，无信息量。）
2. **上游塔是圆，我们用外接正方形画**（2×2）。这是沿用 2026-09-10 用户口径「圆形换方形」，
   与其它建筑/部队的方形口径一致；**不**为上游单独画圆。
3. **人机对战画布 `playArena` 与回放共用 `drawInterpOn`**：`towerGeom` 只在 `openGame()` →
   `computeTowerMax()` 时被设置 ⇒ 人机对战用**回落常量**（本引擎，正确）；但若同一页面里
   先打开过**别的引擎**的回放，`towerGeom` 会沿用（**与既有的 `towerPos`/`towerMax` 同一种陈旧耦合**，
   不是本次新引入的）。人机对战是实时流，未加逐帧几何。
4. **前端写死的三个常量（塔几何 / 塔位缺省 / 竞技场 18×32）** 与引擎的一致性靠**对账测试**保证，
   不是"物理上单一来源"（前端跑在浏览器里，拿不到引擎对象）。对账覆盖到 `TileGrid` 三处字段；
   **未**覆盖"塔数量变化""塔类型变化"（引擎若加塔，前端 `towerPos` 六个缺省槽位需人工跟进）。
5. **本次只改渲染**：`battle.py` / `arena.py` 的行为**一行未改**（§7 的文件清单可核）。

## §6 复现命令

```bash
# 1) 塔几何闸门（不需要 run 目录；Python 对账 + node 真路径）
cd src/clasher_new && PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe \
    ../../scripts/check_dashboard_js.py --tower-only

# 2) 既有全量前端闸门（三层 + 塔几何）
cd src/clasher_new && PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe \
    ../../scripts/check_dashboard_js.py --run runs/nostall20k --league-run runs/run100k

# 3) 相关子集 selftest【R19】
cd src/clasher_new && PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe \
    ../../scripts/run_selftests.py test_dashboard_replays test_league_replays

# 4) 上游录像重生成（可选；需 fastcore 依赖与上游仓库）
/usr/bin/python3 -m pip install --quiet --target /tmp/deps_upstream fastcore
cd /mnt/e/clash-royale-simulator-main-by-jason/src/clasher_new && \
  PYTHONPATH=/tmp/deps_upstream PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B \
  /mnt/e/clash-royale-simulator-main/scripts/skarmy_probe_upstream.py --out /tmp/skarmy_upstream.json
cd /mnt/e/clash-royale-simulator-main && python3 scripts/convert_upstream_frames.py \
  --json /tmp/skarmy_upstream.json --run-dir src/clasher_new/runs/skarmy_probe_upstream

# 5) dashboard 改完必须**重启**才生效（_HTML 是模块级常量）
cd src/clasher_new && PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe -u \
  rl/dashboard.py --solo runs/skarmy_probe --replays runs/skarmy_probe/replays \
  --host 127.0.0.1 --port 8700
```

## §7 本次改动的文件

| 文件 | 性质 |
|---|---|
| `src/clasher_new/rl/replay.py` | `tower_geometry()` + schema 5.x 可选键文档 |
| `src/clasher_new/rl/run_league.py` | recorder 首帧写 `meta.tower_geom` |
| `src/clasher_new/rl/dashboard_html.py` | JS：塔按引擎几何画（矩形）+ `meta` 优先 + 回落常量 |
| `scripts/skarmy_probe_upstream.py` | 上游探针写**上游自己的**几何 |
| `scripts/check_dashboard_js.py` | 第 4 层：三方对账 + 真绘制路径实宽 + `--tower-only` |
| `src/clasher_new/rl/selftests/part2.py` | `test_dashboard_replays` 内**加强**断言（不新增测试） |
| `src/clasher_new/runs/skarmy_probe_upstream/replays/league_1800.pkl`（gitignored） | 重生成：**帧逐值相同** + 多 `meta.tower_geom` |
| `AGENTS.md` / `docs/README.md` | 一行指针（【R18】） |

**未改**：`arena.py`、`battle.py`、`pathfinding_heap.py`、任何训练/奖励逻辑。
