# 骷髅军团定点投放探针：两台引擎并列（2026-09-19，用户指定）

> **用户指定**：「改用骷髅军团做测试，分别在国王塔后、桥头放置骷髅军团，我来观测效果，观测方式复用 dashboard」→ 追加「按照上游引擎再跑一遍」。
> **性质**：这是**观测工具 + 一次读数**，**不是判决**。判据**没有预注册**（【R3】）⇒ 本文只写"观察到什么"，**不写"谁更像真机"**。
> **只读/低风险**：两个探针都只写自己的 `runs/<name>/`（gitignored）；未改引擎、未跑训练。

---

## §1 交付物

| 物 | 路径 | 说明 |
|---|---|---|
| 我方探针 | `scripts/skarmy_probe.py` | 我方引擎，走生产同一录像路径（`LeagueGameRecorder` → `save_league_replays`，**schema 5**） |
| 上游探针 | `scripts/skarmy_probe_upstream.py` | 上游引擎（`by-jason` @ `f616f19`），吐**中性 JSON**（不依赖我们任何格式） |
| 转换器 | `scripts/convert_upstream_frames.py` | 上游 JSON → **我们 schema 5 录像** + `solo_state.json` ⇒ dashboard 能播 |
| 并列读数 | `scripts/skarmy_probe_compare.py` | 读两份录像，逐秒并排列出存活数 / 平均 y / x 区间 |

**两个 dashboard（都已起，互不干扰）**：

| 端口 | 指向 | 内容 |
|---|---|---|
| **8700** | `runs/skarmy_probe` | **我方**引擎 3 局（`league_1800.pkl`） |
| **8701** | `runs/skarmy_probe_upstream` | **上游**引擎 3 局（转成 schema 5 后的 `league_1800.pkl`） |

旧的 dashboard 已按用户要求**全部停掉**（8700 的 `et_ctrl100k`、8701 的 `rand100_eval`，实测端口 curl 回到 `000`）。

---

## §2 方法（两台引擎**逐项对齐**，只有引擎不同）

| 项 | 值 |
|---|---|
| 投放点 | `behind_king` =(9.5, 0.5)（蓝王塔 y∈[1,5]、x∈[7,11] 的**后方那一行**）／`bridge_left` =(3.5,13.5)／`bridge_right` =(14.5,13.5) |
| 牌 | `SkeletonArmy`（`spawn_number=15`，即 15 只骷髅） |
| 对手 | **静音**（我方 `ActionBundle([])`；上游不发牌） |
| 采样 | 每帧推进 **6 个引擎 tick**，`dt = 1/60` ⇒ **0.1 s/帧**、600 帧 = 60 s |
| ⚠️ tick 口径 | 上游**生产**用 1/20（`environment.py` 的 `self.fps=20`），本文两边**都用 1/60** ⇒ 隔离掉"tick 粒度"这一个变量，只比**引擎代码** |
| 录像 | 我方原生 schema 5；上游经转换器写成 schema 5（帧字段逐项同构） |

**上游的运行姿势**（本次实测唯一缺件 = `fastcore`，且它仍要求 cwd = `src/clasher_new`）：

```bash
/usr/bin/python3 -m pip install --quiet --target /tmp/deps_upstream fastcore
cd /mnt/e/clash-royale-simulator-main-by-jason/src/clasher_new
PYTHONPATH=/tmp/deps_upstream PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B \
    /mnt/e/clash-royale-simulator-main/scripts/skarmy_probe_upstream.py --out /tmp/skarmy_upstream.json
# 转成我们的录像（转换器只依赖 stdlib + rl.replay，不需要 torch）
cd /mnt/e/clash-royale-simulator-main
/usr/bin/python3 scripts/convert_upstream_frames.py --json /tmp/skarmy_upstream.json \
    --run-dir src/clasher_new/runs/skarmy_probe_upstream
```

**两局的可比性前置（已核对）**：两边首帧塔血都是 `4824 / 3052 / 3052`（= lv11 锚点，与我们引擎 `card_level=11` 一致）；
两边都成功生成 **15** 只；两边 `invalid_count`/`deploy` 均正常。

---

## §3 读数（`scripts/skarmy_probe_compare.py` + 河带内专项统计）

### 3.1 桥头投放（差异最明显）

| 投放点 | 指标 | 我方 | 上游 |
|---|---|---|---|
| `bridge_left` | 河带 `y∈[15,17]` 内出现的 x 区间 | **1.80 ~ 5.00**（n=222 个观测） | **2.40 ~ 4.70**（n=280） |
| `bridge_left` | 河带内**越出桥廊**（`x<2` 或 `x>5`）的帧数 | **2 / 56** | **0 / 80** |
| `bridge_right` | 河带内 x 区间 | **12.80 ~ 16.00** | **13.40 ~ 15.70** |
| `bridge_right` | 河带内越出桥廊（`x<13` 或 `x>16`）的帧数 | **2 / 56** | **0 / 80** |
| 两者 | 到 5.1 s 时的平均 y | **18.7** | **16.8** |
| 两者 | 到 10.1 s 时的平均 y | **23.8** | **22.1** |

⇒ **两条可复述的观察**（描述性）：
1. **过河速度**：我方骷髅在同样 10 s 内推进得更远（mean y 快 **1.7 格**左右）——与"我们桥面代价 `'.'=5`、上游 `'.'=8`"（`pathfinding_heap.py`）方向一致；
2. **横向散度**：我方在河带内的 x 跨度 **3.2 格**（1.8~5.0）、上游只有 **2.3 格**（2.4~4.7）；
   并且我方有 **2/56 帧**存在个体**在桥廊之外**（`x<2` 或 `x>5`，即几何上的河面上），上游 **0/80**。

⚠️ 第 2 条的机制**未定**【R10】：可及的解释至少有三个 —— ① 我们的**队形车道偏移** `_lane_offset`（`battle.py:1277-1287`，上游没有）；
② 我们的**卡死自救**会丢弃路点（`battle.py:1248-1263`，上游没有）；③ `collision` 推挤发生在 `ensure_walkability` **之后**（`battle.py:2881-2884`）⇒ 被撞出桥面的那一帧到下一 tick 才被拉回。**本轮不做归因。**

### 3.2 王塔后投放（差异很小）

| 指标 | 我方 | 上游 |
|---|---|---|
| 河带内 x 区间 | 3.90 ~ 14.40（n=192） | 3.70 ~ 14.30（n=189） |
| 河带内越出桥廊帧数 | **0 / 69** | **0 / 75** |
| 平均 y（10.1 s） | 12.4 | 11.4 |

⇒ 双腿分开、**左右两桥各走一半**的形态两边一致；差异只是我方**略快**。

### 3.3 三局的终局

三局都在 60 s 内全员阵亡（无对手，死于敌塔火力）；`winner` 由塔血合计的同一口径给出
（我方 `timeout_winner`，上游侧由转换脚本用同一规则算）⇒ `behind_king` 平、两个桥头局蓝方胜。**这是同一个口径，不是独立证据。**

---

## §4 未验证 / 不构成结论

1. **没有预注册判据** ⇒ 上表全部是**描述性读数**，**不得**写成"我们的寻路过桥更差/更好"。
2. **tick 口径**：两边都用 1/60，而上游生产是 1/20 ⇒ 本文**不代表**"上游按自己的 20 Hz 跑"的结果。
3. **n=1 局/投放点**（【R5】）⇒ 只看**形态**，不看小差异。
4. **上游录像是我方转换器写的**（帧字段逐项同构、数值逐字来自上游快照），但转换器本身**未经独立对账**（没有任何地方能逐位验证它与上游内部状态一致）⇒ 这是本文最弱的环节。
5. `runs/` 不入库 ⇒ 本文的 pkl 只在本机；重跑脚本在 §2（上游侧需先装 `fastcore`）。
6. **P3（jump 闩锁）/ P2（桥面 sliver）** 在本探针里**没有被触发**（没有 jumper；sliver 是否被踩到未统计）⇒ 本探针**不构成**对那两条的验证。
