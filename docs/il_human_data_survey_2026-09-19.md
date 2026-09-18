# CR 领域人类对局数据 & 开源 IL/BC 项目 尽职调查（2026-09-19）

> **执行者**：调研子智能体（只陈述可核实事实；每条附 URL 与取数时间）
> **取数时间**：2026-09-18 20:00–20:30 UTC（本机时钟）；下文统一记作「2026-09-19 取数」
> **环境限制**：`web_search` 报 HTTP 402 不可用，改用 `curl` + `web_fetch` 直取。
> **网络可达性实测**（这决定了取证手段，也构成部分「未核实」的原因）：
> - ✅ 可达：`github.com`(git clone)、`api.github.com`(60 req/h)、`cdn.jsdelivr.net`、`www.kaggle.com`、
>   `hf-mirror.com`（HuggingFace 镜像）、`developer.clashroyale.com`、`export.arxiv.org`、`pypi.org`、`www.bing.com`/`www.ecosia.org`（搜索质量差，均被地域重定向到「Clash 代理客户端」）
> - ❌ 不可达（HTTP 000）：`huggingface.co`、`datasets-server.huggingface.co`、`raw.githubusercontent.com`、
>   `web.archive.org`、`html.duckduckgo.com`、`lite.duckduckgo.com`、`www.mojeek.com`、`search.brave.com`
> - ⚠️ 因此 HuggingFace 数据用 `hf-mirror.com` 取证（API 语义与 HF 一致，`/api/datasets/{id}`、`/tree/`、`/resolve/` 均可用）；
>   官方 API 文档站是 SPA + 登录墙，正文**未能直取**

---

## §1 人类回放数据源

### 1.1 一句话结论

**没有任何公开源提供「逐帧完整状态 + 人类动作」的配对语料**（标准 IL/BC 语料）。
公开可得的最好形态是**动作侧**：`(card, x, y, side, tick@20Hz)`，来自 **RoyaleAPI 网站的 `/data/replay` 时间线**
（不是官方 API）。**圣水、单位 HP、随时间的位置轨迹全部缺失**——而这三者正是「攒费→打 Xbow」的状态前提。

### 1.2 数据源表

| # | 名称 | URL | 提供方 | 许可 | 数据量 | **逐帧动作-时间戳** | 逐帧**状态** | 可得性 |
|---|---|---|---|---|---|---|---|---|
| 1 | 官方 Clash Royale API `battlelog` | https://developer.clashroyale.com/ ・ `https://api.clashroyale.com/v1/players/{tag}/battlelog` | Supercell | Supercell API ToS | 每玩家最近对局摘要 | **❌ 否**（间接证据见 1.3） | ❌ | 需 developer key + **IP 白名单**；文档正文登录墙 |
| 2 | RoyaleAPI 回放端点 `/data/replay?tag=…` | https://royaleapi.com/data/replay | RoyaleAPI（第三方） | 受 Supercell + RoyaleAPI ToS 约束；数据集本身多标 MIT | 见 #3/#4 | **✅ 是**（20 Hz tick + 坐标 + 卡牌 + 边） | **❌ 否**（详见 1.4） | 公开网页，但有 Cloudflare；批量采集靠浏览器扩展抓响应 |
| 3 | HF `VanguardX101/IL_Replay`（**FirstLight_CR 用的那个**） | https://hf-mirror.com/datasets/VanguardX101/IL_Replay | VanguardX101 | **未声明**（`croissant.license = None`，tags 里无 license） | **252,238 回放 / 17,836,160 动作**；parquet **1,888,381,137 B ≈ 1.89 GB**（`replays` 表 52 分片 = 825,394,311 B ≈ 0.83 GB） | **✅ 是**（实测 schema 见 1.5） | ❌（仅终局塔血 + 全场聚合圣水） | 公开、**非 gated**、无需 token |
| 4 | HF `Cochon123/clash-royale-replays` | https://hf-mirror.com/datasets/Cochon123/clash-royale-replays | Cochon123 | MIT（README 自注受 Supercell/RoyaleAPI ToS 约束，仅限研究/个人） | 15,420 个 `/data/replay` 原始 JSON；cleaned-legacy：**1,592 唯一对局 / 120,416 出牌事件 / 4,677 ability** | **✅ 是**（实测 schema 见 1.5） | ❌ | 公开、非 gated |
| 5 | HF `chrisrca/clash-royale-tv-replays` | https://hf-mirror.com/datasets/chrisrca/clash-royale-tv-replays | chrisrca | MIT | ~10 fps TV Royale 帧，覆盖 31 个 arena | **❌ 否**（schema 仅 `frame_id`/`image`/`hash`） | ❌（纯像素，需自跑 CV） | 公开、非 gated |
| 6 | HF `Chutdedededede/clash-royale-tv-replays` | https://hf-mirror.com/datasets/Chutdedededede/clash-royale-tv-replays | Chutdedededede | MIT | 文件清单与 #5 **逐名相同**（疑似镜像） | ❌ 同 #5 | ❌ | 公开 |
| 7 | Kaggle `s1m0n38/clash-royale-games` | https://www.kaggle.com/datasets/s1m0n38/clash-royale-games | s1m0n38 | CC BY-NC 4.0 | **481M matches**，18.4 GB | **❌ 否** | ❌ | 公开 |
| 8 | Kaggle `bwandowando/clash-royale-season-18-dec-0320-dataset` | https://www.kaggle.com/datasets/bwandowando/clash-royale-season-18-dec-0320-dataset | BwandoWando | CC BY-NC-SA 4.0 | **37.9M matches**，5.4 GB | **❌ 否** | ❌ | 公开 |
| 9 | Kaggle `jackmangione/clash-royale-matchups-june2026` | https://www.kaggle.com/datasets/jackmangione/clash-royale-matchups-june2026 | jackmangione | CC BY-NC-SA 4.0 | ~170M matchups，6.1 GB | ❌ | ❌ | 公开 |
| 10 | Kaggle `nonrice/clash-royale-battles-upper-ladder-december-2021` | https://www.kaggle.com/datasets/nonrice/clash-royale-battles-upper-ladder-december-2021 | nonrice | CC BY-SA 4.0 | 11.6 MB（**我实际下载并读了列名**） | **❌ 否** | ❌ | 公开 |
| 11 | Kaggle `pepe93/my-clash-royale-ladder-battles` | https://www.kaggle.com/datasets/pepe93/my-clash-royale-ladder-battles | pepe93 | CC BY-NC-SA 4.0 | 29 KB（**我实际下载并读了列名**） | **❌ 否** | ❌ | 公开 |
| 12 | HF `Grandediw/clash-royale-battle` / `raymond9326/clash-royale-battles` / `CrabGuyy/ClashRoyaleTrophyMatches` / `lillyem/clash-royale-data` | https://hf-mirror.com/datasets?search=clash+royale | 各自 | 部分 MIT / 未声明 | 1M<n<10M 行（#12a）等 | **未核实**（仅取到 card 层） | 未核实 | 公开 |
| 13 | GitHub `wty-yy/Clash-Royale-Replay-Dataset` | https://github.com/wty-yy/Clash-Royale-Replay-Dataset | wty-yy | **无许可文件** | 8 stars / 0 forks | 未核实（**README 取不到**，见 §6） | 未核实 | 公开仓，但 clone 失败、jsDelivr 404 |
| 14 | GitHub `wty-yy/Clash-Royale-Detection-Dataset` | https://github.com/wty-yy/Clash-Royale-Detection-Dataset | wty-yy | MIT | 56 stars；含 1,388 张人工标注真实帧 | ❌（视觉检测集） | ❌ | 公开 |

**「不存在」条目**：**公开的「逐帧完整对局状态（圣水/单位 HP/位置轨迹）+ 人类动作」配对数据集 = 不存在。**
检索范围：HF datasets API（`search=` `clash royale` / `royale` / `replay` / `clash` / `firstlight` / `IL_Replay`，
共 5 组查询，2026-09-19 取数）；Kaggle datasets API（`search=` `clash royale` / `royale` / `clash royale matches`）；
GitHub repository search（`clash royale offline rl` / `demonstration` / `replay learning` / `imitation` /
`behavior cloning` / `dataset` / `yolo` / `cv bot` 等 14 组）；arXiv 全库 `all:"Clash Royale"`（4 条命中）。

### 1.3 官方 API：**不含回放/逐帧**（间接证据，但两条独立）

官方文档正文需登录（`https://developer.clashroyale.com/api/documentation` → `401 {"error":"session-not-found"}`，
2026-09-19 实测），故用**由该 API 派生的公开数据集的真实字段**反证：

1. **我实际下载并读了列名**（`nonrice/...` 数据集，11.6 MB）：
   列 = `p1card1..p1card8, p2card1..p2card8, p1trophies, p2trophies, outcome`。
   ⇒ 只有卡组 / 奖杯 / 胜负，**无圣水、无坐标、无 tick、无回放体**。
2. **我实际下载并读了列名**（`pepe93/...` 数据集，29 KB）：
   列含 `my_result, my_score, opponent_score, my_trophies, match_type, my_deck_elixir, op_...`（40+ 卡牌 one-hot）。
   ⚠️ 注意 `match_type` 的取值是 **`replay__ladderBattleType`** —— 这**只是官方 API 的对局类型枚举名**，
   **不代表返回了回放数据**；该行其余字段全是**对局级聚合**，没有任何逐帧字段。
3. 官方 API 端点探针（2026-09-19）：`/v1/players/{tag}/battlelog` 与 `/v1/cards` 均返回
   `403 {"reason":"accessDenied","message":"Missing authorization"}`。
   ⚠️ **此探针不能证明路径存在**（所有路径都先鉴权失败），仅证明**必须有 developer key**。

**2019 年前后的 API 变更**：**未能核实具体日期与细节**（原始公告需要 RoyaleAPI 博客/官方公告，
`royaleapi.com` 与 `web.archive.org` 在本环境均不可达 403/000）。可核实的相邻事实是：
`RoyaleAPI/cr-api` 仓库（171 stars，**已 archived**，最后 push 2020-10-25）README 原文：
> "Clash Royale Analytics, Profiles and Insights. **We no longer publish a public API. Please use the official API from Supercell.**"
> —— https://github.com/RoyaleAPI/cr-api（元数据经 `api.github.com/repos/RoyaleAPI/cr-api`，2026-09-19）
且其旧文档 `RoyaleAPI/cr-api-docs@master:docs/endpoints/player_battles.md` 记录的
`GET https://api.royaleapi.com/player/:tag/battles` 响应字段为
`type / mode / winCountBefore / utcTime / deckType / teamSize / winner / teamCrowns / opponentCrowns / team[] / opponent[]`
（team[] 内为 `tag/name/crownsEarned/startTrophies/clan/deckLink/deck[]`）——**同样没有回放**。
⇒ 可核实的是「RoyaleAPI 自建 API 已停、转向官方 API」，**但不能据此断定 2019 年那次变更的具体内容**。

### 1.4 RoyaleAPI `/data/replay` 里到底有什么（**这是本次最关键的取证**）

我下载了 `Cochon123/clash-royale-replays` 的一个**原始 payload**（`raw/00/000YYC2UJPVG.json`，58,520 B）并解析：

- 结构：`{schema_version, captured_at, request_url, referrer_url, payload:{success, html}}`，`html` 长 56,238 B
- HTML 内：`data-t`(tick) × 99、`data-x` × 49、`marker` × 64 —— 即时间线事件 + 地图标记
- 「elixir」出现 27 次，但**几乎全是 UI 图标/倍率标记**（`class="elixir_icon"`、「2:00 2x」）
- `class="...replay_elixir_table"` 里只有**全场聚合**：如 `Leaked 6.41`
- ⚠️ **`class` 白名单里没有任何 per-tick 状态容器**（`replay_stats` 下是 `card_count`，非时间序列）

**`VanguardX101/IL_Replay` 的 `payload_json` 实测结构**（我用 `pyarrow` 解 parquet，见 1.5）：

```
battle.{battle_type, game_mode, result, team/opponent:{crowns,
        players[0]:{average_elixir, deck[8], elixir_leaked, final_tower_hitpoints, four_card_cycle_elixir, tower_card}}}
events[37] = {card_key, card_play_number, coordinates{arena_label_from_blue, display_royaleapi_units,
              grid_cell_floor, native_world_units, raw_royaleapi_units}, kind, replay_tick_20hz,
              side, source_fields{data_c,data_i,data_s,data_t,data_x,data_y}, time_seconds, ...}
replay.{aggregate_stats.{team,opponent}:{ability,building,spell,troop,total:[count, elixir], leaked},
        card_counts, duration}
```

**逐项判定「有没有逐帧状态」**：

| IL 需要的量 | 是否可得 | 证据 |
|---|---|---|
| 动作卡牌 + 落点 + 时间戳 | **✅ 有** | `events[].{card_key, native_x, native_y, grid_x, grid_y, replay_tick_20hz, time_seconds, side}`；tick/20 = 秒 |
| 双方卡组/卡等 | **✅ 有** | `battle.*.players[0].deck[8]`（含 `level`） |
| 塔血随时间 | **❌ 无** | 只有 `final_tower_hitpoints`（**终局**快照，现场实测 team `{king:7728, princess_left:4239, princess_right:4802, total:16769}` / opp 全 0） |
| 圣水随时间 | **❌ 无** | 只有**全场聚合** `aggregate_stats.{team}[total]` 形如 `[15, 68]` = (张数, 总圣水)；`leaked` 全场值（team 1.04 / opp 2.75）。**没有逐事件圣水** ⇒ 无法直接条件化「圣水≥6」 |
| 单位 HP / 位置轨迹 | **❌ 无** | `"hp"` 在 payload 中出现 **0 次**；只有部署瞬间坐标 |
| 塔卡（`tower_card`）/平均圣水/四卡循环圣水 | ✅ 有 | 全场常数 |

作者自述（`Cochon123/clash-royale-replays` README「Limitations」原文）：
> "Replay actions do not include live unit HP, positions over time, or projectile state—**only discrete play/ability events**."

### 1.5 实测 schema（我下载 parquet 后用 pyarrow 读取，2026-09-19）

`VanguardX101/IL_Replay` → `actions/part-000000.parquet`（22,697,255 B，384,275 行）：

```
schema_version: string        replay_tag: string        event_index: int32
source_index: int32           kind: string              side: string
card_key: string              card_play_number: int32   replay_tick_20hz: int32
time_seconds: double          form_at_play: string      ability_source_authoritative: bool
native_x: int32  native_y: int32  grid_x: int32  grid_y: int32
arena_column: string          arena_row: int32
deck_card_key_candidates: list<string>   ability_source_candidates: list<string>
event_json: large_string
```
样例行：`{"card_key":"guards","side":"opponent","replay_tick_20hz":216,"time_seconds":10.8,
"native_x":16501,"native_y":30499,"grid_x":16,"grid_y":30,"arena_column":"Q","arena_row":31}`

`VanguardX101/IL_Replay` → `replays/part-000000.parquet`（17,432,059 B，5,000 行，每行 1 局）：

```
schema_version, replay_tag, battle_type, game_mode, result,
team_crowns:int16, opponent_crowns:int16, event_count:int32, warning_count:int32,
payload_json:large_string, requested_player_tag:string
```

`Cochon123/clash-royale-replays` → `cleaned-legacy/events.jsonl`（我读前 12 行）：
```
{"battle_id":"000PV2P00YL9","sequence":0,"ticks":184,"seconds":9.2,"side":"opponent",
 "event_type":"card_play","card":"skeletons","x":9500,"y":31499,
 "ability_card":null,"attribution":null,"ability_icon":null}
```

**结论（对 IL 唯一要紧的一条）**：两个最大的公开人类回放集**都只有动作侧 + 20 Hz 时间戳 + 坐标**；
**状态侧（尤其圣水）必须靠模型反推**（起始 5 圣水 + 固定回费 − 卡费，全场 `leaked` 只能做整体校正），
**反推误差没有任何一方标定过**。

---

## §2 开源 CR IL/BC 项目

> stars / 许可 / 最后提交均经 `https://api.github.com/repos/{full_name}` 取数，**取数时间 2026-09-19**。
> 类别：(a) 视觉/内存读取 bot（只读画面或内存）・(b) 基于模拟器/真引擎的策略学习・(c) 其他

| repo | stars | 许可 | 最后提交 | 类别 | 方法 | 奖励 / 损失 | 用了人类数据? | 可复现性 |
|---|---|---|---|---|---|---|---|---|
| **`Jaasssoooonnnnn/FirstLight_CR`** | **不可访问（GitHub API 404）** | — | ClashAI 记录 commit `28d66cc`(2026-09-06, "first commit") | (b) 借壳原生引擎（Null's Royale 私服 APK + ARM64 hook） | **IL 预热 → PPO** | IL loss：gate ACT class weight + delay 邻域平滑 NLL + **Huber value loss on discounted returns**；PPO reward：terminal outcome + tower-health shaping + elixir-overflow penalty +（specialist 1）Hog deploy / first-Hog timing shaping + **demonstration-BC 项**（`ppo_expert_bc.py`） | **✅ 252,238 回放 / 17.8M 动作**（= HF `VanguardX101/IL_Replay`，**我已独立核实**） | 极低：需自备 APK + root VM + 8 GPU + Slurm |
| **`vegetableleaf/ClashAI`** | 163 | **无许可文件** | 2026-09-18 | (b) 真引擎 re-drive + (c) 视觉 live | **BC 职业回放（S1）→ 引擎 DAgger + Gumbel 搜索教师（S3）→ live（S4）** | S1 = **监督**（pro placement 目标）；旧路 = win/loss PPO on imitation init，作者结论「**KL leash defended the init and bought nothing；without it the placement head railed**」；S3 搜索教师门禁**失败**（teacher 距 pro 9.478 格 vs student 3.34–3.48，exact cell 0.00 vs 21.9–23.9） | **✅ 1,253 icebow + 598 hogeq 职业回放（RoyaleAPI）+ HF `VanguardX101/IL_Replay`（603 局）** | 中高：`HANDOFF.md` 777 KB 逐条留证；但依赖非公开引擎运行时 |
| **`wty-yy/KataCR`** | 480（68 forks） | MIT | 2024-06-06 | (a)+(b) 非嵌入式 CV + **离线 RL** | `StARformer` / `Decision Transformer` **离线 RL**，数据由自采专家视频经 `replay_data/offline_data_builder.py` 做特征融合 | 我读了 `katacr/policy/perceptron/reward_builder.py`（OCR 塔血条）：公主塔**被毁 ±1**、国王塔**被毁 ±3**、塔伤 **±(ΔHP/full_HP)**、开国王塔 **−0.1**、**圣水溢出 −0.05 / 每 10 帧**；`MAX_DELTA_HP=1600` | ⚠️ 自采「专家」= 作者/对手录像（非公开人类数据集）；对外发布的是 `Clash-Royale-Replay-Dataset` | 高（中文文档详尽、有 eval 脚本）；权重/数据需另取 |
| **`weihaog1/The-Elixir-Optimizers`** | 0 | **无许可文件** | 2026-03-19 | (c) 视觉 live | **BC（3 头：play/card/position）→ MaskablePPO 微调** | `L_total = L_play + L_card + L_position`；`L_play` = 加权 BCE（正文一处写 play:noop = **10:1**，表格里写 **6:1**，**同一文档内不一致**）；`L_card` = 4-way CE（**只在动作帧上算**）；`L_position` = **576-way label-smoothed CE**（ε=0.1，FiLM 按 card_id 条件化）。PPO 终局：**Win/Loss ±3.0、Crowns ±1.0，全部 ×0.1** | **✅ 自采 40 局 / 5,366 帧 / ~1,600 动作帧**（`pynput`+`mss` 点击记录器） | 中：有完整报告与图表（`docs/final.md`），但**训练数据/权重未公开** |
| **`shawnxu0407/Clash_Royale_agent`** | 123 | MIT | 2025-08-28 | (c) 视觉 live | 专家数据集 → YOLOv8 状态 → ResNet 出牌检测 → cross-attention 策略 | **reward = 对方塔 HP 下降为正、否则为负**（`torch_reward_builder.py`，用 cnocr 读塔血） | ⚠️ 用 `scrcpy`+FFmpeg 采的 Expert 帧；README 写「later add the link for the dataset collection」⇒ **数据未发布** | 低：数据/权重缺失 |
| `hastylmao/Hasty-CR` | 33 | MIT | 2026-08-29 | (b) 自研整数精确模拟器（1 tile=1000 millitile，50 ms tick，1,635 tests） | PPO + self-play league，2,321 掩码动作 | reward 权重开关 `--crown --win --chip --elixir`（README 只给开关名，未给数值）；监督只看**行为**（hog share / plays per match / score drift）不看 loss | ❌ | 高（可装可跑）；但 README 自陈「**No trained policy has beaten a human**」「**This is the simulator marking its own homework**」 |
| `Jaso1024/Real-Time-Strategy-RL-Clash-Royale` | 50 | 无许可 | 2023-06-20 | (c) 视觉 live | PPO + 3 agent（出牌 / 48 格 / 9 格）+ autoencoder 编码 9 段输入 | 我读了 `CRBot.py`：`crowns_reward = round(4.96392*ln(4.86466*player_crowns+0.753851)+1.40353)`；**win +300.0 / loss −200.0 / draw 0**；`total = crowns_reward + end_reward` | ❌（无监督数据，纯在线 PPO） | 中（权重已提交 `TrainedWeights/`）；仅对训练营对手 |
| `Jason-XII/clash-royale-simulator` | 98（21 forks） | 无许可 | 2026-09-18 | (b) 模拟器（实时多人 + RL） | README 引 Supercell 论文 arXiv:2012.12186 说明「经验收集是瓶颈」 | 未核实（README 未给 reward 细节） | ❌ | 未核实 |
| `MSU-AI/clash-royale-gym` | 14 | MIT | 2024-03-13 | (b) gym 环境 | `obs, reward, terminated, _, info = env.step(action)` | README 未给 reward 定义 | ❌ | 低（README 2.2 KB，12 open issues） |
| `artumont/SICrMLB` | 0 | MIT | 2025-12-27 | (a)+(b) 非嵌入式 RL+CV | 自述 "**heavily inspired by KataCr**" | 未核实 | ❌ | 中（有 pytest、pyproject） |
| `chrisrca/CS541-Deep-Learning-Clash-Royale-Project`（分支 `emulation`） | 4 | MIT | 2025-12-12 | (a) 录制 + 模板匹配 | MuMu 模拟器 540×960 录帧 → `tv_royale_extractor.py`；`cr_detection/` 卡图模板 | 课程项目；**产出 HF `chrisrca/clash-royale-tv-replays`** | ⚠️ 录的是 TV Royale（人类）**帧**，但**无动作标注** | 高（步骤清晰） |
| **`Jason-XII/cr-memory-reader`** | 53 | 无许可 | 2026-08-25 | **(a) 内存读取** | Frida hook Null's Royale 私服内存 | **不是学习项目**：读圣水 / 游戏时间 / 手牌 / 皇家塔血量 / 场上单位，**30 Hz** | ❌ | 需 root + 私服 + frida-server |
| `Pbatch/ClashRoyaleBuildABot` | 376 | MIT | 2026-07-24 | (a) 视觉 bot **平台** | 对象检测器 + HP 检测器 + GUI 框架 | **无学习** | ❌ | 高 |
| `pyclashbot/py-clash-bot` | 327 | NOASSERTION | 2026-09-01 | 自动化脚本 | 屏幕自动化 | **无学习** | ❌ | 高 |
| `wty-yy/Clash-Royale-Replay-Dataset` | 8 | 无许可 | 2024-05-12 | 数据仓 | 描述 "Replays for training offline-rl policy"，作 KataCR `--replay-dataset .../golem_ai` | — | 未核实 | **README 取不到**（jsDelivr 404、clone 失败） |
| `wty-yy/Clash-Royale-Detection-Dataset` | 56 | MIT | 2024-06-11 | (a) 视觉数据集 | 生成式合成 + KataCR 的 **1,388 张人工标注真实帧** | — | ❌（无策略数据） | 高 |
| `AlexanderPotiagalov/ClashRoyaleML` | **未核实** | 未核实 | 描述称 2026-08-01 | (a) CV + ML | "pipeline for understanding Clash Royale gameplay from recorded ..." | 未核实 | 未核实 | 未核实（`api.github.com` 返回空响应） |
| `aarohkandy/CRAIzy` | 0 | 无许可 | 2026-07-28 | (c) | bot stack + Kaggle 训练管线 | 未核实 | 未核实 | 未核实 |
| `AngelFireLA/Clash-Royale-AI` | 4 | 无许可 | 2026-02-13 | (a) CV bot | 训练营画面控制 | 未核实 | ❌ | 未核实 |

### 2.1 关键发现

1. **真正做 IL/BC 的只有 4 个**：`FirstLight_CR`（**已 404**）、`ClashAI`、`The-Elixir-Optimizers`、`KataCR`（离线 RL 而非纯 BC）。
   其余全是**视觉/内存读取 bot**（`cr-memory-reader`、`ClashRoyaleBuildABot`、`py-clash-bot`、`Clash Royale AI` 系）
   或**自研模拟器 + 自对弈 RL**（`Hasty-CR`、`Jason-XII`、`MSU-AI`）。
2. **"视觉/内存读取 bot" 与 "基于模拟器的策略学习" 的分界很清楚**：
   - (a) 类（`cr-memory-reader` 53★、`ClashRoyaleBuildABot` 376★、`py-clash-bot` 327★、`cr-yolo` 系）**只提供状态，
     不学策略**，或只学低级动作；
   - (b)/(c) 类才学策略，但**只有 ClashAI 与 FirstLight 真用了大规模人类回放**。
3. **没有找到任何一个项目发布了「人类回放 + BC 权重」的完整可复现包**。
   `The-Elixir-Optimizers` 有完整报告但数据不公开；`ClashAI` 数据公开（HF）但引擎运行时非公开；
   `FirstLight` 仓库已 404（数据仍在 HF）。

---

## §3 CR 领域 RL 项目的奖励设计（含证据）

| 项目 | 胜负 / 终局 | 塔血 | 圣水 | 其他 shaping | 已知效果（原文） |
|---|---|---|---|---|---|
| **`Jaso1024`**（PPO） | **win +300.0 / loss −200.0 / draw 0** | 并入 `crowns_reward`：`round(4.96392*ln(4.86466*crowns+0.753851)+1.40353)` | 无 | 无 | README：目标是打败训练营对手；未给胜率 |
| **`wty-yy/KataCR`**（离线 RL，`reward_builder.py` 实测） | 塔毁即计入（公主±1 / 国王±3） | **公主塔被毁 ±1，国王塔被毁 ±3；塔伤 ±(ΔHP/full_HP)；开国王塔 −0.1** | **溢出 −0.05 / 每 10 帧** | `MAX_DELTA_HP=1600` 作 OCR 跳变过滤 | README：离线策略与 **8000 分 AI** 实时对局，附 12 个获胜对局视频 |
| **`weihaog1/The-Elixir-Optimizers`**（BC→PPO，6 轮迭代） | 终局 **Win/Loss ±3.0、Crowns ±1.0（×0.1）** | 迭代 4「unit advantage 0.03×Δ」、迭代 5「defensive placement +0.15 near enemies」 | 迭代 3「elixir waste penalty −0.1 at 9.5+」、迭代 6「low-elixir noop bonus +0.01 below 3」 | 迭代 2「survival bonus +0.02/step」 | 迭代 6 后 session 19 达 **27%**；最终评测 **10 局 30% 胜率 vs 真人**；总记录 5W/80L/3D，**前 74 局连败** |
| **`hastylmao/Hasty-CR`**（PPO + self-play） | `--win` | `--crown`、`--chip` | `--elixir` | 开关名即设计（**数值未公开**） | 「**No trained policy has beaten a human**」；自陈 sim 胜率是「自己给自己判卷」 |
| **`shawnxu0407/Clash_Royale_agent`** | 无终局项 | **reward = 对方塔 HP 下降为正、否则为负** | 无（但 elixir 作为 state 输入） | 无 | 课程项目，未报告胜率 |
| **`FirstLight_CR`**（IL→PPO；**经 ClashAI 单方审阅**） | terminal outcome | tower-health shaping | elixir-overflow penalties | （specialist 1）Hog deploy / first-Hog timing shaping + **demonstration-BC 项** | 4 行 argmax model-vs-model 胜率表：General vs IL **73.30%**；specialist 1 vs IL/ActiveIL/General **84.74/84.08/79.74%**；specialist 2 **89.87/86.97/88.03%**。⚠️ 作者自注「不同任务分布，不能当作一条连续胜率曲线」 |
| **`MSU-AI/clash-royale-gym`** | 未核实 | 未核实 | 未核实 | — | README 仅给 env 用法 |

**共性**：胜负信号一律稀疏（终局才给），塔血是最普遍的 shaping，**圣水项几乎只以「惩罚溢出/浪费」的形式出现，没有项目把「攒费」正奖励化**。
`KataCR` 的圣水项是**负惩罚**（溢出），`The-Elixir-Optimizers` 迭代 6 的「低圣水时不出牌 +0.01」是**最接近「攒费」的正项**，但其效果只报告到 session 级胜率（27%）。

**arXiv 命中**（`export.arxiv.org/api/query?search_query=all:"Clash Royale"`，2026-09-19 取数，共 4 条）：
- `2012.12186` (2020-12-22) **Learning to Play Imperfect-Information Games by Imitating an Oracle Planner** —— Supercell 作者；
  `Jason-XII/clash-royale-simulator` README 引用它论证「经验收集是瓶颈，Supercell 员工也没法加速引擎」。
  **方法 = 模仿 oracle planner，不是模仿人类**。
- `2504.04783` (2025-04-07) **Playing Non-Embedded Card-Based Games with Reinforcement Learning**（= KataCR 的论文）
- `2605.21868` (2026-05-21) **When to Switch, Not Just What: Transition Quality Prediction in Clash Royale**
- `2512.23718` (2025-12-10) Network Traffic Analysis with Process Mining（**与 CR 无关**，仅关键词命中）

---

## §4 可行性结论（**明确：需自采集**）

### 4.1 公开数据**不能**支撑「示范注入『攒费→打 Xbow』」

三条独立理由，每条都有证据：

1. **状态侧缺失，而目标行为是状态条件的。**
   「攒费→打 Xbow」的判据是「**圣水 ≥6 时**打 Xbow」。我实测两个最大公开集
   （`VanguardX101/IL_Replay` 的 parquet schema 与 `Cochon123` 的 raw payload）：
   **逐帧圣水不可观测**；只有全场 `aggregate_stats.total = [张数, 总圣水]` 与全场 `leaked`。
   ⇒ 可以**反推**圣水（起始 5 + 固定回费 − 卡费），但误差从未被任何一方标定，且
   `IL_Replay` 的 `final_tower_hitpoints` 只是**终局**值，无法校验中间态。
2. **第三方实证：语料规模—一致率的斜率极低。**
   `ClashAI/HANDOFF.md`（777 KB，逐条留证）给出：BC 的 **top-1 pro-cell 一致率 15.44%（v1）/ 46.61% top-5**，
   v5 时 exact-cell ≈ **20.93%**；测得 **+1.50 ± 0.13 pp / 每翻倍语料**（两套牌分别 +1.60 / +1.34），
   **「x10 predicts ~23.5 %, 30 % needs ~81k replays」**。
   且他们把 FirstLight 的 252k 语料折算到 icebow 单套牌只剩 **~2,070 侧（0.41%）**。
   ⇒ **252k 回放 ≠ 252k 条我们卡组的可用示范**；稀有行为（Xbow 是特定卡）在通用语料里被稀释。
3. **量纲/合法性不同构。**
   公开集的动作是 `(card_key, x, y)`；我们的动作是**同一个 `ActionBundle`（多卡包 + 格 + 掩码）**，
   掩码/合法性由 `rl/action_mask.py` 定义（AGENTS.md 已记录 10 条掩码与引擎的冲突）。
   公开数据无法提供掩码态，跨域迁移必须先做一层映射，而 ClashAI 已实测出这种映射的代价：
   引擎→live **exact cell −4.2 pp、card −12.4 pp**（他们的 live shift 读数）。
4. **补充实证**：`The-Elixir-Optimizers` 用 **40 局 / 1,600 动作帧**做 BC 的失败模式是**教科书级**：
   「position loss 占 79%」「576 格每格不到 3 个样本」「val loss 第 5 epoch 后发散」「最好的 F1 0.384」。
   作者结论：「Addressing this requires **20-40 more recorded matches**, combined with PPO fine-tuning」。

### 4.2 所以：**走自采集**——仓内通道已存在且优于公开数据

**`src/clasher_new/rl/human_play.py`（354 行，本仓，我读了源码）** 已经是一条完整的采集 + BC 训练通道：

| 维度 | 事实（源码位置） |
|---|---|
| 对手 | 人 = player-0，模型 = player-1（`FollowerPolicy`，`deterministic=True`）（`:63-80`） |
| 推进粒度 | 每决策 tick（`RLEnv.step` = 0.5 s 战斗时间），人每出一次动作推进一帧（`:9`） |
| **BC 样本** | `(obs, belief_tok, plan_vec, bundle, masks)` → `bc_<ts>.pkl`（`:7-8`） |
| 同时落盘 | `episode_<ts>.pkl`（`EpisodeReplay`，含 `hidden` 特权标签）、`session_<ts>.json`（胜负/步数/累计奖励/样本数）（`:11-14`） |
| BC 训练 | `train_bc_from_human(data_dir, out="follower_human.pt", epochs=5, lr=1e-3)`；产物可 `--init-from` 给 PPO（`:205-233`） |
| CLI | `--policy ... --drive-games N`；`--export <dir>`；`--bc-train --data-dir <dir> --out follower_human.pt`（`:16-19, 305`） |
| 卡组 | `--deck` 可覆盖，默认 `DEFAULT_PLAY_DECK` 8 卡（`:47-49`） |

**这条通道相对公开数据的决定性优势**：观测就是我们训练同款的
`(obs, belief_tok, plan_vec, masks)`，动作就是**同一个 `ActionBundle`** ⇒
**零域差、零量纲映射、掩码一致**——不存在 ClashAI 那个 15% 一致率与 −4.2 pp 的 live 退化问题，
而且**圣水天然逐帧可见**（公开数据最大的缺口）。

**自采集要满足的条件（可核实事实 + 明确标注的推断）**：

| # | 条件 | 依据 |
|---|---|---|
| ① | **人力时间**：一局 = 实时对局，`max_steps` 默认 600 tick × 0.5 s ⇒ 每局约 5 分钟真实时间，且**只能一次一局**（无并行） | `human_play.py:66` `max_steps=600`；`RLEnv.step` 语义见 AGENTS.md §3 关键常量 |
| ② | **必须「定向采集」而非随机对战**：要让「攒费→打 Xbow」出现在样本里，人必须在**圣水 ≥6 且局面合适**时**主动不出牌**再打 Xbow。理由：本仓实测「放弃一张买得起的牌」在 75,891 帧里发生 **0 次**（AGENTS.md 索引：`docs/elixir_saving_audit_2026-09-18.md`）⇒ 随机/自然对战采样**不会**产出这条轨迹 | AGENTS.md C14 / `docs/elixir_saving_audit_2026-09-18.md` |
| ③ | **样本量下限只能靠实测标定，不能抄**：外部参照 `The-Elixir-Optimizers` 的 40 局/1,600 动作帧**不够**（position loss 79%、val 早发散）；ClashAI 的斜率（+1.5 pp/翻倍）说明**买「出哪张牌」便宜、买「打哪个格」很贵** | `docs/final.md` §4b/§4e；`ClashAI/HANDOFF.md` |
| ④ | **现有 BC 实现不能判好坏（工程缺口）**：`train_bc_from_human` 是 **batch=1 的逐样本 SGD**（`for i in perm: loss.backward(); opt.step()`），`epochs=5`，**无 train/val 划分、无任何一致率/准确率指标**，只打印 `mean_logprob` | `human_play.py:219-230` |
| ⑤ | **对手多样性**：当前对手固定为确定性策略 ⇒ 示范分布被对手风格锁定。要扩分布需换对手/换卡组（`--deck` 已有，换对手脚本尚无入口） | `human_play.py:76-80` |
| ⑥ | **合规**：自采集**无第三方 ToS 风险**；公开回放集（RoyaleAPI 派生）则明确受 Supercell + RoyaleAPI ToS 约束 | `Cochon123` dataset card「License」段原文 |
| ⑦ | **建议补记的字段**：为了事后筛选「攒费→Xbow」样本，`session_*.json` 应补记**每帧圣水**与**是否打出目标卡**（replay 已含全状态，属于解析工作而非采集工作） | 推断（基于 ② 的筛选需求） |

### 4.3 可用作**辅助/对照**的公开资产（不构成主语料）

- `VanguardX101/IL_Replay`：252,238 回放 / 17.8M 动作 / 1.89 GB —— 唯一「大规模 + 20 Hz + 坐标」的人类动作集；
  可用来做**先验初始化**、**行为分布统计**（如「人类 Xbow 卡组在第几秒、圣水多少时下 Xbow」的**众数**），
  或作为**对照基线**。⚠️ **许可未声明**，商用/再分发前须自行确认。
- `Cochon123/clash-royale-replays`：MIT（含 ToS 提醒）+ 含 Hero/Champion ability 标注，事件更干净。
- `chrisrca/clash-royale-tv-replays`：像素帧，只对**视觉 bot** 有意义。
- Kaggle 的 481M / 37.9M matches：**对局级**，只适合做**卡组/胜率/元游戏（meta）统计**，对 IL 无用。

---

## §5 来源清单（URL + 取数时间）

**取数时间统一为 2026-09-19（本机时钟 2026-09-18 20:00–20:30 UTC）**

**官方 / 第三方 API**
- https://developer.clashroyale.com/ （SPA，正文需登录）；`/api/documentation` → `401 {"error":"session-not-found"}`
- `https://api.clashroyale.com/v1/players/{tag}/battlelog` → `403 {"reason":"accessDenied","message":"Missing authorization"}`
- https://github.com/RoyaleAPI/cr-api （archived；README：「We no longer publish a public API. Please use the official API from Supercell.」）
- https://github.com/RoyaleAPI/cr-api-docs/blob/master/docs/endpoints/player_battles.md （旧 RoyaleAPI battles 端点字段；无回放）
- https://github.com/RoyaleAPI/cr-api-docs/blob/master/docs/authentication.md ・ `docs/faq.md`
- https://github.com/martincarrera/clash-royale-api （363★, MIT, **archived**, pushed 2018-08-16；仅静态数据 API）

**HuggingFace（经 hf-mirror.com 取证）**
- https://hf-mirror.com/api/datasets?search=clash+royale （11 条 CR 相关数据集）
- https://hf-mirror.com/api/datasets/VanguardX101/IL_Replay ・ `/raw/main/README.md` ・ `/resolve/main/manifest.json` ・ `/resolve/main/compatibility.json` ・ `/api/datasets/VanguardX101/IL_Replay/croissant`
- https://hf-mirror.com/datasets/VanguardX101/IL_Replay/resolve/main/actions/part-000000.parquet （22,697,255 B）
- https://hf-mirror.com/datasets/VanguardX101/IL_Replay/resolve/main/replays/part-000000.parquet （17,432,059 B）
- https://hf-mirror.com/api/datasets/VanguardX101/IL_Replay/tree/main/replays （52 分片，825,394,311 B）
- https://hf-mirror.com/datasets/Cochon123/clash-royale-replays/raw/main/README.md
- https://hf-mirror.com/datasets/Cochon123/clash-royale-replays/resolve/main/cleaned-legacy/manifest.json ・ `.../events.jsonl` ・ `.../matches.jsonl`
- https://hf-mirror.com/datasets/Cochon123/clash-royale-replays/resolve/main/raw/00/000YYC2UJPVG.json （58,520 B，原始 payload 解析）
- https://hf-mirror.com/datasets/chrisrca/clash-royale-tv-replays/raw/main/README.md ・ `/api/datasets/chrisrca/clash-royale-tv-replays`
- https://hf-mirror.com/api/datasets/Chutdedededede/clash-royale-tv-replays ・ `Grandediw/clash-royale-battle`

**Kaggle**
- `https://www.kaggle.com/api/v1/datasets/list?search=clash%20royale` （20 条）
- https://www.kaggle.com/datasets/nonrice/clash-royale-battles-upper-ladder-december-2021 （**已下载 11,617,500 B 并读列名**）
- https://www.kaggle.com/datasets/pepe93/my-clash-royale-ladder-battles （**已下载 29,148 B 并读列名**）
- https://www.kaggle.com/datasets/s1m0n38/clash-royale-games ・ https://www.kaggle.com/datasets/bwandowando/clash-royale-season-18-dec-0320-dataset ・ https://www.kaggle.com/datasets/jackmangione/clash-royale-matchups-june2026

**GitHub 项目（元数据经 `api.github.com/repos/{full_name}`；README/源码经 `cdn.jsdelivr.net/gh/...`）**
- `Jaasssoooonnnnn/FirstLight_CR` → **404**（2026-09-19）
- https://github.com/vegetableleaf/ClashAI ・ `scratchpad/gauntlet/L67/firstlight_review.md` ・ `HANDOFF.md`（777,363 B）・ `tools/README.md`
- https://github.com/wty-yy/KataCR ・ `katacr/policy/perceptron/reward_builder.py`
- https://github.com/weihaog1/The-Elixir-Optimizers ・ `docs/final.md`
- https://github.com/shawnxu0407/Clash_Royale_agent ・ https://github.com/hastylmao/Hasty-CR
- https://github.com/Jaso1024/Real-Time-Strategy-RL-Clash-Royale ・ `CRBot.py`
- https://github.com/Jason-XII/clash-royale-simulator ・ https://github.com/Jason-XII/cr-memory-reader
- https://github.com/MSU-AI/clash-royale-gym ・ https://github.com/artumont/SICrMLB
- https://github.com/Pbatch/ClashRoyaleBuildABot ・ https://github.com/pyclashbot/py-clash-bot
- https://github.com/chrisrca/CS541-Deep-Learning-Clash-Royale-Project （分支 `emulation`）
- https://github.com/wty-yy/Clash-Royale-Replay-Dataset ・ https://github.com/wty-yy/Clash-Royale-Detection-Dataset
- https://github.com/cochon123/clash-royale-ai （采集管线 + 调查报告）
- https://github.com/HumaidAlhuthali/clash-royale-dataset-browser （HF 数据集浏览器）

**arXiv**
- `http://export.arxiv.org/api/query?search_query=all:"Clash Royale"&max_results=40` → 4 条
- https://arxiv.org/abs/2012.12186 ・ https://arxiv.org/abs/2504.04783 ・ https://arxiv.org/abs/2605.21868

**本仓（自采集通道）**
- `src/clasher_new/rl/human_play.py`（354 行，已读）
- `docs/agents/redlines.md` / `docs/elixir_saving_audit_2026-09-18.md`（AGENTS.md §B 索引；「放弃一张买得起的牌 0/75,891 帧」）
- `docs/engine_comparison_firstlight.md`（本仓对 FirstLight_CR 的既有对比记录，含「252K 回放 / 17.8M 动作」）

---

## §6 未验证 / 未能核实项

| # | 未核实项 | 原因 | 可核实的替代路径 |
|---|---|---|---|
| U1 | **官方 API 文档正文**（`battlelog` 到底返回哪些字段） | `developer.clashroyale.com` 是 SPA，`/api/documentation` 需登录会话（401） | 申请 developer key + IP 白名单后直取；或找有 key 的第三方文档 |
| U2 | **2019 年前后官方 API 变更的原始公告** | `royaleapi.com`(403 Cloudflare)、`web.archive.org`(000)、各家搜索引擎均不可达/被地域重定向 | 可达网络环境下查 RoyaleAPI 博客 / Supercell 公告；本报告只核实到「RoyaleAPI 自建 API 已停用（archived，2020-10-25）」 |
| U3 | **`FirstLight_CR` 仓库当前状态**（删除？私有？改名？） | `api.github.com/repos/Jaasssoooonnnnn/FirstLight_CR` → **404**（2026-09-19）。§2 中它的方法/损失/胜率**全部转引自 ClashAI 的审阅文档**，**无法与原文交叉验证** | 找到作者新地址；或经 ClashAI 审阅时的 commit `28d66cc` 的镜像/本地副本 |
| U4 | `wty-yy/Clash-Royale-Replay-Dataset` 的 README / 许可 / 数据量与格式 | `git clone` 失败、`cdn.jsdelivr.net` 对该仓 README 返回 404（疑似仓内无 README 或分支名不同） | 在可达网络下 `git clone` 后直接读；或查 GitHub 网页 |
| U5 | `AlexanderPotiagalov/ClashRoyaleML` 的元数据 | `api.github.com` 返回**空响应**（非 404，疑似瞬时故障） | 重试 `api.github.com/repos/...` |
| U6 | `chrisrca/clash-royale-tv-replays` 的**总对局数** | HF tree API 对该仓 `?recursive=true` 未返回可比结果；我只数了 `arena_01` = 39 条目 ≈ **13 场** | 逐 arena 目录计数，或下载后统计 |
| U7 | `Grandediw/clash-royale-battle` / `raymond9326/clash-royale-battles` / `CrabGuyy/ClashRoyaleTrophyMatches` / `lillyem/clash-royale-data` 的**字段 schema** | 只取到数据集 card 层，未下载 parquet | 逐一下载并读 schema（判定是否含动作侧） |
| U8 | `Hasty-CR` / `MSU-AI` / `Jason-XII` 的**奖励权重数值** | README 只给开关名，未给数字；未逐个读 trainer 源码 | 读各自 `rl/` 目录源码 |
| U9 | `VanguardX101/IL_Replay` 的**许可** | 数据集 card 与 `croissant.license` 均为空；HF tags 无 license 项 | 直接联系作者 VanguardX101；或在 HF 页面确认是否后续补了 license |
| U10 | 公开回放数据中**「人类 Xbow 出手时圣水」的分布** | 圣水不可观测（本报告 §1.4 已证），无法直接统计；只能靠「起始 5 + 固定回费 − 卡费」反推，**反推误差未标定** | 用本仓自有 replay（含全状态）标定反推误差，再决定能否用公开集做先验 |
| U11 | `ClashAI` 引用的 FirstLight 数字（252k / 8 GPU / 1,528 并发 / 各胜率） | **仅来自 ClashAI 单方审阅文档**，原文仓库已 404 | 仅其中 **252,238 回放 / 17,836,160 动作 / 1.89 GB** 已由我**独立核实**（HF API + `manifest.json` + parquet 实读）；其余数字**保持「未核实」** |
