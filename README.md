# 皇室战争模拟器 · 强化学习训练闭环

[English Version](./readme_en.md) ｜ 项目介绍视频：https://www.bilibili.com/video/BV1n3uZ6WE5P/

> **这个仓库是什么**：一个自建的《皇室战争》**确定性模拟器**，以及搭建在它之上的一整套**强化学习训练闭环**。
> 模拟器当世界模型，用来绕开"训练 AI 却收集不到经验"这个瓶颈；仓库的重心是**把 AI 训起来**，模拟器是它的地基。

## 与上游的关系

本项目 fork 自开源模拟器 [Jason-XII/clash-royale-simulator](https://github.com/Jason-XII/clash-royale-simulator)。
**模拟器引擎的最初实现（A\* 寻路、索敌、卡牌交互、游戏数值复刻）来自上游作者**，在此明确致谢。

但本仓库已经远不止"上游的一个补丁"：

- 卡牌覆盖从最初的 47 张扩展到 **122 张可实现卡**（含觉醒、精英 / Hero 机制），数值快照 148 条；
- **从零搭建了完整的 RL 训练闭环**（`src/clasher_new/rl/`，37 个模块、约 1.7 万行），上游没有这部分；
- 引入引擎侧外置工具、推理时浅 MCTS、以及一套**取证驱动的行为诊断方法学**。

换句话说：**上游贡献了"游戏"，本仓库贡献了"让 AI 学会玩这款游戏"。** 下面的内容以本仓库的增量为主线。

## 为什么需要它

想让 AI 学会玩好一款游戏，必须让它收集海量经验。对《皇室战争》来说，经验收集是人机和训练的最大瓶颈——哪怕掌握内部战斗引擎接口的 Supercell 员工，也没法加速引擎来更快地收集经验（*Learning to Play Imperfect-Information Games by Imitating an Oracle Planner*, arXiv:2012.12186）。

本仓库的答案是**自建完整模拟器**：读取准确的游戏数值，复现原引擎的 A\* 寻路与卡牌交互，把一个 180 秒的对局压缩到 **1.1 秒左右**跑完（上游在 M4 上约 150×，普通 CPU 约 70–90×）。**模拟器的上限就是这个项目的上限。**

在模拟器之上，还有一套强化学习环境与训练闭环，用来真的把模型练出来。

## 效果展示

![demo](./demo2.gif)

上图为模拟器界面与游戏实际效果的对比：把真实对局里的下牌时间输入模拟器，前 30 秒的卡牌交互完全一致。

## 项目亮点（本仓库的增量）

### 1. 完整 RL 训练闭环（`src/clasher_new/rl/`）

上游只有 `environment.py` / `train.py` 两个文件；本仓库把它扩展成了一个可长期维护的训练系统：

- **同刻多卡动作**：一个决策步 = 一个 `ActionBundle`（`K_MAX=4`），同 tick 可同时出多张牌 + 英雄技能；提交前**整包原子校验**（手牌解析 / Mirror 语义 / 圣水推演 / 技能就绪），任一非法即拒绝整包。
- **掩码与提交共用坐标契约**：`SubAction(x, y)` 一律是玩家本地坐标，掩码层与提交层共用 `sub_position`，杜绝镜像坐标分裂。
- **信念推断**：对手 8 卡循环队列用 **O(1) 数学锁定**（第 4 张起手牌集合可精确 0/1 推出，避开 40320 种牌序的全量枚举），前 3 张与异常观测才走粒子滤波，再叠一层 GRU 序列编码与**对手出牌事件通道**。
- **计划通道 `PlanToken`**：57 维战术计划 token、21 个宏观意图，作为特征注入策略（不进动作空间）。
- **三种训练模式**（`start_rl.bat` 统一入口）：
  - `solo` —— 单人自对弈（固定卡组镜像 + 周期冻结副本 + 对手池）；
  - `run` —— 联赛（同时维护 5 个流派卡组模型 + main PPO，PFSP 采样）；
  - `flow` —— 全配对分流派联赛（6 个可训练 PPO，卡组池两两全配对，一次训练 148,800 局）。
- **推理时浅 MCTS**（`rl/mcts.py`）：UCT + 引擎确定性叶推演，候选动作复用 `legal_cells`/`validate_bundle`（与提交路径同源），策略 logits 作先验截断。零训练风险，用来量"搜索能赚多少"。

### 2. 引擎侧外置工具（确定性估值服务，不进动作空间）

把领域知识放在引擎侧、网络只管高层选择（借鉴 AlphaStar / TStarBot 谱系）：

| 工具 | 文件 | 用途 |
|---|---|---|
| 塔伤威胁计算器 | `src/clasher_new/threat_calc.py` | 现存部队"不管"情况下的塔损预估 |
| 交换模拟器 | `src/clasher_new/simulate_exchange.py` | "现在打出这张牌"的反事实推演 |
| 法术知识模块 | `src/clasher_new/spell_module.py` | 法术引擎标定档案 + 落点估值 |

### 3. 取证驱动的行为诊断

本项目不只看胜率，而是**用 replay 逐帧证据定位"模型为什么学不会某件事"**，再针对性修机制。几个真实案例：

- **法术砸塔病理** → 加前段对塔 EV 硬闸门，前段砸塔率 54% → 26%；
- **防守"追尾"** → 按威胁深度分派的拦截几何，接敌率 7.4% → 10.5%；
- **攒费链死锁** → 定位到"先有血牛才准攒、先攒满才准沉底"的鸡生蛋结构，改为两阶段状态机。

### 4. 训练基础设施

跨进程并行 worker（绕开 GIL 吃满多核）、断点续训、命名配置 + 奖惩机制、Web 仪表盘（Elo/胜率曲线 + 回放播放器）、人机对战采集（供模仿学习）。

## 快速开始

### 环境准备

```bat
:: 首次：创建 .venv 并安装 torch(CPU) / gymnasium / stable-baselines3 / tqdm
start_training.bat --setup

:: 有 NVIDIA GPU（CUDA 13 / cu130）：
start_training.bat --setup-cuda
```

模拟器的可视化窗口还需要 `pygame`：`pip install pygame`。

### 自检

```bash
cd src/clasher_new
../.venv/Scripts/python.exe rl/selftest.py     # 全链路自检 + 回归测试
```

### 运行模拟器 / 观看对局

```bash
python src/clasher_new/new_visualization.py    # 打开 pygame 可视化窗口跑一局
```

### 启动训练

```bat
:: 交互式向导（选模式 / 配置 / 步数 / 是否开仪表盘）
start_rl.bat

:: 直接指定：solo 自对弈（默认 economy 经济配置）
start_rl.bat --mode solo --config economy

:: 联赛（5 个流派卡组模型 + main）
start_rl.bat --mode run --config aggressive

:: 全配对分流派联赛（6 模型，规模最大）
start_rl.bat --mode flow --config economy
```

训练产物按命名配置落在 `src/clasher_new/runs/<name>/`（checkpoint / 优化器 / 状态 json / 回放），训练中自动打开网页仪表盘（默认 8090 端口）。

> 完整的模块表、每种模式的命令、消融与评估协议见 **[`src/clasher_new/rl/README.md`](src/clasher_new/rl/README.md)**。

### 局域网联机（模拟器自带的测试功能）

1. 查本机局域网 IP（Windows `ipconfig` / macOS `ifconfig | grep inet`，通常 `192.168.` 开头）。
2. 在 `src/clasher_new/server.py` 末尾填入该 IP 并运行。
3. 两台电脑分别运行 `src/clasher_new/client_side/client.py`，选卡组、填 IP 连接。
4. 双方都连上后自动开局。

## 项目结构

```plaintext
src/clasher_new/
├── arena.py / core.py / player.py / card_utils.py       # 基础数据结构
├── battle.py / card_mechanics.py / pathfinding*.py      # 战斗引擎 / 卡牌逻辑 / 寻路
├── new_visualization.py                                 # pygame 可视化（模拟器入口）
├── server.py / client_side/client.py                    # 局域网联机
├── evolutions.py / evo_2025_data.py / elite17_data.py   # 觉醒 / 精英 / Hero 机制
├── threat_calc.py / simulate_exchange.py / spell_module.py   # 引擎侧外置工具
└── rl/                                                  # ★ RL 训练闭环（37 模块）
    ├── env_wrapper.py / action_bundle.py / action_mask.py    # 环境 / 动作包 / 掩码
    ├── observation.py / belief.py / bayes_filter.py          # 观测 / 信念推断
    ├── belief_planner.py / prophet.py / plan_space.py        # 计划通道 / 特权先知
    ├── follower.py / ppo.py / workers.py                     # 策略网络 / PPO / 并行 worker
    ├── train_solo.py / run_league.py / flow_league.py        # 三种训练模式
    ├── league.py / pfsp.py / elo.py / evaluate.py            # 联赛 / 采样 / 评测
    ├── config.py / decks.py / opponents.py / overtime.py     # 配置 / 卡组 / 对手 / 加时
    ├── dashboard.py / replay.py / human_play.py / mcts.py    # 仪表盘 / 回放 / 人机 / MCTS
    └── README.md · selftest.py                               # 导读 · 回归测试

start_rl.bat          # 训练统一启动器（solo / run / flow + 仪表盘）
start_training.bat    # 环境安装 + 启动训练
docs/                 # 314 个条目（41 个 .md）：策划/规格/报告 + 原始采集证据
AGENTS.md             # ★ 红线与结论索引（改方案前先读；历史全文在 docs/agents_archive_2026-09.md）
docs/plan_master.md   # ★ 所有计划的整合视图（现状 / 优先级 / 判据设计）
```

## 模拟器覆盖

- 官方数值快照 **148 条**（含 4 座塔）；批量冒烟全部通过（`scripts/batch_smoke.py`，报告见 `docs/batch_smoke_report.json`）。
- 其中 **122 张为 `implemented`**（机制接入 + 数值验证），另有 2 张 `needs_review`、14 张活动临时卡、6 张变体（默认排除）。
- 已覆盖 **觉醒（Evolution）** 与 **精英 / Hero** 机制。
- 寻路与原游戏一致，大部分角色数值与游戏相同。

完整覆盖矩阵与逐卡状态见 **[`docs/card_coverage.md`](docs/card_coverage.md)**。

## 训练现状与已知问题（2026-09-13）

**一句话**：模拟器与训练闭环已经跑通，**"训练不崩"已解决**（对手分布改造 D1 生效并扛住 4× 训练量），
**"能不能变强"未解决**（上限没动、价值网络仍未拟合），行为质量在爬坡但"不会攒费"的病理没修完。

| 方向 | 状态 | 结论 |
|---|---|---|
| 训练不崩 | ✅ 已解决 | D1（对手分布去镜像化 + 动态历史自身联赛 + PFSP 门禁）：20k 三跑最差锚点 **0.125/0.200/0.250** vs 无干预 **0.000/0.000/0.025**（区间不相交）；100k 长跑 **P1 通过** |
| 能否变强（上限） | ❌ 未解决 | 100k 的最差锚点 0.237 **落在 20k 区间 [0.125,0.250] 内** ⇒ 与 20k 无统计差别；cycling 仍在转 |
| 价值网络（critic） | ⛔ 已闭合 | 根因（GRU 输入饱和）已修且 100k 耐久，但 EV 仍 ≈0/负、`vstd/rstd` ≈0.001；**加量（100k）与四代架构都没救回来** ⇒ 停止投入 |
| 训练量 | ❌ 被否证 | 20k ≈ 55~80 局、100k ≈ 374 局，对标 Atari PPO 的 10M~50M 帧只有 1/500~1/2500 ⇒ "20k 没提升"在样本量上就是必然 |
| 测量与评估口径 | ✅ 已解决 | EV 口径三修、固定随机锚点仪器、相对门禁、**评估节奏 C 方案**（密锚点 + 稀全块）；并立下"判据基线必须脚本复算"的纪律 |
| 行为质量 | ⚠️ 部分 | 防守响应率 38%→**62%**、接敌率 7.4%→**10.5%**、单边堆牌 46%→9.5%；但**攒费链死锁已定位、修复未实现** |

**判读方法学（本项目的特点）**：不看单个胜率数字，而是 **replay 逐帧取证 + 预注册判据 + 脚手架化判读工具**。
判读脚本已入库：`scripts/judge_anchor_blocks.py`（锚点分块判据，基线由磁盘复算）、
`scripts/summarize_solo_run.py`（长 run 诊断汇总）、`scripts/diag_*.py`（critic/表征诊断）。

**去哪看细节**：

- [`docs/plan_master.md`](docs/plan_master.md) —— **所有计划的整合视图**（哪条路走通了 / 被否证 / 未决 + 优先级与判据设计）
- [`AGENTS.md`](AGENTS.md) —— **红线与结论索引**（改方案/写计划前必读，编号可引用：R1-R12 / C1-C10 / X1-X9 / O1-O5）
- [`docs/d1_long_100k_verdict_2026-09-13.md`](docs/d1_long_100k_verdict_2026-09-13.md) —— 最近一次长跑判读
- [`docs/training_audit_2026-09-11.md`](docs/training_audit_2026-09-11.md) —— 训练体系审计（四问）
- [`docs/rl_training_fix_plan_v3.md`](docs/rl_training_fix_plan_v3.md) —— **当前生效**的整改计划（v1/v2 已被它修订或推翻）
- [`docs/agents_archive_2026-09.md`](docs/agents_archive_2026-09.md) —— 历史决策全文存档（过程细节/推理链）

## 文档导航

| 想做什么 | 去哪 |
|---|---|
| 理解模拟器 / 跑起来 | 本文件 |
| 启动训练 | `start_rl.bat`、`start_training.bat` |
| RL 算法代码导读 | [`src/clasher_new/rl/README.md`](src/clasher_new/rl/README.md) |
| **所有计划的整合视图**（现状 / 优先级 / 判据设计） | [`docs/plan_master.md`](docs/plan_master.md) |
| 跨会话红线与决策索引（改方案前**先读**） | [`AGENTS.md`](AGENTS.md) |
| 历史决策全文（过程细节） | [`docs/agents_archive_2026-09.md`](docs/agents_archive_2026-09.md) |
| 文档 ↔ 源码完整索引 | [`docs/README.md`](docs/README.md) |
| 自检 / 回归 | `src/clasher_new/rl/selftest.py`（97 项）、`scripts/test_m*.py`、`scripts/batch_smoke.py` |
| 判读 / 诊断工具 | `scripts/judge_anchor_blocks.py`、`scripts/summarize_solo_run.py`、`scripts/diag_*.py` |

## 参与

模拟器是这个项目的天花板：**我一人无力把 122 张卡的所有交互都做到完全准确**。如果你愿意帮忙实现一两张牌、修正某个机制，或者改进训练算法，非常欢迎提 Issue / PR。

如果这个项目对你有帮助，点个 star 吧。

---

**致谢**：模拟器引擎源自开源项目 [Jason-XII/clash-royale-simulator](https://github.com/Jason-XII/clash-royale-simulator)；经验收集瓶颈与 oracle planner 的论述参考 arXiv:2012.12186。
