# rl/ —— AI 训练算法闭环

对应规划文档 `docs/ai_training_plan.md` 与评审修复手册 `docs/rl_review_fix_plan.md`。
引擎冻结：本包不改 `battle.py` / 卡牌内容。

> 2025 修复状态：`docs/rl_review_fix_plan.md` 的 6 条 P0 Critical 已全部修复并有回归测试
> （`selftest.py` 18 项全绿）；P1 大部完成（信念序列训练、联赛主循环、Exploiter 闭环、
> 先知信号有效化、replay 链路、评测指标、BC 初始化）；P2 清理批次同步进行中。

## 模块

| 文件 | 说明 |
|---|---|
| `action_bundle.py` | 同刻多卡动作包 `ActionBundle`（`K_MAX=4`，含英雄技能 `kind="ability"`）；`sub_position` 是唯一坐标换算入口 |
| `action_mask.py` | 动态动作掩码 + 整包原子校验（含 Mirror 引擎语义、技能耗蓝模拟、`at_cap` 强制 STOP） |
| `observation.py` | 玩家视角观测 + 特权隐藏状态标签 |
| `env_wrapper.py` | `RLEnv`：整包校验 → 同 tick 批量 `deploy_card` → 统一推进决策帧；掩码指纹含 `player_id`/手牌 |
| `belief.py` | 信念推断：规则队列锁定（第 4 张起 0/1）+ 早期/异常粒子滤波 + 统计倾向 + 神经 GRU 编码；`opp_played` 结构化多卡契约 |
| `bayes_filter.py` | 对手 8 卡循环队列信念：O(1) 队列锁定（手牌=卡组−最近4张）+ 前 3 张/异常粒子相；无 40320 全量重建 |
| `belief_planner.py` | 基于 `b_t` 的可部署规划器（过滤静态塔、region 由 intent 推导）；Phase 2 v1 **bp 组 12 意图**（含 protect_backline 反应+信念预判、king_activate），与 ProphetPlanner 同链同序 |
| `prophet.py` | 特权完整状态启发式先知（训练期教师 30% 帧；消费对手手牌/圣水）；Phase 2 v1 pp 组：punish/anti_spell/save_ace 特权精确版 + king_activate/protect_backline（bp 尚未实现的意图先示范）+ 与 bp 同链标签一致 |
| `plan_space.py` | `PlanToken` 计划空间与向量化（`PLAN_DIM` 唯一常量源） |
| `follower.py` | 跟随者策略：CNN+GRU + autoregressive bundle head；`evaluate` 返回真实熵；`save/load_checkpoint` 带元数据 |
| `ppo.py` | 轻量 PPO（GAE + clip）：hidden 重放一致、真实熵正则、截断 bootstrap |
| `league.py` / `pfsp.py` / `elo.py` | 联赛（快照隔离/持久化）/ PFSP（乐观先验）/ Elo（按局数缩放 K） |
| `config.py` | **命名训练配置**：一组超参 + **奖惩机制奖励权重** → 命名 → 独立输出文件夹；预设 standard/aggressive/defensive/**lockdown**/elixir/**economy**/fast + 按流派覆盖 `MODEL_REWARD_OVERRIDES`，支持导出/载入 JSON |
| `workers.py` | **跨进程训练 worker**：独立进程跑 env+信念+规划（绕开 GIL 吃满多核），与主进程以 (obs, belief_tok, plan_vec) 同步协议交互 |
| `replay.py` | 对局 replay（schema v2 统一容器，含特权标签）+ **紧凑联赛录像**（每 2000 步保存，供回放） |
| `train_belief.py` | 信念监督训练：**整局序列** GRU 训练 + 验证集/温度缩放/ECE |
| `train_bc.py` | 跟随者行为克隆预训练（规则专家）→ 供 `train_follower --init-from` |
| `train_follower.py` | 跟随者 PPO：belief/plan dropout、蒸馏注入、`FollowerOpponent` 完整链路 |
| `train_exploiter.py` | Exploiter：换边胜率评估 + 阈值入联赛 |
| `train_prophet.py` | 特权先知 PPO + `prophet_policy_to_plan` 适配器（产物可被蒸馏） |
| `run_league.py` | `--mode eval` 轮转评估（换边/三态/逐局 Elo）+ `--mode run` 联赛主循环（**同时维护 5 个卡组模型**：推进流 / 防守反击流 / 自闭流 / 全 200 卡组 / 全随机，+ 训练中的 main）；支持**命名配置**、**断点续训 --resume**、**CUDA --device**、**每 2000 步联赛录像** |
| `flow_league.py` | **全配对分流派联赛（`--mode flow`）**：6 个**可训练 PPO**（main/推进/防反/自闭/全量/随机）卡组池两两全配对，每对数据只喂该对双方模型（对内流式），双侧轨迹收集 + 镜像奖励；一次训练 148,800 局 |
| `decks.py` | **三分类卡组加载器**：读取 `docs/leaderboard_decks_classified.json`（200 副天梯卡组，推进流 60 / 防守反击流 120 / 自闭流 20），RoyaleAPI 卡名 → 引擎卡名映射 + 兜底补位 |
| `opponents.py` | 脚本策略 `ScriptedPolicy`：random / heuristic / **卡组完全随机**（每局从 139 张引擎卡池重采样 8 张）/ **deck_pool 随机抽整副卡组**（三分类/全 200 模型用） |
| `dashboard.py` | **训练网页 UI**：Elo-训练次数 曲线仪表盘 + **最近训练回放列表 / Canvas 播放器**（纯 Canvas 自绘、离线可用、3s 轮询 `/api/state`、5s 轮询 `/api/replays`） |
| `evaluate.py` | 评测：Win/Lose/Draw、Bundle 合法率、Next-Card Acc/Brier/ECE、消融、`--belief-only` 协议；消融含 **逐意图采纳探针**（region 吻合率 + save_ace hold 服从率，full vs plan-off Δ） |
| `selftest.py` | 全链路自检 + 评审回归测试（P0-1..P0-6、P1-4/5/9/18/21） |

## 训练入口

```bash
# 环境在 venv 中（.venv/bin/python）
cd src/clasher_new

# 0) 自检（含评审回归测试）
python rl/selftest.py

# 1) 信念监督训练（整局序列 GRU + 校准）
python rl/train_belief.py --n-games 50 --epochs 10 --out belief_encoder.pt

# 2) 跟随者 BC 预训练（可选，PPO 的非零起点）
python rl/train_bc.py --n-games 50 --epochs 3 --out follower_bc.pt

# 3) 跟随者 PPO（同刻多卡 + 信念 + plan；可 --init-from follower_bc.pt）
python rl/train_follower.py --total-steps 100000 --save follower.pt --init-from follower_bc.pt

# 4) 单卡 baseline（对照）
python rl/train_baseline.py --total-timesteps 200000 --save baseline_ppo

# 5) 特权先知（Route A；产物经 prophet_policy_to_plan 接入蒸馏）
python rl/train_prophet.py --total-timesteps 100000 --save prophet_ppo

# 6) Exploiter（针对固定 Main Agent；达标自动入联赛）
python rl/train_exploiter.py --main-policy-path follower.pt --save exploiter.pt \
    --league-state league_state.json --n-eval-games 10

# 7) 联赛：轮转评估 / 主循环（同时维护 5 个卡组模型 + main）
python rl/run_league.py --mode eval --policies follower.pt exploiter.pt --n-games 20
python rl/run_league.py --mode run --total-steps 20000 --save-state league_state.json \
    --decks-path docs/leaderboard_decks_classified.json   # 缺省自动探测

# 7a) 命名配置 + 奖惩机制（每个配置一个文件夹，产物全在 out_dir/<name>/ 下）
python rl/run_league.py --mode run --config aggressive --out-dir runs      # 预置奖励方案（推进：费差 0.7）
python rl/run_league.py --mode run --config lockdown --out-dir runs        # 自闭：费差≈0（鼓励费差换塔血）
python rl/run_league.py --mode run --config economy --out-dir runs         # 费差机制别名（默认：塔血%归一化+费差 0.5）
python rl/run_league.py --mode run --config economy --card-level 16        # 跨等级训练（奖励语义不变）
python rl/run_league.py --mode run --config defensive --config-name my-run # 自定义文件夹名
python rl/run_league.py --mode run --config standard --save-config my.json # 导出参数
python rl/run_league.py --mode run --load-config my.json --config-name v2  # 载入并改名

# 7b) 断点续训 + CUDA（cu130）+ 每 2000 步联赛录像（默认开启）
python rl/run_league.py --mode run --config aggressive --resume --device cuda
python rl/run_league.py --mode run --config fast --no-replays --device cpu   # 关录像/用 CPU

# 7c) 并行多环境（n_envs>1）：
#   parallel=mp  （默认）跨进程 worker：env+信念+规划在独立进程多核并行，主进程批量 GPU 推理
#   parallel=proc 单进程批量化（无多核加速，仅 batch 推理/更新）
python rl/run_league.py --mode run --config aggressive --n-envs 4 --device cuda
python rl/run_league.py --mode run --config aggressive --n-envs 8 --parallel mp --device cuda
python rl/run_league.py --mode run --no-eval-start   # 关掉"启动先评估一次"（默认开，WebUI 立即有真实数据）

# 8) 训练网页 UI：各模型 Elo-训练次数 曲线 + 最近回放/播放器（指向命名配置的联赛状态）
python rl/dashboard.py --state runs/aggressive/league_state.json --port 8090
python rl/dashboard.py --state runs/aggressive/league_state.json --sweep runs/economy --port 8090  # 同时显示 flow-sweep 进度/曲线
python rl/dashboard.py --solo runs/economy/solo_state.json --port 8090   # solo 自对弈：胜率曲线±SE + 训练进度（无联赛）
python rl/dashboard.py --play runs/solo --port 8090       # 人机对战：浏览器里打训练模型，对局数据实时落盘（EpisodeReplay + BC）
python rl/dashboard.py --demo --port 8090                       # 无状态时生成演示数据（Elo+sweep+solo+回放）直接看 UI
python rl/dashboard.py --state runs/aggressive/league_state.json --replays runs/aggressive/replays   # 手动指定回放目录
#   --sweep 指向 runs/<name>/（自动扫 flow_sweep_stream / flow_sweep_games5）或单个策略目录；
#   训练进行中 dashboard 每 3s 读取逐轮增量写的 summary.json → 进度条（run x/N + ETA）+ main 曲线 ±1σ 误差棒

# 7d) 全配对分流派联赛（6 个可训练 PPO，卡组池两两全配对；默认一次训练 148,800 局）
#    注意：flow 模式按模型奖惩（MODEL_REWARD_OVERRIDES 覆盖费差：main/all/random=0.5、
#    推进=0.7、防反=0.3、自闭=0.05），--config 只决定共享超参与基线权重。
#    引擎稳定性修复（battle.py/card_mechanics.py）：所有迭代 self.entities 的循环改为
#    list() 快照，消除"击杀 on_death 生成新实体 → 迭代中字典被改"的 RuntimeError 崩溃。
python rl/run_league.py --mode flow --config economy --device cuda
python rl/run_league.py --mode flow --n-random-decks 30 --out-dir runs   # 随机卡组套数（默认 30）
#   产物：runs/<name>/flow_<id>.pt（6 个模型各一个 checkpoint）；--max-ep-steps 控制单局决策步
#   断点续练：--mode flow --resume —— 从 flow_run_state.json 恢复对进度（pair/game）+ 6 模型
#   与优化器（flow_opt_<id>.pt），跳过已完成对继续（每对结束落盘）
#   注意：规模大，建议先小池试跑（详见关键语义「全配对分流派联赛」）

# 7e) flow 数据效率 A/B（缩小 10× 池，验证曲线上涨再上 148,800）
python rl/run_league.py --mode flow-sweep-stream --config economy --device cuda   # 每对 1 局 × 20 次训练
python rl/run_league.py --mode flow-sweep-games5 --config economy --device cuda   # 每对 5 局 × 4 次训练
#   两策略总对局预算相同（20×1,488 = 4×7,440 ≈ 29,760）；产物 flow_sweep_<strategy>/
#   summary.json(csv)：逐轮 main 轮内估计 ±SE + 首/末趋势判定（Δ/SE≥2σ 才算上涨）
#   可选 --sweep-runs / --sweep-scale / --sweep-eval-games 覆盖

# 7f) solo 自对弈（原版 train.py 思路的现代版；固定卡组镜像，无联赛机制）
#     单模型 main，双方同一副固定 8 卡（Knight/MiniPekka/Arrows/Minions/Musketeer/
#     Fireball/Giant/Archer），对手 = main 的周期冻结副本（每 solo_copy_every 步同步，
#     即原版 WeightsCopyingCallback）；不写 Elo/PFSP/league_state。
python rl/run_league.py --mode solo --config economy --device cuda
python rl/run_league.py --mode solo --config economy --solo-copy-every 2000   # 冻结副本同步间隔（步）
#   断点续练：--mode solo --resume —— 从 run_state.json 恢复 step + solo_main.pt + 优化器
#   (solo_opt.pt) + 历史曲线（solo_state.json），不重跑起始评估、曲线不断裂
#   产物：runs/<name>/solo_state.json（胜率±SE/mean_reward/进度，dashboard --solo 实时读）、
#   solo_main.pt（最新指针）+ solo_main_<step>.pt（每次评估的历史检查点，回溯用）、
#   replays/league_<step>.pkl（评估回放，复用回放面板）
#   dashboard：python rl/dashboard.py --solo runs/economy/solo_state.json --port 8090

# 7g) 人机对战 + 人类数据采集（人在浏览器打训练模型，对局转训练数据）
#     人 = player-0（蓝），对手 = FollowerPolicy（player-1，deterministic）；
#     固定 8 卡镜像。每步记录两类数据并落盘 --play-out：
#       episode_<ts>.pkl  → EpisodeReplay（含 hidden 特权标签）→ train_belief
#       bc_<ts>.pkl       → (obs,bundle,belief,plan,masks) → 模仿学习/行为克隆
python rl/dashboard.py --play runs/solo --play-out runs/solo/human_data --port 8090   # 浏览器对战+采集
#   导出与训练：
python rl/human_play.py --export --data-dir runs/solo/human_data --out-belief belief.pkl --out-bc bc.pkl
python rl/train_belief.py --replays-path belief.pkl --out belief_human.pt            # 信念监督（人类数据）
python rl/human_play.py --bc-train --data-dir runs/solo/human_data --out follower_human.pt  # 模仿学习预训练
python rl/run_league.py --mode run --main-init follower_human.pt --config economy    # 用人类 BC 初始化后 PPO
#   无 UI 冒烟：python rl/human_play.py --policy runs/solo/solo_main.pt --drive-games 3

# 7h) 训练提速：评估加速（训练慢的主因 = 评估开销爆炸，不是训练本身）
#   评估是全配对 C(6,2)=15 对 × n_eval_games 局，每局最多 max_ep_steps 步的完整 CPU 模拟
#   + belief（前 3 张 128 粒子，第 4 张后 O(1) 队列锁定）推理；n_eval_games 4→40 时评估量是训练步数的 ~180 倍上限。
#   三招压评估开销：
#     1) 配置（config.py 默认）：n_eval_games 40→16（轮内 SE≈39，仍可区分学习信号）、
#        steps_per_eval 2000→4000（评估频率减半、每 ckpt 训练量翻倍）、
#        economy 预设 only_vs_main=True（league 评估 15 对→5 对；solo 不受影响）。
#     2) play_pair 复用单个 env：每局 reset(seed=...) 换对局，不再逐局重建 RLEnv/BattleState。
#     3) 僵局早停（run_league._stall_probe）：连续 100 步双方塔血合计零变化 → 判平提前结束。
#        CR 无塔治疗，塔血只降不升，"长时间零塔损"是可靠僵局信号；只影响本来就会平局的局，
#        胜负判定不变，只是更快得出"平局"。
#   实测（aggressive ckpt，CPU）：600 步僵局局 22.9→4.05s/局（÷5.7）；solo 评估
#   40 局≈15 分钟/2000 步 → 16 局≈1 分钟/4000 步（约 1/30）。
#   4) 并行评估（solo，进程池）：战斗模拟是纯 Python（GIL 受限），线程并行无效，必须跨进程。
#      --eval-workers N（>1 时 spawn N 进程，每 worker 独立 env+信念+策略，主进程汇总；0=串行）。
#      与串行 eval_solo 同种子逐局等价（worker 走完整 reset 链还原同一信念先验，selftest 验证）。
#      实测（economy，16 局 eval@0，RTX 4070 laptop）：串行 126.1s → 16 进程 ~15-25s。
#   用法：
#     python rl/run_league.py --mode solo --config economy --eval-workers 16

# 9) 评测（含消融 / 信念协议）
python rl/evaluate.py --policy follower.pt --n-games 50
python rl/evaluate.py --policy follower.pt --n-games 200 --ablation all --ablation-out ablation_result.json
#   消融 all：full/plan-off/belief-off/both-off 四变体 + Δ±SE + z 判定，落盘 JSON+CSV
#   （验证 belief/plan 注入是加分还是纯噪声；注意 RNN hidden 仍含历史信息，属保守消融）
python rl/evaluate.py --policy follower.pt --n-games 500 --belief-only

# 9) 导出 replay（供 train_belief --replays-path 消费）
python rl/export_replay.py --n-games 20 --out replays.pkl --opponent heuristic
python rl/train_belief.py --replays-path replays.pkl --epochs 10 --out belief_encoder.pt
```

仓库根目录另有 `scripts/rl/*` 包装脚本，用法一致。

## Windows 一键启动

仓库根目录 `start_training.bat`：自动定位 `.venv`、可选建环境装依赖、启动
**联赛训练窗口 + 仪表盘窗口 + 自动开浏览器**。

```bat
start_training.bat                        :: 默认参数（standard 配置，20000 步，每 2000 步评估，端口 8090）
start_training.bat --config aggressive     :: 用「推进」奖惩机制配置（输出 runs\aggressive\）
start_training.bat --config aggressive --resume   :: 断点续训
start_training.bat --device cuda           :: GPU 训练（需先 --setup-cuda 装 cu130 torch）
start_training.bat --total-steps 50000 --port 8090
start_training.bat --setup                :: 首次：创建 .venv 并安装 torch(CPU)/gymnasium/sb3
start_training.bat --setup-cuda           :: 首次：创建 .venv 并安装 cu130 (CUDA 13) torch
start_training.bat --selftest             :: 先跑自检再启动
start_training.bat --help
```

注意：值参数用 `--opt=value` 形式拼接，路径含空格时请改用命令行直接运行
（`scripts\rl\run_league.py` / `scripts\rl\dashboard.py`）。仪表盘会自动指向
`runs\<config>\league_state.json`。

## 关键语义

- **5 卡组模型同时维护**（`run_league --mode run`，接入你的三分类数据集）：
  `push_flow`（推进流 60 副）/ `counter_flow`（防守反击流 120 副）/ `lockdown_flow`（自闭流 20 副）/
  `all_decks`（全 200 副）/ `random_deck`（全随机 8 卡）。有卡组的模型每局从对应集合**随机抽一副完整卡组**；
  `main`（跟随者 PPO）为训练目标，PFSP 从这 5 个对手采样。卡名经 `rl/decks.py` 映射到引擎卡，对不上的槽位用引擎卡池补位。
- **评估逐 pair 独立采样**：`eval_round_robin` 每对卡组用独立种子流（`_pair_seed_offset`），
  避免所有 pair 复用同一批逐局种子打出"同构局面"（曾见 main 对 5 个对手胜率全等
  `0.40725312499999994 = 0.5×0.95⁴`：n_eval_games=4 全败 + 种子复用所致，**非写入 bug**）。
  每个 pair 的 PFSP 胜率流由自己的比分序列独立 EMA 演进（`pfsp.update_winrate` 按
  `(agent_a, agent_b)` 独立 key），互不污染；回归测试：`test_winrate_streams_independent`。
- **评估粒度 / 噪声地板**（曲线可信度上限）：K=32 逐局 Elo 是**有限记忆跟踪器**
  （MC：单轮噪声 1σ≈±40 即饱和，加局数不收窄运行 Elo）。因此每轮评估额外计算
  **轮内聚合估计** `D̂=400·log₁₀((w+0.5)/(N−w+0.5))`，`SE≈347.5/√N`（N=该 agent 本轮
  总对局数，p=0.5 最坏情形；无偏且 SD 已 MC 验证）。`round_stats` 随 state 持久化，
  dashboard 曲线画 ±1σ 竖线误差棒、表格给"Δ上轮 / σ 信号/噪声"（≥2σ 才可信）：
  `n_eval_games=4` → main(5对,20局) SE≈78，纯噪声下 |Δ|≥100 概率≈36%（**±100 移动
  不可区分信号**）；`n_eval_games=40`（默认）→ main(200局) SE≈25，该概率<1%
  （±100 移动≈2.9σ，可区分学习信号）。回归测试：`test_elo_eval_granularity`。
- **belief/plan 注入消融**（上全规模前的设计验证，`evaluate.py --ablation all`）：
  prophet / belief_planner 是启发式，天花板受限于手写规则质量——`--ablation all` 跑
  full / plan-off / belief-off / both-off 四变体，输出各变体 WinRate±SE（二项 SE=
  √(p(1-p)/N)）与相对 full 的 Δ±SE、z=Δ/SE（|z|≥2 视为有真实贡献），**落盘
  JSON+CSV**。注意 token 置零是保守消融（RNN hidden 仍含历史 belief/plan 信息），
  结论应结合 z 与样本量。Phase 2 追加**逐意图采纳探针**：每帧记录 bp plan 意图标签
  下模型首个部署格与 focus_region 的几何吻合率（region_rate）与 save_ace 的
  hold_mask 服从率（hold_rate），比较 full vs plan-off → Δ>0 = plan 注入被该意图采纳。
  回归测试：`test_ablation_recorded`。
- **flow 数据效率 A/B**（`run_flow_sweep`，`--mode flow-sweep-stream / -games5`）：
  把卡组池缩小一个数量级（`scale_pools(×0.1)` → 6/12/2/20/3/20，一次训练 1,488 局），
  对比两种数据效率策略：**stream**＝每对 1 局忠实流式×20 次完整训练；**games5**＝每对
  5 局×4 次。总对局预算相同（≈29,760）。每轮训练后对 6 模型做全配对换边评估，记录
  main 的**轮内聚合估计**±SE，输出 summary.json/csv + 首/末趋势判定（Δ/SE≥2σ 才算
  "曲线确实上涨"）——先验证设计有效再投入 148,800 局全规模。回归测试：
  `test_flow_sweep_smoke`。
- **全配对分流派联赛**（`flow_league.py`，`run_league --mode flow`）：6 个模型全部为**可训练
  PPO**（`main` / `push_flow` / `counter_flow` / `lockdown_flow` / `all_decks` /
  `random_deck`），每个 `FollowerPolicy` + 独立 `PPOTrainer`。6 个卡组池两两全配对
  （C(6,2)=15 对），每副卡组 vs 每副卡组打 1 局、不换边：推进 60 × 防反 120 × 自闭 20 ×
  全量 200 × 随机 30（`--n-random-decks`，每次训练生成）× main 200 →
  **一次训练 = 148,800 局**。**每对（pair）数据只喂该对双方模型**（不跨对混合）；
  因 obs grid 每条约 32KB、整对收集会爆内存，"每对打完即训"落地为**对内流式**：
  攒够 `update_interval` 条即更新该对双方模型再丢弃。双侧都采探索轨迹：
  player-1 侧由 `FollowerOpponent.take_last_step()` 收集，player-1 的 reward 用
  `compute_reward` 交换 blue/red 视角镜像计算。数据归属（on-policy，每模型只用自己
  一侧轨迹）：推进 34,200 / 防反 61,200 / 自闭 12,200 / 全量 86,000 / 随机 18,000 /
  main 86,000 局。产物 `runs/<name>/flow_<id>.pt`（6 个 checkpoint）。
  回归测试：`test_flow_league_smoke`（mini 池 15 对 50 局全配对 + 双侧轨迹 + 每模型
  至少一次更新 + 落盘；并断言真实池计数 = 148,800）。
- **训练网页 UI**：`dashboard.py` 读取联赛状态 JSON（含 `elo_history`），页面每 3s 轮询 `/api/state`，Canvas 自绘各模型 Elo-训练次数曲线（无 CDN 依赖）。
- **最近回放 + 播放器**：`dashboard.py` 自动扫描 `runs/<config>/replays/league_<step>.pkl`（每评估周期保存的联赛录像，`rl/replay.py` schema 3），
  `/api/replays` 列出最近回放（按修改时间倒序，带局数/大小缓存），`/api/replay?file=..&game=N` 返回单局帧；
  页面内 Canvas 播放器回放（18×32 战场、塔血环、实体插值移动/淡入淡出、圣水条、皇冠、本步动作/奖励/对手出牌），
  支持播放/暂停、逐帧、进度条、0.5×–8× 变速、多局切换。无 CDN 依赖、离线可用。
- **同刻多卡**：一个决策步 = 一个 `ActionBundle`；wrapper 内多次 `deploy_card` / `use_ability` 之间不调用 `battle.step`，全部子动作（出牌 + 英雄技能）在同一 tick 提交。
- **整包原子校验**：提交前校验整个 bundle（含决策时刻手牌解析、Mirror 引擎语义、圣水扣减推演、技能就绪校验）；任一非法即拒绝整包并惩罚；引擎级拒绝会 `RuntimeWarning` 暴露掩码缺口。
- **坐标契约**：`SubAction(x, y)` 一律是玩家本地坐标，掩码层与提交层共用 `sub_position`，杜绝镜像分裂。
- **英雄技能**：`SubAction(kind="ability")` 触发 `battle.use_ability`；跟随者 bundle head 决策空间为「出牌槽位 + ABILITY + STOP」，`at_cap` 时强制 STOP。
- **信念**：`b_t = P(z_t | o_1:t, a_1:t-1)`；`info["opp_played"]` 为结构化列表 `[{"card","x","y"},...]`，技能哨兵由信念入口过滤；统计层记录落点/风格；圣水有独立计数器估计。规则层（`bayes_filter.py`）用 O(1) 队列锁定：8 卡内容已知 + 出牌按序全观测时，**第 4 张起手牌集合 = 卡组 − 最近 4 张、下一张 = 第 k−3 张打出牌（精确 0/1，与开局洗牌无关）**；仅前 3 张与异常观测走粒子近似。
- **PPO 正确性**：重放使用 rollout 记录的掩码序列与 `init_hidden`；熵项为真实分布熵；截断 episode 用 `last_value` bootstrap。
- **维度契约**：`PLAN_DIM = len(PlanToken().to_vector())`，belief 维度由 `belief_token_dim(deck)` 计算；checkpoint 一律带元数据，加载时校验。
- **不泄漏特权**：跟随者只接收 `belief_token`（推断结果）与 plan token，绝不直接接收 `z_t`。
- **命名配置 + 奖惩机制**（`rl/config.py`）：每个 `TrainConfig` = 一组超参 + 一套 `reward` 权重
  （破塔/皇冠/胜负/非法/圣水效率系数），`--config` 选预设、`--load-config` 载入自定义 JSON、
  `--save-config` 导出；所有训练产物按 `out_dir/<name>/` 分文件夹（`config.json` /
  `league_state.json` / `main_ckpt_*` / `main_opt_*` / `run_state.json` / `replays/`）。
- **奖惩机制（2025-06 改版：费差默认打开 + 按流派区分）**：`rl/config.py` 的
  `DEFAULT_REWARD` 现在是**塔血统一 + 费差机制**（不再有"旧公式"）：
  - **塔血统一**（`reward.tower_dmg_opp == reward.tower_dmg_self == 0.001`）：打击与损失
    同价，删掉旧"挨打比打人贵 20%"的不对称。
  - **塔血归一化**（`reward.normalize_tower_dmg=True`）：塔损按 `本局初始总塔血` 归一化到
    lv11 锚（`_TOWER_HP_ANCHOR=10928`，引擎真实 lv11 总塔血 = 2×3052+4824）→ 同一
    "塔血百分比事件"在任何等级给同一奖励。
  - **显式费差项**（`reward.elixir_diff_weight=0.5`）：每步 Δ(我方圣水−对方圣水)，
    potential-style shaping（整局闭环累计归零，只重排不改变总回报），给圣水显式定价：
    **lv11 下 1 圣水 ≈ 500 塔血**（`0.5 / 0.001`；用户校准 300-700 血带中位）——模型只在
    "花 1 费能换 ≥500 塔血"时才愿意花，浪费（花圣水无塔伤）仍是惩罚。
  - **按流派奖惩**（`config.MODEL_REWARD_OVERRIDES`，flow 联赛 6 模型）：在所选预设之上
    按模型覆盖费差权重——`main`/`all_decks`/`random_deck` 用**同一基线**（0.5，1圣水≈500血）；
    `push_flow`（推进）**加码**到 0.7（1圣水≈700血，增加"塔血换费差"）；
    `counter_flow`（防守反击）**减码**到 0.3（1圣水≈300血）；`lockdown_flow`（自闭）
    **压到 ≈0**（0.05，1圣水≈50血，鼓励"费差换塔血"——花圣水换塔伤；浪费仍小惩罚）。
    同一对局内 A/B 各用各的权重算 reward（`flow_league._play_one`）。
  - 命名预设同步更新：`standard` = 新默认（费差 0.5）；`aggressive`=推进（皇冠 8/胜 15 +
    费差 0.7）；`defensive`=防反（非法 0.1 + 费差 0.3）；`lockdown`=自闭（费差 0.05）；
    `elixir`/`economy` 保留为圣水效率 / 费差经济别名。
  配套修复：`RLEnv.reset` 现在会 `battle.update_player_hp()` 同步 PlayerState 塔血到真实
  实体 HP（消除等级>11 时每局第一步"塔血暴涨"的假奖励；lv11 下与旧行为逐位一致）。
  另支持 `--card-level 11-16`（`TrainConfig.card_level`，跨进程 worker 一并贯通）——
  配合费差机制即可跨等级训练/评估而奖励语义不变。
  **塔型感知**：真实游戏里国王塔恒定（lv11=4824），四种公主塔血量各异
  （lv11：PrincessTower 3052 / DaggerDuchess 2768 / RoyalChef 2703 / Cannoneer 2616，
  见 `rl/env_wrapper.TOWER_TROOP_HP_LV11`）。归一化分母取**本局真实总塔血**
  （`tower_total_hp(troop, king)`），因此同一"塔血百分比事件"在任何塔型、任何等级给同一
  奖励（回归测试 `test_tower_troop_hp_reference`）；更弱塔受同等绝对伤害时奖励更高
  （更大的塔血百分比 = 更接近皇冠）。注：引擎目前只模拟标准 Tower Princess（3052/4824），
  其余塔型仅作参考表，待引擎支持后自动生效。
  回归测试：`test_config_reward_weights` / `test_model_reward_overrides` /
  `test_reward_economy_*` + `test_tower_troop_hp_reference`
  （预设/塔血统一/按流派覆盖/等级不变/费差定价/trade 直觉/卡牌等级/塔型不变）。
- **断点续训**：每个评估周期把 `step`、main 权重、Adam 优化器状态写入 `run_state.json`；
  `--resume` 从上次 step 继续（`--total-steps` 可调大续到更远）。
- **CUDA 支持**：`--device cpu|cuda|auto`（默认 auto=可用则 cuda）；跟随者/PPO 全程按
  `policy.device` 搬运张量。需安装匹配的 cu130 torch（`start_training.bat --setup-cuda`）。
- **并行多环境**：`--n-envs N`（默认 1）。N>1 默认走 **跨进程 worker（`--parallel mp`）**：
  env+信念+规划放在独立进程（绕开 GIL、吃满多核），主进程用 `act_parallel` /
  `evaluate_batch` 做批量 GPU 推理与整批 PPO 更新（数值与单条路径逐位等价，有回归测试）。
  `--parallel proc` 则退回单进程批量化。实测（本机 16 核 + RTX 4070，cu130）：
  800 env-steps 训练循环单进程 49s → mp n_envs=4 18.7s（**~2.6×**；更多 worker 可继续提升）。
  注意：league 评估（每 steps_per_eval 一次）走单进程，耗时与并行无关，单独占墙钟时间。
- **启动先评估**：`eval_at_start`（默认开，`--no-eval-start` 关）在训练开始先跑一次
  全轮转评估/快照，WebUI 从第 0 步就有真实 Elo 数据，而不是只显示初始 1500。
- **联赛录像**：每个评估周期（默认 2000 步）把全轮转各局的紧凑帧（时间/动作/塔血/圣水/皇冠/实体）
  存到 `replays/league_<step>.pkl`；`--no-replays` 关闭。
- **Web UI 刷新**：`/api/state` 与页面均带 `no-store` 头，前端 fetch 带时间戳参数防缓存，3s 轮询必拿最新 Elo；
  页面顶部显示**数据来源**（真实状态文件路径 / ⚠ DEMO 合成数据）与"暂无评估数据"提示，避免误以为只显示预设值。
