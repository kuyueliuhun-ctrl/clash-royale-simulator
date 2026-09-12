# `ab_valnorm_20k` 20k 步验证跑 · G1 判读（2026-09-11）

配套文档：审计报告 `training_audit_2026-09-11.md`、整改计划 `rl_training_fix_plan_v1.md`。
run 目录：`src/clasher_new/runs/ab_valnorm_20k/`；日志：`docs/train_ab_valnorm_20k.log`。

---

## 0. 一句话结论

**机制指标达成（价值通道量纲修复生效：`v/p` 2.43 → 0.60~0.95），但没有任何结果类指标改善，
而且发现"训练量"本身被严重误读——20k 步 = 20k 决策帧 ≈ 55~80 局自对弈。**
G1 判为 **部分通过 / 目标未达成**：可以继续推进，但必须先修正步数口径与 critic 诊断，
否则 300k 只是把同样的平台期拉长。

---

## 1. 本次运行配置（与 9k_ft 同口径对照）

| 项 | 值 | 说明 |
|---|---|---|
| config | `economy` | 与 9k/9k_ft 同 |
| `total_steps` | 20000 | 语义见 §2 |
| `steps_per_eval` | 4000 | 中途由 8000 收窄 |
| `n_eval_games` | 40 | 计划 P0-2 落地 |
| `eval_workers` | 16 | 与 `start_rl.bat` 默认一致 |
| `value_norm` | `running` | **本次唯一变量** |
| `adv_norm` | `scale` | 计划 P0-1（原默认 `batch`） |
| `diagnose_every` | 10 | 新增梯度成分诊断 |
| `main_init` | `runs/economy_9k_ft/solo_main.pt` | 热启动 |
| `hist_seed_dirs` | `economy_9k_ft`, `economy_9j` | 计划 P1-1 |
| `max_ep_steps` | 360 | |
| reward | win/lose/draw = +10/−10/−10 | |

启动日志三项验收全对：resume 断点、`ReturnScaler` 跨断点恢复（`mean=1.588 std=10.55`
→ 终态 `mean=1.588 std=11.36, n=19840`）、对手池 `hist ckpts=12`（其中 10 来自补种目录）。

---

## 2. **重要口径修正：`step` = 决策帧，不是 PPO 更新次数，更不是"训练了很多"**

`train_solo.py:1064` `for step in range(start_step+1, cfg.total_steps+1)`，循环体每次只做
一次 `env.step(bundle)`（`:1104`）→ **1 step = 1 个决策帧**。PPO 更新在
`len(transitions) >= update_interval(128)` 时触发（`:1150`），因此：

- 20000 步 ≈ **156 次 PPO 更新**（20000/128；日志实测 199 行覆盖 13603+12000 两段，吻合）
- `max_ep_steps=360`，实测每局约 250~360 帧（由 deploy 占比 ≈8%、`deploy_per_game`≈20~30 反推）
- **整个 20k run ≈ 55~80 局自对弈**

与之前审计里的对标放在同一口径下：

| | 本仓库 | 对标 |
|---|---|---|
| 20k 步 | 2 万个决策帧 ≈ 60 局 | Atari PPO 常见 10M~50M 帧 = 本项目的 **500~2500 倍** |
| 100k 步（9j） | ≈ 280 局 | FirstLight 25M 动作的 **0.4%** |

→ 结论：**"20k 步看不出提升"在样本量上就是必然的**，这不是本次实验的失败证据，
但反过来说明：**继续按 `step` 计预算会系统性高估进度**。步数口径必须改成"局数/决策帧数"。

---

## 3. 机制指标（G1 三项 + 新增诊断）

| 指标 | 目标 | 实测 | 判读 |
|---|---|---|---|
| `value_loss`（归一化后） | < 10 | **1.59** | ✅ 名义达成 |
| `ratio` 离开 1.000 | 离开 | 1.000 / clip 0.0% | ⚪ **指标作废，非失败** |
| `v/p`（价值/策略梯度范数） | < 3 | **0.47 → 0.95**（原 2.43） | ✅ 达成 |
| `p_gnorm` / `v_gnorm` | 新增 | 140 / 133（末段 `v/p=0.95`） | ✅ 诊断通路可用 |
| `value_loss_raw` (`vraw`) | — | 中位 **≈130**，峰值 312 | ⚠️ **新发现：critic 没学到东西** |

### 3.1 `ratio ≡ 1.000` 是结构性的，G1 这条判据本身写错了

`ppo.py` 的模块 docstring（`:12-22`）在实现时就已注明：rollout 与更新共用同一份权重，
单次 update 内 `evaluate_batch` 重算的 logprob 与 rollout 记录的 `old_logprob` **恒等**，
且 `n_epochs=1`、`lr=3e-4`、梯度被裁到 0.5 → `ratio` 的偏离量级只有 1e-3。
`clip_frac ≡ 0%` 同理。**这条判据永远不会触发，应从 G1 移除**，替换为下面的 EV。

### 3.2 新发现（本次最硬的一条）：critic 的解释方差 ≈ 0

`ReturnScaler` 终态 `count=19840, mean=1.588, m2=2559652` →
回报方差 `129.0`、标准差 `11.36`。而 `vraw` 中位 ≈130、末值 203。

```
explained_variance = 1 − MSE / Var(returns) ≈ 1 − 130/129 ≈ 0.00   （末值 1 − 203/129 = −0.57）
```

**即价值网络对回报的解释力约等于"恒定预测均值"。** 这说明 P0-1 修复解决的是
"梯度配比"（v/p 达标），**没有也不可能解决 critic 本身的拟合失败**——它是另一个病：
要么回报的自对弈噪声本就不可从状态预测（镜像自对弈的对称性 → 状态几乎不含胜负信息），
要么价值头与策略共享 trunk 被策略梯度持续拖走。两者都需要单独取证，不能靠调 `vf_coef`。

**建议（低成本、决定性）**：在 `ppo.py` 的诊断里加 `explained_variance`，与
`p_gnorm/v_gnorm` 同频落盘。它比 `value_loss` 无量纲歧义，是"critic 是否在学"的唯一干净答案。

### 3.3 另一条旁证：优势均值系统性偏正

日志里 `adv=+13.978±2.884`、`+13.354±9.868` 等，`adv_mean` 常达 +5~+14（回报 std 仅 11），
说明 critic 系统性低估回报（欠拟合的必然结果）。与 §3.2 同源。

---

## 4. 结果指标：三条曲线全部不动

| step | main（vs frozen_copy） | 对照 vs `baseline0` | 对照 vs `baseline_prev` |
|---|---|---|---|
| 0 | 0.625±0.077 | 0.525±0.079 | 0.600±0.078 |
| 8000 | 0.550 | **0.850±0.057** | 0.700 |
| 12000 | 0.525 | **0.450** | 0.500 |
| 16000 | 0.625 | 0.625 | 0.475 |
| 20000 | **0.450±0.079** | **0.525±0.079** | **0.500±0.079** |

- **main 曲线结构性无意义**：`solo_copy_every=2000` 且 eval 前刚同步冻结副本
  （日志 `冻结副本已同步 @step 20000` 紧邻 `eval@20000`）→ 对手与 main 权重相同，
  恒为镜像局，期望必为 0.5。这就是 AGENTS 记过的"自我对冲无区分度"。
- **唯二有意义的对照口径，20k 步后净变化为零**：vs 起点 0.525（≈0.5）、vs 上一评估点 0.500。
- **噪声地板（本次自校准）**：step 0 时 main / baseline0 / baseline_prev 三者权重**完全相同**，
  却测出 0.625 / 0.525 / 0.600、engagement 28.3 / 34.0 / 38.1。
  → 40 局下胜率 1σ ≈ 0.078（与上报 SE 吻合）、**接敌率 1σ ≈ ±5pp**。
  8000 点的 0.850 与 12000 点的 0.450 分别偏离 0.5 约 4.4σ / 0.6σ，
  是**真实波动**（不是纯噪声），但一涨一跌互相抵消，终点回到起点。

**判读：20k 步内既无提升也无显著退化。** 结合 §2，"无提升"主要由样本量决定；
真正需要回答的是 **§3.2 的 critic 失能**与 §5 的门禁口径。

---

## 5. 行为指标：门禁能跑，但**阈值标定错了口径**

`gates.json`（终态）两项 PASS：`engagement_rate=14.6 ≥ 9.5`、`ghost_rate=3.0 ≤ 10.0`。
但阈值不可信：

- `config.py:150` 注释写的是"engagement_rate **9k_ft 实测 10.5%** → 阈值 9.5"，
  9.5 来自 `scripts/forensics_response.py`（一次性取证脚本）的口径；
- 本次训练内建指标（`train_solo.py:349`"部署后 8s 内 5 格内敌我 troop 同框"）对
  **同一批 9k_ft 权重**测出的是 **28.3 / 34.0 / 38.1**（step 0 的 main / baseline0 / baseline_prev）。
- 即 **阈值 9.5 与实测口径差约 3 倍**，PASS/FAIL 的语义是假的。

行为指标全趋势（main 曲线）：

| step | engagement | intercept | unilateral | ghost | deploy/game | elixir_avg | tower_diff | bundle_multi |
|---|---|---|---|---|---|---|---|---|
| 0 | 28.3 | 32.0 | 13.9 | 24.7 | 29.95 | 1.75 | +644 | 3.9 |
| 8000 | 1.1 | 3.1 | 17.6 | 13.9 | 26.5 | 1.80 | +452 | 38.3 |
| 12000 | 0.3 | 2.7 | 17.0 | 17.1 | 49.05 | 2.26 | −156 | 59.2 |
| 16000 | 1.0 | 24.2 | 15.5 | 2.3 | 29.57 | 2.03 | +281 | 51.0 |
| 20000 | 14.6 | 15.1 | 9.6 | 3.0 | 19.73 | 1.57 | −33 | 13.7 |

值得记的现象（**不当作结论，样本量不足**）：

1. **中段出现"防守消失"**：engagement 28.3 → 1.1 → 0.3（远超 ±5pp 噪声），
   同时 `deploy_per_game` 冲到 49、`bundle_multi` 冲到 59 → 像是**从"防守型"翻到"单边堆牌型"**；
   末段又回落到 14.6 / 19.7 / 13.7。这种大幅摆动说明**行为指标本身是高方差状态量**，
   在 40 局、单一对手分布下很难当门禁。
2. `elixir_avg` 全程 1.57~2.26（**没有攒费行为**）、`deploy_per_game` 末段反而降到 19.7
   —— AGENTS 记的"不会攒费/不会组波"病理**没有改善**。
3. `ghost_rate`（落点 y≥20 的 deploy 占比）末段 3.0，比 step 0 的 24.7 低一个量级——
   但 step 0 与 16000/20000 的巨大差异也可能受对手（frozen/hist/defend 混合）变化影响，
   不宜单独解读。

---

## 6. 环境可靠性：`WinError 1455` 复现 2 次，1 次降级

- 日志第 185-186 行：`OSError: [WinError 1455] ... Error loading cufft64_12.dll` →
  `[eval] 并行评估 worker 失败，降级串行: 1 个 worker 退出`。
  **只崩 1 个 worker，训练未中断**，但 eval@12000 全程串行 ≈8 分钟（并行本应 ~1 分钟）。
- 与宿主 shell shim 失灵（`dirname: command not found`）**同时发生**，但实测 commit
  余量（物理 31.3G / 提交上限 47.3G / 空闲 20.6G）在 16×1.3GB ≈ 20.8GB 的边界上。
- 按 AGENTS 新增的"全局操作约定"：**成因仍记为未定**，不在代码里写自信的错答案。

---

## 7. 结论与下一步

**G1 判读**

| G1 判据 | 结果 |
|---|---|
| `value_loss < 10` | ✅ 1.59（但见 §3.2，此数被 `s²` 除过，不独立说明问题） |
| `ratio` 离开 1.000 | ⚪ 判据作废（结构性恒等） |
| `v/p < 3` | ✅ 0.95 |
| **整体** | **部分通过**：机制修复生效，但"策略真的在学"仍未验证 |

**推进建议（按顺序）**

1. **先修口径，再堆量**（都属低成本）：
   - `ppo.py` 诊断加 `explained_variance`（§3.2）——这是判断 critic 是否可救的唯一干净指标；
   - G1 判据把 `ratio` 换成 `EV`（例如 `EV > 0.2` 才算 critic 可用）；
   - `gates.json` 阈值**按本次口径重标**（用 step 0 同权重三连测的均值 ±2σ 作基线），
     或先降级为纯记录不判定（当前是"会 PASS/FAIL 但阈值无意义"，最危险）。
2. **步数预算改口径**：把 `steps` 换算成"局数"，并把"最小可验证"从 200k 步
   重新表述为 **≥500 局**（≈150k~180k 步）。§2 的表已给出换算。
3. **critic 单独取证**（这是本轮唯一的新结构性问题）：先加 EV，再看
   "回报是否可从状态预测"。若镜像自对弈本身使状态-胜负近乎独立，
   则价值头不是调参问题，而是**必须引入非镜像对手**（P1 对手池的 hist/defend 方向是对的，
   但当前 frozen 仍占 70%）。
4. `value_norm=running` 建议**保留**（它确实把 v/p 拉回平衡，且 `none` 时逐位等于旧行为，
   可随时回退）；但不要指望它解决 critic 拟合。

**不建议现在做的事**：直接开 300k/1M。在 `EV≈0` 且步数口径不明的情况下，
堆量无法区分"样本不够"与"机制不对"。

---

## 8. 本次改动清单（代码）

| 文件 | 改动 |
|---|---|
| `rl/ppo.py` | `ReturnScaler` + `value_norm` + `p_gnorm/v_gnorm/value_loss_raw/value_scale` 诊断 |
| `rl/config.py` | `value_norm`/`diagnose_every`/`hist_seed_dirs`/`gates` 字段；`adv_norm` 默认 `scale`；`eval_workers` 默认 `min(16,cpu_count())`；`n_eval_games` 16→40 |
| `rl/train_solo.py` | 梯度诊断打印、`vraw` 日志字段、history 去重、行为指标门禁 `gates.json`、对照组带行为指标、`_collect_hist_ckpts` 多目录、`_OpponentPool` 补种 |
| `rl/run_league.py` | `--value-norm/--diagnose-every/--hist-seed-dir` CLI 透传 |
| `rl/selftest.py` | 新增 `test_value_channel_norm_and_gnorm_split`、`test_history_dedup_and_gates`、`test_opponent_pool_mix_multi_dir`（全绿，整套 selftest 3m35s 通过） |
