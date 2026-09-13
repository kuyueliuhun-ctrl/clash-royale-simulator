# 预注册：critic 惰性检验（2026-09-13）

> **状态**：判据已写死，**尚未开跑**。跑完只按本文件读，不许现编。
> **上游依据**：`docs/d1_long_100k_cause_analysis_2026-09-13.md` §5.3（"价值函数恒为常数 ⇒ 优势失去状态级信用分配"）
> 与该文 §6 明列的**最大缺口**：「塌缩 → advantage 失真 → 退化」这条传导链**一次都没有测量过**。
> 本实验就是去测这条链，而不是再跑一次 A/B 看曲线。

---

## 0. 要回答的问题（可否证形式）

**Q**：当前（`d1` 结构下）critic 对**策略梯度**的实际贡献是多少？

**H0（原假设，本文要检验的）**：critic 只是给优势加了一个近似**状态无关的常数**，
把它换成标量后策略更新方向基本不变 ⇒ 它对策略的贡献 ≈ 0。

**H1（对立假设）**：critic 在实质参与信用分配 ⇒ 去掉它的状态依赖会显著改变策略更新方向。

**为什么必须测这个**：若 H0 成立，则"修 critic"这件事的**收益上界就是"恢复信用分配"**，
而 X6 把 critic 侧关掉的处置需要重新评估；若 H1 成立，则本文的因果链作废，
O2 退回"训练过程未解"，且"修 critic"有直接收益、优先级应上调。

---

## 1. 设计：两层，主判据在机制层（高功效），功能层只作描述

### Layer 1 —— 纯测量探针（**主判据**，零行为改动）

在**基线配置**上只加日志（不参与任何梯度计算、不消耗 RNG），每个"诊断更新"
（`--diagnose-every 10`，即每 10 次 PPO 更新里的第 1 个 minibatch）记录：

| 量 | 定义 |
|---|---|
| `adv_inert_corr` | `corr(adv_real, adv_const)`（同一批 128 帧，逐帧相关；对正缩放不变） |
| **`adv_inert_resid_frac_norm`** | **`std(Â_real − Â_const) / std(Â_const)`，其中 `Â` = 按 `adv_norm="scale"` 归一化后的优势** ← **主判据用量：critic 的状态依赖占比（进损失的那个向量）** |
| `adv_inert_level_shift_norm` | `|mean(Â_real − Â_const)| / std(Â_const)` —— 归一化后的**水平偏移**（PPO 的 clip 项不消除常数偏移，故需单列） |
| `adv_inert_resid_frac` | `std(adv_real − adv_const) / std(adv_const)`（**原始**尺度；描述性，同时含形状差与尺度差） |
| `adv_inert_std_ratio` / `adv_inert_norm_std_ratio` | 原始 / 归一化后的 std 比 |
| `adv_level_gap` | `|mean(V) − c|`，其中 `c = ret_scaler.mean`（验证"常数水平是否真的匹配"） |
| `grad_cos` | `cos(g_p(adv_real), g_p(adv_const))`，**同一 minibatch、同一尺度化口径**，只对策略参数求梯度 |
| `grad_norm_ratio` | `‖g_p(adv_real)‖ / ‖g_p(adv_const)‖` |

其中 `adv_const` ≡ 用 `compute_gae` 在 **V 恒等于 c** 时算出的优势（`c = ppo.ret_scaler.mean`，
即迄今全部回报的运行均值 —— 这正是塌缩后的 critic 收敛到的那个常数），
`adv_real` ≡ 现行口径（`V` 为网络输出）。两者都按现行 `adv_norm="scale"` 各自除以自身 std。

> **⚠️ 修订 1（2026-09-13，开跑前，实现在先、判据随实现口径修正）**：
> 初稿把 `resid_frac` 定义在**原始**优势上。写单测时发现这不成立：
> `adv_norm="scale"` 会把优势除以自身 std ⇒ **优势整体缩放对梯度是恒等变换**
> （`grad_cos ≡ 1`，已单测固化）。若用原始口径，一个只把优势放大 2 倍的替代方案会被
> 读成 `resid_frac = 0.5`（"critic 贡献很大"），而它其实对策略**毫无影响**。
> ⇒ 主判据一律用 `*_norm` 系列；原始量只作描述。此修订发生在**任何 run 之前**，
> 且方向是**收紧**判据（原口径会把惰性判得更严/更松取决于缩放方向），不构成事后调参。

> **为什么用运行均值 c 而不是"本局平均回报"**：后者是**事后最优常数基线**，比塌缩的 critic 更强，
> 用它会把 H0 判得过宽（人为制造差异）。用运行均值才是"V ≡ 常数"的忠实代理。
> 代价：`adv_level_gap` 用于确认水平是否真的匹配；若该值很大，说明常数水平没对上，
> 主判据需按 §4 的分支降级读。

### Layer 2 —— 干预跑（**次级，描述性，不判决**）

`--critic-baseline const`：把优势计算里的 `V` 整体替换成标量 c（其余全部不变：
`returns`/价值损失/PPO 预算/对手池/奖励全不动）。跑 1~3 个 repeat，与基线 3 跑的锚点序列对比。

**为什么 Layer 2 只作描述**：R5（n=1/臂）—— 3 跑/臂下 20k 的 MDE ≈ 0.35，
锚点 1σ=0.078、相邻点差分 SE=0.110；"没差别"这种零假设在小样本上天然判不出。
**能判的只有大效应**：若干预跑出现"整段归零"（worst ≤ 0.025，即退化成无干预组的形态）
或大幅改善（末点 > 0.675，比基线 3 跑最高末点还高 0.11），那是大效应，作判决。

> **⚠️ 修订 2（2026-09-13，开跑前）**：初稿把 P1′（`level_gap` / `std_ratio` 一致性）写成
> **gate P1** 的前置条件。写单测与 700 步 smoke 后判定这个定位是错的：
> `grad_cos` 量的是**实际进入优化的那个梯度方向**，critic 带来的形状差与水平差都已包含其中；
> `level_gap` 大只说明"常数 c 与 V̄ 不重合"，**不影响**"换成标量后梯度方向变不变"这个问题的有效性。
> ⇒ P1′ 降级为**分解项**，只在 Layer 2（干预跑）的归因里起作用（见 §4 限制 2）。
> 两处修订都发生在任何 run 之前，且都是为了**不让判据把惰性判得过宽**。

---

## 2. 判据（跑之前写死）

| # | 判据 | 阈值 | 读法 |
|---|---|---|---|
| **P1** | **H0 成立（critic 惰性）** | 全 run `grad_cos` **中位数 ≥ 0.99** 且 `adv_inert_resid_frac_norm` **中位数 ≤ 0.10** | critic 对策略更新方向的改变 <1%（cos）、归一化后状态依赖占比 <10% ⇒ "修 critic"的收益上界 = 恢复信用分配 |
| **P2** | **H0 被否证（critic 在工作）** | `grad_cos` **中位数 ≤ 0.95** 或 `resid_frac_norm` **中位数 ≥ 0.30** | critic 在实质参与 ⇒ §5.3 的因果链作废，O2 退回"训练过程未解"，修 critic 有直接收益 |
| **P3** | **灰区** | 0.95 < cos < 0.99，或 0.10 < resid_norm < 0.30 | **不判决**，写"部分惰性"，需更长 run / 更多诊断点 |
| **P1′** | **分解检查（不 gate P1；只用于 Layer 2 的归因）** | 报告 `adv_level_gap` 与 `adv_inert_level_shift_norm` 的中位数；若 `level_shift_norm` 中位 **> 0.10** | 则 Layer 2 的"无差别"**不得**全部归因于"状态依赖无关"——水平偏移也是干预的一部分。主判据 P1 不受此影响（`grad_cos` 是**实际梯度方向**，形状差与水平差都已包含其中） |

**次级（描述性，不作判决）**：

- S1：干预跑 9 点锚点序列的 **min** 与基线 3 跑 min 分布 {**0.125, 0.200, 0.250**} 对比；
- S2：干预跑 **末点（20k）** 与基线 3 跑末点 {**0.675, 0.475, 0.3625**} 对比；
- S3：干预跑自身 `vstd/rstd`、`h_std`、`n_abs`、EV 序列与基线同点对比（只描述）。

> **基线数字全部脚本复算**（R4）：`src/clasher_new/runs/d1_league_20k{,_r2,_r3}/solo_state.json`
> 的 `_controls_history` 里 `vs=="baseline_rand"` 的 9 个点，见本文件 §5。

---

## 3. 协议（单变量）

**基线（对照）**：`d1_league_20k{,_r2,_r3}` 既有 3 跑，配置如下（逐项从 `config.json` 读出）：

```
mode=solo  config=economy  fresh  无 main_init
total_steps=20000  steps_per_eval=2500  n_eval_games=40  eval_workers=12  seed=0
value_norm=running  adv_norm=scale  ppo_epochs=4  ppo_minibatch=32  ppo_shuffle=1
gamma=0.997  gae_lambda=0.99  vf_coef=0.5  ent_coef=0.01  lr=3e-4
update_interval=128  batch_size=128  solo_copy_every=2000
value_bypass=1  value_independent=1
hist_seed_dirs=[runs/economy_9k_ft, runs/economy_9j]
```

**Layer 1 跑**：上述配置 **+ `--adv-inert-probe --diagnose-every 1`**（纯日志），
`--steps-per-eval 20000 --anchor-every 2500`（**只改评估节奏以省墙钟，不改训练语义**；
锚点的构造与全点路径完全相同 —— 同一个 `eval_control(..., RAND_ANCHOR_EVAL_SEED)` 调用，
且 `baseline_rand` 是固定独立模型、不受 `_sync_controls_once` 影响）。
> `--diagnose-every 1` 只是把主判据的**样本数**从 ~8 提到 ~78（每个更新一个 `grad_cos`）；
> 多出的 `autograd.grad` 不写参数、不消耗 RNG ⇒ 训练轨迹不变（单测 ① 已固化：
> 无论 diag 是否命中、无论 `adv_alt` 取何值，训练后参数逐位相同）。
> 评估节奏不影响训练轨迹的论证：评估在 spawn 的 worker 进程里跑，父进程训练环的 RNG
> （`ppo.rng` / env `_rng`）不被 eval 触碰；`_persist` 不消耗 RNG。**（该论证为推理，非实测，见 §4 限制 3。）**

**Layer 2 跑**：Layer 1 配置 **+ `--critic-baseline const`**（可同时带 `--adv-inert-probe` 复用测量）。

---

## 4. 限制与混淆项（必须在判读时写明）

1. **`grad_cos` 是有偏样本**：只在每 10 次更新的第 1 个 minibatch 上测，且是该轮第一个小批
   （ratio 尚≈1）。它测的是"更新方向的**首步**差异"，不是整轮更新的差异。⇒ 只用于**判量级**，不用于精确归因。
2. **常数水平混淆（Layer 2）**：干预把 `V` 换成 c，同时改掉了"状态依赖"与"水平"两件事。
   `adv_level_gap` 是在 Layer 1 里量化该混淆的手段；若 Layer 1 显示 gap 大，Layer 2 的"无差别"
   就不能全归因于"状态依赖无关"。
3. **评估节奏不变性未实测**：Layer 1/2 用的 C 式节奏（1 全点 + 8 轻点）与基线的 9 全点不同；
   "训练轨迹不受评估节奏影响"目前是推理。（若判读出现异常，先按此条排查。）
4. **X10**：锚点每评估点重抽 40 局 ⇒ 与基线**未配对**，差分 SE=√2×0.078=0.110，不得按配对读。
5. **R5**：同 seed 同代码两次 run 也不可复现 ⇒ 3 跑/臂仍是小样本，S1/S2 只作描述。
6. **探针自身开销**：每诊断更新多 1 次前向 + 2 次反传（仅策略损失梯度），
   与既有 `diagnose_every` 的梯度分解同量级；不改变任何被写入梯度的量。

---

## 5. 基线数字（脚本复算，R4；取自各 run `solo_state.json._controls_history`）

| step | `d1_league_20k` | `_r2` | `_r3` |
|---|---|---|---|
| 0 | 0.625 | 0.625 | 0.625 |
| 2500 | 0.400 | 0.500 | 0.750 |
| 5000 | 0.125 | 0.500 | 0.500 |
| 7500 | 0.4625 | 0.2625 | 0.750 |
| 10000 | 0.550 | 0.875 | 0.825 |
| 12500 | 0.600 | 0.200 | 0.475 |
| 15000 | 0.750 | 0.2875 | 0.700 |
| 17500 | 0.375 | 0.2875 | 0.250 |
| 20000 | 0.675 | 0.475 | 0.3625 |
| **min** | **0.125** | **0.200** | **0.250** |
| 均值 | 0.5069 | 0.4450 | 0.5819 |

三跑合计 27 点：均值 **0.511**、标准差 **0.208**、min **0.125**、max 0.875。
零点（step 0，随机初始化 main）**6 个 run 同值 0.625**。

---

## 6. 成本

| 阶段 | 内容 | 预估 |
|---|---|---|
| Layer 1 | 1 跑 20k + 探针，C 式评估节奏 | 训练 ≈13 min + 评估 ≈11 min ≈ **25 min** |
| Layer 2 | 每 repeat 同上 | **25 min/跑** |

外部 CUDA 占用者若仍在（R1），吞吐可能降到 0.46×，届时成本按比例上升；
**若出现降级/崩溃指纹（`WinError 1455` / spawn 失败 / Traceback），按 R1 暂停留证，不自动降级。**

---

## 7. 判读后的处置分支（跑之前写死）

| 结果 | 处置 |
|---|---|
| **P1 成立**（+ P1′ 成立） | "修 critic"= 恢复信用分配，是**唯一**能提高样本信息量的杠杆；按 cause_analysis §7 实验 2 做（去末端 LN / 改目标归一化），并先把探针常驻化 |
| **P2 成立** | §5.3 因果链作废，O2 退回"训练过程未解"；**改奖励/拆价值头的老路重新开放**，且优先查"critic 为什么提供的是坏方向"（余弦接近 −1 属"帮倒忙"，需单列判读） |
| **P3 灰区** | 不判决；先加长到 100k 的诊断点密度（不改结构）再判 |
| **P1 成立但 P1′ 不成立** | 先修代理（改 `c` 的定义：用 V 的实际均值或逐局最优常数），重跑 Layer 1 |

---

## 8. 落地清单（改了什么）

- `rl/ppo.py`：`update(..., adv_alt=None)` —— 新增可选参数，**默认 None = 逐位旧行为**（R2）；
  仅在 `diag_on` 命中的更新上多算 1 次前向 + 1 次反传，用于 `grad_cos`。
- `rl/config.py`：`adv_inert_probe: bool = False`、`critic_baseline: str = "value"`（= 旧行为）。
- `rl/run_league.py`：`--adv-inert-probe`、`--critic-baseline {value,const}`。
- `rl/train_solo.py`：两处 GAE 调用点旁边算 `adv_const`；探针累积 + 落盘；
  `critic_baseline=="const"` 时用 `adv_const` 作 `transitions[*]["adv"]`。
- `rl/selftest.py`：默认路径逐位不变、探针不改变被训练的优势、`grad_cos` 在 `adv_alt` 与 real 相同时 =1.000。
