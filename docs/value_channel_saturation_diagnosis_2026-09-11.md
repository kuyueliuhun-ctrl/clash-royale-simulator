# 价值通道失效根因诊断：GRU 输入饱和（2026-09-11）

> 结论一句话：**critic 恒为常数不是价值头坏了，而是它的输入 `h`（GRU 隐状态）
> 在数值上被冻成了常数**——`enc = relu(enc_fc(fused))` 的量级高达 **533**（其中
> CNN 输出 `grid_feat` 常数分量 ≈ **468**），把 `GRUCell` 的 tanh 候选压到
> `abs_mean≈0.994`（完全饱和）。后果是 **整个神经网络看不见局面**：
> critic 退化为常数预测器（EV = −bias²/Var(R) < 0 可精确复现），
> 策略的动作分布也只剩手工 `BeliefPlanner` 的偏置在起状态依赖作用。
>
> 该病理**跨全部历史 run 系统性存在**。此前 v2 计划里"EV 持续为负 → 怀疑奖励尺度/
> 价值头"的分支判断**病因判错了**。

---

## 1. 现象与原始判据

`prod_200k_valnorm_ev` 跑到 step 5 万，`solo_state.json` 里 EV 全负：

| step | 0 | 10k | 20k | 30k | 40k | 50k |
|---|---|---|---|---|---|---|
| EV（dashboard 显示） | – | -4.91 | -3.74 | -3.37 | -4.26 | -2.26 |
| 胜率 | – | – | – | – | 0.425 | 0.600 |

日志逐批 `vraw`(= 原始 MSE) 波动极大（1.2 → 766），远超全局回报方差（`ret_scaler` 给出
Var(R)≈111.7, std≈10.6）。

## 2. 取证方法与脚本

| 脚本 | 作用 |
|---|---|
| `scripts/diag_critic_ev.py` | 用 ckpt 在真环境 rollout，逐帧收集 (value, GAE return)，同时算 5 种 EV 口径 |
| `scripts/diag_value_head.py` | 解剖各模块参数量级 + h/enc 跨帧方差 + GRU 门饱和 |
| `scripts/diag_encoder_scale.py` | 拆解 `fused` 各分量尺度，定位"把 encoder 顶起来的元凶" |
| `scripts/diag_gru_ablation.py` | 对照实验：raw / 去常量 / 标准化 enc 喂 GRU，测 h 跨帧方差 |

全部用 `runs/prod_200k_valnorm_ev/solo_main.pt`（step 50000）+ 该 run 的 `config.json`
实际超参（gamma=0.997, lam=0.99, max_ep_steps=360, batch=128）。

## 3. 证据链

### 3.1 critic 输出是常数（2 局 743 帧的 rollout）

```
R (GAE 回报): mean=7.671 std=5.503 min=0.57 max=27.74
v (critic)  : mean=3.465 std=0.028 min=3.40 max=3.54     ← 几乎不动
```

`bias = E[v−R] = −4.206`。**恒定预测器的 EV 恒等式**：

```
EV_const = 1 − (Var(R)+bias²)/Var(R) = −bias²/Var(R)
         = −4.206² / 30.284 = −0.584
```

实测 `EV_global = −0.5817`。**精确吻合** → critic 就是常数预测器。

### 3.2 EV 判据本身也被"连续批"污染（测量问题，独立于模型问题）

同一批帧，5 种口径：

| 口径 | EV |
|---|---|
| EV_global（全体帧池化） | **−0.58** |
| EV_batch_contig（训练口径：128 连续帧/批，取均值） | **−1.74** |
| EV_batch_shuffle（随机 128 帧/批） | −0.57 |
| EV_episode（每局一批） | −0.63 |

相邻帧 `corr(R_t, R_{t+1}) = 0.990`，批内 Var(R) 只有全局的 **0.32 倍**。
即：**训练用的"128 连续帧批"把 EV 系统性放大约 3 倍**（−0.58 → −1.74）。
dashboard 上那个 EV = 各更新批 EV 的均值（`train_solo.py:1037-1038`），
它是"批内 EV"，不是 critic 的真实解释力。

### 3.3 逐层定位：enc 量级 533，GRU 饱和

`scripts/diag_value_head.py`（250 帧）：

```
h 跨帧 每维std: mean=0.000026      ← 冻死
||h||: 9.1098 ~ 9.1104             ← 数值上恒定
value_head(h): std=0.000051
GRU: z=0.352  r=0.603  n(候选) abs_mean=0.994   ← tanh 完全饱和
value_head 权重正常（||w||=0.86, rms=0.076）    ← 不是死头
```

`scripts/diag_encoder_scale.py`（60 帧，各分量 L2 范数）：

| 分量 | ‖·‖ 均值 | 跨帧 std |
|---|---|---|
| **grid_feat（CNN 输出）** | **468.4** | **1.06** |
| scalar (elixir/time/next) | 17.2 | 8.77 |
| hand_feat | 5.6 | 0.50 |
| plan_f | 1.36 | 0.15 |
| belief_f | 0.97 | 0.21 |
| **fused**（拼接） | **468.9** | 1.38 |
| enc（= relu(enc_fc(fused))） | **533.1** | 1.77 |

`obs['grid']` 本身很稀疏（mean 0.014，max 10），CNN 输出几乎是**常数偏置**
（相对波动 1.06/468 ≈ 0.2%），却带着 468 的量级。GRU 的 `Wn·enc` 因此巨大，
`tanh` 饱和 → 候选 `n` 恒定 → `h = (1−z)·n + z·h` 收敛到一个与状态无关的固定点。

### 3.4 因果反证：把 enc 归一化，GRU 立刻解冻

`scripts/diag_gru_ablation.py`（10 局 2759 帧）：

| variant | ‖enc‖ | **h 跨帧 std** | h_norm |
|---|---|---|---|
| A_raw（现状） | 530.9 | **0.000005** | 0.00007 |
| B_center（enc − 运行均值） | 8.0 | **0.116** | 0.802 |
| C_standardized（标准化） | 7.4 | **0.114** | 0.705 |

跨局层面同样：A 的"局均 h 差"范数 std = 0.00008，B/C = 0.20 / 0.19。
**相隔 4~5 个数量级** → 信号是被量级压死的，不是输入里没有信息。

> **诚实边界**：同实验里"线性探针 h→R 的留出 EV"在 10 局样本下三组**均为负**
> （A −0.66 / B −0.56 / C −0.81），因为留出测试集只有 3 局、局间回报方差主导。
> 因此**不能**据此宣称"修好后 critic 能达到 EV≈0.6"。本实验只能证明
> "GRU 解冻"，修复后的实际上限需要重新训练后才能测。

### 3.5 系统性：全部历史 run 同一病理

| run | h 跨帧 std | value_head 输出 std | GRU n(abs) |
|---|---|---|---|
| economy_9k_ft | 0.0031 | 0.0093 | 0.996 |
| economy_9j | 0.0022 | 0.0235 | 0.998 |
| ab_valnorm_20k | 0.0027 | 0.0100 | 0.993 |
| economy_10e | 0.0016 | 0.0103 | 0.979 |
| prod_200k | 0.000026 | 0.00005 | 0.994 |

`n(abs)` 全部 ≥0.98 → **没有一个 run 的 GRU 是活的**。

## 4. 影响面（重要）

由于 `h` 恒定：

- `value_head(h)` 常数 → critic = 常数预测器 → **EV 结构性为负**（与奖励尺度、
  vf_coef、value_norm 均无关）。
- `slot_head(h)` / `cell_head(h)` 的输出是**常数向量**；策略的全部状态依赖
  **只来自** `_plan_biases(plan_token)`（手工的 `BeliefPlanner` 偏置）与动作掩码。
  即：**神经网络在策略里基本是开环的**，此前的"行为爬坡"主要是手工规划器在打。
- 梯度经饱和 tanh 传回 CNN 时接近 0（梯度消失），网络**无法自救**——这解释了
  为什么 5 万步毫无改善且越训越饱和（200k 的 h std 比旧 run 还小一个量级）。

## 5. 修复方案（P0）

### P0-A（模型，根因）在 encoder 输出后加归一化

`rl/follower.py`：

```python
# __init__
self.enc_ln = nn.LayerNorm(hidden)          # 新增

# _encode / _encode_batch 的返回
return self.enc_ln(torch.relu(self.enc_fc(fused)))   # 原为 return torch.relu(self.enc_fc(fused))
```

理由：LayerNorm 逐样本沿 hidden 维归一化，输出量级恒为 O(√hidden)，与常数偏置
方向无关，直接消灭 533 → GRU 不再饱和；且逐样本可用（act 的 batch=1 也成立）。

补充（可选，验证后决定）：把 GRU 更新门偏置初始化到 −1
（`gru_cell.bias_ih[hidden:2*hidden] = -1`），让 h 偏向"更新"而非"保持"。

**兼容性**：架构变了 → 旧 ckpt **不能直接续训**（`solo_state`/`solo_opt` 里的
GRU/enc 权重是按饱和态学的）。需**从头训练**，`main_init` 只用于对照不作为初始化。

### P0-B（测量）EV 改为"池化口径"

`train_solo.py` 现在把各更新批 EV 求均值进 history——口径本身把 EV 拉负 3 倍。
改为：在每个评估窗口累积原始 `(v, R)` 数组，评估时对**全体帧**算一次 EV
（或随机打乱后分批）。同时把 `h 跨帧 std`、`GRU n(abs)`、`value_head 输出 std`
加入周期日志 / dashboard，让"GRU 是否活着"成为一线可见指标。

### P0-C（诊断常态化）把三个脚本纳入 selfcheck

`scripts/diag_value_head.py` 的门槛：`h 跨帧 std > 0.02` 且 `GRU n(abs) < 0.9`；
不达标即报警。避免以后再用"EV"这种间接量去猜。

## 6. 修复验收门槛（训练 5k 步即可判）

| 指标 | 现状 | 目标 |
|---|---|---|
| h 跨帧 std | 2.6e-5 | **> 0.05** |
| GRU n(abs_mean) | 0.994 | **< 0.9** |
| value_head 输出 std / 批内 R std | ~0.001 | **> 0.3** |
| EV（池化口径） | −0.58 | **> 0，且随步数上升** |

## 7. 尚未定论 / 待验证

1. 归一化后 critic 的实际上限（EV 能到多少）——需重训后测，见 §3.4 边界。
2. `grid_feat` 常数分量 468 的来源（CNN 末层无归一化 + grid 稀疏）；
   P0-A 的 LayerNorm 在 enc 处已能覆盖，是否还要在 CNN 内部再加一层，待验证。
3. 归一化后 `plan_f`（norm 1.36）与 `grid_feat` 量级会反转——策略对规划的依赖权重
   会重新分配，行为可能短期退化，属预期。
