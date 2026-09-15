# 预注册：plan / belief 辅助通道的「梯度死活」判定（P1）

日期：2026-09-14
分支：`channel-gradient-p1`
上游：`docs/tool_usage_audit_2026-09-14.md`（工具接线与使用审计）
目的：回答「模型没学会用 plan/belief，是**梯度精确为零（吸收态）**还是**梯度活着但信息用不上**」
——这个区分决定后续所有动作（加量 / 改结构 / 接新工具）的走向，见【R14】。

---

## 0. 先说清本预注册**不**回答什么

- **不**回答"plan/belief 通道有没有用"（那是信息侧问题，已由 `pomdp_ceiling_v2_verdict` 给出 +0.4pp 上限）。
- **不**回答"哪个架构更好"。
- **不**调参、**不**改任何训练代码、**不**新增 ckpt。
- 全部仪器**只读**（读 ckpt、跑 rollout、算梯度后丢弃）。

---

## 1. 被测量对象（脚本从 ckpt 键名取，不手抄）

`FollowerPolicy` 前向（`rl/follower.py:284-298`）：

```
grid_feat   = grid_ln(cnn(x))                      # 2560
hand_feat   = ...                                  #   40
scalar      = ...                                  #    3
plan_f      = plan_mlp(plan_v)                     #   64   <- 被测通道
belief_f    = belief_mlp(belief_v)                 #   64   <- 被测通道
fused       = cat([grid_feat, hand_feat, scalar, plan_f, belief_f])   # 2731
enc         = enc_ln(relu(enc_fc(fused)))          #  128
```

被判决的张量（4 个，全部在**前向路径**上）：

| 张量 | 形状 |
|---|---|
| `plan_mlp.0.weight` | (64, 58) |
| `plan_mlp.0.bias` | (64,) |
| `belief_mlp.0.weight` | (64, 563) |
| `belief_mlp.0.bias` | (64,) |

---

## 2. 仪器 A：参数位移指纹（【R14】，静态、秒级）

对每个 run，按 step 排序取全部 `solo_main_<step>.pt`，逐**相邻窗口**算
`‖ΔW‖ / ‖W‖`，并统计"差**逐位恰为 0**"的窗口数 `z` 与总窗口数 `n`。

原理：Adam 对 loss 的常数缩放不变 ⇒ **位移恰为 0 ⟺ 该张量梯度精确为 0**。

### 判据 A（结构判据，无阈值）

| 形式判决 | 条件 |
|---|---|
| **A-M1（通道冻结）** | 存在被测张量 `z == n`（全窗恰零） |
| **A-M2（通道活着）** | 所有被测张量 `z < n` |
| **A-UNDET** | 窗口数 `n < 3`（样本不足，不判决） |

**复现要求**：`d1_long_100k` 与至少一个 20k run 给出同一形式判决才算数；
否则写「跨 run 不一致」并**不判决**。

### 强制对照（防止"零位移 = 死"被误读）

- `enc_fc.*`：已知在策略前向 + 策略目标下被训练 ⇒ 应**非全窗冻结**。
- `value_enc_fc.*` / `value_head.*`：在 `value_independent=True` 下**不在策略前向**
  ⇒ 若全窗冻结，那是"不在前向"而非"死亡"（【R14】2026-09-13 晚补充 ③）。
  **若本 ckpt 的 `value_independent=False`，该项作废并注明。**

---

## 3. 仪器 B：真梯度反传（判决性、分钟级）

流程（全部走真实代码路径，不重写损失）：

1. `RLEnv` + `BeliefInference` + `BeliefPlanner` + `FollowerPolicy` 载入 ckpt；
2. 真 rollout（`deterministic=False`，与训练同分布）采 `N_total` 帧，按**局**分组；
3. 每局用 `PPOTrainer.compute_gae` 真 GAE（`γ`/`λ` 取自 `TrainConfig.resolve('economy')`）；
4. 构造 `PPOTrainer`（`adv_norm="scale"`、`value_norm="running"`、`n_epochs=4`、
   `minibatch=32`、`shuffle=True`，即当前主线协议），取 **第 0 个小批**，
   调 `PPOTrainer._loss_pass`（**真实损失函数**）；
5. `torch.autograd.grad(p_loss, params)` 与 `(v_loss, params)` **分别**取梯度
   （`retain_graph=True`），记录逐张量 `‖g‖`（`None` 记 0）；
6. 同一 rollout 上**重抽小批再算一次**（第 2 个样本），两个样本都为零才判死。

### 判据 B（结构判据，无阈值）

对 `plan_mlp.0.weight/bias` 与 `belief_mlp.0.weight/bias`：

| 形式判决 | 条件 |
|---|---|
| **B-M1（梯度冻结）** | 某张量在 `p_loss` 与 `v_loss` **两条**梯度上、**两个样本**上 `‖g‖` 全为 0 |
| **B-M2（梯度活着）** | 某张量至少一处 `‖g‖ > 0` |
| **B-UNDET** | 反传自检 I2/I3 未过 ⇒ 仪器不可用，**不判决** |

**量级（描述性，不参与判决）**：报告 `‖g_plan_mlp‖ / ‖g_enc_fc‖` 与
`‖g_belief_mlp‖ / ‖g_enc_fc‖`。**明确不设"够大"阈值**——理由见【R15】/【R16】：
本实验没有为该比值标定过的对照，也没有量出 run 间散布，任何阈值都会是照抄或单次观测。

### 强制自检（不过则整轮读数作废）

| 编号 | 内容 | 通过条件 |
|---|---|---|
| **I1 重放一致性** | `_loss_pass` 返回的 `ratio_mean` | `|ratio_mean − 1| < 1e-3`（transitions 与可微重放一致） |
| **I2 反传可分辨** | `enc_fc.weight` 在 `p_loss` 下 `‖g‖` | `> 0`（否则反传链路坏了） |
| **I3 前向路径对照** | `value_enc_fc.weight` 在 `p_loss` 下 `‖g‖` | `== 0`（它不在策略前向；证明仪器能把「不在前向」与「在前向但梯度为零」分开） |

---

## 4. 同步采集的**描述性**读数（为 P2 提供依据，不判决）

- `fused` 五个分块的 `‖·‖` / 跨帧 std：`grid_feat(2560) / hand_feat(40) / scalar(3) / plan_f(64) / belief_f(64)`；
- `enc_fc.weight` 五个列块的 `‖W_block‖` 及其占全矩阵的比例；
- **上游梯度分块**：`∂p_loss/∂fused` 按同样五块切开的 `‖·‖` 与占比
  —— 这是"每块到底收到多少学习信号"的直接读数（比参数位移更靠前一步）。

---

## 5. 判决互斥性与后续动作（跑之前写死）

| A/B 组合 | 结论 | 后续动作 |
|---|---|---|
| A-M1 或 B-M1 命中 **plan** 通道 | plan 通道**梯度死** | 接新工具到 plan **无效**；P3 改为把精确塔伤接到**不经过网络的路径**（规划器决策规则 / 掩码闸门） |
| A-M2 且 B-M2，但上游梯度分块占比 ≪ 1% | 通道活着但**信号被淹没** | 进 P2（融合层尺度 / 逐块学习率 / 归一化），**先改尺度再谈接工具** |
| A-M2 且 B-M2 且分块占比可观 | 通道既活又有信号 | 说明"没学会用"是**优化动力学**问题（非结构）⇒ P3 可以直接接工具进 plan 通道，并预注册"接后 plan 通道梯度变化"判据 |
| 0 个被测张量被判死但 `belief_mlp` 与 `plan_mlp` 结论**相反** | 两通道机制不同 | 分别处置；**不得**用一个通道的结论推广到另一个 |

---

## 6. 明确不做（【R11】）

- 不改 `rl/` 下任何文件；不动 ckpt；不新增 run 目录；
- 不因本判读就扩模型参数 / 上 MCTS / 改奖励 / 拆价值头；
- 不用单次观测（单 rollout 单小批）下"梯度小"的结论 —— 判"小"需要标定，本预注册**不做**。
