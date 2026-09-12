# critic「常数预测器」判别实验（2026-09-12）

> 目的：判别 critic 恒为"常数预测器"（EV≈0、RMSE≈std(R)、corr(v,R)≈0）是
> **程序侧**（代码/实现）还是**训练过程**（数据/动力学/样本量）导致。
> 背景：enc_ln/grid_ln 已修复"数值常数"形态（value std 2.6e-5→0.156），
> 残余是"对中的常数"（bias −3.56→−0.56，但拟合度仍 0）。

## 实验 3：静态审计（GAE/terminal/早停标签路径）——已完成

审阅 `rl/ppo.py::compute_gae` 与 `rl/train_solo.py` 主循环结算路径：

| 检查项 | 结论 |
|---|---|
| GAE 递推公式（delta/gae/returns） | ✅ 正确（dones/truncated/last_value 分支齐全） |
| 截断（max_ep_steps）bootstrap | ✅ P1-7 已修：`ep_trunc[-1]=True` + `main.value(末状态)`，仅真平局（virt=None）时启用 |
| 早停路径（`_stall_probe` early） | ⚠️ 候选：把未终止局当"价值 0 终端"处理（`last_value=0.0`、无 bootstrap）——设计意图可辩护（早停=按 timeout_winner 裁定终结），但早停局占比高（eval 侧随机对局 28~40%），其裁定标签噪声大 |
| 早停裁定（timeout_winner 按塔血/皇冠差） | ⚠️ 候选：与"策略实力"关系弱（A′ 结论 4），这些局的 return 标签是"裁定噪声" |
| value_norm/ReturnScaler 口径 | ✅ 已审计：last_ev_pairs 未缩放、v/p 比值正常 |

**结论：未发现致命 bug；两个候选都指向"标签噪声/可预测性"（偏训练过程侧），
程序侧无实锤。**（早停处理是程序定义，但后果是数据侧。）

## 实验 2：可预测性探针（post-LN enc → GAE return）——设计就绪，待跑

工具：`scripts/diag_critic_ev.py --probe`（已加；合成数据冒烟 PASS：
线性 R²=0.998 / 噪声 R²≈0 / 非线性 lin≈0、mlp≈0.98）。

```bash
cd src/clasher_new
PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe \
    ../../scripts/diag_critic_ev.py --ckpt runs/e2_rand_anchor_20k/solo_main.pt \
    --run-dir runs/e2_rand_anchor_20k --games 100 --device cuda --probe \
    --save-npz ../../docs/_diag_probe_e2.npz
```

判别逻辑：
- **探针 R² ≫ critic EV_global** → 信息在表征里、网络没吸收（程序侧/优化侧）；
- **探针 R² ≈ critic EV_global ≈ 0** → 标签在该表征下就不可预测（训练过程侧数据）；
- **线性≈0、MLP>0** → 关系非线性，critic 含非线性 head 应能学（学不到偏优化/欠训练）。

对照 ckpt：`runs/fix_gru_ln_norm_20k/solo_main.pt`（同探针跑一遍，看跨 run 一致性）。

## 实验 5：探针样本量补足（diag_critic_ev 24→100 局）

`--games 100`：局均 corr(局均 v,局均 R) 的 1σ 从 ~0.22 降到 ~0.10，
让"信息在不在 h 里"可显著化。与实验 2 共用一次 rollout（--probe 一起跑）。

## 实验 4：合成标签对抗检查（可选，需拍板）

把回报替换为确定性可预测函数（如塔血差）短训，看 critic 能否拟合：
能 → 程序链路（梯度/头/GAE）OK，问题在真实标签；不能 → 程序侧 bug。
**高侵入（需临时改训练奖励），暂缓，先看实验 2/5 结果。**

## 实验 1：长跑判别（欠训练 vs 学不动）——需拍板

同一程序、同协议跑 60k~100k+ 看 EV/value_std_ratio 是否爬升。
成本 ~2.5~5h；**若实验 2 显示标签不可预测，长跑大概率无意义**，故排在探针之后。

## 实验 2/5 结果：可预测性探针（50 局 × 2 ckpt，2026-09-12）——已完成

工具 `scripts/diag_critic_ev.py --probe`（post-LN enc → GAE return，留出 20%；
线性 OLS + 小 MLP）。**两个独立 ckpt 结果一致**：

| ckpt（50 局） | 线性探针 R2 | MLP 探针 R2 | critic EV_global | R std / v std |
|---|---|---|---|---|
| `e2_rand_anchor_20k` | +0.111 | **+0.242** | −0.219 | 5.43 / 0.117 |
| `fix_gru_ln_norm_20k` | +0.104 | **+0.306** | −0.001 | 11.05 / 0.219 |

（3 局初步值 0.83 是 MLP 过拟合假象——train 仅 1041 帧；50 局 20524 帧可信。）

**判读（首次定量分离"程序侧 vs 训练过程侧"）**：

1. **表征里有真实可预测信息，且主要是非线性**：线性上界 ~0.10~0.11、
   非线性（MLP）上界 ~0.24~0.31，两 run 一致 ⇒ 这是该训练分布下
   `post-LN enc → GAE return` 的本质可预测性，非单 run 噪声。
2. **critic 完全没有吸收这部分信息**：两个 ckpt 的 critic EV 都 ≤0
   （−0.22 / −0.001，即"不优于常数预测"），而同一批帧用 MLP 可达 +0.24~0.31。
   ⇒ **"信息在表征里、网络没吸收"成立**（至少 24~31% 可解释方差被浪费）。
3. **大部分回报方差在 enc 里不可预测**（~69~76%）：支持"标签噪声大 /
   自对弈对称性使状态→回报部分不可预测"（训练过程侧数据责任）。
4. **结论：两者并存、方向相反，已定量分离**——
   - 程序侧/优化侧：critic 未吸收可预测信息（0.24~0.31 被浪费）；
   - 训练过程侧数据：不可预测部分占多数（~70%+），即使网络完美也够不到高 EV。
   - 两 run 的 `v std`（0.117 / 0.219）≪ `R std`（5.43 / 11.05）再次佐证 critic 输出
     波动远小于回报波动。

**Caveat**：
- enc 是 CNN+scalar+hand+plan+belief 融合后的**压缩特征**，全局可预测上界（用原始
  obs/belief 全特征）可能更高 ⇒ 0.24~0.31 是"以 enc 为特征"的下界性上界；
- 探针是**监督式纯回归**，critic 是 **TD/GAE 自举的 on-policy 联合训练**（更难收敛），
  两者不可直接等比 ⇒ 0.24 不代表"训练后 EV 应到 0.24"，但证明监督信号本身可达。

**下一步判别候选**（需拍板）：
- **A（已执行，见下节）critic 监督微调**：用收集的 (enc, R) 对单独监督训练 value_head
  几百步，看 EV 能否到 0.2+。能 → 程序链路 OK、是 on-policy 联合优化/欠训练问题
  （训练过程侧优化）；不能 → 价值头结构问题（程序侧）。
  **实测结果：+0.054（峰值 +0.128），未到 0.2+ ⇒ 链路通但吸收有瓶颈（见实验 A 节）。**
- **B 长跑 100k**：同程序看 EV 是否向探针上界爬升（判别欠训练 vs 结构，成本高）。
- **C 数据侧**：提高对手多样性 / 降低早停裁定噪声（E2′ 多锚点等）提高可预测上界。

## 实验 A：critic 监督微调判别（e2 ckpt，50 局，4 epoch）——已完成（2026-09-12）

命令：`diag_critic_ev.py --ckpt runs/e2_rand_anchor_20k/solo_main.pt --run-dir runs/e2_rand_anchor_20k
--games 50 --device cuda --finetune --ft-epochs 4`（日志 `docs/ft_e2.log`）。
数据：50 局 / 20012 帧（均局 400.2 帧）；局级留出 **train 40 局 16271 帧 / test 10 局 3741 帧**；
逐帧 (obs,tok,plan,hidden_prev)→GAE return 的 value MSE，Adam lr=3e-4，全网络梯度、hidden detach
（与训练前向完全一致）。本 run 未带 --probe，探针数字沿用实验 2/5 同 ckpt（线性 +0.111 / MLP +0.242）。

| 阶段 | test EV（10 留出局） | train MSE |
|---|---|---|
| 监督前 critic | −0.1386 | — |
| epoch 1 | +0.0148 | 27.68 |
| epoch 2 | +0.0005 | 10.57 |
| epoch 3 | **+0.1283** | 4.64 |
| epoch 4 | +0.0537 | 2.47 |

（critic 对照：EV_global −0.1571、RMSE 6.68 ≈ std(R) 6.21、v std 0.116 ≪ R std 6.21。）

**判读**：

1. **程序链路（前向+梯度）是通的**：监督微调把留出局 test EV 从 −0.139 拉到 +0.054
   （峰值 +0.128）——"恒定预测器"变成"微弱正预测器"。冒烟（2 局 1 epoch −0.81→+0.05）
   与正式 50 局方向一致 ⇒ **排除"价值头/GRU 结构坏到梯度到不了"的强程序侧假设**。
2. **监督吸收远低于探针上界**：+0.05~0.13 vs MLP 探针 +0.24。即便纯监督、零 on-policy
   干扰、无 TD 自举，4 epoch 也只到探针上界的一半以下 ⇒ 不能把"critic 吸收不了"全归
   训练过程侧；**共享 trunk 价值通路存在实质吸收瓶颈（程序侧证据增强）**。
3. **epoch 3 后 test EV 回落（+0.128→+0.054）而 train MSE 继续降（4.64→2.47）⇒ 过拟合
   开始**；16k 训练帧下 4 epoch 未收敛但泛化已见顶，无脑加 epoch 大概率不解决。
4. **探针 vs 监督架构差**：探针是 enc 直连回归（无 GRU、逐帧独立），监督走完整 GRU 时序
   + 共享 trunk。+0.054 可能是"GRU 通路真实可达水平"，0.24 是探针专有乐观上界。要区分
   "欠训练"还是"GRU 瓶颈"需 A′/B′。

**合成结论（与实验 2/5 拼接）**：信息在 enc（24~31% 非线性可预测）→ **critic 吸收效率低**：
on-policy EV≤0，纯监督也只到 5~13%。不再是"程序侧 bug"或"标签全噪声"单边结论——
**吸收瓶颈（程序侧）与标签噪声（数据侧）同时为真**，且新证据把"吸收低"从"优化没学好"
推向"价值通路结构/容量"。**下一步候选**：

- **A′** 长监督收敛曲线（10~20 epoch + wd/lr 调度），看 test EV 是否逼近 0.2+（区分欠拟合 vs 结构瓶颈）；
- **B′** bypass-GRU 监督对照（value_head 直连 enc 回归），直接量化"GRU 是不是瓶颈"；
- **C′** 数据侧降噪（early-stop 裁定标签）仍是独立可并行方向。

> ⚠️ **复测修正（B′ 同跑时发现，2026-09-12）**：同一程序同一配方在**另一次 rollout**
> （GPU 非确定性，20803 帧 vs 本实验的 20012 帧）上，全路径监督 **1 epoch 即达 test EV
> +0.2174**（>0.20 提前停），并非本实验的 +0.054（4 epoch 含过拟合回落）。⇒ 本节的
> "+0.05~0.13 远低于探针上界"不是稳定结论，**run 间方差大（0.05~0.22）**；"吸收瓶颈
> （程序侧）证据增强"应降级——GRU 通路净损耗仅 ~0.08，见实验 B′ 节。

## 实验 B′：bypass-GRU 监督对照（e2 ckpt，50 局，4 epoch）——已完成（2026-09-12）

命令：`diag_critic_ev.py --ckpt runs/e2_rand_anchor_20k/solo_main.pt --run-dir runs/e2_rand_anchor_20k
--games 50 --device cuda --finetune --bypass --ft-epochs 4`（日志 `docs/ft_bp.log`；新增 `--bypass`）。
与实验 A 共享同一次 rollout（20803 帧，40 训/10 测局）；三变体与全路径**同 split、同配方**
（lr=3e-4、逐帧 value MSE、Adam、局级留出 seed 0），唯一差异 = 输入通路：

| 通路 | 监督前 test EV | 监督后 test EV | 备注 |
|---|---|---|---|
| lin-joint（trunk 可训，**无 GRU**） | −0.0896 | **+0.2936** | 1 epoch 即 >0.20 提前停 |
| 全路径（带 GRU）——实验 A 复测 | −0.0079 | **+0.2174** | 1 epoch 提前停（上次 run +0.054/+0.128） |
| lin-frozen（Linear 头，enc 冻结） | −0.0591 | +0.0015 | MSE(t) 47.0→46.9，连均值都没拟合 |
| mlp-frozen（MLP 头，enc 冻结） | −0.0868 | +0.0001 | 同上（探针配方含标准化+lr1e-3，本变体未标准化） |

（探针 MLP 上界参照 +0.242。）

**判读**：

1. **GRU 不是主导瓶颈**：同 run 内去掉 GRU（trunk 可训）EV=+0.294，带 GRU 全路径 EV=+0.217
   ——两者都超过/接近探针上界。GRU 净代价 ≈0.08 EV（次级因素），远非"信息被 GRU 摧毁"。
2. **主导因素是 trunk 可训性（特征条件化）**：frozen enc 上连 MLP 头都学不动（~0.00），
   trunk 一可训立刻跳到 0.22~0.29 ⇒ 探针"0.24 上界"是冻结特征 + 标准化 + 3 层 MLP + lr1e-3
   配方下的下界性上界；训练 trunk 可超它（0.29）。
3. **frozen 变体失速是配方假象**：未标准化 enc 各维方差悬殊（LN 只做帧内归一），lr=3e-4 下
   新鲜头连均值都拟合不了；探针可达性依赖标准化预处理——"信息在 enc"需重新定性为
   "信息在，但需条件化/训练才能取用"。
4. **run 间方差大（重要教训）**：本次 rollout 20803 帧 vs 实验 A 的 20012 帧（同 ckpt 同 seed）
   ——GPU 非确定性导致轨迹不同；全路径监督从 +0.054 漂到 +0.217 ⇒ **跨 run 数字不可直接比，
   判读只用 run 内对照**。

**合成结论（修订实验 A 的表述）**：

- **"critic 结构吸收不了"的强程序侧假设被否定**：同一带 GRU 网络监督配方下 test EV 可达
  +0.22（本次），结构/梯度链路完全够用；
- 实验 A 的 +0.054 是**坏轨迹 + 4 epoch 过拟合回落**的组合，不是稳定上限；
- **主因回到训练过程侧**：on-policy PPO 联合训练（TD 自举/GAE 标签噪声/trunk 梯度冲突）
  下 EV≤0，而监督训练同网络 0.22——差距在训练过程，不在结构；
- 次级发现：GRU 通路有 ~0.08 EV 损耗 ⇒ **价值头直连 enc（bypass）是低成本可试的架构改进**；
- 数据侧（early-stop 裁定噪声等）仍是独立责任项（探针 ~70% 方差不可预测）。

**实现备注**：`--bypass` 三变体共享一次冻结 enc 预计算（pristine trunk deepcopy，不动
实验 A 的原模型）；`lin-joint` 每 epoch 重算 enc（trunk 可训）；判别自动打印汇总。

## 落地：B′ → 正式架构 + C′ → 训练侧标签降噪（2026-09-12）

用户拍板顺序「先 B′ 落地，随后 C′」。详细协议/判读口径见
`docs/rl_training_fix_plan_v3.md` §3.10。

**B′ 落地（`value_bypass`）**：`FollowerPolicy(..., value_bypass=False)`；bypass 时
`value = value_head(enc)`（跳过 GRU），策略头仍走 GRU 隐状态。5 处 value 计算点全改
（`act`/`act_parallel`/`evaluate`/`evaluate_batch`/`value`）；ckpt 元数据带该标志并在
不一致时告警；`TrainConfig.value_bypass`（默认 False=旧行为）+ **economy 预设 True**；
CLI `--no-value-bypass`。标志传播到 train_solo / run_league / flow_league / league 快照。

**C′ 落地（`stall_draw_margin=0.05`）**：早停局结算用新的 `settle_stall`——皇冠不同或
塔血%差 ≥ margin → 决定性 ±胜负；皇冠相同且差 < margin → **记平局=失败**（去掉掷硬币级
胜负标签）。**只改训练侧**（eval 保留真实 CR 规则，保证评估可对比）。新增
`stall_games`/`stall_close_draws` 计数落盘，用于量化 C′ 实际生效比例。

**验证**：官方全量 selftest 94 项 100% PASS（新增 `test_value_bypass`：同权重下 GRU 通路
vs bypass 通路 value 不同、元数据往返、告警不崩；`test_stall_settlement_margin`：决定性
保留 / 细差记平局 / 边界正确）。冒烟 `runs/smoke_bypass_cprime`（2000 步）0 Traceback、
GRU 仍活（h_std 0.042~0.068）。20k 验证跑 `byp_cprime_20k` 进行中
（日志 `docs/train_byp_cprime_20k.log`）。

## 运行约束

- 等训练速度基准（job bash-9）完成后才跑引擎类实验（否则污染基准数字）。
- 判读纪律：探针与 critic 必须用**同一批 rollout 帧**（--probe 内置）。
