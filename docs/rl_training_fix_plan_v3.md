# 训练整改计划 v3：GRU 饱和修复（2026-09-11）

> v2 的 P0（价值通道量纲 / 指标可信化 / 对手池）已实施并跑完 20k+50k 验证，但
> **G1（EV）始终为负**。v3 的依据是同日取证
> `docs/value_channel_saturation_diagnosis_2026-09-11.md`：
> **负 EV 的根因不是奖励尺度也不是价值头，而是 GRU 输入饱和（`enc` 量级 533）
> 导致隐状态 h 被冻成常数**。v2 §4 "转奖励/偏置修复或拆价值头"的**分支判断错误**，
> 在病因纠正前不应再堆步数。

## 0. 病因（一句话）

`enc = relu(enc_fc(fused))` 的 L2 范数 ≈ **533**（CNN 输出 `grid_feat` 常数分量
≈468，跨帧 std 仅 1.06）→ `GRUCell` 的 tanh 候选饱和
（`n(abs_mean) ≈ 0.994`）→ `h` 数值恒定（跨帧 std 2.6e-5）→
`value_head(h)` 常数 → EV = −bias²/Var(R) = −0.58（与实测完全一致）。
全部历史 run 同一病理。

## 1. 行动项

### P0-A｜模型：enc 后加 LayerNorm（根因修复）

文件：`src/clasher_new/rl/follower.py`

```python
# __init__：新增模块
self.enc_ln = nn.LayerNorm(hidden)

# _encode(...) 与 _encode_batch(...) 的返回改为：
return self.enc_ln(torch.relu(self.enc_fc(fused)))
```

- 兼容：架构变更 → **旧 ckpt 不可续训**，需 `--fresh` 从头训练。
  `main_init` 不用于初始化（旧权重是饱和态学的），仅在对照评估里当基线。
- 备选（若 LayerNorm 后仍饱和）：给 CNN 输出加独立归一化 / 把 `grid_feat`
  经一层 LN；或把 GRU 更新门偏置初始化到 −1。**先只上 enc_ln 做单变量验证。**

### P0-B｜测量：EV 改"池化口径" + 加 GRU 活力指标

文件：`src/clasher_new/rl/train_solo.py`、`rl/ppo.py`、`rl/dashboard.py`

1. 评估窗口内**累积原始 `(v, R)`**，评估时对全体帧算**一次** EV（或先随机打乱
   再分批），替换现在"各更新批 EV 求均值"（该口径把 EV 拉负约 3 倍）。
2. 周期日志 + dashboard 增加：`h 跨帧 std`、`GRU n(abs_mean)`、
   `value_head 输出 std / 批内 R std`。

### P0-C｜诊断常态化

把 `scripts/diag_value_head.py` 的门槛并入 selfcheck / 启动前检查：
`h 跨帧 std > 0.05` 且 `GRU n(abs) < 0.9`，不达标直接报警。
（三个诊断脚本 `diag_critic_ev.py` / `diag_value_head.py` / `diag_gru_ablation.py`
已落地，见诊断报告 §2。）

> 口径统一（2026-09-11 修订）：门槛值以 **§2 验收表的 0.05** 为准（代码
> `rl/diagnostics.THRESHOLDS`）。本节原先写的 0.02 是笔误，现仅作 dashboard 的
> 黄色预警带（`rl/dashboard.py`，>0.02 黄 / ≤0.02 红）。

## 2. 验证协议（5k 步即可判定，不必等长跑）

命令（`--fresh`，其余同 v2）：

```
python rl/run_league.py --mode solo --config economy --config-name fix_gru_ln_5k \
  --fresh --total-steps 5000 --steps-per-eval 2500 --n-eval-games 40 \
  --eval-workers 16 --device cuda --value-norm running --adv-norm scale \
  --diagnose-every 10
```

验收门槛：

| 指标 | 现状 | 门槛 |
|---|---|---|
| h 跨帧 std | 2.6e-5 | **> 0.05** |
| GRU n(abs_mean) | 0.994 | **< 0.9** |
| value_head 输出 std / 批内 R std | ~0.001 | **> 0.3** |
| EV（池化口径） | −0.58 | **> 0**，且随步数上升 |

## 3. 分支

- **全达标** → 继续长跑（100k~200k），v1/v2 的奖励与门禁体系按原样沿用；
  critic 实际上限在此时才可测（诊断报告 §3.4 明确该问题此前无法回答）。
- **h 解冻但 EV 仍 ≤0** → 此时才是 v2 §4 的适用场景（奖励尺度 / 价值头结构），
  因为"输入无信息"这一层已被排除。
- **h 仍冻结** → 上 §1 P0-A 备选（CNN 侧归一化 / GRU 门偏置），并复查
  `enc_fc` 权重是否被重新拉大。

## 3.5 实施情况（2026-09-11 已落地）

| 行动项 | 落地位置 | 状态 |
|---|---|---|
| P0-A enc 后 LayerNorm | `rl/follower.py`（`enc_ln`；`_encode`/`_encode_batch`） | ✅ |
| P0-B EV 池化口径 | `rl/train_solo.py::eval_and_write`（窗口内累积逐帧 `(v,R)` 合并算一次；另存 `explained_variance_batched` 对照）；`rl/ppo.py::last_ev_pairs` | ✅ |
| P0-B GRU 活力指标 | 新增 `rl/diagnostics.py`（`gru_vitality`/`check_vitality`）；主循环维护最近 96 帧；评估行落 `h_std`/`gru_n_abs`/`value_std` | ✅ |
| P0-B dashboard 列 | `rl/dashboard.py` solo 表加 `h 跨帧 std`、`GRU n(abs)`（带阈值着色 + tooltip）；EV 列注明池化口径 | ✅ |
| P0-C 诊断常态化（selfcheck） | `rl/selftest.py::test_enc_layernorm_gru_vitality` | ✅ |

**实施期新增的两条实测事实（重要，写入 AGENTS.md）**：

1. **随机初始化时不饱和**。无 `enc_ln` 的 `enc` 范数仅 ~0.5（hidden=64），
   而历史 ckpt 是 533 ⇒ 量级是**训练过程中被推大**的，不是初始化就有的。
   因此回归测试的负对照不用裸随机网络，而是把 `enc_fc` 输出 ×300
   （enc 54→203、GRU |n| 0.92→0.98）确定性复现饱和。
2. **`enc_ln` 对 `enc_fc` 的全局缩放基本不变**（7.976 → 8.000，仅 eps 相对变化），
   `h_std` 在 ×300 下保持 0.21 不变 ⇒ 归一化确实吃掉了量级漂移这条通路。

**未做**：P0-A 的备选（CNN 侧独立归一化 / GRU 门偏置）—— 按计划先单变量验证；
P0-C 尚未并入启动前检查（目前只在评估点报警 + selftest 覆盖）。

---

## 3.6 5k 验证跑判读 + 分支判定（2026-09-11 第二轮）

run：`src/clasher_new/runs/fix_gru_ln_5k/`（12:37 完成，13 局，训练循环 1079.4s），
日志 `docs/train_fix_gru_ln_5k.log`。

### 3.6.1 对照 §2 验收表

| 指标 | 修复前（饱和态 `prod_200k`） | 门槛 | @2500 | @5000 | 判定 |
|---|---|---|---|---|---|
| h 跨帧 std | 2.6e-5 | **>0.05** | **0.1838** | **0.0446** | 先过线、末点又跌破（边界） |
| GRU n(abs_mean) | 0.994 | **<0.9** | **0.4556** | **0.5520** | ✅ 明确解冻（差 4~5 个数量级） |
| value_std | ~0.001 | — | 0.3607 | 0.1667 | 由常数变为有变化 |
| value_std / 批内 R std | 不可判读（`r_std` 全仓未实现） | **>0.3** | — | — | ⚠️ 本轮补齐后才可判读（见 3.6.3） |
| EV（池化口径） | −0.58 | **>0 且上升** | **−0.238** | **−0.0595** | 单调上升（4× 缩减），**未过零** |

**结论一（根因修复生效）**：§0 的病因链（`enc` 未归一化 → tanh 饱和 → h 冻结 → critic 常数）
被切断——饱和的直接判据 `n(abs)` 0.994 → 0.46/0.55，`h` 跨帧 std 从 2.6e-5 提升 3~4 个数量级。

**结论二（EV 未过期）**：−0.58 → −0.238 → −0.0595 单调上升 4×，但 5k 步只有 **13 局、
2 个评估点**，无法区分"卡在 ≤0"与"仍在爬"。严格按 §3 分支表落在第二条
（"h 解冻但 EV ≤0 → v2 §4"），但 v2 §4 的动作（拆价值头 / 改奖励 / P1-4 偏置）高侵入、
且多条自带"G1 之后"门控——**在 5k 就执行正是本计划序言批评的"病因未定就换药"**。

**结论三（行为门禁相对退化）**：`gates.json` `ok=false`——engagement 35.6 → 13.9(@2500)
→ 7.9(@5000)；ghost 3.6 → 15.4 → 17.9。混杂变量：本轮 `--fresh` 冷启动 **且未传
`--hist-seed-dir`** ⇒ 对手池退化为 frozen 0.714 / defend 0.286（`_OpponentPool` 的 hist
槽为空时的自动降级），偏离 v2 §5 的 0.5/0.3/0.2。

### 3.6.2 分支判定：**不执行 v2 §4 的高侵入项**

改为「先补全测量 + 恢复文档配比 → 跑 20k 定论」。依据：①EV 单调上升，趋势未收敛；
②§2 验收表第 3 行此前**不可判读**（3.6.3），在一个测不全的仪表盘上换药无法归因；
③v2 §4 的"拆价值头"在 v2 原文里要求 **"EV 仍 ≈0 **且** 对局多样性提高后行为指标改善"**
的联合信号，本轮两者都不成立（行为指标反而退化，且退化可由对手池偏离解释）。
④v2 §4 的"强化非镜像对手（frozen 再降）"**不需要改代码**——它就是 20k 命令里补上
`--hist-seed-dir`（见 3.6.4），属可逆的运行期变量。

### 3.6.3 第二轮补齐（本轮已落地，全部为"让 20k 可判读"的低风险项）

| 项 | 落地位置 | 说明 |
|---|---|---|
| **P0-B-2 第 3 项指标** | `rl/train_solo.py::eval_and_write`、`rl/diagnostics.py`、`rl/dashboard.py` | 从评估窗口的 `PPOTrainer.last_ev_pairs`（**未缩放**量纲，与 value_head 原始输出同尺度）算 `r_std`（池化）与 `r_std_batch`（每 128 帧批内 std 再平均 → 对应计划原文"批内"），落盘 `value_std_ratio`；新增门槛 `THRESHOLDS["value_std_ratio"]=0.3` 告警；评估行与 dashboard 新列 |
| **P0-C 启动前检查** | `rl/diagnostics.py::check_policy_architecture` | 静态护栏：`enc_ln` 缺失 / 被 `nn.Identity` 替换即报警，在 `train_solo` 建好 main 后调用。诚实口径：**启动时一帧都没有，测不了 h_std/n_abs**；真实活力仍由评估点 `check_vitality` 报警，且告警现已落盘 `stats["vitality_warns"]`（旧实现只 `print`，事后无法追溯） |
| 旧 ckpt 静默陷阱 | `rl/follower.py::load_checkpoint` | 源 ckpt 缺 `enc_ln.*` 时显式告警（原先 `load_state_dict(target)` 形状恒匹配 → 完全静默） |
| 内存上界 | `rl/train_solo.py` 主循环 | `_probe["ev_pairs"]` 超 20 万帧丢最老批（`--steps-per-eval 0` 时原先无界增长） |
| **编码崩溃（新发现，非 v3 引入）** | `rl/run_league.py::_force_utf8_stdout`、`rl/train_solo.py::_print_safe`、`rl/selftest.py::main` | GBK 控制台/管道下 `⚠️` 活力告警 `print` 抛 `UnicodeEncodeError`，被 `except Exception` 吞掉后 **except 处理器里的 `{e!r}` 又内嵌同一字符 → 二次抛错、直接崩训练**（`test_solo_resume` 实际复现）。已改为 UTF-8 兜底 + 安全 print + repr 转 ASCII |
| 计划文本矛盾 | 本文 §1 P0-C | `h_std > 0.02` → `> 0.05`，与 §2/代码统一 |

**未做（仍按计划门控）**：P0-A 备选（CNN 侧归一化 / GRU 门偏置）——触发条件是"h 仍冻结"，
本轮不成立；v2 §4 的拆价值头 / 改奖励 / P1-4 偏置——见 3.6.2。

### 3.6.4 20k 验证跑协议（`fix_gru_ln_20k`）

```bash
cd src/clasher_new
python rl/run_league.py --mode solo --config economy --config-name fix_gru_ln_20k \
  --fresh --total-steps 20000 --steps-per-eval 2500 --n-eval-games 40 --eval-workers 16 \
  --device cuda --value-norm running --adv-norm scale --diagnose-every 10 \
  --hist-seed-dir runs/economy_9k_ft --hist-seed-dir runs/economy_9j
```

逐项理由：

| 参数 | 值 | 理由 |
|---|---|---|
| `--steps-per-eval` | **2500**（8 个环内点） | 本轮唯一要买的是**曲线形状**（h_std 的非单调、EV 是否出现拐点），不是水平差；2500 栅格与 5k 跑一致 ⇒ 同 seed 下 @2500/@5000 两点应**确定性复现** 5k 跑的读数（第二个独立样本）。墙钟 76~96 min（按 5k 实测推算） |
| `--n-eval-games` | 40（不改） | 40 局地板：胜率 1σ≈0.078、接敌率 1σ≈±5pp（`ab_valnorm_20k` 判读附录）。降到 16 全噪声；升到 80 只翻倍主导成本项，而**主判据 EV/h_std 完全不受评估局数影响**（来自训练窗口） |
| `--fresh` **且不传 `--main-init`** | — | v3 §1：旧 ckpt 无 `enc_ln`、权重是饱和态学的 ⇒ 不可续训（`--main-init` 只在 `resume` 缺失时生效，`--fresh` 挡不住它，必须显式不传） |
| `--hist-seed-dir` ×2 | — | 恢复 v2 §5 的 frozen 0.5 / hist 0.3 / defend 0.2（`opp_mix` 无 CLI flag，这是唯一受支持途径）。Caveat：hist ckpt 是 pre-v3 的，会带默认 LN 出场（行为与其记录 elo 有偏），PFSP 按胜负重权 —— 换来的是 EV 能否爬过 0 所要检验的那个"非镜像对手多样性"假设 |
| 新 `--config-name` | — | 复用 `fix_gru_ln_5k` 会覆写其 `solo_state.json` / `solo_main_<step>.pt` |

**判读门槛（跑完照此读）**：

1. 主判据 **EV（池化口径）**：看**趋势**而非单点——8 个点里后 4 点均值 vs 前 4 点均值，
   以及是否出现 ≥0 的点；**配合 `value_std_ratio` > 0.3**（本轮才可判读）。
2. 机理判据：`h_std` 是否稳定 >0.05（区分 5k 的 0.184→0.0446 是噪声还是真实衰减）；
   `n_abs` 是否持续 <0.9。
3. 行为指标只读 `gates.json` 的**相对退化**报警，不读绝对水平（口径见 AGENTS 20k 判读 §5）。
4. 不判 main 曲线与单点胜率。

**分支（20k 之后）**：EV 出现 >0 且 `value_std_ratio`>0.3 → 按 §3 第一条继续堆量
（100k~200k，此时 critic 上限才可测）；h 稳定解冻但 EV 仍 ≤0 且**不再上升** →
此时才执行 v2 §4（奖励尺度 / 价值头结构）；h_std 再次跌破 0.05 且 `enc_fc` 权重量级
被重新拉大 → 上 P0-A 备选（`scripts/diag_encoder_scale.py` 定位元凶分量）。

---

## 3.7 20k 验证跑判读（2026-09-11 第二轮，`fix_gru_ln_20k`）

run `src/clasher_new/runs/fix_gru_ln_20k/`，日志 `docs/train_fix_gru_ln_20k.log`；
训练循环 2096.9s（20k 步 + 9 个评估点）、**62 局**、**0 次降级 / 0 个 Traceback**，
`opp_mix` 已按 §3.6.4 恢复为 frozen 0.5 / hist 0.3 / defend 0.2（hist ckpts=12）。

### 3.7.1 逐点读数

| step | cum局 | 胜率 | EV(池化) | h_std | n_abs | value_std | r_std(批内) | v/R std |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.450 | – | – | – | – | – | – |
| 2500 | 7 | 0.388 | −0.0734 | 0.0531 | 0.4927 | 0.1781 | 7.909 | **0.0225** |
| 5000 | 13 | 0.400 | −0.4183 | 0.0320 | 0.4459 | 0.0333 | 9.236 | 0.0036 |
| 7500 | 22 | 0.475 | −0.0876 | 0.0324 | 0.4519 | 0.0353 | 10.142 | 0.0035 |
| 10000 | 30 | 0.200 | −0.0360 | 0.0356 | 0.4681 | 0.0798 | 11.667 | 0.0068 |
| 12500 | 37 | 0.600 | −0.2057 | 0.0370 | 0.4864 | 0.0925 | 8.559 | 0.0108 |
| 15000 | 45 | 0.475 | −0.0758 | 0.0364 | 0.4994 | 0.0787 | 12.420 | 0.0063 |
| 17500 | 54 | 0.588 | −0.0482 | 0.0376 | 0.5349 | 0.0974 | 10.653 | 0.0091 |
| 20000 | 62 | 0.463 | **−0.0079** | 0.0381 | 0.5330 | 0.1110 | 9.769 | 0.0114 |

### 3.7.2 结论

1. **饱和修复成立且稳定**（这是本轮最重要的确认）：
   - `n_abs` 8/8 点 < 0.9（0.446~0.535），对照修复前 0.994；
   - `scripts/diag_value_head.py`：GRU 门 `z=0.4707 / r=0.4937 / |n|=0.5247` —— 门控与候选
     都健康，**不是饱和**；
   - `enc_ln` 确实生效且未被绕过：`||W||=11.31、RMS=0.707`（γ≈1），post-LN `||enc||=11.386`
     （min 11.385 / max 11.387，LayerNorm 应有的定长）；
   - `scripts/diag_encoder_scale.py`：**pre-LN `||enc||=115.6`，对照修复前 533（降 4.6×）**
     ⇒ §3 分支要求复查的"`enc_fc` 权重量级被重新拉大"**不成立**。
2. **但 critic 仍是有效的"常数预测器"**（§2 四条门槛里 3 条 FAIL）：
   - **`value_std_ratio` 0/8 点达标**（0.0035~0.0225 vs 门槛 0.3）——这是本轮**新增指标**
     给出的最干净判据：value_head 输出的波动只有回报波动的 **0.4%~2.3%**；
   - **EV 0/8 点过零**，max = −0.0079（末点，已贴近 0）；前 4 点均值 −0.1538 → 后 4 点
     −0.0844（**略有上升，幅度在噪声内**）；
   - `h_std` 只有 1/8 点 ≥ 0.05（0.032~0.053，后半均值 0.0373）。
3. **`scripts/diag_critic_ev.py`（24 局 / 7179 帧）的机制读数**：
   - `R: mean=5.524 std=9.747`；`v: mean=1.964 **std=0.124**`；`MSE=107.7 ≈ Var(R)=95.0`
     ⇒ `RMSE=10.38 ≈ std(R)=9.75`，即价值预测的信息量≈0；
   - `corr(v,R) 全局 = −0.0077`；**`回归斜率 R~v = −0.603`（负！）**；
   - `corr(局均 v, 局均 R) = +0.2239`、`EV_between = −0.1005` —— **弱正相关存在但
     24 局下 1σ≈0.22，不显著**（正是 §3.4 警告的"样本量不足，不得据此宣称"）。
4. **行为指标**：`gates.json` `ok=false`，起因只有 ghost_rate（3.6→**14.8** > 7.2）；
   engagement 35.6→**24.5** ≥ 17.8 **PASS**（5k 跑该项是 7.9 FAIL，本轮恢复配比后转正）。
   胜率 0.200~0.600、末 0.463，全程在 1σ=0.078 内 —— 无净漂移。
   对照组剧烈振荡（vs `baseline_prev`：0.625→0.087→0.700→0.200→0.900→0.188→…），
   幅度远超 1σ ⇒ **自对弈策略循环（cycling）指纹**，非统计噪声。

### 3.7.3 分支判定：**落在 v3 §3 表之外的新状态**，不擅自换药

v3 §3 的三行分支都不精确匹配本状态：不是"全达标"（EV 未过零），不是"h 解冻且 EV 不再上升"
（EV 后段略升、末点 −0.008），也不是"h 仍冻结 / enc_fc 被拉大"（n_abs 健康、enc 115.6）。
**新状态 = 饱和已修、GRU 已活，但 encoder 给 GRU 的输入跨帧几乎不变。**

证据：`diag_value_head.py` 报 `enc 跨帧每维 std: mean=0.00424 / median=0.00043`（post-LN
向量范数恒为 11.386）；`diag_encoder_scale.py` 报各分量尺度严重失衡：

| fused 分量 | 范数均值 | 跨帧 std | 说明 |
|---|---|---|---|
| `grid_feat` | **101.018** | 0.426 | 占 fused(102.99) 的 98%，却几乎无变化 |
| `scalar` | 17.623 | **6.971** | 相对变化最大（40%），但被 grid 淹没 |
| `hand_feat` | 6.406 | 0.501 | 被压 16× |
| `plan_f` | 1.649 | 0.199 | 被压 61× |
| `belief_f` | 0.649 | 0.104 | 被压 156× |

⇒ **推断（待单变量验证，不得当已知）**：`enc_ln` 只解决了"量级压死 tanh"，没有解决
**融合层尺度失衡**——一个范数 101、跨帧 std 0.43 的通道主导了 fused/enc，把真正在变化的
scalar/hand/plan/belief 稀释掉，于是 GRU 收到的是"几乎恒定的方向"。

**下一步候选（按机制证据强度排序，需拍板）**：

- **A（首选，即 §1 P0-A 的备选）**：`grid_feat` 或**逐分量**加归一化再 `fused`
  （CNN 输出 LN / 分量 LN），单变量、改动小、直击测到的失衡；改完需 `--fresh` 重训并
  重跑 20k 协议（§3.6.4）。
- **B（§3.4 的遗留问题）**：先把探针样本量做够（24 局 1σ≈0.22 不足以宣称"信息不在 h 里"），
  否则 A 与"状态本就不预测胜负"无法区分。
- **C（v2 §4，高风险）**：拆价值头 / 改奖励尺度——**本轮证据不支持**：`enc_fc` 未拉大、
  GRU 未饱和、EV 后段略升，且 v2 §4 的联合前提（多样性提高后行为指标改善）只部分成立
  （engagement 转正、ghost 恶化）。

**本轮不做**：不实现 A/B/C 中任何一项——v3 的纪律是"一次一个变量、先测量后换药"，
而 A 会改架构（旧 ckpt 全废），必须先有明确拍板。

---

## 3.8 P0-A 备选 A（`grid_ln`）落地 + 20k 验证判读（2026-09-12，`fix_gru_ln_norm_20k`）

用户拍板走 A（§3.7.3）。落地：`rl/follower.py` 新增 `self.grid_ln = nn.LayerNorm(cnn_out)`，
`_encode`/`_encode_batch` 在 `self.cnn(x)` 后立即归一化（**只上 grid_feat** 一个变量：
它是被测到的元凶、占 fused 范数 98%；不动 scalar——3 个语义通道被 LN 强加 sum≈0 会耦合
物理量；不动 plan_f/belief_f——量级小不是压死方）。`diagnostics.check_policy_architecture`
与 `follower.load_checkpoint` 的缺失键告警同步覆盖 `grid_ln`。selftest 增"grid 分量 LN 的
尺度不变性"（CNN ×300 后每元素 RMS 仍 ≈1；去掉 grid_ln 后 RMS→85 万复现失衡）。
官方全量 selftest 100% PASS（顺带修了一个 float32 ULP 级不可达断言：`ratio_mean` 1e-9 →
1e-5，实测偏差 1.19e-7 = 0.998 ULP）。

run：`src/clasher_new/runs/fix_gru_ln_norm_20k/`，日志 `docs/train_fix_gru_ln_norm_20k.log`；
**59 局、8 评估点、训练循环 2709.3s、0 降级 / 0 Traceback**，协议同 §3.6.4（同 seed，唯一变量 = `grid_ln`）。

### 3.8.1 逐点对照（v1=`fix_gru_ln_20k` 仅 enc_ln，v2=本 run）

| step | EV v1 → v2 | h_std v1 → v2 | n_abs v1 → v2 | v/R std v1 → v2 | 胜率 v1 → v2 |
|---|---|---|---|---|---|
| 2500 | −0.073 → −0.097 | 0.053 → 0.064 | 0.493 → 0.461 | 0.0225 → 0.0141 | 0.388 → 0.600 |
| 5000 | −0.418 → −0.126 | 0.032 → 0.038 | 0.446 → 0.503 | 0.0036 → 0.0150 | 0.400 → 0.637 |
| 7500 | −0.088 → −0.315 | 0.032 → 0.042 | 0.452 → 0.572 | 0.0035 → 0.0238 | 0.475 → 0.338 |
| 10000 | −0.036 → −0.015 | 0.036 → 0.041 | 0.468 → 0.608 | 0.0068 → 0.0126 | 0.200 → 0.475 |
| 12500 | −0.206 → −0.528 | 0.037 → 0.039 | 0.486 → 0.651 | 0.0108 → 0.0218 | 0.600 → 0.688 |
| 15000 | −0.076 → −0.089 | 0.036 → 0.041 | 0.499 → 0.706 | 0.0063 → 0.0312 | 0.475 → 0.850 |
| 17500 | −0.048 → −0.013 | 0.038 → 0.040 | 0.535 → 0.739 | 0.0091 → 0.0232 | 0.588 → 0.637 |
| 20000 | −0.008 → **+0.0002** | 0.038 → 0.040 | 0.533 → **0.745** | 0.0114 → 0.0239 | 0.463 → **0.850** |

`gates.json`（v2）：`ok=true`——engagement 10.5→**48.7**、ghost 18.7→**0.4**。

### 3.8.2 结论

1. **尺度修复达成（机制层确认）**，三份诊断（`diag_encoder_scale` / `diag_value_head` /
   `diag_critic_ev`，均跑在 `solo_main.pt` 上）：
   - `grid_feat` 范数 **101→5.87**（占 fused 从 98%→29%）；`fused` 跨帧 std **1.64→7.52**（4.6×）；
     pre-LN `enc` **115.6→4.46**；post-LN `enc` 跨帧每维 std **0.0042→0.0215**（5×，中位 0.00043→0.00125）；
     `value_head(h)` 输出 std **0.079→0.156**（2×）。`grid_ln` 的 γ≈0.707 正常（未绕过）。
2. **critic 从"偏置很大的常数"变成"对中的常数"**：
   - `diag_critic_ev`：EV_global **−0.134 → −0.0051**，与恒定预测器恒等式 `−bias²/Var(R)`
     （bias **−3.56→−0.56**）精确吻合 ⇒ 修好的是**对中**，不是**拟合**；
   - 但 **RMSE 8.74 ≈ std(R) 8.72**，`corr(v,R)=−0.007`，`value_std_ratio` 仍 **0/8 达标**
     （0.014~0.031，门槛 0.3）⇒ **价值头依旧没有真正拟合回报**；
   - 局均 `corr(v,R)`：v1 +0.223 → v2 **−0.209**（n=24，1σ≈0.21，不显著但**符号翻转**）。
3. **`n_abs` 单调上升 0.461→0.745**（门槛 0.9，饱和态基线 0.994）。注意此时输入已归一化
   （post-LN `‖enc‖≡11.47`），饱和压力来自 **GRU 自身权重/h 的增长**（h 范数 6.72→8.99、
   门 z=0.354/r=0.571），**不是输入量级** ⇒ `grid_ln`+`enc_ln` 只修了输入侧，没约束
   GRU 内部漂移。20k 内 <0.9，但趋势外推到 100k 会再超标 ⇒ **修复耐久性存疑**。
4. **`vs baseline0` 崩塌（本轮最大红旗）**：v2 从 step 5000 起对训练起点随机策略
   **0.05~0.125**（v1 同期 0.28~0.6 噪声带），同时打冻结副本（2000 步滞后）**0.85**、
   打上一评估点 **0.80** ⇒ **非传递性 / 策略循环（cycling）指纹**：自对弈无外部锚点时，
   策略可以在"打得过自己近期副本、打不过更早/随机对手"的循环里漂移。v1 无此现象，
   **与 `grid_ln` 相关联出现**。行为门禁却全绿（engagement 48.7 / ghost 0.4）——相对基线
   口径被自身起点定标，**抵消不了对随机对手的崩塌**。
5. **胜率曲线（main vs 冻结副本）末段 0.85 不能当作"训练变强"的证据**——在同一 run 里它
   与 baseline0 崩塌并存，正是第 4 条说的自引用指标失真。

### 3.8.3 分支判定（需拍板，本轮不擅自实施）

**当前状态定性**：尺度/对中修复 ✓；**价值头拟合 ✗**（EV≈0、RMSE≈std、v/R 0.03 vs 0.3）；
**GRU 耐久 ✗**（n_abs 回升）；**绝对强度可疑 ✗**（vs baseline0 崩塌）。

候选下一步（按成本/风险排序）：

- **A′ 取证 cycling（首选，零改动）**：先用现成工具确认"vs baseline0 崩塌"是否真实——
  ① 加跑 main@20000 vs 新随机策略 100 局（确认 0.125 不是 40 局噪声）；
  ② 对 `runs/fix_gru_ln_norm_20k/replays/` 做主评估局的僵局早停/出牌节奏取证
  （`scripts/forensics_response.py` 思路）——区分"绝对变弱" vs "随机对手触发早停裁定"
  vs "真循环"。**在弄清这个之前，任何"堆量/上价值头"都可能是在给一个在循环里打转的
  策略加码。**
- **B′ 修 GRU 耐久**：GRU 门/权重正则或 h 范数约束，压住 n_abs 回升趋势（单变量，
  但需再 `--fresh` 一轮 20k）。
- **C′ 价值头结构（v2 §4）**：表征已解冻、对中已修，剩余"拟合不了"才真正指向
  价值头/GAE 目标结构——但 A′ 未排除 cycling 前不启动。
- **D′ 直接堆量（100k+）**：成本最高；n_abs 趋势与 baseline0 崩塌均未解释，风险最大。

**不做**：不把 v2 的 0.85 胜率当进步写进结论；不因 EV 接近 0 就宣称"修好了"；
不在 A′ 取证前改奖励/拆价值头。

### 3.8.4 A′ 取证结果：cycling 确凿，绝对强度 ≈ 随机（2026-09-12）

工具：`scripts/_forensics_cycling.py`（只读训练产物；三对阵 ×100 局 + 自带回放参考，
worker=16，`max_ep_steps=360`）。证据全部来自 `runs/fix_gru_ln_norm_20k/`：

| 对阵（main 侧胜率） | W/L/D | 疑似早停 | 帧数 mean |
|---|---|---|---|
| main@20000 vs 冻结副本（run 自带回放 n=40） | 0.850 | 3/40 (7.5%) | 368.8 |
| main@20000 vs **baseline0**（起点随机） | **0.130** (10/84/6) | **28/100 (28%)** | 318.0 |
| main@20000 vs **全新随机** | **0.505** (50/49/1) | 2/100 (2%) | 246.7 |
| baseline0 vs 全新随机（sanity） | 0.340 (33/65/2) | **40/100 (40%)** | 308.9 |

**结论：**

1. **`vs baseline0` 崩塌是真实的**：100 局 0.130（SE≈0.034），不是 40 局噪声。
2. **但不是"绝对变弱"**：main@20000 vs 全新随机 = **0.505**（100 局）→ 训练 20k 后
   **绝对强度 ≈ 随机水平**（"随机"含 belief_planner 的 plan 偏置注入，见 §3.8.3 备注）。
3. **非传递性确凿（RPS 三角）**：全新随机 > baseline0（0.660）、baseline0 > main@20000
   （0.870）、main@20000 ≈ 全新随机（0.505）。三个对手两两成环 ⇒ 自对弈策略循环。
   因此 run 内的自引用指标（打冻结副本 0.85、打上一评估点 0.80）**全部失真**，
   "训练在变强"不成立；行为门禁全绿同样不作数。
4. **僵局早停裁定是主要混淆因素**：凡涉及 baseline0 的对局早停 28~40%（随机对随机 40%），
   帧数 min 100~112（远低于 360）——大量对局以 `_stall_probe` 早停 + `timeout_winner`
   （塔血/皇冠）裁定收场，该类裁定方差大、与"策略实力"关系弱，会把胜率差异放大。
   机制推测（未证）：随机策略的高方差出牌让训练策略进入"零塔损僵持"，被按塔血判负。
5. **对 §3.8 分支的影响**：在 cycling 未解决前，**堆量（D′）无意义**——在循环里打转的
   策略加码只会烧算力；价值头结构（C′）同样无意义——critic 学的是"循环内的伪标签"。
   下一个动作必须是**给自对弈加外部锚点**（评估侧锚点门禁 和/或 训练侧对手/奖励改造），
   否则任何内部指标（EV/行为门禁/对照曲线）都不可信。

**遗留（未在本轮做）**：main vs `SelfDefenderPolicy`（真防守脚本）的外部锚点对阵未跑
（`eval_solo_parallel` 会经 `FollowerOpponent` 包装对手，而脚本对手需直接挂 `env.opponent`，
需自定义评估循环，留作下一步可选取证）。

### 3.8.5 E1 落地：固定随机锚点（绝对强度门禁）+ 短验证（2026-09-12）

用户拍板 E1（评估侧锚点，不动训练动力学/架构）。落地 `rl/train_solo.py`：

- **常量**：`RAND_ANCHOR_SEED=99999`（锚点权重种子）、`RAND_ANCHOR_EVAL_SEED=90000`
  （锚点对局固定评估种子）、`RAND_ANCHOR_WARN_FLOOR=0.35`（报警线，未标定，≈起点 0.5
  下方 2σ，只报警不阻断）。
- **锚点构造**：`run_solo` 里用「保存 torch CPU RNG → manual_seed → 构造 → 恢复 RNG」
  生成固定随机策略 `baseline_rand`（不扰动训练随机性）；**永不参与训练、永不同步**。
- **每评估点**：`eval_and_write` 追加第三组对照 `vs baseline_rand`（固定 40 局重打，
  跨评估点可比），写进 `_controls_history`；`vs baseline_rand < 0.35` 时打印
  「绝对强度警报」（`_rand_anchor_warns`，先只报警）。
- **selftest**：`test_solo_rand_anchor`（同种子逐位一致 / 异种子不同 / RNG 无扰动 /
  报警线语义）。官方全量 selftest **100% PASS**。

**短验证（真实 ckpt）**：`main@20000`（`runs/fix_gru_ln_norm_20k/solo_main.pt`）
vs 固定随机锚点 ×40 局 ×2 次：

```
run0: winrate=0.15 (6W/34L/0D)
run1: winrate=0.15 (6W/34L/0D)   → deterministic: True
```

1. **E1 机制确定且生效**：两次逐位一致；0.15 < 0.35 ⇒ 若该 run 跑了 E1，每个评估点都会
   拉响绝对强度警报。
2. **与 §3.8.4 互相印证**：3 个随机对手里 2 个把 main@20000 打到 **0.13~0.15**
   （run 起点随机 0.13 / 固定锚点 0.15），1 个打平 0.505 —— 胜率高度依赖"抽到哪个随机
   权重"，这正是 cycling/"绝对强度≈随机"的表现；**"main vs 冻结副本 0.85"确认为自引用
   失真**。
3. **后续**：E1 列进每个 run 的 `_controls_history` 后，未来任何 run 的"绝对强度"都可
   逐点读取；下一步（待拍板）才是 E2（训练侧锚点对手改造）或 B′（GRU 耐久）。

---

## 3.9 E2 落地：固定随机锚点进训练对手池 + 20k 验证协议（2026-09-12）

用户拍板 E2（训练侧锚点对手改造）。**目标**：A′/E1 已证明自对弈在 RPS 循环里打转、
绝对强度≈随机；E2 把固定随机锚点从"评估侧对照"升级为**训练分布的第 4 槽**，让
"输给固定外部基准"成为可见负样本，打破"只赢近亲"的循环漂移。

### 3.9.1 改动（`rl/config.py` + `rl/train_solo.py` + `rl/selftest.py`）

| 项 | 落地位置 | 说明 |
|---|---|---|
| **对手池第 4 槽 `rand_anchor`** | `config.DEFAULT_OPP_MIX` / `train_solo._OPP_MIX` | `{frozen: 0.4, hist: 0.3, defend: 0.2, rand_anchor: 0.1}`（从 frozen 挪 0.1 给锚点，frozen 仍主力防 curriculum 断裂） |
| **共用锚点构造** | `train_solo._make_rand_anchor(cfg, belief_dim, device)` | RNG 保存/恢复 + `FollowerPolicy(RAND_ANCHOR_SEED)`。**E1 评估侧 `baseline_rand` 与 E2 训练侧锚点同源 ⇒ 逐位一致**（绝对强度测量与训练目标对齐） |
| **训练侧锚点 side** | `_OpponentPool.__init__` | `mix.get("rand_anchor",0)>0` 时构造 `FollowerOpponent(锚点策略, env, belief=..., deterministic=True)`（镜像卡组、deterministic 决策、永不参与训练/同步/PFSP） |
| **sample() 四槽分支** | `_OpponentPool.sample` | hist → defend → rand_anchor → frozen；无 hist ckpt 时按剩余概率归一化（rand 槽保留） |
| **附带修复（无 hist 退化归一化缺陷）** | 同上 | 旧实现无 hist 时 `r < hist+defend` 未受 hist 保护 ⇒ 实测 defend **0.78**/frozen 0.22，而打印宣称 0.286/0.714。现改为归一化（frozen 0.571/defend 0.286/rand 0.143，与打印一致） |
| **兼容** | `sample()` 用 `mix.get("rand_anchor", 0)` | 旧式三槽 dict（无 rand_anchor 键）仍合法：rand 概率 0、行为不变（含无 hist 归一化修复） |
| **selftest** | `test_opponent_pool_rand_anchor` | 默认配比含 0.1；有/无 hist 分布合规（±35%）；锚点权重与 E1 逐位一致；锚点局 record no-op；旧三槽兼容 |

### 3.9.2 验证协议（`e2_rand_anchor_20k`，--fresh）

```bash
cd src/clasher_new
python rl/run_league.py --mode solo --config economy --config-name e2_rand_anchor_20k \
  --fresh --total-steps 20000 --steps-per-eval 2500 --n-eval-games 40 --eval-workers 12 \
  --device cuda --value-norm running --adv-norm scale --diagnose-every 10 \
  --hist-seed-dir runs/economy_9k_ft --hist-seed-dir runs/economy_9j
```

（同 seed、同协议、唯一变量 = 对手池加 rand_anchor 槽；eval-workers 12 按 AGENTS
全局操作约定的 empirically 安全档位。）

**判读口径**（对照 §3.8 的 `fix_gru_ln_norm_20k`）：

1. **机制**：启动日志确认 `rand_anchor=0.1`；`_controls_history` 含 `vs baseline_rand`。
2. **主判据（cycling 是否被打破）**：`vs baseline0` 不再崩塌（对照 run 是 0.05~0.125）；
   `vs baseline_rand`（E1 绝对强度）随训练上升、末点 >0.35（不再拉响绝对强度警报）。
3. **critic 趋势**：EV（池化）/ `value_std_ratio` 是否开始爬升（锚点对局给"状态→胜负"
   提供固定外部信号，理论上应改善可学性）。
4. **GRU 耐久（顺带观测）**：`n_abs` 趋势（对照 run 0.461→0.745 单调上升）。
5. **不判** main 曲线单点胜率；行为门禁只读相对退化报警。

### 3.9.3 E2 20k 验证判读（2026-09-12，`e2_rand_anchor_20k`）：**干预无效，cycling 未破**

run `src/clasher_new/runs/e2_rand_anchor_20k/`，日志 `docs/train_e2_rand_anchor_20k.log`；
**59 局、9 评估点、训练循环 3419.4s、0 降级 / 0 Traceback**。协议同 §3.9.2
（同 seed，唯一变量 = 对手池加 rand_anchor 0.1；注意 eval-workers 12 vs 对照 run 的 16，
时长差异含此因素，见 §3.9.3 末条）。

| step | 胜率(main) | vs baseline0 | vs baseline_prev | **vs baseline_rand** | EV(池化) | h_std | n_abs | v/R std |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.625 | 0.500 | 0.525 | 0.625 | – | – | – | – |
| 2500 | 0.975 | 0.875 | 0.950 | 0.950 | −0.136 | 0.0449 | 0.517 | 0.0077 |
| 5000 | 0.613 | 0.300 | 0.575 | **0.300** | −0.024 | 0.0749 | 0.560 | 0.0210 |
| 7500 | 0.575 | 0.350 | 0.662 | **0.175** | −0.118 | 0.0541 | 0.611 | 0.0205 |
| 10000 | 0.525 | 0.138 | 0.375 | **0.200** | −0.138 | 0.0426 | 0.660 | 0.0321 |
| 12500 | 0.650 | 0.775 | 0.713 | 0.812 | −0.016 | 0.0423 | 0.696 | 0.0196 |
| 15000 | 0.825 | 0.800 | 0.850 | 0.787 | −0.189 | 0.0671 | 0.638 | 0.0186 |
| 17500 | 0.250 | 0.725 | 0.300 | 0.850 | −0.045 | 0.0430 | 0.572 | 0.0145 |
| 20000 | 0.575 | **0.075** | 0.375 | **0.050** | −0.031 | 0.0418 | 0.596 | 0.0098 |

**结论（对照 §3.9.2 判据逐条）**：

1. **机制落地 ✓**：`rand_anchor=0.1` 生效、`_controls_history` 含 `vs baseline_rand`；
   锚点权重与 E1 同源（selftest 逐位验证）。
2. **主判据 FAIL —— 训练侧固定随机锚点未能打破 cycling / 绝对强度依旧崩溃**：
   - `vs baseline_rand` 末点 **0.050**（1σ=0.035，2σ 显著低于报警线 0.35），E1 绝对强度
     警报从 step 5000 起**全程拉响**；对照 fix_gru_ln_norm_20k 的 E1 短验证是 0.15。
   - `vs baseline0` 末点 **0.075**（对照 run 0.125）——对训练起点随机策略的崩塌依旧。
   - 对照组仍剧烈振荡、17500 点出现非传递（main 0.25 但 vs baseline0 0.725、vs
     baseline_rand 0.85）⇒ **RPS 循环仍在转**。
3. **critic 依旧常数**：EV 0/8 过零（−0.016~−0.189，无上升趋势）；`value_std_ratio`
   0/8 达标（0.008~0.032 vs 门槛 0.3）——E2 未改善"状态→胜负"的可学性。
4. **GRU 耐久顺带观测：本轮 n_abs 峰值 0.696、末点 0.596，未再单调攀升**（对照 run
   0.461→0.745）——不能当 E2 的功劳（非目标变量），但至少 20k 内未超标。
5. **为什么 0.1 锚点无效（机制推断，非结论）**：59 局里 rand_anchor 只占 ~6 局，
   PPO 联合损失里锚点对局的梯度占比太低，压不住 90% 自对弈对局里的策略漂移；
   且锚点强度≈随机，20k 步样本不足以学会针对它。**10% 弱锚点不足以锚定自对弈动力学。**

**分支判定（需拍板，本轮不擅自实施）**：

- **E2′**：提高锚点占比（0.1→0.3）或做多锚点（多个随机种子/不同强度）——单变量重验；
  若仍无效，说明"固定弱锚点"路线本身不行。
- **B′**：GRU 耐久（本轮 n_abs 未恶化，优先级可再降）。
- **C′/D′ 依旧无意义**：critic 未拟合 + cycling 未破，堆量/价值头结构都不会有可信收益。
- **更根本的方向**（讨论备选）：自对弈绝对强度爬坡需要**对手模型**（随训练变强的对手 /
  多源对手），或**奖励结构**引入"击败固定基准"的显式信号——均属高侵入，需用户拍板。

**附：时长**：训练循环 3419.4s（59 局）。对照 fix_gru_ln_norm_20k 是 2709.3s（59 局、
8 点、eval-workers **16**）——e2 用 12 使评估更慢，**不能直接归因 E2**；精确速度分解
见 `docs/train_speed_benchmark_2026-09-12.md`（本 session 基准）。

## 3.10 B′ 落地（value 直连 enc）+ C′ 落地（早停裁定降噪）+ 20k 验证（2026-09-12）

来源：`docs/critic_probe_experiment_2026-09-12.md` 实验 A/B′；用户拍板顺序 **先 B′ 落地，随后 C′**。

### 3.10.1 B′ 落地：`value_bypass`（value 头直连 post-LN enc，跳过 GRU）

依据（实验 B′，同 rollout / 同 split / 同监督配方）：无 GRU 通路 test EV **+0.294** vs
带 GRU **+0.217**（GRU 净损耗 ~0.08 EV）；探针 MLP 上界 +0.242。

改动（单变量 = **value 通路的输入**）：
- `rl/follower.py`：`FollowerPolicy(..., value_bypass=False)` 新参数；5 处 value 计算
  （`act` / `act_parallel` / `evaluate` / `evaluate_batch` / `value`）在 bypass 时走
  `value_head(enc)`；**策略头 slot/cell 仍吃 GRU 隐状态 `h`**（GRU 推进逻辑不变，
  策略的状态依赖不受影响）；
- checkpoint 元数据新增 `value_bypass`（save/load）；`load_checkpoint(..., value_bypass=)`
  与元数据不一致时**告警**（旧 ckpt 语义错位，须 `--fresh`）；
- 标志传播：`train_solo`（main/opp/baseline0/baseline_prev/eval worker）、
  `run_league::_build_league`、`flow_league::build_flow_models`、`league` 快照；
- `rl/config.py`：`TrainConfig.value_bypass=False`（dataclass 默认=旧行为，兼容其它入口），
  **`economy` 预设设 True**（新训练默认走新架构）；CLI `--no-value-bypass`（消融用）。

### 3.10.2 C′ 落地：早停低置信裁定降噪（`stall_draw_margin`）

依据（实验 3 静态审计）：早停局占 28~40%，`timeout_winner` 的塔血%细差裁定是标签噪声候选。

改动（**只改训练侧标签**；eval 保留真实 CR 规则 `timeout_winner` ⇒ 评估口径与历史可比）：
- `rl/run_league.py`：新增纯函数 `settle_stall_from_counts` + 包装 `settle_stall`：
  皇冠不同 → 决定性 0/1；皇冠相同且塔血%差 ≥ margin → 决定性 0/1；
  **皇冠相同且差 < margin → None（记平局=失败）**，去掉掷硬币级胜负标签；
- `rl/config.py`：`stall_draw_margin: float = 0.05`；CLI `--stall-draw-margin`（0=旧行为）；
- `train_solo` 训练环早停结算改用它；`_probe` 增 `stall_games` / `stall_close_draws`
  计数并落盘 history（**测量 C′ 实际生效比例**；冒烟 7 局：2 次早停、0 次降级）；
- `diag_critic_ev` 的 rollout 结算同步该口径（诊断标签与训练一致）。

### 3.10.3 验证与当前状态

- 官方全量 `rl/selftest.py` **94 项 100% PASS**（含新增 `test_value_bypass`、
  `test_stall_settlement_margin`）；日志 `docs/selftest_bypass_cprime.log`。
- 冒烟 `runs/smoke_bypass_cprime`（2000 步、2 局/点）：0 Traceback、GRU 仍活
  （h_std 0.042~0.068、n_abs 0.37~0.42）；EV 仍负（2k 步无判读价值）。
- **20k 验证跑** `byp_cprime_20k`：economy 预设（bypass 开 + C′ margin 0.05），
  同协议（40 局/点、eval-workers 12、两个 hist-seed 目录）；日志
  `docs/train_byp_cprime_20k.log`。
- **判读口径**：主判据 = EV 池化是否过零/上升 + `value_std_ratio` 是否改善
  （对照 `fix_gru_ln_norm_20k`：EV −0.0051、ratio 0/8）；次判据 = 对照曲线净漂移；
  **新增** `stall_close_draws/stall_games` 量化 C′ 生效比例。
- ⚠️ **归因声明**：本轮同时带 B′+C′；若 `stall_close_draws` 占比高，需另跑
  `--stall-draw-margin 0` 做消融才能分离两者贡献。

### 3.10.4 落地副作用修复：诊断探头的 value 口径（2026-09-12，重要）

**踩坑**：B′ 改了 value 通路（`value_head(enc)`），但两个诊断探头**硬编码了旧通路**
`value_head(hidden)`：
- `rl/diagnostics.py::gru_vitality` 的 `value_std`（→ `value_std_ratio`，是 v3 §2 验收表
  第 3 行、也是本 run 的次判据）；
- `scripts/diag_value_head.py` 的价值输出统计。

⇒ 对 bypass 模型，`value_std` 测的是 `value_head(gru(enc))`——**不是被训练的那个量**。
已修（bypass 时取 `value_head(enc)`）+ 回归护栏：`test_value_bypass` 现断言
`gru_vitality(pol)["value_std"] == std(value_head(enc))` 且 bypass/非 bypass 不同。
**教训（同类第二次）**：**架构变更后必须全仓搜"硬编码的前向通路"**——上次是
"验收表的分母没人测"（§3.6 教训 3），这次是"探头测了旧通路"。
⚠️ 本轮 `byp_cprime_20k` 进程已加载旧模块 ⇒ **该 run 日志的 `vstd/rstd` 列无效**
（EV 列来自真实 rollout，不受影响）；每个评估点存了 `solo_main_<step>.pt`，
跑完后用修复版 `diag_value_head.py` 离线重算真值。

### 3.10.5 20k 验证判读（`byp_cprime_20k`）：B′ 未改善 critic、C′ 训练侧几乎不触发（2026-09-12）

run `src/clasher_new/runs/byp_cprime_20k/`（65 局、8 评估点、训练循环 1993.4s、
**0 降级 / 0 Traceback**；economy 预设 = `value_bypass=True` + `stall_draw_margin=0.05`；
hist ckpts=12）。

| step | 胜率 | EV(池化) | h_std | n_abs | stall_games | close_draws |
|---|---|---|---|---|---|---|
| 2500 | 0.425 | −0.0082 | 0.0607 | 0.457 | 1 | 0 |
| 5000 | 0.700 | −0.2802 | 0.0407 | 0.452 | 2 | 1 |
| 7500 | 0.725 | −0.2939 | 0.0399 | 0.462 | 2 | 1 |
| 10000 | 0.713 | −0.0103 | 0.0371 | 0.463 | 2 | 1 |
| 12500 | 0.613 | **−0.0044** | 0.0365 | 0.446 | 2 | 1 |
| 15000 | 0.450 | −0.1186 | 0.0376 | 0.471 | 2 | 1 |
| 17500 | 0.500 | −0.0210 | 0.0386 | 0.487 | 2 | 1 |
| 20000 | 0.550 | −0.0643 | 0.0386 | 0.477 | 2 | 1 |

**① B′（bypass）未改善 critic**：EV **0/8 过零**（最好 −0.0044），与参照
`fix_gru_ln_norm_20k`（0/8 过零、最好 −0.0051）同级。诊断（50 局，`docs/diag_ev_byp_20k.log`）：
`v std=0.025` vs `R std=13.71`（0.18%）、`EV_global=−0.0060`、`RMSE 13.76 ≈ std(R)`、
bias −1.09 ⇒ **仍是常数预测器**。离线重算真实 `value_std=0.0221`
（`docs/diag_vh_byp_20k.log`，修复版脚本）→ `value_std_ratio≈0.0017~0.0036`
（门槛 0.3；比参照 0.014~0.031 还低）。

**② 但表征的可预测性反而更高（本轮最重要发现）**：同一批帧上
**MLP 探针 `post-LN enc → GAE return` test R² = +0.4246**（线性 +0.1355）
——**高于此前两个 ckpt（+0.242 / +0.306），是迄今最高**。
⇒ "信息在 enc 里（42%）"与"critic 吸收 ≈0"的**落差本轮最大**；结合 B′ 监督实验
（同网络 value-only 目标可达 EV +0.22~0.29）⇒ **瓶颈不在表征、也不在 value 通路接线**
（bypass 已排除接线假设），而在 **on-policy PPO 联合训练的价值吸收/优化动力学**
（共享 trunk 的 value 梯度与策略梯度相互干扰、TD 自举目标噪声、单轮 on-policy 样本）。

**③ C′ 在训练侧几乎不触发（修正审计假设）**：65 局里早停仅 **2 局**（3%），被 C′ 降级为
平局的 **1 局**（1.5%）。审计所说"早停局占 28~40%"是 **eval 侧随机对手对局**
（baseline_rand），那些局不进训练标签 ⇒ **C′ 对训练标签的实际影响 ≈ 无**。
本轮结果因此可近似按 **B′ 单变量**判读。

**④ 行为/对照**：本轮**未复现** `fix_gru_ln_norm_20k` 的两处崩塌——`vs baseline0` 全程
0.33~0.50（参照 5000 起 0.05~0.125）、`baseline_rand` ≥0.31（E2 曾跌到 0.05）；
但 `vs baseline_prev` 仍剧烈振荡（0.89→0.40）⇒ **cycling 指纹仍在**（与 E2 结论一致）。

**分支判定（需拍板）**：

- **E′（首选，直击本轮结论）**：**给价值通路独立 encoder**（= v2 §4 的"拆价值头/价值编码器"。
  当时因病因误判被搁置；现有三条新证据支撑：enc 可预测 0.42、监督同网络可达 0.22~0.29、
  on-policy EV≈0 且 bypass（只改接线）无效）。**可证伪预测**：独立 encoder 的价值通路
  应能到 EV 0.2+（机制同 B′ `lin-joint`——value-only 梯度塑造自己的表示）。
- **F′**：显式 aux value loss（把 B′ lin-joint 的"value-only 梯度"以辅助损失并入 PPO 联合训练）。
- **G′**：回到对手/数据侧（cycling 是未解主问题；本轮 control 更健康但循环仍在）。
- **`value_bypass` 默认值**：证据不支持它带来收益（EV 同级、真实 ratio 略差、无副作用）
  ⇒ 可选**回退 economy 预设为 False**（保留 flag/代码与消融记录）；若保留，它的价值是
  一条"value 接线与瓶颈无关"的对照基线。

## 3.11 E′ 落地：独立价值编码器 + 非线性价值头（2026-09-12，用户拍板执行）

**依据（三条，全部来自 §3.10.5）**：
1. **MLP 探针 `enc → return` R² = +0.4246**，而**线性探针仅 +0.1355** ⇒ 价值信息**主要是
   非线性的**；现行 `value_head = nn.Linear(hidden,1)`（单线性层）**先天上限 ~0.13**，
   这正是 bypass 通路（纯线性吃 enc）实测 ≈0 的结构原因之一；
2. 监督实验（value-only 目标、trunk 可训）可达 EV **+0.22~0.29**，而 on-policy 联合训练 EV≈0
   ⇒ 差距在"价值吸收"而非表征；
3. bypass（只换接线、仍共享参数/线性头）**无效** ⇒ 需要给价值通路**自己的参数与容量**。

**改动（单变量 = 价值通路的参数独立性与头容量）**：
- `rl/follower.py`：`FollowerPolicy(..., value_independent=)`；新增
  `value_enc_fc`（fused→hidden）+ `value_enc_ln` + `value_head_mlp`
  （`hidden → max(32, hidden//2) → 1`，ReLU）；`_encode_parts` / `_encode_batch_parts`
  暴露 `fused`；**统一入口 `_value_from(enc, h, fused)`**（优先级 **independent > bypass >
  shared**），act / act_parallel / evaluate / evaluate_batch / value / diagnostics /
  诊断脚本全部走它（避免"探头硬编码旧通路"再次发生）；
- `rl/config.py`：`value_independent: bool = False`（dataclass 默认=旧行为）；
  **economy 预设 True**；CLI `--no-value-independent`（消融）；
- ckpt 元数据 `value_independent` + load 不一致告警；train_solo 启动一致性检查覆盖两个标志；
- 传播：train_solo（main/opp/baseline0/baseline_prev/**rand_anchor**/eval worker）、
  `run_league::_build_league`、`flow_league`、`league` 快照。

**踩坑（当场修复，重要）**：`_make_rand_anchor` 漏传架构标志 → 它作为 control 的
`opp_model` 传进 `eval_solo_parallel`，worker 按 `env_kwargs`（= cfg 架构）建网后
`load_state_dict(opp_sd)` **键集不匹配** → **每周期 worker 启动失败 + 静默降级串行**
（冒烟日志 `Missing key(s): value_enc_fc.*/value_head_mlp.*`）。修复：锚点构造随 cfg 标志。
**教训：架构标志必须传播到所有"会经 eval worker 做 state_dict 往返"的策略构造点；
worker 降级是静默的，判读前必须 grep 日志的 Traceback/降级行。**

**验证协议**：run `eind_20k`（economy 预设 = independent；同 20k 协议：`--fresh`、40 局/点、
eval-workers 12、hist ckpts=12、`--value-norm running --adv-norm scale`）。日志
`docs/train_eind_20k.log`。**可证伪预测**：EV 应升到 **0.2+**（对照 `byp_cprime_20k`：
0/8 过零、最好 −0.0044；`fix_gru_ln_norm_20k`：最好 −0.0051）。
**若不升** ⇒ 指向 **F′**（aux value loss / 更高 vf_coef / 更长的价值训练），即瓶颈在
"梯度与目标"而不是"容量与参数独立性"。

### 3.11.1 E′ 20k 判读（`eind_20k`）：**预测被证伪 → 根因定位到"优化预算与批次构成"**（2026-09-12）

run `runs/eind_20k`（65 局、8 评估点、训练循环 **3717.1s**、**0 降级 / 0 Traceback**）。

| step | 镜像胜率 | EV(池化) | h_std | n_abs | vstd/rstd(修正口径) |
|---|---|---|---|---|---|
| 2500 | 0.750 | −0.038 | 0.0435 | 0.508 | 0.0008 |
| 5000 | 0.900 | −0.131 | 0.0387 | 0.478 | 0.0006 |
| 7500 | 0.350 | −0.027 | 0.0609 | 0.471 | 0.0041 |
| 10000 | 0.850 | −0.162 | 0.0611 | 0.481 | 0.0003 |
| 12500 | 0.625 | −0.092 | 0.0388 | 0.469 | 0.0003 |
| 15000 | 0.538 | **−0.004** | 0.0375 | 0.481 | 0.0022 |
| 17500 | 0.075 | −0.026 | 0.0397 | 0.525 | 0.0003 |
| 20000 | 0.975 | −0.643 | 0.0378 | 0.531 | 0.0002 |

**① 预测被证伪**：EV **0/8 过零**（最好 −0.004），与 bypass（−0.0044）、grid_ln（−0.0051）
**同一水平**。离线真值 `value_std = 0.0067`（200 帧，`docs/diag_vh_eind_20k.log`，
E′ 独立 MLP 头）→ ratio ≈0.0005（门槛 0.3）⇒ **仍是常数预测器**。
诊断（50 局，`docs/diag_ev_eind_20k.log`）：`v std=0.007` vs `R std=7.30`（0.1%）、
`EV_global=−0.0356`、`corr(v,R)=+0.126`；**探针：线性 R²=+0.041、MLP R²=+0.236**
（表征仍含 ~24% 可预测信息，critic 吸收 ≈0）。

**跨 ckpt 的"表征 vs 吸收"总表**（同一诊断口径，50 局）：

| ckpt（value 架构） | v std | R std | EV_global | 线性探针 R² | MLP 探针 R² |
|---|---|---|---|---|---|
| `e2_rand_anchor_20k`（共享 GRU+线性头） | 0.117 | 5.43 | −0.219 | +0.111 | +0.242 |
| `fix_gru_ln_norm_20k`（共享 GRU+线性头 + grid_ln） | 0.219 | 11.05 | −0.001 | +0.104 | +0.306 |
| `byp_cprime_20k`（bypass：线性头吃 enc） | 0.025 | 13.71 | −0.006 | +0.136 | **+0.425** |
| `eind_20k`（E′：独立编码器 + MLP 头） | 0.007 | 7.30 | −0.036 | +0.041 | +0.236 |

⇒ **四种价值架构下 critic 吸收都 ≈0，而表征始终含 24~42% 可预测信息。**

对照曲线：`baseline0` 0.46→**0.975**（@15k）→0.50、`baseline_rand` 0.625→**0.925**（@10k）
→0.54（绝对强度有爬升），但 `baseline_prev` 0.10↔1.0 剧烈振荡 ⇒ **cycling 仍在**。

**② 根因（本轮决定性发现，代码级证据）——不是架构，是"优化预算 + 批次构成"**：
- `PPOTrainer.update()`（`rl/ppo.py:199-259`）= **1 次 forward + 1 次 backward + 1 次
  `opt.step()`**：**无 `n_epochs`、无 minibatch、无 shuffle**；
- `train_solo` 每次 update 取 `transitions[:cfg.batch_size]`
  （`rl/train_solo.py:1385-1392`）= **同一局连续的 128 帧**；
- ⇒ **20k 步 = 156 次梯度步**（AGENTS 早已记："20k ≈ 156 次更新"），且每步的目标是
  "局段均值"（批内 Var(R) 仅全局 0.32×、`corr(R_t,R_{t+1})=0.99`）——
  **价值头（无论共享/线性、bypass、独立 MLP 头）在 156 次单步更新内不可能拟合**。

**③ 三条独立证据互相印证**：
- 同一程序、同数据：**监督微调 65k 次逐帧随机序梯度步 → EV 0.22~0.29**（≈400× 的梯度步）；
- 表征可预测性（MLP 探针 `enc→return`）**0.24→0.31→0.42**（跨 run 上升），而 critic 吸收 ≈0；
- B′（接线）、E′（参数独立 + 非线性头）**两轮架构干预全部无效** ⇒ 架构不是瓶颈。

**④ 结论与 F′ 设计（需拍板）**：把更新做成**真正的 PPO**——
`n_epochs`（4~8）× **shuffle** × **minibatch**（32~64）在收集到的 rollout 上多轮随机小批更新：
- 价值头梯度步数 ×4~8，且**打破"连续帧同质批"**（每步看到跨局、跨状态的目标多样性）；
- 代价：训练循环变慢（估计 20k 从 ~2000s → 4000~5000s）；
- **可证伪预测：EV 应在 20k 内 >0 并显著上升**；
- ⚠️ 口径影响：`n_epochs>1` 后 `ratio` 会离开 1.000、clip 开始生效 —— AGENTS §②
  "ratio≡1.000 是结构性的"**只对 `n_epochs=1` 成立**，届时 ratio/clip 重新成为有效诊断。

**⑤ 三次"架构标志未传播"事故（教训）**：B′/E′ 引入的构造参数必须传播到**所有**会做
`state_dict` 往返的构造点——本轮连踩三处：`_make_rand_anchor`（→ eval worker 键集不匹配 →
**静默降级串行**）、`_OpponentPool._ensure_hist`（hist ckpt 可能是新架构）、三个 `diag_*.py`
的镜像对手（`diag_value_head`/`diag_encoder_scale`/`diag_gru_ablation`/`diag_critic_ev`）。
均已修（hist 改为直接采用 `load_checkpoint` 返回的架构正确策略；诊断脚本镜像对手随源策略标志）。
**纪律：新增架构参数时，必须 `grep load_state_dict` 全仓核对一遍。**

## 3.12 F′ 落地：真正的 PPO 更新预算（2026-09-12，用户：持续推进）

分支 `no-human-watch-A`（远端 `origin/no-human-watch-A`），提交 `8c33830 → 75ac7c8 → b408924 → 2648baa`。

### 3.12.1 改动（`rl/ppo.py` 为主）

| 项 | 内容 |
|---|---|
| `PPOTrainer` 新增 | `n_epochs`（默认 1）/ `minibatch_size`（默认 0=整批）/ `shuffle`（默认 False）/ `seed`；`grad_steps` 累计 `opt.step()` 次数 |
| 重构 | 抽出 `_loss_pass`（一批的 loss+诊断）/ `_apply_grad`（backward+诊断+clip+step）/ `_plan_batches`（每轮的划分）/ `_update_epochs`（多轮主循环） |
| **兼容性红线** | `n_epochs=1 & minibatch=0 & shuffle=False`（默认）走**原单一 pass 分支**（loss 仍 `sum` 口径），逐位等于旧实现——`run_league`/`flow_league`/`train_follower`/`train_prophet` 共用本类 |
| 新分支口径 | loss 用 `mean`；每小批一次 `opt.step()`；`ratio/clip/grad_norm` 取**末轮**（标准 PPO 报告口径） |
| 接线 | `rl/config.py`（`ppo_epochs/ppo_minibatch/ppo_shuffle`，默认=旧行为）、`rl/run_league.py`（`--ppo-epochs/--ppo-minibatch/--ppo-shuffle`）、`rl/train_solo.py`（构造 + 启动打印"每 N 帧 X 次梯度步" + 步日志 `gs=`） |
| resume 护栏 | PPO 预算写入 `run_state`，续训时与断点记录不一致即告警（`config.json` 不参与 resume 解析 ⇒ 忘传参数会静默退回旧行为） |
| 回归测试 | `selftest.py::test_ppo_multi_epoch_minibatch`（①默认=1 次梯度步且 ratio≡1/clip≡0；②`grad_steps=epochs×批数`；③`_plan_batches` 是**划分**且打乱可复现；④同行数据多轮后 `vraw` 必降；⑤ratio 离开 1.000 是期望行为；⑥⑦见 §3.12.3） |

**口径影响**：`n_epochs>1` 后 `ratio` 会离开 1.000、clip 开始生效 —— AGENTS §②
"ratio≡1.000 是结构性的"**只对 `n_epochs=1` 成立**，此后 ratio/clip 重新成为有效诊断。

### 3.12.2 20k 验证跑（`fprime_20k`）：训练期 EV 首次 >0，但**判读口径先被查出三个 bug**

协议与 E′ 完全一致（同 seed、同 hist 种子目录、`--fresh`），唯一变量 =
`--ppo-epochs 4 --ppo-minibatch 32 --ppo-shuffle`。
run `runs/fprime_20k`（67 局、8 评估点、训练循环 **2557.8s**、0 降级 / 0 Traceback）。

| step | ES 池化 EV（原始日志） | h_std | n_abs | 镜像胜率 |
|---|---|---|---|---|
| 2500 | **+0.1011** | 0.0587 | 0.574 | 0.412 |
| 5000 | −0.0513 | 0.0585 | 0.641 | 0.662 |
| 7500 | −0.0477 | 0.0634 | 0.667 | 0.575 |
| 10000 | −0.1224 | 0.0691 | 0.709 | 0.575 |
| 12500 | **+0.5575** | 0.0440 | 0.540 | 0.825 |
| 15000 | **+0.4621** | 0.0699 | 0.765 | 0.650 |
| 17500 | **+0.2711** | 0.0668 | 0.759 | 0.662 |
| 20000 | **+0.4129** | 0.0817 | 0.726 | 0.900 |

**这是全项目第一次出现 EV > 0**（历史所有 run 每点 ≤0）。但**不能就此宣布 critic 修好**——
先做了口径核对，结果发现三个测量问题（都不是训练数学问题，但都会把结论带偏）：

### 3.12.3 三个测量 bug（同一天查出，全部已修 + 加回归哨兵）

**① `ratio/clip` 聚合误除全部轮次（假警报）**
步日志读出 `ratio≈0.26 / clip≈2%` —— 内部自相矛盾（ratio 0.26 意味着绝大多数样本
都在 clip 区间外，clip 不可能只有 2%）。逐 epoch 埋点实测：**每个小批 ratio 都是
0.96~1.02（健康）**，只是聚合时 `r_acc` 每轮清零而分母 `w_tot` 是全部轮次 ⇒ 末轮
均值被稀释成 `1/n_epochs`。修：ratio/clip 用本轮权重 `w_ep` 作分母。
哨兵：`lr=0.0` 时 `ratio_mean` 必须落在 float32 ULP 级（<1e-5），旧写法会读出 ≈0.25。

**② `value_std_ratio` 的分子分母来自两个样本集（不可判读）**
@2500 读出 `EV=+0.1011` 而 `vstd/rstd=0.0006` —— 两个数不可能同时为真。核对：
分子 `value_std` 来自 GRU 探针的**最近 96 帧**、分母 `r_std_batch` 来自**整个评估
窗口**。修：`value_std_ratio = std(last_ev_pairs 的 v) / std(last_ev_pairs 的 R)`
（同窗口）；跨窗口的旧比值另存 `value_std_ratio_probe` 仅留档。
数学依据：`EV ≤ 2σ_v/σ_R`（由 `|Cov(v,R)| ≤ σ_v σ_R` 推出），
所以 `σ_v/σ_R=0.0006` 时 EV 上界只有 0.0012，读不出 0.10。

**③ EV 用的是"更新后"预测 = in-sample 读数（最严重）**
旧实现只有一次 forward 且它发生在 backward 之前 ⇒ 旧 EV 天然是**更新前**口径；
F′ 若沿用"末轮预测"，读到的就是**刚在这 128 帧上训过 12 步**的值。三路证据：

- 训练日志末点 EV = **+0.4129**，而**同一权重**在 50 局独立 rollout 上
  （`scripts/diag_critic_ev.py`）`EV_global = −0.0044`（另一次 50 局 −0.0255）；
- 受控实验（同批 128 帧）：更新前 EV **+0.011** / 更新后同批 **+0.042** /
  更新后**换一条独立 rollout −0.384**；
- 结论：把"记住这 128 帧"当成了"critic 学会了"。

修：非旧分支在更新前多做一次整批 no_grad 前向取 `(v_pre, R)` 作为
`last_ev_pairs`/`explained_variance`（与全部历史 run 可比），末轮 in-sample 值另存
`explained_variance_insample`，步日志加 `EVin=`。代价 ≈ 每次 update 多 1/16 更新开销。

**纪律（本轮第三次栽在指标上）**：
- 写出一个**比值**门槛前，先确认分子分母是**同一个样本集**（前两次：`gru_vitality`
  硬编码 `value_head(h)`；`r_std` 分母根本没实现）；
- 报出一个**跨版本可比**的指标前，先确认它**不依赖被改动的那些参数状态**
  （EV 拿"更新后的预测"去比"更新前的历史曲线"就是这类错）。

### 3.12.4 F′ 的真实成效（修正口径后的权威测量）

同一权重、同一 50 局协议、离线权威口径：

| 指标 | E′（`eind_20k`） | F′（`fprime_20k`） | 判读 |
|---|---|---|---|
| `EV_global`（50 局） | −0.0356 | **−0.0044 / −0.0255** | 仍在 0 附近、未过零 |
| v std | 0.007 | **0.070** | 输出动态范围 **10×** |
| R std | 7.30 | 10.36 | — |
| σ_v/σ_R | 0.0010 | **0.0068**（门槛 0.3） | 仍差 40× |
| h 跨帧 std | 0.038 | 0.063（诊断 200 帧） | 健康 |
| GRU n_abs | 0.48 | 0.71~0.77 | <0.9 但单调上升（同 grid_ln 观察） |
| `diag_value_head` 200 帧 value std | 0.0067 | **0.0382** | 5.7× |

⇒ **F′ 让 critic 从"完全不动"变成"能动 10 倍"，但离线 EV 仍 ≈0。**
训练期那个 +0.41 是 in-sample（§3.12.3 ③）。

**关键新证据（新鲜随机初始化对照）**：未训练的 value 头
（`tmp_fresh_value_scale.py`，200 帧真实 rollout，同架构）**σ_v/σ_R = 0.08~0.18**，
而训练 20k 步后是 **0.0074** —— **训练是主动把输出方差收缩到条件均值**。

### 3.12.5 病因转向：探针的"可预测性"是时间泄漏（推翻 §3.11.1 的推理链）

给 `lin_probe`/`mlp_probe` 加 `groups=`（按局分组留出）后，`fprime_20k`（50 局/16721 帧）：

| 留出口径 | 线性探针 R² | MLP 探针 R² |
|---|---|---|
| 逐帧随机（**旧口径**） | +0.0953 / +0.1385 | **+0.2699 / +0.3806** |
| **按局分组（新）** | **−0.3486 / −0.2123** | **−0.3595 / −0.4498** |

（两列 = 两次独立 50 局运行；分组 MLP 逐点为 −0.36 / −0.45。）

根因：相邻帧 `corr(R_t,R_{t+1})≈0.99`，逐帧随机留出会把测试帧的"邻居"放进训练集。
⇒ **跨 ckpt 总表里那些 0.24~0.42 的"表征可预测性"是时间泄漏，跨局并不泛化**
（分组口径下比"预测测试局均值"还差）。
§3.11.1 里"表征有信息、critic 吸收不了 ⇒ 优化预算问题"的**推理链由此失效**——
B′/E′/F′ 三轮都在攻一个被测量假象指出的靶子（F′ 本身仍是正确的训练器改进，保留）。

新的图像（与全部实测自洽）：
1. critic 输出收缩到条件均值，是**MSE 最优行为**，不是"学不动"；
2. `EV≈0` 对应"逐帧回报在该表征下跨局不可预测"；
3. 剩下的问题变成**数据/标签侧**：这个"不可预测"是表征不够、还是回报标签
   （逐帧 GAE 回报）本身就是帧级噪声主导、还是镜像自对弈使局结果与状态无关
   —— §3.12.6 用"换标签"实验分离。

### 3.12.6 换标签探针（按局分组留出）：**没有任何一种回报标签能跨局泛化**

同一批 50 局帧（`runs/fprime_20k`，纯镜像自对弈）、同一 `enc`，只换预测目标
（`docs/diag_ev_fprime_targets.log`）：

| 目标 | 分组 线性 R² | 分组 MLP R² | 逐帧 MLP R²（对照） |
|---|---|---|---|
| 逐帧 GAE 回报 `R_t` | −0.2123 | **−0.4498** | +0.3806 |
| **局结果**广播（终局帧 R = 终局奖励） | −0.2711 | **−0.4880** | +0.3494 |
| 局均回报广播 | −0.2955 | **−0.7118** | — |
| critic `EV_global`（对照） | — | **−0.0048** | — |

三条读法：
1. **逐帧口径全部为正、分组口径全部为强负** —— 泄漏是系统性的，与目标选择无关；
2. **换标签救不了 critic**：连"这局最后是赢是输"这种最粗的标签，跨局也泛化不了
   （分组 R² −0.49）⇒ 不是"标签太噪"，而是**状态表征 → 局结果 的映射本身不跨局**；
3. critic 的 `EV≈0` 与"分组探针做不到更好"**互相印证** ⇒ 价值头收缩到条件均值
   是当前表征/数据下的**最优解**，不是欠训练、不是架构、不是更新预算。

### 3.12.7 F′ 结论与下一步（本轮定论）

**F′ 本身成立且保留**：更新预算从 1 次梯度步/更新 → 16 次，`ratio/clip` 恢复诊断价值，
critic 输出动态范围 10×（σ_v/σ_R 0.0010 → 0.0068），离线 EV 由 −0.036 到 −0.005/−0.026。
但**离线 EV 仍未过零**，训练曲线的 +0.41 是 in-sample 假象（§3.12.3 ③）。

**更重要的是病因被改写**（推翻 §3.11.1 的"表征有信息但没吸收"）：
- 那个"信息"是时间泄漏；
- 换任何标签、按局分组后都**不可跨局泛化**；
- 新鲜随机 value 头的动态范围本来就有 0.08~0.18（比训练后大 10~25×）
  ⇒ 训练是在**正确地收缩到条件均值**。

**因此 critic 侧不该继续加投入**（B′/E′/F′ 三代都在攻一个测量假象指出的靶子）。

**下一步的第一性问题（G′ 候选，尚未实施）**：跨局不可预测到底卡在哪一层？
用一个**三层对照**（同一批帧、同一分组留出口径）分离：
1. **原始状态标量**（双方塔血/圣水/剩余时间/皇冠）→ 局结果：若**能**预测
   ⇒ 世界可预测、瓶颈在 `enc` 表征（该改观测编码/CNN）；
2. **`enc` → 局结果**：已知**不能**（本节 −0.49）；
3. **随机基线**：打乱局标签后重测，给出"分组口径下 R² 的噪声地板"。
若 1 能、2 不能 ⇒ 表征/编码是瓶颈；若 1 也不能 ⇒ 镜像自对弈的局结果本身接近掷硬币，
critic 工作**应当停**，把算力转回策略侧（cycling / 组织进攻）与对手结构。

### 3.13 G′ 三层可预测性对照（2026-09-12，`docs/diag_predict_fprime2.log`）

同一批 50 局 / 16234 帧，**全部按局分组留出 20%**（10 局测试）。特征三层：
`raw`（12 维真值状态标量：双方塔血 6 + 皇冠 2 + 圣水 2 + 时间 + 塔血差）、
`enc`（post-LN 128 维）、`raw+enc`。标签四个，另加两个**功效对照**。

| 特征 | 标签 | 线性 R²（分组） | MLP R²（分组） | MLP R²（逐帧泄漏口径） |
|---|---|---|---|---|
| raw | frame_R | −0.5126 | −2.7048 | +0.7838 |
| raw | outcome（终局奖励广播） | −0.1027 | +0.0583 | +0.7383 |
| raw | ep_mean | −0.6272 | −2.8357 | +0.6770 |
| raw | crown_diff（终局皇冠差广播） | **+0.1844** | −0.3059 | +0.7589 |
| raw | **CTRL: time** | **+1.0000** | +0.9976 | +0.9989 |
| raw | **CTRL: hp_diff** | **+1.0000** | +0.9986 | +0.9994 |
| enc | frame_R | −0.8421 | −1.6931 | +0.2754 |
| enc | outcome | −0.0311 | −0.5475 | +0.2570 |
| enc | crown_diff | +0.0717 | −0.8053 | +0.3547 |
| enc | **CTRL: time** | **+0.9999** | +0.9993 | +0.9996 |
| enc | **CTRL: hp_diff** | **−0.0288** | −0.2632 | +0.4051 |
| raw+enc | crown_diff | **+0.2774** | −0.0501 | +0.7730 |
| raw+enc | outcome | −0.1514 | −0.4454 | +0.7696 |
| 噪声地板（打乱局标签） | — | raw −0.167 / enc −0.244 / raw+enc −0.356 | MLP −0.85 / −0.43 / −0.90 | — |

**① 测量本身有功效（这两个对照是本节可信度的前提）**：`raw → time` 与 `raw → hp_diff`
分组 R² = **+1.0000**（目标就是特征的确定性函数）⇒ "分组留出 + 10 局测试"能测出真信号；
`enc → time` = **+0.9999** ⇒ 表征确实保留了 `time`。

**② 表征缺口（新的、具体的、可修）**：`enc → hp_diff` 分组 R² = **−0.03**（逐帧口径
+0.41 也是泄漏）——**表征里读不出塔血差**，而同一个 `enc` 读 `time` 是 +1.00。
即：`time` 走 scalar 通道（3 维：elixir/time/next_card）直接进 `fused`，塔血只以
**grid 每格通道**（`log(hp)/10`、`hp/max_hp`）存在，经 CNN → `grid_ln` → `enc_fc`
→ `relu` → `enc_ln` → GRU 之后**在线性可解码意义上丢失**。
而**所有策略头（`slot_head`/`cell_head`）都只吃 `h`**（`follower.py:454/472`）⇒
**决策通路没有可靠的塔血输入**（塔血是最直接决定胜负的状态量）。

**③ 世界层面：局结果跨局不可预测**（即使给特权真值状态）：
`raw → outcome` 线性 −0.10 / MLP +0.06（噪声地板 −0.17/−0.85）⇒ 与"掷硬币"不可区分。
只有**终局皇冠差**有弱信号：`raw` +0.18、`raw+enc` **+0.28**（均高于地板）。
⇒ 镜像自对弈（同权重、同卡组）的对局结果由后期噪声决定，critic 被判给拟合
"逐帧 GAE 回报"这个**几乎不可跨局预测**的目标，`EV≈0` 是必然。

**④ 结论（G′ 定论）**：
- **critic 侧确实该停**：目标本身不可跨局预测（③），换架构/优化预算都无解；
- **但发现了一个独立的、可修的观测/表征缺口（②）**：塔血状态没有可靠地进入
  决策通路。下一步把它当**独立假设**去验证，而不是当作 critic 的解药：
  **G′-fix = 把双方塔血/皇冠/圣水差做成显式 scalar 通道**（`scalar_dim` 3 → ~11），
  预注册判据：`enc → hp_diff` 分组 R² 从 −0.03 升到 ≈+1（架构改动 ⇒ 必须 `--fresh`）。

### 3.14 G′-fix 落地：塔血/皇冠/圣水差进显式 scalar 通道（2026-09-12，分支 `no-human-watch-B`）

依据 §3.13 的独立发现（`enc → 塔血差` 分组 R²=−0.03，同一 enc → time=+0.9999；
策略头只吃 h）——把它当**独立假设**验证，不当 critic 的解药。

| 文件 | 改动 |
|---|---|
| `rl/observation.py` | obs 新增 `tower_state` **9 维**：我方三塔血量比 + 敌方三塔血量比 + 双方皇冠/3 + 圣水差/10；归一化用本局满血（`battle.tower_max_hp`，缺失退回 lv11 锚），clip 到 [0,1] |
| `rl/env_wrapper.py` | reset 时把 `_blue/_red_towers_max` 写给 `battle.tower_max_hp` |
| `rl/follower.py` | `tower_dim=9`，**追加在 `fused` 尾部**（plan_f/belief_f 之后）⇒ 旧 ckpt 列序是前缀；单条/批量两条编码路径同步；`_tower_state_tensor` 对缺键/长度不符的旧 obs 补零；`enc_dim`/`value_enc_fc` 同步 |
| `rl/follower.py::load_checkpoint` | `enc_fc.weight`/`value_enc_fc.weight` 走"前 v.shape[1] 列拷贝 + 尾零"（尾部追加 ⇒ 旧列语义不变；旧 ckpt/对手池 ckpt 照常加载） |
| `rl/selftest.py` | `test_tower_state_observation`：obs 语义（满血=1/破塔=0/皇冠/费差带符号）、`fused` 尾部接线一致、**只改塔血 enc 必变**、旧 obs 尾零兼容。官方全量 selftest PASS |

**为什么必须 tail 追加而不是插进 scalar 块**：`fused = [grid|hand|scalar|plan|belief]`，
把新列插在 scalar 中间会让 plan/belief 列整体后移 ⇒ `enc_fc` 的"前列拷贝"兼容分支
会**语义错位**（旧权重被接到错误的列上）。tail 追加保证旧列是前缀。

**预注册判据**（跑完 20k 后按此判，`scripts/diag_critic_ev.py --predict`）：
`enc → 塔血差` 按局分组 R² 从 **−0.03 → ≈+1**（该信息现在是 `fused` 的直接线性输入）。
注意判据只验证"观测/表征缺口已关闭"，**不预期**局结果变得可预测（§3.13 ③ 的结论不变）。

**附带修掉一个诊断回归**（同日发现）：`train_solo.eval_and_write` 里
`stats["value_std_ratio"] = None` 写在 `if/else` **之外**，会把刚算好的同窗口比值
立刻清空 ⇒ 评估行 `vstd/rstd` 恒 None（`gfix_20k` 前两点实测丢了这个数）。
已移进 `else` 分支（同时补 `value_std_ev` 的清空）。

### 3.15 G′-fix 20k 验证（`gfix_20k`，进行中）

协议与 F′ 完全一致（同 seed / 同 hist 种子目录 / `--ppo-epochs 4 --ppo-minibatch 32
--ppo-shuffle`），**唯一变量 = `tower_state` 9 维标量**。启动 0 降级 / 0 Traceback，
eval@0 40 局 255.7s（12 worker）。

| step | ES 池化 EV（**更新前**口径，可比） | EVb | h_std | n_abs | 镜像胜率 |
|---|---|---|---|---|---|
| 2500 | **+0.2824** | +0.272 | 0.0404 | 0.550 | 0.325 |
| 5000 | **+0.1574** | +0.155 | 0.0493 | 0.622 | 0.075 |

⚠️ 两点不能定论，但值得记：① EV 为正且是**更新前口径**（F′ 的 +0.10/+0.56 是 in-sample，
不可直读；等 F′ 用同口径复跑才能干净对比）；② `eval@5000` 镜像胜率 0.075（3W/37L）
是红旗——但镜像曲线本身在 cycling 区间剧烈振荡（历史 0.10↔1.0），且 1σ≈0.078，
**必须等更多点 + 对照曲线**再判。

## 4. 明确不做

- 不因为 EV 为负而调 `vf_coef` / 奖励权重——病因已定位为输入饱和，这两项
  诊断报告 §4 已论证与负 EV 无因果。
- 不在修复前继续 200k 长跑（v2 的 200k 已在 step ~54k 处停止，产物保留在
  `runs/prod_200k_valnorm_ev/`，作为"饱和态"基线）。
