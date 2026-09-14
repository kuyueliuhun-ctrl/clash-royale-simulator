# 价值通路探针 v3 · 预注册（2026-09-14）— 前端丢在哪一段 + 把比值判据做成正式判据

> **跑之前写死**（【红线 R3】）。判据、闸门、分支、失败分支在下面全部固定；跑完**照单读**，
> 不许临场改阈值或增删分支。
> 前两轮：`docs/value_ln_probe_prereg_2026-09-14.md`、`docs/value_ln_probe2_prereg_2026-09-14.md`
> ⇒ 判读 `docs/value_ln_probe_verdict_2026-09-14.md`（`L0_INVALID`，立 **R15**）、
> `docs/value_ln_probe2_verdict_2026-09-14.md`（`V2_INVALID`，立 **R16**）。
> 仪器：`scripts/probe_value_ln.py --ladder v3`（**只读**，不改训练路径、不写 ckpt）。

---

## §0 本轮要回答的两个问题

**Q1（判据类）**：把"前端丢掉约 2/3 线性可读局内信号"从一个**观察到的比值**升级为一条**正式判据**
——即：先量出该比值在**多条独立轨迹**上的分布，再据此定阈值（满足【红线 R16】），
并在 4 条轨迹上逐条判决。

**Q2（定位类）**：这个损失**落在共享感知前端的哪一段**——
`obs 编码` / `CNN` / `grid_ln` / `concat` / 非 grid 分块？

---

## §1 机理：前端路径（源码 `rl/follower.py:268-298`）

```
obs
 ├─ grid (32,18,15)
 │     card_ids = grid[...,0] ──→ entity_emb(8)              (1,32,18,8)
 │     rest     = grid[...,1:]                               (1,32,18,14)
 │     card_type_oh = one_hot(grid[...,3], 4)                (1,32,18,4)
 │     x = cat([rest, card_vecs, card_type_oh]) → permute    (1,26,32,18)   = 14976 维
 │     cnn_pre_ln = cnn(x)  [conv32 → conv64/s2 → conv64/s2 → flatten]      = 2560 维
 │     grid_ln_out = grid_ln(cnn_pre_ln)  [nn.LayerNorm(2560)]              = 2560 维
 ├─ hand (40) → entity_emb → hand_f                                        =   40 维
 ├─ elixir/time/next_card → scalar                                         =    3 维
 ├─ plan_token   → plan_mlp + ReLU → plan_f                                =   64 维
 └─ belief_token → belief_mlp + ReLU → belief_f                            =   64 维
fused = cat([grid_ln_out, hand_f, scalar, plan_f, belief_f])              = 2731 维
```

**关键结构事实（本轮的逻辑基础）**：
1. `fused ⊇ grid_ln_out`（前者是后者的超集）⇒ **`EV(fused) ≥ EV(grid_ln_out)` 在期望上必然成立**。
   这是一条**不含阈值**的自洽性检查（闸 **G-MONO**），也是本轮"损失不可能发生在 `grid_ln_out → fused`"的依据。
2. `grid_ln_out = LayerNorm(cnn_pre_ln)` ⇒ `cnn_pre_ln` 的信息是 `grid_ln_out` 的**超集**
   （LN 只丢掉每帧的常数方向与尺度这 2 个标量）⇒ **`EV(cnn_pre_ln) ≥ EV(grid_ln_out)`** 期望成立。
   ⇒ **`EV(cnn_pre_ln) ≫ EV(grid_ln_out)` 只能解释为"LN 丢掉了按幅度编码的信号"**。
3. `x` 是 CNN 的**唯一**输入 ⇒ **`EV(cnn_pre_ln) ≤ EV(x)`** 期望成立（数据处理不等式）。
4. `x` 与被测参考集 `raw_obs = feat_obs(obs)` **内容相同、编码不同**：
   `raw_obs = [grid(15) 原样 | hand | elixir | next_card | time]`（8648 维），
   `x = [grid 的 14 个非 id 通道原样 | entity_emb(card_id) | card_type onehot]`（14976 维）。
   ⇒ `raw_obs` vs `x` 的差 = **卡 id 的 8 维学习嵌入 vs 原始整数通道 + 通道重排**，
   不含任何"网络看不到的信息"。

---

## §2 仪器扩展（为什么扩展本身不构成第二个变量）

`--ladder v3` 只在**同一次前向**里多抓张量，**不改变** rollout 的动作、奖励、GAE 目标里的任何一个数。
因此"层数变多"是**测量**的扩展，不是**处理**的改变（【红线 R3】的单变量纪律不受影响）。

**并且要当场验证这一点**（闸 **G-REPRO**）：v2 权威跑（seed 7、30000 帧、stoch）的读数是
**可复算的**（v2 判读 §2 复现闸 PASS，同命令两次逐位一致）。因此：
> **用 v3 仪器重跑同一个 `(ckpt, seed, frames, mode)`，必须复现 v2 的 `EV_within(raw_obs)` 与
> `EV_within(fused)`；否则说明仪器改变了轨迹 ⇒ 本轮一切新读数作废。**

---

## §3 权威协议

- ckpt：`runs/critic_inert_probe_20k/solo_main_20000.pt`（`value_independent=True`，hidden=128）
- mode：`stoch`；`--frames 30000`；device：CPU（该网络 CPU/GPU 同速）
- seed：**7 / 11 / 13 / 17**（seed 7 = 与前两轮对齐的锚点；四条 = 比值分布的分母）
- `--raw-obs --ladder v3 --seed-rng`（默认播种，闸 10）
- **seed 7 额外** `--cnn-input`（多抓 `x`，14976 维）；其余 3 条**不抓**（内存/时间）
  ⇒ `x` 上的读数**只作描述性证据**，用于把"CNN 前"再拆成"CNN"与"输入编码"，
  **不参与任何形式判决**（单条轨迹，无分布）。
- 两阶段执行（省内存、且 npz 留证）：先 `--rollout-only --save-npz`，再 `--npz` 离线逐层拟合。

### §3.1 阶梯（v3）

| 键 | 维度 | 位置 | 四条 seed |
|---|---|---|---|
| `raw_obs` | 8648 | 参考集（与第二轮 A 集同构造） | ✅ |
| `raw_nongrid` | 43 | `raw_obs` 去掉 grid 块（hand 40 + elixir/time/next 3） | ✅ |
| `cnn_pre_ln` | 2560 | CNN 输出、`grid_ln` **之前** | ✅ |
| `grid_ln_out` | 2560 | `grid_ln` 之后（= `fused` 的第 0 块） | ✅ |
| `hand_f` / `scalar_f` / `plan_f` / `belief_f` | 40 / 3 / 64 / 64 | `fused` 的其余 4 块 | ✅ |
| `fused` | 2731 | 价值编码器输入 | ✅ |
| `enc` | 128 | **共享**编码器输出（策略 GRU 的输入；描述性） | ✅ |
| `pre_ln` / `relu_ln` / `post_ln` | 128 | 价值支路（第二轮已测，本轮作一致性对照） | ✅ |
| `mlp0_post` / `value` | 64 / 1 | 价值头（已知退化） | ✅ |
| `grid_x` | 14976 | CNN 输入 | **仅 seed 7** |

### §3.2 估计器与切分（与第二轮逐位同一套，禁止两套估计器）

`Ridge + α 网格 v2(下界 1e-6) + 输出裁剪 + select=within`；目标 = GAE 回报**按局中心化**；
按**局**分组切分 `test 0.25 / val 0.15`（【红线 R9④】逐帧随机留出 = 时间泄漏，禁止）。

---

## §4 闸门（跑前写死；每组闸门都标注它的**认识论地位**）

| 闸 | 判据 | 地位 |
|---|---|---|
| **G-REPRO** | seed 7 必须复现：`frames == 29991` 且 `\|ΔEV(raw_obs) − 0.12145137497621039\| ≤ 0.002` 且 `\|ΔEV(fused) − 0.04312000460119425\| ≤ 0.002` 且 `\|ΔVar(R)_within − 58.83\| / 58.83 ≤ 0.01` | **仪器惰性闸**。0.002 的依据：该量在**不同轨迹**上散布 0.1215~0.2999（Δ≈0.18），0.002 = Δ 的 1.1%；而**同轨迹**复算在 v2 已实测**逐位一致**（|Δ|=0）。⇒ 余量 0.002 远大于数值噪声、远小于轨迹噪声。**不过 ⇒ `V3_INVALID`，本轮全部新读数作废** |
| **G-RATIO** | `ρ(fused) := EV(fused)/EV(raw_obs) ≤ 0.5`，在 **4/4 条 seed** 上成立 | **本轮的主体判据（第 11 道闸门「比值闸」）**。阈值 0.5 的标定依据见 §5 |
| **G-ANCHOR** | `EV(raw_obs) ≥ 0.05` 且 `EV(raw_obs) ≥ 5 × max(EV(打散对照))` | **仅退化护栏**（防止比值分母是噪声）。**声明：它不是判别器**——它单独不能产生任何判决；0.05 取在三条已测轨迹最小值 0.1215 的 2.4× 之下 |
| **G-PC** | 阳性对照 `EV_within ≥ 0.15` | 估计器电池（合成目标，可达到的 EV 由构造决定：`within_var=58 / between_var=166`） |
| **G-CLK** | 时钟基线（局内第几帧）`EV_within ≤ 0.05` | 估计器电池 |
| **G-SCR** | 状态打散（`raw_obs` / `fused` / `grid_ln_out`）`EV_within ≤ 0.05` | 估计器电池（**证伪对照**） |
| **G-VAR** | `Var(R)_within > 1` | 估计器电池 |
| **G-MONO** | 对每个 `fused` 的分块 `B`：`EV(fused) ≥ EV(B) − 0.01·EV(raw_obs)`；且 `EV(cnn_pre_ln) ≥ EV(grid_ln_out) − 0.01·EV(raw_obs)`；且（seed 7）`EV(grid_x) ≥ EV(cnn_pre_ln) − 0.01·EV(raw_obs)` | **集合包含自洽闸**（不含阈值的方向性检查，容差 = EV(raw_obs) 的 1%）。违反 ⇒ 该对比较不可用，须报告为估计器噪声 |

**闸门集 V3 = {G-REPRO, G-RATIO, G-ANCHOR, G-PC, G-CLK, G-SCR, G-VAR, G-MONO}**
（v1 的 `G-UP` 与 v2 的 `G-RAW` **均不计入**：【R15】不许跨实验照抄阈值，【R16】不许单次观测标定阈值。）

---

## §5 G-RATIO 的阈值从哪来（【红线 R16】的合规性声明）

正式判据**不许**用单次观测标定。已知的三条轨迹（前两轮）上 `R0/E0`：

| 来源 | `EV(raw_obs)` | `EV(fused)` | `R0/E0` = `1/ρ(fused)` | `ρ(fused)` |
|---|---|---|---|---|
| ① round-2 cache（未播种，32982 帧；`fused` 来自 v1 运行 ⇒ **跨轨迹**） | +0.2999 | +0.0884 | 3.39× | 0.295 |
| ② v2 首次运行（未播种，29654 帧） | +0.1629 | +0.0534 | 3.05× | 0.328 |
| ③ v2 权威运行（播种，29991 帧） | +0.1215 | +0.0431 | 2.82× | 0.355 |

已观测 `ρ(fused) ∈ [0.295, 0.355]`，**极差 0.060**。
阈值取 **0.5**：距观测上界 **0.145**，**余量/极差 = 2.4×**。
⇒ 相对 R16 的两个反面样板（阈值 0.15 落在 0.12~0.30 散布**内部**）已是**量级上的改善**。
本轮再用 4 条独立（已播种、可复算）轨迹**重新量一遍** `ρ` 的分布，并**报告 min/max/mean**；
若 4 条中有任意一条 `ρ(fused) > 0.5` ⇒ **`G-RATIO` FAIL ⇒ 判 `RATIO_WEAK`（该判据作废，回退为描述量）**。

---

## §6 分支（逐条报命中；分支**只用比值**，不用绝对阈值）

记 `ρ(L) = EV_within(L)/EV_within(raw_obs)`（同 run / 同切分 / 同估计器 ⇒ **配对**）。

- **B-LN**（`grid_ln` 是掉点层）：
  `EV(cnn_pre_ln) ≥ 0.5·EV(raw_obs)` **且** `EV(grid_ln_out) ≤ 0.5·EV(cnn_pre_ln)`
- **B-CNN**（CNN 是掉点层）：
  `EV(cnn_pre_ln) < 0.5·EV(raw_obs)` **且**（seed 7）`EV(grid_x) ≥ 0.5·EV(raw_obs)`
- **B-INPUT**（网络输入编码是掉点层）：
  `EV(cnn_pre_ln) < 0.5·EV(raw_obs)` **且**（seed 7）`EV(grid_x) < 0.5·EV(raw_obs)`
  ⇒ 指向 `entity_emb(card_id)` 的 8 维学习嵌入 / 通道布局，而不是卷积
- **B-NONGRID**（信号本在非 grid 块、grid 只是旁观）：
  `EV(raw_nongrid) ≥ 0.5·EV(raw_obs)` ⇒ 上游可读性主要来自 hand/scalar，
  则"前端丢 2/3"的解读要改写为"grid 块本来就没多少线性可读信号"
- **B-ENC**（共享编码器 `enc` 是掉点层）：
  `EV(cnn_pre_ln) ≥ 0.5·EV(raw_obs)` **且** `EV(grid_ln_out) ≥ 0.5·EV(cnn_pre_ln)`
  **且** `EV(enc) ≤ 0.5·EV(fused)`
- **B-SHALLOW**（前端没问题，损失在价值支路内）：
  上面所有层都 `≥ 0.5` 的上游，但 `EV(pre_ln) ≤ 0.5·EV(fused)` —— 与第二轮的
  "`value_enc_ln` 不是掉点层"直接冲突 ⇒ 须报告为**对第二轮读数的反证**

以上分支**互斥**，按 B-LN → B-CNN → B-INPUT → B-NONGRID → B-ENC → B-SHALLOW 的顺序判定（先命中先报），
并**全部逐条打印真值**（不许只打命中项）。

---

## §7 失败分支（跑前写死）

| 情形 | 处置 |
|---|---|
| `G-REPRO` FAIL | `V3_INVALID`。仪器非惰性 ⇒ **本轮所有新读数作废**，只报告"扩展仪器改变了轨迹"这一事实与差多少 |
| `G-RATIO` FAIL（≥1 条 seed `ρ(fused) > 0.5`） | `RATIO_WEAK`。「前端丢 2/3」**退回描述量**，写进台账；不据此改架构 |
| 其它估计器闸门 FAIL | `V3_INVALID`（估计器不可用），但**报告 G-REPRO 是否 PASS**（那仍证明仪器惰性） |
| `G-MONO` FAIL | 该对比较作废，其余读数照报，并明确标注"含子集关系的层间比较在本轮不可用" |
| 分支全不命中 | `NOT_PREREGISTERED`：**不许**临时造新分支；写"定位不成功"，列出全部 `ρ` 供下一轮设计 |
| `--cnn-input` 因内存/时间失败 | seed 7 的 `x` 读数缺失 ⇒ B-CNN/B-INPUT **不可判**（报 `B-CNN-or-earlier`），其余分支不受影响 |

---

## §8 成本预算

- rollout：~30000 帧 × 4 条，CPU 单进程 ~20.3 f/s（实测 smoke：2386 帧 / 117.8 s）
  ⇒ 单条 ~25 分钟。**峰值内存实测口径**：v3 阶梯 30000 帧 = ~3.9 GB（seed 7，含 `grid_x` 14976 维）、
  ~2.3 GB（其余 seed）；`acc`(list) → `X`(ndarray) 若不释放列表会**翻倍**（已修）。
  WSL 可用内存 ~12.8 GB ⇒ **分两批跑**：批 1 = seed 7/11/13（~8.5 GB），批 2 = seed 17（~2.3 GB）。
  ⚠️ 这是**资源计划**的修订（判据/闸门/分支一字未动），写在**开跑之前**。
- 拟合：每次 `fit_layer` 在 8648 维上 ~45 s、2560 维上 ~25 s、14976 维上 ~5 分钟。
  逐层离线跑，~15 分钟；`grid_x` 单层另计 ~5 分钟。
- 上限：**≤ 1.5 小时墙钟**，不占 GPU、不写 ckpt、不动 `rl/` 一行。

---

## §9 明确不做

- 不改任何架构/损失/奖励（本轮是**测量**，不是干预）；
- 不用 v3 的读数直接改 `grid_ln`（改架构须先有 §6 的定位 + 独立的 A/B 预注册）；
- 不把 seed 7 的 `grid_x` 读数当形式判据（单轨迹）；
- 不重跑未播种的旧轨迹当"分布"（【否证 X-16】：未播种轨迹不可复算）。
