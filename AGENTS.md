# AGENTS — 项目红线与决策索引（跨会话必读）

> **本文件 = 红线 + 结论 + 索引。过程细节与历史推理链在 [`docs/agents_archive_2026-09.md`](docs/agents_archive_2026-09.md)（2026-09-13 冻结：**存档头 + 旧 `AGENTS.md` 正文**，正文逐字未删改）。**
>
> **⚠️ 预算纪律（2026-09-13 起）**：工作区指令注入上限 **65,536 B**。旧 AGENTS.md 已涨到
> 100,109 B ⇒ **末尾约 35 KB 对后续会话完全不可见**（实截 65,142 B，见 `docs/_five_reports_merged_2026-09-13.md`）（最近一个月的决策全在末尾）。因此：
> 1. **本文件目标 ≤ 55 KB**（留 10 KB 余量），超了就把细节移进 archive，不要继续堆；
> 2. 新决策的正确写法：**先写 `docs/`（预注册 / 判读 / 诊断）→ 再在本文件加一行指针 + 必须记住的红线**；
> 3. 只增不改历史结论：结论被推翻时**保留原条目并标注"已被 X 推翻"**（推翻本身是重要信息）。
>
> **条目编号可引用**：`【红线 R3】` `【确证 C5】` `【否证 X2】` `【未决 O1】`。

---

## 1. 红线（违反 = 结论作废或事故，无例外）

| # | 红线 | 详情 |
|---|---|---|
| **R1** | **训练期"性能异常"先归因外部，暂停留证等人类** | 吞吐骤降 / 子进程批量崩 / 页面文件报错（历史指纹 **`WinError 1455`**，加载 `torch\lib\cufft64_12.dll` 失败）/ spawn 失败 / 评估卡死 / 进程莫名退出 ⇒ **不自动降级、不为绕开它改代码或下调配置、不把猜测写进注释**。成因未定就写"未定"。判读任何性能数字前先确认本次有没有发生降级（静默降级会把故障伪装成"只是慢"）。经验安全档位 `--eval-workers 12`；16 有 commit 压力风险（实测提交上限 47.3GB / 空闲 20.6GB，16×CUDA-torch≈20.8GB）。**比较两个 run 的耗时前先确认 `eval-workers` 档位**（E2 3419s vs 对照 2709s 的差里含 12 vs 16，不得归因于被比较的改动）。 |
| **R2** | **训练语义兼容红线** | `rl/ppo.py` 被 `run_league` / `flow_league` / `train_follower` / `train_prophet` **共用** ⇒ **函数默认参数必须恒等于旧行为**，新能力只经 `TrainConfig` 显式开启（`value_norm="none"` + `vf_coef=0.5` + `adv_norm="batch"` = 逐位回旧）。 |
| **R3** | **单变量 + 预注册** | 一次只改一个变量；**判据必须在跑之前写死**（含判定分支、失败分支），跑完照单读，不许现编。 |
| **R4** | **阈值判据的基线列必须脚本复算，禁止手抄** | 本会话连错三次（F′ 末4均值手抄 0.400 实际 0.588；C2 基线 1/3 实际 2/3；预注册写"每块 9 点(9,8,8,8,8)"而脚本按序号实切 (9,9,9,9,5)；把无 hist 退化时的**打印配比** 0.714/0.286 当采样配比——打印值≠采样值，实际是 0.571/0.286/0.143）。工具：`scripts/judge_anchor_blocks.py --groups`。**判据的力量必须来自设计**（区间不重叠 / 多跑聚合 / 机制指纹），不能来自对单跑或手算数字的信任。 |
| **R5** | **n=1/臂 分辨率不足** | 同 seed、同代码的两次 run **也不可复现**（前 435 步逐位一致 → 1e-4 漂移 → `eval@800` 已不同；`PYTHONHASHSEED=0` + 单线程 BLAS 无效，机制未定）。20k 单跑 A/B 只能看**大效应**；跨 run 数字不可直接比，判据优先用**同 run 内对照**。⇒ 追溯处置：**此前"单跑看曲线"得出的小效应结论一律降级**；要判读必须多 seed 配对重复，或把评估局数拉大。 |
| **R6** | **架构变更必须 `--fresh`** | `--fresh` **挡不住 `--main-init`**（须显式不传）。旧 ckpt 无 `enc_ln`/`grid_ln` 键时 `load_checkpoint` **静默**保持新初始化 ⇒ 权重与 LN 都不对。新增架构参数必须全仓 `grep load_state_dict` 核对并传播到所有往返构造点。**热启动 `--main-init` 必须显式传 `plan_dim`/`belief_dim`**（旧 ckpt 元数据 57/23 会把 main 建成旧维度 ⇒ `_sync_frozen_copy` shape 失配崩溃；当前 `plan_dim=58`、`belief_dim=563`）。已加护栏：`load_checkpoint` 对缺 `enc_ln.*`/`grid_ln.*` 的旧 ckpt **显式告警**；`_probe["ev_pairs"]` 有 20 万帧上界。 |
| **R7** | **尺度改动必须单常量源 + 对账 selftest** | 奖励汇率 / 值函数 / 闸门 `edw×卡费` / MCTS 值函数必须同源同步改（已两次踩量纲失配：塔伤项 ×1000 使空砸王塔被误判正 EV）。 |
| **R8** | **每个修复配回归测试；selftest 必须跨局边界** | 只看落盘、不跨局边界的 smoke 是盲的（9j 事故：漏 `nonlocal` ⇒ 首局后缓冲泄漏、adv −671、value loss 43 万，30 步冒烟没拦住）。判"是不是我改坏了"以**官方全量 `rl/selftest.py`** 为准（97 项），不是自定义批次顺序。9j 类事故的**日志指纹**：adv 均值 ≈ −惩罚/帧、value loss 数量级漂移、更新连发 ⇒ **看日志头 30 行即可定位**。 |
| **R9** | **测量口径纪律** | ①`step` = **决策帧**不是更新次数；②`ratio≡1.000`/`clip_frac≡0%` 在 `n_epochs=1` 下是**结构性恒等**（该判据已删除，换 EV）；③EV 用**更新前池化**口径（`EVin` 是 in-sample，假象）；④探针必须**按局分组留出**（逐帧随机留出 = 时间泄漏，`corr(R_t,R_{t+1})≈0.99`）；⑤比值门槛的分子分母必须同样本集。 |
| **R10** | **不确定就写不确定** | 宁可写"成因未定 / 样本不足 / 不显著（n=24 时 1σ≈0.22）"，也不要写一个自信的错答案——历史已吃过一次：把未定的性能异常判成"页面文件不够"并写进注释，后续会话会照抄。 |
| **R11** | **明确不做（未获用户另行拍板前）** | 不扩模型参数（629,359）；不上训练时 MCTS / 专家迭代（≈120× 每帧经验成本）；不引入 LLM 实时决策（LLM 只做局间议程提案，文本永不进网络/梯度）；不重构动作语义（`K_MAX=4` 冻结）；不为绕开外部性能异常改代码/降配置；**不把 `main vs 冻结副本` 的 0.85 当"进步"**（自引用失真，见 §3 判读禁则）；**不因 EV≈0 宣称"修好"**（C3）；**不在 A′ 类取证之前改奖励或拆价值头**（该顺序已被证否）；`value_bypass` 默认关闭（证据不支持收益）。 |
| **R12** | **EBK：能算的不许让网络猜** | 私有信息 + 确定性机制（自己的牌序 / 循环 depth / 引擎数值）应在引擎侧算好做**特征**注入，而不是让网络试错学；知识用特征注入（保留学习自由度），**硬闸门只留给已证实病理**。 |
| **R13** | **改判定逻辑必须位图对账** | 动 `legal_cells` / `_position_legal` / 掩码闸门 / `validate_bundle` 前后，各跑 **`scripts/_mask_diff_snapshot.py`** dump 位图（8 状态 × 双方 × 8 手牌 = **128 张**：空场/压境/双倍期/推进坦克/残血塔/群杂/lv14），**逐位 diff 必须全等**才提交（历史：P0 掩码优化 128/128 一致才提交）。 |
| **R14** | **判 critic 好坏不许只看 EV，必须看「参数有没有在动」** | EV≈0 有两种完全不同的成因：**梯度太小**（加量/调系数可能救）与**梯度精确为零**（加量永远救不了）。区分方法（秒级、零成本，直接从 41 个 ckpt 算）：**逐窗算每个张量 `‖ΔW‖/‖W‖`**——若某张量出现"位移恰为 0"的窗口，在 Adam 下只能解释为**该参数梯度精确为 0**（Adam 对 loss 的常数缩放不敏感，任何非零梯度都会给出 ≈lr 的步长）。100k 实测：策略侧 27/27 窗口全在动，价值侧 `value_enc_fc`/`mlp0`/`mlp2.weight` **22~24/27 窗口位移恰为 0**，而 `mlp2.bias` 每窗都动 ⇒ 价值函数 = 常数。**另注意诊断陷阱**：`diag_value_head.py` 打印的 `value_head.weight/bias` 在 `value_independent=True` 下**不参与前向**（37 个参数里唯二逐位冻结者），对着它们做解剖等于什么都没测。见 `docs/d1_long_100k_cause_analysis_2026-09-13.md`。 |

---

## 2. 环境与命令（照抄即可）

**解释器**：基础 `E:\Python313\python.exe`（3.13.12）。WSL 里直调 `/mnt/e/Python313/python.exe -m venv` 是 **no-op**（exit 0 但没生效）；修 venv 必须走 Windows 侧：

```bat
cmd.exe /c "E:\Python313\python.exe -m venv --upgrade E:\clash-royale-simulator-main\.venv"
:: --upgrade 只重建 Scripts/pyvenv.cfg，保留 site-packages，无需重装 torch
```

pip 缓存 `E:\Python313\pip-cache`（经 `E:\Python313\pip.ini` + `PIP_CONFIG_FILE`）。验证：
`.venv/Scripts/python.exe -c "import torch; torch.cuda.is_available()"` → True。

**统一运行姿势**（在 `src/clasher_new` 下）：

```bash
PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe <脚本>
```
> GBK 控制台/重定向管道下，`print("⚠️ …")` 会抛 `UnicodeEncodeError`；若 except 里再用 `{e!r}`
> 二次抛出会**直接崩训练**。已加 `_force_utf8_stdout` / `diagnostics.print_safe` 兜底——
> 新脚本一律自带 UTF-8 stdout reconfigure（本仓库 `scripts/judge_anchor_blocks.py` 是范例）。
> **规则：不要在 catch-all `except` 里 `print` 同一个异常对象**（`{e!r}` 内嵌不可编码字符 =
> 二次抛错路径，会绕过兜底直接崩）。

### 2.1 三种训练模式

| 模式 | 入口 | 语义 |
|---|---|---|
| `solo` | `rl/run_league.py --mode solo` | 固定卡组镜像 + 周期冻结副本 + 对手池（**当前主线**） |
| `run` | `--mode run` | 5 流派卡组模型 + main，PFSP 采样 |
| `flow` | `--mode flow` | 全配对分流派联赛（一次 148,800 局） |

### 2.2 标准 20k 协议（与历史 run 可比，唯一变量改 `--config-name`）

```bash
python rl/run_league.py --mode solo --config economy --config-name <name> --fresh \
  --total-steps 20000 --steps-per-eval 2500 --n-eval-games 40 --eval-workers 12 --device cuda \
  --value-norm running --adv-norm scale --diagnose-every 10 \
  --hist-seed-dir runs/economy_9k_ft --hist-seed-dir runs/economy_9j \
  --ppo-epochs 4 --ppo-minibatch 32 --ppo-shuffle
```
> `opp_mix` **没有 CLI flag**；恢复文档配比（frozen 0.1 / hist 0.6 / defend 0.2 / rand_anchor 0.1）
> 的唯一途径是 `--hist-seed-dir`（可 append）。`economy` 预设 `eval_workers=0`（串行）⇒ **长跑必带 `--eval-workers`**。

### 2.3 100k 长跑 + 评估节奏 C（密锚点 + 稀全块）

```bash
python rl/run_league.py --mode solo --config economy --config-name d1_long_100k --fresh \
  --total-steps 100000 --steps-per-eval 10000 --anchor-every 2500 \
  --n-eval-games 40 --eval-workers 12 --device cuda \
  --value-norm running --adv-norm scale --diagnose-every 10 \
  --hist-seed-dir runs/economy_9k_ft --hist-seed-dir runs/economy_9j \
  --ppo-epochs 4 --ppo-minibatch 32 --ppo-shuffle
```
`--anchor-every` **默认 0 = 旧行为逐位不变**。详见 `docs/eval_cadence_c_2026-09-13.md`。

### 2.4 判读 / 诊断工具

| 工具 | 用途 |
|---|---|
| `scripts/judge_anchor_blocks.py` | **C1 锚点分块判据**；`--groups` 从磁盘复算对照基线（禁止手抄） |
| `scripts/summarize_solo_run.py` | 长 run 诊断汇总（全点表 / 对照 / 锚点 / 对手池 / gates / 异常计数） |
| `scripts/diag_critic_ev.py` | 5 种 EV 口径 + 探针（`--probe/--predict`） |
| `scripts/diag_value_head.py` `diag_encoder_scale.py` `diag_gru_ablation.py` | h/enc 方差、GRU 门、融合层尺度、归一化对照 |
| `scripts/forensics_response.py` `_forensics_cycling.py` | 防守响应/接敌取证；cycling 取证 |

> **脚本对手陷阱**：`ScriptedPolicy(mode="heuristic")` 实为 **mask 随机**（P0-3 时代占位）；
> 要"会防守的脚本对手"必须用 `SelfDefenderPolicy`——其反制落点是**世界坐标**，
> 塞进 `ActionBundle` 前必须做**世界→本地网格逆变换**（P1 有镜像；曾直接塞导致部署非法）。

### 2.5 仪表盘（回放/曲线）

```bash
python rl/dashboard.py --solo runs/<run>/solo_state.json --replays runs/<run>/replays --port 8090
```
> Windows 侧 python 绑定的是 **Windows loopback**：从 WSL `curl 127.0.0.1:8090` 会失败，
> 这是正常的（用它验证"服务挂了"会误判）；用户浏览器访问 `http://127.0.0.1:8090` 正常。

### 2.6 自检

```bash
cd src/clasher_new && PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe rl/selftest.py
```
当前 **97 项 ALL PASSED**（含 `test_anchor_light_point_state`）。

---

## 3. 训练口径（先记住语义，再看数字）

- **1 step = 1 决策帧**；`update_interval=128` ⇒ **20k 步 ≈ 156 次 PPO 更新 ≈ 55~80 局**
  （`max_ep_steps=360`，实测均局 250~360 帧）；100k ≈ 374 局。
  对标 Atari PPO 的 10M~50M 帧，本项目 20k 只有其 1/500~1/2500 ⇒ "20k 没提升"在样本量上就是必然。
- **PPO 预算**必须写进 `run_state`（`config.json` 不参与 resume 解析；续训忘传 `--ppo-*`
  会静默退回 1 次梯度步/更新——这是最贵的脚枪，已有告警）。
- **评估口径**：主判据 = **池化 EV（更新前）**；`EVb` 是 128 连续帧批内口径（**放大约 3×**，只看趋势）；
  `EVin` 是 in-sample（末轮训练后读数，仅用于量化过拟合）。
- **噪声地板（40 局）**：胜率 **1σ≈0.078**（**实测有效**：锚点的 40 局**每个评估点都会重抽**，见【否证 X10】）、
  接敌率 **1σ≈±5pp**；锚点相邻点 |Δ|≈0.225~0.240 = **≈2.05σ_diff**
  （原写"3σ"是把相邻两点当成同一批局 ⇒ 差分 SE 应为 √2σ=0.110；**该数字作废，方向不变**）
  ⇒ 谷底只有 **1 个评估点宽（≲5000 步）**。
- **成本**：纯训练 **26 步/s**（≥2 处独立互证）；一个**全点**（4 块×40 局=160 局）≈**210 s**、
  一个**轻点**（只跑锚点 40 局）≈**53 s**；20k 协议里**评估占 73% 墙钟**。
- **判读禁则**：**不判** main 曲线与单点胜率。机理：`main vs 冻结副本` 结构性≈0.5，且当
  `copy_every` 与 `steps_per_eval` 整除时，"同步→评估"的顺序会让评估对手**恒为刚同步的 main 自己**
  （已改"先评估后同步"）；⇒ **`main vs 冻结副本` 的 0.85 这类读数不得当作"变强"证据**——
  run 内自引用指标（打冻结副本 / 打上一评估点 / 行为门禁全绿）**全部失真**（见 C4）。
  行为指标是**相位型**（D1 r1 的 engagement 同 run 内 39.8→0.3，r2 0.4→0.3，末点 deploy
  11.9/42.5/54.5 横跨对照 12.5~61.9，见 `docs/d1_league_20k_verdict_2026-09-13.md` §4/§8），
  只读 `gates.json` 的**相对退化**，绝对数不跨 run 比。

### 3.1 关键常量（改代码前先对照这里）

| 常量 | 值 | 位置 / 语义 |
|---|---|---|
| `update_interval` | 128 | 每 128 决策帧一次 PPO 更新 |
| `solo_copy_every` | 2000 | 冻结副本同步间隔（同时触发 `refresh_hist` 重扫本 run 快照） |
| `RAND_ANCHOR_EVAL_SEED` | **90000** | 锚点固定**种子**；但**不是**"每点重打同样 40 局"（已【否证 X10】）⇒ 锚点是 **(权重, 父进程牌序链状态)** 的函数，`1σ=0.078` 是真实噪声地板 |
| `RAND_ANCHOR_WARN_FLOOR` | **0.35** | 绝对强度报警线（1σ=0.078；**只报警不阻断**） |
| `THRESHOLDS`（`rl/diagnostics.py`） | `h_std > 0.05`、`n_abs < 0.9`、`value_std_ratio > 0.3` | GRU 活力 / critic 拟合门槛；`value_std_ratio` = value_head 输出 std ÷ 回报 std（**分母必须同窗口**，曾因分母不存在而整条门槛不可判读） |
| `DEFAULT_OPP_MIX`（= `train_solo._OPP_MIX`） | `frozen 0.1 / hist 0.6 / defend 0.2 / rand_anchor 0.1` | D1 后的对手分布（改它要同步两处） |
| `_HIST_POOL_MAX` | 12 | hist 池上限；本目录 ckpt 优先 ⇒ **长 run 会把 `--hist-seed-dir` 的外部补种挤出**（脚本复算：首次"本目录 12"在 **28k** 步，`N(t)=⌊t/2500⌋+1`；`d1_long_100k_verdict` 写的 40k 属口径漂移）。选择规则是 `linspace(0,N−1,12)` **含端点 0** ⇒ `solo_main_0.pt`（未训练起点）**永久钉在 0 号槽**，池成员平均年龄 ≈ t/2 |
| `_PFSP_ALPHA` / `_PFSP_GATE_HI` / `_PFSP_GATE_PENALTY` | 0.20 / 0.85 / 0.2 | PFSP EMA 学习率 / 易胜对手门槛 / 门禁惩罚（`rl/pfsp.py` 默认值仍 = 旧行为） |
| `eval_workers` 安全档 | 12 | 16 有 commit 压力风险（见 R1）；`economy` 预设为 0（串行） |

---

## 4. 已确证（可作为前提引用）

| # | 结论 | 关键数字 / 证据 |
|---|---|---|
| **C1** | **负 EV 的根因是 GRU 输入饱和**（不是奖励尺度、不是价值头结构） | `enc=relu(enc_fc(fused))` 范数 **533**（`grid_feat` 常分量 468、跨帧 std 1.06）→ GRU tanh 候选饱和 `n_abs=0.994` → h 跨帧 std **2.6e-5** → critic 恒常数，`EV_global −0.5817` 与恒等式 `−bias²/Var(R)=−0.584` 吻合。修复 = `enc_ln`（follower.py）。诊断 `docs/value_channel_saturation_diagnosis_2026-09-11.md`。 |
| **C2** | **饱和修复在 100k 尺度耐久**（v3 的正面确认） | 100k 全程 `h_std` 0.067→0.095（>0.05）、`n_abs` 0.667→0.826（<0.9）；100k 结构下无一个点越过门槛。证据 `docs/d1_long_100k_verdict_2026-09-13.md` §4。 |
| **C3** | **但 critic 仍不拟合，且加训练量救不了** | EV 仅前两点为正（+0.234/+0.385），其后 **9/11 点 ≈0 或负**；`vstd/rstd` 30k 后长期 **≈0.001**（比 20k 的 0.0035~0.0225 更差）；run 自身 8 次 vitality 告警。⇒ **"加量救 critic"被 100k 直接否证**（【否证 X4】）。 |
| **C4** | **自对弈 cycling 确凿** | RPS 三角：main 打冻结副本 0.85 / 打训练起点 0.13 / 打全新随机 0.505；锚点长期低位甚至整段 0.000；对照曲线相邻点可差 0.6+。取证 `scripts/_forensics_cycling.py`。 |
| **C5** | **D1（对手分布去镜像化 + 动态历史自身联赛 + PFSP 门禁）消除"整段输给随机策略"的相位** | 20k 三跑最差锚点 **0.125/0.200/0.250** vs 无干预 **0.000/0.000/0.025**（区间不重叠，Welch t≈4.9）；机制指纹：镜像自对弈 40%→**6.6%/13%/~13%**、本 run 快照 1→9 进池。`docs/d1_league_20k_verdict_2026-09-13.md`。 |
| **C6** | **D1 的防崩在 4× 训练量（100k，≈374 局）下成立** | P1 块 worst **0.388/0.475/0.275/0.237/0.525**（mean 0.380、min 0.237）vs 无变化组 [0.000,0.025]；与 D1 20k 区间 [0.125,0.250] 比 **min 0.237 落在其中 ⇒ 无统计差别**（不退化，也没证明改善）。`docs/d1_long_100k_verdict_2026-09-13.md`。 |
| **C7** | **评估节奏不能均匀放宽** | 锚点谷底只 1 个点宽；真实序列粗采样回放：5000 步即开始漏真谷底、10000 步把"最差点"从 0.192 抬到 **0.462**（病态组持续塌陷留得住、健康组周期性瞬态留不住）。⇒ 用"密锚点 + 稀全块"（C 方案）。`docs/eval_cadence_c_2026-09-13.md`。 |
| **C8** | **critic 的失败是"状态→回报"层面的**，不是表征没信息 | 按局分组留出后探针 R² 为负（MLP −0.36/−0.45）；监督微调同网络 test EV 可到 +0.22~+0.29 而 PPO 联合训练 ≤0 ⇒ 结构够用、主因在训练过程（这一条已被后续"critic 侧闭合"限定，见【否证 X6】）。 |
| **C9** | **行为链三层有真实爬坡**（winrate 平台期下） | 防守响应率 38.1%→**61.5%**（v2 通道）；接敌率 7.4%→**10.5%**、中段带部署占比 51%→59.7%、追尾 64%→31%（9k 拦截几何，提交 94ce339）；单边堆牌 45.8%→9.5%。 |
| **C10** | **词表/卡池/工具层的事实** | `ENTITY_NAMES` 13→**177**（旧 13 位序冻结，行拷贝兼容）；belief token 71→563；`PLAN_DIM` **58**（hint 8 维）；`build_card_pool` 塔类过滤唯一正确口径 = `n.startswith("King_")`；外置工具①②③ 已落地并有引擎对账 selftest，④ 未实现。 |

---

## 5. 已否证 / 已关闭（**勿重走**，除非有新证据）

| # | 已否证的事 | 依据 |
|---|---|---|
| **X1** | F′ 首跑训练期 EV **+0.198** 是 in-sample 假象 | 修正口径后 **+0.031**（降 6.4×）；同权重离线 50 局 `EV_global` 为负。`docs/fprime_rerun_20k_verdict_2026-09-13.md` |
| **X2** | "表征含 24~42% 可预测信息、critic 吸收不了" | **逐帧随机留出 = 时间泄漏**（相邻帧 corr(R)≈0.99）；按局分组 R² 全负；换标签/打乱地板救不了。 |
| **X3** | 价值头架构四代（G'-fix 塔血通道 / B′ value 直连 enc / E′ 独立价值编码器+非线性头 / 重量级 bypass） | 训练期 EV 与离线 EV 全部 ≤0；探针复现显示 G'-fix 的 `enc→塔血差 +0.69` 是**通道接上了**，但不是 EV 改善。 |
| **X4** | "加训练量能救 critic" | 100k：EV 9/11 点 ≈0 或负、`vstd/rstd` ≈0.001（更差）。见 C3。 |
| **X5** | E2（把固定随机锚点按 10% 放进训练对手池）能破 cycling | 20k 末点 vs `baseline0` 0.075、vs `baseline_rand` 0.050，RPS 仍在转；10% 弱锚点压不住自对弈漂移。 |
| **X6** | **critic 侧程序已按预注册闭合**（用户预授权：无进展就转策略侧） | 三步（测量口径修复 → 单变量 A/B → 复跑对比）全部"无进展"⇒ 不再往 critic 加投入；转策略侧（对手分布 / cycling / 上限）。 |
| **X7** | 两条预注册判据因**基线手抄**失去区分力 | 20k 的 P1/P2（F′ 末4均值 0.400 vs 实际 0.588）与 C2（1/3 vs 实际 2/3）⇒ 立 **R4**。 |
| **X8** | early-stop 低置信裁定"降噪 28~40%" | 那是 eval 侧统计口径混入；训练侧实测早停率 **3%**。 |
| **X9** | `gates.json` 的绝对阈值判据 | 阈值 9.5 来自一次性脚本，对同一批权重实测 28.3~38.1（差 3×）⇒ 已改**相对本 run 首点**（engagement ≥50%×起点、ghost ≤2×起点）。 |
| **X10** | 「每评估点重打同样 40 局 ⇒ 锚点是权重的确定性函数」（原 `AGENTS.md` §3.1 / `eval_cadence_c` §5） | **实测 11 个评估点 11 种不同首局牌序**。`RLEnv.reset()` 是**原地链式洗牌**（`env_wrapper.py:349-357`：`deck0=list(self.deck0)` → `shuffle` → `self.deck0=deck0`），而并行评估 worker 的起点是 `env_kwargs["deck0"]=list(env.deck0)`（`train_solo.py:968`）=**父训练进程被反复覆盖后的当前牌序** ⇒ 固定种子只固定了 40 局之间的相对偏移，没固定起点。**后果**：`1σ=0.078` 有效（判据仍可用），但相邻点差分 SE=`√2σ=0.110`，故"|Δ|≈0.225≈3σ"应读作 **≈2.05σ**；C7 的"谷底 1 点宽"方向不变。见 `docs/d1_long_100k_cause_analysis_2026-09-13.md` §2.4。 |

---

## 6. 未决与候选下一步

| # | 未决问题 | 现状 | 候选下一步（含判据设计要点） |
|---|---|---|---|
| **O1** | **上限 / 绝对强度**没动 | D1 20k 与 100k 在最差锚点上无差别；次判据（块中位数末−首）口径敏感（+0.125 或 +0.062）⇒ 按预注册**不判决** | ①**D2 = `rand_anchor` 0.1→0.3**（剂量-反应，单变量）；②若要判"上限"，先设计比"滚动均值"更稳的统计量（建议：**块内中位数 + 多跑聚合**，且基线脚本复算）；③注意**固定随机锚点目前是唯一能测绝对强度的仪器**——run 内自引用指标（打冻结副本/打上一评估点）全部失真（§3 判读禁则） |
| **O2** | **critic 拟合** | **机制已定位（2026-09-13）**：不是"训练过程未解"，而是**价值头隐藏层对几乎全部帧输出恒零**（`value_enc_fc`/`mlp0`/`mlp2.weight` 在 35k→100k 的 22~24/27 个窗口里**位移恰为 0**＝梯度精确为零，而 `mlp2.bias` 每窗都动）⇒ 价值函数 = 常数。第一性原因：价值编码器末端 `LayerNorm` 把近常数输入归一化 ⇒ 价值头可表达跨帧方差 ≈0.14，而目标 std≈12.11 ⇒ 差 60~600 倍 ⇒ 自举失败 ⇒ ReLU 逐个死亡锁死（吸收态）。**已排除**"梯度小/被 `1/s²` 压制"（Adam 对常数缩放不敏感；唯一非零梯度的 bias 跑满速）。见【红线 R14】+ `docs/d1_long_100k_cause_analysis_2026-09-13.md` | ①**先做最便宜的惰性检验**：把优势换成纯 REINFORCE+运行均值基线，若 20k 末点差 <0.11 即证明"critic 当前贡献≈0"；②再做单变量修复（去掉末端 LN **或** 目标归一化 `R/s` 后输出 `×s`），先决是加存活率/输出方差探针；③**不要再靠加量**（X4），④不要再"换一代价值头架构"（X3 四代全败，且四代都在同一形态上打转） |
| **O3** | **行为病理未修完** | `elixir_avg` 仍低（不会攒费）；`setup_wait` 冷启动死锁已定位但**两阶段状态机未实现**；单边堆牌末点 100% | 攒费链重构（不依赖血牛在手的窗口条件）+ 观测/plan 通道配合；判据用 gate 相对退化 + 接敌率 |
| **O4** | 训练期 cycling 只"防崩"未"破除" | D1 让最差锚点不再归零，但 RPS 循环仍在（对照曲线大幅摆动） | D2 剂量；或提高 `_HIST_POOL_MAX`（12→24，长 run 的自身联赛更密）；或课程式提高 `defend` 权重 |
| **O5** | 同 seed 不可复现（机制未定） | 前 435 步逐位一致后 1e-4 漂移；`PYTHONHASHSEED=0` + 单线程 BLAS 无效（嫌疑 CUDA 非确定算子） | 属**方法学**问题：当前靠"多跑聚合 + 大效应判据"绕过；若要定位，需逐算子确定性差分（未做） |

---

## 7. 计划台账（所有计划的定位与状态）

> 完整整合视图（含每项计划的改动清单、v1→v2 修订点、红线全文）见 **[`docs/plan_master.md`](docs/plan_master.md)**。

| 计划文档 | 定位 | 状态 |
|---|---|---|
| `docs/rl_review_fix_plan.md` | 评审版修复手册：6 Critical + ~20 Major + ~40 Minor，逐条配回归测试 | **已落地**（6/6 Critical 有回归测试；§6 的 11 个测试名 2026-09-13 核查全在 `rl/selftest.py`） |
| `docs/training_audit_2026-09-11.md` | 训练体系审计（四问：训练问题/指标/步数预算/参数量） | 已定稿，是 v1 的依据 |
| `docs/rl_training_fix_plan_v1.md` | 整改计划 v1（P0 通道+指标 → P1 结构 → 上量；含 G1/G2/G3 门禁、单变量协议） | 部分被 v2 取代（判据/门禁/预算口径三处已改） |
| `docs/rl_training_fix_plan_v2.md` | v1 修订版：G1 判据 `ratio`→**EV**、门禁改**相对首点**、预算改**局数**、对手池 `0.7/0.3/0.2`→`0.5/0.3/0.2`、**先评估后同步** | §1-3、§5、§6 已落地；**§4 病因判断被 v3 推翻**（见【否证 X2/X3】上下文） |
| `docs/rl_training_fix_plan_v3.md` | **当前生效**：GRU 饱和病因链 + P0-A/B/C + 各验证跑判读（5k/20k/grid_ln/A′/E1/E2）+ 门槛与分支 | P0-A/B 已落地；P0-B-2 第 3 项曾因**分母不存在**不可判读（已补）；P0-C 仅静态护栏；A′/E1 已落地，E2 被证伪 |
| `docs/value_channel_saturation_diagnosis_2026-09-11.md` | 负 EV 真根因诊断（GRU 输入饱和） | 已确证（C1） |
| `docs/critic_probe_experiment_2026-09-12.md` | critic 可预测性探针（含时间泄漏发现） | 已确证（X2） |
| `docs/rl_reward_plan_v2.md` | 奖励与计划通道重构 v2（逐帧账本、两段相位、闸门） | 前置依赖**已落地**；Phase 2 由 plan v1 承接；僵局早停保留未动 |
| `docs/rl_plan_design_v1.md` | PlanToken 战术意图 v1（结构先行 + bp/pp 组意图 + 消融） | **已落地**（`PLAN_DIM=58`，文头"57 维"已过期）；`bait` 留 backlog（无可靠触发） |
| `docs/ai_training_plan.md` | RL 算法闭环总纲（先知/信念/跟随者/POMDP/联赛/同刻多卡） | 代码按要求已落地；§8 阶段验收与 §11.3 N=200/500 协议**无落地记录** |
| `docs/P0-mechanics-plan.md` | 游戏机制补全（族 1-8） | 引擎机制**全部完成**（test_m2 72/72、batch_smoke 156/156）；瓶颈在数据层 |
| `docs/mcts_design.md` | 推理时浅 MCTS（零训练风险，量"搜索赚多少 Elo"） | `rl/mcts.py` + 2 个 selftest 已落地；**评估接入（`evaluate.py --mcts`）未落地**（全仓 `RLMCTS` 仅 selftest 调用） |
| `docs/four_decks_manual.md` | 四卡组逐卡手册（速猪/石头人/X弩/巨骷髅攻城槌） | 已定稿，供 `--deck-set four` |
| `docs/cycling_league_plan_2026-09-13.md` / `docs/d1_long_100k_prereg_2026-09-13.md` | D1 与 100k 的**预注册** | 均已执行并判读（C5/C6） |
| `docs/eval_cadence_c_2026-09-13.md` | 评估节奏 C 方案（密锚点 + 稀全块） | 已落地（`--anchor-every`，默认关） |

---

## 8. Run 台账（结论 + 证据路径）

| run / 实验 | 结论一句话 | 证据 |
|---|---|---|
| `ab_valnorm_20k` | `step`=决策帧、`ratio` 判据作废、critic EV≈0 首次发现、门禁阈值口径错位 | `docs/ab_valnorm_20k_verdict_2026-09-11.md` |
| `prod_200k_valnorm_ev` | 跑到 ~54k 停：EV 全负；**成为饱和态基线**（不可用 `main_init` 续训） | `docs/value_channel_saturation_diagnosis_2026-09-11.md` |
| `fix_gru_ln_5k` / `_20k` | 饱和修复生效（n_abs 0.994→0.46~0.53）；critic 仍常数；查出**融合层尺度失衡**（grid 占 fused 98%） | `docs/train_fix_gru_ln_*.log` |
| `fix_gru_ln_norm_20k`（`grid_ln`） | 尺度修复达成（101→5.87）只修"对中"不修拟合；`n_abs` 回升 0.745（耐久性存疑）；`vs baseline0` 崩塌 = cycling 红旗 | `docs/train_fix_gru_ln_norm_20k.log` |
| `byp_cprime_20k` / `eind_20k` / `fprime_20k` / `fprime_ev_20k` | B′/E′/F′/G′-fix 四代架构干预：EV 全部 ≤0；F′ 真预算保留但未过零 | `docs/train_{byp_cprime,eind,fprime,fprime_ev}_20k.log` |
| `e2_rand_anchor_20k` | 训练侧弱锚点 10% **未破 cycling** | `docs/train_e2_rand_anchor_20k.log` |
| `d1_league_20k{,_r2,_r3}` | **防崩有效**（最差锚点区间不相交）；上限未证明；两处预注册标定错误自披露 | `docs/d1_league_20k_verdict_2026-09-13.md` |
| `d1_long_100k` | **P1 PASS**（防崩在 4× 量级成立）、与 20k 无统计差别、上限仍未解决、加量救不了 critic | `docs/d1_long_100k_verdict_2026-09-13.md` + `docs/diag_d1_long_100k.md` |

---

## 9. 文档地图

| 想做什么 | 去哪 |
|---|---|
| 项目介绍 / 快速开始 | [`README.md`](README.md)（中文）、[`readme_en.md`](readme_en.md) |
| **所有计划的整合视图**（主线 / 已确证 / 已否证 / 未决 / 优先级） | [`docs/plan_master.md`](docs/plan_master.md) |
| **当前问题的归因**（critic 为何塌成常数 / 上限为何不动 / 判别性实验） | [`docs/d1_long_100k_cause_analysis_2026-09-13.md`](docs/d1_long_100k_cause_analysis_2026-09-13.md) |
| 该归因的结构化取证全文（6 路并行 + 评审，含 3 处已更正的口径错误） | [`docs/report_d1_long_100k_structured_2026-09-13.md`](docs/report_d1_long_100k_structured_2026-09-13.md) |
| 本文件的过程细节与历史推理链 | [`docs/agents_archive_2026-09.md`](docs/agents_archive_2026-09.md) |
| 文档 ↔ 源码完整索引 | [`docs/README.md`](docs/README.md) |
| RL 代码导读与训练入口 | [`src/clasher_new/rl/README.md`](src/clasher_new/rl/README.md) |
| 自检 / 回归 | `src/clasher_new/rl/selftest.py`、`scripts/test_m*.py`、`scripts/batch_smoke.py` |
| 判读工具 | `scripts/judge_anchor_blocks.py`、`scripts/summarize_solo_run.py`、`scripts/diag_*.py` |
