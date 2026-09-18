# AGENTS 分册 · 环境与命令

> **来源**：原 `AGENTS.md` §2（2026-09-18 拆分，**逐字未改**）。
> **节号保持不变** ⇒ 历史文档里的「`AGENTS.md` §N.M」引用经索引一跳可达。
> 返回索引 → [`../../AGENTS.md`](../../AGENTS.md)

> 本册内容：解释器 / venv / 三种模式 / 协议 / 工具 / 仪表盘 / 自检。

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
| `solo` | `rl/run_league.py --mode solo` | 固定卡组镜像 + **自我对弈**对手池（frozen/hist/**SelfDefender**/锚点） |
| `run` | `--mode run` | **多卡组**：5 脚本 agent（**均 mask-随机**）+ main + Elo。**2026-09-17 用户选定主线**；⚠️ 对手弱/无自我对弈 ⇒ `docs/run_mode_multideck_2026-09-17.md`。**2026-09-18 起**：两级评估（`--big-eval-every`/`--n-eval-games-big`，默认关）+ 评估并行分片（`--eval-workers`，**默认仍串行**）⇒ `docs/long1m_prereg_2026-09-18.md` |
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
> 的唯一途径是 `--hist-seed-dir`（可 append）。**⚠️ 2026-09-14 更正（脚本实测，见 `docs/training_method.md` §B.9）**：`economy` 预设**并不设置** `eval_workers`，实际继承 dataclass 默认 `config.py:209` = `min(16, os.cpu_count())` ⇒ 本机（16 核）实测 **`TrainConfig.resolve('economy').eval_workers == 16`**，**不是 0（串行）**。原文"economy=0（串行）"已作废。**这反而更危险**：不显式传 `--eval-workers` 就会落到 R1 标注的 **16 有 commit 压力风险**档位 ⇒ **长跑仍必须显式传 `--eval-workers 12`**。**✅ 2026-09-17 更正（我 2026-09-14 写错过）**：`runs/...` 全是 **cwd 相对**路径，命令在 `src/clasher_new` 下跑 ⇒
真实位置是 **`src/clasher_new/runs/`**；仓库根那个 `runs/` 是另一份旧目录。
实测 `demo20k`：对手池 `hist ckpts=12 来自 [economy_9k_ft, economy_9j]`、mix 0.1/0.6/0.2/0.1 ⇒ **协议照抄即生效**。

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
| `scripts/forensics_response.py` `_forensics_cycling.py` | 防守响应/接敌取证；cycling 取证（⚠️ 其 `y≥20` 的"幽灵动作"口径**已被推翻**，见【否证 X-15】） |
| `scripts/s1_plan_gate.py`（2026-09-18） | **S1 零成本门禁**：`BeliefPlanner`→PlanToken 贪心执行器，三臂 `random`/`token_strict`/`token_xbow`，行为层三项（全帧圣水≥6 / Xbow 出手率 / 部署前圣水中位）+ 非法动作阴性对照 + `--repro-replays` 仪器自检。**判读**：`docs/s1_gate_2026-09-18.md` |
| `scripts/phi_offline_check.py`（2026-09-18） | **离线逐帧复算 Φ**（`_v_share`/`_active_v`）并与回放 `reward` 逐项对账；冰人白嫖推论的数值验证（证实：ΔΦ = −2.000000 零散布） |
| `scripts/s2_neutrality_probe.py`（2026-09-18） | **行为中立性探针**：在**两棵树**上跑同一段确定性推演，逐 tick 哈希全部实体状态 ⇒ 纯记录通道改动的开/关逐位对账（1200 tick DIGEST 必须逐字相同） |
| `scripts/_patch_battle_root_cast.py`（2026-09-18） | 一次性补丁生成器：给 `battle.py` 打「出牌溯源」通道（AST 定位全部出生点，不靠正则猜括号）；`--apply` 才写文件 |
| `scripts/offline_engagement_trade.py`（2026-09-18） | **离线「局面圣水交换」仪器**（S2）：只读录像，从实体 `id`+`target_id` 重建索敌关系图 → 并查集切局面 → 算 `Trade = ΔΦ + tower_term`。支持 schema 4/5、9 种 `tower_term` 代理（`--tower-mode`）、`component`/`global` 两种 ΔΦ 口径（`--phi-mode`）、`core`/`members` 两种身份判据（`--identity`）、`--sweep`（代理选择扫描）、`--half`（奇偶对半，量批间稳不稳）、`--json`。100 局约 3 秒。**口径与判读**：`docs/s2_instrument_2026-09-18.md` |
| `scripts/selftest_offline_engagement_trade.py`（2026-09-18） | **预注册 §7 不变量 1–6 的测试**（6/6 PASS，合成帧、零引擎依赖、秒级）：反对称 / 桶守恒 / 产物恒 0 费 / 窗口等式 / 局面切分 8 场景 / 跨局重置。不需要经过 `rl/selftest.py`（【R19】） |
| `scripts/forensics_card_usage.py`（2026-09-14） | **回放行为取证**：卡牌使用分布 / 圣水（部署前·后·全帧）/ 部署节奏 / **费用可达性表** / 合法性核验。**只读**，11 个回放文件约 10 秒 |
| `scripts/et_solo100k_readout.py`（2026-09-18） | **`et_solo100k` 两臂判读执行器**：一条命令跑完 §11.13.2 的**四层**——机制层（主，复用 `judge_critic_inertia` 的 within-run 中位±3×MAD 与跨臂「不可分辨」判定；**含层 1b 的 `EVb`／池化 EV**）、行为层（调 `forensics_card_usage`）、门禁层（调 `analyze_online_trade`，含按批次配对 A−B）、项体检（`et` 的 φ/τ/score 分布与 `τ≠0` 占比）。**只编排+复算，不另造口径**（【R17】）；各仪器**原始 stdout 落盘留证**；缺数据**大声失败**不静默填 0。满量程彩排 5 批次/250 局/臂 = **1 min 13 s**（含逐批兑现率，每个评估点单独跑一次离线仪器）。`--judgment <path>` 额外按预注册 **§11.13.8** 的必写条目顺序产出判读文档骨架：**静态条目写全**（不构成判决声明 / 四处更正清单 / 明确未做 / 台账落点）+ 四层读数 + **失败分支输入表**；其中**分支 3 机械判定**（触发/未触发），分支 1·2·4 只列输入并标「须人工判读」。⚠️ 用 Windows python 跑，`--out-dir` 用仓库内相对路径 |
| `scripts/judge_critic_inertia.py`（2026-09-18） | **critic 惰性 / 机制层判据**：`--baseline within` = 本 run **前 100 个诊断点**中位 ± k×MAD（**原始 MAD**，不乘 1.4826），`--baseline prereg` 为旧的固定基线；`--selftest` **4/4 PASS**。2026-09-18 加**可选** `keys=` 形参（默认仍是原六项，**行为逐位不变**），以便 `scripts/health_curve.py` 复用**同一实现**而不是照抄公式（【R17】不得出现两套口径） |
| `scripts/health_curve.py`（2026-09-18） | **训练健康曲线读数**（用户直接要的两个指标）：**价值损失**（用**原始 MSE `vraw=`**，不是 ÷`v_scale²` 的 `value=`）与**策略熵**（掩码后一次决策内各 decoder 步熵之**和**，nat）。二者**本来就逐 update 打在日志里**（solo `[solo step N]` / run `[step N]`，100k ≈ **781 点**），只是 `solo_state.json` 的 32 键里**没有** ⇒ UI 一直看不见。输出 = 分段表 + **局部平台读数**（末窗 vs 紧邻前窗，阈值 k×前窗 MAD）+ §11.13.2 within-run 带（复用 jci 同一实现）+ 可选 `--log-b` 跨臂配对表；`--json` 与 dashboard **同源**；`--selftest` **5/5 PASS**。⚠️ **全部输出是描述性、非预注册判据**；口径与 A_et 实测读数见 [`docs/train_health_metrics_2026-09-18.md`](../train_health_metrics_2026-09-18.md) |
| `scripts/pass_streak_audit.py`（2026-09-18） | **「攒费」样本审计 / 圣水会计（只读录像，~20 s 跑完两臂 155,866 帧）**：把每一帧分成三类——**A 主动不出牌**（`bundle==[]`）/ **B 整包被拒**（`bundle≠[]` 但**圣水没降** ⇒ 非法包被整包拒绝、白掉一帧）/ **C 实际出牌**。分类判据是**会计恒等式**（圣水只可能因回费上升，回费 ≥0 而任何真实出牌花 ≥1 费），**不依赖录像没记录的 `invalid_count`**；决策圣水**精确重建** `pre[i]=elixir0[i-1]`（第 0 帧 = 5.0）。给出：三类占比、`A∩圣水≥6`（**无假设的「能出却选择不出」**）、A 段长度/段内峰值分布、B 段最长与**非法性自证**（重复槽位 / 总费>圣水）、回费标定（实测 **0.3571 圣水/s**）、仪器自检（A 类却花掉圣水应为 0）。⚠️ **`v1` 用「bundle 是否为空」定义不出牌 ⇒ 会把 B 类记成"出牌了"，已作废**（留证 `docs/pass_streak_audit_2026-09-18/audit_v1_SUPERSEDED.log`）。读数与结论：[`docs/elixir_saving_audit_2026-09-18.md`](../elixir_saving_audit_2026-09-18.md) |
| `scripts/probe_explore_randomization.py`（2026-09-18） | **探索侧随机化可行性取证（只读）**：三个模式——`--mode analytic`（只读录像，**解析上界**：按「连续不花钱帧数 `k`」分桶算每 100k 帧的「攒到 6 费」事件数，`k≥17` 桶在 `p=1/6` 下 = **4.5×10⁻¹⁰**）、`--mode placement`（只读录像，**落点集中度**：84/90 格 / 熵 4.5/4.2 bits / top10 73%/78%）、`--mode engine`（真环境蒙卡，臂 `greedy` / `sample` / `uniform` / `mixP` / `holdD`，量「圣水≥6 帧占比 / 最长不花钱段 / 合法格数」）。⚠️ 两遍调用中 `uniform` 臂逐字相同、`sample`/`mix`/`hold` 臂**不同**（O5 同 seed 不可复现）⇒ 只读定性且两遍一致的量。**2026-09-18 追加（第二问）**：新增臂 `bias_<q>_<α>`（每个 decoder 步以 `P(STOP)=q` 停手、剩余概率按 `卡费^α` 加权 = **非等概率提案**）与 `名@β`（给 **STOP logit 加偏置 β** = **on-policy** 版本），并输出 `realized_pass_rate` / `reward_per_game` / `cards` 计数。⚠️ `xbow_plays` 的累加**曾经从未执行、静默恒 0**（已修）⇒ **修前的 JSON 里该字段不可引用**。留证 `docs/probe_explore_randomization/`（含 `bias_{proposal,onpolicy,cards}` 三组） |
**2026-09-18 第三问追加**：`--mode pressure`（离线量**低压窗口**：开启率／窗口数／最长／≥17·≥25·≥34／**窗口内够不够攒到 6 费**）、`--gate-mode`（**门控解析覆盖率**，按录像门状态逐帧乘积）、臂 `gate@β:<门>`（**只在门开时**加 STOP 偏置 ＝ 状态条件偏置）与 `holde@D:<门>`（**带压力中止的 option**）；门 `t`＝码内阈值／`h`＝半场无敌人／`d`＝最近 15 帧没掉塔血／`hd`＝两者。留证 `docs/probe_explore_randomization/`（含 `bias_*` 三组 + `pressure`／`gated_*`／`pressure_gate`） |
**2026-09-18 新工具** `scripts/random_eval_100.py`：**「初始模型 + 全过程随机」评估 runner** —— p0 换成均匀随机 actor、p1 = ckpt 的确定性副本，逐块落盘**生产同格式（schema 5）录像** + `solo_state.json`，供 dashboard（`--solo <目录>`）边跑边看指标与回放；`--games/--block/--out/--seed` 可调。⚠️ p0 无 logprob ⇒ 产物**不能用于训练/BC**。 |
| `scripts/probe_pass_prob.py`（2026-09-18） | **「不出牌」概率（P(STOP)）实测**：`P(不出牌) = bundle 第一个 decoder 步选 STOP`（掩码后、含 plan 软偏置；同时报**同一帧同一隐状态**下 `plan_biases_enabled=False` 的对照）。**★ 必须分层报**：`①开局帧`（每局第一个决策帧，5 圣水空场 ⇒ 与对手无关，`--open-games 30` 只 reset+一次前向，秒级）/ `②有合法选项时`（**真·选择概率**）/ `③全程`（含被迫帧，只是行为频率）；并把被迫帧拆成**圣水不够**（`slot_mask` 裸算对账）vs **「不裸下」门禁禁掉**（`solo_commit_blocked`）。实测 A_et/B_ctrl/fresh_init 三列 + `--json` 逐帧数组；**只读**，约 2 min 40 s。⚠️ 描述性读数、非判据（【R3】【R5】）。读数与禁则：[`docs/pass_prob_2026-09-18.md`](../pass_prob_2026-09-18.md) |
| `src/clasher_new/rl/train_health.py`（2026-09-18） | 上面脚本与 dashboard `/api/health` 的**共用实现**（`parse_line`/`parse_log`/`blocks`/`mad`/`plateau_read`/`health_summary`/`derive_train_log`）——一份口径两处用，避免 UI 与 CLI 各读各的。**训练路径不导入它**（对训练零影响） |
| `scripts/kill_by_cmdline.ps1`（2026-09-18） | **按命令行精准定位/终止 `python.exe`**：`-Marker <子串>` 杀，`-DryRun` 只列；**退出码可当存活判定**（0=有匹配/1=无匹配）。⚠️ 必须**写成文件**再 `-File` 调用（bash 内联传 `$_` 会变成 `\$` ⇒ 过滤器**静默失配、恒返回 0** ⇒「无孤儿」变**假阴性**）；WSL 里也不能直接跑 `powershell`，须经 `cmd.exe /c`；**脚本保持 ASCII-only**（PS 5.1 无 BOM 按 GBK 读 `.ps1` ⇒ 中文注释会解析失败） |
| `scripts/ps_list_python.ps1`（2026-09-18） | **归因用进程快照**（【R1】）：列出 `python.exe` 的 **PID / PPID / 累计 CPU 秒 / 存活秒 / 命令行**；采样两次即可判断某进程是**在算**还是**阻塞**（本机实测抓到过一次「评测 worker 收尾期主进程空转约 5 min」）。**只读**，ASCII-only |
| `scripts/probe_value_ln.py --ladder v3`（2026-09-14） | **前端定位阶梯**：`raw_obs/raw_nongrid/[grid_x]/cnn_pre_ln/grid_ln_out/fused 五块/enc`；`--rollout-only` + `--save-npz` 两阶段、`--exclude` 省内存。**只读**；`_capture_parts` 与原实现逐帧断言逐位一致 |
| `scripts/probe_v4_ln_pair.py` + `judge_probe_v4.py`（2026-09-14） | **真·同张量的投影/归一化分段对账**：离线重算 `enc_fc→ReLU→enc_ln`（与实抓对账）+ 随机投影容量对照；判据全部同 α 配对。**只读、不跑 rollout** |
| `scripts/judge_probe_v3.py`（2026-09-14） | v3 跨 seed 归约：`ρ(fused)` 分布 + 闸门/分支票数（**脚本复算**） |
| `scripts/probe_v3_mono_check.py` + `summarize_probe_v3_mono.py`（2026-09-14） | **超参一致性对账**：逐位验证包含关系 + 每层 α 曲线 + 共同 α 阶梯（闸门 11/12 的执行器） |
| `scripts/probe_reward_composition.py`（2026-09-14） | **逐帧奖励分量分解**（内存包装 `compute_reward`，不改代码）；拆 crown/edw/tower/unit/terminal 五项。**只读** |
| `scripts/value_displacement_scan.py` | 逐窗 `‖ΔW‖/‖W‖` 参数位移指纹（R14 的 M1/M2 判别） |
| `scripts/kill_orphan_workers.ps1`（2026-09-18） | **清【R1】孤儿 spawn worker**（父进程已死的 `--multiprocessing-fork`）：默认 **dry-run 只列**，`-Kill` 才杀，且只杀父进程确认不存在的。实测 14 个孤儿占 **7.7 GB**、把可用提交从 23.2 GB 压到 **7.51 GB**（⇒ 误触发降档）。**长跑前与 `check_commit.py` 配对跑**。⚠️ 该 `.ps1` **必须带 UTF-8 BOM**（PS 5.1 对无 BOM 的 UTF-8 按 GBK 解码，中文会打乱语法）|
| `scripts/check_commit.py`（2026-09-17） | **长跑前宿主提交内存检查**（【R1】的 `wmic` 替代品，因 wmic 已被 Windows 移除）；可用提交 < 12 GB ⇒ 降 `--eval-workers` 档 |
| `scripts/_agents_split.py --check`（2026-09-18） | **AGENTS 拆分的不丢内容校验**：【1】注入尺寸余量、【2】**原文每个非空行逐字出现在新 `AGENTS.md` 或分册之一（缺 0 行才算过）**、【3】相对链接可解析、【4】原节号 1..9 全被分册覆盖。改动 `AGENTS.md` 或 `docs/agents/*` 后跑一次 |
| `scripts/run_selftests.py`（2026-09-14） | 按名跑**子集** selftest（【R19】默认用法；不改 `selftest.py`） |
| `scripts/check_dashboard_js.py`（2026-09-17，**2026-09-18 加 health 正面路径**） | **dashboard 前端回归**：node + DOM 桩跑内嵌 JS 渲染冒烟（经 stdin 管道；Windows 下回退 `wsl.exe node`）；**`--league-run runs/<name>`** 加跑联赛/长跑面板（进度条+大点+胜率曲线）。2026-09-18 新增「训练健康」面板回归：用后端**真实** `build_health_payload`（同一个实现，不造假数据）跑**正面路径**——断言图例含「策略熵」、注里出现**平台读数**与「非预注册判据」，另加「无日志不抛」空分支。⚠️ 这条正面路径**上线当天就抓到一个真 bug**（仓库根推错一层 ⇒ 面板永远"没有日志"，见 `docs/train_health_metrics_2026-09-18.md` §8.2）——只测"没数据不抛"是抓不到的 |

> **📋 全量登记表（2026-09-19 加，Tier 0 · T0-3）**：**本表只登记判读/诊断主用工具**——
> 实测本表引用了 **34** 个 `.py` 名，而 `scripts/` 顶层实有 **91** 个 `.py` ⇒ **57 个未登记**
> （另有 11 个 `scripts/rl/*.py` 包装脚本与 14 个 `.sh/.ps1/.js`）。
> **全量清单**（每个脚本的 docstring 首行 / 是否含 `--selftest` / `assert` 数 / 是否已被本表登记）
> 见 [`scripts_inventory.md`](scripts_inventory.md) —— 由脚本抽取，**不手抄**。
> ⚠️ **两套口径并列**（【R17】）：本段按**顶层 `scripts/*.py`（91 个）**算「未登记 57」；
> `docs/structure_optimization_plan_2026-09-19.md` §5 写的「69 个未登记」按**含 `.sh/.ps1/.js` 与 `scripts/rl/` 的 105 个文件**算 ⇒ 两者**不调和**，引用时须带口径。

> **脚本对手陷阱**：`ScriptedPolicy(mode="heuristic")` 实为 **mask 随机**（P0-3 时代占位）；
> 要"会防守的脚本对手"必须用 `SelfDefenderPolicy`——其反制落点是**世界坐标**，
> 塞进 `ActionBundle` 前必须做**世界→本地网格逆变换**（P1 有镜像；曾直接塞导致部署非法）。

### 2.5 仪表盘（回放/曲线）

```bash
python rl/dashboard.py --state runs/<run> --port 8700   # run 模式（联赛；--state 可给目录）
python rl/dashboard.py --solo  runs/<run> --port 8700   # solo
```
> **2026-09-18 起（长跑对接）**：`--state` 也接受**目录**（自动找 `league_state.json`）；同目录 `run_state.json`+`config.json` ⇒ **进度条**（步数/评估点 n/N/粗估剩余；`state_age_s>15min` 变红=疑似卡死）；两级评估点按 `kind` 画竖虚线；新增**指标多选器**（默认 Elo / 可切**逐对手胜率**，由 `history`+`round_stats.games` 切分复原）。详见 [`docs/dashboard_long1m_2026-09-18.md`](docs/dashboard_long1m_2026-09-18.md)
> `--solo` 可传**目录**（自动找 `solo_state.json`）。**2026-09-17 起**：solo 曲线改为**指标多选器**
> （默认行为指标，胜率降级进「⚠ 自引用（禁读）」组）；卡牌统计新增「**按卡组**」矩阵（行=卡牌、列=卡组）。
> **2026-09-18 新增「训练健康」组**（策略熵 · 价值损失）：数据源**不是** `solo_state.json`，而是
> **训练日志**（`/api/health`；日志路径由 `--train-log` 显式给，或从 `--solo` 目录名推
> `<repo>/docs/train_<run>.log`；推不到就显示"不可用"而**不会**拿别的 run 顶上）。
> 注里直接给出**局部平台读数**（末窗 vs 前窗、diff、3×MAD、snr）并标注「非预注册判据」。
> 口径与实测读数：[`docs/train_health_metrics_2026-09-18.md`](../train_health_metrics_2026-09-18.md)。
> ⚠️ 早期点数少时该读数**无信息量**（窗只有 5% 点数、MAD 很大）⇒ 看读数前先看 snr 与窗口宽度。
> **⚠️ 2026-09-17 环境漂移（实测）：8090 已不可用**——`netsh int ipv4 show excludedportrange protocol=tcp`
> 显示 **8013–8112 被系统保留**（Hyper-V/WinNAT）⇒ 绑定抛 `PermissionError: [WinError 10013]`，
> 而 `netstat -ano | findstr :8090` **查不到任何监听**（所以别误判成"端口被占"）。**改用 `--port 8700`**（实测 200 OK）。
> Windows 侧 python 绑定 **Windows loopback**：WSL 里 `curl 127.0.0.1:<port>` 一律失败（用它验证"服务挂了"会误判）；
> **验证要用 Windows 侧解释器**：`.venv/Scripts/python.exe -c "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:8700/api/solo').status)"`；
> 用户浏览器直接访问 `http://127.0.0.1:8700` 正常。

### 2.6 自检

```bash
# 默认：只跑与改动相关的测试（【红线 R19】）
cd src/clasher_new && PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe \
    ../../scripts/run_selftests.py --list
cd src/clasher_new && PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe \
    ../../scripts/run_selftests.py test_precise_threat
# 全量（仅 R19 列的例外情形才跑；2026-09-14 最后一次全量 = 97 项 ALL PASSED）
cd src/clasher_new && PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe rl/selftest.py
```

---
