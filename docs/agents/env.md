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
| `scripts/check_dashboard_js.py`（2026-09-17） | **dashboard 前端回归**：node + DOM 桩跑内嵌 JS 渲染冒烟（经 stdin 管道；Windows 下回退 `wsl.exe node`）；**`--league-run runs/<name>`** 加跑联赛/长跑面板（进度条+大点+胜率曲线） |

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
