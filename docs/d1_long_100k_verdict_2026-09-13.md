# D1 100k 长跑判读（2026-09-13）

> **状态：进行中** —— 训练已启动并按 C 评估节奏落盘；本文件的可确定部分（协议、披露、
> 脚本复算的基线、判据）先写好，判据结果表待 run 跑完后由
> `scripts/summarize_solo_run.py` + `scripts/judge_anchor_blocks.py` 的输出填齐。
> **⚠️ 在状态改为"已完成"之前，任何读数都不得作为结论引用。**

## 0. 结论

**待填（跑完后写；只允许写脚本产出支持得住的句子）**

## 1. 协议（实际执行的命令）

```bash
cd src/clasher_new
PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe rl/run_league.py \
  --mode solo --config economy --config-name d1_long_100k --fresh \
  --total-steps 100000 --steps-per-eval 10000 --anchor-every 2500 \
  --n-eval-games 40 --eval-workers 12 --device cuda \
  --value-norm running --adv-norm scale --diagnose-every 10 \
  --hist-seed-dir runs/economy_9k_ft --hist-seed-dir runs/economy_9j \
  --ppo-epochs 4 --ppo-minibatch 32 --ppo-shuffle
```

- 代码 = `main`（`616a71b`，D1）+ 分支 `no-human-watch-E` 上的**纯测量**改动
  （`--anchor-every`，默认 0 ⇒ 训练语义不变；见 `docs/eval_cadence_c_2026-09-13.md`）；
- 唯一训练变量 = **训练量 20k → 100k**；
- 评估节奏 = C 方案（实测打印）：
  `全点每 10000 步（11 个，含 eval@0）/ 轻量锚点每 2500 步（30 个，只跑 baseline_rand + 落盘）`；
- 世界时间偏离预注册：**跑前已披露并获用户确认**——外部 CUDA 进程常驻（启动前即占
  2039 MiB / 31% util），训练吞吐实测 **11~14.5 步/s**（同配置 20k 基线 25.9 步/s），
  ETA 由 2.15 h 拉到 ≈3.5 h。**吞吐不影响判据**（判据只与步数有关），故按用户决定继续；
- 0 降级 / 0 `WinError 1455` / 0 `Traceback`（跑完复核）。

## 2. 预注册判据（照抄 `docs/d1_long_100k_prereg_2026-09-13.md`，跑前写死）

- **P1（主判据，C1-块口径）**：41 个锚点按每 9 点一块切 5 块（9,8,8,8,8），
  每块取最差点 `worst_b`；`mean(worst_b) ≥ 0.10` **且** `min(worst_b) ≥ 0.05` → 判
  "防崩在 4× 训练量下成立"；
- 判读规则：P1 通过 **且** 与无变化组区间不重叠 → "成立"；P1 通过但重叠 → "部分成立"；
  P1 不通过 → "失效"；
- **次判据（描述性，不判决）**：块中位数 `median_末块 − median_首块`，≥+0.10 记"提示上移"、
  ≤−0.10 记"提示下移"、其间"不可判"；**不得写进结论句**。

## 3. 对照基线（**由脚本从磁盘复算**，非手抄）

命令：`python scripts/judge_anchor_blocks.py --groups`

| 组 | 单跑 min(worst) | 区间 | 组均值 |
|---|---|---|---|
| D1 20k 三跑 | 0.125 / 0.200 / 0.250 | [0.125, 0.250] | 0.192 |
| 无变化三跑（fprime / fprime_ev / gfix） | 0.000 / 0.000 / 0.025 | [0.000, 0.025] | 0.008 |

**区间是否重叠：否（判据有判别力）** —— 与 `docs/d1_league_20k_verdict_2026-09-13.md`
的记录逐位一致（该文档的基线是手抄的，本次改为脚本复算后仍一致，属交叉验证通过）。

## 4. 结果（待填）

<!-- 待填：scripts/summarize_solo_run.py --log docs/train_d1_long_100k.log --run d1_long_100k --expect-total 100000 的输出 -->

- 锚点序列（41 点，含 5 块 worst/median）：
- P1 判定：
- 次判据（描述性）：
- 诊断读数（EV / h_std / n_abs / vstd-rstd / gates / 对手池 kind_counts）：

## 5. 披露与修订记录

1. **吞吐**：见 §1（外部 CUDA 进程；不改配置、不降 worker，按 AGENTS 约定留证）。
2. **补种 ckpt 会被挤出**（跑前已披露）：`_HIST_POOL_MAX=12` 且本目录优先 ⇒ 20k 时
   池=本目录 9+外部 3，100k 时本目录快照 ≥12 ⇒ 池子全为本 run 自身快照；
   **待填：本 run 的实际 `对手池刷新` 曲线（本目录 N）以确认发生时刻**。
3. **n=1/arm**：与历史基线非同批（同 seed 不可复现），P1 只回答"是否落在防崩区间"。
4. **判读工具的两个自纠**（写入 `scripts/`，避免第三次手抄错）：
   `judge_anchor_blocks.py` 首版按 `step//20000` 切块 → 20k run 的 `mean(worst)` 算成
   **0.400**（文档值 0.192）；`summarize_solo_run.py` 首版对 `kind_counts` 用 `json.loads`
   而日志打的是 Python dict repr（单引号）→ 崩。均已修，并在 §3 用历史数据交叉验证。
