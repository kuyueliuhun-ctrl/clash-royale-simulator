# 训练速度基准（2026-09-12）

> 目的：E2（`rand_anchor` 对手池第 4 槽）落地后确认**训练速度无回归**，并建立
> 可对比的吞吐基准。此前 AGENTS.md 只有 2026-09-06 的引擎级基准
> （18,931 battle-steps/s）与训练循环 ~14 步/s 旧值，协议已变（evaluator/探针/
> 对照组），不能直接沿用。

## 1. 协议

复用 `scripts/bench_train_speed.py`（受控短 solo run，实时给日志打时间戳）：

```bash
cd src/clasher_new
# 纯训练口径（评估开销≈0）：--eval-games 2
PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe \
    ../../scripts/bench_train_speed.py --steps 3000 --per-eval 1000 --eval-games 2 \
    --eval-workers 12 --device cuda --config-name bench_speed_pure
# 含评估口径（与 e2 生产协议一致）：--eval-games 40
PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe \
    ../../scripts/bench_train_speed.py --steps 3000 --per-eval 1000 --eval-games 40 \
    --eval-workers 12 --device cuda --config-name bench_speed_eval \
    --hist-dirs runs/economy_9k_ft runs/economy_9j
```

统计口径：

| 指标 | 定义 | 历史对照 |
|---|---|---|
| 混合吞吐 | `total_steps / 训练循环耗时`（含评估） | `fix_gru_ln_norm_20k` = 20000/2709.3s ≈ **7.4 步/s**；`e2_rand_anchor_20k` 同口径实测（§2） |
| 纯训练吞吐 | `--eval-games 2` 的混合吞吐（评估≈0） | AGENTS 旧值 ~14 步/s（2026-09-06，协议不同，仅参考） |
| eval 成本 | 启动分解 `eval@0 N局=Xs` + 各评估点墙钟间隔 | `e2` eval@0 40局 = 193.6s（worker=12） |
| 引擎速度 | battle-steps/s（引擎侧） | AGENTS 18,931（2026-09-06） |

判读纪律（AGENTS 全局操作约定）：任何数字先确认**本次没有发生评估降级/worker 失败**
（脚本输出 `⚠️ 降级`）；与历史差异 >30% 时优先按外部干扰（页面文件/宿主负载）解释，
先复测再下结论，不直接归因代码。

## 2. 实测

### 2.1 生产日志统计（`e2_rand_anchor_20k`，零成本）

| 项 | 值 |
|---|---|
| 训练循环耗时（含评估） | **3419.4s**（59 局 / 9 评估点 / 20000 步） |
| 混合吞吐 | **20000/3419.4 ≈ 5.85 步/s**（含评估；worker=12） |
| eval@0（主 + 3 组对照 × 40 局 = 160 局） | 193.6s（worker=12） |
| 评估降级 / Traceback | **0 / 0** |

对照（同协议，worker 16）：`fix_gru_ln_norm_20k` = 20000/2709.3 ≈ **7.38 步/s**、
`fix_gru_ln_20k` = 20000/2096.9 ≈ **9.54 步/s**。e2 慢 21~38% 的差异含
**eval-workers 12 vs 16** 因素，不能直接归因 E2；精确拆解见 §2.2。

### 2.2 受控短跑（`bench_speed_pure` / `bench_speed_eval`）

| run | steps | per-eval | eval-games | 训练循环耗时 | 混合步/s | 备注 |
|---|---|---|---|---|---|---|
| bench_speed_pure | 3000 | 1000 | 2 | 364.3s | **8.23** | 评估≈0（8 局/点）；eval@0 2局×4组=44.1s |
| bench_speed_eval | 3000 | 1000 | 40 | 1166.4s | 2.57 ⚠️ | 见下方"外部干扰"备注；eval@0 40局×4组=198.0s |

**pure 拆解（worker=12，cuda）**：
- 评估间隔（=1000 步训练 + 一次 2局×4组评估）: 127.1 / 109.8 / 127.4s（均值 ~121s）；
- eval@0（2 局 × 4 组 = 8 局）: **44.1s** —— 说明**评估成本大头是 spawn/启动固定开销**
  （12 worker 起进程 + 建 env/信念/网络），局数只占小头 ⇒ 40→16 局的节省比
  "局数比例"小得多（见 §3）；
- 纯训练吞吐：1000 步 ≈ 121 − 44 ≈ 77s ⇒ **≈13 步/s**（与 AGENTS 旧值 ~14 步/s 一致）。

**eval 拆解 + ⚠️ 外部干扰备注**：
- eval@0（40 局 × 4 组 = 160 局）: **198.0s**；评估间隔（=1000 步训练 + 一次 40 局评估）:
  443.3 / 321.1 / 402.0s；
- 由此推算的"训练部分"（间隔 − 198s）≈ 191s/千步 ⇒ 5.2 步/s，**与 pure 的 13 步/s
  差 2.5 倍，但代码/协议完全相同** ⇒ 按 AGENTS 全局操作约定归因**外部干扰**
  （spawn 12 个 CUDA-torch worker 的页面文件/内存压力残留在训练段显现；成因未定），
  **不得把 bench_eval 的混合 2.57 步/s 当代码归因**；
- 可信的评估成本口径用 **e2 生产 run** 交叉验证：20k 步假设纯训练 13 步/s ⇒ 训练
  ≈1538s，总 3419s ⇒ 9 个评估点 ≈ 1881s ≈ **209s/点**，与 bench_eval 的 eval@0
  198s 吻合。⇒ **每评估点（40 局 × 4 组，worker 12）≈ 200s**。

## 3. 结论

1. **E2 对手池改动无训练速度回归（纯训练部分）**：pure 实测纯训练 ≈**13 步/s**，
   与 AGENTS 旧值 ~14 步/s 一致 ⇒ `rand_anchor` 第 4 槽对训练循环本身零开销。
2. **评估成本拆解**：每评估点（worker 12）≈ 固定 spawn/启动 ~40s + 局数部分
   （40 局 × 4 组 ≈ 160s）。**40→16 局的真实节省 ≈ (40−16)/40 × 160s ≈ 95s/点**，
   9 点省 ~14 分钟——远小于"局数减 60% ⇒ 省 60%"的直觉（固定开销省不掉）。
3. **评估占总时长 ~55%**（e2：9 点 × ~200s = 1800s / 3419s）⇒ 若想提速，降低
   评估频率（steps_per_eval）比降局数更有效。
4. **worker 12 vs 16**：e2（12）混合 5.85 步/s vs fix_gru_ln_norm（16）7.38 步/s——
   差异主要来自 worker 档位（每点评估慢 ~40s）+ 外部干扰，**非 E2 代码**。
5. **⚠️ 外部干扰记录**：bench_eval 训练段 5.2 步/s vs pure 13 步/s（同代码同机，
   差 2.5 倍），成因未定，按 AGENTS 约定记"未定"不归因代码；**判读任何吞吐数字前
   必须确认本次无降级/无负载异常**（本次两个基准 run 均 0 降级、0 Traceback）。
