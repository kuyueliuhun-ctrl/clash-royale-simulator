# R1 事件留证：Layer 1（critic 惰性检验）在 eval@0 崩溃（2026-09-13）

> **状态：已暂停，等人类决定。未自动降级、未改代码绕开、未下调配置。**
> 依据【红线 R1】。本文件只记**观测到的事实**；未定的一律写"未定"。

## 1. 事件

- 命令：`run_league.py --mode solo --config economy --config-name critic_inert_probe_20k
  --fresh --total-steps 20000 --steps-per-eval 20000 --anchor-every 2500 --n-eval-games 40
  --eval-workers 12 --device cuda --value-norm running --adv-norm scale --diagnose-every 1
  --adv-inert-probe --hist-seed-dir runs/economy_9k_ft --hist-seed-dir runs/economy_9j
  --ppo-epochs 4 --ppo-minibatch 32 --ppo-shuffle`
- 结果：**exit 1**，日志 `docs/train_critic_inert_probe_20k.log`（209 行）。
- 进度：**0 个训练步**（`grep -c "\[solo step"` = 0）、**0 个探针点**（`grep -c advinert` = 0）、
  run 目录只有 `config.json` + 空 `replays/` ⇒ **无任何可用数据**。

## 2. 崩溃链（逐字取自日志）

```
[eval] 并行评估 worker 失败，降级串行: RuntimeError('[enforce fail at alloc_cpu.cpp:117]
        data. DefaultCPUAllocator: not enough memory: you tried to allocate 539136 bytes.')
  File ".../rl/train_solo.py", line 837, in _eval_worker_main
    main = FollowerPolicy(...)          # ← worker 内建策略时就崩了
  File ".../rl/follower.py", line 202, in __init__
    cnn_out = self.cnn(dummy).shape[1]
  ... torch/nn/modules/conv.py → F.conv2d → DefaultCPUAllocator 失败
→（代码按设计降级串行）
  File ".../rl/train_solo.py", line 1484, in run_solo
    eval_and_write(0)                   # ← 崩溃点：**起始评估**，训练环还没进
  File ".../rl/train_solo.py", line 752, in eval_solo → main.act(...)
  File ".../rl/follower.py", line 275, in _encode_parts
    ... .to(self.device) / 12.0
torch.AcceleratorError: CUDA error: unknown error   (cudaErrorUnknown)
```

**两条独立故障**：① 并行评估 worker **宿主内存分配失败**（539 KB 都拿不到）⇒ 代码降级串行；
② 降级后的串行评估在做 H2D 拷贝时 `cudaErrorUnknown`。

## 3. 环境实测（21:2x–21:3x）

| 量 | 实测 | 备注 |
|---|---|---|
| 空闲物理内存 | **11.14 GB** / 总 32.83 GB | |
| 空闲虚拟内存（提交） | **10.68 GB** / 上限 51.67 GB | ⇒ **提交已用 ≈ 41 GB** |
| R1 历史标定 | "提交上限 47.3 GB / 空闲 **20.6 GB**"（当时据此把 12 定为安全档） | ⇒ 今日余量 ≈ 当年一半 |
| GPU | RTX 4070 Laptop 8 GB：已用 1.33~1.99 GB、外部占用者常驻（util 23~43%） | **显存不是瓶颈** |
| 常驻孤儿进程 | 3 个 `multiprocessing.spawn` worker（父 PID 50592/62264 **已不存在**）：94 MB + 558 MB + 560 MB ≈ **1.2 GB** | 是否本 run 遗留：**未定** |
| dashboard | PID 66568 在 127.0.0.1:8090 LISTENING（16 MB） | 我此前启动的，仍在服务 |

## 4. 探针是否在崩溃路径上（要排除的第一件事）

**不在。** 依据：
- 崩溃发生在 `eval_and_write(0)`，而 `ppo.update()` 一次都没执行（0 个 `[solo step]` 行）；
- 探针代码全部位于 `PPOTrainer.update()` 内（`adv_alt` 分支）与 `run_solo` 的探针累积处，
  两者都在训练环里、在 `eval@0` **之后**；
- 同开关（`--diagnose-every 1 --adv-inert-probe`）的两次 smoke（700/800 步）**正常完成**，
  且这次崩溃的 traceback 里没有任何 `adv_alt` / `adv_inert` 帧。

## 5. 未定（不得当结论引用）

- `cudaErrorUnknown` 的**直接成因**：未定（未做单变量复现；不排除与 ① 的宿主内存耗尽同源，也未证实）。
- **41 GB 提交被谁占用**：未定（本次未逐个进程统计；GPU 上无其他算进程，压力在宿主侧）。
- 3 个孤儿 worker 来自哪一次运行（本次 run 还是之前的 `rl/selftest.py` 并行测试）：未定。
- 12 worker 在当前余量下是否仍安全：**需实测**（当年标定 20.6 GB 空闲，今日 10.68 GB）。

## 6. 处置（待人类拍板，R1 禁止自动做）

1. 清理 3 个孤儿 worker（~1.2 GB）；
2. **按原配置原样重跑**（不降 `eval-workers`）—— 符合 R1；
3. 若再炸同指纹，则需人类决定：先腾宿主内存，或明确批准把 `eval-workers` 降到 8
   （降配置属 R1 禁止我自行执行的范围；注意 `eval_workers` 只影响速度、不影响锚点读数语义）。

**不采用**任何"改代码绕开"的做法。
