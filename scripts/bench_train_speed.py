#!/usr/bin/env python
"""训练速度基准（2026-09-12）：跑一个短 solo run，实时给日志行打时间戳，
输出可对比的速度统计。

为什么做：E2（rand_anchor 对手池）落地后需要确认训练速度无回归，并建立可对比的
吞吐基准（AGENTS 里只有 2026-09-06 的引擎 18,931 battle-steps/s 与 ~14 步/s 旧值，
协议已变）。本脚本以受控短 run 出数字，避免手工数日志。

口径（与历史判读一致）：
- 混合吞吐 = total_steps / 训练循环耗时（含评估；fix_gru_ln_norm_20k 是 20000/2709s
  ≈ 7.4 步/s）。"训练循环耗时"来自 train_solo 日志末尾（run_league 主循环开始到结束）。
- eval 成本：启动耗时分解里的 "eval@0 N局=Xs"（并行 worker）。
- 各评估点墙钟间隔：脚本自己给每行日志打时间戳（日志本身无时间戳）。

用法（在 src/clasher_new 下，gamedata.json 相对 cwd）：
  PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe \
      ../../scripts/bench_train_speed.py --steps 3000 --per-eval 1000 --eval-games 2 \
      --eval-workers 12 --device cuda --config-name bench_speed
（--eval-games 2 时评估开销≈0，混合吞吐≈纯训练吞吐；要含评估成本用 --eval-games 40。）
"""
import argparse
import os
import re
import subprocess
import sys
import threading
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKDIR = os.path.join(REPO_ROOT, "src", "clasher_new")
# Windows .venv（torch 2.14 + CUDA）；WSL2 里也是这个解释器
PY = os.path.join(REPO_ROOT, ".venv", "Scripts", "python.exe")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=3000, help="总训练步数（决策帧）")
    ap.add_argument("--per-eval", type=int, default=1000, help="评估间隔步数")
    ap.add_argument("--eval-games", type=int, default=2, help="每评估点局数（2≈纯训练）")
    ap.add_argument("--eval-workers", type=int, default=12, help="并行评估 worker 数")
    ap.add_argument("--device", default="cuda", help="cuda / cpu")
    ap.add_argument("--config", default="economy", help="配置预设（economy/fast）")
    ap.add_argument("--config-name", default="bench_speed", help="run 目录名（runs/<name>）")
    ap.add_argument("--hist-dirs", nargs="*", default=[],
                    help="对手池 hist 补种目录（runs/economy_9k_ft runs/economy_9j）")
    args = ap.parse_args()

    cmd = [PY, "rl/run_league.py", "--mode", "solo", "--config", args.config,
           "--config-name", args.config_name, "--fresh",
           "--total-steps", str(args.steps), "--steps-per-eval", str(args.per_eval),
           "--n-eval-games", str(args.eval_games), "--eval-workers", str(args.eval_workers),
           "--device", args.device, "--value-norm", "running", "--adv-norm", "scale",
           "--diagnose-every", "10"]
    for d in args.hist_dirs:
        cmd += ["--hist-seed-dir", d]
    if not os.path.exists(PY):
        print(f"[bench] 找不到解释器 {PY}（应在仓库根 .venv/Scripts/）", file=sys.stderr)
        return 1
    if not os.path.isdir(WORKDIR):
        print(f"[bench] 找不到工作目录 {WORKDIR}", file=sys.stderr)
        return 1

    print(f"[bench] cmd: {' '.join(cmd)}")
    print(f"[bench] cwd: {WORKDIR}")
    t0 = time.monotonic()
    stamps = []   # (elapsed, line)
    proc = subprocess.Popen(cmd, cwd=WORKDIR, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace")
    lock = threading.Lock()

    def _reader():
        for line in proc.stdout:
            with lock:
                stamps.append((time.monotonic() - t0, line.rstrip("\n")))
    th = threading.Thread(target=_reader, daemon=True)
    th.start()
    rc = proc.wait()
    th.join()
    wall = time.monotonic() - t0
    with lock:
        lines = list(stamps)

    # —— 解析 ——
    evals = []            # (step, elapsed)
    startup = None        # (elapsed, line)
    loop_time = None
    degraded = 0
    for el, line in lines:
        m = re.search(r"\[solo\] eval@(\d+):", line)
        if m:
            evals.append((int(m.group(1)), el))
        if "启动耗时分解" in line:
            startup = (el, line)
        m2 = re.search(r"训练循环耗时 ([\d.]+)s", line)
        if m2:
            loop_time = float(m2.group(1))
        if "降级" in line or "worker 失败" in line or "降级串行" in line:
            degraded += 1

    if loop_time is None:
        loop_time = wall   # run 异常结束时的兜底（含一切）

    print("\n" + "=" * 70)
    print(f"[bench] run '{args.config_name}' 退出码={rc}  进程墙钟={wall:.1f}s")
    print(f"[bench] 配置: steps={args.steps} per-eval={args.per_eval} "
          f"eval-games={args.eval_games} eval-workers={args.eval_workers} device={args.device}")
    if degraded:
        print(f"[bench] ⚠️ 检测到 {degraded} 次评估降级/worker 失败（数字不可比，按 "
              f"AGENTS 全局操作约定：先归因外部再判读）")
    if startup:
        print(f"[bench] 启动分解 @{startup[0]:6.1f}s: {startup[1].strip()}")
    print(f"[bench] 训练循环耗时（含评估）= {loop_time:.1f}s")
    print(f"[bench] 混合吞吐 = {args.steps / max(loop_time, 0.001):.2f} 步/s "
          f"（= {args.steps} 步 / {loop_time:.0f}s，含评估，与历史判读同口径）")
    if evals:
        print("[bench] 评估点墙钟（相对启动）:")
        for step, el in evals:
            print(f"    eval@{step:<6} {el:7.1f}s")
        if len(evals) >= 2:
            gaps = [evals[i + 1][1] - evals[i][1] for i in range(len(evals) - 1)]
            g = max(1, int(args.per_eval))
            print(f"[bench] 评估间隔（= {g} 步训练 + 一次评估）: "
                  + ", ".join(f"{x:.1f}s" for x in gaps))
    return rc


if __name__ == "__main__":
    sys.exit(main())
