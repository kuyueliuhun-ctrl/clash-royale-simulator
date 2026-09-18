# -*- coding: utf-8 -*-
"""搜索/推演成本实测（只读）：确认"搜索+推演"路线的成本账是否仍成立。

回答「用搜索+推演训练打分器」值不值：先量**引擎推演速度**，再算每个决策帧的搜索成本，
最后算"在一局/一个 20k run 上的墙钟倍数"。
用法（必须在 src/clasher_new 下运行）：
    cd src/clasher_new && PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe \
        ../../scripts/probe_search_cost.py
"""
from __future__ import annotations

import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src", "clasher_new"))
from rl.io_bootstrap import force_utf8_stdout  # noqa: E402  (T1-1: UTF-8 兜底单一来源)
force_utf8_stdout()

from rl.env_wrapper import RLEnv
from rl.action_bundle import ActionBundle

TICKS_PER_FRAME = 30          # decision_frames（env_wrapper.py:301）
DT = 1.0 / 60

env = RLEnv(opponent=None, seed=0)
env.reset(seed=0)

# (1) 纯引擎 battle.step 速度（搜索的推演成本单位）
n_ticks = 12000
t0 = time.perf_counter()
for _ in range(n_ticks):
    if env.battle.game_over:
        env.reset(seed=0)
    env.battle.step(DT)
dt_engine = time.perf_counter() - t0
rate_ticks = n_ticks / dt_engine
print(f"[1] 纯引擎 battle.step：{n_ticks} ticks / {dt_engine:.2f}s = "
      f"{rate_ticks:,.0f} battle-steps/s")
print(f"    ⇒ 1 决策帧（{TICKS_PER_FRAME} ticks）≈ {TICKS_PER_FRAME/rate_ticks*1000:.1f} ms"
      f"；一局 217~360 帧 ≈ {217*TICKS_PER_FRAME/rate_ticks:.1f}~"
      f"{360*TICKS_PER_FRAME/rate_ticks:.1f} s 纯推演")

# (2) 完整 RLEnv.step（含观测构造 + 对手 + 30 ticks）：训练侧的真实单位成本
env.reset(seed=0)
n_fr = 200
t0 = time.perf_counter()
for _ in range(n_fr):
    if env.battle.game_over:
        env.reset(seed=0)
    env.step(ActionBundle.noop())
dt_env = time.perf_counter() - t0
print(f"[2] RLEnv.step（无网络）：{n_fr} 帧 / {dt_env:.2f}s = {n_fr/dt_env:.1f} 帧/s"
      f" ⇒ {dt_env/n_fr*1000:.1f} ms/帧")

# (3) 搜索成本（按 mcts_design.md 的 v1 结构：一次模拟 ≈ 深度3×30 + 叶 8s×60 = 720 ticks）
SIM_TICKS = 3 * TICKS_PER_FRAME + 8 * 60
print(f"\n[3] 单次模拟 ≈ 720 battle-steps（mcts_design.md §预算：深度3×30 + 叶 8s×60）"
      f" = {720/rate_ticks*1000:.1f} ms")
    # 参照系用**训练侧**每帧成本（含网络）：AGENTS §3 实测纯训练 26 步/s ⇒ 38.5 ms/帧
TRAIN_MS_PER_FRAME = 1000.0 / 26.0
base_ms = TRAIN_MS_PER_FRAME
print(f"    训练侧参照：{TRAIN_MS_PER_FRAME:.1f} ms/决策帧（纯训练 26 步/s，含网络）")
print(f"    （上面 [2] 的 {dt_env / n_fr * 1000:.1f} ms 是「无网络」的引擎+观测成本，仅作分解）")
print(f"\n    搜索档位（每决策帧 N 次模拟）       秒/帧      相对训练（{base_ms:.1f} ms/帧）"
      f"   一局 217 帧   20k 帧 run")
for nsim in (8, 24, 50, 120, 400):
    s_per_frame = nsim * SIM_TICKS / rate_ticks
    print(f"    N={nsim:<4}                              {s_per_frame:6.2f}    "
          f"{s_per_frame*1000/base_ms:6.1f}×          {s_per_frame*217/60:5.1f} min"
          f"      {s_per_frame*20000/3600:6.1f} h")
print("\n    注：20k 帧 run 指「搜索跑满全部训练帧」；若只在部分帧搜索，按比例折算。")
print("        R11 记的「≈120×」对应 N≈120；mcts_design.md 的 v1 默认 N=24 ⇒ ≈24×。")
