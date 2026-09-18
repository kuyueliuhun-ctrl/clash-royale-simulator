# -*- coding: utf-8 -*-
"""奖励语义单元验证（只读）：逐帧分数是「对比上一帧的差分」还是「当前帧的绝对局面」？

直接调用**真实的** `rl.env_wrapper.compute_reward`（纯函数，不需要 rollout），逐项验证：
  T1 非归一化塔伤：同 Δ、不同绝对塔血 → 分数是否相同（= 是否含绝对水平）
  T2 归一化 + per-tower 差异化定价：同 Δ、残血塔 → 分数是否不同（= 系数是否依赖绝对局面）
  T3 相位：同 Δ、t<120 vs t>=120 → 系数是否不同（_phase_weights，依赖绝对时间）
  T4 资源账 edw：同端点、不同路径 → 累计分是否相同（telescoping）
  T5 elixir_bonus：唯一按**绝对存量**给分的项（默认 0.0 = 关闭）
用法（**必须在 src/clasher_new 下运行**：`card_utils.py` 按相对路径读 `gamedata.json`）：
    cd src/clasher_new && PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe \
        ../../scripts/probe_reward_semantics.py
实测（2026-09-14）：T1 同 Δ 同分（0.500000/0.500000）；T2 残血溢价 2.398×；
T3 相位 ×2（0.0010→0.0020）；T4 同端点三条路径同为 +5.0000；T5 唯一绝对存量项（默认关）；
T6 事件项 ±10 / −0.05×次数。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "src", "clasher_new"))
from rl.io_bootstrap import force_utf8_stdout  # noqa: E402  (T1-1: UTF-8 兜底单一来源)
force_utf8_stdout()

from rl.env_wrapper import compute_reward, _phase_weights, _DEFAULT_REWARD, PHASE_SWITCH_S

ZERO = {"crown_weight": 0.0, "crown_lose_weight": 0.0, "tower_dmg_opp": 0.0,
        "tower_dmg_self": 0.0, "tower_dmg_late": 0.0, "tower_dmg_self_late": 0.0,
        "win_bonus": 0.0, "lose_penalty": 0.0, "draw_penalty": 0.0,
        "invalid_penalty": 0.0, "elixir_bonus": 0.0, "elixir_diff_weight": 0.0,
        "elixir_diff_late": 0.0, "unit_dmg_k": 0.0, "normalize_tower_dmg": False}


def frame(rw, **kw):
    base = dict(blue_hps_old=0.0, red_hps_old=0.0, blue_hps_new=0.0, red_hps_new=0.0,
                blue_left_old=3, red_left_old=3, blue_left_new=3, red_left_new=3,
                my_elixir_before=0.0, opp_elixir_before=0.0,
                my_elixir_after=0.0, opp_elixir_after=0.0,
                winner=None, invalid_count=0)
    base.update(kw)
    return compute_reward(rw, **base)


print("=== T1 非归一化塔伤：同 Δ、不同绝对塔血 ===")
rw = dict(ZERO, tower_dmg_opp=0.001)
a = frame(rw, red_hps_old=10000.0, red_hps_new=9500.0)   # 满血掉 500
b = frame(rw, red_hps_old=600.0, red_hps_new=100.0)      # 残血掉 500
print(f"  满血区 掉 500 → {a:+.6f}   残血区 掉 500 → {b:+.6f}   "
      f"{'完全相同 ⇒ 严格差分，不含绝对水平' if abs(a-b)<1e-12 else '不同'}")

print("\n=== T2 归一化 + per-tower 差异化定价：同 Δ、残血塔 ===")
rw = dict(ZERO, tower_dmg_opp=0.001, normalize_tower_dmg=True,
          tower_premium_k=2.0, king_gate=0.05)
MX = 3052.0
T2 = lambda old, new: frame(rw,
                            red_towers_old=[4824.0, old, 3052.0],
                            red_towers_new=[4824.0, new, 3052.0],
                            red_towers_max=[4824.0, MX, MX],
                            red_hps_old=4824.0 + old + MX, red_hps_new=4824.0 + new + MX)
fa, fb = T2(MX, MX - 500.0), T2(500.0, 0.0)
print(f"  满血塔 掉 500 (3052→2552) → {fa:+.6f}")
print(f"  残血塔 掉 500 ( 500→   0) → {fb:+.6f}   倍数 {fb/fa:.3f}×")
print(f"  ⇒ 同一 500 塔血，落在残血塔上值 {fb/fa:.2f} 倍 ⇒ **系数**依赖绝对局面（旧血线）")

print("\n=== T3 相位：同 Δ、绝对时间不同（_phase_weights）===")
rw = dict(ZERO, tower_dmg_opp=0.001, tower_dmg_self=0.0012,
          tower_dmg_late=0.002, tower_dmg_self_late=0.0022)
for t in (100.0, 130.0):
    to, ts, edw = _phase_weights(rw, t)
    print(f"  battle.time={t:>6.1f}s → tower_opp={to:.4f} tower_self={ts:.4f} edw={edw:.2f}"
          f"   同 500 塔伤分 = {to*500:+.4f}")
print(f"  ⇒ 切换点 PHASE_SWITCH_S={PHASE_SWITCH_S}（绝对时间，非差分）")

print("\n=== T4 资源账 edw：同端点、不同路径（telescoping）===")
rw = dict(ZERO, elixir_diff_weight=0.5)


def seq(points):
    """points: 圣水路径；返回该路径逐帧 edw 项之和。"""
    tot = 0.0
    for i in range(1, len(points)):
        tot += frame(rw, my_elixir_before=points[i - 1], my_elixir_after=points[i])
    return tot


p1 = [0.0, 3.0, 6.0, 9.0, 10.0]          # 一路上涨
p2 = [0.0, 10.0, 10.0, 10.0, 10.0]       # 一步到顶
p3 = [0.0, 5.0, 2.0, 7.0, 10.0]          # 来回振荡
p4 = [0.0, 3.0, 6.0, 9.0, 8.0]           # 端点不同
print(f"  端点 0→10 的三条路径：{seq(p1):+.4f} / {seq(p2):+.4f} / {seq(p3):+.4f}")
print(f"  端点 0→8  的路径     ：{seq(p4):+.4f}")
print(f"  理论值 edw·ΔΦ = 0.5×(10−0) = +5.0000 ⇒ 前三条**完全相等**（中间怎么走不进分）")

print("\n=== T5 elixir_bonus：唯一按绝对存量给分的项（默认 0.0 关闭）===")
print(f"  默认 elixir_bonus={_DEFAULT_REWARD['elixir_bonus']}（关闭）")
rw = dict(ZERO, elixir_bonus=0.01)
for e in (0.0, 5.0, 10.0):
    print(f"  绝对圣水={e:>5.1f}、Δ=0 → 分 {frame(rw, my_elixir_after=e):+.4f}")
print("  ⇒ 开启后分数只看**当前帧绝对圣水**，与上一帧无关（这是唯一的例外项）")

print("\n=== T6 事件项（非差分、非存量）：终局与非法动作 ===")
rw = dict(ZERO, win_bonus=10.0, lose_penalty=10.0, draw_penalty=10.0, invalid_penalty=0.05)
print(f"  胜 winner=0        → {frame(rw, winner=0):+.4f}")
print(f"  负 winner=1        → {frame(rw, winner=1):+.4f}")
print(f"  平 game_over=True  → {frame(rw, winner=None, game_over=True):+.4f}")
print(f"  非法 3 次          → {frame(rw, invalid_count=3):+.4f}")
print("  ⇒ 这些是**一次性事件**，与逐帧差分无关（只在终止帧/该帧结算）")
