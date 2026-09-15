# -*- coding: utf-8 -*-
"""GAE 核函数体检（只读）：GAE 会不会把逐帧奖励"平均化/抹平"？

问题（2026-09-14）：奖励是**逐决策帧**结算的标量（每 0.5 s 游戏时间一次），
进网络前先经 GAE 摊成逐帧优势。要回答"逐帧分数有没有在这一步被平均掉"，
必须把 GAE 当**线性算子**量它：核系数、守恒量、时间分辨率、边界重置。

用**真实** PPOTrainer.compute_gae（不重写公式），全部读数脚本复算。

用法（仓库根目录）：
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/probe_gae_kernel.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src", "clasher_new"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
from rl.ppo import PPOTrainer

GAMMA = 0.997        # rl/config.py:127（无预设覆盖）
T = 360              # max_ep_steps：360 决策帧 = 180 s 游戏时间
SPIKE = 300          # 单帧尖峰位置（留出后续帧以验证"未来不影响过去"）


def gae(rew, lam, val=None, term=None, trunc=None, last=0.0):
    """V=0 时 adv 就是奖励序列的 GAE 线性变换（returns 未用）。"""
    v = [0.0] * len(rew) if val is None else val
    d = [False] * len(rew) if term is None else term
    a, _ = PPOTrainer.compute_gae(rew, v, d, GAMMA, lam,
                                  truncated=trunc, last_value=last)
    return np.asarray(a, dtype=np.float64)


print(f"gamma={GAMMA}  T={T}  spike@{SPIKE}")
print(f"{'lambda':>7} {'g*lam':>9} {'半衰期(帧)':>10} {'半衰期(s)':>9} "
      f"{'1-g*lam':>9} {'尖峰总权重':>10} {'留在自身帧':>10}")
for lam in (0.95, 0.99):
    gl = GAMMA * lam
    rew = [0.0] * T
    rew[SPIKE] = 1.0
    a = gae(rew, lam)
    tot, own = float(a.sum()), float(a[SPIKE])
    half = float(np.log(0.5) / np.log(gl))
    print(f"{lam:>7.2f} {gl:>9.6f} {half:>10.2f} {half*0.5:>9.1f} {1-gl:>9.6f} "
          f"{tot:>10.4f} {100*own/tot:>9.2f}%")

print("\n[A] 线性性（lam=0.99）：max|adv(r1+r2) - adv(r1) - adv(r2)|")
r1 = [0.0] * T; r1[100] = 1.0
r2 = [0.0] * T; r2[200] = -2.5
r12 = [x + y for x, y in zip(r1, r2)]
print(f"    = {np.abs(gae(r12, 0.99) - (gae(r1, 0.99) + gae(r2, 0.99))).max():.3e}"
      "   （=0 ⇒ 无任何非线性/平均/归一化环节）")

print("\n[B] 核系数（lam=0.99）：单帧尖峰 r_300=1 → adv[300-k] 应恒等于 (g*lam)^k")
rew = [0.0] * T; rew[SPIKE] = 1.0
a = gae(rew, 0.99); gl = GAMMA * 0.99
for k in (0, 1, 2, 5, 10, 26, 53, 100):
    print(f"    k={k:>3}  adv[{SPIKE-k:>3}]={a[SPIKE-k]:.6f}  预测 (g*lam)^k={gl**k:.6f}"
          f"  偏差={abs(a[SPIKE-k]-gl**k):.2e}")
print(f"    k>0 方向（尖峰之后的帧）：adv[301..305] = {np.round(a[301:306], 8).tolist()}"
      "  （未来不影响过去）")
print(f"    k=0 系数恰为 {a[SPIKE]:.6f} ⇒ 该帧奖励**全部**留在该帧优势里（未被摊薄/平均）")

print("\n[C] 守恒量（lam=0.99）：sum_j adv[j] 是否 = sum_t r_t * (1-(g*lam)^(t+1))/(1-g*lam)"
      "（仅因序列两端截断而与常数权相差，无平均）")
rng = np.random.default_rng(0)
rew = rng.normal(0.0, 1.0, size=T).tolist()
a = gae(rew, 0.99)
w = np.array([(1 - gl ** (t + 1)) / (1 - gl) for t in range(T)])
print(f"    sum(adv)={a.sum():.9f}   sum(r*w)={float(np.dot(np.asarray(rew), w)):.9f}"
      f"   相对偏差={abs(a.sum()-float(np.dot(np.asarray(rew), w)))/abs(a.sum()):.2e}")
print(f"    权重范围 w[0]={w[0]:.6f}（局首：只辐射 1 帧） → w[T-1]={w[-1]:.4f}"
      f"（局末：可辐射 {w[-1]:.1f} 帧）")

print("\n[D] 时间分辨率（lam=0.99）：同一分奖励摊到「前 1 帧」与「前 9 帧」的系数")
print(f"    (g*lam)^1={gl**1:.6f}  (g*lam)^9={gl**9:.6f}  比值={gl**1/gl**9:.4f}"
      f" ⇒ 8 帧时间差只换来 {100*(1-gl**8):.1f}% 的权重差（信用分配时间分辨粗）")

print("\n[E] 局边界：GAE 按局独立（done@300，尖峰在 299）")
term = [False] * T; term[300] = True
rew = [0.0] * T; rew[299] = 1.0
print(f"    带 done@{300} 与不带：adv[299]={gae(rew,0.99,term=term)[299]:.6f} vs "
      f"{gae(rew,0.99)[299]:.6f}  max|diff|={np.abs(gae(rew,0.99,term=term)-gae(rew,0.99)).max():.2e}")

print("\n[F] 本项目 critic 实测为常数 c=+0.1425485 时，GAE 里的 V 项还剩什么")
c = 0.1425485
print(f"    delta_t = r_t + gamma*c - c = r_t + {(GAMMA-1)*c:+.6f}"
      f" ⇒ 优势 ≈ 折扣回报 − {-(GAMMA-1)*c:.6f}/帧（V 项几乎不构成减法）")
print(f"    对照：若 critic 能解释 R 的 var 比例 ρ，则被减掉的是 ρ 那部分的**状态可解释分量**"
      "（期望上不改变策略梯度，前提是 baseline 与动作无关）")

print("\n[G] adv_norm='scale' 的跨帧耦合（唯一会让它不是逐帧独立的环节）")
print("    实际口径：advs = adv / std(adv)（整批 128 帧共用一个标量，只除不中心化）")
for s in (0.05, 0.52, 5.0):
    print(f"    批 std={s:>5.2f} ⇒ 同一帧的 adv 被乘 {1.0/s:>7.3f} 倍"
          f"（逐帧相对结构不变，绝对幅度随同批其他帧变化）")
