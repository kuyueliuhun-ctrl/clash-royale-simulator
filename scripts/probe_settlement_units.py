# -*- coding: utf-8 -*-
"""结算单位代价探针（只读、离线）：把"按帧计分"换成事件/局末结算，代价是多少？

回答「能否完全取缔按帧计分」。四项读数全部可复算：
  A) 批结构：128 帧的更新批覆盖几个局、多少批含局末；
  B) 可见度：奖励被推迟 d 帧后，在优势里剩 (γλ)^d（λ=0.99）与纯折现 γ^d；
  C) 信息密度：一个批携带多少"奖励事件"（逐帧 / 决策帧对齐 / 局末 三档）；
  D) 空帧账：录得读数（docs/reward_composition_100k.log）的**闭合校验**——
     edw 项的正贡献是否真的全部来自"没有出牌的帧"。
用法（仓库根）：
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/probe_settlement_units.py
"""
from __future__ import annotations

import argparse
import os
import sys

# T1-1b 补齐（2026-09-19）：原先这里是**手写**的 UTF-8 兜底块（只处理 stdout、且不处理 stderr
# ⇒ traceback 在 GBK 下仍是乱码）。现收敛到 T1-1 的**单一实现**（两路 + errors='replace'）。
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                                 "src", "clasher_new"))
from rl.io_bootstrap import force_utf8_stdout  # noqa: E402
force_utf8_stdout()

import numpy as np

GAMMA, LAM, WIN = 0.997, 0.99, 128          # rl/config.py:127 / :351 / :112


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="src/clasher_new/runs/_probe_cache_v2_stoch.npz")
    ap.add_argument("--deploy-interval-frames", type=float, default=11.0,
                    help="部署间隔中位（帧）：行为取证实测 5.5 s ÷ 0.5 s/帧 = 11 帧")
    ap.add_argument("--deploy-frame-ratio", type=float, default=0.087,
                    help="部署帧占比：forensics 8804/100860 = 8.7%")
    a = ap.parse_args()

    gl = GAMMA * LAM
    z = np.load(a.npz)
    ep = z["ep"].astype(np.int64)
    n, n_ep = len(ep), int(ep.max()) + 1
    ep_len = n / n_ep

    print(f"[load] {a.npz}: frames={n} episodes={n_ep} ep_len_mean={ep_len:.1f} "
          f"| γ={GAMMA} λ={LAM} γλ={gl:.5f} batch={WIN} 帧")

    # A) 批结构
    eps_per, has_term = [], 0
    for s in range(0, n, WIN):
        e = min(n, s + WIN)
        blk = ep[s:e]
        eps_per.append(len(np.unique(blk)))
        if any(int(np.where(ep == v)[0][-1]) in range(s, e) for v in np.unique(blk)):
            has_term += 1
    eps_per = np.array(eps_per)
    nb = len(eps_per)
    print(f"\n[A 批结构] 共 {nb} 批（{WIN} 帧/批）")
    print(f"  批内不同局数：1 局 {100*np.mean(eps_per == 1):.1f}% | 2 局 "
          f"{100*np.mean(eps_per == 2):.1f}% | ≥3 局 {100*np.mean(eps_per >= 3):.1f}%")
    print(f"  批内覆盖的局数均值 = {eps_per.mean():.2f}（<1 ⇒ 一个批装不满一个局）")
    print(f"  含局末（终止帧）的批 = {has_term}/{nb} = {100*has_term/nb:.1f}%"
          f" ⇒ 局末结算**不是**「大多数批全零」")
    print(f"  含部署帧（{100*a.deploy_frame_ratio:.1f}% 的帧）的批 ≈ "
          f"{100*(1-(1-a.deploy_frame_ratio)**WIN):.1f}%")

    # B) 可见度
    print(f"\n[B 可见度] 奖励推迟 d 帧发放后，在优势里剩多少（(γλ)^d）；γ^d 为纯折现")
    for d in (0, 1, 11, 17, 30, 50, 100, 150, 217, 300, 360):
        tag = ""
        if d == 11:
            tag = "  ← 部署间隔中位（决策帧对齐的典型延迟）"
        if d == 17:
            tag = "  ← 部署间隔 p90"
        if d == 217:
            tag = "  ← 本样本均局长（局末结算的典型延迟）"
        print(f"  d={d:>3}：(γλ)^d={gl**d:.4f}  γ^d={GAMMA**d:.4f}{tag}")

    # C) 信息密度
    print(f"\n[C 信息密度] 一个 {WIN} 帧批携带的「奖励事件」数")
    dpr = a.deploy_frame_ratio
    print(f"  逐帧结算（现状）          ：{WIN} 个（每帧 1 个标量）")
    print(f"  决策帧对齐（只在部署帧发放）：{WIN*dpr:.1f} 个（≈ 中位间隔 "
          f"{a.deploy_interval_frames:.0f} 帧 ⇒ {WIN/a.deploy_interval_frames:.1f}）")
    print(f"  局末结算（每局 1 个）      ：{eps_per.mean():.2f} 个"
          f"（= 批内局数；相对现状降 {WIN/eps_per.mean():.0f}×）")

    # D) 空帧账闭合校验（录得读数）
    print("\n[D 空帧账闭合校验] 来自 docs/reward_composition_100k.log §3（2968 帧样本）")
    act_n, act_m = 276, -0.111685
    emp_n, emp_m = 2692, +0.015685
    act_s, emp_s = act_n * act_m, emp_n * emp_m
    print(f"  下牌/施法帧 n={act_n} mean={act_m:+.6f} ⇒ 合计 {act_s:+.3f}")
    print(f"  空帧       n={emp_n} mean={emp_m:+.6f} ⇒ 合计 {emp_s:+.3f}")
    print(f"  两项合计 = {act_s + emp_s:+.3f}；日志记录的 edw Σ(带符号) = +11.400 "
          f"⇒ 闭合偏差 {abs(act_s + emp_s - 11.400):.3f}")
    print("  ⇒ **edw 项的全部正贡献来自「没有出牌的帧」**（对方单位死亡注销份额 = 过去动作的延迟后果）")


if __name__ == "__main__":
    main()
