# -*- coding: utf-8 -*-
"""`rl/threat_precise.py` 的对账 + 标定 + 验收（只读）。

§1 **对账**：`combine_both_directions` 的两个方向是否与 `estimate_tower_threat`
   单调用**逐位相等**（这是"双向一次算完"能成立的前提）。
§2 **标定**：`HP_PER_PRESSURE` 按预注册写死的规则（等均值）标定。
§3 **验收**：把 `PreciseThreat` 真跑在真实分布 rollout 上，量
   触发率 / 摊销成本 / 双向质量（BA、召回、精度）。

用法（必须在 src/clasher_new 下运行）：
    cd src/clasher_new && PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe \
        ../../scripts/probe_precise_threat.py --frames 400
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src", "clasher_new")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np  # noqa: E402


def ba_of(pred, truth):
    pred = np.asarray(pred, bool); truth = np.asarray(truth, bool)
    tp = int((pred & truth).sum()); fn = int((~pred & truth).sum())
    tn = int((~pred & ~truth).sum()); fp = int((pred & ~truth).sum())
    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    tnr = tn / (tn + fp) if (tn + fp) else float("nan")
    return (float(np.nanmean([tpr, tnr])), tpr, tnr, tp, fp, fn, tn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/d1_long_100k/solo_main_100000.pt")
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--H", type=float, default=5.0)
    a = ap.parse_args()

    import torch
    from rl.env_wrapper import RLEnv
    from rl.belief import BeliefInference
    from rl.belief_planner import BeliefPlanner, _enemy_pressure
    from rl.follower import FollowerPolicy, BELIEF_DIM
    from rl.plan_space import PLAN_DIM
    from rl.train_follower import FollowerOpponent
    from rl.threat_precise import (PreciseThreat, combine_both_directions,
                                   trigger_directions, HP_PER_PRESSURE)
    from threat_calc import estimate_tower_threat

    print("=" * 78)
    print("§1 对账：combine_both_directions（一次推演）vs 工具单调用（两次推演）")
    env = RLEnv(opponent=None, seed=a.seed)
    obs, _ = env.reset(seed=a.seed)
    pol = FollowerPolicy(hidden=128, plan_dim=PLAN_DIM, belief_dim=BELIEF_DIM,
                         value_bypass=True, value_independent=True)
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    pol.load_state_dict(sd.get("state_dict", sd), strict=False)
    pol.eval()
    opp_side = FollowerOpponent(
        pol, env, belief=BeliefInference(opp_deck=env.deck1, n_particles=128,
                                         seed=a.seed), deterministic=True)
    pol.plan_biases_enabled = True
    env.opponent = opp_side
    belief = BeliefInference(opp_deck=env.deck1, n_particles=128, seed=a.seed)
    bp = BeliefPlanner()

    provider = PreciseThreat(horizon=a.H)
    acc = {"calls": 0, "triggers": 0, "sim_ms": 0.0}
    rows = []
    hidden = None
    for i in range(a.frames):
        b = env.battle
        comb = combine_both_directions(b, a.H)
        e0 = float(estimate_tower_threat(b, 0, horizon=a.H)["total"])
        e1 = float(estimate_tower_threat(b, 1, horizon=a.H)["total"])
        thr_old, my_old = _enemy_pressure(b)
        # 触发的"电平"判定（用于 §3 的成本对照），以及 provider 的真实行为
        tA, _hA = trigger_directions(b, 1)
        tB, _hB = trigger_directions(b, 0)
        t0 = time.perf_counter()
        pthr, pmy = provider.pressures(b)
        dt = (time.perf_counter() - t0) * 1e3
        rows.append({"comb0": comb["to_p0"], "comb1": comb["to_p1"],
                     "e0": e0, "e1": e1, "thr_old": float(thr_old),
                     "my_old": float(my_old), "tA": bool(tA), "tB": bool(tB),
                     "pthr": float(pthr), "pmy": float(pmy), "ms": dt})
        # 推进
        plan_vec = bp.plan(b, belief.state(), obs).to_vector()
        btok = belief.encode(obs, None)
        bundle, lp, v, hidden, masks = pol.act(
            obs, btok, plan_vec, env.get_action_mask,
            hidden=hidden, deterministic=False)
        obs2, r_, term, trunc, info = env.step(bundle)
        belief.update(obs2, info.get("opp_played"))
        opp_side.observe_opponent_played(info.get("opp_played") or [])
        obs = obs2
        if term or trunc:
            obs, _ = env.reset(seed=a.seed + i + 1)
            belief.reset(env.deck1)
            opp_side.reset()
            for _k in acc:                      # reset 会清零 provider.stats ⇒ 先累计
                acc[_k] += provider.stats[_k]
            provider.reset()
            hidden = None

    c0 = np.array([r["comb0"] for r in rows]); e0 = np.array([r["e0"] for r in rows])
    c1 = np.array([r["comb1"] for r in rows]); e1 = np.array([r["e1"] for r in rows])
    print(f"    {len(rows)} 帧：方向 to_p0  max|Δ|={np.abs(c0 - e0).max():.4f} "
          f"不等帧={int((c0 != e0).sum())}")
    print(f"    {len(rows)} 帧：方向 to_p1  max|Δ|={np.abs(c1 - e1).max():.4f} "
          f"不等帧={int((c1 != e1).sum())}")
    ok_recon = bool((c0 == e0).all() and (c1 == e1).all())
    print(f"    ⇒ 对账 {'PASS（逐位相等，可安全用一次推演替代两次）' if ok_recon else '**FAIL**'}")

    # ---------------- §2 标定 ----------------
    print("\n" + "=" * 78)
    print("§2 折算常量 HP_PER_PRESSURE 标定（规则跑前写死：**等均值**）")
    thr_old = np.array([r["thr_old"] for r in rows])
    my_old = np.array([r["my_old"] for r in rows])
    k_a = float(e0.mean() / thr_old.mean()) if thr_old.mean() > 1e-9 else float("nan")
    k_b = float(e1.mean() / my_old.mean()) if my_old.mean() > 1e-9 else float("nan")
    print(f"    旧刻度均值: threat={thr_old.mean():.4f} my_pressure={my_old.mean():.4f}")
    print(f"    新刻度均值: to_p0={e0.mean():.1f} HP  to_p1={e1.mean():.1f} HP")
    print(f"    等均值常量: 方向A={k_a:.1f}  方向B={k_b:.1f}  "
          f"⇒ 取单一常量（两方向同源）HP_PER_PRESSURE 建议 = "
          f"{(k_a + k_b) / 2:.0f}（当前源码值 {HP_PER_PRESSURE:.0f}）")
    pr_old = float((thr_old >= 2.0).mean())
    best = None
    for k in np.arange(50.0, 2000.0, 5.0):
        d = abs(float((e0 / k >= 2.0).mean()) - pr_old)
        if best is None or d < best[0]:
            best = (d, float(k))
    k_pr = best[1]
    print(f"    规则② 等阳性率：k ≈ {k_pr:.0f}（旧阳性率 {pr_old:.1%}）"
          f" —— **与规则①差 {abs(k_pr - (k_a + k_b) / 2) / ((k_a + k_b) / 2):.0%}**"
          f" ⇒ 单常量服务不了两个用途（【R17】类问题）")
    for k in ((k_a + k_b) / 2, k_pr, HP_PER_PRESSURE, 500.0, 1000.0):
        if not np.isfinite(k) or k <= 0:
            continue
        pr_old = float((thr_old >= 2.0).mean())
        pr_new = float((e0 / k >= 2.0).mean())
        ve_old = np.tanh(my_old - thr_old)
        ve_new = np.tanh((e1 - e0) / k)
        print(f"    k={k:8.1f}: threat≥2 阳性率 旧={pr_old:5.1%} 新={pr_new:5.1%} "
              f"| tanh 饱和 旧={float((np.abs(ve_old) > 0.99).mean()):5.1%} "
              f"新={float((np.abs(ve_new) > 0.99).mean()):5.1%}")

    # ---------------- §3 验收 ----------------
    print("\n" + "=" * 78)
    print("§3 验收：PreciseThreat 真跑在真实分布上的成本与质量")
    acc["calls"] += provider.stats["calls"]
    acc["triggers"] += provider.stats["triggers"]
    acc["sim_ms"] += provider.stats["sim_ms"]
    st = acc
    print(f"    调用 {st['calls']} 次 | 触发 {st['triggers']} 次 "
          f"({st['triggers'] / max(1, st['calls']):.1%} 的调用触发) | "
          f"推演总耗时 {st['sim_ms']:.1f} ms")
    print(f"    ⇒ 摊销 {st['sim_ms'] / max(1, st['calls']):.2f} ms/决策帧"
          f"（预算 3.85）")
    # 电平触发的成本对照
    print(f"    [对照] 若用**电平**触发（条件成立即算）："
          f"A={np.mean([r['tA'] for r in rows]):.1%} "
          f"B={np.mean([r['tB'] for r in rows]):.1%} "
          f"A∪B={np.mean([r['tA'] or r['tB'] for r in rows]):.1%}")
    print(f"    [对照] 双向分别触发 A∧B（同时需要两侧）="
          f"{np.mean([r['tA'] and r['tB'] for r in rows]):.1%} "
          f"⇒ 双向一次算完省的正是这一块")
    # 质量：用 provider 的输出 vs 引擎真值（两个方向各自）
    for tag, key, E in (("A 敌方压我 (threat)", "pthr", e0),
                        ("B 我压敌方 (my_pressure)", "pmy", e1)):
        p = np.array([r[key] for r in rows]) * HP_PER_PRESSURE
        posE = E[E > 0]
        med = float(np.median(posE)) if len(posE) else 0.0
        for tname, truth in (("E>0", E > 0),
                             (f"E≥{med:.0f}HP", E >= med)):
            for thr_name, pred in (("新(精确)", p > 0),
                                   ("旧(粗糙)", (np.array([r["thr_old"] for r in rows]) >= 2.0)
                                    if key == "pthr" else
                                    (np.array([r["my_old"] for r in rows]) >= 2.0))):
                ba, tpr, tnr, tp, fp, fn, tn = ba_of(pred, truth)
                print(f"    {tag:26s} 真值{tname:12s} {thr_name:8s} "
                      f"BA={ba:.4f} 召回={tpr:5.1%} 精确率="
                      f"{(tp / (tp + fp) if (tp + fp) else float('nan')):5.1%} "
                      f"| TP={tp} FP={fp} FN={fn}")

    print("\n    读法：诊断性质的读数；`PreciseThreat` 的产量在未触发帧会回退旧公式，"
          "所以这里的 BA 主要反映**触发帧**的质量。")


if __name__ == "__main__":
    main()
