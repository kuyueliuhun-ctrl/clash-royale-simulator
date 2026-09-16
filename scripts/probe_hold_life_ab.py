# -*- coding: utf-8 -*-
"""保持寿命上限（MAX_HOLD_S）的**同轨迹配对 A/B**（只读）。

为什么需要这个仪器：`docs/hold_recompute_probe_after.log` 显示带寿命上限后摊销涨到
4.03/2.90/6.23 ms/帧（只有 1/3 seed 进预算），但那是**跨 run 比较**——按【红线 R5】，
同 seed 同代码两次 run 也不可复现，跨 run 之差不能归因给改动。
本脚本把「有寿命上限」与「无寿命上限」两个 `PreciseThreat` **挂在同一条轨迹上**
（两者都只做观测者，规划器走生产默认的粗糙口径），于是：

- 成本 / 触发次数 / 到期次数：同一批盘面上的**配对数**；
- 质量：同一批抽样帧、同一真值（此刻重算）上的配对 BA；
- 值龄：直接用 `_hold_t`（该值算出的游戏时刻）算，**不再用"保持段年龄"**
  （带寿命上限后，到期+同帧重算在帧序列上是一个连续保持段 ⇒ 段年龄会系统性偏大）。

⚠️ 口径：规划器不看这两个提供器 ⇒ 本脚本量的是**估计器**的成本与质量，
**不是**策略效果（策略效果要端到端训练 A/B）。

用法（必须在 src/clasher_new 下运行）：
    cd src/clasher_new && PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe \
        ../../scripts/probe_hold_life_ab.py --frames 900 --stride 3
"""
from __future__ import annotations

import argparse
import json
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
    return float(np.nanmean([tpr, tnr])), tp, fp, fn, tn


def _q(a, p):
    a = np.asarray(a, dtype=float)
    return float(np.percentile(a, p)) if len(a) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/d1_long_100k/solo_main_100000.pt")
    ap.add_argument("--frames", type=int, default=900)
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 13])
    ap.add_argument("--H", type=float, default=5.0)
    ap.add_argument("--cap", type=float, default=None, help="寿命上限（缺省=模块常量 MAX_HOLD_S）")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--max-samples", type=int, default=400)
    ap.add_argument("--save", default=os.path.join(_ROOT, "docs", "hold_life_ab_samples.json"))
    a = ap.parse_args()

    import torch
    from rl.env_wrapper import RLEnv
    from rl.belief import BeliefInference
    from rl import belief_planner as bp_mod
    from rl.belief_planner import BeliefPlanner, PRESSURE_THRESHOLD
    from rl.follower import FollowerPolicy, BELIEF_DIM
    from rl.plan_space import PLAN_DIM
    from rl.train_follower import FollowerOpponent
    from rl.threat_precise import (PreciseThreat, combine_both_directions,
                                   HP_PER_PRESSURE, MAX_HOLD_S)

    CAP = float(a.cap) if a.cap is not None else float(MAX_HOLD_S)
    K = HP_PER_PRESSURE
    THR_HP = PRESSURE_THRESHOLD * K

    print("=" * 78)
    print(f"保持寿命上限 · 同轨迹配对 A/B · 上限={CAP}s vs 无上限 · H={a.H}s · "
          f"seeds={a.seeds} · frames/seed={a.frames}")
    print("（规划器走生产默认粗糙口径 ⇒ 两个提供器只做观测者，看到**同一条**盘面序列）")

    # 每个 seed 一条轨迹；两个臂同帧调用
    stats = {arm: {"calls": 0, "triggers": 0, "sim_ms": 0.0, "expired": 0} for arm in ("cap", "raw")}
    samples = []
    sim_ms_calls = []
    n_frames = 0

    for seed in a.seeds:
        env = RLEnv(opponent=None, seed=seed)
        obs, _ = env.reset(seed=seed)
        pol = FollowerPolicy(hidden=128, plan_dim=PLAN_DIM, belief_dim=BELIEF_DIM,
                             value_bypass=True, value_independent=True)
        sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        pol.load_state_dict(sd.get("state_dict", sd), strict=False)
        pol.eval()
        opp_side = FollowerOpponent(
            pol, env, belief=BeliefInference(opp_deck=env.deck1, n_particles=128,
                                             seed=seed), deterministic=True)
        pol.plan_biases_enabled = True
        env.opponent = opp_side
        belief = BeliefInference(opp_deck=env.deck1, n_particles=128, seed=seed)
        bp = BeliefPlanner()
        bp_mod.set_precise_threat(False)                 # 规划器用生产默认（粗糙）
        arms = {"cap": PreciseThreat(horizon=a.H, max_hold_s=CAP),
                "raw": PreciseThreat(horizon=a.H, max_hold_s=float("inf"))}
        cand_n = 0
        hidden = None
        for i in range(a.frames):
            b = env.battle
            n_frames += 1
            vals = {}
            for arm, prov in arms.items():
                pre = dict(prov.stats)
                vals[arm] = prov.pressures(b)
                post = prov.stats
                stats[arm]["calls"] += 1
                stats[arm]["triggers"] += post["triggers"] - pre["triggers"]
                stats[arm]["expired"] += post["expired"] - pre["expired"]
                stats[arm]["sim_ms"] += post["sim_ms"] - pre["sim_ms"]
            # 值龄（精确：该值算出的游戏时刻 → 现在）
            ages = {arm: {t: (None if arms[arm]._hold[t] is None or arms[arm]._hold_t[t] is None
                              else float(b.time) - arms[arm]._hold_t[t]) for t in ("A", "B")}
                    for arm in arms}
            # 抽样真值
            stale_any = any(arms["raw"]._hold[t] is not None or arms["cap"]._hold[t] is not None
                            for t in ("A", "B"))
            if stale_any:
                cand_n += 1
            if stale_any and cand_n % max(1, a.stride) == 0 and len(samples) < a.max_samples:
                crude_t, crude_m = bp_mod._crude_enemy_pressure(b)
                t0 = time.perf_counter()
                comb = combine_both_directions(b, a.H)
                sim_ms_calls.append((time.perf_counter() - t0) * 1e3)
                for t, idx in (("A", 0), ("B", 1)):
                    if arms["raw"]._hold[t] is None and arms["cap"]._hold[t] is None:
                        continue                     # 两臂都没有精确值 ⇒ 与本次改动无关
                    samples.append({
                        "seed": seed, "frame": i, "tag": t,
                        "fresh": float(comb["to_p0" if t == "A" else "to_p1"]),
                        "crude": float(crude_t if t == "A" else crude_m),
                        # 两臂**实际输出**（pressure 刻度；无精确值时为该臂的粗糙回退值）
                        "cap_out": float(vals["cap"][idx]),
                        "raw_out": float(vals["raw"][idx]),
                        "cap_held": arms["cap"]._hold[t] is not None,
                        "raw_held": arms["raw"]._hold[t] is not None,
                        "cap_age": ages["cap"][t], "raw_age": ages["raw"][t],
                    })
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
                obs, _ = env.reset(seed=seed + i + 1)
                belief.reset(env.deck1)
                opp_side.reset()
                hidden = None
                for prov in arms.values():
                    prov.reset()

    ms_call = float(np.mean(sim_ms_calls)) if sim_ms_calls else float("nan")
    print(f"\n{'=' * 78}\n§1 成本与触发（**同一条轨迹**，配对）")
    print(f"  帧数 {n_frames}；单次推演（真值调用实测）均值 {ms_call:.1f} ms")
    for arm, name in (("cap", f"有寿命上限 {CAP:.0f}s"), ("raw", "无寿命上限")):
        st = stats[arm]
        amort = st["sim_ms"] / max(1, n_frames)
        base = st["sim_ms"] / max(1, st["triggers"]) if st["triggers"] else float("nan")
        print(f"  {name:16s}: 推演 {st['triggers']:4d} 次（{st['triggers']/n_frames:5.2%}）"
              f" | 到期 {st['expired']:4d} 次 | 摊销 {amort:5.2f} ms/帧"
              f" | 单次均值 {base:5.1f} ms | {'≤3.85 ✅' if amort <= 3.85 else '**>3.85 ❌**'}")
    d_trig = stats["cap"]["triggers"] - stats["raw"]["triggers"]
    print(f"  ⇒ 寿命上限带来的**配对新增推演** = {d_trig} 次"
          f"（+{d_trig / max(1, stats['raw']['triggers']):.1%}）"
          f" ⇒ 摊销增量 {(stats['cap']['sim_ms'] - stats['raw']['sim_ms']) / max(1, n_frames):+.2f} ms/帧")

    print(f"\n{'=' * 78}\n§2 值龄（该值算出的游戏时刻 → 现在，秒）")
    for arm, name in (("cap", "有上限"), ("raw", "无上限")):
        ag = [s[f"{arm}_age"] for s in samples if s[f"{arm}_age"] is not None]
        print(f"  {name}: n={len(ag)} 中位 {_q(ag,50):.1f}s p90 {_q(ag,90):.1f}s "
              f"最大 {max(ag) if ag else 0:.1f}s | ≥{CAP:.0f}s 占比 "
              f"{np.mean([x >= CAP for x in ag]) if ag else float('nan'):.1%}"
              f" | ≥{2*CAP:.0f}s 占比 "
              f"{np.mean([x >= 2*CAP for x in ag]) if ag else float('nan'):.1%}")

    print(f"\n{'=' * 78}\n§3 质量（同一批抽样帧、同一真值「此刻重算 ≥ 阈值」；【R9⑤】同分母）")
    def _cmp(rows_sub, label):
        if not rows_sub:
            print(f"  {label}: n=0")
            return
        tr = np.array([s["fresh"] >= THR_HP for s in rows_sub])
        out = {}
        for key, name in (("raw_out", "无上限"), ("cap_out", f"有上限 {CAP:.0f}s"),
                          ("crude", "粗糙回退")):
            pred = np.array([s[key] >= PRESSURE_THRESHOLD for s in rows_sub])
            ba, tp, fp, fn, tn = ba_of(pred, tr)
            out[key] = ba
            print(f"  {label} {name:14s}: BA={ba:.4f} TP={tp:4d} FP={fp:4d} FN={fn:4d}")
        print(f"  {label} ⇒ 配对差 BA(有上限) − BA(无上限) = {out['cap_out'] - out['raw_out']:+.4f}"
              f"（真值阳性率 {tr.mean():.1%}，n={len(rows_sub)}）")
    if samples:
        _cmp([s for s in samples if s["raw_held"]], "[无上限有精确值]")
        print()
        _cmp(samples, "[全部抽样帧]")
        print("  按值龄分桶（值龄取无上限臂；两臂同一批帧）：")
        print("    值龄区间(秒)      n    无上限 BA   有上限 BA     差   有上限持有率")
        for lo, hi in ((0, 2.5), (2.5, 5), (5, 10), (10, 20), (20, 40), (40, 1e9)):
            g = [s for s in samples if s["raw_age"] is not None and lo <= s["raw_age"] < hi]
            if not g:
                continue
            tr = np.array([s["fresh"] >= THR_HP for s in g])
            def _ba(key):
                return ba_of([s[key] >= PRESSURE_THRESHOLD for s in g], tr)[0]
            print(f"    [{lo:5.1f},{hi if hi < 1e9 else 999:5.1f})  {len(g):4d}    "
                  f"{_ba('raw_out'):8.4f}   {_ba('cap_out'):8.4f}   "
                  f"{_ba('cap_out') - _ba('raw_out'):+.4f}   "
                  f"{np.mean([s['cap_held'] for s in g]):6.1%}")

    try:
        with open(a.save, "w", encoding="utf-8") as fh:
            json.dump({"meta": {"cap": CAP, "H": a.H, "seeds": a.seeds,
                                "frames": n_frames, "ms_call": ms_call, "stats": stats},
                       "samples": samples}, fh, ensure_ascii=False)
        print(f"\n[已落盘] {a.save}")
    except Exception as e:
        print(f"\n[落盘失败] {type(e).__name__}")


if __name__ == "__main__":
    main()
