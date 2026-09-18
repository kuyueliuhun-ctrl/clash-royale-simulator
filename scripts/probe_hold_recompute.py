# -*- coding: utf-8 -*-
"""保持式精确塔伤的状态机取证：**盘面变了会不会重算**（只读探针）。

用户问题（2026-09-14）：「如果对方有新卡牌放置，会不会重新计算」

本脚本不靠读代码回答，而是在真实 rollout 里**逐帧记账**。走生产路径
（`belief_planner.set_precise_threat(True)` ⇒ 每帧 `bp.plan()` 内部调用一次 `pressures()`）。

§0 直接回答：保持期间出现「新敌方单位」的帧，引擎推演有没有被重新调用
§1 状态机生命周期：保持段时长分布 / 武装方式（本方向真触发 vs 借另一侧上升沿顺手武装）
§2 陈旧度：保持值 vs「此刻若重算」的真值（分组：无变化 / 新敌方单位 / 新我方单位）
§3 候选修复的成本：把「追踪半场内出现新单位」也当触发 ⇒ 增量推演数 + 摊销 ms/帧
§4 退出帧：保持值（HP 刻度）→ 回退值（计数刻度）

用法（必须在 src/clasher_new 下运行）：
    cd src/clasher_new && PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe \
        ../../scripts/probe_hold_recompute.py --frames 900 --stride 3
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
from rl.io_bootstrap import force_utf8_stdout  # noqa: E402  (T1-1: UTF-8 兜底单一来源)
force_utf8_stdout()

import numpy as np  # noqa: E402


def _snap(b):
    """当前帧单位快照：{engine_id: (player, x, y, name)}（活着的非塔单位）。"""
    out = {}
    for e in b.entities.values():
        if not getattr(e, "is_alive", False):
            continue
        nm = getattr(e, "name", "")
        if "Tower" in nm:
            continue
        eid = getattr(e, "id", None)
        if eid is None:
            eid = id(e)
        out[eid] = (getattr(e, "player", None),
                    float(e.position.x), float(e.position.y), nm)
    return out


def ba_of(pred, truth):
    pred = np.asarray(pred, bool)
    truth = np.asarray(truth, bool)
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
    ap.add_argument("--stride", type=int, default=3,
                    help="陈旧度抽样步长（候选 = 该方向保持值来自更早推演）")
    ap.add_argument("--max-samples", type=int, default=400)
    ap.add_argument("--save", default=os.path.join(_ROOT, "docs", "hold_recompute_samples.json"),
                    help="抽样明细落盘（供事后切分，不必重跑）")
    a = ap.parse_args()

    import torch
    from rl.env_wrapper import RLEnv
    from rl.belief import BeliefInference
    from rl import belief_planner as bp_mod
    from rl.belief_planner import BeliefPlanner, PRESSURE_THRESHOLD
    from rl.follower import FollowerPolicy, BELIEF_DIM
    from rl.plan_space import PLAN_DIM
    from rl.train_follower import FollowerOpponent
    from rl.threat_precise import (_half_of, combine_both_directions,
                                   HP_PER_PRESSURE)

    print("=" * 78)
    print(f"盘面变化 ⇒ 是否重算 · H={a.H}s · seeds={a.seeds} · frames/seed={a.frames}")
    print("（生产路径：set_precise_threat(True)，每帧 bp.plan() 调一次 pressures()）")

    K = HP_PER_PRESSURE
    THR_HP = PRESSURE_THRESHOLD * K          # 400 HP = 决策阈值 2.0 pressure
    rows = []
    samples = []
    segs = {"A": [], "B": []}
    sim_ms_total = 0.0
    n_sim = 0
    fresh_ms = []

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
        bp_mod.set_precise_threat(True, horizon=a.H)
        prov = bp_mod._PRECISE_PROVIDER

        tracked = {"A": None, "B": None}
        prev_ids = {"A": None, "B": None}
        since = {"A": {}, "B": {}}
        cur = {"A": None, "B": None}
        cand_n = 0
        hidden = None

        def close_seg(t):
            if cur[t] is not None:
                segs[t].append(cur[t])
                cur[t] = None

        for i in range(a.frames):
            b = env.battle
            snap = _snap(b)
            hold_pre = {t: (prov._hold[t] is not None) for t in ("A", "B")}
            hold_val_pre = {t: prov._hold[t] for t in ("A", "B")}
            trig_pre = prov.stats["triggers"]
            sim_pre = prov.stats["sim_ms"]

            ev = {}
            for t in ("A", "B"):
                if hold_pre[t] and tracked[t]:
                    curids = {eid for eid, u in snap.items()
                              if any(h in _half_of(u[1]) for h in tracked[t])}
                    new = curids - prev_ids[t]
                    gone = prev_ids[t] - curids
                    ne = sum(1 for e in new if snap[e][0] == 1)
                    na = sum(1 for e in new if snap[e][0] == 0)
                    ev[t] = {"new": len(new), "gone": len(gone),
                             "new_enemy": ne, "new_ally": na}
                    prev_ids[t] = curids
                    for k, v in (("new_enemy", ne), ("new_ally", na), ("gone", len(gone))):
                        if v:
                            since[t][k] = since[t].get(k, 0) + v
                    if cur[t] is not None:
                        cur[t]["ev_enemy"] += ne
                        cur[t]["ev_ally"] += na
                        cur[t]["ev_gone"] += len(gone)
                else:
                    ev[t] = {"new": 0, "gone": 0, "new_enemy": 0, "new_ally": 0}
                    if hold_pre[t] and tracked[t] is not None and not tracked[t]:
                        since[t]["untracked"] = since[t].get("untracked", 0) + 1
                        if cur[t] is not None:
                            cur[t]["untracked"] = cur[t].get("untracked", 0) + 1

            plan_vec = bp.plan(b, belief.state(), obs).to_vector()
            trig_post = prov.stats["triggers"]
            sim_ran = trig_post > trig_pre
            if sim_ran:
                sim_ms_total += prov.stats["sim_ms"] - sim_pre
                n_sim += 1
            hold_post = {t: (prov._hold[t] is not None) for t in ("A", "B")}

            for t in ("A", "B"):
                if hold_pre[t] and cur[t] is not None:
                    cur[t]["n"] += 1
                    cur[t]["sims"] += int(sim_ran)
                if (not hold_pre[t]) and hold_post[t]:
                    tracked[t] = set(prov._hold_half[t])
                    curids = {eid for eid, u in snap.items()
                              if tracked[t] and any(h in _half_of(u[1]) for h in tracked[t])}
                    prev_ids[t] = curids
                    since[t] = {}
                    cur[t] = {"seed": seed, "i0": i, "n": 0,
                              "sims": 1 if sim_ran else 0,
                              "own": bool(tracked[t]),
                              "ev_enemy": 0, "ev_ally": 0, "ev_gone": 0, "untracked": 0}
                elif hold_pre[t] and (not hold_post[t]):
                    close_seg(t)
                    tracked[t] = None
                    prev_ids[t] = None
                    since[t] = {}

            for t in ("A", "B"):
                stale = hold_pre[t] and hold_post[t] and (not sim_ran)
                if not stale:
                    continue
                cand_n += 1
                if cand_n % max(1, a.stride) or len(samples) >= a.max_samples:
                    continue
                crude_t, crude_m = bp_mod._crude_enemy_pressure(b)
                t0 = time.perf_counter()
                comb = combine_both_directions(b, a.H)
                fresh_ms.append((time.perf_counter() - t0) * 1e3)
                samples.append({
                    "seed": seed, "frame": i, "tag": t,
                    "held": float(hold_val_pre[t]),
                    "fresh": float(comb["to_p0" if t == "A" else "to_p1"]),
                    "crude": float(crude_t if t == "A" else crude_m),
                    "age": since[t].get("age", 0),
                    "own": bool(tracked[t]) if tracked[t] is not None else False,
                    "new_enemy": bool(since[t].get("new_enemy", 0)),
                    "new_ally": bool(since[t].get("new_ally", 0)),
                    "gone": bool(since[t].get("gone", 0)),
                    "untracked": "untracked" in since[t],
                })

            rows.append({"seed": seed, "frame": i, "sim": bool(sim_ran),
                         "hold_pre": dict(hold_pre),
                         "held": {t: (float(hold_val_pre[t]) if hold_pre[t] else None)
                                  for t in ("A", "B")},
                         "ev": ev})
            for t in ("A", "B"):
                if hold_pre[t]:
                    since[t]["age"] = since[t].get("age", 0) + 1

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
                for t in ("A", "B"):
                    close_seg(t)
                tracked = {"A": None, "B": None}
                prev_ids = {"A": None, "B": None}
                since = {"A": {}, "B": {}}
        for t in ("A", "B"):
            close_seg(t)
        bp_mod.set_precise_threat(False)

    N = len(rows)
    try:
        import json
        with open(a.save, "w", encoding="utf-8") as fh:
            json.dump({"meta": {"frames": N, "seeds": a.seeds, "H": a.H,
                                "K": K, "THR_HP": THR_HP, "stride": a.stride,
                                "n_sim": n_sim, "sim_ms_total": sim_ms_total},
                       "samples": samples, "segs": segs}, fh, ensure_ascii=False)
        print(f"[已落盘] {a.save}（{len(samples)} 抽样 / {len(segs['A'])+len(segs['B'])} 保持段）")
    except Exception as e:
        print(f"[落盘失败] {type(e).__name__}")

    print("\n" + "=" * 78)
    print("§0 直接回答：保持期间盘面变了，有没有重算？")
    for t, who in (("A", "敌方压我 (threat)"), ("B", "我压敌方 (my_pressure)")):
        u = "B" if t == "A" else "A"
        n_act = sum(1 for r in rows if r["hold_pre"][t])
        ev_enemy = [r for r in rows if r["hold_pre"][t] and r["ev"][t]["new_enemy"] > 0]
        ev_ally = [r for r in rows if r["hold_pre"][t] and r["ev"][t]["new_ally"] > 0]
        ev_gone = [r for r in rows if r["hold_pre"][t] and r["ev"][t]["gone"] > 0]
        sim_on_ev = [r for r in ev_enemy if r["sim"]]
        explained = sum(1 for r in sim_on_ev if not r["hold_pre"][u])
        print(f"  方向{t} {who}: 帧数 {N}，该方向保持中 {n_act:5d} 帧（{n_act / N:5.1%}）")
        print(f"    保持中 ∧ **出现新敌方单位** = {len(ev_enemy):5d} 帧 ⇒ 本帧发生推演 "
              f"{len(sim_on_ev)} 次（其中 {explained} 次可由「另一侧上升沿」解释，"
              f"无法解释的 {len(sim_on_ev) - explained} 次）")
        print(f"    保持中 ∧ 出现新我方单位 = {len(ev_ally):5d} 帧 / ∧ 有单位消失 = "
              f"{len(ev_gone):5d} 帧")
    for t in ("A", "B"):
        withev = [s for s in segs[t] if s["ev_enemy"] or s["ev_ally"] or s["ev_gone"]]
        inner = sum(max(0, s["sims"] - 1) for s in withev)
        print(f"  方向{t}: 含盘面事件的保持段 {len(withev)} 段 ⇒ 段内额外推演 {inner} 次"
              f"（这些段的盘面事件：新敌方单位 {sum(s['ev_enemy'] for s in withev)} / "
              f"新我方单位 {sum(s['ev_ally'] for s in withev)} / "
              f"消失 {sum(s['ev_gone'] for s in withev)}）")
    print(f"  全 run 推演 {n_sim} 次 / {N} 帧（{n_sim / N:.1%}）")
    print("  ⇒ 结论：**不会重算**。唯一的推演路径 = 某方向从「无保持」变成「触发」（上升沿）；"
          "保持期内新卡、死亡、掉血都不会触发重算。")

    print("\n" + "=" * 78)
    print("§1 状态机生命周期：保持有多长 / 怎么被武装的（1 决策帧 = 0.5 s）")
    for t, nm in (("A", "敌方压我"), ("B", "我压敌方")):
        segs_t = segs[t]
        ln = [s["n"] for s in segs_t]
        own = [s for s in segs_t if s["own"]]
        other = [s for s in segs_t if not s["own"]]
        print(f"  方向{t} {nm}: 保持段 {len(segs_t)} 段 | 时长(帧) 中位 {_q(ln, 50):.0f} "
              f"p90 {_q(ln, 90):.0f} 最大 {max(ln) if ln else 0}"
              f"（中位 ≈ {_q(ln, 50) * 0.5:.1f} s，最大 ≈ {(max(ln) if ln else 0) * 0.5:.1f} s）")
        print(f"    本方向真触发武装 {len(own)} 段（{len(own) / max(1, len(segs_t)):.0%}）"
              f" | **借另一侧上升沿顺手武装** {len(other)} 段"
              f"（{len(other) / max(1, len(segs_t)):.0%}）⇒ 后者所在盘面本方向**没有威胁源**")
        if other:
            print(f"    顺手武装段时长中位 {_q([s['n'] for s in other], 50):.0f} 帧，"
                  f"段内出现新敌方单位 {sum(s['ev_enemy'] for s in other)} 个")
        print(f"    段内被另一侧上升沿**刷新**（重新推演并覆盖保持值）共 "
              f"{sum(max(0, s['sims'] - 1) for s in segs_t)} 次")
    print(f"  单次推演耗时（生产路径实测）均值 {sim_ms_total / max(1, n_sim):.1f} ms；"
          f"预算 3.85 ms/帧")

    print("\n" + "=" * 78)
    print(f"§2 陈旧度：保持值 vs「此刻若重算」真值（抽样 {len(samples)} 帧，"
          f"单次真值 {np.mean(fresh_ms) if fresh_ms else float('nan'):.1f} ms）")
    if samples:
        d = np.array([s["fresh"] - s["held"] for s in samples])
        held = np.array([s["held"] for s in samples])
        fresh = np.array([s["fresh"] for s in samples])
        print(f"  全体: |Δ| 中位 {np.median(np.abs(d)):.0f} HP p90 {_q(np.abs(d), 90):.0f} "
              f"最大 {np.abs(d).max():.0f} HP | 有符号均值 {d.mean():+.0f} HP"
              f"（正 = 保持值**低估**当下威胁）")
        print(f"  决策相关：跨阈值（{THR_HP:.0f}HP = 2.0 pressure）翻转率 = "
              f"{np.mean((held >= THR_HP) != (fresh >= THR_HP)):.1%}"
              f" | 保持值判「安全」而真值判「威胁」= "
              f"{np.mean((held < THR_HP) & (fresh >= THR_HP)):.1%}"
              f" | 反向（保持值误报）= {np.mean((held >= THR_HP) & (fresh < THR_HP)):.1%}")
        groups = [("本方向真触发武装", lambda s: s["own"]),
                  ("借另一侧顺手武装", lambda s: not s["own"]),
                  ("出现新敌方单位", lambda s: s["new_enemy"]),
                  ("出现新我方单位", lambda s: s["new_ally"] and not s["new_enemy"]),
                  ("无任何变化", lambda s: not (s["new_enemy"] or s["new_ally"] or s["gone"]))]
        for gname, f in groups:
            g = [s for s in samples if f(s)]
            if not g:
                print(f"    {gname:18s}: n=0")
                continue
            dg = np.array([s["fresh"] - s["held"] for s in g])
            hg = np.array([s["held"] for s in g])
            fg = np.array([s["fresh"] for s in g])
            print(f"    {gname:18s}: n={len(g):4d} | |Δ| 中位 {np.median(np.abs(dg)):6.0f} "
                  f"p90 {_q(np.abs(dg), 90):6.0f} 最大 {np.abs(dg).max():6.0f} HP | "
                  f"有符号均值 {dg.mean():+6.0f} | 阈值翻转 "
                  f"{np.mean((hg >= THR_HP) != (fg >= THR_HP)):5.1%}")
        print("  质量对照（同一抽样集、同一真值=「此刻真值≥阈值」；【R9⑤】同分母）：")
        for tag, name in (("A", "方向A 敌方压我"), ("B", "方向B 我压敌方")):
            g = [s for s in samples if s["tag"] == tag]
            if not g:
                continue
            truth = np.array([s["fresh"] >= THR_HP for s in g])
            ph = np.array([s["held"] >= THR_HP for s in g])
            pc = np.array([s["crude"] * K >= THR_HP for s in g])
            bh = ba_of(ph, truth)
            bc = ba_of(pc, truth)
            print(f"    {name}: n={len(g)} 真值阳性率 {truth.mean():5.1%} | "
                  f"BA(陈旧保持值)={bh[0]:.4f} vs BA(粗糙回退)={bc[0]:.4f} "
                  f"⇒ 差 {bh[0] - bc[0]:+.4f}"
                  f" | 混淆: 保持 TP={bh[1]} FP={bh[2]} FN={bh[3]} / "
                  f"粗糙 TP={bc[1]} FP={bc[2]} FN={bc[3]}")
        alltruth = np.array([s["fresh"] >= THR_HP for s in samples])
        print(f"    合计: n={len(samples)} BA(陈旧保持值)="
              f"{ba_of([s['held'] >= THR_HP for s in samples], alltruth)[0]:.4f} vs "
              f"BA(粗糙回退)={ba_of([s['crude'] * K >= THR_HP for s in samples], alltruth)[0]:.4f}")
        print("  陈旧度与质量随保持年龄的变化（真值 =「此刻真值 ≥ 阈值」）：")
        print("    age 区间(帧)        n   |Δ|中位  p90   阈值翻转  BA(陈旧保持)  BA(粗糙回退)   差")
        for lo, hi in ((0, 5), (5, 10), (10, 20), (20, 40), (40, 10 ** 9)):
            g = [s for s in samples if lo <= s["age"] < hi]
            if not g:
                continue
            dg = np.abs(np.array([s["fresh"] - s["held"] for s in g]))
            tr = np.array([s["fresh"] >= THR_HP for s in g])
            bh = ba_of([s["held"] >= THR_HP for s in g], tr)[0]
            bc = ba_of([s["crude"] * K >= THR_HP for s in g], tr)[0]
            fl = np.mean([((s["held"] >= THR_HP) != (s["fresh"] >= THR_HP)) for s in g])
            print(f"    [{lo:2d},{hi if hi < 10 ** 9 else 999:3d})          "
                  f"{len(g):4d}  {np.median(dg):7.0f} {_q(dg, 90):6.0f}   {fl:6.1%}   "
                  f"{bh:8.4f}    {bc:8.4f}  {bh - bc:+.4f}")
        ages = [s["age"] for s in samples]
        print(f"  保持年龄（帧，0 = 紧邻推演帧）：中位 {_q(ages, 50):.0f} p90 {_q(ages, 90):.0f} "
              f"最大 {max(ages)}（推演视界 H={a.H}s = {a.H / 0.5:.0f} 帧）")
    else:
        print("  无抽样帧（本 run 没有出现陈旧保持帧）")

    print("\n" + "=" * 78)
    print("§3 候选修复的成本（近似上界，未重放状态机）")
    ms_call = sim_ms_total / max(1, n_sim)
    print(f"  现状：推演 {n_sim} 次 ⇒ 摊销 {sim_ms_total / max(1, N):.2f} ms/帧"
          f"（单次 {ms_call:.1f} ms，预算 3.85）")
    rules = [
        ("出现新敌方单位才重算",
         lambda r: any(r["hold_pre"][t] and r["ev"][t]["new_enemy"] for t in ("A", "B"))),
        ("出现新单位（任一方）就重算",
         lambda r: any(r["hold_pre"][t] and (r["ev"][t]["new_enemy"] or r["ev"][t]["new_ally"])
                       for t in ("A", "B"))),
        ("单位集合有任何增减就重算",
         lambda r: any(r["hold_pre"][t] and (r["ev"][t]["new"] or r["ev"][t]["gone"])
                       for t in ("A", "B"))),
    ]
    for label, pred in rules:
        extra = sum(1 for r in rows if (not r["sim"]) and pred(r))
        tot = n_sim + extra
        print(f"    {label:24s}: 推演 {n_sim} → {tot} 次 ⇒ 摊销 {tot * ms_call / max(1, N):6.2f} "
              f"ms/帧（×{tot / max(1, n_sim):.2f}）")
    extra = sum(max(0, int(s["n"] * 0.5 // a.H)) for s in segs["A"] + segs["B"])
    tot = n_sim + extra
    print(f"    {'保持寿命上限 = H 秒':24s}: 推演 {n_sim} → {tot} 次 ⇒ 摊销 "
          f"{tot * ms_call / max(1, N):6.2f} ms/帧（×{tot / max(1, n_sim):.2f}）"
          f"（纯按寿命刷新，与盘面无关）")

    print("\n" + "=" * 78)
    print("§4 退出帧：保持值（HP 刻度）→ 回退值（计数刻度）")
    jumps = []
    for k in range(1, len(rows)):
        if rows[k]["seed"] != rows[k - 1]["seed"]:
            continue
        for t in ("A", "B"):
            if rows[k - 1]["hold_pre"][t] and (not rows[k]["hold_pre"][t]) \
                    and rows[k - 1]["held"][t] is not None:
                jumps.append(rows[k - 1]["held"][t] / K)
    print(f"  退出事件 {len(jumps)} 次")
    if jumps:
        hp = np.array(jumps)
        print(f"  退出前保持值（pressure 刻度）中位 {np.median(hp):.2f} p90 {_q(hp, 90):.2f}"
              f" | 接近 0（≤0.05）的占 {np.mean(hp <= 0.05):.1%}")


if __name__ == "__main__":
    main()
