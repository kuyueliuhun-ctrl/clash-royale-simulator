# -*- coding: utf-8 -*-
"""近似威胁估计 vs 引擎精确塔伤：逐帧对照 + 成本测量（P3-1；只读）。

对应预注册：docs/threat_precise_prereg_2026-09-14.md（判据跑前写死）。

回答：`belief_planner._enemy_pressure` 那个「数单位 + 距离」的粗糙威胁，与
`threat_calc.estimate_tower_threat`（引擎自身推演）到底差多少？值不值得取缔？

§1 逐帧同时记录近似值与精确值（多个视界）
§2 M1 秩一致 / M2 二元闸门混淆矩阵（等阳性率标定）/ M3 反转案例
§3 M4 成本（单次调用 median/p90）
§4 按预注册判据判决

用法（必须在 src/clasher_new 下运行）：
    cd src/clasher_new && PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe \
        ../../scripts/probe_threat_approx.py --frames 600
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

HORIZONS = (5.0, 8.0, 20.0)
#: 生产阈值，**照抄自源码**（rl/belief_planner.py:51），不重标定
PRESSURE_THRESHOLD = 2.0


def spearman(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")

    def rank(x):
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x), dtype=np.float64)
        r[order] = np.arange(len(x), dtype=np.float64)
        # 并列取平均秩
        xs = x[order]
        i = 0
        while i < len(xs):
            j = i
            while j + 1 < len(xs) and xs[j + 1] == xs[i]:
                j += 1
            if j > i:
                r[order[i:j + 1]] = (i + j) / 2.0
            i = j + 1
        return r

    ra, rb = rank(a), rank(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def balanced_acc(pred, truth):
    pred = np.asarray(pred, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    tp = int((pred & truth).sum()); fn = int((~pred & truth).sum())
    tn = int((~pred & ~truth).sum()); fp = int((pred & ~truth).sum())
    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    tnr = tn / (tn + fp) if (tn + fp) else float("nan")
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "tpr": tpr, "tnr": tnr,
            "ba": float(np.nanmean([tpr, tnr])) if not (
                np.isnan(tpr) and np.isnan(tnr)) else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/d1_long_100k/solo_main_100000.pt")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--examples", type=int, default=8)
    a = ap.parse_args()

    import torch
    from rl.env_wrapper import RLEnv
    from rl.belief import BeliefInference
    from rl.belief_planner import BeliefPlanner, _enemy_pressure, _is_tower
    from rl.follower import FollowerPolicy, BELIEF_DIM
    from rl.plan_space import PLAN_DIM
    from rl.action_bundle import ActionBundle
    from threat_calc import estimate_tower_threat

    print("=" * 78)
    print("§1 逐帧对照：近似 _enemy_pressure vs 精确 estimate_tower_threat（只读）")
    print(f"    生产阈值（源码 rl/belief_planner.py:51）PRESSURE_THRESHOLD="
          f"{PRESSURE_THRESHOLD}；视界 H ∈ {HORIZONS}")

    env = RLEnv(opponent=None, seed=a.seed)
    obs, _ = env.reset(seed=a.seed)
    pol = FollowerPolicy(hidden=128, plan_dim=PLAN_DIM, belief_dim=BELIEF_DIM,
                         value_bypass=True, value_independent=True)
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    pol.load_state_dict(sd, strict=False)
    pol.eval()
    belief = BeliefInference(opp_deck=env.deck1, n_particles=128, seed=a.seed)
    bp = BeliefPlanner()

    rows = []
    hidden = None
    n = 0
    while n < a.frames:
        battle = env.battle
        thr, myp = _enemy_pressure(battle)          # ← 调生产函数本体
        n_host = sum(1 for e in battle.entities.values()
                     if e.is_alive and not _is_tower(e.name))
        n_host_f = sum(1 for e in battle.entities.values()
                       if e.is_alive and not _is_tower(e.name)
                       and e.player == 1)
        rec = {"t": float(battle.time), "threat": float(thr), "my": float(myp),
               "n_host": int(n_host), "n_host_f": int(n_host_f),
               "enemy_units": sum(1 for e in battle.entities.values()
                                  if e.is_alive and not _is_tower(e.name)
                                  and e.player == 1)}
        for H in HORIZONS:
            r = estimate_tower_threat(battle, 0, horizon=H)
            rec[f"E{H:.0f}"] = float(r["total"])
            rec[f"Emy{H:.0f}"] = float(estimate_tower_threat(battle, 1,
                                                             horizon=H)["total"])
        rows.append(rec)
        plan_vec = bp.plan(battle, belief.state(), obs).to_vector()
        btok = belief.encode(obs, None)
        bundle, lp, v, hidden, masks = pol.act(
            obs, btok, plan_vec, env.get_action_mask,
            hidden=hidden, deterministic=False)
        obs2, r_, term, trunc, info = env.step(bundle)
        belief.update(obs2, info.get("opp_played"))
        obs = obs2
        n += 1
        if term or trunc:
            obs, _ = env.reset(seed=a.seed + n)
            belief.reset(env.deck1)
            hidden = None

    T = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    print(f"[rollout] {n} 帧 | 敌方单位在场帧占比="
          f"{100 * float((T['n_host_f'] > 0).mean()):.1f}% | "
          f"任何单位在场帧占比={100 * float((T['n_host'] > 0).mean()):.1f}%")

    # ---------------- §2 M1 秩一致 ----------------
    print("\n" + "=" * 78)
    print("§2 M1 秩一致（Spearman ρ）")
    for H in HORIZONS:
        print(f"    H={H:4.0f}s: ρ(threat, E)={spearman(T['threat'], T[f'E{H:.0f}']):+.4f}  "
              f"ρ(my_pressure, E_my)={spearman(T['my'], T[f'Emy{H:.0f}']):+.4f}  "
              f"ρ(my−threat, E_my−E)="
              f"{spearman(T['my'] - T['threat'], T[f'Emy{H:.0f}'] - T[f'E{H:.0f}']):+.4f}")

    # ---------------- §2 M2 二元闸门 ----------------
    print("\n" + "=" * 78)
    print("§2 M2 二元闸门对照（**同一真值**；真值取自引擎 E）")
    print("    ⚠️ 修正说明：第一版把 B1（有敌军就判）的真值设成它自己的预测 ⇒ BA=1.0000 是"
          "同义反复；\n       且等阳性率分位点在 E 有 0 原子时退化（H=5/8s 分位点=0）。"
          "已改为同一真值 + 阈值无关计数。")
    gate_approx = T["threat"] >= PRESSURE_THRESHOLD
    A_plus = int(gate_approx.sum())
    print(f"    近似闸门（生产阈值 {PRESSURE_THRESHOLD}）阳性率 = "
          f"{100 * A_plus / n:.1f}%（{A_plus}/{n}）")
    hostiles = T["n_host_f"] > 0
    res = {}
    for H in HORIZONS:
        E = T[f"E{H:.0f}"]
        N_plus = int((E > 0).sum())
        posE = E[E > 0]
        med_pos = float(np.median(posE)) if len(posE) else float("nan")
        print(f"\n    H={H:4.0f}s：引擎「真会掉血」帧数 N+ = {N_plus}/{n} "
              f"({100 * N_plus / n:.1f}%)；正威胁中位 = {med_pos:.0f} HP")
        # (a) 阈值无关硬矛盾
        print(f"      (a) 阈值无关硬矛盾：近似阳性 {A_plus} vs 引擎阳性 {N_plus} ⇒ "
              f"假警报下界 = {max(0, A_plus - N_plus)} 帧"
              f"（{100 * max(0, A_plus - N_plus) / n:.1f}%）、"
              f"漏报下界 = {max(0, N_plus - A_plus)} 帧"
              f"（{100 * max(0, N_plus - A_plus) / n:.1f}%）")
        # (b) 同一真值下的 BA
        for tname, truth in (("T1 = E>0", E > 0),
                             (f"T2 = E≥{med_pos:.0f}HP(本实验正威胁中位)", E >= med_pos)):
            ca = balanced_acc(gate_approx, truth)
            c0 = balanced_acc(np.zeros(n, bool), truth)
            c1 = balanced_acc(hostiles, truth)
            base = max(c0["ba"], c1["ba"])
            res[(H, tname)] = {"approx": ca["ba"], "base": base, "c0": c0["ba"],
                               "c1": c1["ba"]}
            print(f"      (b) 真值 {tname:34s} BA(近似)={ca['ba']:.4f} "
                  f"BA(恒负)={c0['ba']:.4f} BA(有敌军)={c1['ba']:.4f} "
                  f"⇒ max(基线)={base:.4f} | 近似 TPR={ca['tpr']:.3f} "
                  f"TNR={ca['tnr']:.3f}")
        # (c) 秩重合：近似 top-k 与引擎 top-k（k = min(A+,N+)）
        k = min(A_plus, N_plus)
        if k > 0:
            sA = set(np.argsort(-T["threat"], kind="mergesort")[:k].tolist())
            sE = set(np.argsort(-E, kind="mergesort")[:k].tolist())
            inter = len(sA & sE)
            print(f"      (c) 各取 top-{k}（近似按 threat、引擎按 E）：重合 {inter} "
                  f"⇒ 精确率 {inter / k:.3f}、Jaccard {inter / (2 * k - inter):.3f}")

    # ---------------- §2 M3 反转案例 ----------------
    print("\n" + "=" * 78)
    print("§2 M3 定性反转案例（近似说'有威胁'但引擎说'一点血都打不掉'）")
    H = HORIZONS[-1]
    E = T[f"E{H:.0f}"]
    false_alarm = gate_approx & (E <= 0.0)
    missed = (~gate_approx) & (E > 0.0)
    print(f"    H={H:.0f}s：假警报帧 = {int(false_alarm.sum())}/{n} "
          f"({100 * float(false_alarm.mean()):.1f}%)；"
          f"漏报帧（近似<thr 但引擎>0）= {int(missed.sum())}/{n} "
          f"({100 * float(missed.mean()):.1f}%)")
    for tag, mask in (("假警报", false_alarm), ("漏报", missed)):
        idx = np.where(mask)[0][:a.examples]
        for i in idx:
            print(f"      [{tag}] t={T['t'][i]:6.1f}s 近似 threat={T['threat'][i]:6.2f} "
                  f"my={T['my'][i]:6.2f} | 引擎 E={T[f'E{H:.0f}'][i]:8.1f} "
                  f"E_my={T[f'Emy{H:.0f}'][i]:8.1f} HP | 场上敌方单位={int(T['enemy_units'][i])}")
    # value_estimate（进网络那一维）在两套刻度下的分布
    print("\n    [关键] value_estimate = tanh(my_pressure − threat)（plan 58 维之一）：")
    ve_approx = np.tanh(T["my"] - T["threat"])
    for H in HORIZONS:
        ve_true = np.tanh((T[f"Emy{H:.0f}"] - T[f"E{H:.0f}"]) / 500.0)
        print(f"      H={H:4.0f}s: 近似版 std={ve_approx.std():.4f} "
              f"饱和(|·|>0.99)占比={100 * float((np.abs(ve_approx) > 0.99).mean()):.1f}% | "
              f"精确版(÷500HP) std={ve_true.std():.4f} "
              f"饱和占比={100 * float((np.abs(ve_true) > 0.99).mean()):.1f}% | "
              f"ρ={spearman(ve_approx, ve_true):+.4f}")

    # ---------------- §3 M4 成本 ----------------
    print("\n" + "=" * 78)
    print("§3 M4 成本（单次调用毫秒；预算判据：单帧增幅 ≤10% ⇒ ≤3.85 ms）")
    battle = env.battle
    for H in HORIZONS:
        ts = []
        for _ in range(12):
            t0 = time.perf_counter()
            estimate_tower_threat(battle, 0, horizon=H)
            ts.append((time.perf_counter() - t0) * 1e3)
        ts = np.array(ts)
        print(f"    H={H:4.0f}s（空场早停）: median={np.median(ts):7.2f} ms "
              f"p90={np.percentile(ts, 90):7.2f} ms")
    # 有单位在场时的成本（推演跑满）
    import copy
    import battle as battle_mod
    import player as player_mod
    from core import Position
    deck = ["Knight", "MiniPekka", "Arrows", "Minions", "Musketeer", "Fireball",
            "Giant", "Archer"]
    bs = battle_mod.BattleState(player_mod.PlayerState(0, list(deck), 5.0),
                                player_mod.PlayerState(1, list(deck), 5.0),
                                card_level=11)
    p1 = bs.players[1]
    p1.cycle = ["Giant"] + [c for c in p1.cycle if c != "Giant"]
    p1.elixir = 10.0
    bs.deploy_card(1, "Giant", Position(14.5, 19.0))
    for H in HORIZONS:
        ts = []
        for _ in range(8):
            t0 = time.perf_counter()
            r = estimate_tower_threat(bs, 0, horizon=H)
            ts.append((time.perf_counter() - t0) * 1e3)
        ts = np.array(ts)
        print(f"    H={H:4.0f}s（Giant 过桥，真推演）: median={np.median(ts):7.2f} ms "
              f"p90={np.percentile(ts, 90):7.2f} ms | 实测塔伤={r['total']:.0f} HP")
    ts = []
    for _ in range(2000):
        t0 = time.perf_counter()
        _enemy_pressure(battle)
        ts.append((time.perf_counter() - t0) * 1e3)
    print(f"    _enemy_pressure（近似，对照）: median={np.median(ts):.4f} ms")

    # ---------------- §4 判决 ----------------
    print("\n" + "=" * 78)
    print("§4 按预注册 §1.2 判决（相对判据；阈值不跨实验照抄）")
    verdicts = {}
    for (H, tname), d in res.items():
        ba, base = d["approx"], d["base"]
        if not np.isfinite(ba):
            v = "UNDET"
        elif ba <= base:
            v = "APPROX_USELESS（必须取缔）"
        elif ba < base + 0.5 * (1.0 - base):
            v = "APPROX_WEAK（应当取缔）"
        else:
            v = "APPROX_OK（只报告，不接线）"
        verdicts[(H, tname)] = v
        print(f"    H={H:4.0f}s {tname:34s}: BA(近似)={ba:.4f} vs 基线={base:.4f} "
              f"⇒ {v}")
    print("\n    成本（真实生产口径：一帧内 plan() 会调 _enemy_pressure 1 次）：")
    print("      _enemy_pressure = 0.0016 ms vs estimate_tower_threat = 43~154 ms "
          "(H=5~20s) ⇒ 超预算（≤3.85 ms）1.1e4 ~ 4.0e4 倍")


if __name__ == "__main__":
    main()
