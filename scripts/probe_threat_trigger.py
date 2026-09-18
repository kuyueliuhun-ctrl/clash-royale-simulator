# -*- coding: utf-8 -*-
"""评估「己方半场无兵 且 对方半场的兵已到桥头 ⇒ 调一次精确塔伤」这个触发条件（只读）。

几何口径（全部取自引擎自身常量，不手抄）：
  · 河：`arena.RIVER_Y1=15.0` ~ `RIVER_Y2=16.0`（y∈[15,16]）
  · 桥：左 `2.0 <= x < 5.0`、右 `13.0 <= x < 16.0`（`arena.py:114-115`）
  · 左右半场：竞技场自身约定 `arena.py:157-170` 为 左 x∈[0,9)、右 x∈[9,18)，
    即**中轴在 x=9.0（cell 8 与 cell 9 之间）**。
    用户口径「半场要包含中轴右侧/左侧一格」⇒ 两半场**都含 cell 9**
    （左 = cell_x ≤ 9、右 = cell_x ≥ 9）⇒ 每支部队至少落在一个半场里，不会漏。

触发（用户提案）：
  存在一个半场 h，使得  我方在该半场无兵  AND  敌方在该半场有兵已到桥头
  ⇒ 该决策帧调用一次 `estimate_tower_threat`。

回答三件事：
  Q1 触发率 ⇒ 摊销成本 = 触发率 × 单次成本，是否落进预算（3.85 ms/帧）
  Q2 **召回**：引擎判定"真会掉塔血"的帧里，触发能覆盖多少
  Q3 **精度**：触发帧里有多少真有威胁（假警报率）；与零信息基线比 BA

用法（必须在 src/clasher_new 下运行）：
    cd src/clasher_new && PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe \
        ../../scripts/probe_threat_trigger.py --frames 400
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

#: 半场口径（cell 横坐标）
HALVES = {
    # 用户口径：两半场都含轴旁那一格（左含 cell 9、右含 cell 9）
    "user(左≤9,右≥9)": (lambda c: c <= 9, lambda c: c >= 9),
    # 竞技场自身口径：左 [0,9) 即 cell≤8；右 [9,18) 即 cell≥9
    "arena(左≤8,右≥9)": (lambda c: c <= 8, lambda c: c >= 9),
}
#: 桥头判定（敌方 y>16 从上方下来）
BRIDGE_Y1, BRIDGE_Y2 = 15.0, 16.0
LEFT_BRIDGE = (2.0, 5.0)
RIGHT_BRIDGE = (13.0, 16.0)


def on_bridge(x):
    return (LEFT_BRIDGE[0] <= x < LEFT_BRIDGE[1]) or \
           (RIGHT_BRIDGE[0] <= x < RIGHT_BRIDGE[1])


def cell_x(pos):
    return int(pos.x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/d1_long_100k/solo_main_100000.pt")
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--horizons", nargs="*", type=float, default=[3.0, 20.0])
    ap.add_argument("--bridge-band", type=float, default=1.5,
                    help="敌方 y ≤ RIVER_Y2 + band 视为已到桥头（默认 1.5 ⇒ y≤17.5）")
    a = ap.parse_args()

    import torch
    from rl.env_wrapper import RLEnv
    from rl.belief import BeliefInference
    from rl.belief_planner import BeliefPlanner
    from rl.follower import FollowerPolicy, BELIEF_DIM
    from rl.plan_space import PLAN_DIM
    from rl.train_follower import FollowerOpponent
    from threat_calc import estimate_tower_threat

    print("=" * 78)
    print("§1 逐决策帧记录（真实训练分布：镜像对手）")
    print(f"    桥头带 = 敌 y ≤ {BRIDGE_Y2 + a.bridge_band:.1f} 且 x 在桥列"
          f"（左 {LEFT_BRIDGE} / 右 {RIGHT_BRIDGE}）")

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

    rows = []
    costs = []
    hidden = None
    for i in range(a.frames):
        b = env.battle
        # —— 记录几何 ——
        mine, theirs = [], []
        for e in b.entities.values():
            if not e.is_alive or "Tower" in getattr(e, "name", ""):
                continue
            rec = (float(e.position.x), float(e.position.y),
                   int(getattr(e, "speed", 0) or 0) > 0)
            (mine if getattr(e, "player", 0) == 0 else theirs).append(rec)
        row = {"mine": mine, "theirs": theirs, "t": float(b.time)}
        for H in a.horizons:
            t0 = time.perf_counter()
            row[f"E{H:g}"] = float(estimate_tower_threat(b, 0, horizon=H)["total"])
            costs.append((time.perf_counter() - t0) * 1e3)
        rows.append(row)
        # —— 推进 ——
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
            hidden = None

    cost_ms = float(np.median(costs)) if costs else 0.0
    print(f"[rollout] {len(rows)} 决策帧 | 单次精确计算中位成本 = "
          f"{cost_ms:.1f} ms（H∈{a.horizons}）")

    # ---------------- 触发判定 ----------------
    def trigger(row, leftf, rightf, mode="col"):
        """mode: 'col'=要求 y 在河带且 x 落在桥列；'band'=只要求 y 在河带（不挑桥列）。"""
        def is_bridgehead(u):
            if u[1] > BRIDGE_Y2 + a.bridge_band:
                return False
            return on_bridge(u[0]) if mode == "col" else True

        for f in (leftf, rightf):
            my_in = [u for u in row["mine"] if f(int(u[0]))]
            en_bridge = [u for u in row["theirs"]
                         if f(int(u[0])) and is_bridgehead(u)]
            if not my_in and en_bridge:
                return True
        return False

    def held_series(t, K):
        """触发后保持 K 帧（部署口径：只在上沿真正调用，其余帧沿用旧值）。"""
        out = np.zeros(len(t), dtype=bool)
        left = 0
        for i, v in enumerate(t):
            if v:
                left = K
            if left > 0:
                out[i] = True
                left -= 1
        return out

    print("\n" + "=" * 78)
    print("§2 触发率与摊销成本（预算 3.85 ms/帧 = 训练帧 38.5 ms 的 10%）")
    print(f"    {'半场口径':>22s} {'触发帧数':>9s} {'触发率':>8s} "
          f"{'摊销(ms/帧)':>12s} {'判决':>10s}")
    trig = {}
    for mode in ("col", "band"):
        for name, (lf, rf) in HALVES.items():
            if name.startswith("arena") and mode == "band":
                continue          # 半场口径实测无非区分力，band 只跑 user 口径
            t = np.array([trigger(r, lf, rf, mode) for r in rows], dtype=bool)
            trig[f"{mode}|{name}"] = t
            amort = float(t.mean()) * cost_ms
            ok = "PASS" if amort <= 3.85 else "超预算"
            print(f"    {mode:>4s}|{name:>22s} {int(t.sum()):9d} {t.mean():8.1%} "
                  f"{amort:12.2f} {ok:>10s}")
    print("    — 边沿触发（仅在上升沿调用一次，其余帧保持旧值）—")
    HOLD = {}
    for key, t in trig.items():
        edge = t & ~np.concatenate([[False], t[:-1]])
        for K in (1, 4, 8, 16):
            h = held_series(t, K)
            HOLD[(key, K)] = h
        line = "  ".join(f"K={K}:{edge.mean():.1%}"
                         for K in (1, 4, 8, 16))
        print(f"    {key:>28s} 上升沿 {int(edge.sum()):3d} 帧 "
              f"({edge.mean():5.1%}) ⇒ 摊销 {edge.mean() * cost_ms:5.2f} ms/帧")
    print("    （K = 触发后保持帧数；调用次数只与上升沿有关，K 只影响覆盖）")

    # ---------------- 与引擎真值对照 ----------------
    print("\n" + "=" * 78)
    print("§3 与引擎真值对照（真值 = 引擎 E；触发=有威胁 的预测器）")
    for H in a.horizons:
        E = np.array([r[f"E{H:g}"] for r in rows])
        posE = E[E > 0]
        med = float(np.median(posE)) if len(posE) else 0.0
        print(f"\n  H={H:g}s：引擎「真会掉血」帧={int((E > 0).sum())}/{len(rows)} "
              f"({(E > 0).mean():.1%})；正威胁中位={med:.0f} HP")
        for (key, K), t in HOLD.items():
            if K not in (1, 8):
                continue
            for tname, truth in (("T1 E>0", E > 0),
                                 (f"T2 E≥{med:.0f}HP", E >= med)):
                tp = int((t & truth).sum()); fp = int((t & ~truth).sum())
                fn = int((~t & truth).sum()); tn = int((~t & ~truth).sum())
                rec = tp / (tp + fn) if (tp + fn) else float("nan")
                prec = tp / (tp + fp) if (tp + fp) else float("nan")
                tpr = rec
                tnr = tn / (tn + fp) if (tn + fp) else float("nan")
                ba = float(np.nanmean([tpr, tnr]))
                # 零信息基线
                base_on = np.ones(len(rows), bool); base_off = np.zeros(len(rows), bool)
                def _ba(pred):
                    a_ = (pred & truth).sum(); b_ = (~pred & truth).sum()
                    c_ = (~pred & ~truth).sum(); d_ = (pred & ~truth).sum()
                    t1 = a_ / (a_ + b_) if (a_ + b_) else np.nan
                    t2 = c_ / (c_ + d_) if (c_ + d_) else np.nan
                    return float(np.nanmean([t1, t2]))
                print(f"    {key:>28s} K={K:<2d} {tname:12s} 召回={rec:5.1%} "
                      f"精度={prec:5.1%} "
                      f"BA={ba:.4f} (基线 全开={_ba(base_on):.4f} "
                      f"全关={_ba(base_off):.4f}) | TP={tp} FP={fp} FN={fn}")

    # ---------------- 漏报的具体形态 ----------------
    print("\n" + "=" * 78)
    print("§4 漏报形态：触发关闭但引擎说有大威胁（H=20s，取威胁最大的几帧）")
    H = max(a.horizons)
    E = np.array([r[f"E{H:g}"] for r in rows])
    name = list(HALVES)[0]
    t = trig[[k for k in trig if k.startswith("col|user")][0]]
    idx = np.where(~t & (E > 0))[0]
    idx = idx[np.argsort(-E[idx])][:8]
    for i in idx:
        r = rows[i]
        print(f"    t={r['t']:6.1f}s E={E[i]:7.0f} HP | 我方单位 "
              f"{[(round(x,1), round(y,1)) for x, y, _ in r['mine']]} | 敌方 "
              f"{[(round(x,1), round(y,1)) for x, y, _ in r['theirs']]}")

    print("\n    读法：'召回'=引擎说有威胁的帧里触发覆盖了多少（漏报=永远算不到）；"
          "'精度'=触发帧里真有威胁的比例（假警报=白花钱）。")


if __name__ == "__main__":
    main()
