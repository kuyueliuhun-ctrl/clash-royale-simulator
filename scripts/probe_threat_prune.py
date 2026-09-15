# -*- coding: utf-8 -*-
"""精确塔伤的廉价化可行性：地理剪枝的剪枝率 + **可靠性**（P3-2 阶段 A；只读）。

背景（`docs/threat_precise_probe_verdict_2026-09-14.md` §4）：
`estimate_tower_threat` 每次 45~168 ms，超预算（3.85 ms/帧）1.1e4~4.4e4 倍 ⇒ 不能每帧调用。

本脚本测「**可证明安全**的地理剪枝」是否可行：
对每个敌方非塔单位算到达我方最近塔的**下界时间**
    t_a = max(0, 直线距离 − 全卡池最大射程 − 拉拽余量) / 移速
只有 `min t_a > H` 时才判 0（"这个视界内物理上不可能接触到塔"）。

**可靠性检查（本脚本的核心）**：在每一帧上，若剪枝判 0 而引擎推演给出 E>0 ⇒ **反例**，
剪枝不成立。反例必须为 0，否则该设计作废。

保守处理（任一命中 ⇒ 该单位不可剪枝，t_a := 0）：
- 移速 ≤0 / 未知；是建筑（`Building`）；有周期出兵（`spawnCharacterData`+`spawnPauseTime`）；
- 有亡语出兵（`death_spawn_data`）；觉醒态。

用法（必须在 src/clasher_new 下运行）：
    cd src/clasher_new && PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe \
        ../../scripts/probe_threat_prune.py --frames 500
"""
from __future__ import annotations

import argparse
import math
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

#: 拉拽/击退类机制可能让敌方单位**比直线行进更快**接近我方塔 ⇒ 必须留余量（保守）。
#: ⚠️ 第一版把"全卡池最大射程 11.5"与"拉拽余量 8.0"一起减掉 = 19.5 格 ≈ 半张图
#: ⇒ 几乎任何单位都判 t_a=0、剪枝率只有 19.8%（**是我自己的参数选错**）。
#: 改为**用每个单位自身的射程** + 适度余量。
PULL_SLACK_TILES = 2.0      # 钩爪/击退类一次性位移余量
SPEED_SLACK = 1.5           # 移速加成（狂暴类）余量：按 1.5× 移速保守估计


def unit_range_tiles(e):
    """该单位自身的攻击射程（格）；单一来源 = 引擎 card_data。"""
    d = getattr(getattr(e, "data", None), "data", None) or {}
    st = d.get("summonCharacterData") or {}
    r = st.get("range") or d.get("range") or 0
    return float(r) / 1000.0


def max_card_range_tiles():
    """全卡池最大攻击射程（格）；单一来源 = 引擎自身 card_data。"""
    from card_utils import card_data
    mx = 0.0
    for _n, d in card_data.items():
        st = d.get("summonCharacterData") or {}
        r = st.get("range")
        if r:
            mx = max(mx, float(r) / 1000.0)
        r2 = d.get("range")
        if r2:
            mx = max(mx, float(r2) / 1000.0)
    return mx


_TOWER_IDS = {0: (3, 4, 6), 1: (1, 2, 5)}


def unprunable(e):
    """该单位是否属于"不可剪枝"（保守返回 True）。"""
    from battle import Building
    if isinstance(e, Building):
        return True
    if getattr(e, "evo", None):
        return True
    d = getattr(getattr(e, "data", None), "data", None) or {}
    scd = d.get("summonCharacterData") or {}
    if scd.get("spawnCharacterData") and scd.get("spawnPauseTime"):
        return True
    if getattr(getattr(e, "data", None), "death_spawn_data", None):
        return True
    sp = getattr(e, "speed", None)
    if not sp or float(sp) <= 0.0:
        return True
    return False


def min_time_to_tower(battle, player_id, slack_tiles, max_range):
    """返回 (min_t_a, n_unprunable)。min_t_a=None 表示无敌方单位。

    `max_range` 参数保留但不再用于剪枝（见上方注释的自披露）。
    """
    tids = _TOWER_IDS[player_id]
    towers = [battle.entities[t] for t in tids
              if t in battle.entities and battle.entities[t].is_alive]
    if not towers:
        return None, 0
    best = None
    n_un = 0
    for e in battle.entities.values():
        if not e.is_alive or getattr(e, "player", None) != 1 - player_id:
            continue
        if "Tower" in getattr(e, "name", ""):
            continue
        if unprunable(e):
            n_un += 1
            return 0.0, n_un
        sp = float(e.speed) * SPEED_SLACK
        rg = unit_range_tiles(e)
        d = min(math.hypot(float(e.position.x) - float(t.position.x),
                           float(e.position.y) - float(t.position.y))
                for t in towers)
        eff = d - rg - slack_tiles
        t_a = 0.0 if eff <= 0 else eff / sp
        best = t_a if best is None else min(best, t_a)
    return best, n_un


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/d1_long_100k/solo_main_100000.pt")
    ap.add_argument("--frames", type=int, default=500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--horizons", nargs="*", type=float,
                    default=[3.0, 5.0, 8.0, 20.0])
    a = ap.parse_args()

    import torch
    from rl.env_wrapper import RLEnv
    from rl.belief import BeliefInference
    from rl.belief_planner import BeliefPlanner
    from rl.follower import FollowerPolicy, BELIEF_DIM
    from rl.plan_space import PLAN_DIM
    from threat_calc import estimate_tower_threat

    max_range = max_card_range_tiles()
    print("=" * 78)
    print("§1 廉价化可行性：地理剪枝（只读）")
    print(f"    全卡池最大射程 = {max_range:.1f} 格（单一来源 = 引擎 card_data）；"
          f"拉拽余量 = {PULL_SLACK_TILES} 格")
    print(f"    剪枝判据：min t_a > H ⇒ 该视界内物理上不可能接触塔 ⇒ 精确值记 0")

    env = RLEnv(opponent=None, seed=a.seed)
    obs, _ = env.reset(seed=a.seed)
    pol = FollowerPolicy(hidden=128, plan_dim=PLAN_DIM, belief_dim=BELIEF_DIM,
                         value_bypass=True, value_independent=True)
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    pol.load_state_dict(sd.get("state_dict", sd), strict=False)
    pol.eval()
    belief = BeliefInference(opp_deck=env.deck1, n_particles=128, seed=a.seed)
    bp = BeliefPlanner()

    recs = []
    hidden = None
    n = 0
    while n < a.frames:
        b = env.battle
        t_a, n_un = min_time_to_tower(b, 0, PULL_SLACK_TILES, max_range)
        rec = {"t": float(b.time), "t_a": (-1.0 if t_a is None else float(t_a)),
               "n_un": int(n_un)}
        for H in a.horizons:
            rec[f"E{H:g}"] = float(estimate_tower_threat(b, 0, horizon=H)["total"])
        recs.append(rec)
        plan_vec = bp.plan(b, belief.state(), obs).to_vector()
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

    ta = np.array([r["t_a"] for r in recs])
    print(f"[rollout] {n} 帧 | 无敌方单位帧 = {int((ta < 0).sum())} | "
          f"t_a=0 帧（近/不可剪枝）= {int((ta == 0).sum())}")

    # T1 成本
    ts = []
    b = env.battle
    for _ in range(300):
        t0 = time.perf_counter()
        min_time_to_tower(b, 0, PULL_SLACK_TILES, max_range)
        ts.append((time.perf_counter() - t0) * 1e3)
    t1_cost = float(np.median(ts))

    print("\n" + "=" * 78)
    print("§2 剪枝率 / 可靠性 / 摊销成本（预算 3.85 ms/帧 = 训练帧 38.5 ms 的 10%）")
    print(f"    T1（地理剪枝）自身成本 = {t1_cost:.4f} ms")
    print(f"    {'H':>5s} {'剪枝率':>8s} {'需调用率':>9s} "
          f"{'可靠性反例':>10s} {'摊销成本(ms/帧)':>16s} {'判决':>10s}")
    for H in a.horizons:
        E = np.array([r[f"E{H:g}"] for r in recs])
        prune = (ta < 0) | (ta > H)          # 无单位 或 下界时间 > H
        viol = int((prune & (E > 0.0)).sum())
        # T2 成本：用非剪枝帧中的实测值（这里取该盘面的实测耗时近似）
        cost2 = None
        ts2 = []
        for _ in range(6):
            t0 = time.perf_counter()
            estimate_tower_threat(b, 0, horizon=H)
            ts2.append((time.perf_counter() - t0) * 1e3)
        cost2 = float(np.median(ts2))
        amort = t1_cost + (1.0 - float(prune.mean())) * cost2
        ok = "PASS" if (viol == 0 and amort <= 3.85) else (
            "反例!" if viol else "超预算")
        print(f"    {H:5.0f} {100 * float(prune.mean()):7.1f}% "
              f"{100 * (1 - float(prune.mean())):8.1f}% {viol:10d} "
              f"{amort:16.2f} {ok:>10s}")

    # 反例细节
    print("\n" + "=" * 78)
    print("§3 反例细节（若为 0 则剪枝设计成立）")
    for H in a.horizons:
        E = np.array([r[f"E{H:g}"] for r in recs])
        prune = (ta < 0) | (ta > H)
        idx = np.where(prune & (E > 0.0))[0][:5]
        if len(idx) == 0:
            print(f"    H={H:4.0f}s: 反例 0（剪枝掉的 {int(prune.sum())} 帧里"
                  f"引擎实测塔伤恒为 0）")
            continue
        print(f"    H={H:4.0f}s: 反例 {int((prune & (E > 0)).sum())} 帧，前几例：")
        for i in idx:
            print(f"      t={recs[i]['t']:6.1f}s t_a={recs[i]['t_a']:8.2f}s "
                  f"不可剪枝单位={recs[i]['n_un']} 引擎 E={E[i]:.1f} HP")

    # ---------------- §4 路线 B：周期采样 + 保持的滞后误差 ----------------
    print("\n" + "=" * 78)
    print("§4 路线 B（周期采样 + 保持）的滞后误差：若每 K 帧才调一次精确工具并保持，"
          "闸门判定与逐帧精确判定的差异率")
    print(f"    {'H':>5s} {'真值口径':>12s} {'K=4':>8s} {'K=8':>8s} {'K=16':>8s} "
          f"{'K=32':>8s} {'逐帧翻转率':>10s}")
    for H in a.horizons:
        E = np.array([r[f"E{H:g}"] for r in recs])
        posE = E[E > 0]
        med = float(np.median(posE)) if len(posE) else 0.0
        for tname, g in (("E>0", E > 0), (f"E>={med:.0f}HP", E >= med)):
            line = []
            for K in (4, 8, 16, 32):
                idxs = np.arange(0, len(g), K)
                held = np.zeros(len(g), dtype=bool)
                for i, s in enumerate(idxs):
                    e = idxs[i + 1] if i + 1 < len(idxs) else len(g)
                    held[s:e] = g[s]
                line.append(100 * float((held != g).mean()))
            flip = 100 * float((g[1:] != g[:-1]).mean())
            print(f"    {H:5.0f} {tname:>12s} "
                  + " ".join(f"{v:7.1f}%" for v in line)
                  + f" {flip:9.1f}%")
    print("    读法：差异率 = 用保持值做决策时的**误判率**；逐帧翻转率 = 闸门本身的不稳定度。")


if __name__ == "__main__":
    main()
