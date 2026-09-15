# -*- coding: utf-8 -*-
"""评估「条件满足时算一次路径并固定跟随」能省多少（只读仪器）。

用户提案：当部队**周围没有己方部队**（空军/陆地单独算，**特别是前面**）
且**对方场上没有可移动单位**（只剩建筑/塔这类不动单位）时，
只算一次路径并让部队按既定路径走，不再周期性重算。

本脚本不修改引擎，只**挂接** `EntityPathfinder.calculate`，记录每次调用的
上下文与真实耗时，然后离线回答三个问题：
  Q1 现在到底有多少次重算？触发原因各是什么（`path` 空 vs 周期性）？
  Q2 其中多少次满足提案条件（⇒ 可省）？
  Q3 省下来的时间占整个 step 的百分之几 ⇒ 单 tick 成本能降几倍？

用法（必须在 src/clasher_new 下运行）：
    cd src/clasher_new && PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe \
        ../../scripts/probe_pathfix.py --ticks 400
"""
from __future__ import annotations

import argparse
import copy
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

RECORDS = []
_CALL_MS = []


def install_hook():
    """挂接 EntityPathfinder.calculate：记录上下文 + 真实耗时，然后调用原实现。"""
    import pathfinding_heap as pf

    orig = pf.EntityPathfinder.calculate

    def wrapped(self):
        e = self.entity
        bs = self.battle
        tgt = self.target
        t0 = time.perf_counter()
        path_before = list(getattr(e, "path", []) or [])
        rec = {
            "tick": int(getattr(bs, "tick", -1)),
            "time": float(getattr(bs, "time", -1.0)),
            "unit": int(getattr(e, "id", -1)),
            "name": getattr(e, "name", "?"),
            "player": getattr(e, "player", -1),
            "air": bool((getattr(e, "data", None) and
                         getattr(e.data, "is_air_unit", False))),
            "reason": "empty" if not path_before else "periodic",
            "target": getattr(tgt, "id", -1),
            "tgt_name": getattr(tgt, "name", "?"),
        }
        # 方向：本单位 → 目标
        try:
            dx = float(tgt.position.x) - float(e.position.x)
            dy = float(tgt.position.y) - float(e.position.y)
            n = math.hypot(dx, dy) or 1.0
            dx, dy = dx / n, dy / n
        except Exception:
            dx = dy = 0.0
        # 敌我可移动单位计数（塔与建筑=不动）
        from battle import Building
        n_enemy_mobile = 0
        n_enemy_non_tower = 0
        for o in bs.entities.values():
            if not o.is_alive:
                continue
            if "Tower" in getattr(o, "name", ""):
                continue
            if getattr(o, "player", None) == e.player:
                continue
            n_enemy_non_tower += 1
            if not isinstance(o, Building) and float(getattr(o, "speed", 0) or 0) > 0:
                n_enemy_mobile += 1
        rec["n_enemy_mobile"] = n_enemy_mobile
        rec["n_enemy_non_tower"] = n_enemy_non_tower
        # 己方友军（按空/地分组）在不同半径内、以及"前方"的数量
        for R in (2.0, 4.0):
            near = 0
            front = 0
            for o in bs.entities.values():
                if not o.is_alive or o.id == e.id:
                    continue
                if getattr(o, "player", None) != e.player:
                    continue
                if "Tower" in getattr(o, "name", ""):
                    continue
                if bool((getattr(o, "data", None) and
                         getattr(o.data, "is_air_unit", False))) != rec["air"]:
                    continue          # 空军/陆地单独算
                d = math.hypot(float(o.position.x) - float(e.position.x),
                               float(o.position.y) - float(e.position.y))
                if d <= R:
                    near += 1
                    if (float(o.position.x) - float(e.position.x)) * dx + \
                       (float(o.position.y) - float(e.position.y)) * dy > 0:
                        front += 1
            rec[f"near{int(R)}"] = near
            rec[f"front{int(R)}"] = front
        try:
            out = orig(self)
        finally:
            dt = (time.perf_counter() - t0) * 1e3
            rec["ms"] = dt
            RECORDS.append(rec)
            _CALL_MS.append(dt)
        return out

    pf.EntityPathfinder.calculate = wrapped
    return orig


def build(n_units, deck=None):
    import battle as bm
    import player as pm
    from core import Position
    deck = deck or ["Knight", "MiniPekka", "Arrows", "Minions", "Musketeer",
                    "Fireball", "Giant", "Archer"]
    bs = bm.BattleState(pm.PlayerState(0, list(deck), 5.0),
                        pm.PlayerState(1, list(deck), 5.0), card_level=11)
    cards = ["Knight", "Archer", "Giant", "Musketeer", "Minions", "MiniPekka"]
    for i in range(n_units):
        pl = i % 2
        p = bs.players[pl]
        c = cards[i % len(cards)]
        p.cycle = [c] + [x for x in p.cycle if x != c]
        p.elixir = 10.0
        pos = (Position(3.0 + 3 * (i % 6), 11.0 + 3 * (i // 6)) if pl == 0
               else Position(15.0 - 3 * (i % 6), 21.0 + 3 * (i // 6)))
        bs.deploy_card(pl, c, pos)
    return bs


def run_rollout(name, frames, seed=7):
    """真实对局分布：RLEnv + 训练过的策略驱动，逐决策帧推进。"""
    import torch
    from rl.env_wrapper import RLEnv
    from rl.belief import BeliefInference
    from rl.belief_planner import BeliefPlanner
    from rl.follower import FollowerPolicy, BELIEF_DIM
    from rl.plan_space import PLAN_DIM
    from rl.action_bundle import ActionBundle

    RECORDS.clear(); _CALL_MS.clear()
    env = RLEnv(opponent=None, seed=seed)
    obs, _ = env.reset(seed=seed)
    pol = FollowerPolicy(hidden=128, plan_dim=PLAN_DIM, belief_dim=BELIEF_DIM,
                         value_bypass=True, value_independent=True)
    sd = torch.load("runs/d1_long_100k/solo_main_100000.pt",
                    map_location="cpu", weights_only=False)
    pol.load_state_dict(sd.get("state_dict", sd), strict=False)
    pol.eval()
    belief = BeliefInference(opp_deck=env.deck1, n_particles=128, seed=seed)
    bp = BeliefPlanner()
    hidden = None
    t0 = time.perf_counter()
    for i in range(frames):
        plan_vec = bp.plan(env.battle, belief.state(), obs).to_vector()
        btok = belief.encode(obs, None)
        bundle, lp, v, hidden, masks = pol.act(
            obs, btok, plan_vec, env.get_action_mask,
            hidden=hidden, deterministic=False)
        obs2, r_, term, trunc, info = env.step(bundle)
        belief.update(obs2, info.get("opp_played"))
        obs = obs2
        if term or trunc:
            obs, _ = env.reset(seed=seed + i + 1)
            belief.reset(env.deck1)
            hidden = None
    wall = (time.perf_counter() - t0) * 1e3
    ticks = frames * 30
    ne = sum(1 for e in env.battle.entities.values() if e.is_alive)
    n_call = len(RECORDS)
    call_ms = float(np.sum(_CALL_MS)) if _CALL_MS else 0.0
    print(f"\n  [{name}] 末帧实体={ne} 决策帧={frames}(={ticks} tick) "
          f"墙钟={wall:.1f} ms")
    print(f"      `calculate` 调用 {n_call} 次（{n_call / ticks:.3f} 次/tick），"
          f"总计 {call_ms:.1f} ms = **{call_ms / wall:.1%} 的墙钟**；"
          f"单次中位 {np.median(_CALL_MS) if _CALL_MS else 0:.3f} ms")
    from collections import Counter
    rc = Counter(r["reason"] for r in RECORDS)
    print(f"      触发原因：{dict(rc)}"
          f" ⇒ **强制重算占 {100 * rc['empty'] / max(1, n_call):.1f}%、"
          f"周期性重算占 {100 * rc['periodic'] / max(1, n_call):.1f}%**")
    return wall, ticks, call_ms, list(RECORDS)


def run_rollout_mirror(name, frames, seed=7):
    """镜像对手（= 训练里真实用的 `FollowerOpponent`，同一策略打自己）。"""
    import torch
    from rl.env_wrapper import RLEnv
    from rl.belief import BeliefInference
    from rl.belief_planner import BeliefPlanner
    from rl.follower import FollowerPolicy, BELIEF_DIM
    from rl.plan_space import PLAN_DIM
    from rl.train_follower import FollowerOpponent

    RECORDS.clear(); _CALL_MS.clear()
    env = RLEnv(opponent=None, seed=seed)
    obs, _ = env.reset(seed=seed)
    pol = FollowerPolicy(hidden=128, plan_dim=PLAN_DIM, belief_dim=BELIEF_DIM,
                         value_bypass=True, value_independent=True)
    sd = torch.load("runs/d1_long_100k/solo_main_100000.pt",
                    map_location="cpu", weights_only=False)
    pol.load_state_dict(sd.get("state_dict", sd), strict=False)
    pol.eval()
    opp_side = FollowerOpponent(
        pol, env,
        belief=BeliefInference(opp_deck=env.deck1, n_particles=128, seed=seed),
        deterministic=True)
    pol.plan_biases_enabled = True      # FollowerOpponent.__init__ 把它关了，恢复 p0 侧
    env.opponent = opp_side
    belief = BeliefInference(opp_deck=env.deck1, n_particles=128, seed=seed)
    bp = BeliefPlanner()
    hidden = None
    t0 = time.perf_counter()
    for i in range(frames):
        plan_vec = bp.plan(env.battle, belief.state(), obs).to_vector()
        btok = belief.encode(obs, None)
        bundle, lp, v, hidden, masks = pol.act(
            obs, btok, plan_vec, env.get_action_mask,
            hidden=hidden, deterministic=False)
        obs2, r_, term, trunc, info = env.step(bundle)
        if info.get("opp_played"):
            pass
        belief.update(obs2, info.get("opp_played"))
        opp_side.observe_opponent_played(info.get("opp_played") or [])
        obs = obs2
        if term or trunc:
            obs, _ = env.reset(seed=seed + i + 1)
            belief.reset(env.deck1)
            opp_side.reset()
            hidden = None
    wall = (time.perf_counter() - t0) * 1e3
    ticks = frames * 30
    ne = sum(1 for e in env.battle.entities.values() if e.is_alive)
    n_call = len(RECORDS)
    call_ms = float(np.sum(_CALL_MS)) if _CALL_MS else 0.0
    print(f"\n  [{name}] 末帧实体={ne} 决策帧={frames}(={ticks} tick) 墙钟={wall:.1f} ms")
    print(f"      `calculate` 调用 {n_call} 次（{n_call / ticks:.3f} 次/tick），"
          f"总计 {call_ms:.1f} ms = **{call_ms / wall:.1%} 的墙钟**；"
          f"单次中位 {np.median(_CALL_MS) if _CALL_MS else 0:.3f} ms")
    from collections import Counter
    rc = Counter(r["reason"] for r in RECORDS)
    print(f"      触发原因：{dict(rc)}"
          f" ⇒ **强制重算占 {100 * rc['empty'] / max(1, n_call):.1f}%、"
          f"周期性重算占 {100 * rc['periodic'] / max(1, n_call):.1f}%**")
    return wall, ticks, call_ms, list(RECORDS)


def run_board(name, bs, ticks):
    RECORDS.clear(); _CALL_MS.clear()
    s = copy.deepcopy(bs)
    t0 = time.perf_counter()
    for _ in range(ticks):
        s.step(1 / 60)
    wall = (time.perf_counter() - t0) * 1e3
    ne = sum(1 for e in s.entities.values() if e.is_alive)
    n_call = len(RECORDS)
    call_ms = float(np.sum(_CALL_MS)) if _CALL_MS else 0.0
    print(f"\n  [{name}] 实体={ne} tick={ticks} 墙钟={wall:.1f} ms "
          f"({wall / ticks:.4f} ms/tick)")
    print(f"      `calculate` 调用 {n_call} 次（{n_call / ticks:.3f} 次/tick），"
          f"总计 {call_ms:.1f} ms = **{call_ms / wall:.1%} 的墙钟**；"
          f"单次中位 {np.median(_CALL_MS) if _CALL_MS else 0:.3f} ms")
    from collections import Counter
    rc = Counter(r["reason"] for r in RECORDS)
    print(f"      触发原因：{dict(rc)}"
          f" ⇒ **强制重算 占 {100 * rc['empty'] / max(1, n_call):.1f}%、"
          f"周期性重算 占 {100 * rc['periodic'] / max(1, n_call):.1f}%**")
    return wall, ticks, call_ms, list(RECORDS)


def evaluate(records, R, require_front, enemy_mode):
    """按提案条件计算可省调用数与节省比例（**只用传入盘面自己的记录**）。"""
    key_near = f"near{int(R)}"
    key_front = f"front{int(R)}"
    ok = []
    for r in records:
        c = (r[key_near] == 0)
        if require_front:
            c = c and (r[key_front] == 0)
        if enemy_mode == "mobile0":
            c = c and (r["n_enemy_mobile"] == 0)
        elif enemy_mode == "nontower0":
            c = c and (r["n_enemy_non_tower"] == 0)
        ok.append(c)
    ok = np.array(ok, dtype=bool)
    # 仅「周期性重算」可省；「path 空」是必须算的
    skippable = ok & np.array([r["reason"] == "periodic" for r in records], bool)
    # 同一 (unit,target) 的连续可省串里只保留第一次
    saved = 0
    last = {}
    for i, r in enumerate(records):
        k = (r["unit"], r["target"])
        if skippable[i]:
            if last.get(k) is True:
                saved += 1
            else:
                last[k] = True
        else:
            last[k] = False
    ms = np.array([r["ms"] for r in records]) if records else np.array([0.0])
    keep = []
    last = {}
    for i, r in enumerate(records):
        k = (r["unit"], r["target"])
        if skippable[i] and last.get(k) is True:
            continue
        keep.append(i)
        last[k] = bool(skippable[i])
    saved_ms = float(sum(ms[i] for i in range(len(records)) if i not in set(keep)))
    return {"n_ok": int(ok.sum()), "n_skippable": int(skippable.sum()),
            "saved_calls": saved, "saved_ms": saved_ms}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=400)
    ap.add_argument("--units", type=int, default=24)
    ap.add_argument("--frames", type=int, default=400)
    a = ap.parse_args()

    orig = install_hook()
    print("=" * 78)
    print("§1 现状：`EntityPathfinder.calculate` 的调用与耗时（挂接实测，无 profiler 失真）")
    try:
        res = []
        for nm, nu in (("稀疏（8 单位）", 8), ("中局（24 单位）", 24)):
            res.append((nm,) + run_board(nm, build(nu), a.ticks))
        res.append(("真实对局（随机对手）",) + run_rollout(
            "真实对局（随机对手）", a.frames))
        res.append(("真实对局（镜像对手=训练分布）",) + run_rollout_mirror(
            "真实对局（镜像对手=训练分布）", a.frames))
    finally:
        import pathfinding_heap as pf
        pf.EntityPathfinder.calculate = orig

    print("\n" + "=" * 78)
    print("§2 提案生效范围：周期性重算里有多少满足提案条件（⇒ 可省）；"
          "每个盘面**各用自己的记录**")
    print(f"    {'盘面':32s} {'R':>4s} {'前空':>5s} {'敌方口径':>10s} "
          f"{'满足':>5s} {'可省':>5s} {'省下(ms)':>9s} {'占该盘面墙钟':>12s}")
    for (nm, wall, ticks, call_ms, recs) in res:
        for R in (2.0, 4.0):
            for rf in (False, True):
                for em in ("mobile0", "nontower0"):
                    d = evaluate(recs, R, rf, em)
                    share = d["saved_ms"] / wall if wall else 0.0
                    print(f"    {nm:32s} {R:4.1f} {str(rf):>5s} {em:>10s} "
                          f"{d['n_ok']:5d} {d['saved_calls']:5d} "
                          f"{d['saved_ms']:9.1f} {share:12.2%}")

    print("\n" + "=" * 78)
    print("§3 子条件的边际/联合成立率（找出是哪一个条件在挡）——每个盘面各算")
    for (nm, wall, ticks, call_ms, recs) in res:
        n = max(1, len(recs))
        print(f"  [{nm}] 调用 {len(recs)} 次")
        for R in (2.0, 4.0):
            kn, kf = f"near{int(R)}", f"front{int(R)}"
            p_near = sum(1 for r in recs if r[kn] == 0) / n
            p_both = sum(1 for r in recs if r[kn] == 0 and r[kf] == 0) / n
            p_em0 = sum(1 for r in recs if r["n_enemy_mobile"] == 0) / n
            p_nt0 = sum(1 for r in recs if r["n_enemy_non_tower"] == 0) / n
            p_all = sum(1 for r in recs if r[kn] == 0 and r[kf] == 0
                        and r["n_enemy_mobile"] == 0) / n
            per = sum(1 for r in recs if r["reason"] == "periodic") / n
            print(f"      R={R:.0f}: P(周围无同兵种友军)={p_near:6.1%} "
                  f"P(前后都无)={p_both:6.1%} | P(敌方无可移动单位)={p_em0:6.1%} "
                  f"P(敌方无任何非塔单位)={p_nt0:6.1%} | **P(全部满足)={p_all:6.1%}** "
                  f"| P(周期性重算)={per:6.1%}")

    print("\n    读法：'可省调用' 只统计**周期性重算**中满足条件的那些"
          "（同一单位同一目标的连续可省串只保留第一次，第一次仍要算）；"
          "'path 空' 的强制重算**不可能被本提案省掉**。")


if __name__ == "__main__":
    main()
