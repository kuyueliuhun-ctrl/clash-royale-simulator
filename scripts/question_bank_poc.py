# -*- coding: utf-8 -*-
"""题库预训练可行性 POC（2026-09-09）：不写死答案，引擎判卷。

链路：构造"威胁牌已行进到位置 p"的快照 → 枚举防守手牌×粗网格落点 →
simulate_exchange（defender=none，攻方冻结）逐候选结算 → 与 threat_calc
"不动手"基线做差，按训练奖励汇率（config.DEFAULT_REWARD 前段口径）计分 →
top-k 答案集。已知解对账（MiniPekka 贴 Giant / Cannon 拉 Hog / Arrows 换
Minions）+ 现有 checkpoint 答题（EV gap vs oracle）。

计分公式（与训练奖励同量纲，前段汇率）：
  score = 0.0012*(不动手塔损 - 候选塔损)      # tower_dmg_self：少挨的塔伤
        + 0.0005*敌军HP移除量                 # unit_dmg_k
        + 0.0010*对敌塔伤害                   # tower_dmg_opp（反打）
        - 0.5*我方圣水花销                    # elixir_diff_weight
  null（不动手）score = 0（基线即参照系）。

POC 近似（正式生成器需替换）：威胁行进用"部署于桥头后直线下移"近似，
未走引擎步进（避免途中塔仇恨污染切片语义）；引擎确定性已由
threat_calc/simulate_exchange 证实，快照构造不影响判卷口径。
"""
import copy
import io
import os
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
# T1-2：不再硬编码 `E:/clash-royale-simulator-main/...`（换机器/换目录即失效）。
# 本文件在 `scripts/` 下 ⇒ 仓库根 = 上两级目录 ⇒ 引擎源码在 `<root>/src/clasher_new`。
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src", "clasher_new"))

# T1-1 追加（2026-09-19）：UTF-8 兜底。本文件含 **GBK 编不出**的字符 ⇒ 无兜底时
# `print` 抛 UnicodeEncodeError（实测：test_m3_evo 因此产生 **10 个假失败**）。
# 用 T1-1 的**单一实现**；只用在入口脚本上（`rl/` 库模块不加 —— 库不该改宿主 stdout）。
from rl.io_bootstrap import force_utf8_stdout  # noqa: E402
force_utf8_stdout()

import battle as battle_mod
import player as player_mod
from card_utils import Card
from core import Position

from threat_calc import estimate_tower_threat
from simulate_exchange import simulate_exchange

# —— 奖励汇率（config.DEFAULT_REWARD 前段）——
K_TOWER_SAVE = 0.0012
K_UNIT_DMG = 0.0005
K_TOWER_DMG_OPP = 0.0010
EDW = 0.5

HAND_N = 4


def make_battle(defender_deck, attacker_deck, threat_card, threat_pos, elixir=8.0):
    """构造快照：威胁牌已部署并行进到 threat_pos（直线下移近似），防守方满牌序。"""
    bs = battle_mod.BattleState(
        player_mod.PlayerState(0, list(defender_deck), elixir),
        player_mod.PlayerState(1, list(attacker_deck), 5.0), card_level=11)
    # 威胁卡必须在手牌前 4 才能部署（can_play_card 校验）——排到手牌首位
    bs.players[1].cycle = [threat_card] + [c for c in bs.players[1].cycle
                                           if c != threat_card]
    ok = bs.deploy_card(1, threat_card, Position(*threat_pos))
    tries = [(threat_pos[0], 18.5), (threat_pos[0], 19.5), (14.5, 18.5)]
    i = 0
    while not ok and i < len(tries):
        ok = bs.deploy_card(1, threat_card, Position(*tries[i]))
        i += 1
    assert ok, f"威胁部署失败: {threat_card}@{threat_pos}"
    threat = next(e for e in bs.entities.values()
                  if getattr(e, "player", None) == 1 and e.is_alive
                  and e.id not in (1, 2, 5))
    # 直线下移到切片位（攻方冻结语义：塔仇恨不触发、HP 满血）
    while threat.position.y > threat_pos[1] + 1e-6:
        threat.position.y -= max(getattr(threat.data, "speed", 1.0), 0.5) * 0.25
    threat.position.y = threat_pos[1]
    return bs


def question_candidates(bs, horizon, coarse=2):
    """枚举防守方手牌 4 张 × 粗网格落点 → 引擎结算计分 → top-k。"""
    from rl.action_mask import legal_cells
    hand = bs.players[0].cycle[:HAND_N]
    baseline = estimate_tower_threat(bs, 0, horizon=horizon)["total"]
    cands = []
    t0 = time.time()
    n = 0
    xs = [0.5 + coarse * i for i in range(9)]
    ys = [1.5 + coarse * i for i in range(10)]
    for ci, cname in enumerate(hand):
        cells = legal_cells(bs, 0, cname)
        for gy in range(32):
            for gx in range(18):
                if not cells[gy, gx] or gx % coarse or (gy - 1) % coarse:
                    continue
                pos = Position(gx + 0.5, gy + 0.5)
                res = simulate_exchange(bs, 0, cname, pos, horizon=horizon,
                                        defender="none")
                n += 1
                if not res["legal"]:
                    continue
                saved = baseline - res["my_towers"]["total"]
                score = (K_TOWER_SAVE * saved + K_UNIT_DMG * res["opp_units"]["hp_removed"]
                         + K_TOWER_DMG_OPP * res["opp_towers"]["total"]
                         - EDW * res["my_cost"])
                cands.append({"card": cname, "pos": (gx, gy), "score": round(score, 4),
                              "saved": round(saved, 1), "killed": res["opp_units"]["killed"],
                              "cost": res["my_cost"]})
    cands.sort(key=lambda c: -c["score"])
    print(f"    [{n} 候选结算 {time.time()-t0:.1f}s] 不动手基线塔损={baseline:.0f}")
    return baseline, cands


def ev_of(bs, cname, pos, horizon, baseline):
    res = simulate_exchange(bs, 0, cname, pos, horizon=horizon, defender="none")
    if not res["legal"]:
        return None
    saved = baseline - res["my_towers"]["total"]
    return round(K_TOWER_SAVE * saved + K_UNIT_DMG * res["opp_units"]["hp_removed"]
                 + K_TOWER_DMG_OPP * res["opp_towers"]["total"] - EDW * res["my_cost"], 4)


DECK_Q1 = ["MiniPekka", "Knight", "Musketeer", "Arrows", "Minions", "Fireball",
           "Skeletons", "Giant"]
DECK_Q2 = ["Cannon", "Skeletons", "Musketeer", "Fireball", "IceGolemite", "Log",
           "Knight", "Arrows"]
DECK_Q3 = ["Arrows", "Musketeer", "Minions", "Knight", "Fireball", "Skeletons",
           "Archers", "Giant"]

QUESTIONS = [
    dict(qid="giant_bridge", defender_deck=DECK_Q1, attacker_deck=DECK_Q1,
         threat="Giant", threat_pos=(3.5, 17.5), slice_y=14.0, horizon=20.0,
         expect=("MiniPekka", 3.0)),
    dict(qid="hog_bridge", defender_deck=DECK_Q2, attacker_deck=DECK_Q2,
         threat="HogRider", threat_pos=(14.5, 17.5), slice_y=15.0, horizon=12.0,
         expect=("Cannon", 6.0)),
    dict(qid="minions_air", defender_deck=DECK_Q3, attacker_deck=DECK_Q3,
         threat="Minions", threat_pos=(3.5, 17.5), slice_y=15.0, horizon=10.0,
         expect=("Arrows", 99.0)),
]


def build_questions():
    qs = []
    for q in QUESTIONS:
        print(f"\n=== 题 {q['qid']}: {q['threat']} 行进至 y={q['slice_y']} "
              f"（防守手牌 {q['defender_deck'][:4]}，视界 {q['horizon']:.0f}s）===")
        bs = make_battle(q["defender_deck"], q["attacker_deck"], q["threat"],
                         (q["threat_pos"][0], q["slice_y"]))
        baseline, cands = question_candidates(bs, q["horizon"])
        for i, c in enumerate(cands[:6]):
            print(f"    #{i+1} {c['card']}@{c['pos']} score={c['score']:+.3f} "
                  f"少挨塔伤={c['saved']:.0f} 击杀={c['killed']} 费={c['cost']}")
        q["battle"] = bs
        q["baseline"] = baseline
        q["top"] = cands[:8]
        qs.append(q)
    return qs


def reconcile(qs):
    print("\n—— 已知解对账 ——")
    ok_all = True
    for q in qs:
        exp_card, exp_dist = q["expect"]
        rank = next((i + 1 for i, c in enumerate(q["top"])
                     if c["card"] == exp_card), None)
        if rank is None:
            print(f"  [FAIL] {q['qid']}: 期望 {exp_card} 不在 top-8")
            ok_all = False
            continue
        c = q["top"][rank - 1]
        wx, wy = c["pos"][0] + 0.5, c["pos"][1] + 0.5
        threat = next(e for e in q["battle"].entities.values()
                      if getattr(e, "player", None) == 1 and e.is_alive
                      and e.id not in (1, 2, 5))
        dist = ((wx - threat.position.x) ** 2 + (wy - threat.position.y) ** 2) ** 0.5
        geo = "贴身✓" if dist <= exp_dist else f"距离{dist:.1f}"
        print(f"  [PASS] {q['qid']}: {exp_card} 排名 #{rank}（{geo}，score {c['score']:+.3f}）")
    return ok_all


def checkpoint_quiz(qs, ckpt_path, ckpt_label):
    import numpy as np
    import torch
    from rl.follower import load_checkpoint
    from rl.belief import BeliefInference, belief_token_dim
    from rl.belief_planner import BeliefPlanner
    from rl.plan_space import PLAN_DIM
    from rl.env_wrapper import RLEnv

    print(f"\n—— checkpoint 答题：{ckpt_label} ——")
    policy = load_checkpoint(ckpt_path, plan_dim=PLAN_DIM,
                             belief_dim=belief_token_dim(QUESTIONS[0]["attacker_deck"]))
    env = RLEnv(opponent=None, seed=0)
    rows = []
    for q in qs:
        env.battle = copy.deepcopy(q["battle"])
        env.deck0, env.deck1 = list(q["defender_deck"]), list(q["attacker_deck"])
        obs = env.observe(0)
        belief = BeliefInference(opp_deck=q["attacker_deck"], seed=0)
        bp = BeliefPlanner()
        plan_vec = bp.plan(env.battle, belief.state(), obs).to_vector()
        tok = belief.encode(obs, None)
        bundle, _, _, _, _ = policy.act(obs, tok, plan_vec, env.get_action_mask,
                                        hidden=None, deterministic=True)
        dep = next((sa for sa in bundle.sub_actions
                    if sa.kind == "deploy" and 1 <= sa.slot <= 4), None)
        if dep is None:
            ans_card, ans_pos, ans_ev = "WAIT(null)", None, 0.0
        else:
            ans_card = q["defender_deck"][dep.slot - 1]
            ans_pos = (dep.x, dep.y)
            wx, wy = dep.x + 0.5, dep.y + 0.5
            ans_ev = ev_of(env.battle, ans_card, Position(wx, wy), q["horizon"],
                           q["baseline"])
            if ans_ev is None:
                ans_ev = 0.0   # 非法部署（被闸门拒）→ 视作不动手
        best = q["top"][0]["score"]
        hit_rank = next((i + 1 for i, c in enumerate(q["top"])
                         if c["card"] == ans_card and ans_pos
                         and abs(c["pos"][0] - ans_pos[0]) <= 1
                         and abs(c["pos"][1] - ans_pos[1]) <= 1), None)
        gap = best - ans_ev
        rows.append({"qid": q["qid"], "ans": f"{ans_card}@{ans_pos}", "ev": ans_ev,
                     "best": best, "gap": gap, "hit": hit_rank})
        print(f"  {q['qid']}: 答 {ans_card}@{ans_pos} EV={ans_ev:+.3f} "
              f"oracle={best:+.3f} gap={gap:+.3f} "
              f"(≈{gap/K_TOWER_SAVE:.0f} 等效塔伤) top8命中={hit_rank}")
    n_gap = sum(r["gap"] for r in rows) / len(rows)
    n_hit = sum(1 for r in rows if r["hit"]) / len(rows)
    print(f"  ⌀ EV gap={n_gap:+.3f}（≈{n_gap/K_TOWER_SAVE:.0f} 等效塔伤/题），"
          f"top-8 命中 {n_hit:.0%}")
    return rows


if __name__ == "__main__":
    t0 = time.time()
    qs = build_questions()
    ok = reconcile(qs)
    print(f"\n对账 {'全部通过' if ok else '存在 FAIL'}，耗时 {time.time()-t0:.0f}s")

    if "--quiz" in sys.argv:
        # T1-2：路径由 `_ROOT`（= `__file__` 推上两级）派生，不再硬编码盘符
        checkpoint_quiz(qs, os.path.join(_ROOT, "runs", "archive",
                                        "economy_100k_v1_ungated", "solo_main.pt"),
                        "100k archive (9j era)")
        checkpoint_quiz(qs, os.path.join(_ROOT, "src", "clasher_new",
                                        "main_ckpt_32000.pt"), "main_ckpt_32000 (旧底座)")
        print(f"\n总耗时 {time.time()-t0:.0f}s")
