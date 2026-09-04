"""评测（规划文档 11）：对局胜率 + 同刻多卡统计 + 信念校准 + 联赛 Elo。

修复：
- P0-5：load_policy 从 checkpoint 元数据读维度，不再硬编码 belief_dim=23；
- P1-22：指标补齐（Win/Lose/Draw、Crown/Tower 差、Elixir Efficiency、Plan Following、
  Belief Brier/Next-Card Acc/ECE）；对手池可配置（random/heuristic/checkpoint）；
  evaluate_belief 独立 N=500 信念协议。
- P2：清理 use_rule_belief 死参数、循环内重复 import。
"""

import os
import sys
import json
import math
import random
import argparse

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import numpy as np

from rl.env_wrapper import RLEnv
from rl.belief import BeliefInference
from rl.belief_planner import BeliefPlanner
from rl.follower import load_checkpoint
from rl.plan_space import PlanToken
from rl.train_follower import heuristic_opponent, FollowerOpponent
from rl.observation import ENTITY_NAMES
from rl.action_bundle import K_MAX
from card_utils import Card


def _ece(conf, acc, n_bins=10):
    ece = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        m = (conf >= edges[i]) & (conf < edges[i + 1])
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(conf)) * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def _plan_region_hit(region: str, x: float, y: float):
    """plan.focus_region 与模型首个部署格 (x, y)（player0 本地坐标）的粗粒度吻合。

    - own_*：y < 15.5（己方半场）；enemy_*：y > 16.5；bridge_*：桥带 13..19；
    - left/right 以 LANE=9 分边；center 类只查半场。
    仅用于逐意图采纳率统计（Phase 2 消融探针），不参与训练/判分。
    """
    x = float(x); y = float(y)
    side_ok = None
    if region in ("own_left", "enemy_left", "bridge_left"):
        side_ok = x < 9.0
    elif region in ("own_right", "enemy_right", "bridge_right"):
        side_ok = x >= 9.0
    half_ok = None
    if region.startswith("own"):
        half_ok = y < 15.5
    elif region.startswith("enemy"):
        half_ok = y > 16.5
    elif region.startswith("bridge"):
        half_ok = 13.0 <= y <= 19.0
    if side_ok is None and half_ok is None:
        return None
    ok = True
    if side_ok is not None:
        ok = ok and side_ok
    if half_ok is not None:
        ok = ok and half_ok
    return bool(ok)


def _adoption_seed():
    return {"frames": 0, "deploys": 0, "reg_frames": 0, "reg_hit": 0,
            "hold_frames": 0, "hold_ok": 0}


def _adoption_summary(adoption):
    """原始计数 → 可落盘摘要（rate 为空样本返回 None）。"""
    out = {}
    for intent, a in adoption.items():
        out[intent] = {
            "frames": a["frames"], "deploys": a["deploys"],
            "region_frames": a["reg_frames"],
            "region_rate": (round(a["reg_hit"] / a["reg_frames"], 4)
                            if a["reg_frames"] else None),
            "hold_frames": a["hold_frames"],
            "hold_rate": (round(a["hold_ok"] / a["hold_frames"], 4)
                          if a["hold_frames"] else None),
        }
    return out


def _record_adoption(stats, plan, bundle):
    """逐意图采纳探针：plan 意图标签（与注入是否置零无关）下模型本帧动作的吻合度。

    - region_rate：首个部署格与 plan.focus_region 几何吻合（有部署的帧里）；
    - hold_rate（仅 save_ace）：有部署帧里没有打出 hold_mask 指名的槽。
    比较 full vs plan-off 变体 → Δ 即「plan 注入对该意图行为的因果采纳」。
    """
    a = stats["adoption"].setdefault(plan.macro_intent, _adoption_seed())
    a["frames"] += 1
    dep = [(sa.slot, sa.x, sa.y) for sa in bundle.sub_actions
           if sa.kind == "deploy" and 1 <= sa.slot <= K_MAX]
    if not dep:
        return
    a["deploys"] += 1
    hit = _plan_region_hit(plan.focus_region, dep[0][1], dep[0][2])
    if hit is not None:
        a["reg_frames"] += 1
        a["reg_hit"] += int(hit)
    if plan.macro_intent == "save_ace":
        held = set(plan.hold_slots())
        if held:
            a["hold_frames"] += 1
            if not any(slot in held for slot, _, _ in dep):
                a["hold_ok"] += 1


def _make_opponent(env, opponent, opponent_policy=None, rng=None):
    if opponent == "heuristic":
        return heuristic_opponent(env, rng)
    if opponent == "checkpoint":
        if not opponent_policy:
            raise ValueError("--opponent checkpoint 需要 --opponent-policy")
        pol = load_checkpoint(opponent_policy)
        return FollowerOpponent(pol, env)
    return None  # random


def run_eval(policy_path, n_games=50, opponent="random", seed=0, hidden_dim=None,
             max_steps=300, opponent_policy=None, ablation=None):
    env = RLEnv(opponent=None, seed=seed)
    rng = random.Random(seed)
    env.opponent = _make_opponent(env, opponent, opponent_policy, rng)

    policy = load_policy(policy_path, hidden_dim=hidden_dim)
    belief = BeliefInference(opp_deck=env.deck1, n_particles=128, seed=seed)
    bp = BeliefPlanner()

    stats = {"wins": 0, "losses": 0, "draws": 0,
             "rew": 0.0, "bundle_sizes": [], "bundle_illegal": 0, "bundle_total": 0,
             "next_hits": 0, "next_total": 0, "next_brier": 0.0,
             "hand_hits": 0, "hand_total": 0,
             "elixir_spent": 0.0, "plan_intents": {},
             "adoption": {},   # Phase 2：逐意图采纳率探针（region 吻合 / save_ace hold 服从）
             "crown_diff": [], "tower_diff": [],
             "belief_conf": [], "belief_correct": [],
             }
    deck_ids = [ENTITY_NAMES.index(c) for c in env.deck1]

    for g in range(n_games):
        obs, _ = env.reset()
        belief.reset(env.deck1)
        hidden = None
        done = False
        steps = 0
        while not done and steps < max_steps:
            plan = bp.plan(env.battle, belief.state(), obs)
            stats["plan_intents"][plan.macro_intent] = stats["plan_intents"].get(plan.macro_intent, 0) + 1
            plan_vec = plan.to_vector()
            tok = belief.encode(obs, None)
            if ablation in ("plan", "both"):
                plan_vec = np.zeros_like(plan_vec)
            if ablation in ("belief", "both"):
                tok = np.zeros_like(tok)
            bundle, _, _, hidden, _ = policy.act(obs, tok, plan_vec,
                                                 env.get_action_mask, hidden=hidden, deterministic=True)
            stats["bundle_sizes"].append(bundle.size)
            stats["bundle_total"] += 1
            # —— Phase 2 采纳率探针：plan 意图（注入前/置零前同标签）vs 模型本帧动作 ——
            _record_adoption(stats, plan, bundle)
            # 圣水效率：bundle 内出牌费用（决策时刻手牌）
            for sa in bundle.sub_actions:
                if sa.kind == "deploy" and 1 <= sa.slot <= K_MAX:
                    cid = int(obs["hand"][sa.slot - 1])
                    if 0 <= cid < len(ENTITY_NAMES):
                        stats["elixir_spent"] += Card(ENTITY_NAMES[cid]).elixir
            obs, r, term, trunc, info = env.step(bundle)
            if not info.get("bundle_ok"):
                stats["bundle_illegal"] += 1
            belief.update(obs, info.get("opp_played"))
            if info.get("hidden") is not None:
                hid = info["hidden"]
                st = belief.state()
                real_next = int(hid["opp_next"])
                pred_idx = int(np.argmax(st.next_probs))
                pred = deck_ids[pred_idx]
                stats["next_total"] += 1
                if pred == real_next:
                    stats["next_hits"] += 1
                # Brier
                onehot = np.zeros(len(deck_ids), dtype=np.float32)
                if real_next in deck_ids:
                    onehot[deck_ids.index(real_next)] = 1.0
                stats["next_brier"] += float(np.mean((st.next_probs - onehot) ** 2))
                # 置信度校准（next_probs）
                stats["belief_conf"].append(float(st.next_probs.max()))
                stats["belief_correct"].append(1.0 if pred == real_next else 0.0)
                # 手牌命中（top4 与真实手牌的重叠）
                real_hand = set(int(v) for v in hid["opp_hand"])
                pred_hand = {deck_ids[i] for i in np.argsort(-st.hand_probs)[:4]}
                stats["hand_total"] += 4
                stats["hand_hits"] += len(pred_hand & real_hand)
            done = term or trunc
            stats["rew"] += r
            steps += 1
        w = env.battle.winner
        if w is None and not env.battle.game_over:
            # 步数截断早于引擎结算 → 皇冠差/塔血差补判（真平保持 None=draw）
            from rl.run_league import timeout_winner
            w = timeout_winner(env.battle)
            rw = env.reward_weights or {}
            if w == 0:
                stats["rew"] += float(rw.get("win_bonus", 0.0))
            elif w == 1:
                stats["rew"] -= float(rw.get("lose_penalty", 0.0))
        if w == 0:
            stats["wins"] += 1
        elif w == 1:
            stats["losses"] += 1
        else:
            stats["draws"] += 1
        p0, p1 = env.battle.players
        stats["crown_diff"].append(p0.get_crown_count() - p1.get_crown_count())
        stats["tower_diff"].append(
            (p0.king_tower_hp + p0.left_tower_hp + p0.right_tower_hp)
            - (p1.king_tower_hp + p1.left_tower_hp + p1.right_tower_hp))

    print("=== 评测结果 ===")
    print(f"对手: {opponent}  场次: {n_games}")
    print(f"Win / Lose / Draw: {stats['wins']} / {stats['losses']} / {stats['draws']} "
          f"(WinRate={stats['wins']/max(1, n_games):.3f})")
    print(f"Mean Reward: {stats['rew']/max(1, n_games):.3f}")
    if stats["bundle_sizes"]:
        print(f"Bundle 大小分布: {np.bincount(stats['bundle_sizes']) / len(stats['bundle_sizes'])}")
    print(f"Bundle 合法率: {1 - stats['bundle_illegal']/max(1, stats['bundle_total']):.3f} "
          f"({stats['bundle_total']} 次)")
    if stats["elixir_spent"] > 0:
        print(f"Elixir Efficiency (reward/elixir): "
              f"{stats['rew']/stats['elixir_spent']:.4f}")
    if stats["next_total"]:
        conf = np.array(stats["belief_conf"]); acc = np.array(stats["belief_correct"])
        print(f"Next-Card Acc: {stats['next_hits']/stats['next_total']:.3f} "
              f"({stats['next_hits']}/{stats['next_total']})")
        print(f"Next-Card Brier: {stats['next_brier']/stats['next_total']:.4f}")
        print(f"Belief ECE: {_ece(conf, acc):.4f}")
    if stats["hand_total"]:
        print(f"Hand Top4 Overlap: {stats['hand_hits']/stats['hand_total']:.3f}")
    if stats["plan_intents"]:
        print(f"Plan 意图分布: {dict(sorted(stats['plan_intents'].items(), key=lambda kv: -kv[1]))}")
    if stats["crown_diff"]:
        print(f"Mean Crown Diff: {np.mean(stats['crown_diff']):.3f}  "
              f"Mean Tower HP Diff: {np.mean(stats['tower_diff']):.1f}")
    adoption = _adoption_summary(stats["adoption"])
    return {"winrate": stats["wins"] / max(1, n_games),
            "mean_reward": stats["rew"] / max(1, n_games),
            "wins": stats["wins"], "losses": stats["losses"], "draws": stats["draws"],
            "n_games": n_games,
            "adoption": adoption,
            # 二项 SE：sqrt(p(1-p)/N)，用于消融 delta 的显著性判断
            "winrate_se": math.sqrt(
                (stats["wins"] / max(1, n_games)) * (1 - stats["wins"] / max(1, n_games))
                / max(1, n_games))}


# ---------------------------------------------------------------------------
# belief / plan 输入消融（P-flow 前置验证）
# ---------------------------------------------------------------------------

ABLATION_VARIANTS = [
    ("full", None, "基线（belief + plan 都注入）"),
    ("plan-off", "plan", "plan token 置零（belief 保留）"),
    ("belief-off", "belief", "belief token 置零（plan 保留）"),
    ("both-off", "both", "两者都置零"),
]


def run_ablation(policy_path, n_games=200, opponent="random", seed=0, hidden_dim=None,
                 max_steps=300, opponent_policy=None, out_path=None):
    """跑 belief/plan 输入消融并**落盘记录**（JSON + CSV）。

    变体：full / plan-off / belief-off / both-off。输出：
    - 各变体 winrate ± SE（二项 SE=√(p(1-p)/N)）；
    - 相对 full 的 delta、delta SE（√(SE₁²+SE₂²)）、z=delta/SE；
    - 判定：|z|≥2 视为「该输入对胜率有真实贡献（或真实损害）」。
    注意：token 置零是保守消融——FollowerPolicy 的 RNN hidden 仍携带历史
    belief/plan 信息，可能低估这两个输入的价值；结论应结合 z 与样本量。
    """
    results = {}
    print(f"=== 消融评测（belief/plan 注入价值）===")
    print(f"策略: {policy_path}  对手: {opponent}  N={n_games}/变体  max_steps={max_steps}")
    for name, ab, desc in ABLATION_VARIANTS:
        r = run_eval(policy_path, n_games=n_games, opponent=opponent, seed=seed,
                     hidden_dim=hidden_dim, max_steps=max_steps,
                     opponent_policy=opponent_policy, ablation=ab)
        results[name] = r
        print(f"  [{name:10s}] {desc}  WinRate={r['winrate']:.3f}±{r['winrate_se']:.3f} "
              f"({r['wins']}W/{r['losses']}L/{r['draws']}D)  reward={r['mean_reward']:.3f}",
              flush=True)

    def delta_vs(name):
        full, ab = results["full"], results[name]
        d = full["winrate"] - ab["winrate"]
        se = math.hypot(full["winrate_se"], ab["winrate_se"])
        z = d / se if se > 0 else None
        return {"full_winrate": round(full["winrate"], 4),
                "abl_winrate": round(ab["winrate"], 4),
                "delta": round(d, 4), "delta_se": round(se, 4),
                "z": round(z, 2) if z is not None else None,
                "verdict": ("贡献/损害显著" if z is not None and abs(z) >= 2
                            else "无显著差异（噪声内）")}

    deltas = {k: delta_vs(k) for k in ("plan-off", "belief-off", "both-off")}
    print("=== 相对 full 的 delta（ΔWinRate，正=该输入有贡献）===")
    for k, d in deltas.items():
        zs = f"{d['z']}σ" if d["z"] is not None else "—"
        print(f"  {k:11s}: Δ={d['delta']:+.4f} ± {d['delta_se']:.4f}  z={zs}  -> {d['verdict']}")

    # —— Phase 2：逐意图采纳率探针（full vs plan-off = plan 注入的因果采纳）——
    adoption_vs_off = {}
    full_a = results["full"].get("adoption", {})
    off_a = results["plan-off"].get("adoption", {})
    print("=== 逐意图采纳（region 吻合率；Δ=full − plan-off >0 → plan 注入被采纳）===")
    print(f"  {'intent':<16}{'full帧':>6}{'off帧':>6}{'reg_full':>9}{'reg_off':>9}"
          f"{'Δreg':>8}{'hold_full':>10}{'hold_off':>10}{'Δhold':>8}")
    for intent in sorted(set(full_a) | set(off_a), key=lambda s: -full_a.get(s, {}).get("frames", 0)):
        fa, oa = full_a.get(intent, {}), off_a.get(intent, {})
        rf = fa.get("region_rate"); ro = oa.get("region_rate")
        hf = fa.get("hold_rate"); ho = oa.get("hold_rate")
        dr = (rf - ro) if (rf is not None and ro is not None) else None
        dh = (hf - ho) if (hf is not None and ho is not None) else None
        adoption_vs_off[intent] = {
            "full_frames": fa.get("frames", 0), "off_frames": oa.get("frames", 0),
            "region_rate_full": rf, "region_rate_off": ro,
            "region_delta": round(dr, 4) if dr is not None else None,
            "hold_rate_full": hf, "hold_rate_off": ho,
            "hold_delta": round(dh, 4) if dh is not None else None,
        }
        print(f"  {intent:<16}{fa.get('frames', 0):>6}{oa.get('frames', 0):>6}"
              f"{'' if rf is None else f'{rf:.3f}':>9}{'' if ro is None else f'{ro:.3f}':>9}"
              f"{'' if dr is None else f'{dr:+.3f}':>8}"
              f"{'' if hf is None else f'{hf:.3f}':>10}"
              f"{'' if ho is None else f'{ho:.3f}':>10}"
              f"{'' if dh is None else f'{dh:+.3f}':>8}")

    out = {
        "policy": os.path.abspath(policy_path),
        "opponent": opponent, "n_games": n_games, "seed": seed, "max_steps": max_steps,
        "variants": {k: {"winrate": v["winrate"], "winrate_se": v["winrate_se"],
                         "mean_reward": v["mean_reward"], "wins": v["wins"],
                         "losses": v["losses"], "draws": v["draws"], "n_games": v["n_games"],
                         "adoption": v.get("adoption", {})}
                     for k, v in results.items()},
        "deltas_vs_full": deltas,
        "adoption_vs_plan_off": adoption_vs_off,
        "note": "token 置零消融；RNN hidden 仍含历史信息，可能低估输入价值；"
                "adoption 探针：region=首部署格与 plan 半场/分边吻合率，hold=save_ace 未打出 hold 槽率",
    }
    if out_path:
        out_path = os.path.abspath(out_path)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        csv_path = os.path.splitext(out_path)[0] + ".csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("variant,winrate,winrate_se,wins,losses,draws,mean_reward\n")
            for k, v in results.items():
                f.write(f"{k},{v['winrate']:.6f},{v['winrate_se']:.6f},"
                        f"{v['wins']},{v['losses']},{v['draws']},{v['mean_reward']:.6f}\n")
            f.write("\n# delta vs full: variant,delta,delta_se,z,verdict\n")
            for k, d in deltas.items():
                f.write(f"# {k},{d['delta']},{d['delta_se']},{d['z']},{d['verdict']}\n")
            f.write("\n# adoption by intent: variant,intent,frames,deploys,"
                    "region_frames,region_rate,hold_frames,hold_rate\n")
            for variant in ("full", "plan-off"):
                for intent, a in sorted(results[variant].get("adoption", {}).items()):
                    rr = a["region_rate"] if a["region_rate"] is not None else ""
                    hr = a["hold_rate"] if a["hold_rate"] is not None else ""
                    f.write(f"{variant},{intent},{a['frames']},{a['deploys']},"
                            f"{a['region_frames']},{rr},{a['hold_frames']},{hr}\n")
            f.write("\n# adoption delta (full - plan-off): intent,region_rate_full,"
                    "region_rate_off,region_delta,hold_rate_full,hold_rate_off,hold_delta\n")
            for intent, a in sorted(adoption_vs_off.items()):
                f.write(f"# {intent},{a['region_rate_full']},{a['region_rate_off']},"
                        f"{a['region_delta']},{a['hold_rate_full']},{a['hold_rate_off']},"
                        f"{a['hold_delta']}\n")
        print(f"[ablation] 结果已落盘 -> {out_path}")
        print(f"[ablation] 明细 CSV   -> {csv_path}")
    return out


def load_policy(path, hidden_dim=None):
    return load_checkpoint(path, hidden_dim=hidden_dim)


def evaluate_belief(policy_path, n_games=500, seed=0, hidden_dim=None, max_steps=300):
    """独立信念协议：只统计信念预测 vs 特权标签（N=500，规格 §11.2）。"""
    env = RLEnv(opponent=None, seed=seed)
    policy = load_policy(policy_path, hidden_dim=hidden_dim)
    belief = BeliefInference(opp_deck=env.deck1, n_particles=128, seed=seed)
    bp = BeliefPlanner()
    deck_ids = [ENTITY_NAMES.index(c) for c in env.deck1]
    hits = total = 0
    brier = 0.0
    confs, accs = [], []
    for g in range(n_games):
        obs, _ = env.reset()
        belief.reset(env.deck1)
        hidden = None
        done = False
        steps = 0
        while not done and steps < max_steps:
            plan = bp.plan(env.battle, belief.state(), obs)
            tok = belief.encode(obs, None)
            bundle, _, _, hidden, _ = policy.act(obs, tok, plan.to_vector(),
                                                 env.get_action_mask, hidden=hidden, deterministic=True)
            obs, _, term, trunc, info = env.step(bundle)
            belief.update(obs, info.get("opp_played"))
            if info.get("hidden") is not None:
                hid = info["hidden"]
                st = belief.state()
                real_next = int(hid["opp_next"])
                pred = deck_ids[int(np.argmax(st.next_probs))]
                total += 1
                if pred == real_next:
                    hits += 1
                onehot = np.zeros(len(deck_ids), dtype=np.float32)
                if real_next in deck_ids:
                    onehot[deck_ids.index(real_next)] = 1.0
                brier += float(np.mean((st.next_probs - onehot) ** 2))
                confs.append(float(st.next_probs.max()))
                accs.append(1.0 if pred == real_next else 0.0)
            done = term or trunc
            steps += 1
    confs = np.array(confs); accs = np.array(accs)
    print("=== 信念评测协议 (N=%d) ===" % n_games)
    print(f"Next-Card Acc: {hits/max(1, total):.3f} ({hits}/{total})")
    print(f"Next-Card Brier: {brier/max(1, total):.4f}   ECE: {_ece(confs, accs):.4f}")
    uniform = 1.0 / len(deck_ids)
    print(f"vs uniform 先验 Brier: {float(np.mean((np.full(len(deck_ids), uniform) - np.eye(len(deck_ids))[0])**2)):.4f} "
          f"(baseline)")
    return {"next_acc": hits / max(1, total), "next_brier": brier / max(1, total)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=str, required=True)
    ap.add_argument("--n-games", type=int, default=50)
    ap.add_argument("--opponent", type=str, default="random",
                    choices=["random", "heuristic", "checkpoint"])
    ap.add_argument("--opponent-policy", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden-dim", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--belief-only", action="store_true", help="只跑信念协议")
    ap.add_argument("--ablation", type=str, default=None,
                    choices=["plan", "belief", "both", "all"],
                    help="消融：plan=禁用 plan token，belief=禁用 belief token，"
                         "both=两者都禁用，all=跑完整对比(full/plan/belief/both)并落盘")
    ap.add_argument("--ablation-out", type=str, default=None,
                    help="--ablation all 的结果落盘路径（缺省 ablation_result.json）")
    args = ap.parse_args()
    if args.belief_only:
        evaluate_belief(args.policy, n_games=args.n_games, seed=args.seed,
                        hidden_dim=args.hidden_dim, max_steps=args.max_steps)
    elif args.ablation == "all":
        out = args.ablation_out or os.path.join(
            os.path.dirname(os.path.abspath(args.policy)) or ".", "ablation_result.json")
        run_ablation(args.policy, n_games=args.n_games, opponent=args.opponent,
                     seed=args.seed, hidden_dim=args.hidden_dim,
                     max_steps=args.max_steps, opponent_policy=args.opponent_policy,
                     out_path=out)
    else:
        run_eval(args.policy, n_games=args.n_games, opponent=args.opponent, seed=args.seed,
                 hidden_dim=args.hidden_dim, max_steps=args.max_steps,
                 opponent_policy=args.opponent_policy, ablation=args.ablation)
