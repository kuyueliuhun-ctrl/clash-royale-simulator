# -*- coding: utf-8 -*-
"""「初始模型 + 全过程随机」100 局评估 runner（只读，产出 dashboard 可读的录像与状态）。

## 它做什么（用户 2026-09-18 直接要求）

> 按初始模型跑 100 轮评估，按全过程随机试验，并且使用 dashboard，观测指标和对局回放。

* **p0（被测方）= 全过程均匀随机**：每个 decoder 步在**掩码给的合法项**里均匀抽
  （0..3 槽位 / ability / STOP），选中槽位后在 `mask["cells"]` 的**合法格**里均匀抽落点。
  ⇒ 与 `probe_explore_randomization.py --arms uniform` **同一口径**（那边是等概率随机臂）。
* **p1（对手）= 初始模型的冻结确定性副本**（argmax）。⇒ 与生产评估
  （`run_league._run_side0_scripted`，`--only-vs-main`）同一约定，只是把 p0 换成随机 actor。
* **100 局**，每 `--block` 局落一次盘：`replays/league_<累计帧>.pkl` + `solo_state.json`
  ⇒ dashboard（`--solo <目录>`）**边跑边能看**，且可以打开回放。

## 产物与口径

* 录像用**生产同一写入路径**：`LeagueGameRecorder.record()` → `replay.battle_snapshot()`
  → `save_league_replays()`（**schema 5**，含实体 `id`/`target_id`/`root_cast`）。
  帧 = **post-step** 状态 + 本步 bundle/reward（与 `run_league.py:313` 逐字同序）。
* `solo_state.json` 的 `history[]` 逐点是**这一个随机臂**的胜率曲线；
  `step` = **累计决策帧**（【R9】`step` 的定义），不是局数。
* ⚠️ **本 runner 不改任何训练/奖励/判定代码**；随机 actor 只在**进程内**替换 p0 的动作来源。
* ⚠️ 【R5】n=1 ckpt、单 seed ⇒ 读数**描述性**；【R3】非判据。

用法（必须在 src/clasher_new 下运行）：
    cd src/clasher_new && PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe \
        ../../scripts/random_eval_100.py \
        --ckpt runs/et_solo100k/solo_main_0.pt --games 100 --block 10 \
        --out runs/rand100_eval --seed 0
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
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np  # noqa: E402


# ----------------------------------------------------------------------
# 全过程随机 actor（口径与 probe_explore_randomization._uniform_bundle 相同）
# ----------------------------------------------------------------------
def random_bundle(get_mask, rng, torch):
    """每个 decoder 步在**掩码给的合法项**里均匀抽；落点在**合法格**里均匀抽。

    全程只用 `mask["slots"] / mask["ability_legal"] / mask["cells"] / mask["at_cap"]`，
    **绝不自己构造合法集合**（否则会重演 B 类非法包病理，见
    `docs/mask_used_slot_offbyone_fix_2026-09-18.md`）。
    """
    from rl.follower import K_MAX, ABILITY_IDX, STOP_IDX
    from rl.action_mask import ActionBundle
    b = ActionBundle()
    cell = None
    for _ in range(K_MAX + 2):
        m = get_mask(b)
        opts = [s for s in range(K_MAX) if bool(m["slots"][s])]
        if m.get("ability_legal"):
            opts.append(ABILITY_IDX)
        opts.append(STOP_IDX)                      # STOP 恒合法
        if m.get("at_cap"):
            opts = [STOP_IDX]
        o = int(opts[int(rng.integers(len(opts)))])
        if o == STOP_IDX:
            break
        if o == ABILITY_IDX:
            b.add_ability()
            continue
        m2 = np.asarray(m["cells"][o])
        legal = np.flatnonzero(m2.reshape(-1) > 0)
        if legal.size == 0:                        # 掩码自相矛盾：本帧按"不出"处理
            b = ActionBundle()
            break
        c = int(legal[int(rng.integers(legal.size))])
        y, x = (divmod(c, m2.shape[1]) if m2.ndim == 2 else (c, 0))
        b.add(o + 1, int(x), int(y))
        cell = (o + 1, int(x), int(y), int(legal.size))
    return b, cell


def build_policy(ckpt, device, torch, label):
    from rl.follower import FollowerPolicy
    d = torch.load(ckpt, map_location="cpu")
    sd = d["state_dict"] if isinstance(d, dict) and "state_dict" in d else d
    meta = {k: v for k, v in d.items() if k != "state_dict"} if isinstance(d, dict) else {}
    pol = FollowerPolicy(
        hidden=int(meta.get("hidden_dim", 128)),
        plan_dim=int(meta["plan_dim"]),
        belief_dim=int(meta["belief_dim"]),
        value_bypass=bool(meta.get("value_bypass", False)),
        value_independent=bool(meta.get("value_independent", False)),
    )
    miss, unexp = pol.load_state_dict(sd, strict=False)
    if miss or unexp:
        print(f"[warn] {label} 载入 missing={len(miss)} unexpected={len(unexp)}")
    pol.eval()
    pol.to_device(device)
    return pol, meta


def main():
    ap = argparse.ArgumentParser(description="初始模型 + 全过程随机：100 局评估 runner")
    ap.add_argument("--ckpt", required=True, help="初始模型 ckpt（也用作确定性对手）")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--block", type=int, default=10, help="每多少局落一次盘")
    ap.add_argument("--out", required=True, help="run 目录（写 replays/ + solo_state.json）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--json", default=None, help="汇总读数 JSON（缺省 = <out>/stats.json）")
    a = ap.parse_args()

    import torch
    from rl.env_wrapper import RLEnv
    from rl.belief import BeliefInference
    from rl.belief_planner import BeliefPlanner
    from rl import train_solo as ts
    from rl.run_league import LeagueGameRecorder, _bundle_cards, timeout_winner
    from rl.replay import save_league_replays
    from rl.overtime import overtime_open

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = a.out
    rep_dir = os.path.join(out, "replays")
    os.makedirs(rep_dir, exist_ok=True)

    pol, meta = build_policy(a.ckpt, dev, torch, "初始模型")
    opp_pol, _ = build_policy(a.ckpt, dev, torch, "对手(初始模型)")
    bp = BeliefPlanner()
    rng = np.random.default_rng(a.seed)

    print(f"[runner] ckpt={a.ckpt}  device={dev}  局数={a.games}  每块={a.block}  "
          f"out={out}", flush=True)
    print(f"[runner] p0 = 全过程均匀随机（合法项 × 合法格）；p1 = 初始模型 argmax", flush=True)

    games = []
    hist = []
    cw = cl = cd = 0
    frames_total = 0
    rew_total = 0.0
    plays_total = 0
    empty_total = 0
    t0 = time.time()
    block_frames = 0
    block_rew = 0.0
    block_games = 0
    block_w = block_l = block_d = 0

    for g in range(a.games):
        env = RLEnv(opponent=None, seed=a.seed + 31 * g, card_level=11,
                    deck0=list(ts.DEFAULT_SOLO_DECK), deck1=list(ts.DEFAULT_SOLO_DECK))
        env.opponent = ts.FollowerOpponent(
            opp_pol, env,
            belief=BeliefInference(opp_deck=env.deck1, n_particles=128, seed=a.seed + g),
            deterministic=True)
        obs, _ = env.reset(seed=a.seed + 2000 + g)
        rec = LeagueGameRecorder("rand", "init@0", "rand", a.max_steps)
        rec.set_decks(env.deck0, env.deck1)
        done = False
        steps = 0
        while not done and (steps < a.max_steps or overtime_open(env.battle)):
            bundle, _cell = random_bundle(env.get_action_mask, rng, torch)
            cards = _bundle_cards(bundle, obs)
            obs, reward, term, trunc, info = env.step(bundle)
            rec.record(env, bundle, reward, info, cards=cards)
            frames_total += 1
            rew_total += float(reward)
            block_frames += 1
            block_rew += float(reward)
            if bundle.size == 0:
                empty_total += 1
            plays_total += len(cards)
            done = term or trunc
            steps += 1
        w = env.battle.winner
        if w is None and not env.battle.game_over:
            w = timeout_winner(env.battle)
        games.append(rec.done(w))
        cw += int(w == 0)
        cl += int(w == 1)
        cd += int(w is None)
        block_w += int(w == 0)
        block_l += int(w == 1)
        block_d += int(w is None)
        block_games += 1

        if block_games >= a.block or g == a.games - 1:
            n = cw + cl + cd
            hist.append({
                "step": int(frames_total),          # 【R9】step = 累计决策帧
                "wins": block_w, "losses": block_l, "draws": block_d,
                "games": block_games,
                "cum_wins": cw, "cum_losses": cl, "cum_draws": cd, "cum_games": n,
                "winrate": cw / max(1, n),
                "winrate_se": float(np.sqrt(max(1e-9, (cw / max(1, n)) * (1 - cw / max(1, n)))
                                             / max(1, n))),
                "mean_reward": rew_total / max(1, frames_total),
                "block_mean_reward": block_rew / max(1, block_frames),
                "empty_bundle_rate": empty_total / max(1, frames_total),
                "plays_per_frame": plays_total / max(1, frames_total),
            })
            save_league_replays(games, os.path.join(rep_dir, f"league_{int(frames_total)}.pkl"))
            state = {
                "mode": "solo",
                "agents": [{"agent_id": "rand", "kind": "main", "path": None}],
                "history": hist,
                "total_steps": int(frames_total),
                "target_steps": int(frames_total),   # 已跑完 ⇒ 进度条 100%
                "deck": list(ts.DEFAULT_SOLO_DECK),
                "opponent": "init-frozen-copy(deterministic)",
                "copy_every": 0,
                "status": "running" if g < a.games - 1 else "done",
                "demo": False,
                "_runner": {"kind": "random_eval", "arm": "uniform-random",
                            "ckpt": a.ckpt, "games_planned": a.games,
                            "block": a.block, "seed": a.seed},
            }
            with open(os.path.join(out, "solo_state.json"), "w", encoding="utf-8") as fh:
                json.dump(state, fh, ensure_ascii=False, indent=2)
            el = time.time() - t0
            print(f"  [块 {len(hist)}] 已完成 {n}/{a.games} 局、{frames_total} 帧 | "
                  f"胜 {cw} 负 {cl} 平 {cd} = {100*cw/max(1,n):.1f}% | "
                  f"局均帧 {frames_total/max(1,n):.1f} | 用时 {el:.0f}s "
                  f"(预计 {el/max(1,g+1)*a.games:.0f}s)", flush=True)
            block_frames = 0
            block_rew = 0.0
            block_games = 0
            block_w = block_l = block_d = 0

    n = cw + cl + cd
    summary = {
        "ckpt": a.ckpt, "arm": "uniform-random(p0) vs init-argmax(p1)",
        "games": n, "frames": frames_total,
        "wins": cw, "losses": cl, "draws": cd, "winrate": cw / max(1, n),
        "winrate_se": float(np.sqrt(max(1e-9, (cw / max(1, n)) * (1 - cw / max(1, n))) / max(1, n))),
        "mean_reward_per_frame": rew_total / max(1, frames_total),
        "frames_per_game": frames_total / max(1, n),
        "empty_bundle_rate": empty_total / max(1, frames_total),
        "plays_per_frame": plays_total / max(1, frames_total),
        "seconds": round(time.time() - t0, 1),
        "history": hist,
    }
    jp = a.json or os.path.join(out, "stats.json")
    with open(jp, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(f"[runner] 完成 {n} 局 / {frames_total} 帧 / {summary['seconds']}s | "
          f"胜率 {100*summary['winrate']:.1f}%±{100*summary['winrate_se']:.1f} | "
          f"空 bundle {100*summary['empty_bundle_rate']:.1f}% | "
          f"每帧出牌 {summary['plays_per_frame']:.2f}", flush=True)
    print(f"[runner] 落盘：{rep_dir}/league_*.pkl、{out}/solo_state.json、{jp}", flush=True)


if __name__ == "__main__":
    main()
