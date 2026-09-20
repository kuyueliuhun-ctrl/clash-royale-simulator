# -*- coding: utf-8 -*-
"""**IL（行为克隆）策略的对局观测 runner** —— 跑几局、落 schema 5 录像，供 dashboard `--solo` 看/播。

用途（用户 2026-09-20：「给我 dashboard，我看几场 IL 后的对局」）：把一个 BC ckpt
（例如 `runs/_fl_il_bc/bc_fl.pt`，用 FirstLight 的 IL_Replay 训出来的）当成**生产策略**跑真引擎，
逐块落盘**生产同格式录像** + `solo_state.json`，dashboard 边看指标边播回放。

* **p0（被测方）= IL 策略**：`FollowerPolicy.act(...)`，默认 `deterministic=True`（argmax，
  与生产评估的脚本对手同约定）。信念/plan 链路与 `rl/human_play.py:142-154` **逐句同序**
  （`BeliefInference.encode` → `BeliefPlanner.plan` → `act`）。
* **p1（对手）= 同一 ckpt 的独立冻结副本**（自对弈镜像），或 `--opponent main --main-ckpt <我方 ckpt>`。
* **卡组**：`--decks solo` = `rl.train_solo.DEFAULT_SOLO_DECK`（与既有评估可比）；
  `--decks fl` = 从 FL 的 IL_Replay JSONL 里抽**真实人类卡组对**（过滤口径与
  `scripts/fl_il_to_bc.py` 的 `--mode samples` **逐条一致**：模式白名单 + 时长 ≤300 s + 两侧 8 张全部可解析）。
* `--hidden {carry,none}`：**carry**（默认）= 生产约定，跨帧带 GRU 隐状态；
  **none** = 每条帧都当首帧（= BC 训练时的口径，`human_play.py:224`）。

⚠️ 描述性读数、非判据（【R3】【R5】：n=1 ckpt、单 seed）。本 runner **不改任何训练/奖励/判定代码**。

用法（必须在 `src/clasher_new` 下运行，或从仓库根用绝对路径）:
    cd src/clasher_new && ../../.venv/Scripts/python.exe ../../scripts/il_readout_games.py \
        --ckpt ../../runs/_fl_il_bc/bc_fl.pt --decks solo --games 10 --block 5 \
        --out ../../runs/il_readout_solo --seed 0
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src", "clasher_new")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from rl.io_bootstrap import force_utf8_stdout  # noqa: E402

force_utf8_stdout()

import numpy as np  # noqa: E402

#: 与 `scripts/fl_il_to_bc.py` 同口径（改这里必须同步那两处）
ALLOWED_MODES = ("Ranked", "Ladder", "1v1 Battle", "1v1",
                 "Grand Challenge", "Classic Challenge")
VARIANTS = ("-ev1", "-ev2", "-ev3", "-hero")
TIME_CAP = 300.0


def _map_key(key):
    from card_aliases import resolve_card
    k = key or ""
    for v in VARIANTS:
        if k.endswith(v):
            k = k[: -len(v)]
    try:
        return resolve_card(k) if k else None
    except Exception:  # noqa: BLE001
        return None


def load_fl_deck_pairs(jsonl, limit=4000):
    """从 `_fl_il_extract.py` 产出的 JSONL 里取**可用的人类卡组对**（过滤口径同 fl_il_to_bc）。"""
    pairs, seen = [], set()
    with open(jsonl, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            if not line.strip():
                continue
            r = json.loads(line)
            if r["gm"] not in ALLOWED_MODES:
                continue
            if not isinstance(r["dur"], (int, float)) or r["dur"] > TIME_CAP:
                continue
            decks, bad = {}, False
            for side in ("team", "opponent"):
                names = [_map_key(k) for k, _lv in r["decks"][side]]
                if any(n is None for n in names) or len(names) != 8:
                    bad = True
                    break
                decks[side] = names
            if bad:
                continue
            key = tuple(decks["team"]) + tuple(decks["opponent"])
            if key in seen:
                continue
            seen.add(key)
            pairs.append((decks["team"], decks["opponent"], r["tag"]))
    return pairs


def build_policy(ckpt, device, torch, label):
    from rl.follower import FollowerPolicy
    d = torch.load(ckpt, map_location="cpu")
    sd = d["state_dict"] if isinstance(d, dict) and "state_dict" in d else d
    meta = {k: v for k, v in d.items() if k != "state_dict"} if isinstance(d, dict) else {}
    pol = FollowerPolicy(hidden=int(meta.get("hidden_dim", 128)),
                         plan_dim=int(meta["plan_dim"]), belief_dim=int(meta["belief_dim"]),
                         value_bypass=bool(meta.get("value_bypass", False)),
                         value_independent=bool(meta.get("value_independent", False)),
                         intent_options=bool(meta.get("intent_options", False)))
    miss, unexp = pol.load_state_dict(sd, strict=False)
    if miss or unexp:
        print(f"[warn] {label} 载入 missing={len(miss)} unexpected={len(unexp)}", flush=True)
    pol.eval()
    pol.to_device(device)
    return pol, meta


def main():
    ap = argparse.ArgumentParser(description="IL 策略对局观测 runner（落 schema 5 录像供 dashboard）")
    ap.add_argument("--ckpt", required=True, help="IL（BC）ckpt")
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--block", type=int, default=5, help="每多少局落一次盘")
    ap.add_argument("--out", required=True, help="run 目录（写 replays/ + solo_state.json）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--decks", choices=("solo", "fl"), default="solo")
    ap.add_argument("--fl-jsonl", default=None,
                    help="--decks fl 时的回放 JSONL（Windows 路径，如 E:\\fl_il_data\\replays_part000000.jsonl）")
    ap.add_argument("--opponent", choices=("self", "main"), default="self")
    ap.add_argument("--main-ckpt", default=None, help="--opponent main 时我方 ckpt")
    ap.add_argument("--hidden", choices=("carry", "none"), default="carry")
    ap.add_argument("--sample", action="store_true", help="随机采样而非 argmax（默认 argmax）")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    import torch
    from rl.env_wrapper import RLEnv
    from rl.belief import BeliefInference
    from rl.belief_planner import BeliefPlanner
    from rl import train_solo as ts
    from rl.train_follower import FollowerOpponent
    from rl.run_league import LeagueGameRecorder, _bundle_cards, timeout_winner
    from rl.replay import save_league_replays
    from rl.overtime import overtime_open

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = a.out
    os.makedirs(os.path.join(out, "replays"), exist_ok=True)

    pol, meta = build_policy(a.ckpt, dev, torch, "IL 策略")
    if a.opponent == "main":
        if not a.main_ckpt:
            raise SystemExit("--opponent main 需要 --main-ckpt")
        opp_pol, _ = build_policy(a.main_ckpt, dev, torch, "我方主策略")
    else:
        opp_pol, _ = build_policy(a.ckpt, dev, torch, "对手(同 ckpt 冻结副本)")
    bp = BeliefPlanner()

    deck_pairs = None
    if a.decks == "fl":
        if not a.fl_jsonl or not os.path.isfile(a.fl_jsonl):
            raise SystemExit(f"--decks fl 需要 --fl-jsonl（给的：{a.fl_jsonl}）")
        deck_pairs = load_fl_deck_pairs(a.fl_jsonl)
        if not deck_pairs:
            raise SystemExit("FL JSONL 里没有可用的人类卡组对")
        print(f"[runner] FL 卡组对：{len(deck_pairs)} 套（过滤口径同 fl_il_to_bc）", flush=True)

    print(f"[runner] ckpt={a.ckpt} device={dev} 局数={a.games} 每块={a.block} "
          f"decks={a.decks} opponent={a.opponent} hidden={a.hidden} "
          f"deterministic={not a.sample}", flush=True)

    games, hist = [], []
    cw = cl = cd = frames_total = plays_total = empty_total = 0
    rew_total = 0.0
    t0 = time.time()
    bf = bg = bw = bl = bd = 0
    brew = 0.0
    card_hist = collections.Counter()
    playable = forced_stop = stop_when_playable = 0
    opp_card_hist = collections.Counter()

    for g in range(a.games):
        if deck_pairs is not None:
            d0, d1, tag = deck_pairs[(a.seed + g) % len(deck_pairs)]
            d0, d1 = list(d0), list(d1)
        else:
            d0 = d1 = list(ts.DEFAULT_SOLO_DECK)
            tag = None
        env = RLEnv(opponent=None, seed=a.seed + 31 * g, card_level=11, deck0=d0, deck1=d1)
        env.opponent = FollowerOpponent(
            opp_pol, env,
            belief=BeliefInference(opp_deck=list(env.deck0), n_particles=128, seed=a.seed + g),
            deterministic=True)
        obs, _ = env.reset(seed=a.seed + 2000 + g)
        belief0 = BeliefInference(opp_deck=list(env.deck1), n_particles=128, seed=a.seed + g)
        belief0.reset(env.deck1)
        rec = LeagueGameRecorder("il", "fl@0" if tag is None else "fl@1", "il", a.max_steps)
        rec.set_decks(env.deck0, env.deck1)
        done = False
        steps = 0
        hidden = None
        while not done and (steps < a.max_steps or overtime_open(env.battle)):
            tok = belief0.encode(obs, None)
            plan = bp.plan(env.battle, belief0.state(), obs).to_vector()
            h_in = hidden if a.hidden == "carry" else None
            bundle, _lp, _v, h_out, _mk = pol.act(obs, tok, plan, env.get_action_mask,
                                                  hidden=h_in, deterministic=not a.sample)
            hidden = h_out if a.hidden == "carry" else None
            #: ★ 行为画像的关键分层（与 `scripts/probe_pass_prob.py` 同一口径）：
            #: 「本帧买得起吗」用**掩码**判（`slots` 任一为真 或 `ability_legal`），
            #: 分三类：①被迫不出（唯一合法项 = STOP）②买得起却选择不出 ③真出牌
            _m0 = env.get_action_mask()
            _can = bool(np.any(_m0["slots"])) or bool(_m0.get("ability_legal"))
            playable += int(_can)
            forced_stop += int(not _can)
            cards = _bundle_cards(bundle, obs)
            if _can and bundle.size == 0:
                stop_when_playable += 1
            obs, reward, term, trunc, info = env.step(bundle)
            rec.record(env, bundle, reward, info, cards=cards)
            #: 出牌统计（p0 用 `_bundle_cards`；对手用 env 交回的 `opp_played`）
            for nm in cards:
                card_hist[nm] += 1
            for nm in (info.get("opp_played") or []):
                #: 实测 `opp_played` 的元素是 **dict**（不是卡名字符串）⇒ 取名字段再计数
                if isinstance(nm, dict):
                    nm = nm.get("card") or nm.get("card_name") or nm.get("name")
                if isinstance(nm, str):
                    opp_card_hist[nm] += 1
            belief0.update(obs, info.get("opp_played"))
            frames_total += 1
            brew += float(reward)
            rew_total += float(reward)
            bf += 1
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
        bw += int(w == 0)
        bl += int(w == 1)
        bd += int(w is None)
        bg += 1

        if bg >= a.block or g == a.games - 1:
            n = cw + cl + cd
            hist.append({
                "step": int(frames_total),
                "wins": bw, "losses": bl, "draws": bd, "games": bg,
                "cum_wins": cw, "cum_losses": cl, "cum_draws": cd, "cum_games": n,
                "winrate": cw / max(1, n),
                "winrate_se": float(np.sqrt(max(1e-9, (cw / max(1, n)) * (1 - cw / max(1, n)))
                                             / max(1, n))),
                "mean_reward": rew_total / max(1, frames_total),
                "block_mean_reward": brew / max(1, bf),
                "empty_bundle_rate": empty_total / max(1, frames_total),
                "plays_per_frame": plays_total / max(1, frames_total),
                "playable_frame_rate": playable / max(1, frames_total),
                "stop_when_playable_rate": stop_when_playable / max(1, playable),
            })
            save_league_replays(games, os.path.join(out, "replays",
                                                    f"league_{int(frames_total)}.pkl"))
            with open(os.path.join(out, "solo_state.json"), "w", encoding="utf-8") as fh:
                json.dump({
                    "mode": "solo",
                    "agents": [{"agent_id": "il", "kind": "main", "path": a.ckpt}],
                    "history": hist,
                    "total_steps": int(frames_total),
                    "target_steps": int(frames_total),
                    "deck": list(d0),
                    "opponent": ("self-frozen-copy(il)" if a.opponent == "self"
                                 else f"main:{a.main_ckpt}"),
                    "copy_every": 0,
                    "status": "running" if g < a.games - 1 else "done",
                    "demo": False,
                    "_runner": {"kind": "il_readout", "ckpt": a.ckpt, "decks": a.decks,
                                "opponent": a.opponent, "hidden": a.hidden,
                                "games_planned": a.games, "block": a.block, "seed": a.seed},
                }, fh, ensure_ascii=False, indent=2)
            el = time.time() - t0
            print(f"  [块 {len(hist)}] {n}/{a.games} 局、{frames_total} 帧 | "
                  f"胜 {cw} 负 {cl} 平 {cd} = {100*cw/max(1,n):.1f}% | "
                  f"局均帧 {frames_total/max(1,n):.1f} | 空 bundle "
                  f"{100*empty_total/max(1,frames_total):.1f}% | 买得起帧 "
                  f"{playable}/{frames_total}、其中选择不出 "
                  f"{100*stop_when_playable/max(1,playable):.1f}% | {el:.0f}s", flush=True)
            bf = bg = bw = bl = bd = 0
            brew = 0.0

    n = cw + cl + cd
    summary = {
        "ckpt": a.ckpt, "decks": a.decks, "opponent": a.opponent, "hidden": a.hidden,
        "games": n, "frames": frames_total, "wins": cw, "losses": cl, "draws": cd,
        "winrate": cw / max(1, n),
        "winrate_se": float(np.sqrt(max(1e-9, (cw / max(1, n)) * (1 - cw / max(1, n)))
                                    / max(1, n))),
        "mean_reward_per_frame": rew_total / max(1, frames_total),
        "frames_per_game": frames_total / max(1, n),
        "empty_bundle_rate": empty_total / max(1, frames_total),
        "plays_per_frame": plays_total / max(1, frames_total),
        "playable_frame_rate": playable / max(1, frames_total),
        "playable_frames": playable,
        "forced_stop_frames": forced_stop,
        "stop_when_playable": stop_when_playable,
        "stop_when_playable_rate": stop_when_playable / max(1, playable),
        "p0_plays": plays_total,
        "p0_top_cards": card_hist.most_common(20),
        "p1_top_cards": opp_card_hist.most_common(20),
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
    print(f"[runner] p0 出牌 {plays_total} 次 / 买得起 {playable} 帧（被迫不出 {forced_stop} 帧）"
          f"｜买得起却选择不出 {stop_when_playable} = "
          f"{100*stop_when_playable/max(1,playable):.1f}%", flush=True)
    print(f"[runner] p0 出牌 top: {card_hist.most_common(6)}", flush=True)
    print(f"[runner] 落盘：{out}/replays/league_*.pkl、{out}/solo_state.json、{jp}", flush=True)


if __name__ == "__main__":
    main()
