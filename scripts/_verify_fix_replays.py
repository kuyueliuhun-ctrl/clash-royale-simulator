# -*- coding: utf-8 -*-
"""临时验证（下划线前缀 = 不进正式仪器表）：用**修复后**的代码跑评估路径并落**真录像**。

目的：让修复闭环经过**同一台仪器**——本脚本产出的 `league_*.pkl` 直接喂给
`scripts/pass_streak_audit.py`（就是当初发现 B 类的仪器），期望 **B 类 = 0 帧**。

路径与种子严格照 `train_solo.py::_eval_worker_main`：
  RLEnv(seed=worker_id+777) → 每局 `env.reset(seed=seed_base+2000+g)` →
  `BeliefInference(seed=seed_base+g / seed_base+1000+g)` → `main.act(..., deterministic=True)`。
⚠️ 与真实并行评估的唯一差别：初始**牌序**（父进程 env 的当前牌序不可知）。

用法（必须在 src/clasher_new 下运行）：
    cd src/clasher_new && PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe \
        ../../scripts/_verify_fix_replays.py --ckpt runs/et_solo100k/solo_main_8000.pt \
        --seed-base 8000 --n-total 20 --out docs/_fix_verify_replays
"""
from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src", "clasher_new")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from rl.io_bootstrap import force_utf8_stdout  # noqa: E402  (T1-1: UTF-8 兜底单一来源)
force_utf8_stdout()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/et_solo100k/solo_main_8000.pt")
    ap.add_argument("--seed-base", type=int, default=8000)
    ap.add_argument("--worker-id", type=int, default=4)
    ap.add_argument("--n-total", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=360)
    ap.add_argument("--out", default="docs/_fix_verify_replays")
    ap.add_argument("--step", type=int, default=8000)
    a = ap.parse_args()

    import torch
    torch.set_num_threads(1)

    from rl.env_wrapper import RLEnv
    from rl.follower import FollowerPolicy
    from rl.plan_space import PLAN_DIM
    from rl.belief import BeliefInference
    from rl.belief_planner import BeliefPlanner
    from rl.train_follower import FollowerOpponent
    from rl.action_mask import validate_bundle
    from rl.run_league import LeagueGameRecorder, _bundle_cards
    from rl.replay import save_league_replays
    from rl import train_solo as ts

    ckpt = a.ckpt if os.path.isabs(a.ckpt) else os.path.join(_SRC, a.ckpt)
    sd = torch.load(ckpt, map_location="cpu")
    meta = {k: v for k, v in sd.items() if k != "state_dict"}
    sd = sd.get("state_dict", sd)
    env = RLEnv(opponent=None, seed=a.worker_id + 777, card_level=11,
                deck0=list(ts.DEFAULT_SOLO_DECK), deck1=list(ts.DEFAULT_SOLO_DECK))
    bd = len(BeliefInference(opp_deck=env.deck1, n_particles=128, seed=0).encode(None, None))
    kw = dict(hidden=int(meta.get("hidden_dim", 128)), plan_dim=PLAN_DIM, belief_dim=bd,
              value_bypass=bool(meta.get("value_bypass", False)),
              value_independent=bool(meta.get("value_independent", False)))
    pol, opp = FollowerPolicy(**kw), FollowerPolicy(**kw)
    pol.load_state_dict(sd)
    opp.load_state_dict(sd)
    pol.eval().to_device("cpu")
    opp.eval().to_device("cpu")
    bp = BeliefPlanner()
    games, n_frames, n_invalid = [], 0, 0
    for g in range(a.n_total):
        opp_side = FollowerOpponent(
            opp, env, belief=BeliefInference(opp_deck=env.deck1, n_particles=128,
                                             seed=a.seed_base + g),
            deterministic=True)
        belief = BeliefInference(opp_deck=env.deck1, n_particles=128,
                                 seed=a.seed_base + 1000 + g)
        obs, _ = env.reset(seed=a.seed_base + 2000 + g)
        belief.reset(env.deck1)
        env.opponent = opp_side
        hidden = None
        rec = LeagueGameRecorder("main", "frozen_copy", "main", a.max_steps,
                                 steps=(a.step, None), decks=(env.deck0, env.deck1))
        steps, done = 0, False
        while not done and steps < a.max_steps:
            plan = bp.plan(env.battle, belief.state(), obs)
            tok = belief.encode(obs, None)
            bundle, _lp, _v, hidden, _m = pol.act(obs, tok, plan.to_vector(),
                                                  env.get_action_mask, hidden=hidden,
                                                  deterministic=True)
            ok, reason, _res = validate_bundle(env.battle, 0, bundle)
            if not ok:
                n_invalid += 1
                print(f"  [异常] game={g} frame={steps} validate 拒绝: {reason}")
            played = _bundle_cards(bundle, obs)
            obs, reward, term, trunc, info = env.step(bundle)
            rec.record(env, bundle, reward, info, cards=played)
            opp_side.observe_opponent_played(played)
            belief.update(obs, info.get("opp_played"))
            done = term or trunc
            steps += 1
            n_frames += 1
        games.append(rec.done(env.battle.winner))
    out_dir = a.out if os.path.isabs(a.out) else os.path.join(_ROOT, a.out)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"league_{a.step}.pkl")
    save_league_replays(games, path)
    print(f"[verify] 局数 {len(games)} ｜ 帧数 {n_frames} ｜ validate 拒绝帧 {n_invalid}")
    print(f"[verify] 录像已落盘: {os.path.relpath(path, _ROOT)}")


if __name__ == "__main__":
    main()
