"""cycling 取证（2026-09-12，计划 v3 §3.8.3 A′）。

背景：`fix_gru_ln_norm_20k`（grid_ln）的 `vs baseline0`（训练起点随机策略）崩塌到
0.05~0.125，而 `vs baseline_prev` / `vs 冻结副本` 高达 0.80/0.85 —— 非传递性/cycling 指纹。
本脚本判定该崩塌是：
  (a) 绝对变弱（main 真打不过随机），还是
  (b) 随机对手触发僵局早停 → 按塔血/皇冠裁定的假象，还是
  (c) 真·非传递循环（策略序列互克）。

做法（只读训练产物，不改模型/训练代码）：
  对阵 1: main@N  vs  baseline0（= solo_main_0.pt，训练起点随机策略）
  对阵 2: main@N  vs  全新随机策略（另一随机种子）
  对阵 3: baseline0 vs 全新随机策略（随机对随机的公平性 sanity，期望 ≈0.5）
  参考:   解析 run 自带 replays/league_<N>.pkl（main vs 冻结副本）
对每场输出：胜率 / 疑似早停占比（帧数 < max_ep_steps 且末帧双方皇冠都 <3）/
平均与极值帧数。

用法：
  python scripts/_forensics_cycling.py --run-dir runs/fix_gru_ln_norm_20k --games 100
"""

import argparse
import os
import pickle
import statistics
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src", "clasher_new")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np  # noqa: E402

from rl import train_solo  # noqa: E402
from rl.config import TrainConfig  # noqa: E402
from rl.follower import FollowerPolicy, load_checkpoint  # noqa: E402
from rl.plan_space import PLAN_DIM  # noqa: E402
from rl.belief import BeliefInference  # noqa: E402


def _analyze_pkl(path):
    with open(path, "rb") as f:
        games = pickle.load(f)["games"]
    n = len(games)
    w0 = sum(1 for g in games if g["winner"] == 0)
    w1 = sum(1 for g in games if g["winner"] == 1)
    dr = n - w0 - w1
    lens = [len(g["frames"]) for g in games]
    max_steps = (games[0]["meta"].get("max_steps") or 360) if games else 360
    early = 0
    for g in games:
        if len(g["frames"]) >= max_steps:
            continue
        last = g["frames"][-1]
        c0, c1 = last.get("crown0"), last.get("crown1")
        if c0 is None or c1 is None or (c0 < 3 and c1 < 3):
            early += 1
    return {
        "n": n, "w0": w0, "w1": w1, "draw": dr,
        "winrate0": (w0 + 0.5 * dr) / n,
        "early": early, "early_frac": early / n,
        "f_mean": round(statistics.mean(lens), 1),
        "f_min": min(lens), "f_max": max(lens),
    }


def _run_matchup(cfg, main, opp, label, games, workers, tmpdir, seed):
    env = train_solo.solo_env(cfg, seed)
    step = 90001  # 独立 step，避免与 run 自带 league_<step>.pkl 撞名
    stats, _ = train_solo.eval_solo_parallel(
        env, main, opp, games, int(cfg.max_ep_steps), seed, cfg,
        n_workers=workers, record_replays=True, replays_dir=tmpdir,
        step=step, frozen_step=None, save_replays=True)
    rep = _analyze_pkl(os.path.join(tmpdir, f"league_{step}.pkl"))
    print(f"[{label}] 胜率(main侧)={rep['winrate0']:.3f} "
          f"({rep['w0']}W/{rep['w1']}L/{rep['draw']}D, n={rep['n']}) "
          f"疑似早停={rep['early']}/{rep['n']} ({rep['early_frac']:.0%}) "
          f"帧数 mean={rep['f_mean']} min={rep['f_min']} max={rep['f_max']}", flush=True)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/fix_gru_ln_norm_20k")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    cfg = TrainConfig.load(os.path.join(a.run_dir, "config.json"))
    cfg.eval_workers = a.workers
    main = load_checkpoint(os.path.join(a.run_dir, "solo_main.pt"),
                           hidden_dim=cfg.hidden_dim)
    main.to_device(a.device)
    baseline0 = load_checkpoint(os.path.join(a.run_dir, "solo_main_0.pt"),
                                hidden_dim=cfg.hidden_dim)
    baseline0.to_device(a.device)
    bd = len(BeliefInference(opp_deck=list(train_solo.DEFAULT_SOLO_DECK),
                             n_particles=128, seed=0).encode(None, None))
    fresh = FollowerPolicy(hidden=cfg.hidden_dim, plan_dim=PLAN_DIM,
                           belief_dim=bd).to_device(a.device)

    print(f"run={a.run_dir} games={a.games} workers={a.workers} "
          f"max_ep_steps={cfg.max_ep_steps}", flush=True)

    # 参考：run 自带主评估回放（main vs 冻结副本）
    ref = os.path.join(a.run_dir, "replays", f"league_{cfg.total_steps}.pkl")
    if os.path.exists(ref):
        rep = _analyze_pkl(ref)
        print(f"[main vs 冻结副本(自带回放)] 胜率={rep['winrate0']:.3f} "
              f"({rep['w0']}W/{rep['w1']}L/{rep['draw']}D) 疑似早停={rep['early']}/{rep['n']} "
              f"帧数 mean={rep['f_mean']}", flush=True)

    with tempfile.TemporaryDirectory(prefix="fc_") as td:
        seed = int(cfg.seed) + 70000
        _run_matchup(cfg, main, baseline0, "main@N vs baseline0(起点随机)",
                     a.games, a.workers, td, seed)
        _run_matchup(cfg, main, fresh, "main@N vs 全新随机",
                     a.games, a.workers, td, seed + 1000)
        _run_matchup(cfg, baseline0, fresh, "baseline0 vs 全新随机 (sanity)",
                     a.games, a.workers, td, seed + 2000)
    print("[done] 取证完成，产物见上方逐行输出", flush=True)


if __name__ == "__main__":
    main()
