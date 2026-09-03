"""导出对局 replay（规划文档 8.1 / scripts/export_replay.py）。

以随机/启发式动作采集对局，保存含特权隐藏状态标签的 replay，供信念监督训练。
产物为 EpisodeReplay 容器列表（含 schema 与 deck），可被 train_belief --replays-path 消费。
"""

import os
import sys
import random
import argparse
import pickle

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from rl.env_wrapper import RLEnv
from rl.replay import EpisodeReplay
from rl.action_bundle import ActionBundle, K_MAX
from rl.train_follower import heuristic_opponent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-games", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="replays.pkl")
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--opponent", type=str, default=None, choices=[None, "random", "heuristic"])
    args = ap.parse_args()

    env = RLEnv(opponent=None, seed=args.seed)
    rng = random.Random(args.seed)
    if args.opponent == "heuristic":
        env.opponent = heuristic_opponent(env, rng)
    replays = []
    for g in range(args.n_games):
        obs, _ = env.reset()
        ep = EpisodeReplay(record_hidden=True)
        ep.start()
        done = False
        steps = 0
        while not done and steps < args.max_steps:
            mask = env.get_action_mask()
            bundle = ActionBundle()
            slots = [i for i in range(K_MAX) if mask["slots"][i]]
            if slots:
                slot = rng.choice(slots)
                cells = [c for c in range(32 * 18) if mask["cells"][slot][c // 18, c % 18]]
                if cells:
                    cell = rng.choice(cells)
                    bundle.add(slot + 1, cell % 18, cell // 18)
            obs, r, term, trunc, info = env.step(bundle)
            ep.record_step(obs, bundle, r, info)
            done = term or trunc
            steps += 1
        replays.append({**ep.end(), "deck": list(env.deck1)})
        print(f"[replay] game {g+1}/{args.n_games} steps={steps}", flush=True)

    with open(args.out, "wb") as f:
        pickle.dump(replays, f)
    print(f"[save] {args.out} ({len(replays)} games)")


if __name__ == "__main__":
    main()
