# -*- coding: utf-8 -*-
"""把**上游引擎**的中性帧 JSON 转成我们 schema-5 录像（供 dashboard 播放）。

- 输入：`scripts/skarmy_probe_upstream.py --out <json>` 的产物
- 输出：`<run-dir>/replays/league_<帧数>.pkl`（走 `rl.replay.save_league_replays`，schema 5）
        + `<run-dir>/solo_state.json`（dashboard 需要）

⚠️ 只写 `--run-dir`。转换本身**不引入任何新事实**：帧里的每个数值都逐字来自上游引擎的快照。
"""
import argparse
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src", "clasher_new")
sys.path.insert(0, SRC)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args(argv)

    from rl.replay import save_league_replays, LEAGUE_REPLAY_SCHEMA

    with io.open(args.json, encoding="utf-8") as fh:
        blob = json.load(fh)

    games, hist, frames_total = [], [], 0
    for g in blob["games"]:
        frames_total += len(g["frames"])
        games.append({"meta": g["meta"], "winner": g["winner"], "frames": g["frames"]})
        hist.append({
            "step": int(frames_total),
            "games": 1,
            "wins": int(g["winner"] == 0), "losses": int(g["winner"] == 1),
            "draws": int(g["winner"] is None),
            "cum_games": len(games),
            "cum_wins": sum(int(x["winner"] == 0) for x in games),
            "placement": g["name"], "world": g["world"],
            "deploy_ok": g["deploy_ok"], "engine": blob["engine"],
            "end_time_s": g["end_time_s"],
        })

    rep_dir = os.path.join(args.run_dir, "replays")
    os.makedirs(rep_dir, exist_ok=True)
    for fn in os.listdir(rep_dir):
        if fn.startswith("league_") and fn.endswith(".pkl"):
            os.remove(os.path.join(rep_dir, fn))
    rp = os.path.join(rep_dir, "league_%d.pkl" % frames_total)
    save_league_replays(games, rp)

    state = {
        "mode": "solo",
        "agents": [{"agent_id": "skarmy-upstream", "kind": "main", "path": None}],
        "history": hist,
        "total_steps": int(frames_total),
        "target_steps": int(frames_total),
        "deck": blob["games"][0]["meta"]["decks"][0],
        "opponent": "silent(no-op)",
        "copy_every": 0,
        "status": "done",
        "demo": False,
        "_runner": {"kind": "skarmy_probe_upstream", "engine": blob["engine"],
                    "dt": blob["dt"], "frames_per_record": blob["frames_per_record"],
                    "placements": blob["placements"]},
    }
    with io.open(os.path.join(args.run_dir, "solo_state.json"), "w", encoding="utf-8") as fh:
        json.dump(state, fh, ensure_ascii=False, indent=2)

    print("[convert] schema=%d  局=%d  帧=%d  -> %s" % (LEAGUE_REPLAY_SCHEMA, len(games),
                                                        frames_total, rp))
    print("[convert] dashboard: rl/dashboard.py --solo %s --replays %s --port 8701"
          % (args.run_dir, rep_dir))


if __name__ == "__main__":
    main()
