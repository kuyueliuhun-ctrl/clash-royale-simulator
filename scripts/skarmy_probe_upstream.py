# -*- coding: utf-8 -*-
"""骷髅军团定点投放探针 —— **上游引擎侧**（2026-09-19，用户指定）。

与 `scripts/skarmy_probe.py`（我方引擎）**逐项对齐**：同样的三个投放点、静音对手、
0.1 s 一帧、60 s 一局；差别只有引擎实现。

运行姿势（上游有自己的依赖与 cwd 契约，实测唯一缺件是 fastcore）：
    /usr/bin/python3 -m pip install --quiet --target /tmp/deps_upstream fastcore
    cd /mnt/e/clash-royale-simulator-main-by-jason/src/clasher_new      # card_utils 用相对路径 open()
    PYTHONPATH=/tmp/deps_upstream PYTHONDONTWRITEBYTECODE=1 \\
        /usr/bin/python3 -B /mnt/e/clash-royale-simulator-main/scripts/skarmy_probe_upstream.py \\
        --out /tmp/skarmy_upstream.json

输出 = **中性 JSON**（不含我们仓库的任何格式），交给
`scripts/convert_upstream_frames.py` 转成我们 schema-5 录像，供 dashboard 播放。

⚠️ 只写 `--out` 那一个文件。不改上游仓库任何文件。
"""
import argparse
import json
import os
import sys

_OUR_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "src", "clasher_new")
if _OUR_SRC not in sys.path:
    sys.path.append(_OUR_SRC)   # append（不是 insert）：不得抢上游引擎 `import battle` 的优先级
from rl.io_bootstrap import force_utf8_stdout  # noqa: E402
force_utf8_stdout()
#: 本脚本在**我方**仓库里、却要在**上游**的 cwd 下跑 ⇒ Python 的 sys.path[0] 是脚本目录，
#: 不是 cwd ⇒ 必须显式把 cwd 放进来，否则 `import battle` 找不到（实测踩过）。
sys.path.insert(0, os.getcwd())

PLACEMENTS = [
    ("behind_king",  (9.5, 0.5)),
    ("bridge_left",  (3.5, 13.5)),
    ("bridge_right", (14.5, 13.5)),
]
DECK0 = ["SkeletonArmy", "Knight", "Arrows", "Fireball",
         "Musketeer", "Giant", "Minions", "MiniPekka"]
DECK1 = ["Knight", "Arrows", "Fireball", "Musketeer",
         "Giant", "Minions", "MiniPekka", "Skeletons"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames-per-record", type=int, default=6, help="每帧推进的引擎 tick 数")
    ap.add_argument("--dt", type=float, default=1.0 / 60.0,
                    help="引擎 tick（**默认与我们的探针一致 1/60**，以隔离『引擎代码』这一个变量；"
                         "上游生产环境的 environment.py 用的是 1/20）")
    ap.add_argument("--max-frames", type=int, default=600, help="帧数（600 帧 × 0.1 s = 60 s）")
    args = ap.parse_args(argv)

    import battle as B
    import player as P
    from core import Position
    from card_utils import Card

    #: 上游的实体分类（只为我们快照里的 `kind` 字段，与 `rl/replay.py::_kind` 同口径）
    def kind_of(e):
        if isinstance(e, B.Building):
            return "building"
        if isinstance(e, B.Projectile):
            return "projectile"
        return "troop"

    games = []
    for name, (wx, wy) in PLACEMENTS:
        bs = B.BattleState(P.PlayerState(0, list(DECK0), 5.0),
                           P.PlayerState(1, list(DECK1), 5.0))
        # slot=1（cycle[0]）必须是想投放的那张牌
        cyc = list(bs.players[0].cycle)
        cyc.remove("SkeletonArmy")
        cyc.insert(0, "SkeletonArmy")
        bs.players[0].cycle = cyc
        bs.step(args.dt)                      # 与我们的探针一致：先推进一步再投放
        ok = bool(bs.deploy_card(0, "SkeletonArmy", Position(wx, wy)))
        spawned = sum(1 for e in bs.entities.values()
                      if (getattr(e, "name", "") or "").startswith("Skeleton"))
        print("[up][%s] deploy=%s 世界=(%.1f,%.1f) 新骷髅=%d" % (name, ok, wx, wy, spawned),
              flush=True)

        frames = []
        for i in range(args.max_frames):
            for _ in range(args.frames_per_record):
                if bs.game_over:
                    break
                bs.step(args.dt)
            ents = []
            for e in bs.entities.values():
                if not e.is_alive:
                    continue
                ents.append([
                    e.name, round(float(e.position.x), 1), round(float(e.position.y), 1),
                    round(float(e.hp), 1), int(e.player), kind_of(e),
                    float(getattr(e.data, "hp", e.hp) or e.hp),
                    float(getattr(e, "shield_health", 0.0) or 0.0),
                    float(getattr(e.data, "shield_health", 0.0) or 0.0),
                    float(getattr(e.data, "collision_radius", 0.5) or 0.5),
                    int(e.id),
                    (int(e.target_id) if getattr(e, "target_id", None) is not None else None),
                    None, None, None,
                ])
            p0, p1 = bs.players
            frames.append({
                "t": round(float(bs.time), 2),
                "bundle": ([["deploy", 1, 9, 0]] if i == 0 and name == "behind_king"
                           else [["deploy", 1, int(wx - 0.5), int(wy - 0.5)]] if i == 0 else []),
                "reward": 0.0,
                "opp_played": [],
                "towers0": [float(p0.king_tower_hp), float(p0.left_tower_hp), float(p0.right_tower_hp)],
                "towers1": [float(p1.king_tower_hp), float(p1.left_tower_hp), float(p1.right_tower_hp)],
                "elixir0": float(p0.elixir), "elixir1": float(p1.elixir),
                "crown0": int(p0.get_crown_count()), "crown1": int(p1.get_crown_count()),
                "v0": None, "v1": None,
                "entities": ents,
            })
            if i % 50 == 0:
                sk = [(round(e.position.x, 1), round(e.position.y, 1))
                      for e in bs.entities.values()
                      if (getattr(e, "name", "") or "").startswith("Skeleton") and e.is_alive]
                print("  [up][%s] frame %d t=%.1f 存活骷髅 %d %s"
                      % (name, i, bs.time, len(sk), sk[:6]), flush=True)
            if bs.game_over:
                break

        w = bs.winner
        if w is None and not bs.game_over:
            # 与 `rl/league_rules.timeout_winner` 同口径：三塔血量合计多者胜，相等为平
            s0 = bs.players[0].king_tower_hp + bs.players[0].left_tower_hp + bs.players[0].right_tower_hp
            s1 = bs.players[1].king_tower_hp + bs.players[1].left_tower_hp + bs.players[1].right_tower_hp
            w = None if s0 == s1 else (0 if s0 > s1 else 1)
        alive = sum(1 for e in bs.entities.values()
                    if (getattr(e, "name", "") or "").startswith("Skeleton") and e.is_alive)
        print("  [up][%s] 结束 t=%.1fs winner=%s 存活骷髅=%d 帧=%d"
              % (name, bs.time, w, alive, len(frames)), flush=True)
        games.append({
            "name": name, "world": [wx, wy], "deploy_ok": ok, "spawned": spawned,
            "winner": w, "end_time_s": round(float(bs.time), 2),
            "meta": {"pair": ["skarmy@%s" % name, "silent"], "side0": "skarmy@%s" % name,
                     "max_steps": args.max_frames, "decks": [list(DECK0), list(DECK1)]},
            "frames": frames,
        })

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"engine": "upstream(Jason-XII) f616f19", "dt": args.dt,
                   "frames_per_record": args.frames_per_record,
                   "placements": [p[0] for p in PLACEMENTS], "games": games}, fh)
    print("[up] 写出 %s（%d 局 / %d 帧）" % (args.out, len(games), sum(len(g["frames"]) for g in games)))


if __name__ == "__main__":
    main()
