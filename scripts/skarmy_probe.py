# -*- coding: utf-8 -*-
"""骷髅军团定点投放探针（用户指定，2026-09-19）：国王塔后 / 桥头。

目的：给 dashboard 一段**可逐帧观看**的录像，让人类直接观测「同一种部队放在不同起点」时
引擎的走位/绕行/过桥/挤桥行为。

做法：
- 三个投放点各一局（**只有投放点不同**，其余逐字相同）：
    behind_king   = 玩家本地 (9, 0) → 世界 (9.5, 0.5)   蓝方王塔（y∈[1,5], x∈[7,11]）**身后**那一行
    bridge_left   = 玩家本地 (3, 13) → 世界 (3.5, 13.5)  左桥桥头（我方一侧）
    bridge_right  = 玩家本地 (14, 13) → 世界 (14.5, 13.5) 右桥桥头（我方一侧）
- 对手 **静音**（`ActionBundle([])`）⇒ 观测到的位移全部来自我方骷髅军团自己。
- 第 0 个决策步投放，之后不再出牌；跑 `--max-steps` 个决策步（默认 120 = 60 s）。
- 录像走**生产同一写入路径**（`LeagueGameRecorder` → `save_league_replays`，**schema 5**），
  并由本脚本另写 `solo_state.json`，于是 **dashboard 可直接打开**：
      .venv/Scripts/python.exe rl/dashboard.py --solo runs/skarmy_probe \\
          --replays runs/skarmy_probe/replays --host 127.0.0.1 --port 8700

⚠️ 本脚本**只写 `--out` 目录**（replays/ + solo_state.json + stats.json），不碰训练/权重/其他录像。
"""
import argparse
import io
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src", "clasher_new")
#: 引擎数据文件按 cwd 解析 ⇒ 先记住调用者 cwd（本仓踩过这个坑）。
ORIG_CWD = os.getcwd()
sys.path.insert(0, SRC)
os.chdir(SRC)

from rl.io_bootstrap import force_utf8_stdout  # noqa: E402  (T1-1: UTF-8 兜底单一来源)
force_utf8_stdout()

#: 世界坐标（= 校验 / 打印用）；bundle 走「玩家本地坐标」，由 `sub_position` 换算。
PLACEMENTS = [
    ("behind_king",  9, 0,  (9.5, 0.5),  "王塔身后（蓝王塔 y∈[1,5]、x∈[7,11] 的后方那一行）"),
    ("bridge_left",  3, 13, (3.5, 13.5), "左桥桥头（我方一侧，河岸前一格）"),
    ("bridge_right", 14, 13, (14.5, 13.5), "右桥桥头（我方一侧，河岸前一格）"),
]

DECK0 = ["SkeletonArmy", "Knight", "Arrows", "Fireball",
         "Musketeer", "Giant", "Minions", "MiniPekka"]
DECK1 = ["Knight", "Arrows", "Fireball", "Musketeer",
         "Giant", "Minions", "MiniPekka", "Skeletons"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "src", "clasher_new", "runs", "skarmy_probe"))
    ap.add_argument("--max-steps", type=int, default=600,
                    help="决策步数（默认 600 = 60 s @ 0.1 s/步）")
    ap.add_argument("--decision-frames", type=int, default=6,
                    help="每个决策步推进的引擎帧数（6 = 0.1 s；只改采样密度，不改引擎动力学）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default=None, help="只跑某个投放点（逗号分隔）")
    ap.add_argument("--no-lane-offset", action="store_true",
                    help="消融实验：强制 Troop._lane_offset = 0（单变量；默认关 = 旧行为逐字不变）")
    args = ap.parse_args(argv)

    from rl.env_wrapper import RLEnv
    from rl.action_bundle import ActionBundle, SubAction
    from rl import train_solo as ts
    from rl.run_league import LeagueGameRecorder, _bundle_cards, timeout_winner
    from rl.replay import save_league_replays
    from rl.overtime import overtime_open

    if args.no_lane_offset:
        # 消融：外部猴补丁，**不改引擎源码**。只把「队形车道偏移」置零，其余逐字不变。
        # 判据与失败分支见 docs/skarmy_two_engine_observations_2026-09-19.md §5（跑前写死）。
        import battle as _battle
        _orig_troop_init = _battle.Troop.__init__

        def _troop_init_no_lane(self, *a, **kw):
            _orig_troop_init(self, *a, **kw)
            self._lane_offset = 0.0

        _battle.Troop.__init__ = _troop_init_no_lane
        print("[skarmy] 消融已生效：Troop._lane_offset 强制 0（只改这一项）", flush=True)

    out = args.out
    rep_dir = os.path.join(out, "replays")
    os.makedirs(rep_dir, exist_ok=True)
    # 清掉旧录像，避免 dashboard 按 league_*.pkl 通配重复计数
    for fn in os.listdir(rep_dir):
        if fn.startswith("league_") and fn.endswith(".pkl"):
            os.remove(os.path.join(rep_dir, fn))

    wanted = None if not args.only else {p.strip() for p in args.only.split(",")}
    picks = [p for p in PLACEMENTS if wanted is None or p[0] in wanted]

    print("[skarmy] out=%s  placements=%s  max_steps=%d (每步 %.1f s，共 %.0f s)"
          % (out, [p[0] for p in picks], args.max_steps,
             args.decision_frames / 60.0, args.max_steps * args.decision_frames / 60.0), flush=True)

    games, hist = [], []
    t0 = time.time()
    frames_total = 0

    for idx, (name, lx, ly, world, note) in enumerate(picks):
        env = RLEnv(opponent=lambda obs: ActionBundle([]),   # 静音对手
                    seed=args.seed + 17 * idx, card_level=11,
                    decision_frames=args.decision_frames,    # 只改采样密度
                    deck0=list(DECK0), deck1=list(DECK1))
        obs, _ = env.reset(seed=args.seed + 100 + idx)
        # slot=1 必须是我们想放的那张牌
        cyc = list(env.battle.players[0].cycle)
        if "SkeletonArmy" not in cyc:
            cyc[0] = "SkeletonArmy"
        else:
            cyc.remove("SkeletonArmy")
            cyc.insert(0, "SkeletonArmy")
        env.battle.players[0].cycle = cyc

        rec = LeagueGameRecorder("skarmy@%s" % name, "silent", "skarmy@%s" % name, args.max_steps)
        rec.set_decks(env.deck0, env.deck1)

        deploy_step = 0
        done = False
        steps = 0
        accepted = None
        deploy_log = []
        while not done and (steps < args.max_steps or overtime_open(env.battle)):
            if steps == deploy_step:
                bundle = ActionBundle([SubAction(kind="deploy", slot=1, x=lx, y=ly)])
            else:
                bundle = ActionBundle([])
            cards = _bundle_cards(bundle, obs)
            obs, reward, term, trunc, info = env.step(bundle)
            if steps == deploy_step:
                alive = [e for e in env.battle.entities.values()
                         if (getattr(e, "name", "") or "").startswith("Skeleton")]
                accepted = info.get("invalid_count", 0) == 0 and bool(alive)
                deploy_log.append({"step": steps, "world": world, "alive_skeletons": len(alive),
                                   "invalid_count": info.get("invalid_count")})
                print("  [%s] 投放 world=(%.1f,%.1f)  新骷髅=%d  invalid_count=%s"
                      % (name, world[0], world[1], len(alive), info.get("invalid_count")), flush=True)
            rec.record(env, bundle, reward, info, cards=cards)
            frames_total += 1
            done = term or trunc
            steps += 1
            if steps % 50 == 0:
                pos = [(round(e.position.x, 1), round(e.position.y, 1))
                       for e in env.battle.entities.values()
                       if (getattr(e, "name", "") or "").startswith("Skeleton")]
                print("  [%s] step %d: 存活骷髅 %d  位置样本 %s"
                      % (name, steps, len(pos), pos[:6]), flush=True)

        w = env.battle.winner
        if w is None and not env.battle.game_over:
            w = timeout_winner(env.battle)
        games.append(rec.done(w))
        hist.append({
            "step": int(frames_total),                 # 【R9】step = 累计决策帧
            "games": 1, "wins": int(w == 0), "losses": int(w == 1), "draws": int(w is None),
            "cum_games": idx + 1, "cum_wins": sum(int(g.get("winner") == 0) for g in games),
            "placement": name, "note": note, "world": list(world),
            "deploy_ok": accepted, "deploy_log": deploy_log,
            "end_time_s": round(float(env.battle.time), 2),
        })
        print("  [%s] 结束: steps=%d  end_time=%.1fs  winner=%s  用时 %.1fs"
              % (name, steps, env.battle.time, w, time.time() - t0), flush=True)

    rp = os.path.join(rep_dir, "league_%d.pkl" % int(frames_total))
    save_league_replays(games, rp)

    state = {
        "mode": "solo",
        "agents": [{"agent_id": "skarmy", "kind": "main", "path": None}],
        "history": hist,
        "total_steps": int(frames_total),
        "target_steps": int(frames_total),
        "deck": list(DECK0),
        "opponent": "silent(no-op)",
        "copy_every": 0,
        "status": "done",
        "demo": False,
        "_runner": {"kind": "skarmy_probe", "placements": [p[0] for p in picks],
                    "seed": args.seed, "max_steps": args.max_steps},
    }
    with open(os.path.join(out, "solo_state.json"), "w", encoding="utf-8") as fh:
        json.dump(state, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(out, "stats.json"), "w", encoding="utf-8") as fh:
        json.dump({"frames": frames_total, "seconds": round(time.time() - t0, 1),
                   "games": hist}, fh, ensure_ascii=False, indent=2)

    print("[skarmy] 落盘: %s  (%d 局 / %d 帧 / %.0fs)"
          % (rp, len(games), frames_total, time.time() - t0), flush=True)
    print("[skarmy] dashboard: rl/dashboard.py --solo %s --replays %s --port 8700"
          % (out, rep_dir), flush=True)


if __name__ == "__main__":
    main()
