# -*- coding: utf-8 -*-
"""把 `VanguardX101/IL_Replay` 的 parquet 分片里若干局导成 JSON（供引擎侧重放探针使用）。

⚠️ **依赖 pyarrow** ⇒ 必须用 **WSL 侧 `/usr/bin/python3`**（实测 pyarrow 25.0.1 + numpy 2.5.2）；
Windows venv 与 `E:\\Python313` **都没有** pyarrow/pandas（实测），故本脚本刻意只用 pyarrow。

用法（例）:
    /usr/bin/python3 scripts/_il_replay_extract.py \
        --actions /tmp/il_actions.parquet --replays /tmp/il_replays.parquet \
        --n 3 --out docs/il_replay_probe_2026-09-19/one_shard.json

输出结构（供 `scripts/replay_feasibility_probe.py` 消费）:
    {"source": {...文件与行数...},
     "replays": [{"replay_tag":..., "result":..., "crowns":...,
                  "decks": {"team":[{"key","level"}...], "opponent":[...]},
                  "final_tower_hitpoints": {...},
                  "aggregate_elixir": {...}, "events": [ {...够重放的最小字段...} ]}]}

**只读**：不写除 --out 之外的任何文件；不联网。
"""
import argparse
import collections
import json
import os
import sys


def _load(p):
    import pyarrow.parquet as pq
    return pq.read_table(p).to_pydict()


EVENT_KEYS = ("event_index", "kind", "side", "card_key", "card_play_number",
              "replay_tick_20hz", "time_seconds", "native_x", "native_y",
              "grid_x", "grid_y", "form_at_play", "ability_source_authoritative")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--actions", required=True)
    ap.add_argument("--replays", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--tag", default=None, help="指定 replay_tag（缺省取事件最多的 n 局）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    act = _load(args.actions)
    rep = _load(args.replays)
    n_rows = len(act["replay_tag"])
    by_tag = collections.Counter(act["replay_tag"])

    if args.tag:
        tags = [args.tag]
    else:
        tags = [t for t, _ in by_tag.most_common(args.n)]

    rep_index = {}
    for i, t in enumerate(rep["replay_tag"]):
        rep_index.setdefault(t, i)

    replays = []
    for tag in tags:
        if tag not in rep_index:
            print(f"[skip] tag {tag} not found in replays parquet")
            continue
        i = rep_index[tag]
        payload = json.loads(rep["payload_json"][i])
        # ⚠️ 结构实测（2026-09-19）：卡组/终局塔血在 `battle.<side>.players[0]` 下，
        # **不是** `battle.players[]`（第一版写错，导致 decks 恒为空 ⇒ 探针 ABORT）。
        battle = payload.get("battle", {}) or {}
        decks, finals, tower_card, aggs, raw_players = {}, {}, {}, {}, {}
        for side in ("team", "opponent"):
            blk = battle.get(side) or {}
            pls = blk.get("players") or []
            p = pls[0] if pls else {}
            decks[side] = [{"key": (c.get("card_key") or c.get("key")),
                            "level": c.get("level"), "name": c.get("name")}
                           for c in (p.get("deck") or [])]
            finals[side] = p.get("final_tower_hitpoints")
            tower_card[side] = p.get("tower_card")
            raw_players[side] = {"crowns": blk.get("crowns"),
                                 "average_elixir": p.get("average_elixir"),
                                 "elixir_leaked": p.get("elixir_leaked"),
                                 "four_card_cycle_elixir": p.get("four_card_cycle_elixir")}
            aggs[side] = payload.get("replay", {}).get("aggregate_stats", {}).get(side)
        events = [{k: act[k][j] for k in EVENT_KEYS}
                  for j in range(n_rows) if act["replay_tag"][j] == tag]
        events.sort(key=lambda e: (e["replay_tick_20hz"], e["event_index"]))
        replays.append({
            "replay_tag": tag,
            "result": rep["result"][i],
            "team_crowns": rep["team_crowns"][i],
            "opponent_crowns": rep["opponent_crowns"][i],
            "event_count_declared": rep["event_count"][i],
            "schema_version": rep["schema_version"][i],
            "decks": decks,
            "deck_cards_with_levels": {s: [{"key": c.get("card_key"), "level": c.get("level"),
                                            "name": c.get("name")}
                                           for c in (battle.get(s, {}).get("players", [{}])[0]
                                                     .get("deck") or [])]
                                       for s in ("team", "opponent")},
            "final_tower_hitpoints": finals,
            "tower_card": tower_card,
            "players_meta": raw_players,
            "aggregate_elixir": aggs,
            "duration": payload.get("replay", {}).get("duration"),
            "validation": payload.get("validation"),
            "events": events,
        })

    out = {
        "source": {
            "actions": os.path.basename(args.actions),
            "actions_rows": n_rows,
            "distinct_replays_in_shard": len(by_tag),
            "replays_rows": len(rep["replay_tag"]),
            "kinds": dict(collections.Counter(act["kind"])),
            "picked": tags,
        },
        "replays": replays,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[extract] {len(replays)} replays -> {args.out}")
    for r in replays:
        print(f"  {r['replay_tag']} result={r['result']} crowns={r['team_crowns']}-"
              f"{r['opponent_crowns']} events={len(r['events'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
