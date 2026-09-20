# -*- coding: utf-8 -*-
"""把 FirstLight 的 IL 数据集（HF `VanguardX101/IL_Replay`）parquet 分片导成 JSONL + 普查 JSON。

⚠️ **依赖 pyarrow** ⇒ 必须用 **WSL 侧 `/usr/bin/python3`**（实测 pyarrow 25.0.1）；
   Windows venv 与 `E:\\Python313` 都没有 pyarrow（前次调研实测）。
   下游消费方（引擎侧重放 / IL 样本导出）跑在 Windows venv 上，故这里**只做格式转换与普查**，不碰引擎。

为什么只要 `replays/part-*.parquet`：实测 `payload_json` 内部**已经含全部 events**
（顶层 7 键 = battle / collection_schema_version / events / replay / schema_version / source / validation），
`actions/part-*.parquet` 是同一批事件的**冗余展开表**，本脚本用它做**对账**（每个 tag 的事件行数 vs
payload 内 events 长度），不参与下游。

用法:
    /usr/bin/python3 scripts/_fl_il_extract.py \
        --replays /mnt/e/fl_il_data/replays/part-000000.parquet \
        --actions /mnt/e/fl_il_data/actions/part-000000.parquet \
        --out-jsonl /mnt/e/fl_il_data/replays_part000000.jsonl \
        --out-census docs/fl_il_2026-09-20/census_part000000.json

**只读**：除 `--out-jsonl` / `--out-census` 外不写任何文件；不联网。
"""
import argparse
import collections
import json
import os
import sys

KIND_CODE = {"play_card": 0, "activate_ability": 1}
SIDES = ("team", "opponent")


def _load(p):
    import pyarrow.parquet as pq
    return pq.read_table(p)


def _player(payload, side):
    blk = (payload.get("battle") or {}).get(side) or {}
    pls = blk.get("players") or []
    return blk, (pls[0] if pls else {})


def _meta(blk, p):
    fin = p.get("final_tower_hitpoints")
    return {
        "crowns": blk.get("crowns"),
        "avg": p.get("average_elixir"),
        "leak": p.get("elixir_leaked"),
        "cycle4": p.get("four_card_cycle_elixir"),
        "final": (fin if isinstance(fin, dict) else None),
        "tower_card": p.get("tower_card"),
    }


def record(row, idx):
    """一行 parquet → 精简记录（键名刻意短，几万局时体积才可控）。"""
    payload = json.loads(row["payload_json"][idx])
    decks, metas = {}, {}
    for side in SIDES:
        blk, p = _player(payload, side)
        decks[side] = [[c.get("card_key"), c.get("level")] for c in (p.get("deck") or [])]
        metas[side] = _meta(blk, p)
    events = []
    for e in payload.get("events") or []:
        #: ⚠️ **实测（第一版写错）**：payload 的 event 里**没有** `native_x/native_y` 顶层键，
        #: 坐标在 `event["coordinates"]` 里（`native_world_units` / `display_royaleapi_units` /
        #: `raw_royaleapi_units` / `grid_cell_floor`）。当初按 `actions/*.parquet` 的扁平列名写，
        #: 结果坐标全 None ⇒ 下游引擎侧「无坐标事件」计数爆表、0 条标签（实测踩过）。
        co = e.get("coordinates") or {}
        nw = co.get("native_world_units") or {}
        nx, ny = nw.get("x"), nw.get("y")
        gf = co.get("grid_cell_floor") or {}
        gx, gy = gf.get("x"), gf.get("y")
        events.append([
            KIND_CODE.get(e.get("kind"), -1),
            0 if e.get("side") == "team" else 1,
            e.get("card_key"),
            e.get("replay_tick_20hz"),
            nx,
            ny,
            gx,
            gy,
        ])
    events.sort(key=lambda r: (r[3] if r[3] is not None else -1))
    rep = payload.get("replay") or {}
    #: ⚠️ 实测：`replay.duration` **不是数字**而是 dict `{display_label, display_seconds,
    #: timeline_seconds}`（第一版按数字读 ⇒ 普查里 n=0）。取 `timeline_seconds` 为准。
    _dur = rep.get("duration") or {}
    dur = _dur.get("timeline_seconds") if isinstance(_dur, dict) else _dur
    if dur is None and isinstance(_dur, dict):
        dur = _dur.get("display_seconds")
    #: `replay.aggregate_stats[side]` = {类别: [张数, 圣水], "leaked": 泄漏圣水}——
    #: 下游用它做「重建总花费 vs 人类真实总花费」的对账（`total` 那一项）。
    agg = {}
    for side in SIDES:
        blk = (rep.get("aggregate_stats") or {}).get(side) or {}
        tot = blk.get("total")
        agg[side] = {"cards": (tot[0] if isinstance(tot, list) and tot else None),
                     "elixir": (tot[1] if isinstance(tot, list) and len(tot) > 1 else None),
                     "leaked": blk.get("leaked")}
    return {
        "tag": row["replay_tag"][idx],
        "bt": row["battle_type"][idx],
        "gm": row["game_mode"][idx],
        "res": row["result"][idx],
        "tc": row["team_crowns"][idx],
        "oc": row["opponent_crowns"][idx],
        "ec": row["event_count"][idx],
        "wc": row["warning_count"][idx],
        "rpt": row["requested_player_tag"][idx],
        "sv": row["schema_version"][idx],
        "dur": dur,
        "agg": agg,
        "decks": decks,
        "meta": metas,
        "ev": events,
    }


def census(recs, cross_check):
    c = {
        "replays": len(recs),
        "battle_type": dict(collections.Counter(r["bt"] for r in recs)),
        "game_mode": dict(collections.Counter(r["gm"] for r in recs)),
        "result": dict(collections.Counter(r["res"] for r in recs)),
        "schema_version": dict(collections.Counter(r["sv"] for r in recs)),
        "events_total": sum(len(r["ev"]) for r in recs),
        "events_kind": dict(collections.Counter(
            ("play_card" if e[0] == 0 else "activate_ability" if e[0] == 1 else "unknown")
            for r in recs for e in r["ev"])),
        "events_side": dict(collections.Counter(
            SIDES[e[1]] for r in recs for e in r["ev"])),
        "deck_size": dict(collections.Counter(
            len(r["decks"][s]) for r in recs for s in SIDES)),
        "deck_levels": dict(collections.Counter(
            (c2[1] if c2[1] is not None else "null")
            for r in recs for s in SIDES for c2 in r["decks"][s])),
        "warning_count": dict(collections.Counter(r["wc"] for r in recs).most_common(8)),
        "cross_check_events_vs_actions": cross_check,
    }
    dur = sorted(r["dur"] for r in recs if isinstance(r["dur"], (int, float)))
    c["duration_s"] = {
        "n": len(dur),
        "min": dur[0] if dur else None,
        "max": dur[-1] if dur else None,
        "median": dur[len(dur) // 2] if dur else None,
        "gt_300": sum(1 for d in dur if d > 300.0),
        "gt_180": sum(1 for d in dur if d > 180.0),
    }
    #: 人类真实总花费（对账用）：每局两侧 `aggregate_stats[*].total[1]` 之和
    tot_e = [sum((r["agg"][s]["elixir"] or 0) for s in SIDES) for r in recs]
    c["human_total_elixir_per_replay"] = {
        "n": len(tot_e), "sum": sum(tot_e),
        "median": sorted(tot_e)[len(tot_e) // 2] if tot_e else None,
    }
    keys = collections.Counter(k for r in recs for s in SIDES for k, _lv in r["decks"][s])
    c["distinct_deck_card_keys"] = len(keys)
    c["deck_card_keys"] = dict(keys.most_common())
    c["variant_suffixed_keys"] = sorted({k for k in keys
                                         if isinstance(k, str) and ("-ev" in k or "-hero" in k)})
    c["void_occurrences"] = keys.get("void", 0)
    c["replays_with_void"] = sum(
        1 for r in recs for s in SIDES for k, _lv in r["decks"][s] if k == "void")
    return c


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", required=True)
    ap.add_argument("--actions", default=None, help="可选：actions 分片，用于事件数对账")
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-census", default=None)
    ap.add_argument("--limit", type=int, default=0, help="只导前 N 局（0=全部）")
    args = ap.parse_args(argv)

    tbl = _load(args.replays)
    d = tbl.to_pydict()
    n = len(d["replay_tag"])
    take = range(n if not args.limit else min(n, args.limit))

    cross = {"status": "skipped (no --actions)"}
    if args.actions:
        a = _load(args.actions).to_pydict()
        cnt = collections.Counter(a["replay_tag"])
        # actions 分片只含它覆盖到的 tag；对账只在该子集上做
        tags = {d["replay_tag"][i] for i in take}
        common = [t for t in tags if t in cnt]
        if common:
            by_tag = {d["replay_tag"][i]: json.loads(d["payload_json"][i]) for i in take}
            mismatch = [t for t in common
                        if len(by_tag[t].get("events") or []) != cnt[t]]
            #: 坐标来源对账：payload `coordinates.native_world_units` vs actions 表的 `native_x/native_y`
            act_native = {}
            for j in range(len(a["replay_tag"])):
                if a["replay_tag"][j] in tags:
                    act_native[(a["replay_tag"][j], a["source_index"][j])] = (a["native_x"][j],
                                                                             a["native_y"][j])
            n_cmp = n_ok = 0
            for t in common[:80]:
                for e in by_tag[t].get("events") or []:
                    key = (t, e.get("source_index"))
                    if key not in act_native:
                        continue
                    nw = (e.get("coordinates") or {}).get("native_world_units") or {}
                    n_cmp += 1
                    if (nw.get("x"), nw.get("y")) == act_native[key]:
                        n_ok += 1
            cross = {"tags_in_actions_shard": len(cnt), "overlap_with_replays": len(common),
                     "payload_events_equals_action_rows": len(common) - len(mismatch),
                     "mismatch_tags": mismatch[:5], "mismatch_total": len(mismatch),
                     "native_coord_compared": n_cmp, "native_coord_equal": n_ok}

    os.makedirs(os.path.dirname(os.path.abspath(args.out_jsonl)), exist_ok=True)
    recs = []
    with open(args.out_jsonl, "w", encoding="utf-8", newline="\n") as f:
        for i in take:
            r = record(d, i)
            recs.append(r)
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"[extract] {len(recs)} replays -> {args.out_jsonl} "
          f"({os.path.getsize(args.out_jsonl) / 1e6:.1f} MB)")

    if args.out_census:
        cen = census(recs, cross)
        os.makedirs(os.path.dirname(os.path.abspath(args.out_census)), exist_ok=True)
        with open(args.out_census, "w", encoding="utf-8", newline="\n") as f:
            json.dump(cen, f, ensure_ascii=False, indent=1)
        print(f"[census] -> {args.out_census}")
        for k in ("replays", "events_total", "events_kind", "events_side", "battle_type",
                  "game_mode", "result", "deck_size", "deck_levels", "duration_s",
                  "distinct_deck_card_keys", "void_occurrences", "cross_check_events_vs_actions"):
            print(f"  {k}: {cen[k]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
