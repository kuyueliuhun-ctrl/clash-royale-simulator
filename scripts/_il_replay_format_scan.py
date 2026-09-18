# -*- coding: utf-8 -*-
"""`VanguardX101/IL_Replay`（= **Firstlight anonymized battle replays**）格式扫描（只读）。

两种模式：
  --schema  抽 N 局，把 `payload_json` 的**完整嵌套结构**（键路径 → 类型 / 列表长度 / 样例值）
            连同两张 parquet 表的列一并导出 → JSON + 可读文本。
  --stats   对整分片做**字段出现率 + 取值分布**（哪些字段恒存在、哪些可选；取值域多大）。

⚠️ 依赖 pyarrow ⇒ 用 **WSL `/usr/bin/python3`**（实测 pyarrow 25.0.1；Windows venv 没有）。
**stdout 一律 ASCII-only**（本环境 GBK 陷阱；中文只留在源码注释里）。

用法:
  /usr/bin/python3 scripts/_il_replay_format_scan.py --schema \
      --actions /tmp/il_actions.parquet --replays /tmp/il_replays.parquet --n 3 \
      --out docs/il_replay_probe_2026-09-19/format_schema.json
  /usr/bin/python3 scripts/_il_replay_format_scan.py --stats --limit 1000 \
      --actions /tmp/il_actions.parquet --replays /tmp/il_replays.parquet \
      --out docs/il_replay_probe_2026-09-19/format_stats.json
  /usr/bin/python3 scripts/_il_replay_format_scan.py --check-cycle-elixir --limit 1000 \
      --actions /tmp/il_actions.parquet --replays /tmp/il_replays.parquet \
      --out docs/il_replay_probe_2026-09-19/format_cycle_elixir.json

`--check-cycle-elixir`：验证回放自带的 `four_card_cycle_elixir` 是否等于
「**该卡组最便宜 4 张的费用和**」（用我们自己的卡表 `card_utils.Card` 复算；实测 WSL python3
能直接 import 它）。先前只在 3 局 × 双侧（6/6）验过，本模式把它放大到 N 局。
"""
import argparse
import collections
import json
import os
import sys


def _load(p):
    import pyarrow.parquet as pq
    return pq.read_table(p)


def _short(v, n=120):
    s = repr(v)
    return s if len(s) <= n else s[:n] + "..."


def walk(obj, path, out, max_list_sample=1):
    """递归记录键路径：类型集合 / 是否列表 / 列表长度 / 样例值。"""
    t = type(obj).__name__
    rec = out.setdefault(path, {"types": set(), "list_len": set(), "sample": None})
    rec["types"].add(t)
    if rec["sample"] is None:
        rec["sample"] = _short(obj) if not isinstance(obj, (dict, list)) else None
    if isinstance(obj, dict):
        for k, v in obj.items():
            walk(v, f"{path}.{k}" if path else k, out)
    elif isinstance(obj, list):
        rec["list_len"].add(len(obj))
        for item in obj[:max_list_sample]:
            walk(item, f"{path}[]", out)


def schema_mode(rep_tbl, act_tbl, n, tags=None):
    rp = rep_tbl.to_pydict()
    ac = act_tbl.to_pydict()
    n_rows = len(rp["replay_tag"])
    idxs = list(range(min(n, n_rows))) if not tags else [rp["replay_tag"].index(t) for t in tags]
    payload_paths, action_keys = {}, sorted(ac.keys())
    per_replay = []
    for i in idxs:
        payload = json.loads(rp["payload_json"][i])
        walk(payload, "", payload_paths)
        rows = [{k: ac[k][j] for k in action_keys}
                for j in range(len(ac["replay_tag"])) if ac["replay_tag"][j] == rp["replay_tag"][i]]
        per_replay.append({
            "replay_tag": rp["replay_tag"][i],
            "table_row": {k: rp[k][i] for k in rp if k != "payload_json"},
            "payload_top_keys": sorted(payload.keys()),
            "n_action_rows": len(rows),
            "first_action_row": rows[0] if rows else None,
            "kinds": dict(collections.Counter(r["kind"] for r in rows)),
            "sides": dict(collections.Counter(r["side"] for r in rows)),
        })
    paths = {p: {"types": sorted(v["types"]), "list_len": sorted(v["list_len"]),
                 "sample": v["sample"]} for p, v in sorted(payload_paths.items())}
    return {"mode": "schema", "n_replays": len(idxs), "replays_table_cols": sorted(rp.keys()),
            "actions_table_cols": action_keys, "payload_paths": paths, "replays": per_replay}


def stats_mode(rep_tbl, act_tbl, limit):
    rp = rep_tbl.to_pydict()
    ac = act_tbl.to_pydict()
    n_rows = len(rp["replay_tag"])
    use = min(limit or n_rows, n_rows)

    presence = collections.Counter()
    samples = {}
    consts = collections.defaultdict(collections.Counter)
    stats = collections.defaultdict(list)
    variants = collections.Counter()
    levels = collections.Counter()
    tower_cards = collections.Counter()

    def visit(obj, path):
        presence[path] += 1
        if isinstance(obj, dict):
            for k, v in obj.items():
                visit(v, f"{path}.{k}" if path else k)
        elif isinstance(obj, list):
            for it in obj[:1]:
                visit(it, f"{path}[]")

    for i in range(use):
        pj = json.loads(rp["payload_json"][i])
        visit(pj, "")
        for key in ("schema_version", "collection_schema_version"):
            if isinstance(pj.get(key), str):
                consts[key][pj[key]] += 1
        b = pj.get("battle", {})
        for side in ("team", "opponent"):
            blk = b.get(side) or {}
            consts["battle.%s.crowns" % side][blk.get("crowns")] += 1
            for p in (blk.get("players") or [])[:1]:
                for c in (p.get("deck") or []):
                    k = c.get("card_key") or ""
                    levels[c.get("level")] += 1
                    for suf in ("-ev1", "-hero"):
                        if k.endswith(suf):
                            variants[suf] += 1
                tc = p.get("tower_card") or {}
                tower_cards[tc.get("card_key")] += 1
        dur = (pj.get("replay") or {}).get("duration") or {}
        if isinstance(dur.get("timeline_seconds"), (int, float)):
            stats["duration_timeline_seconds"].append(float(dur["timeline_seconds"]))
        v = pj.get("validation") or {}
        for k, val in v.items():
            if isinstance(val, (int, float)):
                stats["validation." + k].append(float(val))
        consts["battle.game_mode"][b.get("game_mode")] += 1
        consts["battle.battle_type"][b.get("battle_type")] += 1
        consts["battle.result"][b.get("result")] += 1

    n_rows_all = len(ac["replay_tag"])
    act_stats = {}
    for k in ac:
        col = ac[k]
        try:
            vals = [x for x in col[:200000] if x is not None]
            if vals and isinstance(vals[0], (int, float, bool)):
                act_stats[k] = {"dtype": type(vals[0]).__name__, "min": min(vals), "max": max(vals),
                                "n_distinct_first200k": len(set(vals[:200000]))}
            elif vals and isinstance(vals[0], str):
                c = collections.Counter(vals)
                act_stats[k] = {"dtype": "str", "n_distinct": len(c), "top": c.most_common(8)}
            else:
                act_stats[k] = {"dtype": "other", "sample": _short(col[0] if col else None)}
        except Exception as e:  # noqa: BLE001
            act_stats[k] = {"error": f"{type(e).__name__}: {e}"}

    def _summ(name, xs):
        xs = sorted(xs)
        if not xs:
            return None
        n = len(xs)
        return {"n": n, "min": xs[0], "p50": xs[n // 2], "max": xs[-1],
                "mean": round(sum(xs) / n, 4), "distinct": len(set(xs))}

    return {"mode": "stats", "replays_scanned": use, "replays_total_in_shard": n_rows,
            "key_presence": dict(presence.most_common()),
            "constant_fields": {k: dict(v) for k, v in consts.items()},
            "variant_suffix_counts": dict(variants), "deck_level_counts": dict(levels),
            "tower_card_counts": dict(tower_cards),
            "numeric_summaries": {k: _summ(k, v) for k, v in stats.items()},
            "actions_rows_in_shard": n_rows_all, "actions_columns": act_stats}


def check_cycle_elixir(rep_tbl, limit, rel_src):
    """declared `four_card_cycle_elixir` vs 我们复算的「最便宜 4 张费用和」。"""
    if rel_src not in sys.path:
        sys.path.insert(0, rel_src)
    from card_aliases import resolve_card
    from card_utils import Card

    def base(k):
        for suf in ("-ev1", "-hero"):
            if k.endswith(suf):
                return k[: -len(suf)]
        return k

    rp = rep_tbl.to_pydict()
    use = min(limit or len(rp["replay_tag"]), len(rp["replay_tag"]))
    rows, flips = [], {"declared_lt": 0, "declared_gt": 0, "equal": 0}
    unmapped = 0
    for i in range(use):
        pj = json.loads(rp["payload_json"][i])
        for side in ("team", "opponent"):
            for p in (pj.get("battle", {}).get(side, {}).get("players") or [])[:1]:
                keys = [c.get("card_key") or "" for c in (p.get("deck") or [])]
                costs, ok = [], True
                for k in keys:
                    try:
                        nm = resolve_card(base(k))
                        if nm is None:
                            ok = False
                            break
                        costs.append(float(Card(nm).elixir))
                    except Exception:  # noqa: BLE001
                        ok = False
                        break
                if not ok or len(costs) < 4:
                    unmapped += 1
                    continue
                ours = sum(sorted(costs)[:4])
                dec = p.get("four_card_cycle_elixir")
                rows.append({"replay": rp["replay_tag"][i][:12], "side": side,
                             "declared": dec, "ours_cheapest4": ours, "deck_size": len(costs)})
                if dec is None:
                    continue
                if abs(float(dec) - ours) < 1e-9:
                    flips["equal"] += 1
                elif float(dec) < ours:
                    flips["declared_lt"] += 1
                else:
                    flips["declared_gt"] += 1
    return {"mode": "check_cycle_elixir", "replays_scanned": use, "sides_compared": len(rows),
            "unmappable_sides": unmapped, "outcome": flips, "examples": rows[:20]}


def check_terminal_state(rep_tbl, act_tbl):
    """终局结果 / 塔血 / 是否逐次记录 —— 穷举取证。"""
    rp = rep_tbl.to_pydict()
    ac = act_tbl.to_pydict()
    n = len(rp["replay_tag"])
    ekeys, kinds, results, lvls = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
    king_vals, princess_vals = collections.Counter(), collections.Counter()
    hp_missing = 0
    total_ok = total_bad = 0
    bad_examples = []
    table_matches_payload = 0
    for i in range(n):
        pj = json.loads(rp["payload_json"][i])
        b = pj.get("battle") or {}
        results[b.get("result")] += 1
        for e in pj.get("events") or []:
            for k in e:
                ekeys[k] += 1
            kinds[e.get("kind")] += 1
        for side in ("team", "opponent"):
            blk = b.get(side) or {}
            p = (blk.get("players") or [{}])[0]
            hp = p.get("final_tower_hitpoints")
            lvls[(p.get("tower_card") or {}).get("level")] += 1
            if not hp:
                hp_missing += 1
                continue
            k, l, r, t = (hp.get("king"), hp.get("princess_left"),
                          hp.get("princess_right"), hp.get("total"))
            if None in (k, l, r, t):
                hp_missing += 1
                continue
            if k + l + r == t:
                total_ok += 1
            else:
                total_bad += 1
                if len(bad_examples) < 3:
                    bad_examples.append({side: [k, l, r, t]})
            king_vals[k] += 1
            princess_vals[l] += 1
            princess_vals[r] += 1
        if (rp["result"][i] == b.get("result")
                and rp["team_crowns"][i] == (b.get("team") or {}).get("crowns")
                and rp["opponent_crowns"][i] == (b.get("opponent") or {}).get("crowns")):
            table_matches_payload += 1

    # 只扫这份 actions 分片里属于本表 replay 的行（分片必须来自同一 dataset_root）
    in_tbl = set(rp["replay_tag"])
    act_rows = sum(1 for t in ac["replay_tag"] if t in in_tbl)
    return {"mode": "check_terminal_state", "replays": n,
            "event_key_union": dict(ekeys), "event_kinds": dict(kinds),
            "action_rows_matched": act_rows,
            "terminal_result": dict(results),
            "terminal_hp_fields_missing": hp_missing,
            "terminal_hp_total_identity_ok": total_ok,
            "terminal_hp_total_identity_bad": total_bad,
            "terminal_hp_bad_examples": bad_examples,
            "table_columns_match_payload": table_matches_payload,
            "tower_card_level_histogram": {str(k): v for k, v in lvls.items()},
            "king_hp_distinct_nonzero": len([v for v in king_vals if v]),
            "king_hp_max": max([v for v in king_vals if v], default=None),
            "princess_hp_distinct_nonzero": len([v for v in princess_vals if v]),
            "princess_hp_max": max([v for v in princess_vals if v], default=None),
            "king_hp_top": king_vals.most_common(8),
            "princess_hp_top": princess_vals.most_common(8),
            "per_event_tower_fields": sorted(
                k for k in ekeys if any(t in k.lower() for t in ("hp", "health", "tower", "damage")))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--actions", required=True)
    ap.add_argument("--replays", required=True)
    ap.add_argument("--schema", action="store_true")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--check-cycle-elixir", action="store_true")
    ap.add_argument("--check-terminal-state", action="store_true")
    ap.add_argument("--src", default=None, help="src/clasher_new 路径（--check-cycle-elixir 用）")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--tag", action="append", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    rt, at = _load(args.replays), _load(args.actions)
    if args.schema:
        out = schema_mode(rt, at, args.n, args.tag)
    elif args.stats:
        out = stats_mode(rt, at, args.limit)
    elif args.check_terminal_state:
        out = check_terminal_state(rt, at)
    elif args.check_cycle_elixir:
        rel_src = args.src or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                           "src", "clasher_new")
        out = check_cycle_elixir(rt, args.limit, rel_src)
    else:
        raise SystemExit("need --schema or --stats")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    if args.schema:
        print(f"[schema] replays={out['n_replays']} payload_paths={len(out['payload_paths'])}")
        for p, v in out["payload_paths"].items():
            ll = f" list_len={v['list_len']}" if v["list_len"] else ""
            print(f"  {p:70s} {','.join(v['types'])}{ll}  {v['sample'] or ''}")
        print(f"  replays_table_cols={out['replays_table_cols']}")
        print(f"  actions_table_cols={out['actions_table_cols']}")
    elif out["mode"] == "check_terminal_state":
        print(f"[terminal] replays={out['replays']} action_rows={out['action_rows_matched']}")
        print(f"  event_kinds={out['event_kinds']}")
        print(f"  event_key_union({len(out['event_key_union'])})={sorted(out['event_key_union'])}")
        print(f"  per-event tower/hp/damage fields={out['per_event_tower_fields']}")
        print(f"  result={out['terminal_result']}")
        print(f"  terminal HP: missing={out['terminal_hp_fields_missing']} "
              f"total_identity ok={out['terminal_hp_total_identity_ok']} bad={out['terminal_hp_total_identity_bad']}")
        print(f"  table cols match payload={out['table_columns_match_payload']}")
        print(f"  tower_card.level={out['tower_card_level_histogram']}")
        print(f"  king HP max={out['king_hp_max']} distinct={out['king_hp_distinct_nonzero']} top={out['king_hp_top'][:4]}")
        print(f"  princess HP max={out['princess_hp_max']} distinct={out['princess_hp_distinct_nonzero']} top={out['princess_hp_top'][:4]}")
    elif out["mode"] == "check_cycle_elixir":
        print(f"[cycle] replays={out['replays_scanned']} sides={out['sides_compared']} "
              f"unmappable={out['unmappable_sides']} outcome={out['outcome']}")
        for e in out["examples"][:6]:
            print("  ", e)
    else:
        print(f"[stats] replays scanned={out['replays_scanned']} keys={len(out['key_presence'])} "
              f"actions rows={out['actions_rows_in_shard']}")
        print("  constant_fields:", json.dumps(out["constant_fields"], ensure_ascii=False)[:400])
        print("  variant_suffix_counts:", out["variant_suffix_counts"])
        print("  tower_card_counts:", out["tower_card_counts"])
        print("  numeric_summaries:", json.dumps(out["numeric_summaries"], ensure_ascii=False)[:600])
    print(f"[scan] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
