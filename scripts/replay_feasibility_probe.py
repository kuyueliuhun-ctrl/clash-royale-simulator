# -*- coding: utf-8 -*-
"""CR 人类回放 → **我们的引擎** 的重放可行性探针（只读，不训练）。

问题：公开人类回放（`VanguardX101/IL_Replay`，**动作侧**）能不能驱动 `src/clasher_new` 的引擎？
——这是「我们无人类数据」能否变成「有输入、缺管道」的**可计算性**检查，**不是**等价性判据。

读数（每个都带可复算的分子/分母）：
  R1 **事件卡键可映射率**：`card_key`（RoyaleAPI kebab，如 `fire-spirit`）经
     `card_aliases.resolve_card()` 落到我们卡表。
     ⚠️ 实测：`resolve_card` 对未知键**返回 `None` 而不是抛异常** ⇒ 判据必须查 None。
  R2 **卡组可重建率**：卡组键带**变体后缀**（`-ev1` 觉醒 / `-hero` 英雄）⇒ 先剥后缀再解析，
     变体写进我们**已有**的 `PlayerState.set_evolution_slots()` / `set_hero_slots()`。
  R5 **手牌循环费用一致性（不需要引擎的自检）**：回放给了 `four_card_cycle_elixir`；我们对**同一卡组**
     算「最便宜 4 张费用和」⇒ 相等说明卡组 / 卡等 / 费用表三者对齐。
  R3 **引擎接受率**（本探针的核心）：按 `replay_tick_20hz` 升序，**先推进引擎**（`BattleState.step`，
     让圣水回费与单位行为发生）再 `deploy_card`，统计 True/False + 失败归因
     （`unaffordable` / `not_in_cycle` / `other`；引擎 L4 是**静默 `return False`**，能算才归因）。
  R4 **坐标约定 A/B（重要）**：回放 `native_x/native_y` 是**世界坐标**（`native/1000` 与
     `rl/action_bundle.py::sub_position` 的 `(x+0.5, y+0.5)` **逐位吻合**：回放 `grid_cell_floor=(8,0)`
     对应 `native_x/1000=8.5` ⇒ 就是我们 `Position` 的 x）。
     ⚠️ **实测判定：不需要翻转**（`raw` 接受率 40.5%/42.6%/53.0% vs `flip_y` 2.3%/2.3%/0%）。
     本脚本仍两种并列报数（【R17】不调和），因为**这是我先写错过的一条假设**：
     我原先按 `_mask_diff_snapshot.py` 的用例命名推断「pid0 拥有 y>16 一侧」⇒ 推出需要 `32−y`，
     结果 flip 后**几乎全部位置非法**（`unaffordable=0`、`other≈100%`）⇒ 假设被自己的 A/B 否证。
  R6 **终局塔血**：重放后塔血 vs 回放 `final_tower_hitpoints`。⚠️ **明确记"不可比"**
     （手牌循环顺序未知、逐事件圣水缺失、卡等未对齐、且回放时长可 > 我们 300 s 硬上限）
     ⇒ 只作可计算性读数，**不得**当等价性证据。

用法（Windows venv python；先跑 `scripts/_il_replay_extract.py` 生成 JSON）：
    .venv/Scripts/python.exe scripts/replay_feasibility_probe.py \
        --events-json docs/il_replay_probe_2026-09-19/one_shard.json --out <result.json>

⚠️ **只读**：不改录像/权重/训练；只在 `--out` 写 JSON。
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src", "clasher_new")
#: ⚠️ 先记住调用者 cwd：下面要 `chdir(SRC)`（引擎数据文件按 cwd 解析），
#: 之后相对路径的 CLI 参数会被解析到 src/clasher_new 下 —— 本仓踩过这个坑，故显式保留。
ORIG_CWD = os.getcwd()
sys.path.insert(0, SRC)
os.chdir(SRC)

from rl.io_bootstrap import force_utf8_stdout  # noqa: E402

force_utf8_stdout()

from battle import BattleState  # noqa: E402
from core import Position  # noqa: E402
from player import PlayerState  # noqa: E402
from card_utils import Card  # noqa: E402
from card_aliases import resolve_card  # noqa: E402

#: 卡等：**默认取回放自己的等级**（回放卡组实测 lv16 ⇒ 塔血 5980/3110 与该等级一致）
LEVEL = 16
VARIANTS = ("-ev1", "-hero")     # 回放卡组键的变体后缀（实测 bats-ev1 / magic-archer-hero）
Y_FLIP = 32.0                    # 竞技场高（world y ∈ [0,32]）；回放 native_y 需 32−y
DT = 0.05                        # 引擎推进步长（秒）
TIME_CAP = 300.0                 # 我们引擎的加时硬上限（battle.py:2856-2866）


def split_variant(key):
    for v in VARIANTS:
        if key.endswith(v):
            return key[: -len(v)], v.lstrip("-")
    return key, None


def map_key(key):
    """→ (resolved | None, variant | None, err | None)。`resolve_card` 对未知键返回 None（实测）。"""
    base, variant = split_variant(key)
    try:
        got = resolve_card(base)
    except Exception as e:  # noqa: BLE001
        return None, variant, f"{type(e).__name__}: {e}"
    if got is None:
        return None, variant, "resolve_card -> None"
    return got, variant, None


def towers(bs, pid):
    out = {"king": None, "princess": []}
    for e in bs.entities.values():
        nm = e.name or ""
        if getattr(e, "player", None) != pid:
            continue
        if "KingTower" in nm:
            out["king"] = e.hp
        elif "PrincessTower" in nm:
            out["princess"].append(e.hp)
    return out


def cheapest_four_elixir(cards):
    costs = []
    for c in cards:
        try:
            costs.append(float(Card(c).elixir))
        except Exception:  # noqa: BLE001
            return None
    costs.sort()
    return sum(costs[:4]) if len(costs) >= 4 else None


def _mapping_and_decks(rep):
    keys = sorted({e["card_key"] for e in rep["events"] if e.get("card_key")})
    mapping, unmapped = {}, []
    for k in keys:
        got, _v, err = map_key(k)
        if got is None:
            unmapped.append({"key": k, "err": err})
        else:
            mapping[k] = got
    renamed = {k: v for k, v in mapping.items() if k.replace("-", "").lower() != v.lower()}

    decks, variants, deck_unmapped = {}, {}, {}
    for side in ("team", "opponent"):
        raw = (rep.get("deck_cards_with_levels", {}).get(side)
               or rep.get("decks", {}).get(side) or [])
        names, vs, bad = [], [], []
        for c in raw:
            got, v, err = map_key(c.get("key"))
            if got is None:
                bad.append({"key": c.get("key"), "err": err})
            else:
                names.append(got)
                if v:
                    vs.append({"key": c.get("key"), "resolved": got, "variant": v,
                               "level": c.get("level")})
        decks[side], variants[side], deck_unmapped[side] = names, vs, bad
    return mapping, unmapped, renamed, decks, variants, deck_unmapped


def replay_once(rep, mapping, decks, variants, coord, level):
    """coord ∈ {"raw","flip_y"}：回放 native_y 是否翻转。返回 (读数 dict, 终局塔血 dict)。"""
    bs = BattleState(PlayerState(0, list(decks["team"]), 5.0),
                     PlayerState(1, list(decks["opponent"]), 5.0),
                     card_level=level)
    bs.time = 0.0
    applied = {}
    for pid, side in ((0, "team"), (1, "opponent")):
        vs = [v for v in variants[side] if v["variant"] == "ev1"]
        hs = [v for v in variants[side] if v["variant"] == "hero"]
        info = {"ev1": [v["resolved"] for v in vs], "hero": [v["resolved"] for v in hs]}
        try:
            if vs:
                bs.players[pid].set_evolution_slots([v["resolved"] for v in vs])
                info["evo_slots_applied"] = sorted(bs.players[pid].evo_slots)
            if hs:
                bs.players[pid].set_hero_slots([v["resolved"] for v in hs])
                info["hero_slots_applied"] = sorted(bs.players[pid].hero_slots)
        except Exception as e:  # noqa: BLE001
            info["apply_error"] = f"{type(e).__name__}: {e}"
        applied[side] = info

    n_play = n_abil = ok = 0
    fails = {"unaffordable": 0, "not_in_cycle": 0, "other": 0, "after_time_cap": 0}
    samples, t_prev = [], 0.0
    for e in rep["events"]:
        if e["kind"] != "play_card":
            n_abil += 1
            continue
        n_play += 1
        t = float(e["replay_tick_20hz"]) / 20.0
        if t > TIME_CAP:
            fails["after_time_cap"] += 1
            continue
        # 推进引擎（圣水回费 + 单位行为）——按 DT 小步走到事件时刻
        while bs.time + DT <= t and not bs.game_over:
            bs.step(DT)
        t_prev = t
        pid = 0 if e["side"] == "team" else 1
        card = mapping.get(e["card_key"])
        if card is None:
            fails["other"] += 1
            continue
        x = float(e["native_x"]) / 1000.0
        y = float(e["native_y"]) / 1000.0
        if coord == "flip_y":
            y = Y_FLIP - y
        st = bs.players[pid]
        in_cycle = card in st.cycle
        cost = None
        try:
            cost = float(Card(card).elixir)
        except Exception:  # noqa: BLE001
            pass
        exc = None
        try:
            good = bool(bs.deploy_card(pid, card, Position(x, y)))
        except Exception as ex:  # noqa: BLE001
            good, exc = False, f"{type(ex).__name__}: {ex}"
        if good:
            ok += 1
            continue
        if exc is not None:
            fails["other"] += 1
        elif cost is not None and st.elixir < cost:
            fails["unaffordable"] += 1
        elif not in_cycle:
            fails["not_in_cycle"] += 1
        else:
            fails["other"] += 1
        if len(samples) < 8:
            samples.append({"tick": e["replay_tick_20hz"], "time": round(t, 2), "card": card,
                            "pid": pid, "pos": [round(x, 2), round(y, 2)],
                            "elixir": round(st.elixir, 3), "cost": cost,
                            "in_cycle": in_cycle, "exc": exc})
    return ({"play_card_events": n_play, "accepted": ok, "rejected": n_play - ok,
             "accept_rate": (ok / n_play) if n_play else None,
             "ability_events_not_replayed": n_abil,
             "buckets": fails, "samples": samples,
             "engine_end_time": round(float(bs.time), 2),
             "engine_game_over": bool(bs.game_over),
             "variant_apply": applied},
            {"ours_team": towers(bs, 0), "ours_opponent": towers(bs, 1)})


def run_replay(rep, level=LEVEL):
    res = {"replay_tag": rep["replay_tag"], "result": rep["result"],
           "crowns": [rep["team_crowns"], rep["opponent_crowns"]],
           "duration_timeline_seconds": (rep.get("aggregate_elixir") or {}).get("duration")}
    mapping, unmapped, renamed, decks, variants, deck_unmapped = _mapping_and_decks(rep)
    res["R1"] = {"distinct_keys": len(mapping) + len(unmapped), "mapped": len(mapping),
                 "unmapped": unmapped, "renamed_total": len(renamed), "renamed": renamed}
    res["R2"] = {"deck_sizes": {s: len(decks[s]) for s in decks},
                 "levels": {s: sorted({c.get("level") for c in
                                       (rep.get("deck_cards_with_levels", {}).get(s) or [])}
                                      - {None}) for s in ("team", "opponent")},
                 "variant_cards": {s: [v["variant"] for v in variants[s]] for s in variants},
                 "unmapped": deck_unmapped, "tower_card": rep.get("tower_card")}
    res["R5"] = {s: {"declared_four_card_cycle_elixir":
                     (rep.get("players_meta", {}).get(s) or {}).get("four_card_cycle_elixir"),
                     "ours_cheapest_four_elixir": cheapest_four_elixir(decks[s])}
                 for s in ("team", "opponent")}

    if any(len(decks[s]) != 8 for s in decks):
        res["R3"] = {"status": "ABORT: deck not reconstructed as 8 cards",
                     "deck_sizes": res["R2"]["deck_sizes"], "unmapped": deck_unmapped}
        return res

    res["R3"] = {}
    res["R6"] = {}
    for coord in ("raw", "flip_y"):
        r3, towers_after = replay_once(rep, mapping, decks, variants, coord, level)
        res["R3"][coord] = r3
        res["R6"][coord] = {"ours": towers_after}
    res["R6"]["replay_final_team"] = (rep.get("final_tower_hitpoints") or {}).get("team")
    res["R6"]["replay_final_opponent"] = (rep.get("final_tower_hitpoints") or {}).get("opponent")
    res["R6"]["verdict"] = ("NOT_COMPARABLE (hand-cycle order unknown / per-event elixir absent / "
                            "levels not aligned / replay duration may exceed our 300 s cap) "
                            "-- computability reading only, NOT an equivalence criterion")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--events-json", required=True)
    ap.add_argument("--level", type=int, default=LEVEL)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    for name in ("events_json", "out"):
        v = getattr(args, name, None)
        if v and not os.path.isabs(v):
            setattr(args, name, os.path.normpath(os.path.join(ORIG_CWD, v)))

    with open(args.events_json, encoding="utf-8") as f:
        blob = json.load(f)
    out = {"source": blob.get("source", {}), "results": []}
    for rep in blob["replays"]:
        r = run_replay(rep, args.level)
        out["results"].append(r)
        r1, r2 = r["R1"], r["R2"]
        print(f"=== {r['replay_tag'][:16]} result={r['result']} crowns={r['crowns']} level={args.level}")
        print(f"  R1 event keys mapped {r1['mapped']}/{r1['distinct_keys']} "
              f"renamed={r1['renamed']} unmapped={r1['unmapped']}")
        print(f"  R2 decks={r2['deck_sizes']} levels={r2['levels']} variants={r2['variant_cards']} "
              f"unmapped={r2['unmapped']} tower_card={r2.get('tower_card')}")
        print(f"  R5 four-card cycle elixir: {r['R5']}")
        if "R3" in r and isinstance(r["R3"], dict) and "status" in r["R3"]:
            print(f"  R3 {r['R3']}")
            continue
        for coord in ("raw", "flip_y"):
            a = r["R3"][coord]
            print(f"  R3[{coord}] accepted={a['accepted']}/{a['play_card_events']} "
                  f"rate={a['accept_rate']:.4f} buckets={a['buckets']} "
                  f"engine_t={a['engine_end_time']} over={a['engine_game_over']} "
                  f"abil_not_replayed={a['ability_events_not_replayed']}")
            for s in a["samples"][:3]:
                print(f"      sample: {s}")
        print(f"  R6 ours[raw]={r['R6']['raw']['ours']}")
        print(f"  R6 ours[flip_y]={r['R6']['flip_y']['ours']}")
        print(f"  R6 replay team={r['R6']['replay_final_team']} opponent={r['R6']['replay_final_opponent']}")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print(f"[probe] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
