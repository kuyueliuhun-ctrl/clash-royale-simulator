# -*- coding: utf-8 -*-
"""FirstLight `IL_Replay` 回放 → **我们引擎的 IL（行为克隆）训练对**。

两种模式（`--mode`）：

* `reconcile` —— **纯查表，不跑引擎**：验证「过滤漏斗」与「卡牌经济对账」（预注册 J1.1）。
  对每局每侧核对 `Σ_{play_card} Card(card_key).elixir` == 回放自带的
  `payload.replay.aggregate_stats[side].total[1]`（人类真实总花费），并核对张数。
* `samples`   —— **跑引擎**：按 0.5 s 决策帧推进，在「人类出牌帧」取 `(obs, masks, bundle)`。

预注册（判据/失败分支/不变量）见 `docs/fl_il_prereg_2026-09-20.md`；**判据在实现前已落盘**。

用法（Windows venv；先跑 `scripts/_fl_il_extract.py` 生成 JSONL）:
    .venv/Scripts/python.exe scripts/fl_il_to_bc.py --mode reconcile \
        --jsonl /mnt/e/fl_il_data/replays_part000000.jsonl \
        --out docs/fl_il_2026-09-20/reconcile_part000000.json

⚠️ 引擎数据文件按 **cwd** 解析 ⇒ 本脚本沿用 `scripts/replay_feasibility_probe.py:43-47` 的既有做法：
   保存 `ORIG_CWD`、`chdir(SRC)`，CLI 相对路径按 `ORIG_CWD` 解析。
"""
import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src", "clasher_new")
ORIG_CWD = os.getcwd()
sys.path.insert(0, SRC)
os.chdir(SRC)

from rl.io_bootstrap import force_utf8_stdout  # noqa: E402

force_utf8_stdout()

from card_aliases import resolve_card  # noqa: E402
from card_utils import Card  # noqa: E402
from evolutions import EVOLUTION_CYCLES, evolution_state  # noqa: E402

from rl.env_wrapper import RLEnv  # noqa: E402
from rl.action_bundle import ActionBundle  # noqa: E402
from rl.action_mask import _card_cost  # noqa: E402
from rl.belief import BeliefInference, belief_token_dim  # noqa: E402
from rl.belief_planner import BeliefPlanner  # noqa: E402
from rl.plan_space import PLAN_DIM  # noqa: E402
from rl.hand_score import plan_extras as plan_extras_vec  # noqa: E402
from rl.observation import ENTITY_NAMES  # noqa: E402
from rl.follower import FollowerPolicy  # noqa: E402
from rl.config import TrainConfig, reward_to_env  # noqa: E402
from core import Position  # noqa: E402
import numpy as np  # noqa: E402
import pickle  # noqa: E402

#: 预注册 §1：决策帧 0.5 s；引擎加时硬上限 300 s
FRAME_DT = 0.5
TIME_CAP = 300.0
#: 预注册 §3：允许的模式（`1v1 Showdown` 选牌 / `Normal Battle` 规则未定 ⇒ 排除）
ALLOWED_MODES = ("Ranked", "Ladder", "1v1 Battle", "1v1",
                 "Grand Challenge", "Classic Challenge")
VARIANTS = ("-ev1", "-ev2", "-ev3", "-hero")
#: 预注册 J3：6 类 option 里的 4 个出牌槽（本实验样本全是出牌帧 ⇒ 随机基线 1/6）
BASE_RATE = 1.0 / 6.0


def split_variant(key):
    for v in VARIANTS:
        if key.endswith(v):
            return key[: -len(v)], v.lstrip("-")
    return key, None


def map_key(key):
    base, variant = split_variant(key)
    try:
        got = resolve_card(base)
    except Exception as e:  # noqa: BLE001
        return None, variant, f"{type(e).__name__}: {e}"
    if got is None:
        return None, variant, "resolve_card -> None"
    return got, variant, None


def cost_of(name):
    try:
        return float(Card(name).elixir)
    except Exception:  # noqa: BLE001
        return None


def deck_of(rep, side):
    """→ (names, variants, bad)；任一键映射失败 ⇒ bad 非空 ⇒ 弃局（预注册 §3.2）。"""
    names, vs, bad = [], [], []
    for key, _lv in rep["decks"][side]:
        got, v, err = map_key(key)
        if got is None:
            bad.append({"key": key, "err": err})
        else:
            names.append(got)
            if v:
                vs.append({"key": key, "resolved": got, "variant": v})
    return names, vs, bad


#: ★ 觉醒 S1（预注册 `docs/il_evo_prereg_2026-09-22.md` §2）：回放牌组后缀 → 本仓觉醒位。
#: **只接 `-ev*`**；`-hero` 是**另一套机制**（`set_hero_slots` / Hero 数值表），
#: 本仓 Hero 表对这批卡覆盖 **0%**（预注册 §5：4,550 槽全落空）⇒ 显式**不接线、单列**，
#: 不许把 Hero 当觉醒混进读数（否则「觉醒生效」的结论会被 Hero 槽污染）。
def evo_slot_cards(vs):
    """`deck_of` 的 variants → `(可声明的觉醒卡, 未映射清单)`。

    可声明 = 变体是 `ev*` **且**本仓**两处**数据都在：`EVOLUTION_CYCLES`（周期表）
    与 `Card(name).evo_raw`（觉醒数值快照）——`battle.py:3068-3070` 的触发判定同时要这两个，
    缺任何一个 ⇒ 声明了也**永不触发**（死代码）⇒ 计入「未映射」，**不静默声明**（预注册 §2 S1 做法）。
    """
    cards, unmapped = [], []
    for v in vs:
        var = v.get("variant") or ""
        if not var.startswith("ev"):
            continue
        name = v["resolved"]
        try:
            has_raw = bool(Card(name).evo_raw)
        except Exception:  # noqa: BLE001
            has_raw = False
        if name in EVOLUTION_CYCLES and has_raw:
            cards.append(name)
        else:
            unmapped.append({"key": v["key"], "resolved": name, "variant": var,
                             "in_cycles": name in EVOLUTION_CYCLES, "has_evo_raw": has_raw})
    return cards, unmapped


def install_evo_counters(battle, st):
    """给重建战局装**两个独立**的觉醒触发计数器（【R13】位图对账的同一种思路：两路读数互相校验）。

    * `finish`（**主口径**，覆盖部队/建筑/**法术**全部类型）：包 `BattleState._finish_deploy`。
      它在**每次成功出牌**收尾时被调（`battle.py:2934` 在这里自增 `evo_plays`，出口只有一处），
      而自增在**函数体内** ⇒ 在**调用前**按 `battle.py:3068-3070` 的同一组输入读一次，
      与引擎自己的 `evolved` 判定**同值**。
    * `wrap`（**交叉核对**，只覆盖部队/建筑）：包 `BattleState._wrap`，数
      `len(entity_data)==6 and entity_data[5]`——那是引擎从出生队列里带过来的**觉醒标记位**
      （`battle.py:2758-2761` 读它、`:3215` 写它）⇒ **不重算谓词，直接读引擎的位**。
      法术觉醒（觉醒 Zap 领域 `battle.py:3137` / GoblinBarrel 诱饵 `:3181`）不走 `_wrap`
      ⇒ 预期 `finish >= wrap`，差值应能被「法术觉醒」解释；对不上就**照实报**（【R10】）。

    两个计数器都**只读**，不改行为（`battle.py` 一行未改；`rl/` 零改动）。
    """
    counts = {"finish": {"team": 0, "opp": 0}, "wrap": {"team": 0, "opp": 0},
              "finish_by_card": {}, "wrap_by_card": {}}
    orig_finish = battle._finish_deploy
    orig_wrap = battle._wrap

    def _finish(player_id, card_name, *a, **kw):
        p = battle.players[player_id]
        try:
            hit = bool(Card(card_name).evo_raw and card_name in p.evo_slots
                       and card_name not in p.hero_slots
                       and evolution_state(p.evo_plays.get(card_name, 0), card_name))
        except Exception:  # noqa: BLE001
            hit = False
        if hit:
            side = "team" if int(player_id) == 0 else "opp"
            counts["finish"][side] += 1
            counts["finish_by_card"][card_name] = counts["finish_by_card"].get(card_name, 0) + 1
        return orig_finish(player_id, card_name, *a, **kw)

    def _wrap(entity_data):
        try:
            if len(entity_data) == 6 and entity_data[5]:
                side = "team" if int(entity_data[2]) == 0 else "opp"
                counts["wrap"][side] += 1
                nm = str(entity_data[3])
                counts["wrap_by_card"][nm] = counts["wrap_by_card"].get(nm, 0) + 1
        except Exception:  # noqa: BLE001
            pass
        return orig_wrap(entity_data)

    battle._finish_deploy = _finish
    battle._wrap = _wrap
    st["_evo_counts"] = counts
    return counts


def frames_of(rep, side):
    """把某侧 play_card 事件按 0.5 s 帧分桶 → {frame: [event, ...]}（预注册 §1）。"""
    buckets = collections.defaultdict(list)
    for e in rep["ev"]:
        if e[0] != 0:            # 只算 play_card
            continue
        if e[1] != side:
            continue
        tick = e[3]
        if not isinstance(tick, (int, float)):
            continue
        buckets[int(float(tick) / 20.0 / FRAME_DT)].append(e)
    return buckets


def reconcile_one(rep):
    """一局的读数：过滤理由 + 卡牌经济对账 + 可转样本数（**不跑引擎**）。"""
    out = {"tag": rep["tag"], "gm": rep["gm"], "bt": rep["bt"], "dur": rep["dur"],
           "reject": None, "sides": {}, "labels_frames": {}}
    if rep["gm"] not in ALLOWED_MODES:
        out["reject"] = "game_mode"
        return out
    if not isinstance(rep["dur"], (int, float)) or rep["dur"] > TIME_CAP:
        out["reject"] = "duration_gt_300s"
        return out
    decks, bads = {}, {}
    for side in ("team", "opponent"):
        decks[side], _vs, bads[side] = deck_of(rep, side)
        if bads[side] or len(decks[side]) != 8:
            out["reject"] = "deck_unmapped"
            out["sides"][side] = {"bad": bads[side], "n": len(decks[side])}
            return out
    n_team_play = sum(1 for e in rep["ev"] if e[0] == 0 and e[1] == 0)
    if n_team_play < 4:
        out["reject"] = "team_play_lt4"
        return out

    for side, sname in ((0, "team"), (1, "opponent")):
        evs = [e for e in rep["ev"] if e[0] == 0 and e[1] == side]
        n_ab = sum(1 for e in rep["ev"] if e[0] == 1 and e[1] == side)
        costs, unmapped = [], []
        n_mirror = 0
        #: ⚠️ **口径修正（本轮实测）**：`Mirror` 的**实际**费用 = 上一张牌费用 + 1
        #: （引擎自己的语义：`rl/action_mask.py:30-36 _card_cost()`；且 `player.py:47`
        #: `if card_name != 'Mirror': self.last_card = card_name` ⇒ 连打 Mirror 复制同一张）。
        #: 第一版用 `Card('Mirror').elixir`（卡面 1 费）⇒ 每局有 Mirror 的对账**必差**
        #: （实测 24/24 全部如此，且 0/7044 无 Mirror 的侧有差）——**这是本仪器口径错，
        #: 不是引擎错**（引擎实现是对的）。
        last_cost = None
        for e in evs:
            #: ⚠️ 事件里的 `card_key` 也是 **RoyaleAPI kebab 键**（且可能带 `-ev1/-hero` 变体后缀）
            #: ⇒ 必须先 `resolve_card` 再查费用；第一版直接拿 kebab 键喂 `Card()` ⇒ 费用恒 None、
            #: 总花费恒 0（实测踩过）。
            got, _v, _err = map_key(e[2] or "")
            if got == "Mirror":
                n_mirror += 1
                c = (last_cost + 1.0) if last_cost is not None else None
            else:
                c = cost_of(got) if got else None
                if c is not None:
                    last_cost = c          # Mirror 不覆盖 last_card（player.py:47）
            if c is None:
                unmapped.append(e[2])
            else:
                costs.append(c)
        declared_e = rep["agg"][sname]["elixir"]
        declared_cards = rep["agg"][sname]["cards"]
        ours_e = sum(costs)
        #: ⚠️ 实测：`aggregate_stats[*].total[0]` **不含** `activate_ability`
        #: （样本局 team: troop 12 + spell 3 = total 15，而 ability 另记 3 条、0 圣水）
        #: ⇒ 张数只与 `play_card` 事件数比；第一版多加了 ability 数（实测踩过）。
        ours_cards = len(evs)
        ok_e = (declared_e is not None and abs(ours_e - declared_e) <= 1.0)
        out["sides"][sname] = {
            "play_card_events": len(evs), "ability_events": n_ab,
            "mirror_plays": n_mirror,
            "unmapped_event_keys": sorted(set(unmapped))[:5],
            "ours_total_elixir": ours_e, "declared_total_elixir": declared_e,
            "elixir_abs_diff": (None if declared_e is None else ours_e - declared_e),
            "elixir_match": bool(ok_e),
            "ours_card_count": ours_cards, "declared_card_count": declared_cards,
            "cards_match": (declared_cards is not None and ours_cards == declared_cards),
        }

    # 可转样本：team 侧「恰好 1 个 play_card」的帧（且帧起点 ≤ 300 s）
    fb = frames_of(rep, 0)
    out["labels_frames"]["team_frames_with_play"] = len(fb)
    out["labels_frames"]["team_single_event_frames"] = sum(1 for v in fb.values() if len(v) == 1)
    out["labels_frames"]["team_multi_event_frames"] = sum(1 for v in fb.values() if len(v) > 1)
    out["labels_frames"]["team_frames_after_cap"] = sum(
        1 for f in fb if f * FRAME_DT > TIME_CAP)
    return out


def run_reconcile(args):
    recs = []
    with open(args.jsonl, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    if args.limit:
        recs = recs[: args.limit]

    rows = [reconcile_one(r) for r in recs]
    keep = [r for r in rows if r["reject"] is None]
    reasons = collections.Counter(r["reject"] for r in rows if r["reject"])

    def side_totals(rs, side):
        ok_e = sum(1 for r in rs if r["sides"][side]["elixir_match"])
        ok_c = sum(1 for r in rs if r["sides"][side]["cards_match"])
        n = len(rs)
        return {"n": n, "elixir_match": ok_e, "elixir_match_rate": ok_e / n if n else None,
                "cards_match": ok_c, "cards_match_rate": ok_c / n if n else None,
                "declared_total_elixir": sum(r["sides"][side]["declared_total_elixir"] or 0
                                             for r in rs),
                "ours_total_elixir": sum(r["sides"][side]["ours_total_elixir"] for r in rs)}

    both_ok = sum(1 for r in keep
                  if r["sides"]["team"]["elixir_match"] and r["sides"]["opponent"]["elixir_match"])
    both_ok_rate = both_ok / len(keep) if keep else None
    #: 残差归因（脚本复算，禁手抄）：2×2 列联表 —— 「本侧是否出过 Mirror」×「本侧是否对账一致」
    conf = collections.Counter(
        (r["sides"][s]["mirror_plays"] > 0, r["sides"][s]["elixir_match"])
        for r in keep for s in ("team", "opponent"))
    deficit_no_mirror = conf[(False, False)]
    deficit_mirror = conf[(True, False)]
    n_mirror_sides = sum(v for k, v in conf.items() if k[0])
    mirror_verdict = ("ALL_RESIDUAL_HAS_MIRROR" if deficit_no_mirror == 0 and deficit_mirror > 0
                      else "NO_RESIDUAL" if deficit_mirror == 0 and deficit_no_mirror == 0
                      else "RESIDUAL_NOT_EXPLAINED_BY_MIRROR")
    labels = sum(r["labels_frames"]["team_single_event_frames"] for r in keep)
    out = {
        "source": {"jsonl": os.path.basename(args.jsonl), "replays_read": len(recs)},
        "filters": {"allowed_modes": list(ALLOWED_MODES), "time_cap_s": TIME_CAP,
                    "reject_reasons": dict(reasons),
                    "kept": len(keep), "kept_rate": len(keep) / len(recs) if recs else None},
        "J1_1_card_economy": {
            "team": side_totals(keep, "team"),
            "opponent": side_totals(keep, "opponent"),
            "both_sides_match": both_ok,
            "both_sides_match_rate": both_ok_rate,
            "gate": ">= 0.90",
            "verdict": ("PASS" if both_ok_rate is not None and both_ok_rate >= 0.90 else "FAIL"),
        },
        "J1_1_residual_diagnosis": {
            "contingency_has_mirror_x_elixir_match": {
                f"mirror={int(k[0])},match={int(k[1])}": v for k, v in sorted(conf.items())},
            "sides_with_mirror": n_mirror_sides,
            "deficit_sides_without_mirror": deficit_no_mirror,
            "deficit_sides_with_mirror": deficit_mirror,
            "verdict": mirror_verdict,
            "mechanism": ("口径 = 引擎实际费用：`Mirror` = 上一张牌费用 + 1（`rl/action_mask.py:30-36`），"
                          "且 `Mirror` 不覆盖 `last_card`（`player.py:47`）。第一版误用卡面费 "
                          "`Card('Mirror').elixir`(=1) ⇒ 残差**恰好**等于被复制卡的费用；"
                          "**成因在仪器，不在引擎**（引擎实现正确）。"),
        },
        "J1_2_labels": {
            "team_single_event_frames": labels,
            "team_frames_with_play": sum(r["labels_frames"]["team_frames_with_play"] for r in keep),
            "team_multi_event_frames": sum(r["labels_frames"]["team_multi_event_frames"] for r in keep),
            "team_frames_after_cap": sum(r["labels_frames"]["team_frames_after_cap"] for r in keep),
            "gate": ">= 20000",
            "verdict": ("PASS" if labels >= 20000 else "FAIL"),
        },
        "rows": rows,
    }
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[reconcile] read={len(recs)} kept={len(keep)} "
          f"({out['filters']['kept_rate']:.1%}) reasons={dict(reasons)}")
    print(f"  J1.1 card economy  : {out['J1_1_card_economy']['verdict']} "
          f"both-sides={both_ok}/{len(keep)} = {both_ok_rate}")
    for side in ("team", "opponent"):
        t = out["J1_1_card_economy"][side]
        print(f"      {side:8s} elixir {t['elixir_match']}/{t['n']} cards {t['cards_match']}/{t['n']} "
              f"declared_e={t['declared_total_elixir']:.0f} ours_e={t['ours_total_elixir']:.0f}")
    print(f"  J1.2 labels        : {out['J1_2_labels']['verdict']} "
          f"single-event frames={labels} "
          f"(multi={out['J1_2_labels']['team_multi_event_frames']}, "
          f"after_cap={out['J1_2_labels']['team_frames_after_cap']})")
    d = out["J1_1_residual_diagnosis"]
    print(f"  J1.1 residual      : {d['verdict']} "
          f"deficit sides w/o Mirror={d['deficit_sides_without_mirror']} "
          f"w/ Mirror={d['deficit_sides_with_mirror']} (sides with Mirror={d['sides_with_mirror']})")
    if args.out:
        print(f"[reconcile] wrote {args.out}")
    return 0


def _frames_to_next_elixir(elixir, t):
    """「距下 1 点圣水还有几帧」——**确定可算**（`battle.py:2881` 的三段回费）。

    回费周期：`t < 120` → 2.8 s/点；`120 ≤ t < 240` → 1.4 s/点（双倍）；`t ≥ 240` → `2.8/3` s/点。
    每决策帧 `FRAME_DT = 0.5 s` ⇒ 回费 **0.1786 / 0.3571 / 0.5357 点/帧**。
    圣水已封顶（≥10）⇒ 返回 **-1**（不再回费）。
    """
    if elixir >= 10.0 - 1e-9:
        return -1
    base = 2.8 if t < 120 else (1.4 if t < 240 else 2.8 / 3.0)
    per_frame = FRAME_DT / base
    frac = float(elixir) % 1.0
    rem = 1.0 if frac <= 1e-9 else (1.0 - frac)
    return int(np.ceil(rem / per_frame - 1e-9))


def _card_idx(name):
    """卡名 → `ENTITY_NAMES` 下标（`-1` = 未登记）。"""
    return ENTITY_NAMES.index(name) if name in ENTITY_NAMES else -1


def _has_play_option(mask):
    """我方掩码里**是否至少放行 1 个可出牌槽**（= 这一帧「有得选」）。

    口径与 `FollowerPolicy._slot_mask_tensor`（`rl/follower.py:377-380`）对齐：槽位合法性取自
    `mask["slots"]`；但我方**额外**要求该槽至少有一个合法落点（`mask["cells"][i]` 非全 0），
    否则「选得中槽、选不中格」，不是真正可选。两个口径分别计数（`stop_slot_legal_any` /
    `stop_save_cand`），主口径取**更严**的那个。

    ⚠️ 返回 `(slot_legal_any, cell_legal_any)`。
    """
    slots = mask.get("slots")
    cells = mask.get("cells")
    slot_any = bool(np.any(np.asarray(slots, dtype=bool))) if slots is not None else False
    cell_any = False
    if slots is not None and cells is not None:
        for i, ok in enumerate(slots):
            if bool(ok) and bool(np.any(np.asarray(cells[i], dtype=bool))):
                cell_any = True
                break
    return slot_any, cell_any


def _force_hand(ps, card, hand_slot=None):
    """把手牌强制成含 `card`（人类手牌顺序不可观测 ⇒ 这是**假设**，不是重建）。

    返回 `(swapped_out, idx0)`：`swapped_out=None` 表示本来就在手牌里。
    只在 `cycle` 内部做**对换** ⇒ 长度恒 8、牌集合不变；但 `next_card`（`cycle[4]`）会变
    （实测见 `runs/_probe_il/probe_sample_api.py` PART 3 / Q5a）。
    """
    if card in ps.cycle[:4]:
        return None, ps.cycle[:4].index(card)
    j = ps.cycle.index(card, 4)
    i = 3 if hand_slot is None else int(hand_slot) - 1
    ps.cycle[i], ps.cycle[j] = ps.cycle[j], ps.cycle[i]
    return ps.cycle[j], i


def _force_elixir(ps, cost):
    """只抬不降、上限夹到引擎的 10（与 `PlayerState.regenerate_elixir` 的 min(10,·) 一致）。"""
    before = float(ps.elixir)
    ps.elixir = min(10.0, max(before, float(cost)))
    return before, float(ps.elixir)


def _convert_one(idx, rep, is_hold, out_dir, level=11, coord="raw", detail=False,
                 stop_mode="none", stop_stride=1, with_frames=False, dump_dir=None,
                 dump_grid=False, plan_extras=False, evo_slots=False):
    """一局 → IL 样本（写分片 pkl）+ 逐局读数。**必须模块级**（Windows multiprocessing 是 spawn）。"""
    cfg = TrainConfig.resolve("standard")
    deck0 = list(rep["_decks"]["team"])
    deck1 = list(rep["_decks"]["opponent"])
    #: `opponent` 必须显式给**被动**策略：`None` 在 RLEnv 里 = 每帧随机对手（env_wrapper.py:491-492,
    #: 522-537）⇒ 会凭空改动战局。人类对手的出牌由本函数手工注入（见下）。
    env = RLEnv(opponent=lambda obs: ActionBundle.noop(), seed=0,
                reward_weights=reward_to_env(cfg), card_level=level,
                deck0=deck0, deck1=deck1, record_hidden=False)
    env.reset()
    belief = BeliefInference(opp_deck=env.deck1, n_particles=128, seed=0)
    belief.reset(env.deck1)
    bp = BeliefPlanner()
    policy = FollowerPolicy(hidden=128, plan_dim=PLAN_DIM,
                           belief_dim=belief_token_dim(env.deck1))
    p0, p1 = env.battle.players[0], env.battle.players[1]

    st = {"tag": rep["tag"], "dur": rep["dur"], "holdout": bool(is_hold), "frames": 0,
          "team_single": 0, "team_multi": 0, "opp_plays": 0, "hand_forced": 0,
          "elixir_forced": 0, "mask_reject": 0, "bundle_not_ok": 0, "labels": 0,
          "errors": 0, "end_time": 0.0, "game_over": False,
          #: ★ 2026-09-21 新增：**非出牌决策帧普查**（stop_mode != none 时有效）
          "team_off": 0, "stop_slot_legal_any": 0, "stop_save_cand": 0, "stop_forced": 0,
          "stop_labels": 0, "stop_elixir": [], "team_unmapped": 0}

    #: ★ 觉醒 S1（预注册 §2）：从回放牌组后缀声明觉醒位（`--evo-slots` 才开）。
    #: **必须在 `env.reset()` 之后**（`reset()` 会重建 `PlayerState`，之前的声明会被丢掉）、
    #: 且在**第一帧 `deploy_card` 之前**（声明要在任何一次出牌前生效）。
    #: 关掉时 `st` 里一个字段都不多、`env.battle` 一行不碰 ⇒ **旧路径逐位不变**（【R2】）。
    if evo_slots:
        _decl, _unmapped, _trunc = {}, {}, {}
        for _pid, _side in ((0, "team"), (1, "opponent")):
            _cards, _un = evo_slot_cards((rep.get("_evo") or {}).get(_side, []))
            _decl[_side] = _cards
            _unmapped[_side] = _un
            #: `set_evolution_slots` 内部 `[:2]` 截断（`player.py:17-19`，官方上限 2 槽）
            #: ⇒ 多出来的计入 `truncated`（**不静默**：截断会低估觉醒触发率）。
            _trunc[_side] = max(0, len(_cards) - 2)
            if _cards:
                env.battle.players[_pid].set_evolution_slots(_cards)
        st["evo_declared"] = _decl
        st["evo_unmapped"] = _unmapped
        st["evo_declared_any"] = bool(_decl["team"] or _decl["opponent"])
        st["evo_truncated"] = _trunc
        install_evo_counters(env.battle, st)

    sched = collections.defaultdict(lambda: {"t": [], "o": []})
    max_frame = int(TIME_CAP / FRAME_DT)
    #: ⚠️ 实测：**部分事件的 `native_x/native_y` 是 None**（回放自带的 `validation.coordinate_event_count`
    #: 本就会小于 `normalized_event_count`）⇒ 无坐标的事件无法注入引擎，跳过并计数。
    no_coord = 0
    for e in rep["ev"]:
        if e[0] != 0 or not isinstance(e[3], (int, float)):
            continue
        if e[4] is None or e[5] is None:
            no_coord += 1
            continue
        k = int(float(e[3]) / 20.0 / FRAME_DT)
        if 0 <= k <= max_frame:
            sched[k]["t" if e[1] == 0 else "o"].append(e)
    st["no_coord"] = no_coord

    samples = []
    frames_meta = []
    #: ★ W2/W3（`--plan-extras`）：本局**对手已打出过的不同卡数**的原料（确定性可读，【R12】）。
    #: 一局一个集合（跨帧累积、**不重置**）；未开启时恒 None ⇒ 旧路径零改动。
    _known_opp = set() if plan_extras else None
    #: ★ 2026-09-21 Stage 0：逐决策帧特征导出（F_obs + F_oracle）⇒ 上界诊断的数据源
    dump = [] if dump_dir else None
    for k in range(max_frame + 1):
        if env.battle.game_over:
            break
        bucket = sched.get(k)
        injected, label_bundle = [], None
        #: ★ 2026-09-21：「本帧人类有没有出牌」= 是否有 team play_card 事件（**与标签是否可用无关**）。
        #: 只有这一位为 False 的帧才是「人选择不出牌」⇒ 才是 STOP/攒费帧的候选。
        ev_t = (bucket["t"] if bucket else [])
        has_team_play = bool(ev_t)
        if bucket:
            for e in bucket["o"]:                      # 对手侧：只注入，不产标签
                card = map_key(e[2] or "")[0]
                if card is None:
                    continue
                _force_hand(p1, card)
                c = _card_cost(p1, card)
                if c is not None:
                    _force_elixir(p1, c)
                env.battle.deploy_card(1, card,
                                       Position(float(e[4]) / 1000.0, float(e[5]) / 1000.0))
                injected.append(card)
                if _known_opp is not None:
                    _known_opp.add(card)
                st["opp_plays"] += 1

            ev_t = bucket["t"]
            if len(ev_t) == 1:
                e = ev_t[0]
                card = map_key(e[2] or "")[0]
                if card is None:
                    st["team_unmapped"] += 1
                else:
                    st["team_single"] += 1
                    swapped, si = _force_hand(p0, card)
                    if swapped is not None:
                        st["hand_forced"] += 1
                    c = _card_cost(p0, card)
                    if c is not None:
                        before, _after = _force_elixir(p0, c)
                        if before + 1e-9 < c:
                            st["elixir_forced"] += 1
                    else:
                        before = None
                    #: `--detail`：逐标签留证（时间/牌/费用/强制前圣水），用于核「圣水模型保真度」
                    if detail:
                        st.setdefault("detail", []).append(
                            [round(float(env.battle.time), 2), card, c,
                             (None if before is None else round(before, 3)), swapped is not None])
                    wx, wy = float(e[4]) / 1000.0, float(e[5]) / 1000.0
                    if coord == "flip_y":
                        wy = 32.0 - wy
                    lx = min(max(int(wx), 0), 17)
                    ly = min(max(int(wy), 0), 31)
                    bundle = ActionBundle.from_single(si + 1, lx, ly)
                    m = env.get_action_mask()
                    if bool(m["slots"][si]) and bool(m["cells"][si][ly][lx]):
                        label_bundle = bundle
                    else:
                        #: **掩码比引擎更严**（法术空砸/砸塔 EV/后排几何/不裸下闸门）
                        #: ⇒ 真人动作可能「引擎合法但掩码非法」。本轮处置 = 丢弃该标签，
                        #: 但仍把这一手**打进战局**（否则后续状态会发散），并计数。
                        st["mask_reject"] += 1
                        env.battle.deploy_card(0, card, Position(wx, wy))
            elif len(ev_t) > 1:
                st["team_multi"] += 1
                for e in ev_t:                          # 同帧 ≥2 手：不产标签，但打进战局
                    card = map_key(e[2] or "")[0]
                    if card is None:
                        continue
                    _force_hand(p0, card)
                    c = _card_cost(p0, card)
                    if c is not None:
                        _force_elixir(p0, c)
                    env.battle.deploy_card(0, card, Position(float(e[4]) / 1000.0,
                                                             float(e[5]) / 1000.0))

        #: ★ 2026-09-21：STOP / 攒费帧候选判定（**只在人本帧没出牌时**）。
        #: `stop_mode="save"` 只留「掩码放行 ≥1 出牌槽」的帧（有得选而不选 = 真正的攒费决策）；
        #: `stop_mode="all"` 留全部非出牌帧。掩码禁掉一切的帧本来无选择 ⇒ 其 STOP 项梯度 ≈ 0。
        stop_keep = False
        if label_bundle is None and not has_team_play and stop_mode != "none":
            slot_any, cell_any = _has_play_option(env.get_action_mask())
            st["team_off"] += 1
            if slot_any:
                st["stop_slot_legal_any"] += 1
            if cell_any:
                st["stop_save_cand"] += 1
            else:
                st["stop_forced"] += 1
            stop_keep = (cell_any if stop_mode == "save" else True)
            if stop_keep:
                stop_keep = (stop_stride <= 1) or (k % stop_stride == 0)

        if dump is not None:
            m0 = env.get_action_mask()
            #: ★ Stage 0 尾项：**粗网格**（4×4 平均池化 → 8×5×15 = 600 维）
            #: 用于回答「act 能不能从**棋盘**读出来」（`il_act_upper_bound.py --grid-coarse`）
            _g = np.asarray(env.observe(0)["grid"], dtype=np.float32)
            if dump_grid:
                _ph, _pw = (-_g.shape[0]) % 4, (-_g.shape[1]) % 4
                if _ph or _pw:
                    _g = np.pad(_g, ((0, _ph), (0, _pw), (0, 0)))
                _gh, _gw = _g.shape[0] // 4, _g.shape[1] // 4
                grid_c = _g.reshape(_gh, 4, _gw, 4, _g.shape[2]).mean(axis=(1, 3)).reshape(-1)
            slot_any, cell_any = _has_play_option(m0)
            bst = belief.state()
            ev = list(belief.event_history)[-3:]
            ev_c = [_card_idx(c) for c, _x, _y, _dt in ev]
            ev_x = [float(x if x is not None else -1) for _c, x, _y, _dt in ev]
            ev_y = [float(y if y is not None else -1) for _c, _x, y, _dt in ev]
            ev_dt = [float(dt) for _c, _x, _y, dt in ev]
            while len(ev_c) < 3:                       # 左填充到 3 条（早期帧不足）
                ev_c.insert(0, -1); ev_x.insert(0, -1.0); ev_y.insert(0, -1.0); ev_dt.insert(0, -1.0)
            dump.append(dict(
                frame=k, act=int(has_team_play), label_ok=int(label_bundle is not None),
                opp_play_count=int(st["opp_plays"]),
                opt_slot=int(slot_any), opt_cell=int(cell_any),
                my_elixir=float(p0.elixir), time=float(env.battle.time),
                next_card=float(_card_idx(p0.cycle[4])),
                next_card_cost=float(_card_cost(p0, p0.cycle[4]) or -1.0),
                f2n_my=float(_frames_to_next_elixir(p0.elixir, env.battle.time)),
                b_hand=np.asarray(bst.hand_probs, dtype=np.float32),
                b_next=np.asarray(bst.next_probs, dtype=np.float32),
                b_elixir=float(bst.elixir_mean), b_unc=float(bst.uncertainty),
                opp_elixir=float(p1.elixir), f2n_opp=float(_frames_to_next_elixir(p1.elixir, env.battle.time)),
                opp_hand=np.asarray([_card_idx(c) for c in p1.cycle[:4]], dtype=np.int32),
                opp_cycle=np.asarray([_card_idx(c) for c in p1.cycle], dtype=np.int32),
                ev_c=np.asarray(ev_c, dtype=np.int32),
                ev_x=np.asarray(ev_x, dtype=np.float32),
                ev_y=np.asarray(ev_y, dtype=np.float32),
                ev_dt=np.asarray(ev_dt, dtype=np.float32),
                **({"grid_c": grid_c.astype(np.float32)} if dump_grid else {}),
            ))

        emit_bundle, emit_kind = None, None
        if label_bundle is not None:
            emit_bundle, emit_kind = label_bundle, "play"
        elif stop_keep:
            emit_bundle, emit_kind = ActionBundle(), "save"

        if emit_bundle is not None:
            #: 与 `rl/human_play.py:142-151` **逐句同序**：obs → belief → plan → masks → step
            obs = env.observe(0)
            tok = belief.encode(obs, None)
            if plan_extras:
                #: ★ W2/W3（预注册 `docs/il_whiff_handscore_prereg_2026-09-22.md` §2.3/§3.1）：
                #: **plan 尾部追加** 17 维（手牌打分 13 + 前期卡组信息分 4）。追加而不是改布局
                #: ⇒ `plan_mlp.0.weight` 的「前列拷贝 + 尾零」兼容分支仍然有效（旧 ckpt 可加载）。
                #: 原料与 `bp.plan(...)` **同一帧同一 `BeliefState`**（`_bst` 只取一次，两次取也同值）。
                _bst = belief.state()
                plan_vec = bp.plan(env.battle, _bst, obs).to_vector()
                _extras = plan_extras_vec(
                    own_cards=list(p0.cycle[:4]), own_elixir=float(p0.elixir),
                    opp_cards=list(env.deck1), opp_hand_probs=_bst.hand_probs,
                    opp_elixir_est=float(_bst.elixir_mean),
                    time_s=float(env.battle.time), known_opp=len(_known_opp))
                plan_vec = np.concatenate([plan_vec, _extras]).astype(np.float32)
                st["plan_dim"] = int(plan_vec.shape[0])
            else:
                plan_vec = bp.plan(env.battle, belief.state(), obs).to_vector()
            masks = policy.masks_for(obs, tok, plan_vec, emit_bundle, env.get_action_mask)
            samples.append((obs, tok, plan_vec, emit_bundle, masks))
            frames_meta.append([k, emit_kind])
            if emit_kind == "play":
                st["labels"] += 1
            else:
                st["stop_labels"] += 1
                if len(st["stop_elixir"]) < 64:      # 普查留证：save 帧的圣水（重建值）
                    st["stop_elixir"].append(round(float(p0.elixir), 2))
            step_bundle = emit_bundle
        else:
            step_bundle = ActionBundle.noop()

        obs2, _r, term, trunc, info = env.step(step_bundle)
        if label_bundle is not None and info.get("bundle_ok") is False:
            st["bundle_not_ok"] += 1
        belief.update(obs2, injected or info.get("opp_played"))
        st["frames"] += 1
        st["end_time"] = round(float(env.battle.time), 2)
        if term or trunc:
            break
    st["game_over"] = bool(env.battle.game_over)

    if samples:
        path = os.path.join(out_dir, "bc_fl_%04d.pkl" % idx)
        with open(path, "wb") as f:
            pickle.dump(samples, f)
        st["out"] = os.path.basename(path)
        st["out_n"] = len(samples)
        if with_frames:
            #: 帧序旁挂（**不改 pkl 的 5 元组格式**）⇒ 为 L6 时序 BC 预留；`[帧号, "play"/"save"]`
            fp = os.path.join(out_dir, "frames_fl_%04d.json" % idx)
            with open(fp, "w", encoding="utf-8", newline="\n") as f:
                json.dump(frames_meta, f)
    if dump_dir and dump:
        #: ★ Stage 0 逐决策帧特征（F_obs + F_oracle）⇒ `il_act_upper_bound.py` /
        #: `il_opp_prediction.py` 的输入；**每局一个 npz**（按局分组留出，【R9】）
        keys = list(dump[0].keys())
        arr = {}
        for key in keys:
            col = [row[key] for row in dump]
            arr[key] = (np.asarray(col, dtype=np.int32) if key in
                        ("frame", "act", "label_ok", "opt_slot", "opt_cell", "opp_play_count")
                        else np.stack(col) if isinstance(col[0], np.ndarray)
                        else np.asarray(col, dtype=np.float32))
        arr["tag"] = np.array(rep["tag"])
        arr["holdout"] = np.int32(bool(is_hold))
        #: 对手牌组**顺序**（`belief.next_probs` 的下标语义 = `env.deck1` 的位置）
        #: ⇒ 没有它就无法把信念的 argmax 映射回卡名（「预判下一手」的必需字段）
        arr["opp_deck"] = np.asarray([_card_idx(c) for c in env.deck1], dtype=np.int32)
        os.makedirs(dump_dir, exist_ok=True)
        np.savez_compressed(os.path.join(dump_dir, "feat_%04d.npz" % idx), **arr)
        st["dump_n"] = len(dump)
    if "_evo_counts" in st:
        #: 展平成可 JSON 序列化的顶层字段（`_evo_counts` 内部是嵌套 dict，留着也行，但显式更清楚）
        _c = st.pop("_evo_counts")
        st["evo_trig_finish"] = _c["finish"]
        st["evo_trig_wrap"] = _c["wrap"]
        st["evo_trig_finish_by_card"] = _c["finish_by_card"]
        st["evo_trig_wrap_by_card"] = _c["wrap_by_card"]
    return st


def _worker(task):
    (idx, rep, is_hold, out_dir, level, coord, detail,
     stop_mode, stop_stride, with_frames, dump_dir, dump_grid, plan_extras,
     evo_slots) = task
    try:
        return _convert_one(idx, rep, is_hold, out_dir, level, coord, detail,
                            stop_mode, stop_stride, with_frames, dump_dir, dump_grid,
                            plan_extras, evo_slots), 1
    except Exception as e:  # noqa: BLE001
        import traceback
        return {"tag": rep.get("tag"), "errors": 1,
                "err": f"{type(e).__name__}: {e}",
                "tb": traceback.format_exc()[-400:]}, 0


def run_samples(args):
    """回放 → IL 训练对（跑真引擎）。逐局转换，多进程分片。

    与 `rl/human_play.py:142-151` 的采集时序**逐句对齐**（obs → belief → plan → masks → step），
    唯一差别：动作不是人敲的，而是**回放里的人类事件**，并为此做两处显式假设（预注册 §1）：
      * **手牌强制**：人类手牌顺序不可观测 ⇒ 把要出的牌交换进手牌位（`cycle` 长度/牌集合不变）；
      * **圣水强制**：逐事件圣水不可观测 ⇒ `elixir = max(elixir, 实际费用)`；
        实际费用用 `rl/action_mask.py::_card_cost`（Mirror = 上一张牌费用 + 1，**不是卡面费**）。
    """
    import multiprocessing as mp_mod
    import time
    import hashlib

    recs = []
    with open(args.jsonl, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["gm"] not in ALLOWED_MODES:
                continue
            if not isinstance(r["dur"], (int, float)) or r["dur"] > TIME_CAP:
                continue
            names, vs_all = {}, {}
            bad = False
            for side in ("team", "opponent"):
                nm, vs, bd = deck_of(r, side)
                if bd or len(nm) != 8:
                    bad = True
                    break
                names[side] = nm
                vs_all[side] = vs
            if bad:
                continue
            if sum(1 for e in r["ev"] if e[0] == 0 and e[1] == 0) < 4:
                continue
            r["_decks"] = names
            #: ★ 觉醒 S1：**牌组变体后缀**（`-ev1/-hero/...`）必须一起带下去 ——
            #: 旧版只带 `names`（已剥后缀）⇒ 重建战局里觉醒位从来没被声明过（预注册 §1#8）。
            r["_evo"] = vs_all
            recs.append(r)
    if args.games:
        recs = recs[: args.games]
    print(f"[samples] 可转局数 = {len(recs)}（过滤后）")

    out_train = os.path.join(args.out_dir, "train")
    out_hold = os.path.join(args.out_dir, "holdout")
    os.makedirs(out_train, exist_ok=True)
    os.makedirs(out_hold, exist_ok=True)

    #: 留出切分：**按 replay_tag**（预注册 §1，防同局泄漏）；seed 写死 0。
    #: 用 md5 前 8 位（跨平台稳定，不用 Python 内置 hash —— 它有进程随机化）。
    holdout = [bool(int(hashlib.md5(("fl_il|%s" % r["tag"]).encode()).hexdigest()[:8], 16) % 5 == 0)
               for r in recs]
    print(f"[samples] 留出（tag 哈希 %5==0）= {sum(holdout)} / {len(recs)}")

    tasks = [(i, r, holdout[i], out_train if not holdout[i] else out_hold,
              args.level, args.coord, bool(args.detail),
              args.stop_mode, args.stop_stride, bool(args.with_frames),
              args.dump_frames, bool(args.dump_grid), bool(args.plan_extras),
              bool(args.evo_slots))
             for i, r in enumerate(recs)]
    stats, t0 = [], time.time()
    if args.workers and args.workers > 1:
        with mp_mod.Pool(args.workers) as pool:
            for k, (st, _done) in enumerate(pool.imap_unordered(_worker, tasks), 1):
                stats.append(st)
                if k % 25 == 0 or k == len(tasks):
                    el = time.time() - t0
                    print(f"  [{k}/{len(tasks)}] {el:.0f}s  ({el / k:.2f} s/局)  "
                          f"labels_kept={sum(s.get('labels', 0) for s in stats)}", flush=True)
    else:
        for k, t in enumerate(tasks, 1):
            st, _done = _worker(t)
            stats.append(st)
            if k % 10 == 0:
                print(f"  [{k}/{len(tasks)}] labels_kept={sum(s.get('labels', 0) for s in stats)}",
                      flush=True)

    agg = collections.Counter()
    for s in stats:
        for key in ("frames", "team_single", "team_multi", "opp_plays", "hand_forced",
                    "elixir_forced", "mask_reject", "bundle_not_ok", "labels", "errors",
                    "no_coord", "team_off", "stop_slot_legal_any", "stop_save_cand",
                    "stop_forced", "stop_labels", "team_unmapped"):
            agg[key] += s.get(key, 0)
    n_lab = agg["labels"]
    elix = sorted(v for s in stats for v in s.get("stop_elixir", []))
    pct = (lambda q: (elix[min(len(elix) - 1, int(q * len(elix)))] if elix else None))
    j6 = {
        "stop_mode": args.stop_mode, "stop_stride": args.stop_stride,
        "team_frames_total": agg["frames"],
        "team_off_frames": agg["team_off"],
        "stop_slot_legal_any": agg["stop_slot_legal_any"],
        "stop_save_cand": agg["stop_save_cand"],
        "stop_forced": agg["stop_forced"],
        "stop_labels_written": agg["stop_labels"],
        "play_labels_written": n_lab,
        "team_unmapped_events": agg["team_unmapped"],
        "save_frame_share_of_off": (agg["stop_save_cand"] / agg["team_off"]
                                    if agg["team_off"] else None),
        "stop_elixir_quantiles": {"p10": pct(0.10), "p25": pct(0.25), "p50": pct(0.50),
                                  "p75": pct(0.75), "p90": pct(0.90), "n": len(elix)},
        "note": ("J6.1 普查（描述性，无门禁）。`team_off` = 人类本帧没出牌的决策帧（旧口径**被丢弃**）；"
                 "`stop_save_cand` = 其中我方掩码放行 ≥1 个**有合法落点**的出牌槽 ⇒ 真正的攒费决策；"
                 "`stop_forced` = 掩码禁掉一切 ⇒ 无选择（STOP 项梯度 ≈ 0）。"
                 "⚠️ 判定依赖我方重建圣水（elixir_forced 率见 J2.1）"),
    }
    res = {
        "source": {"jsonl": os.path.basename(args.jsonl), "games_converted": len(stats),
                   "games_selected": len(recs), "workers": args.workers,
                   "elapsed_s": round(time.time() - t0, 1),
                   "card_level": args.level, "coord": args.coord,
                   "stop_mode": args.stop_mode, "stop_stride": args.stop_stride,
                   "plan_extras": bool(args.plan_extras),
                   "evo_slots": bool(args.evo_slots),
                   "dump_frames": (os.path.basename(args.dump_frames) if args.dump_frames else None)},
        "split": {"holdout_by": "md5(tag) % 5 == 0", "holdout_games": sum(holdout),
                  "train_games": len(recs) - sum(holdout)},
        "aggregate": dict(agg),
        "J6_1_stop_frames": j6,
        "J1_2_labels": {"labels": n_lab, "gate": ">= 20000",
                        "verdict": "PASS" if n_lab >= 20000 else "FAIL"},
        "J2_1_reconstruction": {
            "hand_force_rate": (agg["hand_forced"] / agg["team_single"]) if agg["team_single"] else None,
            "elixir_force_rate": (agg["elixir_forced"] / agg["team_single"]) if agg["team_single"] else None,
            "mask_reject_rate": (agg["mask_reject"] / agg["team_single"]) if agg["team_single"] else None,
            "bundle_not_ok_rate": (agg["bundle_not_ok"] / agg["team_single"]) if agg["team_single"] else None,
            "gate_bundle_not_ok": "<= 0.05",
            "verdict": ("PASS" if agg["team_single"] and agg["bundle_not_ok"] / agg["team_single"] <= 0.05
                        else "FAIL"),
        },
        "per_game": stats,
    }
    #: ★ 觉醒 S1 读数块（预注册 `docs/il_evo_prereg_2026-09-22.md` §2 S1 判据①/③）。
    #: 只在 `--evo-slots` 时出现（关掉 = 旧 JSON schema 逐字段不变，【R2】）。
    if args.evo_slots:
        _fin = {"team": 0, "opp": 0}
        _wrp = {"team": 0, "opp": 0}
        _trig_by_card = collections.Counter()
        _wrap_by_card = collections.Counter()
        _unmapped_keys = collections.Counter()
        _decl_slots = {"team": 0, "opponent": 0}
        _trunc = 0
        _games_declared = 0
        for s in stats:
            if s.get("evo_declared_any"):
                _games_declared += 1
            for _side in ("team", "opponent"):
                _decl_slots[_side] += len((s.get("evo_declared") or {}).get(_side) or [])
                _trunc += ((s.get("evo_truncated") or {}).get(_side) or 0)
                for _u in ((s.get("evo_unmapped") or {}).get(_side) or []):
                    _unmapped_keys[_u["key"]] += 1
            for _k in ("team", "opp"):
                _fin[_k] += (s.get("evo_trig_finish") or {}).get(_k, 0)
                _wrp[_k] += (s.get("evo_trig_wrap") or {}).get(_k, 0)
            _trig_by_card.update(s.get("evo_trig_finish_by_card") or {})
            _wrap_by_card.update(s.get("evo_trig_wrap_by_card") or {})
        _n_declared = sum(_decl_slots.values())
        _n_unmapped = sum(_unmapped_keys.values())
        _n_ev = _n_declared + _n_unmapped
        _cov = (_n_declared / _n_ev) if _n_ev else None
        res["S1_evo"] = {
            "flag": "--evo-slots",
            "games_converted": len(stats),
            "games_with_declaration": _games_declared,
            "declaration_rate": (_games_declared / len(stats)) if stats else None,
            "ev_slots_total": _n_ev,
            "ev_slots_declared": _n_declared,
            "ev_slots_unmapped": _n_unmapped,
            "ev_slots_truncated_at_2": _trunc,
            "coverage": _cov,
            "coverage_gate": ">= 0.95",
            "coverage_verdict": ("PASS" if (_cov is not None and _cov >= 0.95) else "FAIL"),
            "unmapped_keys": dict(_unmapped_keys.most_common(20)),
            "declared_by_side": _decl_slots,
            "triggers_finish": _fin,
            "triggers_finish_total": _fin["team"] + _fin["opp"],
            "triggers_wrap": _wrp,
            "triggers_by_card": dict(_trig_by_card.most_common(30)),
            "triggers_wrap_by_card": dict(_wrap_by_card.most_common(30)),
            "cross_check_note": ("`finish` = 主口径（包 `_finish_deploy`，覆盖部队/建筑/法术），"
                                 "口径 = **觉醒出牌次数**；`wrap` = 交叉核对（读引擎出生队列的觉醒标记位，"
                                 "只覆盖部队/建筑），口径 = **觉醒实体个数** ⇒ 两者**量纲不同**："
                                 "单兵卡应 `wrap == finish`，n 兵卡应 `wrap == n × finish`，"
                                 "法术觉醒只进 `finish`（不走 `_wrap`）。"
                                 "对不上或两边全 0 都**照实报**，不调和（【R10】/【R17】）。"),
            "prereg": "docs/il_evo_prereg_2026-09-22.md §2 S1",
        }
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"[samples] frames={agg['frames']} team_single={agg['team_single']} "
          f"labels={n_lab} ({res['J1_2_labels']['verdict']})")
    print(f"  hand_force={res['J2_1_reconstruction']['hand_force_rate']} "
          f"elixir_force={res['J2_1_reconstruction']['elixir_force_rate']} "
          f"mask_reject={res['J2_1_reconstruction']['mask_reject_rate']} "
          f"bundle_not_ok={res['J2_1_reconstruction']['bundle_not_ok_rate']} "
          f"({res['J2_1_reconstruction']['verdict']}) errors={agg['errors']}")
    print(f"  train dir = {out_train} / holdout dir = {out_hold}")
    if args.plan_extras:
        _pds = sorted({s.get("plan_dim") for s in stats if s.get("plan_dim")})
        print(f"  [W2/W3] plan_extras=ON | 落盘 plan_dim={_pds}（应为 [75] = 58+17）")
        if _pds != [75]:
            print(f"  [W2/W3] ⚠️ plan_dim 与预期 75 不一致：{_pds} —— 检查 `rl/hand_score.py` 口径")
    if args.evo_slots:
        _s1 = res["S1_evo"]
        print(f"  [S1 觉醒] 声明觉醒位的局数={_s1['games_with_declaration']}/{_s1['games_converted']} "
              f"| 槽位 声明={_s1['ev_slots_declared']} 未映射={_s1['ev_slots_unmapped']} "
              f"截断={_s1['ev_slots_truncated_at_2']} "
              f"| 覆盖率={_s1['coverage']} ({_s1['coverage_verdict']}，闸门 >=0.95)")
        print(f"  [S1 觉醒] **触发次数**(finish/主口径) 我方={_fin['team']} 对手={_fin['opp']} "
              f"| (wrap/交叉核对) 我方={_wrp['team']} 对手={_wrp['opp']} "
              f"| 逐卡={_s1['triggers_by_card']}")
        if _fin["team"] + _fin["opp"] == 0:
            print("  [S1 觉醒] ⚠️ 触发次数 = 0 —— 预注册 §2 S1 判据③ FAIL（照实报，【R10】）")
    if args.stop_mode != "none":
        #: ⚠️ 实测踩过：`save_frame_share_of_off` 在「全部局都报错」（`team_off == 0`）时是 **None**，
        #: 直接 `:.4f` 会抛 `TypeError` ⇒ **把真正的失败（errors=N）盖掉**（本批就是这样被盖了一轮）。
        _so = j6["save_frame_share_of_off"]
        print(f"  [J6.1] stop_mode={args.stop_mode} stride={args.stop_stride} | "
              f"team_off={agg['team_off']} "
              f"slot_legal={agg['stop_slot_legal_any']} save_cand={agg['stop_save_cand']} "
              f"forced={agg['stop_forced']} | 落盘 play={n_lab} + save={agg['stop_labels']} "
              f"| save/off={('n/a' if _so is None else format(_so, '.4f'))}")
    if agg["errors"]:
        #: 同理：**worker 里的异常一律要能看见**（`_worker` 会把它吞进 `st['err']/['tb']`）。
        _e = next((s for s in stats if s.get("errors") and s.get("err")), None)
        print(f"  [errors] {agg['errors']}/{len(stats)} 局转换失败；首个样例 tag={_e.get('tag') if _e else None}")
        if _e:
            print(f"    err = {_e['err']}")
            print("    tb  = " + (_e.get("tb") or "").replace("\n", " | ")[-400:])
    return 0


def _abs(p):
    """把 CLI 相对路径按 `ORIG_CWD` 解析；并把 WSL 的 `/mnt/e/...` 译成 `E:\\...`。

    ⚠️ 必须译：本脚本跑在 **Windows venv** 上，而调用者常在 WSL bash 里敲命令 ⇒
    `/mnt/e/x` 在 Windows 眼里是「当前盘根下的 /mnt/e/x」，`os.path.abspath` 会给出
    `E:\\mnt\\e\\x`（实测踩过）。
    """
    m = p.replace("\\", "/")
    if m.startswith("/mnt/") and len(m) > 6:
        head, rest = m.split("/", 3)[2:]
        return head.upper() + ":\\" + rest.replace("/", "\\")
    return os.path.normpath(os.path.join(ORIG_CWD, p))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--mode", choices=("reconcile", "samples"), default="reconcile")
    ap.add_argument("--out", default=None)
    ap.add_argument("--out-dir", default=None, help="samples 模式的输出目录")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--games", type=int, default=0, help="samples 模式：只转前 N 局（0=全部，按过滤后顺序）")
    ap.add_argument("--workers", type=int, default=1, help="samples 模式：进程数")
    ap.add_argument("--detail", action="store_true", help="samples 模式：逐标签留证（体积大，只用于小样本）")
    ap.add_argument("--coord", choices=("raw", "flip_y"), default="raw")
    ap.add_argument("--level", type=int, default=11)
    ap.add_argument("--stop-mode", choices=("none", "save", "all"), default="none",
                    help="非出牌决策帧（STOP/攒费帧）采集口径：none=旧行为（只采人类出牌帧）；"
                         "save=只采「掩码放行 ≥1 个有合法落点的出牌槽」的帧（真正的攒费决策，本次主用）；"
                         "all=采全部非出牌帧。见 docs/fl_il_stopframe_prereg_2026-09-21.md")
    ap.add_argument("--stop-stride", type=int, default=1,
                    help="save/all 帧按全局帧号 k %% stride == 0 等距抽稀（确定性、无 RNG）；"
                         "候选数一律如实计数，抽稀只影响落盘量")
    ap.add_argument("--with-frames", action="store_true",
                    help="额外落 frames_fl_%04d.json（[帧号, play/save]）⇒ 为时序 BC 预留帧序")
    ap.add_argument("--dump-grid", action="store_true",
                    help="随 --dump-frames 一起落**粗网格**（4×4 平均池化 → 600 维）")
    ap.add_argument("--dump-frames", default=None,
                    help="逐**决策帧**导出 F_obs + F_oracle 特征到该目录（每局一个 npz）"
                         "⇒ Stage 0 上界诊断（docs/fl_il_il2_prereg_2026-09-22.md）")
    ap.add_argument("--plan-extras", action="store_true",
                    help="★ W2/W3：plan 尾部追加 17 维（手牌打分 13 + 前期卡组信息分 4）"
                         "⇒ plan 向量 58→75（`rl/hand_score.py`）。缺省关 = 旧路径**逐位不变**。"
                         "见 docs/il_whiff_handscore_prereg_2026-09-22.md §2.3/§3.1")
    ap.add_argument("--evo-slots", action="store_true",
                    help="★ 觉醒 S1（数据保真）：从回放牌组后缀 `-ev*` 解析并在重建战局里"
                         "`set_evolution_slots`（**必须元数据齐备**才声明：周期表 + `evo_raw`）。"
                         "`-hero` 是另一套机制，本开关**不接线**。缺省关 = 旧路径**逐位不变**。"
                         "见 docs/il_evo_prereg_2026-09-22.md §2 S1")
    args = ap.parse_args(argv)
    for name in ("jsonl", "out", "out_dir", "dump_frames"):
        v = getattr(args, name, None)
        if v and not os.path.isabs(v):
            setattr(args, name, _abs(v))
        elif v and v.replace("\\", "/").startswith("/mnt/"):
            setattr(args, name, _abs(v))
    return run_reconcile(args) if args.mode == "reconcile" else run_samples(args)


if __name__ == "__main__":
    sys.exit(main())
