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

from rl.env_wrapper import RLEnv  # noqa: E402
from rl.action_bundle import ActionBundle  # noqa: E402
from rl.action_mask import _card_cost  # noqa: E402
from rl.belief import BeliefInference, belief_token_dim  # noqa: E402
from rl.belief_planner import BeliefPlanner  # noqa: E402
from rl.plan_space import PLAN_DIM  # noqa: E402
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


def _convert_one(idx, rep, is_hold, out_dir, level=11, coord="raw", detail=False):
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
          "errors": 0, "end_time": 0.0, "game_over": False}

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
    for k in range(max_frame + 1):
        if env.battle.game_over:
            break
        bucket = sched.get(k)
        injected, label_bundle = [], None
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
                st["opp_plays"] += 1

            ev_t = bucket["t"]
            if len(ev_t) == 1:
                e = ev_t[0]
                card = map_key(e[2] or "")[0]
                if card is not None:
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

        if label_bundle is not None:
            #: 与 `rl/human_play.py:142-151` **逐句同序**：obs → belief → plan → masks → step
            obs = env.observe(0)
            tok = belief.encode(obs, None)
            plan_vec = bp.plan(env.battle, belief.state(), obs).to_vector()
            masks = policy.masks_for(obs, tok, plan_vec, label_bundle, env.get_action_mask)
            samples.append((obs, tok, plan_vec, label_bundle, masks))
            st["labels"] += 1
            step_bundle = label_bundle
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
    return st


def _worker(task):
    idx, rep, is_hold, out_dir, level, coord, detail = task
    try:
        return _convert_one(idx, rep, is_hold, out_dir, level, coord, detail), 1
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
            names = {}
            bad = False
            for side in ("team", "opponent"):
                nm, _vs, bd = deck_of(r, side)
                if bd or len(nm) != 8:
                    bad = True
                    break
                names[side] = nm
            if bad:
                continue
            if sum(1 for e in r["ev"] if e[0] == 0 and e[1] == 0) < 4:
                continue
            r["_decks"] = names
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
              args.level, args.coord, bool(args.detail))
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
                    "elixir_forced", "mask_reject", "bundle_not_ok", "labels", "errors", "no_coord"):
            agg[key] += s.get(key, 0)
    n_lab = agg["labels"]
    res = {
        "source": {"jsonl": os.path.basename(args.jsonl), "games_converted": len(stats),
                   "games_selected": len(recs), "workers": args.workers,
                   "elapsed_s": round(time.time() - t0, 1),
                   "card_level": args.level, "coord": args.coord},
        "split": {"holdout_by": "md5(tag) % 5 == 0", "holdout_games": sum(holdout),
                  "train_games": len(recs) - sum(holdout)},
        "aggregate": dict(agg),
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
    args = ap.parse_args(argv)
    for name in ("jsonl", "out", "out_dir"):
        v = getattr(args, name, None)
        if v and not os.path.isabs(v):
            setattr(args, name, _abs(v))
        elif v and v.replace("\\", "/").startswith("/mnt/"):
            setattr(args, name, _abs(v))
    return run_reconcile(args) if args.mode == "reconcile" else run_samples(args)


if __name__ == "__main__":
    sys.exit(main())
