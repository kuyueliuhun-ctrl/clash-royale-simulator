# -*- coding: utf-8 -*-
"""攒费意图（intent-save）**目标归因**仪器：攒的到底是哪一张牌、等了多少帧、还差多少费。

起因（2026-09-20 用户提问）：「你所谓的攒费帧，是不是依旧在为一张最便宜的卡等待？」
⇒ 当时的判读只有 `set/held/ready/fired` **总数**，**没有按目标卡拆开** ⇒ 问不出答案。
本仪器补上三件事：

  ① **逐帧归因**：pending 帧按「目标槽 → 卡名/费用」拆开（`held` 只给总数时，
     一张 1 费卡等 1 帧与一张 6 费卡等 5 帧在数字上无法区分）；
  ② **事件级**：每次下意图那一刻的圣水 / **缺口 gap = 费用 − 圣水** / 该目标是不是
     **当时最便宜的买不起的卡** / 已等帧数 / 结局（fired / cancelled / dropped / 未结）；
  ③ **按卡自己的阈值**看"够不够长"：攒够一张牌需要 `费用 ÷ 每帧回费` 帧，
     6→34、5→28、4→23、3→17、2→12、1→6（`player.regenerate_elixir` 的
     `base_regen_time=2.8` 是**唯一常量源**，本脚本从签名取，不手抄）。

★ **本仪器写出来的直接原因是一处仪器缺陷**（2026-09-20 发现，见
`docs/intent_target_attribution_2026-09-20.md`）：`scripts/audit_intent_save.py` 原版
构造 `RLEnv(opponent=None, seed=..., intent_save=True)` **不传 deck** ⇒ 落到
`DEFAULT_DECK` / `DEFAULT_DECK_1`（原版 8 卡，最高 **5** 费、**没有 Xbow**），
而训练用的是 `rl.train_solo.DEFAULT_SOLO_DECK`（Xbow 2.9，**Xbow = 6 费**）。
后果：判据 J1 的「≥6 费卡」过滤在**那个卡组上按构造恒为 0**（空判据），
且 J1 的 34 帧阈值对应的牌根本不在场上。本仪器默认用**训练卡组**，
并把 `--deck vanilla` 作为**复现旧读数**的对照臂保留。

用法：

    ./.venv/Scripts/python.exe scripts/probe_intent_target.py \
        --ckpt src/clasher_new/runs/intent_on_100k/solo_main.pt --games 20 --deck solo

`--deck`：`solo`（缺省，= 训练卡组 Xbow 2.9）/ `vanilla`（= `DEFAULT_DECK_1`，复现旧读数）。

纪律：只跑**评估**，不训练、不写 ckpt；阈值全部来自常量换算（【R16】不观测标定）；
判读路径退出码恒 0（归因结果是科学结论，不是运行失败），只有参数/架构错误退 2。
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from datetime import datetime

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src", "clasher_new")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from rl.io_bootstrap import force_utf8_stdout  # noqa: E402
force_utf8_stdout()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from player import PlayerState  # noqa: E402  （engine：回费率唯一常量源）
from card_utils import Card  # noqa: E402
from rl.env_wrapper import RLEnv, DEFAULT_DECK_1  # noqa: E402
from rl.action_bundle import K_MAX  # noqa: E402
from rl.action_mask import _card_cost  # noqa: E402
from rl.train_solo import DEFAULT_SOLO_DECK, resolve_deck_set  # noqa: E402
from rl.belief import BeliefInference  # noqa: E402
from rl.belief_planner import BeliefPlanner  # noqa: E402
from rl.follower import load_checkpoint  # noqa: E402
from rl.plan_space import PLAN_DIM  # noqa: E402

#: 回费率唯一常量源：`player.PlayerState.regenerate_elixir(base_regen_time=2.8)`
BASE_REGEN_TIME = float(inspect.signature(PlayerState.regenerate_elixir)
                        .parameters["base_regen_time"].default)
#: 每决策帧回费（决策帧 = 0.5 s；`RLEnv.decision_frames=30 × dt=1/60`）
DECISION_SECONDS = 0.5
ELIXIR_PER_FRAME = (1.0 / BASE_REGEN_TIME) * DECISION_SECONDS
#: J1 原文的「≥6 费卡」下限（判据可满足性检查用；不在这里改判据）
J1_MIN_COST = 6.0
#: gap（费用 − 圣水）分桶
GAP_BUCKETS = ((0.0, 0.5, "<=0.5"), (0.5, 1.0, "0.5-1"), (1.0, 2.0, "1-2"),
               (2.0, 3.0, "2-3"), (3.0, 4.0, "3-4"), (4.0, 10 ** 9, ">=4"))
DEFAULT_JSON = os.path.join(_ROOT, "docs", "probe_intent_target.json")
#: JSON 里的**列式**段记录（键名不逐段重复 ⇒ 体积约 1/4；同数据，逐值可复算）
EPISODE_COLUMNS = ("game", "t_open", "slot", "card", "cost", "elixir_open", "gap_open",
                   "n_unaffordable_open", "unaff_min", "unaff_max", "n_costs", "real_choice",
                   "rank", "frames", "held", "held_info", "hold_raw", "ready_frames",
                   "age_max", "threshold", "reached_threshold", "outcome")


def _cell(e, col):
    """列式落盘的取值（缺失键 → None；`real_choice`/`reached_threshold` 落 bool）。"""
    return e.get(col)


def frames_needed(cost: float) -> int:
    """攒够一张 `cost` 费牌需要的最少决策帧（从 0 圣水起算，向上取整）。"""
    return int(np.ceil(float(cost) / ELIXIR_PER_FRAME))


def _deck_for(which: str):
    if which == "vanilla":
        return list(DEFAULT_DECK_1)
    return list(resolve_deck_set("default")[0])


def _unaffordable(p, elixir):
    """当前买不起的槽 ⇒ [(slot(1-based), card, cost)]（= 掩码里 `intent_slots` 为真的槽）。"""
    out = []
    if p.king_tower_hp <= 0:
        return out
    for i in range(K_MAX):
        c = _card_cost(p, p.cycle[i])
        if c is not None and float(elixir) < float(c):
            out.append((i + 1, str(p.cycle[i]), float(c)))
    return out


def _bucket_gap(v):
    for lo, hi, label in GAP_BUCKETS:
        if lo <= float(v) < hi:
            return label
    return "?"


def _run_game(env, pol, bp, belief, args):
    obs, _ = env.reset()
    belief.reset(env.deck1)
    hidden = None
    frames = []
    done = False
    n = 0
    cur = None
    eps = []
    while not done and n < args.max_frames:
        plan = bp.plan(env.battle, belief.state(), obs)
        tok = belief.encode(obs, None)
        bundle, _lp, _val, hidden, _masks = pol.act(
            obs, tok, plan.to_vector(),
            lambda b: env.get_action_mask_for(0, b),
            hidden=hidden, deterministic=args.deterministic)
        obs, _r, term, trunc, info = env.step(bundle)
        belief.update(obs, info.get("opp_played"))
        it = info.get("intent") or {}
        ev = it.get("event") or {}
        slot = int(it.get("slot", 0))
        age = int(it.get("age", 0))
        hold = bool(it.get("hold", False))
        p0 = env.battle.players[0]
        el = float(p0.elixir)
        # ★ 先结算本帧事件（2026-09-20 修）：`fired`/`cancelled`/`dropped` 都会在同一帧把
        # `_intent_slot` 清 0 ⇒ 若先按"槽位变了"关闭段，本帧事件就记不到该段上，
        # 全部段会被误记成"换目标"（首版真踩到：999 段 Xbow 全记 open、fired/cancelled=0）。
        if cur is not None:
            cur["fired"] = cur["fired"] or bool(ev.get("intent_fired"))
            cur["cancelled"] = cur["cancelled"] or bool(ev.get("intent_cancelled"))
            cur["dropped"] = cur["dropped"] or bool(ev.get("intent_dropped"))
        if cur is not None and cur["slot"] != slot:
            cur["switched"] = True
            eps.append(cur)
            cur = None
        if cur is None and slot != 0:
            _c = _card_cost(p0, p0.cycle[slot - 1]) if slot - 1 < len(p0.cycle) else None
            _un = _unaffordable(p0, el)
            cur = {"slot": slot, "card": str(p0.cycle[slot - 1]) if slot - 1 < len(p0.cycle) else "?",
                   "cost": float(_c) if _c is not None else None,
                   "elixir_open": el, "age_open": age, "t_open": n,
                   "unaffordable_at_open": [c for _s, _cd, c in _un],
                   "chosen_cost": float(_c) if _c is not None else None,
                   "n_unaffordable_open": len(_un),
                   "is_set": bool(ev.get("intent_set") or ev.get("intent_reassert")),
                   "unaff_min": (min([c for _s, _cd, c in _un]) if _un else None),
                   "unaff_max": (max([c for _s, _cd, c in _un]) if _un else None),
                   "n_costs": len(set(c for _s, _cd, c in _un)),
                   "frames": 0, "held": 0, "held_info": 0, "hold_raw": 0, "ready_frames": 0,
                   "age_max": 0,
                   "elixir_min": el, "last_elixir": el,
                   "fired": False, "cancelled": False, "dropped": False, "switched": False}
        if cur is not None:
            cur["frames"] += 1
            # ★ 逐帧 `info["intent"]["hold"]` 与 env 的 `held` 计数**不是同一个量**，两处已实测差：
            #   ① env 的 `held` **不含每段的 set 帧**（`_apply_intent` 在新目标分支提前 return）；
            #   ② step 内 `_apply_intent` 与 `info` 构造之间圣水可能已变化 ⇒ 存在"帧内变可负担"
            #      的帧：env 记 held、本帧后置 `hold` 已是 False。
            # ⇒逐帧计数只作**参考列**（`hold_raw` / `held_info`），**判定列用 env 自己的
            #   `_intent_age`**（`info["intent"]["age"]`，env 每 held 一帧 +1）⇒ `age_max` 即该段
            #   被 env 记入 `intent_stats["held"]` 的帧数，可直接与 env 对账（⑤）。
            if hold:
                cur["hold_raw"] += 1
                if not bool(ev.get("intent_set")):
                    cur["held_info"] += 1
            else:
                cur["ready_frames"] += 1
            cur["age_max"] = max(cur["age_max"], age)
            cur["elixir_min"] = min(cur["elixir_min"], el)
            cur["last_elixir"] = el
            cur["fired"] = cur["fired"] or bool(ev.get("intent_fired"))
            cur["cancelled"] = cur["cancelled"] or bool(ev.get("intent_cancelled"))
            cur["dropped"] = cur["dropped"] or bool(ev.get("intent_dropped"))
        frames.append((slot, hold, el))
        done = bool(term or trunc)
        n += 1
    if cur is not None:
        eps.append(cur)
    for e in eps:
        # 判定列 = env 口径（env 自己的 `_intent_age` 最大值）；`held_info` = 逐帧后置读数（参考）
        e["held"] = int(e["age_max"])
        e["outcome"] = ("fired" if e["fired"] else "cancelled" if e["cancelled"]
                        else "dropped" if e["dropped"] else "switched" if e["switched"]
                        else "open")
        e["gap_open"] = (None if e["cost"] is None else round(e["cost"] - e["elixir_open"], 4))
        e["threshold"] = None if e["cost"] is None else frames_needed(e["cost"])
        e["reached_threshold"] = bool(e["threshold"] is not None and e["held"] >= e["threshold"])
        # ★ 「有真实选择」= 下意图时买不起的槽 >=2 **且** 它们的费用不全相同
        #   （只有一张买不起时，"最便宜"与"最贵"必然是同一张 ⇒ 单独看 cheapest 占比会骗人）
        e["real_choice"] = bool(e["n_unaffordable_open"] >= 2 and e["n_costs"] >= 2)
        if not e["real_choice"] or e["chosen_cost"] is None:
            e["rank"] = None
        elif e["chosen_cost"] <= e["unaff_min"] + 1e-9:
            e["rank"] = "cheapest"
        elif e["chosen_cost"] >= e["unaff_max"] - 1e-9:
            e["rank"] = "dearest"
        else:
            e["rank"] = "mid"
    return {"frames": n, "pending_frames": sum(1 for s, _h, _e in frames if s != 0),
            "held_frames": sum(e["held"] for e in eps),
            "held_raw_frames": sum(1 for _s, h, _e in frames if h),
            "no_pending_frames": sum(1 for s, _h, _e in frames if s == 0),
            "elixir_sum": float(sum(e for _s, _h, e in frames)),
            "slots": frames, "episodes": eps}


def _agg(games, deck):
    """聚合：逐帧归因（按卡）+ 事件归因（按卡）+ 缺口分布 + 「最便宜」判定。"""
    costs = {c: Card(c).elixir for c in deck}
    by_card = {}
    eps = [e for g in games for e in g["episodes"]]
    n_slot = sum(g["pending_frames"] for g in games)
    # ★ held 用 **env 口径**（= `intent_stats["held"]`，不含每段的 set 帧），与判读仪器可对账
    n_held = sum(e["held"] for e in eps)
    n_held_raw = sum(g["held_raw_frames"] for g in games)
    for e in eps:
        k = e["card"]
        b = by_card.setdefault(k, {"card": k, "cost": costs.get(k), "pending_frames": 0,
                                   "held_frames": 0, "episodes": 0, "held_sum": 0,
                                   "held_max": 0, "reached_threshold": 0, "fired": 0,
                                   "ready": 0, "cancelled": 0, "dropped": 0, "open": 0})
        b["pending_frames"] += e["frames"]
        b["held_frames"] += e["held"]
        b["episodes"] += 1
        b["held_sum"] += e["held"]
        b["held_max"] = max(b["held_max"], e["held"])
        b["reached_threshold"] += 1 if e["reached_threshold"] else 0
        b["fired"] += 1 if e["outcome"] == "fired" else 0
        b["ready"] += 1 if e["ready_frames"] > 0 else 0
        b["cancelled"] += 1 if e["outcome"] == "cancelled" else 0
        b["dropped"] += 1 if e["outcome"] == "dropped" else 0
        b["open"] += 1 if e["outcome"] == "open" else 0
    for b in by_card.values():
        b["threshold"] = frames_needed(b["cost"]) if b["cost"] else None
        b["held_share"] = (b["held_frames"] / n_held) if n_held else 0.0
    # 「是不是在为最便宜的卡等」：每次 set 时，目标费用 == 当时买不起的卡里**最低**费用
    _sets = [e for e in eps if e["is_set"] and e["chosen_cost"] is not None]
    _cheapest = [e for e in _sets if e["unaffordable_at_open"]
                 and e["chosen_cost"] <= min(e["unaffordable_at_open"]) + 1e-9]
    _mostexp = [e for e in _sets if e["unaffordable_at_open"]
                and e["chosen_cost"] >= max(e["unaffordable_at_open"]) - 1e-9]
    _real = [e for e in _sets if e["real_choice"]]
    _rcheap = [e for e in _real if e["rank"] == "cheapest"]
    _rdear = [e for e in _real if e["rank"] == "dearest"]
    _rmid = [e for e in _real if e["rank"] == "mid"]
    gaps = {}
    for e in eps:
        if e["gap_open"] is None:
            continue
        lb = _bucket_gap(e["gap_open"])
        gaps[lb] = gaps.get(lb, 0) + 1
    return {"frames_by_card": sorted(by_card.values(),
                                     key=lambda b: -(b["held_frames"] + b["pending_frames"])),
            "episodes": eps, "pending_frames": n_slot, "held_frames": n_held,
            "held_raw_frames": n_held_raw,
            "gap_hist": gaps,
            "n_set_events": len(_sets),
            "n_set_cheapest_unaffordable": len(_cheapest),
            "n_set_most_expensive_unaffordable": len(_mostexp),
            "cheapest_share": (len(_cheapest) / len(_sets)) if _sets else 0.0,
            "n_real_choice": len(_real),
            "n_real_cheapest": len(_rcheap), "n_real_dearest": len(_rdear),
            "n_real_mid": len(_rmid),
            "real_cheapest_share": (len(_rcheap) / len(_real)) if _real else 0.0,
            "real_dearest_share": (len(_rdear) / len(_real)) if _real else 0.0,
            "deck": deck, "deck_costs": costs}


def _run(args):
    deck = _deck_for(args.deck)
    costs = {c: Card(c).elixir for c in deck}
    ge6 = [c for c in deck if costs[c] >= J1_MIN_COST]
    train_deck = list(resolve_deck_set("default")[0])
    env = RLEnv(opponent=None, seed=args.seed, intent_save=True, card_level=args.card_level,
                deck0=list(deck), deck1=list(deck))
    belief = BeliefInference(opp_deck=env.deck1, n_particles=128, seed=args.seed)
    bp = BeliefPlanner()
    belief_dim = int(np.asarray(belief.encode(None, None)).shape[0])

    print("=" * 88)
    print("攒费意图（intent-save）目标归因仪器 —— 攒的是哪一张牌 / 等了多少帧 / 还差多少费")
    print("=" * 88)
    print(f"回费率常量源 : player.regenerate_elixir(base_regen_time={BASE_REGEN_TIME})")
    print(f"每决策帧回费 : {ELIXIR_PER_FRAME:.5f} 圣水/帧（= 1/{BASE_REGEN_TIME} × {DECISION_SECONDS}s）")
    print(f"攒够帧数换算 : " + "  ".join(f"{c}费→{frames_needed(c)}帧"
                                         for c in sorted(set(costs.values()))))
    print(f"自检(判据阈值): 6 费 → {frames_needed(6)} 帧"
          f"{'（= 预注册 J1 的 34，一致）' if frames_needed(6) == 34 else '（⚠️ 与预注册 34 不一致）'}")
    print()
    print(f"本仪器卡组   : --deck {args.deck} = {deck}")
    print(f"             费用 = " + ", ".join(f"{c}:{costs[c]}" for c in deck))
    print(f"训练卡组     : rl.train_solo.resolve_deck_set('default')[0] = {train_deck}")
    print(f"卡组对账     : {'一致（本仪器卡组 == 训练卡组）' if deck == train_deck else '★ 不一致 ★'}"
          f"（--deck solo 才是训练口径）")
    print(f"J1 可满足性  : 本卡组里 >= {J1_MIN_COST:g} 费的卡 = {ge6 or '【无】'}"
          + ("" if ge6 else "  ⇒ ★ J1 的『>=6 费卡』过滤在本卡组上**按构造恒为 0**（空判据）"))
    print()
    print(f"ckpt         : {args.ckpt}")
    print(f"PLAN_DIM={PLAN_DIM}  belief_dim={belief_dim}  hidden_dim={args.hidden_dim or 'ckpt 元数据'}")
    print(f"评估协议     : games={args.games}  max_frames={args.max_frames}  seed={args.seed}  "
          f"deterministic={args.deterministic}（False=采样）")
    print()

    hidden_dim = args.hidden_dim
    if hidden_dim is None and args.ckpt:
        _md = torch.load(args.ckpt, map_location="cpu")
        hidden_dim = (_md.get("hidden_dim") if isinstance(_md, dict) else None) or 128
    pol = load_checkpoint(args.ckpt, hidden_dim=hidden_dim, plan_dim=PLAN_DIM,
                          belief_dim=belief_dim, intent_options=True)
    if not env.intent_save:
        raise RuntimeError("RLEnv 未开启 intent_save ⇒ 无意图读数")

    games = []
    for g in range(args.games):
        rec = _run_game(env, pol, bp, belief, args)
        rec["game"] = g
        games.append(rec)
        print(f"局 {g:>3}: frames={rec['frames']:>4} pending={rec['pending_frames']:>4} "
              f"held={rec['held_frames']:>4} 无pending={rec['no_pending_frames']:>4} "
              f"意图段={len(rec['episodes']):>3}")

    agg = _agg(games, deck)
    total_frames = sum(g["frames"] for g in games)
    print()
    print("-" * 88)
    print(f"总帧数 {total_frames}  pending 帧 {agg['pending_frames']}"
          f"（{agg['pending_frames'] / max(1, total_frames):.1%}）"
          f"  其中 **held 帧（env 口径）** {agg['held_frames']}"
          f"（{agg['held_frames'] / max(1, total_frames):.1%} 全帧 / "
          f"{agg['held_frames'] / max(1, agg['pending_frames']):.1%} pending 帧）")
    print(f"   [口径] held 用 **env 口径**（= `intent_stats['held']`，**不含每段的 set 帧**）；"
          f"含 set 帧的原始计数 = {agg['held_raw_frames']}"
          f"（差 = 意图段数 {len(agg['episodes'])}）")
    print()
    print("① 逐帧归因（pending 帧按目标卡拆开；held = 该帧仍买不起**且不是该段的 set 帧**=承诺期）")
    print(f"   {'卡':<12}{'费用':>5}{'阈值帧':>7}{'pending帧':>10}{'held帧':>8}"
          f"{'占held':>8}{'意图段':>8}{'held均':>8}{'held最大':>9}{'达阈段':>8}")
    for b in agg["frames_by_card"]:
        print(f"   {b['card']:<12}{b['cost']:>5.0f}{str(b['threshold']):>7}"
              f"{b['pending_frames']:>10}{b['held_frames']:>8}{b['held_share']:>8.1%}"
              f"{b['episodes']:>8}"
              f"{(b['held_sum'] / b['episodes']):>8.2f}{b['held_max']:>9}{b['reached_threshold']:>8}")
    print()
    print("② 意图下的那一刻：缺口 gap = 费用 − 圣水（越接近 0 = 越像「快买得起了」而非「攒大件」）")
    print("   gap 分桶: " + "  ".join(f"{lb}={agg['gap_hist'].get(lb, 0)}"
                                     for _lo, _hi, lb in GAP_BUCKETS))
    print(f"③ 「为最便宜的买不起的卡等待」= {agg['n_set_cheapest_unaffordable']}/{agg['n_set_events']}"
          f"（{agg['cheapest_share']:.1%}）"
          f"   对照「为最贵的买不起的卡等待」= {agg['n_set_most_expensive_unaffordable']}"
          f"/{agg['n_set_events']}")
    _rc = agg["n_real_choice"]
    print(f"③★ **有真实选择**（买不起的槽 >=2 且费用不全相同）= {_rc}/{agg['n_set_events']}"
          f"（{_rc / max(1, agg['n_set_events']):.1%}）："
          f"选**最便宜** = {agg['n_real_cheapest']}（{agg['real_cheapest_share']:.1%}）"
          f" / 选**最贵** = {agg['n_real_dearest']}（{agg['real_dearest_share']:.1%}）"
          f" / 中间 = {agg['n_real_mid']}"
          f"  ⇒ {'偏便宜' if agg['real_cheapest_share'] > agg['real_dearest_share'] + 0.05 else '偏贵' if agg['real_dearest_share'] > agg['real_cheapest_share'] + 0.05 else '**基本打平**'}")
    _long = sorted([e for e in agg["episodes"] if e["held"] > 0],
                   key=lambda e: -e["held"])[:10]
    # ★ 与 env 自己的计数器**对账**（这是首版缺的那一步：仪器自己算的和 env 记的必须相等）
    _env_held = int(env.intent_stats.get("held", 0))
    _env_set = int(env.intent_stats.get("set", 0))
    _env_fired = int(env.intent_stats.get("fired", 0))
    reconcile = {"env_held": _env_held, "probe_held": int(agg["held_frames"]),
                 "held_match": bool(_env_held == int(agg["held_frames"])),
                 "env_set": _env_set, "probe_episodes": len(agg["episodes"]),
                 "env_fired": _env_fired,
                 "probe_fired": int(sum(1 for e in agg["episodes"] if e["fired"]))}
    print()
    print(f"⑤ 与 env 计数器对账：held env={_env_held} / 本仪器={agg['held_frames']}"
          f" ⇒ {'一致' if reconcile['held_match'] else '★ 不一致 ★'}"
          f"    set env={_env_set} / 意图段={len(agg['episodes'])}"
          f"    fired env={_env_fired} / 逐段={reconcile['probe_fired']}")
    print()
    print(f"④ held 最长的 10 段 (局, 帧#, 卡, 费用, 阈值, held, 结局):")
    for e in _long:
        print(f"   game={e.get('game', '?'):>2} t={e['t_open']:>4} {e['card']:<10}"
              f" cost={e['cost']}  thr={e['threshold']:>3}  held={e['held']:>3}  {e['outcome']}")
    print("-" * 88)

    for g in games:
        for e in g["episodes"]:
            e["game"] = g["game"]
    out = {
        "instrument": "scripts/probe_intent_target.py",
        "what": "攒费意图目标归因：pending/held 帧按目标卡拆开 + 每次意图的缺口/结局/是否最便宜",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "args": {"ckpt": args.ckpt, "deck": args.deck, "games": args.games,
                 "max_frames": args.max_frames, "seed": args.seed,
                 "deterministic": bool(args.deterministic), "card_level": args.card_level},
        "regen": {"base_regen_time": BASE_REGEN_TIME,
                  "elixir_per_frame": ELIXIR_PER_FRAME,
                  "frames_needed": {str(c): frames_needed(c) for c in sorted(set(costs.values()))},
                  "source": "player.PlayerState.regenerate_elixir 默认参数（唯一常量源）"},
        "deck": {"mode": args.deck, "cards": deck, "costs": costs,
                 "training_deck": train_deck, "matches_training": bool(deck == train_deck),
                 "has_ge_6": ge6, "j1_filter_satisfiable": bool(ge6)},
        "totals": {"frames": total_frames, "pending_frames": agg["pending_frames"],
                   "held_frames": agg["held_frames"],
                   "held_raw_frames": agg["held_raw_frames"],
                   "held_scope": "env 口径（= intent_stats['held']，不含每段 set 帧）",
                   "n_episodes": len(agg["episodes"]),
                   "gap_hist": agg["gap_hist"],
                   "n_set_events": agg["n_set_events"],
                   "n_set_cheapest_unaffordable": agg["n_set_cheapest_unaffordable"],
                   "n_set_most_expensive_unaffordable": agg["n_set_most_expensive_unaffordable"],
                   "cheapest_share": agg["cheapest_share"]},
        "by_card": agg["frames_by_card"],
        "reconcile_env_counters": reconcile,
        "episode_columns": EPISODE_COLUMNS,
        "episode_rows": [[_cell(e, c) for c in EPISODE_COLUMNS] for e in agg["episodes"]],
        "games": [{"game": g["game"], "frames": g["frames"],
                   "pending_frames": g["pending_frames"], "held_frames": g["held_frames"]}
                  for g in games],
        "discipline": [
            "只跑评估，不训练、不写 ckpt",
            "阈值 = 费用 ÷ 每帧回费（唯一常量源 player.regenerate_elixir），非观测标定（【R16】）",
            "gap 用**步后**圣水近似（误差 <= 1 帧回费 = 0.1786）——本仪器声明该口径，不声称精确",
        ],
    }
    if not args.no_json:
        d = os.path.dirname(os.path.abspath(args.json))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"JSON 留证: {args.json}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="攒费意图目标归因仪器（跑评估，不训练）")
    ap.add_argument("--ckpt", required=True, help="策略 ckpt（11-option 架构 = B 臂口径）")
    ap.add_argument("--deck", choices=("solo", "vanilla"), default="solo",
                    help="solo=训练卡组 Xbow 2.9（缺省）；vanilla=DEFAULT_DECK_1（复现旧读数）")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--card-level", type=int, default=None)
    ap.add_argument("--deterministic", action="store_true",
                    help="策略取 argmax（缺省 False = 按分布采样，与 audit_intent_save.py 同口径）")
    ap.add_argument("--hidden-dim", type=int, default=None)
    ap.add_argument("--json", default=DEFAULT_JSON)
    ap.add_argument("--no-json", action="store_true")
    args = ap.parse_args(argv)
    if not os.path.isfile(args.ckpt):
        print(f"[错误] --ckpt 不存在：{args.ckpt}")
        return 2
    return _run(args)


if __name__ == "__main__":
    sys.exit(main())
