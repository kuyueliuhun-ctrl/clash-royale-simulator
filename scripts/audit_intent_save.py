# -*- coding: utf-8 -*-
"""攒费意图动作（intent-save）预注册判据 **J1** 的判读仪器（跑评估，不训练）。

预注册：`docs/intent_save_prereg_2026-09-19.md` §4（判据）+ §8.5（跑前必须补本仪器）。

## 本仪器是什么 / 不是什么

**描述性仪器，不是判据本身。** 它只做两件事：① 用 `intent_save=True` 的评估局把环境侧
`env.intent_stats`（尤其 `fired_held`）读出来；② 按预注册 §4 写死的阈值把 `fired_held`
数成 J1 的条数并打印 PASS/FAIL。**阈值不在这里定，这里也不新增任何判据。**

- **J1 = 数 `env.intent_stats["fired_held"]` 里 ≥ 34 的条数**。`fired_held` 记的是
  **每次"意图真被兑现"时该意图已保持的决策帧数**（滚动保留最后 256 条）。
- **阈值 34 的来源 = 卡费 / 回费率的解析换算，不是观测标定**（【R16】）：
  最贵目标 Xbow 6 费 ÷ 回费率 0.3571 圣水/s ≈ 17 s，决策帧 0.5 s/帧 ⇒ 6 ÷ 0.3571
  ÷ 0.5 ≈ 33.6 ≈ **34 帧**。⇒ 该阈值在本实验自己的量纲上写死，**不许跨实验照抄**（【R15】）。
- **n = 1/臂 ⇒ 只判二值可达性**（【R5】）：J1 是**计数判据**（"有没有 ≥1 条"），
  不是效应量判据；**不得把胜率当判据**，本仪器因此**不打印、也不落盘任何胜率/奖励读数**。
- **现状基线列不重算**：`scripts/pass_streak_audit.py` 的 **0 条** / 主动不出牌最长段
  **22 帧** 引自预注册 §4 §7（复算命令同文档），本仪器只**原样打印这一列**供并列判读；
  要重新复算请跑那条命令，不要以本脚本的输出顶替。

## 口径（照抄 `rl/evaluate.py` 的评估循环）

`RLEnv(opponent=None, seed=..., intent_save=True)` + `BeliefInference(opp_deck=env.deck1,
n_particles=128, seed=...)` + `BeliefPlanner()`；策略按 **B 臂口径**（`intent_options=True`）
载入，逐帧：

    plan = bp.plan(env.battle, belief.state(), obs)
    tok  = belief.encode(obs, None)
    bundle, lp, val, hidden, masks = pol.act(obs, tok, plan.to_vector(),
                                             lambda b: env.get_action_mask_for(0, b),
                                             hidden=hidden,
                                             deterministic=<--deterministic>)
    obs, r, term, trunc, info = env.step(bundle)
    belief.update(obs, info.get("opp_played"))

`PLAN_DIM` 从 `rl.plan_space` 取（唯一常量源）；`belief_dim` 从
`BeliefInference(env.deck1).encode(None, None)` 的真实长度取。观测多出的
`intent_slot` / `intent_age` 由策略自动消费，**调用方无需特殊处理**。

## 纪律 / 已知边界

- 只跑**评估**，绝不训练（不 import 训练入口、不写 ckpt、不落 replay）。
- `env.intent_stats` **跨局保留**（预注册的实现约定）⇒ 每局读数用**前后差分**，
  `fired_held` 用**逐帧事件**归因到局；合计以 `env.intent_stats` 为准。
- `fired_held` 滚动窗口 256 ⇒ 若 `fired > len(fired_held)`，官方 J1 计数会**低估**；
  本仪器会打印该差值，并同时给出**逐帧事件**独立计数的交叉核对。
- 判读路径**退出码恒 0**（"J1 FAIL" 是**科学结论**，不是运行失败）；只有架构门拒绝
  与参数错误退 2。判读请读 `J1 PASS/FAIL` 行与 JSON。
- 【R10】任何"为什么没有/为什么有"的成因**不在本仪器职责内**，这里只给读数。

用法（两种运行位置都可以，路径全部由 `__file__` 推）：

    # 1) 仓库根目录
    ./.venv/Scripts/python.exe scripts/audit_intent_save.py \
        --ckpt src/clasher_new/runs/et_solo100k/solo_main.pt --games 8

    # 2) src/clasher_new 目录下（与其它 probe_*.py 同姿势）
    cd src/clasher_new && ../../.venv/Scripts/python.exe ../../scripts/audit_intent_save.py \
        --ckpt runs/et_solo100k/solo_main.pt --games 8 --json ../../docs/audit_intent_save.json

`--ckpt` 缺省时须给 `--random-init`（随机初始化策略冒烟）。ckpt 若非 11-option 架构，
默认**报错退出**并给用法提示；确要冒烟请显式加 `--allow-arch-mismatch`（见输出里的告警含义）。

## README 一行用法（供 `scripts/README.md` 索引）

`scripts/audit_intent_save.py --ckpt <ckpt.pt> [--games 8] [--max-frames 400] [--seed 0] [--deterministic] [--allow-arch-mismatch] [--json [路径]]`
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src", "clasher_new")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from rl.io_bootstrap import force_utf8_stdout  # noqa: E402  (T1-1: UTF-8 兜底单一来源)
force_utf8_stdout()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rl.env_wrapper import RLEnv  # noqa: E402
from rl.belief import BeliefInference  # noqa: E402
from rl.belief_planner import BeliefPlanner  # noqa: E402
from rl.follower import FollowerPolicy, load_checkpoint  # noqa: E402
from rl.plan_space import PLAN_DIM  # noqa: E402

#: J1 阈值（预注册 §4）：6 费 ÷ 0.3571 圣水/s ÷ 0.5 s/帧 ≈ 33.6 ⇒ 34。**解析换算，非观测标定。**
J1_THRESHOLD = 34
#: J1 原文限定的目标卡费用下限（「同一张 >=6 费卡」；Xbow = 6）
J1_MIN_COST = 6.0
#: 直方图分桶（预注册 §4/§8.5 要求的口径）
HIST_BUCKETS = ((0, 5, "0-5"), (6, 11, "6-11"), (12, 23, "12-23"),
                (24, 33, "24-33"), (34, 10 ** 9, ">=34"))
#: 每局差分用的标量计数键（`fired_held` 单独按事件归因）
SCALAR_KEYS = ("set", "held", "ready", "fired", "cancelled", "dropped")
#: B 臂口径 ckpt 的 slot_head 出维（`intent_options=True`）
INTENT_SLOT_OPTIONS = 11
#: JSON 缺省落点：repo 根的 docs/（相对 `__file__` 解析 ⇒ 两种运行位置都能落对）
DEFAULT_JSON = os.path.join(_ROOT, "docs", "audit_intent_save.json")
#: 现状基线列（**引自预注册 §4/§7，本脚本不重算**）
BASELINE_NOTE = ("现状基线 = scripts/pass_streak_audit.py 的 0 条、主动不出牌最长段 22 帧"
                 "（引自预注册 docs/intent_save_prereg_2026-09-19.md §4/§7；本脚本不重算）")


def _ckpt_arch(path):
    """读 ckpt 元数据 + `slot_head` 真实出维，判它是不是 11-option（B 臂）架构。"""
    data = torch.load(path, map_location="cpu")
    if isinstance(data, dict) and "state_dict" in data:
        md = data
    else:
        md = {"state_dict": data}
    sd = md["state_dict"]
    declared = bool(md.get("intent_options", False))
    rows = int(sd["slot_head.weight"].shape[0]) if "slot_head.weight" in sd else None
    observed = (rows == INTENT_SLOT_OPTIONS) if rows is not None else None
    # 两份证据（元数据 / 张量形状）任一不符即算不一致，取保守侧
    match = bool(declared) and (observed is not False)
    return md, declared, rows, observed, match


def _print_arch_mismatch_notice(path, declared, rows):
    """显式解释 `load_checkpoint` 那条架构不一致告警的**含义**（【R10】不猜、只陈述机制）。"""
    print()
    print("!" * 78)
    print("[架构不一致 · 已按 --allow-arch-mismatch 放行]")
    print(f"  ckpt: {path}")
    print(f"  元数据 intent_options={declared}；slot_head.weight 出维={rows}"
          f"（B 臂口径应为 {INTENT_SLOT_OPTIONS}）")
    print("  含义：本仪器按 B 臂口径用 intent_options=True 载入 ⇒ slot_head(6→11) /")
    print("        sub_emb(8→13) / enc_fc(输入维 +6) 形状不匹配。这三处没有『尾部追加』")
    print("        兼容分支（enc_fc 的标量加宽发生在 fused **中段**），`load_checkpoint`")
    print("        会把它们**整体重置为随机初始化**（静默）。⇒ 本次运行的策略**不是**")
    print("        该 ckpt 训练出的策略：**只能**用来验证脚本能跑通、字段能读（冒烟），")
    print("        **不得**据此读任何行为/疗效结论，也不得续训（要求 --fresh）。")
    print("!" * 78)
    print()


def _snapshot(stats):
    return {k: int(stats.get(k, 0)) for k in SCALAR_KEYS}


def _hist(counts):
    out = {}
    for lo, hi, label in HIST_BUCKETS:
        out[label] = int(sum(1 for c in counts if lo <= int(c) <= hi))
    return out


def _run_game(env, pol, bp, belief, args):
    """跑一局评估，返回本局读数（标量计数用 env 累计值前后差分；fire 事件逐帧归因）。"""
    before = _snapshot(env.intent_stats)
    obs, _ = env.reset()
    belief.reset(env.deck1)
    hidden = None
    done = False
    steps = 0
    fired_events = []      # 本局每次「意图真被兑现」时已保持的帧数
    while not done and steps < args.max_frames:
        plan = bp.plan(env.battle, belief.state(), obs)
        tok = belief.encode(obs, None)
        bundle, _lp, _val, hidden, _masks = pol.act(
            obs, tok, plan.to_vector(),
            lambda b: env.get_action_mask_for(0, b),
            hidden=hidden, deterministic=args.deterministic)
        obs, _r, term, trunc, info = env.step(bundle)
        belief.update(obs, info.get("opp_played"))
        ev = (info.get("intent") or {}).get("event") or {}
        if ev.get("intent_fired"):
            # 判据 J1 原文 = 「同一张 **>=6 费**卡被 pending 覆盖 >=34 帧后真的打出」
            # => 必须带上目标卡与费用才能按原文过滤（否则一张 3 费卡被抱 34 帧会被误算 PASS）。
            fired_events.append({"held": int(ev.get("intent_fired_held", 0)),
                                 "cost": ev.get("intent_fired_cost"),
                                 "card": ev.get("intent_fired_card")})
        done = bool(term or trunc)
        steps += 1
    after = _snapshot(env.intent_stats)
    return {"frames": steps, "counts": {k: after[k] - before[k] for k in SCALAR_KEYS},
            "fired_held": [int(e["held"]) for e in fired_events],
            "fired_events": fired_events}


def _run(args):
    # ---- 架构门：默认要求 ckpt 真的是 11-option（B 臂）架构 ----
    arch = None
    if args.ckpt:
        if not os.path.isfile(args.ckpt):
            print(f"[错误] --ckpt 不存在：{args.ckpt}")
            return 2
        md, declared, rows, observed, match = _ckpt_arch(args.ckpt)
        arch = {"path": args.ckpt, "declared_intent_options": declared,
                "slot_head_rows": rows, "observed_11_option": observed,
                "match_11_option": match, "hidden_dim": md.get("hidden_dim"),
                "plan_dim_meta": md.get("plan_dim"), "belief_dim_meta": md.get("belief_dim"),
                "allow_arch_mismatch": bool(args.allow_arch_mismatch)}
        if not match and not args.allow_arch_mismatch:
            print(f"[错误] ckpt 不是 11-option 架构：{args.ckpt}")
            print(f"       元数据 intent_options={declared}；slot_head.weight 出维={rows}"
                  f"（应为 {INTENT_SLOT_OPTIONS}）。")
            print("       按 B 臂口径用 intent_options=True 载入会把 slot_head/sub_emb/enc_fc")
            print("       整体重置为随机初始化 ⇒ 读数无意义，故默认拒绝。")
            print("       用法提示：确要只冒烟（验证脚本能跑通/字段能读）请显式加")
            print("                 --allow-arch-mismatch")
            print("       要跑真正的 J1 判读：用 intent-save 训练出的 11-option ckpt，")
            print("                 或先跑 `--fresh` 的 B 臂训练（预注册 §8.5，尚未启动）。")
            return 2
        if not match:
            _print_arch_mismatch_notice(args.ckpt, declared, rows)

    hidden_dim = args.hidden_dim
    if hidden_dim is None and arch is not None:
        hidden_dim = arch.get("hidden_dim")

    # ---- 环境 / 信念 / 规划器（口径照抄 rl/evaluate.py）----
    env = RLEnv(opponent=None, seed=args.seed, intent_save=True)
    belief = BeliefInference(opp_deck=env.deck1, n_particles=128, seed=args.seed)
    bp = BeliefPlanner()
    belief_dim = int(np.asarray(belief.encode(None, None)).shape[0])

    if args.ckpt:
        pol = load_checkpoint(args.ckpt, hidden_dim=hidden_dim, plan_dim=PLAN_DIM,
                              belief_dim=belief_dim, intent_options=True)
    else:
        pol = FollowerPolicy(hidden=int(hidden_dim or 128), plan_dim=PLAN_DIM,
                             belief_dim=belief_dim, intent_options=True)
        pol.eval()
    if not env.intent_save:
        raise RuntimeError("RLEnv 未开启 intent_save ⇒ intent_stats 恒空，J1 不可判")

    print("=" * 78)
    print("J1 判读仪器：攒费意图（intent-save）可达性计数 —— 描述性仪器，不是判据本身")
    print("=" * 78)
    print(f"ckpt         : {args.ckpt or '<无：--random-init 随机初始化策略（冒烟用）>'}")
    if arch is not None:
        print(f"架构         : declared intent_options={arch['declared_intent_options']} / "
              f"slot_head 出维={arch['slot_head_rows']} ⇒ "
              f"{'11-option（B 臂口径，匹配）' if arch['match_11_option'] else '不一致（已放行，仅冒烟）'}")
    print(f"PLAN_DIM     : {PLAN_DIM}   belief_dim: {belief_dim}   hidden_dim: {hidden_dim}")
    print(f"评估协议     : games={args.games}  max_frames={args.max_frames}  seed={args.seed}  "
          f"deterministic={args.deterministic}（False=采样）")
    print(f"env          : intent_save=True（B 臂口径；opponent=None=随机对手，只跑评估不训练）")
    print(f"J1 阈值      : >={J1_THRESHOLD} 保持帧"
          f"（= 6 费 ÷ 0.3571 圣水/s ÷ 0.5 s/帧 ≈ 33.6；解析换算，非观测标定 【R16】）")
    print()

    # ---- 跑局 ----
    games = []
    for g in range(args.games):
        rec = _run_game(env, pol, bp, belief, args)
        rec["game"] = g
        games.append(rec)
        c = rec["counts"]
        print(f"局 {g:>3}: frames={rec['frames']:>4}  "
              f"set={c['set']:>3} held={c['held']:>4} ready={c['ready']:>3} "
              f"fired={c['fired']:>3} cancelled={c['cancelled']:>3} dropped={c['dropped']:>3}  "
              f"fired_held={rec['fired_held']}")

    # ---- 合计（env.intent_stats 跨局累计 = 官方口径）----
    stats = env.intent_stats
    total = {k: int(stats.get(k, 0)) for k in SCALAR_KEYS}
    env_list = [int(x) for x in stats.get("fired_held", [])]
    event_list = [int(x) for rec in games for x in rec["fired_held"]]
    ev_events = [e for rec in games for e in rec.get("fired_events", [])]
    _all_ge = [e for e in ev_events if e["held"] >= J1_THRESHOLD]
    # 【J1 原文口径】逐帧事件（本次运行未截断）+ 目标卡费用 >= J1_MIN_COST
    ev_j1 = [e for e in _all_ge if float(e.get("cost") or 0.0) >= J1_MIN_COST]
    j1_count = int(sum(1 for x in env_list if x >= J1_THRESHOLD))
    j1_count_events = len(ev_j1)
    _ev_j1_txt = str([(e.get("card"), e.get("cost"), e["held"]) for e in ev_j1])
    # J1 的语义是「run 内是否出现过 >=1 条 >=34 帧意图后真打出」= **存在性**。
    # `env.intent_stats["fired_held"]` 滚动只留最后 256 条 ⇒ 长跑里早期那条会被滚掉，
    # 单看列表会**假 FAIL**。故判定取两者之并（逐帧事件 = 本次运行的未截断全史）。
    # 判定依据 = 逐帧事件（本次运行未截断）+ J1 原文的费用过滤；列表口径仅作参考
    j1_authoritative = j1_count_events
    hist = _hist(env_list)
    rolling_truncated = max(0, total["fired"] - len(env_list))

    print()
    print("-" * 78)
    print("合计（env.intent_stats 跨局累计）")
    print(f"  set={total['set']}  held={total['held']}  ready={total['ready']}  "
          f"fired={total['fired']}  cancelled={total['cancelled']}  dropped={total['dropped']}")
    print(f"  fired_held_max={int(stats.get('fired_held_max', 0))}")
    print(f"  fired_held 全列表（n={len(env_list)}）: {env_list}")
    print("  直方图: " + "  ".join(f"{label}={hist[label]}"
                                   for _lo, _hi, label in HIST_BUCKETS))
    print(f"  （滚动窗口 256：fired={total['fired']}，列表 {len(env_list)} 条"
          f"{'，已截断 ' + str(rolling_truncated) + ' 条 ⇒ 官方计数可能低估' if rolling_truncated else '，无截断'}）")
    print(f"  交叉核对（逐帧事件独立归因，未截断）: fired_held={event_list} ⇒ "
          f"count(>={J1_THRESHOLD})={j1_count_events}"
          f"{'（与列表口径一致）' if j1_count_events == j1_count else '（⚠️ 与列表口径不一致：列表被滚动窗口截断，见上）'}")
    print()
    print(f"J1 = count(逐帧事件: held>={J1_THRESHOLD} 且 目标卡费用>={J1_MIN_COST:g}) "
          f"= {j1_count_events}   [未截断全史 + J1 原文口径]")
    print(f"   达标事件明细 (card, cost, held) = {_ev_j1_txt}")
    _same = "（与原文口径相同）" if len(_all_ge) == j1_count_events else "（差异 = 低价卡被抱久，不算 J1）"
    print(f"   [参考] 仅 held>={J1_THRESHOLD}（不过滤费用）= {len(_all_ge)} 条 {_same}")
    print(f"   [参考] env 列表口径 = {j1_count}"
          f"（滚动 256 条 ⇒ 长跑会被滚掉，**不作判定依据**）")
    print(f"J1 基线列: A 臂（意图关 intent_save=False）按构造恒 0 —— 不开意图则不产生 "
          f"intent_stats['fired_held']，本仪器对 A 臂的实测 = None（未跑 A 臂）")
    print(f"           {BASELINE_NOTE}")
    print(f"J1 {'PASS' if j1_authoritative >= 1 else 'FAIL'}"
          f"（判据：>=1 即 PASS；判定值 = {j1_authoritative}（逐帧事件 + 费用过滤）；"
          f"n=1/臂 ⇒ 只判二值可达性 【R5】，不看胜率）")
    print("-" * 78)

    result = {
        "instrument": "scripts/audit_intent_save.py",
        "what": "J1（攒费意图 intent-save）二值可达性计数仪器：描述性仪器，不是判据本身",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "args": {"ckpt": args.ckpt, "random_init": bool(args.random_init),
                 "games": args.games, "max_frames": args.max_frames, "seed": args.seed,
                 "deterministic": args.deterministic,
                 "allow_arch_mismatch": bool(args.allow_arch_mismatch)},
        "arch": arch,
        "plan_dim": PLAN_DIM, "belief_dim": belief_dim, "hidden_dim": hidden_dim,
        "env": {"intent_save": bool(env.intent_save), "opponent": "random(None)"},
        "j1_threshold": J1_THRESHOLD,
        "j1_threshold_derivation": "Xbow 6 费 ÷ 0.3571 圣水/s ≈ 17 s ÷ 0.5 s/帧 ≈ 34 帧"
                                   "（解析换算，非观测标定 【R16】）",
        "games": games,
        "total_counts": total,
        "fired_held": env_list,
        "fired_held_rolling_cap": 256,
        "fired_held_rolling_truncated": rolling_truncated,
        "fired_held_hist": hist,
        "fired_held_event_log": event_list,
        "j1_count": j1_count,
        "j1_count_event_log": j1_count_events,
        "j1_count_authoritative": j1_authoritative,
        "j1_pass": bool(j1_authoritative >= 1),
        "baseline": {
            "arm_a": "intent_save=False：按构造恒 0（不产生 intent_stats）",
            "note": BASELINE_NOTE,
            "source": "docs/intent_save_prereg_2026-09-19.md §4/§7（pass_streak_audit.py 复算口径）",
            "recomputed_here": False,
        },
        "discipline": [
            "描述性仪器，不是判据本身；阈值 34 解析换算而来，非观测标定（【R16】）",
            "n=1/臂 ⇒ 只判二值可达性（【R5】）；不得把胜率当判据（本仪器不产胜率）",
            "只跑评估，不训练；不 import 训练入口、不写 ckpt",
            "现状基线列引自预注册，本仪器不重算",
        ],
    }
    if not args.no_json:
        out = args.json
        d = os.path.dirname(os.path.abspath(out))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"JSON 留证: {out}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="攒费意图动作（intent-save）预注册判据 J1 的判读仪器（跑评估，不训练）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="例：./.venv/Scripts/python.exe scripts/audit_intent_save.py "
               "--ckpt src/clasher_new/runs/et_solo100k/solo_main.pt "
               "--games 1 --max-frames 60 --allow-arch-mismatch")
    ap.add_argument("--ckpt", default=None, help="策略 ckpt（*.pt；B 臂口径 = 11-option 架构）")
    ap.add_argument("--random-init", action="store_true",
                    help="不载 ckpt，用随机初始化的 11-option 策略冒烟（--ckpt 缺省时必需）")
    ap.add_argument("--allow-arch-mismatch", action="store_true",
                    help="ckpt 非 11-option 架构时放行（只用于冒烟：enc_fc 会被整体重置为随机）")
    ap.add_argument("--games", type=int, default=8, help="评估局数（默认 8）")
    ap.add_argument("--max-frames", type=int, default=400, help="每局最大决策帧数（默认 400）")
    ap.add_argument("--seed", type=int, default=0, help="随机种子（默认 0）")
    ap.add_argument("--deterministic", action="store_true",
                    help="策略取 argmax（默认 False = 按分布采样）")
    ap.add_argument("--hidden-dim", type=int, default=None,
                    help="覆盖 hidden 维（缺省读 ckpt 元数据；随机会话默认 128）")
    ap.add_argument("--json", nargs="?", const=DEFAULT_JSON, default=DEFAULT_JSON,
                    help=f"JSON 留证路径（缺省 {os.path.relpath(DEFAULT_JSON, _ROOT)}）")
    ap.add_argument("--no-json", action="store_true", help="不写 JSON")
    args = ap.parse_args(argv)
    if not args.ckpt and not args.random_init:
        ap.error("必须给 --ckpt（11-option ckpt），或显式 --random-init 用随机策略冒烟")
    if args.ckpt and args.random_init:
        ap.error("--ckpt 与 --random-init 互斥，只能选一个")
    return _run(args)


if __name__ == "__main__":
    sys.exit(main())
