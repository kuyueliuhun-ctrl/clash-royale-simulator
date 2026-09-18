# -*- coding: utf-8 -*-
"""「随机出牌 + 随机位置 + 随机概率正弦/衰减」方案的**可行性取证**仪器（只读）。

## 它要回答的问题（用户 2026-09-18 提问）

> 能不能对「下什么牌（含不出牌）」和「下牌位置」各加一层随机，随机的概率本身也随机，
> 概率按正弦变化并随训练步数衰减？

这个问题有个**可以算死**的核心：攒到 Xbow（6 费）需要**连续 34 帧不花钱**（回费实测
0.1786/帧 ⇒ 6/0.1786 = 33.6）。逐帧独立的随机（i.i.d.）要在这 34 帧里的**每一个
"有牌可出"的帧**都恰好抽中"不出"，成功概率是**乘积** ⇒ 随帧数**指数衰减**。

本仪器分两路取证：

### A. 解析上界（`--mode analytic`，只读录像，无引擎）
对每一帧（圣水 E<6）假设"从这里开始一路不出牌直到买得起 Xbow"，则
- 需要的**连续不出牌帧数** `n = ceil((6-E)/REGEN)`；
- 其中**必然存在合法出牌项的帧数** `k`（判据：圣水 ≥ 3 ⇒ 必有可出牌——本卡组任意
  4 张手牌里最便宜那张 ≤3，见 `hand_bound`；手牌在不出牌期间**冻结**）；
- 每一帧"抽中不出"的概率**上界 1/2**（最宽松：只有 1 张牌买得起且 ability 非法
  ⇒ 合法项 = {那张牌, STOP}）。
⇒ **单次尝试成功概率上界** `(1/2)^k`；按不重叠窗口求和 ⇒ **每 100k 决策帧的
「攒到 6 费」事件数上界**。这是【R4】脚本复算，不是手抄。

### B. 引擎蒙卡（`--mode engine`，真环境）
四种行为臂逐帧真的跑环境（对手 = 同一策略的冻结副本，deterministic）：

| 臂 | 行为 |
|---|---|
| `greedy` | 训练后策略 argmax（= 评估口径，基线） |
| `sample` | 训练后策略按自身分布采样（= **现状训练口径**） |
| `uniform` | 每个 decoder 步在**合法项**里**均匀**抽；槽位抽中后在**合法格**里均匀抽 |
| `mixP` | 概率 P 走 `uniform`、否则走 `sample`（= 用户方案的等价实现） |

记：圣水分位 / ≥3 / ≥4 / ≥6 帧占比、Xbow 部署数、**"有牌可出却不出"的连续长度**
分布与其 ≥17 / ≥25 / ≥34 的次数、落点经验熵 vs `log(合法格数)`（= 位置随机还剩多少
新信息）。**位置随机必须只落在 `mask["cells"]` 上**，否则会重新制造非法包（见
`docs/mask_used_slot_offbyone_fix_2026-09-18.md` 的 B 类病理）。

## 纪律
- 只读：不写训练产物、不改奖励/判定/网络；
- 【R3】非判据、【R5】n=1 seed ⇒ 描述性 + 上界，**不得**写成"修好了/确认无效"；
- GBK 陷阱：stdout 必须 `reconfigure(encoding='utf-8')`；`--json` 用仓库相对路径。

用法（必须在 src/clasher_new 下运行）：
    cd src/clasher_new && PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe \
        ../../scripts/probe_explore_randomization.py --mode both \
        --replays runs/et_solo100k/replays \
        --ckpt runs/et_solo100k/solo_main.pt \
        --arms greedy,sample,uniform,mix0.5 --games 8 \
        --json ../../docs/probe_explore_randomization.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import pickle
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src", "clasher_new")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np  # noqa: E402

#: 回费速率（圣水/决策帧）；实测两臂 155,866 帧标定 0.1786（= 0.3571/s）
REGEN = 0.1786
#: 攒费目标：Xbow 费用
XBOW_COST = 6.0
#: 单帧"抽中不出牌"的概率上界（最宽松情形：合法项 = {1 张可出牌, STOP}）
P_PASS_UB = 0.5


def hand_bound():
    """本卡组任意 4 张手牌里最便宜那张的费用上界（⇒「圣水 ≥ 该值必有可出牌」）。"""
    from card_utils import Card
    from rl.train_solo import DEFAULT_SOLO_DECK as D
    costs = {c: float(Card(c).elixir) for c in D}
    import itertools
    worst = 0
    worst_hand = None
    for h in itertools.combinations(D, 4):
        m = min(costs[c] for c in h)
        if m > worst:
            worst, worst_hand = m, h
    return costs, float(worst), list(worst_hand or [])


# ----------------------------------------------------------------------
# A. 解析上界（离线，只读录像）
# ----------------------------------------------------------------------
def analytic(replay_dirs, regen=REGEN, target=XBOW_COST,
             p_list=(-1, 0.5, 1.0 / 3.0, 1.0 / 6.0, 0.07,
                     0.7, 0.8, 0.9, 0.95, 0.99)):
    """每 100k 决策帧的「攒到 target 费」事件数**上界**。

    起窗口径（写死，避免事后挑窗口）：**只在"上一帧刚花过钱"的帧起窗**（= 圣水的局部谷底，
    也就是一次真正的"从零开始攒费"尝试）。从该帧起，圣水只能靠回费单调上升（否则不算
    同一段尝试）⇒ 需要 `need = ceil((target-E0)/regen)` 帧连续不花钱；其中圣水 ≥
    `cheapest_ub` 的帧**必然有合法出牌项**（本卡组任意 4 张手牌最便宜那张 ≤3；不出牌期间
    手牌冻结）⇒ 这些帧必须"恰好抽中不出" ⇒ `k = need - skip` 帧 × 每帧概率 `p_pass`。

    `p_pass` 取四个值（**方案自身的参数**，不是从别处照抄的阈值，【R15】）：
      1/2  = **最宽松上界**（只有 1 张牌买得起且 ability 非法 ⇒ 合法项 = {该牌, STOP}）
      1/3、1/6 = 合法项 3 个 / 6 个（4 张牌全买得起 + ability）时的均匀抽取
      0.07 = 实测"有合法选项时"策略自己的 P(不出牌)（`probe_pass_prob.py`；仅作对照）

    ⚠️ 这是**上界**：`p_pass` 逐帧取最宽松值、且忽略"要攒到 6 还必须先有 Xbow 在手"。
    """
    costs, cheapest_ub, worst_hand = hand_bound()
    games = 0
    frames_total = 0
    per_dir = {}
    for label, d in replay_dirs:
        files = sorted(glob.glob(os.path.join(d, "league_*.pkl")))
        if not files:
            raise SystemExit(f"[缺数据] {label}: {d} 里没有 league_*.pkl")
        n_frames = 0
        elix_hist = []
        attempts = []                       # (E0, k)
        n_spends = 0
        for f in files:
            with open(f, "rb") as fh:
                data = pickle.load(fh)
            for g in (data.get("games") or []):
                games += 1
                fr = g.get("frames") or []
                n = len(fr)
                if n == 0:
                    continue
                # 决策前的圣水：第 0 帧 = 开局 5.0，其后 = 上一帧步末
                post = [float(x.get("elixir0", np.nan)) for x in fr]
                pre = [5.0] + post[:-1]
                n_frames += n
                frames_total += n
                elix_hist.extend(pre)
                spent = [bool(post[i] < pre[i] - 1e-6) for i in range(n)]
                n_spends += int(sum(spent))
                for i, E in enumerate(pre):
                    if E >= target:
                        continue
                    if i > 0 and not spent[i - 1]:
                        continue                                # 只在费谷起窗
                    need = int(math.ceil((target - E) / regen))
                    skip = int(math.ceil(max(0.0, cheapest_ub - E) / regen))
                    k = max(1, need - skip)
                    attempts.append((float(E), int(k)))
        ev = {}
        for p in p_list:
            if isinstance(p, (int, float)) and p < 0:
                continue
            ev[f"p_pass={p:.4f}"] = float(sum(p ** k for _E, k in attempts))
        # 按"必须连续不出牌的帧数 k"分桶：k 小 = 圣水本来就高的普通波动（不构成"攒费"学习信号）；
        # **k ≥ 17** 才是真正的"从低圣水攒到 6"（含（3,6) 全段）。
        buckets = [(1, 4), (5, 9), (10, 16), (17, 10 ** 9)]
        by_bucket = {}
        for lo, hi in buckets:
            sub = [k for _E, k in attempts if lo <= k <= hi]
            name = f"k={lo}" if hi >= 10 ** 9 else f"k={lo}-{hi}"
            by_bucket[name] = {
                "n_attempts": len(sub),
                "attempts_per_100k": 100000.0 * len(sub) / max(1, n_frames),
                "ev_per_100k": {kk: 100000.0 * float(sum(p ** k for k in sub)) / max(1, n_frames)
                                for kk, p in [(f"p_pass={p:.4f}", p) for p in p_list
                                              if not (isinstance(p, (int, float)) and p < 0)]},
            }
        ks = np.asarray([k for _E, k in attempts], dtype=float)
        e0s = np.asarray([E for E, _k in attempts], dtype=float)
        per_dir[label] = {
            "n_frames": n_frames,
            "frames_ge3": int(np.sum(np.asarray(elix_hist) >= 3.0)),
            "frames_ge4": int(np.sum(np.asarray(elix_hist) >= 4.0)),
            "frames_ge6": int(np.sum(np.asarray(elix_hist) >= 6.0)),
            "elixir_median": float(np.median(elix_hist)),
            "elixir_p90": float(np.percentile(elix_hist, 90)),
            "n_attempts": len(attempts),
            "attempts_per_100k": 100000.0 * len(attempts) / max(1, n_frames),
            "k_median": float(np.median(ks)) if ks.size else 0.0,
            "start_elixir_median": float(np.median(e0s)) if e0s.size else 0.0,
            "n_spend_frames": n_spends,
            "ev_per_100k": {kk: 100000.0 * v / max(1, n_frames) for kk, v in ev.items()},
            "by_k_bucket": by_bucket,
        }
    return {
        "regen_per_frame": regen,
        "target_cost": target,
        "deck_costs": {k: float(v) for k, v in costs.items()},
        "cheapest_in_any_hand_upper_bound": cheapest_ub,
        "worst_hand": worst_hand,
        "n_games": games,
        "n_frames": frames_total,
        "per_dir": per_dir,
    }


# ----------------------------------------------------------------------
# B. 引擎蒙卡（真环境，四种行为臂）
# ----------------------------------------------------------------------
def load_policy(ckpt, device, torch):
    from rl.follower import FollowerPolicy
    d = torch.load(ckpt, map_location="cpu")
    sd = d["state_dict"] if isinstance(d, dict) and "state_dict" in d else d
    meta = {k: v for k, v in d.items() if k != "state_dict"} if isinstance(d, dict) else {}
    pol = FollowerPolicy(
        hidden=int(meta.get("hidden_dim", 128)),
        plan_dim=int(meta["plan_dim"]),
        belief_dim=int(meta["belief_dim"]),
        value_bypass=bool(meta.get("value_bypass", False)),
        value_independent=bool(meta.get("value_independent", False)),
    )
    miss, unexp = pol.load_state_dict(sd, strict=False)
    if miss or unexp:
        print(f"  [warn] ckpt 载入 missing={len(miss)} unexpected={len(unexp)}")
    pol.eval()
    pol.to_device(device)
    return pol


def _uniform_bundle(env, get_mask, rng, torch):
    """「每步均匀抽合法项」的策略（等价于把 bundle head 换成均匀分布）。

    全程只用 `mask['slots']` / `mask['ability_legal']` / `mask['cells']`，
    **绝不**自己构造合法集合（否则会重演 B 类非法包病理）。
    """
    from rl.follower import K_MAX, ABILITY_IDX, STOP_IDX
    from rl.action_mask import ActionBundle
    b = ActionBundle()
    cell_choice = None
    for _ in range(K_MAX + 2):
        m = get_mask(b)
        opts = [s for s in range(K_MAX) if bool(m["slots"][s])]
        if m.get("ability_legal"):
            opts.append(ABILITY_IDX)
        opts.append(STOP_IDX)                     # STOP 恒合法
        if m.get("at_cap"):
            opts = [STOP_IDX]
        o = int(opts[int(rng.integers(len(opts)))])
        if o == STOP_IDX:
            break
        if o == ABILITY_IDX:
            b.add_ability()
            continue
        cells = np.asarray(m["cells"][o]).reshape(-1)
        legal = np.flatnonzero(cells > 0)
        if legal.size == 0:                       # 掩码自相矛盾：按 STOP 处理并计数
            b = ActionBundle()
            break
        c = int(legal[int(rng.integers(legal.size))])
        w = cells.shape[0]
        # cells 是二维 (H,W) 时按行优先还原
        m2 = np.asarray(m["cells"][o])
        if m2.ndim == 2:
            y, x = divmod(c, m2.shape[1])
        else:
            y, x = divmod(c, w)
        b.add(o + 1, int(x), int(y))
        cell_choice = (o + 1, int(x), int(y), int(legal.size))
    return b, cell_choice


def _biased_bundle(env, get_mask, rng, p0, torch, q_stop=0.9, alpha=0.0):
    """**非等概率**的随机提案（用户 2026-09-18 第二问）：

        P(STOP) = q_stop                                    （抬"不出牌"）
        剩余 (1-q_stop) 在合法槽位/ability 上按 `卡费**alpha` 加权（抬"大费牌"）

    落点仍在**合法格**里均匀抽（与 `uniform` 臂同口径，便于单变量对比）。
    只为**覆盖率取证**：它对应"外部偏置提案"，**不是**训练写法（见文档 §PPO 面）。
    """
    from card_utils import Card
    from rl.follower import K_MAX, ABILITY_IDX, STOP_IDX
    from rl.action_mask import ActionBundle
    costs = []
    for c in list(getattr(p0, "cycle", []) or [])[:K_MAX]:
        try:
            costs.append(float(Card(c).elixir))
        except Exception:
            costs.append(1.0)
    b = ActionBundle()
    cell_choice = None
    for _ in range(K_MAX + 2):
        m = get_mask(b)
        opts = [s for s in range(K_MAX) if bool(m["slots"][s])]
        if m.get("ability_legal"):
            opts.append(ABILITY_IDX)
        if m.get("at_cap"):
            opts = []
        if not opts or float(rng.random()) < q_stop:
            break                                   # STOP（含"没得选"的被迫停）
        w = np.asarray([(costs[s] if s < len(costs) else 1.0) ** alpha for s in opts],
                       dtype=float)
        if not np.isfinite(w).all() or w.sum() <= 0:
            w = np.ones(len(opts))
        w = w / w.sum()
        o = int(opts[int(rng.choice(len(opts), p=w))])
        if o == ABILITY_IDX:
            b.add_ability()
            continue
        m2 = np.asarray(m["cells"][o])
        flat = m2.reshape(-1)
        legal = np.flatnonzero(flat > 0)
        if legal.size == 0:
            b = ActionBundle()
            break
        c = int(legal[int(rng.integers(legal.size))])
        if m2.ndim == 2:
            y, x = divmod(c, m2.shape[1])
        else:
            y, x = c, 0
        b.add(o + 1, int(x), int(y))
        cell_choice = (o + 1, int(x), int(y), int(legal.size))
    return b, cell_choice


def engine(ckpt, arms, n_games, seed, max_frames, device, torch):
    from rl.env_wrapper import RLEnv
    from rl.belief import BeliefInference
    from rl.belief_planner import BeliefPlanner
    from rl import train_solo as ts
    from rl import follower as fol

    pol = load_policy(ckpt, device, torch)
    opp_pol = load_policy(ckpt, device, torch)
    bp = BeliefPlanner()
    out = {}
    #: STOP logit 的 ckpt 原值（按臂加偏置后再复位；**只加在被测策略 p0 上**，对手保持原样）
    _stop_base = float(pol.slot_head.bias.detach()[5].item())
    for arm in arms:
        arm_label = arm          # 完整臂名（含 `@β`）用于打印/落盘键，避免同名覆盖
        # 稳定种子：Python 的 str hash 逐进程随机化，不能用 hash(arm)
        import zlib
        rng = np.random.default_rng(seed * 7919 + zlib.crc32(arm_label.encode("utf-8")))
        # `名@β`：给 STOP logit 加偏置 β（= **on-policy** 版本的"提高不出牌概率"）
        beta = 0.0
        if "@" in arm:
            arm, _b = arm.split("@", 1)
            beta = float(_b)
        with torch.no_grad():
            pol.slot_head.bias.data[5] = _stop_base + beta
        mode, p_uniform = arm, 0.0
        hold_max, hold_q = 0, 0.0
        q_stop, alpha = 0.0, 0.0
        if arm.startswith("mix"):
            mode, p_uniform = "mix", float(arm[3:])
        elif arm.startswith("bias"):
            # bias_<q_stop>_<alpha>：非等概率提案（抬不出牌 / 抬大费牌）
            _p = arm.split("_")
            mode, q_stop, alpha = "bias", float(_p[1]), float(_p[2])
        elif arm.startswith("hold"):
            # 对照臂：**带时长的随机**（"hold d 帧"，d~U{1..hold_max}），进 hold 的概率
            # hold_q 写死 0.3（只为与 i.i.d. 方案比**量级**，不是建议超参）
            mode, hold_max, hold_q = "hold", int(arm[4:] or 40), 0.3
        rec = {"frames": 0, "games": 0, "elix": [], "n_legal_cards": [],
               "declined_legal": 0, "forced_pass": 0, "plays": 0,
               "choice_frames": 0, "reward_sum": 0.0, "ep_frames": [],
               "cards": {},
               "xbow_plays": 0, "spend_flags": [], "cells": [], "log_ncell": [],
               "streaks_all": []}
        t0 = time.time()
        for g in range(n_games):
            env = RLEnv(opponent=None, seed=seed + 31 * g, card_level=11,
                        deck0=list(ts.DEFAULT_SOLO_DECK), deck1=list(ts.DEFAULT_SOLO_DECK))
            env.opponent = ts.FollowerOpponent(
                opp_pol, env,
                belief=BeliefInference(opp_deck=env.deck1, n_particles=128,
                                       seed=seed + g),
                deterministic=True)
            belief = BeliefInference(opp_deck=env.deck1, n_particles=128,
                                     seed=seed + 1000 + g)
            obs, _ = env.reset(seed=seed + 2000 + g)
            hidden = None
            cur_nospend = 0
            hold_left = 0
            for _step in range(max_frames):
                p0 = env.battle.players[0]
                m0 = env.get_action_mask(None)
                nl = int(np.sum(m0["slots"])) + int(bool(m0.get("ability_legal")))
                E = float(p0.elixir)
                try:
                    plan_vec = bp.plan(env.battle, belief.state(), obs).to_vector()
                except Exception:
                    plan_vec = np.zeros(pol.plan_dim, dtype=np.float32)
                if not isinstance(plan_vec, np.ndarray) or plan_vec.shape[0] != pol.plan_dim:
                    plan_vec = np.zeros(pol.plan_dim, dtype=np.float32)
                btok = belief.encode(obs, None)
                cell_info = None
                do_hold = False
                if mode == "hold":
                    if hold_left > 0:
                        hold_left -= 1
                        do_hold = True
                    elif float(rng.random()) < hold_q:
                        hold_left = int(rng.integers(1, hold_max + 1)) - 1
                        do_hold = True
                do_uniform = (mode == "uniform") or \
                    (mode == "mix" and float(rng.random()) < p_uniform)
                if do_hold:
                    from rl.action_mask import ActionBundle
                    bundle = ActionBundle()          # 空 bundle = 不出牌（合法动作）
                elif mode == "bias":
                    bundle, cell_info = _biased_bundle(
                        env, env.get_action_mask, rng, p0, torch,
                        q_stop=q_stop, alpha=alpha)
                elif do_uniform:
                    bundle, cell_info = _uniform_bundle(env, env.get_action_mask, rng, torch)
                else:
                    det = (mode == "greedy")
                    bundle, _lp, _v, hidden, _m = pol.act(
                        obs, btok, plan_vec.astype(np.float32), env.get_action_mask,
                        hidden=hidden, deterministic=det)
                rec["frames"] += 1
                rec["elix"].append(E)
                rec["n_legal_cards"].append(nl)
                if nl >= 1:
                    rec["choice_frames"] += 1
                deploys = [sa for sa in bundle.sub_actions if sa.kind == "deploy"]
                if not deploys:
                    if nl >= 1:
                        rec["declined_legal"] += 1
                    else:
                        rec["forced_pass"] += 1
                    cur_nospend += 1          # 攒费看的是"没花钱"，被迫不花也算
                else:
                    rec["plays"] += 1
                    rec["streaks_all"].append(cur_nospend)
                    cur_nospend = 0
                if deploys:
                    sd = deploys[0]
                    # 打出牌名统计（含 Xbow）——⚠️ 2026-09-18：本计数**曾经从未累加**
                    # （patch 的锚点没匹配上），导致所有臂 `xbow_plays` 恒 0；已修，
                    # 修前的那批 JSON 里 `xbow_plays: 0` **不可引用**。
                    for _sa in deploys:
                        try:
                            _nm = _sa.card_name(p0)
                        except Exception:
                            _nm = None
                        if _nm:
                            rec["cards"][_nm] = rec["cards"].get(_nm, 0) + 1
                    if any(getattr(a_, "card_name", None) and a_.card_name(p0) == "Xbow"
                           for a_ in deploys):
                        rec["xbow_plays"] += 1
                    m_chosen = env.get_action_mask(None)
                    cells = np.asarray(m_chosen["cells"][sd.slot - 1]).reshape(-1)
                    L = int(np.sum(cells > 0))
                    rec["cells"].append((sd.slot, int(sd.x), int(sd.y)))
                    rec["log_ncell"].append(math.log(max(1, L)))
                obs, _r, term, trunc, info = env.step(bundle)
                rec["reward_sum"] += float(_r)
                belief.update(obs, info.get("opp_played"))
                if term or trunc:
                    rec["streaks_all"].append(cur_nospend)
                    rec["ep_frames"].append(int(_step) + 1)
                    break
            else:
                rec["streaks_all"].append(cur_nospend)
            rec["games"] += 1
        rec["seconds"] = round(time.time() - t0, 1)
        el = np.asarray(rec["elix"], dtype=float)
        st = np.asarray(rec["streaks_all"], dtype=int)
        cells = rec["cells"]
        summ = {
            "arm": arm_label, "stop_logit_beta": beta, "stop_logit_abs": _stop_base + beta,
            "games": rec["games"], "frames": rec["frames"],
            "seconds": rec["seconds"],
            "elixir_mean": float(el.mean()), "elixir_median": float(np.median(el)),
            "elixir_p90": float(np.percentile(el, 90)), "elixir_max": float(el.max()),
            "frac_ge3": float(np.mean(el >= 3.0)),
            "frac_ge4": float(np.mean(el >= 4.0)),
            "frac_ge6": float(np.mean(el >= 6.0)),
            "plays": rec["plays"], "xbow_plays": rec["xbow_plays"],
            "declined_while_legal": rec["declined_legal"],
            "choice_frames": rec["choice_frames"],
            "reward_sum": rec["reward_sum"],
            "cards": dict(sorted(rec["cards"].items(), key=lambda kv: -kv[1])),
            "reward_per_game": rec["reward_sum"] / max(1, rec["games"]),
            "frames_per_game": float(np.mean(rec["ep_frames"])) if rec["ep_frames"] else 0.0,
            "realized_pass_rate": (rec["declined_legal"] / rec["choice_frames"]
                                   if rec["choice_frames"] else 0.0),
            "forced_pass": rec["forced_pass"],
            "max_hold_streak": int(st.max()) if st.size else 0,
            "n_streak_ge17": int(np.sum(st >= 17)),
            "n_streak_ge25": int(np.sum(st >= 25)),
            "n_streak_ge34": int(np.sum(st >= 34)),
            "n_distinct_cells": len({(a, b) for _s, a, b in cells}),
            "cell_entropy_bits": _entropy([(a, b) for _s, a, b in cells]),
            "mean_log_legal_cells": float(np.mean(rec["log_ncell"])) if rec["log_ncell"] else 0.0,
        }
        # Xbow 统计兜底：直接看 elixir 是否真的花掉 6（会计口径，不依赖 cards 字段）
        summ["xbow_note"] = "xbow_plays 依赖 bundle 卡名重建；若为 0 请对照 frac_ge6"
        out[arm_label] = summ
        print(f"  [{arm_label}] {summ['frames']} 帧 / {summ['games']} 局 / {summ['seconds']}s "
              f"| 圣水中位 {summ['elixir_median']:.2f} max {summ['elixir_max']:.2f} "
              f"| ≥6 占比 {100*summ['frac_ge6']:.3f}% | 出牌 {summ['plays']} "
              f"| 最长不花 {summ['max_hold_streak']} 帧")
    return out


def _entropy(cells):
    if not cells:
        return 0.0
    vals, cnt = np.unique(np.asarray(cells), axis=0, return_counts=True)
    p = cnt / cnt.sum()
    return float(-(p * np.log2(p)).sum())


# ----------------------------------------------------------------------
# C. 落点集中度（离线，只读录像）——回答"位置随机"还剩多少新信息
# ----------------------------------------------------------------------
def placement(replay_dirs):
    """实测落点分布 vs **整个网格** 的熵（只读录像）。

    ⚠️ 口径：分母用整张 `GRID_H×GRID_W`（**不是**每张卡的合法格数；后者要引擎掩码，
    由引擎模式的 `mean_log_legal_cells` 给，量级 ≈5.5~5.65 nat）。所以这里的
    "归一化熵"是**下界口径**，只用来读**集中度**（分布是不是挤在少数格子上）。
    """
    from rl.follower import GRID_H, GRID_W
    per_dir = {}
    for label, d in replay_dirs:
        files = sorted(glob.glob(os.path.join(d, "league_*.pkl")))
        if not files:
            raise SystemExit(f"[缺数据] {label}: {d} 里没有 league_*.pkl")
        cells = []
        n_games = 0
        for f in files:
            with open(f, "rb") as fh:
                data = pickle.load(fh)
            for g in (data.get("games") or []):
                n_games += 1
                for fr in (g.get("frames") or []):
                    for s in (fr.get("bundle") or []):
                        if s and s[0] == "deploy":
                            cells.append((int(s[2]), int(s[3])))
        a = np.asarray(cells)
        if a.size == 0:
            per_dir[label] = {"n_deploys": 0}
            continue
        vals, cnt = np.unique(a, axis=0, return_counts=True)
        p = cnt / cnt.sum()
        H = float(-(p * np.log2(p)).sum())
        per_dir[label] = {
            "n_games": n_games, "n_deploys": int(a.shape[0]),
            "grid": [int(GRID_H), int(GRID_W)],
            "n_distinct_cells": int(vals.shape[0]),
            "grid_cells": int(GRID_H * GRID_W),
            "empirical_entropy_bits": H,
            "uniform_entropy_bits": float(math.log2(GRID_H * GRID_W)),
            "effective_cells": float(2.0 ** H),
            "top1_mass": float(p.max()),
            "top10_mass": float(np.sort(p)[-10:].sum()),
            "x_hist": np.bincount(a[:, 0], minlength=GRID_W).tolist(),
            "y_hist": np.bincount(a[:, 1], minlength=GRID_H).tolist(),
        }
    return {"per_dir": per_dir}


def mixture_weight_table(an, pi_stop=0.07, ps=(0.3, 0.5, 1.0)):
    """**混合即策略**写法下「抽中不出牌」这一维的梯度权重，与覆盖率的乘积。

    `pi_train = (1-p)·pi_θ + p·q` ⇒ `∇log pi_train(STOP) = w·∇log pi_θ(STOP)`，
    `w = (1-p)·pi_θ / ((1-p)·pi_θ + p·q)`（`pi_θ` = 策略自己的 P(不出牌)，实测 ≈0.07）。
    ⇒ **每 100k 步的可用梯度信号 ∝ 覆盖率(q) × w(q)**（相对量，只看趋势）。
    对照：**on-policy 的 STOP logit 偏置写法 w ≡ 1**（无任何折扣，见文档 §PPO 面）。
    """
    rows = []
    for lab, v in an["per_dir"].items():
        ev = v["by_k_bucket"].get("k=17", {}).get("ev_per_100k", {})
        for key, e in ev.items():
            q = float(key.split("=")[1])
            row = {"label": lab, "q": q, "events_per_100k_k17": float(e)}
            for p_ in ps:
                den = (1 - p_) * pi_stop + p_ * q
                w = ((1 - p_) * pi_stop / den) if den > 0 else 0.0
                row[f"w_p{p_:.1f}"] = w
                row[f"signal_p{p_:.1f}"] = float(e) * w
            rows.append(row)
        break                       # 一张表足够（两臂同结构）
    return {"pi_stop_measured": pi_stop, "ps": list(ps), "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="both",
                    choices=["analytic", "engine", "placement", "both", "all"])
    ap.add_argument("--replays", action="append", default=[],
                    help="LABEL=DIR（可多次）")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--arms", default="greedy,sample,uniform,mix0.5")
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    rep = []
    for item in a.replays:
        if "=" in item:
            lab, d = item.split("=", 1)
        else:
            lab, d = os.path.basename(item.rstrip("/\\")), item
        rep.append((lab, d))

    result = {"mode": a.mode, "args": vars(a)}

    if a.mode in ("analytic", "both", "all"):
        if not rep:
            raise SystemExit("--mode analytic 需要 --replays LABEL=DIR")
        print("=== A. 解析上界（只读录像）===")
        result["analytic"] = analytic(rep)
        A = result["analytic"]
        print(f"  卡组费用 {A['deck_costs']}")
        print(f"  任意手牌最便宜那张 ≤ {A['cheapest_in_any_hand_upper_bound']} "
              f"(最坏手牌 {A['worst_hand']}) ⇒ 圣水 ≥ 该值必有可出牌")
        for lab, v in A["per_dir"].items():
            print(f"  [{lab}] {v['n_frames']} 帧 | 圣水中位 {v['elixir_median']:.2f} "
                  f"p90 {v['elixir_p90']:.2f} | ≥3 {100*v['frames_ge3']/max(1,v['n_frames']):.1f}% "
                  f"≥4 {100*v['frames_ge4']/max(1,v['n_frames']):.1f}% "
                  f"≥6 {100*v['frames_ge6']/max(1,v['n_frames']):.2f}%")
            print(f"      攒费尝试 {v['n_attempts']} 次（{v['attempts_per_100k']:.0f}/100k 帧、"
                  f"起点圣水中位 {v['start_elixir_median']:.2f}、需连续不出牌帧数中位 {v['k_median']:.0f} 且"
                  f"其中**必有合法出牌项**的帧数同量级）")
            for kk, vv in v["ev_per_100k"].items():
                print(f"      → 每 100k 决策帧攒到 6 费的事件数（{kk}）：**{vv:.3e}**")
            for bname, b in v.get("by_k_bucket", {}).items():
                line = " | ".join(
                    f"{kk.split('=')[1]}→{vv:.3e}" for kk, vv in b["ev_per_100k"].items())
                print(f"        · 分桶 {bname}：尝试 {b['n_attempts']} 次"
                      f"（{b['attempts_per_100k']:.0f}/100k 帧）⇒ {line}")

        try:
            result["mixture_weight"] = mixture_weight_table(result["analytic"])
            print("  -- 偏置强度 vs 可用梯度信号（混合即策略写法；on-policy 偏置写法 w≡1）--")
            for r in result["mixture_weight"]["rows"]:
                print(f"     q={r['q']:.3f}  覆盖率(k≥17)={r['events_per_100k_k17']:.4g}/100k  "
                      f"w(p=0.5)={r['w_p0.5']:.4f}  信号(p=0.5)={r['signal_p0.5']:.4g}  "
                      f"信号(p=1.0)={r['signal_p1.0']:.4g}")
        except Exception as exc:                                        # 缺桶时不静默
            print(f"  [warn] mixture_weight 表未生成：{exc}")

    if a.mode in ("placement", "all"):
        if not rep:
            raise SystemExit("--mode placement 需要 --replays LABEL=DIR")
        print("=== C. 落点集中度（只读录像）===")
        result["placement"] = placement(rep)
        for lab, v in result["placement"]["per_dir"].items():
            if not v.get("n_deploys"):
                print(f"  [{lab}] 0 次部署")
                continue
            print(f"  [{lab}] {v['n_deploys']} 次部署 / {v['n_games']} 局 | "
                  f"用到 {v['n_distinct_cells']}/{v['grid_cells']} 格 | "
                  f"经验熵 {v['empirical_entropy_bits']:.2f} bits "
                  f"(≈{v['effective_cells']:.0f} 个有效格；整网格上限 "
                  f"{v['uniform_entropy_bits']:.2f}) | top1 {v['top1_mass']:.3f} "
                  f"top10 {v['top10_mass']:.3f}")

    if a.mode in ("engine", "both", "all"):
        if not a.ckpt:
            raise SystemExit("--mode engine 需要 --ckpt")
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"=== B. 引擎蒙卡（device={dev}）===")
        arms = [x for x in a.arms.split(",") if x]
        result["engine"] = engine(a.ckpt, arms, a.games, a.seed, a.max_frames, dev, torch)

    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2, default=str)
        print(f"[written] {a.json}")


if __name__ == "__main__":
    main()
