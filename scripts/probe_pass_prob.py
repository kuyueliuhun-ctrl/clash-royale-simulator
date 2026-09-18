# -*- coding: utf-8 -*-
"""「不出牌」（空 bundle / noop）概率实测（只读仪器）。

## 口径（先写死，避免事后解释）

策略是自回归 bundle head：第一个 decoder 步在 `NUM_SLOT_OPTIONS = K_MAX+2` 个选项里选
（0..3 = 出牌槽位、4 = ABILITY、5 = STOP）。**选 STOP 就立刻 break ⇒ bundle 为空**
（`ActionBundle.noop()`）= 本决策步一个子动作都不提交 = 「不出牌」。所以

    P(不出牌 | 状态 s) = P(第一个 decoder 步选 STOP | s)

概率在 **掩码之后** 取（与生产 `FollowerPolicy.act` 逐字同口径：`masked_fill(mask==0,-1e9)`
之后 softmax），并且 **含 plan 软偏置**（生产默认开；`PLAN_HOLD_BIAS=2.5` 会推高 STOP，
故同时报 `--biases off` 的对照，把"网络自己学的"与"手写规划偏置推的"分开）。

## ★ 三个必须分开报的量（本仪器存在的主要理由）

实测（`--games 1 --max-frames 40` 调试）发现：**绝大多数决策帧的掩码里一个合法项都没有**
（圣水低于手牌最便宜那张的费）⇒ 那些帧 P(STOP)=1 是**被迫**的，不是"选择"。混在一起报一个
均值（~0.92）会把"被迫"读成"模型不愿意出牌"。故本仪器分层报：

  - `全程`（所有帧，含被迫）—— 只是行为频率，**不是**选择概率；
  - `有合法选项时`（`final_legal ≥ 1`）—— **这才是"选择不出牌"的概率**；
  - `被迫帧`再拆两类：**圣水不够**（ affordability 掩码就为空）vs **不裸下门禁**
    （`solo_commit_blocked` 把本来买得起的槽整槽禁掉；用 `slot_mask` 裸算对账）。

## 三个锚点（不可互相顶替）

  ① `开局帧`（每局第一个决策帧：双方 5 圣水、空场）—— 与对手无关、可复现；
  ② `全程`（整局逐帧）—— 依赖对手与局面分布（自对弈镜像，见下）；
  ③ `网络初始`（`--fresh-init`）—— 回答"初始概率"的另一读法：**未训练的随机初始化**。

## 纪律

- 只读：不写训练产物、不改奖励、不改判定逻辑；
- 【R5】n=1 seed ⇒ **描述性读数**，不是判据、【R3】不得并入任何预注册判据集；
- 全部输出带样本量（局/帧），缺数据大声失败（不静默出 0）。

用法（必须在 src/clasher_new 下运行）：
    cd src/clasher_new && PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe \
        ../../scripts/probe_pass_prob.py --ckpt A_et=runs/et_solo100k/solo_main.pt \
        --ckpt B_ctrl=runs/et_ctrl100k/solo_main.pt --fresh-init --games 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src", "clasher_new")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from rl.io_bootstrap import force_utf8_stdout  # noqa: E402  (T1-1: UTF-8 兜底单一来源)
force_utf8_stdout()

import numpy as np  # noqa: E402


def _q(a, p):
    a = np.asarray(a, dtype=float)
    return float(np.percentile(a, p)) if len(a) else float("nan")


def _f(a):
    a = np.asarray(a, dtype=float)
    return float(a.mean()) if len(a) else float("nan")


def load_meta(path):
    import torch
    d = torch.load(path, map_location="cpu")
    if isinstance(d, dict) and "state_dict" in d:
        return d["state_dict"], {k: v for k, v in d.items() if k != "state_dict"}
    return d, {}


def build_policy(state_dict, meta, device):
    from rl.follower import FollowerPolicy
    pol = FollowerPolicy(
        hidden=int(meta.get("hidden_dim", 128)),
        plan_dim=int(meta["plan_dim"]),
        belief_dim=int(meta["belief_dim"]),
        value_bypass=bool(meta.get("value_bypass", False)),
        value_independent=bool(meta.get("value_independent", False)),
    )
    if state_dict is not None:
        miss, unexp = pol.load_state_dict(state_dict, strict=False)
        if miss or unexp:
            print(f"  [warn] load state_dict: missing={len(miss)} unexpected={len(unexp)}")
    pol.eval()
    pol.to_device(device)
    return pol


def first_step_probs(pol, obs, btok, plan_vec, mask, torch):
    """第一个 decoder 步的槽位分布（含/不含 plan 软偏置），逐字复刻生产 act() 口径。"""
    import torch.nn.functional as F
    with torch.no_grad():
        _fused, enc = pol._encode_parts(obs, btok, plan_vec)
        h = pol.gru_cell(enc, torch.zeros(1, pol.hidden_dim, device=pol.device))
        sm = pol._slot_mask_tensor(mask)
        out = {}
        for tag, biases in (("on", True), ("off", False)):
            pol.plan_biases_enabled = biases
            sb, _cb = pol._plan_biases(plan_vec)
            lg = pol.slot_head(h) + sb
            lg = lg.masked_fill(sm == 0, -1e9)
            out[tag] = F.softmax(lg, dim=-1).squeeze(0).cpu().numpy()
        pol.plan_biases_enabled = True
    return out


def run_model(label, pol, env, n_games, seed, max_frames, device, torch, verbose=True):
    from rl.belief import BeliefInference
    from rl.belief_planner import BeliefPlanner
    from rl.follower import STOP_IDX
    from rl.action_mask import slot_mask
    from rl import train_solo as ts

    opp_pol = build_policy(None, {"plan_dim": pol.plan_dim, "belief_dim": pol.belief_dim,
                                  "hidden_dim": pol.hidden_dim,
                                  "value_bypass": pol.value_bypass,
                                  "value_independent": pol.value_independent}, device)
    opp_pol.load_state_dict(pol.state_dict())
    opp_pol.eval()

    bp = BeliefPlanner()
    rec = {
        "label": label,
        "open_stop": [], "open_stop_offbias": [], "open_legal": [], "open_elixir": [],
        "open_argmax_stop": 0,
        "stop_all": [], "stop_offbias_all": [],
        "stop_choice": [],                 # final_legal >= 1 的帧
        "stop_choice_offbias": [],
        "elixir_all": [],
        "n_frames": 0, "n_choice_frames": 0,
        "n_no_legal": 0, "n_no_afford": 0, "n_gate_blocked": 0,
        "empty_bundle": 0, "no_deploy": 0,
        "games_done": 0, "open_legal_hist": {},
        "by_legal": {},
    }
    t0 = time.time()
    for g in range(n_games):
        opp_side = ts.FollowerOpponent(
            opp_pol, env,
            belief=BeliefInference(opp_deck=env.deck1, n_particles=128, seed=seed + g),
            deterministic=True)
        env.opponent = opp_side
        belief = BeliefInference(opp_deck=env.deck1, n_particles=128, seed=seed + 1000 + g)
        obs, _ = env.reset(seed=seed + 2000 + g)
        hidden = None
        for step in range(max_frames):
            try:
                plan_vec = bp.plan(env.battle, belief.state(), obs).to_vector()
            except Exception:
                plan_vec = np.zeros(pol.plan_dim, dtype=np.float32)
            if not isinstance(plan_vec, np.ndarray) or plan_vec.shape[0] != pol.plan_dim:
                plan_vec = np.zeros(pol.plan_dim, dtype=np.float32)
            btok = belief.encode(obs, None)
            mask0 = env.get_action_mask(None)
            p0 = env.battle.players[0]
            raw_legal = int(np.sum(slot_mask(p0, elixir_override=p0.elixir)))
            final_legal = int(np.sum(mask0["slots"])) + int(bool(mask0.get("ability_legal")))
            probs = first_step_probs(pol, obs, btok, plan_vec, mask0, torch)
            ps_on = float(probs["on"][STOP_IDX])
            ps_off = float(probs["off"][STOP_IDX])

            rec["n_frames"] += 1
            rec["stop_all"].append(ps_on)
            rec["stop_offbias_all"].append(ps_off)
            rec["elixir_all"].append(float(p0.elixir))
            if final_legal == 0:
                rec["n_no_legal"] += 1
                if raw_legal == 0:
                    rec["n_no_afford"] += 1
                else:
                    rec["n_gate_blocked"] += 1
            else:
                rec["n_choice_frames"] += 1
                rec["stop_choice"].append(ps_on)
                rec["stop_choice_offbias"].append(ps_off)
                k = str(final_legal)
                rec["by_legal"].setdefault(k, []).append([ps_on, ps_off])
            if step == 0:
                rec["open_stop"].append(ps_on)
                rec["open_stop_offbias"].append(ps_off)
                rec["open_legal"].append(final_legal)
                rec["open_elixir"].append(float(p0.elixir))
                rec["open_legal_hist"][str(final_legal)] = \
                    rec["open_legal_hist"].get(str(final_legal), 0) + 1
                rec["open_argmax_stop"] += int(int(np.argmax(probs["on"])) == STOP_IDX)

            bundle, _lp, _v, hidden, _m = pol.act(
                obs, btok, plan_vec.astype(np.float32), env.get_action_mask,
                hidden=hidden, deterministic=False)
            if bundle.size == 0:
                rec["empty_bundle"] += 1
            if not any(sa.kind == "deploy" for sa in bundle.sub_actions):
                rec["no_deploy"] += 1
            obs2, _r, term, trunc, info = env.step(bundle)
            belief.update(obs2, info.get("opp_played"))
            obs = obs2
            if term or trunc:
                break
        rec["games_done"] += 1
        if verbose:
            print(f"  [game {g+1}/{n_games}] frames={rec['n_frames']} "
                  f"t={time.time()-t0:.0f}s")
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def summarize(rec):
    o, oo = np.array(rec["open_stop"]), np.array(rec["open_stop_offbias"])
    a = np.array(rec["stop_all"])
    ch = np.array(rec["stop_choice"])
    n = rec["n_frames"]
    L = []
    L.append(f"### {rec['label']}")
    L.append(f"- 局数 {rec['games_done']} / 决策帧 {n} / 用时 {rec['seconds']}s")
    L.append(f"- **① 开局帧 P(不出牌) = {_f(o):.3f}** "
             f"(中位 {np.median(o):.3f}, 极差 {o.min():.3f}..{o.max():.3f}, n={len(o)})"
             f" | 开局 argmax=STOP 的局数 {rec['open_argmax_stop']}/{len(o)}")
    L.append(f"    开局帧去掉 plan 软偏置 = {_f(oo):.3f} (中位 {np.median(oo):.3f})"
             f" | 开局合法选项数分布 {rec['open_legal_hist']}"
             f" | 开局圣水 {_f(rec['open_elixir']):.2f}")
    if len(ch):
        L.append(f"- **② 有合法选项时（选择概率）P(不出牌) = {_f(ch):.3f}** "
                 f"(中位 {np.median(ch):.3f}, p90 {_q(ch,90):.3f}, "
                 f"n={len(ch)}/{n} 帧 = {100.0*len(ch)/max(1,n):.1f}%)"
                 f" | 去 plan 软偏置 = {_f(rec['stop_choice_offbias']):.3f}")
    else:
        L.append("- **② 有合法选项时：0 帧** ⇒ 本读数不可用（大声失败）")
    L.append(f"- ③ 全程 P(不出牌) = {_f(a):.3f}（含被迫帧；阈值口径混合，仅作行为频率）"
             f" | 去偏置 {_f(rec['stop_offbias_all']):.3f}")
    bl = rec.get("by_legal") or {}
    if bl:
        parts = []
        for k in sorted(bl, key=lambda s: int(s)):
            arr = np.asarray(bl[k], dtype=float)
            parts.append(f"合法{k}项 → P(不出牌)={arr[:,0].mean():.3f}"
                         f"(中位 {np.median(arr[:,0]):.3f}, 去偏置 {arr[:,1].mean():.3f}, n={len(arr)})")
        L.append("- **按合法选项数分层**（②的分解）：" + " | ".join(parts))
    L.append(f"- 被迫帧 {rec['n_no_legal']}/{n} ({100.0*rec['n_no_legal']/max(1,n):.1f}%): "
             f"其中**圣水不够** {rec['n_no_afford']}、**不裸下门禁禁掉** {rec['n_gate_blocked']}"
             f" | 圣水 中位 {np.median(rec['elixir_all']):.2f} 均值 {_f(rec['elixir_all']):.2f}")
    L.append(f"- 实测：空 bundle 率 {100.0*rec['empty_bundle']/max(1,n):.1f}% "
             f"({rec['empty_bundle']}/{n}) | 无部署率 "
             f"{100.0*rec['no_deploy']/max(1,n):.1f}% ({rec['no_deploy']}/{n})")
    return "\n".join(L)


def run_open_only(label, pol, env, n_open, seed0, device, torch):
    """只测**开局帧**（每局第一个决策帧）：只 reset + 一次前向，不跑完整对局 ⇒ n 可以做大。

    开局帧 = 双方 5 圣水、空场、无单位 ⇒ **与对手无关**，可复现；只有首手牌（洗牌）随机。
    """
    from rl.belief import BeliefInference
    from rl.belief_planner import BeliefPlanner
    from rl.follower import STOP_IDX
    from rl.action_mask import slot_mask
    from rl import train_solo as ts
    from card_utils import Card

    bp = BeliefPlanner()
    rec = {"label": label, "n_open": 0, "stop": [], "stop_offbias": [], "legal": [],
           "elixir": [], "argmax_stop": 0, "hand_costs": [], "by_legal": {}}
    t0 = time.time()
    for g in range(n_open):
        env.opponent = None
        belief = BeliefInference(opp_deck=list(ts.DEFAULT_SOLO_DECK), n_particles=128,
                                 seed=seed0 + 5000 + g)
        obs, _ = env.reset(seed=seed0 + 9000 + g)
        p0 = env.battle.players[0]
        try:
            plan_vec = bp.plan(env.battle, belief.state(), obs).to_vector()
        except Exception:
            plan_vec = np.zeros(pol.plan_dim, dtype=np.float32)
        if not isinstance(plan_vec, np.ndarray) or plan_vec.shape[0] != pol.plan_dim:
            plan_vec = np.zeros(pol.plan_dim, dtype=np.float32)
        btok = belief.encode(obs, None)
        mask0 = env.get_action_mask(None)
        raw_legal = int(np.sum(slot_mask(p0, elixir_override=p0.elixir)))
        final_legal = int(np.sum(mask0["slots"])) + int(bool(mask0.get("ability_legal")))
        probs = first_step_probs(pol, obs, btok, plan_vec, mask0, torch)
        ps_on = float(probs["on"][STOP_IDX])
        ps_off = float(probs["off"][STOP_IDX])
        rec["n_open"] += 1
        rec["stop"].append(ps_on)
        rec["stop_offbias"].append(ps_off)
        rec["legal"].append(final_legal)
        rec["elixir"].append(float(p0.elixir))
        rec["hand_costs"].append([float(Card(c).elixir) for c in list(p0.cycle)[:4]])
        rec["argmax_stop"] += int(int(np.argmax(probs["on"])) == STOP_IDX)
        rec["by_legal"].setdefault(str(final_legal), []).append([ps_on, ps_off, raw_legal])
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def summarize_open(rec):
    s = np.array(rec["stop"])
    so = np.array(rec["stop_offbias"])
    legal = np.array(rec["legal"])
    hc = np.array(rec["hand_costs"])
    L = [f"### {rec['label']} —— 开局帧专用采样（n={rec['n_open']} 局，用时 {rec['seconds']}s）"]
    L.append(f"- **开局帧 P(不出牌) = {_f(s):.3f}** (中位 {np.median(s):.3f}, "
             f"p10 {_q(s,10):.3f}, p90 {_q(s,90):.3f}, 极差 {s.min():.3f}..{s.max():.3f})"
             f" | 去 plan 软偏置 = {_f(so):.3f}")
    L.append(f"- 开局 argmax=STOP 的局数 = {rec['argmax_stop']}/{rec['n_open']} "
             f"| 开局合法选项数：均值 {_f(legal):.2f} 分布 "
             f"{ {k: int(v) for k, v in zip(*np.unique(legal, return_counts=True))} }"
             f" | 开局圣水 {_f(rec['elixir']):.2f}")
    parts = []
    for k in sorted(rec["by_legal"], key=int):
        arr = np.asarray(rec["by_legal"][k], dtype=float)
        parts.append(f"合法{k}项 → {arr[:,0].mean():.3f}(去偏置 {arr[:,1].mean():.3f}, n={len(arr)})")
    L.append("- 按开局合法选项数分层：" + " | ".join(parts))
    L.append(f"- 开局首手牌费用（4 张，逐局）：均值 {hc.mean():.2f} "
             f"最便宜那张的分布 中位 {np.median(hc.min(axis=1)):.1f}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", default=[], metavar="LABEL=PATH")
    ap.add_argument("--fresh-init", action="store_true",
                    help="额外测一个全新随机初始化（未训练）的网络")
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--open-games", type=int, default=30,
                    help="开局帧专用采样局数（只 reset + 一次前向，0 = 关闭）")
    ap.add_argument("--max-frames", type=int, default=360)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json", default=os.path.join(_ROOT, "docs", "probe_pass_prob.json"))
    a = ap.parse_args()

    import torch
    device = a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu"
    from rl.env_wrapper import RLEnv
    from rl import train_solo as ts
    from rl.plan_space import PLAN_DIM
    from rl.belief import BeliefInference

    print("=" * 78)
    print("P(不出牌) = bundle 第一个 decoder 步选 STOP 的概率（掩码后 + 含 plan 软偏置）")
    print(f"device={device} games={a.games} max_frames={a.max_frames} seed={a.seed}")

    targets = []
    for spec in a.ckpt:
        label, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--ckpt 需要 LABEL=PATH，收到 {spec!r}")
        full = path if os.path.isabs(path) else os.path.join(_SRC, path)
        if not os.path.exists(full):
            raise SystemExit(f"ckpt 不存在: {full}")
        sd, meta = load_meta(full)
        targets.append((label, sd, meta))
    if a.fresh_init:
        bd = len(BeliefInference(opp_deck=list(ts.DEFAULT_SOLO_DECK), n_particles=128,
                                 seed=0).encode(None, None))
        targets.append(("fresh_init", None, {"plan_dim": PLAN_DIM, "belief_dim": bd,
                                             "hidden_dim": 128, "value_bypass": True,
                                             "value_independent": True}))
    if not targets:
        raise SystemExit("至少给一个 --ckpt 或 --fresh-init")

    env = RLEnv(opponent=None, seed=a.seed, card_level=11,
                deck0=list(ts.DEFAULT_SOLO_DECK), deck1=list(ts.DEFAULT_SOLO_DECK))

    out = []
    for label, sd, meta in targets:
        pol = build_policy(sd, meta, device)
        print(f"\n[{label}] plan_dim={pol.plan_dim} belief_dim={pol.belief_dim} "
              f"hidden={pol.hidden_dim} "
              f"slot_head.bias[STOP]={float(pol.slot_head.bias[5].detach()):.4f}")
        rec = run_model(label, pol, env, a.games, a.seed, a.max_frames, device, torch)
        if a.open_games > 0:
            rec["open_only"] = run_open_only(label, pol, env, a.open_games, a.seed,
                                             device, torch)
            print(summarize_open(rec["open_only"]))
        out.append(rec)
        print(summarize(rec))

    print("\n" + "=" * 78)
    print("汇总：①开局帧 P(不出牌) | ②有合法选项时 P(不出牌) | ③全程 | 实测空 bundle 率")
    for rec in out:
        ch = rec["stop_choice"]
        oo = rec.get("open_only")
        oo_txt = (f" [开局专用 n={oo['n_open']}: {_f(oo['stop']):.3f} "
                  f"去偏置 {_f(oo['stop_offbias']):.3f}]") if oo else ""
        print(f"  {rec['label']:<12} ① {_f(rec['open_stop']):.3f} "
              f"(去偏置 {_f(rec['open_stop_offbias']):.3f}){oo_txt} | "
              f"② {_f(ch):.3f} (n={len(ch)}) | "
              f"③ {_f(rec['stop_all']):.3f} | "
              f"空bundle {100.0*rec['empty_bundle']/max(1,rec['n_frames']):.1f}%")
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump({"args": vars(a), "records": out}, f, ensure_ascii=False, indent=1)
    print(f"[saved] {a.json}")
    print("(warn) 描述性读数（【R5】n=1 seed / 【R3】非判据）：只报观察到，不报疗效。")


if __name__ == "__main__":
    main()
