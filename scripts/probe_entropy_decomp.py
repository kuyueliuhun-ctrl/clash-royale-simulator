# -*- coding: utf-8 -*-
"""熵项预算分解探针（只读；上游 `7fc64f2` 借鉴项的量化仪器）。

问题：`ent_coef=0.01` 这笔钱**付给了谁**？
  - 槽位步（deploy 步）的 slot 熵
  - 落点步的 cell 熵（576 类）
  - 终止步（STOP 步）的 slot 熵
上游 7fc64f2 把落点熵整块删掉、把 noop 从熵里剔除、再加 `eligible = affordable_count >= 2` 门控。
本脚本量出**我们这边这三项各占多少**，并给出「照上游口径重算」的反事实量。

口径：与生产 `FollowerPolicy.act(evaluate=False)` / `evaluate()` 逐位同源 ——
  - 轨迹 = `act(..., deterministic=False)` 真实采样（训练看到的就是这个）
  - 分解用的 ops 逐字复刻 `evaluate()`；**自检**：分解之和必须等于 `evaluate()` 返回的总熵（tol 1e-4）
  - 反事实（upstream 口径）= 只对 `cells/slots` 合法的手牌取 softmax 求熵，丢 ability/STOP；再 × 门控

用法（必须在 src/clasher_new 下）：
  cd src/clasher_new && PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe \
      ../../scripts/probe_entropy_decomp.py \
      --ckpt trained=runs/et_solo100k/solo_main.pt --fresh-init --games 4
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SRC = os.path.join(_ROOT, "src", "clasher_new")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from rl.io_bootstrap import force_utf8_stdout  # noqa: E402
force_utf8_stdout()

import numpy as np  # noqa: E402


def decomp(pol, obs, btok, plan_vec, bundle, masks, hidden_in, torch):
    """逐字复刻 evaluate() 的 decoder 循环，但把三项熵分开记账。"""
    import torch.nn.functional as F
    from rl.action_bundle import ActionBundle  # noqa: F401
    from rl.follower import ABILITY_IDX, GRID_H, GRID_W, K_MAX

    with torch.no_grad():
        _fused, enc = pol._encode_parts(obs, btok, plan_vec)
        if hidden_in is None:
            h0 = torch.zeros(1, pol.hidden_dim, device=pol.device)
        else:
            h0 = hidden_in.reshape(1, -1).detach()
        h = pol.gru_cell(enc, h0)
        slot_bias, cell_bias = pol._plan_biases(plan_vec)

        slot_e = 0.0          # deploy 步的 slot 熵
        cell_e = 0.0          # deploy 步的 cell 熵（576 类）
        stop_e = None         # 终止步 slot 熵
        steps = []
        card_only_raw = []    # 反事实：仅对手牌取 softmax 的熵
        for i, sa in enumerate(bundle.sub_actions):
            mask = masks[i]
            n_deploy = int(np.sum(mask["slots"]))
            n_ability = int(bool(mask.get("ability_legal")))
            sm = pol._slot_mask_tensor(mask)
            lg = pol.slot_head(h) + slot_bias
            lg = lg.masked_fill(sm == 0, -1e9)
            sd = torch.distributions.Categorical(logits=F.log_softmax(lg, dim=-1))
            e_slot = float(sd.entropy().item())
            slot_e += e_slot

            # 反事实：只留 K_MAX 张手牌（丢 ability/STOP），逐字照上游 masked_fill
            card_lg = lg.reshape(-1)[:K_MAX].clone()
            card_lg = card_lg.masked_fill(card_lg <= -1e8, -1e9)
            cd_raw = torch.distributions.Categorical(
                logits=F.log_softmax(card_lg, dim=-1))
            card_only_raw.append(float(cd_raw.entropy().item()))

            if sa.kind == "ability":
                steps.append({"kind": "ability", "slot_e": e_slot, "n_deploy": n_deploy})
                h = pol._sub_update(h, ABILITY_IDX)
                continue

            option = sa.slot - 1
            cells = torch.as_tensor(mask["cells"][option], dtype=torch.float32,
                                    device=pol.device)
            n_cells = int(cells.sum().item())
            cl = pol.cell_head(h).view(1, GRID_H, GRID_W) + cell_bias
            cl = cl.masked_fill(cells == 0, -1e9)
            cell_dist = torch.distributions.Categorical(
                logits=F.log_softmax(cl.reshape(1, -1), dim=-1))
            e_cell = float(cell_dist.entropy().item())
            cell_e += e_cell
            steps.append({"kind": "deploy", "slot_e": e_slot, "cell_e": e_cell,
                          "n_cells": n_cells, "n_deploy": n_deploy,
                          "n_ability": n_ability})
            h = pol._sub_update(h, option, sa.x, sa.y)

        # 终止步（evaluate 的 STOP 分支逐字）
        if len(masks) > len(bundle.sub_actions):
            mask = masks[len(bundle.sub_actions)]
        else:
            mask = {"slots": np.ones(K_MAX, dtype=bool),
                    "cells": np.ones((K_MAX, GRID_H, GRID_W), dtype=bool),
                    "ability_legal": False}
        n_deploy_stop = int(np.sum(mask["slots"]))
        sm = pol._slot_mask_tensor(mask)
        lg = pol.slot_head(h) + slot_bias
        lg = lg.masked_fill(sm == 0, -1e9)
        sd = torch.distributions.Categorical(logits=F.log_softmax(lg, dim=-1))
        stop_e = float(sd.entropy().item())
        if bundle.size > 0:
            # 只有"上一子动作之后还允许继续"的帧才有这一步的语义；空包时这一步= 第一决策步
            pass
        card_lg = lg.reshape(-1)[:K_MAX].clone()
        card_lg = card_lg.masked_fill(card_lg <= -1e8, -1e9)
        cd_raw = torch.distributions.Categorical(logits=F.log_softmax(card_lg, dim=-1))
        card_only_raw.append(float(cd_raw.entropy().item()))
    return {"slot_e": slot_e, "cell_e": cell_e, "stop_e": stop_e, "steps": steps,
            "n_deploy_stop_legal": n_deploy_stop,
            "card_only_raw": card_only_raw}


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
            print(f"  [warn] load: missing={len(miss)} unexpected={len(unexp)}")
    pol.eval()
    pol.to_device(device)
    return pol


def load_meta(path):
    import torch
    d = torch.load(path, map_location="cpu")
    if isinstance(d, dict) and "state_dict" in d:
        return d["state_dict"], {k: v for k, v in d.items() if k != "state_dict"}
    return d, {}


def run(label, pol, env, n_games, seed, max_frames, device, torch):
    from rl.belief import BeliefInference
    from rl.belief_planner import BeliefPlanner
    from rl import train_solo as ts

    opp = build_policy(None, {"plan_dim": pol.plan_dim, "belief_dim": pol.belief_dim,
                              "hidden_dim": pol.hidden_dim,
                              "value_bypass": pol.value_bypass,
                              "value_independent": pol.value_independent}, device)
    opp.load_state_dict(pol.state_dict())
    opp.eval()

    bp = BeliefPlanner()
    acc = {"label": label, "frames": 0, "sum_slot_e": 0.0, "sum_cell_e": 0.0,
           "sum_stop_e": 0.0, "sum_total_e": 0.0, "sum_card_only_raw": 0.0,
           "mismatch": 0, "worst_mismatch": 0.0,
           "n_deploy_steps": 0, "sum_n_cells": 0.0, "max_n_cells": 0,
           "n_choice_ge2": 0, "n_choice_1": 0, "n_choice_0": 0,
           "stop_e_when_choice": 0.0, "n_stop_step_with_choice": 0,
           "stop_e_when_forced": 0.0, "n_stop_step_forced": 0,
           "n_empty_bundle": 0, "legal_hist": {}, "n_ability": 0,
           "elixir_all": [], "choice_elixir": [], "choice_ge6": 0,
           "choice_ge6_stop": 0, "choice_ge2_stop": 0, "n_choice": 0,
           "n_elixir_ge6": 0}
    t0 = time.time()
    for g in range(n_games):
        env.opponent = ts.FollowerOpponent(
            opp, env, belief=BeliefInference(opp_deck=env.deck1, n_particles=128,
                                             seed=seed + g), deterministic=True)
        belief = BeliefInference(opp_deck=env.deck1, n_particles=128, seed=seed + 1000 + g)
        obs, _ = env.reset(seed=seed + 2000 + g)
        hidden = None
        for _step in range(max_frames):
            try:
                plan_vec = bp.plan(env.battle, belief.state(), obs).to_vector()
            except Exception:
                plan_vec = np.zeros(pol.plan_dim, dtype=np.float32)
            if not isinstance(plan_vec, np.ndarray) or plan_vec.shape[0] != pol.plan_dim:
                plan_vec = np.zeros(pol.plan_dim, dtype=np.float32)
            btok = belief.encode(obs, None)

            mask0 = env.get_action_mask(None)
            n_deploy0 = int(np.sum(mask0["slots"]))
            bundle, _lp, _v, hidden_new, masks = pol.act(
                obs, btok, plan_vec.astype(np.float32), env.get_action_mask,
                hidden=hidden, deterministic=False)

            d = decomp(pol, obs, btok, plan_vec, bundle, masks, hidden, torch)
            _lpe, _ve, _he, total_e = pol.evaluate(obs, btok, plan_vec.astype(np.float32),
                                                   bundle, masks, hidden=hidden)
            total_e = float(total_e.item())
            mine = d["slot_e"] + d["cell_e"] + d["stop_e"]
            diff = abs(mine - total_e)
            acc["worst_mismatch"] = max(acc["worst_mismatch"], diff)
            if diff > 1e-3:
                acc["mismatch"] += 1

            acc["frames"] += 1
            acc["sum_slot_e"] += d["slot_e"]
            acc["sum_cell_e"] += d["cell_e"]
            acc["sum_stop_e"] += d["stop_e"]
            acc["sum_total_e"] += total_e
            acc["sum_card_only_raw"] += float(np.sum(d["card_only_raw"]))
            for st in d["steps"]:
                if st["kind"] == "deploy":
                    acc["n_deploy_steps"] += 1
                    acc["sum_n_cells"] += st["n_cells"]
                    acc["max_n_cells"] = max(acc["max_n_cells"], st["n_cells"])
                else:
                    acc["n_ability"] += 1
            # 合法的"手牌可选项"个数（不含 ability/STOP）分层
            k = str(n_deploy0)
            acc["legal_hist"][k] = acc["legal_hist"].get(k, 0) + 1
            if n_deploy0 >= 2:
                acc["n_choice_ge2"] += 1
            elif n_deploy0 == 1:
                acc["n_choice_1"] += 1
            else:
                acc["n_choice_0"] += 1
            # STOP 步熵：STOP 是不是"真选择"
            choice_exists = bool(n_deploy0 > 0 or mask0.get("ability_legal"))
            if choice_exists:
                acc["stop_e_when_choice"] += d["stop_e"]
                acc["n_stop_step_with_choice"] += 1
            else:
                acc["stop_e_when_forced"] += d["stop_e"]
                acc["n_stop_step_forced"] += 1
            if bundle.size == 0:
                acc["n_empty_bundle"] += 1

            acc["elixir_all"].append(float(env.battle.players[0].elixir))
            if n_deploy0 >= 1:
                acc["n_choice"] += 1
                acc["choice_elixir"].append(float(env.battle.players[0].elixir))
                if bundle.size == 0:
                    acc["choice_ge2_stop"] += int(n_deploy0 >= 2)
                if float(env.battle.players[0].elixir) >= 6.0:
                    acc["choice_ge6"] += 1
                    if bundle.size == 0:
                        acc["choice_ge6_stop"] += 1
            if float(env.battle.players[0].elixir) >= 6.0:
                acc["n_elixir_ge6"] += 1
            obs2, _r, term, trunc, info = env.step(bundle)
            belief.update(obs2, info.get("opp_played"))
            obs = obs2
            hidden = hidden_new
            if term or trunc:
                break
        print(f"  [game {g+1}/{n_games}] frames={acc['frames']} t={time.time()-t0:.0f}s")
    acc["seconds"] = round(time.time() - t0, 1)
    return acc


def report(acc):
    f = max(1, acc["frames"])
    tot = acc["sum_total_e"]
    print("\n" + "=" * 78)
    print(f"[{acc['label']}] frames={acc['frames']} games4={acc['seconds']}s")
    print(f"  自检 |分解和 − evaluate()|: 超标帧 {acc['mismatch']} / {acc['frames']} "
          f"(worst {acc['worst_mismatch']:.2e})")
    print(f"  总熵 Σ over frames = {tot:.1f}  (每帧均值 {tot/f:.4f} nat)")
    for tag, key in (("槽位步(deploy)", "sum_slot_e"), ("落点步(cell 576)", "sum_cell_e"),
                     ("终止步(STOP)", "sum_stop_e"),
                     ("反事实:仅手牌原始", "sum_card_only_raw")):
        v = acc[key]
        print(f"    {tag:<20} {v:9.1f}  占比 {100*v/tot if tot else float('nan'):6.2f}%"
              f"   每帧均值 {v/f:.4f}")
    nd = max(1, acc["n_deploy_steps"])
    print(f"  deploy 步 {acc['n_deploy_steps']} 次；合法格数 均值 {acc['sum_n_cells']/nd:.1f} "
          f"最大 {acc['max_n_cells']}（ln 均值格 ≈ {math.log(max(1.0, acc['sum_n_cells']/nd)):.2f} nat）")
    print(f"  手牌合法选项分布 {dict(sorted(acc['legal_hist'].items(), key=lambda kv: int(kv[0])))}")
    print(f"    ≥2 项(上游 eligible) {acc['n_choice_ge2']}   "
          f"=1 项 {acc['n_choice_1']}   0 项 {acc['n_choice_0']}")
    print(f"  STOP 步熵:   有得选 {acc['stop_e_when_choice']:.1f} / "
          f"{acc['n_stop_step_with_choice']} 帧 = "
          f"{acc['stop_e_when_choice']/max(1,acc['n_stop_step_with_choice']):.4f}/帧  |  "
          f"没得选 {acc['stop_e_when_forced']:.1f} / {acc['n_stop_step_forced']} 帧 = "
          f"{acc['stop_e_when_forced']/max(1,acc['n_stop_step_forced']):.4f}/帧")
    print(f"  空 bundle {acc['n_empty_bundle']}/{acc['frames']} "
          f"({100.0*acc['n_empty_bundle']/f:.1f}%)  ability 步 {acc['n_ability']}")
    ea = np.array(acc["elixir_all"]) if acc["elixir_all"] else np.zeros(1)
    ce = np.array(acc["choice_elixir"]) if acc["choice_elixir"] else np.zeros(1)
    print(f"  圣水：全帧 中位 {np.median(ea):.2f} 均值 {ea.mean():.2f}   "
          f"≥6 帧 {acc['n_elixir_ge6']}/{f} ({100.0*acc['n_elixir_ge6']/f:.2f}%)")
    print(f"  有≥1 合法手牌选项的帧 {acc['n_choice']}；这些帧圣水 中位 {np.median(ce):.2f} "
          f"均值 {ce.mean():.2f} 最高 {ce.max():.2f}")
    print(f"  其中圣水 ≥6 的帧 {acc['choice_ge6']}；这些帧里 **空 bundle(=选了 STOP)** "
          f"{acc['choice_ge6_stop']}；有≥2 选项且空 bundle {acc['choice_ge2_stop']}")
    print(f"  熵奖励幅度（×ent_coef=0.01）: 总 {0.01*tot/f:.5f}/帧；"
          f"其中落点 {0.01*acc['sum_cell_e']/f:.5f}、STOP {0.01*acc['sum_stop_e']/f:.5f}")
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", default=[], metavar="LABEL=PATH")
    ap.add_argument("--fresh-init", action="store_true")
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--max-frames", type=int, default=360)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--json", default=os.path.join(_ROOT, "docs",
                                                   "probe_entropy_decomp.json"))
    a = ap.parse_args()

    import torch
    device = a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu"
    from rl.env_wrapper import RLEnv
    from rl import train_solo as ts
    from rl.plan_space import PLAN_DIM
    from rl.belief import BeliefInference

    targets = []
    for spec in a.ckpt:
        label, _, path = spec.partition("=")
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
        out.append(report(run(label, pol, env, a.games, a.seed, a.max_frames, device, torch)))
    with open(a.json, "w", encoding="utf-8") as fh:
        json.dump({"args": vars(a), "records": out}, fh, ensure_ascii=False, indent=1)
    print(f"[saved] {a.json}")
    print("(warn) 描述性读数（【R5】n=1 seed / 【R3】非判据）：只报观察到，不报疗效。")


if __name__ == "__main__":
    main()
