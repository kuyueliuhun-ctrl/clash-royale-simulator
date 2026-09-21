# -*- coding: utf-8 -*-
"""**国王塔落点探针**（用户 2026-09-22 提问）：8702（臂 R=0.25）某局出现「火球砸向对方国王塔」，
人类对局不会这么打 —— 到底是不是「网络少输入了东西」？

本探针**不改任何生产代码**：它把 `scripts/il_readout_games.py` 当模块加载并跑**同一套** setup
（同 seed / 同卡组 / 同对手包装 / 同 `--hidden none --opp-hidden same` 口径），只在
`FollowerPolicy.act` 外面套一层**只读钩子**，在指定「第 g 局第 k 个 p0 决策帧」把这一刻的
**输入与内部张量**打出来：

* 掩码：该槽位合法格数、我们关心的若干格是否合法（**这是"模型能不能去那儿"的唯一硬约束**）；
* plan 向量：`focus_region`（v[8:16]）/ `suggested_card`（v[16]）/ hold_mask（尾部 4 维），
  以及 `_plan_biases` 产出的 `slot_bias` / `cell_bias`（**plan 是软偏置，能改 argmax**）；
* 槽位分布：`slot_head(h)+slot_bias` 掩码后的 argmax（= 模型本帧的"出不出/出哪张"）；
* 落点分布：`cell_head(h)` 的**原始** top-k 与**加 plan 偏置后**的 top-k（掩码后/掩码前各一份），
  每格换算成世界坐标并算出「到对方国王塔的距离」「到最近敌方实体的距离」；
* 场上实体（对方部队/塔）与当时的圣水/时间。

⇒ 一句话用途：把「砸王塔」分解成 **① 掩码允许（去得了）② 模型自己选（cell_head 原始 argmax 就在那儿）
③ 还是 plan 软偏置把它推过去的** 三种成因。

用法（仓库根，Windows venv；必须在 `src/clasher_new` 下跑或让本脚本 chdir）:
    ./.venv/Scripts/python.exe scripts/il_probe_kingtower_cast.py \
        --ckpt runs/_fl_il_bc_save/sweep_mixR025/bc_fl_e3_lr0.001.pt \
        --game 2 --frames 0,1 --out docs/fl_il_2026-09-21/probe_kingtower_cast.json
"""
import argparse
import importlib.util
import io
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

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rl.action_bundle import ActionBundle, sub_position  # noqa: E402
from rl.env_wrapper import RLEnv  # noqa: E402
from rl.follower import (FollowerPolicy, GRID_H, GRID_W, K_MAX,  # noqa: E402
                         STOP_IDX, _REGION_CENTERS)
from rl.observation import ENTITY_NAMES  # noqa: E402
from rl.plan_space import FOCUS_REGIONS  # noqa: E402


def _abs(p):
    m = p.replace("\\", "/")
    if m.startswith("/mnt/") and len(m) > 6:
        head, rest = m.split("/", 3)[2:]
        return head.upper() + ":\\" + rest.replace("/", "\\")
    return os.path.normpath(os.path.join(ORIG_CWD, p))


GEN = {"g": -1, "k": -1}          # 当前局号 / 该局内 p0 决策序号
OUT = {"probe": []}


def _towers(battle):
    out = []
    for e in battle.entities.values():
        nm = getattr(e, "name", "") or ""
        if "Tower" in nm:
            out.append({"id": int(getattr(e, "id", -1)), "name": nm, "player": int(e.player),
                        "x": round(float(e.position.x), 2), "y": round(float(e.position.y), 2),
                        "hp": round(float(e.hp), 1), "alive": bool(e.is_alive)})
    return sorted(out, key=lambda r: (r["player"], r["x"]))


def _units(battle, player):
    out = []
    for e in battle.entities.values():
        if int(getattr(e, "player", -1)) != player or not e.is_alive:
            continue
        nm = getattr(e, "name", "") or ""
        if "Tower" in nm:
            continue
        out.append({"name": nm, "x": round(float(e.position.x), 2),
                    "y": round(float(e.position.y), 2), "hp": round(float(e.hp), 1)})
    return out


def _dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _top_cells(logits, cells_mask, k=8):
    """返回 (flat_idx, logit, legal) 的 top-k（在**掩码前**的 logits 上取）。"""
    flat = np.asarray(logits, dtype=np.float64).reshape(-1)
    order = np.argsort(-flat)[:k]
    return [(int(i), float(flat[i]), bool(cells_mask.reshape(-1)[i])) for i in order]


def _cell_class(battle, x, y):
    """该本地格上是什么（塔/己方塔/敌方单位/空）——只用于分类统计。"""
    _pos = sub_position(0, x, y)
    wx, wy = float(_pos.x), float(_pos.y)
    best, bestd = None, 1e9
    for e in battle.entities.values():
        if not e.is_alive:
            continue
        d = float(np.hypot(wx - float(e.position.x), wy - float(e.position.y)))
        if d < bestd:
            bestd, best = d, e
    if best is None or bestd > 2.0:
        return "empty"
    nm = getattr(best, "name", "") or ""
    if "Tower" not in nm:
        return "unit"
    return "tower_own" if int(best.player) == 0 else "tower_enemy"


def capture_light(pol, obs, plan_token, get_mask):
    """轻量版：每个决策帧只记「槽位 / 落点 / cell_head 原始 top-1 / 合法性 / plan 区域」。"""
    env = getattr(get_mask, "__self__", None)
    battle = env.battle
    mask0 = get_mask(ActionBundle())
    v = np.asarray(plan_token, dtype=np.float64).reshape(-1)
    region = None
    if v.shape[0] >= 16 and float(v[8:16].max()) > 0:
        region = FOCUS_REGIONS[int(np.argmax(v[8:16]))]
    with torch.no_grad():
        _f, enc = pol._encode_parts(obs, pol_belief, plan_token)
        h = pol.gru_cell(enc, torch.zeros(1, pol.hidden_dim, device=pol.device))
        slot_bias, cell_bias = pol._plan_biases(plan_token)
        sl = (pol.slot_head(h) + slot_bias).masked_fill(pol._slot_mask_tensor(mask0) == 0, -1e9)
        option = int(torch.argmax(sl, dim=-1).item())
        rec = {"game": GEN["g"], "k": GEN["k"], "t": round(float(battle.time), 2),
               "elixir": round(float(battle.players[0].elixir), 2),
               "hand": [ENTITY_NAMES[int(c)] for c in np.asarray(obs["hand"]).reshape(-1)[:5]],
               "option": option, "region": region,
               "n_legal": 0, "pick": None, "top1": None}
        if option < K_MAX:
            mL = mask0["cells"][option].reshape(-1)
            raw = pol.cell_head(h).view(GRID_H, GRID_W).detach().cpu().numpy().reshape(-1)
            cb = (cell_bias.detach().cpu().numpy().reshape(-1)
                  if torch.is_tensor(cell_bias) else np.asarray(cell_bias).reshape(-1))
            rec["n_legal"] = int(mL.sum())
            rec["card"] = rec["hand"][option]
            t1 = int(np.argmax(raw))
            rec["top1"] = {"cell": [t1 % GRID_W, t1 // GRID_W], "legal": bool(mL[t1]),
                           "cls": _cell_class(battle, t1 % GRID_W, t1 // GRID_W)}
            pk = int(np.argmax(np.where(mL, raw + cb, -1e9)))
            rec["pick"] = {"cell": [pk % GRID_W, pk // GRID_W], "cls": _cell_class(battle, pk % GRID_W, pk // GRID_W)}
            rec["top1_tower_legal"] = bool(mL[t1] and _cell_class(battle, t1 % GRID_W, t1 // GRID_W).startswith("tower"))
    OUT["probe"].append(rec)
    print("[light] " + json.dumps(rec, ensure_ascii=False), flush=True)
    return rec


def capture(pol, obs, plan_token, get_mask, tag):
    env = getattr(get_mask, "__self__", None)
    battle = env.battle
    me, opp = battle.players[0], battle.players[1]
    mask0 = get_mask(ActionBundle())
    slot_bias, cell_bias = pol._plan_biases(plan_token)
    sb = slot_bias.detach().cpu().numpy() if torch.is_tensor(slot_bias) else np.asarray(slot_bias)
    cb = cell_bias.detach().cpu().numpy() if torch.is_tensor(cell_bias) else np.asarray(cell_bias)

    v = np.asarray(plan_token, dtype=np.float64).reshape(-1)
    region = None
    if v.shape[0] >= 16:
        rseg = v[8:16]
        if float(rseg.max()) > 0:
            region = FOCUS_REGIONS[int(np.argmax(rseg))]
    sug = int(round(float(v[16]) * 4.0)) if v.shape[0] > 16 and v[16] > 0 else None

    with torch.no_grad():
        fused, enc = pol._encode_parts(obs, pol_belief, plan_token)
        h = pol.gru_cell(enc, torch.zeros(1, pol.hidden_dim, device=pol.device))
        slot_logits = pol.slot_head(h) + slot_bias
        slot_mask = pol._slot_mask_tensor(mask0)
        slot_logits = slot_logits.masked_fill(slot_mask == 0, -1e9)
        option = int(torch.argmax(slot_logits, dim=-1).item())
        slot_p = torch.softmax(slot_logits, dim=-1).detach().cpu().numpy().reshape(-1)
        raw_slot = pol.slot_head(h) + slot_bias
        raw_slot_np = raw_slot.detach().cpu().numpy().reshape(-1)

        cell_info = None
        if option < K_MAX:
            cells = torch.as_tensor(mask0["cells"][option], dtype=torch.float32, device=pol.device)
            raw_cell = pol.cell_head(h).view(GRID_H, GRID_W).detach().cpu().numpy()
            with_bias = raw_cell + cb
            mL = mask0["cells"][option]
            lg = with_bias.reshape(-1)
            mflat = mL.reshape(-1)
            masked = np.where(mflat, lg, -1e9)
            pick = int(np.argmax(masked))
            # 无 plan 偏置时的 argmax（隔离 plan 的影响）
            masked_nobias = np.where(mflat, raw_cell.reshape(-1), -1e9)
            pick_nobias = int(np.argmax(masked_nobias))
            legal = np.where(mflat)[0]

            king = None
            for e in battle.entities.values():
                if getattr(e, "name", "") == "KingTower" and int(getattr(e, "player", -1)) == 1:
                    king = (float(e.position.x), float(e.position.y))
            def cell_rec(i):
                x, y = int(i % GRID_W), int(i // GRID_W)
                _pos = sub_position(0, x, y)
                wx, wy = float(_pos.x), float(_pos.y)
                rec = {"cell": [x, y], "flat": int(i), "legal": bool(mflat[i]),
                       "logit_with_bias": round(float(lg[i]), 4),
                       "logit_raw": round(float(raw_cell.reshape(-1)[i]), 4),
                       "world": [round(float(wx), 2), round(float(wy), 2)],
                       "cell_bias": round(float(cb.reshape(-1)[i]), 4)}
                if king is not None:
                    rec["dist_to_enemy_king"] = round(_dist((wx, wy), king), 2)
                return rec

            cell_info = {
                "slot_option": option,
                "n_legal_cells": int(mflat.sum()),
                "argmax_with_bias": cell_rec(pick),
                "argmax_raw_no_bias": cell_rec(pick_nobias),
                "top8_raw_logits": [cell_rec(i) for i, _l, _g in _top_cells(raw_cell, mL, 8)],
                "legal_flat_minmax": [int(legal.min()), int(legal.max())] if legal.size else None,
                "enemy_king_world": [round(king[0], 2), round(king[1], 2)] if king else None,
            }
            # ---- 闸门归因：把「掩码为什么放行这一格」逐条拆开 ----
            try:
                from rl.action_mask import (_spell_has_enemy_target,
                                            _spell_tower_ev_illegal, _spell_radius_m)
                from card_utils import Card as _Card
                cname = ENTITY_NAMES[int(np.asarray(obs["hand"]).reshape(-1)[option])]
                _ci = _Card(cname)
                _r = _spell_radius_m(cname, _ci)
                alive_pr = []
                for e in battle.entities.values():
                    if (getattr(e, "name", "") == "King_PrincessTowers"
                            and int(getattr(e, "player", -1)) == 1 and e.is_alive):
                        alive_pr.append((float(e.position.x), float(e.position.y)))
                king_xy = king
                units = [(float(e.position.x), float(e.position.y),
                          float(getattr(getattr(e, "data", None), "collision_radius", 0.5) or 0.5))
                         for e in battle.entities.values()
                         if int(getattr(e, "player", -1)) == 1 and e.is_alive
                         and "Tower" not in (getattr(e, "name", "") or "")]

                def gate(i):
                    x, y = int(i % GRID_W), int(i // GRID_W)
                    _p = sub_position(0, x, y)
                    wx, wy = float(_p.x), float(_p.y)
                    dk = _dist((wx, wy), king_xy) if king_xy else None
                    dpr = [_dist((wx, wy), t) for t in alive_pr]
                    in_pr = any(d <= _r + 1.4 for d in dpr)
                    in_king = (dk is not None and dk <= _r + 1.4)
                    in_unit = any(_dist((wx, wy), (ux, uy)) <= _r + cr
                                  for ux, uy, cr in units)
                    return {"cell": [x, y], "world": [round(wx, 2), round(wy, 2)],
                            "d_king": None if dk is None else round(dk, 2),
                            "d_princess_min": round(min(dpr), 2) if dpr else None,
                            "splash_princess": bool(in_pr), "splash_king": bool(in_king),
                            "splash_unit": bool(in_unit),
                            "has_enemy_target": bool(_spell_has_enemy_target(battle, 0, _p, _r)),
                            "ev_gate_illegal": bool(_spell_tower_ev_illegal(battle, 0, cname, _p, _ci)),
                            "legal": bool(mflat[i])}

                cats = {"princess": 0, "king_only": 0, "unit": 0, "none": 0}
                for i in np.where(mflat)[0]:
                    g = gate(int(i))
                    if g["splash_unit"]:
                        cats["unit"] += 1
                    elif g["splash_princess"]:
                        cats["princess"] += 1
                    elif g["splash_king"]:
                        cats["king_only"] += 1
                    else:
                        cats["none"] += 1
                cell_info["gate"] = {
                    "card": cname, "radius": _r,
                    "chosen_cell_detail": gate(pick),
                    "princess_cell_detail": gate(int(np.argmax(raw_cell.reshape(-1)))),
                    "legal_set_categories": cats,
                }
            except Exception as exc:
                cell_info["gate"] = {"error": f"{type(exc).__name__}: {exc}"}

            # 对方国王塔格的掩码合法性（世界坐标 → 本地格）
            if king is not None:
                kx, ky = int(king[0]), int(king[1])
                cell_info["king_cell_legal"] = bool(mL[ky, kx])
                cell_info["king_cell_flat"] = int(ky * GRID_W + kx)

    rec = {
        "tag": tag, "game": GEN["g"], "p0_step": GEN["k"],
        "t": round(float(battle.time), 2),
        "my_elixir": round(float(me.elixir), 3), "opp_elixir": round(float(opp.elixir), 3),
        "hand": [ENTITY_NAMES[int(c)] for c in np.asarray(obs["hand"]).reshape(-1)[:5]],
        "next_card": ENTITY_NAMES[int(np.asarray(obs["next_card"]).reshape(-1)[0])],
        "towers": _towers(battle),
        "enemy_units": _units(battle, 1),
        "plan": {"focus_region": region,
                 "region_center": list(_REGION_CENTERS.get(region, ())) if region else None,
                 "suggested_slot": sug,
                 "hold_mask": [round(float(x), 2) for x in v[-4:]] if v.shape[0] >= 4 else None,
                 "plan_dim": int(v.shape[0])},
        "plan_bias": {"slot_bias": [round(float(x), 3) for x in sb],
                      "cell_bias_argmax_cell": [int(np.argmax(cb) % GRID_W), int(np.argmax(cb) // GRID_W)],
                      "cell_bias_pos_flat": int((cb.reshape(-1) > 0).sum())},
        "slot": {"argmax_option": option, "is_stop": bool(option == STOP_IDX),
                 "mask_slots": [bool(x) for x in np.asarray(mask0["slots"]).reshape(-1)],
                 "probs": [round(float(x), 4) for x in slot_p],
                 "raw_logits": [round(float(x), 3) for x in raw_slot_np]},
        "cell": cell_info,
    }
    OUT["probe"].append(rec)
    print(json.dumps(rec, ensure_ascii=False, indent=1), flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--game", type=int, default=2, help="第几局（0-based，与 runner 的 g 一致）")
    ap.add_argument("--frames", default="0,1", help="该局里要探的 p0 决策序号（0-based，逗号分隔）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--light-games", default=None,
                    help="对这些局（0-based，逗号分隔）**每个** p0 决策帧做轻量统计")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    want = {int(x) for x in args.frames.split(",") if x.strip() != ""}
    #: `--light-games 0,1,2` ⇒ 对这些局**每个** p0 决策帧做实垂统计（塔落点先验的分布证据）
    global light_games
    light_games = ({int(x) for x in args.light_games.split(",") if x.strip() != ""}
                   if args.light_games else set())

    # ---- 载入 runner 模块（同 setup，零漂移）----
    spec = importlib.util.spec_from_file_location(
        "il_readout_games", os.path.join(HERE, "il_readout_games.py"))
    ilr = importlib.util.module_from_spec(spec)
    sys.modules["il_readout_games"] = ilr
    spec.loader.exec_module(ilr)

    orig_reset = RLEnv.reset
    orig_act = FollowerPolicy.act

    def patched_reset(self, *a, **k):
        GEN["g"] += 1
        GEN["k"] = -1
        return orig_reset(self, *a, **k)

    def patched_act(self, obs, belief_token, plan_token, get_mask, hidden=None,
                    deterministic=False):
        caller = sys._getframe(1)
        is_p0 = caller.f_code.co_filename.endswith("il_readout_games.py")
        global pol_belief, pol_plan
        if is_p0:
            GEN["k"] += 1
            if GEN["g"] == args.game and GEN["k"] in want:
                pol_belief, pol_plan = belief_token, plan_token
                try:
                    capture(self, obs, plan_token, get_mask, tag="p0")
                except Exception as exc:            # 探针失败绝不改变对局
                    print(f"[probe] capture 失败（已忽略）：{type(exc).__name__}: {exc}", flush=True)
            elif GEN["g"] in light_games:
                pol_belief = belief_token
                try:
                    capture_light(self, obs, plan_token, get_mask)
                except Exception as exc:
                    print(f"[light] 失败（已忽略）：{type(exc).__name__}: {exc}", flush=True)
        return orig_act(self, obs, belief_token, plan_token, get_mask,
                        hidden=hidden, deterministic=deterministic)

    RLEnv.reset = patched_reset
    FollowerPolicy.act = patched_act
    try:
        sys.argv = ["il_readout_games.py", "--ckpt", _abs(args.ckpt), "--games",
                    str(args.game + 1), "--decks", "solo", "--hidden", "none",
                    "--opp-hidden", "same", "--out", _abs("runs/_il_probe_kt"),
                    "--json", _abs("docs/fl_il_2026-09-21/probe_kingtower_cast_stats.json"),
                    "--seed", str(args.seed)]
        ilr.main()
    finally:
        RLEnv.reset = orig_reset
        FollowerPolicy.act = orig_act

    if args.out:
        p = _abs(args.out)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with io.open(p, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"ckpt": args.ckpt, "game": args.game, "frames": sorted(want),
                       "probe": OUT["probe"]}, f, ensure_ascii=False, indent=1)
        print(f"[probe] 落盘 {p}", flush=True)


pol_belief = None
pol_plan = None
light_games = set()

if __name__ == "__main__":
    main()
