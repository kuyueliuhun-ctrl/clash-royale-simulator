"""只读：100k 策略的**逐帧奖励分量分解**（引擎真值，不做重建）。

预注册：`docs/reward_composition_prereg_2026-09-14.md`（判据 W1-W4 在跑之前写死，【红线 R3】）。

做法：在**内存里**把 `rl.env_wrapper.compute_reward` 包一层记录器（**不修改磁盘代码**），
对同一组输入额外算 4 个"只去掉某一项"的变体——`full` / 去 edw / 去 crown / 去塔伤 / 去终局。
每项对权重都是**线性**的 ⇒ 差值是**精确项**。单位项 = `step_reward − full`
（`env_wrapper.py:701` 在 `compute_reward` 之外加）。

用法（在 src/clasher_new 下；**用 CPU**，因为 GPU 上有别的只读探针在跑）：
  PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe \
      ../../scripts/probe_reward_composition.py \
      --ckpt runs/d1_long_100k/solo_main_100000.pt --frames 4000 --seed 7 --device cpu \
      --out ../../docs/reward_composition_100k.json
"""

import argparse
import json
import os
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src", "clasher_new")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import rl.env_wrapper as EW  # noqa: E402
from rl.config import TrainConfig  # noqa: E402
from rl.belief import BeliefInference  # noqa: E402
from rl.belief_planner import BeliefPlanner  # noqa: E402
from rl.plan_space import PLAN_DIM  # noqa: E402
from rl.follower import FollowerPolicy, load_checkpoint  # noqa: E402
from rl.train_follower import FollowerOpponent  # noqa: E402
from rl.train_solo import solo_env, resolve_deck_set  # noqa: E402
from rl.ppo import PPOTrainer  # noqa: E402

#: 最近一次 `compute_reward` 调用的分量（由下面的包装器写入）
REC = {}
_ORIG = EW.compute_reward


def _install_recorder():
    def wrapped(rw, **kw):
        full = _ORIG(rw, **kw)

        def variant(**over):
            r2 = dict(rw)
            r2.update(over)
            return _ORIG(r2, **kw)

        no_edw = variant(elixir_diff_weight=0.0)
        no_crown = variant(crown_weight=0.0, crown_lose_weight=0.0)
        no_tower = variant(tower_dmg_opp=0.0, tower_dmg_self=0.0)
        no_term = variant(win_bonus=0.0, lose_penalty=0.0, draw_penalty=0.0)
        REC.clear()
        REC.update({
            "full": float(full),
            "edw": float(full - no_edw),
            "crown": float(full - no_crown),
            "tower": float(full - no_tower),
            "terminal": float(full - no_term),
            "edw_coef": float(rw.get("elixir_diff_weight") or 0.0),
            "tower_opp_coef": float(rw.get("tower_dmg_opp") or 0.0),
        })
        return full
    EW.compute_reward = wrapped


def _stat(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {"n": 0}
    return {"n": int(x.size), "mean": round(float(x.mean()), 6),
            "median": round(float(np.median(x)), 6),
            "sum": round(float(x.sum()), 4),
            "sum_abs": round(float(np.abs(x).sum()), 4),
            "p10": round(float(np.quantile(x, 0.10)), 6),
            "p90": round(float(np.quantile(x, 0.90)), 6),
            "max_abs": round(float(np.abs(x).max()), 6)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--mode", choices=["stoch", "det"], default="stoch")
    ap.add_argument("--frames", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    _install_recorder()

    cfg = TrainConfig.resolve("economy")
    run_dir = a.run_dir or os.path.dirname(os.path.abspath(a.ckpt))
    cp = os.path.join(run_dir, "config.json")
    if os.path.exists(cp):
        d = json.load(open(cp, encoding="utf-8"))
        for k, v in d.items():
            if hasattr(cfg, k):
                try:
                    setattr(cfg, k, v)
                except Exception:
                    pass

    device = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"
    env = solo_env(cfg, a.seed)
    bdim = len(BeliefInference(opp_deck=env.deck1, n_particles=128,
                               seed=a.seed).encode(None, None))
    pol = load_checkpoint(a.ckpt, hidden_dim=cfg.hidden_dim, plan_dim=PLAN_DIM,
                          belief_dim=bdim)
    pol.to_device(device).eval()
    print(f"=== 奖励分量分解 | ckpt={a.ckpt} mode={a.mode} frames<={a.frames} "
          f"device={device} ===", flush=True)
    print(f"    γ={cfg.gamma} λ={cfg.gae_lambda} update_interval={cfg.update_interval}",
          flush=True)
    _rw_env = dict(getattr(env, "reward_weights", None) or {})
    print(f"    奖励权重（env 真值）: " + ", ".join(
        f"{k}={_rw_env[k]}" for k in sorted(_rw_env)
        if k in ("crown_weight", "crown_lose_weight", "tower_dmg_opp",
                 "tower_dmg_self", "tower_dmg_late", "tower_dmg_self_late",
                 "elixir_diff_weight", "elixir_diff_late",
                 "unit_dmg_k", "win_bonus", "lose_penalty", "draw_penalty",
                 "normalize_tower_dmg", "tower_premium_k", "king_gate")),
        flush=True)
    if not _rw_env:
        print("    ⚠️ env.reward_weights 为空 ⇒ 权重取自默认表（读数仍为引擎真值）",
              flush=True)

    opp = FollowerPolicy(hidden=cfg.hidden_dim, plan_dim=PLAN_DIM, belief_dim=bdim,
                         value_bypass=bool(pol.value_bypass),
                         value_independent=bool(pol.value_independent))
    opp.to_device(device)
    opp.load_state_dict(pol.state_dict())
    env.opponent = FollowerOpponent(
        opp, env, belief=BeliefInference(opp_deck=env.deck1, n_particles=128,
                                         seed=a.seed + 1),
        deterministic=(a.mode == "det"))
    _, _ = resolve_deck_set(getattr(cfg, "deck_set", None) or "default")

    bp = BeliefPlanner()
    belief = BeliefInference(opp_deck=env.deck1, n_particles=128, seed=a.seed)
    obs, _ = env.reset(seed=a.seed)
    belief.reset(env.deck1)
    hidden = None

    rows = {k: [] for k in ("full", "edw", "crown", "tower", "terminal", "unit",
                            "step_reward")}
    extra = {k: [] for k in ("t", "elixir0", "elixir1", "field_v0", "field_v1",
                             "edw_coef", "phi_me", "phi_opp", "n_deploy")}
    per_ep = []
    ep_rows, ep_extra = [], []
    n_frame, ep_idx, t0 = 0, 0, time.time()
    with torch.no_grad():
        while n_frame < a.frames:
            plan = bp.plan(env.battle, belief.state(), obs)
            tok = belief.encode(obs, None)
            pv = plan.to_vector()
            bundle, _, val, hidden, _ = pol.act(obs, tok, pv, env.get_action_mask,
                                                hidden=hidden,
                                                deterministic=(a.mode == "det"))
            REC.clear()
            obs, r, term, trunc, info = env.step(bundle)
            if not REC:
                raise SystemExit("[abort] 记录器没被调用 ⇒ compute_reward 绑定假设不成立")
            full = REC["full"]
            ep_rows.append({k: float(REC[k]) for k in
                            ("edw", "crown", "tower", "terminal")})
            ep_rows[-1]["full"] = float(full)
            ep_rows[-1]["step_reward"] = float(r)
            ep_rows[-1]["unit"] = float(r) - float(full)
            fv = info.get("field_v") or [0.0, 0.0]
            e0 = float(env.battle.players[0].elixir)
            e1 = float(env.battle.players[1].elixir)
            ep_extra.append({"t": float(info.get("battle_time") or 0.0),
                             "elixir0": e0, "elixir1": e1,
                             "field_v0": float(fv[0]), "field_v1": float(fv[1]),
                             "edw_coef": float(REC["edw_coef"]),
                             "phi_me": e0 + float(fv[0]),
                             "phi_opp": e1 + float(fv[1]),
                             "n_deploy": float(len(bundle.sub_actions))})
            belief.update(obs, info.get("opp_played"))
            n_frame += 1
            if term or trunc:
                for rr in ep_rows:
                    for k, v in rr.items():
                        rows[k].append(v)
                for ee in ep_extra:
                    for k, v in ee.items():
                        extra[k].append(v)
                per_ep.append(len(ep_rows))
                ep_rows, ep_extra = [], []
                obs, _ = env.reset(seed=a.seed + n_frame)
                belief.reset(env.deck1)
                hidden = None
                if a.verbose and ep_idx % 5 == 0:
                    print(f"  [rollout] ep={ep_idx} frames={n_frame} "
                          f"{time.time() - t0:.1f}s", flush=True)
                ep_idx += 1
    dt = time.time() - t0
    print(f"[rollout] frames={len(rows['full'])} episodes={len(per_ep)} "
          f"ep_len mean={np.mean(per_ep):.1f} wall={dt:.1f}s "
          f"({len(rows['full']) / max(1e-9, dt):.1f} frames/s)", flush=True)

    if len(rows["full"]) == 0:
        raise SystemExit("[abort] 没有一局完整结束 ⇒ --frames 太小（本局最长 360 帧）")
    terms = ("edw", "crown", "tower", "unit", "terminal")
    A = float(np.abs(rows["edw"]).sum())
    B = float(np.abs(rows["tower"]).sum())
    C = float(np.abs(rows["crown"]).sum())
    D = float(np.abs(rows["unit"]).sum())
    E = float(np.abs(rows["terminal"]).sum())
    S = A + B + C + D + E
    share = {"edw": A / S, "tower": B / S, "crown": C / S, "unit": D / S,
             "terminal": E / S}
    print("\n--- §1 分量占比（Σ|项| 归一）---", flush=True)
    for k in terms:
        sr = _stat(rows[k])
        print(f"  {k:9s} Σ|·|={sr['sum_abs']:10.3f}  占比={share[k]:7.2%}  "
              f"Σ(带符号)={sr['sum']:10.3f}  mean={sr['mean']:+.6f}  "
              f"max|·|={sr['max_abs']:.4f}", flush=True)
    print(f"  step_reward Σ={float(np.sum(rows['step_reward'])):.3f}  "
          f"Σ|·|={float(np.abs(rows['step_reward']).sum()):.3f}  "
          f"mean={float(np.mean(rows['step_reward'])):+.6f}", flush=True)

    print("\n--- §2 分期（edw 切价 t=120s）---", flush=True)
    t = np.asarray(extra["t"])
    ph = {}
    for lab, m in (("t<120", t < 120.0), ("t>=120", t >= 120.0)):
        if m.sum() == 0:
            continue
        sums = {k: float(np.abs(np.asarray(rows[k])[m]).sum()) for k in terms}
        tot = sum(sums.values()) or 1.0
        ph[lab] = {"n": int(m.sum()),
                   "share": {k: sums[k] / tot for k in terms},
                   "edw_coef": float(np.asarray(extra["edw_coef"])[m].mean())}
        print(f"  [{lab:6s}] n={m.sum():5d} edw_coef≈{ph[lab]['edw_coef']:.2f} | "
              + "  ".join(f"{k}={sums[k] / tot:.1%}" for k in terms), flush=True)

    print("\n--- §3 Φ 轨迹与法术账（描述）---", flush=True)
    phi_me = np.asarray(extra["phi_me"])
    phi_opp = np.asarray(extra["phi_opp"])
    el0 = np.asarray(extra["elixir0"])
    fv0 = np.asarray(extra["field_v0"])
    fv1 = np.asarray(extra["field_v1"])
    print(f"  elixir0 mean={el0.mean():.3f} median={np.median(el0):.3f} | "
          f"field_v0 mean={fv0.mean():.3f} | Φ_me mean={phi_me.mean():.3f}", flush=True)
    print(f"  field_v1 mean={fv1.mean():.3f} | Φ_opp mean={phi_opp.mean():.3f} | "
          f"ΔΦ=(Φ_me−Φ_opp) mean={(phi_me - phi_opp).mean():+.3f}", flush=True)
    dep = np.asarray(extra["n_deploy"])
    m_sp = dep >= 1
    edw_arr = np.asarray(rows["edw"])
    if m_sp.any() and (~m_sp).any():
        print(f"  施法/下牌帧 n={int(m_sp.sum())}：edw 项 mean={edw_arr[m_sp].mean():+.6f}"
              f" | 空帧 n={int((~m_sp).sum())}：edw 项 mean={edw_arr[~m_sp].mean():+.6f}",
              flush=True)

    # ---- 判据 ----
    if share["edw"] >= 0.50 and share["tower"] >= 0.50:
        verdict = "W1+W2"
    elif share["edw"] >= 0.50:
        verdict = "W1_REWARD_DOMINATED_BY_RESOURCE_ACCOUNT"
    elif share["tower"] >= 0.50:
        verdict = "W2_REWARD_DOMINATED_BY_TOWER_DAMAGE"
    else:
        verdict = "W4_NO_DOMINANT_TERM"
    w3 = share["terminal"] < 0.10
    print("\n=== 判据（照预注册 §3 读）===", flush=True)
    print(f"  A/S(edw)={share['edw']:.4f}  B/S(tower)={share['tower']:.4f}  "
          f"C/S(crown)={share['crown']:.4f}  D/S(unit)={share['unit']:.4f}  "
          f"E/S(terminal)={share['terminal']:.4f}", flush=True)
    print(f"  W3 终局 <10% ? {w3}（E/S={share['terminal']:.2%}）", flush=True)
    print(f"  VERDICT = {verdict}" + ("  + W3" if w3 else ""), flush=True)

    res = {"ckpt": a.ckpt, "mode": a.mode, "seed": a.seed,
           "frames": int(len(rows["full"])), "episodes": len(per_ep),
           "device": device,
           "weights_env": {k: v for k, v in
                           dict(getattr(env, "reward_weights", None) or {}).items()},
           "weights_cfg_reward": dict(cfg.reward),
           "sum_abs": {"edw": A, "tower": B, "crown": C, "unit": D, "terminal": E},
           "share": share, "verdict": verdict, "w3_terminal_lt_10pct": bool(w3),
           "stats": {k: _stat(rows[k]) for k in list(terms) + ["step_reward", "full"]},
           "phases": ph,
           "phi": {"phi_me_mean": float(phi_me.mean()),
                   "phi_opp_mean": float(phi_opp.mean()),
                   "dphi_mean": float((phi_me - phi_opp).mean()),
                   "elixir0_mean": float(el0.mean()),
                   "field_v0_mean": float(fv0.mean()),
                   "field_v1_mean": float(fv1.mean())},
           "n_prereg": "docs/reward_composition_prereg_2026-09-14.md"}
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        print(f"[out] {a.out}", flush=True)
    return res


if __name__ == "__main__":
    main()
