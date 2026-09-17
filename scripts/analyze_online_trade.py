# -*- coding: utf-8 -*-
"""用**在线口径**量「局面圣水交换」（S2 第四轮；预注册 §11.9.4 的放行前置）。

为什么要单独一个脚本：**离线的局面切分与在线不是同一个量**（同一局 67 vs 2 个窗口，
见 `docs/engagement_trade_online_2026-09-18.md` §3）。所以"`p4b` 是否值得接线"这件事
**必须用在线口径重测**。做法：让评估局以 **measure-only**（跑监视器、权重 0、行为逐位不变）
把每个决策帧结算掉的局面写进录像帧的 `et` 键，再在这里做**同批配对**相关分析。

`et` = `[phi_part, tau, score, n_windows, tau0, tau1]`（**p0 视角**）。
本脚本把 main 视角还原出来（联赛每场打两局、双方互换 ⇒ 必须按 `meta.side0` 翻面），
再重组出两个变体：
    `trade_none  = phi_main`               （不加 `tower_term`）
    `trade_p4b   = phi_main + tau_main`    （加 P4b）
`τ` 是**零和且两侧对称**的 `(τ₀−τ₁)/2`（第四轮修掉的在线同型 bug，见 §7.3 of s2_instrument）。

用法（cwd = src/clasher_new）：
    PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe ../../scripts/analyze_online_trade.py \
        --replays runs/et_measure/replays --json ../../docs/online_trade_measure.json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

import offline_engagement_trade as O            # noqa: E402  （复用 _spearman/_stats/load_replay）

#: 结局量（与离线仪器同定义，避免两套口径）
OUTCOMES = ("crown", "tower_hp", "win")


def _game_rows(game):
    """把一局压成 {变体: trade_main} + 结局量。返回 None 表示缺 `et`。"""
    frames = game.get("frames") or []
    if not frames or frames[0].get("et") is None:
        return None
    meta = game.get("meta") or {}
    main_p0 = str(meta.get("side0", "main")) == "main"
    s = [0.0] * 6
    for f in frames:
        e = f["et"]
        for i in range(6):
            s[i] += float(e[i])
    phi, tau, score, n, tau0, tau1 = s
    sign = 1.0 if main_p0 else -1.0          # trade 反对称 ⇒ main 视角 = ±p0 视角
    last = frames[-1]
    cd = (int(last["crown1"]) - int(last["crown0"])) * sign
    hp = (sum(float(x) for x in last["towers0"])
          - sum(float(x) for x in last["towers1"])) * sign
    w = game.get("winner")
    win = None if w is None else (1 if int(w) == (0 if main_p0 else 1) else 0)
    return {
        "n_frames": len(frames), "main_p0": main_p0,
        "phi": sign * phi, "tau": sign * tau, "score": sign * score,
        "tau0": sign * tau0, "tau1": sign * tau1,
        "n_windows": n,
        "trade_none": sign * phi,
        "trade_p4b": sign * (phi + tau),
        # ★ 代理**自身的**信号（不加 phi）：若这一列也是 ~0，说明 `τ` 本身与结果无关，
        # 而不是"被 phi 淹没了"—— 这两件事的处置完全不同。
        "trade_tau_only": sign * tau,
        "crown": cd, "tower_hp": hp, "win": win,
    }


#: ★ 权重扫描：`trade(λ) = φ + λ·τ`。因为录像里 **φ 与 τ 是分开记的**，
#: 这一步**不需要重跑**。它回答一个关键分叉：
#: * 若 λ 调大后 Δρ 仍≈0 ⇒ **冗余**（τ 的信息已被 φ 包含）⇒ 死路；
#: * 若 Δρ 随 λ 单调变好 ⇒ **太弱**（只是量级只有 φ 的 3~7%）⇒ 出路是**权重/尺度**，不是换代理。
LAMBDAS = (1.0, 3.0, 10.0, 30.0)


def _boot_delta_rho(rows, lam, outcome, n_boot=400, seed=7):
    """配对 bootstrap：对**局**重采样，给 `Δρ(λ) = ρ(φ+λτ) − ρ(φ)` 一个标准误。

    为什么必须配对：Δρ 是两个**在同一批局上**算出的 ρ 之差，独立假设会高估其方差。
    用同一份重采样同时算两个 ρ 再相减 ⇒ 正确反映它们的相关性。
    """
    import random
    rnd = random.Random(seed)
    n = len(rows)
    if n < 8:
        return None
    ds = []
    for _ in range(n_boot):
        idx = [rnd.randrange(n) for _ in range(n)]
        rs = [rows[i] for i in idx]
        xs0 = [r["phi"] for r in rs]
        xs1 = [r["phi"] + lam * r["tau"] for r in rs]
        ys = [(1.0 if r[outcome] == 1 else 0.0) if outcome == "win" else float(r[outcome])
              for r in rs]
        r0, r1 = O._spearman(xs0, ys), O._spearman(xs1, ys)
        if r0 is not None and r1 is not None:
            ds.append(r1 - r0)
    if len(ds) < 8:
        return None
    ds.sort()
    m = sum(ds) / len(ds)
    sd = (sum((d - m) ** 2 for d in ds) / (len(ds) - 1)) ** 0.5
    return {"mean": m, "sd": sd, "lo": ds[int(0.025 * len(ds))],
            "hi": ds[int(0.975 * len(ds)) - 1], "n_boot": len(ds)}


def _rho(variant, outcome, rows):
    xs = [r[variant] for r in rows]
    ys = [(1.0 if r[outcome] == 1 else 0.0) if outcome == "win" else float(r[outcome])
          for r in rows]
    return O._spearman(xs, ys)


def analyze(files, theta=1.0):
    batches = []
    for p in files:
        schema, games = O.load_replay(p)
        rows = []
        for g in games:
            r = _game_rows(g)
            if r:
                rows.append(r)
        if not rows:
            batches.append({"file": os.path.basename(p), "schema": schema,
                            "n_games": 0, "note": "该文件没有 et 字段（不是 measure-only 跑出来的）"})
            continue
        b = {"file": os.path.basename(p), "schema": schema, "n_games": len(rows),
             "frames": sum(r["n_frames"] for r in rows),
             "windows_per_game": statistics.mean(r["n_windows"] for r in rows),
             "tau_nonzero_game_share": statistics.mean(1.0 if abs(r["tau"]) > 1e-9 else 0.0
                                                      for r in rows),
             "score_gt0_game_share": {
                 v: statistics.mean(1.0 if (r[v] - theta) > 0 else 0.0 for r in rows)
                 for v in ("trade_none", "trade_p4b")},
             "mean": {v: statistics.mean(r[v] for r in rows)
                      for v in ("trade_none", "trade_p4b", "tau")},
             "sd": {v: statistics.stdev([r[v] for r in rows]) if len(rows) > 2 else 0.0
                    for v in ("trade_none", "trade_p4b")},
             "tau_mean_abs": statistics.mean(abs(r["tau"]) for r in rows),
             "phi_mean_abs": statistics.mean(abs(r["phi"]) for r in rows)}
        for v in ("trade_none", "trade_p4b", "trade_tau_only"):
            for oc in OUTCOMES:
                b["rho_%s_%s" % (v, oc)] = _rho(v, oc, rows)
        # 冗余检查：τ 与 φ 的相关（高 ⇒ τ 的信息已被 φ 包含）
        xs = [r["phi"] for r in rows]
        ys = [r["tau"] for r in rows]
        b["rho_tau_phi"] = O._spearman(xs, ys)
        # 权重扫描
        b["lambda_rho"] = {}
        for lam in LAMBDAS:
            v = {}
            for oc in OUTCOMES:
                xs = [r["phi"] + lam * r["tau"] for r in rows]
                yv = [(1.0 if r[oc] == 1 else 0.0) if oc == "win" else float(r[oc])
                      for r in rows]
                v[oc] = O._spearman(xs, yv)
            b["lambda_rho"][str(lam)] = v
            b.setdefault("lambda_boot", {})[str(lam)] = {
                oc: _boot_delta_rho(rows, lam, oc) for oc in ("tower_hp", "win")}
        for oc in OUTCOMES:
            a, c = b.get("rho_trade_p4b_%s" % oc), b.get("rho_trade_none_%s" % oc)
            b["d_rho_p4b_minus_none_%s" % oc] = (None if (a is None or c is None)
                                                 else a - c)
        batches.append(b)
    return batches


def summarize(batches):
    """跨批配对汇总（【R16】：同批配对，而不是单批显著性）。"""
    ok = [b for b in batches if b.get("n_games")]
    out = {"n_batches": len(ok), "n_games": sum(b["n_games"] for b in ok)}
    for oc in OUTCOMES:
        ds = [b.get("d_rho_p4b_minus_none_%s" % oc) for b in ok]
        ds = [d for d in ds if d is not None]
        rn = [b.get("rho_trade_none_%s" % oc) for b in ok]
        rn = [x for x in rn if x is not None]
        rp = [b.get("rho_trade_p4b_%s" % oc) for b in ok]
        rp = [x for x in rp if x is not None]
        out["paired_" + oc] = {
            "n": len(ds), "pos": sum(1 for d in ds if d > 0),
            "mean_d": (statistics.mean(ds) if ds else None),
            "sd_d": (statistics.stdev(ds) if len(ds) > 1 else 0.0),
            "mean_rho_none": (statistics.mean(rn) if rn else None),
            "mean_rho_p4b": (statistics.mean(rp) if rp else None),
            "vals": [round(d, 3) for d in ds]}
    return out


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="用在线口径量局面圣水交换（measure-only 录像）")
    ap.add_argument("--replays", nargs="+", required=True)
    ap.add_argument("--theta", type=float, default=1.0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    files = []
    for p in a.replays:
        if os.path.isdir(p):
            files += [os.path.join(p, n) for n in sorted(os.listdir(p)) if n.endswith(".pkl")]
        else:
            files.append(p)
    batches = analyze(files, theta=a.theta)
    s = summarize(batches)

    print("=" * 92)
    print("在线口径 · 局面圣水交换（measure-only 录像；口径见 docs/engagement_trade_online_2026-09-18.md）")
    print("=" * 92)
    print("%-22s %6s %8s %9s %9s %9s %9s %9s %10s"
          % ("批次", "局数", "窗口/局", "|τ|/|φ|", "ρn(塔血)", "ρp(塔血)", "ρn(胜)",
             "ρp(胜)", "Δρp-n(塔血)"))
    for b in batches:
        if not b.get("n_games"):
            print("%-22s %s" % (b["file"], b.get("note")))
            continue
        print("%-22s %6d %8.1f %9s %9s %9s %9s %9s %10s"
              % (b["file"], b["n_games"], b["windows_per_game"],
                 _f(b["tau_mean_abs"] / b["phi_mean_abs"]) if b["phi_mean_abs"] else "n/a",
                 _r(b.get("rho_trade_none_tower_hp")), _r(b.get("rho_trade_p4b_tower_hp")),
                 _r(b.get("rho_trade_none_win")), _r(b.get("rho_trade_p4b_win")),
                 _r(b.get("d_rho_p4b_minus_none_tower_hp"))))
    # ★ 代理自身的信号（不看配对、只看 τ 自己与结果的关系）
    print("\n★ `τ` **自身**的信号（不是配对差；若这里也是 0 ⇒ 代理本身无信息）")
    for oc in OUTCOMES:
        xs = [b.get("rho_trade_tau_only_%s" % oc) for b in batches
              if b.get("n_games") and b.get("rho_trade_tau_only_%s" % oc) is not None]
        if xs:
            print("  %-9s ρ(τ, ·) 逐批 %s  均值 %s（|ρ|max %.3f）"
                  % (oc, [round(x, 3) for x in xs], _r(statistics.mean(xs)),
                     max(abs(x) for x in xs)))
    # ★ 权重扫描汇总（相对 λ=0 即 none）
    print("\n★ 权重扫描 `trade(λ) = φ + λ·τ`：Δρ 相对 λ=0（none）")
    print("   %-8s %9s %9s %9s   %s" % ("λ", "Δρ(塔血)", "Δρ(胜)", "Δρ(冠差)", "逐批 Δρ(塔血)"))
    for lam in LAMBDAS:
        row = {}
        for oc in OUTCOMES:
            ds = []
            for b in batches:
                if not b.get("n_games"):
                    continue
                _v = b["lambda_rho"][str(lam)][oc]
                _c = b.get("rho_trade_none_%s" % oc)
                if _v is not None and _c is not None:
                    ds.append(_v - _c)
            row[oc] = ds
        print("   %-8s %9s %9s %9s   %s"
              % (lam, _r(statistics.mean(row["tower_hp"])) if row["tower_hp"] else "n/a",
                 _r(statistics.mean(row["win"])) if row["win"] else "n/a",
                 _r(statistics.mean(row["crown"])) if row["crown"] else "n/a",
                 [round(d, 3) for d in row["tower_hp"]]))
    print("\n★ 配对 bootstrap（对**局**重采样 400 次）——Δρ(λ) 的 95% 区间")
    for lam in LAMBDAS:
        line = []
        for oc in ("tower_hp", "win"):
            per = [b["lambda_boot"][str(lam)][oc] for b in batches
                   if b.get("n_games") and b.get("lambda_boot", {}).get(str(lam), {}).get(oc)]
            if per:
                lo = min(x["lo"] for x in per)
                hi = max(x["hi"] for x in per)
                line.append("%s [%+.3f, %+.3f]" % (oc, lo, hi))
        print("   λ=%-5s %s" % (lam, "   ".join(line)))
    rtp = [b.get("rho_tau_phi") for b in batches if b.get("n_games")]
    rtp = [x for x in rtp if x is not None]
    print("   ρ(τ, φ) 逐批 %s  均值 %s  ⇒ %s"
          % ([round(x, 3) for x in rtp], _r(statistics.mean(rtp)),
             "**冗余**（τ 的信息已在 φ 里）" if rtp and statistics.mean(rtp) > 0.5
             else "**不冗余**（τ 有独立信息）" if rtp else "n/a"))

    print("\n★ 跨批**配对**汇总（同批内 p4b − none；【R16】）")
    for oc in OUTCOMES:
        p = s["paired_" + oc]
        print("  %-9s n=%d  同向(正) %d/%d   Δρ 均值 %s（sd %s）  |  ρnone %s  ρp4b %s"
              % (oc, p["n"], p["pos"], p["n"], _r(p["mean_d"]), _f(p["sd_d"]),
                 _r(p["mean_rho_none"]), _r(p["mean_rho_p4b"])))
        print("            逐批 Δρ: %s" % p["vals"])
    print("\n总局数 %d（%d 个批次）；θ=%.2f" % (s["n_games"], s["n_batches"], a.theta))
    if a.json:
        d = os.path.dirname(os.path.abspath(a.json))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"batches": batches, "summary": s, "theta": a.theta}, f,
                      ensure_ascii=False, indent=1)
        print("[JSON] %s" % a.json)
    return 0


def _r(x):
    return "n/a" if x is None else "%+.3f" % x


def _f(x):
    return "n/a" if x is None else "%.3f" % x


def _p(x):
    return "n/a" if x is None else "%.1f%%" % (100.0 * x)


if __name__ == "__main__":
    raise SystemExit(main())
