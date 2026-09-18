"""`et_solo100k` 判读执行器：一条命令跑完 §11.13.2 的四层。

用法（在**仓库根**下）：
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/et_solo100k_readout.py \
      --log-a docs/train_et_solo100k.log --log-b docs/train_ctrl100k.log \
      --run-a src/clasher_new/runs/et_solo100k --run-b src/clasher_new/runs/et_ctrl100k \
      --out-dir docs/_readout_et_solo100k --markdown docs/_readout_et_solo100k.md

设计原则（【R3】【R4】【R17】）
---------------------------
1. **只编排 + 复算，不重复实现仪器**：机制层直接 `import judge_critic_inertia` 复用其
   `parse_log` / `within_run_baseline`（含 `mad`），保证与 §11.13.2 写死的 within-run 口径**同一个实现**；
   行为层/门禁层**调用既有仪器脚本**并把**原始 stdout 落盘留证**（不另算一套口径，避免 R17 的"两套约定"）。
2. **缺数据就大声失败**：任何一层拿不到数就标 `MISSING`/`PARSE_FAIL` 并让进程退出码非 0，
   **绝不静默填 0**。
3. **不做判决**：本脚本只出**读数与程序性谓词**（如 §11.13.4-3 的"不可分辨"判定），
   判决文字由 `docs/et_solo100k_2026-09-18.md` 人工写（§11.13.3：缺阳性对照 ⇒ 不出 §5 判决）。

⚠️ 本脚本用 **Windows python** 运行：`--out-dir` / `--markdown` 请用**仓库内相对路径**。
   WSL 的 `/tmp/...` 会被 Windows 解析成 `C:\\tmp\\...`（2026-09-18 实测踩到，raw 文件"消失"）。
   `--run-a/--run-b` 用仓库根相对路径即可（脚本内部会 `abspath`）。
"""

import argparse
import json
import os
import re
import subprocess
import sys

try:  # GBK 控制台兜底（同仓库其它脚本纪律）
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SRC = os.path.join(_ROOT, "src", "clasher_new")
sys.path.insert(0, _HERE)  # 以便 import judge_critic_inertia

import judge_critic_inertia as jci  # noqa: E402

_METRICS = ["cos", "resid_norm", "corr", "lvl", "gap", "gr"]
_METRIC_NAME = {"cos": "grad_cos", "resid_norm": "resid_norm", "corr": "corr",
                "lvl": "level_shift", "gap": "level_gap", "gr": "gnorm_ratio"}


def _fmt(x, nd=5):
    return "None" if x is None else f"{x:.{nd}f}"


def run_instrument(cmd, out_path):
    """跑一个既有仪器，stdout+stderr 原文落盘，返回 (text, returncode)。"""
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    r = subprocess.run(cmd, cwd=_SRC, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env)
    text = (r.stdout or "") + (r.stderr or "")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"$ {' '.join(cmd)}\n\n{text}")
    return text, r.returncode


# ---------------------------------------------------------------- 层 1：机制（主）
def layer_mechanism(log_a, log_b, n_base, k):
    rows_a = jci.parse_log(log_a) if os.path.exists(log_a) else []
    rows_b = jci.parse_log(log_b) if os.path.exists(log_b) else []
    out = {
        "log_a": log_a, "log_b": log_b,
        "n_points_a": len(rows_a), "n_points_b": len(rows_b),
        "n_base": n_base, "k": k,
        "arms": {}, "compare": {}, "indistinguishable_metrics": [],
    }
    if not rows_a or not rows_b:
        out["status"] = "MISSING"
        out["note"] = ("缺 advinert 诊断点 ⇒ 机制层不可算。"
                       "检查两臂是否都带 --adv-inert-probe（§11.13.6）。")
        return out
    wa = jci.within_run_baseline(rows_a, n_base=n_base, k=k)
    wb = jci.within_run_baseline(rows_b, n_base=n_base, k=k)
    out["status"] = "OK"
    for tag, w in (("A_et", wa), ("B_ctrl", wb)):
        out["arms"][tag] = {
            "n_base": w["n_base"], "n_later": w["n_later"],
            "metrics": {m: {
                "base_median": w["metrics"][m]["base_median"],
                "base_mad": w["metrics"][m]["base_mad"],
                "lo": w["metrics"][m]["lo"], "hi": w["metrics"][m]["hi"],
                "degenerate": w["metrics"][m]["degenerate"],
                "later_median": w["metrics"][m]["later_median"],
                "later_inside": w["metrics"][m]["later_inside"],
                "later_n": w["metrics"][m]["later_n"],
            } for m in _METRICS}}
    # 跨臂：§11.13.4-3「差异小于两者自身的窗口散布」⇒ 不可分辨
    # 「窗口散布」的**操作化定义**（写于两臂跑完之前，见预注册 §11.13.4 注）：
    #   spread = max(MAD_A, MAD_B) —— MAD 就是本实验 §11.13.2 指定的离散度统计量。
    for m in _METRICS:
        sa, sb = wa["metrics"][m], wb["metrics"][m]
        la, lb = sa["later_median"], sb["later_median"]
        ma, mb = sa["base_mad"], sb["base_mad"]
        if la is None or lb is None or ma is None or mb is None:
            out["compare"][m] = {"status": "MISSING"}
            continue
        diff = abs(la - lb)
        spread = max(ma, mb)
        deg = bool(sa["degenerate"] or sb["degenerate"])
        indist = (diff < spread)
        out["compare"][m] = {
            "name": _METRIC_NAME[m], "later_median_A": la, "later_median_B": lb,
            "diff": diff, "spread_max_mad": spread, "degenerate": deg,
            "indistinguishable": indist,
        }
        if indist:
            out["indistinguishable_metrics"].append(_METRIC_NAME[m])
    out["all_indistinguishable"] = bool(
        out["indistinguishable_metrics"]
        and len(out["indistinguishable_metrics"]) == len(_METRICS))
    return out


# ------------------------------------------------- 层 1b：EV（§11.13.2 机制层的另两项）
# §11.13.2 把「批内 EV（**更新前池化**）」与「`EVb` 形状」也算在机制层里。
# `EVb` 在 `[solo step N]` 行上（**每个 update 都打印**，不随 diagnose_every 稀疏化）；
# 「池化 EV」在 `eval@N` 行上（eval@0 为 None，第一个真实评估点起才有值）。
_RE_STEP = re.compile(
    r"^\[solo step (?P<step>\d+)\].*?EVb=(?P<evb>[-+0-9.eE]+)"
    r"(?:\s+EVin=(?P<evin>[-+0-9.eE]+))?")
_RE_EVAL = re.compile(
    r"eval@(?P<step>\d+):\s*胜率\s*(?P<wr>[\d.]+)±(?P<ci>[\d.]+)\s*"
    r"\((?P<w>\d+)W/(?P<l>\d+)L/(?P<d>\d+)D,\s*(?P<n>\d+)局\)\s*"
    r"mean_reward=(?P<mr>[-+0-9.eE]+)\s+EV=(?P<ev>\S+)")


def parse_step_ev(path):
    """返回 [(step, evb, evin_or_None), ...]；`EVb` 是**批内、更新前**口径（§11.13.2）。"""
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            m = _RE_STEP.match(ln)
            if m:
                evin = m.group("evin")
                out.append((int(m.group("step")), float(m.group("evb")),
                            (float(evin) if evin is not None else None)))
    return out


def parse_eval_ev(path):
    """返回 [(step, winrate, n_games, mean_reward, pooled_ev_or_None), ...]。"""
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            m = _RE_EVAL.search(ln)
            if not m:
                continue
            ev = m.group("ev")
            for junk in ("（池化）", "(池化)"):
                ev = ev.replace(junk, "")
            out.append((int(m.group("step")), float(m.group("wr")), int(m.group("n")),
                        float(m.group("mr")), (None if ev.strip() == "None" else float(ev))))
    return out


def _within_series(vals, n_base, k):
    """给一条标量序列套 §11.13.2/§11.13.4 注 的 within-run 口径（中位 ± k×原始 MAD）。"""
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"status": "MISSING"}
    nb = min(int(n_base), len(vals))
    base, later = vals[:nb], vals[nb:]
    med = jci._median(base)
    m = jci.mad(base, med)
    lo, hi = med - k * (m or 0.0), med + k * (m or 0.0)
    inside = sum(1 for x in later if lo <= x <= hi) if later else None
    return {"status": "OK", "n_base": nb, "n_later": len(later),
            "base_median": med, "base_mad": m, "lo": lo, "hi": hi,
            "degenerate": (m == 0.0), "later_median": (jci._median(later) if later else None),
            "later_inside": inside}


def layer_ev(log_a, log_b, n_base, k):
    res = {"status": "OK", "arms": {}, "compare": {}, "indistinguishable_metrics": []}
    for tag, path in (("A_et", log_a), ("B_ctrl", log_b)):
        evb_rows = parse_step_ev(path)
        res["arms"][tag] = {
            "n_evb_points": len(evb_rows),
            "evb": _within_series([r[1] for r in evb_rows], n_base, k),
            "evin_base_median": jci._median([r[2] for r in evb_rows[:n_base] if r[2] is not None]),
            "pooled_ev_series": [(s, ev) for s, _wr, _n, _mr, ev in parse_eval_ev(path)],
        }
    a, b = res["arms"]["A_et"], res["arms"]["B_ctrl"]
    for key, name in (("evb", "EVb"),):
        sa, sb = a[key], b[key]
        if sa.get("status") != "OK" or sb.get("status") != "OK":
            res["compare"][name] = {"status": "MISSING"}
            continue
        la, lb = sa["later_median"], sb["later_median"]
        if la is None or lb is None:
            res["compare"][name] = {"status": "MISSING"}
            continue
        diff = abs(la - lb)
        spread = max(sa["base_mad"] or 0.0, sb["base_mad"] or 0.0)
        ind = diff < spread
        res["compare"][name] = {
            "name": name, "later_median_A": la, "later_median_B": lb,
            "diff": diff, "spread_max_mad": spread,
            "degenerate": bool(sa["degenerate"] or sb["degenerate"]),
            "indistinguishable": ind}
        if ind:
            res["indistinguishable_metrics"].append(name)
    if a["n_evb_points"] == 0 or b["n_evb_points"] == 0:
        res["status"] = "PARTIAL"
    return res


def layer_realization(run_a, run_b, out_dir):
    """§11.13.2 门禁行的**主指标**：优势兑现率（N=40 帧 = 20 s，θ=1.0）。

    ⚠️ **仪器命名不一致（§11.13.10）**：§11.13.2 把门禁的指标写成「优势兑现率」、仪器写成
    `analyze_online_trade.py`；但**兑现率只在 `offline_engagement_trade.py` 里实现**
    （`realization_gate`，`REALIZE_N_FRAMES = 40` = 20 s）。⇒ 本层调**后者**取兑现率，
    并另由 `layer_gate` 调前者取配对 Δρ；**两者都报**，不各取所需地拼一个口径（【R17】）。
    """
    res = {"status": "OK", "arms": {}}
    for tag, run in (("A_et", run_a), ("B_ctrl", run_b)):
        rep = os.path.abspath(os.path.join(run, "replays"))
        if not os.path.isdir(rep) or not os.listdir(rep):
            res["arms"][tag] = {"status": "MISSING", "note": f"无录像 {rep}"}
            continue
        text, rc = run_instrument(
            [sys.executable, os.path.join(_HERE, "offline_engagement_trade.py"),
             "--replays", rep, "--phi-mode", "global", "--tower-mode", "p4b"],
            os.path.join(out_dir, f"realization_{tag}.txt"))
        m_rate = _RE_REALIZE.search(text)
        m_elig = _RE_ELIGIBLE.search(text)
        arm = {"status": "OK" if m_rate else "PARSE_FAIL",
               "raw": f"realization_{tag}.txt", "rc": rc,
               "realize_rate_pct": float(m_rate.group(1)) if m_rate else None,
               "spend_ok_pct": float(m_rate.group(2)) if m_rate else None,
               "tower_dmg_pct": float(m_rate.group(3)) if m_rate else None,
               "eligible_windows": int(m_elig.group(1)) if m_elig else None,
               "eligible_per_game": float(m_elig.group(2)) if m_elig else None}
        res["arms"][tag] = arm
    if any(v.get("status") != "OK" for v in res["arms"].values()):
        res["status"] = "PARTIAL"
    a, b = res["arms"].get("A_et", {}), res["arms"].get("B_ctrl", {})
    res["paired"] = _delta(a.get("realize_rate_pct"), b.get("realize_rate_pct"))
    # 逐批（每个评估点单独跑一次仪器）⇒ 可做 A/B **按批次配对**，为失败分支 2 提供
    # 「方向稳不稳」的证据（不只是两个聚合数字相减）。每个 pkl 约 1.5 s。
    per_dir = os.path.join(out_dir, "realization_perbatch")
    os.makedirs(per_dir, exist_ok=True)
    per = {}
    for tag, run in (("A_et", run_a), ("B_ctrl", run_b)):
        rep = os.path.abspath(os.path.join(run, "replays"))
        rows = []
        if os.path.isdir(rep):
            for fn in sorted(os.listdir(rep)):
                if not fn.endswith(".pkl"):
                    continue
                t_i, rc_i = run_instrument(
                    [sys.executable, os.path.join(_HERE, "offline_engagement_trade.py"),
                     "--replays", os.path.join(rep, fn),
                     "--phi-mode", "global", "--tower-mode", "p4b"],
                    os.path.join(per_dir, f"{tag}_{fn}.txt"))
                mr = _RE_REALIZE.search(t_i)
                me = _RE_ELIGIBLE.search(t_i)
                rows.append({"batch": fn, "rc": rc_i,
                             "realize_rate_pct": float(mr.group(1)) if mr else None,
                             "spend_ok_pct": float(mr.group(2)) if mr else None,
                             "tower_dmg_pct": float(mr.group(3)) if mr else None,
                             "eligible_windows": int(me.group(1)) if me else None})
        per[tag] = rows
    by = {t: {r["batch"]: r for r in per[t]} for t in per}
    paired_pb = []
    for name in sorted(set(by.get("A_et", {})) & set(by.get("B_ctrl", {}))):
        ra, rb = by["A_et"][name], by["B_ctrl"][name]
        d = (None if (ra["realize_rate_pct"] is None or rb["realize_rate_pct"] is None)
             else ra["realize_rate_pct"] - rb["realize_rate_pct"])
        paired_pb.append({"batch": name, "A": ra["realize_rate_pct"], "B": rb["realize_rate_pct"],
                          "A_minus_B": d,
                          "eligible_A": ra["eligible_windows"], "eligible_B": rb["eligible_windows"]})
    res["per_batch"] = paired_pb
    ds = [x["A_minus_B"] for x in paired_pb if x["A_minus_B"] is not None]
    res["per_batch_summary"] = {
        "n": len(ds),
        "n_pos": sum(1 for d in ds if d > 0), "n_neg": sum(1 for d in ds if d < 0),
        "n_zero": sum(1 for d in ds if d == 0),
        "mean_A_minus_B": (sum(ds) / len(ds)) if ds else None}
    return res


# ------------------------------------------------- 层 2/3：调用既有仪器 + 解析
_RE_REALIZE = re.compile(
    r"兑现率\s*=\s*([\d.]+)%（花费达标\s*([\d.]+)%\s*／\s*打出塔伤\s*([\d.]+)%）")
_RE_ELIGIBLE = re.compile(r"合格窗口\s*(\d+)\s*个（([\d.]+)/局）")
_RE_POOL_FULL = re.compile(r"\[POOLED\]\s*全帧 elixir0\s+n=(\d+).*?≥6=([\d.]+)%")
_RE_PRE = re.compile(r"\[POOLED\]\s*pre \(=post\+费\)\s+n=(\d+).*?median=([\d.]+)")
_RE_NEVER = re.compile(r"牌组里从未打出的卡:\s*\[(.*?)\]")
_RE_GATE_BATCH = re.compile(r"^(\S+\.pkl)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([+-][\d.]+)\s+([+-][\d.]+)\s+([+-][\d.]+)\s+([+-][\d.]+)\s+([+-][\d.]+)",
                            re.M)
_RE_TAU_PHI = re.compile(r"ρ\(τ, φ\)\s*逐批.*?均值\s*([+-][\d.]+)")


def layer_behaviour(run_a, run_b, out_dir):
    res = {"status": "OK", "arms": {}}
    for tag, run in (("A_et", run_a), ("B_ctrl", run_b)):
        # ⚠️ 必须 abspath：仪器以 cwd=src/clasher_new 启动，仓库根相对路径会解析错位
        # ⇒ 仪器报 "No such file" / 解析失败（2026-09-18 实测踩到）。
        rep = os.path.abspath(os.path.join(run, "replays"))
        if not os.path.isdir(rep):
            res["arms"][tag] = {"status": "MISSING", "note": f"无录像目录 {rep}"}
            continue
        text, rc = run_instrument(
            [sys.executable, os.path.join(_HERE, "forensics_card_usage.py"),
             "--replays", rep],
            os.path.join(out_dir, f"forensics_{tag}.txt"))
        m6 = _RE_POOL_FULL.search(text)
        mpre = _RE_PRE.search(text)
        mnever = _RE_NEVER.search(text)
        xbow_never = None
        if mnever:
            xbow_never = "Xbow" in [s.strip().strip("'\"") for s in mnever.group(1).split(",")]
        arm = {"status": "OK" if (m6 and mpre) else "PARSE_FAIL",
               "raw": f"forensics_{tag}.txt", "rc": rc}
        arm["elixir_ge6_pct"] = float(m6.group(2)) if m6 else None
        arm["full_frames_n"] = int(m6.group(1)) if m6 else None
        arm["pre_deploy_median"] = float(mpre.group(2)) if mpre else None
        arm["pre_deploy_n"] = int(mpre.group(1)) if mpre else None
        arm["xbow_never_played"] = xbow_never
        arm["xbow_play_rate"] = 0 if xbow_never else None  # 未打出 ⇒ 0；否则需查表
        res["arms"][tag] = arm
    if any(v.get("status") != "OK" for v in res["arms"].values()):
        res["status"] = "PARTIAL"
    # 配对 A − B（§11.13.2 的对照 = `B_ctrl` 同批；【R17】两侧同口径同一实现）
    a, b = res["arms"].get("A_et", {}), res["arms"].get("B_ctrl", {})
    res["paired"] = {
        "elixir_ge6_pct": _delta(a.get("elixir_ge6_pct"), b.get("elixir_ge6_pct")),
        "pre_deploy_median": _delta(a.get("pre_deploy_median"), b.get("pre_deploy_median")),
        "xbow_play_rate": _delta(a.get("xbow_play_rate"), b.get("xbow_play_rate")),
    }
    return res


def _delta(x, y):
    if x is None or y is None:
        return {"A": x, "B": y, "A_minus_B": None}
    return {"A": x, "B": y, "A_minus_B": x - y}


def layer_gate(run_a, run_b, out_dir):
    res = {"status": "OK", "arms": {}}
    for tag, run in (("A_et", run_a), ("B_ctrl", run_b)):
        rep = os.path.abspath(os.path.join(run, "replays"))  # 同 layer_behaviour 的理由
        if not os.path.isdir(rep) or not os.listdir(rep):
            res["arms"][tag] = {"status": "MISSING", "note": f"无录像 {rep}"}
            continue
        text, rc = run_instrument(
            [sys.executable, os.path.join(_HERE, "analyze_online_trade.py"),
             "--replays", rep],
            os.path.join(out_dir, f"gate_{tag}.txt"))
        arms_pairs = []
        for m in _RE_GATE_BATCH.finditer(text):
            arms_pairs.append({"batch": m.group(1), "games": int(m.group(2)),
                               "windows_per_game": float(m.group(3)),
                               "tau_phi_ratio": float(m.group(4)),
                               "rho_none_tower": float(m.group(5)),
                               "rho_p4b_tower": float(m.group(6)),
                               "rho_none_win": float(m.group(7)),
                               "rho_p4b_win": float(m.group(8)),
                               "d_rho_tower": float(m.group(9))})
        m_tp = _RE_TAU_PHI.search(text)
        res["arms"][tag] = {
            "status": "OK" if arms_pairs else "PARSE_FAIL",
            "raw": f"gate_{tag}.txt", "rc": rc,
            "batches": arms_pairs,
            "rho_tau_phi": float(m_tp.group(1)) if m_tp else None,
        }
    if any(v.get("status") != "OK" for v in res["arms"].values()):
        res["status"] = "PARTIAL"
    # 按**批次名**配对（两臂评估节奏相同 ⇒ 批次名可比对），供失败分支 2 用
    by = {t: {x["batch"]: x for x in res["arms"].get(t, {}).get("batches", [])}
          for t in ("A_et", "B_ctrl")}
    paired = []
    for name in sorted(set(by["A_et"]) & set(by["B_ctrl"])):
        xa, xb = by["A_et"][name], by["B_ctrl"][name]
        row = {"batch": name}
        for key in ("d_rho_tower", "rho_none_tower", "rho_p4b_tower"):
            row[key + "_A"] = xa[key]
            row[key + "_B"] = xb[key]
            row[key + "_A_minus_B"] = xa[key] - xb[key]
        paired.append(row)
    res["paired"] = paired
    return res


# ---------------------------------------------------------------- 层 4：项体检
def layer_item_health(run_a, out_dir):
    import pickle
    rep = os.path.abspath(os.path.join(run_a, "replays"))
    if not os.path.isdir(rep):
        return {"status": "MISSING", "note": f"无录像 {rep}"}
    phi = tau = score = 0.0
    n_win = n_tau_ne0 = n_frames = n_frames_with_et = 0
    per_batch = []
    for fn in sorted(os.listdir(rep)):
        if not fn.endswith(".pkl"):
            continue
        with open(os.path.join(rep, fn), "rb") as f:
            d = pickle.load(f)
        b_phi = b_tau = b_score = 0.0
        b_win = b_ne0 = b_fr = 0
        for g in d.get("games", []):
            for fr in g.get("frames", []):
                b_fr += 1
                n_frames += 1
                et = fr.get("et")
                if not et:
                    continue
                n_frames_with_et += 1
                b_phi += float(et[0]); b_tau += float(et[1]); b_score += float(et[2])
                b_win += int(et[3])
                if abs(float(et[1])) > 0.0:
                    b_ne0 += 1
        phi += b_phi; tau += b_tau; score += b_score
        n_win += b_win; n_tau_ne0 += b_ne0
        per_batch.append({"batch": fn, "phi": b_phi, "tau": b_tau, "score": b_score,
                          "windows": b_win, "tau_ne0": b_ne0, "frames": b_fr})
    res = {"status": "OK", "n_frames": n_frames, "n_frames_with_et": n_frames_with_et,
           "sum_phi": phi, "sum_tau": tau, "sum_score": score,
           "n_windows": n_win, "windows_tau_ne0": n_tau_ne0,
           "tau_ne0_frac": (n_tau_ne0 / n_win) if n_win else None,
           "abs_tau_over_abs_phi": (abs(tau) / abs(phi)) if phi else None,
           "per_batch": per_batch}
    with open(os.path.join(out_dir, "item_health.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    return res


# ---------------------------------------------------------------- 报告
def render(report):
    L = []
    A = L.append
    A("# `et_solo100k` §11.13.2 读数（脚本复算，非手抄）\n")
    A("> 本文件由 `scripts/et_solo100k_readout.py` 生成。**只出读数与程序性谓词，不做判决**")
    A("> （预注册 §11.13.3：缺阳性对照 ⇒ 不出 §5 意义上的判决）。\n")

    A("## 层 1 · 机制（主）\n")
    m = report["mechanism"]
    A(f"- 诊断点数：`A_et` = **{m['n_points_a']}**、`B_ctrl` = **{m['n_points_b']}**；"
      f"基线窗口 = 前 **{m['n_base']}** 点，带 = 中位 ± **{m['k']:g}**×MAD（**原始 MAD**）。")
    if m["status"] != "OK":
        A(f"- ⚠️ **{m['status']}**：{m.get('note','')}")
        return "\n".join(L)
    A("")
    A("### 臂内（本 run 自己的前 100 点作对照）\n")
    A("| 指标 | A_et 基线中位 | A_et MAD | A_et 后续中位 | A_et 带内 | B_ctrl 基线中位 | B_ctrl MAD | B_ctrl 后续中位 | B_ctrl 带内 |")
    A("|---|---|---|---|---|---|---|---|---|")
    for k in _METRICS:
        va, vb = m["arms"]["A_et"]["metrics"][k], m["arms"]["B_ctrl"]["metrics"][k]
        ia = "n/a" if va["later_inside"] is None else f"{va['later_inside']}/{va['later_n']}"
        ib = "n/a" if vb["later_inside"] is None else f"{vb['later_inside']}/{vb['later_n']}"
        if va["degenerate"]:
            ia += " ⚠退化"
        if vb["degenerate"]:
            ib += " ⚠退化"
        A(f"| {_METRIC_NAME[k]} | {_fmt(va['base_median'])} | {_fmt(va['base_mad'])} | "
          f"{_fmt(va['later_median'])} | {ia} | {_fmt(vb['base_median'])} | {_fmt(vb['base_mad'])} | "
          f"{_fmt(vb['later_median'])} | {ib} |")
    A("")
    A("### 跨臂（§11.13.4-3「差异 < 两者自身的窗口散布」⇒ 不可分辨）\n")
    A("> 「窗口散布」的操作化 = `max(MAD_A, MAD_B)`（MAD 即 §11.13.2 指定的离散度统计量）。")
    A("> 该操作化在两臂跑完**之前**写定（预注册 §11.13.4 注）。\n")
    A("| 指标 | A_et 后续中位 | B_ctrl 后续中位 | 差 \\|Δ\\| | 散布 max(MAD) | 不可分辨？ |")
    A("|---|---|---|---|---|---|")
    for k in _METRICS:
        c = m["compare"].get(k) or {}
        if c.get("status") == "MISSING":
            A(f"| {_METRIC_NAME[k]} | — | — | — | — | MISSING |")
            continue
        A(f"| {c['name']} | {_fmt(c['later_median_A'])} | {_fmt(c['later_median_B'])} | "
          f"{_fmt(c['diff'])} | {_fmt(c['spread_max_mad'])} | "
          f"{'**是**' if c['indistinguishable'] else '否'} |")
    A("")
    A(f"- 判「不可分辨」的指标：**{', '.join(m['indistinguishable_metrics']) or '（无）'}**")
    A(f"- 全部指标都不可分辨 ⇒ **{ '是' if m['all_indistinguishable'] else '否' }**"
      "（对应 §11.13.4 失败分支 3）")

    # —— 层 1b：EV（§11.13.2 机制层的另两项）——
    e = report.get("ev") or {}
    A("\n### 层 1b · EV（§11.13.2 机制层的另两项：批内 EV / `EVb` 形状）\n")
    if e.get("status") not in ("OK", "PARTIAL") or not e.get("arms"):
        A("- ⚠️ MISSING：日志里没有可解析的 `[solo step N] … EVb=` 行")
    else:
        A(f"- 诊断点（`[solo step]` 行）数：`A_et` = **{e['arms']['A_et']['n_evb_points']}**、"
          f"`B_ctrl` = **{e['arms']['B_ctrl']['n_evb_points']}**"
          f"（**每个 update 都打印**，不随 `diagnose_every` 稀疏化）\n")
        A("| 指标 | A_et 基线中位 | A_et MAD | A_et 后续中位 | A_et 带内 | B_ctrl 基线中位 | B_ctrl MAD | B_ctrl 后续中位 | B_ctrl 带内 |")
        A("|---|---|---|---|---|---|---|---|---|")
        for key, name in (("evb", "EVb（批内·更新前）"),):
            sa, sb = e["arms"]["A_et"][key], e["arms"]["B_ctrl"][key]
            ia = "n/a" if sa.get("later_inside") is None else f"{sa['later_inside']}/{sa['n_later']}"
            ib = "n/a" if sb.get("later_inside") is None else f"{sb['later_inside']}/{sb['n_later']}"
            if sa.get("degenerate"):
                ia += " ⚠退化"
            if sb.get("degenerate"):
                ib += " ⚠退化"
            A(f"| {name} | {_fmt(sa['base_median'])} | {_fmt(sa['base_mad'])} | "
              f"{_fmt(sa['later_median'])} | {ia} | {_fmt(sb['base_median'])} | "
              f"{_fmt(sb['base_mad'])} | {_fmt(sb['later_median'])} | {ib} |")
        A("")
        c = (e.get("compare") or {}).get("EVb") or {}
        if c.get("status") == "MISSING":
            A("- 跨臂 EVb 判定：**MISSING**（某一臂没有后续窗口）")
        else:
            A(f"- 跨臂 EVb：差 \\|Δ\\| **{_fmt(c['diff'])}** vs 散布 max(MAD) "
              f"**{_fmt(c['spread_max_mad'])}** ⇒ "
              f"{'**不可分辨**' if c['indistinguishable'] else '可分辨'}")
        A("")
        A("**池化 EV（评估点，描述性）**\n")
        A("| 评估点 | A_et 池化 EV | B_ctrl 池化 EV |")
        A("|---|---|---|")
        pa = {s: ev for s, ev in e["arms"]["A_et"]["pooled_ev_series"]}
        pb = {s: ev for s, ev in e["arms"]["B_ctrl"]["pooled_ev_series"]}
        for s in sorted(set(pa) | set(pb)):
            A(f"| {s} | {pa.get(s)} | {pb.get(s)} |")

    A("\n## 层 2 · 行为（可复算）\n")
    b = report["behaviour"]
    A("| 指标 | A_et | B_ctrl |")
    A("|---|---|---|")
    for key, name in (("elixir_ge6_pct", "全帧圣水 ≥6 占比 (%)"),
                      ("full_frames_n", "全帧 n"),
                      ("pre_deploy_median", "部署前圣水中位"),
                      ("pre_deploy_n", "部署前 n"),
                      ("xbow_play_rate", "Xbow 出手率"),
                      ("xbow_never_played", "Xbow 从未打出")):
        va = b["arms"].get("A_et", {}).get(key)
        vb = b["arms"].get("B_ctrl", {}).get(key)
        A(f"| {name} | {va} | {vb} |")
    A("")
    p = b.get("paired") or {}
    if any(v.get("A_minus_B") is not None for v in p.values()):
        A("**配对差 `A_et − B_ctrl`**（失败分支 1 的输入；n=1 seed/臂 ⇒ **不下显著结论**，【R5】/【R16】）\n")
        A("| 指标 | A_et | B_ctrl | A − B |")
        A("|---|---|---|---|")
        for key, name in (("elixir_ge6_pct", "全帧圣水≥6 占比 (%)"),
                          ("pre_deploy_median", "部署前圣水中位"),
                          ("xbow_play_rate", "Xbow 出手率")):
            v = p.get(key)
            if not v:
                continue
            A(f"| {name} | {v['A']} | {v['B']} | {v['A_minus_B']} |")
        A("")

    A("## 层 3 · 门禁（优势兑现率 / 配对 Δρ）\n")
    rz = report.get("realization") or {}
    A("### 层 3a · 优势兑现率（§11.13.2 门禁的**主指标**；N=40 帧 = 20 s，θ=1.0）\n")
    A("> 仪器：`offline_engagement_trade.py`（兑现率**只**在这里实现；见 §11.13.10 的命名更正）\n")
    if rz.get("status") not in ("OK", "PARTIAL") or not rz.get("arms"):
        A("- ⚠️ MISSING：未能取到兑现率")
    else:
        A("| 臂 | 兑现率 % | 花费达标 % | 打出塔伤 % | 合格窗口 | 窗口/局 |")
        A("|---|---|---|---|---|---|")
        for t in ("A_et", "B_ctrl"):
            x = rz["arms"].get(t, {})
            A(f"| {t} | {x.get('realize_rate_pct')} | {x.get('spend_ok_pct')} | "
              f"{x.get('tower_dmg_pct')} | {x.get('eligible_windows')} | "
              f"{x.get('eligible_per_game')} |")
        pp = rz.get("paired") or {}
        A("")
        A(f"- 配对 A − B：**{pp.get('A_minus_B')}** 个百分点"
          "（**只报数**；n=1 seed/臂 ⇒ 不下结论，【R5】/【R16】）")
        pb = rz.get("per_batch") or []
        if pb:
            sm = rz.get("per_batch_summary") or {}
            A("")
            A("**逐批兑现率（每个评估点单独跑一次仪器；失败分支 2 的「方向稳不稳」证据）**\n")
            A("| 批次 | A_et 兑现率 % | B_ctrl 兑现率 % | A − B | A 合格窗口 | B 合格窗口 |")
            A("|---|---|---|---|---|---|")
            for x in pb:
                A(f"| {x['batch']} | {x['A']} | {x['B']} | {x['A_minus_B']} | "
                  f"{x['eligible_A']} | {x['eligible_B']} |")
            A("")
            A(f"- 配对差方向：正 **{sm.get('n_pos')}** / 负 **{sm.get('n_neg')}** / 零 **{sm.get('n_zero')}**"
              f"（n = {sm.get('n')}），均值 **{_fmt(sm.get('mean_A_minus_B'), 4)}** 个百分点")
        A("")
    A("### 层 3b · 配对 Δρ（`analyze_online_trade.py`，在线口径）\n")
    g = report["gate"]
    for tag in ("A_et", "B_ctrl"):
        arm = g["arms"].get(tag, {})
        A(f"**{tag}**（status={arm.get('status')}）")
        if arm.get("batches"):
            A("")
            A("| 批次 | 局数 | 窗口/局 | \\|τ\\|/\\|φ\\| | ρnone(塔血) | ρp4b(塔血) | Δρ(塔血) |")
            A("|---|---|---|---|---|---|---|")
            for x in arm["batches"]:
                A(f"| {x['batch']} | {x['games']} | {x['windows_per_game']} | {x['tau_phi_ratio']} | "
                  f"{x['rho_none_tower']:+.3f} | {x['rho_p4b_tower']:+.3f} | {x['d_rho_tower']:+.3f} |")
        A(f"")
        A(f"- ρ(τ, φ) = **{arm.get('rho_tau_phi')}**（不冗余的判据）")
        A("")
    gp = g.get("paired") or []
    if gp:
        A("**按批次配对：`Δρ(p4b − none, 塔血)`**（失败分支 2 的输入；**只报数、不下结论**）\n")
        A("| 批次 | A_et Δρ | B_ctrl Δρ | A − B |")
        A("|---|---|---|---|")
        for x in gp:
            A(f"| {x['batch']} | {x['d_rho_tower_A']:+.3f} | {x['d_rho_tower_B']:+.3f} | "
              f"{x['d_rho_tower_A_minus_B']:+.3f} |")
        A("")

    A("## 层 4 · 项自身体检（`et` 明细）\n")
    h = report["item_health"]
    if h.get("status") == "OK":
        A(f"- 帧数 **{h['n_frames']}**，其中带 `et` **{h['n_frames_with_et']}**"
          f"（{100.0 * h['n_frames_with_et'] / max(1, h['n_frames']):.1f}%）")
        A(f"- Σφ = **{h['sum_phi']:.2f}**、Στ = **{h['sum_tau']:.2f}**、Σscore = **{h['sum_score']:.2f}**")
        A(f"- 窗口数 **{h['n_windows']}**，其中 `τ≠0` **{h['windows_tau_ne0']}**"
          f"（{100.0 * (h['tau_ne0_frac'] or 0):.1f}%）")
        A(f"- \\|Στ\\|/\\|Σφ\\| = **{_fmt(h['abs_tau_over_abs_phi'], 4)}**")
    else:
        A(f"- ⚠️ {h.get('status')}: {h.get('note','')}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-a", required=True)
    ap.add_argument("--log-b", required=True)
    ap.add_argument("--run-a", required=True)
    ap.add_argument("--run-b", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--markdown", default=None)
    ap.add_argument("--json", dest="json_path", default=None)
    ap.add_argument("--baseline-n", type=int, default=100)
    ap.add_argument("--baseline-k", type=float, default=3.0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[readout] out_dir 解析为 {os.path.abspath(args.out_dir)}")
    report = {
        "mechanism": layer_mechanism(args.log_a, args.log_b, args.baseline_n, args.baseline_k),
        "ev": layer_ev(args.log_a, args.log_b, args.baseline_n, args.baseline_k),
        "behaviour": layer_behaviour(args.run_a, args.run_b, args.out_dir),
        "gate": layer_gate(args.run_a, args.run_b, args.out_dir),
        "realization": layer_realization(args.run_a, args.run_b, args.out_dir),
        "item_health": layer_item_health(args.run_a, args.out_dir),
    }
    md = render(report)
    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as f:
            f.write(md + "\n")
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(md)
    bad = [k for k, v in report.items() if v.get("status") not in ("OK",)]
    if bad:
        print(f"\n[readout] 非 OK 的层: {bad}（**不许静默当 0 用**；见各层 raw 文件）")
        return 1
    print("\n[readout] 四层全部 OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
