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

# T1-1b 补齐（2026-09-19）：原先这里是**手写**的 UTF-8 兜底块（只处理 stdout、且不处理 stderr
# ⇒ traceback 在 GBK 下仍是乱码）。现收敛到 T1-1 的**单一实现**（两路 + errors='replace'）。
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                                 "src", "clasher_new"))
from rl.io_bootstrap import force_utf8_stdout  # noqa: E402
force_utf8_stdout()

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


def _fx(x, nd=3):
    """带正号的定长格式，`None` → ``n/a``（零方差批次会导致某些 ρ **无定义**；见 `_num_or_none`）。

    表格里**必须**保留这些行并标 `n/a` —— 丢掉它们等于把"这个评估点全胜、ρ(win) 不可算"
    这个**事实**从判读产物里删掉（2026-09-18 第五处更正实测踩到）。
    """
    return "n/a" if x is None else f"{x:+.{nd}f}"


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
_RE_GATE_BATCH = re.compile(
    r"^(\S+\.pkl)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+"
    r"([+-][\d.]+|n/a)\s+([+-][\d.]+|n/a)\s+([+-][\d.]+|n/a)\s+([+-][\d.]+|n/a)\s+([+-][\d.]+|n/a)",
    re.M)
_RE_GATE_ROW = re.compile(r"^(\S+\.pkl)\s+\d+\s+[\d.]+\s+[\d.]+\s", re.M)
_RE_TAU_PHI = re.compile(r"ρ\(τ, φ\)\s*逐批.*?均值\s*([+-][\d.]+)")
_RE_POOL_PLAYS = re.compile(r"卡牌打出（帧内去重槽位）=(\d+)")
_RE_XBOW_ROW = re.compile(r"^\s+Xbow\s+(\d+)\s+([\d.]+)%", re.M)


def parse_xbow_rate(text):
    """**按 S1 门禁写死的口径**算 Xbow 出手率（`docs/s1_gate_2026-09-18.md:97`）：

        「帧内**去重槽位**后 `Xbow` 次数 ÷ 卡牌打出总次数」，且**只取 POOLED 段**。

    ⚠️ 仪器 `forensics_card_usage.py` 会打**两个** `§2 卡牌使用` 段：`（POOLED）` 与
    `（仅 <最后一个回放>）`。两段的「从未打出的卡」列表**可以不同**（实测 A_et：POOLED `[]`
    而仅-league_100000 `['Xbow']`）—— 取错段就会得到相反的 `never` 结论。
    本函数显式切出 POOLED 段，避免"取第一个匹配"这种靠运气的事。
    """
    i = text.find("§2 卡牌使用（POOLED）")
    seg = text
    if i >= 0:
        j = text.find("§2 卡牌使用（仅", i)
        seg = text[i:j] if j > i else text[i:]
    m_tot = _RE_POOL_PLAYS.search(seg)
    if not m_tot:
        return {"rate": None, "plays": None, "total": None}
    total = int(m_tot.group(1))
    m_row = _RE_XBOW_ROW.search(seg)
    plays = int(m_row.group(1)) if m_row else 0     # 表里没有该行 = 一次没打
    return {"rate": (plays / total if total else None), "plays": plays, "total": total}


def _num_or_none(s):
    """``n/a`` → None（**不许**让 n/a 把整行从解析结果里挤出去）。

    ★ 2026-09-18 第五处更正（测量侧，判读产物）：仪器对**零方差批次**会打 ``n/a`` ——
    实例 = `A_et` 的 `eval@32000`（**20W/0L/0D**，`winrate=1.000±0.000`）⇒ 该批 `ρ(win)` **无定义**。
    旧正则只认数字 ⇒ 那一行**匹配失败、被静默丢掉**（层 3b 的 A 表 14 批变 **13** 批，
    md 与"按批次配对"表都少一行），而 `status` 仍是 `OK`。这正是本仓反复吃亏的
    「静默漏数」。现在：① 允许 `n/a` 并记 `None`；② 与原始表**行数对账**，不一致就降级 + 写 note。
    """
    if s is None or s == "n/a":
        return None
    return float(s)


def parse_gate_batches(text):
    """仪器 stdout → 逐批 dict 列表（`n/a` 记 `None`，**不丢行**；见 `_num_or_none`）。"""
    out = []
    for m in _RE_GATE_BATCH.finditer(text):
        out.append({"batch": m.group(1), "games": int(m.group(2)),
                    "windows_per_game": float(m.group(3)),
                    "tau_phi_ratio": float(m.group(4)),
                    "rho_none_tower": _num_or_none(m.group(5)),
                    "rho_p4b_tower": _num_or_none(m.group(6)),
                    "rho_none_win": _num_or_none(m.group(7)),
                    "rho_p4b_win": _num_or_none(m.group(8)),
                    "d_rho_tower": _num_or_none(m.group(9))})
    return out


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
        mnever = _RE_NEVER.search(text)   # 注意：取**第一处** = POOLED 段（不是"仅某回放"段）
        xbow_never = None
        if mnever:
            xbow_never = "Xbow" in [s.strip().strip("'\"") for s in mnever.group(1).split(",")]
        xb = parse_xbow_rate(text)
        arm = {"status": "OK" if (m6 and mpre and xb["rate"] is not None) else "PARSE_FAIL",
               "raw": f"forensics_{tag}.txt", "rc": rc}
        arm["elixir_ge6_pct"] = float(m6.group(2)) if m6 else None
        arm["full_frames_n"] = int(m6.group(1)) if m6 else None
        arm["pre_deploy_median"] = float(mpre.group(2)) if mpre else None
        arm["pre_deploy_n"] = int(mpre.group(1)) if mpre else None
        arm["xbow_never_played"] = xbow_never
        # ★ 2026-09-18 第六处更正（测量侧）：旧写法 `0 if xbow_never else None`
        #   —— 注释写着「否则需查表」，而**查表从未实现** ⇒ 只要该臂打过 1 次 Xbow，
        #   行为层第 3 项（预注册 §11.13.2 明列）在判读产物里就**恒为 None**（两臂都是）。
        #   现在按 S1 门禁写死的口径直接算：`docs/s1_gate_2026-09-18.md:97`
        #   「帧内去重槽位后 Xbow 次数 ÷ 卡牌打出总次数」，且**只取 POOLED 段**。
        arm["xbow_plays"] = xb["plays"]
        arm["xbow_plays_total"] = xb["total"]
        arm["xbow_play_rate"] = xb["rate"]
        if (xbow_never is True and (xb["plays"] or 0) > 0) or \
           (xbow_never is False and xb["plays"] == 0):
            arm["xbow_note"] = (f"⚠️ 内部不一致：POOLED「从未打出」={xbow_never} 而计数={xb['plays']}"
                                "（仪器口径需复核）")
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
        arms_pairs = parse_gate_batches(text)
        # 行数对账：原始表的行数 vs 解析到的行数。**不静默丢批次**（第五处更正，见 _num_or_none）。
        n_raw = len(_RE_GATE_ROW.findall(text))
        notes = []
        if n_raw != len(arms_pairs):
            notes.append(f"原始表 {n_raw} 行 vs 解析 {len(arms_pairs)} 行（有批次未被解析）")
        n_win_na = sum(1 for x in arms_pairs
                       if x["rho_none_win"] is None or x["rho_p4b_win"] is None)
        if n_win_na:
            bad = [x["batch"] for x in arms_pairs
                   if x["rho_none_win"] is None or x["rho_p4b_win"] is None]
            notes.append(f"ρ(win) 无定义的批次 {n_win_na} 个（零方差，如全胜/全负）：{', '.join(bad)}")
        if rc != 0:
            notes.append(f"仪器 returncode={rc}（该臂整体非零退出，逐批读数可能不全）")
        m_tp = _RE_TAU_PHI.search(text)
        _st = "OK" if arms_pairs else "PARSE_FAIL"
        if _st == "OK" and (n_raw != len(arms_pairs) or rc != 0):
            _st = "PARTIAL"
        res["arms"][tag] = {
            "status": _st, "raw": f"gate_{tag}.txt", "rc": rc,
            "batches": arms_pairs, "n_raw_rows": n_raw,
            "n_win_rho_undefined": n_win_na,
            "rho_tau_phi": float(m_tp.group(1)) if m_tp else None,
        }
        if notes:
            res["arms"][tag]["note"] = "；".join(notes)
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

    def _xbow_cell(arm):
        """出手率**连同计数**一起显示（【R4】：判据的分子/分母要看得见，别只给一个比值）。"""
        r = arm.get("xbow_play_rate")
        if r is None:
            return "None"
        return f"{r:.6f}（{arm.get('xbow_plays')}/{arm.get('xbow_plays_total')}）"

    A("| 指标 | A_et | B_ctrl |")
    A("|---|---|---|")
    for key, name in (("elixir_ge6_pct", "全帧圣水 ≥6 占比 (%)"),
                      ("full_frames_n", "全帧 n"),
                      ("pre_deploy_median", "部署前圣水中位"),
                      ("pre_deploy_n", "部署前 n"),
                      ("xbow_play_rate", "Xbow 出手率"),
                      ("xbow_never_played", "Xbow 从未打出")):
        if key == "xbow_play_rate":
            A(f"| {name} | {_xbow_cell(b['arms'].get('A_et', {}))} | "
              f"{_xbow_cell(b['arms'].get('B_ctrl', {}))} |")
            continue
        va = b["arms"].get("A_et", {}).get(key)
        vb = b["arms"].get("B_ctrl", {}).get(key)
        A(f"| {name} | {va} | {vb} |")
    for tag in ("A_et", "B_ctrl"):
        if b["arms"].get(tag, {}).get("xbow_note"):
            A(f"| `{tag}` 提示 | {b['arms'][tag]['xbow_note']} |  |")
    A("")
    A("> 层 2 的口径（预注册未逐字写死比值，故在这里显式标注）：**全帧圣水 ≥6 占比** = 全部帧里"
      "圣水≥6 的占比（`[POOLED] 全帧`）；**部署前圣水中位** = `[POOLED] pre (=post+费)` 的中位；"
      "**Xbow 出手率** = 帧内去重槽位后 Xbow 次数 ÷ 卡牌打出总次数（口径同 `docs/s1_gate_2026-09-18.md:97`）。\n")
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
                  f"{_fx(x['rho_none_tower'])} | {_fx(x['rho_p4b_tower'])} | {_fx(x['d_rho_tower'])} |")
        A(f"")
        A(f"- ρ(τ, φ) = **{arm.get('rho_tau_phi')}**（不冗余的判据）")
        if arm.get("note"):
            A(f"- ⚠️ {arm['note']}")
        if arm.get("n_win_rho_undefined"):
            A(f"- ⚠️ `ρ(win)` 无定义（**零方差**：该评估点全胜或全负）⇒ 该批的胜率侧 Δρ **不参与**任何判定；"
              f"`ρ(塔血)` 列仍有效")
        A("")
    gp = g.get("paired") or []
    if gp:
        A("**按批次配对：`Δρ(p4b − none, 塔血)`**（失败分支 2 的输入；**只报数、不下结论**）\n")
        A("| 批次 | A_et Δρ | B_ctrl Δρ | A − B |")
        A("|---|---|---|---|")
        for x in gp:
            A(f"| {x['batch']} | {_fx(x['d_rho_tower_A'])} | {_fx(x['d_rho_tower_B'])} | "
              f"{_fx(x['d_rho_tower_A_minus_B'])} |")
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


def render_judgment(report, logs):
    """按预注册 §11.13.8 的**必写条目**顺序，生成判读文档骨架。

    ⚠️ 本函数只做两件事：① 把**静态条目**（声明/更正清单/明确未做/台账落点）写全；
    ② 把四层读数与**可机械计算的谓词**填进去。它**不替人下判决** —— 依赖方向判断的分支
    （§11.13.4 的 1、2）只列**输入**并标「须人工判读」，因为 §11.13.11 已实测
    n=1 时胜率/奖励层差异**不可归因**。
    """
    L = []
    A = L.append
    A("# `et_solo100k` 判读（`A_et` vs `B_ctrl`）—— 按预注册 §11.13.8 结构\n")

    # ---- 1) 置顶声明（§11.13.3，必须置顶）----
    A("## 0. ★ 不构成判决声明（预注册 §11.13.3，逐条）\n")
    A("1. **缺阳性对照**（§5.2 的 `C_pos = elixir_diff_weight 0.5→0`）⇒ 按 §5.2 **无法判 `VALID`**。")
    A("2. **n = 1 seed/臂**（§8 原本要求 2 seed/臂）⇒ 按【R5】**只能看大效应**；跨 run 数字不可直接比。")
    A("3. ⇒ 本轮**只出「观察 + 机制层读数」**；**不得**写成「修好了」或「确认无效」；")
    A("   只允许三种写法：**观察到什么 / 未达行为可见阈值 / 不可分辨**。")
    A("4. ★ **§11.13.11 实测标定**：同 seed 同配置、仅差一个测量开关的运行之间 `eval@8000` 可相差 **0.60**"
      "（0.800 vs 0.200），而既有**同配置三连跑**末点胜率极差已达 **0.313**"
      "⇒ **胜率/奖励层的 A vs B 差异在 n=1/臂 下一律不可归因**；失败分支 1、2 **只能读方向**。\n")

    # ---- 2) 臂与命令 + 更正清单 ----
    A("## 1. 臂、命令与四处更正（必须显式列出，§11.13.8 第 2 条）\n")
    A("| 臂 | 配置 | 说明 |")
    A("|---|---|---|")
    A("| `A_et`（干预） | `--config economy_et` | `engagement_trade = 0.5`，与 `elixir_diff_weight` **同汇率**（【R7】） |")
    A("| `B_ctrl`（阴性对照） | `--config economy_etm` | `= economy`；**唯一**有效差异 `engagement_trade_measure_only 0→1` |")
    A("")
    A("两臂逐字同参同 seed、`solo` 模式、顺序串跑，且**均带** `--adv-inert-probe` 与 `--diagnose-every 1`。")
    A("")
    A("**六处发射/口径更正（都不是最初的命令）**：")
    A("")
    A("| # | 节 | 问题 | 处置 |")
    A("|---|---|---|---|")
    A("| 1 | §11.13.6 | 首次发射**漏 `--adv-inert-probe`** ⇒ `resid_norm`/`grad_cos` 读不出来（日志 0 行） | 两臂对称补上后重启（纯测量） |")
    A("| 2 | §11.13.7 | `_set_et_measure` 使 `--config economy` 的**评估录像没有 `et`** ⇒ 门禁对照列为空 | 对照臂改 `economy_etm` |")
    A("| 3 | §11.13.9 | §11.13.2 **自身不一致**：`diagnose_every 10` 下 100k 只有 ~76 点 ⇒ `n_later = 0`、**主判据不可执行** | **不改 N**（那会犯【R16】），改采集：两臂 `--diagnose-every 1` |")
    A("| 4 | §11.13.10 | 门禁行把**指标**（兑现率）与**仪器**（`analyze_online_trade`）写岔了 —— 兑现率只在 `offline_engagement_trade` 里 | 两者都算：层 3a 离线口径 + 层 3b 在线口径 |")
    A("| 5 | §11.13.12 | **读数脚本静默丢批次**：仪器对**零方差批次**（`A_et` 的 `eval@32000` = **20W/0L** ⇒ `ρ(win)` 无定义）打 `n/a`，而解析正则只认数字 ⇒ 该行被丢掉（层 3b 的 A 表 14→**13** 批，`status` 仍 `OK`） | 正则接受 `n/a`（记 `None`）+ **与原始表行数对账**（不符 ⇒ `PARTIAL` + note）+ 仪器零方差批次不再抛 `StatisticsError`；回归 `--selftest` |")
    A("| 6 | §11.13.13 | **行为层第 3 项恒 `None`**：`xbow_play_rate` 写成 `0 if never else None`（注释「否则需查表」）而**查表从未实现** ⇒ 只要该臂打过 1 次 Xbow，预注册明列的「Xbow 出手率」在产物里就永远读不到 | 按 S1 门禁口径（`docs/s1_gate_2026-09-18.md:97`）**显式切 POOLED 段**算 `次数/总打出`，并把**分子分母一起显示**；回归 `--selftest` |")
    A("")
    A(f"日志：`A_et` = `{logs[0]}`；`B_ctrl` = `{logs[1]}`\n")

    # ---- 3) 四层读数（固定顺序）----
    A("## 2. 四层读数（**脚本复算**，顺序 = 机制（主）→ 行为 → 门禁 → 项体检）\n")
    # 内嵌时把 render() 的 `##` 降一级，避免与外层章节号打架
    A(render(report).replace("\n## ", "\n### "))
    A("")

    # ---- 4) 失败分支判定表 ----
    mech = report.get("mechanism") or {}
    ev = report.get("ev") or {}
    beh = report.get("behaviour") or {}
    rz = report.get("realization") or {}
    ind = list(mech.get("indistinguishable_metrics") or []) + list(ev.get("indistinguishable_metrics") or [])
    all_ind = bool(mech.get("all_indistinguishable")) and bool(
        (ev.get("compare") or {}).get("EVb", {}).get("indistinguishable"))
    A("## 3. 失败分支逐条判定（§11.13.4；依据必须写出）\n")
    A("| # | 分支 | 机械输入 | 判定 |")
    A("|---|---|---|---|")
    A(f"| 1 | 机制层改善但行为层三项全无改善 ⇒ 未达行为可见阈值 | 行为层配对差见层 2 的 `A_et − B_ctrl` 表 | **须人工判读**（依赖方向；且 §11.13.11 限制） |")
    A(f"| 2 | 门禁（兑现率）显著低于 `B_ctrl` ⇒ 买到只守不推 | 层 3a 逐批兑现率配对差（方向计数见该表下方） | **须人工判读**（n=1 ⇒ 只能读方向） |")
    A(f"| 3 | 跨臂差异 < 两者自身窗口散布 ⇒ 不可分辨 | 机制层判「不可分辨」的指标 = `{', '.join(ind) or '（无）'}` | **{'触发' if all_ind else '未触发'}** |")
    A(f"| 4 | 训练跑不动/降级/异常 ⇒ 按【R1】先归因外部、不自动降档 | 各层 status：机制 `{mech.get('status')}` / 行为 `{beh.get('status')}` / 兑现率 `{rz.get('status')}` | **须人工判读**（异常须在正文留证） |")
    A("")

    # ---- 5) 明确未做 ----
    A("## 4. 明确未做（§11.13.8 第 5 条）\n")
    A("- **没有**跑阳性对照 `C_pos`（`elixir_diff_weight 0.5→0`）⇒ 按 §5.2 无法判 `VALID`。")
    A("- **没有**跑 2 seed/臂（§8 的原始要求）⇒ 无法把差异与 run 间散布区分开。")
    A("- **没有**用 §11.10 的 λ 扫描去挑权重（那是 in-sample 事后选择，【R16】禁止）。")
    A("- **没有**因为单点胜率难看而改档/降级（【R1】：先归因外部，本轮实测一次 worker 收尾空转即属此类）。")
    A("- **没有**改任何判据（N 仍 = 100）；**六处**更正全部在**测量侧**（含 §11.13.12「不静默丢批次」、§11.13.13「行为层第 3 项恒 None」），依据是代码核对与回归测试。")
    A("")

    # ---- 6) 台账落点 ----
    A("## 5. 台账落点（§11.13.8 第 6 条）\n")
    A("- `docs/agents/ledger.md`：新增/更新结论条目（X 或 O）。")
    A("- `docs/agents/plans_runs_docs.md` §8：run 台账行。")
    A("- `AGENTS.md`：加/改一行指针（`python3 scripts/_agents_split.py --check` 必须 PASS）。")
    A("- 本文件同目录的原始仪器输出（`*.txt`）作为证据一并保留。")
    return "\n".join(L)


def _selftest():
    """【R8】回归测试：护栏 —— 仪器输出里的 `n/a` 行**不许**被解析丢掉（§11.13.12）。

    真实触发例：`A_et` 的 `eval@32000` = **20W/0L/0D**（`winrate=1.000±0.000`）⇒ 该批 `ρ(win)`
    **零方差无定义** ⇒ 仪器打 `n/a` ⇒ 旧正则（只认数字）匹配失败 ⇒ 该行被**静默丢弃**。
    """
    table = "\n".join([
        "league_0.pkl               20     19.8     0.033    +0.329    +0.353    +0.409    +0.427     +0.024",
        "league_100000.pkl          20      8.0     0.091    -0.095    +0.005    +0.000    +0.088     +0.099",
        "league_32000.pkl           20      3.7     0.000    +0.096    +0.096       n/a       n/a     +0.000",
        "league_40000.pkl           20     35.3     0.034    +0.701    +0.708    +0.763    +0.763     +0.008",
    ])
    old = re.compile(r"^(\S+\.pkl)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([+-][\d.]+)\s+([+-][\d.]+)\s+"
                     r"([+-][\d.]+)\s+([+-][\d.]+)\s+([+-][\d.]+)", re.M)
    ok = True

    got = parse_gate_batches(table)
    t1 = len(got) == 4 and len(old.findall(table)) == 3
    print(f"[{'PASS' if t1 else 'FAIL'}] 含 n/a 的表：新解析 {len(got)} 行（期望 4）、"
          f"旧正则 {len(old.findall(table))} 行（期望 3，即**会丢**该行）")
    ok &= t1

    row = [x for x in got if x["batch"] == "league_32000.pkl"][0]
    t2 = (row["rho_none_win"] is None and row["rho_p4b_win"] is None
          and row["rho_none_tower"] == 0.096 and row["d_rho_tower"] == 0.0)
    print(f"[{'PASS' if t2 else 'FAIL'}] n/a 行：win 侧 = None、塔血侧仍为数字 "
          f"（tower={row['rho_none_tower']} d={row['d_rho_tower']}）")
    ok &= t2

    t3 = _RE_GATE_ROW.findall(table).__len__() == 4
    print(f"[{'PASS' if t3 else 'FAIL'}] 行数对账正则 `_RE_GATE_ROW` 也认这一行（4/4）")
    ok &= t3

    t4 = _fx(None) == "n/a" and _fx(0.0) == "+0.000" and _fmt(None) == "None"
    print(f"[{'PASS' if t4 else 'FAIL'}] None 安全格式化：_fx(None)={_fx(None)!r}、"
          f"_fx(0.0)={_fx(0.0)!r}")
    ok &= t4

    # ---- 第六处更正：Xbow 出手率必须真的算出来（旧实现只要打过就恒 None），且必须取 POOLED 段 ----
    forensics = "\n".join([
        "--- §2 卡牌使用（POOLED）---",
        "  卡牌打出（帧内去重槽位）=8769  子动作直方图={'1': 7113}",
        "    Skeletons      1872   21.35%  费=1.0  最大落点 y=20",
        "    Xbow              5    0.06%  费=6.0  最大落点 y=9",
        "  牌组里从未打出的卡: []",
        "--- §2 卡牌使用（仅 league_100000.pkl）---",
        "  卡牌打出（帧内去重槽位）=306  子动作直方图={'1': 232}",
        "  牌组里从未打出的卡: ['Xbow']",
    ])
    xb = parse_xbow_rate(forensics)
    t5 = (xb["plays"] == 5 and xb["total"] == 8769 and abs(xb["rate"] - 5 / 8769) < 1e-12)
    print(f"[{'PASS' if t5 else 'FAIL'}] Xbow 出手率真算出来：{xb['plays']}/{xb['total']}"
          f"={xb['rate']:.6f}（旧实现 `0 if never else None` 恒 None）")
    ok &= t5
    t6 = _RE_NEVER.search(forensics).group(1).strip() == ""
    print(f"[{'PASS' if t6 else 'FAIL'}] 『从未打出』取到的是 **POOLED** 段（空表），"
          f"不是『仅-回放』段的 ['Xbow']（取错段会得到相反结论）")
    ok &= t6
    xb0 = parse_xbow_rate(forensics.replace(
        "    Xbow              5    0.06%  费=6.0  最大落点 y=9\n", ""))
    t7 = (xb0["plays"] == 0 and xb0["rate"] == 0.0)
    print(f"[{'PASS' if t7 else 'FAIL'}] 表里没有 Xbow 行（真的没打过）⇒ plays=0、rate=0.0")
    ok &= t7

    print(f"\n{'ALL PASS' if ok else 'FAILED'} (7/7 期望)")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-a", default=None)
    ap.add_argument("--log-b", default=None)
    ap.add_argument("--run-a", default=None)
    ap.add_argument("--run-b", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--selftest", action="store_true",
                    help="只跑护栏回归（不碰数据；§11.13.12 的 n/a 行不许被丢）")
    ap.add_argument("--markdown", default=None)
    ap.add_argument("--judgment", default=None,
                    help="按预注册 §11.13.8 结构输出判读文档（静态条目 + 四层读数 + 分支输入表）")
    ap.add_argument("--json", dest="json_path", default=None)
    ap.add_argument("--baseline-n", type=int, default=100)
    ap.add_argument("--baseline-k", type=float, default=3.0)
    args = ap.parse_args()

    if args.selftest:
        return _selftest()
    for req in ("log_a", "log_b", "run_a", "run_b", "out_dir"):
        if not getattr(args, req):
            ap.error(f"--{req.replace('_', '-')} 必填（或 --selftest）")

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
    if args.judgment:
        jm = render_judgment(report, (args.log_a, args.log_b))
        with open(args.judgment, "w", encoding="utf-8") as f:
            f.write(jm + "\n")
        print(f"\n[judgment] §11.13.8 结构判读骨架已写出：{args.judgment}")
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
