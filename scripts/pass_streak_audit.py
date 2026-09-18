# -*- coding: utf-8 -*-
"""攒费样本审计 v2：把「不出牌」拆成三类，并**自己独立验**非法性（只读录像）。

## 为什么 v1 不够（v1 = 本脚本第一版，已作废，留证见文档）

v1 用「`bundle` 是否为空」定义"不出牌"，于是有一类帧被**记成"出牌了"**：
模型提交了一个**非法包**（例如同一槽位重复 3 次、或总费 11 > 手上 5 费），
`validate_bundle` **整包拒绝** ⇒ 该帧**什么都没发生**，但 `bundle` 非空。
实测在 `et_solo100k` 的 `league_8000.pkl` 里，**同一套非法包被连发 13 帧**，
期间圣水从 5.18 一路涨到 7.50（因为花不掉）——
**这恰好会被误读成"模型在攒费"**。

## v2 的口径（三个类别，互斥且穷尽）

判据不依赖录像没记录的 `invalid_count`，而是用**客观会计**：
一帧里圣水**只可能因为回费上升**（回费 ≤ 0.36/帧），任何真实出牌都要花掉 ≥1 费。故

    `elixir0[i] >= pre[i] - 1e-6`  ⟺  **这一帧一个子动作都没真的执行**（没花任何费）

| 类别 | 定义 | 含义 |
|---|---|---|
| **A 主动不出牌** | `bundle == []`（且圣水没降） | 合法地选择"本帧不出" |
| **B 整包被拒** | `bundle != []` 且**圣水没降** | **非法动作、白掉一帧**（此前所有行为统计都看不见） |
| **C 实际出牌** | `bundle != []` 且圣水下降 | 真的部署了 |

`pre[i] = elixir0[i-1]`（**精确**：上一帧步末值 = 下一帧决策值；第 0 帧 = 开局 5.0）。

**B 的非法性由脚本自己独立复算**（不靠引擎的 `invalid_count`）：
① **重复槽位**（`len(set(slots)) != len(slots)`）⇒ `validate_bundle` 必拒；
② **总费 > 决策圣水** ⇒ 必拒；③ 以上都不满足的 B 记作 `B_unknown`（掩码缺口 / 引擎级拒绝类，P1-20）。

## 关键读数（本脚本要回答的问题）

    ① **A ∩ {决策圣水 ≥ 6}**：`≥6` 时**整副 8 张卡任何一张都付得起**且手牌恒 4 张
       ⇒ 该帧必然有合法出牌项 ⇒ 这才是**无假设的"能出却选择不出"**证据；
    ② B 的规模、最长连续长度、其中 `pre ≥ 6` 的帧数（= 会被误读成"攒费"的那些）；
    ③ A 段的长度分布与**段内圣水峰值**（真正"主动积攒"到过多少）；
    ④ 回费标定（本帧没花钱的相邻帧差分）⇒ 直接给出"买得起 Xbow(6 费) 要几帧"。

## 纪律

只读；不改训练/奖励/判定；【R3】非判据、【R5】n=1 seed 但同录像内逐帧计数是**穷举**。

用法（必须在 src/clasher_new 下运行）：
    cd src/clasher_new && PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe \
        ../../scripts/pass_streak_audit.py \
        --replays A_et=runs/et_solo100k/replays --replays B_ctrl=runs/et_ctrl100k/replays \
        --json docs/pass_streak_audit.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src", "clasher_new")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from rl.io_bootstrap import force_utf8_stdout  # noqa: E402  (T1-1: UTF-8 兜底单一来源)
force_utf8_stdout()

import numpy as np  # noqa: E402

#: 圣水 ≥ 该值 ⇒ 整副卡组（最贵 Xbow 6 费）都付得起 ⇒ 该帧必然存在合法出牌项
ALL_AFFORD = 6.0
LEVELS = (3.0, 4.0, 5.0, 6.0, 8.0)
#: 回费上限（双倍期 0.357/帧）；用于"圣水没降"的容差
REGEN_MAX = 0.36


def _cost(name):
    from card_utils import Card
    try:
        return float(Card(name).elixir)
    except Exception:
        return float("nan")


def audit_game(game, threshold=ALL_AFFORD):
    frames = game.get("frames") or []
    n = len(frames)
    if n == 0:
        return None
    pre, post, cls, sig = [], [], [], []
    for i, f in enumerate(frames):
        post.append(float(f.get("elixir0", np.nan)))
        pre.append(5.0 if i == 0 else post[i - 1])
    for i, f in enumerate(frames):
        b = f.get("bundle") or []
        no_spend = not (post[i] < pre[i] - 1e-6)
        if len(b) == 0:
            cls.append("A")
        else:
            cls.append("B" if no_spend else "C")
        slots = tuple(int(s[1]) for s in b if s and s[0] == "deploy")
        cost = float(np.nansum([_cost(c) for c in (f.get("cards") or [])])) \
            if f.get("cards") else 0.0
        dup = len(set(slots)) != len(slots)
        over = cost > pre[i] + 1e-6
        sig.append({"slots": slots, "cost": round(cost, 2), "dup": dup, "over": over})

    regen = []
    for i in range(1, n):
        if cls[i] == "A" and not np.isnan(post[i]) and not np.isnan(post[i - 1]):
            d = post[i] - post[i - 1]
            if d > -1e-9:
                regen.append(d)

    runs = []          # 连续 A 段
    i = 0
    while i < n:
        if cls[i] == "A":
            j = i
            while j + 1 < n and cls[j + 1] == "A":
                j += 1
            seg = pre[i:j + 1]
            runs.append({"start": i, "len": j - i + 1, "pre_start": float(seg[0]),
                         "pre_max": float(np.nanmax(seg)),
                         "next_cls": (cls[j + 1] if j + 1 < n else None)})
            i = j + 1
        else:
            i += 1
    bruns = []         # 连续 B 段
    i = 0
    while i < n:
        if cls[i] == "B":
            j = i
            while j + 1 < n and cls[j + 1] == "B":
                j += 1
            bruns.append({"start": i, "len": j - i + 1,
                          "pre_start": float(pre[i]), "pre_max": float(np.nanmax(pre[i:j + 1])),
                          "slots": sig[i]["slots"]})
            i = j + 1
        else:
            i += 1

    anomaly = int(sum(1 for i in range(n) if cls[i] == "A" and post[i] < pre[i] - 1e-6))
    return {"n": n, "pre": pre, "post": post, "cls": cls, "sig": sig,
            "regen": regen, "runs": runs, "bruns": bruns, "anomaly": anomaly,
            "winner": game.get("winner")}


def load_dir(d):
    files = sorted(glob.glob(os.path.join(d, "league_*.pkl")),
                   key=lambda p: int(re.search(r"league_(\d+)\.pkl", p).group(1)))
    if not files:
        raise SystemExit(f"目录里没有 league_*.pkl: {d}")
    out = []
    for p in files:
        step = int(re.search(r"league_(\d+)\.pkl", p).group(1))
        with open(p, "rb") as fh:
            out.append((step, pickle.load(fh).get("games") or []))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", action="append", default=[], metavar="LABEL=DIR")
    ap.add_argument("--threshold", type=float, default=ALL_AFFORD)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    if not a.replays:
        raise SystemExit("至少给一个 --replays LABEL=DIR")

    summary = {}
    for spec in a.replays:
        label, _, d = spec.partition("=")
        d = d if os.path.isabs(d) else os.path.join(_SRC, d)
        if not os.path.isdir(d):
            raise SystemExit(f"目录不存在: {d}")
        points = load_dir(d)
        print("=" * 108)
        print(f"[{label}] {os.path.basename(d)}：{len(points)} 个评估点")
        print(f"{'step':>7} {'帧数':>7} {'A主动不出%':>10} {'B整包被拒%':>10} {'C实际出牌%':>10} "
              f"{'A∩≥6':>6} {'B∩≥6':>6} {'A段最长':>7} {'A段峰值':>7} {'B段最长':>7} {'非法可自证%':>10}")
        agg = {"n": 0, "A": 0, "B": 0, "C": 0, "A_ge6": 0, "B_ge6": 0, "C_ge6": 0,
               "a_run_max": 0, "a_run_peak": 0.0, "b_run_max": 0, "anomaly": 0,
               "b_dup": 0, "b_over": 0, "b_unknown": 0, "runs": [], "bruns": [],
               "regen": [], "lvl": {str(int(l)): 0 for l in LEVELS},
               "per_ge6_frames": 0}
        per_point = []
        for step, games in points:
            st = {k: 0 for k in ("n", "A", "B", "C", "A_ge6", "B_ge6", "C_ge6",
                                 "b_dup", "b_over", "b_unknown", "anomaly")}
            st.update({"a_run_max": 0, "a_run_peak": 0.0, "b_run_max": 0,
                       "lvl": {str(int(l)): 0 for l in LEVELS}})
            for g in games:
                rec = audit_game(g, a.threshold)
                if rec is None:
                    continue
                n = rec["n"]
                st["n"] += n
                st["anomaly"] += rec["anomaly"]
                agg["regen"].extend(rec["regen"])
                for i in range(n):
                    c, p = rec["cls"][i], rec["pre"][i]
                    st[c] += 1
                    if p >= a.threshold:
                        st[c + "_ge6"] += 1
                for i, s in enumerate(rec["sig"]):
                    if rec["cls"][i] == "B":
                        if s["dup"]:
                            st["b_dup"] += 1
                        elif s["over"]:
                            st["b_over"] += 1
                        else:
                            st["b_unknown"] += 1
                for r in rec["runs"]:
                    st["a_run_max"] = max(st["a_run_max"], r["len"])
                    st["a_run_peak"] = max(st["a_run_peak"], r["pre_max"])
                    for l in LEVELS:
                        if r["pre_max"] >= l:
                            st["lvl"][str(int(l))] += 1
                for r in rec["bruns"]:
                    st["b_run_max"] = max(st["b_run_max"], r["len"])
                    agg["bruns"].append(r)
            print(f"{step:>7} {st['n']:>7} {100.0*st['A']/max(1,st['n']):>9.1f}% "
                  f"{100.0*st['B']/max(1,st['n']):>9.1f}% {100.0*st['C']/max(1,st['n']):>9.1f}% "
                  f"{st['A_ge6']:>6} {st['B_ge6']:>6} {st['a_run_max']:>7} "
                  f"{st['a_run_peak']:>7.2f} {st['b_run_max']:>7} "
                  f"{100.0*(st['b_dup']+st['b_over'])/max(1,st['B']):>9.0f}%")
            per_point.append(st)
            for k in ("n", "A", "B", "C", "A_ge6", "B_ge6", "C_ge6",
                      "b_dup", "b_over", "b_unknown", "anomaly"):
                agg[k] += st[k]
            agg["a_run_max"] = max(agg["a_run_max"], st["a_run_max"])
            agg["a_run_peak"] = max(agg["a_run_peak"], st["a_run_peak"])
            agg["b_run_max"] = max(agg["b_run_max"], st["b_run_max"])
            for l in LEVELS:
                agg["lvl"][str(int(l))] += st["lvl"][str(int(l))]

        reg = np.asarray(agg["regen"], dtype=float)
        reg = reg[~np.isnan(reg)]
        med = float(np.median(reg)) if len(reg) else float("nan")
        n = max(1, agg["n"])
        print("-" * 108)
        print(f"[{label}] 帧 {agg['n']} ｜ **A 主动不出牌 {agg['A']} ({100.0*agg['A']/n:.1f}%)** ｜ "
              f"**B 整包被拒 {agg['B']} ({100.0*agg['B']/n:.1f}%)** ｜ "
              f"C 实际出牌 {agg['C']} ({100.0*agg['C']/n:.1f}%)")
        print(f"[{label}] ★ **A ∩ 圣水≥6 = {agg['A_ge6']} 帧**（无假设的『能出却选择不出』）｜ "
              f"B ∩ 圣水≥6 = {agg['B_ge6']} 帧 ｜ C ∩ 圣水≥6 = {agg['C_ge6']} 帧")
        print(f"[{label}] B 的非法性自证：重复槽位 {agg['b_dup']} ｜ 总费>决策圣水 {agg['b_over']} ｜ "
              f"两者都不满足（引擎级/掩码缺口） {agg['b_unknown']} ｜ 最长连续 B = {agg['b_run_max']} 帧")
        print(f"[{label}] A 段（主动不出牌段）：最长 {agg['a_run_max']} 帧 ｜ 段内圣水峰值 max "
              f"{agg['a_run_peak']:.2f} ｜ 峰值≥阈值 的段数："
              + " ｜ ".join(f"≥{int(l)}: {agg['lvl'][str(int(l))]}" for l in LEVELS))
        if len(reg):
            print(f"[{label}] 回费标定（A 类相邻帧差分，n={len(reg)}）：中位 {med:.4f} 圣水/帧 "
                  f"= **{2*med:.4f} 圣水/s** ⇒ 从 0 攒到 Xbow(6 费) 需 **{6.0/max(1e-9, med):.0f} 帧"
                  f"（{6.0/max(1e-9, 2*med):.1f} s）**")
        print(f"[{label}] 仪器自检：A 类却花掉圣水（不可能）的帧 = {agg['anomaly']}（应为 0）")
        summary[label] = {"dir": os.path.relpath(d, _ROOT),
                          "agg": {k: v for k, v in agg.items()
                                  if k not in ("runs", "bruns", "regen")},
                          "regen_median_per_frame": med if len(reg) else None,
                          "per_point": per_point,
                          "b_run_top": sorted(agg["bruns"], key=lambda r: -r["len"])[:8],
                          "a_run_top": sorted(agg["runs"], key=lambda r: -r["pre_max"])[:8]}

    print("=" * 108)
    print("读法：**A∩圣水≥6 = 无假设的『能出却选择不出』**（≥6 时整副卡组都付得起）；")
    print("      **B 是「提交了非法包、整包被拒、白掉一帧」**——它会让圣水被动上涨，**极易被误读成攒费**。")
    print("(warn) 描述性读数（【R3】非判据）。")
    if a.json:
        out = a.json if os.path.isabs(a.json) else os.path.join(_ROOT, a.json)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=1)
        print(f"[saved] {out}")


if __name__ == "__main__":
    main()
