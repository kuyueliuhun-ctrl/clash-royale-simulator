# -*- coding: utf-8 -*-
"""平局解剖 + 新旧裁决口径**配对**对照（只读，数据来自已落盘回放）。

数据来源：`runs/<run>/replays/league_*.pkl`（`LeagueGameRecorder` 落盘；schema=3）。
每局含：`winner`（当时结算结果）+ 每帧 `t / towers0 / towers1 / crown0 / crown1`。
⇒ 可以在**同一批真实对局**上同时算出旧口径与新口径的标签（零重算、零新偏置）。

口径：
  - **旧**（2026-09-17 之前）：皇冠（被拆塔数）优先；皇冠平 → 存活塔**最低血量百分比**小者输；
    百分比完全相等 → 平局。`--margin>0` 时追加 C′ 的"细差 < margin 记平局"。
  - **新**（2026-09-17 用户指定）：皇冠优先；皇冠平 → **三塔血量合计**多者胜；完全相等 → 平局。
  - 各塔"最大血"以该局**第 0 帧**为基准（开局满血；塔兵改上限的极端情形会失真，见预注册 §3）。

复现校验：脚本用"旧口径"重算全部标签，必须与回放里记录的 `winner` **逐局一致**——
一致才说明本脚本的重算可信（这是新口径读数的前提）。

用法：
    cd src/clasher_new && PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe \
        ../../scripts/analyze_draw_anatomy.py --run runs/demo20k
"""
from __future__ import annotations

import argparse
import glob
import io
import os
import pickle
import sys
from collections import Counter

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src", "clasher_new")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _label_old(tot_old, min0, min1, lost0, lost1, margin):
    """旧口径（重算；与 timeout_winner / settle_stall_from_counts 同语义）。"""
    if lost1 > lost0:
        return 0
    if lost0 > lost1:
        return 1
    if min0 is None or min1 is None:
        return None
    if margin and margin > 0:
        if min0 > min1 + margin:
            return 0
        if min1 > min0 + margin:
            return 1
        return None
    if min0 > min1 + 1e-9:
        return 0
    if min1 > min0 + 1e-9:
        return 1
    return None


def _label_new(hp0, hp1, lost0, lost1):
    """新口径（2026-09-17）：皇冠优先 → 三塔血量合计多者胜 → 完全相等平局。"""
    if lost1 > lost0:
        return 0
    if lost0 > lost1:
        return 1
    if hp0 > hp1 + 1e-9:
        return 0
    if hp1 > hp0 + 1e-9:
        return 1
    return None


def _hits(pred, truth):
    return sum(1 for a, b in zip(pred, truth) if (a is None and b is None) or a == b)


def analyze(path, margin):
    d = pickle.load(io.open(path, "rb"))
    games = d["games"]
    rows = []
    for g in games:
        fr = g["frames"]
        if not fr:
            continue
        f0, fl = fr[0], fr[-1]
        t0 = [float(x) for x in f0["towers0"]]
        t1 = [float(x) for x in f0["towers1"]]
        h0 = [float(x) for x in fl["towers0"]]
        h1 = [float(x) for x in fl["towers1"]]
        hp0, hp1 = sum(h0), sum(h1)
        # 最低血量百分比（用开局血量当各塔最大血）
        p0 = [h / m for h, m in zip(h0, t0) if m > 0]
        p1 = [h / m for h, m in zip(h1, t1) if m > 0]
        min0 = min(p0) if p0 else None
        min1 = min(p1) if p1 else None
        lost0, lost1 = int(fl["crown0"]), int(fl["crown1"])
        t = float(fl["t"])
        zero_dmg = (h0 == t0) and (h1 == t1)
        rows.append({
            "t": t, "n": len(fr), "recorded": g["winner"],
            "old": _label_old(None, min0, min1, lost0, lost1, 0.0),   # 评估口径（复现校验基准）
            "new": _label_new(hp0, hp1, lost0, lost1),
            "old_margin": _label_old(None, min0, min1, lost0, lost1, 0.05),
            "hp0": hp0, "hp1": hp1, "min0": min0, "min1": min1,
            "lost0": lost0, "lost1": lost1, "zero_dmg": zero_dmg,
        })
    return rows


def table(rows, key):
    c = Counter(("W" if r[key] == 0 else "L" if r[key] == 1 else "D") for r in rows)
    return c["W"], c["L"], c["D"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/demo20k")
    ap.add_argument("--margin", type=float, default=0.05,
                    help="训练侧 C′ 边距（仅用于第三条诊断列；复现校验用的是评估口径=无边距）")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.run, "replays", "league_*.pkl")))
    if not files:
        print(f"[FAIL] 没找到回放: {a.run}/replays/league_*.pkl")
        return 1

    print("=" * 108)
    print(f"平局解剖 · {a.run} · 回放 {len(files)} 个（每个 = 一个评估点的 40 局）")
    print("口径：旧=最低塔血百分比；旧+margin=叠加 C′ 细差<margin 判平；新=三塔血量合计（2026-09-17）")
    print("=" * 108)
    print(f"{'评估点':>10} {'局':>4} {'t中位':>7} {'引擎300s':>9} {'零塔损截断':>11} {'打满上限':>9} {'其他早终':>9} "
          f"{'旧(评估)W/L/D':>13} {'旧+margin':>12} {'新 W/L/D':>12} {'旧标签复现':>10}")
    all_rows = []
    for p in files:
        rows = analyze(p, a.margin)
        all_rows += rows
        tag = os.path.basename(p).replace("league_", "").replace(".pkl", "")
        ts = sorted(r["t"] for r in rows)
        med = ts[len(ts) // 2]
        n_eng = sum(1 for r in rows if r["t"] >= 300.0)
        n_max = sum(1 for r in rows if r["n"] >= 360 and r["t"] < 300.0)
        # ⚠️ 2026-09-17 更正：早停的**充要签名**是"全程零塔损"(zero_dmg) —— 零塔损局的结束
        # 只可能来自 (a) 僵局早停 (b) 打满 max_ep_steps；而"n<360 且 t<300"对**任何**未到上限的
        # 对局都成立（含正常的拆王塔结束）⇒ 旧标签 n_stall 是**度量假象**（曾给出 96.8%）。
        n_stall = sum(1 for r in rows if r["zero_dmg"] and r["n"] < 360)
        n_other = len(rows) - n_eng - n_max - n_stall
        o, om, nw = table(rows, "old"), table(rows, "old_margin"), table(rows, "new")
        ok = _hits([r["old"] for r in rows], [r["recorded"] for r in rows])
        d2wl = sum(1 for r in rows if r["old"] is None and r["new"] is not None)
        print(f"{tag:>10} {len(rows):>4} {med:>7.1f} {n_eng:>9} {n_stall:>11} {n_max:>9} {n_other:>9} "
              f"{str(o):>12} {str(om):>12} {str(nw):>12} {ok:>7}/{len(rows)}   D→胜负 {d2wl:>3}")

    print("-" * 108)
    o, om, nw = table(all_rows, "old"), table(all_rows, "old_margin"), table(all_rows, "new")
    ok = _hits([r["old"] for r in all_rows], [r["recorded"] for r in all_rows])
    print(f"{'合计':>10} {len(all_rows):>4} {'':>7} "
          f"{sum(1 for r in all_rows if r['t'] >= 300.0):>9} "
          f"{sum(1 for r in all_rows if r['zero_dmg'] and r['n'] < 360):>11} "
          f"{sum(1 for r in all_rows if r['n'] >= 360 and r['t'] < 300.0):>9} "
          f"{sum(1 for r in all_rows if not (r['t'] >= 300.0) and r['n'] < 360 and not r['zero_dmg']):>9} "
          f"{str(o):>12} {str(om):>12} {str(nw):>12} {ok:>7}/{len(all_rows)}")
    print(f"\n旧口径复现校验：{ok}/{len(all_rows)} 与回放记录的 winner 一致"
          f"{'（✅ 可信）' if ok == len(all_rows) else '（❌ 不一致 ⇒ 下面的新口径读数不可信）'}")

    # 平局解剖
    draws_old = [r for r in all_rows if r["old"] is None]
    print(f"\n{'=' * 108}\n旧口径平局解剖（{len(draws_old)} 局）：")
    cat_a = [r for r in draws_old if r["zero_dmg"]]
    cat_b = [r for r in draws_old if not r["zero_dmg"] and abs(r["hp0"] - r["hp1"]) <= 1e-9]
    cat_c = [r for r in draws_old if not r["zero_dmg"] and abs(r["hp0"] - r["hp1"]) > 1e-9]
    print(f"  (a) **全程零塔损**（双方塔血都没掉过）      : {len(cat_a):>3} 局 → 新口径仍判平（塔血确实相同）")
    print(f"  (b) 有塔损、且三塔合计恰好相等            : {len(cat_b):>3} 局 → 新口径仍判平")
    print(f"  (c) 有塔损、合计不等（旧口径漏判的胜负）   : {len(cat_c):>3} 局 → 新口径改判")
    if cat_a:
        print(f"      (a) 的 t 中位 {sorted(r['t'] for r in cat_a)[len(cat_a)//2]:.1f}s"
              f"（僵局早停），帧数中位 {sorted(r['n'] for r in cat_a)[len(cat_a)//2]}")
    if cat_c:
        d = sorted(abs(r["hp0"] - r["hp1"]) for r in cat_c)
        print(f"      (c) 的塔血合计差：中位 {d[len(d)//2]:.0f} HP，最大 {d[-1]:.0f} HP")

    # 停用僵局早停后的帧数代价（2026-09-17 用户拍板：零塔损局必须打满）
    print(f"\n{'=' * 108}\n停用僵局早停后的**帧数代价**（估算，标注上下界）：")
    print("  ⚠️ 2026-09-17 更正：本节的判据原为 `n<360 and t<300`，那是**度量假象**"
          "（对任何未到上限的对局都成立，含正常拆塔结束）\n"
          "     曾据此得出「96.8% 的对局被早停、帧数 +96%~+222%」。真判据 = **全程零塔损**"
          "（零塔损局只能以早停或打满上限结束）。")
    stall_games = [r for r in all_rows if r["zero_dmg"] and r["n"] < 360]
    cur = sum(r["n"] for r in all_rows)
    hi = sum(600 if (r["zero_dmg"] and r["n"] < 360) else r["n"] for r in all_rows)
    lo = sum(max(r["n"], 360) if (r["zero_dmg"] and r["n"] < 360) else r["n"] for r in all_rows)
    print(f"  被早停的对局 = {len(stall_games)}/{len(all_rows)}（{len(stall_games)/len(all_rows):.1%}）"
          f"，其帧数中位 {sorted(r['n'] for r in stall_games)[len(stall_games)//2] if stall_games else 0}")
    print(f"  总帧数：现状 {cur} → 停用后【下界】{lo}（+{(lo-cur)/cur:.1%}）"
          f" / 【上界】{hi}（+{(hi-cur)/cur:.1%}，假设这些局仍无人破塔 ⇒ 打满 600 帧）")
    print(f"  ⇒ 评估墙钟近似按同比例上升；同 `--total-steps 20000` 对应的**局数**从 "
          f"{20000*len(all_rows)/cur:.0f} 降到 {20000*len(all_rows)/hi:.0f}~{20000*len(all_rows)/lo:.0f}")
    print("  交叉校验（独立证据）：`scripts/_probe_eval_frame_ab.py` 比对两个 run 的 eval@0（同初始权重）"
          "⇒ 逐局帧数 40/40 相同\n      —— 因为 eval@0 的 40 局**零塔损局数为 0**，早停从未触发 ⇒ 与本节口径一致。")

    # 配对变化
    print(f"\n{'=' * 108}\n配对变化（同一批对局，只换标签口径）：")
    ch = Counter()
    for r in all_rows:
        ch[(("W" if r["old"] == 0 else "L" if r["old"] == 1 else "D"),
            ("W" if r["new"] == 0 else "L" if r["new"] == 1 else "D"))] += 1
    for (a1, b1), n in sorted(ch.items(), key=lambda kv: -kv[1]):
        mark = "  ← 变了" if a1 != b1 else ""
        print(f"  旧 {a1} → 新 {b1}: {n:>3} 局{mark}")
    print(f"  平局率：旧 {o[2]}/{len(all_rows)} = {o[2]/len(all_rows):.1%}"
          f" → 新 {nw[2]}/{len(all_rows)} = {nw[2]/len(all_rows):.1%}"
          f"（降 {o[2]-nw[2]} 局）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
