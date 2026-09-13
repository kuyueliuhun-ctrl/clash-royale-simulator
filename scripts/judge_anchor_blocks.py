#!/usr/bin/env python3
"""C1 判据判读器：锚点序列 → 分块 worst/median → 预注册判据（含基线组复算）。

纪律（本仓两次标定错误的教训，见 `docs/d1_league_20k_verdict_2026-09-13.md` §披露）：
**基线列一律由本脚本从磁盘读数计算，禁止手抄**（手抄曾把 F′ 首跑末4均值写成 0.400，
实际 0.588；把 C2 基线"≥0.35 的 run 数"写成 1/3，实际 2/3）。

数据来源：`runs/<run>/solo_state.json` 的 `_controls_history`（`vs == "baseline_rand"`，
逐条带 step）——轻量锚点（`--anchor-every`）与全点写的是同一个字段，故 100k 长跑的
2500 分辨率锚点序列天然可读。旧的 run 若无 state，可用 `--log` 回退解析日志。

口径：
- 锚点 = main vs 固定随机策略（`RAND_ANCHOR_EVAL_SEED=90000`）、40 局/点、1σ≈0.078；
- **块口径**：按 `--block`（缺省 20000 步）切块，块内取最差点 → `worst_b`。
  20k 的 run 自动退化成 1 块（= 全程最差点），与 D1 20k 判读口径一致；
  100k 预注册 = 5 块（9,8,8,8,8 点，`docs/d1_long_100k_prereg_2026-09-13.md` §2）。
- **P1（主判据）**：`mean(worst_b) ≥ 0.10` 且 `min(worst_b) ≥ 0.05`。
- **Q2（次，描述性，不判决）**：`median_末块 - median_首块`，≥+0.10 提示上移、≤-0.10 提示下移。

用法：
    python scripts/judge_anchor_blocks.py --groups                    # 复算历史基线表
    python scripts/judge_anchor_blocks.py --run d1_long_100k           # 判 100k 主判据
    python scripts/judge_anchor_blocks.py --run <run> --per-block-points 9
"""

import argparse
import json
import os
import re
import statistics
import sys

# GBK 控制台/重定向管道兜底（教训同 rl/run_league.py::_force_utf8_stdout）：
# 本脚本输出含中文与数学符号，cp936 下会抛 UnicodeEncodeError 直接崩。
for _s in ("stdout", "stderr"):
    try:
        getattr(sys, _s).reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
RUNS = os.path.join(_ROOT, "src", "clasher_new", "runs")

#: 预注册的对照分组（名字只能在这里改；数值一律由磁盘计算）
D1_GROUP = ["d1_league_20k", "d1_league_20k_r2", "d1_league_20k_r3"]
NOCHANGE_GROUP = ["fprime_20k", "fprime_ev_20k", "gfix_20k"]

P1_MEAN_MIN = 0.10          # mean(worst_b) ≥ 0.10
P1_MIN_MIN = 0.05           # min(worst_b) ≥ 0.05
Q2_BAND = 0.10              # 描述性带宽


def load_anchors_from_state(run):
    """从 solo_state.json 的 _controls_history 取锚点序列 [(step, winrate)]（升序）。"""
    p = os.path.join(RUNS, run, "solo_state.json")
    with open(p, encoding="utf-8") as f:
        st = json.load(f)
    out = []
    for c in st.get("_controls_history") or []:
        if c.get("vs") != "baseline_rand":
            continue
        wr = c.get("winrate")
        if wr is None:
            continue
        out.append((int(c["step"]), float(wr)))
    return sorted(out), st


def load_anchors_from_log(path):
    """回退：从日志解析锚点序列（`eval@<step>` 或 `anchor@<step>` 后的 `vs baseline_rand`）。"""
    out, cur = [], None
    pat_step = re.compile(r"(?:eval|anchor)@(\d+):")
    pat_wr = re.compile(r"vs baseline_rand: .*?([\d.]+)±")
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            m = pat_step.search(ln)
            if m:
                cur = int(m.group(1))
                # 新日志格式把锚点写在同一行
                m2 = pat_wr.search(ln)
                if m2:
                    out.append((cur, float(m2.group(1))))
                continue
            m2 = pat_wr.search(ln)
            if m2 and cur is not None:
                out.append((cur, float(m2.group(1))))
    return sorted(set(out))


def blocks_of(series, per_block):
    """按**锚点序号**连续切块（每块 per_block 点）→ [(块号, 点数, worst, median, step区间)]。

    为什么按序号而不是按 `step//block`：按 step 切会把边界点（如 step=20000 落在
    [20000,40000) 块）单独切成一块，20k 的 run 会变成"8 点块 + 1 点块"，worst 口径
    与 D1 20k 判读（全程 9 点最差）不一致——本脚本第一版就是这么错的，`--groups`
    复算出的 mean(worst) 0.400 与文档 0.192 对不上时才暴露。
    按序号切：9 点/块时 20k run = 1 块（= 全程最差点，与基线同口径），
    100k run 的 41 点 = 5 块（9,8,8,8,8，与预注册一致）。
    """
    rows = []
    for b in range(0, len(series), per_block):
        pts = series[b:b + per_block]
        if not pts:
            continue
        wrs = [w for _, w in pts]
        rows.append({
            "block": len(rows),
            "n": len(wrs),
            "worst": min(wrs),
            "median": statistics.median(wrs),
            "lo": pts[0][0],
            "hi": pts[-1][0],
            "steps": [s for s, _ in pts],
        })
    return rows


def evaluate(series, per_block):
    rows = blocks_of(series, per_block)
    worsts = [r["worst"] for r in rows]
    medians = [r["median"] for r in rows]
    res = {
        "rows": rows,
        "n_points": len(series),
        "global_min": min((w for _, w in series), default=None),
        "mean_worst": (sum(worsts) / len(worsts)) if worsts else None,
        "min_worst": min(worsts) if worsts else None,
        "max_worst": max(worsts) if worsts else None,
        "median_first": medians[0] if medians else None,
        "median_last": medians[-1] if medians else None,
    }
    res["p1_pass"] = (res["mean_worst"] is not None
                      and res["mean_worst"] >= P1_MEAN_MIN
                      and res["min_worst"] >= P1_MIN_MIN)
    res["q2_delta"] = (res["median_last"] - res["median_first"]
                       if medians else None)
    return res


def print_run(run, per_block, series, res, note="", complete=True):
    print(f"\n### {run}{('  ' + note) if note else ''}")
    print(f"锚点点数 {res['n_points']}（{series[0][0]}..{series[-1][0]}）"
          f"  全程最小 {res['global_min']:.3f}"
          if series else f"### {run}: 无锚点数据")
    if not series:
        return
    print("| 块 | step 区间 | 点数 | worst | median |")
    print("|---|---|---|---|---|")
    for r in res["rows"]:
        print(f"| b{r['block'] + 1} | {r['lo']}–{r['hi']} | {r['n']} | "
              f"**{r['worst']:.3f}** | {r['median']:.3f} |")
    print(f"worsts = {[round(r['worst'], 3) for r in res['rows']]} → "
          f"mean {res['mean_worst']:.3f} / min {res['min_worst']:.3f} / "
          f"max {res['max_worst']:.3f}")
    print(f"medians = {[round(r['median'], 3) for r in res['rows']]} → "
          f"末-首 = {res['q2_delta']:+.3f}（描述性："
          f"{'提示上移' if res['q2_delta'] >= Q2_BAND else ('提示下移' if res['q2_delta'] <= -Q2_BAND else '不可判')}）")
    print(f"**P1（mean(worst) ≥ {P1_MEAN_MIN} 且 min(worst) ≥ {P1_MIN_MIN}）= "
          f"{'PASS' if res['p1_pass'] else 'FAIL'}**"
          + ("" if complete else
             "  ← **数据不完整（run 未跑完 / 锚点缺失），此读数不得作为判决**"))


def main():
    ap = argparse.ArgumentParser(description="C1 锚点判据判读器（基线由磁盘计算）")
    ap.add_argument("--run", type=str, default=None, help="runs/<run> 目录名")
    ap.add_argument("--log", type=str, default=None, help="回退：日志文件路径")
    ap.add_argument("--per-block-points", type=int, default=9, dest="per_block_points",
                    help="每块的锚点点数（缺省 9 = 与 20k 基线的点数粒度一致）")
    ap.add_argument("--expect-total", type=int, default=0, dest="expect_total",
                    help="预期总步数（如 100000）：state.total_steps 未达到则标注数据不完整")
    ap.add_argument("--groups", action="store_true",
                    help="复算历史对照基线表（D1 三跑 vs 无变化三跑）")
    a = ap.parse_args()

    if a.groups:
        print(f"# C1 基线表（**由磁盘计算**，每块 {a.per_block_points} 个锚点）")
        summary = {}
        for name, group in (("D1", D1_GROUP), ("NOCHANGE", NOCHANGE_GROUP)):
            worsts, means = [], []
            print(f"\n## {name} 组")
            print("| run | 点数 | 块数 | worsts | mean(worst) | min(worst) | P1 |")
            print("|---|---|---|---|---|---|---|")
            for run in group:
                try:
                    series, _ = load_anchors_from_state(run)
                except (OSError, ValueError):
                    print(f"| {run} | — | — | 数据缺失 | | | |")
                    continue
                if not series:
                    print(f"| {run} | 0 | — | 无锚点 | | | |")
                    continue
                r = evaluate(series, a.per_block_points)
                ws = [round(x["worst"], 3) for x in r["rows"]]
                worsts.append(r["min_worst"])
                means.append(r["mean_worst"])
                print(f"| {run} | {r['n_points']} | {len(r['rows'])} | {ws} | "
                      f"{r['mean_worst']:.3f} | {r['min_worst']:.3f} | "
                      f"{'PASS' if r['p1_pass'] else 'FAIL'} |")
            if worsts:
                summary[name] = {
                    "min": min(worsts), "max": max(worsts),
                    "mean_of_means": sum(means) / len(means),
                }
                print(f"\n{name} 组单跑 min(worst) = "
                      f"{[round(w, 3) for w in worsts]} → 区间 "
                      f"[{min(worsts):.3f}, {max(worsts):.3f}]，"
                      f"组均值 {summary[name]['mean_of_means']:.3f}")
        if "D1" in summary and "NOCHANGE" in summary:
            d, n = summary["D1"], summary["NOCHANGE"]
            overlap = not (d["min"] > n["max"] or n["min"] > d["max"])
            print(f"\n**区间是否重叠：{'是（判据力不足）' if overlap else '否（判据有判别力）'}**"
                  f"  D1[{d['min']:.3f},{d['max']:.3f}] vs "
                  f"NOCHANGE[{n['min']:.3f},{n['max']:.3f}]")
        return 0

    if a.log:
        series = load_anchors_from_log(a.log)
        st = {}
        name = os.path.basename(a.log)
    elif a.run:
        series, st = load_anchors_from_state(a.run)
        name = a.run
    else:
        ap.error("需要 --run 或 --log（或 --groups）")

    res = evaluate(series, a.per_block_points)
    note, complete = "", True
    if a.run and st.get("total_steps") is not None:
        ts = int(st["total_steps"])
        note = f"(state total_steps={ts})"
        if ts < int(getattr(a, "expect_total", 0) or 0) or (series and series[-1][0] < ts):
            complete = False
    print_run(name, a.per_block_points, series, res, note=note, complete=complete)
    return 0


if __name__ == "__main__":
    sys.exit(main())
