#!/usr/bin/env python3
"""长 run 诊断汇总器：日志 + solo_state.json + gates.json → 一段 markdown（判读用）。

为什么要脚本化（本仓教训）：本轮会话两次"手抄基线/手抄统计量"都抄错（F′ 末4均值
0.400 vs 实际 0.588；C2 基线 1/3 vs 实际 2/3）。判读材料的**每一个数字**都必须由脚本
从磁盘产出，人只做解释。

输出（全部由磁盘读数计算）：
1. 全点评估表：step / 胜率±SE / EV(池化) / EVb / h_std / n_abs / vstd-rstd / 累计局数
2. 每个全点的三条对照（baseline0 / baseline_prev / baseline_rand）
3. 锚点序列（含 <0.35 警报点）+ C1 分块判据（复用 judge_anchor_blocks）
4. 对手池组成随时间（本目录/外部）+ 最终 kind_counts
5. gates.json
6. 异常计数：WinError 1455 / Traceback / 降级 / 锚点警报
7. 训练循环墙钟

用法：
    python scripts/summarize_solo_run.py --log docs/train_d1_long_100k.log \
        --run d1_long_100k --expect-total 100000 > docs/_d1_long_100k_diag.md
"""

import argparse
import ast
import json
import os
import re
import sys

for _s in ("stdout", "stderr"):
    try:
        getattr(sys, _s).reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
RUNS = os.path.join(_ROOT, "src", "clasher_new", "runs")

#: 与 train_solo.RAND_ANCHOR_WARN_FLOOR 对齐（旧日志的锚点报警行不带 step 关联）
ANCHOR_WARN_FLOOR = 0.35

import judge_anchor_blocks as jab   # noqa: E402  （复用锚点读数/分块判据）

_RE_EVAL = re.compile(
    r"\[solo\] eval@(\d+): 胜率 ([\d.]+)±([\d.]+) \((\d+)W/(\d+)L/(\d+)D, (\d+)局\)"
    r" mean_reward=([-\d.]+|nan) EV=([-\d.]+\S*|None)（池化）(?: EVb=([-\d.]+\S*|None))?"
    r"(?: h_std=([\d.eE+-]+|n/a))?(?: n_abs=([\d.eE+-]+|None))?"
    r"(?: vstd/rstd=([\d.eE+-]+|None))? 累计局数=(\d+)")
_RE_CTRL = re.compile(r"\[solo\]\s+vs (\w+): 胜率 ([\d.]+)±([\d.]+)")
_RE_ANCHOR = re.compile(r"\[solo\] anchor@(\d+): vs baseline_rand 胜率 ([\d.]+)±([\d.]+)")
_RE_ANCHOR_WARN = re.compile(r"绝对强度警报 @step (\d+): (.*)")
_RE_POOL = re.compile(r"对手池刷新 @step (\d+): hist ckpts=(\d+)（本目录 (\d+)，新增 (\d+)）"
                      r" \| 累计对手局 (\{.*\})")
_RE_LOOP = re.compile(r"训练循环耗时 ([\d.]+)s")
_RE_RATE = re.compile(r"\[solo\] 评估节奏: (.*)")


def _parse_log(path):
    d = {"evals": [], "controls": {}, "anchors": [], "anchor_warns": [], "pools": [],
         "loop_s": None, "cadence": None, "counts": {"1455": 0, "traceback": 0,
                                                    "degrade": 0, "error": 0}}
    cur_step = None
    ctrl_buf = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            m = _RE_EVAL.search(ln)
            if m:
                if cur_step is not None:
                    d["controls"][cur_step] = list(ctrl_buf)
                ctrl_buf = []
                cur_step = int(m.group(1))
                d["evals"].append({
                    "step": cur_step, "winrate": float(m.group(2)), "se": float(m.group(3)),
                    "w": int(m.group(4)), "l": int(m.group(5)), "dr": int(m.group(6)),
                    "games": int(m.group(7)), "mean_reward": m.group(8),
                    "ev": m.group(9), "evb": m.group(10), "h_std": m.group(11),
                    "n_abs": m.group(12), "vstd_rstd": m.group(13),
                    "cum_games": int(m.group(14)),
                })
                continue
            m = _RE_CTRL.search(ln)
            if m and cur_step is not None:
                ctrl_buf.append((m.group(1), float(m.group(2)), float(m.group(3))))
                continue
            m = _RE_ANCHOR.search(ln)
            if m:
                d["anchors"].append((int(m.group(1)), float(m.group(2))))
                continue
            m = _RE_ANCHOR_WARN.search(ln)
            if m:
                d["anchor_warns"].append((int(m.group(1)), m.group(2).strip()))
                continue
            m = _RE_POOL.search(ln)
            if m:
                d["pools"].append({"step": int(m.group(1)), "n": int(m.group(2)),
                                   "own": int(m.group(3)), "added": int(m.group(4)),
                                   # train_solo 打的是 Python dict repr（单引号），json.loads 会炸 → ast 安全解析
                                   "kinds": ast.literal_eval(m.group(5))})
                continue
            m = _RE_LOOP.search(ln)
            if m:
                d["loop_s"] = float(m.group(1))
                continue
            m = _RE_RATE.search(ln)
            if m:
                d["cadence"] = m.group(1).strip()
                continue
            if "1455" in ln:
                d["counts"]["1455"] += 1
            if "Traceback" in ln:
                d["counts"]["traceback"] += 1
            if "降级" in ln:
                d["counts"]["degrade"] += 1
            if "Error" in ln or "错误" in ln:
                d["counts"]["error"] += 1
    if cur_step is not None:
        d["controls"][cur_step] = list(ctrl_buf)
    return d


def main():
    ap = argparse.ArgumentParser(description="长 run 诊断汇总（markdown）")
    ap.add_argument("--log", required=True)
    ap.add_argument("--run", default=None, help="runs/<run>（读 solo_state/gates/锚点序列）")
    ap.add_argument("--expect-total", type=int, default=0, dest="expect_total")
    ap.add_argument("--per-block-points", type=int, default=9, dest="per_block_points")
    a = ap.parse_args()

    d = _parse_log(a.log)
    if not d["anchors"] and d["controls"]:
        # 兼容旧日志：锚点写在控制行里（新架构的轻点才打 anchor@<step>）
        d["anchors"] = [(st, wr) for st in sorted(d["controls"])
                        for k, wr, _ in d["controls"][st] if k == "baseline_rand"]
    print(f"# 长 run 诊断汇总：`{a.log}`\n")
    if d["cadence"]:
        print(f"- 评估节奏：{d['cadence']}")
    print(f"- 训练循环墙钟：{d['loop_s']}s" if d["loop_s"] else "- 训练循环墙钟：（未结束）")
    print(f"- 异常计数：WinError1455={d['counts']['1455']} / Traceback="
          f"{d['counts']['traceback']} / 降级={d['counts']['degrade']} / "
          f"含'Error|错误'行={d['counts']['error']}")
    print(f"- 全点数 {len(d['evals'])} / 锚点数 {len(d['anchors'])} / "
          f"锚点警报 {len(d['anchor_warns'])}\n")

    print("## 全点评估\n")
    print("| step | 胜率±SE | W/L/D | EV(池化) | EVb | h_std | n_abs | vstd/rstd | 累计局数 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for e in d["evals"]:
        fmt = lambda v: "—" if v in (None, "None", "n/a") else v  # noqa: E731
        print(f"| {e['step']} | {e['winrate']:.3f}±{e['se']:.3f} | "
              f"{e['w']}/{e['l']}/{e['dr']} | {fmt(e['ev'])} | {fmt(e['evb'])} | "
              f"{fmt(e['h_std'])} | {fmt(e['n_abs'])} | {fmt(e['vstd_rstd'])} | "
              f"{e['cum_games']} |")

    print("\n## 对照（每全点三条）\n")
    print("| step | baseline0 | baseline_prev | baseline_rand |")
    print("|---|---|---|---|")
    for step in sorted(d["controls"]):
        row = {k: v for k, v, _ in d["controls"][step]}
        se = {k: s for k, _, s in d["controls"][step]}
        cells = []
        for k in ("baseline0", "baseline_prev", "baseline_rand"):
            cells.append(f"{row[k]:.3f}±{se[k]:.3f}" if k in row else "—")
        print(f"| {step} | " + " | ".join(cells) + " |")

    print("\n## 锚点序列（C1 判据原料）\n")
    series = sorted(set(d["anchors"]))
    if a.run:
        try:
            s2, _st = jab.load_anchors_from_state(a.run)
            if len(s2) >= len(series):
                series = s2      # state 更全（轻点也在里面）
        except (OSError, ValueError):
            pass
    if series:
        warns = {s for s, _ in d["anchor_warns"]}
        # 旧日志的锚点报警没带 step 关联，用阈值直接标（<0.35 即报警线）
        warns |= {st for st, wr in series if wr < ANCHOR_WARN_FLOOR}
        print("| step | 锚点胜率 | <0.35 警报 |")
        print("|---|---|---|")
        for step, wr in series:
            print(f"| {step} | {wr:.3f} | {'WARN' if step in warns else ''} |")
        res = jab.evaluate(series, a.per_block_points)
        complete = True
        if a.expect_total and (series[-1][0] < a.expect_total):
            complete = False
        jab.print_run(f"{a.run or 'log'}", a.per_block_points, series, res,
                      note=f"(锚点 {len(series)} 点)", complete=complete)
    else:
        print("（无锚点数据）")

    print("\n## 对手池\n")
    if d["pools"]:
        print("| step | hist ckpts | 本目录 | 新增 | 累计对手局 |")
        print("|---|---|---|---|---|")
        for p in d["pools"]:
            print(f"| {p['step']} | {p['n']} | {p['own']} | {p['added']} | "
                  f"{json.dumps(p['kinds'], ensure_ascii=False)} |")

    if a.run:
        gp = os.path.join(RUNS, a.run, "gates.json")
        if os.path.isfile(gp):
            with open(gp, encoding="utf-8") as f:
                g = json.load(f)
            print(f"\n## gates.json\n\n- ok = {g.get('ok')}\n"
                  f"- baseline = {json.dumps(g.get('baseline'), ensure_ascii=False)}\n"
                  f"- checks = {json.dumps(g.get('checks'), ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
