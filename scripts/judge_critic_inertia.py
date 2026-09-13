"""critic 惰性检验判读器（预注册 docs/critic_inertia_prereg_2026-09-13.md）。

从**原始日志**复算全部统计量（不信任 run_state.json 里的汇总，R4），按预注册的
P1/P2/P3 阈值输出判决，并把次级锚点对比一并算出来（描述性）。

用法（在 src/clasher_new 下）：
  PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe \
      ../../scripts/judge_critic_inertia.py --log ../../docs/train_critic_inert_probe_20k.log \
      --run runs/critic_inert_probe_20k
"""

import argparse
import json
import os
import re
import sys

try:  # GBK 控制台兜底（同仓库其它脚本纪律）
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src", "clasher_new")

# 预注册阈值（写死，不许现编）
P1_COS = 0.99      # grad_cos 中位 >= 0.99
P1_RESID = 0.10    # resid_frac_norm 中位 <= 0.10
P2_COS = 0.95      # grad_cos 中位 <= 0.95
P2_RESID = 0.30    # resid_frac_norm 中位 >= 0.30
LEVEL_NOTE = 0.10  # level_shift_norm 中位 > 0.10 ⇒ Layer 2 归因受限

_LINE = re.compile(
    r"\[solo advinert (?P<step>\d+)\]\s+corr=(?P<corr>[-+0-9.eE]+)\s+"
    r"resid=(?P<resid>[-+0-9.eE]+)\s+resid_norm=(?P<resid_norm>[-+0-9.eE]+)\s+"
    r"lvl_shift=(?P<lvl>[-+0-9.eE]+)\s+level_gap=(?P<gap>[-+0-9.eE]+)\s+"
    r"grad_cos=(?P<cos>[-+0-9.eE]+)\s+gnorm_ratio=(?P<gr>[-+0-9.eE]+)")

# 基线（D1 20k 三跑）锚点序列 —— 脚本复算，见预注册 §5
_BASE_RUNS = ["d1_league_20k", "d1_league_20k_r2", "d1_league_20k_r3"]


def _median(v):
    v = sorted(x for x in v if x is not None)
    if not v:
        return None
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def _fmt(x, nd=4):
    return "None" if x is None else f"{x:.{nd}f}"


def parse_log(path):
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            m = _LINE.search(ln)
            if m:
                rows.append({k: (int(v) if k == "step" else float(v))
                             for k, v in m.groupdict().items()})
    return rows


def anchor_series(run_dir):
    p = os.path.join(run_dir, "solo_state.json")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        st = json.load(f)
    return [(int(c["step"]), float(c["winrate"]))
            for c in (st.get("_controls_history") or [])
            if c.get("vs") == "baseline_rand"]


def block_worst(series, n_blocks=5):
    """按锚点**序号**切块取最差（与 scripts/judge_anchor_blocks.py 同口径：
    近等分块，块数 = min(n_blocks, 点数)；**不是**固定每块 9 点）。"""
    if not series:
        return []
    n = len(series)
    k = min(int(n_blocks), n)
    base, rem = divmod(n, k)
    out, i = [], 0
    for b in range(k):
        size = base + (1 if b < rem else 0)
        chunk = series[i:i + size]
        i += size
        if chunk:
            out.append(min(w for _, w in chunk))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--run", required=True, help="探针 run 目录（相对 src/clasher_new）")
    ap.add_argument("--runs-root", default=os.path.join(_SRC, "runs"))
    args = ap.parse_args()

    rows = parse_log(args.log)
    print("=" * 78)
    print("critic 惰性检验判读（预注册 docs/critic_inertia_prereg_2026-09-13.md）")
    print("=" * 78)
    print(f"日志: {args.log}")
    if not rows:
        print("!! 没有解析到任何 [solo advinert ...] 行 —— 探针没生效或日志路径错")
        return 2

    cos = [r["cos"] for r in rows]
    resid_norm = [r["resid_norm"] for r in rows]
    corr = [r["corr"] for r in rows]
    lvl = [r["lvl"] for r in rows]
    gap = [r["gap"] for r in rows]
    gr = [r["gr"] for r in rows]

    print(f"\n诊断更新数 = {len(rows)}（覆盖 step {rows[0]['step']} … {rows[-1]['step']}）")
    print("\n--- 逐点序列（这是最值得看的部分：critic 何时停止起作用）---")
    print(f"{'step':>7} {'corr':>8} {'resid_norm':>11} {'lvl_shift':>10} "
          f"{'level_gap':>10} {'grad_cos':>10} {'gnorm_ratio':>11}")
    for r in rows:
        print(f"{r['step']:>7} {r['corr']:>8.4f} {r['resid_norm']:>11.4f} "
              f"{r['lvl']:>10.4f} {r['gap']:>10.4f} {r['cos']:>+10.4f} {r['gr']:>11.3f}")

    half = len(rows) // 2
    late = rows[half:] or rows
    print("\n--- 汇总（脚本复算）---")
    print(f"  grad_cos        中位 {_fmt(_median(cos), 5)}   "
          f"min {_fmt(min(cos), 5)}   max {_fmt(max(cos), 5)}   "
          f"<0.99 的点数 {sum(1 for c in cos if c < P1_COS)}/{len(cos)}")
    print(f"  resid_frac_norm 中位 {_fmt(_median(resid_norm))}   max {_fmt(max(resid_norm))}")
    print(f"  corr            中位 {_fmt(_median(corr), 5)}   min {_fmt(min(corr), 5)}")
    print(f"  level_shift_norm 中位 {_fmt(_median(lvl))}   max {_fmt(max(lvl))}")
    print(f"  level_gap       中位 {_fmt(_median(gap))}   max {_fmt(max(gap))}")
    print(f"  gnorm_ratio     中位 {_fmt(_median(gr))}")
    print(f"  [后半段] step>={late[0]['step']}: grad_cos 中位 {_fmt(_median([r['cos'] for r in late]), 5)}、"
          f"resid_norm 中位 {_fmt(_median([r['resid_norm'] for r in late]))}")

    m_cos, m_resid = _median(cos), _median(resid_norm)
    print("\n--- 预注册判决 ---")
    if m_cos is not None and m_resid is not None:
        if m_cos >= P1_COS and m_resid <= P1_RESID:
            verdict = "P1 成立 ⇒ critic 惰性（换标量后策略更新方向不变）"
        elif m_cos <= P2_COS or m_resid >= P2_RESID:
            verdict = "P2 成立 ⇒ critic 在工作（因果链作废，O2 退回未解）"
        else:
            verdict = "P3 灰区 ⇒ 部分惰性，不判决（需更长 run / 更多诊断点）"
        print(f"  P1 阈值: grad_cos >= {P1_COS} 且 resid_norm <= {P1_RESID}"
              f"   实测 ({_fmt(m_cos, 5)}, {_fmt(m_resid)})")
        print(f"  P2 阈值: grad_cos <= {P2_COS} 或 resid_norm >= {P2_RESID}")
        print(f"  ⇒ {verdict}")
    if _median(lvl) is not None and _median(lvl) > LEVEL_NOTE:
        print(f"  [分解项] level_shift_norm 中位 {_fmt(_median(lvl))} > {LEVEL_NOTE}"
              f" ⇒ 若做 Layer 2，'无差别'不得全归因于状态依赖（水平偏移也是干预的一部分）")

    # 次级：锚点对比（描述性，不判决）
    mine = anchor_series(os.path.join(args.runs_root, args.run))
    print("\n--- 次级（描述性，R5：n=1/臂 判不了零假设）---")
    if not mine:
        print("  本 run 无锚点记录")
    else:
        print(f"  本 run 锚点序列: " + " ".join(f"{s//1000}k:{w:.3f}" for s, w in mine))
        print(f"  本 run min={min(w for _, w in mine):.3f}  "
              f"mean={sum(w for _, w in mine)/len(mine):.4f}  "
              f"块 worst={[round(x,3) for x in block_worst(mine)]}")
    base = []
    for b in _BASE_RUNS:
        s = anchor_series(os.path.join(args.runs_root, b))
        if s:
            base.append((b, s))
    if base:
        mins = [min(w for _, w in s) for _, s in base]
        ends = [s[-1][1] for _, s in base]
        means = [sum(w for _, w in s) / len(s) for _, s in base]
        print(f"  基线 3 跑（{', '.join(b for b, _ in base)}）: "
              f"min={[round(x,3) for x in mins]}  末点={[round(x,3) for x in ends]}  "
              f"均值={[round(x,4) for x in means]}")
        print("  ⇒ 若本 run min 落在基线 min 区间内、末点落在末点区间内，与'critic 惰性'一致；"
              "若出现整段归零(min<=0.025)或显著改善，按预注册 §2 次级条款读。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
