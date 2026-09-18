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

# T1-1b 补齐（2026-09-19）：原先这里是**手写**的 UTF-8 兜底块（只处理 stdout、且不处理 stderr
# ⇒ traceback 在 GBK 下仍是乱码）。现收敛到 T1-1 的**单一实现**（两路 + errors='replace'）。
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                                 "src", "clasher_new"))
from rl.io_bootstrap import force_utf8_stdout  # noqa: E402
force_utf8_stdout()

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


def mad(v, med=None):
    """中位绝对偏差 —— **原始 MAD**（不乘 1.4826）。

    与 `engagement_trade_prereg_2026-09-18.md` §11.13.2 的「中位 ± 3×MAD」**字面一致**：
    该节写的是 MAD，不是 σ 的一致估计（1.4826×MAD）。此处不做换算，并在输出里显式标注，
    避免读者按 σ 解读（【R10】：口径含糊时写清口径，不留歧义）。
    """
    v = [x for x in v if x is not None]
    if not v:
        return None
    if med is None:
        med = _median(v)
    return _median([abs(x - med) for x in v])


_WITHIN_KEYS = ("cos", "resid_norm", "corr", "lvl", "gap", "gr")


def within_run_baseline(rows, n_base=100, k=3.0, keys=_WITHIN_KEYS):
    """§11.13.2 的**机制层主判据**：本 run 自己**前 `n_base` 次诊断更新**的中位 ± k×MAD。

    为什么用「run 自己的前段」当对照（而不是跨 run 基线）：『R15』『R16』—— 阈值必须在**本实验
    自己的量纲**上标定，且不许用单次观测；『R5』—— 跨 run 数字不可直接比。故对照 = 同一次训练
    的前段自身（程序写死，数值由本函数**脚本复算**，【R4】禁手抄）。

    ⚠️ MAD 可能为 0（前段该指标恒定）⇒ 带退化成一个点。此时 `degenerate=True`，
    后续任何偏离都算越界；调用方**不得**据此判显著，只能记为「带退化，不可判」。

    `keys`（2026-09-18 新增，**向后兼容**）：要统计的指标键名，默认 = §11.13.2 写死的六项。
    加这个形参是为了让 `scripts/health_curve.py`（策略熵 / 价值损失）**复用同一个实现**，
    而不是照抄一份公式出来（【R17】不得出现两套口径）。传自定义 `keys` 时，`rows` 需是对应
    形状的 dict 列表（键名即 `keys`）。默认调用行为**逐位不变**。
    """
    if not rows:
        return {}
    nb = min(int(n_base), len(rows))
    base_rows, later = rows[:nb], rows[nb:]
    out = {"n_base": nb, "n_later": len(later), "k": float(k), "metrics": {}}
    for key in keys:
        bv = [r[key] for r in base_rows]
        med = _median(bv)
        m = mad(bv, med)
        if med is None:
            lo = hi = None
        else:
            lo, hi = med - k * (m or 0.0), med + k * (m or 0.0)
        lv = [r[key] for r in later]
        inside = None
        if later and lo is not None:
            inside = sum(1 for x in lv if lo <= x <= hi)
        out["metrics"][key] = {
            "base_median": med, "base_mad": m, "lo": lo, "hi": hi,
            "degenerate": (m == 0.0),
            "later_median": (_median(lv) if lv else None),
            "later_inside": inside, "later_n": len(lv)}
    return out


def print_within_run(w):
    """打印 §11.13.2 的 within-run 对照块。"""
    if not w:
        print("\n--- 机制层对照（§11.13.2 within-run）：无诊断点，不可算 ---")
        return
    print(f"\n--- 机制层对照（§11.13.2；【R4】脚本复算）---")
    print(f"  基线窗口 = 本 run 前 {w['n_base']} 次诊断更新；比较窗口 = 其后 {w['n_later']} 次")
    print(f"  带定义 = 基线中位 ± {w['k']:g} × MAD（**原始 MAD**，未乘 1.4826）")
    print(f"  {'指标':<14}{'基线中位':>12}{'基线MAD':>12}{'带下':>12}{'带上':>12}"
          f"{'后续中位':>12}{'带内':>10}")
    _NAME = {"cos": "grad_cos", "resid_norm": "resid_norm", "corr": "corr",
             "lvl": "level_shift", "gap": "level_gap", "gr": "gnorm_ratio"}
    for key, s in w["metrics"].items():
        if s["base_median"] is None:
            continue
        ins = ("n/a" if s["later_inside"] is None
               else f"{s['later_inside']}/{s['later_n']}")
        if s["degenerate"]:
            ins += " ⚠退化"
        print(f"  {_NAME.get(key, key):<14}{_fmt(s['base_median'], 5):>12}"
              f"{_fmt(s['base_mad'], 5):>12}{_fmt(s['lo'], 5):>12}{_fmt(s['hi'], 5):>12}"
              f"{_fmt(s['later_median'], 5):>12}{ins:>10}")


def _selftest():
    """本判读器 within-run 对照的回归测试（【R8】：改判读逻辑必须配测试）。

    合成数据，零引擎依赖：
      ① MAD 用**原始**口径（不乘 1.4826）；
      ② 恒定前段 ⇒ degenerate=True 且带退化；
      ③ 后续整体平移 4×MAD ⇒ 全部越界；
      ④ n_base 大于总点数 ⇒ 比较窗口为空，不报错。
    """
    ok = 0

    def _rows(vals):
        return [{"step": i + 1, "cos": v, "resid_norm": v, "corr": v,
                 "lvl": v, "gap": v, "gr": v} for i, v in enumerate(vals)]

    # ① 原始 MAD：数据 [1,2,3,4,100] 中位 3，|x-3| = [2,1,0,1,97] ⇒ 中位 1
    assert mad([1, 2, 3, 4, 100]) == 1.0, mad([1, 2, 3, 4, 100])
    ok += 1
    # ② 恒定前段
    w = within_run_baseline(_rows([0.5] * 10 + [0.0, 1.0]), n_base=10, k=3.0)
    assert w["metrics"]["cos"]["base_mad"] == 0.0
    assert w["metrics"]["cos"]["degenerate"] is True
    ok += 1
    # ③ 平移 4×MAD 全部越界：前段 [0,1,2,...,9] 中位 4.5，MAD 中位|.|=2.5 ⇒ 带 [-3,12]
    #    后续用 100 ⇒ 越界
    w2 = within_run_baseline(_rows(list(range(10)) + [100.0]), n_base=10, k=3.0)
    s = w2["metrics"]["cos"]
    assert s["base_median"] == 4.5 and s["base_mad"] == 2.5, (s["base_median"], s["base_mad"])
    assert s["later_inside"] == 0, s
    ok += 1
    # ④ n_base 超过总点数
    w3 = within_run_baseline(_rows([1.0, 2.0]), n_base=100, k=3.0)
    assert w3["n_base"] == 2 and w3["n_later"] == 0
    assert w3["metrics"]["cos"]["later_inside"] is None
    ok += 1
    print(f"[selftest] judge_critic_inertia within-run 对照 {ok}/4 PASS")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=False, default=None)
    ap.add_argument("--run", required=False, default=None,
                    help="探针 run 目录（相对 src/clasher_new）")
    ap.add_argument("--runs-root", default=os.path.join(_SRC, "runs"))
    ap.add_argument("--baseline", choices=["prereg", "within"], default="prereg",
                    help="机制层对照口径：prereg=critic 惰性预注册的 P1/P2 固定阈值（默认，"
                         "向后兼容）；within=**本 run 自己前 N 次诊断**的中位±k×MAD"
                         "（engagement_trade_prereg §11.13.2）")
    ap.add_argument("--baseline-n", type=int, default=100,
                    help="--baseline within 的基线窗口点数（§11.13.2 写死 =100）")
    ap.add_argument("--baseline-k", type=float, default=3.0,
                    help="--baseline within 的 MAD 倍数（§11.13.2 写死 =3）")
    ap.add_argument("--selftest", action="store_true",
                    help="跑本判读器 within-run 对照的回归测试后退出")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()
    if not args.log:
        ap.error("--log 必填（除非 --selftest）")
    if not args.run:
        ap.error("--run 必填（除非 --selftest）")

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

    # —— §11.13.2 的 within-run 对照（engagement_trade 干预长跑用）——
    if args.baseline == "within":
        print_within_run(within_run_baseline(rows, n_base=args.baseline_n,
                                             k=args.baseline_k))

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
