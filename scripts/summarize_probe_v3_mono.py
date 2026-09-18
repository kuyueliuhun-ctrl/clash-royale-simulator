"""只读：把 v3 的 `mono_check` 日志解析成**脚本复算**的汇总（【红线 R4】禁止手抄）。

产出 `docs/value_ln_probe3_mono_summary.json`，供判读文档引用。

用法（仓库根）：
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/summarize_probe_v3_mono.py \
      --logs docs/value_ln_probe3_mono_seed7.log ... --out docs/value_ln_probe3_mono_summary.json
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

ALPHAS = (1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6)
RE_SUP = re.compile(r"X_fused\[:, :(\d+)\] == X_grid_ln_out \? (\w+)")
RE_HDR = re.compile(r"^\s*([\d.eE+-]+)\s+((?:[+-][\d.]+|—)(?:\s+(?:[+-][\d.]+|—))*)\s*$")
RE_OWN = re.compile(r"^\s*(\S+)\s+α\*=\s*([\d.eE+-]+)\s+test=([+-][\d.]+)\s+ρ=(\S+)\s*$")


def parse(path):
    txt = open(path, encoding="utf-8").read()
    seed = int(re.search(r"mono_seed(\d+)", path).group(1))
    out = {"seed": seed, "path": path}
    m = RE_SUP.search(txt)
    if m:
        out["superset_cols"] = int(m.group(1))
        out["superset_bitwise_equal"] = (m.group(2) == "True")
    # 共同 α 表
    tbl = {}
    in3 = False
    for line in txt.splitlines():
        if "§3 共同 α 阶梯" in line:
            in3 = True
            continue
        if in3 and "§3b" in line:
            in3 = False
        if not in3:
            continue
        m2 = RE_HDR.match(line)
        if not m2:
            continue
        toks = m2.group(2).split()
        try:
            al = float(m2.group(1))
        except ValueError:
            continue
        if abs(al - round(al)) > 1e-12 and al not in ALPHAS:
            continue
        vals = [None if t == "—" else float(t) for t in toks]
        if vals:
            tbl[al] = vals
    out["common_alpha_table"] = {str(k): v for k, v in sorted(tbl.items())}
    # 每层自己 α*
    own = {}
    for line in txt.splitlines():
        m3 = RE_OWN.match(line)
        if m3:
            own[m3.group(1)] = {"alpha": float(m3.group(2)),
                                "test": float(m3.group(3)),
                                "rho": (None if m3.group(4) == "—" else float(m3.group(4)))}
    out["own_alpha"] = own
    # §4
    m4 = re.search(r"同一\s*\*{0,2}\s*α\*\(([\d.eE+-]+)\)\s*\*{0,2}\s*下："
                   r"fused=([+-][\d.]+) vs grid_ln_out=([+-][\d.]+) ⇒ 差 ([+-][\d.]+)", txt)
    if m4:
        out["same_alpha"] = {"alpha": float(m4.group(1)), "fused": float(m4.group(2)),
                             "grid_ln_out": float(m4.group(3)), "diff": float(m4.group(4))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="*", default=None,
                    help="列名（默认从日志的 α 表头推导固定顺序）")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    labels = a.labels or ["raw_obs", "plan_f", "cnn_pre_ln", "grid_ln_out", "fused",
                          "enc", "pre_ln"]
    recs = [parse(p) for p in a.logs]
    recs.sort(key=lambda r: r["seed"])

    print("=" * 96)
    print("v3 mono 汇总（脚本复算）")
    print("=" * 96)
    print("\n--- §A 超集逐位验证 ---")
    for r in recs:
        print(f"  seed{r['seed']:<3} X_fused[:, :{r.get('superset_cols')}] == "
              f"X_grid_ln_out ? {r.get('superset_bitwise_equal')}")

    print("\n--- §B 同 α 配对：cnn_pre_ln vs grid_ln_out（grid_ln 有没有丢信号）---")
    stats = {}
    for r in recs:
        t = r["common_alpha_table"]
        d = []
        for al, vals in t.items():
            if len(vals) >= 4 and vals[2] is not None and vals[3] is not None:
                d.append(vals[2] - vals[3])
        stats[r["seed"]] = {"n_alpha": len(d), "max_abs_diff": max(map(abs, d)) if d else None,
                            "signed_diffs": d}
        print(f"  seed{r['seed']:<3} over {len(d)} 个 α：max|cnn_pre_ln − grid_ln_out| = "
              f"{max(map(abs, d)) if d else float('nan'):.5f}")

    print("\n--- §C 同 α 配对：enc / pre_ln vs 它们的输入 grid_ln_out ---")
    for r in recs:
        t = r["common_alpha_table"]
        for al in sorted(t, key=float):
            v = t[al]
            if len(v) >= 7 and v[3] is not None:
                print(f"  seed{r['seed']:<3} α={float(al):<8g} grid_ln_out={v[3]:+.5f} "
                      f"fused={v[4]:+.5f} enc={v[5]:+.5f} pre_ln={v[6]:+.5f}")

    print("\n--- §D 各自 α* 读法 vs 同 α 读法（fused vs 其子块 grid_ln_out）---")
    for r in recs:
        own = r["own_alpha"]
        sa = r.get("same_alpha") or {}
        f_own = (own.get("fused") or {}).get("test")
        g_own = (own.get("grid_ln_out") or {}).get("test")
        sdiff = sa.get("diff")
        print(f"  seed{r['seed']:<3} 各自 α*：fused={f_own:+.5f} (α*={own['fused']['alpha']:g}) "
              f"grid_ln_out={g_own:+.5f} (α*={own['grid_ln_out']['alpha']:g}) "
              f"Δ={f_own - g_own:+.5f}   ‖  同 α={sa.get('alpha')}：Δ="
              f"{'—' if sdiff is None else f'{sdiff:+.5f}'}")

    print("\n--- §F 同 α：pre_ln（价值支路 2731→128，无 LN）vs enc（共享 2731→128 + LN）---")
    f_cnt = {"pre>enc": 0, "pre<enc": 0, "tie": 0}
    for r in recs:
        t = r["common_alpha_table"]
        w = l = 0
        for al in sorted(t, key=float):
            v = t[al]
            if len(v) >= 7 and v[5] is not None and v[6] is not None:
                if v[6] > v[5]:
                    w += 1
                elif v[6] < v[5]:
                    l += 1
        f_cnt["pre>enc" if w > l else ("pre<enc" if l > w else "tie")] += 1
        print(f"  seed{r['seed']:<3} over {w + l} 个 α：pre_ln > enc 的 α 数 = {w}，"
              f"< 的 = {l}")

    print("\n--- §E 逐层 ρ（各自 α*，相对 raw_obs）---")
    print(f"{'seed':>5} " + " ".join(f"{k:>12}" for k in labels))
    for r in recs:
        own = r["own_alpha"]
        cells = []
        for k in labels:
            v = own.get(k)
            cells.append(f"{'—':>12}" if not v else f"{v['test']:>+12.5f}")
        print(f"{r['seed']:>5} " + " ".join(cells))

    summary = {"records": recs, "labels": labels,
               "grid_ln_neutrality": stats, "pre_vs_enc": f_cnt}
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1)
        print(f"\n[out] {a.out}")
    return summary


if __name__ == "__main__":
    main()
