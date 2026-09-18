"""只读：v3 探针的**跨 seed 归约与判读**（对应预注册 §5/§6/§7）。

判据必须由脚本复算，不许手抄（【红线 R4】）。本脚本只读 `runs/_probe_v3/seed*.json`，
不做任何拟合、不写 ckpt、不碰 `rl/`。

用法（仓库根）：
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/judge_probe_v3.py \
      --dir runs/_probe_v3 --seeds 7 11 13 17 --out runs/_probe_v3/summary.json
"""

import argparse
import glob
import json
import os
import sys

# T1-1b 补齐（2026-09-19）：原先这里是**手写**的 UTF-8 兜底块（只处理 stdout、且不处理 stderr
# ⇒ traceback 在 GBK 下仍是乱码）。现收敛到 T1-1 的**单一实现**（两路 + errors='replace'）。
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                                 "src", "clasher_new"))
from rl.io_bootstrap import force_utf8_stdout  # noqa: E402
force_utf8_stdout()

#: 预注册 §5 写死的比值闸阈值（跑前固定，禁止事后调整）
RHO_MAX = 0.5
#: 预注册 §6 分支用的相对阈值
HALF = 0.5

LAYERS = [
    ("raw_obs", "参考集 obs"),
    ("raw_nongrid", "raw 非 grid 块(8)"),
    ("grid_x", "CNN 输入(14976)"),
    ("cnn_pre_ln", "CNN 输出(2560)"),
    ("grid_ln_out", "grid_ln 之后(2560)"),
    ("hand_f", "hand_f(40)"),
    ("scalar_f", "scalar(3)"),
    ("plan_f", "plan_f(64)"),
    ("belief_f", "belief_f(64)"),
    ("fused", "fused(2731)"),
    ("enc", "共享 enc(128)"),
    ("pre_ln", "value_enc_fc(128)"),
    ("relu_ln", "relu(128)"),
    ("post_ln", "value_enc_ln(128)"),
    ("mlp0_post", "mlp0(64)"),
    ("value", "value(1)"),
]


def load(d, s):
    p = os.path.join(d, f"seed{s}.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def ev_of(run, key):
    r = (run.get("rows") or {}).get(key)
    return None if r is None else r.get("EV_within")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/_probe_v3")
    ap.add_argument("--seeds", nargs="*", type=int, default=[7, 11, 13, 17])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    runs = {}
    for s in a.seeds:
        r = load(a.dir, s)
        if r is None:
            print(f"[warn] seed{s}: 缺 json，跳过（{a.dir}/seed{s}.json）", flush=True)
            continue
        runs[s] = r
    if not runs:
        raise SystemExit("[abort] 没有任何 seed json")

    seeds = sorted(runs)
    print("=" * 100)
    print(f"v3 探针跨 seed 归约 | seeds={seeds} | dir={a.dir}")
    print("=" * 100)

    # ---- §1 逐 seed 基本量 ----
    print("\n--- §1 逐 seed 基本量 ---")
    print(f"{'seed':>5} {'frames':>7} {'eps':>4} {'Var(R)w':>9} {'EV(raw)':>9} "
          f"{'EV(fused)':>10} {'ρ(fused)':>9} {'R0/E0':>7} {'verdict':>22}")
    rho_list, r0e0_list = [], []
    for s in seeds:
        r = runs[s]
        R = ev_of(r, "raw_obs")
        F = ev_of(r, "fused")
        rho = (F / R) if (R not in (None, 0) and F is not None and R > 0) else None
        if rho is not None:
            rho_list.append(rho)
            r0e0_list.append(1.0 / rho)
        v3 = r.get("v3") or {}
        print(f"{s:>5} {r['frames']:>7} {r['episodes']:>4} "
              f"{(v3.get('var_within') or float('nan')):>9.3f} "
              f"{R:>9.4f} {F:>10.4f} "
              f"{('None' if rho is None else f'{rho:.4f}'):>9} "
              f"{('None' if rho is None else f'{1.0 / rho:.2f}×'):>7} "
              f"{r.get('verdict', ''):>22}")

    # ---- §2 逐层表（行=层，列=seed）----
    print("\n--- §2 逐层 EV_within（行=层，列=seed；括号内 ρ=EV/EV(raw_obs)）---")
    hdr = f"{'layer':>14} {'dim':>6} " + " ".join(f"{'s' + str(s):>17}" for s in seeds)
    print(hdr)
    for key, label in LAYERS:
        if not any(ev_of(runs[s], key) is not None for s in seeds):
            continue
        dim = None
        for s in seeds:
            rr = (runs[s].get("rows") or {}).get(key)
            if rr is not None:
                dim = rr.get("dim_raw")
                break
        cells = []
        for s in seeds:
            e = ev_of(runs[s], key)
            R = ev_of(runs[s], "raw_obs")
            if e is None:
                cells.append(f"{'—':>17}")
            else:
                rho = (e / R) if (R not in (None, 0) and R > 0) else None
                cells.append(f"{e:>+7.4f}({('—' if rho is None else f'{rho:.3f}'):>5})")
        print(f"{label:>14} {str(dim):>6} " + " ".join(cells))

    # ---- §3 比值分布（判据 of record）----
    print("\n--- §3 ρ(fused)=EV(fused)/EV(raw_obs) 的分布（判据 of record，预注册 §5）---")
    if rho_list:
        import statistics as st
        print(f"  n={len(rho_list)}  min={min(rho_list):.4f}  max={max(rho_list):.4f}  "
              f"mean={st.mean(rho_list):.4f}  "
              f"std={st.pstdev(rho_list) if len(rho_list) > 1 else 0.0:.4f}  "
              f"极差={max(rho_list) - min(rho_list):.4f}")
        print(f"  R0/E0 = 1/ρ: " + " / ".join(f"{x:.2f}×" for x in r0e0_list))
        print(f"  阈值 ρ ≤ {RHO_MAX}：距观测上界余量 {RHO_MAX - max(rho_list):.4f}"
              f"（= 极差的 "
              f"{(RHO_MAX - max(rho_list)) / max(1e-9, max(rho_list) - min(rho_list)):.2f} 倍）")
        gate_ok = all(x <= RHO_MAX for x in rho_list)
        print(f"  ⇒ G-RATIO({'all ≤ ' + str(RHO_MAX)}) = "
              f"{'PASS' if gate_ok else 'FAIL'}（{sum(x <= RHO_MAX for x in rho_list)}"
              f"/{len(rho_list)} 条通过）")
    else:
        gate_ok = False
        print("  ⇒ 无法计算（存在 R0 ≤ 0 或无 raw_obs）")

    # ---- §4 逐 seed 闸门 & 分支 ----
    print("\n--- §4 逐 seed 闸门与分支 ---")
    branch_votes = {}
    for s in seeds:
        v3 = runs[s].get("v3") or {}
        g = runs[s].get("gates") or {}
        gs = list(v3.get("gate_set_v3") or [])
        gtxt = " ".join(f"{k}={'N/A' if g.get(k) is None else ('P' if g.get(k) else 'F')}"
                        for k in gs)
        print(f"  seed{s:<3} gates_ok={v3.get('gates_ok')}  {gtxt}")
        br = v3.get("branches") or {}
        tr = " ".join(f"{k}={int(bool(x))}" for k, x in br.items())
        print(f"          branches: {tr}")
        print(f"          hits={v3.get('hits')}  verdict={v3.get('verdict')}")
        for h in (v3.get("hits") or []):
            branch_votes[h] = branch_votes.get(h, 0) + 1

    # ---- §5 汇总裁决 ----
    print("\n--- §5 汇总（预注册 §7 失败分支照单读）---")
    all_valid = all((runs[s].get("v3") or {}).get("gates_ok") for s in seeds)
    seq_ok = all((runs[s].get("gates") or {}).get("G-REPRO") is not False for s in seeds)
    print(f"  全部 seed gates_ok ? {all_valid}")
    print(f"  G-REPRO 无 FAIL ? {seq_ok}")
    if branch_votes:
        print("  分支票数: " + "  ".join(f"{k}={v}/{len(seeds)}"
                                         for k, v in sorted(branch_votes.items(),
                                                            key=lambda x: -x[1])))
    summary = {
        "seeds": seeds, "rho_fused": rho_list, "r0e0": r0e0_list,
        "rho_max": max(rho_list) if rho_list else None,
        "G_RATIO_pass": bool(gate_ok), "all_gates_ok": bool(all_valid),
        "branch_votes": branch_votes,
        "per_seed": {str(s): {
            "frames": runs[s]["frames"], "episodes": runs[s]["episodes"],
            "verdict": runs[s].get("verdict"),
            "ev_within": {k: ev_of(runs[s], k) for k, _ in LAYERS},
            "gates": runs[s].get("gates"),
            "gates_ok": (runs[s].get("v3") or {}).get("gates_ok"),
            "hits": (runs[s].get("v3") or {}).get("hits"),
            "branches": (runs[s].get("v3") or {}).get("branches"),
            "var_within": (runs[s].get("v3") or {}).get("var_within"),
        } for s in seeds},
        "rho_threshold": RHO_MAX,
        "prereg": "docs/value_ln_probe3_prereg_2026-09-14.md",
    }
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1)
        print(f"\n[out] {a.out}")
    return summary


if __name__ == "__main__":
    main()
