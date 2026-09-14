"""只读：v4（真·同一张量的投影/归一化对账）跨 seed 归约 —— 判据全部脚本复算（R4）。

用法（仓库根）：
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/judge_probe_v4.py \
      --dir runs/_probe_v4 --seeds 7 11 13 --out runs/_probe_v4/summary.json
"""

import argparse
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

COMMON_ALPHAS = (1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6)
SHOW = ("raw_obs", "grid_ln_out", "fused", "z_enc", "r_enc", "enc", "z_rand",
        "pre_ln", "relu_ln", "post_ln")
BRANCHES = ("P-PROJ", "P-LN", "P-BOTH", "P-RANDOM", "P-VLN", "P-VLN-NEUTRAL")


def S(rec, layer, al):
    c = (rec.get("curves") or {}).get(layer)
    if c is None:
        return None
    v = c.get(str(float(al))) or c.get(f"{al:g}")
    return None if v is None else v[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/_probe_v4")
    ap.add_argument("--seeds", nargs="*", type=int, default=[7, 11, 13])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    recs = {}
    for s in a.seeds:
        p = os.path.join(a.dir, f"seed{s}.json")
        if os.path.exists(p):
            recs[s] = json.load(open(p, encoding="utf-8"))
        else:
            print(f"[warn] 缺 {p}", flush=True)
    if not recs:
        raise SystemExit("[abort] 没有可归约的 json")
    seeds = sorted(recs)

    print("=" * 104)
    print(f"v4 真·同张量投影/归一化对账 | seeds={seeds} | dir={a.dir}")
    print("=" * 104)

    print("\n--- §1 闸门 ---")
    for s in seeds:
        g = recs[s]["gates"]
        print(f"  seed{s:<3} gates_ok={recs[s]['gates_ok']}  " +
              " ".join(f"{k}={'N/A' if v is None else ('P' if v else 'F')}"
                       for k, v in g.items()))
        print(f"          offline max|enc_off − X_enc| = "
              f"{recs[s].get('offline_max_abs_diff')}")

    print("\n--- §2 判据命中（每 seed）---")
    print(f"{'seed':>6} " + " ".join(f"{b:>14}" for b in BRANCHES))
    votes = {b: 0 for b in BRANCHES}
    for s in seeds:
        br = recs[s]["branches"]
        cells = []
        for b in BRANCHES:
            hit = bool(br.get(b))
            votes[b] += int(hit)
            cells.append(f"{'HIT' if hit else '-':>14}")
        print(f"{s:>6} " + " ".join(cells))
    print(f"{'票数':>6} " + " ".join(f"{votes[b]}/{len(seeds):>12}" for b in BRANCHES))

    print("\n--- §3 同 α 阶梯（fused / z_enc / enc / z_rand / relu_ln / post_ln）---")
    for s in seeds:
        r = recs[s]
        print(f"\n  seed {s}")
        print(f"  {'α':>9} " + " ".join(f"{k:>10}" for k in SHOW if S(r, k, 1e-2) is not None))
        for al in COMMON_ALPHAS:
            cells = []
            for k in SHOW:
                v = S(r, k, al)
                if v is None:
                    continue
                cells.append(f"{v:>+10.5f}")
            print(f"  {al:>9g} " + " ".join(cells))

    print("\n--- §4 两个关键比值（同 α，脚本复算）---")
    print("  (a) z_enc / z_rand —— 训练出来的 2731→128 投影 vs 随机投影（≈1 表示两者一样）")
    print("  (b) post_ln / relu_ln —— 价值支路 LN 前后（≈1 表示 LN 中性）")
    for s in seeds:
        r = recs[s]
        ra, rb = [], []
        for al in COMMON_ALPHAS:
            ze, zr = S(r, "z_enc", al), S(r, "z_rand", al)
            po, re_ = S(r, "post_ln", al), S(r, "relu_ln", al)
            ra.append(None if (ze is None or zr in (None, 0) or abs(zr) < 1e-9)
                      else ze / zr)
            rb.append(None if (po is None or re_ in (None, 0) or abs(re_) < 1e-9)
                      else po / re_)
        print(f"  seed{s:<3} z_enc/z_rand = " +
              " ".join("—" if v is None else f"{v:+.2f}" for v in ra))
        print(f"        post_ln/relu_ln = " +
              " ".join("—" if v is None else f"{v:+.2f}" for v in rb))
        # 描述性：3 个随机投影的曲线是否都在同一量级
        for k in ("z_rand2", "z_rand3"):
            if all(S(r, k, al) is not None for al in COMMON_ALPHAS):
                print(f"        {k} 曲线: " + " ".join(
                    f"{S(r, k, al):+.4f}" for al in COMMON_ALPHAS))

    print("\n--- §5 汇总裁决（预注册 §6：方向一致的才写进台账）---")
    for b in BRANCHES:
        if votes[b] == len(seeds):
            print(f"  ✅ {b}：{votes[b]}/{len(seeds)} 条 seed **全部命中** ⇒ 可写结论")
        elif votes[b] == 0:
            print(f"  ❌ {b}：0/{len(seeds)} ⇒ 本轮不支持")
        else:
            print(f"  ⚠️ {b}：{votes[b]}/{len(seeds)} ⇒ **不稳定**，只报票数不合并")

    out = {"seeds": seeds, "votes": votes,
           "per_seed": {str(s): {"gates": recs[s]["gates"],
                                 "gates_ok": recs[s]["gates_ok"],
                                 "branches": recs[s]["branches"],
                                 "hits": recs[s]["hits"],
                                 "verdict": recs[s]["verdict"],
                                 "offline_max_abs_diff":
                                     recs[s].get("offline_max_abs_diff")}
                        for s in seeds},
           "prereg": "docs/value_ln_probe4_prereg_2026-09-14.md"}
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print(f"\n[out] {a.out}")
    return out


if __name__ == "__main__":
    main()
