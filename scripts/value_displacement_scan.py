#!/usr/bin/env python
"""只读：参数位移指纹扫描（R14）。

对每个 run 目录，按步数排序取 `solo_main_*.pt`（不含 `solo_main.pt`），
逐窗口算每个张量的相对位移 `‖ΔW‖/‖W‖`，并统计"位移**恰为 0**"的窗口数。

原理（R14）：Adam 对 loss 的常数缩放不变 ⇒ **位移恰为 0 ⟺ 该张量梯度精确为 0**
⇒ 这是"梯度小"与"梯度恒零（吸收态）"的唯一廉价区分手段。

用法（在 src/clasher_new 下）：
  PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe \
      ../../scripts/value_displacement_scan.py runs/byp_cprime_20k runs/eind_20k ...
"""

import argparse
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src", "clasher_new")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np  # noqa: E402
import torch  # noqa: E402


def load_ckpts(run_dir):
    """返回 [(step, path)]，按 step 升序；忽略 solo_main.pt / solo_opt.pt。"""
    out = []
    for fn in os.listdir(run_dir):
        m = re.fullmatch(r"solo_main_(\d+)\.pt", fn)
        if m:
            out.append((int(m.group(1)), os.path.join(run_dir, fn)))
    out.sort()
    return out


def state_dict_of(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(d, dict) and "state_dict" in d:
        d = d["state_dict"]
    return {k: v.float().numpy() for k, v in d.items() if hasattr(v, "numpy")}


def scan(run_dir, filt, topk):
    cks = load_ckpts(run_dir)
    if len(cks) < 2:
        print(f"  [跳过] {run_dir}: 少于 2 个 solo_main_<step>.pt")
        return None
    sds = [(st, state_dict_of(p)) for st, p in cks]
    names = [k for k in sds[0][1] if filt in k and "num_batches" not in k]
    rows = []
    for k in names:
        zeros = 0
        rels = []
        for i in range(len(sds) - 1):
            a, b = sds[i][1].get(k), sds[i + 1][1].get(k)
            if a is None or b is None or a.shape != b.shape:
                continue
            diff = b - a
            if not np.any(diff):
                zeros += 1
            denom = max(1e-12, float(np.linalg.norm(a)))
            rels.append(float(np.linalg.norm(diff)) / denom)
        if rels:
            rows.append((k, len(rels), zeros, float(np.median(rels))))
    # 对照：策略侧（非 value 的）张量整体冻结情况
    pol_names = [k for k in sds[0][1] if filt not in k and "num_batches" not in k]
    pol_zero = 0
    pol_tot = 0
    for k in pol_names:
        for i in range(len(sds) - 1):
            a, b = sds[i][1].get(k), sds[i + 1][1].get(k)
            if a is None or b is None or a.shape != b.shape:
                continue
            pol_tot += 1
            if not np.any(b - a):
                pol_zero += 1
    nw = len(sds) - 1
    print(f"  [{os.path.basename(run_dir.rstrip('/'))}] 窗口数={nw} "
          f"steps={[s for s, _ in sds]} 匹配张量={len(rows)}")
    if pol_tot:
        print(f"      策略侧（不含 '{filt}'）冻结窗口占比 = {pol_zero}/{pol_tot} "
              f"= {pol_zero / pol_tot:.3f}")
    rows.sort(key=lambda r: (-(r[2] / max(1, r[1])), r[0]))
    for k, nwv, z, med in rows[:topk]:
        flag = "  <== 全窗冻结" if z == nwv else ("  <== 多数窗冻结" if z * 2 > nwv else "")
        print(f"      {k:26s} 窗口={nwv:3d} 恰零={z:3d} ({z / nwv:.2f}) "
              f"中位相对位移={med:.3e}{flag}")
    return {"run": run_dir, "windows": nw, "rows": rows,
            "policy_frozen_frac": (pol_zero / pol_tot if pol_tot else None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--filter", default="value",
                    help="只扫参数名含该子串的张量（默认 value）")
    ap.add_argument("--topk", type=int, default=12)
    a = ap.parse_args()
    print("=== 参数位移指纹扫描（R14；只读，不改任何东西）===", flush=True)
    print(f"    过滤子串='{a.filter}'；判定：位移恰为 0 ⟺ 梯度精确为 0", flush=True)
    allr = []
    for r in a.runs:
        if os.path.isdir(r):
            res = scan(r, a.filter, a.topk)
            if res:
                allr.append(res)
        else:
            print(f"  [缺失] {r}")
    print("\n=== 汇总：value 路径全窗冻结的张量 ===", flush=True)
    for res in allr:
        frozen = [k for k, nwv, z, _ in res["rows"] if z == nwv]
        print(f"  {os.path.basename(res['run'].rstrip('/')):22s} "
              f"全窗冻结={len(frozen)} 个 {frozen if frozen else ''}")


if __name__ == "__main__":
    main()
