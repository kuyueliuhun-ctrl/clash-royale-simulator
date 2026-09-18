# -*- coding: utf-8 -*-
"""评估侧局长 A/B：同一批权重（eval@0 = 全新初始化，seed 0）下，早停开关对局长/结果的影响。

口径：`runs/<run>/replays/league_0.pkl` 是 eval@0 的 40 局 ⇒ 两个 run 的权重逐位相同
（eval@0 在任何训练步之前，R5 的"不可复现"从 eval@800 才开始）
⇒ 逐局帧数可直接配对比较。只读。

用法：
    cd src/clasher_new && PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe \
        ../../scripts/_probe_eval_frame_ab.py runs/demo20k runs/nostall20k
"""
from __future__ import annotations

import io
import os
import pickle
import statistics
import sys

# T1-1b 补齐（2026-09-19）：原先这里是**手写**的 UTF-8 兜底块（只处理 stdout、且不处理 stderr
# ⇒ traceback 在 GBK 下仍是乱码）。现收敛到 T1-1 的**单一实现**（两路 + errors='replace'）。
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                                 "src", "clasher_new"))
from rl.io_bootstrap import force_utf8_stdout  # noqa: E402
force_utf8_stdout()


def load(path):
    if not os.path.exists(path):
        return None
    with io.open(path, "rb") as f:
        return pickle.load(f)


def main():
    runs = sys.argv[1:] or ["runs/demo20k", "runs/nostall20k"]
    print("=" * 100)
    print("评估侧局长 A/B（eval@0，同一初始权重）")
    print("=" * 100)
    data = {}
    for r in runs:
        p = os.path.join(r, "replays", "league_0.pkl")
        d = load(p)
        if d is None:
            print(f"[skip] {p} 不存在")
            continue
        games = d["games"]
        lens, wins = [], []
        for g in games:
            fr = g.get("frames") or []
            if not fr:
                continue
            lens.append(len(fr))
            wins.append(g.get("winner"))
        data[r] = lens
        from collections import Counter
        c = Counter("D" if w is None else ("W" if w == 0 else "L") for w in wins)
        print(f"{r:28s} 局={len(lens):3d} 帧数 min={min(lens):4d} 中位={int(statistics.median(lens)):4d} "
              f"均值={statistics.mean(lens):7.1f} max={max(lens):4d} | W/L/D={c['W']}/{c['L']}/{c['D']} "
              f"| 总帧={sum(lens)}")
    if len(data) == 2:
        (ra, la), (rb, lb) = list(data.items())
        print("-" * 100)
        print(f"逐局配对（前 {min(len(la), len(lb))} 局）：")
        n = min(len(la), len(lb))
        diffs = [lb[i] - la[i] for i in range(n)]
        same = sum(1 for d in diffs if d == 0)
        print(f"  逐局帧数差 {rb} − {ra}: 相同={same}/{n} 中位差={statistics.median(diffs):.1f} "
              f"均值差={statistics.mean(diffs):.1f}")
        if same == n:
            print("  ⇒ **逐局帧数完全一致** ⇒ 早停开关在评估路径上未改变任何一局的长度")
        print(f"  总帧数: {sum(la)} → {sum(lb)}  ({(sum(lb) / max(1, sum(la)) - 1) * 100:+.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
