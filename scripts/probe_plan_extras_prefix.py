# -*- coding: utf-8 -*-
"""**探针**（【R2】/【R13】）：`--plan-extras` 必须**只追加、不改前 58 维**。

为什么需要它：BC 语料一旦重新生成，**跨进程不可复现**（同代码同 seed、两次独立进程样本差
~5.9%，`docs/fl_il_2026-09-21.md` C17 已登记）⇒ 「新旧语料逐位全等」原理上不可达。
但**同进程内**是确定的（C17 实测 0/42 差）⇒ 把同一局在**同一进程**里转两遍
（`plan_extras=False` / `True`）再逐样本比对，才是这个不变量的可判据形式。

判据（跑前写死）：
  * **P1** 两遍的 `obs` / `belief_token` / `masks` / label 序列**逐位相同**；
  * **P2** `plan[:58]` **逐位相同**（`np.array_equal`）；
  * **P3** `plan[58:]` 有 17 维、**非全零**（证明特征真算了，不是静默补零）；
  * **P4** 关掉 `--plan-extras` 时 `plan.shape == (58,)`（旧路径维度不变）。

用法（仓库根；需 FL jsonl）：
    ./.venv/Scripts/python.exe scripts/probe_plan_extras_prefix.py --games 2
"""
import argparse
import importlib.util
import os
import pickle
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src", "clasher_new")
sys.path.insert(0, SRC)
os.chdir(ROOT)

from rl.io_bootstrap import force_utf8_stdout  # noqa: E402

force_utf8_stdout()

import numpy as np  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


def _load_mod():
    path = os.path.join(HERE, "fl_il_to_bc.py")
    spec = importlib.util.spec_from_file_location("_fl_il_probe", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_fl_il_probe"] = mod
    spec.loader.exec_module(mod)
    return mod


def _recs(mod, jsonl, games):
    """与 `fl_il_to_bc.run_samples` 的过滤**同口径**（该过滤内联在函数体里，无法复用 ⇒ 此处复制）。

    出处：`scripts/fl_il_to_bc.py:633-658`（ALLOWED_MODES / dur<=TIME_CAP / 卡组 8 张 /
    team 出牌 >= 4）。若那边改了过滤，这里必须同步改（本探针会在计数不符时显式报错）。
    """
    import json
    out = []
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["gm"] not in mod.ALLOWED_MODES:
                continue
            if not isinstance(r["dur"], (int, float)) or r["dur"] > mod.TIME_CAP:
                continue
            names, bad = {}, False
            for side in ("team", "opponent"):
                nm, _vs, bd = mod.deck_of(r, side)
                if bd or len(nm) != 8:
                    bad = True
                    break
                names[side] = nm
            if bad:
                continue
            if sum(1 for e in r["ev"] if e[0] == 0 and e[1] == 0) < 4:
                continue
            r["_decks"] = names
            out.append(r)
    return out[:games] if games else out


def _convert(mod, rep, out_dir, plan_extras):
    os.makedirs(out_dir, exist_ok=True)
    return mod._convert_one(0, rep, False, out_dir, 11, "raw", False,
                            "save", 4, False, None, False, plan_extras)


def _load_samples(out_dir):
    out = []
    for fn in sorted(os.listdir(out_dir)):
        if fn.endswith(".pkl"):
            with open(os.path.join(out_dir, fn), "rb") as f:
                out.extend(pickle.load(f))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default=os.path.join(ROOT, "..", "fl_il_data",
                                                    "replays_part000000.jsonl"))
    ap.add_argument("--games", type=int, default=2)
    args = ap.parse_args(argv)
    if not os.path.isfile(args.jsonl):
        print(f"[probe] 找不到 {args.jsonl} ⇒ 无法执行（本探针需要 FL 回放）")
        return 2

    mod = _load_mod()
    recs = _recs(mod, args.jsonl, args.games)
    print(f"[probe] 取 {len(recs)} 局（同一进程、同一 rep 对象，唯一变量 = plan_extras）")
    tmp = tempfile.mkdtemp(prefix="_probe_plan_extras_")
    try:
        for gi, rep in enumerate(recs):
            da = os.path.join(tmp, "off_%d" % gi)
            db = os.path.join(tmp, "on_%d" % gi)
            _convert(mod, rep, da, False)
            _convert(mod, rep, db, True)
            sa, sb = _load_samples(da), _load_samples(db)
            print(f"  [局 {gi}] {rep['tag']}: 样本 off={len(sa)} on={len(sb)}")
            if not sa or not sb or len(sa) != len(sb):
                check(f"P0.{gi} 两遍样本数一致", False, f"{len(sa)} vs {len(sb)}")
                continue
            check(f"P0.{gi} 两遍样本数一致（{len(sa)}）", True)
            obs_ok = all(np.array_equal(x[0]["grid"], y[0]["grid"])
                         and np.array_equal(x[0]["hand"], y[0]["hand"])
                         and np.array_equal(x[0]["elixir"], y[0]["elixir"])
                         and np.array_equal(x[0]["time"], y[0]["time"])
                         for x, y in zip(sa, sb))
            bel_ok = all(np.array_equal(x[1], y[1]) for x, y in zip(sa, sb))
            msk_ok = all(len(x[4]) == len(y[4]) and all(
                np.array_equal(a["slots"], b["slots"]) and np.array_equal(a["cells"], b["cells"])
                for a, b in zip(x[4], y[4])) for x, y in zip(sa, sb))
            check(f"P1.{gi} obs / belief_token / masks **逐位相同**", obs_ok and bel_ok and msk_ok,
                  f"obs={obs_ok} belief={bel_ok} masks={msk_ok}")
            dims_off = sorted({x[2].shape[0] for x in sa})
            dims_on = sorted({x[2].shape[0] for x in sb})
            check(f"P4.{gi} 关闭时 plan 恒 (58,)", dims_off == [58], f"{dims_off}")
            check(f"P3.{gi} 开启时 plan 恒 (75,) = 58+17", dims_on == [75], f"{dims_on}")
            pre_ok = all(np.array_equal(x[2], y[2][:58]) for x, y in zip(sa, sb))
            check(f"P2.{gi} 【核心】plan[:58] **逐位相同**（只追加、不改前缀）", pre_ok)
            nz = float(np.abs(np.array([y[2][58:] for y in sb])).sum())
            check(f"P3b.{gi} 追加的 17 维非全零（特征真算了）", nz > 0.0,
                  f"Σ|extras| = {nz:.3f}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print()
    print(f"=== {'PASS' if not FAILS else 'FAIL'}（{len(FAILS)} 项失败）===")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
