# -*- coding: utf-8 -*-
"""两个 IL（BC）样本目录的**逐样本全等**对账 —— J6.2 的执行者（见 `docs/fl_il_stopframe_prereg_2026-09-21.md`）。

用途：验证「给 `scripts/fl_il_to_bc.py` 加 `--stop-mode` 之后，`--stop-mode none` 的旧口径
**逐位不变**」——这是**单变量**实验的前置条件（若这里不等，后面所有「加 save 帧」的读数都不可解释）。

对账口径（逐样本、逐字段，全部用 `numpy.array_equal` / `==`，**不做容差**）：
* `obs`：键集合 + 每个键的数组逐元素相等；
* `belief_tok` / `plan_vec`：逐元素相等；
* `bundle`：`intent` + 子动作序列 `(kind, slot, x, y)`；
* `masks`：序列长度 + 每个掩码的 `slots` / `cells`（逐位）/ `ability_legal` 存在性与取值。

用法（Windows venv）:
    .venv/Scripts/python.exe scripts/il_bc_equiv.py \
        --dir-a runs/_fl_il_bc --dir-b runs/_fl_il_bc_eq30 \
        --out docs/fl_il_2026-09-21/equiv_none_vs_old.json

只读两个目录；只写 `--out`。
"""
import argparse
import glob
import json
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG_CWD = os.getcwd()

import numpy as np  # noqa: E402

#: ⚠️ **必须**把 `src/clasher_new` 放进 `sys.path`：pkl 里存的是 `rl.action_bundle.ActionBundle`
#: ⇒ 反序列化时 unpickler 要能 import `rl.action_bundle`，否则 `ModuleNotFoundError`
#: （第一版漏了，实测踩过——J6.2 因此没跑成，而 `echo $?` 接在管道后还报了假绿的 0）。
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src", "clasher_new"))

#: ⚠️ **GBK 陷阱**：Windows 控制台默认 cp936 ⇒ 打印里的 `⇒` 等字符会 `UnicodeEncodeError`
#: （实测踩过：对账本身已跑完，倒在最后一行 print 上）。与 `rl/io_bootstrap.force_utf8_stdout` 同效，
#: 但**不 import 引擎**（保持本脚本无 torch 依赖、可单跑）。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass


def _abs(p):
    m = p.replace("\\", "/")
    if m.startswith("/mnt/") and len(m) > 6:
        head, rest = m.split("/", 3)[2:]
        return head.upper() + ":\\" + rest.replace("/", "\\")
    return os.path.normpath(os.path.join(ORIG_CWD, p))


def _eq(a, b):
    try:
        return bool(np.array_equal(np.asarray(a), np.asarray(b)))
    except Exception:  # noqa: BLE001
        return a == b


def _obs_keys(o):
    return sorted(o.keys()) if isinstance(o, dict) else None


def cmp_sample(i, sa, sb):
    """返回差异描述列表（空 = 全等）。"""
    d = []
    oa, ta, pa, ba, ma = sa
    ob, tb, pb, bb, mb = sb
    if _obs_keys(oa) != _obs_keys(ob):
        d.append(f"obs 键不同: {_obs_keys(oa)} vs {_obs_keys(ob)}")
    else:
        for k in _obs_keys(oa):
            if not _eq(oa[k], ob[k]):
                d.append(f"obs[{k}] 不等")
    if not _eq(ta, tb):
        d.append("belief_tok 不等")
    if not _eq(pa, pb):
        d.append("plan_vec 不等")
    if int(getattr(ba, "intent", 0)) != int(getattr(bb, "intent", 0)):
        d.append("bundle.intent 不等")
    ta_ = [(s.kind, int(s.slot), int(s.x), int(s.y)) for s in ba.sub_actions]
    tb_ = [(s.kind, int(s.slot), int(s.x), int(s.y)) for s in bb.sub_actions]
    if ta_ != tb_:
        d.append(f"bundle 子动作不等: {ta_} vs {tb_}")
    if len(ma) != len(mb):
        d.append(f"masks 长度不等: {len(ma)} vs {len(mb)}")
    else:
        for j, (x, y) in enumerate(zip(ma, mb)):
            if not _eq(x.get("slots"), y.get("slots")):
                d.append(f"masks[{j}].slots 不等")
            if not _eq(x.get("cells"), y.get("cells")):
                d.append(f"masks[{j}].cells 不等")
            if bool(x.get("ability_legal", False)) != bool(y.get("ability_legal", False)):
                d.append(f"masks[{j}].ability_legal 不等")
    return d


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir-a", required=True, help="参考目录（旧口径产物）")
    ap.add_argument("--dir-b", required=True, help="待验目录（`--stop-mode none` 重跑产物）")
    ap.add_argument("--sub", default="train", help="子目录（train/holdout，缺省 train）")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-report", type=int, default=5)
    args = ap.parse_args(argv)
    for n in ("dir_a", "dir_b", "out"):
        v = getattr(args, n, None)
        if v:
            setattr(args, n, _abs(v))

    da = os.path.join(args.dir_a, args.sub)
    db = os.path.join(args.dir_b, args.sub)
    fa = {os.path.basename(p): p for p in glob.glob(os.path.join(da, "bc_*.pkl"))}
    fb = {os.path.basename(p): p for p in glob.glob(os.path.join(db, "bc_*.pkl"))}
    only_a = sorted(set(fa) - set(fb))
    only_b = sorted(set(fb) - set(fa))
    common = sorted(set(fa) & set(fb))
    n_samples = n_bad = 0
    first = []
    for fn in common:
        A = pickle.load(open(fa[fn], "rb"))
        B = pickle.load(open(fb[fn], "rb"))
        if len(A) != len(B):
            first.append({"file": fn, "diff": [f"样本数不等 {len(A)} vs {len(B)}"]})
            n_bad += 1
            continue
        for i, (x, y) in enumerate(zip(A, B)):
            n_samples += 1
            dd = cmp_sample(i, x, y)
            if dd:
                n_bad += 1
                if len(first) < args.max_report:
                    first.append({"file": fn, "i": i, "diff": dd})
    res = {
        "dir_a": args.dir_a, "dir_b": args.dir_b, "sub": args.sub,
        "files_a": len(fa), "files_b": len(fb), "files_common": len(common),
        "files_only_a": only_a[:10], "files_only_b": only_b[:10],
        "samples_compared": n_samples, "samples_mismatch": n_bad,
        "J6_2": {"gate": "files_only_a==0 and files_only_b==0 and samples_mismatch==0",
                 "verdict": ("PASS" if (not only_a and not only_b and n_bad == 0) else "FAIL")},
        "first_mismatches": first,
        "note": ("逐样本逐字段**全等**对账（无容差）。PASS ⇒ 加 `--stop-mode` 没有动旧口径 ⇒ 单变量成立。"),
    }
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"[equiv] {args.dir_a} vs {args.dir_b} ({args.sub})："
          f"files {len(fa)}/{len(fb)}（共同 {len(common)}），样本 {n_samples}，不等 {n_bad} "
          f"⇒ J6.2 {res['J6_2']['verdict']}")
    for m in first[:3]:
        print("   ", m)
    return 0 if res["J6_2"]["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
