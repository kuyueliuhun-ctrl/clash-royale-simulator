# -*- coding: utf-8 -*-
"""IL 转换（`scripts/fl_il_to_bc.py --mode samples`）**可复现性探针**。

背景（2026-09-21 实测）：同一份代码、同一 `--games 30`、同一 `seed=0`，
* 不同 `--workers`（1 / 6 / 10）⇒ 标签数 635 / 633 / 633，样本级不等 11–22 条；
* **同 `--workers 10`、`PYTHONHASHSEED=0` 固定**，两次独立进程 ⇒ 仍不等 24/379（6.3%）。
⇒ J6.2（「旧口径逐样本全等」）**在原理上不可达**，因为转换本身不是可复现的。

本探针把「不可复现」切成两半，判据二分：

* **同进程内**：把同一局转换两次（`_convert_one` 调两次）⇒ 若**不等** ⇒ 成因在**进程内状态**
  （模块级计数器 / 全局 RNG 流 / 缓存）；
* 若同进程内**全等** ⇒ 再跑一个**新进程**做同样的事 ⇒ 与上一进程比 ⇒ 不等即**跨进程**成因
  （`id()` 哈希 / ASLR / 未播种的全局 RNG）。

⚠️ 只读回放 JSONL；产物写 `--out-dir`（探测样本 + JSON 读数）。不改引擎一行。

用法（Windows venv）:
    .venv/Scripts/python.exe scripts/il_determinism_probe.py \
        --jsonl E:\\fl_il_data\\replays_part000000.jsonl --games 3 \
        --out-dir runs/_il_det_probe --out docs/fl_il_2026-09-21/det_probe_p1.json
"""
import argparse
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src", "clasher_new")
sys.path.insert(0, SRC)
sys.path.insert(0, HERE)          # 以便 import fl_il_to_bc / il_bc_equiv
ORIG_CWD = os.getcwd()

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

import il_bc_equiv  # noqa: E402
import fl_il_to_bc as conv  # noqa: E402   （模块级会 chdir(SRC)，故在上面先存 ORIG_CWD）


def _abs(p):
    m = p.replace("\\", "/")
    if m.startswith("/mnt/") and len(m) > 6:
        head, rest = m.split("/", 3)[2:]
        return head.upper() + ":\\" + rest.replace("/", "\\")
    return os.path.normpath(os.path.join(ORIG_CWD, p))


def load_games(jsonl, n):
    """与 `fl_il_to_bc.run_samples` 的过滤逐条一致（只取前 n 局）。"""
    recs = []
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["gm"] not in conv.ALLOWED_MODES:
                continue
            if not isinstance(r["dur"], (int, float)) or r["dur"] > conv.TIME_CAP:
                continue
            names, bad = {}, False
            for side in ("team", "opponent"):
                nm, _vs, bd = conv.deck_of(r, side)
                if bd or len(nm) != 8:
                    bad = True
                    break
                names[side] = nm
            if bad:
                continue
            if sum(1 for e in r["ev"] if e[0] == 0 and e[1] == 0) < 4:
                continue
            r["_decks"] = names
            recs.append(r)
            if len(recs) >= n:
                break
    return recs


def convert(rep, idx, tmpdir, stop_mode="none"):
    st = conv._convert_one(idx, rep, False, tmpdir, 11, "raw", False, stop_mode, 1, False)
    import pickle
    with open(os.path.join(tmpdir, "bc_fl_%04d.pkl" % idx), "rb") as f:
        return pickle.load(f), st


def compare(a, b):
    n_bad, first = 0, []
    if len(a) != len(b):
        return 1, [{"diff": [f"样本数不等 {len(a)} vs {len(b)}"]}]
    for i, (x, y) in enumerate(zip(a, b)):
        d = il_bc_equiv.cmp_sample(i, x, y)
        if d:
            n_bad += 1
            if len(first) < 3:
                first.append({"i": i, "diff": d})
    return n_bad, first


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--games", type=int, default=3)
    ap.add_argument("--tag", default="p")
    ap.add_argument("--out-dir", default="runs/_il_det_probe")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    args.jsonl = _abs(args.jsonl)
    out_dir = _abs(args.out_dir)
    #: ⚠️ 第一版漏了这句：`--out` 没走 `_abs` ⇒ 被 `fl_il_to_bc` 模块级 `chdir(SRC)` 带到
    #: `src/clasher_new/docs/...`（实测踩过，已清理）
    if args.out:
        args.out = _abs(args.out)
    os.makedirs(out_dir, exist_ok=True)

    recs = load_games(args.jsonl, args.games)
    in_proc = []
    with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
        for gi, rep in enumerate(recs):
            A, stA = convert(rep, gi * 2, t1)
            B, stB = convert(rep, gi * 2 + 1, t2)
            bad, first = compare(A, B)
            in_proc.append({"game": gi, "tag": rep["tag"], "n_a": len(A), "n_b": len(B),
                            "frames": [stA["frames"], stB["frames"]], "mismatch": bad,
                            "first": first})
    n_bad = sum(r["mismatch"] for r in in_proc)
    n_samp = sum(r["n_a"] for r in in_proc)
    res = {
        "jsonl": os.path.basename(args.jsonl), "games": len(recs), "tag": args.tag,
        "in_process": {"samples": n_samp, "mismatch": n_bad, "per_game": in_proc},
        "verdict": ("IN_PROCESS_NONDETERMINISTIC" if n_bad else "IN_PROCESS_DETERMINISTIC"),
        "note": ("同进程内把同一局转换两次。不等 ⇒ 成因在**进程内状态**（模块级计数器/全局 RNG 流/缓存）；"
                 "全等 ⇒ 需再跑一个新进程比 ⇒ 跨进程成因（id() 哈希/ASLR/未播种全局 RNG）。"),
    }
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"[det-probe {args.tag}] 局数={len(recs)} 样本={n_samp} 同进程不等={n_bad} "
          f"⇒ {res['verdict']}")
    for r in in_proc[:3]:
        print(f"   game{r['game']} n={r['n_a']}/{r['n_b']} frames={r['frames']} bad={r['mismatch']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
