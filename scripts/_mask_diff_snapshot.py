# -*- coding: utf-8 -*-
"""P0 掩码逐位对账（【R13】）：快照 / 比较 / 自检 三用。

用法:
  python scripts/_mask_diff_snapshot.py <out.npz>            # 快照（改前、改后各跑一次）
  python scripts/_mask_diff_snapshot.py --compare A.npz B.npz  # 门禁: 逐位全等 -> rc=0
  python scripts/_mask_diff_snapshot.py --selftest            # 判别力自检（必须有用例 FAIL）

【R13】要求「8 状态 × 双方 × 8 手牌 = 128 张位图逐位 diff 全等」才许提交。
旧版（2026-09-19 之前）有两个洞，本版堵上：
  ① **只 dump、不比较、无退出码** ⇒ 「逐位全等」是**人工纪律**，不是脚本保证；
  ② `legal_cells` 抛异常时被 `except` **吞成全零位图** ⇒ 两侧同时坏（或同一 bug 两侧复现）
     仍然「逐位全等」= 假绿。本版把异常计数写进快照（`__meta_errors__`），
     快照本身 rc=2、`--compare` 直接拒判。

stdout 一律 **ASCII-only**：本环境 `PYTHONIOENCODING=utf-8` 实测不生效（stdout=gbk），
中文输出会踩编码陷阱；中文只留在源码/docstring 里。
"""
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src", "clasher_new")
sys.path.insert(0, SRC)
os.chdir(SRC)

import numpy as np

#: 异常计数的存放键（不是位图；比较时单独判定，且要求两侧都为 0）
META_ERRORS = "__meta_errors__"


def build_states():
    """构造一组有代表性的对局状态：空场/敌压境/双倍期/有坦克推进/塔残血。"""
    from battle import BattleState
    from player import PlayerState
    from core import Position

    deck = ["Knight", "MiniPekka", "Arrows", "Minions",
            "Musketeer", "Fireball", "Giant", "Archer"]
    states = []

    def make(cards0=None, cards1=None, time=0.0, level=11):
        bs = BattleState(PlayerState(0, deck, 5.0), PlayerState(1, deck, 5.0),
                         card_level=level)
        bs.time = time
        if cards0:
            for c, (x, y) in cards0:
                bs.deploy_card(0, c, Position(x, y))
        if cards1:
            for c, (x, y) in cards1:
                bs.deploy_card(1, c, Position(x, y))
        return bs

    # ① 空场（前段）
    states.append(("empty_early", make()))
    # ② 空场（双倍期后）— EV 闸门放行口径
    states.append(("empty_late", make(time=130.0)))
    # ③ 敌方 Giant 过河压境（防守豁免口径）
    states.append(("enemy_giant_crossed", make(cards1=[("Giant", (8.5, 12.0))])))
    # ④ 我方 Giant 已在场推进（8h 坦克后屯兵口径）+ 敌方防守单位
    states.append(("my_giant_push", make(cards0=[("Giant", (8.5, 18.0))],
                                         cards1=[("Musketeer", (8.5, 10.0))])))
    # ⑤ 我方 Giant + 后排（Musketeer 在手）同路 — backline gap 全链路
    states.append(("giant_plus_enemy_tank", make(cards0=[("Giant", (8.5, 20.0))],
                                                 cards1=[("Giant", (8.5, 6.0)),
                                                         ("Knight", (9.5, 8.0))])))
    # ⑥ 前段 + 敌公主塔残血（EV 闸门对塔口径）
    s = make()
    for e in s.entities.values():
        if e.player == 1 and "Princess" in (e.name or ""):
            e.hp = 150
    states.append(("late_low_tower", make(time=125.0)))
    # ⑦ 敌方成群小单位（空砸闸门"罩到目标"口径）
    states.append(("enemy_cluster", make(cards1=[("Skeletons", (8.5, 14.0)),
                                                 ("Minions", (9.5, 13.0))])))
    # ⑧ 等级 14（per-level 数值路径）
    states.append(("level14", make(level=14)))

    return states


HAND = ["Knight", "MiniPekka", "Arrows", "Minions",
        "Musketeer", "Fireball", "Giant", "Archer"]


def collect_blobs():
    """返回 (blobs, errors)。异常的位图**仍写零**（保持键集合稳定）但计入 errors。"""
    from rl.action_mask import legal_cells

    blobs = {}
    errors = []
    for name, bs in build_states():
        for pid in (0, 1):
            for card in HAND:
                key = f"{name}|p{pid}|{card}"
                try:
                    cells = legal_cells(bs, pid, card)
                except Exception as e:  # noqa: BLE001 —— 记录而非吞掉
                    errors.append(f"{key}: {e!r}")
                    cells = np.zeros((32, 18), dtype=bool)
                blobs[key] = cells.astype(np.uint8)
    return blobs, errors


def snapshot(out_path):
    blobs, errors = collect_blobs()
    blobs[META_ERRORS] = np.asarray(len(errors), dtype=np.int64)
    np.savez_compressed(out_path, **blobs)
    n_maps = len(blobs) - 1
    print(f"snapshot: {n_maps} maps -> {out_path}")
    for line in errors:
        print(f"[ERR] {line}")
    if errors:
        print(f"[FAIL] {len(errors)} map(s) raised -> snapshot is NOT usable as a gate")
        return 2
    return 0


def _errors_of(z):
    if META_ERRORS not in z.files:
        return -1  # 旧版快照：无法判定是否假绿
    return int(np.asarray(z[META_ERRORS]).reshape(-1)[0])


def compare(a_path, b_path):
    """逐位比较两个快照。rc=0 当且仅当：键集相同 + 每张位图逐位相等 + 两侧异常数均为 0。"""
    a = np.load(a_path)
    b = np.load(b_path)
    ka, kb = set(a.files), set(b.files)
    if ka != kb:
        print(f"[FAIL] key sets differ: only-A={sorted(ka - kb)[:5]} only-B={sorted(kb - ka)[:5]}")
        return 1

    maps = sorted(k for k in ka if k != META_ERRORS)
    bad = []
    for k in maps:
        x, y = a[k], b[k]
        if x.shape != y.shape:
            bad.append((k, f"shape {x.shape} vs {y.shape}"))
        elif not np.array_equal(x, y):
            bad.append((k, f"{int(np.count_nonzero(x != y))} cells differ"))

    ea, eb = _errors_of(a), _errors_of(b)
    if bad:
        print(f"[FAIL] {len(bad)}/{len(maps)} maps differ:")
        for k, why in bad[:10]:
            print(f"  - {k}: {why}")
        if len(bad) > 10:
            print(f"  ... and {len(bad) - 10} more")
        return 1
    if ea or eb:
        print(f"[FAIL] bitmaps identical but snapshots had build errors: a={ea} b={eb} "
              f"(a negative value = legacy snapshot without {META_ERRORS})")
        return 1
    print(f"[OK] {len(maps)}/{len(maps)} maps bitwise identical; build errors a=0 b=0")
    return 0


def _write_synth(path, *, keys=("s0|p0|Knight", "s1|p1|Giant"), flip=None, errs=0):
    d = {k: np.zeros((32, 18), dtype=np.uint8) for k in keys}
    first = keys[0]
    d[first][3, 4] = 1
    if flip is not None:
        d[flip[0]][flip[1], flip[2]] ^= 1
    d[META_ERRORS] = np.asarray(errs, dtype=np.int64)
    np.savez_compressed(path, **d)


def selftest():
    """判别力断言：每个用例都必须给出**预期**的 rc（尤其「必须 FAIL」的那些）。"""
    tmp = tempfile.mkdtemp(prefix="maskdiff_")
    cases = []
    try:
        base = os.path.join(tmp, "base.npz")
        _write_synth(base)

        same = os.path.join(tmp, "same.npz")
        _write_synth(same)
        cases.append(("identical pair -> PASS", base, same, 0))

        flip = os.path.join(tmp, "flip.npz")
        _write_synth(flip, flip=("s1|p1|Giant", 7, 9))
        cases.append(("one bit flipped -> FAIL", base, flip, 1))

        missing = os.path.join(tmp, "missing.npz")
        _write_synth(missing, keys=("s0|p0|Knight",))
        cases.append(("key set mismatch -> FAIL", base, missing, 1))

        err = os.path.join(tmp, "err.npz")
        _write_synth(err, errs=1)
        cases.append(("build error on one side -> FAIL", base, err, 1))

        shape = os.path.join(tmp, "shape.npz")
        d = {"s0|p0|Knight": np.zeros((16, 9), dtype=np.uint8),
             "s1|p1|Giant": np.zeros((32, 18), dtype=np.uint8),
             META_ERRORS: np.asarray(0, dtype=np.int64)}
        np.savez_compressed(shape, **d)
        cases.append(("shape mismatch -> FAIL", base, shape, 1))

        ok = 0
        for name, p, q, want in cases:
            got = compare(p, q)
            mark = "PASS" if got == want else "FAIL"
            ok += (mark == "PASS")
            print(f"[selftest] {mark} {name} (rc={got}, want {want})")
        print(f"[selftest] {ok}/{len(cases)} assertions PASS")
        return 0 if ok == len(cases) else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv):
    if len(argv) >= 2 and argv[1] == "--selftest":
        return selftest()
    if len(argv) >= 2 and argv[1] == "--compare":
        if len(argv) != 4:
            print("usage: python scripts/_mask_diff_snapshot.py --compare A.npz B.npz")
            return 64
        return compare(argv[2], argv[3])
    out_path = argv[1] if len(argv) > 1 else "docs/_mask_baseline.npz"
    return snapshot(out_path)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
