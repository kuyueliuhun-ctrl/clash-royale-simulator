# -*- coding: utf-8 -*-
"""法术掩码**全卡位图**对账（【R13】扩展口径）：全部 `type=="spell"` × 8 状态 × 2 方。

## 为什么必须另开一个脚本（不是重复造轮子）

`_mask_diff_snapshot.py` 的 `HAND` 是 **8 张固定手牌**（Knight/MiniPekka/Arrows/Minions/
Musketeer/Fireball/Giant/Archer），其中**只有 `Arrows(3)` 与 `Fireball(4)` 是法术**
⇒ 那 128 张语料对**法术谓词类**改动是**结构性盲的**：
2026-09-22 的 F4 批（`_spell_deals_damage` 与引擎同源）实改 `Zap/Poison/Vines/...`，
而 128 张语料**逐位全等**——差一点被误判成"无影响"。
（当时手工补了 384 张 sweep，但**产出脚本没有入库** ⇒ 快照无法复算。
本脚本把它变成**可复算的门禁**，并把语料清单写进快照自身。）

## 三用（与主脚本同纪律：异常不吞、rc 有语义）

    python scripts/_mask_diff_snapshot_spells.py --dump <out.npz>
    python scripts/_mask_diff_snapshot_spells.py --compare A.npz B.npz   # 逐位全等 -> rc=0
    python scripts/_mask_diff_snapshot_spells.py --ab                    # 同进程 A/B 方向门禁

## `--ab` 的口径（2026-09-22 W1：低费法术空砸豁免）

把 `SPELL_WHIFF_FREE_MAX_COST` **单变量地**设为 `-1.0`（= 改前语义：所有伤害型法术都受
8h 空砸闸门约束）与**当前值**，在**同一进程内**逐张位图求差。比"文件 vs 文件"更强：
状态、代码、解释器、随机性全部同源，差异**只可能**来自那一个常量。

断言（两条，都是"只许往一个方向走"）：
  ① **零收紧**：`legal→illegal` 的格数 = **0**（低费豁免只许**放松**落点）；
  ② **零越界**：发生变化的卡 ⊆ {费用 ≤ 当前阈值 的**落点伤害型**法术}。

`standard` 语料（128 张）另由 `_mask_diff_snapshot.py --compare` 单独跑；
本脚本在 `--ab` 末尾会**顺带**提示该命令（不代跑，避免 cwd 副作用叠加）。

stdout 一律 **ASCII-only**（本环境 stdout=gbk，中文只留在源码/docstring）。
"""
import argparse
import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src", "clasher_new")
sys.path.insert(0, SRC)
sys.path.insert(0, HERE)          # 复用 _mask_diff_snapshot.build_states（保证状态集同源）
os.chdir(SRC)

import numpy as np  # noqa: E402

_mds = importlib.import_module("_mask_diff_snapshot")

META_ERRORS = "__meta_errors__"
META_CARDS = "__meta_cards__"
META_THRESHOLD = "__meta_threshold__"

#: 改前语义的等价阈值：`cost > -1.0` 对一切非负费用恒真 ⇒ 所有伤害型法术都受空砸闸门约束。
BASELINE_THRESHOLD = -1.0


def spell_cards():
    """全部 `type=="spell"` 的卡名（来自本仓真实卡数据，非硬编码清单）。"""
    from card_utils import Card, card_data
    out = []
    for n in sorted(card_data.keys()):
        try:
            if Card(n).type == "spell":
                out.append(n)
        except Exception:      # noqa: BLE001 —— 构造失败的卡名直接不算法术
            continue
    return out


def collect(threshold):
    """在 `threshold` 下采集全部位图。返回 (blobs, errors)。异常**不吞**，计入 errors。"""
    import rl.action_mask as am
    old = am.SPELL_WHIFF_FREE_MAX_COST
    am.SPELL_WHIFF_FREE_MAX_COST = threshold
    try:
        blobs, errors = {}, []
        cards = spell_cards()
        for name, bs in _mds.build_states():
            for pid in (0, 1):
                for card in cards:
                    key = f"{name}|p{pid}|{card}"
                    try:
                        cells = am.legal_cells(bs, pid, card)
                    except Exception as e:      # noqa: BLE001 —— 记录而非吞掉
                        errors.append(f"{key}: {e!r}")
                        cells = np.zeros((32, 18), dtype=bool)
                    blobs[key] = np.asarray(cells).astype(np.uint8)
        return blobs, errors, cards
    finally:
        am.SPELL_WHIFF_FREE_MAX_COST = old


def _write(out, blobs, errors, cards, threshold):
    blobs = dict(blobs)
    blobs[META_ERRORS] = np.asarray(len(errors), dtype=np.int64)
    blobs[META_CARDS] = np.asarray(cards)
    blobs[META_THRESHOLD] = np.asarray(float(threshold))
    np.savez_compressed(out, **blobs)


def _p(path):
    """把相对路径按**仓库根**解析（本模块 import 时已 `chdir(SRC)`，相对路径会指错地方）。"""
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def cmd_dump(args):
    import rl.action_mask as am
    th = float(am.SPELL_WHIFF_FREE_MAX_COST) if args.threshold is None else float(args.threshold)
    blobs, errors, cards = collect(th)
    out = _p(args.dump)
    _write(out, blobs, errors, cards, th)
    print(f"snapshot: {len(blobs)} maps ({len(cards)} spells x 8 states x 2 sides) "
          f"threshold={th} -> {out}")
    for line in errors:
        print(f"[ERR] {line}")
    if errors:
        print(f"[FAIL] {len(errors)} map(s) raised -> snapshot NOT usable as a gate")
        return 2
    return 0


def cmd_compare(args):
    a, b = np.load(_p(args.compare[0])), np.load(_p(args.compare[1]))
    ka, kb = set(a.files), set(b.files)
    if ka != kb:
        print(f"[FAIL] key sets differ: only-A={sorted(ka - kb)[:5]} only-B={sorted(kb - ka)[:5]}")
        return 1
    maps = sorted(k for k in ka if not k.startswith("__meta_"))
    bad = [(k, int(np.count_nonzero(a[k] != b[k]))) for k in maps if not np.array_equal(a[k], b[k])]
    ea = int(np.asarray(a[META_ERRORS]).reshape(-1)[0]) if META_ERRORS in a.files else -1
    eb = int(np.asarray(b[META_ERRORS]).reshape(-1)[0]) if META_ERRORS in b.files else -1
    if bad:
        print(f"[FAIL] {len(bad)}/{len(maps)} maps differ:")
        for k, n in bad[:15]:
            print(f"  - {k}: {n} cells differ")
        if len(bad) > 15:
            print(f"  ... and {len(bad) - 15} more")
        return 1
    if ea or eb:
        print(f"[FAIL] identical but build errors a={ea} b={eb}")
        return 1
    print(f"[OK] {len(maps)}/{len(maps)} maps bitwise identical; build errors 0/0")
    return 0


def cmd_ab(args):
    """单变量同进程 A/B：只动 `SPELL_WHIFF_FREE_MAX_COST`。"""
    import rl.action_mask as am
    cur = float(am.SPELL_WHIFF_FREE_MAX_COST)
    base, eb, cards = collect(BASELINE_THRESHOLD)
    after, ea, _ = collect(cur)
    print(f"baseline threshold={BASELINE_THRESHOLD} (cost>-1 always true = pre-W1 semantics)")
    print(f"current  threshold={cur}")
    if eb or ea:
        print(f"[FAIL] build errors baseline={len(eb)} current={len(ea)}")
        for line in (eb + ea)[:10]:
            print(f"  [ERR] {line}")
        return 2

    #: 允许发生变化的卡：费用 ≤ 当前阈值 **且** 是落点伤害型（滚动/非伤害型本来就不受闸门）
    from card_utils import Card
    allowed = sorted(c for c in cards
                     if am._spell_requires_placement_target(c)
                     and float(Card(c).elixir) <= cur)
    print(f"allowed-to-change set (cost<={cur} and placement-damage): {allowed}")

    changed, widened_total, tightened_total = [], 0, 0
    for k in sorted(base):
        x, y = base[k].astype(np.int16), after[k].astype(np.int16)
        w = int(np.count_nonzero((x == 0) & (y == 1)))     # illegal -> legal（放松）
        t = int(np.count_nonzero((x == 1) & (y == 0)))     # legal -> illegal（收紧）
        if w or t:
            changed.append((k, w, t))
        widened_total += w
        tightened_total += t

    changed_cards = sorted({k.split("|")[2] for k, _, _ in changed})
    print(f"\nchanged maps = {len(changed)} / {len(base)}   "
          f"widened cells = {widened_total}   tightened cells = {tightened_total}")
    by_card = {}
    for k, w, t in changed:
        c = k.split("|")[2]
        a = by_card.setdefault(c, [0, 0])
        a[0] += w
        a[1] += t
    for c in sorted(by_card):
        print(f"  {c:<16} widened={by_card[c][0]:<6} tightened={by_card[c][1]}")

    ok = True
    if tightened_total != 0:
        print(f"[FAIL] (1) zero-tightening violated: {tightened_total} cells went legal->illegal")
        ok = False
    else:
        print("[PASS] (1) zero-tightening: no cell went legal->illegal")
    outside = sorted(set(changed_cards) - set(allowed))
    if outside:
        print(f"[FAIL] (2) out-of-scope cards changed: {outside}")
        ok = False
    else:
        print(f"[PASS] (2) zero-out-of-scope: changed cards {changed_cards} subset of allowed")
    #: 方向性期望（W1 专属）：每个允许的卡都必须**真的**发生变化，否则"豁免没生效"
    silent = sorted(set(allowed) - set(changed_cards))
    if silent:
        print(f"[WARN] allowed cards unchanged (threshold may be inert for them): {silent}")
    print("\n[OK] A/B gate PASS" if ok else "\n[FAIL] A/B gate FAIL")
    print("next: standard 128-map corpus (spells are Arrows/Fireball only):")
    print("  python scripts/_mask_diff_snapshot.py --dump <new.npz> && "
          "python scripts/_mask_diff_snapshot.py --compare docs/mask_snapshots/kingtower_after2.npz <new.npz>")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=None)
    ap.add_argument("--compare", nargs=2, default=None)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--ab", action="store_true")
    a = ap.parse_args(argv)
    if a.ab:
        return cmd_ab(a)
    if a.dump:
        return cmd_dump(a)
    if a.compare:
        return cmd_compare(a)
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
