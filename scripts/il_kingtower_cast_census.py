# -*- coding: utf-8 -*-
"""**「法术砸敌方王塔」普查器**（台账 C20 的回归仪器）—— 事件侧口径。

为什么需要两个仪器：
* `scripts/il_probe_kingtower_cast.py` 看**指定帧的内部张量**（机制取证：掩码/plan/cell logits）；
* 本脚本看**整批录像的行为结果**（改前/改后对比），用**客观事件**判命中：
  `frames[i]["towers1"][0]` 的敌方王塔血量落差 **恰好 −206**（= Fireball lv11 对王塔的精确伤害），
  再用「该帧之前 ≤4 帧内有 p0 的 `FireballSpell` 在飞」做归属。

⚠️ 为什么不用投射物**最后一帧位置**当落点（第一版就这么写，**漏判**）：
实测 `runs/il_readout_mixR00` 局2：FireballSpell 可见位置是 y = 8.0→13.0→18.0→**23.0**（帧 0..3），
帧 4 时实体已消失、**王塔正好在这一帧掉 206** ⇒ 真实落点 ≈ y 25~26，而最后一帧位置到王塔中心
还有 6.0 > 3.9 ⇒ 第一版判「未命中」。所以落点用**外推**（`last + (last − prev)`）并同时报两种距离。
⚠️ 也不用 `meta["decks"]`/`deck0[slot-1]` 映射卡名（本仓踩过，会把 Fireball 判成 Tesla）。

用法（仓库根；纯解析可用 python3，stdout 故意 ASCII-only 避开 GBK 陷阱）：
    ./.venv/Scripts/python.exe scripts/il_kingtower_cast_census.py \
        --dir runs/il_readout_mixR00 --dir runs/il_readout_fix_R00 --json out.json
"""
import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np

KING_ENEMY = (9.0, 29.0)
KING_COL = 1.4
SPELL_RADIUS = {"FireballSpell": 2.5, "ArrowsSpell": 3.5}
SPELL_NAMES = {"FireballSpell": "Fireball", "ArrowsSpell": "Arrows"}
KING_DMG = {"FireballSpell": 206.0, "ArrowsSpell": 75.0}


def _newest_replays(d):
    fs = sorted(glob.glob(os.path.join(d, "replays", "*.pkl")))
    best, bestn = None, -1
    for f in fs:
        try:
            with open(f, "rb") as fh:
                n = len(pickle.load(fh).get("games", []))
        except Exception:
            continue
        if n > bestn:
            best, bestn = f, n
    return best


def _proj(frames, i, pid=0):
    """帧 i 里 player=pid 的法术投射物 → [(eid, name, x, y)]。"""
    out = []
    for e in (frames[i].get("entities") or []):
        if e[0] in SPELL_RADIUS and int(e[4]) == pid:
            out.append((int(e[10]) if len(e) > 10 else -1, e[0], float(e[1]), float(e[2])))
    return out


def census(d):
    f = _newest_replays(d)
    if f is None:
        return {"dir": d, "error": "no replays"}
    with open(f, "rb") as fh:
        blob = pickle.load(fh)
    games = blob.get("games", [])
    events, casts, frames_total, unattributed = [], 0, 0, []
    for gi, g in enumerate(games):
        frames = g["frames"]
        frames_total += len(frames)
        #: 法术施放计数（实体首次出现的帧 = 施放帧）
        seen = set()
        for i, fr in enumerate(frames):
            for e in (fr.get("entities") or []):
                if e[0] in SPELL_RADIUS and int(e[4]) == 0:
                    eid = int(e[10]) if len(e) > 10 else -1
                    if eid not in seen:
                        seen.add(eid)
                        casts += 1
        #: 敌方王塔血量序列（towers1[0]）
        hp = []
        for fr in frames:
            t1 = fr.get("towers1") or []
            hp.append(float(t1[0]) if t1 else None)
        for i in range(1, len(hp)):
            if hp[i] is None or hp[i - 1] is None:
                continue
            drop = hp[i - 1] - hp[i]
            if abs(drop - 206.0) > 0.51:
                continue
            cand = None
            for j in (i - 1, i - 2, i - 3, i - 4):
                if j < 0:
                    continue
                for (eid, nm, x, y) in _proj(frames, j):
                    if nm != "FireballSpell":
                        continue
                    if cand is None or j > cand[0]:
                        cand = (j, eid, nm, x, y)
                    if j == i - 1:
                        prev = None
                        for k in range(j - 1, max(-1, j - 4), -1):
                            for (e2, n2, x2, y2) in _proj(frames, k):
                                if e2 == eid:
                                    prev = (x2, y2)
                                    break
                            if prev:
                                break
                        if prev:
                            vx, vy = x - prev[0], y - prev[1]
                            cand = (j, eid, nm, x, y, x + vx, y + vy)
            if cand is None:
                unattributed.append({"game": gi, "frame": i, "t": frames[i]["t"], "drop": drop})
                continue
            if len(cand) == 7:
                _j, eid, nm, x, y, ex, ey = cand
                extrap = True
            else:
                _j, eid, nm, x, y = cand
                ex, ey = x, y
                extrap = False
            r = SPELL_RADIUS[nm]
            d_last = float(np.hypot(x - KING_ENEMY[0], y - KING_ENEMY[1]))
            d_imp = float(np.hypot(ex - KING_ENEMY[0], ey - KING_ENEMY[1]))
            units = []
            for e in (frames[i].get("entities") or []):
                if int(e[4]) != 1 or "Tower" in str(e[0]):
                    continue
                du = float(np.hypot(float(e[1]) - ex, float(e[2]) - ey))
                if du <= r + 1.5:
                    units.append([str(e[0]), round(du, 2)])
            events.append({"game": gi, "frame": i, "t": frames[i]["t"], "spell": SPELL_NAMES[nm],
                           "proj_last": [round(x, 2), round(y, 2)], "impact_est": [round(ex, 2), round(ey, 2)],
                           "extrapolated": extrap, "d_king_last": round(d_last, 2), "d_king_impact": round(d_imp, 2),
                           #: 真实落点在「最后可见位置」与「外推点」**之间** ⇒ 取两者到王塔距离的
                           #: **较小值**判命中（只取外推会高估、只取最后位置会低估，实测两种都漏过）。
                           "hits_king": bool(min(d_last, d_imp) <= r + KING_COL),
                           "enemy_units_in_splash": units,
                           "class": "unit" if units else "king_only"})
    hits = [e for e in events if e["hits_king"]]
    return {"dir": d, "replay": os.path.basename(f), "games": len(games), "frames": frames_total,
            "p0_spell_casts": casts, "king_206_events": len(events), "king_hits": len(hits),
            "king_only": sum(1 for h in hits if h["class"] == "king_only"),
            "unit_splash": sum(1 for h in hits if h["class"] == "unit"),
            "unattributed_206": unattributed, "events": events}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", action="append", required=True)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    rows = []
    for d in a.dir:
        r = census(d)
        rows.append(r)
        if "error" in r:
            print("%s: ERROR %s" % (d, r["error"]))
            continue
        print("%-26s games=%3d frames=%5d p0_spell_casts=%3d | king-206 events=%2d -> hits=%2d "
              "(king_only=%2d / unit_splash=%2d) unattributed=%d"
              % (os.path.basename(d.rstrip("/")), r["games"], r["frames"], r["p0_spell_casts"],
                 r["king_206_events"], r["king_hits"], r["king_only"], r["unit_splash"],
                 len(r["unattributed_206"])))
        for h in r["events"]:
            print("    g%-2d f%-3d t=%-6s %-8s impact~%s d=%-5s hit=%s [%s] units=%s"
                  % (h["game"], h["frame"], h["t"], h["spell"], h["impact_est"], h["d_king_impact"],
                     int(h["hits_king"]), h["class"], h["enemy_units_in_splash"]))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)
        print("[census] saved %s" % a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
