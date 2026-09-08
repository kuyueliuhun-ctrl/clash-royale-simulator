# -*- coding: utf-8 -*-
"""P0 掩码优化逐位对账脚本。

在改动前后各跑一次，dump 各状态 legal_cells 位图 → npy，逐位 diff。
用法: python scripts/_mask_diff_snapshot.py <out.npy>
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src", "clasher_new")
sys.path.insert(0, SRC)
os.chdir(SRC)

import numpy as np


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


def main(out_path):
    from rl.action_mask import legal_cells

    blobs = {}
    for name, bs in build_states():
        for pid in (0, 1):
            for card in HAND:
                try:
                    cells = legal_cells(bs, pid, card)
                except Exception as e:
                    cells = np.zeros((32, 18), dtype=bool)
                    print(f"[ERR] {name}/p{pid}/{card}: {e!r}")
                blobs[f"{name}|p{pid}|{card}"] = cells.astype(np.uint8)
    np.savez_compressed(out_path, **blobs)
    print(f"snapshot: {len(blobs)} maps -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "docs/_mask_baseline.npz")
