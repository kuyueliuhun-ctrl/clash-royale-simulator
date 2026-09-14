#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""掩码 ↔ 引擎 部署合法性对账（R13 类取证；**只读**，不改任何判定逻辑）。

目的：`rl/action_mask.py` 认为可放的位置，引擎 `battle.BattleState.deploy_card`
是否真的接受？反之呢？逐格穷举 18×32 = 576 格 × 双方，对若干代表性卡牌做双向对账。

统计口径：
  maskFalseEngTrue  = 掩码判否、引擎接受   ⇒ **掩码过严**（policy 被剥夺合法动作）
  maskTrueEngFalse  = 掩码判可、引擎拒绝   ⇒ **掩码缺口**（env_wrapper 会打 P1-20 告警）

用法（在 src/clasher_new 下）：
    PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe \
        ../../scripts/_mask_vs_engine_reconcile.py
"""
from __future__ import annotations

import copy
import io
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.getcwd())

from core import Position  # noqa: E402
import battle as B  # noqa: E402
import player as P  # noqa: E402
from rl.action_mask import legal_cells  # noqa: E402
from rl.action_bundle import sub_position  # noqa: E402

DECK = ["Knight", "MiniPekka", "Arrows", "Minions", "Musketeer", "Fireball", "Giant", "Archer"]

# 代表性卡：部队 / 建筑 / 特殊部署 / 法术 / 滚动物体
CARDS = [
    "Knight", "Giant", "Miner", "Musketeer", "Minions", "Archer", "MiniPekka",
    "Cannon", "Mortar", "XBow",
    "Arrows", "Fireball", "Zap", "Poison", "Tornado", "Earthquake", "Freeze",
    "Rage", "Heal", "Lightning", "Log", "BarbLog", "GoblinBarrel", "Mirror",
]


def make(card: str) -> "B.BattleState":
    dc = [card] + [c for c in DECK if c != card]
    return B.BattleState(
        P.PlayerState(0, list(dc), 10.0),
        P.PlayerState(1, list(dc), 10.0),
        card_level=11,
    )


def engine_accepts(bs, card: str, pid: int, pos):
    sim = copy.deepcopy(bs)
    sim.players[pid].elixir = 10.0
    cyc = list(sim.players[pid].cycle)
    if card in cyc:
        cyc.remove(card)
    sim.players[pid].cycle = [card] + cyc
    sim.players[pid].last_card = "Knight"
    try:
        return bool(sim.deploy_card(pid, card, Position(pos.x, pos.y)))
    except Exception as e:  # 运行时异常 = 引擎拒绝路径的另一种表现
        return "ERR:" + type(e).__name__


def main() -> int:
    print("[reconcile] 掩码 vs 引擎 · 逐格穷举 18x32 x 双方；card_level=11")
    print(f"[reconcile] 卡牌 {len(CARDS)} 张：{', '.join(CARDS)}")
    print()
    print("| 卡牌 | pid | 掩码可放格数 | 掩码否/引擎可 (过严) | 掩码可/引擎否 (缺口) | 异常 |")
    print("|---|---|---|---|---|---|")
    rows = []
    for card in CARDS:
        try:
            bs = make(card)
            legal_cells(bs, 0, card)
        except Exception as e:
            print(f"| {card} | - | - | - | - | 前置失败 {type(e).__name__} |")
            continue
        for pid in (0, 1):
            cells = legal_cells(bs, pid, card)
            mfe, mte, exc = [], [], []
            for y in range(32):
                for x in range(18):
                    pos = sub_position(pid, x, y)
                    m = bool(cells[y, x])
                    e = engine_accepts(bs, card, pid, pos)
                    if isinstance(e, str):
                        exc.append((x, y, e))
                        continue
                    if (not m) and e:
                        mfe.append((x, y))
                    if m and (not e):
                        mte.append((x, y))
            rows.append((card, pid, int(cells.sum()), len(mfe), len(mte), len(exc)))
            print(
                f"| {card} | {pid} | {int(cells.sum())} | {len(mfe)} | {len(mte)} | {len(exc)} |"
            )
    print()
    print("## 有差异的条目（逐格样本）")
    print()
    for card, pid, n, a, b, e in rows:
        if a or b or e:
            print(f"- **{card}** pid={pid}: 过严 {a} 格 / 缺口 {b} 格 / 异常 {e}")
    print()
    tot_a = sum(r[3] for r in rows)
    tot_b = sum(r[4] for r in rows)
    print(f"[reconcile] 合计：过严 {tot_a} 格，缺口 {tot_b} 格，异常 {sum(r[5] for r in rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
