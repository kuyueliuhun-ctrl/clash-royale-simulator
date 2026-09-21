# -*- coding: utf-8 -*-
"""觉醒 S1 门禁回归（【R8】：每个修复/新通道配回归测试）。

被测对象 = `scripts/fl_il_to_bc.py` 的**两个新件**：
  * `evo_slot_cards(vs)` —— 回放牌组后缀 → 本仓觉醒位的**映射与拒收**（预注册 §2 S1 做法）；
  * `install_evo_counters(battle, st)` —— 觉醒触发的**双口径计数器**（只读，不改行为）。

【R19】只跑本文件，不跑全量 selftest：
    PYTHONDONTWRITEBYTECODE=1 ./.venv/Scripts/python.exe scripts/selftest_evo_slots.py

判据（跑前写死；**不达标即 FAIL 并打印哪一条**）：
  T1 映射：`-ev1` 且两处元数据齐备 ⇒ 声明；`-hero` ⇒ **既不声明也不计入未映射**
     （Hero 是另一套机制，预注册 §5 覆盖 0%，必须单列）；`ev*` 但缺周期表/缺 `evo_raw` ⇒ 未映射。
  T2 触发计数（**引擎真跑**）：Knight（cycle=2）连续出 6 次 ⇒ `finish == 2`（第 3、6 次），
     且单兵卡 `wrap == finish`（两口径对账）。
  T3 **负对照**：同样出 6 次但**不声明觉醒位** ⇒ 两个计数器都 == 0
     （证明计数器数的是「觉醒」，不是「出牌」）。
  T4 触发节律：第 4、5 次**不**觉醒（`plays % 3 == 2` 只在 3 的倍数命中）。
  T5 **Hero 互斥**（引擎 `battle.py:3069`）：同时声明觉醒位与 Hero 位 ⇒ 两边都是 0
     （验证我的谓词与引擎的 `not in hero_slots` **同口径**，而不是我自己另写一套）。
  T6 法术觉醒只进 `finish`（不走 `_wrap`）：Zap（cycle=2）出 3 次 ⇒ `finish == 1`，`wrap == 0`。
  T7 `evolution_state` 与计数一致：整局触发数 == `#{k ∈ 1..n : k % (cycle+1) == cycle}`。

⚠️ 本脚本**只读**：`battle.py` / `action_mask.py` / `player.py` 一行未改；
   引擎用**显式构造**的 `BattleState`（不跑 RLEnv、不训练、不落盘）。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src", "clasher_new")
sys.path.insert(0, SRC)
os.chdir(SRC)
#: 复用被测实现（**同一份**函数，不复制粘贴 ⇒ 测的就是脚本里跑的那段）
sys.path.insert(0, HERE)

from rl.io_bootstrap import force_utf8_stdout  # noqa: E402

force_utf8_stdout()

from battle import BattleState  # noqa: E402
from player import PlayerState  # noqa: E402
from card_utils import Card  # noqa: E402
from core import Position  # noqa: E402
from evolutions import EVOLUTION_CYCLES, evolution_state  # noqa: E402
from fl_il_to_bc import evo_slot_cards, install_evo_counters, _force_hand, _force_elixir  # noqa: E402

FAIL = []


def check(name, ok, extra=""):
    print(("  PASS  " if ok else "  FAIL  ") + name + (("  | " + str(extra)) if extra else ""))
    if not ok:
        FAIL.append(name)


#: 固定卡组（8 张，全部本仓在册）：Knight(cycle=2, 单兵) / Zap(cycle=2, 法术) 是主角
DECK = ["Knight", "Zap", "Archers", "Giant", "Musketeer", "Skeletons", "Cannon", "Fireball"]


def _battle():
    bs = BattleState(PlayerState(0, list(DECK), 5.0),
                     PlayerState(1, list(DECK), 5.0), card_level=11)
    bs.time = 0.0
    return bs


def _deploy_pos(bs, player_id=0):
    """用**引擎自己的** `arena.can_deploy_at` 找第一个合法落点（不靠猜坐标）。

    ⚠️ 实测：pid0 的可部署区在 **y < 15**（`get_deploy_zones(0)` = `[(0,1,18,15.0), (6,0,12,6)]`）
    ⇒ 第一版按「y>16」猜坐标，全部 False（本文件自己的踩坑记录，【R10】）。
    """
    for y in (10.0, 8.0, 12.0, 6.0, 4.0, 3.0, 2.0):
        for x in (9.0, 8.0, 10.0, 7.0, 11.0):
            p = Position(x, y)
            if bs.arena.can_deploy_at(p, player_id, battle_state=bs, is_spell=False):
                return p
    return None


def _play(bs, card, pos, n):
    """出 `card` n 次（每次强制进手牌 + 抬圣水），返回成功次数。`pos` 为 None 时直接返回 0。"""
    if pos is None:
        return 0
    ok = 0
    for _ in range(n):
        _force_hand(bs.players[0], card)
        _force_elixir(bs.players[0], float(Card(card).elixir))
        if bs.deploy_card(0, card, pos):
            ok += 1
    return ok


def _counts(bs, declare=(), hero=()):
    st = {}
    if declare:
        bs.players[0].set_evolution_slots(list(declare))
    if hero:
        bs.players[0].set_hero_slots(list(hero))
    c = install_evo_counters(bs, st)
    return c


# ————————————————————————— T1 映射（纯函数） —————————————————————————
print("[T1] evo_slot_cards 映射与拒收")
vs = [
    {"key": "knight-ev1", "resolved": "Knight", "variant": "ev1"},
    {"key": "berserker-hero", "resolved": "Berserker", "variant": "hero"},
    {"key": "notacard-ev1", "resolved": "NotACard", "variant": "ev1"},
]
cards, un = evo_slot_cards(vs)
check("T1.1 `-ev1` + 元数据齐备 ⇒ 声明", cards == ["Knight"], cards)
check("T1.2 `-hero` **不进未映射**（另一套机制，单列）",
      all(u["variant"] != "hero" for u in un) and len(un) == 1, un)
check("T1.3 不在周期表 ⇒ 未映射且标 `in_cycles=False`",
      un and un[0]["resolved"] == "NotACard" and un[0]["in_cycles"] is False, un)
#: 找一张「在周期表里但 `evo_raw` 为空」的卡（若无 ⇒ 照实记 NO-SAMPLE，不假冒 PASS）
#: ⚠️ 周期表里有 `Furnace` 这类**别名键**（`Card('Furnace')` 直接 KeyError，真名 `FirespiritHut`）
#: ⇒ 逐键 try 包住；查不到就把它本身当作「不可声明」的样本（`evo_slot_cards` 的 try 分支）。
_noraw = []
for _n in EVOLUTION_CYCLES:
    try:
        if not getattr(Card(_n), "evo_raw", None):
            _noraw.append(_n)
    except Exception:  # noqa: BLE001
        _noraw.append(_n)
if _noraw:
    c2, u2 = evo_slot_cards([{"key": x + "-ev1", "resolved": x, "variant": "ev1"} for x in _noraw[:1]])
    check("T1.4 在周期表但缺 `evo_raw`（或卡表里没有该键）⇒ 未映射",
          c2 == [] and u2 and u2[0]["has_evo_raw"] is False, (_noraw[:1], u2))
else:
    print("  NO-SAMPLE  T1.4 周期表全部 42 张都有 evo_raw 且都能建 Card（该分支本轮无样本）")

# ————————————————————————— T2/T4 触发计数（引擎真跑） —————————————————————————
print("[T2/T4] Knight（cycle=2）连续 6 次：触发节律 + 双口径对账")
bs = _battle()
pos = _deploy_pos(bs)
check("T2.0 找到引擎认可的合法落点", pos is not None, pos)
if pos is not None:
    c = _counts(bs, declare=("Knight",))
    n_ok = _play(bs, "Knight", pos, 6)
    check("T2.1 6 次出牌全部被引擎接受", n_ok == 6, n_ok)
    check("T2.2 `finish` == 2（第 3、6 次觉醒）", c["finish"]["team"] == 2, c["finish"]["team"])
    check("T2.3 单兵卡双口径对账 `wrap == finish`",
          c["wrap"]["team"] == c["finish"]["team"], (c["wrap"], c["finish"]))
    check("T2.4 对手侧为 0（只数我方）", c["finish"]["opp"] == 0, c["finish"])
    #: T4：逐次检查节律 —— 第 4、5 次不该觉醒（plays=3,4 ⇒ 3%3==0, 4%3==1）
    bs2 = _battle()
    pos2 = _deploy_pos(bs2)
    c2 = _counts(bs2, declare=("Knight",))
    seq = []
    for i in range(1, 7):
        _force_hand(bs2.players[0], "Knight")
        _force_elixir(bs2.players[0], float(Card("Knight").elixir))
        bs2.deploy_card(0, "Knight", pos2)
        seq.append(c2["finish"]["team"])
    check("T4.1 触发节律 = 第 3、6 次（累积计数 [0,0,1,1,1,2]）",
          seq == [0, 0, 1, 1, 1, 2], seq)
    check("T7.1 与 `evolution_state` 解析式一致（n=6, cycle=2 ⇒ 2 次）",
          seq[-1] == sum(1 for k in range(1, 7) if evolution_state(k - 1, "Knight")), seq[-1])

# ————————————————————————— T3 负对照（不声明） —————————————————————————
print("[T3] 负对照：同样出 6 次但**不声明觉醒位**")
bs = _battle()
pos = _deploy_pos(bs)
c = _counts(bs)          # 不 declare
n_ok = _play(bs, "Knight", pos, 6)
check("T3.1 出牌仍被接受（6 次）", n_ok == 6, n_ok)
check("T3.2 `finish` == 0（不声明 ⇒ 永不触发）",
      c["finish"]["team"] == 0 and c["finish"]["opp"] == 0, c["finish"])
check("T3.3 `wrap` == 0", c["wrap"]["team"] == 0 and c["wrap"]["opp"] == 0, c["wrap"])

# ————————————————————————— T5 Hero 互斥 —————————————————————————
print("[T5] Hero 与觉醒同槽互斥（引擎 `battle.py:3069` `card_name not in hero_slots`）")
bs = _battle()
pos = _deploy_pos(bs)
c = _counts(bs, declare=("Knight",), hero=("Knight",))
n_ok = _play(bs, "Knight", pos, 6)
check("T5.1 出牌仍被接受（6 次）", n_ok == 6, n_ok)
check("T5.2 双口径都 == 0（Hero 优先，觉醒被禁）",
      c["finish"]["team"] == 0 and c["wrap"]["team"] == 0,
      (c["finish"], c["wrap"]))

# ————————————————————————— T6 法术觉醒只进 finish —————————————————————————
print("[T6] 法术觉醒（Zap, cycle=2）：只进 `finish`，不走 `_wrap`")
bs = _battle()
#: ⚠️ 本仓**没有**任何生产代码用 `can_deploy_at(..., is_spell=True)`，而那条分支会在
#: `arena.py:181` 调 `TileGrid._is_rolling_projectile_spell()` —— **该方法在 TileGrid 上不存在**
#: ⇒ 走那条分支必 `AttributeError`（**死路里的潜伏 bug**，实测踩到）。本测试因此**照法术真实路径**
#: 用固定落点（`_deploy_card_impl` 对 `type == 'spell'` 跳过全部几何检查，`battle.py:3073`）
#: 并把该 bug 记进文档（**本轮不修**：不在 S1 范围内，【R11】逐项预注册）。
sp = Position(9.0, 16.0)
check("T6.0 法术落点（固定；`is_spell=True` 分支不可用，见上方注释）", True, sp)
if sp is not None:
    c = _counts(bs, declare=("Zap",))
    n_ok = _play(bs, "Zap", sp, 3)
    check("T6.1 Zap 出 3 次全部被接受", n_ok == 3, n_ok)
    check("T6.2 `finish` == 1（第 3 次觉醒）", c["finish"]["team"] == 1, c["finish"]["team"])
    check("T6.3 `wrap` == 0（法术不走出生队列）", c["wrap"]["team"] == 0, c["wrap"]["team"])

# ————————————————————————— 汇总 —————————————————————————
print("")
print("=" * 64)
print("觉醒 S1 门禁：" + ("全部 PASS" if not FAIL else f"FAIL {len(FAIL)} 条 → {FAIL}"))
print("=" * 64)
sys.exit(1 if FAIL else 0)
