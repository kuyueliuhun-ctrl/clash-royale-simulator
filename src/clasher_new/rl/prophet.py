"""先知规划器 ProphetPlanner（规划文档 5.1 / 5.2）。

使用特权完整状态（get_prophet_state），输出与 BeliefPlanner 同构的 PlanToken，
作为训练期教师 / 监督信号（flow/run_league 30% 帧由 prophet 出 plan）。

Phase 2 v1 pp 组（docs/rl_plan_design_v1.md §4/§5）：
- 特权精确版意图（bp 只能靠信念/弱规则，pp 直读精确状态）：
  * punish —— 对手圣水精确读数（opp_elixir），另一路反推；
  * anti_spell —— 直读对手手牌/进手序（opp_cycle）里的法术 → opp_spell_threat 精确；
  * save_ace —— 藏/解除时机用对手圣水 + 对手手牌反制法术精确判定；
  * spell_finish —— 对手塔血精确读数；
  * king_activate / protect_backline —— bp 尚未实现的意图由 pp 先示范
    （protect_backline 含「对手手牌有切后排单位」的预判版）。
- 其余 bp 可产出的新意图（soft_control/spell_trade/pull/push_commit/setup_wait/
  cycle_small）pp 同步实现 → 30% prophet 帧与 70% bp 帧标签一致（消噪），
  不实现 pre_defend/bait（与 bp 相同，设计 backlog）。
- 一帧只输出一个 macro_intent：优先链与 BeliefPlanner 同序（紧急度降序），
  未命中回退旧 8 意图逻辑（保留 P1-4/15/16 修复语义）。

修复：
- P1-4：实体压力统计过滤静态塔（"Tower"），空场开局不再恒为 defend_*；
- P1-15：focus_region 由 intent + 敌方重心推导，不再用风险标量线性映射；
- P1-16：消费对手手牌/牌序/圣水特权信息（低圣水 → 偏向进攻；对面有法术 → 降低 bundle 规模）。
"""

import os
import sys

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import numpy as np

from card_utils import Card
from rl.plan_space import PlanToken
from rl.belief_planner import (
    PRESSURE_THRESHOLD, KING_ACTIVATE_PRINCESS_HP,
    LANE_SPLIT_X, BRIDGE_Y, OWN_HALF_EDGE, LATE_S,
    SOFT_CONTROL_CARDS, TRADE_SPELL_CARDS, FINISH_SPELL_CARDS, TANK_CARDS,
    PULL_TARGET_CARDS, BACKLINE_CARDS, BACKLINE_HARASSER_CARDS, ACE_CARDS,
    _SPELL_THREAT_KIND,
)

def _is_tower(name: str) -> bool:
    return "Tower" in name


def _region_from_intent(intent: str) -> str:
    mapping = {
        "defend_left": "own_left",
        "defend_right": "own_right",
        "defend_king": "own_center",
        "push_left": "enemy_left",
        "push_right": "enemy_right",
        "counterpush": "enemy_center",
        "spell_value": "enemy_center",
        "cycle_and_wait": "own_center",
    }
    return mapping.get(intent, "own_center")


def _units(fs, player_id: int):
    """特权状态里的非塔部署实体（塔 id/名称除外，行为与 bp 压力统计一致）。"""
    return [e for e in fs["entities"]
            if e["player"] == player_id and not _is_tower(e["name"])]


def _mean_x(units) -> float:
    xs = [float(e["pos"][0]) for e in units]
    return float(np.mean(xs)) if xs else 9.0


def _threat_and_my_pressure(fs):
    threat = 0.0
    my_pressure = 0.0
    for e in fs["entities"]:
        if _is_tower(e["name"]):
            continue
        y = float(e["pos"][1])
        if e["player"] == 1:
            threat += 1.0 + max(0.0, (16 - y) / 16.0)
        else:
            my_pressure += 1.0 + max(0.0, (y - 16) / 16.0)
    return threat, my_pressure


def _closest_enemy(fs):
    """最接近我方塔的敌方部署单位（None=无）。口径与 bp._closest_threat 一致。"""
    best, best_w = None, -1.0
    for e in _units(fs, 1):
        w = 1.0 + max(0.0, (16 - float(e["pos"][1])) / 16.0)
        if w > best_w:
            best_w, best = w, e
    return best


def _pressing_enemy(fs) -> bool:
    """存在已进入我方半场/桥头的敌方单位（y ≤ 河中心+1）→ 防守优先守卫。"""
    return any(float(e["pos"][1]) <= BRIDGE_Y + 1.0 for e in _units(fs, 1))


def _side(x: float) -> str:
    return "left" if x < LANE_SPLIT_X else "right"


def _own_region(x: float) -> str:
    return "own_left" if _side(x) == "left" else "own_right"


def _enemy_region(x: float) -> str:
    return "enemy_left" if _side(x) == "left" else "enemy_right"


def _opposite_enemy_region(x: float) -> str:
    """敌方重心在 x → 建议进攻的另一路 region（punish 用）。"""
    return "enemy_right" if x < LANE_SPLIT_X else "enemy_left"


def _hand_slot(cycle, card_name):
    return cycle.index(card_name) + 1 if card_name in cycle[:4] else None


def _spell_threat_in(cycle, depth: int = 6):
    """对手 cycle 前 depth 张（手牌+进手序）里第一张强法术 → OPP_SPELL_THREATS 值。

    pp 特权版：直读对手手牌/牌序，威胁类型精确；depth 覆盖手牌 + 接下来两张进手。
    """
    for card in cycle[:depth]:
        kind = _SPELL_THREAT_KIND.get(card)
        if kind is not None:
            return kind
    return None


def _pick_suggested(fs, intent: str):
    """从可出手牌里按启发式选一张（与 bp._pick_suggested_card 同启发式）。"""
    my_cycle = fs["my_cycle"]
    my_elixir = float(fs["my_elixir"])
    best, best_score = None, -1e9
    for i in range(4):
        card = my_cycle[i]
        if my_elixir < Card(card).elixir or card == "Mirror":
            continue
        ctype = Card(card).type
        score = 0.0
        if intent.startswith("defend"):
            score = 2.0 if ctype == "spell" else 0.0
            score += (10.0 - Card(card).elixir)
        elif intent.startswith("push") or intent == "counterpush":
            score = Card(card).elixir if ctype == "character" else Card(card).elixir * 0.5
        else:
            score = (10.0 - Card(card).elixir) * 0.5
        if score > best_score:
            best_score, best = score, i + 1
    return best


class ProphetPlanner:
    """特权状态启发式先知（Phase 2 v1：全意图组 + 特权精确字段）。"""

    # ---- Phase 2 v1 新意图检测（与 bp 同优先链，读特权 full_state）----

    def _soft_control(self, fs):
        tu = _closest_enemy(fs)
        if tu is None or float(tu["pos"][1]) > BRIDGE_Y + 1.0:
            return None
        for card in SOFT_CONTROL_CARDS:
            slot = _hand_slot(fs["my_cycle"], card)
            if slot is not None and fs["my_elixir"] >= Card(card).elixir:
                return PlanToken(
                    macro_intent="soft_control",
                    focus_region=_own_region(float(tu["pos"][0])),
                    suggested_card=slot, target_kind="unit",
                    placement_hint="none", elixir_budget=0.4, risk_profile=0.6,
                    value_estimate=-2.0)
        return None

    def _spell_trade(self, fs):
        tu = _closest_enemy(fs)
        if tu is None or float(tu["pos"][1]) > OWN_HALF_EDGE + 1.5:
            return None
        if tu["name"] in PULL_TARGET_CARDS:
            return None  # 血牛解法术亏 → 交给 _pull/单位
        cost = float(Card(tu["name"]).elixir)
        if cost < 3.0:
            return None
        for card in TRADE_SPELL_CARDS:
            slot = _hand_slot(fs["my_cycle"], card)
            if slot is not None and fs["my_elixir"] >= Card(card).elixir:
                return PlanToken(
                    macro_intent="spell_trade",
                    focus_region=_own_region(float(tu["pos"][0])),
                    suggested_card=slot, target_kind="unit",
                    placement_hint="none", elixir_budget=0.5, risk_profile=0.5,
                    value_estimate=float(min(cost, 6.0)) * 0.5)
        return None

    def _protect_backline(self, fs):
        """保后排：a) 近战已贴近/正接近 → 放单位吸仇恨；
        b) pp 预判版——对手手牌有切后排单位且我方后排暴露 → 提前保护。"""
        my_units = _units(fs, 0)
        backlines = [e for e in my_units
                     if float(e["pos"][1]) <= OWN_HALF_EDGE + 0.5
                     and e["name"] in BACKLINE_CARDS]
        predictive = False
        target = None
        for bl in backlines:
            bx, by = float(bl["pos"][0]), float(bl["pos"][1])
            for e in _units(fs, 1):
                if e["name"] in PULL_TARGET_CARDS:
                    continue  # 血牛交给 _pull
                if float(e["pos"][1]) > BRIDGE_Y + 1.5:
                    continue
                dx = float(e["pos"][0]) - bx
                dy = float(e["pos"][1]) - by
                if dx * dx + dy * dy <= 36.0:  # 距离 ≤ 6
                    target = bl
                    break
            if target is not None:
                break
        if target is None and backlines and not _pressing_enemy(fs):
            # 预判：对手手牌有切后排突进（≤4 费近战）且我方后排还没被贴
            if any(Card(c).type == "character"
                   and Card(c).elixir <= 4.0
                   and c in BACKLINE_HARASSER_CARDS
                   for c in fs["opp_cycle"][:4]):
                # 只对"已暴露"的后排（过桥前压、离桥近）做预判
                exposed = [e for e in backlines if float(e["pos"][1]) >= 9.0]
                if exposed:
                    target = exposed[0]
                    predictive = True
        if target is None:
            return None
        for i, card in enumerate(fs["my_cycle"][:4]):
            c = Card(card)
            if c.type in ("character", "building") and 1.0 <= c.elixir <= 4.0 \
                    and fs["my_elixir"] >= c.elixir and card != "Mirror":
                return PlanToken(
                    macro_intent="protect_backline",
                    focus_region=_own_region(float(target["pos"][0])),
                    suggested_card=i + 1, target_kind="my_backline",
                    placement_hint="none", elixir_budget=0.4, risk_profile=0.5,
                    value_estimate=0.3 if predictive else 0.5)
        return None

    def _pull(self, fs):
        target = None
        for e in _units(fs, 1):
            if e["name"] not in PULL_TARGET_CARDS:
                continue
            y = float(e["pos"][1])
            if OWN_HALF_EDGE - 4.0 <= y <= BRIDGE_Y + 2.0:
                target = e
                break
        if target is None:
            return None
        for i, card in enumerate(fs["my_cycle"][:4]):
            c = Card(card)
            if c.type in ("character", "building") and 1.0 <= c.elixir <= 3.0 \
                    and fs["my_elixir"] >= c.elixir and card != "Mirror":
                near_edge = float(target["pos"][0]) < LANE_SPLIT_X - 4.0 \
                    or float(target["pos"][0]) > LANE_SPLIT_X + 4.0
                return PlanToken(
                    macro_intent="pull",
                    focus_region="own_center", suggested_card=i + 1,
                    target_kind="unit",
                    placement_hint="pull_across" if near_edge else "pull_aggro",
                    elixir_budget=0.3, risk_profile=0.5, value_estimate=-1.0)
        return None

    def _punish(self, fs):
        """趁虚另一路（pp 特权精确版）：对手圣水直读 + 沉底重心反侧。"""
        if float(fs["opp_elixir"]) >= 2.5:
            return None
        if _pressing_enemy(fs):
            return None  # 压境先防
        enemy_units = _units(fs, 1)
        if enemy_units:
            region = _opposite_enemy_region(_mean_x(enemy_units))
        else:
            my_x = _mean_x(_units(fs, 0))
            region = "enemy_left" if my_x >= LANE_SPLIT_X else "enemy_right"
        for i, card in enumerate(fs["my_cycle"][:4]):
            c = Card(card)
            if (card in TANK_CARDS or (c.type == "character" and c.elixir >= 5.0)) \
                    and fs["my_elixir"] >= c.elixir:
                return PlanToken(
                    macro_intent="punish", focus_region=region,
                    suggested_card=i + 1, target_kind="tower",
                    placement_hint="none", elixir_budget=0.7, risk_profile=0.8,
                    value_estimate=2.0)
        return None

    def _push_commit(self, fs):
        tank = None
        for e in _units(fs, 0):
            if e["name"] not in TANK_CARDS:
                continue
            y = float(e["pos"][1])
            if 8.0 <= y <= 22.0:
                tank = e
                break
        if tank is None:
            return None
        for i, card in enumerate(fs["my_cycle"][:4]):
            c = Card(card)
            if c.type == "character" and 3.0 <= c.elixir <= 6.0 \
                    and card not in TANK_CARDS and fs["my_elixir"] >= c.elixir:
                return PlanToken(
                    macro_intent="push_commit",
                    focus_region=_own_region(float(tank["pos"][0])),
                    suggested_card=i + 1, target_kind="unit",
                    placement_hint="support_zone", elixir_budget=0.6,
                    risk_profile=0.7, value_estimate=1.5)
        return None

    def _spell_finish(self, fs):
        if float(fs["time"]) < LATE_S:
            return None
        candidates = []
        if fs["opp_towers"][1] > 0:
            candidates.append(("enemy_left", float(fs["opp_towers"][1])))
        if fs["opp_towers"][2] > 0:
            candidates.append(("enemy_right", float(fs["opp_towers"][2])))
        if not candidates:
            return None
        region, hp = min(candidates, key=lambda kv: kv[1])
        if hp > 1200.0:
            return None
        for card in FINISH_SPELL_CARDS:
            slot = _hand_slot(fs["my_cycle"], card)
            if slot is not None and fs["my_elixir"] >= Card(card).elixir:
                return PlanToken(
                    macro_intent="spell_finish", focus_region=region,
                    suggested_card=slot, target_kind="tower",
                    placement_hint="none", elixir_budget=0.45, risk_profile=0.6,
                    value_estimate=1.2)
        return None

    def _setup_wait(self, fs):
        threat, _ = _threat_and_my_pressure(fs)
        if threat >= PRESSURE_THRESHOLD or _pressing_enemy(fs):
            return None
        if any(e["name"] in TANK_CARDS for e in _units(fs, 0)):
            return None  # 已有坦克在场上 → 轮不到 setup
        for card in TANK_CARDS:
            slot = _hand_slot(fs["my_cycle"], card)
            if slot is not None and fs["my_elixir"] >= Card(card).elixir - 0.5:
                return PlanToken(
                    macro_intent="setup_wait",
                    focus_region="own_center", suggested_card=slot,
                    target_kind="none", placement_hint="none",
                    elixir_budget=0.6, risk_profile=0.4, value_estimate=0.5)
        return None

    def _king_activate(self, fs):
        """激活国王塔：公主塔残血/被破 + 敌方血牛/大单位接近国王塔中轴。"""
        if min(float(fs["my_towers"][1]), float(fs["my_towers"][2])) > \
                KING_ACTIVATE_PRINCESS_HP:
            return None
        heavy = None
        for e in _units(fs, 1):
            c = Card(e["name"])
            if e["name"] not in TANK_CARDS and not (c.type == "character"
                                                    and c.elixir >= 5.0):
                continue
            x, y = float(e["pos"][0]), float(e["pos"][1])
            if 3.5 <= x <= 14.5 and y <= BRIDGE_Y + 2.5:
                heavy = e
                break
        if heavy is None:
            return None
        for i, card in enumerate(fs["my_cycle"][:4]):
            c = Card(card)
            if c.type in ("character", "building") and 1.0 <= c.elixir <= 3.0 \
                    and fs["my_elixir"] >= c.elixir and card != "Mirror":
                return PlanToken(
                    macro_intent="king_activate",
                    focus_region="own_center", suggested_card=i + 1,
                    target_kind="unit", placement_hint="king_front",
                    elixir_budget=0.35, risk_profile=0.5, value_estimate=0.8)
        return None

    def _anti_spell(self, fs):
        """防法术（pp 特权精确版）：直读对手手牌+进手序里的强法术。"""
        if _pressing_enemy(fs):
            return None
        kind = _spell_threat_in(fs["opp_cycle"])
        if kind is None:
            return None
        for i, card in enumerate(fs["my_cycle"][:4]):
            c = Card(card)
            if c.type == "character" and 3.0 <= c.elixir <= 6.0 \
                    and fs["my_elixir"] >= c.elixir and card not in TANK_CARDS:
                return PlanToken(
                    macro_intent="anti_spell", focus_region="own_center",
                    suggested_card=i + 1, target_kind="none",
                    placement_hint="anti_spell_zone", opp_spell_threat=kind,
                    elixir_budget=0.5, risk_profile=0.4, value_estimate=0.0)
        return None

    def _save_ace(self, fs):
        """藏终结卡（pp 时机版）：对手低圣水 + 手牌无反制法术 + 己方坦克进场
        = 最强一波（解除藏）；否则 hold_mask 指名别出 + 留费。"""
        if not fs["my_cycle"]:
            return None
        ace_slots = []
        for card in ACE_CARDS:
            slot = _hand_slot(fs["my_cycle"], card)
            if slot is not None:
                ace_slots.append(slot)
        if not ace_slots:
            return None
        if _pressing_enemy(fs):
            return None  # 防守中 ace 可能当解牌用，不硬藏
        # 最强一波：己方坦克推进中 + 对手低圣水 + 对手手牌/进手序无强法术
        tank_pushing = any(
            e["name"] in TANK_CARDS and float(e["pos"][1]) >= 10.0
            for e in _units(fs, 0))
        if tank_pushing and float(fs["opp_elixir"]) < 3.0 \
                and _spell_threat_in(fs["opp_cycle"]) is None:
            return None  # 解除藏：ace 该出场（交给后面意图/模型出）
        hold = 0
        for slot in ace_slots:
            hold |= 1 << (slot - 1)
        suggested = None
        for i, card in enumerate(fs["my_cycle"][:4]):
            c = Card(card)
            if (i + 1) not in ace_slots and c.elixir <= fs["my_elixir"] \
                    and c.type != "spell" and card != "Mirror":
                suggested = i + 1
                break
        return PlanToken(
            macro_intent="save_ace", focus_region="own_center",
            suggested_card=suggested, target_kind="none",
            placement_hint="none", elixir_budget=0.4, risk_profile=0.4,
            hold_mask=hold, value_estimate=-0.3)

    def _cycle_small(self, fs):
        threat, _ = _threat_and_my_pressure(fs)
        if threat >= PRESSURE_THRESHOLD or _pressing_enemy(fs):
            return None
        for i, card in enumerate(fs["my_cycle"][:4]):
            c = Card(card)
            if c.elixir <= 2.0 and fs["my_elixir"] >= c.elixir + 3.0 \
                    and card != "Mirror":
                return PlanToken(
                    macro_intent="cycle_small",
                    focus_region="own_center", suggested_card=i + 1,
                    target_kind="none", placement_hint="none",
                    elixir_budget=0.25, risk_profile=0.3, value_estimate=0.2)
        return None

    # ---- 主入口 ----

    def plan(self, full_state: dict) -> PlanToken:
        fs = full_state
        threat, my_pressure = _threat_and_my_pressure(fs)

        # —— Phase 2 v1 优先链（与 bp 同序 + pp 专属 protect/king）——
        for detector in (self._soft_control, self._spell_trade,
                         self._protect_backline, self._pull, self._punish,
                         self._push_commit, self._spell_finish, self._setup_wait,
                         self._king_activate, self._anti_spell, self._save_ace,
                         self._cycle_small):
            tok = detector(fs)
            if tok is not None:
                return tok

        # —— 回退：旧 8 意图逻辑（保留 P1-4/15/16）——
        enemy_x = _mean_x(_units(fs, 1))
        intent = "cycle_and_wait"
        if threat >= PRESSURE_THRESHOLD and threat >= my_pressure * 0.8:
            intent = "defend_left" if enemy_x < LANE_SPLIT_X else "defend_right"
        elif my_pressure >= PRESSURE_THRESHOLD:
            intent = "push_left" if enemy_x < LANE_SPLIT_X else "push_right"
        elif fs["opp_crown"] > fs["my_crown"]:
            intent = "push_left" if enemy_x < LANE_SPLIT_X else "push_right"
        elif fs["time"] > 180 and fs["my_elixir"] >= fs["opp_elixir"] + 1:
            intent = "counterpush"
        elif fs["opp_elixir"] < 2.0:
            intent = "push_left" if enemy_x < LANE_SPLIT_X else "push_right"

        suggested = _pick_suggested(fs, intent)
        risk = 0.5
        if fs["my_elixir"] > fs["opp_elixir"] + 2:
            risk = 0.7
        elif fs["my_elixir"] < fs["opp_elixir"] - 1:
            risk = 0.3
        if fs["opp_elixir"] < 2.0:
            risk = max(risk, 0.75)
        bundle_hint = 2 if (fs["my_elixir"] >= 6 and threat >= PRESSURE_THRESHOLD
                            and my_pressure >= PRESSURE_THRESHOLD) else 1
        # P1-16：对手手牌有法术且我方要 bundle 进攻 → 收敛 bundle 规模
        if bundle_hint >= 2 and any(Card(c).type == "spell"
                                    for c in fs["opp_cycle"][:4]):
            bundle_hint = 1
        region = _region_from_intent(intent)
        return PlanToken(
            macro_intent=intent,
            focus_region=region,
            suggested_card=suggested,
            bundle_size_hint=bundle_hint,
            combo_hint=1 if bundle_hint >= 2 else 0,
            risk_profile=risk,
            value_estimate=float((fs["my_elixir"] - fs["opp_elixir"]) / 10.0
                                 + my_pressure - threat),
        )
