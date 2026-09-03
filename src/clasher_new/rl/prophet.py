"""先知规划器 ProphetPlanner（规划文档 5.1 / 5.2）。

使用特权完整状态（get_prophet_state），输出与 BeliefPlanner 同构的 PlanToken，
作为训练期教师 / 监督信号。

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
from rl.plan_space import PlanToken, FOCUS_REGIONS

#: 推进压力阈值（与单位数量/HP×费用匹配，避免 6 座塔把意图钉死在 defend_*）
PRESSURE_THRESHOLD = 2.0


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


class ProphetPlanner:
    """特权状态启发式先知。"""

    def plan(self, full_state: dict) -> PlanToken:
        time = full_state["time"]
        my_elixir = full_state["my_elixir"]
        opp_elixir = full_state["opp_elixir"]
        my_crown = full_state["my_crown"]
        opp_crown = full_state["opp_crown"]
        opp_towers = full_state["opp_towers"]
        my_towers = full_state["my_towers"]
        opp_cycle = full_state.get("opp_cycle") or []

        # 敌我推进压力（完整信息，排除静态塔，P1-4）
        threat, my_pressure = 0.0, 0.0
        enemy_xs = []
        for e in full_state["entities"]:
            if _is_tower(e["name"]):
                continue
            if e["player"] == 1:
                threat += 1.0 + max(0.0, (16 - e["pos"][1]) / 16.0)
                enemy_xs.append(e["pos"][0])
            else:
                my_pressure += 1.0 + max(0.0, (e["pos"][1] - 16) / 16.0)
        enemy_x = float(np.mean(enemy_xs)) if enemy_xs else 9.0

        intent = "cycle_and_wait"
        if threat >= PRESSURE_THRESHOLD and threat >= my_pressure * 0.8:
            intent = "defend_left" if enemy_x < 9 else "defend_right"
        elif my_pressure >= PRESSURE_THRESHOLD:
            intent = "push_left" if enemy_x < 9 else "push_right"
        elif opp_crown > my_crown:
            intent = "push_left" if enemy_x < 9 else "push_right"
        elif time > 180 and my_elixir >= opp_elixir + 1:
            intent = "counterpush"
        elif opp_elixir < 2.0:
            # P1-16：对手低圣水 → 趁虚进攻
            intent = "push_left" if enemy_x < 9 else "push_right"

        # 特权信息：选牌用对手塔血与圣水差
        best, best_score = None, -1e9
        my_cycle = full_state["my_cycle"]
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

        risk = 0.5
        if my_elixir > opp_elixir + 2:
            risk = 0.7
        elif my_elixir < opp_elixir - 1:
            risk = 0.3
        if opp_elixir < 2.0:
            risk = max(risk, 0.75)
        bundle_hint = 2 if (my_elixir >= 6 and threat >= PRESSURE_THRESHOLD
                            and my_pressure >= PRESSURE_THRESHOLD) else 1
        # P1-16：对手手牌有法术且我方要 bundle 进攻 → 收敛 bundle 规模，防被清场
        if bundle_hint >= 2 and opp_cycle and any(
                Card(c).type == "spell" for c in opp_cycle[:4]):
            bundle_hint = 1
        region = _region_from_intent(intent)
        return PlanToken(
            macro_intent=intent,
            focus_region=region,
            suggested_card=best,
            bundle_size_hint=bundle_hint,
            combo_hint=1 if bundle_hint >= 2 else 0,
            risk_profile=risk,
            value_estimate=float((my_elixir - opp_elixir) / 10.0 + my_pressure - threat),
        )
