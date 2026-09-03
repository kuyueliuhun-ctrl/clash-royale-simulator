"""信念规划器 BeliefPlanner（规划文档 5.5 / 3.7）。

只用可见观测 + b_t 出可部署计划；规则版 + 后验采样版。

修复：
- P1-4：实体压力统计过滤静态塔，空场开局不再恒为 defend_*；
- P1-15：focus_region 由 intent 推导（defend_left → own_left、push_right → enemy_right）。
"""

import os
import sys

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from typing import Optional

import numpy as np

from card_utils import Card
from rl.belief import BeliefState, BeliefInference
from rl.plan_space import PlanToken, MACRO_INTENTS, FOCUS_REGIONS

#: 推进压力阈值（与单位数量/HP×费用匹配）
PRESSURE_THRESHOLD = 2.0


def _is_tower(name: str) -> bool:
    return "Tower" in name


def _enemy_pressure(battle):
    """敌我双方实体在各自半场的推进压力（粗略威胁估计，排除静态塔）。"""
    threat = 0.0
    my_pressure = 0.0
    for e in battle.entities.values():
        if not e.is_alive:
            continue
        if _is_tower(e.name):
            continue
        # 粗略：敌方单位越靠近我方塔威胁越大
        if e.player == 1:
            threat += 1.0 + max(0.0, (16 - e.position.y) / 16.0)
        else:
            my_pressure += 1.0 + max(0.0, (e.position.y - 16) / 16.0)
    return threat, my_pressure


def _enemy_main_x(battle):
    xs = [e.position.x for e in battle.entities.values()
          if e.is_alive and e.player == 1 and not _is_tower(e.name)]
    return (float(np.mean(xs)) if xs else 9.0)


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


def _pick_suggested_card(battle, player_id, belief: Optional[BeliefState], intent: str):
    """从可出手牌里按启发式选一张：防守选威胁应对，进攻选最高费推进。"""
    p = battle.players[player_id]
    best, best_score = None, -1e9
    for i in range(4):
        card = p.cycle[i]
        if p.elixir < Card(card).elixir or card == "Mirror":
            continue
        ctype = Card(card).type
        score = 0.0
        if intent.startswith("defend"):
            # 防守：法术/低费应对价值高
            if ctype == "spell":
                score = 2.0
            score += (10.0 - Card(card).elixir)
        elif intent.startswith("push") or intent == "counterpush":
            # 进攻：坦克/高费推进
            if ctype == "character":
                score = Card(card).elixir
            else:
                score = Card(card).elixir * 0.5
        else:
            score = (10.0 - Card(card).elixir) * 0.5
        if score > best_score:
            best_score, best = score, i + 1
    return best


class BeliefPlanner:
    """基于信念状态的规划器。"""

    def __init__(self, use_posterior_sampling: bool = False, n_samples: int = 8):
        self.use_posterior_sampling = use_posterior_sampling
        self.n_samples = n_samples

    def plan(self, battle, belief: BeliefState, obs=None) -> PlanToken:
        threat, my_pressure = _enemy_pressure(battle)
        intent = "cycle_and_wait"
        if threat >= PRESSURE_THRESHOLD and threat >= my_pressure * 0.8:
            intent = "defend_left" if _enemy_main_x(battle) < 9 else "defend_right"
        elif my_pressure >= PRESSURE_THRESHOLD:
            intent = "push_left" if _enemy_main_x(battle) < 9 else "push_right"
        elif threat > 0.0:
            intent = "defend_king"
        # 后验采样调节 risk_profile：信念越确定越敢激进
        risk = 0.5
        if belief is not None:
            unc = belief.uncertainty
            risk = float(np.clip(0.75 - 0.5 * unc, 0.1, 0.9))
            if belief.next_probs is not None and belief.next_probs.max() > 0.6:
                # 高置信知道对手下一张 → 更敢抓机会
                risk = min(0.95, risk + 0.15)
        suggested = _pick_suggested_card(battle, 0, belief, intent)
        bundle_hint = 2 if (my_pressure >= PRESSURE_THRESHOLD
                            and threat >= PRESSURE_THRESHOLD and risk > 0.6) else 1
        region = _region_from_intent(intent)
        return PlanToken(
            macro_intent=intent,
            focus_region=region,
            suggested_card=suggested,
            bundle_size_hint=bundle_hint,
            combo_hint=1 if bundle_hint >= 2 else 0,
            risk_profile=risk,
            value_estimate=float(my_pressure - threat),
        )
