"""信念规划器 BeliefPlanner（规划文档 5.5 / 3.7）。

只用可见观测 + b_t 出可部署计划；规则版 + 后验采样版。

Phase 2 v1 扩展（docs/rl_plan_design_v1.md）：
- 意图库补 6 个 bp 可产出新意图（70% 帧即可示范，样本量大）：
  ``soft_control``（对威胁单位放冰冻/藤蔓）→ ``spell_trade``（法术解过桥/威胁单位）→
  ``pull``（拉扯血牛改道/横穿）→ ``push_commit``（坦克推进中身后跟后排）→
  ``setup_wait``（低压力憋组合沉底）→ ``cycle_small``（无压力小费过牌保手牌质量）；
- 一帧只输出一个 macro_intent：新意图按紧急度优先，未命中回退旧 8 意图逻辑；
- 位置类意图只给**策略位 hint**（placement_hint）不给坐标——模型从 grid 自学精确格。

修复：
- P1-4：实体压力统计过滤静态塔，空场开局不再恒为 defend_*；
- P1-15：focus_region 由 intent 推导（defend_left → own_left、push_right → enemy_right）。
"""

import os
import sys

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from typing import Optional, Tuple

import numpy as np

from card_utils import Card
from rl.belief import BeliefState, BeliefInference
from rl.plan_space import PlanToken, MACRO_INTENTS, FOCUS_REGIONS

#: 推进压力阈值（与单位数量/HP×费用匹配）
PRESSURE_THRESHOLD = 2.0

#: 蓝方视角坐标（player0 塔 y≈3-6.5；河 y≈15-17；红方塔 y≈25.5）
BRIDGE_Y = 16.0        # 河中心（约 16）
OWN_HALF_EDGE = 15.0   # 己方半场边界（y < 15 为纯己方半场）
LANE_SPLIT_X = 9.0     # 左/右路分界

#: 软控法术（对威胁单位打断/拖延）
SOFT_CONTROL_CARDS = ("Freeze", "Vines", "Tornado")
#: 伤害/解牌法术（spell_trade 用）
TRADE_SPELL_CARDS = ("Fireball", "Arrows", "Rocket", "Lightning", "Poison",
                     "Earthquake", "Void", "TheLog", "Snowball", "Zap", "Tornado")
#: 推进坦克（血牛 / 推进主角）
TANK_CARDS = ("Giant", "Golem", "ElixirGolem", "RoyalGiant", "GoblinGiant",
              "ElectroGiant", "MegaKnight", "Pekka", "GiantSkeleton", "HogRider")
#: 血牛（pull 的目标：只锁塔 / 大体积推进）
PULL_TARGET_CARDS = ("Giant", "Golem", "ElixirGolem", "RoyalGiant", "GoblinGiant",
                     "ElectroGiant", "MegaKnight", "Pekka", "GiantSkeleton")


def _is_tower(name: str) -> bool:
    return "Tower" in name


def _deployable_entity(e) -> bool:
    """可部署单位/建筑（排除塔、法术弹道/区域效果等临时实体）。"""
    if not e.is_alive or _is_tower(getattr(e, "name", "")):
        return False
    return getattr(getattr(e, "data", None), "type", "") in ("character", "building")


def _enemy_pressure(battle):
    """敌我双方实体在各自半场的推进压力（粗略威胁估计，排除静态塔）。"""
    threat = 0.0
    my_pressure = 0.0
    for e in battle.entities.values():
        if not e.is_alive:
            continue
        if _is_tower(e.name):
            continue
        # 粗略：敌方单位越靠近我方塔威胁越大（y 越小越近 player0 塔）
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


def _side(x: float) -> str:
    return "left" if x < LANE_SPLIT_X else "right"


def _own_region(x: float) -> str:
    s = _side(x)
    return "own_left" if s == "left" else "own_right"


def _enemy_region(x: float) -> str:
    s = _side(x)
    return "enemy_left" if s == "left" else "enemy_right"


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


def _hand_slot(p, card_name):
    return p.cycle.index(card_name) + 1 if card_name in p.cycle[:4] else None


def _hand_card_matching(p, predicate):
    """手牌（前 4）中第一个满足 predicate 的卡名。"""
    for card in p.cycle[:4]:
        if card != "Mirror" and predicate(card):
            return card
    return None


def _closest_threat(battle):
    """最接近我方塔的敌方部署单位（None=无）。"""
    best, best_w = None, -1.0
    for e in battle.entities.values():
        if e.player != 1 or not _deployable_entity(e):
            continue
        w = 1.0 + max(0.0, (16 - e.position.y) / 16.0)   # 与 _enemy_pressure 同口径
        if w > best_w:
            best_w, best = w, e
    return best


def _threat_unit_is_pressing(e) -> bool:
    """威胁单位已进入我方半场/桥头（y ≤ 河中心），正在输出或即将过桥。"""
    return float(e.position.y) <= BRIDGE_Y + 1.0


class BeliefPlanner:
    """基于信念状态的规划器。"""

    def __init__(self, use_posterior_sampling: bool = False, n_samples: int = 8):
        self.use_posterior_sampling = use_posterior_sampling
        self.n_samples = n_samples

    # ---- Phase 2 v1 新意图检测（每帧最多命中一个，返回 None 表示回退旧逻辑）----

    def _soft_control(self, battle, p, threat):
        """最紧急：威胁单位正在我方半场输出 → 手牌有软控（冰冻/藤蔓/龙卷）。"""
        threat_unit = _closest_threat(battle)
        if threat_unit is None or not _threat_unit_is_pressing(threat_unit):
            return None
        for card in SOFT_CONTROL_CARDS:
            slot = _hand_slot(p, card)
            if slot is not None and p.elixir >= Card(card).elixir:
                return PlanToken(
                    macro_intent="soft_control",
                    focus_region=_own_region(threat_unit.position.x),
                    suggested_card=slot, target_kind="unit",
                    placement_hint="none", elixir_budget=0.4, risk_profile=0.6,
                    value_estimate=-2.0)
        return None

    def _spell_trade(self, battle, p, threat):
        """解牌：敌方高价值单位已进入我方半场 → 手牌有伤害法术直接解（赚费差/保塔）。"""
        threat_unit = _closest_threat(battle)
        if threat_unit is None or float(threat_unit.position.y) > OWN_HALF_EDGE + 1.5:
            return None
        if threat_unit.name in PULL_TARGET_CARDS:
            return None  # 血牛解法术亏（Fireball 解不动 Golem）→ 交给 _pull/单位
        cost = float(Card(threat_unit.name).elixir)
        # 只解"值得交法术"的目标：≥3 费部署单位（1-2 费杂鱼留给塔/单位）
        if cost < 3.0:
            return None
        for card in TRADE_SPELL_CARDS:
            slot = _hand_slot(p, card)
            if slot is not None and p.elixir >= Card(card).elixir:
                return PlanToken(
                    macro_intent="spell_trade",
                    focus_region=_own_region(threat_unit.position.x),
                    suggested_card=slot, target_kind="unit",
                    placement_hint="none", elixir_budget=0.5, risk_profile=0.5,
                    value_estimate=float(min(cost, 6.0)) * 0.5)
        return None

    def _pull(self, battle, p, threat):
        """拉扯：敌方血牛已到桥头/过桥 → 放低费单位/建筑改变其路径（距离制胜）。"""
        target = None
        for e in battle.entities.values():
            if e.player != 1 or not _deployable_entity(e):
                continue
            if e.name not in PULL_TARGET_CARDS:
                continue
            if float(e.position.y) <= BRIDGE_Y + 2.0 and float(e.position.y) >= OWN_HALF_EDGE - 4.0:
                target = e
                break
        if target is None:
            return None
        # 拉饵：≤3 费可部署单位/建筑（血牛会锁定它并改道）
        for i, card in enumerate(p.cycle[:4]):
            c = Card(card)
            if c.type in ("character", "building") and 1.0 <= c.elixir <= 3.0 \
                    and p.elixir >= c.elixir and card != "Mirror":
                near_edge = target.position.x < LANE_SPLIT_X - 4.0 \
                    or target.position.x > LANE_SPLIT_X + 4.0
                return PlanToken(
                    macro_intent="pull",
                    focus_region="own_center",
                    suggested_card=i + 1, target_kind="unit",
                    placement_hint="pull_across" if near_edge else "pull_aggro",
                    elixir_budget=0.3, risk_profile=0.5,
                    value_estimate=-1.0)
        return None

    def _push_commit(self, battle, p, threat):
        """推进跟进：己方坦克在地图上推进中（y ≥ 己方半场中前段）→ 在能走到坦克
        后方的区域跟后排（不必紧贴正后方，不等过桥）。"""
        tank = None
        for e in battle.entities.values():
            if e.player != 0 or not _deployable_entity(e):
                continue
            if e.name not in TANK_CARDS:
                continue
            y = float(e.position.y)
            # 坦克已沉底并走过己方半场中段（约 y≥8）即视为推进中；过桥后依然有效
            if 8.0 <= y <= 22.0:
                tank = e
                break
        if tank is None:
            return None
        # 后排支持：非坦克角色 3-6 费（Musketeer/Wizard/Archer/Minions...）
        for i, card in enumerate(p.cycle[:4]):
            c = Card(card)
            if c.type == "character" and 3.0 <= c.elixir <= 6.0 \
                    and card not in TANK_CARDS and p.elixir >= c.elixir:
                return PlanToken(
                    macro_intent="push_commit",
                    focus_region=_own_region(tank.position.x),
                    suggested_card=i + 1, target_kind="unit",
                    placement_hint="support_zone", elixir_budget=0.6,
                    risk_profile=0.7, value_estimate=1.5)
        return None

    def _setup_wait(self, battle, p, threat):
        """蓄力：低压力 + 手牌有坦克且圣水够沉底 + 场上无己方坦克 → 沉底开局/憋组合。"""
        if threat >= PRESSURE_THRESHOLD:
            return None
        tu = _closest_threat(battle)
        if tu is not None and _threat_unit_is_pressing(tu):
            return None  # 敌方单位已压境（哪怕单单位数值低）→ 防守优先
        for e in battle.entities.values():
            if e.player == 0 and _deployable_entity(e) and e.name in TANK_CARDS:
                return None  # 已有坦克在场上 → 轮不到 setup
        for card in TANK_CARDS:
            slot = _hand_slot(p, card)
            if slot is not None and p.elixir >= Card(card).elixir - 0.5:
                return PlanToken(
                    macro_intent="setup_wait",
                    focus_region="own_center", suggested_card=slot,
                    target_kind="none", placement_hint="none",
                    elixir_budget=0.6, risk_profile=0.4, value_estimate=0.5)
        return None

    def _cycle_small(self, battle, p, threat):
        """过牌：无压力 + 手牌含 1-2 费小牌 + 圣水充足 → 下小费轮转手牌质量。"""
        if threat >= PRESSURE_THRESHOLD:
            return None
        tu = _closest_threat(battle)
        if tu is not None and _threat_unit_is_pressing(tu):
            return None  # 同上：压境时不过牌
        for i, card in enumerate(p.cycle[:4]):
            c = Card(card)
            if c.elixir <= 2.0 and p.elixir >= c.elixir + 3.0 and card != "Mirror":
                return PlanToken(
                    macro_intent="cycle_small",
                    focus_region="own_center", suggested_card=i + 1,
                    target_kind="none", placement_hint="none",
                    elixir_budget=0.25, risk_profile=0.3, value_estimate=0.2)
        return None

    # ---- 主入口 ----

    def plan(self, battle, belief: BeliefState, obs=None) -> PlanToken:
        threat, my_pressure = _enemy_pressure(battle)
        p = battle.players[0]

        # —— Phase 2 v1 优先链（紧急度降序）：软控→解牌→拉扯→推进跟牌→蓄力→过牌 ——
        for detector in (self._soft_control, self._spell_trade, self._pull,
                         self._push_commit, self._setup_wait, self._cycle_small):
            tok = detector(battle, p, threat)
            if tok is not None:
                return tok

        # —— 回退：旧 8 意图逻辑 ——
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
