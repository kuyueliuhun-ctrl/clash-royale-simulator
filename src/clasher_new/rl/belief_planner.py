"""信念规划器 BeliefPlanner（规划文档 5.5 / 3.7）。

只用可见观测 + b_t 出可部署计划；规则版 + 后验采样版。

Phase 2 v1 扩展（docs/rl_plan_design_v1.md）：
- **bp 组 12 个新意图**（70% 帧即可示范）：soft_control → spell_trade → protect_backline →
  pull → punish → push_commit → spell_finish → setup_wait → king_activate → anti_spell →
  save_ace → cycle_small（与 ProphetPlanner **同链同序** → 30/70 帧标签一致；
  7g 起部分谓词 BP 侧重启发式口径，与 PP 特权直读不同，但链序不变）；
- 圣水/手牌按"记忆即明牌"处理：punish 读 belief.elixir_mean，anti_spell/save_ace 读
  belief.hand_probs（粒子后验/后期确定性；信息不足时用概率阈值保守化），
  protect_backline 预判版读 belief.hand_probs 里切后排突进卡；
- 一帧只输出一个 macro_intent：新意图按紧急度优先，未命中回退旧 8 意图逻辑；
- 位置类意图只给**策略位 hint**（placement_hint）不给坐标——模型从 grid 自学精确格。

修复：
- P1-4：实体压力统计过滤静态塔，空场开局不再恒为 defend_*；
- P1-15：focus_region 由 intent 推导（defend_left → own_left、push_right → enemy_right）；
- 7g（BP 侧；PP 暂未同步）：血牛按“高血量”口径（lv11 hp≥1600）理解，不再死名单——
  spell_trade 只解远程脆皮；只打建筑的攻城单位/高血血牛/一切近战都交给拉扯；
  只打建筑的只能由建筑拉扯，其余可用便宜单位拦在行进路线（身板优先于输出：
  修正“骑士放弓箭手后面”→ 近战身板前置吸仇恨）；setup_wait 改为主动攒费
  （费≥坦克费+储备）才沉底血牛，push_commit 承认 Knight/Valkyrie/Prince 等
  高血近战为前排并跟输出。
"""

import os
import sys

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from typing import Optional

import numpy as np

from card_utils import Card, card_data
from rl.belief import BeliefState, BeliefInference
from rl.plan_space import (PlanToken, MACRO_INTENTS, FOCUS_REGIONS, ACE_CARDS,
                           OPP_SPELL_THREATS)

#: 推进压力阈值（与单位数量/HP×费用匹配）
PRESSURE_THRESHOLD = 2.0

#: 蓝方视角坐标（player0 塔 y≈3-6.5；河 y≈15-17；红方塔 y≈25.5）
BRIDGE_Y = 16.0        # 河中心（约 16）
OWN_HALF_EDGE = 15.0   # 己方半场边界（y < 15 为纯己方半场）
LANE_SPLIT_X = 9.0     # 左/右路分界
LATE_S = 120.0         # 双倍圣水（后期磨塔/塔血贵）起点

#: 软控法术（对威胁单位打断/拖延）
SOFT_CONTROL_CARDS = ("Freeze", "Vines", "Tornado")
#: 伤害/解牌法术（spell_trade 用）
TRADE_SPELL_CARDS = ("Fireball", "Arrows", "Rocket", "Lightning", "Poison",
                     "Earthquake", "Void", "TheLog", "Snowball", "Zap", "Tornado")
#: 后期磨塔法术（spell_finish：能稳定打到塔）
FINISH_SPELL_CARDS = ("Fireball", "Rocket", "Lightning", "Poison", "Earthquake", "Arrows")
#: 推进坦克（血牛 / 推进主角；角色启发式用：punish/save_ace/king_activate 等）
TANK_CARDS = ("Giant", "Golem", "ElixirGolem", "RoyalGiant", "GoblinGiant",
              "ElectroGiant", "MegaKnight", "Pekka", "GiantSkeleton", "HogRider")
#: 血牛（只锁塔的大体积推进：pull 的目标；注：7g 后 BP 的拉扯判定不再依赖此名单，
#: 改按高血量/近战/建筑目标口径；此名单保留给 ProphetPlanner 与旧角色启发式共用）
PULL_TARGET_CARDS = ("Giant", "Golem", "ElixirGolem", "RoyalGiant", "GoblinGiant",
                     "ElectroGiant", "MegaKnight", "Pekka", "GiantSkeleton")

#: —— 7g 血牛/拉扯新口径 ——
#: 高血量判定（lv11 基准，同一等级可比）：hp≥1600 视为“血牛”。
#: 实测 lv11：Knight 1766 / Valkyrie 1908 / Prince 1920 / HogRider 1696 均≥1600；
#: MiniPekka 1390 / DarkPrince 1200 / Archer 304 等输出/脆皮不算。
TANKY_HP = 1600.0
#: 近战判定：攻击距离 ≤2.0（近战无论血多或少都可以用拉扯/拦路解决）
MELEE_RANGE = 2.0
#: 拉扯/拦路所用便宜单位费用上界
PULL_CHEAP_COST = 3.0
#: 沉底出手需要圣水 = 坦克费用 + 输出储备（沉底后能留费跟输出/防守）
SETUP_RESERVE = 2.0
#: 沉底名单（推进型血牛；HogRider 属桥头快攻，不沉底）
SINK_TANK_CARDS = ("Giant", "Golem", "ElixirGolem", "RoyalGiant", "GoblinGiant",
                   "ElectroGiant", "MegaKnight", "Pekka", "GiantSkeleton")
#: 前排主体（push_commit 的“坦克”判定：推进型血牛 + 高血近战身板如 Knight/Valkyrie/Prince）
FRONT_TANK_CARDS = SINK_TANK_CARDS + ("HogRider", "Knight", "Valkyrie", "Prince")
#: 后排（远程/高价值支援单位：protect_backline 保护对象；pp 预判版共用）
BACKLINE_CARDS = ("Musketeer", "Archer", "Wizard", "IceWizard", "ElectroWizard",
                  "MagicArcher", "Princess", "DartGoblin", "Bomber", "Firecracker",
                  "Executioner", "ThreeMusketeers", "MotherWitch", "Zappies")
#: 切后排的威胁单位（近战突进；pp 预判版查对手手牌用）
BACKLINE_HARASSER_CARDS = ("MiniPekka", "Knight", "Valkyrie", "Bandit", "Prince",
                           "DarkPrince", "RoyalGhost", "Guards")
#: 国王塔激活：公主塔残血判定（lv11 3052 的 ~26%）
KING_ACTIVATE_PRINCESS_HP = 800.0

#: 卡名 → opp_spell_threat 枚举（anti_spell 用；大范围/斩杀法术保守归 big_unknown）
_SPELL_THREAT_KIND = {
    "Fireball": "fireball", "Poison": "poison", "Lightning": "lightning",
    "Freeze": "freeze", "Vines": "freeze", "Tornado": "freeze",
    "Rocket": "big_unknown", "Earthquake": "big_unknown", "Void": "big_unknown",
}
_HAND_PROB_THRESHOLD = 0.55   # anti_spell：手牌概率高于此视为"对面有这张法术"


def _is_tower(name: str) -> bool:
    return "Tower" in name


def _deployable_entity(e) -> bool:
    """可部署单位/建筑（排除塔、法术弹道/区域效果等临时实体）。"""
    name = getattr(e, "name", "")
    if not e.is_alive or _is_tower(name):
        return False
    if name not in card_data:
        # 箭矢（ArcherArrow）等弹道实体不在卡表 → 不是可部署威胁/目标
        return False
    return getattr(getattr(e, "data", None), "type", "") in ("character", "building")


def _unit_card(name):
    """取卡名对应 Card（lv11 统一口径）；非卡名实体/数据缺失 → None。"""
    try:
        return Card(name)
    except KeyError:
        return None


def _tanky_or_melee(c) -> bool:
    """拉扯对象口径（7g）：只打建筑的攻城单位 / 高血量血牛 / 近战，全部可拉扯。

    - 只打建筑的血牛 → 只能由建筑拉扯（单位拦不住攻城单位）；
    - 只是血多/近战 → 任意单位都能拦（近战身板放最前面）。
    """
    return c is not None and (c.target_only_buildings or c.hp >= TANKY_HP
                              or c.range <= MELEE_RANGE)


def _is_front_tank_name(name) -> bool:
    """前排主体判定（push_commit 跟输出用）：沉底名单/桥头快攻/高血近战身板。"""
    c = _unit_card(name)
    if c is None:
        return False
    return name in FRONT_TANK_CARDS or (c.type == "character" and c.hp >= TANKY_HP)


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


def _my_main_x(battle):
    xs = [e.position.x for e in battle.entities.values()
          if e.is_alive and e.player == 0 and not _is_tower(e.name)]
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


def _opposite_enemy_region(x: float) -> str:
    """敌方重心在 x → 建议进攻的另一路 region。"""
    return "enemy_right" if x < LANE_SPLIT_X else "enemy_left"


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


def _opp_spell_threat_of(belief: Optional[BeliefState]):
    """从信念手牌后验估计对手的强法术威胁 → OPP_SPELL_THREATS 值（None=无）。

    后期粒子收敛/确定性锁定时 hand_probs 近 0/1 → 判断精确；前期概率 > 阈值才报。
    """
    if belief is None or belief.hand_probs is None or len(belief.hand_probs) == 0:
        return None
    best, best_p = None, _HAND_PROB_THRESHOLD
    for card, p in zip(belief.deck, belief.hand_probs):
        kind = _SPELL_THREAT_KIND.get(card)
        if kind is not None and float(p) > best_p:
            best_p, best = float(p), kind
    return best


class BeliefPlanner:
    """基于信念状态的规划器。"""

    def __init__(self, use_posterior_sampling: bool = False, n_samples: int = 8):
        self.use_posterior_sampling = use_posterior_sampling
        self.n_samples = n_samples

    # ---- Phase 2 v1 新意图检测（每帧最多命中一个；belief 驱动型读 b_t）----

    def _soft_control(self, battle, p, threat, belief):
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

    def _spell_trade(self, battle, p, threat, belief):
        """解牌：敌方远程脆皮已进入我方半场 → 手牌有伤害法术直接解（赚费差/保塔）。

        7g：法术只留给远程脆皮（远程小单位法术收益最高）；只打建筑的攻城单位、
        高血量血牛、以及一切近战都放行给 _protect_backline/_pull——近战无论血多
        血少都能用拉扯/拦路解决（“骑士放弓箭手后面”的反例：近战身板要前置）。
        """
        threat_unit = _closest_threat(battle)
        if threat_unit is None or float(threat_unit.position.y) > OWN_HALF_EDGE + 1.5:
            return None
        cu = _unit_card(threat_unit.name)
        if cu is None:
            return None  # 防御：非卡名实体（弹道等）不做法术交易
        if cu.target_only_buildings or cu.hp >= TANKY_HP or cu.range <= MELEE_RANGE:
            return None  # 血牛/攻城/近战 → 交给拉扯（_pull）；解不动或不该用法术
        cost = float(cu.elixir)
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

    def _protect_backline(self, battle, p, threat, belief):
        """保后排（bp 版）：a) 反应——敌方近战已贴近/正接近我方后排 → 前置单位吸仇恨；
        b) 信念预判——belief 显示对手手牌高概率有切后排突进（>0.55）且后排暴露。"""
        backlines = [e for e in battle.entities.values()
                     if e.player == 0 and _deployable_entity(e)
                     and e.name in BACKLINE_CARDS
                     and float(e.position.y) <= OWN_HALF_EDGE + 0.5]
        target, predictive = None, False
        for bl in backlines:
            bx, by = float(bl.position.x), float(bl.position.y)
            for e in battle.entities.values():
                if e.player != 1 or not _deployable_entity(e):
                    continue
                cu = _unit_card(e.name)
                if cu is None or cu.target_only_buildings:
                    continue  # 只锁塔的攻城单位拦不住 → 交给 _pull（建筑拉扯）                if float(e.position.y) > BRIDGE_Y + 1.5:
                    continue
                dx = float(e.position.x) - bx
                dy = float(e.position.y) - by
                if dx * dx + dy * dy <= 36.0:   # 距离 ≤ 6
                    target = bl
                    break
            if target is not None:
                break
        if target is None and backlines and belief is not None \
                and belief.hand_probs is not None:
            tu = _closest_threat(battle)
            if tu is None or not _threat_unit_is_pressing(tu):
                probs = {c: float(q) for c, q in zip(belief.deck, belief.hand_probs)}
                if any(probs.get(c, 0.0) > _HAND_PROB_THRESHOLD
                       for c in BACKLINE_HARASSER_CARDS):
                    exposed = [e for e in backlines if float(e.position.y) >= 9.0]
                    if exposed:
                        target = exposed[0]
                        predictive = True
        if target is None:
            return None
        for i, card in enumerate(p.cycle[:4]):
            c = Card(card)
            if c.type in ("character", "building") and 1.0 <= c.elixir <= 4.0 \
                    and p.elixir >= c.elixir and card != "Mirror":
                return PlanToken(
                    macro_intent="protect_backline",
                    focus_region=_own_region(float(target.position.x)),
                    suggested_card=i + 1, target_kind="my_backline",
                    # 7g：吸仇恨单位放在后排/塔前方（pull_aggro），不是放在输出之后
                    placement_hint="pull_aggro", elixir_budget=0.4, risk_profile=0.5,
                    value_estimate=0.3 if predictive else 0.5)
        return None

    def _pull(self, battle, p, threat, belief):
        """拉扯/拦路（7g 口径，不再死等 PULL_TARGET_CARDS 名单）：
        - 目标：进入本方桥头带（y∈[11,18]）的敌方单位中——只打建筑的攻城单位、
          高血量血牛（hp≥TANKY_HP）、以及一切近战（近战血多血少都可拉扯）；
        - 打法：只打建筑的只能用建筑拉扯（单位拦不住攻城单位）；血牛/近战用便宜
          单位拦在其行进路线上——近战身板放最前面吸仇恨，输出留在身板后面。
        """
        target = None
        for e in battle.entities.values():
            if e.player != 1 or not _deployable_entity(e):
                continue
            y = float(e.position.y)
            if not (OWN_HALF_EDGE - 4.0 <= y <= BRIDGE_Y + 2.0):
                continue
            cu = _unit_card(e.name)
            if cu is None or not _tanky_or_melee(cu):
                continue
            target = (e, cu)
            break
        if target is None:
            return None
        e, cu = target
        # 卡选择：攻城目标必须建筑；普通目标优先近战身板（挡在输出前），
        # 其次任意便宜角色；空中目标要求我方单位能对空。
        pick = None
        for i, card in enumerate(p.cycle[:4]):
            if card == "Mirror":
                continue
            c = Card(card)
            if p.elixir < c.elixir or c.elixir > PULL_CHEAP_COST:
                continue
            if cu.target_only_buildings:
                if c.type == "building":
                    pick = i + 1
                    break
            elif c.type == "character":
                if cu.is_air_unit and not c.attack_air:
                    continue
                if c.range <= MELEE_RANGE:
                    pick = i + 1   # 近战身板最优
                    break
                if pick is None:
                    pick = i + 1   # 兜底：便宜角色也行（拦截射程内）
        if pick is None:
            return None
        near_edge = float(e.position.x) < LANE_SPLIT_X - 4.0 \
            or float(e.position.x) > LANE_SPLIT_X + 4.0
        if cu.target_only_buildings:
            # 攻城单位：拉到中路/塔前（建筑拉仇恨换路）
            focus, hint, val = "own_center", \
                ("pull_across" if near_edge else "pull_aggro"), -1.0
        else:
            # 血牛/近战：拦在它正走的那条自家半场路上（身板前置）
            focus, hint, val = _own_region(e.position.x), "pull_aggro", 0.4
        return PlanToken(
            macro_intent="pull",
            focus_region=focus, suggested_card=pick,
            target_kind="unit",
            placement_hint=hint, elixir_budget=0.3, risk_profile=0.5,
            value_estimate=val)

    def _punish(self, battle, p, threat, belief):
        """趁虚另一路：对手低圣水（记忆追踪，圣水=明牌）→ 压与敌方重心相反的路。"""
        if belief is None or belief.elixir_mean > 2.5:
            return None  # 对手圣水充足 → 不算"趁虚"
        if _closest_threat(battle) is not None and \
                _threat_unit_is_pressing(_closest_threat(battle)):
            return None  # 压境先防
        # 另一路：敌方单位重心反侧；无敌方单位时取我方压力反侧
        enemy_x = _enemy_main_x(battle)
        region = _opposite_enemy_region(enemy_x) if enemy_x != 9.0 or \
            any(e.player == 1 and _deployable_entity(e) for e in battle.entities.values()) \
            else ("enemy_left" if _my_main_x(battle) >= 9.0 else "enemy_right")
        # 推进主力：坦克或 ≥5 费角色
        for i, card in enumerate(p.cycle[:4]):
            c = Card(card)
            if (card in TANK_CARDS or (c.type == "character" and c.elixir >= 5.0)) \
                    and p.elixir >= c.elixir:
                return PlanToken(
                    macro_intent="punish", focus_region=region,
                    suggested_card=i + 1, target_kind="tower",
                    placement_hint="none", elixir_budget=0.7, risk_profile=0.8,
                    value_estimate=2.0)
        return None

    def _push_commit(self, battle, p, threat, belief):
        """推进跟进：己方前排主体（坦克/高血近战身板，7g 起含 Knight/Valkyrie/Prince）
        在地图上推进中（y ∈ 己方中前段 8-22）→ 在能走到前排后方的区域跟输出
        （先血牛/身板沉底或前置，输出跟在其后——修正“输出在前、身板在后”）。"""
        tank = None
        for e in battle.entities.values():
            if e.player != 0 or not _deployable_entity(e):
                continue
            if not _is_front_tank_name(e.name):
                continue
            y = float(e.position.y)
            if 8.0 <= y <= 22.0:
                tank = e
                break
        if tank is None:
            return None
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

    def _spell_finish(self, battle, p, threat, belief):
        """后期磨塔（t≥120 双倍期塔血贵）：敌方公主塔血量偏低 → 持续法术压血线。"""
        if battle.time < LATE_S:
            return None
        p1 = battle.players[1]
        # 选血量最低的存活公主塔（国王塔只在 2 塔全破后）
        candidates = []
        if p1.left_tower_hp > 0:
            candidates.append(("enemy_left", p1.left_tower_hp))
        if p1.right_tower_hp > 0:
            candidates.append(("enemy_right", p1.right_tower_hp))
        if not candidates:
            return None
        region, hp = min(candidates, key=lambda kv: kv[1])
        if hp > 1200.0:
            return None  # 血还多 → 交给 push 打，法术磨只对低血线有意义
        for card in FINISH_SPELL_CARDS:
            slot = _hand_slot(p, card)
            if slot is not None and p.elixir >= Card(card).elixir:
                return PlanToken(
                    macro_intent="spell_finish", focus_region=region,
                    suggested_card=slot, target_kind="tower",
                    placement_hint="none", elixir_budget=0.45, risk_profile=0.6,
                    value_estimate=1.2)
        return None

    def _setup_wait(self, battle, p, threat, belief):
        """主动攒费沉底（7g）：低压力 + 手牌有推进血牛（沉底名单）+ 圣水 ≥ 坦克费
        + 输出储备(SETUP_RESERVE) → 沉底塔后；血牛走过中前段后由 _push_commit 跟输出。
        费没攒够就返回 None（落到 cycle_and_wait 等费），不裸下血牛。
        注：HogRider 属桥头快攻不沉底；Knight 等高血近战由 push_commit 承认当前排。"""
        if threat >= PRESSURE_THRESHOLD:
            return None
        tu = _closest_threat(battle)
        if tu is not None and _threat_unit_is_pressing(tu):
            return None  # 敌方单位已压境（哪怕单单位数值低）→ 防守优先
        for e in battle.entities.values():
            if e.player == 0 and _deployable_entity(e) and e.name in SINK_TANK_CARDS:
                return None  # 已有沉底血牛在场上 → 轮不到 setup
        best_slot, best_cost = None, -1.0
        for i, card in enumerate(p.cycle[:4]):
            if card == "Mirror" or card not in SINK_TANK_CARDS:
                continue
            c = Card(card)
            if c.type != "character":
                continue
            if p.elixir < c.elixir + SETUP_RESERVE:
                continue  # 还在攒费：不急着裸沉
            if c.elixir > best_cost:
                best_slot, best_cost = i + 1, float(c.elixir)
        if best_slot is None:
            return None
        return PlanToken(
            macro_intent="setup_wait",
            focus_region="own_center", suggested_card=best_slot,
            target_kind="none", placement_hint="none",
            elixir_budget=0.6, risk_profile=0.4, value_estimate=0.5)

    def _king_activate(self, battle, p, threat, belief):
        """激活国王塔（bp 版）：公主塔残血/被破 + 敌方血牛/大单位接近国王塔中轴
        → 放低费单位拉仇恨位让国王塔参战。"""
        if min(p.left_tower_hp, p.right_tower_hp) > KING_ACTIVATE_PRINCESS_HP:
            return None
        heavy = None
        for e in battle.entities.values():
            if e.player != 1 or not _deployable_entity(e):
                continue
            c = Card(e.name)
            if e.name not in TANK_CARDS and not (c.type == "character"
                                                 and c.elixir >= 5.0):
                continue
            x, y = float(e.position.x), float(e.position.y)
            if 3.5 <= x <= 14.5 and y <= BRIDGE_Y + 2.5:
                heavy = e
                break
        if heavy is None:
            return None
        for i, card in enumerate(p.cycle[:4]):
            c = Card(card)
            if c.type in ("character", "building") and 1.0 <= c.elixir <= 3.0 \
                    and p.elixir >= c.elixir and card != "Mirror":
                return PlanToken(
                    macro_intent="king_activate",
                    focus_region="own_center", suggested_card=i + 1,
                    target_kind="unit", placement_hint="king_front",
                    elixir_budget=0.35, risk_profile=0.5, value_estimate=0.8)
        return None

    def _save_ace(self, battle, p, threat, belief):
        """藏终结卡：手牌有 ace（大闪/藤蔓/冰冻/火箭）且不是最强一波帧 →
        hold_mask 指名别出 + 留费（suggested 指向普通牌）。"""
        ace_slots = []
        for card in ACE_CARDS:
            slot = _hand_slot(p, card)
            if slot is not None:
                ace_slots.append(slot)
        if not ace_slots:
            return None
        tu = _closest_threat(battle)
        if tu is not None and _threat_unit_is_pressing(tu):
            return None  # 防守中 ace 可能当解牌用，不硬藏
        # 己方坦克推进中 = 最强一波窗口期 → 不藏（让 ace 出场）
        for e in battle.entities.values():
            if e.player == 0 and _deployable_entity(e) and e.name in TANK_CARDS:
                y = float(e.position.y)
                if y >= 10.0:
                    return None
        hold = 0
        for slot in ace_slots:
            hold |= 1 << (slot - 1)
        # 建议一张普通可部署卡（非 ace）
        suggested = None
        for i, card in enumerate(p.cycle[:4]):
            c = Card(card)
            if (i + 1) not in ace_slots and c.elixir <= p.elixir \
                    and c.type != "spell" and card != "Mirror":
                suggested = i + 1
                break
        return PlanToken(
            macro_intent="save_ace", focus_region="own_center",
            suggested_card=suggested, target_kind="none",
            placement_hint="none", elixir_budget=0.4, risk_profile=0.4,
            hold_mask=hold, value_estimate=-0.3)

    def _anti_spell(self, battle, p, threat, belief):
        """防法术：对手手牌高概率有强法术（belief）且我方要下后排 → 提示防溅射站位。"""
        kind = _opp_spell_threat_of(belief)
        if kind is None:
            return None
        if _closest_threat(battle) is not None and \
                _threat_unit_is_pressing(_closest_threat(battle)):
            return None
        for i, card in enumerate(p.cycle[:4]):
            c = Card(card)
            if c.type == "character" and 3.0 <= c.elixir <= 6.0 \
                    and p.elixir >= c.elixir and card not in TANK_CARDS:
                return PlanToken(
                    macro_intent="anti_spell", focus_region="own_center",
                    suggested_card=i + 1, target_kind="none",
                    placement_hint="anti_spell_zone", opp_spell_threat=kind,
                    elixir_budget=0.5, risk_profile=0.4, value_estimate=0.0)
        return None

    def _cycle_small(self, battle, p, threat, belief):
        """过牌：无压力 + 手牌含 1-2 费小牌 + 圣水充足 → 下小费轮转手牌质量。"""
        if threat >= PRESSURE_THRESHOLD:
            return None
        tu = _closest_threat(battle)
        if tu is not None and _threat_unit_is_pressing(tu):
            return None
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

        # —— Phase 2 v1 优先链（与 ProphetPlanner 同链同序：紧急度降序，30/70 帧标签一致）——
        for detector in (self._soft_control, self._spell_trade,
                         self._protect_backline, self._pull, self._punish,
                         self._push_commit, self._spell_finish, self._setup_wait,
                         self._king_activate, self._anti_spell, self._save_ace,
                         self._cycle_small):
            tok = detector(battle, p, threat, belief)
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
