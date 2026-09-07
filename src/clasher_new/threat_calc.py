"""塔伤威胁计算器（外置工具 ①）：当前盘面下，若双方都不再部署，敌方现存
部队能对我方防御塔造成多少伤害。

实现：deepcopy 整个 BattleState 后用引擎自身确定性推演（2026-09-06 实测
deepcopy ≈1.5ms、推演 ≈2.4ms/秒模拟时间），索敌/攻速/位移/塔兵反击/国王塔
激活/飞行物收尾全部由引擎结算——与真实对局口径零偏差，不重复建模。

语义（"不管" = 不新增任何部署，不冻结已有单位）：
- 我方现存部队照常防守 → 威胁 = "我不再投入资源" 时将承受的塔损；
- 对手同样按不部署处理（纯盘面威胁，不含对手后续圣水能换出的新攻势）；
- 双方圣水照常回复但无人可花（不影响塔损结算）。

用法::

    from threat_calc import estimate_tower_threat
    threat = estimate_tower_threat(battle, player_id=0, horizon=20.0)
    # threat = {"left": float, "right": float, "king": float, "total": float,
    #           "towers_lost": ["left", ...], "sim_time": float}

消费方（规划内）：belief_planner / prophet 可将其作为特征或候选动作估值器；
RL 侧可作为观测附加通道（每决策帧一次 ≈ 几十 ms，训练循环需掂量）。
"""

import copy

__all__ = ["estimate_tower_threat", "THREAT_HORIZON_S"]

#: 缺省推演视界：Giant 类坦克从桥头到拆完公主塔约 15-20s，20s 覆盖绝大多数
#: "现存部队的剩余威胁"；更长的威胁本来就该由后续部署应对，不属于本工具语义。
THREAT_HORIZON_S = 20.0

#: 塔实体 id 约定（BattleState.__init__ 固定首批生成）：1/2/5 = P1(红), 3/4/6 = P0(蓝)
_TOWER_IDS = {0: (3, 4, 6), 1: (1, 2, 5)}
_TOWER_NAMES = {3: "left", 4: "right", 6: "king", 1: "left", 2: "right", 5: "king"}


def _hostiles_present(battle, player_id):
    """是否存在仍可能威胁我方塔的敌方实体（非双方塔的存活实体）。"""
    tower_ids = set(_TOWER_IDS[0]) | set(_TOWER_IDS[1])
    for e in battle.entities.values():
        if not getattr(e, "is_alive", False):
            continue
        if e.id in tower_ids:
            continue
        # 敌方单位/建筑/飞行物都算；己方残兵不算（不构成对我方塔的威胁输入）
        if getattr(e, "player", None) == 1 - player_id:
            return True
    return False


def estimate_tower_threat(battle, player_id: int, horizon: float = THREAT_HORIZON_S,
                          dt: float = 1 / 60):
    """返回玩家 player_id 在"双方不再部署"假设下未来 horizon 秒内的塔损预估。

    返回 dict（键见模块 docstring）；towers_lost = 推演期内被打掉的塔名列表；
    sim_time = 实际推演秒数（早停时 < horizon）。原 battle 不被修改。
    """
    assert player_id in (0, 1), f"player_id 必须是 0/1，收到 {player_id}"
    zero = {"left": 0.0, "right": 0.0, "king": 0.0, "total": 0.0,
            "towers_lost": [], "sim_time": 0.0}
    if battle.game_over or not _hostiles_present(battle, player_id):
        return zero

    sim = copy.deepcopy(battle)
    tids = _TOWER_IDS[player_id]
    hp0 = {tid: float(sim.entities[tid].hp) for tid in tids}

    t = 0.0
    while t < horizon - 1e-9:
        if sim.game_over:
            break
        sim.step(dt)
        t += dt
        if not _hostiles_present(sim, player_id):
            break   # 敌方威胁源已清空，剩余推演不会产生新塔损
        if all(not sim.entities[tid].is_alive for tid in tids):
            break   # 我方塔已全破

    out = {"towers_lost": [], "sim_time": round(t, 3)}
    total = 0.0
    for tid in tids:
        e = sim.entities[tid]
        dmg = max(0.0, hp0[tid] - float(e.hp))
        name = _TOWER_NAMES[tid]
        out[name] = round(dmg, 1)
        if not e.is_alive:
            out["towers_lost"].append(name)
        total += dmg
    out["total"] = round(total, 1)
    return out


if __name__ == "__main__":
    # 冒烟演示：Giant 单独过桥 vs 空场
    import battle as _b
    import player as _p
    from core import Position

    deck = ["Knight", "MiniPekka", "Arrows", "Minions", "Musketeer", "Fireball", "Giant", "Archer"]
    bs = _b.BattleState(_p.PlayerState(0, list(deck), 5.0),
                        _p.PlayerState(1, list(deck), 5.0), card_level=11)
    print("空场:", estimate_tower_threat(bs, 0))
    bs.players[1].cycle = ["Giant"] + [c for c in bs.players[1].cycle if c != "Giant"]
    bs.players[1].elixir = 10.0
    bs.deploy_card(1, "Giant", Position(14.5, 19.0))   # P1 的 Giant 过桥攻 P0 左塔
    print("敌方 Giant 过桥:", estimate_tower_threat(bs, 0))
