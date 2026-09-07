"""交换模拟器（外置工具 ②）：给定盘面 + 我方候选出牌（卡名, 落点），推演到视界，
返回这波交换的双方塔损 / 部队伤亡 / 圣水花销。

实现：deepcopy 整个 BattleState → 真的 deploy 候选牌（走引擎全部合法性校验）→
确定性推演。推演期我方按"不再投入"处理（与 threat_calc 同语义，只评估这一张牌）；
对手防守方按 defender 参数决定：
- "none"   : 对手不响应（= threat_calc 的纯盘面口径，作为对照下界）；
- "script" : 内置确定性脚本防守（见 script_defender），基线对手；
- "fn"     : 调用方注入 opponent_fn(sim, defender_id) -> [(card, pos), ...]，
             用于多源对手采样（历史 checkpoint / 信念采样防守），AlphaStar
             对手模型偏差的对策入口。

与 threat_calc 的关系：threat_calc 回答"什么都不做会损失多少"（无条件威胁）；
simulate_exchange 回答"打出这张牌之后交换结果如何"（反事实）。两者输出塔损口径
一致（同一 _TOWER_IDS 约定），可直接做差得到"这张牌挽回了多少塔损"。

用法::

    from simulate_exchange import simulate_exchange
    res = simulate_exchange(battle, player_id=0, card="MiniPekka", pos=Position(9.0, 14.0),
                            horizon=10.0, defender="script")
    # res["legal"] / res["my_towers"]["total"] / res["opp_towers"]["total"]
    # res["my_cost"] / res["opp_cost"] / res["my_units"]["hp_frac"] / res["opp_units"]

消费方（规划内）：卡牌知识模块的对账老师 / 条件威胁评估头的数据集生成器 /
浅 MCTS 叶估值。确定性、无副作用（selftest: test_simulate_exchange）。
"""

import copy

from card_utils import Card
from core import Position

__all__ = ["simulate_exchange", "script_defender", "EXCHANGE_HORIZON_S",
           "DEFENDER_TICK_S"]

#: 缺省推演视界：一次防守/进攻交锋的典型结算窗（Giant 类结算需 15s+，用 threat_calc
#: 看长程威胁；本工具聚焦"这一张牌的交换"，10s 覆盖大多数解牌/换塔结果）。
EXCHANGE_HORIZON_S = 10.0

#: 脚本防守方的决策周期（秒）：0.5s 检查一次是否需要部署反制。
DEFENDER_TICK_S = 0.5

#: 塔实体 id 约定（与 threat_calc 一致）：1/2/5 = P1(红), 3/4/6 = P0(蓝)
_TOWER_IDS = {0: (3, 4, 6), 1: (1, 2, 5)}
_TOWER_NAMES = {3: "left", 4: "right", 6: "king", 1: "left", 2: "right", 5: "king"}
_PRINCESS = {0: (3, 4), 1: (1, 2)}


def _tower_report(sim, player_id):
    """player_id 各塔的累计掉血（相对开打前快照由调用方记录）。"""
    out = {}
    for tid in _TOWER_IDS[player_id]:
        e = sim.entities[tid]
        out[_TOWER_NAMES[tid]] = float(e.hp)
    return out


def _threats_to(sim, defender_id, dist_to_tower=10.0):
    """对 defender_id 构成威胁的敌方部队：已过河进入我方半场，或逼近我方存活公主塔。"""
    attacker = 1 - defender_id
    towers = [sim.entities[tid] for tid in _PRINCESS[defender_id]
              if sim.entities[tid].is_alive]
    out = []
    for e in sim.entities.values():
        if not getattr(e, "is_alive", False) or getattr(e, "player", None) != attacker:
            continue
        if e.id in _TOWER_IDS[0] + _TOWER_IDS[1]:
            continue
        if getattr(e.data, "speed", 0) <= 0:
            continue   # 建筑/法术实体不构成推进威胁
        y = e.position.y
        crossed = (y <= 17.5) if defender_id == 1 else (y >= 14.5)
        if crossed or any(e.position.distance_to(t.position) < dist_to_tower
                          for t in towers):
            out.append(e)
    return out


def script_defender(sim, defender_id):
    """确定性基线防守（纯函数，无副作用）：威胁出现时，从手牌选性价比最高的可出
    反制部队，给出"塔前迎击"候选落点（按优先级排序）。返回 [(card, pos), ...]，
    调用方依次尝试 deploy，首个成功者生效（每 tick 至多一张）。

    反制选择口径（全部来自 gamedata，不拍脑袋）：
    - 跳过法术与仅攻建筑单位（都不解场）；
    - 威胁全为空军时要求 attack_air；
    - 评分 = (DPS + HP/15) / 费用，取最高（平分按卡名字典序打破）。
    """
    threats = _threats_to(sim, defender_id)
    if not threats:
        return []
    p = sim.players[defender_id]
    best_name, best_score = None, None
    all_air = all(t.data.is_air_unit for t in threats)
    for c in dict.fromkeys(p.cycle[:4]):
        if not p.can_play_card(c):
            continue
        info = Card(c)
        if info.type == 'spell' or info.target_only_buildings:
            continue
        if all_air and not info.attack_air:
            continue
        dps = (info.damage or 0) / max(info.hit_speed, 0.1)
        score = (dps + info.hp / 15.0) / max(info.elixir, 1)
        if best_score is None or score > best_score + 1e-9 or \
           (abs(score - best_score) <= 1e-9 and c < best_name):
            best_name, best_score = c, score
    if best_name is None:
        return []

    cx = sum(t.position.x for t in threats) / len(threats)
    cy = sum(t.position.y for t in threats) / len(threats)
    towers = [sim.entities[tid] for tid in _PRINCESS[defender_id]
              if sim.entities[tid].is_alive]
    tower = min(towers, key=lambda t: (t.position.x - cx) ** 2 + (t.position.y - cy) ** 2)
    dx, dy = cx - tower.position.x, cy - tower.position.y
    norm = (dx * dx + dy * dy) ** 0.5 or 1.0
    # 候选落点 = 塔前迎击线（朝威胁 2/3.5/1 格）+ 塔侧翼；调用方逐个尝试
    cands = []
    for d in (2.0, 3.5, 1.0):
        cands.append((tower.position.x + dx / norm * d,
                      tower.position.y + dy / norm * d))
    cands.append((tower.position.x + 2.0, tower.position.y))
    cands.append((tower.position.x - 2.0, tower.position.y))
    out = []
    for x, y in cands:
        x = min(max(x, 0.5), 17.5)
        y = min(max(y, 0.5), 31.5)
        out.append((best_name, (x, y)))
    return out


_ILLEGAL = {"legal": False, "reason": "deploy_failed", "card": None, "pos": None,
            "my_cost": 0.0, "opp_cost": 0.0,
            "my_towers": {}, "opp_towers": {}, "towers_lost": {"mine": [], "opp": []},
            "my_units": {}, "opp_units": {}, "sim_time": 0.0}


def simulate_exchange(battle, player_id: int, card: str, pos, horizon: float = EXCHANGE_HORIZON_S,
                      defender: str = "script", opponent_fn=None, dt: float = 1 / 60,
                      defender_tick: float = DEFENDER_TICK_S):
    """推演"我方现在打出 card@pos"到 horizon 秒的交换结果。原 battle 不被修改。

    defender: "none" / "script" / "fn"（用 opponent_fn(sim, defender_id) 自定义对手）。
    返回 dict：
      legal            部署是否成功（False 时其余字段为空壳）
      my_cost/opp_cost 双方本次推演实际圣水花销（含引擎动态费用如 Mirror/MergeMaiden）
      my_towers/opp_towers  {"left","right","king","total"} 掉血
      towers_lost      {"mine": [...], "opp": [...]} 被打掉的塔
      my_units         {"deployed","alive","hp_frac"} 我方这张牌产出的实体存活情况
      opp_units        {"count","killed","hp_removed"} 对手部队（现存+推演期新部署）
      sim_time         实际推演秒数
    """
    assert player_id in (0, 1), f"player_id 必须是 0/1，收到 {player_id}"
    assert defender in ("none", "script", "fn"), f"未知 defender 模式: {defender}"
    if defender == "fn":
        assert opponent_fn is not None, "defender='fn' 需要 opponent_fn"
    out = dict(_ILLEGAL)
    out["towers_lost"] = {"mine": [], "opp": []}
    opp_id = 1 - player_id

    sim = copy.deepcopy(battle)
    my_tids, opp_tids = _TOWER_IDS[player_id], _TOWER_IDS[opp_id]
    my_hp0 = {tid: float(sim.entities[tid].hp) for tid in my_tids}
    opp_hp0 = {tid: float(sim.entities[tid].hp) for tid in opp_tids}

    pre_ids = set(sim.entities)
    my_elixir0 = float(sim.players[player_id].elixir)
    if not sim.deploy_card(player_id, card, pos):
        return out
    out["legal"] = True
    out["reason"] = None
    out["card"] = card
    out["pos"] = (float(pos.x), float(pos.y)) if hasattr(pos, "x") else (float(pos[0]), float(pos[1]))
    out["my_cost"] = round(my_elixir0 - float(sim.players[player_id].elixir), 3)

    # 对手部队：现存 + 推演期新部署都记 hp0；我方这张牌产出的实体同理
    # （死亡单位会从实体表移除，因此记 hp0 留底）
    import battle as _battle_mod

    def _is_unit(e):
        # 只统计部队/建筑；塔攻击的 Projectile、法术 AreaEffect 等效果实体不算
        return isinstance(e, (_battle_mod.Troop, _battle_mod.Building))

    opp_units_hp0 = {}
    for eid, e in sim.entities.items():
        if eid in pre_ids and getattr(e, "player", None) == opp_id \
                and eid not in opp_tids and _is_unit(e) \
                and getattr(e, "is_alive", False):
            opp_units_hp0[eid] = float(e.hp)
    my_spawn_hp0 = {}

    def _absorb_new_entities():
        for eid, e in sim.entities.items():
            if eid in opp_units_hp0 or eid in my_spawn_hp0:
                continue
            pl = getattr(e, "player", None)
            if not _is_unit(e) or not getattr(e, "is_alive", False):
                continue
            if pl == opp_id and eid not in opp_tids:
                opp_units_hp0[eid] = float(e.hp)
            elif pl == player_id and eid not in my_tids:
                my_spawn_hp0[eid] = float(max(getattr(e.data, "hp", 0) or 0, 1.0))

    _absorb_new_entities()

    opp_cost = 0.0
    t, next_tick = 0.0, defender_tick
    while t < horizon - 1e-9:
        if sim.game_over:
            break
        sim.step(dt)
        t += dt
        _absorb_new_entities()
        if t >= next_tick - 1e-9 and not sim.game_over:
            next_tick += defender_tick
            if defender == "script":
                acts = script_defender(sim, opp_id)
            elif defender == "fn":
                acts = opponent_fn(sim, opp_id) or []
            else:
                acts = []
            for c, ppos in acts:
                if not hasattr(ppos, "x"):
                    ppos = Position(ppos[0], ppos[1])
                e0 = float(sim.players[opp_id].elixir)
                if sim.deploy_card(opp_id, c, ppos):
                    opp_cost += e0 - float(sim.players[opp_id].elixir)
                    _absorb_new_entities()
                    break   # 每 tick 至多一张；候选按优先级排序，失败试下一个
        if all(not sim.entities[tid].is_alive for tid in my_tids + opp_tids):
            break   # 已无塔可结算
    _absorb_new_entities()
    out["opp_cost"] = round(opp_cost, 3)
    out["sim_time"] = round(t, 3)

    def _report(hp0, tids):
        r, total, lost = {}, 0.0, []
        for tid in tids:
            e = sim.entities[tid]
            dmg = max(0.0, hp0[tid] - float(e.hp))
            r[_TOWER_NAMES[tid]] = round(dmg, 1)
            total += dmg
            if not e.is_alive:
                lost.append(_TOWER_NAMES[tid])
        r["total"] = round(total, 1)
        r["lost"] = lost
        return r

    mt = _report(my_hp0, my_tids)
    ot = _report(opp_hp0, opp_tids)
    out["my_towers"] = {k: v for k, v in mt.items() if k != "lost"}
    out["opp_towers"] = {k: v for k, v in ot.items() if k != "lost"}
    out["towers_lost"] = {"mine": mt["lost"], "opp": ot["lost"]}

    deployed = alive = 0
    hp_now = hp_max = 0.0
    for eid, h0 in my_spawn_hp0.items():
        deployed += 1
        hp_max += h0
        e = sim.entities.get(eid)
        if e is not None and getattr(e, "is_alive", False):
            alive += 1
            hp_now += float(e.hp)
    out["my_units"] = {"deployed": deployed, "alive": alive,
                       "hp_frac": round(hp_now / hp_max, 3) if hp_max else 0.0}

    killed = 0
    removed = 0.0
    count = len(opp_units_hp0)
    for eid, h0 in opp_units_hp0.items():
        e = sim.entities.get(eid)
        if e is None:
            killed += 1   # 尸体已清出实体表 = 击杀
            removed += h0
            continue
        if not getattr(e, "is_alive", False):
            killed += 1
        removed += max(0.0, h0 - float(e.hp))
    out["opp_units"] = {"count": count, "killed": killed,
                        "hp_removed": round(removed, 1)}
    return out


if __name__ == "__main__":
    import battle as _b
    import player as _p
    from core import Position

    deck = ["Knight", "MiniPekka", "Arrows", "Minions", "Musketeer", "Fireball", "Giant", "Archer"]
    bs = _b.BattleState(_p.PlayerState(0, list(deck), 8.0),
                        _p.PlayerState(1, list(deck), 8.0), card_level=11)
    bs.players[0].cycle = ["Giant"] + [c for c in bs.players[0].cycle if c != "Giant"]
    for mode in ("none", "script"):
        res = simulate_exchange(bs, 0, "Giant", Position(3.5, 14.0), horizon=12.0,
                                defender=mode)
        print(mode, "→", {k: res[k] for k in
                          ("legal", "my_cost", "opp_cost", "opp_towers", "my_units",
                           "opp_units", "sim_time")})
