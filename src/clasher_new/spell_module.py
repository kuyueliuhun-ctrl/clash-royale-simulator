"""法术知识模块（外置工具 ③，exposure level 1-2 参照实现）：把"某张法术能做什么、
在什么情况用、砸下去值多少"做成引擎侧确定性查询服务，不进动作空间。

设计原则（AGENTS.md 验证纪律）：模块里所有伤害数字**不做静态公式复刻**——引擎法术
数值路径含 OFFICIAL_OVERRIDES / damage_per_level / DOT buff_data / Arrows 25-122 特例，
静态复刻必漂移。本模块的数值全部来自**引擎实测标定**：对每张法术在一次性一次性标定局
（静止 Giant / 公主塔 / Cannon 三靶）里量出真实伤害，缓存复用；gamedata 只提供能力标志
（眩晕/冻结/拉拽/召唤/增益）与半径口径。对账 selftest（test_spell_module）保证
evaluate_cast 预测 ≈ engine_resolution 实测。

三层 API：
- get_spell_profile(card, level)   静态能力档案（标定缓存）
- evaluate_cast(battle, pid, card, pos)  落点估值：罩到谁、各掉多少血、谁能砸死、
  击杀折多少费（零推演成本，供规划器/观测特征批量调用）
- engine_resolution(battle, pid, card, pos)  引擎实测口径（对账老师 / 数据集标签）
- best_cast(battle, pid, card)     粗网格扫最优落点

用法::

    from spell_module import evaluate_cast, best_cast
    ev = evaluate_cast(battle, 0, "Fireball", Position(9.0, 19.0))
    # ev["targets"]=[{id,name,damage,dies,value}...] / ev["elixir_killed_value"]
    # ev["tower_damage_total"]

注意口径：evaluate_cast 是"施法瞬间"的静态预测——飞行时间内目标会移动（Fireball
弹道 ~1-2s），对快速单位预测偏乐观；精确交换请用 engine_resolution 或
simulate_exchange。DOT 类（Poison/Earthquake）的部队伤害按"目标满驻留"上界口径。
"""

import copy

from card_utils import Card

__all__ = ["get_spell_profile", "evaluate_cast", "best_cast", "engine_resolution",
           "clear_profile_cache"]

# ---------------------------------------------------------------------------
# 引擎标定
# ---------------------------------------------------------------------------

_PROFILE_CACHE = {}

#: 标定用假人（gamedata 常青卡，数值稳定）：
#: Giant = 慢速大血量部队靶（对部队伤害）；公主塔 = 对塔伤害（含降伤口径）；
#: Cannon = 普通建筑靶（含建筑加成如 Earthquake ×4.5）。
_CAL_TROOP_DUMMY = "Giant"
_CAL_BUILDING_DUMMY = "Cannon"
#: 部队靶候选位：P1 半场标准位；BarbLog 等部署区受限法术回退到 P0 半场
_CAL_TROOP_SPOTS = [(9.0, 19.0), (9.0, 12.0)]
_CAL_BUILDING_SPOTS = [(9.0, 19.0)]
_TOWER_IDS = {0: (3, 4, 6), 1: (1, 2, 5)}


def _deals_damage(card_name):
    """是否输出伤害的法术（口径同 rl/action_mask，另补 spells 行 damage/DOT）。"""
    from card_utils import spells
    info = Card(card_name)
    row = spells.get(card_name, {})
    pdmg = info.projectile_data.damage if info.projectile_data else 0
    return bool(pdmg or info.data.get('damage') or row.get('damage')
                or (row.get('buff_data') or {}).get('damage_per_second'))


def _radius_m(card_name):
    """溅射半径（世界单位）。引擎口径：AreaEffect 用 spells 行 radius，弹道法术用
    projectileData.radius；与 rl/action_mask._spell_radius_m 一致并补行级回退。"""
    from card_utils import spells
    row = spells.get(card_name, {})
    raw = row.get('radius')
    if not raw:
        data = getattr(Card(card_name), "data", None) or {}
        raw = data.get("radius")
        if raw is None:
            raw = (data.get("projectileData") or {}).get("radius")
    return (float(raw) if raw else 0.0) / 1000.0


def _spell_effects(card_name):
    """gamedata 能力标志（值伤之外的非伤害效果，供"什么情况用"层消费）。"""
    from card_utils import spells
    row = spells.get(card_name, {}) or {}
    bd = row.get('buff_data') or {}
    return {
        "stun": bool(row.get('stun') or bd.get('stun')),
        "freeze": row.get('buff') == 'Freeze' or bool(bd.get('freeze')),
        "pull": bool(row.get('controls_buff')) and card_name == 'Tornado',
        "buff": row.get('buff'),
        "spawn": row.get('spawn_character'),
        "clone": bool(row.get('clone')),
        "ignore_buildings": bool(row.get('ignore_buildings')),
        "deploy_restricted": card_name == 'BarbLog',   # 勘误批1：仅部队部署区可施放
    }


def engine_resolution(battle, player_id: int, card_name: str, pos,
                      max_t: float = 15.0, dt: float = 1 / 60):
    """引擎实测口径：deepcopy → deploy 该法术 → 推演到法术实体消亡（DOT 走满
    生命周期）→ 逐实体 hp 差。对账老师/数据集标签；原 battle 不被修改。"""
    assert player_id in (0, 1)
    from core import Position
    if not hasattr(pos, "x"):
        pos = Position(pos[0], pos[1])
    sim = copy.deepcopy(battle)
    opp_tids = _TOWER_IDS[1 - player_id]
    pre_ids = set(sim.entities)
    hp0 = {}
    for eid, e in sim.entities.items():
        pl = getattr(e, "player", None)
        if pl != player_id and getattr(e, "is_alive", False):
            hp0[eid] = float(e.hp)
    if not sim.deploy_card(player_id, card_name, pos):
        return {"legal": False, "per_entity": {}, "towers": {}, "sim_time": 0.0}
    cast_ids = set(sim.entities) - pre_ids
    t = 0.0
    while t < max_t - 1e-9:
        if sim.game_over:
            break
        sim.step(dt)
        t += dt
        # 延迟生成也归入本次施法实体（Arrows 三连波 / 墓园散兵等），否则会提前停表
        for eid in sim.entities:
            if eid not in pre_ids:
                cast_ids.add(eid)
        cast_gone = all(not (cid in sim.entities and sim.entities[cid].is_alive)
                        for cid in cast_ids)
        if t > 0.2 and cast_gone and not sim.schedule:
            break
    per, towers = {}, {}
    for eid, h0 in hp0.items():
        e = sim.entities.get(eid)
        dmg = h0 if e is None else max(0.0, h0 - float(e.hp))
        if dmg <= 0:
            continue
        if eid in opp_tids:
            towers[eid] = round(dmg, 1)
        else:
            per[eid] = round(dmg, 1)
    return {"legal": True, "per_entity": per, "towers": towers,
            "sim_time": round(t, 3)}


def _measure(card_name, level, dummy, spots, tower_target=False):
    """单场景标定：返回该场景下引擎实测的总伤害；部署/施法非法 → 0.0。"""
    import battle as battle_mod
    import player as player_mod
    from core import Position
    deck = [card_name] * 8
    opp_deck = [dummy] * 8 if dummy else ["Knight"] * 8
    for sx, sy in spots:
        bs = battle_mod.BattleState(player_mod.PlayerState(0, list(deck), 10.0),
                                    player_mod.PlayerState(1, list(opp_deck), 10.0),
                                    card_level=level)
        dummy_id = None
        if dummy:
            bs.players[1].cycle = [dummy] + [c for c in bs.players[1].cycle if c != dummy]
            if not bs.deploy_card(1, dummy, Position(sx, sy)):
                continue   # 该假人在此位部署非法 → 换下一位
        target = Position(bs.entities[1].position.x, bs.entities[1].position.y) \
            if tower_target else Position(sx, sy)
        res = engine_resolution(bs, 0, card_name, target)
        if not res["legal"]:
            continue   # 施法非法（如 BarbLog 越区）→ 换下一位
        if tower_target:
            return sum(res["towers"].values())
        if dummy:
            dummy_id = [eid for eid, e in bs.entities.items()
                        if eid > 6 and e.player == 1]
            return sum(v for eid, v in res["per_entity"].items() if eid in dummy_id)
    return 0.0


def _calibrate(card_name, level):
    from card_utils import spells
    row = spells.get(card_name, {}) or {}
    info = Card(card_name)
    lifetime = row.get('life_duration') or 0
    prof = {
        "name": card_name, "level": level,
        "elixir": info.elixir,
        "type": info.type,
        "deals_damage": _deals_damage(card_name),
        "radius": _radius_m(card_name),
        # 引擎标定伤害（对单个满血靶；DOT 为满驻留口径，见模块 docstring）
        "troop_damage": 0.0, "tower_damage": 0.0, "building_damage": 0.0,
        "dot": bool(lifetime >= 2500), "lifetime_s": lifetime / 1000.0,
        "effects": _spell_effects(card_name),
        "calibrated": False,
    }
    if not prof["deals_damage"]:
        return prof
    if row.get('spawn_character') or row.get('clone'):
        # 召唤/克隆类：伤害无"单点落点"语义，不做伤害标定
        return prof
    prof["troop_damage"] = _measure(card_name, level, _CAL_TROOP_DUMMY,
                                    _CAL_TROOP_SPOTS)
    prof["tower_damage"] = _measure(card_name, level, None, [(0, 0)],
                                    tower_target=True)
    prof["building_damage"] = _measure(card_name, level, _CAL_BUILDING_DUMMY,
                                       _CAL_BUILDING_SPOTS)
    prof["calibrated"] = True
    return prof


def get_spell_profile(card_name, level=None, refresh=False):
    """法术能力档案（含引擎标定伤害，按 (卡名, 等级) 惰性缓存）。"""
    level = level if level is not None else Card.default_level
    key = (card_name, level)
    if key not in _PROFILE_CACHE or refresh:
        _PROFILE_CACHE[key] = _calibrate(card_name, level)
    return _PROFILE_CACHE[key]


def clear_profile_cache():
    _PROFILE_CACHE.clear()


# ---------------------------------------------------------------------------
# 落点估值
# ---------------------------------------------------------------------------

def _hits_filters(info, e):
    """引擎溅射的空中/地面过滤口径（_deal_splash_damage）。仅对真有 projectileData
    的弹道法术应用（AreaEffect 类如 Zap 无该字段 → 空/地全命中；Projectile({})
    包装器的默认名 'Unknown' 不能作为判据）。"""
    if not (info.data.get('projectileData') or {}):
        return True
    proj = info.projectile_data
    if e.data.is_air_unit and not proj.hits_air:
        return False
    if (not e.data.is_air_unit) and not proj.hits_ground:
        return False
    return True


def evaluate_cast(battle, player_id: int, card_name: str, pos):
    """静态落点估值：罩到哪些敌方实体、各吃多少伤（标定口径）、谁被砸死、
    击杀折多少费。零推演成本；原 battle 不被修改。

    返回 dict：
      castable       该落点是否可施（BarbLog 部署区限制等）
      affordable     手牌/圣水口径是否可出（can_play_card）
      targets        [{"id","name","kind","hp","damage","dies","value"}]
      n_targets / elixir_killed_value / troop_damage_total / tower_damage_total
      dot            是否 DOT（部队伤害为满驻留上界）
    """
    from core import Position
    if not hasattr(pos, "x"):
        pos = Position(pos[0], pos[1])
    opp = 1 - player_id
    opp_tids = _TOWER_IDS[opp]
    prof = get_spell_profile(card_name, getattr(battle, "card_level", None))
    info = Card(card_name)
    out = {
        "card": card_name, "pos": (round(pos.x, 2), round(pos.y, 2)),
        "castable": True, "affordable": battle.players[player_id].can_play_card(card_name),
        "deals_damage": prof["deals_damage"], "dot": prof["dot"],
        "targets": [], "n_targets": 0, "elixir_killed_value": 0.0,
        "troop_damage_total": 0.0, "tower_damage_total": 0.0,
    }
    if card_name == 'BarbLog':
        out["castable"] = bool(battle.arena.can_deploy_at(
            pos, player_id, battle_state=battle, is_spell=False))
    if not prof["deals_damage"] or not out["castable"]:
        return out
    for e in battle.entities.values():
        if not getattr(e, "is_alive", False) or getattr(e, "invincible", False):
            continue
        if getattr(e, "player", None) != opp:
            continue
        if not _hits_filters(info, e):
            continue
        col = getattr(e.data, "collision_radius", 0.0) or 0.0
        if e.position.distance_to(pos) > prof["radius"] + col + 1e-9:
            continue
        if e.id in opp_tids:
            kind, dmg = "tower", prof["tower_damage"]
        elif type(e).__name__ == "Building":
            kind, dmg = "building", prof["building_damage"]
        else:
            kind, dmg = "troop", prof["troop_damage"]
        hp = float(e.hp) + float(getattr(e, "shield_health", 0) or 0)
        dies = dmg >= hp - 1e-9
        value = 0.0
        if dies:
            value = (getattr(e.data, "elixir", 0) or 0) / max(getattr(e.data, "spawn_number", 1), 1)
            out["elixir_killed_value"] += value
        tgt = {"id": e.id, "name": e.card_name, "kind": kind,
               "hp": round(hp, 1), "damage": round(dmg, 1), "dies": dies,
               "value": round(value, 2)}
        out["targets"].append(tgt)
        out["n_targets"] += 1
        if kind == "tower":
            out["tower_damage_total"] += dmg
        else:
            out["troop_damage_total"] += dmg
    return out


#: 落点评分缺省权重：击杀折费 + 塔/部队伤害折费。前期 1 费 ≈ 500 塔 HP（奖励口径）；
#: 部队血量 ≈ 650 HP/费（Knight 1766/3 ≈ 590、Giant 3968/5 ≈ 794 的折中）。消费者可传
#: 自定义评分覆盖。
TOWER_HP_PER_ELIXIR = 500.0
TROOP_HP_PER_ELIXIR = 650.0


def _score(ev):
    return (ev["elixir_killed_value"] + ev["tower_damage_total"] / TOWER_HP_PER_ELIXIR
            + ev["troop_damage_total"] / TROOP_HP_PER_ELIXIR)


def best_cast(battle, player_id: int, card_name: str, grid: float = 1.0):
    """在敌方实体的包围盒（外扩半径）上粗网格扫最优落点。
    返回 (pos, eval, score)；无伤害目标时返回 (None, None, -inf)。"""
    from core import Position
    opp = 1 - player_id
    prof = get_spell_profile(card_name, getattr(battle, "card_level", None))
    pts = [(e.position.x, e.position.y) for e in battle.entities.values()
           if getattr(e, "is_alive", False) and getattr(e, "player", None) == opp]
    if not pts:
        return None, None, float("-inf")
    r = prof["radius"] + grid
    x0 = max(0.5, min(p[0] for p in pts) - r)
    x1 = min(17.5, max(p[0] for p in pts) + r)
    y0 = max(0.5, min(p[1] for p in pts) - r)
    y1 = min(31.5, max(p[1] for p in pts) + r)
    best = (None, None, float("-inf"))
    ny = max(1, int((y1 - y0) / grid))
    nx = max(1, int((x1 - x0) / grid))
    for j in range(ny + 1):
        for i in range(nx + 1):
            pos = Position(x0 + i * grid, y0 + j * grid)
            ev = evaluate_cast(battle, player_id, card_name, pos)
            if not ev["castable"]:
                continue
            s = _score(ev)
            if s > best[2] + 1e-9:
                best = (pos, ev, s)
    return best


if __name__ == "__main__":
    import battle as _b
    import player as _p
    from core import Position

    deck = ["Knight", "MiniPekka", "Arrows", "Minions", "Musketeer", "Fireball", "Giant", "Archer"]
    bs = _b.BattleState(_p.PlayerState(0, list(deck), 8.0),
                        _p.PlayerState(1, list(deck), 8.0), card_level=11)
    bs.players[0].cycle = ["Fireball"] + [c for c in bs.players[0].cycle if c != "Fireball"]
    bs.players[1].cycle = ["Giant"] + [c for c in bs.players[1].cycle if c != "Giant"]
    bs.players[1].elixir = 10.0
    bs.deploy_card(1, "Giant", Position(9.0, 19.0))
    pos, ev, s = best_cast(bs, 0, "Fireball")
    print("best Fireball:", pos.x if pos else None, pos.y if pos else None,
          f"score={s:.2f}")
    print("  targets:", [(t["name"], t["damage"], t["dies"]) for t in ev["targets"]])
    res = engine_resolution(bs, 0, "Fireball", pos)
    print("engine towers:", res["towers"], "per_entity:", res["per_entity"])
