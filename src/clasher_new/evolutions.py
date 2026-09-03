"""M4 族7 觉醒系统数据层。
周期表来源：Fandom Card Evolution 主表 + 1/2-Cycles 分类页三源交叉验证（见 docs/evolution_cycles.json
与 docs/数值规则查证汇总.md，取数 2026-08-28）。周期语义：cycle=N = 先出 N 次普通形态，第 N+1 次为觉醒形态，之后交替。
键名映射勘误（经 data_official/cards_i18n.json 确认）：RageBarbarian=Lumberjack、AxeMan=Executioner。
Elite Barbarians（AngryBarbarians）不在 34 张快照觉醒内，其 2026 觉醒周期=1（RoyaleZone/gamer.org，v6 主册已录）。
"""
import math

EVOLUTION_CYCLES = {
    # —— 1 周期（打 1 回普通，第 2 次觉醒）——
    'AxeMan': 1, 'Barbarians': 1, 'ElectroDragon': 1, 'GoblinGiant': 1, 'MegaKnight': 1,
    'Pekka': 1, 'RoyalGiant': 1, 'RoyalRecruits': 1, 'Witch': 1, 'Wizard': 1,
    # —— 2 周期（打 2 回普通，第 3 次觉醒）——
    'Archer': 2, 'Bats': 2, 'BattleRam': 2, 'BlowdartGoblin': 2, 'Bomber': 2,
    'Cannon': 2, 'Firecracker': 2, 'GoblinBarrel': 2, 'GoblinCage': 2, 'GoblinDrill': 2,
    'Hunter': 2, 'IceSpirits': 2, 'InfernoDragon': 2, 'Knight': 2, 'Mortar': 2,
    'Musketeer': 2, 'RageBarbarian': 2, 'SkeletonBalloon': 2, 'Skeletons': 2,
    'Snowball': 2, 'Tesla': 2, 'Valkyrie': 2, 'Wallbreakers': 2, 'Zap': 2,
    # 2026 觉醒（快照外，已人工确认）：觉醒野蛮人精锐周期=1
    'AngryBarbarians': 1,
}

# 官方数值覆盖（数值按官方现行值；快照旧值处以研究结论为准，来源见 docs/数值规则查证汇总.md）
OFFICIAL_OVERRIDES = {
    'Rage': {'speed_mult': 1.30, 'duration': 4.5, 'residue': 1.0, 'damage_lv11': 179},  # 2025/10/6 +35%→+30%；2025/8/4 6s→4.5s
    'Log': {'crown_tower_percent': 0.15, 'pushback': 0.7},   # 伤害用快照 dpl（lv11=290）；Fandom 现行 266 冲突已记录 review_needed
    'Tornado': {'damage_per_tick_lv11': 84, 'tick': 0.55, 'crown_tower_percent': 0.35},
    'Heal': {'heal_per_tick_lv11': 96, 'tick': 0.5, 'duration': 2.0},  # 旧版 Heal 卡（2020 已删除，快照基准）
    'Fisherman': {'pull_speed': 8.5, 'slow_removed': True},  # 2026/4/6 减速已移除
    'GoldenKnight': {'ability_cooldown': 12.0},  # wiki 12s vs gamedata 8s 冲突，按官方
    'Monk': {'damage_reduction': 0.65},          # wiki 65% vs gamedata 80% 冲突，按官方
    'SkeletonKing': {'spawn_radius': 3.5},       # wiki 3.5 vs gamedata 4.0 冲突，按官方
}


def evolution_state(plays: int, card_name: str):
    """按出牌次数判断本手是否为觉醒形态：cycle=1 → 第2/4/6…次觉醒；cycle=2 → 第3/6…次觉醒。
    plays = 本次出牌前该卡已打出次数（0-based）。"""
    cycle = EVOLUTION_CYCLES.get(card_name)
    if not cycle:
        return False
    return plays % (cycle + 1) == cycle


def derive_evolved_stats(card_name, evolved_scd, base_card, characters, buildings, level=11):
    """从 evolvedSpellsData.summonCharacterData 推导觉醒形态指定等级属性（默认 lv11，支持 11-16）。
    觉醒数值快照为稀有度起始级（如 Knight_EV1 hp=690 为 lv1），按基础卡 per-level 曲线等比放大到目标等级。
    返回 dict：{hp, damage, shield_hitpoints, 特殊字段...}"""
    from card_utils import _rarity_level_index, level_scale
    out = {}
    base_scd = base_card.data['summonCharacterData']
    char_row = characters.get(base_scd.get('name')) or buildings.get(base_scd.get('name')) or {}
    hp_pl = char_row.get('hitpoints_per_level') or []
    li = _rarity_level_index(base_card.data.get('rarity') or 'Common', level)
    evo_hp = evolved_scd.get('hitpoints')
    if evo_hp and hp_pl and 0 <= li < len(hp_pl) and hp_pl[0]:
        out['hp'] = evo_hp * (hp_pl[li] / hp_pl[0])
    else:
        out['hp'] = base_card.hp
    out['damage'] = base_card.damage
    if evolved_scd.get('shieldHitpoints'):
        out['shield_hitpoints'] = evolved_scd['shieldHitpoints'] * level_scale(level)  # 护盾按统一曲线放大（置信度中）
    # 特殊机制字段原样透传（引擎按字段族接入）
    for k in ('buffWhenNotAttackingData', 'buffAfterHitsData', 'buffAfterHitsTime', 'groupMaxSize',
              'projectile2Data', 'onAttackActionData', 'onStartChargingActionData',
              'onStartingActionData', 'deathSpawnCharacterData', 'onKilledActionData',
              'shieldLostActionData', 'chargeSpeedMultiplier', 'damageSpecial', 'specialAttackRangeForStats',
              'attackSequence', 'attackSequenceMode', 'attackSequenceList'):
        if k in evolved_scd:
            out[k] = evolved_scd[k]
    return out
