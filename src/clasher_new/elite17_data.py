"""M8 — Elite17 精英卡（Hero 化）数据层。

规格：docs/elite17_spec.md（17 节）；原始数据：re/official/extracted/elite17/*.json +
base_stats.json + elite17_summary.json。

来源标注约定（与 evo_2025_data.py 一致）：
- [Fandom]  = docs/_elite_<卡名>.txt（本会话 CDP 抓取，规格书逐卡引用）
- [内存]    = re/official 官方 15.535 一致性快照（base_stats.json / overlay 表）
- [暂借]    = 数值缺失卡按规格书口径借用同类卡参数，逐条标注【暂借/待实测】
- [假设]    = 双源均无数值、采用标注假设的合理值【待实测】

设计决策（规格书 §0 + 数据提取子代理结论）：
1. 17/17 走冠军式 abilityData/use_ability 模式：圣水扣费 + 单次使用
   （BattleState.use_ability M8 分支统一处理扣费/单次，holder 只做效果）。
2. Hero 数值 = 独立表（base_stats/overlay + Fandom），非普通卡倍率；
   只覆写规格书给出 Hero 差异值的卡，其余沿用普通表（逐条标注）。
3. Wild slot 互斥在引擎层（battle.deploy_card）：卡组声明 hero 化 → 该卡觉醒禁用。
4. HeroCoin 不建模（元进度货币）。

本模块由 card_utils.py 导入末尾 apply() 驱动，只做三件事：
1. 注入 HERO_ABILITIES（card_data[name]['summonCharacterData']['abilityData'] 等价结构，
   供 Card.ability / BattleState.use_ability 消费——冠军同链路）；
2. 注册 Hero 专属派生角色为可构造卡（炮塔/TombQueen/Skeletrooper/Rhino/假人）；
3. 暴露 HERO_BASE_STATS（overlay 数值表，battle.apply_hero_overlay 消费）。
"""
import math


def _curve(value, src_level, length=17):
    """按官方统一曲线（×1.1/级）从 src_level 基准值生成 Common 轴 per-level 数组
    （index = level-1）。仅用于注册派生角色（引擎 set_level 消费）。"""
    base = value / (1.1 ** (src_level - 1))
    return [round(base * (1.1 ** i)) for i in range(length)]


def hval(value, src_level, level):
    """Hero 独立表数值：src_level 基准 → 当前战斗等级（官方统一曲线，标注口径）。"""
    return round(value * (1.1 ** (level - src_level)))


# ---------------------------------------------------------------------------
# Hero 独立数值表（overlay：部署即 Hero 形态时覆写实体数值）
# 只收录规格书给出 Hero 差异值的卡；未收录卡 = Hero 主数值沿用普通表（规格书未给差异）。
# 元组 = (数值, 基准等级, 来源标注)
# ---------------------------------------------------------------------------
HERO_BASE_STATS = {
    # §1 Knight [Fandom L1: 681/94/Shield 197]；内存 overlay ShieldHitpoints=200（双记，取 Fandom）
    'Knight': {'hp': (681, 1, '[Fandom L1]'), 'damage': (94, 1, '[Fandom L1]'),
               'shield': (197, 1, '[Fandom L1]; 内存 overlay=200 双记')},
    # §4 Valkyrie [Fandom L11: 1907/266]
    'Valkyrie': {'hp': (1907, 11, '[Fandom L11]'), 'damage': (266, 11, '[Fandom L11 AreaDmg]')},
    # §5 Wizard [Fandom L11: 832/281]
    'Wizard': {'hp': (832, 11, '[Fandom L11]'), 'damage': (281, 11, '[Fandom L11]')},
    # §6 Bowler [Fandom L11: 2081/289（AreaDmg，弹丸伤害）]
    'Bowler': {'hp': (2081, 11, '[Fandom L11]'), 'damage': (289, 11, '[Fandom L11 AreaDmg→弹丸]')},
    # §7 Giant [Fandom L11: 3968/253]
    'Giant': {'hp': (3968, 11, '[Fandom L11]'), 'damage': (253, 11, '[Fandom L11]')},
    # §9 MegaMinion [Fandom L11: 837/312]；Hero 版近战化 [Fandom Range Melee:Long] → 伤害走直接槽
    'MegaMinion': {'hp': (837, 11, '[Fandom L11]'), 'damage': (312, 11, '[Fandom L11]'),
                   'damage_target': 'direct'},
    # 其余 11 张：规格书未给 Hero 主数值差异（≈普通表）或全缺 → 不覆写，沿用普通表。
}

# ---------------------------------------------------------------------------
# Hero 能力表（abilityData 等价结构；费用/时长/半径等全部来自规格书对应节，
# 缺失处逐条标注【假设】/【暂借】。单次使用语义由引擎统一处理，不在此建模。）
# ---------------------------------------------------------------------------
KNIGHT_SHIELD = (197, 1, '[Fandom L1]; 内存=200 双记')

HERO_ABILITIES = {
    # §1 Knight — Triumphant Taunt（2 费）：6.5 格嘲讽 5s + 自身护盾 5s
    'Knight': {'name': 'Knight_hero_Ability', 'manaCost': 2,
               'tauntRadius': 6.5, 'tauntDuration': 5.0, 'shieldDuration': 5.0,
               'shieldValue': KNIGHT_SHIELD, 'source': '[Fandom 2026-03-02 削 7.5→6.5]'},
    # §2 Musketeer — Trusty Turret（3 费）：前方 3 格炮塔，寿命 10s 线性衰减，落地 AOE
    'Musketeer': {'name': 'Musketeer_hero_Ability', 'manaCost': 3,
                  'turretCard': 'MusketeerHeroTurret', 'frontOffset': 3.0,
                  'spawnDamage': (95, 3, '[Fandom L3 TurretSpawnDmg]'),
                  'spawnRadius': 2.0,  # [假设] 半径无字段，取中型 AOE 口径
                  'source': '[Fandom L3]'},
    # §3 MiniPekka — Breakfast Boost（1 费）：煎饼 22s/格或每击 +10s 进度（≤3 格）；
    #    加级 0/1/2/3 格 → +1/+2/+3/+5 级；回复 30% maxHP
    'MiniPekka': {'name': 'MiniPekka_hero_Ability', 'manaCost': 1,
                  'meterSeconds': 22.0, 'onHitProgress': 10.0, 'maxMeter': 3,
                  'levelsByMeter': [1, 2, 3, 5], 'healPct': 0.30,
                  'source': '[Fandom L17-21 表 2539/2793/3072/3379/3717]'},
    # §4 Valkyrie — Wild Whirlwind（3 费）：3.5s 旋风 0.25s/击 半径 2.5（AbilityDmg 97@L11），
    #    减伤 15%，塔伤 ×0.5；结束冲刺 5.5 格；旋风后禁攻
    'Valkyrie': {'name': 'ValkyrieHero_Ability', 'manaCost': 3,
                 'whirlDuration': 3.5, 'tick': 0.25, 'radius': 2.5,
                 'tickDamage': (97, 11, '[Fandom L11 AbilityDmg]'),
                 'damageReduction': 0.15, 'crownMult': 0.5,
                 'speedMult': 1.20,        # [假设] 「移速提升」无数值
                 'dashRange': 5.5, 'forbidAttack': 1.0,  # [假设] 禁攻时长无数值
                 'source': '[Fandom+内存 overlay DashLandingTime=100]'},
    # §5 Wizard — Fiery Flight（1 费；wiki 属性表 1 与信息框 2 矛盾，双记取 1）：
    #    1s 延迟升空 5s，移速 +50%，火球命中生成半径 4 / 2s 火旋风（TornadoDmg 43@L11）
    'Wizard': {'name': 'WizardHero_Ability', 'manaCost': 1,
               'flyDelay': 1.0, 'flyDuration': 5.0, 'speedMult': 1.5,
               'tornadoRadius': 4.0, 'tornadoDuration': 2.0,
               'tornadoDps': (43, 11, '[Fandom L11 TornadoDmg]'),
               'source': '[Fandom L11]'},
    # §6 Bowler — Stone Swish（2 费）：2.5s 蓄力（禁攻）→ 7.3s 迫击炮模式：
    #    射程 11.5、共 3 发、攻速 1.9s、伤害 AbilityDmg 508@L11（塔伤 ×0.5）
    'Bowler': {'name': 'BowlerHero_activate_ability', 'manaCost': 2,
               'chargeTime': 2.5, 'siegeDuration': 7.3, 'siegeRange': 11.5,
               'siegeShots': 3, 'siegeHitSpeed': 1.9,
               'siegeDamage': (508, 11, '[Fandom L11 AbilityDmg]'), 'crownMult': 0.5,
               'source': '[Fandom 2026-08-04 弹程 7.5→7（投射段，引擎简化为直接 AOE 命中）]'},
    # §7 Giant — Heroic Hurl（2 费）：抓 2 格内最高 HP 敌军部队扔出 9 格，
    #    落地 ImpactDmg 135@L11 + 眩晕 2s
    'Giant': {'name': 'GiantHero_Ability', 'manaCost': 2,
              'grabRadius': 2.0, 'throwRange': 9.0,
              'impactDamage': (135, 11, '[Fandom L11 ImpactDmg]'),
              'impactRadius': 1.5,  # [假设] 落地溅射半径无字段
              'stun': 2.0, 'source': '[Fandom L11]'},
    # §8 Goblins — Banner Brigade（1 费，条件窗）：最后一只哥布林死亡后落旗开窗 5s，
    #    窗口内按按钮 → 召出 2 只 Brigade Goblins（2026-08-04 x3→x2）
    'Goblins': {'name': 'GoblinHero_Ability', 'manaCost': 1,
                'window': 5.0, 'brigadeCount': 2,
                'source': '[Fandom Hero 主体≈普通表 78/48]'},
    # §9 MegaMinion — Wounding Warp（2 费）：瞬移到部署时标记的最低 HP 敌人处，
    #    WarpDmg 399@L11，之后永久对皇冠塔伤害 ×0.25（2026-08-04）
    'MegaMinion': {'name': 'MegaMinion_Teleport_Ability', 'manaCost': 2,
                   'warpDamage': (399, 11, '[Fandom L11 WarpDmg; 内存 Damage=330 双记]'),
                   'warpRadius': 1.5,  # [假设] 落点溅射半径无字段
                   'crownMult': 0.25, 'source': '[Fandom L11]'},
    # §10 IceWizard — 冰封自身→破碎冻结 AOE（机制结构 [内存 54 动作组]；数值全缺）：
    #     冰块 3s【假设】，破碎 AOE 冻结 2s【暂借 Freeze 语义】半径 3【暂借】+ 减速 30% 2s【暂借 Ice Golem/Hero】
    'IceWizard': {'name': 'IceWizardHero_Ability', 'manaCost': 2,  # 费用【假设】
                  'cubeDuration': 3.0,      # 【假设-待实测】
                  'freezeRadius': 3.0,      # 【暂借】
                  'freeze': 2.0,            # 【暂借 Freeze 语义-待实测】
                  'slowMult': 0.70, 'slowDuration': 2.0,  # 【暂借 Ice Golem/Hero】
                  'source': '[内存动作组 54 个; Fandom 页缺失]'},
    # §11 Tombstone — Regal Revive（5 费）：墓碑破碎 → Tomb Queen 升起（HP4224/Dmg422@L11），
    #     只攻建筑 sight 7；Queen 计时器数值缺【假设 15s-待实测】
    'Tombstone': {'name': 'Tombstone_hero_Ability', 'manaCost': 5,
                  'queenCard': 'TombQueen', 'queenLifetime': 15.0,  # 【假设-待实测】
                  'source': '[内存 TID_HERO_TOMBSTONE_MONSTER 1650@lv基准 vs Fandom L11 4224 双记]'},
    # §12 Berserker — Savage Survival（3 费）：熊灵 4s：攻速 0.2s / 移速 UltraFast(135) /
    #     HP 不低于 1 / 塔伤 ×0.25 / BearDmg 167@L11
    'Berserker': {'name': 'BerserkerHero_ability_group', 'manaCost': 3,
                  'duration': 4.0, 'hitSpeed': 0.2, 'speed': 135,
                  'bearDamage': (167, 11, '[Fandom L11 BearDmg]'), 'crownMult': 0.25,
                  'source': '[Fandom 属性表; 正文缺(机制按规格书)]'},
    # §13 DarkPrince — Destructive Dismount（3 费）：下马：本体落地溅射 + 犀牛独立冲塔
    #     （Rhino L11: HP1356/Dmg179/ChargeDmg358）
    'DarkPrince': {'name': 'DarkPrinceHero_Ability', 'manaCost': 3,
                   'mountCard': 'DarkPrinceHeroRhino',
                   'landingRadius': 1.2,
                   'landingDamageMult': 1.0,  # [假设] 落地溅射伤害=普攻（数值缺）
                   'source': '[内存 overlay+动作组; Fandom L6-16 表]'},
    # §14 Balloon — Coffin Cadets（2 费）：骷髅伞兵飞向 6 格内最近地面敌人：
    #     落地伤害 263@L11（对塔 ×0.1）+ 驻场（HP473/Dmg204@L11, 攻速 1.1s, VeryFast）
    'Balloon': {'name': 'BalloonHero_Ability', 'manaCost': 2,
                'cadetCard': 'Skeletrooper', 'seekRadius': 6.0,
                'landingDamage': (263, 11, '[Fandom L11 LandingDmg]'),
                'landingRadius': 1.5,  # [假设] 落地 AOE 半径无字段
                'crownMult': 0.10, 'flySpeed': 3.0,  # [假设] 伞降弹速
                'source': '[Fandom L11]'},
    # §15 BarbarianBarrel — Rowdy Reroll（1 费）：桶再滚一次 + 野蛮人回满血（勘误批1 用户口径）
    'BarbLog': {'name': 'BarbLogHero_spawn_reroll', 'manaCost': 1,
                'window': 10.0,  # [假设] 按钮窗口=桶滚出后 10s 内（单次）
                'healFull': True,  # 勘误批1（2026-09-04 用户口径）：开启技能后野蛮人回满血（原 healPct 0.50 作废）
                'source': '[Fandom Range 3 / Width 2.6]'},
    # §16 EliteArcher — Warp + Triple Shot + 假人分身（机制结构 [内存 17 动作组]；数值全缺）：
    #     瞬移 3 格【假设-暂借】；三连射攻速 0.3s【暂借-待实测】；假人 HP104 [内存·中置信] 寿命 5s【假设】
    'EliteArcher': {'name': 'EliteArcherHero_Ability', 'manaCost': 2,  # 费用【假设】
                    'warpRange': 3.0,        # 【假设-暂借】
                    'tripleShots': 3, 'tripleHitSpeed': 0.3,  # 【暂借-待实测】
                    'dummyCard': 'EliteArcherHeroDummy', 'dummyLifetime': 5.0,  # 【假设】
                    'source': '[内存动作组 17 个; Fandom 无页面]'},
    # §17 IceGolemite — 冰雪光环（机制结构 [内存 buff 分档 10 个]）：半径 4 / 3 次脉冲 /
    #     脉冲伤害 69@L11【暂借 Ice Golem/Hero】/ 减速 30% 2s；按体型分档 freeze/slow【假设规则】
    'IceGolemite': {'name': 'IceGolemiteHero_OnAbilityActivationGroup', 'manaCost': 2,  # 费用【假设】
                    'radius': 4.0, 'pulses': 3, 'interval': 1.0,  # 【暂借 Ice Golem/Hero】
                    'pulseDamage': (69, 11, '[暂借 Ice Golem/Hero L11]'),
                    'slowMult': 0.70, 'slowDuration': 2.0,
                    'smallFreeze': 1.0,   # 【假设】小体型档冻结 1s
                    'source': '[内存 buff 分档命名; wiki Ice Golem/Hero Snowstorm]'},
}

# ---------------------------------------------------------------------------
# Hero 专属派生角色（注册为可构造卡；数值全显式，禁止引擎静默编造）
# ---------------------------------------------------------------------------

# §2 火枪手炮塔 [Fandom L3: TurretHP 717 / 衰减 71.7/s / TurretDmg 65 / SpawnDmg 95；
#     内存 0x76c726382a80 HP600/Range4000 双记]。攻速【假设 0.8s-待实测】。
MUSKETEER_TURRET_SCD = {
    'name': 'MusketeerHeroTurret', 'rarity': 'Rare', 'sightRange': 5500,
    'deployTime': 500, 'hitpoints': 717, 'damage': 65,
    'hitSpeed': 800, 'loadTime': 300, 'range': 4000, 'lifeTime': 10000,
    'attacksGround': True, 'attacksAir': True, 'collisionRadius': 500,
    'tidTarget': 'TID_TARGETS_AIR_AND_GROUND',
    'data_note': '[Fandom L3 Turret 表]; hitSpeed [假设]; 寿命衰减走 Building 通用 lifeTime 管线; '
                 '弹丸飞行段简化为直接结算（MusketeerTurret_Projectile 数值行缺失）',
}

# §11 Tomb Queen [Fandom L11: 4224/422; 内存 1650@lv基准 双记]；只攻建筑 sight 7
#     [规格书 2026-08-04 sight 5.5→7]；攻速【假设 1.5s-待实测】。
TOMB_QUEEN_SCD = {
    'name': 'TombQueen', 'rarity': 'Legendary', 'sightRange': 7000,
    'deployTime': 500, 'speed': 60, 'hitpoints': 4224, 'damage': 422,
    'hitSpeed': 1500, 'loadTime': 600, 'range': 1600,
    'attacksGround': True, 'collisionRadius': 600,
    'tidTarget': 'TID_TARGETS_BUILDINGS',
    'data_note': '[Fandom L11 Queen 表; 内存 1650 双记]; hitSpeed [假设]',
}

# §14 Skeletrooper（骷髅伞兵）[Fandom L11: HP473/Dmg204/攻速 1.1s/VeryFast(120)/Range 6.5/对地]
SKELETROOPER_SCD = {
    'name': 'Skeletrooper', 'rarity': 'Legendary', 'sightRange': 7000,
    'deployTime': 0, 'speed': 120, 'hitpoints': 473, 'damage': 204,
    'hitSpeed': 1100, 'loadTime': 400, 'range': 6500,
    'attacksGround': True, 'collisionRadius': 450,
    'tidTarget': 'TID_TARGETS_GROUND',
    'data_note': '[Fandom L11 Skeletrooper 表]',
}

# §13 犀牛坐骑（独立单位冲塔）[Fandom L11: HP1356/Dmg179/ChargeDmg358]；
#     冲锋复用 Prince 管线（chargeRange【假设 3.5 格-同 Prince】攻速【假设 1.4s】）。
RHINO_SCD = {
    'name': 'DarkPrinceHeroRhino', 'rarity': 'Epic', 'sightRange': 5500,
    'deployTime': 0, 'speed': 90, 'hitpoints': 1356, 'damage': 179,
    'damageSpecial': 358, 'chargeRange': 3500,
    'hitSpeed': 1400, 'loadTime': 600, 'range': 1200,
    'attacksGround': True, 'collisionRadius': 550,
    'tidTarget': 'TID_TARGETS_BUILDINGS',
    'data_note': '[Fandom L11 Rhino 表]; chargeRange/hitSpeed [假设]',
}

# §16 EliteArcher 假人分身 [内存 0x76c72648ce00: HP104/Range5500/Speed40（中置信，列混排）]
ELITE_ARCHER_DUMMY_SCD = {
    'name': 'EliteArcherHeroDummy', 'rarity': 'Common', 'sightRange': 5500,
    'deployTime': 0, 'speed': 40, 'hitpoints': 104, 'damage': 0,
    'hitSpeed': 10000, 'range': 0,
    'attacksGround': False, 'collisionRadius': 400,
    'tidTarget': 'TID_TARGETS_NONE',
    'data_note': '[内存 Dummy 表 HP104 中置信]; 寿命由引擎 hero 能力控制',
}

# 注册清单：(角色定义, hp 基准值, 基准等级)
# 稀有度统一按 Common 轴注册（_curve 数组 index=level-1；派生角色不走稀有度起始级语义）
_DERIVED = (
    (MUSKETEER_TURRET_SCD, 717, 3),
    (TOMB_QUEEN_SCD, 4224, 11),
    (SKELETROOPER_SCD, 473, 11),
    (RHINO_SCD, 1356, 11),
    (ELITE_ARCHER_DUMMY_SCD, 104, 1),
)


def apply(card_data, characters, character_to_card):
    """card_utils 导入末尾调用：注入 17 卡 abilityData + 注册 Hero 派生角色。"""
    # ① 能力表挂载（冠军同链路：Card.ability → BattleState.use_ability 消费）
    for card_name, ab in HERO_ABILITIES.items():
        entry = card_data.get(card_name)
        if entry is None:
            continue
        scd = entry.setdefault('summonCharacterData', entry)
        scd['abilityData'] = ab
    # ② 派生角色注册（显式 per-level 曲线，来源标注于 SCD.data_note）
    for scd, base_val, src_lv in _DERIVED:
        name = scd['name']
        if name in card_data:
            continue
        card_data[name] = {'name': name, 'tidType': 'TID_CARD_TYPE_CHARACTER',
                           'summonCharacterData': dict(scd)}
        characters[name] = {
            'name': name, 'rarity': 'Common',
            'hitpoints_per_level': _curve(base_val, src_lv),
            'damage_per_level': _curve(scd.get('damage') or 0, src_lv) if scd.get('damage') else [],
        }
        character_to_card.setdefault(name, name)
