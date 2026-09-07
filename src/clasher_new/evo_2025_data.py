"""M6 — 2025 新觉醒 7 张数据层（快照 gamedata.json 无 evolvedSpellsData，自建等价数据）。

来源标注约定：
- [Fandom]  = docs/_page_N.txt（CDP 采集 2026-09-03，页码映射见 docs/evo_2025_new7.md）
- [内存]    = re/official/extracted/all_tables_final.json 官方 15.535 一致性快照（权威性 > Fandom）
- [假设]    = Fandom/内存均无数值，采用标注假设的合理值（待 L4 对拍）

本模块只做两件事（由 card_utils.py 末尾 apply() 驱动）：
1. 为 7 张卡注入 card_data[name]['evolvedSpellsData'] 等价结构——
   summonCharacterData 仅含 name（数值与基础形态一致，由 derive_evolved_stats 保底）；
   evo2025Hooks 为引擎钩子参数包（已加入 evolutions.M5_EVO_PASSTHROUGH透传）。
2. 注册 3 个派生角色（Souldier / SkeletonArmy_EV1_General / SkeletonArmy_EV1_Shadow）
   为可构造卡 + 官方数值表行（显式 per-level 数组，不走 1.1 曲线推导）。
"""

# ---------------------------------------------------------------------------
# per-level 数值（全部显式列出，禁止引擎内静默编造）
# ---------------------------------------------------------------------------

# 觉醒皇家幽灵 Souldier [Fandom _page_4.txt Souldiers Attributes 表 lv9-16]
# 列：Spawn Damage = Hitpoints = Area Damage（DPS=值/1.8 与表吻合）
SOULDDIER_PER_LEVEL = [67, 74, 81, 89, 98, 108, 119, 130]  # lv9..16, Legendary 轴(index=lv-9)
SOULDDIER_SPAWN_RARITY = 'Legendary'

# 觉醒皇家野猪落地伤害 [Fandom _page_5.txt Landing Damage 表 lv3-16: 54,59,65,71,79,86,95,105,115,
# 127,139,153,168,185] × 两次平衡削弱（History: 2/3/2026 -27%、4/5/2026 -49%，×0.73×0.51=×0.3723）。
# ⚠️ 口径说明：该页为 stub，表值可能未随平衡更新（Ghost 页表值已反映平衡，此页存疑）；
# 取「平衡后现行值」口径，原表值(115@lv11)记录于此备查。若表值已含削弱则此 ×0.3723 为重复削弱。
ROYAL_HOGS_LANDING_RAW = [54, 59, 65, 71, 79, 86, 95, 105, 115, 127, 139, 153, 168, 185]  # lv3..16
ROYAL_HOGS_LANDING_NERF = 0.73 * 0.51
ROYAL_HOGS_LANDING_PER_LEVEL = [round(v * ROYAL_HOGS_LANDING_NERF) for v in ROYAL_HOGS_LANDING_RAW]
ROYAL_HOGS_LANDING_RARITY = 'Rare'  # index=lv-3

# 觉醒骷髅军团 General Gerry [内存 0x76c726384680]：Hitpoints=32 / ShieldHitpoints=32 / Damage=32
# （10 行同值 → 等级无关）；Range=1600、Speed=90、SightRange=5500、TID 实证。
# Hit Speed=1.0s / First Hit=0.5s [Fandom _page_3.txt General Gerry Attributes]。
GERRY_LEVEL_VALUE = 32

# 亡影（Shadow）[Fandom _page_3.txt Shadow Skeleton Attributes]：Hit Speed 1.1s / First Hit 0.5s /
# Speed Medium(60) / Melee Short(0.5)；HP「unlimited」[假设→法术可伤害血池 999（详见矩阵文档）]；
# 伤害与骷髅本体一致（从官方数值表 Skeleton 行拷贝）。
SHADOW_HP_POOL = 999


# ---------------------------------------------------------------------------
# 7 卡 evolvedSpellsData 等价数据（evo2025Hooks = 引擎钩子参数包）
# ---------------------------------------------------------------------------

EVO_2025 = {
    # ① Princess：攻击不变（快照本已是 AoE+对空：projectileData radius 2000 + tidTarget AIR&GROUND），
    #    觉醒语义 = 减速箭 + 死亡减速领域 [Fandom _page_7.txt]
    #    首发减速、之后每 2 发减速（4/8/2026 平衡：every 2 hits from every 3）；半径 3.0 / 30%（Ice Wizard 同款）；
    #    时长 5.5s（4/8/2026 平衡 7s→5.5s；死亡领域时长同口径 [假设：平衡同时覆盖领域]）。
    'Princess': {
        'summonCharacterData': {'name': 'Princess'},
        'evo2025Hooks': {
            'slowAttack': {'radius': 3.0, 'speedMult': 0.70, 'duration': 5.5, 'everyNHits': 2},
            'deathSlowZone': {'radius': 3.0, 'speedMult': 0.70, 'duration': 5.5},
        },
    },
    # ② MinionHorde：首击面纱（Dark Elixir veil）。时长无直接字段 [假设：Fandom History
    #    「invisible hit speed multiplier x0.5→x0.67」解读为 面纱时长=hitSpeed(1.0s)×倍率 → 现行 0.67s；
    #    首击本身被闪避（无效）]。
    'MinionHorde': {
        'summonCharacterData': {'name': 'Minion'},
        'evo2025Hooks': {
            'firstHitVeil': 0.67,
        },
    },
    # ③ RoyalHogs：起飞直达最近建筑；受击或攻击时落地 + 落地 AoE [Fandom _page_5.txt]。
    #    落地半径无字段 [假设 1.5 格]；数值同原版（identical stats）。
    'RoyalHogs': {
        'summonCharacterData': {'name': 'RoyalHog'},
        'evo2025Hooks': {
            'flyingAssault': {
                'landingDamagePerLevel': ROYAL_HOGS_LANDING_PER_LEVEL,
                'landingDamageRarity': ROYAL_HOGS_LANDING_RARITY,
                'landingRadius': 1.5,  # [假设]
            },
        },
    },
    # ④ Ghost：显形时召唤 2 Souldier（召唤伤害）[Fandom _page_4.txt]。
    #    召唤伤害半径无字段 [假设 1.0 格，与 Ghost 自身 areaDamageRadius 同口径]；
    #    Souldier 存活时长：Fandom 策略节明言「do not despawn naturally」（与卡面引文「会消失」冲突，
    #    取策略节为准=不自然消失，冲突记录于矩阵文档）。
    'Ghost': {
        'summonCharacterData': {'name': 'Ghost'},
        'evo2025Hooks': {
            'souldierSummon': {
                'count': 2,
                'spawnDamagePerLevel': SOULDDIER_PER_LEVEL,
                'spawnDamageRarity': SOULDDIER_SPAWN_RARITY,
                'radius': 1.0,  # [假设]
            },
        },
    },
    # ⑤ SkeletonArmy：15 骷髅散布 + General Gerry 后排 [Fandom _page_3.txt + 内存 0x76c726384680]。
    #    Gerry 在场：骷髅死亡 → 亡影（无敌+不可选取，法术可伤害）；Gerry 阵亡：亡影全灭 + 不再转化。
    'SkeletonArmy': {
        'summonCharacterData': {'name': 'Skeleton'},
        'evo2025Hooks': {
            'generalGerry': {
                'card': 'SkeletonArmy_EV1_General',
                'shadowCard': 'SkeletonArmy_EV1_Shadow',
                'backOffset': 1.0,  # [假设] Gerry 后排 = 部署中心向己方后方 1 格
            },
        },
    },
    # ⑥ BabyDragon：攻击期间气流 [Fandom _page_2.txt]：8×9 格区域 敌 -30% / 友 +30%（1/12/2025
    #    平衡后 30%）；死后残留 2s；6/7/2026 平衡伤害 +4%。
    #    8×9 矩形 → 圆形半径 4.0 [假设]；脉冲 0.25s [假设： aura 节拍]。
    'BabyDragon': {
        'summonCharacterData': {'name': 'BabyDragon'},
        'evo2025Hooks': {
            'gust': {
                'radius': 4.0, 'allySpeedMult': 1.30, 'enemySpeedMult': 0.70,
                'pulse': 0.25, 'deathLinger': 2.0, 'damageMult': 1.04,
            },
        },
    },
    # ⑦ Furnace（gamedata 卡名 FirespiritHut）：数值同原版（identical stats）；
    #    热生成 2.4s/灵 + 左右侧向交替（原正面）[Fandom _page_1.txt：Hot Spawn Speed 2.4s，
    #    「spawn first to the left, then to the right」]。
    #    基础出兵循环引擎未建模（矩阵已录），觉醒钩子自带出兵循环。
    'FirespiritHut': {
        'summonCharacterData': {'name': 'FirespiritHut'},
        'evo2025Hooks': {
            'hotSpawn': {
                'interval': 2.4, 'card': 'FireSpirits',
                'sideOffset': 0.8, 'forwardOffset': 0.8,  # [假设：侧向/前向偏移格数]
            },
        },
    },
}

# 周期表键别名（Furnace 为 Fandom/官方叫法，gamedata 卡名 = FirespiritHut）
CARD_NAME_ALIASES = {'Furnace': 'FirespiritHut'}


# ---------------------------------------------------------------------------
# 派生角色定义（显式 per-level 注册，不走 1.1 曲线）
# ---------------------------------------------------------------------------

SOULDDIER_SCD = {
    'name': 'Souldier', 'rarity': 'Common', 'sightRange': 5500, 'deployTime': 200,
    'speed': 90, 'hitpoints': SOULDDIER_PER_LEVEL[0], 'damage': SOULDDIER_PER_LEVEL[0],
    'hitSpeed': 1800, 'loadTime': 600,
    'range': 1200, 'attacksGround': True, 'areaDamageRadius': 1000, 'collisionRadius': 450,
    'tid': 'TID_CHARACTER_SOULDDIER', 'source': 'characters',
    'tidTarget': 'TID_TARGETS_GROUND', 'tidSpeed': 'TID_SPEED_4',
}

GERRY_SCD = {
    'name': 'SkeletonArmy_EV1_General', 'rarity': 'Common', 'sightRange': 5500,
    'deployTime': 1000, 'speed': 90, 'hitpoints': GERRY_LEVEL_VALUE,
    'damage': GERRY_LEVEL_VALUE,
    'shieldHitpoints': GERRY_LEVEL_VALUE,  # [内存 ShieldHitpoints=32] Guard 同款护盾
    'hitSpeed': 1000, 'loadTime': 500, 'range': 1600, 'attacksGround': True,
    'collisionRadius': 500, 'tid': 'TID_CHARACTER_SKELETON_ARMY_EV1_GENERAL',
    'source': 'characters', 'tidTarget': 'TID_TARGETS_GROUND', 'tidSpeed': 'TID_SPEED_4',
}

SHADOW_SCD = {
    'name': 'SkeletonArmy_EV1_Shadow', 'rarity': 'Common', 'sightRange': 5500,
    'deployTime': 0, 'speed': 60, 'hitpoints': SHADOW_HP_POOL,
    'damage': 32,  # 基准（lv1 Common 轴, 官方 Skeleton 行 per-level 覆盖）
    'hitSpeed': 1100, 'loadTime': 500, 'range': 500, 'attacksGround': True,
    'collisionRadius': 400, 'tid': 'TID_CHARACTER_SKELETON_ARMY_EV1_SHADOW',
    'source': 'characters', 'tidTarget': 'TID_TARGETS_GROUND', 'tidSpeed': 'TID_SPEED_3',
}


def _register(char_def, hp_pl, dmg_pl, rarity, card_data, characters, character_to_card):
    """登记派生角色：card_data 合成卡入口 + characters 官方数值表行（显式数组）。"""
    name = char_def['name']
    if name in card_data:
        return
    card_data[name] = {'name': name, 'tidType': 'TID_CARD_TYPE_CHARACTER',
                       'summonCharacterData': dict(char_def)}
    characters[name] = {'name': name, 'rarity': rarity,
                        'hitpoints_per_level': list(hp_pl), 'damage_per_level': list(dmg_pl)}
    character_to_card.setdefault(name, name)


def apply(card_data, characters, character_to_card):
    """card_utils 导入末尾调用：注入 7 卡 evolvedSpellsData + 注册派生角色。"""
    for card_name, evo in EVO_2025.items():
        if card_name in card_data and 'evolvedSpellsData' not in card_data[card_name]:
            card_data[card_name]['evolvedSpellsData'] = evo

    # Souldier： Legendary 轴 lv9..16 [Fandom]
    _register(SOULDDIER_SCD, SOULDDIER_PER_LEVEL, SOULDDIER_PER_LEVEL, 'Legendary',
              card_data, characters, character_to_card)
    # General Gerry：内存 10 行同值 32 → 等级无关常量数组（Epic 轴对齐 SkeletonArmy）
    _const = [GERRY_LEVEL_VALUE] * 16
    _register(GERRY_SCD, _const, _const, 'Epic', card_data, characters, character_to_card)
    # Shadow：HP 血池常量（法术可伤害）；伤害拷贝官方 Skeleton 行（缺行回退 1.1 曲线）
    sk = characters.get('Skeleton') or {}
    sh_dmg = list(sk.get('damage_per_level') or [])
    if not sh_dmg:
        sh_dmg = [round(32 * (1.1 ** i)) for i in range(16)]
    _register(SHADOW_SCD, [SHADOW_HP_POOL] * 16, sh_dmg, 'Common',
              card_data, characters, character_to_card)


# ===========================================================================
# M7 — 数据接入三卡引擎钩子的数据补全（Ronin / Vines / Spirit Empress）
# gamedata.json 的 Ronin.summonCharacterData / Vines.areaEffectObjectData /
# MergeMaiden 双形态条目已注入（数值行验证：Ronin lv11 1779/371、VinesProjectile
# Epic 轴 lv11=153、MergeMaiden_Mounted Legendary lv9 轴 lv11 hp=1798/dmg=309）。
# 本节只补引擎消费缺口，来源标注约定同上（[Fandom]=docs/_fp_*.txt 2026-09-03 CDP）。
# ===========================================================================

# Vines 投掷物弹速：gamedata projectileData 仅 name/radius（快照缺 speed 字段）。
# [假设] 取 600（Fireball 同款法术弹速, 官方法术弹速带 350~1100, 待 L4 对拍）。
VINES_PROJECTILE_SPEED = 600

# Spirit Empress 双形态角色定义 [Fandom _fp_SpiritEmpress.txt 属性表 2026-09-03 CDP]：
#   Air Form：   Cost 6 / Hit Speed 1.6s / First Hit 0.6s / Speed Medium(60) / Range 5 / Air&Ground
#   Ground Form：Cost 3 / Hit Speed 1.2s / First Hit 0.3s / Speed Fast(90) / Melee Medium(1.2) / Ground
# 数值基准 = Legendary 轴起始 lv9（hp 1486 / dmg 255, 与官方数值表行 per-level 数组首值一致）；
# 两形态共享 hp/dmg per-hit（Fandom 表 air dps 159.4=255/1.6、ground dps 212.5=255/1.2 反推吻合）。
# collisionRadius 快照无字段 → [假设] 500（中型单位同款, 待 L4）。
MERGE_MAIDEN_MOUNTED_SCD = {
    'name': 'MergeMaiden_Mounted', 'rarity': 'Legendary', 'sightRange': 5500,
    'deployTime': 1000, 'speed': 60, 'hitpoints': 1486, 'damage': 255,
    'hitSpeed': 1600, 'loadTime': 600, 'range': 5000,
    'attacksGround': True, 'attacksAir': True, 'collisionRadius': 500,
    'tid': 'TID_CHARACTER_MERGE_MAIDEN_MOUNTED', 'source': 'characters',
    'tidTarget': 'TID_TARGETS_AIR_AND_GROUND', 'tidSpeed': 'TID_SPEED_3',
    'data_note': 'Spirit Empress 飞行远程形态 [Fandom Air Form Attributes]; collisionRadius [假设 500]',
}

MERGE_MAIDEN_NORMAL_SCD = {
    'name': 'MergeMaiden_Normal', 'rarity': 'Legendary', 'sightRange': 5500,
    'deployTime': 1000, 'speed': 90, 'hitpoints': 1486, 'damage': 255,
    'hitSpeed': 1200, 'loadTime': 300, 'range': 1200,
    'attacksGround': True, 'attacksAir': False, 'collisionRadius': 500,
    'tid': 'TID_CHARACTER_MERGE_MAIDEN_NORMAL', 'source': 'characters',
    'tidTarget': 'TID_TARGETS_GROUND', 'tidSpeed': 'TID_SPEED_4',
    'data_note': 'Spirit Empress 地面近战形态 [Fandom Ground Form Attributes]; collisionRadius [假设 500]',
}


def apply_m7(card_data, characters, character_to_card, air_units=None):
    """M7 数据接入补全（card_utils 导入末尾调用，紧随 apply()）：
    ① VinesProjectile 弹速注入（快照缺 speed 字段, 弹道永不落地的引擎缺口）。
    ② Spirit Empress 双形态 summonCharacterData 挂载——gamedata 的
       MergeMaiden_Mounted/Normal 条目只有 summonCharacter 引用（无数值体）,
       引擎 Card 构造需要 summonCharacterData；数值由官方 characters 行
       （MergeMaiden_Mounted 已有 / Normal 行在此补建）在 set_level 覆盖。
    ③ MergeMaiden_Mounted 飞行标记（air_units 名册, Card.is_air_unit 数据源）。
    ④ MergeMaiden_Normal 官方数值表行补建（Legendary lv9 轴, hp/dmg 与 Mounted
       行同源：Fandom 表两形态共享 hp 1798@lv11 与 per-hit 伤害, 仅攻速/射程/位面不同）。"""
    # ① Vines 弹速
    _vines = card_data.get('Vines') or {}
    _vpd = _vines.get('projectileData')
    if _vpd is not None and not _vpd.get('speed'):
        _vpd['speed'] = VINES_PROJECTILE_SPEED

    # ② 双形态 summonCharacterData（保留 gamedata 原 manaCost 等卡面字段）
    for form_name, scd in (('MergeMaiden_Mounted', MERGE_MAIDEN_MOUNTED_SCD),
                           ('MergeMaiden_Normal', MERGE_MAIDEN_NORMAL_SCD)):
        entry = card_data.get(form_name)
        if entry is not None and 'summonCharacterData' not in entry:
            entry['summonCharacterData'] = dict(scd)

    # ③ Mounted 飞行标记（air_units 由 flying_height!=0 派生, 此处直接补名册）
    _mm = characters.get('MergeMaiden_Mounted')
    if _mm is not None:
        if not _mm.get('flying_height'):
            _mm['flying_height'] = 1000  # [假设] 任意非 0 即判定飞行
        if air_units is not None and 'MergeMaiden_Mounted' not in air_units:
            air_units.append('MergeMaiden_Mounted')

    # ④ Normal 官方数值表行补建（挂载后 set_level 走 official_table 缩放）
    if _mm is not None and 'MergeMaiden_Normal' not in characters:
        characters['MergeMaiden_Normal'] = {
            'name': 'MergeMaiden_Normal',
            'rarity': _mm.get('rarity') or 'Legendary',
            'hitpoints_per_level': list(_mm.get('hitpoints_per_level') or []),
            'damage_per_level': list(_mm.get('damage_per_level') or []),
            'flying_height': 0,
            'data_note': 'M7 补建：与 Mounted 行同源（Fandom 两形态共享 hp/per-hit dmg, '
                         '差异仅在攻速 1.2s/射程 melee/位面 ground）',
        }
        character_to_card.setdefault('MergeMaiden_Normal', 'MergeMaiden_Normal')
