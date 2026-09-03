import json
# （原依赖 fastcore.nested_idx，仅一处嵌套取值，已内联等价实现——消除非必要依赖）

with open('gamedata.json') as f:
    data = json.load(f)

with open('cards_stats_characters.json') as f:
    characters_data = json.load(f)
    air_units = [each['name'] for each in characters_data if each['flying_height'] != 0]
    characters = {each['name']:each for each in characters_data}
with open('cards_stats_spell.json') as f:
    spells_data = json.load(f)
    spells = {each['name']:each for each in spells_data}
with open('cards_stats_building.json') as f:
    buildings_data = json.load(f)
    buildings = {each['name']:each for each in buildings_data}
with open('cards_stats_projectile.json') as f:
    projectiles = {each['name']:each for each in json.load(f)}

data = data['items']['spells']
card_data = {each['name']: each for each in data}

card_data['Golemite'] = {'name': 'Golemite', 'summonCharacterData':card_data['Golem']['summonCharacterData']['deathSpawnCharacterData']}

lava_pups = card_data['LavaHound']['summonCharacterData']['deathSpawnCharacterData']
barbarian = card_data['BattleRam']['summonCharacterData']['deathSpawnCharacterData']
card_data['LavaPups'] = {'name': 'LavaPups', 'summonCharacterData':lava_pups} | lava_pups
card_data['Barbarian'] = {'name': 'Barbarian', 'summonCharacterData': barbarian} | barbarian

# The king tower is not defined in `gamedata.json`, have to hard code it here.
king_tower_stats = {
    'name': 'KingTower',
    'tidType': 'TID_TYPE_TOWER_TROOP',
    'summonCharacterData': {
        'name': 'KingTower',
        'hitpoints': 2100,
        'hitSpeed': 1000,
        'damage': 109,
        'sightRange': 7000,
        'range': 7000,
        'collisionRadius': 1400,
        'tidTarget': 'TID_TARGETS_AIR_AND_GROUND',
        'deployTime': 3300,
        'loadTime': 700,
        'projectileData': {
            'name': 'KingProjectile',
            'speed': 600,
            'damage': 109,
        }
    }
}
card_data['KingTower'] = king_tower_stats
card_data['King_PrincessTowers']['summonCharacterData'] = card_data['King_PrincessTowers']['statCharacterData']
# 2025 塔防兵系统：King_CannonTowers / King_KnifeTowers / King_ChefTowers 与公主塔同构（statCharacterData 挂载）
for _kt in ('King_CannonTowers', 'King_KnifeTowers', 'King_ChefTowers'):
    if _kt in card_data and card_data[_kt].get('statCharacterData'):
        card_data[_kt]['summonCharacterData'] = card_data[_kt]['statCharacterData']

# 角色名 → 卡名 反查表（Goblin→Goblins、Skeleton→Skeletons），供落地出兵/亡语生成使用
character_to_card = {}
for _cname, _centry in card_data.items():
    _s = _centry.get('summonCharacterData') or _centry
    _nm = _s.get('name')
    if _nm and _nm not in character_to_card:
        character_to_card[_nm] = _cname


def _register_derived_character(char_def):
    """把 gamedata 里内嵌的角色定义（能力召唤物/觉醒亡语等，数值为起始级基准）
    注册为可构造卡：生成合成 per-level 数组（×1.1/级，实测统一曲线），
    使标准 set_level 管线自动缩放到目标等级（11-16 全支持）。"""
    name = char_def.get('name')
    if not name or name in card_data:
        return
    hp0 = char_def.get('hitpoints') or 0
    dmg0 = char_def.get('damage') or 0
    entry = {'name': name, 'tidType': 'TID_CARD_TYPE_CHARACTER',
             'summonCharacterData': dict(char_def)}
    card_data[name] = entry
    if name not in characters and hp0:
        characters[name] = {
            'name': name,
            'hitpoints_per_level': [round(hp0 * (1.1 ** i)) for i in range(19)],
            'damage_per_level': [round(dmg0 * (1.1 ** i)) for i in range(19)] if dmg0 else [],
        }
        character_to_card.setdefault(name, name)


# 英雄能力召唤物 / 觉醒特殊生成物：从 abilityData 与 evolvedSpellsData 中登记（M3/M4）
def _scan_and_register(obj):
    """递归扫描 abilityData/action 树，登记所有带 hitpoints 的内嵌角色定义"""
    if isinstance(obj, dict):
        if obj.get('hitpoints') and obj.get('name'):
            _register_derived_character(obj)
        for v in obj.values():
            _scan_and_register(v)
    elif isinstance(obj, list):
        for v in obj:
            _scan_and_register(v)


for _c in list(card_data.values()):
    _scd = _c.get('summonCharacterData') or {}
    _ab = _scd.get('abilityData') or {}
    if _ab:
        _scan_and_register(_ab)
    _pd = _scd.get('projectileData') or {}
    _tb = _pd.get('targetBuffData') or {}
    if _tb.get('deathSpawnData'):
        # 女巫妈妈诅咒：目标死亡生成 VoodooHog（注册该生成物为可构造卡）
        _scan_and_register(_tb['deathSpawnData'])
    _evo = _c.get('evolvedSpellsData') or {}
    _evo_scd = _evo.get('summonCharacterData') or {}
    for _k in ('deathSpawnCharacterData', 'onKilledActionData'):
        if _evo_scd.get(_k):
            _scan_and_register(_evo_scd[_k])

def _rarity_level_index(rarity, level):
    """稀有度 → 等级索引（数组 0 为该稀有度起始等级）：Common=lv1, Rare=lv3, Epic=lv6,
    Legendary=lv9, Champion=lv11。未知稀有度按 Common。"""
    if rarity == 'Rare':
        return level - 3
    if rarity == 'Epic':
        return level - 6
    if rarity == 'Legendary':
        return level - 9
    if rarity == 'Champion':
        return level - 11
    return level - 1


def _value_at_level(arr, rarity, level, base):
    """按稀有度等级轴取数组值：
    - 索引在数组内 → 直接取值
    - 索引越界（支持到 16 级及更远）→ 按实测曲线 ×1.1 延续（与数据文件延伸口径一致）
    - 数组为空 → 按 base（lv1 基准）×1.1^(level-1) 曲线推导
    """
    li = _rarity_level_index(rarity, level)
    if arr and 0 <= li < len(arr):
        return arr[li]
    if arr:
        return round(arr[-1] * (1.1 ** (li - len(arr) + 1)))
    return round(base * (1.1 ** (level - 1))) if base else 0


def level_scale(level):
    """lv1 基准 → 指定等级的放大系数（官方曲线 1.1/级，11-16 级均适用）。
    注：导出数组相邻级比值平均 1.0985 系逐级舍入的观察值，底层曲线即 1.1
    （wiki 验证值如 Guardienne 1621 = base×1.1^10 吻合）。用于能力/觉醒/特殊机制换算。"""
    return 1.1 ** (level - 1)


class Card:
    # 全局默认等级：BattleState 在构造时设为自身 card_level，使战斗内所有 Card() 构造
    # 自动继承该等级（单战斗串行假设；多战斗并行场景应显式传 level）。
    default_level = 11

    def __init__(self, card_name, level=None):
        self.level = level if level is not None else Card.default_level
        self.data = card_data[card_name]
        self.data.setdefault('summonCharacterData', self.data)
        self.hp = self.data['summonCharacterData'].get('hitpoints', 0)
        self.elixir = self.data.get('manaCost', 0) # princess towers don't have elixir cost
        self.name = self.data['name']
        self.damage = self.data['summonCharacterData'].get('damage', 0)
        self._base_damage0 = self.damage  # M4.5 动作链：缩放基准（set_level 前的原始基准伤害）
        self._base_hp0 = self.hp          # 缺失 per-level 表时的曲线推导基准（lv1）
        self.spawn_number = self.data.get('summonNumber', 1)
        self.spawn_delay = self.data.get('summonDeployDelay', 0) / 1000
        self.spawn_radius = self.data.get('summonRadius', 550) / 1000

        self.area_damage_radius = self.data['summonCharacterData'].get('areaDamageRadius', 0) / 1000
        self.projectile_damage_radius = ((self.data.get('summonCharacterData') or {})
                                         .get('projectileData', {}) or {}).get('spawnProjectileData', {}).get('radius')
        self.collision_radius = self.data['summonCharacterData'].get('collisionRadius', 1000) / 1000
        self.hit_speed = self.data['summonCharacterData'].get('hitSpeed', 0) / 1000
        self.load_time = self.data['summonCharacterData'].get('loadTime', 0) / 1000
        self.speed = self.data['summonCharacterData'].get('speed', 0)/50
        self.target_only_buildings = self.data['summonCharacterData'].get('tidTarget', '') == "TID_TARGETS_BUILDINGS"
        self.is_air_unit = self.name in air_units or self.data['summonCharacterData'].get('name', '') in air_units
        self.attack_air = 'AIR' in self.data['summonCharacterData'].get("tidTarget", '')
        self.attack_ground = ('GROUND' in self.data['summonCharacterData'].get('tidTarget', '')) or self.target_only_buildings
        self.range = self.data['summonCharacterData'].get('range', 0) / 1000
        self.sight_range = self.data['summonCharacterData'].get('sightRange', 0) / 1000
        self.deploy_time = self.data['summonCharacterData'].get('deployTime', 0) / 1000
        self.charge_range = self.data['summonCharacterData'].get('chargeRange', 0) / 1000

        self.projectiles = 'projectileData' in self.data['summonCharacterData']
        self.projectile_data = Projectile(self.data['summonCharacterData'].get('projectileData', {}))
        # —— M1 弹道生成链：二段弹（数值表 spawn_projectile）与落地出兵（gamedata spawnCharacterData）——
        if self.projectile_data.name:
            _prow = projectiles.get(self.projectile_data.name)
            if _prow:
                self.projectile_data.spawn_projectile = _prow.get('spawn_projectile')
        _pdat = self.data['summonCharacterData'].get('projectileData', {})
        if _pdat.get('spawnCharacterData'):
            _cnt = _pdat.get('spawnCharacterCount', 1)
            _cn = character_to_card.get(_pdat['spawnCharacterData'].get('name'),
                                        _pdat['spawnCharacterData'].get('name'))
            self.projectile_data.spawn_characters = (_cnt, _cn)
        # —— M2：Log 类「落地变滚木」链（gamedata projectileData.spawnProjectileData）——
        if _pdat.get('spawnProjectileData'):
            self.projectile_data.spawn_projectile = _pdat['spawnProjectileData']['name']
        self.projectile_waves = self.data.get('projectileWaves', 1)
        self.wave_interval = self.data.get("projectileWaveInterval", 0) / 1000

        self.charge_damage = self.data['summonCharacterData'].get('damageSpecial', 0)
        self.shield_health = self.data['summonCharacterData'].get('shieldHitpoints', 0)

        # M2 修复：lifeTime 单位为毫秒，此前未除 1000（Cannon 30s 寿命被当成 30000s → 永不衰减）
        _lt = self.data['summonCharacterData'].get('lifeTime')
        self.lifetime = _lt / 1000 if _lt else float('inf')

        self.death_spawn_data = self.data['summonCharacterData'].get('deathSpawnCharacterData', {})
        self.death_area_effect = self.data['summonCharacterData'].get('deathAreaEffectData', {})
        self.death_damage = self.data['summonCharacterData'].get('deathDamage', 0)

        self.jump_height = self.data['summonCharacterData'].get('jumpHeight', 0)
        self.jump_speed = self.data['summonCharacterData'].get('jumpSpeed', 0) / 60

        self.spawn_data = self.data['summonCharacterData'].get("spawnAreaObjectData", {})
        self.kamikaze = self.data['summonCharacterData'].get('kamikaze', False)

        self.tower_damage_mult = 1+self.data['summonCharacterData'].get('crownTowerDamagePercent', 0)/100

        self.type = self.data.get('tidType', '').split('_')[-1].lower()
        self.rarity = self.data.get('rarity', 'Common')

        # —— M1 递增伤害（地狱飞龙/地狱塔/同类）：官方数值表 variable_damage_* 字段 ——
        _scd_name = self.data['summonCharacterData'].get('name', '')
        _stats_row = characters.get(_scd_name) or buildings.get(_scd_name) or {}
        _t1, _t2 = _stats_row.get('variable_damage_time1'), _stats_row.get('variable_damage_time2')
        _v2, _v3 = _stats_row.get('variable_damage2'), _stats_row.get('variable_damage3')
        # 假设（置信度：中，待 L4 对拍）：time 为每段持续毫秒（累计和=切换阈值）；value 为百分比增量
        _th, _acc = [], 0.0
        for _t in (_t1, _t2):
            if _t:
                _acc += _t / 1000.0
                _th.append(_acc)
        self.ramp_stage_times = _th  # 累计切换阈值 [秒]
        self.ramp_values_raw = [x for x in (_v2, _v3) if x is not None]   # 基准级(9级)绝对阶段伤害
        self._ramp_stats_row = _stats_row
        self.ramp_stage_damages = []  # set_level 内按当前级缩放填充
        # time1=0 的卡（如 Monk）不是常规蓄力递增，排除
        self.has_ramp = bool(self.ramp_stage_times and self.ramp_stage_times[0] > 0 and self.ramp_values_raw)

        # —— M2 族3：建筑最小射程（官方数值表 minimum_range，Mortar=3.5 / GoblinCannon=3.5 / BarbarianLauncher=3.5）——
        self.min_range = (_stats_row.get('minimum_range') or 0) / 1000 if _stats_row else 0

        # —— M2 族5：渔夫特殊攻击（钩拉）字段 ——
        self.special_min_range = self.data['summonCharacterData'].get('specialMinRange', 0) / 1000
        self.special_range = self.data['summonCharacterData'].get('specialRange', 0) / 1000
        self.special_load_time = self.data['summonCharacterData'].get('specialLoadTime', 0) / 1000

        # —— M3 族6：英雄能力（abilityData 体系，=「精英卡」定义）——
        self.ability = self.data['summonCharacterData'].get('abilityData')  # None = 无能力

        # —— M4 族7：觉醒形态（evolvedSpellsData）——
        self.evo_raw = self.data.get('evolvedSpellsData')

        # —— M4.5 动作链：攻击序列（attackSequenceList）——
        # 数据形态（gamedata 快照已含）：
        #   InfernoDragon_EV1: attackSequence=[0,1,2,3] + attackSequenceMode=Manual +
        #                      attackSequenceList=[{damage:14},{47},{165},{330}]（跨攻击递增，逐档推进）
        #   Berserker:        attackSequenceList=[{damage:40},{40},{40}]（单次攻击多段命中）
        _aseq = self.data['summonCharacterData'].get('attackSequenceList')
        self.attack_seq_raw = _aseq
        self.attack_seq = self.data['summonCharacterData'].get('attackSequence')
        self.attack_seq_mode = self.data['summonCharacterData'].get('attackSequenceMode')
        self.attack_seq_damages = []  # set_level 内按当前级缩放填充
        # Berserker 类：基础伤害缺失（damage=0，全部经攻击序列结算）——以序列首档伤害作为缩放基准
        if not self.damage and _aseq:
            _d0 = next((s.get('damage') for s in _aseq if s.get('damage')), 0)
            if _d0:
                self.damage = _d0
                self._base_damage0 = _d0

        self.set_level(self.level)

    def set_level(self, level):
        li_card = _rarity_level_index(self.rarity, level)

        if self.projectiles:
            projectile_name = self.data['summonCharacterData']['projectileData']['name']
            _prow = projectiles.get(projectile_name) or {}
            _dpl = _prow.get('damage_per_level')
            if _dpl:
                # 弹丸行自带稀有度轴（与卡片稀有度可能不同，如派生/共享弹丸）
                self.projectile_data.damage = _value_at_level(_dpl, _prow.get('rarity') or self.rarity,
                                                              level, _prow.get('damage') or 0)
            elif _prow.get('damage') is not None:
                self.projectile_data.damage = _prow.get('damage')
            # damage_per_level 与 damage 均缺失时保持 0（如烟花射手：伤害全部经爆裂弹结算）

        if self.type == 'troop':
            building_name = self.data['summonCharacterData']['name']
            if building_name in buildings:
                _row = buildings[building_name]
                self.hp = _value_at_level(_row.get('hitpoints_per_level') or [],
                                          _row.get('rarity') or self.rarity, level, self._base_hp0)

        if self.type == 'character':
            character_name = self.data['summonCharacterData']['name']
            if character_name in characters:
                _row = characters[character_name]
                _rar = _row.get('rarity') or self.rarity  # 以数值表行稀有度为准（VoodooHog 等派生卡）
                self.hp = _value_at_level(_row.get('hitpoints_per_level') or [], _rar, level, self._base_hp0)
                if self.damage:
                    self.damage = _value_at_level(_row.get('damage_per_level') or [], _rar, level, self._base_damage0)
                    self.stats_source = 'official_table'
            else:
                # 快照数值表缺行（LittlePrince/Berserker/GoblinMachine/BossBandit 等 9 张）：
                # 按全稀有度统一曲线推导（实测 947+192+208+240+40 个相邻级比值 avg≈1.0985，
                # lv1→lv11 ≈ ×2.556，对所有稀有度含 Champion 一致；gamedata base=lv1）。
                # 置信度：中——存在 1~2% 舍入偏差（对照 Knight 690→1766 实测 2.559）
                _step = 1.1 ** (level - 1)
                self.hp = round(self._base_hp0 * _step)
                if self.damage:
                    self.damage = round(self._base_damage0 * _step)
                self.stats_source = 'derived_curve'
        elif self.type == 'building':
            # M2 修复：建筑卡此前完全没走 per-level 缩放（InfernoTower 17 vs 官方 51）
            building_name = self.data['summonCharacterData']['name']
            if building_name in buildings:
                _row = buildings[building_name]
                _rar = _row.get('rarity') or self.rarity
                self.hp = _value_at_level(_row.get('hitpoints_per_level') or [], _rar, level, self._base_hp0)
                if self.damage:
                    self.damage = _value_at_level(_row.get('damage_per_level') or [], _rar, level, self._base_damage0)
                self.stats_source = 'official_table'
            else:
                self.stats_source = 'gamedata_base'
        elif self.type == 'spell':
            # For simplicity, just assume that spells are projectiles, which is already handled
            pass

        # M1 递增伤害：阶段绝对伤害按当前级缩放（对照 Fandom 地狱龙 35/120/422 验证）
        # 注意必须在 damage 赋值之后执行
        if self.has_ramp:
            _row = self._ramp_stats_row or {}
            _dpl = _row.get('damage_per_level') or []
            _base0 = _dpl[0] if _dpl and _dpl[0] else 0
            _scale = (self.damage / _base0) if _base0 else 1.0
            self.ramp_stage_damages = [x * _scale for x in self.ramp_values_raw]

        # M4.5 动作链：攻击序列各档伤害按当前级等比缩放（基准 = 基础卡 lv1 伤害）
        if getattr(self, 'attack_seq_raw', None):
            _b0 = getattr(self, '_base_damage0', 0) or 1
            _scale = (self.damage / _b0) if _b0 else 1.0
            self.attack_seq_damages = []
            for _st in self.attack_seq_raw:
                _d = _st.get('damage')
                self.attack_seq_damages.append((_d * _scale) if _d else None)

        return level

class Projectile:
    def __init__(self, projectile_data):
        self.data = projectile_data
        self.damage = self.data.get('damage', 0)
        self.speed = self.data.get('speed', 0) / 60
        self.radius = (self.data.get('spawnProjectileData', {}).get('radius', 0) or self.data.get('radius', 0)) / 1000
        self.target_buff = self.data.get('targetBuffData', {})
        self.buff_time = self.data.get('buffTime', 0) / 1000
        # 女巫妈妈诅咒：buff 带 deathSpawnData → 目标死亡时生成物（VoodooHog）
        self.target_buff_death_spawn = self.target_buff.get('deathSpawnData')
        self.hits_air = 'AIR' in self.data.get('tidTarget', '')
        self.hits_ground = 'GROUND' in self.data.get('tidTarget', '') or 'BUILDING' in self.data.get('tidTarget', '')
        self.pushback = self.data.get('pushback', 0) / 1000
        self.name = self.data.get('name', 'Unknown')
        self.roll_range = self.data.get('projectileRange', 0) / 1000
        self.crown_tower_percent = (self.data.get("crownTowerDamagePercent", 0) + 100)/100
        # —— M1 弹道生成链 ——
        self.spawn_projectile = None    # 命中后生成的二段弹名（数值表 spawn_projectile）
        self.spawn_characters = None    # 命中后生成的部队 (数量, 卡名)（gamedata spawnCharacterData）
        if self.data.get('name') == 'TowerPrincessProjectile':
            self.hits_air = True
            self.hits_ground = True


def projectile_from_row(row, level=11):
    """M1 弹道生成链：从 cards_stats_projectile 数值表行构建二段弹包装。
    行格式与 gamedata projectileData 不同（snake_case + per_level 数组），单独适配。"""
    li = {'Common': 10, 'Rare': 8, 'Epic': 5, 'Legendary': 2, 'Champion': 0}.get(
        (row.get('rarity') or 'Common').lower().capitalize(), 10)
    w = Projectile(row)
    dpl = row.get('damage_per_level')
    if dpl and li < len(dpl):
        w.damage = dpl[li] or 0
    w.hits_air = bool(row.get('aoe_to_air', w.hits_air))
    w.hits_ground = bool(row.get('aoe_to_ground', w.hits_ground))
    w.roll_range = (row.get('projectile_range') or 0) / 1000.0
    w.spawn_projectile = row.get('spawn_projectile')
    w.spawn_count = row.get('spawn_count', 1) or 1
    return w

class TimedExplosiveData:
    def __init__(self, death_spawn_data):
        self.data = death_spawn_data
        self.name = self.data['name']
        self.damage = self.data['deathDamage']
        self.deploy_time = self.data['deployTime'] / 1000
        self.collision_radius = self.data['collisionRadius'] / 1000
        self.range = 3.0
        self.crown_tower_damage_percent = self.data.get('crownTowerDamagePercent', 100) / 100

class AreaEffectData:
    def __init__(self, source_card_name):
        # This only works for lumberjack, will modify later.
        self.data = Card(source_card_name)['summonCharacterData'].get('deathSpawnCharacterData', {}).get('deathAreaEffectData', {})
        self.duration = self.data.get('lifeDuration', 0) / 1000
        self.radius = self.data.get('radius', 0) / 1000
        self.buff_time = self.data.get('buffTime', 0)
        self.buff_data = self.data.get('buffData', {})
        self.speed_multiplier = self.buff_data.get('speedMultiplier')
        self.damage = self.data.get('spawnAreaEffectObjectData', {}).get('damage', 0)
        self.crown_tower_damage_percent = self.buff_data.get('crown', 0) or self.data.get('crownTowerDamagePercent', 0)

if __name__ == '__main__':
    deck = ['Knight', 'MiniPekka', 'Arrows', 'Minions', 'Musketeer', 'Fireball', 'Giant', 'Archer']
    for each in deck:
        print(Card(each).type)
