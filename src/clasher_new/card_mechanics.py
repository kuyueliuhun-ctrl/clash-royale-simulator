import math

from core import BasicCharacter, Position
from card_utils import Card, level_scale, _rarity_level_index
from arena import TileGrid
from evolutions import OFFICIAL_OVERRIDES

class Ghost(BasicCharacter):
    """皇家幽灵：脱战隐身（gamedata buffWhenNotAttackingData=Invisibility, 1800ms）。
    隐身 = 不可被选取（targetable=False）但可被范围伤害/法术命中（不无敌, 官方语义）；
    首次攻击显形, 脱战（不攻击）到时限后再次隐身。
    M6 觉醒：每次由隐身转显形瞬间召唤 2 名 Souldier（召唤伤害, 数据层 evo_2025_data.py）；
    觉醒隐身延迟 2.0s（Fandom History 2/3/2026：1.8s→2.0s）。
    （此前基础 Ghost 隐身引擎未建模, 本类补全基础+觉醒两条路径。）"""
    EVOLVED_INVIS_DELAY = 2.0  # [Fandom _page_4.txt History]

    def __init__(self, entity):
        super().__init__(entity)
        self.visible = False
        self._visible_timer = 0.0
        entity.targetable = False  # 部署即隐身（官方：spawn invisible, 攻击时显形）

    def on_tick(self, dt):
        super().on_tick(dt)
        if not self.visible: return
        self._visible_timer -= dt
        if self._visible_timer <= 0:
            self.visible = False
            self.entity.targetable = False

    def on_attack(self, current_target=None):
        e = self.entity
        was_visible = self.visible
        super().on_attack(current_target)
        if not self.visible:
            self.visible = True
            e.targetable = True
            # M6 ④：显形瞬间召唤 Souldier（觉醒形态；每次隐身→显形都触发）
            hooks = (getattr(e, 'evo', None) or {}).get('evo2025Hooks') or {}
            ss = hooks.get('souldierSummon')
            if ss and e.battle_state is not None:
                self._evo2025_summon_souldiers(ss)
        scd = e.data.data.get('summonCharacterData') or {}
        self._visible_timer = self.EVOLVED_INVIS_DELAY if e.evo else \
            (scd.get('buffWhenNotAttackingTime') or 1800) / 1000.0

    def _evo2025_summon_souldiers(self, ss):
        """显形召唤：count 名 Souldier 分立两侧, 各自对周围造成召唤伤害。"""
        from battle import Troop
        e = self.entity
        bs = e.battle_state
        li = _rarity_level_index(ss.get('spawnDamageRarity', 'Legendary'), e.level)
        arr = ss.get('spawnDamagePerLevel') or []
        dmg = arr[li] if 0 <= li < len(arr) else (arr[-1] if arr else 0)
        for i in range(int(ss.get('count', 2))):
            side = 0.7 if i % 2 == 0 else -0.7
            t = Troop(bs.next_entity_id, Position(e.position.x + side, e.position.y),
                      e.player, 'Souldier', bs)
            t.deploy_delay_remaining = t.data.deploy_time
            bs._spawn_entity(t)
            if dmg:
                # 召唤伤害：半径 [假设 1.0]（数据无字段）, 对空对地
                bs.deal_area_damage(e.player, t.position, ss.get('radius', 1.0), dmg, True, True)


class Witch(BasicCharacter):
    def __init__(self, entity):
        super().__init__(entity)
        self.next_spawn_remaining = 1.0
    def on_tick(self, dt):
        super().on_tick(dt)
        if not self.entity.is_alive: return
        if self.next_spawn_remaining > 0:
            self.next_spawn_remaining -= dt
            return
        # spawn skeletons!
        from battle import get_spawn_position, Troop
        skeleton = Card('Skeletons')
        skeleton.spawn_number = 4
        skeleton.spawn_radius = 2
        skeleton.spawn_delay = 0
        positions = get_spawn_position(skeleton, self.entity.position, self.entity.player, False)
        for each in positions:
            self.battle_state._spawn_entity(Troop(self.battle_state.next_entity_id, each, self.entity.player, 'Skeletons'))
        # —— M5 觉醒补全：出兵间隔走数据值（基础 Witch 7000ms；此前硬编码 7.0）——
        _evo = getattr(self.entity, 'evo', None) or {}
        self.next_spawn_remaining = (_evo.get('spawnPauseTime') or 7000) / 1000.0

class Balloon(BasicCharacter):
    def on_death(self):
        from battle import TimedExplosive
        bomb = TimedExplosive(self.battle_state.next_entity_id, self.entity.position, self.entity.player, self.entity.name)
        self.battle_state._spawn_entity(bomb)

class Golem(BasicCharacter):
    def on_death(self):
        from battle import Troop, Position
        self.battle_state = self.entity.battle_state

        positions = [Position(self.entity.position.x-0.5, self.entity.position.y),
                     Position(self.entity.position.x+0.5, self.entity.position.y)]
        for position in positions:
            self.battle_state._spawn_entity(Troop(self.battle_state.next_entity_id, position, self.entity.player, 'Golemite'))

class LavaHound(BasicCharacter):
    def on_death(self):
        from battle import get_spawn_position, Troop
        self.battle_state = self.entity.battle_state
        positions = get_spawn_position(Card('LavaPups'), self.entity.position, self.entity.player)
        for position in positions:
            self.battle_state._spawn_entity(Troop(self.battle_state.next_entity_id, position, self.entity.player, 'LavaPups'))

class Prince(BasicCharacter):
    """Implements charging abilities."""
    def __init__(self, entity):
        super().__init__(entity)
        self.starting_position = Position(self.entity.position.x, self.entity.position.y)
        self.charging = False

    def on_tick(self, dt):
        super().on_tick(dt)
        distance = self.entity.position.distance_to(self.starting_position)
        if distance > self.entity.data.charge_range and not self.charging:
            self.charging = True
            self.entity.speed *= 2
            # —— M4 族7：觉醒冲锋羊出场冲锋推击（onStartChargingActionData）——
            # M5 修复：e 引用先于赋值（原代码 NameError，觉醒冲锋羊触发即崩）
            e = self.entity
            evo = getattr(e, 'evo', None)
            oscd = (evo or {}).get('onStartChargingActionData') if evo else None
            if oscd:
                strength = oscd.get('pushBackStrength', 2500) / 1000
                dmg = oscd.get("pushBackDamage", 83) * level_scale(e.level)
                e.battle_state.deal_area_damage(e.player, e.position, 2.0, dmg, False, True)
                e.battle_state.push_enemies(e.player, e.position, 2.0, strength)
        if self.charging: self.entity.attack_cooldown = 0

    def on_attack(self, current_target=None):
        if not self.charging:
            current_target.take_damage(self.entity.data.damage)
            self.starting_position = Position(self.entity.position.x, self.entity.position.y)
        else:
            _cd = self.entity.data.charge_damage or self.entity.data.damage
            _b0 = getattr(self.entity.data, '_base_damage0', 0) or self.entity.data.damage
            _scale = (self.entity.data.damage / _b0) if _b0 else 1.0   # 勘误批10：冲锋伤害随级缩放（原恒 lv1 基准）
            current_target.take_damage(_cd * _scale)
            self.charging = False
            self.starting_position = Position(self.entity.position.x, self.entity.position.y)
            self.entity.speed = self.entity.data.speed
        self.entity.attack_cooldown = self.entity.data.hit_speed
        if self.entity.data.kamikaze:
            self.entity.is_alive = False
            self.on_death()

class DarkPrince(Prince):
    pass

class BattleRam(Prince):
    def on_death(self):
        from battle import get_spawn_position, Troop
        positions = get_spawn_position(Card('Barbarian'), self.entity.position, self.entity.player)
        for position in positions:
            self.battle_state._spawn_entity(Troop(self.battle_state.next_entity_id, position, self.entity.player, 'Barbarian'))


class GiantSkeleton(BasicCharacter):
    def __init__(self, entity):
        super().__init__(entity)
    def on_death(self):
        from battle import TimedExplosive
        bomb = TimedExplosive(self.battle_state.next_entity_id, self.entity.position, self.entity.player,
                              self.entity.name)
        self.battle_state._spawn_entity(bomb)

class IceWizard(BasicCharacter):
    """落地冰雾（gamedata spawnAreaObjectData=IceWizardCold / FL AEO.IceWizardCold）：
    伤害 33×等级（Common 轴 lv11=84）、半径 3.0、对空对地、对塔 0%（crownTowerDamagePercent=-100）、
    减速 2.5s（FL BUFF.IceWizardCold：speedMultiplier=-35 → 快照 -35 → 速度 0.65×；
    FL 逻辑层另带 HitSpeedMultiplier=-35 —— buffData.hitSpeedMultiplier 消费为攻速减速）。
    此前口径错误（1.0s 无攻速减速）已按 FL 修正；塔不吃伤害也不吃减速。"""
    def on_spawn(self):
        from battle import Building, Troop
        spawn_data = self.entity.data.spawn_data
        e = self.entity
        bs = e.battle_state
        radius = spawn_data['radius'] / 1000
        bf = spawn_data.get('buffData') or {}
        slow = 1.0 + (bf.get('speedMultiplier') or 0) / 100.0
        slow_hs = 1.0 + (bf.get('hitSpeedMultiplier') or 0) / 100.0
        duration = (bf.get('buffTime') or spawn_data.get('buffTime') or 2500) / 1000.0
        dmg_mult = 1.0 + (spawn_data.get('crownTowerDamagePercent') or 0) / 100.0
        for entity in list(bs.entities.values()):
            if not entity.is_alive or entity.player == e.player: continue
            if not entity.position.distance_to(e.position) < radius + entity.data.collision_radius:
                continue
            is_tower = isinstance(entity, Building) and entity.id <= 6
            if not is_tower:
                entity.take_damage(spawn_data['damage'] * level_scale(e.level))
            if not isinstance(entity, Building):
                entity.apply_buff(speed_mult=slow, hit_speed_mult=slow_hs, duration=duration)

class Miner(BasicCharacter):
    def __init__(self, entity):
        super().__init__(entity)
        self.distance = self.entity.position.distance_to(TileGrid.RED_KING_TOWER if self.entity.player == 1 else TileGrid.BLUE_KING_TOWER)
        self.freeze_time = self.distance/(650/60)
        self.entity.targetable = False
        self.entity.invincible = True

    def on_tick(self, dt):
        if self.freeze_time > 0:
            self.freeze_time -= dt
            self.entity.deploy_delay_remaining = self.entity.data.deploy_time
        else:
            self.entity.targetable = True
            self.entity.invincible = False

class Rage(BasicCharacter):
    def __init__(self, entity):
        super().__init__(entity)
        self.deploy_delay_remaining = entity.data.deploy_time
        self.data = entity.data.death_area_effect
        self.lifetime = self.data['lifeDuration']/1000

        self.radius = self.data['radius']/1000
        self.hit_speed = self.data['hitSpeed']/1000
        self.buff_time = self.data['buffTime']/1000
        self.speed_multiplier = self.data['buffData']['hitSpeedMultiplier']/100
        self.damage = self.data['spawnAreaEffectObjectData']['damage']
        self.crown_percent = self.data['spawnAreaEffectObjectData']['crownTowerDamagePercent']/100 + 1
        self.attack_cooldown = 0
        self.entity.battle_state.deal_area_damage(self.entity.player, self.entity.position, self.radius, self.damage,
                                                  True, True,
                                                  self.crown_percent)

    def on_tick(self, dt):
        super().on_tick(dt)
        from battle import Troop, Building, Projectile
        # print(self.attack_cooldown)
        if self.deploy_delay_remaining > 0:
            self.deploy_delay_remaining -= dt
            return
        if self.lifetime <= 0:
            self.entity.is_alive = False
            return
        else:
            self.lifetime -= dt
        if self.attack_cooldown <= 0:
            for entity in list(self.entity.battle_state.entities.values()):
                if not entity.is_alive or entity.player != self.entity.player: continue
                if entity.position.distance_to(self.entity.position) > self.radius + entity.data.collision_radius: continue
                if isinstance(entity, Troop) or isinstance(entity, Building):
                    entity.speed_buff = max(entity.speed_buff, self.speed_multiplier)
                    entity.buff_time_remaining = max(entity.buff_time_remaining, self.buff_time)

            self.attack_cooldown = self.hit_speed
            pass
        else:
            self.attack_cooldown -= dt

class RageBarbarian(BasicCharacter):
    def on_death(self):
        from battle import Entity
        self.battle_state._spawn_entity(Entity(self.battle_state.next_entity_id, self.entity.position, self.entity.player, "Rage", self.battle_state))


# ==================== M2 族5：渔夫钩拉 ====================

class Fisherman(BasicCharacter):
    """钩子 = 唯一攻击方式：3.5~7 格内地面目标，蓄力 1.3s 抛钩（弹速 800）。
    命中：拉向自己（内部值 8.5 格/s，官方无文字数值）+ 钩伤。
    减速已被官方移除（2026/4/6 平衡，见 docs/数值规则查证汇总.md）。近身 (<3.5) 无攻击手段（官方弱点）。"""
    def __init__(self, entity):
        super().__init__(entity)
        self.special_cd = 0.0

    def on_attack(self, current_target=None):
        # 普攻无伤害——钩子是唯一输出（近身无攻击手段是官方已知弱点）
        self.entity.attack_cooldown = self.entity.data.hit_speed

    def on_tick(self, dt):
        super().on_tick(dt)
        e = self.entity
        self.special_cd = max(0.0, self.special_cd - dt)
        if self.special_cd > 0 or e.deploy_delay_remaining > 0: return
        t = e.battle_state.entities.get(e.target_id) if e.target_id else None
        from battle import Building
        if t is None or not t.is_alive or t.data.is_air_unit: return
        # —— 勘误批4：钩建筑 → 把自己拉向建筑（官方机制, 原跳过）——
        if isinstance(t, Building):
            d = e.position.distance_to(t.position)
            min_r = getattr(e.data, 'special_min_range', 0) or 3.5
            max_r = getattr(e.data, 'special_range', 0) or 7.0
            if min_r <= d <= max_r:
                e.hook_pull = {'x': t.position.x, 'y': t.position.y,
                               'speed': OFFICIAL_OVERRIDES.get('Fisherman', {}).get('pull_speed', 8.5),
                               'time': 2.0}
                self.special_cd = getattr(e.data, 'special_load_time', 0) or 1.3
            return
        if not hasattr(t, 'hook_pull'): return
        d = e.position.distance_to(t.position)
        min_r = getattr(e.data, 'special_min_range', 0) or 3.5
        max_r = getattr(e.data, 'special_range', 0) or 7.0
        if d < min_r or d > max_r: return
        ov = OFFICIAL_OVERRIDES.get('Fisherman', {})
        t.hook_pull = {'x': e.position.x, 'y': e.position.y,
                       'speed': ov.get('pull_speed', 8.5), 'time': 2.0}
        hook_dmg = getattr(e.data, 'special_damage', 0) or e.data.damage
        if hook_dmg: t.take_damage(hook_dmg, delayed=True)
        self.special_cd = getattr(e.data, 'special_load_time', 0) or 1.3


# ==================== M3 族6：英雄能力（abilityData 体系 =「精英卡」）====================

class _HeroBase(BasicCharacter):
    def use_ability(self): return True


class SkeletonKing(_HeroBase):
    """灵魂召唤：耗蓝 2 / 冷却 20s。数量 = 6(基础) + 灵魂数(场上任意部队死亡+1, 上限10)，上限 16。
    以墓园形式 0.25s/只 持续放出（gamedata: lifeDuration 10s, spawnInterval 250ms, 半径 4→官方 3.5）。"""
    def __init__(self, entity):
        super().__init__(entity)
        self.remaining = 0
        self.timer = 0.0
        self.index = 0

    def use_ability(self):
        souls = self.battle_state.souls[self.entity.player] if hasattr(self.battle_state, 'souls') else 0
        self.remaining = min(6 + souls, 16)
        if hasattr(self.battle_state, 'souls'): self.battle_state.souls[self.entity.player] = 0
        self.timer = 0.9  # 施法前摇 0.933s（官方）
        return True

    def on_tick(self, dt):
        super().on_tick(dt)
        if self.remaining <= 0: return
        if not self.entity.is_alive: return  # 简化：本体死亡即停止（官方：墓园继续刷完，待 L4）
        self.timer -= dt
        if self.timer > 0: return
        self.timer = 0.25
        self.remaining -= 1
        import math
        radius = OFFICIAL_OVERRIDES.get('SkeletonKing', {}).get('spawn_radius', 3.5)
        ang = self.index * 2.399963
        self.index += 1
        rad = 0.4 + 0.6 * ((self.index * 7) % 10) / 9.0 * radius
        pos = Position(self.entity.position.x + math.cos(ang) * rad,
                       self.entity.position.y + math.sin(ang) * rad)
        from battle import Troop
        t = Troop(self.battle_state.next_entity_id, pos, self.entity.player, 'SkeletonKingSkeleton')
        self.battle_state._spawn_entity(t)


class ArcherQueen(_HeroBase):
    """隐身斗篷：耗蓝 1 / 冷却 17s。3.5s 不可被选取（非无敌，仍吃法术/AOE），
    攻速 +180%（×2.8）、移速 -25%。"""
    def use_ability(self):
        e = self.entity
        e.targetable = False
        e.cloak_time = 3.5
        e.apply_buff(hit_speed_mult=2.8, duration=3.5)
        e.apply_buff(speed_mult=0.75, duration=3.5)
        return True

    def on_tick(self, dt):
        super().on_tick(dt)
        t = getattr(self.entity, 'cloak_time', 0)
        if t > 0:
            t -= dt
            self.entity.cloak_time = t
            if t <= 0: self.entity.targetable = True


class GoldenKnight(_HeroBase):
    """连环突进：耗蓝 1 / 冷却 12s（官方；gamedata 8s 冲突，按官方）。
    最多 10 段、每段 5.5 格内最近未突进目标，突进伤害 ≈340（gamedata dashDamage 131 为起始级 ×1.1^10，wiki 335），
    突进期间无敌；命中公主塔即停。"""
    def use_ability(self):
        e = self.entity
        e.dash_remaining = 10
        e.dashed_ids = set()
        e.invincible = True
        return True

    def on_tick(self, dt):
        super().on_tick(dt)
        e = self.entity
        if getattr(e, 'dash_remaining', 0) <= 0:
            if getattr(e, 'dashing', False):
                e.dashing = False
                e.invincible = False
            return
        e.dashing = True
        from battle import Troop, Building
        best, best_d = None, 5.5
        for ent in list(e.battle_state.entities.values()):
            if not ent.is_alive or ent.player == e.player: continue
            if not isinstance(ent, (Troop, Building)) or not ent.targetable: continue
            if ent.id in e.dashed_ids: continue
            d = e.position.distance_to(ent.position)
            if d < best_d: best, best_d = ent, d
        if best is None:
            e.dash_remaining = 0
            e.dashing = False
            e.invincible = False
            return
        e.dashed_ids.add(best.id)
        e.dash_remaining -= 1
        e.position = Position(best.position.x + 0.4, best.position.y)
        best.take_damage(131 * level_scale(e.level))
        if 'PrincessTower' in best.name:
            e.dash_remaining = 0


class Monk(_HeroBase):
    """禅定护持：耗蓝 1 / 冷却 17s。4s 内减伤 65%（官方；gamedata 80% 冲突，按官方），
    投射物反弹给发射者、法术反弹至最近敌方公主塔（battle.Projectile 侧实现），免击退/拉扯（未建模，待 L4）。"""
    def use_ability(self):
        e = self.entity
        e.deflect_active = True
        e.deflect_time = 4.0
        e.apply_buff(damage_reduction=OFFICIAL_OVERRIDES.get('Monk', {}).get('damage_reduction', 0.65), duration=4.0)
        return True

    def on_tick(self, dt):
        super().on_tick(dt)
        t = getattr(self.entity, 'deflect_time', 0)
        if t > 0:
            t -= dt
            self.entity.deflect_time = t
            if t <= 0: self.entity.deflect_active = False


class MightyMiner(_HeroBase):
    """爆破脱身：耗蓝 1 / 冷却 13s。钻地瞬移到镜像换路位置（期间不可选取），
    原地留下炸弹 1s 后爆炸（lv11 ≈332 伤害 / 半径 2 / 击退 1.8）。"""
    def use_ability(self):
        e = self.entity
        e.targetable = False
        e.invincible = True
        e.drill_time = 0.6  # 钻地过渡（简化）
        from battle import GenericBomb
        bomb = GenericBomb(e.battle_state.next_entity_id, Position(e.position.x, e.position.y),
                           e.player, damage=130 * level_scale(e.level), radius=2.0, delay=1.0, knockback=1.8)
        e.battle_state._spawn_entity(bomb)
        e.position = Position(18.0 - e.position.x, e.position.y)
        return True

    def on_tick(self, dt):
        super().on_tick(dt)
        t = getattr(self.entity, 'drill_time', 0)
        if t > 0:
            t -= dt
            self.entity.drill_time = t
            if t <= 0:
                self.entity.targetable = True
                self.entity.invincible = False


class LittlePrince(_HeroBase):
    """皇家救援：耗蓝 3 / 冷却 30s。召唤守护者 Guardienne（lv11 ≈1621 血 / 205 伤）冲锋入场，
    对沿途地面敌人造成 ≈233 伤害并击退最多 2 格；守护者留场至被击杀。"""
    def use_ability(self):
        e = self.entity
        from battle import Troop
        gy = -0.6 if e.player == 0 else 0.6
        guard = Troop(e.battle_state.next_entity_id, Position(e.position.x, e.position.y + gy),
                      e.player, 'ChampionGuard')
        e.battle_state._spawn_entity(guard)
        dmg = 90 * level_scale(e.level)   # pushBackDamage 90（起始级）→ lv16 ≈370
        e.battle_state.deal_area_damage(e.player, guard.position, 1.5, dmg, False, True)
        e.battle_state.push_enemies(e.player, guard.position, 1.5, 2.0)
        return True


class BossBandit(_HeroBase):
    """金蝉脱壳手雷：耗蓝 1 / 每局限 2 次（间隔 3s）。隐身 1s 并向身后传送 6 格。
    被动冲刺（3.5~6 格 / 0.8s 蓄力 / 双倍伤害 / 期间无敌）在 on_attack 中实现。"""
    def use_ability(self):
        e = self.entity
        if e.ability_uses >= 2: return False
        e.targetable = False
        e.grenade_time = 1.0
        dy = -6.0 if e.player == 0 else 6.0
        ny = max(0.5, min(31.5, e.position.y + dy))
        e.position = Position(e.position.x, ny)
        return True

    def on_tick(self, dt):
        super().on_tick(dt)
        t = getattr(self.entity, 'grenade_time', 0)
        if t > 0:
            t -= dt
            self.entity.grenade_time = t
            if t <= 0: self.entity.targetable = True

    def on_attack(self, current_target=None):
        e = self.entity
        # 被动冲刺（3.5~6 格触发段）：双倍伤害（官方：冲刺 = 2×普攻）；近身段按普攻
        d = e.position.distance_to(current_target.position)
        mult = 2.0 if 3.5 <= d <= 6.0 else 1.0
        current_target.take_damage(e.data.damage * mult, delayed=True)
        e.attack_cooldown = e.data.hit_speed


# ==================== M5 觉醒补全 ====================

class MegaKnight(BasicCharacter):
    """觉醒超级骑士（MegaKnight_EV1）。普通形态 evo=None → 行为与基础完全一致。
    · 落地溅射（卡面顶层 projectileData=MegaKnightAppear：伤168/半径2.2/击退1.0格/
      仅地面/对塔全额；FirstLight PROJECTILE.MegaKnightAppear 同值）——部署延迟
      （deployTime 1s=落地动画）结束时结算，GenericBomb 延时弹挂载。
      等级轴：projectiles 表 MegaKnightAppear 行 damage_per_level（Common 轴
      lv1=355...lv11=908，含快照轴整体缩放），与 TowerPrincessProjectile 同消费模式。
      这就是「刺客突进规避落地伤害」技巧的规避对象。
    · 冲刺跳（baseData：dashDamage 210 / dashMinRange 3500 / dashMaxRange 5000，lv1 基准）：
      目标在 3.5~5.0 格且不在近战范围时跳跃贴脸，落地对 areaDamageRadius(1.3) 内地面敌人
      造成 dashDamage×等级缩放；跳跃视为一次攻击（进入攻击冷却）。
      dashFilter=filter_mega_knight_jump_evolution → 仅地面目标（本卡本就只打地面）。
    · 上勾拳（onAttackActionData：pushBackStrength 4000）：攻击把周围敌人击退 4.0 格
      （口径与觉醒冲锋羊一致：/1000 → 格）。
    · doFollowUpJump=false → 无后续连跳；dashFollowUpMin/MaxRange 仅随数据存档。"""
    def __init__(self, entity):
        super().__init__(entity)
        self.dash_cd = 0.0
        self._spawn_bomb_laid = False
        # —— 冲刺跳滞空状态：起跳→空中直线位移→落地结算溅射。空中不无敌（仍被索敌/受击），
        # 只是位移改沿起跳点→落点直线（用户口径 2026-09-09：超骑起跳不无敌，空中仍然会被打）。——
        self._mk_jump = None      # {x0,y0,x1,y1,timer,dmg,dur,landed}
        self._mk_jump_landed = False

    def on_spawn(self):
        # 落地溅射：延迟 = 部署动画（deploy_delay_remaining 首帧扣减前即调用本钩子，
        # 直接用 data.deploy_time）；GenericBomb 首帧 update 仍会再等 delay 秒。
        e = self.entity
        if self._spawn_bomb_laid or e.battle_state is None: return
        self._spawn_bomb_laid = True
        from battle import GenericBomb
        from card_utils import projectiles, _value_at_level
        prj = projectiles.get('MegaKnightAppear') or {}
        dmg = _value_at_level(prj.get('damage_per_level') or [], prj.get('rarity') or 'Common',
                              e.level, prj.get('damage') or 168)
        bomb = GenericBomb(e.battle_state.next_entity_id, Position(e.position.x, e.position.y),
                           e.player, damage=dmg, radius=(prj.get('radius') or 2200) / 1000.0,
                           delay=max(e.data.deploy_time, 0.1), knockback=(prj.get('pushback') or 1000) / 1000.0,
                           hits_air=bool(prj.get('aoe_to_air', False)),
                           hits_ground=bool(prj.get('aoe_to_ground', True)))
        bomb.name = 'MegaKnightAppear'
        e.battle_state._spawn_entity(bomb)

    def on_tick(self, dt):
        super().on_tick(dt)
        e = self.entity
        self.dash_cd = max(0.0, self.dash_cd - dt)
        # —— 冲刺跳两阶段推进：起跳预备（原地蓄力, 不无敌可被打）→ 空中直线位移 → 落地溅射。
        # 总时长（预备+空中）= 1.7s（用户观察 2026-09-09：真实游戏超骑起跳到落地约 2 秒，
        # 引擎取 1.7s 预备+空中）。——
        if self._mk_jump is not None:
            j = self._mk_jump
            j['timer'] -= dt
            if j['phase'] == 'prep':
                # 起跳预备：原地蓄力（位置不动），计时结束进入空中
                if j['timer'] <= 0:
                    j['phase'] = 'air'
                    j['timer'] = j['air_dur']
            elif j['phase'] == 'air':
                if j['timer'] > 0:
                    t = 1.0 - j['timer'] / j['air_dur']
                    e.position.x = j['x0'] + (j['x1'] - j['x0']) * t
                    e.position.y = j['y0'] + (j['y1'] - j['y0']) * t
                else:
                    # 落地：落到目标落点 + 溅射
                    e.position.x, e.position.y = j['x1'], j['y1']
                    e.battle_state.deal_area_damage(e.player, Position(j['x1'], j['y1']),
                                                    e.data.area_damage_radius or 1.3,
                                                    j['dmg'], False, True)
                    e.attack_cooldown = e.data.hit_speed
                    self.dash_cd = e.data.hit_speed
                    self._mk_jump = None
                    self._mk_jump_landed = True
            return
        # —— 勘误：普通超骑也有冲刺跳（基础机制, 非觉醒专属）——
        # gamedata 卡面 summonCharacterData 自带 dashDamage/dashMinRange/dashMaxRange
        # （dashFilter=filter_mega_knight_jump_evolution → 仅地面目标）。
        # 觉醒形态可在 evo 里覆盖这些字段（dashFollowUpMin/MaxRange 等）。
        scd = getattr(e.data, 'data', {}) or {}
        scd = scd.get('summonCharacterData') or {} if isinstance(scd, dict) else {}
        evo = getattr(e, 'evo', None) or {}
        dash_dmg = evo.get('dashDamage') or scd.get('dashDamage')
        if not dash_dmg: return
        if self.dash_cd > 0 or e.deploy_delay_remaining > 0: return
        # —— 起跳目标：边缘距离（center − 双方碰撞半径）最近的地面敌人，不受索敌视距限制。
        # 真实游戏口径（2026-09-10 用户观察）：MK 对进入起跳距离的目标「放置即起跳」——
        # 例如桥头 MK 与刚部署的中场刺客，刺客还在部署动画/前摇（deploy_delay_remaining>0）
        # 且超出普通 5.5 格索敌视距时，MK 依然立刻起跳扑过去。
        # 实现：逐实体扫 中心距−MK碰撞−目标碰撞 最近者（与塔射程 battle.py edge 口径一致），
        # 跳过视距门槛（起跳距离是「跳得到」的物理判定，不依赖普通索敌）。——
        t = None
        best_edge = 1e9
        from battle import Building, Troop as _Troop
        for ent in list(e.battle_state.entities.values()):
            if not isinstance(ent, (_Troop, Building)): continue
            if not ent.is_alive or ent.player == e.player or not ent.targetable: continue
            if ent.data.is_air_unit: continue
            ed = e.position.distance_to(ent.position) - e.data.collision_radius \
                - getattr(ent.data, 'collision_radius', 0)
            if ed < best_edge:
                best_edge = ed
                t = ent
        if t is None or not t.is_alive or t.data.is_air_unit: return
        d = e.position.distance_to(t.position)
        if e.in_attack_range(t): return
        lo = (evo.get('dashMinRange') or scd.get('dashMinRange') or 3500) / 1000
        hi = (evo.get('dashMaxRange') or scd.get('dashMaxRange') or 5000) / 1000
        if not (lo <= best_edge <= hi): return
        # 冲刺跳：起跳预备（原地 0.5s 蓄力）→ 空中直线位移 → 落地溅射（areaDamageRadius 1.3）。
        # 总时长（预备+空中）= 1.7s（用户观察：真实超骑起跳到落地约 2 秒）。
        n = max(d, 0.1)
        jx = t.position.x + (e.position.x - t.position.x) / n * 0.6
        jy = t.position.y + (e.position.y - t.position.y) / n * 0.6
        prep = 0.5
        air_dur = max(1.7 - prep, 0.3)   # 空中 = 总 1.7s − 预备 0.5s = 1.2s
        self._mk_jump = {'x0': e.position.x, 'y0': e.position.y,
                         'x1': jx, 'y1': jy, 'dmg': dash_dmg * level_scale(e.level),
                         'phase': 'prep', 'prep_dur': prep,
                         'air_dur': air_dur, 'timer': prep}
        self._mk_jump_landed = False

    def on_attack(self, current_target=None):
        super().on_attack(current_target)
        evo = getattr(self.entity, 'evo', None) or {}
        if evo.get('pushBackStrength'):
            tiles = evo['pushBackStrength'] / 1000.0   # 4000 → 4.0 格（与冲锋羊同口径）
            self.battle_state.push_enemies(self.entity.player, self.entity.position,
                                           self.entity.data.area_damage_radius or 1.3, tiles)


class Musketeer(BasicCharacter):
    """觉醒火枪手（Musketeer_EV1）。attackSequenceList=[普射弹, 狙击弹(customRange 30000)]
    + attackSequenceMode=None → 按序列逐发交替：每第 2 发为狙击弹，可锁定 30 格内任意目标
    （数据未给狙击弹独立伤害 → 沿用普攻伤害，标注低置信度）。
    普通形态 evo=None → 行为与基础一致。"""
    def __init__(self, entity):
        super().__init__(entity)
        self.shot_index = 0

    def on_tick(self, dt):
        super().on_tick(dt)
        e = self.entity
        evo = getattr(e, 'evo', None) or {}
        seq = evo.get('attackSequenceList') or []
        if not any(st.get('customRange') for st in seq if isinstance(st, dict)):
            e._snipe_range_active = 0
            return
        # 下一发是否为狙击弹：按序列长度取模；是则临时扩展有效射程/索敌范围
        idx = self.shot_index % max(len(seq), 1)
        cur = seq[idx] if idx < len(seq) and isinstance(seq[idx], dict) else {}
        e._snipe_range_active = (cur.get('customRange') or 0) / 1000.0

    def on_attack(self, current_target=None):
        self.shot_index += 1
        self.entity._snipe_range_active = 0   # 攻击后恢复普射射程（下一 tick 按序列重设）
        super().on_attack(current_target)


# ==================== M7 数据接入：Ronin 格挡反击（机制 1, 被动）====================

class Ronin(BasicCharacter):
    """【M7】浪人格挡反击（被动）。数据：gamedata Ronin.summonCharacterData
    （parryReflectPercent=200 / parryCooldownMs=3500 / parryMeleeOnly=True, 引擎消费字段）。
    语义：被近战攻击命中时, 若格挡冷却完毕 → 本次伤害被格挡（不受伤）+ 反弹 200% 所受
    伤害给攻击者, 并进入 3.5s 冷却；远程攻击与冷却期间正常受伤。
    钩子路径：Entity.take_damage 受击钩子（entity_holder.on_take_damage, 返回 True=格挡短路）。
    近战判定：攻击者射程 ≤1.5 格（官方近战上限 Melee Medium 1.2）【口径待 L4 对拍】；
    空中近战单位免疫格挡（官方：Bats/Phoenix 免受格挡影响 [Fandom _fp_Ronin 策略节]）。
    【口径待 L4 对拍】官方口径里 Ronin 或有主动技能形态（gamedata abilityData.RoninParry）,
    当前按纯被动实现——部署即就绪（首次格挡无预热延迟, [假设]）。"""
    # 近战射程阈值：官方 Melee 最长 Medium=1.2, 1.5 容错覆盖圆整误差
    MELEE_RANGE_THRESHOLD = 1.5

    def __init__(self, entity):
        super().__init__(entity)
        scd = entity.data.data.get('summonCharacterData') or {}
        self.parry_percent = (scd.get('parryReflectPercent') or 200) / 100.0
        self.parry_cooldown = (scd.get('parryCooldownMs') or 3500) / 1000.0
        self.parry_melee_only = bool(scd.get('parryMeleeOnly', True))
        self.parry_timer = 0.0  # >0 = 冷却中

    def on_tick(self, dt):
        super().on_tick(dt)
        if self.parry_timer > 0:
            self.parry_timer = max(0.0, self.parry_timer - dt)

    def on_take_damage(self, amount, source):
        """受击钩子（Entity.take_damage 调用）。返回 True = 本次伤害被格挡（引擎短路）。"""
        e = self.entity
        if self.parry_timer > 0 or not e.is_alive:
            return False
        if not hasattr(source, 'data') or not hasattr(source, 'take_damage'):
            return False
        if source.player == e.player:
            return False
        if self.parry_melee_only:
            if (getattr(source.data, 'range', 99) or 0) > self.MELEE_RANGE_THRESHOLD:
                return False   # 远程攻击：正常受伤
            if getattr(source.data, 'is_air_unit', False):
                return False   # 空中近战（Bats/Phoenix）免疫格挡
        source.take_damage(amount * self.parry_percent, delayed=True, source=e)
        self.parry_timer = self.parry_cooldown
        return True

# ==================== 勘误批1（2026-09-04 用户口径校对）====================

class Assassin(BasicCharacter):
    """刺客突进（勘误批1, 用户口径 2026-09-04；gamedata dashDamage/dashMinRange/dashMaxRange/jumpSpeed）。
    - 部署与普通单位无异；
    - 与最近敌人的距离进入 [3.5, 6.0] 格时起跑突进（dashMinRange/dashMaxRange）；
    - 突进一旦开始不可被打断（眩晕/冰冻忽略, Troop.apply_buff 突进分支 + Troop.update 直调 dash_tick）；
    - 突进中若出现更近的敌人（如新部署单位）→ 改变突进目标；
    - 目标死亡 → 改突进最近的敌人（突进态保持）；
    - 抵达目标造成突进伤害 dashDamage（152@起始级 × 等级曲线, 高于普攻 76）并进入普攻冷却；
    - 冲刺过程无敌：官方无明确字段【待用户确认】→ 当前不无敌, 仅不可打断。
    位移速度 = jumpSpeed/60 ≈ 8.33 格/s（同跳河速度口径）。"""
    DASH_MIN = 3.5
    DASH_MAX = 6.0

    def __init__(self, entity):
        super().__init__(entity)
        entity._dash_active = False
        entity._dash_target_id = None
        entity._dash_invincible_timer = 0.0   # 冲刺无敌剩余时长（用户口径 2026-09-09：0.8s 全程无敌）

    def _nearest_enemy(self, exclude_id=None):
        e = self.entity
        best, best_d = None, float('inf')
        for o in e.battle_state.entities.values():
            if not o.is_alive or o.player == e.player or o.id == exclude_id:
                continue
            d = o.position.distance_to(e.position) - getattr(o.data, 'collision_radius', 0)
            if d < best_d:
                best, best_d = o, d
        return best, best_d

    def on_tick(self, dt):
        super().on_tick(dt)
        e = self.entity
        if not e.is_alive:
            return
        # 冲刺无敌 0.8s 计时：每帧推进（冲刺已结束后仍走完剩余无敌，对应「冲刺全程无敌」）
        if e._dash_invincible_timer > 0:
            e._dash_invincible_timer = max(0.0, e._dash_invincible_timer - dt)
            if e._dash_invincible_timer <= 0:
                e.invincible = False
        if e._dash_active:
            return
        if e.deploy_delay_remaining > 0:
            return
        # 未突进：检测触发窗口（正常攻击蓄力/移动不受影响, 由 Troop.update 常规逻辑处理）
        # 触发窗口用中心距离（与 MK 起跳距离同口径：dashMin/MaxRange 是「跳/冲得到」的
        # 物理判定，按中心点量）。若用边缘距离（−碰撞半径），MK 落地贴脸时窗口会被
        # 碰撞半径吃掉而打不开，突进永远触发不了（2026-09-10 用户技巧实测）。——
        tgt, _ = self._nearest_enemy()
        if tgt is not None:
            d_center = tgt.position.distance_to(e.position)
            if self.DASH_MIN <= d_center <= self.DASH_MAX:
                e._dash_active = True
                e._dash_target_id = tgt.id
                # —— 用户口径 2026-09-09：冲刺前摇即进入无敌状态，持续 0.8s（Boss Bandit 同款
                # Dash Time 0.8s / invulnerable while dashing）——
                e.invincible = True
                e._dash_invincible_timer = 0.8

    def dash_tick(self, dt):
        """突进推进（由 Troop.update 在常规索敌/移动之前直调, bypass 冰冻/眩晕与 A*）。"""
        e = self.entity
        bs = e.battle_state
        tgt = bs.entities.get(e._dash_target_id)
        if tgt is None or not tgt.is_alive:
            nxt, _ = self._nearest_enemy()
            if nxt is None:
                e._dash_active = False
                e._dash_target_id = None
                return
            e._dash_target_id = nxt.id
            tgt = nxt
        # 更近的敌人出现（如新部署单位）→ 改变突进目标（用户口径）
        nearer, nd = self._nearest_enemy(exclude_id=e._dash_target_id)
        cur_d = tgt.position.distance_to(e.position) - getattr(tgt.data, 'collision_radius', 0)
        if nearer is not None and nd < cur_d - 0.05:
            e._dash_target_id = nearer.id
            tgt = nearer
        dist = tgt.position.distance_to(e.position)
        reach = getattr(tgt.data, 'collision_radius', 0) + e.data.collision_radius
        if dist <= max(reach, 0.3):
            # 抵达：突进伤害（高于普攻）+ 普攻冷却（无敌由 0.8s 计时器走完, 不在此解除）
            dmg = (e.data.data.get('summonCharacterData') or {}).get('dashDamage', 0) * level_scale(e.level)
            tgt.take_damage(dmg, delayed=True, source=e)
            e.attack_cooldown = e.data.hit_speed
            e._dash_active = False
            e._dash_target_id = None
            return
        step = min(e.data.jump_speed * dt, dist - max(reach, 0.3) * 0.5)
        e.position.x += (tgt.position.x - e.position.x) / dist * step
        e.position.y += (tgt.position.y - e.position.y) / dist * step
        # 突进期间冷却照常恢复（不可攻击：抵达才结算）
        e.attack_cooldown = max(e.data.hit_speed - e.data.load_time, e.attack_cooldown - dt)


class BattleHealer(BasicCharacter):
    """战斗天使双治疗光环（勘误批1, 用户口径 2026-09-04）。
    - 部署时生成 2.5 格治疗光环; 每次攻击时生成 3.0 格治疗光环;
    - 所有治疗光环：寿命 1s, 每 0.25s 一跳共 4 跳（HealAuraZone）;
    - lv11：部署光环每跳 50 / 攻击光环每跳 25; 多等级：攻击 = ceil(部署/2);
    - 部署光环随等级 1.1/级曲线缩放【假设, 待对拍】;
    - 不治疗自己; 仅治疗友军部队（建筑是否受疗【待确认】, 当前不含）;
    - buffWhenNotAttackingTime 判定为旧版「脱战回血」定义残留, 不消费。"""
    DEPLOY_RADIUS = 2.5
    ATTACK_RADIUS = 3.0
    TICKS = 4
    INTERVAL = 0.25
    DEPLOY_HEAL_LV11 = 50

    def _deploy_heal(self):
        return self.DEPLOY_HEAL_LV11 * (1.1 ** (self.entity.level - 11))

    def _spawn_aura(self, radius, heal_per_tick):
        from battle import HealAuraZone
        bs = self.battle_state
        e = self.entity
        zone = HealAuraZone(bs.next_entity_id, Position(e.position.x, e.position.y),
                            e.player, radius, heal_per_tick,
                            ticks=self.TICKS, interval=self.INTERVAL,
                            exclude_id=e.id)
        zone.battle_state = bs
        bs.entities[zone.id] = zone
        bs.next_entity_id += 1

    def on_spawn(self):
        # 部署光环延迟到首次 on_tick 生成：on_spawn 在 Troop.__init__ 内运行,
        # 此刻 bs.next_entity_id 尚未被调用方递增（直接建实体会与本体 id 冲突被覆盖）。
        self._deploy_aura_pending = True

    def on_tick(self, dt):
        super().on_tick(dt)
        if getattr(self, '_deploy_aura_pending', False):
            self._deploy_aura_pending = False
            self._spawn_aura(self.DEPLOY_RADIUS, self._deploy_heal())

    def on_attack(self, current_target=None):
        super().on_attack(current_target)
        if self.entity.is_alive:
            self._spawn_aura(self.ATTACK_RADIUS, math.ceil(self._deploy_heal() / 2))



# T2-5：M8 段（Elite17，35 个定义 + HERO_CLASSES）已拆到 `card_mechanics_elite17`。
# 用 `import *` 聚合 ⇒ 本模块的**公开命名空间与拆分前逐名相同**（见 docs 的 65 名断言）；
# `battle.py:5` 的 `from card_mechanics import *` **一字未动**。
from card_mechanics_elite17 import *  # noqa: E402,F401,F403
