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


# ==================== M8：Elite17 精英卡（Hero 化）机制类 ====================
# 规格：docs/elite17_spec.md §1-17；数据层：elite17_data.py（HERO_ABILITIES 能力表，
# 费用/时长/半径等参数逐条来源标注）。装配链路见 battle.apply_hero_overlay。
# 激活模型：全部走冠军式 use_ability（圣水扣费 + 单次使用由 BattleState.use_ability
# M8 分支统一处理；holder.use_ability 只做效果，返回 False = 未生效→返还圣水）。
# Goblins（条件窗）与 BarbLog（法术二次滚）的按钮在实体死后仍可按——走
# BattleState.hero_windows 条件窗分支，效果体在下方模块级函数。

from elite17_data import hval as _elite_hval, HERO_ABILITIES as _HERO_ABILITIES


def _ab(e):
    """取实体能力参数包（elite17_data.HERO_ABILITIES 注入的 abilityData）。"""
    return getattr(e.data, 'ability', None) or {}


def _sv(triple, level):
    """(数值, 来源等级, 标注) 三元组 → 当前战斗等级数值（官方 1.1/级曲线）。"""
    return _elite_hval(triple[0], triple[1], level)


def open_goblin_window(bs, player, pos):
    """M8 ⑧：Goblins 旗窗——最后一只哥布林阵亡时开 5s 条件窗（窗口内按按钮出增援）。"""
    ab = _HERO_ABILITIES['Goblins']
    bs.hero_windows[player] = {
        'until': bs.time + ab['window'], 'used': False, 'card': 'Goblins',
        'pos': Position(pos.x, pos.y),
        'effect': _goblin_brigade_effect}


def _goblin_brigade_effect(bs):
    """M8 ⑧：Brigade Goblins 增援（x2，2026-08-04 由 x3 削；属性与本体相同）。"""
    for pid in (0, 1):
        w = bs.hero_windows.get(pid)
        if w and w.get('card') == 'Goblins' and w.get('used'):
            info = Card('Goblins')
            info.spawn_number = _HERO_ABILITIES['Goblins']['brigadeCount']
            info.spawn_delay = 0
            from battle import Troop as _T, get_spawn_position as _gsp
            for p in _gsp(info, w['pos'], pid, False):
                bs._spawn_entity(_T(bs.next_entity_id, p, pid, 'Goblins', bs))
            return


class HeroKnight(_HeroBase):
    """【M8 §1】Hero Knight — Triumphant Taunt（2 费，单次）：
    嘲讽 6.5 格内所有敌军部队+建筑攻击自己（5s，强制锁定走 Entity._hero_taunt_override），
    同时自身获得护盾（ShieldHitpoints 197 [Fandom L1]/内存 200 双记，5s 到期消失）。"""
    def __init__(self, entity):
        super().__init__(entity)
        self.shield_timer = 0.0

    def use_ability(self):
        e, ab = self.entity, _ab(self.entity)
        bs = e.battle_state
        from battle import Troop, Building
        for t in list(bs.entities.values()):
            if not t.is_alive or t.player == e.player: continue
            if not isinstance(t, (Troop, Building)): continue
            if t.position.distance_to(e.position) > ab['tauntRadius'] + t.data.collision_radius: continue
            t._taunt_until = bs.time + ab['tauntDuration']
            t._taunt_target_id = e.id
            t.path = []
        v, src, _ = ab['shieldValue']
        self.entity.shield_health = _elite_hval(v, src, self.entity.level)
        self.shield_timer = ab['shieldDuration']
        return True

    def on_tick(self, dt):
        super().on_tick(dt)
        if self.shield_timer > 0:
            self.shield_timer -= dt
            if self.shield_timer <= 0:
                self.entity.shield_health = 0   # 能力护盾到期（与护盾被打掉同一槽位）


class HeroMusketeer(_HeroBase):
    """【M8 §2】Heroic Musketeer — Trusty Turret（3 负，单次）：
    前方 3 格放置自动炮塔（MusketeerHeroTurret：HP717@L3/射程 4/寿命 10s 线性衰减
    [Fandom L3]），落地 AOE 95@L3 [Fandom TurretSpawnDmg]。"""
    def use_ability(self):
        e, ab = self.entity, _ab(self.entity)
        from battle import Building
        gy = -ab['frontOffset'] if e.player == 0 else ab['frontOffset']  # 前方=敌方方向
        pos = Position(e.position.x, e.position.y + gy)
        t = Building(e.battle_state.next_entity_id, pos, e.player, ab['turretCard'])
        e.battle_state._spawn_entity(t)
        e.battle_state.deal_area_damage(e.player, pos, ab['spawnRadius'],
                                        _sv(ab['spawnDamage'], e.level), True, True)
        return True


class HeroMiniPekka(_HeroBase):
    """【M8 §3】Heroic Mini P.E.K.K.A — Breakfast Boost（1 费，单次）：
    煎饼进度：22s 自动 1 格或每击 +10s 进度（≤3 格）；能力 = 吃煎饼升级
    （0/1/2/3 格 → +1/+2/+3/+5 级，等级走官方 1.1/级曲线 [口径假设：沿等级表跳级]）
    + 回复 30% maxHP [Fandom L17-21 表 2539..3717]。"""
    def __init__(self, entity):
        super().__init__(entity)
        self.progress = 0.0   # 煎饼进度（秒）
        self.meter = 0        # 已烹饪格数（0..3）

    def on_tick(self, dt):
        super().on_tick(dt)
        ab = _ab(self.entity)
        if self.meter < ab['maxMeter']:
            self.progress += dt
            if self.progress >= ab['meterSeconds']:
                self.meter += 1
                self.progress = 0.0

    def on_attack(self, current_target=None):
        super().on_attack(current_target)
        ab = _ab(self.entity)
        if self.meter < ab['maxMeter']:
            self.progress += ab['onHitProgress']
            while self.progress >= ab['meterSeconds'] and self.meter < ab['maxMeter']:
                self.meter += 1
                self.progress -= ab['meterSeconds']

    def use_ability(self):
        e, ab = self.entity, _ab(self.entity)
        steps = ab['levelsByMeter'][min(self.meter, 3)]
        mult = 1.1 ** steps   # 等级 = 沿等级表跳级（引擎口径：官方 1.1/级曲线）
        old_max, old_dmg = e.data.hp, e.data.damage
        e.data.hp = round(old_max * mult)
        e.data.damage = round(old_dmg * mult)
        # 升级补差 + 回复 30% maxHP（口径假设：两者叠加，封顶新上限）
        e.hp = min(e.data.hp, e.hp + (e.data.hp - old_max) + e.data.hp * ab['healPct'])
        self.meter, self.progress = 0, 0.0
        return True


class HeroValkyrie(_HeroBase):
    """【M8 §4】Hero Valkyrie — Wild Whirlwind（3 费，单次）：
    3.5s 旋风：0.25s/击 AOE 半径 2.5（AbilityDmg 97@L11，对皇冠塔 ×0.5）、减伤 15%、
    移速提升 [假设 ×1.2，数值缺]；结束位移冲刺 5.5 格；旋风后禁攻 [假设 1s，时长缺]。
    与觉醒（EV1 Tornado）同 Wild 槽互斥——引擎由 hero_slots 声明决定，不叠加。"""
    def __init__(self, entity):
        super().__init__(entity)
        self.phase = None      # None / 'whirl' / 'forbid'
        self.timer = 0.0
        self.tick_timer = 0.0

    def use_ability(self):
        e, ab = self.entity, _ab(self.entity)
        self.phase, self.timer, self.tick_timer = 'whirl', ab['whirlDuration'], 0.0
        e.apply_buff(speed_mult=ab['speedMult'], duration=ab['whirlDuration'])
        e.apply_buff(damage_reduction=ab['damageReduction'], duration=ab['whirlDuration'])
        return True

    def _dash(self):
        """旋风结束位移冲刺 5.5 格（朝敌方公主塔方向；地面单位受可走性约束）。"""
        e = self.entity
        from arena import TileGrid
        tgt = TileGrid.RED_KING_TOWER if e.player == 0 else TileGrid.BLUE_KING_TOWER
        dx, dy = tgt.x - e.position.x, tgt.y - e.position.y
        n = (dx * dx + dy * dy) ** 0.5 or 1.0
        ab = _ab(e)
        step = ab['dashRange'] / n
        for k in (1.0, 0.66, 0.33):   # 落点不可走则逐级收缩（河道/建筑）
            nx, ny = e.position.x + dx * step * k, e.position.y + dy * step * k
            if e.data.is_air_unit or e.battle_state.ground_walkable(Position(nx, ny), e.data.collision_radius):
                e.position = Position(nx, ny)
                e.path = []
                break

    def on_tick(self, dt):
        super().on_tick(dt)
        e, ab = self.entity, _ab(self.entity)
        if self.phase == 'whirl':
            self.timer -= dt
            e.attack_cooldown = max(e.attack_cooldown, 0.25)   # 旋风期间不进行普通攻击
            self.tick_timer -= dt
            if self.tick_timer <= 0:
                self.tick_timer = ab['tick']
                e.battle_state.deal_area_damage(
                    e.player, e.position, ab['radius'], _sv(ab['tickDamage'], e.level),
                    e.data.attack_air, e.data.attack_ground, ab['crownMult'])
            if self.timer <= 0:
                self._dash()
                self.phase, self.timer = 'forbid', ab['forbidAttack']
        elif self.phase == 'forbid':
            self.timer -= dt
            e.attack_cooldown = max(e.attack_cooldown, self.timer)  # 禁攻 buff
            if self.timer <= 0:
                self.phase = None

    def on_attack(self, current_target=None):
        if self.phase == 'whirl':   # 旋风期间普攻关闭（DisableMeleeAeoDamageEffect）
            self.entity.attack_cooldown = max(self.entity.attack_cooldown, 0.25)
            return
        super().on_attack(current_target)


class HeroWizard(_HeroBase):
    """【M8 §5】Hero Wizard — Fiery Flight（1 费；wiki 属性表 1 与信息框 2 矛盾双记取 1，单次）：
    1s 延迟后升空 5s（临时空中单位 + 移速 +50%），期间每次火球命中处生成
    半径 4 / 2s 火旋风（TornadoDmg 43@L11 独立伤害，对空对地）。
    [近似] 旋风落点以攻击时刻目标位置代替弹道落点，待 L4 对拍。"""
    def __init__(self, entity):
        super().__init__(entity)
        self.fly_pending = 0.0
        self.fly_timer = 0.0

    def use_ability(self):
        self.fly_pending = _ab(self.entity)['flyDelay']
        return True

    def _end_fly(self):
        e = self.entity
        e.data.is_air_unit = Card(e.card_name).is_air_unit   # 落地还原位面
        e.path = []
        self.fly_timer = 0.0

    def on_tick(self, dt):
        super().on_tick(dt)
        e, ab = self.entity, _ab(self.entity)
        if self.fly_pending > 0:
            self.fly_pending -= dt
            if self.fly_pending <= 0:
                e.data.is_air_unit = True
                e.apply_buff(speed_mult=ab['speedMult'], duration=ab['flyDuration'])
                e.path = []
                self.fly_timer = ab['flyDuration']
            return
        if self.fly_timer > 0:
            self.fly_timer -= dt
            if self.fly_timer <= 0:
                self._end_fly()

    def on_attack(self, current_target=None):
        super().on_attack(current_target)
        if self.fly_timer > 0 and current_target is not None:
            from battle import EvoEffectZone
            ab = _ab(self.entity)
            dps = _sv(ab['tornadoDps'], self.entity.level)
            zone = EvoEffectZone(self.entity.battle_state.next_entity_id,
                                 Position(current_target.position.x, current_target.position.y),
                                 self.entity.player, self.entity.battle_state,
                                 radius=ab['tornadoRadius'], lifetime=ab['tornadoDuration'],
                                 dps=dps, tick=0.5, level=1,  # dps 已按级换算 → level=1 不二次缩放
                                 label='WizardHero_MiniTornadoBuff')
            self.entity.battle_state._spawn_entity(zone)


class HeroBowler(_HeroBase):
    """【M8 §6】Hero Bowler — Stone Swish（2 费，单次）：
    2.5s 蓄力（禁攻）→ 7.3s 迫击炮模式：射程 11.5、共 3 发、攻速 1.9s、
    伤害 AbilityDmg 508@L11（对皇冠塔 ×0.5）；弹丸 7 格飞行段引擎简化为直接命中 [口径注]。
    Hero 主伤害（AreaDmg 289@L11）在弹丸上——overlay 同步覆写 projectile_data.damage。"""
    def __init__(self, entity):
        super().__init__(entity)
        self.phase = None    # None / 'charge' / 'siege'
        self.timer = 0.0
        self.shots_left = 0
        self._orig = None

    def _enter_siege(self):
        e, ab = self.entity, _ab(self.entity)
        self._orig = (e.data.range, e.data.sight_range, e.data.hit_speed,
                      e.data.tower_damage_mult, e.data.projectile_data.damage)
        e.data.range = ab['siegeRange']
        e.data.sight_range = max(e.data.sight_range, ab['siegeRange'])
        e.data.hit_speed = ab['siegeHitSpeed']
        e.data.tower_damage_mult = ab['crownMult']
        e.data.projectile_data.damage = _sv(ab['siegeDamage'], e.level)
        e.target_id = None
        e.path = []
        self.phase, self.timer, self.shots_left = 'siege', ab['siegeDuration'], ab['siegeShots']

    def _exit_siege(self):
        e = self.entity
        if self._orig is not None:
            (e.data.range, e.data.sight_range, e.data.hit_speed,
             e.data.tower_damage_mult, e.data.projectile_data.damage) = self._orig
        self.phase = None

    def use_ability(self):
        self.phase = 'charge'
        self.timer = _ab(self.entity)['chargeTime']
        return True

    def on_tick(self, dt):
        super().on_tick(dt)
        e, ab = self.entity, _ab(self.entity)
        if self.phase == 'charge':
            self.timer -= dt
            e.attack_cooldown = max(e.attack_cooldown, self.timer)  # 蓄力期间禁攻
            if self.timer <= 0:
                self._enter_siege()
        elif self.phase == 'siege':
            self.timer -= dt
            if self.timer <= 0 or self.shots_left <= 0:
                self._exit_siege()

    def on_attack(self, current_target=None):
        super().on_attack(current_target)
        if self.phase == 'siege':
            self.shots_left -= 1   # 3 发用尽立即还原（BowlerHeroDeactivateGroup）


class HeroGiant(_HeroBase):
    """【M8 §7】Hero Giant — Heroic Hurl（2 费，单次）：
    抓取 2 格内最高 HP 敌军部队（1 个，对空对地）水平扔出 9 格（朝本方进攻方向），
    落地 ImpactDmg 135@L11 + 眩晕 2s。
    [简化] 飞行中段不可被地面选取未建模（瞬时位移），待 L4。"""
    def use_ability(self):
        e, ab = self.entity, _ab(self.entity)
        bs = e.battle_state
        from battle import Troop
        best, best_hp = None, -1
        for t in list(bs.entities.values()):
            if not t.is_alive or t.player == e.player or not isinstance(t, Troop): continue
            if t.position.distance_to(e.position) > ab['grabRadius'] + t.data.collision_radius: continue
            hp = t.hp + t.shield_health
            if hp > best_hp:
                best, best_hp = t, hp
        if best is None:
            return False   # 无可抓目标 → 不扣费（use_ability 返还语义）
        from arena import TileGrid
        tgt = TileGrid.RED_KING_TOWER if e.player == 0 else TileGrid.BLUE_KING_TOWER
        dx, dy = tgt.x - e.position.x, tgt.y - e.position.y
        n = (dx * dx + dy * dy) ** 0.5 or 1.0
        step = ab['throwRange'] / n
        nx, ny = best.position.x + dx * step, best.position.y + dy * step
        if not best.data.is_air_unit:
            if not bs.ground_walkable(Position(nx, ny), best.data.collision_radius):
                k = 0.5    # 落点不可走 → 折半投掷距离（河道/边界兜底）
                nx, ny = best.position.x + dx * step * k, best.position.y + dy * step * k
        best.position = Position(nx, ny)
        best.path = []
        bs.deal_area_damage(e.player, best.position, ab['impactRadius'],
                            _sv(ab['impactDamage'], e.level), True, True)
        for t in list(bs.entities.values()):
            if not t.is_alive or t.player == e.player: continue
            if isinstance(t, (Troop,)) and t.position.distance_to(best.position) <= ab['impactRadius'] + t.data.collision_radius:
                t.apply_buff(stun=ab['stun'])
        return True


class HeroGoblins(_HeroBase):
    """【M8 §8】Hero Goblins — Banner Brigade（1 费，条件窗，单次）：
    部署按钮禁用；最后一只哥布林死亡时落旗开 5s 窗口（BattleState.hero_windows），
    窗口内按按钮 → 后方召出 2 只 Brigade Goblins（属性与本体相同）。
    Hero 数值≈普通表（78/48 vs 79/49，沿用普通表不覆写）。"""
    def __init__(self, entity):
        super().__init__(entity)
        e, bs = entity, entity.battle_state
        cur = bs._hero_goblin_group_state.get(e.player)
        # 2s 内同批部署视为同组（x4 一次出 4 只）
        if cur and bs.time - cur[1] < 2.0:
            self.group_id = cur[0]
            bs._hero_goblin_group_state[e.player] = (self.group_id, bs.time)
        else:
            bs._hero_goblin_group_seq += 1
            self.group_id = bs._hero_goblin_group_seq
            bs._hero_goblin_group_state[e.player] = (self.group_id, bs.time)

    def use_ability(self):
        return False   # 本体存活时按钮禁用（官方：Flag_Disable_Ability_Button）

    def on_death(self):
        e = self.entity
        bs = e.battle_state
        alive = 0
        for t in list(bs.entities.values()):
            if (t.is_alive and t.player == e.player and t.card_name == 'Goblins'
                    and isinstance(t.entity_holder, HeroGoblins)
                    and t.entity_holder.group_id == self.group_id):
                alive += 1
        # 分批部署（0.2s 间隔）未落地的同组成员不算「最后一只阵亡」
        pending = any(item[0][3] == 'Goblins' and item[0][2] == e.player for item in bs.schedule)
        if alive == 0 and not pending and self.group_id is not None:
            open_goblin_window(bs, e.player, e.position)


class HeroMegaMinion(_HeroBase):
    """【M8 §9】Hero Mega Minion — Wounding Warp（2 费，单次）：
    部署被动标记最低 HP 敌人（目标死亡标记转移）；按钮瞬移到标记处
    （WarpDmg 399@L11 溅射 [半径假设 1.5]），之后永久对皇冠塔伤害 ×0.25（2026-08-04）。"""
    def __init__(self, entity):
        super().__init__(entity)
        self.mark_id = None
        self._mark()

    def _mark(self):
        e = self.entity
        best, best_hp = None, None
        for t in list(e.battle_state.entities.values()):
            if not t.is_alive or t.player == e.player: continue
            from battle import Troop
            if not isinstance(t, Troop): continue
            hp = t.hp + t.shield_health
            if best_hp is None or hp < best_hp:
                best, best_hp = t, hp
        self.mark_id = best.id if best is not None else None

    def on_tick(self, dt):
        super().on_tick(dt)
        # 标记转移：被标记者死亡 → 转移到新的最低 HP 敌人
        t = self.entity.battle_state.entities.get(self.mark_id) if self.mark_id else None
        if t is None or not t.is_alive:
            self._mark()

    def use_ability(self):
        e, ab = self.entity, _ab(self.entity)
        bs = e.battle_state
        t = bs.entities.get(self.mark_id) if self.mark_id else None
        if t is None or not t.is_alive:
            self._mark()
            t = bs.entities.get(self.mark_id) if self.mark_id else None
            if t is None:
                return False   # 无标记目标 → 不扣费
        e.position = Position(t.position.x + 0.5, t.position.y)   # 贴脸落点（空中单位）
        e.path = []
        bs.deal_area_damage(e.player, t.position, ab['warpRadius'],
                            _sv(ab['warpDamage'], e.level), True, True)
        e.data.tower_damage_mult = ab['crownMult']   # 永久（不设时限）
        return True


class HeroTombstone(_HeroBase):
    """【M8 §11】Hero Tombstone — Regal Revive（5 费，单次）：
    按钮预付 → 墓碑破碎时 Tomb Queen 从墓中升起（HP4224/Dmg422@L11 [Fandom]；
    只攻建筑 sight 7 [2026-08-04 5.5→7]）；Queen 计时器 [假设 15s-待实测] 到期/死亡即终。
    [2026-07-06] Hero 形态移除持续产骷髅（overlay 摘除 spawnCharacterData）。"""
    def __init__(self, entity):
        super().__init__(entity)
        _scd = entity.data.data.get('summonCharacterData') or {}
        if isinstance(_scd, dict):
            _scd.pop('spawnCharacterData', None)

    def use_ability(self):
        self.entity._hero_queen_armed = True
        return True

    def on_death(self):
        e = self.entity
        if not getattr(e, '_hero_queen_armed', False):
            return
        from battle import Troop
        ab = _ab(e)
        q = Troop(e.battle_state.next_entity_id, Position(e.position.x, e.position.y),
                  e.player, ab['queenCard'], e.battle_state)
        q._evo_temp_lifetime = ab['queenLifetime']   # 复用临时寿命管线（到期即亡）
        e.battle_state._spawn_entity(q)


class HeroBerserker(_HeroBase):
    """【M8 §12】Hero Berserker — Savage Survival（3 费，单次）：
    熊灵附体 4s：攻速 0.2s / 移速 UltraFast(135) / HP 不低于 1 / 对皇冠塔伤害 ×0.25 /
    BearDmg 167@L11；结束 change_back 还原。HP 下限走 on_take_damage 受击钩子
    （重伤结算 clamp 到 1——口径假设：clamp 前不享受减伤/护盾细分，待 L4）。"""
    def __init__(self, entity):
        super().__init__(entity)
        self.bear_timer = 0.0
        self._orig = None

    def use_ability(self):
        e, ab = self.entity, _ab(self.entity)
        self._orig = (e.data.hit_speed, e.speed, e.data.tower_damage_mult, e.data.damage)
        e.data.hit_speed = ab['hitSpeed']
        e.speed = ab['speed'] / 50.0   # 内部速度单位换算（Card.speed 同口径 /50）
        e.data.tower_damage_mult = ab['crownMult']
        e.data.damage = _sv(ab['bearDamage'], e.level)
        self.bear_timer = ab['duration']
        return True

    def on_tick(self, dt):
        super().on_tick(dt)
        if self.bear_timer > 0:
            self.bear_timer -= dt
            if self.bear_timer <= 0:
                # change_back_from_bearform 还原
                (self.entity.data.hit_speed, self.entity.speed,
                 self.entity.data.tower_damage_mult, self.entity.data.damage) = self._orig

    def on_take_damage(self, amount, source):
        """熊灵 HP 下限：致死伤害 clamp 到 1 HP（minimum hitpoints）。返回 True=引擎短路。"""
        e = self.entity
        if self.bear_timer > 0 and e.is_alive and e.hp - amount <= 0:
            e.hp = 1.0
            return True
        return False


class HeroDarkPrince(DarkPrince):
    """【M8 §13】Hero Dark Prince — Destructive Dismount（3 费，单次）：
    按钮下马：本体跳跃落地溅射（伤害=普攻 [假设，数值缺]，半径 1.2）、失去冲锋改徒步
    溅射攻击；犀牛独立单位冲塔（HP1356/Dmg179/ChargeDmg358@L11 [Fandom]，冲锋复用
    Prince 管线）。两者独立存活。"""
    def use_ability(self):
        e, ab = self.entity, _ab(self.entity)
        bs = e.battle_state
        # ① 本体落地溅射（Change_To_Walking 落地帧）
        bs.deal_area_damage(e.player, e.position, ab['landingRadius'],
                            e.data.damage * ab['landingDamageMult'], True, False)
        # ② 本体徒步化：失去冲锋（charge_range=0）+ 普攻带溅射（area_damage_radius）
        e.data.charge_range = 0
        self.charging = False
        e.data.area_damage_radius = ab['landingRadius']
        # ③ 犀牛坐骑独立 spawn（Target=Buildings，charge 行为走 DarkPrinceHeroRhino 类）
        from battle import Troop
        gy = 0.8 if e.player == 0 else -0.8   # 坐骑落位本体侧后（口径假设）
        rhino = Troop(bs.next_entity_id, Position(e.position.x, e.position.y + gy),
                      e.player, ab['mountCard'], bs)
        bs._spawn_entity(rhino)
        return True


class DarkPrinceHeroRhino(Prince):
    """【M8 §13】犀牛坐骑（独立单位）：只攻建筑，冲锋伤害 damageSpecial=358@L11。
    复用 Prince 冲锋管线（charge_range 3.5 [假设同 Prince]，数据见 elite17_data.RHINO_SCD）。"""
    pass


class HeroBalloon(Balloon):
    """【M8 §14】Hero Balloon — Coffin Cadets（2 负，单次）：
    召出骷髅伞兵（Skeletrooper）飞向 6 格内最近地面敌人，落地伤害 263@L11（对塔 ×0.1）
    并驻场攻击（HP473/Dmg204@L11/攻速 1.1s/VeryFast/对地）。
    本体死亡炸弹（Balloon 死亡 spawn）继承基础 Balloon 机制类。"""
    def use_ability(self):
        e, ab = self.entity, _ab(self.entity)
        bs = e.battle_state
        from battle import Troop
        best, best_d = None, ab['seekRadius']
        for t in list(bs.entities.values()):
            if not t.is_alive or t.player == e.player: continue
            if t.data.is_air_unit: continue   # 只找地面敌人
            d = e.position.distance_to(t.position)
            if d < best_d:
                best, best_d = t, d
        if best is None:
            return False   # 无可投放目标 → 不扣费
        c = Troop(bs.next_entity_id, Position(e.position.x, e.position.y),
                  e.player, ab['cadetCard'], bs)
        c._sk_target_id = best.id
        bs._spawn_entity(c)
        return True


class Skeletrooper(_HeroBase):
    """【M8 §14】骷髅伞兵：伞降段不可选取/无敌，落点 AOE（对塔 ×0.1）后转常规驻场攻击。
    目标中途死亡 → 原地落地（口径假设，官方有 Failsafe_Spawn_Delayer 兜底分支）。"""
    def __init__(self, entity):
        super().__init__(entity)
        self.landed = False
        self.land_pos = None
        # 直部署（无 abilityData，如 batch_smoke）回退 Balloon 能力参数
        self.ab = _ab(entity) or _HERO_ABILITIES['Balloon']
        self.fly_timer = 3.0   # [假设] 伞降兜底时限（官方 _Failsafe_Spawn_Delayer 分支）
        entity.targetable = False
        entity.invincible = True

    def _land(self, pos):
        e = self.entity
        ab = self.ab
        self.landed, self.land_pos = True, Position(pos.x, pos.y)
        e.targetable = True
        e.invincible = False
        e.battle_state.deal_area_damage(e.player, pos, ab['landingRadius'],
                                        _sv(ab['landingDamage'], e.level), True, True,
                                        ab['crownMult'])

    def on_tick(self, dt):
        super().on_tick(dt)
        if self.landed: return
        e, ab = self.entity, self.ab
        t = e.battle_state.entities.get(getattr(e, '_sk_target_id', None))
        if t is None or not t.is_alive:
            self._land(e.position)   # 兜底：目标消失原地落地
            return
        d = e.position.distance_to(t.position)
        # 落地阈值 = 双方碰撞半径之和 + 0.3（目标移动牵引下仍可落地）
        if d <= e.data.collision_radius + t.data.collision_radius + 0.3:
            self._land(Position(t.position.x, t.position.y))
            return
        self.fly_timer -= dt
        if self.fly_timer <= 0:
            self._land(e.position)   # 兜底：时限内未抵达 → 原地落地
        e.position = Position(e.position.x + (t.position.x - e.position.x) / d * ab['flySpeed'] * dt,
                              e.position.y + (t.position.y - e.position.y) / d * ab['flySpeed'] * dt)

    def on_attack(self, current_target=None):
        if not self.landed:
            self.entity.attack_cooldown = max(self.entity.attack_cooldown, 0.1)
            return   # 伞降段不攻击
        super().on_attack(current_target)


class HeroIceWizard(IceWizard):
    """【M8 §10】Hero Ice Wizard — 冰封自身→破碎冻结 AOE（机制先行，数值全缺）：
    能力 = 自身变冰块（不可选取+无敌+定身）3s【假设-待实测】；破碎时 AOE 冻结 2s【暂借
    Freeze 语义】半径 3【暂借】+ 减速 30% 2s【暂借 Ice Golem/Hero】。
    [内存结构 54 动作组：cube/attached/reapper/spawn_freeze/FreezeAeo——击杀联动的
    from_kill 分支未建模（参数全缺），待实测后补]。"""
    def __init__(self, entity):
        super().__init__(entity)
        self.cube_timer = 0.0
        self._orig_speed = None

    def use_ability(self):
        e, ab = self.entity, _ab(self.entity)
        self.cube_timer = ab['cubeDuration']
        e.targetable = False
        e.invincible = True
        # 定身不用 freeze_timer（其会使 Troop.update 早退 → on_tick 饥饿），
        # 改为移速归零 + on_attack 禁攻，由本类驱动冰块计时
        self._orig_speed = e.speed
        e.speed = 0.0
        return True

    def on_tick(self, dt):
        super().on_tick(dt)
        e, ab = self.entity, _ab(self.entity)
        if self.cube_timer > 0:
            self.cube_timer -= dt
            if self.cube_timer <= 0:
                # 破碎复现（attached_wait_to_reapper）+ 冻结 AOE（spawn_freeze/FreezeAeo）
                from battle import EvoEffectZone
                e.targetable = True
                e.invincible = False
                e.speed = self._orig_speed
                zone = EvoEffectZone(e.battle_state.next_entity_id,
                                     Position(e.position.x, e.position.y),
                                     e.player, e.battle_state,
                                     radius=ab['freezeRadius'], lifetime=0.5, tick=0.5,
                                     slow=ab['slowMult'], stun_pulse=ab['freeze'],
                                     level=1, label='IceWizardHero_FreezeAeo')
                e.battle_state._spawn_entity(zone)

    def on_attack(self, current_target=None):
        if self.cube_timer > 0:   # 冰封中不攻击不移动
            return
        super().on_attack(current_target)


class HeroEliteArcher(_HeroBase):
    """【M8 §16】Hero Elite Archer — Warp + Triple Shot（机制先行，数值全缺）：
    能力 = 向最近敌人瞬移 3 格【假设-暂借】+ 接下来 3 发三连射（攻速 0.3s【暂借-待实测】）
    + 部署假人分身（HP104 [内存 Dummy 表·中置信]，寿命 5s【假设】）。"""
    def __init__(self, entity):
        super().__init__(entity)
        self.shots_left = 0
        self._orig_hit_speed = None

    def use_ability(self):
        e, ab = self.entity, _ab(self.entity)
        bs = e.battle_state
        # ① Warp：向最近敌人瞬移（不可走则折半兜底）
        best, best_d = None, float('inf')
        from battle import Troop, Building
        for t in list(bs.entities.values()):
            if not t.is_alive or t.player == e.player or not isinstance(t, (Troop, Building)): continue
            d = e.position.distance_to(t.position)
            if d < best_d:
                best, best_d = t, d
        if best is not None:
            dx, dy = best.position.x - e.position.x, best.position.y - e.position.y
            n = (dx * dx + dy * dy) ** 0.5 or 1.0
            step = ab['warpRange'] / n
            nx, ny = e.position.x + dx * step, e.position.y + dy * step
            if not bs.ground_walkable(Position(nx, ny), e.data.collision_radius):
                nx, ny = e.position.x + dx * step * 0.5, e.position.y + dy * step * 0.5
            e.position = Position(nx, ny)
            e.path = []
        # ② Triple Shot：接下来 3 发攻速 0.3s（打完还原）
        self._orig_hit_speed = e.data.hit_speed
        e.data.hit_speed = ab['tripleHitSpeed']
        self.shots_left = ab['tripleShots']
        # ③ 假人分身（Dummy_Start_Group；承受火力用）
        d = Troop(bs.next_entity_id, Position(e.position.x, e.position.y + (0.8 if e.player == 0 else -0.8)),
                  e.player, ab['dummyCard'], bs)
        d._evo_temp_lifetime = ab['dummyLifetime']
        bs._spawn_entity(d)
        return True

    def on_attack(self, current_target=None):
        super().on_attack(current_target)
        if self.shots_left > 0:
            self.shots_left -= 1
            if self.shots_left <= 0:
                self.entity.data.hit_speed = self._orig_hit_speed


class HeroIceGolemite(_HeroBase):
    """【M8 §17】Hero Ice Golemite — 冰雪光环（机制先行，数值【暂借 Ice Golem/Hero】）：
    能力 = 冰雪光环 AOE（半径 4 / 3 次脉冲 / 间隔 1s / 脉冲伤 69@L11 / 减速 30% 2s），
    按目标体型分档 Freeze/Slow（分档规则见 battle.IceGolemiteSnowZone）。"""
    def use_ability(self):
        e, ab = self.entity, _ab(self.entity)
        from battle import IceGolemiteSnowZone
        zone = IceGolemiteSnowZone(
            e.battle_state.next_entity_id, Position(e.position.x, e.position.y),
            e.player, e.battle_state,
            radius=ab['radius'], pulses=ab['pulses'], interval=ab['interval'],
            damage=_sv(ab['pulseDamage'], e.level),
            slow=ab['slowMult'], slow_duration=ab['slowDuration'],
            small_freeze=ab['smallFreeze'], level=1,   # 伤害已按级换算 → level=1 不二次缩放
            label='IceGolemiteHero_ice_aura')
        e.battle_state._spawn_entity(zone)
        return True


# M8：卡名 → Hero 机制类（apply_hero_overlay 消费；BarbLog 为法术无实体 → 条件窗）
HERO_CLASSES = {
    'Knight': HeroKnight,
    'Musketeer': HeroMusketeer,
    'MiniPekka': HeroMiniPekka,
    'Valkyrie': HeroValkyrie,
    'Wizard': HeroWizard,
    'Bowler': HeroBowler,
    'Giant': HeroGiant,
    'Goblins': HeroGoblins,
    'MegaMinion': HeroMegaMinion,
    'Tombstone': HeroTombstone,
    'Berserker': HeroBerserker,
    'DarkPrince': HeroDarkPrince,
    'Balloon': HeroBalloon,
    'IceWizard': HeroIceWizard,
    'EliteArcher': HeroEliteArcher,
    'IceGolemite': HeroIceGolemite,
}


# ==================== 勘误批3/8/10/12/14：专项机制类 ====================

class ElectroGiant(BasicCharacter):
    """Zap Pack 反射（勘误批3）：被 2 格内敌方部队造成伤害时, 对攻击者反射 75@基准 + 0.5s 眩晕；
    自身冰冻期间不反射【官方】；一击多段按一次计【简化, 官方按命中次数】。"""
    def on_damaged(self, amount, source):
        from battle import Troop
        e = self.entity
        if not e.is_alive or e.freeze_timer > 0:
            return
        scd = e.data.data.get('summonCharacterData') or {}
        rad = (scd.get('reflectedAttackRadius') or 2000) / 1000
        if not isinstance(source, Troop) or source.player == e.player or not source.is_alive:
            return
        if source.position.distance_to(e.position) > rad:
            return
        dmg = (scd.get('reflectedAttackDamage') or 75) * level_scale(e.level)
        source.take_damage(dmg, source=e)
        source.apply_buff(stun=(scd.get('reflectedAttackBuffDuration') or 500) / 1000)


class _AttackStunMixin:
    """攻击附带 ZapFreeze 眩晕（buffOnDamageData 消费；仅部队, 塔不吃眩晕）。"""
    _stun_time = 0.5
    def on_attack(self, current_target=None):
        from battle import Troop
        super().on_attack(current_target)
        if current_target is not None and current_target.is_alive and isinstance(current_target, Troop):
            current_target.apply_buff(stun=self._stun_time)


class ElectroWizard(_AttackStunMixin, BasicCharacter):
    """勘误批3：部署 Zap（半径 3 / 75@基准【Fandom, 快照缺字段, 待对拍】/ 0.5s 眩晕）+ 攻击眩晕。"""
    def on_spawn(self):
        from battle import Troop
        bs = self.entity.battle_state
        e = self.entity
        bs.deal_area_damage(e.player, e.position, 3.0, 75 * level_scale(e.level), True, True)
        for o in list(bs.entities.values()):
            if isinstance(o, Troop) and o.is_alive and o.player != e.player \
                    and o.position.distance_to(e.position) <= 3.0 + o.data.collision_radius:
                o.apply_buff(stun=0.5)


class MiniSparkys(_AttackStunMixin, BasicCharacter):
    """勘误批8：Zappies 每次攻击 0.5s 眩晕（buffOnDamageData=ZapFreeze, 此前零消费）。"""


class ElectroSpirit(BasicCharacter):
    """勘误批3：自毁 + 链电（stats 行 chained_hit_count 9 / radius 4.0）。
    攻击命中后向最近敌人链式弹射（同伤害）, 攻击后自毁【kamikaze】。"""
    def on_attack(self, current_target=None):
        from battle import Troop, Projectile, SpawnProjectile
        from card_utils import projectiles
        super().on_attack(current_target)
        e = self.entity
        scd = e.data.data.get('summonCharacterData') or {}
        count = int(scd.get('chainedHitCount') or 9)
        radius = (scd.get('chainedHitRadius') or 4000) / 1000
        _pd = (scd.get('projectileData') or {})
        _prow = projectiles.get(_pd.get('name') or '') or {}
        _base = _prow.get('damage') or e.data.damage
        dmg = _base * (1.1 ** (e.level - 1))   # 投射物 lv1 基准 → 等级缩放
        hit_ids = {current_target.id} if current_target else set()
        cur = current_target
        for _ in range(max(0, count - 1)):
            nxt, nd = None, float('inf')
            for o in e.battle_state.entities.values():
                if not o.is_alive or o.id in hit_ids or o.player == e.player: continue
                if isinstance(o, (Projectile, SpawnProjectile)): continue
                d = (cur.position if cur is not None else e.position).distance_to(o.position)
                if d <= radius and d < nd:
                    nxt, nd = o, d
            if nxt is None: break
            nxt.take_damage(dmg, source=e)
            hit_ids.add(nxt.id)
            cur = nxt
        e.is_alive = False   # 自毁（官方：命中后湮灭）


class RamRider(Prince):
    """勘误批10：羊冲锋（Prince 管线继承）+ 骑手缠网：攻击命中部队施加 -70% 减速 2s
    （BolaSnare, 快照嵌套字段缺 → Fandom 机制【待对拍】）；骑手独立攻击未建模【简化】。"""
    def on_attack(self, current_target=None):
        from battle import Troop
        super().on_attack(current_target)
        if current_target is not None and current_target.is_alive and isinstance(current_target, Troop):
            current_target.apply_buff(speed_mult=0.30, duration=2.0)


class MovingCannon(BasicCharacter):
    """勘误批9：Cannon Cart——HP 降至 50% 变身 BrokenCannon（定身建筑态, 保留当前 HP,
    寿命 15s【Fandom, 快照缺】；变身不重置索敌【简化：新建建筑自然重索敌】）。"""
    def on_tick(self, dt):
        super().on_tick(dt)
        e = self.entity
        if not e.is_alive or getattr(self, '_transformed', False):
            return
        if e.hp <= e.data.hp * 0.5:
            self._transformed = True
            from battle import Building
            bs = e.battle_state
            nb = Building(bs.next_entity_id, Position(e.position.x, e.position.y),
                          e.player, 'BrokenCannon')
            nb.hp = e.hp
            nb.deploy_delay_remaining = 0
            bs._spawn_entity(nb)   # _spawn_entity 统一接管 battle_state/id
            e.is_alive = False   # 变身不是死亡：绕过 die() 亡语


class Phoenix(BasicCharacter):
    """勘误批9：凤凰蛋链——死亡 → 亡语火球（64@基准/半径 2.5/击退）+ 产蛋（孵化 4.3s【Fandom 现行】）
    → 孵化出满血再生体（2025/11 取消 80% 规则）；再生体死亡不再产蛋/爆火球。"""
    def on_death(self):
        if getattr(self, '_is_rebirth', False):
            return
        e = self.entity
        bs = e.battle_state
        scd = e.data.data.get('summonCharacterData') or {}
        pfp = scd.get('deathSpawnProjectileData') or {}
        # 亡语火球
        dmg = (pfp.get('damage') or 64) * level_scale(e.level)
        radius = (pfp.get('radius') or 2500) / 1000
        bs.deal_area_damage(e.player, e.position, radius, dmg, True, True)
        bs.push_enemies(e.player, e.position, radius, (pfp.get('pushback') or 2000) / 1000)
        # 蛋
        from battle import Troop
        egg = Troop(bs.next_entity_id, Position(e.position.x, e.position.y), e.player, 'PhoenixEgg', bs)
        egg._soul_excluded = True
        bs._spawn_entity(egg)
        bs.next_entity_id += 1


class PhoenixEgg(BasicCharacter):
    """凤凰蛋（勘误批9）：不可移动/不攻击, 孵化 4.3s 后出满血再生 Phoenix（不再产蛋）。"""
    def __init__(self, entity):
        super().__init__(entity)
        self.hatch = 4.3   # 【Fandom 现行 4.3s; 快照 deployTime=1000 为部署动画】

    def on_tick(self, dt):
        super().on_tick(dt)
        e = self.entity
        if not e.is_alive:
            return
        self.hatch -= dt
        if self.hatch <= 0:
            from battle import Troop
            bs = e.battle_state
            t = Troop(bs.next_entity_id, Position(e.position.x, e.position.y), e.player, 'Phoenix', bs)
            bs._spawn_entity(t)
            bs.next_entity_id += 1
            t.entity_holder._is_rebirth = True
            t._soul_excluded = True
            e.is_alive = False   # 破壳不是死亡


class ThreeMusketeers(BasicCharacter):
    """勘误批12（2025/11 重构）：敌部队 3 格内 → 近战形态（射程 1.2, 仅地面【简化: 射程覆盖】）；
    敌远离 → 自动切回远程 6。双向动态切换, 无次数限制。"""
    def on_tick(self, dt):
        super().on_tick(dt)
        e = self.entity
        if not e.is_alive:
            return
        from battle import Troop
        near = False
        for o in e.battle_state.entities.values():
            if isinstance(o, Troop) and o.is_alive and o.player != e.player \
                    and o.position.distance_to(e.position) <= 3.0:
                near = True
                break
        want = 1.2 if near else 6.0
        if getattr(self, '_form_range', None) != want:
            self._form_range = want
            e._range_override = want
            e.target_id = None
            e.path = []


class GoblinGiant(BasicCharacter):
    """勘误批5：哥布林巨人背载枪哥布林——独立投掷（射程 5.5 / 攻速 1.7s, 直接结算【简化: 无弹道】）；
    死亡枪哥布林落地 → SpearGoblin（官方 dismount）。"""
    def __init__(self, entity):
        super().__init__(entity)
        self.rider_cd = 1.0

    def on_tick(self, dt):
        from battle import Troop
        from card_utils import projectiles
        super().on_tick(dt)
        e = self.entity
        if not e.is_alive or e.deploy_delay_remaining > 0:
            return
        self.rider_cd -= dt
        if self.rider_cd > 0:
            return
        self.rider_cd = (e.data.data.get('summonCharacterData', {}).get('spawnCharacterData', {})
                         or {}).get('hitSpeed', 1700) / 1000
        best, best_d = None, float('inf')
        for o in e.battle_state.entities.values():
            if not o.is_alive or o.player == e.player or not isinstance(o, Troop): continue
            d = o.position.distance_to(e.position)
            if d <= 5.5 and d < best_d:
                best, best_d = o, d
        if best is None:
            return
        rider = Card('SpearGoblinGiant')
        _prow = projectiles.get('SpearGoblinProjectile') or {}
        _rd = _prow.get('damage') * (1.1 ** (e.level - 1)) if _prow.get('damage') else 0   # 快照 scd damage=None → 投射物行 32@lv1【标注】
        if _rd:
            best.take_damage(_rd, source=e)

    def on_death(self):
        # 枪哥布林落地（dismount）；专属 on_death 使通用亡语跳过, 不双出
        from battle import Troop
        bs = self.entity.battle_state
        t = Troop(bs.next_entity_id, Position(self.entity.position.x, self.entity.position.y),
                  self.entity.player, 'SpearGoblin', bs)
        t._soul_excluded = True
        bs._spawn_entity(t)
        bs.next_entity_id += 1


# ==================== 勘误批7（用户裁决）：塔兵机制 ====================

class King_KnifeTowers(BasicCharacter):
    """Dagger Duchess 飞刀蓄能（用户口径 2026-09-04）：
    - 8 支飞刀；有刀时 0.5s 连发（用户勘误 2026-09-04, 覆盖快照 450ms）, 每次攻击消耗 1 支；
    - 回充：从消耗第一支飞刀开始计时, 每 0.9s 回 1 支（独立计时器, 不因断攻重置）；8 支封顶；
    - 刀耗尽期间无法攻击（等回充）。"""
    MAX_KNIVES = 8
    RECHARGE = 0.9

    def __init__(self, entity):
        super().__init__(entity)
        self.knives = self.MAX_KNIVES
        self._regen_started = False
        self._regen_timer = 0.0

    def on_attack(self, current_target=None):
        if self.knives <= 0:
            # 无刀：跳过本次攻击（等回充），不进入冷却长眠
            self.entity.attack_cooldown = self.RECHARGE
            return
        super().on_attack(current_target)
        self.entity.attack_cooldown = 0.5   # 用户勘误：攻速 0.5s（覆盖快照 450ms）
        self.knives -= 1
        if not self._regen_started:
            self._regen_started = True   # 从消耗第一支开始计时（用户口径）
            self._regen_timer = self.RECHARGE

    def on_tick(self, dt):
        super().on_tick(dt)
        if self._regen_started and self.knives < self.MAX_KNIVES:
            self._regen_timer -= dt
            if self._regen_timer <= 0:
                self.knives += 1
                self._regen_timer = self.RECHARGE


class King_ChefTowers(BasicCharacter):
    """Royal Chef 烹饪（用户裁决的简单拟合 2026-09-04）：
    - 基础烹饪 23s（首饼 7s【Fandom troop 报告】）, 烹饪期间自己每次攻击延长烹饪：
      Δ = (38-23)s × 攻速 / 38s —— 即「持续攻击 38s」恰好把 23s 拉满到 38s；
    - 一座公主塔已失 → 烹饪变慢 ×2【假设系数】；双塔失 → 停止烹饪；
    - 出餐：+1 级 = 治疗 10% max hp（可叠, 允许超上限【简化】）+ 伤害 ×1.1（_damage_mult 载体）；
      只喂 >33% 血的友军部队（不喂建筑）, 优先未喂过的最高血量。"""
    BASE_COOK = 23.0
    FIRST_COOK = 7.0
    MAX_COOK = 38.0

    def __init__(self, entity):
        super().__init__(entity)
        self._cook_left = self.FIRST_COOK
        self._fed_ids = set()

    def _attack_extension(self):
        hit = (self.entity.data.hitSpeed or 1000) / 1000
        return (self.MAX_COOK - self.BASE_COOK) * hit / self.MAX_COOK

    def on_attack(self, current_target=None):
        super().on_attack(current_target)
        # 烹饪期间攻击 → 延长烹饪（用户拟合）
        if self._cook_left > 0:
            self._cook_left = min(self.MAX_COOK, self._cook_left + self._attack_extension())

    def on_tick(self, dt):
        from battle import Troop
        super().on_tick(dt)
        e = self.entity
        bs = e.battle_state
        # 塔况修正
        p = bs.players[e.player]
        lost = (p.left_tower_hp <= 0) + (p.right_tower_hp <= 0)
        if lost >= 2:
            return                      # 双塔失 → 停止烹饪
        rate = 2.0 if lost == 1 else 1.0   # 一塔失 → 变慢【假设 ×2】
        self._cook_left -= dt / rate
        if self._cook_left > 0:
            return
        self._cook_left = self.BASE_COOK   # 下一炉回归基础时长
        # 选feeding目标：>33% 血的友军部队, 优先未喂过的最高血量
        best, best_key = None, None
        for o in bs.entities.values():
            if not isinstance(o, Troop) or not o.is_alive or o.player != e.player: continue
            if o.hp <= o.data.hp * 0.33: continue
            key = (o.id not in self._fed_ids, o.hp)
            if best_key is None or key > best_key:
                best, best_key = o, key
        if best is None:
            return
        self._fed_ids.add(best.id)
        best._damage_mult = getattr(best, '_damage_mult', 1.0) * 1.1
        best.hp = best.hp + best.data.hp * 0.1   # 可叠/可超上限【简化】
