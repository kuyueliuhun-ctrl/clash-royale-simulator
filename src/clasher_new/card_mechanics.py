from core import BasicCharacter, Position
from card_utils import Card, level_scale
from arena import TileGrid
from evolutions import OFFICIAL_OVERRIDES

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
        self.next_spawn_remaining = 7.0

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
            evo = getattr(self.entity, 'evo', None)
            oscd = (evo or {}).get('onStartChargingActionData') if evo else None
            if oscd:
                strength = oscd.get('pushBackStrength', 2500) / 1000
                dmg = oscd.get("pushBackDamage", 83) * level_scale(e.level)
                e = self.entity
                e.battle_state.deal_area_damage(e.player, e.position, 2.0, dmg, False, True)
                e.battle_state.push_enemies(e.player, e.position, 2.0, strength)
        if self.charging: self.entity.attack_cooldown = 0

    def on_attack(self, current_target=None):
        if not self.charging:
            current_target.take_damage(self.entity.data.damage)
            self.starting_position = Position(self.entity.position.x, self.entity.position.y)
        else:
            current_target.take_damage(self.entity.data.charge_damage)
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
    def on_spawn(self):
        spawn_data = self.entity.data.spawn_data
        for entity in self.entity.battle_state.entities.values():
            if not entity.is_alive or entity.player == self.entity.player: continue
            if not entity.position.distance_to(self.entity.position) < spawn_data['radius']/1000 + entity.data.collision_radius:
                continue
            entity.take_damage(spawn_data['damage'])
            entity.speed_debuff = min(1 + spawn_data['buffData']['speedMultiplier'] / 100, entity.speed_debuff)
            entity.debuff_time_remaining = spawn_data['buffTime']/1000

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
            for entity in self.entity.battle_state.entities.values():
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
        if t is None or not t.is_alive or t.data.is_air_unit: return
        if not hasattr(t, 'hook_pull'): return  # 仅部队可钩（对建筑官方为把自己拉过去，简化跳过，待 L4）
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