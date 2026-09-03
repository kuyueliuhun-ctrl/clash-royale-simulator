from dataclasses import dataclass
import math

@dataclass
class Position:
    x: float
    y: float
    def distance_to(self, other):
        return math.hypot(self.x-other.x, self.y-other.y)

class BlankEntity:
    """A placeholder that only encodes the position value."""
    def __init__(self, position):
        self.position = position

class BasicCharacter:
    def __init__(self, entity):
        self.entity = entity
        self.battle_state = self.entity.battle_state
        self.data = self.entity.data
    def on_spawn(self): pass
    def on_tick(self, dt): self.battle_state = self.entity.battle_state
    def on_death(self): pass
    def on_attack(self, current_target=None):
        self.entity.last_attack_time = self.battle_state.time  # M4 fortify 脱战判定
        # M4.5 动作链：Berserker 等伤害仅在攻击序列内（data.damage=0）也须结算
        if self.entity.data.damage or getattr(self.entity, 'attack_seq', None):
            damage = self.entity.ramped_damage(self.data.damage)  # M1 递增伤害 / M4.5 攻击序列
            if self.entity.data.area_damage_radius:
                self.battle_state.deal_area_damage(self.entity.player, self.entity.position, self.data.area_damage_radius,
                                                   damage,
                                                   self.data.attack_air, self.data.attack_ground)
            else:
                if 'King' in current_target.name:
                    current_target.take_damage(damage*self.entity.data.tower_damage_mult, delayed=True)
                else:
                    current_target.take_damage(damage, delayed=True)
        elif self.entity.data.projectiles:
            # must have projectiles
            self.entity.create_projectile(current_target)
        self.entity.attack_cooldown = self.data.hit_speed
        # —— M4 族7：觉醒攻击后钩子（骷髅分裂/蝙蝠自愈/野蛮人狂暴/弓箭手二段箭/RG推击/女武神龙卷）——
        if hasattr(self.entity, '_evo_on_attack'):
            self.entity._evo_on_attack(current_target)
        # —— M4.5 动作链：攻击完成钩子（InfernoDragon_EV1 递增推进 / Berserker 多段命中）——
        if hasattr(self.entity, '_on_attack_done'):
            self.entity._on_attack_done(current_target)
        # —— M3 族6：渔夫特殊攻击（钩拉）由 Fisherman 钩子在 on_tick 中处理 ——
        if self.entity.data.kamikaze:
            self.entity.is_alive = False