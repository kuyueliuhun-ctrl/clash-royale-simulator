from core import BlankEntity
from player import PlayerState
from pathfinding_heap import EntityPathfinder, position_to_cell, cell_to_position
from card_mechanics import *
from card_utils import Card, TimedExplosiveData, spells, buildings, projectiles, character_to_card, projectile_from_row
from card_utils import _rarity_level_index, _value_at_level, level_scale
from evolutions import evolution_state, derive_evolved_stats, OFFICIAL_OVERRIDES
import math
from itertools import combinations


class Entity:
    def __init__(self, id, position, player, card_name, battle_state: "BattleState" = None):
        # Stores permanent information about this entity like `player` and `card_name`.
        self.id, self.position, self.player, self.card_name, self.battle_state = (id, position, player, card_name, battle_state)
        self.data = Card(self.card_name)
        self.level = self.data.level  # 11-16 全等级支持：实体当前卡牌等级
        self.name = self.data.name

        # Stores state information that is likely to change.
        self.is_alive = True
        self.attack_cooldown = self.data.hit_speed-self.data.load_time
        self.speed = self.data.speed
        self.hp = self.data.hp
        self.shield_health = self.data.shield_health
        self.target_id = None

        # Why use both targetable and invincible? Because some entities like the royal ghost/archer queen can be invisible but
        # still takes damage. Other entities like the bandit/boss bandit/golden knight/miner(underground) can not be hit in a
        # certain state.
        self.targetable = True
        self.invincible = False

        # There are a lot of entities that can jump across the arena's river, and the movement pattern is
        # significantly different, so I dedicated a special variable to store this information.
        self.jumping_across_river = False

        # This affects both speed and hit speed. I will rewrite this when poison comes out.
        self.speed_buff = 1.0
        self.speed_debuff = 1.0
        self.buff_time_remaining = 0.0
        self.debuff_time_remaining = 0.0
        self.hit_speed_mult = 1.0  # M3：独立攻速倍率（ArcherQueen 隐身 +180% 等不改变移速的攻速 buff）

        # This part is where flexibility comes in - some cards have special mechanics that can't be handled in
        # the entity/troop/buildings classes. So I created `BasicCharacter` to delegate most of the logic.
        # If a card doesn't have special logic like the knight and mini-pekka, then only `BasicCharacter` will be
        # used.
        self.entity_holder = BasicCharacter(self)
        if self.card_name in globals() and not isinstance(self, Projectile):
            self.entity_holder = eval(f"{self.card_name}(self)")
        self.entity_holder.on_spawn()

        self.path = []

        self.pending_damage = []

        # —— M1 递增伤害状态（仅 has_ramp 卡生效：InfernoDragon/InfernoTower）——
        self.ramp_target_id = None
        self.ramp_timer = 0.0
        self.ramp_stage = 0

        # —— M2 族4：通用 buff 槽 ——
        self.freeze_timer = 0.0          # 冰冻/眩晕：停移停攻（Zap=0.5s、Freeze=4s、Tesla 脉冲…）
        self.damage_reduction = 0.0      # 减伤比例（Knight 觉醒 fortify / Monk 禅定）
        self.damage_reduction_timer = 0.0
        self.regen_buffs = []            # 治疗导槽 [{'hps': 每秒, 'time': 剩余}]（Heal / 觉醒蝙蝠）
        self.hook_pull = None            # M2 族5：钩拉 {'x','y','speed'}（渔夫钩/Tornado 备用）
        # —— M4 族7：觉醒形态特殊字段（derive_evolved_stats 输出）——
        self.evo = None
        self._fortify_dr = 0.0  # M4：脱战减伤独立槽（fortify），与 buff 减伤（Monk）取 max
        self.last_attack_time = -999.0  # M4：上次攻击时刻（fortify 脱战判定用）
        self.ability_cd = 0.0            # M3 族6：能力冷却剩余
        self.ability_uses = 0            # M3 族6：已用次数（BossBandit 限 2 次）
        # —— M4.5 动作链：攻击序列状态（attackSequenceList）——
        # attack_seq = 序列原始定义（[{damage,...}]）；None = 无攻击序列
        # attack_seq_mode = "Manual"（跨攻击逐档推进，如 InfernoDragon_EV1）或 None（单次攻击多段命中，如 Berserker）
        self.attack_seq = None
        self.attack_seq_mode = None
        self.attack_seq_damages = []    # 各档伤害（已按当前级缩放）
        self.attack_seq_stage = 0       # 当前档位索引（Manual 递增用）
        self.attack_seq_pending = 0     # 多段命中：剩余待打段数
        self.attack_seq_hit_timer = 0.0 # 多段命中：段间隔计时

    # —— M2 族4：统一 buff 入口 ——
    def apply_buff(self, speed_mult=None, hit_speed_mult=None, duration=0.0,
                   damage_reduction=None, stun=0.0, heal=None, retarget=False):
        """speed_mult/hit_speed_mult: 倍率（>1 加速、<1 减速）；damage_reduction: 0~1 减伤；
        stun: 定身秒数（附带攻击蓄力重置与可选重索敌）；heal: {'hps','time'}"""
        if stun > 0:
            self.freeze_timer = max(self.freeze_timer, stun)
            # Zap 类眩晕：重置攻击蓄力与冲锋（官方：眩晕重置蓄力类攻击，普通攻击仅暂停）
            self.attack_cooldown = max(self.attack_cooldown, self.data.hit_speed)
            holder = self.entity_holder
            if hasattr(holder, 'charging') and holder.charging:
                holder.charging = False
                self.speed = self.data.speed
                if hasattr(holder, 'starting_position'):
                    holder.starting_position = Position(self.position.x, self.position.y)
            if retarget:
                self.target_id = None
                self.path = []
            return
        if speed_mult is not None:
            if speed_mult >= 1.0:
                self.speed_buff = max(self.speed_buff, speed_mult)
            else:
                self.speed_debuff = min(self.speed_debuff, speed_mult)
            self.buff_time_remaining = max(self.buff_time_remaining, duration)
        if hit_speed_mult is not None and hit_speed_mult >= 1.0:
            # M3：独立攻速槽（不影响移速；Rage 类走 speed_buff 双驱动语义不变）
            self.hit_speed_mult = max(self.hit_speed_mult, hit_speed_mult)
            self.buff_time_remaining = max(self.buff_time_remaining, duration)
        if damage_reduction is not None:
            self.damage_reduction = max(self.damage_reduction, damage_reduction)
            self.damage_reduction_timer = max(self.damage_reduction_timer, duration)
        if heal:
            self.regen_buffs.append({'hps': heal['hps'], 'time': heal.get('time', 1.0)})

    def to_dict(self):
        """If I want to render a certain entity on the screen, what's the minimal information I'll need?"""
        return {
            'type': 'entity',
            'card_name': self.card_name,
            'player': self.player,
            'x': self.position.x,
            'y': self.position.y,
            'hp': self.hp,
            'max_hp': self.data.hp,
            'shield_max_hp': self.data.shield_health,
            'shield_hp': self.shield_health,
            'collision_radius': self.data.collision_radius if not isinstance(self, Projectile) else 0.3
        }

    def die(self):
        """Automatically call entity holder's on_death to prevent bugs"""
        self.is_alive = False
        self.entity_holder.on_death()
        if isinstance(self, Troop):
            self._evo_on_death()
        # 女巫妈妈诅咒：被诅咒单位死亡 → 生成 VoodooHog（属施法者阵营）
        curse = getattr(self, 'voodoo_curse', None)
        if curse and curse.get('name') and self.battle_state is not None:
            from card_utils import Card as _Card
            try:
                _Card(curse['name'])
                t = Troop(self.battle_state.next_entity_id, Position(self.position.x, self.position.y),
                          curse['player'], curse['name'])
                self.battle_state._spawn_entity(t)
            except Exception:
                pass
        self.battle_state.on_death(self)

    def update(self, dt):
        # This part may be a bit confusing because it doesn't check the `is_alive` and `deploy_delay_remaining` attribute.
        # Reasons: this will be eventually called by `super()` and won't terminate the actual update function. And
        # there are miner and drill that needs to be moving before it's even deployed. So this function only updates the buff_time
        # and debuff_time attribute.

        # I assume this function will be called after the deployment and alive check.
        self.entity_holder.on_tick(dt)
        if self.buff_time_remaining > 0:
            self.buff_time_remaining -= dt
        else:
            self.speed_buff = 1.0
            self.hit_speed_mult = 1.0
        if self.debuff_time_remaining > 0:
            self.debuff_time_remaining -= dt
        else:
            self.speed_debuff = 1.0
        # —— M2 族4：治疗导槽 / 减伤到期 / 能力冷却 ——
        if self.regen_buffs:
            still = []
            for rb in self.regen_buffs:
                rb['time'] -= dt
                if rb['time'] > 0:
                    self.hp = min(self.data.hp, self.hp + rb['hps'] * dt)
                    still.append(rb)
            self.regen_buffs = still
        if self.damage_reduction_timer > 0:
            self.damage_reduction_timer -= dt
            if self.damage_reduction_timer <= 0:
                self.damage_reduction = 0.0
        if self.ability_cd > 0:
            self.ability_cd = max(0.0, self.ability_cd - dt)
        # —— M4 族7：觉醒 fortify（脱战减伤，如觉醒骑士 damageReduction=60%）——
        # 独立槽位，不与 apply_buff 减伤（Monk 禅定）互相覆盖
        fortify = (self.evo or {}).get('buffWhenNotAttackingData') if self.evo else None
        if fortify:
            dr = fortify.get('damageReduction', 0) / 100.0
            # 「脱战」判定：距上次攻击超过 1s（攻击动画窗口，简化，待 L4）
            self._fortify_dr = dr if (self.battle_state.time - self.last_attack_time) > 1.0 else 0.0
        elif getattr(self, '_fortify_dr', 0.0):
            self._fortify_dr = 0.0

        for pending_damage in self.pending_damage:
            self.take_damage(pending_damage, delayed=False)
        self.pending_damage = []

        # —— M1 递增伤害：锁定同一目标时蓄力计时，换目标/脱锁即重置 ——
        if getattr(self.data, 'has_ramp', False):
            target = self.battle_state.entities.get(self.target_id) if self.target_id else None
            if target is None or target.player == self.player or not self.in_attack_range(target):
                self.ramp_target_id, self.ramp_timer, self.ramp_stage = None, 0.0, 0
            elif target.id != self.ramp_target_id:
                self.ramp_target_id, self.ramp_timer, self.ramp_stage = target.id, 0.0, 0
            else:
                self.ramp_timer += dt
                stage = 0
                for i, t in enumerate(self.data.ramp_stage_times):
                    if self.ramp_timer >= t:
                        stage = i + 1
                self.ramp_stage = stage

        # —— M4.5 动作链：Manual 递增序列——脱锁/脱攻击范围即重置到首档（与 M1 蓄力同语义）——
        if getattr(self, 'attack_seq', None) and self.attack_seq_mode is not None:
            _t = self.battle_state.entities.get(self.target_id) if self.target_id else None
            if _t is None or _t.player == self.player or not self.in_attack_range(_t):
                self.attack_seq_stage = 0

        # —— M4.5 动作链：多段命中排队（Berserker 连击）——每 hit_speed/n 打一段
        if getattr(self, 'attack_seq_pending', 0) > 0:
            self.attack_seq_hit_timer -= dt
            if self.attack_seq_hit_timer <= 0:
                self.attack_seq_pending -= 1
                self.attack_seq_hit_timer = self.data.hit_speed / max(len(self.attack_seq_damages), 1)
                _tid = getattr(self, 'attack_seq_target_id', None) or self.target_id
                _tgt = self.battle_state.entities.get(_tid) if _tid else None
                if _tgt is not None and _tgt.is_alive:
                    _d = self.attack_seq_damages[min(self.attack_seq_stage, len(self.attack_seq_damages) - 1)]
                    _tgt.take_damage(_d, delayed=True)

    def ramped_damage(self, base=None):
        """M1 递增伤害：按当前锁定阶段返回阶段伤害。
        数值模型（已对照 Fandom 地狱龙 35/120/422 验证）：
        阶段伤害 = variable_damageN(基准级绝对值) × (当前级伤害/基准级伤害)，间隔=hitSpeed 恒定
        M4.5 动作链：有攻击序列（attack_seq）时优先于时间蓄力——InfernoDragon_EV1 的
        离散四档（14/47/165/330）取代基础时间蓄力模型。"""
        if getattr(self, 'attack_seq', None):
            if self.attack_seq_damages:
                stage = min(self.attack_seq_stage, len(self.attack_seq_damages) - 1)
                return self.attack_seq_damages[stage] if self.attack_seq_damages[stage] else self.data.damage
            return base if base is not None else self.data.damage
        if not getattr(self.data, 'has_ramp', False) or self.ramp_stage == 0:
            return base if base is not None else self.data.damage
        idx = min(self.ramp_stage, len(self.data.ramp_stage_damages)) - 1
        return self.data.ramp_stage_damages[idx]

    # —— M4.5 动作链：攻击序列解析与推进 ——
    def _resolve_attack_seq(self, raw, mode=None):
        """挂载攻击序列（raw=[{damage,...},...]）。伤害按当前级等比缩放（基准=seq 首档 lv1 伤害）。"""
        if not raw:
            self.attack_seq = None
            self.attack_seq_mode = None
            self.attack_seq_damages = []
            self.attack_seq_stage = 0
            return
        self.attack_seq = raw
        self.attack_seq_mode = mode
        _d0 = None
        for _st in raw:
            if _st.get('damage'):
                _d0 = _st['damage']
                break
        # 缩放基准：data.damage 为当前级伤害；若基础伤害缺失（如 Berserker 伤害仅在序列内）
        # 或序列首档即为基准，则保持原值（scale=1，标注低置信度，待 per-level 表补齐）
        _scale = (self.data.damage / _d0) if (_d0 and self.data.damage) else 1.0
        self.attack_seq_damages = [(_st.get('damage') or 0) * _scale for _st in raw]
        self.attack_seq_stage = 0
        self.attack_seq_pending = 0
        self.attack_seq_hit_timer = 0.0

    def _on_attack_done(self, current_target=None):
        """攻击完成钩子（BasicCharacter.on_attack 末段调用）：
        Manual 序列 → 跨攻击推进一档（封顶末档）；无 mode 序列 → 单次攻击多段命中（Berserker 连击）。"""
        if not getattr(self, 'attack_seq', None):
            return
        if self.attack_seq_mode is not None:
            # 跨攻击递增：本次攻击已结算当前档，推进到下一档
            self.attack_seq_stage = min(self.attack_seq_stage + 1, len(self.attack_seq_damages) - 1)
        else:
            # 多段命中：ramped_damage 已打第一段，剩余段排队（记录目标，避免索敌状态干扰结算）
            n = len(self.attack_seq_damages)
            if n > 1:
                self.attack_seq_pending = n - 1
                self.attack_seq_hit_timer = 0.0
                self.attack_seq_target_id = current_target.id if (current_target is not None and hasattr(current_target, 'id')) else self.target_id


    def take_damage(self, amount: float, delayed=False):
        """Apply damage to entity"""
        if self.invincible: return
        dr = max(self.damage_reduction if self.damage_reduction_timer > 0 else 0.0,
                 getattr(self, '_fortify_dr', 0.0))
        if dr > 0:
            amount *= (1.0 - dr)
        if delayed:
            self.pending_damage.append(amount)
            return
        if not self.shield_health: self.hp -= amount
        else: self.shield_health = max(0, self.shield_health - amount)

        if self.hp <= 0 and self.is_alive:
            self.die()
            if self.data.death_damage:
                # I assume that all death damage deals attack to both air and ground troops.
                # The game data file hasn't specified what's the radius of the death damage,
                # so here I just set it to 1 tile
                self.battle_state.deal_area_damage(self.player, self.position, 1.0+self.data.collision_radius, self.data.death_damage,
                                                   attack_air=True, attack_ground=True)

    def in_attack_range(self, target):
        if target is None: return False
        if 'PrincessTower' in target.name:
            bonus = 0.5
        else:
            bonus = 0
        dist = self.position.distance_to(target.position)
        # M2 族3：建筑最小射程（Mortar 贴脸不攻击）
        if getattr(self.data, 'min_range', 0) and dist < self.data.min_range:
            return False
        return dist <= self.data.range + target.data.collision_radius + bonus
    def in_sight_range(self, target):
        if target is None: return False
        if 'PrincessTower' in target.name:
            bonus = 0.5
        else:
            bonus = 0
        return self.position.distance_to(target.position) <= self.data.sight_range + target.data.collision_radius + bonus

    def get_nearest_target(self):
        """Find nearest valid target with priority rules"""
        building_targets = []
        troop_targets = []

        for entity in list(self.battle_state.entities.values()):
            if not isinstance(entity, Troop) and not isinstance(entity, Building): continue
            if not entity.is_alive or entity.player == self.player: continue
            if not entity.targetable: continue
            distance = self.position.distance_to(entity.position)
            # M2 族3：最小射程内目标直接排除（Mortar 转火射程外目标，全被贴脸时待机）
            if getattr(self.data, 'min_range', 0) and distance < self.data.min_range: continue
            if (entity.data.is_air_unit and not self.data.attack_air) or ((not entity.data.is_air_unit) and not self.data.attack_ground):
                continue
            if self.in_sight_range(entity):
                if isinstance(entity, Building):
                    building_targets.append((distance, entity))
                elif not self.data.target_only_buildings:
                    troop_targets.append((distance, entity))
        closest_building = min(building_targets, key=lambda x: x[0])[1] if building_targets else None
        closest_troop = min(troop_targets, key=lambda x: x[0])[1] if troop_targets else None

        if self.data.target_only_buildings:
            targets = building_targets
        elif self.in_attack_range(closest_building) or self.in_attack_range(closest_troop):
            targets = troop_targets + building_targets
        else:
            targets = troop_targets if troop_targets else building_targets

        targets.sort(key=lambda x: x[0])
        if not targets: return None
        else: return targets[0][1]

    def _should_switch_target(self, current_target, new_target):
        """Determine if we should switch from current target to new target"""
        # if self.position.distance_to(new_target.position)-current_target.data.collision_radius < self.data.sight_range: return False
        if self.data.target_only_buildings and not isinstance(new_target, Building): return False
        if not new_target:
            return True
        # M2：防御建筑优先转火进入攻击范围的部队（现实行为：迫击炮/加农炮会停下打塔转而防御）
        if isinstance(self, Building) and isinstance(new_target, Troop) and not isinstance(current_target, Troop):
            if self.in_attack_range(new_target):
                return True
        if self.in_attack_range(current_target):
            return False
        # Always switch to troops in sight range (higher priority than buildings)
        is_current_building = isinstance(current_target, Building)
        is_new_troop = not isinstance(new_target, Building)
        if is_new_troop and is_current_building:
            return True
        if self.position.distance_to(current_target.position) > self.position.distance_to(new_target.position):
            return True
        return False

    def update_current_target(self):
        # If target is killed or no longer in sight, update the target_id to None
        current_target = None
        if self.target_id is None or \
                self.target_id not in self.battle_state.entities or \
                not self.battle_state.entities.get(self.target_id).is_alive:
            # doesn't have a valid prior target
            self.target_id = None
            self.path = []
        else:
            current_target = self.battle_state.entities.get(self.target_id)
            if not self.in_sight_range(current_target):
                if 'PrincessTower' not in current_target.name and 'KingTower' not in current_target.name:
                    self.path = []
                current_target = None
                self.target_id = None

        best_target = self.get_nearest_target()
        if self.target_id:
            if self._should_switch_target(self.battle_state.entities[self.target_id], best_target):
                current_target = best_target
                self.target_id = current_target.id if current_target else None
        else:
            current_target = best_target
            self.target_id = current_target.id if current_target else None

        # Now, the current target can still be None (example: a knight deployed at the back)
        # This case we update the target to the nearest enemy princess tower, so we can do A* globally!
        if self.target_id is None:
            min_distance = float('inf')
            self.target_id = 1
            for i in range(1, 7):
                if not self.battle_state.entities[i].is_alive: continue
                possible_princess_tower = self.battle_state.entities[i]
                if possible_princess_tower.player == self.player: continue
                distance = possible_princess_tower.position.distance_to(self.position) - possible_princess_tower.data.collision_radius
                if distance < min_distance:
                    min_distance = distance
                    self.target_id = i
            current_target = self.battle_state.entities[self.target_id]
        return current_target

    def create_projectile(self, target, damage_override=None):
        if not self.data.projectiles: raise Exception('Entity does not have any projectiles.')
        projectile = Projectile(
            id=self.battle_state.next_entity_id, position=Position(self.position.x, self.position.y),
            player=self.player, source_card_name=self.data.name, target=target,
            damage_override=damage_override, source=self)
        projectile.battle_state = self.battle_state
        self.battle_state.entities[projectile.id] = projectile
        self.battle_state.next_entity_id += 1

    def on_both_sides_of_river(self, e2):
        if isinstance(e2, Entity):
            y = e2.position.y
        else: y = e2.y
        if y < 15.0: return self.position.y > 17.0
        else: return self.position.y < 15.0

    def near_river(self):
        return abs(self.position.y-15.0)<self.data.collision_radius or abs(self.position.y-17.0)<self.data.collision_radius


class Troop(Entity):
    def __init__(self, id, position, player, card_name, battle_state=None, evolved=False):
        super().__init__(id, position, player, card_name, battle_state)
        self.deploy_delay_remaining = self.data.deploy_time
        self.name = self.data.name
        self.path_blocked_counter = 0
        self.jumping_across_river = False
        self.start_jumping_position = None
        self.spawned = False
        self.evo_hits = 0
        self.evo_extra_spawned = 0
        if evolved:
            self._apply_evolution()
        # —— M4.5 动作链：基础卡攻击序列（Berserker 三连击等，非觉醒形态）——
        if not self.attack_seq and getattr(self.data, 'attack_seq_raw', None):
            self._resolve_attack_seq(self.data.attack_seq_raw, self.data.attack_seq_mode)

    def _apply_evolution(self):
        """M4 族7：觉醒形态——数值按基础卡曲线推导 + 特殊机制字段挂载
        兼容两种数据格式：旧版 evolvedSpellsData.summonCharacterData 嵌套；
        新版扁平格式（source=ext，如 InfernoDragon_EV1）整个 evolvedSpellsData 即角色定义。"""
        from card_utils import characters, buildings
        evo_raw = self.data.evo_raw
        if not evo_raw: return
        evolved_scd = evo_raw.get('summonCharacterData') or {}
        if not evolved_scd.get('name') and evo_raw.get('name'):
            evolved_scd = evo_raw  # 扁平格式：顶层即角色定义
        stats = derive_evolved_stats(self.card_name, evolved_scd,
                                     self.data, characters, buildings, level=self.level)
        self.hp = stats.get('hp', self.hp)
        if stats.get('damage'):
            self.data.damage = stats['damage']
        if stats.get('shield_hitpoints'):
            self.shield_health = stats['shield_hitpoints']
        self.evo = stats
        # —— M4.5 动作链：觉醒形态攻击序列（InfernoDragon_EV1 四级递增 / ElectroDragon_EV1 动作组）——
        if stats.get('attackSequenceList'):
            self._resolve_attack_seq(stats['attackSequenceList'], stats.get('attackSequenceMode'))
        # —— onStartingActionData：出场动作（觉醒特斯拉出场眩晕脉冲；哥布林笼捕获从简）——
        osad = stats.get('onStartingActionData')
        if osad and 'Tesla' in str(osad.get('name', '')) and self.battle_state is not None:
            sd = osad.get('spawnDataData') or {}
            radius = (sd.get('maxRadius') or 6000) / 1000
            for entity in list(self.battle_state.entities.values()):
                if not entity.is_alive or entity.player == self.player: continue
                if isinstance(entity, (Projectile, SpawnProjectile, AreaEffect)): continue
                if entity.position.distance_to(self.position) <= radius:
                    entity.apply_buff(stun=0.5)  # 简化：出场脉冲半径内全部眩晕 0.5s（官方为链式，待 L4）

    def _evo_on_attack(self, target):
        """M4 族7：觉醒攻击后钩子（按字段族分发）"""
        evo = self.evo
        if not evo: return
        # ① 攻击序列 buff（buffAfterHitsData：觉醒骷髅分裂 / 觉醒蝙蝠自愈 / 觉醒野蛮人狂暴）
        buffs = evo.get('buffAfterHitsData') or []
        times = evo.get('buffAfterHitsTime') or [0] * len(buffs)
        for i, buff in enumerate(buffs):
            window = (times[i] if i < len(times) else 0) / 1000.0
            name = buff.get('name', '')
            if 'Duplication' in name:
                # 觉醒骷髅：每次攻击多生出 1 只，直到组上限（按实际存活组员计数）
                max_group = evo.get('groupMaxSize', 8)
                alive_group = sum(1 for e2 in self.battle_state.entities.values()
                                  if e2.card_name == self.card_name and e2.is_alive
                                  and e2.position.distance_to(self.position) < 5.0)
                if alive_group < max_group:
                    self.evo_extra_spawned += 1
                    self.battle_state.spawn_arrival_troops(self.card_name, 1,
                        Position(self.position.x + 0.3, self.position.y + 0.3), self.player)
            elif 'Heal' in name:
                hps = buff.get("healPerSecond", 30) * level_scale(self.level)  # 起始级基准 → 当前级
                self.apply_buff(heal={'hps': hps, 'time': window or 1.0})
            elif 'Rage' in name:
                self.apply_buff(speed_mult=buff.get('speedMultiplier', 135) / 100.0, duration=window or 3.0)
        # ② 觉醒弓箭手：二段箭（projectile2Data，双发射击）
        p2 = evo.get('projectile2Data')
        if p2 and p2.get('damage'):
            target.take_damage(p2["damage"] * level_scale(self.level), delayed=True)  # 84(起始级)→lv16 ≈351
        # ③ onAttackActionData：觉醒皇家巨人推击 / 觉醒女武神迷你龙卷
        oaad = evo.get('onAttackActionData')
        if oaad:
            sd = oaad.get('spawnDataData') or {}
            if 'PushBack' in oaad.get('name', ''):
                dmg = (sd.get("damage") or 32) * level_scale(self.level)
                radius = (sd.get('radius') or 3000) / 1000
                self.battle_state.deal_area_damage(self.player, self.position, radius, dmg, True, False)
                self.battle_state.push_enemies(self.player, self.position, radius, 1.0)
            elif 'Tornado' in oaad.get('name', ''):
                # 觉醒女武神：攻击时把周围敌人微微拉向自己（官方为 0.5s 迷你龙卷，此处按攻击同步简化）
                radius = (sd.get('radius') or 5500) / 1000
                self.battle_state.pull_enemies(self.player, self.position, radius, 1.0, dt=0.1)

    def _evo_on_death(self):
        """M4 族7：觉醒死亡钩子（觉醒瓦基丽亡语 / 觉醒哥布林囚笼亡语覆盖）"""
        evo = self.evo
        if not evo: return
        oka = evo.get('onKilledActionData') or {}
        sd = oka.get('spawnDataData')
        if sd and sd.get('hitpoints'):
            card = Card(sd['name'])
            for p in get_spawn_position(card, self.position, self.player):
                t = Troop(self.battle_state.next_entity_id, p, self.player, sd['name'])
                self.battle_state._spawn_entity(t)
        dsd = evo.get('deathSpawnCharacterData')
        if dsd and dsd.get('hitpoints') and dsd.get('name') not in ('GoblinBrawler',):
            # 觉醒囚笼亡语出兵（基础卡自身的 deathSpawn 走原路径，不重复）
            t = Troop(self.battle_state.next_entity_id, Position(self.position.x, self.position.y),
                      self.player, dsd['name'])
            self.battle_state._spawn_entity(t)

    def to_dict(self):
        d = super().to_dict()
        d.update({'type': 'troop', })
        return d

    def move_towards(self, position, dt: float, can_overshoot=False) -> None:
        dx, dy = position.x-self.position.x, position.y-self.position.y
        distance = math.hypot(dx, dy)
        if distance == 0: return
        if not can_overshoot:
            move_distance = min(self.speed * dt * self.speed_buff * self.speed_debuff, distance)
        else:
            move_distance = self.speed * dt * self.speed_buff * self.speed_debuff
        move_x, move_y = (dx / distance) * move_distance, (dy / distance) * move_distance
        self.position.x += move_x
        self.position.y += move_y

    def update(self, dt):
        if not self.is_alive: return
        # —— M2 族4：冰冻/眩晕（Zap 0.5s、Freeze 4s）：停移停攻，冷却一并暂停 ——
        if self.freeze_timer > 0:
            self.freeze_timer -= dt
            return
        if self.name == 'Miner':
            super().update(dt)
        if self.deploy_delay_remaining > 0:
            self.deploy_delay_remaining = max(0.0, self.deploy_delay_remaining - dt)
            return # Haven't finished deploying yet
        # Logic: the troop may have a current target (or doesn't), and `get_nearest_target` also gives a
        # recommended target. If current target exists, compare that with the recommendation to see
        # if it needs to switch. If it doesn't exist, use the best target. However, the best target may also
        # be none.
        if self.name != 'Miner':
            super().update(dt)
        # The miner needs to update before deployment.
        if self.jumping_across_river and self.on_both_sides_of_river(self.start_jumping_position):
            self.jumping_across_river = False
            self.data.is_air_unit = Card(self.name).is_air_unit
            self.speed = self.data.speed
        current_target = self.update_current_target()
        # —— M2 族5：钩拉位移（渔夫钩）：直线拽向钩点，期间不攻击、无视河道 ——
        if self.hook_pull:
            pull = self.hook_pull
            pull['time'] -= dt
            dist = math.hypot(pull['x']-self.position.x, pull['y']-self.position.y)
            if dist > 0.15 and pull['time'] > 0:
                step = min(pull['speed']*dt, dist)
                self.position.x += (pull['x']-self.position.x)/dist*step
                self.position.y += (pull['y']-self.position.y)/dist*step
                self.attack_cooldown = max(self.data.hit_speed-self.data.load_time, self.attack_cooldown-dt)
                return
            self.hook_pull = None
        # After the modification, we always have a target, sometimes it's in sight range, sometimes it's not
        # We use A* search for all cases to pathfind towards the target.
        # The case is even the same with ground troops and air troops.

        # Move towards target if out of attack range
        if (not self.in_attack_range(current_target)) or self.jumping_across_river:
            has_jump_ability = self.data.jump_speed and self.on_both_sides_of_river(current_target) and self.near_river() and self.in_sight_range(current_target)
            if not self.jumping_across_river and has_jump_ability:
                self.start_jumping_position = Position(self.position.x, self.position.y)
                self.jumping_across_river = True
                self.data.is_air_unit = True
                self.speed = self.data.jump_speed
            if self.data.is_air_unit:
                self.move_towards(current_target.position, dt, True)
            else:
                if not self.path:
                    self.path = EntityPathfinder(self, current_target, self.battle_state).calculate()
                elif self.in_sight_range(current_target) and self.battle_state.tick % 10 == 0:
                    self.path = EntityPathfinder(self, current_target, self.battle_state).calculate()

                # determine the next waypoint and move towards that waypoint
                min_point = min(self.path, key=lambda pos: pos.distance_to(self.position))
                index = self.path.index(min_point)
                start_vector = (self.position.x-self.path[0].x, self.position.y-self.path[0].y)
                close_vector = (self.position.x-min_point.x, self.position.y-min_point.y)
                dot = start_vector[0]*close_vector[0] + start_vector[1]*close_vector[1]
                if dot >= 0:
                    # move towards next waypoint
                    index += 1
                if index == len(self.path):
                    self.move_towards(current_target.position, dt, True)
                else:
                    self.move_towards(self.path[index], dt, True)
            self.attack_cooldown = max(self.data.hit_speed-self.data.load_time, self.attack_cooldown-dt*self.speed_buff*self.speed_debuff*self.hit_speed_mult)
        else:
            if self.attack_cooldown <= 0:
                self.entity_holder.on_attack(current_target)
            else:
                self.attack_cooldown -= dt*self.speed_buff*self.speed_debuff*self.hit_speed_mult



class Building(Entity):
    def __init__(self, id, position, player, card_name, persistent=False, evolved=False):
        super().__init__(id, position, player, card_name)
        self.deploy_delay_remaining = self.data.deploy_time
        self.lifetime_elapsed = 0.0
        self.target_id = None
        self.tower_active = False
        self.persistent = persistent
        self.name = self.data.name
        self.evo_hits = 0
        self.evo_extra_spawned = 0
        if evolved and self.data.evo_raw:
            # 觉醒建筑（ Cannon/Mortar/Tesla ）：数值按曲线推导；出场脉冲延迟到首次 update（battle_state 届时才挂上）
            from card_utils import characters, buildings
            _evo_raw = self.data.evo_raw
            _evo_scd = _evo_raw.get('summonCharacterData') or {}
            if not _evo_scd.get('name') and _evo_raw.get('name'):
                _evo_scd = _evo_raw  # 扁平格式兼容
            stats = derive_evolved_stats(self.card_name, _evo_scd,
                                         self.data, characters, buildings, level=self.level)
            self.hp = stats.get('hp', self.hp)
            self.evo = stats
            self._evo_starting_pending = bool(stats.get('onStartingActionData'))

    def to_dict(self):
        d = super().to_dict()
        d.update({'type': 'building'})
        return d

    def take_damage(self, amount: float, delayed=False):
        super().take_damage(amount, delayed)
        if self.data.name == 'KingTower' and not self.tower_active:
            self.tower_active = True

    def update(self, dt: float):
        """Update building - only attack, no movement"""
        if not self.is_alive: return
        if getattr(self, '_evo_starting_pending', False):
            # M4 觉醒建筑出场动作（特斯拉出场眩晕脉冲）
            self._evo_starting_pending = False
            osad = (self.evo or {}).get('onStartingActionData') or {}
            if 'Tesla' in str(osad.get('name', '')):
                sd = osad.get('spawnDataData') or {}
                radius = (sd.get('maxRadius') or 6000) / 1000
                for entity in list(self.battle_state.entities.values()):
                    if not entity.is_alive or entity.player == self.player: continue
                    if isinstance(entity, (Projectile, SpawnProjectile, AreaEffect)): continue
                    if entity.position.distance_to(self.position) <= radius:
                        entity.apply_buff(stun=0.5)  # 简化：链式眩晕待 L4
        if self.data.name == 'KingTower' and not self.tower_active: return
        if self.deploy_delay_remaining > 0:
            self.deploy_delay_remaining = max(0.0, self.deploy_delay_remaining - dt)
            return
        # —— M2 族4：建筑同样受冰冻（Freeze/Tesla 脉冲）——
        if self.freeze_timer > 0:
            self.freeze_timer -= dt
            return
        super().update(dt)
        if self.data.lifetime > 0 and not self.persistent:
            # M2 修复：此前同一帧 take_damage 两次（衰减速率 ×2）
            decay = (self.data.hp / float(self.data.lifetime)) * dt
            self.take_damage(decay)
        if self.attack_cooldown > 0:
            self.attack_cooldown = max(0, self.attack_cooldown-dt*self.speed_buff*self.speed_debuff*self.hit_speed_mult)
        target = self.update_current_target()
        if target and self.in_attack_range(target) and self.attack_cooldown <= 0:
            if self.data.projectiles:
                # M1: 建筑（地狱塔类）的蓄力阶段对弹道伤害生效
                self.create_projectile(target, damage_override=self.ramped_damage(self.data.projectile_data.damage))
            else:
                target.take_damage(self.ramped_damage(self.data.damage))
            self.attack_cooldown = self.data.hit_speed

class Projectile(Entity):
    def __init__(self, id, position, player, source_card_name, target, homing=True, battle_state=None, damage_override=None, source=None):
        super().__init__(id, position, player, source_card_name)
        self.target_position = Position(target.position.x, target.position.y)
        self.initial_position = Position(self.position.x, self.position.y)
        self.proj = self.data.projectile_data # a shortcut
        self.rolling = bool(self.proj.roll_range)
        self.homing = homing
        self.target = target
        self.battle_state = battle_state
        self.name = self.proj.name
        self.source = source  # M3: 发射者实体引用（Monk 反弹需要）
        if self.data.type == 'spell':
            self.data.collision_radius = self.proj.radius
        else: self.data.collision_radius = 0.3

        self.damage_dealt = []
        self.damage_override = damage_override  # M1: 蓄力建筑/生成链的伤害覆盖值

    def _damage(self):
        """实际伤害：蓄力建筑/生成链可传入覆盖值"""
        return self.proj.damage if self.damage_override is None else self.damage_override

    def _arrival_direction(self):
        dx, dy = self.target_position.x-self.initial_position.x, self.target_position.y-self.initial_position.y
        n = math.hypot(dx, dy) or 1.0
        return (dx/n, dy/n)

    def _on_arrive(self):
        """M1：到达目标点——结算伤害/buff，并触发弹道生成链（二段弹/落地出兵）"""
        if not self.proj.radius:
            tgt = self.target
            # —— M3 族6：Monk 禅定反弹（投射物 → 反弹给发射者；来源已死 → 最近公主塔）——
            if tgt is not None and getattr(tgt, 'deflect_active', False):
                src = getattr(self, 'source', None)
                dmg = self._damage()
                if isinstance(src, Entity) and src.is_alive and not isinstance(src, Projectile):
                    src.take_damage(dmg)
                else:
                    self.battle_state.reflect_to_tower(tgt, dmg)
                impact = self.target_position
                self._chain(impact)
                return
            if tgt is not None and hasattr(tgt, 'take_damage'):
                tgt.take_damage(self._damage())
            if self.proj.buff_time and hasattr(self.target, 'speed_debuff'):
                self.target.speed_debuff = min(1 + self.proj.target_buff.get('speedMultiplier', 0) / 100, self.target.speed_debuff)
                self.target.debuff_time_remaining = self.proj.buff_time
                # 女巫妈妈诅咒：标记目标，死亡时生成 VoodooHog（属施法者阵营）
                if getattr(self.proj, 'target_buff_death_spawn', None):
                    self.target.voodoo_curse = {'player': self.player, 'name': self.proj.target_buff_death_spawn.get('name')}
        else:
            self._deal_splash_damage()
        impact = self.target_position
        self._chain(impact)

    def _chain(self, impact):
        """M1 弹道生成链：二段弹 / 落地出兵"""
        sp = getattr(self.proj, 'spawn_projectile', None)
        if sp:
            self.battle_state.spawn_projectile_chain(sp, impact, self.player, self._arrival_direction())
        sc = getattr(self.proj, 'spawn_characters', None)
        if sc:
            self.battle_state.spawn_arrival_troops(sc[1], sc[0], impact, self.player)

    def to_dict(self):
        d = super().to_dict()
        d.update({'type': 'projectile'})
        return d

    def update(self, dt):
        """Update projectile - move towards target"""
        if not self.is_alive: return
        if self.rolling:
            distance = self.position.distance_to(self.initial_position)
            if distance > self.proj.roll_range:
                self.is_alive = False
                return
            # now deal area damage
            for each in list(self.battle_state.entities.values()):
                if type(each).__name__ in {'Projectile', 'SpawnProjectile', 'RollingProjectile', 'AreaEffect',
                                              'TimedExplosive'}: continue  # exclude spells or stealth entities
                if each in self.damage_dealt or each.data.is_air_unit: continue
                if not each.is_alive or each.player == self.player: continue
                if each.position.distance_to(self.position) < each.data.collision_radius + self.proj.radius:
                    each.take_damage(self._damage(), delayed=True)
                    self.damage_dealt.append(each)
                    # now knockback
                    direction_vector = complex(each.position.x-self.position.x, each.position.y-self.position.y)
                    direction_vector /= abs(direction_vector)
                    direction_vector *= self.proj.pushback
                    if isinstance(each, Troop):
                        new_x = each.position.x + direction_vector.real
                        new_y = each.position.y + direction_vector.imag
                        if self.battle_state.ground_walkable(Position(new_x, new_y), each.data.collision_radius):
                            each.position = Position(new_x, new_y)
            direction_vector = complex(self.target_position.x-self.initial_position.x,
                                       self.target_position.y-self.initial_position.y)
            direction_vector /= abs(direction_vector)
            direction_vector *= self.proj.speed * dt
            self.position.x += direction_vector.real
            self.position.y += direction_vector.imag
            return

        target_position_final = self.target_position if not self.homing else self.target.position
        distance = self.position.distance_to(target_position_final)
        if distance <= self.proj.speed * dt:
            self._on_arrive()
            self.is_alive = False
        else:
            self._move_towards(target_position_final, dt)

    def _deal_splash_damage(self) -> None:
        """Deal damage to entities in splash radius using hitbox overlap detection"""
        # —— M3 族6：法术类溅射命中 Monk 禅定 → 整个法术反弹至最近敌方公主塔（官方规则）——
        for entity in list(self.battle_state.entities.values()):
            if (entity.is_alive and getattr(entity, 'deflect_active', False)
                    and entity.player != self.player
                    and entity.position.distance_to(self.target_position) <= self.proj.radius + entity.data.collision_radius):
                self.battle_state.reflect_to_tower(entity, self._damage())
                return
        for entity in list(self.battle_state.entities.values()):
            if entity.invincible: continue
            if entity.player == self.player or not entity.is_alive: continue
            if entity.data.is_air_unit and not self.proj.hits_air: continue
            if (not entity.data.is_air_unit) and not self.proj.hits_ground: continue

            # Use hitbox-based collision detection for more accurate splash damage
            if entity.position.distance_to(self.target_position) <= (self.proj.radius + entity.data.collision_radius):
                base = self._damage()
                amount_dealt = base if "King" not in entity.name else round(base * self.proj.crown_tower_percent)
                entity.take_damage(amount_dealt)
                if self.proj.buff_time:
                    entity.speed_debuff = min(1 + self.proj.target_buff.get('speedMultiplier', 0) / 100, entity.speed_debuff)
                    entity.debuff_time_remaining = self.proj.buff_time
                    if getattr(self.proj, 'target_buff_death_spawn', None):
                        entity.voodoo_curse = {'player': self.player, 'name': self.proj.target_buff_death_spawn.get('name')}

    def _move_towards(self, target_pos, dt):
        """Move towards target position"""
        # Note: I used a much cleaner way of writing the code.
        direction = complex(target_pos.x - self.position.x, target_pos.y - self.position.y)
        step = direction / abs(direction) * self.proj.speed * dt
        self.position.x += step.real
        self.position.y += step.imag


class _ProjectileShim:
    """M1 二段弹的 data 兼容垫片：提供下游代码访问的最小属性集（无卡牌身份）"""
    def __init__(self, proj):
        self.proj = proj
        self.type = 'projectile'
        self.name = proj.name
        self.collision_radius = 0.3
        self.is_air_unit = False
        self.death_damage = 0
        self.hp = 0
        self.range = 0
        self.elixir = 0
        self.tower_damage_mult = 1.0
        self.area_damage_radius = 0
        self.shield_health = 0


class SpawnProjectile(Projectile):
    """M1 弹道生成链的二段弹（如烟花射手的爆裂弹）。
    从数值表行构建的 wrapper 直接驱动，有意跳过 Entity.__init__（无卡牌身份可查），
    手工补齐下游代码访问的最小状态；invincible 使其在任何伤害结算中被跳过。"""
    def __init__(self, id, position, player, proj_wrapper, target_position, battle_state):
        self.id, self.position, self.player = id, position, player
        self.is_alive = True
        self.targetable = False
        self.invincible = True
        self.battle_state = battle_state
        self.proj = proj_wrapper
        self.name = proj_wrapper.name
        self.card_name = proj_wrapper.name
        self.homing = False
        self.target = BlankEntity(target_position)
        self.target_position = Position(target_position.x, target_position.y)
        self.initial_position = Position(position.x, position.y)
        self.damage_override = None
        self.damage_dealt = []
        self.pending_damage = []
        self.rolling = False
        self.jumping_across_river = False
        self.path = []
        self.shield_health = 0
        self.hp = 0
        self.data = _ProjectileShim(proj_wrapper)

    def to_dict(self):
        return {'type': 'projectile', 'card_name': self.name, 'player': self.player,
                'x': self.position.x, 'y': self.position.y, 'hp': 0, 'max_hp': 0,
                'shield_max_hp': 0, 'shield_hp': 0, 'collision_radius': 0.3}

    def die(self): self.is_alive = False

    def update(self, dt): Projectile.update(self, dt)




class _EffectShim:
    """AreaEffect 的 data 兼容垫片"""
    def __init__(self, radius):
        self.type = 'area_effect'
        self.name = 'AreaEffect'
        self.collision_radius = radius
        self.is_air_unit = False
        self.death_damage = 0
        self.hp = 0
        self.range = 0
        self.elixir = 0
        self.tower_damage_mult = 1.0
        self.area_damage_radius = 0
        self.shield_health = 0


class AreaEffect(Entity):
    """M2 族4/族5：瞬发区域法术实体（Zap/Freeze/Heal/Rage/Tornado/Earthquake/Poison）。
    由 cards_stats_spell 行驱动（radius/life_duration/hit_speed/buff_data），官方 lv11 数值经
    evolutions.OFFICIAL_OVERRIDES 覆盖。此前这些法术在引擎里是「无行为隐形实体」。
    注意：刻意不走 Entity.__init__（否则同名机制类如 Rage 会在 battle_state 挂载前执行钩子）。"""
    def __init__(self, id, position, player, card_name, battle_state=None):
        self.id, self.position, self.player = id, position, player
        self.card_name = card_name
        self.name = card_name
        self.battle_state = battle_state
        self.is_alive = True
        self.targetable = False
        self.invincible = True
        self.path = []
        self.pending_damage = []
        self.jumping_across_river = False
        self.hp = 0
        self.shield_health = 0
        row = spells.get(card_name, {})
        ov = OFFICIAL_OVERRIDES.get(card_name, {})
        bd = row.get('buff_data') or {}
        self.radius = (row.get('radius') or 3000) / 1000
        self.lifetime = ov.get('duration', (row.get('life_duration') or 1000) / 1000)
        self.tick = ov.get('tick', (row.get('hit_speed') or 500) / 1000) or 0.5
        self.tick_timer = 0.0
        self.only_enemies = bool(row.get('only_enemies'))
        self.only_own_troops = bool(row.get('only_own_troops'))
        self.ignore_buildings = bool(row.get('ignore_buildings'))
        self.controls = bool(row.get('controls_buff'))
        self.buff_name = row.get('buff')
        self.buff_time = (row.get('buff_time') or 0) / 1000
        self.stun_applied = False  # Freeze 类：整场仅施加一次（总时长从施法起算）
        # —— 11-16 级全支持：法术伤害按战斗卡牌等级缩放（此前硬编码 lv11）——
        lv = Card.default_level
        row_rar = row.get('rarity') or 'Common'
        dpl = row.get('damage_per_level') or []
        if ov.get('damage_per_tick_lv11') is not None:
            self.damage_per_tick = ov['damage_per_tick_lv11'] * (1.1 ** (lv - 11))
        elif ov.get('damage_lv11') is not None:
            self.damage_per_tick = ov['damage_lv11'] * (1.1 ** (lv - 11))
        elif dpl:
            self.damage_per_tick = _value_at_level(dpl, row_rar, lv, row.get('damage') or 0)
        elif row.get('damage'):
            self.damage_per_tick = row['damage']
        else:
            # DOT 型（Earthquake/Poison）：buff_data.damage_per_second 为起始级基准，按统一曲线放大
            self.damage_per_tick = (bd.get('damage_per_second') or 0) * self.tick * level_scale(lv)
        self.crown_pct = ov.get('crown_tower_percent',
                                (row.get('crown_tower_damage_percent') or 0) + 100) / 100 \
            if (ov.get('crown_tower_percent') or row.get('crown_tower_damage_percent')) else None
        self.heal_per_tick = ov.get('heal_per_tick_lv11', 0) * (1.1 ** (lv - 11))
        self.data = _EffectShim(self.radius)
        self.entity_holder = BasicCharacter(self)  # 无机制钩子的空 holder（data 先于 holder）

    def _in_radius(self, e):
        return e.position.distance_to(self.position) <= self.radius + e.data.collision_radius

    def _pulse(self):
        ov = OFFICIAL_OVERRIDES.get(self.card_name, {})
        for e in list(self.battle_state.entities.values()):
            if not e.is_alive or isinstance(e, (Projectile, SpawnProjectile, AreaEffect)): continue
            if self.only_enemies and e.player == self.player: continue
            if self.only_own_troops and e.player != self.player: continue
            if self.ignore_buildings and isinstance(e, Building): continue
            if not self._in_radius(e): continue
            # —— buff 分发 ——
            bn = self.buff_name
            if bn == 'ZapFreeze':
                e.apply_buff(stun=self.buff_time, retarget=True)   # 眩晕 + 重索敌（官方）
            elif bn == 'Freeze':
                # 冻结总时长 = 4s（从施法起算）：整场冰冻只施加一次，重复 pulse 不刷新
                if not self.stun_applied:
                    e.apply_buff(stun=self.buff_time)
                    self.stun_applied = True
            elif bn == 'Rage':
                # 官方现行：+30%、效果残留 1s（2025/10 调整），覆盖快照旧值 +35%
                e.apply_buff(speed_mult=ov.get('speed_mult', 1.30), duration=ov.get('residue', 1.0))
            elif bn == 'Heal':
                if self.heal_per_tick:
                    e.apply_buff(heal={'hps': self.heal_per_tick / self.tick, 'time': self.tick})
            # Tornado/Earthquake/Poison 的 buff 仅表现层，无属性效果
            # —— 伤害 ——
            if self.damage_per_tick:
                if isinstance(e, Building) and 'King' in e.name and self.crown_pct is not None:
                    e.take_damage(self.damage_per_tick * self.crown_pct)
                else:
                    e.take_damage(self.damage_per_tick)

    def update(self, dt):
        if not self.is_alive: return
        # —— M2 族5：Tornado 拉拽（官方拉力 360%，质量抵抗建模待 L4；此处用收敛模型：
        # 剩余时间内匀速拉到中心，冲锋单位抗性从简）——
        if self.controls:
            for e in list(self.battle_state.entities.values()):
                if not isinstance(e, Troop) or not e.is_alive or e.player == self.player: continue
                dist = e.position.distance_to(self.position)
                if dist > self.radius or dist < 0.05: continue
                speed = dist / max(self.lifetime, 0.1)
                step = min(speed * dt, dist)
                nx = e.position.x + (self.position.x - e.position.x) / dist * step
                ny = e.position.y + (self.position.y - e.position.y) / dist * step
                if self.battle_state.arena.is_walkable(Position(nx, ny)):
                    e.position.x, e.position.y = nx, ny
        # 结算先行于寿命判定：Zap 类 life_duration=1ms 的瞬发法术也要打满一次 pulse
        self.tick_timer -= dt
        if self.tick_timer <= 0:
            self.tick_timer = self.tick
            self._pulse()
        self.lifetime -= dt
        if self.lifetime <= 0:
            self.is_alive = False

    def to_dict(self):
        return {'type': 'area_effect', 'card_name': self.card_name, 'player': self.player,
                'x': self.position.x, 'y': self.position.y, 'hp': 0, 'max_hp': 0,
                'shield_max_hp': 0, 'shield_hp': 0, 'collision_radius': self.radius}


class _BombShim:
    """GenericBomb 的 data 兼容垫片"""
    def __init__(self, radius):
        self.type = 'bomb'
        self.name = 'GenericBomb'
        self.collision_radius = radius
        self.is_air_unit = False
        self.death_damage = 0
        self.hp = 0
        self.range = 0
        self.elixir = 0
        self.tower_damage_mult = 1.0
        self.area_damage_radius = 0
        self.shield_health = 0


class GenericBomb(Entity):
    """M3：可编程定时炸弹（Mighty Miner 能力）。显式伤害/半径/延迟/击退，不可被攻击不可被选取。"""
    def __init__(self, id, position, player, damage, radius, delay, knockback=0.0):
        self.id, self.position, self.player = id, position, player
        self.is_alive = True
        self.targetable = False
        self.invincible = True
        self.battle_state = None
        self.damage, self.radius, self.delay, self.knockback = damage, radius, delay, knockback
        self.card_name = 'GenericBomb'
        self.name = 'GenericBomb'
        self.data = _BombShim(radius)
        self.pending_damage = []
        self.path = []
        self.jumping_across_river = False
        self.shield_health = 0
        self.hp = 0

    def update(self, dt):
        if not self.is_alive: return
        self.delay -= dt
        if self.delay > 0: return
        for e in list(self.battle_state.entities.values()):
            if not e.is_alive or e.player == self.player: continue
            if isinstance(e, (Projectile, SpawnProjectile, AreaEffect, GenericBomb)): continue
            if e.position.distance_to(self.position) <= self.radius + e.data.collision_radius:
                e.take_damage(self.damage)
                if self.knockback:
                    d = e.position.distance_to(self.position)
                    if d > 0.05:
                        nx = e.position.x + (e.position.x - self.position.x) / d * self.knockback
                        ny = e.position.y + (e.position.y - self.position.y) / d * self.knockback
                        if self.battle_state.ground_walkable(Position(nx, ny), e.data.collision_radius) or e.data.is_air_unit:
                            e.position.x, e.position.y = nx, ny
        self.is_alive = False

    def take_damage(self, amount): pass

    def to_dict(self):
        return {'type': 'bomb', 'card_name': self.name, 'player': self.player,
                'x': self.position.x, 'y': self.position.y, 'hp': 0, 'max_hp': 0,
                'shield_max_hp': 0, 'shield_hp': 0, 'collision_radius': self.radius}


class TimedExplosive(Entity):
    def __init__(self, id, position, player, card_name):
        super().__init__(id, position, player, card_name)
        self.dsd = TimedExplosiveData(self.data.death_spawn_data)
        self.deploy_delay_remaining = self.dsd.deploy_time
        self.name = self.dsd.name

    def update(self, dt):
        if not self.is_alive: return
        if self.deploy_delay_remaining > 0:
            self.deploy_delay_remaining = max(0.0, self.deploy_delay_remaining - dt)
            return
        for entity in list(self.battle_state.entities.values()):
            if not entity.is_alive or entity.player == self.player: continue
            if entity.position.distance_to(self.position) - entity.data.collision_radius < self.dsd.range:
                if entity.name in ('King_PrincessTowers', 'KingTower'):
                    entity.take_damage(self.dsd.damage*self.dsd.crown_tower_damage_percent)
                else:
                    entity.take_damage(self.dsd.damage)
        self.is_alive = False

    def take_damage(self, amount: float):
        # Bombs does not take damage!
        pass


def get_spawn_position(card_info, position, player, offset_angle=True):
    spawn_number, spawn_delay, r = card_info.spawn_number, card_info.spawn_delay, card_info.spawn_radius
    if spawn_number == 1: return [Position(position.x, position.y)]
    positions = []
    angle_offset = {2: 0, 3: math.pi/2, 4: math.pi/4, 6: 0}
    for i in range(spawn_number):
        angle = 2*math.pi*i/spawn_number
        if offset_angle: angle += angle_offset.get(spawn_number, 0)
        if player == 1: angle += math.pi
        dx, dy = r*math.cos(angle), r*math.sin(angle)
        positions.append(Position(position.x+dx, position.y+dy))
    return positions


class BattleState:
    def __init__(self, player_0: PlayerState, player_1: PlayerState, card_level=None):
        self.card_level = card_level if card_level is not None else Card.default_level
        Card.default_level = self.card_level  # 战斗内所有 Card() 构造继承该等级（单战斗串行）
        self.entities = {}
        self.players = [player_0, player_1]
        self.arena = TileGrid()
        self.time = 0.0
        self.tick = 0
        self.game_over = False
        self.winner = None
        self.next_entity_id = 1
        self.regen = 2.8

        self._spawn_entity(Building(1, self.arena.RED_LEFT_TOWER, 1, 'King_PrincessTowers', True))
        self._spawn_entity(Building(2, self.arena.RED_RIGHT_TOWER, 1, 'King_PrincessTowers', True))
        self._spawn_entity(Building(3, self.arena.BLUE_LEFT_TOWER, 0, 'King_PrincessTowers', True))
        self._spawn_entity(Building(4, self.arena.BLUE_RIGHT_TOWER, 0, 'King_PrincessTowers', True))
        self._spawn_entity(Building(5, self.arena.RED_KING_TOWER, 1, 'KingTower', True))
        self._spawn_entity(Building(6, self.arena.BLUE_KING_TOWER, 0, 'KingTower', True))

        self.schedule = []
        self.building_positions = []
        self.building_cache = None
        self.cache_fresh = False
        self.souls = [0, 0]  # M3：Skeleton King 灵魂计数（场上任意部队死亡 +1，上限 10）

    def in_river(self, position):
        river_tiles = [(0, 15), (0, 16), (1, 15), (1, 16),
            *[(i, j) for i in range(5, 13) for j in range(15, 17)], # (5, 15) to (12, 16)
            (16, 15), (16, 16), (17, 15), (17, 16)]
        return (int(position.x), int(position.y)) in river_tiles

    def ensure_walkability(self, entity):
        if entity.jumping_across_river and self.in_river(entity.position): return
        if isinstance(entity, Building) or isinstance(entity, Projectile): return
        if isinstance(entity, (SpawnProjectile, AreaEffect, GenericBomb)): return  # M2/M3：静态效果实体不参与走位修正

        if not self.ground_walkable(entity.position, entity.data.collision_radius):

            x, y, r = entity.position.x, entity.position.y, entity.data.collision_radius
            push_ratio = 0.5
            if y < push_ratio*r: y=push_ratio*r
            elif y > 32-push_ratio*r: y=32-push_ratio*r
            if x < push_ratio*r: x=r
            elif x > 18-push_ratio*r: x=18-push_ratio*r
            if 15-push_ratio*r < y < 17+push_ratio*r and not entity.data.is_air_unit:
                y = 15-push_ratio*r if y-15 < 17-y else 17+push_ratio*r
            entity.position.x = x
            entity.position.y = y

    def _spawn_entity(self, entity):
        self.ensure_walkability(entity)
        entity.battle_state = self
        entity.id = self.next_entity_id
        self.entities[self.next_entity_id] = entity
        self.next_entity_id += 1

    def _wrap(self, entity_data):
        card_name = entity_data[3]
        entity_data = list(entity_data)
        entity_data[0] = self.next_entity_id
        self.next_entity_id += 1
        evolved = False
        if len(entity_data) == 6:  # M4：第 6 位 = 觉醒形态标记
            evolved = bool(entity_data[5])
            entity_data = entity_data[:5]
        if len(entity_data) == 7:
            return Projectile(*entity_data)
        if card_name in spells:
            return Entity(*entity_data)
        elif card_name in buildings:
            self.cache_fresh = False
            # M2 修复：此前把 battle_state 对象传入 persistent 槽（恒真值 → 部署建筑寿命衰减失效）
            return Building(entity_data[0], entity_data[1], entity_data[2], entity_data[3],
                            persistent=False, evolved=evolved)
        else:
            return Troop(entity_data[0], entity_data[1], entity_data[2], entity_data[3],
                         entity_data[4], evolved=evolved)

    def delayed_spawn(self, entity, delay):
        if delay:
            self.schedule.append((entity, self.time+delay))
        else:
            self._spawn_entity(self._wrap(entity))

    def update_player_hp(self):
        p0, p1 = self.players
        p0.king_tower_hp = self.entities[6].hp
        p0.left_tower_hp = self.entities[3].hp
        p0.right_tower_hp = self.entities[4].hp
        p1.king_tower_hp = self.entities[5].hp
        p1.left_tower_hp = self.entities[1].hp
        p1.right_tower_hp = self.entities[2].hp

    def step(self, dt):
        if self.game_over: return
        self.update_player_hp()
        p0 = self.players[0].get_crown_count()
        p1 = self.players[1].get_crown_count()
        p0h = self.players[0]
        p1h = self.players[1]
        if p0 == 3:
            self.game_over = True
            self.winner = 1
            return
        elif p1 == 3:
            self.game_over = True
            self.winner = 0
            return
        elif 300>self.time >= 180:
            if p0 > p1:
                self.game_over = True
                self.winner = 1
                return
            elif p0 < p1:
                self.game_over = True
                self.winner = 0
                return
        elif self.time >= 300:
            self.game_over = True
            min_0_hp = min(each for each in (p0h.king_tower_hp, p0h.left_tower_hp, p0h.right_tower_hp) if each > 0)
            min_1_hp = min(each for each in (p1h.king_tower_hp, p1h.left_tower_hp, p1h.right_tower_hp) if each > 0)
            if min_0_hp > min_1_hp:
                self.winner = 0
            else:
                self.winner = 1
        for each in self.players:
            each.regenerate_elixir(dt, 2.8 if self.time < 120 else 1.4 if self.time < 240 else 2.8/3)
        self.entities = {key:value for key,value in self.entities.items() if (value.is_alive or key <= 6)}
        self.building_positions = [(entity.position.x, entity.position.y, entity.data.collision_radius) for entity in self.entities.values() if isinstance(entity, Building)]
        if not self.cache_fresh:
            self.calculate_building_cache()
            self.cache_fresh = True
        for entity in list(self.entities.values()):
            entity.update(dt)
            self.ensure_walkability(entity)
        self.resolve_collisions()

        for entity, spawn_time in self.schedule:
            if self.time >= spawn_time: self._spawn_entity(self._wrap(entity))
        self.schedule = [each for each in self.schedule if each[1] > self.time]
        self.time += dt
        self.tick += 1

    def _finish_deploy(self, player_id, card_name, from_mirror=False):
        """M1：出牌收尾——扣费并更新卡序；镜像重放的卡不在手牌中，手动送回循环末尾"""
        p = self.players[player_id]
        if from_mirror:
            if card_name in p.cycle:
                p.cycle.remove(card_name)
                p.cycle.append(card_name)
        else:
            p.play_card(card_name)

    def spawn_projectile_chain(self, projectile_name, position, player, direction):
        """M1 弹道生成链：按名称从数值表构建二段弹（FirecrackerProjectile→FirecrackerExplosion）"""
        row = projectiles.get(projectile_name)
        if not row: return
        wrapper = projectile_from_row(row)
        count = max(1, int(wrapper.spawn_count or 1))
        base_angle = math.atan2(direction[1], direction[0]) if direction else 0.0
        travel = wrapper.roll_range or 2.0   # 二段弹飞行距离（假设：无数据时 2 tile，待 L4）
        for i in range(count):
            a = base_angle if count == 1 else base_angle + (i - (count - 1) / 2) * 0.5
            tgt = Position(position.x + math.cos(a) * travel, position.y + math.sin(a) * travel)
            sp = SpawnProjectile(self.next_entity_id, Position(position.x, position.y),
                                 player, wrapper, tgt, self)
            self.entities[sp.id] = sp
            self.next_entity_id += 1

    def spawn_arrival_troops(self, card_name, count, position, player):
        """M1 落地出兵（哥布林飞桶类）：弹道到达后在落点部署 count 个单位"""
        info = Card(card_name)
        info.spawn_number = count
        info.spawn_delay = 0
        for p in get_spawn_position(info, position, player):
            self.delayed_spawn((self.next_entity_id, p, player, card_name, self), 0.0)

    def deploy_card(self, player_id, card_name, position, _from_mirror=False):
        # —— M1 镜像法术：重放上一张使用的卡，费用 = 基础费 + 1 ——
        if card_name == 'Mirror':
            p = self.players[player_id]
            if not p.can_play_card('Mirror'): return False
            last = p.last_card
            if not last: return False
            if p.elixir < Card(last).elixir + 1: return False
            ok = self.deploy_card(player_id, last, position, _from_mirror=True)
            if ok: p.elixir -= Card(last).elixir + 1  # _from_mirror 路径不扣基础费，此处一并扣
            return ok
        if not _from_mirror and not self.players[player_id].can_play_card(card_name):
            return False
        card_info = Card(card_name)

        if card_info.type != 'spell':
            # Check the deployment area is legit
            if self.is_position_occupied_by_building(position, 0): return False
            if player_id == 0:
                if position.y <= 1.0 and (position.x <= 6.0 or position.x > 12.0): return False
                if position.y >= 21.0: return False
                elif position.y >= 15.0:
                    if position.x <= 9:
                        if self.players[1].left_tower_hp > 0: return False
                    else:
                        if self.players[1].right_tower_hp > 0: return False
            elif player_id == 1:
                if position.y > 31.0 and (position.x <= 6.0 or position.x > 12.0): return False
                if position.y <= 10: return False
                elif position.y <= 17.0:
                    if position.x <= 9:
                        if self.players[0].left_tower_hp > 0: return False
                    else:
                        if self.players[0].right_tower_hp > 0: return False

        if card_info.type == 'spell':
            srow = spells.get(card_name, {})
            if srow.get('spawn_character'):
                # —— M1 区域持续出兵法术（墓园类）：在持续时间内按间隔确定性散布出兵 ——
                radius = (srow.get('radius') or 3000) / 1000.0
                duration = (srow.get('life_duration') or 5000) / 1000.0
                interval = (srow.get('spawn_interval') or 500) / 1000.0
                initial = (srow.get('spawn_initial_delay') or 0) / 1000.0
                count = max(0, int((duration - initial) / interval)) if interval else 0
                char_card = character_to_card.get(srow['spawn_character'], srow['spawn_character'])
                for i in range(count):
                    t = initial + i * interval
                    angle = i * 2.399963  # 黄金角，确定性散布（引擎整体确定性，不引入随机）
                    rad = radius * (0.35 + 0.65 * ((i * 7) % 10) / 9.0)
                    pos = Position(position.x + math.cos(angle) * rad, position.y + math.sin(angle) * rad)
                    self.delayed_spawn((len(self.entities)+1, pos, player_id, char_card, self), t)
                self._finish_deploy(player_id, card_name, _from_mirror)
                return True
            if srow.get('clone'):
                # —— M1 克隆法术：复制范围内的友军部队（克隆体 1 血，待 L4 校准）——
                radius = (srow.get('radius') or 3000) / 1000.0
                for entity in list(self.entities.values()):
                    if not isinstance(entity, Troop) or not entity.is_alive or entity.player != player_id: continue
                    if entity.position.distance_to(position) > radius: continue
                    clone = Troop(self.next_entity_id, Position(entity.position.x+0.3, entity.position.y+0.3),
                                  player_id, entity.card_name, battle_state=self)
                    clone.hp = 1.0
                    self._spawn_entity(clone)
                self._finish_deploy(player_id, card_name, _from_mirror)
                return True
            # —— M2 族4：瞬发区域法术（Zap/Freeze/Heal/Rage/Tornado/Earthquake/Poison）——
            # 此前这些法术走 get_spawn_position 生成「无行为隐形实体」，伤害/眩晕/拉拽全部无效。
            # 以数值表 projectile 字段判断是否飞行（Heal 有 spellAsDeploy 且 projectile=null → 瞬发）
            if (srow.get('buff') or srow.get('controls_buff')) and not srow.get('projectile'):
                ae = AreaEffect(self.next_entity_id, Position(position.x, position.y), player_id, card_name)
                self._spawn_entity(ae)
                self._finish_deploy(player_id, card_name, _from_mirror)
                return True

        if card_info.type == 'spell' and card_info.projectiles:
            initial_position = self.arena.BLUE_KING_TOWER if player_id == 0 else self.arena.RED_KING_TOWER

            target = BlankEntity(position)
            delayed_counter = 0
            for wave in range(card_info.projectile_waves):
                initial_position = Position(initial_position.x, initial_position.y)
                # I know that I should not use `len(self.entities)+1` here because it would cause bugs.
                # so in the actual `delay_spawn` function, I added another layer that corrects the entity id to a legit one.
                self.delayed_spawn((len(self.entities)+1, initial_position, player_id, card_name, target, False, self), delayed_counter)
                delayed_counter += card_info.wave_interval
            self._finish_deploy(player_id, card_name, _from_mirror)
            return True

        positions = get_spawn_position(card_info, position, player_id)
        # —— M4 族7：觉醒形态判定（卡组携带觉醒位 + 周期表：cycle=N → 第 N+1 次觉醒，交替）——
        p = self.players[player_id]
        evolved = bool(card_info.evo_raw and card_name in p.evo_slots
                       and evolution_state(p.evo_plays.get(card_name, 0), card_name))
        p.evo_plays[card_name] = p.evo_plays.get(card_name, 0) + 1
        delayed_counter = 0
        for pos in positions:
            self.delayed_spawn((len(self.entities)+1, pos, player_id, card_name, self, evolved), delayed_counter)
            delayed_counter += card_info.spawn_delay
        self._finish_deploy(player_id, card_name, _from_mirror)
        return True

    def calculate_building_cache(self):
        self.building_cache = []
        for x_cell in range(0, 36):
            self.building_cache.append([])
            for y_cell in range(0, 64):
                self.building_cache[x_cell].append(float('inf'))
        for x_cell in range(0, 36):
            for y_cell in range(0, 64):
                pos = cell_to_position((x_cell, y_cell))
                m = min(self.building_positions, key=lambda x: math.sqrt((pos.x-x[0])**2+(pos.y-x[1])**2)-x[2])
                minimum_distance = math.sqrt((pos.x-m[0])**2+(pos.y-m[1])**2)-m[2]
                self.building_cache[x_cell][y_cell] = minimum_distance
    def pathfind_ground_walkable(self, position, mover_radius):
        if not self.arena.is_walkable(position): return False
        x, y = position_to_cell(position)
        return self.building_cache[x][y] > mover_radius

    def ground_walkable(self, position, mover_radius):
        if not self.arena.is_walkable(position): return False
        return not self.is_position_occupied_by_building(position, mover_radius)

    def is_position_occupied_by_building(self, position, mover_radius: float = 0.5) -> bool:
        """Return True when a position overlaps any live building footprint."""
        for x,y,r in self.building_positions:
            # I choose not to use math.hypot to speed things up. This functino gets called several millions times per game
            if (x-position.x)**2+ (y-position.y)**2 < (r + mover_radius)**2:
                return True
        return False

    def resolve_collisions(self):
        entities_alive = [each for each in self.entities.values() if each.is_alive and (isinstance(each, Troop) or isinstance(each, Building))]
        ground_troops = combinations([each for each in entities_alive if not each.data.is_air_unit], 2)
        flying_troops = combinations([each for each in entities_alive if each.data.is_air_unit], 2)
        for troop in (ground_troops, flying_troops):
            for e1, e2 in troop:
                if e1.position.distance_to(e2.position) < e1.data.collision_radius + e2.data.collision_radius:
                    overlap = e1.data.collision_radius + e2.data.collision_radius - e1.position.distance_to(e2.position)
                    # the direction vector points from e1 to e2
                    direction_vector = complex(e2.position.x-e1.position.x, e2.position.y-e1.position.y)
                    if abs(direction_vector) == 0: return
                    direction_vector /= abs(direction_vector)
                    total_speed = e1.data.speed + e2.data.speed
                    if total_speed == 0: continue  # M1: 双建筑重叠（速度均为0）时跳过推挤，防除零
                    movement_ratio = e2.data.speed / total_speed
                    e2.position.x += direction_vector.real*movement_ratio*overlap
                    e2.position.y += direction_vector.imag*movement_ratio*overlap
                    e1.position.x += -direction_vector.real * (1-movement_ratio)*overlap
                    e1.position.y += -direction_vector.imag * (1-movement_ratio)*overlap

    def on_death(self, entity):
        if entity.name == 'King_PrincessTowers':
            player = entity.player
            for each in list(self.entities.values()):
                if each.name == 'KingTower' and each.player == player:
                    each.tower_active = True
                    break
        if isinstance(entity, Building): self.cache_fresh = False
        # —— M3：Skeleton King 灵魂收集（场上任意部队死亡 +1，上限 10；能力召唤物除外——简化）——
        if isinstance(entity, Troop) and entity.card_name != 'SkeletonKingSkeleton':
            for pid in (0, 1):
                self.souls[pid] = min(10, self.souls[pid] + 1)

    # —— M3 族6：Monk 法术反弹兜底（来源已死/法术来源 → 反弹至最近敌方公主塔）——
    def reflect_to_tower(self, monk_entity, damage):
        best, best_d = None, float('inf')
        for e in list(self.entities.values()):
            if not e.is_alive or e.player == monk_entity.player: continue
            if 'PrincessTower' not in e.name: continue
            d = e.position.distance_to(monk_entity.position)
            if d < best_d: best, best_d = e, d
        if best is not None: best.take_damage(damage)

    # —— M4 族7：觉醒女武神/皇家巨人的位移辅助 ——
    def push_enemies(self, player, position, radius, tiles):
        for e in list(self.entities.values()):
            if not e.is_alive or e.player == player or isinstance(e, Building): continue
            if isinstance(e, (Projectile, SpawnProjectile, AreaEffect)): continue
            d = e.position.distance_to(position)
            if d > radius or d < 0.05: continue
            nx = e.position.x + (e.position.x - position.x) / d * tiles
            ny = e.position.y + (e.position.y - position.y) / d * tiles
            if self.ground_walkable(Position(nx, ny), e.data.collision_radius) or e.data.is_air_unit:
                e.position.x, e.position.y = nx, ny

    def pull_enemies(self, player, position, radius, tiles, dt=0.1):
        for e in list(self.entities.values()):
            if not e.is_alive or e.player == player or isinstance(e, Building): continue
            if isinstance(e, (Projectile, SpawnProjectile, AreaEffect)): continue
            d = e.position.distance_to(position)
            if d > radius or d < 0.05: continue
            step = min(tiles, d)
            nx = e.position.x + (position.x - e.position.x) / d * step
            ny = e.position.y + (position.y - e.position.y) / d * step
            if self.ground_walkable(Position(nx, ny), e.data.collision_radius) or e.data.is_air_unit:
                e.position.x, e.position.y = nx, ny

    # —— M3 族6：英雄能力释放入口（演示/环境共用；自动选取第一个就绪英雄）——
    def use_ability(self, player_id):
        for e in list(self.entities.values()):
            if not e.is_alive or e.player != player_id: continue
            ability = getattr(e.data, 'ability', None)
            if not ability or e.ability_cd > 0: continue
            holder = e.entity_holder
            if not hasattr(holder, 'use_ability'): continue
            if not holder.use_ability(): continue
            cost = ability.get('manaCost', 0)
            if self.players[player_id].elixir < cost: return False
            self.players[player_id].elixir -= cost
            e.ability_cd = OFFICIAL_OVERRIDES.get(e.card_name, {}).get(
                'ability_cooldown', ability.get('cooldown', 0) / 1000)
            e.ability_uses += 1
            return True
        return False

    def deal_area_damage(self, from_player, position, range, amount, attack_air, attack_ground, crown_tower_damage_percent=1.0):
        for entity in list(self.entities.values()):
            if not entity.is_alive or entity.player == from_player: continue
            if entity.invincible: continue
            amount_dealt = amount if "King" not in entity.name else amount*crown_tower_damage_percent
            if attack_air and entity.data.is_air_unit:
                if entity.position.distance_to(position) < range:
                    entity.take_damage(amount_dealt)
            elif attack_ground and not entity.data.is_air_unit:
                if entity.position.distance_to(position) < range:
                    entity.take_damage(amount_dealt)


