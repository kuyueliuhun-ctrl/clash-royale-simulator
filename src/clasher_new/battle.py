from core import BlankEntity
from player import PlayerState
from pathfinding_heap import EntityPathfinder, position_to_cell, cell_to_position
from card_mechanics import *
from card_utils import Card, TimedExplosiveData, spells, buildings, projectiles, character_to_card, projectile_from_row
from card_utils import _rarity_level_index, _value_at_level, level_scale, card_data
from evolutions import evolution_state, derive_evolved_stats, OFFICIAL_OVERRIDES, collect_evo_mechanics
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
        # —— M8：精英卡（Hero 化）状态 ——
        self.hero_mode = False           # Wild slot：按 Hero 形态部署（apply_hero_overlay 置位）
        self._taunt_until = 0.0          # Knight Hero 嘲讽窗截止（绝对时刻）
        self._taunt_target_id = None     # 嘲讽者实体 id
        # —— M4.5 动作链：攻击序列状态（attackSequenceList）——
        # attack_seq = 序列原始定义（[{damage,...}]）；None = 无攻击序列
        # attack_seq_mode = "Manual"（跨攻击逐档推进，如 InfernoDragon_EV1）或 None（单次攻击多段命中，如 Berserker）
        self.attack_seq = None
        self.attack_seq_mode = None
        self.attack_seq_damages = []    # 各档伤害（已按当前级缩放）
        self.attack_seq_stage = 0       # 当前档位索引（Manual 递增用）
        self.attack_seq_pending = 0     # 多段命中：剩余待打段数
        self.attack_seq_hit_timer = 0.0 # 多段命中：段间隔计时
        # —— M5 觉醒补全：觉醒弹道钩子状态 ——
        self.evo_chain = None            # 链电参数 {'count','radius','stun'}（ElectroDragon_EV1）
        self.evo_bomber_chain = None     # 二段爆炸定义（Bomber_EV1 spawnChain）
        self._snipe_range_active = 0.0   # 狙击临时射程（Musketeer_EV1 customRange，0=普射）
        self._evo_temp_lifetime = None   # 觉醒临时复活体剩余寿命（Pekka tempResurrect）
        self._evo_resurrect_used = False # 复活已用过（每场一次）
        # —— M6：2025 新觉醒 7 张钩子状态 ——
        self._evo_veil_time = 0.0        # MinionHorde 首击面纱时长（>0 启用, 每成员一次）
        self._evo_veil_used = False      # 面纱已消耗
        self._evo_veil_timer = 0.0       # 面纱剩余时间（>0 期间 invincible）
        self._evo_landing = None         # RoyalHogs 落地定义 {'radius','damage'}（None=未起飞/已落地）

    # —— M2 族4：统一 buff 入口 ——
    def apply_buff(self, speed_mult=None, hit_speed_mult=None, duration=0.0,
                   damage_reduction=None, stun=0.0, heal=None, retarget=False):
        """speed_mult/hit_speed_mult: 倍率（>1 加速、<1 减速）；damage_reduction: 0~1 减伤；
        stun: 定身秒数（附带攻击蓄力重置与可选重索敌）；heal: {'hps','time'}"""
        if stun > 0:
            # —— 勘误批1：刺客突进不可被打断（用户口径 2026-09-04）——突进期间忽略眩晕/冰冻
            if getattr(self, '_dash_active', False):
                return
            self.freeze_timer = max(self.freeze_timer, stun)
            # Zap 类眩晕：重置攻击蓄力与冲锋（官方：眩晕重置蓄力类攻击，普通攻击仅暂停）
            self.attack_cooldown = max(self.attack_cooldown, self.data.hit_speed)
            # —— 勘误批6：眩晕重置蓄能（官方：Freeze/Zap 重置 Inferno 类 ramp-up）——
            self.ramp_stage = 0
            self.ramp_timer = 0.0
            self.ramp_target_id = None
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
                self.buff_time_remaining = max(self.buff_time_remaining, duration)
            else:
                # M6 修复：减速窗口此前误写 buff_time_remaining → speed_debuff 每 tick 被
                # debuff 分支重置（减速实际只生效 1 tick 的潜在 bug）。改写独立减速窗口。
                self.speed_debuff = min(self.speed_debuff, speed_mult)
                self.debuff_time_remaining = max(self.debuff_time_remaining, duration)
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
        if isinstance(self, (Troop, Building)):
            self._evo_on_death()
        # —— 勘误批2+：通用基础亡语（deathSpawnCharacterData；跳过专属机制类与觉醒同名亡语）——
        self._generic_death_spawn()
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
        # —— M5 觉醒补全：觉醒击杀治疗归因（PekkaEV1_Heal：onKilledDoneAction）——
        # 击杀者为本方觉醒 Pekka → 治疗 resurrectParameters[2]=500（lv1 基准，语义假设标注）
        killer = getattr(self, 'last_hit_by', None)
        if (isinstance(killer, Entity) and killer.is_alive and isinstance(killer, Troop)
                and killer.battle_state is not None):
            _kevo = killer.evo or {}
            if _kevo.get('onKilledDoneAction'):
                rp = _kevo.get('resurrectParameters') or []
                heal = (rp[2] if len(rp) > 2 else 500) * level_scale(killer.level)
                killer.hp = min(killer.data.hp, killer.hp + heal)
        self.battle_state.on_death(self)

    def _collector_tick(self, dt):
        """【勘误批3】圣水收集器：每 manaGenerateTimeMs 产 manaCollectAmount 圣水;
        冻结期暂停（官方 2016/7/4）; 死亡返还 manaOnDeath（走 _death_elixir_gift）。"""
        if self.card_name != 'Elixir Collector':
            return
        scd = self.data.data.get('summonCharacterData') or {}
        period = (scd.get('manaGenerateTimeMs') or 12000) / 1000
        amt = scd.get('manaCollectAmount') or 1
        if getattr(self, '_collect_cd', None) is None:
            self._collect_cd = period
        self._collect_cd -= dt
        if self._collect_cd <= 0 and self.freeze_timer <= 0:
            self._collect_cd = period
            p = self.battle_state.players[self.player]
            p.elixir = min(10.0, p.elixir + amt)

    def _troop_spawner_tick(self, dt):
        """【勘误批2+】部队通用周期出兵：gamedata spawnCharacterData/spawnNumber/spawnPauseTime。
        首波 1.0s（官方 Night Witch 部署后 1s, 与 Witch 机制类惯例一致）, 之后每 spawnPauseTime 一波。
        仅限无专属出兵机制类的部队（Witch 跳过；建筑走各自管线）。"""
        if self.card_name in ('Witch',) or not isinstance(self, Troop):
            return
        scd = self.data.data.get('summonCharacterData') or {}
        sp = scd.get('spawnCharacterData')
        pause = scd.get('spawnPauseTime')
        if not sp or not sp.get('name') or not pause:
            return
        if getattr(self, '_spawner_cd', None) is None:
            self._spawner_cd = 1.0   # 首波延迟【官方 Night Witch 1s】
        self._spawner_cd -= dt
        if self._spawner_cd > 0:
            return
        self._spawner_cd = pause / 1000.0
        from card_utils import character_to_card as _c2cs
        _nm = _c2cs.get(sp['name'], sp['name'])
        if _nm not in card_data:
            return
        _cnt = int(scd.get('spawnNumber') or 1)
        _info = Card(_nm)
        _info.spawn_number = _cnt
        _info.spawn_delay = 0
        for p in get_spawn_position(_info, self.position, self.player, False):
            self.battle_state._spawn_entity(Troop(self.battle_state.next_entity_id, p, self.player, _nm, self.battle_state))
            self.battle_state.next_entity_id += 1

    def _generic_death_spawn(self):
        """【勘误批2+】通用基础亡语：deathSpawnCharacterData 消费（此前仅觉醒路径可走）。
        - 炸弹型（dsd 带 deathDamage 无 hitpoints, 如 BombTowerBomb）：TimedExplosive(本卡)
          ——deployTime 引信后按 deathDamage 爆炸（TimedExplosiveData 读本卡 death_spawn_data）;
        - 单位型：deathSpawnCount 个（缺省 1）散布出在原地;
        - SkeletonBalloon：容器 0.6s 后出 7 骷髅（container deathSpawnCount）;
        - ElixirGolem 三级链由单位型递归覆盖（Golem→2×Golem2→各 2×Golem4）,
          圣水馈赠对手在 _death_elixir_gift 处理（Fandom 机制【数值待对拍】）。
        跳过：① 有专属 on_death 的机制类（Golem/LavaHound/Balloon/GiantSkeleton/BattleRam）
              ② 觉醒态且觉醒亡语同名（_evo_on_death 已出, 防双倍）。"""
        if not isinstance(self, (Troop, Building)) or self.battle_state is None:
            return
        dsd = getattr(self.data, 'death_spawn_data', None) or {}
        name = dsd.get('name')
        if not name:
            return
        holder_handled = type(self.entity_holder).on_death is not BasicCharacter.on_death
        evo_dsd = ((getattr(self, 'evo', None) or {}).get('deathSpawnCharacterData') or {}).get('name')
        if holder_handled or (getattr(self, 'evo', None) and evo_dsd):
            return
        bs = self.battle_state
        self._death_elixir_gift()
        if dsd.get('deathDamage') and not dsd.get('hitpoints'):
            bomb = TimedExplosive(bs.next_entity_id, Position(self.position.x, self.position.y),
                                  self.player, self.card_name)
            bs._spawn_entity(bomb)
            return
        if self.card_name == 'SkeletonBalloon' or name == 'SkeletonContainer':
            _cnt = int(dsd.get('deathSpawnCount') or 7)
            _delay = (dsd.get('deployTime') or 600) / 1000.0
            for i in range(_cnt):
                bs.delayed_spawn((bs.next_entity_id + i, Position(self.position.x, self.position.y),
                                  self.player, 'Skeleton', bs), _delay)
            bs.next_entity_id += _cnt
            return
        from card_utils import character_to_card as _c2c
        target = _c2c.get(name, name)
        if target not in card_data:
            return
        count = int((self.data.data.get('summonCharacterData') or {}).get('deathSpawnCount') or 1)
        _info = Card(target)
        _info.spawn_number = count
        _info.spawn_delay = 0
        for p in get_spawn_position(_info, self.position, self.player, False):
            t = Troop(bs.next_entity_id, p, self.player, target, bs)
            t._soul_excluded = True   # 勘误批11：亡语衍生不计魂
            bs._spawn_entity(t)
            bs.next_entity_id += 1

    def _death_elixir_gift(self):
        """【勘误批4】Elixir Golem 死亡给对手圣水（gamedata 无字段, Fandom 机制）：
        本体 1.0 / Golemite(ElixirGolem2) 0.5 / Blob(ElixirGolem4) 0.25——总量 3【待对拍】。"""
        scd = self.data.data.get('summonCharacterData') or {}
        if scd.get('manaOnDeath'):
            p = self.battle_state.players[self.player]
            p.elixir = min(10.0, p.elixir + scd['manaOnDeath'])
        _gift = {'ElixirGolem': 1.0, 'ElixirGolem2': 0.5, 'ElixirGolem4': 0.25}
        amt = _gift.get(self.card_name)
        if amt:
            opp = self.battle_state.players[1 - self.player]
            opp.elixir = min(10.0, opp.elixir + amt)

    def update(self, dt):
        # This part may be a bit confusing because it doesn't check the `is_alive` and `deploy_delay_remaining` attribute.
        # Reasons: this will be eventually called by `super()` and won't terminate the actual update function. And
        # there are miner and drill that needs to be moving before it's even deployed. So this function only updates the buff_time
        # and debuff_time attribute.

        # I assume this function will be called after the deployment and alive check.
        # —— 勘误批13：Mother Witch 诅咒 5s 过期（快照无字段, Fandom 机制）——
        _vc = getattr(self, 'voodoo_curse', None)
        if _vc and _vc.get('until') is not None and self.battle_state.time > _vc['until']:
            self.voodoo_curse = None
        # —— M6：MinionHorde 首击面纱计时（期间 invincible, 到期恢复）——
        if getattr(self, '_evo_veil_timer', 0.0) > 0:
            self._evo_veil_timer -= dt
            self.invincible = self._evo_veil_timer > 0
        # —— M7：Vines 拽落空中单位的临时落地（snare 结束复飞）。
        # 用绝对时刻判定（束缚期间 Troop.update 因 freeze_timer 提前 return 不走此处,
        # 相对计时不会递减）：束缚结束后的首个 update tick 立即复飞。
        if getattr(self, '_vines_grounded_time', 0.0) > 0:
            if self.battle_state.time >= getattr(self, '_vines_grounded_until', 0.0):
                self._vines_grounded_time = 0.0
                if not self.data.is_air_unit and not self.jumping_across_river:
                    self.data.is_air_unit = Card(self.name).is_air_unit
                    self.path = []
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
            # M5 觉醒补全：过量治疗上限（Bats_EV1 allowedOverHealPerc 200 → 可治疗至 2×max_hp；
            # 字段位于 buffAfterHitsData[].allowedOverHealPerc）
            _oh = 0
            if self.evo:
                for _bf in self.evo.get('buffAfterHitsData') or []:
                    _oh = max(_oh, _bf.get('allowedOverHealPerc') or 0)
            _cap = self.data.hp * (1 + _oh / 100.0) if _oh else self.data.hp
            still = []
            for rb in self.regen_buffs:
                rb['time'] -= dt
                if rb['time'] > 0:
                    self.hp = min(_cap, self.hp + rb['hps'] * dt)
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
        # M5 修复：无伤害档位的序列（ElectroDragon_EV1 的 attackSequenceList 仅引用
        # doAttackAction 动作组）不挂载——否则劫持 on_attack 结算路径，data.damage=0 时打 0 伤
        if not any(_st.get('damage') for _st in raw):
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


    def take_damage(self, amount: float, delayed=False, source=None, pierce_invincible=False):
        """Apply damage to entity.  M5：source=伤害来源（觉醒击杀治疗归因用）
        M6：pierce_invincible=法术伤害可穿透亡影无敌（觉醒骷髅军团亡影「can be affected by spells」）"""
        # —— M6：MinionHorde 首击面纱（每成员一次：首击被闪避 + 进入短无敌窗口）——
        if getattr(self, '_evo_veil_time', 0.0) > 0 and not self._evo_veil_used:
            self._evo_veil_used = True
            self._evo_veil_timer = self._evo_veil_time
            self.invincible = True
            return
        if self.invincible and not pierce_invincible: return
        if source is not None: self.last_hit_by = source
        # —— M7：受击钩子（Ronin 格挡反击等经 entity_holder.on_take_damage 挂载）。
        # delayed 与即时路径统一在此触发（delayed 攻击经 pending_damage 重放时 source 已失,
        # 因此必须在首次调用处拿 source）；钩子返回 True = 本次伤害被格挡, 直接短路。
        # M8：放宽为 source=None 也触发（Berserker 熊灵 HP 下限需覆盖法术/AOE 伤害；
        # 各钩子对 source=None 自行短路, Ronin 语义不变）。
        if hasattr(self.entity_holder, 'on_take_damage'):
            if self.entity_holder.on_take_damage(amount, source):
                return
        dr = max(self.damage_reduction if self.damage_reduction_timer > 0 else 0.0,
                 getattr(self, '_fortify_dr', 0.0))
        if dr > 0:
            amount *= (1.0 - dr)
        # —— M6：RoyalHogs 受击落地（起飞状态下被任意伤害命中 → 落地 + 落地 AoE, 一次性）——
        land = getattr(self, '_evo_landing', None)
        if land is not None and isinstance(self, Troop):
            self._evo_landing = None
            self.data.is_air_unit = False
            self.battle_state.deal_area_damage(self.player, self.position, land['radius'],
                                               land['damage'], True, True)
        if delayed:
            self.pending_damage.append(amount)
            return
        had_shield = self.shield_health
        if not self.shield_health: self.hp -= amount
        else: self.shield_health = max(0, self.shield_health - amount)
        # —— M5 觉醒补全：护盾破碎爆炸（Wizard_EV1_ShieldLostAoE：radius 3000 / damage 110）——
        if had_shield > 0 and self.shield_health == 0:
            # —— 勘误批6：破盾重置蓄能（官方 Inferno 类机制）——
            self.ramp_stage = 0
            self.ramp_timer = 0.0
            self.ramp_target_id = None
            sla = (self.evo or {}).get('shieldLostActionData') if self.evo else None
            if sla and self.battle_state is not None:
                for sub in sla.get('subActionsData') or []:
                    sd = sub.get('spawnDataData') or {}
                    if sd.get('damage'):
                        self.battle_state.deal_area_damage(
                            self.player, self.position, (sd.get('radius') or 3000) / 1000,
                            sd['damage'] * level_scale(self.level), True, True)

        if self.hp <= 0 and self.is_alive:
            self.die()
            if self.data.death_damage:
                # I assume that all death damage deals attack to both air and ground troops.
                # The game data file hasn't specified what's the radius of the death damage,
                # so here I just set it to 1 tile
                self.battle_state.deal_area_damage(self.player, self.position, 1.0+self.data.collision_radius, self.data.death_damage,
                                                   attack_air=True, attack_ground=True)
        # —— 勘误批3：受击后钩子（ElectroGiant Zap Pack 反射等；不短路伤害）——
        if self.is_alive and hasattr(self.entity_holder, 'on_damaged'):
            self.entity_holder.on_damaged(amount, source)

    def in_attack_range(self, target):
        if target is None: return False
        if 'PrincessTower' in target.name:
            bonus = 0.5
        else:
            bonus = 0
        dist = self.position.distance_to(target.position)
        # —— 勘误批12：形态射程覆盖（ThreeMusketeers 近战形态等）——
        _ro = getattr(self, '_range_override', None)
        if _ro is not None:
            return dist <= _ro + bonus + self.data.collision_radius
        # M2 族3：建筑最小射程（Mortar 贴脸不攻击）
        if getattr(self.data, 'min_range', 0) and dist < self.data.min_range:
            return False
        # M5 觉醒补全：觉醒火枪手狙击弹临时射程（customRange 30000）
        _snipe = getattr(self, '_snipe_range_active', 0)
        if _snipe and dist <= _snipe: return True
        return dist <= self.data.range + target.data.collision_radius + bonus
    def in_sight_range(self, target):
        if target is None: return False
        if 'PrincessTower' in target.name:
            bonus = 0.5
        else:
            bonus = 0
        dist = self.position.distance_to(target.position)
        # M5 觉醒补全：狙击弹索敌范围同步扩展（否则 30 格目标因视距 6 永远锁不到）
        _snipe = getattr(self, '_snipe_range_active', 0)
        if _snipe and dist <= _snipe: return True
        return dist <= self.data.sight_range + target.data.collision_radius + bonus

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
            best_same_side = None   # 与本单位同侧的最近塔（越河索敌的合法目标）
            best_any = None         # 无侧向限制兜底（仅当同侧无塔存活，如王塔已死=终局前）
            for i in range(1, 7):
                if not self.battle_state.entities[i].is_alive: continue
                possible_princess_tower = self.battle_state.entities[i]
                if possible_princess_tower.player == self.player: continue
                distance = possible_princess_tower.position.distance_to(self.position) - possible_princess_tower.data.collision_radius
                if best_any is None or distance < best_any[0]:
                    best_any = (distance, i)
                # 索敌范围不跨中轴：永远不把对侧公主塔作为回退目标。王塔(位于中轴)
                # 不受限。实证：王塔(y=29)比公主塔(y=25.5)深，破左塔后左桥头 x>=4.5
                # 处欧氏距离反而是右塔更近（等距线恰切在桥面 x~4.2-4.5），桥头部队
                # 会被吸去对侧车道；镜像方向同样成立。与真实 CR 一致：破边塔后
                # 本侧部队转国王塔，不横穿去打对侧公主塔。
                _cx = self.battle_state.arena.width / 2
                if (possible_princess_tower.position.x - _cx) * (self.position.x - _cx) >= 0:
                    if distance < min_distance:
                        min_distance = distance
                        best_same_side = (distance, i)
            pick = best_same_side or best_any
            if pick is not None:
                self.target_id = pick[1]
            current_target = self.battle_state.entities[self.target_id]
        return current_target

    def _hero_taunt_override(self, current_target):
        """M8 ①：Knight Hero 嘲讽覆盖（Triumphant Taunt）——嘲讽窗内强制锁定嘲讽者。
        在 update_current_target 之后调用（覆盖普通索敌/换目标逻辑，含建筑目标）。"""
        if getattr(self, '_taunt_until', 0.0) > self.battle_state.time:
            t = self.battle_state.entities.get(getattr(self, '_taunt_target_id', None))
            if t is not None and t.is_alive:
                return t
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
        # 卡死自救探针（多地面单位在桥头/墙角互相顶住的死锁检测，见 Troop.update 移动分支）
        self._stuck_check_pos = None
        self._stuck_check_time = 0.0
        # 队形车道偏移：部署时按出兵位置相对部署中心的横向偏移记录（deploy_card 打标），
        # 行进时把路点沿行进方向法线平移，让同卡多单位并排走位而不是前后互挤
        self._lane_offset = getattr(position, '_lane_offset', 0.0)
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
        # —— M5 觉醒补全：全树机制字段收集（含动作组内嵌参数）——
        for k, v in collect_evo_mechanics(evo_raw).items():
            stats.setdefault(k, v)
        self.hp = stats.get('hp', self.hp)
        if stats.get('damage'):
            self.data.damage = stats['damage']
        if stats.get('shield_hitpoints'):
            self.shield_health = stats['shield_hitpoints']
        self.evo = stats
        # —— M5 觉醒补全：觉醒弹道钩子参数 ——
        _epd = stats.get('evoProjectileData') or {}
        if _epd.get('chainedHitCount'):
            # 觉醒雷电飞龙链电（doAttackAction 攻击组；链半径无字段 → 默认 3.0，待 L4）
            self.evo_chain = {'count': int(_epd['chainedHitCount']), 'radius': 3.0,
                              'stun': ((_epd.get('targetBuffData') or {}).get('buffTime') or 500) / 1000}
        if _epd.get('spawnChain') and (_epd.get('spawnProjectileData') or {}).get('damage'):
            # 觉醒炸弹人二段爆炸链（BombSkeletonProjectile_2_EV1，damage 88）
            self.evo_bomber_chain = _epd['spawnProjectileData']
        # —— M4.5 动作链：觉醒形态攻击序列（InfernoDragon_EV1 四级递增 / ElectroDragon_EV1 动作组）——
        if stats.get('attackSequenceList'):
            self._resolve_attack_seq(stats['attackSequenceList'], stats.get('attackSequenceMode'))
        # —— onStartingActionData：出场动作（觉醒特斯拉出场眩晕脉冲）——
        osad = stats.get('onStartingActionData')
        if osad and 'Tesla' in str(osad.get('name', '')) and self.battle_state is not None:
            _tesla_evo_pulse(self.battle_state, self, osad)
        # —— M6：2025 新觉醒 7 张钩子（evo2025Hooks, 数据层 evo_2025_data.py）——
        hooks = stats.get('evo2025Hooks') or {}
        if hooks.get('firstHitVeil'):
            # MinionHorde 首击面纱：时长见数据层（假设值 0.67s, 标注）
            self._evo_veil_time = float(hooks['firstHitVeil'])
            self._evo_veil_used = False
        fa = hooks.get('flyingAssault')
        if fa:
            # RoyalHogs 起飞：部署即飞行, 直达最近建筑（data.is_air_unit=True 走直线移动）；
            # 跳河能力关闭（飞行期跳河复位会污染落地状态）
            li = _rarity_level_index(fa.get('landingDamageRarity') or self.data.rarity, self.level)
            _arr = fa.get('landingDamagePerLevel') or []
            _dmg = _arr[li] if 0 <= li < len(_arr) else (_arr[-1] if _arr else 0)
            self._evo_landing = {'radius': fa.get('landingRadius', 1.5), 'damage': _dmg}
            self.data.is_air_unit = True
            self.data.jump_speed = 0
        if hooks.get('generalGerry'):
            # SkeletonArmy：确保本方 General Gerry 存在（15 只同批部署只生成 1 个）
            self._evo2025_ensure_gerry(hooks['generalGerry'])
        gust = hooks.get('gust')
        if gust:
            # BabyDragon 气流：伤害 ×1.04（6/7/2026 平衡）；aura tick 由 _evo2025_gust_tick 驱动。
            # 基础幼龙伤害在弹道（data.damage=0）→ 同步缩放弹道伤害
            self._evo_gust = gust
            if gust.get('damageMult'):
                if self.data.damage:
                    self.data.damage = round(self.data.damage * gust['damageMult'])
                elif self.data.projectiles:
                    self.data.projectile_data.damage = round(
                        self.data.projectile_data.damage * gust['damageMult'])

    # ==================== M6：2025 新觉醒 7 张钩子 ====================
    def _evo2025_ensure_gerry(self, gg):
        """SkeletonArmy_EV1：本方 General Gerry 单例（同批 15 骷髅只生成一个, 后排=己方后方）。"""
        bs = self.battle_state
        if bs is None: return
        for e in list(bs.entities.values()):
            if getattr(e, '_evo2025_is_gerry', False) and e.is_alive and e.player == self.player:
                self._evo2025_army_gerry = e.id
                return
        name = gg.get('card', 'SkeletonArmy_EV1_General')
        # 后排方向：蓝方(player 0)部署区 y<15 → 己方后方 = -y；红方相反（口径假设, 数据无落点字段）
        back = -gg.get('backOffset', 1.0) if self.player == 0 else gg.get('backOffset', 1.0)
        g = Troop(bs.next_entity_id, Position(self.position.x, self.position.y + back),
                  self.player, name, bs)
        g._evo2025_is_gerry = True
        bs._spawn_entity(g)
        self._evo2025_army_gerry = g.id

    def _evo2025_gust_tick(self, dt, current_target=None):
        """BabyDragon_EV1 气流：攻击期间（目标在攻击范围内）友军 +30% / 敌军 -30% 移速，
        半径 4.0（8×9 格 → 圆形假设）。脱战即气流停止；死亡残留由 _evo_on_death 落地。"""
        gust = getattr(self, '_evo_gust', None)
        if not gust: return
        if current_target is None or not self.in_attack_range(current_target):
            return
        pulse = gust.get('pulse', 0.25)
        self._evo_gust_timer = getattr(self, '_evo_gust_timer', 0.0) - dt
        if self._evo_gust_timer > 0: return
        self._evo_gust_timer = pulse
        r = gust.get('radius', 4.0)
        for e2 in list(self.battle_state.entities.values()):
            if not e2.is_alive or e2 is self: continue
            if isinstance(e2, (Projectile, SpawnProjectile, AreaEffect, EvoEffectZone,
                               EvoZapZone, GenericBomb, TimedExplosive)): continue
            if e2.position.distance_to(self.position) > r + e2.data.collision_radius: continue
            if e2.player == self.player:
                e2.apply_buff(speed_mult=gust.get('allySpeedMult', 1.30), duration=pulse * 1.5)
            else:
                e2.apply_buff(speed_mult=gust.get('enemySpeedMult', 0.70), duration=pulse * 1.5)

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
                # M5 觉醒补全：狂暴同时覆盖攻速与移速（Barbarian_EVO_Rage：hitSpeedMultiplier 135
                # + speedMultiplier 135；此前只加移速）
                self.apply_buff(speed_mult=buff.get('speedMultiplier', 135) / 100.0,
                                hit_speed_mult=buff.get('hitSpeedMultiplier', 135) / 100.0,
                                duration=window or 3.0)
        # ② 觉醒弓箭手：二段箭（projectile2Data，双发射击）
        p2 = evo.get('projectile2Data')
        if p2 and p2.get('damage'):
            # M5 觉醒补全：specialAttackRangeForStats 4500 → 仅 4.5 格内附加二段箭（数据边界）
            _max_d = (evo.get('specialAttackRangeForStats') or 0) / 1000
            if not _max_d or self.position.distance_to(target.position) <= _max_d:
                target.take_damage(p2["damage"] * level_scale(self.level), delayed=True)  # 84(起始级)→lv16 ≈351
        # ②b M5 觉醒补全：觉醒猎人网缚（onStartingActionData=Hunter_EV1_net_attack：
        # range 4000 / 熊陷阱弹 / targetFilter 无建筑 → 仅对部队；眩晕时长数据无字段
        # → 默认 1.0s，标注假设）。仅首次攻击生效。
        _osad = evo.get('onStartingActionData') or {}
        if 'net' in str(_osad.get('name', '')) and not getattr(self, '_evo_net_used', False):
            self._evo_net_used = True
            if target is not None and isinstance(target, Troop) and target.is_alive:
                target.apply_buff(stun=1.0)
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
                # M5 觉醒补全：觉醒女武神迷你龙卷走数据值（Valkyrie_MiniTornado_EV1：
                # radius 5500 / lifeDuration 500 / hitFrequency 400 / damagePerSecond 83 /
                # attractPercentage 300）——EvoEffectZone：0.5s 内 1 次脉冲 + 向心吸引
                _bf = sd.get('buffData') or {}
                _zone = EvoEffectZone(
                    self.battle_state.next_entity_id, Position(self.position.x, self.position.y),
                    self.player, self.battle_state,
                    radius=(sd.get('radius') or 5500) / 1000,
                    lifetime=(sd.get('lifeDuration') or 500) / 1000,
                    dps=_bf.get('damagePerSecond', 0),
                    tick=max(_bf.get('hitFrequency') or sd.get('hitSpeed') or 400, 50) / 1000,
                    attract=(_bf.get('attractPercentage') or 0) / 100.0,  # 300 → 3 格/s（口径假设）
                    level=self.level, label=sd.get('name') or 'Valkyrie_MiniTornado_EV1')
                self.battle_state._spawn_entity(_zone)
        # —— M6 ①：Princess 减速箭（首发减速, 之后每 everyNHits 发减速一次；
        # 半径 3.0 / 30% / 5.5s, 4/8/2026 平衡口径, 数据来源 evo_2025_data.py）——
        hooks = evo.get('evo2025Hooks') or {}
        sa = hooks.get('slowAttack')
        if sa and target is not None:
            self._evo_attack_count = getattr(self, '_evo_attack_count', 0) + 1
            _n = sa.get('everyNHits', 2)
            if self._evo_attack_count == 1 or (self._evo_attack_count - 1) % _n == 0:
                for e2 in list(self.battle_state.entities.values()):
                    if not e2.is_alive or e2.player == self.player: continue
                    if isinstance(e2, (Projectile, SpawnProjectile, AreaEffect, EvoEffectZone,
                                       EvoZapZone, GenericBomb, TimedExplosive)): continue
                    if e2.position.distance_to(target.position) <= sa.get('radius', 3.0) + e2.data.collision_radius:
                        e2.apply_buff(speed_mult=sa.get('speedMult', 0.70),
                                      duration=sa.get('duration', 5.5))
        # —— M6 ③：RoyalHogs 攻击即落地（落地 AoE 后恢复地面近战, 一次性）——
        land = getattr(self, '_evo_landing', None)
        if land is not None:
            self._evo_landing = None
            self.data.is_air_unit = False
            self.battle_state.deal_area_damage(self.player, self.position, land['radius'],
                                               land['damage'], True, True)

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
        # —— M5 觉醒补全：Pekka 觉醒临时复活（tempResurrect + resurrectParameters）——
        # 数值取自 gamedata resurrectParameters=[0,2000,500,500,5000,5000,900,10000,200]：
        #   rp[1]=2000 复活延迟 ms；rp[2]=500 基础复活 HP（lv1 基准）；rp[8]=200 每灵魂加成
        #   （resurrectChargeFilter=skeleton_king_charge_souls → 与骷髅王同源灵魂计数）；
        #   rp[7]=10000 HP 上限；rp[4]=5000 临时存活时长（语义假设，置信度低，待 L4 对拍）
        if evo.get('tempResurrect') and not self._evo_resurrect_used and hasattr(self.battle_state, 'resurrect_queue'):
            rp = evo.get('resurrectParameters') or []
            delay = (rp[1] if len(rp) > 1 else 2000) / 1000.0
            base_hp = (rp[2] if len(rp) > 2 else 500) * level_scale(self.level)
            per_soul = (rp[8] if len(rp) > 8 else 0) * level_scale(self.level)
            hp_cap = (rp[7] if len(rp) > 7 else 10000) * level_scale(self.level)
            souls = self.battle_state.souls[self.player] if hasattr(self.battle_state, 'souls') else 0
            if hasattr(self.battle_state, 'souls'): self.battle_state.souls[self.player] = 0
            hp = min(base_hp + souls * per_soul, hp_cap)
            lifetime = (rp[4] if len(rp) > 4 else 5000) / 1000.0
            self.battle_state.resurrect_queue.append(
                (self.card_name, Position(self.position.x, self.position.y), self.player,
                 hp, lifetime, self.battle_state.time + delay))
            self._evo_resurrect_used = True
        # —— M6 ⑤：SkeletonArmy 亡影转化（Gerry 存活时骷髅阵亡 → 原地生成亡影：
        # 无敌+不可选取, 法术可伤害; Gerry 阵亡后不再转化）——
        hooks = evo.get('evo2025Hooks') or {}
        ggh = hooks.get('generalGerry')
        if ggh:
            _gerry = self.battle_state.entities.get(getattr(self, '_evo2025_army_gerry', None))
            if _gerry is not None and _gerry.is_alive and _gerry.player == self.player:
                sh = Troop(self.battle_state.next_entity_id,
                           Position(self.position.x, self.position.y),
                           self.player, ggh.get('shadowCard', 'SkeletonArmy_EV1_Shadow'),
                           self.battle_state)
                sh._evo2025_is_shadow = True
                sh._evo2025_shadow_of = _gerry.id
                # 官方语义：亡影不可被部队/建筑/塔选取, 无限 HP（普攻无效）, 但可被法术伤害
                # （AreaEffect._pulse 对 _evo2025_is_shadow 走 pierce_invincible）
                sh.targetable = False
                sh.invincible = True
                sh.deploy_delay_remaining = 0.0
                self.battle_state._spawn_entity(sh)
        # —— M6 ①：Princess 死亡减速领域（半径 3.0 / 30% / 5.5s）——
        dz = hooks.get('deathSlowZone')
        if dz:
            _zone = EvoEffectZone(
                self.battle_state.next_entity_id, Position(self.position.x, self.position.y),
                self.player, self.battle_state,
                radius=dz.get('radius', 3.0), lifetime=dz.get('duration', 5.5),
                slow=dz.get('speedMult', 0.70), level=self.level,
                label='Princess_EV1_DeathZone')
            self.battle_state._spawn_entity(_zone)
        # —— M6 ⑥：BabyDragon 死后气流残留（约 2s：友 +30% / 敌 -30%）——
        gust = hooks.get('gust')
        if gust:
            _zone = EvoEffectZone(
                self.battle_state.next_entity_id, Position(self.position.x, self.position.y),
                self.player, self.battle_state,
                radius=gust.get('radius', 4.0), lifetime=gust.get('deathLinger', 2.0),
                slow=gust.get('enemySpeedMult', 0.70),
                ally_buff={'speed_mult': gust.get('allySpeedMult', 1.30), 'duration': 0.3},
                level=self.level, label='BabyDragon_EV1_Gust')
            self.battle_state._spawn_entity(_zone)

    def _evo_giant_tick(self, dt, current_target=None):
        """M5 觉醒补全：GoblinGiant_EV1 哥布林投掷（onStartingActionData：healthPercentages
        [50,50,50] 血量阈值触发 + actionsData interval=1800ms 连续出兵；
        actionToExecuteData → Goblin 内嵌定义 hp79/dmg47，向目标方向掷出）。"""
        evo = self.evo
        if not evo or not evo.get('healthPercentages') or not evo.get('actionsData'): return
        if not getattr(self, '_evo_giant_triggered', False):
            if self.hp <= self.data.hp * 0.5:
                self._evo_giant_triggered = True
                self._evo_giant_timer = 0.0
            else:
                return
        self._evo_giant_timer = getattr(self, '_evo_giant_timer', 0.0) - dt
        if self._evo_giant_timer > 0: return
        self._evo_giant_timer = (evo['actionsData'][0].get('interval') or 1800) / 1000.0
        act = evo['actionsData'][0].get('actionToExecuteData') or {}
        # 投掷落点：朝当前目标方向 1 格（spawn_forward 语义）
        pos = self.position
        if current_target is not None:
            d = self.position.distance_to(current_target.position) or 1.0
            pos = Position(self.position.x + (current_target.position.x - self.position.x) / d,
                           self.position.y + (current_target.position.y - self.position.y) / d)
        interpret_action_group(self, act, target=current_target, position=pos)

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
        # —— M5 觉醒补全：觉醒临时复活体寿命（Pekka tempResurrect：到期即亡，不再二次复活）——
        if self._evo_temp_lifetime is not None:
            self._evo_temp_lifetime -= dt
            if self._evo_temp_lifetime <= 0:
                self.die()
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
        # —— M8 ①：Knight Hero 嘲讽覆盖（窗内强制锁定嘲讽者）——
        current_target = self._hero_taunt_override(current_target)
        # —— M5 觉醒补全：GoblinGiant_EV1 血量阈值连续出兵 ——
        self._evo_giant_tick(dt, current_target)
        # —— 勘误批3：Elixir Collector 产水（manaCollectAmount/manaGenerateTimeMs; 冻结期停止【官方 2016】）——
        self._collector_tick(dt)
        # —— M6 ⑥：BabyDragon_EV1 攻击期间气流 ——
        self._evo2025_gust_tick(dt, current_target)
        # —— 勘误批2+：部队通用周期出兵（spawnCharacterData+spawnPauseTime, Night Witch 等；
        # Witch 有专属机制类自出骷髅, 跳过防双倍）——
        self._troop_spawner_tick(dt)
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
        # —— 勘误批1：刺客突进（Assassin dash）——突进一旦开始不可被打断：
        # 在冰冻/眩晕检查之后（apply_buff 已对突进态豁免眩晕, 此处再 bypass 常规索敌/移动）
        if getattr(self, '_dash_active', False):
            self.entity_holder.dash_tick(dt)
            return
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

                # —— 卡死自救：多地面单位在窄桥/墙角互相顶住时，朝当前路点的移动分量被
                # resolve_collisions 沿连线抵消，而上面的 dot 法要求“穿过路点”才切下一个
                # 路点 → 永久死锁（实证：一对弓箭手楔死在桥头，60s 纹丝不动）。
                # 每 0.5s 检查一次位移，近似为零则丢弃当前路点让单位继续走向下一个；
                # 路点丢光后由 `if not self.path` 分支按当前位置重算全程路径。
                if self._stuck_check_pos is None:
                    self._stuck_check_pos = Position(self.position.x, self.position.y)
                    self._stuck_check_time = self.battle_state.time
                elif self.battle_state.time - self._stuck_check_time >= 0.5:
                    if self.path and self.position.distance_to(self._stuck_check_pos) < 0.1:
                        min_point = min(self.path, key=lambda pos: pos.distance_to(self.position))
                        self.path = [p for p in self.path if p is not min_point]
                    if not self.path:
                        self.path = EntityPathfinder(self, current_target, self.battle_state).calculate()
                    self._stuck_check_pos = Position(self.position.x, self.position.y)
                    self._stuck_check_time = self.battle_state.time

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
                    # 队形车道：把路点沿“本单位→路点”方向的法线平移 _lane_offset，
                    # 同卡多单位各自保持部署时的横向车道并排推进（法线方向随行进自适应，
                    # 纵向推进时即 x 平移；最后一格收尾仍直取目标不偏移）。
                    _wp = self.path[index]
                    _off = self._lane_offset
                    if _off:
                        _dxw, _dyw = _wp.x - self.position.x, _wp.y - self.position.y
                        _d = math.hypot(_dxw, _dyw)
                        if _d > 1e-6:
                            _wp = Position(_wp.x - _dyw / _d * _off, _wp.y + _dxw / _d * _off)
                    self.move_towards(_wp, dt, True)
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
            # —— M5 觉醒补全：全树机制字段收集（与 Troop._apply_evolution 同口径）——
            for k, v in collect_evo_mechanics(_evo_raw).items():
                stats.setdefault(k, v)
            self.hp = stats.get('hp', self.hp)
            self.evo = stats
            self._evo_starting_pending = bool(stats.get('onStartingActionData'))

    def to_dict(self):
        d = super().to_dict()
        d.update({'type': 'building'})
        return d

    def take_damage(self, amount: float, delayed=False, source=None, pierce_invincible=False):
        # pierce_invincible 仅对亡影类无敌部队有意义；建筑无该状态，收下保持 AreaEffect
        # 调用口径一致（AreaEffect._pulse else 分支对所有非塔目标统一传参）
        super().take_damage(amount, delayed, source)
        if self.data.name == 'KingTower' and not self.tower_active:
            self.tower_active = True

    def update(self, dt: float):
        """Update building - only attack, no movement"""
        if not self.is_alive: return
        if getattr(self, '_evo_starting_pending', False):
            # M4/M5 觉醒建筑出场动作（觉醒特斯拉出场眩晕脉冲 → _tesla_evo_pulse 数据化）
            self._evo_starting_pending = False
            osad = (self.evo or {}).get('onStartingActionData') or {}
            if 'Tesla' in str(osad.get('name', '')):
                _tesla_evo_pulse(self.battle_state, self, osad)
        if self.data.name == 'KingTower' and not self.tower_active: return
        if self.deploy_delay_remaining > 0:
            self.deploy_delay_remaining = max(0.0, self.deploy_delay_remaining - dt)
            return
        # —— M2 族4：建筑同样受冰冻（Freeze/Tesla 脉冲）——
        if self.freeze_timer > 0:
            self.freeze_timer -= dt
            return
        # —— M5 觉醒补全：觉醒建筑逐 tick 钩子（GoblinDrill 隐匿 / GoblinCage 捕获）——
        if self._evo_building_tick(dt): return
        super().update(dt)
        if self.data.lifetime > 0 and not self.persistent:
            # M2 修复：此前同一帧 take_damage 两次（衰减速率 ×2）
            decay = (self.data.hp / float(self.data.lifetime)) * dt
            self.take_damage(decay)
        if self.attack_cooldown > 0:
            self.attack_cooldown = max(0, self.attack_cooldown-dt*self.speed_buff*self.speed_debuff*self.hit_speed_mult)
        target = self.update_current_target()
        # —— M8 ①：Knight Hero 嘲讽覆盖（建筑同样可被嘲讽，含公主/国王塔）——
        target = self._hero_taunt_override(target)
        if target and self.in_attack_range(target) and self.attack_cooldown <= 0:
            if self.data.projectiles:
                # M1: 建筑（地狱塔类）的蓄力阶段对弹道伤害生效
                self.create_projectile(target, damage_override=self.ramped_damage(self.data.projectile_data.damage))
            else:
                target.take_damage(self.ramped_damage(self.data.damage))
            self.attack_cooldown = self.data.hit_speed

    # ==================== M5 觉醒补全：觉醒建筑钩子 ====================
    def _evo_on_death(self):
        """M5 觉醒补全：觉醒建筑死亡钩子。
        · GoblinDrill_EV1：deathSpawnCount=2 + deathSpawnCharacterData(Goblin) 钻出哥布林
        · GoblinCage_EV1：亡语出觉醒斗士（GoblinCage_EV1_GoblinBrawler，旧 Troop 路径管不到建筑）"""
        evo = self.evo
        if not evo: return
        dsd = evo.get('deathSpawnCharacterData') or {}
        n = int(evo.get('deathSpawnCount') or 0)
        if dsd.get('name'):
            try:
                card = Card(dsd['name'])
            except Exception:
                return
            count = max(1, n) if n else 1
            card.spawn_number = count
            card.spawn_delay = 0
            for p in get_spawn_position(card, self.position, self.player):
                self.battle_state._spawn_entity(
                    Troop(self.battle_state.next_entity_id, p, self.player, dsd['name'], self.battle_state))

    def _evo_building_tick(self, dt):
        """M5 觉醒补全：觉醒建筑逐 tick 钩子。返回 True = 本帧跳过常规行为（隐匿中）。
        ① GoblinDrill_EV1 隐匿迁移：hideHpThresholds [66,33]% 各触发一次 → hideTime 1000ms
           不可选取/无敌 + spawnCharacterOnHide(Goblin)×spawnCharacterOnHideCounts[1,1]；
           官方同时迁移位置（数据无落点字段 → 原地隐匿，标注假设）。
        ② GoblinCage_EV1 捕获：captureRadius 3000 / hitFrequency 1000 / damagePerHit 132 /
           numberOfUnitsToCapture 1 / targetFilter 地面部队（不含建筑）——被捕获者持续眩晕+受伤。"""
        evo = self.evo
        if not evo: return False
        bs = self.battle_state
        # ① 隐匿迁移
        th = evo.get('hideHpThresholds')
        if th and not getattr(self, '_evo_hidden', False):
            pending = getattr(self, '_evo_hide_pending', None)
            if pending is None:
                pending = list(th)
                self._evo_hide_pending = pending
            frac = self.hp / max(self.data.hp, 1) * 100
            if pending and frac <= pending[0]:
                pending.pop(0)
                self._evo_hidden = True
                self._evo_hide_timer = (evo.get('hideTime') or 1000) / 1000.0
                self.targetable = False
                self.invincible = True
                _name = evo.get('spawnCharacterOnHide')
                _counts = evo.get('spawnCharacterOnHideCounts') or [1, 1]
                _n = _counts[len(th) - len(pending) - 1] if len(_counts) > len(th) - len(pending) - 1 >= 0 else 1
                if _name:
                    for i in range(int(_n)):
                        _t = Troop(bs.next_entity_id,
                                   Position(self.position.x + 0.4 * (i + 1), self.position.y),
                                   self.player, _name, bs)
                        bs._spawn_entity(_t)
                return True
        if getattr(self, '_evo_hidden', False):
            self._evo_hide_timer -= dt
            if self._evo_hide_timer <= 0:
                self._evo_hidden = False
                self.targetable = True
                self.invincible = False
            return True  # 隐匿期间不攻击不衰减
        # ② 哥布林笼捕获
        cap_r = evo.get('captureRadius')
        if cap_r:
            self._evo_capture_cd = max(0.0, getattr(self, '_evo_capture_cd', 0.0) - dt)
            cap_r = cap_r / 1000.0
            tgt = bs.entities.get(getattr(self, '_evo_captured_id', None)) if getattr(self, '_evo_captured_id', None) else None
            if tgt is None or not tgt.is_alive or not isinstance(tgt, Troop) \
                    or tgt.data.is_air_unit or tgt.position.distance_to(self.position) > cap_r + 1.0:
                tgt = None
                self._evo_captured_id = None
            if tgt is None and self._evo_capture_cd <= 0:
                best, best_d = None, cap_r
                for e in list(bs.entities.values()):
                    if not isinstance(e, Troop) or not e.is_alive or e.player == self.player: continue
                    if e.data.is_air_unit or not e.targetable: continue  # targetFilter：地面部队
                    d = e.position.distance_to(self.position)
                    if d < best_d: best, best_d = e, d
                if best is not None:
                    tgt = best
                    self._evo_captured_id = best.id
            if tgt is not None and self._evo_capture_cd <= 0:
                # numberOfUnitsToCapture=1：同一时刻只捕获一个；每捕获周期（hitFrequency 1s）
                # 刷新束缚 + damagePerHit（132×等级缩放）
                tgt.apply_buff(stun=(evo.get('hitFrequency') or 1000) / 1000.0)
                tgt.take_damage((evo.get('damagePerHit') or 132) * level_scale(self.level))
                self._evo_capture_cd = (evo.get('hitFrequency') or 1000) / 1000.0
            return False
        # ③ M6 ⑦：Furnace（FirespiritHut_EV1）热生成——攻击期间 2.4s/灵 + 左右侧向交替
        # （原正面；数值同原版 identical stats；基础建筑出兵循环引擎未建模, 觉醒钩子自带）。
        hs = (evo.get('evo2025Hooks') or {}).get('hotSpawn')
        if hs:
            if self.deploy_delay_remaining <= 0:
                self._evo_hotspawn_timer = getattr(self, '_evo_hotspawn_timer', hs.get('interval', 2.4))
                self._evo_hotspawn_timer -= dt
                if self._evo_hotspawn_timer <= 0:
                    self._evo_hotspawn_timer = hs.get('interval', 2.4)
                    side = getattr(self, '_evo_hotspawn_side', -1)   # 首个左侧（Fandom 策略节）
                    self._evo_hotspawn_side = -side                  # 左右交替
                    fwd = hs.get('forwardOffset', 0.8) * (1 if self.player == 0 else -1)
                    pos = Position(self.position.x + hs.get('sideOffset', 0.8) * side,
                                   self.position.y + fwd)
                    bs._spawn_entity(Troop(bs.next_entity_id, pos, self.player,
                                           hs.get('card', 'FireSpirits'), bs))
        return False

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
        # —— M7：Vines 藤蔓束缚（弹道落地 → 束缚领域, 不走通用溅射：
        # 官方语义只打 3 个最高 HP 目标 + 拽落 + 束缚, 与 splash 全域伤害不同）——
        if self.proj.name == 'VinesProjectile':
            spawn_vines_zone(self.battle_state, self, self.target_position)
            self._chain(self.target_position)
            return
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
                tgt.take_damage(self._damage(), source=self.source)
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
        self._evo_impact(impact)

    def _evo_impact(self, impact):
        """M5 觉醒补全：觉醒弹道命中钩子（弹道到达点触发）。
        ① 链电（chainedHitCount，ElectroDragon_EV1 doAttackAction 攻击组）：
           主目标命中后向周围链式弹射 count-1 次，每次同伤害 + ZapFreeze 眩晕。
        ② 二段爆炸（spawnChain，Bomber_EV1：BombSkeletonProjectile_2_EV1 damage 88）。
        ③ 效果领域（spawnAreaEffectObjectData：Firecracker_EV1 烟花灼烧带减速 /
           IceSpirits_EV1 冰雾伤害+冰冻；onHitTargetActionData → 友方狂暴领域）。"""
        src = getattr(self, 'source', None)
        if not isinstance(src, Entity) or not getattr(src, 'evo', None): return
        pd = src.evo.get('evoProjectileData') or {}
        lv = level_scale(src.level)
        # ① 链电
        chain = getattr(src, 'evo_chain', None)
        if chain and self.target is not None and hasattr(self.target, 'take_damage'):
            hit_ids = {self.target.id}
            node = self.target
            for _ in range(max(0, chain['count'] - 1)):
                nxt, nxt_d = None, chain['radius']
                for e in list(self.battle_state.entities.values()):
                    if not isinstance(e, (Troop, Building)) or not e.is_alive: continue
                    if e.player == src.player or e.id in hit_ids or not e.targetable: continue
                    if e.data.is_air_unit and not src.data.attack_air: continue
                    if (not e.data.is_air_unit) and not src.data.attack_ground: continue
                    d = node.position.distance_to(e.position)
                    if d < nxt_d: nxt, nxt_d = e, d
                if nxt is None: break
                nxt.take_damage(self._damage(), delayed=True, source=src)
                if chain['stun']: nxt.apply_buff(stun=chain['stun'])
                hit_ids.add(nxt.id)
                node = nxt
        # ② 二段爆炸链
        bc = getattr(src, 'evo_bomber_chain', None)
        if bc:
            # 二段弹半径无字段 → 默认 1.0（待 L4）
            self.battle_state.deal_area_damage(src.player, impact, 1.0, bc['damage'] * lv, True, True)
        # ③ 效果领域
        ae = pd.get('spawnAreaEffectObjectData')
        if ae and isinstance(ae, dict):
            spawn_evo_zone(self.battle_state, src, ae, impact)
            # IceSpirits：onHitTargetActionData（spawnTime 3000）→ 命中点留下友方狂暴领域
            # （IceSpirits_Target_EV1 tid=TID_SPELL_RAGE；数据无倍率 → 按官方 Rage 现行 1.30，标注默认）
            ohta = pd.get('onHitTargetActionData') or {}
            if ohta.get('spawnTime'):
                _bf = ohta.get('spawnDataData') or {}
                if _bf:
                    zone = EvoEffectZone(
                        self.battle_state.next_entity_id, Position(impact.x, impact.y), src.player,
                        self.battle_state,
                        radius=(ae.get('radius') or 2000) / 1000,
                        lifetime=ohta['spawnTime'] / 1000.0,
                        ally_buff={'speed_mult': 1.30, 'hit_speed_mult': 1.30, 'duration': 0.6},
                        level=src.level, label=_bf.get('name') or 'IceSpirits_Target_EV1')
                    self.battle_state._spawn_entity(zone)

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
                # 勘误批2（2026-09-08）：滚动弹到终点也要走出兵链（BarbLog 滚木终点
                # 出 1 野蛮人；此前 rolling 分支直接消亡，_chain 永不触发）。
                self._chain(self.position)
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
                # —— 勘误批2+：法术 impact 击退（pushback ms→格；此前仅 Log rolling 分支生效）——
                _pb = getattr(self.proj, 'pushback', 0)
                if _pb and isinstance(entity, Troop):
                    _d = entity.position.distance_to(self.target_position) or 1.0
                    _step = min(_pb, _d)
                    entity.position.x += (entity.position.x - self.target_position.x) / _d * _step
                    entity.position.y += (entity.position.y - self.target_position.y) / _d * _step
                if self.proj.buff_time:
                    entity.speed_debuff = min(1 + self.proj.target_buff.get('speedMultiplier', 0) / 100, entity.speed_debuff)
                    entity.debuff_time_remaining = self.proj.buff_time
                    if getattr(self.proj, 'target_buff_death_spawn', None):
                        entity.voodoo_curse = {'player': self.player, 'name': self.proj.target_buff_death_spawn.get('name'),
                                               'until': self.battle_state.time + 5.0}   # 勘误批13：诅咒 5s 过期【Fandom】

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
        # crown_pct：行级缺省回退 buff_data（Earthquake/Poison 的对塔降伤在 bd 里）
        _crown_raw = (row.get('crown_tower_damage_percent')
                      or (bd.get('crown_tower_damage_percent') or 0))
        self.crown_pct = ov.get('crown_tower_percent', _crown_raw + 100) / 100 \
            if (ov.get('crown_tower_percent') or _crown_raw) else None
        # —— 勘误批3：对建筑伤害加成（Earthquake building_damage_percent=350 → ×4.5）——
        self.building_mult = 1.0 + (bd.get('building_damage_percent') or 0) / 100.0
        # —— 勘误批3：减速（Earthquake -50% / Poison -15%），时长=buff_time（残留窗口）——
        _sm = bd.get('speed_multiplier') or 0
        self.speed_mult = (100 + _sm) / 100.0 if _sm else None
        self.freeze_single_shot = (self.buff_name == 'Freeze')  # Freeze 伤害施法单次（官方裁决）
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
            elif bn == 'Earthquake' or bn == 'Poison':
                # 勘误批3：减速不再仅表现层（bd.speed_multiplier 消费）
                if self.speed_mult:
                    e.apply_buff(speed_mult=self.speed_mult, duration=self.buff_time or 1.0)
            elif bn == 'Rage':
                # 官方现行：+30%、效果残留 1s（2025/10 调整），覆盖快照旧值 +35%
                e.apply_buff(speed_mult=ov.get('speed_mult', 1.30), duration=ov.get('residue', 1.0))
            elif bn == 'Heal':
                if self.heal_per_tick:
                    e.apply_buff(heal={'hps': self.heal_per_tick / self.tick, 'time': self.tick})
            # Tornado/Earthquake/Poison 的 buff 仅表现层，无属性效果
            # —— 伤害 ——
            if self.damage_per_tick and not self.only_own_troops:
                # 勘误批10 修复：Rage 类 only_own_troops 光环不再反向伤害己方
                if isinstance(e, Building) and self.crown_pct is not None and (
                        'King' in e.name or 'PrincessTower' in e.name):
                    # 勘误批3：对塔降伤覆盖公主塔（此前只对 'King' 名生效）
                    e.take_damage(self.damage_per_tick * self.crown_pct)
                elif isinstance(e, Building) and self.building_mult != 1.0:
                    e.take_damage(self.damage_per_tick * self.building_mult)
                else:
                    # M6：法术伤害穿透亡影无敌（觉醒骷髅军团亡影「can be affected by spells」）
                    e.take_damage(self.damage_per_tick,
                                  pierce_invincible=bool(getattr(e, '_evo2025_is_shadow', False)))
        # —— 勘误批6（裁决包）：Freeze 伤害=施法单次, 首次 pulse 后关闭伤害轨 ——
        if self.freeze_single_shot and self.damage_per_tick:
            self.damage_per_tick = 0

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
        # —— 勘误批10：Rage 施法伤害（对敌一次性, 官方现行机制; 快照无字段 → Fandom 179@lv11【待对拍】）——
        if self.card_name == 'Rage' and not getattr(self, '_rage_hit', False):
            self._rage_hit = True
            from card_utils import level_scale as _ls
            _rd = 179 * (1.1 ** (Card.default_level - 11))   # 179 为 lv11 基准【Fandom】
            for e in list(self.battle_state.entities.values()):
                if not e.is_alive or e.player == self.player: continue
                if isinstance(e, (Projectile, SpawnProjectile, AreaEffect)): continue
                if not self._in_radius(e): continue
                if isinstance(e, Building):
                    e.take_damage(_rd * 0.3)   # 塔伤 ~30%【Fandom】
                else:
                    e.take_damage(_rd)
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


# ==================== M5 觉醒补全：通用动作组解释器 + 觉醒效果领域 ====================
# 动作语义参考：内存 dump re/cr_dump iDrag_chaos_1 动作链（ActionTakeDamage 的
# Damage.BaseDamage + SubActionsDelay + Resolver/Shape/Filter 结构）；
# 数值一律取自 gamedata.json evolvedSpellsData，缺失即跳过或用注释标注的默认值。

class EvoEffectZone(Entity):
    """M5 觉醒补全：觉醒效果领域通用实体（由动作组解释器从 spawnDataData 构建，
    area_effect_objects_evo / character_buffs_evo 定义驱动）。
    支持：持续伤害（damagePerSecond × hitFrequency 脉冲）、减速/冰冻（speedMultiplier /
    hitSpeedMultiplier=-100）、进入吸引（attractPercentage，格/s）、友方增益（狂暴领域）。
    刻意不走 Entity.__init__（同 AreaEffect：无卡牌身份，避免同名机制类钩子）。"""
    def __init__(self, id, position, player, battle_state, radius, lifetime,
                 dps=0.0, tick=0.5, slow=None, stun_pulse=0.0, attract=0.0,
                 ally_buff=None, level=11, label='EvoEffectZone'):
        self.id, self.position, self.player = id, position, player
        self.battle_state = battle_state
        self.is_alive = True
        self.targetable = False
        self.invincible = True
        self.radius, self.lifetime = radius, lifetime
        self.dps, self.tick = dps, tick
        self.slow, self.stun_pulse, self.attract = slow, stun_pulse, attract
        self.ally_buff = ally_buff or {}
        self.level = level
        self.card_name = label
        self.name = label
        self.data = _EffectShim(radius)
        self.pending_damage = []
        self.path = []
        self.jumping_across_river = False
        self.shield_health = 0
        self.hp = 0
        self.tick_timer = 0.0

    def update(self, dt):
        if not self.is_alive: return
        scale = level_scale(self.level)
        pulse_dmg = self.dps * self.tick * scale
        for e in list(self.battle_state.entities.values()):
            if not e.is_alive or e is self: continue
            if isinstance(e, (Projectile, SpawnProjectile, AreaEffect, EvoEffectZone,
                              EvoZapZone, GenericBomb, TimedExplosive)): continue
            if e.position.distance_to(self.position) > self.radius + e.data.collision_radius: continue
            if e.player != self.player:
                if self.dps: e.take_damage(pulse_dmg)
                if self.slow: e.apply_buff(speed_mult=self.slow, duration=self.tick * 1.1)
                if self.stun_pulse: e.apply_buff(stun=self.stun_pulse)
                if self.attract and isinstance(e, Troop):
                    self.battle_state.pull_enemies(self.player, self.position, self.radius,
                                                   self.attract * dt)
            elif self.ally_buff and isinstance(e, (Troop, Building)):
                ab = self.ally_buff
                e.apply_buff(speed_mult=ab.get('speed_mult'), hit_speed_mult=ab.get('hit_speed_mult'),
                             duration=ab.get('duration', 0.6))
        self.lifetime -= dt
        if self.lifetime <= 0: self.is_alive = False

    def take_damage(self, amount, delayed=False, source=None): pass

    def to_dict(self):
        return {'type': 'evo_zone', 'card_name': self.name, 'player': self.player,
                'x': self.position.x, 'y': self.position.y, 'hp': 0, 'max_hp': 0,
                'shield_max_hp': 0, 'shield_hp': 0, 'collision_radius': self.radius}


def spawn_evo_zone(bs, src, sd, position, action=None):
    """M5 觉醒补全：由区域/buff 定义（spawnDataData）构建 EvoEffectZone。
    sd 支持 radius/lifeDuration/hitSpeed/buffTime + buffData{damagePerSecond, hitFrequency,
    speedMultiplier, hitSpeedMultiplier, attractPercentage}；sd.damage 为单次 AoE 伤害。"""
    if not isinstance(sd, dict): return None
    radius = (sd.get('radius') or 2000) / 1000
    life = (sd.get('lifeDuration') if sd.get('lifeDuration') is not None else 3000) / 1000
    bf = sd.get('buffData') or {}
    dps = bf.get('damagePerSecond') or sd.get('damagePerSecond') or 0
    hf = bf.get('hitFrequency') or sd.get('hitSpeed') or 500
    tick = max(hf, 50) / 1000.0 if hf else 0.5
    slow = None
    sm = bf.get('speedMultiplier', 0)
    if sm < 0: slow = 1 + sm / 100.0   # -15 → ×0.85 减速
    stun_pulse = 0.0
    if bf.get('hitSpeedMultiplier', 0) == -100:   # -100 = 完全停顿（冰冻/眩晕）
        stun_pulse = (sd.get('buffTime') or bf.get('buffTime') or 500) / 1000.0
    attract = (bf.get('attractPercentage') or 0) / 100.0
    # 单次 AoE 伤害（如 IceSpirits_EV1 冰雾 damage 43）
    if sd.get('damage'):
        bs.deal_area_damage(src.player, position, radius, sd['damage'] * level_scale(src.level),
                            True, True)
    zone = EvoEffectZone(bs.next_entity_id, Position(position.x, position.y), src.player, bs,
                         radius=radius, lifetime=life, dps=dps, tick=tick, slow=slow,
                         stun_pulse=stun_pulse, attract=attract, level=src.level,
                         label=sd.get('name') or 'EvoEffectZone')
    bs._spawn_entity(zone)
    return zone


# ==================== M7：Vines 藤蔓束缚（数据接入机制 2）====================

class HealAuraZone(Entity):
    """【勘误批1】治疗光环（BattleHealer, 用户口径 2026-09-04）：
    寿命 1s、每 0.25s 治疗半径内友军部队一跳（共 4 跳）; 不治疗光环源（exclude_id）;
    仅部队（建筑受疗【待确认】, 当前不含）; 对空对地。刻意不走 Entity.__init__
    （同 EvoEffectZone：无卡牌身份, 避免同名机制类钩子）。"""
    def __init__(self, id, position, player, radius, heal_per_tick,
                 ticks=4, interval=0.25, exclude_id=None, label='HealAura'):
        self.id, self.position, self.player = id, position, player
        self.battle_state = None
        self.is_alive = True
        self.targetable = False
        self.invincible = True
        self.radius = radius
        self.heal = heal_per_tick
        self.ticks_target = ticks
        self.pulses_done = 0
        self.interval = interval
        self.exclude_id = exclude_id
        self.tick_timer = 0.0
        self.card_name = label
        self.name = label
        self.level = 11
        self.data = _EffectShim(radius)
        self.pending_damage = []
        self.path = []
        self.jumping_across_river = False
        self.shield_health = 0
        self.hp = 0

    def update(self, dt):
        if not self.is_alive: return
        self.tick_timer += dt
        # 脉冲计数制（不按寿命扣减：浮点漂移会吞掉最后一跳）
        while self.tick_timer >= self.interval and self.pulses_done < self.ticks_target:
            self.tick_timer -= self.interval
            self.pulses_done += 1
            self._pulse()
        if self.pulses_done >= self.ticks_target:
            self.is_alive = False

    def _pulse(self):
        bs = self.battle_state
        if bs is None: return
        for e in list(bs.entities.values()):
            if not e.is_alive or e.player != self.player or e.id == self.exclude_id:
                continue
            if not isinstance(e, Troop):      # 仅部队【建筑待确认】
                continue
            if e.position.distance_to(self.position) > self.radius + getattr(e.data, 'collision_radius', 0):
                continue
            e.hp = min(e.data.hp, e.hp + self.heal)


class VinesSnareZone(Entity):
    """【M7】Vines 藤蔓束缚领域（gamedata Vines.areaEffectObjectData 驱动）。
    投掷物落地生成 → 锁定半径内 HP 最高的 multipleTargets=3 个敌人（targetHighestHp）
    → 每目标至多 ticks=2 跳伤害（lv11 每跳 153, Epic 轴; 对王塔类 crownTowerDamagePercent=25%）
    → 命中目标立即束缚（Vines_Trap_Snare：2.5s 不可移动/攻击, apply_buff(stun) 语义）
    → 空中单位被拽落（groundsAirUnits：flying 临时关闭, 落地时长=束缚时长, 期间可被地面单位攻击）。
    ⚠️ 口径冲突记录：属性表领域持续 lifeDuration=2s vs 卡面束缚 2.5s——领域存在 2s、
    束缚 buff 2.5s（两跳伤害只在领域存续期间结算, 束缚可越出领域 0.5s）。
    刻意不走 Entity.__init__（同 EvoEffectZone：无卡牌身份, 避免同名机制类钩子）。"""
    def __init__(self, id, position, player, battle_state, radius, lifetime, damage,
                 hits=2, crown_pct=0.25, snare_duration=2.5, max_targets=3,
                 grounds_air=True, level=11, label='Vines_AeO'):
        self.id, self.position, self.player = id, position, player
        self.battle_state = battle_state
        self.is_alive = True
        self.targetable = False
        self.invincible = True
        self.radius, self.lifetime = radius, lifetime
        self.damage, self.hits, self.crown_pct = damage, hits, crown_pct
        self.snare_duration, self.max_targets = snare_duration, max_targets
        self.grounds_air = grounds_air
        self.level = level
        self.card_name = label
        self.name = label
        self.data = _EffectShim(radius)
        self.pending_damage = []
        self.path = []
        self.jumping_across_river = False
        self.shield_health = 0
        self.hp = 0
        self.locked = None            # 落地时锁定的目标（首个 update tick 选定）
        self.hits_dealt = 0           # 已结算跳数
        self.tick_timer = lifetime / max(hits, 1)  # 跳间隔 = 领域时长/跳数（2s/2 = 1s）

    def _lock_targets(self):
        """落地锁定：onlyEnemies + hitsGround/hitsAir 过滤, 按 HP 降序取前 N（含建筑）。"""
        targets = []
        for e in list(self.battle_state.entities.values()):
            if not e.is_alive or e.player == self.player: continue
            if isinstance(e, (Projectile, SpawnProjectile, AreaEffect, EvoEffectZone,
                              EvoZapZone, VinesSnareZone, GenericBomb, TimedExplosive)): continue
            if e.data.is_air_unit and not self.grounds_air: continue
            if e.position.distance_to(self.position) > self.radius + e.data.collision_radius: continue
            targets.append(e)
        targets.sort(key=lambda x: (x.hp + x.shield_health), reverse=True)
        locked = targets[:self.max_targets]
        for t in locked:
            # 束缚：2.5s 不可移动/攻击（Vines_Trap_Snare；分档 Small..XXLarge 待 L4, 统一档）
            t.apply_buff(stun=self.snare_duration)
            # 拽落：空中单位临时落地, 落地时长=束缚时长, 期间可被地面单位攻击
            if self.grounds_air and getattr(t.data, 'is_air_unit', False) and isinstance(t, Troop):
                t.data.is_air_unit = False
                t._vines_grounded_time = max(getattr(t, '_vines_grounded_time', 0.0),
                                             self.snare_duration)
                # 绝对复飞时刻（束缚期间 update 早退, 相对计时不可靠）
                t._vines_grounded_until = self.battle_state.time + self.snare_duration
                t.path = []
        return locked

    def _deal_hit(self):
        for t in self.locked:
            if not t.is_alive: continue
            # 对王塔类按 crownTowerDamagePercent（口径同 AreaEffect._pulse：仅 'King' 类塔降伤,
            # 普通建筑全额）——gamedata 25%（Fandom 2026/6/1 已削至 23%, 快照未跟, 待 L4）
            dmg = self.damage * self.crown_pct if ('King' in t.name) else self.damage
            # 即时结算（不走 delayed）：被束缚目标 update 早退, pending 伤害永不结算
            t.take_damage(dmg, source=None)

    def update(self, dt):
        if not self.is_alive: return
        if self.locked is None:
            self.locked = self._lock_targets()
            self._deal_hit()          # 第 1 跳：落地即时
            self.hits_dealt = 1
        else:
            self.tick_timer -= dt
            if self.tick_timer <= 0 and self.hits_dealt < self.hits:
                self._deal_hit()      # 第 2 跳
                self.hits_dealt += 1
                self.tick_timer += self.lifetime / max(self.hits, 1)
        self.lifetime -= dt
        if self.lifetime <= 0:
            self.is_alive = False

    def take_damage(self, amount, delayed=False, source=None): pass

    def to_dict(self):
        return {'type': 'vines_zone', 'card_name': self.name, 'player': self.player,
                'x': self.position.x, 'y': self.position.y, 'hp': 0, 'max_hp': 0,
                'shield_max_hp': 0, 'shield_hp': 0, 'collision_radius': self.radius}


def spawn_vines_zone(bs, projectile, impact):
    """【M7】Vines 弹道落地 → 束缚领域。数据全量取自 gamedata
    Vines.areaEffectObjectData（radius/lifeDuration/ticks/multipleTargets/targetHighestHp/
    groundsAirUnits/crownTowerDamagePercent/buffData.snareDurationMs）；
    伤害用弹道已按战斗卡牌等级解析的 per-hit 伤害（projectiles 行 Epic 轴 lv11=153）。"""
    aeo = (Card('Vines').data.get('areaEffectObjectData') or {})
    bf = aeo.get('buffData') or {}
    zone = VinesSnareZone(
        bs.next_entity_id, Position(impact.x, impact.y), projectile.player, bs,
        radius=(aeo.get('radius') or 2500) / 1000,
        lifetime=(aeo.get('lifeDuration') or 2000) / 1000,
        damage=projectile._damage(),
        hits=int(aeo.get('ticks') or 2),
        crown_pct=(aeo.get('crownTowerDamagePercent') or 25) / 100,
        snare_duration=(bf.get('snareDurationMs') or 2500) / 1000,
        max_targets=int(aeo.get('multipleTargets') or 3),
        grounds_air=bool(aeo.get('groundsAirUnits')),
        level=Card.default_level,
        label=aeo.get('name') or 'Vines_AeO')
    bs._spawn_entity(zone)
    return zone


def _spawn_action_character(bs, player, sd, position=None):
    """M5 觉醒补全：动作组 SpawnEnemy 类语义——按内嵌角色定义出兵。
    定义已由 card_utils._scan_and_register 登记为可构造卡（Wallbreaker_mini/GoblinDummy/
    Goblin 等），per-level 曲线继承。"""
    name = sd.get('name')
    if not name: return False
    try:
        card = Card(name)
    except Exception:
        return False
    card.spawn_number = 1
    card.spawn_delay = 0
    for p in get_spawn_position(card, position, player, False):
        t = Troop(bs.next_entity_id, p, player, name, bs)
        bs._spawn_entity(t)
    return True


def interpret_action_group(entity, action, target=None, position=None):
    """M5 觉醒补全：通用动作组解释器入口。
    解析 evolvedSpellsData 内嵌动作组（subActionsData / actionsData / onHitActionData /
    onHitTargetActionData / doAttackAction 引用组等），执行其中有战斗语义的动作：
      · ActionTakeDamage 类 → 直接伤害（dump 参考：Damage.BaseDamage）
      · SpawnEnemy 类 → spawnDataData 内嵌角色定义出兵
      · 区域效果类 → radius/lifeDuration/damagePerSecond/buffData → EvoEffectZone
      · buff 类 → hitSpeedMultiplier/speedMultiplier/damagePerSecond
    纯视觉动作（*Effect / *Anim / Export / shadow / tid*）无战斗语义，自然跳过。
    返回已执行的动作类型列表（供测试断言）。"""
    executed = []
    if not isinstance(action, dict) or entity.battle_state is None:
        return executed
    bs = entity.battle_state
    # ① 直接伤害
    dmg = action.get('damage')
    if not dmg and isinstance(action.get('Damage'), dict):
        dmg = action['Damage'].get('BaseDamage')
    if dmg and target is not None and hasattr(target, 'take_damage'):
        target.take_damage(dmg * level_scale(entity.level), delayed=True, source=entity)
        executed.append('damage')
    # ② 出兵 / ③ 区域效果：spawnDataData 分派（角色定义 vs 区域/buff 定义）
    sd = action.get('spawnDataData')
    if isinstance(sd, dict) and sd.get('hitpoints') and not sd.get('radius'):
        if _spawn_action_character(bs, entity.player, sd, position):
            executed.append('spawn')
    elif isinstance(sd, dict) and (sd.get('radius') or sd.get('damagePerSecond') or sd.get('buffData')):
        if spawn_evo_zone(bs, entity, sd, position or (target.position if target is not None else entity.position)):
            executed.append('zone')
    # ④ 动作节点自身即区域定义（Zap_EV1_SpawnAOE_medium 等叶子）
    if not sd and action.get('radius') and action.get('lifeDuration') is not None:
        if spawn_evo_zone(bs, entity, action, position or entity.position):
            executed.append('zone')
    # ⑤ 子动作组递归（ActionGroup 语义）
    for key in ('subActionsData', 'actionsData'):
        for sub in action.get(key) or []:
            executed += interpret_action_group(entity, sub, target, position)
    return executed


def _tesla_evo_pulse(bs, entity, osad):
    """M5 觉醒补全：觉醒特斯拉出场脉冲（Tesla_EV1_AppearStun）。
    半径 maxRadius 6000 内眩晕 lifeDuration=1500ms（此前简化为 0.5s），
    并按 onHitActionData buff（Tesla_EV1_WithDamage：damagePerSecond 68 /
    hitFrequency -1 一次性）结算 DOT 伤害（dps × 持续时间）。"""
    sd = osad.get('spawnDataData') or {}
    radius = (sd.get('maxRadius') or 6000) / 1000
    stun = (sd.get('lifeDuration') or 1500) / 1000.0
    bsd = ((sd.get('onHitActionData') or {}).get('spawnDataData') or {})
    dps = bsd.get('damagePerSecond', 0)
    scale = level_scale(entity.level)
    for e in list(bs.entities.values()):
        if not e.is_alive or e.player == entity.player: continue
        if isinstance(e, (Projectile, SpawnProjectile, AreaEffect, EvoEffectZone,
                          EvoZapZone, GenericBomb)): continue
        if e.position.distance_to(entity.position) <= radius:
            e.apply_buff(stun=stun)
            if dps: e.take_damage(dps * scale * stun)


class EvoZapZone(AreaEffect):
    """M5 觉醒补全：觉醒 Zap 领域（evolvedSpellsData.areaEffectObjectData.Zap_EV1：
    半径 2500 / lifeDuration 5000 / buffTime 500，敌人进入即眩晕 0.5s）。
    首脉冲沿用基础 Zap 伤害+眩晕；onStartingActionData 次级 AOE
    （Zap_EV1_SpawnAOE_medium：radius 3000 / lifeDuration 400）一次性眩晕。"""
    def __init__(self, id, position, player, card_name, battle_state=None):
        super().__init__(id, position, player, card_name, battle_state)
        aod = (Card(card_name).evo_raw or {}).get('areaEffectObjectData') or {}
        self.zone_stun = (aod.get('buffTime') or 500) / 1000.0
        self.lifetime = (aod.get('lifeDuration') or 5000) / 1000.0  # 覆盖基础 Zap 瞬发寿命
        self.stunned_ids = set()
        self.medium = None
        for sub in (aod.get('onStartingActionData') or {}).get('subActionsData') or []:
            if sub.get('radius') and sub.get('lifeDuration') is not None \
                    and 'medium' in str(sub.get('name', '')):
                self.medium = sub
        self.medium_done = False
        self.zap_dmg_done = False

    def update(self, dt):
        if not self.is_alive: return
        # 首脉冲：基础 Zap 伤害 + 眩晕（数值表驱动）
        if not self.zap_dmg_done:
            self.zap_dmg_done = True
            self._pulse()
        # 次级 AOE：一次性眩晕（半径 3.0）
        if self.medium and not self.medium_done:
            self.medium_done = True
            r = self.medium.get('radius', 3000) / 1000
            st = (self.medium.get('buffTime') or 500) / 1000.0
            for e in list(self.battle_state.entities.values()):
                if not e.is_alive or e.player == self.player: continue
                if isinstance(e, (Projectile, SpawnProjectile, AreaEffect, EvoEffectZone,
                                  EvoZapZone, GenericBomb)): continue
                if e.position.distance_to(self.position) <= r:
                    e.apply_buff(stun=st)
        # 领域：进入即眩晕（每敌人一次，避免站桩永冻）
        for e in list(self.battle_state.entities.values()):
            if not e.is_alive or e.player == self.player: continue
            if isinstance(e, (Projectile, SpawnProjectile, AreaEffect, EvoEffectZone,
                              EvoZapZone, GenericBomb)): continue
            if e.id in self.stunned_ids: continue
            if e.position.distance_to(self.position) <= self.radius + e.data.collision_radius:
                self.stunned_ids.add(e.id)
                e.apply_buff(stun=self.zone_stun)
        self.lifetime -= dt
        if self.lifetime <= 0: self.is_alive = False


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


# ==================== M8：Elite17 精英卡（Hero 化）引擎机制 ====================
# 规格：docs/elite17_spec.md；数据层：elite17_data.py（Hero 独立数值表 + 能力表）。
# 装配链路：卡组声明 hero_slots（PlayerState.set_hero_slots，Wild slot 互斥觉醒）
# → deploy_card/_wrap 部署时 apply_hero_overlay（数值覆写 + 护盾 + Hero 机制类 holder）
# → BattleState.use_ability M8 分支（单次使用 + 圣水预检 + 条件窗）。

def apply_hero_overlay(entity, bs):
    """M8：Hero 形态装配（部署即 Hero，非普通卡倍率）。
    · 数值覆写：HERO_BASE_STATS 独立表（含来源等级，按官方 1.1/级曲线换算到战斗等级）；
      弹丸卡（Bowler 类 damage=0 走弹丸）同步覆写弹丸伤害。
    · 护盾：Hero overlay ShieldHitpoints → 既有护盾槽（shield_health）。
    · holder 换成 Hero 机制类（HERO_CLASSES，card_mechanics M8 段）。"""
    from elite17_data import HERO_BASE_STATS, hval
    entity.hero_mode = True
    st = HERO_BASE_STATS.get(entity.card_name) or {}
    if st.get('hp'):
        v, src, _note = st['hp']
        entity.data.hp = hval(v, src, entity.level)
        entity.hp = entity.data.hp
    if st.get('damage'):
        v, src, _note = st['damage']
        slot = st.get('damage_target') or ('direct' if entity.data.damage else
               'projectile' if entity.data.projectiles else 'direct')
        if slot == 'projectile':
            # 弹丸卡：伤害在弹丸上（Bowler 类 Hero 主伤害走 projectile_data）
            entity.data.projectile_data.damage = hval(v, src, entity.level)
        else:
            entity.data.damage = hval(v, src, entity.level)
    if st.get('shield'):
        v, src, _note = st['shield']
        entity.shield_health = hval(v, src, entity.level)
    cls = HERO_CLASSES.get(entity.card_name)
    if cls is not None:
        entity.entity_holder = cls(entity)
    return entity


def _barb_log_reroll_effect(bs):
    """M8 ⑮：BarbLog Hero「Rowdy Reroll」效果体（由 use_ability 条件窗分支回调）：
    桶沿原方向二次全车道滚（reset path 语义 = 重置已命中列表、伤害重新结算；
    引擎直接再生成一枚 Rolling 弹）+ 落点附近野蛮人回复 50% 桶伤害。"""
    win = player = None
    for pid in (0, 1):
        w = bs.hero_windows.get(pid)
        if w and w.get('card') == 'BarbLog' and w.get('used'):
            win, player = w, pid
            break
    if win is None: return
    bs.spawn_projectile_chain('BarbLogProjectileRolling', win['origin'], player, win['dir'])
    # —— 勘误批2（2026-09-08，勘误批1 回滚）：Fandom Rowdy Reroll 属性表
    # Damage Healed=50%（docs/_elite_BarbarianBarrel.txt L119），正文同口径；
    # "回满血"系误读。治疗量 = 桶伤害（BarbLogProjectileRolling，Epic 轴 per-level）
    # × 50%，按出桶方战斗等级取值（HERO_ABILITIES['BarbLog']['healPct']）。
    # 治疗范围 = Rowdy Reroll 属性 Range 3 / Width 2.6 → 二次滚车道带内
    #（origin → origin+dir×(滚程) 矩形带，宽 2.6/2=1.3 格；旧"落点 6 格"假设
    # 实测永远滤不到首滚终点出兵（距 origin 6.33），作废）。
    from elite17_data import HERO_ABILITIES
    heal_pct = HERO_ABILITIES['BarbLog'].get('healPct', 0.50)
    _bp = projectiles.get('BarbLogProjectileRolling') or {}
    barrel_dmg = _value_at_level(_bp.get('damage_per_level') or [],
                                 _bp.get('rarity') or 'Epic',
                                 bs.card_level,
                                 _bp.get('damage') or 0)
    heal = barrel_dmg * heal_pct
    _ox, _oy = win['origin'].x, win['origin'].y
    _dx, _dy = win['dir']
    _range = float(HERO_ABILITIES['BarbLog'].get('rerollRange', 3.0))
    _half_w = float(HERO_ABILITIES['BarbLog'].get('width', 2.6)) / 2.0
    for e in list(bs.entities.values()):
        if (not e.is_alive or e.player != player or e.name not in ('Barbarian', 'Barbarians')
                or not isinstance(e, Troop)):
            continue
        # 二次滚车道带：沿滚向投影 ∈ [-1, _range]（滚程可到 4.5 但官方 Range=3，
        # 取并集覆盖：起点后方 1 格到滚程终点），垂向偏差 ≤ Width/2
        along = (e.position.x - _ox) * _dx + (e.position.y - _oy) * _dy
        perp = abs((e.position.x - _ox) * -_dy + (e.position.y - _oy) * _dx)
        if along < -1.0 or perp > _half_w:
            continue
        e.hp = min(e.data.hp, e.hp + heal)


class IceGolemiteSnowZone(EvoEffectZone):
    """M8 ⑰：冰雪光环（Hero IceGolemite）——按目标体型分档施加 Freeze/Slow。
    数值【暂借 Ice Golem/Hero】：半径 4 / 3 次脉冲 / 间隔 1s / 脉冲伤害 69@L11 / 减速 30% 2s；
    分档规则 [假设]（内存 buff 命名 Base/Small/Medium/Large/Tower，参数未解出）：
    小体型（collisionRadius ≤ 0.5）部队 → 冻结 1s；其余部队 → 减速 30% 2s；建筑/塔 → 仅减速。
    【暂借/待实测】标注详见 elite17_data.py HERO_ABILITIES['IceGolemite']。"""
    SMALL_RADIUS = 0.5  # [假设] 小体型阈值（Skeleton 400 / Goblin 400 vs 中型 500+）

    def __init__(self, id, position, player, battle_state, radius, pulses, interval,
                 damage, slow, slow_duration, small_freeze, level, label='IceGolemiteSnowZone'):
        super().__init__(id, position, player, battle_state, radius=radius,
                         lifetime=pulses * interval + 0.01, level=level, label=label)
        self.pulse_damage = damage
        self.interval = interval
        self.pulses_left = pulses
        self.slow_mult, self.slow_duration = slow, slow_duration
        self.small_freeze = small_freeze
        self.pulse_timer = 0.0

    def update(self, dt):
        if not self.is_alive: return
        self.pulse_timer -= dt
        if self.pulse_timer <= 0 and self.pulses_left > 0:
            self.pulse_timer = self.interval
            self.pulses_left -= 1
            for e in list(self.battle_state.entities.values()):
                if not e.is_alive or e is self or e.player == self.player: continue
                if isinstance(e, (Projectile, SpawnProjectile, AreaEffect, EvoEffectZone,
                                  EvoZapZone, GenericBomb, TimedExplosive)): continue
                if e.position.distance_to(self.position) > self.radius + e.data.collision_radius: continue
                e.take_damage(self.pulse_damage)
                if isinstance(e, Troop) and e.data.collision_radius <= self.SMALL_RADIUS:
                    e.apply_buff(stun=self.small_freeze)          # 小体型档：冻结
                else:
                    e.apply_buff(speed_mult=self.slow_mult,        # 中/大体型与建筑塔：减速
                                 duration=self.slow_duration)
        if self.pulses_left <= 0:
            self.is_alive = False


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

        # —— 勘误批7：塔兵=卡组第 9 张，替换两座公主塔（Cannoneer/Duchess/Chef；数值已在
        # cards_stats_building 与 rl/env_wrapper 参考表核对一致——gamedata 快照权威）——
        _r_tt = player_1.tower_troop or 'King_PrincessTowers'
        _b_tt = player_0.tower_troop or 'King_PrincessTowers'
        self._spawn_entity(Building(1, self.arena.RED_LEFT_TOWER, 1, _r_tt, True))
        self._spawn_entity(Building(2, self.arena.RED_RIGHT_TOWER, 1, _r_tt, True))
        self._spawn_entity(Building(3, self.arena.BLUE_LEFT_TOWER, 0, _b_tt, True))
        self._spawn_entity(Building(4, self.arena.BLUE_RIGHT_TOWER, 0, _b_tt, True))
        self._spawn_entity(Building(5, self.arena.RED_KING_TOWER, 1, 'KingTower', True))
        self._spawn_entity(Building(6, self.arena.BLUE_KING_TOWER, 0, 'KingTower', True))

        self.schedule = []
        self.resurrect_queue = []  # M5：觉醒临时复活队列 [(card,pos,player,hp,lifetime,at_time)]
        self.hero_windows = {}     # M8：Hero 条件窗（Goblins 旗窗 / BarbLog 二次滚窗）player → 窗口态
        self._hero_goblin_group_seq = 0      # M8：Hero Goblins 部署波次组 id
        self._hero_goblin_group_state = {}   # M8：player → (gid, 部署时刻)（2s 内部署视为同组）
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
            ent = Entity(*entity_data)
        elif card_name in buildings:
            self.cache_fresh = False
            # M2 修复：此前把 battle_state 对象传入 persistent 槽（恒真值 → 部署建筑寿命衰减失效）
            ent = Building(entity_data[0], entity_data[1], entity_data[2], entity_data[3],
                           persistent=False, evolved=evolved)
        else:
            ent = Troop(entity_data[0], entity_data[1], entity_data[2], entity_data[3],
                        entity_data[4], evolved=evolved)
        # —— M8：Wild slot——卡组声明 hero 化的卡按 Hero 形态装配
        # （Hero 独立数值表 + 护盾 + abilityData + Hero 机制类 holder；见 apply_hero_overlay）——
        if card_name in self.players[entity_data[2]].hero_slots:
            apply_hero_overlay(ent, self)
        return ent

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
            # 加时硬顶（300s）真实 CR 裁决：双方存活塔中血量百分比最低者输；
            # 完全相等才平局（winner 保持 None）。此前 else 恒判 player 1 胜，
            # 且用绝对血量（塔兵改变公主塔最大血时会失真）。
            self.game_over = True
            _m0 = min(self.entities[i].hp / self.entities[i].data.hp
                      for i in (3, 4, 6) if self.entities[i].is_alive)
            _m1 = min(self.entities[i].hp / self.entities[i].data.hp
                      for i in (1, 2, 5) if self.entities[i].is_alive)
            if _m0 > _m1:
                self.winner = 0
            elif _m1 > _m0:
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

        _bonus = getattr(self, '_mirror_level_bonus', 0)
        if _bonus:
            Card.default_level += _bonus   # 勘误批9：镜像出兵等级 +1（同步法术在 deploy 窗口内生效）
        for entity, spawn_time in self.schedule:
            if self.time >= spawn_time: self._spawn_entity(self._wrap(entity))
        self.schedule = [each for each in self.schedule if each[1] > self.time]
        if _bonus:
            Card.default_level -= _bonus
            self._mirror_level_bonus = 0
        # —— M5 觉醒补全：觉醒临时复活队列（Pekka tempResurrect：延迟 2s 原地复活，
        # HP=500×曲线+200×灵魂，临时存活 5s，每场一次）——
        for item in self.resurrect_queue:
            if self.time >= item[5]:
                _t = Troop(self.next_entity_id, item[1], item[2], item[0], self, evolved=True)
                _t.hp = item[3]
                _t._evo_temp_lifetime = item[4]
                _t._evo_resurrect_used = True  # 复活体不再二次复活
                self._spawn_entity(_t)
        self.resurrect_queue = [i for i in self.resurrect_queue if self.time < i[5]]
        self.time += dt
        self.tick += 1

    def _finish_deploy(self, player_id, card_name, from_mirror=False, hand_card=None,
                       actual_cost=None):
        """M1：出牌收尾——扣费并更新卡序；镜像重放的卡不在手牌中，手动送回循环末尾
        M5 觉醒补全：出牌计数统一在此入口（部队/法术一致，觉醒周期按出牌次数交替）
        M7：actual_cost 非 None = 动态费用出牌（Spirit Empress 双费用形态）——
        扣「实际费用」而非卡面费用, 手牌循环仍按手牌名（MergeMaiden）推进。"""
        p = self.players[player_id]
        if Card(card_name).evo_raw:
            p.evo_plays[card_name] = p.evo_plays.get(card_name, 0) + 1
        if from_mirror:
            if card_name in p.cycle:
                p.cycle.remove(card_name)
                p.cycle.append(card_name)
        elif actual_cost is not None:
            _hand = hand_card or card_name
            p.elixir -= actual_cost
            if _hand in p.cycle:
                p.cycle.remove(_hand)
                p.cycle.append(_hand)
            if _hand != 'Mirror': p.last_card = _hand
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

    def _cast_lightning(self, player_id, position):
        """【勘误批8】Lightning：半径 3.5 内最高 HP 的至多 3 个敌方单位/建筑,
        各受一道雷击（伤害+0.5s 眩晕+重索敌）；对塔伤害 ×0.65【快照无字段, Fandom -35%】。"""
        from card_utils import _value_at_level
        _row = spells.get('Lightning', {})
        _prj = projectiles.get('LighningSpell') or {}
        _dpl = _prj.get('damage_per_level') or []
        dmg = _value_at_level(_dpl, _row.get('rarity') or 'Epic', Card.default_level,
                              _prj.get('damage') or 660) if _dpl else (_prj.get('damage') or 660)
        radius = (_row.get('radius') or 3500) / 1000
        cands = []
        for e in list(self.entities.values()):
            if not e.is_alive or e.player == player_id: continue
            if isinstance(e, (Projectile, SpawnProjectile, AreaEffect)): continue
            if e.position.distance_to(position) <= radius + e.data.collision_radius:
                cands.append(e)
        cands.sort(key=lambda x: (x.data.hp if hasattr(x.data, 'hp') else 0), reverse=True)
        for e in cands[:3]:
            _tower = isinstance(e, Building)
            e.take_damage(dmg * (0.65 if _tower else 1.0))
            if isinstance(e, Troop):
                e.apply_buff(stun=0.5, retarget=True)

    def deploy_card(self, player_id, card_name, position, _from_mirror=False):
        # —— M1 镜像法术：重放上一张使用的卡，费用 = 基础费 + 1 ——
        if card_name == 'Mirror':
            p = self.players[player_id]
            if not p.can_play_card('Mirror'): return False
            last = p.last_card
            if not last: return False
            # —— M7：Spirit Empress 双费用形态——镜像复制上一形态并按其「实际费用+1」
            # （官方：镜像沿用上一形态, 费用不再随圣水切档 [Fandom _fp_SpiritEmpress 策略节]）——
            if last == 'MergeMaiden':
                _mirror_base = getattr(p, 'last_card_cost', None) or 3
            else:
                _mirror_base = Card(last).elixir
            if p.elixir < _mirror_base + 1: return False
            # —— 勘误批9：镜像卡等级 +1（官方现行；延迟出兵窗口内统一提升等级）——
            self._mirror_level_bonus = 1
            ok = self.deploy_card(player_id, last, position, _from_mirror=True)
            if not ok: self._mirror_level_bonus = 0
            if ok: p.elixir -= _mirror_base + 1  # _from_mirror 路径不扣基础费，此处一并扣
            return ok
        # —— M7：Spirit Empress（MergeMaiden）双费用形态（部署规则, 非技能）——
        # 圣水 ≥6 → 6 费飞行远程形态（MergeMaiden_Mounted, air, range 5, A&G）；
        # 圣水 <6 → 3 费地面近战形态（MergeMaiden_Normal, ground, melee）。
        # AI/RL 的 action space 语义不变（仍是选这张牌, 费用动态；手牌校验/圣水扣减/
        # 出牌计数全部用实际消耗）。镜像重放沿用上一形态（官方口径, 见 Mirror 分支）。
        _merge_cost = None
        _hand_card = None
        if card_name == 'MergeMaiden':
            p = self.players[player_id]
            if _from_mirror:
                _form = getattr(p, 'last_card_form', None) or 'MergeMaiden_Mounted'
                _merge_cost = 6 if _form == 'MergeMaiden_Mounted' else 3
            else:
                _mounted = p.elixir >= 6
                _merge_cost = 6 if _mounted else 3
                # 手牌校验用实际费用（非卡面 6 费）：在手牌前 4 位 + 圣水足够 + 王塔存活
                if not (card_name in p.cycle[:4] and p.elixir >= _merge_cost
                        and p.king_tower_hp > 0):
                    return False
                _form = 'MergeMaiden_Mounted' if _mounted else 'MergeMaiden_Normal'
            p.last_card_cost = _merge_cost      # 镜像复制形态/费用用
            p.last_card_form = _form
            _hand_card = card_name              # 手牌循环按 MergeMaiden 推进
            card_name = _form
        elif not _from_mirror and not self.players[player_id].can_play_card(card_name):
            return False
        # —— 勘误批1：BarbLog 仅可在与部队相同的部署区域施放（用户口径 2026-09-04,
        # 与其他法术「全场任意」不同）——
        if card_name == 'BarbLog' and not _from_mirror:
            if not self.arena.can_deploy_at(position, player_id, battle_state=self, is_spell=False):
                return False
        card_info = Card(card_name)
        # —— M5 觉醒补全：法术类觉醒形态判定（Zap 觉醒领域 / GoblinBarrel 觉醒诱饵）——
        # M8：Wild slot 互斥——卡组声明 hero 化的卡觉醒禁用（同一 Wild 槽二选一，Hero 优先）
        _p = self.players[player_id]
        evolved = bool(card_info.evo_raw and card_name in _p.evo_slots
                       and card_name not in _p.hero_slots
                       and evolution_state(_p.evo_plays.get(card_name, 0), card_name))

        if card_info.type != 'spell':
            # Check the deployment area is legit
            if self.is_position_occupied_by_building(position, 0): return False
            # —— 勘误批8：Miner 官方可部署全场（含敌半场）——
            if card_name != 'Miner' and player_id == 0:
                if position.y <= 1.0 and (position.x <= 6.0 or position.x > 12.0): return False
                if position.y >= 21.0: return False
                elif position.y >= 15.0:
                    if position.x <= 9:
                        if self.players[1].left_tower_hp > 0: return False
                    else:
                        if self.players[1].right_tower_hp > 0: return False
            elif card_name != 'Miner' and player_id == 1:
                if position.y > 31.0 and (position.x <= 6.0 or position.x > 12.0): return False
                if position.y <= 10: return False
                elif position.y <= 17.0:
                    if position.x <= 9:
                        if self.players[0].left_tower_hp > 0: return False
                    else:
                        if self.players[0].right_tower_hp > 0: return False

        if card_info.type == 'spell':
            srow = spells.get(card_name, {})
            # —— 勘误批8：Lightning 三目标最高 HP + 0.5s 眩晕 + 对塔降伤（原为无行为隐形实体）——
            if card_name == 'Lightning':
                self._cast_lightning(player_id, position)
                self._finish_deploy(player_id, card_name, _from_mirror)
                return True
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
                    clone._soul_excluded = True   # 勘误批11：克隆体不计魂
                    self._spawn_entity(clone)
                self._finish_deploy(player_id, card_name, _from_mirror)
                return True
            # —— M2 族4：瞬发区域法术（Zap/Freeze/Heal/Rage/Tornado/Earthquake/Poison）——
            # 此前这些法术走 get_spawn_position 生成「无行为隐形实体」，伤害/眩晕/拉拽全部无效。
            # 以数值表 projectile 字段判断是否飞行（Heal 有 spellAsDeploy 且 projectile=null → 瞬发）
            if (srow.get('buff') or srow.get('controls_buff')) and not srow.get('projectile'):
                # —— M5 觉醒补全：觉醒 Zap 领域（areaEffectObjectData.Zap_EV1：5s 眩晕领域）——
                if evolved and card_name == 'Zap' and (card_info.evo_raw or {}).get('areaEffectObjectData'):
                    z = EvoZapZone(self.next_entity_id, Position(position.x, position.y),
                                   player_id, card_name, self)
                    self._spawn_entity(z)
                    self._finish_deploy(player_id, card_name, _from_mirror)
                    return True
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
            # —— M5 觉醒补全：觉醒哥布林飞桶诱饵（decoyData → GoblinDummy 假哥布林，
            # 落点旁偏移 1 格；数量无字段 → 1，标注假设）——
            if evolved and card_name == 'GoblinBarrel':
                _decoy = ((card_info.evo_raw.get('projectileData') or {}).get('decoyData')
                          or {}).get('spawnCharacterData') or {}
                if _decoy.get('name'):
                    _spawn_action_character(self, player_id, _decoy,
                                            Position(position.x + 1.0, position.y))
            # —— M8 ⑮：BarbLog Hero——桶滚出后开「二次滚」按钮窗（Rowdy Reroll，单次）；
            # 窗口记录落点与滚向（二次滚沿原方向全车道重滚 + 治疗野蛮人）——
            if card_name == 'BarbLog' and card_name in _p.hero_slots:
                from elite17_data import HERO_ABILITIES
                _ab = HERO_ABILITIES['BarbLog']
                _kt = self.arena.BLUE_KING_TOWER if player_id == 0 else self.arena.RED_KING_TOWER
                _dx, _dy = position.x - _kt.x, position.y - _kt.y
                _n = math.hypot(_dx, _dy) or 1.0
                self.hero_windows[player_id] = {
                    'until': self.time + _ab['window'], 'used': False, 'card': 'BarbLog',
                    'origin': Position(position.x, position.y),
                    'dir': (_dx / _n, _dy / _n),
                    'effect': _barb_log_reroll_effect}
            self._finish_deploy(player_id, card_name, _from_mirror)
            return True

        positions = get_spawn_position(card_info, position, player_id)
        # —— M4 族7：觉醒形态判定（卡组携带觉醒位 + 周期表：cycle=N → 第 N+1 次觉醒，交替）
        # M5：evolved 已在 deploy_card 入口统一判定；出牌计数统一走 _finish_deploy ——
        p = self.players[player_id]
        # —— 队形车道打标：记录每个出兵位置相对部署中心的横向偏移（蓝方 +y 推进向的
        # 法线 = x 轴；红方镜像取反），Troop 行进时据此平移路点实现并排走位。
        # 单位卡 dx=0 无影响；幅度截断 ±0.8 格防窄桥上车道越界。——
        for pos in positions:
            _dx = max(-0.8, min(0.8, pos.x - position.x))
            pos._lane_offset = (1.0 if player_id == 1 else -1.0) * _dx
        delayed_counter = 0
        for pos in positions:
            self.delayed_spawn((len(self.entities)+1, pos, player_id, card_name, self, evolved), delayed_counter)
            delayed_counter += card_info.spawn_delay
        # —— 勘误批2+：双部队卡第二部队（summonCharacterSecondData/SecondCount）——
        # GoblinGang 3×SpearGoblin / Rascals 2×RascalGirl / Goblinstein 1×doctor；
        # 第二波与主部队间隔 0.2s【假设：官方先后部署, 无精确间隔字段, 待 L4】
        _second = card_info.data.get('summonCharacterSecondData')
        if _second and _second.get('name'):
            from card_utils import character_to_card as _c2c2
            _sec_name = _c2c2.get(_second['name'], _second['name'])
            if _sec_name in card_data:
                _sec_cnt = int(card_info.data.get('summonCharacterSecondCount') or 1)
                _sec_info = Card(_sec_name)
                _sec_info.spawn_number = _sec_cnt
                _sec_info.spawn_delay = 0
                _sec_pos = get_spawn_position(_sec_info, position, player_id, False)
                for _sp in _sec_pos:
                    self.delayed_spawn((len(self.entities)+1, _sp, player_id, _sec_name, self, False),
                                       delayed_counter + 0.2)
        # M7：Spirit Empress 动态费用——扣实际消耗, 手牌循环按 MergeMaiden 推进
        self._finish_deploy(player_id, card_name, _from_mirror,
                            hand_card=_hand_card, actual_cost=_merge_cost)
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
        # —— M6 ⑤：SkeletonArmy General Gerry 阵亡 → 其名下全部亡影即刻消散
        # （官方：击败 Gerry 只消灭亡影, 存活骷髅不受影响且此后不再转化）——
        if getattr(entity, '_evo2025_is_gerry', False):
            for e in list(self.entities.values()):
                if getattr(e, '_evo2025_shadow_of', None) == entity.id and e.is_alive:
                    e.die()
        # —— M3：Skeleton King 灵魂收集（场上任意部队死亡 +1，上限 10；能力召唤物除外——简化）——
        if isinstance(entity, Troop) and entity.card_name != 'SkeletonKingSkeleton' \
                and not getattr(entity, '_soul_excluded', False):
            # 勘误批11：克隆体/亡语衍生/召唤物不计魂（官方只算「最终形态」）
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
    # —— M8：精英卡（Hero 化）扩展：① 条件窗分支（Goblins 旗窗 / BarbLog 二次滚窗，
    #    本体已死仍可按按钮）；② Hero 形态分支 = 单次使用 + 圣水预检（先扣费再执行，
    #    未生效返还——官方「能力未开始施放即死亡/失败 → 返还圣水」语义）。
    #    冠军（Champion）路径保持原样不动（回归红线）。
    def use_ability(self, player_id):
        win = self.hero_windows.get(player_id)
        if win and not win.get('used'):
            if self.time > win['until']:
                win['used'] = True   # 窗口过期关闭（Goblins 旗消失 / BarbLog 按钮失效）
            else:
                from elite17_data import HERO_ABILITIES
                ab = HERO_ABILITIES[win['card']]
                cost = ab.get('manaCost', 0)
                if self.players[player_id].elixir < cost: return False
                self.players[player_id].elixir -= cost
                win['used'] = True
                win['effect'](self)  # 条件窗效果（出增援 / 二次滚），实现见 card_mechanics M8
                return True
        for e in list(self.entities.values()):
            if not e.is_alive or e.player != player_id: continue
            ability = getattr(e.data, 'ability', None)
            if not ability or e.ability_cd > 0: continue
            holder = e.entity_holder
            if not hasattr(holder, 'use_ability'): continue
            # —— M8：Hero 形态（单次使用 + 圣水预检）——
            if getattr(e, 'hero_mode', False):
                if e.ability_uses >= 1: continue
                cost = ability.get('manaCost', 0)
                if self.players[player_id].elixir < cost: return False
                self.players[player_id].elixir -= cost
                if not holder.use_ability():
                    self.players[player_id].elixir += cost  # 未生效返还（官方语义）
                    continue
                e.ability_uses += 1
                return True
            # —— 冠军原路径（不改动）——
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


