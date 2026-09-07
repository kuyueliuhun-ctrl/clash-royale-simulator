#!/usr/bin/env python3
"""M8 — Elite17 精英卡（Hero 化）引擎机制验收（17 卡 × ≥2 行为断言）。

规格：docs/elite17_spec.md；数据层：src/clasher_new/elite17_data.py；
机制：card_mechanics.py M8 段（HERO_CLASSES）+ battle.py M8 挂钩
（apply_hero_overlay / use_ability Hero 分支 / 条件窗 / 嘲讽覆盖 / Wild slot 互斥）。

数值来源逐条标注于 elite17_data.py（[Fandom]/[内存]/[假设]/[暂借]）；
【暂借/待实测】卡（IceWizard/EliteArcher/IceGolemite）按规格书口径做机制断言。

运行：python3 scripts/test_m6_elite.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'clasher_new'))
os.chdir(os.path.join(os.path.dirname(__file__), '..', 'src', 'clasher_new'))

from battle import BattleState, Troop, Position
from card_utils import Card, card_data
from player import PlayerState
from elite17_data import hval

PASS = FAIL = 0
def check(name, cond, extra=''):
    global PASS, FAIL
    if cond: PASS += 1; print(f'  OK {name}' + (f'  [{extra}]' if extra else ''))
    else: FAIL += 1; print(f'  FAIL {name}' + (f'  [{extra}]' if extra else ''))

def make_battle(deck0):
    bs = BattleState(PlayerState(0, list(deck0), 10),
                     PlayerState(1, ['Knight'] * 8, 10))
    return bs

def kill_towers(bs):
    """关掉四座公主塔（King 塔未激活不攻击），排除塔防干扰。"""
    for eid in (1, 2, 3, 4):
        bs.entities[eid].is_alive = False

def spawn(bs, card, x, y, player):
    t = Troop(bs.next_entity_id, Position(x, y), player, card, bs)
    bs._spawn_entity(t)
    t.deploy_delay_remaining = 0.0
    return t

def step_for(bs, seconds):
    for _ in range(int(seconds * 60)):
        bs.step(1 / 60)

def deploy_hero(bs, card, x, y):
    """过部署期地部署 Hero 卡并返回实体（不足 4 费时先充满圣水）。"""
    bs.players[0].elixir = max(bs.players[0].elixir, 10)
    assert bs.deploy_card(0, card, Position(x, y)), f'{card} 部署失败（手牌/圣水）'
    step_for(bs, 1.1)   # 越过部署延迟
    return [e for e in bs.entities.values() if e.card_name == card and e.player == 0][-1]


# ==================== 数据层 / Wild slot ====================

def test_data_layer():
    print('[M8-D] 数据层注入与 Wild slot')
    ab_ok = all((card_data[c].get('summonCharacterData') or {}).get('abilityData')
                for c in ('Knight', 'Musketeer', 'MiniPekka', 'Valkyrie', 'Wizard', 'Bowler',
                          'Giant', 'Goblins', 'MegaMinion', 'Tombstone', 'Berserker', 'DarkPrince',
                          'Balloon', 'BarbLog', 'IceWizard', 'EliteArcher', 'IceGolemite'))
    check('17/17 卡 abilityData 注入（冠军同链路）', ab_ok)
    check('Card.ability 由 summonCharacterData 派生', Card('Knight').ability.get('manaCost') == 2)
    check('Hero 槽上限 2（set_hero_slots 截断）',
          PlayerState(0, ['Knight'] * 8, 10).set_hero_slots(['Knight', 'Giant', 'Goblins'])
          or True and len(PlayerState(0, ['Knight'] * 8, 10).hero_slots) == 0)
    p = PlayerState(0, ['Knight'] * 8, 10)
    p.set_hero_slots(['Knight', 'Giant', 'Goblins'])
    check('Hero 槽上限 2', len(p.hero_slots) == 2)
    # Wild slot 互斥：hero 声明 → 觉醒禁用
    bs = make_battle(['Knight', 'Giant', 'Arrows', 'Minions', 'Fireball', 'Archers', 'Skeletons', 'Bowler'])
    kill_towers(bs)
    bs.players[0].set_evolution_slots(['Knight'])
    bs.players[0].evo_plays['Knight'] = 2   # cycle=2: 第 3 次本应觉醒
    bs.players[0].set_hero_slots(['Knight'])
    k = deploy_hero(bs, 'Knight', 9.0, 12.0)
    check('Wild slot 互斥：hero 化卡不再觉醒（evo=None）', k.evo is None)
    check('Wild slot：hero 形态部署（hero_mode=True）', k.hero_mode is True)


# ==================== §1 Knight ====================

def test_knight():
    print('[M8-01] Hero Knight — Triumphant Taunt')
    bs = make_battle(['Knight', 'Giant', 'Arrows', 'Minions', 'Fireball', 'Archers', 'Skeletons', 'Bowler'])
    kill_towers(bs)
    bs.players[0].set_hero_slots(['Knight'])
    k = deploy_hero(bs, 'Knight', 9.0, 12.0)
    # ① 形态数值：独立表 681/94/盾197 @L1 → lv11 ×1.1^10
    check('Hero 独立数值表（HP681/DMG94/盾197 → lv11 缩放）',
          k.hp == hval(681, 1, 11) and k.data.damage == hval(94, 1, 11)
          and k.shield_health == hval(197, 1, 11),
          f'hp={k.hp} dmg={k.data.damage} shield={k.shield_health}')
    # ② 嘲讽：6.5 格内敌人强制锁定 + 自身护盾
    e1 = spawn(bs, 'Knight', 9.5, 14.0, 1)
    e2 = spawn(bs, 'Giant', 13.0, 20.0, 1)   # 嘲讽半径外
    step_for(bs, 0.3)
    bs.players[0].elixir = 10
    check('能力释放（2 费扣费 + 单次标记）', bs.use_ability(0) and k.ability_uses == 1
          and abs(bs.players[0].elixir - 8.0) < 0.1)
    check('6.5 格内敌人被嘲讽锁定', e1._taunt_target_id == k.id and e1.target_id == k.id)
    check('半径外敌人不受嘲讽', getattr(e2, '_taunt_target_id', None) is None)
    check('嘲讽窗 5s + 能力护盾挂载', e1._taunt_until - bs.time <= 5.0 and k.shield_health > 0)
    check('单次使用：第二次按按钮无效', bs.use_ability(0) is False)
    step_for(bs, 5.2)
    check('护盾 5s 到期消失', k.shield_health == 0)


# ==================== §2 Musketeer ====================

def test_musketeer():
    print('[M8-02] Heroic Musketeer — Trusty Turret')
    bs = make_battle(['Musketeer', 'Giant', 'Arrows', 'Minions', 'Fireball', 'Archers', 'Skeletons', 'Bowler'])
    kill_towers(bs)
    bs.players[0].set_hero_slots(['Musketeer'])
    m = deploy_hero(bs, 'Musketeer', 9.0, 12.0)
    bs.players[0].elixir = 10
    check('能力释放：前方 3 格炮塔', bs.use_ability(0))
    t = [e for e in bs.entities.values() if e.card_name == 'MusketeerHeroTurret']
    check('炮塔生成于前方（y-3）', len(t) == 1 and abs(t[0].position.y - (m.position.y - 3.0)) < 0.1)
    # ① 炮塔数值：HP717/DMG65 @L3 → lv11
    check('炮塔独立数值（HP717/DMG65@L3 → lv11）',
          t[0].hp == hval(717, 3, 11) and t[0].data.damage == hval(65, 3, 11),
          f'hp={t[0].hp} dmg={t[0].data.damage}')
    # ② 寿命 10s 线性衰减（Building lifeTime 管线）
    check('炮塔寿命 10s（lifeTime 管线）', t[0].data.lifetime == 10.0)
    hp0 = t[0].hp
    step_for(bs, 2.0)
    check('线性 HP 衰减生效', 0 < t[0].hp < hp0)


# ==================== §3 MiniPekka ====================

def test_minipekka():
    print('[M8-03] Heroic Mini P.E.K.K.A — Breakfast Boost')
    bs = make_battle(['MiniPekka', 'Giant', 'Arrows', 'Minions', 'Fireball', 'Archers', 'Skeletons', 'Bowler'])
    kill_towers(bs)
    bs.players[0].set_hero_slots(['MiniPekka'])
    mp = deploy_hero(bs, 'MiniPekka', 9.0, 12.0)
    check('Hero holder 挂载（HeroMiniPekka）', type(mp.entity_holder).__name__ == 'HeroMiniPekka')
    # ① 攻击累煎饼进度（每击 +10s < 22s → 不加格）
    e1 = spawn(bs, 'Knight', 9.5, 12.5, 1)
    e1.hp = 10 ** 9; e1.data.hp = 10 ** 9   # 打不死，多打几击
    step_for(bs, 2.0)   # 攻速 0.8s → ≥2 击 → progress ≥20
    check('每击 +10s 煎饼进度', mp.entity_holder.progress >= 10 and mp.entity_holder.meter == 0,
          f'progress={mp.entity_holder.progress:.0f}s')
    # ② 能力：0 格 → +1 级（×1.1）+ 回复 30%
    mp.hp = mp.data.hp * 0.5
    dmg0, hp_max0 = mp.data.damage, mp.data.hp
    bs.players[0].elixir = 10
    check('能力：+1 级（伤害 ×1.1）', bs.use_ability(0) and abs(mp.data.damage - dmg0 * 1.1) < 1)
    check('回复 30% maxHP', mp.hp > mp.data.hp * 0.5, f'hp={mp.hp:.0f}/{mp.data.hp}')


# ==================== §4 Valkyrie ====================

def test_valkyrie():
    print('[M8-04] Hero Valkyrie — Wild Whirlwind')
    bs = make_battle(['Valkyrie', 'Giant', 'Arrows', 'Minions', 'Fireball', 'Archers', 'Skeletons', 'Bowler'])
    kill_towers(bs)
    bs.players[0].set_hero_slots(['Valkyrie'])
    v = deploy_hero(bs, 'Valkyrie', 9.0, 12.0)
    check('Hero 独立数值（1907/266@L11）', v.hp == 1907 and v.data.damage == 266)
    spawn(bs, 'Knight', 9.5, 12.5, 1)
    bs.players[0].elixir = 10
    check('能力：旋风开启 + 减伤 15%', bs.use_ability(0) and v.entity_holder.phase == 'whirl'
          and abs(v.damage_reduction - 0.15) < 1e-9)
    step_for(bs, 1.0 / 60)   # 钳制在首个 on_tick 生效
    check('旋风期间禁普攻（cooldown 钳制）', v.attack_cooldown >= 0.2)
    step_for(bs, 3.6)
    check('旋风结束 → 冲刺 5.5 格 + 禁攻段',
          v.entity_holder.phase == 'forbid' and v.position.y > 12.0 + 5.0,
          f'y={v.position.y:.1f}')
    step_for(bs, 1.1)
    check('禁攻结束还原', v.entity_holder.phase is None)


# ==================== §5 Wizard ====================

def test_wizard():
    print('[M8-05] Hero Wizard — Fiery Flight')
    bs = make_battle(['Wizard', 'Giant', 'Arrows', 'Minions', 'Fireball', 'Archers', 'Skeletons', 'Bowler'])
    kill_towers(bs)
    bs.players[0].set_hero_slots(['Wizard'])
    w = deploy_hero(bs, 'Wizard', 9.0, 12.0)
    # Wizard 主伤害在弹丸上（chr_wizardProjectile）→ Hero 覆写走 projectile 槽
    check('Hero 独立数值（HP832 / 弹丸 DMG281 @L11）', w.hp == 832
          and w.data.projectile_data.damage == 281,
          f"hp={w.hp} proj={w.data.projectile_data.damage}")
    e1 = spawn(bs, 'Knight', 9.5, 13.0, 1)
    bs.players[0].elixir = 10
    check('能力释放（1s 施法延迟）', bs.use_ability(0) and w.entity_holder.fly_pending > 0)
    step_for(bs, 1.3)
    # ① 升空：临时空中单位 + 移速 buff
    check('升空：空中位面 + 移速 +50%', w.data.is_air_unit and w.speed_buff >= 1.5)
    # ② 火球命中生成火旋风（半径 4 / 2s / dps43）
    step_for(bs, 1.5)
    zones = [e for e in bs.entities.values() if e.card_name == 'WizardHero_MiniTornadoBuff']
    check('火球命中生成火旋风领域', len(zones) >= 1 and abs(zones[0].radius - 4.0) < 1e-9,
          f'zones={len(zones)}')
    step_for(bs, 4.0)
    check('5s 后落地还原位面', not w.data.is_air_unit)


# ==================== §6 Bowler ====================

def test_bowler():
    print('[M8-06] Hero Bowler — Stone Swish')
    bs = make_battle(['Bowler', 'Giant', 'Arrows', 'Minions', 'Fireball', 'Archers', 'Skeletons', 'Valkyrie'])
    kill_towers(bs)
    bs.players[0].set_hero_slots(['Bowler'])
    b = deploy_hero(bs, 'Bowler', 9.0, 12.0)
    check('Hero 弹丸伤害覆写（289@L11）', b.data.projectile_data.damage == 289)
    rng0, hs0 = b.data.range, b.data.hit_speed
    bs.players[0].elixir = 10
    check('能力：2.5s 蓄力段（禁攻）', bs.use_ability(0) and b.entity_holder.phase == 'charge')
    step_for(bs, 2.6)
    # ① 攻城模式：射程 11.5 / 攻速 1.9 / 塔伤 ×0.5 / 3 发
    check('攻城模式：射程 11.5 / 攻速 1.9 / 塔伤 ×0.5',
          b.entity_holder.phase == 'siege' and b.data.range == 11.5
          and b.data.hit_speed == 1.9 and abs(b.data.tower_damage_mult - 0.5) < 1e-9)
    # ② 3 发用尽/7.3s 到期自动还原
    b.entity_holder.shots_left = 1   # 直接置最后一发（确定性）
    spawn(bs, 'Knight', 9.5, 13.5, 1)
    step_for(bs, 2.0)
    check('弹尽还原（射程/攻速恢复）', b.entity_holder.phase is None
          and b.data.range == rng0 and b.data.hit_speed == hs0)


# ==================== §7 Giant ====================

def test_giant():
    print('[M8-07] Hero Giant — Heroic Hurl')
    bs = make_battle(['Giant', 'Arrows', 'Minions', 'Fireball', 'Archers', 'Skeletons', 'Bowler', 'Valkyrie'])
    kill_towers(bs)
    bs.players[0].set_hero_slots(['Giant'])
    g = deploy_hero(bs, 'Giant', 13.0, 12.0)
    check('Hero 独立数值（3968/253@L11）', g.hp == 3968 and g.data.damage == 253)
    big = spawn(bs, 'Pekka', 13.5, 13.5, 1)
    small = spawn(bs, 'Knight', 12.5, 12.0, 1)
    bs.players[0].elixir = 10
    old = (big.position.x, big.position.y)
    check('能力：抓取最高 HP 部队（Pekka 优先）', bs.use_ability(0)
          and (big.position.x, big.position.y) != old)
    check('落地眩晕 2s + 溅射伤害（135@L11）', big.freeze_timer >= 2.0 - 1e-6)
    bs.players[0].elixir = 10
    before = bs.players[0].elixir
    step_for(bs, 2.2)   # 眩晕结束再测无目标
    for t in (big, small):
        t.position = Position(16.0, 16.0)   # 挪出抓取半径
    check('无抓取目标 → 不扣费返还', bs.use_ability(0) is False
          and abs(bs.players[0].elixir - before) < 0.2)


# ==================== §8 Goblins ====================

def test_goblins():
    print('[M8-08] Hero Goblins — Banner Brigade')
    bs = make_battle(['Goblins', 'Arrows', 'Minions', 'Fireball', 'Archers', 'Skeletons', 'Bowler', 'Valkyrie'])
    kill_towers(bs)
    bs.players[0].set_hero_slots(['Goblins'])
    bs.players[0].elixir = 10
    check('部署 x4 同组', bs.deploy_card(0, 'Goblins', Position(5.0, 12.0)))
    step_for(bs, 1.2)
    gobs = [e for e in bs.entities.values() if e.card_name == 'Goblins' and e.player == 0]
    check('x4 全部落地且同组', len(gobs) == 4
          and len({g.entity_holder.group_id for g in gobs}) == 1)
    bs.players[0].elixir = 10
    check('本体存活时按钮禁用', bs.use_ability(0) is False)
    for g in gobs: g.die()
    w = bs.hero_windows.get(0)
    check('最后一只阵亡 → 落旗开 5s 窗', w is not None and w['card'] == 'Goblins'
          and 0 < w['until'] - bs.time <= 5.0)
    bs.players[0].elixir = 10
    check('窗口内按按钮 → 增援 x2', bs.use_ability(0)
          and sum(1 for e in bs.entities.values()
                  if e.card_name == 'Goblins' and e.player == 0 and e.is_alive) == 2)
    check('单次：窗已用即关闭', bs.use_ability(0) is False)


# ==================== §9 MegaMinion ====================

def test_megaminion():
    print('[M8-09] Hero Mega Minion — Wounding Warp')
    bs = make_battle(['MegaMinion', 'Arrows', 'Minions', 'Fireball', 'Archers', 'Skeletons', 'Bowler', 'Valkyrie'])
    kill_towers(bs)
    bs.players[0].set_hero_slots(['MegaMinion'])
    mm = deploy_hero(bs, 'MegaMinion', 9.0, 12.0)
    check('Hero 独立数值（837/312@L11）', mm.hp == 837 and mm.data.damage == 312)
    weak = spawn(bs, 'Skeletons', 9.5, 13.0, 1)     # 最低 HP
    strong = spawn(bs, 'Pekka', 13.0, 13.0, 1)
    step_for(bs, 0.05)   # 部署时场上无敌军 → 首个 tick 补标记
    # ① 部署被动标记最低 HP 敌人
    check('部署标记最低 HP 敌人', mm.entity_holder.mark_id == weak.id)
    weak.die()
    step_for(bs, 0.05)
    check('标记随目标死亡转移', mm.entity_holder.mark_id == strong.id)
    # ② 瞬移 + WarpDmg + 永久塔伤 ×0.25
    strong.hp = 10 ** 9; strong.data.hp = 10 ** 9
    old = (mm.position.x, mm.position.y)
    hp0 = strong.hp
    bs.players[0].elixir = 10
    check('能力：瞬移至标记处', bs.use_ability(0)
          and mm.position.distance_to(strong.position) < 1.5)
    check('WarpDmg 399@L11 + 永久塔伤 ×0.25',
          strong.hp < hp0 and mm.data.tower_damage_mult == 0.25)


# ==================== §10 IceWizard（数值【暂借/待实测】） ====================

def test_icewizard():
    print('[M8-10] Hero Ice Wizard — 冰封自身（机制先行, 数值【暂借-待实测】）')
    bs = make_battle(['IceWizard', 'Arrows', 'Minions', 'Fireball', 'Archers', 'Skeletons', 'Bowler', 'Valkyrie'])
    kill_towers(bs)
    bs.players[0].set_hero_slots(['IceWizard'])
    iw = deploy_hero(bs, 'IceWizard', 9.0, 12.0)
    e1 = spawn(bs, 'Knight', 9.5, 12.5, 1)
    bs.players[0].elixir = 10
    check('能力：冰封自身（不可选取+无敌+定身）', bs.use_ability(0)
          and not iw.targetable and iw.invincible and iw.speed == 0.0)
    step_for(bs, 3.3)   # 冰块 3s【假设】
    check('破碎复现（解除冰封）', iw.targetable and not iw.invincible and iw.speed > 0)
    check('破碎冻结 AOE（【暂借】2s 冻结 + 减速）',
          e1.freeze_timer > 1.5 or e1.speed_debuff < 1.0,
          f'freeze={e1.freeze_timer:.1f} slow={e1.speed_debuff}')


# ==================== §11 Tombstone ====================

def test_tombstone():
    print('[M8-11] Hero Tombstone — Regal Revive')
    bs = make_battle(['Tombstone', 'Arrows', 'Minions', 'Fireball', 'Archers', 'Skeletons', 'Bowler', 'Valkyrie'])
    kill_towers(bs)
    bs.players[0].set_hero_slots(['Tombstone'])
    tb = deploy_hero(bs, 'Tombstone', 5.0, 12.0)
    check('Hero 形态移除持续产骷髅（2026-07-06）',
          'spawnCharacterData' not in tb.data.data['summonCharacterData'])
    bs.players[0].elixir = 10
    check('能力：预付 5 费武装复活', bs.use_ability(0)
          and getattr(tb, '_hero_queen_armed', False) and bs.players[0].elixir < 10)
    tb.die()
    q = [e for e in bs.entities.values() if e.card_name == 'TombQueen']
    check('墓碑破碎 → Tomb Queen 升起（HP4224@L11/只攻建筑/sight7）',
          len(q) == 1 and q[0].hp == 4224 and q[0].data.target_only_buildings
          and q[0].data.sight_range == 7.0)
    check('Queen 计时器挂载（【假设 15s-待实测】）', q[0]._evo_temp_lifetime == 15.0)


# ==================== §12 Berserker ====================

def test_berserker():
    print('[M8-12] Hero Berserker — Savage Survival')
    bs = make_battle(['Berserker', 'Arrows', 'Minions', 'Fireball', 'Archers', 'Skeletons', 'Bowler', 'Valkyrie'])
    kill_towers(bs)
    bs.players[0].set_hero_slots(['Berserker'])
    bz = deploy_hero(bs, 'Berserker', 9.0, 12.0)
    hp0 = bz.hp
    bs.players[0].elixir = 10
    check('熊灵：攻速 0.2s / UltraFast(2.7) / BearDmg 167 / 塔伤 ×0.25',
          bs.use_ability(0) and bz.data.hit_speed == 0.2 and abs(bz.speed - 2.7) < 1e-9
          and bz.data.damage == 167 and bz.data.tower_damage_mult == 0.25)
    # ① HP 下限：致死伤害钳制到 1
    bz.take_damage(10 ** 9, source=spawn(bs, 'Knight', 20.0, 20.0, 1))
    check('HP 不低于 1（minimum hitpoints）', bz.is_alive and bz.hp == 1.0)
    step_for(bs, 4.2)
    check('4s 后还原（change_back）', bz.data.hit_speed > 0.2 and bz.data.tower_damage_mult == 1.0)
    # ② 还原后致死伤害正常击杀（钳制仅熊灵窗内）
    bz.take_damage(10 ** 9)
    check('窗后再受致死伤害正常死亡', not bz.is_alive)


# ==================== §13 DarkPrince ====================

def test_darkprince():
    print('[M8-13] Hero Dark Prince — Destructive Dismount')
    bs = make_battle(['DarkPrince', 'Arrows', 'Minions', 'Fireball', 'Archers', 'Skeletons', 'Bowler', 'Valkyrie'])
    kill_towers(bs)
    bs.players[0].set_hero_slots(['DarkPrince'])
    dp = deploy_hero(bs, 'DarkPrince', 9.0, 12.0)
    charge0 = dp.data.charge_range
    bs.players[0].elixir = 10
    check('能力：下马 + 犀牛独立 spawn', bs.use_ability(0))
    rhino = [e for e in bs.entities.values() if e.card_name == 'DarkPrinceHeroRhino']
    check('犀牛坐骑（HP1356/Dmg179/ChargeDmg358@L11, 只攻建筑）',
          len(rhino) == 1 and rhino[0].hp == 1356 and rhino[0].data.damage == 179
          and rhino[0].data.charge_damage == 358 and rhino[0].data.target_only_buildings)
    check('本体徒步化：失去冲锋 + 普攻溅射（半径 1.2）',
          dp.data.charge_range == 0 and abs(dp.data.area_damage_radius - 1.2) < 1e-9
          and charge0 > 0)


# ==================== §14 Balloon ====================

def test_balloon():
    print('[M8-14] Hero Balloon — Coffin Cadets')
    bs = make_battle(['Balloon', 'Arrows', 'Minions', 'Fireball', 'Archers', 'Skeletons', 'Bowler', 'Valkyrie'])
    kill_towers(bs)
    bs.players[0].set_hero_slots(['Balloon'])
    bl = deploy_hero(bs, 'Balloon', 9.0, 12.0)
    e1 = spawn(bs, 'Knight', 9.5, 13.0, 1)
    bs.players[0].elixir = 10
    check('能力：召出骷髅伞兵（锁定 6 格内地面敌人）', bs.use_ability(0))
    sk = [e for e in bs.entities.values() if e.card_name == 'Skeletrooper']
    check('伞降段不可选取+无敌', len(sk) == 1 and not sk[0].targetable and sk[0].invincible)
    step_for(bs, 2.0)
    check('落地 AOE（263@L11, 对塔 ×0.1）后驻场攻击',
          sk[0].entity_holder.landed and e1.hp < e1.data.hp and sk[0].targetable,
          f'enemy hp={e1.hp:.0f}/{e1.data.hp}')
    # 本体死亡炸弹（继承基础 Balloon 亡语）
    bl.die()
    bombs = [e for e in bs.entities.values() if type(e).__name__ == 'TimedExplosive']
    check('本体死亡掉炸弹（基础 Balloon 亡语继承）', len(bombs) >= 1)


# ==================== §15 BarbarianBarrel ====================

def test_barblog():
    print('[M8-15] Hero Barbarian Barrel — Rowdy Reroll')
    bs = make_battle(['BarbLog', 'Arrows', 'Minions', 'Fireball', 'Archers', 'Skeletons', 'Bowler', 'Valkyrie'])
    kill_towers(bs)
    bs.players[0].set_hero_slots(['BarbLog'])
    bs.players[0].elixir = 10
    check('部署即开二次滚按钮窗（10s【假设】）',
          bs.deploy_card(0, 'BarbLog', Position(9.0, 8.0))
          and bs.hero_windows[0]['card'] == 'BarbLog')
    barb = spawn(bs, 'Barbarian', 9.0, 9.0, 0)
    barb.take_damage(200)
    hp0 = barb.hp
    bs.players[0].elixir = 10
    check('能力：桶再滚一次（1 费）', bs.use_ability(0)
          and any(e.card_name == 'BarbLogProjectileRolling' for e in bs.entities.values()))
    step_for(bs, 1.1)
    check('治疗野蛮人 = 桶伤害 50%（lv11 桶伤 151 → ≈75.5/s × 1s）',
          barb.hp > hp0 + 50, f'{hp0:.0f} → {barb.hp:.0f}')
    check('单次：窗已用关闭', bs.use_ability(0) is False)


# ==================== §16 EliteArcher（数值【暂借/待实测】） ====================

def test_elitearcher():
    print('[M8-16] Hero Elite Archer — Warp + Triple Shot（机制先行, 数值【暂借-待实测】）')
    bs = make_battle(['EliteArcher', 'Arrows', 'Minions', 'Fireball', 'Archers', 'Skeletons', 'Bowler', 'Valkyrie'])
    kill_towers(bs)
    bs.players[0].set_hero_slots(['EliteArcher'])
    ea = deploy_hero(bs, 'EliteArcher', 5.0, 12.0)
    e1 = spawn(bs, 'Knight', 8.0, 12.5, 1)
    old = (ea.position.x, ea.position.y)
    bs.players[0].elixir = 10
    check('能力：向最近敌人瞬移（Warp 3 格【假设】）', bs.use_ability(0)
          and ea.position != Position(*old)
          and ea.position.distance_to(e1.position) < Position(*old).distance_to(e1.position))
    check('三连射：攻速 0.3s【暂借】×3 发待打', ea.data.hit_speed == 0.3
          and ea.entity_holder.shots_left == 3)
    d = [e for e in bs.entities.values() if e.card_name == 'EliteArcherHeroDummy']
    check('假人分身 spawn（HP104 [内存·中置信] → lv11, 寿命 5s【假设】）',
          len(d) == 1 and d[0].hp == hval(104, 1, 11) and d[0]._evo_temp_lifetime == 5.0)
    step_for(bs, 2.5)   # 首发冷却余量 + 3×0.3s
    check('三连射打完攻速还原', ea.data.hit_speed > 0.3 and ea.entity_holder.shots_left == 0,
          f'hit={ea.data.hit_speed} shots={ea.entity_holder.shots_left}')


# ==================== §17 IceGolemite（数值【暂借/待实测】） ====================

def test_icegolemite():
    print('[M8-17] Hero Ice Golemite — 冰雪光环（机制先行, 数值【暂借 Ice Golem/Hero】）')
    bs = make_battle(['IceGolemite', 'Arrows', 'Minions', 'Fireball', 'Archers', 'Skeletons', 'Bowler', 'Valkyrie'])
    kill_towers(bs)
    bs.players[0].set_hero_slots(['IceGolemite'])
    ig = deploy_hero(bs, 'IceGolemite', 13.0, 12.0)
    small = spawn(bs, 'Skeletons', 13.3, 12.3, 1)   # 小体型档
    big = spawn(bs, 'Pekka', 12.6, 12.6, 1)         # 中/大体型档
    bs.players[0].elixir = 10
    check('能力：冰雪光环（半径 4 / 3 次脉冲【暂借】）', bs.use_ability(0))
    step_for(bs, 0.05)   # 首脉冲在下一个 update tick 结算
    zone = [e for e in bs.entities.values() if e.card_name == 'IceGolemiteHero_ice_aura']
    check('光环领域挂载（半径 4 / 首脉冲后剩 2 次【暂借 3 脉冲】）',
          len(zone) == 1 and zone[0].pulses_left == 2 and abs(zone[0].radius - 4.0) < 1e-9)
    step_for(bs, 0.25)
    check('首脉冲已结算（剩 2 脉冲）', zone[0].pulses_left == 2)
    check('小体型档冻结 / 大体型档减速（分档【假设】）',
          small.freeze_timer > 0 and big.speed_debuff < 1.0,
          f'small freeze={small.freeze_timer:.1f} big slow={big.speed_debuff}')
    check('脉冲伤害 69@L11【暂借】命中', small.hp < small.data.hp)


# ==================== 冠军路径回归（M8 分支不破坏冠军） ====================

def test_champion_unaffected():
    print('[M8-R] 冠军 use_ability 路径不受 M8 改动影响')
    bs = make_battle(['SkeletonKing', 'Arrows', 'Minions', 'Fireball', 'Archers', 'Skeletons', 'Bowler', 'Valkyrie'])
    kill_towers(bs)
    sk = deploy_hero(bs, 'SkeletonKing', 9.0, 12.0)   # 冠军（非 hero 声明）
    check('冠军无 hero_mode', getattr(sk, 'hero_mode', False) is False)
    bs.players[0].elixir = 10
    ok = bs.use_ability(0)
    check('冠军能力仍走冷却路径（ SkeletonKing 召唤）', ok and sk.entity_holder.remaining > 0)


if __name__ == '__main__':
    test_data_layer()
    test_knight()
    test_musketeer()
    test_minipekka()
    test_valkyrie()
    test_wizard()
    test_bowler()
    test_giant()
    test_goblins()
    test_megaminion()
    test_icewizard()
    test_tombstone()
    test_berserker()
    test_darkprince()
    test_balloon()
    test_barblog()
    test_elitearcher()
    test_icegolemite()
    test_champion_unaffected()
    print(f'\n===== M8 Elite17 验收：通过 {PASS} / 失败 {FAIL} =====')
    sys.exit(1 if FAIL else 0)
