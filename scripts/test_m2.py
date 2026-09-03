#!/usr/bin/env python3
"""M2/M3/M4 机制族验收：族3 最小射程 / 族4 buff 槽 / 族5 拉拽 / 族6 英雄能力 / 族7 觉醒。
运行：cd src/clasher_new && python3 ../../scripts/test_m2.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'clasher_new'))
os.chdir(os.path.join(os.path.dirname(__file__), '..', 'src', 'clasher_new'))

import battle
from battle import BattleState, Troop, Building, Position
from player import PlayerState
from evolutions import EVOLUTION_CYCLES

PASS = FAIL = 0
def check(name, cond, extra=''):
    global PASS, FAIL
    if cond: PASS += 1; print(f'  OK {name}' + (f'  [{extra}]' if extra else ''))
    else: FAIL += 1; print(f'  FAIL {name}' + (f'  [{extra}]' if extra else ''))

def make_battle(elixir=10):
    d0 = ['Knight','Archers','Fireball','Giant','Musketeer','Arrows','Minions','Cannon']
    d1 = ['Knight','MiniPekka','Arrows','Minions','Musketeer','Fireball','Giant','Archer']
    bs = BattleState(PlayerState(0, d0, elixir), PlayerState(1, d1, elixir))
    bs.players[0].elixir = elixir
    bs.players[1].elixir = elixir
    return bs

def give(bs, pid, card, elixir=10):
    """把卡置顶入手并给满圣水（绕开 cycle[:4] 限制）"""
    p = bs.players[pid]
    p.cycle = [card] + [c for c in p.cycle if c != card]
    p.elixir = elixir

def kill_red_towers(bs):
    """击杀红方双公主塔（直接置死，不触发 on_death），用于需要敌方半场测试位的用例"""
    for eid in (1, 2):
        bs.entities[eid].is_alive = False

def step_for(bs, seconds):
    for _ in range(int(seconds * 60)):
        bs.step(1 / 60)

def spawn_troop(bs, card, x, y, player, evolved=False):
    t = Troop(bs.next_entity_id, Position(x, y), player, card, bs, evolved=evolved)
    bs._spawn_entity(t)
    return t

# ---------- 族3：建筑最小射程 ----------
def test_min_range():
    print('[族3] Mortar 最小射程')
    b = make_battle()
    mortar = Building(b.next_entity_id, Position(9.0, 13.5), 1, 'Mortar', False)  # 站位避开蓝塔射程（否则测试前就被塔点死）
    b._spawn_entity(mortar)
    step_for(b, 9.5)  # 部署 3.5s + 攻击冷却 5.0s + 首发弹道飞行
    close = spawn_troop(b, 'Knight', 9.0, 14.8, 0)   # 距离 1.3 < 3.5 贴脸
    step_for(b, 2.0)
    check('贴脸单位不受攻击（白嫖）', close.hp == close.data.hp, f'hp={close.hp:.0f}')
    mortar.attack_cooldown = 0.0  # 跳过迫击炮上一发冷却（它此前在轰蓝塔——真实行为）
    far = spawn_troop(b, 'Giant', 9.0, 8.5, 0)       # 距离 5.0 在射程内（1.45s 后才跨过最小射程线）
    hp0 = far.hp
    step_for(b, 1.3)  # 弹道飞行 1.0s 后命中（先于跨线）
    check('射程内目标正常受击', far.hp < hp0, f'hp {hp0:.0f}→{far.hp:.0f}')
    check('最小射程值 3.5', mortar.data.min_range == 3.5, f'{mortar.data.min_range}')

# ---------- 族4：buff 槽 ----------
def test_zap_stun():
    print('[族4] Zap 眩晕+伤害+重置')
    b = make_battle()
    k = spawn_troop(b, 'Knight', 9.0, 14.5, 1)
    k.attack_cooldown = 0  # 假设正要攻击
    give(b, 0, 'Zap')
    ok = b.deploy_card(0, 'Zap', Position(9.0, 14.0))
    check('Zap 部署成功（此前为无效隐形实体）', ok)
    step_for(b, 0.2)
    check('眩晕生效（0.5s 冻结）', k.freeze_timer > 0, f'freeze={k.freeze_timer:.2f}')
    check('Zap lv11 伤害 192', abs(k.data.hp - k.hp - 192) < 2, f'伤害={k.data.hp-k.hp:.0f}')
    check('攻击蓄力被重置', k.attack_cooldown >= k.data.hit_speed - 0.01, f'cd={k.attack_cooldown:.2f}')
    step_for(b, 0.6)
    check('眩晕到期恢复', k.freeze_timer <= 0)

def test_freeze():
    print('[族4] Freeze 冰冻 4s')
    b = make_battle()
    k = spawn_troop(b, 'Giant', 9.0, 14.5, 1)
    give(b, 0, 'Freeze')
    ok = b.deploy_card(0, 'Freeze', Position(9.0, 14.0))
    check('Freeze 部署成功', ok)
    step_for(b, 0.3)
    check('冰冻生效（4s）', k.freeze_timer > 3.5, f'freeze={k.freeze_timer:.2f}')
    y0 = k.position.y
    step_for(b, 1.5)
    check('冰冻期间不移动', abs(k.position.y - y0) < 0.01, f'位移={abs(k.position.y-y0):.3f}')
    step_for(b, 3.0)
    check('冰冻总时长 4s（不因重复脉冲刷新到 8s）', k.freeze_timer <= 0, f'freeze={k.freeze_timer:.2f}')

def test_heal():
    print('[族4] Heal 治疗')
    b = make_battle()
    k = spawn_troop(b, 'Knight', 9.0, 14.0, 0)
    k.hp = 1000.0
    give(b, 0, 'Heal')
    b.deploy_card(0, 'Heal', Position(9.0, 14.0))
    step_for(b, 2.2)
    healed = k.hp - 1000
    check('治疗量 ≈ 4×96=384（旧版 Heal lv11，低置信度标注）', 300 <= healed <= 400, f'healed={healed:.0f}')

def test_rage():
    print('[族4] Rage 官方现行 +30%/4.5s')
    b = make_battle()
    k = spawn_troop(b, 'Knight', 9.0, 14.0, 0)
    give(b, 0, 'Rage')
    b.deploy_card(0, 'Rage', Position(9.0, 14.0))
    step_for(b, 0.3)
    check('狂暴加速 1.30（官方 2025/10 现行值，非快照 1.35）', abs(k.speed_buff - 1.30) < 0.01, f'buff={k.speed_buff}')

def test_tornado_pull():
    print('[族5] Tornado 拉拽')
    b = make_battle()
    k1 = spawn_troop(b, 'Knight', 7.5, 14.5, 1)
    k2 = spawn_troop(b, 'Knight', 10.5, 14.5, 1)
    give(b, 0, 'Tornado')
    b.deploy_card(0, 'Tornado', Position(9.0, 14.5))
    step_for(b, 1.3)
    d1 = k1.position.distance_to(Position(9.0, 14.5))
    d2 = k2.position.distance_to(Position(9.0, 14.5))
    check('两侧单位被拉向中心', d1 < 2.0 and d2 < 2.0, f'd1={d1:.2f} d2={d2:.2f}')
    dmg = k1.data.hp - k1.hp
    check('Tornado lv11 跳伤 84×2=168', abs(dmg - 168) <= 2, f'dmg={dmg:.0f}')

def test_building_decay():
    print('[族4] 建筑 persistent 错位修复（Cannon 寿命衰减）')
    b = make_battle()
    give(b, 0, 'Cannon')
    ok = b.deploy_card(0, 'Cannon', Position(9.0, 12.0))
    check('Cannon 部署成功', ok)
    cannon = [e for e in b.entities.values() if e.name == 'Cannon' and e.is_alive][0]
    hp_start = cannon.hp
    step_for(b, 10.0)
    check('Cannon 10s 后按寿命自衰减', cannon.hp < hp_start, f'hp {hp_start:.0f}→{cannon.hp:.0f}')

def test_fisherman_hook():
    print('[族5] 渔夫钩拉')
    b = make_battle()
    kill_red_towers(b)
    f = spawn_troop(b, 'Fisherman', 9.0, 18.0, 0)
    k = spawn_troop(b, 'Giant', 9.0, 23.5, 1)  # 5.5 格，在 3.5~7 钩程内
    step_for(b, 4.0)  # 部署 1s + 蓄力 1.3s + 抛钩飞行
    d = k.position.distance_to(Position(9.0, 18.0))
    check('目标被拉近渔夫', d < 2.5, f'dist={d:.2f}')
    check('钩子伤害（lv11 194，唯一输出源）', abs(k.data.hp - k.hp - 194) < 3, f'dmg={k.data.hp-k.hp:.0f}')

# ---------- 族6：英雄能力 ----------
def test_skeleton_king():
    print('[族6] Skeleton King 灵魂召唤')
    b = make_battle()
    kill_red_towers(b)  # 防止骷髅走进塔程被射死干扰计数（真实行为）
    sk = spawn_troop(b, 'SkeletonKing', 9.0, 14.0, 0)
    step_for(b, 1.2)  # 部署 1.0s
    b.souls[0] = 4
    ok = b.use_ability(0)
    check('能力释放成功', ok)
    check('圣水扣费 2', abs(b.players[0].elixir - (10 - 2)) < 0.1, f'elixir={b.players[0].elixir:.2f}')
    step_for(b, 5.0)  # 前摇 0.9 + 10 只 × 0.25s
    n = sum(1 for e in b.entities.values() if e.name == 'SkeletonKingSkeleton' and e.is_alive)
    check('召唤骷髅数 = 6+灵魂(4) = 10', n == 10, f'n={n}')
    check('进入冷却（20s 计时中）', 14.0 <= sk.ability_cd <= 20.0, f'cd={sk.ability_cd:.1f}')

def test_archer_queen():
    print('[族6] Archer Queen 隐身斗篷')
    b = make_battle()
    q = spawn_troop(b, 'ArcherQueen', 9.0, 14.0, 0)
    step_for(b, 1.2)
    b.use_ability(0)
    check('不可被选取', not q.targetable)
    check('攻速倍率 ×2.8', abs(q.hit_speed_mult - 2.8) < 0.01, f'{q.hit_speed_mult}')
    check('移速 -25%', abs(q.speed_debuff - 0.75) < 0.01, f'{q.speed_debuff}')
    step_for(b, 4.0)
    check('隐身到期恢复可选取', q.targetable)
    check('攻速恢复', abs(q.hit_speed_mult - 1.0) < 0.01)

def test_golden_knight_dash():
    print('[族6] Golden Knight 连环突进')
    b = make_battle()
    gk = spawn_troop(b, 'GoldenKnight', 9.0, 14.0, 0)
    t1 = spawn_troop(b, 'Knight', 9.0, 15.8, 1)
    t2 = spawn_troop(b, 'Knight', 9.0, 17.5, 1)
    step_for(b, 1.2)  # 部署 1.0s
    hp1, hp2 = t1.hp, t2.hp
    b.use_ability(0)
    step_for(b, 0.5)
    check('突进造成 ≈340 伤害（131×1.1^10，wiki 335）', t1.hp < hp1 - 300 or t2.hp < hp2 - 300,
          f't1 {hp1-t1.hp:.0f} t2 {hp2-t2.hp:.0f}')
    check('位置位移（瞬移链）', gk.position.distance_to(Position(9.0, 14.0)) > 1.0)

def test_monk_reflect():
    print('[族6] Monk 禅定反弹')
    b = make_battle()
    m = spawn_troop(b, 'Monk', 9.0, 14.0, 0)
    archer = spawn_troop(b, 'Archer', 9.0, 17.0, 1)
    step_for(b, 1.2)
    b.use_ability(0)
    check('减伤 65%（官方）', abs(m.damage_reduction - 0.65) < 0.01, f'{m.damage_reduction}')
    step_for(b, 1.5)  # 弓箭手射几箭
    check('弓箭手被反弹伤害', archer.hp < archer.data.hp, f'hp={archer.hp:.0f}/{archer.data.hp}')
    check('Monk 存活且伤害经减伤', m.is_alive and m.hp > m.data.hp * 0.5, f'hp={m.hp:.0f}/{m.data.hp}')

def test_little_prince_guard():
    print('[族6] Little Prince 皇家救援')
    b = make_battle()
    lp = spawn_troop(b, 'LittlePrince', 9.0, 14.0, 0)
    step_for(b, 1.2)
    b.use_ability(0)
    step_for(b, 0.5)
    guards = [e for e in b.entities.values() if e.name == 'ChampionGuard' and e.is_alive]
    check('守护者 Guardienne 入场', len(guards) == 1)
    check('守护者 lv11 ≈1621 血', abs(guards[0].hp - 625 * 1.1**10) < 3, f'hp={guards[0].hp:.0f}')

def test_mighty_miner():
    print('[族6] Mighty Miner 爆破脱身')
    b = make_battle()
    mm = spawn_troop(b, 'MightyMiner', 6.0, 14.0, 0)
    step_for(b, 1.2)  # 部署 1.0s
    x0 = mm.position.x
    b.use_ability(0)
    step_for(b, 0.3)
    check('镜像换路瞬移', abs(mm.position.x - (18 - x0)) < 0.6, f'x {x0}→{mm.position.x:.1f}')
    check('钻地期间不可选取', not mm.targetable)
    step_for(b, 1.2)
    check('钻地结束恢复', mm.targetable and not mm.invincible)

def test_boss_bandit():
    print('[族6] Boss Bandit 金蝉脱壳（限 2 次）')
    b = make_battle()
    bb = spawn_troop(b, 'BossBandit', 9.0, 14.0, 0)
    step_for(b, 1.2)
    y0 = bb.position.y
    b.use_ability(0)
    check('向后传送 6 格', abs(bb.position.y - (y0 - 6)) < 0.1, f'y {y0}→{bb.position.y:.1f}')
    bb.ability_cd = 0
    b.use_ability(0)
    check('第二次可用（共 2 次）', bb.ability_uses == 2)
    bb.ability_cd = 0
    ok = b.use_ability(0)
    check('第三次被拒（限 2 次）', not ok)

# ---------- 族7：觉醒 ----------
def test_evo_cycle():
    print('[族7] 觉醒周期（Knight cycle=2：第 3 次觉醒）')
    b = make_battle()
    b.players[0].set_evolution_slots(['Knight'])
    for i in range(2):
        give(b, 0, 'Knight')
        b.deploy_card(0, 'Knight', Position(3.0, 10.0))
    normal = [e for e in b.entities.values() if e.name == 'Knight' and e.is_alive and e.evo is None]
    check('前两次普通形态', len(normal) == 2, f'{len(normal)}')
    give(b, 0, 'Knight')
    b.deploy_card(0, 'Knight', Position(3.0, 10.0))
    evos = [e for e in b.entities.values() if e.name == 'Knight' and e.is_alive and e.evo is not None]
    check('第三次觉醒形态', len(evos) == 1, f'{len(evos)}')
    if evos:
        check('觉醒 hp 按曲线推导（=基础 1766）', abs(evos[0].hp - 1766) < 5, f'hp={evos[0].hp:.0f}')
    # 未携带觉醒位的同卡永不觉醒
    b2 = make_battle()
    for i in range(3):
        give(b2, 0, 'Knight')
        b2.deploy_card(0, 'Knight', Position(3.0, 10.0))
    n2 = [e for e in b2.entities.values() if e.name == 'Knight' and e.is_alive and e.evo is not None]
    check('未携带觉醒位不觉醒', len(n2) == 0)

def test_evo_knight_fortify():
    print('[族7] 觉醒骑士 fortify（脱战减伤 60%）')
    b = make_battle()
    k = spawn_troop(b, 'Knight', 9.0, 14.0, 0, evolved=True)
    step_for(b, 1.8)  # 部署 1.0s + 攻击冷却回满（hitSpeed 1.1）
    check('fortify 生效（脱战时）', k._fortify_dr > 0.5, f'dr={getattr(k,"_fortify_dr",0)}')
    k.last_attack_time = b.time  # 模拟刚攻击
    step_for(b, 0.05)
    check('攻击中 fortify 解除', getattr(k, '_fortify_dr', 0) == 0.0, f'dr={getattr(k,"_fortify_dr",0)}')

def test_evo_skeletons_duplication():
    print('[族7] 觉醒骷髅分裂（每击 +1，组上限 8）')
    b = make_battle()
    s = spawn_troop(b, 'Skeletons', 9.0, 14.0, 0, evolved=True)
    dummy = Building(b.next_entity_id, Position(9.0, 14.8), 1, 'Elixir Collector', False)
    b._spawn_entity(dummy)  # 无攻击建筑假人：位置固定、不还手（圣水收集器）
    dummy.data.lifetime = float('inf')  # 关闭寿命衰减
    step_for(b, 8.0)
    group = sum(1 for e in b.entities.values() if e.card_name == 'Skeletons' and e.is_alive)
    check('组内数量向 8 增长', group > 3, f'group={group}')

def test_evo_wizard_shield():
    print('[族7] 觉醒法师护盾')
    b = make_battle()
    w = spawn_troop(b, 'Wizard', 9.0, 14.0, 0, evolved=True)
    check('觉醒护盾 75×1.1^10 ≈194', abs(w.shield_health - 75 * 1.1**10) < 2, f'shield={w.shield_health:.0f}')

def test_evo_archer_double_shot():
    print('[族7] 觉醒弓箭手双发射击')
    b = make_battle()
    a = spawn_troop(b, 'Archer', 9.0, 14.0, 0, evolved=True)
    t = spawn_troop(b, 'Giant', 9.0, 15.8, 1)  # 1.8 格 < 射程 5
    step_for(b, 3.0)
    dmg = t.data.hp - t.hp
    plain_arrow = 106 + 218  # 主箭 + 二段箭（近似）
    check('双发伤害 ≈ 主箭+二段箭', dmg > 200, f'dmg={dmg:.0f}')

def test_evo_royal_giant_push():
    print('[族7] 觉醒皇家巨人推击（对建筑）')
    b = make_battle()
    rg = spawn_troop(b, 'RoyalGiant', 9.0, 14.0, 0, evolved=True)
    cannon = Building(b.next_entity_id, Position(9.0, 15.8), 1, 'Cannon', False)
    b._spawn_entity(cannon)
    hp0 = cannon.hp
    step_for(b, 4.0)
    dmg = hp0 - cannon.hp
    check('RG 对建筑造成伤害（攻击+推击 AoE）', dmg > 300, f'dmg={dmg:.0f}')

def test_evo_all_construct():
    print('[族7] 全 34 张快照觉醒卡 evolved 形态可构造')
    from card_utils import Card
    ok_n = 0
    for name in EVOLUTION_CYCLES:
        if name == 'AngryBarbarians': continue  # 2026 快照外
        try:
            c = Card(name)
            if not c.evo_raw: continue
            if c.type == 'building':
                t = Building(9999, Position(9, 9), 0, name, False, evolved=True)
            else:
                t = Troop(9999, Position(9, 9), 0, name, None, evolved=True)
            ok_n += 1
        except Exception as e:
            check(f'{name} evolved 构造', False, f'{type(e).__name__}: {str(e)[:60]}')
    check(f'觉醒形态构造 {ok_n}/34', ok_n == 34, f'{ok_n}')

# ---------- M4.5 动作链：攻击序列（attackSequenceList） ----------
def test_attack_seq_inferno_evo():
    """觉醒地狱龙：四级递增 14/47/165/330（lv11≈36/121/424/849），Manual 逐攻击推进、封顶末档、脱锁重置。"""
    print('[M4.5] 觉醒地狱龙攻击序列（InfernoDragon_EV1）')
    b = make_battle()
    t = spawn_troop(b, 'InfernoDragon', 9.0, 14.0, 0, evolved=True)
    stages = [round(x) for x in t.attack_seq_damages]
    check('解析四级序列 14/47/165/330 → lv11', stages == [36, 121, 424, 849], f'{stages}')
    dmg0 = round(t.ramped_damage())
    t._on_attack_done(); dmg1 = round(t.ramped_damage())
    t._on_attack_done(); dmg2 = round(t.ramped_damage())
    t._on_attack_done(); dmg3 = round(t.ramped_damage())
    t._on_attack_done(); dmg4 = round(t.ramped_damage())
    check('逐攻击递增 36→121→424→849', (dmg0, dmg1, dmg2, dmg3) == (36, 121, 424, 849), f'{dmg0},{dmg1},{dmg2},{dmg3}')
    check('封顶末档 849（不越界）', dmg4 == 849, f'{dmg4}')
    # 脱锁重置：target_id 置空并走 Entity.update 的序列重置逻辑 → 回到首档
    t.target_id = None
    battle.Entity.update(t, 1 / 60)
    check('脱锁重置到首档', t.attack_seq_stage == 0, f'stage={t.attack_seq_stage}')

def test_attack_seq_inferno_battle():
    """集成：战斗中觉醒地狱龙对同一目标的单次伤害随攻击推进（末档≈849 单发）。"""
    print('[M4.5] 觉醒地狱龙战斗中递增（对巨人对拍）')
    b = make_battle()
    t = spawn_troop(b, 'InfernoDragon', 9.0, 14.0, 0, evolved=True)
    g = spawn_troop(b, 'Giant', 9.0, 13.2, 1)
    # 逐 tick 推进，记录巨人所受每次攻击的伤害序列
    hits = []
    prev = g.hp
    for _ in range(int(9.0 * 60)):
        b.step(1 / 60)
        if g.hp < prev - 1:
            hits.append(round(prev - g.hp))
        prev = g.hp
    # 首击需等部署1s+蓄力1.2s+延迟伤害1tick；9s 足够打出多次
    check('有多次攻击记录', len(hits) >= 4, f'{len(hits)} hits {hits}')
    check('末段单发 ≈849（最高档）', max(hits[-3:]) >= 800, f'tail={hits[-3:]}')
    check('首击 ≈36（首档）', abs(hits[0] - 36) <= 6, f'first={hits[0]}')

def test_attack_seq_berserker():
    """Berserker：攻击序列三连击 40×3（lv1 基准，lv11 按 1.1 曲线 ×2.594 → 104×3），单次攻击周期内打满。"""
    print('[M4.5] Berserker 三连击（40×3 → lv11 104×3）')
    b = make_battle()
    ber = spawn_troop(b, 'Berserker', 9.0, 14.0, 0)
    check('序列解析为三段且按曲线缩放', [round(x) for x in ber.attack_seq_damages] == [104, 104, 104],
          f'{ber.attack_seq_damages}')
    ber._on_attack_done()
    check('攻击后排队 2 段剩余', ber.attack_seq_pending == 2, f'pending={ber.attack_seq_pending}')
    # 集成：直接驱动一次攻击（Berserker 索敌数据缺失 STATS_UNRESOLVED，不走 get_nearest_target）
    # 剩余两段应在 hit_speed/3 ≈0.17s 内依次结算，总伤 = 3×104=312
    b2 = make_battle()
    ber2 = spawn_troop(b2, 'Berserker', 9.0, 14.0, 0)
    k = spawn_troop(b2, 'Knight', 9.0, 13.0, 1)
    ber2.deploy_delay_remaining = 0.0
    k.deploy_delay_remaining = 0.0
    ber2.entity_holder.on_attack(k)   # 第一段 104（delayed）+ 排队 2 段
    for _ in range(int(1.0 * 60)):
        b2.step(1 / 60)
    dmg = 1766 - k.hp
    check('一次攻击打满 312（3×104）', abs(dmg - 312) <= 8, f'dmg={dmg:.0f}')

# ---------- 16 级数据支持（set_level 全卡无越界 + 数值轴正确） ----------
def test_level16_support():
    print('[数据] 16 级支持：全卡 set_level(16) + 稀有度轴修正')
    from card_utils import Card, card_data
    errs = []
    for name in sorted(card_data):
        try:
            Card(name).set_level(16)
        except Exception as e:
            errs.append(f'{name}: {type(e).__name__} {str(e)[:40]}')
    check('全卡 set_level(16) 无异常', not errs, '; '.join(errs[:3]))
    # 主卡数值：Knight lv16 hp=2822（Common 数组 index15）
    k = Card('Knight'); k.set_level(16)
    check('Knight lv16 hp=2822', k.hp == 2822, f'hp={k.hp}')
    # 稀有度轴修正：SkeletonArmy 召唤的 Skeleton（Common 行）lv16 与 Skeletons 一致
    sa = Card('SkeletonArmy'); sa.set_level(16)
    sk = Card('Skeletons'); sk.set_level(16)
    check('SkeletonArmy 单位与 Skeletons 同数值轴', sa.hp == sk.hp, f'sa={sa.hp} sk={sk.hp}')
    # VoodooHog（行稀有度 Legendary）：lv16 不越界且有值
    vh = Card('VoodooHog'); vh.set_level(16)
    check('VoodooHog lv16 可用', vh.hp > 0, f'hp={vh.hp}')
    # 派生曲线卡（9 张缺失表）16 级可用
    bp = Card('LittlePrince'); bp.set_level(16)
    check('LittlePrince（derived_curve）lv16 可用', bp.hp > 0, f'hp={bp.hp}')

def test_battle_level_range():
    """11-16 全等级战斗贯通：BattleState(card_level) → 实体/法术/觉醒全部按该等级。"""
    print('[数据] 11-16 战斗等级贯通')
    from card_utils import Card, _value_at_level, characters as _chars
    from battle import AreaEffect
    from card_utils import spells as _spells
    zap_dpl = (_spells.get('Zap') or {}).get('damage_per_level') or []
    prev_evo = 0
    ok_all = True
    for lv in range(11, 17):
        b = make_battle()
        b.card_level = lv
        battle.Card.default_level = lv
        k = spawn_troop(b, 'Knight', 9.0, 14.0, 0)
        exp_hp = _value_at_level(_chars['Knight']['hitpoints_per_level'], 'Common', lv, 690)
        ae = AreaEffect(b.next_entity_id, Position(9, 14), 0, 'Zap'); ae.battle_state = b
        exp_zap = _value_at_level(zap_dpl, 'Common', lv, 0)
        t = spawn_troop(b, 'InfernoDragon', 9.0, 15.0, 0, evolved=True)
        ok = (k.hp == exp_hp and k.level == lv
              and abs(ae.damage_per_tick - exp_zap) < 1
              and t.hp > prev_evo)  # 觉醒 hp 随级严格递增
        prev_evo = t.hp
        if not ok:
            ok_all = False
            print(f'    lv{lv} 不匹配: knight={k.hp}(exp {exp_hp}) zap={ae.damage_per_tick:.0f}(exp {exp_zap:.0f}) evo_hp={t.hp:.0f}')
    check('11-16 每级实体/法术/觉醒数值贯通', ok_all, f'evo hp 区间 [{battle.Card.default_level}]')


def test_witchmother_curse():
    print('[M4.5] 女巫妈妈诅咒（VoodooCurse → VoodooHog）')
    b = make_battle()
    wm = spawn_troop(b, 'WitchMother', 9.0, 14.0, 0)
    k = spawn_troop(b, 'Knight', 9.0, 12.0, 1)
    wm.deploy_delay_remaining = 0.0
    k.deploy_delay_remaining = 0.0
    voodoo = None
    for _ in range(int(20.0 * 60)):
        b.step(1 / 60)
        vh = [e for e in b.entities.values() if e.card_name == 'VoodooHog' and e.is_alive]
        if vh:
            voodoo = vh[0]
            break
    check('被诅咒骑士死亡生成 VoodooHog', voodoo is not None, '')
    if voodoo:
        check('VoodooHog 归属施法者（player 0）', voodoo.player == 0, f'player={voodoo.player}')
        check('VoodooHog 有血量', voodoo.hp > 0, f'hp={voodoo.hp:.0f}')

if __name__ == '__main__':
    for t in (test_min_range, test_zap_stun, test_freeze, test_heal, test_rage,
              test_tornado_pull, test_building_decay, test_fisherman_hook,
              test_skeleton_king, test_archer_queen, test_golden_knight_dash,
              test_monk_reflect, test_little_prince_guard, test_mighty_miner,
              test_boss_bandit, test_evo_cycle, test_evo_knight_fortify,
              test_evo_skeletons_duplication, test_evo_wizard_shield,
              test_evo_archer_double_shot, test_evo_royal_giant_push,
              test_evo_all_construct, test_attack_seq_inferno_evo,
              test_attack_seq_inferno_battle, test_attack_seq_berserker,
              test_witchmother_curse, test_level16_support, test_battle_level_range):
        try:
            t()
        except Exception as e:
            import traceback; traceback.print_exc()
            check(t.__name__ + ' 异常', False, str(e)[:80])
    print(f'\n通过 {PASS} / 失败 {FAIL}')
    sys.exit(1 if FAIL else 0)
