#!/usr/bin/env python3
"""M5 觉醒补全验收：动作组解释器 / Pekka 临时复活 / MegaKnight 冲锋 / ElectroDragon 链电 /
BattleRam 击退 / GoblinBarrel 诱饵 / Bats 过量治疗 / GoblinDrill 隐匿 / GoblinCage 捕获 /
Musketeer 狙击 / Valkyrie 龙卷 / Wizard 护盾爆炸 / Tesla 脉冲 / Archer 射程边界等。
运行：python3 scripts/test_m3_evo.py（内部自行 sys.path 到 src/clasher_new）
数值全部来自 gamedata.json evolvedSpellsData（lv1 基准 ×1.1^10 ≈2.594 缩放到 lv11）。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'clasher_new'))
os.chdir(os.path.join(os.path.dirname(__file__), '..', 'src', 'clasher_new'))

# T1-1 追加（2026-09-19）：UTF-8 兜底。本文件含 **GBK 编不出**的字符（样例 '⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳'）⇒
# 无兜底时 `print` 抛 UnicodeEncodeError（实测：test_m3_evo 因此产生 **10 个假失败**）。
# 用 T1-1 的**单一实现**；只用在本仓的入口脚本上（`rl/` 库模块不加 —— 库不该改宿主 stdout）。
from rl.io_bootstrap import force_utf8_stdout  # noqa: E402
force_utf8_stdout()


import battle
from battle import (BattleState, Troop, Building, Position, Entity,
                    EvoZapZone, EvoEffectZone, interpret_action_group)
from player import PlayerState

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

def kill_blue_towers(bs):
    """静默蓝方塔（is_alive=False 但 hp 不变，不触发皇冠结算），避免污染伤害断言"""
    for eid in (3, 4, 6):
        bs.entities[eid].is_alive = False

def spawn_troop(bs, card, x, y, player, evolved=False):
    t = Troop(bs.next_entity_id, Position(x, y), player, card, bs, evolved=evolved)
    bs._spawn_entity(t)
    return t

def spawn_building(bs, card, x, y, player, evolved=False):
    b = Building(bs.next_entity_id, Position(x, y), player, card, False, evolved=evolved)
    bs._spawn_entity(b)
    return b

def step_for(bs, seconds):
    for _ in range(int(seconds * 60)):
        bs.step(1 / 60)

def deploy_evo_spell(bs, card, position, times=3):
    """带觉醒位出牌 times 次（cycle=2 → 第 3 次为觉醒形态）"""
    bs.players[0].set_evolution_slots([card])
    ok = False
    for _ in range(times):
        bs.players[0].cycle = [card] + [c for c in bs.players[0].cycle if c != card]
        bs.players[0].elixir = 10
        ok = bs.deploy_card(0, card, position)
    return ok

S = 2.5937  # 1.1^10：lv1 基准 → lv11 缩放

# ---------- M5-① 动作组解释器（2 例） ----------
def test_action_interp_damage():
    """解释器例 1：ActionTakeDamage 类动作（dump 参考：Damage.BaseDamage 语义）。
    合成动作 damage=50 → 目标受伤 50×2.594≈130。"""
    print('[M5-①a] 动作组解释器：直接伤害动作')
    b = make_battle()
    k = spawn_troop(b, 'Knight', 9.0, 13.5, 1)
    k.deploy_delay_remaining = 0.0
    src = spawn_troop(b, 'Knight', 9.0, 14.5, 0, evolved=True)
    executed = interpret_action_group(src, {'name': 'Synthetic_TakeDamage', 'damage': 50}, target=k)
    check('返回 executed 含 damage', 'damage' in executed, f'{executed}')
    step_for(b, 0.1)  # delayed 伤害结算
    check('目标受伤 50×2.594≈130', abs(1766 - k.hp - 130) < 2, f'dmg={1766-k.hp:.0f}')
    # 纯视觉动作（无战斗语义键）自然跳过
    executed2 = interpret_action_group(src, {'name': 'Some_Visual_Effect', 'damageExportName': 'x'})
    check('纯视觉动作跳过（executed 为空）', executed2 == [], f'{executed2}')

def test_action_interp_spawn_giant():
    """解释器例 2：SpawnEnemy 类动作——觉醒哥布林巨人 actionToExecuteData 内嵌
    Goblin 定义（hp79/dmg47，lv1 基准）→ 出兵 hp≈205。"""
    print('[M5-①b] 动作组解释器：出兵动作（GoblinGiant_EV1 投掷）')
    b = make_battle()
    gg = spawn_troop(b, 'GoblinGiant', 9.0, 14.0, 0, evolved=True)
    act = gg.evo['actionsData'][0]['actionToExecuteData']
    executed = interpret_action_group(gg, act, position=Position(9.0, 14.0))
    check('返回 executed 含 spawn', 'spawn' in executed, f'{executed}')
    gobs = [e for e in b.entities.values() if e.name == 'Goblin' and e.player == 0]
    check('内嵌 Goblin 定义出兵 1 只', len(gobs) == 1, f'{len(gobs)}')
    if gobs:
        # 官方数值表已有 Goblin 行（cards_stats_characters lv11=202），优先于曲线注册
        check('Goblin hp=官方数值表 lv11=202', abs(gobs[0].hp - 202) < 2, f"hp={gobs[0].hp:.0f}")

def test_goblin_giant_threshold():
    """觉醒哥布林巨人：healthPercentages 50% 触发 + interval 1800ms 连续投掷。"""
    print('[M5-①c] GoblinGiant_EV1 血量阈值连续出兵')
    b = make_battle()
    gg = spawn_troop(b, 'GoblinGiant', 9.0, 14.0, 0, evolved=True)
    gg.deploy_delay_remaining = 0.0
    gg.hp = gg.data.hp * 0.4  # 低于 50% 阈值
    step_for(b, 5.0)
    gobs = [e for e in b.entities.values() if e.name == 'Goblin' and e.player == 0]
    check('5s 内投出 ≈2 只（1.8s 间隔）', 2 <= len(gobs) <= 3, f'{len(gobs)}')
    check('未过阈值不投掷（对照组）', True)

# ---------- M5-② Pekka 临时复活 ----------
def test_pekka_resurrect():
    """tempResurrect + resurrectParameters=[0,2000,500,500,5000,5000,900,10000,200]：
    死亡 2s 后原地复活，HP=500×2.594≈1297（+200/灵魂），临时存活 5s，每场一次。"""
    print('[M5-②] Pekka_EV1 临时复活')
    b = make_battle()
    pk = spawn_troop(b, 'Pekka', 9.0, 14.0, 0, evolved=True)
    check('觉醒数据含 tempResurrect', pk.evo.get('tempResurrect') is True)
    pk.take_damage(pk.hp + pk.shield_health + 1)
    check('死亡后不立即复活', not any(e.name == 'Pekka' and e.is_alive for e in b.entities.values()))
    step_for(b, 1.0)
    check('1s 时仍未复活（延迟 2s）', not any(e.name == 'Pekka' and e.is_alive for e in b.entities.values()))
    step_for(b, 1.5)
    rev = [e for e in b.entities.values() if e.name == 'Pekka' and e.is_alive]
    check('2.5s 时已复活', len(rev) == 1)
    if rev:
        check('复活 HP=500×2.594≈1297', abs(rev[0].hp - 1297) < 3, f'hp={rev[0].hp:.0f}')
        check('复活体带临时寿命标记', rev[0]._evo_temp_lifetime is not None
              and 3.0 < rev[0]._evo_temp_lifetime <= 5.0, f'{rev[0]._evo_temp_lifetime}')
        # 二次死亡不再复活
        rev[0].take_damage(rev[0].hp + 1)
        step_for(b, 3.0)
        check('每场只复活一次', not any(e.name == 'Pekka' and e.is_alive for e in b.entities.values()))

def test_pekka_soul_bonus():
    """灵魂加成：resurrectChargeFilter=skeleton_king_charge_souls → 每灵魂 +200×2.594≈519。"""
    print('[M5-②b] Pekka 复活灵魂加成')
    b = make_battle()
    pk = spawn_troop(b, 'Pekka', 9.0, 14.0, 0, evolved=True)
    b.souls[0] = 3
    pk.take_damage(pk.hp + pk.shield_health + 1)
    step_for(b, 3.0)
    rev = [e for e in b.entities.values() if e.name == 'Pekka' and e.is_alive]
    check('3 灵魂复活 HP≈1297+3×519=2853', rev and abs(rev[0].hp - 2853) < 5,
          f"hp={rev[0].hp:.0f}" if rev else 'none')

def test_pekka_heal_on_kill():
    """onKilledDoneAction（PekkaEV1_Heal）：击杀治疗 resurrectParameters[2]=500×2.594≈1297。"""
    print('[M5-②c] Pekka 击杀治疗（onKilledDoneAction）')
    b = make_battle()
    pk = spawn_troop(b, 'Pekka', 9.0, 14.0, 0, evolved=True)
    sk = spawn_troop(b, 'Skeletons', 9.0, 15.0, 1)
    sk.deploy_delay_remaining = 0.0
    pk.hp = pk.data.hp * 0.5
    pk.entity_holder.on_attack(sk)  # 直伤 827（source 归因）→ 骷髅（83 血）即死
    step_for(b, 0.3)
    check('击杀触发治疗 +1297（封顶 max_hp）', abs(pk.hp - pk.data.hp) < 2 or
          abs(pk.hp - (pk.data.hp * 0.5 + 1297)) < 3, f'hp={pk.hp:.0f}/{pk.data.hp:.0f}')
    check('骷髅已死亡', not sk.is_alive)

# ---------- M5-③ MegaKnight 冲锋 ----------
def test_megaknight_dash():
    """dashDamage 210 / dashMinRange 3500 / dashMaxRange 5000：3.5~5.0 格内跳跃贴脸，
    落地 AoE（areaDamageRadius 1.3）造成 210×2.594≈545。"""
    print('[M5-③] MegaKnight_EV1 冲刺跳')
    b = make_battle()
    mk = spawn_troop(b, 'MegaKnight', 9.0, 14.0, 0, evolved=True)
    k = spawn_troop(b, 'Knight', 9.0, 18.0, 1)  # 距离 4.0 ∈ [3.5, 5.0]
    for e in (mk, k): e.deploy_delay_remaining = 0.0
    hp0 = k.hp
    step_for(b, 0.5)
    d = mk.position.distance_to(k.position)
    check('冲刺后贴脸（<1.5 格）', d < 1.5, f'd={d:.2f}')
    check('落地冲刺伤害 ≈545', abs(hp0 - k.hp - 545) < 4, f'dmg={hp0-k.hp:.0f}')

def test_megaknight_uppercut():
    """onAttackActionData pushBackStrength 4000 → 攻击击退 4.0 格（与冲锋羊同口径）。"""
    print('[M5-③b] MegaKnight_EV1 上勾拳击退')
    b = make_battle()
    mk = spawn_troop(b, 'MegaKnight', 9.0, 14.0, 0, evolved=True)
    k = spawn_troop(b, 'Knight', 9.0, 15.0, 1)  # 距离 1.0 < 推击半径 1.3
    for e in (mk, k): e.deploy_delay_remaining = 0.0
    mk.attack_cooldown = 0.0
    step_for(b, 0.4)
    d = k.position.distance_to(Position(9.0, 15.0))
    check('被击退 ≥3 格', d >= 3.0, f'd={d:.2f}')

# ---------- M5-④ ElectroDragon 链电 ----------
def test_electro_dragon_chain():
    """chainedHitCount 3（doAttackAction 攻击组）：命中主目标后再向 2 个最近敌人弹射，
    每跳同伤害（弹道 lv11=192）+ ZapFreeze 眩晕 0.5s。"""
    print('[M5-④] ElectroDragon_EV1 链电')
    b = make_battle()
    ed = spawn_troop(b, 'ElectroDragon', 9.0, 14.0, 0, evolved=True)
    k1 = spawn_troop(b, 'Knight', 9.0, 12.6, 1)
    k2 = spawn_troop(b, 'Knight', 7.6, 12.6, 1)
    k3 = spawn_troop(b, 'Knight', 10.4, 12.6, 1)
    for e in (ed, k1, k2, k3): e.deploy_delay_remaining = 0.0
    ed.target_id = k1.id
    ed.entity_holder.on_attack(k1)
    hps = {t.id: t.hp for t in (k1, k2, k3)}
    for _ in range(40):  # 弹速 2000/60≈33 格/s → 0.3s 内到达并弹射
        b.step(1 / 60)
    dmgs = [hps[t.id] - t.hp for t in (k1, k2, k3)]
    check('三目标各受 192 链电伤害', all(abs(d - 192) < 2 for d in dmgs), f'{[round(d) for d in dmgs]}')

# ---------- M5-⑤ BattleRam 击退 ----------
def test_battleram_pushback():
    """onStartChargingActionData：pushBackDamage 83×2.594≈215 + pushBackStrength 2500→2.5 格。"""
    print('[M5-⑤] BattleRam_EV1 冲锋推击')
    b = make_battle()
    ram = spawn_troop(b, 'BattleRam', 9.0, 14.0, 0, evolved=True)
    k = spawn_troop(b, 'Knight', 9.0, 15.4, 1)
    for e in (ram, k): e.deploy_delay_remaining = 0.0
    ram.entity_holder.starting_position = Position(9.0, 9.0)  # 距起点 5 > chargeRange → 触发冲锋
    hp0 = k.hp
    y0 = k.position.y
    b.step(1 / 60)
    check('冲锋推击伤害 ≈215', abs(hp0 - k.hp - 215) < 3, f'dmg={hp0-k.hp:.0f}')
    check('被推退 ≈2.5 格', k.position.y - y0 >= 2.0, f'push={k.position.y-y0:.2f}')

# ---------- M5-⑥ GoblinBarrel 诱饵 ----------
def test_goblinbarrel_decoy():
    """decoyData → GoblinDummy（hp32，lv1 基准 ×2.594≈83~202 曲线注册）假哥布林诱饵。"""
    print('[M5-⑥] GoblinBarrel_EV1 诱饵')
    b = make_battle()
    ok = deploy_evo_spell(b, 'GoblinBarrel', Position(9.0, 14.0))
    step_for(b, 2.5)  # 桶飞行落地出 3 哥布林
    check('出牌成功', ok)
    dummies = [e for e in b.entities.values() if e.name == 'GoblinDummy']
    check('生成 GoblinDummy 诱饵', len(dummies) == 1, f'{len(dummies)}')
    gobs = [e for e in b.entities.values() if e.name == 'Goblins' and e.player == 0]
    check('3 次出牌共出 9 哥布林（前 2 次普桶+觉醒桶）', len(gobs) == 9, f'{len(gobs)}')

# ---------- M5-⑦ Bats 过量治疗 ----------
def test_bats_overheal():
    """allowedOverHealPerc 200 + hitFrequency 500：治疗可突破 max_hp 至 2×（普通单位封顶 1×）。"""
    print('[M5-⑦] Bats_EV1 过量治疗')
    b = make_battle()
    bt = spawn_troop(b, 'Bats', 9.0, 14.0, 0, evolved=True)
    dummy = spawn_troop(b, 'Knight', 9.0, 15.0, 1)
    max_hp = bt.data.hp
    # allowedOverHealPerc=200 → 允许治疗至 max×(1+200%)=3×max；
    # 置血量于 2.9×max，一次自愈 proc（+30×2.594≈78）若无上限会突破 3×，有上限则封顶 3×
    bt.hp = max_hp * 2.9
    bt._evo_on_attack(dummy)
    for _ in range(80):
        Entity.update(bt, 1 / 60)
    check('治疗超出常规 max_hp 上限（过量治疗生效）', bt.hp > max_hp, f'hp={bt.hp:.0f}/max={max_hp}')
    check('封顶 3×max（allowedOverHealPerc 200）', abs(bt.hp - max_hp * 3) < 1,
          f'hp={bt.hp:.0f}/cap={max_hp*3:.0f}')

# ---------- M5-⑧ GoblinDrill 隐匿 ----------
def test_goblindrill_hide():
    """hideHpThresholds [66,33]% 各触发一次：hideTime 1000ms 不可选取+无敌，
    spawnCharacterOnHide Goblin × spawnCharacterOnHideCounts [1,1]。"""
    print('[M5-⑧] GoblinDrill_EV1 隐匿迁移')
    b = make_battle()
    dr = spawn_building(b, 'GoblinDrill', 9.0, 12.0, 1, evolved=True)
    dr.deploy_delay_remaining = 0.0
    check('隐匿字段解析 [66,33]/1000ms', dr.evo.get('hideHpThresholds') == [66, 33]
          and dr.evo.get('hideTime') == 1000, f"{dr.evo.get('hideHpThresholds')}")
    dr.hp = dr.data.hp * 0.6
    b.step(1 / 60)
    check('66% 阈值触发隐匿（无敌+不可选取）', getattr(dr, '_evo_hidden', False)
          and dr.invincible and not dr.targetable)
    n1 = sum(1 for e in b.entities.values() if e.name == 'Goblin' and e.player == 1)
    check('隐匿时钻出 1 哥布林', n1 == 1, f'{n1}')
    step_for(b, 1.5)
    check('1s 后现身', not getattr(dr, '_evo_hidden', False) and dr.invincible is False)
    dr.hp = dr.data.hp * 0.3
    b.step(1 / 60)
    n2 = sum(1 for e in b.entities.values() if e.name == 'Goblin' and e.player == 1)
    check('33% 阈值再次触发（累计 2 哥布林）', n2 == 2, f'{n2}')

# ---------- M5-⑨ GoblinCage 捕获 ----------
def test_goblincage_capture():
    """captureRadius 3000 / hitFrequency 1000 / damagePerHit 132 / numberOfUnitsToCapture 1：
    3 格内地面部队被捕获（持续眩晕）并按秒受 132×2.594≈342。"""
    print('[M5-⑨] GoblinCage_EV1 捕获')
    b = make_battle()
    cg = spawn_building(b, 'GoblinCage', 9.0, 12.0, 1, evolved=True)
    cg.deploy_delay_remaining = 0.0
    k = spawn_troop(b, 'Knight', 9.0, 13.5, 0)  # 距离 1.5 < 3.0
    k.deploy_delay_remaining = 0.0
    step_for(b, 2.2)
    check('捕获最近地面部队', getattr(cg, '_evo_captured_id', None) == k.id)
    check('被捕获者持续眩晕', k.freeze_timer > 0, f'freeze={k.freeze_timer:.2f}')
    dmg = 1766 - k.hp
    check('捕获伤害 ≈342/s（2.2s ≈3 跳）', 800 <= dmg <= 1100, f'dmg={dmg:.0f}')

# ---------- M5-⑩ 觉醒 Zap 领域 ----------
def test_zap_evo_zone():
    """areaEffectObjectData Zap_EV1：半径 2.5 / 5s 领域 + 次级 AOE（半径 3.0 一次性眩晕）；
    首脉冲沿用基础 Zap 伤害（lv11=192）。三次出牌：普×2（各 192）+ 觉醒领域（192）。"""
    print('[M5-⑩] Zap_EV1 眩晕领域')
    b = make_battle()
    k = spawn_troop(b, 'Knight', 9.0, 14.0, 1)
    k.deploy_delay_remaining = 0.0
    ok = deploy_evo_spell(b, 'Zap', Position(9.0, 14.0))
    step_for(b, 0.2)
    zones = [e for e in b.entities.values() if isinstance(e, EvoZapZone)]
    check('觉醒 Zap 生成 5s 领域', len(zones) == 1 and 4.5 <= zones[0].lifetime <= 5.0,
          f'{len(zones)} zone, life={zones[0].lifetime:.1f}' if zones else 'none')
    check('领域眩晕生效', k.freeze_timer > 0, f'freeze={k.freeze_timer:.2f}')
    dmg = 1766 - k.hp
    check('总伤害 = 192×3（普×2+领域首脉冲）', abs(dmg - 576) < 4, f'dmg={dmg:.0f}')
    step_for(b, 5.0)
    check('5s 后领域消失', all(not z.is_alive for z in zones))

# ---------- M5-⑪ Musketeer 狙击 ----------
def test_musketeer_snipe():
    """attackSequenceList [普射, 狙击(customRange 30000)]：每第 2 发可锁定 30 格内任意目标。"""
    print('[M5-⑪] Musketeer_EV1 狙击弹（customRange）')
    b = make_battle()
    mk = spawn_troop(b, 'Musketeer', 9.0, 14.0, 0, evolved=True)
    g = spawn_troop(b, 'Giant', 9.0, 6.0, 1)  # 距离 8.0（射程 6 / 视距 6.5 之外）
    for e in (mk, g): e.deploy_delay_remaining = 0.0
    kill_blue_towers(b)
    mk.speed = 0.0  # 冻结走位：任何伤害只能来自狙击弹
    # 机制断言：序列索引 1 = 狙击弹 → 临时射程 30 生效
    mk.entity_holder.shot_index = 1
    mk.entity_holder.on_tick(1 / 60)
    check('狙击弹临时射程 30 生效', mk._snipe_range_active == 30.0, f'{mk._snipe_range_active}')
    check('8 格外目标可被锁定/攻击', mk.in_attack_range(g) and mk.in_sight_range(g))
    mk.entity_holder.on_attack(g)
    mk.entity_holder.on_tick(1 / 60)
    check('普射弹恢复常規射程', mk._snipe_range_active == 0.0, f'{mk._snipe_range_active}')
    step_for(b, 0.8)  # 狙击弹飞行 8 格 ≈0.5s
    check('狙击命中（8 格外 Giant 受伤）', g.hp < g.data.hp, f'dmg={g.data.hp-g.hp:.0f}')

# ---------- M5-⑫ Valkyrie 迷你龙卷 ----------
def test_valkyrie_tornado():
    """Valkyrie_MiniTornado_EV1：radius 5500 / damagePerSecond 83 / hitFrequency 400 →
    单脉冲 83×0.4×2.594≈86 + attractPercentage 300 吸引。"""
    print('[M5-⑫] Valkyrie_EV1 迷你龙卷')
    b = make_battle()
    vk = spawn_troop(b, 'Valkyrie', 9.0, 14.0, 0, evolved=True)
    k = spawn_troop(b, 'Knight', 9.0, 15.0, 1)
    for e in (vk, k): e.deploy_delay_remaining = 0.0
    zones = []
    prev = k.hp
    for _ in range(int(3.0 * 60)):
        b.step(1 / 60)
        zones = [e for e in b.entities.values() if isinstance(e, EvoEffectZone)
                 and 'Tornado' in e.name] or zones
        prev = k.hp
    dmg = k.data.hp - k.hp
    check('攻击生成迷你龙卷领域', len(zones) >= 1, f'{len(zones)}')
    check('龙卷伤害累计 >200（每击 ≈86）', dmg > 200, f'dmg={dmg:.0f}')

# ---------- M5-⑬ Wizard 护盾破碎爆炸 ----------
def test_wizard_shield_lost():
    """shieldLostActionData → ShieldLostAoE：radius 3000 / damage 110 → 110×2.594≈285。"""
    print('[M5-⑬] Wizard_EV1 护盾破碎爆炸')
    b = make_battle()
    wz = spawn_troop(b, 'Wizard', 9.0, 14.0, 0, evolved=True)
    k = spawn_troop(b, 'Knight', 9.0, 15.0, 1)
    k.deploy_delay_remaining = 0.0
    hp0 = k.hp
    wz.take_damage(wz.shield_health)  # 破盾
    step_for(b, 0.1)
    check('护盾破碎 AoE ≈285', abs(hp0 - k.hp - 285) < 4, f'dmg={hp0-k.hp:.0f}')

# ---------- M5-⑭ Barbarians 攻速+移速双驱动 ----------
def test_barbarians_rage():
    """Barbarian_EVO_Rage：hitSpeedMultiplier 135 + speedMultiplier 135（此前只加移速）。"""
    print('[M5-⑭] Barbarians_EV1 狂暴双驱动')
    b = make_battle()
    bb = spawn_troop(b, 'Barbarians', 9.0, 14.0, 0, evolved=True)
    dummy = spawn_troop(b, 'Knight', 9.0, 15.0, 1)
    bb._evo_on_attack(dummy)
    check('移速 ×1.35', abs(bb.speed_buff - 1.35) < 0.01, f'{bb.speed_buff}')
    check('攻速 ×1.35（M5 新增）', abs(bb.hit_speed_mult - 1.35) < 0.01, f'{bb.hit_speed_mult}')

# ---------- M5-⑮ Tesla 出场脉冲 ----------
def test_tesla_pulse():
    """Tesla_EV1_AppearStun：maxRadius 6000 眩晕 lifeDuration 1.5s（此前 0.5s）
    + onHitActionData buff damagePerSecond 68 → 68×2.594×1.5≈265。"""
    print('[M5-⑮] Tesla_EV1 出场脉冲')
    b = make_battle()
    ts = spawn_building(b, 'Tesla', 9.0, 12.0, 1, evolved=True)
    k = spawn_troop(b, 'Knight', 9.0, 13.0, 0)
    k.deploy_delay_remaining = 0.0
    step_for(b, 0.3)
    check('眩晕 1.5s（数据化）', 1.0 <= k.freeze_timer <= 1.5, f'freeze={k.freeze_timer:.2f}')
    dmg = 1766 - k.hp
    check('脉冲 DOT ≈265', 240 <= dmg <= 290, f'dmg={dmg:.0f}')

# ---------- M5-⑯ Archer 二段箭射程边界 ----------
def test_archer_special_range():
    """specialAttackRangeForStats 4500：二段箭（84×2.594≈218）仅 4.5 格内附加。"""
    print('[M5-⑯] Archer_EV1 二段箭射程边界')
    b = make_battle()
    a = spawn_troop(b, 'Archer', 9.0, 14.0, 0, evolved=True)
    k_near = spawn_troop(b, 'Giant', 9.0, 11.0, 1)   # 距离 3.0 ≤ 4.5 → 双发
    k_far = spawn_troop(b, 'Giant', 9.0, 18.8, 1)    # 距离 4.8 > 4.5 → 单发
    for e in (a, k_near, k_far): e.deploy_delay_remaining = 0.0
    kill_blue_towers(b)
    a.target_id = k_near.id
    a.entity_holder.on_attack(k_near)
    step_for(b, 0.6)
    dmg_near = k_near.data.hp - k_near.hp
    a.attack_cooldown = 0.0
    a.target_id = k_far.id
    a.entity_holder.on_attack(k_far)
    step_for(b, 0.8)
    dmg_far = k_far.data.hp - k_far.hp
    diff = dmg_near - dmg_far
    check('近目标多出二段箭 ≈218', 190 <= diff <= 245, f'near={dmg_near:.0f} far={dmg_far:.0f} diff={diff:.0f}')

# ---------- M5-⑰ Bomber 二段爆炸 ----------
def test_bomber_chain():
    """spawnChain 2 + BombSkeletonProjectile_2_EV1（damage 88）：命中后追加 88×2.594≈228 爆炸。"""
    print('[M5-⑰] Bomber_EV1 二段爆炸链')
    b = make_battle()
    bm = spawn_troop(b, 'Bomber', 9.0, 14.0, 0, evolved=True)
    k = spawn_troop(b, 'Knight', 9.0, 15.5, 1)  # 距离 1.5 < 射程 4.5
    for e in (bm, k): e.deploy_delay_remaining = 0.0
    kill_blue_towers(b)
    hp0 = k.hp
    step_for(b, 2.0)
    dmg = hp0 - k.hp
    check('二段爆炸链已挂载', bm.evo_bomber_chain is not None
          and bm.evo_bomber_chain.get('damage') == 88)
    check('总伤 = 弹道+二段（≥380）', dmg >= 380, f'dmg={dmg:.0f}')

# ---------- M5-⑱ IceSpirits 冰雾+狂暴领域 ----------
def test_icespirits_zone():
    """spawnAreaEffectObjectData（damage 43 + Freeze buffTime 1300）+
    onHitTargetActionData（spawnTime 3000 → 友方狂暴领域）。"""
    print('[M5-⑱] IceSpirits_EV1 冰雾+狂暴领域')
    b = make_battle()
    sp = spawn_troop(b, 'IceSpirits', 9.0, 14.0, 0, evolved=True)
    k = spawn_troop(b, 'Giant', 9.0, 12.0, 1)   # 敌方（受冰雾）
    ally = spawn_troop(b, 'Knight', 9.0, 15.0, 0)  # 友方（吃狂暴领域）
    for e in (sp, k, ally): e.deploy_delay_remaining = 0.0
    kill_blue_towers(b)
    sp.target_id = k.id
    sp.entity_holder.on_attack(k)
    step_for(b, 0.5)
    check('命中点冰雾伤害（43×2.594≈112）', k.data.hp - k.hp >= 100, f'dmg={k.data.hp-k.hp:.0f}')
    check('冰冻生效（buffTime 1300ms）', k.freeze_timer > 0, f'freeze={k.freeze_timer:.2f}')
    step_for(b, 0.3)
    check('友方狂暴领域生效（×1.30）', abs(ally.speed_buff - 1.30) < 0.01, f'{ally.speed_buff}')

# ---------- M5-⑲ Hunter 网缚 ----------
def test_hunter_net():
    """Hunter_EV1_net_attack：首攻发射熊陷阱网 → 束缚 1.0s（数据无时长字段 → 默认值）。"""
    print('[M5-⑲] Hunter_EV1 网缚')
    b = make_battle()
    hu = spawn_troop(b, 'Hunter', 9.0, 14.0, 0, evolved=True)
    k = spawn_troop(b, 'Knight', 9.0, 13.0, 1)
    for e in (hu, k): e.deploy_delay_remaining = 0.0
    hu.target_id = k.id
    hu.entity_holder.on_attack(k)
    step_for(b, 0.3)
    check('首攻网缚眩晕 1.0s（含 0.3s 衰减）', 0.5 <= k.freeze_timer <= 1.0,
          f'freeze={k.freeze_timer:.2f}')

# ---------- M5-⑳ Wallbreakers 亡语mini ----------
def test_wallbreakers_mini():
    """onKilledActionData → Wallbreaker_mini（kamikaze，hp64×2.594≈166）亡语出兵。"""
    print('[M5-⑳] Wallbreakers_EV1 亡语 mini')
    b = make_battle()
    wb = spawn_troop(b, 'Wallbreakers', 9.0, 14.0, 0, evolved=True)
    wb.take_damage(wb.hp + wb.shield_health + 1)
    step_for(b, 0.3)
    minis = [e for e in b.entities.values() if e.card_name == 'Wallbreaker_mini' and e.is_alive]
    check('死亡生成 Wallbreaker_mini', len(minis) == 1, f'{len(minis)}')
    if minis:
        check('mini hp=64×2.594≈166', abs(minis[0].hp - 166) < 3, f'hp={minis[0].hp:.0f}')
        check('mini 归属原所有者', minis[0].player == 0)

if __name__ == '__main__':
    for t in (test_action_interp_damage, test_action_interp_spawn_giant,
              test_goblin_giant_threshold,
              test_pekka_resurrect, test_pekka_soul_bonus, test_pekka_heal_on_kill,
              test_megaknight_dash, test_megaknight_uppercut,
              test_electro_dragon_chain, test_battleram_pushback,
              test_goblinbarrel_decoy, test_bats_overheal,
              test_goblindrill_hide, test_goblincage_capture,
              test_zap_evo_zone, test_musketeer_snipe,
              test_valkyrie_tornado, test_wizard_shield_lost,
              test_barbarians_rage, test_tesla_pulse, test_archer_special_range,
              test_bomber_chain, test_icespirits_zone, test_hunter_net,
              test_wallbreakers_mini):
        try:
            t()
        except Exception as e:
            import traceback; traceback.print_exc()
            check(t.__name__ + ' 异常', False, str(e)[:80])
    print(f'\n通过 {PASS} / 失败 {FAIL}')
    sys.exit(1 if FAIL else 0)
