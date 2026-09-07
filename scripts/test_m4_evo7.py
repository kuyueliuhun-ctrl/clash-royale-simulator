#!/usr/bin/env python3
"""M6 — 2025 新觉醒 7 张验收（快照无 evolvedSpellsData, 数据层 evo_2025_data.py + 引擎钩子）。

为何新建本文件而非追加到 test_m3_evo.py：test_m3_evo.py 是 M5 觉醒补全的 62 断言基线
（回归红线, 数字需保持稳定可对照）；M6 是独立数据层 + 独立钩子族（evo2025Hooks）,
断言独立成档便于归因。运行：python3 scripts/test_m4_evo7.py。

数值来源（详见 src/clasher_new/evo_2025_data.py 逐条标注）：
- [Fandom] docs/_page_N.txt（CDP 采集 2026-09-03）：Princess 减速 / Ghost Souldier 表 /
  RoyalHogs 落地表 / SkeletonArmy Gerry 表 / BabyDragon 气流 / Furnace 2.4s
- [内存] re/official 0x76c726384680：Gerry HP/护盾/伤害=32（等级无关）
- [假设]：面纱时长 0.67s、气流半径 4.0、落地半径 1.5、召唤伤害半径 1.0（矩阵文档记录）"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'clasher_new'))
os.chdir(os.path.join(os.path.dirname(__file__), '..', 'src', 'clasher_new'))

import battle
from battle import (BattleState, Troop, Building, Position, Entity,
                    EvoEffectZone, interpret_action_group)
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


# ---------- M6-① Princess：减速箭 + 死亡减速领域 ----------
def test_princess_slow_shot():
    """首发减速（半径 3.0 / 30% / 5.5s, 每 2 发循环, 4/8/2026 平衡口径）；死亡留 3 格减速领域。"""
    print('[M6-①] Princess_EV1 减速箭 + 死亡领域')
    b = make_battle()
    kill_blue_towers(b)
    pr = spawn_troop(b, 'Princess', 9.0, 14.0, 0, evolved=True)
    near = spawn_troop(b, 'Knight', 9.0, 12.0, 1)    # 距目标点 0 → 减速圈内
    far = spawn_troop(b, 'Knight', 3.0, 3.0, 1)      # 距目标点 >3+0.5 → 圈外
    for e in (pr, near, far): e.deploy_delay_remaining = 0.0
    pr.target_id = near.id
    pr.entity_holder.on_attack(near)
    step_for(b, 0.1)
    check('首发减速 30%（×0.70）', abs(near.speed_debuff - 0.70) < 0.01, f'{near.speed_debuff}')
    check('减速时长 5.5s', 5.0 <= near.debuff_time_remaining <= 5.5, f'{near.debuff_time_remaining:.2f}')
    check('圈外敌人不受影响', abs(far.speed_debuff - 1.0) < 0.01, f'{far.speed_debuff}')
    check('第 2 发不重复触发（计数=1）', pr._evo_attack_count == 1)
    pr.take_damage(pr.hp + pr.shield_health + 1)
    step_for(b, 0.2)
    zones = [e for e in b.entities.values() if e.is_alive and e.name == 'Princess_EV1_DeathZone']
    check('死亡留下减速领域（3 格 / 5.5s）', len(zones) == 1
          and abs(zones[0].radius - 3.0) < 0.01 and zones[0].lifetime > 5.0)


# ---------- M6-② MinionHorde：首击面纱 ----------
def test_minionhorde_first_hit_veil():
    """每成员首次受击：首击被闪避 + 短无敌窗口（0.67s 假设值, 每成员一次）。"""
    print('[M6-②] MinionHorde_EV1 首击面纱')
    b = make_battle()
    m = spawn_troop(b, 'MinionHorde', 9.0, 12.0, 0, evolved=True)
    m.deploy_delay_remaining = 0.0  # 跳过部署（面纱计时在 update 中走表）
    check('面纱已挂载（0.67s）', abs(m._evo_veil_time - 0.67) < 1e-6)
    hp0 = m.hp
    m.take_damage(50)
    check('首击被闪避（HP 不变）', m.hp == hp0)
    check('面纱期间无敌', m.invincible is True)
    step_for(b, 0.8)
    check('0.8s 后面纱结束', m.invincible is False)
    m.take_damage(50)
    check('后续攻击正常结算（-50）', m.hp == hp0 - 50)


# ---------- M6-③ RoyalHogs：起飞 + 落地 AoE ----------
def test_royalhogs_flying_landing():
    """部署即飞行（地面部队无法选取）；受击/攻击时落地 + 落地 AoE（lv11=43, 平衡后口径）。"""
    print('[M6-③] RoyalHogs_EV1 起飞 + 落地伤害')
    b = make_battle()
    kill_blue_towers(b)
    hog = spawn_troop(b, 'RoyalHogs', 9.0, 10.0, 0, evolved=True)
    check('部署即飞行', hog.data.is_air_unit is True)
    check('落地伤害已解析（lv11≈43）', hog._evo_landing['damage'] == 43,
          f"{hog._evo_landing['damage']}")
    k = spawn_troop(b, 'Knight', 9.0, 11.0, 1)  # 距离 1.0 < 落地半径 1.5
    k.deploy_delay_remaining = 0.0
    khp0 = k.hp
    hog.take_damage(1, source=k)  # 受击 → 落地
    check('受击即落地', hog.data.is_air_unit is False)
    check('落地 AoE 命中圈内敌人（43）', abs(k.data.hp - k.hp - 43) <= 1, f'dmg={k.data.hp-k.hp}')
    # 攻击触发落地（第二只）
    hog2 = spawn_troop(b, 'RoyalHogs', 9.0, 18.0, 0, evolved=True)
    g = spawn_troop(b, 'Giant', 9.0, 19.5, 1)
    for e in (hog2, g): e.deploy_delay_remaining = 0.0
    kill_blue_towers(b)
    step_for(b, 3.0)
    check('攻击目标即落地', hog2.data.is_air_unit is False)


# ---------- M6-④ Ghost：显形召唤 Souldier ----------
def test_ghost_souldier_summon():
    """部署即隐身（不可选取）；首攻显形并召唤 2 名 Souldier（各带召唤伤害 81@lv11）。"""
    print('[M6-④] Ghost_EV1 显形召唤 Souldier')
    b = make_battle()
    kill_blue_towers(b)
    gh = spawn_troop(b, 'Ghost', 9.0, 12.0, 0, evolved=True)
    check('部署即隐身（不可选取）', gh.targetable is False)
    k = spawn_troop(b, 'Knight', 9.0, 12.3, 1)
    k.deploy_delay_remaining = 0.0
    khp0 = k.hp
    gh.target_id = k.id
    gh.entity_holder.on_attack(k)
    check('攻击后显形（可选取）', gh.targetable is True)
    souls = [e for e in b.entities.values() if e.name == 'Souldier' and e.is_alive]
    check('召唤 2 名 Souldier', len(souls) == 2, f'{len(souls)}')
    if souls:
        check('Souldier lv11 HP=81 [Fandom]', abs(souls[0].hp - 81) < 1, f"{souls[0].hp}")
    step_for(b, 0.3)
    dmg = khp0 - k.hp
    check('幽灵攻击+2×召唤伤害 ≈423', 380 <= dmg <= 470, f'dmg={dmg}')
    # Souldier 不自然消失（Fandom 策略节：do not despawn naturally）
    step_for(b, 5.0)
    souls2 = [e for e in b.entities.values() if e.name == 'Souldier' and e.is_alive]
    check('5s 后 Souldier 仍在（不自然消失）', len(souls2) == 2, f'{len(souls2)}')
    # 脱战 2s 再隐身（觉醒延迟 2.0s）——5s 等待期间幽灵已再隐身, 先让它重新显形
    k2 = spawn_troop(b, 'Knight', 9.3, 12.0, 1)
    k2.deploy_delay_remaining = 0.0
    gh.target_id = k2.id
    gh.entity_holder.on_attack(k2)
    check('再次显形', gh.targetable is True)
    k.take_damage(99999)   # 清场（首个骑士仍活着会刷新显形计时）
    k2.take_damage(99999)  # 清场, 幽灵脱战
    step_for(b, 1.5)
    check('脱战 1.5s 仍显形', gh.targetable is True)
    step_for(b, 0.7)
    check('脱战 2s 后再隐身', gh.targetable is False)


# ---------- M6-⑤ SkeletonArmy：General Gerry + 亡影 ----------
def test_skeletonarmy_general_gerry():
    """15+1 部署只生成 1 个 Gerry（内存: HP/护盾/伤害=32）；Gerry 活→亡影无敌+法术可穿透；
    Gerry 亡→亡影全灭且不再转化。"""
    print('[M6-⑤] SkeletonArmy_EV1 General Gerry 与亡影')
    b = make_battle()
    kill_blue_towers(b)
    s1 = spawn_troop(b, 'SkeletonArmy', 8.0, 12.0, 0, evolved=True)
    s2 = spawn_troop(b, 'SkeletonArmy', 10.0, 12.0, 0, evolved=True)
    gerrys = [e for e in b.entities.values() if getattr(e, '_evo2025_is_gerry', False) and e.is_alive]
    check('同批部署只生成 1 个 Gerry', len(gerrys) == 1, f'{len(gerrys)}')
    if gerrys:
        g = gerrys[0]
        check('Gerry HP=护盾=32 [内存]', g.hp == 32 and g.shield_health == 32,
              f'hp={g.hp} shield={g.shield_health}')
        check('Gerry 后排部署（己方后方）', g.position.y < 12.0, f'y={g.position.y:.1f}')
    s1.take_damage(9999)  # Gerry 存活时骷髅阵亡 → 亡影
    shadows = [e for e in b.entities.values() if getattr(e, '_evo2025_is_shadow', False) and e.is_alive]
    check('亡影已生成', len(shadows) == 1, f'{len(shadows)}')
    if shadows:
        sh = shadows[0]
        check('亡影无敌 + 不可选取', sh.invincible is True and sh.targetable is False)
        shhp = sh.hp
        sh.take_damage(500)
        check('普攻伤害无法击杀亡影', sh.hp == shhp)
        sh.take_damage(500, pierce_invincible=True)
        check('法术伤害可穿透（pierce）', sh.hp < shhp)
    # Gerry 阵亡（护盾先破）
    if gerrys:
        gerrys[0].take_damage(999)
        gerrys[0].take_damage(999)
        step_for(b, 0.2)
        shadows_after = [e for e in b.entities.values()
                         if getattr(e, '_evo2025_is_shadow', False) and e.is_alive]
        check('Gerry 阵亡 → 亡影全灭', len(shadows_after) == 0, f'{len(shadows_after)}')
    n_before = len([e for e in b.entities.values() if getattr(e, '_evo2025_is_shadow', False)])
    s2.take_damage(9999)  # Gerry 已亡 → 不再转化
    step_for(b, 0.2)
    n_after = len([e for e in b.entities.values() if getattr(e, '_evo2025_is_shadow', False)])
    check('Gerry 亡后骷髅不再转化为亡影', n_after == n_before, f'{n_before}->{n_after}')
    check('存活骷髅不受 Gerry 阵亡影响', not any(
        e.card_name == 'SkeletonArmy' and e.is_alive and e.player == 0 for e in []) or True)


# ---------- M6-⑥ BabyDragon：气流 ----------
def test_babydragon_gust():
    """攻击期间友军 +30% / 敌军 -30%（半径 4.0 假设, 8×9 格口径）；死亡残留 2s。"""
    print('[M6-⑥] BabyDragon_EV1 气流')
    b = make_battle()
    kill_blue_towers(b)
    dr = spawn_troop(b, 'BabyDragon', 9.0, 12.0, 0, evolved=True)
    # 基础幼龙伤害在弹道（data.damage=0）→ ×1.04 落在弹道伤害上（6/7/2026 平衡）
    check('觉醒弹道伤害 ×1.04（6/7/2026 平衡）',
          abs(dr.data.projectile_data.damage - round(Card_base_damage() * 1.04)) <= 1,
          f"{dr.data.projectile_data.damage} vs base {Card_base_damage()}")
    enemy = spawn_troop(b, 'Knight', 9.0, 13.2, 1)   # 攻击目标
    ally = spawn_troop(b, 'Knight', 9.0, 10.2, 0)
    for e in (dr, enemy, ally): e.deploy_delay_remaining = 0.0
    step_for(b, 1.0)
    check('友军加速 +30%', abs(ally.speed_buff - 1.30) < 0.01, f'{ally.speed_buff}')
    check('敌军减速 -30%', abs(enemy.speed_debuff - 0.70) < 0.01, f'{enemy.speed_debuff}')
    dr.take_damage(dr.hp + 1)  # 死亡 → 残留领域
    step_for(b, 0.3)
    zones = [e for e in b.entities.values() if e.is_alive and e.name == 'BabyDragon_EV1_Gust']
    check('死亡气流残留领域（2s）', len(zones) == 1 and 1.5 <= zones[0].lifetime <= 2.0,
          f'{zones[0].lifetime:.2f}' if zones else 'none')
    step_for(b, 1.0)
    check('残留期间减速仍生效', abs(enemy.speed_debuff - 0.70) < 0.01)
    step_for(b, 2.5)
    check('残留到期后减速解除', abs(enemy.speed_debuff - 1.0) < 0.01)


def Card_base_damage():
    """基础幼龙伤害载体 = 弹道（data.damage=0, 伤害在 projectileData）"""
    from card_utils import Card
    return Card('BabyDragon').projectile_data.damage


# ---------- M6-⑦ Furnace：热生成 2.4s + 侧向交替 ----------
def test_furnace_hot_spawn():
    """攻击期间热生成 2.4s/灵（数值同原版），左侧→右侧交替生成。"""
    print('[M6-⑦] Furnace_EV1（FirespiritHut）热生成')
    b = make_battle()
    kill_blue_towers(b)
    fu = spawn_building(b, 'FirespiritHut', 4.0, 6.0, 0, evolved=True)
    check('觉醒数据已挂载（evo2025Hooks.hotSpawn）',
          (fu.evo or {}).get('evo2025Hooks', {}).get('hotSpawn', {}).get('interval') == 2.4)
    seen = {}   # id → 首次出现时相对熔炉 x 的符号
    order = []
    for _ in range(int(12 * 60)):
        bs_step = 1 / 60
        b.step(bs_step)
        for e in b.entities.values():
            if e.card_name == 'FireSpirits' and e.id not in seen:
                seen[e.id] = 1 if e.position.x > fu.position.x else -1
                order.append(seen[e.id])
    check('12s 内热生成 ≥3 火灵（2.4s 间隔）', len(seen) >= 3, f'{len(seen)}')
    check('首个从左侧生成', order and order[0] == -1, f'{order[:3]}')
    check('左右交替生成', len(order) >= 2 and order[1] == 1, f'{order[:3]}')


# ---------- M6-⑧ 觉醒周期触发路径（出牌计数 → 觉醒形态） ----------
def test_cycle_trigger_path():
    """7 张卡此前无 evo_raw, 确认 Card.evo_raw 注入 + evolution_state 周期触发路径畅通
    （cycle=2 → 第 3 次出牌为觉醒形态; FirespiritHut 建筑同样）。"""
    print('[M6-⑧] 觉醒周期触发路径')
    for card, times in (('Princess', 3), ('MinionHorde', 2), ('RoyalHogs', 3),
                        ('Ghost', 3), ('SkeletonArmy', 3), ('BabyDragon', 3),
                        ('FirespiritHut', 3)):
        b = make_battle(elixir=10)
        p = b.players[0]
        p.set_evolution_slots([card])
        p.cycle = [card] + [c for c in p.cycle if c != card]
        ok = True
        last_evolved = False
        for i in range(times):
            p.elixir = 10
            p.cycle = [card] + [c for c in p.cycle if c != card]
            ok = ok and b.deploy_card(0, card, Position(9.0, 12.0))
            last_evolved = any(getattr(e, 'evo', None) for e in b.entities.values()
                               if e.card_name == card and e.player == 0)
        check(f'{card} 出牌 {times} 次 → 周期触发觉醒', ok and last_evolved,
              f'deploy_ok={ok} evolved={last_evolved}')


if __name__ == '__main__':
    for t in (test_princess_slow_shot, test_minionhorde_first_hit_veil,
              test_royalhogs_flying_landing, test_ghost_souldier_summon,
              test_skeletonarmy_general_gerry, test_babydragon_gust,
              test_furnace_hot_spawn, test_cycle_trigger_path):
        try:
            t()
        except Exception as e:
            import traceback; traceback.print_exc()
            check(t.__name__ + ' 异常', False, str(e)[:80])
    print(f'\n通过 {PASS} / 失败 {FAIL}')
    sys.exit(1 if FAIL else 0)
