#!/usr/bin/env python3
"""M1 机制族测试：递增伤害 + 弹道生成链。直接 python3 scripts/test_m1.py 运行。"""
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 'src', 'clasher_new')
sys.path.insert(0, SRC)
os.chdir(SRC)  # card_utils 以相对路径读取 json

import battle
import player
from core import Position

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(('  OK ' if cond else '  FAIL ') + name + (f'  [{detail}]' if detail else ''))


def make_battle(deck0=None, deck1=None):
    return battle.BattleState(
        player.PlayerState(0, deck0 or ['Knight', 'Mirror', 'Firecracker', 'Arrows',
                                        'GoblinBarrel', 'Graveyard', 'Clone', 'InfernoDragon'], 10),
        player.PlayerState(1, deck1 or ['Valkyrie', 'Musketeer', 'Giant', 'Minions',
                                        'Fireball', 'Arrows', 'Knight', 'Skeletons'], 10))


def step_for(b, seconds):
    for _ in range(int(seconds * 60)):
        b.step(1 / 60)


# ---------- 族 1：递增伤害 ----------
def test_ramp_inferno_dragon():
    print('[test] 地狱飞龙三段递增')
    b = make_battle()
    tower = b.entities[1]  # red princess tower
    dragon = battle.Troop(b.next_entity_id, Position(tower.position.x - 1.5, tower.position.y - 0.5), 0, 'InfernoDragon')
    b._spawn_entity(dragon)
    last_hp = tower.hp
    damages = []
    for i in range(int(6.5 * 60)):
        b.step(1 / 60)
        if tower.hp < last_hp:
            damages.append(round(last_hp - tower.hp, 1))
            last_hp = tower.hp
    uniq = sorted(set(damages))
    dragon_card_dmg = dragon.data.damage
    expected = sorted({round(dragon_card_dmg, 1),
                       round(dragon.data.ramp_stage_damages[0], 1),
                       round(dragon.data.ramp_stage_damages[1], 1)})
    check('地狱龙蓄力阶段覆盖 0/1/2', dragon.ramp_stage >= 2, f'final_stage={dragon.ramp_stage}')
    check(f'伤害阶梯符合公式 {expected}（实测 {uniq}）',
          all(any(abs(u - e) < 0.6 for u in uniq) for e in expected), f'damages={uniq}')
    # 对照 Fandom 实测值（11 级：35/120/422，本引擎 dpl 略有舍入差异）
    s1, s2, s3 = dragon.data.damage, *dragon.data.ramp_stage_damages
    check(f'对照 Fandom 官方数值 35/120/422（引擎 {s1:.0f}/{s2:.0f}/{s3:.0f}）',
          abs(s2 - 120) <= 2 and abs(s3 - 422) <= 6)
    check('伤害单调不减', damages == sorted(damages))


def test_ramp_inferno_tower():
    print('[test] 地狱塔（建筑直伤路径）')
    b = make_battle()
    it = battle.Building(b.next_entity_id, Position(3.5, 26.0), 1, 'InfernoTower', False)
    b._spawn_entity(it)
    knight = battle.Troop(b.next_entity_id, Position(3.5, 24.0), 0, 'Knight')
    b._spawn_entity(knight)
    step_for(b, 1.0)  # 部署缓冲
    princess_dmg = battle.Card("King_PrincessTowers").projectile_data.damage  # 红塔弹丸伤害，按源过滤
    last_hp = knight.hp
    damages = []
    for i in range(int(6.0 * 60)):
        b.step(1 / 60)
        if knight.hp < last_hp:
            damages.append(round(last_hp - knight.hp, 1))
            last_hp = knight.hp
    uniq = sorted(set(d for d in damages if abs(d - princess_dmg) > 0.5))
    base = it.data.damage
    exp = [round(base, 1)] + [round(x, 1) for x in it.data.ramp_stage_damages]
    check(f'地狱塔伤害阶梯符合缩放公式 {exp}（实测 {uniq}，已滤红塔 {princess_dmg}）',
          len(uniq) >= 3
          and abs(uniq[1] / uniq[0] - exp[1] / exp[0]) < 0.05
          and abs(uniq[-1] / uniq[0] - exp[2] / exp[0]) < 0.05,
          f'base={base} stage_damages={[round(x,1) for x in it.data.ramp_stage_damages]}')


def test_ramp_reset():
    print('[test] 换目标重置蓄力')
    b = make_battle()
    tower = b.entities[1]
    d = battle.Troop(b.next_entity_id, Position(tower.position.x - 1.5, tower.position.y - 0.5), 0, 'InfernoDragon')
    b._spawn_entity(d)
    step_for(b, 5.5)  # 部署1s + 蓄力4.5s → 应达阶段2（避开2.0s阈值边界）
    stage_before = d.ramp_stage
    d.target_id = None
    step_for(b, 0.1)
    check(f'脱锁后蓄力重置（{stage_before} → {d.ramp_stage}）',
          stage_before >= 2 and d.ramp_stage == 0 and d.ramp_timer < 0.2)  # 重锁后从零重新蓄力


# ---------- 族 2：弹道生成链 ----------
def test_firecracker_chain():
    print('[test] 烟花射手爆裂弹链')
    b = make_battle()
    fc = battle.Troop(b.next_entity_id, Position(9.0, 10.0), 0, 'Firecracker')
    b._spawn_entity(fc)
    knight = battle.Troop(b.next_entity_id, Position(9.0, 12.0), 1, 'Knight')
    b._spawn_entity(knight)
    seen_explosion, knight_min_hp = False, knight.hp
    for i in range(int(6.0 * 60)):
        b.step(1 / 60)
        if any(e.name == 'FirecrackerExplosion' for e in b.entities.values()):
            seen_explosion = True
        if knight.is_alive:
            knight_min_hp = min(knight_min_hp, knight.hp)
    check('命中后生成 FirecrackerExplosion 二段弹', seen_explosion)
    check('爆裂弹造成非零伤害', knight_min_hp < knight.data.hp, f'knight hp {knight.data.hp}→{knight_min_hp}')


def test_goblin_barrel():
    print('[test] 哥布林飞桶落地出兵')
    b = make_battle(deck0=['GoblinBarrel', 'Knight', 'Mirror', 'Firecracker',
                           'Graveyard', 'Clone', 'InfernoDragon', 'Arrows'])
    ok = b.deploy_card(0, 'GoblinBarrel', Position(3.5, 28.0))
    check('飞桶部署成功', ok)
    step_for(b, 4.0)
    goblins = [e for e in b.entities.values()
               if isinstance(e, battle.Troop) and e.player == 0 and e.name == 'Goblins' and e.is_alive]
    check(f'落点出现 3 只哥布林（实测 {len(goblins)}）', len(goblins) >= 3)


def test_graveyard():
    print('[test] 墓园持续出兵')
    b = make_battle(deck0=['Graveyard', 'Knight', 'Mirror', 'Firecracker',
                           'GoblinBarrel', 'Clone', 'InfernoDragon', 'Arrows'])
    ok = b.deploy_card(0, 'Graveyard', Position(3.5, 28.0))
    check('墓园部署成功', ok)
    seen_ids = set()
    for i in range(int(11.0 * 60)):
        b.step(1 / 60)
        for e in b.entities.values():
            if isinstance(e, battle.Troop) and e.player == 0 and e.name == 'Skeletons':
                seen_ids.add(e.id)
    check(f'持续刷出骷髅（约14只，观测 {len(seen_ids)}）', len(seen_ids) >= 10)


def test_clone():
    print('[test] 克隆法术')
    b = make_battle(deck0=['Knight', 'Clone', 'Mirror', 'Firecracker',
                           'GoblinBarrel', 'Graveyard', 'InfernoDragon', 'Arrows'])
    b.deploy_card(0, 'Knight', Position(9.0, 10.0))
    step_for(b, 0.5)
    b.deploy_card(0, 'Clone', Position(9.0, 10.0))
    step_for(b, 0.5)
    knights = [e for e in b.entities.values()
               if isinstance(e, battle.Troop) and e.player == 0 and e.name == 'Knight' and e.is_alive]
    check(f'复制出第二个骑士（{len(knights)}）', len(knights) == 2)
    check('克隆体 1 血', any(k.hp == 1.0 for k in knights), f'hps={[round(k.hp,1) for k in knights]}')


def test_mirror():
    print('[test] 镜像法术重放')
    b = make_battle()
    ok1 = b.deploy_card(0, 'Knight', Position(9.0, 10.0))
    step_for(b, 0.5)
    elixir_before = b.players[0].elixir
    ok2 = b.deploy_card(0, 'Mirror', Position(9.5, 10.0))
    spent = elixir_before - b.players[0].elixir  # 部署后立即读数，排除回复干扰
    step_for(b, 0.5)
    knights = [e for e in b.entities.values()
               if isinstance(e, battle.Troop) and e.player == 0 and e.name == 'Knight' and e.is_alive]
    check('镜像部署成功', ok1 and ok2)
    check(f'再产一个骑士（{len(knights)}）', len(knights) == 2)
    check(f'镜像费用 = 基础费+1（实扣 {spent:.2f}）', abs(spent - 4.0) < 0.01)
    check('last_card 不被 Mirror 覆盖', b.players[0].last_card == 'Knight')


# ---------- 回归冒烟 ----------
def test_smoke_regression():
    print('[test] 回归冒烟：60s 混战不抛异常')
    import random
    random.seed(7)
    b = make_battle(deck0=['Knight', 'Valkyrie', 'InfernoDragon', 'Firecracker',
                           'GoblinBarrel', 'Graveyard', 'Clone', 'Mirror'],
                    deck1=['Giant', 'Musketeer', 'Minions', 'Fireball', 'Arrows',
                           'Skeletons', 'InfernoTower', 'Knight'])
    try:
        for i in range(60 * 60):
            if b.game_over: break
            if i % 45 == 0:  # 每约 0.75s 双方随机出牌
                for pid in (0, 1):
                    card = random.choice(b.players[pid].cycle)
                    b.deploy_card(pid, card, Position(random.uniform(2, 16), random.uniform(3, 28)))
            b.step(1 / 60)
        check('60s 混战无异常', True)
    except Exception as e:
        check('60s 混战无异常', False, f'{type(e).__name__}: {e}')


if __name__ == '__main__':
    for t in (test_ramp_inferno_dragon, test_ramp_inferno_tower, test_ramp_reset,
              test_firecracker_chain, test_goblin_barrel, test_graveyard,
              test_clone, test_mirror, test_smoke_regression):
        t()
    print(f'\n通过 {len(PASS)} / 失败 {len(FAIL)}')
    if FAIL:
        print('失败项:', FAIL)
        sys.exit(1)
