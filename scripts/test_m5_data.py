#!/usr/bin/env python3
"""M7 — 数据接入三卡引擎机制验收（Ronin 格挡反击 / Vines 藤蔓束缚 / Spirit Empress 双费用形态）。

为何新建 test_m5_data.py 而非追加 test_m4_evo7.py：test_m4_evo7.py 是 M6「2025 新觉醒 7 张」
的 51 断言验收基线（回归红线, 数字需保持稳定可对照）；本批三卡为**非觉醒的数据接入机制**
（被动格挡 / 法术领域 / 部署规则）, 机制族与验收口径独立, 独立成档便于归因。

数值来源（详见 src/clasher_new/evo_2025_data.py M7 节与 docs/data_integration_notes.md）：
- [gamedata] Ronin.summonCharacterData（parryReflectPercent=200/parryCooldownMs=3500/parryMeleeOnly=True）、
  Vines.areaEffectObjectData（2s 领域/2 跳/25% 塔/3 目标/拽落/snare 2.5s）、
  MergeMaiden 双形态条目 + cards_stats_characters.MergeMaiden_Mounted（Legendary lv9 轴, lv11 hp=1798/dmg=309）
- [Fandom] docs/_fp_Ronin.txt / _fp_Vines.txt / _fp_SpiritEmpress.txt（2026-09-03 CDP）
- [假设] VinesProjectile 弹速 600、snare 统一档、Ronin 部署即就绪（待 L4）

运行：python3 scripts/test_m5_data.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'clasher_new'))
os.chdir(os.path.join(os.path.dirname(__file__), '..', 'src', 'clasher_new'))

from battle import BattleState, Troop, Position
from card_utils import Card
from player import PlayerState

PASS = FAIL = 0
def check(name, cond, extra=''):
    global PASS, FAIL
    if cond: PASS += 1; print(f'  OK {name}' + (f'  [{extra}]' if extra else ''))
    else: FAIL += 1; print(f'  FAIL {name}' + (f'  [{extra}]' if extra else ''))

def make_battle():
    bs = BattleState(PlayerState(0, ['Ronin', 'MiniPekka', 'Vines', 'MergeMaiden', 'Musketeer',
                                     'Arrows', 'Minions', 'Giant'], 10),
                     PlayerState(1, ['Knight'] * 8, 10))
    return bs

def kill_princess_towers(bs):
    """关掉四座公主塔（King 塔未激活不攻击）, 排除塔防干扰。"""
    for eid in (1, 2, 3, 4):
        bs.entities[eid].is_alive = False

def spawn_troop(bs, card, x, y, player):
    t = Troop(bs.next_entity_id, Position(x, y), player, card, bs)
    bs._spawn_entity(t)
    t.deploy_delay_remaining = 0.0
    return t

def step_for(bs, seconds):
    for _ in range(int(seconds * 60)):
        bs.step(1 / 60)


# ==================== 机制 1：Ronin 格挡反击（被动） ====================

def test_ronin_data():
    """gamedata parry 字段经 card_mechanics.Ronin 挂载。"""
    print('[M7-R] Ronin 格挡反击 — 数据挂载')
    b = make_battle()
    r = spawn_troop(b, 'Ronin', 9.0, 20.0, 0)
    h = r.entity_holder
    check('反射倍率 200%（parryReflectPercent）', abs(h.parry_percent - 2.0) < 1e-9)
    check('冷却 3.5s（parryCooldownMs）', abs(h.parry_cooldown - 3.5) < 1e-9)
    check('仅近战（parryMeleeOnly）', h.parry_melee_only is True)
    check('部署即就绪（timer=0）', h.parry_timer == 0.0)


def test_ronin_parry_reflect():
    """近战首击：格挡（本体不受伤）+ 反弹 200% 给攻击者。"""
    print('[M7-R] Ronin 格挡反击 — 近战格挡 + 反弹 200%')
    b = make_battle()
    kill_princess_towers(b)
    r = spawn_troop(b, 'Ronin', 9.0, 20.0, 0)
    m = spawn_troop(b, 'MiniPekka', 16.0, 12.0, 1)   # 远离战场, 只作手动受击靶
    rhp0, mhp0 = r.hp, m.hp
    r.take_damage(100, source=m)
    step_for(b, 0.1)
    check('本体被格挡（HP 不变）', r.hp == rhp0, f'{r.hp}')
    check('攻击者反弹 200%（-200）', abs(m.hp - (mhp0 - 200)) < 0.01, f'{mhp0 - m.hp}')
    check('进入 3.5s 冷却', abs(r.entity_holder.parry_timer - 3.4) < 0.1,
          f'{r.entity_holder.parry_timer:.2f}')


def test_ronin_cooldown_and_ranged():
    """冷却期间正常受伤；CD 结束再次格挡；远程攻击不触发。"""
    print('[M7-R] Ronin 格挡反击 — 冷却 / 远程')
    b = make_battle()
    kill_princess_towers(b)
    r = spawn_troop(b, 'Ronin', 9.0, 20.0, 0)
    m = spawn_troop(b, 'MiniPekka', 16.0, 12.0, 1)
    mus = spawn_troop(b, 'Musketeer', 16.0, 14.0, 1)
    rhp0 = r.hp
    r.take_damage(100, source=m)          # 第 1 击：格挡
    r.take_damage(100, source=m)          # 冷却期间：正常受伤
    check('冷却期间正常受伤（-100）', abs(r.hp - (rhp0 - 100)) < 0.01, f'{r.hp}')
    step_for(b, 3.6)                      # > 3.5s CD
    r.take_damage(100, source=m)
    step_for(b, 0.1)
    check('CD 结束再次格挡（HP 不变）', abs(r.hp - (rhp0 - 100)) < 0.01, f'{r.hp}')
    check('再次进入冷却', r.entity_holder.parry_timer > 3.0)
    r2 = spawn_troop(b, 'Ronin', 9.0, 24.0, 0)
    r2hp0 = r2.hp
    r2.take_damage(100, source=mus)       # 远程（range 6）不触发
    check('远程攻击正常受伤（-100）', abs(r2.hp - (r2hp0 - 100)) < 0.01, f'{r2.hp}')


# ==================== 机制 2：Vines 藤蔓束缚（法术） ====================

def test_vines_data_and_zone():
    """数据贯通 + 落地生成束缚领域。"""
    print('[M7-V] Vines 藤蔓束缚 — 数据 + 落地领域')
    c = Card('Vines')
    check('弹道伤害 lv11=153（Epic 轴）', abs(c.projectile_data.damage - 153) < 0.01,
          f'{c.projectile_data.damage}')
    check('弹速已注入（快照缺失, [假设] 600）', c.projectile_data.speed > 0,
          f'{c.projectile_data.speed}')
    b = make_battle()
    kill_princess_towers(b)
    g = spawn_troop(b, 'Giant', 9.0, 13.0, 1)
    ok = b.deploy_card(0, 'Vines', Position(9.0, 13.0))
    step_for(b, 1.5)   # 弹道飞行 ~1.0s + 落地
    zones = [e for e in b.entities.values() if e.is_alive and e.name == 'Vines_AeO']
    check('出牌成功且落地生成领域', ok and len(zones) == 1)
    if zones:
        z = zones[0]
        check('领域半径 2.5', abs(z.radius - 2.5) < 0.01, f'{z.radius}')
        check('领域剩余寿命 ≤2s（属性表口径, 飞行 ~1s 后）', 1.0 < z.lifetime <= 2.01,
              f'remaining={z.lifetime:.2f}')
        check('锁定 3 高血目标（targetHighestHp）', z.locked is not None and len(z.locked) <= 3,
              f'{[t.name for t in (z.locked or [])]}')


def test_vines_two_hits_and_tower():
    """两跳伤害（lv11 153×2）；对王塔类 25%（crownTowerDamagePercent）。
    布景：杀蓝方公主塔（防塔击污染 Giant 血量）；王塔 25% 场景直接砸蓝王塔。"""
    print('[M7-V] Vines 藤蔓束缚 — 两跳伤害 + 对塔 25%')
    b = make_battle()
    p = b.players[0]
    for eid in (3, 4):          # 蓝方公主塔关闭（Giant 在蓝塔射程边界内会挨塔击）
        b.entities[eid].is_alive = False
    g = spawn_troop(b, 'Giant', 9.0, 13.0, 1)
    ghp0 = g.hp
    b.deploy_card(0, 'Vines', Position(9.0, 13.0))
    step_for(b, 3.2)            # 飞行 ~1.0s + 领域全程 2s（两跳 @落地/@+1s）
    check('两跳伤害共 306（153×2）', abs(ghp0 - g.hp - 306) < 2.0, f'{ghp0 - g.hp}')
    check('领域 2s 后消失',
          all(not z.is_alive for z in b.entities.values() if z.name == 'Vines_AeO'))
    # 对塔 25%：直接砸红王塔（仅敌方塔受 25% 降伤；无其他敌人 → 锁定集 = 红王塔）
    p.cycle.remove('Vines'); p.cycle.insert(0, 'Vines')
    p.elixir = 10
    king = b.entities[5]    # 红方王塔（player 1）
    khp0 = king.hp
    b.deploy_card(0, 'Vines', Position(9.0, 29.0))
    step_for(b, 5.4)        # 弹道飞行 ~2.6s（横跨全场）+ 领域 2s
    check('对塔 25%/跳（2×153×0.25=76.5）', abs(khp0 - king.hp - 76.5) < 2.0,
          f'{khp0 - king.hp:.2f}')


def test_vines_ground_snare_top3():
    """空中单位拽落 + 可被地面单位攻击 + 束缚不移动 + 3 目标上限。
    拽落靶用 MegaMinion（血厚, 不被旁边骑士在测试窗口内击杀）。"""
    print('[M7-V] Vines 藤蔓束缚 — 拽落 / 束缚 / 3 目标上限')
    b = make_battle()
    kill_princess_towers(b)
    mn = spawn_troop(b, 'MegaMinion', 9.0, 13.0, 1)
    kn = spawn_troop(b, 'Knight', 10.0, 13.0, 0)
    b.deploy_card(0, 'Vines', Position(9.0, 13.0))
    step_for(b, 1.5)
    check('空中单位被拽落（groundsAirUnits）', mn.data.is_air_unit is False)
    kn.update_current_target()
    check('地面单位可攻击被拽落目标', kn.target_id == mn.id,
          f'target={b.entities.get(kn.target_id).name if kn.target_id in b.entities else None}')
    check('束缚中（freeze>0, 不可移动/攻击）', mn.freeze_timer > 2.0, f'{mn.freeze_timer:.2f}')
    mn_pos = Position(mn.position.x, mn.position.y)
    step_for(b, 0.5)
    check('束缚期间不自主移动（无碰撞干扰）',
          abs(mn.position.x - mn_pos.x) < 0.05 and abs(mn.position.y - mn_pos.y) < 0.05)
    step_for(b, 1.8)   # snare 2.5s（自落地 ~t=1.0 起 → t≈3.5 结束）
    check('束缚结束后复飞', mn.is_alive and mn.data.is_air_unit is True,
          f'air={mn.data.is_air_unit} alive={mn.is_alive}')
    # 3 目标上限：4 敌在场（含 2 空中）只锁 3
    b2 = make_battle()
    kill_princess_towers(b2)
    spawn_troop(b2, 'Giant', 9.0, 13.0, 1)
    spawn_troop(b2, 'Knight', 9.5, 13.0, 1)
    m1 = spawn_troop(b2, 'Minions', 8.5, 13.0, 1)
    m2 = spawn_troop(b2, 'Minions', 9.2, 12.8, 1)
    b2.deploy_card(0, 'Vines', Position(9.0, 13.0))
    step_for(b2, 1.5)
    z = [e for e in b2.entities.values() if e.is_alive and e.name == 'Vines_AeO'][0]
    m1_hurt = m1.hp < m1.data.hp
    m2_hurt = m2.hp < m2.data.hp
    check('3 目标上限（只锁 3 个, 第 4 个不受损）',
          len(z.locked) == 3 and (m1_hurt != m2_hurt),
          f'locked={[t.name for t in z.locked]} m1={m1_hurt} m2={m2_hurt}')


# ==================== 机制 3：Spirit Empress 双费用形态（部署规则） ====================

def _mm_battle(elixir):
    b = make_battle()
    p = b.players[0]
    p.cycle = ['MergeMaiden'] + ['Knight'] * 7
    p.elixir = elixir
    kill_princess_towers(b)
    return b, p

def _mm_forms(b):
    return [e for e in b.entities.values()
            if e.id > 6 and e.is_alive and e.card_name.startswith('MergeMaiden')]


def test_mergemaiden_data():
    """两形态数值贯通（官方数值表行 + M7 补建 Normal 行）。"""
    print('[M7-M] Spirit Empress 双费用形态 — 数值')
    m = Card('MergeMaiden_Mounted')
    check('飞行形态 lv11：hp 1798 / dmg 309（Legendary 轴）',
          round(m.hp) == 1798 and round(m.damage) == 309, f'{m.hp}/{m.damage}')
    check('飞行形态：攻速 1.6 / 射程 5 / 对空对地 / 飞行',
          abs(m.hit_speed - 1.6) < 1e-9 and abs(m.range - 5.0) < 1e-9
          and m.attack_air and m.attack_ground and m.is_air_unit)
    n = Card('MergeMaiden_Normal')
    check('地面形态：攻速 1.2 / 近战 1.2 / 仅对地 / 地面',
          abs(n.hit_speed - 1.2) < 1e-9 and abs(n.range - 1.2) < 1e-9
          and (not n.attack_air) and n.attack_ground and (not n.is_air_unit))
    check('两形态共享 hp/dmg（Fandom 表口径）',
          round(n.hp) == 1798 and round(n.damage) == 309)


def test_mergemaiden_forms():
    """圣水 ≥6 → 6 费飞行远程；<6 → 3 费地面；恰好 6 边界；手牌/圣水/last_card 一致性。"""
    print('[M7-M] Spirit Empress 双费用形态 — 部署规则')
    b, p = _mm_battle(8)
    ok = b.deploy_card(0, 'MergeMaiden', Position(9.0, 12.0))
    forms = _mm_forms(b)
    check('圣水 8 → 飞行远程形态（Mounted）',
          ok and len(forms) == 1 and forms[0].card_name == 'MergeMaiden_Mounted')
    check('圣水 8 → 扣 6 费', abs(p.elixir - 2.0) < 1e-9, f'{p.elixir}')
    check('手牌循环按 MergeMaiden 推进 + last_card 记录',
          'MergeMaiden' in p.cycle and p.cycle[-1] == 'MergeMaiden' and p.last_card == 'MergeMaiden',
          f'{p.cycle[-1]}')
    check('镜像费用锚点=实际消耗（last_card_cost=6）', getattr(p, 'last_card_cost', None) == 6)
    # 圣水 <6 → 地面形态（把 MergeMaiden 挪回手牌首位）
    step_for(b, 1.0)
    p.elixir = 5
    p.cycle.remove('MergeMaiden'); p.cycle.insert(0, 'MergeMaiden')
    ok = b.deploy_card(0, 'MergeMaiden', Position(9.0, 12.0))
    forms = [e for e in _mm_forms(b) if e.card_name == 'MergeMaiden_Normal']
    check('圣水 5 → 地面近战形态（Normal）',
          ok and len(forms) == 1 and forms[0].data.is_air_unit is False)
    check('圣水 5 → 扣 3 费', abs(p.elixir - 2.0) < 1e-9, f'{p.elixir}')
    # 边界：恰好 6 → 空中形态
    p.elixir = 6.0
    p.cycle.remove('MergeMaiden'); p.cycle.insert(0, 'MergeMaiden')
    ok = b.deploy_card(0, 'MergeMaiden', Position(9.0, 12.0))
    mounted = [e for e in _mm_forms(b) if e.card_name == 'MergeMaiden_Mounted']
    check('边界恰好 6 → 空中形态且扣 6',
          ok and len(mounted) == 2 and abs(p.elixir - 0.0) < 1e-9, f'elixir={p.elixir}')
    # 圣水不足实际费用 → 拒绝
    p.elixir = 2.0
    p.cycle.remove('MergeMaiden'); p.cycle.insert(0, 'MergeMaiden')
    n_before = len(_mm_forms(b))
    ok = b.deploy_card(0, 'MergeMaiden', Position(9.0, 12.0))
    check('圣水 2 不足 3 费 → 拒绝且不扣费',
          (not ok) and len(_mm_forms(b)) == n_before and abs(p.elixir - 2.0) < 1e-9)


def test_mergemaiden_mirror():
    """镜像复制上一形态并按其「实际费用+1」（官方口径）。"""
    print('[M7-M] Spirit Empress 双费用形态 — 镜像')
    b, p = _mm_battle(10)
    p.cycle = ['MergeMaiden', 'Mirror'] + ['Knight'] * 6
    b.deploy_card(0, 'MergeMaiden', Position(9.0, 12.0))   # 圣水 10 ≥6 → 空中, 扣 6 → 剩 4
    mirror_cost = p.last_card_cost
    p.elixir = 10   # 补足镜像费用（镜像需 实际费用+1 = 7）
    ok = b.deploy_card(0, 'Mirror', Position(9.0, 12.0))
    forms = _mm_forms(b)
    check('镜像复制空中形态（2 个 Mounted）',
          ok and sum(1 for e in forms if e.card_name == 'MergeMaiden_Mounted') == 2)
    check('镜像按「实际费用+1」扣费（6+1=7）',
          abs(p.elixir - (10 - (mirror_cost + 1))) < 1e-9,
          f'remaining={p.elixir:.1f} base={mirror_cost}')


if __name__ == '__main__':
    for t in (test_ronin_data, test_ronin_parry_reflect, test_ronin_cooldown_and_ranged,
              test_vines_data_and_zone, test_vines_two_hits_and_tower,
              test_vines_ground_snare_top3,
              test_mergemaiden_data, test_mergemaiden_forms, test_mergemaiden_mirror):
        try:
            t()
        except Exception as e:
            import traceback; traceback.print_exc()
            check(t.__name__ + ' 异常', False, str(e)[:80])
    print(f'\n通过 {PASS} / 失败 {FAIL}')
    sys.exit(1 if FAIL else 0)
