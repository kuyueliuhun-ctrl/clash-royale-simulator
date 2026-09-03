#!/usr/bin/env python3
"""批量卡牌冒烟测试：全 gamedata 卡 构造→部署→30s 战斗，检查「有行为」（生成单位或造成伤害）。
产出：
  1. 控制台覆盖表（OK / NO-ACT / 异常）
  2. docs/batch_smoke_report.json（供 coverage.py 与人工评审消费）
  3. 退出码：有异常为 1，NO-ACT 不视为失败（可能为测试场景假象，如 Clone 需友军在场）

运行：cd src/clasher_new && python3 ../../scripts/batch_smoke.py
"""
import sys, os, json, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'clasher_new'))
os.chdir(os.path.join(os.path.dirname(__file__), '..', 'src', 'clasher_new'))

from card_utils import Card, card_data
from battle import BattleState, Position, Troop
from player import PlayerState

DURATION = 30  # 秒


def smoke(name):
    """返回 (status, detail)。status ∈ {OK, NO-ACT, 异常前缀}"""
    try:
        Card(name)
    except Exception as e:
        return f'CONSTRUCT {type(e).__name__}: {str(e)[:60]}'
    bs = BattleState(PlayerState(0, [name] + ['Knight'] * 7, 99), PlayerState(1, ['Knight'] * 8, 99))
    p1_king_hp0 = bs.entities[6].hp
    # 敌方下个骑士当靶子：直接生成实体（绕过部署合法性检查，放在待测卡旁边 1 格）
    enemy_knight = Troop(bs.next_entity_id, Position(9.0, 13.0), 1, 'Knight', bs)
    bs._spawn_entity(enemy_knight)
    # Mirror 需要 last_card：先下一张牌建立镜像对象
    if name == 'Mirror':
        bs.deploy_card(0, 'Knight', Position(8.0, 14.0))
    try:
        ok = bs.deploy_card(0, name, Position(9.0, 14.0))
        if not ok:
            return 'DEPLOY-REJECTED'
    except Exception as e:
        return f'DEPLOY {type(e).__name__}: {str(e)[:60]}'
    # 行为信号：①敌方骑士（靶子）掉血；②敌方王塔掉血；③友军实体数超出部署基数（生成单位类）
    def friendly_non_tower():
        return sum(1 for e in bs.entities.values() if e.id > 6 and e.player == 0 and e.is_alive)
    def enemy_knight_hp():
        h = p1_king_hp0
        for e in bs.entities.values():
            if e.card_name == 'Knight' and e.player == 1 and e.is_alive:
                h = min(h, e.hp)
        return h
    base_friendly = friendly_non_tower()
    knight_hp0 = enemy_knight_hp()
    seen_extra = False
    enemy_dmg_peak = 0.0
    try:
        for _ in range(DURATION * 60):
            bs.step(1 / 60)
            if not seen_extra and friendly_non_tower() > base_friendly:
                seen_extra = True
            enemy_dmg_peak = max(enemy_dmg_peak,
                                 p1_king_hp0 - bs.entities[6].hp,
                                 knight_hp0 - enemy_knight_hp())
            if bs.game_over:
                break
    except Exception as e:
        tb = traceback.format_exc().splitlines()
        src = tb[-3].strip()[-70:] if len(tb) >= 3 else ''
        return f'STEP {type(e).__name__}: {str(e)[:50]} @{src}'
    acted = seen_extra or enemy_dmg_peak > 0
    return ('OK' if acted else 'NO-ACT'), {'spawned_extra': seen_extra, 'enemy_dmg': round(enemy_dmg_peak, 1)}


def main():
    cards = sorted(card_data.keys())
    results = {}
    for name in cards:
        results[name] = smoke(name)

    errs = {k: v for k, v in results.items() if isinstance(v, str)}
    noact = {k: v for k, v in results.items() if isinstance(v, tuple) and v[0] == 'NO-ACT'}
    ok = {k: v for k, v in results.items() if isinstance(v, tuple) and v[0] == 'OK'}

    print(f'全 gamedata 卡冒烟：共 {len(cards)} | OK {len(ok)} | NO-ACT {len(noact)} | 异常 {len(errs)}')
    print('\n=== 异常 ===')
    for k, v in sorted(errs.items()):
        print(f'  {k:26s} {v}')
    print('\n=== NO-ACT（需人工确认是否场景假象）===')
    for k, v in sorted(noact.items()):
        print(f'  {k:26s} {v[1]}')

    report = {
        'total': len(cards),
        'ok': sorted(ok),
        'no_act': sorted(noact),
        'errors': {k: v for k, v in sorted(errs.items())},
        'detail': {k: (v[1] if isinstance(v, tuple) else {'error': v}) for k, v in results.items()},
    }
    out = os.path.join(os.path.dirname(__file__), '..', 'docs', 'batch_smoke_report.json')
    with open(out, 'w') as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print(f'\n报告已写入 {out}')
    return 1 if errs else 0


if __name__ == '__main__':
    sys.exit(main())
