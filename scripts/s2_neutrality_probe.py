# -*- coding: utf-8 -*-
"""行为中立性探针：在**两棵树**上跑同一段确定性推演，打印可对账摘要。

用法：python neutrality_probe.py <tree_dir>
对账：两棵树的 DIGEST 行必须**逐字相同**（纯记录通道不得改任何行为）。
"""
import hashlib
import os
import sys

TREE = sys.argv[1]
os.chdir(TREE)                 # card_utils 用相对路径读 gamedata.json
sys.path.insert(0, TREE)

from battle import BattleState                      # noqa: E402
from player import PlayerState                      # noqa: E402
from card_utils import Card                         # noqa: E402

DECK = ['Tombstone', 'Witch', 'Skeletons', 'Giant', 'Minions', 'Fireball',
        'Knight', 'Archers']

bs = BattleState(PlayerState(0, list(DECK), 5.0), PlayerState(1, list(DECK), 5.0),
                 card_level=11)
bs.regen = 2.8

#: (tick, player, card, (x, y)) —— 固定脚本，覆盖：单位/建筑/法术/亡语/召唤物
PLAN = []
for k in range(14):
    # 出兵点必须落在各自合法半场（player0: y<15；player1: y>17），否则引擎整包拒绝
    PLAN.append((40 * k + 5, 0, 'Giant', (9.0, 12.0 + 0.2 * k)))
    PLAN.append((40 * k + 12, 1, 'Tombstone', (4.0, 19.0 - 0.2 * k)))
    PLAN.append((40 * k + 20, 0, 'Skeletons', (10.0, 13.5)))
    PLAN.append((40 * k + 28, 1, 'Witch', (13.0, 20.0)))
    PLAN.append((40 * k + 34, 0, 'Fireball', (13.0, 21.0)))
    PLAN.append((40 * k + 36, 1, 'Knight', (5.0, 19.0)))

PLAN = sorted(PLAN)
h = hashlib.sha256()
fired = 0
for tick in range(1200):
    while fired < len(PLAN) and PLAN[fired][0] <= tick:
        _, pid, card, pos = PLAN[fired]
        bs.players[pid].elixir = 10.0     # 保证出牌不被圣水挡住（两棵树一致）
        try:
            bs.deploy_card(pid, card, __import__('battle').Position(*pos))
        except Exception as e:                       # noqa: BLE001
            h.update(("ERR:%s:%s" % (card, type(e).__name__)).encode())
        fired += 1
    bs.step(1 / 30.0)
    for eid in sorted(bs.entities):
        e = bs.entities[eid]
        h.update(("%d|%d|%.6f|%.6f|%.6f|%d|%d;"
                  % (eid, int(e.is_alive), e.hp, e.position.x, e.position.y,
                     int(getattr(e, 'target_id', -1) or -1), e.player)).encode())

print("DIGEST %s" % h.hexdigest())
print("TIME %.6f  ENTITIES %d  WINNER %s  GAMEOVER %s"
      % (bs.time, len(bs.entities), bs.winner, bs.game_over))
print("ELIXIR %.6f %.6f" % (bs.players[0].elixir, bs.players[1].elixir))

# —— 记录通道读数（未打补丁的树会打印 None）——
ents = list(bs.entities.values())
n_prod = sum(1 for e in ents if getattr(e, 'is_product', None) is True)
n_non = sum(1 for e in ents if getattr(e, 'is_product', None) is False)
n_unk = sum(1 for e in ents if getattr(e, 'is_product', None) is None)
n_root = sum(1 for e in ents if getattr(e, 'root_cast', None) is not None)
print("RECORD is_product True=%d False=%d None=%d | root_cast set=%d/%d | "
      "next_cast_id=%s | cast_log=%s"
      % (n_prod, n_non, n_unk, n_root, len(ents),
         getattr(bs, 'next_cast_id', None),
         len(getattr(bs, 'cast_log', {}) or {})))
cl = getattr(bs, 'cast_log', None)
if cl:
    sample = sorted(cl.items())[:4]
    print("CASTS %s ..." % [(k, v['player'], v['card']) for k, v in sample])
