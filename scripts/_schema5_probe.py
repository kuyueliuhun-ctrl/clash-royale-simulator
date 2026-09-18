# -*- coding: utf-8 -*-
"""sandbox 验证：schema 5（root_cast / is_product / share + 帧内 v0/v1）是否端到端可用。

用法：python schema5_probe.py <tree_dir>
"""
import os
import sys

TREE = sys.argv[1]
os.chdir(TREE)
sys.path.insert(0, TREE)
# T1-2：本文件就在 `scripts/` 下，直接由 `__file__` 推导 ⇒ 不再硬编码 WSL/Windows 两套绝对路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import battle                                       # noqa: E402
from player import PlayerState                      # noqa: E402
from rl.replay import battle_snapshot, LEAGUE_REPLAY_SCHEMA  # noqa: E402
import offline_engagement_trade as O                # noqa: E402

DECK = ['Tombstone', 'Witch', 'Skeletons', 'Giant', 'Minions', 'Fireball',
        'Knight', 'Archers']
bs = battle.BattleState(PlayerState(0, list(DECK), 5.0), PlayerState(1, list(DECK), 5.0),
                        card_level=11)


class _Bundle:
    sub_actions = []


#: 简易"环境影子"：只提供 `_active_v` / `_v_share`（不在真 env 上跑，避免 GPU 依赖）
class _EnvShadow:
    def __init__(self, b):
        self.battle = b
        self._v_share = {0: {}, 1: {}}
        self._active_v = [0.0, 0.0]


env = _EnvShadow(bs)
frames = []
PLAN = []
for k in range(3):
    PLAN += [(40 * k + 5, 0, 'Giant', (9.0, 12.0)), (40 * k + 12, 1, 'Tombstone', (4.0, 19.0)),
             (40 * k + 20, 0, 'Skeletons', (10.0, 13.5)), (40 * k + 28, 1, 'Witch', (13.0, 20.0))]
PLAN.sort()
fired = 0
for tick in range(140):
    while fired < len(PLAN) and PLAN[fired][0] <= tick:
        _, pid, card, pos = PLAN[fired]
        bs.players[pid].elixir = 10.0
        before = bs.players[pid].elixir
        ok = bs.deploy_card(pid, card, battle.Position(*pos))
        if ok:   # 影子记账：与 RLEnv._deploy_ledger 同规则（只用于让 v/share 非空）
            cost = before - bs.players[pid].elixir
            new = [e.id for e in bs.entities.values()
                   if isinstance(e, (battle.Troop, battle.Building)) and e.id > 6
                   and e.id not in env._v_share[pid]]
            if new and cost > 0:
                for eid in new:
                    env._v_share[pid][eid] = cost / len(new)
                    env._active_v[pid] += cost / len(new)
        fired += 1
    bs.step(1 / 30.0)
    # 死亡注销
    for pid in (0, 1):
        for eid in [e for e in env._v_share[pid] if e not in bs.entities]:
            env._active_v[pid] -= env._v_share[pid].pop(eid)
    fr = battle_snapshot(bs, _Bundle(), 0.0, {}, v=env._active_v,
                         shares={**env._v_share[0], **env._v_share[1]})
    frames.append(fr)

print("SCHEMA %s" % LEAGUE_REPLAY_SCHEMA)
e0 = frames[60]["entities"]
print("FRAME keys has v0/v1: %s %s" % ("v0" in frames[60], "v1" in frames[60]))
print("ENTITY arity: %s" % sorted({len(e) for f in frames for e in f["entities"]}))
prod = [e for f in frames for e in f["entities"] if len(e) > 13 and e[13] is True]
print("is_product True 条目数 = %d（样例 %s）"
      % (len(prod), [(e[0], e[12], e[13], e[14]) for e in prod[:3]]))
tab = O.entity_table(frames[60])
sh = [d for d in tab.values() if d.get("share")]
print("share 非零实体 = %d（样例 %s）"
      % (len(sh), [(d["name"], d["share"], d["is_product"]) for d in sh[:4]]))
phi, diag = O.phi_series(frames)
print("phi_series exact=%s  v_source=%s" % (diag.get("exact"), diag.get("v_source")))
print("重建 vs 帧内真值 max 误差 = %s"
      % diag.get("reconstruct_vs_frame_v_max_err"))
print("cost_ring 条目数 = %d" % len(diag.get("cost_ring") or {}))
assert LEAGUE_REPLAY_SCHEMA == 5
assert "v0" in frames[60] and frames[60]["v0"] is not None
assert all(len(e) == 15 for f in frames for e in f["entities"])
assert diag.get("exact") is True
assert prod, "应当有产物（Tombstone→Skeleton / Witch→Skeleton）"
print("SANDBOX SCHEMA5 PROBE: PASS")
