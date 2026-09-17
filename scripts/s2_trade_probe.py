# -*- coding: utf-8 -*-
"""开/关逐位对账 + 在线↔离线交叉复算（S2 §6 第 5 项）。

用法：
    python trade_probe.py <tree_dir> <off|on>

- `off`：在**该树**上跑一段确定性脚本局，打印 **REWARD 序列的 SHA-256** 与状态摘要。
  在**未打补丁的树**与**打了补丁但开关关**的树上必须**逐字相同**（= 开/关逐位对账）。
- `on`：同一条脚本，但把 `engagement_trade` 打开；打印面额摘要 + 逐窗口 `trade`
  （供与 `scripts/offline_engagement_trade.py` 在同一段录像上独立复算的结果对比）。
"""
import hashlib
import json
import os
import sys

TREE = sys.argv[1]
MODE = sys.argv[2] if len(sys.argv) > 2 else "off"
os.chdir(TREE)
sys.path.insert(0, TREE)
SCRIPTS = (r"E:\clash-royale-simulator-main\scripts" if os.name == "nt"
           else "/mnt/e/clash-royale-simulator-main/scripts")
sys.path.insert(0, SCRIPTS)

from rl.env_wrapper import RLEnv                                   # noqa: E402
from rl.action_bundle import ActionBundle, SubAction               # noqa: E402
from rl.replay import battle_snapshot                              # noqa: E402

DECK = ['Tombstone', 'Witch', 'Skeletons', 'Giant', 'Minions', 'Fireball',
        'Knight', 'Archer']

rw = {}
if MODE == "on":
    rw = {"engagement_trade": 1.0, "engagement_trade_theta": 1.0,
          "engagement_trade_t_ref": 2.0, "engagement_trade_gate": 1}

env = RLEnv(deck0=list(DECK), deck1=list(DECK), seed=12345, speed=1.0,
            decision_frames=30, dt=1 / 60, record_hidden=False,
            reward_weights=rw)
obs, _ = env.reset(seed=12345)


def _slot_of(card):
    for i in range(5):
        cid = int(obs["hand"][i])
        from rl.observation import ENTITY_NAMES
        if ENTITY_NAMES[cid] == card:
            return i + 1
    return 0


#: (帧, 卡, (x, y))：双方交替出兵，落点都在各自合法半场
PLAN = []
for k in range(10):
    PLAN += [(3 * k + 0, 'Giant', (9.0, 12.0 + 0.2 * k)),
             (3 * k + 1, 'Tombstone', (4.0, 19.0 - 0.2 * k)),
             (3 * k + 2, 'Skeletons', (10.0, 13.5))]
PLAN = sorted(PLAN)

h = hashlib.sha256()
rewards, frames, et_scores, et_cum = [], [], [], []
fired = 0
for step in range(60):
    sa = []
    while fired < len(PLAN) and PLAN[fired][0] <= step:
        _, card, pos = PLAN[fired]
        slot = _slot_of(card)
        if slot:
            sa.append(SubAction(kind="deploy", slot=slot, x=pos[0], y=pos[1]))
        fired += 1
    bundle = ActionBundle(sa) if sa else ActionBundle.noop()
    obs, reward, term, trunc, info = env.step(bundle)
    rewards.append(round(float(reward), 9))
    h.update(("%.9f;" % reward).encode())
    for eid in sorted(env.battle.entities):
        e = env.battle.entities[eid]
        h.update(("%d|%d|%.6f|%.6f|%.6f|%d;"
                  % (eid, int(e.is_alive), e.hp, e.position.x, e.position.y,
                     int(getattr(e, "target_id", -1) or -1))).encode())
    try:      # schema 5 的树带 v/shares；未打补丁的树没有这两个参数
        fr = battle_snapshot(env.battle, bundle, reward, info,
                             v=list(env._active_v),
                             shares={**env._v_share[0], **env._v_share[1]})
    except TypeError:
        fr = battle_snapshot(env.battle, bundle, reward, info)
    frames.append(fr)
    et_scores.append(float(info.get("engagement_trade", 0.0)))
    et_cum.append(float(info.get("engagement_trade_cum", 0.0)))
    if term or trunc:
        break

outdir = "/tmp/trade_probe_out"
os.makedirs(outdir, exist_ok=True)
with open(os.path.join(outdir, "frames_%s.json" % MODE), "w", encoding="utf-8") as f:
    json.dump(frames, f)
with open(os.path.join(outdir, "reward_%s.txt" % MODE), "w", encoding="utf-8") as f:
    f.write("\n".join("%.9f" % x for x in rewards))

print("MODE %s  STEPS %d" % (MODE, len(rewards)))
print("REWARD_DIGEST %s" % h.hexdigest())
print("REWARD_SUM %.9f" % sum(rewards))
_et = getattr(env, "_et", None)
if _et is not None and not env.battle.game_over:
    # 局末（或步数截断）把仍开着的窗口结掉 —— 与离线仪器的 `episode_end` 对齐
    _et.flush(env.battle, env._active_v)
    et_scores.append(_et.pop_scores())
print("ET_SCORE_SUM %.9f  ET_N %d" % (sum(et_scores), _et.n_settled if _et else 0))
print("ET_CUM_LAST %.9f" % (et_cum[-1] if et_cum else 0.0))
if _et is not None:
    tr = [round(t, 6) for t, _s, _n in _et.settled]
    print("ET_TRADES n=%d sum=%.6f  first=%s" % (len(tr), sum(tr), tr[:8]))
