"""加时（突然死亡）窗口判定：纯逻辑、零重依赖（不 import torch）。

背景：引擎 BattleState 自带 [180, 300) 加时规则——常规时间末（180s）双方被拆塔数
相同 → 加时内谁先被再破一塔谁输（引擎在 [180,300) 一旦出现皇冠差立即终局）。
但 RL 各循环以 max_ep_steps 截断：默认 360 步 = battle.time≈180s，皇冠平的对局在
进入引擎加时前就被截断，并用 timeout_winner 直接按塔血/平局结算 —— 于是出现
“180s 双方失去相同防御塔 → 游戏直接结束”而不是进入突然死亡。

本模块只负责回答“现在是否处于应继续的加时窗口”，供 run_league / train_solo /
flow_league / evaluate / workers 各循环统一消费（放进轻量模块，避免 mp worker
为判个窗口被迫 import torch 重栈）。

规则（用户确认，2026-09）：
  - battle.time ∈ [180, 300) 且双方被拆塔数相同、对局未终局 → overtime_open=True
    （各 RL 循环绕过 max_ep_steps 截断继续打；引擎会先破塔立即终局）；
  - 恰达 300s 仍平 → overtime_open=False（RL 层收手）。终局裁决统一走
    timeout_winner：皇冠平 → 双方存活塔中血量百分比更低者输（真实 CR 加时末
    规则，与引擎 300s 硬顶分支同口径），完全相等才记平局；
  - 皇冠不同 → False（常规时间末已有领先者，直接按皇冠结算胜负）；
  - 终局 → False。
"""

#: 常规时间末（秒）。battle.time ≥ NORMAL_TIME_S 且双方被拆塔数相同 → 进入加时窗口。
NORMAL_TIME_S = 180.0
#: 加时硬顶（秒）。到 OVERTIME_END_S 仍未破塔 → 不再延长，按 timeout_winner 的
#: 最低塔血百分比裁决终局（完全相等才平局）。
OVERTIME_END_S = 300.0


#: 双方三塔实体 id（与 `battle.BattleState.TOWER_IDS` 同约定；mock 战场复算用）
_TOWER_IDS = {0: (3, 4, 6), 1: (1, 2, 5)}


def tower_hp_total(battle, player_id):
    """三塔血量合计：优先用引擎单源方法；mock 战场按同 id 约定复算；无实体信息 → None。"""
    fn = getattr(battle, "tower_hp_total", None)
    if callable(fn):
        return float(fn(player_id))
    ents = getattr(battle, "entities", None)
    if not ents:
        return None
    tot = 0.0
    for i in _TOWER_IDS[int(player_id)]:
        e = ents.get(i)
        if e is None:
            continue
        if getattr(e, "is_alive", False):
            tot += float(getattr(e, "hp", 0.0))
    return tot


def timeout_winner(battle):
    """截断/早停/到点时的结算：皇冠多者胜；皇冠平 → **塔血合计多者胜**；完全相等 → None。

    2026-09-17 口径变更（用户指定）：塔血裁决从"存活塔最低血量百分比"改为
    "三塔血量**合计**"，且**只在合计完全相等时判平局**。单一来源 = 引擎
    `BattleState.timeout_winner()`（本函数在真实 battle 上直接委托它）。
    规则与取舍：`docs/draw_rule_prereg_2026-09-17.md`。
    """
    if battle is None:
        return None
    p0, p1 = battle.players
    lost0 = int(p0.get_crown_count())
    lost1 = int(p1.get_crown_count())
    if lost1 > lost0:
        return 0
    if lost0 > lost1:
        return 1
    fn = getattr(battle, "timeout_winner", None)
    if callable(fn):
        return fn()                      # 真实引擎：单一来源
    h0 = tower_hp_total(battle, 0)
    h1 = tower_hp_total(battle, 1)
    if h0 is None or h1 is None:
        return None                      # mock 无实体信息 → 平局（旧行为）
    if h0 > h1 + 1e-9:
        return 0
    if h1 > h0 + 1e-9:
        return 1
    return None


def overtime_open(battle):
    """加时（突然死亡）窗口是否仍应继续，绕过 max_ep_steps 在常规时间末的截断。"""
    if battle is None or battle.game_over:
        return False
    p0, p1 = battle.players
    if int(p0.get_crown_count()) != int(p1.get_crown_count()):
        return False
    t = float(battle.time)
    return NORMAL_TIME_S - 1e-9 <= t < OVERTIME_END_S - 1e-9
