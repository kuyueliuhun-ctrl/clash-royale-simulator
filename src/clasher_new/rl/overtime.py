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
  - 恰达 300s 仍平 → overtime_open=False（RL 层在触发引擎 ≥300s 塔血兜底分支前收手，
    把加时末判定统一交给 timeout_winner 的“平局”口径）；
  - 皇冠不同 → False（常规时间末已有领先者，直接按皇冠结算胜负）；
  - 终局 → False。
"""

#: 常规时间末（秒）。battle.time ≥ NORMAL_TIME_S 且双方被拆塔数相同 → 进入加时窗口。
NORMAL_TIME_S = 180.0
#: 加时硬顶（秒）。到 OVERTIME_END_S 仍未破塔 → 不再延长，由 timeout_winner 记平局。
OVERTIME_END_S = 300.0


def overtime_open(battle):
    """加时（突然死亡）窗口是否仍应继续，绕过 max_ep_steps 在常规时间末的截断。"""
    if battle is None or battle.game_over:
        return False
    p0, p1 = battle.players
    if int(p0.get_crown_count()) != int(p1.get_crown_count()):
        return False
    t = float(battle.time)
    return NORMAL_TIME_S - 1e-9 <= t < OVERTIME_END_S - 1e-9
