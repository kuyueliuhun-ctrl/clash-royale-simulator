"""AI 训练算法闭环：同刻多卡 ActionBundle + 信念推断 + 先知/跟随者 + 联赛。

对应规划文档: docs/ai_training_plan.md
引擎冻结：本包只做 RL wrapper 层，不修改 battle.py / 卡牌内容。
"""

import os
import sys

# 保证与 clasher_new 平级模块（battle/player/core/card_utils）可用扁平 import
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
