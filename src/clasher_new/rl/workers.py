"""跨进程训练 worker：每个进程独立持有 env + 信念 + 规划器 + 脚本对手。

背景：战斗模拟是纯 Python（GIL 受限），单进程并行无法吃满多核。跨进程方案把
「环境推演 + 信念 + 规划」放进独立 worker 进程，主进程只做批量 GPU 推理与 PPO 更新，
从而真正并行利用多核 CPU，并把 N 个 env 的推理合并成一次前向。

协议（严格同步，每个 worker 一次处理一条消息）：
- parent -> worker ``("mask", partial_bundle)``  → worker -> parent ``("mask", mask)``
- parent -> worker ``("step", bundle)``         → worker -> parent
  ``("step", payload, reward, term, trunc, opp_played, winner)``
- parent -> worker ``("reset", spec)``          → worker -> parent ``("ready", payload)``
- parent -> worker ``None``                      → worker -> parent ``("closed", None)``

payload = (obs, plan_vec, belief_tok)，即下一个决策步的输入（plan/belief 由 worker
在其本地 battle/信念上计算，避免把整盘状态传回主进程）。

对手 spec（脚本策略序列化）：
- {"type": "none"}
- {"type": "scripted", "mode", "pool", "deck_pool", "seed"}

注意：训练对手当前均为 ScriptedPolicy；若联赛采样到学习型对手，worker 回退内置随机。

用法：主进程 ``ctx = mp.get_context("spawn"|"fork")``，然后
``ctx.Process(target=worker_main, args=(i, seed, rw, in_q, out_q))``。
（用 target 函数而非 Process 子类，是为了让子进程严格使用指定 context。）
"""

import os
import sys
import random
import multiprocessing

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from rl.env_wrapper import RLEnv
from rl.belief import BeliefInference
from rl.belief_planner import BeliefPlanner
from rl.prophet import ProphetPlanner
from rl.opponents import ScriptedPolicy


def _payload(env, belief, bp, prophet, obs, rng):
    """计算下一个决策步的 (obs, belief_tok, plan_vec)。"""
    if rng.random() < 0.3:
        plan = prophet.plan(env.get_prophet_state())
    else:
        plan = bp.plan(env.battle, belief.state(), obs)
    return obs, belief.encode(obs, None), plan.to_vector()


def _apply_opponent(env, spec):
    """按 spec 装配对手（脚本策略在 worker 本地重建并绑定 env）。"""
    if spec.get("type") == "scripted":
        pol = ScriptedPolicy(mode=spec.get("mode", "random"),
                             pool=spec.get("pool"),
                             deck_pool=spec.get("deck_pool"),
                             seed=spec.get("seed", 0))
        pol.env = env
        env.deck1_factory = pol.deck if pol.pool else None
        env.opponent = pol
    else:
        env.deck1_factory = None
        env.opponent = None


def worker_main(worker_id, seed, reward_weights, in_q, out_q, card_level=None):
    """跨进程 env 推演循环（独立进程入口，绕开 GIL 实现多核并行）。"""
    try:
        env = RLEnv(opponent=None, seed=seed, reward_weights=dict(reward_weights or {}),
                    card_level=card_level)
        belief = BeliefInference(opp_deck=env.deck1, n_particles=128, seed=seed)
        bp = BeliefPlanner()
        prophet = ProphetPlanner()
        rng = random.Random(seed)
        while True:
            msg = in_q.get()
            if msg is None or msg[0] == "close":
                out_q.put(("closed", None))
                return
            if msg[0] == "mask":
                out_q.put(("mask", env.get_action_mask(msg[1])))
            elif msg[0] == "reset":
                _apply_opponent(env, msg[1])
                obs, _ = env.reset()
                belief.reset(env.deck1)
                out_q.put(("ready", _payload(env, belief, bp, prophet, obs, rng)))
            elif msg[0] == "step":
                bundle = msg[1]
                obs, reward, term, trunc, info = env.step(bundle)
                belief.update(obs, info.get("opp_played"))
                out_q.put(("step", _payload(env, belief, bp, prophet, obs, rng),
                           float(reward), bool(term), bool(trunc),
                           info.get("opp_played"), env.battle.winner))
            else:
                raise ValueError(f"未知消息: {msg[0]}")
    except Exception as e:
        import traceback
        try:
            out_q.put(("error", "%r\n%s" % (e, traceback.format_exc())))
        except Exception:
            pass
