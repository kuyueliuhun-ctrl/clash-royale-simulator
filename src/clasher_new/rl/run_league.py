"""联赛循环（规划文档 7.4 / 7.5）+ 训练网页 UI 数据源。

两个入口：
- ``evaluate_league``（--mode eval）：保存策略轮转对战（换边 + 三态 + 逐局 Elo）；
- ``run_league``（--mode run）：**同时维护 5 个模型** 的联赛主循环——PPO 训练 main →
  PFSP 采对手 → 周期全轮转评估（含新模型 random_deck）→ 逐局 Elo → 快照刷新 →
  Elo 历史曲线 → 状态持久化（供训练网页 UI 读取）。

其它模式：
- ``--mode flow`` / ``flow-sweep-*``：全配对分流派联赛 / 数据效率 A/B（rl/flow_league.py）；
- ``--mode solo``：**单人自对弈**（原版 train.py 思路，固定卡组镜像 + 周期冻结副本，
  无联赛机制，写 solo_state.json 供 dashboard --solo；rl/train_solo.py）。

新增能力（对应需求）：
- ``rl/config.py`` 命名配置：每组参数（超参 + **奖惩机制奖励权重**）命名后
  独立输出到 ``out_dir/<name>/``（断点 / 检查点 / 联赛录像 / Elo 状态 / config.json）；
- ``--resume`` 断点续训：从 run_state.json 恢复 step、main 权重与 Adam 状态
  （8a8dbbc+ 起**默认自动续训**：不加参数 = 有断点就接着训；显式 ``--fresh`` 才从头）；
- ``--device cpu|cuda|auto``：CUDA（cu130）训练支持；
- ``--n-envs N`` 并行多环境 + batch 推理/更新（N>1 用 FollowerPolicy.act_parallel /
  evaluate_batch，单进程内批量化 GPU；注意战斗模拟为纯 Python（GIL），真正的多核
  加速需要跨进程 worker，见 README）；
- 训练开始先跑一次评估/快照（``--no-eval-start`` 关闭），WebUI 立即有真实数据；
- 每个评估周期（默认 2000 步）保存**联赛录像**到 ``replays/league_<step>.pkl``。

5 个模型槽位：
1. ``main``         跟随者 PPO（训练中）
2. ``random_deck``  **卡组完全随机**模型（每局重采样 8 张卡，随机合法出牌，新）
3. ``heuristic``    脚本启发式
4. ``random``       随机固定卡组
5. ``all_decks``    三分类全 200 卡组模型（有卡组数据时）

修复沿用：P1-11 逐局 Elo、P1-14 换边 + 三态、P1-9 快照权重副本、P1-10 持久化。
"""

import os
import sys
import json
import time
import queue
import argparse
import itertools
import random

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import numpy as np
import torch

from rl.env_wrapper import RLEnv
from rl.belief import BeliefInference
from rl.belief_planner import BeliefPlanner
from rl.follower import FollowerPolicy, load_checkpoint, save_checkpoint
from rl.plan_space import PLAN_DIM
from rl.league import League
from rl.train_follower import FollowerOpponent
from rl.ppo import PPOTrainer
from rl.prophet import ProphetPlanner
from rl.action_bundle import ActionBundle, K_MAX
from rl.observation import ENTITY_NAMES
from rl.opponents import ScriptedPolicy, build_card_pool
from rl.decks import load_classified_decks, decks_by_archetype, classify_stats
from rl.config import TrainConfig, reward_to_env
from rl.overtime import (NORMAL_TIME_S, OVERTIME_END_S, overtime_open,
                         timeout_winner as _overtime_timeout_winner)
from rl.replay import battle_snapshot, save_league_replays


def _cuda_hint() -> str:
    """CUDA 不可用时的诊断提示（帮助区分 CPU 构建 / 驱动问题）。"""
    ver = getattr(torch, "__version__", "?")
    cuda_ver = torch.version.cuda
    lines = [
        "[device] CUDA 不可用，回退 cpu。诊断：",
        f"  torch={ver}  torch.version.cuda={cuda_ver}  "
        f"cuda.is_available()={torch.cuda.is_available()}",
    ]
    if not cuda_ver:
        lines += [
            "  当前 torch 是 CPU 构建（+cpu）。pip 装了 CPU 版后再用 cu130 索引装会被"
            "当成'已满足'跳过，不会切换构建——必须强制换装：",
            "    pip uninstall -y torch",
            "    pip install torch --index-url https://download.pytorch.org/whl/cu130",
            "  （或一条命令：pip install --force-reinstall torch "
            "--index-url https://download.pytorch.org/whl/cu130）",
        ]
    else:
        lines.append("  torch 带 CUDA 但仍不可用：请检查 NVIDIA 驱动是否支持 CUDA 13.x"
                     "（nvidia-smi 查看驱动版本，需较新的驱动）")
    return "\n".join(lines)


def resolve_device(device: str) -> str:
    """把 --device 解析成实际设备（auto = cuda 可用则 cuda）。"""
    if device == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    elif device not in ("cpu", "cuda"):
        raise ValueError(f"device 必须是 cpu/cuda/auto，收到 {device}")
    else:
        dev = device
    if dev == "cuda" and not torch.cuda.is_available():
        print(_cuda_hint(), flush=True)
        return "cpu"
    return dev


class LeagueGameRecorder:
    """逐局录像采集器：把每个决策步压缩成轻量帧，供每 2000 步联赛录像。"""

    def __init__(self, a_id, b_id, side0, max_steps, steps=None, decks=None):
        self.meta = {"pair": [a_id, b_id], "side0": side0, "max_steps": max_steps}
        # steps = [a_step, b_step]：双方模型各自所在训练步（None = 无步数概念，
        # 如脚本对手）。dashboard 对局列表据此显示 "main@2000 vs main@0" 这类对阵。
        if steps is not None:
            self.meta["steps"] = [int(s) if s is not None else None for s in steps]
        # decks = (deck0, deck1)：双方本局实际卡组（卡名列表）。dashboard 的卡牌使用
        # 统计据此还原"这一局双方各带了什么"，也是卡组构成统计的数据源。
        if decks is not None:
            self.set_decks(decks[0], decks[1])
        self.frames = []
        self.winner = None

    def set_decks(self, deck0, deck1):
        """记录本局双方实际卡组（play_pair 复用 env 时在 reset 之后才可知）。"""
        self.meta["decks"] = [list(deck0), list(deck1)]

    def record(self, env, bundle, reward, info, cards=None):
        frame = battle_snapshot(env.battle, bundle, reward, info)
        # cards = 本步我方（player-0）实际打出的卡名。opp_played 已含对手卡名，本字段
        # 补齐我方一侧 → dashboard 卡牌使用统计可做双侧完整。旧录像无此字段。
        if cards:
            frame["cards"] = list(cards)
        self.frames.append(frame)

    def done(self, winner):
        self.winner = winner
        return {"meta": self.meta, "winner": winner, "frames": self.frames}


#: 评估早停：连续 STALL_LIMIT 次检查（每 STALL_WINDOW 步一次）双方塔血合计零变化
#: → 判僵局为平局、提前结束对局，省掉拖满 max_ep_steps 的无效模拟（费差 shaping 下
#: 双方都龟缩的对局占比不小，单局可从 ~23s 降到 ~4s）。CR 无塔治疗，塔血只降不升，
#: 长时间零塔损是可靠僵局信号。
STALL_WINDOW = 10
STALL_LIMIT = 10   # 10×10=100 步无塔损判平


def towers_hp(env):
    """双方三塔血量合计（僵局检测用；塔血只降不升）。"""
    p0 = env.battle.players[0]
    p1 = env.battle.players[1]
    return (p0.king_tower_hp + p0.left_tower_hp + p0.right_tower_hp
            + p1.king_tower_hp + p1.left_tower_hp + p1.right_tower_hp)


def _min_alive_tower_pct(battle, player_id):
    """该玩家存活塔中最低的血量百分比（真实 CR 加时末裁决口径）。

    实体不可得（纯 mock/假 battle，如 selftest）返回 None，调用方退回平局。
    """
    ents = getattr(battle, 'entities', None)
    if not ents:
        return None
    best = None
    for eid in ((3, 4, 6) if player_id == 0 else (1, 2, 5)):
        e = ents.get(eid)
        if e is None or not getattr(e, 'is_alive', False):
            continue
        max_hp = float(getattr(getattr(e, 'data', None), 'hp', 0) or 0)
        if max_hp <= 0:
            return None
        pct = float(e.hp) / max_hp
        if best is None or pct < best:
            best = pct
    return best


def timeout_winner(battle, hp_tiebreak=None):
    """截断/早停时的到期结算兜底（不动引擎，只在 episode 提前结束时补判）。

    规则（2026-09 定稿）：
      1) 皇冠多者胜：皇冠 = 对方被拆塔数（players[X].get_crown_count()
         是 X 侧被拆塔数 = 对方得分）；
      2) 皇冠相同 → **三塔血量合计**多者胜（2026-09-17 用户指定口径；
         旧口径是"存活塔最低血量百分比"，见 docs/draw_rule_prereg_2026-09-17.md）；
         完全相等 → None（平局）。
         僵局早停/截断等价于把终局提前到这里，不能一律记平局——否则出现
         "1-1、双方塔血差 1000+ HP 却记 D" 的错误平局（回放实证：
         economy league_6000 局3/局11 等）。mock 战场无实体信息时退回平局。

    hp_tiebreak: 兼容旧签名保留，不再作为开关（塔血裁决总是启用，见规则 2）。
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
    return _overtime_timeout_winner(battle)      # 单一来源（引擎 BattleState.timeout_winner）


def settle_stall_from_counts(lost0, lost1, min_pct0, min_pct1, margin=0.05):
    """C'（2026-09-12）早停局低置信裁定降噪（纯函数，可单测）。

    与 timeout_winner 同口径，但给"皇冠相同"的塔血%细差加置信门槛：
    - 皇冠不同 → 决定性，照常返回 0/1；
    - 皇冠相同且塔血%差 ≥ margin → 决定性，返回 0/1；
    - 皇冠相同且塔血%差 < margin（掷硬币级裁定）→ None（记平局，调用方按
      平局=失败惩罚），**不再按细差判胜负**。

    依据：docs/critic_probe_experiment_2026-09-12.md 实验 3 —— 早停局占比
    28~40%，其中皇冠相同的塔血%细差裁定标签噪声最大（早停把未定局的胜负
    提前裁定，细差方向近乎随机）。仅用于**训练侧**结算降噪；eval 仍用
    timeout_winner（真实 CR 规则），保证评估口径与历史可对比。
    """
    if lost1 > lost0:
        return 0
    if lost0 > lost1:
        return 1
    if min_pct0 is None or min_pct1 is None:
        return None
    if min_pct0 > min_pct1 + margin:
        return 0
    if min_pct1 > min_pct0 + margin:
        return 1
    return None


def settle_stall(battle, margin=0.0):
    """早停局结算。**默认（margin=0）走 2026-09-17 新口径**：皇冠 → 三塔血量**合计**，
    完全相等才平局（= timeout_winner 同一规则）。

    `margin > 0` 时走 2026-09-12 的 C′ 旧路径（存活塔最低血量百分比 + 边距，
    "低置信细差记平局"）——保留它是为了**逐位复现旧标签**（`--stall-draw-margin 0.05`）。
    为什么改：正常防守 ⇒ 100 步无塔损 ⇒ 僵局早停 ⇒ 细差 < margin ⇒ 记 D（且 D 按失败罚），
    即"正常防守被判为平局"。口径与判据见 docs/draw_rule_prereg_2026-09-17.md。
    """
    if battle is None:
        return None
    if margin and float(margin) > 0.0:
        p0, p1 = battle.players
        return settle_stall_from_counts(int(p0.get_crown_count()), int(p1.get_crown_count()),
                                        _min_alive_tower_pct(battle, 0),
                                        _min_alive_tower_pct(battle, 1), float(margin))
    return timeout_winner(battle)


def _stall_probe(env, last_hp, stall_count):
    """僵局探针：每 STALL_WINDOW 步调用一次。

    返回 (early_stop, last_hp, stall_count)。连续 STALL_LIMIT 次零塔血变化 → early_stop。
    """
    hp = towers_hp(env)
    if last_hp is None:
        return False, hp, 0
    if abs(hp - last_hp) < 1e-9:
        stall_count += 1
        if stall_count >= STALL_LIMIT:
            return True, hp, stall_count
        return False, hp, stall_count
    return False, hp, 0


def _run_side0(env, policy, belief, bp, max_steps=300, recorder=None, reset_seed=None,
               stall_stop=False):
    """policy 以 player-0 身份打完整对局；返回 winner（0/1/None=平）。

    支持 FollowerPolicy（完整信念/plan 链路）与 ScriptedPolicy（随机合法出牌）。
    recorder: LeagueGameRecorder 可选，逐帧记录联赛录像。
    reset_seed: 传入时 env.reset(seed=reset_seed)（play_pair 复用 env 时保持逐局种子）。
    僵局早停（**2026-09-17 起默认关闭**，见 `TrainConfig.train_stall_stop`）：连续 100 步双方
    塔血零变化 → 判平（STALL_WINDOW/STALL_LIMIT）；`stall_stop=True` 可复现旧行为。
    """
    if isinstance(policy, ScriptedPolicy):
        # ⚠️ 转发必须带上 stall_stop，否则脚本对手路径会静默丢掉开关（2026-09-17 踩过）
        return _run_side0_scripted(env, policy, max_steps, recorder,
                                   reset_seed=reset_seed, stall_stop=stall_stop)
    obs, _ = env.reset() if reset_seed is None else env.reset(seed=reset_seed)
    belief.reset(env.deck1)
    if recorder is not None:
        recorder.set_decks(env.deck0, env.deck1)   # reset 后才会重采样出本局实际卡组
    hidden = None
    done = False
    steps = 0
    stall_count = 0
    last_hp = None
    opp_side = env.opponent if isinstance(env.opponent, FollowerOpponent) else None
    while not done and (steps < max_steps or overtime_open(env.battle)):
        if stall_stop and steps % STALL_WINDOW == 0:
            early, last_hp, stall_count = _stall_probe(env, last_hp, stall_count)
            if early:
                break   # 僵局判平，提前结束（默认关闭：见 config.train_stall_stop）
        plan = bp.plan(env.battle, belief.state(), obs)
        tok = belief.encode(obs, None)
        bundle, _, _, hidden, _ = policy.act(obs, tok, plan.to_vector(),
                                             env.get_action_mask, hidden=hidden, deterministic=True)
        agent_played = _bundle_cards(bundle, obs)
        obs, reward, term, trunc, info = env.step(bundle)
        if recorder is not None:
            recorder.record(env, bundle, reward, info, cards=agent_played)
        if opp_side is not None:
            opp_side.observe_opponent_played(agent_played)
        belief.update(obs, info.get("opp_played"))
        done = term or trunc
        steps += 1
    w = env.battle.winner
    if w is None and not env.battle.game_over:
        # 僵局早停/步数截断早于引擎结算：皇冠差已定胜负，平才记平局
        w = timeout_winner(env.battle)
    return w


def _run_side0_scripted(env, policy, max_steps=300, recorder=None, reset_seed=None,
                        stall_stop=False):
    obs, _ = env.reset() if reset_seed is None else env.reset(seed=reset_seed)
    if recorder is not None:
        recorder.set_decks(env.deck0, env.deck1)   # reset 后才会重采样出本局实际卡组
    done = False
    steps = 0
    stall_count = 0
    last_hp = None
    opp_side = env.opponent if isinstance(env.opponent, FollowerOpponent) else None
    while not done and (steps < max_steps or overtime_open(env.battle)):
        if stall_stop and steps % STALL_WINDOW == 0:
            early, last_hp, stall_count = _stall_probe(env, last_hp, stall_count)
            if early:
                break   # 僵局判平，提前结束（默认关闭）
        bundle = policy.play(env, 0)
        agent_played = _bundle_cards(bundle, obs)
        obs, reward, term, trunc, info = env.step(bundle)
        if recorder is not None:
            recorder.record(env, bundle, reward, info, cards=agent_played)
        if opp_side is not None:
            opp_side.observe_opponent_played(agent_played)
        done = term or trunc
        steps += 1
    w = env.battle.winner
    if w is None and not env.battle.game_over:
        # 僵局早停/步数截断早于引擎结算：皇冠差已定胜负，平才记平局
        w = timeout_winner(env.battle)
    return w


def _bundle_cards(bundle, obs):
    out = []
    for sa in bundle.sub_actions:
        if sa.kind == "deploy" and 1 <= sa.slot <= K_MAX:
            cid = int(obs["hand"][sa.slot - 1])
            if 0 <= cid < len(ENTITY_NAMES):
                out.append(ENTITY_NAMES[cid])
    return out


def _deck_factory_of(policy):
    """脚本策略的"每局换卡组"工厂：pool（随机 8 张）或 deck_pool（整套抽取）→ deck()；
    其余（含普通策略）→ None = 用固定卡组。

    历史 bug（2026-09-11 由 dashboard 卡牌使用统计取证暴露）：旧写法只判 ``policy.pool``，
    漏掉 deck_pool → push/counter/lockdown/all_decks 四个三分类卡组模型实际一直打
    DEFAULT_DECK 的固定 8 卡，200 副天梯卡组从未生效。
    """
    if isinstance(policy, ScriptedPolicy) and (policy.pool or policy.deck_pool):
        return policy.deck
    return None


def _make_opp(policy, env, deck):
    """把 policy 包成 player-1 对手；None = 内置随机（固定卡组）。"""
    if policy is None:
        env.deck1_factory = None
        return None
    if isinstance(policy, ScriptedPolicy):
        policy.env = env
        env.deck1_factory = _deck_factory_of(policy)
        return policy
    env.deck1_factory = None    # 非脚本对手：显式清空，避免上一对的卡组工厂残留
    return FollowerOpponent(policy, env,
                            belief=BeliefInference(opp_deck=list(deck), n_particles=128, seed=0))


def _prepare_env(env, side0_pol, side1_pol, deck0_prior=None):
    """按双方策略类型装配 env（随机卡组工厂 + 对手），在 reset 前调用。

    deck0_prior: FollowerOpponent 信念先验卡组（缺省 env.deck0；play_pair 复用 env 时
    传构造时的初始卡组快照，与旧"每局新建 env"语义一致）。
    """
    # 显式赋值：非脚本策略 → None（否则会沿用上一对留下的卡组工厂，卡组跨 pair 泄漏）
    env.deck0_factory = _deck_factory_of(side0_pol)
    env.opponent = _make_opp(side1_pol, env,
                             deck0_prior if deck0_prior is not None else env.deck0)
    return env


def _pair_seed_offset(idx, a, b):
    """同一评估周期内不同 pair 的独立种子偏移。

    历史教训：eval_round_robin 曾对每对都用 ``seed + step``，导致所有 pair 的
    逐局种子（seed+g / seed+g+5000）完全相同——若某方弱到每局结果同构
    （如 main 对 5 个对手 4 连败），各 pair 的 PFSP 胜率流就打出完全相同序列、
    收敛到同一值（曾见 5 个 0.40725312499999994 = 0.5×0.95⁴）。加 pair 专属偏移，
    让不同 pair 用不同的 RNG 流，评估采样真正互相独立。
    """
    import hashlib
    return int(hashlib.sha1(f"{idx}|{a}|{b}".encode("utf-8")).hexdigest()[:7], 16)


def play_pair(league, a_id, a_pol, b_id, b_pol, n_games, max_steps, seed, record=False):
    """a vs b 换边 n 局（a 先手 n/2 + b 先手 n/2），逐局更新 Elo/PFSP（P1-11/P1-14）。

    返回 (wins_a, wins_b, draws, replays)。record=True 时采集逐局联赛录像。
    """
    wins_a = wins_b = draws = 0
    replays = []
    # 复用同一个 env：每局 reset(seed=...) 换对局，省去逐局 RLEnv/BattleState 重建
    # （评估每轮数百局，重建固定成本累加起来可观）。信念先验固定用构造时的初始卡组
    # 快照，与旧"每局新建 env（reset 前 deck 仍为默认卡组）"语义一致。
    env = RLEnv(opponent=None, seed=seed)
    deck0_prior = list(env.deck0)
    belief_prior = list(env.deck1)
    for g in range(n_games):
        if g % 2 == 0:
            _prepare_env(env, a_pol, b_pol, deck0_prior)
            belief = BeliefInference(opp_deck=belief_prior, n_particles=128, seed=seed + g)
            rec = LeagueGameRecorder(a_id, b_id, a_id, max_steps) if record else None
            w = _run_side0(env, a_pol, belief, BeliefPlanner(), max_steps, rec,
                           reset_seed=seed + g)
            if rec is not None:
                replays.append(rec.done(w))
            score_a = 1.0 if w == 0 else (0.5 if w is None else 0.0)
        else:
            _prepare_env(env, b_pol, a_pol, deck0_prior)
            belief = BeliefInference(opp_deck=belief_prior, n_particles=128, seed=seed + g + 5000)
            rec = LeagueGameRecorder(a_id, b_id, b_id, max_steps) if record else None
            w = _run_side0(env, b_pol, belief, BeliefPlanner(), max_steps, rec,
                           reset_seed=seed + g + 5000)
            if rec is not None:
                replays.append(rec.done(w))
            score_a = 0.0 if w == 0 else (0.5 if w is None else 1.0)
        league.record_match(a_id, b_id, score_a)
        if score_a == 1.0:
            wins_a += 1
        elif score_a == 0.0:
            wins_b += 1
        else:
            draws += 1
    return wins_a, wins_b, draws, replays


def _round_estimates(pair_results):
    """轮内聚合 Elo 估计（BT-lite + Laplace 平滑）→ 曲线可信度上限。

    逐局 K=32 运行 Elo 是**有限记忆跟踪器**：单轮噪声 1σ≈±40 即饱和（N≥20 不再下降），
    所以"加评估局数"不会收紧运行 Elo 曲线。要真正降噪必须做**轮内聚合**：
      D̂_ab = 400·log10((w+0.5)/(n−w+0.5))，n=该对局数，w=胜+0.5·平（Laplace 防 ±∞）
      est[a] = 1500 + mean_pairs(D̂)，SE[a] ≈ 347.5/√(games_a)（p=0.5 最坏情形，已 MC 验证）
    返回 (est, games)：est: {aid:[R, SE]}，games: {aid: 本轮总对局数}。
    """
    import math
    acc, games = {}, {}
    for a, b, wa, wb, dr in pair_results:
        n = int(wa) + int(wb) + int(dr)
        if n <= 0:
            continue
        w = float(wa) + 0.5 * float(dr)
        d = 400.0 * math.log10((w + 0.5) / (n - w + 0.5))
        acc.setdefault(a, []).append(d)
        acc.setdefault(b, []).append(-d)
        games[a] = games.get(a, 0) + n
        games[b] = games.get(b, 0) + n
    est = {}
    for aid, ds in acc.items():
        est[aid] = [round(1500.0 + sum(ds) / len(ds), 1),
                    round(347.5 / math.sqrt(games[aid]), 1)]
    return est, games


def eval_round_robin(league, n_games, max_steps, seed, step, only_vs_main=False, record=False):
    """全轮转评估：所有有策略的 agent 两两换边对战，逐局 Elo，并记录历史曲线。

    返回采集到的联赛录像列表（record=True 时非空）。
    """
    ids = [aid for aid, ag in league.agents.items() if ag.policy is not None]
    pairs = list(itertools.combinations(ids, 2))
    if only_vs_main:
        pairs = [p for p in pairs if "main" in p]
    replays = []
    pair_results = []
    for idx, (a, b) in enumerate(pairs):
        pair_seed = seed + step + _pair_seed_offset(idx, a, b)
        wins_a, wins_b, draws, rs = play_pair(league, a, league.agents[a].policy,
                                              b, league.agents[b].policy,
                                              n_games, max_steps, pair_seed, record=record)
        replays.extend(rs)
        pair_results.append((a, b, wins_a, wins_b, draws))
        print(f"[eval@{step}] {a} vs {b}: {wins_a}W {wins_b}L {draws}D", flush=True)
    # 轮内聚合估计 + 噪声地板（曲线可信度上限；SE≈347.5/√N 随局数下降）
    est, games = _round_estimates(pair_results)
    league.record_round_stats(step, {"est": est, "games": games})
    for aid in sorted(est):
        r, se = est[aid]
        print(f"[eval@{step}] {aid}: 轮内估计 {r:.0f}±{se:.0f}（本轮 {games[aid]} 局，"
              f"1σ 噪声地板）", flush=True)
    league.record_elo_history(step)
    return replays


# ---------------------------------------------------------------------------
# 模式一：轮转评估
# ---------------------------------------------------------------------------

def evaluate_league(policies, kinds, n_games, seed, hidden_dim, max_steps=600, device="auto"):
    league = League(seed=seed)
    pols = {}
    for i, path in enumerate(policies):
        kind = (kinds[i] if kinds and i < len(kinds)
                else ("main" if i == 0 else "baseline"))
        pols[path] = load_checkpoint(path, hidden_dim=hidden_dim)
        pols[path].to_device(resolve_device(device))
        league.add_agent(path, kind=kind, policy=pols[path])
    pair_results = []
    for idx, (a, b) in enumerate(itertools.combinations(policies, 2)):
        wa, wb, dr, _ = play_pair(league, a, pols[a], b, pols[b], n_games, max_steps,
                                  seed + _pair_seed_offset(idx, a, b))
        pair_results.append((a, b, wa, wb, dr))
        print(f"{os.path.basename(a)} vs {os.path.basename(b)}: "
              f"{wa}W {wb}L {dr}D / {n_games}", flush=True)
    est, games = _round_estimates(pair_results)
    league.record_round_stats(0, {"est": est, "games": games})
    league.record_elo_history(0)
    print("=== League Elo ===")
    for aid, r in league.elo_table().items():
        print(f"{os.path.basename(aid):24s} Elo={r:.1f}")
    return league


# ---------------------------------------------------------------------------
# 模式二：联赛主循环（同时维护 5 个模型）
# ---------------------------------------------------------------------------

def build_five_agents(league, main, seed, decks_path=None):
    """注册 5 个卡组模型：三分类×3 + 全 200 卡组 + 全随机（main 为训练目标）。

    有卡组的模型每局从对应卡组集合里**随机抽一副**完整 8 卡卡组。
    找不到三分类数据集时退回旧的随机/启发式 5 槽位。
    """
    pool = build_card_pool()
    decks = None
    try:
        decks = load_classified_decks(decks_path)
    except FileNotFoundError as e:
        print(f"[league] 未找到三分类卡组，退回旧 5 模型: {e}")
    if decks:
        by_arch = decks_by_archetype(decks)
        agents5 = {
            "push_flow": ScriptedPolicy(mode="random", deck_pool=by_arch["推进流"], seed=seed + 10),
            "counter_flow": ScriptedPolicy(mode="random", deck_pool=by_arch["防守反击流"], seed=seed + 20),
            "lockdown_flow": ScriptedPolicy(mode="random", deck_pool=by_arch["自闭流"], seed=seed + 30),
            "all_decks": ScriptedPolicy(mode="random", deck_pool=decks, seed=seed + 40),
            "random_deck": ScriptedPolicy(mode="random", pool=pool, seed=seed + 50),
        }
        counts, missing = classify_stats(decks)
        print(f"[league] 三分类卡组已接入: {dict(counts)} 平均补位="
              f"{ {k: round(v, 2) for k, v in missing.items()} }", flush=True)
    else:
        agents5 = {
            "random_deck": ScriptedPolicy(mode="random", pool=pool, seed=seed + 10),
            "heuristic": ScriptedPolicy(mode="heuristic", seed=seed + 20),
            "random": ScriptedPolicy(mode="random", seed=seed + 30),
            "all_decks": ScriptedPolicy(mode="random", pool=pool, seed=seed + 40),
            "random_deck_b": ScriptedPolicy(mode="random", pool=pool, seed=seed + 50),
        }
    league.add_agent("main", kind="main", policy=main, replace=True)
    for aid, pol in agents5.items():
        if aid == "main":
            continue
        league.add_agent(aid, kind="baseline", policy=pol, replace=True)
    return agents5


def _make_env(cfg, seed):
    return RLEnv(opponent=None, seed=seed, reward_weights=reward_to_env(cfg),
                 card_level=cfg.card_level)


def _make_trainer(main, cfg):
    return PPOTrainer(main, lr=cfg.lr, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda,
                      clip=cfg.clip, vf_coef=cfg.vf_coef, ent_coef=cfg.ent_coef,
                      max_grad_norm=cfg.max_grad_norm, adv_norm=cfg.adv_norm)


def _load_run_state(cfg):
    if os.path.exists(cfg.run_state_path()):
        try:
            with open(cfg.run_state_path(), "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None
    return None


def _restore(league, cfg, main, device, resume):
    """恢复联赛/训练进度；返回 (start_step, ppo, main)。resume=True 时从 run_state 续训。"""
    ppo = _make_trainer(main, cfg)
    start_step = 0
    if resume:
        rs = _load_run_state(cfg)
        if rs:
            ckpt, opt = rs.get("main_ckpt"), rs.get("opt_ckpt")
            if ckpt and os.path.exists(ckpt):
                main = load_checkpoint(ckpt, hidden_dim=cfg.hidden_dim)
                main.to_device(device)
                ppo = _make_trainer(main, cfg)
                if opt and os.path.exists(opt):
                    ppo.opt.load_state_dict(torch.load(opt, map_location=device))
                if os.path.exists(cfg.state_path()):
                    league.load_state(cfg.state_path(), policies={"main": main})
                start_step = int(rs.get("step", 0))
                print(f"[resume] 从 step {start_step} 续训（继续到 {cfg.total_steps}）", flush=True)
                return start_step, ppo, main
            print(f"[resume] 检查点缺失 {ckpt}，从头开始", flush=True)
        elif os.path.exists(cfg.state_path()):
            league.load_state(cfg.state_path(), policies={"main": main})
    elif os.path.exists(cfg.state_path()):
        league.load_state(cfg.state_path(), policies={"main": main})
    return start_step, ppo, main


def _save_snapshot(league, main, ppo, cfg, step, device):
    ckpt = cfg.ckpt_path(step)
    opt = cfg.opt_path(step)
    save_checkpoint(main, ckpt)
    torch.save(ppo.opt.state_dict(), opt)
    if cfg.keep_snapshot:
        league.refresh_snapshot("main", main, path=ckpt)
    league.save_state(cfg.state_path())
    run_state = {"step": int(step), "total_steps": int(cfg.total_steps),
                 "main_ckpt": ckpt, "opt_ckpt": opt,
                 "config": cfg.name, "device": device}
    with open(cfg.run_state_path(), "w", encoding="utf-8") as f:
        json.dump(run_state, f)


def _eval_and_snapshot(league, main, ppo, cfg, step, device, record_replays):
    replays = eval_round_robin(league, cfg.n_eval_games, cfg.max_ep_steps, cfg.seed, step,
                               only_vs_main=cfg.only_vs_main, record=record_replays)
    if record_replays and replays:
        rpath = os.path.join(cfg.replays_dir(), f"league_{step}.pkl")
        save_league_replays(replays, rpath)
        print(f"[replay@{step}] 已保存 {len(replays)} 局联赛录像 -> {rpath}", flush=True)
    _save_snapshot(league, main, ppo, cfg, step, device)
    print("=== League Elo ===")
    for aid, r in league.elo_table().items():
        print(f"  {aid:20s} Elo={r:.1f}", flush=True)


def _sample_opponent_for(league, env, seed):
    """从联赛 PFSP 采样一个对手并装配到 env（训练数据收集用）。"""
    op = league.sample_opponent("main")
    if op.policy is None:
        env.deck1_factory = None
        env.opponent = None
    elif isinstance(op.policy, ScriptedPolicy):
        op.policy.env = env
        env.deck1_factory = _deck_factory_of(op.policy)
        env.opponent = op.policy
    else:
        env.deck1_factory = None
        env.opponent = FollowerOpponent(op.policy, env,
                                        belief=BeliefInference(opp_deck=env.deck0,
                                                               n_particles=128, seed=seed))


def _build_league(cfg, device, resume):
    """公共前缀：创建 env/belief/main/ppo/league/agents。

    返回 (env0, belief0, main, ppo, league, start_step)。
    """
    env = _make_env(cfg, cfg.seed)
    belief = BeliefInference(opp_deck=env.deck1, n_particles=128, seed=cfg.seed)
    main = (load_checkpoint(cfg.main_init, hidden_dim=cfg.hidden_dim,
                            value_bypass=cfg.value_bypass,
                            value_independent=cfg.value_independent) if cfg.main_init
            else FollowerPolicy(hidden=cfg.hidden_dim, plan_dim=PLAN_DIM,
                                belief_dim=len(belief.encode(None, None)),
                                value_bypass=cfg.value_bypass,
                                value_independent=cfg.value_independent))
    main.to_device(device)
    league = League(seed=cfg.seed)
    start_step, ppo, main = _restore(league, cfg, main, device, resume)
    build_five_agents(league, main, cfg.seed, decks_path=cfg.decks_path)
    return env, belief, main, ppo, league, start_step


# ---------------------------------------------------------------------------
# 单 env 主循环（n_envs<=1，旧行为；供默认/兜底）
# ---------------------------------------------------------------------------

def _run_single(cfg: TrainConfig, resume=False, record_replays=True):
    device = resolve_device(cfg.device)
    cfg.ensure_dirs()
    cfg.save()
    print(f"[league] 配置 '{cfg.name}' -> {cfg.folder()} (device={device})", flush=True)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    env, belief, main, ppo, league, start_step = _build_league(cfg, device, resume)
    rng = random.Random(cfg.seed)
    bp = BeliefPlanner()
    prophet = ProphetPlanner()

    def sample_training_opponent():
        _sample_opponent_for(league, env, cfg.seed)

    def eval_and_snapshot(step):
        _eval_and_snapshot(league, main, ppo, cfg, step, device, record_replays)

    # 训练开始先跑一次评估/快照（WebUI 立即有真实数据而非只有预设 1500）
    if cfg.eval_at_start:
        eval_and_snapshot(0)

    obs, _ = env.reset()
    belief.reset(env.deck1)
    hidden = None
    last_eval_step = None
    ep_obs, ep_belief, ep_plan, ep_bundle, ep_lp, ep_val, ep_rew = [], [], [], [], [], [], []
    ep_term, ep_trunc, ep_masks, ep_init = [], [], [], []
    transitions = []
    sample_training_opponent()
    _t_train0 = time.monotonic()

    for step in range(start_step + 1, cfg.total_steps + 1):
        use_prophet = rng.random() < 0.3
        plan = prophet.plan(env.get_prophet_state()) if use_prophet else bp.plan(env.battle, belief.state(), obs)
        plan_vec = plan.to_vector()
        belief_tok = belief.encode(obs, None)
        init_hidden = hidden
        bundle, lp, val, hidden, masks = main.act(obs, belief_tok, plan_vec, env.get_action_mask,
                                                  hidden=hidden, deterministic=False)
        obs2, reward, term, trunc, info = env.step(bundle)
        done = term or trunc
        ep_obs.append(obs); ep_belief.append(belief_tok); ep_plan.append(plan_vec)
        ep_bundle.append(bundle); ep_lp.append(lp); ep_val.append(val); ep_rew.append(reward)
        ep_term.append(term); ep_trunc.append(trunc); ep_masks.append(masks); ep_init.append(init_hidden)
        belief.update(obs2, info.get("opp_played"))
        obs = obs2

        if done or (len(ep_rew) >= cfg.max_ep_steps and not overtime_open(env.battle)):
            truncated = ((not term) and (len(ep_rew) >= cfg.max_ep_steps
                                         and not overtime_open(env.battle)))
            # 早停/截断补结算（与 solo 一致）：皇冠差已分胜负 → 给终端胜负奖励；
            # 皇冠相同 = 加时窗口内无人再破塔 → 按平局=失败罚（不再用塔血提前判胜）
            if env.battle.winner is None and not env.battle.game_over and ep_rew:
                virt = timeout_winner(env.battle)
                rw = reward_to_env(cfg)
                if virt == 0:
                    ep_rew[-1] += float(rw["win_bonus"])
                elif virt == 1:
                    ep_rew[-1] -= float(rw["lose_penalty"])
                else:
                    ep_rew[-1] -= float(rw.get("draw_penalty", rw.get("lose_penalty", 10.0)))
            # P1-7 修复：env 恒返回 trunc=False，须显式标记截断末步 bootstrap 才生效
            if truncated:
                ep_trunc[-1] = True
                last_val = main.value(obs, belief_tok, plan_vec, hidden)
            else:
                last_val = 0.0
            adv, ret = PPOTrainer.compute_gae(ep_rew, ep_val, ep_term, cfg.gamma, cfg.gae_lambda,
                                              truncated=ep_trunc, last_value=last_val)
            for i in range(len(ep_rew)):
                transitions.append({"obs": ep_obs[i], "belief": ep_belief[i], "plan": ep_plan[i],
                                    "bundle": ep_bundle[i], "old_logprob": ep_lp[i],
                                    "adv": float(adv[i]), "returns": float(ret[i]),
                                    "masks": ep_masks[i], "init_hidden": ep_init[i]})
            obs, _ = env.reset()
            belief.reset(env.deck1)
            hidden = None
            ep_obs, ep_belief, ep_plan, ep_bundle, ep_lp, ep_val, ep_rew = [], [], [], [], [], [], []
            ep_term, ep_trunc, ep_masks, ep_init = [], [], [], []
            sample_training_opponent()

        if len(transitions) >= cfg.update_interval:
            stats = ppo.update(transitions[:cfg.batch_size] if len(transitions) > cfg.batch_size
                               else transitions)
            transitions = transitions[cfg.batch_size:] if len(transitions) > cfg.batch_size else []
            print(f"[step {step}] policy={stats['policy_loss']:.4f} value={stats['value_loss']:.4f} "
                  f"entropy={stats['entropy']:.4f}", flush=True)

        if cfg.steps_per_eval and step % cfg.steps_per_eval == 0:
            eval_and_snapshot(step)
            last_eval_step = step

    print(f"[train] 训练循环耗时 {time.monotonic() - _t_train0:.1f}s", flush=True)
    save_checkpoint(main, cfg.main_final_path())
    if last_eval_step != cfg.total_steps:
        eval_and_snapshot(cfg.total_steps)
    print(f"[done] config '{cfg.name}' 完成，产物在 {cfg.folder()}（含 Elo 历史，供网页 UI 读取）")


# ---------------------------------------------------------------------------
# 并行多 env 主循环（n_envs>1：batch 推理 + batch PPO 更新）
# ---------------------------------------------------------------------------

def _run_vec(cfg: TrainConfig, resume=False, record_replays=True):
    device = resolve_device(cfg.device)
    cfg.ensure_dirs()
    cfg.save()
    n = max(1, int(cfg.n_envs))
    print(f"[league] 配置 '{cfg.name}' -> {cfg.folder()} (device={device}, n_envs={n})",
          flush=True)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    env, belief, main, ppo, league, start_step = _build_league(cfg, device, resume)
    bp = BeliefPlanner()
    prophet = ProphetPlanner()
    rng = random.Random(cfg.seed)

    def eval_and_snapshot(step):
        _eval_and_snapshot(league, main, ppo, cfg, step, device, record_replays)

    # 训练开始先跑一次评估/快照
    if cfg.eval_at_start:
        eval_and_snapshot(0)

    envs = [_make_env(cfg, cfg.seed + i) for i in range(n)]
    beliefs = [BeliefInference(opp_deck=e.deck1, n_particles=128, seed=cfg.seed + i)
               for i, e in enumerate(envs)]
    obs_list = []
    hidden_list = [None] * n
    for i in range(n):
        _sample_opponent_for(league, envs[i], cfg.seed + i)
        envs[i].reset()
        beliefs[i].reset(envs[i].deck1)
        obs_list.append(envs[i].observe(0))

    def new_buf():
        return {"obs": [], "belief": [], "plan": [], "bundle": [], "lp": [], "val": [],
                "rew": [], "term": [], "trunc": [], "masks": [], "init": []}

    ep_bufs = [new_buf() for _ in range(n)]
    transitions = []

    step = start_step
    prev_block = (start_step // cfg.steps_per_eval) if cfg.steps_per_eval else 0
    last_eval_step = None

    while step < cfg.total_steps:
        # 1) 规划 + 信念编码（每个 env 独立，CPU）
        plans = []
        belief_toks = []
        for i in range(n):
            use_prophet = rng.random() < 0.3
            plan = (prophet.plan(envs[i].get_prophet_state()) if use_prophet
                    else bp.plan(envs[i].battle, beliefs[i].state(), obs_list[i]))
            plans.append(plan.to_vector())
            belief_toks.append(beliefs[i].encode(obs_list[i], None))

        # 2) 批量 act（一次前向喂 GPU）
        inits = list(hidden_list)
        get_masks = [envs[i].get_action_mask for i in range(n)]
        bundles, lps, vals, hidden_list, masks_list = main.act_parallel(
            obs_list, belief_toks, plans, get_masks,
            hidden_list=hidden_list, deterministic=False)

        # 3) 逐 env 推进 + 收 transition
        for i in range(n):
            b = ep_bufs[i]
            b["obs"].append(obs_list[i]); b["belief"].append(belief_toks[i])
            b["plan"].append(plans[i]); b["bundle"].append(bundles[i])
            b["lp"].append(lps[i]); b["val"].append(vals[i])
            b["init"].append(inits[i]); b["masks"].append(masks_list[i])
            obs2, reward, term, trunc, info = envs[i].step(bundles[i])
            done = term or trunc
            b["rew"].append(reward); b["term"].append(term); b["trunc"].append(trunc)
            beliefs[i].update(obs2, info.get("opp_played"))
            obs_list[i] = obs2

            if done or (len(b["rew"]) >= cfg.max_ep_steps and not overtime_open(envs[i].battle)):
                truncated = ((not term) and (len(b["rew"]) >= cfg.max_ep_steps
                                             and not overtime_open(envs[i].battle)))
                # 早停/截断补结算：皇冠差已分胜负 → 终端胜负；皇冠相同=加时未破塔 → 平局=失败
                if envs[i].battle.winner is None and not envs[i].battle.game_over and b["rew"]:
                    virt = timeout_winner(envs[i].battle)
                    rw = reward_to_env(cfg)
                    if virt == 0:
                        b["rew"][-1] += float(rw["win_bonus"])
                    elif virt == 1:
                        b["rew"][-1] -= float(rw["lose_penalty"])
                    else:
                        b["rew"][-1] -= float(rw.get("draw_penalty", rw.get("lose_penalty", 10.0)))
                # P1-7 修复：截断末步须显式标记，last_value bootstrap 才生效
                if truncated:
                    b["trunc"][-1] = True
                    last_val = main.value(obs2, belief_toks[i], plans[i], hidden_list[i])
                else:
                    last_val = 0.0
                adv, ret = PPOTrainer.compute_gae(b["rew"], b["val"], b["term"],
                                                  cfg.gamma, cfg.gae_lambda,
                                                  truncated=b["trunc"], last_value=last_val)
                for k in range(len(b["rew"])):
                    transitions.append({"obs": b["obs"][k], "belief": b["belief"][k],
                                        "plan": b["plan"][k], "bundle": b["bundle"][k],
                                        "old_logprob": b["lp"][k],
                                        "adv": float(adv[k]), "returns": float(ret[k]),
                                        "masks": b["masks"][k], "init_hidden": b["init"][k]})
                _sample_opponent_for(league, envs[i], cfg.seed + i)
                envs[i].reset()
                beliefs[i].reset(envs[i].deck1)
                hidden_list[i] = None
                obs_list[i] = envs[i].observe(0)
                ep_bufs[i] = new_buf()

        # 4) batch PPO 更新
        if len(transitions) >= cfg.update_interval:
            batch = transitions[:cfg.batch_size]
            stats = ppo.update(batch)
            transitions = transitions[len(batch):]
            print(f"[step {step}] policy={stats['policy_loss']:.4f} "
                  f"value={stats['value_loss']:.4f} entropy={stats['entropy']:.4f}",
                  flush=True)

        # 5) 按 env-steps 计步 + 评估
        step = min(cfg.total_steps, step + n)
        if cfg.steps_per_eval:
            block = step // cfg.steps_per_eval
            if block > prev_block:
                eval_and_snapshot(step)
                last_eval_step = step
                prev_block = block

    save_checkpoint(main, cfg.main_final_path())
    if last_eval_step != cfg.total_steps:
        eval_and_snapshot(cfg.total_steps)
    print(f"[done] config '{cfg.name}' 完成，产物在 {cfg.folder()}（含 Elo 历史，供网页 UI 读取）")


# ---------------------------------------------------------------------------
# 跨进程 worker 主循环（n_envs>1 且 parallel=mp）：多核真并行
# ---------------------------------------------------------------------------

def _run_mp(cfg: TrainConfig, resume=False, record_replays=True):
    """跨进程并行：env+信念+规划在独立 worker 进程跑（绕开 GIL），主进程批量 GPU 推理。

    协议见 rl/workers.py。worker 只做推演并回传 (obs, plan_vec, belief_tok)；
    掩码也按需回传（autoregressive 解码每步一次轻量 IPC）。
    """
    import multiprocessing as mp
    from rl.workers import worker_main

    device = resolve_device(cfg.device)
    cfg.ensure_dirs()
    cfg.save()
    n = max(1, int(cfg.n_envs))
    print(f"[league] 配置 '{cfg.name}' -> {cfg.folder()} "
          f"(device={device}, n_envs={n}, parallel=mp)", flush=True)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    _, _, main, ppo, league, start_step = _build_league(cfg, device, resume)

    def eval_and_snapshot(step):
        _eval_and_snapshot(league, main, ppo, cfg, step, device, record_replays)

    if cfg.eval_at_start:
        eval_and_snapshot(0)

    def spec_for(pol):
        if pol is None:
            return {"type": "none"}
        if isinstance(pol, ScriptedPolicy):
            return {"type": "scripted", "mode": pol.mode, "pool": pol.pool,
                    "deck_pool": pol.deck_pool, "seed": pol.seed}
        print("[mp] 警告：联赛采样到学习型对手，并行模式回退内置随机", flush=True)
        return {"type": "none"}

    def next_spec():
        return spec_for(league.sample_opponent("main").policy)

    # spawn 避免 fork 继承父进程 CUDA 上下文（GPU 场景安全）；Windows 必须 spawn
    ctx = mp.get_context("spawn" if (os.name == "nt" or device == "cuda") else "fork")
    in_qs, out_qs, procs = [], [], []
    for i in range(n):
        iq, oq = ctx.Queue(), ctx.Queue()
        w = ctx.Process(target=worker_main,
                        args=(i, cfg.seed + i, reward_to_env(cfg), iq, oq, cfg.card_level))
        try:
            w.start()
        except OSError as e:
            for p in procs:
                p.terminate()
            print(f"[mp] 训练 worker 启动失败（{e}），降级单进程模式继续训练", flush=True)
            return _run_single(cfg, resume=resume, record_replays=record_replays)
        in_qs.append(iq)
        out_qs.append(oq)
        procs.append(w)

    def recv(i, expect):
        try:
            msg = out_qs[i].get(timeout=120)
        except queue.Empty:
            if not procs[i].is_alive():
                # worker 启动即崩（如 WinError 1455 页面文件不足）不会发任何消息，
                # 原实现会在这里永久阻塞
                raise RuntimeError(f"worker{i} 已退出（exitcode={procs[i].exitcode}，"
                                   "常见原因：页面文件不足 WinError 1455）")
            raise
        if msg[0] != expect:
            raise RuntimeError(f"worker{i} 异常响应 '{msg[0]}'（期望 '{expect}'）: "
                               f"{msg[1] if len(msg) > 1 else ''}")
        return msg

    def mask_fn(i, partial):
        in_qs[i].put(("mask", partial))
        return recv(i, "mask")[1]

    def get_masks_batch(partials):
        """把 N 个掩码请求一次性并发发出再统一回收（worker 并行算 legal_cells）。"""
        for i in range(n):
            in_qs[i].put(("mask", partials[i]))
        out = []
        for i in range(n):
            out.append(recv(i, "mask")[1])
        return out

    def new_buf():
        return {"obs": [], "belief": [], "plan": [], "bundle": [], "lp": [], "val": [],
                "rew": [], "term": [], "trunc": [], "masks": [], "init": [], "winner": []}

    try:
        for i in range(n):
            in_qs[i].put(("reset", next_spec()))
        payloads = [recv(i, "ready")[1] for i in range(n)]
        get_masks = [lambda partial, i=i: mask_fn(i, partial) for i in range(n)]
        ep_bufs = [new_buf() for _ in range(n)]
        hidden_list = [None] * n
        transitions = []
        step = start_step
        prev_block = (start_step // cfg.steps_per_eval) if cfg.steps_per_eval else 0
        last_eval_step = None
        post_obs_list = [None] * n
        _t_train0 = time.monotonic()

        while step < cfg.total_steps:
            _t0 = time.monotonic()
            # 1) 批量 act（掩码按需回传 worker）
            pre_obs = [p[0] for p in payloads]
            toks = [p[1] for p in payloads]
            plans = [p[2] for p in payloads]
            inits = list(hidden_list)
            bundles, lps, vals, hidden_list, masks_list = main.act_parallel(
                pre_obs, toks, plans, get_masks, hidden_list=hidden_list,
                deterministic=False, get_masks_batch=get_masks_batch)
            _t1 = time.monotonic()
            # 2) 分发给各 worker 并行推演
            for i in range(n):
                in_qs[i].put(("step", bundles[i]))
            done_flags = []
            for i in range(n):
                msg = recv(i, "step")
                _, payload, reward, term, trunc, opp_played, winner, overtime_now = msg
                post_obs_list[i] = payload[0]
                payloads[i] = payload
                b = ep_bufs[i]
                b["obs"].append(pre_obs[i]); b["belief"].append(toks[i])
                b["plan"].append(plans[i]); b["bundle"].append(bundles[i])
                b["lp"].append(lps[i]); b["val"].append(vals[i])
                b["init"].append(inits[i]); b["masks"].append(masks_list[i])
                b["rew"].append(reward); b["term"].append(term); b["trunc"].append(trunc)
                b["winner"].append(winner)
                if term or trunc or (len(b["rew"]) >= cfg.max_ep_steps and not overtime_now):
                    done_flags.append(i)
            _t2 = time.monotonic()

            # 3) 收尾对局：GAE + 入库 + 重置 worker
            for i in done_flags:
                b = ep_bufs[i]
                truncated = (not b["term"][-1]) and (len(b["rew"]) >= cfg.max_ep_steps)
                # 平局=失败：截断且无胜者 → 最后一步加失败罚
                if truncated and b["winner"] and b["winner"][-1] is None and b["rew"]:
                    rw = reward_to_env(cfg)
                    b["rew"][-1] -= float(rw.get("draw_penalty", rw.get("lose_penalty", 10.0)))
                # P1-7 修复：截断末步须显式标记，last_value bootstrap 才生效
                if truncated:
                    b["trunc"][-1] = True
                    last_val = main.value(post_obs_list[i], b["belief"][-1], b["plan"][-1],
                                          hidden_list[i])
                else:
                    last_val = 0.0
                adv, ret = PPOTrainer.compute_gae(b["rew"], b["val"], b["term"],
                                                  cfg.gamma, cfg.gae_lambda,
                                                  truncated=b["trunc"], last_value=last_val)
                for k in range(len(b["rew"])):
                    transitions.append({"obs": b["obs"][k], "belief": b["belief"][k],
                                        "plan": b["plan"][k], "bundle": b["bundle"][k],
                                        "old_logprob": b["lp"][k],
                                        "adv": float(adv[k]), "returns": float(ret[k]),
                                        "masks": b["masks"][k], "init_hidden": b["init"][k]})
                in_qs[i].put(("reset", next_spec()))
            for i in done_flags:
                payloads[i] = recv(i, "ready")[1]
                hidden_list[i] = None
                ep_bufs[i] = new_buf()

            # 4) batch PPO 更新
            if len(transitions) >= cfg.update_interval:
                batch = transitions[:cfg.batch_size]
                stats = ppo.update(batch)
                transitions = transitions[len(batch):]
                print(f"[step {step}] policy={stats['policy_loss']:.4f} "
                      f"value={stats['value_loss']:.4f} entropy={stats['entropy']:.4f}",
                      flush=True)

            # 5) 按 env-steps 计步 + 评估
            step = min(cfg.total_steps, step + n)
            if cfg.steps_per_eval:
                block = step // cfg.steps_per_eval
                if block > prev_block:
                    eval_and_snapshot(step)
                    last_eval_step = step
                    prev_block = block
            if os.environ.get("DSH_MP_TIMING"):
                print(f"[timing] iter act={(_t1-_t0)*1000:.1f}ms "
                      f"sim={(_t2-_t1)*1000:.1f}ms total={(time.monotonic()-_t0)*1000:.1f}ms",
                      flush=True)
    finally:
        for i in range(n):
            try:
                in_qs[i].put(None)
            except Exception:
                pass
        for w in procs:
            w.join(timeout=5)

    print(f"[train] 训练循环耗时 {time.monotonic() - _t_train0:.1f}s", flush=True)
    save_checkpoint(main, cfg.main_final_path())
    if last_eval_step != cfg.total_steps:
        eval_and_snapshot(cfg.total_steps)
    print(f"[done] config '{cfg.name}' 完成，产物在 {cfg.folder()}（含 Elo 历史，供网页 UI 读取）")


def run_league(cfg: TrainConfig, resume=False, record_replays=True):
    """联赛主循环入口：n_envs>1 按 parallel 选择跨进程/单进程并行，否则单 env。"""
    if int(cfg.n_envs) > 1:
        if cfg.parallel == "proc":
            return _run_vec(cfg, resume=resume, record_replays=record_replays)
        return _run_mp(cfg, resume=resume, record_replays=record_replays)
    return _run_single(cfg, resume=resume, record_replays=record_replays)


def _force_utf8_stdout():
    """把 stdout/stderr 切到 UTF-8 + errors='replace'。

    训练日志含中文/emoji（如 GRU 活力告警的 ⚠️）。Windows 控制台默认 cp936，
    管道/重定向时 Python 用 locale 编码 → `print` 抛 UnicodeEncodeError；
    更坏的是它会被 `eval_and_write` 的 `except Exception` 吃掉后**再次**在
    except 处理器里抛（`{e!r}` 内嵌不可编码字符），直接崩掉训练
    （2026-09-11 由 test_solo_resume 实际复现）。这里统一兜底。
    """
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def main():
    _force_utf8_stdout()
    ap = argparse.ArgumentParser(description="联赛主循环：命名配置 + 奖惩机制 + 断点续训 + CUDA")
    ap.add_argument("--mode", choices=["eval", "run", "flow", "solo",
                                       "flow-sweep-stream", "flow-sweep-games5"],
                    default="run")
    # eval 模式
    ap.add_argument("--policies", nargs="+", default=None)
    ap.add_argument("--kinds", nargs="+", default=None)
    ap.add_argument("--n-games", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=600)
    # run 模式：配置
    ap.add_argument("--config", type=str, default="standard",
                    help="命名配置预设（standard/aggressive/defensive/lockdown/elixir/economy/fast，"
                         "或 --load-config 的 JSON）")
    ap.add_argument("--config-name", type=str, default=None,
                    help="覆盖配置名（输出文件夹名），默认用预设名")
    ap.add_argument("--load-config", type=str, default=None,
                    help="从 JSON 载入自定义配置（含奖励权重）")
    ap.add_argument("--save-config", type=str, default=None,
                    help="把解析后的配置导出为 JSON（可编辑后 --load-config 复用）")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="输出根目录（缺省 rl/config.py 里 out_dir=runs）")
    # 续训默认开：存在 run_state.json/solo_state.json/flow_state 就自动接着训；
    # 显式 --fresh 才从头（旧实验/想重跑时用）。
    ap.add_argument("--fresh", action="store_true",
                    help="忽略断点强制从头训练（默认：有断点就自动续训）")
    ap.add_argument("--resume", action="store_true",
                    help=argparse.SUPPRESS)  # 兼容旧脚本；现在默认已自动续训
    ap.add_argument("--no-replays", action="store_true",
                    help="不保存每评估周期的联赛录像（默认保存）")
    ap.add_argument("--no-eval-start", action="store_true",
                    help="训练开始不先跑一次评估（默认跑，WebUI 立即有真实数据）")
    ap.add_argument("--device", type=str, default=None,
                    help="cpu / cuda / auto（缺省 auto=可用则 cuda，cu130 支持）")
    # run 模式：超参覆盖（优先级高于配置预设）
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--steps-per-eval", type=int, default=None)
    ap.add_argument("--anchor-every", type=int, default=None,
                    help="C 方案：轻量锚点评估点间隔（只跑 baseline_rand 锚点+落快照，"
                         "不跑 main/对照两块）；0/缺省=关闭")
    ap.add_argument("--n-envs", type=int, default=None,
                    help="并行多环境数（>1 用批量推理/更新；默认 1）")
    ap.add_argument("--parallel", type=str, choices=["mp", "proc"], default=None,
                    help="n_envs>1 时并行方式：mp=跨进程 worker（多核真并行，默认）/ "
                         "proc=单进程批量化")
    ap.add_argument("--card-level", type=int, default=None,
                    help="本局全部卡牌等级 11-16（默认 11；费差机制跨等级一致）")
    ap.add_argument("--main-init", type=str, default=None)
    ap.add_argument("--decks-path", type=str, default=None,
                    help="三分类卡组 JSON 路径（缺省自动探测 docs/leaderboard_decks_classified.json）")
    ap.add_argument("--deck-set", type=str, default=None,
                    help="solo 镜像/对手卡组：default=原版 8 卡 / four=四种卡组对手池"
                         "（docs/four_decks_manual.md）/ list:卡1,卡2,...=显式 8 卡镜像")
    ap.add_argument("--keep-snapshot", action="store_true",
                    help="同时维护 main_ckpt 快照槽位（默认只维护 5 卡组模型 + main）")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--update-interval", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--n-eval-games", type=int, default=None)
    ap.add_argument("--max-ep-steps", type=int, default=None)
    ap.add_argument("--only-vs-main", action="store_true",
                    help="评估只打 main vs 其它（省时）；默认全轮转 5 模型")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--hidden-dim", type=int, default=None)
    # flow 模式（全配对分流派联赛）
    ap.add_argument("--n-random-decks", type=int, default=30,
                    help="flow 模式：完全随机卡组每次训练生成套数（默认 30）")
    # flow-sweep 模式（缩小 10× 池的数据效率 A/B）
    ap.add_argument("--sweep-runs", type=int, default=None,
                    help="flow-sweep：训练轮数覆盖（stream 默认 20 / games5 默认 4）")
    ap.add_argument("--sweep-scale", type=float, default=0.1,
                    help="flow-sweep：卡组池缩小比例（默认 0.1 = 降低一个数量级）")
    ap.add_argument("--sweep-eval-games", type=int, default=None,
                    help="flow-sweep：每对评估局数（默认 10）")
    # solo 模式（单人自对弈，无联赛）
    ap.add_argument("--solo-copy-every", type=int, default=None,
                    help="solo：冻结副本同步间隔（步，默认 2000）")
    ap.add_argument("--eval-workers", type=int, default=None,
                    help="评估并行进程数（>1 时 spawn N 进程并行打评估局，绕开 GIL；"
                         "默认 0=串行。战斗模拟是纯 Python，跨进程才真正吃满多核）")
    # 纯 RL 冷启动修复的 A/B 开关（覆盖配置预设）
    ap.add_argument("--gae-lambda", type=float, default=None,
                    help="GAE λ（economy 预设已默认 0.99；0.95=旧视野）")
    ap.add_argument("--ent-coef", type=float, default=None,
                    help="熵系数覆盖（默认 0.01；冷启动试探可临时调高）")
    ap.add_argument("--adv-norm", type=str, choices=["batch", "scale", "none"], default=None,
                    help="advantage 归一化：batch=整批中心化(旧) / scale=只除std / none=原始")
    ap.add_argument("--value-norm", type=str, choices=["none", "running"], default=None,
                    help="价值通道量纲：none=旧行为（价值损失不缩放）/ running=按回报运行 "
                         "std 缩放（v_loss/=s²，修 value_loss 压倒策略项，见 rl/ppo.py）")
    ap.add_argument("--diagnose-every", type=int, default=None,
                    help="梯度成分诊断采样间隔（每 N 次 update 打印 p_gnorm/v_gnorm；"
                         "0=关闭。用于证实/证伪'评论家主导更新'）")
    ap.add_argument("--hist-seed-dir", action="append", default=None,
                    help="热启动 run 的对手池补种目录（可多次）。本目录无 solo_main_*.pt 时"
                         "从这些目录抽 hist ckpt，修对手池退化为 frozen+defend 的问题")
    ap.add_argument("--train-stall-stop", action="store_true",
                    help="solo：**开启**僵局早停（2026-09-17 起默认关；开=旧行为：连续 100 步"
                         "零塔损即判平截断）。默认关的理由见 TrainConfig.train_stall_stop 注释")
    ap.add_argument("--no-train-stall-stop", action="store_true",
                    help="solo：关闭训练环僵局早停（**自 2026-09-17 起这是默认值**，本开关保留兼容）")
    ap.add_argument("--adv-inert-probe", action="store_true",
                    help="critic 惰性检验【纯测量】：每个诊断更新额外算一份 V≡常数 的优势，"
                         "报告 corr/resid_frac/grad_cos（不改训练行为）。"
                         "预注册 docs/critic_inertia_prereg_2026-09-13.md")
    ap.add_argument("--critic-baseline", type=str, choices=["value", "const"], default=None,
                    help="critic 惰性检验【干预】：value=旧行为；const=把优势里的 V 项换成"
                         "标量 c（returns/价值损失/预算/奖励全不动）。非推荐配置")
    ap.add_argument("--stall-draw-margin", type=float, default=None,
                    help="C'（2026-09-12）早停低置信裁定降噪：皇冠相同时，塔血%%细差 < 该阈值"
                         "记平局=失败（去掉掷硬币级胜负标签）；0=退化为旧行为（细差也判胜负）")
    ap.add_argument("--no-value-bypass", action="store_true",
                    help="关闭 B'（value 直连 enc，跳过 GRU）：回旧 GRU value 通路，"
                         "架构消融用（economy 预设默认开）")
    ap.add_argument("--no-value-independent", action="store_true",
                    help="关闭 E'（独立价值编码器 + MLP 头）：回共享 trunk value 通路，"
                         "架构消融用（economy 预设默认开；优先级 independent > bypass）")
    # —— F'（2026-09-12）：真正的 PPO 更新预算（多轮 × 打乱 × 小批）——
    ap.add_argument("--ppo-epochs", type=int, default=None,
                    help="F'：同一份 rollout 上重复几轮更新（默认 1 = 旧行为；建议 4~8）。"
                         "旧实现每次 update 只有 1 次 opt.step ⇒ 20k 步仅 156 次梯度步，"
                         "critic（任何架构）都学不动（见 docs/rl_training_fix_plan_v3.md §3.11.1）")
    ap.add_argument("--ppo-minibatch", type=int, default=None,
                    help="F'：每轮切成多大的小批（0=整批/旧行为；建议 32~64）。"
                         "旧实现喂进来的是同一局连续 128 帧（corr(R_t,R_t+1)≈0.99），"
                         "小批+打乱才打破批内同质性")
    ap.add_argument("--ppo-shuffle", action="store_true",
                    help="F'：每轮打乱样本顺序（默认关，配合 --ppo-epochs/--ppo-minibatch 用）")
    args = ap.parse_args()
    # 自动续训为默认：无 --fresh 时 resume=True（断点缺失/不存在时各入口会自行从头并提示）
    resume = not args.fresh

    if args.mode == "eval":
        if not args.policies:
            ap.error("--mode eval 需要 --policies")
        evaluate_league(args.policies, args.kinds, args.n_games, args.seed or 0,
                        args.hidden_dim or 128, max_steps=args.max_steps,
                        device=args.device or "auto")
        return

    # ---- run 模式：解析命名配置 + 命令行覆盖 ----
    overrides = {}
    for k in ("total_steps", "steps_per_eval", "anchor_every", "n_envs", "parallel",
              "card_level",
              "batch_size", "update_interval", "lr", "hidden_dim", "seed",
              "n_eval_games", "max_ep_steps", "device", "main_init", "decks_path",
              "deck_set",
              "solo_copy_every", "eval_workers",
              "gae_lambda", "ent_coef", "adv_norm", "value_norm", "diagnose_every",
              "ppo_epochs", "ppo_minibatch"):
        v = getattr(args, k)
        if v is not None:
            overrides[k] = v
    if args.ppo_shuffle:
        overrides["ppo_shuffle"] = True
    if args.hist_seed_dir:
        overrides["hist_seed_dirs"] = list(args.hist_seed_dir)
    if args.stall_draw_margin is not None:
        overrides["stall_draw_margin"] = float(args.stall_draw_margin)
    if args.no_value_bypass:
        overrides["value_bypass"] = False
    if args.no_value_independent:
        overrides["value_independent"] = False
    if args.no_train_stall_stop:
        overrides["train_stall_stop"] = False
    if args.train_stall_stop:
        overrides["train_stall_stop"] = True
    if args.adv_inert_probe:
        overrides["adv_inert_probe"] = True
    if args.critic_baseline is not None:
        overrides["critic_baseline"] = args.critic_baseline
    if args.keep_snapshot:
        overrides["keep_snapshot"] = True
    if args.only_vs_main:
        overrides["only_vs_main"] = True
    if args.out_dir is not None:
        overrides["out_dir"] = args.out_dir
    if args.config_name:
        overrides["name"] = args.config_name
    if args.no_eval_start:
        overrides["eval_at_start"] = False

    cfg = TrainConfig.resolve(args.config, load_config=args.load_config, **overrides)
    if args.save_config:
        path = cfg.save(args.save_config)
        print(f"[config] 已导出配置 -> {path}")

    if args.mode == "flow":
        from rl.flow_league import run_flow
        run_flow(cfg, resume=resume, n_random_decks=args.n_random_decks)
        return

    if args.mode in ("flow-sweep-stream", "flow-sweep-games5"):
        # 缩小 10× 池的数据效率 A/B：先验证 flow 曲线上涨再上 148,800 全规模
        from rl.flow_league import run_flow_sweep
        strategy = "stream" if args.mode == "flow-sweep-stream" else "games5"
        run_flow_sweep(cfg, strategy=strategy, n_runs=args.sweep_runs,
                       pool_scale=args.sweep_scale,
                       eval_games=args.sweep_eval_games)
        return

    if args.mode == "solo":
        # 单人自对弈：固定卡组镜像 + 周期冻结副本，无联赛机制（原版 train.py 思路）
        from rl.train_solo import run_solo
        run_solo(cfg, resume=resume, record_replays=not args.no_replays)
        return

    run_league(cfg, resume=resume, record_replays=not args.no_replays)


if __name__ == "__main__":
    main()
