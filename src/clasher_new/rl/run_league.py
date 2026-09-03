"""联赛循环（规划文档 7.4 / 7.5）+ 训练网页 UI 数据源。

两个入口：
- ``evaluate_league``（--mode eval）：保存策略轮转对战（换边 + 三态 + 逐局 Elo）；
- ``run_league``（--mode run）：**同时维护 5 个模型** 的联赛主循环——PPO 训练 main →
  PFSP 采对手 → 周期全轮转评估（含新模型 random_deck）→ 逐局 Elo → 快照刷新 →
  Elo 历史曲线 → 状态持久化（供训练网页 UI 读取）。

新增能力（对应需求）：
- ``rl/config.py`` 命名配置：每组参数（超参 + **奖惩机制奖励权重**）命名后
  独立输出到 ``out_dir/<name>/``（断点 / 检查点 / 联赛录像 / Elo 状态 / config.json）；
- ``--resume`` 断点续训：从 run_state.json 恢复 step、main 权重与 Adam 状态；
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

    def __init__(self, a_id, b_id, side0, max_steps):
        self.meta = {"pair": [a_id, b_id], "side0": side0, "max_steps": max_steps}
        self.frames = []
        self.winner = None

    def record(self, env, bundle, reward, info):
        self.frames.append(battle_snapshot(env.battle, bundle, reward, info))

    def done(self, winner):
        self.winner = winner
        return {"meta": self.meta, "winner": winner, "frames": self.frames}


def _run_side0(env, policy, belief, bp, max_steps=300, recorder=None):
    """policy 以 player-0 身份打完整对局；返回 winner（0/1/None=平）。

    支持 FollowerPolicy（完整信念/plan 链路）与 ScriptedPolicy（随机合法出牌）。
    recorder: LeagueGameRecorder 可选，逐帧记录联赛录像。
    """
    if isinstance(policy, ScriptedPolicy):
        return _run_side0_scripted(env, policy, max_steps, recorder)
    obs, _ = env.reset()
    belief.reset(env.deck1)
    hidden = None
    done = False
    steps = 0
    opp_side = env.opponent if isinstance(env.opponent, FollowerOpponent) else None
    while not done and steps < max_steps:
        plan = bp.plan(env.battle, belief.state(), obs)
        tok = belief.encode(obs, None)
        bundle, _, _, hidden, _ = policy.act(obs, tok, plan.to_vector(),
                                             env.get_action_mask, hidden=hidden, deterministic=True)
        agent_played = _bundle_cards(bundle, obs)
        obs, reward, term, trunc, info = env.step(bundle)
        if recorder is not None:
            recorder.record(env, bundle, reward, info)
        if opp_side is not None:
            opp_side.observe_opponent_played(agent_played)
        belief.update(obs, info.get("opp_played"))
        done = term or trunc
        steps += 1
    return env.battle.winner


def _run_side0_scripted(env, policy, max_steps=300, recorder=None):
    obs, _ = env.reset()
    done = False
    steps = 0
    opp_side = env.opponent if isinstance(env.opponent, FollowerOpponent) else None
    while not done and steps < max_steps:
        bundle = policy.play(env, 0)
        agent_played = _bundle_cards(bundle, obs)
        obs, reward, term, trunc, info = env.step(bundle)
        if recorder is not None:
            recorder.record(env, bundle, reward, info)
        if opp_side is not None:
            opp_side.observe_opponent_played(agent_played)
        done = term or trunc
        steps += 1
    return env.battle.winner


def _bundle_cards(bundle, obs):
    out = []
    for sa in bundle.sub_actions:
        if sa.kind == "deploy" and 1 <= sa.slot <= K_MAX:
            cid = int(obs["hand"][sa.slot - 1])
            if 0 <= cid < len(ENTITY_NAMES):
                out.append(ENTITY_NAMES[cid])
    return out


def _make_opp(policy, env, deck):
    """把 policy 包成 player-1 对手；None = 内置随机（固定卡组）。"""
    if policy is None:
        return None
    if isinstance(policy, ScriptedPolicy):
        policy.env = env
        env.deck1_factory = policy.deck if policy.pool else None
        return policy
    return FollowerOpponent(policy, env,
                            belief=BeliefInference(opp_deck=list(deck), n_particles=128, seed=0))


def _prepare_env(env, side0_pol, side1_pol):
    """按双方策略类型装配 env（随机卡组工厂 + 对手），在 reset 前调用。"""
    if isinstance(side0_pol, ScriptedPolicy):
        env.deck0_factory = side0_pol.deck if side0_pol.pool else None
    env.opponent = _make_opp(side1_pol, env, env.deck0)
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
    for g in range(n_games):
        if g % 2 == 0:
            env = _prepare_env(RLEnv(opponent=None, seed=seed + g), a_pol, b_pol)
            belief = BeliefInference(opp_deck=env.deck1, n_particles=128, seed=seed + g)
            rec = LeagueGameRecorder(a_id, b_id, a_id, max_steps) if record else None
            w = _run_side0(env, a_pol, belief, BeliefPlanner(), max_steps, rec)
            if rec is not None:
                replays.append(rec.done(w))
            score_a = 1.0 if w == 0 else (0.5 if w is None else 0.0)
        else:
            env = _prepare_env(RLEnv(opponent=None, seed=seed + g + 5000), b_pol, a_pol)
            belief = BeliefInference(opp_deck=env.deck1, n_particles=128, seed=seed + g + 5000)
            rec = LeagueGameRecorder(a_id, b_id, b_id, max_steps) if record else None
            w = _run_side0(env, b_pol, belief, BeliefPlanner(), max_steps, rec)
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


def eval_round_robin(league, n_games, max_steps, seed, step, only_vs_main=False, record=False):
    """全轮转评估：所有有策略的 agent 两两换边对战，逐局 Elo，并记录历史曲线。

    返回采集到的联赛录像列表（record=True 时非空）。
    """
    ids = [aid for aid, ag in league.agents.items() if ag.policy is not None]
    pairs = list(itertools.combinations(ids, 2))
    if only_vs_main:
        pairs = [p for p in pairs if "main" in p]
    replays = []
    for idx, (a, b) in enumerate(pairs):
        pair_seed = seed + step + _pair_seed_offset(idx, a, b)
        wins_a, wins_b, draws, rs = play_pair(league, a, league.agents[a].policy,
                                              b, league.agents[b].policy,
                                              n_games, max_steps, pair_seed, record=record)
        replays.extend(rs)
        print(f"[eval@{step}] {a} vs {b}: {wins_a}W {wins_b}L {draws}D", flush=True)
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
    for idx, (a, b) in enumerate(itertools.combinations(policies, 2)):
        wa, wb, dr, _ = play_pair(league, a, pols[a], b, pols[b], n_games, max_steps,
                                  seed + _pair_seed_offset(idx, a, b))
        print(f"{os.path.basename(a)} vs {os.path.basename(b)}: "
              f"{wa}W {wb}L {dr}D / {n_games}", flush=True)
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
                      max_grad_norm=cfg.max_grad_norm)


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
        env.deck1_factory = op.policy.deck if op.policy.pool else None
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
    main = (load_checkpoint(cfg.main_init, hidden_dim=cfg.hidden_dim) if cfg.main_init
            else FollowerPolicy(hidden=cfg.hidden_dim, plan_dim=PLAN_DIM,
                                belief_dim=len(belief.encode(None, None))))
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

        if done or len(ep_rew) >= cfg.max_ep_steps:
            truncated = (not term) and (len(ep_rew) >= cfg.max_ep_steps)
            last_val = main.value(obs, belief_tok, plan_vec, hidden) if truncated else 0.0
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

            if done or len(b["rew"]) >= cfg.max_ep_steps:
                truncated = (not term) and (len(b["rew"]) >= cfg.max_ep_steps)
                last_val = (main.value(obs2, belief_toks[i], plans[i], hidden_list[i])
                            if truncated else 0.0)
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
        w.start()
        in_qs.append(iq)
        out_qs.append(oq)
        procs.append(w)

    def recv(i, expect):
        msg = out_qs[i].get()
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
                "rew": [], "term": [], "trunc": [], "masks": [], "init": []}

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
                _, payload, reward, term, trunc, opp_played, winner = msg
                post_obs_list[i] = payload[0]
                payloads[i] = payload
                b = ep_bufs[i]
                b["obs"].append(pre_obs[i]); b["belief"].append(toks[i])
                b["plan"].append(plans[i]); b["bundle"].append(bundles[i])
                b["lp"].append(lps[i]); b["val"].append(vals[i])
                b["init"].append(inits[i]); b["masks"].append(masks_list[i])
                b["rew"].append(reward); b["term"].append(term); b["trunc"].append(trunc)
                if term or trunc or len(b["rew"]) >= cfg.max_ep_steps:
                    done_flags.append(i)
            _t2 = time.monotonic()

            # 3) 收尾对局：GAE + 入库 + 重置 worker
            for i in done_flags:
                b = ep_bufs[i]
                truncated = (not b["term"][-1]) and (len(b["rew"]) >= cfg.max_ep_steps)
                last_val = (main.value(post_obs_list[i], b["belief"][-1], b["plan"][-1],
                                       hidden_list[i]) if truncated else 0.0)
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


def main():
    ap = argparse.ArgumentParser(description="联赛主循环：命名配置 + 奖惩机制 + 断点续训 + CUDA")
    ap.add_argument("--mode", choices=["eval", "run", "flow"], default="run")
    # eval 模式
    ap.add_argument("--policies", nargs="+", default=None)
    ap.add_argument("--kinds", nargs="+", default=None)
    ap.add_argument("--n-games", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=600)
    # run 模式：配置
    ap.add_argument("--config", type=str, default="standard",
                    help="命名配置预设（standard/aggressive/defensive/elixir/fast，或 --load-config 的 JSON）")
    ap.add_argument("--config-name", type=str, default=None,
                    help="覆盖配置名（输出文件夹名），默认用预设名")
    ap.add_argument("--load-config", type=str, default=None,
                    help="从 JSON 载入自定义配置（含奖励权重）")
    ap.add_argument("--save-config", type=str, default=None,
                    help="把解析后的配置导出为 JSON（可编辑后 --load-config 复用）")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="输出根目录（缺省 rl/config.py 里 out_dir=runs）")
    ap.add_argument("--resume", action="store_true", help="从 run_state.json 断点续训")
    ap.add_argument("--no-replays", action="store_true",
                    help="不保存每评估周期的联赛录像（默认保存）")
    ap.add_argument("--no-eval-start", action="store_true",
                    help="训练开始不先跑一次评估（默认跑，WebUI 立即有真实数据）")
    ap.add_argument("--device", type=str, default=None,
                    help="cpu / cuda / auto（缺省 auto=可用则 cuda，cu130 支持）")
    # run 模式：超参覆盖（优先级高于配置预设）
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--steps-per-eval", type=int, default=None)
    ap.add_argument("--n-envs", type=int, default=None,
                    help="并行多环境数（>1 用批量推理/更新；默认 1）")
    ap.add_argument("--parallel", type=str, choices=["mp", "proc"], default=None,
                    help="n_envs>1 时并行方式：mp=跨进程 worker（多核真并行，默认）/ "
                         "proc=单进程批量化")
    ap.add_argument("--card-level", type=int, default=None,
                    help="本局全部卡牌等级 11-16（默认 11；economy 奖励跨等级一致）")
    ap.add_argument("--main-init", type=str, default=None)
    ap.add_argument("--decks-path", type=str, default=None,
                    help="三分类卡组 JSON 路径（缺省自动探测 docs/leaderboard_decks_classified.json）")
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
    args = ap.parse_args()

    if args.mode == "eval":
        if not args.policies:
            ap.error("--mode eval 需要 --policies")
        evaluate_league(args.policies, args.kinds, args.n_games, args.seed or 0,
                        args.hidden_dim or 128, max_steps=args.max_steps,
                        device=args.device or "auto")
        return

    # ---- run 模式：解析命名配置 + 命令行覆盖 ----
    overrides = {}
    for k in ("total_steps", "steps_per_eval", "n_envs", "parallel", "card_level",
              "batch_size", "update_interval", "lr", "hidden_dim", "seed",
              "n_eval_games", "max_ep_steps", "device", "main_init", "decks_path"):
        v = getattr(args, k)
        if v is not None:
            overrides[k] = v
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
        run_flow(cfg, resume=args.resume, n_random_decks=args.n_random_decks)
        return

    run_league(cfg, resume=args.resume, record_replays=not args.no_replays)


if __name__ == "__main__":
    main()
