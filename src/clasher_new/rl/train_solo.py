"""单人自对弈训练（原版 ``train.py`` 思路的现代版；**无联赛机制**）。

- 单模型 main（``FollowerPolicy`` + ``PPOTrainer``），双方使用**同一副固定卡组**
  （默认 8 卡，镜像对局；``DEFAULT_SOLO_DECK``）；
- 对手 = main 的**周期冻结副本**：每 ``cfg.solo_copy_every`` 步把 main 权重拷给
  opponent（原版 ``train.py`` ``WeightsCopyingCallback`` 的现代版），对手 deterministic；
- 无联赛：不建 ``League``、不写 Elo/PFSP/``league_state.json``；
- 周期评估（``cfg.steps_per_eval``）写 ``solo_state.json``
  （winrate±SE / mean_reward / 进度字段），供 dashboard ``--solo`` 实时显示；
- 评估回放存 ``replays/league_<step>.pkl``（复用 dashboard 回放列表/播放器）。

用法（run_league 入口）：
    python rl/run_league.py --mode solo --config economy --device cuda
"""

import os
import sys
import json
import math
import time
import random

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import numpy as np
import torch

from rl.env_wrapper import RLEnv
from rl.belief import BeliefInference
from rl.belief_planner import BeliefPlanner
from rl.prophet import ProphetPlanner
from rl.plan_space import PLAN_DIM
from rl.follower import FollowerPolicy, save_checkpoint, load_checkpoint
from rl.ppo import PPOTrainer
from rl.config import reward_to_env
from rl.train_follower import FollowerOpponent
from rl.run_league import resolve_device, _bundle_cards, LeagueGameRecorder
from rl.replay import save_league_replays

#: 固定卡组（原版默认 8 卡）：双方镜像使用同一副。
DEFAULT_SOLO_DECK = ["Knight", "MiniPekka", "Arrows", "Minions",
                     "Musketeer", "Fireball", "Giant", "Archer"]

#: 训练中先知规划注入概率（与 run_league 主训练一致）
_SOLO_PROPHET_PROB = 0.3


def solo_env(cfg, seed):
    """双方同一副固定卡组的镜像 RLEnv。"""
    return RLEnv(opponent=None, seed=seed, reward_weights=reward_to_env(cfg),
                 card_level=cfg.card_level,
                 deck0=DEFAULT_SOLO_DECK, deck1=DEFAULT_SOLO_DECK)


def _sync_frozen_copy(main, opp):
    """把 main 当前权重同步给冻结副本（周期执行）。"""
    opp.load_state_dict(main.state_dict())


def write_solo_state(path, cfg, history, step, status="running",
                     deck=None, copy_every=None, target_steps=None):
    """把 solo 训练状态落盘（增量写；dashboard --solo 实时读取）。"""
    state = {
        "mode": "solo",
        "agents": [{"agent_id": "main", "kind": "main", "path": None}],
        "history": history,             # [{step,wins,losses,draws,games,winrate,winrate_se,mean_reward}]
        "total_steps": int(step),
        "target_steps": int(target_steps if target_steps is not None else cfg.total_steps),
        "deck": deck if deck is not None else list(DEFAULT_SOLO_DECK),
        "opponent": "self-play-frozen-copy",
        "copy_every": int(copy_every if copy_every is not None else cfg.solo_copy_every),
        "status": status,
        "demo": False,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    return state


def eval_solo(env, main, opp, n_games, max_steps, seed, cfg,
              record_replays=False, replays_dir=None, step=None):
    """main（deterministic）vs 冻结副本（deterministic）打 n_games。

    返回 (stats, replays)。replays 非空时以 league_<step>.pkl 落盘（复用 dashboard 回放）。
    """
    bp = BeliefPlanner()
    wins = losses = draws = 0
    rew_sum = 0.0
    replays = []
    for g in range(n_games):
        opp_side = FollowerOpponent(
            opp, env,
            belief=BeliefInference(opp_deck=env.deck1, n_particles=128, seed=seed + g),
            deterministic=True)
        env.opponent = opp_side
        belief = BeliefInference(opp_deck=env.deck1, n_particles=128, seed=seed + 1000 + g)
        obs, _ = env.reset(seed=seed + 2000 + g)
        belief.reset(env.deck1)
        hidden = None
        rec = LeagueGameRecorder("main", "frozen_copy", "main", max_steps) if record_replays else None
        done = False
        steps = 0
        ep_rew = 0.0
        while not done and steps < max_steps:
            plan = bp.plan(env.battle, belief.state(), obs)
            tok = belief.encode(obs, None)
            bundle, _, _, hidden, _ = main.act(
                obs, tok, plan.to_vector(), env.get_action_mask,
                hidden=hidden, deterministic=True)
            played = _bundle_cards(bundle, obs)
            obs, reward, term, trunc, info = env.step(bundle)
            ep_rew += float(reward)
            if rec is not None:
                rec.record(env, bundle, reward, info)
            opp_side.observe_opponent_played(played)
            belief.update(obs, info.get("opp_played"))
            done = term or trunc
            steps += 1
        w = env.battle.winner
        if w == 0:
            wins += 1
        elif w == 1:
            losses += 1
        else:
            draws += 1
        rew_sum += ep_rew
        if rec is not None:
            replays.append(rec.done(w))
    n = max(1, n_games)
    winrate = (wins + 0.5 * draws) / n
    se = math.sqrt(max(0.0, winrate * (1.0 - winrate)) / n) if n > 1 else 0.5
    stats = {"step": step, "wins": wins, "losses": losses, "draws": draws,
             "games": n, "winrate": round(winrate, 4),
             "winrate_se": round(se, 4), "mean_reward": round(rew_sum / n, 4)}
    if replays and replays_dir and step is not None:
        os.makedirs(replays_dir, exist_ok=True)
        save_league_replays(replays, os.path.join(replays_dir, f"league_{step}.pkl"))
    return stats, replays


def run_solo(cfg, resume=False, record_replays=True):
    """单人自对弈主循环（无联赛；写 solo_state.json + solo_main.pt）。

    流程：镜像固定卡组 → main 训练 vs 冻结副本（每 solo_copy_every 步同步）→
    每 steps_per_eval 评估并写 solo_state.json（dashboard 实时显示）。
    """
    device = resolve_device(cfg.device)
    cfg.ensure_dirs()
    cfg.save()
    print(f"[solo] 单人自对弈 配置 '{cfg.name}' -> {cfg.folder()} "
          f"(device={device}, seed={cfg.seed}, 固定卡组 {len(DEFAULT_SOLO_DECK)} 卡镜像, "
          f"冻结副本同步间隔={cfg.solo_copy_every})", flush=True)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    env = solo_env(cfg, cfg.seed)
    belief_dim = len(BeliefInference(opp_deck=env.deck1, n_particles=128,
                                     seed=0).encode(None, None))
    if cfg.main_init:
        main = load_checkpoint(cfg.main_init, hidden_dim=cfg.hidden_dim)
    else:
        main = FollowerPolicy(hidden=cfg.hidden_dim, plan_dim=PLAN_DIM,
                              belief_dim=belief_dim)
    main.to_device(device)
    opp = FollowerPolicy(hidden=cfg.hidden_dim, plan_dim=PLAN_DIM, belief_dim=belief_dim)
    opp.to_device(device)
    _sync_frozen_copy(main, opp)   # 开局副本 = main 初始权重
    ppo = PPOTrainer(main, lr=cfg.lr, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda,
                     clip=cfg.clip, vf_coef=cfg.vf_coef, ent_coef=cfg.ent_coef,
                     max_grad_norm=cfg.max_grad_norm)
    bp = BeliefPlanner()
    prophet = ProphetPlanner()
    rng = random.Random(cfg.seed)

    opp_side = FollowerOpponent(opp, env,
                                belief=BeliefInference(opp_deck=env.deck1,
                                                       n_particles=128, seed=cfg.seed),
                                deterministic=True)
    env.opponent = opp_side
    belief = BeliefInference(opp_deck=env.deck1, n_particles=128, seed=cfg.seed)

    history = []

    def eval_and_write(step):
        stats, _ = eval_solo(env, main, opp, int(cfg.n_eval_games),
                             int(cfg.max_ep_steps), cfg.seed + step, cfg,
                             record_replays=record_replays,
                             replays_dir=cfg.replays_dir(), step=step)
        history.append(stats)
        write_solo_state(cfg.solo_state_path(), cfg, history, step,
                         status="done" if step >= cfg.total_steps else "running")
        print(f"[solo] eval@{step}: 胜率 {stats['winrate']:.3f}±{stats['winrate_se']:.3f} "
              f"({stats['wins']}W/{stats['losses']}L/{stats['draws']}D, "
              f"{stats['games']}局) mean_reward={stats['mean_reward']:.3f}", flush=True)
        save_checkpoint(main, cfg.solo_main_path())

    # 训练开始先跑一次评估（WebUI 立即有真实数据）
    if cfg.eval_at_start:
        eval_and_write(0)

    obs, _ = env.reset()
    belief.reset(env.deck1)
    hidden = None
    last_eval_step = None
    ep_obs, ep_belief, ep_plan, ep_bundle, ep_lp, ep_val, ep_rew = [], [], [], [], [], [], []
    ep_term, ep_trunc, ep_masks, ep_init = [], [], [], []
    transitions = []
    _t0 = time.monotonic()

    for step in range(1, cfg.total_steps + 1):
        use_prophet = rng.random() < _SOLO_PROPHET_PROB
        plan = prophet.plan(env.get_prophet_state()) if use_prophet \
            else bp.plan(env.battle, belief.state(), obs)
        plan_vec = plan.to_vector()
        belief_tok = belief.encode(obs, None)
        init_hidden = hidden
        bundle, lp, val, hidden, masks = main.act(
            obs, belief_tok, plan_vec, env.get_action_mask,
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
            adv, ret = PPOTrainer.compute_gae(ep_rew, ep_val, ep_term, cfg.gamma,
                                              cfg.gae_lambda, truncated=ep_trunc,
                                              last_value=last_val)
            for i in range(len(ep_rew)):
                transitions.append({"obs": ep_obs[i], "belief": ep_belief[i],
                                    "plan": ep_plan[i], "bundle": ep_bundle[i],
                                    "old_logprob": ep_lp[i], "adv": float(adv[i]),
                                    "returns": float(ret[i]), "masks": ep_masks[i],
                                    "init_hidden": ep_init[i]})
            obs, _ = env.reset()
            belief.reset(env.deck1)
            opp_side.reset()
            hidden = None
            ep_obs, ep_belief, ep_plan, ep_bundle, ep_lp, ep_val, ep_rew = [], [], [], [], [], [], []
            ep_term, ep_trunc, ep_masks, ep_init = [], [], [], []

        if len(transitions) >= cfg.update_interval:
            stats = ppo.update(transitions[:cfg.batch_size] if len(transitions) > cfg.batch_size
                               else transitions)
            transitions = transitions[cfg.batch_size:] if len(transitions) > cfg.batch_size else []
            print(f"[solo step {step}] policy={stats['policy_loss']:.4f} "
                  f"value={stats['value_loss']:.4f} entropy={stats['entropy']:.4f}",
                  flush=True)

        # 周期同步冻结副本（原版 WeightsCopyingCallback 思路）
        if cfg.solo_copy_every and step % cfg.solo_copy_every == 0:
            _sync_frozen_copy(main, opp)
            print(f"[solo] 冻结副本已同步 @step {step}", flush=True)

        if cfg.steps_per_eval and step % cfg.steps_per_eval == 0:
            eval_and_write(step)
            last_eval_step = step

    print(f"[solo] 训练循环耗时 {time.monotonic() - _t0:.1f}s", flush=True)
    save_checkpoint(main, cfg.solo_main_path())
    if last_eval_step != cfg.total_steps:
        eval_and_write(cfg.total_steps)
    print(f"[done] solo '{cfg.name}' 完成，产物在 {cfg.folder()} "
          f"（solo_state.json 供 dashboard --solo 读取）")


if __name__ == "__main__":
    from rl.config import TrainConfig
    run_solo(TrainConfig.resolve("standard"))
