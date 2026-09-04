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
from rl.run_league import (resolve_device, _bundle_cards, LeagueGameRecorder,
                           _stall_probe, STALL_WINDOW, _load_run_state, timeout_winner)
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


def _draw_penalty(cfg) -> float:
    """平局惩罚（= 失败：平局不再免费）。缺省与 lose_penalty 相同。"""
    rw = reward_to_env(cfg)
    return float(rw.get("draw_penalty", rw.get("lose_penalty", 10.0)))


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
              record_replays=False, replays_dir=None, step=None, frozen_step=None):
    """main（deterministic）vs 冻结副本（deterministic）打 n_games。

    返回 (stats, replays)。replays 非空时以 league_<step>.pkl 落盘（复用 dashboard 回放）。
    step: main 当前训练步（录像文件步数）；frozen_step: 冻结副本最后同步时的训练步，
    两者一并写入每局 meta.steps，dashboard 对阵列显示 "main@<step> vs main@<frozen_step>"。
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
        rec = LeagueGameRecorder("main", "frozen_copy", "main", max_steps,
                                 steps=(step, frozen_step)) if record_replays else None
        done = False
        steps = 0
        ep_rew = 0.0
        stall_count = 0
        last_hp = None
        while not done and steps < max_steps:
            if steps % STALL_WINDOW == 0:
                early, last_hp, stall_count = _stall_probe(env, last_hp, stall_count)
                if early:
                    break   # 僵局判平，提前结束
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
        if w is None and not env.battle.game_over:
            # 僵局早停/截断早于引擎结算 → 皇冠差/塔血差已定胜负；真平才判平=失败
            virt = timeout_winner(env.battle)
            if virt is None:
                ep_rew -= _draw_penalty(cfg)
            else:
                w = virt
                rw = reward_to_env(cfg)
                ep_rew += float(rw["win_bonus"] if w == 0 else -rw["lose_penalty"])
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


def _eval_worker_main(worker_id, main_sd, opp_sd, games, env_kwargs,
                      seed_base, max_steps, n_particles, record, out_q):
    """并行评估 worker：独立进程打 games（全局游戏索引列表）里每局。

    战斗模拟是纯 Python（GIL），跨进程才能真正吃满多核。每 worker 自建 env+信念+策略
    （从主进程收 state_dict），逐局回传 (game_idx, winner, ep_rew, replay_or_None)。
    与串行 eval_solo 同种子等价：串行复用单个 env，reset() 会基于上一局牌序继续
    shuffle（belief 先验 = 上一局结束后的牌序），所以 worker 必须把 0..n_games-1 的
    reset 链全部走一遍（reset 本身极便宜），只打分配到的局，才能还原同一信念先验。
    """
    try:
        from rl.env_wrapper import RLEnv
        from rl.belief import BeliefInference
        from rl.belief_planner import BeliefPlanner
        from rl.follower import FollowerPolicy
        from rl.plan_space import PLAN_DIM
        from rl.train_follower import FollowerOpponent
        from rl.run_league import LeagueGameRecorder, _stall_probe, STALL_WINDOW, _bundle_cards

        # 16 进程 × 默认 16 线程 = 256 线程在 16 核上互相争抢（过订阅），
        # 战斗模拟是纯 Python 单线程、推理 batch 极小 → 每 worker 1 线程即可
        import torch as _torch
        _torch.set_num_threads(1)
        try:
            import numpy as _np
            _np.set_num_threads(1)
        except Exception:
            pass

        env = RLEnv(opponent=None, seed=worker_id + 777,
                    reward_weights=dict(env_kwargs.get("reward_weights") or {}),
                    card_level=env_kwargs.get("card_level"),
                    deck0=list(env_kwargs["deck0"]), deck1=list(env_kwargs["deck1"]))
        belief_dim = len(BeliefInference(opp_deck=env.deck1, n_particles=n_particles,
                                         seed=0).encode(None, None))
        main = FollowerPolicy(hidden=env_kwargs["hidden_dim"], plan_dim=PLAN_DIM,
                              belief_dim=belief_dim)
        opp = FollowerPolicy(hidden=env_kwargs["hidden_dim"], plan_dim=PLAN_DIM,
                             belief_dim=belief_dim)
        main.load_state_dict(main_sd)
        opp.load_state_dict(opp_sd)
        main.to_device("cpu")
        opp.to_device("cpu")
        bp = BeliefPlanner()
        n_total = int(env_kwargs["n_total"])
        do = set(int(g) for g in games)
        results = []
        for g in range(n_total):
            # 与串行 eval_solo 同序：先按当前 env.deck1（上一局牌序）构造信念，再 reset 换牌序
            opp_side = FollowerOpponent(
                opp, env,
                belief=BeliefInference(opp_deck=env.deck1, n_particles=n_particles,
                                       seed=seed_base + g),
                deterministic=True)
            belief = BeliefInference(opp_deck=env.deck1, n_particles=n_particles,
                                     seed=seed_base + 1000 + g)
            obs, _ = env.reset(seed=seed_base + 2000 + g)
            belief.reset(env.deck1)
            if g not in do:
                continue
            env.opponent = opp_side
            hidden = None
            rec = LeagueGameRecorder("main", "frozen_copy", "main", max_steps,
                                     steps=(env_kwargs.get("eval_step"),
                                            env_kwargs.get("frozen_step"))) if record else None
            done = False
            steps = 0
            ep_rew = 0.0
            stall_count = 0
            last_hp = None
            while not done and steps < max_steps:
                if steps % STALL_WINDOW == 0:
                    early, last_hp, stall_count = _stall_probe(env, last_hp, stall_count)
                    if early:
                        break
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
            if w is None and not env.battle.game_over:
                # 与串行 eval_solo 一致：早停/截断补结算（皇冠差/塔血差→胜负，真平→平局=失败）
                virt = timeout_winner(env.battle)
                rw = env_kwargs.get("reward_weights") or {}
                if virt is None:
                    ep_rew -= float(rw.get("draw_penalty", rw.get("lose_penalty", 10.0)))
                else:
                    w = virt
                    ep_rew += float(rw["win_bonus"] if w == 0 else -rw["lose_penalty"])
            results.append((g, w, ep_rew, rec.done(w) if rec is not None else None))
        out_q.put(("result", results))
    except Exception as e:
        import traceback
        try:
            out_q.put(("error", "%r\n%s" % (e, traceback.format_exc())))
        except Exception:
            pass


def eval_solo_parallel(env, main, opp, n_games, max_steps, seed, cfg,
                       n_workers=8, record_replays=False, replays_dir=None, step=None,
                       frozen_step=None):
    """eval_solo 的进程池并行版：n_games 局均分到 n_workers 个 spawn 进程打。

    串行 16 局≈126s（主进程单核跑纯 Python 模拟）；16 进程理论 ≈ 126/16 + spawn/import
    开销 ≈ 15-25s。统计与串行同公式（wins/losses/draws + mean_reward + SE）。
    """
    import multiprocessing as mp
    n_workers = max(1, min(int(n_workers), int(n_games)))
    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    main_sd = {k: v.detach().cpu() for k, v in main.state_dict().items()}
    opp_sd = {k: v.detach().cpu() for k, v in opp.state_dict().items()}
    games = list(range(int(n_games)))
    chunks = [games[i::n_workers] for i in range(n_workers)]
    env_kwargs = {"reward_weights": reward_to_env(cfg), "card_level": cfg.card_level,
                  "deck0": list(env.deck0), "deck1": list(env.deck1),
                  "hidden_dim": int(cfg.hidden_dim), "n_total": int(n_games),
                  "eval_step": step, "frozen_step": frozen_step}
    procs = []
    for wid, chunk in enumerate(chunks):
        if not chunk:
            continue
        p = ctx.Process(target=_eval_worker_main,
                        args=(wid, main_sd, opp_sd, chunk, env_kwargs,
                              int(seed), int(max_steps), 128, bool(record_replays), out_q))
        p.start()
        procs.append(p)
    results = []
    for _ in procs:
        msg = out_q.get()
        if msg[0] == "error":
            for p in procs:
                p.terminate()
            raise RuntimeError(f"并行评估 worker 失败: {msg[1]}")
        results.extend(msg[1])
    for p in procs:
        p.join(timeout=10)
    results.sort(key=lambda r: r[0])   # 按游戏索引还原顺序
    wins = losses = draws = 0
    rew_sum = 0.0
    replays = []
    for _g, w, ep_rew, rec in results:
        if w == 0:
            wins += 1
        elif w == 1:
            losses += 1
        else:
            draws += 1
        rew_sum += ep_rew
        if rec is not None:
            replays.append(rec)
    n = max(1, int(n_games))
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
    _t_start = time.monotonic()
    device = resolve_device(cfg.device)   # 首次调用即触发 torch CUDA 上下文初始化
    _t_cuda = time.monotonic()
    cfg.ensure_dirs()
    cfg.save()
    print(f"[solo] 单人自对弈 配置 '{cfg.name}' -> {cfg.folder()} "
          f"(device={device}, seed={cfg.seed}, 固定卡组 {len(DEFAULT_SOLO_DECK)} 卡镜像, "
          f"冻结副本同步间隔={cfg.solo_copy_every})", flush=True)
    _t_cfg = time.monotonic()
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    env = solo_env(cfg, cfg.seed)
    _t_env = time.monotonic()
    belief_dim = len(BeliefInference(opp_deck=env.deck1, n_particles=128,
                                     seed=0).encode(None, None))
    # —— 断点续练（--resume）：恢复 step / main 权重 / 优化器 / 历史曲线 ——
    start_step = 0
    history = []
    rs = None
    if resume:
        rs = _load_run_state(cfg)
        if rs and rs.get("solo_ckpt") and os.path.exists(rs["solo_ckpt"]):
            start_step = int(rs.get("step", 0))
            if os.path.exists(cfg.solo_state_path()):
                try:
                    with open(cfg.solo_state_path(), "r", encoding="utf-8") as f:
                        history = (json.load(f) or {}).get("history", [])
                except (OSError, ValueError):
                    history = []
        else:
            print("[solo] resume 检查点缺失，从头开始", flush=True)

    if cfg.main_init and not (rs and rs.get("solo_ckpt")
                              and os.path.exists(rs["solo_ckpt"])):
        main = load_checkpoint(cfg.main_init, hidden_dim=cfg.hidden_dim)
    else:
        main = FollowerPolicy(hidden=cfg.hidden_dim, plan_dim=PLAN_DIM,
                              belief_dim=belief_dim)
    main.to_device(device)
    if rs and rs.get("solo_ckpt") and os.path.exists(rs["solo_ckpt"]):
        # resume：断点权重为准（覆盖 main_init）
        main = load_checkpoint(rs["solo_ckpt"], hidden_dim=cfg.hidden_dim)
        main.to_device(device)
        print(f"[solo] resume 从 step {start_step} 续训（继续到 {cfg.total_steps}）", flush=True)
    opp = FollowerPolicy(hidden=cfg.hidden_dim, plan_dim=PLAN_DIM, belief_dim=belief_dim)
    opp.to_device(device)
    _sync_frozen_copy(main, opp)   # 开局副本 = main（resume 后即断点权重）
    frozen_step = start_step       # 冻结副本当前所在训练步（录像 meta.steps 用）
    ppo = PPOTrainer(main, lr=cfg.lr, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda,
                     clip=cfg.clip, vf_coef=cfg.vf_coef, ent_coef=cfg.ent_coef,
                     max_grad_norm=cfg.max_grad_norm, adv_norm=cfg.adv_norm)
    if rs and rs.get("solo_opt") and os.path.exists(rs["solo_opt"]):
        try:
            ppo.opt.load_state_dict(torch.load(rs["solo_opt"], map_location=device))
        except Exception:
            print("[solo] resume 优化器状态不匹配，Adam 从头（模型仍续训）", flush=True)
    bp = BeliefPlanner()
    prophet = ProphetPlanner()
    rng = random.Random(cfg.seed)

    opp_side = FollowerOpponent(opp, env,
                                belief=BeliefInference(opp_deck=env.deck1,
                                                       n_particles=128, seed=cfg.seed),
                                deterministic=True)
    env.opponent = opp_side
    belief = BeliefInference(opp_deck=env.deck1, n_particles=128, seed=cfg.seed)
    _t_policy = time.monotonic()

    def eval_and_write(step):
        if int(cfg.eval_workers) > 1:
            stats, _ = eval_solo_parallel(env, main, opp, int(cfg.n_eval_games),
                                          int(cfg.max_ep_steps), cfg.seed + step, cfg,
                                          n_workers=int(cfg.eval_workers),
                                          record_replays=record_replays,
                                          replays_dir=cfg.replays_dir(), step=step,
                                          frozen_step=frozen_step)
        else:
            stats, _ = eval_solo(env, main, opp, int(cfg.n_eval_games),
                                 int(cfg.max_ep_steps), cfg.seed + step, cfg,
                                 record_replays=record_replays,
                                 replays_dir=cfg.replays_dir(), step=step,
                                 frozen_step=frozen_step)
        history.append(stats)
        write_solo_state(cfg.solo_state_path(), cfg, history, step,
                         status="done" if step >= cfg.total_steps else "running")
        print(f"[solo] eval@{step}: 胜率 {stats['winrate']:.3f}±{stats['winrate_se']:.3f} "
              f"({stats['wins']}W/{stats['losses']}L/{stats['draws']}D, "
              f"{stats['games']}局) mean_reward={stats['mean_reward']:.3f}", flush=True)
        save_checkpoint(main, cfg.solo_main_path())
        save_checkpoint(main, cfg.solo_ckpt_path(step))   # 历史版本保留（solo_main_<step>.pt）
        torch.save(ppo.opt.state_dict(), cfg.solo_opt_path())   # 断点续练恢复 Adam
        with open(cfg.run_state_path(), "w", encoding="utf-8") as f:
            json.dump({"step": int(step), "solo_ckpt": cfg.solo_main_path(),
                       "solo_opt": cfg.solo_opt_path(),
                       "config": cfg.name, "device": device}, f)

    # 训练开始先跑一次评估（WebUI 立即有真实数据）；resume 时不重跑起始评估
    if cfg.eval_at_start and start_step == 0:
        _t_eval0 = time.monotonic()
        eval_and_write(0)
        _t_eval1 = time.monotonic()
        print(f"[solo] 启动耗时分解: torch+CUDA init={_t_cuda-_t_start:.1f}s | "
              f"cfg/dirs={_t_cfg-_t_cuda:.1f}s | 环境+卡牌(BattleState/arena/卡池)="
              f"{_t_env-_t_cfg:.1f}s | 策略/信念/PPO={_t_policy-_t_env:.1f}s | "
              f"eval@0 {cfg.n_eval_games}局={_t_eval1-_t_eval0:.1f}s "
              f"(并行worker={cfg.eval_workers}) | "
              f"合计(A→B)={_t_eval1-_t_cfg:.1f}s", flush=True)

    obs, _ = env.reset()
    belief.reset(env.deck1)
    hidden = None
    last_eval_step = start_step
    ep_obs, ep_belief, ep_plan, ep_bundle, ep_lp, ep_val, ep_rew = [], [], [], [], [], [], []
    ep_term, ep_trunc, ep_masks, ep_init = [], [], [], []
    transitions = []
    stall_count = 0                 # 训练环僵局探针（与 eval 同语义）
    last_hp = None
    _t0 = time.monotonic()

    for step in range(start_step + 1, cfg.total_steps + 1):
        # —— 训练环僵局早停（纯 RL 修复）：连续 100 步双方塔血零变化 → 判平结束本局。
        # 否则躺平要拖满 max_ep_steps 才在 360 帧末罚一次 −10，(γλ)^k 视野内不可见，
        # “平局=失败”只修了终局、没修信用视野。与 eval 用同一个探针/判罚语义。
        if cfg.train_stall_stop and len(ep_rew) and len(ep_rew) % STALL_WINDOW == 0:
            early, last_hp, stall_count = _stall_probe(env, last_hp, stall_count)
            if early:
                if env.battle.winner is None and not env.battle.game_over and ep_rew:
                    # 早停补结算：皇冠差/塔血差已分胜负 → 终端胜负；真平才判平=失败
                    virt = timeout_winner(env.battle)
                    rw = reward_to_env(cfg)
                    if virt == 0:
                        ep_rew[-1] += float(rw["win_bonus"])
                    elif virt == 1:
                        ep_rew[-1] -= float(rw["lose_penalty"])
                    else:
                        ep_rew[-1] -= _draw_penalty(cfg)
                adv, ret = PPOTrainer.compute_gae(
                    ep_rew, ep_val, ep_term, cfg.gamma, cfg.gae_lambda,
                    truncated=ep_trunc, last_value=0.0)
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
                stall_count, last_hp = 0, None
                continue
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
            # 平局=失败：对局以无胜者结束（僵局/截断）→ 先补到期结算（皇冠差/塔血差
            # 已分胜负就给终端胜负奖励），真平才按平局=失败惩罚
            virt = None
            if env.battle.winner is None and not env.battle.game_over and ep_rew:
                virt = timeout_winner(env.battle)
                rw = reward_to_env(cfg)
                if virt == 0:
                    ep_rew[-1] += float(rw["win_bonus"])
                elif virt == 1:
                    ep_rew[-1] -= float(rw["lose_penalty"])
                else:
                    ep_rew[-1] -= _draw_penalty(cfg)
            # P1-7 修复：env 恒返回 trunc=False，若不显式标记，截断局会被 compute_gae
            # 当成普通终止（next_val=0），last_value bootstrap 从未生效——躺平拖满 360
            # 帧时最后一步 δ 巨大而前面 ~300 帧毫无信号。达步数上限、非终局且未虚拟
            # 判出胜负（真平需继续）→ 标记末步用 last_value bootstrap。
            if truncated and virt is None:
                ep_trunc[-1] = True
                last_val = main.value(obs, belief_tok, plan_vec, hidden)
            else:
                last_val = 0.0
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
            stall_count, last_hp = 0, None

        if len(transitions) >= cfg.update_interval:
            batch = (transitions[:cfg.batch_size] if len(transitions) > cfg.batch_size
                     else transitions)
            n_play = sum(1 for t in batch
                         if any(sa.kind == "deploy" for sa in t["bundle"].sub_actions))
            avg_size = sum(len(t["bundle"].sub_actions) for t in batch) / max(1, len(batch))
            stats = ppo.update(batch)
            transitions = transitions[cfg.batch_size:] if len(transitions) > cfg.batch_size else []
            print(f"[solo step {step}] policy={stats['policy_loss']:.4f} "
                  f"value={stats['value_loss']:.4f} entropy={stats['entropy']:.4f} "
                  f"| deploy={100.0 * n_play / len(batch):.1f}% bundle={avg_size:.2f} "
                  f"ratio={stats['ratio_mean']:.3f} clip={100.0 * stats['clip_frac']:.1f}% "
                  f"adv={stats['adv_mean']:+.3f}±{stats['adv_std']:.3f} "
                  f"gnorm={stats['grad_norm']:.2f} n={len(batch)}", flush=True)

        # 周期同步冻结副本（原版 WeightsCopyingCallback 思路）
        if cfg.solo_copy_every and step % cfg.solo_copy_every == 0:
            _sync_frozen_copy(main, opp)
            frozen_step = step
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
