"""全配对分流派联赛训练（flow league，P-flow）。

用户方案（P-flow-1）：
- 6 个模型全部为**可训练 PPO**：main / push_flow / counter_flow / lockdown_flow /
  all_decks / random_deck，每个 FollowerPolicy + 独立 PPOTrainer；
- 6 个卡组池两两全配对（C(6,2)=15 对），每副卡组 vs 每副卡组打 1 局、不换边：
  推进 60 × 防反 120 × 自闭 20 × 全量 200 × 随机 30(每次训练生成) × main 200
  → 一次训练 = Σ|pool_i|×|pool_j| = 148,800 局；
- **每对（pair）数据只喂该对双方模型**（不跨对混合，满足"所有用了推进卡组的对局
  数据给推进和全量训练"的流式语义）；
- 对内流式：obs["grid"] 每条约 32KB，整对收集会爆内存，故攒够 update_interval 条
  即更新该对双方模型再丢弃（"每对打完即训"落地为对内流式更新，语义一致）；
- player-1 侧轨迹由 FollowerOpponent.take_last_step() 收集；player-1 的 reward 用
  compute_reward 交换 blue/red 视角镜像计算（invalid_count 视为 0，FollowerPolicy
  从掩码采样一般合法）；
- **按模型奖惩**：每个模型用自己的 reward_weights（config.MODEL_REWARD_OVERRIDES
  在所选预设之上按模型覆盖：推进加码费差 / 防反减码 / 自闭压到≈0；main/all/random
  同一基线），同一对局内 A 用 rw_a 算 reward0、B 用 rw_b 镜像算 reward1。

数据归属（on-policy，每模型只用自己作为对局一方时的轨迹）：
- 推进 34,200 局 / 防反 61,200 / 自闭 12,200 / 全量 86,000 / 随机 18,000 /
  main 86,000（main 用全量 200 套池，与全部 5 个对手池对战）。
"""

import os
import sys
import json
import math
import time
import random
from collections import OrderedDict

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import numpy as np
import torch

from rl.env_wrapper import RLEnv, compute_reward, _phase_weights
from rl.belief import BeliefInference
from rl.belief_planner import BeliefPlanner
from rl.follower import FollowerPolicy, load_checkpoint, save_checkpoint
from rl.plan_space import PLAN_DIM
from rl.train_follower import FollowerOpponent
from rl.ppo import PPOTrainer
from rl.prophet import ProphetPlanner
from rl.opponents import build_card_pool, sample_deck
from rl.decks import load_classified_decks, decks_by_archetype
from rl.config import reward_to_env, model_reward_weights
from rl.run_league import (_bundle_cards, resolve_device, eval_round_robin,
                           overtime_open)
from rl.league import League

#: 6 个可训练模型 id（顺序决定对局矩阵的排列）
FLOW_MODEL_IDS = ["push_flow", "counter_flow", "lockdown_flow",
                  "all_decks", "random_deck", "main"]

#: 训练中先知规划注入概率（与 run_league 主训练一致）
_PROPHET_PROB = 0.3


# ---------------------------------------------------------------------------
# 卡组池
# ---------------------------------------------------------------------------

def build_flow_pools(cfg, n_random_decks=30):
    """构建 6 个卡组池。返回 OrderedDict[id -> (label, [deck,...])]。

    - 推进/防反/自闭/全量：来自三分类卡组数据集（60/120/20/200 套）；
    - 完全随机：每次训练用引擎卡池生成 n_random_decks 套（默认 30）；
    - main：用全量 200 套卡组池（用户确认，与全量模型同池不同策略）。
    """
    decks = load_classified_decks(cfg.decks_path)
    by = decks_by_archetype(decks)
    rng = random.Random(int(cfg.seed) + 999)
    pool_cards = build_card_pool()
    random_decks = [{"archetype": "完全随机", "cards": sample_deck(rng, pool_cards), "missing": 0}
                    for _ in range(int(n_random_decks))]
    return OrderedDict([
        ("push_flow", ("推进流", by["推进流"])),
        ("counter_flow", ("防守反击流", by["防守反击流"])),
        ("lockdown_flow", ("自闭流", by["自闭流"])),
        ("all_decks", ("全量卡组", decks)),
        ("random_deck", (f"完全随机×{n_random_decks}", random_decks)),
        ("main", ("全量卡组(main)", decks)),
    ])


def flow_pair_games(pools):
    """按全配对规则统计一次训练的总对局数（C(6,2)=15 对 Σ|pool_i|×|pool_j|）。"""
    ids = list(pools.keys())
    total = 0
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            total += len(pools[ids[i]][1]) * len(pools[ids[j]][1])
    return total


def scale_pools(pools, factor):
    """把每个卡组池按 factor 缩小（每池截断到 max(1, round(len×factor))）。

    用于"降低一个数量级"的数据效率验证（factor=0.1：60→6 / 120→12 / 20→2 /
    200→20 / 30→3 / 200→20 → 一次训练 1,488 局，为真实 148,800 的 1/100）。
    抽样用固定种子，保证可复现。
    """
    rng = random.Random(20240903)
    out = OrderedDict()
    for aid, (label, decks) in pools.items():
        n = max(1, int(round(len(decks) * factor)))
        if n < len(decks):
            idx = sorted(rng.sample(range(len(decks)), n))
            sel = [decks[i] for i in idx]
        else:
            sel = list(decks)
        out[aid] = (label, sel)
    return out


# ---------------------------------------------------------------------------
# 模型 / 训练器
# ---------------------------------------------------------------------------

def build_flow_models(cfg, device, belief_dim):
    """构建 6 个 FollowerPolicy + PPOTrainer。main 可 --main-init 预训练初始化。"""
    models = {}
    trainers = {}
    for mid in FLOW_MODEL_IDS:
        if mid == "main" and cfg.main_init:
            pol = load_checkpoint(cfg.main_init, hidden_dim=cfg.hidden_dim)
        else:
            pol = FollowerPolicy(hidden=cfg.hidden_dim, plan_dim=PLAN_DIM,
                                 belief_dim=belief_dim)
        pol.to_device(device)
        models[mid] = pol
        trainers[mid] = PPOTrainer(pol, lr=cfg.lr, gamma=cfg.gamma,
                                   gae_lambda=cfg.gae_lambda, clip=cfg.clip,
                                   vf_coef=cfg.vf_coef, ent_coef=cfg.ent_coef,
                                   max_grad_norm=cfg.max_grad_norm,
                                   adv_norm=cfg.adv_norm)
    return models, trainers


def save_flow_models(cfg, models):
    """把 6 个模型保存到 cfg.folder()/flow_<id>.pt。"""
    for mid, pol in models.items():
        save_checkpoint(pol, os.path.join(cfg.folder(), f"flow_{mid}.pt"))


def _flow_progress_path(cfg):
    return os.path.join(cfg.folder(), "flow_run_state.json")


def _save_flow_progress(cfg, pair_ix, game_ix, total_games, trainers):
    """落盘 flow 断点：对进度（pair_ix/game_ix）+ 6 个优化器状态（flow_opt_<id>.pt）。"""
    with open(_flow_progress_path(cfg), "w", encoding="utf-8") as f:
        json.dump({"pair_ix": int(pair_ix), "game_ix": int(game_ix),
                   "total_games": int(total_games)}, f, ensure_ascii=False, indent=2)
    for mid, tr in trainers.items():
        torch.save(tr.opt.state_dict(), os.path.join(cfg.folder(), f"flow_opt_{mid}.pt"))


def _load_flow_resume(cfg, device):
    """resume 时从磁盘恢复 (models, trainers, pair_start, game_ix_start)；失败返回 None。"""
    pp = _flow_progress_path(cfg)
    if not os.path.exists(pp):
        return None
    try:
        with open(pp, "r", encoding="utf-8") as f:
            prog = json.load(f)
        pair_start = int(prog.get("pair_ix", 0))
        game_ix_start = int(prog.get("game_ix", 0))
    except (OSError, ValueError):
        return None
    loaded = {}
    for mid in FLOW_MODEL_IDS:
        p = os.path.join(cfg.folder(), f"flow_{mid}.pt")
        if os.path.exists(p):
            pol = load_checkpoint(p, hidden_dim=cfg.hidden_dim)
            pol.to_device(device)
            loaded[mid] = pol
    if len(loaded) != len(FLOW_MODEL_IDS):
        return None
    trainers = {}
    for mid, pol in loaded.items():
        t = PPOTrainer(pol, lr=cfg.lr, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda,
                       clip=cfg.clip, vf_coef=cfg.vf_coef, ent_coef=cfg.ent_coef,
                       max_grad_norm=cfg.max_grad_norm, adv_norm=cfg.adv_norm)
        op = os.path.join(cfg.folder(), f"flow_opt_{mid}.pt")
        if os.path.exists(op):
            try:
                t.opt.load_state_dict(torch.load(op, map_location=device))
            except Exception:
                pass   # 优化器不匹配则 Adam 从头，模型仍续训
        trainers[mid] = t
    return loaded, trainers, pair_start, game_ix_start


# ---------------------------------------------------------------------------
# 对局 / 轨迹收集
# ---------------------------------------------------------------------------

def _hp_state(p):
    """(总塔血, 剩余皇冠数, 圣水)，供双视角 reward 镜像计算。"""
    return (p.king_tower_hp + p.left_tower_hp + p.right_tower_hp,
            3 - p.get_crown_count(),
            p.elixir)


def new_ep_buf():
    return {"obs": [], "belief": [], "plan": [], "bundle": [], "lp": [], "val": [],
            "rew": [], "term": [], "trunc": [], "masks": [], "init": []}


def _flush_episode(buf, ep, policy, last_obs, last_belief, last_plan, last_hidden,
                   truncated, cfg):
    """整局 GAE 后把 transition 追加进 buf（与 run_league 主训练一致）。"""
    # P1-7 修复：env 恒返回 trunc=False，截断末步须显式标记 bootstrap 才生效
    if truncated and ep["trunc"]:
        ep["trunc"][-1] = True
        last_val = policy.value(last_obs, last_belief, last_plan, last_hidden)
    else:
        last_val = 0.0
    adv, ret = PPOTrainer.compute_gae(ep["rew"], ep["val"], ep["term"],
                                      cfg.gamma, cfg.gae_lambda,
                                      truncated=ep["trunc"], last_value=last_val)
    for i in range(len(ep["rew"])):
        buf.append({"obs": ep["obs"][i], "belief": ep["belief"][i], "plan": ep["plan"][i],
                    "bundle": ep["bundle"][i], "old_logprob": ep["lp"][i],
                    "adv": float(adv[i]), "returns": float(ret[i]),
                    "masks": ep["masks"][i], "init_hidden": ep["init"][i]})


def _drain(buf, trainer, batch_size):
    """按 batch_size 消费 buf 并训练（流式，释放内存）。"""
    while len(buf) >= batch_size:
        trainer.update(buf[:batch_size])
        del buf[:batch_size]
    if buf:
        trainer.update(buf)
        buf.clear()


def _play_one(env, pol_a, pol_b, deckA, deckB, cfg, seed, max_steps,
              buf_a, buf_b, bp, prophet, rng, rw_a, rw_b):
    """deckA→pol_a(player-0)，deckB→pol_b(player-1)，打 1 局不换边。

    双侧都采探索轨迹：player-0 侧外部循环收集，player-1 侧由
    FollowerOpponent.take_last_step() 收集（reward 用 compute_reward 交换视角镜像）。
    rw_a/rw_b：A/B 各自的 reward_weights（按模型奖惩）；reward0 用 rw_a、reward1 用 rw_b。
    返回 winner（0/1/None）。
    """
    env.deck0 = list(deckA["cards"]) if isinstance(deckA, dict) else list(deckA)
    env.deck1 = list(deckB["cards"]) if isinstance(deckB, dict) else list(deckB)
    env.deck0_factory = None
    env.deck1_factory = None
    belief_a = BeliefInference(opp_deck=env.deck1, n_particles=128, seed=seed)
    opp = FollowerOpponent(pol_b, env,
                           belief=BeliefInference(opp_deck=env.deck0,
                                                  n_particles=128, seed=seed + 1),
                           deterministic=False)
    env.opponent = opp
    obs, _ = env.reset()
    belief_a.reset(env.deck1)
    hidden_a = None
    ep_a = new_ep_buf()
    ep_b = new_ep_buf()
    last_tr1 = None
    last_belief_tok = None
    last_plan_tok = None
    done = False
    steps = 0
    while not done and (steps < max_steps or overtime_open(env.battle)):
        use_prophet = rng.random() < _PROPHET_PROB
        plan = (prophet.plan(env.get_prophet_state()) if use_prophet
                else bp.plan(env.battle, belief_a.state(), obs))
        plan_tok = plan.to_vector()
        belief_tok = belief_a.encode(obs, None)
        bundle, lp, val, hidden_a, masks = pol_a.act(
            obs, belief_tok, plan_tok, env.get_action_mask,
            hidden=hidden_a, deterministic=False)
        a_played = _bundle_cards(bundle, obs)
        old0 = _hp_state(env.battle.players[0])
        old1 = _hp_state(env.battle.players[1])
        v_before = [float(env._active_v[0]), float(env._active_v[1])]
        env.reward_weights = rw_a          # 按模型奖惩：reward0 用 A 的权重
        obs2, reward0, term, trunc, info = env.step(bundle)
        tr1 = opp.take_last_step()
        new0 = _hp_state(env.battle.players[0])
        new1 = _hp_state(env.battle.players[1])
        winner = env.battle.winner if env.battle.game_over else None
        # player-1 视角 reward：交换 blue/red、winner 翻转（invalid 视为 0），用 B 的权重
        # v2：与 RLEnv.step 同口径——两段价格 + 资源账 V（份额在 env 内维护）
        v_after = info.get("field_v") or [env._active_v[0], env._active_v[1]]
        tower_b, edw_b = _phase_weights(rw_b, env.battle.time)
        rw_b2 = dict(rw_b)
        rw_b2["tower_dmg_opp"] = tower_b
        rw_b2["tower_dmg_self"] = tower_b
        rw_b2["elixir_diff_weight"] = edw_b
        reward1 = compute_reward(
            rw_b2,
            blue_hps_old=old1[0], red_hps_old=old0[0],
            blue_hps_new=new1[0], red_hps_new=new0[0],
            blue_left_old=old1[1], red_left_old=old0[1],
            blue_left_new=new1[1], red_left_new=new0[1],
            my_elixir_before=old1[2], opp_elixir_before=old0[2],
            my_elixir_after=new1[2], opp_elixir_after=new0[2],
            my_v_before=v_before[1], opp_v_before=v_before[0],
            my_v_after=v_after[1], opp_v_after=v_after[0],
            winner=(1 if winner == 1 else (0 if winner == 0 else None)),
            invalid_count=0,
            blue_hps_max=getattr(env, "_red_hps_max", None),
            red_hps_max=getattr(env, "_blue_hps_max", None),
            game_over=env.battle.game_over)
        # player-0 侧 transition
        ep_a["obs"].append(obs); ep_a["belief"].append(belief_tok)
        ep_a["plan"].append(plan_tok); ep_a["bundle"].append(bundle)
        ep_a["lp"].append(lp); ep_a["val"].append(val)
        ep_a["masks"].append(masks); ep_a["init"].append(hidden_a)
        ep_a["rew"].append(reward0); ep_a["term"].append(term); ep_a["trunc"].append(trunc)
        # player-1 侧 transition
        ep_b["obs"].append(tr1["obs"]); ep_b["belief"].append(tr1["belief"])
        ep_b["plan"].append(tr1["plan"]); ep_b["bundle"].append(tr1["bundle"])
        ep_b["lp"].append(tr1["lp"]); ep_b["val"].append(tr1["val"])
        ep_b["masks"].append(tr1["masks"]); ep_b["init"].append(tr1["hidden"])
        ep_b["rew"].append(reward1); ep_b["term"].append(term); ep_b["trunc"].append(trunc)
        last_tr1 = tr1
        last_belief_tok = belief_tok
        last_plan_tok = plan_tok
        belief_a.update(obs2, info.get("opp_played"))
        opp.observe_opponent_played(a_played)
        obs = obs2
        done = term or trunc
        steps += 1
    truncated = (not term) and (len(ep_a["rew"]) >= max_steps)
    _flush_episode(buf_a, ep_a, pol_a, obs, last_belief_tok, last_plan_tok, hidden_a,
                   truncated, cfg)
    _flush_episode(buf_b, ep_b, pol_b,
                   last_tr1["obs"], last_tr1["belief"], last_tr1["plan"],
                   last_tr1["hidden"], truncated, cfg)
    return winner


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------

def run_flow(cfg, resume=False, n_random_decks=30, pools=None, max_pairs=None,
             games_per_deck_pair=1, save=True, seed=None, quiet=False):
    """全配对分流派联赛主循环。

    参数：
    - pools: 显式注入卡组池（selftest 用 mini 池）；缺省 build_flow_pools()。
    - max_pairs: 只跑前 N 对（试跑用）；缺省全部 15 对。
    - games_per_deck_pair: 每副 deckA×deckB 打的局数（>1 = 数据效率实验"对局翻倍"）。
    - save: 是否每对/结束时落盘 checkpoint（sweep 多轮训练时关掉省 IO）。
    - seed: 覆盖 cfg.seed 的本轮种子（sweep 每轮换种子，避免 20 轮全同）。
    - quiet: 抑制逐对进度打印。
    返回 (total_games, models, trainers)。
    """
    device = resolve_device(cfg.device)
    cfg.ensure_dirs()
    cfg.save()
    seed = cfg.seed if seed is None else seed
    print(f"[flow] 全配对分流派联赛 配置 '{cfg.name}' -> {cfg.folder()} "
          f"(device={device}, seed={seed}, 每对局数={games_per_deck_pair})", flush=True)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    pools = pools or build_flow_pools(cfg, n_random_decks=n_random_decks)
    ids = list(pools.keys())
    env = RLEnv(opponent=None, seed=seed, reward_weights=reward_to_env(cfg),
                card_level=cfg.card_level)
    model_rewards = {mid: model_reward_weights(mid, cfg) for mid in FLOW_MODEL_IDS}
    _rw = model_rewards
    print(f"[flow] 按模型奖惩（覆盖预设 '{cfg.name}'）："
          f"main/all/random 费差{_rw['main']['elixir_diff_weight']} "
          f"推进{_rw['push_flow']['elixir_diff_weight']} "
          f"防反{_rw['counter_flow']['elixir_diff_weight']} "
          f"自闭{_rw['lockdown_flow']['elixir_diff_weight']} "
          f"（1圣水≈{0.001 / _rw['main']['elixir_diff_weight']:.0f}血 @lv11, 塔血统一 "
          f"{_rw['main']['tower_dmg_opp']}/{_rw['main']['tower_dmg_self']}）", flush=True)
    belief_dim = len(BeliefInference(opp_deck=env.deck1, n_particles=128,
                                     seed=0).encode(None, None))
    models = trainers = None
    pair_start = game_ix_start = 0
    if resume:
        _r = _load_flow_resume(cfg, device)
        if _r is not None:
            models, trainers, pair_start, game_ix_start = _r
            print(f"[flow] resume 从第 {pair_start} 对继续（6 模型+优化器已加载）", flush=True)
        else:
            print("[flow] resume 检查点缺失/不完整，从头开始", flush=True)
    if models is None:
        models, trainers = build_flow_models(cfg, device, belief_dim)
    bp = BeliefPlanner()
    prophet = ProphetPlanner()
    rng = random.Random(seed)
    max_steps = int(cfg.max_ep_steps)
    total_games = 0
    t0 = time.monotonic()

    pair_ix = pair_start
    game_ix = game_ix_start
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if max_pairs is not None and pair_ix >= max_pairs:
                break
            id_a, (label_a, pool_a) = ids[i], pools[ids[i]]
            id_b, (label_b, pool_b) = ids[j], pools[ids[j]]
            n_pair = len(pool_a) * len(pool_b) * int(games_per_deck_pair)
            if pair_ix < pair_start:
                # 断点前已完成的对：只累计局数，不重打
                total_games += n_pair
                pair_ix += 1
                continue
            if not quiet:
                print(f"[flow] 对 {label_a}({len(pool_a)}) × {label_b}({len(pool_b)}) "
                      f"= {n_pair} 局 ...", flush=True)
            buf_a, buf_b = [], []
            for deckA in pool_a:
                for deckB in pool_b:
                    for _k in range(int(games_per_deck_pair)):
                        _play_one(env, models[id_a], models[id_b], deckA, deckB,
                                  cfg, seed + game_ix, max_steps,
                                  buf_a, buf_b, bp, prophet, rng,
                                  model_rewards[id_a], model_rewards[id_b])
                        game_ix += 1
                        if len(buf_a) >= cfg.update_interval:
                            _drain(buf_a, trainers[id_a], cfg.batch_size)
                        if len(buf_b) >= cfg.update_interval:
                            _drain(buf_b, trainers[id_b], cfg.batch_size)
            _drain(buf_a, trainers[id_a], cfg.batch_size)
            _drain(buf_b, trainers[id_b], cfg.batch_size)
            total_games += n_pair
            pair_ix += 1
            if not quiet:
                print(f"[flow] {label_a} × {label_b} 完成 {n_pair} 局（累计 {total_games}），"
                      f"耗时 {time.monotonic() - t0:.1f}s", flush=True)
            if save:
                save_flow_models(cfg, models)
                _save_flow_progress(cfg, pair_ix, game_ix, total_games, trainers)
    if save:
        save_flow_models(cfg, models)
    print(f"[done] flow 联赛完成 {total_games} 局，6 模型已保存 -> {cfg.folder()}", flush=True)
    return total_games, models, trainers


# ---------------------------------------------------------------------------
# 数据效率 A/B（缩小 10× 池，验证曲线上涨再上全规模）
# ---------------------------------------------------------------------------

SWEEP_STRATEGIES = {
    "stream": {"n_runs": 20, "games_per_pair": 1,
               "desc": "每对 1 局（忠实流式）× 20 次完整训练"},
    "games5": {"n_runs": 4, "games_per_pair": 5,
               "desc": "每对 5 局 × 4 次完整训练"},
}


def _sweep_trend(rows):
    """首 vs 末（当前累计）的 main 轮内估计趋势判定：|Δ|/SE≥2σ 才算上涨/下跌。"""
    first, last = rows[0]["main_est"], rows[-1]["main_est"]
    delta = last[0] - first[0]
    se_delta = math.hypot(first[1], last[1])
    z = delta / se_delta if se_delta > 0 else None
    verdict = ("上涨（≥2σ）" if z is not None and z >= 2
               else ("下跌（≤-2σ）" if z is not None and z <= -2 else "持平（噪声内）"))
    return {"first_main_est": first, "last_main_est": last,
            "delta": round(delta, 1), "delta_se": round(se_delta, 1),
            "z": round(z, 2) if z is not None else None, "verdict": verdict}


def _write_sweep(out_dir, spec, strategy, pool_scale, sizes, n_runs,
                 games_per_pair, per_run, total_budget, eval_games, device,
                 rows, status, elapsed_s):
    """把当前累计 rows 落盘 summary.json/csv（含进度字段；dashboard 实时轮询）。

    status: running=训练进行中（逐轮增量写，dashboard 显示进度）/ done=全部完成。
    """
    summary = {
        "strategy": strategy, "desc": spec["desc"],
        "pool_scale": pool_scale, "pool_sizes": sizes,
        "n_runs": n_runs, "games_per_pair": games_per_pair,
        "per_run_games": per_run, "total_games": total_budget,
        "eval_games_per_pair": eval_games, "device": device,
        "status": status,
        "run_current": len(rows),
        "elapsed_s": round(elapsed_s, 1),
        "eta_s": round(elapsed_s / max(1, len(rows)) * max(0, n_runs - len(rows)), 1)
                 if status == "running" else 0.0,
        "rows": rows,
        "trend": _sweep_trend(rows),
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "summary.csv"), "w", encoding="utf-8") as f:
        f.write("run,games,main_est,main_se")
        for mid in FLOW_MODEL_IDS:
            f.write(f",{mid}_est,{mid}_se")
        f.write("\n")
        for row in rows:
            f.write(f"{row['run']},{row['games']},{row['main_est'][0]},{row['main_est'][1]}")
            for mid in FLOW_MODEL_IDS:
                e = row["est"].get(mid)
                f.write(f",{e[0] if e else ''},{e[1] if e else ''}")
            f.write("\n")
    return summary


def run_flow_sweep(cfg, strategy="stream", n_runs=None, games_per_pair=None,
                   pool_scale=0.1, n_random_decks=30, pools=None, eval_games=None):
    """缩小 10× 卡组池的 flow 数据效率 A/B 实验（先验证曲线确实上涨再跑 148,800）。

    - ``stream``（--mode flow-sweep-stream）：每对 1 局、对内流式，**跑 20 次完整训练**；
    - ``games5``（--mode flow-sweep-games5）：每对 5 局（对局次数翻 5 倍），**跑 4 次**。
    两者总对局预算相同（20×1,488 = 4×7,440 ≈ 29,760 局），对比"多轮流式小步快跑"
    vs "单轮更多数据"哪种数据效率策略让曲线上涨。每次训练后对 6 个模型做一轮
    全配对换边评估（eval_games/对），记录**轮内聚合估计**（SE≈347.5/√N 噪声地板）。

    产出 -> ``runs/<name>/flow_sweep_<strategy>/``：
    - summary.json：**逐轮增量写**（每轮结束即更新，含 status/run_current/eta_s 进度字段，
      dashboard 可实时显示训练进度）+ 逐轮 main 强度 + 全模型估计 + 趋势判定（首 vs 末，Δ/SE）；
    - summary.csv：曲线数据（可直接画图）；
    - final_flow_<id>.pt：最后一轮 6 个模型。

    pools 可注入 mini 池（selftest 用）；返回 (rows, summary)。
    """
    if strategy not in SWEEP_STRATEGIES:
        raise ValueError(f"未知 sweep 策略 {strategy}，可用 {list(SWEEP_STRATEGIES)}")
    spec = SWEEP_STRATEGIES[strategy]
    n_runs = int(n_runs if n_runs is not None else spec["n_runs"])
    games_per_pair = int(games_per_pair if games_per_pair is not None else spec["games_per_pair"])
    eval_games = int(eval_games or 10)
    device = resolve_device(cfg.device)
    cfg.ensure_dirs()
    cfg.save()
    raw = pools if pools is not None else build_flow_pools(cfg, n_random_decks=n_random_decks)
    scaled = scale_pools(raw, pool_scale)
    per_run = flow_pair_games(scaled) * games_per_pair
    total_budget = per_run * n_runs
    out_dir = os.path.join(cfg.folder(), f"flow_sweep_{strategy}")
    os.makedirs(out_dir, exist_ok=True)
    sizes = {k: len(v[1]) for k, v in scaled.items()}
    print(f"[sweep] 策略={strategy}（{spec['desc']}） 池缩小×{pool_scale} -> {sizes} "
          f"每轮 {per_run} 局 × {n_runs} 轮 = {total_budget} 局", flush=True)

    rows = []
    t0 = time.monotonic()
    for run in range(n_runs):
        run_seed = cfg.seed + run * 10000
        total, models, _tr = run_flow(cfg, pools=scaled,
                                      games_per_deck_pair=games_per_pair,
                                      save=False, seed=run_seed, quiet=True)
        lg = League(seed=run_seed + 7)
        for mid in FLOW_MODEL_IDS:
            lg.add_agent(mid, kind="main" if mid == "main" else "baseline",
                         policy=models[mid])
        eval_round_robin(lg, eval_games, int(cfg.max_ep_steps), run_seed + 77, run,
                         only_vs_main=False, record=False)
        rs = lg.round_stats[-1]
        row = {"run": int(run), "games": int(total),
               "main_est": [float(x) for x in rs["est"]["main"]],
               "est": {k: [float(a), float(b)] for k, (a, b) in rs["est"].items()}}
        rows.append(row)
        r, se = row["main_est"]
        print(f"[sweep] run {run:02d}: main 轮内估计 {r:.0f}±{se:.0f} "
              f"({rs['games'].get('main', 0)} 评估局) [{time.monotonic() - t0:.1f}s]",
              flush=True)
        # 逐轮增量落盘（running 状态），dashboard 实时显示进度
        elapsed = time.monotonic() - t0
        summary = _write_sweep(out_dir, spec, strategy, pool_scale, sizes, n_runs,
                               games_per_pair, per_run, total_budget, eval_games,
                               device, rows,
                               "done" if run == n_runs - 1 else "running", elapsed)
        if run == n_runs - 1:
            for mid, pol in models.items():
                save_checkpoint(pol, os.path.join(out_dir, f"final_flow_{mid}.pt"))

    trend = summary["trend"]
    first, last = trend["first_main_est"], trend["last_main_est"]
    delta, se_delta, z = trend["delta"], trend["delta_se"], trend["z"]
    verdict = trend["verdict"]
    print(f"[sweep] 完成 {n_runs} 轮 × {per_run} 局 = {total_budget} 局，耗时 "
          f"{time.monotonic() - t0:.1f}s", flush=True)
    print(f"[sweep] 曲线判定：main {first[0]:.0f}±{first[1]:.0f} -> "
          f"{last[0]:.0f}±{last[1]:.0f}  Δ={delta:+.0f}±{se_delta:.0f}  z={z}σ  "
          f"=> {verdict}", flush=True)
    print(f"[sweep] 结果 -> {out_dir}/summary.json / summary.csv", flush=True)
    return rows, summary


if __name__ == "__main__":
    from rl.config import TrainConfig
    _cfg = TrainConfig.resolve("standard")
    run_flow(_cfg)
