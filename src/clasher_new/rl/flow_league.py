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
  从掩码采样一般合法）。

数据归属（on-policy，每模型只用自己作为对局一方时的轨迹）：
- 推进 34,200 局 / 防反 61,200 / 自闭 12,200 / 全量 86,000 / 随机 18,000 /
  main 86,000（main 用全量 200 套池，与全部 5 个对手池对战）。
"""

import os
import sys
import time
import random
from collections import OrderedDict

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import numpy as np
import torch

from rl.env_wrapper import RLEnv, compute_reward
from rl.belief import BeliefInference
from rl.belief_planner import BeliefPlanner
from rl.follower import FollowerPolicy, load_checkpoint, save_checkpoint
from rl.plan_space import PLAN_DIM
from rl.train_follower import FollowerOpponent
from rl.ppo import PPOTrainer
from rl.prophet import ProphetPlanner
from rl.opponents import build_card_pool, sample_deck
from rl.decks import load_classified_decks, decks_by_archetype
from rl.config import reward_to_env
from rl.run_league import _bundle_cards, resolve_device

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
                                   max_grad_norm=cfg.max_grad_norm)
    return models, trainers


def save_flow_models(cfg, models):
    """把 6 个模型保存到 cfg.folder()/flow_<id>.pt。"""
    for mid, pol in models.items():
        save_checkpoint(pol, os.path.join(cfg.folder(), f"flow_{mid}.pt"))


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
    last_val = policy.value(last_obs, last_belief, last_plan, last_hidden) if truncated else 0.0
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
              buf_a, buf_b, bp, prophet, rng):
    """deckA→pol_a(player-0)，deckB→pol_b(player-1)，打 1 局不换边。

    双侧都采探索轨迹：player-0 侧外部循环收集，player-1 侧由
    FollowerOpponent.take_last_step() 收集（reward 用 compute_reward 交换视角镜像）。
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
    while not done and steps < max_steps:
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
        obs2, reward0, term, trunc, info = env.step(bundle)
        tr1 = opp.take_last_step()
        new0 = _hp_state(env.battle.players[0])
        new1 = _hp_state(env.battle.players[1])
        winner = env.battle.winner if env.battle.game_over else None
        # player-1 视角 reward：交换 blue/red、winner 翻转（invalid 视为 0）
        reward1 = compute_reward(
            env.reward_weights,
            blue_hps_old=old1[0], red_hps_old=old0[0],
            blue_hps_new=new1[0], red_hps_new=new0[0],
            blue_left_old=old1[1], red_left_old=old0[1],
            blue_left_new=new1[1], red_left_new=new0[1],
            my_elixir_before=old1[2], opp_elixir_before=old0[2],
            my_elixir_after=new1[2], opp_elixir_after=new0[2],
            winner=(1 if winner == 1 else (0 if winner == 0 else None)),
            invalid_count=0,
            blue_hps_max=getattr(env, "_red_hps_max", None),
            red_hps_max=getattr(env, "_blue_hps_max", None))
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

def run_flow(cfg, resume=False, n_random_decks=30, pools=None, max_pairs=None):
    """全配对分流派联赛主循环。

    参数：
    - pools: 显式注入卡组池（selftest 用 mini 池）；缺省 build_flow_pools()。
    - max_pairs: 只跑前 N 对（试跑用）；缺省全部 15 对。
    返回 (total_games, models, trainers)。
    """
    device = resolve_device(cfg.device)
    cfg.ensure_dirs()
    cfg.save()
    print(f"[flow] 全配对分流派联赛 配置 '{cfg.name}' -> {cfg.folder()} "
          f"(device={device})", flush=True)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    pools = pools or build_flow_pools(cfg, n_random_decks=n_random_decks)
    ids = list(pools.keys())
    env = RLEnv(opponent=None, seed=cfg.seed, reward_weights=reward_to_env(cfg),
                card_level=cfg.card_level)
    belief_dim = len(BeliefInference(opp_deck=env.deck1, n_particles=128,
                                     seed=0).encode(None, None))
    models, trainers = build_flow_models(cfg, device, belief_dim)
    bp = BeliefPlanner()
    prophet = ProphetPlanner()
    rng = random.Random(cfg.seed)
    max_steps = int(cfg.max_ep_steps)
    total_games = 0
    t0 = time.monotonic()

    pair_ix = 0
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if max_pairs is not None and pair_ix >= max_pairs:
                break
            id_a, (label_a, pool_a) = ids[i], pools[ids[i]]
            id_b, (label_b, pool_b) = ids[j], pools[ids[j]]
            n_pair = len(pool_a) * len(pool_b)
            print(f"[flow] 对 {label_a}({len(pool_a)}) × {label_b}({len(pool_b)}) "
                  f"= {n_pair} 局 ...", flush=True)
            buf_a, buf_b = [], []
            g = 0
            for deckA in pool_a:
                for deckB in pool_b:
                    _play_one(env, models[id_a], models[id_b], deckA, deckB,
                              cfg, cfg.seed + g, max_steps,
                              buf_a, buf_b, bp, prophet, rng)
                    g += 1
                    if len(buf_a) >= cfg.update_interval:
                        _drain(buf_a, trainers[id_a], cfg.batch_size)
                    if len(buf_b) >= cfg.update_interval:
                        _drain(buf_b, trainers[id_b], cfg.batch_size)
            _drain(buf_a, trainers[id_a], cfg.batch_size)
            _drain(buf_b, trainers[id_b], cfg.batch_size)
            total_games += n_pair
            pair_ix += 1
            print(f"[flow] {label_a} × {label_b} 完成 {n_pair} 局（累计 {total_games}），"
                  f"耗时 {time.monotonic() - t0:.1f}s", flush=True)
            save_flow_models(cfg, models)
    save_flow_models(cfg, models)
    print(f"[done] flow 联赛完成 {total_games} 局，6 模型已保存 -> {cfg.folder()}", flush=True)
    return total_games, models, trainers


if __name__ == "__main__":
    from rl.config import TrainConfig
    _cfg = TrainConfig.resolve("standard")
    run_flow(_cfg)
