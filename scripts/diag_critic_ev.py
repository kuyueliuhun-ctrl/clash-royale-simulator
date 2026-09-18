"""Critic EV 取证脚本（2026-09-11，§4 诊断）。

目的：判断 `explained_variance`（EV）持续为负，是

  (A) critic 真的太差（全局都预测不了回报），还是
  (B) 训练用 "128 帧连续批" 算 EV 造成的测量假象
      （同一局连续 128 帧回报高度同质 → 批内 Var(R) 很小 → EV=1-MSE/Var 天然为负）。

做法：用指定 checkpoint 在真环境里 rollout（与训练同超参），逐帧记录
`(value, reward, done, truncated)`，按训练同款 GAE 构造 return，然后离线同时算：

  1. EV_global        —— 全体帧池化（最诚实的"critic 到底会不会预测"）
  2. EV_batch_contig  —— 训练口径：每 128 连续帧一批，取均值（dashboard 上那个数）
  3. EV_batch_shuffle —— 随机分组 128 帧（打散批内同质性）
  4. EV_episode       —— 每局一批
  5. EV_between       —— 只用"每局平均 R"这一层（跨局可预测性）
  6. 批内/全局 Var(R) 与 value 误差分解
  7. corr(v,R)、回归斜率、按局聚合后的 corr

用法：
    python scripts/diag_critic_ev.py --ckpt runs/prod_200k_valnorm_ev/solo_main.pt \
        --games 24 --device cuda --out docs/_diag_ev.npz
"""

import argparse
import copy
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src", "clasher_new")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# T1-1 追加（2026-09-19）：UTF-8 兜底。本文件含 **GBK 编不出**的字符 ⇒ 无兜底时
# `print` 抛 UnicodeEncodeError（实测：test_m3_evo 因此产生 **10 个假失败**）。
# 用 T1-1 的**单一实现**；只用在入口脚本上（`rl/` 库模块不加 —— 库不该改宿主 stdout）。
from rl.io_bootstrap import force_utf8_stdout  # noqa: E402
force_utf8_stdout()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rl.config import TrainConfig, reward_to_env  # noqa: E402
from rl.belief import BeliefInference  # noqa: E402
from rl.belief_planner import BeliefPlanner  # noqa: E402
from rl.prophet import ProphetPlanner  # noqa: E402
from rl.plan_space import PLAN_DIM  # noqa: E402
from rl.follower import FollowerPolicy, load_checkpoint  # noqa: E402
from rl.env_wrapper import (KING_TOWER_HP_LV11,  # noqa: E402
                            TOWER_TROOP_HP_LV11)
_TOWER_HP_ANCHOR_K = KING_TOWER_HP_LV11
_TOWER_HP_ANCHOR_P = float(TOWER_TROOP_HP_LV11['PrincessTower'])
_TOWER_HP_ANCHOR_ALL = 2.0 * _TOWER_HP_ANCHOR_P + _TOWER_HP_ANCHOR_K
from rl.ppo import PPOTrainer  # noqa: E402
from rl.train_follower import FollowerOpponent  # noqa: E402
from rl.run_league import timeout_winner, overtime_open, _stall_probe, STALL_WINDOW, settle_stall  # noqa: E402
from rl.train_solo import solo_env, _draw_penalty, resolve_deck_set, _SOLO_PROPHET_PROB  # noqa: E402


def load_run_cfg(run_dir):
    """从 run 的 config.json 还原超参（覆盖 economy 预设），保证与训练一致。"""
    cfg = TrainConfig.resolve("economy")
    p = os.path.join(run_dir, "config.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        for k, v in d.items():
            if hasattr(cfg, k):
                try:
                    setattr(cfg, k, v)
                except Exception:
                    pass
    return cfg


def rollout(cfg, main, n_games, seed, device, max_frames=400000,
            collect_features=False, collect_supervised=False,
            collect_raw=False):
    """与训练同口径 rollout：返回逐帧数组。

    collect_features=True 时额外收集每帧的 post-LN enc（GRU 输入，
    `main._encode` 输出），供"可预测性探针"判别：状态→回报的信息
    到底在不在表征里（判别 critic 问题是程序侧吸收失败还是标签本身
    不可预测）。

    collect_supervised=True 时额外收集监督微调所需逐帧数据
    (obs, belief_tok, plan_vec, hidden_prev, R, EP)（第 9 个返回值；
    hidden_prev 是 act 前的隐状态，detach 输入，与训练语义一致）。
    """
    import random
    rng = random.Random(seed)
    mirror_deck, _ = resolve_deck_set(getattr(cfg, "deck_set", None) or "default")
    env = solo_env(cfg, seed, deck0=mirror_deck, deck1=mirror_deck)
    belief_dim = len(BeliefInference(opp_deck=env.deck1, n_particles=128,
                                     seed=0).encode(None, None))
    # B'/E'：镜像对手必须用与 ckpt 相同的 value 架构（否则键集不匹配）
    opp = FollowerPolicy(hidden=cfg.hidden_dim, plan_dim=PLAN_DIM, belief_dim=belief_dim,
                         value_bypass=bool(getattr(main, "value_bypass", False)),
                         value_independent=bool(getattr(main, "value_independent", False)))
    opp.to_device(device)
    opp.load_state_dict(main.state_dict())
    bp = BeliefPlanner()
    prophet = ProphetPlanner()

    V, R, EP, STEP_IDX, TERM, TRUNC, EPID = [], [], [], [], [], [], []
    X = [] if collect_features else None
    RAW = [] if collect_raw else None
    S = ({"obs": [], "tok": [], "plan": [], "hid": [], "R": [], "ep": []}
         if collect_supervised else None)
    ep_meta = []
    total = 0
    for g in range(n_games):
        if g % 10 == 0:
            print(f"[diag] rollout {g}/{n_games} 局...", flush=True)
        _tg = time.monotonic()
        opp_side = FollowerOpponent(
            opp, env, belief=BeliefInference(opp_deck=env.deck1, n_particles=128,
                                             seed=seed + g), deterministic=True)
        env.opponent = opp_side
        belief = BeliefInference(opp_deck=env.deck1, n_particles=128, seed=seed + 1000 + g)
        obs, _ = env.reset(seed=seed + 2000 + g)
        belief.reset(env.deck1)
        hidden = None
        last_hp, stall = None, 0
        ep_val, ep_rew, ep_term, ep_trunc = [], [], [], []
        ep_s_obs, ep_s_tok, ep_s_plan, ep_s_hid = [], [], [], []
        steps = 0
        done = False
        while not done and (steps < cfg.max_ep_steps or overtime_open(env.battle)):
            if cfg.train_stall_stop and steps and steps % STALL_WINDOW == 0:
                early, last_hp, stall = _stall_probe(env, last_hp, stall)
                if early:
                    break
            plan = (prophet.plan(env.get_prophet_state()) if rng.random() < _SOLO_PROPHET_PROB
                    else bp.plan(env.battle, belief.state(), obs))
            tok = belief.encode(obs, None)
            plan_vec = plan.to_vector()
            if collect_features:
                with torch.no_grad():
                    X.append(main._encode(obs, tok, plan_vec).detach().cpu()
                             .numpy().reshape(-1))
            if collect_raw:
                # G'（2026-09-12）：可观测原始状态标量（无表征、无网络）。
                # 直接读 `battle.players`：塔血/皇冠/圣水/时间 都是 `observation.py`
                # 已经喂给策略的量（该 sim 无敌方战争迷雾：`opp_elixir`/`opp_towers`
                # 都在 obs 字典里），所以这不是特权信息。
                # 注意：塔血用 lv11 锚归一化（所有 economy run 都是 card_level=11）。
                _b = env.battle
                _p0, _p1 = _b.players
                _mt = (_p0.king_tower_hp, _p0.left_tower_hp, _p0.right_tower_hp)
                _ot = (_p1.king_tower_hp, _p1.left_tower_hp, _p1.right_tower_hp)
                RAW.append(np.array([
                    _mt[0] / _TOWER_HP_ANCHOR_K, _mt[1] / _TOWER_HP_ANCHOR_P,
                    _mt[2] / _TOWER_HP_ANCHOR_P,
                    _ot[0] / _TOWER_HP_ANCHOR_K, _ot[1] / _TOWER_HP_ANCHOR_P,
                    _ot[2] / _TOWER_HP_ANCHOR_P,
                    float(_p0.get_crown_count()), float(_p1.get_crown_count()),
                    float(_p0.elixir) / 10.0, float(_p1.elixir) / 10.0,
                    float(_b.time) / 180.0,
                    (sum(_mt) - sum(_ot)) / _TOWER_HP_ANCHOR_ALL,
                ], dtype=np.float64))
            if collect_supervised:
                ep_s_obs.append(obs)
                ep_s_tok.append(tok)
                ep_s_plan.append(plan_vec)
                ep_s_hid.append(hidden)
            bundle, _, val, hidden, _ = main.act(
                obs, tok, plan_vec, env.get_action_mask,
                hidden=hidden, deterministic=False)
            obs, reward, term, trunc, info = env.step(bundle)
            ep_val.append(float(val))
            ep_rew.append(float(reward))
            ep_term.append(bool(term))
            ep_trunc.append(bool(trunc))
            belief.update(obs, info.get("opp_played"))
            done = term or trunc
            steps += 1
            total += 1
            if total > max_frames:
                break
        # —— 与训练同款结算（僵尸局/截断补终端）——
        truncated = ((not ep_term[-1]) and (len(ep_rew) >= cfg.max_ep_steps
                                           and not overtime_open(env.battle))
                     if ep_rew else False)
        virt = None
        if env.battle.winner is None and not env.battle.game_over and ep_rew:
            # C'（2026-09-12）：与训练同口径的早停结算（低置信裁定降噪）
            virt = settle_stall(env.battle, getattr(cfg, "stall_draw_margin", 0.05))
            rw = reward_to_env(cfg)
            if virt == 0:
                ep_rew[-1] += float(rw["win_bonus"])
            elif virt == 1:
                ep_rew[-1] -= float(rw["lose_penalty"])
            else:
                ep_rew[-1] -= _draw_penalty(cfg)
        if truncated and virt is None:
            ep_trunc[-1] = True
            # 末步 bootstrap：与训练一致（训练取 main.value(末状态)）。
            # 这里用最后记录帧的 value 近似（同一次前向的 cache 值，量级一致）。
            last_val = float(ep_val[-1])
        else:
            last_val = 0.0
        adv, ret = PPOTrainer.compute_gae(ep_rew, ep_val, ep_term, cfg.gamma,
                                          cfg.gae_lambda, truncated=ep_trunc,
                                          last_value=last_val)
        for i in range(len(ep_rew)):
            V.append(ep_val[i]); R.append(float(ret[i])); EP.append(g)
            STEP_IDX.append(i); TERM.append(ep_term[i]); TRUNC.append(ep_trunc[i])
            if collect_supervised:
                S["obs"].append(ep_s_obs[i]); S["tok"].append(ep_s_tok[i])
                S["plan"].append(ep_s_plan[i]); S["hid"].append(ep_s_hid[i])
                S["R"].append(float(ret[i])); S["ep"].append(g)
        ep_meta.append({"game": g, "len": len(ep_rew), "winner": env.battle.winner,
                        "ep_rew": float(np.sum(ep_rew)),
                        "R0": float(ret[0]) if len(ret) else 0.0,
                        "v_mean": float(np.mean(ep_val)) if ep_val else 0.0,
                        "R_mean": float(np.mean(ret)) if len(ret) else 0.0})
        print(f"[diag]   局 {g} 完成 {time.monotonic() - _tg:.1f}s "
              f"(len={len(ep_rew)})", flush=True)
        if total > max_frames:
            break
    X_arr = np.asarray(X, np.float64) if X is not None else None
    RAW_arr = np.asarray(RAW, np.float64) if RAW is not None else None
    return (np.asarray(V, np.float64), np.asarray(R, np.float64),
            np.asarray(EP, np.int64), np.asarray(STEP_IDX, np.int64),
            np.asarray(TERM, bool), np.asarray(TRUNC, bool), ep_meta, X_arr, S,
            RAW_arr)



#: G' 三层可预测性对照（2026-09-12）。原始标量的列名（与 rollout 里构造顺序一致）。
RAW_FEATURE_NAMES = [
    "my_king_hp", "my_left_hp", "my_right_hp",
    "opp_king_hp", "opp_left_hp", "opp_right_hp",
    "my_crown", "opp_crown", "my_elixir", "opp_elixir",
    "time", "hp_diff",
]


def _outcome_target(R, EP, TERM):
    """局结果标签：每局**最后一个终局帧**的 R（终止步 next_val=0 => R_T = 终局奖励），
    广播到该局所有帧。返回 (target, mask)。"""
    last = {}
    for i, (g, t) in enumerate(zip(EP, TERM)):
        if t:
            last[int(g)] = i
    tgt = np.zeros(len(R), dtype=np.float64)
    mask = np.zeros(len(R), dtype=bool)
    for i, g in enumerate(EP):
        j = last.get(int(g))
        if j is not None:
            tgt[i] = R[j]
            mask[i] = True
    return tgt, mask


def _three_layer_predictability(X, RAW, R, EP, TERM, STEP_IDX):
    """G'：同一批帧、同一**按局分组**留出口径，换特征层与标签。

    三层特征：① 原始可观测标量（塔血/圣水/时间/皇冠）② post-LN enc ③ 两者拼接。
    三个标签：逐帧 GAE 回报 R_t / 局结果广播（终局帧 R）/ 局均回报广播。
    外加**打乱局标签**的噪声地板（分组口径下 R² 的随机水平）。
    """
    print("\n=== G' 三层可预测性对照（全部按**局分组**留出 20%）===", flush=True)
    print("特征列: " + ", ".join(RAW_FEATURE_NAMES), flush=True)
    print(f"帧数={len(R)} 局数={len(np.unique(EP))}", flush=True)

    out_t, out_m = _outcome_target(R, EP, TERM)
    ug = np.unique(EP)
    gm = np.asarray([R[EP == g].mean() for g in ug])
    mean_t = np.zeros(len(R), dtype=np.float64)
    for k, g in enumerate(ug):
        mean_t[EP == g] = gm[k]

    # 局终皇冠差（从 RAW 的 my_crown/opp_crown 列取终局帧）——比终局奖励更干净的局结果
    _last = {}
    for i, (g, t) in enumerate(zip(EP, TERM)):
        if t:
            _last[int(g)] = i
    crown_t = np.zeros(len(R), dtype=np.float64)
    crown_m = np.zeros(len(R), dtype=bool)
    for i, g in enumerate(EP):
        j = _last.get(int(g))
        if j is not None:
            crown_t[i] = float(RAW[j, 7] - RAW[j, 6])   # opp_crown - my_crown
            crown_m[i] = True

    feats = [("raw", RAW)]
    if X is not None and len(X) == len(R):
        feats.append(("enc", X))
        feats.append(("raw+enc", np.concatenate([RAW, X], axis=1)))

    targets = [("frame_R", R, np.ones(len(R), dtype=bool)),
               ("outcome", out_t, out_m),
               ("ep_mean", mean_t, np.ones(len(R), dtype=bool)),
               ("crown_diff", crown_t, crown_m),
               # —— 功效对照（planted signal）：目标本身是 raw 特征的确定性函数。
               # 分组口径下若这两行不是 ≈+1，说明"分组留出 + 10 局测试集"这套
               # 测量本身没有功效，前面的负 R2 就不能解读为"不可预测"。
               ("CTRL:time", RAW[:, 10], np.ones(len(R), dtype=bool)),
               ("CTRL:hp_diff", RAW[:, 11], np.ones(len(R), dtype=bool))]

    print(f"{'特征':<10}{'标签':<10}{'线性R2(分组)':>14}{'MLP R2(分组)':>14}"
          f"{'MLP R2(逐帧泄漏口径)':>22}", flush=True)
    for fname, F in feats:
        for tname, y, mask in targets:
            if mask.sum() < 100:
                continue
            Fm, ym, Em = F[mask], y[mask], EP[mask]
            le, _, _ = lin_probe(Fm, ym, groups=Em)
            me, _, _ = mlp_probe(Fm, ym, groups=Em)
            lf, _, _ = mlp_probe(Fm, ym)
            print(f"{fname:<10}{tname:<10}{le:>+14.4f}{me:>+14.4f}{lf:>+22.4f}",
                  flush=True)

    # 噪声地板：把**局标签**在局之间打乱后重测（保持每局帧数不变）
    print("\n[噪声地板] 局标签随机置换后重测（分组口径）:", flush=True)
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(ug))
    gmap = {int(g): float(gm[perm[k]]) for k, g in enumerate(ug)}
    sh_t = np.asarray([gmap[int(g)] for g in EP])
    for fname, F in feats:
        le, _, _ = lin_probe(F, sh_t, groups=EP)
        me, _, _ = mlp_probe(F, sh_t, groups=EP)
        print(f"  {fname:<10} 线性 {le:+.4f}  MLP {me:+.4f}", flush=True)

    print("\n解读：① raw 能、② enc 不能 => 瓶颈在表征/观测编码（改 CNN/编码）；"
          "① 也不能（且低于噪声地板）=> 局结果本身不可从状态预测，critic 工作应停。",
          flush=True)


def ev(v, r):
    v = np.asarray(v, np.float64); r = np.asarray(r, np.float64)
    if r.size < 2:
        return float("nan")
    var = float(r.var())
    if var <= 1e-12:
        return float("nan")
    return float(1.0 - ((v - r) ** 2).mean() / var)


def batch_ev(v, r, bs=128, shuffle=False, seed=0):
    v = np.asarray(v, np.float64); r = np.asarray(r, np.float64)
    n = len(r)
    idx = np.arange(n)
    if shuffle:
        rs = np.random.RandomState(seed)
        rs.shuffle(idx)
    out = []
    for s in range(0, n - bs + 1, bs):
        sel = idx[s:s + bs]
        e = ev(v[sel], r[sel])
        if np.isfinite(e):
            out.append(e)
    return np.asarray(out, np.float64)


def _probe_ev(pred, r):
    pred = np.asarray(pred, np.float64)
    r = np.asarray(r, np.float64)
    var = float(r.var())
    if var <= 1e-12:
        return float("nan")
    return float(1.0 - ((pred - r) ** 2).mean() / var)


def _grouped_split(groups, seed=0, test_frac=0.2):
    """按**组**（局）留出：测试组的帧一条都不出现在训练集里。

    为什么必须有这个口径：本数据相邻帧 `corr(R_t,R_{t+1})≈0.99`，
    逐帧随机留出会让测试帧的邻居落在训练集 => R2 被时间泄漏抬高。
    """
    rng = np.random.RandomState(seed)
    gs = np.unique(groups)
    if len(gs) < 2:
        return None
    perm = rng.permutation(len(gs))
    n_test = max(1, int(len(gs) * test_frac))
    test_g = set(gs[perm[:n_test]].tolist())
    mask = np.array([g in test_g for g in groups])
    return np.where(~mask)[0], np.where(mask)[0]


def lin_probe(X, y, seed=0, test_frac=0.2, groups=None):
    """post-LN enc → return 的线性可预测性上界（留出法 R2）。

    `groups` 非空 → 按局分组留出（无时间泄漏）；None → 逐帧随机留出（旧口径）。
    """
    rng = np.random.RandomState(seed)
    n = len(y)
    if groups is not None:
        sp = _grouped_split(groups, seed=seed, test_frac=test_frac)
        if sp is None:
            return float("nan"), 0, 0
        tr, te = sp
    else:
        idx = rng.permutation(n)
        n_test = max(1, int(n * test_frac))
        tr, te = idx[n_test:], idx[:n_test]
    Xtr = np.column_stack([X[tr], np.ones(len(tr))])
    w, *_ = np.linalg.lstsq(Xtr, y[tr], rcond=None)
    pred = np.column_stack([X[te], np.ones(len(te))]) @ w
    return _probe_ev(pred, y[te]), len(tr), len(te)


def mlp_probe(X, y, seed=0, hidden=64, iters=400, lr=1e-3, test_frac=0.2,
              groups=None):
    """post-LN enc → return 的非线性可预测性上界（小 MLP 留出 R2）。

    判别逻辑：线性≈0 而 MLP>0 → 关系非线性（critic 含非线性 head 应能学，
    学不到则偏训练侧）；两者都≈0 → 该表征下标签不可预测（偏训练侧数据）。
    `groups` 非空 → 按局分组留出（无时间泄漏）；None → 逐帧随机留出（旧口径）。
    """
    import torch
    import torch.nn as nn
    rng = np.random.RandomState(seed)
    n = len(y)
    if groups is not None:
        sp = _grouped_split(groups, seed=seed, test_frac=test_frac)
        if sp is None:
            return float("nan"), 0, 0
        tr, te = sp
    else:
        idx = rng.permutation(n)
        n_test = max(1, int(n * test_frac))
        tr, te = idx[n_test:], idx[:n_test]
    Xt = torch.tensor(X[tr], dtype=torch.float32)
    yt = torch.tensor(y[tr], dtype=torch.float32).unsqueeze(1)
    Xe = torch.tensor(X[te], dtype=torch.float32)
    ye = torch.tensor(y[te], dtype=torch.float32).unsqueeze(1)
    mu, sd = Xt.mean(0), Xt.std(0).clamp_min(1e-6)
    Xt = (Xt - mu) / sd
    Xe = (Xe - mu) / sd
    ym, ys = yt.mean(), yt.std().clamp_min(1e-6)
    yt = (yt - ym) / ys
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.ReLU(),
                          nn.Linear(hidden, hidden), nn.ReLU(),
                          nn.Linear(hidden, 1))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    model.train()
    for _ in range(iters):
        opt.zero_grad()
        loss = lossf(model(Xt), yt)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(Xe).squeeze(1) * ys + ym
    return _probe_ev(pred.numpy(), y[te]), len(tr), len(te)


def _finetune_critic(args, model, S, R_all, device):
    """critic 监督微调判别（实验 A，2026-09-12）。

    用 rollout 记录的逐帧 (obs, belief_tok, plan_vec, hidden_prev) 监督回归
    GAE return：前向与训练完全一致（enc -> gru_cell(enc, hidden.detach())
    -> value_head），只算 value 项的 MSE 梯度。判别逻辑：
    - 监督微调后 test EV 明显上升（接近探针 0.24~0.31）=> 程序链路 OK，
      on-policy 联合训练没吸收（训练侧优化/欠训练）；
    - test EV 纹丝不动（仍 <=0）=> 价值头/GRU 结构问题（程序侧）。
    """
    import torch.nn as nn
    from torch.optim import Adam

    n = len(S["obs"])
    print(f"\n=== critic 监督微调（{n} 帧，lr={args.ft_lr}，"
          f"{args.ft_epochs} epoch，全网络 value 梯度）===", flush=True)

    games = sorted(set(S["ep"]))
    rng2 = np.random.RandomState(0)
    n_test_g = max(1, len(games) // 5)
    test_games = set(rng2.choice(games, size=n_test_g, replace=False).tolist())
    tr_idx = [i for i, e in enumerate(S["ep"]) if e not in test_games]
    te_idx = [i for i, e in enumerate(S["ep"]) if e in test_games]
    print(f"[ft] 局 {len(games)}（train {len(games)-n_test_g} / test {n_test_g}）; "
          f"帧 train {len(tr_idx)} / test {len(te_idx)}", flush=True)

    def _to_t(x, f32=True):
        if isinstance(x, torch.Tensor):
            return x.detach().to(device)
        return torch.as_tensor(x, dtype=torch.float32 if f32 else torch.long,
                               device=device)

    def predict(i):
        obs = S["obs"][i]          # dict 观测，_encode 内部取字段并 unsqueeze
        tok = _to_t(S["tok"][i])   # 1D，_encode 内部 unsqueeze(0)
        plan = _to_t(S["plan"][i])
        hid = S["hid"][i]
        if hid is None:
            hid = torch.zeros(1, model.hidden_dim, device=device)
        else:
            hid = _to_t(hid).unsqueeze(0) if hid.dim() == 1 else _to_t(hid)
        fused, enc = model._encode_parts(obs, tok, plan)
        # B'/E'（2026-09-12）：value 走策略真实通路（independent → 自己的编码器+MLP 头；
        # bypass → value_head(enc)；否则 GRU 隐状态），与 follower._value_from 同源
        h = model.gru_cell(enc, hid.detach())
        return model._value_from(enc, h, fused).squeeze(-1)

    def eval_ev(idx, label):
        vs = np.empty(len(idx), np.float64)
        rs = np.empty(len(idx), np.float64)
        with torch.no_grad():
            for k, i in enumerate(idx):
                vs[k] = float(predict(i).item())
                rs[k] = S["R"][i]
        e = ev(vs, rs)
        print(f"[ft] {label}: test EV = {e:+.4f}", flush=True)
        return e

    ev0 = eval_ev(te_idx, "监督前 critic(test 局)")
    opt = Adam(model.parameters(), lr=args.ft_lr)
    model.train()
    for ep in range(args.ft_epochs):
        order = np.random.RandomState(1000 + ep).permutation(len(tr_idx))
        tot_loss = 0.0
        cnt = 0
        for j in order:
            i = tr_idx[j]
            r = torch.tensor([S["R"][i]], dtype=torch.float32, device=device)
            loss = nn.functional.mse_loss(predict(i), r)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += float(loss.item())
            cnt += 1
        te_ev = eval_ev(te_idx, f"监督微调后 epoch {ep+1}")
        print(f"[ft] epoch {ep+1}/{args.ft_epochs}: 平均 MSE(train)="
              f"{tot_loss/max(1,cnt):.2f}", flush=True)
        if te_ev > 0.20:
            print("[ft] test EV 已 >0.20，接近探针上界 => 提前停止", flush=True)
            break
    print(f"\n[ft] 判别：初始 {ev0:+.4f} -> 最终 {te_ev:+.4f}"
          f"（探针 MLP 上界约 0.24~0.31）", flush=True)


def _bypass_gru(args, model, S, R_all, device):
    """B' 实验：value_head 直连 enc（绕过 GRU）的监督对照（2026-09-12）。

    与实验 A 同数据、同局级留出（seed 0、1/5 局）、同配方（ft_lr、ft_epochs、
    逐帧 value MSE、Adam），唯一差别 = 输入是 post-LN enc 而非
    `gru_cell(enc, hidden.detach())`。三个变体：

      lin-frozen : Linear(hidden,1) 头（= value_head 同容量），enc 冻结
                   -> value_head 容量直接吸 enc 信息的上限
      lin-joint  : Linear(hidden,1) 头 + 共享 trunk 一起训（= 模型去掉 GRU）
                   -> 与实验 A 唯一差异是 GRU，直接量化"GRU 是不是瓶颈"
      mlp-frozen : MLP(hidden->64->1) 头（对齐探针容量），enc 冻结
                   -> 探针容量 + 本配方，判别"容量 vs 配方"哪个限制 A

    判别（对照实验 A 全路径 +0.054/+0.128、探针 MLP +0.242）：
      lin-joint 明显高于 A => GRU 是瓶颈；
      全变体都约等于 A     => 瓶颈不在 GRU（head 容量/配方/标签）；
      mlp-frozen 约 0.24   => 配方没问题、是 value_head 太浅（容量问题）。
    """
    import torch.nn as nn
    from torch.optim import Adam

    n = len(S["obs"])
    print(f"\n=== B' bypass-GRU 监督对照（{n} 帧，lr={args.ft_lr}，"
          f"{args.ft_epochs} epoch）===", flush=True)

    games = sorted(set(S["ep"]))
    rng2 = np.random.RandomState(0)
    n_test_g = max(1, len(games) // 5)
    test_games = set(rng2.choice(games, size=n_test_g, replace=False).tolist())
    tr_idx = [i for i, e in enumerate(S["ep"]) if e not in test_games]
    te_idx = [i for i, e in enumerate(S["ep"]) if e in test_games]
    print(f"[bp] 局 {len(games)}（train {len(games)-n_test_g} / test {n_test_g}）; "
          f"帧 train {len(tr_idx)} / test {len(te_idx)}", flush=True)

    def _to_t(x, f32=True):
        if isinstance(x, torch.Tensor):
            return x.detach().to(device)
        return torch.as_tensor(x, dtype=torch.float32 if f32 else torch.long,
                               device=device)

    def _enc_of(i, m):
        obs = S["obs"][i]          # dict 观测，_encode 内部取字段并 unsqueeze
        tok = _to_t(S["tok"][i])   # 1D，_encode 内部 unsqueeze(0)
        plan = _to_t(S["plan"][i])
        return m._encode(obs, tok, plan)   # (1,hidden)

    # pristine trunk（deepcopy，不动传给实验 A 的原模型）
    base = copy.deepcopy(model)

    # 冻结模式：预计算一次 enc
    print("[bp] 预计算冻结 enc ...", flush=True)
    encs = []
    with torch.no_grad():
        for i in range(n):
            encs.append(_enc_of(i, base).reshape(-1))
    enc_all = torch.stack(encs).to(device)          # (n, hidden)

    def _run_variant(name, head, enc_mode, trunk):
        """enc_mode='frozen' 用缓存 enc；'joint' 每 epoch 重算（trunk 可训）。"""
        params = list(head.parameters())
        if enc_mode == "joint":
            params += list(trunk.parameters())
        opt = Adam(params, lr=args.ft_lr)

        def _pred(i):
            if enc_mode == "frozen":
                e = enc_all[i:i + 1]
            else:
                e = _enc_of(i, trunk)
            return head(e).squeeze(-1)

        def _eval(idx):
            vs = np.empty(len(idx), np.float64)
            rs = np.empty(len(idx), np.float64)
            with torch.no_grad():
                for k, i in enumerate(idx):
                    vs[k] = float(_pred(i).item())
                    rs[k] = S["R"][i]
            return ev(vs, rs)

        head.train()
        if enc_mode == "joint":
            trunk.train()
        ev0 = _eval(te_idx)
        te_ev = ev0
        for ep in range(args.ft_epochs):
            order = np.random.RandomState(1000 + ep).permutation(len(tr_idx))
            tot = 0.0
            cnt = 0
            for j in order:
                i = tr_idx[j]
                r = torch.tensor([S["R"][i]], dtype=torch.float32, device=device)
                loss = nn.functional.mse_loss(_pred(i), r)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss.item())
                cnt += 1
            te_ev = _eval(te_idx)
            print(f"[bp:{name}] epoch {ep+1}/{args.ft_epochs}: "
                  f"test EV={te_ev:+.4f}  (MSE(t)={tot/max(1,cnt):.2f})", flush=True)
            if te_ev > 0.20:
                print(f"[bp:{name}] test EV >0.20 => 提前停止", flush=True)
                break
        print(f"[bp:{name}] 初始 {ev0:+.4f} -> 最终 {te_ev:+.4f}", flush=True)
        return ev0, te_ev

    hd = model.hidden_dim
    res = {}
    res["lin-frozen"] = _run_variant(
        "lin-frozen", nn.Linear(hd, 1).to(device), "frozen", base)
    res["lin-joint"] = _run_variant(
        "lin-joint", nn.Linear(hd, 1).to(device), "joint", base)
    res["mlp-frozen"] = _run_variant(
        "mlp-frozen",
        nn.Sequential(nn.Linear(hd, 64), nn.ReLU(), nn.Linear(64, 1)).to(device),
        "frozen", base)

    print("\n[bp] 判别汇总（对照：实验 A 全路径 +0.054/+0.128；探针 MLP +0.242）", flush=True)
    for k, (e0, e1) in res.items():
        print(f"[bp] {k}: {e0:+.4f} -> {e1:+.4f}", flush=True)
    best_lin_joint = res["lin-joint"][1]
    if best_lin_joint > 0.18:
        print("[bp] 判别：lin-joint 明显高于实验 A => GRU 通路是吸收瓶颈", flush=True)
    elif max(r[1] for r in res.values()) < 0.10:
        print("[bp] 判别：全变体接近实验 A => 瓶颈不在 GRU，"
              "在 head 容量/配方/标签", flush=True)
    else:
        print("[bp] 判别：介于两者之间（见各变体数值）", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--run-dir", default="runs/prod_200k_valnorm_ev")
    ap.add_argument("--games", type=int, default=24)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--out", default="")
    ap.add_argument("--save-npz", default="")
    ap.add_argument("--probe", action="store_true",
                    help="训练线性/MLP 可预测性探针（post-LN enc -> GAE return）")
    ap.add_argument("--predict", action="store_true",
                    help="G'（2026-09-12）三层可预测性对照："
                         "① 原始可观测状态标量（塔血/圣水/时间/皇冠）"
                         "② post-LN enc  ③ 两者拼接，对同一批标签（逐帧回报/局结果/"
                         "局均回报）都按**局分组**留出，并给打乱标签的噪声地板")
    ap.add_argument("--finetune", action="store_true",
                    help="critic 监督微调判别：用 (obs,tok,plan,hidden)->GAE return 监督"
                         "训练 value 通路，看 EV 能否升到探针水平")
    ap.add_argument("--bypass", action="store_true",
                    help="B' 对照：value_head 直连 enc（绕过 GRU）监督训练，"
                         "量化 GRU 是否是吸收瓶颈（与 --finetune 可同跑）")
    ap.add_argument("--ft-epochs", type=int, default=4)
    ap.add_argument("--ft-lr", type=float, default=3e-4)
    args = ap.parse_args()

    cfg = load_run_cfg(args.run_dir)
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    print(f"[diag] cfg: gamma={cfg.gamma} lam={cfg.gae_lambda} max_ep={cfg.max_ep_steps} "
          f"stall={cfg.train_stall_stop} device={device}", flush=True)

    dummy = solo_env(cfg, 0)
    belief_dim = len(BeliefInference(opp_deck=dummy.deck1, n_particles=128,
                                     seed=0).encode(None, None))
    main_pol = load_checkpoint(args.ckpt, hidden_dim=cfg.hidden_dim,
                               plan_dim=PLAN_DIM, belief_dim=belief_dim)
    main_pol.to_device(device)
    main_pol.train()  # 与训练一致（dropout/BN 若有）
    print(f"[diag] loaded {args.ckpt}", flush=True)

    V, R, EP, SI, TERM, TRUNC, meta, X, S, RAW = rollout(
        cfg, main_pol, args.games, args.seed, device,
        collect_features=args.probe, collect_supervised=(args.finetune or args.bypass),
        collect_raw=args.predict)
    n = len(R)
    print(f"\n[diag] 帧数={n}  局数={len(meta)}  "
          f"平均局长={n/max(1,len(meta)):.1f}", flush=True)
    wins = sum(1 for m in meta if m["winner"] == 0)
    losses = sum(1 for m in meta if m["winner"] == 1)
    draws = sum(1 for m in meta if m["winner"] is None)
    print(f"[diag] 结果：main(先手视角) win={wins} lose={losses} draw={draws}", flush=True)

    print("\n=== 回报/价值分布 ===", flush=True)
    print(f"R: mean={R.mean():.3f} std={R.std():.3f} min={R.min():.2f} max={R.max():.2f}", flush=True)
    print(f"v: mean={V.mean():.3f} std={V.std():.3f} min={V.min():.2f} max={V.max():.2f}", flush=True)
    mse = float(((V - R) ** 2).mean())
    var = float(R.var())
    print(f"MSE(v,R)={mse:.3f}  Var(R)={var:.3f}  RMSE={np.sqrt(mse):.3f}", flush=True)
    bias = float((V - R).mean())
    print(f"bias E[v-R]={bias:.3f}  (E[v]={V.mean():.3f}, E[R]={R.mean():.3f})", flush=True)

    print("\n=== EV 各口径 ===", flush=True)
    print(f"EV_global        = {ev(V, R):+.4f}", flush=True)
    bc = batch_ev(V, R, args.batch, shuffle=False)
    bsh = batch_ev(V, R, args.batch, shuffle=True, seed=1)
    print(f"EV_batch_contig  = {np.nanmean(bc):+.4f}  (n={len(bc)}, "
          f"min={np.nanmin(bc):+.2f} max={np.nanmax(bc):+.2f})", flush=True)
    print(f"EV_batch_shuffle = {np.nanmean(bsh):+.4f}", flush=True)
    # 每局一批
    ep_evs = [ev(V[EP == g], R[EP == g]) for g in np.unique(EP)]
    ep_evs = np.asarray([e for e in ep_evs if np.isfinite(e)])
    print(f"EV_episode       = {ep_evs.mean():+.4f}  (n={len(ep_evs)})", flush=True)

    # 批内 vs 全局 方差
    print("\n=== 方差结构（解释负 EV 的来源）===", flush=True)
    within = []
    for s in range(0, n - args.batch + 1, args.batch):
        within.append(R[s:s + args.batch].var())
    within = np.asarray(within)
    print(f"全局 Var(R)={var:.3f} (std {np.sqrt(var):.2f})", flush=True)
    print(f"批内 Var(R): mean={np.nanmean(within):.3f} (std {np.sqrt(np.nanmean(within)):.2f}) "
          f"median={np.nanmedian(within):.3f}", flush=True)
    print(f"批内/全局 方差比 ≈ {np.nanmean(within)/max(var,1e-9):.3f}", flush=True)
    # 序列相关性（同局相邻帧 R 的相关）
    r1 = np.corrcoef(R[:-1], R[1:])[0, 1]
    print(f"相邻帧 corr(R_t, R_t+1)={r1:.4f}", flush=True)

    # 相关性 / 可预测性
    c_all = float(np.corrcoef(V, R)[0, 1]) if V.std() > 0 and R.std() > 0 else float("nan")
    print(f"\ncorr(v,R) 全体 = {c_all:+.4f}", flush=True)
    # 局内去均值相关（critic 能不能解释"局内"变化）
    Vc = V.copy(); Rc = R.copy()
    for g in np.unique(EP):
        m = EP == g
        Vc[m] -= V[m].mean(); Rc[m] -= R[m].mean()
    c_within = float(np.corrcoef(Vc, Rc)[0, 1]) if Vc.std() > 0 and Rc.std() > 0 else float("nan")
    print(f"corr(v,R) 局内去均值 = {c_within:+.4f}", flush=True)
    # 跨局相关（每局均值）
    gm_v = np.asarray([V[EP == g].mean() for g in np.unique(EP)])
    gm_r = np.asarray([R[EP == g].mean() for g in np.unique(EP)])
    c_between = float(np.corrcoef(gm_v, gm_r)[0, 1]) if gm_v.std() > 0 else float("nan")
    print(f"corr(局均 v, 局均 R) 跨局 = {c_between:+.4f}", flush=True)

    # 回归斜率
    if V.std() > 0:
        slope = float(np.cov(V, R)[0, 1] / V.var())
        print(f"回归 R~v 斜率 = {slope:.3f}  (1.0=标准差对齐；<0=方向相反)", flush=True)

    # 一个诊断上界：EV_between（只用局均值，等价于"局级 critic"能做到多少）
    print(f"EV_between(局均值层) = {ev(gm_v, gm_r):+.4f}", flush=True)

    if args.probe and X is not None and len(X) == len(R) and X.shape[1] > 0:
        print("\n=== 可预测性探针（post-LN enc -> GAE return，留出 20%）===", flush=True)
        le, ntr, nte = lin_probe(X, R)
        print(f"线性探针  test R2 = {le:+.4f}   (train {ntr} / test {nte} 帧)", flush=True)
        me, _, _ = mlp_probe(X, R)
        print(f"MLP 探针  test R2 = {me:+.4f}   (enc dim={X.shape[1]})", flush=True)
        # —— 2026-09-12 新增：**按局分组**留出（诊断时间泄漏）——
        # 逐帧随机留出在本数据集上是无效的：相邻帧 corr(R_t,R_{t+1})≈0.99，
        # 测试帧的"邻居"几乎必然在训练集里 => R2 被时间泄漏抬高。
        # 分组口径 = 训练局/测试局完全不重叠，才是"跨局面泛化"的可预测性。
        leg, ntrg, nteg = lin_probe(X, R, groups=EP)
        meg, _, _ = mlp_probe(X, R, groups=EP)
        print(f"[按局分组留出] 线性 R2 = {leg:+.4f}  MLP R2 = {meg:+.4f}"
              f"   (train {ntrg} / test {nteg} 帧, {len(np.unique(EP))} 局)", flush=True)
        print(f"  => 逐帧 R2 - 分组 R2 = MLP {me - meg:+.4f} / 线性 {le - leg:+.4f}"
              "（该差就是时间泄漏的量级）", flush=True)

        # —— 2026-09-12：**换标签**再探（都按局分组留出）——
        # 问题：跨局不可预测到底是"表征不行"还是"逐帧 GAE 回报本身就是帧级噪声主导"。
        # 终局帧的 R 恰等于终局奖励（终止步 next_val=0 => R_T = r_T），故"局结果"
        # 可从 R 中取出：每局最后一个**终局**帧的 R 作该局结果，广播到该局所有帧。
        # 若它的分组 R2 明显为正 => 状态能预测局结果、只是预测不了帧级回报
        # => critic 该换标签（G' 候选）。
        _last_idx = {}
        for _i, (_g, _t) in enumerate(zip(EP, TERM)):
            if _t:
                _last_idx[int(_g)] = _i
        if _last_idx:
            _out = np.zeros(len(R), dtype=np.float64)
            _has = np.zeros(len(R), dtype=bool)
            for _i, _g in enumerate(EP):
                _j = _last_idx.get(int(_g))
                if _j is not None:
                    _out[_i] = R[_j]
                    _has[_i] = True
            if _has.sum() > 100:
                _Xo, _Ro, _Eo = X[_has], _out[_has], EP[_has]
                _lo, _, _ = lin_probe(_Xo, _Ro, groups=_Eo)
                _mo, _, _ = mlp_probe(_Xo, _Ro, groups=_Eo)
                _mo_f, _, _ = mlp_probe(_Xo, _Ro)
                print(f"[换标签：局结果广播] 分组 线性 R2 = {_lo:+.4f} / "
                      f"MLP R2 = {_mo:+.4f}（逐帧口径 MLP = {_mo_f:+.4f}；"
                      f"{len(np.unique(_Eo))} 局有终局帧 / 共 {len(np.unique(EP))} 局）",
                      flush=True)
            # 另一候选标签：局均回报（对每局所有帧广播局均值）
            _ug = np.unique(EP)
            _gm_all = np.asarray([R[EP == g].mean() for g in _ug])
            _m2v = np.zeros(len(R), dtype=np.float64)
            for _k, _g in enumerate(_ug):
                _m2v[EP == _g] = _gm_all[_k]
            _lm, _, _ = lin_probe(X, _m2v, groups=EP)
            _mm, _, _ = mlp_probe(X, _m2v, groups=EP)
            print(f"[换标签：局均回报广播] 分组 线性 R2 = {_lm:+.4f} / "
                  f"MLP R2 = {_mm:+.4f}", flush=True)
        print(f"critic 对照 EV_global = {ev(V, R):+.4f}（critic 对同一批帧的解释力）",
              flush=True)
        print("解读：探针≈critic => 标签在该表征下就不可预测（偏训练侧数据）；"
              "探针>>critic => 信息在表征里但网络没吸收（偏程序侧/训练侧优化）",
              flush=True)

    if args.predict and RAW is not None and len(RAW) == len(R):
        _three_layer_predictability(X, RAW, R, EP, TERM, SI)

    if args.bypass:
        _bypass_gru(args, main_pol, S, R, device)
    if args.finetune:
        _finetune_critic(args, main_pol, S, R, device)

    if args.save_npz:
        np.savez(args.save_npz, V=V, R=R, EP=EP, STEP=SI, TERM=TERM, TRUNC=TRUNC,
                 meta=np.array(meta, dtype=object))
        print(f"\n[diag] 原始数据已存 {args.save_npz}", flush=True)


if __name__ == "__main__":
    main()
