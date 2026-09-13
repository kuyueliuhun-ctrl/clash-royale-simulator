#!/usr/bin/env python
"""只读探针：POMDP 信息天花板检验。

问题（可否证）：当前逐帧输入能否解释**局内**回报变化？把 oracle 隐藏量（对手手牌/
循环/圣水，引擎真值）加进去能提升多少？判据见
`docs/pomdp_ceiling_prereg_2026-09-13.md`（跑前写死）。

本脚本**只读**：不改训练路径、不写 run 目录；rollout + 监督拟合都在探针内完成。

用法（在 src/clasher_new 下）：
  PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe \
      ../../scripts/pomdp_ceiling_probe.py \
      --ckpt runs/critic_inert_probe_20k/solo_main_20000.pt \
      --mode stoch --frames 30000 --device cuda
"""

import argparse
import json
import os
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SRC = os.path.join(_ROOT, "src", "clasher_new")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rl.config import TrainConfig  # noqa: E402
from rl.belief import BeliefInference  # noqa: E402
from rl.belief_planner import BeliefPlanner  # noqa: E402
from rl.plan_space import PLAN_DIM  # noqa: E402
from rl.follower import FollowerPolicy, load_checkpoint  # noqa: E402
from rl.train_follower import FollowerOpponent  # noqa: E402
from rl.train_solo import solo_env, resolve_deck_set  # noqa: E402
from rl.ppo import PPOTrainer  # noqa: E402
from rl.observation import ENTITY_NAMES  # noqa: E402


# --------------------------------------------------------------------------
# 特征
# --------------------------------------------------------------------------
def feat_obs(obs) -> np.ndarray:
    """A 集：跟随者实际看到的（obs 原样展平）。"""
    parts = [np.asarray(obs["grid"], dtype=np.float32).ravel(),
             np.asarray(obs["hand"], dtype=np.float32).ravel(),
             np.asarray(obs["elixir"], dtype=np.float32).ravel(),
             np.asarray(obs["next_card"], dtype=np.float32).ravel(),
             np.asarray(obs["time"], dtype=np.float32).ravel()]
    return np.concatenate(parts)


def make_hidden_mapper(deck):
    """返回 (eid2slot, ndeck)：卡名 -> 本副牌槽位。"""
    eid2slot = {}
    for slot, name in enumerate(deck):
        if name in ENTITY_NAMES:
            eid2slot[ENTITY_NAMES.index(name)] = slot
    return eid2slot, len(deck)


def feat_hidden(hid, eid2slot, ndeck) -> np.ndarray:
    """B 集增量：对手循环序 / 手牌 / 进手 / 圣水（引擎真值）。"""
    cyc = np.asarray(hid["opp_cycle"]).ravel()
    hand = np.asarray(hid["opp_hand"]).ravel()
    nxt = int(np.asarray(hid["opp_next"]).ravel()[0])
    pos = np.zeros(ndeck, dtype=np.float32)
    for k, eid in enumerate(cyc):
        s = eid2slot.get(int(eid))
        if s is not None:
            pos[s] = float(k) / max(1.0, float(ndeck - 1))
    hp = np.zeros(ndeck, dtype=np.float32)
    for eid in hand:
        s = eid2slot.get(int(eid))
        if s is not None:
            hp[s] = 1.0
    np_ = np.zeros(ndeck, dtype=np.float32)
    s = eid2slot.get(nxt)
    if s is not None:
        np_[s] = 1.0
    elx = np.asarray(hid["opp_elixir"], dtype=np.float32).ravel()
    return np.concatenate([pos, hp, np_, elx])


# --------------------------------------------------------------------------
# rollout
# --------------------------------------------------------------------------
def rollout(a, cfg, device):
    env = solo_env(cfg, a.seed)
    bdim = len(BeliefInference(opp_deck=env.deck1, n_particles=128,
                               seed=a.seed).encode(None, None))
    pol = load_checkpoint(a.ckpt, hidden_dim=cfg.hidden_dim, plan_dim=PLAN_DIM,
                          belief_dim=bdim)
    pol.to_device(device).eval()
    opp = FollowerPolicy(hidden=cfg.hidden_dim, plan_dim=PLAN_DIM, belief_dim=bdim,
                         value_bypass=bool(pol.value_bypass),
                         value_independent=bool(pol.value_independent))
    opp.to_device(device)
    opp.load_state_dict(pol.state_dict())
    det = (a.mode == "det")
    env.opponent = FollowerOpponent(
        opp, env, belief=BeliefInference(opp_deck=env.deck1, n_particles=128,
                                         seed=a.seed + 1), deterministic=det)
    _, _ = resolve_deck_set(getattr(cfg, "deck_set", None) or "default")
    eid2slot, ndeck = make_hidden_mapper(env.deck0)

    bp = BeliefPlanner()
    belief = BeliefInference(opp_deck=env.deck1, n_particles=128, seed=a.seed)
    obs, _ = env.reset(seed=a.seed)
    belief.reset(env.deck1)
    hidden = None

    FA, FC, FB, RT, VT, EP = [], [], [], [], [], []
    ep_rew, ep_val, ep_term, ep_trunc = [], [], [], []
    epA, epC, epB = [], [], []
    ep_idx = 0
    ep_lens = []
    n_frame = 0
    t0 = time.time()
    with torch.no_grad():
        step = 0
        while n_frame < a.frames:
            plan = bp.plan(env.battle, belief.state(), obs)
            tok = belief.encode(obs, None)
            pv = plan.to_vector()
            bundle, _, val, hidden, _ = pol.act(obs, tok, pv, env.get_action_mask,
                                                hidden=hidden, deterministic=det)
            hid = env.get_hidden_state()
            epA.append(feat_obs(obs))
            epC.append(np.concatenate([epA[-1], np.asarray(tok, dtype=np.float32).ravel()]))
            epB.append(np.concatenate([epA[-1], feat_hidden(hid, eid2slot, ndeck)]))
            ep_val.append(float(np.asarray(val, dtype=np.float64).ravel()[0]))
            obs, r, term, trunc, info = env.step(bundle)
            ep_rew.append(float(r))
            ep_term.append(bool(term))
            ep_trunc.append(bool(trunc))
            belief.update(obs, info.get("opp_played"))
            n_frame += 1
            step += 1
            if term or trunc:
                last_val = 0.0
                if not term:
                    nb = bp.plan(env.battle, belief.state(), obs)
                    last_val = float(np.asarray(
                        pol.value(obs, belief.encode(obs, None), nb.to_vector(),
                                  hidden), dtype=np.float64).ravel()[0])
                _, ret = PPOTrainer.compute_gae(ep_rew, ep_val, ep_term, cfg.gamma,
                                                cfg.gae_lambda, truncated=ep_trunc,
                                                last_value=last_val)
                FA.extend(epA); FC.extend(epC); FB.extend(epB)
                RT.extend(np.asarray(ret, dtype=np.float64).tolist())
                VT.extend(ep_val)
                EP.extend([ep_idx] * len(ep_rew))
                ep_lens.append(len(ep_rew))
                ep_idx += 1
                ep_rew, ep_val, ep_term, ep_trunc = [], [], [], []
                epA, epC, epB = [], [], []
                obs, _ = env.reset(seed=a.seed + step)
                belief.reset(env.deck1)
                hidden = None
                if a.verbose and ep_idx % 10 == 0:
                    print(f"  [rollout] ep={ep_idx} frames={n_frame} "
                          f"{time.time() - t0:.1f}s", flush=True)
    dt = time.time() - t0
    X = {"A": np.asarray(FA, dtype=np.float32),
         "C": np.asarray(FC, dtype=np.float32),
         "B": np.asarray(FB, dtype=np.float32)}
    y = np.asarray(RT, dtype=np.float64)
    v = np.asarray(VT, dtype=np.float64)
    ep = np.asarray(EP, dtype=np.int64)
    print(f"[rollout] mode={a.mode} frames={len(y)} episodes={ep_idx} "
          f"ep_len mean={np.mean(ep_lens):.1f} wall={dt:.1f}s "
          f"({len(y) / max(1e-9, dt):.1f} frames/s)", flush=True)
    return X, y, v, ep


# --------------------------------------------------------------------------
# 指标
# --------------------------------------------------------------------------
def ev(pred, y):
    mse = float(np.mean((np.asarray(pred, dtype=np.float64) - y) ** 2))
    var = float(np.var(y))
    return None if var <= 1e-12 else 1.0 - mse / var


def ev_within(pred, y, ep):
    """局内中心化 EV：优势算子只关心 V 的局内变化。"""
    p = np.asarray(pred, dtype=np.float64).copy()
    yy = np.asarray(y, dtype=np.float64).copy()
    for e in np.unique(ep):
        m = (ep == e)
        p[m] -= p[m].mean()
        yy[m] -= yy[m].mean()
    var = float(np.var(yy))
    if var <= 1e-12:
        return None
    return 1.0 - float(np.mean((p - yy) ** 2)) / var


def ev_within_raw(pred, y, ep):
    yc = np.asarray(y, dtype=np.float64).copy()
    for e in np.unique(ep):
        m = (ep == e)
        yc[m] -= yc[m].mean()
    var = float(np.var(yc))
    if var <= 1e-12:
        return None
    return 1.0 - float(np.mean((np.asarray(pred, dtype=np.float64) - y) ** 2)) / var


def ev_win128_med(pred, y, ep, win=128):
    vals = []
    p = np.asarray(pred, dtype=np.float64)
    for e in np.unique(ep):
        idx = np.where(ep == e)[0]
        for s in range(0, len(idx) - win + 1, win):
            k = idx[s:s + win]
            if np.var(y[k]) <= 1e-12:
                continue
            vals.append(1.0 - float(np.mean((p[k] - y[k]) ** 2)) / float(np.var(y[k])))
    return (float(np.median(vals)) if vals else None), len(vals)


def var_decomp(y, ep):
    tot = float(np.var(y))
    means, wsum, within = [], 0.0, 0.0
    for e in np.unique(ep):
        m = (ep == e)
        means.append(y[m].mean())
        wsum += float(m.sum())
        within += float(m.sum()) * float(np.var(y[m]))
    between = float(np.var(np.asarray(means, dtype=np.float64)))
    return tot, between, within / max(1.0, wsum)


def clock_perm(n, ep, seed=0):
    """按『局内位置 k 相同、但来自不同局』构造行置换（跨局配对打散用）。"""
    rng = np.random.default_rng(int(seed) + 707)
    k = np.zeros(n, dtype=np.int64)
    for e in np.unique(ep):
        idx = np.where(ep == e)[0]
        k[idx] = np.arange(len(idx))
    perm = np.arange(n)
    for kk in np.unique(k):
        idx = np.where(k == kk)[0]
        if len(idx) < 2:
            continue
        tgt = idx[rng.permutation(len(idx))]
        perm[idx] = np.roll(tgt, 1)   # roll 保证不落在自己那一行
    return perm


def scramble_rows_by_clock(X, ep, seed=0):
    """修订 5 证伪对照用的行置换：同一『局内位置 k』内跨局换行。

    置换后每帧的特征来自**另一局**、但**同一个局内位置**⇒ 时钟信息完全保留、
    状态信息被打散。若真实目标上的 `EV_within` 在这种输入下不塌，说明模型用的
    不是状态，而是某种与局无关的公共结构（那 V3 的增益就不能解读为状态级信号）。
    """
    return np.asarray(X)[clock_perm(len(ep), ep, seed)]


def scramble_target_by_clock(y, ep, seed=0):
    """下一轮预注册的**阴性对照**：把**目标**按同样的置换打散。

    与 `scramble_rows_by_clock` 破坏的是同一条配对（状态↔回报），但作用在目标侧：
    目标的**边缘分布**、**局内位置结构**、**局内/局间分解**全部逐值保留，
    只有"哪一帧的特征对上哪个回报"被随机化。
    ⇒ 一个只能用状态做预测的估计器在这个靶子上**必须**给出 ≈0；
    若它仍能拿到正 `EV_within`，说明估计器在用与配对无关的公共结构。
    """
    return np.asarray(y)[clock_perm(len(y), ep, seed + 909)]



def load_npz(path):
    """从 `--save-npz` 落盘的逐帧数据离线重建 (X, y, v, ep)，免重跑 rollout。"""
    z = np.load(path)
    X = {k.split("_", 1)[1]: z[k].astype(np.float32) for k in z.files
         if k.startswith("X_")}
    y = z["y"].astype(np.float64)
    v = z["v"].astype(np.float64)
    ep = z["ep"]
    return X, y, v, ep


def make_positive_control(Xa, ep, seed=0, within_var=58.0, between_var=166.0):
    """闸门 G5 的阳性对照目标（修订 4）。

    **动机**：如果连"信号按构造必然存在"的目标，这条流水线都做不到 `EV_within > 0`，
    那么它在真实回报上给出的"局内不可预测"就是**估计器无能**，不是科学结论。

    构造：取方差最大的 64 个真实特征列里的 8 列，合成一个非线性函数作为局内信号，
    再叠加局间水平偏移；两者方差按**真实回报的局内/局间方差**配平 ⇒ 与真问题同形，
    但**答案已知**（局内分量 100% 由这些特征决定）。
    """
    Xa = np.asarray(Xa, dtype=np.float64)
    rng = np.random.default_rng(int(seed) + 101)
    sd = Xa.std(axis=0)
    cand = np.argsort(-sd)[:64]
    cols = rng.choice(cand, size=8, replace=False)
    z = [(Xa[:, c] - Xa[:, c].mean()) / max(1e-8, Xa[:, c].std()) for c in cols]
    s = 1.0 * z[0] + 0.7 * np.tanh(1.5 * z[1]) + 0.5 * z[2] * z[3] + 0.3 * z[4]
    for e in np.unique(ep):
        m = (ep == e)
        s[m] -= s[m].mean()
    s = s / max(1e-12, s.std()) * np.sqrt(max(1e-12, within_var))
    off = np.zeros_like(s)
    lvl = float(np.sqrt(max(1e-12, between_var)))
    for e in np.unique(ep):
        m = (ep == e)
        off[m] = rng.normal(0.0, lvl)
    y = s + off
    return y, {"cols": [int(c) for c in cols], "within_var": float(within_var),
               "between_var": float(between_var), "realized_var": float(np.var(y))}


# --------------------------------------------------------------------------
# 拟合
# --------------------------------------------------------------------------
def standardize(Xtr, Xs):
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd[sd < 1e-8] = 1.0
    return [(x - mu) / sd for x in Xs]


def fit_ridge(Xtr, ytr, Xva, yva, Xte, alphas=None, ep_va=None, select="pooled"):
    """Ridge + **输出裁剪**。

    2026-09-13 修订 3：初版 alpha 网格上限只有 10（对 d=8648、n≈2e4 相当于 0.05% 正则
    ⇒ 几乎 OLS），在**整局留出**（分布外）上外推爆炸（实测 EV_within ≈ −60）。
    ⇒ 网格放宽到 1e8（alpha→∞ 即"预测训练均值"= EV 0 的下界解），并把预测裁剪到
    训练目标区间内（估计器修正，不改问题）。

    2026-09-13 修订 4：`select` 决定**模型选择口径**——`pooled`（= EV，预注册初版）
    或 `within`（= EV_within，与主判据同口径）。选择口径与评价口径不匹配会让
    "局内"读数被系统性压低；两种口径都跑，见判据一致性闸 G6。
    """
    if alphas is None:
        alphas = (1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8)
    lo, hi = float(np.min(ytr)), float(np.max(ytr))
    mu = float(ytr.mean())
    XtX = Xtr.T @ Xtr
    Xty = Xtr.T @ (ytr - mu)
    I = np.eye(Xtr.shape[1])
    best, best_a = None, None
    for al in alphas:
        w = np.linalg.solve(XtX + al * I, Xty)
        pv = np.clip(Xva @ w + mu, lo, hi)
        s = ev(pv, yva) if select == "pooled" else ev_within(pv, yva, ep_va)
        if s is None:
            continue
        if best is None or s > best:
            best, best_a = s, (al, w, pv)
    if best_a is None:
        return None, None, None, None
    al, w, pv = best_a
    return np.clip(Xte @ w + mu, lo, hi), al, best, pv


def fit_mlp(Xtr, ytr, Xva, yva, Xte, seed=0, epochs=400, hidden=128, lr=1e-3,
            device="cpu", ep_va=None, select="pooled"):
    """MLP + 早停。`select` 同 fit_ridge（pooled=EV / within=EV_within，见修订 4）。"""
    torch.manual_seed(seed)
    d = Xtr.shape[1]
    net = torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.ReLU(),
                              torch.nn.Linear(hidden, hidden // 2), torch.nn.ReLU(),
                              torch.nn.Linear(hidden // 2, 1)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    lo, hi = float(np.min(ytr)), float(np.max(ytr))
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=device)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=device)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)
    n = Xtr_t.shape[0]
    best, best_state, bad = None, None, 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, 1024):
            k = perm[i:i + 1024]
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(net(Xtr_t[k]).squeeze(-1), ytr_t[k])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            pv = np.clip(net(Xva_t).squeeze(-1).cpu().numpy().astype(np.float64),
                         lo, hi)
        s = ev(pv, yva) if select == "pooled" else ev_within(pv, yva, ep_va)
        if s is None:
            continue
        if best is None or s > best:
            best, best_state, bad = s, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 40:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        p = np.clip(net(Xte_t).squeeze(-1).cpu().numpy().astype(np.float64), lo, hi)
        pva = np.clip(net(Xva_t).squeeze(-1).cpu().numpy().astype(np.float64), lo, hi)
    return p, best, pva


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--mode", choices=["stoch", "det"], default="stoch")
    ap.add_argument("--frames", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--models", default="ridge,mlp")
    ap.add_argument("--repeats", type=int, default=1,
                    help="MLP 多 seed 重复，报均值（主判据用 mean）")
    ap.add_argument("--save-npz", default=None,
                    help="把逐帧特征/回报/价值存成 npz（离线重拟合用，免重跑 rollout）")
    ap.add_argument("--out", default=None)
    ap.add_argument("--verbose", action="store_true")
    # ---- 修订 4（2026-09-13 深夜，见预注册 §8）：估计器必须与主判据同口径，且必须过阳性对照 ----
    ap.add_argument("--npz", default=None,
                    help="从 --save-npz 落盘的逐帧数据离线重拟合（免重跑 rollout）")
    ap.add_argument("--select", choices=["pooled", "within"], default="pooled",
                    help="模型选择口径（修订 4）：pooled=EV（预注册初版）；"
                         "within=EV_within（与主判据同口径）")
    ap.add_argument("--center-target", action="store_true",
                    help="修订 4：目标改为**局内中心化回报** y−ȳ_ep（只留局内分量；"
                         "免费给估计器每局的均值 ⇒ 对 P2 保守）")
    ap.add_argument("--positive-control", action="store_true",
                    help="修订 4 闸门 G5：用**构造的、必然可学**的局内信号跑同一条流水线；"
                         "连它也做不到 EV_within≥0.15 ⇒ 估计器不合格，实验作废")
    ap.add_argument("--tag", default=None, help="输出 JSON 名后缀（多口径并存）")
    ap.add_argument("--sets", default="A,C,B", help="要拟合的特征集（逗号分隔）")
    ap.add_argument("--scramble-x", action="store_true",
                    help="修订 5 的**证伪对照**：把每个特征集的行按『局内位置 k 相同、"
                         "但来自不同局』置换（状态对不上、时钟对得上）。若真实目标的 "
                         "EV_within 不塌 ⇒ V3 的增益不是来自『状态』")
    ap.add_argument("--scramble-y", action="store_true",
                    help="下一轮预注册的**阴性对照**：把**目标**做同样的置换"
                         "（保留边缘分布/时钟结构，只破坏状态↔回报配对）。"
                         "合格估计器在此必须给 ≈0")
    ap.add_argument("--clock-baseline", action="store_true",
                    help="时钟基线对照：特征集换成**只有『局内第几帧』一个特征**"
                         "（其余流程不变）⇒ 量出『时间剖面』能解释多少局内方差")
    a = ap.parse_args()

    cfg = TrainConfig.resolve("economy")
    run_dir = a.run_dir or os.path.dirname(os.path.abspath(a.ckpt))
    cp = os.path.join(run_dir, "config.json")
    if os.path.exists(cp):
        d = json.load(open(cp, encoding="utf-8"))
        for k, vv in d.items():
            if hasattr(cfg, k):
                try:
                    setattr(cfg, k, vv)
                except Exception:
                    pass
    device = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"
    print(f"=== POMDP ceiling probe | ckpt={a.ckpt} mode={a.mode} "
          f"frames={a.frames} device={device} ===", flush=True)

    X, y, v, ep = rollout(a, cfg, device) if not a.npz else load_npz(a.npz)

    if a.save_npz and not a.npz:
        d = os.path.dirname(a.save_npz)
        if d:
            os.makedirs(d, exist_ok=True)
        np.savez_compressed(a.save_npz, y=y.astype(np.float32), v=v.astype(np.float32),
                            ep=ep, **{f"X_{k}": X[k].astype(np.float16) for k in X})
        print(f"[dump] 逐帧特征/回报已存 {a.save_npz}"
              f"（可离线重拟合，免重跑 {len(y)} 帧 rollout）", flush=True)

    # ---- 修订 4：阳性对照（闸门 G5）与目标口径 ----
    tot_real, bet_real, wit_real = var_decomp(y, ep)
    res_ctrl = None
    if a.positive_control:
        y, res_ctrl = make_positive_control(X["A"], ep, seed=a.seed,
                                            within_var=wit_real, between_var=bet_real)
        print(f"[control G5] 阳性对照目标已构造：局内方差={wit_real:.4f}、"
              f"局间方差={bet_real:.4f}（与真实回报一致）；信号 = 8 个真实特征列的"
              f"非线性函数 ⇒ 局内信号**必然存在且可学**", flush=True)
    if a.center_target:
        yc = y.astype(np.float64).copy()
        for e in np.unique(ep):
            m = (ep == e)
            yc[m] -= yc[m].mean()
        y = yc
        print("[target] 目标已改为局内中心化回报（y−ȳ_ep）", flush=True)
    if a.scramble_y:
        # 注意顺序：先中心化再打散 ⇒ 打散后每局拿到的是**别的局的整条中心化序列**，
        # 边缘分布/时钟结构逐值保留（见 scramble_target_by_clock 的 docstring）。
        y = scramble_target_by_clock(y, ep, seed=a.seed)
        print("[scramble-Y] 目标已按同一置换打散（阴性对照：状态↔回报配对被破坏）",
              flush=True)

    # ---- 时钟基线对照（只给"局内第几帧"一个特征）----
    if a.clock_baseline:
        kk = np.zeros(len(ep), dtype=np.float32)
        for e in np.unique(ep):
            idx = np.where(ep == e)[0]
            kk[idx] = np.arange(len(idx))
        X = {"A": kk.reshape(-1, 1)}
        a.sets = "A"
        print("[clock-baseline] 特征集已替换为单一『局内第几帧』特征", flush=True)

    # ---- 修订 5：证伪对照（打散状态、保留时钟）----
    if a.scramble_x:
        X = {k: scramble_rows_by_clock(v, ep, seed=a.seed) for k, v in X.items()}
        print("[scramble-X] 已按『局内位置相同、跨局置换』打散状态（时钟保留）", flush=True)

    # ---- 闸门 G1/G3 ----
    n_ep = int(ep.max()) + 1
    ok_g1 = (len(y) >= 30000 and n_ep >= 60)
    ok_g3 = bool(np.all(np.isfinite(y)) and np.all(np.isfinite(v))
                 and all(np.all(np.isfinite(X[k])) for k in X))

    # ---- 局分组留出（R9④）----
    rng = np.random.default_rng(a.seed)
    eps = np.unique(ep)
    rng.shuffle(eps)
    n_test = max(1, int(round(len(eps) * a.test_frac)))
    n_val = max(1, int(round(len(eps) * a.val_frac)))
    test_eps = set(eps[:n_test].tolist())
    val_eps = set(eps[n_test:n_test + n_val].tolist())
    m_te = np.array([e in test_eps for e in ep])
    m_va = np.array([e in val_eps for e in ep])
    m_tr = ~(m_te | m_va)
    print(f"[split] train={m_tr.sum()} val={m_va.sum()} test={m_te.sum()} "
          f"(episodes {len(eps) - n_test - n_val}/{n_val}/{n_test})", flush=True)

    tot, between, within = var_decomp(y, ep)
    print(f"[var] Var(R)={tot:.4f}  between(局均值)={between:.4f} "
          f"within(局内)={within:.4f}  within/total={within / max(1e-12, tot):.4f}",
          flush=True)
    print(f"[critic] EV_pooled={ev(v, y):+.4f}  EV_within={ev_within(v, y, ep):+.4f}  "
          f"EV_within_raw={ev_within_raw(v, y, ep):+.4f}  "
          f"std(V)/std(R)={v.std() / max(1e-12, y.std()):.4f}", flush=True)
    wm, nw = ev_win128_med(v, y, ep)
    print(f"[critic] EV_win128_med={wm if wm is None else round(wm, 4)} (n={nw})",
          flush=True)

    res = {"ckpt": a.ckpt, "mode": a.mode, "frames": int(len(y)),
           "episodes": int(n_ep), "seed": a.seed,
           "select": a.select, "center_target": bool(a.center_target),
           "positive_control": bool(a.positive_control),
           "scramble_x": bool(a.scramble_x), "scramble_y": bool(a.scramble_y),
           "source_npz": a.npz,
           "var_real_total": tot_real, "var_real_between": bet_real,
           "var_real_within": wit_real,
           "control_spec": res_ctrl,
           "gate_g1_frames_episodes": bool(ok_g1), "gate_g3_finite": bool(ok_g3),
           "var_total": tot, "var_between": between, "var_within": within,
           "critic": {"EV_pooled": ev(v, y), "EV_within": ev_within(v, y, ep),
                      "EV_within_raw": ev_within_raw(v, y, ep),
                      "EV_win128_med": wm,
                      "vstd_rstd": float(v.std() / max(1e-12, y.std()))},
           "sets": {}}

    ytr, yva, yte = y[m_tr], y[m_va], y[m_te]
    ep_te = ep[m_te]
    ep_va = ep[m_va]
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    for name in [s.strip() for s in a.sets.split(",") if s.strip()]:
        Xa = X[name]
        # 保信息的列过滤（修订 3b）：丢掉**训练集内零方差**的列——它们不含任何信息，
        # 却会让 d 虚高（8648 维展平网格里绝大多数格子从未出现）并放大外推爆炸风险。
        sd_all = Xa[m_tr].std(axis=0)
        keep = sd_all > 1e-8
        Xk = Xa[:, keep]
        Xtr, Xva, Xte = standardize(Xk[m_tr], [Xk[m_tr], Xk[m_va], Xk[m_te]])
        row = {"dim_raw": int(Xa.shape[1]), "dim": int(Xk.shape[1])}
        if "ridge" in models:
            p, al, sv, pva = fit_ridge(Xtr, ytr, Xva, yva, Xte, ep_va=ep_va,
                                       select=a.select)
            if p is not None:
                row["ridge"] = {"alpha": al, "val_sel": sv,
                                "val_EV": ev(pva, yva),
                                "val_EV_within": ev_within(pva, yva, ep_va),
                                "EV_pooled": ev(p, yte),
                                "EV_within": ev_within(p, yte, ep_te),
                                "EV_within_raw": ev_within_raw(p, yte, ep_te),
                                "EV_win128_med": ev_win128_med(p, yte, ep_te)[0]}
        if "mlp" in models:
            accs = {"EV_pooled": [], "EV_within": [], "EV_within_raw": [],
                    "EV_win128_med": [], "val_EV": [], "val_EV_within": [],
                    "val_sel": []}
            for r in range(a.repeats):
                p, sv, pva = fit_mlp(Xtr, ytr, Xva, yva, Xte, seed=a.seed + r,
                                     device=device, ep_va=ep_va, select=a.select)
                accs["EV_pooled"].append(ev(p, yte))
                accs["EV_within"].append(ev_within(p, yte, ep_te))
                accs["EV_within_raw"].append(ev_within_raw(p, yte, ep_te))
                accs["EV_win128_med"].append(ev_win128_med(p, yte, ep_te)[0])
                accs["val_EV"].append(ev(pva, yva))
                accs["val_EV_within"].append(ev_within(pva, yva, ep_va))
                accs["val_sel"].append(sv)
            row["mlp"] = {k: {"mean": (float(np.mean([x for x in vs if x is not None]))
                                       if any(x is not None for x in vs) else None),
                              "all": vs} for k, vs in accs.items()}
        res["sets"][name] = row
        print(f"[fit] set={name} dim={row['dim']}（原始 {row['dim_raw']}，"
              f"丢弃训练集零方差列 {row['dim_raw'] - row['dim']}）", flush=True)
        for mk in ("ridge", "mlp"):
            if mk in row:
                if mk == "ridge":
                    g = row[mk]
                    print(f"       ridge a={g['alpha']:<8g} val_EV={g['val_EV']:+.4f} "
                          f"val_EV_within={g['val_EV_within']:+.4f} | "
                          f"EV_pooled={g['EV_pooled']:+.4f} "
                          f"EV_within={g['EV_within']:+.4f} "
                          f"EV_within_raw={g['EV_within_raw']:+.4f} "
                          f"win128={g['EV_win128_med']}", flush=True)
                else:
                    g = row[mk]
                    print(f"       mlp   val_EV={g['val_EV']['mean']:+.4f} "
                          f"val_EV_within={g['val_EV_within']['mean']:+.4f} | "
                          f"EV_pooled={g['EV_pooled']['mean']:+.4f} "
                          f"EV_within={g['EV_within']['mean']:+.4f} "
                          f"EV_within_raw={g['EV_within_raw']['mean']:+.4f} "
                          f"win128={g['EV_win128_med']['mean']}", flush=True)
    # ---- 预注册判据（照 §3 读）----
    def g(setname, model, key):
        try:
            x = res["sets"][setname][model][key]
            return x.get("mean") if isinstance(x, dict) else x
        except Exception:
            return None

    eB, eC, eA = (g("B", "mlp", "EV_within"), g("C", "mlp", "EV_within"),
                  g("A", "mlp", "EV_within"))
    crit_w = res["critic"]["EV_within"]
    # 估计器发散闸（预注册修订 3，2026-09-13）：若**全部**模型在**留出验证集**上都不如
    # "预测训练均值"（val_EV ≤ 0），说明估计器在整局留出上发散 ⇒ 判据不可读，run 作废。
    vvals = []
    for nm in ("A", "C", "B"):
        rr = res["sets"].get(nm) or {}
        if "ridge" in rr and rr["ridge"].get("val_EV") is not None:
            vvals.append(rr["ridge"]["val_EV"])
        if "mlp" in rr:
            mv = (rr["mlp"].get("val_EV") or {}).get("mean")
            if mv is not None:
                vvals.append(mv)
    estimator_ok = bool(vvals) and max(vvals) > 0.0
    # 闸门 G5（修订 4）：阳性对照——目标里**按构造存在**局内信号，估计器必须能把它捞出来。
    # 捞不出来（三集最好者仍 < 0.15）⇒ 估计器无能 ⇒ 作废，**不得**读成 P2。
    ctrl_best, ctrl_pass = None, None
    if a.positive_control:
        cand = [x for x in (eA, eC, eB) if x is not None]
        ctrl_best = (max(cand) if cand else None)
        ctrl_pass = bool(ctrl_best is not None and ctrl_best >= 0.15)
    gate_ok = bool(ok_g1 and ok_g3 and estimator_ok and ctrl_pass is not False
                   and crit_w is not None and crit_w <= 0.05)
    if not estimator_ok or ctrl_pass is False:
        verdict = "INVALID_ESTIMATOR"
    else:
        verdict = "GATE_FAIL"
    if gate_ok and None not in (eB, eC):
        dBC = eB - eC
        if eB >= 0.15 and dBC >= 0.08:
            verdict = "P1_H_POMDP_SUPPORT"
        elif eB <= 0.05:
            verdict = "P2_H_STRUCT_FALSIFIED_POMDP"
        elif eC >= 0.15 and dBC < 0.08:
            verdict = "P3_H_LEARN"
        else:
            verdict = "P0_GRAY"
    res["criteria"] = {"EV_within_A": eA, "EV_within_C": eC, "EV_within_B": eB,
                       "delta_B_minus_C": (None if None in (eB, eC) else eB - eC),
                       "critic_EV_within": crit_w,
                       "gate_g1": bool(ok_g1), "gate_g3": bool(ok_g3),
                       "gate_estimator_ok": bool(estimator_ok),
                       "estimator_val_EVs": vvals,
                       "gate_g5_positive_control": ctrl_pass,
                       "control_EV_within_best": ctrl_best,
                       "gate_critic_le_0.05": (crit_w is not None and crit_w <= 0.05),
                       "verdict": verdict}
    print("\n=== 判据（照预注册 §3 读）===", flush=True)
    print(f"  EV_within  A={eA} C={eC} B={eB}  (ΔB−C="
          f"{res['criteria']['delta_B_minus_C']})", flush=True)
    print(f"  critic EV_within={crit_w}  gate G1={ok_g1} G3={ok_g3} "
          f"critic<=0.05={res['criteria']['gate_critic_le_0.05']}", flush=True)
    print(f"  估计器闸 estimator_ok={estimator_ok}  val_EVs="
          f"{[None if x is None else round(x, 4) for x in vvals]}", flush=True)
    if a.positive_control:
        print(f"  G5 阳性对照（目标里必然有局内信号）: best EV_within="
              f"{None if ctrl_best is None else round(ctrl_best, 4)}  通过={ctrl_pass}",
              flush=True)
    print(f"  VERDICT = {verdict}", flush=True)

    tag = f"_{a.tag}" if a.tag else ""
    out = a.out or os.path.join(_ROOT, "docs",
                                f"pomdp_ceiling_probe_{a.mode}{tag}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"[out] {out}", flush=True)


if __name__ == "__main__":
    main()
