"""信念模块监督训练（规划文档 8.3 / 阶段三）。

流程：采集 replay（含特权隐藏状态标签）→ 训练神经信念编码器
（下一张牌预测 + 手牌概率预测）→ 保存权重。

修复：
- P1-1：GRU 按 **episode 序列**训练（整局前向展开 + 逐时间步损失），不再是长度 1 的
  单帧 MLP；训练/推理分布一致（推理端喂最长 32 帧历史）。
- P1-17：按局切分验证集；报告 val next-acc / Brier / NLL；训练后做温度缩放并报告 ECE。
- P1-21：可直接消费 export_replay 产物（--replays-path），避免双采集。
"""

import os
import sys
import random
import argparse

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import numpy as np

from rl.env_wrapper import RLEnv
from rl.replay import EpisodeReplay
from rl.belief import NeuralBeliefEncoder, build_feature
from rl.observation import ENTITY_NAMES

DEFAULT_DECK_1 = ["Minions", "Archer", "MiniPekka", "Musketeer",
                  "Giant", "Fireball", "Arrows", "Knight"]


def sample_bundle(mask, rng):
    from rl.action_bundle import ActionBundle
    bundle = ActionBundle()
    slots = np.flatnonzero(mask["slots"])
    if slots.size == 0:
        return bundle
    slot = int(rng.choice(slots))
    cells = np.flatnonzero(mask["cells"][slot])
    if cells.size == 0:
        return bundle
    cell = int(rng.choice(cells))
    x, y = cell % 18, cell // 18
    bundle.add(slot + 1, x, y)
    return bundle


def collect_replays(n_games, seed, opponent=None, max_steps=600):
    env = RLEnv(opponent=opponent, seed=seed)
    rng = random.Random(seed)
    replays = []
    for g in range(n_games):
        obs, _ = env.reset()
        ep = EpisodeReplay(record_hidden=True)
        ep.start()
        done = False
        steps = 0
        while not done and steps < max_steps:
            mask = env.get_action_mask()
            bundle = sample_bundle(mask, rng)
            obs, r, term, trunc, info = env.step(bundle)
            ep.record_step(obs, bundle, r, info)
            done = term or trunc
            steps += 1
        replays.append({**ep.end(), "deck": list(env.deck1)})
        if (g + 1) % 10 == 0:
            print(f"[collect] game {g+1}/{n_games} done, steps={steps}", flush=True)
    return replays, env.deck1


def _episode_arrays(ep):
    """把一个 episode 容器转成 (feat_seq (T,D), y_next (T,), y_hand (T,deck))；无标签步跳过。"""
    steps = ep["steps"] if isinstance(ep, dict) else ep.steps
    deck = ep.get("deck") if isinstance(ep, dict) else None
    xs, yn, yh = [], [], []
    for st in steps:
        hidden = st.get("hidden")
        if hidden is None:
            continue
        xs.append(build_feature(st["obs"], st["opp_played"]))
        yn.append(int(hidden["opp_next"]))
        hand_ids = set(int(v) for v in hidden["opp_hand"])
        yh.append([1.0 if ENTITY_NAMES.index(c) in hand_ids else 0.0 for c in deck])
    if not xs:
        return None
    return np.stack(xs), np.array(yn, dtype=np.int64), np.stack(yh).astype(np.float32)


def build_episodes(replays):
    """返回 (episodes, deck)：episodes = [(T,D),(T,),(T,deck)] 序列列表。"""
    eps = []
    deck = DEFAULT_DECK_1
    for ep in replays:
        if isinstance(ep, dict) and ep.get("deck"):
            deck = list(ep["deck"])
        arr = _episode_arrays(ep)
        if arr is not None:
            eps.append(arr)
    return eps, deck


def brier_of(probs, y):
    """probs: (N,K)；y: (N,) one-hot 目标。返回平均 Brier 分数。"""
    k = probs.shape[1]
    yh = np.eye(k)[y]
    return float(np.mean(np.sum((probs - yh) ** 2, axis=1)))


def nll_of(logits, y):
    import torch
    lg = torch.as_tensor(logits, dtype=torch.float32)
    loss = torch.nn.functional.cross_entropy(lg, torch.as_tensor(y, dtype=torch.long))
    return float(loss.item())


def ece_of(probs, y, n_bins=10):
    """期望校准误差（按预测置信度分箱）。"""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == y).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf >= bins[i]) & (conf < bins[i + 1])
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(conf)) * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def fit_temperature(logits, y):
    """在验证集上拟合温度 T（最小化 NLL），返回 T。"""
    import torch
    lg = torch.as_tensor(logits, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.long)
    best_t, best_nll = 1.0, float("inf")
    for t in [0.5, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0, 5.0]:
        n = float(torch.nn.functional.cross_entropy(lg / t, yt).item())
        if n < best_nll:
            best_nll, best_t = n, t
    return best_t


def train(epochs=10, lr=1e-3, batch_size=64, n_games=50, seed=0, out="belief_encoder.pt",
          replays_path=None, opponent=None, val_frac=0.2, max_steps=600):
    import torch
    import torch.nn as nn

    if replays_path:
        import pickle
        with open(replays_path, "rb") as f:
            replays = pickle.load(f)
    else:
        replays, _ = collect_replays(n_games, seed, opponent=opponent, max_steps=max_steps)
    episodes, deck = build_episodes(replays)
    assert episodes, "无带标签的 episode 可训练"
    in_dim = episodes[0][0].shape[1]
    hand_dim = len(deck)
    print(f"[data] {len(episodes)} episodes  in_dim={in_dim} hand_dim={hand_dim}")

    # 按局切分验证集（P1-17：绝不按帧切，避免时序泄漏）
    rng = random.Random(seed)
    order = list(range(len(episodes)))
    rng.shuffle(order)
    n_val = max(1, int(len(order) * val_frac))
    val_idx = set(order[:n_val])
    tr_ep = [episodes[i] for i in order if i not in val_idx]
    va_ep = [episodes[i] for i in order if i in val_idx]

    enc = NeuralBeliefEncoder(in_dim=in_dim, hidden=64, num_classes=13, hand_dim=hand_dim)
    opt = torch.optim.Adam(list(enc.gru.parameters()) + list(enc.next_head.parameters())
                           + list(enc.hand_head.parameters()) + list(enc.belief_proj.parameters()), lr=lr)
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()

    def eval_metrics(episodes_, temperature=1.0):
        with torch.no_grad():
            logits_all, y_all, briers = [], [], []
            for (xs, yn, yh) in episodes_:
                seq = torch.from_numpy(xs).unsqueeze(0)       # (1, T, D)
                gru_out, _ = enc.gru(seq)                          # (1, T, H)
                lg = enc.next_head(gru_out[0]) / temperature       # (T, K)
                prob = torch.softmax(lg, dim=-1).numpy()
                logits_all.append(lg.numpy()); y_all.append(yn)
                # 手牌 BCE（Brier 用 next 概率即可）
                briers.append(brier_of(prob, yn))
            logits_all = np.concatenate(logits_all)
            y_all = np.concatenate(y_all)
            prob = torch.softmax(torch.as_tensor(logits_all), dim=-1).numpy()
            acc = float((prob.argmax(1) == y_all).mean())
            return (acc, float(np.mean(briers)), nll_of(logits_all, y_all),
                    ece_of(prob, y_all), prob, y_all)

    for ep_i in range(epochs):
        tot_loss, n_batch = 0.0, 0
        for (xs, yn, yh) in tr_ep:
            seq = torch.from_numpy(xs).unsqueeze(0)            # (1, T, D)
            yn_t = torch.from_numpy(yn)
            yh_t = torch.from_numpy(yh)
            gru_out, _ = enc.gru(seq)                               # (1, T, H)
            next_logits = enc.next_head(gru_out[0])                 # (T, K)
            hand_logits = enc.hand_head(gru_out[0])                 # (T, deck)
            loss = ce(next_logits, yn_t) + bce(hand_logits, yh_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += float(loss.item())
            n_batch += 1
        va_acc, va_brier, va_nll, va_ece, _, _ = eval_metrics(va_ep)
        print(f"[train] epoch {ep_i+1}/{epochs} loss={tot_loss/max(1,n_batch):.4f} "
              f"val_next_acc={va_acc:.3f} val_brier={va_brier:.3f} val_nll={va_nll:.3f} "
              f"val_ece={va_ece:.3f}", flush=True)

    # 温度缩放（P1-17）：在验证集拟合，降低 NLL / ECE
    with torch.no_grad():
        _, _, _, _, prob, y_all = eval_metrics(va_ep)
    # 用训练后输出拟合温度
    val_logits, val_y = [], []
    with torch.no_grad():
        for (xs, yn, yh) in va_ep:
            seq = torch.from_numpy(xs).unsqueeze(0)
            gru_out, _ = enc.gru(seq)
            val_logits.append(enc.next_head(gru_out[0]).numpy())
            val_y.append(yn)
    val_logits = np.concatenate(val_logits)
    val_y = np.concatenate(val_y)
    temp = fit_temperature(val_logits, val_y)
    va_acc_t, va_brier_t, va_nll_t, va_ece_t, _, _ = eval_metrics(va_ep, temperature=temp)
    print(f"[calib] temperature={temp:.2f}  val_next_acc={va_acc_t:.3f} val_brier={va_brier_t:.3f} "
          f"val_nll={va_nll_t:.3f} val_ece={va_ece_t:.3f}", flush=True)

    torch.save({
        "gru": enc.gru.state_dict(),
        "next_head": enc.next_head.state_dict(),
        "hand_head": enc.hand_head.state_dict(),
        "belief_proj": enc.belief_proj.state_dict(),
        "in_dim": in_dim,
        "hidden": 64,
        "num_classes": 13,
        "hand_dim": hand_dim,
        "temperature": temp,
    }, out)
    print(f"[save] {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--n-games", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="belief_encoder.pt")
    ap.add_argument("--replays-path", type=str, default=None,
                    help="直接消费 export_replay 产物（.pkl）")
    ap.add_argument("--opponent", type=str, default=None, choices=[None, "heuristic"],
                    help="采集对手类型（None=random）")
    ap.add_argument("--max-steps", type=int, default=600)
    args = ap.parse_args()
    train(epochs=args.epochs, n_games=args.n_games, seed=args.seed, out=args.out,
          replays_path=args.replays_path, opponent=args.opponent, max_steps=args.max_steps)
