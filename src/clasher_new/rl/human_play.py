"""人机对战 + 人类数据采集（--mode 无；独立模块 + dashboard --play 集成）。

- 人（player-0）vs 训练模型（player-1，FollowerPolicy，deterministic）实时对战；
- 每步记录两类训练数据：
  * **EpisodeReplay**（rl/replay.py）：(obs, bundle, reward, opp_played, hidden)
    → ``to_belief_dataset()`` 喂 ``train_belief --replays-path``（信念监督）；
  * **BC 样本**：(obs, belief_tok, plan_vec, bundle, masks)
    → ``train_bc_from_human`` 行为克隆 / 模仿学习预训练。
- 对局按决策 tick 推进（RLEnv.step 每次 0.5s 战斗时间），人每出一个动作推进一帧。

数据落盘（--play-out，默认 runs/<name>/human_data/）：
- episode_<ts>.pkl    EpisodeReplay（含 hidden 特权标签）
- bc_<ts>.pkl         BC 样本列表
- session_<ts>.json   元数据（胜负/步数/累计奖励/样本数）

用法（直接跑/无 UI 采集测试）：
    python rl/human_play.py --policy runs/solo/solo_main.pt --drive-games 3
    python rl/human_play.py --export runs/solo/human_data --out-belief belief.pkl --out-bc bc.pkl
    python rl/human_play.py --bc-train --data-dir runs/solo/human_data --out follower_human.pt
"""

import os
import sys
import json
import time
import random
import argparse
import datetime

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import numpy as np

from rl.env_wrapper import RLEnv
from rl.action_bundle import ActionBundle, K_MAX
from rl.belief import BeliefInference
from rl.belief_planner import BeliefPlanner
from rl.plan_space import PLAN_DIM
from rl.follower import FollowerPolicy, save_checkpoint, load_checkpoint
from rl.config import TrainConfig, reward_to_env
from rl.replay import EpisodeReplay, battle_snapshot
from rl.train_follower import FollowerOpponent
from rl.run_league import _bundle_cards

#: 固定卡组（默认镜像 8 卡；可 --deck 覆盖）
DEFAULT_PLAY_DECK = ["Knight", "MiniPekka", "Arrows", "Minions",
                     "Musketeer", "Fireball", "Giant", "Archer"]


def _make_env(cfg, deck, seed):
    return RLEnv(opponent=None, seed=seed, reward_weights=reward_to_env(cfg),
                 card_level=cfg.card_level, deck0=deck, deck1=deck,
                 record_hidden=True)


def load_policy(path, hidden_dim=128):
    """加载 FollowerPolicy checkpoint（带元数据）。"""
    return load_checkpoint(path, hidden_dim=hidden_dim)


class HumanPlaySession:
    """一局人机对战：人出 ActionBundle，模型对手每 tick 应对；全程记录训练数据。"""

    def __init__(self, policy, cfg=None, deck=None, seed=0, max_steps=600,
                 out_dir=None, session_id=None):
        self.cfg = cfg or TrainConfig.resolve("standard")
        self.deck = list(deck) if deck else list(DEFAULT_PLAY_DECK)
        self.seed = seed
        self.max_steps = int(max_steps)
        self.out_dir = out_dir
        self.session_id = session_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.env = _make_env(self.cfg, self.deck, seed)
        self.policy = policy
        self.opp = FollowerOpponent(policy, self.env,
                                    belief=BeliefInference(opp_deck=self.deck,
                                                           n_particles=128, seed=seed),
                                    deterministic=True)
        self.env.opponent = self.opp
        self.belief = BeliefInference(opp_deck=self.deck, n_particles=128, seed=seed)
        self.bp = BeliefPlanner()
        self.rec = EpisodeReplay(record_hidden=True)
        self.bc_samples = []
        self.winner = None
        self.reward_total = 0.0
        self.steps = 0
        self.game_over = False
        self._hidden = None
        self.start()

    # ---- 生命周期 ----

    def start(self):
        obs, _ = self.env.reset()
        self.belief.reset(self.env.deck1)
        self.rec.start()
        self._hidden = None
        return self.state()

    def state(self):
        """返回给前端的战场帧 + 手牌 + 合法落点（battle_snapshot 帧可直接复用渲染器）。"""
        fr = battle_snapshot(self.env.battle, ActionBundle.noop(), 0.0, {})
        mask = self.env.get_action_mask_for(0)
        hand = [self.env.battle.players[0].cycle[i] for i in range(len(mask["slots"]))]
        legal = {i: [[int(c % 18), int(c // 18)] for c in np.flatnonzero(mask["cells"][i])]
                 for i in range(len(mask["slots"])) if mask["slots"][i]}
        return {
            "ok": True,
            "frame": fr,
            "hand": hand,
            "legal": legal,
            "elixir": float(self.env.battle.players[0].elixir),
            "game_over": bool(self.game_over),
            "winner": self.winner,
            "reward_total": round(self.reward_total, 3),
            "steps": int(self.steps),
            "max_steps": int(self.max_steps),
            "deck": list(self.deck),
        }

    # ---- 动作 ----

    def act(self, slot, x, y):
        """人类出牌：(slot 1..K_MAX, x, y 本地坐标)；slot=0 = 空过等待（noop 推进 0.5s 回圣水）。
        返回 state；非法动作返回 error。"""
        if self.game_over:
            return {"ok": False, "error": "本局已结束，请开新局"}
        slot = int(slot)
        if slot == 0:
            bundle = ActionBundle.noop()   # 空过：推进 0.5s 回圣水
        else:
            mask = self.env.get_action_mask_for(0)
            si = slot - 1
            if not (0 <= si < len(mask["slots"])) or not mask["slots"][si]:
                return {"ok": False, "error": f"slot {slot} 非法（圣水不足或手牌槽空）"}
            cells = mask["cells"][si]          # (GRID_H=32, GRID_W=18) 布尔网格
            if (y < 0 or y >= cells.shape[0] or x < 0 or x >= cells.shape[1]
                    or not bool(cells[y][x])):
                return {"ok": False, "error": f"({x},{y}) 对该牌非法"}
            bundle = ActionBundle.from_single(slot, int(x), int(y))
        obs = self.env.observe(0)
        belief_tok = self.belief.encode(obs, None)
        plan = self.bp.plan(self.env.battle, self.belief.state(), obs)
        plan_vec = plan.to_vector()
        # BC 样本：掩码在采集时刻生成（battle 状态随后会变）
        masks = self.policy.masks_for(obs, belief_tok, plan_vec, bundle,
                                      self.env.get_action_mask)
        obs2, reward, term, trunc, info = self.env.step(bundle)
        self.rec.record_step(obs, bundle, reward, info)   # hidden 随 info 记录
        self.bc_samples.append((obs, belief_tok, plan_vec, bundle, masks))
        played = _bundle_cards(bundle, obs)
        self.opp.observe_opponent_played(played)
        self.belief.update(obs2, info.get("opp_played"))
        self.reward_total += float(reward)
        self.steps += 1
        self.game_over = bool(term or trunc) or self.steps >= self.max_steps
        if self.game_over:
            self.winner = self.env.battle.winner
        return self.state()

    # ---- 落盘 ----

    def save(self, out_dir=None, keep_bc=True):
        """把本局 EpisodeReplay + BC 样本落盘；返回 (ep_path, bc_path, meta_path)。"""
        out_dir = out_dir or self.out_dir
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        ep = self.rec.end()
        meta = {"session_id": self.session_id, "winner": self.winner,
                "steps": self.steps, "reward_total": round(self.reward_total, 3),
                "n_bc": len(self.bc_samples), "deck": list(self.deck),
                "ts": time.time()}
        ep_path = bc_path = meta_path = None
        if out_dir:
            ep_path = os.path.join(out_dir, f"episode_{self.session_id}.pkl")
            import pickle
            with open(ep_path, "wb") as f:
                pickle.dump([{**ep, "deck": list(self.deck)}], f)
            if keep_bc and self.bc_samples:
                bc_path = os.path.join(out_dir, f"bc_{self.session_id}.pkl")
                with open(bc_path, "wb") as f:
                    pickle.dump(self.bc_samples, f)
            meta_path = os.path.join(out_dir, f"session_{self.session_id}.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        return ep_path, bc_path, meta_path


# ---------------------------------------------------------------------------
# 采集到的数据 → 训练
# ---------------------------------------------------------------------------

def load_bc_samples(data_dir):
    """读取 data_dir 下所有 bc_*.pkl 的 BC 样本并合并。"""
    out = []
    for fn in sorted(os.listdir(data_dir)):
        if fn.startswith("bc_") and fn.endswith(".pkl"):
            with open(os.path.join(data_dir, fn), "rb") as f:
                import pickle
                out.extend(pickle.load(f))
    return out


def train_bc_from_human(data_dir, out="follower_human.pt", epochs=5, lr=1e-3,
                        hidden_dim=128, seed=0):
    """在人类 BC 样本上做行为克隆（模仿学习预训练），产物可 --init-from 给 PPO。"""
    import torch
    samples = load_bc_samples(data_dir)
    if not samples:
        raise RuntimeError(f"{data_dir} 下无 bc_*.pkl 样本")
    torch.manual_seed(seed)
    np.random.seed(seed)
    belief_dim = len(samples[0][1])
    plan_dim = len(samples[0][2])
    policy = FollowerPolicy(hidden=hidden_dim, plan_dim=plan_dim, belief_dim=belief_dim)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    print(f"[bc-human] {len(samples)} 样本，belief_dim={belief_dim} plan_dim={plan_dim}")
    for ep in range(epochs):
        perm = np.random.permutation(len(samples))
        tot = 0.0
        for i in perm:
            obs, tok, plan, bundle, masks = samples[i]
            lp, _, _, _ = policy.evaluate(obs, tok, plan, bundle, masks, hidden=None)
            loss = -lp
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(lp.item())
        print(f"[bc-human] epoch {ep + 1}/{epochs} mean_logprob={tot / len(samples):.3f}")
    save_checkpoint(policy, out)
    print(f"[save] {out}")
    return policy


def export_data(data_dir, out_belief=None, out_bc=None):
    """把 human_data 目录导出成可直接训练的文件：
    - out_belief：EpisodeReplay 列表 pickle（喂 train_belief --replays-path）
    - out_bc：合并的 BC 样本 pickle（喂 train_bc_from_human）
    """
    import pickle
    if out_belief:
        replays = []
        for fn in sorted(os.listdir(data_dir)):
            if fn.startswith("episode_") and fn.endswith(".pkl"):
                with open(os.path.join(data_dir, fn), "rb") as f:
                    replays.extend(pickle.load(f))
        with open(out_belief, "wb") as f:
            pickle.dump(replays, f)
        print(f"[export] 信念回放 {len(replays)} 局 -> {out_belief}")
    if out_bc:
        samples = load_bc_samples(data_dir)
        with open(out_bc, "wb") as f:
            pickle.dump(samples, f)
        print(f"[export] BC 样本 {len(samples)} -> {out_bc}")
    return out_belief, out_bc


def drive_games(policy, n_games, seed=0, max_steps=600, out_dir=None,
                cfg=None, deck=None):
    """无 UI 冒烟：用随机合法动作驱动 n 局，验证数据采集链路（headless 测试）。"""
    cfg = cfg or TrainConfig.resolve("standard")
    meta = []
    for g in range(n_games):
        sess = HumanPlaySession(policy, cfg=cfg, deck=deck, seed=seed + g,
                                max_steps=max_steps, out_dir=out_dir,
                                session_id=f"drive{g}_{int(time.time())}")
        rng = random.Random(seed + g)
        while not sess.game_over:
            st = sess.state()
            legal = st["legal"]
            if not legal:
                # 圣水不足：空过等待（noop 推进 0.5s 回圣水）
                st = sess.act(0, 0, 0)
                if not st.get("ok", True):
                    break
                continue
            slot = rng.choice(list(legal.keys()))
            cell = rng.choice(legal[slot])
            st = sess.act(slot + 1, cell[0], cell[1])
            if not st.get("ok", True):
                break
        ep, bc, mt = sess.save(out_dir=out_dir)
        print(f"[drive] game {g}: winner={sess.winner} steps={sess.steps} "
              f"reward={sess.reward_total:.3f} bc={len(sess.bc_samples)}")
        meta.append({"winner": sess.winner, "steps": sess.steps,
                     "reward_total": sess.reward_total, "bc": len(sess.bc_samples)})
    return meta


def main():
    ap = argparse.ArgumentParser(description="人机对战数据采集/导出/BC 训练")
    ap.add_argument("--policy", type=str, default=None, help="FollowerPolicy checkpoint")
    ap.add_argument("--config", type=str, default="standard")
    ap.add_argument("--out-dir", type=str, default="runs/human_data")
    ap.add_argument("--drive-games", type=int, default=0,
                    help="无 UI 冒烟：随机动作打 N 局验证采集链路")
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deck", type=str, default=None,
                    help="逗号分隔卡名（缺省默认 8 卡镜像）")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--out-belief", type=str, default=None)
    ap.add_argument("--out-bc", type=str, default=None)
    ap.add_argument("--bc-train", action="store_true")
    ap.add_argument("--data-dir", type=str, default=None)
    ap.add_argument("--bc-out", type=str, default="follower_human.pt")
    ap.add_argument("--bc-epochs", type=int, default=5)
    ap.add_argument("--hidden-dim", type=int, default=128)
    args = ap.parse_args()

    deck = args.deck.split(",") if args.deck else None

    if args.export:
        data_dir = args.data_dir or args.out_dir
        export_data(data_dir, out_belief=args.out_belief, out_bc=args.out_bc)
        return

    if args.bc_train:
        data_dir = args.data_dir or args.out_dir
        train_bc_from_human(data_dir, out=args.bc_out, epochs=args.bc_epochs,
                            hidden_dim=args.hidden_dim, seed=args.seed)
        return

    if not args.policy:
        ap.error("需要 --policy 或 --export / --bc-train")
    policy = load_policy(args.policy, hidden_dim=args.hidden_dim)
    cfg = TrainConfig.resolve(args.config)
    if args.drive_games > 0:
        drive_games(policy, args.drive_games, seed=args.seed,
                    max_steps=args.max_steps, out_dir=args.out_dir, cfg=cfg, deck=deck)
    else:
        # 交互模式：标准输入读取 "slot,x,y" 或 "quit"
        sess = HumanPlaySession(policy, cfg=cfg, deck=deck, seed=args.seed,
                                max_steps=args.max_steps, out_dir=args.out_dir)
        print("[play] 输入 slot,x,y 出牌；quit 结束并保存")
        while not sess.game_over:
            line = input("> ").strip()
            if not line or line == "quit":
                break
            try:
                s, x, y = [int(t) for t in line.split(",")]
                st = sess.act(s, x, y)
                if not st.get("ok", True):
                    print("!!", st.get("error"))
            except (ValueError, IndexError):
                print("!! 格式: slot,x,y")
        sess.save(out_dir=args.out_dir)
        print(f"[play] 结束 winner={sess.winner} steps={sess.steps} "
              f"reward={sess.reward_total:.3f} bc={len(sess.bc_samples)}")


if __name__ == "__main__":
    main()
