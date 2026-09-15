# -*- coding: utf-8 -*-
"""信念输入准确度实测（只读）：belief_token 里那几项**到底准不准**。

背景：belief_token 563 维里含"对手手牌后验 / 下一张 / 圣水估计"。设计改动前必须先量：
  A) 手牌/下一张：`CycleBayesFilter` 锁定后是**精确 0/1**（bayes_filter.py:57-59）⇒ 只量
     未锁定（前几帧）阶段的准确率 + 锁定帧占比；
  B) 圣水：`BeliefInference._tick_elixir`（belief.py:288-297）用的是 **硬编码 dt/2.8**
     ——而引擎圣水回复是**分段**的（battle.py:2690：t<120 → 2.8s/点；120≤t<240 → 1.4s/点；
     t≥240 → 2.8/3 s/点）⇒ 疑似**系统性低估**，本脚本用真值量化它；
  C) 给出"分段正确的圣水账"在同一条轨迹上的 MAE，作为改进量的**上界读数**。

用法（必须在 src/clasher_new 下运行）：
    cd src/clasher_new && PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe \
        ../../scripts/probe_belief_accuracy.py --episodes 6
"""
from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src", "clasher_new"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

from rl.env_wrapper import RLEnv
from rl.belief import BeliefInference
from card_utils import Card


def regen_rate(t: float) -> float:
    """引擎真实圣水回复速率（秒/点），与 battle.py:2690 同源。"""
    return 2.8 if t < 120 else (1.4 if t < 240 else 2.8 / 3.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--particles", type=int, default=128)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    env = RLEnv(opponent=None, seed=a.seed)          # None = 随机对手
    obs, _ = env.reset(seed=a.seed)
    belief = BeliefInference(opp_deck=env.deck1, n_particles=a.particles, seed=a.seed)
    rows = []       # 每决策帧一行（严格 1 帧 1 行，训练环同序）
    exact = []      # 分段正确的圣水账（只用可见事实）
    loses, caps = [], []
    e_exact, last_t = 5.0, None
    for ep in range(a.episodes):
        obs, _ = env.reset(seed=a.seed + ep)
        belief.reset(env.deck1)                     # ⚠️ reset 会重洗 deck ⇒ 每局重算映射
        deckpos = {c: i for i, c in enumerate(env.deck1)}
        e_exact, last_t = 5.0, None
        for _ in range(400):
            p1 = env.battle.players[1]
            t = float(env.battle.time)
            st = belief.state()                      # 训练环同序：encode 在 step 之前
            true_hand = set(p1.cycle[:4])
            hp = st.hand_probs
            top4 = {env.deck1[i] for i in np.argsort(-hp)[:4]}
            np_ok = env.deck1[int(np.argmax(st.next_probs))] == p1.cycle[4]
            e_true = float(p1.elixir)
            rows.append((t, bool(belief.rule.locked), top4 == true_hand,
                         float(sum(hp[deckpos[c]] for c in true_hand)),
                         np_ok, float(st.elixir_mean), e_true))
            # 分段正确的账：时间推进那段用**推进起点时刻**的回复率
            if last_t is not None:
                e_exact = e_exact + (t - last_t) / regen_rate(last_t)
            last_t = t
            exact.append((t, min(10.0, e_exact), e_true))
            caps.append(e_true >= 9.99)
            # 推进一帧
            obs2, _r, term, trunc, info = env.step(_noop_bundle())
            for it in (info.get("opp_played") or []):
                nm = it.get("card") if isinstance(it, dict) else None
                if nm and nm != "__ability__" and nm in deckpos:
                    e_exact = e_exact - Card(nm).elixir
                    loses.append(Card(nm).elixir)
            e_exact = float(np.clip(e_exact, 0.0, 10.0))
            belief.update(obs2, info.get("opp_played"))   # 训练环同序
            obs = obs2
            if term or trunc:
                break

    # 列序：0 t, 1 locked, 2 hand_ok, 3 mass, 4 next_ok, 5 est, 6 true
    R = np.array(rows, dtype=float)
    T, locked = R[:, 0], R[:, 1] > 0.5
    hand_ok, mass, next_ok = R[:, 2], R[:, 3], R[:, 4]
    est, true = R[:, 5], R[:, 6]
    print(f"[rollout] {a.episodes} 局，{len(R)} 决策帧，粒子={a.particles}")
    print(f"\n[A] 循环信念（手牌/下一张）")
    print(f"  锁定帧占比（概率精确 0/1） = {100*locked.mean():.1f}%"
          f"  ⇒ 未锁定 {int((~locked).sum())} 帧")
    for tag, m in (("全部帧", np.ones(len(R), bool)), ("锁定帧", locked),
                   ("未锁定帧", ~locked)):
        if m.sum() == 0:
            continue
        print(f"  [{tag:<6}] 手牌集合 top-4 命中率 = {100*hand_ok[m].mean():5.1f}% | "
              f"真手牌概率质量和 = {mass[m].mean():.4f}（4.0=完全正确）| "
              f"下一张 top-1 命中 = {100*next_ok[m].mean():5.1f}%")

    # 锁定流是否被反复打断（未锁定帧的成因）
    runs, cur, n_sw = [], 0, 0
    for i in range(len(R)):
        if not locked[i]:
            cur += 1
        elif cur:
            runs.append(cur); cur = 0; n_sw += 1
    if cur:
        runs.append(cur); n_sw += 1
    if runs:
        runs = np.array(runs)
        print(f"  未锁定「段」：{n_sw} 段 / {len(R)} 帧 ⇒ 每局 {n_sw/a.episodes:.1f} 段、"
              f"段长 中位 {int(np.median(runs))} 帧（最长 {int(runs.max())}）"
              f" ⇒ {'反复打断' if n_sw/a.episodes > 1.5 else '多为开局未锁定'}")

    print(f"\n[B] 圣水估计（现状：_tick_elixir 硬编码 dt/2.8）")
    err = est - true
    print(f"  全部帧 MAE = {np.abs(err).mean():.3f} 圣水 | 偏置(估计−真值) = {err.mean():+.3f}")
    for lo, hi, tag in ((0, 120, "t<120s"), (120, 240, "120≤t<240s（双倍）"),
                        (240, 1e9, "t≥240s（三倍）")):
        m = (T >= lo) & (T < hi)
        if m.sum():
            e = err[m]
            print(f"  [{tag:<16}] n={int(m.sum()):>4} MAE={np.abs(e).mean():.3f} "
                  f"偏置={e.mean():+.3f} 最负={e.min():+.3f}")

    E = np.array([[r[1], r[2]] for r in exact], dtype=float)
    Tn = np.array([r[0] for r in exact], dtype=float)
    if len(E):
        ee = E[:, 0] - E[:, 1]
        print(f"\n[C] 对照：**分段正确**的圣水账（只用时间+观测出牌）")
        print(f"  全部帧 MAE = {np.abs(ee).mean():.3f} | 偏置 = {ee.mean():+.3f}")
        for lo, hi, tag in ((0, 120, "t<120s"), (120, 240, "120≤t<240s"),
                            (240, 1e9, "t≥240s")):
            m = (Tn >= lo) & (Tn < hi)
            if m.sum():
                print(f"  [{tag:<16}] n={int(m.sum()):>4} MAE={np.abs(ee[m]).mean():.3f} "
                      f"偏置={ee[m].mean():+.3f}")
        print(f"  ⇒ 改动潜在收益（MAE 下降）："
              f"{np.abs(err).mean():.3f} → {np.abs(ee).mean():.3f} "
              f"（{100*(1-np.abs(ee).mean()/max(1e-9,np.abs(err).mean())):.0f}% 降幅）")


def _noop_bundle():
    from rl.action_bundle import ActionBundle
    return ActionBundle.noop()


def _noop_step(env):
    return env.step(_noop_bundle())


if __name__ == "__main__":
    main()
