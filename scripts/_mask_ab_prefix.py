# -*- coding: utf-8 -*-
"""临时 A/B（下划线前缀 = 不进正式仪器表）：`used` off-by-one 修复前后的对照。

- 旧实现 = **修复前**的 `env_wrapper.py`，由脚本**按提交哈希现取**
  （`git show 792736a:src/clasher_new/rl/env_wrapper.py`）写到临时目录再 import
  —— 树里**不放生产代码副本**（【R10】口径要能复算，但不要第二份真源）；
- 新实现 = 当前工作树。

三组对照（**同种子**）：
  A. 强制 `elixir = 10` 的部分 bundle 掩码：统计「已用槽位仍被判合法」的组数
     （圣水不再是瓶颈 ⇒ 暴露掩码层病灶）
  B. 真实 rollout（确定性采样，与评估路径同）：每帧
     · `mask` 里已用槽位是否仍合法（掩码层）
     · 采样出的 bundle 是否被 `validate_bundle` 拒（行为层）
     · 新加的掩码不变式是否抛 `RuntimeError`（守卫层）
  C. 逐例对照打印

用法（必须在 src/clasher_new 下运行）：
    cd src/clasher_new && PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe \
        ../../scripts/_mask_ab_prefix.py --games 10 --max-steps 200
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src", "clasher_new")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


#: 修复前的最后一个提交（`used` off-by-one 仍在内）
PREFIX_COMMIT = "792736a"
PREFIX_REL = "src/clasher_new/rl/env_wrapper.py"


def _load_prefix_env(prefix_py=None):
    """取**修复前**的 `env_wrapper.py`（`git show <commit>:<path>`，或 `--prefix-py` 给的副本），
    写到临时目录后 import，返回其 `RLEnv`。

    ⚠️ 目的是在**同一进程**里对照「修复前 vs 修复后」，不是在源码树里再放一份生产代码。
    """
    import importlib.util
    import subprocess
    import tempfile
    if prefix_py:
        src = open(prefix_py, encoding="utf-8").read()
    else:
        try:
            r = subprocess.run(["git", "show", f"{PREFIX_COMMIT}:{PREFIX_REL}"],
                               cwd=_ROOT, capture_output=True, text=True,
                               encoding="utf-8", check=True)
        except Exception as e:                                  # noqa: BLE001
            raise SystemExit(
                f"取修复前源码失败（{e!r}）。请先确认提交 {PREFIX_COMMIT} 在本仓库可达：\n"
                f"  git show {PREFIX_COMMIT}:{PREFIX_REL} | head\n"
                f"或显式给副本：--prefix-py <path>") from e
        src = r.stdout
    tmp = tempfile.mkdtemp(prefix="prefix_env_")
    path = os.path.join(tmp, "env_wrapper_prefix.py")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(src)
    spec = importlib.util.spec_from_file_location("env_wrapper_prefix", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.RLEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--seed-base", type=int, default=8000)
    ap.add_argument("--prefix-py", default=None,
                    help="可选的修复前 env_wrapper.py 副本（缺省用 git show 792736a）")
    a = ap.parse_args()

    import numpy as np
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, _SRC)

    from rl.env_wrapper import RLEnv as NewEnv
    OldEnv = _load_prefix_env()
    from rl.action_bundle import ActionBundle, K_MAX
    from rl.action_mask import validate_bundle
    from rl.follower import FollowerPolicy
    from rl.plan_space import PLAN_DIM
    from rl.belief import BeliefInference
    from rl.belief_planner import BeliefPlanner
    from rl.train_follower import FollowerOpponent
    from rl import train_solo as ts

    DECK = list(ts.DEFAULT_SOLO_DECK)

    # ---------- A. 强制 elixir=10 的掩码层对照 ----------
    print("=" * 96)
    print("A. 部分 bundle 掩码（**强制 elixir=10**，排除『买不起』的掩盖）")
    for name, C in (("旧(修复前 792736a)", OldEnv), ("新(修复后 工作树)", NewEnv)):
        env = C(opponent=None, seed=3, card_level=11, deck0=list(DECK), deck1=list(DECK))
        tot = bad_used = bad_used_slots = 0
        for sd in (100, 101, 102, 103):
            env.reset(seed=sd)
            env.battle.players[0].elixir = 10.0
            for n in (1, 2, 3, 4):
                for combo in itertools.permutations(range(1, K_MAX + 1), n):
                    b = ActionBundle()
                    for s in combo:
                        b.add(s, 8, 20)
                    m = env.get_action_mask(b)
                    tot += 1
                    leaked = [s for s in combo if bool(m["slots"][s - 1])]
                    if leaked:
                        bad_used += 1
                    bad_used_slots += len(leaked)
        print(f"   {name}: 组合 {tot} 组 ｜ 『已用槽位仍合法』{bad_used} 组 "
              f"({100.0*bad_used/tot:.1f}%) ｜ 泄漏槽位次数 {bad_used_slots}")

    # ---------- B. 真实 rollout 对照 ----------
    print("=" * 96)
    print("B. 真实 rollout（确定性采样，与评估路径同）")
    sd_ckpt = torch.load(os.path.join(_SRC, "runs/et_solo100k/solo_main_8000.pt"),
                         map_location="cpu")
    sd_ckpt = sd_ckpt.get("state_dict", sd_ckpt)
    for name, C in (("旧(修复前 792736a)", OldEnv), ("新(修复后 工作树)", NewEnv)):
        env = C(opponent=None, seed=781, card_level=11, deck0=list(DECK), deck1=list(DECK))
        bd = len(BeliefInference(opp_deck=env.deck1, n_particles=128, seed=0).encode(None, None))
        kw = dict(hidden=128, plan_dim=PLAN_DIM, belief_dim=bd,
                  value_bypass=True, value_independent=True)
        pol, opp = FollowerPolicy(**kw), FollowerPolicy(**kw)
        pol.load_state_dict(sd_ckpt)
        opp.load_state_dict(sd_ckpt)
        pol.eval().to_device("cpu")
        opp.eval().to_device("cpu")
        bp = BeliefPlanner()
        n_frames = n_mask_leak = n_invalid = n_guard = 0
        for g in range(a.games):
            opp_side = FollowerOpponent(
                opp, env, belief=BeliefInference(opp_deck=env.deck1, n_particles=128,
                                                 seed=a.seed_base + g),
                deterministic=True)
            belief = BeliefInference(opp_deck=env.deck1, n_particles=128,
                                     seed=a.seed_base + 1000 + g)
            obs, _ = env.reset(seed=a.seed_base + 2000 + g)
            belief.reset(env.deck1)
            env.opponent = opp_side
            hidden = None
            steps = 0
            done = False
            while not done and steps < a.max_steps:
                plan = bp.plan(env.battle, belief.state(), obs)
                tok = belief.encode(obs, None)
                try:
                    bundle, _lp, _v, hidden, masks = pol.act(
                        obs, tok, plan.to_vector(), env.get_action_mask,
                        hidden=hidden, deterministic=True)
                except RuntimeError as e:
                    if "mask 不变式" not in str(e):
                        raise
                    n_guard += 1
                    bundle = ActionBundle()
                    hidden = None
                n_frames += 1
                # 掩码层：每一步里，已出现在 partial 的槽位是否仍被放行
                for j in range(len(bundle.sub_actions)):
                    if j >= len(masks):
                        break
                    seen = {sa.slot - 1 for sa in bundle.sub_actions[:j] if sa.kind == "deploy"}
                    if any(bool(masks[j]["slots"][i]) for i in seen):
                        n_mask_leak += 1
                        break
                ok, reason, _res = validate_bundle(env.battle, 0, bundle)
                if not ok:
                    n_invalid += 1
                try:
                    obs, reward, term, trunc, info = env.step(bundle)
                except RuntimeError as e:
                    if "mask 不变式" not in str(e):
                        raise
                    n_guard += 1          # 守卫在**对手侧**路径命中（FollowerOpponent → act）
                    break
                try:
                    belief.update(obs, info.get("opp_played"))
                except Exception:
                    pass
                done = term or trunc
                steps += 1
        print(f"   {name}: 帧 {n_frames} ｜ 掩码泄漏帧 {n_mask_leak} ({100.0*n_mask_leak/max(1,n_frames):.2f}%) "
              f"｜ validate 拒绝帧 {n_invalid} ({100.0*n_invalid/max(1,n_frames):.2f}%) "
              f"｜ 不变式命中 {n_guard} ({100.0*n_guard/max(1,n_frames):.2f}%)")
    print("=" * 96)
    print("读法：旧实现应出现泄漏/拒绝（或不变式命中）；新实现三项应全为 0。")


if __name__ == "__main__":
    main()
