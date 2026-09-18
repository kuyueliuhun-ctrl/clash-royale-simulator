# -*- coding: utf-8 -*-
"""外置工具接线核查 + 模型敏感度实测（只读）：

回答「我们已经有工具让模型算未来塔伤，模型真的用了吗？」

§1 **静态接线**（脚本扫描全仓，不手抄）：三个外置工具的入口函数在**生产代码**里被谁调用
   （排除定义处与 selftest）——决定"模型有没有机会用"。
§2 **动态敏感度**（真 ckpt + 真 rollout）：对**确实接进了决策路径**的通道做消融——
   同一帧、同一 GRU 隐状态，比较四种条件下的**贪心动作是否翻转**与**同一动作的 logprob 变化**：
     A 完整 | B 关掉 plan 软偏置（隔离"网络自己学的部分"）
     C plan 置零 + 偏置关（隔离"只靠 obs/belief"） | D belief 置零（对照已知 +0.4pp）
用法（必须在 src/clasher_new 下运行）：
    cd src/clasher_new && PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe \
        ../../scripts/probe_tool_usage.py --frames 200 --ckpt runs/d1_long_100k/solo_main_100000.pt
"""
from __future__ import annotations

import argparse
import os
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src", "clasher_new"))
from rl.io_bootstrap import force_utf8_stdout  # noqa: E402  (T1-1: UTF-8 兜底单一来源)
force_utf8_stdout()

import numpy as np

# 工具入口 → 所属文件（"生产调用点"= 排除该文件自身与 selftest）
TOOLS = [
    ("① 未来塔伤", "estimate_tower_threat", "threat_calc.py"),
    ("② 交换模拟器", "simulate_exchange", "simulate_exchange.py"),
    ("③ 法术知识", "get_spell_profile", "spell_module.py"),
    ("③ 法术知识", "best_cast", "spell_module.py"),
]


def scan_callers():
    root = os.path.join(_ROOT, "src", "clasher_new")
    hits = {name: [] for _, name, _ in TOOLS}
    for dirpath, _dirs, files in os.walk(root):
        if "__pycache__" in dirpath:
            continue
        for fn in files:
            if not fn.endswith(".py"):
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, _ROOT).replace("\\", "/")
            try:
                txt = open(full, encoding="utf-8").read()
            except OSError:
                continue
            for _tag, name, owner in TOOLS:
                if fn == owner:
                    continue
                for m in re.finditer(rf"\b{name}\b", txt):
                    line = txt[:m.start()].count("\n") + 1
                    hits[name].append((rel, line, "selftest" in fn))
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--ckpt", default="runs/d1_long_100k/solo_main_100000.pt")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    print("=" * 78)
    print("§1 静态接线：三个外置工具在**生产代码**里被谁调用（脚本扫描，排除定义处/selftest）")
    hits = scan_callers()
    for tag, name, owner in TOOLS:
        prod = [(f, l) for f, l, is_st in hits[name] if not is_st]
        st = [(f, l) for f, l, is_st in hits[name] if is_st]
        state = "✅ 有生产调用" if prod else "❌ **无任何生产调用**（只在 selftest 里）"
        print(f"  {tag}  {name:<22} {state}")
        for f, l in prod[:5]:
            print(f"        → {f}:{l}")
        if st:
            print(f"        （selftest 命中 {len(st)} 处，已排除）")

    # ---------------- 动态敏感度 ----------------
    print("\n" + "=" * 78)
    print("§2 动态敏感度：真 ckpt 上做 plan/belief 消融（同帧同隐状态）")
    import torch
    from rl.env_wrapper import RLEnv
    from rl.belief import BeliefInference
    from rl.belief_planner import BeliefPlanner
    from rl.follower import FollowerPolicy, BELIEF_DIM
    from rl.plan_space import PLAN_DIM
    from rl.action_bundle import ActionBundle

    env = RLEnv(opponent=None, seed=a.seed)
    obs, _ = env.reset(seed=a.seed)
    pol = FollowerPolicy(hidden=128, plan_dim=PLAN_DIM, belief_dim=BELIEF_DIM)
    sd = torch.load(a.ckpt, map_location="cpu")
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    missing, unexpected = pol.load_state_dict(sd, strict=False)
    print(f"[ckpt] {a.ckpt}: missing={len(missing)} unexpected={len(unexpected)}")
    pol.eval()
    belief = BeliefInference(opp_deck=env.deck1, n_particles=128, seed=a.seed)
    bp = BeliefPlanner()

    import torch.nn.functional as F
    from rl.action_bundle import ActionBundle
    from rl.follower import STOP_IDX

    def slot_dist_vec(plan_vec, btok, biases=True, h_in=None):
        """第一决策步的槽位分布（logits→softmax，掩码已上）——分布层面的可比量。"""
        pol.plan_biases_enabled = biases
        with torch.no_grad():
            fused, enc = pol._encode_parts(obs, btok, plan_vec)
            h0 = (torch.zeros(1, pol.hidden_dim, device=pol.device) if h_in is None
                  else h_in)
            h = pol.gru_cell(enc, h0)
            sb, _cb = pol._plan_biases(plan_vec)
            mask = env.get_action_mask(ActionBundle())
            sm = pol._slot_mask_tensor(mask)
            lg = pol.slot_head(h) + sb
            lg = lg.masked_fill(sm == 0, -1e9)
            p = torch.softmax(lg, dim=-1)
        pol.plan_biases_enabled = True
        return p.squeeze(0).cpu().numpy()

    zero_plan = np.zeros(PLAN_DIM, dtype=np.float32)
    zero_belief = np.zeros(BELIEF_DIM, dtype=np.float32)
    n = n_nz_plan = 0
    hidden = None
    plan_err = []
    KL = {"B": [], "C": [], "D": []}
    gap_BC, gap_D = [], []
    flips = {"B": 0, "C": 0, "D": 0}
    stop = {"A": [], "B": [], "C": [], "D": []}
    for _ in range(a.frames):
        try:
            plan_vec = bp.plan(env.battle, belief.state(), obs).to_vector()
        except Exception as exc:
            if not plan_err:
                plan_err.append(repr(exc))
            plan_vec = zero_plan
        if float(np.abs(plan_vec).sum()) > 1e-6:
            n_nz_plan += 1
        btok = belief.encode(obs, None)
        h_cur = None if hidden is None else hidden.clone()
        pA = slot_dist_vec(plan_vec, btok, True, h_cur)
        pB = slot_dist_vec(plan_vec, btok, False, h_cur)
        pC = slot_dist_vec(zero_plan, btok, False, h_cur)
        pD = slot_dist_vec(plan_vec, zero_belief, True, h_cur)
        eps = 1e-12
        kb = float(np.sum(pA * (np.log(pA + eps) - np.log(pB + eps))))
        kc = float(np.sum(pA * (np.log(pA + eps) - np.log(pC + eps))))
        gap_BC.append(abs(kb - kc))          # = plan 学习分支的贡献上界
        gap_D.append(float(np.abs(pA - pD).max()))
        for tag, p in (("B", pB), ("C", pC), ("D", pD)):
            KL[tag].append(float(np.sum(pA * (np.log(pA + eps) - np.log(p + eps)))))
            flips[tag] += int(np.argmax(pA) != np.argmax(p))
        stop["A"].append(float(pA[STOP_IDX])); stop["B"].append(float(pB[STOP_IDX]))
        stop["C"].append(float(pC[STOP_IDX])); stop["D"].append(float(pD[STOP_IDX]))
        n += 1
        # 推进：用分布 A 的采样动作（非贪心，避免"93% 空 bundle"退化）
        bundle, _lp, _v, hidden_new, _m = pol.act(
            obs, btok, plan_vec, env.get_action_mask, hidden=h_cur, deterministic=False)
        hidden = hidden_new
        obs2, _r, term, trunc, info = env.step(bundle)
        belief.update(obs2, info.get("opp_played"))
        obs = obs2
        if term or trunc:
            obs, _ = env.reset(seed=a.seed + n)
            belief.reset(env.deck1)
            hidden = None

    print(f"[健康检查] 规划器异常(首条)={plan_err or '无'} | plan 非零帧={n_nz_plan}/{n}"
          f" | 递归：hidden 跨帧携带")
    print(f"\n[rollout] n={n} 帧 | 第一决策步**槽位分布**层面的比较（同帧同隐状态）")
    print(f"  P(STOP/不出牌) 均值: A={np.mean(stop['A']):.3f} B={np.mean(stop['B']):.3f} "
          f"C={np.mean(stop['C']):.3f} D={np.mean(stop['D']):.3f}")
    print(f"  **分离读数**：plan 学习分支的贡献上界 = 逐帧 |KL_B−KL_C| 最大 "
          f"{max(gap_BC):.3e}（中位 {np.median(gap_BC):.3e}）")
    print(f"  **分离读数**：belief 的影响上界 = 逐帧 max|p_A−p_D| 最大 {max(gap_D):.3e}"
          f"（中位 {np.median(gap_D):.3e}）")
    for tag, label in (("B", "plan 学习分支（偏置关） vs A"),
                       ("C", "plan 置零+偏置关       vs A"),
                       ("D", "belief 置零            vs A")):
        k = np.array(KL[tag])
        print(f"  {label}: KL(A‖·) 中位={np.median(k):.3e} 均值={k.mean():.3e} "
              f"| 槽位 argmax 翻转率={100*flips[tag]/max(1,n):5.1f}%")
    print("\n读法：翻转率≈0 且 |Δlogprob|≈0 ⇒ **该通道对当前策略几乎无影响**；"
          "翻转率高 ⇒ 网络确实在用它（或只是被软偏置推着走，看 B/C 对比）。")


if __name__ == "__main__":
    main()
