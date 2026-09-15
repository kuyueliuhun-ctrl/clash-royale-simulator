# -*- coding: utf-8 -*-
"""plan / belief 辅助通道的「梯度死活」判决（P1；只读）。

对应预注册：docs/channel_gradient_prereg_2026-09-14.md（判据在跑之前写死）。

§1 静态位移指纹（【R14】）：逐窗 ‖ΔW‖/‖W‖ + 「差逐位恰为 0」的窗口数。
§2 真梯度反传：真 ckpt + 真 rollout + **真实损失函数**（PPOTrainer._loss_pass），
   把 p_loss / v_loss 的梯度**分开**取，回答「∃ 非零梯度」这个结构问题。
§3 描述性读数（为 P2 提供依据）：fused 五块的尺度 / enc_fc 列块权重范数 /
   ∂p_loss/∂fused 的分块上游梯度。
§4 分块零消融：把 fused 的每一块单独置零，测策略第一决策步槽位分布的 KL
   ——「策略到底在读哪一块」的直接读数。

用法（必须在 src/clasher_new 下运行）：
    cd src/clasher_new && PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe \
        ../../scripts/probe_channel_gradient.py --frames 256 \
        --ckpt runs/d1_long_100k/solo_main_100000.pt
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src", "clasher_new")
for _p in (_SRC, os.path.join(_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np  # noqa: E402

# fused 分块（与 follower.py:295 的 cat 顺序逐位对应）
BLOCKS = [("grid_feat", 0, 2560), ("hand_feat", 2560, 2600),
          ("scalar", 2600, 2603), ("plan_f", 2603, 2667),
          ("belief_f", 2667, 2731)]
# 被测通道张量（前向路径上；见预注册 §1）
TARGETS = ["plan_mlp.0.weight", "plan_mlp.0.bias",
           "belief_mlp.0.weight", "belief_mlp.0.bias"]
# 强制对照
CTRL_ALIVE = ["enc_fc.weight", "enc_fc.bias"]
# 本 ckpt 里 **不在策略前向**（值通路专用的独立编码器 + 头）
CTRL_VAL_ONLY = ["value_enc_fc.weight", "value_enc_fc.bias",
                 "value_enc_ln.weight", "value_head_mlp.0.weight",
                 "value_head_mlp.0.bias", "value_head_mlp.2.weight"]


def _load_vds():
    """复用已有的位移指纹实现（单一来源，避免口径漂移）。"""
    path = os.path.join(_ROOT, "scripts", "value_displacement_scan.py")
    spec = importlib.util.spec_from_file_location("value_displacement_scan", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- §1
def part1_fingerprint(runs):
    vds = _load_vds()
    print("=" * 78)
    print("§1 静态位移指纹（R14）：差逐位恰为 0 ⟺ 该张量梯度精确为 0")
    print("   原理：Adam 对 loss 的常数缩放不变，任何非零梯度都会给出 ≈lr 的步长")
    for r in runs:
        if not os.path.isdir(r):
            print(f"  [缺失] {r}")
            continue
        cks = vds.load_ckpts(r)
        if len(cks) < 2:
            print(f"  [跳过] {r}: 少于 2 个 solo_main_<step>.pt")
            continue
        sds = [(st, vds.state_dict_of(p)) for st, p in cks]
        nwin = len(sds) - 1

        def measure(key):
            z, rels = 0, []
            for i in range(nwin):
                a, b = sds[i][1].get(key), sds[i + 1][1].get(key)
                if a is None or b is None or a.shape != b.shape:
                    return None
                diff = b - a
                if not np.any(diff):
                    z += 1
                rels.append(float(np.linalg.norm(diff))
                            / max(1e-12, float(np.linalg.norm(a))))
            return z, float(np.median(rels))

        print(f"\n  [{os.path.basename(r.rstrip('/'))}] 窗口数={nwin} "
              f"steps={[s for s, _ in sds]}")
        for key in TARGETS + CTRL_ALIVE + CTRL_VAL_ONLY:
            m = measure(key)
            if m is None:
                print(f"      {key:34s} [键缺失/形状不符，跳过]")
                continue
            z, med = m
            tag = ("被测" if key in TARGETS else
                   "对照·应活" if key in CTRL_ALIVE else "对照·值通路专用")
            flag = ""
            if key in TARGETS:
                flag = "  <<< 全窗冻结(A-M1)" if z == nwin else (
                    "  <多数冻结" if z * 2 > nwin else "")
            print(f"      {key:34s} 恰零={z:3d}/{nwin:<3d} ({z / max(1, nwin):.2f}) "
                  f"中位相对位移={med:.3e}  [{tag}]{flag}")


# ---------------------------------------------------------------- §2/§3/§4
def run(a):
    import torch
    from rl.config import TrainConfig
    from rl.env_wrapper import RLEnv
    from rl.belief import BeliefInference
    from rl.belief_planner import BeliefPlanner
    from rl.follower import FollowerPolicy, BELIEF_DIM, STOP_IDX
    from rl.plan_space import PLAN_DIM
    from rl.ppo import PPOTrainer
    from rl.action_bundle import ActionBundle

    cfg = TrainConfig.resolve("economy")
    print("\n" + "=" * 78)
    print("§2 真梯度反传（真 ckpt + 真 rollout + 真实损失 _loss_pass）")
    print(f"    协议：gamma={cfg.gamma} lambda={cfg.gae_lambda} "
          f"adv_norm=scale value_norm=running epochs=4 minibatch=32 shuffle=True")

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def fresh_policy():
        p = FollowerPolicy(hidden=128, plan_dim=PLAN_DIM, belief_dim=BELIEF_DIM,
                           value_bypass=True, value_independent=True)
        return p.to_device(dev)

    def make_ppo(pol):
        return PPOTrainer(pol, lr=cfg.lr, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda,
                          clip=cfg.clip, vf_coef=cfg.vf_coef, ent_coef=cfg.ent_coef,
                          max_grad_norm=cfg.max_grad_norm, adv_norm="scale",
                          value_norm="running", n_epochs=4, minibatch_size=32,
                          shuffle=True, seed=a.seed)

    env = RLEnv(opponent=None, seed=a.seed)
    obs, _ = env.reset(seed=a.seed)
    pol = fresh_policy()
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    meta = {k: sd[k] for k in ("value_bypass", "value_independent")
            if isinstance(sd, dict) and k in sd}
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    missing, unexpected = pol.load_state_dict(sd, strict=False)
    print(f"[ckpt] {a.ckpt}\n       元数据 {meta}；"
          f"missing={len(missing)} unexpected={len(unexpected)}")
    for p in pol.parameters():
        p.requires_grad_(True)
    pol.train()

    belief = BeliefInference(opp_deck=env.deck1, n_particles=128, seed=a.seed)
    bp = BeliefPlanner()
    col = {k: [] for k in ("obs", "belief", "plan", "bundle", "lp", "val",
                           "masks", "init")}
    ep_rew, ep_term, ep_trunc = [], [], []
    hidden = None
    plan_err, n_nz = [], 0
    ep_start, n = 0, 0

    # §4：分块零消融的累积器（用**当时的真掩码**，同帧同隐状态）
    # ⚠️ 掩码退化陷阱：若某帧只有 1 个合法槽位，分布恒为 one-hot ⇒ KL 结构性为 0，
    #    与被消融的块无关。故必须同时记录**掩码无关的** max|Δlogit|。
    abl_kl = {nm: [] for nm, _, _ in BLOCKS}
    abl_dlogit = {nm: [] for nm, _, _ in BLOCKS}
    abl_flip = {nm: [0, 0] for nm, _, _ in BLOCKS}   # [翻转数, 可比帧数]
    stop_full = []
    n_legal_hist = []
    elixir_hist = []

    def slot_measure(obs_, btok, plan_vec, h_in, mask, abl=None):
        """返回 (masked_p, raw_logits, n_legal)。"""
        handle = None
        if abl is not None:
            lo, hi = abl

            def pre_hook(_m, args):
                # ⚠️ 必须用 forward_pre_hook：forward_hook 在 forward **之后**才触发，
                # 在那里改输入等于什么都没改（本轮踩过，§4 曾全零）。
                x = args[0].clone()
                x[:, lo:hi] = 0.0
                return (x,) + tuple(args[1:])
            handle = pol.enc_fc.register_forward_pre_hook(pre_hook)
        try:
            with torch.no_grad():
                _f, enc = pol._encode_parts(obs_, btok, plan_vec)
                h0 = (torch.zeros(1, pol.hidden_dim, device=pol.device)
                      if h_in is None else h_in)
                h = pol.gru_cell(enc, h0)
                sb, _cb = pol._plan_biases(plan_vec)
                sm = pol._slot_mask_tensor(mask)
                raw = pol.slot_head(h) + sb
                lg = raw.masked_fill(sm == 0, -1e9)
                p = torch.softmax(lg, dim=-1)
            return (p.squeeze(0).cpu().numpy(),
                    raw.squeeze(0).cpu().numpy(), int((sm > 0).sum().item()))
        finally:
            if handle is not None:
                handle.remove()

    while n < a.frames:
        try:
            plan_vec = bp.plan(env.battle, belief.state(), obs).to_vector()
        except Exception as exc:
            if not plan_err:
                plan_err.append(repr(exc))
            plan_vec = np.zeros(PLAN_DIM, dtype=np.float32)
        if float(np.abs(plan_vec).sum()) > 1e-6:
            n_nz += 1
        btok = belief.encode(obs, None)
        init_hidden = hidden
        mask = env.get_action_mask(ActionBundle())
        # —— §4 消融（同帧同隐状态同掩码）——
        h_cur = None if hidden is None else hidden.clone()
        p_full, raw_full, n_legal = slot_measure(obs, btok, plan_vec, h_cur, mask)
        stop_full.append(float(p_full[STOP_IDX]))
        n_legal_hist.append(n_legal)
        elixir_hist.append(float(env.battle.players[0].elixir))
        eps = 1e-12
        for nm, lo, hi in BLOCKS:
            p_abl, raw_abl, _nl = slot_measure(obs, btok, plan_vec, h_cur, mask,
                                               (lo, hi))
            abl_kl[nm].append(float(np.sum(p_full * (np.log(p_full + eps)
                                                     - np.log(p_abl + eps)))))
            abl_dlogit[nm].append(float(np.max(np.abs(raw_full - raw_abl))))
            if n_legal > 1:
                abl_flip[nm][0] += int(np.argmax(p_full) != np.argmax(p_abl))
                abl_flip[nm][1] += 1
        # —— 真 rollout 推进 ——
        bundle, lp, val, hidden, masks = pol.act(
            obs, btok, plan_vec, env.get_action_mask,
            hidden=hidden, deterministic=False)
        for k, v in (("obs", obs), ("belief", btok), ("plan", plan_vec),
                     ("bundle", bundle), ("lp", lp), ("val", val),
                     ("masks", masks), ("init", init_hidden)):
            col[k].append(v)
        obs2, reward, term, trunc, info = env.step(bundle)
        ep_rew.append(reward); ep_term.append(term); ep_trunc.append(trunc)
        belief.update(obs2, info.get("opp_played"))
        obs = obs2
        n += 1
        if term or trunc:
            ep_start = len(ep_rew)
            obs, _ = env.reset(seed=a.seed + n)
            belief.reset(env.deck1)
            hidden = None
    if ep_start < len(ep_rew):
        pass
    ex = np.array(elixir_hist)
    print(f"[rollout] {n} 帧 | 规划器异常={plan_err or '无'} | "
          f"plan 非零帧={n_nz}/{n}")
    print(f"[rollout] P(STOP)均值={np.mean(stop_full):.3f} | "
          f"圣水中位={np.median(ex):.2f} p90={np.percentile(ex, 90):.2f} "
          f"| 圣水≥4 帧占比={100 * float((ex >= 4).mean()):.1f}% "
          f"| **可出手牌数≤0 帧占比={100 * float((np.array(n_legal_hist) <= 1).mean()):.1f}%**")

    # 按局分组算真 GAE（口径与 train_solo 相同：局内 GAE，局边界断开）
    # 末尾未结束的局也收进来（末步标 truncated → 用 last_value bootstrap），
    # 否则 rollout 短于一局时转移为空（本轮踩过：256 帧 = 0 局结束）。
    bounds = []
    s = 0
    for i in range(len(ep_rew)):
        if ep_term[i] or ep_trunc[i]:
            bounds.append((s, i + 1, False))
            s = i + 1
    if s < len(ep_rew):
        bounds.append((s, len(ep_rew), True))
    trans = []
    for (s, e, partial) in bounds:
        tr = list(ep_trunc[s:e])
        if partial:
            tr[-1] = True
        adv, ret = PPOTrainer.compute_gae(
            ep_rew[s:e], col["val"][s:e], ep_term[s:e], cfg.gamma, cfg.gae_lambda,
            truncated=tr, last_value=0.0)
        for j in range(s, e):
            trans.append({"obs": col["obs"][j], "belief": col["belief"][j],
                          "plan": col["plan"][j], "bundle": col["bundle"][j],
                          "old_logprob": col["lp"][j], "adv": float(adv[j - s]),
                          "returns": float(ret[j - s]), "masks": col["masks"][j],
                          "init_hidden": col["init"][j],
                          "n_legal": int(n_legal_hist[j])})
    print(f"[GAE] {len(bounds)} 局（含末局 partial={bounds[-1][2]}）"
          f" | 转移 {len(trans)} 条 | "
          f"adv std={np.std([t['adv'] for t in trans]):.4f}")

    ppo = make_ppo(pol)
    adv_raw = np.array([t["adv"] for t in trans], dtype=np.float32)
    advs = adv_raw.copy()
    _std = float(advs.std())
    if _std > 1e-6:
        advs = advs / (_std + 1e-8)
    rets_np = np.array([t["returns"] for t in trans], dtype=np.float32)
    ppo.ret_scaler.update(rets_np)
    _sc = ppo.ret_scaler.scale()
    v_scale = _sc if _sc > 0 else 1.0
    coef = ppo.ent_coef

    names = [nm for nm, _ in pol.named_parameters()]
    params = [p for _, p in pol.named_parameters()]
    grabbed = []
    hh = pol.enc_fc.register_forward_hook(lambda m, i, o: grabbed.append(i[0]))
    batches = ppo._plan_batches(len(trans))
    print(f"[minibatch] 计划批次 {len(batches)} 个（epochs=4）；取前 2 个小批各算一次")

    def grad_report(model, ppo_, trans_, idx, advs_, rets_, vs):
        pnames = [nm for nm, _ in model.named_parameters()]
        pparams = [p for _, p in model.named_parameters()]
        grabbed.clear()
        pack = ppo_._loss_pass(trans_, idx, advs_, rets_, ppo_.ent_coef, vs,
                               reduction="mean")
        fused_t = grabbed[-1] if grabbed else None
        gp = torch.autograd.grad(pack["p_loss"], pparams, retain_graph=True,
                                 allow_unused=True)
        gf = None
        if fused_t is not None and fused_t.requires_grad:
            gf = torch.autograd.grad(pack["p_loss"], [fused_t], retain_graph=True,
                                     allow_unused=True)[0]
        gv = torch.autograd.grad(pack["v_loss"], pparams, retain_graph=True,
                                 allow_unused=True)
        d = {"ratio_mean": pack["ratio_mean"],
             "p_loss": float(pack["p_loss"].item()),
             "v_loss": float(pack["v_loss"].item()),
             "ev": pack["ev"], "n_idx": len(idx),
             "adv_absmax": float(np.abs(advs_[idx]).max()),
             "adv_std_batch": float(np.std(advs_[idx])),
             "n_deploy": sum(1 for i in idx
                             if trans_[i]["bundle"].sub_actions),
             "gp": {nm: (0.0 if g is None else float(g.detach().norm().item()))
                    for nm, g in zip(pnames, gp)},
             "gv": {nm: (0.0 if g is None else float(g.detach().norm().item()))
                    for nm, g in zip(pnames, gv)},
             "gp_fused": None if gf is None else gf.detach().cpu().numpy(),
             "nl_row": np.array([trans_[i]["n_legal"] for i in idx])}
        return d

    results = [grad_report(pol, ppo, trans, batches[bi], advs, rets_np, v_scale)
               for bi in range(min(2, len(batches)))]
    for bi, r in enumerate(results):
        print(f"\n  [样本{bi}] 小批={r['n_idx']} 条 | ratio_mean={r['ratio_mean']:.8f}"
              f" | p_loss={r['p_loss']:+.6f} v_loss={r['v_loss']:.6f}"
              f" | EV={r['ev']:+.4f}")
        print(f"          诊断：批内 |adv|max={r['adv_absmax']:.4e} "
              f"std={r['adv_std_batch']:.4e} | 含部署的帧={r['n_deploy']}/{r['n_idx']}"
              f" | p 梯度非零张量={sum(1 for v in r['gp'].values() if v > 0)}"
              f"/{len(r['gp'])}")
    hh.remove()

    print("\n  逐张量 ‖g‖（p_loss=策略损失，v_loss=价值损失；4 列 = 2 样本 × 2 损失）：")
    print(f"      {'张量':30s} {'p(样本0)':>13s} {'v(样本0)':>13s} "
          f"{'p(样本1)':>13s} {'v(样本1)':>13s}")
    for nm in names:
        cells = []
        for r in results:
            cells += [r["gp"][nm], r["gv"][nm]]
        mark = "  <被测通道" if nm in TARGETS else ("  <值通路(不在策略前向)"
                                                if nm in CTRL_VAL_ONLY else "")
        print(f"      {nm:30s} {cells[0]:13.4e} {cells[1]:13.4e} "
              f"{cells[2]:13.4e} {cells[3]:13.4e}{mark}")

    # ---------------- 自检 ----------------
    print("\n  —— 强制自检（见预注册 §3；I3b 已按实测重设计，见判读自披露）——")
    info0 = [r for r in results if r["adv_absmax"] > 0]
    i1 = all(abs(r["ratio_mean"] - 1.0) < 1e-3 for r in results)
    i2 = bool(info0) and all(all(r["gp"][k] > 0 for k in CTRL_ALIVE) for r in info0)
    i3a = all(all(r["gp"][k] == 0.0 for k in CTRL_VAL_ONLY) for r in results)
    # I3'：新鲜初始化策略上，同一套反传必须给出 **非零** 值通路梯度
    #（若本 ckpt 的值通路梯度为零，那是真发现，不是仪器坏 —— 需要一个"已知活"的对照）
    fresh = fresh_policy()
    ppo_f = make_ppo(fresh)
    pf = grad_report(fresh, ppo_f, trans, batches[0], advs, rets_np, v_scale)
    i3p = all(pf["gv"][k] > 0 for k in CTRL_VAL_ONLY) and \
        all(pf["gv"][k] > 0 for k in TARGETS) and \
        pf["gv"]["value_head_mlp.2.weight"] > 0
    print(f"      I1 重放一致性 |ratio_mean−1|<1e-3        : "
          f"{'PASS' if i1 else 'FAIL'} "
          f"({[round(r['ratio_mean'], 8) for r in results]})")
    print(f"      I2 enc_fc 在 p_loss 下 ‖g‖>0（反传没坏） : {'PASS' if i2 else 'FAIL'}")
    print(f"      I3a 值通路专用张量在 p_loss 下 ‖g‖==0     : "
          f"{'PASS' if i3a else 'FAIL'}")
    print(f"      I3' 新鲜初始化策略上值通路 v_loss 梯度>0   : "
          f"{'PASS' if i3p else 'FAIL'}  "
          f"(value_enc_fc={pf['gv']['value_enc_fc.weight']:.4e}, "
          f"head_mlp.0.w={pf['gv']['value_head_mlp.0.weight']:.4e})")
    ok = i1 and i2 and i3a and i3p

    # 真发现：本 ckpt 的值通路梯度分布
    dead_val = [k for k in names if k.startswith("value") and
                results[0]["gv"][k] == 0.0 and results[1]["gv"][k] == 0.0]
    live_val = [k for k in names if k.startswith("value") and
                (results[0]["gv"][k] > 0 or results[1]["gv"][k] > 0)]
    print(f"\n  [附带发现·本 ckpt 值通路] v_loss 下**恒零**梯度的张量 "
          f"{len(dead_val)}/{len(dead_val) + len(live_val)} 个；"
          f"唯一活的是 {live_val}")

    # ---------------- 判据 B ----------------
    print("\n  —— 判据 B（结构判据，无阈值；信息量不足的样本单独标注）——")
    b_verdict = {}
    info = [r for r in results if r["adv_absmax"] > 0]
    print(f"      有信息量的小批（批内 |adv|max>0）：{len(info)}/{len(results)}")
    for key in TARGETS:
        if not info:
            b_verdict[key] = "B-UNDET"
        else:
            dead = all(r["gp"][key] == 0.0 and r["gv"][key] == 0.0 for r in info)
            b_verdict[key] = "B-M1 冻结" if dead else "B-M2 活着"
        cells = " ".join(f"({r['gp'][key]:.2e},{r['gv'][key]:.2e})" for r in results)
        print(f"      {key:30s} (‖g_p‖,‖g_v‖)={cells} ⇒ {b_verdict[key]}")
    print("\n  —— 量级（描述性，不参与判决；本实验**无**标定阈值）——")
    for i, r in enumerate(results):
        b = r["gp"]["enc_fc.weight"]
        parts = [f"enc_fc={b:.3e}"]
        for key in TARGETS:
            tag = f"{key.replace('.0.', '.')}"
            parts.append(f"{tag}={r['gp'][key] / b:.2e}" if b > 0 else f"{tag}=nan")
        print(f"      样本{i} ‖g_p‖/‖g_p(enc_fc)‖：{'  '.join(parts)}")

    # ---------------- §3 描述性：分块尺度 / 列块 / 上游梯度 ----------------
    print("\n" + "=" * 78)
    print("§3 描述性读数（为 P2 提供依据）：fused 分块尺度 / enc_fc 列块 / 上游梯度")
    fused_all = None
    W = pol.enc_fc.weight.detach().cpu().numpy()
    tot_w = float(np.linalg.norm(W))
    # fused 的实际数值（单独抓一次前向）
    with torch.no_grad():
        f_t, _e = pol._encode_batch_parts(
            [t["obs"] for t in trans[:32]], [t["belief"] for t in trans[:32]],
            [t["plan"] for t in trans[:32]])
    F = f_t.detach().cpu().numpy()
    print(f"      （上游梯度：对同一小批 {results[0]['n_idx']} 帧梯度取逐元素平均后"
          f"按块求范数；fused 统计用前 32 帧）")
    print(f"      {'分块':11s} {'维度':>5s} {'‖·‖中位':>11s} {'跨帧std中位':>12s} "
          f"{'‖W列块‖':>10s} {'占比':>7s} {'‖∂p/∂blk‖':>11s} {'占比':>7s}")
    gmean = np.mean([r["gp_fused"] for r in results], axis=0)   # (N,2731)
    rows_g = np.concatenate([r["gp_fused"] for r in results], axis=0)   # (ΣN,2731)
    rows_nl = np.concatenate([r["nl_row"] for r in results])
    has_choice = rows_nl > 1
    tmp = []
    for nm, lo, hi in BLOCKS:
        seg = F[:, lo:hi]
        nrm = float(np.median(np.linalg.norm(seg, axis=1)))
        sd = float(np.median(seg.std(axis=0)))
        wb = float(np.linalg.norm(W[:, lo:hi]))
        # 逐行范数；只统计**有选择余地**的行（其余行结构性为零，见判据 C）
        row_n = np.linalg.norm(rows_g[:, lo:hi], axis=1)
        sel = row_n[has_choice]
        ub = float(np.median(sel)) if len(sel) else 0.0
        tmp.append((nm, hi - lo, nrm, sd, wb, ub))
    ub_sum = sum(t[5] for t in tmp) or 1.0
    print(f"      （上游梯度用 {len(rows_g)} 行 = 2 样本 × 32 帧；其中"
          f"有选择余地的 {int(has_choice.sum())} 行参与中位统计）")
    for nm, dim, nrm, sd, wb, ub in tmp:
        print(f"      {nm:11s} {dim:5d} {nrm:11.4e} {sd:12.4e} {wb:10.4e} "
              f"{wb / tot_w:7.2%} {ub:11.4e} {ub / ub_sum:7.2%}")
    print("      读法：'‖W列块‖占比' = 该块在共享投影里的权重质量；"
          "'‖∂p/∂blk‖' = **只在有选择余地的帧上**逐行梯度范数的中位，"
          "占比分母是五块中位之和（对全部帧取平均会被结构性零行稀释）。")
    # —— 判据 C：掩码导致的「逐行零梯度」——
    print("\n  —— 判据 C（机制验证）：策略对 fused 的梯度**逐行**是否为零 ——")
    for bi, r in enumerate(results):
        g = r["gp_fused"]
        rn = np.linalg.norm(g, axis=1)
        nl = r["nl_row"]
        z = rn == 0.0
        z1 = z[nl <= 1]
        z2 = z[nl > 1]
        print(f"      样本{bi}: 零梯度行={int(z.sum())}/{len(rn)} "
              f"({100 * z.mean():.1f}%) | 其中 合法槽位≤1 的行={int(z1.sum())}"
              f"/{int((nl <= 1).sum())}、合法槽位>1 的行={int(z2.sum())}"
              f"/{int((nl > 1).sum())} | max‖g_row‖={rn.max():.3e}")
        if (nl > 1).any():
            print(f"               合法槽位>1 的行的 ‖g_row‖ 中位="
                  f"{np.median(rn[nl > 1]):.3e}；合法槽位≤1 的中位="
                  f"{np.median(rn[nl <= 1]):.3e}")
    print("      机制：掩码只剩 STOP 时 softmax 恒为 1 ⇒ slot logprob≡0 ⇒ "
          "该帧对任何参数**梯度精确为 0**（与网络权重无关）。")
    print(f"      （注意 plan_f/belief_f 的下游只有 enc_fc 的对应列块，"
          f"其自身参数梯度 ‖g_p‖ 见上表）")

    # ---------------- §4 分块零消融 ----------------
    print("\n" + "=" * 78)
    print("§4 分块零消融（同帧、同隐状态、同掩码；把 fused 该块置零）")
    nl = np.array(n_legal_hist)
    n_play = nl - 1          # 真正可出的手牌张数（去掉恒合法的 STOP）
    print(f"      【关键诊断】可出手牌张数分布 "
          f"{ {int(k): int(v) for k, v in zip(*np.unique(n_play, return_counts=True))} }"
          f" ⇒ **{100 * float((n_play <= 0).mean()):.1f}% 的决策帧一张牌都出不了**"
          f"（此时 STOP 是唯一合法动作，策略没有选择余地）")
    mask2 = nl > 1
    n_cmp = int(mask2.sum())
    print(f"      可比帧（至少 1 张牌可出）= {n_cmp}/{n} = {100 * n_cmp / n:.1f}%")
    print(f"      {'分块':11s} {'max|Δlogit|中位(全)':>19s} "
          f"{'中位(可比帧)':>14s} {'均值(可比帧)':>14s} {'KL中位(可比)':>13s} "
          f"{'翻转(可比)':>11s}")
    for nm, _, _ in BLOCKS:
        d = np.array(abl_dlogit[nm])
        k = np.array(abl_kl[nm])
        dc = d[mask2] if len(d) == n else d
        kc = k[mask2] if len(k) == n else k
        fl, tot = abl_flip[nm]
        print(f"      {nm:11s} {np.median(d):19.3e} "
              f"{(np.median(dc) if len(dc) else float('nan')):14.3e} "
              f"{(dc.mean() if len(dc) else float('nan')):14.3e} "
              f"{(np.median(kc) if len(kc) else float('nan')):13.3e} "
              f"{100 * fl / max(1, tot):10.1f}%")
    print("      读法：'max|Δlogit|'是**掩码无关**的敏感性读数（首选）；"
          "因 93% 帧无牌可出，'可比帧'列才是策略真正在做选择时的读数。")
    return ok, results, b_verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/d1_long_100k/solo_main_100000.pt")
    ap.add_argument("--frames", type=int, default=256)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--fingerprint-runs", nargs="*", default=[
        "runs/d1_long_100k", "runs/d1_league_20k", "runs/d1_league_20k_r2",
        "runs/d1_league_20k_r3"])
    ap.add_argument("--skip-fingerprint", action="store_true")
    ap.add_argument("--skip-gradient", action="store_true")
    a = ap.parse_args()
    if not a.skip_fingerprint:
        part1_fingerprint(a.fingerprint_runs)
    if not a.skip_gradient:
        run(a)


if __name__ == "__main__":
    main()
