# -*- coding: utf-8 -*-
"""**卡牌使用率分布**口径：源数据（人类标签）vs 训练后策略（用户 2026-09-20 提议）。

为什么另开一件仪器（预注册 `docs/fl_il_scaling_prereg_2026-09-20.md` §4bis）：
`il_eval_holdout.py` 量的是 **argmax 命中率**（top-1 / 逐槽 recall），它被『恒选某槽』的 trivial 上界压住
（实测 trivial = 39.84%，预注册闸门 40% 因此**无分辨力**）。本仪器改量**分布形状**：

* 逐帧配对（**同一批留出帧**）：人类打的牌 = `obs["hand"][slot_label-1]`；策略打的牌 = 策略确定性 argmax 的那个槽的手牌。
  策略选 `ABILITY`/`STOP`（或无可出）记成 `能力`/`未出牌` **两个独立桶**，不藏。
* 主读数 **TVD = 0.5·Σ|P_human − P_policy|**（含非卡桶）。
* 偏好口径 **`cond_X(c) = P(X 打 c | c 在该帧手牌中)`** ⇒ 把「这张牌常在手」与「这张牌被偏好」分开。
* 参考点（脚本复算）：`hand_uniform`（每帧手牌 4 张均分）、`slot4`（恒选第 4 槽）、`random-init`（`--random-init`）。

用法（Windows venv，从仓库根跑）:
    ./.venv/Scripts/python.exe scripts/il_card_usage.py \
        --ckpt runs/_fl_il_bc/sweep/bc_fl_e10_lr0.001.pt \
        --data-dir runs/_fl_il_bc/holdout --out docs/fl_il_2026-09-20/sweep/usage_m1.json
"""
import argparse
import collections
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import il_eval_holdout as ie  # noqa: E402  （复用 _abs / load_dir / make_get_mask；导入时 chdir 到 SRC）

import torch  # noqa: E402
from rl.follower import load_checkpoint  # noqa: E402
from rl.observation import ENTITY_NAMES  # noqa: E402

STOP_BUCKET = "未出牌"
ABILITY_BUCKET = "能力"


def card_of(obs, slot):
    """1-based 槽位 → 卡名（用 ENTITY_NAMES 下标；越界/异常返回 None）。"""
    idx = int(obs["hand"][slot - 1])
    if 0 <= idx < len(ENTITY_NAMES):
        return ENTITY_NAMES[idx]
    return None


def spearman(xs, ys):
    """Spearman ρ = 秩的 Pearson 相关（不依赖 scipy）。并列取平均秩。"""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return (num / (dx * dy)) if dx > 0 and dy > 0 else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--random-init", action="store_true")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--min-in-hand", type=int, default=30, help="Spearman 只取在手帧数 ≥ 此值的牌")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if not args.ckpt and not args.random_init:
        ap.error("需要 --ckpt 或 --random-init")
    for name in ("ckpt", "data_dir", "out"):
        v = getattr(args, name, None)
        if v:
            setattr(args, name, ie._abs(v))

    holdout = ie.load_dir(args.data_dir, args.limit)
    if not holdout:
        print(f"[usage] {args.data_dir} 下没有 bc_*.pkl")
        return 2
    if args.random_init:
        torch.manual_seed(0)
        from rl.follower import FollowerPolicy
        policy = FollowerPolicy(hidden=128, plan_dim=len(holdout[0][2]), belief_dim=len(holdout[0][1]))
        policy.eval()
        tag = "random-init|torch.manual_seed(0)"
    else:
        policy = load_checkpoint(args.ckpt)
        tag = os.path.basename(args.ckpt)

    hum = collections.Counter()
    pol = collections.Counter()
    hand_cnt = collections.Counter()        # 每张牌「在手里」的帧数
    hum_given = collections.Counter()       # 在手且人类打了它
    pol_given = collections.Counter()       # 在手且策略打了它
    #: **「恰好一张」变体（混淆控制）**：一张牌在手牌里出现两次时，「随便挑一个槽」的策略命中它的概率天然翻倍，
    #: 而人类也可能因此更常打它 ⇒ 这会**人为抬高** cond 的相关性。只在「该牌恰好占 1 个槽」的帧上统计即可去掉该混淆。
    h1_cnt = collections.Counter()
    h1_hum = collections.Counter()
    h1_pol = collections.Counter()
    h1_s4 = collections.Counter()
    hu_uniform = collections.Counter()      # hand_uniform 参考
    sl4 = collections.Counter()             # 恒选第 4 槽参考
    n = 0
    #: **位置受控的「真选择」读数**：人类 39.8% 的标签就落在第 4 槽（= 我们重建手牌的第 3 号位），
    #: 而「恒选第 4 槽」在这一子集上必中、在其余 60.2% 上必错 ⇒
    #: 只有在**人类打了别的槽**的子集上，命中率才是「学会了选牌」的读数（机会线 = 1/4 = 0.25）。
    g = collections.Counter()               # all/out/in 的 n 与 hit
    for obs, tok, plan, bundle, masks in holdout:
        n += 1
        hand = [card_of(obs, s) for s in range(1, 5)]
        occ = collections.Counter(c for c in hand if c)
        for c in set(c for c in hand if c):
            hand_cnt[c] += 1
            if occ[c] == 1:
                h1_cnt[c] += 1
        for c in hand:
            if c:
                hu_uniform[c] += 1.0 / 4
        lab = bundle.sub_actions[0]
        lab_slot0 = int(lab.slot) - 1
        if getattr(lab, "kind", "deploy") == "ability":
            hum[ABILITY_BUCKET] += 1
        else:
            hc = card_of(obs, int(lab.slot))
            hum[hc or "?"] += 1
            if hc:
                hum_given[hc] += 1
                if occ[hc] == 1:
                    h1_hum[hc] += 1
        if hand[3]:                      # 恒选第 4 槽
            sl4[hand[3]] += 1
            if occ[hand[3]] == 1:
                h1_s4[hand[3]] += 1

        pred, _lp, _v, _h, _mk = policy.act(obs, tok, plan, ie.make_get_mask(masks),
                                            hidden=None, deterministic=True)
        if not pred.sub_actions:
            pol[STOP_BUCKET] += 1
            g["all_n"] += 1
            g["out_n" if lab_slot0 != 3 else "in_n"] += 1
        else:
            sa = pred.sub_actions[0]
            hit = (getattr(sa, "kind", "deploy") != "ability") and (int(sa.slot) - 1 == lab_slot0)
            g["all_n"] += 1
            g["all_hit"] += int(hit)
            if lab_slot0 != 3:
                g["out_n"] += 1
                g["out_hit"] += int(hit)
            else:
                g["in_n"] += 1
                g["in_hit"] += int(hit)
            if getattr(sa, "kind", "deploy") == "ability":
                pol[ABILITY_BUCKET] += 1
            else:
                pc = card_of(obs, int(sa.slot))
                pol[pc or "?"] += 1
                if pc:
                    pol_given[pc] += 1
                    if occ[pc] == 1:
                        h1_pol[pc] += 1

    def tvd(a, b):
        keys = set(a) | set(b)
        return 0.5 * sum(abs(a.get(k, 0) / n - b.get(k, 0) / n) for k in keys)

    t_hum_slot4 = tvd(hum, sl4)
    t_hum_unif = tvd(hum, hu_uniform)
    t_hum_pol = tvd(hum, pol)

    # ---- 偏好口径（去手牌组成）：这是**判别式**读数；边际 TVD 会被手牌组成主导 ----
    # 实测（3 epoch 对照）：TVD(human, policy)=0.1281 **大于** TVD(human, 恒选槽4)=0.0617
    # ⇒ 预注册 J-U1 的「边际 TVD」判据与 40% 闸门同病（trivial 参考更强）；故另立 `mad_cond`/`U`：
    #   cond_X(c) = P(X 打 c | c 在该帧手牌中)；「不看偏好」基线 = 0.25（4 槽等权）
    #   mad(h, X) = mean_c |cond_h(c) − cond_X(c)|；U(X) = 1 − mad(h,X)/mad(h, 0.25) = **人类偏好被复现的比例**
    cards = [c for c in hand_cnt if c not in ("?",) and hand_cnt[c] >= args.min_in_hand]
    cards.sort(key=lambda c: -hand_cnt[c])
    cond_h = {c: hum_given[c] / hand_cnt[c] for c in cards}
    cond_p = {c: pol_given[c] / hand_cnt[c] for c in cards}
    cond_4 = {c: sl4.get(c, 0) / hand_cnt[c] for c in cards}
    zero = {c: 0.25 for c in cards}

    def mad(a, b):
        return sum(abs(a[c] - b[c]) for c in cards) / max(1, len(cards))

    mad_hp, mad_h0, mad_h4 = mad(cond_h, cond_p), mad(cond_h, zero), mad(cond_h, cond_4)
    u_pol = (1 - mad_hp / mad_h0) if mad_h0 > 0 else None
    u_slot4 = (1 - mad_h4 / mad_h0) if mad_h0 > 0 else None
    rho = spearman([cond_h[c] for c in cards], [cond_p[c] for c in cards])
    rho_slot4 = spearman([cond_h[c] for c in cards], [cond_4[c] for c in cards])

    #: ★ **残差口径（去「槽位占用」）**：`resid_X(c) = cond_X(c) − cond_4(c)`。
    #: 本数据里人类的「卡牌偏好」有 ~90% 可由「这张牌恰好在第 4 槽」解释
    #: （实测 ρ(cond_human, cond_slot4) = 0.9006）⇒ 只有**残差**才是「除占用之外的偏好」。
    resid_h = {c: cond_h[c] - cond_4[c] for c in cards}
    resid_p = {c: cond_p[c] - cond_4[c] for c in cards}
    rho_resid = spearman([resid_h[c] for c in cards], [resid_p[c] for c in cards])
    mad_resid = sum(abs(resid_h[c] - resid_p[c]) for c in cards) / max(1, len(cards))
    mad_resid_zero = sum(abs(resid_h[c]) for c in cards) / max(1, len(cards))

    def boot_ci(rng_seed=0, draws=2000):
        """**卡牌级**自助 95% CI（重抽 100 张牌）—— 判「ρ_resid 是否只是噪声」用（【R16】：不用单点观测标定）。"""
        rnd = random.Random(rng_seed)
        vals = []
        for _ in range(draws):
            idx = [rnd.randrange(len(cards)) for _ in cards]
            xs = [resid_h[cards[i]] for i in idx]
            ys = [resid_p[cards[i]] for i in idx]
            r = spearman(xs, ys)
            if r is not None:
                vals.append(r)
        if not vals:
            return None, None
        vals.sort()
        return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]

    ci_lo, ci_hi = boot_ci()

    #: **「恰好一张」变体**：同一套残差口径，但分母只算「该牌恰好占 1 个槽」的帧（去掉重复牌混淆）。
    def _metrics(sub_cnt, sub_hum, sub_pol, sub_s4, boot=True):
        cs = [c for c in sub_cnt if sub_cnt[c] >= args.min_in_hand]
        cs.sort(key=lambda c: -sub_cnt[c])
        if len(cs) < 3:
            return None
        cH = {c: sub_hum[c] / sub_cnt[c] for c in cs}
        cP = {c: sub_pol[c] / sub_cnt[c] for c in cs}
        c4 = {c: sub_s4.get(c, 0) / sub_cnt[c] for c in cs}
        rH = {c: cH[c] - c4[c] for c in cs}
        rP = {c: cP[c] - c4[c] for c in cs}
        rho_c = spearman([cH[c] for c in cs], [cP[c] for c in cs])
        rho_r = spearman([rH[c] for c in cs], [rP[c] for c in cs])
        lo = hi = None
        if boot:
            rnd = random.Random(0)
            vals = []
            for _ in range(2000):
                idx = [rnd.randrange(len(cs)) for _ in cs]
                r = spearman([rH[cs[i]] for i in idx], [rP[cs[i]] for i in idx])
                if r is not None:
                    vals.append(r)
            if vals:
                vals.sort()
                lo, hi = vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]
        return {"cards": len(cs), "spearman_cond": rho_c, "spearman_resid": rho_r,
                "spearman_resid_ci95": [lo, hi],
                "mad_cond_human_policy": sum(abs(cH[c] - cP[c]) for c in cs) / len(cs),
                "mad_cond_human_flat": sum(abs(cH[c] - 0.25) for c in cs) / len(cs)}

    once = _metrics(h1_cnt, h1_hum, h1_pol, h1_s4)

    per_card = [{
        "card": c,
        "human_n": hum.get(c, 0), "human_rate": hum.get(c, 0) / n,
        "policy_n": pol.get(c, 0), "policy_rate": pol.get(c, 0) / n,
        "in_hand_n": hand_cnt[c],
        "cond_human": cond_h.get(c, hum_given[c] / hand_cnt[c]),
        "cond_policy": cond_p.get(c, pol_given[c] / hand_cnt[c]),
        "cond_slot4": cond_4.get(c, sl4.get(c, 0) / hand_cnt[c]),
        "hand_uniform_rate": hu_uniform.get(c, 0) / n, "slot4_rate": sl4.get(c, 0) / n,
    } for c in sorted(hand_cnt, key=lambda c: -hand_cnt[c]) if c != "?"]

    res = {
        "ckpt": tag, "holdout_samples": n,
        "tvd_human_policy": t_hum_pol,
        "tvd_human_slot4": t_hum_slot4,
        "tvd_human_hand_uniform": t_hum_unif,
        "mad_cond_human_policy": mad_hp,
        "mad_cond_human_flat": mad_h0,
        "mad_cond_human_slot4": mad_h4,
        "pref_capture_policy": u_pol,
        "pref_capture_slot4": u_slot4,
        "policy_stop_rate": pol.get(STOP_BUCKET, 0) / n,
        "policy_ability_rate": pol.get(ABILITY_BUCKET, 0) / n,
        "policy_unknown_rate": pol.get("?", 0) / n,
        "spearman_cond": rho, "spearman_cond_slot4": rho_slot4,
        "spearman_resid": rho_resid, "spearman_resid_ci95": [ci_lo, ci_hi],
        "mad_resid_human_policy": mad_resid, "mad_resid_human_zero": mad_resid_zero,
        "pref_capture_resid": (1 - mad_resid / mad_resid_zero) if mad_resid_zero > 0 else None,
        "exactly_once": once,
        "choice_gate": {
            "all_n": g["all_n"], "all_acc": (g["all_hit"] / g["all_n"]) if g["all_n"] else None,
            "out_of_pos4_n": g["out_n"],
            "out_of_pos4_acc": (g["out_hit"] / g["out_n"]) if g["out_n"] else None,
            "in_pos4_n": g["in_n"],
            "in_pos4_acc": (g["in_hit"] / g["in_n"]) if g["in_n"] else None,
            "chance": 0.25, "pos4_prior_share": g["in_n"] / max(1, g["all_n"]),
        },
        "spearman_cards": len(cards), "min_in_hand": args.min_in_hand,
        "distinct_cards_human": len([c for c in hum if c not in ("?", STOP_BUCKET, ABILITY_BUCKET)]),
        "distinct_cards_policy": len([c for c in pol if c not in ("?", STOP_BUCKET, ABILITY_BUCKET)]),
        "per_card": per_card,
    }
    print(f"[usage] {tag}  n={n}")
    print(f"  ★ 偏好复现率 U = {u_pol if u_pol is None else round(u_pol, 4)}  "
          f"（mad(h,policy)={mad_hp:.4f} / mad(h,不看偏好)={mad_h0:.4f}；trivial 恒选槽4 的 U={u_slot4 if u_slot4 is None else round(u_slot4, 4)}）")
    print(f"  Spearman(cond_human, cond_policy) = {rho if rho is None else round(rho, 4)}"
          f"   [恒选槽4 参考 ρ = {rho_slot4 if rho_slot4 is None else round(rho_slot4, 4)}]"
          f"  （{len(cards)} 张牌，在手帧 ≥ {args.min_in_hand}）")
    print(f"  ★★ 残差口径（去占用）ρ_resid = {rho_resid if rho_resid is None else round(rho_resid, 4)}  "
          f"95%CI=[{None if ci_lo is None else round(ci_lo, 3)}, {None if ci_hi is None else round(ci_hi, 3)}]  "
          f"U_resid = {round(1 - mad_resid / mad_resid_zero, 4) if mad_resid_zero > 0 else None}")
    if once:
        print(f"  ★★ 恰好一张变体（去重复牌混淆）ρ_resid = {round(once['spearman_resid'], 4)} "
              f"95%CI=[{round(once['spearman_resid_ci95'][0], 3)}, {round(once['spearman_resid_ci95'][1], 3)}]  "
              f"ρ_cond = {round(once['spearman_cond'], 4)}  U = "
              f"{round(1 - once['mad_cond_human_policy'] / once['mad_cond_human_flat'], 4)}  "
              f"（{once['cards']} 张牌）")
    cg = res["choice_gate"]
    print(f"  ★★★ 位置受控的选牌命中率：全体 {cg['all_acc']:.4f}（{cg['all_n']} 帧）  "
          f"**人类打非第4槽子集 {cg['out_of_pos4_acc']:.4f}**（{cg['out_of_pos4_n']} 帧；机会线 0.25，"
          f"恒选槽4 在此子集 = 0.0000）  人类打第4槽子集 {cg['in_pos4_acc']:.4f}（{cg['in_pos4_n']} 帧）")
    print(f"  [边际口径，预注册 J-U1] TVD(human,policy)={t_hum_pol:.4f}  "
          f"TVD(human,恒选槽4)={t_hum_slot4:.4f}  TVD(human,手牌均分)={t_hum_unif:.4f}")
    print(f"  策略非卡桶：未出牌 {res['policy_stop_rate']:.3f}  能力 {res['policy_ability_rate']:.3f}  "
          f"未知 {res['policy_unknown_rate']:.3f}")
    print(f"  {'卡':<16}{'人类n':>7}{'策略n':>7}{'Δ率':>9}{'在手n':>7}{'cond人':>9}{'cond策':>9}{'cond槽4':>9}")
    for r in per_card[:args.top]:
        print(f"  {r['card']:<16}{r['human_n']:>7}{r['policy_n']:>7}"
              f"{r['policy_rate'] - r['human_rate']:>+9.4f}{r['in_hand_n']:>7}"
              f"{r['cond_human']:>9.4f}{r['cond_policy']:>9.4f}{r['cond_slot4']:>9.4f}")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        print(f"[usage] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
