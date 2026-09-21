# -*- coding: utf-8 -*-
"""**对照表**（脚本复算，禁手抄，【R4】）：W2 手牌打分 / W3 前期卡组信息分的剂量-反应。

预注册：`docs/il_whiff_handscore_prereg_2026-09-22.md` **§6**（跑前写死）。
三臂（**同语料、同形状、同初值**，唯一变量 = plan 尾 17 列**有没有信息**）：

| 臂 | `--plan-extras-zero` | 含义 |
|---|---|---|
| `HS17` | 17 | **对照组**（尾 17 列存在但抹零） |
| `HS04` | 4 | 只有 W2 活着（手牌打分 13 维） |
| `HS00` | 0 | **W2 + W3** 全活 |

判据（跑前冻结，逐条实现）：
  * **J-W2.3**：配对 `HS04 − HS17` ⇒ `mean_k ΔNLL ≤ 0` ∧ `mean_k Δact-AUC ≥ 0` ∧ ≥2/3 seed 方向一致；
    `mean_k ΔNLL > 0 ∧ mean_k Δact-AUC < 0` ⇒ **无收益 ⇒ 回滚特征**。
  * **J-W2.4**：`plays/game` 落在 [18, 28.5] —— **只登记不判定**（【R15】带值不可跨语料照抄）。
  * **J-W3.2**：配对 `HS00 − HS04` ⇒ `mean_k Δ(known_opp_median_in_window) ≥ 0`（方向判据）。
  * 逐 seed 值**全部列出**（不藏噪声）。

用法（仓库根）：
    ./.venv/Scripts/python.exe scripts/il_hs_report.py \
        --root runs/_fl_il_bc_hs --docs docs/fl_il_2026-09-21 \
        --out docs/fl_il_2026-09-21/hs_report.json
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

ARMS = [("HS17", 17), ("HS04", 4), ("HS00", 0)]
BAND_PLAYS = [18.0, 28.5]


def _abs(p):
    m = p.replace("\\", "/")
    if m.startswith("/mnt/") and len(m) > 6:
        head, rest = m.split("/", 3)[2:]
        return head.upper() + ":\\" + rest.replace("/", "\\")
    return os.path.normpath(os.path.join(ROOT, p))


def _load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _load_opt(p):
    return _load(p) if p and os.path.isfile(p) else None


def readout_stats(out_dir):
    """读 runner 的统计 json（目录 ⇒ 找 `*_stats.json`；与 `il_mix_report.py` 同口径）。"""
    if not out_dir or not os.path.isdir(out_dir):
        return None
    cand = glob.glob(os.path.join(out_dir, "*_stats.json")) or \
        glob.glob(os.path.join(out_dir, "*.json"))
    return _load(cand[0]) if cand else None


def collect(root, docs, seeds):
    rows = []
    for arm, z in ARMS:
        for k in seeds:
            ckdir = os.path.join(root, "sweep_mix%s" % arm, "s%d" % k)
            ck = os.path.join(ckdir, "bc_fl_e3_lr0.001.pt")
            man = _load_opt(os.path.join(ckdir, "sweep_manifest.json"))
            ho = _load_opt(os.path.join(docs, "holdout_hs%s_s%d.json" % (arm.lower(), k)))
            ro = readout_stats(os.path.join(ROOT, "runs", "il_readout_hs%s_s%d" % (arm.lower(), k)))
            ro = ro or _load_opt(os.path.join(docs, "readout_hs%s_s%d_stats.json"
                                              % (arm.lower(), k)))
            us = _load_opt(os.path.join(docs, "usage_hs%s_s%d.json" % (arm.lower(), k)))
            r0 = (man or {}).get("runs", [{}])[0] if man else {}
            row = {
                "arm": arm, "zero": z, "seed": k,
                "ckpt_exists": os.path.isfile(ck),
                "plan_dim": r0.get("plan_dim"), "plan_extras_zero": r0.get("plan_extras_zero"),
                "epoch_pool": r0.get("epoch_pool"),
                "mean_logprob_last": (r0["curve"][-1]["mean_logprob"] if r0.get("curve") else None),
                "total_sec": r0.get("total_sec"),
                "act_auc": (ho or {}).get("act_decision", {}).get("auc_model_act"),
                "play_top1": (ho or {}).get("J3", {}).get("top1_option"),
                "play_nll": (ho or {}).get("J3", {}).get("nll_mean"),
                "play_cell": (ho or {}).get("J3", {}).get("cell_exact"),
                "macro": (ho or {}).get("macro_recall_option"),
                "plays_per_game": ((ro or {}).get("p0_plays", 0) / (ro or {}).get("games", 1)
                                   if ro and ro.get("games") else None),
                "frames_per_game": (ro or {}).get("frames_per_game"),
                "winrate": (ro or {}).get("winrate"),
                "stop_when_playable_rate": (ro or {}).get("stop_when_playable_rate"),
                "playable_frame_rate": (ro or {}).get("playable_frame_rate"),
                "known_opp_median_in_window": ((ro or {}).get("info_window") or {})
                .get("known_opp_median_in_window"),
                "plays_in_window_per_game": ((ro or {}).get("info_window") or {})
                .get("plays_in_window_per_game"),
                "first_play_t_median": ((ro or {}).get("info_window") or {})
                .get("first_play_t_median"),
                "usage_tvd": (us or {}).get("tvd"),
                "usage_top": (us or {}).get("top") if us else None,
            }
            rows.append(row)
    return rows


def check_ablation_columns(root, seeds):
    """**设计有效性自检**（脚本复算）：三臂的 `plan_mlp.0.weight` 只有该动的列在动。

    这是本实验「同形状 + 同初值 + 唯一变量 = 尾列有无信息」的**直接证据**：
      * `HS17 vs HS04`：W2 段（58..70）**必须不同**（HS04 用了它）、W3 段（71..74）**必须逐位相同**
        （两臂都把 W3 抹零 ⇒ 零梯度）；
      * `HS04 vs HS00`：W3 段**必须不同**（HS00 才用）。
    「静默失败」的样子：若 `plan_extras_zero` 没接上，三臂逐位相同 ⇒ 下面全 FAIL。
    """
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        return {"skipped": f"无 torch：{type(e).__name__}"}
    out = {"pairs": [], "ok": True}
    for k in seeds:
        w = {}
        for arm, _z in ARMS:
            p = os.path.join(root, "sweep_mix%s" % arm, "s%d" % k,
                             "bc_fl_e3_lr0.001.pt")
            if not os.path.isfile(p):
                w = None
                break
            w[arm] = torch.load(p, map_location="cpu")["state_dict"]["plan_mlp.0.weight"]
        if not w:
            out["pairs"].append({"seed": k, "skipped": "ckpt 缺失"})
            continue
        def md(a, b, lo, hi):
            return float((w[a][:, lo:hi] - w[b][:, lo:hi]).abs().max())
        rec = {"seed": k,
               "HS17_vs_HS04_W2_58_70": md("HS17", "HS04", 58, 71),
               "HS17_vs_HS04_W3_71_74": md("HS17", "HS04", 71, 75),
               "HS04_vs_HS00_W3_71_74": md("HS04", "HS00", 71, 75),
               "HS04_vs_HS00_W2_58_70": md("HS04", "HS00", 58, 71)}
        rec["PASS"] = bool(rec["HS17_vs_HS04_W2_58_70"] > 0
                           and rec["HS17_vs_HS04_W3_71_74"] == 0.0
                           and rec["HS04_vs_HS00_W3_71_74"] > 0)
        out["ok"] = out["ok"] and rec["PASS"]
        out["pairs"].append(rec)
    return out


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def _delta(rows, arm_a, arm_b, field):
    """逐 seed 配对差 `arm_a − arm_b`（同 seed）⇒ 返回 (逐 seed 列表, 均值)。"""
    by = {(r["arm"], r["seed"]): r for r in rows}
    ds = []
    for k in sorted({r["seed"] for r in rows}):
        ra, rb = by.get((arm_a, k)), by.get((arm_b, k))
        if not ra or not rb:
            continue
        va, vb = ra.get(field), rb.get(field)
        ds.append((k, (None if (va is None or vb is None) else va - vb)))
    return ds, _mean([d for _k, d in ds])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/_fl_il_bc_hs")
    ap.add_argument("--docs", default="docs/fl_il_2026-09-21")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    root, docs = _abs(args.root), _abs(args.docs)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    rows = collect(root, docs, seeds)

    hdr = ["臂", "zero", "seed", "pool", "logp@3", "actAUC", "top1", "NLL", "cell",
           "macro", "出牌/局", "帧/局", "胜率", "买得起不出", "窗口已知卡", "TVD"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for r in rows:
        f = lambda v, n=4: ("—" if v is None else (f"{v:.{n}f}" if isinstance(v, float) else str(v)))
        print("| " + " | ".join([
            r["arm"], str(r["zero"]), str(r["seed"]), str(r["epoch_pool"]),
            f(r["mean_logprob_last"], 3), f(r["act_auc"]), f(r["play_top1"]),
            f(r["play_nll"]), f(r["play_cell"]), f(r["macro"]),
            f(r["plays_per_game"], 2), f(r["frames_per_game"], 1), f(r["winrate"], 2),
            f(r["stop_when_playable_rate"]), f(r["known_opp_median_in_window"], 2),
            f(r["usage_tvd"]),
        ]) + " |")

    # ——— 配对判据 ———
    out = {"prereg": "docs/il_whiff_handscore_prereg_2026-09-22.md §6",
           "band_plays_per_game": BAND_PLAYS, "rows": rows, "judgements": {},
           "ablation_columns": check_ablation_columns(root, seeds)}

    d_nll, m_nll = _delta(rows, "HS04", "HS17", "play_nll")
    d_auc, m_auc = _delta(rows, "HS04", "HS17", "act_auc")
    d_top1, m_top1 = _delta(rows, "HS04", "HS17", "play_top1")
    n_ok = sum(1 for (_k, a), (_k2, b) in zip(d_nll, d_auc)
               if a is not None and b is not None and a <= 0 and b >= 0)
    n_pair = sum(1 for (a, b) in zip(d_nll, d_auc) if a[1] is not None and b[1] is not None)
    j233 = bool(m_nll is not None and m_auc is not None and n_pair
                and m_nll <= 0 and m_auc >= 0 and n_ok * 3 >= 2 * n_pair)
    rollback = bool(m_nll is not None and m_auc is not None and m_nll > 0 and m_auc < 0)
    out["judgements"]["J-W2.3"] = {
        "pair": "HS04 - HS17 (W2 手牌打分)", "delta_nll_per_seed": d_nll,
        "delta_nll_mean": m_nll, "delta_act_auc_per_seed": d_auc, "delta_act_auc_mean": m_auc,
        "delta_top1_per_seed": d_top1, "delta_top1_mean": m_top1,
        "seeds_ok": n_ok, "seeds_paired": n_pair,
        "verdict": "PASS" if j233 else ("FAIL(无收益⇒回滚特征)" if rollback else "FAIL"),
    }

    d_kn, m_kn = _delta(rows, "HS00", "HS04", "known_opp_median_in_window")
    d_pw, m_pw = _delta(rows, "HS00", "HS04", "plays_in_window_per_game")
    d_nll3, m_nll3 = _delta(rows, "HS00", "HS04", "play_nll")
    d_auc3, m_auc3 = _delta(rows, "HS00", "HS04", "act_auc")
    out["judgements"]["J-W3.2"] = {
        "pair": "HS00 - HS04 (W3 信息分)", "delta_known_opp_per_seed": d_kn,
        "delta_known_opp_mean": m_kn,
        "delta_plays_in_window_per_seed": d_pw, "delta_plays_in_window_mean": m_pw,
        "delta_nll_mean": m_nll3, "delta_act_auc_mean": m_auc3,
        "verdict": ("PASS" if (m_kn is not None and m_kn >= 0) else "FAIL"),
        "note": "方向判据（R15/R16）：只看符号，不设绝对阈值",
    }

    band = []
    for r in rows:
        p = r["plays_per_game"]
        band.append({"arm": r["arm"], "seed": r["seed"], "plays_per_game": p,
                     "in_band": (None if p is None else BAND_PLAYS[0] <= p <= BAND_PLAYS[1])})
    out["judgements"]["J-W2.4"] = {
        "band": BAND_PLAYS, "per_row": band,
        "n_in_band": sum(1 for b in band if b["in_band"]),
        "n_row": sum(1 for b in band if b["in_band"] is not None),
        "note": "只登记不判定（R15：带值来自另一批语料，不可照抄）",
    }

    if args.out:
        os.makedirs(os.path.dirname(_abs(args.out)), exist_ok=True)
        with open(_abs(args.out), "w", encoding="utf-8", newline="\n") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print(f"\n[hs_report] → {args.out}")
    for k, v in out["judgements"].items():
        print(f"[hs_report] {k}: {v.get('verdict', '（描述性，无判决）')}")
    ab = out["ablation_columns"]
    print(f"[hs_report] 消融列自检: "
          f"{'PASS' if ab.get('ok') else ab.get('skipped', 'FAIL——三臂可能逐位相同！')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
