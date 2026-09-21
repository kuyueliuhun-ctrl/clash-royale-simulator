# -*- coding: utf-8 -*-
"""混比剂量-反应的**对照表**（脚本复算，禁手抄）—— 预注册 `docs/fl_il_il2_prereg_2026-09-22.md` §3 J8。

四个臂（`--mix-ratio R` = 每 epoch 抽 `round(R × n_play)` 条 save 帧）：

| 臂 | R | 含义 |
|---|---|---|
| `R00` | 0.0 | 等价 play-only 目标（对照组） |
| `R025` | 0.25 | 轻度混入 |
| `R10` | 1.0 | 均权 |
| `R337` | 3.37 | **= 本轮塌成的那个点**（全量 168,587，复用 `sweep/`） |

判据（跑前写死）：**J8.1** act-AUC > Stage 0 噪声带 ／ **J8.2** play 组 top1 ≥ 0.35 且 NLL ≤ 5.75
／ **J8.3** 部署局均出牌 ∈ **[18, 28.5]**（人类 22.8 的 ±20%）／ 人类分箱靶 8/9/10 费 = 0.107/0.186/0.094。

用法：
    .venv/Scripts/python.exe scripts/il_mix_report.py \
        --arms runs/_fl_il_bc_save/sweep_mixR00=runs/il_readout_mixR00 \
               runs/_fl_il_bc_save/sweep_mixR025=runs/il_readout_mixR025 \
               ... --docs docs/fl_il_2026-09-21 --out docs/fl_il_2026-09-21/mix_report.json
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG_CWD = os.getcwd()


def _abs(p):
    m = p.replace("\\", "/")
    if m.startswith("/mnt/") and len(m) > 6:
        head, rest = m.split("/", 3)[2:]
        return head.upper() + ":\\" + rest.replace("/", "\\")
    return os.path.normpath(os.path.join(ORIG_CWD, p))


def _load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def readout_stats(path):
    """读 runner 的**统计 json**（`--json` 的产物：`p0_plays / frames_per_game / winrate` …）。

    ⚠️ 第一版误读 run 目录里的 `solo_state.json`（那是 dashboard 的状态文件，没有 `p0_plays`）
    ⇒ `KeyError`（实测踩过）。若给的是目录，则在其中找 `*_stats.json`。
    """
    if os.path.isdir(path):
        cand = glob.glob(os.path.join(path, "*_stats.json")) or \
            glob.glob(os.path.join(path, "*.json"))
        if not cand:
            return None
        path = sorted(cand)[0]
    return _load(path) if os.path.isfile(path) else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True, help="holdout_mix*.json 所在目录")
    ap.add_argument("--arm", action="append", required=True,
                    help="`<ckpt_dir>=<readout_stats.json>`（可多次）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    docs = _abs(args.docs)
    rows = []
    for spec in args.arm:
        ck_dir, ro_dir = spec.split("=", 1)
        ck_dir, ro_dir = _abs(ck_dir), _abs(ro_dir)
        tag = os.path.basename(ck_dir).replace("sweep_mix", "").replace("sweep", "R337")
        man = _load(os.path.join(ck_dir, "sweep_manifest.json"))["runs"][0]
        ho = _load(os.path.join(docs, "holdout_mix%s.json" % tag))
        ro = readout_stats(ro_dir)
        xbow = 0
        if ro:
            xbow = sum(n for c, n in ro.get("p0_top_cards", []) if c in ("XBow", "Xbow"))
        rows.append({
            "tag": tag, "mix_ratio": man.get("mix_ratio"),
            "epoch_pool": man.get("epoch_pool"),
            "mean_logprob_ep3": (man["curve"][-1]["mean_logprob"] if man.get("curve") else None),
            "act_auc": ho["act_decision"]["auc_model_act"],
            "act_base_rate": ho["act_decision"]["base_rate_act"],
            "play_top1": ho["J3"]["top1_option"], "play_nll": ho["J3"]["nll_mean"],
            "play_cell": ho["J3"]["cell_exact"], "macro": ho["macro_recall_option"],
            "stop_gap_unweighted": ho["stop_frames"]["stop_calibration_abs_gap"],
            "stop_gap_weighted": (abs(ho["stop_frames"]["weighted"]["model_stop_rate_argmax"]
                                      - ho["stop_frames"]["weighted"]["human_stop_rate"])
                                  if ho["stop_frames"].get("weighted") else None),
            "plays_per_game": (ro["p0_plays"] / ro["games"]) if ro else None,
            "frames_per_game": ro.get("frames_per_game") if ro else None,
            "winrate": ro.get("winrate") if ro else None,
            "stop_when_playable_rate": ro.get("stop_when_playable_rate") if ro else None,
            "xbow_plays": xbow,
        })
    rows.sort(key=lambda r: (r["mix_ratio"] is None, r["mix_ratio"]))
    band = [18.0, 28.5]
    for r in rows:
        r["J8_2"] = ("PASS" if (r["play_top1"] is not None and r["play_top1"] >= 0.35
                                and r["play_nll"] is not None and r["play_nll"] <= 5.75) else "FAIL")
        r["J8_3"] = ("PASS" if (r["plays_per_game"] is not None
                                and band[0] <= r["plays_per_game"] <= band[1]) else
                     ("高于带上界（过度出牌）" if r["plays_per_game"] is not None
                      and r["plays_per_game"] > band[1] else "低于带下界（不出牌）"))
    res = {"band_plays_per_game": band, "human_plays_per_game": 22.8, "rows": rows}
    if args.out:
        with open(_abs(args.out), "w", encoding="utf-8", newline="\n") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    hdr = ["mix", "pool", "logp@3", "actAUC", "playTop1", "playNLL", "macro",
           "STOPΔ", "STOPΔw", "出牌/局", "帧/局", "胜率", "买得起不出", "Xbow", "J8.2", "J8.3"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for r in rows:
        print("| " + " | ".join([
            str(r["mix_ratio"]), str(r["epoch_pool"]), f"{r['mean_logprob_ep3']:.3f}",
            f"{r['act_auc']:.4f}" if r["act_auc"] is not None else "—",
            f"{r['play_top1']:.4f}" if r["play_top1"] is not None else "—",
            f"{r['play_nll']:.4f}" if r["play_nll"] is not None else "—",
            f"{r['macro']:.4f}" if r["macro"] is not None else "—",
            f"{r['stop_gap_unweighted']:.4f}" if r["stop_gap_unweighted"] is not None else "—",
            f"{r['stop_gap_weighted']:.4f}" if r["stop_gap_weighted"] is not None else "—",
            f"{r['plays_per_game']:.1f}" if r["plays_per_game"] is not None else "—",
            f"{r['frames_per_game']:.0f}" if r["frames_per_game"] is not None else "—",
            f"{r['winrate']:.2f}" if r["winrate"] is not None else "—",
            f"{r['stop_when_playable_rate']:.3f}" if r["stop_when_playable_rate"] is not None else "—",
            str(r["xbow_plays"]), r["J8_2"], r["J8_3"],
        ]) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
