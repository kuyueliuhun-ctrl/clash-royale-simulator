# -*- coding: utf-8 -*-
"""觉醒 S1 判读：`--evo-slots` 关（off）vs 开（on），同一 200 局。

判据**跑前写死**在 `docs/il_evo_prereg_2026-09-22.md` §2 S1：
  ① **覆盖率**：回放里出现的 `-ev*` 后缀能映射到本仓觉醒表的比例 **≥ 0.95**；
  ② **不变量**：同一 200 局重跑，`labels` 与旧读数之差 **≤ 5%**（= 掩码/跨进程漂移量级）、`errors = 0`；
  ③ **生效性**：觉醒触发次数 **> 0**，并给出「声明觉醒位的局数 / 总局数」。

本脚本另加三项**本报告自己标定**的检查（非预注册判据，标 `extra`）：
  E1 **JSON schema 不变**：off 产物里**不得**出现任何 `evo_*` 字段 / `S1_evo` 块
     （【R2】「关掉 = 旧路径不变」在产物层面的可执行检查）；
  E2 **双口径对账**：`finish`（觉醒出牌次数）vs `wrap`（觉醒实体个数）—— 量纲不同，
     报告 `wrap/finish` 并逐卡列出，**不调和**（【R17】）；
  E3 **跨进程对账**（非判据）：与 2200 局旧语料 `samples_stop_save_hs.json` 里**同一批 tag**
     的聚合比 —— 用于标定「跨进程漂移量级」，**不是** PASS/FAIL。

用法（仓库根；`/usr/bin/python3` 即可，**不需要 torch**）：
    PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 scripts/il_evo_probe_report.py \
        --off docs/fl_il_2026-09-21/evo200_off.json \
        --on  docs/fl_il_2026-09-21/evo200_on.json \
        --archive docs/fl_il_2026-09-21/samples_stop_save_hs.json \
        --out docs/fl_il_2026-09-21/evo200_verdict.json
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src", "clasher_new")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

#: ⚠️ UTF-8 兜底必须走**单一实现**（`scripts/_structure_check.py` 检查 ⑧：手写
#: `sys.stdout.reconfigure` 块 = FAIL）。本脚本**不需要 torch**（`rl/__init__.py` 是纯 stdlib）
#: ⇒ `/usr/bin/python3` 直接跑即可（实测 import 通过）。
from rl.io_bootstrap import force_utf8_stdout  # noqa: E402

#: 参与判据②的计数字段（分子/分母同口径：都是**整 200 局求和**）
COUNTED = ("frames", "team_single", "team_multi", "labels", "mask_reject", "bundle_not_ok",
           "errors", "team_off", "stop_save_cand", "stop_labels", "opp_plays", "hand_forced")


def _sum(games, keys=COUNTED):
    out = {k: 0 for k in keys}
    for g in games:
        for k in keys:
            out[k] += int(g.get(k) or 0)
    return out


def _rel(a, b):
    """相对差（分母取 |b|，b=0 时返回 None）。"""
    if not b:
        return None
    return (a - b) / abs(b)


def _evo_fields(obj):
    """找**逐局数据**里所有 `evo*` 键名（E1 schema 检查）。

    ⚠️ 只扫 `per_game`：`source` 块里**本来就有** `evo_slots: false/true` 这个**配置回显**
    （第一版把它当成了「off 侧出现了 evo 字段」⇒ 假 FAIL，本报告自己踩的坑，【R10】）。
    """
    hits = set()

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k == "S1_evo" or str(k).startswith("evo_"):
                    hits.add(k)
                walk(v)
        elif isinstance(x, list):
            for v in x[:400]:
                walk(v)
    walk(obj.get("per_game") or [])
    return sorted(hits)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--off", required=True)
    ap.add_argument("--on", required=True)
    ap.add_argument("--archive", default=None, help="2200 局旧语料读数（E3，非判据）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    off = json.load(open(args.off, encoding="utf-8"))
    on = json.load(open(args.on, encoding="utf-8"))
    goff, gon = off["per_game"], on["per_game"]
    rep = {"prereg": "docs/il_evo_prereg_2026-09-22.md §2 S1",
           "inputs": {"off": os.path.basename(args.off), "on": os.path.basename(args.on)},
           "games": {"off": len(goff), "on": len(gon)}}

    # ———————————————— ① 覆盖率 ————————————————
    s1 = on.get("S1_evo") or {}
    cov = s1.get("coverage")
    #: ★ 口径区分（本批发现的**预注册缺口**）：预注册 §2 S1 判据①字面说的是「后缀**能映射**」
    #: （§5 实测 100%），而本探针报的是更严的「**能声明**」= 能映射 ∧ 引擎两处数据齐备
    #: （周期表 + `evo_raw`）——后者对 `elite-barbarians-ev1` 不成立 ⇒ 两个定义差 8%。
    _det = {}
    for g in gon:
        for side in ("team", "opponent"):
            for u in ((g.get("evo_unmapped") or {}).get(side) or []):
                d = _det.setdefault(u["key"], {"n": 0, "resolved": u["resolved"],
                                               "in_cycles": u["in_cycles"],
                                               "has_evo_raw": u["has_evo_raw"]})
                d["n"] += 1
    rep["gate1_coverage"] = {
        "coverage_definition": ("**能声明率** = 可声明槽 / (可声明槽 + 未映射槽)，"
                                "可声明 = 后缀可映射 ∧ `name ∈ EVOLUTION_CYCLES` ∧ `Card(name).evo_raw` 为真。"
                                "⚠️ 预注册 §2 S1 判据①的字面口径是「**能映射**」（§5 实测 100%）"
                                "——两者**不是同一个量**，差集恰好是下面 `unmapped_detail` 里那张卡。"),
        "ev_slots_total": s1.get("ev_slots_total"),
        "ev_slots_declared": s1.get("ev_slots_declared"),
        "ev_slots_unmapped": s1.get("ev_slots_unmapped"),
        "ev_slots_truncated_at_2": s1.get("ev_slots_truncated_at_2"),
        "coverage": cov, "threshold": ">= 0.95",
        "unmapped_keys": s1.get("unmapped_keys"),
        "unmapped_detail": _det,
        "verdict": "PASS" if (cov is not None and cov >= 0.95) else "FAIL",
    }

    # ———————————————— ② 不变量 ————————————————
    so, sn = _sum(goff), _sum(gon)
    deltas = {}
    for k in COUNTED:
        r = _rel(sn[k], so[k])
        deltas[k] = {"off": so[k], "on": sn[k], "abs": sn[k] - so[k],
                     "rel": r, "rel_pct": (None if r is None else round(100 * r, 4))}
    lab_rel = deltas["labels"]["rel"]
    rep["gate2_invariance"] = {
        "counted": deltas,
        "labels_rel_diff": lab_rel,
        "threshold": "|rel(labels)| <= 0.05 且 errors == 0",
        "errors": {"off": so["errors"], "on": sn["errors"]},
        "verdict": ("PASS" if (lab_rel is not None and abs(lab_rel) <= 0.05
                               and sn["errors"] == 0 and so["errors"] == 0) else "FAIL"),
    }

    # ———————————————— ③ 生效性 ————————————————
    fin = s1.get("triggers_finish") or {}
    wrp = s1.get("triggers_wrap") or {}
    fin_tot = (fin.get("team", 0) or 0) + (fin.get("opp", 0) or 0)
    wrp_tot = (wrp.get("team", 0) or 0) + (wrp.get("opp", 0) or 0)
    rep["gate3_liveness"] = {
        "games_with_declaration": s1.get("games_with_declaration"),
        "games_total": s1.get("games_converted"),
        "declaration_rate": s1.get("declaration_rate"),
        "triggers_finish": fin, "triggers_finish_total": fin_tot,
        "triggers_wrap": wrp, "triggers_wrap_total": wrp_tot,
        "triggers_by_card": s1.get("triggers_by_card"),
        "triggers_wrap_by_card": s1.get("triggers_wrap_by_card"),
        "threshold": "triggers_finish_total > 0",
        "verdict": "PASS" if fin_tot > 0 else "FAIL",
        "note": ("旧路径（off）**必然** 0（`rl/` 从不声明觉醒位）；off 产物里没有该字段 ⇒ "
                 "只有 on 侧有数。`finish` = 觉醒出牌次数，`wrap` = 觉醒实体个数（量纲不同）。"),
    }

    # ———————————————— E1 schema 不变（R2） ————————————————
    f_off, f_on = _evo_fields(off), _evo_fields(on)
    rep["extra_schema"] = {
        "evo_keys_in_off": f_off, "evo_keys_in_on": f_on,
        "S1_evo_block": {"off": "S1_evo" in off, "on": "S1_evo" in on},
        "expected": "off：无 `S1_evo` 块且逐局无 `evo_*` 字段；on：两者都有",
        "verdict": ("PASS" if (not f_off and f_on and ("S1_evo" in on) and ("S1_evo" not in off))
                    else "FAIL"),
    }

    # ———————————————— E2 双口径 ————————————————
    ratio = (wrp_tot / fin_tot) if fin_tot else None
    rep["extra_two_gauges"] = {
        "finish_total": fin_tot, "wrap_total": wrp_tot, "wrap_per_finish": ratio,
        "read": ("单兵卡应 ≈1、n 兵卡应 ≈n、法术觉醒只进 finish ⇒ 期望 wrap/finish 落在 [0, 1]"
                 "（多兵卡存在时也可 >1，**不设判据**）"),
    }

    # ———————————————— E3 跨进程对账（非判据） ————————————————
    if args.archive and os.path.exists(args.archive):
        arch = json.load(open(args.archive, encoding="utf-8"))
        tags = {g.get("tag") for g in goff}
        ag = [g for g in arch["per_game"] if g.get("tag") in tags]
        sa = _sum(ag)
        rep["extra_archive_crossproc"] = {
            "archive_file": os.path.basename(args.archive),
            "archive_games_matched": len(ag), "archive_games_total": len(arch["per_game"]),
            "archive_labels": sa["labels"], "off_labels": so["labels"],
            "rel_diff_labels": _rel(so["labels"], sa["labels"]),
            "archive_mask_reject": sa["mask_reject"], "off_mask_reject": so["mask_reject"],
            "note": "**非判据**：跨进程漂移（C17）⇒ 只用于标定本批 5% 阈值的噪声地板。",
        }
    else:
        rep["extra_archive_crossproc"] = {"skipped": True, "reason": "归档读数不存在"}

    verdicts = [rep["gate1_coverage"]["verdict"], rep["gate2_invariance"]["verdict"],
                rep["gate3_liveness"]["verdict"]]
    rep["S1_verdict"] = ("PASS" if all(v == "PASS" for v in verdicts) else "FAIL")
    rep["S1_verdicts"] = dict(zip(("①覆盖率", "②不变量", "③生效性"), verdicts))

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(rep, f, ensure_ascii=False, indent=1)

    print("=" * 70)
    print("觉醒 S1 判读（预注册 §2 S1）")
    print("=" * 70)
    g1, g2, g3 = rep["gate1_coverage"], rep["gate2_invariance"], rep["gate3_liveness"]
    print(f"① 覆盖率   = {g1['coverage']}  ({g1['ev_slots_declared']}/{g1['ev_slots_total']} 槽, "
          f"截断 {g1['ev_slots_truncated_at_2']})  → {g1['verdict']}")
    print(f"② 不变量   = labels off={g2['counted']['labels']['off']} on={g2['counted']['labels']['on']} "
          f"rel={g2['labels_rel_diff']} | errors off={g2['errors']['off']} on={g2['errors']['on']} "
          f"→ {g2['verdict']}")
    print(f"   mask_reject off={g2['counted']['mask_reject']['off']} "
          f"on={g2['counted']['mask_reject']['on']} rel={g2['counted']['mask_reject']['rel']}")
    print(f"③ 生效性   = 触发(finish)={g3['triggers_finish_total']} wrap={g3['triggers_wrap_total']} | "
          f"声明局数={g3['games_with_declaration']}/{g3['games_total']} "
          f"({g3['declaration_rate']}) → {g3['verdict']}")
    print(f"  逐卡触发 = {g3['triggers_by_card']}")
    print(f"E1 schema  = off {rep['extra_schema']['evo_keys_in_off']} / on 非空 "
          f"→ {rep['extra_schema']['verdict']}")
    print(f"E2 双口径  = wrap/finish = {ratio}")
    e3 = rep["extra_archive_crossproc"]
    if not e3.get("skipped"):
        print(f"E3 跨进程  = 归档 labels={e3['archive_labels']} vs off={e3['off_labels']} "
              f"rel={e3['rel_diff_labels']} （非判据，噪声地板）")
    print("-" * 70)
    print(f"S1 总判 = {rep['S1_verdict']}   {rep['S1_verdicts']}")
    return 0 if rep["S1_verdict"] == "PASS" else 2


if __name__ == "__main__":
    force_utf8_stdout()
    sys.exit(main())
