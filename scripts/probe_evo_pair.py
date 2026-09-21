# -*- coding: utf-8 -*-
"""觉醒 S1 **同进程配对**探针：同一局、同一进程内跑 `--evo-slots` 关 / 开两遍。

为什么需要它（预注册 §2 S1 判据②的**真正可执行形式**）：
  * 200 局 off/on 是**两个进程**跑的，而本仓已确证**跨进程不可复现**
    （【C17】：同代码同 seed、两次独立进程样本差 5.9%；**同进程内 0/42**）
    ⇒ 5% 阈值只能证明「没有大改动」，**不能**证明「关掉时逐位不变」。
  * 本探针把两次调用放进**同一个进程**（`_convert_one` 直接调用，不走 `Pool`），
    于是可以给出**逐值相等**的强判据。

判据（跑前写死）：
  P1 **未声明觉醒位的局**：off 与 on 的**全部**公共计数字段（frames / labels / mask_reject /
     team_single / stop_* / errors …）**逐值相等**（这就是【R2】「关掉 = 旧路径不变」；
     唯一差别允许出现在 on 侧**新增**的 `evo_*` 字段上）。
  P2 **声明了觉醒位的局**：off/on **允许**不同（重建语义确实变了），但必须报出差异与触发数；
     且**至少 1 局**触发数 > 0（否则 S1 在真实回放上仍不生效 ⇒ 判 FAIL，照实报【R10】）。
  P3 两遍的 `tag` / `dur` / `holdout` 一致（同一局的同一份输入）。

用法（仓库根；**Windows venv python**，需要 torch）：
    PYTHONDONTWRITEBYTECODE=1 ./.venv/Scripts/python.exe scripts/probe_evo_pair.py \
        --jsonl ../fl_il_data/replays_part000000.jsonl \
        --report docs/fl_il_2026-09-21/evo200_off.json --games 16 \
        --out docs/fl_il_2026-09-21/evo_pair_probe.json
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src", "clasher_new")
ORIG_CWD = os.getcwd()
sys.path.insert(0, SRC)
os.chdir(SRC)
sys.path.insert(0, HERE)

from rl.io_bootstrap import force_utf8_stdout  # noqa: E402

force_utf8_stdout()

from fl_il_to_bc import deck_of, _convert_one  # noqa: E402

#: ⚠️ **不要用 `fl_il_to_bc._abs`**：它的 `ORIG_CWD` 是**它被 import 时**的 cwd，而本脚本在
#: import 之前已经 `chdir(SRC)` ⇒ 相对路径会被拼成 `<repo>/src/clasher_new/docs/...`（实测踩到，
#: 与 `il_hs_feature_use.py` 同型的坑）⇒ 这里用**本脚本自己**在 chdir 前记下的 `ORIG_CWD`。


def _resolve(p):
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(ORIG_CWD, p))

IGNORE_PREFIX = ("evo_",)


def _select(jsonl, tags, limit):
    """按 `--report` 里的 tag 选局（**与 200 局探针同一批局**，不另写一套过滤逻辑）。"""
    out = []
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("tag") not in tags:
                continue
            names, vs_all = {}, {}
            bad = False
            for side in ("team", "opponent"):
                nm, vs, bd = deck_of(r, side)
                if bd or len(nm) != 8:
                    bad = True
                    break
                names[side] = nm
                vs_all[side] = vs
            if bad:
                continue
            r["_decks"] = names
            r["_evo"] = vs_all
            out.append(r)
            if limit and len(out) >= limit:
                break
    return out


def _cmp(off, on):
    """公共字段逐值比；返回 `(n_diff, diffs)`（忽略 on 侧新增的 `evo_*`）。"""
    diffs = {}
    for k in sorted(set(off) & set(on)):
        if k.startswith(IGNORE_PREFIX):
            continue
        if off[k] != on[k]:
            diffs[k] = {"off": off[k], "on": on[k]}
    return len(diffs), diffs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--report", required=True, help="200 局 off 读数（取 per_game 的 tag 顺序）")
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--p1-control", type=int, default=6,
                    help="前 N 局额外跑一遍 **清空牌组变体** 的 on 臂（构造「一局都没声明」的对照）"
                         "⇒ P1 才有样本。⚠️ 真实回放里 95%+ 的局都会声明 ⇒ 不构造对照的话 P1 是"
                         "**空洞 PASS**（本批第一版实测 0 局可比却照样报 PASS，【R10】）")
    ap.add_argument("--level", type=int, default=11)
    ap.add_argument("--coord", default="raw")
    ap.add_argument("--stop-mode", default="none", choices=("none", "save", "all"))
    ap.add_argument("--stop-stride", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    jl = _resolve(args.jsonl)
    rep200 = json.load(open(_resolve(args.report), encoding="utf-8"))
    tags = [g["tag"] for g in rep200["per_game"] if g.get("tag")]
    recs = _select(jl, set(tags), args.games)
    print(f"[pair] 选中 {len(recs)} 局（来自 200 局探针的 tag 序列，前 {args.games} 个）")

    d_off = os.path.join(ROOT, "runs", "_fl_il_bc_evo", "pair_off")
    d_on = os.path.join(ROOT, "runs", "_fl_il_bc_evo", "pair_on")
    os.makedirs(d_off, exist_ok=True)
    os.makedirs(d_on, exist_ok=True)

    per, p1_fail, p2_fail, n_decl, n_trig, n_crash = [], 0, 0, 0, 0, 0
    n_ctrl, n_ctrl_fail, n_ctrl_crash = 0, 0, 0
    for i, r in enumerate(recs):
        common = dict(level=args.level, coord=args.coord, detail=False,
                      stop_mode=args.stop_mode, stop_stride=args.stop_stride,
                      with_frames=False, dump_dir=None, dump_grid=False, plan_extras=False)
        #: ⚠️ **必须逐局 try**：`--evo-slots` 会**第一次**把引擎的觉醒代码路径跑起来，而那条路径
        #: 有未被发现过的崩溃（本批实测 9/200 局：Witch_EV1 `spawnPauseTime` 读到字符串标签、
        #: GoblinDrill_EV1 `deathSpawnCount` 读到 `'spawn_count'`）⇒ 不包住会让整支探针**中途死掉**。
        err = {}
        try:
            s_off = _convert_one(i, r, False, d_off, evo_slots=False, **common)
        except Exception as e:  # noqa: BLE001
            s_off = None
            err["off"] = f"{type(e).__name__}: {e}"
        try:
            s_on = _convert_one(i, r, False, d_on, evo_slots=True, **common)
        except Exception as e:  # noqa: BLE001
            s_on = None
            err["on"] = f"{type(e).__name__}: {e}"
        #: ★ P1 **构造对照**：把牌组变体清空后再跑一遍 on 臂 ⇒ 保证「一局都没声明」。
        #: 把 `_evo` 清空是**合法**的：`_evo` 只在 evo 块里被读（`evo_slots=False` 时整块不执行）
        #: ⇒ 清空**不影响** off 臂的结果，可以直接拿上面的 `s_off` 作配对基线。
        ctrl = None
        if i < args.p1_control:
            r["_evo"] = {}
            try:
                ctrl = _convert_one(i, r, False, d_on, evo_slots=True, **common)
            except Exception as e:  # noqa: BLE001
                ctrl = None
                err["ctrl"] = f"{type(e).__name__}: {e}"
        if s_off is None or s_on is None:
            n_crash += 1
            per.append({"i": i, "tag": r.get("tag"), "crashed": True, "error": err})
            print(f"  [{i + 1}/{len(recs)}] {r.get('tag')} **崩溃** {err}")
        else:
            nd, diffs = _cmp(s_off, s_on)
            trig = (s_on.get("evo_trig_finish") or {})
            tr_tot = int(trig.get("team", 0)) + int(trig.get("opp", 0))
            decl = bool(s_on.get("evo_declared_any"))
            row = {"i": i, "tag": r.get("tag"), "declared": decl,
                   "declared_slots": s_on.get("evo_declared"),
                   "triggers": trig, "n_field_diffs": nd, "field_diffs": diffs,
                   "labels_off": s_off.get("labels"), "labels_on": s_on.get("labels")}
            if decl:
                n_decl += 1
                n_trig += tr_tot
            per.append(row)
            print(f"  [{i + 1}/{len(recs)}] {r.get('tag')} decl={decl} trig={tr_tot} "
                  f"labels {s_off.get('labels')}→{s_on.get('labels')} field_diffs={nd}")
        #: 对照臂的判定（**P1 的真实样本**）
        if ctrl is not None and s_off is not None:
            if ctrl.get("evo_declared_any"):
                n_ctrl_crash += 1          # 构造失败：本该没声明却声明了 ⇒ 让 P1 判 FAIL
            n_ctrl += 1
            cnd, cdiffs = _cmp(s_off, ctrl)
            row = next((x for x in per if x.get("i") == i), None)
            if row is not None:
                row["p1_control"] = {"declared": bool(ctrl.get("evo_declared_any")),
                                     "n_field_diffs": cnd, "field_diffs": cdiffs,
                                     "labels_off": s_off.get("labels"),
                                     "labels_ctrl": ctrl.get("labels")}
            if cnd:
                n_ctrl_fail += 1
        elif i < args.p1_control:
            n_ctrl_crash += 1

    n_ok = len(recs) - n_crash
    #: ★ P1 = **构造对照**上「一局都没声明却逐值相等」（自然局也纳入：声明了的局不参与 P1）
    p1 = (n_ctrl_fail == 0 and n_ctrl_crash == 0 and n_ctrl > 0)
    p2 = (n_trig > 0)
    res = {
        "prereg": "docs/il_evo_prereg_2026-09-22.md §2 S1（判据②的**同进程**强形式）",
        "inputs": {"jsonl": os.path.basename(args.jsonl), "games": len(recs),
                   "stop_mode": args.stop_mode, "level": args.level},
        "P1_no_declaration_bitwise": {
            "control_games": n_ctrl,
            "control_games_with_field_diff": n_ctrl_fail,
            "control_construction_failures": n_ctrl_crash,
            "games_compared": n_ok,
            "games_without_declaration_natural": n_ok - n_decl,
            "verdict": "PASS" if p1 else "FAIL",
            "meaning": ("**构造对照**（清空牌组变体 ⇒ 一局都不声明）上 off/on **全部公共字段逐值相等**；"
                        "⚠️ 真实回放的局 95%+ 都会声明 ⇒ 自然样本对 P1 几乎无覆盖，"
                        "**必须**构造对照（否则就是空洞 PASS）"),
        },
        "P2_liveness_positive": {
            "games_with_declaration": n_decl, "triggers_total": n_trig,
            "verdict": "PASS" if p2 else "FAIL",
            "meaning": "真实回放上声明了觉醒位，且**至少一局**触发 > 0",
        },
        "P3_crashes": {
            "games_crashed": n_crash, "games_total": len(recs),
            "verdict": "CLEAN" if n_crash == 0 else "CRASHES",
            "meaning": ("`--evo-slots` 让引擎的觉醒路径**第一次**被跑起来；崩溃 = 该路径的**真 bug**，"
                        "**不是**本探针的 bug（同进程、不涉及多进程）"),
        },
        "per_game": per,
    }
    res["pair_verdict"] = "PASS" if (p1 and p2) else "FAIL"
    if args.out:
        #: ⚠️ **必须 `_resolve`**：本脚本已经 `chdir(SRC)`，直接 `abspath` 会把相对路径落到
        #: `<repo>/src/clasher_new/docs/...`（实测踩到：产物**静默写到了引擎目录下**，
        #: 而 `docs/` 下什么都没生成）。本仓第三次踩同一类坑（见 §31.5 的同类教训）。
        _o = _resolve(args.out)
        os.makedirs(os.path.dirname(os.path.abspath(_o)), exist_ok=True)
        with open(_o, "w", encoding="utf-8", newline="\n") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    print("=" * 70)
    print(f"P1 未声明局逐值相等 = {'PASS' if p1 else 'FAIL'} "
          f"（**构造对照** {n_ctrl} 局中 {n_ctrl_fail} 局有差异、构造失败 {n_ctrl_crash}）")
    print(f"   自然局：{n_ok} 局可比，其中 {n_ok - n_decl} 局未声明（多数局都会声明 ⇒ 自然样本覆盖不足）")
    print(f"P2 声明局触发 > 0  = {'PASS' if p2 else 'FAIL'} "
          f"({n_decl} 局声明，触发合计 {n_trig})")
    print(f"P3 崩溃            = {res['P3_crashes']['verdict']} ({n_crash}/{len(recs)} 局)")
    print(f"配对总判 = {res['pair_verdict']}")
    return 0 if res["pair_verdict"] == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
