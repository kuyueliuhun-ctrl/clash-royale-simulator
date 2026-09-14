# -*- coding: utf-8 -*-
"""只读取证：联赛回放的**卡牌使用 / 圣水 / 部署节奏 / 动作合法性**（只读、离线）。

动机（2026-09-14）：外部独立调研在回放里发现了一组**文档里没有**的行为事实
（`docs/external_review_verification_2026-09-13.md` →「真正的新贡献」）：
Xbow 几乎不出、少数便宜卡占绝对多数、**部署时圣水几乎见底**、同路重部署间隔很长。
它们集中指向一个文档里没有的问题层次 ——「**模型连局部最优都做不到**」——
正好是价值/critic 那条线之外、解释"上限不动"的另一半。

本脚本把它们变成**可复算的仪器**（【红线 R4】：一切数字脚本复算，禁止手抄）。

──────────────────────────────────────────────────────────────────────────
四个必须在读数字前定死的口径（每一个都会让"看起来相同"的数字差出几倍）
──────────────────────────────────────────────────────────────────────────
1. **`frame["elixir0"]` 是「部署之后」的读数**（已扣本帧牌费）。实测互证：首帧 t=0.5
   打出 Knight(3) 后 `elixir0=2.179`，而起手 5.0 + 0.5s×回复 0.179 = 5.179。
   ⇒ 本脚本同时报 **post（回放原值）** 与 **pre = post + Σ去重卡费**，并显式标注。
2. **回放文件是「每 10000 步一个」的全量转储，每个文件各自含 40 局。**
   ⇒ 把 11 个文件当集合与只看最后一个文件会得到完全不同的数字（外部调研据此产生过
   "11 文件 / 40 局 / 733 部署"三者互相矛盾的写法）。本脚本**强制分开报**
   （§1 per-file 表 + 显式 POOLED）。
3. **同一帧的 bundle 可以对同一个槽位重复出手**（K_MAX=4 的 4 个子动作里出现
   `('deploy', slot, x, y)` 重复）⇒ `run_league._bundle_cards` 会把同一张卡记多次。
   ⇒ 本脚本按**帧内去重槽位**统计卡牌使用；重复子动作单列（浪费的动作容量）。
4. **「幽灵动作 y≥20」这条旧口径是错的**（`scripts/forensics_response.py` 注释里的
   "引擎未执行"）。引擎规则（`rl/action_mask._position_legal`）：**部队/建筑 y≥21 非法**、
   **法术可打任意格**。实测本 run：部队最大 y=20（从未越界）、法术最大 y=31。
   判"到底执行了没有"的独立证据 = **本帧费用有没有被扣**（`pre ≤ 10 + 回复`）：
   y=20 的部队落点与 y≥21 的法术落点，`pre` 全部 ≤ 5.2 ⇒ **全部执行了**。
   ⇒ **本回放里没有非法部署**；旧 `y≥20` 过滤器把合法进攻法术误判成幽灵。
   §5 把这个缺陷**量化**出来（以免后续会话照抄）。

对照：防守响应 / 拦截 / 接敌率在 `scripts/forensics_response.py`（注意其 y≥20 口径）；
参数位移指纹在 `scripts/value_displacement_scan.py`。本脚本不重复。

用法（在 src/clasher_new 下）：
  PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe \
      ../../scripts/forensics_card_usage.py --replays runs/d1_long_100k/replays \
      --out ../../docs/forensics_card_usage_d1_long_100k.json
"""

import argparse
import glob
import json
import os
import pickle
import sys
from collections import Counter

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src", "clasher_new")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

#: 世界坐标口径（与 `forensics_response.py` 同源）：河 y≈16；x<9 左路，x≥9 右路。
RIVER = 16.0
LANE_SPLIT = 9.0
#: 动作语义冻结的槽位数（【红线 R11】`K_MAX=4`）；`_bundle_cards` 只统计 1..K_MAX 槽。
K_MAX = 4
#: 引擎对**部队/建筑**的硬边界：`_position_legal` 里 `pos.y >= 21.0` ⇒ 本地 y≥21 非法。
UNIT_Y_MAX = 20
#: 圣水回复容差（0.5s/帧 ÷ 2.8s/点 = 0.179，双倍圣水期 0.357）⇒ 0.6 当"费用已扣"上界。
REGEN_TOL = 0.6
#: 旧口径（`forensics_response.py`）的幽灵阈值 —— 只为 §5 缺陷量化保留。
LEGACY_GHOST_Y = 20


# ---------------------------------------------------------------------------
def load_cost_map(src=_SRC):
    """卡名 → 圣水费。**权威源 = 引擎运行时同一张表**（不在脚本里手抄费率）。

    优先级（与引擎一致）：`gamedata.json['items']['spells'][i]['manaCost']`
    ——`card_utils.Card.elixir = data.get('manaCost', 0)`，`data = gamedata['items']['spells']`，
    即**引擎扣费就是读这里**。实测该表 148 行、144 行有费用，8 张本副牌全部命中
    （Knight3/Archer3/Skeletons1/IceWizard3/Xbow6/Fireball4/Tesla4/Log2）。
    兜底：`cards_stats_*.json[].elixir`（覆盖不全，仅作交叉核对）。
    """
    cost = {}
    p = os.path.join(src, "gamedata.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            gd = json.load(f)
        for row in ((gd.get("items") or {}).get("spells") or []):
            if not isinstance(row, dict):
                continue
            nm, el = row.get("name"), row.get("manaCost")
            if nm and isinstance(el, (int, float)) and nm not in cost:
                cost[nm] = float(el)
    for fn in ("cards_stats_characters.json", "cards_stats_building.json",
               "cards_stats_spell.json"):
        p = os.path.join(src, fn)
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            continue
        for row in data:
            if not isinstance(row, dict):
                continue
            nm, el = row.get("name"), row.get("elixir")
            if nm and isinstance(el, (int, float)) and nm not in cost:
                cost[nm] = float(el)
    return cost


def load_files(path):
    """path 可以是目录（取全部 league_*.pkl，按步数排序）或单个 pkl。"""
    if os.path.isdir(path):
        def _step(p):
            digits = "".join(ch for ch in os.path.basename(p) if ch.isdigit())
            return int(digits or 0)
        files = sorted(glob.glob(os.path.join(path, "league_*.pkl")), key=_step)
    else:
        files = [path]
    out = []
    for p in files:
        with open(p, "rb") as f:
            d = pickle.load(f)
        games = d["games"] if isinstance(d, dict) and "games" in d else d
        out.append((os.path.basename(p), games))
    return out


def _blank(step=None):
    return {
        "step": step,
        "n_games": 0, "n_games_side0_main": 0, "n_frames": 0,
        "n_deploy_frames": 0, "n_subactions": 0, "n_subactions_dedup": 0,
        "dup_subaction_frames": 0, "subaction_hist": Counter(),
        "n_card_plays": 0, "card_counts": Counter(), "card_y_max": {},
        "n_unit_deploys": 0, "n_unit_y_ge21": 0, "n_spell_plays": 0,
        "n_spell_y_ge21": 0,
        "legacy_ghost_plays": 0, "legacy_ghost_frames": 0,
        "elix_post": [], "elix_pre": [], "dt_all": [], "dt_same_lane": [],
        "elix_all": [], "n_frames_main": 0,
        "unknown_cost_frames": 0, "exec_violations": 0,
        "n_single_frames": 0, "elixir_spent": 0.0,
        "deck_cards": Counter(), "pairs": Counter(),
    }


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", required=True,
                    help="回放目录（全部 league_*.pkl）或单个 pkl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--topk", type=int, default=5)
    a = ap.parse_args()

    cost_map = load_cost_map()
    files = load_files(a.replays)
    print(f"=== 卡牌使用 / 圣水 / 部署节奏取证 | {a.replays} | "
          f"{len(files)} 个回放文件 | 费率表 {len(cost_map)} 张 ===", flush=True)

    import numpy as np

    # 法术判定直接问引擎（`card_utils.Card.type == 'spell'`），不在脚本里手抄卡表
    _spell_cache = {}

    def _is_spell(nm):
        if nm not in _spell_cache:
            try:
                from card_utils import Card
                _spell_cache[nm] = (Card(nm).type == "spell")
            except Exception:
                _spell_cache[nm] = False
        return _spell_cache[nm]

    per_file = {}
    pooled = _blank()
    for name, games in files:
        step = int("".join(ch for ch in name if ch.isdigit()) or 0)
        st = _blank(step)
        st["n_games"] = len(games)
        for g in games:
            meta = g.get("meta") or {}
            pair = tuple(meta.get("pair") or [])
            st["pairs"]["/".join(str(x) for x in pair)] += 1
            if str(meta.get("side0")) != "main":
                # 只有 player-0 的 bundle/cards/elixir0 被记录 ⇒ 非 main 先手局不进"我方"统计
                continue
            st["n_games_side0_main"] += 1
            decks = meta.get("decks") or []
            if decks and decks[0]:
                for c in decks[0]:
                    st["deck_cards"][c] += 1
            frames = g.get("frames") or []
            st["n_frames"] += len(frames)
            # 「会不会攒费」的对照组：**全部帧**的 elixir 分布（含不出手的帧）
            st["elix_all"].extend(float(fr.get("elixir0") or 0.0) for fr in frames)
            st["n_frames_main"] += len(frames)
            prev_t, prev_lane = None, None
            for fr in frames:
                # 与 `run_league._bundle_cards` **逐字同口径的过滤**（否则 cards 与 bundle 错位）
                deploys = [b for b in (fr.get("bundle") or [])
                           if b and b[0] == "deploy" and 1 <= int(b[1]) <= K_MAX]
                if not deploys:
                    continue
                st["n_deploy_frames"] += 1
                st["n_subactions"] += len(deploys)
                st["subaction_hist"][len(deploys)] += 1
                cards = list(fr.get("cards") or [])
                # ⚠️ `_bundle_cards` 是**按出现顺序 append** 的（不是按槽位号索引）⇒
                #    `cards[i]` 对应 `deploys[i]`。绝不能写 `cards[slot-1]`。
                slots, seen = [], set()
                uniq_cards = []
                for i, b in enumerate(deploys):
                    k = int(b[1])
                    if k in seen:
                        continue
                    seen.add(k)
                    slots.append((k, int(b[2]), int(b[3])))
                    uniq_cards.append(cards[i] if i < len(cards) else None)
                st["n_subactions_dedup"] += len(slots)
                if len(slots) < len(deploys):
                    st["dup_subaction_frames"] += 1
                uniq_pairs = [(c, s) for c, s in zip(uniq_cards, slots) if c]
                for c, (_, _, y) in uniq_pairs:
                    st["card_counts"][c] += 1
                    st["n_card_plays"] += 1
                    st["card_y_max"][c] = max(st["card_y_max"].get(c, -1), y)
                    if _is_spell(c):
                        st["n_spell_plays"] += 1
                        if y >= UNIT_Y_MAX + 1:
                            st["n_spell_y_ge21"] += 1
                    else:
                        st["n_unit_deploys"] += 1
                        if y >= UNIT_Y_MAX + 1:
                            st["n_unit_y_ge21"] += 1
                    if y >= LEGACY_GHOST_Y:
                        st["legacy_ghost_plays"] += 1
                if any(int(b[3]) >= LEGACY_GHOST_Y for b in deploys):
                    st["legacy_ghost_frames"] += 1
                # ---- 圣水：只取"恰好 1 个去重子动作"的帧（此时 pre=post+费 精确）----
                if len(slots) != 1 or len(uniq_pairs) != 1:
                    continue
                c = uniq_pairs[0][0]
                cc = cost_map.get(c)
                if cc is None:
                    st["unknown_cost_frames"] += 1
                    continue
                st["n_single_frames"] += 1
                post = float(fr.get("elixir0") or 0.0)
                pre = post + cc
                st["elixir_spent"] += cc
                if pre > 10.0 + REGEN_TOL:
                    # 费用没被扣 ⇒ 该动作未被引擎执行。本 run 实测应恒为 0（合法性与扣费互证）
                    st["exec_violations"] += 1
                st["elix_post"].append(post)
                st["elix_pre"].append(pre)
                t = float(fr.get("t") or 0.0)
                lane = 0 if slots[0][1] < LANE_SPLIT else 1
                if prev_t is not None:
                    st["dt_all"].append(t - prev_t)
                    if lane == prev_lane:
                        st["dt_same_lane"].append(t - prev_t)
                prev_t, prev_lane = t, lane
        per_file[name] = st

    for st in per_file.values():
        for k, v in st.items():
            if isinstance(v, Counter):
                pooled[k].update(v)
            elif isinstance(v, list):
                pooled[k].extend(v)
            elif k == "card_y_max":
                for kk, vv in v.items():
                    pooled[k][kk] = max(pooled[k].get(kk, -1), vv)
            elif isinstance(v, int) and k != "step":
                pooled[k] += v

    def _stat(x):
        if len(x) == 0:
            return {"n": 0}
        return {"n": int(len(x)), "mean": round(float(x.mean()), 4),
                "median": round(float(np.median(x)), 4),
                "p10": round(float(np.quantile(x, 0.10)), 4),
                "p90": round(float(np.quantile(x, 0.90)), 4),
                "min": round(float(x.min()), 4), "max": round(float(x.max()), 4)}

    def _q(x, q):
        return None if len(x) == 0 else round(float(np.quantile(x, q)), 4)

    def _frac(x, f):
        return None if len(x) == 0 else round(float(f(x).mean()), 4)

    def _cost_table(counts, elix_all, cmap):
        """§7：把「出手分布」与「费用可达性」并排放（脚本复算，禁手抄）。

        对牌组里每个**出现过的费用档 c**：`per_card_share` = 该档每张卡的平均出手占比；
        `p_afford` = 全部帧里 `elixir0 >= c` 的比例（= 该档的**可达窗口**上限）。
        若策略只是"打得起什么就打什么"，两列应同向且量级相近。
        """
        tot = sum(counts.values())
        el = np.asarray(elix_all, dtype=np.float64)
        by_cost = {}
        for c, n in counts.items():
            cc = cmap.get(c)
            if cc is None:
                continue
            by_cost.setdefault(float(cc), []).append((c, n))
        out = []
        for c in sorted(by_cost):
            items = by_cost[c]
            share = sum(n for _, n in items) / max(1, tot)
            out.append({
                "cost": c, "n_cards": len(items),
                "plays": int(sum(n for _, n in items)),
                "bucket_share": round(share, 4),
                "per_card_share": round(share / len(items), 4),
                "cards": [k for k, _ in sorted(items, key=lambda x: -x[1])],
                "p_afford_all_frames": (round(float((el >= c).mean()), 6)
                                        if el.size else None),
                "n_frames_afford": (int((el >= c).sum()) if el.size else None),
            })
        return out

    def summarize(st):
        cc = st["card_counts"]
        tot = sum(cc.values())
        top = cc.most_common(a.topk)
        cheap = sum(v for k, v in cc.items()
                    if cost_map.get(k) is not None and cost_map[k] <= 3)
        ep = np.asarray(st["elix_post"], dtype=np.float64)
        er = np.asarray(st["elix_pre"], dtype=np.float64)
        out = {
            "step": st["step"],
            "n_games": st["n_games"], "n_games_side0_main": st["n_games_side0_main"],
            "n_frames": st["n_frames"], "n_deploy_frames": st["n_deploy_frames"],
            "n_subactions": st["n_subactions"],
            "n_subactions_dedup": st["n_subactions_dedup"],
            "dup_subaction_frames": st["dup_subaction_frames"],
            "subaction_hist": {str(k): v for k, v in sorted(st["subaction_hist"].items())},
            "n_card_plays": int(tot),
            "card_counts": dict(cc.most_common()),
            "card_share": {k: round(v / max(1, tot), 4) for k, v in cc.most_common()},
            "card_y_max": st["card_y_max"],
            "card_cost": {k: cost_map.get(k) for k in cc},
            "n_spells_distinct": sum(1 for k in cc if _is_spell(k)),
            "topk": a.topk, "topk_cards": [k for k, _ in top],
            "topk_share": round(sum(v for _, v in top) / max(1, tot), 4),
            "cheap_le3_share": round(cheap / max(1, tot), 4),
            "deck_cards": dict(st["deck_cards"]),
            "unplayed_in_deck": sorted(set(st["deck_cards"]) - set(cc)),
            "legality": {
                "unit_y_max_legal": UNIT_Y_MAX,
                "n_unit_deploys": st["n_unit_deploys"],
                "n_unit_y_ge21_illegal": st["n_unit_y_ge21"],
                "n_spell_plays": st["n_spell_plays"],
                "n_spell_y_ge21_legal": st["n_spell_y_ge21"],
                "legacy_ghost_y": LEGACY_GHOST_Y,
                "legacy_filter_marks": st["legacy_ghost_plays"],
                "legacy_filter_frames": st["legacy_ghost_frames"],
                "n_single_frames": st["n_single_frames"],
                "exec_violations_should_be_0": st["exec_violations"],
            },
            "unknown_cost_frames": st["unknown_cost_frames"],
            "elixir_spent": round(float(st["elixir_spent"]), 3),
            "elix_post": _stat(ep), "elix_pre": _stat(er),
            "dt_all": _stat(np.asarray(st["dt_all"], dtype=np.float64)),
            "dt_same_lane": _stat(np.asarray(st["dt_same_lane"], dtype=np.float64)),
            "elix_all_frames": _stat(np.asarray(st["elix_all"], dtype=np.float64)),
            "cost_table": _cost_table(st["card_counts"], st["elix_all"], cost_map),
            "all_frames_frac_ge6": _frac(np.asarray(st["elix_all"], dtype=np.float64),
                                         lambda x: x >= 6.0),
            "all_frames_frac_ge8": _frac(np.asarray(st["elix_all"], dtype=np.float64),
                                         lambda x: x >= 8.0),
            "post_median": _q(ep, 0.5), "pre_median": _q(er, 0.5),
            "post_frac_lt_0p5": _frac(ep, lambda x: x < 0.5),
            "post_frac_lt_1": _frac(ep, lambda x: x < 1.0),
            "pre_frac_ge_6": _frac(er, lambda x: x >= 6.0),
            "pre_frac_ge_8": _frac(er, lambda x: x >= 8.0),
            "lane_dt_median": _q(np.asarray(st["dt_same_lane"], dtype=np.float64), 0.5),
        }
        if st.get("pairs"):
            out["pairs"] = dict(st["pairs"])
        return out

    res = {"replays": a.replays, "cost_map_size": len(cost_map),
           "per_file": {k: summarize(v) for k, v in per_file.items()},
           "pooled": summarize(pooled)}
    pf, p = res["per_file"], res["pooled"]
    last = list(pf)[-1]

    print("\n--- §1 作用域表（外部调研『11 文件 / 40 局 / 733 部署』矛盾的根源）---")
    print(f"  {'file':22s} {'局':>4s} {'帧':>7s} {'部署帧':>7s} {'子动作':>7s} "
          f"{'去重后':>7s} {'卡牌次':>7s} {'部署后中位':>10s} {'同路间隔中位':>12s}")
    for name, r in pf.items():
        pm = r["post_median"] if r["post_median"] is not None else float("nan")
        lm = r["lane_dt_median"] if r["lane_dt_median"] is not None else float("nan")
        print(f"  {name:22s} {r['n_games']:4d} {r['n_frames']:7d} "
              f"{r['n_deploy_frames']:7d} {r['n_subactions']:7d} "
              f"{r['n_subactions_dedup']:7d} {r['n_card_plays']:7d} "
              f"{pm:10.3f} {lm:12.3f}")
    print(f"  {'POOLED(全部)':22s} {p['n_games']:4d} {p['n_frames']:7d} "
          f"{p['n_deploy_frames']:7d} {p['n_subactions']:7d} "
          f"{p['n_subactions_dedup']:7d} {p['n_card_plays']:7d} "
          f"{p['post_median']:10.3f} {p['lane_dt_median']:12.3f}")

    for tag, r in (("POOLED", p), (f"仅 {last}", pf[last])):
        print(f"\n--- §2 卡牌使用（{tag}）---")
        print(f"  卡牌打出（帧内去重槽位）={r['n_card_plays']}  "
              f"子动作直方图={r['subaction_hist']}  含重复子动作的帧={r['dup_subaction_frames']}")
        for c, n in r["card_counts"].items():
            print(f"    {c:12s} {n:6d}  {r['card_share'][c]:7.2%}  "
                  f"费={r['card_cost'].get(c)}  最大落点 y={r['card_y_max'].get(c)}")
        print(f"  top{r['topk']} = {r['topk_cards']} 合计 {r['topk_share']:.2%}；"
              f"费≤3 的卡合计 {r['cheap_le3_share']:.2%}")
        print(f"  牌组里从未打出的卡: {r['unplayed_in_deck']}")

    print("\n--- §3 部署时的圣水（只统计『恰好 1 个子动作』的帧，pre=post+费 精确）---")
    for tag, r in (("POOLED", p), (f"仅 {last}", pf[last])):
        if not r["elix_post"]["n"]:
            continue
        print(f"  [{tag}] post(回放原值) n={r['elix_post']['n']} "
              f"mean={r['elix_post']['mean']:.3f} median={r['post_median']:.3f} "
              f"p10={r['elix_post']['p10']:.3f} p90={r['elix_post']['p90']:.3f} "
              f"max={r['elix_post']['max']:.3f} | <0.5={r['post_frac_lt_0p5']:.2%} "
              f"<1={r['post_frac_lt_1']:.2%}")
        print(f"  [{tag}] pre (=post+费)   n={r['elix_pre']['n']} "
              f"mean={r['elix_pre']['mean']:.3f} median={r['pre_median']:.3f} "
              f"p10={r['elix_pre']['p10']:.3f} p90={r['elix_pre']['p90']:.3f} "
              f"max={r['elix_pre']['max']:.3f} | ≥6={r['pre_frac_ge_6']:.2%} "
              f"≥8={r['pre_frac_ge_8']:.2%}")

    print("\n--- §3b 「会不会攒费」：**全部帧**（含不出手的帧）的圣水分布 ---")
    for tag, r in (("POOLED", p), (f"仅 {last}", pf[last])):
        s2 = r["elix_all_frames"]
        if not s2.get("n"):
            continue
        print(f"  [{tag}] 全帧 elixir0 n={s2['n']} mean={s2['mean']:.3f} "
              f"median={s2['median']:.3f} p90={s2['p90']:.3f} max={s2['max']:.3f} "
              f"| ≥6={r['all_frames_frac_ge6']:.2%} ≥8={r['all_frames_frac_ge8']:.2%}")

    print("\n--- §4 部署节奏 ---")
    for key, lab in (("dt_all", "任意两次部署间"), ("dt_same_lane", "同路两次部署间")):
        s = p[key]
        if s.get("n"):
            print(f"  [POOLED] {lab:12s} n={s['n']:5d} mean={s['mean']:.3f} "
                  f"median={s['median']:.3f} p10={s['p10']:.3f} p90={s['p90']:.3f} "
                  f"max={s['max']:.3f}")

    print("\n--- §7 费用可达性：出手分布 vs 「圣水 ≥ 费用」的帧占比 ---")
    print(f"  {'费':>3s} {'张数':>4s} {'出手':>6s} {'档占比':>8s} {'每张占比':>8s} "
          f"{'可达窗口':>10s} {'可达帧数':>9s}  卡")
    for r in p["cost_table"]:
        print(f"  {r['cost']:3.1f} {r['n_cards']:4d} {r['plays']:6d} "
              f"{r['bucket_share']:8.2%} {r['per_card_share']:8.2%} "
              f"{(r['p_afford_all_frames'] if r['p_afford_all_frames'] is not None else float('nan')):10.6f} "
              f"{(r['n_frames_afford'] if r['n_frames_afford'] is not None else -1):9d}  "
              f"{','.join(r['cards'])}")

    lg = p["legality"]
    print("\n--- §5 合法性核验（推翻旧『幽灵动作 y≥20』口径）---")
    print(f"  引擎规则：部队/建筑 y≥21 非法（=本地 y>{UNIT_Y_MAX}）；法术可打任意格。")
    print(f"  实测：部队打出 {lg['n_unit_deploys']} 次、越界(y≥21) {lg['n_unit_y_ge21_illegal']} 次；"
          f"法术打出 {lg['n_spell_plays']} 次，其中 y≥21（**合法**远距离施放）"
          f"{lg['n_spell_y_ge21_legal']} 次。")
    print(f"  独立证据（费用是否被扣）：单子动作帧 {lg['n_single_frames']} 个，"
          f"pre>10+{REGEN_TOL} 的**未扣费**帧 = {lg['exec_violations_should_be_0']}（应为 0）。")
    print(f"  旧口径会误标：{lg['legacy_filter_marks']} 次打出 / "
          f"{lg['legacy_filter_frames']} 帧 ⇒ 旧口径的『幽灵』绝大多数是**我方进攻法术落点**。")

    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        print(f"\n[out] {a.out}", flush=True)
    return res


if __name__ == "__main__":
    main()
