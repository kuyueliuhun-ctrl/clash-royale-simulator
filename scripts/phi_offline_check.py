# -*- coding: utf-8 -*-
"""离线逐帧复算 Phi（资源账势函数）+ 与回放 reward 对账（只读、纯 CPU、零训练成本）。

回答 `docs/frame_credit_proposal_review_2026-09-18.md` §7.8 第 2 条登记的**未做数值验证**
的推论（同文件 §7.1 下半格）：

    「一个 2 费 IceGolemite 出去、没打出任何伤害、被吃掉
      ⇒ 我方 Phi 净 -2 费 ⇒ 奖励 -1.0（edw=0.5）」

Phi 权威定义（`rl/env_wrapper.py`）：
    ``_v_share[pid] = {entity_id: 部署份额}``、``_active_v[pid] = sum(share)``；
    ``Phi = 手牌圣水 + _active_v``；奖励的 edw 项 = ``edw(t) * [dPhi_我 - dPhi_敌]``。
    部署帧：``share = cost / len(该帧新增 Troop|Building 实体)``（``_deploy_ledger``），
    spell 卡（除 Mirror）不入账（产物免费）；死亡即注销份额（``_collect_deaths``）；
    塔（id<=6）不入账（reset 时 ``_seen_max_id = max(entities)``，新实体从 7 起）。

本脚本做两件事（都不改任何既有源码）：
  ① **离线重建 dPhi**：用回放帧的 ``cards`` / ``opp_played`` / ``entities`` 重建
     ``_v_share`` 台账 + 出牌花费，逐帧算出 ``dPhi`` 与 edw 项；
  ② **对账**：把 reward 其余可离线复算的项（crown / 塔血 / unit_dmg / terminal）一并复算，
     与回放帧 ``reward`` 比较并报残差。

口径（读数字前先看）：
- 回放帧 schema >= 4 才有实体 ``id``（12 元组，`rl/replay.py:98`）；schema <= 3 只有 10 位
  ⇒ 本脚本用 (name, player) + 最近位置贪心匹配**合成**实体 id（失配会打印）。
- 代价真源 = `card_utils.Card(name).elixir`（引擎扣费读同一张表），脚本内不手抄费率。
- reward 的 ``d_elixir`` 只含**出牌花费**（部署期间不推进时间）；回放帧 ``elixir0``
  是「已扣费 + 本步 0.5s 回复已计入」的读数 ⇒ 两者不可混用，本脚本用卡费直接算。
- 残差里**必然**含非法动作罚（回放不记录 ``bundle_ok``）、Mirror 变价、
  引擎 ``_death_elixir_gift`` 等；这些**不影响** ① 的 dPhi 台账（只用卡费与实体生死）。
- 对账口径写死：``recon = crown + tower + edw*dPhi + unit_dmg + terminal``；
  ``terminal = +10 / -10 / -10``（胜 / 负 / 平局，`config.DEFAULT_REWARD.draw_penalty=10`）。

用法（在 src/clasher_new 下）：
  PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe \
      ../../scripts/phi_offline_check.py \
      --replays runs/run_schema4/replays --replays runs/long1m/replays \
      --out ../../docs/phi_offline_check.json
"""

import argparse
import glob
import json
import os
import pickle
import sys
from collections import Counter, defaultdict

# T1-1b 补齐（2026-09-19）：原先这里是**手写**的 UTF-8 兜底块（只处理 stdout、且不处理 stderr
# ⇒ traceback 在 GBK 下仍是乱码）。现收敛到 T1-1 的**单一实现**（两路 + errors='replace'）。
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                                 "src", "clasher_new"))
from rl.io_bootstrap import force_utf8_stdout  # noqa: E402
force_utf8_stdout()

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src", "clasher_new")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

DT = 1.0 / 60.0
PHASE_SWITCH_S = 120.0           # rl/env_wrapper.PHASE_SWITCH_S
UNIT_DMG_K = 0.0005              # config.DEFAULT_REWARD["unit_dmg_k"]
CROWN_W = 8.0
CROWN_LOSE_W = 10.0
WIN_BONUS = 10.0
LOSE_PENALTY = 10.0
DRAW_PENALTY = 10.0
NON_UNIT_KINDS = ("effect", "projectile")
#: 「冰人」候选卡名（引擎里 2 费冰人 = IceGolemite；进化形态 FreezeIceGolemite）
ICE_GOLEM_NAMES = ("IceGolemite", "FreezeIceGolemite")


def _stdout(s):
    try:
        print(s, flush=True)
    except UnicodeEncodeError:
        print(s.encode("utf-8", "replace").decode("utf-8", "replace"), flush=True)


def load_files(path):
    """目录（全部 league_*.pkl，按步数排序）或单个 pkl。"""
    if os.path.isdir(path):
        def _step(p):
            digits = "".join(ch for ch in os.path.basename(p) if ch.isdigit())
            return int(digits or 0)
        files = sorted(glob.glob(os.path.join(path, "league_*.pkl")), key=_step)
    else:
        files = [path]
    out = []
    for p in files:
        try:
            with open(p, "rb") as f:
                d = pickle.load(f)
        except Exception as e:                                    # noqa: BLE001
            _stdout(f"[skip] {p}: {type(e).__name__}")
            continue
        games = d["games"] if isinstance(d, dict) and "games" in d else d
        schema = d.get("schema", 1) if isinstance(d, dict) else 1
        out.append((os.path.basename(p), schema, games))
    return out


class _IdSynth:
    """schema<=3 的实体 id 合成：同 (name, player) 内按最近位置贪心配对。"""

    def __init__(self):
        self.next_id = 1000
        self.live = {}          # key -> [(synth_id, x, y)]

    def assign(self, rows):
        by_key = defaultdict(list)
        for r in rows:
            by_key[(r[0], r[1])].append(r)
        out = {}
        for key, rs in by_key.items():
            prev = list(self.live.get(key, []))
            used = [False] * len(prev)
            pairs = []
            for i, r in enumerate(rs):
                for j, pv in enumerate(prev):
                    pairs.append(((r[2] - pv[1]) ** 2 + (r[3] - pv[2]) ** 2, i, j))
            pairs.sort()
            match = {}
            for _d2, i, j in pairs:
                if i in match or used[j]:
                    continue
                match[i] = j
                used[j] = True
            new_live = []
            for i, r in enumerate(rs):
                if i in match:
                    sid = prev[match[i]][0]
                else:
                    self.next_id += 1
                    sid = self.next_id
                new_live.append((sid, r[2], r[3]))
                out[(r[0], r[1], r[2], r[3])] = sid
            self.live[key] = new_live
        return out


def frame_entities(fr, schema, synth):
    """返回 [(eid, name, player, kind, x, y, hp)]（eid 为 None 表示无法归属）。"""
    ents = fr.get("entities") or []
    if schema >= 4:
        return [(int(e[10]), str(e[0]), int(e[4]), str(e[5]), float(e[1]), float(e[2]),
                 float(e[3])) for e in ents if len(e) > 10]
    tag = synth.assign([(str(e[0]), int(e[4]), float(e[1]), float(e[2])) for e in ents])
    out = []
    for e in ents:
        nm, pl = str(e[0]), int(e[4])
        key = (nm, pl, float(e[1]), float(e[2]))
        out.append((tag.get(key), nm, pl, str(e[5]) if len(e) > 5 else "",
                    float(e[1]), float(e[2]), float(e[3])))
    return out


def phase_weights(t):
    """(tower_opp, tower_self, edw)：直接复用 `rl/env_wrapper._phase_weights`（单一来源）。"""
    from rl.env_wrapper import _phase_weights, _DEFAULT_REWARD
    return _phase_weights(dict(_DEFAULT_REWARD), t)


def regen_between(t_prev, t_now):
    """帧间圣水回复量（引擎 `battle.py:2786` 每 tick：t<120 → 2.8 / t<240 → 1.4 / 否则 2.8/3）。

    逐步精确积分（tick 数由帧间真实 `t` 差反推，最后一帧常被提前截断）。
    """
    n = int(round(max(0.0, (t_now - t_prev)) / DT))
    tot = 0.0
    for i in range(1, n + 1):
        t = t_prev + i * DT
        rt = 2.8 if t < 120.0 else (1.4 if t < 240.0 else 2.8 / 3.0)
        tot += DT / rt
    return tot


def tower_dmg(prev, now, mx):
    """per-tower 差异化定价 + lv11 锚归一化：复用 env_wrapper 的权威实现。"""
    from rl.env_wrapper import (_per_tower_norm_dmg, DEFAULT_TOWER_PREMIUM_K,
                                DEFAULT_KING_GATE)
    return _per_tower_norm_dmg(
        [prev[0] - now[0], prev[1] - now[1], prev[2] - now[2]],
        prev, now, mx,
        king=True, k=DEFAULT_TOWER_PREMIUM_K, king_gate=DEFAULT_KING_GATE)


def _cost(name, cost_map):
    return cost_map.get(name)


def analyse_game(g, schema, cost_map, gid=None):
    frames = g.get("frames") or []
    if not frames:
        return None
    from card_utils import Card
    synth = None if schema >= 4 else _IdSynth()
    meta = g.get("meta") or {}
    winner = g.get("winner")
    frames0 = frames[0]
    mx0 = list(frames0.get("towers0") or [0, 0, 0])
    mx1 = list(frames0.get("towers1") or [0, 0, 0])

    share = {0: {}, 1: {}}          # pid -> {eid: share}
    active_v = {0: 0.0, 1: 0.0}
    prev = None                      # 上一帧：{"ents": {eid: (pid,name,kind,hp)}, "fr": fr}
    rows, ice_events = [], []
    unknown_cost = 0
    multi_card_with_new = 0

    for k, fr in enumerate(frames):
        t = float(fr.get("t") or 0.0)
        ents = {}
        for eid, nm, pl, kind, x, y, hp in frame_entities(fr, schema, synth):
            if eid is None:
                continue
            ents[eid] = (pl, nm, kind, hp)

        # ---- 死亡注销（上一帧存活、本帧不见）----
        dead_shares = {0: 0.0, 1: 0.0}
        dead_names = {0: [], 1: []}
        if prev is not None:
            for eid, (pl, nm, kind, _hp) in prev["ents"].items():
                if eid in ents:
                    continue
                sh = share[pl].pop(eid, None)
                if sh is not None:
                    dead_shares[pl] += sh
                    dead_names[pl].append((eid, nm, sh))
                    active_v[pl] -= sh

        # ---- 新增实体 ----
        new_ids = {0: [], 1: []}
        for eid, (pl, nm, kind, _hp) in ents.items():
            if prev is not None and eid in prev["ents"]:
                continue
            if eid <= 6 or kind not in ("troop", "building"):
                continue
            new_ids[pl].append(eid)

        # ---- 出牌（我方 cards / 对手 opp_played）----
        cards = {0: [c for c in (fr.get("cards") or []) if c], 1: []}
        for o in (fr.get("opp_played") or []):
            c = o.get("card") if isinstance(o, dict) else o
            if c and c != "__ability__":
                cards[1].append(c)

        # ---- 花费交叉核对：从 elixir0 差分 + 逐步回复量反解（= 实际花费 - 死亡馈赠）----
        spend_elix = {0: 0.0, 1: 0.0}
        if prev is not None:
            tp = float(prev["fr"].get("t") or 0.0)
            rg = regen_between(tp, t)
            for pid, key in ((0, "elixir0"), (1, "elixir1")):
                before = float(prev["fr"].get(key) or 0.0)
                now = float(fr.get(key) or 0.0)
                spend_elix[pid] = before + rg - now

        # ---- 花费 + 份额入账 ----
        spend = {0: 0.0, 1: 0.0}
        additions = {0: 0.0, 1: 0.0}
        for pid in (0, 1):
            ledger_cards = []
            for c in cards[pid]:
                cc = _cost(c, cost_map)
                if cc is None:
                    unknown_cost += 1
                    continue
                spend[pid] += cc          # 所有成功出牌的卡费都进 elixir 账
                try:
                    is_spell = (Card(c).type == "spell")
                except Exception:                                  # noqa: BLE001
                    is_spell = False
                if is_spell and c != "Mirror":
                    continue              # 法术产物免费（_deploy_ledger 的 return）
                ledger_cards.append(cc)
            ids = new_ids[pid]
            if ids and ledger_cards:
                if len(ledger_cards) > 1:
                    multi_card_with_new += 1
                tot = float(sum(ledger_cards))
                sh = tot / len(ids)
                for eid in ids:
                    share[pid][eid] = sh
                    active_v[pid] += sh
                additions[pid] = tot

        # ---- 冰人实例登记 / 回填 ----
        for pid in (0, 1):
            for eid in new_ids[pid]:
                if ents[eid][1] in ICE_GOLEM_NAMES:
                    ice_events.append({
                        "eid": eid, "pid": pid, "name": ents[eid][1],
                        "deploy_frame": k, "deploy_t": t,
                        "share": share[pid].get(eid, 0.0),
                        "death_frame": None, "death_t": None, "lost": None,
                    })
            for eid, nm, sh in dead_names[pid]:
                if nm not in ICE_GOLEM_NAMES:
                    continue
                for ev in ice_events:
                    if (ev["eid"] == eid and ev["pid"] == pid
                            and ev["death_frame"] is None):
                        ev["death_frame"] = k
                        ev["death_t"] = t
                        ev["lost"] = sh
                        break

        # ---- reward 各项复算 ----
        dphi = {0: 0.0, 1: 0.0}
        crown_term = tower_term = edw_term = unit_term = 0.0
        if prev is not None:
            dphi[0] = -spend[0] + additions[0] - dead_shares[0]
            dphi[1] = -spend[1] + additions[1] - dead_shares[1]
            to, ts, edw = phase_weights(t)
            edw_term = edw * (dphi[0] - dphi[1])
            dc0 = int(fr.get("crown0") or 0) - int(prev["fr"].get("crown0") or 0)
            dc1 = int(fr.get("crown1") or 0) - int(prev["fr"].get("crown1") or 0)
            crown_term = CROWN_W * dc1 - CROWN_LOSE_W * dc0
            tower_term = (to * tower_dmg(list(prev["fr"].get("towers1") or [0, 0, 0]),
                                        list(fr.get("towers1") or [0, 0, 0]), mx1)
                          - ts * tower_dmg(list(prev["fr"].get("towers0") or [0, 0, 0]),
                                          list(fr.get("towers0") or [0, 0, 0]), mx0))
            hurt = {0: 0.0, 1: 0.0}
            for eid, (pl, nm, kind, bhp) in prev["ents"].items():
                if kind in NON_UNIT_KINDS or eid <= 6:
                    continue
                ahp = ents[eid][3] if eid in ents else 0.0
                if bhp > ahp:
                    hurt[pl] += bhp - ahp
            unit_term = UNIT_DMG_K * (hurt[1] - hurt[0])
        terminal = 0.0
        if k == len(frames) - 1:
            terminal = (WIN_BONUS if winner == 0
                        else -LOSE_PENALTY if winner is not None else -DRAW_PENALTY)
        recon_core = crown_term + tower_term + edw_term + unit_term
        recon = recon_core + terminal
        reward = float(fr.get("reward") or 0.0)
        rows.append({
            "k": k, "t": t, "last": (k == len(frames) - 1),
            "reward": reward, "recon": recon, "recon_core": recon_core,
            "resid": reward - recon, "resid_core": reward - recon_core,
            "crown": crown_term, "tower": tower_term,
            "edw_term": edw_term, "unit": unit_term, "terminal": terminal,
            "dphi0": dphi[0], "dphi1": dphi[1],
            "spend0": spend[0], "add0": additions[0], "dead0": dead_shares[0],
            "cards0": list(cards[0]), "cards1": list(cards[1]),
            "new0": len(new_ids[0]), "new1": len(new_ids[1]),
            "deadn0": len(dead_names[0]), "deadn1": len(dead_names[1]),
            "spend_elix0": spend_elix[0], "spend_elix1": spend_elix[1],
            "_gid": gid,
        })
        prev = {"ents": ents, "fr": fr}
    # 冰人证据链回填：死亡帧的**复算 edw 项**与该帧 residual
    # （residual 小 ⇒ 复算出的 edw 项确实是回放 reward 的组成部分）
    for ev in ice_events:
        df = ev.get("death_frame")
        if df is not None and 0 <= df < len(rows):
            ev["edw_term_at_death"] = rows[df]["edw_term"]
            ev["resid_core_at_death"] = rows[df]["resid_core"]
            ev["reward_at_death"] = rows[df]["reward"]
        d0 = ev.get("deploy_frame")
        if d0 is not None and 0 <= d0 < len(rows):
            ev["edw_term_at_deploy"] = rows[d0]["edw_term"]
    return {"rows": rows, "ice": ice_events, "unknown_cost": unknown_cost,
            "multi_card_with_new": multi_card_with_new, "meta": meta,
            "winner": winner, "decks": meta.get("decks") or [], "n_frames": len(frames)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", action="append", required=True,
                    help="回放目录或单个 pkl（可重复）")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-games", type=int, default=0, help="0 = 不限制（每文件）")
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--dump-outliers", type=int, default=0,
                    help="按 |resid| 打印最大的 N 帧（含出牌/新增/死亡上下文）")
    a = ap.parse_args()

    from forensics_card_usage import load_cost_map
    cost_map = load_cost_map(_SRC)

    all_rows, ice_all, per_file = [], [], []
    n_games = unknown_cost = multi_card = 0
    for path in a.replays:
        files = load_files(path)
        if a.max_files:
            files = files[:a.max_files]
        for name, schema, games in files:
            if a.max_games:
                games = games[:a.max_games]
            gm = 0
            for gi, g in enumerate(games):
                try:
                    res = analyse_game(g, schema, cost_map, gid=f"{name}#{gi}")
                except Exception as e:                              # noqa: BLE001
                    _stdout(f"[err] {path}/{name}: {type(e).__name__}: {e}")
                    continue
                if res is None:
                    continue
                gm += 1
                n_games += 1
                unknown_cost += res["unknown_cost"]
                multi_card += res["multi_card_with_new"]
                all_rows.extend(res["rows"])
                for ev in res["ice"]:
                    ev["file"] = name
                    ev["pair"] = "/".join(str(x) for x in (res["meta"].get("pair") or []))
                    ice_all.append(ev)
            per_file.append({"path": path, "file": name, "schema": schema, "games": gm})
            _stdout(f"  [{os.path.basename(path)}/{name}] schema={schema} games={gm}")

    import numpy as np

    def _stat(x, name):
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            return {"name": name, "n": 0}
        return {"name": name, "n": int(x.size),
                "mean": round(float(x.mean()), 6),
                "median": round(float(np.median(x)), 6),
                "p10": round(float(np.quantile(x, 0.10)), 6),
                "p90": round(float(np.quantile(x, 0.90)), 6),
                "min": round(float(x.min()), 6), "max": round(float(x.max()), 6),
                "mean_abs": round(float(np.abs(x).mean()), 6),
                "frac_abs_lt_1e-6": round(float((np.abs(x) < 1e-6).mean()), 6),
                "frac_abs_lt_0p01": round(float((np.abs(x) < 0.01).mean()), 6)}

    #: 口径：**末帧单列**。回放不记录 game_over 时刻 ⇒ 无法离线判定该帧 reward 里
    #: 有没有终局 ±10（实测两种都出现：走满 max_steps 时 winner 事后由 timeout_winner
    #: 判定 ⇒ 无终局项；引擎在步内 game_over ⇒ 有）。主对账只用**非末帧**。
    nonlast = [r for r in all_rows if not r["last"]]
    lastf = [r for r in all_rows if r["last"]]
    rc_nl = np.asarray([r["resid_core"] for r in nonlast], dtype=np.float64)
    rc_l = np.asarray([r["resid_core"] for r in lastf], dtype=np.float64)
    res_arr = np.asarray([r["resid"] for r in all_rows], dtype=np.float64)
    rew = np.asarray([r["reward"] for r in all_rows], dtype=np.float64)
    edw_all = np.asarray([r["edw_term"] for r in all_rows], dtype=np.float64)
    n = res_arr.size
    rs = _stat(res_arr, "resid")
    rnl = _stat(rc_nl, "resid_core_nonlast")
    rl = _stat(rc_l, "resid_core_last")

    _stdout(f"\n=== 对账：逐帧 reward vs 离线复算 | {n_games} 局 / {n} 帧 "
            f"| 费率表 {len(cost_map)} 张 | 未知费卡次={unknown_cost} "
            f"| 同帧多卡+新实体={multi_card} ===")
    _stdout("  recon_core = crown + tower(差异化定价/lv11锚) + edw*dPhi + unit_dmg")
    _stdout("  resid_core = reward - recon_core")
    _stdout(f"  非末帧 n={rnl['n']}：mean={rnl.get('mean')} median={rnl.get('median')} "
            f"mean_abs={rnl.get('mean_abs')} | 严格相等(<1e-6)={rnl.get('frac_abs_lt_1e-6')} "
            f"| <0.01={rnl.get('frac_abs_lt_0p01')} | min={rnl.get('min')} max={rnl.get('max')}")
    _stdout(f"  末帧   n={rl['n']}：mean={rl.get('mean')} min={rl.get('min')} max={rl.get('max')} "
            f"| 严格相等(<1e-6)={rl.get('frac_abs_lt_1e-6')} | <0.01={rl.get('frac_abs_lt_0p01')}")
    if rl["n"]:
        vals, cnts = np.unique(np.round(rc_l, 6), return_counts=True)
        top = sorted(zip(vals.tolist(), cnts.tolist()), key=lambda x: -x[1])[:6]
        _stdout(f"  末帧残差取值分布 top6 = {top}")
        _stdout("    （末帧的 reward **有时含、有时不含**终局 ±10："
                "`_run_side0` 走满 max_steps 时 winner 事后由 timeout_winner 判定，"
                "该帧 reward 里就没有终局项；引擎在步内 game_over 时才有。"
                "回放不记录 game_over 时刻 ⇒ 末帧无法离线判归属，主对账只用非末帧。）")
    # 残差来源分解①：把「本帧无塔伤/无单位受伤」的帧单独看 —— 这些帧的复算里
    # 不含任何**被回放取整到 0.1 的 hp** 进去，若它们严格相等 ⇒ 严格相等的缺口基本
    # 来自 hp 取整（回放 `replay.py:127-133` 对 position/hp 一律 round(...,1)）。
    def _split(rows_):
        nd = [r for r in rows_ if abs(r["tower"]) < 1e-12 and abs(r["unit"]) < 1e-12]
        d = [r for r in rows_ if not (abs(r["tower"]) < 1e-12 and abs(r["unit"]) < 1e-12)]
        return nd, d
    nd, d = _split(nonlast)
    for lab, grp in (("无塔伤/无单位伤", nd), ("有塔伤/有单位伤", d)):
        g = np.asarray([r["resid_core"] for r in grp], dtype=np.float64)
        _stdout(f"    分组[{lab}] n={g.size} mean_abs={float(np.abs(g).mean()) if g.size else None} "
                f"<1e-6={float((np.abs(g) < 1e-6).mean()) if g.size else None} "
                f"<0.01={float((np.abs(g) < 0.01).mean()) if g.size else None}")
    _stdout(f"  全帧（含末帧，recon 带 terminal） resid：mean={rs.get('mean')} "
            f"mean_abs={rs.get('mean_abs')} <0.01={rs.get('frac_abs_lt_0p01')}")
    _stdout(f"  reward.mean={float(rew.mean()):.6f} reward.std={float(rew.std()):.6f} "
            f"edw_term.mean={float(edw_all.mean()):.6f} "
            f"edw_term.std={float(edw_all.std()):.6f}")

    # 花费交叉核对（卡费 vs elixir 差分反解；差值 = 死亡馈赠 - 假出牌扣费）
    cx = np.asarray([r["spend0"] - r["spend_elix0"] for r in all_rows], dtype=np.float64)
    cxs = _stat(cx, "spend_cardcost_minus_elixir")
    _stdout(f"  花费交叉核对（全部帧）：cardcost - elixir反解 mean={cxs.get('mean')} "
            f"median={cxs.get('median')} mean_abs={cxs.get('mean_abs')} "
            f"<1e-6={cxs.get('frac_abs_lt_1e-6')} <0.01={cxs.get('frac_abs_lt_0p01')}")

    if a.dump_outliers and n:
        core_abs = np.asarray([abs(r['resid_core']) for r in all_rows], dtype=np.float64)
        order = np.argsort(-core_abs)[:a.dump_outliers]
        _stdout(f"\n--- |resid| 最大的 {len(order)} 帧（用于判断残差来源）---")
        for i in order:
            r = all_rows[int(i)]
            _stdout(f"  {r['_gid']} k={r['k']} t={r['t']:.2f} last={r['last']} "
                    f"reward={r['reward']:+.6f} recon_core={r['recon_core']:+.6f} "
                    f"resid_core={r['resid_core']:+.6f} | crown={r['crown']:+.4f} "
                    f"tower={r['tower']:+.6f} edw={r['edw_term']:+.6f} unit={r['unit']:+.6f} "
                    f"| dphi0={r['dphi0']:+.4f} dphi1={r['dphi1']:+.4f} "
                    f"cards0={r['cards0']} cards1={r['cards1']} new0={r['new0']} "
                    f"dead0={r['deadn0']} dead1={r['deadn1']} "
                    f"spend0={r['spend0']:.1f} spend_elix0={r['spend_elix0']:.4f}")

    # ---- 冰人实例 ----
    alive = [e for e in ice_all
             if e.get("death_frame") is not None and cost_map.get(e["name"]) is not None]
    for e in alive:
        cost = float(cost_map[e["name"]])
        e["cost"] = cost
        e["dphi_end_to_end"] = (-cost + e["share"]) + (-e["lost"])
        ew = 0.5 if float(e["death_t"]) < PHASE_SWITCH_S else 0.1
        e["edw_at_death"] = ew
        e["reward_contrib"] = ew * e["dphi_end_to_end"]
    nt = np.asarray([e["dphi_end_to_end"] for e in alive], dtype=np.float64)
    rc = np.asarray([e["reward_contrib"] for e in alive], dtype=np.float64)
    by_name = Counter(e["name"] for e in alive)
    _stdout(f"\n=== 冰人（IceGolemite / FreezeIceGolemite）| 出现 {len(ice_all)} 实例 "
            f"| 观测到死亡帧 {len(alive)} ===")
    if alive:
        pre = np.asarray([float(e["death_t"]) < PHASE_SWITCH_S for e in alive])
        _stdout(f"  死亡发生在 t<120（edw=0.5）的实例占比 = {float(pre.mean()):.4f} "
                f"({int(pre.sum())}/{len(alive)})")
        _stdout(f"  端到端 dPhi = (-cost + 部署份额) + (-死亡注销)："
                f"mean={float(nt.mean()):.6f} median={float(np.median(nt)):.6f} "
                f"min={float(nt.min()):.6f} max={float(nt.max()):.6f}")
        _stdout(f"  edw*dPhi：mean={float(rc.mean()):.6f} "
                f"median={float(np.median(rc)):.6f} "
                f"min={float(rc.min()):.6f} max={float(rc.max()):.6f}")
        _stdout(f"  恰好 = -2.0（= -cost）的占比 = "
                f"{float((np.abs(nt + 2.0) < 1e-9).mean()):.4f}")
        _stdout(f"  恰好 = -1.0 的占比 = {float((np.abs(rc + 1.0) < 1e-9).mean()):.4f} "
                f"| 恰好 = -0.2 的占比 = {float((np.abs(rc + 0.2) < 1e-9).mean()):.4f}")
    # 证据链：① 孤立归因（部署 +share、死亡 -lost）的 edw 敏感度 = edw(t_death)*(-lost)；
    #         ② 死亡帧的 resid_core（复算保真度 ⇒ 该敏感度确实是回放 reward 的一部分）
    ev_iso = np.asarray([e["edw_at_death"] * (-e["lost"]) for e in alive], dtype=np.float64)
    ev_res = np.asarray([e.get("resid_core_at_death", np.nan) for e in alive], dtype=np.float64)
    ev_res = ev_res[~np.isnan(ev_res)]
    if ev_iso.size:
        _stdout(f"  证据链①孤立归因 edw(t_death)*(-lost)：mean={float(ev_iso.mean()):.6f} "
                f"median={float(np.median(ev_iso)):.6f} min={float(ev_iso.min()):.6f} "
                f"max={float(ev_iso.max()):.6f} | = -1.0 占比="
                f"{float((np.abs(ev_iso + 1.0) < 1e-9).mean()):.4f} "
                f"= -0.2 占比={float((np.abs(ev_iso + 0.2) < 1e-9).mean()):.4f}")
        _stdout(f"  证据链②死亡帧 resid_core：mean_abs={float(np.abs(ev_res).mean()):.6f} "
                f"<1e-6={float((np.abs(ev_res) < 1e-6).mean()):.4f} "
                f"<0.01={float((np.abs(ev_res) < 0.01).mean()):.4f}（n={ev_res.size}）")
    _stdout(f"  按卡名：{dict(by_name)}")

    out = {
        "replays": a.replays, "cost_map_size": len(cost_map),
        "n_games": n_games, "n_frames": n, "per_file": per_file,
        "unknown_cost_card_plays": unknown_cost,
        "multi_card_frames_with_new_entities": multi_card,
        "recon_terms": ["crown", "tower(per-tower premium + lv11 anchor)",
                        "edw*dPhi(ledger)", "unit_dmg_k"],
        "resid_all_with_terminal": rs,
        "resid_core_nonlast": rnl,
        "resid_core_last": rl,
        "spend_crosscheck": cxs,
        "reward": _stat(rew, "reward"), "edw_term": _stat(edw_all, "edw_term"),
        "ice_golem": {
            "n_seen": len(ice_all), "n_with_death_frame": len(alive),
            "names": dict(by_name),
            "dphi_end_to_end": _stat(nt, "dphi_e2e"),
            "reward_contrib": _stat(rc, "edw*dphi"),
            "frac_dphi_eq_minus2": (round(float((np.abs(nt + 2.0) < 1e-9).mean()), 6)
                                    if alive else None),
            "frac_reward_eq_minus1": (round(float((np.abs(rc + 1.0) < 1e-9).mean()), 6)
                                      if alive else None),
            "examples": [{kk: e.get(kk) for kk in
                          ("file", "pair", "eid", "pid", "name", "deploy_frame", "deploy_t",
                           "death_frame", "death_t", "share", "lost", "cost",
                           "dphi_end_to_end", "edw_at_death", "reward_contrib",
                           "edw_term_at_death", "resid_core_at_death", "reward_at_death")}
                         for e in alive[:8]],
        },
    }
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        _stdout(f"\n[out] {a.out}")
    return out


if __name__ == "__main__":
    main()
