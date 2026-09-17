# -*- coding: utf-8 -*-
"""离线「局面圣水交换」仪器（S2，零训练成本，只读回放）。

规格来源（本文件每个口径都指向它们，改口径必须同步改这两份）：
  * ``docs/frame_credit_proposal_review_2026-09-18.md`` §7（缺口确证 / 修订判决）、
    §8（局面划分规格 v1）、§9（溯源路由 / 货币泵否决）
  * ``docs/engagement_trade_prereg_2026-09-18.md`` §1（定义）、§2（指标）、§3（溯源）、
    §4（两条否决证明）、§7（不变量）

**本文件不 import 引擎、不跑训练、不写任何既有文件。** 它读的是录像
（``rl/replay.py::save_league_replays`` 的容器：``{"schema": s, "games": [{meta,winner,frames}, ...]}``）。

────────────────────────────────────────────────────────────────────────────
§0 口径登记（每条都可脚本复算；不变量测试见 ``scripts/selftest_offline_engagement_trade.py``）
────────────────────────────────────────────────────────────────────────────
* **索敌关系图** 每一帧用 ``(a, b)`` 建无向边 ⟺ ``a.target_id == b.id`` 或 ``b.target_id == a.id``
  （schema ≥ 4 才有 ``target_id``；见预注册 §11.1）。局面 = 该图的**连通分量**（并查集）。
* **交战对** 边 ``(a,b)`` 满足 ①``a.player != b.player``（跨边）②**至少一侧非产物**（规则①）。
  ⚠️ schema 4 没有 ``is_product`` ⇒ 规则①**退化为恒真**（全部当非产物）。该退化在
  文档 §11.5 登记；只有 schema ≥ 5 才是规格的忠实实现。
* **结算** 分量**连续 ``settle_frames`` 个采样帧无交战对** ⇒ 结算；分量**分裂/合并/清空**
  也立即结算。``K = 30`` engine tick = 1 决策帧 ⇒ 回放分辨率下 ``settle_frames = 1``
  （规格见 §1；``settle_frames * 30 == settle_ticks`` 由测试断言）。
* **窗口基线** 窗口 ``[t0, t1]``（帧下标）的 Φ 基线取 **``t0-1`` 帧的边界状态**
  （= 该窗口第一次动作**之前**）。这是 §2.1 两种形式（ΔΦ 形式 与 净支出形式）等价的
  必要条件，见 ``test_trade_equals_phi_window``。
* **净交换** ``Trade_me = ΔΦ_me - ΔΦ_op + tower_term_me``，
  ``ΔΦ_x = Φ_x(t1) - Φ_x(t0-1)``，``Φ_x = 手牌圣水 + 残值 V_x``。
* **残值 V** 权威定义在 ``rl/env_wrapper.py::RLEnv._deploy_ledger`` / ``_collect_deaths``
  （``_v_share`` / ``_active_v``）。schema ≥ 5 的录像**直接带**在线真值 ⇒ 离线零误差；
  schema 4 只能**重建**（`reconstruct_v`），误差来源已在 §11.5 登记。
* **产物体恒 0 费**（§4 否决证明）：任何产物体出现/死亡都不得改变 V。
* **`tower_term` 的零和性**：预注册 §7-1 要求 ``trade_p0 == -trade_p1``。只有当
  ``tower_term`` 是**零和转移**（``+τ`` / ``−τ``）时才成立；若两侧各按自己视角取正
  （``per-side``），该不变量**不成立**。仪器两种口径都算，并报出差异 ⇒ 见文档 §11.5。
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import pickle
import sys

# ---------------------------------------------------------------------------
# 卡费 / 卡型：**单一来源** = 引擎自己的 card_utils.Card（【R7】不许另建一张表）
# ---------------------------------------------------------------------------
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO, "src", "clasher_new")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

_CARD_CACHE: dict = {}


def card_info(name):
    """{cost, type, dps}；查不到 ⇒ 全 None（产物名如 FireSpirit 不在卡表里）。"""
    if name in _CARD_CACHE:
        return _CARD_CACHE[name]
    info = {"cost": None, "type": None, "dps": None}
    try:
        from card_utils import Card
        c = Card(name)
        info["cost"] = float(c.elixir)
        info["type"] = str(c.type)
        hs = float(c.hit_speed)
        if hs > 0:
            info["dps"] = float(c.damage) / hs
    except Exception:
        pass
    _CARD_CACHE[name] = info
    return info


def card_cost(name):
    """卡费；未知 ⇒ 0.0（产物体按 0 费处理，与 §4 一致）。"""
    v = card_info(name)["cost"]
    return 0.0 if v is None else v


def card_is_spell(name):
    return card_info(name)["type"] == "spell"


def card_dps(name):
    return card_info(name)["dps"]


#: 塔的实体 id（`battle.py::BattleState.__init__` 硬编码 1..6，`update_player_hp` 同源）
TOWER_IDS = frozenset({1, 2, 3, 4, 5, 6})
#: 塔 id -> 归属玩家（1,2,5 = player 1；3,4,6 = player 0）
TOWER_OWNER = {1: 1, 2: 1, 5: 1, 3: 0, 4: 0, 6: 0}

TICK_PER_FRAME = 30          # 1 决策帧 = 30 engine tick（0.5 s）
SETTLE_TICKS = 30            # K（§1，登记为「有原则的选择」，非标定阈值）
THETA = 1.0                  # θ 圣水（§2.3，登记为「有原则的选择」）
#: §5.1 门禁的 N = 20 s：**必须登记为「有原则的选择」，不是标定值**（【R15】【R16】）。
#: 回放分辨率 = 1 决策帧/0.5 s ⇒ 20 s = 40 帧。
REALIZE_N_FRAMES = 40

#: CLI 用：本批原始局（--sweep 需要在同一批局上反复换口径）
_CACHE_GAMES: list = []


# ---------------------------------------------------------------------------
# §1 回放读取
# ---------------------------------------------------------------------------
def load_replay(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "games" in data:
        return int(data.get("schema", 0)), list(data["games"])
    raise ValueError("无法识别的录像容器: %s" % path)


def entity_table(frame):
    """帧 -> {id: {name,x,y,hp,player,kind,max_hp,id,target_id,root_cast,is_product,share}}。

    字段下标（schema 4 = 12 元组 / schema 5 = 15 元组，见 `rl/replay.py::battle_snapshot`）：
    ``[name,x,y,hp,player,kind,max_hp,shield,shield_max,radius,id,target_id,root_cast,is_product,share]``
    """
    out = {}
    for e in frame.get("entities", ()):
        if len(e) < 11 or e[10] is None:
            continue
        eid = int(e[10])
        out[eid] = {
            "name": e[0], "x": e[1], "y": e[2], "hp": e[3],
            "player": int(e[4]), "kind": e[5], "max_hp": e[6],
            "id": eid,
            "target_id": (int(e[11]) if len(e) > 11 and e[11] is not None else None),
            "root_cast": (int(e[12]) if len(e) > 12 and e[12] is not None else None),
            "is_product": (bool(e[13]) if len(e) > 13 and e[13] is not None else None),
            "share": (float(e[14]) if len(e) > 14 and e[14] is not None else None),
        }
    return out


def replay_entity_arity(games):
    """抽样出实体元组长度分布（判断 schema 与字段可用性）。"""
    c = collections.Counter()
    for g in games[:8]:
        for f in (g.get("frames") or [])[:25]:
            for e in f.get("entities", ()):
                c[len(e)] += 1
    return dict(c)


# ---------------------------------------------------------------------------
# §1 索敌关系图 -> 连通分量 -> 交战对
# ---------------------------------------------------------------------------
def components(ents):
    """索敌关系图的连通分量（并查集；用实体 id 为键）。返回 [frozenset, ...]，确定性排序。"""
    parent = {i: i for i in ents}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, d in ents.items():
        t = d["target_id"]
        if t is not None and t in ents:
            ra, rb = find(i), find(t)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    groups = collections.defaultdict(set)
    for i in ents:
        groups[find(i)].add(i)
    return sorted((frozenset(v) for v in groups.values()), key=lambda c: min(c))


def cross_edges(comp, ents):
    """分量内的**跨边**（不论是否产物）。返回确定性排序的 ``[(a,b), ...]``，a<b。"""
    out = set()
    for i in comp:
        t = ents[i]["target_id"]
        if t is None or t not in comp:
            continue
        if ents[i]["player"] == ents[t]["player"]:
            continue
        out.add((i, t) if i < t else (t, i))
    return sorted(out)


def engagement_edges(comp, ents, rule1=True):
    """分量内的**交战对** = 跨边 ∧（rule1 时）至少一侧非产物（规格 §1 规则①）。"""
    out = []
    for a, b in cross_edges(comp, ents):
        if rule1 and ents[a]["is_product"] is True and ents[b]["is_product"] is True:
            continue
        out.append((a, b))
    return out


# ---------------------------------------------------------------------------
# §1 局面分段（去抖 + 分裂/合并/清空结算）
# ---------------------------------------------------------------------------
def segment_windows(frames, settle_frames=1, rule1=True, identity="core"):
    """把一局的帧序列切成局面窗口。

    ★ **身份判据**（``identity``）—— 本仪器在实现中实测到的一个规格缺口，见文档 §11.5：

    - ``members``：按**全部成员**重叠续接（= 规格字面读法）。**实测不可用**：
      塔（id ≤ 6）恒在场上 ⇒ 任何"同一座塔被不同单位先后攻击"都会续接成**一个**
      长达数百帧的窗口（实测出现 span=247 帧的"局面"）。
    - ``core``（默认）：按**交战核心**（= 参与至少一个交战对的**非塔**实体）重叠续接。
      塔仍计入 ``members_union``（信用归属要用，§7.5），但**不参与身份判定**。
      ★ 窗口的 ``key`` **只在活跃帧更新**（= 最后一次活跃时的核心），于是"目标瞬态丢失
      一两帧"不会立刻把窗口掐断 ⇒ 去抖 ``settle_frames`` 才有意义（若每帧都重设 key，
      不活跃帧的核心为空 ⇒ key 为空 ⇒ 窗口立刻结算 ⇒ ``K`` 是死代码 —— 初版实测
      ``no_engagement_K`` 结算占比 **0/2174**，就是这个 bug）。

    - 分量**连续 ``settle_frames`` 个采样帧无交战对** ⇒ 结算（``no_engagement_K``）；
    - 分量**分裂/合并**（一个窗口被多个分量争用，或一个分量跨多个窗口）⇒ 在**最后一个
      共同帧 ``t-1``** 结算（``merge`` / ``split``）；
    - 分量**清空**（无分量可续接）⇒ 结算（``vanish``）；
    - 局末仍在开着的窗口 ⇒ 结算（``episode_end``）。

    返回 [dict]：``rep/t0/t1/last_active/reason/members_union/members_final/core_union``。
    ``t1`` = 结算**发生的**那一帧（首次观测到"够久没有交战对"的那一帧）。
    """
    windows = []
    open_w = {}          # rep -> window dict

    def _core(comp, ents, active):
        if not active:
            return frozenset()
        if identity == "members":
            return frozenset(comp)
        c = set()
        for a, b in engagement_edges(comp, ents, rule1=rule1):
            if a > 6:
                c.add(a)
            if b > 6:
                c.add(b)
        return frozenset(c)

    def _key(comp, core, active):
        """分量用于**匹配**的键：活跃 ⇒ 交战核心；不活跃 ⇒ 全成员（去抖窗内续接用）。"""
        if identity == "members":
            return set(comp)
        return set(core) if active else set(comp)

    for t, frame in enumerate(frames):
        ents = entity_table(frame)
        comps = components(ents)
        act = [bool(engagement_edges(c, ents, rule1=rule1)) for c in comps]
        cores = [_core(c, ents, act[i]) for i, c in enumerate(comps)]
        keys = [_key(c, cores[i], act[i]) for i, c in enumerate(comps)]

        # ---- 匹配（**按窗口**判定，不按分量）----
        # 为什么不能按分量：交战断开的那一帧，原分量会碎成若干小分量（8 与 10 各自单独），
        # 若按分量逐个匹配，"一个窗口被 2 个分量重叠"就会被误判成 merge 而立刻结算
        # ⇒ 去抖 K 失效（实测：K=2 的场景被结算在 t1=0）。
        hits = {}                      # rep -> {"active": [ci], "all": [ci]}
        for ci in range(len(comps)):
            if not keys[ci]:
                continue
            for rep, w in open_w.items():
                if w["key"] & keys[ci]:
                    h = hits.setdefault(rep, {"active": [], "all": []})
                    h["all"].append(ci)
                    if act[ci]:
                        h["active"].append(ci)
        # 合并（merge）：≥2 个**活跃**分量争用同一窗口 ⇒ 在最后一个共同帧结算
        contested = {rep for rep, h in hits.items() if len(h["active"]) > 1}
        # 分裂（split）：一个活跃分量同时重叠 ≥2 个窗口
        adeg = collections.Counter(ci for rep, h in hits.items()
                                  if rep not in contested for ci in h["active"])
        spanning = {ci for ci, d in adeg.items() if d > 1}

        new_open, closed = {}, set()
        for rep, w in sorted(open_w.items(), key=lambda kv: repr(kv[0])):
            if rep in contested:
                windows.append(settle_window(w, max(w["t0"], t - 1), "merge"))
                closed.add(rep)
                continue
            h = hits.get(rep)
            if h is None:
                windows.append(settle_window(w, t, "split_or_vanish"))
                closed.add(rep)
                continue
            for ci in h["all"]:
                w["members_union"] |= set(comps[ci])
                w["members_final"] = set(comps[ci])
                w["core_union"] |= set(cores[ci])
            if h["active"]:
                best = max(h["active"], key=lambda ci: (len(w["key"] & keys[ci]), -ci))
                if identity == "members":
                    for ci in h["all"]:
                        w["key"] |= set(comps[ci])
                else:
                    w["key"] = set(cores[best])
                w["last_active"] = t
                w["inactive_run"] = 0
            elif identity == "members":
                for ci in h["all"]:
                    w["key"] |= set(comps[ci])
            if not h["active"]:
                w["inactive_run"] += 1
            if w["inactive_run"] >= settle_frames:
                windows.append(settle_window(w, t, "no_engagement_K"))
                closed.add(rep)
            else:
                new_open[rep] = w

        claimed = set(ci for rep, w in new_open.items()
                      for ci in hits.get(rep, {}).get("all", []))
        for ci, c in enumerate(comps):
            if not act[ci] or ci in claimed or ci in spanning:
                continue
            rep = ("new", t, ci)
            new_open[rep] = {"rep": rep, "t0": t, "t1": None, "last_active": t,
                             "inactive_run": 0, "members_union": set(c),
                             "members_final": set(c), "core_union": set(cores[ci]),
                             "key": set(cores[ci]) if identity != "members" else set(c)}
        open_w = new_open
    last = len(frames) - 1
    for _rep, w in sorted(open_w.items(), key=lambda kv: repr(kv[0])):
        windows.append(settle_window(w, last, "episode_end"))
    windows.sort(key=lambda w: (w["t0"], w["t1"]))
    return windows


def settle_window(w, t, reason):
    out = dict(w)
    out["t1"] = max(int(t), int(w["t0"]))
    out["reason"] = reason
    out["members_union"] = set(w["members_union"])
    out["members_final"] = set(w.get("members_final") or w["members_union"])
    out["core_union"] = set(w.get("core_union") or w.get("key") or set())
    return out


# ---------------------------------------------------------------------------
# §2 残值 V 与 Φ
# ---------------------------------------------------------------------------
def plays_of_frame(frame):
    """本帧双方实际出的牌 -> [(player, card_name), ...]（顺序：先 0 后 1）。

    player 0：``frame["cards"]``（``rl/run_league.py::_bundle_cards`` 由手牌解析出的卡名）。
    player 1：``frame["opp_played"]``（``env_wrapper._deploy_opponent`` 的 ``[{"card":...}]``）。
    """
    out = []
    for c in (frame.get("cards") or ()):
        out.append((0, c))
    for d in (frame.get("opp_played") or ()):
        if isinstance(d, dict) and d.get("card"):
            out.append((1, d["card"]))
        elif isinstance(d, str):
            out.append((1, d))
    return out


def estimate_regen(frames):
    """从「双方都没出牌、且不在圣水上限」的帧里估圣水自然回复速率（每帧，中位数）。"""
    deltas = []
    for a, b in zip(frames, frames[1:]):
        dt = float(b["t"]) - float(a["t"])
        if dt <= 0:
            continue
        if plays_of_frame(a) or plays_of_frame(b):
            continue
        if any(float(a[k]) >= 9.999 or float(b[k]) >= 9.999 for k in ("elixir0", "elixir1")):
            continue
        for k in ("elixir0", "elixir1"):
            d = float(b[k]) - float(a[k])
            if d >= -1e-9:
                deltas.append(d / dt)
    if not deltas:
        return 2.8 * 0.5, 0            # 引擎 BattleState.regen=2.8/s，帧=0.5s
    deltas.sort()
    return deltas[len(deltas) // 2], len(deltas)


def reconstruct_v(frames, regen_per_frame=None, recon="costname"):
    """重建在线 ``_active_v``（schema 4 的**近似**；schema ≥ 5 请用帧内真值）。

    与 ``RLEnv._deploy_ledger`` 同规则（**同源**）：
      - ``Card(c).type == "spell" and c != "Mirror"`` ⇒ 不入账（法术成本走 E 账、产物免费）；
      - 否则把该卡实际花费 ``cost`` 摊给「本帧新增的 ``troop|building`` 实体」，
        ``share = cost / len(new_ids)``；实体死亡即注销其 share。

    ``recon``：
      - ``naive``：新增实体**全收**（含帧内延迟出兵 / 建筑产物 ⇒ 会被污染，是误差上界）；
      - ``costname``（默认）：只收**卡表里有正费用**的新增实体（滤掉 0 费产物名），
        过滤后为空则退回 naive 并计数。

    返回 ``(v, diag)``；``diag["share_map"][pid][eid]`` **永久保留**每个实体的部署份额
    （死亡也不删）——它同时是 P3/P2 的费用来源与组件口径的残值来源。
    """
    if regen_per_frame is None:
        regen_per_frame = estimate_regen(frames)[0]
    v_share = {0: {}, 1: {}}
    share_map = {0: {}, 1: {}}
    deploy_cost = {0: {}, 1: {}}
    active = [0.0, 0.0]
    v, deploys = [], {}
    prev_ids, prev_elixir, prev_t = set(), [None, None], None
    diag = {"recon": recon, "regen_per_frame": regen_per_frame, "v_source": "reconstruct",
            "exact": False, "filtered_names": collections.Counter(), "fallback": 0,
            "cost_residual": [], "deploy_frames": 0, "at_cap": 0}
    for t, frame in enumerate(frames):
        ents = entity_table(frame)
        ids = set(ents)
        new_ids = ids - prev_ids
        now_elixir = [float(frame["elixir0"]), float(frame["elixir1"])]
        dt = (float(frame["t"]) - prev_t) if prev_t is not None else 0.5
        plays = plays_of_frame(frame)
        if plays:
            diag["deploy_frames"] += 1
            per_player = collections.defaultdict(list)
            for pid, name in plays:
                per_player[pid].append(name)
            for pid, names in sorted(per_player.items()):
                if prev_elixir[pid] is None:
                    cost_total = 0.0
                else:
                    cost_total = prev_elixir[pid] + regen_per_frame * dt - now_elixir[pid]
                    if prev_elixir[pid] >= 9.999:
                        diag["at_cap"] += 1
                    cost_total = max(0.0, cost_total)
                tw = [card_cost(n) for n in names]
                if sum(tw) <= 1e-9:
                    tw = [1.0] * len(names)
                tot_w = sum(tw)
                for j, name in enumerate(names):
                    cost = cost_total * (tw[j] / tot_w)
                    # 全部出牌（含法术）都计入实际花费（对账用；法术只是不入 V 账）
                    deploy_cost[pid][t] = deploy_cost[pid].get(t, 0.0) + cost
                    if card_is_spell(name) and name != "Mirror":
                        continue
                    cands = [eid for eid in new_ids
                             if eid > 6 and ents[eid]["kind"] in ("troop", "building")]
                    if recon == "costname":
                        keep = [eid for eid in cands if card_cost(ents[eid]["name"]) > 0.0]
                        for eid in cands:
                            if eid not in keep:
                                diag["filtered_names"][ents[eid]["name"]] += 1
                        if keep:
                            cands = keep
                        elif cands:
                            diag["fallback"] += 1
                    if not cands or cost <= 1e-9:
                        continue
                    share = cost / len(cands)
                    for eid in cands:
                        v_share[pid][eid] = v_share[pid].get(eid, 0.0) + share
                        share_map[pid][eid] = share_map[pid].get(eid, 0.0) + share
                        active[pid] += share
                    deploys.setdefault(t, []).append(
                        {"player": pid, "card": name, "eids": list(cands),
                         "cost": cost, "cost_elixir_total": cost_total})
                    diag["cost_residual"].append(cost_total - sum(card_cost(n) for n in names))
        for pid in (0, 1):
            for eid in [e for e in v_share[pid] if e not in ids]:
                active[pid] -= v_share[pid].pop(eid)
        v.append([active[0], active[1]])
        prev_ids, prev_elixir, prev_t = ids, now_elixir, float(frame["t"])
    diag["share_map"] = share_map
    diag["deploy_cost"] = deploy_cost
    diag["v"] = v
    diag["deploys"] = deploys
    return v, diag


def phi_series(frames, recon="costname", regen_per_frame=None):
    """每帧 ``(Φ_0, Φ_1)`` + V 来源与费用环。

    schema ≥ 5 且帧内带 ``v0``/``v1`` ⇒ **直接取在线真值**（``exact=True``，费用环取帧内
    ``share``）；否则用 ``reconstruct_v``（``exact=False``，近似）。
    """
    have_v = bool(frames) and all(("v0" in f and "v1" in f) for f in frames[:5])
    if have_v:
        cost_ring, share_map = {}, {0: {}, 1: {}}
        for f in frames:
            for e in f.get("entities", ()):
                if len(e) > 14 and e[14] is not None:
                    eid, sh = int(e[10]), float(e[14])
                    cost_ring[eid] = sh
                    share_map[int(e[4])][eid] = sh
        diag = {"v_source": "frame", "exact": True, "cost_ring": cost_ring,
                "share_map": share_map, "deploys": {},
                "v": [[float(f["v0"]), float(f["v1"])] for f in frames]}
        phi = [[float(f["elixir0"]) + float(f["v0"]),
                float(f["elixir1"]) + float(f["v1"])] for f in frames]
        # ★ schema 5 才有这步：用**重建**去对账**帧内真值** ⇒ 直接验证 schema 4 近似法
        _v, rdiag = reconstruct_v(frames, regen_per_frame=regen_per_frame, recon=recon)
        err = max(abs(_v[t][p] - diag["v"][t][p])
                  for t in range(len(frames)) for p in (0, 1))
        diag["reconstruct_vs_frame_v_max_err"] = err
        diag["deploy_cost"] = rdiag["deploy_cost"]
        diag["reconstruct_diag"] = {"filtered_names": dict(rdiag["filtered_names"]),
                                    "fallback": rdiag["fallback"],
                                    "regen_per_frame": rdiag["regen_per_frame"]}
        return phi, diag
    v, diag = reconstruct_v(frames, regen_per_frame=regen_per_frame, recon=recon)
    cost_ring = {}
    cost_ring.update(diag["share_map"][0])
    cost_ring.update(diag["share_map"][1])
    diag["cost_ring"] = cost_ring
    phi = [[float(f["elixir0"]) + v[t][0], float(f["elixir1"]) + v[t][1]]
           for t, f in enumerate(frames)]
    return phi, diag


# ---------------------------------------------------------------------------
# §2.2 tower_term 三个代理（P3 / P1 / P2）
# ---------------------------------------------------------------------------
def _frame_meta(frames):
    """预扫：``first_seen[eid]/name_of[eid]/player_of[eid]/tables[t]``。"""
    first_seen, name_of, player_of, tables = {}, {}, {}, []
    for t, f in enumerate(frames):
        tab = entity_table(f)
        tables.append(tab)
        for eid, d in tab.items():
            if eid not in first_seen:
                first_seen[eid] = t
                name_of[eid] = d["name"]
                player_of[eid] = d["player"]
    return first_seen, name_of, player_of, tables


def _partner_map(w, tables):
    """窗口内 ``{eid: {t: [对手 eid, ...]}}``（跨边，不看规则①）。"""
    partner = collections.defaultdict(lambda: collections.defaultdict(list))
    for t in range(max(0, w["t0"] - 1), min(w["t1"], len(tables) - 1) + 1):
        tab = tables[t]
        for a, b in cross_edges(set(tab), tab):
            partner[a][t].append(b)
            partner[b][t].append(a)
    return partner


def tower_term(w, tables, partner, *, me=0, mode="p3a", t_ref=2.0, cost_of=None,
               frames=None):
    """按 §2.2 从 ``me`` 视角算 τ。

    - ``p3a``：``Σ cost(E)``，E 在「最后一次存活采样帧」处于跨边（**任意**对手，含塔）；
    - ``p3b``：同 p3a，但要求对手是**非塔**实体（``id > 6``）⇒ 排除「塔自己打死」的情形；
    - ``p1`` ：``Σ dps(E) × 被钉住秒数``（dps 由引擎 ``damage/hit_speed`` 算）；
    - ``p2`` ：``Σ cost(E) × min(1, 被钉住秒数 / T_ref)``（``T_ref`` **待标定**）；
    - ``p4`` ：**塔血门控版 P2**（本仪器新加的首选候选）：只有当**我方三塔在窗口内零掉血**
      时才给 ``Σ_E cost(E) × min(1, pinned_unit_s(E) / T_ref)``，否则 0。
      ⇒ 有界（≤ Σ cost(E)、与 ΔΦ 同量纲）、**不可刷**（不真挡住塔伤就拿不到）、且
      **不是已有 Φ 的重新开窗**（塔血不在 Φ 里）。
    - ``p4b``：同 p4，但按**单位次数**计（``Σ_E min(1, pinned_unit_s/T_ref)``，G=1 圣水/单位）。
    - ``p5`` ★：**「塔的击杀不算我的交换」**（本仪器从 p3b 的实测里反推出来的候选）：
      ``τ = -Σ cost(E)``，E 死亡时的交战伙伴**全是塔**（``id <= 6``）⇒ 扣掉这部分
      已有 Φ 的击杀信用。动机：现行 ΔΦ 对「不防守、让塔自己打死 Hog」也记 +4 费，
      于是**塔扛伤反而比拉扯更赚**；扣掉它，拉扯（花 2 费、塔不挨打）才严格更优。
      有界（|τ| ≤ Σ cost）、同量纲、**不可刷**（负项刷不出来）、**不冗余**（是 p3b 的补集）。
    - ``p5b``：同 p5，但只在"我方该窗口**零塔损**"时才扣（更保守）。
    - ``none``：恒 0。

    ``cost_of(eid, name) -> float`` 可注入（schema ≥ 5 用帧内 ``share``）。
    """
    if cost_of is None:
        def cost_of(eid, name):
            return card_cost(name)
    t0, t1 = w["t0"], w["t1"]
    # ★ 只统计**本局面成员**的账（否则同一个死亡会被所有重叠窗口各记一次 ⇒ 重复计数，
    #   与 phi_mode="global" 的问题是同一类）。成员集合 = 窗口生命周期内出现过的
    #   连通分量成员（含塔）。
    mem = w.get("members_union") or set()
    tau = 0.0
    deaths, kill_credit = [], 0.0
    pinned_frames = collections.Counter()        # 任意跨边（含塔）
    pinned_unit_frames = collections.Counter()   # ★ 被**非塔**实体钉住（"拉扯"口径）
    for t in range(t0, t1 + 1):
        tab = tables[t] if t < len(tables) else {}
        for eid, d in tab.items():
            if d["player"] == me or eid not in mem:
                continue
            opps = list(partner.get(eid, {}).get(t, ())) + \
                list(partner.get(eid, {}).get(t - 1, ()))
            if opps:
                pinned_frames[eid] += 1
            if any(o > 6 for o in opps):
                pinned_unit_frames[eid] += 1
    prev_tab = tables[max(0, t0 - 1)] if tables else {}
    for t in range(t0, min(t1, len(tables) - 1) + 1):
        tab = tables[t]
        for eid, d in prev_tab.items():
            if d["player"] == me or eid in tab or eid not in mem:
                continue
            opp = list(partner.get(eid, {}).get(t - 1, ()))
            cost = cost_of(eid, d["name"])
            deaths.append({"id": eid, "name": d["name"], "cost": cost, "t": t,
                           "opponents": opp})
            kill_credit += cost
            if mode == "p3a" and opp:
                tau += cost
            elif mode == "p3b" and any(o > 6 for o in opp):
                tau += cost
            elif mode == "p5" and opp and all(o <= 6 for o in opp):
                tau -= cost
        prev_tab = tab
    # 采样间隔 = 1 决策帧 = 0.5 s（录像本身就是决策帧粒度；窗口长短由采样帧数决定
    # ⇒ 一律用帧数 × 0.5 s，已在 §0 登记）
    dt_med = TICK_PER_FRAME / 60.0
    if mode == "p1":
        for eid, cnt in pinned_frames.items():
            tau += (card_dps(_name_of(tables, eid)) or 0.0) * cnt * dt_med
    elif mode == "p2":
        for eid, cnt in pinned_frames.items():
            tau += cost_of(eid, _name_of(tables, eid)) * min(1.0, (cnt * dt_med) / t_ref)
    elif mode == "p5b":
        gate = 1.0
        if frames is not None and frames:
            k = "towers%d" % me
            tb, t1c = max(0, w["t0"] - 1), min(w["t1"], len(frames) - 1)
            loss = sum(float(frames[tb][k][i]) for i in range(3)) - \
                sum(float(frames[t1c][k][i]) for i in range(3))
            gate = 1.0 if loss <= 1e-6 else 0.0
        if gate:
            for d in deaths:
                if d["opponents"] and all(o <= 6 for o in d["opponents"]):
                    tau -= d["cost"]
    elif mode in ("p4", "p4b"):
        # 塔血门控：我方三塔在窗口内必须**零掉血**（塔血是 Φ 里没有的量 ⇒ 真新信息）
        gate = 1.0
        if frames is not None and frames:
            k = "towers%d" % me
            tb, t1c = max(0, w["t0"] - 1), min(w["t1"], len(frames) - 1)
            loss = sum(float(frames[tb][k][i]) for i in range(3)) - \
                sum(float(frames[t1c][k][i]) for i in range(3))
            gate = 1.0 if loss <= 1e-6 else 0.0
        for eid, cnt in pinned_unit_frames.items():
            frac = min(1.0, (cnt * dt_med) / t_ref)
            if mode == "p4":
                tau += gate * cost_of(eid, _name_of(tables, eid)) * frac
            else:
                tau += gate * frac
    return {"tau": tau, "mode": mode, "kill_credit": kill_credit, "deaths": deaths,
            "pinned_frames": dict(pinned_frames),
            "pinned_unit_frames": dict(pinned_unit_frames),
            "n_pinned": len(pinned_frames), "n_pinned_unit": len(pinned_unit_frames),
            "pinned_and_killed": sum(1 for d in deaths if d["opponents"]),
            "dt": dt_med}


def _name_of(tables, eid):
    for tab in tables:
        if eid in tab:
            return tab[eid]["name"]
    return ""


# ---------------------------------------------------------------------------
# §2 一个窗口的 Trade
# ---------------------------------------------------------------------------
def member_v(tab, members, cost_ring):
    """窗口成员在当前帧的残值 ``[V_0, V_1]``（产物体份额恒 0 ⇒ 与 §4 一致）。"""
    out = [0.0, 0.0]
    for eid in members:
        d = tab.get(eid)
        if d is None:
            continue
        sh = d.get("share")
        if sh is None:
            sh = cost_ring.get(eid)
        if sh is None:
            sh = card_cost(d["name"])
        out[d["player"]] += float(sh)
    return out


def window_trade(w, frames, phi, *, tables=None, cost_ring=None, first_seen=None,
                 name_of=None, player_of=None, deploy_cost=None, tower_mode="p3a",
                 tower_transfer="zero-sum", phi_mode="component", t_ref=2.0,
                 rule1=True, me=0):
    """算一个窗口的 ``Trade_p0`` / ``Trade_p1``（及 §7-4 的对账残差）。

    ``phi_mode``：
      - ``global``：``ΔΦ`` 用**整块场地**（预注册 §2.1 的字面读法；双路同时开战时
        多个窗口会重叠 ⇒ **同一次账会被重复计入多个窗口**）；
      - ``component``（默认）：``ΔΦ`` 只统计窗口 ``members_union`` 成员（+ 其部署事件）⇒ 不重复。

    ``tower_transfer``：
      - ``zero-sum``（默认）：``tower_term = [+τ, -τ]`` ⇒ 满足预注册 §7-1；
      - ``per-side``：两侧各按自己视角取正（``τ_p1`` 由翻面重算）⇒ **会让 §7-1 不成立**，
        仅作对照臂（差异会打印出来）。

    ``identity_err_elx``（仅 ``phi_mode='global'`` 有意义）：把 **净支出**（由原始圣水差
    独立算出）代进 §2.1 的等价形式，与 ΔΦ 形式对账 ⇒ ``test_trade_equals_phi_window``
    用的就是这个量；它同时验证「出牌检测 + regen 估计」自洽。
    """
    cost_ring = cost_ring or {}
    deploy_cost = deploy_cost or {0: {}, 1: {}}
    if tables is None:
        tables = [entity_table(f) for f in frames]
    partner = _partner_map(w, tables)

    def _cost(eid, name):
        sh = cost_ring.get(eid)
        return card_cost(name) if sh is None else float(sh)

    t0, t1 = w["t0"], w["t1"]
    tb = max(0, t0 - 1)
    members = w["members_union"]
    mv_b = member_v(tables[tb], members, cost_ring)
    mv_t = member_v(tables[t1], members, cost_ring)

    # `d_phi`：整块场地的 Φ 变化（诊断用）。**组件口径下它不能直接当 Trade** ——
    # 见下面的 ★ 缺陷记录。
    d_phi = [phi[t1][p] - phi[tb][p] for p in (0, 1)]
    d_phi_elx = [0.0, 0.0]      # 只含"手牌圣水 + 成员残值"的版本（组件口径诊断）
    for p in (0, 1):
        elx = float(frames[t1]["elixir%d" % p]) - float(frames[tb]["elixir%d" % p])
        d_phi_elx[p] = elx + (mv_t[p] - mv_b[p])

    # 只算"**本窗口成员**的部署花费"（对齐 §2.1 的净支出形式）
    spend = [0.0, 0.0]
    for eid in members:
        fs = (first_seen or {}).get(eid)
        if fs is None or not (t0 <= fs <= t1):
            continue
        p = (player_of or {}).get(eid, 0)
        spend[p] += float(cost_ring.get(eid, 0.0))
    net_spend = [spend[p] - (mv_t[p] - mv_b[p]) for p in (0, 1)]

    # §7-4 对账：全盘口径下用**原始圣水花费**独立算净支出，再与 ΔΦ 形式比
    identity_err_elx = None
    if phi_mode == "global":
        spend_elx = [sum(v for t, v in deploy_cost[p].items() if t0 <= t <= t1)
                     for p in (0, 1)]
        dv_all = [(phi[t1][p] - phi[tb][p])
                  - (float(frames[t1]["elixir%d" % p])
                     - float(frames[tb]["elixir%d" % p])) for p in (0, 1)]
        net_elx = [spend_elx[p] - dv_all[p] for p in (0, 1)]
        identity_err_elx = (net_elx[1] - net_elx[0]) - (d_phi[0] - d_phi[1])

    # ★ 两侧各自视角的 τ（bug 记录见文档 §11.5）：只算 p0 视角、再让 p1 取 −τ
    # ⇒ **另一半局（main 坐 player 1）拿到的 τ 恒 ≤ 0**，相关性被系统性偏置压成 0。
    tt = tower_term(w, tables, partner, me=0, mode=tower_mode, t_ref=t_ref,
                    cost_of=_cost, frames=frames)
    tau = tt["tau"]
    tau_opp = tower_term(w, tables, partner, me=1, mode=tower_mode, t_ref=t_ref,
                         cost_of=_cost, frames=frames)
    tau1 = tau_opp["tau"]
    if tower_transfer == "zero-sum":
        # 零和且**两侧对称**：term[0] = -term[1]，且两侧都拿到自己那份净额
        term = [0.5 * (tau - tau1), 0.5 * (tau1 - tau)]
    elif tower_transfer == "per-side":
        # 各按自己视角取正 ⇒ **违反**预注册 §7-1（对照臂）
        term = [tau, tau1]
    else:                      # "own"：p0 视角、p1 取负（先前实现，保留作对照）
        term = [tau, -tau]

    # ★ 缺陷 6（2026-09-18 由门禁测试暴露）：**组件口径必须走净支出形式，不能走 ΔΦ 形式**。
    # ΔΦ 形式的圣水部分是**整块场地**的（`elixir0/1` 是全局的）而残值部分是**窗口成员**的
    # ⇒ 只要窗口期间我们在**别的路**下了牌，那笔花费会被算进本窗口却拿不到对应的 V 记入
    # ⇒ 窗口 Trade 被系统性压成负数（实测：本窗口 +3 的场景被算成 −1）。
    # 净支出形式两侧都是"成员口径"，自洽，且**不需要 regen**（§2.1 明说自然回复不计）。
    if phi_mode == "global":
        trade = [d_phi[0] - d_phi[1] + term[0], d_phi[1] - d_phi[0] + term[1]]
    else:
        trade = [net_spend[1] - net_spend[0] + term[0],
                 net_spend[0] - net_spend[1] + term[1]]
    return {
        "t0": t0, "t1": t1, "reason": w["reason"], "span_frames": t1 - t0 + 1,
        "n_members": len(members), "n_core": len(w.get("core_union") or ()),
        "d_phi": d_phi, "d_phi_elx": d_phi_elx,
        "net_spend": net_spend, "identity_err_elx": identity_err_elx,
        "tower_term": term, "tau": tau, "tau_p1": tau1, "kill_credit": tt["kill_credit"],
        "pinned_unit_frames_o0": tt["n_pinned_unit"],
        "pinned_unit_frames_o1": tau_opp["n_pinned_unit"],
        "trade_p0": trade[0], "trade_p1": trade[1],
        "score_p0": max(0.0, trade[0] - THETA),
        "pinned_frames": tt["n_pinned"], "pinned_unit_frames": tt["n_pinned_unit"],
        "pinned_and_killed": tt["pinned_and_killed"], "deaths": len(tt["deaths"]),
    }


# ---------------------------------------------------------------------------
# §3 溯源路由（桶守恒）
# ---------------------------------------------------------------------------
def route_buckets(w, tables, cost_ring, *, me=0, mode="p3a"):
    """把窗口内「敌方单位死亡注销」的账按 ``root_cast`` 路由到桶（§3.2/§3.5）。

    桶 = **出牌事件 id**（写在产物体自己身上 ⇒ 不查召唤者，召唤者可能已死）。
    ``buckets[None]`` = 无法溯源（无交战伙伴 / 伙伴没有 root_cast）。
    返回 ``(buckets, total)``；``total`` = 该窗口全部被路由的账（**不漏不重**由测试断言）。
    """
    partner = _partner_map(w, tables)

    def _cost(eid, name):
        sh = cost_ring.get(eid)
        return card_cost(name) if sh is None else float(sh)

    tt = tower_term(w, tables, partner, me=me, mode=mode, t_ref=2.0, cost_of=_cost)
    buckets = collections.defaultdict(float)
    total = 0.0
    for d in tt["deaths"]:
        amt = d["cost"]
        if amt <= 1e-12:
            continue
        total += amt
        key = None
        t_prev = max(0, d["t"] - 1)
        tab_prev = tables[t_prev] if t_prev < len(tables) else {}
        for o in d["opponents"]:
            od = tab_prev.get(o)
            if od is not None and od.get("root_cast") is not None:
                key = od["root_cast"]
                break
        buckets[key] += amt
    return dict(buckets), total


# ---------------------------------------------------------------------------
# §5 整局分析
# ---------------------------------------------------------------------------
def analyze_game(game, *, settle_frames=1, tower_mode="p3a", tower_transfer="zero-sum",
                 phi_mode="component", recon="costname", t_ref=2.0, rule1=True,
                 identity="core", regen_per_frame=None):
    frames = game.get("frames") or []
    if not frames:
        return None
    meta = game.get("meta") or {}
    # ★ 口径修正（2026-09-18 实测踩到）：联赛录像**每一场对阵打两局、双方互换**，
    # `meta["side0"]` 指明谁坐在 player 0。若一律用 `Trade_p0` / `winner==0` 当
    # "我方"，一半的局会把**脚本对手**当主角 ⇒ 相关性分析直接是垃圾。
    main_player = 0 if str(meta.get("side0", "main")) == "main" else 1
    phi, pdiag = phi_series(frames, recon=recon, regen_per_frame=regen_per_frame)
    cost_ring = pdiag.get("cost_ring") or {}
    first_seen, name_of, player_of, tables = _frame_meta(frames)
    windows = segment_windows(frames, settle_frames=settle_frames, rule1=rule1,
                              identity=identity)
    common = dict(tables=tables, cost_ring=cost_ring, first_seen=first_seen,
                  name_of=name_of, player_of=player_of,
                  deploy_cost=pdiag.get("deploy_cost"), tower_mode=tower_mode,
                  tower_transfer=tower_transfer, phi_mode=phi_mode, t_ref=t_ref,
                  rule1=rule1)
    trades = [window_trade(w, frames, phi, **common) for w in windows]
    gate = realization_gate(frames, windows, trades, pdiag.get("deploy_cost"),
                            main_player)
    if phi_mode == "component":
        g_trades = [window_trade(w, frames, phi, **dict(common, phi_mode="global"))
                    for w in windows]
    else:
        g_trades = trades
    last = frames[-1]
    t0_end = sum(float(x) for x in last["towers0"])
    t1_end = sum(float(x) for x in last["towers1"])
    trade_main = [t["trade_p0"] if main_player == 0 else t["trade_p1"] for t in trades]
    score_main = [max(0.0, x - THETA) for x in trade_main]
    w = game.get("winner")
    win_main = None if w is None else (1 if int(w) == main_player else 0)
    return {"meta": meta, "winner": game.get("winner"), "main_player": main_player,
            "trade_main": trade_main, "score_main": score_main, "win_main": win_main,
            "n_frames": len(frames), "phi": phi, "phi_diag": pdiag,
            "windows": windows, "trades": trades, "global_trades": g_trades,
            "gate": gate,
            "cost_ring": cost_ring,
            # ★ 口径陷阱：`player.get_crown_count()` 返回的是**本方被拆掉的公主塔数**
            # （= 让给对面的皇冠），所以"我方拿到的皇冠"= 对面的 `crown` 字段。
            # 实测：用错方向会让 ρ(冠差, 塔血差) 从 +0.92 变成 −0.92（符号全反）。
            "crown_diff": (int(last["crown1"]) - int(last["crown0"]))
            * (1 if main_player == 0 else -1),
            "tower_hp_diff": (t0_end - t1_end) * (1 if main_player == 0 else -1),
            "elixir0_end": float(last["elixir0"]),
            "elixir1_end": float(last["elixir1"])}


def realization_gate(frames, windows, trades, deploy_cost, main_player, *,
                     n_frames=REALIZE_N_FRAMES, theta=THETA):
    """预注册 §5.1 第三行（**门禁**）：优势兑现率。

    定义（逐字照预注册，只把"我方"解析成 main 侧）：
        `Trade_me >= θ` 之后的 `N = 20 s` 内，
        **我方累计部署花费 >= Trade**（真的把攒下的优势花出去）
        **或** 我方打出的**塔伤 > 0**（真的推进了）
        ⇒ 记一次"兑现"；兑现率 = 兑现数 / 合格窗口数。

    ⚠️ **这是门禁，不是判据**：它的对照必须是**同批阴性对照臂**（§5.1 表第三行的"本实验同批阴性对照"），
    离线 S2 没有对照臂 ⇒ 本函数只给**原始率与分布**，判决留给 S3（§11.7.2）。
    ⚠️ `N = 20 s` 与 `θ = 1.0` 都是"有原则的选择"，**不得**当已标定阈值（§10.2）。
    """
    me = 1 if int(main_player) == 1 else 0
    opp = 1 - me
    dcost = (deploy_cost or {}).get(me) or {}
    elig = 0
    ok_spend = ok_dmg = realized = 0
    spends, trades_pos = [], []
    for w, t in zip(windows, trades):
        trade_me = t["trade_p0"] if me == 0 else t["trade_p1"]
        if trade_me < theta:
            continue
        elig += 1
        t0 = w["t0"]
        t_end = min(len(frames) - 1, t0 + n_frames)
        spend = sum(v for tt, v in dcost.items() if t0 <= tt <= t_end)
        dmg = (sum(float(x) for x in frames[t0]["towers%d" % opp])
               - sum(float(x) for x in frames[t_end]["towers%d" % opp]))
        s_ok, d_ok = spend >= trade_me, dmg > 1e-6
        ok_spend += int(s_ok)
        ok_dmg += int(d_ok)
        realized += int(s_ok or d_ok)
        spends.append(spend)
        trades_pos.append(trade_me)
    return {"n_eligible": elig,
            "n_realized": realized,
            "rate_all": (realized / elig) if elig else None,
            "rate_spend": (ok_spend / elig) if elig else None,
            "rate_dmg": (ok_dmg / elig) if elig else None,
            "spend": _stats(spends), "trade_pos": _stats(trades_pos),
            "n_frames": n_frames, "theta": theta}


def _spearman(xs, ys):
    """秩相关（无 scipy）；样本不足/无方差 ⇒ None。"""
    n = len(xs)
    if n < 3:
        return None

    def rank(v):
        idx = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[idx[j + 1]] == v[idx[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[idx[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def summarize(all_games, *, tower_mode, theta=THETA, tower_transfer="zero-sum",
              phi_mode="component"):
    """汇总成 S2 出口判据要的读数（预注册 §9 S2 的 ①②③④）。"""
    gs = [g for g in all_games if g]
    tw = [t for g in gs for t in g["trades"]]
    n = len(tw)
    by_reason = collections.Counter(t["reason"] for t in tw)
    # 「钉住」的两个口径：任意跨边（含塔，宽）／被非塔实体钉住（"拉扯"口径，窄）
    pinned_any = [t for t in tw if t["pinned_frames"] > 0]
    pinned_unit = [t for t in tw if t["pinned_unit_frames"] > 0]
    pinned_unit_no_kill = [t for t in pinned_unit if t["deaths"] == 0]
    per_game = []
    for g in gs:
        per_game.append({
            "score": sum(g.get("score_main") or [x["score_p0"] for x in g["trades"]]),
            "trade": sum(g.get("trade_main") or [x["trade_p0"] for x in g["trades"]]),
            "trade_phi_only": sum(g.get("trade_main") or [x["trade_p0"] for x in g["trades"]])
            - sum(t["tower_term"][0] if g.get("main_player", 0) == 0
                  else t["tower_term"][1] for t in g["trades"]),
            "winner": g["winner"], "win_main": g.get("win_main"), "n_frames": g["n_frames"],
            "phi_end_diff": (g["phi"][-1][0] - g["phi"][-1][1])
            * (1 if g.get("main_player", 0) == 0 else -1),
            "crown_diff": g.get("crown_diff", 0),
            "tower_hp_diff": g.get("tower_hp_diff", 0.0),
        })
    gates = [g["gate"] for g in gs if g.get("gate") and g["gate"]["n_eligible"]]
    ge = sum(g["n_eligible"] for g in gates)
    gr = sum(g["n_realized"] for g in gates)
    ident = [abs(t["identity_err_elx"]) for t in tw if t["identity_err_elx"] is not None]
    v_err = [g["phi_diag"]["reconstruct_vs_frame_v_max_err"] for g in gs
             if g["phi_diag"].get("reconstruct_vs_frame_v_max_err") is not None]
    return {
        "n_games": len(gs), "n_windows": n,
        "windows_per_game": (n / len(gs)) if gs else None,
        "span_frames": _stats([t["span_frames"] for t in tw]),
        "trade_p0": _stats([t["trade_p0"] for t in tw]),
        "tau": _stats([t["tau"] for t in tw]),
        "kill_credit": _stats([t["kill_credit"] for t in tw]),
        "by_reason": dict(by_reason),
        "tower_mode": tower_mode, "tower_transfer": tower_transfer,
        "phi_mode": phi_mode, "theta": theta,
        "pinned_any_share": (len(pinned_any) / n) if n else None,
        "pinned_unit_share": (len(pinned_unit) / n) if n else None,
        "pinned_no_kill_share": (len(pinned_unit_no_kill) / n) if n else None,
        "pinned_unit_no_kill_frac": (len(pinned_unit_no_kill) / len(pinned_unit))
        if pinned_unit else None,
        "pinned_no_kill_trade": _stats([t["trade_p0"] for t in pinned_unit_no_kill]),
        "frac_tau_eq_kill_credit": (sum(abs(t["tau"] - t["kill_credit"]) < 1e-9
                                        for t in tw) / n) if n else None,
        "frac_tau_le_kill_credit": (sum(t["tau"] <= t["kill_credit"] + 1e-9
                                        for t in tw) / n) if n else None,
        "score_nonzero_share": (sum(t["score_p0"] > 0 for t in tw) / n) if n else None,
        "identity_err_elx": _stats(ident),
        "antisym_max_err": max((abs(t["trade_p0"] + t["trade_p1"]) for t in tw),
                              default=0.0),
        "reconstruct_vs_frame_v_max_err": max(v_err) if v_err else None,
        "gate": {"n_eligible": ge, "n_realized": gr,
                 "rate_all": (gr / ge) if ge else None,
                 "rate_spend": (sum(g["n_eligible"] * g["rate_spend"] for g in gates) / ge)
                 if ge else None,
                 "rate_dmg": (sum(g["n_eligible"] * g["rate_dmg"] for g in gates) / ge)
                 if ge else None,
                 "n_frames": REALIZE_N_FRAMES, "theta": theta,
                 "per_game_eligible": (ge / len(gates)) if gates else None},
        "per_game": per_game,
    }


def _stats(v):
    if not v:
        return {"n": 0}
    s = sorted(v)
    n = len(s)
    med = s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
    return {"n": n, "min": s[0], "p25": s[n // 4], "median": med, "p75": s[3 * n // 4],
            "max": s[-1], "mean": sum(s) / n, "std": _std(s), "mad": _mad(s, med)}


def _std(v):
    n = len(v)
    if n < 2:
        return 0.0
    m = sum(v) / n
    return math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1))


def _mad(s, med):
    return sorted(abs(x - med) for x in s)[len(s) // 2]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _iter_files(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            for nm in sorted(os.listdir(p)):
                if nm.endswith(".pkl"):
                    out.append(os.path.join(p, nm))
        else:
            out.append(p)
    return out


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="离线「局面圣水交换」仪器（只读回放）")
    ap.add_argument("--replays", nargs="+", default=[],
                    help="录像文件或目录（目录取全部 *.pkl）")
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--half", type=int, default=0, choices=[0, 1, 2],
                    help="按奇偶把每份录像的局**交错对半**切（1=奇数位，2=偶数位）"
                         "⇒ 用来量「同一批数据的两个半批之间排名稳不稳」（【R16】）")
    ap.add_argument("--settle-frames", type=int, default=1)
    ap.add_argument("--settle-ticks", type=int, default=SETTLE_TICKS)
    ap.add_argument("--tower-mode", default="p4",
                    choices=["none", "p3a", "p3b", "p1", "p2", "p4", "p4b", "p5", "p5b"])
    ap.add_argument("--tower-transfer", default="zero-sum",
                    choices=["zero-sum", "per-side", "own"],
                    help="zero-sum=两侧对称零和（默认，满足 §7-1）；"
                         "per-side=各取正（违 §7-1，对照）；own=p0 视角 p1 取负（有偏，对照）")
    ap.add_argument("--phi-mode", default="component", choices=["global", "component"])
    ap.add_argument("--identity", default="core", choices=["core", "members"],
                    help="局面续接的身份判据（core=交战核心，members=全成员）")
    ap.add_argument("--recon", default="costname", choices=["naive", "costname"])
    ap.add_argument("--t-ref", type=float, default=2.0)
    ap.add_argument("--json", default=None)
    ap.add_argument("--sweep", action="store_true",
                    help="扫全部 tower_mode（none/p3a/p3b/p1/p2）并打印选代理表")
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args(argv)

    if a.settle_frames * TICK_PER_FRAME != a.settle_ticks:
        print("[FAIL] settle_frames*%d != settle_ticks (%d*%d != %d)"
              % (TICK_PER_FRAME, a.settle_frames, TICK_PER_FRAME, a.settle_ticks))
        return 2
    files = _iter_files(a.replays)
    if not files:
        ap.error("--replays 为空")

    all_games, file_report = [], []
    _CACHE_GAMES.clear()
    for p in files:
        schema, games = load_replay(p)
        if a.half:
            games = games[a.half - 1::2]
        if a.limit_games:
            games = games[:a.limit_games]
        file_report.append({"file": p, "schema": schema, "games": len(games),
                            "entity_len": replay_entity_arity(games)})
        for g in games:
            _CACHE_GAMES.append(g)
            all_games.append(analyze_game(
                g, settle_frames=a.settle_frames, tower_mode=a.tower_mode,
                tower_transfer=a.tower_transfer, phi_mode=a.phi_mode,
                recon=a.recon, t_ref=a.t_ref, identity=a.identity))

    rep = {"files": file_report,
           "config": {"settle_frames": a.settle_frames, "settle_ticks": a.settle_ticks,
                      "tower_mode": a.tower_mode, "tower_transfer": a.tower_transfer,
                      "phi_mode": a.phi_mode, "identity": a.identity,
                      "recon": a.recon, "t_ref": a.t_ref, "theta": THETA},
           "summary": summarize(all_games, tower_mode=a.tower_mode,
                                tower_transfer=a.tower_transfer, phi_mode=a.phi_mode)}

    print("=" * 80)
    print("离线「局面圣水交换」仪器（S2 仪器，只读回放）")
    print("=" * 80)
    for fr in file_report:
        print("  %-44s schema=%-2s games=%-4d 实体元组长度=%s"
              % (os.path.basename(fr["file"]), fr["schema"], fr["games"],
                 fr["entity_len"]))
    s = rep["summary"]
    exact = any(g and g["phi_diag"].get("exact") for g in all_games)
    print("\n[v 来源] %s（schema≥5 带帧内 v0/v1 ⇒ 精确；否则重建 ⇒ 近似，见文档 §11.5）"
          % ("帧内真值" if exact else "离线重建"))
    print("[口径] settle_frames=%d（=%d tick）｜identity=%s｜tower_mode=%s｜"
          "tower_transfer=%s｜phi_mode=%s｜recon=%s｜θ=%.1f"
          % (a.settle_frames, a.settle_ticks, a.identity, a.tower_mode, a.tower_transfer,
             a.phi_mode, a.recon, THETA))
    print("\n① 局面窗口 %d 个（%.2f 个/局）｜跨度(帧): %s"
          % (s["n_windows"], s["windows_per_game"] or 0, _fmt(s["span_frames"])))
    print("   结算原因: %s" % s["by_reason"])
    print("   Trade_p0: %s" % _fmt(s["trade_p0"]))
    print("   τ(%s): %s" % (a.tower_mode, _fmt(s["tau"])))
    print("   窗口内敌方死亡注销额: %s" % _fmt(s["kill_credit"]))
    print("   ★ τ == kill_credit 的窗口占比 = %s（τ ≤ kill_credit 占比 %s）"
          % (_pct(s["frac_tau_eq_kill_credit"]), _pct(s["frac_tau_le_kill_credit"])))
    print("\n③ 「被非塔实体钉住」的窗口占比 = %s；其中零死亡（钉住但没打死）= %s"
          "（占有钉住窗口 %s）"
          % (_pct(s["pinned_unit_share"]), _pct(s["pinned_no_kill_share"]),
             _pct(s["pinned_unit_no_kill_frac"])))
    print("   宽口径（任意跨边含塔）窗口占比 = %s" % _pct(s["pinned_any_share"]))
    print("   钉住未杀窗口的 Trade_p0: %s" % _fmt(s["pinned_no_kill_trade"]))
    print("\n④ score>0 窗口占比 = %s" % _pct(s["score_nonzero_share"]))
    g = s.get("gate") or {}
    print("   ★ 门禁·优势兑现率（N=%d 帧 = %.0f s，θ=%.1f）: 合格窗口 %d 个（%.1f/局）"
          % (g.get("n_frames", 0), g.get("n_frames", 0) * 0.5, g.get("theta", 0),
             g.get("n_eligible", 0), g.get("per_game_eligible") or 0.0))
    print("     兑现率 = %s（花费达标 %s ／ 打出塔伤 %s）"
          % (_pct(g.get("rate_all")), _pct(g.get("rate_spend")), _pct(g.get("rate_dmg"))))
    print("   对账 |净支出(原始圣水) − ΔΦ| : %s" % _fmt(s["identity_err_elx"]))
    print("   反对称：max|Trade_p0 + Trade_p1| = %.3e" % s["antisym_max_err"])
    if s["reconstruct_vs_frame_v_max_err"] is not None:
        print("   ★ 重建 V vs 帧内真值 V：max 误差 = %.3e（schema 5 专有对账）"
              % s["reconstruct_vs_frame_v_max_err"])
    if s["per_game"]:
        pg = s["per_game"]
        print("   秩相关(局级 Trade vs 终局 Φ 差) = %s"
              % _r(_spearman([g["trade"] for g in pg], [g["phi_end_diff"] for g in pg])))
        print("   秩相关(局级 Trade 去掉 τ vs 终局 Φ 差) = %s"
              % _r(_spearman([g["trade_phi_only"] for g in pg],
                             [g["phi_end_diff"] for g in pg])))
    tw = sorted(tw_all(all_games), key=lambda t: (-t["span_frames"], t["t0"]))
    print("\n窗口样例（跨度前 %d）:" % a.top)
    for t in tw[:a.top]:
        print("   [%4d,%4d] %-16s span=%-4d ΔΦ=(%+.2f,%+.2f) τ=%+.2f kill=%+.2f "
              "Trade0=%+.2f score=%+.2f" % (t["t0"], t["t1"], t["reason"],
                                            t["span_frames"], t["d_phi"][0],
                                            t["d_phi"][1], t["tau"], t["kill_credit"],
                                            t["trade_p0"], t["score_p0"]))
    if a.json:
        d = os.path.dirname(os.path.abspath(a.json))
        if d:
            os.makedirs(d, exist_ok=True)
        slim = {"files": file_report, "config": rep["config"], "summary": s,
                "games": [None if g is None else
                          {"winner": g["winner"], "n_frames": g["n_frames"],
                           "trades": g["trades"]} for g in all_games]}
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(slim, f, ensure_ascii=False, indent=1)
        print("\n[JSON] %s" % a.json)
    if a.sweep:
        print("\n" + "=" * 80)
        print("★ S2 出口判据 ②：tower_term 代理选择扫描（其它口径完全固定）")
        print("=" * 80)
        print("%-6s %7s %7s %7s %8s %8s %9s %9s %9s %9s"
              % ("代理", "τ中位", "τ均值", "τ≠0", "score>0", "兑现率", "ρ(冠差)",
                 "ρ(塔血)", "ρ(胜)", "ρ(Φ差)*"))
        raw = [g for g in all_games if g]
        for mode in ("none", "p3a", "p3b", "p1", "p2", "p4", "p4b", "p5", "p5b"):
            gs = [analyze_game(g_src, settle_frames=a.settle_frames, tower_mode=mode,
                               tower_transfer=a.tower_transfer, phi_mode=a.phi_mode,
                               recon=a.recon, t_ref=a.t_ref, identity=a.identity)
                  for g_src in _CACHE_GAMES]
            sm = summarize(gs, tower_mode=mode, tower_transfer=a.tower_transfer,
                           phi_mode=a.phi_mode)
            pg = sm["per_game"]
            twm = list(_tw(gs))
            print("%-6s %7.2f %7.2f %7s %8s %8s %9s %9s %9s %9s"
                  % (mode, (sm["tau"] or {}).get("median", 0.0),
                     (sm["tau"] or {}).get("mean", 0.0),
                     _pct(None if not twm else sum(1 for t in twm if abs(t["tau"]) > 1e-12)
                          / len(twm)),
                     _pct(sm["score_nonzero_share"]),
                     _pct((sm.get("gate") or {}).get("rate_all")),
                     _r(_spearman([x["trade"] for x in pg],
                                  [x["crown_diff"] for x in pg])),
                     _r(_spearman([x["trade"] for x in pg],
                                  [x["tower_hp_diff"] for x in pg])),
                     _r(_spearman([x["trade"] for x in pg],
                                  [1.0 if x.get("win_main") == 1 else 0.0 for x in pg])),
                     _r(_spearman([x["trade"] for x in pg],
                                  [x["phi_end_diff"] for x in pg]))))
        print("\n★ 前三个 ρ 用**非自引用**结果（皇冠差 / 终局塔血差 / 胜负）；"
              "带 * 的 ρ(Φ差) 是**自引用**的（Trade 由 ΔΦ 构成）⇒ 只作 sanity，不作判据。"
              "\nn=%d 局 ⇒ ρ 的 1σ ≈ %.3f ⇒ 小于 ~0.26 的差异不可解读。"
              % (len(raw), 1.0 / max(1.0, (len(raw) - 1) ** 0.5)))
    return 0


def _tw(gs):
    for g in gs:
        if g:
            for t in g["trades"]:
                yield t


def tw_all(all_games):
    for g in all_games:
        if g:
            for t in g["trades"]:
                yield t


def _fmt(st):
    if not st or st.get("n", 0) == 0:
        return "(空)"
    return ("n=%d min=%+.2f p25=%+.2f med=%+.2f p75=%+.2f max=%+.2f mean=%+.2f "
            "σ=%.2f MAD=%.2f" % (st["n"], st["min"], st["p25"], st["median"], st["p75"],
                                 st["max"], st["mean"], st["std"], st["mad"]))


def _pct(x):
    return "n/a" if x is None else "%.1f%%" % (100.0 * x)


def _r(x):
    return "n/a" if x is None else "%+.3f" % x


if __name__ == "__main__":
    raise SystemExit(main())
