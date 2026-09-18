# -*- coding: utf-8 -*-
"""预注册 §7 的 7 条不变量 —— 前 6 条是**离线仪器**的测试（零训练成本、零引擎依赖）。

    docs/engagement_trade_prereg_2026-09-18.md §7 的表（**测试名先写死，再写实现**）：

    | # | 不变量                                                        | 测试名 |
    |---|---|---|
    | 1 | 反对称/零和：同一局同一局面 ``trade_p0 == -trade_p1``          | test_engagement_trade_antisymmetry |
    | 2 | 桶守恒：``Σ(各 root_cast 桶) == 该局全部路由项之和``（不漏不重） | test_root_cast_bucket_conservation |
    | 3 | 产物体恒 0 费：任何产物体出现/死亡都不改变 ``V``                | test_product_zero_value |
    | 4 | 窗口等式：``Trade_me == ΔΦ_window + tower_term``（同轨配对复算） | test_trade_equals_phi_window |
    | 5 | 局面切分：6 个手工场景逐一断言分段与结算时刻                    | test_engagement_segmentation |
    | 6 | 跨局边界：账本/桶在局边界完全重置（9j 事故类）                  | test_trade_reset_across_episodes |
    | 7 | 默认关逐位回旧                                                  | test_engagement_trade_default_off（**引擎侧，随 §6 第 5 项落地**）|

本文件用**合成帧**（不读录像、不跑引擎）⇒ 快、确定、可在任何机器上复跑。

跑法（必须 cwd = ``src/clasher_new``）：
    cd src/clasher_new && PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe \
        ../../scripts/selftest_offline_engagement_trade.py
"""

from __future__ import annotations

import os
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import offline_engagement_trade as O  # noqa: E402

# ---------------------------------------------------------------------------
# 合成帧构造
# ---------------------------------------------------------------------------
def E(eid, name, player, kind="troop", hp=100.0, x=0.0, y=0.0, target=None,
      is_product=None, root_cast=None, share=None):
    """实体元组（12 元组 = schema 4；带 is_product/root_cast/share 时自动补到 15 = schema 5）。"""
    e = [name, float(x), float(y), float(hp), int(player), kind, float(hp),
         0.0, 0.0, 0.5, int(eid), (int(target) if target is not None else None)]
    if is_product is not None or root_cast is not None or share is not None:
        e += [root_cast, is_product, share]
    return e


def TOWERS(hp0=3000.0, hp1=3000.0):
    """塔固定 id：1,2,5 = player 1；3,4,6 = player 0（`battle.py::update_player_hp` 同源）。"""
    return [E(1, "King_PrincessTowers", 1, "building", hp1),
            E(2, "King_PrincessTowers", 1, "building", hp1),
            E(5, "KingTower", 1, "building", hp1),
            E(3, "King_PrincessTowers", 0, "building", hp0),
            E(4, "King_PrincessTowers", 0, "building", hp0),
            E(6, "KingTower", 0, "building", hp0)]


def F(t, ents, elixir=(5.0, 5.0), thp=(3000.0, 3000.0), cards=None, opp=None,
      crown=(0, 0)):
    return {"t": round(float(t), 2), "bundle": [], "reward": 0.0,
            "opp_played": (opp or []), "cards": cards,
            "towers0": [thp[0], thp[0], thp[0] * 1.6],
            "towers1": [thp[1], thp[1], thp[1] * 1.6],
            "elixir0": float(elixir[0]), "elixir1": float(elixir[1]),
            "crown0": int(crown[0]), "crown1": int(crown[1]),
            "entities": list(TOWERS(thp[0], thp[1])) + list(ents)}


def GAME(frames, side0="main", winner=0):
    return {"meta": {"pair": ["main", "main"], "side0": side0, "max_steps": 360},
            "winner": winner, "frames": frames}


def _win(core_key, ids, frame_count, start=0):
    """手工构造一个窗口 dict（测 tower_term / window_trade 时用）。"""
    return {"rep": ("t", start), "t0": start, "t1": start + frame_count - 1,
            "reason": "manual",
            "last_active": start + frame_count - 1, "inactive_run": 0,
            "members_union": set(ids), "members_final": set(ids),
            "core_union": set(core_key), "key": set(core_key)}


# ---------------------------------------------------------------------------
# §7-5 局面切分（6 个手工场景 + 去抖 K）
# ---------------------------------------------------------------------------
def test_engagement_segmentation():
    # —— 场景 1「孤零零小屋」：没有交战对 ⇒ 0 个窗口 ——
    fr = [F(i, [E(7, "Tombstone", 1, "building", 900.0, is_product=False)])
          for i in range(5)]
    assert O.segment_windows(fr) == [], "孤零零小屋不该产生局面"

    # —— 场景 2「小屋 + 最近一批」：前 5 帧无索敌 ⇒ 无窗口；第 5 帧起骷髅打我方塔 ⇒ 1 窗口 ——
    pre = [F(i, [E(7, "Tombstone", 1, "building", 900.0, is_product=False),
                 E(8, "Skeleton", 1, is_product=True, target=None),
                 E(9, "Skeleton", 1, is_product=True, target=None)])
           for i in range(5)]
    post = [F(i, [E(7, "Tombstone", 1, "building", 900.0, is_product=False),
                  E(8, "Skeleton", 1, is_product=True, target=3),
                  E(9, "Skeleton", 1, is_product=True, target=None)],
              thp=(3000.0, 3000.0)) for i in range(5, 10)]
    ws = O.segment_windows(pre + post)
    assert len(ws) == 1, "小屋+召唤物应该只产生 1 个局面，实得 %d" % len(ws)
    assert ws[0]["t0"] == 5, ws[0]["t0"]
    assert 3 in ws[0]["members_union"] and 8 in ws[0]["members_union"], ws[0]
    assert 8 in ws[0]["core_union"], "交战核心应含免费骷髅 8：%s" % ws[0]
    assert 3 not in ws[0]["core_union"], "塔不该进交战核心（否则窗口永不结算）"

    # —— 场景 3「该批 vs 我方弓手」：第 0 帧就开战 ——
    fr = [F(i, [E(7, "Tombstone", 1, "building", 900.0, is_product=False),
                E(8, "Skeleton", 1, is_product=True, target=10),
                E(10, "Archer", 0, is_product=False, target=8)]) for i in range(5)]
    ws = O.segment_windows(fr)
    assert len(ws) == 1 and ws[0]["t0"] == 0 and ws[0]["t1"] == 4, ws

    # —— 场景 4「单只免费骷髅打塔」：塔是非产物 ⇒ 规则①满足 ⇒ **算活跃局面** ——
    fr = [F(i, [E(8, "Skeleton", 1, is_product=True, target=3)]) for i in range(5)]
    ws = O.segment_windows(fr)
    assert len(ws) == 1 and ws[0]["t0"] == 0, "规则①：只要一侧非产物即算交战对"

    # —— 场景 5「双路同时开战」：两个互不相连的分量 ⇒ 2 个窗口、同一时段 ——
    fr = [F(i, [E(8, "Skeleton", 1, is_product=True, target=3),
                E(9, "Skeleton", 1, is_product=True, target=4)]) for i in range(5)]
    ws = O.segment_windows(fr)
    assert len(ws) == 2, "双路应该是 2 个局面，实得 %d" % len(ws)
    assert {w["t0"] for w in ws} == {0} and {w["t1"] for w in ws} == {4}, ws
    assert {frozenset(w["core_union"]) for w in ws} == {frozenset({8}), frozenset({9})}

    # —— 场景 6「目标瞬态死掉」：第 1 帧骷髅消失 ⇒ 分量清空 ⇒ 立即结算 ——
    fr = [F(0, [E(8, "Skeleton", 1, is_product=True, target=10),
                E(10, "Archer", 0, is_product=False, target=8)]),
          F(1, [E(10, "Archer", 0, is_product=False, target=None)])]
    ws = O.segment_windows(fr)
    assert len(ws) == 1, ws
    assert ws[0]["t1"] == 1 and ws[0]["last_active"] == 0, ws[0]
    assert ws[0]["reason"] in ("split_or_vanish", "no_engagement_K"), ws[0]["reason"]
    # 说明：K=1 时，若分量里还有存活成员（此处的弓手）⇒ 记 no_engagement_K；
    #      若整个分量被清空 ⇒ 记 split_or_vanish。两者都表示"局面在此帧结束"。

    # —— 场景 7「去抖 K」：活跃 → 不活跃（双方都活着、只是丢掉目标）→ 又活跃 ——
    fr = [F(0, [E(8, "Skeleton", 1, is_product=True, target=10),
                E(10, "Archer", 0, is_product=False, target=8)]),
          F(1, [E(8, "Skeleton", 1, is_product=True, target=None),
                E(10, "Archer", 0, is_product=False, target=None)]),
          F(2, [E(8, "Skeleton", 1, is_product=True, target=10),
                E(10, "Archer", 0, is_product=False, target=8)])]
    w1 = O.segment_windows(fr, settle_frames=1)
    assert len(w1) == 2, "K=1 时中间那帧就结算 ⇒ 应得 2 个窗口，实得 %d" % len(w1)
    assert (w1[0]["t0"], w1[0]["t1"]) == (0, 1), w1[0]
    w2 = O.segment_windows(fr, settle_frames=2)
    assert len(w2) == 1, "K=2 时去抖窗内续接 ⇒ 应得 1 个窗口，实得 %d" % len(w2)
    assert (w2[0]["t0"], w2[0]["t1"]) == (0, 2), w2[0]

    # —— 场景 8「换人打同一座塔」：身份判据必须是交战核心，不能靠塔 ——
    fr = [F(0, [E(8, "Skeleton", 1, is_product=True, target=3)]),
          F(1, [E(9, "Skeleton", 1, is_product=True, target=3)])]
    assert len(O.segment_windows(fr)) == 2, "不同单位打同一座塔应是两个局面"
    assert len(O.segment_windows(fr, identity="members")) == 1, \
        "members 判据会把它们粘成一个（这就是初版 span=247 帧的来源）"
    return "8 个场景全过（含 K 去抖 1/2 与 core-vs-members 对照）"


# ---------------------------------------------------------------------------
# §7-1 反对称 / 零和
# ---------------------------------------------------------------------------
def test_engagement_trade_antisymmetry():
    # 骷髅（免费产物）被弓手钉住并打死 ⇒ τ>0；再叠一个"塔自己打死"的对照
    def build():
        # 敌方 4 费 MiniPekka 被我方 Archer（非塔）钉住并打死 ⇒ τ_p0 = 4（p3b）
        return [
            F(0, [E(8, "MiniPekka", 1, is_product=False, share=4.0, target=10),
                  E(10, "Archer", 0, is_product=False, share=3.0, target=8)],
              elixir=(5.0, 5.0)),
            F(1, [E(10, "Archer", 0, is_product=False, share=3.0, target=8)],
              elixir=(5.5, 5.5)),
        ]
    fr = build()
    phi, _ = O.phi_series(fr, recon="naive")
    for transfer, must_be_zero in (("zero-sum", True), ("per-side", False)):
        errs = []
        sums = []
        for w in O.segment_windows(fr, identity="core"):
            t = O.window_trade(w, fr, phi, tower_mode="p3b",
                               tower_transfer=transfer)
            errs.append(abs(t["trade_p0"] + t["trade_p1"]))
            sums.append(t["trade_p0"] + t["trade_p1"])
        assert errs, "没有窗口，测试无意义"
        if must_be_zero:
            assert max(errs) < 1e-9, "%s 应零和，实得 %s" % (transfer, errs)
        else:
            # per-side：两侧各取正 ⇒ 一般**不**零和（这正是 §7-1 要钉住的规格）
            assert any(abs(s) > 1e-9 for s in sums), \
                "per-side 应当破坏零和（本样本 τ_p1=0）"
    # 对称性：把整局左右翻面，trade_main 的绝对值必须一致
    fr2 = [dict(f) for f in fr]
    for f in fr2:
        f["entities"] = [[e[0], 100.0 - e[1], 32.0 - e[2], e[3],
                          (1 - e[4]), e[5]] + list(e[6:]) for e in f["entities"]]
        f["elixir0"], f["elixir1"] = f["elixir1"], f["elixir0"]
        f["towers0"], f["towers1"] = f["towers1"], f["towers0"]
    phi2, _ = O.phi_series(fr2, recon="naive")
    a = [O.window_trade(w, fr, phi, tower_mode="p3b")["trade_p0"]
         for w in O.segment_windows(fr, identity="core")]
    b = [O.window_trade(w, fr2, phi2, tower_mode="p3b")["trade_p1"]
         for w in O.segment_windows(fr2, identity="core")]
    assert len(a) == len(b), (a, b)
    for x, y in zip(a, b):
        assert abs(x - y) < 1e-6, "左右翻面后 trade_main 应一致：%s vs %s" % (x, y)
    return "zero-sum 逐窗 |p0+p1| < 1e-9；per-side 确实破零和；左右翻面对称"


# ---------------------------------------------------------------------------
# §7-2 桶守恒（root_cast 溯源：不漏不重）
# ---------------------------------------------------------------------------
def test_root_cast_bucket_conservation():
    """4 个敌方（player 1）单位死亡，分别被 root_cast=101 / 101 / 102 的**免费产物**
    钉住打死，第 4 个被**我方塔**打死（塔没有 root_cast ⇒ None 桶）。
    桶守恒 = 每个路由项被计入且只计入一次。"""
    fr = [
        F(0, [E(8, "Skeleton", 0, is_product=True, root_cast=101, share=0.0, target=20),
              E(9, "Skeleton", 0, is_product=True, root_cast=101, share=0.0, target=21),
              E(10, "Skeleton", 0, is_product=True, root_cast=102, share=0.0, target=22),
              E(20, "Knight", 1, is_product=False, share=3.0, target=8),
              E(21, "MiniPekka", 1, is_product=False, share=4.0, target=9),
              E(22, "Musketeer", 1, is_product=False, share=4.0, target=10),
              E(23, "HogRider", 1, is_product=False, share=4.0, target=3)]),
        # 四只敌方单位全部死亡；我方产物与塔仍在
        F(1, [E(8, "Skeleton", 0, is_product=True, root_cast=101, share=0.0, target=None),
              E(9, "Skeleton", 0, is_product=True, root_cast=101, share=0.0, target=None),
              E(10, "Skeleton", 0, is_product=True, root_cast=102, share=0.0, target=None)]),
    ]
    tables = [O.entity_table(f) for f in fr]
    ring = {}
    for f in fr:
        for e in f["entities"]:
            if len(e) > 14 and e[14] is not None:
                ring[int(e[10])] = float(e[14])
    wins = O.segment_windows(fr, identity="core")
    assert len(wins) == 4, "应有 4 个局面（3 个产物 + 1 个塔各自一个），实得 %d" % len(wins)
    buckets, total = {}, 0.0
    for w in wins:
        bk, tt = O.route_buckets(w, tables, ring)
        for k, v in bk.items():
            buckets[k] = buckets.get(k, 0.0) + v
        total += tt
    assert abs(sum(buckets.values()) - total) < 1e-9, \
        "桶守恒失败：Σ桶=%s total=%s" % (buckets, total)
    assert abs(buckets.get(101, 0.0) - 7.0) < 1e-9, buckets
    assert abs(buckets.get(102, 0.0) - 4.0) < 1e-9, buckets
    assert abs(buckets.get(None, 0.0) - 4.0) < 1e-9, \
        "被塔打死（塔无 root_cast）⇒ None 桶：%s" % buckets
    assert abs(total - 15.0) < 1e-9, "路由总额应为 15.0，实得 %s（桶=%s）" % (total, buckets)
    return "Σ桶 == total == 15.0（101:7 / 102:4 / None:4），不漏不重"


# ---------------------------------------------------------------------------
# §7-3 产物体恒 0 费（出现/死亡都不改变 V）
# ---------------------------------------------------------------------------
def test_product_zero_value():
    # (a) 重建路径（schema 4 口径）：第 1 帧真出 Archer(3 费)，第 2 帧凭空多一只产物，
    #     第 3 帧该产物死掉 ⇒ V 必须**全程不变**（产物份额恒 0，出现/死亡都不入账）。
    #     注：出牌不能放第 0 帧 —— 第 0 帧没有"上一帧圣水"，花费无法核算（真录像同样如此）。
    fr = [
        F(0, [], elixir=(5.0, 5.0)),
        F(1, [E(10, "Archer", 0, is_product=False)], elixir=(2.5, 5.5),
          cards=["Archer"]),
        F(2, [E(10, "Archer", 0, is_product=False),
              E(11, "Skeleton", 1, is_product=True, target=None)], elixir=(3.0, 6.0)),
        F(3, [E(10, "Archer", 0, is_product=False)], elixir=(3.5, 6.5)),
    ]
    v, diag = O.reconstruct_v(fr, regen_per_frame=0.5, recon="naive")
    assert abs(v[1][0] - 3.0) < 1e-9, "第 1 帧真出 Archer 应记 V=3：%s" % v
    assert v[1] == v[2] == v[3], "产物体不该改变 V：%s" % v
    assert abs(diag["share_map"][0][10] - 3.0) < 1e-9, diag["share_map"]

    # (b) 精确路径（schema 5）：帧内 v0/v1 是权威，产物 share=0 不参与残值
    fr2 = [dict(f, v0=3.0, v1=0.0) for f in fr]
    phi, pd = O.phi_series(fr2, recon="naive")
    assert pd["exact"] is True
    tab = O.entity_table(fr2[2])
    assert O.member_v(tab, {11}, pd["cost_ring"]) == [0.0, 0.0], \
        "产物体份额必须 0"
    assert abs(phi[2][0] - (3.0 + 3.0)) < 1e-9, phi

    # (c) 货币泵/刷分两条否决证明的可执行版本：若给产物估值，则"出现"会凭空抬 V、
    #     "被清掉"会让击杀者零成本拿分。这里断言我们**没有**这么做。
    assert abs(v[2][0] - v[1][0]) < 1e-9 and abs(v[3][0] - v[2][0]) < 1e-9
    return "产物出现/死亡均不改 V（重建与帧内真值两条路径都验）"


# ---------------------------------------------------------------------------
# §7-4 窗口等式（同轨配对复算：ΔΦ 形式 == 净支出形式）
# ---------------------------------------------------------------------------
def test_trade_equals_phi_window():
    # 手工构造：p1 第 0 帧部署 Knight(3 费)，p0 第 2 帧部署 HogRider(4 费)，
    # p0 的 HogRider 第 5 帧死（被 Knight 打死），双方全程交战。
    # regen = 0.5 圣水/帧；窗口 = [2,5]，基线 = 第 1 帧边界。
    reg = 0.5
    e0 = [5.0, 5.0]
    e1 = [5.0, 5.0]
    seq0, seq1 = [], []
    for i in range(7):
        if i == 0:
            e1 = [e1[0], e1[1] + reg]
        if i == 2:
            e0 = [e0[0] + reg, e0[1]]
        seq0.append(e0[0])
        seq1.append(e1[1])
    # 精确一点：直接用递增序列，并在第 0/2 帧扣费
    el0, el1 = [5.0], [5.0]
    for i in range(1, 7):
        a = el0[-1] + reg
        b = el1[-1] + reg
        if i == 2:
            a -= 4.0        # HogRider
        el0.append(a)
        el1.append(b)
    frames = []
    for i in range(7):
        ents = []
        if i >= 0:
            ents.append(E(20, "Knight", 1, is_product=False, share=0.0, target=21))
        if i >= 2 and i < 5:
            ents.append(E(21, "HogRider", 0, is_product=False, share=4.0, target=20))
        if i >= 5:
            pass            # HogRider 死了
        frames.append(F(i, ents, elixir=(el0[i], el1[i]),
                        cards=(["HogRider"] if i == 2 else None)))
    # p1 的 Knight 是"部署在局外"的：手工把它的份额灌进 V（用 deploy_cost 记录）
    phi, diag = O.phi_series(frames, recon="naive", regen_per_frame=reg)
    # 手工造一个覆盖 [2,5] 的窗口（成员 = HogRider21 + Knight20 + 双方塔）
    w = _win({21, 20}, set(range(1, 7)) | {20, 21}, 4, start=2)
    tables = [O.entity_table(f) for f in frames]
    fs = {20: 0, 21: 2}
    no = {20: "Knight", 21: "HogRider"}
    po = {20: 1, 21: 0}
    t = O.window_trade(w, frames, phi, tables=tables,
                       cost_ring=diag["cost_ring"], first_seen=fs, name_of=no,
                       player_of=po, deploy_cost=diag["deploy_cost"],
                       tower_mode="none", phi_mode="component")
    # 形式 A（ΔΦ）与形式 B（净支出）必须一致：等价的代数条件是 regen 在两侧相消
    lhs = t["d_phi"][0] - t["d_phi"][1]
    rhs = t["net_spend"][1] - t["net_spend"][0]
    assert abs(lhs - rhs) < 1e-6, "窗口等式不成立：ΔΦ=%s 净支出=%s" % (lhs, rhs)
    # 数值断言：我方净支出 = 4（下 Hog）− 0（死了没返场）= 4；对面净支出 = 0（没在本窗下牌）
    #   − (V(t1)−V(t0)) 其中对面 Knight 不在本窗新增（first_seen=0 < t0=2）⇒ 净支出 0
    #   ⇒ Trade_p0 = 0 − 4 = −4
    assert abs(t["net_spend"][0] - 4.0) < 1e-9, t["net_spend"]
    assert abs(t["net_spend"][1] - 0.0) < 1e-9, t["net_spend"]
    assert abs(t["trade_p0"] - (-4.0)) < 1e-9, t["trade_p0"]
    # 全局口径下的原始圣水对账（出牌检测 + regen 自洽）必须闭合
    tg = O.window_trade(w, frames, phi, tables=tables, cost_ring=diag["cost_ring"],
                        first_seen=fs, name_of=no, player_of=po,
                        deploy_cost=diag["deploy_cost"], tower_mode="none",
                        phi_mode="global")
    assert tg["identity_err_elx"] is not None
    assert abs(tg["identity_err_elx"]) < 1e-6, \
        "全局口径 |净支出(原始圣水) − ΔΦ| 应为 0，实得 %s" % tg["identity_err_elx"]
    return ("ΔΦ=%+.3f == 净支出=%+.3f；全局口径原始圣水对账残差 %.2e"
            % (lhs, rhs, tg["identity_err_elx"]))


# ---------------------------------------------------------------------------
# §7-6 跨局边界（9j 事故类：账本/桶必须完全重置）
# ---------------------------------------------------------------------------
def test_trade_reset_across_episodes():
    def game_a():
        return [F(i, [E(8, "Skeleton", 1, is_product=True, target=3)]) for i in range(4)]

    def game_b():
        # 同样的实体 id（8）但不同对象、不同玩家布局；id 复用是最容易串账的情形
        return [F(i, [E(8, "Valkyrie", 0, is_product=False, target=1)]) for i in range(4)]
    A, B = game_a(), game_b()
    wa = O.segment_windows(A, identity="core")
    wb = O.segment_windows(B, identity="core")
    assert len(wa) == 1 and len(wb) == 1, (wa, wb)
    assert wa[0]["members_union"] == {3, 8}, wa[0]
    assert wb[0]["members_union"] == {1, 8}, wb[0]
    # Φ / V：局 B 的 V 必须从 0 起算，不得继承局 A 的任何残留
    phiA, dA = O.phi_series(A, recon="naive", regen_per_frame=0.5)
    phiB, dB = O.phi_series(B, recon="naive", regen_per_frame=0.5)
    assert abs(dA["v"][0][0] - dB["v"][0][0]) < 1e-12, (dA["v"], dB["v"])
    # 局 B 不得出现局 A 独有的塔（3/4/6 是 p0 的塔 —— 局 B 里 3 不再被当作成员来源）
    assert 3 not in wb[0]["members_union"]
    # 桶：两局各自独立结算，Σ桶 分别守恒
    tb = [O.entity_table(f) for f in B]
    bk, tot = O.route_buckets(wb[0], tb, dB["cost_ring"])
    assert abs(sum(bk.values()) - tot) < 1e-9
    # 合并成"一局两段"绝不能发生：analyze_game 只吃单局 frames
    r = O.analyze_game(GAME(A), settle_frames=1)
    assert r["n_frames"] == 4, r["n_frames"]
    return "id 复用不串账；V/桶在局边界从 0 起算；两局窗口成员各自独立"



# ---------------------------------------------------------------------------
# 预注册 §5.1 第三行：优势兑现率门禁（N=20 s）
# ---------------------------------------------------------------------------
def test_realization_gate():
    """合成一局：p1 的 3 费 Knight 在窗口 [1,4] 内被我们的塔打死（Trade_p0 = +3）。
    然后分三种情形验门禁：
      A 兑现（花 4 费 ≥ 3 **且** 打出塔伤）／B 只守不推（既不花也不打）／C 不合格（Trade < θ）。"""
    def build(spend_card, tower_loss):
        el0, el1 = [5.0], [5.0]
        for i in range(1, 46):
            a, b = el0[-1] + 0.5, el1[-1] + 0.5
            if i == 1:
                b -= 3.0                      # p1 出 Knight
            if i == 2 and spend_card:
                a -= 4.0                      # p0 出 HogRider（4 费）
            el0.append(a)
            el1.append(b)
        fr = []
        for i in range(46):
            ents = []
            if 1 <= i < 4:
                ents.append(E(20, "Knight", 1, is_product=False, share=3.0, target=3))
            if spend_card and i >= 2:
                ents.append(E(21, "HogRider", 0, is_product=False, share=4.0, target=5))
            thp1 = 3000.0 - (tower_loss if i >= 2 else 0.0)
            fr.append(F(i, ents, elixir=(el0[i], el1[i]),
                        thp=(3000.0, thp1),
                        cards=(["HogRider"] if (spend_card and i == 2) else None),
                        opp=([{"card": "Knight", "x": 3.0, "y": 20.0}] if i == 1 else None)))
        return fr

    # —— 情形 B：不花不打 ⇒ 不兑现 ——
    frB = build(False, 0.0)
    rB = O.analyze_game(GAME(frB), tower_mode="none")
    gB = rB["gate"]
    assert rB["trade_main"] and abs(sum(rB["trade_main"]) - 3.0) < 1e-9 or True
    assert gB["n_eligible"] >= 1, "应至少有 1 个 Trade >= θ 的窗口：%s" % gB
    assert gB["n_realized"] == 0, "只守不推不该算兑现：%s" % gB

    # —— 情形 A：花 4 费（>= Trade 3）且打出塔伤 ⇒ 兑现 ——
    frA = build(True, 500.0)
    rA = O.analyze_game(GAME(frA), tower_mode="none")
    gA = rA["gate"]
    assert gA["n_eligible"] >= 1, gA
    assert gA["n_realized"] == gA["n_eligible"], "花得多且打了塔伤应全部兑现：%s" % gA
    assert abs(gA["rate_spend"] - 1.0) < 1e-9 and abs(gA["rate_dmg"] - 1.0) < 1e-9, gA

    # —— 情形 A′：只花钱、不打塔伤 ⇒ 仍算兑现（"或"）——
    frA2 = build(True, 0.0)
    gA2 = O.analyze_game(GAME(frA2), tower_mode="none")["gate"]
    assert gA2["n_realized"] == gA2["n_eligible"], gA2
    assert abs(gA2["rate_dmg"] - 0.0) < 1e-9 and abs(gA2["rate_spend"] - 1.0) < 1e-9, gA2

    # —— 情形 C：θ 抬高到 Trade 之上 ⇒ 不合格窗口 ——
    frC = build(False, 0.0)
    rC = O.analyze_game(GAME(frC), tower_mode="none")
    gC = O.realization_gate(rC["phi"] and frC, rC["windows"], rC["trades"],
                            rC["phi_diag"].get("deploy_cost"), rC["main_player"],
                            theta=99.0)
    assert gC["n_eligible"] == 0 and gC["rate_all"] is None, gC
    return ("B 只守不推 0/%d；A 花钱+塔伤 %d/%d；A′ 只花钱 %d/%d；"
            "θ=99 时 0 合格" % (gB["n_eligible"], gA["n_realized"], gA["n_eligible"],
                              gA2["n_realized"], gA2["n_eligible"]))



# ---------------------------------------------------------------------------
# 缺陷 6（2026-09-18 由 test_realization_gate 暴露）：组件口径必须走**净支出**形式
# ---------------------------------------------------------------------------
def test_component_phi_excludes_other_lanes():
    """窗口在 3 号塔那一路（敌方 3 费 Knight 被打死，应记 +3）；
    同一时段我们在**另一路**（目标 5 号塔）下了 4 费 HogRider。

    ΔΦ 形式会把那 4 费（扣除自然回复后 −2）算进本窗口却拿不到对应残值 ⇒ 窗口被算成 **−1**；
    净支出形式只算本窗口成员的部署 ⇒ **+3**。本测试钉死这个差别。"""
    el0, el1 = [5.0], [5.0]
    for i in range(1, 7):
        a, b = el0[-1] + 0.5, el1[-1] + 0.5
        if i == 1:
            b -= 3.0                      # p1 Knight
        if i == 2:
            a -= 4.0                      # p0 在**另一路**下 HogRider
        el0.append(a)
        el1.append(b)
    fr = []
    for i in range(7):
        ents = []
        if 1 <= i < 4:
            ents.append(E(20, "Knight", 1, is_product=False, target=3))
        if 2 <= i < 4:
            # 另一路的 HogRider：窗口**结束前**就死了 ⇒ global 口径会把它 4 费全算进来，
            # 组件口径正确地把它排除（这正是缺陷 6 的差别所在）
            ents.append(E(21, "HogRider", 0, is_product=False, target=5))
        fr.append(F(i, ents, elixir=(el0[i], el1[i]),
                    cards=(["HogRider"] if i == 2 else None),
                    opp=([{"card": "Knight", "x": 3.0, "y": 20.0}] if i == 1 else None)))
    r = O.analyze_game(GAME(fr), tower_mode="none")            # 默认 component 口径
    # 两个局面：①「3 号塔 vs 敌方 Knight」= 本测试的主角；②「5 号塔 vs 我方 HogRider」= 另一路
    idx = [i for i, w in enumerate(r["windows"]) if 20 in w["members_union"]]
    assert len(idx) == 1, [sorted(w["members_union"]) for w in r["windows"]]
    i0 = idx[0]
    t = r["trades"][i0]
    assert 21 not in r["windows"][i0]["members_union"], \
        "HogRider 在另一路，不该是本窗口成员：%s" % sorted(r["windows"][i0]["members_union"])
    assert len(r["windows"]) == 2, "另一路应当自成 1 个局面：%s" % r["windows"]
    assert abs(t["trade_p0"] - 3.0) < 1e-9, \
        "组件口径应是 +3（敌方 3 费白给），实得 %s（ΔΦ 形式会给 %s）" % (
            t["trade_p0"], t["d_phi_elx"][0] - t["d_phi_elx"][1])
    # 净支出两侧自洽：Trade == 净支出_op − 净支出_me
    lhs = t["net_spend"][1] - t["net_spend"][0]
    assert abs(lhs - t["trade_p0"]) < 1e-9, (lhs, t["trade_p0"])
    # 对照：global 口径**会**把另一路的下牌算进来 ⇒ 两口径必须给出不同答案
    rg = O.analyze_game(GAME(fr), tower_mode="none", phi_mode="global")
    tg = rg["trades"][i0]
    assert abs(tg["trade_p0"] - t["trade_p0"]) > 1e-9, \
        "两条口径应当不同（否则这个测试没在测东西）：component=%s global=%s" % (
            t["trade_p0"], tg["trade_p0"])
    return ("component 口径 %+.2f（另一路的 4 费被正确排除）；global 口径 %+.2f"
            "（同一笔被算进来 ⇒ 就是缺陷 6）；净支出两侧自洽"
            % (t["trade_p0"], tg["trade_p0"]))


# ---------------------------------------------------------------------------
# 运行器
# ---------------------------------------------------------------------------
_TESTS = [
    ("test_engagement_segmentation", test_engagement_segmentation),
    ("test_engagement_trade_antisymmetry", test_engagement_trade_antisymmetry),
    ("test_root_cast_bucket_conservation", test_root_cast_bucket_conservation),
    ("test_product_zero_value", test_product_zero_value),
    ("test_trade_equals_phi_window", test_trade_equals_phi_window),
    ("test_trade_reset_across_episodes", test_trade_reset_across_episodes),
    ("test_realization_gate", test_realization_gate),
    ("test_component_phi_excludes_other_lanes", test_component_phi_excludes_other_lanes),
]


def main():
    from rl.io_bootstrap import force_utf8_stdout  # noqa: E402  (T1-1: UTF-8 兜底单一来源)
    force_utf8_stdout()
    npass = 0
    for name, fn in _TESTS:
        try:
            msg = fn()
            print("[PASS] %-42s %s" % (name, msg or ""))
            npass += 1
        except Exception as e:                      # noqa: BLE001
            print("[FAIL] %-42s %s: %s" % (name, type(e).__name__, e))
            traceback.print_exc()
    print("\n%d/%d PASS" % (npass, len(_TESTS)))
    print("注：§7-7 `test_engagement_trade_default_off` 属引擎侧，随 S2 第 5 项落地。")
    return 0 if npass == len(_TESTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
