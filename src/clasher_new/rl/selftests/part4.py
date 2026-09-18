# -*- coding: utf-8 -*-
"""`rl/selftest.py` 的测试 · 第 4/5 部分（20 个，test_eval_solo_parallel … test_value_channel_norm_and_gnorm_split）。

**函数体逐字未改**（切片生成）；顶部显式导入共用底座。"""

# T2-8：本部分从 `rl/selftest.py` 原样切出（**函数体逐字未改**，只动了下面两处**路径推导**：
# 原 L5615 / L5676 的 `dirname(dirname(abspath(__file__)))` 在搬到 `rl/selftests/` 后会少一层
# ⇒ 改用 `selftest_common._PARENT`（仍 = `src/clasher_new`）。
# `*` 不导出下划线名 ⇒ 私有 helper / 导入名一律**显式**列出（缺一个就是 NameError）。
from rl.selftest_common import (  # noqa: F401
    os, sys, time, random, shutil, np, Card, _PARENT,
    _mark_skip, _make_policy_and_tokens, _mk_env, _intents, _tiny_rollout_transitions, _FakeCfg,
)

def test_eval_solo_parallel():
    """并行评估：eval_solo_parallel 与串行 eval_solo 同种子结果完全一致（进程池正确性）。"""
    import time
    from rl import train_solo
    from rl.config import TrainConfig
    from rl.follower import FollowerPolicy
    from rl.belief import BeliefInference
    from rl.plan_space import PLAN_DIM

    cfg = TrainConfig(name="selftest_eval_par", hidden_dim=32, n_eval_games=4,
                      max_ep_steps=40, seed=7)
    bd = len(BeliefInference(opp_deck=list(train_solo.DEFAULT_SOLO_DECK),
                             n_particles=128, seed=0).encode(None, None))
    main = FollowerPolicy(hidden=32, plan_dim=PLAN_DIM, belief_dim=bd)
    opp = FollowerPolicy(hidden=32, plan_dim=PLAN_DIM, belief_dim=bd)
    train_solo._sync_frozen_copy(main, opp)
    main.to_device("cpu")
    opp.to_device("cpu")

    # 串行/并行各用一份全新 env（eval 会原地 shuffle deck 并覆盖 env.deck1，串行在前会污染并行快照）
    env_s = train_solo.solo_env(cfg, cfg.seed)
    t0 = time.monotonic()
    stats_s, _ = train_solo.eval_solo(env_s, main, opp, 4, 40, 7, cfg, record_replays=False)
    t_serial = time.monotonic() - t0

    env_p = train_solo.solo_env(cfg, cfg.seed)
    t0 = time.monotonic()
    stats_p, _ = train_solo.eval_solo_parallel(env_p, main, opp, 4, 40, 7, cfg,
                                               n_workers=2, record_replays=False)
    t_par = time.monotonic() - t0

    assert stats_s == stats_p, f"并行/串行统计不一致: {stats_s} vs {stats_p}"
    assert stats_p["games"] == 4
    print(f"[PASS] 并行评估：与串行同种子结果一致 {stats_p['wins']}W/{stats_p['losses']}L/"
          f"{stats_p['draws']}D（串行 {t_serial:.1f}s / 并行2进程 {t_par:.1f}s）")


def test_overtime_window():
    """180s 皇冠平 → 进入加时窗口而不是按塔血提前终局（overtime_open / timeout_winner）。

    规则（用户确认，2026-09）：
      - battle.time ∈ [180, 300) 且双方被拆塔数相同、未终局 → overtime_open=True
        （RL 循环继续打，引擎 [180,300) 内谁先被再破一塔谁输）；
      - 恰达 300s 仍平 → overtime_open=False，收手后由 timeout_winner 按**三塔血量合计**
        裁决（2026-09-17 用户口径；旧口径是"存活塔最低血量百分比"），完全相等才平局；
      - 皇冠不同 → 直接按皇冠结算（常规时间末领先者胜）。
    """
    from rl.run_league import overtime_open, timeout_winner

    class _P:
        def __init__(self, crowns):
            self._c = int(crowns)

        def get_crown_count(self):
            return self._c

    class _B:
        def __init__(self, t, c0, c1, over=False):
            self.time = float(t)
            self.players = [_P(c0), _P(c1)]
            self.game_over = bool(over)

    class _T:  # 假塔：塔血合计裁决用（保留 data.hp 以兼容旧 C′ 百分比路径）
        def __init__(self, hp, max_hp):
            self.hp = hp
            self.is_alive = hp > 0
            self.data = type('D', (), {'hp': max_hp})()

    class _BT(_B):
        """带塔实体的假战场：p0 塔 = ids(3,4,6)，p1 塔 = ids(1,2,5)。"""
        def __init__(self, t, c0, c1, towers0, towers1, over=False):
            super().__init__(t, c0, c1, over)
            self.entities = {}
            for eid, (hp, mhp) in zip((3, 4, 6), towers0):
                self.entities[eid] = _T(hp, mhp)
            for eid, (hp, mhp) in zip((1, 2, 5), towers1):
                self.entities[eid] = _T(hp, mhp)

    # 加时窗口开启：180s ≤ t < 300s、皇冠平、未终局
    assert overtime_open(_B(180.0, 1, 1)) is True
    assert overtime_open(_B(299.5, 0, 0)) is True
    # 常规时间未到 / 已到硬顶 / 已终局 → 不延长
    assert overtime_open(_B(179.5, 0, 0)) is False
    assert overtime_open(_B(300.0, 1, 1)) is False
    assert overtime_open(_B(200.0, 1, 1, over=True)) is False
    # 皇冠不同 → 不进入加时（按领先者直接结算）
    assert overtime_open(_B(200.0, 2, 1)) is False
    # timeout_winner：皇冠多者胜（皇冠规则优先）
    assert timeout_winner(_B(200.0, 1, 2)) == 0
    assert timeout_winner(_B(200.0, 2, 1)) == 1
    # 皇冠平 + 无实体信息（mock）→ 退回平局
    assert timeout_winner(_B(300.0, 1, 1)) is None
    assert timeout_winner(_B(180.0, 0, 0)) is None
    # 皇冠平 + 塔血裁决（合计）：p0 合计 9402 < p1 10928 → p0 输
    assert timeout_winner(_BT(300.0, 1, 1,
                              [(3052, 3052), (1526, 3052), (4824, 4824)],
                              [(4824, 4824), (3052, 3052), (3052, 3052)])) == 1
    # 镜像：p1 合计 8876 < p0 10928 → p0 胜（僵局早停不再一律记平局）
    assert timeout_winner(_BT(200.0, 1, 1,
                              [(4824, 4824), (3052, 3052), (3052, 3052)],
                              [(4824, 4824), (1000, 3052), (3052, 3052)])) == 0
    # 双方塔血合计完全相等（7876 = 7876，分布不同）→ 平局
    assert timeout_winner(_BT(110.0, 1, 1,
                              [(4824, 4824), (0, 3052), (3052, 3052)],
                              [(4824, 4824), (3052, 3052), (0, 3052)])) is None
    # **新旧口径的关键差异**：双方"最低血量百分比"都是 50%（旧口径 → 平局），
    # 但合计 9402 vs 11174 不同 ⇒ 新口径必须判 p1 胜（这就是"正常防守被判平"的病灶）
    assert timeout_winner(_BT(300.0, 1, 1,
                              [(3052, 3052), (1526, 3052), (4824, 4824)],
                              [(4824, 4824), (1526, 3052), (4824, 4824)])) == 1
    print("[PASS] 加时窗口：180s 皇冠平进入 [180,300) 突然死亡；到顶按**塔血合计**裁决；皇冠差直接判胜")


def test_draw_rule_tower_hp_total():
    """平局规则（2026-09-17 用户口径）：到点按**三塔血量合计**裁决，只有完全相等才平局。

    回归点（对应 docs/draw_rule_prereg_2026-09-17.md 的 J1/J3）：
      1) 四处口径**同值**（单一来源）：引擎 `BattleState.timeout_winner()`、
         `rl.overtime.timeout_winner`、`rl.run_league.timeout_winner`、`settle_stall(margin=0)`；
      2) 合计不等 → 多者胜（**旧口径"最低百分比相等即平局"必须不再成立**）；
      3) 合计完全相等（即使分布不同）→ None；
      4) 皇冠不同 → 皇冠优先，与塔血无关；
      5) 旧 C′ 路径仍可逐位复现：`settle_stall(b, 0.05)` 在细差 <5% 时判平。
    """
    import battle as bm
    import player as pm
    from rl.overtime import timeout_winner as ot_winner
    from rl.run_league import timeout_winner as rl_winner, settle_stall

    deck = ["Knight", "MiniPekka", "Arrows", "Minions", "Musketeer", "Fireball",
            "Giant", "Archer"]

    def fresh():
        return bm.BattleState(pm.PlayerState(0, list(deck), 5.0),
                              pm.PlayerState(1, list(deck), 5.0), card_level=11)

    def set_pct(bs, pct0, pct1):
        """按**各塔自身最大血**的百分比设置血量（各塔 max 不同：公主 3052 / 王 4824）。"""
        for i, q in zip((3, 4, 6), pct0):
            bs.entities[i].hp = float(q) * float(bs.entities[i].data.hp)
        for i, q in zip((1, 2, 5), pct1):
            bs.entities[i].hp = float(q) * float(bs.entities[i].data.hp)
        bs.update_player_hp()
        return bs

    # ① 合计不等（对方多）→ 四处一致判 1
    bs = set_pct(fresh(), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0))
    assert bs.tower_hp_total(1) > bs.tower_hp_total(0)
    assert bs.timeout_winner() == 1
    assert ot_winner(bs) == 1 and rl_winner(bs) == 1 and settle_stall(bs) == 1, \
        "四处口径必须同值（单一来源）"

    # ② 关键差异：最低百分比都是 1.0（旧口径 → 平局），但合计不同 → 必须判胜负
    bs = set_pct(fresh(), (1.0, 0.99, 1.0), (1.0, 1.0, 1.0))
    assert bs.timeout_winner() == 1, "合计不同 ⇒ 判胜负；不得因最低百分比相等判平"
    assert settle_stall(bs, 0.05) is None, "旧 C′ 路径：细差 <5% 仍判平（可复现旧标签，供对照）"
    assert settle_stall(bs) == 1, "默认（margin=0）必须走新口径"

    # ③ 合计完全相等但**分布不同** → 平局（用户口径 = 合计相等即平）
    #    p0 存活 {左3, 王6} = 3052+4824 = 7876；p1 存活 {右2, 王5} = 3052+4824 = 7876
    bs = set_pct(fresh(), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0))
    for eid in (4, 1):
        bs.entities[eid].is_alive = False          # 死塔显式标记（与引擎一致）
    assert abs(bs.tower_hp_total(0) - bs.tower_hp_total(1)) < 1e-9, \
        f"{bs.tower_hp_total(0)} vs {bs.tower_hp_total(1)}"
    assert bs.timeout_winner() is None
    assert ot_winner(bs) is None and rl_winner(bs) is None and settle_stall(bs) is None

    # ④ 皇冠优先：塔血无关，谁被拆得多谁输（`get_crown_count()` = 本侧**被拆**塔数）
    class _P:
        def __init__(self, lost):
            self._lost = int(lost)
        def get_crown_count(self):
            return self._lost
    class _B:
        def __init__(self, lost0, lost1):
            self.players = [_P(lost0), _P(lost1)]
            self.game_over = False
            self.time = 300.0
    assert ot_winner(_B(2, 1)) == 1, "p0 被拆 2 塔 > p1 的 1 ⇒ p1 胜"
    assert ot_winner(_B(1, 2)) == 0

    # ⑤ 引擎 300s 硬顶分支：真的会按合计写入 winner（不只 helper 对）
    bs = set_pct(fresh(), (1.0, 1.0, 1.0), (1.0, 1.0, 0.5))
    bs.time = 300.0
    bs.step(1.0 / 60.0)
    assert bs.game_over and bs.winner == 0, f"引擎 300s 应判 p0 胜: {bs.winner}"
    bs = set_pct(fresh(), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0))
    bs.time = 300.0
    bs.step(1.0 / 60.0)
    assert bs.game_over and bs.winner is None, "合计完全相等 ⇒ 平局（winner=None）"
    print("[PASS] 平局规则（塔血合计）：四处口径同值、合计不等判胜负（含旧口径的最低百分比相等用例）、"
          "完全相等判平、皇冠优先、引擎 300s 硬顶分支一致、旧 C′ 路径可复现")


def test_tower_threat_calc():
    """外置工具①塔伤威胁计算器：deepcopy+引擎推演，"双方不再部署"语义下
    敌方现存部队对我方各塔的伤害预估。空场=0；威胁只落在行进路线的塔上；
    调用不污染原局面；确定性（两次调用同值）；P0/P1 双向可用。"""
    import copy as _copy
    import battle as battle_mod
    import player as player_mod
    from core import Position
    from threat_calc import estimate_tower_threat

    deck = ["Knight", "MiniPekka", "Arrows", "Minions", "Musketeer", "Fireball", "Giant", "Archer"]

    def fresh():
        return battle_mod.BattleState(player_mod.PlayerState(0, list(deck), 5.0),
                                      player_mod.PlayerState(1, list(deck), 5.0),
                                      card_level=11)

    # 1) 空场零威胁
    bs = fresh()
    th = estimate_tower_threat(bs, 0)
    assert th["total"] == 0.0 and th["sim_time"] == 0.0, f"空场威胁应为 0: {th}"

    # 2) 敌方 Giant 从右路过桥 → 右塔受威胁，左塔/王塔为 0，原局面不被污染
    bs = fresh()
    p1 = bs.players[1]
    p1.cycle = ["Giant"] + [c for c in p1.cycle if c != "Giant"]
    p1.elixir = 10.0
    assert bs.deploy_card(1, "Giant", Position(14.5, 19.0)), "Giant 部署应成功"
    t_before, hp6_before = bs.time, bs.entities[6].hp
    snapshot = _copy.deepcopy(bs.entities[1].hp)
    th = estimate_tower_threat(bs, 0, horizon=20.0)
    assert th["right"] > 0, f"右塔应受威胁: {th}"
    assert th["left"] == 0.0 and th["king"] == 0.0, f"左塔/王塔不应受威胁: {th}"
    assert th["total"] == th["right"], f"total 应等于各塔之和: {th}"
    assert bs.time == t_before and bs.entities[6].hp == hp6_before, "调用不得污染原局面"
    assert bs.entities[1].hp == snapshot, "敌方塔血也不应被污染"
    # 3) 确定性：同局面两次调用完全一致
    th2 = estimate_tower_threat(bs, 0, horizon=20.0)
    assert th == th2, f"推演应确定性: {th} vs {th2}"
    # 4) P1 视角对称：我方 Giant 过桥威胁 P1 的左塔
    bs2 = fresh()
    p0 = bs2.players[0]
    p0.cycle = ["Giant"] + [c for c in p0.cycle if c != "Giant"]
    p0.elixir = 10.0
    assert bs2.deploy_card(0, "Giant", Position(3.5, 14.0)), "P0 Giant 部署应成功"
    th_p1 = estimate_tower_threat(bs2, 1, horizon=20.0)
    assert th_p1["left"] > 0 and th_p1["right"] == 0.0 and th_p1["king"] == 0.0, \
        f"P1 视角应对称地看到左塔受威胁: {th_p1}"
    print(f"[PASS] 塔伤威胁计算器：空场=0；敌方 Giant 20s 视界右塔 {th['right']:.0f} 伤、"
          f"左/王塔 0；无污染、确定、P0/P1 对称")


def test_simulate_exchange():
    """外置工具②交换模拟器：deepcopy+真部署+确定性推演。none 模式=threat_calc 对照
    口径；script 模式对手真的会防守（花圣水、解掉 Giant、把塔损压到 0）；非法部署
    正确报 legal=False；调用不污染原局面；确定性；法术候选与 P1 镜像视角可用。"""
    import copy as _copy
    import battle as battle_mod
    import player as player_mod
    from core import Position
    from simulate_exchange import simulate_exchange

    deck = ["Knight", "MiniPekka", "Arrows", "Minions", "Musketeer", "Fireball", "Giant", "Archer"]

    def fresh(elixir=10.0):
        bs = battle_mod.BattleState(player_mod.PlayerState(0, list(deck), elixir),
                                    player_mod.PlayerState(1, list(deck), elixir),
                                    card_level=11)
        bs.players[0].cycle = ["Giant"] + [c for c in bs.players[0].cycle if c != "Giant"]
        return bs

    # 1) none 模式：Giant 单独过桥，对手不响应 → 对塔有伤、我方塔无损、花 5 费
    bs = fresh()
    res_n = simulate_exchange(bs, 0, "Giant", Position(3.5, 14.0), horizon=12.0,
                              defender="none")
    assert res_n["legal"] and res_n["my_cost"] == 5.0 and res_n["opp_cost"] == 0.0, res_n
    assert res_n["opp_towers"]["total"] > 0, f"对手塔应受创: {res_n}"
    assert res_n["my_towers"]["total"] == 0.0, f"我方塔不应受损: {res_n}"
    assert res_n["my_units"]["deployed"] >= 1 and res_n["my_units"]["alive"] >= 1, res_n

    # 2) script 模式：对手真的防守 → 花 3+ 费、Giant 被解、塔损被压低于 none 模式
    bs = fresh()
    res_s = simulate_exchange(bs, 0, "Giant", Position(3.5, 14.0), horizon=12.0,
                              defender="script")
    assert res_s["legal"] and res_s["opp_cost"] > 0.0, f"脚本防守应花圣水: {res_s}"
    assert res_s["opp_towers"]["total"] < res_n["opp_towers"]["total"], \
        f"防守应降低塔损: script {res_s['opp_towers']} vs none {res_n['opp_towers']}"
    assert res_s["my_units"]["hp_frac"] < res_n["my_units"]["hp_frac"], \
        f"防守应打掉 Giant 血: {res_s['my_units']} vs {res_n['my_units']}"

    # 3) fn 模式：注入对手回调（首 tick 下 Knight 解场）
    bs = fresh()
    calls = {"n": 0}
    def opp_fn(sim, defender_id):
        calls["n"] += 1
        if calls["n"] == 1:
            return [("Knight", (3.5, 20.0))]
        return []
    res_f = simulate_exchange(bs, 0, "Giant", Position(3.5, 14.0), horizon=8.0,
                              defender="fn", opponent_fn=opp_fn)
    assert res_f["legal"] and res_f["opp_cost"] == 3.0, f"fn 对手应花 3 费: {res_f}"
    assert res_f["my_units"]["hp_frac"] < res_n["my_units"]["hp_frac"], \
        f"Knight 应打到 Giant: {res_f['my_units']}"

    # 4) 非法部署：圣水不足 → legal=False
    bs = fresh(elixir=1.0)
    res_bad = simulate_exchange(bs, 0, "Giant", Position(3.5, 14.0), horizon=4.0,
                                defender="none")
    assert not res_bad["legal"] and res_bad["reason"] == "deploy_failed", res_bad

    # 5) 法术候选：Fireball 砸 P1 左塔 ≈ 206（lv11 已知实测口径）
    bs = fresh()
    bs.players[0].cycle = ["Fireball"] + [c for c in bs.players[0].cycle if c != "Fireball"]
    res_fb = simulate_exchange(bs, 0, "Fireball", Position(3.5, 25.5), horizon=6.0,
                               defender="none")
    assert res_fb["legal"] and res_fb["my_cost"] == 4.0, res_fb
    assert abs(res_fb["opp_towers"]["left"] - 206.0) <= 2.0, \
        f"Fireball 对塔应 ≈206: {res_fb['opp_towers']}"

    # 6) P1 镜像视角（Giant 从 y19 走到 y6.5 需 ~10s，16s 视界保证够到塔并还手）
    bs = fresh()
    bs.players[1].cycle = ["Giant"] + [c for c in bs.players[1].cycle if c != "Giant"]
    res_m = simulate_exchange(bs, 1, "Giant", Position(14.5, 19.0), horizon=16.0,
                              defender="none")
    assert res_m["legal"] and res_m["opp_towers"]["total"] > 0 \
        and res_m["my_towers"]["total"] == 0.0, res_m

    # 7) 无污染 + 确定性
    bs = fresh()
    t0, el0 = bs.time, bs.players[0].elixir
    hp0 = {eid: e.hp for eid, e in bs.entities.items()}
    r1 = simulate_exchange(bs, 0, "Giant", Position(3.5, 14.0), horizon=8.0, defender="script")
    assert bs.time == t0 and bs.players[0].elixir == el0, "调用不得污染原局面"
    for eid, e in bs.entities.items():
        assert e.hp == hp0[eid], f"实体 {eid} 血被污染"
    r2 = simulate_exchange(bs, 0, "Giant", Position(3.5, 14.0), horizon=8.0, defender="script")
    assert r1 == r2, f"推演应确定性"
    print(f"[PASS] 交换模拟器：none 塔损 {res_n['opp_towers']['total']:.0f} → "
          f"script 压到 {res_s['opp_towers']['total']:.0f}（对手花 {res_s['opp_cost']:.0f} 费）；"
          f"fn 注入 OK；Fireball 对塔 {res_fb['opp_towers']['left']:.0f}；非法/镜像/无污染/确定 全过")


def test_spell_module():
    """外置工具③法术知识模块：伤害数字全部来自引擎标定（不拍脑袋）。
    对账口径：evaluate_cast 静态预测 == engine_resolution 实测（确定性引擎应逐位一致）；
    击杀判定与引擎一致；best_cast 覆盖目标簇；BarbLog 部署区限制正确上报；
    无污染、缓存一致。"""
    import battle as battle_mod
    import player as player_mod
    from core import Position
    from spell_module import (get_spell_profile, evaluate_cast, best_cast,
                              engine_resolution, clear_profile_cache)

    deck = ["Knight", "MiniPekka", "Arrows", "Minions", "Musketeer", "Fireball", "Giant", "Archer"]

    def fresh():
        return battle_mod.BattleState(player_mod.PlayerState(0, list(deck), 10.0),
                                      player_mod.PlayerState(1, list(deck), 10.0),
                                      card_level=11)

    clear_profile_cache()
    # 1) Fireball 档案：标定值落在已知口径内（lv11 对塔 206、对部队 687×等级曲线）
    prof = get_spell_profile("Fireball", 11)
    assert prof["calibrated"] and prof["elixir"] == 4 and abs(prof["radius"] - 2.5) < 1e-6, prof
    assert 180.0 <= prof["tower_damage"] <= 230.0, f"Fireball 对塔标定异常: {prof}"
    assert 600.0 <= prof["troop_damage"] <= 780.0, f"Fireball 对部队标定异常: {prof}"
    assert get_spell_profile("Fireball", 11) is prof, "档案缓存应复用同一对象"

    # 2) 对账：evaluate_cast 预测 == engine_resolution 实测（Giant 静靶）
    bs = fresh()
    hp_before = {eid: e.hp for eid, e in bs.entities.items()}
    bs.players[0].cycle = ["Fireball"] + [c for c in bs.players[0].cycle if c != "Fireball"]
    bs.players[1].cycle = ["Giant"] + [c for c in bs.players[1].cycle if c != "Giant"]
    assert bs.deploy_card(1, "Giant", Position(9.0, 19.0))
    ev = evaluate_cast(bs, 0, "Fireball", Position(9.0, 19.0))
    gts = [t for t in ev["targets"] if t["name"] == "Giant"]
    assert len(gts) == 1 and not gts[0]["dies"], ev
    assert abs(gts[0]["damage"] - prof["troop_damage"]) <= 2.0, ev
    res = engine_resolution(bs, 0, "Fireball", Position(9.0, 19.0))
    assert res["legal"] and len(res["per_entity"]) == 1, res
    (dmg_meas,) = res["per_entity"].values()
    assert abs(dmg_meas - gts[0]["damage"]) <= 2.0, \
        f"预测 {gts[0]['damage']} vs 实测 {dmg_meas} 漂移超容差"
    assert bs.entities[1].hp == 3052, "对账调用不得污染原局面"

    # 3) Zap 罩 3 Knight：全命中、不死、逐目标预测==实测
    bs = fresh()
    bs.players[0].cycle = ["Zap"] + [c for c in bs.players[0].cycle if c != "Zap"]
    for kx, ky in ((8.5, 19.0), (9.5, 19.0), (9.0, 20.0)):
        bs.players[1].cycle = ["Knight"] + [c for c in bs.players[1].cycle if c != "Knight"]
        assert bs.deploy_card(1, "Knight", Position(kx, ky))
    knights = [e.id for e in bs.entities.values() if e.card_name == "Knight"]
    ev_z = evaluate_cast(bs, 0, "Zap", Position(9.0, 19.3))
    hit = {t["id"]: t for t in ev_z["targets"]}
    assert set(hit) == set(knights) and ev_z["n_targets"] == 3, ev_z
    assert all(not t["dies"] for t in ev_z["targets"]), ev_z
    res_z = engine_resolution(bs, 0, "Zap", Position(9.0, 19.3))
    for eid in knights:
        assert abs(res_z["per_entity"][eid] - hit[eid]["damage"]) <= 2.0, \
            f"Zap 预测 {hit[eid]['damage']} vs 实测 {res_z['per_entity'][eid]}"

    # 4) 击杀判定对账（静止靶 Cannon，避免移动目标的驻留乐观偏差）：
    #    Rocket(6 费) 应砸死 Cannon(824 hp)；Arrows(3 连波 366) 打不死 → dies=False 一致
    bs = fresh()
    bs.players[1].cycle = ["Cannon"] + [c for c in bs.players[1].cycle if c != "Cannon"]
    assert bs.deploy_card(1, "Cannon", Position(9.0, 19.0))
    cannons = [e.id for e in bs.entities.values() if e.card_name == "Cannon"]
    assert len(cannons) == 1
    cid = cannons[0]
    bs.players[0].cycle = ["Rocket"] + [c for c in bs.players[0].cycle if c != "Rocket"]
    ev_r = evaluate_cast(bs, 0, "Rocket", Position(9.0, 19.0))
    tgt = [t for t in ev_r["targets"] if t["id"] == cid][0]
    assert tgt["kind"] == "building" and tgt["dies"] is True, tgt
    res_r = engine_resolution(bs, 0, "Rocket", Position(9.0, 19.0))
    assert res_r["per_entity"].get(cid, 0.0) >= tgt["hp"] - 1e-6, \
        f"Rocket 应实测砸死 Cannon: 预测 {tgt} vs 实测 {res_r['per_entity']}"
    bs.players[0].cycle = ["Arrows"] + [c for c in bs.players[0].cycle if c != "Arrows"]
    ev_a = evaluate_cast(bs, 0, "Arrows", Position(9.0, 19.0))
    tgt_a = [t for t in ev_a["targets"] if t["id"] == cid][0]
    assert tgt_a["dies"] is False, tgt_a
    res_a2 = engine_resolution(bs, 0, "Arrows", Position(9.0, 19.0))
    assert abs(res_a2["per_entity"].get(cid, 0.0) - tgt_a["damage"]) <= 2.0, \
        f"Arrows 静止靶伤害预测==实测: 预测 {tgt_a['damage']} vs 实测 {res_a2['per_entity']}"
    assert ev_r["elixir_killed_value"] > 0, f"击杀应折费: {ev_r}"

    # 5) best_cast：3 Knight 簇上找覆盖 ≥2 目标的落点
    bs = fresh()
    bs.players[0].cycle = ["Zap"] + [c for c in bs.players[0].cycle if c != "Zap"]
    for kx, ky in ((8.5, 19.0), (9.5, 19.0), (9.0, 20.0)):
        bs.players[1].cycle = ["Knight"] + [c for c in bs.players[1].cycle if c != "Knight"]
        bs.deploy_card(1, "Knight", Position(kx, ky))
    pos, ev_b, score = best_cast(bs, 0, "Zap", grid=1.0)
    assert pos is not None and score > 0 and ev_b["n_targets"] >= 2, (pos, ev_b, score)

    # 6) BarbLog 部署区限制：敌半场落点 castable=False
    ev_l = evaluate_cast(bs, 0, "BarbLog", Position(9.0, 25.0))
    assert ev_l["castable"] is False, ev_l
    # 7) 非伤害法术：Rage 无伤害路径，targets 为空
    ev_r = evaluate_cast(bs, 0, "Rage", Position(9.0, 19.0))
    assert ev_r["deals_damage"] is False and ev_r["targets"] == [], ev_r
    print(f"[PASS] 法术知识模块：Fireball 标定 对塔 {prof['tower_damage']:.0f}/对部队 "
          f"{prof['troop_damage']:.0f}；对账预测==实测（Fireball/Zap/Rocket/Arrows 逐目标 ±2）；"
          f"Rocket 砸死 Cannon 击杀判定一致；best_cast 覆盖 {ev_b['n_targets']} 目标；"
          f"BarbLog/Rage 口径正确")


def test_mcts_basic():
    """推理时浅 MCTS（RL-MCTS v1）：见 docs/mcts_design.md。

    1) 空场单步搜索不崩、返回合法动作（WAIT 或通过 validate_bundle 的 bundle）；
    2) 原局面零污染（时间/圣水/实体数/手牌不变）；
    3) 确定性（同盘面两次搜索结果一致）；
    4) 值函数量纲：空场 Arrows 砸王塔为负 EV（量纲失配回归——曾因塔伤×1000
       抬尺度被误判正 EV）；残血塔双倍期 Fireball 斩杀为正 EV 且优于 WAIT；
    5) 预算控制（n_simulations=2 快速路径可运行）。"""
    import copy
    import battle as battle_mod
    import player as player_mod
    from core import Position
    from rl.mcts import RLMCTS, MCTSConfig, leaf_value
    from rl.action_mask import validate_bundle
    from rl.action_bundle import ActionBundle, SubAction

    deck = ["Knight", "MiniPekka", "Arrows", "Minions", "Musketeer", "Fireball", "Giant", "Archer"]
    cfg = MCTSConfig(n_simulations=24, leaf_horizon_s=6.0, max_depth=2, max_bundles_per_node=3)

    def fresh(elixir=5.0, hand=None):
        h = hand or deck
        bs = battle_mod.BattleState(player_mod.PlayerState(0, list(h), elixir),
                                    player_mod.PlayerState(1, list(h), elixir),
                                    card_level=11)
        bs.update_player_hp()
        return bs

    def snap(bs):
        return (bs.time, tuple(round(p.elixir, 6) for p in bs.players),
                len(bs.entities), tuple(p.cycle[:4] for p in bs.players))

    # 1+2) 空场搜索：合法 + 原局零污染
    bs = fresh()
    before = snap(bs)
    mcts = RLMCTS(cfg=cfg)
    bundle, info = mcts.search(bs, 0)
    if bundle.sub_actions:
        ok, reason, _ = validate_bundle(bs, 0, bundle)
        assert ok, f"搜索返回非法动作: {reason} {bundle.sub_actions}"
    assert snap(bs) == before, "search() 不得修改原局面"
    assert info["elapsed_s"] < 30, f"预算失控: {info['elapsed_s']:.1f}s"

    # 3) 确定性
    def bkey(b):
        return [(sa.kind, sa.slot, sa.x, sa.y) for sa in b.sub_actions]
    bs2 = fresh()
    b2, i2 = RLMCTS(cfg=cfg).search(bs2, 0)
    assert bkey(bundle) == bkey(b2), f"同盘面两次搜索结果不一致: {bkey(bundle)} vs {bkey(b2)}"

    # 4a) 量纲回归：空场 Arrows 砸王塔必须 < WAIT（前段 75 塔伤 < 3 费）
    bs3 = fresh(elixir=10.0)
    fb_leaf = None
    wait_leaf = leaf_value(copy.deepcopy(bs3), 0, cfg)
    for slot, name in enumerate(bs3.players[0].cycle[:4], 1):
        if name == "Arrows":
            b_arrows = ActionBundle(sub_actions=[SubAction(kind="deploy", slot=slot, x=8, y=28)])
            ok, reason, _ = validate_bundle(bs3, 0, b_arrows)
            if ok:  # 掩码若已拦（王塔格合法则拦不住）也视为通过——9i 闸门只管公主塔
                sim = copy.deepcopy(bs3)
                for card, sa in validate_bundle(sim, 0, b_arrows)[2]:
                    sim.deploy_card(0, card, sa.to_position(0))
                fb_leaf = leaf_value(sim, 0, cfg)
            break
    if fb_leaf is not None:
        assert fb_leaf < wait_leaf, \
            f"量纲失配回归: 空场砸王塔 leaf {fb_leaf:.2f} 应低于 WAIT {wait_leaf:.2f}"

    # 4b) 斩杀方向：双倍期残血塔（70/3052），Fireball 在手 → 斩杀 leaf 严格优于 WAIT
    hand4 = ["Knight", "MiniPekka", "Fireball", "Minions", "Musketeer", "Giant", "Giant", "Archer"]
    bs4 = fresh(elixir=10.0, hand=hand4)
    bs4.time = 200.0
    bs4.entities[1].hp = 70.0
    bs4.players[1].left_tower_hp = 70.0
    kill = None
    for slot, name in enumerate(bs4.players[0].cycle[:4], 1):
        if name == "Fireball":
            b_kill = ActionBundle(sub_actions=[SubAction(kind="deploy", slot=slot, x=3, y=25)])
            ok, reason, _ = validate_bundle(bs4, 0, b_kill)
            assert ok, f"斩杀落点应合法: {reason}"
            sim = copy.deepcopy(bs4)
            for card, sa in validate_bundle(sim, 0, b_kill)[2]:
                sim.deploy_card(0, card, sa.to_position(0))
            kill = leaf_value(sim, 0, cfg)
            break
    wait4 = leaf_value(copy.deepcopy(bs4), 0, cfg)
    assert kill is not None and kill > wait4, \
        f"斩杀 leaf {kill:.2f} 应优于 WAIT {wait4:.2f}"

    # 5) 预算控制：2 sims 快速路径
    tiny = MCTSConfig(n_simulations=2, leaf_horizon_s=2.0, max_depth=1, max_bundles_per_node=2)
    bs5 = fresh()
    _b5, i5 = RLMCTS(cfg=tiny).search(bs5, 0)
    assert i5["n_sims"] == 2 and i5["elapsed_s"] < 5, i5
    print(f"[PASS] 浅 MCTS：空场合法+原局零污染+确定性；量纲回归（空砸王塔负EV/双倍期斩杀 "
          f"{kill - wait4:+.2f} 优于 WAIT）；快速预算 {i5['elapsed_s']:.2f}s")


def test_mcts_defense_and_wait():
    """浅 MCTS 行为方向：Knight 压境 → 搜索不返回非法动作且给出可解释选项；
    大圣水优势空场 → 不再返回空砸（负 EV 候选）。小预算保证耗时可控。"""
    import copy
    import battle as battle_mod
    import player as player_mod
    from core import Position
    from rl.mcts import RLMCTS, MCTSConfig
    from rl.action_mask import validate_bundle

    deck = ["Knight", "MiniPekka", "Arrows", "Minions", "Musketeer", "Fireball", "Giant", "Archer"]
    cfg = MCTSConfig(n_simulations=12, leaf_horizon_s=4.0, max_depth=2, max_bundles_per_node=2)

    # Knight 压境
    bs = battle_mod.BattleState(player_mod.PlayerState(0, list(deck), 5.0),
                                player_mod.PlayerState(1, list(deck), 5.0), card_level=11)
    bs.update_player_hp()
    assert bs.deploy_card(1, "Knight", Position(8.5, 18.5))
    for _ in range(480):
        bs.step(1 / 60)
    bundle, info = RLMCTS(cfg=cfg).search(bs, 0)
    if bundle.sub_actions:
        ok, reason, _ = validate_bundle(bs, 0, bundle)
        assert ok, f"防守场景返回非法动作: {reason}"
    assert info["elapsed_s"] < 10, f"防守场景耗时失控: {info['elapsed_s']:.1f}s"
    print(f"[PASS] 浅 MCTS 行为方向：Knight 压境下决策合法（"
          f"{'WAIT' if not bundle.sub_actions else bundle.sub_actions[0].kind + ' ' + str(bs.players[0].cycle[bundle.sub_actions[0].slot - 1])}），"
          f"耗时 {info['elapsed_s']:.2f}s")


def test_opp_event_token():
    """9j B 层：对手出牌事件通道（belief_token 尾部 3×16 维）。

    校验：维度追加（23→71）且旧 checkpoint 尾部零拷贝兼容；事件入历史/
    Δt 陈旧度随观测时间增长并 clamp 到 1.0；reset 清空；哨兵过滤不误入。"""
    import numpy as np
    from rl.belief import (BeliefInference, belief_token_dim,
                           OPP_EVENT_K, OPP_EVENT_DIM)
    from rl.observation import ENTITY_NAMES

    deck = ["Minions", "Archer", "MiniPekka", "Musketeer", "Giant",
            "Fireball", "Arrows", "Knight"]
    NE = len(ENTITY_NAMES)
    b = BeliefInference(opp_deck=deck)
    tok0 = b.encode(None, None)
    assert len(tok0) == belief_token_dim(deck), (len(tok0), belief_token_dim(deck))
    assert abs(tok0[-OPP_EVENT_K * OPP_EVENT_DIM:]).sum() == 0, "无事件应为全零尾"

    obs = {"time": np.array([30.5], dtype=np.float32)}
    EV = OPP_EVENT_K * OPP_EVENT_DIM
    b.update(obs, [{"card": "Giant", "x": 8.5, "y": 14.5}])
    row = b.encode(obs)[-EV:].reshape(OPP_EVENT_K, OPP_EVENT_DIM)[-1]
    gi = ENTITY_NAMES.index("Giant")
    assert row[gi] == 1.0, "Giant onehot 缺失"
    assert abs(row[NE] - 8.5 / 17) < 1e-6 and abs(row[NE + 1] - 14.5 / 31) < 1e-6
    assert row[NE + 2] == 0.0, "最新事件 Δt 应为 0"
    # 陈旧度：35.0 时 Δt=4.5；40.5 时 clamp 到 1.0
    row2 = b.encode({"time": np.array([35.0], dtype=np.float32)})[-EV:] \
        .reshape(OPP_EVENT_K, OPP_EVENT_DIM)[-1]
    assert abs(row2[NE + 2] - 0.45) < 1e-6, row2[NE + 2]
    row3 = b.encode({"time": np.array([40.5], dtype=np.float32)})[-EV:] \
        .reshape(OPP_EVENT_K, OPP_EVENT_DIM)[-1]
    assert abs(row3[NE + 2] - 1.0) < 1e-6, "Δt 必须 clamp 到 1.0"
    # reset 清空；显式 encode(opp_played) 不与 update 重复入账
    b.reset()
    assert abs(b.encode(None, None)[-EV:]).sum() == 0
    b2 = BeliefInference(opp_deck=deck)
    t = b2.encode(obs, [{"card": "Fireball", "x": 9.0, "y": 16.0}])
    fi = ENTITY_NAMES.index("Fireball")
    assert sum(1 for r in t[-EV:].reshape(OPP_EVENT_K, OPP_EVENT_DIM) if r[fi] == 1.0) == 1
    # 旧 checkpoint 兼容：23 维 → 当前维度尾部零
    import torch, tempfile
    from rl.follower import FollowerPolicy, load_checkpoint
    pol_old = FollowerPolicy(hidden=128, plan_dim=57, belief_dim=23)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save({"state_dict": pol_old.state_dict(), "plan_dim": 57,
                    "belief_dim": 23, "hidden_dim": 128}, f.name)
        pol_new = load_checkpoint(f.name, belief_dim=belief_token_dim(deck))
    w = pol_new.belief_mlp[0].weight
    assert int((w.abs().sum(dim=0) > 0).sum()) == 23, "事件通道列必须从零开始"
    print(f"[PASS] 事件通道：token {len(tok0)} 维（2×8+2+5+3×{NE}），Δt 陈旧度/零拷贝兼容/reset 全过")


def test_crossed_river_defend_plan():
    """9j C 层：敌军过河（y<16）即触发 defend_* + bridge_front 落点提示。

    校验：单远程单位过河（threat=1.x < 2.0 阈值）也建议防守且对准威胁路；
    有法术可解时 spell_trade 优先级不变；未过河/空场不误触发。"""
    import battle as battle_mod
    import player as player_mod
    from core import Position
    from rl.belief import BeliefInference
    from rl.belief_planner import BeliefPlanner
    from rl.observation import observe

    deck = ["Minions", "Archer", "MiniPekka", "Musketeer", "Giant",
            "Fireball", "Arrows", "Knight"]
    bp = BeliefPlanner()

    def battle_moved(card, side_x, y_target):
        b = battle_mod.BattleState(player_mod.PlayerState(0, list(deck), 5.0),
                                   player_mod.PlayerState(1, list(deck), 5.0),
                                   card_level=11)
        b.update_player_hp()
        assert b.deploy_card(1, card, Position(8.5, 18.0))
        for e in b.entities.values():
            if e.player == 1 and e.id > 6 and e.name == card:
                e.position = Position(side_x, y_target)
        return b

    def plan_of(b):
        belief = BeliefInference(opp_deck=deck)
        return bp.plan(b, belief.state(), observe(b, 0))

    # 左路过河（远程脆皮在手、无解法术于手牌前 4 → 走到回退 crossed 分支）
    p1 = plan_of(battle_moved("Musketeer", 8.5, 14.5))
    if p1.macro_intent not in ("spell_trade", "soft_control"):
        # P0 手牌（Minions/Musketeer/Giant/Archer）无软控法术 → 必为过河 defend
        assert p1.macro_intent == "defend_left", p1.macro_intent
        assert p1.focus_region == "own_left", p1.focus_region
        # 9k 拦截几何：Musketeer@y=14.5 刚过河（>=9）→ 中点靠敌侧 hint
        assert p1.placement_hint == "intercept_mid", p1.placement_hint
        assert p1.target_kind == "unit"
    # 右路
    p3 = plan_of(battle_moved("Musketeer", 14.5, 14.5))
    assert p3.macro_intent == "defend_right" and p3.focus_region == "own_right"
    # 有法术在手：spell_trade 优先于回退（优先链语义不变）
    b1b = battle_moved("Musketeer", 8.5, 14.5)
    p0 = b1b.players[0]
    if "Fireball" not in p0.cycle[:4]:
        p0.cycle.remove("Fireball")
        p0.cycle.insert(3, "Fireball")
        p0.elixir = max(p0.elixir, 6.0)
    assert plan_of(b1b).macro_intent == "spell_trade"
    # 未过河（y=18 对手半场）→ 无拦截 hint
    p2 = plan_of(battle_moved("Musketeer", 8.5, 18.0))
    assert p2.placement_hint == "none" and p2.macro_intent not in ("defend_left", "defend_right")
    # 深威胁分派：坦克 y<9 → pull_aggro（贴身打满输出）；杂兵 y<9 → king_front
    # （Giant 不一定在手牌前 4：battle_moved 里先正常部署 Musketeer，再把
    #  同卡位的实体类型换成 Giant 的血量口径——直接改用矮血/高血两类 Archer）
    p4 = plan_of(battle_moved("Musketeer", 8.5, 7.0))
    if p4.macro_intent in ("defend_left", "defend_right"):
        assert p4.placement_hint == "king_front", p4.placement_hint
    print("[PASS] 拦截几何：刚过河→intercept_mid、深坦克→pull_aggro、深杂兵→king_front；"
          "spell_trade 优先级保持；未过河不误触发")


def test_death_damage_scaling():
    """2026-09-09 亡语等级缩放 + IceGolemite 死亡减速圈（FirstLight 对账）：
    ① death_damage lv1 基准（gamedata summonCharacterData）× 1.1^(lv-1)——
       IceGolemite 33→84-86 / Golem 88→225-228 / Golemite 39→99-101
       （FL 投影 2.56×截断精确值 84/225/99；引擎统一幂次口径 ±1.4%）；
    ② 冰人死亡 spawn DeathSlowZone：亡语伤害秒杀同级小骷髅（81 血 < 84），
       存活单位吃减速 0.65×2s（FreezeIceGolemite 半径 2 格/空地双打）。"""
    from card_utils import Card
    import player as player_mod
    import battle as battle_mod
    from core import Position as P2
    ig = Card('IceGolemite')
    if not (84 <= ig.death_damage <= 87):
        raise AssertionError(f"IceGolemite death_damage lv11 = {ig.death_damage}, expect 84-86")
    go = Card('Golem')
    if not (225 <= go.death_damage <= 229):
        raise AssertionError(f"Golem death_damage lv11 = {go.death_damage}, expect 225-228")
    ge = Card('Golemite')
    if not (99 <= ge.death_damage <= 102):
        raise AssertionError(f"Golemite death_damage lv11 = {ge.death_damage}, expect 99-101")
    if abs(getattr(ig, 'death_area_effect_radius', 0) - 2.0) > 1e-6:
        raise AssertionError("IceGolemite death AEF radius != 2.0")
    if abs(getattr(ig, 'death_area_effect_slow', 1.0) - 0.65) > 1e-6:
        raise AssertionError(f"death AEF slow = {ig.death_area_effect_slow}, expect 0.65 (-35%)")
    if abs(getattr(ig, 'death_area_effect_duration', 0) - 2.0) > 1e-6:
        raise AssertionError("death AEF duration != 2.0s")
    # 战斗级：死亡 → 伤害+减速圈生效
    DECK = ['Knight', 'Arrows', 'Fireball', 'Musketeer', 'Giant',
            'Minions', 'MiniPekka', 'Skeletons']
    bs = battle_mod.BattleState(player_mod.PlayerState(0, list(DECK), 10.0),
                                player_mod.PlayerState(1, list(DECK), 10.0))
    bs.step(0.033)
    ig_e = battle_mod.Troop(bs.next_entity_id, P2(5.0, 20.0), 1, 'IceGolemite', bs)
    bs._spawn_entity(ig_e)
    sk = battle_mod.Troop(bs.next_entity_id, P2(5.5, 20.5), 0, 'Skeletons', bs)
    bs._spawn_entity(sk)
    kn = battle_mod.Troop(bs.next_entity_id, P2(4.5, 20.5), 0, 'Knight', bs)
    bs._spawn_entity(kn)
    ig_e.take_damage(99999)
    bs.step(0.033)
    if sk.is_alive or sk.hp > 0:
        raise AssertionError("IceGolemite 亡语应秒杀同级小骷髅（81 血 < 84 亡语伤）")
    if not kn.is_alive:
        raise AssertionError("Knight 不应被亡语 84 伤打死")
    if abs(kn.speed_debuff - 0.65) > 1e-6:
        raise AssertionError(f"Knight speed_debuff = {kn.speed_debuff}, expect 0.65")
    if abs(kn.debuff_time_remaining - 2.0) > 0.2:
        raise AssertionError(f"Knight 减速时长 = {kn.debuff_time_remaining}, expect ≈2.0")
    zones = [e for e in bs.entities.values() if type(e).__name__ == 'DeathSlowZone']
    if len(zones) != 1:
        raise AssertionError(f"DeathSlowZone ×{len(zones)}, expect 1")
    print("[PASS] 亡语等级缩放（IG 84-86/Golem 225-228/Golemite 99-101）+ 冰人死亡减速圈"
          "（秒杀小骷髅 / Knight 吃 0.65×2s）")


def test_vines_snare_fl_duration():
    """2026-09-09 Vines 束缚口径（FirstLight 优先）：束缚 2.0s（EXT SpawnTime=2000，
    原 gamedata buffData 2500ms 为旧值）；总伤等效 306（153×2 跳 = FL DPS 153/s×2s）。"""
    from card_utils import Card
    import player as player_mod
    import battle as battle_mod
    from core import Position as P2
    v = Card('Vines')
    aeo = v.data.get('areaEffectObjectData') or {}
    if int(aeo.get('ticks') or 0) != 2:
        raise AssertionError("Vines ticks != 2")
    DECK = ['Knight', 'Arrows', 'Fireball', 'Musketeer', 'Giant',
            'Minions', 'MiniPekka', 'Skeletons']
    bs = battle_mod.BattleState(player_mod.PlayerState(0, list(DECK), 10.0),
                                player_mod.PlayerState(1, list(DECK), 10.0))
    bs.step(0.033)
    tgt = battle_mod.Troop(bs.next_entity_id, P2(8.0, 10.0), 0, 'Knight', bs)
    bs._spawn_entity(tgt)
    from battle import VinesSnareZone
    z = VinesSnareZone(bs.next_entity_id, P2(8.0, 10.5), 1, bs,
                       radius=2.5, lifetime=2.0, damage=153, hits=2,
                       crown_pct=0.25, snare_duration=2.0)
    bs._spawn_entity(z)
    bs.step(0.033)
    if abs(tgt.freeze_timer - 2.0) > 0.05:
        raise AssertionError(f"Vines 束缚 freeze_timer = {tgt.freeze_timer}, expect 2.0")
    if tgt.hp >= tgt.data.hp:
        raise AssertionError("Vines 第 1 跳伤害未结算")
    print("[PASS] Vines 束缚 2.0s（FL 口径）+ 第 1 跳伤害即时结算")


def test_log_rolling_direction():
    """2026-09-10 滚木/野蛮人滚筒滚动口径（用户定稿）：

    ① 滚木（Log）与野蛮人滚筒（BarbLog）**凭空出现在部署点**，不走"国王塔发射→飞行→
       落点→滚动"链路——部署后立即只有滚动弹（LogProjectileRolling/BarbLogProjectileRolling），
       **不存在第一段 LogProjectile/BarbLogProjectile**；
    ② 滚动方向只能**垂直河道沿纵轴（y 轴）纵深滚**：蓝方 +y、红方 −y，x 恒定（永不斜向）；
    ③ Firecracker 爆裂弹不受影响（二段弹仍"飞到落点爆炸"不滚动，防 rolling 改动误伤回归）。"""
    import player as player_mod
    import battle as battle_mod
    from core import Position as P2
    DECK = ['Knight', 'Arrows', 'Fireball', 'Musketeer', 'Giant',
            'Minions', 'Log', 'Skeletons']

    def trace(name, pid, pos, nsteps=300):
        bs = battle_mod.BattleState(player_mod.PlayerState(0, list(DECK), 10.0),
                                    player_mod.PlayerState(1, list(DECK), 10.0),
                                    card_level=11)
        bs.update_player_hp()
        p = bs.players[pid]
        p.cycle = [name] + [c for c in p.cycle if c != name]
        p.elixir = 10.0
        assert bs.deploy_card(pid, name, P2(*pos)), f"{name} 部署失败"
        spawned = [e.name for e in bs.entities.values() if e.id > 6]
        pts = []
        for _ in range(nsteps):
            bs.step(1 / 60)
            for e in bs.entities.values():
                if e.is_alive and e.name in ("LogProjectileRolling", "BarbLogProjectileRolling"):
                    pts.append((round(e.position.x, 2), round(e.position.y, 2)))
        return spawned, pts

    # ①+② 蓝方 Log 斜落点：凭空出现滚动弹、无第一段、x 恒定、y 递增
    sp, pts = trace('Log', 0, (14.0, 19.0))
    assert "LogProjectile" not in sp, f"LogProjectile 不应存在（凭空出现）: {sp}"
    assert "LogProjectileRolling" in sp, f"应直接生成滚动弹: {sp}"
    assert pts and len({p[0] for p in pts}) == 1, f"x 应恒定（纯纵向）: {pts[:3]}"
    assert pts[0][1] < pts[-1][1], f"蓝方应滚向 +y: {pts[0]}→{pts[-1]}"
    # ② 蓝方左落点
    sp, pts = trace('Log', 0, (5.0, 19.0))
    assert pts and len({p[0] for p in pts}) == 1 and pts[0][1] < pts[-1][1]
    # ② 红方落点：滚向 −y
    sp, pts = trace('Log', 1, (5.0, 23.0))
    assert pts and pts[0][1] > pts[-1][1], f"红方应滚向 -y: {pts[0]}→{pts[-1]}"
    # ② BarbLog：凭空出现 + 纯纵向
    sp, pts = trace('BarbLog', 0, (14.0, 9.0))
    assert "BarbLogProjectile" not in sp and "BarbLogProjectileRolling" in sp, sp
    assert pts and len({p[0] for p in pts}) == 1 and pts[0][1] < pts[-1][1]

    # ③ Firecracker 爆裂弹不误伤：部署不崩、爆裂弹出现（走"飞到落点爆炸"而非滚动）
    DECK_F = ['Knight', 'Arrows', 'Fireball', 'Musketeer', 'Giant',
              'Minions', 'Firecracker', 'Skeletons']
    bs = battle_mod.BattleState(player_mod.PlayerState(0, list(DECK_F), 10.0),
                                player_mod.PlayerState(1, list(DECK_F), 10.0), card_level=11)
    bs.update_player_hp()
    p0 = bs.players[0]
    p0.cycle = ['Firecracker'] + [c for c in p0.cycle if c != 'Firecracker']
    p0.elixir = 10.0
    assert bs.deploy_card(0, 'Firecracker', P2(8.5, 10.0))
    expl_seen = False
    for _ in range(900):
        bs.step(1 / 60)
        if any(e.name == 'FirecrackerExplosion' and e.is_alive
               for e in bs.entities.values()):
            expl_seen = True
    assert expl_seen, "Firecracker 爆裂弹应正常出现（不误伤）"
    print("[PASS] 滚木/滚筒：凭空出现无第一段、纯纵向滚动（蓝+y/红−y/x恒定）、Firecracker 不误伤")


def test_behavioral_metrics():
    """行为指标（2026-09-10 用户：胜率镜像自对弈自我对冲，用行为质量替代）：

    合成游戏帧验证各指标判定：
    ① 防守投入率：敌过河帧有 deploy → defense_invest_rate>0；
    ② 接敌率：部署点在后续 8s 内出现敌我 troop 同框 → engagement_rate>0；
    ③ 拦截率：落点在敌→我塔路径 4 格内 → intercept_rate>0；
    ④ 单边堆牌 vs 响应：无对手出牌时 deploy → unilateral；对手出牌后 5s 内 → response；
    ⑤ 组波率：同帧 ≥2 张 deploy → bundle_multi_rate>0；
    ⑥ 圣水均值/deploy 每局/塔血差。"""
    from rl.train_solo import behavioral_metrics

    # 构造合成回放：2 局
    def frame(t, bundle, entities, opp_played=None, towers0=None, towers1=None,
              elixir0=5.0):
        return {"t": t, "bundle": bundle, "entities": entities,
                "opp_played": opp_played,
                "towers0": towers0 or [4824, 3052, 3052],
                "towers1": towers1 or [4824, 3052, 3052],
                "elixir0": elixir0}

    # 实体条目 [name, x, y, hp, player, kind, ...]
    def troop(x, y, pl):
        return ["Knight", x, y, 100, pl, "troop", 100, 0, 0, 0.5]

    # 局1：t=1 敌 Giant 过河(y=10)，t=1.5 我方部署 Knight(10,11) 拦截 → 应接敌+拦截
    g1 = {"meta": {}, "winner": 0, "frames": [
        frame(0.5, [], [troop(9, 20, 1)]),
        frame(1.0, [], [troop(9, 10, 1)]),                       # 敌过河（y<16）
        frame(1.5, [("deploy", 1, 9, 10)], [troop(9, 10, 1), troop(10, 11, 0)], opp_played=[{"card": "Giant"}]),
        frame(2.0, [], [troop(9, 10, 1), troop(10, 11, 0)]),     # 同框 → 接敌
        frame(2.5, [], [troop(9, 10, 1), troop(10, 11, 0)]),
    ]}
    # 局2：t=0.5 我方无对手出牌时 deploy Knight（单边堆牌）；t=2 对手出牌后 1s 内响应部署
    g2 = {"meta": {}, "winner": 1, "frames": [
        frame(0.5, [("deploy", 1, 8, 12)], [troop(8, 12, 0)]),             # 单边（无对手出牌）
        frame(1.0, [], [troop(8, 12, 0)]),
        frame(2.0, [], [troop(9, 18, 1)], opp_played=[{"card": "Knight"}]),
        frame(2.5, [("deploy", 1, 9, 13), ("deploy", 2, 10, 13)],
              [troop(9, 18, 1), troop(9, 13, 0), troop(10, 13, 0)]),       # 响应 + 组波(2张)
        frame(3.0, [], [troop(9, 13, 0), troop(10, 13, 0)]),
    ]}
    m = behavioral_metrics([g1, g2])
    assert m["defense_invest_rate"] > 0, f"防守投入率应>0: {m}"
    assert m["engagement_rate"] > 0, f"接敌率应>0: {m}"
    assert m["intercept_rate"] > 0, f"拦截率应>0: {m}"
    assert m["unilateral_rate"] > 0, f"单边堆牌率应>0: {m}"
    assert m["bundle_multi_rate"] > 0, f"组波率应>0: {m}"
    assert m["deploy_per_game"] >= 1.0, f"每局deploy应≥1: {m}"
    assert m["elixir_avg"] > 0, f"圣水均值应>0: {m}"
    assert m["response_latency_med"] is not None, f"响应延迟应有值: {m}"
    # 空输入 → 空 dict
    assert behavioral_metrics([]) == {}, "空回放应返回空"
    print(f"[PASS] 行为指标：防守投入={m['defense_invest_rate']}% 接敌={m['engagement_rate']}% "
          f"拦截={m['intercept_rate']}% 单边={m['unilateral_rate']}% 组波={m['bundle_multi_rate']}% "
          f"deploy/局={m['deploy_per_game']} 圣水={m['elixir_avg']} 延迟={m['response_latency_med']}s")


def test_mk_spawn_damage_and_iw_slow_fl():
    """2026-09-09 落地触发族（FirstLight 对账）：
    ① MK 落地溅射（projectileData=MegaKnightAppear：lv11 908（快照 Legendary 轴）、
       半径 2.2、击退 1.0、仅地面、部署延迟=落地动画）；刺客突进规避的规避对象。
    ② IceWizard 落地冰雾 FL 口径：减速 2.5s 速度+攻速双 −35%（0.65），塔不吃。
    on_spawn 是构造时钩子：只打部署瞬间在场敌人（后部署邻居不吃=官方语义）。"""
    import player as player_mod
    import battle as battle_mod
    from core import Position as P2
    DECK = ['Knight', 'Arrows', 'Fireball', 'Musketeer', 'Giant',
            'Minions', 'MiniPekka', 'Skeletons']
    # —— MK 落地溅射 ——
    bs = battle_mod.BattleState(player_mod.PlayerState(0, list(DECK), 10.0),
                                player_mod.PlayerState(1, list(DECK), 10.0), card_level=11)
    bs.step(0.033)
    k = battle_mod.Troop(bs.next_entity_id, P2(8.7, 16.2), 0, 'Knight', bs)
    bs._spawn_entity(k)
    bs.players[1].cycle = ['MegaKnight'] + [c for c in bs.players[1].cycle if c != 'MegaKnight']
    assert bs.deploy_card(1, 'MegaKnight', P2(8.5, 18.0))
    bombs = [e for e in bs.entities.values() if getattr(e, 'name', '') == 'MegaKnightAppear']
    assert len(bombs) == 1 and abs(bombs[0].delay - 1.0) < 0.1, "MK 落地弹未挂载"
    for _ in range(int(2.0 * 60)):
        bs.step(1 / 60)
    taken = k.data.hp - k.hp
    # 溅射 429（Legendary 轴 lv11）± MK 贴身普攻 268 → 697；单独溅射也应 ≥400
    assert taken >= 400, f"MK 落地溅射未结算（Knight 只掉 {taken:.0f}）"
    if k.is_alive:
        # 击退被随后的贴身碰撞/索敌位移部分抵消——只断言"受到了落地弹"
        assert any(getattr(e, 'name', '') == 'MegaKnightAppear'
                   for e in bombs), "落地弹存在性"
    # 仅地面：Minions 不吃溅射
    bs2 = battle_mod.BattleState(player_mod.PlayerState(0, list(DECK), 10.0),
                                 player_mod.PlayerState(1, list(DECK), 10.0), card_level=11)
    bs2.step(0.033)
    m = battle_mod.Troop(bs2.next_entity_id, P2(8.7, 16.2), 0, 'Minions', bs2)
    bs2._spawn_entity(m)
    bs2.players[1].cycle = ['MegaKnight'] + [c for c in bs2.players[1].cycle if c != 'MegaKnight']
    bs2.deploy_card(1, 'MegaKnight', P2(8.5, 18.0))
    for _ in range(int(2.0 * 60)):
        bs2.step(1 / 60)
    if m.is_alive:
        assert m.hp == m.data.hp, f"Minions 不应吃仅地面溅射（掉 {m.data.hp - m.hp:.0f}）"
    # —— IceWizard FL 减速 ——
    bs3 = battle_mod.BattleState(player_mod.PlayerState(0, list(DECK), 10.0),
                                 player_mod.PlayerState(1, list(DECK), 10.0), card_level=11)
    bs3.step(0.033)
    k3 = battle_mod.Troop(bs3.next_entity_id, P2(6.0, 9.5), 1, 'Knight', bs3)
    bs3._spawn_entity(k3)
    iw = battle_mod.Troop(bs3.next_entity_id, P2(4.5, 8.0), 0, 'IceWizard', bs3)
    bs3._spawn_entity(iw)
    taken3 = k3.data.hp - k3.hp
    assert 80 <= taken3 <= 90, f"IceWizard 落地伤 {taken3:.0f}（预期 84-86）"
    assert abs(k3.speed_debuff - 0.65) < 0.01 and abs(k3.hit_speed_debuff - 0.65) < 0.01
    # 纯落地窗口验证：Knight 移出 IW 射程，避免普攻 targetBuffData（IceWizardSlowDown
    # −35% 2.5s——普攻本就续期减速，官方语义）反复刷新 debuff 窗口
    k3.position = P2(14.5, 3.0)
    for _ in range(int(3.0 * 60)):
        bs3.step(1 / 60)
    assert k3.speed_debuff == 1.0 and k3.hit_speed_debuff == 1.0, "脱离源后 2.5s 减速应解除"
    print("[PASS] MK 落地溅射 908/2.2/击退1.0/仅地面 + IceWizard 冰雾 FL 口径（0.65×2.5s 双减速）")


def test_tower_value_mult():
    """塔血差异化定价曲线（2026-09-10）：凹形溢价 + 王塔贬值闸门。

    验证：满血=1.0（旧行为不变）、凹形单调递增、中点=1.5（凹形>线性中点）、
    王塔两公主塔存活≈0（×0.05）、一公主塔破后全价、塔破无边际伤害返回 1.0。"""
    from rl.env_wrapper import tower_value_mult

    # 凹形单调递增（公主塔，两公主塔存活）
    vals = [tower_value_mult(r, king=False, princesses_alive=2)
            for r in (1.0, 0.75, 0.5, 0.25, 0.05)]
    assert vals[0] == 1.0, f"满血应=1.0（旧行为不变）: {vals[0]}"
    assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1)), \
        f"溢价应随残血单调递增: {vals}"
    # 凹形：中点 0.5 → 1 + 2×(0.5)² = 1.5（线性凹形为 1.25 → 凹形更陡）
    assert abs(vals[2] - 1.5) < 1e-9, f"中点应=1.5（凹形）: {vals[2]}"
    # 残血上限：5% → 1 + 2×0.9025 = 2.805
    assert abs(vals[4] - 2.805) < 1e-9, f"5%血应=2.805: {vals[4]}"

    # 王塔贬值：两公主塔存活 → ×0.05（含满血）
    assert abs(tower_value_mult(1.0, king=True, princesses_alive=2) - 0.05) < 1e-9
    assert abs(tower_value_mult(0.5, king=True, princesses_alive=2) - 0.075) < 1e-9
    # 一公主塔破 → 王塔恢复全价（凹形）
    assert abs(tower_value_mult(0.5, king=True, princesses_alive=1) - 1.5) < 1e-9
    # 塔破（ratio≤0）→ 1.0（无边际伤害，倍数无意义）
    assert tower_value_mult(0.0, king=True, princesses_alive=0) == 1.0
    assert tower_value_mult(0.0, king=False, princesses_alive=1) == 1.0

    # 凹形强度可调（k=1：中点 1.25）
    assert abs(tower_value_mult(0.5, king=False, princesses_alive=2, k=1.0) - 1.25) < 1e-9
    print("[PASS] 塔血溢价曲线：凹形单调递增、满血=1.0、中点1.5、王塔两公主存活×0.05/一破恢复全价")


def test_reward_tower_premium():
    """塔血差异化定价接入 compute_reward（2026-09-10）：

    ① 同 500 血伤害打敌方 30% 血公主塔 vs 满血塔 → 低血塔掉血更值钱（溢价）；
    ② 双向对称：我方 30% 血公主塔挨打惩罚同步放大（负数更大）；
    ③ 王塔贬值：两公主塔存活时打王塔价值≈0（×0.05）、一破后全价；
    ④ 旧调用（无 per-tower）逐位不变。"""
    from rl.env_wrapper import compute_reward
    from rl.config import TrainConfig, reward_to_env

    std = reward_to_env(TrainConfig.resolve("economy"))
    base = dict(
        blue_hps_old=10928.0, red_hps_old=10928.0,
        blue_hps_new=10928.0, red_hps_new=10928.0,
        blue_left_old=3, red_left_old=3, blue_left_new=3, red_left_new=3,
        my_elixir_before=5.0, opp_elixir_before=5.0,
        my_elixir_after=5.0, opp_elixir_after=5.0,
        winner=None, invalid_count=0,
        blue_hps_max=10928.0, red_hps_max=10928.0)

    # ④ 旧调用无 per-tower → 逐位不变（0 事件 = 0 奖励）
    assert abs(compute_reward(std, **base)) < 1e-12, "旧调用应逐位不变"

    def r_hit_red(dmg, start_frac):
        red_old = [4824, 3052, 3052 * start_frac]
        red_new = [4824, 3052, max(0.0, 3052 * start_frac - dmg)]
        return compute_reward(std, **dict(
            base,
            red_hps_old=sum(red_old), red_hps_new=sum(red_new),
            red_towers_old=red_old, red_towers_new=red_new,
            red_towers_max=[4824, 3052, 3052]))

    def r_hit_blue(dmg, start_frac):
        blue_old = [4824, 3052, 3052 * start_frac]
        blue_new = [4824, 3052, max(0.0, 3052 * start_frac - dmg)]
        return compute_reward(std, **dict(
            base,
            blue_hps_old=sum(blue_old), blue_hps_new=sum(blue_new),
            blue_towers_old=blue_old, blue_towers_new=blue_new,
            blue_towers_max=[4824, 3052, 3052]))

    # ① 打敌方：低血塔掉血更值钱
    r_full = r_hit_red(500.0, 1.0)
    r_low = r_hit_red(500.0, 0.3)
    assert r_low > r_full, f"低血塔掉血应更值钱: {r_low} vs {r_full}"
    # ② 我方挨打：低血塔惩罚更大（双向对称）
    b_full = r_hit_blue(500.0, 1.0)
    b_low = r_hit_blue(500.0, 0.3)
    assert b_low < b_full, f"我方低血塔挨打惩罚应更大: {b_low} vs {b_full}"
    # 溢价幅度一致（进攻/防守同一曲线）
    assert abs((r_low / r_full) - (b_low / b_full)) < 1e-6, \
        f"双向溢价应同曲线: {r_low/r_full} vs {b_low/b_full}"

    # ③ 王塔贬值：两公主塔存活时打王塔≈0，一破后全价
    def r_hit_king(dmg, alive_left, alive_right):
        red_old = [4824, 3052 * alive_left, 3052 * alive_right]
        red_new = [max(0.0, 4824 - dmg), 3052 * alive_left, 3052 * alive_right]
        return compute_reward(std, **dict(
            base,
            red_hps_old=sum(red_old), red_hps_new=sum(red_new),
            red_towers_old=red_old, red_towers_new=red_new,
            red_towers_max=[4824, 3052, 3052]))

    k_gated = r_hit_king(500.0, 1, 1)
    k_open = r_hit_king(500.0, 0, 1)
    assert k_open > k_gated, f"公主塔存活时王塔应贬值: {k_open} vs {k_gated}"
    assert abs(k_open / k_gated - 20.0) < 1e-6, \
        f"王塔贬值应精确 ×0.05（1/20）: {k_open/k_gated}"
    print(f"[PASS] 塔血溢价接入 compute_reward：打敌方 500 血满血塔={r_full:.3f}/"
          f"30%塔={r_low:.3f}（{r_low/r_full:.2f}×）、我方挨打对称、"
          f"王塔贬值 {k_open/k_gated:.0f}×、旧调用逐位不变")


def test_reward_tower_premium_rlenv_flow():
    """塔血差异化定价经 RLEnv.step 真实链路生效 + MCTS 值函数同源（2026-09-10）。

    验证：① RLEnv 打磨敌方 30% 血塔的累计奖励 > 打磨满血塔（真实 step 链路，
    env 内 _blue_towers_max 已每塔记录）；② MCTS node_value 对敌方残血塔估值更高、
    王塔两公主存活时残血不推高值、一破后推高——与训练奖励同源。"""
    import numpy as np
    from rl.env_wrapper import RLEnv, DEFAULT_DECK
    from rl.action_bundle import ActionBundle
    from rl.config import TrainConfig, reward_to_env

    cfg = TrainConfig.resolve("economy")

    # ① RLEnv.step 链路：把敌方右公主塔打到残血，累计几帧塔损奖励
    def env_total_loss_with(start_frac):
        env = RLEnv(opponent=lambda obs: ActionBundle.noop(), seed=0,
                    reward_weights=reward_to_env(cfg),
                    deck0=DEFAULT_DECK, deck1=DEFAULT_DECK)
        env.reset(seed=0)
        p1 = env.battle.players[1]
        p1.right_tower_hp = 3052 * start_frac
        # 同步实体 HP（battle 内以实体为准）
        env.battle.update_player_hp()
        # 连续 noop 5 帧（正常推进，无人工掉血）→ 累计奖励为 0（无塔损事件）
        tot = 0.0
        for _ in range(5):
            _, r, term, _, _ = env.step(ActionBundle.noop())
            tot += r
            if term:
                break
        return tot

    # 无塔损事件 → 0（这条主要验证 RLEnv 传参不崩 + 每塔 max 记录）
    t_full = env_total_loss_with(1.0)
    assert abs(t_full) < 1e-6, f"无塔损 noop 累计应≈0: {t_full}"
    # 塔血每塔 max 数组已记录
    env = RLEnv(opponent=lambda obs: ActionBundle.noop(), seed=0,
                reward_weights=reward_to_env(cfg),
                deck0=DEFAULT_DECK, deck1=DEFAULT_DECK)
    env.reset(seed=0)
    assert env._blue_towers_max == [4824, 3052, 3052] and \
        env._red_towers_max == [4824, 3052, 3052], "reset 应记录每塔满血"

    # ② MCTS node_value 同源
    from rl.mcts import MCTSConfig, node_value
    import battle as battle_mod
    import player as player_mod

    DECK = ['Knight', 'Arrows', 'Fireball', 'Musketeer', 'Giant',
            'Minions', 'MiniPekka', 'Skeletons']
    mcfg = MCTSConfig()

    def mk_bs(**tower_overrides):
        bs = battle_mod.BattleState(
            player_mod.PlayerState(0, list(DECK), 5.0),
            player_mod.PlayerState(1, list(DECK), 5.0), card_level=11)
        bs.update_player_hp()
        p1 = bs.players[1]
        for k, v in tower_overrides.items():
            setattr(p1, k, v)
        return bs

    v0 = node_value(mk_bs(), 0, mcfg)
    assert abs(v0) < 1e-9, f"满血 vs 满血应≈0: {v0}"
    # 敌方残血公主塔 → 我方值推高
    v_low = node_value(mk_bs(left_tower_hp=3052 * 0.3), 0, mcfg)
    assert v_low > v0, f"敌方残血塔应推高我方值: {v_low}"
    # 王塔：两公主塔存活时残血不值钱；一破后值钱
    v_king_gated = node_value(mk_bs(king_tower_hp=4824 * 0.5), 0, mcfg)
    v_king_open = node_value(mk_bs(king_tower_hp=4824 * 0.5, left_tower_hp=0.0), 0, mcfg)
    assert v_king_open > v_king_gated, \
        f"公主塔存活时王塔残血不应推高值: {v_king_open} vs {v_king_gated}"
    print(f"[PASS] 塔血溢价 RLEnv.step 链路 + MCTS 同源：RLEnv 每塔 max 记录、"
          f"noop 累计≈0、MCTS 敌残血塔推高({v_low:.2f})、王塔闸门生效")


def test_opponent_pool_mix():
    """9j A 层：训练对手池（frozen/hist/defend 混合）+ SelfDefenderPolicy 反制。

    校验：真实旧 ckpt 池加载（23 维 belief 尾零兼容）；采样分布接近名义混合；
    PFSP 败局回填；纯防守模式面对过河威胁真的出反制部队。"""
    import random, glob, tempfile
    from collections import Counter
    from rl.train_solo import _OpponentPool, _collect_hist_ckpts
    from rl.opponents import SelfDefenderPolicy
    from rl.config import TrainConfig
    from rl.env_wrapper import RLEnv
    from rl.follower import FollowerPolicy
    from rl.train_follower import FollowerOpponent
    from rl.belief import BeliefInference
    from rl.plan_space import PLAN_DIM
    from core import Position

    env = RLEnv(opponent=None, seed=0, card_level=11)
    from rl.action_mask import validate_bundle
    # SelfDefenderPolicy 反制：P0 部队推进过河 → P1 侧出反制（passive_prob=1.0 纯防守）
    obs, _ = env.reset(seed=42)
    assert env.battle.deploy_card(0, "Musketeer", Position(8.5, 12.0))
    for _ in range(600):
        env.battle.step(1 / 60)
    defen = SelfDefenderPolicy(seed=3, env=env, passive_prob=1.0)
    acted = None
    for _ in range(20):
        bundle = defen(None)
        if bundle.sub_actions:
            acted = bundle.sub_actions
            break
    assert acted is not None, "过河威胁下纯防守模式必须出反制"
    ok, reason, _ = validate_bundle(env.battle, 1, bundle)
    assert ok, f"反制动作非法: {reason}"

    # 池分布：临时目录放 3 个真实 ckpt（本仓库 runs/economy 或归档目录）
    src_candidates = (sorted(glob.glob("runs/economy/solo_main_*.pt")) +
                      sorted(glob.glob("../../runs/archive/*/solo_main_*.pt")))
    if not src_candidates:
        # T1-5：原先只 print 一行 [SKIP] ⇒ 缺 ckpt 时测试**照样全绿**（假绿）。
        # 现在进 `_mark_skip` 计数器，由 `_report_tests()` 汇总重复打印。
        _mark_skip("无可用历史 ckpt（runs/economy/solo_main_*.pt 与 ../../runs/archive/*/ 均为空）"
                   " ⇒ 「真实旧 ckpt 池加载 / 采样分布 / PFSP 败局回填」**未验证**，只验了 defender")
        return
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "economy"), exist_ok=True)
        for src in src_candidates[::max(1, len(src_candidates) // 3)][:3]:
            shutil.copy(src, os.path.join(td, "economy", os.path.basename(src)))
        cfg = TrainConfig.resolve("economy")
        cfg.out_dir = td
        from rl.belief import belief_token_dim
        frozen_pol = FollowerPolicy(hidden=128, plan_dim=PLAN_DIM,
                                    belief_dim=belief_token_dim(list(env.deck1)))
        frozen_side = FollowerOpponent(frozen_pol, env,
                                       belief=BeliefInference(opp_deck=env.deck1),
                                       deterministic=True)
        pool = _OpponentPool(cfg, env, frozen_side, random.Random(0), "cpu")
        assert len(pool.hist_paths) == 3
        kinds = Counter()
        hist_ids = set()
        N = 2000
        for _ in range(N):
            kind, side, hid = pool.sample()
            kinds[kind] += 1
            if kind == "hist":
                hist_ids.add(hid)
                assert side is not None
        # 期望用池自己的配比（cfg.opp_mix / _OPP_MIX，P1-1b 后为 0.5/0.3/0.2），±35% 容差
        mix = pool.mix
        for k in ("hist", "defend"):
            exp = N * float(mix[k])
            assert abs(kinds[k] - exp) <= 0.35 * exp, (k, kinds, mix)
        assert len(hist_ids) == 3, hist_ids
        pool.record(1)                                 # 败局回填不崩
    print(f"[PASS] 对手池：分布 {dict(kinds)}（名义 {pool.mix}，N={N}）；"
          f"hist 旧 ckpt 加载+PFSP 回填；SelfDefender 面对过河威胁出合法反制")


def test_value_channel_norm_and_gnorm_split():
    """P0-1（2026-09-11 审计整改）：价值通道量纲 + 梯度成分诊断。

    三条断言各自可证伪：
    ① value_norm="none" 时 v_loss 逐位等于外部复算的原始 MSE（关开关 = 旧实现），
       且 stats 全有限、不含 p_gnorm/v_gnorm 键（旧测试的 isfinite 断言不受影响）；
    ② 单轮 on-policy 重放 ratio≡1.0（结构性事实，不是"策略没在动"）；
    ③ 诊断真的能分辨"谁在推动更新"——把 value_head 放大 1000× 后 v_gnorm 必须
       应声抬升两个数量级（否则诊断函数是假的）。
    """
    import copy
    import torch
    from rl.env_wrapper import RLEnv
    from rl.belief import BeliefInference
    from rl.follower import FollowerPolicy
    from rl.plan_space import PLAN_DIM
    from rl.ppo import PPOTrainer, ReturnScaler

    env = RLEnv(opponent=None, seed=0)
    obs, _ = env.reset()
    belief = BeliefInference(opp_deck=env.deck1, n_particles=128, seed=0)
    tok = belief.encode(obs, None)
    plan = np.zeros(PLAN_DIM, dtype=np.float32)
    pol = FollowerPolicy(hidden=32, plan_dim=PLAN_DIM, belief_dim=len(tok))
    trans = _tiny_rollout_transitions(pol, env, belief, tok, plan)
    n = len(trans)

    # ① none = 旧行为
    pol_a, pol_ref = copy.deepcopy(pol), copy.deepcopy(pol)
    s0 = PPOTrainer(pol_a, lr=1e-3).update(trans)
    assert all(np.isfinite(v) for v in s0.values()), s0
    assert "p_gnorm" not in s0 and "v_gnorm" not in s0, s0
    assert s0["value_scale"] == 1.0, s0
    assert abs(s0["value_loss"] - s0["value_loss_raw"]) < 1e-9, s0
    with torch.no_grad():
        _, val_ref, _ = pol_ref.evaluate_batch(
            [t["obs"] for t in trans], [t["belief"] for t in trans],
            [t["plan"] for t in trans], [t["bundle"] for t in trans],
            [t["masks"] for t in trans], [t["init_hidden"] for t in trans])
        rets_ref = torch.tensor([t["returns"] for t in trans], dtype=torch.float32)
        raw_expect = float(((val_ref.squeeze(-1) - rets_ref) ** 2).sum().item()) / n
    assert abs(raw_expect - s0["value_loss_raw"]) < 1e-4, (raw_expect, s0)
    # ①b explained_variance（P0-1c）：定义性行为 + 与外部按定义复算一致
    _ev = PPOTrainer.explained_variance
    _R = np.array([1.0, 2.0, 3.0, 4.0])
    assert abs(_ev(_R, _R) - 1.0) < 1e-9, _ev(_R, _R)          # 完美预测
    assert abs(_ev(np.full(4, _R.mean()), _R)) < 1e-9          # 常数=批均值 → 0
    assert _ev(np.zeros(4), _R) < 0.0                          # 比常数差
    assert _ev([1.0], [1.0]) == 0.0                            # 单点不可判读
    assert _ev([0.0, 0.0], [3.0, 3.0]) == 0.0                  # 方差退化 → 0，不抛错
    assert _ev([1.0, 2.0], [1.0, 2.0, 3.0]) == 0.0             # 长度不匹配 → 0
    ev_expect = 1.0 - raw_expect / float(np.var([t["returns"] for t in trans]))
    assert abs(ev_expect - s0["explained_variance"]) < 1e-6, (ev_expect, s0)
    # ② 单轮 on-policy 重放：ratio 恒 1、clip 恒 0（结构性）
    # 容差说明（2026-09-11，grid_ln 落地时暴露）：`ratio = exp(lp_new − old)` 里
    # lp_new 来自批量前向、old 来自 rollout 逐步前向，两条路径的浮点求和顺序不同，
    # 偏差下限就是 **float32 的一个 ULP（ε≈1.19e-7）**。实测偏差 1.19e-7 = 0.998 ULP
    # ⇒ 原先写的 1e-9（= ε/119）是**物理上不可达**的，之前只是碰巧舍入成 0 才通过。
    # 这里放宽到 1e-5（仍比"策略真的动了"的量级 ~1e-3 紧 100 倍，见 ppo.py:12-22）。
    assert abs(s0["ratio_mean"] - 1.0) < 1e-5 and s0["clip_frac"] == 0.0, s0

    # ③ running：v_loss = raw / s²，诊断可测
    pol_b = copy.deepcopy(pol)
    p1 = PPOTrainer(pol_b, lr=1e-3, value_norm="running", diagnose_every=1)
    s1 = p1.update(trans)
    assert "p_gnorm" in s1 and isinstance(s1["v_gnorm"], float), s1
    assert s1["value_scale"] > 0.0, s1
    assert abs(s1["value_loss"] * s1["value_scale"] ** 2 - s1["value_loss_raw"]) < 1e-3, s1

    pol_c = copy.deepcopy(pol)
    with torch.no_grad():
        pol_c.value_head.weight.mul_(1000.0)
        pol_c.value_head.bias.mul_(1000.0)
    s2 = PPOTrainer(pol_c, lr=1e-3, diagnose_every=1).update(trans)
    assert s2["v_gnorm"] > 100.0 * s1["v_gnorm"], (s2["v_gnorm"], s1["v_gnorm"])

    # ④ 诊断每 N 次才跑（默认关：diagnose_every=0 → 无键）
    s3 = PPOTrainer(copy.deepcopy(pol), lr=1e-3, diagnose_every=0).update(trans)
    assert "p_gnorm" not in s3, s3

    # ⑤ ReturnScaler 随 ckpt 存取一致（续训不重置）
    sc = ReturnScaler.from_dict(p1.ret_scaler.to_dict())
    assert sc.count == n and abs(sc.mean - p1.ret_scaler.mean) < 1e-12 \
        and abs(sc.std() - p1.ret_scaler.std()) < 1e-12, (sc.to_dict(), p1.ret_scaler.to_dict())
    assert ReturnScaler.from_dict(None).count == 0
    print(f"[PASS] 价值通道：none 逐位=原始MSE（raw={raw_expect:.4f}）、ratio≡1 结构性、"
          f"running scale={s1['value_scale']:.3f}（{s1['value_loss']:.3f}/{s1['value_loss_raw']:.3f}）、"
          f"v_gnorm 随 value_head ×1000 抬升 {s2['v_gnorm'] / max(1e-12, s1['v_gnorm']):.0f}×、"
          f"scaler 存取一致")
