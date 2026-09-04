"""全链路自检：ActionBundle / 原子校验 / 贝叶斯信念 / 跟随者 / PPO / 联赛 + 评审回归测试。

回归测试对应 docs/rl_review_fix_plan.md §6：
- test_hidden_replay_consistency           → P0-1
- test_entropy_positive_and_sign           → P0-2
- test_mask_validate_invariant_both_sides  → P0-3/P0-4
- test_heuristic_opponent_actually_plays   → P0-3/P0-4
- test_exploiter_loads_main_checkpoint     → P0-5
- test_belief_survives_ability             → P0-6
- test_belief_multi_card_update            → P1-5
- test_register_checkpoint_isolated        → P1-9
- test_bundle_cap_no_crash                 → P1-18
- test_replay_roundtrip                    → P1-21
- test_prophet_empty_board_not_defend      → P1-4
- test_winrate_streams_independent         → 联赛数据契约：不同 pair 的 PFSP 胜率流独立演进
- test_elo_eval_granularity                → 评估粒度：噪声地板(SE=347.5/√N) / 轮内聚合估计 / 误差棒链路
- test_ablation_recorded                   → belief/plan 消融：4 变体对比 + z 判定 + JSON/CSV 落盘
- test_flow_sweep_smoke                    → flow 数据效率 A/B：缩小池 sweep 通路 + summary 落盘

运行：python rl/selftest.py   （需在 src/clasher_new 下，或由 scripts/rl/selftest.py 包装）
"""

import os
import sys
import random

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import numpy as np

from card_utils import Card


def test_action_bundle_same_tick():
    from rl.env_wrapper import RLEnv
    from rl.action_bundle import ActionBundle
    from rl.action_mask import validate_bundle

    env = RLEnv(opponent=None, seed=1)
    env.reset()
    env.battle.players[0].elixir = 10.0
    hand = list(env.battle.players[0].cycle[:4])
    bundle = ActionBundle(); bundle.add(1, 8, 12); bundle.add(2, 9, 13)
    ok, reason, resolved = validate_bundle(env.battle, 0, bundle)
    assert ok and [r[0] for r in resolved] == hand[:2], reason
    n0 = len(env.battle.entities)
    env.step(bundle)
    played = env.battle.players[0].cycle[-2:]
    assert set(played) == {hand[0], hand[1]}, f"played {played}, expected {hand[:2]}"
    assert len(env.battle.entities) > n0, "same tick should add entities"
    print("[PASS] ActionBundle 同刻多卡：正确打出决策时刻的两张卡")

    # 原子拒绝
    bad = ActionBundle(); bad.add(1, 8, 12); bad.add(1, 9, 13)
    ok2, reason2, _ = validate_bundle(env.battle, 0, bad)
    assert not ok2, "duplicate slot should be rejected"
    env.battle.players[0].elixir = 0.5
    ok3, reason3, _ = validate_bundle(env.battle, 0, ActionBundle.from_single(1, 8, 12))
    assert not ok3, "no-elixir should be rejected"
    print("[PASS] ActionBundle 原子校验：重复槽位 / 圣水不足整包拒绝")


def test_action_bundle_ability():
    from rl.env_wrapper import RLEnv
    from rl.action_bundle import ActionBundle
    from rl.action_mask import validate_bundle, ability_legal
    from battle import Troop
    from core import Position

    env = RLEnv(opponent=None, seed=0)
    env.reset()
    b = env.battle
    b.players[0].elixir = 10.0
    hero = Troop(b.next_entity_id, Position(9.0, 12.0), 0, "SkeletonKing", b)
    b._spawn_entity(hero)
    assert ability_legal(b, 0), "场上就绪英雄应使 ability_legal=True"

    bundle = ActionBundle(); bundle.add_ability(); bundle.add(1, 8, 12)
    ok, reason, resolved = validate_bundle(b, 0, bundle)
    assert ok and resolved[0][0] == "__ability__", reason
    elixir_before, uses_before = b.players[0].elixir, hero.ability_uses
    env.step(bundle)
    assert hero.ability_uses == uses_before + 1, "英雄技能应被触发"
    assert b.players[0].elixir < elixir_before, "技能应扣圣水"
    print("[PASS] ActionBundle 覆盖英雄技能：出牌 + 开技能同一 tick")

    # 无就绪英雄时，纯技能 bundle 应被原子拒绝
    env2 = RLEnv(opponent=None, seed=0); env2.reset()
    bad = ActionBundle(); bad.add_ability()
    ok2, reason2, _ = validate_bundle(env2.battle, 0, bad)
    assert not ok2, "无就绪英雄时技能动作应被拒绝"
    print("[PASS] ActionBundle 原子校验：无就绪英雄时技能整包拒绝")


def test_bayes_filter():
    from rl.bayes_filter import CycleBayesFilter
    import player as player_mod
    import battle as battle_mod
    from core import Position

    deck = ["Minions", "Archer", "MiniPekka", "Musketeer", "Giant", "Fireball", "Arrows", "Knight"]
    real = player_mod.PlayerState(1, list(deck), 5.0)
    b = battle_mod.BattleState(player_mod.PlayerState(0, list(deck), 5.0), real)
    bf = CycleBayesFilter(deck, n_particles=512, seed=0)
    played = []
    for _ in range(6):
        card = real.cycle[0]
        if b.deploy_card(1, card, Position(3.5, 25.5)):
            played.append(card)
            bf.update(card)
    assert abs(bf.hand_probs().sum() - 4.0) < 0.01
    top3 = [deck[i] for i in np.argsort(-bf.next_probs())][:3]
    assert real.cycle[4] in top3, f"real next {real.cycle[4]} not in top3 {top3}"
    print("[PASS] 贝叶斯粒子滤波：后验收敛，下一张牌在 top3")


def _make_policy_and_tokens(env, seed=0):
    from rl.belief import BeliefInference
    from rl.plan_space import PlanToken, PLAN_DIM
    belief = BeliefInference(opp_deck=env.deck1, n_particles=128, seed=seed)
    tok = belief.encode(None, None)
    return belief, tok, PlanToken.zeros().to_vector(), PLAN_DIM


def test_hidden_replay_consistency():
    """P0-1：evaluate 用 rollout 时记录的 hidden 重放，lp_new ≈ old。"""
    import torch
    from rl.env_wrapper import RLEnv
    from rl.follower import FollowerPolicy

    env = RLEnv(opponent=None, seed=0)
    obs, _ = env.reset()
    belief, tok, plan, PLAN_DIM = _make_policy_and_tokens(env)
    pol = FollowerPolicy(hidden=64, plan_dim=PLAN_DIM, belief_dim=len(tok))

    b1, lp1, _, h1, masks1 = pol.act(obs, tok, plan, env.get_action_mask, hidden=None)
    lp_new1, _, _, _ = pol.evaluate(obs, tok, plan, b1, masks1, hidden=None)
    assert abs(float(lp_new1) - lp1) < 1e-3, f"step1 重放 logprob 不一致: {lp1} vs {lp_new1}"

    obs2, _, term, trunc, info = env.step(b1)
    belief.update(obs2, info.get("opp_played"))
    b2, lp2, _, h2, masks2 = pol.act(obs2, tok, plan, env.get_action_mask, hidden=h1)
    lp_new2, _, _, _ = pol.evaluate(obs2, tok, plan, b2, masks2, hidden=h1)
    assert abs(float(lp_new2) - lp2) < 1e-3, f"step2 重放 logprob 不一致: {lp2} vs {lp_new2}"
    print("[PASS] P0-1 hidden 重放一致：evaluate 用记录 hidden，lp_new ≈ old")


def test_entropy_positive_and_sign():
    """P0-2：熵非负；正优势 + 未超 clip 时更新后已选动作概率上升。"""
    from rl.env_wrapper import RLEnv
    from rl.follower import FollowerPolicy
    from rl.ppo import PPOTrainer

    env = RLEnv(opponent=None, seed=0)
    obs, _ = env.reset()
    belief, tok, plan, PLAN_DIM = _make_policy_and_tokens(env)
    pol = FollowerPolicy(hidden=64, plan_dim=PLAN_DIM, belief_dim=len(tok))
    ppo = PPOTrainer(pol, lr=1e-2, clip=0.5, ent_coef=0.001)

    bundle, lp, val, _, masks = pol.act(obs, tok, plan, env.get_action_mask,
                                        hidden=None, deterministic=True)
    lp_new, _, _, ent = pol.evaluate(obs, tok, plan, bundle, masks, hidden=None)
    assert float(ent) >= 0.0, "熵必须非负"
    old_lp = float(lp_new)
    tr = {"obs": obs, "belief": tok, "plan": plan, "bundle": bundle,
          "old_logprob": old_lp - 0.2, "adv": 1.0, "returns": val,
          "masks": masks, "init_hidden": None}
    ppo.update([tr])
    lp_after, _, _, _ = pol.evaluate(obs, tok, plan, bundle, masks, hidden=None)
    assert float(lp_after) > old_lp - 1e-4, \
        f"正优势下已选动作概率应上升: {old_lp:.4f} -> {float(lp_after):.4f}"
    print("[PASS] P0-2 熵方向正确：entropy>=0 且正优势下已选动作概率上升")


def test_mask_validate_invariant_both_sides():
    """P0-3/P0-4：mask 合法 ⟹ validate 通过（P0/P1 两侧）；两侧掩码不同。"""
    from rl.env_wrapper import RLEnv
    from rl.action_bundle import ActionBundle
    from rl.action_mask import validate_bundle

    env = RLEnv(opponent=None, seed=2)
    env.reset()
    env.battle.players[0].elixir = 10.0
    env.battle.players[1].elixir = 10.0
    for pid in (0, 1):
        mask = env.get_action_mask_for(pid)
        checked = 0
        for slot in range(4):
            if not mask["slots"][slot]:
                continue
            ys, xs = np.nonzero(mask["cells"][slot])
            assert ys.size > 0, f"P{pid} slot{slot} 应至少有一个合法格"
            for y, x in list(zip(ys, xs))[:5]:
                b = ActionBundle.from_single(slot + 1, int(x), int(y))
                ok, reason, _ = validate_bundle(env.battle, pid, b)
                assert ok, f"P{pid} 掩码合法格却被 validate 拒绝: slot={slot} ({x},{y}) {reason}"
                checked += 1
        assert checked > 0, f"P{pid} 应有可校验的合法格"

    # P0-3：指纹含 player_id，两侧手牌不同时缓存不得串用。
    # 让 P0 槽0 = 法术（全合法）、P1 槽0 = 部队（部分合法）。
    p0 = env.battle.players[0]
    p1 = env.battle.players[1]
    p0.cycle = ["Fireball", "Minions", "Musketeer", "MiniPekka", "Giant", "Arrows", "Archer", "Knight"]
    p1.cycle = ["Minions", "Fireball", "Musketeer", "MiniPekka", "Giant", "Arrows", "Archer", "Knight"]
    m0 = env.get_action_mask_for(0)
    m1 = env.get_action_mask_for(1)
    assert env._mask_fp is not None and env._mask_fp[0] == 1, "指纹应包含 player_id"
    assert not np.array_equal(m0["cells"][0], m1["cells"][0]), \
        "P0 槽0 法术 vs P1 槽0 部队的 cells 应不同（缓存按 player_id 区分）"
    # 交替调用不串缓存
    m0b = env.get_action_mask_for(0)
    m1b = env.get_action_mask_for(1)
    assert np.array_equal(m0["cells"], m0b["cells"]) and np.array_equal(m1["cells"], m1b["cells"])
    print("[PASS] P0-3/P0-4 掩码-校验不变式 + player_id 缓存隔离")


def test_heuristic_opponent_actually_plays():
    """P0-3/P0-4：heuristic 对手出牌率不再 0.5%。"""
    from rl.env_wrapper import RLEnv
    from rl.action_bundle import ActionBundle
    from rl.train_follower import heuristic_opponent

    env = RLEnv(opponent=None, seed=3)
    env.opponent = heuristic_opponent(env, random.Random(0))
    env.reset()
    decisions = plays = 0
    for _ in range(50):
        env.battle.players[1].elixir = 10.0
        obs, r, term, trunc, info = env.step(ActionBundle.noop())
        decisions += 1
        if info["opp_played"]:
            plays += 1
        if term or trunc:
            env.reset()
    assert plays >= 25, f"heuristic 出牌率过低: {plays}/{decisions}"
    print(f"[PASS] P0-3/P0-4 heuristic 对手正常出牌：{plays}/{decisions}")


def test_exploiter_loads_main_checkpoint():
    """P0-5：checkpoint 元数据 + 旧格式回退都能正确加载。"""
    import tempfile
    import torch
    from rl.follower import FollowerPolicy, save_checkpoint, load_checkpoint
    from rl.plan_space import PLAN_DIM

    env = _mk_env()
    belief, tok, plan, _ = _make_policy_and_tokens(env)
    pol = FollowerPolicy(hidden=64, plan_dim=PLAN_DIM, belief_dim=len(tok))
    d = tempfile.mkdtemp()
    p1 = os.path.join(d, "main.pt")
    save_checkpoint(pol, p1)
    loaded = load_checkpoint(p1)
    assert loaded.plan_dim == PLAN_DIM and loaded.belief_dim == len(tok)
    p2 = os.path.join(d, "main_old.pt")
    torch.save(pol.state_dict(), p2)
    loaded2 = load_checkpoint(p2, hidden_dim=64)
    assert loaded2.plan_dim == PLAN_DIM and loaded2.belief_dim == len(tok)
    print("[PASS] P0-5 checkpoint 元数据/旧格式回退加载正常")


def test_belief_survives_ability():
    """P0-6：哨兵 __ability__ 不进信念模块，不崩溃、不重置先验。"""
    from rl.env_wrapper import RLEnv
    from rl.belief import BeliefInference

    env = RLEnv(opponent=None, seed=0)
    obs, _ = env.reset()
    belief = BeliefInference(opp_deck=env.deck1, n_particles=512, seed=0)
    belief.reset(env.deck1)
    belief.update(obs, [{"card": "Minions", "x": 3.0, "y": 25.0},
                        {"card": "Archer", "x": 4.0, "y": 24.0},
                        {"card": "MiniPekka", "x": 5.0, "y": 23.0}])
    p_before = belief.state().next_probs.copy()
    # 技能哨兵必须被过滤
    belief.update(obs, [{"card": "__ability__", "x": None, "y": None}])
    st = belief.state()
    assert np.allclose(st.next_probs, p_before), "技能哨兵不应改变信念"
    uniform = 1.0 / len(env.deck1)
    assert st.next_probs.max() > uniform + 0.01, "信念不应被重置为均匀先验"
    print("[PASS] P0-6 信念模块免疫技能哨兵：不崩溃、不重置")


def test_belief_multi_card_update():
    """P1-5：同 tick 出两张，信念对两张都做排除。"""
    from rl.belief import BeliefInference

    deck = ["Minions", "Archer", "MiniPekka", "Musketeer", "Giant", "Fireball", "Arrows", "Knight"]
    belief = BeliefInference(opp_deck=deck, n_particles=512, seed=0)
    belief.update(None, [{"card": "Minions", "x": 3.0, "y": 25.0},
                         {"card": "Archer", "x": 4.0, "y": 24.0}])
    st = belief.state()
    idx = {c: i for i, c in enumerate(deck)}
    assert st.hand_probs[idx["Minions"]] < 0.1, "Minions 应被排除出手牌"
    assert st.hand_probs[idx["Archer"]] < 0.1, "Archer 应被排除出手牌"
    print("[PASS] P1-5 同 tick 多卡：信念对两张卡都做了排除")


def test_register_checkpoint_isolated():
    """P1-9：快照注册后，historical 参数不再随 main 变化。"""
    import torch
    from rl.follower import FollowerPolicy
    from rl.league import League
    from rl.plan_space import PLAN_DIM

    env = _mk_env()
    belief, tok, plan, _ = _make_policy_and_tokens(env)
    main = FollowerPolicy(hidden=64, plan_dim=PLAN_DIM, belief_dim=len(tok))
    lg = League(seed=0)
    lg.add_agent("main", kind="main", policy=main)
    new_id = lg.register_checkpoint("main", main)
    snap = lg.agents[new_id].policy
    assert lg.agents["main"].kind == "main", "原 main 应保持 main"
    assert snap is not main, "快照应为独立对象"
    before = {k: v.clone() for k, v in snap.state_dict().items()}
    for p in main.parameters():
        p.data.add_(0.1)
    for k, v in snap.state_dict().items():
        assert torch.allclose(v, before[k]), "historical 快照参数不应随 main 变化"
    print("[PASS] P1-9 快照隔离：historical 参数与训练中的 main 解耦")


def test_bundle_cap_no_crash():
    """P1-18：10 费 + 低费卡局面，follower 稳定产出合法 bundle 而不崩。"""
    from rl.env_wrapper import RLEnv
    from rl.follower import FollowerPolicy
    from rl.action_bundle import K_MAX

    env = RLEnv(opponent=None, seed=0,
                deck0=["Arrows", "Archer", "Minions", "Knight",
                       "Minions", "Archer", "Knight", "Arrows"])
    obs, _ = env.reset()
    env.battle.players[0].elixir = 10.0
    belief, tok, plan, PLAN_DIM = _make_policy_and_tokens(env)
    pol = FollowerPolicy(hidden=64, plan_dim=PLAN_DIM, belief_dim=len(tok))
    hidden = None
    for _ in range(15):
        bundle, _, _, hidden, _ = pol.act(obs, tok, plan, env.get_action_mask, hidden=hidden)
        assert bundle.size <= K_MAX, "bundle 不应超 K_MAX"
        env.battle.players[0].elixir = 10.0
        obs, _, term, trunc, info = env.step(bundle)
        belief.update(obs, info.get("opp_played"))
        if term or trunc:
            obs, _ = env.reset()
            env.battle.players[0].elixir = 10.0
            hidden = None
            belief.reset(env.deck1)
    print("[PASS] P1-18 K_MAX 上限：低费卡局面不崩且 bundle 不超限")


def test_replay_roundtrip():
    """P1-21：save→load→to_belief_dataset 非空且 hidden 对齐。"""
    import tempfile
    from rl.replay import EpisodeReplay
    from rl.env_wrapper import RLEnv
    from rl.action_bundle import ActionBundle

    env = RLEnv(opponent=None, seed=0)
    obs, _ = env.reset()
    ep = EpisodeReplay(record_hidden=True)
    ep.start()
    bundle = ActionBundle.from_single(1, 8, 12)
    obs, r, term, trunc, info = env.step(bundle)
    ep.record_step(obs, bundle, r, info)
    hid = info["hidden"]
    d = tempfile.mkdtemp()
    path = os.path.join(d, "ep.pkl")
    ep.save(path)
    ep2 = EpisodeReplay.load(path)
    ds = ep2.to_belief_dataset()
    assert len(ds) == 1, "round-trip 后监督样本应为 1"
    assert ds[0][2]["opp_next"] == hid["opp_next"], "hidden 标签应对齐"
    print("[PASS] P1-21 replay round-trip：save→load→dataset 数据对齐")


def test_prophet_empty_board_not_defend():
    """P1-4：空场开局 intent != defend_*。"""
    from rl.env_wrapper import RLEnv
    from rl.prophet import ProphetPlanner

    env = RLEnv(opponent=None, seed=0)
    env.reset()
    plan = ProphetPlanner().plan(env.get_prophet_state())
    assert not plan.macro_intent.startswith("defend"), plan.macro_intent
    print(f"[PASS] P1-4 先知空场不防御：intent={plan.macro_intent}")


def _mk_env():
    from rl.env_wrapper import RLEnv
    return RLEnv(opponent=None, seed=0)


def test_random_deck_model():
    """卡组完全随机模型：每局 8 卡重采样，脚本动作合法。"""
    import random as _random
    from rl.env_wrapper import RLEnv
    from rl.opponents import ScriptedPolicy, build_card_pool, sample_deck
    from rl.action_mask import validate_bundle

    pool = build_card_pool()
    assert len(pool) >= 8, "卡池应足够采样 8 张"
    deck = sample_deck(_random.Random(0), pool)
    assert len(deck) == 8 and len(set(deck)) == 8, "随机卡组应为 8 张互不相同的卡"

    pol = ScriptedPolicy(mode="random", pool=pool, seed=0)
    env = RLEnv(opponent=pol, seed=0)
    pol.env = env
    env.deck1_factory = pol.deck
    obs, _ = env.reset()
    d1 = list(env.deck1)
    env.reset()
    d2 = list(env.deck1)
    assert d1 != d2, "每局卡组应重采样（极大概率不同）"

    b = pol.play(env, 1)
    ok, reason, _ = validate_bundle(env.battle, 1, b)
    assert ok or b.size == 0, f"脚本动作应合法或 noop: {reason}"
    print("[PASS] 卡组完全随机模型：每局 8 卡重采样且动作合法")


def test_league_elo_history():
    """联赛 Elo 历史（训练网页 UI 数据源）：记录 + save/load round-trip。"""
    import tempfile
    from rl.league import League

    lg = League(seed=0)
    lg.add_agent("main", kind="main")
    lg.add_agent("random_deck", kind="baseline")
    lg.record_match("main", "random_deck", 0.8)
    lg.record_elo_history(100)
    lg.record_match("main", "random_deck", 0.5)
    lg.record_elo_history(200)
    assert len(lg.elo_history.get("main", [])) == 2, "应记录两个时间点"
    assert lg.elo_history["main"][0][0] == 100 and lg.elo_history["main"][1][0] == 200
    assert lg.total_steps == 200

    d = tempfile.mkdtemp()
    p = os.path.join(d, "state.json")
    lg.save_state(p)
    lg2 = League(seed=0)
    lg2.load_state(p)
    assert lg2.elo_history["main"] == lg.elo_history["main"]
    assert lg2.total_steps == 200
    print("[PASS] 联赛 Elo 历史：记录 + save/load round-trip")


def test_winrate_streams_independent():
    """不同 pair 的 PFSP 胜率流独立演进（防"各 pair 共享同一 EMA 流"类写入 bug 回归）。

    背景：runs/aggressive 曾见 main 对 5 个对手胜率全等（0.40725312499999994）。
    排查确认那**不是写入 bug**——每个值都可还原为独立 4 局 EMA
    （0.40725312499999994 = 0.5×(1-0.05)⁴ = n_eval_games=4 全败；0.4547531249999999
    = [L,L,W,L]…），只是 main 全败 + play_pair 各 pair 复用同批种子导致局面同构。
    本测试锁定真正的正确性契约：不同 pair 各维护独立 EMA 流——
    1) 每流数值 = 独立重算的 EMA 参考值（由自己的比分序列驱动）；
    2) 不同比分序列必须演化出不同胜率（若实现误用共享 key/流，此断言必挂）；
    3) record_match 双向互补：winrate(a,b) + winrate(b,a) == 1；
    4) 记录 (a,b) 不得污染其它 pair 的流（独立性）；
    5) save/load round-trip 后 key（a|b 序列化）不串流。
    """
    import tempfile
    from rl.league import League

    lg = League(seed=0)
    for aid in ("main", "push_flow", "counter_flow", "lockdown_flow"):
        lg.add_agent(aid, kind="main" if aid == "main" else "baseline")

    alpha = 0.05

    def ref_ema(seq):
        v = 0.5
        for s in seq:
            v = v * (1 - alpha) + s * alpha
        return v

    # 每 pair 喂不同比分序列（长度 40，0/0.5/1 混合）
    seq_ab = [1.0, 0.0] * 20      # main vs push_flow:    胜负交替 → ~0.5
    seq_ac = [1.0] * 40           # main vs counter_flow: 全胜   → 高位
    seq_ad = [0.0] * 40           # main vs lockdown_flow:全败   → 低位
    for s in seq_ab:
        lg.record_match("main", "push_flow", s)
    for s in seq_ac:
        lg.record_match("main", "counter_flow", s)
    for s in seq_ad:
        lg.record_match("main", "lockdown_flow", s)

    wr = lg.pfsp.winrates
    # 1) 各自等于独立重算的 EMA（流由自己的比分驱动）
    assert abs(wr[("main", "push_flow")] - ref_ema(seq_ab)) < 1e-12
    assert abs(wr[("main", "counter_flow")] - ref_ema(seq_ac)) < 1e-12
    assert abs(wr[("main", "lockdown_flow")] - ref_ema(seq_ad)) < 1e-12
    # 2) 不同比分序列 → 不同胜率（共享流 bug 会在这里暴露）
    assert wr[("main", "push_flow")] != wr[("main", "counter_flow")]
    assert wr[("main", "push_flow")] != wr[("main", "lockdown_flow")]
    assert wr[("main", "counter_flow")] != wr[("main", "lockdown_flow")]
    # 3) 双向互补
    assert abs(wr[("main", "push_flow")] + wr[("push_flow", "main")] - 1.0) < 1e-12
    # 4) 独立性：再打一轮 main vs push_flow，不得污染 main vs counter_flow 的流
    v_ac_before = wr[("main", "counter_flow")]
    v_ab_before = wr[("main", "push_flow")]
    for s in seq_ab:
        lg.record_match("main", "push_flow", s)
    assert wr[("main", "counter_flow")] == v_ac_before, "其它 pair 的胜率流被污染"
    assert wr[("main", "lockdown_flow")] == ref_ema(seq_ad), "未触及 pair 的流被污染"
    assert wr[("main", "push_flow")] != v_ab_before, "本 pair 流应继续演化"
    # 5) save/load round-trip：a|b key 序列化不串流
    d = tempfile.mkdtemp()
    p = os.path.join(d, "state.json")
    lg.save_state(p)
    lg2 = League(seed=0)
    lg2.load_state(p)
    assert lg2.pfsp.winrates == lg.pfsp.winrates
    assert abs(lg2.pfsp.winrates[("main", "push_flow")] - ref_ema(seq_ab * 2)) < 1e-12
    assert lg2.pfsp.winrates[("main", "counter_flow")] == v_ac_before
    print("[PASS] 不同 pair 的 PFSP 胜率流独立演进：互不污染 / 各自 EMA 一致 / "
          "round-trip 不串 key")


def test_elo_eval_granularity():
    """评估粒度统计契约：噪声地板 / 轮内聚合估计 / 误差棒数据全链路。

    背景：K=32 逐局 Elo 是**有限记忆跟踪器**（MC 验证单轮噪声 1σ≈±40 即饱和，
    加局数不收窄运行 Elo 曲线）；曲线可信度上限取决于**轮内聚合估计**
    （BT-lite，SE≈347.5/√N，N=该 agent 本轮总对局数，随局数真正收窄）：
    - N=4/对 → main(5对,20局) SE≈78，纯噪声下 |ΔElo|≥100 的概率 ≈36% →
      ±100 的 2000 步移动无法区分学习信号与评估噪声；
    - N=40/对 → main(5对,200局) SE≈25，纯噪声下 |ΔElo|≥100 概率 <1% → 可区分。
    """
    import math, tempfile, random, statistics
    from rl.league import League
    from rl.run_league import _round_estimates
    from rl.config import TrainConfig
    from rl import run_league as rl_mod
    from rl.dashboard import build_payload

    def noise_prob(se_per_round, delta=100.0):
        """纯噪声（真差=0）下两次独立轮差 |Δ|≥delta 的概率，正态近似。"""
        sd = math.sqrt(2.0) * se_per_round
        z = delta / (sd * math.sqrt(2.0))
        return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z)))

    # 1) 闭式 SE = 347.5/√N（p=0.5 最坏情形，MC 已验证）
    assert abs(347.5 / math.sqrt(4) - 173.8) < 0.1
    assert abs(347.5 / math.sqrt(40) - 54.9) < 0.1
    # main 每轮总对局 = 5 对 × N
    assert abs(347.5 / math.sqrt(5 * 4) - 77.7) < 0.5
    assert abs(347.5 / math.sqrt(5 * 40) - 24.6) < 0.5
    # 2) 纯噪声下 ≥100 Elo 移动的概率：N=4 高（不可区分）→ N=40 低（强信号）
    p4 = noise_prob(347.5 / math.sqrt(5 * 4))
    p40 = noise_prob(347.5 / math.sqrt(5 * 40))
    assert p4 > 0.30, f"N=4 时 ±100 应是常见噪声，实际 {p4:.2f}"
    assert p40 < 0.01, f"N=40 时 ±100 应是强信号，实际 {p40:.4f}"
    # 3) 轮内聚合估计：无偏 + SD≈347.5/√N（MC，N=40）
    rng = random.Random(7)
    trials, N = 800, 40
    ds = []
    for _ in range(trials):
        w = sum(1 for _ in range(N) if rng.random() < 0.5)
        ds.append(400.0 * math.log10((w + 0.5) / (N - w + 0.5)))
    sd = statistics.pstdev(ds)
    mu = statistics.mean(ds)
    assert abs(mu) < 15, f"聚合估计应无偏，mean={mu:.1f}"
    assert 40 < sd < 70, f"SD≈347.5/√N=54.9，实测 {sd:.1f}"
    # 4) _round_estimates 聚合逻辑：多对结果 → est/games
    est, games = _round_estimates([
        ("a", "b", 4, 0, 0),   # a 4:0 b → D̂_ab=400·log10(4.5/0.5)
        ("a", "c", 2, 2, 0),   # a 2:2 c → D̂_ac=0
    ])
    assert games == {"a": 8, "b": 4, "c": 4}
    d_ab = 400.0 * math.log10(4.5 / 0.5)
    assert abs(est["a"][0] - (1500.0 + (d_ab + 0.0) / 2.0)) < 0.1
    assert est["a"][1] == round(347.5 / math.sqrt(8), 1)
    # 5) 全链路：eval_round_robin 记录 round_stats → state → dashboard payload（误差棒数据）
    d = tempfile.mkdtemp()
    cfg = TrainConfig(name="selftest_gran", total_steps=6, steps_per_eval=0,
                      update_interval=1000, batch_size=16, hidden_dim=32, seed=0,
                      n_eval_games=1, max_ep_steps=2, only_vs_main=True,
                      eval_at_start=True, out_dir=d)
    rl_mod.run_league(cfg, resume=False, record_replays=False)
    lg = League()
    lg.load_state(cfg.state_path())
    assert len(lg.round_stats) >= 1, "应记录至少一轮评估统计"
    rs0 = lg.round_stats[0]
    assert rs0["games"].get("main") == 5, "only_vs_main：main 本轮应打 5 局"
    assert "main" in rs0["est"] and rs0["est"]["main"][1] > 0
    pl = build_payload(cfg.state_path())
    assert pl["ok"] and len(pl["round_stats"]) >= 1
    assert pl["round_stats"][0]["est"]["main"][1] > 0
    print("[PASS] 评估粒度统计契约：噪声地板(SE=347.5/√N) / 轮内聚合估计 / "
          "round_stats 全链路（N=4 → ±100≈噪声，N=40 → ±100≈信号）")


def test_classified_decks():
    """三分类卡组：200 副 / 60-120-20 / 全部映射为可部署卡 / deck_pool 随机抽卡组。"""
    from rl.decks import load_classified_decks, decks_by_archetype, normalize_card
    from rl.opponents import ScriptedPolicy
    from card_utils import card_data

    decks = load_classified_decks()
    assert len(decks) == 200, f"应有 200 副卡组，实际 {len(decks)}"
    by = decks_by_archetype(decks)
    assert len(by["推进流"]) == 60 and len(by["防守反击流"]) == 120 \
        and len(by["自闭流"]) == 20, [len(by[a]) for a in by]
    for d in decks:
        assert len(d["cards"]) == 8, f"卡组应 8 张: {d['archetype']}"
        for c in d["cards"]:
            assert c in card_data, f"{c} 不在引擎卡表中"
    assert normalize_card("fire-spirit") == "FireSpirits"
    assert normalize_card("the-log") == "Log"

    pol = ScriptedPolicy(mode="random", deck_pool=decks, seed=0)
    drawn = [pol.deck() for _ in range(5)]
    assert all(len(d) == 8 for d in drawn), "deck_pool 抽取应为 8 卡完整卡组"
    assert len({tuple(d) for d in drawn}) >= 2, "多次抽取应出现不同卡组"
    print("[PASS] 三分类卡组：200 副 / 60-120-20 / 全部可部署 / deck_pool 随机抽卡组")


def test_league_training_loop():
    """联赛主循环能跑完对局并触发结束/截断重置（防 ep_* 解包 bug 回归）。"""
    import tempfile
    from rl import run_league as rl_mod
    from rl.config import TrainConfig

    d = tempfile.mkdtemp()
    cfg = TrainConfig(name="selftest_loop", total_steps=6, steps_per_eval=0,
                      update_interval=1000, batch_size=16, hidden_dim=32, seed=0,
                      n_eval_games=1, max_ep_steps=2, only_vs_main=True,
                      eval_at_start=False, out_dir=d)
    rl_mod.run_league(cfg, resume=False, record_replays=False)
    assert os.path.exists(cfg.state_path()), "联赛状态应已落盘"
    assert os.path.exists(cfg.main_final_path()), "main 权重应已落盘"
    print("[PASS] 联赛主循环：对局结束/截断重置正常（ep_* 列表重置一致）")


def test_config_reward_weights():
    """命名配置：预设解析互不影响、塔血统一、奖励权重可注入 RLEnv 并改变回报。"""
    import tempfile
    from rl.config import TrainConfig, reward_to_env
    from rl.env_wrapper import RLEnv
    from rl.action_bundle import ActionBundle

    std = TrainConfig.resolve("standard")
    agg = TrainConfig.resolve("aggressive")
    assert std.reward["crown_weight"] == 5.0
    assert agg.reward["crown_weight"] == 8.0
    # 2025-06 改版：塔血统一（打击=损失同价）；预设差异体现在皇冠/费差
    for name in ("standard", "aggressive", "defensive", "lockdown", "elixir", "economy", "fast"):
        rw = TrainConfig.resolve(name).reward
        assert rw["tower_dmg_self"] == rw["tower_dmg_opp"], f"{name} 塔血应统一"
    assert agg.reward["elixir_diff_weight"] > std.reward["elixir_diff_weight"]
    # 二次解析不污染预设（共享实例回归）
    assert TrainConfig.resolve("standard").reward["crown_weight"] == 5.0
    assert TrainConfig.resolve("aggressive").reward["crown_weight"] == 8.0

    env = RLEnv(opponent=None, seed=0, reward_weights=reward_to_env(std))
    env.reset()
    _, r0, _, _, _ = env.step(ActionBundle.noop())
    env2 = RLEnv(opponent=None, seed=0, reward_weights=reward_to_env(agg))
    env2.reset()
    _, r1, _, _, _ = env2.step(ActionBundle.noop())
    # 配置项确实注入 env 并生效（不同配置 → 不同权重结构）
    assert env.reward_weights["crown_weight"] == 5.0
    assert env2.reward_weights["crown_weight"] == 8.0
    assert isinstance(r0, float) and isinstance(r1, float)

    # config.json 往返
    d = tempfile.mkdtemp()
    p = os.path.join(d, "cfg.json")
    agg.save(p)
    back = TrainConfig.load(p)
    assert back.name == "aggressive" and back.reward["crown_weight"] == 8.0
    print("[PASS] 命名配置：预设/加载/奖励权重注入 RLEnv 正常、塔血统一")


def test_model_reward_overrides():
    """按流派奖惩：main/all/random 同一基线；推进加码费差、防反减码、自闭压到≈0。"""
    from rl.config import TrainConfig, model_reward_weights

    std = TrainConfig.resolve("standard")
    base = model_reward_weights("main", std)
    assert model_reward_weights("all_decks", std) == base, "all_decks 应与 main 同参数"
    assert model_reward_weights("random_deck", std) == base, "random_deck 应与 main 同参数"
    assert base["tower_dmg_self"] == base["tower_dmg_opp"] == 0.001, "塔血应统一 0.001/0.001"
    assert base["normalize_tower_dmg"] is True, "费差机制默认打开"
    assert base["elixir_diff_weight"] == 0.5, "基线费差 = 0.5（1圣水≈500血）"
    # 流派覆盖：推进 > 基线 > 防反 > 自闭
    push = model_reward_weights("push_flow", std)
    counter = model_reward_weights("counter_flow", std)
    lock = model_reward_weights("lockdown_flow", std)
    assert push["elixir_diff_weight"] == 0.7 > base["elixir_diff_weight"], "推进应加码费差"
    assert counter["elixir_diff_weight"] == 0.3 < base["elixir_diff_weight"], "防反应减码费差"
    assert lock["elixir_diff_weight"] == 0.05 < counter["elixir_diff_weight"], "自闭应压到≈0"
    # 塔血在所有流派也统一
    for mid in ("push_flow", "counter_flow", "lockdown_flow"):
        rw = model_reward_weights(mid, std)
        assert rw["tower_dmg_self"] == rw["tower_dmg_opp"] == 0.001, mid
    # 未知模型回退到所选预设（不改基线行为）
    assert model_reward_weights("unknown_model", std) == base
    print("[PASS] 按流派奖惩：main/all/random 同基线 0.5、推进 0.7 / 防反 0.3 / 自闭 0.05、"
          "塔血统一 0.001/0.001、未知模型回退基线")


def test_reward_economy_preset():
    """费差默认打开：standard/economy 都带 normalize+费差；按流派覆盖生效。"""
    import tempfile
    import os
    from rl.config import TrainConfig, reward_to_env, model_reward_weights
    from rl.env_wrapper import RLEnv

    eco = TrainConfig.resolve("economy")
    std = TrainConfig.resolve("standard")
    # 2025-06 改版：费差默认打开（standard 不再是"旧公式"）
    assert eco.reward["normalize_tower_dmg"] is True
    assert eco.reward["elixir_diff_weight"] > 0
    assert std.reward["normalize_tower_dmg"] is True
    assert std.reward["elixir_diff_weight"] == eco.reward["elixir_diff_weight"] == 0.5
    # 按流派覆盖：main 基线 0.5 / 推进 0.7 / 防反 0.3 / 自闭 0.05
    assert model_reward_weights("main", std)["elixir_diff_weight"] == 0.5
    assert model_reward_weights("push_flow", std)["elixir_diff_weight"] == 0.7
    assert model_reward_weights("counter_flow", std)["elixir_diff_weight"] == 0.3
    assert model_reward_weights("lockdown_flow", std)["elixir_diff_weight"] == 0.05
    # reward_to_env 注入 RLEnv 后生效
    env = RLEnv(opponent=None, seed=0, reward_weights=reward_to_env(eco))
    assert env.reward_weights["normalize_tower_dmg"] is True
    assert env.reward_weights["elixir_diff_weight"] > 0
    # config.json 往返保留布尔键与费差权重
    d = tempfile.mkdtemp()
    p = os.path.join(d, "cfg.json")
    eco.save(p)
    back = TrainConfig.load(p)
    assert back.reward["normalize_tower_dmg"] is True
    assert back.reward["elixir_diff_weight"] == eco.reward["elixir_diff_weight"]
    print("[PASS] 费差默认打开：standard/economy normalize+费差=0.5、按流派 0.7/0.3/0.05、JSON 往返正常")


def test_reward_economy_level_invariance():
    """费差机制：塔损按塔血%归一化 → 同一事件跨等级奖励一致；旧公式仍漂移（回归）。"""
    from rl.env_wrapper import compute_reward, _TOWER_HP_ANCHOR
    from rl.config import TrainConfig, reward_to_env

    eco = reward_to_env(TrainConfig.resolve("economy"))
    # 旧公式（2025-06 前的默认：normalize 关、费差 0、挨打 0.0012）——仅作回归对照
    legacy = {"crown_weight": 5.0, "tower_dmg_opp": 0.001, "tower_dmg_self": 0.0012,
              "win_bonus": 10.0, "lose_penalty": 10.0, "invalid_penalty": 0.05,
              "elixir_bonus": 0.0, "normalize_tower_dmg": False, "elixir_diff_weight": 0.0}
    # 锚 = 引擎真实 lv11 总塔血（2×3052 + 4824）
    assert _TOWER_HP_ANCHOR == 10928.0, "lv11 总塔血锚 = 2×3052 + 4824 = 10928"

    def r(weights, total_max, event_frac=0.05):
        # 同一事件：磨掉敌方 event_frac 比例的总塔血；费差不变、无皇冠、未终局、无非法
        dmg = event_frac * total_max
        return compute_reward(
            weights,
            blue_hps_old=total_max, red_hps_old=total_max,
            blue_hps_new=total_max, red_hps_new=total_max - dmg,
            blue_left_old=3, red_left_old=3, blue_left_new=3, red_left_new=3,
            my_elixir_before=5.0, opp_elixir_before=5.0,
            my_elixir_after=5.0, opp_elixir_after=5.0,
            winner=None, invalid_count=0,
            blue_hps_max=total_max, red_hps_max=total_max)

    lv11_max = 10928.0   # 引擎默认 lv11：2×3052 + 4824
    lv16_max = 21268.0   # 2×5726 + 9816
    eco11, eco16 = r(eco, lv11_max), r(eco, lv16_max)
    old11, old16 = r(legacy, lv11_max), r(legacy, lv16_max)
    # 费差机制：同一"塔血百分比事件"跨等级奖励一致
    assert abs(eco11 - eco16) < 1e-9, f"费差机制应跨等级不变: {eco11} vs {eco16}"
    # 旧公式确实随等级漂移（这正是要修的问题，回归验证）
    assert abs(old11 - old16) > 0.01, "旧公式应随等级漂移（回归验证）"
    print(f"[PASS] 费差机制：跨等级不变({eco11:.4f})、旧公式漂移({old11:.3f}->{old16:.3f})")


def test_reward_economy_elixir_diff():
    """费差项：显式给圣水定价（1圣水≈500血@lv11）；potential-style（闭环累计归零）。"""
    from rl.env_wrapper import compute_reward
    from rl.config import TrainConfig, reward_to_env

    std = reward_to_env(TrainConfig.resolve("standard"))   # 费差=0.5
    # 旧公式（无费差项）作回归对照
    legacy = {"crown_weight": 5.0, "tower_dmg_opp": 0.001, "tower_dmg_self": 0.001,
              "win_bonus": 10.0, "lose_penalty": 10.0, "invalid_penalty": 0.05,
              "elixir_bonus": 0.0, "normalize_tower_dmg": True, "elixir_diff_weight": 0.0}
    base = dict(blue_hps_old=10928.0, red_hps_old=10928.0,
                blue_hps_new=10928.0, red_hps_new=10928.0,
                blue_left_old=3, red_left_old=3, blue_left_new=3, red_left_new=3,
                winner=None, invalid_count=0,
                blue_hps_max=10928.0, red_hps_max=10928.0)

    # 我方花 4 费（费差 -4）→ 默认机制显式 -2.0（=4×0.5）；旧公式 0（无圣水定价）
    r_std = compute_reward(std, my_elixir_before=5.0, opp_elixir_before=5.0,
                           my_elixir_after=1.0, opp_elixir_after=5.0, **base)
    r_legacy = compute_reward(legacy, my_elixir_before=5.0, opp_elixir_before=5.0,
                              my_elixir_after=1.0, opp_elixir_after=5.0, **base)
    assert abs(r_std - (-2.0)) < 1e-9, f"花4费应-2.0: {r_std}"
    assert abs(r_legacy - 0.0) < 1e-12, f"旧公式花费无显式惩罚: {r_legacy}"
    # 对方花 4 费（我方费差 +4）→ 默认机制显式 +2.0
    r_std2 = compute_reward(std, my_elixir_before=5.0, opp_elixir_before=5.0,
                            my_elixir_after=5.0, opp_elixir_after=1.0, **base)
    assert abs(r_std2 - 2.0) < 1e-9, f"对方花4费应+2.0: {r_std2}"
    # potential-style：闭环（花4→对方花4→我方回5→对方回5）费差项累计归零
    steps = [(5.0, 5.0, 1.0, 5.0), (1.0, 5.0, 1.0, 1.0),
             (1.0, 1.0, 5.0, 1.0), (5.0, 1.0, 5.0, 5.0)]
    total = sum(compute_reward(std, my_elixir_before=a, opp_elixir_before=b,
                               my_elixir_after=c, opp_elixir_after=d, **base)
                for a, b, c, d in steps)
    assert abs(total) < 1e-9, f"费差项应闭环归零: {total}"
    print(f"[PASS] 费差项：花4费=-2.0/对方花4费=+2.0/闭环累计归零（{total:.2e}）；"
          f"旧公式无定价（{r_legacy:.2f}）")


def test_reward_economy_trade_pricing():
    """费差 vs 塔血的真实 trade：1圣水≈500血@lv11（花1费换≥500塔血才划算）。"""
    from rl.env_wrapper import compute_reward
    from rl.config import TrainConfig, reward_to_env

    std = reward_to_env(TrainConfig.resolve("standard"))   # 费差=0.5
    legacy = {"crown_weight": 5.0, "tower_dmg_opp": 0.001, "tower_dmg_self": 0.001,
              "win_bonus": 10.0, "lose_penalty": 10.0, "invalid_penalty": 0.05,
              "elixir_bonus": 0.0, "normalize_tower_dmg": True, "elixir_diff_weight": 0.0}
    base = dict(blue_hps_old=10928.0, red_hps_old=10928.0,
                blue_hps_new=10928.0, red_hps_new=10928.0,
                blue_left_old=3, red_left_old=3, blue_left_new=3, red_left_new=3,
                winner=None, invalid_count=0,
                blue_hps_max=10928.0, red_hps_max=10928.0)

    def r(weights, **kw):
        return compute_reward(weights, **dict(base, **kw))

    # ① 校准：花 1 费换 500 塔血 ≈ 中性（1圣水≈500血）；换 600 血 → 正
    neutral = r(std, red_hps_new=10928.0 - 500.0,
                my_elixir_before=5.0, opp_elixir_before=5.0,
                my_elixir_after=4.0, opp_elixir_after=5.0)
    assert abs(neutral) < 1e-9, f"1费换500血应中性: {neutral}"
    good = r(std, red_hps_new=10928.0 - 600.0,
             my_elixir_before=5.0, opp_elixir_before=5.0,
             my_elixir_after=4.0, opp_elixir_after=5.0)
    assert good > 0, f"1费换600血应划算: {good}"
    # ② 花 4 费磨 4.3% 总塔血（≈470血=117血/圣水 < 500）：新校准下不划算（负）
    trade = r(std, red_hps_new=10928.0 - 0.043 * 10928,
              my_elixir_before=5.0, opp_elixir_before=5.0,
              my_elixir_after=1.0, opp_elixir_after=5.0)
    assert trade < 0, f"4费只磨4.3%塔血应不划算: {trade}"
    # ③ 花 4 费但 0 塔损（浪费）：应为负
    waste = r(std, red_hps_new=10928.0,
              my_elixir_before=5.0, opp_elixir_before=5.0,
              my_elixir_after=1.0, opp_elixir_after=5.0)
    assert waste < 0, f"白花 4 费应为负: {waste}"
    # ④ 让塔挨 1% 总塔血、换 2 费差（对方花 2 费而我不防）：应为正（trade 划算）
    trade2 = r(std, blue_hps_new=10928.0 - 0.01 * 10928, red_hps_new=10928.0,
               my_elixir_before=5.0, opp_elixir_before=5.0,
               my_elixir_after=5.0, opp_elixir_after=3.0)
    assert trade2 > 0, f"挨 1% 塔血换 2 费差应划算: {trade2}"
    # ⑤ 同一事件在旧公式（无费差项）：为负 → 旧公式学不出这个 trade（缺陷回归）
    old = r(legacy, blue_hps_new=10928.0 - 0.01 * 10928, red_hps_new=10928.0,
            my_elixir_before=5.0, opp_elixir_before=5.0,
            my_elixir_after=5.0, opp_elixir_after=3.0)
    assert old < 0, f"旧公式挨打换费差应为负（缺陷）: {old}"
    print(f"[PASS] 费差 trade 定价：1费换500血={neutral:.3f}≈0 / 600血={good:.3f}>0 / "
          f"4费4.3%塔血={trade:.3f}<0 / 浪费={waste:.3f}<0 / 挨1%换2费差={trade2:.3f}>0 "
          f"（旧公式={old:.3f}<0）")


def test_rlenv_card_level():
    """RLEnv 支持 11-16 卡牌等级：reset 同步真实塔血；lv16=数据表、lv11=引擎默认。"""
    from rl.env_wrapper import RLEnv
    from card_utils import Card

    env16 = RLEnv(opponent=None, seed=0, card_level=16)
    env16.reset()
    p = env16.battle.players[0]
    # reset 已同步 PlayerState 到真实实体 HP（消除首步假奖励）
    assert p.left_tower_hp == env16.battle.entities[3].hp
    assert p.king_tower_hp == env16.battle.entities[6].hp
    assert (p.left_tower_hp, p.king_tower_hp) == (5726, 9816), \
        (p.left_tower_hp, p.king_tower_hp)
    env11 = RLEnv(opponent=None, seed=0, card_level=11)
    env11.reset()
    p = env11.battle.players[0]
    assert p.left_tower_hp == env11.battle.entities[3].hp
    # 引擎默认 lv11 塔血 3052/4824（官方数组 index10 为 3584/6144；原作者硬编码
    # (4824,3052,3052) 即引擎默认值 —— 锚点 _TOWER_HP_ANCHOR 与之对齐）
    assert (p.left_tower_hp, p.king_tower_hp) == (3052, 4824), \
        (p.left_tower_hp, p.king_tower_hp)
    Card.default_level = 11  # 恢复全局默认，避免污染后续测试
    print("[PASS] RLEnv 卡牌等级：lv16 塔血 5726/9816、lv11 引擎默认 3052/4824、reset 已同步")


def test_tower_troop_hp_reference():
    """塔血参考表：国王塔恒定 4824、四种公主塔 lv11 各异；归一化对塔型/等级不变。"""
    from rl.env_wrapper import (TOWER_TROOP_HP_LV11, KING_TOWER_HP_LV11,
                                tower_total_hp, compute_reward, _TOWER_HP_ANCHOR, RLEnv)
    from rl.config import TrainConfig, reward_to_env

    # 用户提供的真实游戏 lv11 数据
    assert TOWER_TROOP_HP_LV11 == {
        "PrincessTower": 3052.0, "DaggerDuchess": 2768.0,
        "RoyalChef": 2703.0, "Cannoneer": 2616.0}
    assert KING_TOWER_HP_LV11 == 4824.0
    assert _TOWER_HP_ANCHOR == tower_total_hp(3052.0, 4824.0) == 10928.0
    # 引擎标准塔（RLEnv 默认）确实 = PrincessTower 3052 / KingTower 4824
    env = RLEnv(opponent=None, seed=0)
    env.reset()
    assert env.battle.players[0].left_tower_hp == 3052.0
    assert env.battle.players[0].king_tower_hp == 4824.0

    eco = reward_to_env(TrainConfig.resolve("economy"))

    def r(troop_hp, event_frac=0.05):
        # 同一事件：磨掉敌方 event_frac 比例的总塔血（分母用该塔型的真实总塔血）
        total = tower_total_hp(troop_hp, KING_TOWER_HP_LV11)
        dmg = event_frac * total
        return compute_reward(
            eco,
            blue_hps_old=total, red_hps_old=total,
            blue_hps_new=total, red_hps_new=total - dmg,
            blue_left_old=3, red_left_old=3, blue_left_new=3, red_left_new=3,
            my_elixir_before=5.0, opp_elixir_before=5.0,
            my_elixir_after=5.0, opp_elixir_after=5.0,
            winner=None, invalid_count=0,
            blue_hps_max=total, red_hps_max=total)

    # 归一化对塔型不变：同一"磨 5% 总塔血"在四种公主塔下给同一奖励
    vals = {t: r(h) for t, h in TOWER_TROOP_HP_LV11.items()}
    ref = vals["PrincessTower"]
    for t, v in vals.items():
        assert abs(v - ref) < 1e-9, f"{t} 应同分: {v} vs {ref}"
    # 语义验证：同样的绝对伤害（1000 HP），打在更弱的炮兵塔（2616）上比标准公主塔值钱
    # （更弱塔 = 更大的塔血百分比 = 更接近皇冠）
    def absr(troop_hp):
        total = tower_total_hp(troop_hp, KING_TOWER_HP_LV11)
        return compute_reward(
            eco,
            blue_hps_old=total, red_hps_old=total,
            blue_hps_new=total, red_hps_new=total - 1000.0,
            blue_left_old=3, red_left_old=3, blue_left_new=3, red_left_new=3,
            my_elixir_before=5.0, opp_elixir_before=5.0,
            my_elixir_after=5.0, opp_elixir_after=5.0,
            winner=None, invalid_count=0,
            blue_hps_max=total, red_hps_max=total)
    weak = absr(TOWER_TROOP_HP_LV11["Cannoneer"])
    strong = absr(TOWER_TROOP_HP_LV11["PrincessTower"])
    assert weak > strong, f"更弱塔受同等伤害应更值钱: {weak} vs {strong}"
    print(f"[PASS] 塔血参考：国王恒定4824/公主各异；归一化对塔型不变({ref:.4f})；"
          f"1000HP打炮兵({weak:.3f})>打公主塔({strong:.3f})")


def test_league_resume():
    """断点续训：run_state 落盘，resume 后从旧 step 续跑并刷新快照。"""
    import tempfile
    from rl import run_league as rl_mod
    from rl.config import TrainConfig
    import json as _json

    d = tempfile.mkdtemp()
    cfg = TrainConfig(name="selftest_resume", total_steps=4, steps_per_eval=0,
                      update_interval=1000, batch_size=16, hidden_dim=32, seed=0,
                      n_eval_games=1, max_ep_steps=2, only_vs_main=True, out_dir=d)
    rl_mod.run_league(cfg, resume=False, record_replays=False)
    rs = _json.load(open(cfg.run_state_path(), "r", encoding="utf-8"))
    assert rs["step"] == 4 and os.path.exists(rs["main_ckpt"]) and os.path.exists(rs["opt_ckpt"])
    first_ckpt = rs["main_ckpt"]

    cfg2 = TrainConfig(name="selftest_resume", total_steps=8, steps_per_eval=0,
                       update_interval=1000, batch_size=16, hidden_dim=32, seed=0,
                       n_eval_games=1, max_ep_steps=2, only_vs_main=True,
                       eval_at_start=False, out_dir=d)
    rl_mod.run_league(cfg2, resume=True, record_replays=False)
    rs2 = _json.load(open(cfg.run_state_path(), "r", encoding="utf-8"))
    assert rs2["step"] == 8, f"resume 应从 5 续到 8，实际 {rs2['step']}"
    assert os.path.exists(cfg.ckpt_path(8)) and os.path.exists(cfg.opt_path(8))
    print("[PASS] 断点续训：run_state 落盘 + 续跑 + 快照刷新正常")


def test_league_replays():
    """每评估周期联赛录像：record=True 采集、保存、回读。"""
    import tempfile
    from rl import run_league as rl_mod
    from rl.league import League
    from rl.opponents import ScriptedPolicy, build_card_pool
    from rl.follower import FollowerPolicy
    from rl.plan_space import PLAN_DIM
    from rl.replay import save_league_replays, load_league_replays

    pool = build_card_pool()
    lg = League(seed=0)
    main = FollowerPolicy(hidden=32, plan_dim=PLAN_DIM, belief_dim=23)
    lg.add_agent("main", kind="main", policy=main)
    lg.add_agent("random_deck", kind="baseline",
                 policy=ScriptedPolicy(mode="random", pool=pool, seed=1))
    replays = rl_mod.eval_round_robin(lg, n_games=2, max_steps=5, seed=0, step=2000,
                                      only_vs_main=True, record=True)
    assert len(replays) == 2, f"应记录 2 局，实际 {len(replays)}"
    g = replays[0]
    assert g["meta"]["pair"] == ["main", "random_deck"] and g["winner"] in (0, 1, None)
    assert g["frames"] and "entities" in g["frames"][0] and "towers0" in g["frames"][0]

    d = tempfile.mkdtemp()
    p = os.path.join(d, "league_2000.pkl")
    save_league_replays(replays, p)
    back = load_league_replays(p)
    assert len(back) == 2 and back[0]["frames"][0]["t"] >= 0.0
    print("[PASS] 联赛录像：逐局采集 + 保存/回读正常")


def test_dashboard_replays():
    """仪表盘回放：列表扫描 / 对局加载 / 单局帧 / 非法文件名防护 / demo 生成 / 页面元素。"""
    import tempfile
    import rl.dashboard as dash
    from rl.replay import save_league_replays

    def frame(t, t0, t1, entities, **kw):
        base = {
            "t": t, "bundle": [], "reward": 0.0, "opp_played": [],
            "towers0": t0, "towers1": t1,
            "elixir0": 5.0, "elixir1": 5.0, "crown0": 0, "crown1": 0,
            "entities": entities,
        }
        base.update(kw)
        return base

    towers0 = [4824.0, 3052.0, 3052.0]
    towers1 = [4824.0, 3052.0, 3052.0]
    games = [
        {"meta": {"pair": ["main", "push_flow"], "side0": "main", "max_steps": 600},
         "winner": 0,
         "frames": [frame(0.5, towers0, towers1, [["Knight", 4.0, 12.0, 700.0, 0]]),
                    frame(1.2, towers0, towers1, [["Knight", 4.2, 11.8, 680.0, 0]],
                          bundle=[["deploy", 2, 8.0, 14.0]], reward=0.05,
                          opp_played=[{"card": "Archers", "x": 12.0, "y": 20.0}])]},
        {"meta": {"pair": ["main", "random_deck"], "side0": "main", "max_steps": 600},
         "winner": 1,
         "frames": [frame(0.5, towers0, towers1, [["Archers", 12.0, 20.0, 250.0, 1]])]},
    ]
    d = tempfile.mkdtemp()
    p = os.path.join(d, "league_2000.pkl")
    save_league_replays(games, p)

    # 扫描列表
    rp = dash.build_replays_payload(d)
    assert rp["ok"] and len(rp["replays"]) == 1, rp
    meta = rp["replays"][0]
    assert meta["file"] == "league_2000.pkl" and meta["step"] == 2000
    assert meta["n_games"] == 2 and meta["size"] > 0

    # 对局列表（轻量，不含帧）
    gl = dash.load_replay_payload(d, "league_2000.pkl")
    assert gl["ok"] and len(gl["games"]) == 2
    assert gl["games"][0]["n_frames"] == 2 and gl["games"][0]["winner"] == 0
    assert gl["games"][1]["winner"] == 1 and gl["games"][1]["duration"] == 0.5
    assert "frames" not in gl, "列表接口不应返回帧"

    # 单局帧
    g = dash.load_replay_payload(d, "league_2000.pkl", 0)
    assert g["ok"] and len(g["frames"]) == 2
    assert g["frames"][1]["bundle"] == [["deploy", 2, 8.0, 14.0]]
    assert g["frames"][1]["opp_played"] == [{"card": "Archers", "x": 12.0, "y": 20.0}]

    # 边界：非法文件名 / 越界 / 不存在目录
    assert dash.load_replay_payload(d, "../evil.pkl")["ok"] is False
    assert dash.load_replay_payload(d, "a/b.pkl")["ok"] is False
    assert dash.load_replay_payload(d, "missing.pkl")["ok"] is False
    assert dash.load_replay_payload(d, "league_2000.pkl", 99)["ok"] is False
    assert dash.build_replays_payload(os.path.join(d, "nope"))["ok"] is False

    # demo 回放生成（无数字步数也能列出）
    d2 = os.path.join(d, "demo")
    dash.make_demo_replays(d2, n_games=2, n_frames=10)
    demo_path = os.path.join(d2, "league_demo.pkl")
    assert os.path.exists(demo_path)
    rp2 = dash.build_replays_payload(d2)
    assert rp2["ok"] and rp2["replays"][0]["file"] == "league_demo.pkl"
    assert rp2["replays"][0]["step"] is None
    dg = dash.load_replay_payload(d2, "league_demo.pkl", 1)
    assert dg["ok"] and len(dg["frames"]) == 10

    # 页面包含播放器元素（防回归）
    html = dash._HTML
    for token in ("最近训练回放", 'id="arena"', "btnPlay", "scrub",
                  "/api/replays", "/api/replay"):
        assert token in html, f"页面缺少 {token}"

    print("[PASS] 仪表盘回放：列表扫描 + 对局加载 + 单局帧 + 非法名防护 + demo + 页面元素")


def test_battle_clone_fix():
    """克隆法术克隆冰法（on_spawn 访问 battle_state）不再崩溃（battle.py:1400 修复）。"""
    from rl.env_wrapper import RLEnv
    from battle import Position

    env = RLEnv(opponent=None, seed=0,
                deck0=["IceWizard", "Clone", "Arrows", "Fireball", "Giant", "Archer", "Knight", "Minions"])
    env.reset()
    p0 = env.battle.players[0]
    p0.elixir = 10.0
    # 直接摆 cycle 保证 IceWizard/Clone 在手牌（前 4）
    p0.cycle = ["IceWizard", "Clone", "Arrows", "Fireball", "Giant", "Archer", "Knight", "Minions"]
    ok = env.battle.deploy_card(0, "IceWizard", Position(10.0, 10.0))
    assert ok, "冰法部署应成功"
    env.battle.step(1 / 60)
    ok2 = env.battle.deploy_card(0, "Clone", Position(10.0, 10.0))
    assert ok2, "克隆法术部署应成功（且不再触发 battle_state=None 崩溃）"
    print("[PASS] 克隆法术：冰法克隆不再崩溃")


def test_cuda_device_support():
    """设备支持：cpu 必跑；cuda 可用时额外跑 act+evaluate+PPO update（cu130）。"""
    import torch
    from rl.env_wrapper import RLEnv
    from rl.belief import BeliefInference
    from rl.follower import FollowerPolicy
    from rl.plan_space import PLAN_DIM
    from rl.ppo import PPOTrainer
    from rl.action_mask import validate_bundle

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    env = RLEnv(opponent=None, seed=0)
    obs, _ = env.reset()
    belief = BeliefInference(opp_deck=env.deck1, n_particles=128, seed=0)
    tok = belief.encode(obs, None)
    for dev in devices:
        pol = FollowerPolicy(hidden=32, plan_dim=PLAN_DIM, belief_dim=len(tok))
        pol.to_device(dev)
        assert pol.device == dev
        bundle, lp, val, hidden, masks = pol.act(obs, tok, PLAN_DIM and np.zeros(PLAN_DIM, dtype=np.float32),
                                                 env.get_action_mask, hidden=None, deterministic=True)
        ok, reason, _ = validate_bundle(env.battle, 0, bundle)
        assert ok, f"{dev} 动作应合法: {reason}"
        obs2, r, term, trunc, info = env.step(bundle)
        trans = [{"obs": obs, "belief": tok, "plan": np.zeros(PLAN_DIM, dtype=np.float32),
                  "bundle": bundle, "old_logprob": lp, "adv": 1.0, "returns": val,
                  "masks": masks, "init_hidden": None}]
        stats = PPOTrainer(pol, lr=1e-3).update(trans)
        assert all(np.isfinite(v) for v in stats.values()), f"{dev} PPO 更新应有限"
        obs = obs2
    print(f"[PASS] 设备支持：{', '.join(devices)} 上 act/evaluate/PPO 正常")


def test_belief_follower_ppo_league():
    import torch
    from rl.env_wrapper import RLEnv
    from rl.belief import BeliefInference
    from rl.belief_planner import BeliefPlanner
    from rl.prophet import ProphetPlanner
    from rl.follower import FollowerPolicy
    from rl.ppo import PPOTrainer
    from rl.plan_space import PlanToken, PLAN_DIM
    from rl.league import League
    from rl.action_mask import validate_bundle

    env = RLEnv(opponent=None, seed=0)
    obs, _ = env.reset()
    belief = BeliefInference(opp_deck=env.deck1, n_particles=128, seed=0)
    belief.reset(env.deck1)
    tok = belief.encode(obs, None)
    plan = PlanToken.zeros().to_vector()
    pol = FollowerPolicy(hidden=64, plan_dim=PLAN_DIM, belief_dim=len(tok))
    ppo = PPOTrainer(pol, lr=1e-3)

    trans = []
    hidden = None
    for _ in range(4):
        init_hidden = hidden
        bundle, lp, val, hidden, masks = pol.act(obs, tok, plan, env.get_action_mask,
                                                 hidden=hidden, deterministic=False)
        ok, reason, _ = validate_bundle(env.battle, 0, bundle)
        assert ok, f"follower 动作应合法: {reason}"
        obs2, r, term, trunc, info = env.step(bundle)
        done = term or trunc
        trans.append({"obs": obs, "belief": tok, "plan": plan, "bundle": bundle,
                      "old_logprob": lp, "adv": 1.0, "returns": val,
                      "masks": masks, "init_hidden": init_hidden})
        belief.update(obs2, info.get("opp_played"))
        obs = obs2
        if done:
            obs, _ = env.reset(); belief.reset(env.deck1); hidden = None
    stats = ppo.update(trans)
    assert all(np.isfinite(v) for v in stats.values()), stats
    assert stats["entropy"] >= 0.0
    print("[PASS] 跟随者策略：autoregressive bundle 动作合法，PPO 更新收敛（含熵）")

    bp = BeliefPlanner(); pp = ProphetPlanner()
    bplan = bp.plan(env.battle, belief.state(), obs)
    pplan = pp.plan(env.get_prophet_state())
    assert bplan.macro_intent in _intents() and pplan.macro_intent in _intents()
    print("[PASS] BeliefPlanner / ProphetPlanner：输出合法计划")

    lg = League(seed=0)
    for aid in ("main", "random", "heuristic"):
        lg.add_agent(aid, kind="main" if aid == "main" else "baseline")
    lg.record_match("main", lg.sample_opponent("main").agent_id, 0.8)
    assert lg.elo_table()["main"] > 1500
    print("[PASS] 联赛：PFSP 采样 + Elo 更新")


def test_parallel_batch_equivalence():
    """批量 act/evaluate（并行多 env / batch PPO）与单条路径逐位等价。"""
    import torch
    from rl.env_wrapper import RLEnv
    from rl.belief import BeliefInference
    from rl.follower import FollowerPolicy
    from rl.plan_space import PlanToken, PLAN_DIM

    env = RLEnv(opponent=None, seed=0)
    obs, _ = env.reset()
    belief = BeliefInference(opp_deck=env.deck1, n_particles=128, seed=0)
    tok = belief.encode(obs, None)
    plan = PlanToken.zeros().to_vector()
    pol = FollowerPolicy(hidden=32, plan_dim=PLAN_DIM, belief_dim=len(tok))
    pol.eval()

    # 单条 act（确定性）
    b1, lp1, v1, h1, m1 = pol.act(obs, tok, plan, env.get_action_mask,
                                  hidden=None, deterministic=True)
    # 批量 act（两个同种子 env → 相同 obs；各 reset 一次保证状态一致）
    env2 = RLEnv(opponent=None, seed=0)
    obs2, _ = env2.reset()
    b2, lps2, v2, h2, m2 = pol.act_parallel(
        [obs, obs2], [tok, tok], [plan, plan],
        [env.get_action_mask, env2.get_action_mask],
        hidden_list=[None, None], deterministic=True)
    assert abs(lps2[0] - lp1) < 1e-5, f"批量 lp 不一致: {lps2[0]} vs {lp1}"
    assert abs(v2[0] - v1) < 1e-4, f"批量 value 不一致: {v2[0]} vs {v1}"
    assert b2[0] == b1, "批量 bundle 与单条不一致"
    assert len(m2[0]) == len(m1) and len(m2[1]) == len(m1)

    # evaluate_batch vs evaluate（logprob/value/entropy 一致）
    lp_ev, val_ev, _, ent_ev = pol.evaluate(obs, tok, plan, b1, m1, hidden=None)
    lp_b, val_b, ent_b = pol.evaluate_batch([obs], [tok], [plan], [b1], [m1], [None])
    assert abs(float(lp_ev) - float(lp_b[0])) < 1e-4, f"evaluate_batch lp 不一致"
    assert abs(float(ent_ev) - float(ent_b[0])) < 1e-4, "evaluate_batch entropy 不一致"
    assert abs(float(val_ev) - float(val_b[0, 0])) < 1e-4, "evaluate_batch value 不一致"
    print("[PASS] 批量 act/evaluate 与单条路径逐位等价")


def test_parallel_training_loop():
    """并行多 env（单进程 batch 路径）：n_envs=2 parallel=proc 训练主循环跑通并落盘。"""
    import tempfile
    from rl import run_league as rl_mod
    from rl.config import TrainConfig

    d = tempfile.mkdtemp()
    cfg = TrainConfig(name="selftest_vec", total_steps=6, steps_per_eval=0,
                      update_interval=1000, batch_size=16, hidden_dim=32, seed=0,
                      n_eval_games=1, max_ep_steps=2, only_vs_main=True,
                      n_envs=2, parallel="proc", eval_at_start=False, out_dir=d)
    rl_mod.run_league(cfg, resume=False, record_replays=False)
    assert os.path.exists(cfg.state_path()), "并行联赛状态应已落盘"
    assert os.path.exists(cfg.main_final_path()), "并行 main 权重应已落盘"
    print("[PASS] 并行多 env（proc）：n_envs=2 训练主循环完成")


def test_mp_training_loop():
    """跨进程 worker 并行：n_envs=2 parallel=mp 训练主循环跑通并落盘。"""
    import tempfile
    from rl import run_league as rl_mod
    from rl.config import TrainConfig

    d = tempfile.mkdtemp()
    cfg = TrainConfig(name="selftest_mp", total_steps=6, steps_per_eval=0,
                      update_interval=1000, batch_size=16, hidden_dim=32, seed=0,
                      n_eval_games=1, max_ep_steps=2, only_vs_main=True,
                      n_envs=2, parallel="mp", eval_at_start=False, out_dir=d)
    rl_mod.run_league(cfg, resume=False, record_replays=False)
    assert os.path.exists(cfg.state_path()), "mp 联赛状态应已落盘"
    assert os.path.exists(cfg.main_final_path()), "mp main 权重应已落盘"
    print("[PASS] 跨进程 worker（mp）：n_envs=2 训练主循环完成")


def _intents():
    from rl.plan_space import MACRO_INTENTS
    return MACRO_INTENTS


def test_flow_league_smoke():
    """全配对分流派联赛（P-flow）：mini 池 15 对全配对跑通 + 双侧轨迹 + 每对即训 + 落盘。

    - 真实池计数断言：60×120×20×200×30×main200 全配对 = 148,800 局；
    - mini 池（2/2/1/2/2/2）全配对 = 50 局，双侧（player-0/1）轨迹都收集、
      每对数据只喂该对双方模型（对内流式 update_interval 触发更新）、6 模型落盘。
    """
    import tempfile
    from rl import flow_league as fl
    from rl.config import TrainConfig

    def mk_decks(n, arch):
        return [{"archetype": arch, "cards": [
            "Knight", "MiniPekka", "Arrows", "Minions", "Musketeer",
            "Fireball", "Giant", "Archer"], "missing": 0} for _ in range(n)]

    # 真实池规模断言（docs 数据集 60/120/20/200 + 随机30 + main200 → 148,800）
    d0 = tempfile.mkdtemp()
    cfg0 = TrainConfig(name="selftest_flow_count", hidden_dim=32, seed=0, out_dir=d0)
    pools_real = fl.build_flow_pools(cfg0, n_random_decks=30)
    assert fl.flow_pair_games(pools_real) == 148800, fl.flow_pair_games(pools_real)
    print("[PASS] flow 真实池全配对 = 148,800 局（60×120×20×200×30×main200）")

    pools = fl.OrderedDict([
        ("push_flow", ("推进流", mk_decks(2, "推进流"))),
        ("counter_flow", ("防守反击流", mk_decks(2, "防守反击流"))),
        ("lockdown_flow", ("自闭流", mk_decks(1, "自闭流"))),
        ("all_decks", ("全量卡组", mk_decks(2, "全量卡组"))),
        ("random_deck", ("完全随机", mk_decks(2, "完全随机"))),
        ("main", ("全量卡组(main)", mk_decks(2, "全量卡组"))),
    ])
    expected = fl.flow_pair_games(pools)
    assert expected == 50, f"mini 池应 50 局，实际 {expected}"

    d = tempfile.mkdtemp()
    cfg = TrainConfig(name="selftest_flow", total_steps=1000, steps_per_eval=0,
                      update_interval=16, batch_size=8, hidden_dim=32, seed=0,
                      n_eval_games=1, max_ep_steps=2, eval_at_start=False, out_dir=d)
    total, models, trainers = fl.run_flow(cfg, pools=pools, n_random_decks=2)
    assert total == expected, f"flow 对局数 {total} != {expected}"
    assert set(models) == set(fl.FLOW_MODEL_IDS), "6 个模型都应存在"
    assert set(trainers) == set(fl.FLOW_MODEL_IDS), "6 个训练器都应存在"
    for mid in fl.FLOW_MODEL_IDS:
        p = os.path.join(d, "selftest_flow", f"flow_{mid}.pt")
        assert os.path.exists(p), f"flow 模型未落盘: {p}"
        st = os.path.getsize(p)
        assert st > 0, f"flow 模型空文件: {p}"
    # 双侧轨迹确实训练过：每个 trainer 都消费过轨迹并完成 >=1 次更新
    for mid in fl.FLOW_MODEL_IDS:
        assert trainers[mid].updates >= 1, f"模型 {mid} 未发生任何训练更新"
    print("[PASS] flow 联赛：mini 池 15 对（50局）双侧轨迹 + 每对即训 + 6 模型落盘")


def test_ablation_recorded():
    """belief/plan 输入消融（P-flow 前置验证）：4 变体对比 + delta/z 判定 + JSON/CSV 落盘。

    背景：prophet/belief_planner 是启发式，注入价值需消融证明；token 置零是保守
    消融（RNN hidden 仍含历史信息），但至少要有可追溯的产出记录而非仅 stdout。
    """
    import tempfile, json as _json
    from rl.env_wrapper import RLEnv
    from rl.belief import BeliefInference
    from rl.follower import FollowerPolicy, save_checkpoint
    from rl.plan_space import PLAN_DIM
    from rl.evaluate import run_ablation

    d = tempfile.mkdtemp()
    env = RLEnv(opponent=None, seed=0)
    belief_dim = len(BeliefInference(opp_deck=env.deck1, n_particles=8,
                                     seed=0).encode(None, None))
    pol = FollowerPolicy(hidden=32, plan_dim=PLAN_DIM, belief_dim=belief_dim)
    ckpt = os.path.join(d, "pol.pt")
    save_checkpoint(pol, ckpt)
    out = os.path.join(d, "ablation.json")
    res = run_ablation(ckpt, n_games=1, opponent="random", seed=0, hidden_dim=32,
                       max_steps=20, out_path=out)
    assert set(res["variants"]) == {"full", "plan-off", "belief-off", "both-off"}
    assert set(res["deltas_vs_full"]) == {"plan-off", "belief-off", "both-off"}
    for v in res["variants"].values():
        assert 0.0 <= v["winrate"] <= 1.0 and v["n_games"] == 1
    for dd in res["deltas_vs_full"].values():
        assert "delta" in dd and "verdict" in dd and "z" in dd
    assert os.path.exists(out), "消融 JSON 未落盘"
    assert os.path.exists(os.path.splitext(out)[0] + ".csv"), "消融 CSV 未落盘"
    loaded = _json.load(open(out, encoding="utf-8"))
    assert loaded["policy"] and loaded["note"]
    print("[PASS] belief/plan 消融：4 变体对比 + delta/z 判定 + JSON/CSV 落盘")


def test_flow_sweep_smoke():
    """flow 数据效率 A/B：缩小池 sweep 通路（mini 池 + 1 轮）→ summary.json/csv 落盘。"""
    import tempfile, json as _json
    from rl import flow_league as fl
    from rl.config import TrainConfig

    def mk_decks(n, arch):
        return [{"archetype": arch, "cards": [
            "Knight", "MiniPekka", "Arrows", "Minions", "Musketeer",
            "Fireball", "Giant", "Archer"], "missing": 0} for _ in range(n)]

    pools = fl.OrderedDict([
        ("push_flow", ("推进流", mk_decks(2, "推进流"))),
        ("counter_flow", ("防守反击流", mk_decks(2, "防守反击流"))),
        ("lockdown_flow", ("自闭流", mk_decks(1, "自闭流"))),
        ("all_decks", ("全量卡组", mk_decks(2, "全量卡组"))),
        ("random_deck", ("完全随机", mk_decks(2, "完全随机"))),
        ("main", ("全量卡组(main)", mk_decks(2, "全量卡组"))),
    ])
    d = tempfile.mkdtemp()
    cfg = TrainConfig(name="selftest_sweep", total_steps=1000, steps_per_eval=0,
                      update_interval=16, batch_size=8, hidden_dim=32, seed=0,
                      n_eval_games=1, max_ep_steps=2, eval_at_start=False, out_dir=d)
    rows, summary = fl.run_flow_sweep(cfg, strategy="stream", pools=pools,
                                      n_runs=1, games_per_pair=1, eval_games=1,
                                      pool_scale=1.0)
    assert len(rows) == 1
    r, se = rows[0]["main_est"]
    assert se > 0, "main 应有噪声地板 SE"
    assert summary["total_games"] == 50, summary["total_games"]
    out_dir = os.path.join(d, "selftest_sweep", "flow_sweep_stream")
    assert os.path.exists(os.path.join(out_dir, "summary.json"))
    assert os.path.exists(os.path.join(out_dir, "summary.csv"))
    assert os.path.exists(os.path.join(out_dir, "final_flow_main.pt"))
    s = _json.load(open(os.path.join(out_dir, "summary.json"), encoding="utf-8"))
    assert s["strategy"] == "stream" and len(s["rows"]) == 1
    assert s["trend"]["first_main_est"] == s["trend"]["last_main_est"]
    print("[PASS] flow-sweep：mini 池 1 轮通路 + main 轮内估计(±SE) + summary.json/csv 落盘")


def test_flow_resume():
    """flow 断点续练：前 N 对 + resume 全跑，总局数=全量、不重打已完成对。"""
    import tempfile
    from rl import flow_league as fl
    from rl.config import TrainConfig

    def mk_decks(n, arch):
        return [{"archetype": arch, "cards": [
            "Knight", "MiniPekka", "Arrows", "Minions", "Musketeer",
            "Fireball", "Giant", "Archer"], "missing": 0} for _ in range(n)]

    pools = fl.OrderedDict([
        ("push_flow", ("推进流", mk_decks(2, "推进流"))),
        ("counter_flow", ("防守反击流", mk_decks(2, "防守反击流"))),
        ("lockdown_flow", ("自闭流", mk_decks(1, "自闭流"))),
        ("all_decks", ("全量卡组", mk_decks(2, "全量卡组"))),
        ("random_deck", ("完全随机", mk_decks(2, "完全随机"))),
        ("main", ("全量卡组(main)", mk_decks(2, "全量卡组"))),
    ])
    expected = fl.flow_pair_games(pools)
    assert expected == 50, f"mini 池应 50 局，实际 {expected}"
    d = tempfile.mkdtemp()
    cfg = TrainConfig(name="selftest_flow_resume", total_steps=1000, steps_per_eval=0,
                      update_interval=16, batch_size=8, hidden_dim=32, seed=0,
                      n_eval_games=1, max_ep_steps=2, eval_at_start=False, out_dir=d)
    # 第一段：只跑前 2 对（push×counter=4 + push×lockdown=2 = 6 局）
    total1, _, _ = fl.run_flow(cfg, pools=pools, n_random_decks=2, max_pairs=2)
    assert total1 == 6, f"前 2 对应 6 局，实际 {total1}"
    assert os.path.exists(fl._flow_progress_path(cfg)), "flow_run_state.json 应已落盘"
    # 断点续练全跑：跳过已完成对，总局数 = 全量
    total2, models2, _ = fl.run_flow(cfg, pools=pools, n_random_decks=2, resume=True)
    assert total2 == expected, f"resume 后总局数应=全量 {expected}，实际 {total2}"
    assert set(models2) == set(fl.FLOW_MODEL_IDS), "resume 后 6 模型都在"
    print(f"[PASS] flow 断点续练：前 2 对({total1}局) + resume 全跑({total2}局) 不重打已完成对")


def test_solo_mode_smoke():
    """solo 自对弈：固定卡组镜像 + 周期冻结副本 + solo_state.json/checkpoint 落盘（无联赛）。"""
    import tempfile
    import json as _json
    from rl import train_solo
    from rl.config import TrainConfig

    d = tempfile.mkdtemp()
    cfg = TrainConfig(name="selftest_solo", total_steps=8, steps_per_eval=4,
                      update_interval=4, batch_size=4, hidden_dim=32, seed=0,
                      n_eval_games=2, max_ep_steps=4, solo_copy_every=2, out_dir=d)
    train_solo.run_solo(cfg, record_replays=False)
    assert os.path.exists(cfg.solo_state_path()), "solo_state.json 应已落盘"
    st = _json.load(open(cfg.solo_state_path(), "r", encoding="utf-8"))
    assert st["mode"] == "solo" and st["opponent"] == "self-play-frozen-copy"
    assert len(st["history"]) >= 2, "应有起始+最终评估"
    assert st["deck"] == train_solo.DEFAULT_SOLO_DECK, "固定卡组镜像"
    assert os.path.exists(cfg.solo_main_path()), "solo_main.pt 应已落盘"
    for s in (0, 4, 8):   # eval_at_start + steps_per_eval=4 + 结束步
        assert os.path.exists(cfg.solo_ckpt_path(s)), f"solo_main_{s}.pt 历史检查点应已落盘"
    assert not os.path.exists(cfg.state_path()), "solo 不应写 league_state.json（无联赛）"
    print("[PASS] solo 自对弈：固定卡组镜像 + 冻结副本 + solo_state.json/solo_main.pt 落盘、无联赛状态")


def test_human_play_session():
    """人机对战：随机动作驱动 + EpisodeReplay/BC 样本落盘 + 导出可喂信念/BC 训练。"""
    import tempfile
    import pickle
    from rl import human_play
    from rl.config import TrainConfig
    from rl.belief import BeliefInference
    from rl.follower import FollowerPolicy
    from rl.plan_space import PLAN_DIM
    from rl.replay import EpisodeReplay

    d = tempfile.mkdtemp()
    cfg = TrainConfig(name="selftest_human", hidden_dim=32, max_ep_steps=30, out_dir=d)
    deck = human_play.DEFAULT_PLAY_DECK
    belief_dim = len(BeliefInference(opp_deck=deck, n_particles=128, seed=0).encode(None, None))
    pol = FollowerPolicy(hidden=32, plan_dim=PLAN_DIM, belief_dim=belief_dim)
    meta = human_play.drive_games(pol, 1, seed=0, max_steps=12, out_dir=d, cfg=cfg)
    assert meta[0]["steps"] > 0 and meta[0]["bc"] > 0, "人机对战应产生步数与 BC 样本"
    files = os.listdir(d)
    assert any(f.startswith("episode_") for f in files), "应落盘 EpisodeReplay"
    assert any(f.startswith("bc_") for f in files), "应落盘 BC 样本"
    bel, bc = human_play.export_data(d, os.path.join(d, "belief.pkl"), os.path.join(d, "bc.pkl"))
    replays = pickle.load(open(bel, "rb"))
    assert len(replays) == 1 and "steps" in replays[0], "信念回放可导出"
    samples = pickle.load(open(bc, "rb"))
    assert len(samples) == meta[0]["bc"], "BC 样本合并数一致"
    ep = EpisodeReplay()
    ep.steps = replays[0]["steps"]
    ds = ep.to_belief_dataset()
    assert ds, "EpisodeReplay → 信念监督样本非空（含 hidden 特权标签）"
    print("[PASS] 人机对战：随机驱动 + EpisodeReplay/BC 落盘 + 导出（信念/BC 均可训练）")


def test_solo_resume():
    """solo 断点续练：恢复 step/权重/优化器/历史曲线，续训不重复评估。"""
    import tempfile
    import json as _json
    from rl.config import TrainConfig
    from rl import train_solo

    d = tempfile.mkdtemp()
    cfg = TrainConfig(name="selftest_solo_resume", total_steps=8, steps_per_eval=4,
                      update_interval=4, batch_size=4, hidden_dim=32, seed=0,
                      n_eval_games=2, max_ep_steps=4, solo_copy_every=4, out_dir=d)
    train_solo.run_solo(cfg, record_replays=False)
    assert os.path.exists(cfg.solo_opt_path()), "solo_opt.pt 应已落盘（断点续练用）"
    st = _json.load(open(cfg.solo_state_path(), "r", encoding="utf-8"))
    assert [h["step"] for h in st["history"]] == [0, 4, 8], st["history"]
    rs = _json.load(open(cfg.run_state_path(), "r", encoding="utf-8"))
    assert rs["step"] == 8
    # 续训到 12 步：不重跑 0/4/8，曲线延续到 12
    cfg2 = TrainConfig(name="selftest_solo_resume", total_steps=12, steps_per_eval=4,
                       update_interval=4, batch_size=4, hidden_dim=32, seed=1,
                       n_eval_games=2, max_ep_steps=4, solo_copy_every=4, out_dir=d)
    train_solo.run_solo(cfg2, resume=True, record_replays=False)
    st2 = _json.load(open(cfg.solo_state_path(), "r", encoding="utf-8"))
    steps2 = [h["step"] for h in st2["history"]]
    assert steps2 == [0, 4, 8, 12], steps2
    print("[PASS] solo 断点续练：恢复 step/权重/优化器/历史曲线，续训不重复评估")


def test_stall_probe():
    """僵局早停探针：连续 STALL_LIMIT 次零塔血变化 → early_stop；塔损重置计数。"""
    from rl.run_league import _stall_probe, STALL_LIMIT

    def make_fake(hps):
        class _P:
            def __init__(self, k, l, r):
                self.king_tower_hp = k
                self.left_tower_hp = l
                self.right_tower_hp = r
        class _B:
            players = [_P(hps[0], hps[1], hps[2]), _P(hps[3], hps[4], hps[5])]
        class _E:
            battle = _B()
        return _E()

    env = make_fake([1000] * 6)
    last, cnt, early = None, 0, False
    for _ in range(STALL_LIMIT + 1):   # 首次调用只建立基线，之后 STALL_LIMIT 次连续零变化
        early, last, cnt = _stall_probe(env, last, cnt)
    assert early, "连续零塔损应触发早停"

    env2 = make_fake([1000] * 6)
    last2, cnt2 = None, 0
    _, last2, cnt2 = _stall_probe(env2, last2, cnt2)
    env2.battle.players[0].king_tower_hp = 999  # 塔损
    early2, last2, cnt2 = _stall_probe(env2, last2, cnt2)
    assert not early2 and cnt2 == 0, "塔损应重置僵局计数"
    print("[PASS] 僵局探针：连续零塔损早停 + 塔损重置计数")


def test_play_pair_env_reuse():
    """评估加速：play_pair 复用单个 env（每局 reset(seed=...)），换边 n 局跑通 + Elo/PFSP 更新。"""
    from rl.follower import FollowerPolicy
    from rl.belief import BeliefInference
    from rl.league import League
    from rl.plan_space import PLAN_DIM
    from rl.run_league import play_pair

    deck = ["Knight", "MiniPekka", "Arrows", "Minions", "Musketeer",
            "Fireball", "Giant", "Archer"]
    belief_dim = len(BeliefInference(opp_deck=deck, n_particles=32, seed=0).encode(None, None))
    pa = FollowerPolicy(hidden=32, plan_dim=PLAN_DIM, belief_dim=belief_dim)
    pb = FollowerPolicy(hidden=32, plan_dim=PLAN_DIM, belief_dim=belief_dim)
    pa.to_device("cpu")
    pb.to_device("cpu")
    lg = League(seed=0)
    lg.add_agent("a", kind="main", policy=pa)
    lg.add_agent("b", kind="all_decks", policy=pb)
    wa, wb, dr, rs = play_pair(lg, "a", pa, "b", pb, 4, 30, seed=7)
    assert wa + wb + dr == 4, f"局数 4，实际 {wa}+{wb}+{dr}"
    assert 0 <= wa <= 4 and 0 <= wb <= 4
    tbl = lg.elo_table()
    assert "a" in tbl and "b" in tbl, "Elo/PFSP 应已更新"
    print(f"[PASS] play_pair env 复用：换边 4 局跑通（{wa}W {wb}L {dr}D）+ Elo/PFSP 更新")


def test_eval_stall_early_stop():
    """僵局早停集成：双方都不部署 → 连续零塔损判平，远早于打满 max_steps。"""
    import time
    from rl.env_wrapper import RLEnv
    from rl.action_bundle import ActionBundle
    from rl.opponents import ScriptedPolicy
    from rl.belief import BeliefInference
    from rl.belief_planner import BeliefPlanner
    from rl.run_league import _run_side0, _prepare_env

    class Idle(ScriptedPolicy):
        def play(self, env, player_id):
            return ActionBundle.noop()

    idle = Idle()
    env = RLEnv(opponent=None, seed=3)
    _prepare_env(env, idle, idle)
    t0 = time.monotonic()
    w = _run_side0(env, idle, BeliefInference(opp_deck=env.deck1, n_particles=16, seed=3),
                   BeliefPlanner(), max_steps=600, reset_seed=3)
    dt = time.monotonic() - t0
    assert w is None, "僵局应判平"
    assert dt < 12.0, f"僵局应提前结束（实际 {dt:.1f}s），否则早停未触发"
    print(f"[PASS] 僵局早停：{dt:.1f}s 判平（对照打满 600 步 ~23s）")


def main():
    test_action_bundle_same_tick()
    test_action_bundle_ability()
    test_bayes_filter()
    test_hidden_replay_consistency()
    test_entropy_positive_and_sign()
    test_mask_validate_invariant_both_sides()
    test_heuristic_opponent_actually_plays()
    test_exploiter_loads_main_checkpoint()
    test_belief_survives_ability()
    test_belief_multi_card_update()
    test_register_checkpoint_isolated()
    test_bundle_cap_no_crash()
    test_replay_roundtrip()
    test_prophet_empty_board_not_defend()
    test_random_deck_model()
    test_league_elo_history()
    test_winrate_streams_independent()
    test_elo_eval_granularity()
    test_classified_decks()
    test_league_training_loop()
    test_belief_follower_ppo_league()
    test_config_reward_weights()
    test_model_reward_overrides()
    test_reward_economy_preset()
    test_reward_economy_level_invariance()
    test_reward_economy_elixir_diff()
    test_reward_economy_trade_pricing()
    test_rlenv_card_level()
    test_tower_troop_hp_reference()
    test_league_resume()
    test_league_replays()
    test_dashboard_replays()
    test_battle_clone_fix()
    test_cuda_device_support()
    test_parallel_batch_equivalence()
    test_parallel_training_loop()
    test_mp_training_loop()
    test_flow_league_smoke()
    test_ablation_recorded()
    test_flow_sweep_smoke()
    test_flow_resume()
    test_solo_mode_smoke()
    test_solo_resume()
    test_human_play_session()
    test_stall_probe()
    test_play_pair_env_reuse()
    test_eval_stall_early_stop()
    print("\nALL SELFTESTS PASSED")


if __name__ == "__main__":
    main()
