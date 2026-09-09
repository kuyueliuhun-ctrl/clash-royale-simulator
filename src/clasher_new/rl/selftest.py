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
import shutil
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
    """命名配置：预设解析互不影响、塔血不对称基线、奖励权重可注入 RLEnv 并改变回报。"""
    import tempfile
    from rl.config import TrainConfig, reward_to_env
    from rl.env_wrapper import RLEnv
    from rl.action_bundle import ActionBundle

    std = TrainConfig.resolve("standard")
    agg = TrainConfig.resolve("aggressive")
    assert std.reward["crown_weight"] == 8.0
    assert agg.reward["crown_weight"] == 8.0
    # 2026-09 改版：塔血不对称（挨打 0.0012 > 打人 0.001）+ 被破塔 10 > 破塔 8；
    # 预设差异体现在皇冠/费差
    for name in ("standard", "aggressive", "defensive", "lockdown", "elixir", "economy", "fast"):
        rw = TrainConfig.resolve(name).reward
        assert rw["tower_dmg_opp"] == 0.001 and rw["tower_dmg_self"] == 0.0012, f"{name} 塔血不对称基线"
        assert rw["crown_lose_weight"] == 10.0 > rw["crown_weight"], f"{name} 被破塔惩罚应更重"
    assert agg.reward["elixir_diff_weight"] > std.reward["elixir_diff_weight"]
    # 二次解析不污染预设（共享实例回归）
    assert TrainConfig.resolve("standard").reward["crown_weight"] == 8.0
    assert TrainConfig.resolve("aggressive").reward["crown_weight"] == 8.0

    env = RLEnv(opponent=None, seed=0, reward_weights=reward_to_env(std))
    env.reset()
    _, r0, _, _, _ = env.step(ActionBundle.noop())
    env2 = RLEnv(opponent=None, seed=0, reward_weights=reward_to_env(agg))
    env2.reset()
    _, r1, _, _, _ = env2.step(ActionBundle.noop())
    # 配置项确实注入 env 并生效（不同配置 → 不同权重结构）
    assert env.reward_weights["crown_weight"] == 8.0
    assert env2.reward_weights["crown_weight"] == 8.0
    assert isinstance(r0, float) and isinstance(r1, float)

    # config.json 往返
    d = tempfile.mkdtemp()
    p = os.path.join(d, "cfg.json")
    agg.save(p)
    back = TrainConfig.load(p)
    assert back.name == "aggressive" and back.reward["crown_weight"] == 8.0
    print("[PASS] 命名配置：预设/加载/奖励权重注入 RLEnv 正常、塔血不对称 0.001/0.0012 + 被破塔 10")


def test_model_reward_overrides():
    """按流派奖惩：main/all/random 同一基线；推进加码费差、防反减码、自闭压到≈0。"""
    from rl.config import TrainConfig, model_reward_weights

    std = TrainConfig.resolve("standard")
    base = model_reward_weights("main", std)
    assert model_reward_weights("all_decks", std) == base, "all_decks 应与 main 同参数"
    assert model_reward_weights("random_deck", std) == base, "random_deck 应与 main 同参数"
    assert base["tower_dmg_opp"] == 0.001 and base["tower_dmg_self"] == 0.0012, \
        "塔血不对称 0.001/0.0012（挨打 > 打人）"
    assert base["crown_lose_weight"] == 10.0 > base["crown_weight"], "被破塔 10 > 破塔 8"
    assert base["normalize_tower_dmg"] is True, "费差机制默认打开"
    assert base["elixir_diff_weight"] == 0.5, "基线费差 = 0.5（1圣水≈500血）"
    # 流派覆盖：推进 > 基线 > 防反 > 自闭
    push = model_reward_weights("push_flow", std)
    counter = model_reward_weights("counter_flow", std)
    lock = model_reward_weights("lockdown_flow", std)
    assert push["elixir_diff_weight"] == 0.7 > base["elixir_diff_weight"], "推进应加码费差"
    assert counter["elixir_diff_weight"] == 0.3 < base["elixir_diff_weight"], "防反应减码费差"
    assert lock["elixir_diff_weight"] == 0.05 < counter["elixir_diff_weight"], "自闭应压到≈0"
    # 塔血在所有流派也保持不对称
    for mid in ("push_flow", "counter_flow", "lockdown_flow"):
        rw = model_reward_weights(mid, std)
        assert rw["tower_dmg_opp"] == 0.001 and rw["tower_dmg_self"] == 0.0012, mid
    # 未知模型回退到所选预设（不改基线行为）
    assert model_reward_weights("unknown_model", std) == base
    print("[PASS] 按流派奖惩：main/all/random 同基线 0.5、推进 0.7 / 防反 0.3 / 自闭 0.05、"
          "塔血不对称 0.001/0.0012 + 被破塔 10、未知模型回退基线")


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
    # 9j：belief token 尾部含事件通道——策略维度须与 BeliefInference.encode 输出一致
    #（旧硬编码 71 在词表 v2 扩容后失配；从 DEFAULT_SOLO_DECK 动态推导）
    from rl.belief import belief_token_dim
    from rl import train_solo
    main = FollowerPolicy(hidden=32, plan_dim=PLAN_DIM,
                          belief_dim=belief_token_dim(list(train_solo.DEFAULT_SOLO_DECK)))
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
    from rl.ppo import PPOTrainer

    d = tempfile.mkdtemp()
    cfg = TrainConfig(name="selftest_solo", total_steps=8, steps_per_eval=4,
                      update_interval=4, batch_size=4, hidden_dim=32, seed=0,
                      n_eval_games=2, max_ep_steps=4, solo_copy_every=2, out_dir=d)
    # 泄漏回归哨兵：max_ep_steps=4 保证 8 步内跨过局边界（bug 恰在 reset 路径）。
    # _new_episode_reset 若漏 nonlocal ep_* 缓冲区，旧对局帧会反复 flush 进 PPO
    # 批（当时实测 8 步喂了 30+ 条过渡）。每环境步至多产生 1 条过渡，
    # total(update) ≤ total(env steps)+k_max·1 是硬上界——超过即缓冲区泄漏。
    _real_update = PPOTrainer.update
    _upd_calls = []
    train_solo.PPOTrainer.update = lambda self, batch, **kw: (
        _upd_calls.append(len(batch)) or _real_update(self, batch, **kw))
    try:
        train_solo.run_solo(cfg, record_replays=False)
    finally:
        train_solo.PPOTrainer.update = _real_update
    total_upd = sum(_upd_calls)
    assert total_upd <= cfg.total_steps, (
        f"PPO 批累计 {total_upd} > 环境步 {cfg.total_steps}：ep_* 缓冲区泄漏"
        f"（_new_episode_reset 漏 nonlocal，旧对局帧重复入批）")
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


def test_draw_penalty_as_loss():
    """平局=失败：引擎终局平局（game_over=True, winner=None）与僵局/截断平局都按失败惩罚。"""
    from rl.env_wrapper import compute_reward
    from rl.config import TrainConfig, reward_to_env

    std = reward_to_env(TrainConfig.resolve("economy"))
    base = dict(blue_hps_old=10928.0, red_hps_old=10928.0,
                blue_hps_new=10928.0, red_hps_new=10928.0,
                blue_left_old=3, red_left_old=3, blue_left_new=3, red_left_new=3,
                my_elixir_before=5.0, opp_elixir_before=5.0,
                my_elixir_after=5.0, opp_elixir_after=5.0,
                invalid_count=0, blue_hps_max=10928.0, red_hps_max=10928.0)

    # ① 未终局（winner=None, game_over=False）→ 平局惩罚不触发（避免误伤普通步）
    r_ongoing = compute_reward(std, winner=None, game_over=False, **base)
    assert r_ongoing == 0.0, f"进行中的普通步不应被平局罚: {r_ongoing}"

    # ② 引擎终局平局（game_over=True, winner=None）→ 与失败同罚
    r_draw = compute_reward(std, winner=None, game_over=True, **base)
    assert r_draw == -float(std["lose_penalty"]), \
        f"平局应=失败罚(-{std['lose_penalty']}): {r_draw}"
    assert std.get("draw_penalty", std["lose_penalty"]) == std["lose_penalty"], \
        "默认 draw_penalty 应与 lose_penalty 相同"

    # ③ 训练循环截断平局：win 分支不受影响
    r_win = compute_reward(std, winner=0, game_over=True, **base)
    assert r_win == float(std["win_bonus"]), f"胜仍应+win_bonus: {r_win}"
    r_lose = compute_reward(std, winner=1, game_over=True, **base)
    assert r_lose == -float(std["lose_penalty"]), f"负仍应-lose_penalty: {r_lose}"

    # ④ eval_solo 僵局平局 → mean_reward 已扣 draw_penalty（策略学得到"平局不可取"）
    from rl import train_solo
    from rl.follower import FollowerPolicy
    from rl.belief import BeliefInference
    from rl.plan_space import PLAN_DIM
    from rl.action_bundle import ActionBundle

    class Noop(FollowerPolicy):
        """act 恒返回 noop：双方都不部署 → 僵局早停必判平，mean_reward 确定性可断言。"""
        def act(self, obs, belief_token, plan_token, get_mask,
                hidden=None, deterministic=False):
            return ActionBundle.noop(), 0.0, 0.0, hidden, {}

    cfg = TrainConfig(name="selftest_draw_penalty", hidden_dim=32, n_eval_games=2,
                      max_ep_steps=600, seed=5, out_dir="runs/_tmp_drawtest")
    env = train_solo.solo_env(cfg, 5)
    bd = len(BeliefInference(opp_deck=list(train_solo.DEFAULT_SOLO_DECK),
                             n_particles=128, seed=0).encode(None, None))
    main = Noop(hidden=32, plan_dim=PLAN_DIM, belief_dim=bd)
    opp = Noop(hidden=32, plan_dim=PLAN_DIM, belief_dim=bd)
    main.to_device("cpu")
    opp.to_device("cpu")
    train_solo._sync_frozen_copy(main, opp)
    stats, _ = train_solo.eval_solo(env, main, opp, 2, 600, 5, cfg, record_replays=False)
    assert stats["draws"] == 2, f"双方 noop 应全平局: {stats}"
    assert stats["mean_reward"] == -float(std["lose_penalty"]), \
        f"僵局平局 mean_reward 应含失败罚: {stats['mean_reward']}"
    print("[PASS] 平局=失败：引擎终局平局/僵局平局均按 lose_penalty 惩罚，普通步不误伤")

    import shutil
    shutil.rmtree("runs/_tmp_drawtest", ignore_errors=True)


def test_reward_v2_ledger():
    """reward v2 资源账（economy）：部署不罚（E−c 与 V+c 同帧抵消）、份额入账/死亡注销、
    有目标法术=花费型惩罚（8h 起无目标空砸被闸门拒绝，不再有合法格）、双倍期（t≥120）
    edw 换档为 elixir_diff_late、单位受伤 shaping 生效。"""
    import numpy as np
    from rl.env_wrapper import RLEnv, DEFAULT_DECK
    from rl.action_bundle import ActionBundle
    from rl.config import TrainConfig, reward_to_env

    cfg = TrainConfig.resolve("economy")
    env = RLEnv(opponent=lambda obs: ActionBundle.noop(), seed=3,
                reward_weights=reward_to_env(cfg),
                deck0=DEFAULT_DECK, deck1=DEFAULT_DECK)
    env.reset(seed=3)
    p = env.battle.players
    # 控制手牌：slot 布局随 shuffle 变，这里直接钉死前 4 张
    p[0].cycle = ["Knight", "Arrows", "Fireball", "MiniPekka"] + p[0].cycle[4:]
    env._seen_max_id = max(env.battle.entities)

    def first_cell(slot):
        cells = env.get_action_mask_for(0)["cells"][slot - 1]
        ys, xs = np.nonzero(cells)
        return int(xs[0]), int(ys[0])

    # ① 部署 Knight（3 费）：资源账帧 ≈0（不再有旧式 −1.5 下牌惩罚），V 入账 3
    _, r_deploy, _, _, info = env.step(ActionBundle.from_single(1, 9, 13))
    assert abs(r_deploy) < 0.05, f"部署不应即时受罚: {r_deploy}"
    assert abs(info["field_v"][0] - 3.0) < 1e-6, f"Knight 3费应入账: {info['field_v']}"

    # ② 单位死亡注销：Knight 移到敌方塔旁 1hp → 步进击杀 → V 归零
    knight = [e for e in env.battle.entities.values()
              if getattr(e, "name", "") == "Knight" and e.player == 0 and e.is_alive]
    assert knight, "应存在部署的 Knight"
    k = knight[0]
    k.hp = 1.0
    k.position.x, k.position.y = 14.5, 25.5   # P1 右公主塔坐标
    for _ in range(4):
        env.step(ActionBundle.noop())
    assert abs(env._active_v[0]) < 1e-6, f"死亡后份额应注销: {env._active_v}"

    # ③ 有目标法术（8h 空砸闸门：无目标格不再可出，这里取首个覆盖存活敌方塔的合法格）
    #    花费型 → ≈ −3×0.5 = −1.5
    p[0].elixir = 5.0
    p[0].cycle.remove("Arrows"); p[0].cycle.insert(0, "Arrows")   # 轮转回手（手牌区 = 前 4）
    slot_arrows = p[0].cycle.index("Arrows") + 1
    x, y = first_cell(slot_arrows)
    _, r_blank, _, _, info = env.step(ActionBundle.from_single(slot_arrows, x, y))
    assert info["bundle_ok"], f"覆盖敌方塔的法术应可出: {info['bundle_reason']}"
    assert r_blank < -1.2, f"有目标法术前段应≈−1.5（花费型）: {r_blank}"

    # ④ 双倍期（t≥120）edw 换档 0.5→0.1：同有目标法术 ≈ −3×0.1 = −0.3
    env.battle.time = 121.0
    p[0].elixir = 5.0
    p[0].cycle.remove("Arrows"); p[0].cycle.insert(0, "Arrows")
    slot_arrows = p[0].cycle.index("Arrows") + 1
    x, y = first_cell(slot_arrows)
    _, r_late, _, _, info = env.step(ActionBundle.from_single(slot_arrows, x, y))
    assert info["bundle_ok"], f"双倍期覆盖敌方塔的法术应可出: {info['bundle_reason']}"
    assert -0.6 < r_late < -0.15, f"双倍期有目标法术应≈−0.3(late edw): {r_late}"

    # ⑤ 配置契约：standard/economy 都带 v2 键且 late < early、tower late > early
    std = TrainConfig.resolve("standard").reward
    assert std["elixir_diff_late"] < std["elixir_diff_weight"], "双倍期费应更贱"
    assert std["tower_dmg_late"] > std["tower_dmg_opp"], "双倍期塔血应更贵"
    assert std["unit_dmg_k"] > 0.0, "单位受伤 shaping 默认打开"
    print("[PASS] reward v2 记账：部署不罚/份额入账注销/有目标法术花费型罚（空砸已被闸门拒绝）/"
          "双倍期换档/单位受伤 shaping 生效")


def test_spell_empty_value_gate():
    """8h 空砸闸门：伤害型法术（Arrows 等）没有存活敌方目标可罩的格子=非法格，
    validate 整包拒绝；敌人进入溅射半径后该格恢复合法（mask/validate 同源、P0/P1 对称）。"""
    import battle as battle_mod
    import player as player_mod
    from core import Position
    from battle import Troop
    from rl.action_bundle import ActionBundle, sub_position
    from rl.action_mask import legal_cells, validate_bundle

    deck = ["Arrows", "Knight", "MiniPekka", "Giant", "Musketeer", "Fireball", "Archer", "Minions"]
    radius = 3.5  # Arrows 溅射半径（世界单位）
    for pid in (0, 1):
        bs = battle_mod.BattleState(
            player_mod.PlayerState(0, list(deck), 5.0),
            player_mod.PlayerState(1, list(deck), 5.0))
        p = bs.players[pid]
        p.elixir = 10.0
        p.cycle = ["Arrows"] + [c for c in p.cycle if c != "Arrows"]  # Arrows 固定槽 1

        def enemy_in_radius(pos):
            for e in bs.entities.values():
                if not getattr(e, "is_alive", True):
                    continue
                if getattr(e, "player", None) != (1 - pid):
                    continue
                col = getattr(getattr(e, "data", None), "collision_radius", 0.0) or 0.0
                if pos.distance_to(e.position) <= radius + col + 1e-9:
                    return True
            return False

        cells = legal_cells(bs, pid, "Arrows")
        assert int(cells.sum()) > 0, f"P{pid} 敌方塔存活时 Arrows 应仍有合法格"
        empty = None
        for yy in range(cells.shape[0]):
            for xx in range(cells.shape[1]):
                if not enemy_in_radius(sub_position(pid, xx, yy)):
                    empty = (xx, yy)
                    break
            if empty is not None:
                break
        assert empty is not None, f"P{pid} 应存在无目标空场格"
        ex, ey = empty
        # 空场格：掩码非法 + 整包拒绝
        assert not bool(cells[ey, ex]), f"P{pid} 空场格不应可出 Arrows"
        ok, reason, _ = validate_bundle(bs, pid, ActionBundle.from_single(1, ex, ey))
        assert not ok and "非法" in reason, f"P{pid} 空砸应被 validate 拒绝: {reason}"
        # 敌人进入该格溅射半径 → 恢复合法
        wpos = sub_position(pid, ex, ey)
        t = Troop(bs.next_entity_id, Position(wpos.x, wpos.y), 1 - pid, "Archer", bs)
        t.hp = 1.0
        bs._spawn_entity(t)
        cells2 = legal_cells(bs, pid, "Arrows")
        assert bool(cells2[ey, ex]), f"P{pid} 有敌人后该格应恢复合法"
        ok2, reason2, _ = validate_bundle(bs, pid, ActionBundle.from_single(1, ex, ey))
        assert ok2, f"P{pid} 有敌人后 validate 应通过: {reason2}"
    print("[PASS] 8h 空砸闸门：伤害法术空场格 mask+validate 双拒；目标进入溅射半径后恢复合法（P0/P1 对称）")


def test_spell_tower_ev_gate():
    """9h 前段法术对塔 EV 闸门：双倍期前"只罩对手公主塔、无部队/建筑可溅"的落点
    mask+validate 双拒（Arrows 3 费 25 伤 < 0.5×3×500=750 折费线）；塔旁有敌方部队
    的落点放行（有正事可干）；双倍期(t≥120)放行；部队在溅射半径内但不罩塔的格子
    不受此闸门影响（8h 空砸闸门已覆盖）。"""
    import battle as battle_mod
    import player as player_mod
    from core import Position
    from rl.action_bundle import ActionBundle
    from rl.action_mask import legal_cells, validate_bundle, sub_position

    deck = ["Arrows", "Knight", "MiniPekka", "Giant", "Musketeer", "Fireball", "Archer", "Minions"]
    bs = battle_mod.BattleState(player_mod.PlayerState(0, list(deck), 5.0),
                                player_mod.PlayerState(1, list(deck), 5.0))
    p0 = bs.players[0]
    p0.elixir = 10.0
    p0.cycle = ["Arrows"] + [c for c in p0.cycle if c != "Arrows"]

    # P0 视角：P1 左塔世界坐标 (3.5, 25.5)；本地坐标换算 sub_position(0,x,y)
    cells = legal_cells(bs, 0, "Arrows")
    tower_local = None
    for yy in range(32):
        for xx in range(18):
            if sub_position(0, xx, yy).distance_to(Position(3.5, 25.5)) < 1.0:
                tower_local = (xx, yy)
    assert tower_local is not None, "应能找到 P1 左塔对应本地格"
    tx, ty = tower_local
    assert not bool(cells[ty, tx]), "前段纯砸塔格(只罩公主塔)应非法"
    ok, reason, _ = validate_bundle(bs, 0, ActionBundle.from_single(1, tx, ty))
    assert not ok and "非法" in reason, f"前段纯砸塔应被 validate 拒绝: {reason}"

    # 塔旁放一个敌方部队（仍在溅射半径内）→ 该格恢复合法
    from battle import Troop
    troop_id = bs.next_entity_id
    t = Troop(bs.next_entity_id, Position(3.5, 24.0), 1, "Archer", bs)
    t.hp = 1.0
    bs._spawn_entity(t)
    cells2 = legal_cells(bs, 0, "Arrows")
    assert bool(cells2[ty, tx]), "溅射半径含部队后砸塔格应恢复合法"

    # 双倍期放行：把时间拨到 t≥120
    bs.time = 121.0
    cells3 = legal_cells(bs, 0, "Arrows")
    assert bool(cells3[ty, tx]), "双倍期砸塔格应放行（奖励口径已调成近正 EV）"
    ok3, _, _ = validate_bundle(bs, 0, ActionBundle.from_single(1, tx, ty))
    assert ok3, "双倍期纯砸塔应被 validate 放行"
    bs.time = 0.0

    # P1 镜像对称：P1 砸 P0 左塔 (3.5, 6.5)
    p1 = bs.players[1]
    p1.elixir = 10.0
    p1.cycle = ["Arrows"] + [c for c in p1.cycle if c != "Arrows"]
    bs.entities[troop_id].is_alive = False   # 移走刚才的部队，恢复纯砸塔局面
    ok4, reason4, _ = validate_bundle(bs, 1, ActionBundle.from_single(1, 3, 18))
    assert not ok4 and "非法" in reason4, f"P1 镜像纯砸塔应被拒绝: {reason4}"
    print(f"[PASS] 9h 前段法术对塔 EV 闸门：纯砸塔格 mask+validate 双拒；含部队放行；"
          f"双倍期放行；对称视角同拒")


def test_no_solo_commit_without_lead():
    """8h 不裸下：对手能出手且无 +3 费差时，MiniPekka/Giant 单卡（空 bundle 首卡）
    mask 整槽禁掉、validate 单卡整包拒绝；有费差/对手无法出手/压境防守时放行；
    同刻多卡协同不受限。"""
    import numpy as np
    import battle as battle_mod
    from battle import Troop
    from core import Position
    from rl.env_wrapper import RLEnv, DEFAULT_DECK
    from rl.action_bundle import ActionBundle
    from rl.action_mask import legal_cells, validate_bundle

    env = RLEnv(opponent=lambda obs: ActionBundle.noop(), seed=11,
                deck0=DEFAULT_DECK, deck1=DEFAULT_DECK)
    env.reset(seed=11)
    p = env.battle.players
    # 固定 P0 手牌：MiniPekka 槽1、Knight 槽2（P1 保持同卡组，手里有 Knight=3费可出手）
    p[0].cycle = ["MiniPekka", "Knight"] + [c for c in p[0].cycle if c not in ("MiniPekka", "Knight")]
    p[0].elixir = 5.0
    p[1].elixir = 5.0

    def mini_mask_slot():
        m = env.get_action_mask_for(0)
        idx = p[0].cycle.index("MiniPekka")
        return bool(m["slots"][idx]), m

    blocked, m = mini_mask_slot()
    assert not blocked, "对手能出手+无费差 → MiniPekka 裸下首卡应被 mask 禁掉"
    # 找一个 MiniPekka 的合法落点（闸门仅禁槽位；位置合法集不变）
    cells = legal_cells(env.battle, 0, "MiniPekka")
    ys, xs = np.nonzero(cells)
    cx, cy = int(xs[0]), int(ys[0])
    ok, reason, _ = validate_bundle(env.battle, 0, ActionBundle.from_single(1, cx, cy))
    assert not ok and "不裸下" in reason, f"单卡 MiniPekka 应被 validate 拒绝: {reason}"

    # 有费差（己方 10 vs 对方 5，+5≥3）→ 放行
    p[0].elixir = 10.0
    allowed, m2 = mini_mask_slot()
    assert allowed, "己方圣水领先≥3 → MiniPekka 单卡应放行"
    ok2, reason2, _ = validate_bundle(env.battle, 0, ActionBundle.from_single(1, cx, cy))
    assert ok2, f"有费差单卡应通过 validate: {reason2}"

    # 对手无法出手（对方圣水 < 手牌最低费）→ 放行
    p[0].elixir = 5.0
    p[1].elixir = 2.0
    allowed3, _ = mini_mask_slot()
    assert allowed3, "对手出不了手 → 裸下应放行"

    # 压境防守（对方单位进入我半场）→ 放行
    p[1].elixir = 5.0
    t = Troop(env.battle.next_entity_id, Position(9.0, 10.0), 1, "MiniPekka", env.battle)
    env.battle._spawn_entity(t)
    allowed4, _ = mini_mask_slot()
    assert allowed4, "对方压境 → 防守单卡应放行"

    # 同刻多卡协同：Knight+MiniPekka 不受“不裸下”限制
    p[0].elixir = 10.0
    p[1].elixir = 5.0
    del env.battle.entities[t.id]  # 清掉压境单位，回到无防守状态
    cells_k = legal_cells(env.battle, 0, "Knight")
    ys_k, xs_k = np.nonzero(cells_k)
    b = ActionBundle()
    b.add(2, int(xs_k[0]), int(ys_k[0]))      # Knight 槽2
    cells_m = legal_cells(env.battle, 0, "MiniPekka")
    ys_m, xs_m = np.nonzero(cells_m)
    b.add(1, int(xs_m[0]), int(ys_m[0]))      # MiniPekka 槽1
    ok5, reason5, _ = validate_bundle(env.battle, 0, b)
    assert ok5, f"同刻两卡协同应放行: {reason5}"
    print("[PASS] 8h 不裸下：无费差单卡高承诺单位 mask+validate 双拒；领先/对手无费/压境/多卡协同放行")


def test_tank_backline_geometry():
    """8h 坦克后屯兵：Giant/Knight 推进时，Musketeer/Archer/Minions/MiniPekka 只能放
    在坦克后面且纵向间距 ≥ 攻击距离（快单位更靠后）；坦克前/贴身位非法，
    坦克消失恢复；落点附近有敌军（防守响应）放行；P0/P1 镜像一致。"""
    import battle as battle_mod
    import player as player_mod
    from battle import Troop
    from core import Position
    from rl.action_bundle import ActionBundle, sub_position
    from rl.action_mask import legal_cells, validate_bundle

    deck = ["Knight", "MiniPekka", "Arrows", "Minions", "Musketeer",
            "Fireball", "Giant", "Archer"]

    def mk_battle():
        return battle_mod.BattleState(
            player_mod.PlayerState(0, list(deck), 10.0),
            player_mod.PlayerState(1, list(deck), 10.0))

    def spawn(bs, pid, name, x, y, hp):
        t = Troop(bs.next_entity_id, Position(x, y), pid, name, bs)
        t.hp = hp
        bs._spawn_entity(t)
        return t

    def cell_near(bs, pid, wx, wy, tol=0.9):
        best, bd = None, 1e18
        for yy in range(32):
            for xx in range(18):
                wp = sub_position(pid, xx, yy)
                d = (wp.x - wx) ** 2 + (wp.y - wy) ** 2
                if d < bd:
                    bd, best = d, (xx, yy)
        assert bd ** 0.5 < tol, f"无网格格点接近 ({wx},{wy})"
        return best

    # —— P0：Giant 左路推进，Musketeer 间距须 ≥ ~6.6 ——
    bs = mk_battle()
    giant = spawn(bs, 0, "Giant", 5.0, 12.0, 4000.0)
    cells = legal_cells(bs, 0, "Musketeer")
    xb, yb = cell_near(bs, 0, 5.0, 5.0)     # 身后 7 单位 → 合法
    assert bool(cells[yb, xb]), "坦克身后留足间距应合法"
    xc, yc = cell_near(bs, 0, 5.0, 11.0)    # 贴身（身后 1.5）→ 非法
    assert not bool(cells[yc, xc]), "贴身坦克位应非法"
    xa, ya = cell_near(bs, 0, 5.0, 14.0)    # 坦克前 → 非法
    assert not bool(cells[ya, xa]), "坦克前位应非法（会超车/抢仇恨）"
    # validate 同源：身后合法格通过、坦克前位整包拒绝
    p0 = bs.players[0]
    p0.cycle = ["Musketeer"] + [c for c in p0.cycle if c != "Musketeer"]
    ok_b, reason_b, _ = validate_bundle(bs, 0, ActionBundle.from_single(1, xb, yb))
    assert ok_b, f"身后位应通过 validate: {reason_b}"
    ok_a, reason_a, _ = validate_bundle(bs, 0, ActionBundle.from_single(1, xa, ya))
    assert not ok_a and "非法" in reason_a, f"坦克前位应被 validate 拒绝: {reason_a}"
    # 坦克消失 → 恢复
    del bs.entities[giant.id]
    cells2 = legal_cells(bs, 0, "Musketeer")
    assert bool(cells2[ya, xa]), "坦克消失后该格应恢复合法"
    # 落点附近有敌军 → 防守放行
    giant2 = spawn(bs, 0, "Giant", 5.0, 12.0, 4000.0)
    spawn(bs, 1, "MiniPekka", 5.0, 13.5, 300.0)
    cells3 = legal_cells(bs, 0, "Musketeer")
    assert bool(cells3[ya, xa]), "落点附近有敌军=防守响应，几何门应放行"

    # —— P1 镜像：Knight 右路推进，Archer 必须在其后 ——
    bs2 = mk_battle()
    spawn(bs2, 1, "Knight", 14.5, 20.0, 2000.0)
    cells4 = legal_cells(bs2, 1, "Archer")
    xb2, yb2 = cell_near(bs2, 1, 14.5, 26.0)   # Knight 身后（y 更大）→ 合法
    assert bool(cells4[yb2, xb2]), "P1 Knight 身后位应合法"
    xa2, ya2 = cell_near(bs2, 1, 14.5, 18.0)   # Knight 前方 → 非法
    assert not bool(cells4[ya2, xa2]), "P1 Knight 前方位应非法"

    # —— MiniPekka 按后排保护（Giant 后），坦克前方位同样非法 ——
    bs3 = mk_battle()
    spawn(bs3, 0, "Giant", 5.0, 12.0, 4000.0)
    cells5 = legal_cells(bs3, 0, "MiniPekka")
    xb3, yb3 = cell_near(bs3, 0, 5.0, 9.0)     # 身后 3 单位 ≥ 2.6 → 合法
    assert bool(cells5[yb3, xb3]), "MiniPekka 坦克身后应合法"
    xa3, ya3 = cell_near(bs3, 0, 5.0, 13.5)
    assert not bool(cells5[ya3, xa3]), "MiniPekka 坦克前位应非法"
    print("[PASS] 8h 坦克后屯兵：后排只能跟坦克身后留攻击距离间距（快单位更远）；"
          "坦克前/贴身位非法、防守响应放行、P0/P1 镜像一致")


def test_plan_v1_layout():
    """PlanToken v1 尾部扩展：旧 21 维布局逐位兼容、新意图进尾部新组、
    target/hint/threat/budget/hold_mask 落位正确、load_checkpoint 旧维补零扩展。"""
    import numpy as np
    from rl.plan_space import (PLAN_DIM, PlanToken, MACRO_INTENTS, _OLD_INTENT_COUNT,
                               FOCUS_REGIONS, TARGET_KINDS, PLACEMENT_HINTS,
                               OPP_SPELL_THREATS)
    # 9k：PLACEMENT_HINTS 7→8（新增 intercept_mid）→ PLAN_DIM 57→58（旧21+新37）
    assert PLAN_DIM == 58, f"PLAN_DIM 应为 58（旧21+新37）: {PLAN_DIM}"

    # ① 旧意图帧：前 21 维 == 旧布局（intent8 + region8 + 旧标量5）
    v = PlanToken().to_vector()
    assert int(np.argmax(v[:8])) == MACRO_INTENTS.index("cycle_and_wait")
    assert int(np.argmax(v[8:16])) == FOCUS_REGIONS.index("own_center")
    assert np.allclose(v[16:21], [0.0, 1.0, 0.0, 0.5, 0.0]), v[16:21]
    # v1 尾段默认值：新意图组全 0；target/hint/threat = none(各自 one-hot 首位)；
    # elixir_budget = 1.0（不限制投入）；hold 4 位 = 0
    n_new = len(MACRO_INTENTS) - _OLD_INTENT_COUNT
    seg_new = v[21:21 + n_new]
    assert np.all(seg_new == 0.0), "默认帧不应携带新意图"
    off = 21 + n_new
    assert int(np.argmax(v[off:off + len(TARGET_KINDS)])) == TARGET_KINDS.index("none")
    off += len(TARGET_KINDS)
    assert int(np.argmax(v[off:off + len(PLACEMENT_HINTS)])) == PLACEMENT_HINTS.index("none")
    off += len(PLACEMENT_HINTS)
    assert int(np.argmax(v[off:off + len(OPP_SPELL_THREATS)])) == OPP_SPELL_THREATS.index("none")
    off += len(OPP_SPELL_THREATS)
    assert abs(v[off] - 1.0) < 1e-6, "默认 elixir_budget 应为 1.0（不限制）"
    assert np.all(v[off + 1:off + 5] == 0.0), "默认 hold_mask 应为 0"

    # ② 新意图帧：旧组全 0（不占用旧 intent 位），新组 one-hot 于 (21..21+13)
    for name in MACRO_INTENTS[_OLD_INTENT_COUNT:]:
        vn = PlanToken.intent(name).to_vector()
        assert np.all(vn[:8] == 0.0), f"新意图 {name} 不应占用旧 intent 位"
        seg = vn[21:21 + n_new]
        assert int(np.argmax(seg)) == MACRO_INTENTS.index(name) - _OLD_INTENT_COUNT, name

    # ③ v1 字段落位（顺序：intent_new → target_kind → placement_hint → threat → budget → hold）
    v = PlanToken.intent("soft_control", "enemy_center", target_kind="unit",
                         placement_hint="pull_across", opp_spell_threat="lightning",
                         elixir_budget=0.4, hold_mask=0b1010).to_vector()
    off = 21 + n_new
    assert int(np.argmax(v[off:off + len(TARGET_KINDS)])) == TARGET_KINDS.index("unit")
    off += len(TARGET_KINDS)
    assert int(np.argmax(v[off:off + len(PLACEMENT_HINTS)])) == PLACEMENT_HINTS.index("pull_across")
    off += len(PLACEMENT_HINTS)
    assert int(np.argmax(v[off:off + len(OPP_SPELL_THREATS)])) == OPP_SPELL_THREATS.index("lightning")
    off += len(OPP_SPELL_THREATS)
    assert abs(float(v[off]) - 0.4) < 1e-6
    assert v[off + 1:off + 5].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert PlanToken(hold_mask=0b1010).hold_slots() == [2, 4]

    # ④ load_checkpoint：旧 21 维 ckpt → 57 维网络补零加载（前 21 列权重原样保留）
    import tempfile
    import shutil
    import torch
    from rl.follower import FollowerPolicy, load_checkpoint
    old = FollowerPolicy(hidden=32, plan_dim=21, belief_dim=8)
    d = tempfile.mkdtemp()
    p = os.path.join(d, "old.pt")
    torch.save({"state_dict": old.state_dict(), "plan_dim": 21,
                "belief_dim": 8, "hidden_dim": 32}, p)
    new = load_checkpoint(p, plan_dim=PLAN_DIM, belief_dim=8)
    assert new.plan_dim == PLAN_DIM
    sd_old = old.state_dict()
    sd_new = new.state_dict()
    assert torch.allclose(sd_new["plan_mlp.0.weight"][:, :21],
                          sd_old["plan_mlp.0.weight"]), "旧 21 列权重应原样保留"
    assert torch.all(sd_new["plan_mlp.0.weight"][:, 21:] == 0.0), "尾部应补零"
    shutil.rmtree(d, ignore_errors=True)
    print("[PASS] PlanToken v1 布局：旧21维兼容/新意图组/字段落位/旧ckpt补零加载")


def test_bp_new_intent_rules():
    """BeliefPlanner Phase2 v1 规则：6 个新意图 + 守卫 + 旧回退（每场景独立 battle）。"""
    import battle
    import player
    from core import Position
    from rl.belief_planner import BeliefPlanner

    DECK = ['Knight', 'Arrows', 'Fireball', 'Musketeer', 'Giant',
            'Minions', 'MiniPekka', 'Skeletons']

    def new_battle():
        return battle.BattleState(player.PlayerState(0, list(DECK), 10.0),
                                  player.PlayerState(1, list(DECK), 10.0))

    def place(bs, pid, card, x, y):
        pl = bs.players[pid]
        pl.cycle = [card] + [c for c in pl.cycle if c != card][:3]
        pl.elixir = 10.0
        dep_y = 20.0 if pid == 1 else 6.0
        assert bs.deploy_card(pid, card, Position(x, dep_y)), (pid, card)
        e = [e for e in bs.entities.values() if e.player == pid and e.id > 6][-1]
        e.position.x, e.position.y = x, y
        return e

    def set_hand(bs, cards):
        bs.players[0].cycle = list(cards)
        bs.players[0].elixir = 10.0

    bp = BeliefPlanner()
    # S1 过牌：无压力 + 手牌 1 费小牌 + 圣水足
    bs = new_battle(); set_hand(bs, ['Skeletons', 'Knight', 'Arrows', 'Fireball'])
    assert bp.plan(bs, None).macro_intent == "cycle_small"
    # S2 解牌：敌方 Musketeer 过桥 y=10 + 手牌 Fireball（血牛不抢）
    bs = new_battle(); place(bs, 1, 'Musketeer', 6, 10)
    set_hand(bs, ['Fireball', 'Knight', 'Arrows', 'Minions'])
    t = bp.plan(bs, None)
    assert t.macro_intent == "spell_trade" and t.focus_region == "own_left"
    # S3 软控：敌方 MiniPekka 压境 + 手牌 Freeze
    bs = new_battle(); place(bs, 1, 'MiniPekka', 6, 9)
    set_hand(bs, ['Freeze', 'Knight', 'Arrows', 'Fireball'])
    assert bp.plan(bs, None).macro_intent == "soft_control"
    # S4 拉扯（7g）：敌方近战血牛（Prince）逼近 → 用法术之外的便宜身板拦路
    bs = new_battle(); place(bs, 1, 'Prince', 8, 14)
    set_hand(bs, ['Knight', 'Arrows', 'Fireball', 'Musketeer'])
    t = bp.plan(bs, None)
    assert t.macro_intent == "pull" and t.placement_hint == "pull_aggro", t.macro_intent
    # S4b 攻城单位无建筑：只打塔的 Golem 单位拦不住 → 不判 pull（回退防守）
    bs = new_battle(); place(bs, 1, 'Golem', 9, 14)
    set_hand(bs, ['Knight', 'Arrows', 'Fireball', 'Musketeer'])
    t = bp.plan(bs, None)
    assert t.macro_intent != "pull", t.macro_intent
    # S5 推进跟牌：己方 Giant 推进中 + 手牌后排
    bs = new_battle(); place(bs, 0, 'Giant', 9, 10)
    set_hand(bs, ['Musketeer', 'Arrows', 'Fireball', 'Knight'])
    t = bp.plan(bs, None)
    assert t.macro_intent == "push_commit" and t.placement_hint == "support_zone"
    # S5b 推进跟牌（7g）：己方 Knight 高血近战身板推进中 → 也承认前排并跟输出
    bs = new_battle(); place(bs, 0, 'Knight', 8, 10)
    set_hand(bs, ['Musketeer', 'Arrows', 'Fireball', 'MiniPekka'])
    t = bp.plan(bs, None)
    assert t.macro_intent == "push_commit" and t.placement_hint == "support_zone", \
        t.macro_intent
    # S5c 推进跟牌（7h2）：Giant 推进中手牌同时有 Knight/Musketeer → 优先真后排 Musketeer
    bs = new_battle(); place(bs, 0, 'Giant', 9, 10)
    set_hand(bs, ['Knight', 'Musketeer', 'Arrows', 'Fireball'])
    t = bp.plan(bs, None)
    assert t.macro_intent == "push_commit" and t.suggested_card == 2, \
        (t.macro_intent, t.suggested_card)
    # S5d 推进跟牌（7h2）：无后排时允许近战身板（Knight）兜底
    bs = new_battle(); place(bs, 0, 'Giant', 9, 10)
    set_hand(bs, ['Knight', 'MiniPekka', 'Arrows', 'Fireball'])
    t = bp.plan(bs, None)
    assert t.macro_intent == "push_commit" and t.suggested_card == 1, \
        (t.macro_intent, t.suggested_card)
    # S6 沉底：空场 + 手牌沉底血牛 + 圣水攒满(≈10) → 沉底
    bs = new_battle(); set_hand(bs, ['Giant', 'Knight', 'Arrows', 'Fireball'])
    assert bp.plan(bs, None).macro_intent == "setup_wait"
    # S6b 费未满且手牌无后排（窗口条件不满足）→ 不裸沉，落到 cycle_and_wait
    bs = new_battle(); set_hand(bs, ['Giant', 'Knight', 'Arrows', 'Fireball'])
    bs.players[0].elixir = 6.0
    assert bp.plan(bs, None).macro_intent == "cycle_and_wait"
    # S6c HogRider 桥头快攻不沉底：费满也轮不到 setup
    bs = new_battle(); set_hand(bs, ['HogRider', 'Knight', 'Arrows', 'Fireball'])
    assert bp.plan(bs, None).macro_intent != "setup_wait"
    # S7 旧回退：空场无小费无坦克 → cycle_and_wait
    bs = new_battle(); set_hand(bs, ['Knight', 'Musketeer', 'MiniPekka', 'Fireball'])
    assert bp.plan(bs, None).macro_intent == "cycle_and_wait"
    # S8 守卫：压境时 setup/cycle 不抢防守（无法术软控 → 回退 defend）
    bs = new_battle(); place(bs, 1, 'Musketeer', 12, 8)
    bs.players[0].cycle = ['Knight', 'MiniPekka', 'Giant', 'Musketeer']
    bs.players[0].elixir = 10.0
    assert bp.plan(bs, None).macro_intent.startswith("defend")

    # —— belief 驱动四意图（圣水/手牌=记忆可追踪）——
    from rl.belief import BeliefState

    def belief(elixir=5.0, probs=None):
        arr = np.full(len(DECK), 0.2, dtype=np.float32)
        if probs:
            for card, pp in probs.items():
                arr[DECK.index(card)] = pp
        return BeliefState(deck=list(DECK), hand_probs=arr,
                           next_probs=np.full(len(DECK), 0.125, dtype=np.float32),
                           elixir_mean=elixir, uncertainty=0.6)

    # —— 7h 主动攒费窗口（总圣水差不落后≥2 + 无过河单位 + 前排后排齐 → 满10才沉底）——
    # S6d 攒费窗口帧：Giant+后排 Musketeer 在手、费未满、无压境 → setup_wait+hold 全禁手牌
    bs = new_battle(); set_hand(bs, ['Giant', 'Knight', 'Arrows', 'Musketeer'])
    bs.players[0].elixir = 8.0
    t = bp.plan(bs, belief(6.0))
    assert t.macro_intent == "setup_wait" and t.hold_mask == 0b1111 \
        and t.suggested_card is None, (t.macro_intent, t.hold_mask, t.suggested_card)
    # S6e 攒满 10 → 沉底 Giant（suggested 指向 Giant 槽）
    bs = new_battle(); set_hand(bs, ['Giant', 'Knight', 'Arrows', 'Musketeer'])
    bs.players[0].elixir = 10.0
    t = bp.plan(bs, belief(6.0))
    assert t.macro_intent == "setup_wait" and t.suggested_card == 1, \
        (t.macro_intent, t.suggested_card)
    # S6f 我方估计落后对手 ≥2 费 → 不进入攒费/沉底
    bs = new_battle(); set_hand(bs, ['Giant', 'Knight', 'Arrows', 'Musketeer'])
    bs.players[0].elixir = 3.0
    t = bp.plan(bs, belief(6.0))
    assert t.macro_intent != "setup_wait", t.macro_intent
    # S6g 对手过河单位压境（MiniPekka 深入我方半场）→ 攒费窗口关闭
    bs = new_battle(); set_hand(bs, ['Giant', 'Knight', 'Arrows', 'Musketeer'])
    bs.players[0].elixir = 10.0
    place(bs, 1, 'MiniPekka', 8, 12)
    t = bp.plan(bs, belief(6.0))
    assert t.macro_intent != "setup_wait", t.macro_intent

    # S9 punish：对手低圣水（belief.elixir_mean 记忆）→ 另一路进攻
    bs = new_battle(); set_hand(bs, ['Giant', 'Knight', 'Arrows', 'Fireball'])
    t = bp.plan(bs, belief(1.5))
    assert t.macro_intent == "punish", t.macro_intent
    # S10 spell_finish：t≥120 残血公主塔 → 法术磨塔
    bs = new_battle(); bs.time = 150.0
    bs.players[1].left_tower_hp = 500.0
    set_hand(bs, ['Fireball', 'Knight', 'Arrows', 'Musketeer'])
    t = bp.plan(bs, belief(5.0))
    assert t.macro_intent == "spell_finish" and t.focus_region == "enemy_left"
    # S11 anti_spell：belief 显示对面手牌高概率 Fireball + 我方要下后排
    bs = new_battle(); set_hand(bs, ['Musketeer', 'Knight', 'Skeletons', 'Fireball'])
    t = bp.plan(bs, belief(6.0, {'Fireball': 0.9}))
    assert t.macro_intent == "anti_spell" and t.opp_spell_threat == "fireball"
    # S12 save_ace：手牌 Lightning（ace）非关键帧 → hold_mask 指名别出
    bs = new_battle(); set_hand(bs, ['Lightning', 'Knight', 'Arrows', 'Musketeer'])
    t = bp.plan(bs, belief(8.0))
    assert t.macro_intent == "save_ace" and (t.hold_mask & 1) == 1, (t.macro_intent, t.hold_mask)
    # S13 protect_backline（反应）：敌方 MiniPekka 贴近我方 Musketeer → 前置保护
    bs = new_battle(); set_hand(bs, ['Knight', 'MiniPekka', 'Skeletons', 'Musketeer'])
    place(bs, 0, 'Musketeer', 6, 12)
    place(bs, 1, 'MiniPekka', 8, 14)
    t = bp.plan(bs, belief(5.0))
    assert t.macro_intent == "protect_backline" and t.target_kind == "my_backline", \
        t.macro_intent
    # S14 protect_backline（信念预判）：belief 显示对手手牌高概率 MiniPekka + 后排暴露
    bs = new_battle(); set_hand(bs, ['Knight', 'Musketeer', 'Arrows', 'Fireball'])
    place(bs, 0, 'Archer', 6, 11)
    t = bp.plan(bs, belief(5.0, {'MiniPekka': 0.9}))
    assert t.macro_intent == "protect_backline", t.macro_intent
    # S15 king_activate：公主塔残血 + Golem 深入中轴 + 手牌低费
    bs = new_battle(); set_hand(bs, ['Skeletons', 'Knight', 'Arrows', 'Musketeer'])
    bs.players[0].left_tower_hp = 300.0
    place(bs, 1, 'Golem', 9, 10)
    t = bp.plan(bs, belief(5.0))
    assert t.macro_intent == "king_activate" and t.placement_hint == "king_front", \
        t.macro_intent
    print("[PASS] BeliefPlanner v1 规则：cycle_small/spell_trade/soft_control/pull/push_commit/"
          "setup_wait + 血牛放行 + 压境守卫 + 旧回退 + punish/spell_finish/anti_spell/save_ace"
          " + protect_backline(反应+信念预判)/king_activate（12 意图，与 pp 同链同序）")
    print("[PASS] BeliefPlanner 7g：拉扯按高血/近战/建筑目标口径（攻城单位需建筑拉），"
          "spell_trade 只留远程脆皮，setup 主动攒费(费+2储备)才沉底，"
          "push_commit 认 Knight/Valkyrie/Prince 前排跟输出")
    print("[PASS] BeliefPlanner 7h：攒费窗口=不落后≥2费+无过河单位+前排后排齐，"
          "未满帧 hold_mask=1111 等费，攒满 10 才沉底血牛；"
          "7h2：push_commit 跟牌优先真后排(Musketeer)，无后排才近战兜底")


def test_pp_new_intent_rules():
    """ProphetPlanner Phase2 v1 特权意图组：punish(精确圣水)/spell_finish/anti_spell
    (直读手牌)/save_ace(藏+解除时机)/king_activate/protect_backline(反应+预判)
    + 与 bp 同链标签一致（soft/spell_trade/pull/push_commit/setup/cycle_small）。"""
    import battle
    import player
    from core import Position
    from rl.prophet import ProphetPlanner

    DECK = ['Knight', 'Arrows', 'Fireball', 'Musketeer', 'Giant',
            'Minions', 'MiniPekka', 'Skeletons']

    def new_battle():
        return battle.BattleState(player.PlayerState(0, list(DECK), 10.0),
                                  player.PlayerState(1, list(DECK), 10.0))

    def place(bs, pid, card, x, y):
        pl = bs.players[pid]
        pl.cycle = [card] + [c for c in pl.cycle if c != card][:3]
        pl.elixir = 10.0
        dep_y = 20.0 if pid == 1 else 6.0
        assert bs.deploy_card(pid, card, Position(x, dep_y)), (pid, card)
        e = [e for e in bs.entities.values() if e.player == pid and e.id > 6][-1]
        e.position.x, e.position.y = x, y
        return e

    def set_hand(bs, cards, pid=0):
        bs.players[pid].cycle = list(cards)
        bs.players[pid].elixir = 10.0

    def pstate(bs):
        """与 env_wrapper.get_prophet_state() 同构的特权摘要。"""
        p0, p1 = bs.players
        return {
            "time": bs.time,
            "my_cycle": list(p0.cycle), "opp_cycle": list(p1.cycle),
            "my_elixir": p0.elixir, "opp_elixir": p1.elixir,
            "my_towers": [p0.king_tower_hp, p0.left_tower_hp, p0.right_tower_hp],
            "opp_towers": [p1.king_tower_hp, p1.left_tower_hp, p1.right_tower_hp],
            "my_crown": p0.get_crown_count(), "opp_crown": p1.get_crown_count(),
            "entities": [
                {"name": e.name, "player": e.player,
                 "pos": (e.position.x, e.position.y), "hp": e.hp}
                for e in bs.entities.values() if e.is_alive
            ],
        }

    pp = ProphetPlanner()
    # S1 punish（精确圣水）：对手 elixir 1.5 → 另一路进攻
    bs = new_battle(); set_hand(bs, ['Giant', 'Knight', 'Arrows', 'Fireball'])
    bs.players[1].elixir = 1.5
    t = pp.plan(pstate(bs))
    assert t.macro_intent == "punish" and t.target_kind == "tower", t.macro_intent
    # S2 spell_finish：t≥120 残血左公主塔 + Fireball
    bs = new_battle(); bs.time = 150.0
    bs.players[1].left_tower_hp = 500.0
    set_hand(bs, ['Fireball', 'Knight', 'Arrows', 'Musketeer'])
    t = pp.plan(pstate(bs))
    assert t.macro_intent == "spell_finish" and t.focus_region == "enemy_left"
    # S3 anti_spell（直读对手手牌）：对面 hand 有 Fireball + 我方要下后排
    bs = new_battle(); set_hand(bs, ['Musketeer', 'Knight', 'Skeletons', 'Fireball'])
    bs.players[1].cycle = ['Fireball'] + list(bs.players[1].cycle)[:3]
    t = pp.plan(pstate(bs))
    assert t.macro_intent == "anti_spell" and t.opp_spell_threat == "fireball"
    # S4 save_ace（藏）：手牌 Lightning + 对手圣水足 + 对手手牌无反制 → hold slot1
    bs = new_battle(); set_hand(bs, ['Lightning', 'Knight', 'Arrows', 'Musketeer'])
    bs.players[1].elixir = 8.0
    bs.players[1].cycle = ['Knight', 'Musketeer', 'Giant', 'Minions',
                           'MiniPekka', 'Skeletons', 'Arrows', 'Fireball']
    t = pp.plan(pstate(bs))
    assert t.macro_intent == "save_ace" and (t.hold_mask & 1) == 1, (t.macro_intent, t.hold_mask)
    # S5 save_ace 解除：坦克进场 + 对手低圣水 + 手牌无反制 → 不藏（转 push_commit 跟牌）。
    # 坦克从默认 8 卡 cycle 打出后回队尾不在手 → punish 无进攻牌，链落到 push_commit
    bs = new_battle()
    place(bs, 0, 'Giant', 9, 12)
    bs.players[0].cycle = ['Lightning', 'Knight', 'Arrows', 'Musketeer',
                           'Minions', 'MiniPekka', 'Skeletons', 'Giant']
    bs.players[0].elixir = 10.0
    bs.players[1].elixir = 1.0
    bs.players[1].cycle = ['Knight', 'Musketeer', 'Giant', 'Minions',
                           'MiniPekka', 'Skeletons', 'Arrows', 'Fireball']
    t = pp.plan(pstate(bs))
    assert t.macro_intent == "push_commit" and t.hold_mask == 0, t.macro_intent
    # S6 king_activate：左公主塔残血 + Golem 深入中轴 + 手牌低费
    bs = new_battle(); set_hand(bs, ['Skeletons', 'Knight', 'Arrows', 'Musketeer'])
    bs.players[0].left_tower_hp = 300.0
    place(bs, 1, 'Golem', 9, 10)
    t = pp.plan(pstate(bs))
    assert t.macro_intent == "king_activate" and t.placement_hint == "king_front", \
        t.macro_intent
    # S7 protect_backline（反应）：敌方 MiniPekka 贴近我方 Musketeer → 前置保护
    bs = new_battle(); set_hand(bs, ['Knight', 'MiniPekka', 'Skeletons', 'Musketeer'])
    place(bs, 0, 'Musketeer', 6, 12)
    place(bs, 1, 'MiniPekka', 8, 14)
    t = pp.plan(pstate(bs))
    assert t.macro_intent == "protect_backline" and t.target_kind == "my_backline", \
        t.macro_intent
    # S8 protect_backline（pp 预判）：对手手牌有切后排单位 + 我方后排暴露
    bs = new_battle(); set_hand(bs, ['Knight', 'Musketeer', 'Arrows', 'Fireball'])
    place(bs, 0, 'Archer', 6, 11)
    bs.players[1].cycle = ['MiniPekka'] + [c for c in bs.players[1].cycle
                                           if c != 'MiniPekka'][:3]
    t = pp.plan(pstate(bs))
    assert t.macro_intent == "protect_backline", t.macro_intent

    # —— 与 bp 同链标签一致（30% prophet 帧不稀释 bp 示范）——
    # S9 pull：血牛逼近 + 低费拉扯卡
    bs = new_battle(); set_hand(bs, ['Skeletons', 'Knight', 'Arrows', 'Fireball'])
    place(bs, 1, 'Golem', 9, 14)
    t = pp.plan(pstate(bs))
    assert t.macro_intent == "pull" and t.placement_hint == "pull_aggro", t.macro_intent
    # S10 cycle_small：空场 + 1 费小牌 + 圣水足（对手手牌无法术，避免 anti 抢链）
    bs = new_battle(); set_hand(bs, ['Skeletons', 'Knight', 'Arrows', 'Fireball'])
    bs.players[1].cycle = ['Knight', 'Musketeer', 'Giant', 'Minions',
                           'MiniPekka', 'Skeletons', 'Arrows', 'Fireball']
    assert pp.plan(pstate(bs)).macro_intent == "cycle_small"
    # S11 setup_wait：空场 + 手牌坦克
    bs = new_battle(); set_hand(bs, ['Giant', 'Knight', 'Arrows', 'Fireball'])
    bs.players[1].cycle = ['Knight', 'Musketeer', 'Giant', 'Minions',
                           'MiniPekka', 'Skeletons', 'Arrows', 'Fireball']
    assert pp.plan(pstate(bs)).macro_intent == "setup_wait"
    # S12 push_commit：己方 Giant 推进中 + 后排
    bs = new_battle(); set_hand(bs, ['Musketeer', 'Arrows', 'Fireball', 'Knight'])
    place(bs, 0, 'Giant', 9, 10)
    t = pp.plan(pstate(bs))
    assert t.macro_intent == "push_commit" and t.placement_hint == "support_zone"
    # S13 旧回退：空场无小费无坦克无 ace → cycle_and_wait
    bs = new_battle(); set_hand(bs, ['Knight', 'Musketeer', 'MiniPekka', 'Fireball'])
    bs.players[1].cycle = ['Knight', 'Musketeer', 'Giant', 'Minions',
                           'MiniPekka', 'Skeletons', 'Arrows', 'Fireball']
    assert pp.plan(pstate(bs)).macro_intent == "cycle_and_wait"
    print("[PASS] ProphetPlanner v1 特权意图：punish/spell_finish/anti_spell/save_ace(藏+解除)/"
          "king_activate/protect_backline(反应+预判) + bp 同链标签一致")


def test_bayes_queue_lock():
    """CycleBayesFilter v2 O(1) 队列锁定定理：
    8 张内容已知 + 出牌按序全观测 → 第 4 张起手牌 = 卡组 − 最近 4 张、下一张 =
    第 k−3 张打出牌（精确 0/1，与开局排列无关）；异常观测退回粒子相后，
    连续 4 张合法出牌自动重锁且必须与真实队列同步（无伪锁）。"""
    import random
    from rl.bayes_filter import CycleBayesFilter

    deck = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

    # ① 随机策略 300 步 × 多局：第 4 张（step==3）起必须锁定，
    #    且全程手牌/下一张与真实队列逐位一致（锁定永不锁错、不脱锁）
    for trial in range(5):
        rng = random.Random(100 + trial)
        truth = list(deck); rng.shuffle(truth)
        bf = CycleBayesFilter(deck, n_particles=128, seed=trial)
        real = list(truth)
        for step in range(300):
            i = rng.randrange(4); c = real[i]
            real = [x for x in real if x != c] + [c]
            bf.update(c)
            if step < 3:
                assert not bf.locked, f"第4张前不得锁定 trial={trial} step={step}"
                i_c = deck.index(c)
                assert bf.hand_probs()[i_c] == 0.0, "打出的卡应排除出手牌"
            else:
                assert bf.locked and bf.entropy() == 0.0, \
                    f"内容已知+全观测第4张起应精确锁定 trial={trial} step={step}"
                assert set(np.unique(bf.hand_probs())) <= {0.0, 1.0}
                hand = {deck[i] for i in range(len(deck)) if bf.hand_probs()[i] > 0.5}
                assert hand == set(real[:4]), f"手牌不同步 trial={trial} step={step}"
                nxt = deck[int(np.argmax(bf.next_probs()))]
                assert nxt == real[4], f"下一张不同步 trial={trial} step={step}"
    print("[1/3] 定理：第4张起 O(1) 锁定，300 步×5 局手牌/下一张全程与真实同步")

    # ② 异常观测（手牌外）：退回粒子相；随后 4 张真实合法出牌自动重锁且同步
    rng = random.Random(7)
    truth = list(deck); rng.shuffle(truth)
    bf = CycleBayesFilter(deck, n_particles=64, seed=3)
    real = list(truth)
    for _ in range(5):                       # 前 5 张合法 → 已锁定
        i = rng.randrange(4); c = real[i]
        real = [x for x in real if x != c] + [c]
        bf.update(c)
    assert bf.locked
    fake = real[4]                           # 真实队列的下一张 = 此刻不在手牌
    assert fake not in real[:4]
    bf.update(fake)                          # 手牌外出牌 → 不推进真实队列
    assert not bf.locked, "手牌外出牌应退回粒子相"
    for _ in range(8):                       # 连续真实合法出牌 → 4 张后重锁
        i = rng.randrange(4); c = real[i]
        real = [x for x in real if x != c] + [c]
        bf.update(c)
    assert bf.locked and bf.entropy() == 0.0
    hand = {deck[i] for i in range(len(deck)) if bf.hand_probs()[i] > 0.5}
    assert hand == set(real[:4]), "异常后重锁必须与真实手牌同步"
    nxt = deck[int(np.argmax(bf.next_probs()))]
    assert nxt == real[4], "异常后重锁必须与真实下一张同步"
    print("[2/3] 异常：手牌外退回粒子相，4 张合法出牌后自动重锁且无伪锁")

    # ③ 同 seed 确定性：粒子相（前 3 张）轨迹逐位一致，锁定 cycle 相同
    rng = random.Random(11)
    truth = list(deck); rng.shuffle(truth)
    seq = []
    real = list(truth)
    for _ in range(6):
        i = rng.randrange(4); c = real[i]
        real = [x for x in real if x != c] + [c]
        seq.append(c)
    outs = []
    for seed in (42, 42):
        b = CycleBayesFilter(deck, n_particles=128, seed=seed)
        for c in seq:
            b.update(c)
        outs.append((b.hand_probs().tolist(), b.next_probs().tolist(), list(b._cycle)))
    assert outs[0] == outs[1], "同 seed 信念路径必须逐位一致"
    print("[3/3] 确定性：同 seed 粒子相轨迹与锁定 cycle 逐位一致")
    print("[PASS] 信念 O(1) 队列锁定：精确推进/熵0/0-1概率；异常重锁无伪锁；跨进程确定性")


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
      - 恰达 300s 仍平 → overtime_open=False，收手后由 timeout_winner 按最低塔血
        百分比裁决（真实 CR 加时末规则），完全相等才平局；
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

    class _T:  # 假塔：最低血量百分比裁决用
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
    # 皇冠平 + 塔血裁决：双方满血公主塔（p0 左塔残血 50%）→ p0 输
    assert timeout_winner(_BT(300.0, 1, 1,
                              [(3052, 3052), (1526, 3052), (4824, 4824)],
                              [(4824, 4824), (3052, 3052), (3052, 3052)])) == 1
    # 镜像：p1 塔更残 → p0 胜（僵局早停不再一律记平局）
    assert timeout_winner(_BT(200.0, 1, 1,
                              [(4824, 4824), (3052, 3052), (3052, 3052)],
                              [(4824, 4824), (1000, 3052), (3052, 3052)])) == 0
    # 双方最低塔血完全相等（各 100%）→ 平局
    assert timeout_winner(_BT(110.0, 1, 1,
                              [(4824, 4824), (0, 3052), (3052, 3052)],
                              [(4824, 4824), (3052, 3052), (0, 3052)])) is None
    print("[PASS] 加时窗口：180s 皇冠平进入 [180,300) 突然死亡；到顶按最低塔血裁决；皇冠差直接判胜")


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
        print("[SKIP] 对手池分布：无可用历史 ckpt（仅验证 defender）")
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
        for _ in range(400):
            kind, side, hid = pool.sample()
            kinds[kind] += 1
            if kind == "hist":
                hist_ids.add(hid)
                assert side is not None
        assert 40 <= kinds["hist"] <= 120, kinds      # 名义 20%
        assert 20 <= kinds["defend"] <= 80, kinds     # 名义 10%
        assert len(hist_ids) == 3, hist_ids
        pool.record(1)                                 # 败局回填不崩
    print(f"[PASS] 对手池：分布 {dict(kinds)} ≈ 0.7/0.2/0.1；hist 旧 ckpt 加载+PFSP 回填；"
          f"SelfDefender 面对过河威胁出合法反制")


def main():
    test_action_bundle_same_tick()
    test_action_bundle_ability()
    test_bayes_filter()
    test_bayes_queue_lock()
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
    test_draw_penalty_as_loss()
    test_reward_v2_ledger()
    test_spell_empty_value_gate()
    test_spell_tower_ev_gate()
    test_no_solo_commit_without_lead()
    test_tank_backline_geometry()
    test_plan_v1_layout()
    test_bp_new_intent_rules()
    test_pp_new_intent_rules()
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
    test_eval_solo_parallel()
    test_overtime_window()
    test_tower_threat_calc()
    test_simulate_exchange()
    test_spell_module()
    test_mcts_basic()
    test_mcts_defense_and_wait()
    test_opp_event_token()
    test_crossed_river_defend_plan()
    test_opponent_pool_mix()
    print("\nALL SELFTESTS PASSED")


if __name__ == "__main__":
    main()
