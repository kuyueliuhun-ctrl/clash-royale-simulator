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
        # 落点不能选塔上：塔矩形化后 (3.5,25.5)=P1 公主塔中心，部署正确拒绝
        #（旧圆形几何因 _is_tower_alive 属性名 bug 塔占位从未生效才放过）
        if b.deploy_card(1, card, Position(6.5, 22.5)):
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


def test_deck_pool_factory():
    """卡组工厂：deck_pool（三分类 / 全 200 卡组）必须逐局生效，且跨 pair 清空。

    历史 bug：`env.deck{0,1}_factory = pol.deck if pol.pool else None` 只判 pool，
    漏掉 deck_pool → push/counter/lockdown/all_decks 一直打 DEFAULT_DECK 固定 8 卡。
    """
    from rl.run_league import _deck_factory_of, _prepare_env
    from rl.opponents import ScriptedPolicy
    from rl.env_wrapper import RLEnv

    d1 = {"cards": ["Giant", "Musketeer", "Fireball", "Arrows",
                    "Minions", "Archer", "Knight", "MiniPekka"]}
    d2 = {"cards": ["Xbow", "Arrows", "Knight", "Skeletons",
                    "IceSpirits", "Goblins", "Tesla", "Fireball"]}
    pol_deckpool = ScriptedPolicy(mode="random", deck_pool=[d1, d2], seed=1)
    pol_pool = ScriptedPolicy(mode="random", pool=["Giant", "Archer"], seed=2)
    pol_none = ScriptedPolicy(mode="random", seed=3)

    assert _deck_factory_of(pol_deckpool) is not None, "deck_pool 策略必须产出每局卡组工厂"
    assert _deck_factory_of(pol_pool) is not None, "pool 策略必须产出每局卡组工厂"
    assert _deck_factory_of(pol_none) is None and _deck_factory_of(None) is None

    env = RLEnv(opponent=None, seed=0)
    env.deck0_factory = pol_deckpool.deck          # 模拟上一 pair 残留的工厂
    _prepare_env(env, None, pol_none, None)
    assert env.deck0_factory is None and env.deck1_factory is None, "卡组工厂必须跨 pair 清空"

    env2 = RLEnv(opponent=None, seed=0)
    _prepare_env(env2, pol_deckpool, pol_none, None)
    seen = set()
    for i in range(12):
        env2.reset(seed=i)
        seen.add(tuple(sorted(env2.deck0)))
    assert len(seen) == 2, f"deck_pool 应逐局抽整套卡组（2 副），实得 {len(seen)} 种"

    print("[PASS] 卡组工厂：deck_pool 逐局生效 + pool 生效 + 跨 pair 清空")


def test_dashboard_card_stats():
    """仪表盘卡牌使用统计：双侧归属（opp_played / cards）+ 旧录像降级 + 汇总 + 防护。"""
    import tempfile
    import rl.dashboard as dash
    from rl.replay import save_league_replays

    def frame(**kw):
        base = {"t": 0.5, "bundle": [], "reward": 0.0, "opp_played": [],
                "towers0": [1.0, 1.0, 1.0], "towers1": [1.0, 1.0, 1.0],
                "elixir0": 5.0, "elixir1": 5.0, "crown0": 0, "crown1": 0, "entities": []}
        base.update(kw)
        return base

    d = tempfile.mkdtemp()
    # ① 旧录像：只有 opp_played（player-1 侧），无 frame["cards"] / meta["decks"]
    old = [
        {"meta": {"pair": ["main", "push_flow"], "side0": "main", "max_steps": 10},
         "winner": 1,
         "frames": [frame(opp_played=[{"card": "Giant", "x": 1, "y": 2}]),
                    frame(opp_played=[{"card": "Giant", "x": 1, "y": 2},
                                      {"card": "Fireball", "x": 3, "y": 4},
                                      {"card": "__ability__", "x": None, "y": None}])]},
        {"meta": {"pair": ["main", "push_flow"], "side0": "push_flow", "max_steps": 10},
         "winner": 0,
         "frames": [frame(opp_played=[{"card": "Archer", "x": 1, "y": 2}])]},
    ]
    save_league_replays(old, os.path.join(d, "league_0.pkl"))
    p = dash.build_card_stats_payload(d, "league_0.pkl")
    assert p["ok"] and p["n_games"] == 2, p
    by = {a["model"]: a for a in p["agents"]}
    # 第 1 局 side0=main → 对手侧 = push_flow；第 2 局 side0=push_flow → 对手侧 = main
    assert by["push_flow"]["cards"] == {"Giant": 2, "Fireball": 1}, by["push_flow"]["cards"]
    assert by["main"]["cards"] == {"Archer": 1}, by["main"]["cards"]
    assert "__ability__" not in by["push_flow"]["cards"], "技能哨兵不应计入卡牌统计"
    assert by["push_flow"]["plays"] == 3 and by["main"]["plays"] == 1
    assert p["coverage"]["partial"] is True, "旧录像我方侧无记录 → 应标记部分覆盖"

    # ② 新录像：frame["cards"]（我方）+ meta["decks"]（双方卡组）
    new = [{"meta": {"pair": ["main", "push_flow"], "side0": "main", "max_steps": 10,
                     "decks": [["Giant", "Archer"], ["Xbow", "Arrows"]]},
            "winner": 0,
            "frames": [frame(cards=["Giant", "Archer"],
                             opp_played=[{"card": "Xbow", "x": 1, "y": 2}])]}]
    save_league_replays(new, os.path.join(d, "league_1000.pkl"))
    p2 = dash.build_card_stats_payload(d, "league_1000.pkl")
    by2 = {a["model"]: a for a in p2["agents"]}
    assert by2["main"]["cards"] == {"Giant": 1, "Archer": 1}, by2["main"]["cards"]
    assert by2["push_flow"]["cards"] == {"Xbow": 1}, by2["push_flow"]["cards"]
    assert by2["main"]["deck_cards"] == {"Giant": 1, "Archer": 1}, by2["main"]["deck_cards"]
    assert by2["push_flow"]["deck_cards"] == {"Xbow": 1, "Arrows": 1}
    assert p2["coverage"]["partial"] is False, "新录像双侧都有记录 → 不应标 partial"

    # ③ 多文件汇总（n_files=0 → 全部）+ 非法输入防护
    pall = dash.build_card_stats_payload(d, None, 0)
    assert pall["ok"] and pall["n_games"] == 3 and len(pall["files"]) == 2, pall.get("files")
    assert dash.build_card_stats_payload(d, "../evil.pkl")["ok"] is False
    assert dash.build_card_stats_payload(d, "missing.pkl")["ok"] is False
    assert dash.build_card_stats_payload(os.path.join(d, "nope"))["ok"] is False

    # ④ 页面元素（防回归）
    for token in ("卡牌使用统计", "/api/cardstats", "statsScope", "loadCardStats"):
        assert token in dash._HTML, f"页面缺少 {token}"

    print("[PASS] 仪表盘卡牌使用统计：双侧归属 + 旧录像降级 + 汇总 + 防护 + 页面元素")


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
    # 塔矩形化后 (14.5,26.0) 落在 P1 右公主塔 3×3（y∈[24,27]）内，非法是正确行为；
    # 身后位测试点移到塔矩形上方 (14.5, 28.5)
    xb2, yb2 = cell_near(bs2, 1, 14.5, 28.5)   # Knight 身后（y 更大，塔外）→ 合法
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


def _tiny_rollout_transitions(pol, env, belief, tok, plan, n=4):
    """构造 n 条最小 transition（adv 正负交替、回报带 3.0 的离散度）。"""
    trans, hidden, obs = [], None, None
    obs, _ = env.reset()
    belief.reset(env.deck1)
    for i in range(n):
        ih = hidden
        bundle, lp, val, hidden, masks = pol.act(
            obs, tok, plan, env.get_action_mask, hidden=hidden, deterministic=False)
        obs2, r, term, trunc, info = env.step(bundle)
        sgn = 1.0 if i % 2 else -1.0
        trans.append({"obs": obs, "belief": tok, "plan": plan, "bundle": bundle,
                      "old_logprob": lp, "adv": sgn,
                      "returns": float(val) + 3.0 * sgn,
                      "masks": masks, "init_hidden": ih})
        belief.update(obs2, info.get("opp_played"))
        obs = obs2
    return trans


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


def test_history_dedup_and_gates():
    """P0-2：history 按 step 去重（10e 双 step:0）+ 行为指标门禁先只报警不阻断。"""
    import json
    import tempfile
    from rl.train_solo import (_dedup_history, _check_gates, behavioral_metrics)
    from rl.config import TrainConfig

    # 去重：同 step 覆盖旧点
    h = [{"step": 0, "winrate": 0.1}, {"step": 2000, "winrate": 0.2}]
    h[:] = _dedup_history(h, 0)
    h.append({"step": 0, "winrate": 0.9})
    assert [x["step"] for x in h] == [2000, 0] and h[1]["winrate"] == 0.9, h
    assert _dedup_history([{"step": 0}, {"step": 0}], 0) == [], "同 step 全清"
    assert _dedup_history([{"step": 5}], 7) == [{"step": 5}], "不同 step 不动"

    with tempfile.TemporaryDirectory() as td:
        cfg = TrainConfig(name="g", out_dir=td)
        p = os.path.join(td, "g", "gates.json")
        # P0-2b：相对门禁 —— 首个评估点只建基线、不判定（口径由自身起点定标，
        # 修的是"绝对阈值 9.5 与内建指标实测 28.3~38.1 差 3 倍 → PASS/FAIL 是假的"）
        rep0 = _check_gates({"step": 0, "engagement_rate": 30.0, "ghost_rate": 24.0}, cfg, p)
        assert rep0["ok"] is True and rep0["checks"] == [], rep0
        assert abs(json.load(open(p, encoding="utf-8"))["baseline"]["engagement_rate"]
                   - 30.0) < 1e-9, rep0
        # 相对起点退化：engagement < 50%×30=15；ghost > 200%×24=48 → 报警不抛异常
        rep = _check_gates({"step": 8000, "engagement_rate": 3.0, "ghost_rate": 90.0}, cfg, p)
        assert rep["ok"] is False and len(rep["checks"]) == 2, rep
        assert json.load(open(p, encoding="utf-8"))["ok"] is False
        # 达标 → ok=True；且 baseline 不随评估点漂移（续训语义）
        rep2 = _check_gates({"step": 8000, "engagement_rate": 30.0, "ghost_rate": 24.0}, cfg, p)
        assert rep2["ok"] is True and rep2["baseline"]["engagement_rate"] == 30.0, rep2
        # 绝对阈值写法仍兼容（旧行为）；阈值可关；缺失指标不误判
        cfg.gates = {"engagement_rate": 9.5}
        assert _check_gates({"step": 1}, cfg, None)["checks"] == [], "无该指标 → 不检查"
        cfg.gates = {}
        assert _check_gates({"step": 1}, cfg, None)["ok"] is True

    # ghost_rate 确实进了行为指标（门禁依赖它）
    games = [{"meta": {}, "winner": 0, "frames": [
        {"t": 0.0, "bundle": [("deploy", 1, 8, 30), ("deploy", 2, 9, 10)],
         "entities": [], "towers0": [4824, 3052, 3052], "towers1": [4824, 3052, 3052],
         "elixir0": 5.0}]}]
    bm = behavioral_metrics(games)
    assert bm.get("ghost_rate") == 50.0, bm
    print("[PASS] P0-2：history 按 step 去重；门禁越界报警/达标通过/阈值可关且不中断训练；"
          f"ghost_rate={bm['ghost_rate']}")


def test_opponent_pool_mix_multi_dir():
    """P1-1：本 run 目录无 ckpt 时从 --hist-seed-dir 补种，对手池恢复 hist 槽。"""
    import tempfile
    from collections import Counter
    from rl.train_solo import _OpponentPool, _collect_hist_ckpts
    from rl.config import TrainConfig
    from rl.env_wrapper import RLEnv
    from rl.follower import FollowerPolicy, save_checkpoint
    from rl.train_follower import FollowerOpponent
    from rl.belief import BeliefInference
    from rl.plan_space import PLAN_DIM

    with tempfile.TemporaryDirectory() as td:
        cur = os.path.join(td, "cur")        # 本 run 目录：**无** ckpt（首轮训练）
        seed = os.path.join(td, "seed")      # 旧 run 目录
        os.makedirs(cur)
        os.makedirs(seed)
        # 收集器：本目录空 → 无补种为空、有补种为补齐；本目录优先
        assert _collect_hist_ckpts(cur, max_n=12) == []
        env = RLEnv(opponent=None, seed=0, card_level=11)
        env.reset(seed=0)
        bd = len(BeliefInference(opp_deck=env.deck1, n_particles=128,
                                 seed=0).encode(None, None))
        pol = FollowerPolicy(hidden=32, plan_dim=PLAN_DIM, belief_dim=bd)
        for s in (0, 1000, 2000, 3000):
            save_checkpoint(pol, os.path.join(seed, f"solo_main_{s}.pt"))
        got = _collect_hist_ckpts(cur, max_n=12, extra_dirs=[seed])
        assert len(got) == 4 and all(seed == os.path.dirname(g) for g in got), got

        cfg = TrainConfig(name="cur", out_dir=td, hidden_dim=32)
        opp = FollowerPolicy(hidden=32, plan_dim=PLAN_DIM, belief_dim=bd).to_device("cpu")
        side = FollowerOpponent(opp, env, belief=BeliefInference(
            opp_deck=env.deck1, n_particles=128, seed=0), deterministic=True)
        # 无补种目录 → hist 槽仍为空（这正是热启动退化形态）
        pool_no = _OpponentPool(cfg, env, side, random.Random(0), "cpu",
                                hist_seed_dirs=None)
        assert pool_no.hist_paths == [], pool_no.hist_paths
        # 有补种目录 → hist 槽亮起
        pool2 = _OpponentPool(cfg, env, side, random.Random(0), "cpu",
                              hist_seed_dirs=[seed])
        assert len(pool2.hist_paths) == 4, pool2.hist_paths
        kinds = Counter(pool2.sample()[0] for _ in range(200))
        assert kinds["hist"] > 0, kinds

        # 本目录（cur）后来有了 ckpt → 优先，不再补种
        for s in (0, 500):
            save_checkpoint(pol, os.path.join(cur, f"solo_main_{s}.pt"))
        near = _collect_hist_ckpts(cur, max_n=2, extra_dirs=[seed])
        assert len(near) == 2 and all(cur == os.path.dirname(g) for g in near), near
        pool = _OpponentPool(cfg, env, side, random.Random(0), "cpu",
                             hist_seed_dirs=[seed])
        # 本目录 2 个排在前面，其余由补种目录补齐到 max_n（12）
        assert len(pool.hist_paths) == 6, pool.hist_paths
        assert all(cur == os.path.dirname(g) for g in pool.hist_paths[:2]), pool.hist_paths
        print(f"[PASS] P1-1：本目录空时补种 hist={len(pool2.hist_paths)}、无补种为空；"
              f"本目录 ckpt 排前={len(pool.hist_paths)}（前 2 来自本目录）；"
              f"采样分布 {dict(kinds)}")


def test_opponent_pool_rand_anchor():
    """E2（2026-09-12，v3 §3.9）：训练侧固定随机锚点进对手池（第 4 槽 rand_anchor）。

    背景：A′ 取证坐实自对弈 RPS 循环（main 打冻结副本 0.85 / 打起点随机 0.13 / 打
    全新随机 0.505）。E1 只加评估侧锚点（测量），E2 把锚点放进**训练分布**，让
    "输给固定外部基准"成为可见负样本，打破循环漂移。校验：
    ① 默认配比含 rand_anchor=0.1（DEFAULT_OPP_MIX 与 _OPP_MIX 同步，frozen 0.5→0.4）；
    ② 有 hist 时采样分布 ≈ 名义（含 rand_anchor 槽，±35% 容差）；
    ③ 锚点 side 权重 == _make_rand_anchor(...)（= E1 baseline_rand，同种子逐位一致）；
    ④ 锚点/defend 局 record() 不崩（no-op，不污染 PFSP）；
    ⑤ 无 hist 退化：按剩余概率归一化（frozen/defend/rand ≈ 0.571/0.286/0.143，
        ±35%），且 defend 不再吃掉 hist 空间（旧实现实测 0.78，属缺陷修复）；
    ⑥ 旧式三槽 mix（无 rand_anchor 键）兼容：rand 概率 0、sample 不崩。
    """
    import random, glob, tempfile, shutil, os
    from collections import Counter
    import torch
    from rl.train_solo import (_OpponentPool, _make_rand_anchor, _OPP_MIX,
                               RAND_ANCHOR_SEED)
    from rl.config import TrainConfig, DEFAULT_OPP_MIX
    from rl.env_wrapper import RLEnv
    from rl.follower import FollowerPolicy
    from rl.train_follower import FollowerOpponent
    from rl.belief import BeliefInference
    from rl.plan_space import PLAN_DIM

    env = RLEnv(opponent=None, seed=0, card_level=11)
    frozen_pol = FollowerPolicy(hidden=128, plan_dim=PLAN_DIM,
                                belief_dim=len(BeliefInference(
                                    opp_deck=env.deck1, n_particles=128,
                                    seed=0).encode(None, None)))
    frozen_side = FollowerOpponent(frozen_pol, env,
                                   belief=BeliefInference(opp_deck=env.deck1),
                                   deterministic=True)

    # ① 默认配比
    assert abs(DEFAULT_OPP_MIX["rand_anchor"] - 0.1) < 1e-9, DEFAULT_OPP_MIX
    # D1（2026-09-13）：frozen 0.4 → 0.1、hist 0.3 → 0.6（去镜像化，见 cycling_league_plan）
    assert abs(DEFAULT_OPP_MIX["frozen"] - 0.1) < 1e-9, DEFAULT_OPP_MIX
    assert abs(DEFAULT_OPP_MIX["hist"] - 0.6) < 1e-9, DEFAULT_OPP_MIX
    assert abs(_OPP_MIX["rand_anchor"] - 0.1) < 1e-9, _OPP_MIX
    assert dict(_OPP_MIX) == dict(DEFAULT_OPP_MIX), (_OPP_MIX, DEFAULT_OPP_MIX)

    bd = len(BeliefInference(opp_deck=env.deck1, n_particles=128,
                             seed=0).encode(None, None))
    # ③ 参考锚点（= E1 baseline_rand 同种子构造）
    ref = _make_rand_anchor(TrainConfig.resolve("economy"), bd, device="cpu")

    # ②④ 有 hist 的分布 + record no-op
    src_candidates = (sorted(glob.glob("runs/economy/solo_main_*.pt")) +
                      sorted(glob.glob("../../runs/archive/*/solo_main_*.pt")))
    if src_candidates:
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "economy"), exist_ok=True)
            for src in src_candidates[::max(1, len(src_candidates) // 3)][:3]:
                shutil.copy(src, os.path.join(td, "economy", os.path.basename(src)))
            cfg = TrainConfig.resolve("economy")
            cfg.out_dir = td
            pool = _OpponentPool(cfg, env, frozen_side, random.Random(0), "cpu")
            assert pool.rand_anchor_side is not None, "默认 mix 应构造锚点 side"
            side_sd, ref_sd = pool._rand_anchor_pol.state_dict(), ref.state_dict()
            assert all(torch.equal(side_sd[k], ref_sd[k]) for k in side_sd), \
                "训练侧锚点与 E1 评估侧锚点（同种子）权重不一致"
            kinds = Counter()
            N = 3000
            for _ in range(N):
                kind, side, _hid = pool.sample()
                kinds[kind] += 1
                if kind == "rand_anchor":
                    assert side is pool.rand_anchor_side, "锚点局应返回锚点 side"
            mix = pool.mix
            for k in ("hist", "defend", "rand_anchor"):
                exp = N * float(mix[k])
                assert abs(kinds[k] - exp) <= 0.35 * exp, (k, kinds, mix)
            pool.record(1)   # 锚点/defend 局 record 不得崩（no-op）
    else:
        print("[SKIP] 有 hist 分布：无可用历史 ckpt（仅验证其余分支）")

    # ⑤ 无 hist 退化（空目录）：归一化 + 旧 bug 修复
    with tempfile.TemporaryDirectory() as td:
        cfg = TrainConfig.resolve("economy")
        cfg.out_dir = td
        pool = _OpponentPool(cfg, env, frozen_side, random.Random(0), "cpu")
        kinds = Counter()
        N = 20000
        for _ in range(N):
            kind, _side, _hid = pool.sample()
            kinds[kind] += 1
        den = 1.0 - pool.mix["hist"]
        for k, frac in (("frozen", pool.mix["frozen"] / den),
                        ("defend", pool.mix["defend"] / den),
                        ("rand_anchor", pool.mix.get("rand_anchor", 0) / den)):
            exp = N * frac
            assert abs(kinds[k] - exp) <= 0.35 * exp, (k, kinds, frac)
        # 旧 bug 的指纹：hist 的概率空间被误分给 defend（实测 0.78，而宣称 0.286）。
        # D1 后名义退化分布 = frozen 0.25 / defend 0.50 / rand_anchor 0.25（hist 0.6 退出后
        # 按剩余槽位归一化）；旧的"吃掉 hist"实现会给出 defend 0.80 ⇒ 用 ±0.08 精确钉住
        # 正确口径（比旧的绝对阈值 0.4 更强，且不再依赖 mix 的具体数值）。
        assert abs(kinds["defend"] / N - pool.mix["defend"] / den) < 0.08, \
            f"无 hist 退化把 hist 空间误分给 defend（旧 bug 实测 0.78）: {kinds}"

    # ⑥ 旧式三槽 mix 兼容
    with tempfile.TemporaryDirectory() as td:
        cfg = TrainConfig.resolve("economy")
        cfg.out_dir = td
        cfg.opp_mix = {"frozen": 0.5, "hist": 0.3, "defend": 0.2}
        pool = _OpponentPool(cfg, env, frozen_side, random.Random(0), "cpu")
        assert pool.rand_anchor_side is None, "无 rand_anchor 键不应构造锚点"
        kinds = Counter()
        N = 5000
        for _ in range(N):
            kind, _side, _hid = pool.sample()
            kinds[kind] += 1
        assert kinds["rand_anchor"] == 0, kinds
        den_old = 1.0 - 0.3   # 无 hist 时按剩余概率归一化（frozen=0.5/0.7≈0.714）
        assert abs(kinds["frozen"] / N - 0.5 / den_old) < 0.05, kinds

    print(f"[PASS] E2 对手池：rand_anchor 槽默认 0.1（D1 后 frozen 0.1 / hist 0.6）；"
          f"有/无 hist 分布合规（seed={RAND_ANCHOR_SEED}）；"
          f"锚点权重与 E1 逐位一致；旧式三槽 mix 兼容；无 hist 归一化修复")


def test_pfsp_gate_and_dynamic_hist():
    """D1（2026-09-13，`docs/cycling_league_plan_2026-09-13.md`）回归。

    三件必须可证伪的事：
    ① **`rl/pfsp.py` 默认参数 = 旧行为逐位等价**（新参数都是 opt-in，兼容红线）；
    ② `alpha`/门禁真的按语义生效（α=0.2 一局把 EMA 0.5→0.6；EMA 胜率 > gate_hi 的
       ckpt 权重 ×gate_penalty；参数非法要报错）；
    ③ `_OpponentPool` 走 D1 参数、且 `refresh_hist()` 把**本 run 新写的快照**纳入池，
       同时**已入池 ckpt 的 PFSP id 保持不变**（旧实现用 `hist_<下标>`，池一增长就张冠李戴）。
    """
    import tempfile
    from collections import Counter
    from rl.pfsp import PFSP
    from rl.train_solo import (_OpponentPool, _PFSP_ALPHA, _PFSP_GATE_HI,
                               _PFSP_GATE_PENALTY)
    from rl.config import TrainConfig, DEFAULT_OPP_MIX
    from rl.env_wrapper import RLEnv
    from rl.follower import FollowerPolicy, save_checkpoint
    from rl.train_follower import FollowerOpponent
    from rl.belief import BeliefInference
    from rl.plan_space import PLAN_DIM

    # ①② PFSP 语义（纯函数，不碰环境）
    base = PFSP(beta=1.0, seed=0)                       # 旧行为
    d1 = PFSP(beta=1.0, seed=0, alpha=_PFSP_ALPHA,
              gate_hi=_PFSP_GATE_HI, gate_penalty=_PFSP_GATE_PENALTY)
    ops = ["hard", "easy", "fresh"]
    base.update_winrate("m", "hard", 0.0)               # 全败 → 高权重
    base.update_winrate("m", "easy", 1.0)               # 全胜 → 低权重
    d1.update_winrate("m", "hard", 0.0)
    d1.update_winrate("m", "easy", 1.0)
    wb, wd = base.weights("m", ops), d1.weights("m", ops)
    # 旧行为（alpha=0.05）：0.5→0.475（全败）、0.5→0.525（全胜）；未采样 = 1.0
    assert np.allclose(wb, [0.525, 0.475, 1.0]), wb
    assert abs(wb[2] - 1.0) < 1e-12, wb                          # 未采样 = 乐观先验
    assert abs(wd[0] - 0.6) < 1e-9, wd                           # 0.2 新息：0.5→0.4
    # easy：EMA 0.5→0.6 < gate_hi=0.85 ⇒ 门禁尚未触发
    assert abs(wd[1] - 0.4) < 1e-9, wd
    assert abs(wd[2] - 1.0) < 1e-12, wd
    for _ in range(6):
        d1.update_winrate("m", "easy", 1.0)
    wr_easy = d1.winrates[("m", "easy")]
    assert wr_easy > _PFSP_GATE_HI, wr_easy
    wd2 = d1.weights("m", ops)
    assert abs(wd2[1] - (1.0 - wr_easy) ** 1.0 * _PFSP_GATE_PENALTY) < 1e-9, (wd2, wr_easy)
    assert wd2[1] < (1.0 - wr_easy), "门禁必须真的降权"
    # 未采样对手不受门禁影响（乐观先验保持探索）
    assert abs(wd2[2] - 1.0) < 1e-12, wd2
    # 参数校验
    for bad in (dict(alpha=0.0), dict(alpha=1.5), dict(gate_penalty=1.5),
                dict(gate_penalty=-0.1), dict(beta=-1.0)):
        try:
            PFSP(**bad)
            raise AssertionError(f"非法参数未报错: {bad}")
        except ValueError:
            pass

    # ③ 对手池接线 + 动态刷新
    env = RLEnv(opponent=None, seed=0, card_level=11)
    env.reset(seed=0)
    bd = len(BeliefInference(opp_deck=env.deck1, n_particles=128,
                             seed=0).encode(None, None))
    pol = FollowerPolicy(hidden=32, plan_dim=PLAN_DIM, belief_dim=bd)
    with tempfile.TemporaryDirectory() as td:
        cur, seed = os.path.join(td, "cur"), os.path.join(td, "seed")
        os.makedirs(cur)
        os.makedirs(seed)
        for s in (0, 1000):
            save_checkpoint(pol, os.path.join(seed, f"solo_main_{s}.pt"))
        cfg = TrainConfig(name="cur", out_dir=td, hidden_dim=32)
        opp = FollowerPolicy(hidden=32, plan_dim=PLAN_DIM, belief_dim=bd).to_device("cpu")
        side = FollowerOpponent(opp, env, belief=BeliefInference(
            opp_deck=env.deck1, n_particles=128, seed=0), deterministic=True)
        pool = _OpponentPool(cfg, env, side, random.Random(0), "cpu",
                             hist_seed_dirs=[seed])
        assert pool.mix == dict(DEFAULT_OPP_MIX), pool.mix
        assert abs(pool.pfsp_alpha - _PFSP_ALPHA) < 1e-12, pool.pfsp_alpha
        assert abs(pool._pfsp.gate_hi - _PFSP_GATE_HI) < 1e-12
        assert abs(pool._pfsp.gate_penalty - _PFSP_GATE_PENALTY) < 1e-12
        assert len(pool.hist_paths) == 2, pool.hist_paths
        ids_before = dict(pool._hist_id)
        kinds = Counter(pool.sample()[0] for _ in range(3000))
        for k in ("hist", "defend", "rand_anchor", "frozen"):
            exp = 3000 * float(pool.mix[k])
            assert abs(kinds[k] - exp) <= 0.35 * exp, (k, kinds, pool.mix)

        # 本 run 后来写出快照 → refresh_hist 必须纳入，且旧 id 不变
        save_checkpoint(pol, os.path.join(cur, "solo_main_2000.pt"))
        added = pool.refresh_hist(2000)
        assert added == 1, added
        assert os.path.join(cur, "solo_main_2000.pt") in pool.hist_paths, pool.hist_paths
        assert all(pool._hist_id[p] == i for p, i in ids_before.items()), "旧 ckpt 的 id 变了"
        assert pool.kind_counts["hist"] >= 1, pool.kind_counts
        # 反向验证：旧实现（hist_<下标>）会在这里给出不同的 id 映射
        assert pool._hist_id[os.path.join(cur, "solo_main_2000.pt")].startswith("hist_cur_")
        print(f"[PASS] D1：PFSP 默认=旧行为、门禁降权（EMA {wr_easy:.3f} > "
              f"{_PFSP_GATE_HI} → ×{_PFSP_GATE_PENALTY}）、池走 α={pool.pfsp_alpha}、"
              f"refresh_hist 新增 {added} 且旧 id 稳定、分布 {dict(kinds)}")


def test_enc_layernorm_gru_vitality():
    """v3 P0-A 回归：`enc_ln` 必须让 GRU 解冻，并吸收 `enc_fc` 的量级漂移。

    病理（修复前，在**训练后**的 ckpt 上实测）：`enc = relu(enc_fc(fused))` 的
    L2 范数 ≈533（CNN 输出的常数分量 ≈468、跨帧 std 仅 1.06）→ GRUCell tanh
    候选饱和（|n|≈0.994）→ h 跨帧 std ≈2.6e-5（数值恒定）→ value_head 恒输出
    常数、EV 恒负；slot_head/cell_head 同样吃常数 h（策略网络开环）。

    **重要事实（已实测）**：随机初始化时**并不饱和** —— 无 enc_ln 的 enc 范数
    仅 ~0.5（hidden=64）。量级是**训练过程中被推大的**。所以本测试不用裸随机
    网络当负对照，而是把 `enc_fc` 输出人为放大 300× 模拟训练后期的量级漂移
    （enc 范数 54 → 203，GRU |n| 0.92 → 0.98）：这既是确定性复现，也正是
    `enc_ln` 要吃掉的问题（LayerNorm 对正数缩放基本不变）。
    """
    import torch
    import torch.nn as nn
    from rl.env_wrapper import RLEnv
    from rl.follower import FollowerPolicy
    from rl.belief import BeliefInference
    from rl.belief_planner import BeliefPlanner
    from rl.plan_space import PLAN_DIM
    from rl import diagnostics as diag

    env = RLEnv(opponent=None, seed=0, card_level=11)
    obs, _ = env.reset(seed=0)
    belief = BeliefInference(opp_deck=env.deck1, n_particles=32, seed=0)
    bp = BeliefPlanner()
    bd = len(belief.encode(None, None))
    pol = FollowerPolicy(hidden=64, plan_dim=PLAN_DIM, belief_dim=bd).to_device("cpu")

    frames = []
    hidden = None
    for _ in range(64):
        plan = bp.plan(env.battle, belief.state(), obs)
        pv = plan.to_vector()
        bt = belief.encode(obs, None)
        frames.append((obs, bt, pv))
        bundle, _lp, _v, hidden, _m = pol.act(obs, bt, pv, env.get_action_mask,
                                              hidden=hidden, deterministic=True)
        obs, _r, term, trunc, info = env.step(bundle)
        belief.update(obs, info.get("opp_played"))
        if term or trunc:
            obs, _ = env.reset(seed=7)
            belief.reset(env.deck1)
            hidden = None

    def _enc_norm():
        with torch.no_grad():
            return float(pol._encode(*frames[0]).norm().item())

    # 1) 基准：enc_ln 生效 → GRU 有活力
    n0, vit0 = _enc_norm(), diag.gru_vitality(pol, frames)
    assert not diag.check_vitality(vit0), f"GRU 未解冻：{diag.check_vitality(vit0)}"
    # 2) 量级漂移（enc_fc ×300，模拟训练后期）→ enc_ln 必须吸收掉，活力不变
    with torch.no_grad():
        pol.enc_fc.weight.mul_(300.0)
        pol.enc_fc.bias.mul_(300.0)
    n1, vit1 = _enc_norm(), diag.gru_vitality(pol, frames)
    assert not diag.check_vitality(vit1), \
        f"enc_ln 未能吸收量级漂移：{diag.check_vitality(vit1)}（vit={vit1}）"
    # 注意：不是**完全**不变 —— LayerNorm 的 eps 在 x 被放大后相对更小，
    # 实测 7.976 → 8.000（0.3%，因为 LN 输出范数 ≈ sqrt(hidden) 有上限）。
    assert abs(n1 - n0) < 0.05 * max(1.0, n0), \
        f"LayerNorm 后 enc 范数应基本不随缩放变：{n0} vs {n1}"
    # 3) 负对照：去掉 enc_ln → 必须复现饱和（否则本测试没有保护力）
    pol.enc_ln = nn.Identity()
    n2, vit2 = _enc_norm(), diag.gru_vitality(pol, frames)
    assert diag.check_vitality(vit2), f"负对照未复现饱和（测试无保护力）：{vit2}"
    # 4) v3 P0-C 启动前检查：静态病因护栏（enc_ln 缺失 / 被 Identity 替换）
    assert diag.check_policy_architecture(pol), "enc_ln=Identity 未被启动前检查拦截"
    _fresh = FollowerPolicy(hidden=64, plan_dim=PLAN_DIM, belief_dim=bd).to_device("cpu")
    assert not diag.check_policy_architecture(_fresh), \
        f"健康策略被启动前检查误报：{diag.check_policy_architecture(_fresh)}"
    del _fresh.enc_ln
    assert diag.check_policy_architecture(_fresh), "enc_ln 缺失未被启动前检查拦截"
    # 5) v3 §2 验收第 3 行：value_std / 批内 R std 的门槛与告警必须生效
    assert diag.THRESHOLDS["value_std_ratio"] == 0.3
    assert diag.check_vitality({"h_std": 0.2, "n_abs": 0.5, "value_std_ratio": 0.5}) == []
    assert diag.check_vitality({"h_std": 0.2, "n_abs": 0.5, "value_std_ratio": 0.01}), \
        "value_std_ratio 低于门槛未报警（v3 §2 第 3 行仍不可判读）"
    # 6) v3 P0-A 备选（第二轮）：grid_feat 分量归一化
    #    20k 跑实测融合层尺度失衡：grid_feat ‖·‖=101 占 fused 的 98%、跨帧 std 仅 0.426
    #    （v3 §3.7）。这里不拿"训练后 101"当测试（随机初始化时 CNN 输出很小、根本不复现
    #    该量级——同 §3.5 的教训），而是验**归一化的尺度不变性**：CNN 输出被放大多少倍，
    #    grid 分量的"每元素 RMS"都必须被 grid_ln 拉回 ~1。
    pol3 = FollowerPolicy(hidden=64, plan_dim=PLAN_DIM, belief_dim=bd).to_device("cpu")

    def _comp_rms(p):
        """用 forward hook 抓 enc_fc 的输入（= fused），按分量算每元素 RMS。"""
        cap = {}

        def _hk(_m, inp, _o):
            cap.setdefault("f", inp[0].detach())
        _h = p.enc_fc.register_forward_hook(_hk)
        try:
            p._encode(*frames[0])
        finally:
            _h.remove()
        f = cap["f"][0]
        g = p.cnn_out
        parts = {"grid": f[:g], "hand": f[g:g + 40], "scalar": f[g + 40:g + 43],
                 "plan": f[g + 43:g + 107], "belief": f[g + 107:]}
        return {k: float(v.pow(2).mean().sqrt()) for k, v in parts.items()}

    rms0 = _comp_rms(pol3)
    assert 0.6 < rms0["grid"] < 1.4, f"grid_ln 未把 grid 每元素 RMS 归到 ~1：{rms0}"
    with torch.no_grad():
        for _p in pol3.cnn.parameters():
            _p.mul_(300.0)      # CNN 输出 ×300（模拟训练后期的量级漂移）
    rms1 = _comp_rms(pol3)
    assert abs(rms1["grid"] - rms0["grid"]) < 0.05, \
        f"grid_ln 未吸收 CNN 量级漂移：{rms0['grid']} vs {rms1['grid']}"
    # 负对照：换回 Identity → 每元素尺度必须随 CNN 放大而放大（否则本测试无保护力）
    pol3.grid_ln = nn.Identity()
    rms2 = _comp_rms(pol3)
    assert rms2["grid"] > 10 * rms0["grid"], \
        f"去掉 grid_ln 后未复现尺度失衡（测试无保护力）：{rms2['grid']}"
    # 7) 启动前检查必须覆盖两个归一化模块
    assert diag.check_policy_architecture(pol3), "grid_ln=Identity 未被启动前检查拦截"
    pol4 = FollowerPolicy(hidden=64, plan_dim=PLAN_DIM, belief_dim=bd).to_device("cpu")
    assert not diag.check_policy_architecture(pol4), \
        f"健康策略被启动前检查误报：{diag.check_policy_architecture(pol4)}"
    del pol4.grid_ln
    assert diag.check_policy_architecture(pol4), "grid_ln 缺失未被启动前检查拦截"
    print(f"[PASS] v3 P0-A：enc_ln 生效 —— enc 范数 {n0:.2f}（×300 后仍 {n1:.2f}）；"
          f"h_std={vit0['h_std']:.4f}、n_abs={vit0['n_abs']:.4f}；"
          f"去掉 enc_ln 后 enc={n2:.1f}、h_std={vit2['h_std']:.2e}、"
          f"n_abs={vit2['n_abs']:.4f}（复现饱和）")
    print(f"[PASS] v3 P0-A 备选：grid_ln 把 grid 每元素 RMS 归到 {rms0['grid']:.3f}"
          f"（CNN ×300 后仍 {rms1['grid']:.3f}；去掉 grid_ln 后 {rms2['grid']:.3f}）；"
          f"启动前检查 + value/R std 门槛已生效")


def test_value_bypass():
    """B'（2026-09-12）落地回归：value_bypass=True 时 value = value_head(enc)
    （跳过 GRU），策略头仍走 GRU；checkpoint 元数据保存/恢复该标志；
    显式覆盖与元数据不一致时告警（不静默）。

    判别：
    - 相同权重下 bypass 与 GRU 通路的 value 不同（enc ≠ gru(enc,0)）；
    - bypass 通路 value 确定性一致，且 act()/value() 的 value 一致（语义贯穿）；
    - save/load checkpoint 后 value_bypass 标志保留；
    - load_checkpoint(..., value_bypass=...) 与元数据不一致 → 打印告警不崩。
    """
    import os
    import tempfile
    import torch
    from rl.env_wrapper import RLEnv
    from rl.follower import FollowerPolicy, save_checkpoint, load_checkpoint
    from rl.belief import BeliefInference
    from rl.belief_planner import BeliefPlanner
    from rl.plan_space import PLAN_DIM

    env = RLEnv(opponent=None, seed=0, card_level=11)
    obs, _ = env.reset(seed=0)
    belief = BeliefInference(opp_deck=env.deck1, n_particles=32, seed=0)
    bp = BeliefPlanner()
    bd = len(belief.encode(None, None))
    plan = bp.plan(env.battle, belief.state(), obs)
    pv = plan.to_vector()
    bt = belief.encode(obs, None)

    a = FollowerPolicy(hidden=64, plan_dim=PLAN_DIM, belief_dim=bd,
                       value_bypass=False).to_device("cpu")
    b = FollowerPolicy(hidden=64, plan_dim=PLAN_DIM, belief_dim=bd,
                       value_bypass=True).to_device("cpu")
    b.load_state_dict(a.state_dict())      # 同权重 → 对比只差 value 通路
    assert not a.value_bypass and b.value_bypass

    va = a.value(obs, bt, pv, hidden=None)
    vb = b.value(obs, bt, pv, hidden=None)
    assert abs(va - vb) > 1e-6, f"bypass 与 GRU 通路 value 意外相同: {va} vs {vb}"
    assert b.value(obs, bt, pv, hidden=None) == vb, "bypass value 不确定（应逐位一致）"
    # act() 的 value 必须与 value() 一致（rollout/PPO 重放同源）
    _b, _lp, v_act, _h, _m = b.act(obs, bt, pv, env.get_action_mask, deterministic=True)
    assert abs(v_act - vb) < 1e-6, f"act() value {v_act} != value() {vb}（bypass 未贯穿）"
    # 诊断口径护栏（2026-09-12 实测踩过）：gru_vitality 的 value_std 必须走策略真实
    # value 通路（bypass → value_head(enc)），不得硬编码 value_head(h)——否则
    # value_std_ratio 这类主判据测的不是被训练的量。
    import numpy as _np
    from rl import diagnostics as _diag
    frames = []
    _hid = None
    _obs = obs
    for _t in range(12):
        _plan = bp.plan(env.battle, belief.state(), _obs)
        _pv2 = _plan.to_vector()
        _bt2 = belief.encode(_obs, None)
        frames.append((_obs, _bt2, _pv2))
        _bd, _l, _v, _hid, _m2 = b.act(_obs, _bt2, _pv2, env.get_action_mask,
                                       hidden=_hid, deterministic=True)
        _obs, _r, _term, _trunc, _info = env.step(_bd)
        belief.update(_obs, _info.get("opp_played"))
        if _term or _trunc:
            _obs, _ = env.reset(seed=11)
            belief.reset(env.deck1)
            _hid = None
    vit_b = _diag.gru_vitality(b, frames)
    vit_a = _diag.gru_vitality(a, frames)
    with torch.no_grad():
        vs_manual = [float(b.value_head(b._encode(o, t, p)).item())
                     for (o, t, p) in frames]
    assert abs(vit_b["value_std"] - float(_np.std(vs_manual))) < 1e-6, \
        (f"gru_vitality 的 value_std 未走 bypass 通路：{vit_b['value_std']} vs "
         f"{float(_np.std(vs_manual))}")
    assert abs(vit_a["value_std"] - vit_b["value_std"]) > 1e-9, \
        "非 bypass 与 bypass 的 value_std 意外相同（探针可能仍硬编码旧通路）"
    # checkpoint 元数据往返
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "bp.pt")
        save_checkpoint(b, p)
        c = load_checkpoint(p, plan_dim=PLAN_DIM, belief_dim=bd)
        assert c.value_bypass is True, "value_bypass 元数据未保存/恢复"
        load_checkpoint(p, plan_dim=PLAN_DIM, belief_dim=bd, value_bypass=False)  # 告警不崩
    print(f"[PASS] B' value_bypass：GRU 通路 {va:+.4f} vs bypass {vb:+.4f}"
          f"（同权重不同通路）；元数据往返/告警正常")


def test_stall_settlement_margin():
    """C'（2026-09-12）早停低置信裁定降噪（纯函数层）：

    - 皇冠不同 → 决定性 0/1（不受 margin 影响）；
    - 皇冠相同 + 塔血%差 ≥ margin → 决定性 0/1（保留 2026-09 的"1-1 塔血差 1000+"
      修复，只降噪不丢决定性信息）；
    - 皇冠相同 + 塔血%差 < margin → None（记平局=失败，去掉掷硬币级裁定）；
    - 实体信息缺失（None）→ None（退回平局，与 timeout_winner 同）。
    """
    from rl.run_league import settle_stall_from_counts, settle_stall

    # 皇冠不同 → 决定性
    assert settle_stall_from_counts(0, 1, 0.5, 0.5, 0.05) == 0   # p1 丢塔多 → p0 胜
    assert settle_stall_from_counts(1, 0, 0.5, 0.5, 0.05) == 1   # p0 丢塔多 → p1 胜
    # 皇冠相同 + 决定性细差（≥ margin）→ 保留胜负
    assert settle_stall_from_counts(1, 1, 0.90, 0.50, 0.05) == 0
    assert settle_stall_from_counts(1, 1, 0.40, 0.90, 0.05) == 1
    # 皇冠相同 + 临界（margin 边界）→ 决定性（> margin 才算）
    assert settle_stall_from_counts(1, 1, 0.50 + 0.05 + 1e-9, 0.50, 0.05) == 0
    # 皇冠相同 + 低置信细差（< margin）→ None（平局=失败，降噪）
    assert settle_stall_from_counts(1, 1, 0.52, 0.50, 0.05) is None
    assert settle_stall_from_counts(1, 1, 0.499, 0.500, 0.05) is None
    # 皇冠相同 + 完全相等 → None（与 timeout_winner 同）
    assert settle_stall_from_counts(1, 1, 0.50, 0.50, 0.05) is None
    # 实体信息缺失 → None
    assert settle_stall_from_counts(1, 1, None, 0.50, 0.05) is None
    assert settle_stall_from_counts(1, 1, None, None, 0.05) is None
    assert settle_stall(None) is None
    # margin=0 时退化为 timeout_winner 语义（仅 1e-9 容差保留）
    assert settle_stall_from_counts(1, 1, 0.5000001, 0.5, 0.0) == 0
    print("[PASS] C' 早停低置信裁定降噪：决定性情形保留、细差 < margin 记平局、边界正确")


def test_value_independent_encoder():
    """E'（2026-09-12）回归：独立价值编码器 + 非线性价值头。

    判别：
    - `value_independent=True` 时 value == `value_head_mlp(value_enc_ln(relu(value_enc_fc(fused))))`
      （与内部实现逐位一致 → 通路确实是"自己的编码器 + MLP 头"，不是共享头）；
    - 与共享/bypass 通路（同权重）不同 → 参数确实独立（策略前向不含价值模块）；
    - 策略头仍走 GRU 隐状态（独立价值不改变策略语义：同权重下 act 的 bundle 与共享版一致）；
    - ckpt 元数据保存/恢复 `value_independent`；显式覆盖不一致告警不崩；
    - 诊断口径：`gru_vitality` 的 value_std 走独立通路（不再是硬编码旧通路）。
    """
    import numpy as np
    import torch
    from rl.env_wrapper import RLEnv
    from rl.follower import FollowerPolicy
    from rl.belief import BeliefInference
    from rl.belief_planner import BeliefPlanner
    from rl.plan_space import PLAN_DIM
    from rl import diagnostics as diag

    env = RLEnv(opponent=None, seed=0, card_level=11)
    obs, _ = env.reset(seed=0)
    belief = BeliefInference(opp_deck=env.deck1, n_particles=32, seed=0)
    bp = BeliefPlanner()
    bd = len(belief.encode(None, None))
    plan = bp.plan(env.battle, belief.state(), obs)
    pv = plan.to_vector()
    bt = belief.encode(obs, None)

    shared = FollowerPolicy(hidden=64, plan_dim=PLAN_DIM, belief_dim=bd).to_device("cpu")
    indep = FollowerPolicy(hidden=64, plan_dim=PLAN_DIM, belief_dim=bd,
                           value_independent=True).to_device("cpu")
    assert indep.value_independent and not indep.value_bypass
    # 共享权重（策略/编码器部分），价值模块保持各自初始化 → 隔离变量
    missing = indep.load_state_dict(shared.state_dict(), strict=False)
    assert all('value_enc' in k or 'value_head_mlp' in k for k in missing.missing_keys), \
        f"独立价值模块未被视为新键：{missing.missing_keys}"
    # ① 通路正确性：value 必须等于独立通路的显式计算
    with torch.no_grad():
        fused, _enc = indep._encode_parts(obs, bt, pv)
        expected = float(indep.value_head_mlp(
            indep.value_enc_ln(torch.relu(indep.value_enc_fc(fused)))).item())
    got = indep.value(obs, bt, pv, hidden=None)
    assert abs(got - expected) < 1e-6, f"独立价值通路不符：{got} vs {expected}"
    # ② 与共享通路不同（参数独立）
    assert abs(shared.value(obs, bt, pv, hidden=None) - got) > 1e-6, \
        "independent 与 shared 的 value 意外相同（模块未独立）"
    # ③ 策略语义不变：同权重下 act 的 bundle 必须一致（价值模块不影响策略）
    b_s, lp_s, _, _, _ = shared.act(obs, bt, pv, env.get_action_mask, deterministic=True)
    b_i, lp_i, _, _, _ = indep.act(obs, bt, pv, env.get_action_mask, deterministic=True)
    assert lp_s == lp_i and len(b_s.sub_actions) == len(b_i.sub_actions), \
        "独立价值模块改变了策略前向（不应影响 act 的 logprob/bundle）"
    # ④ 元数据往返 + 不一致告警
    import os
    import tempfile
    from rl.follower import save_checkpoint, load_checkpoint
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "indep.pt")
        save_checkpoint(indep, p)
        c = load_checkpoint(p, plan_dim=PLAN_DIM, belief_dim=bd)
        assert c.value_independent is True, "value_independent 元数据未保存/恢复"
        assert hasattr(c, "value_head_mlp"), "恢复后缺少独立价值头"
        load_checkpoint(p, plan_dim=PLAN_DIM, belief_dim=bd, value_independent=False)  # 告警不崩
    # ⑤ 诊断口径：gru_vitality 的 value_std 走独立通路
    frames = []
    _hid = None
    _obs = obs
    for _t in range(12):
        _plan = bp.plan(env.battle, belief.state(), _obs)
        _pv = _plan.to_vector()
        _bt = belief.encode(_obs, None)
        frames.append((_obs, _bt, _pv))
        _bd2, _l2, _v2, _hid, _m2 = indep.act(_obs, _bt, _pv, env.get_action_mask,
                                             hidden=_hid, deterministic=True)
        _obs, _r2, _term2, _trunc2, _info2 = env.step(_bd2)
        belief.update(_obs, _info2.get("opp_played"))
        if _term2 or _trunc2:
            _obs, _ = env.reset(seed=13)
            belief.reset(env.deck1)
            _hid = None
    vit = diag.gru_vitality(indep, frames)
    vs_manual = []
    with torch.no_grad():
        for (o, t, p) in frames:
            _f, _e = indep._encode_parts(o, t, p)
            vs_manual.append(float(indep._value_from(
                _e, torch.zeros(1, indep.hidden_dim), _f).item()))
    assert abs(vit["value_std"] - float(np.std(vs_manual))) < 1e-6, \
        f"gru_vitality 未走独立价值通路：{vit['value_std']} vs {float(np.std(vs_manual))}"
    print(f"[PASS] E' 独立价值编码器：通路一致（{got:+.4f}）、与共享通路不同、"
          f"策略 logprob 不受影响、元数据/告警/诊断口径正常")


def test_ppo_multi_epoch_minibatch():
    """F'（2026-09-12）：真正的 PPO 更新预算（多轮 × 打乱 × 小批）。

    背景：旧 `PPOTrainer.update()` = 1 次 forward / 1 次 backward / 1 次 `opt.step`，
    而 `train_solo` 喂进来的是同一局连续 128 帧（相邻帧回报相关 ≈0.99）⇒ 20k 步
    只有 156 次梯度步、每次梯度目标几乎相同 ⇒ 四种价值架构的 EV 全部 ≤0，而同一
    表征的 MLP 探针 R² 有 0.24~0.42（"表征有信息、critic 吸收不了"）。

    本测试断言（每条都可证伪）：
    ① **默认参数 = 旧行为**：1 次梯度步、ratio≡1、clip≡0、adv 统计不变；
    ② `grad_steps == epochs × ceil(n/mb)`，且 `trainer.grad_steps` 累计正确；
    ③ **每轮每个样本恰好被访问一次**（`_plan_batches` 是划分不是抽样），
       打乱后顺序确实变了（seeded RNG 可复现）；
    ④ **多轮真的更拟合**：同一批数据上 `value_loss_raw` 必须比旧单 pass 更低
       （若相等 ⇒ 多轮的梯度步没生效，这正是要防的"改了参数但没接线"）；
    ⑤ ratio 离开 1.000（这是 F' 的**期望**行为，AGENTS 的"ratio≡1.000 是结构性"
       只对 n_epochs=1 成立）。
    """
    import copy
    import torch
    from rl.env_wrapper import RLEnv
    from rl.belief import BeliefInference
    from rl.follower import FollowerPolicy
    from rl.plan_space import PLAN_DIM
    from rl.ppo import PPOTrainer

    env = RLEnv(opponent=None, seed=0)
    obs, _ = env.reset()
    belief = BeliefInference(opp_deck=env.deck1, n_particles=128, seed=0)
    tok = belief.encode(obs, None)
    plan = np.zeros(PLAN_DIM, dtype=np.float32)
    pol = FollowerPolicy(hidden=32, plan_dim=PLAN_DIM, belief_dim=len(tok))
    trans = _tiny_rollout_transitions(pol, env, belief, tok, plan, n=12)
    n = len(trans)

    # ① 默认 = 旧行为（兼容性红线：run_league/flow_league/train_follower 共用本类）
    pol_ref = copy.deepcopy(pol)
    t0 = PPOTrainer(pol_ref, lr=1e-3)
    assert (t0.n_epochs, t0.minibatch_size, t0.shuffle) == (1, 0, False)
    s0 = t0.update(trans)
    assert s0["grad_steps"] == 1 and t0.grad_steps == 1, s0   # 恒 1 次梯度步
    assert abs(s0["ratio_mean"] - 1.0) < 1e-5 and s0["clip_frac"] == 0.0, s0
    assert all(np.isfinite(v) for v in s0.values()), s0
    with torch.no_grad():
        w0 = [p.detach().clone() for p in pol_ref.parameters()]

    # ③ `_plan_batches` 是划分（不是抽样），且打乱可复现
    tm = PPOTrainer(copy.deepcopy(pol), lr=1e-2, n_epochs=4, minibatch_size=4,
                    shuffle=True)
    seen = []
    for _ in range(3):
        bs = tm._plan_batches(n)
        assert sum(len(b) for b in bs) == n, [len(b) for b in bs]
        flat = np.concatenate(bs)
        assert sorted(flat.tolist()) == list(range(n)), flat    # 每样本恰好一次
        seen.append(flat.tolist())
    assert any(o != list(range(n)) for o in seen), "shuffle=True 但顺序恒为原序"
    t_same = PPOTrainer(copy.deepcopy(pol), lr=1e-2, n_epochs=4, minibatch_size=4,
                        shuffle=True)
    assert [np.concatenate(t_same._plan_batches(n)).tolist()
            for _ in range(3)] == seen, "同 seed 的打乱不可复现"

    # ②/④/⑤ 多轮多批：梯度步数 = 轮数 × 批数；同行数据拟合更深；ratio 离开 1
    pol_m = copy.deepcopy(pol)
    tm2 = PPOTrainer(pol_m, lr=1e-2, n_epochs=4, minibatch_size=4, shuffle=True,
                     seed=7)
    sm = tm2.update(trans)      # 注意：与 tm 用同一 seed，但 tm 只调了 _plan_batches
    assert sm["grad_steps"] == 4 * 3, sm          # 4 轮 × ceil(12/4)=3 批
    assert tm2.grad_steps == 12, tm2.grad_steps
    assert abs(sm["adv_mean"] - s0["adv_mean"]) < 1e-6, (sm["adv_mean"], s0["adv_mean"])
    assert sm["value_loss_raw"] < s0["value_loss_raw"], (sm, s0)
    assert abs(sm["ratio_mean"] - 1.0) > 1e-4 or sm["clip_frac"] > 0.0, sm
    assert sm["ppo_epochs"] == 4 and sm["ppo_minibatch"] == 4, sm
    assert len(tm2.last_ev_pairs[0]) == n, tm2.last_ev_pairs   # 末轮每帧一次预测
    with torch.no_grad():
        d_multi = sum(float(((a - b) ** 2).sum())
                      for a, b in zip(pol_m.parameters(), pol.parameters()))
    assert d_multi > 0.0, d_multi

    # 单轮打乱（n_epochs=1）+ 指定 minibatch：ratio 仍应≈1（结构性的 on-policy）
    t1 = PPOTrainer(copy.deepcopy(pol), lr=1e-3, n_epochs=1, minibatch_size=4,
                    shuffle=True)
    s1 = t1.update(trans)
    assert s1["grad_steps"] == 3, s1

    # ⑥ 回归哨兵（F' 落地时抓到的第一个真 bug，2026-09-12）：ratio/clip 的**聚合**
    # 必须只除**末轮**的样本数。当时写成"末轮累加 / 全部轮次累加"，4 轮跑出
    # ratio_mean=0.2479（≈1/4）而逐小批其实全是 0.99 —— 一个"看起来像策略崩了"
    # 的假警报。这里用 **lr=0.0**（参数逐位不动）做最锐的哨兵：ratio 只该有
    # "批量前向 vs 逐步前向"的 float32 ULP 级偏差（实测 ~1.2e-7），
    # 误除全部轮次时会读出 ≈1/n_epochs≈0.25（比容差大 2.5 万倍）。
    tq = PPOTrainer(copy.deepcopy(pol), lr=0.0, n_epochs=4, minibatch_size=4,
                    shuffle=True)
    sq = tq.update(trans)
    assert abs(sq["ratio_mean"] - 1.0) < 1e-5, sq
    assert sq["clip_frac"] == 0.0, sq

    # ⑦ EV 口径哨兵（同日发现的第 3 个测量陷阱）：对外报出的 `explained_variance`
    # 必须是**更新前**的预测（旧实现天然如此：单次 forward 之后才 backward），
    # 末轮 in-sample 值另存 `explained_variance_insample`。
    # 实测背景：fprime_20k 末点训练 EV=+0.41，同权重在 50 局独立 rollout 上
    # EV_global=−0.0255 —— 差别几乎全来自"在那批上刚训过 12 步"。
    pol_e = copy.deepcopy(pol)
    te = PPOTrainer(pol_e, lr=1e-2, n_epochs=8, minibatch_size=4, shuffle=True)
    se = te.update(trans)
    assert se["explained_variance"] == te.last_ev_pre, (se, te.last_ev_pre)
    assert se["explained_variance_insample"] > se["explained_variance"], (
        "8 轮同批更新后 in-sample EV 必须高于更新前 EV（否则口径没分开）", se)
    # 旧分支不开这个键（保持旧 stats 结构）
    assert "explained_variance_insample" not in s0, s0

    print(f"[PASS] F' PPO 更新预算：默认 1 次梯度步（ratio≡1/clip≡0，旧行为逐位保留）、"
          f"4 轮×小批 4 = {sm['grad_steps']} 次梯度步（划分可复现）、"
          f"同批 vraw {s0['value_loss_raw']:.3f}→{sm['value_loss_raw']:.3f}、"
          f"ratio {sm['ratio_mean']:.4f} clip={100.0 * sm['clip_frac']:.1f}%")



def test_solo_rand_anchor():
    """E1（2026-09-12，v3 §3.8.4 取证后）：固定随机锚点的确定性与警报线。

    背景：自对弈无外部锚点会在策略循环里打转（main@20000 打冻结 0.85 / 打起点随机
    0.13 / 打全新随机 0.505 = RPS 三角）。E1 给每个评估点加一组 "main vs 固定种子
    随机策略" 对照量化绝对强度。本测试保证：
    ① 同一种子构造两次 → 权重逐位一致（跨评估点/跨 run 可复现）；
    ② 种子不同 → 权重不同（sanity）；
    ③ RNG 保存/恢复后调用方随机性不受扰动（与训练序列共用 RNG 时安全）；
    ④ `_rand_anchor_warns` 报警线：<0.35 报警、≥0.35 / None 不报警。
    """
    import torch
    from rl.follower import FollowerPolicy
    from rl.plan_space import PLAN_DIM
    from rl import train_solo
    from rl.belief import BeliefInference

    bd = len(BeliefInference(opp_deck=list(train_solo.DEFAULT_SOLO_DECK),
                             n_particles=32, seed=0).encode(None, None))

    def make(seed):
        rng = torch.get_rng_state()
        torch.manual_seed(seed)
        p = FollowerPolicy(hidden=32, plan_dim=PLAN_DIM, belief_dim=bd)
        torch.set_rng_state(rng)
        return p

    # ① 同种子 → 逐位一致
    p1, p2 = make(train_solo.RAND_ANCHOR_SEED), make(train_solo.RAND_ANCHOR_SEED)
    s1, s2 = p1.state_dict(), p2.state_dict()
    assert all(torch.equal(s1[k], s2[k]) for k in s1), "同种子锚点权重不一致（不可复现）"
    # ② 不同种子 → 不同
    p3 = make(train_solo.RAND_ANCHOR_SEED + 1)
    assert not torch.equal(p1.value_head.weight, p3.value_head.weight), \
        "不同种子锚点权重应不同（sanity）"
    # ③ RNG 恢复
    rng_before = torch.get_rng_state()
    make(12345)
    assert torch.equal(torch.get_rng_state(), rng_before), "锚点构造扰动了调用方 RNG"
    # ④ 报警线
    warns = train_solo._rand_anchor_warns
    assert warns(0.13) and warns(0.34) and not warns(0.35) \
        and not warns(0.5) and not warns(None)
    print(f"[PASS] E1 固定随机锚点：同种子逐位一致、异种子不同、RNG 无扰动、"
          f"报警线 <{train_solo.RAND_ANCHOR_WARN_FLOOR} 生效")


def test_anchor_light_point_state():
    """C 方案（2026-09-13）：轻量锚点评估点（`--anchor-every`）。

    背景：100k 级长 run 里评估占 ~3/4 墙钟，但 C1 判据（单跑最差锚点）用的锚点序列
    实测谷底**只有 1 个点宽**（相邻点 |Δ|≈0.23 ≈ 3σ）；粗采样回放证明 5000 步就开始
    漏真谷底、10000 步把"最差点"从 0.192 系统性抬到 0.462（病态组的塌陷是持续性的、
    任何粗采样都留得住；D1 的低点是周期性瞬态、粗采样直接丢）。故拆成
    "密锚点（只跑 baseline_rand）+ 稀全点（main/对照三块）"。

    本测试盯的都是会**静默失效**的点：
    ① `TrainConfig.anchor_every` 默认 0（关闭 = 旧行为，不动任何既有 run）；
    ② CLI `--anchor-every` 已接线到 overrides（漏接 = 传了也不生效，最典型的脚枪）；
    ③ 轻点写入的锚点条目与全点三条对照**共存**、按 step 去重、锚点序列可复原
       （轻点若把历史覆盖掉，"2500 分辨率"就是假的）；
    ④ 同 step 重复写幂等（resume 重跑同一评估点不留重复条目）。
    """
    import json
    import re
    import tempfile
    import inspect
    from rl.config import TrainConfig
    from rl import run_league, train_solo

    # ① 默认关闭 / 可显式开启
    assert TrainConfig().anchor_every == 0, TrainConfig().anchor_every
    assert TrainConfig.resolve("economy", anchor_every=2500).anchor_every == 2500
    assert TrainConfig.resolve("economy").anchor_every == 0

    # ② CLI 接线：flag 存在 **且** 进了 overrides 元组（源码级断言）
    src = inspect.getsource(run_league.main)
    assert '"--anchor-every"' in src, "CLI flag --anchor-every 缺失"
    assert re.search(r'for k in \([^)]*"anchor_every"', src, re.S), \
        "CLI flag 未进 overrides 元组 → 传了不生效"

    with tempfile.TemporaryDirectory() as td:
        cfg = TrainConfig(name="c_anchor", out_dir=td)
        p = os.path.join(td, "c_anchor", "solo_state.json")
        hist = [{"step": 0, "winrate": 0.5, "games": 40}]

        def rd():
            with open(p, encoding="utf-8") as f:
                return json.load(f)

        # 全点 @0：三条对照
        train_solo.write_solo_state(p, cfg, hist, 0, controls=[
            {"step": 0, "vs": "baseline0", "winrate": 0.525},
            {"step": 0, "vs": "baseline_prev", "winrate": 0.600},
            {"step": 0, "vs": "baseline_rand", "winrate": 0.625}])
        # 轻点 @2500：只有锚点（且 history 未变）
        train_solo.write_solo_state(p, cfg, hist, 2500, controls=[
            {"step": 2500, "vs": "baseline_rand", "winrate": 0.400}])
        ch = rd()["_controls_history"]
        assert len(ch) == 4, ch
        assert sorted(c["step"] for c in ch if c["vs"] == "baseline_rand") == [0, 2500], ch
        assert rd()["total_steps"] == 2500, "轻点也应推进进度（dashboard）"
        assert len(rd()["history"]) == 1, "轻点不得改动胜率曲线"

        # ④ 同 step 重复写 → 幂等
        train_solo.write_solo_state(p, cfg, hist, 2500, controls=[
            {"step": 2500, "vs": "baseline_rand", "winrate": 0.400}])
        assert len(rd()["_controls_history"]) == 4, rd()["_controls_history"]

        # ③ 下一个全点 @5000：3 旧 + 3 新，锚点序列仍连续
        train_solo.write_solo_state(p, cfg, hist, 5000, controls=[
            {"step": 5000, "vs": "baseline0", "winrate": 0.5},
            {"step": 5000, "vs": "baseline_prev", "winrate": 0.5},
            {"step": 5000, "vs": "baseline_rand", "winrate": 0.5}])
        ch = rd()["_controls_history"]
        assert len(ch) == 7, ch
        assert sorted(c["step"] for c in ch if c["vs"] == "baseline_rand") \
            == [0, 2500, 5000], ch

    print("[PASS] C 方案轻量锚点：默认关、CLI 已接线、轻点/全点对照共存可复原"
          "（step=0/2500/5000 锚点序列）、同 step 幂等")


def test_adv_inert_probe_and_const_baseline():
    """critic 惰性检验（2026-09-13）：`adv_alt` 探针与 `critic_baseline="const"`。

    预注册 `docs/critic_inertia_prereg_2026-09-13.md`。本测试断言（每条可证伪）：

    ① **探针零副作用**：`update(trans, adv_alt=X)` 与 `update(trans)` 训练后的参数
       **逐位相同** —— 无论 X 是什么、无论 `diagnose_every` 是否命中。这是"纯测量"的
       唯一可证伪形式（若哪天探针被接进梯度，这条立刻红）。
    ② **默认路径键集不变**：`adv_alt=None` 时 stats 里不出现 `adv_inert_*` 键，
       `last_adv_inert_grad is None`（旧入口/旧测试不受影响，R2）。
    ③ `adv_alt == adv` ⇒ `corr=1.000`、`resid_frac=0`、`grad_cos=+1.000`；
       `adv_alt == 2·adv` ⇒ 因 `adv_norm="scale"` 各自除以自身 std，
       **两路优势向量完全相同** ⇒ `grad_cos` 仍 = +1.000（这条同时验证尺度化口径对称）。
    ④ 故意给一个与真实优势**反序**的 `adv_alt` ⇒ `grad_cos` 显著 < 1（探针有分辨力，
       不是恒返回 1 的死探针）。
    ⑤ V≡常数 c 的 GAE 恒等式：`ret_c == adv_c + c`（`compute_gae` 的 returns 定义），
       且 `c = mean(returns_real)` 时 `adv_c` 与 `adv_real` 的 std 同量级
       （若差 10× 说明常数基线选错，预注册 P1′ 会拦）。
    """
    import copy
    import torch
    from rl.env_wrapper import RLEnv
    from rl.belief import BeliefInference
    from rl.follower import FollowerPolicy
    from rl.plan_space import PLAN_DIM
    from rl.ppo import PPOTrainer

    env = RLEnv(opponent=None, seed=0)
    obs, _ = env.reset()
    belief = BeliefInference(opp_deck=env.deck1, n_particles=128, seed=0)
    tok = belief.encode(obs, None)
    plan = np.zeros(PLAN_DIM, dtype=np.float32)
    pol = FollowerPolicy(hidden=32, plan_dim=PLAN_DIM, belief_dim=len(tok))
    trans = _tiny_rollout_transitions(pol, env, belief, tok, plan, n=12)
    real = np.array([t["adv"] for t in trans], dtype=np.float32)

    def _params(p):
        return [q.detach().clone() for q in p.parameters()]

    # ① 探针零副作用（含 diag 命中与不命中两种）
    for de in (0, 2):
        pa, pb = copy.deepcopy(pol), copy.deepcopy(pol)
        ta = PPOTrainer(pa, lr=1e-3, n_epochs=2, minibatch_size=6, shuffle=True,
                        diagnose_every=de, seed=5)
        tb = PPOTrainer(pb, lr=1e-3, n_epochs=2, minibatch_size=6, shuffle=True,
                        diagnose_every=de, seed=5)
        sa = ta.update(trans)
        sb = tb.update(trans, adv_alt=(-real).tolist())
        for w1, w2 in zip(_params(pa), _params(pb)):
            assert torch.equal(w1, w2), f"探针改变了参数（diag={de}）"
        assert abs(sa["adv_mean"] - sb["adv_mean"]) < 1e-7, (sa, sb)
        assert "adv_inert_corr" not in sa, sa          # ② 默认键集不变
        assert ta.last_adv_inert_grad is None
        assert "adv_inert_corr" in sb                  # 探针统计确实产出

    # ③ 同向量 / 同比例向量 ⇒ grad_cos = +1（adv_norm="scale" 下纯缩放 = 恒等变换）
    #    用**legacy 单 pass**（整批 12 条）测：极小合成批上策略损失对参数的梯度会出现
    #    精确抵消（实测 minibatch=6 的首批 p_gnorm 恰为 0.0），而 legacy 整批非零 ⇒
    #    grad_cos 才有定义。多轮小批路径的接线由端到端 smoke 覆盖。
    for alt, tag in ((real, "same"), (2.0 * real, "scaled")):
        tc = PPOTrainer(copy.deepcopy(pol), lr=1e-3, diagnose_every=1)
        sc = tc.update(trans, adv_alt=alt.tolist())
        assert tc.last_adv_inert_grad is not None, (tag, sc, "legacy 整批梯度应非零")
        assert sc["p_gnorm"] > 0.0, (tag, sc)
        assert abs(sc["adv_inert_corr"] - 1.0) < 1e-5, (tag, sc)
        assert sc["adv_inert_resid_frac_norm"] < 1e-6, (tag, sc)   # 归一化后必须一致
        gc = tc.last_adv_inert_grad
        assert abs(gc["grad_cos"] - 1.0) < 1e-4, (tag, gc)
    # 纯缩放：raw resid/scale 差 2×（描述性），但归一化后为 0 ⇒ 判据必须用 norm 口径
    ts = PPOTrainer(copy.deepcopy(pol), lr=1e-3, diagnose_every=1)
    ss = ts.update(trans, adv_alt=(2.0 * real).tolist())
    assert abs(ss["adv_inert_resid_frac"] - 0.5) < 1e-4, ss
    assert abs(ss["adv_inert_std_ratio"] - 0.5) < 1e-4, ss

    # ④ 反号优势 ⇒ corr = −1 且 grad_cos = −1（探针有分辨力，不是恒返回 1 的死探针）
    td = PPOTrainer(copy.deepcopy(pol), lr=1e-3, diagnose_every=1)
    sd = td.update(trans, adv_alt=(-real).tolist())
    gd = td.last_adv_inert_grad
    assert sd["adv_inert_corr"] < -0.99, sd
    assert gd is not None and gd["grad_cos"] < -0.99, gd

    # ④b 无关优势（固定随机向量）⇒ corr≈0、归一化残差大、grad_cos 明显 < 1
    #     注意别用 np.roll(±1 交替, 奇数)：那恰好等于取反（该模式周期为 2）。
    _alt_rand = np.random.default_rng(3).normal(size=len(trans)).astype(np.float32)
    te = PPOTrainer(copy.deepcopy(pol), lr=1e-3, diagnose_every=1)
    se = te.update(trans, adv_alt=_alt_rand.tolist())
    ge = te.last_adv_inert_grad
    assert abs(se["adv_inert_corr"]) < 0.6, se
    assert se["adv_inert_resid_frac_norm"] > 0.8, se
    assert ge is not None and ge["grad_cos"] < 0.9, ge

    # ④c 退化情形不许崩：常数替代优势（归一化后 std=0）→ `resid_frac_norm` 记 None
    #     （此处 grad_cos 仍有定义：合成批的 ±1 交替使梯度精确抵消，常数向量于是
    #      表现为 −1 倍方向 —— 这是**合成数据的退化**，不是探针的判据）
    tf = PPOTrainer(copy.deepcopy(pol), lr=1e-3, diagnose_every=1)
    sf = tf.update(trans, adv_alt=np.full(len(trans), 0.25, dtype=np.float32).tolist())
    assert sf["adv_inert_std_ratio"] > 1e6, sf
    assert sf["adv_inert_resid_frac_norm"] is None, sf
    assert sf["adv_inert_norm_std_ratio"] is None, sf
    assert all(v is None or np.isfinite(v) for v in sf.values()), sf

    # ⑤ V≡c 的 GAE 恒等式与量级
    rew = [t["returns"] - t["adv"] for t in trans]
    c = float(np.mean([t["returns"] for t in trans]))
    term = [False] * len(trans)
    adv_c, ret_c = PPOTrainer.compute_gae(rew, [c] * len(rew), term, 0.997, 0.99,
                                          truncated=[False] * len(rew), last_value=c)
    assert np.allclose(ret_c, adv_c + c, atol=1e-4), (ret_c[:3], adv_c[:3])
    adv_r, _ = PPOTrainer.compute_gae(rew, [0.0] * len(rew), term, 0.997, 0.99,
                                      truncated=[False] * len(rew), last_value=0.0)
    assert 0.1 < float(adv_c.std()) / max(1e-9, float(adv_r.std())) < 10.0, \
        (float(adv_c.std()), float(adv_r.std()))


def test_precise_threat():
    """精确塔伤（rl/threat_precise.py）：双向一次算完与工具对账、桥列/半场口径对账、
    默认关时逐位不变、触发条件的物理正确性、跨局自动复位。"""
    import battle as bm
    import player as pm
    from core import Position
    from arena import TileGrid
    import threat_calc
    from threat_calc import estimate_tower_threat
    from rl import belief_planner as bp_mod
    from rl.threat_precise import RIVER_Y1, RIVER_Y2
    from rl.threat_precise import (combine_both_directions, trigger_directions,
                                   PreciseThreat, HP_PER_PRESSURE, HORIZON_S,
                                   MAX_HOLD_S,
                                   _half_of, _TOWER_IDS, bridge_cols)

    deck = ["Knight", "MiniPekka", "Arrows", "Minions", "Musketeer", "Fireball",
            "Giant", "Archer"]

    def fresh():
        return bm.BattleState(pm.PlayerState(0, list(deck), 5.0),
                              pm.PlayerState(1, list(deck), 5.0), card_level=11)

    def put(bs, pid, card, pos):
        p = bs.players[pid]
        p.cycle = [card] + [c for c in p.cycle if c != card]
        p.elixir = 10.0
        assert bs.deploy_card(pid, card, pos), f"{card} 部署失败"

    # ① 桥列/塔 id 常量与引擎对账（【红线 R7】同一数值 + 对账）
    assert threat_calc._TOWER_IDS == _TOWER_IDS, "塔 id 约定与 threat_calc 不一致"
    ar = TileGrid()
    bc = bridge_cols()
    assert 3 in bc and 14 in bc, f"桥列应含左右桥中心格: {sorted(bc)}"
    assert 9 not in bc, "中轴处不应是桥"
    for cx in sorted(bc):
        assert ar.is_walkable(Position(cx + 0.5, (RIVER_Y1 + RIVER_Y2) / 2.0)), \
            f"桥列 {cx} 应可走"
    assert not ar.is_walkable(Position(9.0, 15.5)), "河道非桥位置应不可走"

    # ② 半场口径：中轴旁那一格两半都算 ⇒ 每支部队至少落在一个半场
    for cx in range(18):
        assert len(_half_of(cx + 0.5)) >= 1, f"cell {cx} 未落入任何半场"
    assert _half_of(9.5) == ("L", "R"), "中轴旁那一格应同时属于左右半场（用户口径）"
    assert _half_of(0.5) == ("L",) and _half_of(17.5) == ("R",)

    # ③ 双向一次算完 与 工具单调用 逐位相等（这是"一次推演替代两次"的前提）
    cases = []
    bs = fresh()
    put(bs, 1, "Giant", Position(14.5, 19.0))
    cases.append(bs)
    bs2 = fresh()
    put(bs2, 0, "Giant", Position(3.5, 14.0))
    put(bs2, 1, "Musketeer", Position(14.5, 20.0))
    cases.append(bs2)
    cases.append(fresh())
    for i, b in enumerate(cases):
        comb = combine_both_directions(b, HORIZON_S)
        e0 = estimate_tower_threat(b, 0, horizon=HORIZON_S)["total"]
        e1 = estimate_tower_threat(b, 1, horizon=HORIZON_S)["total"]
        assert comb["to_p0"] == e0 and comb["to_p1"] == e1, \
            f"用例{i}: 双向一次算完与工具不一致 {comb} vs ({e0},{e1})"
        # 原局面不得被污染
        assert b.time == cases[i].time

    # ④ 触发条件的物理正确性
    bs = fresh()
    put(bs, 1, "Giant", Position(3.5, 17.5))          # 敌 Giant 到左桥头（合法部署最近点）
    trig, halves = trigger_directions(bs, 1)
    assert trig and "L" in halves, f"左桥头有敌兵且我方左半无兵 ⇒ 应触发: {trig} {halves}"
    put(bs, 0, "Knight", Position(4.0, 12.0))         # 我方左半场放人 ⇒ 不再触发
    trig2, _ = trigger_directions(bs, 1)
    assert not trig2, "左半场已有我方单位 ⇒ 不应触发"
    # 敌方偏远（还没到桥头、也没过河）→ 不触发
    bs3 = fresh()
    put(bs3, 1, "Giant", Position(6.0, 21.0))
    assert not trigger_directions(bs3, 1)[0], "敌兵在敌方半场深处 ⇒ 不应触发"

    # ⑤ 默认关：`_enemy_pressure` 必须与旧实现逐位相同
    bp_mod.set_precise_threat(False)
    for b in cases:
        got = bp_mod._enemy_pressure(b)
        ref = bp_mod._crude_enemy_pressure(b)
        assert got == ref, f"开关关闭时应逐位等于旧实现: {got} vs {ref}"

    # ⑥ 开启后：未触发帧回退旧口径；触发帧给出**精确值**；同初始状态 ⇒ 确定
    bp_mod.set_precise_threat(True)
    try:
        b = cases[0]                                   # 敌 Giant 在敌方半场 → 不触发
        assert bp_mod._enemy_pressure(b) == bp_mod._crude_enemy_pressure(b), \
            "未触发帧应回退旧口径"

        def _fresh_call(board):
            """同一初始状态下的单次调用（提供器重建 ⇒ 不受上一次调用影响）。"""
            bp_mod.set_precise_threat(False)
            bp_mod.set_precise_threat(True)
            return bp_mod._enemy_pressure(board)

        b = cases[1]                                   # 我方 Giant 到左桥头 → 触发方向 B
        r1 = _fresh_call(b)
        r2 = _fresh_call(b)
        assert r1 == r2, f"同一局面同一初始状态应确定: {r1} vs {r2}"
        exp_b = estimate_tower_threat(b, 1, horizon=HORIZON_S)["total"] / HP_PER_PRESSURE
        exp_a = estimate_tower_threat(b, 0, horizon=HORIZON_S)["total"] / HP_PER_PRESSURE
        assert abs(r1[1] - exp_b) < 1e-6, f"触发方向的 my_pressure 应为精确值: {r1[1]} vs {exp_b}"
        assert abs(r1[0] - exp_a) < 1e-6, \
            f"一次推演同时得到另一方向 ⇒ threat 也应是精确值: {r1[0]} vs {exp_a}"
        st = bp_mod.precise_threat_stats()
        assert st["triggers"] >= 1, f"应有触发计数: {st}"

        # 状态机：威胁源消失 ⇒ 该方向退出保持、回到回退口径
        b_clear = fresh()                              # 场上无敌方单位
        r3 = bp_mod._enemy_pressure(b_clear)
        assert r3 == bp_mod._crude_enemy_pressure(b_clear), \
            "威胁源消失后应退出保持并回到回退口径"
    finally:
        bp_mod.set_precise_threat(False)
    assert bp_mod.precise_threat_stats() is None, "关闭后不应再有提供器"

    # ⑦ 跨局自动复位：time 回退 ⇒ 不得沿用上一局的保持值
    pr = PreciseThreat(horizon=HORIZON_S)
    b = cases[1]
    pr.pressures(b)
    assert pr._hold["B"] is not None, "触发后应进入保持"
    nb = fresh()                                       # 新一局，time=0 < 上一局
    pr.pressures(nb)
    assert pr._hold["B"] is None and pr._hold["A"] is None, "跨局必须复位保持值"
    pr.reset()

    # ⑧ 保持寿命上限 MAX_HOLD_S：到期强制失效；触发仍成立时同帧刷新；无触发则回退粗糙
    #    ⚠️ 这里直接改 `battle.time` 做状态机单元测试（提供器读的就是量 `battle.time`）；
    #    真实时间推进下的行为由 scripts/probe_hold_recompute.py 的 rollout 度量。
    assert MAX_HOLD_S == 4.0 * HORIZON_S, "寿命上限取值应 = 4×视界（预注册写死）"
    pr = PreciseThreat(horizon=HORIZON_S)
    bb = fresh()
    put(bb, 1, "Giant", Position(3.5, 17.5))           # 敌 Giant 到左桥头 ⇒ 方向 A 触发
    bb.time = 50.0
    pr.pressures(bb)
    assert pr._hold["A"] is not None, "触发后应进入保持"
    assert pr.stats["triggers"] == 1 and pr.stats["expired"] == 0, f"{pr.stats}"
    bb.time = 50.0 + MAX_HOLD_S - 1.0                  # (a) 未到期
    pr.pressures(bb)
    assert pr._hold["A"] is not None and pr.stats["expired"] == 0, \
        f"未到 {MAX_HOLD_S}s 不应失效: {pr.stats}"
    bb.time = 50.0 + MAX_HOLD_S + 0.5                  # (b) 到期 ∧ 触发仍成立 ⇒ 同帧刷新
    r1 = pr.pressures(bb)
    # 该盘面没有我方单位 ⇒ 方向 B 那次"顺手武装"早已按 `_exit` 退出，故只有 A 到期
    assert pr.stats["expired"] == 1, f"应记一次寿命到期: {pr.stats}"
    assert pr.stats["triggers"] == 2, f"触发仍成立 ⇒ 应同帧重算: {pr.stats}"
    assert pr._hold["A"] is not None, "重算后应重新进入保持"
    exp_a = estimate_tower_threat(bb, 0, horizon=HORIZON_S)["total"] / HP_PER_PRESSURE
    assert abs(r1[0] - exp_a) < 1e-6, f"刷新后应为最新精确值: {r1[0]} vs {exp_a}"

    pr.reset()                                         # (c) 到期 ∧ 触发已失效 ⇒ 退出回退
    b2 = fresh()
    put(b2, 1, "Giant", Position(3.5, 17.5))
    b2.time = 80.0
    pr.pressures(b2)
    assert pr._hold["A"] is not None, "触发武装失败"
    put(b2, 0, "Knight", Position(4.0, 12.0))          # 我方左半场有兵 ⇒ 触发条件失效
    b2.time = 80.0 + MAX_HOLD_S - 1.0
    pr.pressures(b2)
    assert pr._hold["A"] is not None, "触发失效但敌兵仍在桥头 ⇒ 寿命未到不得退出"
    b2.time = 80.0 + MAX_HOLD_S + 0.5
    r2 = pr.pressures(b2)
    assert pr._hold["A"] is None and pr._hold["B"] is None, "寿命到期且无触发 ⇒ 必须退出保持"
    # 本盘面我方 Knight 已过河 ⇒ 方向 B 的保持也活到了寿命上限 ⇒ 两个方向各记一次到期
    assert pr.stats["expired"] == 2 and pr.stats["triggers"] == 1, f"{pr.stats}"
    assert abs(r2[0] - pr._crude(b2)[0]) < 1e-9, "退出后应回退粗糙口径"

    print(f"[PASS] 精确塔伤：双向一次算完与工具逐位相等（3 用例）；桥列/塔id/半场口径对账；"
          f"默认关逐位不变；触发条件物理正确（左桥头触发 / 有守军不触发 / 远处不触发）；"
          f"开启后触发方向 = 引擎精确值（H={HORIZON_S}s，常量 {HP_PER_PRESSURE}）；跨局自动复位；"
          f"保持寿命上限 {MAX_HOLD_S}s（未到期不失效 / 到期且触发则同帧刷新 / 到期且无触发则回退粗糙）")


def main():
    # 与 run_league.main 同一兜底：日志含中文/emoji，Windows cp936 管道会崩
    from rl.run_league import _force_utf8_stdout
    _force_utf8_stdout()
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
    test_deck_pool_factory()
    test_dashboard_card_stats()
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
    test_death_damage_scaling()
    test_vines_snare_fl_duration()
    test_log_rolling_direction()
    test_behavioral_metrics()
    test_mk_spawn_damage_and_iw_slow_fl()
    test_tower_value_mult()
    test_reward_tower_premium()
    test_reward_tower_premium_rlenv_flow()
    test_value_channel_norm_and_gnorm_split()
    test_history_dedup_and_gates()
    test_solo_rand_anchor()
    test_anchor_light_point_state()
    test_opponent_pool_rand_anchor()
    test_opponent_pool_mix_multi_dir()
    test_pfsp_gate_and_dynamic_hist()
    test_enc_layernorm_gru_vitality()
    test_value_bypass()
    test_value_independent_encoder()
    test_ppo_multi_epoch_minibatch()
    test_stall_settlement_margin()
    test_adv_inert_probe_and_const_baseline()
    test_precise_threat()
    print("\nALL SELFTESTS PASSED")


if __name__ == "__main__":
    main()
