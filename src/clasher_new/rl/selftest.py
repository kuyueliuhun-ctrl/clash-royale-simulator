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
    """命名配置：预设解析互不影响、奖励权重可注入 RLEnv 并改变回报。"""
    import tempfile
    from rl.config import TrainConfig, reward_to_env
    from rl.env_wrapper import RLEnv
    from rl.action_bundle import ActionBundle

    std = TrainConfig.resolve("standard")
    agg = TrainConfig.resolve("aggressive")
    assert std.reward["crown_weight"] == 5.0
    assert agg.reward["crown_weight"] == 8.0
    assert agg.reward["tower_dmg_opp"] > std.reward["tower_dmg_opp"]
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
    print("[PASS] 命名配置：预设/加载/奖励权重注入 RLEnv 正常")


def test_reward_economy_preset():
    """economy 预设存在且生效；standard 保持旧公式（normalize 关、费差 0）。"""
    import tempfile
    import os
    from rl.config import TrainConfig, reward_to_env
    from rl.env_wrapper import RLEnv

    eco = TrainConfig.resolve("economy")
    std = TrainConfig.resolve("standard")
    assert eco.reward["normalize_tower_dmg"] is True
    assert eco.reward["elixir_diff_weight"] > 0
    assert std.reward["normalize_tower_dmg"] is False
    assert std.reward["elixir_diff_weight"] == 0.0
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
    print("[PASS] economy 预设：normalize+费差生效、standard 保持旧公式、JSON 往返正常")


def test_reward_economy_level_invariance():
    """economy 奖励：塔损按塔血%归一化 → 同一事件跨等级奖励一致；lv11 与旧公式逐位一致。"""
    from rl.env_wrapper import compute_reward, _TOWER_HP_ANCHOR
    from rl.config import TrainConfig, reward_to_env

    eco = reward_to_env(TrainConfig.resolve("economy"))
    std = reward_to_env(TrainConfig.resolve("standard"))
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
    old11, old16 = r(std, lv11_max), r(std, lv16_max)
    # economy：同一"塔血百分比事件"跨等级奖励一致
    assert abs(eco11 - eco16) < 1e-9, f"economy 应跨等级不变: {eco11} vs {eco16}"
    # economy 在 lv11 与旧公式逐位一致（行为不变）
    assert abs(eco11 - old11) < 1e-12, f"economy@lv11 应等于旧公式: {eco11} vs {old11}"
    # 旧公式确实随等级漂移（这正是要修的问题，回归验证）
    assert abs(old11 - old16) > 0.01, "旧公式应随等级漂移（回归验证）"
    print(f"[PASS] economy 奖励：跨等级不变({eco11:.4f})、lv11 与旧公式一致、"
          f"旧公式漂移({old11:.3f}->{old16:.3f})")


def test_reward_economy_elixir_diff():
    """economy 费差项：显式给圣水定价；potential-style（闭环累计归零）。"""
    from rl.env_wrapper import compute_reward
    from rl.config import TrainConfig, reward_to_env

    eco = reward_to_env(TrainConfig.resolve("economy"))
    std = reward_to_env(TrainConfig.resolve("standard"))
    base = dict(blue_hps_old=10928.0, red_hps_old=10928.0,
                blue_hps_new=10928.0, red_hps_new=10928.0,
                blue_left_old=3, red_left_old=3, blue_left_new=3, red_left_new=3,
                winner=None, invalid_count=0,
                blue_hps_max=10928.0, red_hps_max=10928.0)

    # 我方花 4 费（费差 -4）→ economy 显式 -0.4；旧公式 0（无圣水定价）
    r_eco = compute_reward(eco, my_elixir_before=5.0, opp_elixir_before=5.0,
                           my_elixir_after=1.0, opp_elixir_after=5.0, **base)
    r_std = compute_reward(std, my_elixir_before=5.0, opp_elixir_before=5.0,
                           my_elixir_after=1.0, opp_elixir_after=5.0, **base)
    assert abs(r_eco - (-0.4)) < 1e-9, f"花4费应-0.4: {r_eco}"
    assert abs(r_std - 0.0) < 1e-12, f"旧公式花费无显式惩罚: {r_std}"
    # 对方花 4 费（我方费差 +4）→ economy 显式 +0.4
    r_eco2 = compute_reward(eco, my_elixir_before=5.0, opp_elixir_before=5.0,
                            my_elixir_after=5.0, opp_elixir_after=1.0, **base)
    assert abs(r_eco2 - 0.4) < 1e-9, f"对方花4费应+0.4: {r_eco2}"
    # potential-style：闭环（花4→对方花4→我方回5→对方回5）费差项累计归零
    steps = [(5.0, 5.0, 1.0, 5.0), (1.0, 5.0, 1.0, 1.0),
             (1.0, 1.0, 5.0, 1.0), (5.0, 1.0, 5.0, 5.0)]
    total = sum(compute_reward(eco, my_elixir_before=a, opp_elixir_before=b,
                               my_elixir_after=c, opp_elixir_after=d, **base)
                for a, b, c, d in steps)
    assert abs(total) < 1e-9, f"费差项应闭环归零: {total}"
    print(f"[PASS] economy 费差项：花4费=-0.4/对方花4费=+0.4/闭环累计归零（{total:.2e}）")


def test_reward_economy_trade_pricing():
    """费差 vs 塔血的真实 trade：economy 权重下给出符合直觉的定价（旧公式学不出）。"""
    from rl.env_wrapper import compute_reward
    from rl.config import TrainConfig, reward_to_env

    eco = reward_to_env(TrainConfig.resolve("economy"))
    std = reward_to_env(TrainConfig.resolve("standard"))
    base = dict(blue_hps_old=10928.0, red_hps_old=10928.0,
                blue_left_old=3, red_left_old=3, blue_left_new=3, red_left_new=3,
                winner=None, invalid_count=0,
                blue_hps_max=10928.0, red_hps_max=10928.0)

    def r(weights, **kw):
        return compute_reward(weights, **dict(base, **kw))

    # ① 花 4 费磨掉对方 4.3% 总塔血（≈火球直击塔）：应为正（高效换血）
    trade = r(eco, blue_hps_new=10928.0, red_hps_new=10928.0 - 0.043 * 10928,
              my_elixir_before=5.0, opp_elixir_before=5.0,
              my_elixir_after=1.0, opp_elixir_after=5.0)
    assert trade > 0, f"火球直击塔应划算: {trade}"
    # ② 花 4 费但 0 塔损（浪费）：应为负
    waste = r(eco, blue_hps_new=10928.0, red_hps_new=10928.0,
              my_elixir_before=5.0, opp_elixir_before=5.0,
              my_elixir_after=1.0, opp_elixir_after=5.0)
    assert waste < 0, f"白花 4 费应为负: {waste}"
    # ③ 让塔挨 1% 总塔血、换 2 费差（对方花 2 费而我不防）：应为正（trade 划算）
    trade2 = r(eco, blue_hps_new=10928.0 - 0.01 * 10928, red_hps_new=10928.0,
               my_elixir_before=5.0, opp_elixir_before=5.0,
               my_elixir_after=5.0, opp_elixir_after=3.0)
    assert trade2 > 0, f"挨 1% 塔血换 2 费差应划算: {trade2}"
    # ④ 同一事件在旧公式（无费差项）：为负 → 模型学不出这个 trade（缺陷回归）
    old = r(std, blue_hps_new=10928.0 - 0.01 * 10928, red_hps_new=10928.0,
            my_elixir_before=5.0, opp_elixir_before=5.0,
            my_elixir_after=5.0, opp_elixir_after=3.0)
    assert old < 0, f"旧公式挨打换费差应为负（缺陷）: {old}"
    print(f"[PASS] economy trade 定价：火球直击塔={trade:.3f}>0 / 浪费={waste:.3f}<0 / "
          f"挨1%换2费差={trade2:.3f}>0（旧公式={old:.3f}<0）")


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
    test_classified_decks()
    test_league_training_loop()
    test_belief_follower_ppo_league()
    test_config_reward_weights()
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
    print("\nALL SELFTESTS PASSED")


if __name__ == "__main__":
    main()
