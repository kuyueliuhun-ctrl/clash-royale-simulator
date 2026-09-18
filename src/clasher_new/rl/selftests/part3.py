# -*- coding: utf-8 -*-
"""`rl/selftest.py` 的测试 · 第 3/5 部分（20 个，test_flow_league_smoke … test_bayes_queue_lock）。

**函数体逐字未改**（切片生成）；顶部显式导入共用底座。"""

# T2-8：本部分从 `rl/selftest.py` 原样切出（**函数体逐字未改**，只动了下面两处**路径推导**：
# 原 L5615 / L5676 的 `dirname(dirname(abspath(__file__)))` 在搬到 `rl/selftests/` 后会少一层
# ⇒ 改用 `selftest_common._PARENT`（仍 = `src/clasher_new`）。
# `*` 不导出下划线名 ⇒ 私有 helper / 导入名一律**显式**列出（缺一个就是 NameError）。
from rl.selftest_common import (  # noqa: F401
    os, sys, time, random, shutil, np, Card, _PARENT,
    _mark_skip, _make_policy_and_tokens, _mk_env, _intents, _tiny_rollout_transitions, _FakeCfg,
)

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
    """僵局早停：**2026-09-17 起默认关闭**（用户拍板：零塔损局必须打满，接受时间损失）。

    断言（配套 `config.train_stall_stop` 默认 True→False）：
      - **默认**（stall_stop=False）：双方都不部署 → **不早停**，一路打到 t≈300s 由引擎裁决（判平）；
      - **显式 stall_stop=True**：旧行为仍可逐位复现（约 50s 就 break 判平）。
    """
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

    # ① 默认：不早停 → 打满到加时硬顶（300s）
    env = RLEnv(opponent=None, seed=3)
    _prepare_env(env, idle, idle)
    w = _run_side0(env, idle, BeliefInference(opp_deck=env.deck1, n_particles=16, seed=3),
                   BeliefPlanner(), max_steps=600, reset_seed=3)
    assert w is None, "零塔损僵局最终仍由引擎按塔血合计判平"
    assert env.battle.time >= 299.0, \
        f"默认必须打满到 300s（实际 {env.battle.time:.1f}s）——早停应当已关闭"

    # ② 显式开启：旧行为（约 50s 判平）仍可复现
    env2 = RLEnv(opponent=None, seed=3)
    _prepare_env(env2, idle, idle)
    t0 = time.monotonic()
    w2 = _run_side0(env2, idle, BeliefInference(opp_deck=env2.deck1, n_particles=16, seed=3),
                    BeliefPlanner(), max_steps=600, reset_seed=3, stall_stop=True)
    dt = time.monotonic() - t0
    assert w2 is None, "开启早停时僵局判平"
    assert env2.battle.time < 180.0, \
        f"开启早停应远早于 180s（实际 {env2.battle.time:.1f}s）"
    print(f"[PASS] 僵局早停：默认关闭（打满 {env.battle.time:.0f}s 由引擎裁决）；"
          f"显式 stall_stop=True 复现旧行为（{env2.battle.time:.0f}s / {dt:.1f}s 判平）")


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
        """act 恒返回 noop：双方都不部署 → 打满 300s 后引擎按塔血合计判平（早停已默认关闭），
        mean_reward 仍然确定性可断言（= 平局罚）。"""
        def act(self, obs, belief_token, plan_token, get_mask,
                hidden=None, deterministic=False):
            return ActionBundle.noop(), 0.0, 0.0, hidden, {}

    # T1-4：临时 run 目录从硬编码 `runs/_tmp_drawtest` 改为 `tempfile.mkdtemp()` + `finally` 清理。
    # 两个原因（均为 T0 盘点实测）：
    #   ① 旧写法把清理放在**函数末尾**：断言一旦中途失败，`rmtree` **不会执行**
    #      ⇒ 在仓库工作区里**留下残留目录**；
    #   ② 它写的是**相对**路径 ⇒ 只有 cwd=src/clasher_new 时才落在预期位置。
    # 现在用**绝对临时目录** ⇒ 无论断言成败都不在仓库留痕，且与 cwd 无关。
    import tempfile
    _tmp_out = tempfile.mkdtemp(prefix="selftest_draw_penalty_")
    try:
        cfg = TrainConfig(name="selftest_draw_penalty", hidden_dim=32, n_eval_games=2,
                          max_ep_steps=600, seed=5, out_dir=_tmp_out)
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
    finally:
        shutil.rmtree(_tmp_out, ignore_errors=True)


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
