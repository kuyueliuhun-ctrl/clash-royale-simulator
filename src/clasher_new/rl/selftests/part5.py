# -*- coding: utf-8 -*-
"""`rl/selftest.py` 的测试 · 第 5/5 部分（20 个，test_history_dedup_and_gates … test_mask_partial_bundle_invariants）。

**函数体逐字未改**（切片生成）；顶部显式导入共用底座。"""

# T2-8：本部分从 `rl/selftest.py` 原样切出（**函数体逐字未改**，只动了下面两处**路径推导**：
# 原 L5615 / L5676 的 `dirname(dirname(abspath(__file__)))` 在搬到 `rl/selftests/` 后会少一层
# ⇒ 改用 `selftest_common._PARENT`（仍 = `src/clasher_new`）。
# `*` 不导出下划线名 ⇒ 私有 helper / 导入名一律**显式**列出（缺一个就是 NameError）。
from rl.selftest_common import (  # noqa: F401
    os, sys, time, random, shutil, np, Card, _PARENT,
    _mark_skip, _make_policy_and_tokens, _mk_env, _intents, _tiny_rollout_transitions, _FakeCfg,
)

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
        # T1-5：这个 `else` 原先只 print 一行 [SKIP]（且整段 ②③④ 静默跳过）⇒ 假绿。
        _mark_skip("无可用历史 ckpt ⇒ 本条 ②③（有 hist 时的采样分布 / 锚点权重与 E1 同种子一致）"
                   "**未验证**，只跑了 ①④⑤⑥ 其余分支")

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


def test_eval_scheduler_two_tier():
    """两级评估调度（2026-09-18）：小评估（`steps_per_eval`）+ 稀疏大评估（`big_eval_every`）。

    本测试盯的都是会**静默失效**的点：
    ① 默认全关（`big_eval_every=0` / `n_eval_games_big=0`）⇒ 与旧实现逐位一致；
    ② `big_eval_every=0` 时 `due()` 触发序列 **恰好等于** 旧的
       `step % steps_per_eval == 0`，且 `n_games` 恒 = `n_eval_games`；
    ③ 两网格**并集**：8000 与 100000 不整除 ⇒ 100k/200k/…/900k 共 9 个大评估点被
       **插入**（不被 8000 网格吞掉），1e6 同时命中两网格 → 只评一次、按大预算；
    ④ CLI 已接线到 overrides（漏接 = 传了不生效，最典型的脚枪）；
    ⑤ `_eval_and_snapshot(n_games=…)` 真的把局数传给了 `eval_round_robin`
       （漏传 ⇒ 大评估静默退化成小评估，曲线噪声不变而你以为降了）。
    """
    import re
    import inspect
    from types import SimpleNamespace
    from rl.config import TrainConfig
    from rl import run_league

    # ① 默认关 / 可显式开启
    d = TrainConfig()
    assert d.big_eval_every == 0, d.big_eval_every
    assert d.n_eval_games_big == 0, d.n_eval_games_big
    assert d.eval_big_at_start is True
    assert TrainConfig.resolve("economy", big_eval_every=100000,
                               n_eval_games_big=20).big_eval_every == 100000
    assert TrainConfig.resolve("economy").big_eval_every == 0

    # ② 关掉大评估 ⇒ 与旧触发条件逐位一致
    cfg = TrainConfig(name="t2", steps_per_eval=8000, n_eval_games=10,
                      total_steps=40000)
    s = run_league._EvalScheduler(cfg, 0)
    got = []
    for st in range(0, cfg.total_steps + 1):
        r = s.due(st)
        if r:
            got.append((st, r[0], r[1]))
    old = [(st, "small", 10) for st in range(1, cfg.total_steps + 1)
           if cfg.steps_per_eval and st % cfg.steps_per_eval == 0]
    assert got == old, (got, old)
    # 0 号点由 eval_at_start 单独触发（due 不重复触发）⇒ 旧序列里 0 也不该被 due 吃掉
    assert got[0][0] == 8000, got[0]
    assert s.start_budget() == ("small", 10), s.start_budget()

    # ③ 两网格并集 + 不整除时插入 + 同格点只评一次
    cfg = TrainConfig(name="t2b", steps_per_eval=8000, big_eval_every=100000,
                      n_eval_games=10, n_eval_games_big=20, total_steps=1000000)
    s = run_league._EvalScheduler(cfg, 0)
    pts = s.expected_points(1000000)
    # 8000 网格 126 点（含 0）；100000 网格 11 点，其中 6 点（k 偶：0/200k/…/1000k）
    # 已被 8000 网格覆盖 ⇒ 并集 = 126 + (11−6) = 131；多出来的是 100k/300k/500k/700k/900k
    assert len(pts) == 131, len(pts)
    assert pts[:2] == [0, 8000], pts[:2]
    assert 100000 in pts and 96000 in pts and 104000 in pts, "大评估点必须被插入"
    kinds = {p: s.budget_for(p) for p in pts}
    assert kinds[0] == ("big", 20), kinds[0]        # 起始点用大预算（低噪声基线）
    assert kinds[8000] == ("small", 10), kinds[8000]
    assert kinds[100000] == ("big", 20), kinds[100000]
    assert kinds[200000] == ("big", 20), kinds[200000]   # 200k 同时在 8000 网格上 ⇒ 大
    assert kinds[96000] == ("small", 10), kinds[96000]
    assert kinds[1000000] == ("big", 20), kinds[1000000]   # 1e6 同时命中两网格 ⇒ 大
    n_big = sum(1 for p in pts if kinds[p][0] == "big")
    assert n_big == 11, n_big                       # 0/100k/../900k + 1e6
    # 走一遍 due()：每个评估点恰好触发一次，且预算与 budget_for 一致
    seen = []
    for st in range(0, 1000000 + 1):
        r = s.due(st)
        if r:
            seen.append((st, r[0], r[1]))
    seen.append((0, *s.start_budget()))             # eval_at_start 的 0 号点
    seen.sort()
    assert [p for p, _, _ in seen] == pts, "due() 的评估点集合必须等于并集"
    assert seen == [(p, *kinds[p]) for p in pts], "due() 的预算必须等于 budget_for"
    assert sum(g for _, _, g in seen) == 120 * 10 + 11 * 20, "局数预算对账"
    plan = s.plan_str(1000000, 5)
    assert "评估点 131 个" in plan and "大 11 / 小 120" in plan, plan
    assert "预计总对局 7100 局" in plan, plan       # (120*10 + 11*20) * 5 对

    # n_eval_games_big=0 ⇒ 回退到 n_eval_games（只改间隔不改局数）
    cfg2 = TrainConfig(name="t2c", steps_per_eval=8000, big_eval_every=100000,
                       n_eval_games=10, n_eval_games_big=0, total_steps=200000)
    assert run_league._EvalScheduler(cfg2, 0).budget_for(100000) == ("big", 10)
    # eval_big_at_start=False ⇒ 起始点用小预算
    cfg3 = TrainConfig(name="t2d", steps_per_eval=8000, big_eval_every=100000,
                       n_eval_games=7, n_eval_games_big=21, total_steps=200000,
                       eval_big_at_start=False)
    assert run_league._EvalScheduler(cfg3, 0).start_budget() == ("small", 7)

    # 续训：游标从 start_step 起步 ⇒ 不重复评估已评过的格点（run_state 是评估后才写的）
    cfg5 = TrainConfig(name="t2f", steps_per_eval=8000, big_eval_every=100000,
                       n_eval_games=10, n_eval_games_big=20)
    s5 = run_league._EvalScheduler(cfg5, 8000)
    assert s5.due(8000) is None, "resume 后不得重复评估同一格点"
    assert s5.due(16000) == ("small", 10)
    s6 = run_league._EvalScheduler(cfg5, 100000)
    assert s6.due(100000) is None, "resume 后不得重复评估大评估格点"
    assert s6.due(104000) == ("small", 10)

    # ④ CLI 接线（源码级断言：flag 存在 **且** 进了 overrides 元组）
    src = inspect.getsource(run_league.main)
    assert '"--big-eval-every"' in src, "CLI flag --big-eval-every 缺失"
    assert '"--n-eval-games-big"' in src, "CLI flag --n-eval-games-big 缺失"
    assert re.search(r'for k in \([^)]*"big_eval_every"', src, re.S), \
        "CLI flag 未进 overrides 元组 → 传了不生效"
    assert re.search(r'for k in \([^)]*"n_eval_games_big"', src, re.S), \
        "CLI flag 未进 overrides 元组 → 传了不生效"

    # ⑤ _eval_and_snapshot 真的把 n_games 传下去（打桩 eval_round_robin / _save_snapshot）
    calls = []
    orig_eval, orig_save = run_league.eval_round_robin, run_league._save_snapshot
    try:
        run_league.eval_round_robin = lambda lg, n, ms, sd, st, **kw: calls.append((st, n)) or []
        run_league._save_snapshot = lambda *a, **kw: None
        cfg4 = TrainConfig(name="t2e", n_eval_games=10, max_ep_steps=360)
        fake = SimpleNamespace(elo_table=lambda: {}, agents={}, only_vs_main=True)
        run_league._eval_and_snapshot(fake, None, None, cfg4, 8000, "cpu", False)
        run_league._eval_and_snapshot(fake, None, None, cfg4, 100000, "cpu", False, n_games=20)
        assert calls == [(8000, 10), (100000, 20)], calls
    finally:
        run_league.eval_round_robin = orig_eval
        run_league._save_snapshot = orig_save

    print("[PASS] 两级评估调度：默认关（旧行为逐位一致）、并集插入 100k 网格、"
          "同格点只评一次按大预算、1e6 共 131 点 / 7100 局、CLI 已接线、"
          "n_games 真传到 eval_round_robin")


def test_eval_round_robin_parallel_equivalence():
    """run 模式评估并行分片（2026-09-18）：`eval_round_robin(..., n_workers=N)`。

    动机：run 模式评估原本**串行**跑每个 pair 的每一局（纯 Python 引擎模拟 ≈9 s/局），
    1M 步长跑里 7100 局 = 17+ 小时。并行化后 Elo/PFSP 必须仍在**父进程按 (pair, g)
    规范顺序逐局补记**，否则 league 状态演化会与串行不同。

    本测试断言（全部可证伪）：
    ① `_policy_spec` → `_spec_to_policy` 往返：脚本策略重建后同 seed 同行为；
       学习型策略权重逐位相同（漏传 dim/flag 会在此炸）；
    ② **n_games=1 时并行与串行结果逐位一致**：单局不需要链式洗牌（每 pair 的 env
       只 shuffle 一次，且种子相同）⇒ 并行分片不得改变任何一局的结果；
    ③ 并行分支真的被走到（源码级：`_run_eval_pairs_parallel` 被调用），且
       `eval_round_robin` 的默认 `n_workers=0` = 旧串行行为。
    """
    import inspect
    import torch
    from rl.config import TrainConfig
    from rl import run_league
    from rl.league import League
    from rl.opponents import ScriptedPolicy
    from rl.follower import FollowerPolicy
    from rl.belief import BeliefInference
    from rl.env_wrapper import RLEnv
    from rl.plan_space import PLAN_DIM

    # ① spec 往返
    sp = ScriptedPolicy(mode="random", pool=None, deck_pool=None, seed=123)
    sp2 = run_league._spec_to_policy(run_league._policy_spec(sp))
    assert (sp2.mode, sp2.pool, sp2.deck_pool, sp2.seed) == \
        (sp.mode, sp.pool, sp.deck_pool, sp.seed)
    assert sp2.deck() == sp.deck(), "同 seed 的脚本策略重建后 deck() 必须一致"
    env0 = RLEnv(seed=0)
    belief0 = BeliefInference(opp_deck=env0.deck1, n_particles=128, seed=0)
    main = FollowerPolicy(hidden=32, plan_dim=PLAN_DIM,
                          belief_dim=len(belief0.encode(None, None)),
                          value_bypass=True, value_independent=True)  # = economy 架构
    p2 = run_league._spec_to_policy(run_league._policy_spec(main))
    assert (p2.value_bypass, p2.value_independent) == (True, True), "架构标志必须随 spec 传递"
    sd_a, sd_b = main.state_dict(), p2.state_dict()
    assert set(sd_a) == set(sd_b), (set(sd_a) ^ set(sd_b))
    assert all(torch.equal(sd_a[k], sd_b[k]) for k in sd_a), "权重往返必须逐位相同"

    # ② 并行 vs 串行（n_games=1 ⇒ 无洗牌链差异 ⇒ 必须逐位一致）
    def build():
        lg = League(seed=7)
        for aid, sd in (("a", 11), ("b", 22), ("c", 33)):
            lg.add_agent(aid, kind="baseline", policy=ScriptedPolicy(mode="random", seed=sd),
                         replace=True)
        return lg

    lg1 = build()
    r1 = run_league.eval_round_robin(lg1, 1, 60, 5, 0, n_workers=1, record=True)
    ser_elo = {k: round(float(v), 4) for k, v in lg1.elo_table().items()}

    lg2 = build()
    r2 = run_league.eval_round_robin(lg2, 1, 60, 5, 0, n_workers=2, record=True)
    par_elo = {k: round(float(v), 4) for k, v in lg2.elo_table().items()}
    assert ser_elo == par_elo, (ser_elo, par_elo)
    assert len(r1) == len(r2) == 3, (len(r1), len(r2))

    # ③ 默认串行 + 并行分支已接线
    sig = inspect.signature(run_league.eval_round_robin)
    assert sig.parameters["n_workers"].default == 0, sig
    src = inspect.getsource(run_league.eval_round_robin)
    assert "_run_eval_pairs_parallel" in src, "并行分支未接线"
    assert run_league._eval_pair_worker_main.__module__ == "rl.run_league", \
        "worker 必须是模块级函数（spawn 需要可 pickle 的 target）"

    print("[PASS] run 模式评估并行：spec 往返逐位一致、n_games=1 并行==串行（Elo + 录像）、"
          "默认 n_workers=0 仍是串行、worker 为模块级函数")


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


def test_take_damage_signature_consistency():
    """引擎全体 `take_damage` 覆盖必须接受**调用点用到的全部关键字实参**。

    事故（2026-09-17）：`card_mechanics.dash_tick` 对突进目标调
    `take_damage(dmg, delayed=True, source=e)`，而 `TimedExplosive.take_damage(self, amount)`
    与 `GenericBomb.take_damage(self, amount)` 是"不受伤害"的 no-op 桩、**签名偏窄**
    ⇒ 突进把爆炸物当目标时 `TypeError: ... unexpected keyword argument 'delayed'` **直接崩训练**。
    solo 打固定 X弩牌不触发；`--mode run` 第一次上 200 副多卡组牌就崩（见
    `docs/run_mode_multideck_2026-09-17.md`）。另有 2 个桩（EvoEffectZone / VinesSnareZone）
    缺 `pierce_invincible`，被法术穿透那一支命中会同样崩。

    判据（静态、确定性、零成本）：AST 扫引擎源码，收集每个 `take_damage` 调用用到的关键字，
    再要求**每个定义都接受它们**（`**kwargs` 视为万能）。这是【R12】"能算的不许靠试"在 API 层
    的落地——比"跑一局碰运气"快且穷尽。
    """
    import ast
    import glob
    import io as _io
    import os as _os

    # T2-8：改用共用 `_PARENT`（= src/clasher_new）；自己算会因搬目录而少一层
    root = _PARENT
    files = [_os.path.join(root, "battle.py")] + sorted(glob.glob(_os.path.join(root, "rl", "*.py")))
    defs, used = [], {}
    for f in files:
        try:
            tree = ast.parse(_io.open(f, encoding="utf-8").read())
        except SyntaxError:
            continue
        cls_of = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for sub in node.body:
                    cls_of[id(sub)] = node.name
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "take_damage":
                defs.append((f, node.lineno, cls_of.get(id(node)),
                             [a.arg for a in node.args.args], bool(node.args.kwarg)))
            if isinstance(node, ast.Call):
                fn = node.func
                if (getattr(fn, "attr", None) or getattr(fn, "id", None)) == "take_damage":
                    for kw in node.keywords:
                        if kw.arg:
                            used.setdefault(kw.arg, []).append((f, node.lineno))
    assert defs, "没扫到任何 take_damage 定义 —— 扫描路径不对"
    assert used, "没扫到任何 take_damage 调用 —— 扫描路径不对"
    need = set(used)
    bad = []
    for f, ln, cls, params, has_kw in defs:
        if has_kw:
            continue
        missing = sorted(need - set(params))
        if missing:
            bad.append("%s:%d %s 缺 %s" % (_os.path.relpath(f, root), ln, cls or "-", missing))
    assert not bad, ("take_damage 签名与调用点不一致（会 TypeError 崩训练）：\n  " + "\n  ".join(bad))
    # 出现新的调用关键字时，本测试的语义要跟着复核（不是自动放行）
    assert need <= {"delayed", "source", "pierce_invincible"}, "出现未预期的 take_damage kwargs: %s" % sorted(need)
    print("    [ok] take_damage：%d 个定义 / %d 类 kwargs 全部兼容（%s）"
          % (len(defs), len(need), ", ".join(sorted(need))))


def test_noncombat_entity_contract():
    """非战斗实体契约（2026-09-17 事故，run 模式第一次上 200 副牌连撞两次崩溃）。

    背景：`AreaEffect` / `GenericBomb` / `EvoEffectZone` / `HealAuraZone` / `VinesSnareZone`
    **刻意不走 `Entity.__init__`**（无卡牌身份，避免同名机制类钩子），但仍 `class X(Entity)`。
    于是任何通用路径碰到它们都会对**不存在**的属性赋值/读取 ⇒ `AttributeError` 崩训练：
      - `battle.py::_pulse` 给范围内实体 `apply_buff(heal=…)` ⇒ `VinesSnareZone` 无 `regen_buffs`；
      - 突进 `card_mechanics::dash_tick` 对目标 `take_damage(delayed=…)` ⇒ `TimedExplosive`
        桩签名偏窄（该类其实走了 `Entity.__init__`，属同一族的"签名不一致"）。

    判据：
      A. 静态（双向）：battle.py 里"定义了 `__init__` 且不调 `super().__init__`/`Entity.__init__`"
         的类（Entity 自身除外）**必须**声明 `is_combat_entity = False`；
         反之声明了 `False` 的类**必须**真的跳过 `Entity.__init__`（防反向错标）。
      B. 动态：5 个类各造一个实例，把通用入口全调一遍（`apply_buff` 的四条分支 + `take_damage`
         全 kwargs）⇒ 必须**不抛异常**（就是 run 模式实测崩掉的那条路径）。
    """
    import ast
    import io as _io
    import os as _os

    # T2-8：改用共用 `_PARENT`（= src/clasher_new）；自己算会因搬目录而少一层
    root = _PARENT
    path = _os.path.join(root, "battle.py")
    src = _io.open(path, encoding="utf-8").read()
    tree = ast.parse(src)

    # 只考察 **Entity 的后代**（模块内直接/间接继承）；`_BombShim`/`BattleState` 这类
    # 数据垫片不是 Entity，不适用本契约（2026-09-17 首版扫描漏了这一步，被测试自身抓出）。
    bases_of = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            bases_of[node.name] = [b.id for b in node.bases if isinstance(b, ast.Name)]
    def _is_entity_desc(name):
        """向上找 Entity（bases_of 只含本模块的类；外部基类视为非 Entity）。"""
        todo, gone = [name], set()
        while todo:
            cur = todo.pop()
            if cur in gone:
                continue
            gone.add(cur)
            for b in bases_of.get(cur, []):
                if b == "Entity":
                    return True
                todo.append(b)
        return False

    skip_init, declared_false = set(), set()
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name == "Entity":
            continue
        if not any((b.id == "Entity") or _is_entity_desc(b.id)
                   for b in node.bases if isinstance(b, ast.Name)):
            continue
        for sub in node.body:
            if isinstance(sub, ast.Assign):
                for t in sub.targets:
                    if isinstance(t, ast.Name) and t.id == "is_combat_entity" \
                            and isinstance(sub.value, ast.Constant) and sub.value.value is False:
                        declared_false.add(node.name)
        inits = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"]
        if not inits:
            continue
        txt = ast.get_source_segment(src, inits[0]) or ""
        if ("super().__init__" not in txt) and ("Entity.__init__" not in txt):
            skip_init.add(node.name)

    missing = sorted(skip_init - declared_false)
    reverse = sorted(declared_false - skip_init)
    assert not missing, ("这些类跳过 Entity.__init__ 但没声明 is_combat_entity = False"
                        "（通用路径会 AttributeError 崩训练）：%s" % missing)
    assert not reverse, ("这些类声明了 is_combat_entity = False 但其实走了 Entity.__init__"
                        "（错标会让通用路径错误跳过真战斗实体）：%s" % reverse)
    assert declared_false, "一个非战斗实体都没扫到 —— 扫描逻辑不对"

    # B. 动态：通用入口对它们必须无异常（就是实测崩溃的那条路径）
    from battle import (AreaEffect, GenericBomb, EvoEffectZone, HealAuraZone, Position,
                        VinesSnareZone)
    pos = Position(9.0, 10.0)
    objs = [
        AreaEffect(9001, pos, 0, "Fireball", None),
        GenericBomb(9002, pos, 1, 100.0, 1.5, 0.5),
        EvoEffectZone(9003, pos, 0, None, 2.0, 5.0),
        HealAuraZone(9004, pos, 0, 2.0, 100.0),
        VinesSnareZone(9005, pos, 1, None, 2.0, 2.0, 153.0),
    ]
    for o in objs:
        assert getattr(o, "is_combat_entity", True) is False, type(o).__name__
        o.apply_buff(speed_mult=1.3, duration=1.0)          # Rage 分支
        o.apply_buff(hit_speed_mult=1.5, duration=1.0)      # M3 攻速槽
        o.apply_buff(damage_reduction=0.3, duration=1.0)    # Monk 减伤
        o.apply_buff(heal={"hps": 10.0, "time": 1.0})       # Heal（实测崩点）
        o.apply_buff(stun=1.0, retarget=True)               # Freeze/Zap 分支
        o.take_damage(10.0, delayed=True, source=None, pierce_invincible=False)
    print("    [ok] 非战斗实体契约：%d 个类声明 False 且跳过 Entity.__init__；"
          "5 个实例的通用入口全部无异常" % len(declared_false))


def test_death_spawn_routing_data_invariant():
    """通用亡语路由的数据不变量（2026-09-17 run 模式第三例崩溃）。

    `_generic_death_spawn` 把"带 `deathDamage` 且无 `hitpoints`"的卡交给 `TimedExplosive`，
    而 `TimedExplosiveData.__init__` 硬读 `name / deathDamage / deployTime / collisionRadius`。
    `SkeletonBalloon` 的容器 dsd **同样带 `deathDamage` 却没有 `collisionRadius`**
    （它是"0.6s 后出 7 骷髅"的容器，同一个函数里有专门分支）⇒ 被误判成炸弹，
    `KeyError: 'collisionRadius'` 直接崩训练。

    判据（数据驱动、穷举全卡表）：
      A. 对所有**满足路由判定** `battle.is_death_bomb(dsd)` 的卡，`TimedExplosiveData(dsd)`
         必须能构造成功（= 键齐全）；
      B. `SkeletonBalloon` 必须**不**被判为炸弹（它应走容器分支）。
    """
    from battle import is_death_bomb
    from card_utils import card_data, TimedExplosiveData

    rows = [(n, r) for n, r in card_data.items() if isinstance(r, dict)]
    assert rows, "卡表为空 —— 数据加载不对"
    bombs, bad = [], []
    for name, row in rows:
        dsd = (row.get('summonCharacterData') or {}).get('deathSpawnCharacterData') or {}
        if not dsd or not is_death_bomb(dsd):
            continue
        bombs.append(name)
        try:
            TimedExplosiveData(dsd)
        except Exception as e:                     # noqa: BLE001 - 要的就是任何异常
            bad.append("%s: %s" % (name, e))
    assert not bad, ("被判为亡语炸弹、但 TimedExplosiveData 解析不了（会 KeyError 崩训练）：\n  "
                     + "\n  ".join(bad))
    assert bombs, "一张炸弹卡都没扫到 —— 路由判定或卡表不对"
    sb = card_data.get('SkeletonBalloon')
    if isinstance(sb, dict):
        dsd = (sb.get('summonCharacterData') or {}).get('deathSpawnCharacterData') or {}
        assert not is_death_bomb(dsd), \
            "SkeletonBalloon 被判成炸弹了 —— 它应走容器分支（0.6s 后出 7 骷髅），否则漏掉容器语义"
    print("    [ok] 亡语炸弹路由：%d 张（%s）全部可被 TimedExplosiveData 解析；"
          "SkeletonBalloon 走容器分支" % (len(bombs), ", ".join(bombs)))


def test_normalize_dir_zero_vector():
    """方向向量归一化：**零向量必须返回 None**（而不是除零崩溃）。

    2026-09-17 run 模式第四例崩溃：滚动弹（BarbLog 类）按
    `target_position - initial_position` 归一化行进方向，起点与终点重合时
    `direction_vector /= abs(direction_vector)` ⇒ `ZeroDivisionError: complex division by zero`。
    同一个文件里"击退"与"推挤"两处各自写了零向量守卫，唯独这一处漏了 ⇒ 抽 `normalize_dir`
    做单一来源（`battle.py::normalize_dir`）并在此固定语义。
    """
    from battle import normalize_dir

    assert normalize_dir(0.0, 0.0) is None, "零向量必须返回 None"
    assert normalize_dir(1e-12, -1e-12) is None, "数值等价于零的向量也必须返回 None"
    v = normalize_dir(3.0, 4.0)
    assert v is not None and abs(v - complex(0.6, 0.8)) < 1e-12, v
    assert abs(abs(normalize_dir(-5.0, 0.0)) - 1.0) < 1e-12
    assert abs(abs(normalize_dir(0.0, -2.5)) - 1.0) < 1e-12
    print("    [ok] normalize_dir：零向量→None（不除零）；非零向量单位化正确")


def test_mask_partial_bundle_invariants():
    """掩码不变式（2026-09-18 回归）：**已用槽位不得再被判合法**。

    病灶：`RLEnv.get_action_mask_for` 里 `used` 存的是 0-based 下标（`sa.slot - 1`），
    而成员测试写成 `sa.slot not in used`（1-based）⇒ partial bundle 一旦出现「(s, s−1)」
    这种**相邻降序对**，低槽位 `s−1` 会被整段跳过：既漏记 `used`（掩码放行**重复槽位**
    ⇒ `validate_bundle` **整包拒绝** ⇒ 白掉一帧 + 吃 `invalid_penalty`），又漏扣该卡费用。
    实测在 `et_solo100k` 两臂录像里造成 **497 帧非法包**（最长**连续 225 帧**卡死）——
    取证 `docs/mask_used_slot_offbyone_fix_2026-09-18.md`。

    本测试固定四条不变式（**跨局边界**，【R8】）：
      ① 出现过的槽位 ⇒ `mask["slots"]` 必须为 False（枚举全部 1/2/3 元**有序**组合）；
      ② `mask["used_slots"]` 与 partial 的槽位集合**逐位一致**（0-based）；
      ③ 掩码里每个仍合法的槽位 ⇔ 卡费 ≤ 剩余圣水（与 `validate_bundle` 同口径、不重复扣费）；
      ④ 采样器产出的**每个** bundle 都通过 `validate_bundle`，且每一步选中的槽位在
         该步掩码里确实合法（跨 2 局 × sample/argmax 两分支）。
    """
    import itertools
    from rl.action_bundle import ActionBundle, K_MAX
    from rl.action_mask import validate_bundle
    from rl.env_wrapper import RLEnv
    from rl.train_solo import DEFAULT_SOLO_DECK

    env = RLEnv(opponent=None, seed=3, card_level=11,
                deck0=list(DEFAULT_SOLO_DECK), deck1=list(DEFAULT_SOLO_DECK))
    n_partial = 0
    for ep in range(2):                                  # R8：跨局边界
        env.reset(seed=100 + ep)
        p = env.battle.players[0]
        for n in (1, 2, 3):
            for combo in itertools.permutations(range(1, K_MAX + 1), n):
                b = ActionBundle()
                for s in combo:
                    b.add(s, 8, 20)
                m = env.get_action_mask(b)
                bad = [s for s in combo if bool(m["slots"][s - 1])]
                assert not bad, (f"局{ep} partial={combo}: 已用槽位仍被判合法 {bad}"
                                 f"（这正是 off-by-one 复发的指纹）")
                used = {s - 1 for s in combo}
                assert set(np.asarray(m["used_slots"]).tolist()) == used, \
                    f"局{ep} partial={combo}: used_slots={m['used_slots']} != {sorted(used)}"
                expect = max(0.0, p.elixir - sum(Card(p.cycle[s - 1]).elixir for s in combo))
                for i in range(K_MAX):
                    if i in used:
                        continue
                    cost = Card(p.cycle[i]).elixir
                    assert bool(m["slots"][i]) == (cost <= expect + 1e-9), (
                        f"局{ep} partial={combo}: 槽{i + 1} 合法性 {bool(m['slots'][i])} "
                        f"与费用口径不符（费 {cost} / 剩余 {expect:.2f}）")
                n_partial += 1

    from rl.belief import BeliefInference
    from rl.belief_planner import BeliefPlanner
    from rl.follower import FollowerPolicy
    from rl.plan_space import PLAN_DIM
    bd = len(BeliefInference(opp_deck=list(DEFAULT_SOLO_DECK), n_particles=32,
                             seed=0).encode(None, None))
    pol = FollowerPolicy(hidden=32, plan_dim=PLAN_DIM, belief_dim=bd).to_device("cpu")
    pol.eval()
    bp = BeliefPlanner()
    frames = 0
    cell_gap = 0
    for ep in range(2):
        obs, _ = env.reset(seed=900 + ep)
        belief = BeliefInference(opp_deck=env.deck1, n_particles=32, seed=900 + ep)
        hidden = None
        for k in range(40):
            plan = bp.plan(env.battle, belief.state(), obs)
            tok = belief.encode(obs, None)
            bundle, _lp, _v, hidden, masks = pol.act(
                obs, tok, plan.to_vector(), env.get_action_mask,
                hidden=hidden, deterministic=(k % 2 == 1))
            for j, sa in enumerate(bundle.sub_actions):
                if sa.kind == "deploy" and j < len(masks):
                    assert bool(masks[j]["slots"][sa.slot - 1]), (
                        f"局{ep} 帧{k} decoder步{j}: 选中的槽位 {sa.slot} 在该步掩码里非法")
            ok, reason, _res = validate_bundle(env.battle, 0, bundle)
            if not ok:
                if "位置非法" in str(reason):
                    cell_gap += 1                      # 另一类缺口，见打印（本测试不掩盖）
                else:
                    raise AssertionError(
                        f"局{ep} 帧{k}: 采样产出非法包（{reason}）"
                        f"bundle={[(s.kind, s.slot, s.x, s.y) for s in bundle.sub_actions]}")
            obs, _r, term, trunc, info = env.step(bundle)
            belief.update(obs, info.get("opp_played"))
            frames += 1
            if term or trunc:
                break
    assert frames >= 20, f"采样帧数太少（{frames}），不变式覆盖不足"
    print(f"    [ok] 掩码不变式：{n_partial} 组 partial 逐位一致（含 (s,s-1) 降序对）；"
          f"{frames} 帧采样 bundle 全部通过 validate_bundle"
          + (f"；⚠️ 另有 {cell_gap} 帧因『部署位置非法』被拒（掩码-格子行缺口，另案）"
             if cell_gap else ""))


def test_intent_save_mechanism():
    """攒费意图动作（intent-save，2026-09-19 用户拍板扩参）：状态机 + 默认关逐位不变 + 跨局。

    预注册 `docs/intent_save_prereg_2026-09-19.md`。覆盖：
      ① **默认关**：观测 dict 无意图键、`step` 不触碰意图状态、`info` 无 `"intent"`；
      ② **单变量**：**无 pending** 时开/关两臂的 `slots`/`cells`/`ability_legal`/`at_cap`/
         `used_slots` **逐位相同** ⇒ 打开开关只在"有意图"时才改变掩码；
      ③ **架构**：关 = 6 option / scalar 3；开 = 11 option（**尾部追加**）/ scalar 9，
         参数增量 == slot_head 行 + sub_emb 列 + enc_fc 列（逐项复算，非手抄）；
      ④ **状态机**：SAVE ⇒ 承诺期（slots 全 False、ability False、CANCEL 可用）；
         纯 STOP **保持不需要重抽**（age 递增）；攒够 ⇒ ready + slots 恢复；真打出 ⇒ fired
         且记录**当时已保持帧数**（判据 J1 的数据源）；CANCEL ⇒ 清空；
      ⑤ **【R8】跨局**：`reset()` 清当前 pending，但**累计计数保留**（run 级取证量）。
    """
    from rl.env_wrapper import RLEnv
    from rl.action_bundle import ActionBundle, K_MAX
    from rl.action_mask import _card_cost
    from rl.follower import FollowerPolicy

    e0 = RLEnv(opponent=None, seed=1, intent_save=False); e0.reset()
    e1 = RLEnv(opponent=None, seed=1, intent_save=True); e1.reset()
    # ① 默认关
    o0, o1 = e0.observe(0), e1.observe(0)
    assert "intent_slot" not in o0 and "intent_age" not in o0, "默认关不得改观测 dict"
    assert "intent_slot" in o1 and "intent_age" in o1
    # ② 单变量（无 pending：这里 e1 也还没下意图；★ 两臂必须在同一状态下比掩码）
    e0.battle.players[0].elixir = 1.0
    e1.battle.players[0].elixir = 1.0
    m0, m1 = e0.get_action_mask_for(0), e1.get_action_mask_for(0)
    for k in ("slots", "cells", "ability_legal", "at_cap", "used_slots"):
        assert np.array_equal(np.asarray(m0[k]), np.asarray(m1[k])),             f"无 pending 时掩码 key {k} 必须逐位相同（单变量）"
    assert set(m1) - set(m0) == {"intent_slots", "intent_cancel", "intent_hold"}
    # ①默认关的 step 检查放在②之后：两侧必须在**同一状态**下比掩码
    e0.battle.players[0].elixir = 1.0
    _, _, _, _, info0 = e0.step(ActionBundle.noop())
    assert "intent" not in info0, "默认关不得出现 intent 取证字段"
    assert e0.intent_stats["set"] == 0 and e0._intent_slot == 0
    # ③ 架构 + 参数增量（逐项复算）
    h = 128
    p_off = FollowerPolicy(hidden=h, plan_dim=58, belief_dim=563, intent_options=False)
    p_on = FollowerPolicy(hidden=h, plan_dim=58, belief_dim=563, intent_options=True)
    n_off = sum(x.numel() for x in p_off.parameters())
    n_on = sum(x.numel() for x in p_on.parameters())
    assert p_off.num_slot_options == 6 and p_on.num_slot_options == 11
    assert p_off.scalar_dim == 3 and p_on.scalar_dim == 9
    assert tuple(p_on.slot_head.weight.shape) == (11, h)
    assert tuple(p_on.sub_emb.weight.shape) == (h, 13)
    _expect = (5 * h + 5) + (5 * h) + (6 * h)      # slot_head 行 + sub_emb 列 + enc_fc 列
    assert n_on - n_off == _expect, f"参数增量 {n_on - n_off} != 复算 {_expect}"
    # ④ 状态机
    p = e1.battle.players[0]
    _un = [i for i in range(K_MAX) if (_card_cost(p, p.cycle[i]) or 0) > p.elixir]
    assert _un, "圣水 1.0 时应至少有一张买不起的牌"
    i = _un[0]
    assert m1["intent_slots"][i] and not m1["slots"][i], "买不起的槽应可下攒费意图"
    env = e1
    _, _, _, _, info = env.step(ActionBundle.intent_save(i + 1))
    assert info["intent"]["slot"] == i + 1 and info["intent"]["stats"]["set"] == 1
    mh = env.get_action_mask_for(0)
    assert mh["intent_hold"] and not mh["slots"].any() and not mh["ability_legal"], \
        "承诺期必须压制一切花费（否则 k 帧合取 p^k 采样不到）"
    assert mh["intent_cancel"], "承诺期必须可 CANCEL（用户要的『中途改变想法』）"
    _, _, _, _, info = env.step(ActionBundle.noop())          # ★ 保持：不重抽意图
    assert info["intent"]["age"] == 1 and info["intent"]["stats"]["held"] == 1
    # ★ 回归（2026-09-19 修）：**同一目标的再次表态不得重置年龄**。原实现无条件
    # `age=0`，而承诺期掩码**同时放行 SAVE** ⇒ 策略每重述一次年龄就归零 ⇒
    # 「同一张卡被 pending 覆盖 >=34 帧」这条 J1 语义**机械上不可能成立**。
    _set_before = info["intent"]["stats"]["set"]
    _, _, _, _, info = env.step(ActionBundle.intent_save(i + 1))
    assert info["intent"]["age"] == 2, f"同目标重述必须续龄，实测 age={info['intent']['age']}"
    assert info["intent"]["stats"]["set"] == _set_before, "同目标重述不算新 set"
    assert info["intent"]["event"].get("intent_reassert") == 1
    assert info["intent"]["stats"]["held"] == 2
    env.battle.players[0].elixir = 10.0
    mr = env.get_action_mask_for(0)
    assert not mr["intent_hold"] and mr["slots"][i], "攒够后必须恢复正常可出牌"
    _, _, _, _, info = env.step(ActionBundle.noop())
    assert info["intent"]["stats"]["ready"] == 1 and info["intent"]["slot"] == i + 1
    cells = env.get_action_mask_for(0)["cells"][i]
    _y, _x = np.unravel_index(int(np.argmax(cells)), cells.shape)
    _, _, _, _, info = env.step(ActionBundle.from_single(i + 1, int(_x), int(_y)))
    assert info["intent"]["stats"]["fired"] == 1
    assert info["intent"]["stats"]["fired_held"] == [2], \
        "fired 必须记录当时已保持帧数（J1 数据；含同目标重述那一帧）"
    # J1 原文限定的前提：事件必须带**目标卡与费用**（否则 3 费卡被抱 34 帧会误算 PASS）
    _ev = info["intent"]["event"]
    assert _ev.get("intent_fired_card") and _ev.get("intent_fired_cost") is not None
    assert _ev["intent_fired_cost"] == _card_cost(env.battle.players[0], _ev["intent_fired_card"])
    # ⑤ 跨局（【R8】）：pending 清空、累计保留
    _set_before = env.intent_stats["set"]
    env.reset()
    assert env._intent_slot == 0 and env._intent_age == 0
    assert env.intent_stats["set"] == _set_before, "累计计数必须跨局保留（run 级取证量）"
    # CANCEL 路径
    ps = env.battle.players[0]; ps.elixir = 1.0
    _j = [k for k in range(K_MAX) if (_card_cost(ps, ps.cycle[k]) or 0) > ps.elixir][0]
    env.step(ActionBundle.intent_save(_j + 1))
    assert env._intent_slot == _j + 1
    env.step(ActionBundle.intent_cancel())
    assert env._intent_slot == 0 and env.intent_stats["cancelled"] == 1
    # 回归（2026-09-19，端到端冒烟抓到的真事故）：`act()` 的分支里引用了 `INTENT_CANCEL`，
    # 而子集测试是**直达** `ActionBundle.intent_cancel()` 的 ⇒ 绕过 act() 那条分支，
    # 曾出现「selftest 全绿但一采样到 CANCEL 就 NameError」。（【R8】每个修复配回归测试）
    from rl import follower as _f
    assert hasattr(_f, "INTENT_CANCEL"), "act() 会引用 follower.INTENT_CANCEL（曾 NameError）"
    assert _f.FollowerPolicy.terminal_option(ActionBundle.intent_cancel()) == _f.CANCEL_IDX
    assert _f.FollowerPolicy.terminal_option(ActionBundle.intent_save(2)) == _f.INTENT_SAVE_BASE + 1
    print(f"    [ok] intent-save：默认关逐位不变；无 pending 时掩码逐位相同；"
          f"开=11 option/scalar 9（参数 +{_expect}，逐项复算）；"
          f"SAVE→承诺期压制花费→保持不重抽→ready→fired(held=1)→CANCEL；跨局 pending 清空/计数保留")
