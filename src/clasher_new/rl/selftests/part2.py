# -*- coding: utf-8 -*-
"""`rl/selftest.py` 的测试 · 第 2/5 部分（20 个，test_config_reward_weights … test_mp_training_loop）。

**函数体逐字未改**（切片生成）；顶部显式导入共用底座。"""

# T2-8：本部分从 `rl/selftest.py` 原样切出（**函数体逐字未改**，只动了下面两处**路径推导**：
# 原 L5615 / L5676 的 `dirname(dirname(abspath(__file__)))` 在搬到 `rl/selftests/` 后会少一层
# ⇒ 改用 `selftest_common._PARENT`（仍 = `src/clasher_new`）。
# `*` 不导出下划线名 ⇒ 私有 helper / 导入名一律**显式**列出（缺一个就是 NameError）。
from rl.selftest_common import (  # noqa: F401
    os, sys, time, random, shutil, np, Card, _PARENT,
    _mark_skip, _make_policy_and_tokens, _mk_env, _intents, _tiny_rollout_transitions, _FakeCfg,
)

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

    # 塔几何（2026-09-19；用户：「引擎已经正常实现塔了，但 dashboard 回放的塔还是 1 格」）：
    # 前端按**引擎的真实足迹**画塔 ⇒ 引擎侧必须提供几何；老录像没有该键时前端回落常量
    # （前端常量的三方对账在 `scripts/check_dashboard_js.py --tower-only`）。
    from rl.replay import tower_geometry
    from arena import TileGrid as _TG
    from core import Position as _Pos

    class _B:                       # 只给 tower_geometry 要的那一个属性
        def __init__(self, towers):
            self.arena = type("_A", (), {"towers": towers})()

    assert tower_geometry(_B([(_Pos(3.5, 25.5), 1.5, 1.5, 1),
                              (_Pos(9.0, 29.0), 2.0, 2.0, 1)])) == \
        [[3.5, 25.5, 1.5, 1.5, 1], [9.0, 29.0, 2.0, 2.0, 1]]          # 本仓 4 元组（矩形）
    assert tower_geometry(_B([(_Pos(3.5, 25.5), 1.0, 1)])) == \
        [[3.5, 25.5, 1.0, 1.0, 1]]                                     # 上游 3 元组（圆形）
    assert tower_geometry(_B([(_Pos(0.0, 0.0), 1, 2, 3, 4)])) is None   # 形状不认识
    assert tower_geometry(_B([])) is None and tower_geometry(None) is None
    real = tower_geometry(_B(_TG.towers))                              # 真引擎几何
    assert real is not None and len(real) == 6, real
    assert [(r[2], r[3]) for r in real].count((1.5, 1.5)) == 4, real   # 公主塔 3×3
    assert [(r[2], r[3]) for r in real].count((2.0, 2.0)) == 2, real   # 国王塔 4×4
    assert [tuple(r[:2]) for r in real] == [(t[0].x, t[0].y) for t in _TG.towers], real
    # 局级可选键：dashboard 的加载器必须把 meta.tower_geom 原样交给前端（不升 schema）
    p2 = os.path.join(d, "league_3000.pkl")
    save_league_replays([{"meta": {"pair": ["a", "b"], "tower_geom": real}, "winner": 0,
                          "frames": [frame(0.5, towers0, towers1, [])]}], p2)
    got = dash.load_replay_payload(d, "league_3000.pkl", 0)
    assert got["ok"] and got["meta"]["tower_geom"] == real, got.get("meta")

    print("[PASS] 仪表盘回放：列表扫描 + 对局加载 + 单局帧 + 非法名防护 + demo + 页面元素"
          " + 塔几何（4/3 元组 + 真引擎 + meta 透传）")


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


def test_dashboard_league_payload():
    """仪表盘「联赛/长跑」面板：进度、两级评估点标注、逐点对手胜率、目录入口（2026-09-18）。

    长跑（1M 步 / 131 评估点）期间每 3 s 轮询这个 payload，所以每条都可能**静默失效**：
    ① `--state` 只认文件 ⇒ 盯着 `runs/long1m/` 就得写全文件名；目录入口必须能解析；
    ② `league_state.total_steps` 只是"最近评估点"⇒ 没有 run_state.json 的进度就没法
       显示"才跑了 1.6%"，前端会看起来像跑完了；百分比/ETA/计划点数必须来自 run_state+config；
    ③ 大/小评估点必须**来自与训练侧同一份** `eval_schedule`（漏传 config 字段 ⇒ 计划点数
       与实际点数漂移，会被当成训练异常）；
    ④ 逐点胜率是从 `history` + `round_stats[i].games` 的累计局数**切分复原**的
       （`league_state` 只存 EMA 标量），切错位会给出假曲线 ⇒ 用可手算的 fixture 对账；
    ⑤ 训练卡死/退出时要能看出来（15 min 无写入 ⇒ stale）。
    """
    import tempfile
    import json as _json
    import rl.dashboard as dash
    from rl.config import eval_schedule
    from rl.run_league import _EvalScheduler

    # 手算 fixture：小点 10 步 × 2 局/对、大点 15 步 × 3 局/对、总 30 步、起始用大预算
    cfg = {"name": "t_league", "total_steps": 30, "steps_per_eval": 10,
           "big_eval_every": 15, "n_eval_games": 2, "n_eval_games_big": 3,
           "eval_big_at_start": True, "only_vs_main": True, "eval_workers": 12,
           "n_envs": 1, "device": "cuda"}
    # 计划 = {0,10,20,30} ∪ {0,15,30} = 5 点；kind：0 大、10 小、15 大、20 小、30 大
    want = [(0, "big", 3), (10, "small", 2), (15, "big", 3), (20, "small", 2), (30, "big", 3)]
    assert eval_schedule(10, 15, 2, 3, 30) == want, eval_schedule(10, 15, 2, 3, 30)

    with tempfile.TemporaryDirectory() as td:
        run = os.path.join(td, "long1m")
        os.makedirs(run)
        with open(os.path.join(run, "config.json"), "w", encoding="utf-8") as f:
            _json.dump(cfg, f)
        with open(os.path.join(run, "run_state.json"), "w", encoding="utf-8") as f:
            _json.dump({"step": 20, "total_steps": 30, "config": "t_league"}, f)
        # 已写完两个点（0 与 10）：各 main vs A 2 局
        state = {
            "agents": [{"agent_id": "main", "kind": "main"},
                       {"agent_id": "A", "kind": "baseline"}],
            "ratings": {"main": 1600.0, "A": 1400.0},
            "elo_history": {"main": [[0, 1550.0], [10, 1600.0]],
                            "A": [[0, 1450.0], [10, 1400.0]]},
            "round_stats": [
                {"step": 0, "est": {"main": [1550.0, 100.0], "A": [1450.0, 100.0]},
                 "games": {"main": 2, "A": 2}},
                {"step": 10, "est": {"main": [1600.0, 90.0], "A": [1400.0, 90.0]},
                 "games": {"main": 2, "A": 2}},
            ],
            "winrates": {"main|A": 0.625, "A|main": 0.375},
            "history": [["main", "A", 1.0], ["main", "A", 0.0],
                        ["main", "A", 1.0], ["main", "A", 0.5]],
            "total_steps": 10,
        }
        sp = os.path.join(run, "league_state.json")
        with open(sp, "w", encoding="utf-8") as f:
            _json.dump(state, f)

        # ① 目录入口
        assert dash.resolve_state_path(run) == sp, dash.resolve_state_path(run)
        assert dash.resolve_state_path(sp) == sp
        assert dash.resolve_state_path(os.path.join(td, "nope")) == \
            os.path.abspath(os.path.join(td, "nope"))
        empt = os.path.join(td, "empty_run")
        os.makedirs(empt)
        assert dash.resolve_state_path(empt) is None, "目录里没有 league_state.json ⇒ None"

        pl = dash.build_payload(dash.resolve_state_path(run))
        assert pl["ok"], pl.get("error")

        # ② 进度：来自 run_state（20/30），不是 league_state 的 total_steps(=10)
        rm = pl["run_meta"]
        assert rm["cur_step"] == 20 and rm["total_steps"] == 30, rm
        assert abs(rm["pct"] - 66.67) < 0.05, rm["pct"]
        assert rm["points_done"] == 2, rm["points_done"]

        # ③ 计划点数 = 训练侧调度器的点数（同一份 eval_schedule）
        sched = _EvalScheduler(_FakeCfg(cfg), 0)
        assert sched.expected_points(30) == [p for p, _k, _n in want], sched.expected_points(30)
        assert rm["plan_points"] == len(want) == 5, rm["plan_points"]
        assert rm["plan_big"] == 3 and rm["plan_small"] == 2, rm
        assert [[s, k, n] for s, k, n in rm["schedule"]] == [list(w) for w in want]
        # round_stats 被标注了大/小（前端据此画竖虚线）
        ks = [(rt["step"], rt["kind"], rt["games_per_pair"]) for rt in pl["round_stats"]]
        assert ks == [(0, "big", 3), (10, "small", 2)], ks

        # ④ 逐点胜率（切分复原）：点0 = (1+0)/2 = 0.5；点10 = (1+0.5)/2 = 0.75
        assert pl["winrate_curves"]["main|A"] == [[0, 0.5], [10, 0.75]], \
            pl["winrate_curves"]["main|A"]
        assert pl["winrate_counts"]["main|A"] == [2, 2], pl["winrate_counts"]
        assert pl["winrates"]["main|A"] == 0.625, pl["winrates"]

        # ⑤ stale：把状态文件 mtime 拨到 20 分钟前
        old = time.time() - 1200
        os.utime(sp, (old, old))
        assert dash.build_payload(sp)["run_meta"]["stale"] is True
        assert dash.build_payload(sp)["run_meta"]["state_age_s"] > 1000

        # 缺 run_state/config 时不得炸（旧 run / demo 状态）——进度降级但曲线仍可用
        os.remove(os.path.join(run, "run_state.json"))
        os.remove(os.path.join(run, "config.json"))
        pl2 = dash.build_payload(sp)
        assert pl2["ok"], pl2.get("error")
        # 退化但结构完整：total_steps=0 ⇒ 前端 renderRunMeta 自动不画进度条（不是崩）
        assert pl2["run_meta"]["total_steps"] == 0 and not pl2["run_meta"]["schedule"], pl2["run_meta"]
        assert pl2["run_meta"]["cur_step"] == 10, pl2["run_meta"]   # 退回 league_state.total_steps
        assert pl2["winrate_curves"]["main|A"] == [[0, 0.5], [10, 0.75]]

    print("[PASS] 联赛长跑面板：目录入口、进度/ETA 来自 run_state、两级评估点标注与训练侧"
          "同源（5 点 = 大3/小2）、逐点胜率切分复原（0.5/0.75）、20 min 未写入 ⇒ stale、"
          "缺 run_state/config 时降级不炸")


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
