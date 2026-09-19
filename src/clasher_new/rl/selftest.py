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


# ⚠️ **必须在任何 `from rl.` 之前**：以 `python rl/selftest.py` 运行时 `sys.path[0]` 是 `rl/`
# （不是 `src/clasher_new`）⇒ 不先插路径，下一行就 `ModuleNotFoundError: No module named 'rl'`。
# 旧文件也是靠这一段（它当时写在本文件里）；拆分后这段**只能留在这里**。
import os as _os
import sys as _sys
_PARENT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _PARENT not in _sys.path:
    _sys.path.insert(0, _PARENT)

# T2-8（2026-09-19）：本文件已拆为「**聚合 + main()**」。
#   共用底座 → `rl/selftest_common.py`；100 个测试按定义序 → `rl/selftests/part{1..5}.py`。
#   这里**显式重导出**全部公开/私有名字 ⇒ `dir(rl.selftest)` 仍看得到 100 个 `test_*`
#   （`scripts/run_selftests.py` 的 `dir()` 反射、`--list`、`--order-check` 都靠这条）。
# ⚠️ `main()` 的 100 行调用清单**逐字未改**（拆分前后 AST 对账一致）；
#   唯一改动是 `_instrument_tests()` → `_instrument_tests(globals())`（否则它包装不到任何测试）。
from rl.selftest_common import (  # noqa: F401
    os, sys, time, random, shutil, np, Card, _PARENT,
    _mark_skip, _instrument_tests, _report_tests, discover_tests, register_namespace,
    _make_policy_and_tokens, _mk_env, _intents, _tiny_rollout_transitions, _FakeCfg,
)

from rl.selftests.part1 import *  # noqa: F401,F403
from rl.selftests.part2 import *  # noqa: F401,F403
from rl.selftests.part3 import *  # noqa: F401,F403
from rl.selftests.part4 import *  # noqa: F401,F403
from rl.selftests.part5 import *  # noqa: F401,F403

# T2-8：**导入期**把本模块交给共用底座的计时包装器。两个作用：
#   ① `discover_tests()` / `--order-check` 不必先跑 `main()` 就能拿到全集与定义序（原先靠 main() 副作用）；
#   ② 聚合模块是**唯一**能看到 100 个 `test_*` 的地方（各 part 的 globals 彼此独立）。
register_namespace(globals())
_instrument_tests(globals())


def main():
    # 与 run_league.main 同一兜底：日志含中文/emoji，Windows cp936 管道会崩
    from rl.run_league import _force_utf8_stdout
    _force_utf8_stdout()
    # T1-3/T1-5：把下面的 `test_*` 换成计时/计数包装。**只是想下面那 100 行一行都别动** ——
    # 顺序与内容即全量路径的契约（外部调用方 `scripts/_apply_s2_channel_when_idle.sh:70-71`）。
    _instrument_tests(globals())   # 已在导入期做过（幂等）；留着是为了让 main() 自解释
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
    test_dashboard_league_payload()
    test_battle_clone_fix()
    test_take_damage_signature_consistency()
    test_noncombat_entity_contract()
    test_death_spawn_routing_data_invariant()
    test_normalize_dir_zero_vector()
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
    test_draw_rule_tower_hp_total()
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
    test_eval_scheduler_two_tier()
    test_eval_round_robin_parallel_equivalence()
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
    test_mask_partial_bundle_invariants()
    test_intent_save_mechanism()
    test_intent_audit_deck_invariant()
    # T1-3/T1-5：以上 100 行**逐字未改**（顺序即契约）；这里只汇总计数/耗时/跳过清单。
    _report_tests()
    print("\nALL SELFTESTS PASSED")


if __name__ == "__main__":
    main()
