# `scripts/` 分类索引（**自动生成，勿手改**）

> 生成器 / 校验器：`scripts/_structure_check.py`（`--write-scripts-readme` 重写；检查 **⑩** 防漂移）。
> 逐文件的 docstring / `--selftest` / assert 数 / 是否已被 `env.md` 登记 ⇒ 见
> [`../docs/agents/scripts_inventory.md`](../docs/agents/scripts_inventory.md)。

## ★ 为什么**没有**把 `scripts/` 拆成子目录（Tier 2 · T2-6 的决定）

方案原写「`scripts/` 分目录（tools / probes / tests）」。**实测代价后判定不做**，理由是数字：

| 家族 | 文件数 | 文档提及 | 其中活跃索引 | 涉及**历史留证**文档数 |
|---|---:|---:|---:|---:|
| `test_m*.py` | 6 | 193 | 4 | 13 |
| `_survey_*.py` | 9 | 128 | 0 | 8 |
| 其它 `_*.py` | 13 | 138 | 15 | 36 |
| `probe_/diag_/forensics_*` | 30 | 264 | 30 | 59 |
| 其它 | 35 | 552 | 75 | 72 |

- **全仓 `scripts/<名>.py` 被提及 1,293 次 / 113 份文档**，其中 **~1,167 次落在 `docs/*.md` 的历史留证文档里**；
- 本仓纪律是「新决策先写 docs（可改）」+「**只增不改历史结论**」⇒ 搬迁必然要么**漏改**（文档说谎），
  要么**改写历史**（违反纪律）。**即使只搬最小的家族（`_survey_*` 9 个）也要碰 8 份历史文档**；
- **功能面其实很小**：`.sh` 5 处（`finalize_et_solo100k.sh` / `run_probe_v3.sh` / `_apply_s2_channel_when_idle.sh`）、
  `.bat` 只引用 `scripts/rl/*`（方案本就要求**保持原位**）、`.ps1` 1 处且是散文 ⇒ **风险不在功能面，全在文档面**；
- 而 T2-6 想要的「可寻性」已经由 **T0-3 的 `docs/agents/scripts_inventory.md`**（91 个脚本 + docstring + 4 分类）
  与本文件（分类 + 漂移检查）**给到**，无需搬文件。

> 若将来确实要搬：**必须同批**改 `env.md §2.4` / `AGENTS.md` / `docs/README.md` / 3 个 `.sh` /
> `docs/agents/scripts_inventory.md`，并在 98 份历史文档**顶部加一行「路径已迁移」备注**（而不是改写正文）。

## 分类

### ① 测试（引擎验收）（6 个）

引擎机制验收脚本（`test_m2/m3_evo/m4_evo7/m5_data/m6_elite/m1`）。**不属于** `run_selftests.py` 那套。

`test_m1.py`, `test_m2.py`, `test_m3_evo.py`, `test_m4_evo7.py`, `test_m5_data.py`, `test_m6_elite.py`

### ② 一次性取证 / 归档工具（`_` 前缀）（28 个）

按仓内既有约定：`_` = 不进正式仪器表；多为**一次性**取证/迁移工具，**保留作证据**。

`_agents_split.py`, `_astar_cost_compare.py`, `_converge_utf8_bootstrap.py`, `_fl_il_extract.py`, `_forensics_cycling.py`, `_il_replay_extract.py`, `_il_replay_format_scan.py`, `_mask_ab_prefix.py`, `_mask_diff_snapshot.py`, `_mask_vs_engine_reconcile.py`, `_patch_battle_root_cast.py`, `_path_shape_probe.py`, `_probe_eval_frame_ab.py`, `_probe_value_collapse.py`, `_probe_value_path.py`, `_schema5_probe.py`, `_skarmy_spawn_probe.py`, `_structure_check.py`, `_survey_audit.py`, `_survey_brief.py`, `_survey_groups.py`, `_survey_inventory.py`, `_survey_md_to_docx.py`, `_survey_merge.py`, `_survey_merge_docs.py`, `_survey_reverse_check.py`, `_survey_verify.py`, `_verify_fix_replays.py`

### ③ 探针 / 诊断（可复用仪器）（69 个）

可复跑的判读/诊断仪器（多数已在 `docs/agents/env.md` §2.4 登记）。

`analyze_draw_anatomy.py`, `analyze_online_trade.py`, `batch_smoke.py`, `bench_train_speed.py`, `check_commit.py`, `check_dashboard_js.py`, `coverage.py`, `diag_critic_ev.py`, `diag_encoder_scale.py`, `diag_gru_ablation.py`, `diag_value_head.py`, `duel_search.py`, `forensics_card_usage.py`, `forensics_response.py`, `health_curve.py`, `judge_anchor_blocks.py`, `judge_critic_inertia.py`, `judge_probe_v3.py`, `judge_probe_v4.py`, `offline_engagement_trade.py`, `pass_streak_audit.py`, `phi_offline_check.py`, `pomdp_ceiling_probe.py`, `probe_belief_accuracy.py`, `probe_channel_gradient.py`, `probe_collision_block.py`, `probe_collision_order.py`, `probe_collision_trap.py`, `probe_credit_baseline.py`, `probe_emb_capacity.py`, `probe_encoding.py`, `probe_engine_profile.py`, `probe_entropy_decomp.py`, `probe_explore_randomization.py`, `probe_formation.py`, `probe_gae_kernel.py`, `probe_grid_collision_loss.py`, `probe_hold_life_ab.py`, `probe_hold_recompute.py`, `probe_intent_target.py`, `probe_obs_channels_static.py`, `probe_pass_prob.py`, `probe_pathfix.py`, `probe_precise_threat.py`, `probe_reward_composition.py`, `probe_reward_semantics.py`, `probe_run_trainer_params.py`, `probe_scorer_offline.py`, `probe_search_cost.py`, `probe_settlement_units.py`, `probe_threat_approx.py`, `probe_threat_prune.py`, `probe_threat_trigger.py`, `probe_tool_usage.py`, `probe_v3_mono_check.py`, `probe_v4_ln_pair.py`, `probe_value_ln.py`, `random_eval_100.py`, `run_selftests.py`, `s1_plan_gate.py`, `s2_neutrality_probe.py`, `s2_trade_probe.py`, `selftest_engagement_trade_online.py`, `selftest_io_bootstrap.py`, `selftest_offline_engagement_trade.py`, `selftest_reward_tables.py`, `summarize_probe_v3_mono.py`, `summarize_solo_run.py`, `value_displacement_scan.py`

### ④ 其它（脚本 / 引擎侧辅助 / 待归类）（24 个）

未落入上面三类者：领域脚本、引擎侧辅助、以及**待归类**项。

`assassin_left_bridge_test.py`, `assassin_vs_megaknight.py`, `assassin_vs_sparky.py`, `audit_intent_save.py`, `cdp_forward.py`, `convert_upstream_frames.py`, `count_vectors_ckpt.py`, `et_solo100k_readout.py`, `extend_level16.py`, `fl_il_to_bc.py`, `il_act_upper_bound.py`, `il_bc_bench.py`, `il_bc_sweep.py`, `il_bc_equiv.py`, `il_determinism_probe.py`, `il_card_usage.py`, `il_carry_diag.py`, `il_ckpt_equal.py`, `il_curve_report.py`, `il_eval_holdout.py`, `il_kingtower_cast_census.py`, `il_mix_report.py`, `il_probe_kingtower_cast.py`, `il_nll_decompose.py`, `il_opp_prediction.py`, `il_paired_compare.py`, `il_readout_games.py`, `question_bank_poc.py`, `selftest_spell_kingtower.py`, `replay_feasibility_probe.py`, `skarmy_probe.py`, `skarmy_probe_compare.py`, `skarmy_probe_upstream.py`, `skarmy_traj_analyze.py`

**合计 127 个 `*.py`**（另有 `scripts/rl/*.py` **11 个**：`start_rl.bat` 依赖其位置的入口包装脚本，**不参与**本分类）。
