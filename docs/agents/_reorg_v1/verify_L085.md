row: L085
target_docs: docs/agents/reward_surveys.md; docs/il_reward_reference_analysis_2026-09-19.md; docs/il_replay_feasibility_2026-09-19.md; docs/replay_format_comparison_2026-09-19.md; src/clasher_new/rl/run_league.py
counts: PRESENT=15 PARTIAL=1 MISSING=0 INDEX-ONLY=1
MISSING/PARTIAL 清单：
- 「三条长条目逐字搬入」 ｜ PARTIAL ｜ 仅 2/3 逐字：`git show 2e8e0e9^:AGENTS.md` 第 85–86 行 vs `docs/agents/reward_surveys.md:9-10` 逐字节相同（5082 vs 5080 B，差仅 CRLF）；第 3 行（录像格式）在提交 2e8e0e9 随其源文档一起**新建**（`--name-status` = `A docs/replay_format_comparison_2026-09-19.md`），拆前 AGENTS.md 只有 2 行 ⇒ 该行无「搬入」前身
关键证据（≤5 条）：
- 逐字搬入（2/3 成立）→ docs/agents/reward_surveys.md:9-10 "（与 2e8e0e9^:AGENTS.md:85-86 逐字节相同，仅 CRLF 差异）"
- FL 六项 IL 损失全掩码 CE/Huber、penalty 0 命中 → docs/il_reward_reference_analysis_2026-09-19.md:22 "FL 的 IL 损失 6 项全是**掩码 CE + Huber**（`native_runner/training/v4/imitation.py:283-290`），`penalty` 在 7 个 IL 文件里 **0 命中**"
- 卡覆盖 177/178、唯一缺 void、≈3% → docs/replay_format_comparison_2026-09-19.md:170 "**178** 个不同值 → 可用 **177** / 不可用 **1（`void`）**"；:171 "**59 / 1941 侧（≈3.0%）** 的卡组含它"
- logprob/masks 在 run_league.py:190 被丢弃 → src/clasher_new/rl/run_league.py:190 "bundle, _, _, hidden, _ = policy.act(obs, tok, plan.to_vector(),"（`docs/replay_format_comparison_2026-09-19.md:377` 逐字引用该行）
- 直接重放只走通 35–62% → docs/il_replay_feasibility_2026-09-19.md:127 "`deploy_card` 在 `raw` 坐标下**接受 35%–62%** 的人类出牌"；主缺口见 :104 "**主缺口是"圣水不可观测"**"（`unaffordable` 62–86，:96-104）
