row: L080
target_docs: docs/exploration_bias_pass_2026-09-18.md；scripts/probe_explore_randomization.py；docs/probe_explore_randomization/bias_{proposal,onpolicy,cards}.{txt,json}；docs/agents/ledger.md O10
counts: PRESENT=19 PARTIAL=0 MISSING=0 INDEX-ONLY=0
MISSING/PARTIAL 清单：
- none
关键证据（≤5 条）：
- 「不出牌」偏置有效 / 解析 1,264 事件 → docs/exploration_bias_pass_2026-09-18.md:16 “q=0.9 … = 1,264 事件 /100k 帧（等概率是 4.5×10⁻¹⁰）”（analytic.json q=0.9 = 1264.45）
- 剂量-反应 β=0/4/5/6/7 → 同档:20 “≥6 帧占比 0.000% → 0.178% → 12.70% → 51.27% → 82.98%”
- 回报 −4.46→−41.5(β=6)/−54.1(β=7) → 同档:25；同档:95-99 表
- stop_logit_bias=-1.0 仅初始化生效 → 同档:21；src/clasher_new/rl/follower.py:156/253-255
- xbow_plays 静默恒 0 仪器 bug 留档 → 同档:210-213
