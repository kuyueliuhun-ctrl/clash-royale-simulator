row: L083
target_docs: docs/illegal_action_layers_2026-09-18.md；docs/agents/ledger.md（§O9）
counts: PRESENT=24 PARTIAL=1 MISSING=0 INDEX-ONLY=0
MISSING/PARTIAL 清单：
- L4 行号范围 `battle.py:2702→2996-3090` ｜ PARTIAL ｜ 目标文档全篇无 3090，同一事实写作 `battle.py:2996-3077`（IALL:41「`battle.py:2702`（包装）→ `battle.py:2996-3077`」）；两处均未覆盖 `_deploy_card_impl` 实际终点（源码 next def 在 3221），属范围口径差
关键证据（≤5 条）：
- 四层总览（L1 构成 + 只产 bool / L2 9 条 reason / L3 常量 / L4 静默） → docs/illegal_action_layers_2026-09-18.md:38-41
- validate_bundle 拒收理由 9 条 + `action_mask.py:521-573` → 同上:419-431
- `invalid_penalty` 默认 0.05 / defensive 0.1 / `env_wrapper.py:262-263` / 两来源 → 同上:519-548
- F1 根因 `_effective_card` `action_mask.py:39-43` 只处理 Mirror → 同上:1053
- 日志读数 1,644 = BarbLog 1,644（100.0%）+ 台账 O9（P1-20 首次被量化） → 同上:1084；docs/agents/ledger.md:64
