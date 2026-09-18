row: L081
target_docs: docs/exploration_pressure_gate_2026-09-18.md（+ scripts/probe_explore_randomization.py、docs/probe_explore_randomization/、docs/agents/ledger.md §6 O10）
counts: PRESENT=31 PARTIAL=0 MISSING=0 INDEX-ONLY=1
MISSING/PARTIAL 清单：
- none
关键证据（≤5 条）：
- 门率/窗口（d 60.9%、544 帧、838/100k；h 44.7%、195、752） → docs/exploration_pressure_gate_2026-09-18.md:17-18 “开启 **60.9%** 的帧、最长连续 **544 帧**、其中 **838 个窗口/100k 帧**”
- 六臂读数（sample@0 0.000%/+1.63；sample@6 48.209%/−56.13；gate@6:d 6.765%/+1.68；gate@6:h 0.385%/−13.21；holde@40:d 16.131%/−38.04） → 同文:114-119 表格逐格一致
- 门的选择：码内阈值只开 4.7% → 5.0/100k；中位 threat 4.99 → 同文:33-34 “只开 **4.7%** 的帧，门控后覆盖率 **5.0/100k 帧**”；belief_planner.py:51 “PRESSURE_THRESHOLD = 2.0”、:159 “def _crude_enemy_pressure(battle)”
- 代价 ÷7 比离线 ÷1.6 更贵；引擎门率 42.8%/24.2% vs 离线 60.9%/44.7%；离线=乐观上界 → 同文:30-31、128-129
- 推理错误留档（平均场 17 帧当独立抽签 ⇒ 掉 10 个数量级；门自相关 开12/关15） → 同文:98-100 “把 17 帧当成 17 次独立抽签 …（掉 10 个数量级）”“门状态高度自相关（开 12 帧 / 关 15 帧交替）”
- 仪器/留证/台账：script:862 “--mode pressure”、:871 “--gate-mode”、:401-467 “名@β:门”解析；pressure.json / gated_d.json / pressure_gate.json 均存在；台账 O10 → docs/agents/ledger.md:68（INDEX-ONLY）
