item: C0
target_docs: docs/run100k_2026-09-18.md；docs/agents/plans_runs_docs.md §8 Run 台账；docs/agents/ledger.md
counts: PRESENT=12 PARTIAL=0 MISSING=5 INDEX-ONLY=2
MISSING/PARTIAL 清单：
- `run100k` **已跑完**（状态）｜MISSING｜三目标文档无：T1 仅「启动记录」且 :56 停在 cur_step 0，§8 无 run100k 行，ledger 无；实证 `runs/run100k/run_state.json` step=100000、`league_state.json` history=800（本次读）；旁证 docs/cmp_manual_2026-09-19.md:540
- `runs/run100k/` 路径｜MISSING｜目标三文档无；docs/README.md:72「数据 `runs/run100k/`」、docs/s2_instrument_2026-09-18.md:386
- 录像文件名 `league_{0,8000,…,100000}.pkl`｜MISSING｜目标三文档无；s2_instrument:386 只列 `{0,8000,16000}`；实存 14 个（ls）
- `+ 14 ckpt`｜MISSING｜目标三文档无；实存 `main_ckpt_*.pt` 14 个（ls）；docs/cmp_train_ours_2026-09-19.md:525
- 新源 `runs/run_schema5/`（短 smoke、schema 5、60 局）｜MISSING｜目标三文档无；docs/engagement_trade_online_2026-09-18.md:150
关键证据（≤5 条）：
- run 模式 100k 步 / 两级评估 / `--only-vs-main` → docs/run100k_2026-09-18.md:4-5 “`run` 模式 **100k 步**基线跑，**沿用 `long1m` 的两级评估**…`--only-vs-main`”
- 14 点 / 800 局 → 同 :29 “14 个评估点：大 2 / 小 12；5 对/点（only_vs_main）⇒ 预计总对局 800 局”（:33 训练日志逐字一致）
- 14 个 schema-4 录像 + S2 数据源 → 同 :7-8 “产出 **14 个 schema-4 录像**，供新方案 S2 的离线仪器直接消费”
- 孤儿 worker 7.7 GB / 7.51→23.24 GB / 12 档 / 工具 → 同 :16-20 “可用提交 7.51 GB < 12 GB ⇒ 降档”；“**14 个** `--multiprocessing-fork`（≈7.7 GB）”；“`[OK] 可用 23.24 GB ≥ 12 GB ⇒ 12 档可用`”；:18 “`scripts/kill_orphan_workers.ps1 -Kill`”
- §11.9.3 指针（INDEX-ONLY）→ docs/engagement_trade_prereg_2026-09-18.md:491 起（非指定 target_docs，故记 INDEX-ONLY）
