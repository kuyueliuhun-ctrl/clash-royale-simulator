item: C4
target_docs: docs/train_health_metrics_2026-09-18.md（主）；代码取证 src/clasher_new/rl/train_health.py、scripts/health_curve.py、src/clasher_new/rl/ppo.py::_loss_pass、src/clasher_new/rl/dashboard_html.py
counts: PRESENT=14 PARTIAL=1 MISSING=0 INDEX-ONLY=0
MISSING/PARTIAL 清单：
- dashboard「推不到就显示『不可用』＋不拿别的 run 顶上」｜PARTIAL｜主文档无该明述（无「不可用」字样、无「不拿别的 run 顶上」不变式）：仅 §8.2 以事故叙述写“面板静默显示"没有日志"”；行为有代码证据 src/clasher_new/rl/train_health.py:283,296-297（“都不存在则返回 None（不猜一个不存在的路径给上层用）”）与 src/clasher_new/rl/dashboard_html.py:1031（“训练健康不可用”）
关键证据（≤5 条）：
- 两量本就逐 update 计算/打印（100k≈781 点）→ docs/train_health_metrics_2026-09-18.md:26,29-32 “`rl/ppo.py::PPOTrainer._loss_pass`…”“100k 步 ≈ 781 点”；代码 src/clasher_new/rl/ppo.py:361 `_loss_pass`、:406-407 `value_loss_raw`/`entropy`
- solo_state.json history[] 逐评估点 14 点且 32 键无 → 同上:35-36 “逐评估点（A_et 只有 14 点）且 32 个键里没有 `entropy` / `value_loss`”
- 口径：熵=掩码后各 decoder 步熵之和（nat）；价值损失用 vraw → 同上:53 “（nat）”、:68-69,78 “`vraw=` 原始 MSE（未缩放）”“`0.46 / 24.87² = 0.00074`”
- A_et 熵 0.596→0.459→0.322、其后 70k 在 0.26–0.36、平台读数 → 同上:103,108-111 “diff=-0.0568 阈值 k×MAD=0.1988 snr=…0.86 ⇒ 判定：平台”；其后 679 点带内 664/679（:113-114）
- 两工程坑（GBK／_PARENT 推错）→ 同上:222-226 “仍然是 gbk”“之前…GBK 字节、之后…UTF-8 字节”、:231-235 “`_PARENT = dirname(dirname(__file__))` = `src/clasher_new`…一个也推不出来”
