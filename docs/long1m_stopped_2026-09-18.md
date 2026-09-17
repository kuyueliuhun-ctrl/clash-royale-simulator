# `long1m` 提前终止记录（2026-09-18）

> **触发**：用户指令「停掉训练，然后实现我们的新想法」（按局面结算的圣水交换信用分配，见
> [`frame_credit_proposal_review_2026-09-18.md`](frame_credit_proposal_review_2026-09-18.md)）。
> **性质**：**部分 run**，不是完成态。按【R3】/预注册纪律如实登记：**J1 按定义不成立**，
> 其余判据只能用"已完成的那 12.8%"来读，且**不得**当作完整 run 的结论。

---

## 1. 终止时状态（脚本复算，非手抄）

| 量 | 值 | 复算命令 |
|---|---|---|
| 最终 `step` | **128,000 / 1,000,000（12.8%）** | `runs/long1m/run_state.json` 的 `step` / `total_steps` |
| 评估点 | **18 个**：`0, 8000, …, 96000, 100000, 104000, …, 128000` | `league_state.json` 的 `round_stats[].step` |
| 大点 | `0` 与 `100000` 两个（**被插入**在 8000 网格里，符合 `eval_schedule` 两网格取并集） | 同上 + `rl/config.py::eval_schedule` |
| 评估点 `games` | `main` 在每点 **50 局**（大点 **100 局**）= `10 局/对 × 5 对`（大点 20×5） | `round_stats[].games` |
| ckpt | `main_ckpt_{0,8000,…,128000}.pt`，共 18 个 | `ls runs/long1m/` |
| 录像 | **18 个** `replays/league_<step>.pkl` | `ls runs/long1m/replays/` |
| 目录体积 | **393 MB** | `du -sh runs/long1m` |
| 降级/异常 | **0 次**（日志里唯一的 `降级串行` 命中是启动时的说明行 `[eval-plan] … 失败自动降级串行`） | `grep -c "降级串行" docs/train_long1m.log` → 1（说明行） |
| 孤儿 worker | **无**（终止后 `tasklist` 无 `python.exe`） | `cmd.exe /c tasklist` |

**终止方式**：后台作业 `bash-17` 被取消（同时释放 8700 端口的 dashboard 作业 `bash-18`）。

## 2. 能读与不能读（按预注册 `long1m_prereg_2026-09-18.md`）

| 判据 | 能否读 | 说明 |
|---|---|---|
| **J1（跑完 / 无静默降级）** | ❌ **不成立** | 人为终止 ⇒ 按定义不算"跑完"。**只能记录"到 12.8% 为止无降级"**（0 次） |
| **J2（critic 健康）** | ⚠️ **只能描述前 12.8%** | `diagnose_every=10` 的诊断点在日志里；**不得**外推到 1M |
| **J3（对固定脚本池的非自引用 Elo 曲线，门槛 71 Elo）** | ⚠️ **部分可读** | `ratings.main` 由 1000 → **1921.8**；这是**对着固定脚本池**的读数 ⇒ 非自引用，但**样本只到 128k**，且脚本对手弱（见 §3） |
| **J4（行为门禁）** | ⚠️ **部分可读** | 见 §3 |

## 3. 终止时可直接引用的读数（只作描述）

- **Elo**：`ratings` = `main 1921.8` / `push_flow 1585.8` / `random_deck 1717.8` / `lockdown_flow 1335.6` /
  `all_decks 1248.1` / `counter_flow 1190.9`（`league_state.json`）。
- **逐对手胜率（EMA）**：`main|push_flow 0.836`、`main|counter_flow 0.99998`、`main|lockdown_flow 0.9864`、
  `main|all_decks 0.9945`；对手侧镜像值互为 1−p。
- **环境画像（`replays/league_88000.pkl`，50 局）**：平局 **0**、**49/50 局一方打满 3 冠**、
  局长中位 **168 帧（84 s）**、`opp_played` 非空 **643/9079 帧**。
  ⇒ run 模式对手**弱且不组织推进**（5 个 mask-随机脚本）⇒ **这也是新方案必须挂 `solo` 的实测依据**
  （见评审 §7.4）。
- **当前牌池含 3 个持续产兵建筑**：`Tombstone`（10 局）、`FirespiritHut`（10 局）、`GoblinDrone/GoblinDrill`
  ⇒ **40% 的评估局会踩到"局面永不结算"的坑**（评审 §8.2 的实测依据）。

## 4. 未做（明确列出，避免下次误以为已做）

1. **未出判读**：`docs/long1m_verdict_2026-09-18.md` **不会**产出（run 未完成）。本文件即其替代。
2. **未清理**：`runs/long1m` 393 MB 保留（含 18 个 ckpt + 18 个录像）⇒ 它是新方案的**现成回放素材**，
   **不要**按旧习惯当 smoke 残留清掉。
3. **未复用 ckpt**：新方案上线时若要从这里热启动，注意【R6】——`--main-init` 必须显式传
   `plan_dim=58` / `belief_dim=563`；但**新方案若只加奖励项、不改架构，则不需 `--fresh`**（待预注册决定）。
