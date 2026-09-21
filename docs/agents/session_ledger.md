# 会话协作台账（多会话并行：谁在改哪些文件）

> **本册内容**：多会话并行的**登记协议** + **会话表** + **文件占用表** + **冲突处理** + **现场快照**。
> **用途**：同一个工作区里可能同时有多个 DSH 会话在干活（会话 A / B / C 由用户指定字母）。
> **落笔前先查本表，动手前先登记，收尾再回写**——避免两会话抢改同一文件、互相覆盖。
> **纪律**：本表**只增删改自己那一行**；别人的行**只读**。

---

## 1. 登记协议（4 步，缺一不可）

| 步 | 时机 | 动作 |
|---|---|---|
| **① 开工登记** | 动手改**第一个文件之前** | 在 §2 自己的行写：在做什么 / 状态 🔄 / **将要碰的文件** / 模式（独占·只读）/ 起始时间 |
| **② 改前查表** | 每次要编辑一个文件前 | 查 §3：该文件已被别人标 **独占** ⇒ **停下问人**，不要抢写（纯读取不受限） |
| **③ 变更通知** | 改动了别人标过的文件时 | 在 §4 追加一行：时间 + 文件 + 改了什么 + 证据（commit / 留证路径） |
| **④ 收尾回写** | 任务完成 | 本行状态改 ✅ + 补「**改了哪些文件**」清单 + 按要求登记到 `AGENTS.md` / 文档地图（【R18】改代码同步维护文档） |

**模式口径**：
- **独占** = 我改期间别人别改；
- **只读** = 我在读，别人可改（但改了请在 §4 说一句）；
- **已让出** = 我不再碰这个文件。

**会话身份**：DSH 会话**没有可自动读取的全局 ID**（本机 `~/.dsh/` 下无会话库，实测只有 `dsh-think-summary.json`）
⇒ 身份**只能人为约定**：用户说 A / B / C，就固定用这个字母；行内另写清「工作目录 / 正在跑的命令」，便于对号。

---

## 2. 会话表（当前在场）

| 会话 | 在做的事 | 状态 | 正在 / 将要修改的文件 | 模式 | 起始 | 最近更新 |
|---|---|---|---|---|---|---|
| **A** | **W1 低费空砸豁免（`8f93de4`）→ C23/C24/C25 拆头（**路线已关闭**）→ **C26 W2/W3 手牌打分+前期信息分** → **C27 觉醒 S1（`--evo-slots`，已跑通；判据②在字面上 FAIL 且**暴露引擎觉醒路径两个真崩溃** ⇒ 按判据④**停手：不做 S2/S3**）**；分支 `il-whiff-handscore-w1` / `il-handscore-w2` / `il-evo-s1` | 🔄 进行中 | `rl/{action_mask,hand_score(新),plan_space,follower}.py`、`scripts/{selftest_spell_kingtower,selftest_hand_score(新),_mask_diff_snapshot_spells(新),probe_plan_extras_prefix(新),il_hs_report(新),il_hs_feature_use(新),selftest_evo_slots(新),probe_evo_pair(新),il_evo_probe_report(新),il_bc_sweep,fl_il_to_bc,il_readout_games,il_eval_holdout,il_card_usage,_run_hs_arms.sh(新),_run_hs_readouts.sh(新),_run_evo_probe.sh(新)}.py|sh`、`scripts/README.md`（**自动生成**，用 `_structure_check.py --write-scripts-readme` 重写）、`docs/{il_whiff_handscore_prereg_2026-09-22,il_evo_prereg_2026-09-22,fl_il_il2_prereg_2026-09-22}.md`、`docs/agents/{ledger,agents_long_entries,session_ledger}.md`、`AGENTS.md`、`docs/fl_il_2026-09-21/{hs_*,evo*}.json`（新） | 独占（上述文件） | 2026-09-22 | 2026-09-22 |
| **B** | 建**多会话协作台账**（本文件）+ 在 `AGENTS.md`、`docs/README.md` 各加一行索引 | 🔄 进行中 | `docs/agents/session_ledger.md`（新建）、`AGENTS.md`（+1 行）、`docs/README.md`（+1 行） | 独占 | 2026-09-21 22:25 | 2026-09-21 22:25 |
| **C** | （待 C 会话自己填写） | ⏳ 待登记 | — | — | — | — |

> 状态用词：🔄 进行中 / ⏸ 暂停 / ✅ 完成 / ⛔ 中止。**完成后保留该行**（本仓纪律：只增不改历史）。

---

## 3. 文件占用表

| 文件 / 目录 | 会话 | 模式 | 说明 | 登记时间 |
|---|---|---|---|---|
| `docs/agents/session_ledger.md` | B | 独占 | 本文件，仅创建会话可改结构 | 2026-09-21 22:25 |
| `AGENTS.md` | B | 独占 | 仅追加 1 行决策索引；**A 已按 §5 串行规则追加 1 行**（见 §4） | 2026-09-21 22:25 |
| `docs/README.md` | B | 独占 | 仅追加 1 行快速入口 | 2026-09-21 22:25 |
| `src/clasher_new/rl/action_mask.py` | A | 独占 | W1 低费空砸豁免（`SPELL_WHIFF_FREE_MAX_COST`） | 2026-09-22 |
| `scripts/selftest_spell_kingtower.py` | A | 独占 | W1 回归断言（23 → 32） | 2026-09-22 |
| `scripts/_mask_diff_snapshot_spells.py` | A | 独占 | 新建：全法术位图 A/B 门禁 | 2026-09-22 |
| `src/clasher_new/rl/hand_score.py` | A | 独占 | 新建：W2/W3 的 17 维纯函数特征（plan 尾部追加） | 2026-09-22 |
| `src/clasher_new/rl/plan_space.py` | A | 独占 | 新增 `PLAN_BASE_DIM`/`PLAN_HOLD_OFFSET`（不变量修复） | 2026-09-22 |
| `src/clasher_new/rl/follower.py` | A | 独占 | `_plan_biases` 用 `PLAN_HOLD_OFFSET` + `plan_extras_zero` 消融开关 | 2026-09-22 |
| `scripts/il_card_usage.py` | A | 独占 | 修 save 语料 `IndexError`（人类不出牌 ⇒ 「未出牌」桶）；**旧语料 R2 逐位复现** | 2026-09-22 |
| `runs/_fl_il_bc_hs/` | A | 独占 | W2/W3 语料（2200 局 / plan_dim=75）+ 9 个臂 ckpt | 2026-09-22 |

> **热点文件**（多会话常碰，谁动谁登记）：`AGENTS.md`、`docs/README.md`、`docs/agents/*.md`、
> `src/clasher_new/rl/action_mask.py`、`src/clasher_new/battle.py`、`src/clasher_new/rl/ppo.py`、
> `scripts/` 下的共享仪器、以及 `runs/<run>/`（**同一 run 目录不要两个会话同时写**）。

---

## 4. 变更通知（改动了别人标过的文件 / 重要工作区状态变化）

| 时间 | 会话 | 文件 | 改了什么 | 证据 |
|---|---|---|---|---|
| 2026-09-21 22:25 | B | — | （空表，等待填写） | — |
| 2026-09-22 | A | `src/clasher_new/rl/action_mask.py`、`scripts/selftest_spell_kingtower.py` | W1：低费法术（cost≤2）**空砸豁免**（只豁免 8h 空砸闸门，9h 砸塔 EV 闸门不动 ⇒ F1 不回退）；回归 23→32 断言 | 32/32 PASS；全法术 A/B 432 张：32 张变化、**收紧 0**、越界 0；标准 128 张**逐位全等** |
| 2026-09-22 | A | `AGENTS.md` | 按 §5「两会话都改 ⇒ 串行」：**写前已重读**，仅追加 1 行决策索引（C22 行） | 本条即通知；`scripts/_agents_split.py --check` 待跑 |
| 2026-09-22 | A | `AGENTS.md`、`docs/agents/agents_long_entries.md` | 追加 **C23 行**（解耦组合臂）；首版 952 B **超 800 B 闸门** ⇒ 按头部纪律 ⑥ **逐字下沉**到 `agents_long_entries.md` **§27**，行内压缩到 793 B | `--check` **PASS**（34,245 B / 40,960；最长行 793 B；分册 27 节、缺 0） |
| 2026-09-22 | A | `scripts/il_readout_games.py` | 增 `--gate-ckpt` / `--gate-threshold`（§6 解耦组合臂）；**新增路径默认关闭**⇒ 不给 flag 时行为不变 | **F0 恒等回归 PASS**（τ=1.0 vs 无门，3 局 13 项统计逐值全等、`gate_stop_frames=0`、`p0_top_cards` 一致） |
| 2026-09-22 | A | `docs/fl_il_il2_prereg_2026-09-22.md` | 增 **§6**（组合臂口径封口：6.1 恒等式证伪 / 6.2 封口 6 条 / 6.3 K1–K6 / 6.4 分支 / 6.5 不变量 / 6.7 结果 / 6.8 第三分支失效登记） | 判据**跑前写死**（R3）；结果：**K1 FAIL**（34.70 > 28.5）⇒ 按 6.4 关闭组合臂路线 |
| 2026-09-22 | A | `docs/agents/ledger.md` | 增 **C23** 条目；头 `C1–C22` → `C1–C23` | 与 `AGENTS.md` C23 行同步（R18） |
| 2026-09-22 | A | ⚠️ **工作区意外（照实登记）** | 跑组合臂时 `--out` 传了 `/mnt/e/...` 给 **Windows python** ⇒ 产物落到**仓外**的 `E:\mnt\e\...` 废路径；我在清理该废树时**发现里面另有别的会话早先留下的重复件** `docs/grid_collision_loss_2026-09-20/stats_replay.json`。**已核实仓内正本完好**（26,039 B、Sep 20 07:42）⇒ **未丢数据**。教训：**Windows python 不认 `/mnt/...`**，runner 类脚本一律用**相对路径** | 全程未 `git add -A` / 未 `git checkout .`；废树已删；组合臂产物已用相对路径重跑落回 `runs/il_readout_gateR337actR00/`（`runs/` 被 gitignore ⇒ 读数另存 `docs/fl_il_2026-09-21/gate_compose_R337gate_R00act_stats.json` 入库） |
| 2026-09-22 | A | `src/clasher_new/rl/follower.py`、`scripts/il_bc_sweep.py` | **C24 独立 act 头**（预注册 §7）：`act_head(2)` + `slot_head` 出维 6→5；新增 `SUB_SPACE_DIM` 隔离形状破坏面；旧路径走**分离实现**；PPO 批量路径在拆头下显式 `NotImplementedError`；ckpt 元数据加 `decoupled_act` + 不一致告警。**新增** `scripts/selftest_decoupled_act.py` | 回归 **15/15 PASS**（含 P3 嵌入口径 `max\|Δ\|=6e-07`）；**F0 位图门 `29/29 逐位相同`** |
| 2026-09-22 | A | `scripts/il_eval_holdout.py`、`scripts/il_readout_games.py` | §7.8 判定的**必改评估路径**：`first_option_probs` → `first_option_dist6(...)[STOP]`（嵌入口径）；拆头臂 top1 取嵌入 argmax；`build_policy` 补读 `decoupled_act`（**不加会静默随机**） | 同 ckpt 读数**逐位不变**（`top1 0.40419060493409936` / `nll 5.596222589020527`）⇒ 共享头路径零影响 |
| 2026-09-22 | A | ⚠️ **自伤事故（照实登记）** | 在 `il_eval_holdout.py` 上做了一次声称「只删尾换行」的编辑，**实际把两行粘成一行** ⇒ `SyntaxError`。**被紧随的回归跑立刻抓到**（两个长跑用 `il_bc_sweep.py`，未受影响），已修复 + 三文件语法检查通过 | 教训：**`edit` 的 old/new 末尾换行必须逐字对齐**，改完**立刻跑语法检查** |
| 2026-09-22 | A | `docs/fl_il_il2_prereg_2026-09-22.md` §7.13、`docs/agents/{ledger,agents_long_entries}.md`、`AGENTS.md` | **C25**：修正配对（R10 池）判读 —— **负结果**：拆头**没有**解开耦合（ΔNLL **+0.0697**、出牌/局 30.70→**33.20**），且 **`stop_when_playable` 0.2228→0.0879（方向相反）** ⇒ 按 §7.5 判**共享头诊断不成立** ⇒ **拆头路线关闭** | `--check` PASS（35,772 B / 40,960；最长行 793 B；分册 29 节缺 0）；★ 这是本项目第一次用「**正确配对**」把一条已投入的路线**关掉** |

---

## 5. 冲突处理

1. **独占冲突** ⇒ **停止编辑**，在 §4 记一行，然后**问人**（不要"我先写完再说"）。
2. **两个会话都要改 `AGENTS.md`** ⇒ **串行**：写前**重读一次**（另一会话可能刚改过），写完立刻回写本表。
3. **`git` 操作**：
   - 不要 `git add -A` / `git add .`——会把别的会话的中间产物一起提交；
   - 不要 `git checkout .` / `git restore .`——见 §6，会复活 270 份已迁移档案；
   - 提交**按路径点名**；发现别人未提交的改动**不要替他提交/回滚**，先问人。
4. **发现别人正在改的文件被自己改了** ⇒ 立刻在 §4 留痕并告知用户（宁可暴露，不静默覆盖）。
5. **拿不准就写"拿不准"**（【R10】）：本表不写猜测当事实，推断一律标「推断·未确认」。

---

## 6. 现场快照（**只读观测**，2026-09-21 22:25 CST）

> 这是**观测**，不是任何会话的声明；**冲突判断以 §2 / §3 为准**。取数命令：
> `git status --short`、`git log --oneline -5`、`find . -type f -newermt '-3 days'`（排除 `.git/ runs/ node_modules/`）。

**① 工作区状态**：`git status --short` = **270 项删除（未暂存）/ 9 项修改 / 19 项未跟踪**；
HEAD = `37d6d83`「修复：法术『只罩王塔』的落点非法（掩码 EV 闸门）…」。

**★★ 270 项删除 = 2026-09-20 docs 目录精简留下的未提交状态，不是新事故**：
`docs/_archive/` **整个目录尚未纳入 git**（`git ls-files docs/_archive` 为空），而 `docs/` 下的原件被删
⇒ git 看到 270 个「删除 + 未跟踪副本」。
⇒ **对所有会话的硬约束**：不要 `git add -A`、不要 `git checkout .`（会把 264 份档案当重复文件复活／整批误提交）。

**② 最近 3 天改动过的文件（分线，⚠️ 归线系推断·未与任何会话确认）**：

| 线 | 涉及文件（观测到的最近改动） | 备注 |
|---|---|---|
| **线 1 · IL / 掩码** | `src/clasher_new/rl/action_mask.py`、`scripts/selftest_spell_kingtower.py`、`docs/il_spell_kingtower_gap_2026-09-22.md`、`docs/mask_snapshots/{spell_before,spell_after}.npz`、`docs/fl_il_2026-09-21/readout_fix3_*` | 与 HEAD 提交同一条线，**疑似已收尾** |
| **线 2 · 文档形态 / AGENTS v2** | `AGENTS.md`、`docs/agents/{ledger,agents_long_entries}.md`、`docs/README.md`、`scripts/README.md` | 与 AGENTS.md 头部「文件形态纪律」同批 |
| **线 3 · IL 二期预注册** | `docs/il_evo_prereg_2026-09-22.md`（未跟踪，22:21） | 单文件，**疑似刚开笔** |

**③ 未跟踪文件 19 项**中含 `docs/_archive/`（整目录）、`docs/{model_decision_path,model_vector_inventory,observation_gap_verdict,grid_collision_loss}_2026-09-20.md`
与 `scripts/{count_vectors_ckpt,probe_emb_capacity,probe_grid_collision_loss,probe_obs_channels_static}.py`
⇒ **这些是别人还没提交的成果，别当成垃圾清理**。

---

## 7. 待确认（未决）

- **A / C 两会话各自在做什么、要碰哪些文件** —— **未确认**（本表 §2 留空待填）。
  用户给出后（或由 A/C 自己写入）本表即为权威；在填写前，**冲突判断只能靠 §6 的只读观测**。
- 是否需要把本表升级成**机器可读**（JSON/YAML）+ 一个小工具做「改前查表」自动提示 —— **未定**，待用户拍板。
