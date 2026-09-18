# -*- coding: utf-8 -*-
"""生成 AGENTS.md 整理草稿（B 区 10 行巨型行 → 一行指针 + 分册全文）。

只读 AGENTS.md，写出两个文件：
  1) docs/agents/index_notes.md      —— 10 行巨型行的逐字搬迁件（只增不改）
  2) AGENTS.reorg-draft.md           —— 除 B 区这 10 行被换成一行式索引外，其余逐字相同
不修改 AGENTS.md 本身。可重复运行（幂等）。

用法：python docs/agents/_reorg_v1/build_draft.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)

SRC = "AGENTS.md"
DRAFT = "AGENTS.reorg-draft.md"
NOTES = "docs/agents/index_notes.md"

# 搬迁前后的行号（1-based）与字节数（搬迁时实测，用作完整性断言）
ROWS = [77, 78, 79, 80, 81, 82, 83, 84, 85, 86]
EXPECT_BYTES = [1103, 1183, 2645, 2492, 2595, 2073, 3229, 6918, 1417, 3167]

TITLES = {
    77: "攒费穷举实测：「学不会攒费」成立且更狠",
    78: "修复：掩码 `used` 0/1-based off-by-one（B 类根因）",
    79: "探索随机（第一问）：字面「随机×随机+位置随机+正弦衰减」做不到事",
    80: "偏置随机（第二问）：抬「不出牌」概率是唯一正确杠杆",
    81: "状态条件随机（第三问）：低压时才随机 —— 方向对，买回了回报",
    82: "「初始模型 + 全过程随机」100 局评估 + dashboard 观测",
    83: "「非法动作」是怎么定义的 —— 四层取证",
    84: "全项目结构优化（盘点 + Tier 0/1/2/3 执行记录）",
    85: "外部调研类台账（IL 奖惩参考 / 跨人类回放 / 录像格式对照）",
    86: "上游引擎增量核查（原作者 `f616f19` vs 我们）",
}

NOTES_EXTRA = {
    83: "⚠️ **已知口径不一致（2026-09-19 取证查出，未改）**：本行 L4 写 `battle.py:2702→2996-3090`，"
        "而 `docs/illegal_action_layers_2026-09-18.md:41` 写作 `battle.py:2996-3077`（该分册全篇无 3090）。"
        "两处均未覆盖 `_deploy_card_impl` 实际终点（源码下一个 `def` 在 3221）。**成因未定，不得自行改写**（【R10】）。",
    84: "⚠️ **已知自相矛盾（2026-09-19 取证查出，未改）**：本行中段写「余 6 项全部判定『不做』」，"
        "但同一行后文又写 T2-7 / T3-4「完成」，且 `docs/structure_remaining_2026-09-19.md:18-20` 的最终结算是"
        "**完全完成 15 / 部分完成 3 / 判定不做 3**。⇒ 摘要在压缩时失真，**以分册为准**；本行正文按原文冻结不改。",
    85: "⚠️ **已知不精确（2026-09-19 取证查出，未改）**：本行写「三条长条目逐字搬入」，实测只有 **2/3** ——"
        "第 85–86 行（提交 `2e8e0e9^`）与 `docs/agents/reward_surveys.md:9-10` 逐字节相同（仅 CRLF 差），"
        "而第 3 条（录像格式）是随其源文档 `docs/replay_format_comparison_2026-09-19.md` **新建**的，没有「搬入」前身。",
}

# 一行式替换行（第二列 = 一句话结论 + 指针 + 分册锚点）
COMPACT_ROWS = {
77: "| **★ 攒费穷举实测：「学不会攒费」成立且更狠** | 「放弃一张买得起的牌」**75,891 帧 0 次**；7,238 段主动不出牌**0 段**在圣水 ≥4 时仍不出（y≥4 必有可出）；A 段最长 **22 < 34 帧** ⇒ **这条轨迹从未被采样**（0 样本 ⇒ 0 梯度，step 0 与 100k 同结构）。★ 附带查出 **B 类病理**（非法卡包整包被拒 = 白掉一帧：497 帧/0.7%、39/280 局、最长 225 帧 ≈112 s、跨 9 评估点），污染了「圣水≥6 = 0.52%」读数（**96% 是它**），5 次 Xbow 全在 B 解锁帧 | 审计 [`docs/elixir_saving_audit_2026-09-18.md`](docs/elixir_saving_audit_2026-09-18.md)、仪器 `scripts/pass_streak_audit.py`、台账 **C14 / O9**（O3 已补记）；长条目全文 → `docs/agents/index_notes.md` **§1** |",
78: "| **★★ 修复：掩码 `used` 0/1-based off-by-one（B 类根因）** | `used` 存 **0-based** 却用 **1-based** `sa.slot` 做成员测试 ⇒ partial 现「(s, s−1)」时低槽位被整段跳过 ⇒ **掩码放行已用槽位** ⇒ 提交重复槽位包 ⇒ `validate_bundle` **整包拒绝**（真实 100k：497 帧/0.7%、39/280 局、最长 225 帧）。**一行修复 + 两条零成本掩码不变式 + 回归测试** `test_mask_partial_bundle_invariants`（判别力已证）；同种子 **44/256→0/256**、拒绝帧 **474/1707→0/1609**、端到端 **B = 0/3413**。⚠️ 与攒费无关（`A ∩ 圣水≥6` 仍 = 0，C14 不变） | [`docs/mask_used_slot_offbyone_fix_2026-09-18.md`](docs/mask_used_slot_offbyone_fix_2026-09-18.md)、留证 `docs/mask_used_slot_offbyone_fix_2026-09-18/`、台账 **O9**；长条目全文 → `index_notes.md` **§2** |",
79: "| **★ 探索随机（第一问）：字面「随机×随机+位置随机+正弦衰减」做不到事** | 攒到 Xbow 是**合取**（连续 34 帧不花钱、其中 ~17 帧须逐帧抽中「不出」）⇒ 成功率 `p^k`（**指数在帧数**）。实测 `uniform`（随机开到最满）**2,084 帧圣水一次没到 6**（峰值 5.36，≥25/≥34 帧段均 0）；解析 **4.5×10⁻¹⁰ /100k 帧**（即便放宽到 `p=1/2` 也只有 5.8×10⁻²）。**决定成败的是「时长」不是「随机」**：`hold d~U{1..40}` ⇒ ≥6 帧占 **28.96%/38.81%**、≥34 帧段 **978 次/100k 帧**（差 ~12 个数量级）。落点那半 = 均匀化（实测高度集中 84/90 格、top10 占 73%/78%）；正弦+衰减是**训练步**上的调度 ⇒ 对局内合取**零作用**且裁梯度；三新超参在 n=1 下不可判。§7 三写法对照（推荐**混合即策略**、记 `log π_train`、`p=0` 逐位回旧、【R2】） | [`docs/exploration_randomization_analysis_2026-09-18.md`](docs/exploration_randomization_analysis_2026-09-18.md)、仪器 `scripts/probe_explore_randomization.py`、留证 `docs/probe_explore_randomization/`、台账 **O10**；长条目全文 → `index_notes.md` **§3** |",
80: "| **★ 偏置随机（第二问）：抬「不出牌」概率是唯一正确杠杆** | 逐帧 `1/6→q` 把成功率抬成 `q^k`（每帧 5 倍 ⇒ 12 个数量级）；解析 `q=0.9` = **1,264 事件/100k 帧**；实测 `uniform` 0.000% → `bias_0.9` **18.9%**（峰值 10.00）。**可零 off-policy 实现**：给 **STOP logit 加正偏置 β** ⇒ `ratio≡1`，剂量 β=0/4/5/6/7 ⇒ **0.000→0.178→12.70→51.27→82.98%**；仓内已有 `stop_logit_bias=-1.0`（`follower.py:156/253-255`，**仅初始化生效**）。「抬大费牌概率」方向是**反的**（α=1.5 使 ≥6 占比 **−36%**）。★★ **但覆盖率 ≠ 会学**：每局回报 **−4.46 → −41.5/−54.1** ⇒ 采样到了 PPO 也会学「别这么干」（γ 折扣已排除） | [`docs/exploration_bias_pass_2026-09-18.md`](docs/exploration_bias_pass_2026-09-18.md)、仪器同前、留证 `docs/probe_explore_randomization/bias_*`、台账 **O10**；长条目全文 → `index_notes.md` **§4** |",
81: "| **★ 状态条件随机（第三问）：低压时才随机 —— 方向对，买回了回报** | 前提成立：门 `d`（最近 15 帧没掉塔血）开 **60.9%** 帧、最长 544 帧、**838 窗口/100k**。同 ckpt / 同 8 局：`sample@0` 0.000% / **+1.63**；`sample@6`（未门控）48.209% / **−56.13**；**`gate@6:d` 6.765% / +1.68**（三轮里**第一个同时成立**）；`gate@6:h` 0.385% / −13.21。代价 **÷7**（离线预测 ÷1.6 是**乐观上界**）；用码内 `PRESSURE_THRESHOLD=2.0` 只开 **4.7%** ⇒ 等于没有。限定：n=8 且基线回报跨 run 摆动（【O5】）⇒ 回报差异**不可当判据**；**「省下的费没花出去」仍未解决**（攒到 6 费 133 帧、Xbow 0 次）⇒ **仍要 O7** | [`docs/exploration_pressure_gate_2026-09-18.md`](docs/exploration_pressure_gate_2026-09-18.md)、仪器同前（`--mode pressure` / `--gate-mode`）、留证 `docs/probe_explore_randomization/{pressure,gated_*,pressure_gate}.*`、台账 **O10**；长条目全文 → `index_notes.md` **§5** |",
82: "| **★ 「初始模型 + 全过程随机」100 局评估 + dashboard 观测** | p0 = 全过程均匀随机 actor、p1 = `solo_main_0.pt` 确定性 argmax（同 `--only-vs-main` 约定）；**100 局 / 31,491 帧 / 1,367.9 s**（schema 5、生产同一写入路径）。读数 **58W/42L/0D = 58.0%±4.9**、局均 **314.9 帧**、空 bundle **90.4%**。独立复算：**A 90.4% / B 0.0%（掩码修复的更大样本复核，31,491 帧）/ C 9.6%**；`A ∩ 圣水≥6 = 0`；**圣水峰值 5.357 = 开局 5.0 + 2 帧回费、只出现在每局第 3 帧 ⇒ 100 局里圣水从未超过开局水平**；A 段最长 24 < 34；**Xbow 0 次**（3,062 次部署）；落点 484/576 格、熵 8.47 bits。dashboard **http://127.0.0.1:8701**。⚠️ p0 是**随机 actor 不是策略**（不能用于训练/BC）；对手与 p0 同源 ⇒ 胜率**不含绝对强度** | [`docs/rand100_eval_2026-09-18.md`](docs/rand100_eval_2026-09-18.md)、runner `scripts/random_eval_100.py`、独立复算 `docs/rand100_eval_2026-09-18/`、台账 **O10**；长条目全文 → `index_notes.md` **§6** |",
83: "| **★ 「非法动作」四层取证（只读取证，未改代码）** | 四层**并列且互不推导**：**L1 掩码**（只产 `bool`、**不产 reason**）→ **L2 整包校验**（`validate_bundle` **9 条中文 reason**，任一非法**整包拒收**）→ **L3 惩罚/统计**（`invalid_penalty=0.05`（defensive 0.1）× `invalid_count`；两来源：整包拒 =1、引擎拒逐卡 +1）→ **L4 引擎物理**（失败一律**静默 `return False`**）。**10 条冲突 C1–C10 并列不调和**；**F1** MergeMaiden 掩码层「费用过严 + 位置过松」（根因 `_effective_card` 只处理 Mirror）；**F2** 三冲突卡均不在 `DEFAULT_SOLO_DECK`，BarbLog 在 `FOUR_DECK_SET` ⇒ **solo 实时暴露**，`run` 模式 p1 侧非法**不被计数**。真实日志：`P1-20` 共 **1,644 行、BarbLog 100.0%**；**未定 U8–U12** | [`docs/illegal_action_layers_2026-09-18.md`](docs/illegal_action_layers_2026-09-18.md)（§1 分层 / §6 C1–C10 / §10 复核 + F1/F2）、台账 **O9**；长条目全文 → `index_notes.md` **§7** |",
84: "| **★★ 全项目结构优化（盘点 + Tier 0/1/2/3）** | **已拆到分册** → [`docs/agents/structure.md`](docs/agents/structure.md)。盘点：**199 `.py` / 62,402 行**、9 个 God file、**引擎→`rl/` 反向边 = 0**、`card_utils` cwd 隐性契约。**Tier 0/1 完成**（死件清理 31、只读核对器 `scripts/_structure_check.py` 自检 13/13、UTF-8 兜底单一实现、硬编码绝对路径 9→0）；**Tier 2**：`rl/dashboard.py` **3,171→566**、`league_rules.py` **8/8 逐字相同**、`card_mechanics.py` **1,896→861**（65 名逐名相同）、T2-7 奖励簇 → `rl/reward.py`（`env_wrapper.py` **812→598**，逐位对账 4,000 例 0 不一致）、`selftest.py` **6,023→172**（100/100 逐字相同、全量 **100 通过 / 0 失败**）；**Tier 3**：T3-2 **取消 cwd 契约**、T3-4 交付核对器 **⑪**「path 引导须先于产品 import」。**方案 §6 的 21 项：完全完成 15 / 部分完成 3 / 判定不做 3**（每条不做带证据） | [`docs/agents/structure.md`](docs/agents/structure.md)、[`docs/structure_remaining_2026-09-19.md`](docs/structure_remaining_2026-09-19.md)、`docs/agents/scripts_inventory.md`；长条目全文 → `index_notes.md` **§8** |",
85: "| **★ 外部调研类台账（IL 奖惩参考 / 人类回放 / 录像格式对照）** | 全文**已拆到分册** → [`docs/agents/reward_surveys.md`](docs/agents/reward_surveys.md)（两条长条目**逐字搬入**，第三条随源文档新建）。四条一句话结论：① **IL 阶段结构性没有奖惩**（FL 六项损失全掩码 CE/Huber、`penalty` 0 命中），**IL→PPO 锚定项默认关闭**，有一手 KL 锚定先例的是 **VPT**；② **「奖励塑形退火」无一手实例**（Five 的 `anneal` 是超参/环境特性，VPT 退的是 KL 系数 ρ）；③ **人类回放**：卡键 100% 可映射、卡组 8/8、手牌循环费 6/6 精确相等、坐标**无需翻转**，但直接重放只走通 **35–62%**（主缺口 = **逐事件圣水不可观测**）⇒ 可用于 IL 但**必须先状态重建**；④ 两格式互补（我方 schema 5 无卡组元数据 / 无 seed，`logprob`/`masks` 在 `run_league.py:190` 被丢弃）；★ 卡覆盖 **177/178，唯一缺 `void`**（≈3% 卡组） | [`docs/agents/reward_surveys.md`](docs/agents/reward_surveys.md)、源分册 `docs/il_reward_reference_analysis_2026-09-19.md` / `docs/il_replay_feasibility_2026-09-19.md` / `docs/replay_format_comparison_2026-09-19.md`；长条目全文 → `index_notes.md` **§9** |",
86: "| **★ 上游引擎增量核查（原作者 Jason-XII `f616f19` vs 我们）** | 分叉点 **`f20fa4d`**（两仓 `merge-base --is-ancestor` 都 rc=0；我方分叉后 **242** 提交 / 上游 **24**）。★ **退出码陷阱**：`merge-base --is-ancestor f616f19 …` = **rc=128 + `fatal: Not a valid object name`**（上游对象不在本机），**不是**「不是祖先」（那才是 rc=1）⇒ 不得据此断言「两仓无共同祖先」（与 `docs/upstream_claim_audit_2026-09-19.md` 并列不调和）。上游 24 提交里 **6 个动引擎**（4 文件，净 **+52/−33**），**引擎数据与基座零改动**。逐条：① 建筑掉血 `372ed0e` **我们已独立修好**；② 碰撞 `*0.5` **未跟进**（建议先测）；③ 20 Hz **等价**（`env_wrapper.py:88-89`）；④ 河面 A\\* **不照搬**（上游最终版是死代码）；⑤ `1e9716b` 仅注释。★ **顺带查出我们自己的四处**（**本轮一行未改，待拍板**）：**P1** `battle.py:3292` 应为 `continue`；**P2** 桥面 16 半格是 `W` ⇒ 名义 3 格实 2 格；**P3（高）** 过河 jump **无界闩锁**（贴脸 18 s、承伤 0、被误判空军）；**P4** `arena.walkable_cache` 模块级全局 + `int()` 截断 ⇒ **一次越界查询永久改图** | [`docs/upstream_engine_delta_2026-09-19.md`](docs/upstream_engine_delta_2026-09-19.md)、合并件 `docs/upstream_four_sections_merge_2026-09-19.md`；长条目全文 → `index_notes.md` **§10** |",
}


def to_two_cells(row):
    """把「标题 | 结论 | 指针」三格合并成 B 表要求的「标题 | 结论 ｜ 指针」两格。"""
    parts = row.split("|")
    if len(parts) != 5 or parts[0].strip() or parts[4].strip():
        raise AssertionError("替换行不是三格结构: %r" % row[:60])
    return "|" + parts[1] + "|" + parts[2].rstrip() + " ｜ " + parts[3].lstrip() + "|"


def main():
    raw = open(SRC, "rb").read().decode("utf-8")
    lines = raw.split("\n")

    # ── 完整性断言：搬家前的 10 行必须还是预期的 10 行（防行号漂移）──
    for i, ln in enumerate(ROWS):
        got = len(lines[ln - 1].encode("utf-8"))
        if got != EXPECT_BYTES[i]:
            print("ABORT: 第 %d 行字节数 %d != 预期 %d —— AGENTS.md 已变动，请重新抽取" % (ln, got, EXPECT_BYTES[i]))
            return 1
        if not lines[ln - 1].startswith("| **"):
            print("ABORT: 第 %d 行不是表格行 —— AGENTS.md 结构已变" % ln)
            return 1

    before = len(raw.encode("utf-8"))

    # ── 1) 分册：逐字搬迁件 ──
    notes = []
    notes.append("# 索引长条目全文（`AGENTS.md` B 区巨型行的逐字搬迁件）")
    notes.append("")
    notes.append("> **来源**：`AGENTS.md`（2026-09-19 搬迁时版，%d B）**第 77–86 行**，**逐字搬入**；" % before)
    notes.append("> 原处改为**一行式索引行 + 指向本文件的锚点**（草稿见 `AGENTS.reorg-draft.md`）。")
    notes.append("> **纪律**：本文件**只增不改**（【本仓纪律②】）；被推翻时保留原条目并标注「已被 X 推翻」。")
    notes.append("> **不要**把本文件当成一手证据 —— 它只是 `AGENTS.md` 曾经的一行摘要；一手证据在各条目自己给的 `docs/*.md`。")
    notes.append("")
    notes.append("## 〇、搬迁与取证状态（2026-09-19）")
    notes.append("")
    notes.append("- 搬迁完整性：10 行**逐字**（字节数与搬迁前一致，见每条出处行）；`build_draft.py` 内含行号+字节数双断言。")
    notes.append("- 事实取证：**10/10 行**做了「行内承重事实 → 目标分册 file:line」核对；**7 行 0 MISSING、0 PARTIAL**"
                 "（§1 §2 §3 §4 §5 §6 §10 及 §9 除一条措辞）；**3 行各查出 1 处口径/措辞不符**，逐条记在对应 § 的备注里，**原文一字未改**。")
    notes.append("- 取证结论的原始记录：`docs/agents/_reorg_v1/verify_L0*.md`（L80/L81/L83/L84/L85/L86）"
                 "与 `docs/agents/_reorg_v1/row_L0*.txt`（全部 10 行）。")
    notes.append("")
    for n, ln in enumerate(ROWS, 1):
        text = lines[ln - 1]
        notes.append("---")
        notes.append("")
        notes.append("## %d. %s" % (n, TITLES[ln]))
        notes.append("")
        notes.append("> 出处：`AGENTS.md` 第 %d 行，逐字，%d B" % (ln, len(text.encode("utf-8"))))
        notes.append("")
        notes.append(text)
        notes.append("")
        if ln in NOTES_EXTRA:
            notes.append(NOTES_EXTRA[ln])
            notes.append("")
    notes_text = "\n".join(notes)
    os.makedirs(os.path.dirname(NOTES), exist_ok=True)
    open(NOTES, "w", encoding="utf-8", newline="\n").write(notes_text)

    # ── 2) 草稿：只替换 B 区这 10 行 ──
    draft_lines = list(lines)
    for idx, ln in enumerate(ROWS):
        draft_lines[ln - 1] = to_two_cells(COMPACT_ROWS[ln])
    # 头部加一行草稿标识（不改原有文字）
    draft_lines.insert(2, "> **⚠️ 这是整理草稿（2026-09-19），不是生效文件**：与 `AGENTS.md` 的差异**只有 B 区第 77–86 行**"
                          "（10 行巨型行 → 一行式索引 + `docs/agents/index_notes.md` 锚点），其余逐字相同。"
                          "正文全文已**逐字**搬到 [`docs/agents/index_notes.md`](docs/agents/index_notes.md)。")
    draft_text = "\n".join(draft_lines)
    open(DRAFT, "w", encoding="utf-8", newline="\n").write(draft_text)

    after = len(draft_text.encode("utf-8"))
    notes_bytes = len(notes_text.encode("utf-8"))
    row_bytes = [len(text.encode("utf-8")) for text in COMPACT_ROWS.values()]

    print("AGENTS.md            %6d B  (%d 行)" % (before, len(lines)))
    print("AGENTS.reorg-draft   %6d B  -> 省 %d B (%.1f%%)" % (after, before - after, 100.0 * (before - after) / before))
    print("index_notes.md       %6d B" % notes_bytes)
    print("原 10 行合计         %6d B" % sum(EXPECT_BYTES))
    print("新 10 行合计         %6d B  (max %d B / min %d B)" % (sum(row_bytes), max(row_bytes), min(row_bytes)))
    print("净变化               %+d B (AGENTS.md) %+d B (含分册)" % (after - before, after + notes_bytes - before))
    print("余量 vs 65,244 B 注入上限：旧 %d B -> 新 %d B" % (65244 - before, 65244 - after))
    return 0


if __name__ == "__main__":
    sys.exit(main())
