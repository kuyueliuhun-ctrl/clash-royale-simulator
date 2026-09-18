# -*- coding: utf-8 -*-
"""生成 AGENTS.md 整理草稿 v2（三层形态：A 全量 / B 一行式 / C 状态速览）。

只读 AGENTS.md，写出：
  1) docs/agents/agents_long_entries.md  —— B 区 10 行 + C 区 6 条的**逐字**搬迁件
  2) AGENTS.reorg-draft.md               —— v2 草稿（A 不动、B 压紧、C 换状态速览表）

不修改 AGENTS.md 本身。幂等；行号/字节数双断言，AGENTS.md 一变动即 ABORT。

用法：python docs/agents/_reorg_v1/build_v2.py
"""
import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)

SRC = "AGENTS.md"
DRAFT = "AGENTS.reorg-draft.md"
NOTES = "docs/agents/agents_long_entries.md"

B_ROWS = [77, 78, 79, 80, 81, 82, 83, 84, 85, 86]
B_BYTES = [1103, 1183, 2645, 2492, 2595, 2073, 3229, 6918, 1417, 3167]
A_HEAD, A_TAIL = 1, 76          # 草稿中逐字保留的前缀区间
C_HEAD_LINE = 90                # '## C. 当前活跃工作'
C_ITEMS = [92, 100, 104, 193, 268, 295]

B_TITLES = {
    77: "攒费穷举实测：「学不会攒费」成立且更狠",
    78: "修复：掩码 `used` 0/1-based off-by-one（B 类根因）",
    79: "探索随机（第一问）：字面「随机×随机+位置随机+正弦衰减」做不到事",
    80: "偏置随机（第二问）：抬「不出牌」概率是唯一正确杠杆",
    81: "状态条件随机（第三问）：低压时才随机 —— 方向对，买回了回报",
    82: "「初始模型 + 全过程随机」100 局评估 + dashboard 观测",
    83: "「非法动作」是怎么定义的 —— 四层取证",
    84: "全项目结构优化（盘点 + Tier 0/1/2/3 执行记录）",
    85: "外部调研类台账（IL 奖惩参考 / 人类回放 / 录像格式对照）",
    86: "上游引擎增量核查（原作者 `f616f19` vs 我们）",
}

# 2026-09-19 事实取证查出的口径/措辞不符（原文一字未改，只加备注）
B_NOTES = {
    83: "⚠️ **已知口径不一致（2026-09-19 取证查出，未改）**：本行 L4 写 `battle.py:2702→2996-3090`，"
        "而 `docs/illegal_action_layers_2026-09-18.md:41` 写作 `battle.py:2996-3077`（该分册全篇无 3090）；"
        "两者都未覆盖 `_deploy_card_impl` 实际终点（下一个 `def` 在 3221）。**成因未定，不得自行改写**（【R10】）。",
    84: "⚠️ **已知自相矛盾（2026-09-19 取证查出，未改）**：本行中段写「余 6 项全部判定『不做』」，"
        "但同一行后文又写 T2-7 / T3-4「完成」，且 `docs/structure_remaining_2026-09-19.md:18-20` 的最终结算是"
        "**完全完成 15 / 部分完成 3 / 判定不做 3**。⇒ 以分册为准；本行正文按原文冻结不改。",
    85: "⚠️ **已知不精确（2026-09-19 取证查出，未改）**：本行写「三条长条目逐字搬入」，实测 **2/3** ——"
        "第 85–86 行（提交 `2e8e0e9^`）与 `docs/agents/reward_surveys.md:9-10` 逐字节相同（仅 CRLF 差），"
        "第 3 条（录像格式）是随源文档 `docs/replay_format_comparison_2026-09-19.md` **新建**的，没有「搬入」前身。",
}

# C 区取证查出的缺口（原文/目标文档均未改，2026-09-19）
C_NOTES = {
    0: "⚠️ **本条的 5 个事实在任何目标文档里都不存在**（2026-09-19 取证）：`run100k` **已跑完**这个状态"
       "（`docs/run100k_2026-09-18.md` 仍停在启动记录、`plans_runs_docs.md §8` 无 run100k 行、ledger 无；"
       "实证是 `runs/run100k/run_state.json` step=100000）、`runs/run100k/` 路径、录像名 `league_{0,8000,…,100000}.pkl`、"
       "`+ 14 ckpt`、新源 `runs/run_schema5/`。⇒ **这些事实只存在于 `AGENTS.md`（现搬入本文件）**；"
       "若要压缩 C 区，必须先补写进台账（待拍板）。",
    1: "⚠️ **已知状态相反（2026-09-19 取证查出，未改）**：`docs/long1m_stopped_2026-09-18.md:50` 写"
       "`long1m_verdict_2026-09-18.md` **不会产出**（本文件即其替代），而 `docs/agents/plans_runs_docs.md:51`（§8 台账）"
       "仍写它「**待出**」——§8 未随终止更新。两处相反，**以终止文档为准**，台账待修正（待拍板）。",
    3: "⚠️ **已知：17 处细节不在判读文档内**（2026-09-19 取证：PARTIAL=17、MISSING=0）——它们分散在"
       "`docs/et_solo100k_2026-09-18.md`（§10/§11/§12）与预注册 §11.13.6–13 里，判读文档只有总述。"
       "⇒ 压缩本条时必须保留「不构成判决 + n=1 分辨率标定」这两条在场。",
    4: "⚠️ **已知：一条不变式只在代码里**（2026-09-19 取证：PARTIAL=1）——「日志推不到就显示『不可用』、"
       "不拿别的 run 顶上」在 `docs/train_health_metrics_2026-09-18.md` 无明述，只有代码证据"
       "（`src/clasher_new/rl/train_health.py:283,296-297`、`dashboard_html.py:1031`）。",
    5: "⚠️ **已知：一条纪律只有 2/5 份明写**（2026-09-19 取证：PARTIAL=1）——「五份各自独立、不合并」"
       "仅 ④`:9`、⑤`:9` 明写；①②③ 只有「仅加声明、不改正文」的编排者声明（①`:18`、②`:10`、③`:9`）。",
}

C_TITLES = {
    0: "C0 `run100k`：已跑完（14 点两级评估 / 14 个 schema-4 录像）",
    1: "C1 `long1m`：按用户指令提前终止（128,000 / 1,000,000 步）",
    2: "C2 新方案（按局面结算的圣水交换信用分配）：已完整取证，判「不接线」",
    3: "C3 `economy_et` 100k 两臂干预长跑：已跑完 + 已判读（不构成判决）",
    4: "C4 训练健康指标（价值损失 / 策略熵）：已交付",
    5: "C5 五份对象报告：跨项目 RL 阶段奖惩设计",
}

# ── B 区替换行（一行式：标题 + 一句话结论 + 指针 + 分册锚点；目标 ≤ 800 B）──
B_COMPACT = {
77: "| **★ 攒费穷举实测：「学不会攒费」成立且更狠** | 「放弃一张买得起的牌」**75,891 帧 0 次**、7,238 段主动不出牌**0 段**在圣水≥4 时仍不出、A 段最长 **22 < 34 帧** ⇒ **轨迹从未被采样**（0 样本 ⇒ 0 梯度；step 0 与 100k 同结构）。★ 附带查出 **B 类病理**（非法卡包整包被拒、白掉一帧），**污染**了「圣水≥6 = 0.52%」读数（**96% 是它**） ｜ 审计 [`docs/elixir_saving_audit_2026-09-18.md`](docs/elixir_saving_audit_2026-09-18.md)、仪器 `scripts/pass_streak_audit.py`、台账 **C14 / O9**；全文 → 分册 **§1** |",
78: "| **★★ 修复：掩码 `used` 0/1-based off-by-one（B 类根因）** | `used` 存 **0-based** 却用 **1-based** `sa.slot` 测试 ⇒ **掩码放行已用槽位** ⇒ 提交重复槽位包 ⇒ `validate_bundle` **整包拒绝**（真实 100k：497 帧 / 0.7%）。**一行修复 + 两条零成本掩码不变式 + 回归测试**（判别力已证）；同种子 **44/256→0/256**、拒绝帧 **474/1707→0/1609**、端到端 **B = 0/3413**；⚠️ 与攒费无关（C14 不变） ｜ [`docs/mask_used_slot_offbyone_fix_2026-09-18.md`](docs/mask_used_slot_offbyone_fix_2026-09-18.md)、留证目录 `docs/mask_used_slot_offbyone_fix_2026-09-18/`、台账 **O9**；全文 → 分册 **§2** |",
79: "| **★ 探索随机（第一问）：「随机×随机+位置随机+正弦衰减」做不到事** | 攒到 Xbow 是**合取**（连续 34 帧不花钱、其中 ~17 帧须逐帧抽中「不出」）⇒ 成功率 `p^k`（指数在帧数）；解析 **4.5×10⁻¹⁰ /100k 帧**；**决定成败的是「时长」不是「随机」**（`hold d~U{1..40}` ⇒ ≥6 帧占 **28.96%/38.81%**，差 ~12 个数量级）；**正弦+衰减是训练步上的调度** ⇒ 对局内合取**零作用**且裁梯度；三新超参在 n=1 下不可判 ｜ [`docs/exploration_randomization_analysis_2026-09-18.md`](docs/exploration_randomization_analysis_2026-09-18.md)、仪器 `scripts/probe_explore_randomization.py`、留证 `docs/probe_explore_randomization/`、台账 **O10**；全文 → 分册 **§3** |",
80: "| **★ 偏置随机（第二问）：抬「不出牌」概率是唯一正确杠杆** | 逐帧 `1/6→q` ⇒ 成功率 `q^k`（每帧 5 倍 = 12 个数量级）；**可零 off-policy 实现**：给 **STOP logit 加正偏置 β**（`ratio≡1`），剂量 β=0/4/5/6/7 ⇒ **0.000 → 0.178 → 12.70 → 51.27 → 82.98%**；仓内已有 `stop_logit_bias=-1.0`（**仅初始化生效**）；「抬大费牌概率」方向是**反的**。★★ **覆盖率 ≠ 会学**：每局回报 **−4.46 → −41.5/−54.1** ⇒ PPO 会学到「别这么干」 ｜ [`docs/exploration_bias_pass_2026-09-18.md`](docs/exploration_bias_pass_2026-09-18.md)、仪器同前、留证 `docs/probe_explore_randomization/bias_*`、台账 **O10**；全文 → 分册 **§4** |",
81: "| **★ 状态条件随机（第三问）：低压时才随机 —— 方向对，买回了回报** | 门 `d`（最近 15 帧没掉塔血）开 **60.9%** 帧；`sample@6`（未门控）**48.209% / −56.13** → **`gate@6:d` 6.765% / +1.68**（三轮里**第一个同时成立**）；代价 **÷7**；用码内 `PRESSURE_THRESHOLD=2.0` 只开 **4.7%** ⇒ 等于没有；n=8 且基线回报跨 run 摆动 ⇒ **回报差异不可当判据**；**「省下的费没花出去」未解决**（133 帧 / 0 次 Xbow）⇒ **仍要 O7** ｜ [`docs/exploration_pressure_gate_2026-09-18.md`](docs/exploration_pressure_gate_2026-09-18.md)、仪器同前、留证 `docs/probe_explore_randomization/{pressure,gated_*,pressure_gate}.*`、台账 **O10**；全文 → 分册 **§5** |",
82: "| **★ 「初始模型 + 全过程随机」100 局评估 + dashboard 观测** | p0 = 全过程均匀随机 actor、p1 = `solo_main_0.pt` 确定性 argmax；**100 局 / 31,491 帧**（schema 5）⇒ **58.0%±4.9**、局均 **314.9 帧**、空 bundle **90.4%**；独立复算 **A 90.4% / B 0.0%**（掩码修复的更大样本复核）/ **C 9.6%**；**圣水峰值 5.357 = 开局 5.0 + 2 帧回费 ⇒ 100 局里从未超过开局水平**；**Xbow 0 次**（3,062 部署）。dashboard **8701**。⚠️ p0 **不能用于训练/BC**；对手同源 ⇒ 胜率不含绝对强度 ｜ [`docs/rand100_eval_2026-09-18.md`](docs/rand100_eval_2026-09-18.md)、runner `scripts/random_eval_100.py`、复算 `docs/rand100_eval_2026-09-18/`、台账 **O10**；全文 → 分册 **§6** |",
83: "| **★ 「非法动作」四层取证（只读取证，未改代码）** | 四层**并列且互不推导**：**L1 掩码**（只产 `bool`）→ **L2 整包校验**（**9 条 reason**，任一非法**整包拒收**）→ **L3 惩罚/统计**（`invalid_penalty=0.05` × `invalid_count`；两来源）→ **L4 引擎物理**（失败一律**静默 `return False`**）；**10 条冲突 C1–C10 并列不调和**；**F1** MergeMaiden「费用过严 + 位置过松」；**F2** BarbLog 在 `FOUR_DECK_SET` ⇒ **solo 实时暴露**、`run` 模式 p1 侧非法**不被计数**；真实日志 P1-20 = **1,644 行全 BarbLog**；未定 U8–U12 ｜ [`docs/illegal_action_layers_2026-09-18.md`](docs/illegal_action_layers_2026-09-18.md)、台账 **O9**；全文 → 分册 **§7** |",
84: "| **★★ 全项目结构优化（Tier 0/1/2/3）** | 已拆到 [`docs/agents/structure.md`](docs/agents/structure.md)：盘点 **199 `.py` / 62,402 行**、引擎→`rl/` 反向边 = 0；Tier 0/1（死件 31、核对器自检 13/13、UTF-8 单一实现、绝对路径 9→0）；Tier 2（`dashboard` **3,171→566**、`card_mechanics` **1,896→861**、奖励簇 → `rl/reward.py` 逐位对账、`selftest` **6,023→172** 且全量 **100 通过 / 0 失败**）；Tier 3（取消 cwd 契约、核对器 **⑪**）。**21 项 = 完全完成 15 / 部分 3 / 不做 3** ｜ [`docs/agents/structure.md`](docs/agents/structure.md)、[`docs/structure_remaining_2026-09-19.md`](docs/structure_remaining_2026-09-19.md)、`docs/agents/scripts_inventory.md`；全文 → 分册 **§8** |",
85: "| **★ 外部调研类台账（IL 奖惩 / 人类回放 / 录像格式）** | 全文 → [`docs/agents/reward_surveys.md`](docs/agents/reward_surveys.md)。① **IL 阶段结构性没有奖惩**（FL 六项损失全掩码 CE/Huber、`penalty` 0 命中），**IL→PPO 锚定项默认关闭**（有一手 KL 锚定先例的是 **VPT**）；② **「奖励塑形退火」无一手实例**；③ **人类回放**：卡键/卡组/费用全可映射、坐标**无需翻转**，但直接重放只走通 **35–62%**（主缺口 = **圣水不可观测**）⇒ **必须先状态重建**；④ 两格式互补；★ 卡覆盖 **177/178**（唯一缺 `void`） ｜ 源 `docs/agents/reward_surveys.md`（+ 三份源分册）；全文 → 分册 **§9** |",
86: "| **★ 上游引擎增量核查（`f616f19` vs 我们）** | 分叉点 **`f20fa4d`**（我方 **242** 提交 / 上游 **24**）；★ **退出码陷阱**：`--is-ancestor f616f19` = **rc=128**（对象不在本机）≠「不是祖先」（rc=1）⇒ 不得据此断言无共同祖先；上游 24 提交里 **6 个动引擎**：① 建筑掉血**已独立修好** ② 碰撞 `*0.5` **未跟** ③ 20 Hz **等价** ④ 河面 A\\* **不照搬**。★ **顺带查出我们自己的四处**：**P1** `battle.py:3292` 应为 `continue`；**P2** 桥面 16 半格 `W`（3 格实 2 格）；**P3（高）** 过河 jump **无界闩锁**；**P4** `walkable_cache` 全局污染 ｜ [`docs/upstream_engine_delta_2026-09-19.md`](docs/upstream_engine_delta_2026-09-19.md)；全文 → 分册 **§10** |",
}

DISCIPLINE = [
    "> **（v2 草稿新增）文件形态纪律**：① 三段在场必要性 —— **A 红线全量在场**（必须遵守，不可拆）／",
    "> **B 索引一行式**（标题 + 一句话结论 + 指针，正文在一手文档）／**C 状态速览**（只留在做什么、状态、未决）；",
    "> ② 总量闸门 **≤ 40 KB**、单行 **≤ 800 B**（避开注入截断与 `read` 工具单行 2000 字符截断）；",
    "> ③ 新决策先写 `docs/` → 本文件只加**一行指针**；④ **只增不改**历史结论（被推翻时保留原条目并标注）；",
    "> ⑤ 拆分一律**逐字搬迁 + 指针**，**不得**用重写的摘要取代正文；下沉的长条目全文 →",
    "> [`docs/agents/agents_long_entries.md`](docs/agents/agents_long_entries.md)。",
]

# ── C 区替换：状态速览表 + 未决指针 ──
C_TABLE_HEAD = [
    "### C-速览（状态 + 必须在场的要点 + 全文锚点）",
    "",
    "| # | 项目 | 状态 | 必须在场的要点 | 全文 / 指针 |",
    "|---|---|---|---|---|",
]
C_TABLE_ROWS = [
    "| **C0** | `run100k` | ✅ **已跑完**（`run` 模式 100k 步） | 14 点两级评估 / 800 局（`--only-vs-main`）；**`runs/run100k/`：14 个 schema-4 录像**（`league_{0,8000,…,100000}.pkl`）**+ 14 ckpt**，**它就是 S2 全部读数的数据源**（schema 4 ⇒ 带 §11.9.3 重建误差）；新源 `runs/run_schema5/`（schema 5、60 局） | 记录 [`docs/run100k_2026-09-18.md`](docs/run100k_2026-09-18.md)（⚠️ 仍是启动记录，终局未回写）；全文 → 分册 **§11** |",
    "| **C1** | `long1m` | ⛔ **已按用户指令提前终止**（128,000 / 1,000,000 = **12.8%**） | 18 个评估点、**0 次降级**、**无判读产出且 J1 按定义不成立**；`runs/long1m/`（393 MB、18 ckpt + 18 录像）**保留** = 新方案现成回放素材 | [`docs/long1m_stopped_2026-09-18.md`](docs/long1m_stopped_2026-09-18.md)；全文 → 分册 **§12** |",
    "| **C2** | 新方案（圣水交换信用分配） | ⏸ **已完整取证，判「不接线」**（**不启动 S3**） | 三门禁：**S1 未通过**（约束在探索/机制侧）、**S2 在正确口径下不通过**（Δρ **+0.008/+0.014**，离线 +0.145/+0.108 ⇒ **÷18 / ÷7.7**）、`τ` 有信号但增量低于一切仪器分辨率；**放行条件写死在预注册 §11.9.4**；出路 = **O7 修机制**（须单独预注册） | 预注册 §11.5/§11.9/§11.10、[`docs/s2_instrument_2026-09-18.md`](docs/s2_instrument_2026-09-18.md)、[`docs/s1_gate_2026-09-18.md`](docs/s1_gate_2026-09-18.md)、[`docs/online_measure_2026-09-18.md`](docs/online_measure_2026-09-18.md)、台账 **X19 / O7**；全文 → 分册 **§13** |",
    "| **C3** | `et_solo100k` 两臂 | ✅ **已跑完 + 已判读**（⚠️ **不构成判决**） | 缺阳性对照 + n=1/臂 ⇒ **不得**写成「修好了 / 确认无效」；三种允许写法：**观察到**（两臂都没做出「攒费→打 Xbow」：全帧圣水≥6 **0.52% vs 0.05%**、Xbow **5/8769 vs 1/8370**）、**不可分辨**（机制层 4/6）、**门禁饱和无分辨力**（兑现率 **98.5% vs 98.2%**）；★ 分辨率标定：同 seed 同配置仅 `--diagnose-every` 10 vs 1 ⇒ `eval@8000` **0.800 vs 0.200（差 0.60）** | 判读 [`docs/et_solo100k_judgment_2026-09-18.md`](docs/et_solo100k_judgment_2026-09-18.md)、读数 [`docs/readout_et_solo100k.md`](docs/readout_et_solo100k.md)、run 记录 §10/§11/§12、台账 **O8**；全文 → 分册 **§14** |",
    "| **C4** | 训练健康指标 | ✅ **已交付**（**描述性、非判据**） | 价值损失 / 策略熵**本来就逐 update 在算、只是 UI 看不见**；口径红线：熵 = 掩码后各 decoder 步熵之**和**（nat）、价值损失必须用 **`vraw=`**（`value=` 是 ÷`v_scale²`：A_et 实测 1.75→24.87）；A_et 熵 0.596→0.459→0.322 后 70k 在 0.26–0.36（**平台**）；⚠️ **不得**写成「熵平台 ⇒ 步数瓶颈」 | [`docs/train_health_metrics_2026-09-18.md`](docs/train_health_metrics_2026-09-18.md)；全文 → 分册 **§15** |",
    "| **C5** | 五份对象报告 | ✅ **已落盘**（**各自独立、不合并**） | 对象 ①Five+AlphaStar ②绝悟/JueWu ③VPT+MineRL ④RLHF+离线 IL ⑤我们 vs FirstLight_CR；共用字段集 ①–⑧ + 统一单位口径；**禁止跨项目数字对拍**、`未验证` **不得升格**；⚠️ ⑤ 返回文本在 **§1.D 截断**（见其 §G） | 五份：`docs/rl_reward_crossproject_tables_2026-09-19.md`、`docs/juewu_reward_table_unified_2026-09-19.md`、`docs/vpt_minerl_rl_reward_tables_2026-09-19.md`、`docs/rlhf_offline_il_reward_audit_2026-09-19.md`、`docs/ours_vs_firstlight_reward_tables_2026-09-19.md`；全文 → 分册 **§16** |",
]
C_OPEN = [
    "",
    "### C-未决 / 待拍板（在场的一行指针）",
    "",
    "- **O7**（机制自锁；候选下一步，**须单独预注册**）・**O8 / O9 / O10**（未决项）⇒ [`docs/agents/ledger.md`](docs/agents/ledger.md) §6。",
    "- **上游核查顺带查出的 P1–P4**（`battle.py:3292` / 桥面 16 半格 / 过河 jump 无界闩锁 / `walkable_cache` 全局污染）——**一行未改，待拍板** ⇒ 见 B 区该行与 [`docs/upstream_engine_delta_2026-09-19.md`](docs/upstream_engine_delta_2026-09-19.md)。",
    "- **C 区取证查出的 5 处缺口**（C0 状态/路径 5 项、C1 §8 台账与终止文档**相反**、C3 判读文档缺 17 处细节、C4 一条不变式只在代码、C5 一条纪律只有 2/5 明写）⇒ 见分册 §11–§16 各条备注与 [`docs/agents/_reorg_v1/REPORT_v2.md`](docs/agents/_reorg_v1/REPORT_v2.md)。",
]


def main():
    global SRC
    ap = argparse.ArgumentParser(description="生成 AGENTS.md 整理草稿 v2（--apply = 直接上线）")
    ap.add_argument("--from", dest="src", default=SRC, help="源文件（缺省 AGENTS.md；上线后重生成用 AGENTS.pre-v2.md）")
    ap.add_argument("--apply", action="store_true", help="写 AGENTS.md 本体（去掉草稿标识行、措辞去「草稿」）")
    args = ap.parse_args()
    SRC = args.src
    term = "（2026-09-19 新增）" if args.apply else "（v2 草稿新增）"
    disc = [DISCIPLINE[0].replace("（v2 草稿新增）", term)] + DISCIPLINE[1:]
    target = "AGENTS.md" if args.apply else DRAFT
    raw = open(SRC, "rb").read().decode("utf-8")
    lines = raw.split("\n")
    before = len(raw.encode("utf-8"))

    # ── 断言 1：B 区 10 行未漂移 ──
    for i, ln in enumerate(B_ROWS):
        got = len(lines[ln - 1].encode("utf-8"))
        if got != B_BYTES[i] or not lines[ln - 1].startswith("| **"):
            print("ABORT: B 区第 %d 行已变动（%d B != %d B）—— 请重新抽取" % (ln, got, B_BYTES[i]))
            return 1
    # ── 断言 2：C 区条目边界未漂移 ──
    got_items = [i for i, l in enumerate(lines, 1) if i > C_HEAD_LINE and re.match(r"^\d+\. ", l)]
    if got_items != C_ITEMS or not lines[C_HEAD_LINE - 1].startswith("## C."):
        print("ABORT: C 区结构已变动（items=%s）" % got_items)
        return 1
    c_bounds = []
    for k, a in enumerate(C_ITEMS):
        b = C_ITEMS[k + 1] - 1 if k + 1 < len(C_ITEMS) else len(lines)
        c_bounds.append((a, b))
    c_texts = ["\n".join(lines[a - 1:b]).rstrip("\n") for a, b in c_bounds]

    # ── 1) 分册：B 区 10 行 + C 区 6 条，逐字 ──
    n = []
    n.append("# AGENTS.md 下沉长条目全文（B 区 10 行 + C 区 6 条，逐字搬迁件）")
    n.append("")
    n.append("> **来源**：`AGENTS.md`（2026-09-19 搬迁时版，%d B）**第 77–86 行（B 区）**与**第 92–296 行（C 区）**，" % before)
    n.append("> **逐字搬入**；原处改为**一行式索引 / 状态速览 + 指向本文件的锚点**（**2026-09-19 v2 已上线**；上线前原文留证 `_reorg_v1/AGENTS.pre-v2.md`）。")
    n.append("> **纪律**：本文件**只增不改**；被推翻时保留原条目并标注「已被 X 推翻」。")
    n.append("> **不要**把本文件当成一手证据 —— 它只是 `AGENTS.md` 曾经的摘要；一手证据在各条目自己给的 `docs/*.md`。")
    n.append("")
    n.append("## 〇、搬迁与取证状态（2026-09-19）")
    n.append("")
    n.append("- 搬迁完整性：B 区 10 行 + C 区 6 条均**逐字**（字节数/条目边界有双断言，见 `build_v2.py`）。")
    n.append("- 事实取证（行内承重事实 → 目标文档 `file:line`）：**B 区 10/10 行**（7 行 0 MISSING、3 行各 1 处口径/措辞不符）、"
             "**C 区 6/6 条**（C2 0 缺口；**C0 有 5 处 MISSING**、C1 1 处状态相反、C3 17 处细节不在判读文档、C4/C5 各 1 处）。")
    n.append("- **C0 的 5 处 MISSING 意味着：那 5 个事实此前只存在于 `AGENTS.md`**（现随本文件保存）——压缩 C 区前须先补写进台账。")
    n.append("- 取证原始记录：`docs/agents/_reorg_v1/verify_L0*.md`（B 区 6 份）、`verify_C*.md`（C 区 6 份）、"
             "`row_L0*.txt` / `citem_*.txt`（全部抽取件）。")
    n.append("")
    for i, ln in enumerate(B_ROWS, 1):
        t = lines[ln - 1]
        n += ["---", "", "## %d. %s" % (i, B_TITLES[ln]), "", "> 出处：`AGENTS.md` 第 %d 行（B 区），逐字，%d B" % (ln, len(t.encode())), "", t, ""]
        if ln in B_NOTES:
            n += [B_NOTES[ln], ""]
    for k, txt in enumerate(c_texts, 11):
        n += ["---", "", "## %d. %s" % (k, C_TITLES[k - 11]), "",
              "> 出处：`AGENTS.md` 第 %d–%d 行（C 区），逐字，%d B" % (c_bounds[k - 11][0], c_bounds[k - 11][1], len(txt.encode())), "", txt, ""]
        if (k - 11) in C_NOTES:
            n += [C_NOTES[k - 11], ""]
    notes_text = "\n".join(n)
    os.makedirs(os.path.dirname(NOTES), exist_ok=True)
    open(NOTES, "w", encoding="utf-8", newline="\n").write(notes_text)

    # ── 2) 草稿：头部 + A 不动 + B 压紧 + C 换速览 ──
    out = list(lines[:A_TAIL])                      # 1..76 逐字
    if not args.apply:                              # 草稿才需要标识行
        out.insert(2, "> **⚠️ 这是整理草稿 v2（2026-09-19），不是生效文件**：与 `AGENTS.md` 的差异 = ① 头部新增「文件形态纪律」；"
                      "② B 区第 77–86 行 → 一行式索引；③ C 区第 92–296 行 → 状态速览表 + 未决指针。"
                      "被换下的正文**逐字**存于 [`docs/agents/agents_long_entries.md`](docs/agents/agents_long_entries.md)。")
    out[12:12] = disc                               # 头部纪律块（插在原引言之后）
    for ln in B_ROWS:
        out.append(B_COMPACT[ln])                   # B 区替换
    out.append("")
    out.append("---")
    out.append("")
    out.append(lines[C_HEAD_LINE - 1])              # '## C. 当前活跃工作' 标题保持原样
    out.append("")
    out += C_TABLE_HEAD + C_TABLE_ROWS + C_OPEN
    draft_text = "\n".join(out) + "\n"
    open(target, "w", encoding="utf-8", newline="\n").write(draft_text)

    # ── 3) 校验与度量 ──
    d = [l for l in draft_text.split("\n") if not l.startswith("> **⚠️ 这是整理草稿 v2")]
    di = next(i for i in range(len(d) - 5) if d[i:i + 6] == disc)
    restored_head = d[:di] + d[di + 6:]             # 去掉纪律块后应与源文件第 1–76 行逐字相同
    assert restored_head[:76] == lines[:76], "头部 / A 区被改动"
    a_block = "\n".join(lines[14:38])               # 源文件 A 区（第 15–38 行）
    assert a_block in draft_text, "A 区被改动"
    for ln in B_ROWS:
        assert lines[ln - 1] in notes_text, "B 区第 %d 行未逐字进分册" % ln
    for txt in c_texts:
        assert txt in notes_text, "C 区某条未逐字进分册"
    for ln in B_ROWS:
        row = B_COMPACT[ln]
        assert row.count("|") == 3, "B 区第 %d 行格数异常" % ln

    after = len(draft_text.encode("utf-8"))
    nb = len(notes_text.encode("utf-8"))
    b_rows_before = sum(B_BYTES)
    b_rows_after = sum(len(B_COMPACT[l].encode()) for l in B_ROWS)
    c_before = sum(len(t.encode()) for t in c_texts)
    c_after = sum(len(r.encode()) for r in C_TABLE_ROWS) + sum(len(r.encode()) for r in C_TABLE_HEAD + C_OPEN)

    print("A 区校验 ✅（逐字未动） | 搬迁校验 ✅ B 10 行 + C 6 条全部逐字命中")
    print("-" * 74)
    print("AGENTS.md            %6d B" % before)
    print("v1 草稿              %6d B" % len(open('docs/agents/_reorg_v1/AGENTS.reorg-draft.v1.md','rb').read()))
    print("%s %6d B   -> 较原文省 %d B (%.1f%%)" % (("AGENTS.md（v2 上线）" if args.apply else "v2 草稿            "), after, before - after, 100.0 * (before - after) / before))
    print("-" * 74)
    print("A 区（红线 19 条）   %6d B  ->  %6d B（不动）" % (sum(len(l.encode()) + 1 for l in lines[14:38]), sum(len(l.encode()) + 1 for l in lines[14:38])))
    print("B 区 10 行           %6d B  ->  %6d B  (max %d / min %d)" % (b_rows_before, b_rows_after, max(len(B_COMPACT[l].encode()) for l in B_ROWS), min(len(B_COMPACT[l].encode()) for l in B_ROWS)))
    print("C 区正文             %6d B  ->  %6d B  (速览表 %d 行)" % (c_before, c_after, len(C_TABLE_ROWS)))
    print("-" * 74)
    print("分册 agents_long_entries.md  %6d B（B %d + C %d 逐字）" % (nb, b_rows_before, c_before))
    print("余量 vs 65,244 B 上限：原文 %d B -> v2 %d B" % (65244 - before, 65244 - after))
    return 0


if __name__ == "__main__":
    sys.exit(main())
