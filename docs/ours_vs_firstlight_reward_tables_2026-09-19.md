# 我方 vs FirstLight_CR 奖惩设计对照（同 schema 抽取）

> **本文件的性质（先读这一段 · 截断如实声明）**
> 本轮 5 个对象报告里，对象 1/2/3 已落盘；**对象 4/5 本轮未落盘**。
> 本文件是对象 5 的子任务返回文本的**逐字落盘件**；该文本**在编排上下文到达本节末行时被截断**，**后半部分（§1.D 尾行之后、§2 FirstLight 起、§3 及以后）在本会话不可恢复**（全仓 grep 无命中、无 session transcript 命中）。
> ⇒ 截断点以 **`<!-- ⛔ 原文在此处被截断 -->`** 显式标出。**未做任何补写或推测**；缺口见 **§G**。
>
> **取数时间**：2026-09-19（CST）
> **纪律**：只写文件内容支持的结论；无证据写「未见证据」；**未与 FL 对拍任何数字**；**不与其它四份对象报告合并**。

> **★ 统一 schema（五份对象报告共用：字段名 / 顺序 / 单位口径一致 — 2026-09-19 编排者补，仅加声明、不改正文）**
> **字段集 ①–⑧**：① 奖励项清单（项名 / 权重 / 量纲 / 作用域 / 是否零和）・② 惩罚项（显式 / 隐式 / 用约束或掩码替代）・③ 塑形与调度（是否 PBRS？是否退火？局内时间权重？）・④ 归一化三问（奖励 / 价值 / 优势）・⑤ value head 设计・⑥ 已知坑（带原文句；二手必须标注）・⑦ 一手出处（URL / `file:line` + 取数时间）・⑧ 阶段归属（IL / RL / 跨阶段接口）
> **本报告落点**：①=§1.A ・ ②=§1.B ・ ③=§1.C ・ ④=§1.D ・ ⑤=**（截断，未取回）** ・ ⑥=**（截断，未取回）** ・ ⑦=逐行内嵌 `file:line`（实测 / `[rc1记]` 二分标注） ・ ⑧=**（截断，未取回）**
> **统一单位口径**：权重一律**按来源原样记录**并显式标注量纲（分母单位）；无单位者写「无单位」；**同名项若量纲不同则并列不调和**，不折算、不换算、不对拍。
> **证据等级图例**：`[实测]` = 子任务在**当前树** `5ced8ce` 回读/运行所得；`[rc1记]` = 仅见于盘上旧文档、**未在当前树复核到**的锚点；其余沿用盘上约定。
> **三条硬纪律**：① **禁止跨项目数字对拍**（本对象 = 我们 vs FirstLight，**两边数字一律不得相减/相除/排名**）；② **标「未验证」的项一律保持未验证**；③ **不得把 IL 阶段的事写成 RL 阶段**，跨阶段项必须落在 ⑧。

---

> **观测时间**：2026-09-19（本会话实测，`date` 环境为 WSL/Linux；时区标注见各源文档）
> **我方树**：`/mnt/e/clash-royale-simulator-main`，`git log -1` = **`5ced8ce`**（2026-09-19 04:34:06 +0800，"docs: 其他 IL…"）——**注意**：两份待读文档锚定的是**旧提交** `267455b`（rc2 自述："本报告的行号全部以 `267455b` 为准"）。此后发生了 T2-7（奖励块拆分）与 T2-8（selftest 拆分）等**纯搬运**重构 ⇒ **行号已漂移**。下表我按纪律区分两种证据：**`实测`** = 我在当前树 `5ced8ce` 上回读/运行得到的行号；**`[rc1记]`** = 仅见于待读文档、我**未在当前树复核到**的锚点。
> **FL 树**：`/mnt/e/FirstLight_CR`，HEAD = **`28d66cc0a5d65888515e22fdf22f11d783b65efb`**（单 commit，工作树 clean，`git rev-list --count HEAD` = **1**）——**实测复核**，与待读文档一致。关键文件 SHA-256 **实测与会话文档逐字相同**：`battle_env.py` = `a6657739…8280`、`contracts.py` = `02f979f8…0942`、`async_cluster_self_play.py` = `68b495b2…8e794`、`ppo_runtime.py` = `91da5010…6609`。
> **纪律**：只写文件内容支持的结论；无证据写「未见证据」；两边口径不同**并列不调和**；**未与 FL 对拍任何数字**；**未运行**任何训练 / 全量 selftest。

---

## §1 我们项目：RL 阶段奖惩（`schema` 口径）

### 1.A 奖励项清单（22 键单点表 · `config.py` 实测 `:37-68` / `:299`）

默认值来自 `config.DEFAULT_REWARD`（**实测** `python -c` 逐键 dump）。「量纲」一栏 **并列不调和**：同一表达式同时携带三种单位。

| # | 奖励项 | 默认权重 | 量纲 | 触发条件 | 来源（`file:line`，实测） |
|---|---|---|---|---|---|
| 1 | `crown_weight` | 8.0 | 分/座 | `Δp1.get_crown_count()`>0（敌塔被破里程碑） | `rl/config.py:49`；`rl/reward.py:227` |
| 2 | `crown_lose_weight` | 10.0 | 分/座 | `Δp0` 破塔；缺省回退 `crown_weight` | `rl/config.py:50`；`rl/reward.py:228` |
| 3 | `tower_dmg_opp` | 0.001 | 分/HP（经归一化层） | 敌方塔掉血 `red_dmg>0`；帧内被相位改写 | `rl/config.py:51`；`rl/reward.py:160-161`、`:229` |
| 4 | `tower_dmg_self` | 0.0012 | 分/HP（不对称：挨打 > 打人） | 我方塔掉血 `blue_dmg>0`（取负） | `rl/config.py:52`；`rl/reward.py:162-163`、`:230` |
| 5 | `tower_dmg_late` | 0.002 | 分/HP | **仅 `battle.time ≥ 120`** 时替换 `tower_dmg_opp` | `rl/config.py:53`；`rl/reward.py:160-161` |
| 6 | `tower_dmg_self_late` | 0.0022 | 分/HP | **仅 `t ≥ 120`** 时替换 `tower_dmg_self` | `rl/config.py:54`；`rl/reward.py:162-163` |
| 7 | `elixir_diff_weight` | 0.5 | 分/圣水（= ⊗ Δ，见 1.C） | `edw` 真值非 0；帧内被相位改写 | `rl/config.py:61-62`；`rl/reward.py:164-165`、`:233-241` |
| 8 | `elixir_diff_late` | 0.1 | 分/圣水 | **仅 `t ≥ 120`** 时替换 `elixir_diff_weight` | `rl/config.py:63`；`rl/reward.py:164-165` |
| 9 | `elixir_bonus` | 0.0 | 分/圣水 | 每帧无条件：`+ w × my_elixir_after`（**帧末**取水） | `rl/config.py:59`；`rl/reward.py:231` |
| 10 | `unit_dmg_k` | 0.0005 | 分/HP | **后置项**：帧起已存在实体（`id>6`、排除 6 类临时实体）掉血 | `rl/config.py:64`；`rl/reward.py:539`（实测；`[rc1记] env_wrapper.py:751-761`） |
| 11 | `win_bonus` | 10.0 | 分（跳变） | `winner == 0`（仅在 `battle.game_over` 时传入） | `rl/config.py:55`；`rl/reward.py:242-243` |
| 12 | `engagement_trade` | 0.0 | 分/「Trade」单位 | **后置项**，`(self._et_w or measure_only)` 真才构造监视器；`>0` 才动奖励 | `rl/config.py:41`；`rl/env_wrapper.py:138`、`:565`（实测） |
| 13 | `engagement_trade_theta` | 1.0 | 圣水 | 每个**结算**窗口各一次：`score=max(0, trade−θ)` | `rl/config.py:42`（注释称"平滑 hinge"，**实现是硬铰链**）；`[rc1记] engagement.py:204` |
| 14 | `engagement_trade_t_ref` | 2.0 | 秒（自标"待标定"） | `if self.t_ref and (pin0 or pin1)`；取 0 ⇒ τ 整项归零 | `rl/config.py:43`；`[rc1记] engagement.py:189/197-201` |
| 15 | `engagement_trade_gate` | 1（int） | 布尔 | 结算时判我方窗口内三塔是否掉血 ⇒ `g0=0` | `rl/config.py:44`；`[rc1记] engagement.py:191-195` |
| 16 | `engagement_trade_measure_only` | 0（int） | 布尔 | 抑制奖励但照跑监视器（**单向**，与 #12 不互斥） | `rl/config.py:48`；`rl/env_wrapper.py:565` |
| 17 | `normalize_tower_dmg` | True | 布尔（**真值判断**） | 逐侧独立：per-tower 三参数齐 ⇒ 溢价分支；否则聚合锚 `10928` | `rl/config.py:60`；`rl/reward.py:205-225` |
| 18 | `tower_premium_k` | 2.0 | 倍率 | 凹形 `(1+k(1−r)²)`，上限 ×3；**需 #17 真且 per-tower 齐** | `rl/config.py:65-66`；`rl/reward.py:103-106`、`:145-147` |
| 19 | `king_gate` | 0.05 | 倍率 | 王塔 **且** 帧末两公主塔存活 ⇒ ×0.05 | `rl/config.py:67`；`rl/reward.py:104-105` |
| 20 | `elixir_diff_*` 的势 Φ | — | 圣水 | `Φ = 手牌圣水 + 场上部署份额`（法术产物份额记 0） | `rl/reward.py:237-241`；`[rc1记] env_wrapper.py:611-631` |
| 21 | `engagement_trade_*` 的 τ | — | 圣水 | `τ=0.5(τ₀−τ₁)`，`τ₀ += g0×shares[1][eid]×min(1, cnt·dt/t_ref)` | `[rc1记] engagement.py:196-202` |
| 22 | `engagement_trade_*` 的 Trade | — | 圣水 | `trade = d0 − d1 + τ`（`d = (elixir+v) − phi0`） | `[rc1记] engagement.py:186-187/203` |

**两处后置项不在 `compute_reward` 内**（浮点加法顺序必须为 `compute_reward → unit_dmg_k → engagement_trade`）：`rl/reward.py` 只到 `:251`（`return reward`），后置项在 `rl/env_wrapper.py:539` / `:565`（实测）。

**三份常量副本键集（实测）**：`config.DEFAULT_REWARD` = **22** 键；`rl/reward.py::_DEFAULT_REWARD` = **16** 键（缺 `draw_penalty` + 全部 5 个 `engagement_trade*`，**env-only = 空集**）；`rl/mcts.py:52` `MCTSConfig.reward` = **8** 键。⇒ 22−16 = **6**，与任务书口径一致。

### 1.B 惩罚项（显式惩罚）

| 惩罚项 | 权重 | 量纲 | 触发条件 | 来源（实测） |
|---|---|---|---|---|
| `invalid_penalty` | **0.05** | 分/次 | `invalid_count` 真；**唯一相乘点** `reward -= w × invalid_count`；计数 3 源（RL 预校验整包记 1 / `use_ability` 失败 +1 / `deploy_card` 失败 +1） | `rl/config.py:58`；`rl/reward.py:249-250`；`rl/env_wrapper.py:249`（`if invalid_count`） |
| `lose_penalty` | 10.0 | 分 | `winner is not None and ≠0` | `rl/config.py:56`；`rl/reward.py:244-245` |
| `draw_penalty` | 10.0 | 分 | `winner is None and game_over`（**平局=败**） | `rl/config.py:57`；`rl/reward.py:246-248` |
| `crown_lose_weight` | 10.0 | 分/座 | 被破塔（> 得塔 8.0 的不对称） | `rl/config.py:50`；`rl/reward.py:228` |
| 预设覆盖：`defensive` | `invalid_penalty=0.1` | — | 仅预设 | `[rc1记] config.py:390/396/402/408/414/441/462` |

**惩罚不在 `engagement_trade` 侧**：`#14 t_ref=0` 是"关掉 P4b 钉住项"的开关，`#15 gate=0` 不是关 τ 而是"不限门控的 P2"（`[rc1记] config.py:44` 注释 + `engagement.py:191`）。

### 1.C 塑形与调度

| 问题 | 事实 | 来源（实测 / `[rc1记]`） |
|---|---|---|
| `PHASE_SWITCH_S = 120.0` 只作用于哪 3 键？ | **只作用于 3 个「被覆盖键」**：`tower_dmg_opp`←`tower_dmg_late`、`tower_dmg_self`←`tower_dmg_self_late`、`elixir_diff_weight`←`elixir_diff_late`。`_phase_weights(rw, battle_time)` 返回 3 元组，调用方把三值**写回 `rw`** 再调 `compute_reward`。**其余 16 键不参与相位**（含塔血溢价 `tower_premium_k`/`king_gate`、`unit_dmg_k`、5 个 `engagement_trade*`） | 实测 `rl/reward.py:77`（常量）、`:152-166`（函数）、`:160-165`（三键） |
| 相位注入路径 | 主路径 `rl/env_wrapper.py` 内每决策帧；**另有两条独立复刻**：`flow_league.py:296` 每帧**整字典重绑定** `env.reward_weights = rw_a`、`mcts.py` 搜索值函数。⚠️ `RLEnv.__init__` 缓存的 `_et_*`（`:138-145`）**不随重绑定更新** | 实测 `flow_league.py:296`；`[rc1记] env_wrapper.py:351-359` |
| 阈值副本 | **4 份、互不 import**：`rl/reward.py:77`（唯一 import 源）、`belief_planner.py:57`（`LATE_S`）、`action_mask.py:172`（**裸字面量** `>= 120.0`）、`scripts/phi_offline_check.py:60`（手抄） | 实测前三处；第 4 处 `[rc1记]` |
| **`elixir_diff` 是否 PBRS？** | **不是**（**未见证据**支持 PBRS）。实现为 `Δ` 形式的势差，**但缺 PBRS 的承重条件**：① **无 γ 折扣**——纯 `Φ'−Φ`，而 PBRS 要求 `γΦ'−Φ`；② 无终态势置零；③ 注释自称 *"potential-style shaping"*（`rl/reward.py:183`）。⇒ 并列两种口径：**语义上像势差，形式上不满足 Ng et al. 的 PBRS 条件** | `rl/reward.py:183`（注释）、`:233-241`（实现）；**无 γ 在场**：`grep γ/gamma` 在该块 0 命中 |
| 退火 | **未见任何奖励权重退火**；`MODEL_REWARD_OVERRIDES`（`config.py:73-80`）是**按模型静态覆盖** `elixir_diff_weight`（push_flow 0.7 / counter_flow 0.3 / lockdown_flow 0.05） | 实测 `rl/config.py:73-80` |
| **引擎回费三档 vs 奖励两档错配** | 引擎：`battle.py:2870` = `2.8 if t<120 else 1.4 if t<240 else 2.8/3`；奖励：只有 120 s 一个切换点 ⇒ `t∈[240,300)` 用**双倍期价格**给**三倍回费**记账。**全仓无人承认** | `[rc1记] battle.py:2870`；`rl/reward.py:77`（单阈值） |
| 边界语义 | 120 s 处两侧一致（引擎 `t<120` 用 2.8 / 奖励 `t≥120` 用 late），**无 off-by-one**；`:724` 读的是**推进后**的 `battle.time`，引擎回费用**推进前**的 ⇒ 跨边界那帧奖励可能提前 ≤1 帧进新档 | `[rc1记]` + 代码直读 |

### 1.D 归一化三问（奖励 / 价值 / 优势）

| 通道 | 是否有归一化 | 证据（实测） |
|---|---|---|
| **奖励** | **零归一化**（既无 clip 也无 normalize）；`v_scale` **只出现在价值损失分母**，不进 `rewards` | `rl/ppo.py:392`（`v_loss = v_mse / (v_scale*v_scale)`）；`rl/config.py:195` |
| **价值** | **有**，但**入口不一致**：`value_norm="running"` ⇒ `v_scale = ret_scaler.scale()`，`v_loss /= v_scale²`。⚠️ `_make_trainer`（实测 `rl/run_league.py:722-725`）**只传 8 个参数**，**不传 `value_norm`** ⇒ `run` 模式价值损失**恒 1.0 倍（不缩放）**；`train_solo.py:1141-1146` 则**齐全** | 实测 `rl/run_league.py:722-725`；`rl/train_solo.py:1141-1146`；`rl/ppo.py:241-246/392`；`[rc1记] run_league.py:840-843`（**旧行号，已漂移**） |
| **优势** | **有**，`adv_norm="scale"` 缺省 = **只除批 std、不中心化**（`advs = advs/(std+1e-8)`）；另有 `"batch"`（中心化+除 std）与 `"none"` | 实测 `rl/config.py:195`；`rl/ppo.py:222-233` |
| `value_norm` 默认 | `"none"`（dataclass 默认 = 旧行为）；`economy`/`economy_et`/`economy_etm` 预设设 `"running"` | <!-- ⛔ 原文在此处被截断 --> |

<!-- ⛔ 原文在此处被截断（本行 = 子任务返回文本的最后一个字符） -->

---

## §G 缺口声明

- 对象 5 的返回文本在 `§1.D` 的 `value_norm` 默认行处被截断。**未取回**：§1.D 该行来源列、§2（FirstLight 侧 ①②③④⑤⑥⑦⑧ 全部）、§3（设计意图 / 迁移性）、§4（来源清单）、§5（未验证项）、§6（对既有报告的更正）。
- 本文件**只承载已取回的部分**；**未补写、未推测**。
- 若需补齐：建议**单独重跑对象 5 的抽取子任务**（输入 = 本文件 + 盘上 [`docs/cmp_reward_ours_detail_2026-09-19.md`](cmp_reward_ours_detail_2026-09-19.md)、[`docs/cmp_reward_fl_detail_2026-09-19.md`](cmp_reward_fl_detail_2026-09-19.md)、根目录 `三轴对照_奖惩_训练_非法动作_我们vsFirstLight_2026-09-19.md`）+ 本文件顶部的统一 schema。
- 本文件**不含**任何跨项目数字对拍：`§1.A–1.D` 全部数字都是**我们本仓**的读数（`[实测]` / `[rc1记]` 二分标注），FL 侧数字尚未取回。
