# 预注册：低费法术空砸豁免 + 手牌打分机制 + 前期卡组信息分（2026-09-22）

> **用户指令原文**：「首先 Poison 并不便宜，它 4 费。空砸这个问题，首先是放开一些低费卡牌的限制，
> 其次是，加入手牌打分机制：己方手牌打分+对方手牌打分+双方手牌打分，并且打分机制要与圣水量结合。
> 在此基础上，增加一个前期特有的，卡组信息分，让模型前期进行一些试探」
>
> 纪律：【R3】判据跑前写死（含失败分支）｜【R7】阈值单常量源｜【R12】能算的不许让网络猜｜
> 【R13】位图对账｜【R17】包含关系逐位验证｜【R6】观测维变更必须 `--fresh`。
> 本文件分三批 **W1 / W2 / W3**，各自独立可回滚。

---

## 0. 口径更正（先记，防止继续沿用错口径）

| 上一轮的说法 | 事实 | 出处 |
|---|---|---|
| 「Zap/Poison 这类**便宜**伤害法术」 | **Poison = 4 费**，不便宜；上一轮我把它算进「便宜」是**错的** | 用户 2026-09-22；`Card("Poison").elixir == 4` 实测 |

「低费」的可执行定义（本预注册采用）：**`cost <= 2`**。
实测本仓 **27 张 `type=="spell"`**（`card_data` 全枚举，非硬编码）：

| 费用 | 张数 | 卡 |
|---|---|---|
| 1 | 4 | GlobalLightning / Heal / Mirror / WarmSpell |
| **2** | **6** | **BarbLog / GoblinCurse / Log / Rage / Snowball / Zap** |
| 3 | 9 | Arrows / Clone / DarkMagic / Earthquake / GlobalClone / GoblinBarrel / RoyalDelivery / Tornado / Vines |
| 4 | 3 | Fireball / Freeze / Poison |
| 5 | 2 | GoblinPartyRocket / Graveyard |
| 6 | 3 | Lightning / MergeMaiden / Rocket |

**受落点几何闸门约束（`_spell_requires_placement_target`，即"落点伤害型 ∧ 非滚动"）的按费用分组**：

| 费用 | 受闸门法术 |
|---|---|
| **2** | **Snowball、Zap** |
| 3 | Arrows、Earthquake、RoyalDelivery\*、Tornado、Vines |
| 4 | Fireball、Freeze\*、Poison |
| 6 | Rocket |

\* `RoyalDelivery` 的 `_spell_tower_damage == 0.0`、`Freeze` 的 `_spell_radius_m == 0.0`
⇒ 两者在 9h EV 闸门内部**提前 return**（等效未受约束，既有语义如此，本批不动）。

⇒ **阈值取 2 而不是 3**：cost≤2 恰好等于「过牌法术」集合
（Zap/Snowball/Log/BarbLog + 非伤害的 GoblinCurse/Rage），
而 3 费里混着 Earthquake/Tornado 这类**不是过牌卡**的解场法术。

---

## 1. W1：低费法术**空砸豁免**（本批已实现）

### 1.1 改动（单点、单常量）

| # | 改动 | 位置 |
|---|---|---|
| W1-a | 新增 `SPELL_WHIFF_FREE_MAX_COST = 2.0`（**单一来源**，R7） | `rl/action_mask.py`（`_spell_requires_placement_target` 之后） |
| W1-b | 新增 `_spell_whiff_gate_applies(name, info)` = 落点几何闸门适用 **∧** `cost > 阈值` | 同上 |
| W1-c | 8h 空砸闸门的**两处**调用点改用 W1-b | `_position_legal`（提交路径）、`legal_cells`（掩码路径） |

### 1.2 关键设计：**只豁免空砸闸门，不豁免砸塔 EV 闸门**

`_spell_tower_ev_illegal` 自带「落点没罩到任何存活塔 → 提前 `return False`（=不非法）」
⇒ 低费豁免后：

- **空地上的 Zap/Snowball → 合法**（过牌/空放被解锁，用户要的）；
- **只罩敌方王塔的 Zap/Snowball → 仍然非法**（F1 修复**不回退**，用户上一轮的要求）。

两件事由**两个不同谓词**管：空砸（有没有目标）与砸塔 EV（值不值）。
把它们合成一个开关是本批**明确避免**的做法——否则「放开低费空砸」会顺手拆掉「不许砸王塔」。

### 1.3 判据（跑前写死）

| 判据 | 内容 | 结果 |
|---|---|---|
| **J-W1.1** | 空场（只有塔）`Zap/Snowball` 合法格 **0 → >0** | ✅ 0 → **458**（各） |
| **J-W1.2** | ≥3 费伤害法术在**每张位图**上逐位全等 | ✅ Arrows/Tornado/Fireball/Poison/Rocket 全 **0** |
| **J-W1.3** | `Zap/Snowball`「只罩敌方王塔」格**仍全非法**（F1 不回归） | ✅ n=44，仍合法 **0** |
| **J-W1.4** | 同 200 局 IL 管线 `mask_reject` **不增**、`labels` **不减** | ✅ **219 → 203**（−16）、`labels` **4440 → 4461**（**+0.47%**）、errors 0 |
| **J-W1.5** | `selftest_spell_kingtower.py` 全断言 PASS | ✅ **32/32** |
| **J-W1.6** | 【R13】标准 128 张语料**逐位全等** + 全法术 A/B **零收紧/零越界** | ✅ 128/128 全等；432 张中 **32 张**变化、**收紧 0**、越界 0 |

### 1.4 J-W1.4 基线（**脚本复算，禁手抄**，R4）

基线 = `docs/fl_il_2026-09-21/maskimpact200.json`。
复算命令见 §4。**判据：不增 / 不减**（低费放宽**只可能**让闸门少拒，故两者原则上只会变好或不变）。

**实测结果**（脚本复算，未手抄）：

| 指标 | 基线（F4 后） | W1 | Δ |
|---|---|---|---|
| `mask_reject` | 219 | **203** | **−16** |
| `labels` | 4440 | **4461** | **+21（+0.47%）** |
| `frames` | 66,064 | 66,455 | +391 |
| `team_single` | 4,659 | 4,664 | +5 |
| `errors` | 0 | 0 | 0 |

⇒ **J-W1.4 成立，且方向是"回血"**：与 F4 批次（`labels` **−1.27%**）**反号**。
机制解释：低费法术的空砸落点不再被掩码拒收 ⇒ 重建战局的走样次数变少
（`mask_reject` ↓）⇒ 更多人类决策帧保住 ⇒ `labels` ↑；`frames` 随之 +391（重建少走样）。
⚠️ 两者**不是同一批完全一致的局**（重建一旦被拒就会分叉），故只作**方向判据**，
不作逐局配对判据（R17）。

### 1.5 失败分支

- J-W1.2 出现任何一位变化 ⇒ **回滚**（说明改动越界到贵牌）。
- J-W1.3 失败 ⇒ **回滚**（低费豁免误伤了砸王塔保护）。
- J-W1.4 `labels` 下降 ⇒ **保留改动但如实记录**（说明人类数据里也没有低费法术空砸；
  此时 W1 是「给模型一个人类不用的自由度」，需在 W2 的判据里体现，并考虑把阈值退回 0）。

---

## 2. W2：手牌打分机制（己方 / 对方 / 双方）＋ 圣水结合

### 2.1 可算性前提（R12：能算的不许让网络猜）

| 量 | 可得性 | 出处 |
|---|---|---|
| **己方手牌** | 完全可观测（`p.cycle[:4]`） | `rl/observation.py:132-136` |
| **对方手牌** | **精确队列推断**（卡组已知时）：从第 4 张起「当前手牌集合 = 卡组 − 最近 4 张互异牌」，与洗牌顺序无关 | `rl/bayes_filter.py:9-14`（暴力验证 3000 局 × 59 步零反例）；锁定流 `entropy()==0.0` |
| **对方圣水** | **只有规则估计器，且已知有偏**：硬编码 2.8 s/费 vs 引擎**分阶段** 2.8/1.4/2.8÷3 | `rl/belief.py:288-298` vs `battle.py:2880-2881` |
| **对方卡组 8 张** | 对局内已知（自对弈/`--fresh` 训练时双方卡组由 env 给定） | `rl/env_wrapper.py` env 构造 |

⇒ **己方/对方手牌打分都是确定性可算量**，对方圣水打分**必须显式携带不确定性**（不得当真值）。

### 2.2 特征定义（首版写死，纯函数、可单测）

新增 `src/clasher_new/rl/hand_score.py`（无副作用纯函数）：

| 特征 | 定义 | 值域 |
|---|---|---|
| `S_own` | 己方手牌：`n_playable = #{c ∈ hand : cost(c) ≤ elixir}`；`cheapest`；`total_cost`；`afford_frac = n_playable/4` | `[0,1]` |
| `S_opp` | 对方手牌（贝叶斯后验推断集合）同式，用 `opp_elixir_est` | `[0,1]` |
| `S_diff` | `S_own − S_opp`（**双方**） | `[-1,1]` |
| `E_*` | `elixir` / `opp_elixir_est` / 差（**显式带 `uncertainty`**） | 归一 |
| `t_afford` | 「还要几拍才买得起最贵手牌」= `max(0, max_cost − elixir)/2.8` | 秒/10 |

### 2.3 注入通道（含一处**必修坑**）

- 走 **plan 尾部追加**：`plan_mlp.0.weight` 有「前列拷贝 + 尾零」兼容分支
  （`follower.py:135-139`）⇒ 旧 ckpt **可加载**，新维从零学。
- **不需**用 scalar 通道：scalar 在 `fused` **中段** ⇒ `enc_fc` 形状变化**无兼容分支、静默重置**
  （`follower.py:172-183`）⇒ 必然 `--fresh`。
- ⚠️ **必修坑**：`follower.py:424-427` 用 `v[PLAN_DIM - 4 + i]` 读 **hold**。
  尾追加会让它**静默错位**（指向新特征而非 hold）⇒ 必须新增显式常量
  `PLAN_HOLD_OFFSET`（`plan_space.py`）并让 `follower.py` 改用它。
  这是**不变量修复**：改完对旧 58 维向量 `hold` 偏置必须**逐值不变**。

### 2.4 判据

| 判据 | 内容 |
|---|---|
| **J-W2.1** | 对旧 58 维 plan 向量，`_plan_biases` 的 `slot_bias/cell_bias` **逐值全等**（`torch.equal`） |
| **J-W2.2** | `PLAN_DIM` 58 → 58+K；旧 ckpt 加载**不报错**且 `plan_mlp.0.weight` 新列为**严格 0** |
| **J-W2.3** | `--fresh` 同种子配对 **≥3 seed**：`ΔNLL ≤ 0` ∧ `Δact-AUC ≥ 0` ∧ 行为双侧带不越界；**无收益 ⇒ 回滚** |

### 2.5 失败分支

任一 seed `ΔNLL > 0` **且** `Δact-AUC < 0` ⇒ 判 W2 无收益 ⇒ **回滚特征**
（**保留 `PLAN_HOLD_OFFSET` 修复**，因为那是 bug 修复，不是特征）。

---

## 3. W3：前期特有的**卡组信息分**（试探）

### 3.1 定义

| 特征 | 定义 | 标定依据 |
|---|---|---|
| `info_unknown` | `1 − known_opp/8`，`known_opp` = 对手**已打出过**的不同卡数（确定性可读，EBK） | — |
| `w(t)` | 前期门控 = `clip(1 − t/T_info, 0, 1)`，**`T_info = 30.0`** | 人类开局窗口 T=30 s：median **4** 次出牌、**4** 张不同卡；首牌中位 **t=9.85 s** |
| `probe_value` | 当前可负担牌数 × `info_unknown`（"这手牌还能逼出多少未知"的代理） | — |

维度：`{info_unknown, w(t), probe_value, known_opp/8}` 四维进 plan 尾（与 W2 同批）。

### 3.2 人类侧经验标定（由脚本复算，T4 侦察）

| 读数 | 值 |
|---|---|
| 开局窗口 T=30 s 内出牌数（局×侧） | mean **4.21** / median **4** |
| 窗口内**不同卡**数 | median **4** |
| 窗口内花费中位 | **13.00** 圣水（30 s 回费 ≈10.7 ⇒ 动用了开局 5 费） |
| 首牌费用档 | ≤2 费 **37.49%**、≥4 费 **37.97%**（**接近平坦**，不是"开局必出便宜牌"） |
| 首牌时圣水（上界/下界代理） | 中位 **8.95 / 6.33** ⇒ **等到接近 10 才动** |
| ≤2 费法术落**己方侧** | **4,076 次 = 窗口法术的 49.71%**（覆盖 **40.78%** 单位）⇒ **过牌行为真实存在** |
| 1 费牌进敌方半场 | **2 / 6,038 = 0.03%** |
| 出牌位置（T=30 s） | 己方 **64.21%** / 桥头 **26.32%** / 敌方 **9.47%** |

### 3.3 判据

| 判据 | 内容 |
|---|---|
| **J-W3.1** | `T_info` 窗口内 `w(t) > 0`、窗口外**恒 0**（逐帧验证） |
| **J-W3.2** | `--fresh` 训练后，同 T=30 s 窗口内 `known_opp` 中位 **≥ 基线**（**方向判据**，不设绝对阈值——R15/R16） |

### 3.4 边界（不许越界）

「信息分」**只进观测/软偏置，不改奖励**（【R11】不在 A′ 类取证之前改奖励）。

---

## 4. 执行顺序、复算命令与回滚

**顺序**：W1（本批，纯掩码，已实现并过门禁）→ W2（特征 + `PLAN_HOLD_OFFSET` 修复，需 `--fresh`）
→ W3（前期门控，与 W2 同批 `--fresh`）。每批：新建分支 + 推送 + 配回归。

**复算命令**（全部从仓库根，相对路径；⚠️ 本环境 `_abs()` 会把 `/mnt/...` 译成 `E:\...`，故**必须传相对路径**）：

```bash
# W1 回归测试（32 断言）
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 scripts/selftest_spell_kingtower.py

# W1 全法术位图 A/B（同进程单变量：阈值 -1.0 vs 当前）
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 scripts/_mask_diff_snapshot_spells.py --ab

# W1 标准 128 张语料（R13 主口径）
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 scripts/_mask_diff_snapshot.py /tmp/kt_w1.npz
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 scripts/_mask_diff_snapshot.py --compare \
  /mnt/e/clash-royale-simulator-main/docs/mask_snapshots/kingtower_after2.npz /tmp/kt_w1.npz

# W1 人类标签回归（200 局；基线 mask_reject=219 / labels=4440）
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 scripts/fl_il_to_bc.py \
  --jsonl ../fl_il_data/replays_part000000.jsonl \
  --mode samples --games 200 --workers 10 --level 11 --coord raw \
  --stop-mode none --stop-stride 1 \
  --out docs/fl_il_2026-09-21/maskimpact200_w1.json --out-dir /tmp/il_mask_w1_samples
```

**回滚**：W1 = 把 `SPELL_WHIFF_FREE_MAX_COST` 设回 `-1.0`（等价改前语义，一行）。
W2/W3 = plan 尾维不追加即可（`PLAN_HOLD_OFFSET` 修复**不回滚**）。

---

## 5. 本批**明确不动**的已知缺口（照实记录，不夹带）

| # | 缺口 | 说明 |
|---|---|---|
| O-a | **`Lightning`（6 费，塔伤 686.4）`_spell_deals_damage` 读 `False`** | ⇒ 它**完全不受**两个闸门约束（可以空砸/砸王塔）。谓词四路取或仍未覆盖它的伤害字段来源。**本批不修**，单列。 |
| O-b | `Log` 部署敌我半场**都**放行（576/576） | 与用户口径「落点同普通部队」**不符**；改它会动人类数据重建 ⇒ **待拍板** |
| O-c | `BarbLog` 两侧 `deploy_card` **均被拒** | 已知 O9 / P1-20；本仓根本打不出去 |
| O-d | 对方圣水估计器偏差（2.8 vs 分阶段回费） | 本批**不修**，只在 W2 特征里**显式带不确定性** |
| O-e | 标准 128 张语料对法术改动**结构性盲** | 已用 `_mask_diff_snapshot_spells.py` 补可复算门禁（本批新增）；`_mask_diff_snapshot.py` 的 `HAND` 保持不动 |

---

## 6. W2/W3 实施期修正（**跑前写死**，2026-09-22；本节**取代** §2.2/§2.3/§2.4/§3.1/§3.3/§4 的相应条文）

> 【R3】纪律：本节在**任何训练跑之前**写完。§2/§3 的原文**不删除**（只增不改），有冲突以本节为准。
> 触发原因：写代码时发现草稿有 3 处**实施级缺陷**，若照跑会得到**构造性的假结论**。

### 6.1 §2.2/§3.1 特征定义：两处**线性重复**修正

| # | 草稿原文 | 缺陷 | 改为 |
|---|---|---|---|
| 修-1 | `S_own` 第 4 维 = `afford_frac = n_playable/4` | 与第 1 维 `n_playable/4` **逐值恒等**（同一量写两遍） | `mean_cost_own/10`（手牌均值费用，不同信息；仍 ∈ [0.1,1]） |
| 修-2 | 信息分第 4 维 = `known_opp/8` | 与第 1 维 `1 − known_opp/8` **仿射等价**（`x = 1 − y`） | `info_unknown · w(t)`（**时间门控后的剩余未知度**；与另三维不共线） |

两处都是**跑前**发现的草稿笔误，修正后**每一维都携带独立信息**。修正后的 17 维布局（冻结）：

| idx | 名 | 公式 | 值域 |
|---|---|---|---|
| 0 | `own_n_playable` | `#{c ∈ own_hand : cost(c) ≤ elixir} / 4` | [0,1] |
| 1 | `own_cheapest` | `min cost / 10` | [0.1,1] |
| 2 | `own_total_cost` | `Σcost / 40` | [0.04,1] |
| 3 | `own_mean_cost` | `mean cost / 10` **（修-1）** | [0.1,1] |
| 4 | `opp_exp_n_playable` | `Σ_i p_i·1[cost_i ≤ ê_opp] / 4` | [0,1] |
| 5 | `opp_cheapest` | `min{cost_i : p_i > 0} / 10` | [0.1,1] |
| 6 | `opp_exp_cost` | `Σ_i p_i·cost_i / 10` | [0.1,1] |
| 7 | `opp_p_big` | `Σ_i p_i·1[cost_i ≥ 5]` | [0,1] |
| 8 | `S_diff` | `own_n_playable − opp_exp_n_playable`（§2.2 的「双方」） | [−1,1] |
| 9 | `elixir` | `elixir / 10` | [0,1] |
| 10 | `opp_elixir_est` | `ê_opp / 10`（**规则估计，非真值**，O-d） | [0,1] |
| 11 | `elixir_diff` | `(elixir − ê_opp) / 10` | [−1,1] |
| 12 | `t_afford` | `max(0, max_cost_own − elixir)/2.8/10`（秒/10） | [0,0.36] |
| 13 | `info_unknown` | `1 − known_opp/8` | [0,1] |
| 14 | `w(t)` | `clip(1 − t/30, 0, 1)` | [0,1] |
| 15 | `probe_value` | `(n_playable_own/4)·info_unknown` | [0,1] |
| 16 | `info_gated` | `info_unknown · w(t)` **（修-2）** | [0,1] |

`PLAN_EXTRA_DIM = 17`（W2 = idx 0..12 = 13 维；W3 = idx 13..16 = 4 维）。
唯一实现：`src/clasher_new/rl/hand_score.py`（纯函数，可单测）。

### 6.2 §2.3 注入通道：**对照臂必须同形状**（本批最关键的方法修正）

草稿把基线写成「旧 58 维网络」。**这是错的**，理由是一条可执行的 RNG 事实：

> `il_bc_sweep.train_one` 先 `torch.manual_seed(seed)`，**再**构造 `FollowerPolicy`。
> `plan_dim=58` 与 `plan_dim=75` 会让 `plan_mlp[0] = nn.Linear(·, 64)` 消耗**不同数量**的
> RNG 抽样 ⇒ **其后每一个参数（cnn / gru / slot_head / cell_head / value_head …）都拿到
> 不同的初始值** ⇒ 「加了信息」与「换了初值」**混淆**，不是单变量。

⇒ **对照臂改为：同一个 75 维网络，前向里把最后 17 列置零**（`FollowerPolicy(plan_extras_zero=17)`）。
这样三臂**形状、参数量、初始化逐值、permutation 顺序完全一致**，唯一差别 = 那 17 列**有没有携带信息**
（置零列的梯度恒 0 ⇒ 它们只是 17 个死输入）。这也顺带把「加了信息」与「纯加容量」**彻底分开**。

| # | 改动 | 位置 |
|---|---|---|
| W2-a | 新增 `PLAN_BASE_DIM`(=58) / `PLAN_HOLD_OFFSET`(=54)；`PLAN_DIM = PLAN_BASE_DIM` | `rl/plan_space.py` |
| W2-b | `_plan_biases` 的 hold 下标由 `PLAN_DIM − 4` 改为 `PLAN_HOLD_OFFSET` | `rl/follower.py` |
| W2-c | 新增 `plan_extras_zero=K`（前向里抹掉 plan 末 K 列）+ ckpt 元数据 + 不一致告警 | `rl/follower.py` |
| W2-d | 新增 `--plan-extras`：plan 尾部追加 17 维 ⇒ 语料落盘 75 维 | `scripts/fl_il_to_bc.py` |
| W2-e | 新增 `--plan-extras-zero {17,4,0}`（+ manifest 记录） | `scripts/il_bc_sweep.py` |
| W2-f | 推理侧同口径产 75 维 plan（`build_plan_vec`，**plan_dim 不符即显式报错**） | `scripts/il_readout_games.py` |
| W2-g | readout 增 `info_window` 读数（J-W3.2 的仪器） | 同上 |

**W2-b 是纯不变量修复，不是本批特征**：对旧 58 维向量 hold 偏置**逐位不变**（回归 T2.1）。
⚠️ 注意坑的**精确形状**：草稿说「尾追加会让 `PLAN_DIM−4` 静默错位」——**不准确**：
`PLAN_DIM` 是常量 58，尾追加后 `v[54+i]` **仍然正确**；真正会错的是「按**向量长度**取尾 4 维」
（`v.shape[0]−4 = 71`）。⇒ 本修复的作用是**把正确读法写成显式常量、杜绝将来有人"顺手改成
`len(v)−4`"**，并在回归 T2.4 里**构造性地演示**那条错误路径会读到 extras（实测错值
`slot 3 = −1.7` vs 正确 `slot 2/4 = −2.5`）。

### 6.3 语料必须**重生成一次**（§4 的「顺序」修正）

`plan` 向量是**生产期烤进样本**的（`scripts/fl_il_to_bc.py:550`），尾部追加 ⇒ **必须重生成语料**。
但**不能**「旧 58 语料 vs 新 75 语料」这样配对，原因已登记（**C17**）：

> 同代码同 seed、**两次独立进程**，样本差 **23/388（5.9%）**，而同进程内 **0/42** —— 跨进程不可复现。

⇒ 采用§6.2 的**同形状对照臂**：**只生成一份 75 维语料，三臂共用**。
「前 58 维在开关前后逐位不变」这条不变量**不能**用两份语料比对（原理上不可达），
改用**同进程双跑探针**证明：`scripts/probe_plan_extras_prefix.py`（P1 obs/belief/masks 逐位、
**P2 `plan[:58]` 逐位**、P3 追加维非零、P4 关闭时恒 58 维）。

### 6.4 臂表与判据（**跑前冻结**）

语料：`runs/_fl_il_bc_hs/{train,holdout}`（`--stop-mode save --stop-stride 4 --plan-extras`，
**2200 局**，与 C17/H1 那批**同源命令**）。
训练：`--configs 3:1e-3 --mix-ratio 1.0 --seed {0,1,2}`（3 epoch = 与混比四臂同强度）。

| 臂 | out-dir | plan_dim | `--plan-extras-zero` | 含义 |
|---|---|---|---|---|
| **Z17** | `runs/_fl_il_bc_hs/sweep_mixHS17/s{k}` | 75 | **17** | **对照组**：17 维存在但抹零 ⇒ 零新信息 |
| **Z04** | `runs/_fl_il_bc_hs/sweep_mixHS04/s{k}` | 75 | **4** | 只有 **W2（手牌打分 13 维）** 活着 |
| **Z00** | `runs/_fl_il_bc_hs/sweep_mixHS00/s{k}` | 75 | **0** | **W2 + W3** 全活 |

| 判据 | 内容 |
|---|---|
| **J-W2.3（取代 §2.4 原式）** | 配对 **Z04 − Z17**（同 corpus、同 seed）：**`mean_k ΔNLL ≤ 0` ∧ `mean_k Δact-AUC ≥ 0`** ∧ ≥2/3 seed 方向一致 ⇒ **PASS**。若 `mean_k ΔNLL > 0 ∧ mean_k Δact-AUC < 0` ⇒ **无收益 ⇒ 回滚特征**（保留 §6.2 的不变量修复）。**逐 seed 值必须全部列出**（【R4】脚本复算）。 |
| **J-W2.4（行为带）** | 两臂 `plays/game` 均落在 **[18, 28.5]**；越界**只登记不判负**（§2.4 原写「双侧带不越界」⇒ 此处**放宽**：本语料与四臂**不同源**，带值 **R15 不可跨实验照抄**，故带**只作描述**，判定**只用配对方向**）。 |
| **J-W3.2（取代 §3.3 原式）** | 配对 **Z00 − Z04**：`info_window.known_opp_median_in_window` 的 **seed 均值 ≥ 0**（**方向判据**，R15/R16：不设绝对阈值）。另报 `plays_in_window_per_game` / `first_play_t_median` 作**描述**。 |
| **J-W4.1（【R2】前缀不变量）** | `probe_plan_extras_prefix.py` 全 PASS（同进程 P1/P2/P3/P4）。 |
| **J-W4.2（回归）** | `scripts/selftest_hand_score.py` **44/44 PASS**。 |
| **J-W4.3（ckpt 兼容，§2.4 J-W2.2）** | 58 维 ckpt 载入 75 维网络：不报错 ∧ 新增 17 列**严格 0** ∧ 前 58 列逐位拷贝（回归 T7.3/T7.4）。 |

**为什么新语料不必与混比四臂可比**：`R00/025/10/337` 是**另一批语料**（无 extras）⇒ 本批
**自带宽度的对照臂**（Z17），不跨批对拍（【R17】口径不同不共用数字）。

### 6.5 复算命令（全部相对路径；⚠️ 本环境 Windows python 不认 `/mnt/...`）

```bash
# 1) 回归（44 断言）
PYTHONDONTWRITEBYTECODE=1 ./.venv/Scripts/python.exe scripts/selftest_hand_score.py

# 2) 前缀不变量（同进程双跑；需 FL 回放）
PYTHONDONTWRITEBYTECODE=1 ./.venv/Scripts/python.exe scripts/probe_plan_extras_prefix.py --games 2

# 3) 语料（2200 局；约 40 min @12 workers）
./.venv/Scripts/python.exe -u scripts/fl_il_to_bc.py --mode samples \
  --jsonl ../fl_il_data/replays_part000000.jsonl --games 2200 --workers 12 \
  --level 11 --coord raw --stop-mode save --stop-stride 4 --plan-extras \
  --out-dir runs/_fl_il_bc_hs --out docs/fl_il_2026-09-21/samples_stop_save_hs.json

# 4) 三臂 × 三 seed（每跑约 50 min；串行 3 个/批）
for z in 17 04 00; do for s in 0 1 2; do
  ./.venv/Scripts/python.exe scripts/il_bc_sweep.py \
    --data-dir runs/_fl_il_bc_hs/train --out-dir runs/_fl_il_bc_hs/sweep_mixHS$z/s$s \
    --configs 3:1e-3 --seed $s --mix-ratio 1.0 --plan-extras-zero $((10#$z)) --threads 4
done; done

# 5) 留出 + 部署读数（每臂每 seed）
./.venv/Scripts/python.exe scripts/il_eval_holdout.py \
  --ckpt runs/_fl_il_bc_hs/sweep_mixHS17/s0/bc_fl_e3_lr0.001.pt \
  --data-dir runs/_fl_il_bc_hs/holdout --out docs/fl_il_2026-09-21/holdout_hs17_s0.json
./.venv/Scripts/python.exe -u scripts/il_readout_games.py \
  --ckpt runs/_fl_il_bc_hs/sweep_mixHS17/s0/bc_fl_e3_lr0.001.pt --games 10 --decks solo \
  --hidden none --opp-hidden same --out runs/il_readout_hs17_s0 \
  --json docs/fl_il_2026-09-21/readout_hs17_s0_stats.json

# 6) 台账（脚本复算，禁手抄）
PYTHONDONTWRITEBYTECODE=1 ./.venv/Scripts/python.exe scripts/il_hs_report.py \
  --out docs/fl_il_2026-09-21/hs_report.json
```

### 6.6 成本与分辨率（【R5】/【R16】，跑前登记）

| 项 | 值 |
|---|---|
| 语料规模 | 2200 局（实测 **~1.0 s/局** @10 workers ⇒ **~37 min**） |
| 单跑 | 3 epoch × (38.7k play + 38.7k save) ≈ **50 min**（按混比四臂 `2629 s / 77.5k×3ep` 折算） |
| 总跑数 | **9**（3 臂 × 3 seed）⇒ 3 并发 ≈ **2.5–3 h** |
| 分辨率声明 | 3 seed 是**同数据同处理**的重复（只有初值/permutation 不同）⇒ 用**均值方向**判定，逐 seed 值全部登记。**不做**跨批（混比四臂）比较 |
| 已知不确定性 | `S_opp` 的 `hand_probs` **本来就在 belief token 里**（§2.1 已写）⇒ 这 4 维是**再编码**不是新信息；`ê_opp` 是**有偏估计**（O-d）。⇒ 若 W2 无疗效，**不能**据此断言「手牌打分无用」，只能说「**在这个估计器 + 这个再编码下**无疗效」（【R10】） |

### 6.7 本批仍**明确不做**（照实登记）

- **不改奖励**（【R11】）；不改掩码（W1 已交付，本批零掩码改动）。
- 不修 `Lightning` 谓词（O-a）、`Log`/`BarbLog` 落点（O-b/O-c）、对方圣水分期回费估计器（O-d）。
- 不把 W3 的信息分接进 `BeliefPlanner` 的意图选择（那是 planner 的行为，不是观测；本批只注入特征）。

### 6.8 归因仪器（**描述性，非门禁**；跑前写死）

主判据 J-W2.3 只给**总效应**。若它判「无收益」，有两种解释、出路完全不同：
(a) 特征无增量信息；(b) 信息在但没学进去。⇒ 预先写死一件**归因**仪器把它分开：

`scripts/il_hs_feature_use.py`（**只对全活臂 HS00 适用**；对 `plan_extras_zero>0` 的 ckpt 显式拒绝，
因为它的前向已经把尾列抹零、扰动无意义）：

| 干预 | 做法 | 读法 |
|---|---|---|
| `zero` | 17 列置零 | 同时改了输入尺度 ⇒ 会被 `plan_mlp` 偏置部分吸收 ⇒ **低估**使用度，只作参考 |
| `perm` | 17 列**整块跨帧打乱** n=4 次 | 保留边际分布、只破坏「帧 ↔ 特征」对应 ⇒ **干净**检验 |

判读（【R10】：不设绝对阈值）：`ΔNLL(perm) ≈ 0`（相对 `perm_lp_std_across_shuffles/√n` 的抖动尺度）
⇒ 模型**没用**这些列；明显大于 ⇒ 在用。**n=4000 帧、seed 0、逐帧配对**，结果**照实登记**。

本仪器**不改任何判决**，只在汇报里回答「为什么不生效」（若确实不生效）。

---

## 6.9 结果（2026-09-22 跑完；**未预注册的第三分支已登记**）

语料：`runs/_fl_il_bc_hs`，2200 局 / **1711 s** / `plan_dim=[75]` / play **47,270** + save **160,668** /
`frames=736,277` / `team_off=685,867` / `save_cand=638,279` / `mask_reject=0.0577` / **errors 0**。
九跑：`--configs 3:1e-3 --mix-ratio 1.0 --threads 4`，池 **76,926**（= play 38,463 + save 38,463），
单跑 **37.7–40.0 min**，**0 报错**（`docs/fl_il_2026-09-21/hs_arms.log`）。

### 6.9.1 逐 seed 全表（脚本复算：`docs/fl_il_2026-09-21/hs_report.json`）

| 臂 | seed | act-AUC | play top1 | play NLL | cell | macro | 出牌/局 | 帧/局 | 胜率 | 买得起不出 | 窗口已知卡 | TVD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **HS17**（对照） | 0 | 0.6483 | 0.2022 | 6.1438 | 0.0508 | 0.1617 | 32.6 | 329.9 | 0.70 | 0.0659 | 4.00 | 0.0889 |
| HS17 | 1 | 0.6518 | 0.2040 | 6.0469 | 0.0545 | 0.1720 | 34.3 | 339.9 | 0.40 | 0.1160 | 4.00 | 0.0901 |
| HS17 | 2 | 0.6495 | 0.2648 | 6.0017 | 0.0560 | 0.2049 | 32.4 | 328.1 | 0.40 | 0.0663 | 4.00 | 0.0992 |
| **HS04**（W2） | 0 | 0.6505 | 0.2257 | 6.0682 | 0.0612 | 0.1708 | 33.4 | 332.4 | 0.50 | 0.0722 | 4.00 | 0.0761 |
| HS04 | 1 | 0.6620 | 0.1959 | 6.0990 | 0.0512 | 0.1619 | 35.8 | 350.7 | 0.40 | 0.1268 | 4.00 | 0.0965 |
| HS04 | 2 | 0.6523 | 0.2591 | 6.1319 | 0.0452 | 0.2038 | 31.4 | 319.3 | 0.60 | 0.0710 | 4.00 | 0.0909 |
| **HS00**（W2+W3） | 0 | 0.6523 | 0.2116 | 6.1301 | 0.0534 | 0.1673 | 37.0 | 361.4 | 0.30 | 0.1127 | 4.00 | 0.0778 |
| HS00 | 1 | 0.6553 | 0.2053 | 6.0506 | 0.0554 | 0.1766 | 33.6 | 337.1 | 0.30 | 0.0795 | 4.00 | 0.0873 |
| HS00 | 2 | 0.6585 | 0.2403 | 5.9724 | 0.0593 | 0.1922 | 36.0 | 348.3 | 0.50 | 0.0296 | 4.00 | 0.0680 |

**设计有效性（脚本自检，3/3 seed PASS）**：`HS17 vs HS04` 的 `plan_mlp.0.weight`
—— W2 段（58..70）逐 seed `max|Δ| = 3.93 / 3.49 / 3.16`、**W3 段（71..74）`== 0.0`**；
`HS04 vs HS00` 的 W3 段 `2.13 / 1.74 / 1.50`。⇒ **只有该动的列在动**，`plan_extras_zero` 真接上了。

### 6.9.2 J-W2.3（W2 手牌打分）：**FAIL** —— 且落进了**未预注册的第三分支**

| 配对（HS04 − HS17） | seed 0 | seed 1 | seed 2 | **均值** |
|---|---|---|---|---|
| Δact-AUC | +0.0022 | +0.0103 | +0.0028 | **+0.0051**（**3/3 同号**） |
| **ΔNLL** | **−0.0755** | +0.0521 | +0.1302 | **+0.0356** |
| Δtop1 | +0.0235 | −0.0082 | −0.0057 | +0.0032 |

- 判据要求 `mean ΔNLL ≤ 0 且 mean Δact-AUC ≥ 0` ⇒ **ΔNLL 不满足 ⇒ J-W2.3 FAIL**；
- ⚠️ **§6.4 只注册了两条分支**：PASS，或「`mean ΔNLL > 0` **且** `mean Δact-AUC < 0` ⇒ 无收益 ⇒ 回滚」。
  实测落在 **`ΔNLL > 0` 而 `Δact-AUC > 0`** 的**第三分支（未预注册）** ⇒ **照 §6.8 先例登记为
  预注册缺口，不自作判决**。本次的处置（保守）：
  **不把 W2 采用为已验证的改进**（一条必要条件未过），**也不回滚代码**（回滚条件未触发），
  将其作为**默认关、可选**的能力保留（与 `decoupled_act` 同处置）。
- **【R10】不许写「W2 有害」**：ΔNLL 的**逐 seed 散布（−0.076 … +0.130，极差 0.206）远大于均值 0.036**
  ⇒ 在该池/该训练强度下 **W2 的端到端效应与种子噪声同量级**，方向上不可分（1/3 seed 反而更好）。
  唯一**方向一致**的是 act-AUC（3/3 正，均值 +0.0051）—— 幅度小，符号检验 p=0.125，**不足以当结论**。

### 6.9.3 归因（§6.8 仪器，**这是本批最有信息量的读数**）：特征**确实被用上了**

| 干预（HS00 s0，前 4000 帧：play 934 / stop 3066） | ΔNLL |
|---|---|
| `zero`（17 列置零） | **+0.0132** |
| **`perm`（整块跨帧打乱，n=4）** | **+0.0811** |
| `perm` − play 帧 | **+0.1902** |
| `perm` − stop 帧 | +0.0478 |
| perm 抖动尺度（逐帧 std 0.1824 / √4000） | **≈ 0.0029** |

- ⇒ `ΔNLL(perm) = +0.0811` **≈ 28× 抖动尺度** ⇒ **模型在用这 17 列**，且**出牌帧上的依赖（+0.190）
  远大于不出牌帧（+0.048）** —— 与「手牌打分主要是给『出不出/出哪张』用的」一致。
- **负对照（仪器无假阳性）**：同一仪器跑**抹零臂 HS17 s0** ⇒ `ΔNLL(perm) = ΔNLL(zero) = 0.0`
  **恰好为 0**（`hs_feature_use_HS17_s0_NEGCTRL.json`）。⇒ +0.081 不是「任意扰动都会变差」的假象。
- **合起来是本批最要紧的一句**：**信息进得去、网络也学得会用**（归因强阳性），
  **但端到端配对指标分辨不出收益**（J-W2.3 FAIL，效应在种子噪声内）⇒ 瓶颈**不在「特征没被利用」**，
  而在**该训练强度/该池构成下端到端可分辨的收益本身就不存在或极小**。

### 6.9.4 J-W3.2（W3 信息分）：**判据失效（无分辨力）——「PASS」是空的**

- 配对（HS00 − HS04）的 `known_opp_median_in_window` **三个 seed 全部 = 0.0**，
  而该量在**全部 9 个 run** 上**都等于 4.00**（`frames_in_window` 也恒 610）⇒ **统计量饱和**，
  ⇒ 按 §6.4 的字面判据（`均值 ≥ 0`）**形式上 PASS，实质零分辨力**。
  **照实登记为「判据失效」，不得写成「W3 有效」**（同 J6.3① 「只写上界」那次的病）。
- 换更细的同源读数看：`known_opp` 的**均值** 3.398–3.648、**max 恒 5**、
  `plays_in_window` 5.6–5.9、`distinct` 恒 5.0 —— **臂间差 < 臂内 seed 散布** ⇒ **不可分辨**。
- 附带的（未预注册、**描述性**）配对：ΔNLL 均值 **−0.0487**（−0.048/+0.062/−0.160 中的两负），
  Δact-AUC 均值 +0.0004、Δ出牌/局 均值 +2.0（+3.6/−2.2/+4.6）
  ⇒ **方向偏正但逐 seed 不稳、散布 > 均值** ⇒ **不构成结论**。

### 6.9.5 J-W2.4（行为带，**只登记不判定**）

`plays/game` 全部 9 个 run 落在 **31.4–37.0** ⇒ **0/9 在 [18, 28.5] 带内，全部高于上界**。
与 C19/C25 的「混比池会过度出牌」**同一现象**；带值来自**另一批语料** ⇒ 按【R15】**不作判据**。
`winrate` 0.30–0.70（10 局/跑，SE≈0.15）⇒ **不可分辨**。

### 6.9.6 本批净结论

1. **基础设施**：plan 尾部注入通道（17 维，纯函数、可单测）、`PLAN_HOLD_OFFSET` 不变量修复、
   同形状消融开关（`plan_extras_zero`）、推理侧同口径、归因仪器 —— **全部交付且门禁 PASS**
   （回归 44/44、前缀探针全 PASS、消融列自检 3/3、负对照恰好 0）。
2. **W2 疗效**：**未过判据**（J-W2.3 FAIL），**效应与种子噪声同量级**；**但不是「没用」** ——
   归因显示网络**显著在用**这些特征（+0.081 ≫ 0.0029）。⇒ 下一刀的着力点**不是再加特征**，
   而是**训练强度/损失/池构成**（与 C19/C25 的结论同向）。
3. **W3 疗效**：**不可分辨**，且**原判据饱和失效** ⇒ 若要继续验 W3，**必须先换一个有分辨力的仪器**
   （例如按 t=10/20/30 s 分桶的 `known_opp` 曲线，而不是中位数），**并重新预注册**。
4. **未预注册的第三分支**（`ΔNLL>0 ∧ Δact-AUC>0`）已登记；处置 = **不采用、不回滚、默认关**。

