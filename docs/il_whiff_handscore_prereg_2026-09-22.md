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
