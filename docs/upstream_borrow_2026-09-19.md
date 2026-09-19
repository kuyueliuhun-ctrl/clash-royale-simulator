# 原作者新版训练方法：**我们能借鉴什么**（2026-09-19）

> **触发**：用户问「按照原作者新版的训练方法，文档在 `docs/upstream_update_vs_ours_2026-09-19.md`，我们可以借鉴什么」。
> **取证对象**：上游 `/mnt/e/clash-royale-simulator-main-by-jason`（HEAD `f616f19`）、我们 `/mnt/e/clash-royale-simulator-main`（工作区）。
> **取数时间**：2026-09-19。
> **证据纪律**：每条带 `文件:行号` / commit sha；区分【事实】【实测读数】【推断】【未定】；没有读到的写「查不到」，
> 不臆造。**本文不主张任何疗效**（【R3】非判据、【R5】n=1 seed）。
> **与前文的关系**：`docs/upstream_update_vs_ours_2026-09-19.md`（下称「前文」）做的是**机制对照**（已核证的部分见本文 §1）；
> 本文做的是**「借什么、不借什么、借之前要预注册什么」**，并补一台前文没有的仪器（§2）。

---

## 0. 结论速览（先看这张表）

| # | 借鉴项 | 上游做法（sha） | 我们现状（实测） | 判决 |
|---|---|---|---|---|
| **B1** | **熵项里删掉落点熵** | `7fc64f2` 把 `Σ_slot p(slot)·(y_ent + x_ent)` 整块删除，只留**卡熵** | ★ **我们的熵奖励 85.5%–90.0% 付给了落点（576 类）头**（§2） | **借**（唯一一处「上游改了 learner 损失、我们没跟」） |
| **B2** | **对手课程：把「会攒费 + 会解场」做进脚本对手** | `8744065`/`e049d55`：`elixir < 8 → 不出牌`、聚团法术（≥3 且 2.5 格内 ≥3）、取**最深入**威胁、无解不裸下 | `SelfDefenderPolicy`（权重 0.2）**四条都没有**；`run` 模式 5 个对手**全是 mask 随机** | **借**（唯一对准 S1 判定「约束在机制/探索侧」的方向；台账 O4 已登记候选） |
| **B3** | **记 `approx_kl`**（先量再谈 `target_kl=0.03`） | `target_kl=0.03`（`train_autoregressive.py:325` 上下文） | 我们自研 `rl/ppo.py` **没有任何 KL 统计**（只有 `ratio_mean`/`clip_frac`/`grad_norm`） | **借（零风险，只加记录）** |
| **B4** | **「被迫不出牌 vs 主动不出牌」做成**训练内**诊断** | `1a86041` 加进 `evaluate.py` | 我们只有**离线**探针（`probe_pass_prob.py` / `pass_streak_audit.py`） | **借（描述性）** |
| **B5** | **吞吐：多环境 + 更长训练** | 16 env ×（单次 5M，**热启动自 24,694,976 步 ckpt**） | `--n-envs` 默认 **1**，已记录的所有启停命令都没用它；`act_parallel` 已实现且 run 模式在用，**solo 完全没接线** | **可选**；★ **但按我们自己的证据它不解 O7**（§3.5） |
| **N1** | 把 STOP 负偏置改中性 | `noop_head` 独立、无偏置 | `stop_logit_bias=-1.0`；★ 实测「改中性」≡ 剂量 β=1，**落在死区**（β=0/4 实测 0.000%/0.178%） | **不借**（§4.1） |
| **N2** | 8 帧观测堆叠 | `(8,32,18,15)` | 无堆叠；O2 P3′ 已证**信息不是缺口** | **不借**（R11 扩参 + R6 `--fresh`，§4.2） |
| **N3** | 576 类联合落点头 | `f616f19` `placement_head = Linear(hidden, 32*18)` | ★ **我们本来就是**：`rl/follower.py:238` `cell_head = nn.Linear(hidden, GRID_H*GRID_W)` | **零增量**（§4.3） |
| **N4** | 往奖励里加圣水项 | 上游**完全没有**圣水项 | 我们有 `edw=0.5`，且 `engagement_trade` 已判**不接线** | **不借**（与前文④一致） |

**一句话**：前文 §3.5 的四条建议里，**③ 的事实前提是错的**（我们有 `defend=0.2`，缺的是**强度**不是**存在**）；
①②④ 分别是「已被我们的剂量-反应实测否证」「与 R11/R6 硬冲突且方向已被证伪」「与前文自己一致」。
**真正的空白是前文漏掉的 B1**——上游这次唯一动 learner 损失的地方（熵项），我们**没有跟**，而且实测它占了我们熵奖励的 **~90%**。

---

## 1. 先纠正四处事实（前文与本仓台账的口径对齐）

### 1.1 前文 §3.5 建议③ 的前提与它自己的 §2.1 矛盾 —— **我们有，但弱**

- 前文 `:136`（§2.1 表）自己写着 `solo` 对手池 = `frozen .1 / hist .6 / **defend .2** / rand_anchor .1`；
  `:161`（§2.3）又承认「`solo` 对手池里有 `SelfDefenderPolicy`」；但 `:216`（§3.5 ③）写成「给我们**补一个**外生的强防守脚本对手 —— 这是我们与作者最直接的结构差」。
- 【事实】`defend` 早就在池里：`rl/config.py:105` `DEFAULT_OPP_MIX = {"frozen":0.1,"hist":0.6,"defend":0.2,"rand_anchor":0.1}`、
  兜底 `rl/train_solo.py:145`、实例化 `rl/train_solo.py:346-347`、类定义 `rl/opponents.py:125`。
- ⇒ **正确的改写是「把已有的 20% defend 变强」，不是「补一个」。** 强度差距见 §3.2。

### 1.2 「STOP 被结构性压制」：机制描述要改，**实测结论反而更强**

- 【事实·机制】`rl/follower.py:156` `stop_logit_bias=-1.0`，施加点 `:253-255`
  `if stop_logit_bias: self.slot_head.bias[STOP_IDX] += stop_logit_bias` —— 位于 `__init__` 内、**只加一次**，
  所以严格说它是**初始化先验**（梯度理论上可以把它推回去），不是「每次前向都加的常量」。
- 【实测】扫 53 个 ckpt（`et_solo100k` 14 个 + `et_solo` 6 个 + `run100k` 14 个 + `long1m` 19 个）：
  `slot_head.bias[STOP]` ∈ **[−0.9931, −0.9382]**，均值 **−0.9491**；第 0 步 ckpt（`solo_main_0.pt`）−0.9435 → `long1m/main_ckpt_128000.pt` −0.9440。
  ⇒ **128,000 步、三种模式，这个先验的位移 < 0.06**。机制上是先验，**功能上确实等于固定偏置**。
  （仪器 `.tmp/stop_bias_scan.py`，只读；53 个 ckpt 全部 `dim=(6,)` / `plan=58` / `belief=563`。）

### 1.3 前文 §1.6 说「learner 的网络与奖励一行没动」不精确

- 【事实】`7fc64f2`「Update entropy calculation method」**改的正是 learner 的损失项**（`train_autoregressive.py` 的
  `_conditional_entropy`，见 §3.1 逐字 diff）。前文把它列作「熵改算法 · 纠偏」，却在 §1.5 写「learner 的网络与奖励**一行没动**」。
  ⇒ 本文把 B1 提为一等候选，正是因为**这是 9 个提交里唯一动 learner 的地方**，而前文 §3.5 的建议清单里没有它。

### 1.4 规模：上游不是「5M 步」，是**从 24,694,976 步热启动**

- 【事实】HEAD `train_autoregressive.py:318` 附近：`start_checkpoint = "cr_masked_moe_dir/cr_24694976_steps.zip"`（`7fc64f2` 的 diff 上下文行，未改动）；
  `7fc64f2` 新增 `load_checkpoint = model_name if os.path.exists(f"{model_name}.zip") else start_checkpoint`。
- 【事实】`evaluate.py:295-297` 评估 `cr_autoregressive_moe_dir/cr_{5,10,15}000000_steps.zip` ⇒ ckpt 谱系 **≥15M 步**。
- 【事实】`train_autoregressive.py:310-315` `n_envs=16`、`n_steps=8192//16=512`；`:338` `total_timesteps=5_000_000`（= **单次续跑预算**，不是累计量）。
- ⇒ 对比我们：`long1m` 提前终止于 **128,000 步**（12.8%），`run100k` = 100k 步。**差距 ≈ 193×**（24,694,976 / 128,000）。
  ⚠️ 这个数字只能用于**拒绝「按步数对拍」**，不能用于「他们更强」（前文 §2.1 的口径声明仍然成立）。

---

## 2. 新仪器：熵项预算分解（`scripts/probe_entropy_decomp.py`）

### 2.1 方法（为什么可信）

- 轨迹 = 生产 `FollowerPolicy.act(..., deterministic=False)` 的**真实采样**（训练看到的就是这个）；
- 分解用的 ops **逐字复刻** `evaluate()` 的三个记帳点：deploy 步 `slot_dist.entropy()`（`rl/follower.py:775`）、
  cell 步 `cell_dist.entropy()`（`:790`）、终止（STOP）步 `slot_dist.entropy()`（`:804`）；
- **自检**：分解之和必须等于 `evaluate()` 返回的总熵 —— 六次跑（含 40 帧 smoke）共 **5,782 帧，超标 0 帧**，最大偏差 **5.96e-07**。
  ⇒ 分解不是「另一份实现」，而是同一前向的**同一批张量**分开记账。

### 2.2 读数（同 ckpt 四次独立轨迹；**采样 RNG 未播种 ⇒ 这是散布，不是重复**）

| 臂 | 帧数 | 总熵/帧 (nat) | **落点(cell 576)** | deploy 槽位 | STOP 步 |
|---|---|---|---|---|---|
| `et_solo100k/solo_main.pt` | 945 | 0.2981 | **89.82%** | 9.60% | 0.57% |
| 同上 | 924 | 0.2817 | **90.04%** | 9.36% | 0.60% |
| 同上 | 762 | 0.3032 | **87.59%** | 10.88% | 1.53% |
| 同上（40 帧 smoke） | 40 | 0.4769 | **85.50%** | 14.50% | 0.00% |
| `fresh_init`（未训练） | 1195 | 0.5701 | **92.92%** | 6.31% | 0.77% |
| `fresh_init` | 812 | 0.5234 | **89.84%** | 8.29% | 1.87% |

- ⇒ **落点项占 85.5%–90.0%（已训练）/ 89.8%–92.9%（未训练）**；熵奖励幅度（×`ent_coef=0.01`）= **0.0028–0.0030/帧**（40 帧 smoke 0.0048），其中落点 0.0025–0.0027。
- 合法格数：每个 deploy 步平均 **300–313 格**（上限 576；`ln 306 ≈ 5.72 nat` 可用），**落点是 576 类的大空间**。
- 合法手牌选项分布（已训练 945 帧）：**0 项 860、1 项 78、≥2 项 7** ⇒ **91.0% 的决策帧没有任何可出的牌**。
- STOP 步熵：**没得选的帧 860/860 恰好 = 0.0000**；有得选时 0.0190/帧 ⇒ STOP 这一项在我们这里是**舍入误差**（0.57%–1.53%）。
- 圣水：全帧中位 **1.50**；**≥6 的帧 0/945**；有合法选项的帧圣水中位 **3.07**、最高 **5.00**
  ⇒ 「能出却不出且圣水 ≥6」在这批轨迹里**结构性不可达**（自洽于 `docs/elixir_saving_audit_2026-09-18.md:113`：
  真实 run 里 390 帧 ≥6 有 **375 帧（96%）是 B 类整包被拒的被动上涨**，真正「主动不出牌且圣水≥6」= **0 帧**）。
- ★ **反事实**：照上游口径「只对手牌取熵」（`logits[:, 1:]`）而**不加** `eligible` 门控，会得到 **1.374–1.389/帧 = 已训练总熵的 4.55–4.57 倍**（未训练 2.4–2.6 倍）——
  因为 0 张可打时那 4 个 logit 全是 `-1e9` ⇒ 分布**退化成均匀**（熵 = `ln 4 ≈ 1.386`），熵奖励反而**暴涨**。
  ⇒ 上游那个 `eligible = affordable_count >= 2` 门控**不是精修，是这项改动的必要组成部分**（它自己的注释也这么写）。

### 2.3 这些读数同时**证否**了两件事

1. 「我们的熵奖励在鼓励随机 STOP」—— **不成立**：STOP 项只占 0.57%–1.53%，且 91% 的帧它的熵恰好是 0（STOP 是唯一合法项）。
2. 「上游的 `eligible` 门控能救我们」—— **几乎没有作用面**：我们的 `masked_fill` 在求熵**之前**，0 项/1 项可打的帧熵本来就是 0；
   门控唯一的作用是掐掉那 1 项类帧的小尾巴（≤ 12% 的熵预算），而它**代价**是引入一个逐 minibatch 浮动的有效 `ent_coef`。

---

## 3. 可借项（按「证据强度 × 是否对准病根」排序）

### 3.1 B1 · 熵项：删掉落点熵（上游 `7fc64f2` 的落点那一半）

**【上游逐字】**（`git show 7fc64f2 -- src/clasher_new/train_autoregressive.py`）

```diff
-    def _conditional_entropy(self, latent_pi, card_dist, hand):
-        entropy = card_dist.entropy()
-        for slot in range(1, 5):
-            selected_card = hand[:, slot - 1]
-            y_dist, x_dist = self._placement_distributions(latent_pi, selected_card)
-            entropy = entropy + card_dist.probs[:, slot] * (
-                y_dist.entropy() + x_dist.entropy()
-            )
-        return entropy
+    def _conditional_entropy(self, card_dist, hand, obs):
+        """Card entropy only, conditioned on playing an affordable card."""
+        card_distribution = Categorical(logits=card_dist.logits[:, 1:])
+        entropy = card_distribution.entropy()
+        elixir = obs["elixir"].float().reshape(-1, 1)
+        affordable_count = (self.card_cost_table[hand] <= elixir).sum(dim=1)
+        eligible = affordable_count >= 2
+        scale = eligible.numel() / eligible.sum().clamp(min=1)
+        return entropy * eligible.float() * scale
```

三件事一起做：**(a) 把 noop 从熵里剔除**（`logits[:, 1:]`）、**(b) 落点熵整块删除**、**(c) `eligible` 门控 + batch 归一 `scale`**。

**【我们的现状·实测】** 见 §2：`(b)` 正是我们熵预算的 **~90%**；`(a)` 对我们**几乎无影响**（STOP 项 0.57%–1.53%）；
`(c)` 对我们**几乎没有作用面**（掩码后 0/1 项帧的熵已经恰好是 0）。
⇒ **应借的只有 (b)**，且 (c) 不借 —— 这是一个**干净的单变量改动**：只停付落点熵，卡/槽位探索压力**逐值不变**。

- **成本**：低。纯函数（`evaluate()` 与 `evaluate_batch()` 两处累加点），**不动网络结构 ⇒ 不必 `--fresh`**。
- **红线**：改训练语义 ⇒ **R2**（只能经 `TrainConfig` 显式开，默认保持旧行为）；**R3**（须另行预注册）；
  **R19 例外**（`rl/ppo.py`/`follower.py` 属共享底层 ⇒ 要跑全量 selftest）。
- **【未定·必须写死】** 疗效：**未知**。请特别注意 §2.3①——这项改动**不解决 O7**（我们的熵奖励本来就不是付给 STOP 的）。
  它的正当性是「不再为 576 格的大空间均匀探索付钱」，**不是**「能学会攒费」。
- **预注册应写死的判据**：分解读数（落点项占比 → 0）× 「能出却不出」计数 × 落点熵时间序列；
  **失败分支照抄** `docs/exploration_pressure_gate_2026-09-18.md:156-159` 的写法：
  「若……不高于阴性对照，记『本分辨率下无结论』，**不得**写『无效』」。

### 3.2 B2 · 对手课程：把已有 `defend`（0.2）从「会防守」升级为「会攒费 + 会解场」

**【上游逐字可搬的四条】**（`src/clasher_new/defensive_strategy.py`，提交 `8744065` + `e049d55`）

| 机制 | 上游逐字 | 行号 |
|---|---|---|
| 硬攒费门槛 | `if elixir < 8: return 0, 0, 0` | `:139-141` |
| 聚团法术 | 敌部队 ≥3 **且** 局部 2.5 格内 ≥3 才交 Fireball/Arrows | `:85-104` |
| 取最深入威胁 | `min(intruders, key=lambda e: e["y"])` | `:109` |
| 无解不裸下 | 解不掉就 `return 0, 0, 0` | `:121,:158` |
| 只看敌**部队**（塔/建筑/法术不算） | `owner==1 and type==1` | `:66-69` |

**【我们的现状】**（`rl/opponents.py:125-215` + `simulate_exchange.py`）
- 有：真反制（`script_defender`：跳过法术与仅攻建筑卡、按 `(DPS + HP/15)/费` 选、塔前迎击线）
- **没有**：攒费门槛（只有 `passive_prob=0.6` 的停手概率，费用约束是隐含的 `p.can_play_card`）、聚团法术判据、
  「解不掉就不出」、最深入威胁（它取的是威胁**质心** `simulate_exchange.py:112-113`）
- 权重只有 **0.2**，且**从未被加强**（`git log -- src/clasher_new/rl/opponents.py` 只有建档 + 四卡组换血）
- `run` 模式的 5 个脚本对手在有三分类卡组时**全是 `ScriptedPolicy(mode="random")`**（`rl/run_league.py:695-701`）
  = **mask 随机**；`heuristic` 只出现在找不到卡组的退回分支（`:708-712`）

**为什么这条最值得考虑**：S1 门禁已判「**约束主要在探索/机制侧**」，且手写「攒费→Xbow」规则能打出 **156× 基线**
（`docs/s1_gate_2026-09-18.md:23,231-241`）⇒ 「把攒费/解场做成**环境侧课程**而不是奖励项」正好落在这个判定上，
而台账 O4 候选里已经登记了「课程式提高 `defend` 权重」（`docs/agents/ledger.md:59`）。

- **上游自己的前车之鉴**：`d428b79` 给对手加了 `DiverseOpponent` 随机化，但把 `gap/offset/reserve/push_y/spell_count`
  **写死**，注释自陈 *"Broad placement jitter weakened tower defense in benchmarks."* ⇒ **只随机化门槛/取舍，不随机化落点**。
- **成本**：中。改 mix 须**同步两处**（`docs/agents/metrics.md:55`）；脚本本体动 `rl/opponents.py` + `simulate_exchange.py`。
- **红线**：R2（对手分布 = 训练语义）、R3（须预注册）、R7（`opp_mix` 单常量源两处同源）。
- **判据禁则**：**不得用胜率**（R5/R16 + `et_solo100k` 判读 §11.13.11 实测同配置单点差 **0.60**）。

### 3.3 B3 · 记 `approx_kl`（先量，再决定要不要 `target_kl`）

- 【事实】`grep -n "kl" src/clasher_new/rl/ppo.py` 只有注释里提到 `ratio≡1`；实际记录项仅 `ratio_mean` / `clip_frac` / `grad_norm`（`:396-401`）。
- 【事实】上游 `target_kl=0.03`（`train_autoregressive.py:325` 上下文），且是 **SB3 内置**行为（本机未装 SB3，**未核实**其精确语义 ⇒ 写「未核实」，不臆造）。
- ⇒ **借法 = 只加一条记录**（不设阈值）。加完之后再看我们自己的 KL 分布落在哪 —— 直接照抄 0.03 触 **R15**（阈值不许跨实验照抄）。
  成本极低；仍触 R2/R19（改 `rl/ppo.py`）。

### 3.4 B4 · 「被迫不出牌 vs 主动不出牌」做进训练内诊断

- 【事实】上游 `1a86041` 在 `evaluate.py` 里加了 `noop` 诊断（前文 §1.6 记为「被迫不出牌 vs 主动不出牌」）。
- 我们**已有等价概念的离线仪器**（`scripts/probe_pass_prob.py`、`scripts/pass_streak_audit.py` 的 A/B/C 分类），
  缺的只是「**训练循环内**每 `--diagnose-every` 就顺带记一次」。这是**描述性**读数，不进判据。

### 3.5 B5 · 吞吐（可选，且**不要**把它当解法）

- 【事实】`rl/run_league.py:1523-1524` `--n-envs` 默认 `None`→1；`docs/run100k_2026-09-18.md:38-43` 的启停命令**没有**用它；
  `act_parallel` 已实现（`rl/follower.py:529+`，含「与 `act()` 逐位一致」的 selftest `rl/selftests/part2.py:855`）
  且 run 模式已接线（`rl/run_league.py:1173,1366`），但 `grep -n "n_envs\|act_parallel" rl/train_solo.py` = **0 命中**。
- ⇒ 借法 = 复用已有设施（不是写新东西）。
- ★ **但按我们自己的证据，更多步数不解 O7**：`docs/elixir_saving_audit_2026-09-18.md` 的 0 样本论证
  （step 0 与 100k **同结构**：放弃买得起的牌 0/75,891）；`docs/exploration_randomization_analysis_2026-09-18.md`
  解析成功率 **4.5×10⁻¹⁰ / 100k 帧** ⇒ 就算跑 5M 帧也只多 ~1.7 个数量级，仍在「从未采样」区。
  ⇒ 这条是**迭代速度**投资，不是**行为改变**投资，两者不要混。

---

## 4. 明确**不借**的（每条都有我们自己的反证）

### 4.1 N1 · 「把 STOP 负偏置改成中性」—— 我们的剂量-反应实测把它落在**死区**

- 【事实·仪器口径】偏置探针的 β 是**叠加在现有 bias 之上**的增量：
  `scripts/probe_explore_randomization.py:393` `_stop_base = float(pol.slot_head.bias.detach()[5].item())`、
  `:409` `pol.slot_head.bias.data[5] = _stop_base + beta`。
  ⇒ 「把 −1.0 改成中性」≡ **β = +1**。
- 【实测读数】β = 0 / 4 / 5 / 6 / 7 ⇒ 「≥6 圣水帧占比」**0.000% / 0.178% / 12.70% / 51.27% / 82.98%**
  （`docs/exploration_bias_pass_2026-09-18.md:20`）⇒ **β=1 与 β=0 不可分辨**。
- 【实测代价】β=6/7 时**每局回报 −41.52 / −54.10**（基线 −4.46，`docs/exploration_bias_pass_2026-09-18.md:168`）
  ⇒ 要起效就得进「回报崩溃区」，必须配课程/门控（唯一没崩的形态是**门控** `gate@6:d` 6.765% / **+1.68**，
  `docs/exploration_pressure_gate_2026-09-18.md:25,117`，且那条**仍未解决**「省下的费没花出去」）。
- ⇒ **借 B1/B2 之前不要动这个旋钮**；上游 `noop_head` 的「无负偏置」对我们**等价于 β=1**，**剂量不足**。

### 4.2 N2 · 8 帧观测堆叠 —— 三重冲突

- **R11**「不扩模型参数」（`docs/agents/redlines.md:25`）；改 CNN 输入通道会破坏全部旧卷积权重
  （`rl/observation.py:79` 已写死这条警告）；**R6** 架构/观测变更必须 `--fresh`（`docs/agents/redlines.md:20`）。
- 且**方向已被证伪**：O2 P3′ `belief token 无增量`（`C−A=+0.0039<0.05`）+「**信息不是缺口**：局内回报线性可预测而 critic 拿到 0」
  （`docs/agents/ledger.md:57`）。

### 4.3 N3 · 576 联合落点头 —— **我们本来就是，零增量**

- 【事实】`rl/follower.py:238` `self.cell_head = nn.Linear(hidden, GRID_H * GRID_W)`；
  `rl/observation.py:83` `GRID_H, GRID_W = 32, 18`；`rl/follower.py:499-500` 单 `Categorical` 采样；
  `grep -rn "y_head\|x_head\|placement_head" src/clasher_new/` = **0 命中**（我们这边）。
- ⇒ 上游 `f616f19` 是**从** (32)+(18) 两个因子化头**改成** 576 单头；我们**从一开始就是 576 单头**。
  前文 §1.3 把它当作上游新东西列出是对的，但**对我们没有可借性**（顺带：前文 `:173` 的 `train_autoregressive.py` 是**上游**路径，
  我们仓内同名的 `src/clasher_new/train_autoregressive.py` 是 36 行的 SB3 起步脚本，**不是同一份文件** —— 引用时勿混）。

### 4.4 N4 · 往奖励里加圣水项

- 与 `docs/agents/ledger.md:48`（X19：Δρ(塔血) +0.000、Δρ(胜) +0.001 ⇒ 幅度归零、方向不稳）、
  `docs/online_measure_2026-09-18.md:166-167`、R11 一致 ⇒ **不做**（前文④ 与我们一致）。
- ★ 但它与 B2 的关系要写清：**O7 修机制是它的前置，不是替代**（`docs/agents/ledger.md:62` ④）。

---

## 5. 我们已经有、因此**不算借鉴**的（避免重复劳动）

| 项 | 上游有 | 我们 | 备注 |
|---|---|---|---|
| 非法动作检查 | `1e9716b` 加了「比对 cycle 前后」的检查 | **四层显式**（掩码 / 整包校验 9 条 reason / 惩罚 / 统计） | 且已修真实病理（`used` off-by-one，`docs/mask_used_slot_offbyone_fix_2026-09-18.md`） |
| 被迫 vs 主动不出牌 | `1a86041` 训练内诊断 | **有离线仪器 + A/B/C 记账** | 只差「训练内顺手记」 |
| 课程做在环境侧 | ✅ 9 个脚本对手 | ✅ `solo` 有 `defend`（0.2） | 差的是**强度**（§3.2） |
| 对手自适应 | ❌ 上游**没有**（9 个固定脚本，`agent_pool.py` 是死件） | ✅ `frozen/hist/defend/rand_anchor` + Elo/PFSP + `flow` 全配对 | **这一层我们领先，不要为了"像上游"砍掉** |
| 特权先验注入 | ❌ 无 | ✅ `plan_dim=58` + `belief_dim=563` | 但注意 O2 P3′：**belief 无增量** |
| 测量纪律 | ❌ 24 个提交里无评估纪律 | ✅ 预注册 / 复算 / 分辨率认识 / 口径红线 | 这是我们**能判「不可分辨」**的原因 |

**附**：R11 括号里写「不扩模型参数（**629,359**）」，而实测当前策略 **993,008** 个参数
（37 个张量全量求和；`runs/et_solo100k/solo_main.pt` 与 `runs/run100k/main_ckpt_100000.pt` 同值）——
两个数**不一致，成因未定**【R10】（可能是 `belief_dim` 71→563 之前的旧读数）。规则本身不受影响。

---

## 6. 限制（【R10】本条最重要）

1. §2 全部读数是**描述性**的：n=1 seed、每臂 4 局 / 812–1,195 帧、**1 个 ckpt**、对手是**自身冻结副本**（`solo` 自对弈臂）；
   采样 RNG 未播种 ⇒ 四次同 ckpt 跑出 762/924/945 帧三种轨迹，**落点占比给出的是区间（85.5%–90.0%）而不是单点**。
2. **未跑任何训练**：本文所有「借」都是**建议**，不是判决；按 **R3** 每条都要另行预注册（含失败分支）。
3. **未核实**：SB3 `target_kl` 的精确语义（本机未装 SB3，仓内无 requirements）⇒ 只写「上游设了 0.03」，不写它的行为。
4. **未测**：上游模型的实际行为读数（上游仓内**没有**训练日志 / ckpt / 评估录像）⇒ 「学会了攒费/解场」仍**无第三方可复算证据**。
5. `SelfDefenderPolicy` 与 `run` 五脚本的**强度**没有行为级读数（全仓查不到）——
   §3.2 的「弱」是**代码级**判定（缺四条机制），不是**实测胜率**判定。
6. 本文涉及的两个上游 commit（`7fc64f2`/`f616f19`）都是**逐字 `git show`** 核过的；
   `8744065`/`e049d55` 的四条机制**取自前文**（本文只核了 `defensive_strategy.py:139-141` 一行），**其余四行未逐字复核**。

---

## 7. 复现命令

```bash
# ① 熵项预算分解（本文 §2；产出 docs/probe_entropy_decomp.json）
cd src/clasher_new && PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe \
  ../../scripts/probe_entropy_decomp.py \
  --ckpt trained=runs/et_solo100k/solo_main.pt --fresh-init --games 4 --max-frames 360

# ② STOP 偏置穿越 128k 步的轨迹（§1.2）
cd src/clasher_new && PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe \
  ../../.tmp/stop_bias_scan.py          # 临时件（.tmp/ 已 gitignore）

# ③ 上游熵项改动的逐字 diff（§3.1）
git -C /mnt/e/clash-royale-simulator-main-by-jason show 7fc64f2 -- src/clasher_new/train_autoregressive.py
```

---

## 8. 文件清单（本次改动）

| 文件 | 性质 |
|---|---|
| `scripts/probe_entropy_decomp.py` | **新增仪器**（只读；熵项预算分解；自带与 `evaluate()` 的对账自检） |
| `docs/probe_entropy_decomp.json` | 新增留证（四次轨迹的原始读数 + 合法项分布 + 圣水分层） |
| `docs/upstream_borrow_2026-09-19.md` | 本文 |
| `AGENTS.md` | B 区加一行指针（≤800 B） |
| `.tmp/stop_bias_scan.py` | 临时件（**不提交**，`.tmp/` 已 gitignore）—— 命令见 §7② |
