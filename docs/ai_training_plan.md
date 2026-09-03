# AI 训练算法实现规划：先知规划器 + 信念推断 + 跟随者策略 + POMDP + 联赛机制 + 同刻多卡

> **实现状态**：本文档对应的代码已按计划落地于 `src/clasher_new/rl/`（`ActionBundle` 同刻多卡、贝叶斯信念推断、BeliefPlanner/ProphetPlanner、FollowerPolicy + PPO、联赛/PFSP/Elo、训练与评测脚本）。详见 `src/clasher_new/rl/README.md`。规划章节保留为设计与演进依据。

## 0. 范围与边界

本阶段**不再继续扩展游戏引擎机制，也不再继续补卡牌内容**。

目标是建立一套可复现的 RL 训练闭环：

1. 以当前模拟器作为环境侧黑盒；
2. 用 POMDP 表达“玩家只知道自己可见信息，不完全知道对手隐藏信息”的对局；
3. 用“先知规划器 / 跟随者策略”分层训练；
4. 用联赛机制维持对手多样性，避免自我博弈退化；
5. 后续如果游戏文件逆向产出更多规则数据，只影响观测与动作编码层，不影响整体训练框架。

当前已有基础：

- `src/clasher_new/environment.py`：已有简单 gym 环境，32x18x15 grid 观测、5-slot hand、elixir、`MultiDiscrete([5, 32, 18])` 动作。
- `src/clasher_new/train.py`：已有 CNN feature extractor + PPO。
- `src/clasher_new/agent_pool.py`：已有简单 Elo/对局评估雏形。

---

## 1. 设计目标

### 1.1 短期目标

建立一个最小可训练、可评估、可复现的 POMDP 强化学习闭环。

### 1.2 中期目标

引入“先知规划器 + 跟随者策略”：

- 先知规划器：使用特权信息或更完整的状态信息进行规划；
- 跟随者策略：只使用玩家真实可见观测执行动作；
- 跟随者动作升级为同刻多卡 `ActionBundle`，解决单卡动作表达力不足的问题；
- 显式维护隐藏状态的贝叶斯信念 `b_t`，并基于信念进行规划（Belief Planner）；
- 通过模仿、蒸馏、DAgger 或 RL 微调，把先知与信念规划的能力迁移到跟随者。

### 1.3 长期目标

建立稳定联赛：

- 主策略不断进化；
- 历史快照与专门针对型对手共同训练；
- Elo / PFSP / 联赛评分维护策略多样性；
- 最终得到一个不过拟合单一对手的通用 1v1 策略。

---

## 2. 核心原则

1. **引擎冻结**：不继续扩展 `battle.py` / 卡牌机制 / 卡牌数值。
2. **环境包装器优先**：AI 训练相关代码尽量放在独立 RL wrapper 中，而不是污染模拟器本体。
3. **POMDP 显式化**：不要把对手隐藏信息直接给跟随者策略。
4. **先知只做训练侧辅助**：先知规划器可以使用特权信息，但最终部署的是跟随者策略。
5. **信念显式化**：对隐藏状态做贝叶斯推断 `b_t`，并让规划/决策基于 `b_t`，而不是假装隐藏状态已知。
6. **联赛优先于单点调参**：策略能力必须经过多样对手验证，而不是只针对 random 或最近策略。
7. **可复现**：每个训练 run 都要记录 config、seed、对手池、观测编码版本、奖励函数版本。

---

## 3. POMDP 形式化

### 3.1 基本定义

一个 POMDP 可以写成：

```text
M = (S, A, O, T, Ω, R, γ)
```

其中：

| 符号 | 含义 |
|---|---|
| `S` | 完整对局状态，包括双方单位、塔、手牌、牌序、圣水、冷却、隐藏信息 |
| `A` | 玩家可执行动作集合 |
| `O` | 玩家实际可见观测 |
| `T` | 环境转移函数，由模拟器实现 |
| `Ω` | 观测函数，把完整状态映射成某个玩家视角的观测 |
| `R` | 奖励函数 |
| `γ` | 折扣因子 |

### 3.2 状态空间 `S`

完整状态包括：

- 双方塔生命；
- 双方场上单位；
- 双方圣水；
- 双方当前手牌；
- 双方牌库顺序 / 下一步牌；
- 单位 buff、冻结、减速、部署冷却；
- 投射物、法效果、生成物；
- 随机数状态。

这些信息**只允许给先知规划器**，不给跟随者策略。

### 3.3 观测空间 `O`

玩家真实可见观测包括：

- 自己手牌、下一张牌、圣水；
- 自己场上单位；
- 对方场上可见单位；
- 双方塔状态；
- 对方刚刚打出的牌；
- 时间；
- 塔区域、可部署区域。

跟随者策略**不应该直接看到**：

- 对手未来牌序；
- 对手完整手牌；
- 对手尚未打出的隐藏决策；
- 环境内部随机数状态。

### 3.4 动作空间 `A`

正式动作形式使用 `ActionBundle`，见 `3.4.1`。单卡动作 `(slot, x, y)` 只作为早期兼容 baseline 或 `ActionBundle` 长度为 1 的特例。

单卡动作仍需动作掩码：

- `slot = 0`：等待 / no-op；
- `slot > 0`：打出第 `slot - 1` 张手牌；
- `x, y`：部署坐标；
- 圣水不足时禁用对应 slot；
- 非法坐标、非法区域禁用；
- 部署冷却中禁用对应牌。

它也可以展开成更稳定的 autoregressive 动作：

```text
1. 是否出牌
2. 选择哪张牌
3. 选择部署区域
4. 选择区域内的精确位置
```

这比直接 `MultiDiscrete([5, 32, 18])` 更容易学习，也更适合加入动作掩码。

#### 3.4.1 同刻多卡动作包 `ActionBundle`

现有 `MultiDiscrete([5, 32, 18])` 的局限是：一个决策步只能打出一张牌，无法表达“同一时刻连续下多张牌”。新框架必须把动作从单卡动作升级为**同刻多卡动作包**。

一个决策步动作定义为：

```text
A_t = ActionBundle(
    sub_actions = [a_1, a_2, ..., a_n],
    stop        = True / False
)
```

其中：

```text
a_i = (slot_i, x_i, y_i)
0 <= n <= K_max
```

关键语义：

1. **同刻执行**：同一个 `ActionBundle` 内的所有合法 `a_i` 都在当前决策步内提交给引擎，之后再统一推进模拟帧。  
2. **不重置观测**：策略在生成 `a_1, a_2, ...` 时，不重新观测战场；只根据原始观测 `o_t`、plan token 和已经生成的历史子动作来决定后续动作。  
3. **动态掩码**：每生成一个子动作后，重新计算动作掩码：
   - 扣除该卡圣水；
   - 移除已使用手牌槽；
   - 禁止重复选择同一 slot；
   - 重新检查部署区域合法性。
4. **停止条件**：策略可以输出 `STOP` 结束本步，也可以因为圣水不足、无合法动作、达到 `K_max` 而强制停止。  
5. **失败处理**：默认采用“整包校验、整包提交”的原子语义。若 bundle 中任一子动作非法，则拒绝整个 bundle 并施加惩罚，避免出现半执行状态；调试期可以另设宽松模式，仅记录非法子动作。

建议 `K_max` 初版设为 `4`，即最多一次打出四张手牌。当前模拟器中 `cycle[:4]` 是可出手牌，`cycle[4]` 是下一张牌；新 wrapper 应把二者显式区分，而不是继续沿用 5-slot hand 的旧观测。

`ActionBundle` 的校验建议采用两段式：

1. **先校验**：根据当前状态和 bundle 内已有子动作，推导剩余圣水、剩余手牌、合法部署区；
2. **再提交**：只有整包合法时，才按 bundle 内顺序依次调用 `deploy_card`；
3. **不推进时间**：bundle 内多次 `deploy_card` 之间不调用 `battle.step`，因此所有牌都发生在同一决策 tick。

v1 建议只允许 bundle 使用“决策开始时已在手牌中的牌”；bundle 内打出一张牌后新进入手牌的下一张牌，留到下一个决策步再决定，避免无新观测下的隐藏信息推断。`Mirror` 等依赖 `last_card` 的牌需要在 bundle 内有明确顺序规则，或 v1 先禁止进入多卡 bundle。

`ActionBundle` 同时覆盖**英雄技能触发**：`SubAction` 增加 `kind="ability"` 类型（引擎侧调用 `battle.use_ability(player_id)`，自动选取场上就绪英雄并扣费/进入冷却），可与出牌子动作在同一 bundle 内同 tick 组合；无就绪英雄时技能子动作按原子校验拒绝整包。跟随者 bundle head 的决策空间为「出牌槽位 + ABILITY + STOP」。

`ActionBundle` 不要求改模拟器本体：wrapper 可以在同一决策步内连续调用多次 `deploy_card`，期间不推进模拟时间，然后再统一调用一次 `battle.step`。因此它对策略来说是“同刻多卡”，对模拟器来说是“同 tick 批量提交”。

需要注意：由于 `deploy_card` 会立刻更新圣水、手牌和部分即时效果，bundle 内的顺序可能影响结果。v1 可以保留策略生成的顺序；如果后续发现顺序导致学习困难，再引入显式 order head 或受限排序规则。

这个动作包应该成为正式 follower 的默认动作形式；单卡动作只是它的 `n = 1` 特例，可继续作为早期 baseline。

### 3.5 奖励函数 `R`

初版奖励建议保留当前结构：

```text
reward =
  w1 * (敌方塔伤害 - 我方塔伤害)
+ w2 * (敌方掉塔数 - 我方掉塔数)
+ w3 * win/loss
+ w4 * 非法动作惩罚
```

之后引入更细的辅助奖励：

- 圣水浪费惩罚；
- 防守成功奖励；
- 反打节奏奖励；
- 塔剩余血量差；
- 单位交换价值。

但要注意：辅助奖励只作为 shaping，不掩盖胜负主目标。

### 3.6 隐藏状态与信念状态 `b_t`

POMDP 中的隐藏状态不只是“看不见的完整局面”，重点是那些会影响决策、但跟随者不能直接读取的量：

```text
z_t = {
  opponent_hand,         # 对手当前手牌
  opponent_cycle,        # 对手牌序 / 下一张可能出现的牌
  opponent_elixir,       # 对手圣水，如果观测不完整则推断
  opponent_macro_intent, # 防守 / 进攻 / 换路 / 反打 / 拖节奏
  opponent_tendency,     # 激进、防守、法术习惯、卡组倾向
  unresolved_effects,    # 尚未结算的投射物 / 法术 / 部署延迟
}
```

信念状态定义为：

```text
b_t = P(z_t | o_1:t, a_1:t-1)
```

v1 不必一开始就做完整贝叶斯滤波。可以分成三层：

1. **显式规则信念**
   - 根据对手已经打出的牌，维护“剩余可能手牌 / 已消耗牌 / 下一张可能牌”；
   - 对 8 卡循环这类小状态空间，可以用枚举或带权样本表示。

2. **统计信念**
   - 维护对手倾向的离散分布，例如：
     - `aggressive / defensive / cycle / beatdown / spell_heavy`；
     - 左路偏重、右路偏重、双路均衡；
     - 喜欢反打、喜欢防守反击、喜欢法术清场。

3. **神经信念编码**
   - 用 GRU / Transformer / RSSM 式模型从历史观测中压缩出 `belief_token`；
   - 这是最通用方案，但要额外加校准和辅助监督，避免变成不可解释黑盒。

理想形态是：

```text
b_t = {
    hand_probs,        # 每张未出牌在对手手牌中的概率
    cycle_probs,       # 牌序 / 下一张牌分布
    elixir_estimate,   # 对手圣水估计
    intent_probs,      # 对手宏观意图分布
    tendency_probs,    # 对手风格分布
    uncertainty,       # 当前信念的不确定度
}
```

这个 `b_t` 会成为后续规划的核心输入之一。

### 3.7 基于信念状态的规划

这里要把两类规划器分开：

| 名称 | 可用信息 | 用途 |
|---|---|---|
| Prophet Planner | 可使用特权完整状态 | 训练期教师 / 监督信号 |
| Belief Planner | 只能使用可见历史和 `b_t` | 可部署的真实规划器 |

贝叶斯更新的抽象形式是：

```text
b_t ∝ p(o_t | z_t) * Σ_{z_t-1} p(z_t | z_t-1, a_t-1) * b_t-1
```

规划目标可以写成：

```text
a_t* = argmax_a E_{z_t ~ b_t}[ R_t + γ E[V(s_t+1)] ]
```

工程上可以按难度分四层实现：

1. **规则化信念规划**
   - 利用对手已出牌推断剩余牌和下一张牌；
   - 如果对手大概率没有某张关键法术，就更大胆堆单位；
   - 如果对手大概率有关键法术，就避免高价值单位密集站位。

2. **后验采样规划**
   - 从 `b_t` 采样若干个可能的对手手牌 / 牌序 / 意图；
   - 对每个样本做短程 lookahead；
   - 取期望收益最高或风险调整后收益最高的动作。

3. **信念树搜索**
   - 类似 POMCP / belief tree search；
   - 节点状态是 `b_t`，而不是单一确定状态；
   - 成本更高，但更贴近 POMDP。

4. **神经信念规划器**
   - 用 Transformer / GRU 编码历史，输出 `belief_token`；
   - 再由小型 planner head 输出宏观 intent、风险偏好、bundle size hint；
   - 这是 RL 训练里最实用的中间路线。

对 follower 的意义是：策略不再只看“当前可见棋盘”，而是显式考虑“对手可能有什么、可能想做什么、这种判断有多不确定”。

---

## 4. 总体架构

```text
                 ┌────────────────────┐
                 │   Game Simulator   │
                 └─────────┬──────────┘
                           │ full state
                           ▼
                 ┌────────────────────┐
                 │  RL Env Wrapper    │
                 │  obs / action mask │
                 └─────────┬──────────┘
                           │
              partial obs  │  privileged state
              ▼            ▼
       ┌────────────┐   ┌────────────────┐
       │  Belief    │──►│ Prophet Planner │
       │  Inference │   │  advice / plan  │
       └─────┬──────┘   └────────────────┘
             │ belief token       │ plan token
             ▼                     ▼
       ┌────────────┐   ┌────────────────┐
       │ Belief     │──►│   Follower     │
       │ Planner    │   │   Policy       │
       └────────────┘   └─────┬──────────┘
             (plan token)      │
                               ▼
                       ┌────────────┐
                       │  League    │
                       │  opponents │
                       └────────────┘
```

核心思路：

- 模拟器只负责状态转移；
- RL wrapper 负责输出玩家视角观测和动作掩码；
- Belief Inference 从可见历史推断对手隐藏状态，产出 `b_t` 与 `belief_token`；
- 先知规划器使用完整状态，输出**特权监督** plan；
- 信念规划器只使用可见历史 + `b_t`，输出**可部署** plan；
- 跟随者只使用可见观测、belief token、plan token（先知或信念规划器提供），输出同刻多卡 `ActionBundle`；
- 联赛机制负责提供多样对手。

---

## 5. 先知规划器设计

### 5.1 角色定义

先知规划器不是最终 agent。

它是训练阶段的“教师 / 规划器”，可以使用：

- 完整实体状态；
- 双方隐藏手牌；
- 对手牌库顺序；
- 更多前瞻信息；
- 甚至可以在内部调用模拟器做短程 lookahead。

与它互补的是 **信念规划器 Belief Planner**（见 5.5）：先知用特权完整状态出“标准答案”，信念规划器只用可见历史 + `b_t` 出“可部署答案”。两者的 plan token 可以互相监督、互相消融，最终部署的是信念规划器或由信念规划器提供 plan token 的 follower。

### 5.2 输出形式

建议先知规划器输出三层信息：

```text
ProphetOutput = {
    macro_intent,       # 宏观意图：防守 / 进攻 / 换路 / 拖节奏 / 法术控场
    focus_region,       # 粗粒度重点区域，例如左路、右路、王塔前、桥头
    suggested_card,     # 可选：建议使用的卡
    suggested_action,   # 可选：具体动作
    bundle_size_hint,   # 可选：本次建议单卡还是多卡联动
    combo_hint,         # 可选：坦克+后排 / 法术+单位 / 双路压制等组合意图
    value_estimate,     # 可选：当前局面价值
}
```

初版可以简化成：

```text
macro_intent ∈ {
  defend_left,
  defend_right,
  defend_king,
  push_left,
  push_right,
  counterpush,
  spell_value,
  cycle_and_wait
}
```

`focus_region` 可以先离散成粗区域：

```text
focus_region ∈ {
  own_left,
  own_center,
  own_right,
  bridge_left,
  bridge_right,
  enemy_left,
  enemy_center,
  enemy_right
}
```

不建议一开始就让先知输出非常精确的 `(x, y)`，否则跟随者容易退化成纯模仿器。

### 5.3 先知的两种实现路线

#### 路线 A：特权观测 RL 先知

让一个 PPO / Recurrent PPO agent 直接观测完整状态训练。

优点：

- 实现简单；
- 可以直接复用现有 PPO 基础；
- 不需要手写规划规则。

缺点：

- 训练成本高；
- 可能仍然依赖特权信息，迁移到跟随者时会损失。

#### 路线 B：启发式 + 搜索先知

用规则或短程搜索生成计划。

例如：

1. 判断当前最大威胁：对方哪一侧推进最快；
2. 判断防守收益；
3. 判断反打收益；
4. 判断法术价值；
5. 输出宏观 intent。

优点：

- 可解释；
- 冷启动快；
- 可以给 follower 提供稳定监督信号。

缺点：

- 上限受限于规则设计；
- 不能自动发现人类未知策略。

建议：先做路线 B 作为冷启动，再训练路线 A 作为高级先知。

### 5.4 先知与跟随者的关系

优先采用以下方式：

```text
1. 先知给出 plan token / advice vector。
2. 跟随者观测 partial obs + plan token。
3. 跟随者输出最终动作。
4. 训练时同时使用：
   - 行为克隆：模仿先知或启发式策略；
   - RL 微调：在真实 reward 下继续优化。
```

可选升级：

- 使用 DAgger，让跟随者在自己的状态分布上查询先知建议，减少 covariate shift；
- 使用 plan dropout，随机遮蔽先知建议，避免跟随者完全依赖先知；
- 使用 adversarial league，让跟随者在多样对手下学会在错误建议下纠错。

### 5.5 信念规划器 Belief Planner

信念规划器与先知规划器的本质区别只有一条：**先知看到 `s_t`，信念规划器只看到 `o_1:t` 与 `b_t`**。

信念规划器是真正要部署的那一个，所以它的质量直接决定最终 agent 的真实水平。

输入：

```text
o_1:t        # 完整可见历史（观测 + 动作 + 结果）
b_t          # 信念状态，见 3.6
plan_ctx     # 可选：卡组、地图、局面摘要
```

输出（与先知同构，便于蒸馏和监督）：

```text
BeliefPlan = {
    macro_intent,       # 在信念下最可能的意图
    focus_region,
    suggested_card,
    bundle_size_hint,
    combo_hint,
    risk_profile,       # 信念不确定度高时更保守 / 更低置信出牌
}
```

v1 实现路线（按成本递增）：

1. **规则信念规划**：只靠手牌/循环/圣水推断，加一组启发式 if-then；
2. **后验采样规划**：从 `b_t` 采样 `K` 个隐藏状态，各自短程 lookahead 后按期望选动作；
3. **神经信念规划**：`BeliefEncoder → plan head`，直接学 `b_t → plan token`；
4. **信念树搜索**：POMCP 式扩展，作为远期目标。

信念规划器与先知的关系：

- 先知产生**特权监督**：`plan_prophet(s_t)`；
- 信念规划器产生**可部署监督**：`plan_belief(o_1:t, b_t)`；
- follower 同时学习两者，但只在**信念规划器无 plan**（plan dropout）时也不能崩溃；
- 消融实验对比“先知 plan / 信念 plan / 无 plan”三档，量化信念推断带来的收益。

---

## 6. 跟随者策略设计

### 6.1 输入

跟随者策略只接收：

- 当前玩家可见观测；
- 自己动作历史；
- 先知 plan token；
- **信念状态 `b_t` 的编码 `belief_token`**（手牌概率、牌序概率、圣水估计、意图分布、不确定度）；
- 当前 `ActionBundle` 已生成的部分子动作；
- 动作合法性掩码。

> 原则：跟随者永远不直接看到隐藏状态，只能看到“对隐藏状态的推断结果”。因此 `b_t` 本身必须由独立模块提供，且不允许把 `z_t` 混进观测。

观测建议包含：

- 网格实体编码；
- 单位类型；
- 阵营；
- 生命值；
- 是否空军；
- 是否攻击空军 / 地面；
- 射程；
- 移动速度；
- 部署状态；
- 塔生命；
- 圣水；
- 手牌；
- 下一张牌；
- 时间；
- **belief token**（作为独立特征组接入）。

### 6.2 网络结构

初版建议：

```text
partial_obs
   │
   ├── grid CNN / ResNet block
   ├── hand embedding
   ├── scalar features
   │
   ▼
 fused feature
   │
   ├── recurrent memory / GRU / Transformer
   │
   ▼
 policy head + value head

belief_token
   │
   └── belief encoder ──► 与 fused feature 拼接后进 policy/value head
```

推荐实现顺序：

1. CNN + MLP + PPO；
2. CNN + GRU + PPO；
3. CNN + Transformer / memory + PPO；
4. 加 plan-conditioned head；
5. 加 belief-conditioned head：`belief_token` 与 fused feature 拼接后进 policy/value head。

### 6.3 动作头

#### 6.3.1 单卡动作头：过渡 baseline

初版可以先保留单卡动作头用于对照：

```text
policy = P(slot, x, y | obs, plan, mask)
```

或结构化为：

```text
p1 = P(play_or_wait | obs, plan)
p2 = P(card_slot | obs, plan, play)
p3 = P(region | obs, plan, card_slot)
p4 = P(x, y | obs, plan, region)
```

#### 6.3.2 同刻多卡动作头：正式版

正式 follower 不应再输出单卡动作，而应输出 `ActionBundle`。

推荐使用 **autoregressive bundle head**：

```text
h_t = Encoder(o_t, plan_t, belief_token_t, history_t)
```

并且 bundle head 的每一步都条件于信念：

```text
a_1 ~ π(a_1 | h_t, belief_token_t, mask_1)
a_2 ~ π(a_2 | h_t, belief_token_t, a_1, mask_2)
...
a_n ~ π(a_n | h_t, belief_token_t, a_{<n}, mask_n)
STOP ~ π(STOP | h_t, belief_token_t, a_{<=n})
```

联合动作概率为：

```text
log π(A_t | o_t, plan_t, belief_token_t)
= Σ_i log π(a_i | o_t, plan_t, belief_token_t, a_<i)
+ log π(STOP | o_t, plan_t, belief_token_t, a_<=n)
```

实现要点：

1. **不要枚举所有卡牌子集**，否则动作空间会组合爆炸。
2. 每个 sub-action 生成后立即更新动态动作掩码。
3. 可以按 `slot` 升序或其他固定规则规范化 bundle，减少同一语义动作的重复表示。
4. bundle 内的所有合法子动作都在同一物理时刻提交给模拟器，然后再统一推进模拟帧。
5. 若某个 sub-action 非法，默认拒绝整个 bundle 并施加惩罚，避免产生“半执行”状态；调试期可选择宽松模式记录非法项。
6. PPO 需要自定义 composite action distribution，不直接等同于普通 `MultiDiscrete`。

### 6.4 POMDP 记忆

跟随者必须维护历史信息。

最小版本：

```text
h_t = GRU(f(o_t, a_{t-1}), h_{t-1})
```

更完整版本：

```text
h_t = TransformerEncoder(o_{t-k:t}, a_{t-k:t-1})
```

记忆要能学到：

- 对手刚才打了什么牌；
- 哪条路已经被压制；
- 对手可能缺哪张牌；
- 自己刚才的防守是否成功。

记忆与信念的关系：

- **GRU/Transformer 记忆**是“隐式”的信念：网络内部隐含了对历史的理解，但不可解释、难校准；
- **显式信念 `b_t`** 是“外置”的信念：由 `BeliefInference` 独立维护，可作为额外输入，也可作为辅助监督目标；
- 推荐两者并用：外置信念给规划器/可解释分析用，隐式记忆给策略的细粒度反应用。v1 可以先只做隐式记忆 + 规则信念，再逐步升级为神经信念。

---

## 7. 联赛机制

### 7.1 目的

避免以下问题：

- 自我博弈过拟合；
- 策略循环克制；
- 只会打 random；
- 单一风格垄断；
- 灾难遗忘。

### 7.2 联赛成员类型

建议维护四类成员：

| 类型 | 说明 |
|---|---|
| Main Agent | 正在持续训练的主策略 |
| Historical Agent | 过去的 checkpoint，冻结参数 |
| Exploiter Agent | 专门针对某个 Main Agent 的克制策略 |
| Baseline Agent | random / heuristic / scripted 策略 |

### 7.3 对手采样

使用 PFSP，Prioritized Fictitious Self-Play：

```text
P(opponent) ∝ (1 - winrate(main, opponent))^β
```

含义：

- 如果 Main Agent 很容易打赢某个对手，就降低采样概率；
- 如果某个对手很难打，就提高采样概率。

可以再叠加多样性因子，避免一直只针对某个对手。

### 7.4 Elo / 评分机制

每次联赛周期后：

1. 所有活跃 agent 两两对战；
2. 更新 Elo 或 TrueSkill；
3. 记录胜率、平均奖励、塔伤害差；
4. 保存表现显著变化的 checkpoint。

注意：Elo 只用于联赛调度和分析，不直接作为 RL reward。

### 7.5 Exploiter 机制

每隔 N 个训练周期：

1. 选择当前 Main Agent；
2. 训练一个 Exploiter；
3. Exploiter 的目标是从 Main Agent 身上获得高胜率；
4. 如果 Exploiter 胜率超过阈值，把 Main Agent 加入下一轮以 Exploiter 为重要对手的训练；
5. 防止 Main Agent 出现明显漏洞。

Exploiter 不需要成为最终策略，只需要暴露 Main Agent 的弱点。

---

## 8. 训练流程

### 8.1 阶段一：环境与数据基线

目标：确认 RL wrapper 稳定。

工作项：

- 实现独立 `RLEnv` wrapper；
- 支持部分观测；
- 支持 `ActionBundle` 同刻多卡动作；
- 支持动作掩码；
- 支持对称坐标转换；
- 支持结果记录；
- 支持固定 seed；
- 支持批量并行环境；
- replay 中同步记录特权隐藏状态（对手手牌/牌序/意图标签），供信念模块监督训练。

验收：

- 能稳定跑完对局；
- 无 illegal state；
- `ActionBundle` 能正确校验、执行或拒绝；
- reward 曲线可复现；
- 能保存 episode replay（含隐藏状态标签）。

### 8.2 阶段二：基础 POMDP baseline

目标：先有一个可比较的 baseline。

工作项：

- 实现 CNN + MLP PPO；
- 实现 CNN + GRU PPO；
- 优先支持 `ActionBundle`，早期可先用 `n <= 1` 作为兼容模式；
- 加入 action mask；
- 加入 opponent pool；
- 记录 TensorBoard。

验收指标：

- vs Random 胜率显著超过 90%；
- vs Heuristic Bot 有明显胜率提升；
- `ActionBundle` 训练与评估链路可跑通；
- 相同 seed 下训练曲线稳定。

### 8.3 阶段三：信念推断与贝叶斯规划 v1

目标：建立对隐藏状态的显式信念，并验证“基于信念的规划”收益。

工作项：

- 实现 `BeliefInference` 三层：
  1. 规则信念（手牌 / 循环 / 圣水枚举或加权样本）；
  2. 统计信念（对手倾向离散分布）；
  3. 神经信念编码器（GRU / Transformer / RSSM）。
- 实现信念更新接口：`update(o_t, a_t) -> b_{t+1}`；
- 用**特权状态作为监督标签**训练信念模块：
  - 对手真实手牌 / 牌序 / 意图作为 supervised target；
  - 交叉熵 / KL 损失；
  - 校准损失（reliability / Brier score）。
- 实现 `BeliefPlanner`（先规则版，再后验采样版）；
- 在 follower 中接入 `belief_token`，跑“有 / 无信念”对照。

验收：

- 信念校准：Brier score / log-loss 在验证集上显著优于 uniform 先验；
- 下一张牌预测准确率明显提升；
- 加入 `belief_token` 后 follower 胜率不低于 baseline；
- 基于信念的规划 vs 无信念规划有明显收益或至少不劣化。

### 8.4 阶段四：先知规划器 v1

目标：产生稳定的宏观建议。

工作项：

- 定义 `macro_intent` 和 `focus_region`；
- 实现启发式先知；
- 实现 plan token 编码；
- 将 plan 注入 follower policy；
- 同步实现信念规划器（见 5.5），让先知与信念规划输出同构，便于蒸馏。

验收：

- 先知建议覆盖率接近 100%；
- plan token 能显著改变 follower 行为；
- follower 可以根据同一 partial obs 在不同 plan 下输出不同动作。

### 8.5 阶段五：跟随者训练 v1

目标：跟随者学会在可见观测 + 信念 + 计划下执行动作。

工作项：

- Behavior cloning（同时克隆先知 plan 与信念 plan）；
- PPO fine-tune；
- plan dropout；
- belief dropout（随机遮蔽 `belief_token`，防止过度依赖外置信念）；
- DAgger 可选；
- 辅助监督：在策略训练的同时，用自监督目标强化信念模块（预测下一张牌 / 对手手牌 / 对手意图）。

验收：

- follower vs random 胜率不低于 baseline；
- follower 在有 plan 条件下比无 plan 条件表现更好；
- 无 plan、无 belief 时不应崩溃，说明没有完全依赖特权信息；
- 信念模块在训练中不退化（持续评估校准指标）。

### 8.6 阶段六：联赛 v1

目标：建立稳定 self-play league。

工作项：

- 实现对手池；
- 实现 PFSP；
- 实现 Elo；
- 实现 checkpoint 管理；
- 实现 Exploiter 训练入口。

验收：

- 能持续训练不崩；
- 不会只过拟合最近对手；
- 不同 checkpoint 之间有 Elo 排序；
- 最终 agent 对 random / heuristic / historical agents 均保持优势。

### 8.7 阶段七：高级先知与消融实验

目标：验证算法贡献。

实验组：

| 实验 | 目的 |
|---|---|
| PPO baseline | 基准 |
| Recurrent PPO | 验证 POMDP 记忆价值 |
| Prophet + Follower | 验证规划迁移价值 |
| Belief + Follower | 验证贝叶斯信念推断价值 |
| Prophet + Belief + Follower | 验证两者叠加 |
| League only | 验证联赛对鲁棒性的贡献 |
| Prophet + League | 验证组合收益 |
| No plan dropout | 检查 follower 是否过度依赖先知 |
| No belief dropout | 检查 follower 是否过度依赖外置信念 |
| Belief calibration off | 检查信念校准对胜率的贡献 |

---

## 9. 代码结构建议

不建议继续改模拟器本体。

建议新增：

```text
src/clasher_new/rl/
  __init__.py
  env_wrapper.py
  observation.py
  action_bundle.py
  action_mask.py
  plan_space.py
  belief.py            # BeliefState / BeliefInference 数据结构与更新接口
  bayes_filter.py      # 规则/枚举/加权样本式贝叶斯滤波（手牌、循环、圣水）
  belief_encoder.py    # 神经信念编码器（GRU / Transformer / RSSM）
  belief_planner.py    # BeliefPlanner：基于 b_t 的规划器
  belief_train.py      # 信念模块的监督训练 / 校准
  prophet.py
  follower.py
  league.py
  pfsp.py
  elo.py
  replay.py
  train_baseline.py
  train_belief.py      # 信念模块训练入口
  train_follower.py
  train_exploiter.py
  evaluate.py
```

脚本层：

```text
scripts/
  train_rl_baseline.py
  train_belief.py
  train_prophet.py
  train_follower.py
  run_league.py
  evaluate_league.py
  export_replay.py
```

数据目录：

```text
runs/
  baseline/
  belief_v1/
  prophet_v1/
  follower_v1/
  league_v1/
  exploiters/
```

---

## 10. 关键接口草案

### 10.1 环境接口

```python
class RLEnv:
    def reset(self, seed=None, opponent=None) -> ObsDict
    def step(self, action_bundle: ActionBundle) -> tuple[ObsDict, float, bool, bool, InfoDict]
    def get_action_mask(self, partial_bundle=None) -> ActionMask
    def get_prophet_state(self) -> ProphetState
    def get_hidden_state(self) -> HiddenState   # 仅训练期：信念监督用
```

### 10.2 先知接口

```python
class ProphetPlanner:
    def plan(self, full_state, visible_obs) -> PlanToken
```

### 10.3 信念接口

```python
@dataclass
class BeliefState:
    hand_probs: np.ndarray        # 每张未出牌在对手手牌中的概率
    cycle_probs: np.ndarray       # 下一张牌 / 牌序分布
    elixir_estimate: np.ndarray   # 对手圣水估计（分布或均值±std）
    intent_probs: np.ndarray      # 对手宏观意图分布
    tendency_probs: np.ndarray    # 对手风格分布
    uncertainty: float            # 信念不确定度

class BeliefInference:
    def reset(self) -> None
    def update(self, o_t, a_t, b_t) -> BeliefState   # b_t -> b_{t+1}
    def encode(self, b_t) -> np.ndarray              # -> belief_token
    def train_step(self, hidden_state, o_hist, a_hist) -> dict  # 监督/校准 loss

class BeliefPlanner:
    def plan(self, o_1t, b_t) -> BeliefPlan   # 输出与 Prophet 同构的 plan token
```

### 10.4 跟随者接口

```python
class FollowerPolicy:
    def act(
        self,
        obs,
        plan_token,
        belief_token,           # 信念模块编码结果
        partial_bundle=None,
        deterministic=False,
    ) -> ActionBundle
```

### 10.5 联赛接口

```python
class League:
    def sample_opponent(self, agent_id) -> AgentHandle
    def register_checkpoint(self, path, metadata) -> None
    def update_ratings(self, results) -> None
    def select_training_opponents(self, agent_id, n) -> list[AgentHandle]
```

---

## 11. 评测方案

### 11.1 必测对手

- Random Policy；
- Heuristic Bot；
- 历史Checkpoint；
- Main Agent；
- Exploiter Agent；
- 不同 deck 组合。

### 11.2 指标

| 指标 | 含义 |
|---|---|
| Win Rate | 总体胜负 |
| Elo | 联赛相对强度 |
| Crown Difference | 王冠差 |
| Tower Damage Difference | 塔伤害差 |
| Elixir Efficiency | 圣水交换效率 |
| Action Legality Rate | 动作合法率 |
| Bundle Size Distribution | 每个决策步同刻出牌数量分布 |
| Bundle Legality Rate | 同刻多卡动作包整体合法率 |
| Bundle Success Rate | 同刻多卡动作包成功执行率 |
| Plan Following Rate | 是否按 plan 行动 |
| Robustness | 对不同 deck / 对手的稳定性 |
| Belief Brier / Log-loss | 信念校准质量（预测对手手牌/牌序/意图） |
| Next-Card Prediction Accuracy | 对手下一张牌预测准确率 |
| Hand Inference Accuracy | 对手手牌推断准确率 |
| Intent Prediction Accuracy | 对手宏观意图预测准确率 |
| Belief Uncertainty Calibration | 信念不确定度与实际正确率是否匹配（reliability curve） |
| Belief Ablation Delta | 有 / 无 `belief_token` 的胜率差，量化信念贡献 |

### 11.3 评估协议

每个版本至少评估：

```text
N = 200 场
opponents = fixed benchmark pool
seeds = fixed
decks = fixed benchmark decks
```

避免用训练中对手池做唯一评估来源。

信念模块单独评估：

```text
N = 500 场（覆盖多种 deck / 对手风格）
指标 = Brier / log-loss / next-card accuracy / hand inference accuracy / intent accuracy
baseline = uniform 先验 或“只看上一张牌”的启发式
```

---

## 12. 风险与规避

| 风险 | 规避方式 |
|---|---|
| Follower 完全依赖先知 | plan dropout、无 plan 评估、DAgger |
| 联赛过拟合历史对手 | PFSP + diversity sampling |
| 奖励塑形导致不对局 | 主奖励始终以胜负和塔差为主 |
| POMDP 记忆学不到 | 显式历史特征 + recurrent/Transformer |
| 动作空间稀疏 | action mask + autoregressive action |
| `ActionBundle` 组合爆炸 | autoregressive bundle head + `STOP` + 动态掩码 |
| Bundle 内顺序影响结果 | 固定 canonical order、显式 order head、或先做顺序敏感性实验 |
| Bundle 半执行导致状态污染 | 默认整包校验、整包提交 |
| 信念过自信（overconfidence） | 校准损失 + reliability curve 监控 + 温度缩放 |
| 信念推断错误导致错误规划 | 信念作为“软输入”而非硬决策，允许后验采样 + plan dropout |
| 特权信息泄漏进跟随者 | 信念只给推断结果 `b_t`，绝不直接给 `z_t`；代码层隔离 + 数据审计 |
| 信念模块训练数据不足 | 用特权状态自动生成大量监督标签（对手手牌/牌序/意图） |
| 信念与策略耦合导致灾难遗忘 | 信念模块先冻结训练，再联合微调；定期回测校准指标 |
| 模拟器速度限制 | headless 并行环境 + 减少 render |
| 训练不稳定 | 固定 seed、checkpoint、KL 限制、学习率调度 |

---

## 13. 实施里程碑

### M1：RL 环境契约

- `RLEnv` wrapper；
- partial obs；
- `ActionBundle` 同刻多卡动作契约；
- action mask；
- replay 记录；
- 并行环境。

### M2：Baseline POMDP Agent

- Recurrent PPO；
- 先支持 `n <= 1` 的 `ActionBundle` 兼容模式，再开放多卡；
- opponent pool；
- benchmark 评估。

### M3：Belief Inference & Planner v1

- `BeliefInference`（规则 + 统计 + 神经三层）；
- 特权状态监督训练 + 校准评估；
- `BeliefPlanner` 规则版 / 后验采样版；
- follower 接入 `belief_token` 并与无信念 baseline 对比。

### M4：Prophet v1

- macro intent；
- focus region；
- plan token；
- 启发式先知；
- 信念规划器与先知输出同构，便于蒸馏。

### M5：Follower v1

- plan-conditioned + belief-conditioned policy；
- 完整 `ActionBundle` 动作头；
- imitation + RL；
- plan dropout + belief dropout。

### M6：League v1

- opponent pool；
- PFSP；
- Elo；
- Exploiter。

### M7：实验报告

- ablation（含 Belief / No-belief）；
- 不同 agent 胜率；
- Elo 表；
- 信念校准曲线；
- 行为分析。

---

## 14. 与游戏文件逆向的关系

游戏文件逆向不影响本规划的整体算法结构，只影响以下部分：

1. 观测编码：更多真实字段可以进入 `observation.py`；
2. 动作约束：更准确的合法部署区域和卡牌限制；
3. 数值奖励：更准确的单位价值、塔价值、圣水价值；
4. 隐藏信息建模：更真实的 deck / hand / cycle 规则。

因此本阶段先搭训练算法闭环，逆向数据成熟后直接替换观测与规则映射层。

---

## 15. 当前建议的下一步

1. 定义并实现 `ActionBundle`：数据结构、整包校验、动态动作掩码；
2. 新建 `src/clasher_new/rl/env_wrapper.py`，让 `step()` 接收同刻多卡动作包；
3. 把当前 `CREnv` 重构为更适合 POMDP 的训练 wrapper；
4. 显式加入 action mask；
5. 加入 replay 记录（并记录特权隐藏状态，作为信念模块监督标签）；
6. 训练第一个 Recurrent PPO baseline，先从 `n <= 1` 过渡到 `n <= 4`；
7. 实现 `BeliefInference` v1：规则信念（手牌/循环/圣水）→ 统计信念 → 神经信念编码器，并用特权状态监督 + 校准；
8. 实现 `BeliefPlanner` v1：规则版 / 后验采样版，跑“有 / 无信念”对照；
9. 再实现 Prophet Planner 与 League。
