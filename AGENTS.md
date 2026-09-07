# AGENTS — 项目决策与方案存档

本文件记录跨会话需要记住的方案评估与决策，供后续会话/协作者直接引用，避免重复推导。
（文档↔源码总索引见 `docs/README.md`；训练代码导读见 `src/clasher_new/rl/README.md`。）

---

## 游戏 AI 工具使用：先例与接入方案（2026-09-06 探索 + 评估定稿）

### 背景结论

游戏 AI 中"外置工具/技能"有两条谱系：

- **经典 RL 谱系**：宏动作 / options framework。OpenAI Five 的出装模板、AlphaStar 的开局先验、TStarBot 的宏动作——把领域知识做成引擎侧确定性模块，网络只管高层选择。
- **LLM agent 谱系**：Voyager（技能库，Minecraft）、TextStarCraft II（LLM 战略顾问 + RL 执行层，与本仓库架构最接近）、ACM/SwarmBrain（2025 后续）。
- **不照搬 Voyager 实时写代码模式**：CR 决策粒度亚秒级，LLM 单次调用数百 ms~秒级，进不了 tick 循环；适合局间节奏。

### LLM 顾问的作用机制（原理定稿）

LLM 文本**永远不进网络或梯度**，只通过"改训练议程"起作用：

1. 改经验分布（最安全）：诊断 → 转 PFSP 采样权重（`rl/pfsp.py` 的 `weights()`）；
2. 改奖励权重（最危险）：改 `rl/config.DEFAULT_REWARD` 类字典，改错方向梯度会高效放大错误；
3. 改约束与课程：动作闸门开关、双倍期切换点、卡组池。

可信性靠**提案-验证分离**：LLM 文本只是提案 → 结构化配置 → 下一轮按新议程训练 → `evaluate`/elo 客观分验证 → 涨了保留、跌了回滚。错误代价被限制在"浪费一轮训练"。

### MCTS / 搜索（原理定稿）

- 本仓库有确定性模拟器 = 现成世界模型，不需要 MuZero 式模型学习。
- 正确形态：策略网络给先验分（top-k 剪枝）+ 价值网络评叶 + 浅深度（2-4 决策帧）搜索。CR 的"出牌-响应"结构使深度 2-3 已覆盖一次完整交锋。
- 本质困难是**对手问题**（部分可观测），三条路：最可能对手（便宜、脆弱）、信念采样 POMCP（正确、贵）、特权 oracle 蒸馏 ReBeal（有 `get_prophet_state()` 原料）。
- 训练时带搜索、推理时不带也能变强 = 专家迭代（Expert Iteration）：蒸馏完整访问分布而非最优动作（防熵坍缩）；搜索必须显著强于当前策略才有正收益；搜索对手模型的偏见会被一并蒸馏（AlphaStar 的教训）。

### 实测修正（评估时补的账，勿凭直觉推翻）

1. **引擎速度**（2026-09-06 实测，lv11 交火局面）：
   - 18,931 battle-steps/s，1 battle step 0.05ms，1 个 env 决策步（30 battle steps）约 2ms（不含网络推理）。
   - 浅 MCTS（深 3 × top-8 × 每叶推演 5 决策步 ≈ 3600 battle steps）≈ **0.19s/次搜索**；一局 360 决策步挂搜索只多花约 70s → **推理时搜索非常便宜**。
   - 训练时专家迭代 ≈ **120× 每帧经验成本**，当前训练量级（万步级）烧不起。
2. **路线 1（宏动作）已有大量基础**：`rl/plan_space.py` 已是 57 维 plan token、21 个宏观意图（含 pull/anti_spell/setup_wait/spell_finish/punish）+ placement_hint(7) + hold_mask(4)。剩余工作是"给意图层补执行绑定"，而非从零建宏动作。注意：plan 现在是**特征输入**（Prophet/BeliefPlanner 产 token 喂策略），"每 k tick 选技能"意味着改成**动作层**，是动作语义重造 + 旧 checkpoint（plan_dim=21）兼容问题，工程量大。
3. **路线 3（观测增强查询工具）已经存在**：`rl/belief.py` / `bayes_filter.py` / `prophet.py` 就是该形态，无新增工作量。
4. **评估基线注意**：搜索/蒸馏收益需要强对手才能测出；script policy 太弱。现成通道：main vs frozen_copy（`eval_solo_parallel`）与 exploiter 检查点。

### 优先级（已确认的排序）

1. **纯 RL 训练量拉满 + 已诊断行为病理的修复**——基线快速爬坡期，任何外挂收益都会被淹没，且这是最便宜的确定性收益。法术砸塔病理首选 SpellModule 特征路线（见"卡牌知识模块"节），前段 EV 闸门（硬约束）作学不动时的兜底，二选一起步。
   - **2026-09-07 三项全部落地**（100k solo 取证证实软惩罚无效：37 次砸塔 100% 选择型、68% 纯空砸塔、前段砸塔率 16%→54%，取证报告见会话）：
     - **9h 前段法术对塔 EV 闸门（硬约束）**：`rl/action_mask._spell_tower_ev_illegal`——双倍期前，伤害法术落点只罩对手公主塔、溅射内无部队/建筑、且对塔伤折费（引擎标定值/500）< edw×卡费 → mask+validate 双拒；含部队放行、双倍期放行（selftest: `test_spell_tower_ev_gate`）。
     - **评估对照机制**：solo 曲线对手自我对冲无区分度 → `train_solo.run_solo` 每周期加两组对照局（vs baseline0 训练起点 / vs baseline_prev 上一评估点），种子错开、不落盘、不进训练迭代；写 `solo_state.json` 的 `_controls_history`（[{step,vs,winrate,...}]）。
     - **工具③接入 belief_planner**：`_spell_cast_value` 调 spell_module.best_cast 引擎估值——`_spell_trade` 罩不到任何目标的法术跳过、value_estimate 混入账面分；`_spell_finish` 用双倍期汇率（1费≈50塔HP）核算磨塔账面，纯罩塔且亏费否决（注意：磨塔账要用 late 汇率，用前段 500 汇率会把设计内磨塔误杀——bp 规则测试 S10 踩过）。
   - **2026-09-07 带闸门 100k 复训实测（fresh，同卡组同 16 局/2k 步协议）**：①前段砸塔率 54%→26%（终点局取证，205 施法中前段 53 次），闸门方向生效但未归零——残余多为"法术是唯一打得起的牌"+双倍期边界情形；②对照曲线 vs baseline0 全程 0.458±0.025、50k 后 0.512±0.025，**起点对照首次可测**（旧曲线对手每 2k 步同步自抵消，无法证伪），但涨幅在 16 局/点的噪声内（±0.12），结论=无大幅退化、无快速爬坡，100k 步量级仍是平台期；③vs baseline_prev 后 10 点 0.563>0.5，缓慢正向漂移；④出牌节奏 20→38/局（对手 21→36 无摆烂）、引擎级拒绝 0/602；⑤高承诺卡（Giant 4.1%→0.5%、MiniPekka 4.8%→2.7%）被 8g 不裸下门明显抑制，Fireball 占比升（10%→14%，含更多解场用法）。**结论：闸门有效、评估口径修复完成；训练量是下一个瓶颈（FirstLight 同期用 2500 万动作的 IL 底座）。**
2. **推理时浅 MCTS** 挂评估路径（零训练风险）——直接量出"搜索能赚多少 Elo"，作为一切搜索路线的前置验证。
   - **2026-09-07 已实现（`rl/mcts.py`，设计文档 `docs/mcts_design.md`）**：UCT + 引擎确定性叶推演（economy 口径值函数：塔损归一 + 两段相位权重 + 圣水差资源账）；候选动作复用 `legal_cells`/`validate_bundle`（8h/8g/9i 闸门与提交路径同源，搜索不另造规则）；策略 (slot,cell) logits 作先验截断（可选）；对手响应经 `opponent_fn` 回调注入（None=不响应下界）。实测 24 sims ≈ 1.4-3.2s/决策帧（受候选数影响），一局挂搜索可承受于评估。selftest: `test_mcts_basic`/`test_mcts_defense_and_wait`（空场合法+原局零污染+确定性、量纲回归、预算控制）。
   - **实现教训（搜索值函数勿重蹈）**：①**量纲必须与训练奖励自洽**——曾把塔伤项 ×1000 抬尺度而 edw 保持 0.5/费，塔伤/费差失配 1000 倍，空砸王塔被误判正 EV（75 血×0.001×1000=75 vs 3 费×0.5=1.5），去掉 ×1000 后自然一致；②`get_crown_count()` 语义是**自己丢的塔数**（非拿到的皇冠），皇冠差要取 op−me；③v1 决策帧语义="我方出手+对手同帧响应→仍轮我方"，树全部节点同视角，回传**不做符号翻转**（翻转只适用真交替树，误用会把高分动作回传成负值）；④候选枚举每槽独立 cap（总 cap 会把 4 槽挤成 1 槽），缺省格序按"距最近敌军/罩到敌方价值"排序（纯 y 主序在 cap 内全是底线格，防守/斩杀格永远进不了候选）。
3. **LLM 局间顾问**（实验性）——把人工录像取证（行为病理诊断）自动化；注意它只解决"发现问题"，修复仍需人。
4. **专家迭代 / 训练时搜索**——等基线收敛、GPU 空闲（YOLO 训练结束后）、且步骤 2 证明搜索增益为正之后再启动。

### 9j「单边堆牌/不对牌」三层修复（2026-09-08 落地，待复训验证）

**问题定性**（FirstLight 对比分析 + replay 取证 `scripts/forensics_response.py`）：
模型防守响应率 41.9%（对照对手 60.7%）、威胁→首次响应延迟中位 4.0s、威胁场景
62% 的落点距过河敌军 >8 格——**时机响应尚可、空间上不对牌**。根因两层：①训练对手是
frozen_copy，两边都不防守时"换家"是合法策略（单边堆牌在训练分布里不受惩罚）；
②对手事件信息（"0.5s 前在桥头下了 Giant"）要靠 ~9 格感受野的 CNN 自己从 grid 重新发现。
**取证教训：P0 塔 y≈3-6.5（y 小半场）、P1 塔 y≈25.5-29，"敌军过河"= P1 troop y<16——
第一版取证把半场方向量反了（crown0=被 P0 承认的皇冠数是旁证），坐标口径必须用
entity 实测 + `belief_planner.BRIDGE_Y` 注释互证。**

- **A 层 训练对手池**（`train_solo._OpponentPool`）：frozen 副本 70% + 历史 checkpoint
  PFSP 20%（`_collect_hist_ckpts` 按步数均匀抽 ≤12 个，`pfsp.PFSP` 乐观先验，
  每局结束回填胜负）+ **真防守脚本 10%**（`opponents.SelfDefenderPolicy`：
  复用 `simulate_exchange.script_defender` 的 (DPS+HP/15)/费 反制 + 塔前迎击线，
  无威胁帧 60% 停手缓出）——单边推进在防守对手面前直接亏塔损，换家 meta 失效。
  selftest: `test_opponent_pool_mix`。
- **B 层 对手出牌事件通道**（`belief.py`）：最近 3 次对手出牌 → 每条
  [card_onehot(13), x/17, y/31, Δt/10] 3×16=48 维追加在 belief_token 尾部
  （23→71 维）。**旧 checkpoint 尾零兼容**：`follower.load_checkpoint` 对
  `belief_mlp.0.weight` 复制 plan_mlp 的"前列拷贝+尾部清零"模式（真实 100k 旧 ckpt
  验证：前 23 列语义保留、事件列从零学）。Δt 以当前决策时刻为基准=事件陈旧度可学，
  clamp 1.0。调用方零改动自动生效（现有 encode(obs,None)→step→update(opp_played)
  序列天然因果正确）。selftest: `test_opp_event_token`。
- **C 层 过河即防**（`belief_planner.plan` 回退段）：敌军过河（y<16）即建议
  defend_left/right + `placement_hint="bridge_front"` + target_kind="unit"，
  focus_region 对准威胁 x——不再等威胁当量攒够 2.0（单单位过河 threat=1.0 会被
  旧条件漏给 push/cycle）。软偏置语义；spell_trade/soft_control 优先级不变。
  selftest: `test_crossed_river_defend_plan`。
- **配套教训**：①`ScriptedPolicy(mode="heuristic")` 实为 mask 随机（P0-3 时代占位），
  "会防守的脚本对手"必须用 SelfDefenderPolicy（其反制落点是**世界坐标**，
  塞 ActionBundle 前必须做世界→本地网格逆变换，P1 有镜像——曾直接塞导致
  部署非法）；②belief token 维度 71 硬编码进 selftest 的策略构造（23 会在
  act 时 mat1/mat2 失配）；③对手经 `env.opponent(obs1)` 调用时 env 引用需在
  构造时注入（`SelfDefenderPolicy(env=...)`）。

### 外置工具（2026-09-06 起步，用户确认的路线）

外置工具 = 引擎侧确定性服务，模型/规划器按需调用，不进动作空间。第一个已落地：

- **①塔伤威胁计算器 `threat_calc.estimate_tower_threat(battle, player_id, horizon=20)`**（2026-09-07 已落地）
  - 语义："双方都不再部署"假设下，敌方现存部队未来 horizon 秒内对我方各塔的伤害；
    我方现存部队照常防守（= "我不再投入资源" 的塔损）。
  - 实现：deepcopy BattleState + 引擎确定性推演（索敌/攻速/位移/塔兵/王塔激活零口径偏差）。
  - 成本实测（2026-09-06）：空场快路径 0ms；Giant 单兵 ≈25ms；13 实体交火盘面 ≈256ms
    （20s 视界 × 1200 步）。适合规划器/评估每决策帧一次；训练循环逐帧调用需掂量。
  - 返回 `{"left","right","king","total","towers_lost","sim_time"}`；确定性、无副作用
    （selftest: `test_tower_threat_calc`）。
  - 后续接入点：belief_planner 候选动作估值 / prophet 特征 / RL 观测附加通道。

- **②交换模拟器 `simulate_exchange.simulate_exchange(battle, player_id, card, pos, horizon=10, defender=...)`**（2026-09-07 已落地）
  - 语义："我方现在打出这张牌"的反事实推演：deepcopy → 真的 deploy（走引擎全部
    合法性校验，非法报 `legal=False`）→ 我方按"不再投入"处理 → 推演到视界。
  - 对手防守三模式：`none`（= threat_calc 对照下界）/ `script`（内置确定性基线防守：
    威胁过河或逼近塔 10 格时，手牌按 (DPS+HP/15)/费 选最优反制、塔前迎击线候选落点，
    纯函数无副作用）/ `fn`（注入 `opponent_fn(sim, defender_id)`，多源对手采样入口，
    对应 AlphaStar 对手模型偏差对策）。
  - 返回：双方各塔掉血/被破、`my_cost/opp_cost`（引擎动态费用口径）、我方产出实体
    存活率、对手部队击杀数与总伤。确定性、无副作用（selftest: `test_simulate_exchange`）。
  - 实现教训（写类似工具必读）：**实体记账必须 `isinstance(e, (Troop, Building))` 过滤**
    ——塔攻击的 Projectile（player=对手、hp=塔血量级）会被误计成对手部队；
    死亡单位会从 `battle.entities` 移除，存活统计需在吸收时留 hp0 底稿；
    候选部署必须由调用方执行（引擎 deploy 无 dry-run，策略函数内试部署=真部署）。
  - 用途：与 threat_calc 做差得"这张牌挽回多少塔损"→ 条件威胁数据集标签 /
    模块对账老师 / 浅 MCTS 叶估值。

- **③法术知识模块 `spell_module.py`**（2026-09-07 已落地，exposure level 1-2 参照实现）
  - 核心决策：伤害数字**不做静态公式复刻**，全部**引擎实测标定**（gamedata 数值路径
    含 OFFICIAL_OVERRIDES/damage_per_level/DOT buff_data/Arrows 特例，复刻必漂移）。
    每法术对三靶（Giant/公主塔/Cannon）一次性标定，按 (卡名, 等级) 缓存。
  - API：`get_spell_profile`（档案+标定伤害）/ `evaluate_cast`（落点估值：罩到谁、各
    掉多少血、谁被砸死、击杀折费——零推演成本）/ `best_cast`（敌实体包围盒粗网格扫
    最优落点，缺省评分 = 击杀折费 + 塔伤/500 + 部队伤/650）/ `engine_resolution`
    （逐实体实测口径，对账老师/数据集标签）。
  - 对账 selftest（`test_spell_module`）：Fireball 对塔 206/对部队 688 标定值；
    evaluate_cast 预测 == engine_resolution 实测（Fireball/Zap/Rocket/Arrows 逐目标 ±2）；
    Rocket 砸死 Cannon 击杀判定一致。注意：**移动目标是静态预测的固有乐观项**
    （Minions 出牌即飞，可能脱离后续波次半径），对账用静止靶。
  - 实现教训：延迟生成实体必须逐步归入施法实体集（Arrows 三连波只量到第一波的坑）；
    空/地过滤只对真有 projectileData 的法术应用（Projectile({}) 包装器默认名
    'Unknown' 不能作判据，Zap 被误滤成无目标）。
  - 顺带修复引擎既有 bug：`Building.take_damage` 缺 `pierce_invincible` 形参，
    AreaEffect 类法术（Zap 等）砸普通建筑（Cannon/哥布林小屋）必崩 —— 真实对局路径，
    非 selftest 独有。

### 卡牌知识模块（外置工具③④，2026-09-07 设计定稿，未实现）

**核心思想**：现有信息通路（grid 观测 / belief token / plan token）缺"每张卡的条件化知识"
这一中间层——plan token 是卡无关的，模型只能靠 RL 试错学每张卡的用法，8k 步量级学不完。
给每张（类）卡配一个引擎侧确定性评估器，输出**特征而非动作/闸门**（老原则：硬闸门只留给
已证实病理，知识用特征注入保留学习自由度）。

- **架构**：每模块两接口——静态能力（gamedata 数值）+ 情境估值（复用 threat_calc 差分/
  几何查询）；只评估当前手牌 4 张，每帧 4 次求值成本可控；输出定长向量并入观测或 plan
  通道（plan_space v1 的"尾部追加 + 旧维度逐位不动"模式，旧 checkpoint 兼容）。
- **暴露机制（分批解锁）**：按知识深度 exposure 等级（0=纯RL / 1=静态能力 / 2=防守交换
  估值 / 3=完整建议含最优落点）；训练配置 `card_knowledge_level`，eval 平台期升一级，
  禁用通道置零。按卡类分批：先法术模块（病理已确诊），部队模块验证有正收益后再加。
- **验证纪律**：每个模块必须配引擎对账 selftest（估值 ≈ rollout 实测 ± 容差）——
  教训：crownTowerDamagePercent 存 −70 非 wiki −65、Arrows 25/122 特例、手写成本表
  Archer 2 费 bug；模块里所有数字必须来自 gamedata 或实测标定，禁止拍脑袋。

**循环窗口求值契约（关键修订）**：模块求值范围是"循环队列窗口"而非仅手牌——
只评估手牌 4 张会让模型失去"过牌换更有价值牌"的跨期决策能力（plan_space 已预留
cycle_and_wait / cycle_small / save_ace 意图槽，缺的就是接地信号）。

- 手牌 4 张 + next card（cycle[4]，公开信息）都要求值；cycle_gain =
  value(next) − max_value(想弃手牌) 作为过牌增益特征；决策仍留给网络。
- next card 价值是"若此刻在手"的一阶估计（到手时盘面已变）；盘面快进后再求值是
  level 3 的精化，不起步就做。

**循环规划器（外置工具④，与 SpellModule 互补）**：管"想要的牌多久能到"。
- 动机：小费卡组一次惨烈防守可能要转一整组牌（如出两张建筑过牌）——多卡循环防守的
  功劳分配横跨 20+ env 步，RL 试错学不动；而自己牌序是完全自知的私有信息、队列机制
  确定性 → 该引擎算的不该让网络猜。也不写成硬策略（守/弃塔/换路是复合权衡，规则会
  变成硬伤）。
- 输出特征（全部是事实非决策）：deck 每张牌的 depth（还要出几张进手）、过牌 ETA、
  循环总成本（圣水+位置损失）、对照 threat_calc 的"最佳防守牌 depth/ETA/期间塔损"。
- 信息口径：合法已知 = 手牌4 + next 确定位置 + 末尾 3 张仅集合已知（顺序未揭示）——
  末尾 3 张的顺序不确定性由规划器按分布处理，不假装全知。
- 病理关联："一够费就出最便宜牌"的倾倒行为是该策略的雏形，缺的只是"倾倒是在往哪个
  counter 靠近"的价值信号；补上后病理直接转化为正确过牌行为。

**落地顺序（已确认）**：① 手写 SpellModule 参照实现 + 引擎对账 selftest（不依赖 LLM，
同时把验证链路调通，也是 LLM 的 few-shot 范例）；② LLM 生成流水线：录像按卡筛选+奖励
归桶作证据、gamedata 为 ground truth 防幻觉、API 白名单沙箱（禁 get_prophet_state 等
特权访问，人工审查防泄漏）、双验证门（对账 selftest → 训练 eval 涨留跌滚）；③ 之后再做
循环规划器。风险预案：模块建议过拟合到取证录像分布 → PFSP 多对手池 + 局外验证局。

### 条件威胁评估（2026-09-07 讨论定稿，用户核心关切：泛化能力）

**问题定义**：威胁是关系性的——"地狱塔 vs 我方石人核心"是 N×N 卡对条件判断，手写规则
每卡需上百条交互，不可行。目标 = 自训练权重学习"对手现状对我方的条件威胁"（含牌序错开：
地狱塔在手 depth=0 与 depth=5 是两个局面）。

**为什么学习式成立**：卡对交互是低维属性空间（费用/HP/DPS/射程/对空/单体/AOE/建筑/
渐增等）上的连续交互函数——函数逼近内插任意卡对，无需枚举 N²。

**标签来源（引擎白送）**：`threat_calc` 回答无条件威胁（现存部队不管会怎样）；其
deepcopy+rollout 模式直接扩展出反事实版：

- **工具② simulate_exchange(battle, player_id, card, pos, horizon)**（下一个该写）：
  deepcopy → 真的 deploy 候选牌 → 对手按手牌信念/冻结副本防守 → 推演到视界 →
  输出双方塔损/部队存活率/圣水交换。
- 一物三用：① 卡牌模块对账老师；② 条件威胁数据集生成器（扫"我的候选牌 × 对手盘面/
  手牌状态"）；③ **浅 MCTS 缺的价值网络**（评叶 = "这个交换值不值"），训练数据免费。

**交互评估头（"训练一种权重"）**：输入 = 属性化盘面编码 + 我方候选牌属性 + 对手手牌/
牌序状态（counter depth/ETA 是一等特征——"现在攻 vs 等一张再攻"的依据）；输出 = 期望
交换结果。喂给 PPO 当特征或搜索先验。威胁头/解牌价值头/对手手牌头（已有 hidden_labels
监督）同属 auxiliary tasks 家族。

**必须处理的两个偏差**：
1. **对手模型偏差**（AlphaStar 教训）：标签生成时对手若只用冻结副本，评估头过拟合该
   对手。对策：rollout 对手多源采样（历史 checkpoints + script + 信念采样防守）。
2. **条件完整性**：评估头输入必须含对手牌序深度，否则"错开牌序"无从谈起；对手出牌
   历史完全可观测，depth 追踪是合法信息（belief 模块增量工作）。

**路线关系**：threat_calc（已落地）→ simulate_exchange → 交互评估头串成主链；
SpellModule/循环规划器降格为冷启动脚手架，评估头成熟后接管，模块退居冷门卡兜底
+ LLM few-shot 范例。

### 相关诊断结论存档（2026-09-06）

- 模型观测无战争迷雾，敌方部队全可见；预判能力不是行为病理的瓶颈。
- 引擎法术对塔降伤正确生效（`King_PrincessTowers` 命中 "King" 分支；Fireball 对塔 30%，lv11 实测 206 伤）。
- 法术砸塔行为病理（economy 8k 步录像，480 次施法取证）：
  - 双倍期砸塔仅 1%（设计上 `tower_dmg_late=0.002` 使砸塔双倍期近似正 EV，属有意设计）；
  - 前段砸塔 16%，净 EV 约 −1.79/次（tower_dmg_opp 0.001 × 206 − edw 0.5 × 4 费），惩罚数值已足够但欠训练未吸收；
  - **63% 的砸塔发生在"法术是手里唯一打得起的牌"时**（手牌重建法：8! 枚举初始牌序 + 槽位/圣水流水双重校验），机制 = 全高费手牌 + 最便宜卡是 3 费法术 + "一够费就出牌"习惯 + 空砸闸门把落点限制到塔，四者叠加；
  - 干预方向：前段法术对塔 EV 闸门（`伤害×30% < edw×费用` 时塔格非法，双倍期放行），把账面惩罚升级为硬约束。
