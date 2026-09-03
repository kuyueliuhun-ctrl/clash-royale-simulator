# RL 训练链路修复手册（评审版）

> 适用范围：`src/clasher_new/rl/`（25 个文件，约 2840 行）
> 依据：`docs/ai_training_plan.md` 规格 + 四轮模块评审 + 本人逐条代码复核
> 状态：`selftest` 8/8 通过；`py_compile` 全过。但存在 **6 条 Critical**，修完之前任何训练曲线与评测结果都不可信。
> 本手册每条均给出：问题 → 证据位置 → 根因 → 修复方案 → 验收标准。编号即修复引用号。

> **修复进度（2025）**：P0-1~P0-6 全部修复并有回归测试（`selftest.py` 现 18 项，含
> §6 清单的 hidden 重放、熵方向、掩码不变式、heuristic 出牌率、exploiter 加载、
> 信念免疫技能哨兵等）。P1：1、2、3、4、5、6、7、8、9、10、11、12、13、14、15、16、
> 17、18、19、20、21、22、23 已完成；P2 清理批次大部分完成（死代码/维度/坐标/空间
> 声明/联赛细节/env 边界），4.2 包相对导入重构暂缓（低风险、跨模块，另行安排）。
> 详见 `src/clasher_new/rl/README.md`。

---

## 0. 结论速览

| 级别 | 数量 | 含义 |
|---|---|---|
| 🔴 Critical（P0） | 6 | 学习信号错误 / 必崩路径 / 评测基线失效，必须先修 |
| 🟠 Major（P1） | ~20 | 训练信号质量 / 规格闭环缺失 |
| 🟡 Minor（P2） | ~40 | 死代码、命名、seed、性能等清理 |

**核心判断**：
1. PPO 的 hidden 重放不一致（P0-1）使整个 PPO 目标失效——**现在是"能跑但学不到正确的东西"**；
2. 掩码坐标/缓存两处 bug（P0-3、P0-4）使 heuristic 对手实际出牌率 0.5%——**"vs Heuristic" 评测基线无效**；
3. 熵项方向反（P0-2）、维度硬编码（P0-5）、哨兵崩溃（P0-6）分别破坏探索、exploiter 链路与信念模块。

---

## 1. 修复优先级总览

| 批次 | 编号 | 一句话 |
|---|---|---|
| **P0-A** | P0-1 | PPO hidden 重放不一致 → 目标失效 |
| **P0-B** | P0-2 | 熵项方向反了 |
| **P0-C** | P0-3 + P0-4 | 掩码缓存缺 player_id + 掩码/提交坐标分裂（同一批改） |
| **P0-D** | P0-5 | plan/belief 维度硬编码 20/32 vs 实际 21/23 |
| **P0-E** | P0-6 | `"__ability__"` 哨兵 → 信念 KeyError / 整组重置 |
| **P0-F** | P1-1 | train_belief 的 GRU 在单帧上训练（信念模块无效） |
| **P1** | P1-2 … P1-20 | 见第 3 节 |
| **P2** | P2-1 … | 见第 4 节 |

---

## 2. P0：Critical（阻塞训练有效性 / 直接崩溃）

### P0-1　PPO hidden 重放不一致 → 整个 PPO 目标失效

- **证据**：`rl/ppo.py:58-59` 调 `policy.evaluate(obs, belief, plan, bundle, masks)` **不传 hidden**；`rl/follower.py:169-170` 在 `hidden=None` 时置零初始化 GRU。
- **根因**：rollout 时（`follower.act`，`follower.py:119-122`）hidden 跨决策步链式传递并回传下一个 `act`；但 `update` 重放每条 transition 时从**零隐状态**只用该步 obs 重算 `lp_new`。重放时的策略状态 ≠ 采样时的策略状态，`ratio = exp(lp_new - old)` 不再是有效重要性采样比。
- **影响**：PPO 目标被破坏，梯度方向不可信。这是本仓库最致命的一条。
- **修复方案**（二选一，推荐 A）：
  - **A（存 hidden）**：rollout 时每个 transition 存 `init_hidden`（进入该步时的 hidden，`detach()`）；`update` 中 `evaluate(..., hidden=t["init_hidden"])`。
  - **B（按 episode 重建）**：transitions 带 `episode_id`/`seq_idx`，update 内按 episode 顺序前向展开重建 hidden。
- **验收**：`evaluate` 用 rollout 时记录的 hidden 重放，`lp_new ≈ old`（同一策略、同一 transition，未更新前差值 < 1e-3）。给 selftest 加 `test_hidden_replay_consistency`。

### P0-2　PPO 熵项方向反了

- **证据**：`rl/ppo.py:66` `ent = -lp_new.mean()`。
- **根因**：`lp_new` 是整个 bundle 的联合 logprob（非分布熵）。代入 `loss - ent_coef*ent` 后实为 `loss + ent_coef*lp_new.mean()`——最小化会**压低已选动作概率**，与探索目标完全相反。
- **影响**：熵正则变"反探索"，配合 clip 会抑制学习。
- **修复方案**：让 `follower.evaluate` 额外返回真熵：`entropy = slot_dist.entropy().sum() + cell_dist.entropy().sum()`（对所有 decoder 步求和），`ppo.py` 用该值。
- **验收**：熵项非负且对随机策略接近理论值；selftest 断言 `entropy >= 0` 且更新后已选动作在正优势下概率上升。

### P0-3　掩码缓存指纹缺 `player_id`

- **证据**：`rl/env_wrapper.py:149-161`，`fp = (tick, tuple(hp...), tuple(building_positions))`，无 `player_id`。
- **根因**：训练主循环先 `get_action_mask()`（P0）→ `step()` → `_run_opponent()` 调 `get_action_mask_for(1)`。此时 tick/塔血/建筑未变 → 指纹命中 → **P1 拿到按 P0 手牌与 P0 部署规则算的 cells**。
- **影响**：所有自博弈/评测对手的掩码全错。
- **修复方案**：`fp` 加入 `player_id`（建议再含 `tuple(p.cycle)`）；`reset()` 显式清空 `_mask_fp/_mask_cells`。
- **验收**：`get_action_mask_for(0)` 与 `get_action_mask_for(1)` 在同一 tick 返回不同 cells；selftest 断言两侧掩码与各自 `validate_bundle` 一致。

### P0-4　掩码坐标系与提交坐标系差一次中心镜像

- **证据**：`rl/action_mask.py:84` `legal_cells` 用**世界坐标** `Position(x+0.5, y+0.5)` 校验；提交路径 `rl/action_bundle.py` `sa.to_position(player_id=1)` 会**镜像**为世界坐标（`env_wrapper.py:204` 沿用旧约定）。
- **根因**：掩码说"格 (x,y) 合法"的语义是世界坐标，执行语义是"玩家本地坐标、需镜像"。两者对 P1 相差一次镜像。
- **影响**：P1 的 cells 掩码大面积失真，基于掩码出牌的对手大半被 `deploy_card` 拒绝——实测 heuristic 出牌率 0.5%（vs Heuristic 基线失效，`evaluate.py:40-41` 受影响）。
- **修复方案**：统一约定 **"SubAction 的 (x,y) 一律是玩家本地坐标"**。`legal_cells` 对 `player_id==1` 在 `_position_legal` 前先镜像：`pos = Position(17.5-x, 31.5-y)`；或抽一个 `sub_position(player_id, x, y)` 供掩码与提交共用。
- **验收**：新增 selftest 不变式：`mask["cells"][slot][y,x] == True ⟹ validate_bundle(from_single(slot, x, y)).ok`（对两个 player_id 都成立）。heuristic 对手出牌率恢复到 random 同量级。

### P0-5　plan/belief 维度硬编码，三处口径互相矛盾

- **证据**：`rl/train_follower.py:95`（`plan_dim=20, belief_dim=32`）；`rl/follower.py:34` 默认值同样 20/32；`rl/run_league.py:28` 硬编码 23。真实维度：plan=21（`PlanToken.to_vector()`）、belief=23（`BeliefInference.encode(None,None)`）。
- **影响**：exploiter 加载 main checkpoint **必崩**（size mismatch，已复现）；`evaluate.py:24`、`run_league.py` 等加载训练主策略时同样有崩溃/静默截断风险。
- **修复方案**：
  1. 单一常量源：`PLAN_DIM = len(PlanToken().to_vector())`、`BELIEF_DIM = len(BeliefInference(opp_deck=...).encode(None, None))`；
  2. checkpoint 保存元数据 `{"state_dict", "plan_dim", "belief_dim", "hidden_dim"}`，加载时读取并校验；
  3. 删除 `follower.py:34` 的魔法默认值，改为必需参数或工厂函数。
- **验收**：`train_exploiter` 能加载 main checkpoint 正常训练；`evaluate`/`run_league` 加载任意 checkpoint 不崩。

### P0-6　`"__ability__"` 哨兵泄漏进信念模块 → KeyError / 整组重置

- **证据**：`rl/env_wrapper.py:209` 上报 `played = "__ability__"` → `rl/belief.py:65` `Card("__ability__").type` → `card_data` **KeyError 崩溃**；粒子滤波 `bayes_filter.py:57-61` 把哨兵当"未知牌"→ **全量重置信念**。
- **影响**：对手（如 FollowerOpponent）一触发英雄技能，训练/评测下一轮 `belief.update` 即崩；不崩时信念被清零。
- **修复方案**：
  1. 最小修复：`BeliefInference.update` 入口过滤非 `ENTITY_NAMES` 的字符串（`"__ability__"`、`None`）；
  2. 治本：`_run_opponent` 改为结构化上报 `[{"card": ..., "x": ..., "y": ...}, ...]`（见 P1-5），技能用独立字段。
- **验收**：selftest 新增 `test_belief_survives_ability`：连续出牌后触发一次技能，`next_probs` 不跌回均匀先验、不抛异常。

---

## 3. P1：Major（训练信号质量 / 规格闭环）

### P1-1　train_belief 在长度 1 的序列上训 GRU（信念模块无效）
- **证据**：`rl/train_belief.py:106-107`（`seq = xs.unsqueeze(1)`），逐样本独立、隐状态零初始化、batch 内 `permutation` 打乱时序。
- **根因**：GRU 被训练成单帧 MLP；单帧特征从数学上推不出 8 卡循环（`y_next/y_hand` 只能学到先验）。推理时 `belief.py:124` 却喂最长 32 帧真实历史——训练/推理分布不一致。
- **修复**：按 episode 组织 `(B, T, D)` 序列（逐局 pad+mask 或滑窗），隐状态沿时间展开训练；或逐局顺序流式训练携带 hidden。
- **验收**：验证集 next-card 准确率显著优于 uniform 先验；Brier/log-loss 优于"只看上一张牌"启发式（对应规格 §8.3）。

### P1-2　StatisticalBelief 大面积失效
- **证据**：`rl/belief.py:60-84`：`tendency_counts` 从未 +1、`push_back_count` 恒 0、`side_counts` 因所有调用点 `opp_x=None` 永不更新。
- **修复**：让 env 上报对手出牌落点 x 与卡类型（结构化 `opp_played` 后自然解决）；或删除该层只留神经编码，避免"看似三层实为一层"。
- **验收**：两侧出牌后 `side_counts`/`tendency_probs` 有实际区分度。

### P1-3　对手圣水估计缺失
- **证据**：`rl/belief.py:196` `elixir_est` 参数被忽略，`elixir_mean/std` 恒为 5.0/1.0；`encode()` 里的 `elixir_mean` 是死特征。
- **修复**：实现简单圣水计数器（初始 5.0，按决策间隔回复，按观测出牌扣费），写入 `elixir_mean/std`。
- **验收**：随机对手下 `elixir_mean` 与真实圣水的 MAE 显著低于常数 5.0。

### P1-4　先知 macro_intent 恒为 defend_*（塔被算进推进压力）
- **证据**：`rl/prophet.py:33-51`（复制到 `belief_planner.py:29-35, 82-89`）：双方 6 座塔都是 `battle.entities` 成员且 `is_alive` → `threat ≥ 6.0`、`my_pressure ≥ 6.0` 恒成立 → intent 几乎永远 `defend_left/right`；`bundle_hint` 的 `threat>0.3 and my_pressure>0.3` 恒真。
- **修复**：实体循环过滤静态建筑（`e["name"]` 含 `Tower` 跳过，或 `Card(e["name"]).type == "building"` 排除）；阈值改到与单位数量/HP×费用匹配（如 `threat >= 2.0`）。
- **验收**：新增单测断言"空场开局 intent != defend_*"；`push_*`/`counterpush` 分支可达。

### P1-5　对手多卡出牌只上报最后一张、丢落点
- **证据**：`rl/env_wrapper.py:205-214`（`played = card` 循环覆盖）。
- **影响**：信念监督样本与规则信念系统性漏记先出的卡；`side_counts` 无数据源（P1-2 根因）。
- **修复**：`info["opp_played"]` 改为列表 `[{"card", "x", "y"}, ...]`（或新增 `opp_played_all` 兼容键），`replay.py`、`belief.update` 按列表逐条更新。
- **验收**：对手一 tick 出两张时，信念对两张卡都做了排除。

### P1-6　动态掩码不排已用槽 / 技能已用
- **证据**：`rl/action_mask.py:35-41`、`rl/follower.py:133-160`、`rl/env_wrapper.py:166-180`：`slot_mask` 不感知 `partial_bundle` 已消耗资源；`ability_legal` 用真实圣水而非模拟扣费后。
- **影响**：同刻多卡（规格 6.3.2 核心）几乎学不出，奖励被 `-0.05*invalid` 污染。
- **修复**：`slot_mask` 增加 `used_slots` 参数并内置 False；`cells[used]` 同理置 False；`ability_legal` 在 bundle 已含 ability 后置 False；`follower._slot_mask_tensor` 消费 `used_slots`。
- **验收**：bundle 内第 2 个子动作的掩码不会再包含第 1 个已选 slot。

### P1-7　截断 episode 的 GAE 把截断当终止
- **证据**：`rl/ppo.py:38-41`：`t == T-1` 强制 `next_val = 0`。
- **影响**：`max_ep_steps` 截断被当作真实终止，value/advantage 系统性低估（截断是唯一未正确处理的路径）。
- **修复**：transitions 存 `truncated` 标志；截断时 `next_val = value(T)`（bootstrap）。
- **验收**：构造一个截断 episode，断言其 GAE 不再等于"当终止"的结果。

### P1-8　FollowerOpponent 是"残废版 main agent"
- **证据**：`rl/train_follower.py:52-78`：belief/plan 恒零向量、mask 全 1（非法部署被静默拒）、hidden 跨 episode 不重置。
- **修复**：装配独立 `BeliefInference` + `BeliefPlanner`；mask 用 `get_action_mask_for(1)`（依赖 P0-3/P0-4 修复）；reset 时清对手 hidden。
- **验收**：exploiter vs main 的胜率能反映真实强弱（换边对打，见 P1-14）。

### P1-9　联赛 checkpoint 快照语义错误
- **证据**：`rl/league.py:49-52` `register_checkpoint` 把 main 本体 `kind` 改为 `"historical"`，但 `policy` 仍指向**训练中的同一活对象**。
- **修复**：注册新条目（`f"{agent_id}_ckpt{n}"`），保存**权重副本**（新 `FollowerPolicy` + `load_state_dict` + `eval()` + `requires_grad_(False)`）；`metadata` 存入条目；原 main 保持 `kind="main"`。
- **验收**：训练 N 步后，historical 快照的参数不变（与注册时一致）。

### P1-10　run_league 实为有偏评估脚本，联赛主循环缺失
- **证据**：`rl/run_league.py`：无训练循环、无快照注册、无 exploiter 轮换、无持久化；`League.sample_opponent` / `PFSP.sample` 全仓库无调用方。
- **修复**：拆成两个入口——`evaluate_league`（保留轮转评估）与真正的 `run_league` 主循环（训练 N 步 → PFSP 采对手 → 周期 round-robin 评估 → Elo → 快照 → exploiter 轮换）；联赛状态（ratings/winrates/成员）序列化到 json/torch 文件。
- **验收**：`run_league` 能持续训练并产生 Elo 排序；重启后可加载联赛状态继续。

### P1-11　Elo / PFSP 每对只更新一次
- **证据**：`rl/run_league.py:94` 把 20 局聚合成一次 `record_match`（k=32 只走一次）；`pfsp.py:24-28` 对聚合分数做一次 EMA。
- **影响**：16:4 vs 11:9 的 Elo 差距被压缩到 ~10 vs ~1.6，量表失真；PFSP 采样趋近均匀。
- **修复**：逐局调用 `record_match(a, b, 1.0/0.0/0.5)`；`record_match` 支持 `n_games` 参数按局数缩放 K；PFSP 同步逐局录入。
- **验收**：20 局 16:4 与 11:9 的 Elo 差显著拉开；PFSP 对克星对手提升采样概率。

### P1-12　Exploiter 闭环缺失
- **证据**：`rl/train_exploiter.py`：`eval_every=0`、无胜率评估、无阈值回灌、checkpoint 文件名不区分 main。
- **修复**：训练后换边 n 局测 exploiter vs main 胜率；`--winrate-threshold`（如 0.55）达标时 `add_exploiter` + `register_checkpoint`；文件名 `exploiter_{main_name}.pt`；默认 `total_steps` 加大（如 20000+）。
- **验收**：能产出"克制 main"的 exploiter 并进入联赛池。

### P1-13　先知 RL 路线（train_prophet）产物无人消费
- **证据**：`rl/train_prophet.py:105-114` 保存的 `prophet_ppo.zip` 全仓库无加载点；蒸馏只用启发式 `ProphetPlanner`。
- **修复**（二选一）：(a) 给 RL 先知加 plan head 输出 `MACRO_INTENTS`/`FOCUS_REGIONS` 分布 + value，与 `PlanToken.to_vector()` 对齐；(b) 写 `prophet_policy_to_plan(model, obs) -> PlanToken` 适配器接入 `train_follower.make_plan`。至少 README 标注"RL 先知目前是孤儿实验"。
- **验收**：train_prophet 产物能被 follower 蒸馏链路消费。

### P1-14　评测对战不对称（B 方残废 + 无换边 + 平局偏置）
- **证据**：`rl/run_league.py:71-93`：B 方 `np.zeros(23)` belief + `full_mask()`；A 恒为 player 0 先手；平局计为 A 负；引擎 300s tie-break 偏袒 player 1（`battle.py:1263-1266`）。
- **修复**：B 方走完整 belief 链路（封装可 reset 对手类）；每对组合换边各打 n/2 局；区分 win/lose/draw 三态。
- **验收**：对同一对 agent，换边后胜率之和 ≈ 1（含平局）。

### P1-15　先知 focus_region 由风险标量线性映射（空间无意义）
- **证据**：`rl/prophet.py:78`（`region = FOCUS_REGIONS[int(risk*8)]`），`belief_planner.py:100` 同。
- **修复**：region 由 intent + 敌方重心 `enemy_x` 推导（`defend_left → own_left/bridge_left`、`push_right → enemy_right`）；`risk_profile` 保持独立标量。
- **验收**：`intent=defend_left` 时 region 不会落在 enemy_right。

### P1-16　先知未消费对手手牌/牌序（特权信息闲置）
- **证据**：`rl/prophet.py:53-70` 只读 `my_cycle`；`full_state["opp_cycle"]` 未用。
- **修复**：至少两条规则——对手 `opp_elixir < 2` 提高 push 权重；`opp_cycle[:4]` 含法术且我方要 bundle 进攻时降低 `bundle_size_hint`。
- **验收**：构造"对手低圣水"局面，intent 显著偏向 push。

### P1-17　信念训练无校准、数据源单一
- **证据**：`rl/train_belief.py:40-60, 97-121`：只用随机对手 + 固定卡组；无验证集/ECE/温度缩放/早停；`next_acc` 是训练 batch 上的。
- **修复**：混入 heuristic 对手与多副卡组（借 league/pfsp 对手池）；加验证集 + 温度缩放 + ECE/NLL 报告；`hand_head` 加"和为 4"约束。
- **验收**：验证集 Brier 显著优于 uniform 先验（规格 §8.3）。

### P1-18　K_max 强制停止缺失 → bundle 超限 raise 崩 episode
- **证据**：`rl/follower.py:129` 循环 `K_MAX+2` 次 → 第 5 次 `bundle.add` 触发 `action_bundle.py:71-73` 抛 `ValueError`。
- **修复**：掩码层在 `len(partial_bundle.sub_actions) >= K_MAX` 时只放行 STOP；`ActionBundle` 超限由 raise 改为校验拒绝。
- **验收**：10 费 + 四张低费卡局面，follower 能稳定产出合法 bundle 而不崩。

### P1-19　Mirror 校验与引擎语义不符
- **证据**：`rl/action_mask.py:25-32, 187-191` vs `battle.py:1343-1351`：引擎实际费用 = `Card(last_card).elixir + 1` 且按 `last` 复查部署区。
- **修复**：v1 无条件拒绝 Mirror（含单卡），或按 `p.last_card` 复算费用与部署区。
- **验收**：单卡 Mirror bundle 不再通过校验后被引擎拒绝。

### P1-20　整包原子语义被引擎级拒绝破坏
- **证据**：`rl/env_wrapper.py:254-257`：validate 通过后逐个 `deploy_card`，引擎一旦拒绝（掩码误判、`building_positions` 同 tick 快照滞后——`battle.py:1292`），前面子动作已生效 → 半执行状态。
- **修复**：validate 阶段复刻引擎全部拒绝条件（含同 tick 新建筑占位）；`invalid_count>0` 时记 warning/断言；长期给 `BattleState` 加事务式部署。
- **验收**：整包提交后引擎拒绝率降为 0（或全部拒绝路径有日志）。

### P1-21　replay 数据链路脆弱（双格式 / 无消费者 / 体积失控 / 缺意图标签）
- **证据**：`rl/replay.py:41-53` 与 `export_replay.py:52-56` 两套格式；`load()`/`to_belief_dataset()` 无调用方；3 局 41MB（20 局 ≈ 270–400MB）；缺"意图标签"；`record_hidden=False` 静默丢数据。
- **修复**：统一容器格式（带 schema 版本号）；`train_belief` 消费 `export_replay` 产物、删除重复 `collect_replays`；grid 转 float16/int8 或 gzip；补意图代理标签（如对手当时 BeliefPlanner intent）；`record_hidden=True` 且 `hidden is None` 时 warn/raise。
- **验收**：round-trip 单测（save→load→to_belief_dataset 非空且 hidden 对齐）。

### P1-22　evaluate 指标与协议不达标（实现度约 30%）
- **证据**：`rl/evaluate.py`：只实现 Win Rate/Mean Reward/Bundle Size/Bundle Legality/Next-Card Acc；缺 Elo/Crown/Tower 差/Elixir Efficiency/Plan Following/Belief Brier/Hand & Intent Acc/校准曲线；对手池仅 random/heuristic；N=50 ≠ 200。
- **修复**：① 指标补齐（从 `info`/`hidden` 现成数据算）；② 对手池可配置（random/heuristic/checkpoint/main/deck 组合），每组合 N 场、seed 固定；③ 拆 `evaluate_belief()` 独立 N=500 协议。
- **验收**：输出与规格 §11.2 指标表对齐。

### P1-23　缺 BC 初始化阶段与消融评估
- **证据**：`rl/train_follower.py` 无 `--init-from`；评估只 vs random。
- **修复**：补 BC 预训练脚本（对先知/信念 plan 做行为克隆）+ `run_training(init_from=...)`；评估加"有/无 plan、无 belief"三档消融。
- **验收**：能复现规格 §8.5 验收项（有 plan 优于无 plan、无 plan/belief 不崩溃）。

### P1-24　`act()` 在 `no_grad` 外构建计算图
- **证据**：`rl/follower.py:119-122`：`_encode` 与 `gru_cell` 在 `no_grad` 之外执行，返回的 `h` 跨 600 步串联、永不 backward。
- **影响**：内存/CPU 浪费，长 episode 可能 OOM。
- **修复**：`act` 全程 `no_grad`，或返回前 `h = h.detach()`。
- **验收**：rollout 300 步后显存/内存不增长。

---

## 4. P2：Minor（清理批次）

### 4.1 死代码 / 死参数
- `action_bundle.py:60-63,107-112`：`from_tuple`/`bundle_to_legacy`/`legacy_to_bundle` 无调用方且与 `env_wrapper.py:300-305` 的 `legacy_action_to_bundle` **参数约定相反**（`(slot,x,y)` vs `(slot,y,x)`）——统一或删除；
- `action_bundle.py:69`：`stop` 死字段（STOP 由 `STOP_IDX`+空 bundle 承担）——注释或统一 `is_stop`；
- `action_mask.py:132-135`：`bundle_cell_masks` 死代码且"接上即错"（ability 不扣费）——补扣费或删除；
- `belief_planner.py:66-68`：空 `if ... pass` 块；
- `belief.py:139` `predict_next`、`belief.py:86` `side_pref`、`bayes_filter.py:89` `most_likely_hand`：无调用方；
- `train_follower.py:34` `heuristic_opponent` 里 `bp = BeliefPlanner()` 创建后未用（docstring 还声称用信念规划）；
- `train_follower.py:62,65-68`：死赋值（`plan_tok` zeros 后无条件覆盖）、函数内重复 import、永不触发的 `np.pad` 分支；
- `train_baseline.py:86`：`--n-envs` 参数未使用（用户传了不生效）；
- `train_follower.py:81` / `train_baseline.py`：`n_envs` 未接入并行（SB3 单 env + n_steps=2048 极慢）；
- `run_league.py:34,37,79-80`：`opponent_fn` 参数、`belief_b`/`bp_b`、`play` 的 `seed` 参数未使用；
- `replay.py:5,12,19,55`：未用 import、`_active` 只写不读；
- `env_wrapper.py:313-317`：`random_strategy` 死代码、`battle_entity_names` 冗余转口；
- `observation.py:92-99`：`get_crown_count()` 语义反直觉（返回"被摧毁的塔数"）——改名或 docstring 明示。

### 4.2 导入与全局态（跨模块 X6）
- `action_bundle.py:9-14`、`env_wrapper.py:11-13`、`ppo.py:10-12` 独立 sys.path hack → `battle`/`player`/`core` 可能以双身份加载（`isinstance` 失效、`Card.default_level` 分裂，`card_utils.py:154-156` 是"单战斗串行假设"）。
- **修复**：统一包相对导入；RLEnv 构造时显式设置/恢复 `Card.default_level`，或文档限定 SubprocVecEnv。

### 4.3 seed / 复现性
- `env_wrapper.py:76,91-105`：`self.seed` 属性未用；未调 `super().reset(seed=seed)`；`ActionBundleSpace.sample`（L40-42）用全局 `random` 破坏可复现性（§8.1）——sample 注入 rng。
- `train_follower.py:196-210`：`hidden_dim` 未暴露 CLI（恒 128 与默认 256 不一致，checkpoint 隐式耦合）。

### 4.4 观测 / space 声明
- `env_wrapper.py:80,82`：`hand` 上界 13（实际 0..12）、`next_card` 上界 32 魔数（应 12）——`high=len(ENTITY_NAMES)-1`；
- `observation.py:64-68`：`hand` 取 `cycle[:5]` 与 `next_card` 重复编码，规格 3.4.1 要求 `cycle[:4]` + 单独 next_card——统一约定（belief 侧已是 4 张语义）；
- `belief.py:23`：`NUM_CARDS = 13` 与 `ENTITY_NAMES` 重复硬编码——`len(ENTITY_NAMES)`；
- `belief.py:169`：`obs["time"]`（0~180+）未归一化——`/180.0`；
- `follower.py:82-84`：`card_type = rest[..., 0]` 实为 `is_opponent` 通道，`num_classes=4` 浪费 2 维；真类型通道是 `rest[..., 2]`；
- `follower.py:74-77`：未接入 `obs["next_card"]`（规格 6.1）；
- `train_prophet.py:41,52-55,71,79`：priv 观测 `Box(high=13)` 但塔血 ~4000+ 未归一化；`in_ch` 硬编码 14/32/18——复用 `observation.py` 常量并归一化塔血/圣水；
- `prophet.py:58`：Mirror 守卫死代码（卡组无 Mirror）；
- `plan_space.py:47,51`：`suggested_card` 原始整数混入向量（改 one-hot 或 /4）；`value_estimate` clip ±1 吞信息（先 tanh 归一化）；
- `plan_space.py:41,43`：非法 `macro_intent` 字符串会 `ValueError`——加成员校验。

### 4.5 联赛细节
- `league.py:31-33`：`add_agent` 静默覆盖同 id 且 Elo 残留——同 id 抛异常或 `replace=True` 重置 rating；
- `league.py:54-58`：`add_exploiter` id 用"现有数量"，无删除 API 下可碰撞——递增计数器 + 补 `remove_agent`/`retire`（含清理 `pfsp.winrates` 键）；
- `run_league.py:63-64`：第一个 policy 默认标 `baseline`（误导）——默认 `main`；
- `pfsp.py:24-28`：EMA 先验 0.5 + 低 alpha，新对手几乎不被提升采样——未采样对手 winrate 置 0（乐观先验）或加多样性下限；校验 `beta >= 0`；
- `elo.py`：无持久化、无按局数缩放 K 选项；
- `train_baseline.py`：无训练后评估、未加 action mask（§8.2 要求，可用 MaskablePPO/sb3-contrib）。

### 4.6 env 边界
- `env_wrapper.py:266-271`：`speed<1` 取整为 0 模拟冻结；`visualize` 在 step 内 `sleep(1/60)` 拖慢 30×；`assert` 在 `-O` 下失效——`speed<=0` raise、sleep 仅在 render 命中、改 `raise TypeError`；
- `env_wrapper.py:297`：`terminated == truncated == game_over`——外层再包 TimeLimit 会双计，建议 `truncated=False`；
- `env_wrapper.py:216-228`：`_random_opponent` 均匀采样约六成落禁区（"vs Random>90%"基准被弱化）——用修复后的 `get_action_mask_for(1)` 采样；
- `action_mask.py:62-73`：`player_id` 非 0/1 静默按 1——显式断言；
- `action_mask.py:108-121`：`cost <= 0` 兼作"无就绪英雄"哨兵（`manaCost=0` 时误拒）——`ability_mana` 返回 `Optional[float]`；
- `action_mask.py:89-105`：技能就绪判定与引擎偏差（掩码查王塔、引擎查 `use_ability()` 返回值）——注释写明"必要非充分"约定；
- `battle.py:1292`：同 tick 建筑占位快照滞后——`_finish_deploy` 后追加占位或注释接受偏差。

### 4.7 评估 / 测试补强
- `evaluate.py:24`：`belief_dim=23` 硬编码（同 P0-5）；`use_rule_belief` 死参数；循环内重复 import；
- `evaluate.py:85`：平局静默计为不胜 + 引擎 tie-break 偏袒 player 1——输出 win/lose/draw 三态；
- `export_replay.py`：docstring 声称可随机/启发式采集但无 `--opponent` 参数；`range(4)` 硬编码应 `K_MAX`；循环内 import 上移；
- `selftest.py`：`assert sum(hand_probs)==4` 恒真（无效断言，有效的是下一行 top3）；调私有 `battle._spawn_entity` 强耦合引擎——暴露测试友好的 spawn 接口。

---

## 5. 跨模块契约（改一次，到处受益）

| 契约 | 现状 | 统一为 |
|---|---|---|
| **坐标** | `legal_cells` 世界坐标 vs `to_position` 镜像 | "SubAction (x,y) = 玩家本地坐标"，抽 `sub_position(player_id, x, y)` 单一换算函数 |
| **掩码** | 校验层 vs 掩码层对"已消耗资源"理解不一致 | 掩码内置 `used_slots`，`validate_bundle` 与 `get_action_mask_for` 共用同一模拟扣费逻辑 |
| **info 契约** | `opp_played` 字符串 / 哨兵 / 丢坐标 | 结构化 `[{"card","x","y"},...]`，技能独立字段，哨兵不出 info |
| **维度** | plan 20/21、belief 23/32 三处硬编码 | 单一常量 + checkpoint 元数据 |
| **导入** | 多处 sys.path hack | 统一包相对导入 |
| **STOP/K_max** | `stop` 死字段 + raise 超限 | 掩码强制 STOP + 超限校验拒绝 |
| **legacy 转换** | 两个约定相反的函数 | 统一 `(slot, y, x)` 并删冗余 |

---

## 6. 回归测试清单（selftest 新增）

修完一批，把对应断言加进 `rl/selftest.py`，防止回退：

| 测试 | 防的是 |
|---|---|
| `test_hidden_replay_consistency` | P0-1（evaluate 用记录 hidden 重放，`lp_new ≈ old`） |
| `test_entropy_positive_and_sign` | P0-2（熵非负、正优势下已选动作概率上升） |
| `test_mask_validate_invariant_both_sides` | P0-3/P0-4（`mask合法 ⟹ validate 通过`，P0/P1 两侧） |
| `test_heuristic_opponent_actually_plays` | P0-3/P0-4（heuristic 出牌率 ≈ random 量级） |
| `test_exploiter_loads_main_checkpoint` | P0-5 |
| `test_belief_survives_ability` | P0-6 |
| `test_belief_multi_card_update` | P1-5（同 tick 两张都排除） |
| `test_register_checkpoint_isolated` | P1-9（训练后 historical 参数不变） |
| `test_bundle_cap_no_crash` | P1-18（10 费 + 四低费卡不崩） |
| `test_replay_roundtrip` | P1-21 |
| `test_prophet_empty_board_not_defend` | P1-4 |

---

## 7. 建议修复顺序（里程碑）

- **M-0（0.5 天）**：P0-1、P0-2、P0-5、P0-6 —— 纯局部改动，让"训练信号"先变成真的；
- **M-1（0.5 天）**：P0-3 + P0-4 + 契约"坐标/掩码"统一 —— 让评测基线可信，并加两条不变式测试；
- **M-2（1 天）**：P1-1（信念序列化训练）、P1-5、P1-2/P1-3 —— 信念模块真正能用；
- **M-3（1 天）**：P1-6、P1-7、P1-8、P1-18、P1-19 —— 学习循环质量；
- **M-4（1–2 天）**：P1-9 ~ P1-12、P1-14 —— 联赛闭环（快照/Elo/PFSP/exploiter/主循环）；
- **M-5（1–2 天）**：P1-4、P1-13、P1-15、P1-16 —— 先知信号有效化；
- **M-6（1 天）**：P1-21 ~ P1-23 —— 数据链路与评测完备化；
- **M-7（随时）**：P2 清理批次 + 回归测试补全。

每完成一个里程碑跑一次 `selftest`；P0 全部完成后，第一次"有意义的训练曲线"（vs random / heuristic / 历史快照）才有参考价值。

---

## 8. 一句话总结

> **先修 P0-1/P0-2（PPO 两个核心 bug），再修 P0-3/P0-4（评测基线），然后 P0-5/P0-6（链路断点）——这六条修完，"能跑"才变成"学得对"；之后按 M-2 → M-4 → M-5 补信念、联赛、先知，最终用新增的回归测试守住每个修复。**
