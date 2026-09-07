# 推理时浅 MCTS（RL-MCTS v1）— 方案定稿（2026-09-07）

目标：回答 AGENTS 优先级 2 的问题——「搜索能赚多少 Elo」。零训练风险，只挂评估/推理路径。

## 核心设计

**搜索形态**：单玩家决策时点的确定性推演树搜索（MCTS 变体，UCT 选择 + 引擎 rollout 估值）。
不是 AlphaZero 式自博弈搜索——本仓库无迷雾但保留对手不确定性，v1 取「对手按冻结副本策略响应」的
单信念口径（对手模型的偏见不进训练，只在推理时用，风险可控）。

**一次决策的搜索结构**：

```
节点 = (deepcopy 后的 BattleState, 待行动方)
决策帧 dt_decision = 0.5s（与 RLEnv 决策帧一致）
动作 = ActionBundle 的可执行子集（slot+cell，复用 legal_cells 掩码，含 EV 闸门/不裸下门）
叶估值 = 引擎确定性推演 horizon 秒 → 塔血差变化 + 资源账（与训练奖励同构，规则化）
```

- **选择**：UCT（c_uct 可调，默认 c=1.4）；未扩展动作按「策略先验 + 掩码」排序扩展。
- **先验**：FollowerPolicy 的 (slot_logits, cell_logits) 归一化后作为扩展次序先验
  （top-k 剪枝，k 可配，默认 8 个候选 bundle）；无策略时退化为均匀先验（脚本模式可用）。
- **推演**：节点间转移 = 我方 bundle 部署 + 对手 bundle 部署（对手策略以回调注入）+
  battle.step × decision_frames；叶处再向前确定性推演 horizon_s（默认 8s）计塔血差。
- **回传**：value = Δ(对方塔血%归一) − Δ(我方塔血%归一) + edw×(对方费 − 我方差)（economy 口径），
  根节点按玩家视角取 max。
- **确定性假设**：引擎 step 确定性、双方策略 deterministic=True → 同一节点同一动作结果一致，
  树可以复用统计量。

## 预算与参数

| 参数 | 默认 | 说明 |
|---|---|---|
| n_simulations | 24 | 每决策帧模拟次数（根的 24 个候选 ≈ top-8×3 深度） |
| max_depth | 3 | 决策帧深度（一次完整交锋 ≈ 1.5s） |
| leaf_horizon_s | 8.0 | 叶确定性推演秒数（≈ threat_calc 语义） |
| prior_top_k | 8 | 策略先验保留的候选 bundle 数/帧 |
| c_uct | 1.4 | UCT 探索系数 |

预算账（AGENTS 实测引擎 18.9k battle-steps/s）：
单次模拟 ≈ 深度 3 × 30 battle steps + 叶 8s × 60 = 480+240 ≈ 720 battle-steps ≈ 38ms；
24 次 ≈ 0.9s/决策帧。一局 360 决策帧 ≈ 5.5 分钟/局 —— **评估可接受，训练不可用**（符合定位）。

## 对手口径（v1 明确边界）

- 对手 bundle 由回调 `opponent_fn(battle, player_id) -> ActionBundle` 生成；
  评估路径传冻结副本策略（deterministic），传 None = 对手不再部署（下界）。
- v1 不做信念采样（POMCP 留待 v2）；根节点直接用 ProphetPlanner 全知对手手牌做
  「对手可选动作枚举」的掩码输入（当前工具链已全可见，无新增泄漏——本仓库观测本就无迷雾）。

## 与既有闸门的关系

搜索候选动作直接复用 `legal_cells`（含 8h 空砸门、8g 不裸下门、9i EV 闸门）——
搜索在"已合法"的动作集合上选优，不重新发明部署规则。

## 文件与接口

- `src/clasher_new/rl/mcts.py`：`RLMCTS(policy=None, opponent_fn=None, cfg=MCTSConfig())`，
  `search(battle, player_id, obs=None) -> (ActionBundle, info)`；内部 deepcopy 战场，不改原局。
- `src/clasher_new/rl/selftest.py`：`test_mcts_*`——(1) 空场 Giant 单步搜索不崩且落点合法；
  (2) 明显交换（满血塔 vs 残血塔 finisher）方向正确；(3) 确定性（同种子同结果）；
  (4) 预算控制（n_simulations=2 快速路径不超时）。

## 评估接入

`rl/evaluate.py` 增加 `--mcts` 开关：同一 checkpoint 打两组 n 局（MCTS vs 无搜索），
对手为同一冻结脚本/模型，输出胜率差 + 平均决策耗时（`mcts_info['elapsed_s']`）。
结果写 `docs/` 下评估报告，回答「搜索收益是否为正」。
