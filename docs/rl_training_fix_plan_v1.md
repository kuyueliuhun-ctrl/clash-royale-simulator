# RL 训练整改计划 v1（2026-09-11）

依据 `docs/training_audit_2026-09-11.md` 的四问审计结论制定。
**总原则：先让策略真的在学（打通梯度/价值通道）→ 再让指标可信（门禁+锚点）→ 最后才堆训练量。**
顺序不可颠倒：在"策略每步位移测不出"的状态下堆步数，等于把算力倒进黑洞。

---

## 0. 阶段与门禁总览

| 阶段 | 目标 | 关键交付 | 入口条件 | 出口条件（Gate） |
|---|---|---|---|---|
| **P0**（1–2 天） | 打通策略学习通道 + 指标可信化 | `ppo.py` 价值通道修复 + 梯度成分诊断；评估门禁/去重/并行默认 | 无 | **G1**：A/B 显示 value 通道恢复 |
| **P1**（3–5 天） | 修结构性机制 | 热启动 PFSP 池、组波奖励、锚点基准集、换边、STOP 门控 | G1 | **G2**：300k 步行为+锚点双改善 |
| **P2**（按需） | 工程与整洁 | 崩溃自愈、临时文件归档、观测降维 spike | G2 | — |
| **上量** | 300k → 500k–1M | — | G1 / G2 | **G3**：1M 仍平台则转 IL+MCTS |

> 全计划约束：**`rl/ppo.py` 被 `run_league` / `flow_league` / `train_follower` / `train_prophet`
> 共同依赖，所有新能力必须"默认参数 = 旧行为"，通过 config 显式开启**，否则会一次性污染
> 所有训练入口。

---

## P0 — 打通"策略真正在学"的通道

### P0-1 价值通道修复 + 梯度成分诊断（最高优先）

**问题**：`gnorm` 中位 6,803 vs `max_grad_norm=0.5`（`rl/ppo.py:26`）；
`value_loss` 中位 39–57（RMSE≈6–7）而单步奖励量级仅 3；`clip_frac` 恒 0.0%、
`ratio` p50 = 1.000 → **策略每步位移测不出，更新方向被价值拟合主导**。

**改动点**：`rl/ppo.py`（+ `rl/config.py` 开关）

1. **梯度成分诊断（先做，用于证伪）**——新增 stats 字段 `p_gnorm` / `v_gnorm`：
   对 `p_loss`、`vf_coef*v_loss` 分别 `torch.autograd.grad(..., retain_graph=True)`
   取范数，再走原有的合并 backward。为控制开销，**按 `diagnose_every=10` 抽样**
   （网络仅 7.13ms/步，即使每次都做也只 +2 次 backward）。
   → 这一步本身就产出结论：`v_gnorm/p_gnorm` 若远大于 1，"评论家主导"被直接证实。
2. **回报归一化（核心修法）**——`ReturnScaler`（Welford running mean/var）：
   - 训练目标 `targets = (returns − mean) / (std + 1e-6)`；
   - critic 前向在归一化域，**bootstrap 需反归一化**：`main.value()` 的返回值
     经 `trainer.denormalize_value()`（`train_solo.py:1015` 的唯一调用点）；
   - scaler 统计量随 checkpoint 落盘（`solo_state.json` 或 ckpt 的 `meta`），
     保证续训不重置。
3. **备选/叠加开关**（各自独立可开）：`vf_coef` 0.5 → 0.25；critic 单独参数组给
   更高 lr；`adv_norm` 默认 `batch` → `scale`（`rl/config.py:104`，本项目自己的注释
   已指出 batch 中心化把躺平零优势帧抬成伪正优势）。
4. 新增 config 字段：`value_norm: str = "none"`、`diagnose_every: int = 10`。

**验收（G1）**——同 ckpt 起点、同 seed、同 20k 步、单变量 A/B（见 P0-3）：

| 指标 | 现状基线 | 通过阈值 |
|---|---|---|
| `value_loss` 中位 | ~48 | **< 10** |
| `ratio` 分布 | 恒定 1.000（min 0.992 / max 1.006） | **std > 0.01**（策略真的在动） |
| `clip_frac` | 0.0% | **> 0**（偶发出现即可，不要求高） |
| `v_gnorm/p_gnorm` | 待诊断 | **< 3** |
| 行为指标（接敌率/幽灵动作率） | 10.5% / ~20% | 不退化 |
| 胜率 | — | **不作判据**（16 局 2σ≈±0.25，噪声内） |

**成本**：实现 ~半天；A/B 两臂 2×20k 步 ≈ 40–60 分钟（CUDA/mp=4）。
**回滚**：`value_norm="none"` + `vf_coef=0.5` + `adv_norm="batch"` = 逐位回旧行为。
**风险**：改值函数量纲必须与奖励同锚——AGENTS 已记录"塔伤项 ×1000 导致空砸塔被误判正 EV"
的教训。**任何尺度改动都必须配一条对账 selftest**（见"回归测试清单"）。

### P0-2 评估口径可信化

| 子项 | 改动点 | 做法 | 验收 |
|---|---|---|---|
| 历史去重 | `train_solo.py:867` | `history.append` 前按 `step` 去重（同 step 覆盖） | 多次启动不再出现重复点（10e 有两个 `step:0`） |
| **门禁化** | `train_solo.py` 新增 `_check_gates(stats)` | 写 `runs/<name>/gates.json` + 控制台 `[gate] WARN`；"**先只报警不阻断**" | 接敌率/幽灵动作率越界时报警可见 |
| 门禁阈值 | `rl/config.py` 新增 `gates: dict` | 初值 `engagement_rate >= 0.095`（9k_ft 实测 10.5% 留余量）、`ghost_rate <= 0.10`、`winrate_se` 存在、`mean_reward` 有限 | 阈值可配、可关 |
| 对照组行为指标 | `train_solo.py:841-846` | 对照组改 `record_replays=True, save_replays=False`（只算不落盘） | `_controls_history` 带行为指标 |
| 并行默认 | `rl/config.py:122` | `eval_workers: 0 → min(16, cpu_count())` | eval@0 从 ~126s（串行）降到 ~15–25s |
| 评估局数 | `rl/config.py:84` | economy 预设 `n_eval_games 16 → 40`（2σ 从 ±0.25 → ±0.16） | 用 `steps_per_eval 4000→8000` 抵消墙钟 |

### P0-3 单变量 A/B 实验协议（纪律，不是代码）

- **固定**：同 ckpt 起点（建议 `runs/economy_9k_ft/solo_main.pt` 或 9j 100k）、同 `seed`、
  同 `deck_set`、同 `total_steps=20000`、同对手池配置。
- **单变量**：一次只改一项；P0-1 的四条子项若同时改，需先跑"只改诊断"的 arm 0 建立基线。
- **判读口径**：主判据 = **机制指标**（`value_loss` / `ratio` 分布 / `v_gnorm:p_gnorm`），
  次判据 = **行为指标**（接敌率 / 幽灵动作率 / 组波率 / 满费占比）；
  **胜率在本分辨率下不作判据**。
- **产物**：`runs/ab_<tag>/`，`dashboard --sweep` 并列对比曲线。

---

## P1 — 修结构性机制

### P1-1 热启动 run 的对手池退化（低成本，先做）

**问题（已复核）**：`runs/economy_9k_ft` 日志第 2 行实测
`无历史 ckpt（本目录首轮训练），退化为 frozen=0.875 / defend=0.125`
—— 设计上的 `70/20/10`（`train_solo.py:86 _OPP_MIX`）在热启动 run 里退化成 87.5/12.5，
PFSP 历史槽为空。而热启动恰恰是当前推荐做法（hint 需热启动才吸收）。

- **改动**：`_collect_hist_ckpts(folder, max_n)`（`:110`）→ 支持多目录；
  `_OpponentPool`（`:153`）接受 `hist_seed_dirs`；新增 CLI `--hist-seed-dir`（可多次）；
  `TrainConfig.hist_seed_dirs: list = None`。
- **验收**：热启动 run 日志第 2 行显示 `frozen=0.7/hist=0.2/defend=0.1`；
  `selftest.test_opponent_pool_mix` 增加多目录用例。
- **成本**：半天。

### P1-2 组波/攒费奖励（凹形定价）

**问题**：`elixir_diff` 是 potential-style shaping（`env_wrapper.py:245-253`），
下牌瞬间 `E−c`/`V+c` 同帧抵消 ⇒ **攒费零奖励**；实测 `bundle` 均值 0.05–0.11（≈94% 纯 STOP）。

- **前置（必须先做）**：写"量纲一致性对账"selftest——把 `rl/config.py` 奖励的圣水汇率
  （前段 1 费 ≈ 500 血、双倍期 ≈ 50 血）、`rl/action_mask._spell_tower_ev_gate` 的
  `edw×卡费` 判据、`rl/mcts.py` 值函数的汇率**放在同一处常量源**并对账断言一致。
  AGENTS 已记录两次量纲失配事故，这是硬前置。
- **方案 B1（优先，低风险）**：同一 `ActionBundle` 内 ≥2 张且总费 ≥6 → 小额 bonus
  （建议 **+0.5**，与"1 圣水 ≈ 500 塔血"同锚）。仅训练奖励，**不进评估**。
- **方案 B2（进阶，可选）**：凹形 elixir 定价（存 6–10 费时每费 +δ；泄到 <2 费时撤回 δ）。
- **验收**：组波率（bundle 均值）> 1.5 张/组合帧、3s 内连发占比 > 5%；
  且 `tower_dmg` 累计收益不下降、胜率不退化。
- **成本**：1–2 天（含对账测试）。**风险最高项，必须有 config 开关可一键回滚。**

### P1-3 固定锚点基准集 + 换边

**(a) 固定锚点基准集（高性价比，无风险）**
- 每轮评估固定对三方各跑 N 局，落 `runs/<name>/anchor_state.json`：
  ① 固定 seed 的随机初始化 `FollowerPolicy`（衡量绝对下限与"是否学会")；
  ② `SelfDefenderPolicy` + 四卡组（跨 run 恒定，可跨版本比）；
  ③ 指定历史 ckpt 目录中步数最大的一个（衡量能力保留/遗忘）。
- **成本**：3×40 局 ≈ 2 分钟/轮。**收益**：10d 式"对手隐形导致曲线失真"事件可被立即报警。

**(b) 换边（先 spike，再排期）**
- 现状：`eval_solo` 中 main 恒为 P0，无换边（`_pair_seed_offset` 只存在于 `run_league.py:322`）；
  若成立则先手/地图偏置会混进 0.5 零点，与 ±0.25 噪声叠加。
- **利好**：掩码层已支持 `get_action_mask_for(player_id)`（`env_wrapper.py:421`）、
  坐标换算已有 `to_position(player_id=1)`。
- **阻碍**：`RLEnv.step` 的提交路径硬编码 `deploy_card(0, ...)`（`env_wrapper.py:630`）。
- **行动**：**先做 30 分钟 spike** 判定工程量（参数化 env `player_id` vs 外层 side-swap wrapper）。
  可行 → 排期实现 + `selftest` 断言正反手同种子可复现；
  **不可行 → 降级方案**：种子流按 side 分离并在报告里显式标注该偏差，不假装已消除。

### P1-4 STOP 偏置门控

**问题**：`follower.py:119,160-162` 的 `stop_logit_bias=-1.0` 是**状态无关常量**，
无条件加在 STOP logit 上 ⇒ 与"攒费"需求直接冲突（高费时也推着出牌）。

- **改动**：按当前圣水水平线性调制偏置（高费时减弱/归零，低费时保持），
  圣水可从 obs 取得；**加载旧 ckpt 时行为不变**（保持既有兼容语义）。
- **验收**：满费时间占比从 **0.1%** 上升、圣水均值从 **2.04** 上升；
  同时 `deploy%` 不得跌破 4%（防止塌成"永不出牌"）。
- **成本**：1 天。**风险**：中（可能退化成憋牌），必须 A/B。

---

## P2 — 工程与整洁（按需）

| 项 | 改动点 | 说明 |
|---|---|---|
| 早停 `ep_term` 显式标记 | `train_solo.py:963-965` | **非 bug**（已复核：该帧已补完整终端结算，`next_val=0` 语义正确），只是隐式依赖 `compute_gae` 在 `T-1` 必然截断的写法，加 `ep_term[-1]=True` 作防御，零行为变更 |
| 崩溃自愈 | 新增 wrapper 脚本 | 9j 在 90k 被显卡驱动静默带走（无 traceback）→ 检测日志静默退出 + 自动 `--resume` 重启；确保每个 run 默认带 `--resume` |
| 临时文件归档 | `src/clasher_new/tmp_*.py`（4 个）、根目录 `0631.pdf` / `libg_strings.txt` / `re_lib.py` / `:memory:.ses` | 清理或移入 `archive/`，减少误读 |
| 观测降维 spike | `rl/observation.py`、`rl/env_wrapper.py` | grid 32KB/条是 NN 前向主因；但 NN 只占 10% 吞吐 → **优先级低，等 G2 后再看** |

---

## 3. 训练量阶梯与 Gate 判据

| Gate | 判据 | 通过后做什么 |
|---|---|---|
| **G1** | P0-1 的 A/B 达标（`value_loss`<10、`ratio` 分布离开 1.000、`v/p`<3）+ 行为指标不退化 | 允许跑 **300k 步**（≈4–10h，mp=4/CUDA） |
| **G2** | 300k 时：①行为指标（接敌率/组波率）改善 **≥2σ**；②锚点基准集胜率改善 **≥2σ** | 允许上 **500k–1M** |
| **G3** | 1M 时锚点胜率仍平台 | **停止堆纯 PPO 步数**，转 IL/BC 底座 + 推理时浅 MCTS |

**为什么不是直接冲 1M**：现状胜率在 100k 已平台，而行为指标仍在爬坡；在 G1 未通过前
堆量只会在"策略位移测不出"的状态下空转。

---

## 4. 明确不做（防范围蔓延）

1. **不扩模型参数**——实测 629,359，比 Atari DQN(1.7M) 还小，但容量分布病态
   （`enc_fc` 单层占 55.6%、GRU 仅 128、entity_emb 仅 8 维），且吞吐被引擎主导
   （NN 仅占 10%）。扩容既提不了步速也不解决瓶颈。**除非 G2 后"锚点胜率不动
   且行为指标也不动"**，才考虑表征改造（实体注意力 / GRU 128→256）。
2. **不上训练时 MCTS / 专家迭代**——AGENTS 已实测 ≈ **120× 每帧经验成本**，当前量级烧不起。
3. **不引入 LLM 实时决策**——决策粒度亚秒级，LLM 延迟进不了 tick 循环。
4. **不重构动作语义**——`K_MAX=4` 保持不变，只改偏置（P1-4）与奖励（P1-2）；
   把 plan 从特征层改成动作层是动作语义重造 + 旧 ckpt 兼容问题，工程量不可控。

---

## 5. 资源与时间估算（假设：CUDA + `--n-envs 4 --parallel mp`）

| 阶段 | 实现 | 实验/训练 | 备注 |
|---|---|---|---|
| P0-1 | ~0.5 天 | A/B 2×20k 步 ≈ 40–60 min | 参考：9k 的 20k 步 CPU 单 env = 1.04h |
| P0-2 | ~0.5 天 | — | 门禁/去重/默认值，基本零风险 |
| P0-3 | 纪律，无实现 | — | — |
| P1-1 | ~0.5 天 | 短验证 run | 最低成本的结构性修复 |
| P1-2 | 1–2 天 | A/B 2×20k 步 | 含量纲对账测试 |
| P1-3 | 锚点 0.5 天；换边 spike 0.5h + 视结果排期 | — | 锚点每轮 +2 min |
| P1-4 | 1 天 | A/B 2×20k 步 | — |
| **300k 步** | — | **≈4–10 h** | 中性吞吐 20.8 步/s（mp=4）→ 4h；单 env 8 步/s → 10.4h |

---

## 6. 回归测试清单（新增，遵循本仓库 selftest 文化）

所有新增项必须配 selftest，且**必须能跨过局边界**（AGENTS 教训：只查落盘不查统计量的
冒烟测试是盲的，9j 的 `nonlocal` 泄漏事故就是这么漏过去的）。

1. `test_return_scaler_roundtrip`：归一化 → 反归一化可逆；scaler 随 ckpt 存取一致。
2. `test_gnorm_split_diagnostic`：构造一个"评论家误差 >> 策略误差"的合成批，
   断言 `v_gnorm/p_gnorm` 显著 >1（诊断本身可证伪）。
3. `test_value_norm_off_is_bitwise_old`：`value_norm="none"` 时与旧实现逐位一致。
4. `test_history_dedup_by_step`：同 step 重复评估只留一条。
5. `test_gate_warn_not_block`：越界只报警不中断训练。
6. `test_reward_dimension_consistency`：奖励汇率 == 闸门汇率 == MCTS 值函数汇率（同一常量源）。
7. `test_opponent_pool_mix_multi_dir`：多目录历史池仍满足 70/20/10。
8. （换边可行时）`test_eval_side_swap_same_seed`：正反手同种子统计可复现。

---

## 7. 风险登记册

| 风险 | 触发条件 | 缓解 |
|---|---|---|
| 改值/奖励尺度引发量纲失配 | P0-1 / P1-2 | 每项配对账 selftest（清单 #2/#6）；config 开关可回滚 |
| 改 `ppo.py` 污染其他训练入口 | 所有 P0-1 改动 | 默认参数 = 旧行为；`test_value_norm_off_is_bitwise_old` 守门 |
| 换边工程量失控 | P1-3(b) | 先 0.5h spike；不可行则降级为"标注偏差"而非硬做 |
| 门禁阈值误杀 | P0-2 | 先只报警不阻断；阈值可配 |
| 憋牌退化（P1-4 反向） | stop 偏置门控过强 | `deploy%` 下线 4% 作硬监测 |
| 显卡驱动崩溃丢进度 | 长 run | P2 自愈 wrapper + 默认 `--resume` |

---

## 8. 建议的执行顺序（最小闭环）

```
P0-1 诊断（半天）
  → 拿到 v_gnorm:p_gnorm 真值，证实/证伪"评论家主导"
  → 开 value_norm（+ adv_norm=scale）跑 A/B
  → G1 达标 → 跑 300k
P0-2 同步进行（门禁/去重/并行默认，零风险）
P1-1 与 P0 并行（半小时级）
P1-2 / P1-4 在 G1 之后做（改奖励/偏置必须在值通道健康的前提下才可归因）
P1-3 锚点：随时可做，建议在 300k 之前落地（否则 300k 的结论没有可信尺子）
```
