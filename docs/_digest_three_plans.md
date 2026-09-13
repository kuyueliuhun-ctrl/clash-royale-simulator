# 三份计划整合 digest（2026-09-13 核查）

同链迭代：评审手册修 P0 → v1 打通价值通道/指标可信 → v2 按 20k 实测改判据。v1 的 ratio/value_loss 判据作废；v2 §4 分支被 v3 推翻。标记 [文]自述/[核]grep/[未核]。

## 1 定位与状态
- `rl_review_fix_plan.md`：评审手册（6 Critical／~20 Major／~40 Minor）。**已落地**，§6 的 11 测试名均在 selftest.py [核]；未被取代。
- `rl_training_fix_plan_v1.md`：四问审计整改（P0 通道→P1 机制→上量；G1/G2/G3）。**部分被 v2 取代**：P0-1/P0-2/P1-1 落地，P1-2/P1-3/P1-4 未做，G1 两判据作废。
- `rl_training_fix_plan_v2.md`：v1 修订版（5 处 20k 证伪项＋1 处结构缺陷，收口 200k）。**6 处全落地** [文][核]；§4 分支被 v3 明示"判断错误"。

## 2 v1 改动＋状态
1. **P0-1 价值通道**（value_norm/ReturnScaler/梯度诊断）：落地（economy=running）；**value_loss<10、ratio 判据作废** [核]
2. **P0-2 评估口径**：落地（相对门禁、n_eval_games=40、workers≤16）；绝对阈值作废 [文][核]
3. **P0-3 A/B 纪律**：由 ab_valnorm_20k 承载 [未核]
4. **P1-1 热启动对手池退化**：落地（hist_seed_dirs）[核]
5. **P1-2 组波/攒费奖励**：**未做**（无 bundle≥2 费≥6 bonus、无凹形定价；仅 elixir_bonus 线性 shaping）[核]；v2 §4 列为后续项
6. **P1-3a 固定锚点基准集**：**未按原文落地**（无 anchor_state.json）[核]；后以 rand_anchor 等替代
7. **P1-3b 换边**：run_league/flow_league 已换边；eval_solo 仍恒 P0、无换边测试 [核]
8. **P1-4 STOP 门控**：**未做**（stop_logit_bias 仍常量）[核]

## 3 v1→v2 修订（作者表）
| # | v1 | 依据 | v2 |
|---|---|---|---|
|1| ratio 离开 1.000 | ppo.py 结构性恒等（单轮 on-policy＋n_epochs=1） | 换 EV≥0.2 |
|2| value_loss<10 判 critic | 被 s² 除过、跨版本不可比；达标时 EV≈0 | 加 EV 诊断，value_loss 只作量纲回归 |
|3| 门禁绝对阈值 9.5 | 阈值来自一次性脚本；内建指标对同批 9k_ft 实测 28.3~38.1，差 3 倍 | 改相对本 run 首评估点 |
|4| 预算按 step | step=决策帧；20k≈156 更新≈55~80 局 | 改局数，落 cum_games |
|5| 对手池 0.7/0.2/0.1 | EV≈0 疑镜像对称致状态→胜负不可预测 | 0.5/0.3/0.2；后被 E2 改 0.1/0.6/0.2/0.1 [核] |
|6| 非计划内：copy_every 整除 steps_per_eval ⇒ 评估对手恒为刚同步的自己 | 20k 判读 §4 | 先评估后同步（train_solo.py:1629<:1638）[核] |

## 4 6 条 Critical 终态＋回归测试
作者自述全修 [文]；测试名均见 selftest.py [核]。1 P0-1 hidden 重放 → `test_hidden_replay_consistency`；2 P0-2 熵方向反 → `test_entropy_positive_and_sign`；3 P0-3 掩码指纹缺 player_id → `test_mask_validate_invariant_both_sides`；4 P0-4 掩码/提交坐标差镜像 → `test_heuristic_opponent_actually_plays`；5 P0-5 维度 20/32 vs 21/23 → `test_exploiter_loads_main_checkpoint`；6 P0-6 `"__ability__"` 哨兵 → `test_belief_survives_ability`。

## 5 红线/兼容/不要改回去
- 共享模块 ppo.py（4 入口共用）：新能力默认=旧行为、经 TrainConfig 显式开启；none＋vf_coef0.5＋adv_norm batch＝逐位回旧
- 尺度改动（奖励/值/闸门/MCTS 汇率）必须同常量源＋对账 selftest
- ratio≡1.000／clip_frac≡0 结构性，非"策略没动"（判据已删）；EV 是诊断非目标，不设 EV 奖励
- 勿按 step 高估进度；不判 main 曲线与单点胜率；行为指标只读 gates.json 相对退化
- 明确不做：不扩参数、不上训练时 MCTS/专家迭代、不引入 LLM 实时决策、不重构动作语义(K_MAX=4)；不为绕开外部异常改代码/降配置（宿主干扰→暂停等人类）
- 契约：坐标=本地坐标＋sub_position｜掩码内 used_slots 与 validate 同扣费｜opp_played 结构化＋技能独立字段＋哨兵不入 info｜维度单常量＋ckpt 元数据｜统一包相对导入｜超 K_MAX 强制 STOP＋校验拒绝｜legacy 统一 (slot,y,x)
