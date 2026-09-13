# v3 digest（供总计划）

源 `docs/rl_training_fix_plan_v3.md`(09-11~12)；据 `docs/value_channel_saturation_diagnosis_2026-09-11.md`；P1 编号出 `docs/rl_training_fix_plan_v1.md`。`rl/`=`src/clasher_new/rl/`，`runs/`=`src/clasher_new/runs/`。取数 09-13（只读文件）。

## 1 病因与推翻

- 负 EV 根因=**GRU 输入饱和**，非奖励尺度、非价值头：`enc=relu(enc_fc(fused))` ‖enc‖533（grid_feat 常分量 468/std 1.06）→ `n_abs`0.994 → h 跨帧 std 2.6e-5 → critic 恒常数，EV 吻合 `−bias²/Var(R)=−0.584`（实测 −0.5817）。
- 系统性：9k_ft/9j/10e/ab_valnorm_20k/prod_200k 的 `n(abs)`≥0.98；`slot/cell_head(h)` 亦常数 ⇒ 状态依赖只剩手工 `BeliefPlanner` 偏置。
- **推翻①**(09-11) v2 §4「EV≈0→奖励尺度/拆价值头」病因判错；**②**(09-11) v3 §1 P0-C `h_std>0.02` 笔误，统一 0.05（0.02 仅 dashboard 黄带）；**③**(09-12)「表征有信息(0.24~0.42)而 critic 吸收不了⇒优化预算」失效——逐帧留出系时间泄漏(`corr(R_t,R_t+1)≈0.99`)，按局分组 R² MLP −0.36/−0.45。
- 新缺口(09-12)：`enc→hp_diff` 分组 R² **−0.03**（同一 enc 读 `time` +0.9999）⇒ 决策通路无可靠塔血输入。

## 2 行动项状态

- **P0(v1)**：P0-1 价值通道(`value_norm=running`/`adv_norm=scale`/梯度诊断)已落地、**所依病因被推翻**；P0-2 评估口径(去重/`gates.json`/对照行为指标/`eval_workers`/40 局点)已落地、**绝对阈值被推翻**→相对首点 `{"rel":">=","frac":0.5}`、首点只建基线（9.5 作废：同权重实测 28.3~38.1，差 ~3×）；P0-3 单变量 A/B 纪律沿用。`rl/ppo.py`、`rl/config.py`、`rl/train_solo.py::_check_gates`。
- **v3 本体**：**P0-A 已落地** `enc_ln=nn.LayerNorm(hidden)`（旧 ckpt 不可续训，须 `--fresh`）→ `rl/follower.py` `enc_ln`/`_encode`/`_encode_batch`；备选 `grid_ln` 已落地(09-12)、GRU 门偏置**未做**。**P0-B 已落地**：EV 池化+逐帧 `(v,R)`(`rl/train_solo.py::eval_and_write`、`rl/ppo.py::last_ev_pairs`)+活力指标(`rl/diagnostics.py`、`rl/dashboard.py`)；P0-B-2 第 3 项 `value_std_ratio` **部分落地**（5k 轮分母 `r_std` 全仓未实现⇒不可判读，09-11 补齐；09-12 改同窗口，旧值存 `_probe`）。**P0-C 部分落地**：静态护栏(`enc_ln`/`grid_ln` 缺失或被 Identity 替换即报警，落盘 `stats["vitality_warns"]`；启动测不了 h_std/n_abs，活力在评估点测)→ `rl/diagnostics.py::check_policy_architecture`、`rl/selftest.py::test_enc_layernorm_gru_vitality`。随附已落地：GBK 告警二次抛错崩溃、旧 ckpt 缺键静默、`_probe` 20 万帧上界→`rl/run_league.py::_force_utf8_stdout`、`rl/diagnostics.print_safe`、`rl/follower.py::load_checkpoint`。
- **P1(v1)**：P1-1 多目录补种已落地（`--hist-seed-dir` 是恢复配比唯一途径，`opp_mix` 无 CLI flag）；P1-3a 固定随机锚点=E1(`RAND_ANCHOR_SEED=99999`/`RAND_ANCHOR_EVAL_SEED=90000`/`RAND_ANCHOR_WARN_FLOOR=0.35`)；P1-3b 换边、P1-2 组波/攒费凹形奖励、P1-4 STOP 门控**未做**（门控在 G1/EV 后）。
- **后续分支**：A′ 已做；E1 已落地；**E2 被证伪**；B′ `value_bypass` 已落地(economy 预设 True)**证据不支持收益**；C′ `stall_draw_margin` 已落地但训练侧几乎不触发（65 局早停 2、降级 1；28~40% 是 eval 随机对手局）；E′ 独立价值编码器+MLP 头已落地、**20k 预测被证伪**；F′ 真 PPO(epochs4/minibatch32/shuffle)已落地保留、离线 EV 未过零；G′ 三层对照已做，**G′-fix（塔血/皇冠/圣水差进 scalar 3→~11 维）未做**。

## 3 门槛/协议/分支

- **验收表(5k 可判)**：`h 跨帧 std>0.05`；`GRU n(abs_mean)<0.9`；`value_head std/批内 R std>0.3`；`EV(池化)>0 且随步上升`。代码源 `rl/diagnostics.py::THRESHOLDS={"h_std":0.05,"n_abs":0.9,"value_std_ratio":0.3}`。修复前基线 2.6e-5/0.994/~0.001/−0.58。
- **口径**：主判据 EV=池化（窗口累积逐帧 `(v,R)` 合并算一次），对照 `explained_variance_batched`；`EVb`=批内(放大 ~3×，只看趋势)；F′ 后非旧分支取**更新前**整批 no_grad 前向，in-sample 另存 `EVin=`。v2：EV≥0.2→可救堆量；≈0→停堆量。次判据=对照曲线 vs `baseline0`/`baseline_prev` 净漂移 **≥2σ(≈0.16)**。40 局地板：胜率 1σ≈0.078、接敌率 1σ≈±5pp。行为只读 `gates.json` 相对退化；**不判** main 曲线与单点胜率。
- **协议**（`cd src/clasher_new`，均 `--fresh --value-norm running --adv-norm scale --diagnose-every 10`）：5k=`--config-name fix_gru_ln_5k --total-steps 5000 --steps-per-eval 2500`；20k=`--config-name fix_gru_ln_20k --total-steps 20000 --steps-per-eval 2500 --n-eval-games 40 --eval-workers 16 --hist-seed-dir runs/economy_9k_ft --hist-seed-dir runs/economy_9j`；E2 同 20k+`--config-name e2_rand_anchor_20k --eval-workers 12`；F′ 同 E′+`--ppo-epochs 4 --ppo-minibatch 32 --ppo-shuffle`。`--fresh` 挡不住 `--main-init`，须显式不传。
- **分支**：全达标→长跑 100k~200k；h 解冻但 EV≤0 且不再上升→才走 v2 §4；h 仍冻结或 `enc_fc` 重被拉大→上 P0-A 备选。

## 4 不要做

- 不因 EV 负调 `vf_coef`/奖励权重；修复前不继续 200k（prod_200k 停于 ~54k，留作饱和态基线）。
- 不用 `main_init` 续训 v3（旧 ckpt 无 `enc_ln`/`grid_ln`，静默陷阱）；架构改动一律 `--fresh`。
- 5k/20k 不执行 v2 §4 高侵入项（拆价值头/改奖励/P1-4）；不在 A′ 取证前改奖励或拆头。
- 不把 `main vs 冻结副本 0.85` 当变强（自引用失真）；不因 EV≈0 宣称修好；cycling 未解时堆量(D′)/价值头结构(C′)无效。
- 不扩模型参数、不上训练时 MCTS/专家迭代、不引入 LLM 实时决策、不重构动作语义(`K_MAX=4`)；不为绕外部性能异常改代码/降配置。
- `rl/ppo.py` 被 4 入口共用 ⇒ 默认参数必须=旧行为。

## 5 验证跑结论+证据

| 跑 | 结论一句话 | 日志 |
|---|---|---|
| 5k `fix_gru_ln_5k` | 修复生效(n_abs 0.994→0.46/0.55、h_std 升 3~4 数量级)，EV 单调 −0.58→−0.06 未过零、判不了分支；行为退化由对手池偏离解释 | `docs/train_fix_gru_ln_5k.log` |
| 20k `fix_gru_ln_20k` | 饱和修复稳定(n_abs 8/8<0.9)但 critic 仍常数(ratio 0/8、EV 0/8 过零)；融合层失衡 grid 占 fused 98% | `docs/train_fix_gru_ln_20k.log` |
| grid_ln 20k `fix_gru_ln_norm_20k` | 尺度修复达成(grid 101→5.87)只修对中(EV −0.134→−0.005)，价值头仍未拟合；n_abs 回升 0.745；`vs baseline0` 崩塌 0.05~0.125=cycling 红旗 | `docs/train_fix_gru_ln_norm_20k.log` |
| A′ 取证 | cycling 确凿(RPS 三角：随机>baseline0>main≈随机)，绝对强度≈随机(vs 全新随机 0.505)，早停裁定是主要混淆 | `scripts/_forensics_cycling.py`(只读 `runs/fix_gru_ln_norm_20k/`，v3 §3.8.4) |
| E1 固定锚点 | 机制确定生效(两次 0.15/0.15 逐位一致)，0.15<0.35 ⇒ 每评估点都会报警 | v3 §3.8.5；`rl/selftest.py::test_solo_rand_anchor` |
| E2 `e2_rand_anchor_20k` | 干预无效、cycling 未破：`vs baseline_rand` 末点 0.050(<0.35)、`vs baseline0` 0.075，critic 仍常数；10% 弱锚点压不住漂移 | `docs/train_e2_rand_anchor_20k.log` |
| E′ `eind_20k` | 预测被证伪：EV 0/8 过零(最好 −0.004，与 bypass/grid_ln 同级)，独立编码器+MLP 头仍常数 | `docs/train_eind_20k.log` |
| F′ `fprime_20k` | 真 PPO 保留：训练期 EV 首 >0 属 in-sample 假象；离线 EV −0.004/−0.026 未过零、v std 10×；查出 3 个测量 bug | `docs/train_fprime_20k.log` |
| G′ 三层对照 | `raw→outcome` 跨局≈掷硬币(+0.06，地板 −0.85)⇒critic 侧该停；`enc→hp_diff` −0.03=可修观测缺口 | `docs/diag_predict_fprime2.log` |

## 6 未核实

归一化后 critic 实际上限（留出探针 10 局全负，不得宣称）；`grid_feat` 468 来源；量级反转致行为短期退化（预期未验证）；局均 `corr(v,R)` 24 局 1σ≈0.21~0.22、E′ 符号翻转不显著；E2 与对照时长差含 `eval-workers 12 vs 16`(`docs/train_speed_benchmark_2026-09-12.md`)；`byp_cprime_20k` 日志 `vstd/rstd` 列无效(旧诊断模块)；未核实 09-13 后 d1/gfix/fprime_ev 等 run 的改写。
