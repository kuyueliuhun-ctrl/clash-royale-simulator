# 其他优秀项目（使用了 IL）的奖惩机制 —— 对我们的参考

> **合成者**：本次调研的编排/合成者（**7 份子报告** + 本合成本轮**补取 6 篇一手文献**闭合 U-04~U-08，见 §7.4）
> **落盘时间**：2026-09-19（CST）；**第二轮闭合时间**：2026-09-19（CST）
> **取数时间**：子报告统一 **2026-09-19 03:33Z**（= CST 11:33，WSL 只读）；**本轮补取 = 2026-09-19 CST**（arXiv API + ar5iv 直连 `curl`，逐篇见 §7.4）；本文合成的仓内复核单独标注
> **对象**
> - **我们** = `E:\clash-royale-simulator-main`（WSL `/mnt/e/clash-royale-simulator-main`）。下文路径以 `src/clasher_new/` 为前缀根，简写 `rl/xxx.py`；`src/clasher_new/` 下**没有** `rl/` 之外的并列目录。
> - **FL** = `E:\FirstLight_CR`（`/mnt/e/FirstLight_CR`），HEAD `28d66cc0a5d65888515e22fdf22f11d783b65efb`（commit date 2026-09-06 22:17:43 -0400）。
> - 其余外部项目：AlphaStar / 绝悟 JueWu / RLHF(InstructGPT·HH·TRL·DS-Chat) / 离线 IL-RL 家族(AWAC·DT·IQL·TD3+BC·RLPD·Cal-QL·robomimic·DAPG) / OpenAI Five。
>
> **纪律与禁则（本文自缚，逐条遵守）**
> ① 无 URL 或 `file:line` 支撑的陈述一律标 **`未验证`**，不写成事实；
> ② 描述性观察不写成判据（**【R3】**）；
> ③ **禁止**用外部项目数字与我们的数字对拍（量纲 / step 语义 / 参数规模不可比；**【R15】**）；
> ④ **禁止**写「修好了 / 确认无效」这类超证据措辞；
> ⑤ 口径冲突**并列不调和**（**【R17】**）。
>
> **本文的证据分级**（沿用子报告约定）：`[一手]` 论文/代码原文逐字；`[复现件]` 第三方复现论文；`[二手]` 分析/issue；`[未验证]` 未取到；`[本地]` 本仓 `file:line` 复核。

---

## §0 一页结论（含第一问答复）

**我们已确证的病理**（[本地] `docs/agents/ledger.md:28` **C14**、`:62` **O7**、`:48` **X-19**、`docs/s1_gate_2026-09-18.md`）：
「攒费 → 打 Xbow」这条轨迹**从未被采样**（`0/7,238` 段主动不出牌发生在圣水 ≥4；A 段最长 **22 帧 < 34 帧**）⇒ **0 样本 ⇒ 0 梯度**；奖励侧手段**已完整取证并判不接线**（X-19：换成奖励真正会用的口径后 Δρ(塔血) `+0.145 → +0.008`、Δρ(胜) `+0.108 → +0.014`，第五轮补尾部 flush 后归零且方向不稳；O7：机制自锁，Xbow 不在 `ACE_CARDS`/`SINK_TANK_CARDS` ⇒ `hold_mask` 在 Xbow 卡组 0 帧触发）。

**第一问答复（明确）**：IL 家族**确实能提供不碰奖励函数的手段**，但**只有 3 条**能在我们仓内落地并自洽；其余 4 条会被「0 样本」这一前件反噬。逐条如下（详见 §3.1）：

| # | 机制名 | 落点（我们的文件） | 能否逐位/统计口径验证 | 成本 | 判定 |
|---|---|---|---|---|---|
| I1 | **BC/NLL 示范损失本身**（IL 阶段结构性没有奖励） | `rl/train_bc.py:107-111`（`loss=-lp`）、`rl/human_play.py:205-233` | ✅ **能逐位**：单帧 NLL 可脚本复算；`p=0` 时逐位回旧行为可对账 | 极低（代码已存在） | **可做（作为先决条件与仪器），但不针对病根** |
| I2 | **时序性（stateful / TBPTT）** —— 让 BC 学得到「跨帧不花钱」 | `rl/train_bc.py:107`（`hidden=None` 每帧归零）+ `rl/follower.py:753-806`（`evaluate`） | ✅ **能逐位**：同帧 hidden 输入 A/B，hidden 归零 vs 携带；梯度链长度可断言 | 低（改 `evaluate` 调用契约 + TBPTT 循环） | **可做（最高性价比之一，须预注册）** |
| I3 | **动作掩码不变式（采样即校验）** —— 把「非法」变成接口契约而非奖励惩罚 | `rl/follower.py:433/529`（`act`/`act_parallel`） | ✅ **能逐位**：`validate_bundle` 逐包断言 + 【R13】128 张位图对账 | 低（O9 的修复已完成一半） | **可做（先决条件，与奖励无关）** |
| I4 | **数据/缓存侧质量过滤** | `rl/export_replay.py`（只产 `EpisodeReplay`）、`rl/replay.py` | ✅ 能统计，但**过滤是减法**（只丢样本） | 低 | **不做**（补不了 0 样本：减法不能造样本） |
| I5 | **样本加权 / 优势加权（AWAC·AWR·IQL）** | `rl/train_bc.py:107-111` + `rl/follower.py:765`（`value`） | ✅ 逐位对账容易（权重张量可比） | 中 | **待预注册**（⚠️ **优势在零样本区也是零** ⇒ 学不到新轨迹；仅对缓解「便宜牌 95.47% 主导」可能有量级作用） |
| I6 | **return-conditioning（DT / RvS）** —— 以「目标回报」为条件生成动作 | `rl/train_bc.py:107`（需把 return 注入 `plan_vec` 或新输入头） | ✅ 逐位可对账（输入向量可比）；性能判据 ❌ n=1 不可判 | 中高 | **待预注册**（⚠️ 我们**无人类数据、无高质量轨迹**；`未验证`） |
| I7 | **DAgger / 交互式纠正（含 expert iteration）** | `rl/follower.py:433`（`act`）、`rl/belief_planner.py`（规则专家） | ✅ 逐位（纠正帧的 `(obs,bundle,masks)` 可复算） | 高（需在线专家 + 重采样预算） | **待预注册**（⚠️ 若专家本身不产「攒费」帧，迭代只是放大同一局部最优；【R11】不含此禁令，但触碰成本） |

**一句话结论**：IL 家族给我们的**不是新奖励**，而是 **(a) 一条不碰奖励函数就能注入「时序/长时域」能力的通路（I2）**、**(b) 把「非法动作」从奖励里拿出来的结构方案（I3，IL 项目的一致性做法）**、**(c) 一个可以把「罕见局面」变成可训练信号的接口（I6/I5，但都**依赖先能采到那种局面**）**。**没有任何 IL 手段能绕过 C14**——因为 IL 的数据侧与奖励侧共用同一个采样器，采样器到不了的地方，两条路都到不了。真正对准病根的仍是 **O7（机制侧）**，IL 手段的价值是**在 O7 之后把新轨迹变成可学的参数**。

---

## §1 概念澄清：「IL 项目的奖惩机制」到底指什么

「IL 项目有奖惩机制」这句话本身是**含糊的**。逐字核查后，它至少落在 **5 个互不相同的层**。混淆层与层是本次调研里最容易出错的地方，故先分层，每层给 ≥2 个实例 + URL。

### ① IL 本身**没有奖励**，只有示范损失（BC / NLL / CE）——「奖惩」在 IL 阶段结构性缺席

| 实例 | 事实 | 出处 |
|---|---|---|
| **FirstLight_CR** | IL 唯一损失 = **6 个掩码交叉熵/回归项之和**（`gate` + `candidate` + `target` + `delay` + `continue` + `value`），权重默认全 `1.0`；`penalty` 在 12 个 IL 文件里 **0 命中** | `[本地]` `/mnt/e/FirstLight_CR/native_runner/training/v4/imitation.py:283-290`、`:145-152` |
| **AlphaStar** | IL 阶段 = 6 个 action 分量的交叉熵之和 + grad clip 0.5 + Adam；**监督动作，不是监督 z** | `[复现件]` mini-AlphaStar <https://ar5iv.labs.arxiv.org/html/2104.06890> §3.4 |
| **绝悟 JueWu-SL** | 目标 = **双层动作标签 + 多视图意图标签的四项交叉熵加权和**（四个权重都设 1，λ=1，**无 KL、无熵项**） | `[一手]` arXiv:2011.12582（见 `docs/juewu_research_2026-09-19.md` §3.2） |
| **InstructGPT（SFT 段）** | **论文 v1 没有给出 SFT 损失方程**，只有散文「using supervised learning」；IL 损失 = 对示范的最大似然（token 级 CE）的可靠出处是 Stiennon et al. 2020 §1 | `[一手]` <https://arxiv.org/html/2203.02155v1> §3.5；<https://ar5iv.labs.arxiv.org/html/2009.01325> §1 |
| **Decision Transformer** | 训练 = 纯监督动作预测（离散 CE / 连续 MSE），**没有奖励项、没有 critic、没有 bootstrap** | `[一手]` arXiv:2106.01345 <https://arxiv.org/e-print/2106.01345> |

> **对我们**：我们的 `rl/train_bc.py:107-111` 也正是这一层——`loss = -lp`，无 entropy、无 value loss、无 KL（`[本地]`；`rl/ppo.py:393` 的 `p_loss + vf_coef*v_loss − coef*ent_term` 是 PPO 的，不是 BC 的）。**所以「IL 阶段有哪些奖惩可借鉴」这个问题的答案在①层是：没有。**

### ② 隐式奖惩落在**数据侧**：示范质量过滤、样本加权、DAgger/交互式纠正、专家混合（expert iteration）

| 实例 | 机制 | 出处 |
|---|---|---|
| **FirstLight_CR** | **预检拒绝规则**（`accepted` / `rejection_reason`：`no_expert_actions` / `missing_or_mixed_data_i` / `action_before_first_decision` / `more_than_two_actions_in_window`）；**shadow 合法性校验**；**旧缓存非法标签清洗**。三者都是「**丢样本**」而非「扣分」 | `[本地]` `.../v4/dataset.py:22-41`、`:44-90`；`expert.py:299-326`；`cache.py:87-124` |
| **FirstLight_CR（PPO 阶段）** | **按胜负加权采样**：`weight = 1.0 if won else 0.25` → `rng.choices(..., weights=[r["weight"]])` | `[本地]` `.../tools/experiments/build_hog_expert_manifest.py:79`；`.../v4/ppo_expert_bc.py:343-345` |
| **绝悟 P2** | **Scene Identification**（按轨迹分段：推塔/团战/清线/打野）+ **欠采样** + **attack sample normalization**；预处理后**只留 1/20 帧** | `[一手]` arXiv:2011.12582（见 `docs/juewu_research_2026-09-19.md` §7.1 A3） |
| **LIMA** | **数据质量 > 数据数量**：1000 条精选 + 不做 RLHF，65% 胜过做过 RLHF 的 DaVinci003；`filtered vs unfiltered` 差 **0.5 分**；**数据翻倍不改善** | `[一手]` <https://arxiv.org/abs/2305.11206>；<https://arxiv.org/html/2305.11206v1> |
| **DAgger** | 交互式纠正：用当前策略 rollout + 专家标注，缓解 BC 的 `O(T²ε)` 误差累积 | `[二手]`（AlphaStar 相关分析：<https://www.alexirpan.com/2019/02/22/alphastar-part2.html> §1）；原始 DAgger 参考文献本轮**未逐字核** |

> **对我们**：`rl/export_replay.py` 产出的 `EpisodeReplay` **不含 masks/plan_vec**（`[本地]` `rl/export_replay.py:1-5` 与 `:44-52` 的 `ActionBundle` 只记 bundle），**不足以直接喂 BC**；能产 BC 样本的只有 `rl/human_play.py`（需要人类或外部驱动，`[本地]` `rl/human_play.py:1-20`）。我们**没有**数据侧过滤/加权/纠正的任何实现（`[本地]` `grep -naE "behavior_clone|imitation|demos|teacher" rl/ scripts/` → **0 命中**）。

### ③ 把奖励**塞进 IL 损失**：advantage-weighted regression（AWAC/AWR）/ return-conditioning（Decision Transformer）/ BC 辅助损失 / 优势加权

> **关键说明**：这类使「奖励」以**样本权重**或**输入条件**的形式出现，**不是环境奖励**。

| 实例 | 精确形式 | 出处 |
|---|---|---|
| **AWAC** | KKT 闭式解 `π*(a\|s) = (1/Z(s))·π_β(a\|s)·exp((1/λ)A^{π_k}(s,a))`；投影到参数化策略（**前向 KL**）：`θ_{k+1} = argmax_θ E_{s,a~β}[ log π_θ(a\|s) · exp((1/λ)·A^{π_k}(s,a)) ]`；**λ=0.3（manipulation）/ 1.0（MuJoCo）**；**实践中丢弃 Z(s)**（消融：pen 84%→98%、door 0%→95%、relocate 0%→54%） | `[一手]` arXiv:2006.09359v6 <https://arxiv.org/e-print/2006.09359>；官方代码指 rlkit <https://awacrl.github.io/> |
| **AWAC 的「实现真相」** | 官方参考实现**默认不是 `exp(A/λ)`**：rlkit `normalize_over_batch` **默认 True** ⇒ `weights = F.softmax(score/beta, dim=0)`；jaxrl 的 AWR **只有** batch-softmax，并留注释「`exp(a/beta)` is unbiased but high variance, `softmax(a/beta)` is biased but lower variance」 | `[一手]` <https://github.com/vitchyr/rlkit/blob/master/rlkit/torch/sac/awac_trainer.py>（line 618–631）；<https://github.com/ikostrikov/jaxrl/blob/main/jaxrl/agents/awac/actor.py> |
| **IQL** | `L_π(φ) = E_{(s,a)~D}[ exp(β(Q_θ̂(s,a) − V_ψ(s))) · log π_φ(a\|s) ]`（advantage weighted regression）；`L_V` 用**非对称 expectile**；`τ=0.9/β=10.0`（AntMaze）、`τ=0.7/β=3.0`（locomotion） | `[一手]` arXiv:2110.06169 <https://arxiv.org/e-print/2110.06169> |
| **Decision Transformer** | `τ = (R̂₁,s₁,a₁,…,R̂_T,s_T,a_T)`，`R̂_t = Σ_{t'=t}^T r_{t'}`；**奖励只被用来算 R̂ 再当输入 token**；测试时 `target_return` 起手并逐步 decrement | `[一手]` arXiv:2106.01345 <https://arxiv.org/e-print/2106.01345> |
| **TD3+BC** | **BC 正则项**（③ 的「BC 辅助损失」形态，**同时是 ④ 的锚定项**）：`π = argmax_π E_{(s,a)~D}[ λ·Q(s,π(s)) − (π(s)−a)² ]`，其中 **`λ = α / Σ_{(s,a)}\|Q(s,a)\|`**（原文强调这是「normalize by the **average absolute value**」的启发式，用于平衡 value maximization 与 BC）；**α=2.5 默认**，在 (1, 2, 2.5, 3, 4) 上调过 | `[一手]` arXiv:2106.06860 <https://ar5iv.labs.arxiv.org/html/2106.06860>（**本轮补取**，U-05 已闭合） |
| **DAPG** | 演示数据上的**辅助 BC 损失**，其权重**同时含优势加权与退火**：`w(s,a) = λ₀·λ₁^k·max_{(s',a')~ρ_π} A^π(s',a')`（`k` = 迭代计数），**λ₀=0.1、λ₁=0.95**；原文动机逐字：「we **asymptotically decay the auxiliary objective**」（初期示范至少不差于策略，末期不再偏置梯度）；流程 = 先 **BC 预训练**再以该 augmented loss 微调 | `[一手]` **arXiv:1709.10087** <https://ar5iv.labs.arxiv.org/html/1709.10087>（**本轮补取**，U-06 已闭合）｜⚠️ **本稿更正**：上一稿写的 `arXiv:1910.10314` 经 arXiv API 复算是 *The pure-quartic soliton laser*（物理论文），**与 DAPG 无关** |

> **对我们**：我们**已经有一个「优势加权」的实例**——FL 的 `ppo_expert_bc.py` 的胜负加权（②层）与 `value` 回归项（③层的边缘形态）。而**我们仓内完全没有 ③ 类实现**：`rl/train_bc.py:107-111` 是纯 `-logprob`。**落地 ③ 只需改一处 loss 表达式**（见 §3.1 I5），但**它救不了 0 样本**（优势在零样本区也退化）。

### ④ **IL→RL 微调**阶段的奖励 + **锚定惩罚**（KL-to-reference / BC regularizer / trust region）—— 这是 IL 项目里最真实的「惩罚项」

| 实例 | 惩罚/锚定项 | 取值 | 出处 |
|---|---|---|---|
| **InstructGPT** | `objective = E[r_θ(x,y) − β·log(π_RL(y\|x)/π_SFT(y\|x))] + γ·E[log π_RL(x)]`；理由原句：「we add a **per-token KL penalty** from the SFT model at each token to **mitigate over-optimization of the reward model**」 | **β=0.02**（E.7：最优 0.01~0.02）；γ=27.8 | `[一手]` <https://arxiv.org/html/2203.02155v1> §3.5/E.7 |
| **Anthropic HH** | `r_total = r_PM − λ_KL·D_KL(policy ‖ policy_0)`；作者自评 λ_KL=0.001「**might actually be wholly unnecessary**」 | **λ_KL=0.001** | `[一手]` <https://arxiv.org/html/2204.05862v1> §4.1/§B.1 |
| **TRL v0.11.2** | 逐 token `non_score_reward = -kl_ctl.value * kl`；RM 分数**只加在最后一个非 mask token**；`AdaptiveKLController`：`proportional_error = np.clip(current/target − 1, −0.2, 0.2)` | `init_kl_coef=0.2`、`target=6.0`、`horizon=10000`、`target_kl=1.0`(早停门限) | `[一手]` <https://github.com/huggingface/trl/blob/v0.11.2/trl/trainer/ppo_config.py>、`.../ppo_trainer.py`（line 1135–1162）、`.../utils.py`（line 54–69） |
| **DeepSpeed-Chat** | `kl_divergence_estimate = -self.kl_ctl * (log_probs - ref_log_probs)`；`reward_clip = clamp(score, −5, 5)` 只加末 token | **`kl_ctl = 0.1`**（硬编码） | `[一手]` <https://github.com/deepspeedai/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/dschat/rlhf/ppo_trainer.py>（line 65–71、181–194） |
| **AlphaStar** | RL 更新损失的**第 3 项 = KL（当前 actor 轨迹 ↔ 监督人类策略）**；外部分析表述「penalized whenever their action probabilities differ from the supervised policy」。**KL 系数 / 是否退火：本轮未取回** | `[复现件]`+`[二手]` | `[复现件]` <https://ar5iv.labs.arxiv.org/html/2104.06890> §3.4；`[二手]` <https://cyk1337.github.io/notes/2019/07/21/RL/DRL/Decipher-AlphaStar-on-StarCraft-II/> |
| **绝悟** | **❌ 未找到** KL 锚定项；P3 的蒸馏用**交叉熵 `H^×(π_i‖π_θ)`**（教师→学生），不是 KL 锚定 | — | `[一手]`（见 `docs/juewu_research_2026-09-19.md` §5.2） |

> **两处必须并列的口径冲突（【R17】，不调和）**
> - **「KL 惩罚防 reward hacking」vs「KL 惩罚不改善真奖励前沿」**：InstructGPT 确实这么写；但**同一研究组的受控实验**（Gao/Schulman/Hilton）发现「The KL penalty only causes the gold RM score to converge earlier, but **does not affect the KL_RL-gold reward frontier**, and so the effect of the penalty on the gold score is **akin to early stopping**」，且「using KL penalty has a **strictly larger proxy-gold gap**, we set KL penalty to 0 for all other RL experiments」。**两者并列才算诚实**——`[一手]` <https://arxiv.org/html/2210.10760v1> §3.6。
> - **InstructGPT 的 KL 是 per-token（散文写了，公式无 Σ）vs Anthropic HH 的 KL 口径 `[NOT-FOUND]`**（HH 全文 grep `per-token`/`at each token` = **0 命中**，**不做推断**）。

### ⑤ **纯 RL + 奖励塑形并退火**（OpenAI Five 类，**非 IL**，作对照，**明确标注不可当 IL 证据**）

| 实例 | 事实 | 出处 |
|---|---|---|
| **OpenAI Five（Dota 2）** | **本轮已取回原文**（arXiv:1912.06680）：① 奖励 = **密集 shaped reward**，原文自述「**modeled loosely after potential-based shaping functions**, though the guarantees therein do not apply here」；② **奖励函数是项目开始时一次性构造的**（「constructed the reward function once at the start of the project」）；③ ⚠️ **关键辨别**：全文 5 处 `anneal` **没有一处**是「奖励塑形权重退火」——它指 **(a) 超参**（`learning rate and horizon`）与 **(b) 新环境/动作 feature 的 0%→100% 渐进引入**（「starting with 0% of rollout games played with the new environment or actions」），并「losing TrueSkill during the annealing → revert / 放慢」；④ 奖励权重只在 surgery 段以「**minor environment changes (such as improvements to partial reward weights or scripted logic)**」一笔带过，**无调度/退火描述**。⇒ **「OpenAI Five 用奖励塑形退火」这一流行说法在本轮取回的一手文本中未获直接支持**；且绝悟 P1 只说 `reward design is inspired by OpenAI Five's Dota reward`（**只提奖励设计灵感，未提退火**）⇒ **不得由 OpenAI Five 反推绝悟用了退火** | `[一手]` arXiv:1912.06680 <https://ar5iv.labs.arxiv.org/html/1912.06680>（§3.2 / §4.2 / Appendix G；**本轮补取**，U-04 部分闭合） |
| **绝悟（非 IL 对照）** | 奖励 = **人工设计、手工调参**（腾讯受访原话「所有的『奖励机制』都需要人类从头设计」）；**「奖励权重退火」在一手文献中找不到任何直接证据** | `[一手]` <https://cloud.tencent.com.cn/developer/article/2160236>（见 `docs/juewu_research_2026-09-19.md` §4.5） |

> **禁则重申**：⑤ 类**不可当作 IL 证据**；对我们而言它的价值是**反面案例**（§5 反面清单 **F2**，另见 F5 的 KL-to-human）。且 OpenAI Five 的数字（若有）**不得**与我们对比（禁则③）。

---

## §2 逐项目事实表

> 口径：**范式 / IL 数据规模与来源 / 奖励项 / 惩罚项 / 锚定项 / 塑形与退火 / 已知的坑 / 出处**。
> **所有外部项目的绝对数字仅用于说明其内部方案，不得与我们的数字对拍（禁则③【R15】）。**

### 2.1 FirstLight_CR（`[本地]` 直读，HEAD `28d66cc`）

| 维度 | 事实 | 出处 |
|---|---|---|
| **范式** | 两段式：**stateful IL**（独立入口 `launch_stateful_il.sh` → `train_imitation_cache.py`）→ **PPO 自对弈**；IL 用**时序 RNN + teacher forcing + TBPTT** | `.../v4/launch_stateful_il.sh:13-20`；`train_imitation_cache.py:1`、`:544-579`；`learning.py:59` |
| **IL 数据规模/来源** | **真人天梯回放**（RoyaleAPI 采集），非规则专家、非脚本 agent；`cache_builder.py:476` `--source-count` 默认 **100** | `.../v4/expert.py:1`（`"Expert timing/window contracts for online RoyaleAPI replay training."`）、`dataset.py:1`（`"…for RoyaleAPI IL replays."`）、`:60-75`、`:156-327`；`cache_builder.py:476` |
| **IL 损失（唯一权威）** | 6 项掩码 CE/Huber 之和：`gate`（ACT 类权重 8×）+ `candidate` + `target` + `delay`（邻域平滑 0.2）+ `continue` + `value`（Huber，target = 引擎奖励折扣回报）；权重默认全 1.0，`train_imitation_cache.py:436` **硬编码无 CLI 覆盖** | `.../v4/imitation.py:283-290`、`:145-152`；`train_imitation_cache.py:436`、`:353-354` |
| **奖励项** | IL 阶段**无**；唯一的「奖励成分」= 第 6 项 `value` 回归（target 由 `producer.py:395-405` 的 `discounted_returns` 产生，`gamma_per_decision`） | `.../v4/producer.py:395-412`；`imitation.py:271-281` |
| **惩罚项** | IL 阶段**无**（`penalty` 在 12 个 IL 文件里 0 命中；`advantage`/GAE 只在 `ppo.py:127`） | 穷举检索见 §2.1 备注 |
| **锚定项** | **PPO 损失内不存在** KL-to-reference / KL-to-expert / BC 正则 / trust-region（穷举检索 0 命中）；`ppo_expert_bc.py` 是「**每次 PPO update 之后的一步独立 BC 更新**」（自己 `zero_grad` + `step`），**默认关闭**（`--expert-bc-manifest` 无 default ⇒ `None`；生产 launch 脚本不传） | `.../v4/distributed_ppo.py:476/491-495`（`approx_kl` 是相对**自己旧行为策略**的 KL 早停，非锚定）；`ppo_runtime.py`/`checkpoint.py`/`factory.py` 0 命中；`ppo_expert_bc.py:1-7`、`:492`、`:576-578`；`train_ppo_self_play_cluster.py:976`、`:1742` |
| **塑形与退火** | IL 阶段无；PPO 的 `target_kl` 默认 `None`（未启用） | `.../v4/ppo.py:147/162-163` |
| **已知的坑** | ① IL 对齐动作**只到 5-tick 决策窗**，窗口外动作降级 WAIT（`producer.py:170-209`）② 引擎拒绝 → `gate_loss=False`（`:229-267`）③ **BC 步不碰 critic**（三重证据 + 回归测试）④ BC 的 probe KL 明确**不是**全 rollout 的 KL（`ppo_expert_bc.py:422`）⑤ `_sanitize_legacy_illegal_expert_targets` 说明**旧缓存存在非法标签**（数据侧历史债） | `producer.py:170-209`、`:229-267`；`ppo_expert_bc.py:387/525`、`:398-417`、`:422`；`tests/test_training_v4_ppo_expert_bc.py:192/211/248-251`；`cache.py:87-124` |
| **对我们的直接含义** | FL 是**唯一一个「IL 有完整实现、但 IL 阶段零奖惩」且我们能逐行读的对照**；它的锚定项**实际也不存在**（默认关）。⇒ 「IL 项目里有真实惩罚项」这一直觉，在 FL 上**不成立** | — |

### 2.2 AlphaStar（`[复现件]`+`[二手]`；**Nature 正文本轮未取回**）

| 维度 | 事实 | 出处 |
|---|---|---|
| **范式** | IL **先行独立阶段** → RL（再用 KL 把它拴住）；league（main player / main exploiter / league exploiter）+ PFSP | `[复现件]` <https://ar5iv.labs.arxiv.org/html/2104.06890> §3.4/§3.5 |
| **IL 数据规模/来源** | **971K replays / 44 天**（第三方复现件转述，**非原文**）；⚠️ 同作者两版 CPU 核数自相矛盾（128,000 vs 12,000）⇒ 子报告**不采任何一版** | `[复现件]`（见 `docs/alphastar_reward_il_survey_2026-09-19.md` §2.1） |
| **奖励项** | **二值 win/loss 终局信号（唯一环境奖励）** + **由人类统计量 z 导出的 pseudo-rewards**（build order 的 edit distance + 累计统计量的 Hamming distance） | `[二手-分析]`（同上 §3.2） |
| **惩罚项** | 「penalized whenever their action probabilities differ from the supervised policy」= **就是 KL 项的另一说法**，不是独立惩罚 | `[二手]` <https://cyk1337.github.io/notes/2019/07/21/RL/DRL/Decipher-AlphaStar-on-StarCraft-II/> |
| **锚定项** | **KL（当前 actor ↔ 监督人类策略）**，是 RL 更新损失第 3 项；**KL 系数与是否退火：`未验证`** | `[复现件]` ar5iv 2104.06890 §3.4 |
| **塑形与退火** | z 作 baseline 进 TD(λ)；值头 `baseline=(2/π)·arctan((π/2)b)`；**权重/退火两路独立检索均未找到** ⇒ `[未验证]` | `[复现件]`；同上 §3.2 |
| **已知的坑** | ① 去掉人类统计量奖励的 ablation **未取证**（最接近的是二手转述「utilzation of human data is critical」）② IL-RL 目标冲突（保真 vs 求胜）**有机制性证据，论文是否讨论未取证**③ league 中策略循环/被 exploiter 攻击 ④ 公平性质疑（未用 human action space）⑤ BC 误差累积 `O(T²ε)` | `[二手]`/`[未验证]`，逐条见 `docs/alphastar_reward_il_survey_2026-09-19.md` §5 |
| **对我们的直接含义** | 「KL-to-human」在我们这里**参考分布不存在**（我们无人类数据）；可换成 KL-to-**规则专家**，但**那是另一个机制**，不能声称「我们在用 AlphaStar 的 KL-to-human」 | `[本地]` 见 §3.1 |

### 2.3 绝悟 / JueWu（`[一手]`，arXiv API 逐字核对）

| 维度 | 事实 | 出处 |
|---|---|---|
| **范式** | IL（JueWu-SL）与 RL（1v1 / 5v5 / 开悟）**两段式**；P2 原文只说「**正在**结合 SL 与 RL」，**5v5 的 P3 通篇未提 SL 初始化** | `[一手]` arXiv:2011.12582；arXiv:2011.12692（见 `docs/juewu_research_2026-09-19.md` §3.3） |
| **IL 数据规模/来源** | **top 1% 人类玩家**（**不是职业选手**），**每英雄约 12 万局 / 1 亿样本**，预处理后**只留 1/20 帧** | `[一手]` arXiv:2011.12582 §3.1 |
| **IL 损失** | 双层动作标签 + 多视图意图标签的**四项交叉熵加权和**（四权重全 1，λ=1，**无 KL、无熵项**） | `[一手]` 同上 §3.2 |
| **奖励项** | 三张表全部抽出：1v1 Table 6、5v5 Table 4（**5 个价值头 = 奖励分解** `V̂=Σw_k V̂^k`，**`w_k` 数值论文没给**）、开悟 Table 5。dense 项：hp、mana^4、gold、exp、tower_hp 等 | `[一手]` arXiv:1912.09729 / 2011.12692 / 2209.08483（同上 §4.1–4.3） |
| **惩罚项** | 唯一显式消极惩罚 = 5v5 的 **`No-op −0.00001`**（**1v1 与开悟的奖励表里没有这一项**）；非法动作走**剔除**而非惩罚 | `[一手]` 同上 §5.1/§5.3 |
| **锚定项** | **❌ 未找到** KL 锚定（P3 蒸馏用交叉熵 `H^×`）；**✅ 有** Dual-clip PPO 与 PPO ratio clip（标准信任域） | `[一手]` 同上 §5.2 |
| **塑形与退火** | 密集项 = 塑形；课程 = **lineup 级三阶段**（固定阵容 → 多教师蒸馏 → 随机阵容，按 Elo 晋升）；**「奖励权重退火」在一手可核实文献中找不到任何直接证据**（逐字检索 `anneal`/`schedule`/`reward weight` 等）；`w_k` 调度 **`未验证`** | `[一手]`（同上 §4.4/§4.5） |
| **已知的坑（作者原文）** | ① learning collapse ② 泛化崩溃到**胜率恒为 0** ③ **没有 legal action 就快速收敛到局部最优**（消融 Fig.13）④ **通用奖励函数导致所有阵容打同一种打法**（macro-state entropy 0.000/0.014/0.408）⑤ IL 三处数据分布问题 + 「**人类微操有噪声，完全模仿会掉性能**」 | `[一手]`（同上 §6.1–6.6） |
| **口径冲突（并列不调和【R17】）** | Kill 的符号：**1v1 = −0.5、开悟 = −0.6、5v5 = +1**（三张表不一致）；折扣因子 **0.998 vs 0.997**（实测 P1 与 P6 一致、只有 P3 不同） | `[一手]`（同上 §5.1、§9 U5） |
| **对我们的直接含义** | 最可迁移的是 **A1「把非法动作做成交互契约而非奖励惩罚」**（正对我们 C1–C10 四层互不推导）；**A3 场景分层采样**正对「0 样本 ⇒ 0 梯度」；**A4 只模仿高层意图**（因为人类微操有噪声） | `[一手]`（同上 §7.1） |

### 2.4 RLHF 家族：InstructGPT / Anthropic HH / TRL / DeepSpeed-Chat（`[一手]`）

| 维度 | 事实 | 出处 |
|---|---|---|
| **范式** | 三阶段：**SFT/IL → RM → PPO（奖励 = RM 分数 − β·KL）**；HH **无显式 IL 阶段**（PM 先在 LM 语料预训练） | `[一手]` <https://arxiv.org/html/2203.02155v1> §3.5；<https://arxiv.org/html/2204.05862v1> §4.1 |
| **IL 数据规模/来源** | InstructGPT：**人类示范**（SFT 13k prompts / RM 33k / PPO 31k）；**v1 没有给出 SFT 损失方程** | `[一手]` 同上 §3.2；`[NOT-FOUND]` SFT 损失式 |
| **奖励项** | **学出来的 RM**（InstructGPT Eq.1 `−(1/(K choose 2))·E[log σ(r(y_w)−r(y_l))]`；HH **该文未写 PM 损失公式**，须引 Askell 2021 Eq.3.1） | `[一手]` 同上；<https://arxiv.org/html/2112.00861v3> |
| **惩罚项** | **KL to reference**（见 §1④ 取值表） | `[一手]` 同上 |
| **锚定项** | β=0.02（InstructGPT）/ λ_KL=0.001（HH）/ `init_kl_coef`=0.2（TRL）/ `kl_ctl`=0.1（DS-Chat） | `[一手]`（见 §1④） |
| **塑形与退火** | 无塑形退火；**有** KL 自适应控制器（TRL `AdaptiveKLController`：比例误差 + ±0.2 裁剪） | `[一手]` <https://github.com/huggingface/trl/blob/v0.11.2/trl/trainer/utils.py>（line 54–69） |
| **已知的坑** | ① RM 过优化 ⇒ **真偏好与 RM 反相关**（Stiennon §4.3）；② PM 高分 ≠ 真人高评（HH §4.1/§B.4）；③ **KL 惩罚只等价早停、让代理-金标缺口更大**（Gao §3.6）；④ **加大 KL 救不了 alignment tax**（InstructGPT §E.6：β 提到 2.0 = 默认 100 倍仍修不好）；⑤ TRL 负 KL 是失败前兆（issue #235 / #417：3/10 实验出现负 KL 爆炸） | `[一手]`+`[二手]`，逐条见 `docs/rlhf_kl_penalty_survey_2026-09-19.md` §4 |
| **对我们的直接含义** | **「无人类数据下 KL 仍有用」这一命题在本轮核查的 RLHF 文献中查无依据**（`[NOT-FOUND]`）；KL 是**把策略拉回参考分布**的项，而我们的唯一确证瓶颈是**探索不足** ⇒ **方向相反** | `[本地]`+`[一手]`（同上 §5） |

### 2.5 离线 IL + RL 微调家族（AWAC / IQL / DT / TD3+BC / RLPD / Cal-QL / robomimic / DAPG）

| 维度 | 事实 | 出处 |
|---|---|---|
| **范式** | 全部是「示范数据集 + 离线/半在线 RL 微调」；**奖励几乎从不直接进入模仿损失** | `[一手]` arXiv:2006.09359 / 2110.06169 / 2106.01345 |
| **IL 数据规模/来源** | 多为**人类示范**（robomimic 的 Square-MH 等）或**机器/合成数据**（D4RL locomotion） | `[一手]`；robomimic = **arXiv:2108.03298**（**本轮补取**，U-08 闭合） |
| **奖励项** | AWAC/IQL：奖励**只经 critic 变成标量优势**，再给模仿项当权重；DT：奖励**只被用来算 return-to-go 再当输入 token**；RLPD/Cal-QL：奖励**只进 TD/CQL 目标**，不进模仿项 | `[一手]` 同上 + arXiv:2302.02948 / 2303.05479 |
| **惩罚项** | AWAC 的 `KL(π‖π_β) ≤ ε` 是**推导前提**（Lagrangian），落成实现后表现为**批内 softmax 权重**；DT **无惩罚机制**；**RLPD 原文明确「we do not restrict the policy using a behavior cloning term」⇒ 无 BC 惩罚**；Cal-QL 亦无 BC 项 | `[一手]` arXiv:2006.09359；2302.02948 §1 |
| **锚定项** | **TD3+BC**：`λ·Q(s,π(s)) − (π(s)−a)²`，`λ = α/Σ\|Q\|`、α=2.5（**真正的 BC 正则**）；**DAPG**：辅助 BC 权重 `λ₀λ₁^k max A^π`（λ₀=0.1、λ₁=0.95，**渐近衰减**）；**RLPD/Cal-QL**：无 IL 锚定（Cal-QL 的机制是**保守价值下界的校准**） | `[一手]` arXiv:2106.06860 / 1709.10087 / 2302.02948 / 2303.05479（**本轮全部补取**，U-05/U-06/U-07 闭合） |
| **塑形与退火** | **DAPG**：`λ₁^k`（λ₁=0.95）**逐迭代渐近退火**辅助 BC 项（已逐字取回）；**TD3+BC 不退火**（固定 α=2.5）；RLPD/Cal-QL 无 IL 退火项 | `[一手]` arXiv:1709.10087 §IV-C2 |
| **已知的坑（证据分裂，【R17】并列）** | ① **人类示范上 BC-RNN 显著强于离线 RL**（robomimic Table 1「Square (MH)」：BC 52.7±6.6 / **BC-RNN 78.0±4.3** / BCQ 14.0±4.3 / CQL 0.7±0.9；原文结论逐字「methods that model **temporal correlations** (BC-RNN, HBC, IRIS) exhibit strong performance on **human datasets**」，而「Batch RL algorithms like BCQ … perform poorly on human datasets」）② **合成/机器数据与低数据量上离线 RL 反超**（D4RL locomotion 合计 BC 466.7 vs IQL 692.4）③ DT 自报 `56.1` vs 10%BC `56.7` ⇒ **DT ≈ 掐尖百分位 BC**，且「When data is plentiful … %BC can match or beat other offline RL methods」④ **反方决定性**：Paster et al.「You Can't Count on Luck」——return-conditioning 在**随机环境**下「**fail dramatically**」，Connect Four 上 RvS「cannot achieve more than 0.2 average return（win-rate 60%）」、2048「can only win about 60%」，且「**this lack of performance … is not due to a lack of data**」⑤ AWAC 论文附录有**内部不一致**（Z(s) 积分内用 π_θ 而非 π_β、期望式漏 exp，属排版/笔误级） | `[一手]` arXiv:2205.15967（Paster）；arXiv:2106.01345（DT 表）；arXiv:2006.09359（AWAC 附录）；robomimic = arXiv:2108.03298 Table 1（**本轮补取**，逐字核对）；IQL 数字见子报告 |
| **对我们的直接含义** | ③③④ 合起来给出一条**对我们最有用的判断**：**「掐尖百分位 BC」在有充足数据时能与离线 RL 打平，而它与「AWAC 式样本加权」在数学上是近亲**；但 ④ 提醒：**我们的环境含强随机性（卡序、对手）** ⇒ return-conditioning 属高风险方向 | `[一手]` |

### 2.6 OpenAI Five（Dota 2，**纯 RL，非 IL**，对照）

| 维度 | 事实 | 出处 |
|---|---|---|
| **范式** | 纯 RL（大规模 PPO，**无 IL 阶段**）+ **手工密集 shaped reward** | `[一手]` arXiv:1912.06680 §3.2 |
| **奖励项** | 「a more detailed reward function」，含角色死亡/资源收集等附加信号；利用零和多人体做**对称化**（减去对手所得）；原文自述 shaped reward「modeled loosely after potential-based shaping functions, though the guarantees therein do not apply here」；细节在 Appendix G；**奖励函数在项目开始时一次性构造** | `[一手]` 同上 §3.2 + Appendix G |
| **惩罚项** | shaped reward 本身**双向**（reward **or penalty**）；**无** KL/BC 类锚定惩罚 | `[一手]` 同上 Appendix G |
| **锚定项** | **无**（纯 RL，无参考策略/示范） | `[一手]`（以本轮取回文本为限） |
| **塑形与退火** | ⚠️ 全文 5 处 `anneal` **均非奖励权重退火**：(a) 超参（`learning rate and horizon`）；(b) 新环境/动作 feature 的 **0%→100%** 渐进引入（TrueSkill 掉则回退/放慢）。奖励权重只在 surgery 段以「minor environment changes (such as improvements to partial reward weights or scripted logic)」提及，**无调度** | `[一手]` 同上 §4.2 + surgery 段 + Appendix G |
| **已知的坑** | 环境/动作 surgery 会掉 TrueSkill；奖励函数一次性构造 ⇒ 后期只能靠 minor env change 微调 | `[一手]` 同上 |
| **对我们的直接含义** | **不可当 IL 证据**；「其奖励塑形退火」的常见引法**本轮未获一手支持**；且其数字**禁止**与我们对比 | 禁则③；§5 F2 |

---

## §3 对照我们项目的缺口表

> **只列我们已取证的缺口**（引用 `docs/` 文件名与 AGENTS.md 台账编号）。
> 每行写：**补法 / 触发哪条红线 / 验证手段（有没有逐位对账）/ 成本 / 判定**。

### 3.0 缺口清单（来源 = 台账，逐条可查）

| 编号 | 缺口（已取证） | 出处 |
|---|---|---|
| **C14** | ★「放弃一张买得起的牌」在 **75,891 帧**里发生 **0 次**（`0/7,238` 段；`max y = 4.000000`，0 段超过 4.0）；A 段最长 **22 帧 < 34 帧** ⇒ **0 样本 ⇒ 0 梯度**，且 step 0 与 100k **同结构**（与训练无关） | `docs/agents/ledger.md:28`；`docs/elixir_saving_audit_2026-09-18.md` |
| **O7** | ★ 机制自锁：Xbow **不在** `ACE_CARDS` 也不在 `SINK_TANK_CARDS` ⇒ `save_ace`/`setup_wait` 在 Xbow 卡组 **0 帧触发**；`_pick_suggested_card` **只推荐付得起的牌**（Xbow 要 6 费）⇒ 自锁回路。S1 门禁：`token_strict` 全帧圣水≥6 = **0/8191**、Xbow = **0/929**；而只多加一条不在 PlanToken 语义里的规则就打出 **156×** 基线 | `docs/agents/ledger.md:62`；`docs/s1_gate_2026-09-18.md` |
| **X-19** | ★ 「按局面结算的圣水交换信用分配」**已完整取证、判【不接线】**：离线口径 P4b 配对 10/10 正（Δρ(塔血)+0.145/Δρ(胜)+0.108）→ **在线正确口径 ÷18 / ÷7.7**（+0.008/+0.014）→ **补尾部 flush 后归零且方向不稳**（+0.000 / +0.001，3/5 与 2/5 正）；`τ` 太稀疏（`Στ/Σφ ≈ 1/12`）；门禁兑现率在 run 模式 **99.0–99.7% = 无分辨力** | `docs/agents/ledger.md:48`；`docs/s2_instrument_2026-09-18.md`；`docs/online_measure_2026-09-18.md` |
| **O9** | ★★ 模型会提交**非法卡包**被整包拒绝（B 类，A_et **497 帧 / 最长连续 225 帧 ≈ 112 s**，每帧 −0.05 ⇒ 累计 ≈ **−11.25 > 胜利奖励 +10**）；**已于 2026-09-18 定位并修复**（掩码 `used` 0/1-based off-by-one，一行修复 + 2 条掩码不变式 + 回归测试；端到端 B 类 = **0/3413 帧**）。另并列 `P1-20`（validate 通过但引擎拒绝）1,644 行**全为 BarbLog**（与掩码缺口 C2 逐项一致）。**⚠️ 修复后 C14 不变** | `docs/agents/ledger.md:64`；`docs/mask_used_slot_offbyone_fix_2026-09-18.md`；`docs/illegal_action_layers_2026-09-18.md` |
| **O8** | `economy_et` 100k 两臂：**没有任何一项能用现有仪器分辨出疗效**（机制层 4/6 不可分辨；两臂都没有把「攒费→打 Xbow」做出来，0.52% vs 0.05%、Xbow 5/8769 vs 1/8370）；**不构成判决**（缺阳性对照 + n=1/臂） | `docs/agents/ledger.md:63`；`docs/et_solo100k_judgment_2026-09-18.md` |
| **O10** | 探索侧「随机×随机 + 位置随机 + 正弦/衰减」**字面方案被定量否证**（`p_pass^k` 合取结构；`uniform` 臂 2,084 帧圣水**一次没到 6**）；有效轴是**时长**（`hold d~U{1..40}` ⇒ ≥6 帧占 28.96%/38.81%）；`bias_0.9` 达 18.9% 但**每局回报 −4.46 → −41.5**（⇒ 采样到了 PPO 会学到「别这么干」）；**门控偏置**是目前唯一同时抬覆盖率且不崩回报的形态（`gate@6:d`：6.765% vs 0.000%，回报 +1.68 vs +1.63，⚠️ n=8 且基线摆动） | `docs/agents/ledger.md:67`；`docs/exploration_randomization_analysis_2026-09-18.md`；`docs/exploration_bias_pass_2026-09-18.md`；`docs/exploration_pressure_gate_2026-09-18.md` |
| **C1–C10**（非法动作四层） | **四层并列且互不推导**：L1 掩码层（只产 bool、不产 reason）→ L2 整包校验（9 条中文 reason，任一非法整包拒收）→ L3 惩罚/统计（`invalid_penalty=0.05`，两来源：整包拒 =1、引擎拒逐卡 +1）→ L4 引擎物理（失败一律静默 `return False`、无理由串）。**10 条冲突并列不调和**，含掩码过严（C1 MergeMaiden 静态 6 vs 引擎动态 3/6；C6 Miner 全区豁免）、掩码缺口（C2 BarbLog）、作用域空隙（C3 8h 不裸下）、**同名两种基**（C9 `used` L1 0-based vs L2 1-based） | `docs/illegal_action_layers_2026-09-18.md`；`docs/agents/ledger.md:64` |
| **C3** | critic 仍不拟合，**加训练量救不了**（EV 11 点里 9 点 ≈0 或负、`vstd/rstd` 长期 ≈0.001） | `docs/agents/ledger.md:17` |
| **O2** | 价值头隐藏层对几乎全部帧**输出恒零**（35k→100k 的 22~24/27 窗口位移**恰为 0**）；价值函数 = 常数；`V≡常数时 resid≈0/grad_cos≈1 是恒等式` | `docs/agents/ledger.md:57` |
| **X6** | critic 侧程序**已按预注册闭合**（用户预授权：无进展就转策略侧）⇒ **不再往 critic 加投入** | `docs/agents/ledger.md:41` |
| **O5** | 同 seed 不可复现（机制未定；前 435 步逐位一致后 1e-4 漂移） | `docs/agents/ledger.md:60` |
| **O6** | 精确塔伤（触发式）已落地但**默认关闭**；端到端 A/B 是改任何生产默认值的前置条件 | `docs/agents/ledger.md:61` |

### 3.1 第一问详表：IL 家族**不碰奖励函数**的手段（机制名 → 落点 → 验证 → 成本 → 判定）

> **共同前提（先说清）**：`[本地]` 我们**无人类数据**——BC 目标是**规则专家在线自产**（`rl/train_bc.py:40-57` 的 `expert_bundle` = `BeliefPlanner.plan()` → `suggested_card` → `REGION_CENTERS` 最近合法格；`rl/train_bc.py:60-83` 的 `collect()` 跑 `RLEnv(opponent=None)` **只有 player-0**；**全文件无 `pickle.dump`/`json.dump`，不落盘**）。因此本节一律**不得**把「人类示范」当可用资产。

| # | 机制名 | 在我们仓内的落点 | 能否被逐位/统计口径验证 | 成本 | 判定 |
|---|---|---|---|---|---|
| **I1** | **BC / NLL 示范损失**（IL 无奖励的**证明**，不是修复） | `rl/train_bc.py:107-111`（`lp,_,_,_ = policy.evaluate(...)`；`loss = -lp`；单样本一步一更新、Adam、`np.random.permutation` 打乱）；`rl/human_play.py:205-233`（`train_bc_from_human`） | ✅ **逐位**：单帧 NLL 可由脚本用同一 ckpt 复算；`evaluate` 是**存储专家动作**的 log-prob（`rl/follower.py:753-806`：slot 分布 `:772-782`、cell 分布 `:784-790`、STOP `:791-795`），可逐帧对账 | **极低**（代码已存在，131 行） | **可做（作为仪器与先决条件）**。**但它本身不针对病根**：它只会把「便宜牌 95.47%」这一现状学得更牢（`docs/agents/ledger.md:58` O3 的实测）。⚠️ **不利证据**：IL 专家 = `BeliefPlanner` 建议卡 + 最近合法格，**只推付得起的牌** ⇒ 专家自身**在 Xbow 卡组上 0 帧产出激进的攒费动作**（O7） |
| **I2** | **时序性：stateful RNN + teacher forcing + TBPTT** —— 唯一能给 BC 注入「跨帧不花钱」能力的通路（**最高性价比**） | **现状（必须修的点）**：`rl/train_bc.py:107` 每帧传 `hidden=None`（`evaluate` 内 `rl/follower.py:763-765` 归零）⇒ **帧间无梯度、无状态**；`evaluate_batch`（`:650`）同样是**只重算存档 bundle**。**改造落点**：`rl/follower.py:753-806` 的 `evaluate` 签名与 `:765` 的 hidden 传递；`rl/train_bc.py:97-111` 的采样→训练循环改成 TBPTT 分块 | ✅ **逐位可对账**：(a) **同帧 A/B**：把「hidden 归零」与「hidden 携带」两版在**同一批帧**上跑，逐帧 logprob 与 loss 应一致（隐藏态只影响帧间耦合，不影响单帧语义）⇒ **可写成回归测试**；(b) **梯度链长度可断言**（`loss.grad_fn` 穿越的帧数）；(c) 行为层统计：STOP 概率 / A 段最长帧数（用 `scripts/pass_streak_audit.py` 复算） | **低**（改动局限在 2 个文件；无新常量，若按 F 类不改架构则**不触发 R6**） | **可做（须先预注册）**。与 FL 的对照：`[本地]` FL 的 IL **是** stateful + teacher forcing + TBPTT（`.../v4/learning.py:59` `"""Teacher-force actions; callers detach state between TBPTT chunks."""`、`:93-105`；`train_imitation_cache.py:544-579`；`imitation.py:121-123` "Batch equal-length chronological sequences **without shuffling frames**"）。**另一条一手支持**：`[一手]` robomimic Table 1 的结论是「methods that model **temporal correlations** (BC-RNN, HBC, IRIS) exhibit strong performance on **human datasets**」，而 Batch RL 方法（BCQ/CQL）在人类数据上很差（**本轮补取**，arXiv:2108.03298）⇒ **「时序性」是 BC 能不能用好示范的关键变量**，这独立支持 I2 的方向。⚠️ **代价与限定**：这**不增加覆盖率**，只让**未来**采样到的长时域片段**可学**；且「能不能学会」本轮**未验证**（与 O10 的「采样到了≠会学」同性质）。 |
| **I3** | **把「非法动作」做成接口契约而非奖励惩罚**（绝悟 P6 的 `legal_action` + `sub_action_mask` 同构；AlphaStar 亦走掩码） | `rl/follower.py:433`（`act`）、`:529`（`act_parallel`）的采样点；`rl/action_mask.py:521-573`（`validate_bundle`） | ✅ **逐位**：**O9 的修复已给出模板**——「采样出的 bundle 必须过 `validate_bundle`」+ 2 条掩码不变式（已用槽位必须非法 / 被掩槽位不得被选中）+ 回归测试 `test_mask_partial_bundle_invariants`；掩码类改动走 **【R13】** `scripts/_mask_diff_snapshot.py` **128 张位图逐位全等** | **低**（O9 已完成一半；剩余 = 把 L1/L2/L4 的合法性判据收敛到**单一权威源**） | **可做（先决条件）**。理由：`docs/illegal_action_layers_2026-09-18.md` 已证 **L3 的 `invalid_penalty` 在承担「纠错」职能**（累计 ≈ −11.25 > 胜利 +10），而绝悟的做法是**把非法性从奖励里拿出来**。⚠️ **不触碰奖励函数**（`invalid_penalty` 保留但不再当主要纠错手段）⇒ **不触发【R11】**。 |
| **I4** | **数据/缓存侧质量过滤**（FL 的 preflight reject / shadow 校验 / 非法标签清洗） | 我们侧对应物：`rl/export_replay.py`（只产 `EpisodeReplay`，**无 masks/plan_vec** ⇒ 不足以喂 BC）；`rl/replay.py` | ✅ 统计口径容易；❌ **逐位无意义**（它是减法） | 低 | **不做**。理由：**过滤只丢样本，不能造样本**；C14 是「0 样本」，减法无法解决。 |
| **I5** | **样本加权 / 优势加权（AWAC·AWR·IQL 形态）** | `rl/train_bc.py:107-111` 的 loss 改成 `-(w * lp)`，其中 `w = exp(A/λ)` 或 **批内 softmax**（AWAC 参考实现默认是后者）；`A` 的来源 = `rl/follower.py:765` 的 `value` 或 `rl/ppo.py` 的 GAE | ✅ **逐位**：权重张量可直接对账（同 batch 同 ckpt → 权重逐元素相同；softmax 分母口径可断言）；⚠️ **必须报「两套约定」**：`exp(A/λ)`（论文形式）vs `softmax(A/λ)`（实现默认）——【R17】要求分子/分母同超参 | 中 | **待预注册**。⚠️ **核心限制（原理性）**：优势加权只在**已有样本**上重分配权重；「攒费→Xbow」的样本数是 **0** ⇒ 它在零样本区**同样退化**（无 Q/V 估计可依）。**它不能替代 O7**。可能有用的地方是**缓解「便宜牌主导 95.47%」**（把 AD 序列的概率质量从高频项挪走），但那属于**另一个靶子**，须与 C14 分开预注册。 |
| **I6** | **return-conditioning（DT / RvS）**：以「目标回报」为条件生成动作 | `rl/train_bc.py:107` 需把 `R̂` 注入输入：最省的做法是**加进 `plan_vec`**（`rl/plan_space.py:124-158` 的 `to_vector`，注意【R6】的 `PLAN_DIM`）或**新增输入头**（后者触碰架构 ⇒ **必须 `--fresh`**，【R6】） | ✅ **逐位**：输入向量可逐元素对账；❗ **性能判据不可判**：`docs/agents/ledger.md:63` 的 §11.13.11 标定——**同 seed 同配置重跑单点差 0.60** ⇒ n=1 下胜率层差异一律不可归因 | **中高** | **待预注册**。⚠️ **风险最高的方向**：`[一手]` Paster et al. 已证明 return-conditioning 在**随机环境**下「**fail dramatically**」（且「**not due to a lack of data**」）。我们的环境含强随机性（卡序、对手）⇒ 属高风险。另：我们**无高质量示范轨迹**压条件分布（`未验证`）。 |
| **I7** | **DAgger / 交互式纠正 + 专家混合（expert iteration）** | 在线专家 = `rl/belief_planner.py`（但 `plan()` **只返回 PlanToken**，动作仍由 RL 出 ⇒ 须另建 PlanToken 贪心执行器，`docs/agents/ledger.md:58` O3 已点明）；重采样走 `rl/follower.py:433` `act`；专家迭代在 FL 侧的实例是 `ppo_expert_bc.py` 的**独立 BC 步**（默认关） | ✅ **逐位**：每个纠正帧的 `(obs, belief_tok, plan_vec, bundle, masks)` 可复算（`rl/follower.py:511-527` 的 `masks_for` 明确「与 `act()` 中 autoregressive 掩码生成完全一致」，是**为 BC/离线监督设计**的现成接口） | **高**（需在线专家 + 重采样预算 + 判据） | **待预注册（优先级低于 I2/I3）**。⚠️ **致命前提**：若专家本身在 Xbow 卡组上**不产攒费帧**（O7 已证 `save_ace`/`setup_wait` 0 帧触发），则 expert iteration **只是在放大同一个局部最优**。⇒ **必须先做 O7**，或先把专家改成能产目标轨迹的形态。 |
| **I8** | **IL→RL 锚定：KL-to-reference / BC 正则**（本节唯一「真·惩罚项」） | `rl/ppo.py:379-392`（PPO 损失处加项）；参考策略 = 冻结的 BC ckpt（`rl/follower.py:67` `load_checkpoint`） | ✅ 逐位可对账（`p=0` 时**行为与奖励逐位回旧**，参照本仓 `test_measure_only_is_behavior_neutral` 的做法）；❗ **性能判据不可判**（n=1，单点差 0.60） | 中（代码量小，验证成本高） | **不做**（当前）。理由：**方向相反** —— KL/BC 正则把策略**拉回参考分布**，而我们的瓶颈是**探索不足**（C14）；且 RLHF 的 KL 两条主干理由都以「奖励是学出来的代理」为前提，我们**没有 RM**（`docs/rlhf_kl_penalty_survey_2026-09-19.md` §5）；**触发【R11】**（不在 A′ 类取证之前改奖励/加惩罚项）。**出路**：若将来要做锚定，**正确先例是 offline RL 的 BC 正则族，不是 RLHF 的 KL**（【R17】：别混引）。 |
| **I9** | **奖励塑形 + 退火**（OpenAI Five 类） | — | — | — | **不做（非 IL 证据）**。见 §5 F2/F6。 |

**I1–I9 的合并结论（回答第一问）**：
1. **IL 家族能提供的、不碰奖励函数的可用手段 = I1（作为仪器）+ I2（时序，最高性价比）+ I3（掩码契约，先决条件）**；这三条都**不与【R11】冲突**（不改奖励、不改行为语义、I2/I3 均**有逐位对账手段**）。
2. **I5/I6/I7 都是「待预注册」而非「可做」**，因为它们的效力**全部依赖「先能采到目标局面」**，而那是 **O7 的活**。
3. **I8（KL/BC 锚定）方向与病根相反，判不做**；这是本轮**最明确的一条否定结论**。
4. **没有任何 IL 手段能绕过 C14**：IL 的数据侧与奖励侧**共用同一个采样器**。这是本轮合成里最重要的一条结构性判断。

### 3.2 缺口 → 机制映射表（只映射已取证缺口）

| 缺口（编号） | 哪个项目的哪个机制补它 | 补法 | 触发哪条红线 | 验证手段（有无逐位对账） | 成本 | 判定 |
|---|---|---|---|---|---|---|
| **C14**（0 样本 ⇒ 0 梯度） | **绝悟 A3「场景分层采样」**（P2 Scene Identification + 欠采样 + 样本归一化）；**绝悟 A1（掩码契约，间接：把「不可能」与「不划算」分开）** | **不在 IL 侧补**。IL 侧只能做 I2（让新样本可学）；**覆盖率必须靠 O7（机制）或 O10 已证有效的「带时长 option / 门控偏置」** | **【R11】**（未获拍板不改奖励）；**【R3】**（预注册）；若动 `hold_mask` 语义则撞 **动作语义冻结（`K_MAX=4`）** | ❗ **部分**：覆盖率可统计（`scripts/pass_streak_audit.py` 的 A/B/C 三类 + `scripts/probe_explore_randomization.py`）；**疗效不可判**（O8：n=1，单点差 0.60） | 中（机制改动小，验证成本高） | **可做（机制侧 O7，须单独预注册）**；IL 侧仅 **I2 配合** |
| **O7**（机制自锁：Xbow 不在 `ACE_CARDS`/`SINK_TANK_CARDS`） | **无外部机制可替** —— 这是**我们自己的机制缺陷**；绝悟 A1 只能提供**设计原则**（「合法性/可达性应是接口契约」） | 候选（O7 已写）：① 把高费致胜卡纳入 `ACE_CARDS`（或按卡费自动判定）；② 修 `_pick_suggested_card` 的「只推付得起的牌」 | **【R3】**（须单独预注册）；**【R7】**（若同时改多个常量 ⇒ 必须同源 + 对账 selftest）；**【R2】**（默认参数 = 旧行为） | ✅ **有较强的对账面**：`plan` 通道是**确定性可算量**（`rl/belief_planner.py` 是纯规则）⇒ 可逐帧对账 `hold_mask` 触发集合（PASS/FAIL 断言）；行为侧用 S1 的 `token_strict` 与 `random` 两臂对照（【R4】禁手抄基线） | 中 | **可做（最高优先级）** |
| **X-19**（奖励侧已完整取证、判不接线） | **IL 家族的「不碰奖励」手段（I1/I2/I3）**正是它给出的出路之一 | 把「信用分配」从**奖励结算**挪到**数据/表示侧**：I2（时序）+ I5（加权，待预注册） | **【R11】**（A′ 类取证之前不改奖励 —— X-19 的**取证已完成且判否**，故「不做奖励改动」是**唯一合规动作**） | ✅ I2/I3 有逐位对账；❗ I5 的性能判据在当前分辨率下不可判 | 低–中 | **可做（I2/I3）；I5 待预注册** |
| **O9 / P1-20（非法动作四层）** | **绝悟 A1**（`legal_action` + `sub_action_mask`：合法性与动作间依赖都是 **API 契约**，不是奖励项）；**FL 的「掩码直接委托引擎校验」结构** | 把 L1/L2/L4 的合法性判据收敛到**单一权威源** + 在 `act()` 采样点加「采样出的 bundle 必须过 `validate_bundle`」不变量 | **【R13】**（掩码/判定改动 ⇒ **128 张位图逐位全等**，用 `scripts/_mask_diff_snapshot.py`）；**【R17】**（C9「同名两种基」= 包含关系必须逐位验证）；**【R8】**（跨局边界回归） | ✅ **有**（O9 已完成：掩码层 44/256→0/256、rollout 拒绝帧 474/1707→0/1609、端到端 B 类 0/3413）；**【R13】的 128 张位图是硬门禁** | 低 | **可做（已部分完成）** |
| **C3 / O2 / X6（critic 侧）** | **❌ 无 IL 机制可补**。IL 家族要么不做价值（DT），要么把价值当加权工具（AWAC/IQL） | 若强行用 I5，则 `A` 的质量取决于**已死的 critic** ⇒ 加权在数值上退化（`value` 恒常数时 `exp(A/λ)` 恒常数 ≡ 均匀权重） | **【R11】不在 A′ 类取证之前拆价值头**；**【R14】**（判 critic 不许只看 EV，必须看参数有没有在动） | ⚠️ 可对账：`value` 的**唯一值计数**（本仓已有 `scripts/_probe_value_path.py` 的做法：600 帧实测 `value` 唯一值 1/600、`std=0`） | 低（诊断） | **不做（X6 已闭合 critic 侧）**。**但 I5 的前置**是「critic 还活着」⇒ 在 C3/O2 未解决前，I5 的收益上限受压制 |
| **O8（分辨率不足）** | **绝悟 A6「用累计价值损失挑选奖励项」**（形式化为**奖励项体检指标**）；**FL 的「两臂对称 + 逐位同参」纪律** | 把 I5/I2 的任何实验都**先解决可分辨性**：≥2 seed/臂 + 配对判据 | **【R5】**（n=1/臂分辨率不足）；**【R16】**（阈值余量 ≫ run 间散布）；**【R15】**（不跨实验照抄阈值） | ✅ **有先例**：本仓 `scripts/judge_anchor_blocks.py`（【R4】脚本复算）、`scripts/et_solo100k_readout.py`（含 `--selftest` 7/7） | 中 | **可做（前置）**——**在 I5/I6 之前必须先做** |
| **C1–C10（四层互不推导）** | **绝悟 A1**（单一权威源 + 动作间依赖）；**FL 的严格参数校验 + 掩码喂进网络** | 建一层**可复算的合法性产物**；`invalid_penalty` **不再承担纠错职能**（保留但降级） | **【R13】**【R17】【R7】（若动汇率） | ✅ 有（同 O9 一行） | 低–中 | **可做** |
| **O3（`elixir_avg` 低 / `setup_wait` 冷启动死锁）** | **绝悟 A4「只模仿高层意图而非底层动作」**（因为人类微操有噪声，完全模仿会掉性能）；**FL 的「不可表示窗口降级 WAIT」**（`producer.py:170-209`） | 把手写规则当**macro-goal 先验/辅助信号**，不直接替代策略动作 | **【R11】**（动作语义冻结 `K_MAX=4`）；**【R3】** | ✅ `plan` 通道确定性可对账 | 中 | **待预注册** |
| **O10（探索侧）** | **绝悟 A1/A3**（掩码契约 + 场景分层）；**绝悟的「无 action mask 就快速收敛到局部最优」消融**（P6 §H.2）**与我们 O7 结论同向** | 保留已证有效的形态（**带时长 option / 门控偏置**），**不做字面随机** | **【R3】**；若改行为分布 ⇒ **【R2】**（`p=0` 逐位回旧行为） | ✅ 覆盖率可统计；❗ 疗效不可判（n=8 且基线摆动） | 中 | **可做（但优先级低于 O7）** |
| **O5（同 seed 不可复现）** | **❌ 无补**（IL 项目里也普遍存在：FL 是单 commit 仓库、AlphaStar 未取到确定性说明） | — | **【R16】**（阈值余量 ≫ run 间散布） | — | — | **不做（记为方法学限制）** |
| **O6（精确塔伤默认关）** | **❌ 无 IL 机制**（这是**奖励项/特征**，不是 IL 手段） | — | **【R11】**（A′ 类之前不改奖励）；**【R2】** | ✅ 有逐位（关闭时逐位不变） | — | **不做（前置 A/B 未做）** |

---

## §4 硬约束校验

> 任何建议在落地前必须逐条通过本节。**本节不产生新结论，只做合规拦截。**

### 4.1 【R11】「不在 A′ 类取证之前改奖励」+ **X-19**（已完整取证、判不接线）

- **A′ 类取证的状态**：X-19 **已完成**（`docs/agents/ledger.md:48`：预注册首选 P3 被推翻、P1/P2 否决、P4b 在正确口径下 ÷18/÷7.7 且补尾部 flush 后归零）。⇒ 就**该方案**而言，「取证之前不许改」的条件**已满足并且结论是「不接线」**。
- **因此**：任何把「按局面结算的圣水交换」重新包装成奖励项的建议，**直接以 X-19 否决**（不是「证据不足」，是**已否证**）。
- **允许的动作**：把信用分配挪到**非奖励侧**（I2 时序 / I5 加权 / 表示侧），因为【R11】的字面禁令是「改**奖励**或拆价值头」。
- ✅ **§3.1 的 I1/I2/I3 全部不触碰奖励函数** ⇒ 合规。
- ⚠️ **边界**：I3 虽然保留 `invalid_penalty`，但如果建议**改它的数值**（0.05 → x），就落入【R11】+【R7】双重管辖 ⇒ **须单独预注册**。

### 4.2 【R13】128 张位图（任何掩码/判定改动）

- **硬门禁原句**：动 `legal_cells` / `_position_legal` / 掩码闸门 / `validate_bundle` 前后，各跑 `scripts/_mask_diff_snapshot.py` dump 位图（**8 状态 × 双方 × 8 手牌 = 128 张**：空场/压境/双倍期/推进坦克/残血塔/群杂/lv14），**逐位 diff 必须全等才提交**（历史：P0 掩码优化 128/128 一致才提交）。`[本地]` `docs/agents/redlines.md:27`；`scripts/_mask_diff_snapshot.py:1-4`
- **对本轮 IL 建议的约束**：
  - **I3**（掩码契约）若只是**加不变量断言**（不改变掩码输出）⇒ **128 张应逐位全等**（这是**必须**的）；若收敛 `_effective_card`（F1：只处理 Mirror）⇒ **几乎必然改变位图** ⇒ 必须**报出变更集**并说明为什么全等不成立（【R17】要求包含关系逐位验证）。
  - **I1/I2**（BC 训练侧）**不动掩码**，但 `rl/follower.py:511-527` 的 `masks_for` 明确承诺「与 `act()` 中 autoregressive 掩码生成完全一致」⇒ **任何让两者不一致的改动都是回归**，须加断言测试。

### 4.3 【R7】多常量改动（单常量源 + 对账 selftest）

- **原句**：奖励汇率 / 值函数 / 闸门 `edw×卡费` / MCTS 值函数必须**同源同步改**（已两次踩量纲失配：塔伤项 ×1000 使空砸王塔被误判正 EV）。`[本地]` `docs/agents/redlines.md:21`
- **对本轮建议的约束**：
  - **I5** 引入 `λ`（优势温度）⇒ **新常量**。按【R7】精神，`λ` 必须**单一来源**（不得在 `train_bc.py` 与配置文件各写一份）；且【R2】要求默认行为不变。
  - **I6** 若要注入 `R̂`，**注意【R6】**：`PLAN_DIM` 是「唯一常量源」（`rl/plan_space.py:3`），改它 = 架构变更 ⇒ **必须 `--fresh`**（热启动须显式 `plan_dim=58` / `belief_dim=563`；缺 `enc_ln`/`grid_ln` 会**静默错**）。
  - **I2** 若只改 hidden 传递（不改维度）⇒ 无新常量 ⇒ 不触发【R7】。✅

### 4.4 「我们**无人类数据**，BC 目标是规则专家在线自产」这一事实

- **逐条落点**（[本地]）：`rl/train_bc.py:40-57`（`expert_bundle` = `BeliefPlanner` → `suggested_card` → 最近合法格；**形参 `rng` 从未被使用** ⇒ 专家是**确定性函数**）；`:60-83`（`collect()` 用 `RLEnv(opponent=None)`，**只有 player-0**，**不落盘**）；`rl/human_play.py:1-20`（人类通道**另在别的文件**，产 `bc_<ts>.pkl`，与 `train_bc.py` **无代码耦合**）；磁盘实测：`find . -name "follower_*.pt"` / `bc_*.pkl` / `human_data/` → **全空**。
- **对本轮建议的约束（三条）**：
  1. **不得**引用任何「人类示范」类机制作为**可用资产**：AlphaStar 的 KL-to-human、JueWu 的 12 万局/英雄、InstructGPT 的 SFT 人类示范、RLHF 的 RM 偏好数据 —— 全部**降级为「机制形态参考」**。
  2. **「IL 数据质量决定上限」（LIMA）** 这条**不得**外推到我们（`[未验证]`）：LIMA 的证据是「1000 条**人类精选** > 5 万条未筛选」，我们的 BC 专家是**规则**且自动产 50 局（`rl/train_bc.py:121`）⇒ **两者的「数据质量」不是同一个量**。
  3. **I7（DAgger）的专家必须是「能产目标轨迹的专家」**：O7 已证现有规则专家在 Xbow 卡组上 **0 帧**触发攒费（`save_ace`/`setup_wait`）⇒ **先修 O7，再谈 DAgger**。

### 4.5 「外部数字 vs 我们数字」的对比一律禁止

- **禁止清单（举例，非穷举）**：AlphaStar 的 971K replays / 44 天 / CPU 核数；JueWu 的 12 万局/英雄、1 亿样本、60 万 CPU 核 + 1,064 GPU、17M 参数；InstructGPT 的 β=0.02 / SFT 13k prompts；AWAC 的 λ=0.3；IQL 的 τ=0.9/β=10.0；LIMA 的 1000 条。
  - ⚠️ **注**：FL 的「12.7M 参数」是**既有对照文档**（`三轴对照_奖惩_训练_非法动作_我们vsFirstLight_2026-09-19.md:27`）的记载，**本文未复算**（`[本地]` 复核：`.../v4/model.py:301-302` 有 `parameter_count()`，但本文**未实例化模型**）⇒ 引用时标 **`未验证`**，仅作「量级不可比」的示例，**不作为任何判据**。
- **理由**：量纲 / `step` 语义 / 参数规模不可比（【R15】；禁则③）。
- **允许的引用形态**：只引「**机制名 + 其内部逻辑**」，**不引数值作为我们的判据阈值**。
- ⚠️ **口径冲突并列不调和（【R17】）**：本轮的**三处**冲突必须**并列写出**，不得择一：
  1. **「KL 防 reward hacking」vs「KL 只等价早停 / 让代理-金标缺口更大」**（InstructGPT §3.5 vs Gao §3.6）。
  2. **「人类示范上 BC > 离线 RL」vs「合成数据上离线 RL > BC」**（robomimic Square-MH vs D4RL locomotion）。
  3. **「绝悟 1v1/开悟 Kill = 负」vs「5v5 Kill = 正」**；**折扣因子 0.998 vs 0.997**；**我们的奖励表键数 22（`rl/config.py:37-68`）vs 16（`rl/reward.py:28-46`）**。

### 4.6 【R3】与【R2】的额外拦截

- **【R3】**：I2/I5/I6/I7 的**判据必须在跑之前写死**（含失败分支）。本轮的判定一律写成「可做 / 待预注册 / 不做」，**不写成「建议这么做」**。
- **【R2】**：I2（hidden 传递）与 I5（权重）都会改变**训练时的行为分布**（若用于在线采样）⇒ 必须保证默认参数 = 旧行为、新能力只经显式开关开启。**本仓已有可照抄的做法**：`measure_only` 的 `test_measure_only_is_behavior_neutral`（逐位中性、复跑 3/3 PASS）。

---

## §5 反面清单（明确**不建议借鉴** + 理由）

| # | 不建议借鉴 | 理由 | 证据 |
|---|---|---|---|
| **F1** | **KL-to-reference 惩罚（RLHF 形态）** | ① 两条主干理由（防 RM 过优化、别跑出 RM 训练分布）**以「奖励是学出来的代理」为前提**，我们**没有 RM** ⇒ 无对象；② 剩下的熵/防塌缩已被 `ent_coef=0.01` 覆盖；③ **方向与病根相反**（KL 拉回参考分布 vs 我们的探索不足）；④ Gao et al. 受控证据：显式 KL 惩罚**不改善真奖励前沿**、只等价早停 | `[一手]` <https://arxiv.org/html/2203.02155v1> §3.5；<https://arxiv.org/html/2210.10760v1> §3.6；`[本地]` `docs/rlhf_kl_penalty_survey_2026-09-19.md` §5；`docs/agents/redlines.md:25`【R11】 |
| **F2** | **「OpenAI Five 的奖励塑形退火」这一说法本身**（含以此为样板改奖励） | ① **非 IL 证据**（不能当 IL 项目借鉴）；② 本轮已取回原文并**纠正误读**：全文 `anneal` 指的是**超参**与**新环境/动作 feature 的渐进引入**，**不是**奖励权重退火；奖励函数是项目开始时一次性构造（仅有「improvements to partial reward weights」这类 minor env change）⇒ **以它为「奖励塑形退火」先例属误读**；③ 我们的统计分辨率**不支持**「小步改奖励、快跑对比」的调参循环 | `[一手]` arXiv:1912.06680 §3.2/§4.2/Appendix G（U-04 部分闭合）；`[本地]` `docs/agents/ledger.md:63` O8 + §11.13.11（同配置单点差 **0.60**）；【R5】【R16】 |
| **F3** | **照搬绝悟的 `No-op −0.00001` 惩罚** | 我们的「不作为」是**圣水经济/机制**造成的（88%–91% 帧买不起任何手牌；`pre ≥6` 只占 0.02%），**不是奖励缺项** ⇒ 照抄会**误治病灶**。已实测：抬 stop 概率能把圣水≥6 帧从 **0% → 48%**，但每局回报 **−4.46 → −41.5** | `[一手]` arXiv:2011.12692 Table 4；`[本地]` `docs/pass_prob_2026-09-18.md`；`docs/exploration_bias_pass_2026-09-18.md`；`docs/juewu_research_2026-09-19.md` §7.2 N8 |
| **F4** | **AlphaStar 的 z-statistic pseudo-rewards（原始形态）** | ① **无人类数据** ⇒ z 的采样池不存在；② **量纲不可比**（z 是「整条 build order / 整套累计统计量」级对象，其距离数值与我们的单帧奖励 `tower_dmg=0.001` 级**差数个数量级**）；③ 单机 `n_envs=1` 的吞吐承受不起 | `[二手-分析]`；`[本地]` `docs/alphastar_reward_il_survey_2026-09-19.md` §6.2 |
| **F5** | **AlphaStar 的 KL-to-human** | 我们**没有人类策略** ⇒ 参考分布不存在。换成 KL-to-prior 是**另一个机制**，不得声称「我们在用 AlphaStar 的 KL-to-human」 | `[复现件]` ar5iv 2104.06890 §3.4；`[本地]` 同上 §6.2 |
| **F6** | **任何「IL→RL 的 RL 阶段奖励」搬用（含 JueWu 三张奖励表的具体数值）** | 量纲 / 卡牌机制 / `step` 语义**都不可比**；且 P6 的消融已证「**没有 action mask 就快速收敛到局部最优**」⇒ 该先解决的是**合法性接口**，不是奖励数值 | 【R15】；`[一手]` 同上 §5.3、§6.3 |
| **F7** | **AWAC 的「`exp(A/λ)` 就是被跑的形式」这一说法** | **参考实现默认不是它**：rlkit `normalize_over_batch` 默认 `True` ⇒ `softmax(score/beta)`；jaxrl 只有 batch-softmax。⇒ 引用时必须**两套约定并列**（【R17】） | `[一手]` <https://github.com/vitchyr/rlkit/blob/master/rlkit/torch/sac/awac_trainer.py>（line 618–631）；<https://github.com/ikostrikov/jaxrl/blob/main/jaxrl/agents/awac/actor.py> |
| **F8** | **用外部项目的失败读数当「我们也会这样」的论据** | 例如「JueWu 泛化崩溃到胜率恒为 0」不能推出「我们也该预期这个」——【R3】禁止把描述性观察写成判据；且各家环境/规模不可比 | 【R3】；禁则③ |
| **F9** | **把「掐尖百分位 BC」当通用解** | DT 自报 `56.1` vs 10%BC `56.7` ⇒ DT ≈ 掐尖 BC，且原文说「When data is plentiful … %BC can match or beat other offline RL methods」——**限定条件是「data is plentiful」**，而我们**连一条目标轨迹都没有** | `[一手]` arXiv:2106.01345 Discussion；`[本地]` C14 |
| **F10** | **return-conditioning 直接上线** | Paster et al. 已证它在**随机环境**下「**fail dramatically**」（且不是数据不足）；我们的环境含强随机性（卡序/对手）⇒ 属高风险，**只能待预注册** | `[一手]` arXiv:2205.15967 |
| **F11** | **把 FL 的 IL 阶段当作「有奖惩可借鉴」的样本** | FL 的 IL 损失 6 项**全部是掩码 CE/Huber**；`penalty` 在 12 个 IL 文件里 **0 命中**；其 PPO 阶段的锚定项（`ppo_expert_bc`）**默认关闭**且是**独立更新步**而非 PPO 损失项 | `[本地]` `.../v4/imitation.py:283-290`；`train_ppo_self_play_cluster.py:976/1742` |
| **F12** | **把「奖励项开了但没样本」当文献里有对应失败模式** | RLHF 的失败是「**奖励被利用**」（reward hacking），我们的是「**奖励没有样本可作用**」——**反向**的病。⇒ 任何从 RLHF 借来的「防奖励被利用」手段（含 KL）**都不对准** | `[本地]` `docs/rlhf_kl_penalty_survey_2026-09-19.md` §6.2 |

---

## §6 不确定 / 未能验证 / 需外部信息清单

### 6.1 本轮**未能验证**（不得当事实引用）

| # | 项 | 状态 |
|---|---|---|
| U-01 | **AlphaStar 的 Nature 论文正文** | `未验证`。`nature.com` 全部变体 **302 → `idp.nature.com`（SSO）**，取回工具不跟随跨域重定向；Springer 补充材料是 **PDF/ZIP**，工具明确拒收；`deepmind.google` 博客为客户端渲染只取到标题/日期；`web.archive.org` 直连超时。⇒ 报告里**没有任何一句来自论文原文** |
| U-02 | **AlphaStar 的 KL 系数与是否退火** | `未验证`（两路独立检索均未找到） |
| U-03 | **AlphaStar 的 z-pseudo-reward 权重/退火** | `未验证` |
| U-04 | **OpenAI Five 的塑形退火原文** | **部分闭合（本轮补取）**：原文已取回（arXiv:1912.06680）；「**奖励权重退火**」在一手文本中**未获直接支持**（5 处 `anneal` 均为超参 / 新环境·动作 feature）；**剩余未验证**：Appendix G 完整奖励权重表未逐项核对 |
| U-05 | **TD3+BC 的 BC 正则公式与 λ 构造** | ✅ **已闭合**（本轮补取 arXiv:2106.06860）：`λ = α/Σ\|Q\|`、α=2.5、`−(π(s)−a)²` 正则项 |
| U-06 | **DAPG 的辅助 BC 损失形式与退火调度** | ✅ **已闭合**（本轮补取 **arXiv:1709.10087**）：`w = λ₀λ₁^k max A^π`、λ₀=0.1、λ₁=0.95。⚠️ **并更正了本稿上一版引错的 arXiv id**（`1910.10314` 实为 *The pure-quartic soliton laser*） |
| U-07 | **RLPD / Cal-QL 的惩罚/正则细节** | ✅ **已闭合**（本轮补取 arXiv:2302.02948 / 2303.05479）：**RLPD 原文明确不使用 BC 项**；Cal-QL = CQL 预训练 + 保守价值校准（**无 IL 惩罚**） |
| U-08 | **robomimic 的具体数字（如 Square-MH 78.0）** | ✅ **已闭合**（本轮补取 arXiv:2108.03298 Table 1）：Square (MH) `BC-RNN 78.0±4.3` / `BCQ 14.0±4.3` / `CQL 0.7±0.9` **逐字核对** |
| U-09 | **InstructGPT 的 SFT 损失公式** | `[NOT-FOUND]`：v1 全文（HTML + TeX 源码）无 SFT 损失方程，也无 "per-token cross entropy" 措辞；只能引 Stiennon et al. §1 并标注**转引** |
| U-10 | **Anthropic HH 的 KL 是 per-token 还是序列级** | `[NOT-FOUND]`：全文 grep `per-token`/`at each token`/`sum over tokens` = **0 命中**。**不做推断** |
| U-11 | **Anthropic HH 的 PM 损失方程** | `[NOT-FOUND]`（该文确实没写；须引 Askell et al. 2021 §3.1 Eq.3.1） |
| U-12 | **HH 的 KL 系数消融** | `[NOT-FOUND]`（全文无 `ablation` 一词、无 β/λ 扫描报告） |
| U-13 | **Gao et al. 的 KL 扫描具体 β 数值表（图 9）** | `未验证`（只验证到结论句与「其余实验 KL=0」） |
| U-14 | **Stiennon Fig.5 的 β 取值序列** | `未验证` |
| U-15 | **DeepSpeed-Chat 的 reward running average** | `[NOT-FOUND]`：`ppo_trainer.py` grep `mean|std|moving|average|ema` 无命中；仅见 `main.py` import 了 `moving_average`，但**取不到该文件** ⇒ **不断言** |
| U-16 | **DeepSpeed-Chat 的 per-stage 详细文档** | `未验证`（取不到 README） |
| U-17 | **TRL `main` 上 `PPOTrainer` 移除后的替代路径** | `未验证`（`PPOTrainer` 已于 2026-09-04 从 `main` 移除，commit `700b845c`） |
| U-18 | **TRL 官方文档页 `huggingface.co/docs/trl/main/en/ppo_trainer`** | `未验证`：本环境**无法访问**（host 不可达，`web_fetch` 与 `curl` 均失败，重试 3 次 rc=000）；改用 repo 文档源 v0.15.2 |
| U-19 | **JueWu 的 `w_k` 数值与调度** | `未验证`（P3 只给 `V̂=Σw_k V̂^k`，**未给数值**） |
| U-20 | **JueWu 的奖励权重退火** | **在一手可核实文献中找到 0 处直接证据** ⇒ 不能断言其使用，也不能断言其未使用 |
| U-21 | **JueWu 的 U8 `hok_env` 源码** | `未验证`：本会话 `raw.githubusercontent.com` SSL 失败、`github.com` clone 133 s 超时 ⇒ 只有 P6 的 API 文档与论文描述 |
| U-22 | **JueWu 是否有 reward hacking / 卡 bug 的一手记载** | `[NOT-FOUND]`（六篇论文全文均无） |
| U-23 | **「手工规则奖励 + RL 产生可复现 exploit」的受控论文证据** | `[NOT-FOUND]`（CoastRunners 类案例本轮**未验证**：`blog.openai.com` DNS 失败、`openai.com/index/faulty-reward-functions/` 返回 403）⇒ **不得**在正文引用未验证的 CoastRunners |
| U-24 | **DAgger 原始文献** | 本轮**未逐字核**（§1② 的引用是二手转述） |
| U-25 | **「弱 SFT → RM 上限」类命题** | `[NOT-FOUND]`：未找到任何论文写这句话。**不要把它当引用** |
| U-26 | **`web_search` 工具可用性** | 本会话多次遇到 **HTTP 402（Insufficient Balance）** ⇒ 后半程全部改用 `web_fetch` 直连；**star/fork 等热度指标一律留空**（禁止臆造） |
| U-27 | **FL 的参数量 12.7M** | `未验证`：本文未实例化 FL 模型（`.../v4/model.py:301-302` 只提供 `parameter_count()`）；该数字来自既有对照文档 `三轴对照_奖惩_训练_非法动作_我们vsFirstLight_2026-09-19.md:27` |
| U-28 | **本仓奖励表的两套约定** | **并列不调和（【R17】）**：`rl/config.py:37-68` 的 `DEFAULT_REWARD` = **22 键**（经 `reward_to_env` 进 `RLEnv.reward_weights`）；`rl/reward.py:28-46` 的 `_DEFAULT_REWARD` = **16 键**（`rl/env_wrapper.py:99` 在无参时的初值）。两者**都会被执行**，且 `rl/reward.py` 用 `.get()` 读取其中 6 个可选键（`:111/160-165/199/205/213/223/228`）⇒ **「键数」不是单一数字**，引用时必须报两套 |

### 6.2 需要我们**自己产出**才能判定的信息（外部无法提供）

| # | 缺什么 | 为什么外部无法提供 | 出路 |
|---|---|---|---|
| N-01 | **「时序 BC（I2）在我们这里能不能学会攒费」** | 这是**我们环境 + 我们规则专家**的性质；外部项目的 IL 都是人类数据，条件不同 | 须做 I2 的小样本 A/B（含逐位对账 + 预注册判据） |
| N-02 | **「优势加权在 critic 近死（C3/O2）时是否退化成均匀权重」** | 取决于**我们**的 critic 数值状态 | `[本地]` 可实测：`value` 的唯一值计数（有现成做法 `scripts/_probe_value_path.py`） |
| N-03 | **「I5/I6 的疗效能否分辨」** | 取决于**我们**的 run 间散布（已实测单点差 **0.60**） | 先做 O8 的候选①（阳性对照）与②（**2 seed/臂**） |
| N-04 | **O7 机制修复的**具体形态**（纳入 `ACE_CARDS` vs 按卡费自动判定）** | 是**我们**的 `belief_planner` 设计问题 | 须单独预注册（O7 已列候选） |
| N-05 | **F1（`_effective_card` 只处理 Mirror）的可达性** | 我们代码的静态路径问题 | `docs/illegal_action_layers_2026-09-18.md` §8 已列为未定 U8–U12 |

### 6.3 需要**外部信息**（换通道才能拿到）

| # | 需要什么 | 为什么拿不到 | 建议通道 |
|---|---|---|---|
| E-01 | **AlphaStar Nature 正文原句**（IL 损失细节、KL 系数、消融） | SSO 重定向 + PDF/ZIP 拒收 | 能跟随跨域重定向且支持 PDF 解析的取回工具，或机构订阅 |
| E-02 | ~~OpenAI Five 的奖励塑形 + 退火原始描述~~ | ✅ **本轮已取回**（arXiv:1912.06680，`ar5iv` 直连）；结论见 §1⑤/§2.6 | — |
| E-03 | ~~TD3+BC / DAPG / RLPD / Cal-QL 原文~~ | ✅ **本轮已取回**（2106.06860 / 1709.10087 / 2302.02948 / 2303.05479） | — |
| E-04 | ~~robomimic 的数字原表~~ | ✅ **本轮已取回**（arXiv:2108.03298 Table 1） | — |
| E-05 | **JueWu 的 `hok_env` 源码** | SSL / 超时 | 换镜像或离线包 |

---

## §7 来源清单（全部 URL + 取数时间；本地项 file:line）

**统一取数时间**：子报告 **2026-09-19 03:33Z**（= CST 11:33）；本文的仓内复核为 **2026-09-19 CST**（逐条标注 `[本地]`）。

### 7.1 本地（我们项目）—— 全部 `file:line` 可直接查

| 用途 | 位置 |
|---|---|
| 红线全文 | `docs/agents/redlines.md`（【R2】`:11`、【R3】`:17`、【R7】`:21`、【R11】`:25`、【R13】`:27`、【R16】`:30`、【R17】`:31`） |
| 台账 | `docs/agents/ledger.md`：**C14** `:28`、**X-19** `:48`、**O2** `:57`、**O3** `:58`、**O7** `:62`、**O8** `:63`、**O9** `:64`、**O10** `:67`；C3 `:17`；X6 `:41` |
| BC 本体 | `src/clasher_new/rl/train_bc.py:33-37`（`REGION_CENTERS`）、`:40-57`（`expert_bundle`，`rng` 未使用）、`:60-83`（`collect`，`opponent=None`、不落盘）、`:86-116`（`train_bc`）、`:102-113`（`loss=-lp`）、`:107`（`hidden=None`）、`:121-127`（CLI 默认 50 局/600 步） |
| BC 的人类通道（**另一文件，无耦合**） | `src/clasher_new/rl/human_play.py:1-20`（模块 docstring）、`:194-202`（`load_bc_samples`）、`:205-233`（`train_bc_from_human`）、`:236-256`（`export_data`）、`:259-288`（`drive_games`，驱动者是 `rng.choice` 不是专家）、`:319-323`（`--bc-train`） |
| 策略与掩码 | `src/clasher_new/rl/follower.py:47`、`:55-65`（`save_checkpoint`）、`:67-100`（`load_checkpoint`，维度兼容）、`:235`（`nn.GRUCell`）、`:433`（`act`）、`:511-527`（`masks_for`，承诺与 `act` 一致）、`:529`（`act_parallel`）、`:650`（`evaluate_batch`，只重算存档 bundle）、`:753-806`（`evaluate`，`:763-765` hidden 归零、`:772-790` slot/cell logprob、`:791-795` STOP） |
| PPO（无 KL） | `src/clasher_new/rl/ppo.py:88`（`ent_coef=0.01`）、`:379-392`（`ratio`/`surr1`/`surr2`/`v_loss`）、`:393` |
| 主线加载 BC 的通道 | `src/clasher_new/rl/config.py:253`（`main_init`）；`rl/train_solo.py:1088-1093`；`rl/run_league.py:927-929`；`rl/flow_league.py:129-130` |
| 奖励表（**口径冲突：22 vs 16**） | `src/clasher_new/rl/config.py:37-68`（`DEFAULT_REWARD`，**22 键**）、`:298-299`（预设字段）；`src/clasher_new/rl/reward.py:28-46`（`_DEFAULT_REWARD`，**16 键**） |
| 机制自锁落点 | `src/clasher_new/rl/belief_planner.py:91-94`（`SINK_TANK_CARDS`/`FRONT_TANK_CARDS`）、`:263`（`_pick_suggested_card`）、`:663-707`（`setup_wait` + `hold_mask`）、`:740-774`（`_save_ace`）；`src/clasher_new/rl/action_mask.py:521-573`（`validate_bundle`）；`rl/plan_space.py:3`、`:11-12`、`:83`（**`ACE_CARDS`**，Xbow 不在其中）、`:103-158`、`:187-188`（`PLAN_DIM` 唯一常量源） |
| 观测（无历史栈） | `src/clasher_new/rl/observation.py:83-84`（`GRID_H,GRID_W=32,18`；`GRID_C=15`）、`:141`（`elixir` 当前帧）、`:165/169`（`opp_elixir`/`my_elixir`） |
| 取证文档（本文引用） | `docs/s1_gate_2026-09-18.md`（S1 门禁：约束在机制/探索侧）；`docs/elixir_saving_audit_2026-09-18.md`（C14）；`docs/s2_instrument_2026-09-18.md`（S2 仪器 8/8 + 6 个口径缺陷）；`docs/online_measure_2026-09-18.md`（在线口径 ÷18/÷7.7）；`docs/engagement_trade_prereg_2026-09-18.md` §11（六轮修订）；`docs/et_solo100k_judgment_2026-09-18.md`（O8）；`docs/readout_et_solo100k.md` + `docs/readout_et_solo100k/`；`docs/pass_prob_2026-09-18.md`；`docs/exploration_randomization_analysis_2026-09-18.md`；`docs/exploration_bias_pass_2026-09-18.md`；`docs/exploration_pressure_gate_2026-09-18.md`；`docs/illegal_action_layers_2026-09-18.md`；`docs/mask_used_slot_offbyone_fix_2026-09-18.md`；`docs/reward_composition_verdict_2026-09-14.md`；`docs/train_health_metrics_2026-09-18.md`；`docs/rand100_eval_2026-09-18.md` |
| 仪器（可复算） | `scripts/_mask_diff_snapshot.py`（【R13】128 张位图）；`scripts/pass_streak_audit.py`（A/B/C 三类）；`scripts/probe_explore_randomization.py`；`scripts/offline_engagement_trade.py`（8/8 不变量）；`scripts/analyze_online_trade.py`；`scripts/judge_anchor_blocks.py`；`scripts/forensics_card_usage.py`；`scripts/random_eval_100.py`；`scripts/et_solo100k_readout.py`；`scripts/finalize_et_solo100k.sh` |
| 跨项目对照（既有） | `三轴对照_奖惩_训练_非法动作_我们vsFirstLight_2026-09-19.md`（仓库根）；`docs/cmp_train_ours_2026-09-19.md`；`docs/cmp_reward_ours_detail_2026-09-19.md`；`docs/cmp_illegal_ours_2026-09-19.md`；`docs/cmp_hyperparams_2026-09-19.md`；`docs/cmp_manual_2026-09-19.md` |

### 7.2 FirstLight_CR（本地只读，HEAD `28d66cc0a5d65888515e22fdf22f11d783b65efb`）

| 用途 | 位置 |
|---|---|
| IL 入口 | `native_runner/training/v4/launch_stateful_il.sh:13-20` |
| IL 训练器 | `.../v4/train_imitation_cache.py:1`、`:293-294`、`:317-330`、`:345`、`:353-354`、`:436`、`:544-579` |
| **IL 损失（唯一权威）** | `.../v4/imitation.py:145-152`（`ILLossWeightsV4` 默认全 1.0）、`:181-199`（`_masked_mean`/`_masked_gate_mean`）、`:216-226`（delay 邻域平滑）、`:229-305`（`imitation_loss`）、`:258-281`（六项）、`:283-290`（总式） |
| IL 数据源与预检 | `.../v4/expert.py:1`、`:60-75`、`:156-327`、`:299-326`；`.../v4/dataset.py:1`、`:22-41`、`:44-90`；`.../v4/cache_builder.py:203-204`、`:476`；`.../v4/cache.py:87-124`、`:629-688` |
| 序列与掩码产地 | `.../v4/producer.py:170-209`、`:229-267`、`:382-418`；`.../v4/learning.py:45-112`（teacher forcing）、`:128-174`（`discounted_returns`）、`:220-240`（`trusted_decision_mask`） |
| 模型 | `.../v4/model.py:85`、`:164`、`:403`、`:573`、`:612-616`、`:662-671`；`.../v4/components.py:47-55`、`:58-68` |
| PPO 锚定（**不存在**） | `.../v4/distributed_ppo.py:241/387/391/476/491-495`；`ppo.py:147`、`:162-163`；`ppo_runtime.py`/`checkpoint.py`/`factory.py` 0 命中 |
| BC 辅助步（独立、默认关） | `.../v4/ppo_expert_bc.py:1-7`、`:156-201`、`:208`、`:251-253`、`:323`、`:342`、`:387`、`:398-417`、`:420-422`、`:445-459`、`:492`、`:525`、`:533`、`:555`、`:576-578`、`:645`；`train_ppo_self_play_cluster.py:976`、`:1231-1232`、`:1729-1758`、`:1740-1742` |
| 回归测试（可作模板） | `.../tests/test_training_v4_ppo_expert_bc.py:192`、`:211`、`:248-251` |
| 权重来源（胜负加权） | `native_runner/tools/experiments/build_hog_expert_manifest.py:79` |

### 7.3 外部项目（URL + 取数时间）

| 项目 | 来源 | URL | 时间 | 状态 |
|---|---|---|---|---|
| AlphaStar | Nature 论文（正文） | <https://www.nature.com/articles/s41586-019-1724-z> | 2026-09-19 | ❌ 302→SSO（**未验证**） |
| AlphaStar | 复现件 mini-AlphaStar | <https://ar5iv.labs.arxiv.org/html/2104.06890> | 2026-09-19 | ✅ |
| AlphaStar | 复现/批评论文 | <https://ar5iv.labs.arxiv.org/html/2209.11553> | 2026-09-19 | ✅ |
| AlphaStar | 二手分析（Deciphering） | <https://cyk1337.github.io/notes/2019/07/21/RL/DRL/Decipher-AlphaStar-on-StarCraft-II/> | 2026-09-19 | ✅ 二手 |
| AlphaStar | 二手分析（Alex Irpan） | <https://www.alexirpan.com/2019/02/22/alphastar-part2.html> | 2026-09-19 | ✅ 二手 |
| 绝悟 | 1v1（AAAI-2020） | arXiv:1912.09729 | 2026-09-19（arXiv API 复算） | ✅ |
| 绝悟 | JueWu-SL（TNNLS） | arXiv:2011.12582 | 同上 | ✅ |
| 绝悟 | 5v5（NeurIPS 2020） | arXiv:2011.12692 | 同上 | ✅ |
| 绝悟 | 开悟官方环境（NeurIPS 2022） | arXiv:2209.08483 | 同上 | ✅ |
| 绝悟 | MGG（NeurIPS 2021） | arXiv:2110.14221 | 同上 | ✅ |
| 绝悟 | 中文一手受访（奖励需人工设计 / 惩罚力度痛点） | <https://cloud.tencent.com.cn/developer/article/2160236> | 2026-09-19 | ✅ 一手访谈 |
| RLHF | InstructGPT | <https://arxiv.org/abs/2203.02155>；<https://arxiv.org/html/2203.02155v1>；<https://arxiv.org/pdf/2203.02155v1>；<https://arxiv.org/e-print/2203.02155v1> | 2026-09-19 | ✅ |
| RLHF | Anthropic HH | <https://arxiv.org/html/2204.05862v1>；<https://github.com/anthropics/hh-rlhf> | 2026-09-19 | ✅ |
| RLHF | Askell et al. 2021（PM 损失真出处） | <https://arxiv.org/html/2112.00861v3> | 2026-09-19 | ✅ |
| RLHF | Stiennon et al. 2020（KL 两个 purposes） | <https://ar5iv.labs.arxiv.org/html/2009.01325> | 2026-09-19 | ✅ |
| RLHF | Gao et al.（KL ≈ early stopping） | <https://arxiv.org/abs/2210.10760>；<https://arxiv.org/html/2210.10760v1>；<https://ar5iv.labs.arxiv.org/html/2210.10760> | 2026-09-19 | ✅ |
| RLHF | LIMA | <https://arxiv.org/abs/2305.11206>；<https://arxiv.org/html/2305.11206v1> | 2026-09-19 | ✅ |
| RLHF | Secrets of RLHF Part I | <https://arxiv.org/abs/2307.04964>；<https://ar5iv.labs.arxiv.org/html/2307.04964> | 2026-09-19 | ✅ |
| RLHF | TRL v0.11.2 代码 | <https://github.com/huggingface/trl/blob/v0.11.2/trl/trainer/ppo_trainer.py>；`.../ppo_config.py`；`.../utils.py` | 2026-09-19 | ✅ |
| RLHF | TRL v0.15.2 文档源 | <https://github.com/huggingface/trl/blob/v0.15.2/docs/source/ppo_trainer.md> | 2026-09-19 | ✅ |
| RLHF | TRL 官方文档页 | <https://huggingface.co/docs/trl/main/en/ppo_trainer> | 2026-09-19 | ❌ **不可访问（U-18）** |
| RLHF | DS-Chat 代码 | <https://github.com/deepspeedai/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/dschat/rlhf/ppo_trainer.py> | 2026-09-19 | ✅ |
| RLHF | TRL issue/PR（二手） | <https://github.com/huggingface/trl/issues/235>；`/issues/417`；`/issues/1178`；<https://github.com/huggingface/trl/pull/6429> | 2026-09-19 | ✅ 二手 |
| 离线 IL | AWAC | <https://arxiv.org/abs/2006.09359>；<https://arxiv.org/e-print/2006.09359>；<https://awacrl.github.io/> | 2026-09-19 | ✅ |
| 离线 IL | AWAC 参考实现 | <https://github.com/vitchyr/rlkit/blob/master/rlkit/torch/sac/awac_trainer.py>；<https://github.com/ikostrikov/jaxrl/blob/main/jaxrl/agents/awac/actor.py>；<https://github.com/ikostrikov/jaxrl/blob/main/jaxrl/agents/awac/value.py> | 2026-09-19 | ✅ |
| 离线 IL | Decision Transformer | <https://arxiv.org/e-print/2106.01345>；<https://arxiv.org/abs/2106.01345> | 2026-09-19 | ✅ |
| 离线 IL | Paster et al.（RvS 在随机环境失败） | <https://arxiv.org/e-print/2205.15967>；<https://arxiv.org/abs/2205.15967> | 2026-09-19 | ✅ |
| 离线 IL | IQL | <https://arxiv.org/e-print/2110.06169>；<https://arxiv.org/abs/2110.06169>；<https://github.com/ikostrikov/implicit_q_learning> | 2026-09-19 | ✅ |
| 离线 IL | TD3+BC / DAPG / RLPD / Cal-QL / robomimic | arXiv:2106.06860 / **1709.10087** / 2302.02948 / 2303.05479 / 2108.03298 | 2026-09-19 | ✅ **本轮补取**（U-05~U-08 闭合；DAPG 的 id 已更正） |
| OpenAI Five | Dota 2 with Large Scale DRL | arXiv:1912.06680；<https://ar5iv.labs.arxiv.org/html/1912.06680> | 2026-09-19 | ✅ **本轮补取**（U-04 部分闭合） |

### 7.4 本轮补取的一手文献（合成阶段新增，取数时间 **2026-09-19 CST**）

> **通道**：`curl -L` 直连 arXiv API（`export.arxiv.org`）与 `ar5iv` HTML → 去标签为纯文本 → **本地逐字 grep**（不是二手转述）。
> **纪律**：本节只用于闭合「机制形态」问题，**不引任何外部数值与我们的数字对比**（禁则③）。

| # | 文献 | URL | 闭合项 | 关键取证（逐字） |
|---|---|---|---|---|
| 1 | OpenAI Five, *Dota 2 with Large Scale DRL* (1912.06680) | <https://ar5iv.labs.arxiv.org/html/1912.06680> | U-04（部分） | shaped reward「modeled loosely after potential-based shaping functions」；奖励函数「constructed … once at the start of the project」；`anneal` = 超参 / 新环境·动作 feature（**非奖励权重退火**） |
| 2 | TD3+BC, *A Minimalist Approach to Offline RL* (2106.06860) | <https://ar5iv.labs.arxiv.org/html/2106.06860> | U-05 | `π=argmax_π E[λQ(s,π(s))−(π(s)−a)²]`；`λ=α/Σ\|Q(s,a)\|`；α=2.5（调过 1/2/2.5/3/4） |
| 3 | DAPG (1709.10087) | <https://ar5iv.labs.arxiv.org/html/1709.10087> | U-06（+ id 更正） | `w(s,a)=λ₀λ₁^k max A^π`；λ₀=0.1、λ₁=0.95；「we asymptotically decay the auxiliary objective」 |
| 4 | RLPD (2302.02948) | <https://ar5iv.labs.arxiv.org/html/2302.02948> | U-07 | 「we **do not** restrict the policy using a behavior cloning term, and do not reset to demonstration states」 |
| 5 | Cal-QL (2303.05479) | <https://ar5iv.labs.arxiv.org/html/2303.05479> | U-07 | CQL 离线预训练（Eq. 5.1）→ 在线微调；机制 = 保守价值下界**校准**（非 IL 惩罚） |
| 6 | robomimic (2108.03298) | <https://ar5iv.labs.arxiv.org/html/2108.03298> | U-08 | Table 1「Square (MH)」：BC 52.7±6.6 / **BC-RNN 78.0±4.3** / BCQ 14.0±4.3 / CQL 0.7±0.9；「methods that model **temporal correlations** … strong performance on **human datasets**」 |

**arXiv id 复核（同一通道，2026-09-19）**：`1910.10314` = *The pure-quartic soliton laser*（**与本调研无关**）；`1709.10087` = *Learning Complex Dexterous Manipulation with Deep RL and Demonstrations*（**DAPG 真身**）。⇒ 上一稿 §1③/§7.3 的 DAPG id 系误引，本稿已改。

---

## 附：本报告对纪律条款的自查（【R3】【R10】【R17】）

1. **无 URL / `file:line` 的陈述**：已用 `未验证` / `[NOT-FOUND]` 标记，共 **24 项**未闭合或部分闭合列于 §6.1（**U-05/U-06/U-07/U-08 已于本轮闭合，U-04 部分闭合**，见 §7.4）；正文中凡涉外部措辞处均带来源等级。
2. **描述性 vs 判据**：全文凡描述性读数（如 O6 的触发率、O8 的两臂计数差异、C1–C10 的 1,644 行）**一律标注「描述性、非判据」**，未升格为判断依据。
3. **外部数字**：全文**未**用任何外部项目的数值作为我们的判据阈值；外部数值**仅**用于描述其机制内部构造，并在 §4.5 明列禁止清单。
4. **措辞**：全文未出现「修好了 / 确认无效」；一律写「观察到 / 未达阈值 / 不可分辨 / **不构成判决**」。
5. **口径冲突**：三处冲突（KL 效果、BC vs 离线 RL、JueWu Kill 符号 + 折扣 + 我们的 22/16 键）**并列不调和**，见 §4.5。
6. **本文的性质**：**综合与映射**，不含新的实验；所有引用均可回溯到 §7 的 `file:line` / URL。
7. **本轮补取（2026-09-19）**：新增 **6 篇一手文献**并更正 **1 处 arXiv id 误引**（DAPG）；新增/修改的结论均标 `[一手]` 并附 URL 与取数时间，且**未**引用其数值作为我们的判据（仅用于描述其机制内部构造）。仓内 `file:line` 已用脚本**逐条复核**（31 个锚点：命中 28、需修正 3、已修正）。
