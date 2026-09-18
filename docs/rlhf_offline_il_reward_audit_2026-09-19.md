# 对象 4 —— RLHF + 离线 IL 族奖惩设计 · 本轮「勘误与补强」

> **本文件的性质（先读这一段 · 缺口如实声明）**
> 本轮 5 个对象报告里，对象 1/2/3 已落盘；**对象 4/5 本轮未落盘**。
> 对象 4 的子任务**受只读约束**，其字段级正文报告**在本会话从未写入工作目录**；本文件保存的是该子任务返回的**勘误与补强全文（逐字）**。
> ⇒ **本文件不是对象 4 的完整对象报告**：字段级（①–⑧）正文见下面「正文落点」列的盘上既有材料，本文件只承载**本轮新增/更正**的部分。缺口见 **§G**，**未做任何补写或推测**。
>
> **取数时间**：2026-09-19（CST）
> **纪律**：标 `[未验证]` / 「未验证」的项**原样保留**；**禁止跨项目数字对拍**；**不与其它四份对象报告合并**。

> **★★ 顶部补记（2026-09-19 本轮，编排者加）**：本文件**已补齐**对象 4 的字段级正文。
> - **本文件开头完整保留上一轮的 70 行原文、逐字未改一字**；仅在其第 9 行之后插入了本补记（6 行）⇒ 原文现分布于**第 1–9 行与第 16–76 行**（顶部原文里的「§G 缺口声明」也原样留在第 71–76 行）。
> - **正文（①–⑧ 固定 schema，两族各一套）从 §G 补记之后的「§0」开始**；上一轮 §G 第 1 条的缺口**自此关闭**。
> - 本轮新增 = 完整字段级抽取 + **本轮独立复取的一手文献**（RLPD `2302.02948` / robomimic `2108.03298` / Paster `2205.15967`，以及 Gao `2210.10760`・LIMA `2305.11206`・InstructGPT `2203.02155`・HH `2204.05862`・DT `2106.01345` 的逐字 grep）。
> - 本文件不含任何跨项目数字对拍；口径冲突**并列不调和**（【R17】）。

> **★ 统一 schema（五份对象报告共用：字段名 / 顺序 / 单位口径一致 — 2026-09-19 编排者补，仅加声明、不改正文）**
> **字段集 ①–⑧**：① 奖励项清单（项名 / 权重 / 量纲 / 作用域 / 是否零和）・② 惩罚项（显式 / 隐式 / 用约束或掩码替代）・③ 塑形与调度（是否 PBRS？是否退火？局内时间权重？）・④ 归一化三问（奖励 / 价值 / 优势）・⑤ value head 设计・⑥ 已知坑（带原文句；二手必须标注）・⑦ 一手出处（URL / `file:line` + 取数时间）・⑧ 阶段归属（IL / RL / 跨阶段接口）
> **本报告落点**：本文件 = **勘误轮**；①–⑧ 的字段级正文见下表「正文落点」。
> **统一单位口径**：权重一律**按来源原样记录**并显式标注量纲（分母单位）；无单位者写「无单位」；**同名项若量纲不同则并列不调和**，不折算、不换算、不对拍。
> **证据等级图例**（沿用盘上约定，不新建体系）：`[一手-论文原文]`・`[一手-官方代码]`・`[复现件-OA论文]`・`[二手-分析]`・`[未验证]`・`[本地-实测]`
> **三条硬纪律**：① **禁止跨项目数字对拍**（一切数字只用于标识与各自内部一致性论证）；② **标「未验证」的项一律保持未验证**，不因抽成表而升级证据等级；③ **不得把 IL 阶段的事写成 RL 阶段**，跨阶段项必须落在 ⑧。

---

## 正文落点（对象 4 在盘上的既有材料）

| # | 文件 | 覆盖 |
|---|---|---|
| 1 | [`docs/rlhf_kl_penalty_survey_2026-09-19.md`](rlhf_kl_penalty_survey_2026-09-19.md)（450 行） | RLHF 家族：三阶段结构与损失、KL 惩罚项的作用/实现/取值、开源实现代码位置、实践坑、与我们项目的类比边界、来源清单、未验证项（**其编号体系为 §8，无 U 编号**） |
| 2 | [`docs/il_reward_mechanism_synthesis_2026-09-19.md`](il_reward_mechanism_synthesis_2026-09-19.md)（488 行） | 使用 IL 的项目奖惩机制合成（含 offline RL：DT / TD3+BC / Cal-QL / RLPD 等） |
| 3 | [`docs/il_reward_reference_analysis_2026-09-19.md`](il_reward_reference_analysis_2026-09-19.md) | 参考分析（盘上复核出三处需修正） |

⚠️ **编号不可混用**：既有文件的未验证编号与本轮子任务的 **U1–U15 不是同一套**。本文件保留子任务的 U 编号**原样**，**不做映射、不做合并**。

---

## §1 本轮勘误与补强（1 处；U15 关闭）——子任务返回原文逐字

**触发原因**：子智能体 B 的正文与收尾信息对 RLPD 引文给出**互相矛盾**的归属（正文："as a characterization of DDPG+BC"／收尾："describing DDPG+BC"）。我不接受二手归属，直接取回一手 HTML 看上下文（`web_fetch` HTTP 200，**2026-09-18 20:49–20:55 UTC**，ar5iv 镜像；本会话已记录 raw/API 不可达）。

### 1. U15 判定结果：原报告写「未能确定」是对的，现在可以关掉——**「we」= RLPD 自己**

一手上下文（<https://ar5iv.labs.arxiv.org/html/2302.02948> **§2 Related work → "Constraining to prior data"** 段，逐字）：

> "Also similar to our work is that by [Nair et al. 2018a], who also use an off-policy algorithm with a fixed offline replay buffer. **However, we do not restrict the policy using a behavior cloning term, and do not reset to demonstration states.** Moreover, we note that these approaches generally require the offline data to be high quality … while **our approach** is, importantly, agnostic to the quality of the data."

⇒ 该句是 **RLPD 在与 Nair et al. 2018a（DDPGfD）对比时的第一人称自述**，不是「描述 DDPG+BC」。**结论「RLPD 无 BC 惩罚项」成立**，且证据面比原报告写的更宽：

- Intro 自设问题："…without offline RL pre-training or **explicit imitation terms** that privilege the prior offline data?"
- §2 "Offline RL pre-training" 段："our normalized update is **not an offline RL method** — we do not perform any offline pre-training but run online RL from scratch with offline data included in a replay buffer."
- §4 开头："an approach based on off-policy model-free RL, **without pre-training or explicit constraints**."

### 2. §2 表 RLPD 行需要改两处

| 字段 | 原报告写法 | 修正后 |
|---|---|---|
| **② 惩罚项** | "明确没有 BC 项……（位于 §2 Related Work……是**对其自身方法**的陈述）" | ✅ 结论不变；**引用措辞**改为「**§2 "Constraining to prior data" 段，第一人称自述（与 Nair et al. 2018a / DDPGfD 对比）**」；并补上 Intro 的 "explicit imitation terms" 与 §4 的 "without pre-training or **explicit constraints**" 两条旁证 |
| **④ value/critic** | "TD3 式 twin critic + LayerNorm；对称采样 + 高 UTD" | **补上机制原话**：其防过外推靠的是 **critic 里的 LayerNorm——一个「界」，不是惩罚项**：「LayerNorm **bounds** the values and empirically prevents catastrophic value extrapolation」；「it **does not explicitly constrain** the policy to remain close to the offline data」；「instead of creating a mechanism that **explicitly discourages OOD actions**, which can be viewed as **anti-exploration**」。算法侧：**SAC(MaxEnt)**（actor 目标含 `−α log π`）+ **CDQ 取子集 `Z∈{1,2}`** + 大 ensemble `E` + **对称采样 50/50** + 高 UTD |

### 3. 顺带一条与本仓相关的交叉核对（新增，非原报告内容）

**不能因为「RLPD 用了 LayerNorm」就照搬**：RLPD 的 LayerNorm 是**用来修 critic 过外推**的；而本仓 **O2** 已定位**价值头末端 LayerNorm 是 critic 死亡的第一性原因之一**（近常数输入被归一化 ⇒ 价值头可表达跨帧方差 ≈0.14 而目标 std≈12.11，ReLU 逐个死亡锁死；实测 `value` 唯一值 1/600、std=0）——**同名机制、相反作用**。这条属我们的本地事实，须自证，**不得引 RLPD 当依据**。

### 4. 未受影响的其余部分

§1–§7 的其他结论**不变**；§7 的 **U15 从「未验证」移入「已关闭」**，U1（TRL/DS-Chat 代码未能独立复验）、U11（DT 论文 `[NOT-FOUND]` vs 代码 `scale=1000.`）、U13（TD3+BC λ 含/不含 `1/N` 两套并存）、U14（Cal-QL 的「无 BC 项」属 verified-by-absence）、U12（DT Average 行列标签我未亲自复核）等**仍按原样保留为未验证**。交付物性质不变：**未写入工作目录任何文件**（只读约束）。

---

## §G 缺口声明（**不补写**）

1. 对象 4 的**字段级对象报告（①–⑧ 表）本轮未落盘**，且本会话**无法恢复**（子任务只读 + 全仓 grep 无命中 + 无 session transcript 命中）。
2. 因此「五份对象报告」中，对象 4 目前只有**勘误轮**——缺陷如实记录在此，**未用推测补写**。
3. 若需补齐：建议**单独重跑对象 4 的抽取子任务**，输入 = 上面「正文落点」3 份盘上材料 + 本文件顶部的统一 schema。
4. 本文件**不含**任何跨项目数字对拍；文中出现的数值均为**上游各自项目内部**的读数或**本仓**读数（§1.3 已显式声明「须自证，不得引 RLPD 当依据」）。

---

## §G 补记（2026-09-19 本轮 · 缺口**已补齐**）

上面 §G 的 1–4 条是**上一轮**的如实记录，**原样保留、一字未改**。本轮按 §G 第 3 条的建议**单独重跑了对象 4 的抽取子任务**，字段级正文（①–⑧）已写入下面 §0–§10；**§G 第 1 条的缺口自此关闭**（该条作为事故账继续留在原处）。

> **阅读顺序**：先读本文顶部的「统一 schema」与这段补记，再从 §0 起读。**§G 之后的内容全部是新增**，顶部 70 行原文未做任何删改。

---

# 对象 4 完整抽取正文 —— RLHF 与「离线 IL + RL 微调族」的 RL 阶段奖惩设计

> **写入时间**：2026-09-19（CST）。**取数时间**：仓内 `file:line` 复核与外部**直取**均为 2026-09-19（本子任务）；上游子报告另记录其外部一手源取回窗口为 2026-09-18T19:33Z–20:30Z。
> **性质**：**抽取 + 归档**，不含新实验、未跑训练、未改任何训练/奖励代码。
> **素材**：`rlhf_kl_penalty_survey_2026-09-19.md`（450 行）、`il_reward_mechanism_synthesis_2026-09-19.md`（488 行）、`il_reward_reference_analysis_2026-09-19.md`（279 行）= 上游三份盘上材料；+ 本轮**独立复取**的一手文献（见 §10.2）。
> **本文件纪律（自缚，逐条遵守）**：① **禁止跨项目数字对拍**（一切数字只用于标识其各自项目内部构造，**不得**当我们的判据阈值）；② 标 `未验证` 的项**保持未验证**，不因抽成表而升格；③ 不得把 IL 阶段的事写成 RL 阶段（跨阶段项必须落在 ⑧）；④ 口径冲突**并列不调和**（【R17】）；⑤ 描述性观察不得升格为判据（【R3】）。

## §0 本文件的结构、证据等级与解读约定

**「每族一张表」的落地方式**（两种读法都满足）：每族先给**一张族总表**（行 = 对象、列 = ①–⑧），再给**逐对象字段级明细表**（行 = ①–⑧、列 = 内容 + 证据/出处）。族总表的单元是压缩版，明细表是完整字段级正文。

| | 族 A | 族 B |
|---|---|---|
| 名称 | **RLHF 族** | **离线 IL + RL 微调族** |
| 成员 | InstructGPT（`2203.02155`）・Anthropic HH（`2204.05862`）・TRL `v0.11.2`・DeepSpeed-Chat | AWAC（`2006.09359`）・IQL（`2110.06169`）・Decision Transformer（`2106.01345`）・TD3+BC（`2106.06860`）・DAPG（`1709.10087`）・RLPD（`2302.02948`）・Cal-QL（`2303.05479`）・robomimic（`2108.03298`） |
| 辅助一手源（**非族成员**，只用于 ②/⑥ 的判据） | Askell 2021 `2112.00861`（HH 的 PM 损失真出处）・Stiennon 2020 `2009.01325`（KL 两个 purposes）・Gao 2022 `2210.10760`（KL 受控反证）・LIMA `2305.11206` | Paster et al. `2205.15967`（RvS 随机环境反证） |

**证据等级图例**（沿用盘上约定，不新建体系）：`[一手-论文原文]`・`[一手-官方代码]`・`[复现件-OA论文]`・`[二手-分析]`・`[未验证]`・`[本地-实测]`・`[本地-推理]`。
**本轮的独立复取标记**：`[本会话直取]` = 本子任务在 2026-09-19 用 `web_fetch`/`curl` 直取原文并逐字 grep 命中（与上游子报告的取数**互相独立**）；**未标**者 = 只依据上游盘上材料（其来源等级照抄，不升级）。
**单位口径**：权重/系数一律**按来源原样记录**并显式标注量纲（分母单位）；无单位者写「无单位」；**同名项量纲不同则并列不调和**，不折算、不换算、不对拍。
**⑧ 阶段的三个取值**：`IL 阶段` / `RL 阶段` / `跨阶段接口`（接口 = 数据、奖励、参考分布、条件变量在阶段之间的传递物）。

**编号作用域（防混引）**：本 H1（「对象 4 完整抽取正文」）之下的 `§0`–`§11` **自成一套编号**；其上的原文里另有 `§1`（勘误轮）与 `§G`，属**另一套**。**正文凡写 `§x` 一律指本 H1 之下者**；两套**不互相引用**。

---

## §1 族 A 总表 —— RLHF（列 = 固定 schema ①–⑧）

| 对象 | ① 奖励从哪来 | ② 惩罚项（KL 的 β 与口径；RM 分数加在哪） | ③ 塑形与调度 | ④ 归一化三问（奖励/价值/优势） | ⑤ value/critic | ⑥ 已知坑（带原文） | ⑦ 一手出处 | ⑧ 阶段归属 |
|---|---|---|---|---|---|---|---|---|
| **InstructGPT** | **人类偏好训出的 RM**（pairwise 排序损失 Eq.1，`(K choose 2)` 归一） | `per-token KL` to `π_SFT`，**β=0.02**；Eq.2 **无可见 Σ**、散文写 "at each token"；RM 分数是**序列级标量** `r_θ(x,y)`，**加在何处未写** ⇒ 不适用「末 token」写法 | **无奖励塑形/退火**；β 为常数（E.7 扫过 0 / 0.01 / 0.02 / 2） | **奖励**：RM 用 bias 归一使示范均分 0（原文）；**价值**：value function initialized from the RM（原文）；**优势**：GAE 细节**未取到** `未验证` | per-token value head，**初始化自 RM** | β=0 与 β=2 都差、最优 ≈0.01~0.02；β 提到 **2.0（100×）**仍修不好 alignment tax；RM 过优化 ⇒ **真偏好与 RM 反相关**（转引 Stiennon §4.3）；LIMA **1,000 条不做 RLHF 65% 胜 DaVinci003**；Gao 受控：**KL ≈ early stopping** 且**其余实验 KL=0** | `2203.02155v1` §3.1/§3.2/§3.5/§C.1–C.4/§E.6/§E.7 | IL：SFT（**论文无损失公式**）；RM：偏好建模；**RL**：Eq.2 的 KL 与 PPO；**接口**：RM 标量 → RL 奖励 |
| **Anthropic HH** | **PM（偏好模型）分数本身**；**无显式 IL/SFT 阶段**（PM 先在 LM 语料 PMP） | `r_total = r_PM − λ_KL·D_KL(policy‖policy_0)`（Eq.4.1，**序列级写法**），**λ_KL=0.001**；**per-token 还是序列级 = `[NOT-FOUND]`（不做推断）**；RM 分数位置 = 直接用 PM 分数当 RL 奖励（原文） | **无塑形/退火**；λ_KL 常数；**KL 系数消融 `[NOT-FOUND]`**（全文无 `ablation`） | **三问全部未取到** ⇒ `未验证`（本轮未核其 PPO 归一细节） | **未取到** ⇒ `未验证` | λ_KL=0.001「likely has a very minor impact … **might actually be wholly unnecessary**」；PM 在更高分处失校准（Fig.4 caption，约 150k 训练样本后）；PM 高分 ≠ 真人高评（§4.1/§B.4）；过优化 = 真偏好变差（§4.2） | `2204.05862v1` §3.1/§4.1/§4.2/§4.3/§A.2/§B.1/§B.4；PM 损失须引 Askell 2021 `2112.00861v3` §3.1 Eq.(3.1) | IL：**不存在**；RL：Eq.4.1；**接口**：PM 分数 → RL 奖励（PM 自身三阶段在 §A.2，≠ 助手三阶段） |
| **TRL `v0.11.2`** | **外部传入的 RM 分数** `score`（本库不训 RM） | **per-token**：`non_score_reward = -kl_ctl.value * kl`；**RM 分数只加在最后一个非 mask token**；`init_kl_coef=0.2`、`target=6.0`、`horizon=10000`；`target_kl=1.0` 是**另一个参**（早停门限）；`_kl_penalty` 四口径 `kl/abs/mse/full` | **有**：`AdaptiveKLController` 按比例误差调**系数**（`proportional_error = clip(current/target−1, −0.2, 0.2)`；`mult = 1 + err*n_steps/horizon`；`self.value *= mult`）；`FixedKLController` = 空操作；**调的是惩罚系数，不是奖励塑形/退火** | 文档写 `objective/non_score_reward = beta * kl.sum(1)`（**符号与代码相反**，代码逐 token 是**减** KL ⇒ 并列）；价值/优势归一 **未取到** | critic 由调用方提供（`value_model`），**本库不含 value 设计** | **负 KL 是失败前兆**（维护者原话）；「3 out of 10 experiments」出现负 KL 爆炸（issue #417）；「cares more about lowering the kl and stops boosting the reward」（#1178）；无 scorable completion 拿 **+0.264 advantage**（PR #6429，二手） | 代码 `trl/trainer/ppo_trainer.py` `_kl_penalty` **L1150–1162**、`compute_rewards` **L1135–1146**、负 KL 告警 **L1311–1317**；`ppo_config.py` **L161–181**；`utils.py` **L54–79**；文档源 `docs/source/ppo_trainer.md` @ `v0.15.2` | IL：Step 1 SFT；RL：KL 项；**接口**：`score` 入参 = RM→RL |
| **DeepSpeed-Chat** | **外部传入的 RM 分数** `reward_score`，先 `clamp(·, −5, +5)`（`clip_reward_value=5`） | **per-token（不跨 token 求和）**：`kl_divergence_estimate = -kl_ctl*(log_probs - ref_log_probs)`；**RM 分数只加在最后一个 response token**（`rewards[j,start:ends[j]][-1] += reward_clip[j]`）；`kl_ctl=0.1`（**硬编码**，注释 `# Those value can be changed`） | **无调度**（固定 `kl_ctl`）；`clip_reward_value=5` 是**奖励裁剪**，不是塑形/退火 | **奖励**：只有 ±5 裁剪；**价值/优势**归一 `[NOT-FOUND]`；⚠️「reward running average」= `[NOT-FOUND]`（grep `mean\|std\|moving\|average\|ema` 无命中；`main.py` import 了 `moving_average` 但**该文件取不到** ⇒ 不断言） | actor/critic 双损失（`actor_loss_fn`/`critic_loss_fn`），`cliprange_value=0.2`、`gamma=1.0`、`lam=0.95`；其中一行 `torch.clamp` 属**两次独立转写共识、非逐字节验证** ⇒ `未验证` | `kl_ctl` 硬编码；per-stage 详细文档**未取到**（U-11）；README 自述三阶段 | 代码 `applications/DeepSpeed-Chat/dschat/rlhf/ppo_trainer.py` 默认值 **L65–71**、`compute_rewards` **L181–194**；README（三阶段自述）；重命名 commit `e86d0c6d`（旧路径 `training/step3_rlhf_finetuning/ppo_trainer.py` 已不存在） | IL：Step 1 SFT；RL：Step 3；**接口**：Step 2 RM → Step 3 |

**族 A 的两条前置更正（必须与表同读）**：① InstructGPT **v1 全文只有 2 个编号公式**（Eq.1 RM 损失、Eq.2 RL 目标），**没有 Eq.3/4/5**，**也没有 SFT 损失公式**；② HH **没有写出 pairwise sigmoid 损失**，该公式在 **Askell et al. 2021 §3.1 Eq.(3.1)**；③ **TRL 的 `PPOTrainer` 已于 2026-09-04 从 `main` 移除**（commit `700b845c`，「Remove PPOTrainer (#7020)」）⇒ 本文件引的是**存档 tag `v0.11.2`**。

---

## §2 族 A 逐对象字段级明细（①–⑧）

### §2.1 InstructGPT（`arXiv:2203.02155v1`）

| 字段 | 内容 | 证据 / 出处 |
|---|---|---|
| **① 奖励从哪来** | **人类偏好训出的 RM**：`(K choose 2)` pairwise 排序损失，Eq.(1)。RM 是**学出来的代理**；RL 的奖励 = RM 标量（+ 可选的预训练梯度项，见 ②）。 | `[一手-论文原文]` <https://arxiv.org/html/2203.02155v1> §3.5 Reward modeling（Eq.1）；**本轮直取命中** |
| **② 惩罚项** | `objective(φ) = E[r_θ(x,y) − β·log(π_RL(y\|x)/π_SFT(y\|x))] + γ·E[log π_RL(x)]`（Eq.2）。**逐字理由**：「In addition, we add a **per-token KL penalty** from the SFT model **at each token** to **mitigate over-optimization of the reward model**.」**β=0.02**（§C.4）。**注意三点**：㈠ 公式里**没有可见的 Σ**，逐 token 只见于散文；㈡ `r_θ(x,y)` 是**序列级标量**，**加在哪个位置论文未写**（**不要**照 TRL/DS-Chat 的「末 token」写法回填）；㈢ `γ=27.8` 是**预训练 mix 项**，不是 KL（PPO 模型 γ=0）。 | `[一手-论文原文]` 同上 §3.5（Eq.2 + 理由句）、§C.4（`β=0.02`、`γ=27.8`）；**本轮直取命中**（"per-token KL penalty … at each token"、`β=0.02`） |
| **③ 塑形与调度** | **无奖励塑形、无退火**；β 是常数。§E.7 的 β 扫描**不是调度**（是超参选择）。 | `[一手-论文原文]` 同上 §E.7 |
| **④ 归一化三问** | **奖励**：RM loss 对平移不变 ⇒ 用 bias 归一使**示范均分 0**（「we normalize the reward model using a bias so that the labeler demonstrations achieve a mean score of 0 before doing RL」，§3.5）。**价值**：「The value function is **initialized from the RM**」（§3.5）。**优势**：GAE / 优势归一细节 **本轮未取到** ⇒ `未验证`。 | `[一手-论文原文]` 同上 §3.5；优势一项 `未验证` |
| **⑤ value/critic** | per-token value head，**初始化自 RM**（这一条是本报告里唯一明确到 ⑤ 的 InstructGPT 事实）。 | `[一手-论文原文]` 同上 §3.5 |
| **⑥ 已知坑（带原文）** | ㈠ **β 的最优区间窄且两端都坏**：`"Both 0 and 2 for KL reward coefficient result in poor performance. The optimal value is around 0.01 and 0.02."`（§E.7）。㈡ **加大 KL 救不了回退**：`"even by increasing the KL reward coefficient to 2.0, which is 100 times of the default value, the regressions still cannot be fixed."`（§E.6）㈢ **KL 只等价早停（同一研究组受控实验，必须并列）**：`"The KL penalty only causes the gold RM score to converge earlier, but does not affect the KL_RL-gold reward frontier, and so the effect of the penalty on the gold score is **akin to early stopping**"`，且 `"Because we observe that using KL penalty has a **strictly larger proxy-gold gap**, we set **KL penalty to 0** for all other RL experiments in this paper."`（Gao `2210.10760v1` §3.6）。㈣ **学出来的奖励会与真人偏好反相关**（转引 Stiennon §4.3）。㈤ **LIMA**：`"fine-tuned with the standard supervised loss on only **1,000 carefully curated** prompts and responses, **without any reinforcement learning or human preference modeling**"`，且 `"as high as … 58% when compared to Bard and **65% versus DaVinci003, which was trained with human feedback**"`、`"surprisingly, doubling the training set does not improve response quality"`。 | `[一手-论文原文]` InstructGPT §E.6/§E.7；Gao **本轮直取命中**（三处逐字）；LIMA `2305.11206` 摘要/正文，**本轮直取命中**；Stiennon 为**转引**（标二手归属） |
| **⑦ 一手出处** | <https://arxiv.org/html/2203.02155v1> §3.1（三阶段原文）、§3.2（13k/33k/31k prompts）、§3.5（Eq.1/Eq.2/理由句/RM 归一/value 初始化）、§C.1–C.4（超参、β=0.02、γ=27.8）、§E.6（β=2.0 反证）、§E.7（0/0.01/0.02/2）；取数 2026-09-19（本子任务直取）。 | — |
| **⑧ 阶段归属** | **IL 阶段**：SFT（**论文无损失公式**；「IL 损失 = 对示范的最大似然」的可靠出处是 Stiennon 2020 §1，属**转引**）。**RM**：偏好建模（数据侧）。**RL 阶段**：Eq.2 的 KL 项与 PPO。**跨阶段接口**：RM 标量 → RL 奖励、`π_SFT` → KL 的参考分布。 | `[一手-论文原文]` + `[NOT-FOUND]`（SFT 损失式） |

### §2.2 Anthropic HH（`arXiv:2204.05862v1`；PM 损失真出处 = Askell 2021 `2112.00861`）

| 字段 | 内容 | 证据 / 出处 |
|---|---|---|
| **① 奖励从哪来** | **PM（偏好模型）分数本身**。逐字（**本轮直取命中**）：`"Throughout this paper we use r_PM = the preference model score itself for the RL reward."` HH **无显式 IL/SFT 阶段**（PM 先在 LM 语料上预训练）。 | `[一手-论文原文]` <https://arxiv.org/html/2204.05862v1> §4.1 附近；**本轮直取命中** |
| **② 惩罚项** | `r_total = r_PM − λ_KL·D_KL(policy ‖ policy_0)`（**Eq.(4.1)**，**序列级写法**），`λ_KL ≥ 0` 为超参；**λ_KL = 0.001**（§B.1）。**逐字自评（本轮直取命中）**：`"In practice we use a very small value of λ_KL = 0.001, which likely has a very minor impact during most of RL training (as D_KL < 100 typically), and **might actually be wholly unnecessary**."` ⚠️ **per-token 还是序列级 = `[NOT-FOUND]`**：全文 grep `per-token` / `at each token` / `sum over tokens` = **0 命中** ⇒ **不做推断**（不因 "D_KL < 100 typically" 去反推口径）。**RM 分数位置**：HH 用自己的话说 `r_PM` **就是** RL 奖励（不是「在某 token 上加一次」）。 | `[一手-论文原文]` 同上 §4.1（Eq.4.1 + λ_KL=0.001 + wholly unnecessary 句）、§4.3（`D_KL` 的经验估计说明）、§B.1（`λ_KL=0.001`、`ϵ=0.2`、`γ=1`、**no entropy bonus**）；**本轮直取命中（λ_KL / wholly unnecessary / Eq.4.1）** |
| **③ 塑形与调度** | **无奖励塑形、无退火**；λ_KL 常数。**KL 系数消融 `[NOT-FOUND]`**（全文无 `ablation` 一词，只有 §B.1「We performed a variety of hyperparameter scans…」）。⚠️ **别名冲突**：HH 另有 `L_Total = L_Helpfulness + λ·L_Harmlessness`（λ∈{1,2,3,4,10}）—— 那是**损失混合权重**，**不是 KL**，不得混引。 | `[一手-论文原文]` 同上 §B.1；消融 `[NOT-FOUND]` |
| **④ 归一化三问** | **奖励 / 价值 / 优势三问本轮均未取到** ⇒ `未验证`（本轮**只**核实了 Eq.4.1、λ_KL、自评句、PM=奖励；PPO 的归一细节未核）。 | `未验证`（本轮未核） |
| **⑤ value/critic** | **未取到** ⇒ `未验证`。 | `未验证`（本轮未核） |
| **⑥ 已知坑（带原文）** | ㈠ **作者自评 KL 可能完全没必要**（见 ②）。㈡ **PM 在更高分处失校准**：`"the train and test PM's disagree, with the train PM assigning a higher mean reward"`（Fig.4 caption，**约 150k 训练样本之后**）。㈢ **PM 高分 ≠ 真人高评**：`"PMs also become less calibrated at higher scores, so higher rewards do not necessarily imply better performance."`（§4.1）；§B.4：`"the naive PM predictions significantly overestimate the empirical Elos"`。㈣ **过优化 = 真偏好变差**：`"The divergence is likely an indication that the preference model is less robust and more easily exploited at higher rewards. That is, the policy has been over-optimized on the train PM"`（§4.2）。㈤ **必须另引**：HH 的 PM 损失公式**不在本文**，在 **Askell 2021 §3.1 Eq.(3.1)**：`L_PM = log(1 + e^{r_bad − r_good})`（等价 `−log σ(r_good − r_bad)`），「for batched sample pairs we take the mean over all pairs」；HH §3.1 自称 `"Our preference model training setup is also identical to that in [Askell et al., 2021]"`。 | `[一手-论文原文]` HH §4.1/§4.2/§B.4/Fig.4 caption；Askell `2112.00861v3` §3.1 Eq.(3.1) |
| **⑦ 一手出处** | HH <https://arxiv.org/html/2204.05862v1> §3.1、§4.1、§4.2、§4.3、§A.2、§B.1、§B.4、Fig.4 caption（PDF 74 页）；Askell <https://arxiv.org/html/2112.00861v3> §3.1；取数 2026-09-19（HH 关键四句为本轮直取命中）。 | — |
| **⑧ 阶段归属** | **IL 阶段：不存在**（HH 助手流程在 §4.1 只写成**两步** PM→RLHF）。**RL 阶段**：Eq.4.1 的 KL 与 PPO。**跨阶段接口**：PM 分数 → RL 奖励；`policy_0`（初始策略）→ KL 的参考分布。⚠️ **PM 自己的三阶段（§A.2：LM 预训练 → PMP → 人类反馈微调）不是助手的三阶段**，不得混写。 | `[一手-论文原文]` 同上 §4.1 / §A.2 |

### §2.3 TRL `v0.11.2`（`PPOTrainer`；`main` 上已于 2026-09-04 移除）

| 字段 | 内容 | 证据 / 出处 |
|---|---|---|
| **① 奖励从哪来** | **RM 分数以 `score` 入参传入**（本库不训 RM）；奖励 = `score`（末 token 加一次）+ 逐 token 的 KL 惩罚。 | `[一手-官方代码]` `trl/trainer/ppo_trainer.py` `compute_rewards` **L1135–1146** |
| **② 惩罚项** | **per-token**：`kl = self._kl_penalty(logprob, ref_logprob)`；`non_score_reward = -self.kl_ctl.value * kl`；`reward = non_score_reward.clone()`；**RM 分数只加在最后一个非 mask token**：`reward[last_non_masked_index] += score`。`_kl_penalty` **四种口径**：`kl`（`logprob − ref_logprob`）、`abs`、`mse`（`0.5·(·)²`）、`full`（`F.kl_div(..., reduction="none").sum(-1)`）。**默认 β**：`init_kl_coef = 0.2`（`ppo_config.py:161`）、`target = 6.0`、`horizon = 10000.0`；**`target_kl = 1.0` 是另一个参**（早停门限，docstring「Stop early if we exceed this value by over 50%」，`ppo_config.py:181`）—— **不是自适应 KL 的目标**。 | `[一手-官方代码]` `ppo_trainer.py` **L1150–1162**（`_kl_penalty`）、**L1135–1146**（`compute_rewards`）；`ppo_config.py` **L161–181**；Git blob SHA 逐字节核对 `42f916dd…`（上游） |
| **③ 塑形与调度** | **有自适应控制器**：`AdaptiveKLController.update`：`proportional_error = np.clip(current / target − 1, −0.2, 0.2)`；`mult = 1 + proportional_error * n_steps / self.horizon`；`self.value *= mult`（`utils.py` **L54–69**，docstring 指向 HF 论文 `1909.08593`）。更新调用：`self.kl_ctl.update(stats["objective/kl"], batch_size * num_processes)`（`ppo_trainer.py` **L881–884**）。`FixedKLController.update` = **空操作**（`utils.py` **L72–79**），选择逻辑在 `ppo_trainer.py` **L319–322**。⭐ **重要区分**：这调的是**惩罚系数的调度**，**不是奖励塑形/退火**。 | `[一手-官方代码]` 同上 |
| **④ 归一化三问** | **奖励**：文档把 `objective/non_score_reward` 定义成 `beta * kl.sum(1)`（**per-token KL 的求和**），`objective/rlhf_reward = score − non_score_reward` —— ⚠️ **文档符号与代码相反**（代码里逐 token 是**减** KL）⇒ **并列不调和**。**价值/优势**：critic 由调用方提供，**本库不含相应归一设计** ⇒ 不适用。 | `[一手-官方代码]` 文档源 `docs/source/ppo_trainer.md` @ `v0.15.2`；代码 `compute_rewards` |
| **⑤ value/critic** | **本库不含 value 设计**：`value_model`（critic）由调用方传入；本库也不定义奖励归一。 | `[一手-官方代码]`（以本轮核到的调用形态为限） |
| **⑥ 已知坑（带原文）** | ㈠ **负 KL 是失败前兆**（维护者 lvwerra 亲述）：`"A common failure mode of the training is that the generation kwargs are not correctly set which leads to **negative KL-divergence which is almost always a precursor for failed training**."`（issue #235，**[二手]**）。TRL 代码自身也告警：`if mean_kl.item() < -1.0: warnings.warn(f"KL divergence is starting to become negative: … this might be a precursor for failed training.")`（`ppo_trainer.py` **L1311–1317**）。㈡ 实测失败率：`"I noticed **3 out 10 experiments** would experience some amount of **negative KL explosion**, though one of them recovered"`（issue #417，**[二手]**）。㈢ **KL 压倒奖励**：`"after step12 it looks like the model cares more about lowering the kl and stops boosting the reward"`（issue #1178，配置 `init_kl_coef: 0.2, target: 6, target_kl: 1`，**[二手]**）。㈣ **reward hacking 的工程数字**：`"The unscorable completion (idx 7) gets a **+0.264 advantage** without the defense — a positive policy gradient despite carrying no learning signal."`（PR #6429，**[二手]**）。㈤ **文档 ↔ 代码符号相反**（见 ④）。 | `[一手-官方代码]` + `[二手-issue/PR]`（逐条已标） |
| **⑦ 一手出处** | 代码：<https://github.com/huggingface/trl/blob/v0.11.2/trl/trainer/ppo_trainer.py>、`.../ppo_config.py`、`.../utils.py`；文档源：<https://github.com/huggingface/trl/blob/v0.15.2/docs/source/ppo_trainer.md>；移除 commit `700b845c5d4b1cfd292cf9c067626d8a19fc0fe1`（2026-09-04）。⚠️ 官方文档页 `https://huggingface.co/docs/trl/main/en/ppo_trainer` **本环境不可访问**（U-A14）。 | — |
| **⑧ 阶段归属** | **IL 阶段**：Step 1 SFT（本库另有）。**RL 阶段**：KL 项与 PPO。**跨阶段接口**：`score` 入参（RM→RL）、`ref_model`/`ref_logprobs`（参考分布，`ppo_trainer.py` **L745–763**，`torch.no_grad()` 前向）。 | `[一手-官方代码]` |

### §2.4 DeepSpeed-Chat（`deepspeedai/DeepSpeedExamples`，`master`）

| 字段 | 内容 | 证据 / 出处 |
|---|---|---|
| **① 奖励从哪来** | **Step 2 训出的 RM 分数** `reward_score`；先裁剪 `reward_clip = torch.clamp(reward_score, -self.clip_reward_value, self.clip_reward_value)`（±5）。 | `[一手-官方代码]` `applications/DeepSpeed-Chat/dschat/rlhf/ppo_trainer.py` `compute_rewards` **L181–194** |
| **② 惩罚项** | **per-token（不跨 token 求和）**：`kl_divergence_estimate = -self.kl_ctl * (log_probs - ref_log_probs)`；`rewards = kl_divergence_estimate`；**RM 分数只加在最后一个 response token**：`rewards[j, start:ends[j]][-1] += reward_clip[j]`。**默认值**（`# Those value can be changed`）：`self.kl_ctl = 0.1`、`clip_reward_value = 5`、`cliprange = 0.2`、`cliprange_value = 0.2`、`gamma = 1.0`、`lam = 0.95`。 | `[一手-官方代码]` 同上 默认值 **L65–71**、`compute_rewards` **L181–194** |
| **③ 塑形与调度** | **无调度**（`kl_ctl` 固定）；`clip_reward_value = 5` 是**奖励裁剪**，不是塑形/退火。 | `[一手-官方代码]` 同上 |
| **④ 归一化三问** | **奖励**：只有 ±5 裁剪（**这是本族唯一写死在代码里的奖励尺度处理**）。**价值/优势**：`[NOT-FOUND]`。⚠️ **「reward running average」= `[NOT-FOUND]`**：`ppo_trainer.py` 全文 grep `mean\|std\|moving\|average\|ema` **无命中**；仅见 Step-3 `main.py` import 了 `moving_average`/`ExponentialMovingAverage`（来自 `dschat.utils.utils`），但**该文件取不到**（GitHub API 限流 + raw 不可达）⇒ **不断言**。 | `[一手-官方代码]` + `[NOT-FOUND]` |
| **⑤ value/critic** | actor/critic 双损失（`actor_loss_fn` / `critic_loss_fn`）；`cliprange_value=0.2`、`gamma=1.0`、`lam=0.95`。⚠️ 其中一行 `torch.clamp` 属**两次独立转写共识、非逐字节验证** ⇒ `未验证`。参考模型：`self.ref_model = self.rlhf_engine.ref`；`'ref_logprobs': gather_log_probs(logits_ref[:, :-1, :], seq[:, 1:])`（同文件 **L154–174**）。 | `[一手-官方代码]` + 一行 `未验证` |
| **⑥ 已知坑** | 以本轮取回文本为限：㈠ `kl_ctl` **硬编码**在源码里（注释自承「Those value can be changed」）；㈡ per-stage 详细文档**未取到**（`training/README.md` 限流）⇒ Step-3 双损失的官方描述 `未验证`；㈢ 旧文档路径 `training/step3_rlhf_finetuning/ppo_trainer.py` **已不存在**，commit `e86d0c6d`（2023-11-06）rename 到 `dschat/rlhf/ppo_trainer.py` ⇒ **引用旧路径即为失效引用**。 | `[一手-官方代码]`；`未验证`（per-stage 文档） |
| **⑦ 一手出处** | <https://github.com/deepspeedai/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/dschat/rlhf/ppo_trainer.py> **L65–71 / L154–194**；README（三阶段自述）；commit `e86d0c6d38d65158836c275e2d7b6a8f8e92f026`。 | — |
| **⑧ 阶段归属** | **IL 阶段**：Step 1 SFT；**RM**：Step 2；**RL 阶段**：Step 3（KL 与 PPO）；**跨阶段接口**：Step 2 RM → Step 3 的 `reward_score`、`rlhf_engine.ref` → 参考分布。⭐ 本族里**唯一**把「逐 token KL」与「RM 分数位置」都写死在代码里、可点击核对的实现。 | `[一手-官方代码]` |

---

## §3 族 B 总表 —— 离线 IL + RL 微调（列 = 固定 schema ①–⑧）

| 对象 | ① 奖励从哪来 | ② 惩罚项 | ③ 塑形与调度 | ④ 归一化三问 | ⑤ value/critic | ⑥ 已知坑（带原文） | ⑦ 一手出处 | ⑧ 阶段归属 |
|---|---|---|---|---|---|---|---|---|
| **AWAC** | **数据集内 reward**（不建 RM）；奖励**只经 critic 变标量优势**，再当模仿项权重 | **`KL(π‖π_β) ≤ ε` 是推导前提（Lagrangian）**，落成实现后表现为**优势加权**。**两套约定并列**：论文 `exp((1/λ)A)`（λ=0.3 manipulation / 1.0 MuJoCo）vs **参考实现默认批内 softmax**（rlkit `normalize_over_batch=True`、jaxrl 只有 batch-softmax） | **无退火**；λ 常数；**实践中丢弃 Z(s)**（消融 pen 84%→98%、door 0%→95%、relocate 0%→54%） | **奖励**：不适用（只进 critic）；**价值**：TD 目标；**优势**：**归一化就落在权重里**（softmax = 批内归一）。⚠️ 论文附录有两处内部不一致（Z(s) 积分内用 π_θ 而非 π_β、期望式漏 exp） | twin Q（TD3 式，rlkit）；不做 expectile；**不做 OOD 回避**（那正是 KL 前提要做的事） | 见 ② 的「论文形式 ≠ 实现默认」；附录排版/笔误级不一致；`[本地-推理]` 零样本区退化（优势在无样本处无定义） | `2006.09359`（e-print 逐字）+ rlkit `awac_trainer.py` **L618–631** + jaxrl `awac/actor.py`；**本轮未复取** | RL 阶段（离线/在线微调）；**接口**：数据集 `β(a\|s)` → 加权项、critic → `A(s,a)` |
| **IQL** | **数据集内 reward**；不建 RM | **无显式惩罚**；机制 = **非对称 expectile 的 V + 优势加权回归**：`L_π(φ) = E[exp(β(Q_θ̂(s,a) − V_ψ(s)))·log π_φ(a\|s)]`；τ=0.9/β=10.0（AntMaze）、τ=0.7/β=3.0（locomotion）。优势权重形式 = `exp(βA)`（论文） | **无退火**；τ、β 常数 | **奖励**：未取到；**价值**：expectile 本身是**价值目标的非对称加权**（τ 定分位）；**优势**：`A = Q − V`，用同一 critic 内的 V ⇒ 天然标量化 | **expectile V** + twin Q；**刻意不查询数据集外动作**（这是「不需要 BC 惩罚项」的机制来源；⚠️ 本轮**未取回该设计理由的逐字原句**） | `[上游-一手逐字]` D4RL locomotion 合计 IQL **692.4** > BC **466.7**（**本轮未复现**）；IQL 官方实现是否也用 softmax **未核** | `2110.06169`（e-print 逐字，上游）；`github.com/ikostrikov/implicit_q_learning`；**本轮未复取** | RL 阶段（离线 RL）；**接口**：数据集 → expectile 目标与加权项 |
| **Decision Transformer** | **不建 critic 的条件回报**：奖励**只被用来算 return-to-go** `R̂_t = Σ_{t'≥t} r_{t'}`，再作为**输入 token** | **无**（无 KL、无 BC 正则、无 critic、无 bootstrap） | **无训练调度**；测试时 `target_return` 起手并**逐步 decrement**（推理期条件，不是调度） | **奖励**：`scale=1000.`（**代码**；**论文 `[NOT-FOUND]`** ⇒ 并列）；**价值/优势**：**N/A**（不存在 critic） | **无 critic**（核心设计） | `[本轮直取命中]` DT Table 3 Average **DT 56.1 vs 10%BC 56.7**；原文限定「When data is plentiful … %BC can match or beat other offline RL methods」；**Paster 反证**：`"can fail dramatically in stochastic environments"`、`"this lack of performance … is **not due to a lack of data**"`、Connect Four `"an RvS agent cannot achieve more than 0.2 average return (corresponding to a win-rate of 60%)"` | `2106.01345` Table 3（**本轮直取命中**）；Paster `2205.15967` 摘要/§3.3（**本轮直取命中**）；`scale=1000.` 见上游（`未验证`） | **IL 阶段（纯监督动作预测）**；**无 RL 阶段**；**接口**：return-to-go 条件 token |
| **TD3+BC** | **数据集内 reward** → critic（TD3 式）；不建 RM | **BC 正则项**（同时是锚定项）：`π = argmax_π E[λ·Q(s,π(s)) − (π(s)−a)²]`，**`λ = α / Σ\|Q(s,a)\|`**（原文强调这是 normalize by the **average absolute value** 的启发式）；**α=2.5**（在 1/2/2.5/3/4 上调过） | **不退火**（α 固定）；λ 随 `Σ\|Q\|` 自适应 —— 是**归一化**，不是调度 | **奖励/价值**：**λ 的分母 `Σ\|Q\|` 就是对价值尺度的归一**（全族里最直接回答 ④ 的机制）；**优势**：无显式优势（直接最大化 Q） | TD3 式 twin critic + target smoothing；不做 expectile；**不回避 OOD**（BC 项就是 OOD 抑制） | α 敏感（调过 5 个值）；**λ 含/不含 `1/N` 两套并存**（`未验证`）；在人类数据上可能仍远弱于时序 BC（robomimic 表内无 TD3+BC ⇒ **N/A**，不得替它下结论） | `2106.06860`（ar5iv 逐字，上游）；**本轮未复取** | RL 阶段（离线）；**接口**：数据集动作 `a` → BC 正则；`\|Q\|` → 自适应 λ |
| **DAPG** | 演示数据上的**辅助 BC 损失** + 环境 reward（on-policy PG：NPG/TRPO） | **辅助项权重同时含优势加权与退火**：`w(s,a) = λ₀·λ₁^k·max_{(s',a')~ρ_π} A^π(s',a')`（k = 迭代计数），**λ₀=0.1、λ₁=0.95**。原文动机逐字：`"we asymptotically decay the auxiliary objective"` | **有**：`λ₁^k` **逐迭代渐近退火**辅助 BC 项；流程 = 先 **BC 预训练**再以该 augmented loss 微调 | **奖励**：未取到；**价值**：NPG 的 on-policy value；**优势**：权重里含 **`max A^π`**（一个批内/轨迹内的最大化归一形态） | on-policy value function；不做 expectile；**无 OOD 约束**（on-policy 采样自然覆盖） | ⚠️ **引用纪律**：上一稿误引 `1910.10314`（实为 *The pure-quartic soliton laser*）⇒ **DAPG 真身 = `1709.10087`**；λ₀/λ₁ 敏感度未取到 ⇒ `未验证` | `1709.10087` §IV-C2（ar5iv 逐字，上游）；**本轮未复取** | **跨阶段接口**：BC 预训练（IL）→ RL 微调；辅助项把演示留在 RL 损失里并**退火** |
| **RLPD** | **数据集内 reward**（offline buffer + online replay）；**不建 RM、无 BC 项** | **明确没有 BC 惩罚**。逐字（§2「Constraining to prior data」，第一人称自述，与 Nair et al. 2018a / DDPGfD 对比）：`"However, we **do not** restrict the policy using a behavior cloning term, and do not reset to demonstration states."` 旁证：Intro `"without offline RL pre-training or explicit imitation terms that privilege the prior offline data?"`；§4 `"an approach based on off-policy model-free RL, without pre-training or explicit constraints."` **无 KL to reference** | **无奖励塑形/退火**；「调度」只有**对称采样 50/50**（每个 batch 一半 replay、一半 offline）与 **UTD `G`**（≠ 奖励调度） | **核心就在 ④**：靠 **critic 内的 LayerNorm 当「界」** —— `"LayerNorm **bounds** the values and empirically prevents catastrophic value extrapolation"`、`"it **does not explicitly constrain** the policy to remain close to the offline data"`；**奖励归一** `未验证`；**优势**：SAC(MaxEnt)，actor 目标含 `−α log π`（Algorithm 1 line 17/23，**本轮直取命中**）；CDQ 取子集 `Z∈{1,2}` + 大 ensemble `E` | **twin/大 ensemble critic + LayerNorm = 一个「界」，不是惩罚项**。逐字（§4.2）：`"instead of creating a mechanism that **explicitly discourages OOD actions**, which can be viewed as **anti-exploration**, we instead need to simply ensure that the learned functions do not extrapolate in an unconstrained manner"` | ⚠️ **同名机制、相反作用**：本仓 **O2** 已证**价值头末端 LayerNorm 是 critic 死亡的第一性原因之一**（近常数输入被归一化 ⇒ 价值头可表达跨帧方差 ≈0.14 而目标 std≈12.11，ReLU 逐个死亡锁死；实测 `value` 唯一值 1/600、std=0）⇒ **不能因「RLPD 用了 LayerNorm」就照搬**；这条属**本地事实，须自证，不得引 RLPD 当依据** | `[本会话直取]` <https://ar5iv.labs.arxiv.org/html/2302.02948> §2 / §4 / §4.2 / Algorithm 1（HTTP 200，2026-09-19） | **RL 阶段**（在线 off-policy 从零训）；**接口**：offline buffer + 对称采样 = 跨阶段数据接口，**不是 IL 损失** |
| **Cal-QL** | **数据集内 reward** → CQL 离线预训练 → 在线微调 | **无 IL 惩罚、无 BC 项**（**verified-by-absence**，上游 U14 ⇒ 保留为 `未验证`）。机制 = **保守价值下界（CQL）的校准**，不是惩罚项、也不是锚定到行为 | **无奖励退火**；校准发生在**离线→在线切换点**（阶段切换，不是退火） | **价值**：校准 = 让 Q 的尺度在离线与在线之间**对齐**（正是 ④ 的「价值归一化」问）；**奖励/优势**归一 `未验证` | CQL critic + 保守下界校准；不做 expectile | **本轮取回文本未记录作者的失败/坑条目** ⇒ 记「⑥：本轮未取到」，**不臆造** | `2303.05479` Eq. 5.1（ar5iv 逐字，上游）；**本轮未复取** | **跨阶段接口**：离线 CQL 预训练 → 在线 RL；**IL 阶段不存在** |
| **robomimic** | **数据集内 reward**（离线人类示范；部分数据集带 reward label）；**不建 RM**。其定位是**评测/研究协议**，不是单一算法 | **不是「一个奖惩设计」**：它把 ② 的差异**暴露成对照实验** —— BC/BC-RNN 纯 NLL 无惩罚；BCQ/CQL 有约束/保守项；HBC/IRIS 有时序抽象 | **无塑形/退火**；作者指出 **(C4) 训练目标只是替代指标、策略表现逐 epoch 波动大 ⇒ 模型选择困难**（= 「目标 vs 真目标分离」的实证） | 各算法的归一化细节**未逐条取到** ⇒ `未验证`；作者的核心 ④ 结论是 **(C5) 超参敏感** | 按算法而异（BCQ/CQL 有 critic，BC/BC-RNN 没有）。作者结论：**时序建模（BC-RNN/HBC/IRIS）在人类数据上强**，Batch RL（BCQ/CQL）在人类数据上差 | `[本轮直取命中]` Table 1「Square (MH)」：BC **52.7±6.6** / **BC-RNN 78.0±4.3** / **BCQ 14.0±4.3** / **CQL 0.7±0.9**；Table 3 说明逐字：`"methods that model **temporal correlations** (BC-RNN, HBC, IRIS) exhibit strong performance on **human datasets**. Furthermore, while Batch RL algorithms like BCQ are proficient on machine-generated data, they perform poorly on human datasets."` | `[本会话直取]` <https://arxiv.org/html/2108.03298v1> Table 1 / Table 3（HTTP 200，2026-09-19） | **IL 阶段**（BC 家族）与**离线 RL 阶段**（BCQ/CQL）**并列**；**接口** = 数据集与评测协议 |
| **（辅助反证）Paster `2205.15967`** | 环境 reward，用**条件回报**（RvS）；不建 RM | **无惩罚** | 无 | **奖励**：条件化在**轨迹回报**上（非期望）⇒ 归一问题 | 无 critic（RvS 对照）；其 ESPER 用 cluster 平均回报 | `[本轮直取命中]` 摘要：`"can **fail dramatically** in stochastic environments since trajectories that result in a return may have only achieved that return due to luck"`；Figure 1 caption：`"will fail, **even with infinite data**"`；§3.3：`"Figure 6 shows that **this lack of performance … is not due to a lack of data**"`；Connect Four：`"an RvS agent cannot achieve more than **0.2 average return** (corresponding to a win-rate of **60%**)"` | `[本会话直取]` <https://arxiv.org/html/2205.15967v1> 摘要 / Fig.1 / §3.3（HTTP 200，curl+去标签+grep，2026-09-19） | IL 阶段（RvS 对照）；**结论作用于「以回报为条件」的跨阶段接口设计** |

---

## §4 族 B 逐对象补充明细（逐字引用 / 口径 / 本地落点）

> **读法说明**：**§3 的每一行就是该对象的 ①–⑧ 全覆盖**（固定 schema 的字段级正文）。本节**只补**三类 §3 装不下的东西：㈠ **逐字引用**；㈡ **两套约定的原文对照**；㈢ **与我们仓的落点**。**不重复 §3 的字段内容**，因此本节的行**不是**每个对象的 ①–⑧ 全表。

### §4.1 AWAC —— ② 的两套约定（**并列不调和**）

| 约定 | 逐字 / 形式 | 出处 |
|---|---|---|
| **论文形式** | KKT 闭式解 `π*(a\|s) = (1/Z(s))·π_β(a\|s)·exp((1/λ)A^{π_k}(s,a))`；投影到参数化策略（前向 KL）：`θ_{k+1} = argmax_θ E_{s,a~β}[ log π_θ(a\|s)·exp((1/λ)·A^{π_k}(s,a)) ]`；**λ=0.3（manipulation）/ 1.0（MuJoCo）** | `[一手-论文原文]` `arXiv:2006.09359`（e-print 逐字，上游）；**本轮未复取** |
| **参考实现默认** | rlkit `normalize_over_batch` **默认 True** ⇒ `weights = F.softmax(score/beta, dim=0)`（**批内 softmax**）；jaxrl 的 AWR **只有** batch-softmax，并留注释：`"exp(a/beta) is unbiased but high variance, softmax(a/beta) is biased but lower variance"` | `[一手-官方代码]` `github.com/vitchyr/rlkit` `rlkit/torch/sac/awac_trainer.py` **L618–631**；`github.com/ikostrikov/jaxrl` `jaxrl/agents/awac/actor.py`；**本轮未复取** |
| **④ 的接口** | 两套约定**分子/分母不同**（`exp` 的归一常数 vs 批内 softmax 的批和）⇒ 按【R17】必须**报「两套约定」并只采信方向一致的部分** | — |
| **Z(s) 的处理** | **实践中丢弃 Z(s)**；消融（丢弃后）：pen 84%→98%、door 0%→95%、relocate 0%→54% | `[一手-论文原文]` 同上（上游记录） |

### §4.2 IQL —— expectile 与「刻意不查询数据集外动作」

| 项 | 内容 | 出处 |
|---|---|---|
| **⑤ expectile 形式** | `L_V` 用**非对称 expectile**（τ>0.5 上偏），`τ=0.9/β=10.0`（AntMaze）、`τ=0.7/β=3.0`（locomotion）；`L_π(φ) = E_{(s,a)~D}[ exp(β(Q_θ̂(s,a) − V_ψ(s))) · log π_φ(a\|s) ]` | `[一手-论文原文]` `arXiv:2110.06169`（e-print 逐字，上游）；**本轮未复取** |
| **⑤ 设计理由（⚠️ 非逐字）** | 「**刻意不查询数据集外动作**」是本报告与上游合成文档的**表述**，指 IQL 用 expectile + 只用数据集动作做 Q 学习，从而**不需要**在策略改进时查询 OOD 动作；⚠️ **本轮未取回该设计理由的逐字原句** ⇒ 只作机制描述，**不得当引用原文**（见 §9.1 B4） | `[上游-分析]` + `未验证`（逐字原句） |
| **② 与 AWAC 的差别** | IQL **没有** AWAC 的 KL 前提；它的「约束」被**吸收进 expectile 的价值目标**里 ⇒ 这是「同一目的、两种落点」的并列对照 | `[一手-论文原文]`（形式层） |
| **⑥ 数字** | 上游记录：D4RL locomotion 合计 **IQL 692.4 > BC 466.7**；⚠️ **本轮未独立复现** ⇒ 保留 `未验证` 标记（§9.2） | `[上游-一手逐字]` |

### §4.3 RLPD —— ⑤ 的逐字原话（「界」而非惩罚项）★ 本轮独立复取

| 项 | 逐字 | 出处 |
|---|---|---|
| **不做 BC 惩罚** | `"However, we do not restrict the policy using a behavior cloning term, and do not reset to demonstration states."`（§2「Constraining to prior data」，**第一人称自述**，与 Nair et al. 2018a / DDPGfD 对比） | `[本会话直取]` ar5iv `2302.02948` §2 |
| **不是离线 RL 方法** | `"our normalized update is not an offline RL method—we do not perform any offline pre-training but run online RL from scratch with offline data included in a replay buffer."` ⚠️ **OCR 存疑**：`normalized update` 疑为 `method`（该句在 ar5iv 渲染中如此）；引用时**并列标注** | `[本会话直取]` 同上 §2「Offline RL pre-training」段 |
| **无预训练或显式约束** | `"an approach based on off-policy model-free RL, without pre-training or explicit constraints"` | `[本会话直取]` 同上 §4 开头 |
| **LayerNorm = 界** | `"we demonstrate that LayerNorm **bounds** the values and empirically prevents catastrophic value extrapolation"`；`"it **does not explicitly constrain** the policy to remain close to the offline data"` | `[本会话直取]` 同上 §4.2（标题即 "Design Choice 2: Layer Normalization Mitigates Catastrophic Overestimation"） |
| **不做 OOD 抑制的理由** | `"instead of creating a mechanism that explicitly discourages OOD actions, which can be viewed as anti-exploration, we instead need to simply ensure that the learned functions do not extrapolate in an unconstrained manner"` | `[本会话直取]` 同上 §4.2 |
| **算法侧（本轮逐字命中）** | MaxEnt actor 目标 `(1/E)Σ_i Q_{θ_i}(s,ã) − α log π_φ(ã\|s)`（Algorithm 1 line 23）；TD 目标加 `γα log π_φ(ã'\|s')`（line 17）；CDQ 取子集 `Z∈{1,2}`（line 3）；对称采样 **50/50**（line 12–14）；UTD `G`（line 11）；大 ensemble `E`（line 1） | `[本会话直取]` 同上 Algorithm 1 |
| **⚠️ 本地反证（不得引 RLPD 当依据）** | 本仓 **O2**：**价值头末端 LayerNorm 是 critic 死亡的第一性原因之一**（近常数输入被归一化 ⇒ 可表达跨帧方差 ≈0.14 而目标 std≈12.11，ReLU 逐个死亡；实测 `value` 唯一值 **1/600**、`std=0`）⇒ **同名机制、相反作用**，RLPD 的结论**不能**推出「我们也该加 LayerNorm」 | `[本地-实测]` `docs/agents/ledger.md` **O2**（并由 `il_reward_reference_analysis_2026-09-19.md` §1.3 独立复核） |

### §4.4 TD3+BC / DAPG / Cal-QL / DT / robomimic —— 逐字与落点补充

| 对象 | 项 | 逐字 / 内容 | 出处 |
|---|---|---|---|
| **TD3+BC** | **②（BC 正则 = 锚定）** | `π = argmax_π E_{(s,a)~D}[ λ·Q(s,π(s)) − (π(s)−a)² ]`，**`λ = α / Σ_{(s,a)}\|Q(s,a)\|`**；原文强调这是「normalize by the **average absolute value**」的启发式；**α=2.5 默认**，在 (1, 2, 2.5, 3, 4) 上调过 | `[一手-论文原文]` `arXiv:2106.06860`（ar5iv 逐字，上游）；**本轮未复取** |
| **TD3+BC** | **③ 不退火** | α 固定；λ 随 `Σ\|Q\|` 自适应 = **归一化**，不是调度 | 同上 |
| **DAPG** | **②③ 权重含退火** | `w(s,a) = λ₀·λ₁^k·max_{(s',a')~ρ_π} A^π(s',a')`（k = 迭代计数），**λ₀=0.1、λ₁=0.95**；原文：`"we asymptotically decay the auxiliary objective"`（初期示范至少不差于策略、末期不再偏置梯度）；流程 = 先 **BC 预训练**再以该 augmented loss 微调 | `[一手-论文原文]` `arXiv:1709.10087` §IV-C2（ar5iv 逐字，上游）；**本轮未复取** |
| **DAPG** | **⑦ 引用纪律** | ⚠️ 上一稿引 `1910.10314` 是**物理论文**（*The pure-quartic soliton laser*）⇒ **DAPG 真身 = `1709.10087`**；本条是**本族唯一的 arXiv id 更正**，写进正文以防再引错 | 上游 `il_reward_mechanism_synthesis_2026-09-19.md` §7.4（arXiv id 复核） |
| **Cal-QL** | **②** | CQL 离线预训练（Eq. 5.1）→ 在线微调；机制 = **保守价值下界校准**（不是 IL 惩罚、不是 BC 锚定）；「无 BC 项」属 **verified-by-absence** ⇒ `未验证` | `[一手-论文原文]` `arXiv:2303.05479` Eq. 5.1（ar5iv 逐字，上游）；**本轮未复取** |
| **DT** | **②⑤ 无惩罚、无 critic** | 训练 = 纯监督动作预测（离散 CE / 连续 MSE）；**没有奖励项、没有 critic、没有 bootstrap**；奖励只被用来算 `R̂_t = Σ_{t'≥t} r_{t'}` 再当**输入 token** | `[一手-论文原文]` `arXiv:2106.01345`（e-print 逐字，上游） |
| **DT** | **⑥ 与 %BC 的关系（本轮直取命中）** | Table 3「Average」列：**DT 56.1 vs 10%BC 56.7**（25%BC 52.7 / 40%BC 49.4 / 100%BC 39.5 / CQL 43.5）；原文限定：`"When data is plentiful … %BC can match or beat other offline RL methods"` ⇒ **DT ≈ 掐尖百分位 BC**，且**限定条件是数据充足** | `[本会话直取]` `arXiv:2106.01345` Table 3 |
| **DT** | **④ 的两套口径** | 论文 `return` 的缩放 **`[NOT-FOUND]`** vs **代码 `scale=1000.`** ⇒ 并列（`未验证`） | 上游 U11；**本轮未核代码** |
| **robomimic** | **⑥ 证据分裂的一侧（本轮直取命中）** | Table 1「Square (MH)」：**BC 52.7±6.6 / BC-RNN 78.0±4.3 / BCQ 14.0±4.3 / CQL 0.7±0.9**；Table 3 说明逐字：`"methods that model temporal correlations (BC-RNN, HBC, IRIS) exhibit strong performance on human datasets. Furthermore, while Batch RL algorithms like BCQ are proficient on machine-generated data, they perform poorly on human datasets."` | `[本会话直取]` `arXiv:2108.03298v1` Table 1 / Table 3 |
| **robomimic** | **④ 的实证教训** | **(C4) 训练目标只是替代指标（success rate 才是真目标）、策略表现逐 epoch 波动大 ⇒ 模型选择困难**；**(C5) 对智能体设计决策高度敏感** | `[一手-论文原文]` 同上 §2 |

---

## §5 惩罚项与归一化的横向对照（按 schema ② 与 ④ 抽平）

### §5.1 ② 惩罚项的四种落点（**不是同一类东西**）

| 落点 | 机制 | 成员 | 参考分布存在吗 | 依赖「学出来的奖励」吗 |
|---|---|---|---|---|
| **RL 奖励里的 KL 项** | `−β·KL(π‖π_ref)`（per-token 或序列级） | InstructGPT・HH・TRL・DS-Chat | **是**（π_SFT / policy_0 / ref_model） | **依赖**（InstructGPT 理由③；Stiennon 理由②） |
| **IL/RL 损失里的 BC 正则** | `λ·Q − (π−a)²`（TD3+BC） | TD3+BC | 数据集行为分布 | 不依赖 RM（但依赖数据集内 Q） |
| **IL 损失里的优势加权** | `exp(A/λ)` 或**批内 softmax**（AWAC / IQL / AWR） | AWAC・IQL | 数据集行为分布 `β(a\|s)` | 不依赖 RM（奖励来自数据集） |
| **IL 损失里的辅助项 + 退火** | `λ₀λ₁^k·max A^π`（DAPG）；演示 BC 预训练 + 渐近衰减 | DAPG | 演示数据 | 不依赖 RM |
| **不做惩罚，改用结构 / 内建界** | 无 critic（DT）；critic 内 LayerNorm（RLPD）；CQL 保守下界校准（Cal-QL）；expectile 不查 OOD（IQL）；离线预训练+对称采样 | DT・RLPD・Cal-QL・IQL | — | — |

⭐ **这行是本报告最重要的一条结构判断**：本族里**只有 RLHF 那一列**把惩罚项做成「RL 奖励的组成部分」；离线族里**没有任何成员**把惩罚写进环境奖励 —— 它们要么写进**损失**，要么改用**结构**（无 critic / 内建界 / 保守下界）。**「惩罚项 ≠ 奖励扣分」是本族的一致范式。**

### §5.2 ② 的两套约定并列（【R17】不调和）

| 项 | 约定 A | 约定 B | 谁在用 |
|---|---|---|---|
| **KL 口径** | **per-token**（逐 token 一阶差分） | **序列级**（一条序列一个 KL 标量） | A：InstructGPT（散文）・TRL（代码）・DS-Chat（代码）；B：HH（Eq.4.1 写法，口径 `[NOT-FOUND]`） |
| **RM 分数加在哪** | **末 token 加一次** | **序列级标量**（位置未写） | 末 token：TRL（末个非 mask token）・DS-Chat（末个 response token，先 ±5 裁剪）；序列级：InstructGPT Eq.2（**位置未写**）・HH（`r_PM` 即奖励） |
| **优势加权形式** | **`exp(A/λ)` / `exp(βA)`**（论文形式，归一常数 `Z(s)`） | **批内 softmax**（归一 = 批和） | A：AWAC 论文・IQL 论文；B：**rlkit 默认**（`normalize_over_batch=True`）・**jaxrl 只有** batch-softmax |

### §5.3 ④ 归一化三问对照（能回答的才写，不能写「未取到」）

| 对象 | 奖励归一 | 价值归一 | 优势归一 |
|---|---|---|---|
| InstructGPT | ✅ RM 用 bias 归一到「示范均分 0」 | ✅ value 初始化自 RM | **未取到** |
| HH / TRL / DS-Chat | DS-Chat 有 ±5 裁剪；HH/TRL 未取到 | TRL 不在本库；DS-Chat 一行 `未验证` | **未取到** |
| AWAC | 不适用（只进 critic） | TD 目标 | ✅ **落在权重**（softmax = 批内归一） |
| IQL | **未取到** | ✅ expectile（非对称加权 = 价值目标的分位归一） | ✅ `A = Q − V`（同 critic 内标量化） |
| DT | 代码 `scale=1000.`（论文 `[NOT-FOUND]`） | **N/A**（无 critic） | **N/A** |
| TD3+BC | **未取到**（D4RL 标准） | ✅ **`λ = α/Σ\|Q\|` 的分母就是对价值尺度的归一** | 无显式优势 |
| DAPG | **未取到** | on-policy value | ✅ 权重里含 `max A^π` |
| RLPD | **未取到** | ✅ **critic 内 LayerNorm = 「界」** | SAC MaxEnt（`−α log π`） |
| Cal-QL | **未取到** | ✅ 校准 = 离线/在线 Q 尺度对齐 | **未取到** |
| robomimic | **未取到**（各算法） | 按算法而异 | 按算法而异；作者结论是 **(C5) 超参敏感** |

---

## §6 口径冲突清单（**并列不调和**，【R17】）

| # | 冲突 | 两侧并列 | 出处 |
|---|---|---|---|
| **C1** | **KL 惩罚的作用** | ① InstructGPT：`"to mitigate over-optimization of the reward model"`；② **同一研究组受控实验** Gao：`"the effect of the penalty on the gold score is **akin to early stopping**"` + `"using KL penalty has a **strictly larger proxy-gold gap**, we **set KL penalty to 0** for all other RL experiments"` | InstructGPT §3.5 vs Gao `2210.10760v1` §3.6（**本轮直取命中**） |
| **C2** | **KL 是 per-token 还是序列级** | ① InstructGPT：散文 "at each token"、**公式无 Σ**；② HH：Eq.4.1 **序列级写法**、per-token `[NOT-FOUND]`；③ TRL/DS-Chat：**代码逐 token** | 三处并列，**不取一** |
| **C3** | **RM 分数加在哪** | ① InstructGPT Eq.2：序列级标量 `r_θ(x,y)`，**位置未写**；② TRL：末个非 mask token；③ DS-Chat：末个 response token（先 ±5 裁剪） | 三处并列 |
| **C4** | **优势加权形式** | ① `exp(A/λ)`（AWAC/IQL 论文）；② **批内 softmax**（rlkit 默认 / jaxrl 唯一） | `2006.09359` vs `awac_trainer.py:618–631` / `jaxrl/awac/actor.py` |
| **C5** | **BC vs 离线 RL（证据分裂）** | ① **人类示范**：BC-RNN **78.0** ≫ BCQ **14.0** / CQL **0.7**（Square-MH）；② **合成/低数据量**：D4RL locomotion 合计 **IQL 692.4 > BC 466.7**；③ **DT ≈ 掐尖 %BC**（**56.1 vs 56.7**，且原文限定 "When data is plentiful"） | robomimic Table 1（**本轮直取命中**）・IQL（上游）・DT Table 3（**本轮直取命中**） |
| **C6** | **我们自己的两套奖励表** | `rl/config.py:37-68` 的 `DEFAULT_REWARD` = **22 键**（**本轮复算 = 22**）vs `rl/reward.py:28-46` 的 `_DEFAULT_REWARD` = **16 键**（**本轮复算 = 16**）；⚠️ **且 `rl/reward.py:25` 的注释写着「与 `rl/config.DEFAULT_REWARD` 保持一致；勿单独改一处」—— 注释与代码不符**（差集 6 = 5 个 `engagement_trade*`（config-only）+ 1 个 `draw_penalty`（真实缺口））⇒ 直接关系【R7】「单一常量源」，**本轮只定位、未修** | `[本地-实测]` `src/clasher_new/rl/config.py:37-68`、`src/clasher_new/rl/reward.py:25/28-46` |
| **C7** | **TRL 文档符号 vs 代码符号** | 文档：`objective/non_score_reward` = `beta * kl.sum(1)`、`rlhf_reward = score − non_score_reward`；代码：`non_score_reward = -kl_ctl.value * kl`（**减**） | `docs/source/ppo_trainer.md` @ `v0.15.2` vs `ppo_trainer.py:1135–1146` |
| **C8** | **DT 的 return 缩放** | 论文 **`[NOT-FOUND]`** vs 代码 **`scale=1000.`** | 上游 U11（`未验证`） |
| **C9** | **TD3+BC 的 λ** | **含 `1/N`** vs **不含 `1/N`** 两套并存 | 上游 U13（`未验证`） |
| **C10** | **RLPD 一句的 OCR** | ar5iv 渲染为 `"our **normalized update** is not an offline RL method"`，语义疑为 `"our **method** is not an offline RL method"` ⇒ 引用时**并列标注** | `[本会话直取]` ar5iv `2302.02948` §2 |

---

## §7 另必答：与游戏类（环境胜负 + 手工项）的本质差异，以及我们的能迁 / 不能迁

### §7.0 我方事实（本轮复核，作为判断前提）

| 项 | 事实 | 落点（本轮复核） |
|---|---|---|
| 奖励 | **手工 22 键表**（硬编码默认值），**不是学出来的 RM** | `src/clasher_new/rl/config.py:37-68`（**本轮复算键数 = 22**） |
| 另一套默认 | `_DEFAULT_REWARD` = **16 键**，且 `:25` 注释声称与 config 一致（**不符**） | `src/clasher_new/rl/reward.py:25/28-46`（**本轮复算键数 = 16**） |
| PPO | 手写 `PPOTrainer`：`clip=0.2`、`ent_coef=0.01`、`n_epochs=1`；**无任何 KL 项** | `src/clasher_new/rl/ppo.py:87-89`（本轮复核）；`target_kl`/`approx_kl`/`kl_` 在 `rl/ppo.py`・`rl/train_solo.py`・`rl/run_league.py` **零命中**（上游 §5.0） |
| IL 侧 | `loss = -lp`，**无 entropy / value / KL 正则**；每帧 `hidden=None`（帧间无状态、无梯度） | `src/clasher_new/rl/train_bc.py:107-108`（本轮复核） |
| 人类偏好数据 / RM | **没有** | 上游 `rlhf_kl_penalty_survey_2026-09-19.md` §5.0 |
| 唯一真锚 | 我们**自己的引擎**（规则可判定的胜负 / 塔血 / 费差） | 本仓；`docs/reward_composition_verdict_2026-09-14.md`（五项拆解、无主导项） |

### §7.1 必答（**≤5 行**）

1. **本质差异**：本族的「奖惩」主干是**学出来的代理**（RM / PM 分数）或**数据集内回报的再分配**（优势加权 / return-to-go），惩罚项（KL-to-reference、BC 正则）是围绕**参考分布**的约束；游戏类（我们）= 环境给的**真信号**（胜负 / 塔血 / 费差 / 非法计数）在手写规则上的**确定性函数** ⇒ **没有「代理 vs 金标」的分裂，也没有「RM 训练分布」这个概念**。
2. **推论**：本族全部惩罚项都在服务「**奖励是代理**」这一前提（防代理过优化 / 防跑出代理训练分布 / 把数据内回报变成可学信号），且惩罚对象是**策略分布**而非环境状态 ⇒ **「本族这么做有效」不能推出「我们也该有惩罚项」**。
3. **能迁（3 条）**：① **BC 正则 / 参考分布锚定的数学形态**（TD3+BC 的 `λ·Q − (π−a)²`、AWAC/IQL 的优势加权）—— 只作 **IL 损失内的样本权重或辅助项**，**不动奖励函数**；② **「用结构替代惩罚」的范式**（RLPD 的「内建界」思路、IQL 的不查 OOD、DT 的不建 critic）—— 形态可借，**方向必须自证**（RLPD 的 LayerNorm 在本仓 O2 里作用相反）；③ **两个诊断视角**：robomimic 的「训练目标只是替代指标（C4）」与 Gao 的「KL 系数 = **速率旋钮**而非安全装置」，可直接用来给我们的判据设计避坑。
4. **不能迁（3 条）**：① **KL-to-reference 的理由**（InstructGPT「mitigate over-optimization of the RM」、Stiennon 理由②③）**以学出来的奖励为前提 —— 我们无人类偏好数据、无 RM**，且它把策略**拉回参考分布**，与已确证的唯一病理 **C14（探索不足：放弃一张买得起的牌 0/7,238 段、A 段最长 22 帧 < 34 帧）方向相反**；② 本族**全部数值**（β=0.02 / λ_KL=0.001 / `init_kl_coef`=0.2 / `kl_ctl`=0.1 / λ=0.3 / τ=0.9 / β=10 / α=2.5 / λ₀λ₁）—— 量纲与奖励尺度不可比，【R15】明禁跨实验照抄阈值；③ **return-conditioning 直接上线** —— Paster 一手反证：随机环境 `"fail dramatically"` 且 `"not due to a lack of data"`，而我们的环境含卡序与对手随机。
5. **一句话**：本族给我们的不是新奖励，而是 (a) 一条**不碰奖励**的样本加权 / 条件化接口、(b) 一个「**惩罚项 ≠ 奖励扣分**」的结构范式、(c) 一组「代理奖励会坏」的反面证据；它治的病（奖励被利用 / 数据内回报再分配）与我们的病（**奖励开了但没有样本可作用**，C14/O7）**方向相反** ⇒ 本族**不构成翻转 X-19「奖励侧不接线」结论的依据**。

---

## §8 已知坑汇总（按对象 × 与我们病根的方向）

| 坑 | 原文/数字 | 属于谁 | 与我们的关系（**只写方向，不做对拍**） |
|---|---|---|---|
| 显式 KL 惩罚**不改善**真奖励前沿、只等价**早停** | `"akin to early stopping"`；`"strictly larger proxy-gold gap"` ⇒ 其余实验 **KL=0** | 族 A（Gao，**本轮直取**） | **反向**：我们的是「奖励没样本可作用」，不是「奖励被利用」 |
| 加大 KL 修不好 alignment tax | β 提到 **2.0 = 默认 100 倍**仍修不好；β=0/2 都差，最优 0.01~0.02 | InstructGPT（**本轮直取**） | **反向**（同上） |
| **负 KL 是失败前兆** | `"almost always a precursor for failed training"`；**3/10** 出现负 KL 爆炸 | TRL（二手 + 代码告警） | 我们**无 KL** ⇒ 无此坑，但也说明我们**没有这个安全阀**（替代物是 PPO clip = 隐式信任域） |
| **IL 数据质量 / 不做 RLHF 也能赢** | LIMA：**1,000 条精选**、**不用 RLHF / 人类偏好建模**，`65% versus DaVinci003, which was trained with human feedback`；数据翻倍不改善 | 族 A（LIMA，**本轮直取**） | **同名不同量**：我们的 BC 专家是**规则**、自动产 50 局 ⇒ **禁止**把 LIMA 外推为我们（`未验证`） |
| **BC vs 离线 RL 证据分裂** | 人类数据：BC-RNN **78.0** ≫ BCQ **14.0** / CQL **0.7**；合成/低数据量：IQL **692.4** > BC **466.7**；DT **56.1** ≈ 10%BC **56.7** | robomimic / IQL / DT（前两者：robomimic **本轮直取**） | **并列不调和**；对我们的用法是「**先看数据源与时序建模**，不要先选算法」 |
| **return-conditioning 在随机环境崩** | `"fail dramatically"`；`"not due to a lack of data"`；Connect Four `"cannot achieve more than 0.2 average return (win-rate 60%)"`；`"even with infinite data"` | Paster（**本轮直取**） | **高风险**：我们的卡序 / 对手含随机性 ⇒ I6 类方案只能待预注册 |
| **同名机制、相反作用** | RLPD：`"LayerNorm bounds the values"`；本仓 **O2**：价值头末端 LayerNorm 是 critic 死亡的第一性原因之一（`value` 唯一值 1/600、std=0） | RLPD（**本轮直取**）+ 本仓 O2 | **禁止照搬**；须自证 |
| **目标与真目标的错配** | robomimic **(C4)**：训练目标只是替代指标、策略表现逐 epoch 波动 ⇒ 模型选择困难；**(C5)** 超参高度敏感 | robomimic（同上） | **同向警告**：我们的性能层在 n=1 下不可归因（同配置单点差 **0.60**）⇒ 判据必须预注册 |
| **「奖励项开了但没有可学样本」** | 本仓 **X-19/O7/C14**：奖励侧已完整取证判不接线；奖励开了没样本 = RLHF 失败模式的**反向** | 本仓 | **这是我们唯一的病根**；任何从族 A 借来的「防奖励被利用」手段都**不对准** |

---

## §9 未验证 / 未闭合清单

### §9.1 未验证 / 未能核实（**共 34 项**：族 A 18・族 B 14・本地 2）

| # | 项 | 状态与原因 |
|---|---|---|
| **A1** | HH 的 KL 是 per-token 还是序列级 | `[NOT-FOUND]`：全文 grep `per-token`/`at each token`/`sum over tokens` = **0 命中**；Eq.4.1 是序列级写法。**不做推断** |
| **A2** | HH 的 PM 损失方程 | `[NOT-FOUND]`（该文确实没写）⇒ 引 Askell 2021 §3.1 Eq.(3.1) |
| **A3** | HH 的 KL 系数消融 | `[NOT-FOUND]`（全文无 `ablation`，只有 §B.1「a variety of hyperparameter scans」） |
| **A4** | HH 的 ④ 归一化三问（奖励/价值/优势） | **本轮未取到** ⇒ `未验证` |
| **A5** | HH 的 ⑤ value/critic 设计 | **本轮未取到** ⇒ `未验证` |
| **A6** | InstructGPT 的 SFT 损失公式 | `[NOT-FOUND]`：v1 全文（HTML + TeX 源码）无 SFT 损失方程，也无 "per-token cross entropy" 措辞 ⇒ 只能引 Stiennon §1 并标**转引** |
| **A7** | InstructGPT Eq.(2) 是否显式对 token 求和 | **公式里没有可见的 Σ**；`"per-token … at each token"` 只见于散文 ⇒ 「公式未写、散文写了」已如实标注 |
| **A8** | InstructGPT 的优势归一（GAE 细节） | **本轮未取到** ⇒ `未验证` |
| **A9** | InstructGPT 的 RM 分数在序列中的**位置** | **论文未写** ⇒ 不得照 TRL/DS-Chat 的「末 token」写法回填 |
| **A10** | Gao 的 KL 扫描具体 β 取值表（图 9） | 只验证到结论句与「其余实验 KL=0」；图的横轴具体数值**未验证** |
| **A11** | Stiennon Fig.5 里 β 的具体取值序列 | 未提取（只验证到 §4.3 结论文字与 KL 两个 purposes 的原文） |
| **A12** | DeepSpeed-Chat 的「reward running average」 | `[NOT-FOUND]`：`ppo_trainer.py` grep `mean\|std\|moving\|average\|ema` **无命中**；`main.py` import 了 `moving_average` 但**取不到该文件** ⇒ **不断言** |
| **A13** | DeepSpeed-Chat 的 per-stage 文档 + 一行 `torch.clamp` | 文档**取不到**（限流）；`torch.clamp` 属**两次独立转写共识、非逐字节验证** |
| **A14** | TRL 官方文档页 `huggingface.co/docs/trl/main/en/ppo_trainer` | **本环境不可访问**（host 不可达，重试 3 次 rc=000）；且该文档文件已从 `main` 移除 ⇒ 用 repo 文档源 `v0.15.2` 替代 |
| **A15** | TRL 的 value/critic 与优势归一 | **不在本库**（critic 由调用方传入）⇒ 字段不适用，非「漏抽」 |
| **A16** | TRL `main` 上 `PPOTrainer` 移除后的替代路径；最新 tag 的发布日期 | 替代路径**未调查**；最新 tag 观察到 **v1.13.0**，**发布日期未验证** |
| **A17** | 「弱 SFT → RM 上限」类命题 | `[NOT-FOUND]`：**没有任何论文**写这句话 ⇒ **不要当引用** |
| **A18** | 「手工/规则奖励 + RL 产生可复现 exploit」的受控论文证据 | `[NOT-FOUND]`（CoastRunners 类案例本轮**未验证**：`blog.openai.com` DNS 失败、`openai.com/index/faulty-reward-functions/` 403）⇒ **不得**引用 |
| **B1** | AWAC 论文附录两处内部不一致的具体归属 | 上游记为「Z(s) 积分内用 π_θ 而非 π_β、期望式漏 exp，属排版/笔误级」；**本轮未复取** ⇒ `未验证` |
| **B2** | AWAC 参考实现 `normalize_over_batch` 默认值的**代码行号** | 行号 **L618–631** 来自上游；**本轮未复取** ⇒ `未验证` |
| **B3** | IQL 官方实现是否也用批内 softmax | **本轮未核** ⇒ 与 AWAC 同样**两套并列**，但不得声称 IQL 实现也是 softmax |
| **B4** | IQL「刻意不查询数据集外动作」的**逐字原句** | **本轮未取回** ⇒ 只作机制描述，**不得当引用原文** |
| **B5** | IQL 的奖励归一化 | **未取到** ⇒ `未验证` |
| **B6** | DT 的 return 缩放 | 论文 `[NOT-FOUND]` vs 代码 `scale=1000.` ⇒ **并列**（`未验证`） |
| **B7** | DT Table 3 的「Average」行列标签 | **本轮未亲自复核**（上游 U12） |
| **B8** | TD3+BC 的 λ 含/不含 `1/N` | **两套并存**（上游 U13）⇒ 并列，不择一 |
| **B9** | TD3+BC 的奖励归一化（D4RL 标准） | **未取到** ⇒ `未验证` |
| **B10** | DAPG 的 λ₀/λ₁ 敏感度与其它超参 | **未取到** ⇒ `未验证` |
| **B11** | Cal-QL 的「无 BC 项」 | **verified-by-absence**（上游 U14）⇒ 保留 `未验证`，不升格为「证实没有」 |
| **B12** | Cal-QL 的 ⑥ 已知坑 | **本轮取回文本未记录** ⇒ 记「未取到」，**不臆造** |
| **B13** | robomimic 各算法的归一化细节 | **未取到**（只有 (C5)「超参敏感」）⇒ `未验证` |
| **B14** | robomimic Table 1/3 之外的附录全表 | **未逐行核**（本轮直取到 Table 1 与 Table 3） |
| **L1** | `rl/reward.py:25` 注释与代码不符（22 vs 16） | **本轮只定位、未修**（改动受【R7】管辖，须先定「单一常量源」形态） |
| **L2** | 「RLPD 的 LayerNorm 照搬有害」这一步 | O2 是本仓**确证**，但「RLPD 的用法 ⇒ 我们的用法必然有害」的因果链**未做实验** ⇒ `[本地-推理]` |

### §9.2 上游已核、**本轮未独立复现**（provenance 标记，**不计入 34**）

IQL 的 D4RL locomotion **692.4 / 466.7**；AWAC 的 λ=0.3/1.0 与 Z(s) 消融数字（pen 84%→98%、door 0%→95%、relocate 0%→54%）；DAPG 的 `λ₀=0.1 / λ₁=0.95` 与「we asymptotically decay the auxiliary objective」；TD3+BC 的 `α=2.5` 与 λ 公式；Cal-QL 的 Eq. 5.1；TRL/DS-Chat 的全部行号与 Git blob SHA；Stiennon 的 KL 两个 purposes 原文；HH 的 §B.1 超参串与 §B.4 Elo 过度估计串；Askell 的 Eq.(3.1) 原文；Paster 的 2048 具体数字（上游记「can only win about 60%」，**本轮未复现**）。
**说明**：这些项在上游盘上材料中标为 `[一手]` 并附 URL/行号，本报告**照抄其等级不改**，但**明确区分**「本轮独立命中」与「仅依上游」两种 provenance。

---

## §10 来源清单（URL + 章节号 / `file:line` + 取数时间）

### §10.1 本轮独立直取命中（`[本会话直取]`，2026-09-19）

| 来源 | URL | 章节 / 表 | 命中的关键字串 |
|---|---|---|---|
| Gao/Schulman/Hilton 2022 | <https://arxiv.org/html/2210.10760v1> | §3.6 Effect of KL Penalty | `akin to early stopping`・`strictly larger proxy-gold gap`・`set KL penalty to 0` |
| LIMA | <https://arxiv.org/html/2305.11206v1> | 摘要 / 正文 | `1,000 carefully curated`・`without any reinforcement learning or human preference modeling`・`65% versus DaVinci003, which was trained with human feedback`・`doubling the training set does not improve response quality` |
| InstructGPT | <https://arxiv.org/html/2203.02155v1> | §3.5 / §C.4 | `per-token KL penalty`・`at each token`・`mitigate over-optimization`・`β=0.02` |
| Anthropic HH | <https://arxiv.org/html/2204.05862v1> | §4.1 / §B.1（Eq.4.1） | `wholly unnecessary`・`λ_KL=0.001`・`r_PM = the preference model score itself for the RL reward` |
| Decision Transformer | <https://ar5iv.labs.arxiv.org/html/2106.01345>（Table 3 经 ar5iv；`2106.01345`） | Table 3 | `Average 56.1` / `10%BC 56.7` |
| RLPD | <https://ar5iv.labs.arxiv.org/html/2302.02948> | §2 / §4 / §4.2 / Algorithm 1 | `we do not restrict the policy using a behavior cloning term`・`LayerNorm bounds the values`・`does not explicitly constrain`・`anti-exploration`・`without pre-training or explicit constraints` |
| robomimic | <https://arxiv.org/html/2108.03298v1> | Table 1 / Table 3 / §2 | `BC-RNN 78.0`・`BCQ 14.0`・`CQL 0.7`・`temporal correlations` |
| Paster et al. | <https://arxiv.org/html/2205.15967v1> | 摘要 / Fig.1 caption / §3.3 / §3.4 | `fail dramatically`・`not due to a lack of data`・`even with infinite data`・`0.2 average return`・`win-rate of 60%` |

### §10.2 上游盘上材料（等级照抄，不升级；取数 2026-09-18T19:33Z–2026-09-19 CST）

| 来源 | URL | 章节 / 行号 |
|---|---|---|
| InstructGPT | <https://arxiv.org/html/2203.02155v1>；TeX <https://arxiv.org/e-print/2203.02155v1> | §3.1・§3.2・§3.5（Eq.1/Eq.2）・§C.1–C.4・§E.6・§E.7 |
| Anthropic HH | <https://arxiv.org/html/2204.05862v1>；<https://arxiv.org/pdf/2204.05862v1>（74 页） | §3.1・§4.1（Eq.4.1/4.2）・§4.2・§4.3・§A.2・§B.1・§B.4・Fig.4 |
| Askell et al. 2021（PM 损失真出处） | <https://arxiv.org/html/2112.00861v3> | §3.1 Eq.(3.1) |
| Stiennon et al. 2020 | <https://ar5iv.labs.arxiv.org/html/2009.01325> | §1・§3.4（KL 两个 purposes）・§4.3（过优化 / 反相关） |
| TRL `v0.11.2` | <https://github.com/huggingface/trl/blob/v0.11.2/trl/trainer/ppo_trainer.py>；`.../ppo_config.py`；`.../utils.py` | `_kl_penalty` **L1150–1162**・`compute_rewards` **L1135–1146**・负 KL 告警 **L1311–1317**・`ppo_config.py` **L161–181**・`utils.py` **L54–79** |
| TRL 文档源 | <https://github.com/huggingface/trl/blob/v0.15.2/docs/source/ppo_trainer.md> | `objective/non_score_reward` / `objective/rlhf_reward` |
| TRL 移除 commit | `700b845c5d4b1cfd292cf9c067626d8a19fc0fe1`「Remove PPOTrainer (#7020)」，2026-09-04 | — |
| TRL issue/PR（**二手**） | `/issues/235`・`/issues/417`・`/issues/1178`・`/pull/6429` | 逐条见 §2.3 |
| DeepSpeed-Chat | <https://github.com/deepspeedai/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/dschat/rlhf/ppo_trainer.py> | 默认值 **L65–71**・`compute_rewards` **L181–194**・ref_logprobs **L154–174** |
| DeepSpeed-Chat README / rename commit | `.../applications/DeepSpeed-Chat/README.md`；`e86d0c6d38d65158836c275e2d7b6a8f8e92f026` | 三阶段自述 |
| AWAC | <https://arxiv.org/abs/2006.09359>；<https://arxiv.org/e-print/2006.09359>；<https://awacrl.github.io/> | 闭式解 / λ=0.3・1.0 / Z(s) 消融 |
| AWAC 参考实现 | <https://github.com/vitchyr/rlkit/blob/master/rlkit/torch/sac/awac_trainer.py>（**L618–631**）；<https://github.com/ikostrikov/jaxrl/blob/main/jaxrl/agents/awac/actor.py> | 批内 softmax |
| IQL | <https://arxiv.org/e-print/2110.06169>；<https://github.com/ikostrikov/implicit_q_learning> | expectile + 优势加权回归；τ/β 取值 |
| Decision Transformer | <https://arxiv.org/e-print/2106.01345>；<https://arxiv.org/abs/2106.01345> | return-to-go；Table 3；Discussion |
| Paster et al. | <https://arxiv.org/e-print/2205.15967>；<https://arxiv.org/abs/2205.15967> | RvS 随机环境失败 |
| TD3+BC | <https://ar5iv.labs.arxiv.org/html/2106.06860> | `λ=α/Σ\|Q\|`、α=2.5 |
| DAPG | <https://ar5iv.labs.arxiv.org/html/1709.10087> | §IV-C2（`w=λ₀λ₁^k max A^π`、λ₀=0.1、λ₁=0.95） |
| RLPD | <https://ar5iv.labs.arxiv.org/html/2302.02948> | §1・§2（两段）・§4・§4.2・§5・Algorithm 1 |
| Cal-QL | <https://ar5iv.labs.arxiv.org/html/2303.05479> | CQL 离线预训练（Eq. 5.1）→ 在线微调 |
| robomimic | <https://ar5iv.labs.arxiv.org/html/2108.03298> | Table 1（Square (MH)） |

### §10.3 本地项（`file:line`，**本轮复核**）

| 用途 | 位置 |
|---|---|
| 手工 22 键奖励表 | `src/clasher_new/rl/config.py:37-68`（**本轮复算键数 = 22**） |
| 另一套 16 键默认 + 与代码不符的注释 | `src/clasher_new/rl/reward.py:25`（注释）、`:28-46`（**本轮复算键数 = 16**） |
| 手写 PPO（无 KL）默认值 | `src/clasher_new/rl/ppo.py:87-89`（`clip=0.2`、`ent_coef=0.01`、`n_epochs=1`） |
| BC 损失（IL 阶段无奖励/无正则/无状态） | `src/clasher_new/rl/train_bc.py:107`（`hidden=None`）、`:108`（`loss=-lp`） |
| 本仓台账（本文引用的缺口） | `docs/agents/ledger.md`：**C3**・**C14**・**O2**・**O7**・**X-19**・**O8**・**O9**・**O10** |
| 本仓取证文档 | `docs/s1_gate_2026-09-18.md`・`docs/elixir_saving_audit_2026-09-18.md`・`docs/s2_instrument_2026-09-18.md`・`docs/online_measure_2026-09-18.md`・`docs/pass_prob_2026-09-18.md`・`docs/exploration_pressure_gate_2026-09-18.md`・`docs/et_solo100k_judgment_2026-09-18.md`・`docs/reward_composition_verdict_2026-09-14.md`・`docs/train_health_metrics_2026-09-18.md` |
| 上游三份盘上材料 | `docs/rlhf_kl_penalty_survey_2026-09-19.md`・`docs/il_reward_mechanism_synthesis_2026-09-19.md`・`docs/il_reward_reference_analysis_2026-09-19.md` |

---

## §11 纪律自查

| 条款 | 执行情况 |
|---|---|
| **【R17】口径冲突并列不调和** | §5.2 与 §6 共 **10 条**冲突**全部并列**（KL 作用、per-token/序列级、RM 位置、`exp`/softmax、BC vs 离线 RL、我方 22/16 键、TRL 文档↔代码符号、DT scale、TD3+BC λ、RLPD OCR），**未择一、未调和、未折算** |
| **禁止跨项目数字对拍** | 全部数字**只**出现在「该对象自身构造」的语境（β / λ / α / τ / 表格成功率）；**未**用任何外部数字作我们的判据阈值；§7.1 第 4 条已明列禁止清单 |
| **「未验证」不升格** | §9.1 **34 项**原样保留；§9.2 另标 **10 组**「上游已核、本轮未复现」的 provenance 标记，**不混入**已确证 |
| **不把 IL 写成 RL** | 每个对象都有 ⑧ 阶段归属；DT 明确标「IL 阶段、**无 RL 阶段**」；RLPD 明确标「offline buffer + 对称采样是**跨阶段数据接口、不是 IL 损失**」；HH 明确标「IL 阶段**不存在**」 |
| **【R3】描述性 ≠ 判据** | §8 的每一条坑只写**方向**，不写成阈值；§7.1 的能迁/不能迁写成「可做 / 待预注册 / 不做」的**方向判断**，不含新判据 |
| **【R11】不在 A′ 类取证之前改奖励** | §7.1 第 4 条与 §8 末行明确：本族**不构成**翻转 X-19「奖励侧不接线」的依据；本轮**未改任何**训练/奖励代码 |
| **【R10】不确定就写不确定** | 全部 `[NOT-FOUND]` 与「未取到」**原样保留**；`未验证` 项未做任何推断（如 HH 的 per-token 口径、Cal-QL 的坑条目） |
| **【R7】单一常量源** | 发现并记录 `rl/reward.py:25` 注释与代码不符（22 vs 16）⇒ 列 §9.1 **L1**（**本轮只定位、未修**） |
| **未做声明** | **未跑训练**、**未跑全量 selftest**（【R19】）、**未改任何训练/奖励代码**；本轮唯一的文件改动 = 本文件自身的补齐（保顶部 70 行原文） |

