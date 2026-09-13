# AGENTS — 项目决策与方案存档

本文件记录跨会话需要记住的方案评估与决策，供后续会话/协作者直接引用，避免重复推导。
（文档↔源码总索引见 `docs/README.md`；训练代码导读见 `src/clasher_new/rl/README.md`。）

> ⚠️ **本文已超出工作区指令预算（65,536 B），文件末尾内容会被截断、后续会话看不到。**
> 因此新决策一律在文件**顶部**这一小块登记指针（正文仍在末尾按时间追加，仅供人工查阅）。

## 最新决策索引（置顶指针，2026-09-13）

- **评估节奏 C 方案（密锚点 + 稀全块）**：`docs/eval_cadence_c_2026-09-13.md`。
  20k 协议里评估占 **73% 墙钟**（纯训练 26 步/s、全点 ≈210 s、轻点 ≈53 s）；
  100k 由 3.46 h 降到 **2.15 h**。**不要均匀放宽评估周期**——实测 D1 三跑锚点谷底
  **只有 1 个评估点宽**（相邻点 |Δ|≈0.23 ≈3σ），粗采样回放：5000 步即开始漏真谷底、
  10000 步把"最差点"从 0.192 抬到 0.462（病态组持续塌陷留得住、健康组周期性瞬态留不住）。
  工具 = `--anchor-every`（默认 0 = 旧行为逐位不变）。
- **D1 100k 长跑预注册**：`docs/d1_long_100k_prereg_2026-09-13.md`（主判据 = C1 块口径；
  天花板问题只做描述、不设判决线；跑前披露补种 ckpt 在 100k 被挤出等 5 条偏差）。
- **D1 100k 长跑判读（已完成）**：`docs/d1_long_100k_verdict_2026-09-13.md`。
  **P1 通过**（5 块 worst 0.388/0.475/0.275/0.237/0.525 → mean 0.380、min 0.237；
  无变化组 [0.000,0.025] ⇒ 区间不重叠）⇒ **防崩在 4× 训练量下成立**；min 0.237 落在
  D1 20k 区间 [0.125,0.250] 内 ⇒ 该判据上与 20k 无统计差别；**上限仍未解决**
  （次判据口径敏感：+0.125 或 +0.062，按预注册只描述不判决；cycling 未消除）。
  **对 v3 的正面确认**：GRU 活力 100k 全程健康（h_std 0.067→0.095、n_abs 0.667→0.826）。
  **否证"加量能救 critic"**：EV 仅前两点为正（+0.234/+0.385）其后 9/11 点 ≈0 或负，
  `vstd/rstd` 30k 后长期 ≈0.001，run 自身打 8 次 vitality 告警。
  判读工具：`scripts/judge_anchor_blocks.py`（C1 分块，基线脚本复算）+ `scripts/summarize_solo_run.py`。
- 新增回归：`rl/selftest.py::test_anchor_light_point_state`。

---

## 环境：Python 已迁到 E 盘（2026-09-12，用户删除了 C 盘 Python）

- **基础解释器**：`E:\Python313\python.exe`（3.13.12，pip 25.3；安装包保留在
  `E:\python-install\python-3.13.12-amd64.exe`，安装日志同目录）。
  原 C 盘 `C:\Users\枯月流魂\AppData\Local\Programs\Python\Python313` **已被用户删除**，
  venv shim 报 `did not find executable at 'C:\Users\????\...'` 就是这个原因。
- **`.venv` 修复方式（教训）**：WSL 直接调 `/mnt/e/Python313/python.exe -m venv <win路径>`
  是**无效 no-op**（互操作层参数改写，venv 没跑或跑错地方，exit 0 且 cfg 不变）。
  必须经 Windows 侧执行：
  `cmd.exe /c "E:\Python313\python.exe -m venv --upgrade E:\clash-royale-simulator-main\.venv"`
  （`--upgrade` 只重建 Scripts/pyvenv.cfg，**保留 site-packages，无需重装 torch**）。
- **pip 缓存**：`E:\Python313\pip-cache`，经 `E:\Python313\pip.ini`
  （`[global] cache-dir`）+ 用户环境变量 `PIP_CONFIG_FILE=E:\Python313\pip.ini` 全局生效。
- 验证口径：`.venv/Scripts/python.exe -c "import torch; torch.cuda.is_available()"` 为 True。

---

## ⚠️ 全局操作约定：训练期"性能异常"先归因外部，暂停等人工（2026-09-11）

**当前阶段，训练过程中任何"看起来像性能问题"的现象——多数是因为外部问题，而不是本项目的性能 bug。**

遇到这类现象（吞吐骤降、子进程批量崩溃、内存/页面文件报错、spawn 失败、评估卡死、
进程莫名退出等）：

> **暂停训练，保留现场（日志 / 报错原文 / 进程状态），等人类操作后再继续。**
> 不要自动降级、不要为了"绕开"它去改代码或下调配置、不要把猜测写进注释。

**为什么（实证，成因未定）**：2026-09-11 `ab_valnorm_20k` 训练中，`--eval-workers 16` 多次触发
`WinError 1455「页面文件太小」`（加载 `torch\lib\cufft64_12.dll` 时失败）→ worker 启动即崩
（第一次全部崩、第二次只崩 1 个）→ **静默降级串行**（eval 从 ~1min 变成 ~8min）。
当时的处置是把它当成"页面文件不够"，把默认并行度下调到 10 并写进代码注释。
**事后复核时，成因其实未定**，正反两面证据都有：

- 支持"外部干扰"：故障与宿主 shell shim 报错同时发生（`dirname: command not found` /
  `cd: null directory`）；丢弃小 run（`_tmp_worker16`）复测 16 并行 **0 报错、全跑通**。
- 支持"真实 commit 压力"：实测**提交上限 47.3GB / 空闲 20.6GB**，而 16 个 CUDA-torch
  worker ≈ 16×1.3GB ≈ **20.8GB，正好压在空闲 commit 边界上**；训练进程自身还占 ~2GB。
  这也解释了"小 run 通过、真 run 失败"——复测时训练进程很小，余量充足。
- **empirically 安全的档位是 12**（9k eval 周期实测 167s，未降级）。

**处置约定**：遇到就按上面的规则**暂停、留证、等人类**，不要自己下调配置或改注释。
历史上已经吃过一次亏：当时直接判成"页面文件不够"→ 默认降到 10 并写进注释 →
后续会话会照抄这条错误结论。**宁可把成因写成"未定"，也不要写一个自信的错答案。**

**推论（同等对待）**：
- 现象出现时，先问"是不是宿主 / 杀软 / 磁盘 / 别的进程干的"，再问"是不是代码干的"；
- **静默降级是危险设计**——它把故障伪装成"只是慢"，会让归因错误长期不被发现。
  暂不改动该机制，但**判读任何性能数字前，先确认本次有没有发生降级**；
- 本文件里的性能基准（引擎 18,931 battle-steps/s、训练循环 ~14 步/s 等）若与实测严重不符，
  **优先按外部干扰解释**，先复测再下结论。

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

### 「不会留费/不会沉底/不会组织进攻」取证与诊断（2026-09-09，用户定题）

**行为取证（economy_9k_ft 112 局，回放逐帧）**：
- 满费时间占比 **0.1%**（圣水一满立刻泄掉，11 次满费→下牌延迟中位 0.5s）；
- 圣水均值 **2.04/2.11**（双倍期前后几乎无差）——产能 1费/2.8s × 180s ≈ 64 费，
  实际每局只花 ~29 费，一半产能被"一够费就花"的节律锁死在低价值单卡上；
- **3s 内连发（组波）占比 0.0%**，bundle 内 2 张率 0.8%、3-4 张 0——
  K_MAX=4 的组合动作容量形同虚设；
- 进攻部署 45% 顶桥头（y16-19）、36% 沉底（y24+）——沉底不是完全没有，
  但与攒费/组波无关（都是单张随机深度）。

**三层诊断（为何学不会组织进攻）**：
1. **动作结构**：bundle K_MAX=4 容量在，但训练里从未被用满——镜像自对弈中
   "多张同帧"从无奖励差异，且 STOP 在 PPO 熵正则下永远是最安全的默认动作；
   94% 决策帧 STOP + 6% 单张帧 = 从未出现"攒 20 帧费然后 1 帧泄 3 张"的模式。
2. **奖励结构**：elixir_diff 是 potential-style shaping——下牌瞬间 E−c 与 V+c
   同帧抵消，**攒费本身零奖励**；"一波 8 费换塔"与"四次 2 费换塔"在 tower_dmg
   账面上等价，组织进攻的额外收益（时间窗口/塔仇恨/法术配合）没有任何显式项。
3. **对手结构**：镜像 frozen_copy 同样裸奔 → 攒费方在训练分布里得不到对手
   压力差（对面也 2 费裸奔），"攒费→被打一波"的负样本从未出现。

**修复方向（按侵入性排序，待拍板）**：
- A（观测+课程）：把"当前圣水/满费倒计时/双倍期倒计时"显式进观测（已有）+
  plan 通道加 `save_up` 意图（攒费蓄势态，plan_space 已有 setup_wait 雏形），
  热启动微调让策略先"见过攒费解"；
- B（奖励）：加"组波奖励"——bundle 内 ≥2 张且总费 ≥6 时给小额 bonus，或
  把 elixir_diff 改成**凹形定价**（存 6-10 费时每费额外 +δ，泄到 <2 费时 δ 撤回），
  让"蓄势"有正账面；危险点：与 MCTS 值函数/闸门汇率必须同步改，否则重蹈量纲失配；
- C（对手结构）：~~对手池加 rush 脚本~~ **已否决（2026-09-09 用户）**——对手模型
  也不会有意识保留圣水，rush 脚本在双盲环境下 100% 触发（镜像 main 恒满费真空），
  学到的只是"泄得更勤"而非"攒费打波"。

**C 否决后的深挖（2026-09-09，setup_wait 死锁确认）**：
行为学根因不在"没有课程"，在**课程不可达**。7h 已实现完整攒费链
`setup_wait（攒费+hold_mask=1111 禁花手牌）→ 满费沉底血牛 → push_commit 跟后排`，
但实测 4 局 898 决策帧 **100% cycle_and_wait**，setup_wait 从未触发。死锁结构：
①触发要求"当下手里就有血牛（Giant 等，手牌 1/8≈12% 概率）"——血牛不在手
直接 return None，连 hold 分支都不给；②hold（攒费等待）分支又要求"后排在手"
（再 ×44%）且沉底要攒到 9.95——而策略 2 费裸奔的均衡态下永远到不了 9.95。
**"先攒到满费才沉底、先有血牛在手才准攒"互为先决条件 = 鸡生蛋死锁**，
任何 RL 梯度都够不着这个从未被采样过的分支（cold-start 不可达，非欠训练）。

**解法（重构 setup_wait 触发结构，提交前定稿）**：
攒费窗口条件改为**不依赖血牛在手**——"无 pressing 威胁 + 圣水≥4 + 双倍期前"
即进入 hold（禁花手牌，仅保留防守豁免：threat 或过河敌出现立即解禁）；
血牛在手时才输出 suggested_card（沉底预告），不在手就先攒——血牛轮转到手
时圣水大概率已接近满。本质变化：从"满费+血牛齐备才启动"的一次性门，改成
"先攒费、后等件"的**两阶段状态机**，让 hold 帧在训练分布里真正出现，
PPO 才有样本学"攒费比泄费好"。

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
- **9j 复训事故与修复（2026-09-09，勿重蹈）**：首启 ~360 步起 value loss 冲到
  43 万、adv≈-671、gnorm≈90 万、clip 84%、更新连发——根因是
  `train_solo._new_episode_reset` 漏了 `nonlocal ep_obs/ep_rew/...` 12 个缓冲区，
  Python 把重新赋值绑成嵌套函数局部名，外层缓冲从未清空 → 第一局 done 后每迭代
  重复 flush/结算，平局惩罚 −10 在泄漏缓冲上逐帧堆积（δ≈−10×γλ^k，adv −671 正是
  该形态；首局恰好打满 360 步才结束，30 步冒烟测试跨不过局边界所以没拦住）。
  修复：缓冲区一并 nonlocal；回归哨兵 `test_solo_mode_smoke`（max_ep_steps=4 强制
  跨局，断言 `ΣPPO批 ≤ 环境步数`——每步至多产 1 条过渡是硬上界；已做 buggy 版
  反向验证：8 步入批 20 条被正确拦截）。**教训：①嵌套函数重绑定外层变量必须逐个
  nonlocal；②冒烟测试必须跨过局边界（max_ep_steps < total_steps），只查落盘不查
  统计量的 smoke 是盲的；③此类事故日志自带指纹——adv 均值 ≈ −惩罚/帧、value loss
  数量级漂移、更新连发，看日志头 30 行即可定位。**
- **9j 复训结果取证（2026-09-09，`scripts/forensics_response.py`，816 局对照）**：
  训练在 90k 被显卡驱动二次崩溃带走（无 traceback 的静默退出=外部终止），从 88k
  checkpoint 续训完成；终点对照 vs baseline0 0.625 / vs baseline_prev 0.688，全程
  vs baseline0 均值 ≈0.42（平台期，与带闸门旧 run 结论一致——winrate 不动是预期内）。
  **行为指标**：①时序病理基本消失——防守响应率 38.1%→**61.5%**（与对手 59.1% 持平），
  单边堆牌 45.8%→**9.5%**（A 层惩罚单边推进的直接证据）；②**幽灵动作率**
  （bundle 里部署位置非法、被引擎静默拒绝的尝试，y≥20 格即越区）20.2% vs 带闸门
  36.0% / 无闸门 28.7%——非法尝试下降但未归零，训练时白白浪费 1/5 出牌机会且
  吃 invalid_penalty；③空间拦截率（落点在最近过河敌军→我方塔之间带内）4.6% vs
  旧 run 21.7%——**看似倒退实为口径混淆**：新 run 威胁部署集中在 y=12-16（桥头）
  而旧 run 在 y=4-8（塔前）。桥头拦截在战术上未必更差（若敌军也聚在桥头），但
  C 层 placement_hint="bridge_front" 可能过度吸引了落点；待"落点-敌军相对几何"
  细口径定论。
  **取证口径教训**：①replay bundle 记录的是**策略尝试**，validate 拒绝的非法动作
  也会被记下并留在录像里——空间统计必须先剔除幽灵动作（本次 709 个"威胁部署"
  里 208 个是幽灵，直接污染落点分布）；②评估 replay 的 bundle 是 side0(main)=P0
  本地格、世界=(x+0.5,y+0.5)，opp_played 是世界坐标，半场判定用 P0 塔 y≈3-6.5
  互证（同 v2 教训）；③中途变更 replay 目录相对路径会拼错 games 列表，导致
  "部署成功产生 y=24 单位"的假象——diff 出的"新单位"其实是另一局的数据。
- **「防守」口径重审（2026-09-09，用户纠偏："放牌不是为消灭敌军就不叫防守"）**：
  上述 response_defense（敌过河期间出牌）是**时机口径**，不是防守语义。改用
  **接敌口径**（部署点 5 格内 8s 内我方新单位与敌军 troop 是否同框交战）重判
  economy_9j（80 局抽样，剔开局 10s 铺场与幽灵动作）：
  - **威胁场景部署 551 次中 93% 从未接敌**；未接敌部署 vs 最近敌军相对位置：
    **64% 部署在敌军身后**（敌 y≈4-8 已站塔前、我兵 y≈12-16 桥头，背向而行），
    仅 10% 在敌军前进方向前方。桥头防守不是"流派选择"而是**追尾式病理**：
    防守单位被放在敌人**已经离开**的位置，8 秒内碰不到敌人。
  - 对照带闸门无三层 run 接敌率 25%、旧无闸门 38%——9j 的"防守响应率 61.5%"
    是时机上的进步、语义上的退步：模型学会了"敌过河时要出牌"，但把牌扔在
    桥头模板格，不是扔在敌军行进线上。C 层 placement_hint="bridge_front" 的
    固定桥点偏置是首要嫌疑（威胁在 y=4 也提示去桥头）。
  - **修复方向（待做）**：①C 层拦截点改为"威胁单位当前位置→其目标塔的连线
    前方 2-3 格"，随敌军深度滚动，而非固定桥格；②奖励侧给"防守部署→接敌"
    记账（threat_calc 已能标注交战区间，可做 shaping 或至少做诊断工具）；
    ③观测侧确认部署后单位行进方向是否可见（远程单位站桩属性已知）。
- **9k 拦截几何落地 + 短程验证（2026-09-09，提交 94ce339）**：
  - **几何分派定稿（用户防守原则）**：防守核心 = 增加可操作时间/减少消耗/
    尽量双塔同打。C 层 hint 废除固定 bridge_front → 按威胁深度×类型分派：
    刚过河/中段（y≥9）→ `intercept_mid`（威胁→塔连线中点靠敌侧，用户口径
    "中间靠敌军的那一格"）；深威胁（y<9）高血坦克/攻建筑 → `pull_aggro`
    （贴身敌侧 1-1.5 格，MiniPekka 贴 Giant 打满输出）；深威胁其它 →
    `king_front`（塔后拉扯借塔输出）。script_defender 同步三档（中点靠敌
    1.5 格/再退 1.5/再进 2；近战反制卡对高血威胁自动贴身档）。
    PLACEMENT_HINTS 7→8 维（PLAN_DIM 57→58，旧 ckpt 尾零兼容）。
  - **接敌率进 forensics 正式指标**（部署后 8s 内、部署点 5 格内敌我 troop
    同框；同时剔除 y≥20 幽灵动作）。基线：economy_9j 7.4%。
  - **9k fresh 20k 验证的三个发现**：①方向分布翻转——追尾 64%→31%、迎前
    26%→69%，几何语义已被 hint 通道表达；②但 fresh 20k 模型中段带（y8-16）
    部署仅占 2.6%（9j 100k 是 51%）——hint 是软偏置，fresh 探索期吸收不了，
    **接敌率验证必须热启动**（fresh 反而把 50% 部署扔进 y=0-4 塔后死角，
    接敌 4.4%<9j，属训练量不足而非几何错误）；③接敌机制真相：当前接敌主要
    靠"敌军撞上来"（追尾桶接敌 10.4% vs 迎前 2%）——因为迎前桶大量是塔后
    死角蹲守（敌被塔吸仇恨停在塔前，我兵在塔后 5 格外永远碰不到）。
    **落点几何 + 单位主动走位**两者都影响接敌，后者靠训练吸收。
  - **教训：main_init 热启动必须显式传 plan_dim/belief_dim**——旧 ckpt 元数据
    （57/23）会把 main 建成旧维度，随后 _sync_frozen_copy 拷进新维度网络即
    shape 失配崩溃（run_league 的 main_init 路径无兼容分支，已修 train_solo）。
  - **9k_ft 热启动验证（12k 步，从 9j 100k ckpt）——拦截几何生效确认**：
    接敌率 7.4%→**10.5%**；中段带（y8-16）部署占比 51%（9j）→2.6%（9k fresh）
    →**59.7%**（9k_ft，hint 被吸收的直接证据）；塔后死角（y0-4）50%→**0.3%**。
    按方向拆接敌率：迎前 **16.0%**、平行 15.5%、追尾 6.8%——"部署在敌军前进
    方向前方"的单位主动接敌能力是追尾的 2.4 倍，**主动拦截取代被动挨撞**。
    剩余病理：追尾部署仍占 60%（敌军 y=4-8 深驻时 hint 判 intercept_mid 但
    落点仍偏河侧）——下一刀是 hint 的"敌深度"分界从 y=9 下调（或按敌军速度
    预判行进线交点而非当前位置）。
  - **结论**：防守行为链（时机→落点→主动接敌）三层逐级改善：
    9j 修时机（响应率 38→62%）、9k_ft 修落点语义（中段 60%+接敌 10.5%）。
    winrate 平台期下这是行为质量的真实爬坡。

### 四卡组对手池 + P0 掩码优化（2026-09-08，FirstLight 参照）

**FirstLight CR 参照（`E:/FirstLight_CR`，读后定稿）**：native 引擎 + IL 252k 回放 +
PPO，速猪专精模型上过名人堂。可借鉴的是**卡组口径的三表**：`card_support.json`（122 张
支持卡）、`card_specs.json.gz`（152 条 native 提取数值）、`card_logic.json.gz`（机制规则）。
本引擎 148 张卡全覆盖其 122 张（Card() 全部可构造）；唯一缺口 = IceGolem（gamedata 无
此卡），速猪位用 IceGolemite 同价替代。其训练史教训（固定 IL 高胜率歪路 = 296/306 胜局
对手 ≤3 次出牌）与本仓库"对照曲线防自欺"结论同源。

- **四种卡组**（`docs/four_decks_manual.md`，逐卡数值+战术角色手册；2026-09-08 用户换血）：
  速猪2.6（2.62）/ 石头人（3.62，替换皇家巨人）/ X弩（3.25）/ 巨骷髅攻城槌（3.88，
  替换双线快攻，用户指定卡单）——覆盖速攻/重推/自闭/攻城组合四 archetype，
  全部 deploy 实测通过。`FOUR_DECK_SET` 定义在 `rl/opponents.py`。
  **卡名映射陷阱：官方 Zappies = 引擎 `MiniSparkys`（gamedata id 26000052，
  TID_SPELL_MINI_ZAPMACHINE），不是 `ZapMachine`（26000033，6费电击机器）——
  对卡先对 id 再对名字；GiantSkeleton 亡语炸弹走 TimedExplosive 链路
  （death_spawn_data 带 deathDamage 无 hitpoints），实测 209伤/3.02s引信/3格半径/
  对塔 200%；BarbLog/Vines 等法术受部署区限制（仅己方半场）。**
- **接入**：`--deck-set four`（defend 对手每局从四卡组抽一副，逼出"对牌"能力）/
  `--deck-set list:卡1,...`（显式镜像）；`TrainConfig.deck_set` 落 config.json。
  `_new_episode_reset` 经 `env.deck1_factory` 注入（FollowerOpponent 无 deck 属性 →
  getattr 缺省 None，frozen/hist 保持镜像不变）。
- **卡池修复**：`build_card_pool` 的 `"Tower" in n` 过滤误伤 BombTower/InfernoTower
  （名字带 Tower 的合法建筑卡）——注意 `endswith("Tower")` 同样误伤，唯一正确口径是
  `n.startswith("King_")`（塔类卡全部有 King_ 前缀）。卡池 142→144。
- **P0 掩码优化（预期 2.8×，实测远超）**：`legal_cells` 的 576 格循环内每格重建
  Card（部队 576 次/法术 ~2300 次/调用），profile 显示 `Card.__init__` 占 92% 耗时
  （0.730/0.794s，其中 dict.get 2.25M 次 0.23s + set_level 0.09s + Projectile 0.11s）。
  改法：`card_info` 参数贯通 legal_cells → _position_legal → _spell_* → backline 全链路
  （构造 1 次循环外复用）+ `_card_static_cache` 按 (卡名, Card.default_level) 缓存。
  **实测：legal_cells 单卡 4.37→0.23ms（19×）、整帧 4 手牌 18→3.0ms（6×）、
  训练循环 3-4→14.4 步/s（单 env CPU）**。
  **验证纪律（改判定逻辑必做）**：改动前后各跑 `scripts/_mask_diff_snapshot.py`
  dump 位图（8 状态×双方×8 手牌=128 张：空场/压境/双倍期/推进坦克/残血塔/群杂/lv14），
  逐位 diff 必须全等——本次 128/128 一致才提交。
- **economy_10d 验证 run**（12k warm-start 从 9j 100k，四卡组 defend 池，selftest 74/74）：
  曲线 eval@0 0.625 → 2k 0.875 → 4k-12k 多点 1.000（vs baseline0/prev 同步爬升，
  终点双对照 1.000）；**eval@0 16 局 54.4s（worker 16 全成）vs 9j 时代串行 ~30min**——
  页面文件充足时 spawn worker 是数量级收益。注：本轮 winrate 大幅爬升是 defend 对手
  从默认 8 卡换成四卡组所致（对手变强，对照基线同步变强，*不能*与旧曲线直读对比）；
  行为取证待做。
  **10d 结论追溯失效（2026-09-09 词表 v2 发现）**：当时观测词表 ENTITY_NAMES 只有
  原版 8 卡的 13 个名字——四卡组对手的 HogRider/Golem/Xbow/BattleRam 等实体**全部
  不进 grid 观测、手牌编码 0**，defend 对手对模型近乎隐形（只剩掩码/血量变化的间接
  痕迹）。"四卡组逼出对牌行为"在 10d 并不成立，1.000 曲线含大量 vs 隐形对手的水分。
  econ_10d 的"对手变强"实为"对手出牌合法但模型看不见"。
- **词表 v2 + 连弩镜像卡组（2026-09-09，提交 d71dfc9，用户：原版 8 卡无实战价值）**：
  - **ENTITY_NAMES 13→177**（全卡池实体名 dump：部署+死亡刷出+法术包装器+弹射物；
    加手牌卡名与实体名两套的实测差集 Clone/Log/Snowball/Graveyard/Lightning 等 13 个
    ——手牌编码用卡名 "Snowball"、场上包装器叫 "SnowballSpell"）。**旧 13 位序冻结**
    （前缀语义保留），entity_emb 行兼容分支进 load_checkpoint（旧行拷贝/新行从零，
    真实 100k ckpt 验证逐位保留）。
  - **belief token 事件 one-hot 自动扩容**：8 卡卡组 71→563 维（OPP_EVENT_DIM=
    len(ENTITY_NAMES)+3），belief_mlp 尾零兼容前 71 列语义不变。
  - **CARD_TYPES 不扩**：引擎还有 area_effect/projectile/bomb 三个 data.type
    （垫片实体），类型折叠进 "spell"（one_hot num_classes=4 不动 → CNN 权重兼容）；
    垫片缺的攻击/移动物理字段改 getattr 补零读取（新垫片漏补不再崩观测）。
  - **DEFAULT_SOLO_DECK → 连弩 2.9**（Xbow/Tesla/Skeletons/IceWizard/Archer/Knight/
    Log/Fireball，与 FOUR_DECK_SET X弩同族）。
  - **economy_10e 20k 测试 run**（warm-start ungated 100k，deck-set four）已启动；
    教训：**economy 预设 eval_workers=0（串行）**，10d 的 16 是当时显式传的——
    忘传则每次评估 857s（串行）vs ~240s（16 worker），长 run 必带 --eval-workers 16。

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

---

## `ab_valnorm_20k` 20k 步验证跑：四条跨会话必须记住的结论（2026-09-11）

完整判读见 `docs/ab_valnorm_20k_verdict_2026-09-11.md`；run 目录 `src/clasher_new/runs/ab_valnorm_20k/`。

### ① `step` = 决策帧，不是更新次数 —— 别再高估训练进度

`train_solo.py:1064` 每轮循环只做一次 `env.step`（`:1104`），**1 step = 1 个决策帧**；
PPO 每收满 `update_interval=128` 帧才更新一次（`:1150`）。所以：

- **20k 步 ≈ 156 次更新 ≈ 55~80 局自对弈**（`max_ep_steps=360`，实测均局 250~360 帧）；
- 100k 步（9j）≈ 280 局；对标 Atari PPO 的 10M~50M 帧，**本项目 20k 只有其 1/500~1/2500**。
- **推论**：任何"跑 N 万步看提升"的实验，先换算成局数；"20k 步没提升"在样本量上就是必然。

### ② `ratio ≡ 1.000` / `clip_frac ≡ 0%` 是结构性的 —— 该判据作废

rollout 与更新共用同一份权重、`n_epochs=1`、`lr=3e-4`、梯度裁到 0.5 → ratio 偏离只有 1e-3。
（`rl/ppo.py` 模块 docstring `:12-22` 已写明。）审计计划 v1 的 G1 里"ratio 离开 1.000"
**永远不会触发**，应替换为 **`explained_variance`（EV）**。

### ③ 新发现：critic 的解释方差 ≈ 0（本轮唯一的新结构性问题）

`ReturnScaler` 终态 `count=19840 mean=1.588 m2=2559652` → 回报方差 **129.0**、std **11.36**；
而 `vraw`（未缩放 MSE）中位 ≈130、末值 203 → **EV = 1 − MSE/Var ≈ 0（末值 −0.57）**。
即价值网络对回报的解释力**等于"恒定预测均值"**。

- 所以 P0-1 的 `value_norm=running` 只解决了**梯度配比**（`v/p` 2.43 → 0.60~0.95，达标），
  **没有也不可能解决 critic 拟合失败**。两者是独立的病，别指望调 `vf_coef`；
- 旁证：优势均值系统性偏正（`adv=+13.978±2.884`，收益 std 仅 11）→ critic 系统性低估回报。
- **下一步取证**：加 `explained_variance` 诊断；再判断"回报是否可从状态预测"——
  若镜像自对弈使状态-胜负近乎独立，价值头就不是调参问题，**必须提高非镜像对手占比**
  （当前对手池 frozen 仍占 70%）。

### ④ 门禁阈值口径错位 —— `gates.json` 现在会 PASS/FAIL，但阈值没有意义

`config.py:150` 的注释说"engagement_rate 9k_ft 实测 10.5% → 阈值 9.5"，那个 10.5 来自
一次性脚本 `scripts/forensics_response.py`；而训练内建指标（`train_solo.py:349`）对
**同一批 9k_ft 权重**实测是 **28.3 / 34.0 / 38.1**（step 0 的 main/baseline0/baseline_prev 三连测）。
**差约 3 倍。** 在重标定之前不要把 gates.json 的 PASS 当结论。

### 附：本次自校准的噪声地板（40 局）

step 0 时 main = baseline0 = baseline_prev **权重完全相同**，却测出
胜率 **0.625 / 0.525 / 0.600**、接敌率 **28.3 / 34.0 / 38.1** →
**胜率 1σ ≈ 0.078（与上报 SE 吻合）、接敌率 1σ ≈ ±5pp**。
以后判读行为指标，先用这条地板过滤波动。

### 附：结果指标（20k 步净变化为零，供后续对照）

| step | main（镜像，结构性≈0.5） | vs `baseline0` | vs `baseline_prev` |
|---|---|---|---|
| 0 | 0.625 | 0.525 | 0.600 |
| 8000 | 0.550 | **0.850** | 0.700 |
| 12000 | 0.525 | 0.450 | 0.500 |
| 16000 | 0.625 | 0.625 | 0.475 |
| 20000 | 0.450 | 0.525 | 0.500 |

行为病理未改善：`elixir_avg` 全程 1.57~2.26（**依然不会攒费**）、`deploy_per_game` 末段反降至 19.7。
`WinError 1455` 本次复现 2 次、1 次 eval 降级串行（成因仍按"全局操作约定"记为未定）。

---

## 计划 v2 落地 + `prod_200k_valnorm_ev` 长跑（2026-09-11）

计划 v2 = `docs/rl_training_fix_plan_v2.md`，是 v1 的**修订版**（v1 结构仍有效）。
本次改了 6 处，每条都有 20k 判读的实测依据——**后续会话不要再改回去**：

| # | 改动 | 依据 |
|---|---|---|
| 1 | **G1 判据 `ratio` 离开 1.000 → 删除**，换成 **`explained_variance ≥ 0.2`** | `ppo.py` 单轮 on-policy + n_epochs=1 ⇒ `ratio≡1.000` 是**结构性恒等**，判据永不触发（判读 §3.1） |
| 2 | 新增 **`PPOTrainer.explained_variance`** 诊断 + stats 字段 + 日志 `EV=` | `value_loss` 被 `s²` 除过、跨版本不可比；EV 无量纲，是"critic 是否在学"的唯一干净判据（判读 §3.2） |
| 3 | 行为门禁**绝对阈值 → 相对本 run 首个评估点**（`{"rel":">=","frac":0.5}`；baseline 存 `gates.json`、首点只建基线不判定） | 阈值 9.5 来自一次性脚本，训练内建指标对**同一批 9k_ft 权重**实测 28.3~38.1，差约 3 倍 ⇒ PASS/FAIL 语义是假的（判读 §5） |
| 4 | 预算口径 **`step` → 局数**，history 新增 `cum_games` | `step` = 决策帧；20k 步 ≈ 156 次 PPO 更新 ≈ 55~80 局（判读 §2） |
| 5 | 对手池 **`frozen 0.7/hist 0.2/defend 0.1` → `0.5/0.3/0.2`**（`config.DEFAULT_OPP_MIX` + `TrainConfig.opp_mix`） | critic EV≈0 的怀疑之一 = 镜像自对弈对称性使"状态→胜负"不可预测；提高非镜像对手占比给 critic 可学信号 |
| 6 | 主循环改成 **先评估、后同步冻结副本** | 旧顺序（同步→评估）在两步长整除时使评估对手恒为**刚同步的 main 自己** ⇒ main 曲线恒镜像、期望 0.5、结构性无信息（判读 §4） |

**兼容性红线**：`rl/ppo.py` 被 `run_league`/`flow_league`/`train_follower`/`train_prophet`
共用 ⇒ 函数默认仍是旧行为（`value_norm="none"`、`diagnose_every=0`），新能力只经
`TrainConfig` 显式开启。`economy` 预设的 `value_norm` 已设为 `"running"`（dataclass 默认不动）。

**`prod_200k_valnorm_ev` 启动配置（照抄即可复现）**：

```bash
python rl/run_league.py --mode solo --config economy --config-name prod_200k_valnorm_ev --fresh \
  --total-steps 200000 --steps-per-eval 10000 --n-eval-games 40 --eval-workers 16 --device cuda \
  --value-norm running --adv-norm scale --diagnose-every 10 \
  --main-init runs/economy_9k_ft/solo_main.pt \
  --hist-seed-dir runs/economy_9k_ft --hist-seed-dir runs/economy_9j
```
日志 `docs/train_prod_200k.log`，产物 `src/clasher_new/runs/prod_200k_valnorm_ev/`。
实测吞吐 **26.2 步/s**、eval≈205s/点（16 worker 未被降级）⇒ **总墙钟 ≈3.2 h**。

**判读口径（跑完照此读）**：主判据 = `EV`（≥0.2 → critic 可救，继续堆量；持续 ≈0 或为负 →
停堆量，转 P1-2 奖励 / P1-4 偏置，或把价值头从共享 trunk 拆出）；次判据 = 对照曲线
vs `baseline0`/`baseline_prev` 的净漂移 **≥2σ（≈0.16）**；行为指标只读 `gates.json` 的
**相对退化**报警。**不判** main 曲线与单点胜率。

**首批实测（启动 ~5 分钟）**：`EV` 为 **负值**（-2.4 ~ -10.3），即价值网络**比"恒定预测批均值"
还差**（比 20k 判读里推断的"≈0"更严重）。`vraw` 仍 10~150、`v/p=0.50`。
`gates.json` 基线自动标定为 `engagement 28.3 / ghost 24.7`，**与 20k 跑 step-0 完全一致**
⇒ 相对门禁的口径自校准生效（这两个数就是以后判读行为指标的地板）。

## ⚠️ 负 EV 的真根因：GRU 输入饱和（2026-09-11，**推翻 v2 §4 的病因判断**）

`prod_200k_valnorm_ev` 跑到 step ~54k 时 EV 仍全负，据此做取证（报告
`docs/value_channel_saturation_diagnosis_2026-09-11.md`，计划 `docs/rl_training_fix_plan_v3.md`）。

**跨会话必须记住的结论：**

1. **critic 恒为常数，不是"学得差"**。743 帧 rollout 里 value 输出 std=0.028 而
   回报 R std=5.50；`EV_global = −0.5817` 与"恒定预测器"恒等式
   `−bias²/Var(R) = −0.584` **精确吻合**。所以负 EV 是**结构性**的，不是调参能救的。
2. **根因 = GRU 输入饱和**：`enc = relu(enc_fc(fused))` 的 L2 范数 ≈533
   （CNN 输出 `grid_feat` 常数分量 ≈468、跨帧 std 仅 1.06）→ `GRUCell` 的 tanh
   候选饱和 `n(abs_mean)=0.994` → 隐状态 h 跨帧 std **2.6e-5**（数值恒定）。
   对照实验：把 enc 归一化后 h 跨帧 std 立刻回到 **0.116**（差 4~5 个数量级）。
3. **影响面超出 critic**：`slot_head(h)`/`cell_head(h)` 同样吃常数 h ⇒ 策略的状态
   依赖**只剩手工 `BeliefPlanner` 的 plan 偏置**，神经网络在策略里基本开环。
   这解释了为什么"行为有爬坡但 winrate 不动"。
4. **病理是系统性的**：economy_9k_ft / 9j / 10e / ab_valnorm_20k / prod_200k
   五个 ckpt 的 `GRU n(abs)` 全 ≥0.98。**没有任何一个 run 的 GRU 是活的。**
5. **EV 判据本身也被测量口径污染**：训练用"128 连续帧/批"算 EV，而相邻帧
   `corr(R_t,R_{t+1})=0.990`、批内 Var(R) 仅为全局 0.32 倍 ⇒ 该口径把 EV
   **放大约 3 倍**（实测 −0.58 → −1.74）。dashboard 上那个 EV 是"批内 EV 均值"。
6. **v2 §4 的分支判断（奖励尺度 / 价值头结构）病因判错**，在 GRU 修好前不要再按
   它去调奖励或拆价值头。v3 的 P0-A 是 `follower.py` 的 `enc_ln = nn.LayerNorm(hidden)`
   （enc 后归一化），**旧 ckpt 不可续训，必须 --fresh**。

**待验证（不得当成已知）**：归一化后 critic 的实际上限（EV 能到多少）——
留出法线性探针在 10 局样本下三组全为负，样本量不足，**不能**据此宣称修好就能达标。

**诊断脚本（已入库，复用即可）**：`scripts/diag_critic_ev.py`（5 种 EV 口径）、
`scripts/diag_value_head.py`（h/enc 方差 + GRU 门）、`scripts/diag_encoder_scale.py`
（fused 分量尺度）、`scripts/diag_gru_ablation.py`（归一化对照实验）。

### v3 实施记录（2026-09-11 当日已落地）

1. **P0-A 已改**：`rl/follower.py` 新增 `self.enc_ln = nn.LayerNorm(hidden)`，
   `_encode` / `_encode_batch` 返回 `self.enc_ln(relu(enc_fc(fused)))`。
   随机初始化下实测 enc 范数 7.98（原 533 是**训练后**的量级漂移）；
   把 `enc_fc` ×300 仍被 LN 吸收（范数 8.00、h_std 0.21 不变）。
2. **P0-B 已改**：`rl/train_solo.py` EV 改**池化口径**（评估窗口内累积逐帧
   `(v,R)` 合并算一次），保留 `explained_variance_batched`（随机打乱分批）作对照；
   单步日志的 `EVb` 明确标注为批内口径（放大 ~3 倍，只看趋势）。
   `rl/ppo.py` 增 `PPOTrainer.last_ev_pairs` 承载逐帧 `(v,R)`。
3. **P0-B/C 已改**：新增 `rl/diagnostics.py`（`gru_vitality` / `check_vitality`，
   门槛 `h_std>0.05` 且 `n_abs<0.9`）。评估行与 dashboard 落 `h_std`/`gru_n_abs`/
   `value_std`；`train_solo` 主循环维护最近 96 帧供探针（纯前向，不推进 env）。
4. **回归测试**：`rl/selftest.py::test_enc_layernorm_gru_vitality`。
   负对照用 `enc_fc×300` + `enc_ln=Identity` 复现饱和（enc 203、|n| 0.978），
   **不**用裸随机网络（裸随机网络 enc 范数仅 ~0.5，不复现）。
5. ⚠️ **不要用 `main_init` 续训 v3**：旧 ckpt 无 `enc_ln` 键，加载后 LN 是默认
   1/0（未学过），且旧权重是饱和态下学的 → 验证跑必须 `--fresh`。
   （`load_checkpoint` 对缺失键保持新初始化，不会报错，属静默陷阱。）

### 5k 验证跑判读 + 分支判定（2026-09-11，计划 v3 §3.6）

run `src/clasher_new/runs/fix_gru_ln_5k/`（13 局、2 个评估点、日志 `docs/train_fix_gru_ln_5k.log`）：

| 指标 | 饱和态基线 | 门槛（v3 §2） | @2500 | @5000 |
|---|---|---|---|---|
| GRU n(abs_mean) | 0.994 | <0.9 | **0.456** | **0.552** |
| h 跨帧 std | 2.6e-5 | >0.05 | 0.184 | **0.0446**（末点又跌破） |
| value_std | ~0.001 | — | 0.361 | 0.167 |
| EV（池化口径） | −0.58 | >0 且上升 | −0.238 | **−0.0595**（4× 缩减，未过零） |

1. **根因修复生效**：`n_abs` 0.994→0.46/0.55、`h_std` 提升 3~4 个数量级 ⇒ v3 §0 的病因链
   （enc 未归一化 → tanh 饱和 → h 冻结 → critic 常数）已被切断。
2. **EV 未过期，5k 判不了分支**：单调上升但 13 局/2 点无法区分"卡住"与"仍在爬"。
   **本轮不执行 v2 §4 的高侵入项**（拆价值头 / 改奖励 / P1-4 偏置）——v2 原文要求的是
   "EV ≈0 **且** 多样性提高后行为指标改善"的**联合**信号，而本轮行为指标反而退化
   （engagement 35.6→7.9、ghost 3.6→17.9），且退化可由下一条的对手池偏离解释。
   先补测量、恢复配比，再跑 20k 定论（协议见 v3 §3.6.4）。
3. **测量缺口（本轮最该记住的教训）**：v3 §2 验收表第 3 行 `value_std / 批内 R std > 0.3`
   的**分母 `r_std` 全仓根本没有实现**（`rl/diagnostics.py` 写着"调用方另测"，而调用方
   `train_solo` 没测）⇒ **该门槛自始至终不可判读**，P0-B-2 声称的 3 项指标实际只落地 2 项。
   已补：`eval_and_write` 从 `last_ev_pairs`（**未缩放**量纲，与 value_head 原始输出同尺度）
   算 `r_std`（池化）/`r_std_batch`（128 帧批内 std 再平均）→ `value_std_ratio`，
   评估行 + dashboard 新列 + `THRESHOLDS["value_std_ratio"]=0.3` 告警。
   **教训：写进验收表的指标，必须确认分母真的存在；"调用方另测"= 没人测。**
4. **对手池偏离（5k 的混杂变量）**：未传 `--hist-seed-dir` ⇒ hist 槽为空 ⇒ 自动退化为
   frozen **0.714** / defend **0.286**（`_OpponentPool._ensure_hist`），偏离 v2 §5 的
   0.5/0.3/0.2。`opp_mix` **没有 CLI flag** ⇒ `--hist-seed-dir`（可 append 多次）是恢复
   文档配比的**唯一**受支持途径。注意 hist ckpt 是 pre-v3 的，会带默认 LN 出场（行为与其
   记录 elo 有偏），PFSP 按胜负重权。
5. **编码崩溃（新发现，非 v3 引入，会真崩训练）**：GBK 控制台/重定向管道下
   `print("⚠️ ...")` 抛 `UnicodeEncodeError` → 被 `except Exception` 吞掉 →
   **except 处理器里的 `{e!r}` 又内嵌同一不可编码字符 → 二次抛错、直接崩**
   （`test_solo_resume` 与 `test_opponent_pool_mix` 实际复现，后者崩在对手池加载）。
   已修：`rl/run_league.py::_force_utf8_stdout`（stdout/stderr → UTF-8+replace）、
   `rl/diagnostics.print_safe`（逐条 print 兜底，`follower`/`train_solo` 共用）、
   `train_solo` except 里的 repr 转 ASCII。
   **教训：catch-all except 里再 print 同一个异常对象 = 二次抛错路径。**
6. **P0-C 已落地，但要说清范围**：`check_policy_architecture` 只做**静态**护栏
   （`enc_ln` 缺失 / 被 `nn.Identity` 替换）；启动时一帧都没有，**测不了** h_std/n_abs，
   真实活力仍在评估点测，且告警已落盘 `stats["vitality_warns"]`（旧实现只 print → 事后
   无法在 state/dashboard 追溯）。v3 §1 的 `h_std>0.02` 是笔误，统一为 **0.05**（与 §2/
   代码一致；0.02 仅作 dashboard 黄带）。
7. 顺手护栏：`follower.load_checkpoint` 对缺 `enc_ln.*` 的旧 ckpt 显式告警（原先
   `load_state_dict(target)` 形状恒匹配 → 完全静默）；`_probe["ev_pairs"]` 加 20 万帧上界
   （`--steps-per-eval 0` 时原先无界增长）。

### 20k 验证跑判读（2026-09-11，`fix_gru_ln_20k`，计划 v3 §3.7）

run `src/clasher_new/runs/fix_gru_ln_20k/`（62 局、8 个 EV 点、训练循环 2096.9s、
**0 降级 / 0 Traceback**；`opp_mix` 已恢复 0.5/0.3/0.2，hist ckpts=12）。

| step | 胜率 | EV(池化) | h_std | n_abs | v/R std |
|---|---|---|---|---|---|
| 2500 | 0.388 | −0.0734 | 0.0531 | 0.493 | 0.0225 |
| 5000 | 0.400 | −0.4183 | 0.0320 | 0.446 | 0.0036 |
| 10000 | 0.200 | −0.0360 | 0.0356 | 0.468 | 0.0068 |
| 15000 | 0.475 | −0.0758 | 0.0364 | 0.499 | 0.0063 |
| 20000 | 0.463 | **−0.0079** | 0.0381 | 0.533 | 0.0114 |

1. **饱和修复成立且稳定（本轮最重要的确认）**：`n_abs` **8/8 点 < 0.9**（0.446~0.535，对照
   修复前 0.994）；`diag_value_head.py` 实测 GRU 门 `z=0.471 / r=0.494 / |n|=0.525`（健康）；
   `enc_ln` 生效且未被绕过（`||W||=11.31`、γ≈0.707，post-LN `||enc||≡11.386`）；
   **pre-LN `||enc||=115.6`，对照修复前 533（降 4.6×）⇒ "`enc_fc` 被重新拉大"不成立。**
2. **但 critic 仍是有效的"常数预测器"**：新增的 `value_std_ratio` **0/8 达标**
   （0.0035~0.0225 vs 门槛 0.3）= value_head 输出波动只有回报波动的 **0.4%~2.3%**；
   EV **0/8 过零**（max −0.0079 末点），前 4 点均值 −0.154 → 后 4 点 −0.084（略升，噪声内）；
   `diag_critic_ev.py`（24 局/7179 帧）：`v std=0.124` vs `R std=9.747`、`RMSE 10.38 ≈ std(R)`、
   `corr(v,R)=−0.008`、**回归斜率 R~v = −0.603（负）**、`corr(局均 v, 局均 R)=+0.223`（n=24 下
   1σ≈0.22 → **不显著，不得宣称"信息在 h 里"或"不在 h 里"**）。
3. **新状态（v3 §3 分支表之外）：饱和已修、GRU 已活，但 encoder 给 GRU 的输入跨帧几乎不变。**
   实测 post-LN `enc` 跨帧每维 std `mean=0.00424 / median=0.00043`；**融合层尺度失衡**：
   `grid_feat` 范数 **101.0** / 跨帧 std **0.43**（占 fused 102.99 的 98%，几乎不变化），
   而真正在变的 `scalar` 17.6/std **6.97**、`hand_feat` 6.4/std 0.50、`plan_f` 1.65/0.199、
   `belief_f` 0.65/0.104 被稀释 **16×~156×**。
4. **行为**：`gates.json ok=false`，唯一失败项是 ghost_rate（3.6→**14.8** > 7.2）；
   engagement 35.6→**24.5 ≥ 17.8 PASS**（5k 跑该项 7.9 FAIL，恢复配比后转正）。
   胜率 0.20~0.60、末 0.463（1σ=0.078，无净漂移）。对照组剧烈振荡
   （vs `baseline_prev` 0.625→0.087→0.700→0.200→0.900→0.188→…，远超 1σ）⇒ **自对弈 cycling 指纹**。
5. **下一步候选（未执行，需拍板）**：**A（首选，§1 P0-A 备选）** `grid_feat` / 逐分量加归一化
   再 fused（单变量、直击测到的失衡；改架构 ⇒ 必须 `--fresh`）；**B** 先把 EV/探针样本量做够
   （24 局不足以区分 A 与"状态本就不预测胜负"）；**C（v2 §4 拆价值头/改奖励）本轮证据不支持**
   （enc 未拉大、GRU 未饱和、EV 后段略升，且 v2 §4 的联合前提只部分成立）。
6. **教训（测试纪律）**：`test_eval_solo_parallel` 断言串行≡并行**逐位相等**，但它依赖进程内
   累积状态 —— 我按自定义顺序批量跑测试时它 FAIL（mean_reward 4.095 vs 4.171），
   单独跑 1~2 次全 PASS，**官方 `rl/selftest.py` 全量 100% PASS**。
   ⇒ 判读"是不是我改坏了"的正确姿势是**跑官方全量**，不是自定义批次的顺序结果。
   另：该断言的存在意味着**一旦 eval 静默降级串行，统计量与并行点不再逐位可比**
   （本轮未发生降级，0 次）。

### P0-A 备选 A（`grid_ln`）落地 + 20k 验证判读（2026-09-12，计划 v3 §3.8）

用户拍板走 A。`rl/follower.py` 新增 `self.grid_ln = nn.LayerNorm(cnn_out)`，`_encode`/
`_encode_batch` 在 `self.cnn(x)` 后立即归一化（只上 grid_feat 单变量；不动 scalar/plan/belief，
理由见 §3.8 头注）。`check_policy_architecture`/`load_checkpoint` 告警同步覆盖 `grid_ln`。
官方全量 selftest 100% PASS。

run `src/clasher_new/runs/fix_gru_ln_norm_20k/`（59 局、8 评估点、训练循环 2709.3s、
0 降级/0 Traceback；同 seed、同协议，唯一变量 = `grid_ln`）。

1. **尺度修复达成（机制层）**：`grid_feat` 范数 101→**5.87**（占 fused 98%→29%）、fused
   跨帧 std 1.64→**7.52**、pre-LN enc 115.6→**4.46**、post-LN enc 每维跨帧 std
   0.0042→**0.0215**（5×）、value_head 输出 std 0.079→**0.156**（2×）。grid_ln γ≈0.707 正常。
2. **critic 从"偏置大的常数"变"对中的常数"，仍未拟合**：EV_global **−0.134→−0.0051**
   （与 `−bias²/Var(R)` 吻合：bias −3.56→−0.56）；但 RMSE 8.74 ≈ std(R) 8.72、
   `value_std_ratio` 仍 **0/8 达标**（0.014~0.031，门槛 0.3）、局均 corr(v,R) +0.223→**−0.209**
   （n=24 不显著但符号翻转）。⇒ **修好的是对中，不是拟合。**
3. **`n_abs` 单调上升 0.461→0.745（门槛 0.9）**：此时输入已归一化（post-LN ‖enc‖≡11.47），
   饱和压力来自 **GRU 自身权重/h 增长**（h 范数 6.72→8.99），**不是输入量级** ⇒ enc_ln+grid_ln
   只修输入侧、没约束 GRU 内部漂移，20k 内 <0.9 但外推 100k 会再超标 ⇒ **修复耐久性存疑**。
4. **`vs baseline0` 崩塌（本轮最大红旗）**：step 5000 起对训练起点随机策略 **0.05~0.125**
   （v1 同期 0.28~0.6），同时打冻结副本 0.85、打上一评估点 0.80 ⇒ **非传递性/cycling 指纹**；
   v1 无此现象，与 grid_ln 相关联出现。行为门禁全绿（engagement 48.7/ghost 0.4）但相对基线
   口径被自身起点定标，**抵消不了对随机对手的崩塌**。**main vs 冻结副本 0.85 不能当"变强"证据。**
5. **下一步候选（需拍板）**：**A′（首选，零改动）** 取证 cycling——加跑 vs 新随机策略 100 局
   + 对 `runs/fix_gru_ln_norm_20k/replays/` 做早停/节奏取证，区分"绝对变弱/早停裁定/真循环"；
   **B′** 修 GRU 耐久（门/权重正则或 h 范数约束，再 --fresh 一轮 20k）；**C′** 价值头结构
   （v2 §4，须在 A′ 排除 cycling 后）；**D′** 直接堆量 100k+（成本最高、两个异常未解释，风险最大）。
   **不把 0.85 当进步、不因 EV≈0 宣称修好、不在 A′ 前改奖励/拆价值头。**

### A′ 取证结论：cycling 确凿、绝对强度 ≈ 随机（2026-09-12，计划 v3 §3.8.4）

工具 `scripts/_forensics_cycling.py`（三对阵 ×100 局，`runs/fix_gru_ln_norm_20k/`）：

| 对阵（main 侧） | 胜率 | 疑似早停 |
|---|---|---|
| main@20000 vs 冻结副本（run 回放 n=40） | 0.850 | 3/40 (7.5%) |
| main@20000 vs **baseline0**（起点随机） | **0.130** (10/84/6) | **28/100 (28%)** |
| main@20000 vs **全新随机** | **0.505** (50/49/1) | 2/100 (2%) |
| baseline0 vs 全新随机（sanity） | 0.340 | **40/100 (40%)** |

1. **`vs baseline0` 崩塌真实**（100 局 0.130），**但绝对强度≈随机**（vs 全新随机 0.505）。
2. **RPS 三角确凿**：全新随机 > baseline0 > main@20000 ≈ 全新随机 ⇒ **自对弈策略循环**。
   run 内自引用指标（打冻结 0.85 / 打上一评估点 0.80 / 行为门禁全绿）**全部失真**。
3. **僵局早停裁定是主要混淆因素**：涉及 baseline0 的对局早停 28~40%、帧数 min≈100
   （远低于 360），大量对局 `_stall_probe` 早停 + `timeout_winner` 塔血裁定，方差大。
4. **推论**：cycling 未解决前，堆量（D′）与价值头（C′）都无意义——critic 学的是循环内
   伪标签。**下一个动作必须是给自对弈加外部锚点**（评估侧锚点门禁 和/或 训练侧对手/奖励
   改造），否则内部指标（EV/行为门禁/对照曲线）不可信。
5. 遗留：main vs `SelfDefenderPolicy` 外部锚点对阵未跑（需自定义评估循环绕过
   `FollowerOpponent` 包装，留作可选取证）。

### E1 落地：固定随机锚点（绝对强度门禁，2026-09-12，计划 v3 §3.8.5）

用户拍板 E1（评估侧锚点）。`rl/train_solo.py`：每评估点追加第三组对照
`vs baseline_rand`（固定种子 99999 的随机策略，固定评估种子 90000 重打同 40 局；
`RAND_ANCHOR_WARN_FLOOR=0.35` 报警线，只报警不阻断；构造时保存/恢复 torch RNG 不扰动
训练）。selftest `test_solo_rand_anchor`，官方全量 100% PASS。

**短验证（`main@20000` vs 固定随机锚点 ×40×2）**：0.15 / 0.15（逐位一致，deterministic）。
与 §3.8.4 互证：3 个随机对手 2 个把 main 打到 **0.13~0.15**、1 个打平 0.505 ⇒
胜率高度依赖"抽到哪个随机权重"，cycling/"绝对强度≈随机"坐实；**"打冻结副本 0.85"是
自引用失真**。未来任何 run 的 `_controls_history` 都有绝对强度列可逐点读。

### E2 落地：固定随机锚点进训练对手池（2026-09-12，计划 v3 §3.9，用户拍板）

E1 只测量、E2 动手：锚点从评估侧对照升级为**训练分布第 4 槽 `rand_anchor`**，
打破自对弈 RPS 循环漂移。

- **配比**：`DEFAULT_OPP_MIX`/`_OPP_MIX` → `{frozen 0.4, hist 0.3, defend 0.2, rand_anchor 0.1}`
  （frozen 让 0.1 给锚点，仍主力）。
- **共用构造**：`train_solo._make_rand_anchor(cfg, belief_dim, device)`（RNG 保存/恢复 +
  `RAND_ANCHOR_SEED`）⇒ **E1 评估侧 `baseline_rand` 与 E2 训练侧锚点逐位一致**。
- **训练侧锚点**：`_OpponentPool` 构造 `FollowerOpponent(锚点策略, deterministic=True)`
  （镜像卡组，永不参与训练/同步/PFSP）；`sample()` hist→defend→rand_anchor→frozen。
- **附带修复（无 hist 退化归一化缺陷）**：旧实现无 hist 时 `r<hist+defend` 未受 hist
  保护 ⇒ 实测 defend **0.78**/frozen 0.22 而打印宣称 0.286/0.714。现按剩余概率归一化
  （frozen 0.571/defend 0.286/rand 0.143，与打印一致）。**5k 判读里引用的"0.714/0.286"
  是打印值不是采样值，别再照抄。**
- **兼容**：旧式三槽 mix（无 rand_anchor 键）rand 概率 0、行为不变。
- **selftest**：`test_opponent_pool_rand_anchor`（分布/权重一致性/兼容/归一化修复）。
- **验证**：`e2_rand_anchor_20k`（--fresh，20k，2500/点，40 局，worker 12，hist 补种×2）。
  判读：vs baseline0 不再崩塌（对照 run 0.05~0.125）、vs baseline_rand 末点 >0.35、
  EV/value_std_ratio 趋势、n_abs 顺带观测。

### E2 20k 判读：干预无效，cycling 未破（2026-09-12，计划 v3 §3.9.3）

`e2_rand_anchor_20k`（59 局/9 点/3419s/0 降级）：**训练侧固定随机锚点 0.1 未能锚定
自对弈动力学**——

- `vs baseline_rand` 末点 **0.050**（E1 门禁从 5k 起全程警报；对照 run 的 E1 短验证 0.15）；
  `vs baseline0` 末点 **0.075**（对照 0.125）⇒ **绝对强度崩溃依旧**；
- 对照组仍剧烈振荡、17500 点非传递（main 0.25 但 vs baseline0 0.725 / vs baseline_rand
  0.85）⇒ **RPS 循环仍在转**；
- critic 依旧常数（EV 0/8 过零、v/R 0.008~0.032 < 0.3）；
- n_abs 峰值 0.696、末点 0.596（对照单调升 0.745）——本轮未恶化，顺带观测不算 E2 功劳。
- **机制推断（非结论）**：59 局里锚点仅 ~6 局，联合梯度压不住 90% 自对弈漂移；锚点
  强度≈随机，20k 样本学不会针对它。**10% 弱锚点不足以打破 cycling。**
- **下一步候选（需拍板）**：E2′（锚点占比 0.1→0.3 或多锚点）/ B′（GRU 耐久，本轮可降级）/
  C′D′ 依旧无意义（critic 未拟合+cycling 未破）；或转更根本的对手模型/奖励结构（高侵入）。
- 时长 3419s vs 对照 2709s 差异含 **eval-workers 12 vs 16**，不归因 E2；速度分解见
  `docs/train_speed_benchmark_2026-09-12.md`。

### critic 可预测性探针：信息在表征里、critic 没吸收（2026-09-12，实验文档见
`docs/critic_probe_experiment_2026-09-12.md`）

`diag_critic_ev.py --probe`（post-LN enc → GAE return，留出 20%，线性 OLS + 小 MLP）
两个 ckpt 各 50 局，跨 run 一致：**线性探针 R2≈0.10~0.11、MLP≈0.24~0.31、critic EV≤0
（−0.22/−0.001）** ⇒ ①表征里有非线性可预测信息（监督式可达 24~31%），critic 完全没
吸收；②~70%+ 回报方差在 enc 里不可预测（标签噪声/自对弈对称性，训练侧数据责任）。
**首次定量分离：程序侧/优化侧"没吸收" + 训练侧数据"不可预测"并存。**
Caveat：3 局初值 MLP 0.83 是过拟合假象；enc 是压缩特征，上界是下界性；
探针是监督回归、critic 是 TD 联合训练，不可直接等比。
下一步候选：**A 监督微调 value_head**（判别优化 vs 结构，最快）/ B 100k 长跑 / C 数据侧改善。

### 实验 A + B′：监督微调判别最终结论——结构够用、主因在训练过程（2026-09-12）

**A**（`docs/ft_e2.log`，50 局/20012 帧）：全路径监督 4 epoch，test EV −0.139→+0.054
（epoch 3 峰值 +0.128，epoch 4 过拟合回落）。**B′**（`docs/ft_bp.log`，同 ckpt 另一次
rollout 20803 帧，`--finetune --bypass` 同跑同 split）run 内对照：

| 通路（trunk 可训度） | 监督后 test EV |
|---|---|
| lin-joint（**无 GRU**，trunk 训） | **+0.2936** |
| 全路径（带 GRU）——A 复测 | **+0.2174**（1 epoch 提前停） |
| lin-frozen / mlp-frozen（enc 冻结） | ~0.00（失速：未标准化 enc 在 lr3e-4 下连均值都拟合不了） |

**结论（修订 A 初判）**：①**"critic 结构吸收不了"被否定**——同一带 GRU 网络监督配方下
test EV 可达 +0.22（A 的 +0.054 是坏轨迹+过拟合回落，非稳定上限）；②**GRU 非主导瓶颈**，
净损耗仅 ~0.08（+0.29 vs +0.22）；③主导因素 = **trunk 可训性**：enc 冻结时新鲜头学不动，
trunk 一可训即 0.22~0.29（超探针上界 0.242）；④**主因回到训练过程侧**：on-policy PPO
联合训练 EV≤0 vs 监督同网络 0.22，差距在训练过程（TD 自举/GAE 标签噪声/trunk 梯度冲突）。
⚠️ **教训：同 ckpt 同 seed 两次 rollout 帧数不同（20012 vs 20803，GPU 非确定性）⇒ 跨 run
数字不可直接比，判读只用 run 内对照**。次级发现：价值头直连 enc（bypass）是低成本可试
架构改进；数据侧（early-stop 裁定噪声）仍独立。工具：`diag_critic_ev.py --bypass`（三变体）。

### B′/C′ 落地：value 直连 enc（`value_bypass`）+ 早停裁定降噪（`stall_draw_margin`）（2026-09-12）

用户拍板「先 B′ 落地，随后 C′」。**已落地 + 官方全量 selftest 94 项 100% PASS + 冒烟通过**；
20k 验证跑 `byp_cprime_20k` 进行中（日志 `docs/train_byp_cprime_20k.log`）。

- **B′（架构）**：`FollowerPolicy(..., value_bypass=)` —— True 时 `value = value_head(enc)`
  （**跳过 GRU**），策略头 slot/cell 仍吃 GRU 隐状态；5 处 value 计算点全改
  （act / act_parallel / evaluate / evaluate_batch / value）。ckpt 元数据带该标志，
  不一致时**告警**（旧 ckpt 语义错位 ⇒ 须 `--fresh`）。`TrainConfig.value_bypass=False`
  （dataclass 默认=旧行为，兼容 train_follower/BC 等入口）；**economy 预设 True**；
  CLI `--no-value-bypass`（消融）。标志传播：train_solo（main/opp/baseline*/eval worker）、
  run_league::_build_league、flow_league、league 快照。
- **C′（数据侧）**：`settle_stall`（run_league.py，纯函数 `settle_stall_from_counts` 可单测）——
  早停局皇冠不同或塔血%差 ≥ `stall_draw_margin`(=0.05) → 决定性 ±胜负；皇冠相同且差 < margin
  → **记平局=失败**（去掉掷硬币级胜负标签，保留反躺平信号）。**只改训练侧**（eval 仍用
  `timeout_winner` 真实 CR 规则 ⇒ 评估口径与历史可比）；CLI `--stall-draw-margin`（0=旧行为）。
  history 新增 `stall_games`/`stall_close_draws` **量化 C′ 实际生效比例**（冒烟 7 局：2 早停/0 降级）。
- **判读口径**见 v3 计划 §3.10.3：主判据 EV 池化过零/上升 + `value_std_ratio`（对照
  `fix_gru_ln_norm_20k`：EV −0.0051、ratio 0/8）。⚠️ 本轮同时带 B′+C′ 两改动；若
  `stall_close_draws` 占比高，须另跑 `--stall-draw-margin 0` 消融分离贡献。

**⚠️ 落地副作用（2026-09-12，同类第二次）**：B′ 改 value 通路后，两个诊断探头仍**硬编码**
`value_head(hidden)` —— `rl/diagnostics.py::gru_vitality` 的 `value_std`（→ `value_std_ratio`）
与 `scripts/diag_value_head.py`；对 bypass 模型测的不是被训练的量。已修（bypass → `value_head(enc)`）
+ 回归护栏（`test_value_bypass` 断言 `gru_vitality` 的 value_std == std(value_head(enc))）。
**教训：架构变更后必须全仓搜"硬编码的前向通路"**（上次是"验收表分母没人测"，这次是"探头测旧通路"）。
`byp_cprime_20k` 进程加载的是旧模块 ⇒ 该 run 的 `vstd/rstd` 列无效（**EV 列有效**，来自真实
rollout）；评估点存了 `solo_main_<step>.pt`，跑完用修复版脚本离线重算真值。

### 20k 判读：B′ 未改善 critic、表征可预测性反升到 0.42、C′ 训练侧几乎不触发（2026-09-12）

run `runs/byp_cprime_20k`（65 局、8 点、训练循环 1993.4s、0 降级/0 Traceback；
economy 预设 = bypass + `stall_draw_margin=0.05`）。日志 `docs/train_byp_cprime_20k.log`。

- **① B′ 无效**：EV **0/8 过零**（最好 −0.0044，参照 `fix_gru_ln_norm_20k` 最好 −0.0051）
  ——同级。诊断（50 局）：`v std=0.025` vs `R std=13.71`（0.18%）、`EV_global=−0.0060`、
  RMSE≈std(R)；离线真实 `value_std=0.0221` → ratio 0.0017~0.0036（门槛 0.3，比参照更低）
  ⇒ **仍是常数预测器**。⇒ **value 通路的接线（GRU vs enc）不是瓶颈**。
- **② 表征可预测性反升（最重要）**：同批帧 **MLP 探针 R² = +0.4246**（线性 +0.1355）
  ——高于此前 ckpt（+0.242/+0.306），**迄今最高**。"信息在 enc（42%）vs critic 吸收≈0"
  落差最大；结合监督实验（同网络 value-only 可达 0.22~0.29）⇒ **瓶颈在 on-policy PPO
  联合训练的价值吸收/优化动力学**（共享 trunk 梯度冲突 / TD 目标噪声），**非表征、非接线**。
- **③ C′ 训练侧几乎不触发（修正假设）**：65 局早停仅 2 局（3%），被降级 1 局（1.5%）。
  审计的"28~40%"是 **eval 侧随机对手**局（不进训练标签）⇒ C′ 对训练标签影响≈无；
  本轮可近似按 **B′ 单变量**判读。
- **④ 行为**：未复现 `fix_gru_ln_norm_20k` 的 `vs baseline0` 崩塌（全程 0.33~0.50）与
  baseline_rand 崩塌（≥0.31）；但 `vs baseline_prev` 仍振荡（0.89→0.40）⇒ **cycling 仍在**。
- **下一步候选**：**E′（首选）给价值通路独立 encoder**（v2 §4 项，现有三条证据支撑；
  可证伪预测：EV 应到 0.2+）；F′ aux value loss；G′ 回到对手/数据侧（cycling）。
  **`value_bypass` 默认值**：证据不支持收益 ⇒ 可选回退预设 False（保留 flag 做对照基线）。

### E′ 落地：独立价值编码器 + 非线性价值头（2026-09-12，用户拍板）

**依据**：① MLP 探针 `enc→return` **R²=0.4246** vs 线性仅 **0.1355** ⇒ 价值信息主要是
非线性的，`value_head=nn.Linear(hidden,1)` **先天上限 ~0.13**（bypass 纯线性吃 enc 必然≈0）；
② 监督同网络可达 EV 0.22~0.29 而 on-policy 联合训练 ≈0；③ bypass（只换接线）无效
⇒ 需给价值通路**自己的参数与容量**。

**改动**：`FollowerPolicy(..., value_independent=)` → `value_enc_fc` + `value_enc_ln` +
`value_head_mlp`（hidden→max(32,hidden/2)→1）；`_encode_parts`/`_encode_batch_parts` 暴露
`fused`；**统一入口 `_value_from(enc,h,fused)`（independent > bypass > shared）**，
act/act_parallel/evaluate/evaluate_batch/value/diagnostics/诊断脚本全走它（防"探头硬编码旧通路"
复发）；`TrainConfig.value_independent`（默认 False）+ **economy 预设 True**；CLI
`--no-value-independent`；ckpt 元数据 + 一致性检查覆盖两个标志；传播到 train_solo
（含 **rand_anchor**）/run_league/flow_league/league 快照。

**⚠️ 踩坑（当场修复）**：`_make_rand_anchor` 漏传架构标志 → 作为 control 的 `opp_model`
进 `eval_solo_parallel`，worker 按 cfg 建网后 `load_state_dict(opp_sd)` **键集不匹配** →
**每周期 worker 启动失败 + 静默降级串行**（冒烟日志 `Missing key(s): value_enc_fc/value_head_mlp`）。
**教训：架构标志必须传播到所有经 eval worker 做 state_dict 往返的构造点；判读前先 grep
日志的 Traceback/降级行**（静默降级会把 bug 伪装成"只是慢"）。

**验证**：官方全量 selftest PASS（含新 `test_value_independent_encoder`：通路一致性、
与共享通路不同、**策略 logprob 不受影响**、元数据/告警/诊断口径）；冒烟 1500~2000 步
0 Traceback/0 降级。**20k 跑 `eind_20k`**（日志 `docs/train_eind_20k.log`）。
**可证伪预测：EV 应到 0.2+**（对照 `byp_cprime_20k` 最好 −0.0044）；若不升 ⇒ 转 F′
（aux value loss / 更高 vf_coef / 更长价值训练 = 瓶颈在梯度与目标，不在容量）。

### E′ 20k 判读：预测被证伪 ⇒ 根因是"优化预算 + 批次构成"，不是架构（2026-09-12）

run `runs/eind_20k`（65 局、8 点、训练循环 3717.1s、0 降级/0 Traceback；economy = independent）。
**EV 0/8 过零**（最好 −0.004，与 bypass −0.0044、grid_ln −0.0051 同级）；离线真值
`value_std=0.0067`（独立 MLP 头）→ ratio ≈0.0005 ⇒ **仍是常数预测器**。对照曲线
`baseline0` 0.46→0.975(@15k)、`baseline_rand` 0.625→0.925(@10k)（绝对强度有爬升），但
`baseline_prev` 0.10↔1.0 ⇒ **cycling 仍在**。

**⚠️ 本轮决定性发现（代码级）——瓶颈不在架构**：
- `PPOTrainer.update()`（`ppo.py:199-259`）= **1 forward + 1 backward + 1 opt.step**，
  **无 n_epochs / 无 minibatch / 无 shuffle**；
- `train_solo` 每步取 `transitions[:batch_size]` = **同一局连续 128 帧**（`train_solo.py:1385`）；
- ⇒ **20k 步 = 156 次梯度步**，每步目标≈"局段均值"（批内 Var(R)=全局 0.32×、
  `corr(R_t,R_{t+1})=0.99`）⇒ 价值头（共享/线性、bypass、独立 MLP 头都试过）**不可能拟合**。
- 三条独立证据互证：监督微调 **65k 逐帧随机步 → EV 0.22~0.29**（≈400×）；MLP 探针
  `enc→return` **0.24→0.31→0.42** 而 critic 吸收 ≈0；**B′(接线) + E′(参数独立+非线性头)
  两轮架构干预全部无效**。

**F′ 设计（下一步，需拍板）**：把更新做成真正的 PPO —— `n_epochs`(4~8) × **shuffle** ×
**minibatch**(32~64) 多轮随机小批更新 ⇒ 价值头梯度步 ×4~8 且打破"连续帧同质批"。
**可证伪预测：EV 20k 内 >0 并上升**。⚠️ `n_epochs>1` 后 `ratio` 离开 1.000、
clip 生效 —— AGENTS §②"ratio≡1.000 是结构性"**只对 n_epochs=1 成立**。

**⚠️ 三次"架构标志未传播"事故**：新增构造参数必须传播到**所有做 state_dict 往返的构造点**——
本轮连踩 `_make_rand_anchor`（→ eval worker 键集不匹配 → **静默降级串行**）、
`_OpponentPool._ensure_hist`（hist ckpt 可能是新架构）、四个 `diag_*.py` 镜像对手。
**纪律：新增架构参数时 `grep load_state_dict` 全仓核对。**

**E′ 诊断补齐（50 局，`docs/diag_ev_eind_20k.log`）**：`v std=0.007` vs `R std=7.30`（0.1%）、
`EV_global=−0.0356`；探针 线性 R²=+0.041 / **MLP R²=+0.236**。**四 ckpt 总表**（同口径）：
e2（共享GRU+线性）MLP 0.242 / norm(grid_ln) 0.306 / byp(bypass) **0.425** / eind(独立+MLP) 0.236，
而 EV 全部 ≤0（−0.219/−0.001/−0.006/−0.036）⇒ **表征含 24~42% 可预测信息、四种价值架构
吸收都 ≈0** —— 与"156 次梯度步"的机理一致：不是架构问题，是优化预算/批次问题。

> ⚠️ **上面这段推理链已于 2026-09-12 被推翻**（探针是时间泄漏，见下一节）。
> 保留原文以便追溯；**不要**再引用"表征含 24~42% 可预测信息"这个结论。

---

## ⚠️ F′ 落地 + 判读口径三修 + 探针时间泄漏（2026-09-12，计划 v3 §3.12）

分支 `no-human-watch-A`（已推送）。这是本项目**第四次**"指标本身在骗人"。

### ① F′：真正的 PPO 更新预算（代码已落地，保留）

`PPOTrainer(n_epochs / minibatch_size / shuffle)`；默认值 = 旧行为（1 轮/整批/不打乱，
走原单一 pass 分支**逐位等价**——`run_league`/`flow_league`/`train_follower`/`train_prophet`
共用本类，默认不得变）。新分支 = 多轮 × 打乱 × 小批，每小批一次 `opt.step`，
`mean` 口径，`ratio/clip/grad_norm` 取**末轮**。CLI：`--ppo-epochs/--ppo-minibatch/--ppo-shuffle`。
**PPO 预算写进 run_state，续训忘传参数会告警**（`config.json` 不参与 resume 解析，静默退回旧行为）。
`20k` 跑（`runs/fprime_20k`，67 局、循环 2557.8s）：每 128 帧 **16 次梯度步**（原 1 次）。

### ② 口径三修（都不是训练数学问题，但都会把结论带偏）

1. **`ratio/clip` 聚合误除全部轮次** ⇒ 4 轮读出 ratio≈0.26（实际每小批 0.96~1.02）。
   修：用本轮权重 `w_ep`；哨兵 `lr=0.0 → ratio 必须 <1e-5 偏 1`。
2. **`value_std_ratio` 分子分母跨窗口**（分子=`最近 96 帧`、分母=`整个评估窗口`）⇒ 不可判读。
   修：分子分母都用 `last_ev_pairs`。**恒等式 `EV ≤ 2σ_v/σ_R`** 是判断这两个数是否自洽的利器。
3. **EV 用了"更新后"预测 = in-sample**（最严重）：F′ 末轮预测是在那 128 帧上训过 12 步的值。
   实证：训练日志末点 `EV=+0.41`，**同权重** 50 局独立 rollout `EV_global=−0.0044`；
   受控实验同批：更新前 +0.011 / 更新后同批 +0.042 / 更新后换独立 rollout **−0.384**。
   修：更新前多做一次整批 no_grad 前向作为对外 EV（与历史可比），末轮值存
   `explained_variance_insample`，步日志加 `EVin=`。

**纪律**：写一个**比值**门槛前，先确认分子分母是**同一个样本集**；报一个**跨版本可比**的
指标前，先确认它不依赖本次改动的那部分参数状态。

### ③ 决定性发现：探针的"可预测性"是**时间泄漏**

`lin_probe`/`mlp_probe` 加 `groups=`（按局分组留出）。`fprime_20k`（50 局/16721 帧，纯镜像）：

| 留出口径 | 线性 R² | MLP R² |
|---|---|---|
| 逐帧随机（旧口径） | +0.10 / +0.14 | **+0.27 / +0.38** |
| **按局分组（新）** | −0.35 / −0.21 | **−0.36 / −0.45** |

相邻帧 `corr(R_t,R_{t+1})≈0.99` ⇒ 逐帧随机留出把测试帧的邻居放进训练集。
**⇒ 跨 ckpt 的 0.24~0.42 全是泄漏，跨局不泛化**（分组口径比"预测测试局均值"还差）。

**换标签也救不了**（同批帧、同分组口径）：局结果广播（终局帧 R）分组 MLP R² **−0.49**、
局均回报广播 **−0.71**、逐帧 GAE 回报 **−0.45**，而 critic `EV_global=−0.005`。
⇒ **没有任何一种回报标签能从 `enc` 跨局预测**。

**另一直证**：未训练的 value 头 σ_v/σ_R 本来就是 **0.08~0.18**
（`tmp_fresh_value_scale.py`），训练 20k 后是 **0.0074** ⇒ **训练是在正确地收缩到条件均值**，
`EV≈0` 是该表征/数据下的**最优解**，不是欠训练/架构/更新预算问题。

### ④ 结论与下一步

- **F′ 成立且保留**（ratio/clip 恢复诊断价值、critic 动态范围 10×：σ_v/σ_R 0.0010→0.0068、
  离线 EV −0.036→−0.005/−0.026），但**离线 EV 仍未过零**；训练曲线那个 +0.41 是 in-sample。
- **critic 侧不要再加投入**：B′/E′/F′ 三代都在攻一个测量假象指出的靶子。
- **下一步（G′，未实施）三层对照**（同批帧、同分组口径）分离"世界不可预测"vs"表征不行"：
  ① **原始状态标量**（塔血/圣水/时间/皇冠）→ 局结果；② `enc` → 局结果（已知不能 −0.49）；
  ③ 打乱局标签的随机基线（给分组 R² 噪声地板）。① 能/② 不能 ⇒ 表征是瓶颈（改观测编码）；
  ① 也不能 ⇒ 镜像自对弈局结果≈掷硬币，critic 工作应停、算力转回策略侧（cycling/组织进攻）。


---

## F′ 修正口径复跑（2026-09-13，分支 `no-human-watch-C`）：critic 侧判定"无进展" → 转策略侧

完整判读 `docs/fprime_rerun_20k_verdict_2026-09-13.md`；run `src/clasher_new/runs/fprime_ev_20k/`。

### ① 为什么复跑

`fprime_20k` 的训练期 EV（均值 +0.198）是 commit `2648baa` **之前**的代码产出的
（`explained_variance` = 末轮更新后的 in-sample 读数），而 `gfix_20k` 是**更新前池化**。
跨口径比较无效 ⇒ 复跑一次把 F′ 放到同一口径，与 G'-fix 做单变量对照。
分支 C = 分支 A（无 `tower_state`）+ 一个纯度量修复（`stats["value_std_ratio"] = None`
从 `if/else` 之外移回 `else`）⇒ **训练语义与 `fprime_20k` 等价**。

### ② 结果：EV 曲线还是在 0 附近振荡，无趋势

| 评估点 | F′ 首跑（旧口径 in-sample） | **F′ 复跑（修正口径）** | G'-fix | legacy 基线 |
|---|---|---|---|---|
| 2500 / 5000 / 7500 | +0.101 / −0.051 / −0.048 | **−0.041 / +0.103 / +0.336** | +0.282 / +0.157 / −0.104 | −0.097 / −0.126 / −0.315 |
| 10000 / 12500 / 15000 | −0.122 / +0.558 / +0.462 | **−0.005 / −0.115 / −0.156** | +0.039 / +0.005 / −0.012 | −0.015 / −0.528 / −0.089 |
| 17500 / 20000 | +0.271 / +0.413 | **−0.011 / +0.139** | +0.660 / −0.106 | −0.013 / +0.000 |
| **均值** | +0.198 | **+0.031** | +0.115 | −0.148 |

⇒ 首跑那个 +0.198 **是 in-sample 假象**（降 6.4×）；与 G'-fix 的差在自噪声内
⇒ **塔血通道对 critic 既不好也不坏，不是瓶颈**（与 G′ 预注册证伪一致）。
**三代改动没有一次让 EV 结构性过零。**

### ③ 新数据：同窗口 `vstd/rstd`（此前被 bug 清空，从未有过曲线）

F′ 复跑 **0.027~0.463（均值 0.178）** vs legacy 基线 0.021（**8.5×**），
但 EV 没跟着起来 ⇒ **F′ 把 critic"弄活了"，活起来的方差却不可预测**
（= 在同一批 128 帧上多训 12 步换来的拟合方差）。这是"EV≈0 是 MSE 最优收缩"的更强证据。

### ④ 离线权威口径（50 局）：EV 为负有**两个**来源，别再只看第一个

| 指标 | F′ 首跑 | F′ 复跑 | G'-fix |
|---|---|---|---|
| `EV_global` | −0.0025 | −0.1589 | −2.2146 |
| `corr(v,R)` 全体 / 局内去均值 | +0.07 / +0.02 | +0.54 / +0.64 | +0.75 / +0.90 |
| `corr(局均 v, 局均 R)` 跨局 | +0.14 | +0.17 | −0.84 |
| **回归斜率 `R~v`（1.0=量纲对齐）** | 10.5 | 28.5 | 525 |
| `EV_between` | −0.0001 | −0.43 | −22.3 |

1. **跨局不可预测**（跨局 corr 全在 ±0.2，n=50 时 1σ≈0.14）；
2. **量纲失配**（斜率 10~525）—— critic 输出标度只有回报的 1/10~1/500，
   于是 `corr=+0.5~0.75` 也照样 `MSE > Var(R)`，而且那个相关几乎全是**局内时间趋势**
   （去均值后仍在），不是跨局信息。
⇒ 离线 EV 更负 ≠ critic 更差（是 bias/标度更烂）；但**三者都没过零**，判据要的是过零。

### ⑤ 探针复现（分组留出，功效对照 `→time` ≈ +1.00 仍过关）

无塔血通道的架构上 `enc → 塔血差` 分组 MLP R² = **−0.17**（G′ 测到 −0.26）
⇒ 反证 G'-fix 的 **+0.69 是"通道真接上了"**，不是探针噪声。
`enc → 局结果` = −0.65、`raw → 局结果` = −0.28 ⇒ **连特权真值都测不出局结果**，
再次确认不是表征问题。

### ⑥ "末点崩塌"是 cycling 相位，不是 G'-fix 特有

固定随机锚点（门槛 0.35）：F′ 复跑 @12500-17500 = **0.000/0.000/0.000**，
G'-fix 崩在 @20000（0.025），F′ 首跑崩在 5000-10000 —— 同形态、**不同相位**。
锚点均值：G'-fix 0.463（6/9 点过线）≈ F′ 首跑 0.385（6/9）> F′ 复跑 0.178（2/9），
但同代码同 seed 的两次重复就差 0.21（见 ⑦）⇒ **单跑分辨率不足，不能据此排序**。

### ⑦ ⚠️ 方法学（最重要）：**同 seed 同代码的两次 run 也不可复现**

`fprime_20k` 与 `fprime_ev_20k` 训练语义等价、seed/命令行相同，却从 **@2500 起分叉**。
做了两组对照（各 2 次 800 步同参 run）：原始环境下前 **435 步逐位一致**，
之后出现 **1e-4 级漂移**，`eval@800` 已不同；
加 `PYTHONHASHSEED=0 + OMP/MKL/OPENBLAS_NUM_THREADS=1` **仍然不同**
⇒ 排除 hash 随机化与 BLAS 线程数，**机制未定**（嫌疑：CUDA 端非确定性算子，
如 GRUCell/embedding 反向的原子加；`tmp_determinism.py` 测的简化 conv+GRU 反向是逐位可复现的）。
**推论：本项目 20k 单跑 A/B（n=1/臂）分辨率不足，此前"单跑看曲线"的小效应结论
一律降级；要判读必须多 seed 配对重复（或把评估局数拉大）。**
本轮判定不受影响：三次 `EV_global` 全 < 0、曲线全部绕 0 振荡 —— 不是小效应，是**没有效应**。

### ⑧ 判定与下一步

**critic 侧到此为止**（证据链：表征归一化 / 独立价值编码 / 真 PPO 预算 / 塔血通道
＋ 三层可预测性对照 ＋ 功效对照 ＋ 换标签 ＋ 打乱标签地板 → "逐帧 GAE 回报跨局不可预测，
critic 收缩到条件均值是 MSE 最优解"）。**转回策略侧**，优先级：

1. **cycling / 绝对强度**（所有 run 的锚点胜率长期 < 0.35，甚至整段 0.000）——
   固定随机锚点（E2 门禁）是**唯一**能测绝对强度的仪器；
2. **行为病理未解决**：`elixir_avg` 起点 1.55 → 终点 1.9~3.0，末点 `unilateral_rate` 100%
   （攒费 / 组织进攻与 2026-09-09 诊断时一样）；
3. 可复用：推理时浅 MCTS（`rl/mcts.py`，已验证便宜）、`belief_planner` 拦截几何链。

---

## D1 判定（2026-09-13，分支 `no-human-watch-D`）：对手分布去镜像化**有效于"防崩"**，未证明提高上限

完整判读 `docs/d1_league_20k_verdict_2026-09-13.md`；预注册 `docs/cycling_league_plan_2026-09-13.md`。
三跑 run：`runs/d1_league_20k{, _r2, _r3}/`，日志 `docs/train_d1_league_20k{,_r2,_r3}.log`。

### ① 改了什么（一个机制：训练时"打谁"）

- `opp_mix`：frozen **0.4→0.1**、hist **0.3→0.6**（`config.DEFAULT_OPP_MIX` +
  `train_solo._OPP_MIX` 两处同步）；`rand_anchor` 剂量**不变 0.1**（E2 已测过，留给 D2）。
- **`_OpponentPool.refresh_hist()`（新）**：每 `solo_copy_every` 步重扫本 run 目录。
  **旧实现的坑（重要）**：`hist_paths` 只在 `__init__` 扫一次，而 `--fresh` 时本目录为空
  ⇒ **整个 run 的 hist 槽全是 `--hist-seed-dir` 的外部旧架构 ckpt，本 run 自己的快照
  从头到尾进不了池**（"历史联赛"名不副实）。修复后本 run 快照 1→9 进池。
- hist 的 PFSP id 由 `hist_<下标>` 改成**稳定键**（父目录名+文件名）——池增长时下标会错位，
  PFSP 胜率会被张冠李戴。
- `rl/pfsp.py` 新增可选 `alpha/gate_hi/gate_penalty`（**默认 = 旧行为逐位等价**，
  兼容红线）；训练侧 `alpha 0.05→0.20`（20k run 每 ckpt 仅 ~1.5 局，旧 EMA 几乎不动
  ⇒ PFSP 名义存在、实际近似均匀）+ 易胜对手（EMA>0.85）权重 ×0.2。
- 对手局累计直方图落日志（判读指纹）；selftest 新增 `test_pfsp_gate_and_dynamic_hist`。

### ② 结果（三跑 20k，同协议，只改 config-name）

| 统计量 | D1 r1/r2/r3 | 无干预对照（F′ 首跑 / F′ 复跑 / G'-fix） |
|---|---|---|
| **单跑最差锚点**（主判据 C1） | **0.125 / 0.200 / 0.250**（均值 0.192） | 0.000 / 0.000 / 0.025（均值 0.008） |
| 末 4 点均值（用户指定） | 0.600 / 0.312 / 0.447 | 0.588 / 0.019 / 0.356 |
| 全 8 点均值 | 0.492 / 0.423 / 0.577（min 0.423） | 0.355 / 0.122 / 0.416（max 0.416） |

- **C1 两组区间完全不相交**（[0.125,0.250] vs [0.000,0.025]，Welch t≈4.9，n=3/3）
  ⇒ D1 **消除了"整段输给固定随机策略"的相位**（A′ 取证里最病态的那条）。
- **"末 4 点滚动均值"两组重叠**（0.312~0.600 vs 0.019~0.588）⇒ 这条判据被相位主导，
  n=3 下**判不了**；D1 **没有**证明提高绝对强度上限。
- 机制指纹三跑全成立：镜像自对弈 40% → **6.6% / 13% / ~13%**，本 run 快照 1→9 进池，
  0 降级 / 0 Traceback。
- `gates.json` 的 engagement FAIL **不是 D1 引入**：三跑的末点 deploy 11.9/42.5/54.5
  横跨无干预组整个范围（12.5~61.9），是**所有 run 共有的末点相位**。

### ③ 判定与下一步

**有效（防崩层面）**：D1 保留并建议进主线；**上限未动** ⇒ 下一步要么 100k 长跑看
上限是否随步数上移，要么 D2 = `rand_anchor` 0.1→0.3（剂量-反应，单变量）。

### ④ ⚠️ 两条标定教训（本次连错两次，同源，必记）

1. r1 判读时发现预注册里 **F′ 首跑末 4 均值填成 0.400，实际 0.588** ⇒ P1/P2 失去区分力
   （F′ 首跑也会通过）；
2. r2 判读时发现 C2 基线"≥0.35 的 run 数"填成 **1/3，实际 2/3**（漏了 G'-fix 的 0.356）。
   两次都是**把基线算弱**，让判据看起来更有区分力。

⇒ **纪律：阈值判据的基线列必须用脚本算出来贴进预注册，不许手抄。**
判据的力量必须来自**设计**（区间不重叠 / 多跑聚合 / 机制指纹），不能来自对单跑或
手算数字的信任（与 `fprime_rerun_20k_verdict` §8"同 seed 同代码两跑从第一个评估点就分叉、
锚点末 4 均值差 0.57"是同一件事的两面）。
