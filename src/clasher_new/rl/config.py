"""命名训练配置：一组参数（超参 + 奖励权重）→ 命名 → 独立输出文件夹。

用途（对应需求）：
- 「不同奖惩机制分类模型」：每个命名配置自带一套 ``reward`` 权重，即一种奖惩机制；
- 「一组参数设置 + 命名 + 分文件夹保存」：``--config <名>`` 选中预设（或用
  ``--load-config <json>`` 载入自定义），训练产物全部落在 ``out_dir/<name>/`` 下：
  断点状态、main 检查点、联赛录像、config.json 存档、Elo 状态、日志。
- ``--save-config <path>`` 把当前解析结果导出为 JSON，可编辑后 ``--load-config`` 复用。

预设：
- ``standard``   默认机制：塔血统一（打击=损失同价 0.001/0.001）+ 费差默认打开
                 （normalize_tower_dmg=True，elixir_diff_weight=0.5，1 圣水≈500 塔血 @lv11）；
- ``aggressive`` 推进：破塔/皇冠/胜利奖励更高，费差加码（1 圣水≈700 血，塔血换费差）；
- ``defensive``  防守反击：非法动作惩罚更重，费差减码（1 圣水≈300 血）；
- ``lockdown``   自闭：费差压到≈0（1 圣水≈50 血，鼓励费差换塔血，浪费仍小惩罚）；
- ``elixir``     鼓励圣水效率：每步按我方剩余圣水给正向 shaping（叠加默认费差机制）；
- ``economy``    费差经济（默认机制别名）：塔损按塔血%归一化 + 显式 Δ费差 shaping；
- ``fast``       小步快跑：步数/评估频率/单局上限都调小，用于冒烟/设备验证。

按流派区分奖惩（flow 联赛 6 模型）：``MODEL_REWARD_OVERRIDES`` 在所选预设之上按
模型 id 覆盖，main/all_decks/random_deck 用基线，推进加码费差、防反减码、自闭压到≈0。

任何超参都可用命令行覆盖（如 ``--config aggressive --lr 1e-3``）。
"""

import os
import json
from dataclasses import dataclass, field, asdict

#: 默认奖励权重（与 RLEnv 的 _DEFAULT_REWARD 保持一致；勿单独改一处）。
#: reward v2（2026 重构，替代旧"手牌圣水差逐帧费差"）：
#: - 资源账：费差项 = Δ(手牌圣水 + 场上部署份额)差（部署帧 E−c/V+c 抵消 → 下牌不罚；
#:   死亡注销、法术击杀返还 → 送死/空砸有代价、解牌赚费差）；
#: - 价格两段离散（120s 切双倍圣水，不线性）：tower_dmg_late=0.002 塔血贵 /
#:   elixir_diff_late=0.1 费贱 → 双倍期亏费换塔血、法术砸塔自动变正 EV；
#: - unit_dmg_k：单位受伤 shaping（客观伤害事件，非估值：敌方单位每掉 1 血 → 我方 +k）。
DEFAULT_REWARD = {
    "crown_weight": 8.0,        # 皇冠差系数（破敌塔每座 +8：破塔里程碑，胜利太稀疏需中间大奖励）
    "crown_lose_weight": 10.0,  # 被破塔惩罚（> crown_weight：丢塔比破塔更痛，教防守价值）
    "tower_dmg_opp": 0.001,     # 敌方塔损 → 正奖励（前段 t<120）
    "tower_dmg_self": 0.0012,   # 我方塔损 → 负奖励（不对称：塔伤奖励 0.001 末位加"2"）
    "tower_dmg_late": 0.002,    # v2 双倍期塔血系数（t≥120：斩杀/法术砸塔自动变正 EV）
    "tower_dmg_self_late": 0.0022,  # 双倍期我方塔损（同上不对称：0.002 末位加"2"）
    "win_bonus": 10.0,          # 获胜加成
    "lose_penalty": 10.0,       # 失败惩罚
    "draw_penalty": 10.0,       # 平局惩罚（= 失败：平局归类为败，逼策略主动求胜；0=旧行为免费平局）
    "invalid_penalty": 0.05,    # 每次非法动作惩罚
    "elixir_bonus": 0.0,        # 每步按我方剩余圣水的正向 shaping（圣水效率机制）
    "normalize_tower_dmg": True,   # 塔损按塔血%归一化到 lv11 锚（默认打开，跨等级一致）
    "elixir_diff_weight": 0.5,     # v2 资源账 edw 前段（t<120：费贵，教珍惜圣水；
                                   # lv11 下 1 圣水 ≈ 0.5/0.001 = 500 塔血的前段锚）
    "elixir_diff_late": 0.1,       # v2 双倍期 edw（t≥120：费贱，亏费换塔血可接受）
    "unit_dmg_k": 0.0005,          # v2 单位受伤 shaping（敌方单位每掉 1 血 → 我方 +k）
    "tower_premium_k": 2.0,        # 塔血差异化定价：凹形溢价强度（残血塔单位血价值
                                   # ×(1+k×(1−ratio)²)），双向对称；经 reward_to_env 透传
    "king_gate": 0.05,             # 两公主塔都存活时王塔单位血价值系数（≈0，王塔贬值）
}

#: 按流派模型的奖惩覆盖（在所选预设之上按模型 id 覆盖；flow 联赛 6 模型用）。
#: main / all_decks / random_deck = 基线（同一参数）；推进加码费差（塔血换费差）、
#: 防守反击减码、自闭压到≈0（鼓励费差换塔血，浪费仍小惩罚——见 config 模块注释）。
MODEL_REWARD_OVERRIDES = {
    "main": {},                 # 基线
    "all_decks": {},            # 基线（与 main/random_deck 同一参数）
    "random_deck": {},          # 基线
    "push_flow": {"elixir_diff_weight": 0.7},     # 推进：增加"塔血换费差"奖励（1圣水≈700血）
    "counter_flow": {"elixir_diff_weight": 0.3},  # 防守反击：降低该奖励（1圣水≈300血）
    "lockdown_flow": {"elixir_diff_weight": 0.05},# 自闭：压到≈0，鼓励"费差换塔血"（1圣水≈50血）
}

#: solo 训练对手池配比（frozen / hist / defend / rand_anchor）。
#: 2026-09-11 判读整改（P1-1b）：frozen 0.7 → 0.5，hist 0.2 → 0.3，defend 0.1 → 0.2。
#: 依据：20k 验证跑测出 **critic 的 explained_variance ≈ 0**（价值网络对回报的解释力
#: 等于"恒定预测均值"）。怀疑之一 = **镜像自对弈的对称性**使"状态→胜负"近乎不可预测
#: （对手 70% 是自己的冻结副本，任何战术都无差别地被镜像抵消）。提高非镜像对手占比
#: 是给 critic 制造可学习信号的最直接手段；frozen 仍占多数，保持自对弈稳定性。
#: 2026-09-12（E2，v3 §3.9）：再加第四槽 **rand_anchor**（固定随机锚点，frozen 0.5 → 0.4）。
#: 依据：A′ 取证坐实自对弈 RPS 循环（main 打冻结副本 0.85 / 打起点随机 0.13 / 打全新
#: 随机 0.505；`scripts/_forensics_cycling.py`）——自对弈无外部锚点时策略会在"只赢近亲"
#: 的循环里漂移。把固定随机锚点（权重种子 RAND_ANCHOR_SEED，deterministic 决策）放进
#: **训练分布**：每局 10% 打锚点，"输给固定外部基准"成为可见负样本，打破循环漂移。
#: 锚点永不参与训练/同步/PFSP，与 E1 评估侧锚点同权重。
#: 兼容：旧式三槽 dict（无 rand_anchor 键）仍合法，rand_anchor 概率为 0。
#: 想回旧行为：``opp_mix={"frozen":0.7,"hist":0.2,"defend":0.1}``。
#:
#: 2026-09-13（D1，`docs/cycling_league_plan_2026-09-13.md`）：**frozen 0.4 → 0.1、
#: hist 0.3 → 0.6**，rand_anchor 剂量刻意不变（E2 已测过 0.1 无效，留作 D2 剂量-反应）。
#: 依据：① A′ 取证 + 本轮 F′ 复跑判读证明"末点崩塌"是 **cycling 相位**，而 `frozen`
#: 是唯一与当前策略**同步演化**的对手（每 2000 步同步）= RPS 锁步的载体；
#: ② 代码侦察发现 `hist_paths` 只在 `__init__` 扫一次 ⇒ `--fresh` 启动时全是外部旧 ckpt，
#: **本 run 自己的历史快照从未进池**（"历史联赛"名不副实）——D1 同时给 hist 池加
#: 动态刷新（`_OpponentPool.refresh_hist`）+ PFSP 门禁（`alpha` 0.05→0.20、
#: 易胜对手 ×0.2 降权，见 `rl/pfsp.py`）。
DEFAULT_OPP_MIX = {"frozen": 0.1, "hist": 0.6, "defend": 0.2, "rand_anchor": 0.1}


@dataclass
class TrainConfig:
    name: str = "standard"
    description: str = ""
    # —— 训练/评估超参 ——
    total_steps: int = 20000
    steps_per_eval: int = 4000   # 评估频率：每 N 步训满再评（2000→4000：每 ckpt 训练量翻倍、评估频率减半）
    anchor_every: int = 0        # C 方案（2026-09-13）：轻量评估点间隔——只跑固定随机锚点
                                 # （baseline_rand，C1 判据原料）+ 落快照，不跑 main/对照两块
                                 # （省 3/4 成本）。0=关闭（旧行为，逐位不变）。
                                 # 用途：长 run 上把锚点维持在 2500 分辨率、其余块稀疏到
                                 # steps_per_eval，实测（D1 三跑锚点序列粗采样回放）证明
                                 # C1 的谷底只有 1 个点宽、5000 即开始漏，10000 会把最差点
                                 # 系统性抬高 ~2×（0.192→0.462）。
    batch_size: int = 128
    update_interval: int = 128
    lr: float = 3e-4
    hidden_dim: int = 128
    seed: int = 0
    n_eval_games: int = 40    # 每对评估局数。统计契约：轮内聚合估计 SE≈347.5/√(5N)（main 5 对）
                              # N=16→2σ≈±0.25（检测 100 Elo 要 48 局/对、检测 5% 胜率要 ~800 局）
                              # → 2026-09-11 审计整改：16→40（2σ≈±0.16），配合 steps_per_eval
                              # 4000→8000 抵消墙钟；评估开销与 N 成正比，嫌慢用 --n-eval-games 覆盖。
    max_ep_steps: int = 360       # 常规时间 180s（360 步）的截断上限；180s 皇冠相同 → 自动进入
                                  # 加时窗口（overtime_open 延长到最多 300s=600 步，先破塔者胜；
                                  # 到 300s 仍无人破塔 → **按三塔血量合计裁决**：多者胜，
                                  # 只有合计完全相等才记平局（2026-09-17 口径））
    n_envs: int = 1               # 并行多环境（>1 用批量推理；默认 1 与旧行为一致）
    parallel: str = "mp"          # n_envs>1 时：mp=跨进程 worker（多核真并行）/ proc=单进程批量化
    card_level: int = 11          # 本局全部卡牌等级（11-16；配合 economy 的塔血归一化跨等级一致）
    eval_at_start: bool = True    # 训练开始先跑一次评估/快照，WebUI 立即有真实数据
    gamma: float = 0.997          # 终端现值修复：0.997^360≈0.34（原 0.99^600≈0.0024，胜负/平局惩罚几乎不可学）
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    clip: float = 0.2
    max_grad_norm: float = 0.5
    # —— 纯 RL 冷启动修复（2026）——
    # advantage 归一化：batch=整批中心化（旧默认）/ scale=只除以批 std /
    # none=原始。躺平局批内大量零优势帧被中心化抬成伪正优势（推高 STOP），
    # scale 保留原始符号只缩放幅度。
    # 2026-09-11 审计整改：默认 batch → scale（PPOTrainer 的函数默认仍是 batch，
    # 所以只影响走 TrainConfig 的训练入口；显式 --adv-norm batch 可逐位回旧行为）。
    adv_norm: str = "scale"
    # —— 价值通道量纲（2026-09-11 审计整改，P0-1）——
    # none=旧行为（价值损失不缩放）；running=按回报运行 std 缩放价值损失
    # （v_loss /= s²），修"value_loss 量纲压倒策略项"（实测 gnorm 中位 6.8e3
    # vs 裁剪阈值 0.5、value_loss 中位 ~48）。详见 rl/ppo.py 模块 docstring。
    value_norm: str = "none"
    # 梯度成分诊断采样间隔（每次 update 计数；>0 时每 N 次打印 p_gnorm/v_gnorm，
    # 用于证实/证伪"评论家主导更新"。0=关闭。开销 = 每 N 次多两次反传）。
    diagnose_every: int = 10
    # solo 训练环僵局早停判平（与 eval 同语义：连续 100 步双方塔血零变化 → 判平）。
    # 否则躺平要拖满 max_ep_steps 才在 360 帧末罚一次 −10，(γλ)^k 视野内完全不可见。
    train_stall_stop: bool = True
    # —— critic 惰性检验（2026-09-13；预注册 docs/critic_inertia_prereg_2026-09-13.md）——
    # adv_inert_probe：**纯测量**开关（默认 False = 逐位旧行为）。开启后每个诊断更新
    #   额外算一份"V≡常数 c"的优势（c = ret_scaler.mean，即塌缩后的 critic 收敛到的常数），
    #   报告 corr(adv_real, adv_const)、状态依赖占比、策略损失梯度余弦 grad_cos；
    #   不改变任何被写入梯度的量，只多 1 次前向 + 1 次反传（每 diagnose_every 次更新一次）。
    adv_inert_probe: bool = False
    # critic_baseline：**干预**开关。value = 旧行为（优势用网络 V）；const = 把优势里的
    #   V 项整体替换为标量 c（returns / 价值损失 / 预算 / 对手池 / 奖励全不动）。
    #   仅用于"critic 惰性"实验，**不是**推荐配置。
    critic_baseline: str = "value"
    # —— C'（2026-09-12）早停低置信裁定降噪 ——
    # 早停（stall）局按 timeout_winner 结算时：皇冠不同 → 决定性，照常 ±胜负；
    # 皇冠相同 → 若"双方存活塔最低血量百分比差" < 该阈值，视为掷硬币级裁定，
    # 不再按塔血%细差判胜负，统一记平局惩罚（平局=失败，保留反躺平信号）。
    # 依据：docs/critic_probe_experiment_2026-09-12.md 实验 3 —— timeout_winner
    # 塔血裁定是标签噪声候选（早停局占 28~40%，其中皇冠相同的细差裁定噪声最大）。
    # 仅改训练侧标签（eval 仍用真实 CR 规则 timeout_winner，保证评估口径可对比）。
    # 2026-09-17：默认 0.0 = **不再按"低置信细差"判平**；塔血裁决改走"三塔血量合计"
    # 且只在完全相等时平局（docs/draw_rule_prereg_2026-09-17.md）。
    # 旧行为可逐位复现：--stall-draw-margin 0.05。
    stall_draw_margin: float = 0.0
    # —— F′（2026-09-12）：真正的 PPO 更新预算 ——
    # 旧实现 `PPOTrainer.update()` = **1 次 forward / 1 次 backward / 1 次 opt.step**，
    # 而喂进来的批是**同一局连续 128 帧**（corr(R_t,R_{t+1})≈0.99）⇒ 20k 步只有
    # 156 次梯度步、每次梯度目标几乎相同 ⇒ 价值头（四种架构都试过）EV 全部 ≤0，
    # 而同一表征的 MLP 探针 R² 有 0.24~0.42（"表征有信息、critic 吸收不了"）。
    # 依据：docs/rl_training_fix_plan_v3.md §3.11.1、docs/critic_probe_experiment_2026-09-12.md。
    # 默认值 = 旧行为（1 轮、整批、不打乱）；显式开启才切到多轮小批分支。
    # 开启后 ratio 会离开 1.000、clip_frac >0 —— 正常且期望（AGENTS 的
    # "ratio≡1.000 是结构性的"只对 n_epochs=1 成立）。
    ppo_epochs: int = 1
    ppo_minibatch: int = 0        # 0 = 整批（不切）
    ppo_shuffle: bool = False     # 每轮是否打乱样本顺序
    # —— 数据 / 运行时 ——
    decks_path: str = None      # 三分类卡组 JSON（缺省自动探测）
    # solo 镜像卡组选择："default"=原版 8 卡镜像；"four"=四种卡组对手池
    # （docs/four_decks_manual.md：defend 对手每局从四卡组抽一副，镜像卡组不变）；
    # "list:Card1,..."=显式镜像卡组（逗号分隔 8 张引擎卡名）。
    deck_set: str = "default"
    main_init: str = None       # BC 预训练 / 旧检查点
    # 热启动 run 的对手池补种（P1-1，2026-09-11）：本目录首轮训练时
    # solo_main_<step>.pt 池为空 → _OPP_MIX 的 hist 槽退化（实测 9k_ft 日志
    # "frozen=0.875/defend=0.125"，PFSP 历史槽从未生效）。指到旧 run 目录
    # （可多个）后，hist 槽从这些目录里均匀抽 ckpt，恢复 70/20/10。
    hist_seed_dirs: list = None
    device: str = "auto"        # cpu / cuda / auto（=cuda 可用则 cuda）
    only_vs_main: bool = False
    keep_snapshot: bool = False
    out_dir: str = "runs"
    # —— solo 自对弈（--mode solo，无联赛；原版 train.py 思路）——
    solo_copy_every: int = 2000   # 冻结副本同步间隔（步）：每 N 步把 main 权重拷给对手
    # —— 评估并行（eval_solo/play_pair 用进程池绕开 GIL；0 = 串行旧行为）——
    # 2026-09-11 审计整改：默认 0 → min(16, cpu_count())（16 核机器即 16，与
    # start_rl.bat 的 EVAL_WORKERS=16 一致）。实测 9k eval 周期（主+2 对照 × 16 局、
    # worker=12）167s；战斗模拟是纯 Python，跨进程才吃满多核。
    # ⚠️ worker=16 有已知的间歇性失败：加载 torch\lib\cufft64_12.dll 时
    # `WinError 1455「页面文件太小」`→ 部分/全部 worker 启动即崩 → **静默降级串行**
    # （eval ~1min → ~8min，结果等价只是慢）。**成因未定**（见 AGENTS.md 全局操作约定）：
    # 本机提交上限 47.3GB / 空闲 20.6GB，16 个 CUDA-torch worker ≈20.8GB 正好压在边界；
    # 但也有与宿主 shell 故障同时发生的旁证。遇到时按 AGENTS.md 的约定**暂停等人类**，
    # 不要自己猜着改。empirically 安全档位是 --eval-workers 12。
    eval_workers: int = field(default_factory=lambda: min(16, os.cpu_count() or 1))
    # —— 行为指标门禁（P0-2b，2026-09-11 判读整改）——**先只报警不阻断** ——
    # 阈值语义（除绝对值外有两种写法）：
    #   1) 数字       → 绝对阈值（旧行为）。op 推断：名字以 `_max` 结尾或名为
    #                   `ghost_rate` 用 `<=`，其余用 `>=`；
    #   2) {"rel":">=","frac":0.5} → **相对本 run 首个评估点**的比值门禁
    #                   （不低于起点的 50% / 不高于起点的 200%）。
    # 为什么改相对门禁：20k 验证跑证实**绝对阈值标定错了口径**——阈值 9.5 来自一次性
    # 取证脚本（9k_ft=10.5%），而训练内建指标对同一批 9k_ft 权重实测 28.3~38.1，
    # 差约 3 倍 → PASS/FAIL 语义是假的。相对自身起点则口径漂移自免疫。
    # 首个评估点（eval@0）只建基线不判定；baseline 存进 gates.json，续训沿用。
    gates: dict = field(default_factory=lambda: {
        "engagement_rate": {"rel": ">=", "frac": 0.5},
        "ghost_rate": {"rel": "<=", "frac": 2.0},
    })
    # solo 训练对手池配比（P1-1b + E2）：frozen/hist/defend/rand_anchor，缺省见 DEFAULT_OPP_MIX。
    opp_mix: dict = field(default_factory=lambda: dict(DEFAULT_OPP_MIX))
    # —— 价值通路架构（B'，2026-09-12）——
    # True = value_head 直连 post-LN enc（跳过 GRU），策略头仍走 GRU 隐状态。
    # 依据：实验 B' 监督对照 无 GRU +0.294 vs 带 GRU +0.217（docs/critic_probe_experiment...md）。
    # ⚠️ 架构变更：旧 ckpt（value_bypass=False 训练）不可续训到 True，须 --fresh
    #（load_checkpoint 读元数据，不一致时告警）。
    value_bypass: bool = False
    # —— 奖惩机制（每配置一套，见 DEFAULT_REWARD 注释）——
    reward: dict = field(default_factory=lambda: dict(DEFAULT_REWARD))
    # —— E'（2026-09-12）：独立价值编码器 + 非线性价值头 ——
    # True = 价值通路用自己的 `value_enc_fc`/`value_enc_ln`/`value_head_mlp`
    # （只吃 value 梯度、不与策略共享参数），优先级 independent > bypass > shared。
    # 依据：MLP 探针 `enc → return` R²=+0.42（线性仅 +0.135 ⇒ 单个 nn.Linear 头有
    # 先天上限）+ 监督同网络可达 EV 0.22~0.29 而 on-policy 联合训练 ≈0
    # + bypass（只换接线）无效 ⇒ 给价值通路自己的容量与梯度。
    # ⚠️ 架构变更：须 `--fresh`（ckpt 元数据带该标志，不一致时告警）。
    value_independent: bool = False

    # ---- 路径（全部落在 out_dir/<name>/ 下）----
    def folder(self):
        return os.path.join(self.out_dir, self.name)

    def state_path(self):
        return os.path.join(self.folder(), "league_state.json")

    def solo_state_path(self):
        return os.path.join(self.folder(), "solo_state.json")

    def solo_main_path(self):
        return os.path.join(self.folder(), "solo_main.pt")

    def solo_ckpt_path(self, step):
        """solo 历史检查点：solo_main_<step>.pt（每次评估保留一份，回溯用）。"""
        return os.path.join(self.folder(), f"solo_main_{step}.pt")

    def solo_opt_path(self):
        """solo 优化器状态（断点续练恢复 Adam 用）。"""
        return os.path.join(self.folder(), "solo_opt.pt")

    def run_state_path(self):
        return os.path.join(self.folder(), "run_state.json")

    def gates_path(self):
        """行为指标门禁报告（P0-2）：每周期评估后覆盖写。"""
        return os.path.join(self.folder(), "gates.json")

    def config_path(self):
        return os.path.join(self.folder(), "config.json")

    def replays_dir(self):
        return os.path.join(self.folder(), "replays")

    def main_final_path(self):
        return os.path.join(self.folder(), "main_final.pt")

    def ckpt_path(self, step):
        return os.path.join(self.folder(), f"main_ckpt_{step}.pt")

    def opt_path(self, step):
        return os.path.join(self.folder(), f"main_opt_{step}.pt")

    def ensure_dirs(self):
        os.makedirs(self.folder(), exist_ok=True)
        os.makedirs(self.replays_dir(), exist_ok=True)

    # ---- 序列化 ----
    def to_dict(self):
        return asdict(self)

    def save(self, path=None):
        path = path or self.config_path()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return path

    @classmethod
    def from_dict(cls, d):
        d = dict(d or {})
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        cfg = cls(**known)
        rw = dict(DEFAULT_REWARD)
        rw.update(d.get("reward") or {})
        cfg.reward = rw
        return cfg

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    # ---- 预设 ----
    @classmethod
    def presets(cls):
        return {
            "standard": cls(name="standard", description="默认奖惩机制：塔血统一 0.001/0.001、费差打开（1圣水≈500血）"),
            "aggressive": cls(
                name="aggressive",
                description="推进：破塔/皇冠/胜利奖励更高，费差加码（塔血换费差，1圣水≈700血）",
                reward={"crown_weight": 8.0, "win_bonus": 15.0,
                        "lose_penalty": 10.0, "invalid_penalty": 0.05,
                        "elixir_bonus": 0.0, "elixir_diff_weight": 0.7}),
            "defensive": cls(
                name="defensive",
                description="防守反击：非法动作惩罚更重，费差减码（1圣水≈300血）",
                reward={"crown_weight": 3.0, "win_bonus": 10.0,
                        "lose_penalty": 10.0, "invalid_penalty": 0.1,
                        "elixir_bonus": 0.0, "elixir_diff_weight": 0.3}),
            "lockdown": cls(
                name="lockdown",
                description="自闭：费差压到≈0（1圣水≈50血，鼓励费差换塔血，浪费仍小惩罚）",
                reward={"crown_weight": 8.0, "win_bonus": 10.0,
                        "lose_penalty": 10.0, "invalid_penalty": 0.05,
                        "elixir_bonus": 0.0, "elixir_diff_weight": 0.05}),
            "elixir": cls(
                name="elixir",
                description="鼓励圣水效率：默认机制基础上叠加每步按剩余圣水 shaping",
                reward={"crown_weight": 8.0, "win_bonus": 10.0,
                        "lose_penalty": 10.0, "invalid_penalty": 0.05,
                        "elixir_bonus": 0.01, "elixir_diff_weight": 0.5}),
            "economy": cls(
                name="economy",
                description="费差经济（默认机制别名）：塔损按塔血%归一化 + Δ费差 shaping（1圣水≈500血）",
                reward={"crown_weight": 8.0, "win_bonus": 10.0,
                        "lose_penalty": 10.0, "invalid_penalty": 0.05,
                        "elixir_bonus": 0.0, "normalize_tower_dmg": True,
                        "elixir_diff_weight": 0.5},
                gae_lambda=0.99,     # 纯RL冷启动：γλ=0.947→0.987，优势半衰期 13→53 帧
                                     # （终端±10 与 60-150 帧的出牌因果进入 GAE 视野）
                steps_per_eval=8000,  # 2026-09-11 审计整改：n_eval_games 16→40 后
                                     # 评估单次成本 ×2.5，频率减半抵消回来（单位步数的
                                     # 评估墙钟不变，曲线点更少但每点更可信）
                value_norm="running",  # 2026-09-11 判读整改：把审计验证过的价值通道
                                     # 修复设为训练预设默认（v/p 2.43→0.6~0.95）。
                                     # dataclass 默认仍是 "none"（= 旧行为），只影响
                                     # 走本预设的训练入口；--value-norm none 可回退。
                value_bypass=True,   # B'（2026-09-12）落地：value 直连 enc 跳过 GRU
                                     #（实验 B'：+0.294 vs +0.217）。架构变更 ⇒ 须 --fresh。
                value_independent=True,  # E'（2026-09-12）：独立价值编码器 + MLP 头
                                     # （优先级 independent > bypass）。架构变更 ⇒ 须 --fresh。
                only_vs_main=True),   # 联赛模式评估只测 main（15 对→5 对，评估量再 ÷3）；solo 不受影响
            "fast": cls(
                name="fast", description="小步快跑（冒烟/设备验证用）",
                total_steps=2000, steps_per_eval=500,
                n_eval_games=2, max_ep_steps=300),
        }

    @classmethod
    def resolve(cls, preset, load_config=None, **overrides):
        """按名字/JSON 解析出配置，再应用命令行 overrides（只覆盖合法字段）。

        注意：预设是共享实例，先深拷贝再覆盖，避免一次运行污染下次运行。
        参数名用 preset 而非 name，避免与字段 name 的 overrides 冲突。
        """
        if load_config:
            cfg = cls.load(load_config)
            if preset:
                cfg.name = preset
        else:
            presets = cls.presets()
            if preset in presets:
                cfg = cls.from_dict(presets[preset].to_dict())  # 拷贝，不共享
            else:
                raise ValueError(
                    f"未知配置 '{preset}'，可用: {', '.join(presets)}（或 --load-config）")
        for k, v in (overrides or {}).items():
            if v is None:
                continue
            if k in cls.__dataclass_fields__:
                setattr(cfg, k, v)
            else:
                raise ValueError(f"未知配置字段 '{k}'")
        return cfg


def reward_to_env(cfg: TrainConfig) -> dict:
    """把配置的奖励权重转成 RLEnv 接受的 reward_weights 字典。"""
    return dict(DEFAULT_REWARD, **cfg.reward)


def model_reward_weights(model_id: str, cfg: TrainConfig) -> dict:
    """按流派模型的奖励权重 = 所选预设之上叠加 MODEL_REWARD_OVERRIDES。

    flow 联赛 6 模型用（推进/防反/自闭差异化，main/all/random 用基线同一参数）。
    未知 model_id 直接回退到所选预设（不改基线行为）。
    """
    rw = dict(DEFAULT_REWARD, **cfg.reward)
    rw.update(MODEL_REWARD_OVERRIDES.get(model_id) or {})
    return rw
