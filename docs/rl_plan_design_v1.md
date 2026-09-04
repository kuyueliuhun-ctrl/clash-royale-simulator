# PlanToken 战术意图扩展设计 v1

> 状态：设计定稿，实现进度——结构先行(57 维)✅ / bp 组 10 意图 ✅ / **pp 组 ✅（punish、
> spell_finish、anti_spell、save_ace 特权精确版 + king_activate、protect_backline 先行
> 示范 + 与 bp 同链标签一致）/ 消融 ✅（plan 注入 vs 置零 + 逐意图采纳探针 region/hold）**；
> bait 留 backlog；pp 精确层价值经消融探针验证。
> 目标分支：main。前置：reward v2 Phase 1+3 已落地（0c0fbb1），本设计是 Phase 2。
> Plan 定位（复习）：plan = 教师每帧输出的**战术意图提示**，作为输入条件进 actor+critic；
> 模型可学会忽略坏 plan → 扩展价值上限 = planner 规则质量。
> 兼容：旧 21 维语义不动，全部扩展走尾部追加；旧 checkpoint 尾部补零加载。

---

## 0. 本轮修正与补充总览（用户指示）

修正已有候选：
1. `punish`：惩罚对面沉底/低圣水时，**不打同路，改打另一路**（趁对面资源投在别处压空路）；
2. `push_commit`：推进跟进触发条件从"坦克过桥后"改为"**坦克在地图上推进中即可在其后跟进**"；
3. `spell_finish`：语义从"斩杀收尾"改为"**后期靠法术持续磨塔血/压血线**"（配合 reward v2 后期塔血贵）；
4. `king_activate`：确认实现（激活国王塔，拉怪让国王塔参战）。

新增意图：
5. `cycle_small` 过牌：手里有小费牌(1-2费)且圣水充足 → 下小费轮转，保证手牌尽快出现高质量防守/进攻牌；
6. `pull` 拉扯：**不绑定"标准拉扯格"——位置不是关键，距离/改道才是**：用单位/建筑改变血牛的锁定
   与行进路线（可横穿把血牛从左路拉到右路、拉到己方输出集中点、或让它转锁我方建筑），
   拉得越远越久，己方塔/单位输出的时间越多（距离 = 拉扯收益）；
7. `protect_backline` 保后排：己方后排将受/已受近战攻击 → 放卡挤开或吸仇恨；远程输出手被锁定 → 吸仇恨；已被锁定 → 小电等重置目标；
8. `anti_spell` 防法术（泛化原 position_safe）：**关键是对手有什么法术、藏了什么**（信念/特权判威胁），
   目的是让对手法术收益最小化——火球/毒药/大闪/冰冻都要防，不限"防蹭血"；
   落点要考虑**单位会走**（移动轨迹），不是静态贴边格（防火球落点在国王塔前中轴附近这类
   "单位离开塔溅射区"的轨迹位，而非斜后方死位）；
9. `save_ace` 藏卡留费：大闪/藤蔓/冰冻等可能一波终结的卡要藏住（不在非关键时刻暴露），并在关键时刻前保留足够费用。

---

## 1. 意图总表 v1（= 保留 8 + 修正/新增 9 ≈ 17 个）

| # | intent | 做什么 | 目标 kind | 触发者 | 依赖特权？ |
|---|---|---|---|---|---|
| 0a | defend_left/right/king | 常规防守（保留） | unit | bp/pp | 否 |
| 0b | push_left/right | 常规推进（保留） | tower | bp/pp | 否 |
| 0c | counterpush | 防反（保留） | tower/unit | bp/pp | 否 |
| 0d | spell_value | 法术价值（保留，泛） | unit/tower | bp/pp | 否 |
| 0e | cycle_and_wait | 无目的等待（保留） | none | bp/pp | 否 |
| 1 | `soft_control` | 对威胁单位放冰冻/藤蔓 | unit | bp/pp | 否（可见单位） |
| 2 | `spell_trade` | 法术解后排/关键单位 | unit | bp/pp | pp 更强 |
| 3 | `punish` | 对面沉底/低圣水 → **另一路进攻** | tower | pp 强 / bp 弱 | 是（圣水/手牌） |
| 4 | `setup_wait` | 沉底蓄力（有目的等待/憋 combo） | none | bp | 否 |
| 5 | `push_commit` | 坦克推进中 → **部署到能走到坦克后方的区域**（不必紧贴正后方） | unit(坦克) | bp | 否 |
| 6 | `pre_defend` | 预判防守（对面即将进攻前占位） | unit | pp 强 | 是 |
| 7 | `bait` | 骗解/骗费 | unit | 无可靠触发 | —（暂不实现） |
| 8 | `spell_finish` | **后期法术持续磨塔血/压血线** | tower | bp/pp | 部分（time 即可） |
| 9 | `king_activate` | 激活国王塔（拉怪/蹭国王塔） | unit | bp | 否 |
| 10 | `cycle_small` | 小费过牌保手牌质量 | none | bp | 否（自己 cycle 可见） |
| 11 | `pull` | 拉扯：改道/横穿/拉远（**距离制胜**，不绑标准格） | unit(血牛) | bp | 否 |
| 12 | `protect_backline` | 保后排（挤开/吸仇恨/重置目标） | unit(我方后排) | bp/pp | pp 强（预判） |
| 13 | `anti_spell` | 防法术：对手有什么/藏什么 → 法术收益最小化（火球/毒药/大闪/冰冻；考虑移动轨迹） | none(自位置) | pp 强 / bp 中 | 是（对手法术推断） |
| 14 | `save_ace` | 藏终结卡 + 留费（本帧别出 ace，hold_mask 指名） | none | bp（藏）/pp（时机） | pp 强 |

保留 `bait` 进 backlog（无可靠示范源）。

---

## 2. 结构扩展（字段设计）

| 字段 | 现状 | v1 建议 | 说明 |
|---|---|---|---|
| `macro_intent` | 8 one-hot | 尾部追加新意图（含 cycle_and_wait 语义拆分：`cycle_small` 是新意图） | 前 8 位不动 |
| `target_kind` | 无 | 5 one-hot：none / unit / building / tower / my_backline | "对谁行动"；`my_backline` 服务保后排/防蹭血 |
| `elixir_budget` | 无 | 连续 0..1（允许投入/10） | 防守 ≤0.4 / combo ≤0.6 / 磨塔 0.3-0.5 / save_ace 时"保留费"用 1−budget 表达 |
| `phase` | 无 | 4 one-hot：none / setup / commit / cleanup | 跨帧组合阶段（先不加维度也行的备选） |
| `placement_hint` | 无（region 8 区太粗） | 动态策略位（修正版见 §2.1）：pull_across / pull_aggro / support_zone / anti_spell_zone / bridge_front / king_front / none | 表达"放位策略类型"，**不是静态标准格**；与 region 联合定路 |
| `opp_spell_threat` | 无 | one-hot：none / fireball / poison / lightning / freeze / big_unknown | `anti_spell` 核心：对手手牌/牌序里有什么法术（belief 后验或 pp 直读）；"藏了什么"由信念/特权威胁概率表达 |
| `hold_mask` | 无 | **4 bit 槽位掩码（1=本帧别出该槽）** | `save_ace` 用（方案 B 已定）：指名道姓哪张别出；多 ace 可同时标；可选配合动作掩码概率软禁 |
| `suggested_card` | 单卡槽位 | 保持 | combo 第二张由模型在 commit/同帧自行接 |
| `combo_hint` | 0-3 | 保持 0-3 | 1 坦克+后排 / 2 法术+单位 / 3 双路 |

维度预算（尾部追加，旧 21 位不动）：
- 意图追加 9-13 个：≈ +9~13
- target_kind：+5
- phase：+4（备选，暂不加，见取舍）
- placement_hint：+7（none/pull_across/pull_aggro/support_zone/anti_spell_zone/bridge_front/king_front）
- opp_spell_threat：+6（none/fireball/poison/lightning/freeze/big_unknown）
- elixir_budget：+1
- hold_mask：+4（方案 B 定案）
合计 21 → 21+9~13+5+7+6+1+4 ≈ **53-57**（砍 phase → 49-53）

> 取舍建议：v1 加 `target_kind`+`placement_hint`+`opp_spell_threat`+`elixir_budget`+`hold_mask`+新意图
> one-hot；`phase` 由 planner 每帧重规划切换意图表达（暂不加维）。全量意图已确认（用户：只有 2-3 个
> 是特定卡组才用，其余通用；one-hot 稀疏可接受）。

### 2.1 placement_hint 语义修正（用户 2026 评审结论）

placement_hint 表达的是**放位策略类型**，不是"静态标准格"：
- `pull` 类：**位置不是关键，距离/改道是**——`pull_across`（横穿换路，可左→右跨全图）、
  `pull_aggro`（拉到己方输出集中点/转锁建筑）。不存在唯一"标准拉扯格"；目标锁定/路径由 grid 可见，
  收益 = 拉出的距离 × 时间（reward/value 裁决）。
- `anti_spell` 类：**关键是对手有什么、藏了什么**（`opp_spell_threat`），以及**单位会走**——
  防火球选"单位轨迹离开塔溅射区"的落点（如国王塔前中轴附近），不是"斜后方贴边死位"；
  防大闪电 = 防三点成线、防冰冻 = 防扎堆被围杀。
- `support_zone`（push_commit）：部署到**能走到坦克后方**的区域，不必紧贴正后方。

结论（开放问题 1 已定）：离散策略位 + region 联合定路 + grid 自学精确格 + reward 校正；不给连续坐标。

---

## 3. 关键意图卡片（触发/产出/示例）

### 3.1 soft_control（软控打断）
- 触发：敌方 unit 近我方塔/在输出（threat 高）且手牌有 Freeze/Vines。
- target_kind=unit；region=威胁单位所在侧。
- 剧本：MiniPekka 砍塔 → 冰冻放它身上，让它少吃 4 秒塔血（reward v2 已让"塔血少掉"经 value 回流）。

### 3.2 spell_trade（法术解后排）
- 触发：坦克推进中(5) + 敌方在坦克路径/塔旁下防守后排；或敌方后排沉底准备反打。
- target_kind=unit；bundle_hint=2 时模型应同帧 火球+小电 打同一目标。
- 注：是否"亏费但赚"由 reward v2 资源账+塔血回流裁决，plan 只示范时机。

### 3.3 punish（趁虚另一路）【已修正】
- 触发（pp）：对面刚下 ≥5 费且圣水 <2；或对面沉底大费。
- 意图：**不在同路对攻，切另一路压空**——push_left/right 的 lane 与对方沉底路相反。
- 需要新增/复用字段：目标 lane 反推（planner 输出 region 时取 `enemy_另一侧`）。
- 剧本 C 修正：ThreeMusketeers 沉底左路 → 我右路压 Giant+后排（punish, region=enemy_right）。

### 3.4 setup_wait（沉底蓄力）
- 触发：低压力 + 手牌组合成型（如 Giant/Musketeer 都在手或即将入手）。
- 输出：suggested_card=坦克，region=own_center（沉底位）；后续交给 push_commit。

### 3.5 push_commit（坦克后跟进）【已修正触发与落点】
- 触发：我方存在"正在推进的坦克"（不要求过桥——从沉底出发沿路推进中即可）。
- 意图：部署后排**到能走到坦克后方的区域**——不必紧贴正后方，也不要求"过桥后才跟"；
  只要落点让后排沿路追上/跟在坦克身后即可（placement_hint=support_zone + region=坦克所在路）。
- region：坦克所在 lane；模型从 grid 选"可达坦克后方"的具体格。

### 3.6 spell_finish（后期磨塔）【已修正语义】
- 触发：t≥120（双倍期）且对面塔血 < 某阈值（持续压血线，不要求必杀）。
- target_kind=tower；法术选择由模型做（火球/毒药/大闪/火箭）。
- 与 reward v2 关系：late tower_dmg=0.002 已让"磨塔法术"正 EV，plan 负责示范"什么时候开始磨"。

### 3.7 king_activate（激活国王塔）【确认实现】
- 触发：我方一座公主塔被拆（或血量危险）、敌方血牛/大单位接近国王塔可被拉；
  或故意让法术/单位蹭到国王塔激活其参战。
- 剧本：Giant 走到国王塔前 → 放低费单位拉它到国王塔另一侧，国王塔开始输出（长期价值极高）。
- 依赖引擎：单位仇恨/国王塔参战机制已存在（battle 有王塔）；planner 只需示范"什么时候值得卖公主塔血换国王塔激活"。

### 3.8 cycle_small（小费过牌）
- 触发：当前无压力（threat≈0）且手牌含 1-2 费小牌、圣水充足（≥ 小费+3）。
- 意图：下小费牌轮转，使第 5 张高质量卡入手更快。
- 依赖：自己 cycle 完全可见（bp 可产出，无需特权）。
- 与 cycle_and_wait 关系：cycle_and_wait = 无牌可打/等费；cycle_small = 有目的用小费换手牌。

### 3.9 pull（拉扯）【已修正：距离制胜，不绑标准格】
- 触发：敌方血牛（Giant/Golem/PEKKA 类只锁塔或高威胁近战）在推进。
- 本质：**位置不是关键，距离/改道是**——让血牛：
  - 横穿换路（从左路拉到右路，可跨全图）：placement_hint=pull_across；
  - 拉到我方输出集中点（两塔齐射/单位集火）：placement_hint=pull_aggro；
  - 转锁我方建筑（墓碑/小屋），偏离原路拖延时间。
- 表达：intent=pull + target_kind=unit(血牛) + placement_hint ∈ {pull_across, pull_aggro} + region。
- 实现要点：planner 只给"拉它 + 横穿/集火方向"，血牛**当前锁定的目标**与路径由 grid 可见；
  用建筑横拉时由 planner 提示"出建筑换锁"（pull + 建筑卡）。
- 收益裁决：距离 = 拉扯收益，由 reward（塔血少掉/单位多打）+ value 回流教模型学会拉多远、何时放。

### 3.10 protect_backline（保后排）
- 触发：我方后排（Musketeer/火枪/弓手等）正被或将被近战接近（pp 可预判对手下兵）。
- 三档动作（按严重度）：
  a) 将被接近 → 前置肉盾吸仇恨；
  b) 近战已贴身 → 放单位"挤开"（碰撞位移）或在其路径放单位；
  c) 远程已锁定后排 → 小电/眩晕类重置目标。
- 表达：intent=protect_backline + target_kind=my_backline + region=后排所在侧。

### 3.11 anti_spell（防法术）【已泛化：不是"防蹭血站位"】
- 触发：要下后排/关键单位时，对手**可能持有法术**（bp：信念推断；pp：直读手牌/牌序），
  尤其 t≥120 后期"还没显露的卡"一律按可能有法术保守处理。
- 核心信息：`opp_spell_threat`（fireball/poison/lightning/freeze/big_unknown）——
  **对面有什么、藏了什么**决定怎么防：
  - 火球/毒药：落点选"单位移动轨迹上难以连塔带人"的位置（例如国王塔前中轴附近，
    单位向前走会离开塔溅射区）——注意**单位会走**，不存在"斜后方贴边"这种静态死位；
  - 大闪电：防"后排与塔三点成线被一起劈"——分散 + 不叠高价值单位；
  - 冰冻：防"我方单位扎堆被冻住围杀"——分批/拉开。
- 表达：intent=anti_spell + opp_spell_threat + placement_hint=anti_spell_zone；
  具体落点由模型从 grid + threat 解（不是 planner 给坐标）。
- 目的量化：让对手法术覆盖价值最小化（模型从 value 学"这手如果被火球打了亏多少"）。

### 3.12 save_ace（藏终结卡+留费）【方案 B 已定：hold_mask】
- 触发：手牌含 ace（大闪/藤蔓/冰冻/火箭，名单可配置）且当前不是"最强一波"时机
  （我方无坦克进场/对面圣水充足/对面有法术可反制）。
- 表达（方案 B）：
  - `intent=save_ace`
  - `hold_mask`（4bit）：指名道姓"槽 X 的 ace 本帧别出"（多 ace 可同时标）
  - `elixir_budget`：保留费（budget 低 → 这帧只花少量，为 ace 攒费）
  - 可选：概率软禁进动作掩码（0.5-0.7 概率把 hold 槽从可玩 mask 剔除，留探索口）
- ace 出场：planner 判"最强一波"（pp：坦克进场 + 对手低圣水 + 无解牌/单位扎堆）→
  save_ace 解除、hold_mask 清空、intent 转 spell_trade/soft_control/combo —— **解除即示范**。
- 学习闭环：藏（hold）→ 憋费（budget）→ 解除（关键帧）→ ace 打出大收益 → critic 学"藏到关键帧值钱"。

---

## 4. bp / pp 可产出性矩阵

| 意图 | BeliefPlanner（可见观测+信念） | ProphetPlanner（特权） | 需新规则 |
|---|---|---|---|
| soft_control | ✅ threat+手牌查 Freeze/Vines | ✅ | 小段规则 |
| spell_trade | ✅ 敌方单位+法术手牌 | ✅ 强（能看对面手牌决定值不值） | 小段规则 |
| punish | ⚠ 只能看"对面刚下了大费"(可见) | ✅ 低圣水判定 | 规则+**另一路反推** |
| setup_wait | ✅ | ✅ | 组合检测规则 |
| push_commit | ✅ 场上坦克推进中 | ✅ | 坦克检测+身后放 |
| spell_finish | ✅ time+塔血 | ✅ | 阈值规则 |
| king_activate | ✅ | ✅ | 新规则（王塔血量/位置） |
| cycle_small | ✅ | ✅ | 手牌扫描规则 |
| pull | ✅ 血牛检测 | ✅ | 位置启发（中轴） |
| protect_backline | ✅ 已接近可见 | ✅ 预判更强 | 近战距离检测 |
| anti_spell | ⚠ 信念推断"可能有法术"（威胁类型靠后验） | ✅ 直读对面手牌/牌序（威胁类型精确） | 信念查询 + opp_spell_threat 编码 |
| save_ace | ✅ 藏（自己手牌） | ✅ 时机（对手状态） | ace 名单 + 时机判定 |

> 依赖特权的意图（punish/anti_spell 威胁类型/save_ace 时机）只能靠 30% prophet 帧示范；
> bp 能产出的（soft_control/cycle_small/pull 等）70% 帧也示范 → 样本量更大。
> 实现顺序应优先"bp 可产出"组。

---

## 5. 实现顺序建议（每步独立提交、可回退）

1. **结构先行**：PlanToken 尾部追加（新意图位 + target_kind + placement_hint + opp_spell_threat
   + elixir_budget + hold_mask），旧 21 维补零加载；selftest 断言 PLAN_DIM 与旧 checkpoint 兼容。
2. **bp 组**：cycle_small → soft_control → spell_trade → push_commit → setup_wait → pull
   （6 个，BeliefPlanner 规则，70% 帧立即生效示范）。
3. **pp 组**：punish（另一路反推）→ spell_finish → anti_spell → save_ace 时机 → king_activate
   → protect_backline（prediction 版）。
4. **消融**：plan 注入 vs 置零（现有消融框架）；逐个意图统计"该 intent 帧的采纳率/后续行为"。

---

## 6. 开放问题（已定 + 待定）

**已定：**
1. ✅ `placement_hint`：**离散策略位 + region 联合 + grid 自学 + reward 校正，不给连续坐标**；
   语义按 §2.1 修正（距离制胜 / 防法术看对手有什么与单位轨迹 / 可达坦克后方），非静态标准格。
2. ✅ `save_ace` 负向约束 = **方案 B：`hold_mask`（4bit 槽位掩码）** + intent + elixir_budget，
   可选概率软禁进动作掩码。
3. ✅ 意图**全量 17 个**（用户已确认：仅 2-3 个特定卡组才用，one-hot 稀疏可接受）。
4. ✅ `phase` **不加**（靠 planner 每帧切换 intent 表达跨帧阶段，不占维度）。
5. ✅ ace 名单 **固定**（Lightning/Vines/Freeze/Rocket 等配置化固定名单，不按卡组动态）。
6. ✅ `cycle_small` 与 `cycle_and_wait` **保持两个意图**（有无目的之分）。
