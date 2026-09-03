# P0 游戏机制全面补全计划

> 状态：**引擎机制全部完成（族1-8 ✅，test_m2 72/72，batch_smoke 156/156）**；
> 当前唯一瓶颈 = 数据层（15 张 STATS_UNRESOLVED + Ronin/Vines，见 §8 数据攻坚）。
> 范围界定：卡组设计由使用者负责，本计划只保证 `CREnv(agent_deck, opp_deck)` 接口可接受任意合法卡组。
> 已确认的设计决策：**精英卡 = 有技能的卡（abilityData 体系）**；**觉醒触发 = 交替形态，周期分一/两回合**。

---

## 0. 当前总状态（2026-09-03 复盘）

| 模块 | 状态 |
|---|---|
| 族1 递增伤害 / 族2 弹道链 / 族2.5 女巫妈妈 / 族3 最小射程 / 族4 buff槽 / 族5 拉拽 | ✅（M2，test_m1+m2 全绿） |
| 族6 英雄能力 ×7（SkeletonKing/ArcherQueen/GoldenKnight/Monk/MightyMiner/LittlePrince/BossBandit） | ✅（card_mechanics._HeroBase + use_ability） |
| 族7 觉醒系统（34 卡，周期表 evolutions.py + 钩子族全接线） | ✅ |
| 族8 动作链 + 族8.5 曲线推导 | ✅ |
| 数据覆盖 | implemented 107 / needs_review 15 / 快照缺失 2（Ronin/Vines） |

### §8 数据攻坚（本轮启动，双线并行）
1. **内存节点树提取**（权威，官方国服 15.535 驻留池）：`re/pool_extract2.py` 解析 95.7 万节点/6.5 万字符串，
   证实池内含全部 2025 新卡（Vines/Ronin/Berserker/BossBandit/GoblinMachine/SuspiciousBush）与英雄/进化；
   key→value 关联结构重建中 → `re/official/extracted/`
2. **Fandom 采集**（17 张缺卡 per-level）→ `re/fandom_stats/fandom_card_stats.json`
3. 采集完成后走集成管线补录 `cards_stats_*.json` / `gamedata.json`，覆盖矩阵 107 → 124，
   重跑 `scripts/coverage.py` + `scripts/batch_smoke.py`

### 已关闭的历史缺口
- **X-Bow 数据**：`cards_stats_building.json` Xbow 行齐全（minimum_range=0，仅 Mortar 3500 有最小射程）——初版"无数据"判断过时
- **觉醒周期表**：`docs/evolution_cycles.json` 34 卡全 high 置信（Fandom 三源交叉验证，取数 2026-08-28），
  `evolutions.py EVOLUTION_CYCLES` 已落地（含 RageBarbarian=Lumberjack 勘误与 2026 平衡调整）

---

## 1. 数据层现状（已核实）

| 项 | 结论 |
|---|---|
| 卡牌总量 | gamedata.json 146 条 = 100 角色 + 26 法术 + 16 建筑 + 4 塔；已实现 47 张 |
| 待实现 | 约 95~100 张（含名称错位归并后） |
| 觉醒卡 | 34 张带 `evolvedSpellsData`（自带数值 + 机制钩子字段）；**18 张觉醒角色缺 per-level 数值表** |
| 有技能卡 | **9 张带 `abilityData`**：7 英雄（SkeletonKing/ArcherQueen/GoldenKnight/Monk/MightyMiner/LittlePrince/BossBandit）+ 2 活动卡（SuperHogRiderTerry/GiantBuffer，排除）。技能字段含 `cooldown`/`manaCost`/`resurrectBaseCount`/`spawnLimit`，数据齐备 |
| 觉醒周期 | 数据中**无字段**，需按真实游戏规则维护 per-card 配置表 |
| 名称错位 | `AngryBarbarians`=Elite Barbarians、`IceSpirits`/`FireSpirits` 复数形、`Pekka` 等，需三方对照 |
| 内部/活动卡 | `TriWizards`、`MergeMaiden_*`、`SkeletonWarriors_SpookyChess`、`SuperHogRiderTerry`、`GiantBuffer` 排除出可部署池 |
| 数据缺口 | X-Bow 无数据；Mortar 有数据但引擎缺最小射程；18 张觉醒角色数值 |

---

## 2. P0-1 数据治理层（地基）

- **`name_registry.py`**：internal 名 ↔ englishName ↔ cards.json id 三方对照；自动生成卡牌覆盖矩阵
- 内部/活动卡排除名单（白名单常量）
- 派生卡 `manaCost` 补录（Golemite/LavaPups/Barbarian，当前 `Card.elixir=0` 会污染场上价值计算）
- 觉醒数值推导管道：`evolvedSpellsData.summonCharacterData` 自带数值为基底，per-level 数组缺失时按基础卡稀有度索引继承（标注置信度）
- **验收**：全 146 卡 `Card()` 构造通过；数值抽查（火球 688@11、Knight 1766@11 等）；名称对照表评审

## 3. P0-2 引擎机制族扩展（7 族，按依赖排序）

> **进度（2026-09-02）**：族 1 ✅、族 2 ✅ 已实现并通过 `scripts/test_m1.py`（19/19）；
> 递增伤害数值模型已对照 Fandom 实测值（地狱龙 11 级 35/120/422 vs 引擎 36/120/420）验证。
> **进度（2026-09-03，本轮）**：M2 族3/4/5 代码已实现在先（test_m2 53/53）；本轮新增
> **M4.5 动作链（攻击序列 attackSequenceList）族** 并通过 test_m2（66/66 全绿）；
> 9 张 STATS_UNRESOLVED 以统一曲线（×1.0985/级，全稀有度一致，实测 947+ 相邻级比值）推导到 lv11；
> 批量冒烟 156/156（`scripts/batch_smoke.py`），覆盖矩阵 implemented 47→107。
> **进度（2026-09-03，16 级支持）**：`set_level` 等级索引改为按数值表行稀有度（修复 VoodooHog/
> SkeletonArmy/GoblinBrawler 错配）+ 越界曲线延续；**11-16 级全贯通**：`BattleState(card_level)` /
> `CREnv(card_level)` 指定战斗等级，法术/觉醒/英雄技能/动作链全按当前等级缩放（test_m2 72/72）。

### 族 1：递增伤害 ✅ 已完成
- Entity 增加 `ramp_target_id/ramp_timer/ramp_stage`（换目标/脱攻击范围即重置）；攻击伤害经 `ramped_damage()` 计算
- 数值模型（已验证）：阶段伤害 = `variable_damage2/3`（基准级绝对值）× (当前级伤害/基准级伤害)；切换阈值 = `variable_damage_time1/2` 每段毫秒的累计和
- 覆盖：Inferno Dragon / Inferno Tower / Monk（Monk 的 time1=0，非蓄力递增，已排除并标注）

### 族 2：弹道落地生成链 ✅ 已完成
- `Projectile._on_arrive()` 钩子：读 `spawn_projectile` 做二段弹（`SpawnProjectile` 类，从数值表行构建、绕过 Card 查表）；法术落地出兵（Goblin Barrel 经 `spawnCharacterData` 出 3 哥布林）；Graveyard 持续时间内确定性散布刷骷髅（黄金角）；Clone 复制友军（克隆体 1 血）；Mirror = PlayerState 记忆上张牌 + 圣水+1 重放（`_from_mirror` 路径绕过手牌校验）
- 验收：`scripts/test_m1.py` 六项行为断言 + 60s 混战回归全绿

### 族 2.5：女巫妈妈诅咒 ✅ 已完成（本轮）
- `VoodooProjectile` 的 `targetBuffData`（VoodooCurse）无 `speedMultiplier`（旧通用代码假设错误 → 命中即 KeyError 崩溃）
- 修复：`target_buff.get('speedMultiplier', 0)` 防护 + 实现诅咒语义——被诅咒目标死亡时生成 `VoodooHog`（归属施法者；经 `_scan_and_register` 注册为可构造卡）
- 覆盖：WitchMother / SuperWitch
- 验收：`test_witchmother_curse`（诅咒 → 死亡 → VoodooHog 归施法者）

### 族 8（M4.5）：动作链/攻击序列（attackSequenceList）✅ 已完成（本轮）
- **数据源**：`gamedata.json` 已含 `attackSequence`/`attackSequenceMode`/`attackSequenceList`（InfernoDragon_EV1 四级 14/47/165/330、Berserker 三连 40×3、ElectroDragon_EV1 动作组引用）；内存 dump（re/cr_dump）含完整动作链 JSON（觉醒熔炉 `iDrag_chaos_1`：`ActionTakeDamage` 14/47/165 + `SubActionsDelay:[0,50]` + Resolver/Shape/Filter），证实同一机制在更新版本沿用
- **实现**：
  - 数据层：`Card` 读取基础卡 `attackSequenceList`（Berserker 伤害仅在序列内 → 以序列首档作缩放基准）；`derive_evolved_stats` 透传三个字段；兼容觉醒数据两种格式（旧嵌套 + 新扁平 `source=ext`）
  - 引擎：`Entity.attack_seq*` 状态 + `ramped_damage()` 优先返回当前档伤害 + `_on_attack_done()` 推进/排队 + 脱锁重置（与 M1 蓄力同语义）
  - 两种序列语义：**Manual 跨攻击递增**（InfernoDragon_EV1 逐攻击推进、封顶末档）vs **单次多段命中**（Berserker 每 hit_speed/n 打一段）
- **验收**：`test_attack_seq_inferno_evo` / `test_attack_seq_inferno_battle` / `test_attack_seq_berserker`
- **待 L4**：ElectroDragon_EV1 的 `doAttackAction` 动作组解释器（dump 内 `ActionRunActionOnResolvedGameObjects` 等 ClassType 已定位）；觉醒熔炉等快照外新卡待数据接入后启用

### 族 8.5（数据推导）：9 张 STATS_UNRESOLVED 统一曲线推导 ✅（本轮，置信度中）
- **实测**：全稀有度（含 Champion）相邻级比值 avg≈1.0985、lv1→lv11≈×2.556（947+192+208+240+40 样本）；gamedata `summonCharacterData` 基准 = lv1
- **实现**：`Card.set_level` 缺失 per-level 表时按 `base × 1.0985^(level-1)` 推导（Knight 对照 690→1766 误差 ~0.1%）
- 覆盖：LittlePrince/Berserker/GoblinMachine/SuspiciousBush/GoblinDemolisher/Goblinstein/GiantBuffer/BossBandit/GoblinHut
- 状态：registry 仍标 `needs_review`（数据置信度中，不擅自升级）；`Card.stats_source='derived_curve'`

### 族 3：建筑最小射程
- `Building` 增加 `min_range`：目标进入最小射程内不攻击/切换目标
- 覆盖：Mortar（X-Bow 待补数据后自动接入）
- 验收：贴脸单位不受攻击的白嫖测试

### 族 4：通用 buff 槽
- 现有 `speed_buff/speed_debuff` 迁移为统一 `Entity.buffs[]`：heal-over-time（Heal）、freeze（停移停攻）、reset（清 `attack_cooldown` 与冲锋状态）、shield 恢复
- 覆盖：Heal / Freeze / Zap 系 / Electro Wizard / ZapMachine（Log 推退已有）
- 验收：buff 叠加/覆盖/到期规则单测

### 族 5：拉拽位移
- `force_displacement(pos)`：考虑碰撞与河道约束的强制位移 API
- 覆盖：Tornado / Fisherman
- 验收：拉拽落点 + 碰撞求解断言

### 族 6：英雄能力系统（= 已确认的「精英卡」定义）
- action space `MultiDiscrete([5,32,18])` → 扩展技能维度（保持 `slot=0` 语义不变以兼容旧 checkpoint）
- `Entity.ability_state`：冷却（`abilityData.cooldown`）、圣水消耗（`abilityData.manaCost`）、次数/计数限制（`spawnLimit`/`resurrectBaseCount`）
- 七张逐个实现：SkeletonKing 灵魂召唤大军；ArcherQueen 隐身（`targetable`/`invincible` 字段作者已预留）；GoldenKnight 冲刺链（`dashDamage`/`dashCount`）；Monk 反弹（`reflected_attack_damage`）；MightyMiner 钻地；LittlePrince 召唤守护者；BossBandit 冲刺
- 验收：每技能一条单测；旧模型兼容垫片

### 族 7：觉醒系统
- 卡组携带 ≤2 觉醒位；**触发规则：交替形态，per-card `evolution_cycle_count`（1 或 2 回合）**——数据无字段，按真实游戏查证后维护 override 表（默认 2）
- 形态切换读取 `evolvedSpellsData`；34 张觉醒钩子按字段族分批接入：减伤（`buffWhenNotAttackingData`）、攻击序列（`attackSequenceList`/`buffAfterHitsData`）、二段弹道（`projectile2Data`）、死亡行为（`onKilledActionData`/`deathSpawnCharacterData`）、护盾丢失（`shieldLostActionData`）、充能动作（`onStartChargingActionData`）
- 验收：触发时序测试（普通→循环 N 回合→觉醒→交替）+ 每张觉醒卡行为断言

## 4. P0-3 逐卡接入与验收体系

- ~20 张纯数据卡直接启用；85 张按机制族批量接入
- **回归金样**：47 张已实现卡用固定 schedule 回放对比实体轨迹快照，任何机制改动不得破坏
- 每卡验收三件套：行为单测 / 数值校验 / 回归通过
- 自动生成交付矩阵 `docs/card_coverage.md`（卡名 / 机制族 / 状态 / 测试）

## 5. 里程碑

| 里程碑 | 内容 |
|---|---|
| M1 | 数据治理 + 族 1（递增伤害）+ 族 2（弹道生成链） |
| M2 | 族 3（最小射程）+ 族 4（buff 槽）+ 族 5（拉拽） |
| M3 | 族 6 英雄能力系统全量 |
| M4 | 族 7 觉醒系统全量（34 张） |
| M5 | 覆盖矩阵 100% + 回归全绿 |

## 6. 风险与数据缺口

1. **觉醒周期表**需按真实游戏查证（数据无字段），每张的 1/2 回合归属要逐卡确认
2. **X-Bow 数据缺失**：需从新版游戏数据补录（raw-capture 管线可提取）
3. **18 张觉醒角色数值继承**可能与真实数值有偏差，交付矩阵中标注置信度
4. **英雄技能改变 action space**：旧 checkpoint 需兼容垫片（slot=0 语义保留）
5. 活动卡（SuperHogRiderTerry/GiantBuffer/MergeMaiden）不进可部署池

## 7. 与后续阶段的接口

- **P1**（记牌器/ΔΦ）：依赖 P0-1 的名称治理与派生卡 manaCost 修正
- **P2**（联赛）：只需 `CREnv(agent_deck, opp_deck)` 接口，本计划保证任意合法卡组（含觉醒位/英雄位）可跑；卡组池由使用者设计
- **P3**（SampleFactory APPO + v-trace）：与 P0 正交，联赛框架原样迁移

---

## 附录：内容补充方案（数据与资产缺口）

### B.1 盘点结论（已核实，较初版判断乐观）

- **X-Bow 其实存在**：内部名 `Xbow`（小写 b，6 费建筑卡，`summonCharacterData` 齐全）——初版"无数据"判断是名字大小写踩坑，进一步证明名称治理层必要
- 两份 gamedata 副本逐字节一致，无版本漂移
- cards.json（123 张官方元数据）与 gamedata 按 englishName 对照后，**仅缺 2 张新卡**：`Vines`（Season 75 新法术）、`Ronin`（Season 85 新卡），均为 gamedata 快照之后官方推出
- 最终内容缺口清单：① Ronin/Vines 完整战斗数据 ② 18 张觉醒角色 per-level 数值 ③ 觉醒周期表（数据无字段，需查证）④ 各机制族的验收基准值

### B.2 数据来源分级

| 级别 | 来源 | 用途 |
|---|---|---|
| L0 | 站内深挖：cards.json 元数据、stats 四表未消费字段、client_side 副本 | 零成本，先榨干 |
| L1 | **RoyaleAPI cr-api-data**（GitHub）：本项目 stats json 的同源 schema（`flying_height`/`damage_per_level` 完全吻合），直接拉新版 | 补新卡数值、觉醒 per-level、验证现有表 |
| L2 | APK 资产提取：目标包 assets 内 `Logic*.csv`（gamedata.json 的原始出处，`meta.fingerprint` 即游戏内指纹） | 最权威最新，一劳永逸 |
| L3 | 官方 Clash Royale API：元数据/图标（`download_images.py` 管线已在用） | 名称/图片/稀有度 |
| L4 | **raw-capture 行为对拍**：Frida 管线抓真实对局 | 觉醒周期、递增伤害时点等"查证型知识"的实证 |

### B.3 四类内容工作流

| 缺口类型 | 方法 | 产出 |
|---|---|---|
| 新卡数据（Ronin/Vines） | L1 拉新版 stats + L2 补机制字段 | gamedata 增量条目 + 接入测试 |
| 觉醒数值（18 张） | evo 自带数值为基底 + L1 per-level 继承推导 | `set_level` 推导管道（P0-1 已列，标注置信度） |
| 觉醒周期表 | L4 抓包实证 / RoyaleAPI 觉醒卡页查证 | `evolution_cycle_count` override 表 |
| 机制验收基准 | L4 对拍 + 社区文档 | 每机制族的单元测试期望值 |

### B.4 版本锚定策略

- 以 `gamedata.json → meta.fingerprint` 锚定当前快照的游戏版本；所有数值断言相对该版本有效
- 升级流程 = 替换 gamedata 快照 → 重跑覆盖矩阵脚本（B.1 的对照脚本固化到 `scripts/coverage.py`）→ 差异清单进工作队列
- cards.json 中超出快照的卡（Ronin/Vines）在交付矩阵中标注来源 Season
