# M7 数据接入三卡 — 实现决策与待办（2026-09 批次）

> 覆盖：Ronin（格挡反击）/ Vines（藤蔓束缚）/ Spirit Empress（双费用形态）。
> 数据层（gamedata.json 注入 + 官方数值表行）在上批已就绪并验证；本批只做引擎钩子。
> 来源标注约定：[gamedata]=快照注入字段、[Fandom]=docs/_fp_*.txt（2026-09-03 CDP）、[假设]=无数据来源的合理值。
> 验收：`scripts/test_m5_data.py` 40 断言全绿；回归 test_m2 72/72、test_m3_evo 62/62、
> test_m4_evo7 51/51、batch_smoke 197/197。

## 机制 1：Ronin 格挡反击（被动）

**数据契约**（gamedata `Ronin.summonCharacterData`）：`parryReflectPercent=200`、
`parryCooldownMs=3500`、`parryMeleeOnly=True`。

**实现要点**
- `card_mechanics.Ronin(BasicCharacter)`：新增通用**受击钩子**路径——
  `Entity.take_damage` 在 `last_hit_by` 记录处调用 `entity_holder.on_take_damage(amount, source)`，
  返回 `True` = 本次伤害被格挡（引擎短路，本体不掉血）。钩子放在 delayed 分支之前，
  因为 delayed 攻击经 `pending_damage` 重放时 source 已丢失，必须在首次调用处拿攻击者。
- 格挡语义 = 完全格挡（本体不受该次伤害）+ 反弹 200% 所受伤害给攻击者 + 进入 3.5s 冷却；
  冷却期间/远程攻击正常受伤。
- 近战判定：攻击者 `data.range ≤ 1.5`（官方 Melee 最长 Medium=1.2，1.5 为圆整容错）。
- 空中近战单位免疫格挡（官方：Bats/Phoenix 免受格挡影响 [Fandom _fp_Ronin 策略节]）。
- 反弹伤害走 `delayed=True`（下 tick 结算），天然避免双浪人互弹的无穷递归。

**【待 L4】**
- 【口径待 L4 对拍】官方口径里 Ronin 或有**主动技能形态**（gamedata `abilityData.RoninParry`），
  当前按纯被动实现。
- 【假设】部署即就绪（首次格挡无预热延迟）；官方首格挡是否需要充能未查证。
- 【口径待 L4】近战阈值 1.5 为启发式；官方近战/远程分类的精确判据（如 isMelee 字段）未从
  快照中确认。
- 【未建模】格挡触发时的视觉/位移表现；同一帧多目标近战同时命中的结算顺序（按引擎
  update 顺序，只格挡第一个到达的伤害）。

## 机制 2：Vines 藤蔓束缚（法术）

**数据契约**（gamedata `Vines`）：`areaEffectObjectData`（radius 2500 / lifeDuration 2000 /
damage 95（lv6 基准）/ crownTowerDamagePercent 25 / ticks 2 / multipleTargets 3 /
targetHighestHp / groundsAirUnits / buffData `Vines_Trap_Snare`：snareDurationMs 2500）
+ `cards_stats_projectile.VinesProjectile`（Epic 轴 damage_per_level 95..246，lv11=153）。

**实现要点**
- 数据补全（`evo_2025_data.apply_m7`）：gamedata `Vines.projectileData` 缺 `speed` 字段
  （弹道永不落地）→ 注入 600 [假设：Fireball 同款法术弹速；官方法术弹速带 350~1100]。
- 弹道落地钩子：`Projectile._on_arrive` 对 `VinesProjectile` 走专用分支（不走通用溅射——
  官方语义是「3 个最高 HP 目标」而非全域伤害）→ `spawn_vines_zone` → `VinesSnareZone`。
- `VinesSnareZone`（独立实体，仿 EvoEffectZone 的最小初始化）：
  - 落地锁定半径内 HP 最高的 3 个敌方实体（含建筑，Fandom 明示 troops or buildings）；
  - 每目标至多 2 跳伤害：第 1 跳落地即时、第 2 跳 +1s（跳间隔 = 领域时长 2s / 2 跳）；
    伤害用弹道已按战斗等级解析的 per-hit 值（lv11=153）；
  - 对 'King' 类塔按 25% 降伤（口径同 `AreaEffect._pulse`：普通建筑全额）；
  - 命中目标立即束缚：`apply_buff(stun=2.5s)`（不可移动+不可攻击，复用 freeze_timer 语义）；
  - **拽落**：空中单位 `data.is_air_unit` 临时关闭，落地时长=束缚时长；复飞用**绝对时刻**
    （`_vines_grounded_until`）判定——束缚期间 `Troop.update` 因 freeze 提前 return，
    相对计时不可靠；束缚结束后首个 update tick 复飞并清路径。
- 领域伤害用**即时结算**（`delayed=False`）：被束缚目标 update 早退，pending 伤害永不结算
  （实现中踩过的坑，已写进代码注释）。

**【待 L4】**
- ⚠️ **口径冲突记录**：属性表领域持续 `lifeDuration=2s` vs 卡面束缚 2.5s——当前取
  「领域存在 2s（两跳伤害在领域内结算）、束缚 buff 2.5s（可越出领域 0.5s）」。
- ⚠️ **对塔百分比冲突**：gamedata 25% vs Fandom History 1/6/2026 已削至 23%（快照未跟）——
  当前按 gamedata 25%。
- 【待 L4】buff 注册表 `Vines_Trap_Snare` 分 Small/Medium/Large/XLarge/XXLarge 五档
  （内存实证）——当前按统一档（2500ms）实现，分档映射（按目标体积？）未建模。
- 【假设】VinesProjectile 弹速 600。
- 【简化】锁定目标在领域期间死亡不补位（官方是否有二次选取未查证）；目标被束缚后脱离
  半径（被推挤）仍继续吃满两跳。

## 机制 3：Spirit Empress 双费用形态（部署规则，非技能）

**数据契约**：gamedata `MergeMaiden`（28000025，6 费）+ `MergeMaiden_Mounted/Normal`
（26000105/26000104，notVisible 形态条目）；`cards_stats_characters.MergeMaiden_Mounted`
（Legendary lv9 轴，lv11 hp=1798/dmg=309 + air/ground dps 数组）；Fandom 双形态属性
（air：1.6s/range 5/A&G/Medium(60)；ground：1.2s/melee 1.2/Ground/Fast(90)）。

**实现要点**
- 数据补全（`evo_2025_data.apply_m7`）：
  - 双形态条目只有 `summonCharacter` 引用（无数值体）→ 挂载 `summonCharacterData`
    （MERGE_MAIDEN_MOUNTED/NORMAL_SCD，数值基准 = Legendary 轴起始 lv9 1486/255，
    官方数值表行在 set_level 覆盖）；
  - `characters.MergeMaiden_Mounted.flying_height=1000`（内存补名册 air_units）；
  - `MergeMaiden_Normal` 官方数值表行**补建**（与 Mounted 行同源：Fandom 表两形态共享
    hp 1798@lv11 与 per-hit 伤害——air dps 159.4=255/1.6、ground dps 212.5=255/1.2 反推
    吻合；差异仅在攻速/射程/位面）；
  - collisionRadius 快照无字段 → [假设] 500。
- 部署规则（`battle.deploy_card` 专用分支，在通用 `can_play_card` 之前）：
  - 圣水 ≥6 → 6 费部署 `MergeMaiden_Mounted`；<6 → 3 费部署 `MergeMaiden_Normal`；
  - 手牌校验用**实际费用**（圣水 2 时 3 费形态正确拒绝）；手牌循环仍按 `MergeMaiden`
    推进（`_finish_deploy` 新增 `hand_card/actual_cost` 参数，默认值下行为与旧版完全一致）；
  - AI/RL action space 语义不变（仍是选这张牌，费用动态）；
  - `p.last_card_cost / p.last_card_form` 记录实际消耗与形态 → **镜像**复制上一形态并按
    「实际费用+1」扣费（官方：镜像沿用上一形态，费用不随圣水切档 [Fandom 策略节]）。
- 边界：恰好 6 圣水 → 空中形态（≥6 含 6）。

**【待 L4】**
- 【假设】collisionRadius 500（两形态同）；sightRange 5500 沿用通用默认。
- 【简化】ground 形态 loadTime=300（first hit 0.3s）、Mounted loadTime=600（first hit 0.6s）
  取 Fandom First Hit Speed；引擎首次攻击实现为 `hit_speed - load_time` 冷却起算，近似口径。
- 【待 L4】MergeTactics 合体语义（卡名 MergeMaiden 暗示的合体玩法）未建模——当前只实现
  双费用部署规则（用户口径：权威）。
- 【待 L4】镜像+觉醒位/镜像+双形态叠加的极端交互仅覆盖了基础路径。

## 工程记录

- 改动文件：`src/clasher_new/battle.py`（受击钩子 / 拽落恢复 / VinesSnareZone /
  Projectile 落地分支 / deploy_card 双费用 / _finish_deploy 参数）、
  `src/clasher_new/card_mechanics.py`（Ronin）、`src/clasher_new/evo_2025_data.py`（M7 节）、
  `src/clasher_new/card_utils.py`（apply_m7 挂载）、`scripts/test_m5_data.py`（新建）。
- `docs/evo_2025_new7.md` 未改动（M6 觉醒 7 张文档不受本批影响；回归 test_m4_evo7 51/51 佐证）。
- 新测试放 `test_m5_data.py` 而非追加 `test_m4_evo7.py` 的理由：后者是 M6 觉醒 7 张的
  51 断言验收基线（回归红线数字需稳定可对照），本批三卡为非觉醒的数据接入机制，
  机制族与验收口径独立，独立成档便于归因。
