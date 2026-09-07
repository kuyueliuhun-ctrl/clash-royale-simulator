# 觉醒机制钩子缺口矩阵（2026-09-03 审计；M5 觉醒补全后更新）

> **M6 影响确认（2026-09-03）**：2025 新觉醒 7 张（快照无 evolvedSpellsData）已在独立数据层
> `src/clasher_new/evo_2025_data.py` + `evo2025Hooks` 钩子族实现, **不改动本矩阵任何条目**——
> 上表 26 张卡的 evolvedSpellsData 字段消费路径未变。两处共享引擎修复与本矩阵条目语义兼容
> （回归全绿验证）：① `apply_buff` 减速窗口改写 `debuff_time_remaining`（修复减速只生效 1 tick
> 的潜在 bug, 此前矩阵内消费减速的字段：Firecracker 灼烧带/Valkyrie 龙卷）；② `Entity.take_damage`
> 新增 `pierce_invincible` 参数（默认 False, 既有调用行为不变, 仅亡影法术穿透使用）。
> 另 test_m2「觉醒形态构造」计数 34→41（7 张数据层落地后可构造, 属数据补全非语义变更）。

> 生成: scripts 内联审计; 视觉/导出类字段已过滤; 字段名 = evolvedSpellsData 内未消费的键
>
> **M5 觉醒补全（2026-09-03）**: 通用动作组解释器 + 逐卡钩子已落地, 状态标注如下:
> - ✅ = 本轮实现（引擎可执行, 见 src/clasher_new/battle.py / card_mechanics.py / evolutions.py）
> - ✅(已有) = 引擎既有机制已覆盖（如 attacksGround 由 tidTarget 通用驱动）
> - ⏭️ = 跳过（纯视觉/展示字段, 或数据无数值语义——理由逐条标注）
>
> 数值来源: gamedata.json evolvedSpellsData（lv1 基准, ×1.1^10 曲线缩放 11-16 级）;
> 动作组语义参考内存 dump re/cr_dump（iDrag_chaos_1 动作链: ActionTakeDamage/Damage.BaseDamage/
> SubActionsDelay + Resolver/Shape/Filter 结构）。
> 验收: scripts/test_m3_evo.py（62 断言）+ scripts/test_m2.py 72/72 + scripts/batch_smoke.py 全绿。

**26 张卡存在游戏性缺口, 共 78 个未实现字段 → M5 后: 实现/覆盖 61, 跳过 17**

## 有游戏性缺口的觉醒卡

### Archer
- `specialAttackRangeForStatsTid` ✅ 语义实现：`specialAttackRangeForStats=4500` → 二段箭仅 4.5 格内附加（TID 本体为展示字段, 跳过）

### Pekka
- `attacksGround` ✅(已有)（tidTarget 通用）
- `onKilledDoneAction` ✅ 击杀治疗（PekkaEV1_Heal, 治疗=resurrectParameters[2]×曲线, 击杀归因经 take_damage source）
- `resurrectParameters` ✅ [0,2000,500,500,5000,5000,900,10000,200]：延迟 2s / 基础 HP 500 / 灵魂加成 200 / 上限 10000 / 临时存活 5s（[4] 语义假设, 待 L4）
- `tempResurrect` ✅ 一次性临时复活（复活体不再二次复活）

### Witch
- `attacksGround` ✅(已有)
- `spawnPauseTime` ✅ 出兵间隔数据化（7000ms, 此前硬编码 7.0）
- `subActionsData` ⏭️ 跳过：动作组仅含事件名（On_Skeleton_Destroyed）, gamedata 无任何数值语义

### Valkyrie
- `attractPercentage` ✅ 迷你龙卷吸引（300 → 3 格/s 向心吸引, EvoEffectZone）
- `hitFrequency` ✅ 龙卷伤害脉冲周期（400ms × damagePerSecond 83）
- `nextAction.spawnData` ⏭️ 跳过：Valkyrie_NotPushed_BUF 无数值定义

### Bomber
- `spawnChain` ✅ 二段爆炸链（BombSkeletonProjectile_2_EV1 damage 88, 弹道命中点追加 AoE）

### Musketeer
- `attacksGround` ✅(已有)
- `customRange` ✅ 狙击弹 30000 → 每 2 发可锁定 30 格内任意目标（attackSequenceList 交替）
- `subActionsData` ✅ 狙击目标锁定组（snipe_targeting 语义由 customRange 射程/索敌扩展实现; 纯效果子动作自然跳过）

### Wizard
- `subActionsData` ✅ ShieldLostAoE（radius 3000 / damage 110, 护盾破碎即爆）; ShieldVFX 视觉子动作跳过

### IceSpirits
- `onHitTargetActionData` ✅ 命中点友方狂暴领域（spawnTime 3000ms; 数据无倍率 → 官方 Rage 现行 1.30, 标注默认）

### RageBarbarian
- `attacksGround` ✅(已有)

### InfernoDragon
- `actionToExecute` ✅ 常驻 ticker（UpdateDecayCounter/UpdateAttackSequence）语义 = M4.5 攻击序列推进/脱锁重置, 已实现
- `attacksGround` ✅(已有)
- `classType` ⏭️ 跳过：ActionGroup 结构标记, 非游戏逻辑
- `onStartingAttackAction` ✅ ResetDecayCounter = 脱锁重置到首档（M4.5 序列重置语义一致）
- `subActions` ✅ 同 actionToExecute 组成（语义已由攻击序列覆盖）
- `targettedDamageEffect` ⏭️ 跳过：视觉

### BlowdartGoblin
- `attacksGround` ✅(已有)

### Hunter
- `attacksGround` ✅(已有)
- `customFirstProjectile` ⏭️ 跳过：与基础弹道数据完全一致（同名同参）
- `customFirstProjectileData` ⏭️ 跳过：同上
- `damageExportName` ⏭️ 跳过：视觉
- `multipleProjectiles` ⏭️ 跳过：基础卡霰弹 10 弹丸建模缺口（非觉醒新增语义）
- `targetFilter` ✅ 网缚仅地面部队（default_character_targets_no_buildings）; 网缚=首攻束缚 1.0s（时长数据缺失 → 默认值标注）

### AxeMan
- `attacksGround` ✅(已有)
- `targetedHitEffect` ⏭️ 跳过：视觉

### Bats
- `allowedOverHealPerc` ✅ 过量治疗上限（200 → 可治疗至 3×max_hp）
- `hitFrequency` ✅ 自愈脉冲周期（500ms × healPerSecond 30, window=buffAfterHitsTime）

### MegaKnight
- `attacksGround` ✅(已有)
- `dashFilter` ✅ 冲刺仅地面目标（本卡 tidTarget=GROUND 保证）
- `dashFollowUpMaxRange` ✅ 随数据存档（doFollowUpJump=false → 不启用后续跳）
- `dashFollowUpMinRange` ✅ 同上
- `dashMaxRange` ✅ 冲刺跳外边界（5000）
- `dashMinRange` ✅ 冲刺跳内边界（3500）
- `doFollowUpJump` ✅ 数据值 false → 无后续连跳（配置消费）
- `dashDamage`（baseData 内, 矩阵关联字段）✅ 冲刺落地 AoE 210×曲线 + 上勾拳 pushBackStrength 4000→4 格击退

### Wallbreakers
- `attacksGround` ✅(已有)
- `nextAction.spawnData` ⏭️ 跳过：WallbreakerBarrelExplosion_EV1 无数值定义
- `onKilledActionData`（关联字段）✅ 亡语 Wallbreaker_mini 出兵（hp64×曲线, kamikaze）

### GoblinGiant
- `actionToExecuteData` ✅ 投掷哥布林（内嵌 Goblin 定义, 动作组解释器出兵）
- `actionsData` ✅ 血量阈值（healthPercentages 50%）触发 + interval 1800ms 连续投掷
- `attacksGround` ✅(已有)
- `deathSpawnCount` ⏭️ 跳过：嵌套于 SpearGoblinGiant 背部枪哥布林亡语链（基础卡行为, 非觉醒钩子）

### ElectroDragon
- `attacksGround` ✅(已有)
- `chainedHitCount` ✅ 链式弹射（命中后向 count-1 个最近敌人弹射, 同伤害+ZapFreeze 眩晕; 链半径无字段 → 默认 3.0 标注）
- `doAttackAction` ✅ 动作组引用 → 链电攻击组实现（attackSequenceList 无伤害档不挂载, 修复 0 伤劫持 bug）

### Firecracker
- `hitFrequency` ✅ 烟花灼烧领域脉冲（250ms × damagePerSecond 20, 3s 领域 + 减速 15%）

### Cannon
- `subActionsData` ⏭️ 跳过：Cannon_EV1_barrage 动作组在 gamedata 与内存 dump 中均无伤害/弹数参数, 禁止编造（待 L4 对拍补数值）

### Mortar
- `attacksGround` ✅(已有)

### Tesla
- `hitFrequency` ✅ onHitActionData buff hitFrequency=-1 → 一次性 buff 语义（不出周期脉冲）
- `onHitActionData` ✅ 出场脉冲 DOT（Tesla_EV1_WithDamage: damagePerSecond 68 × lifeDuration 1.5s; 眩晕同步数据化为 1.5s）

### GoblinCage
- `attacksGround` ✅(已有)
- `captureRadius` ✅ 捕获半径 3000
- `damagePerHit` ✅ 捕获伤害 132×曲线 / 捕获周期
- `deathSpawnCount` ✅ 亡语出觉醒斗士（GoblinCage_EV1_GoblinBrawler ×1; 修复: 觉醒建筑死亡钩子此前缺失）
- `hitFrequency` ✅ 捕获周期 1000ms（束缚刷新+伤害节拍）
- `numberOfUnitsToCapture` ✅ 同时捕获 1 个
- `subActionsData` ✅ 捕获动作组（CaptureUnit）
- `targetFilter` ✅ GroundCharacterTargetsNoBuildings → 仅地面部队（不含建筑/空军）

### GoblinDrill
- `attacksGround` ✅(已有)
- `deathSpawnCount` ✅ 亡语钻出 2 哥布林（觉醒建筑死亡钩子）
- `hideHpThresholds` ✅ [66,33]% 血量阈值各触发一次隐匿
- `hideTime` ✅ 隐匿 1000ms（不可选取+无敌, 期间不衰减不攻击）
- `spawnCharacterOnHide` ✅ 隐匿时钻出 Goblin
- `spawnCharacterOnHideCounts` ✅ [1,1] 每阈值出 1 只
- `spawnPathfindEffect` ⏭️ 跳过：视觉
- `spawnPauseTime` ⏭️ 跳过：基础卡出兵节流（引擎建筑出兵循环未建模, 非觉醒钩子语义）
- `subActionsData` ✅ relocate 动作组（官方同时迁移位置, 数据无落点字段 → 原地隐匿, 标注假设）

### GoblinBarrel
- `attacksGround` ✅(已有)（法术卡, 字段不适用）
- `decoyData` ✅ 诱饵（decoyData.spawnCharacterData → GoblinDummy 假哥布林落点旁 1 格; 数量无字段 → 1, 标注假设）

### Zap
- `areaEffectObjectData` ✅ Zap_EV1 持续领域（半径 2.5 / 5s, 进入眩晕 0.5s/敌一次; 首脉冲沿用基础 Zap 伤害）
- `subActionsData` ✅ 次级 AOE（Zap_EV1_SpawnAOE_medium: 半径 3.0 一次性眩晕）; DummyZap 视觉子动作跳过
- `tidAction` ⏭️ 跳过：展示字段

## M5 实现清单（引擎侧）

| 模块 | 内容 |
|---|---|
| evolutions.py | M5_EVO_PASSTHROUGH 字段族白名单 + collect_evo_mechanics 全树机制字段收集（动作组内嵌参数） |
| battle.py | interpret_action_group 通用动作组解释器（伤害/出兵/区域效果/递归子动作）; EvoEffectZone 通用效果领域（DOT/减速/冰冻/吸引/友方增益）; EvoZapZone 眩晕领域; spawn_evo_zone; _tesla_evo_pulse; Pekka 临时复活队列（BattleState.resurrect_queue）; MegaKnight 冲刺/上勾拳; ElectroDragon 链电（Projectile._evo_impact）; Bomber 二段爆炸; Firecracker/IceSpirits 命中领域; Hunter 网缚; Musketeer 狙击射程; Wizard 护盾破碎爆炸; Bats 过量治疗上限; Barbarians 攻速+移速双驱动; Archer 二段箭射程边界; Valkyrie 迷你龙卷数据化; GoblinGiant 阈值投掷; GoblinDrill 隐匿; GoblinCage 捕获; 觉醒建筑死亡钩子; 法术出牌觉醒计数统一（_finish_deploy） |
| card_mechanics.py | MegaKnight / Musketeer 机制类; Prince.on_tick 觉醒冲锋羊 e 引用先于赋值 bug 修复; Witch 出兵间隔数据化 |
| core.py | BasicCharacter.on_attack 击杀归因（take_damage source） |
| card_utils.py | evolvedSpellsData 全量扫描登记内嵌角色定义（Wallbreaker_mini / GoblinDummy / Goblin 等） |
