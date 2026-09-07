# Elite17 实现规格 — 17 张精英卡 (Hero 化变体)

> 数据契约: **内存 = 官方 15.535 权威** (re/official/memdump); Fandom = 结构/数值补充 (docs/_elite_*.txt, 抓取于本会话)。
> 冲突双记。证据附录: `re/official/extracted/elite17/_memory_appendix.md`; 每卡原始提取: `re/official/extracted/elite17/<卡名>.json`。
> 本文件只做数据与规格, 不含引擎改动 (src/ 未触碰)。

## 0. 共同激活语义 (适用于全部 17 张)

1. **形态装配**: 精英卡 = 普通卡 + Hero overlay 配置。内存 overlay 表显式给出 `Base = "CHARACTER.<普通卡>"` (Knight 表 0x76c726387cc0 等) → 引擎实现为: 复用普通卡 Character 数据, 叠加 overlay 列 (Ability/PrefabAsset/ShieldHitpoints/Dash* 等) 与 Hero 专属 buff/action group。Hero 数值 = 普通卡等级表重新计费后的数值 (Fandom 各卡 Statistics 的 HP/Damage 与普通卡不同, 例如 Knight 普通 L1 690/80 → Hero L1 681/94: **Hero 是独立数值表, 不是简单倍率**)。
2. **能力模型 = 冠军式 `use_ability`**: 部署后出现能力按钮; 激活耗圣水 (Ability Cost 1~6); 2026-08-04 起全部能力**单次使用** (此前为 N 秒冷却, 各卡 History 可查); 能力未开始施放即死亡 → 返还已付圣水。内存侧佐证: 全局无 `ManaCost` 列, 但存在与冠军同族的符号族 `*_Set_Ability_Button_Pending / *_Reset_Ability_Button / *_Reset_Ability_Charges / *_Execute_Charge / *_set_ability_button_cooldown / *_Limited_Time_Ability_Button / Ability_Button_Defaults`。
   - 引擎建议: 直接复用 Champion 的 `use_ability` 路径 (manaCost 扣费 + pending/button 状态机 + 单次使用 flag), 能力效果走 `interpret_action_group`。
3. **Wild slot**: 2026-03-16 起卡组槽位 = 1 Evolution + 1 Hero + 1 Wild; Wild 槽 Arena 10 解锁, 对局内把该卡激活为**觉醒或 Hero 形态 (二选一, 互斥不叠加)**。每套卡组 Hero 数上限 2 (Hero 槽 + Wild 槽, 与 Champion 槽共享), Arena 5 解锁。HeroCoin/Hero fragments (200/卡) 是元进度解锁货币, 不进战斗模拟。
   - Valkyrie (有觉醒又有 Hero): 同一 Wild 槽二选一。选 Hero → 装配 `CHARACTER.Valkyrie` overlay (0x76c726418b80); 选觉醒 → EV1 装配。**不可同时**。
4. **充能语义分类结论**: 17/17 全部走冠军式 abilityData/use_ability 模式。特例:
   - Goblins: 按钮条件性禁用, 最后一只哥布林死亡后开启 5s 窗口 (`GoblinHero_Flag_Limited_Time_Ability_Button`, `GoblinHero_Reset_Ability_Charges`)。
   - MegaMinion: 部署时被动挂"最低 HP 敌人"标记, 按钮瞬移到标记处。
   - BarbarianBarrel: 唯一 spell Hero, 按钮在桶滚出后触发二次滚 (`BarbLogHero_override_ability_button_when_rolling`)。
5. **数据缺口 (全局)**: ① buff/能力数值参数未从符号注册表解出 (注册表只含名字; 参数在动作序列化流中, 格式已部分破译: 记录头 `03 01 03`/`03 01 02` + 池字符串指针, 待后续攻关); ② 1335 张表中无 HitSpeed 列 (攻速整体缺失, 用 Fandom 值); ③ 5 张 HeroSpell 投掷物参数 missing; ④ IceWizard/EliteArcher/IceGolemite 的 wiki 页缺失 (missing, 见各卡节)。

每卡格式: 普通版基准 (内存 lv1) → Hero 差异 (Fandom) → 能力机制 → 引擎实现建议 → 数据缺口/完备度。
- **实现状态 (M8)**: 已全部实现。数据层 `src/clasher_new/elite17_data.py`（17 卡 abilityData 注入 + HERO_BASE_STATS 独立数值表 + 5 个派生角色注册, 逐条来源标注）；机制 `card_mechanics.py` M8 段（HERO_CLASSES 16 个 Hero 机制类）；引擎挂钩 `battle.py`（`apply_hero_overlay` 装配 / `BattleState.use_ability` M8 分支=单次使用+圣水预检+失败返还+条件窗 / `Entity._hero_taunt_override` 嘲讽覆盖）；Wild slot 互斥 = `PlayerState.set_hero_slots`（≤2）+ `deploy_card` 觉醒禁用; HeroCoin 未建模; 卡组声明 hero 化才生效（向后兼容, action space 不变）。验收 `scripts/test_m6_elite.py`（84 断言）。


---


## 1. Knight (Hero Knight)
- **普通版基准** (内存 0x76c72648bac0): HP 690, Damage 80, Range 1200 (1.2 格), Speed 60 (Medium), Mass 10。Fandom L1 Hero: HP 681 / Dmg 94 / DPS 78 / **Shield HP 197**。
- **Hero 差异**: 数值独立表; 内存 overlay 0x76c726387cc0: `Base=CHARACTER.Knight`, `Ability=Knight_hero_Ability`, **`ShieldHitpoints=200`** (≈Fandom L1 197, 平衡史: 后 -6%、-33%), `OnStartingAction=Knight_hero_SetShieldZero`, PrefabAsset=hero_prefab_knight_battle。
- **能力: Triumphant Taunt (2 费, 单次)**: 1s 延迟后嘲讽 7.5 格内所有敌军部队+建筑 (2026-03-02 削为 6.5, Taunt Trigger Window 0.7s→0.1s) 攻击自己, 同时获得护盾; 嘲讽与护盾均持续 5s (护盾可被提前打掉, 嘲讽目标仍保持锁定直至死亡)。被动: 无。
- **引擎实现**: 能力 = `EvoEffectZone`(半径 6.5 格) 一次性施加 Taunt buff (Knight_hero_IsTauntedBuff, 5s) + 自身 Shield buff (Knight_hero_AbilitySelfBuff + ShieldHitpoints, 5s)。嘲讽 = 强制 retarget 到自身, 需要在 targeting 优先级插入 taunt 覆盖层 (参考既有 taunt/Champion 逻辑; `AbilityPendingEffect=Heal_aura` 列表明复用 pending 特效挂点)。
- **数据缺口**: buff 参数 (时长 5s/半径 6.5) 目前仅 Fandom; 内存侧 buff 键已在注册表确认。
- **完备度**: high。
- **实现状态 (M8)**: 已实现（`HeroKnight`）。独立表 681/94/盾 197 [Fandom L1 → lv11 ×1.1 曲线]；嘲讽 6.5 格 / 5s（部队+建筑, `Entity._hero_taunt_override` 强制锁定）+ 能力护盾 5s（既有护盾槽）；2 费单次。测试 M8-01。


## 2. Musketeer (Heroic Musketeer)
- **普通版基准** (0x76c72637ae40): HP 282, Range 6000 (6 格), Projectile=MusketeerProjectile (Damage 列未解码, Fandom L3 Dmg 101 / L16 349)。
- **Hero 差异**: overlay 0x76c7263875c0: `Ability=Musketeer_hero_Ability`。**Hero 炮塔**为独立单位: 内存 0x76c726382a80 (TID_BUILDING_CANNON 列混排, 中置信): HP 600, Range 4000, Projectile=MusketeerTurret_Projectile, PrefabAsset=hero_prefab_musketeer_turret_battle; Fandom L3: TurretHP 717 / 10s 衰减 71.7/s / TurretDmg 65 / TurretSpawnDmg 95。
- **能力: Trusty Turret (3 费, 单次)**: 1s 延迟后在前方 3 格放置自动炮塔, 射程 4 格, 对空对地, 寿命 10s, 期间线性衰减 (decay/s), 落地时造成 spawn 伤害。
- **引擎实现**: 能力 = spawn building (寿命 10s + 线性 HP 衰减 + deploy 落地 AOE) — 复用 spawner/building 生命周期; 炮塔攻击复用普通攻击管线换 Projectile。
- **数据缺口**: 内存炮塔表行对齐噪声 (与 Cannon 列混排) — 炮塔数值以 Fandom 为准, 双记。
- **完备度**: high。
- **实现状态 (M8)**: 已实现（`HeroMusketeer` + 派生角色 `MusketeerHeroTurret`）。炮塔 HP717/DMG65@L3→lv11、寿命 10s 线性衰减（Building lifeTime 管线）、落地 AOE 95@L3（半径 2.0 [假设]）；攻速 0.8s [假设]；炮塔弹丸飞行段简化为直接结算（MusketeerTurret_Projectile 数值行缺失）。测试 M8-02。


## 3. MiniPekka (Heroic Mini P.E.K.K.A.)
- **普通版基准** (0x76c72638db40): HP 543, Damage 295, Range 800, (Speed Fast 90 由 Fandom 补)。
- **Hero 差异**: overlay 0x76c72648a5c0: `Ability=MiniPekka_hero_Ability`, 动作组 22 个 (`MiniPekka_hero_select_level_SubActions*`, `_on_level_up_actions`, `_level_up_effect_*`, `_heal`, `_run_timer_continuous`, `_on_hit_action`, `_special_cast_animation`)。Fandom: Hero 等级表扩至 **L21** (L17-21 即能力加级后的数值: 2539/2793/3072/3379/3717 HP)。
- **能力: Breakfast Boost (1 费, 单次)**: 立即吃掉已烹饪的煎饼, 升级 + 回复 30% HP。煎饼进度: 22s 自动填 1 格或每次攻击 +10s 进度, 最多 3 格; 加级 0 格=+1 / 1 格=+2 / 2 格=+3 / 3 格=+5 级 (等级 = 沿等级表跳级, HP/伤随表)。
- **引擎实现**: ① 被动计时器 (22s) + on-hit 进度累加器; ② 能力 = level-up 动作: 按 meter 数沿等级表跳跃 N 级 + heal 30% maxHP; ③ 等级表需扩展到 L21 (Fandom 全表已抓)。buff `MiniPekkaHero_buff_for_tag` 挂等级 tag。
- **完备度**: high。
- **实现状态 (M8)**: 已实现（`HeroMiniPekka`）。煎饼 22s/格 + 每击 +10s 进度（≤3 格）；能力 = 0/1/2/3 格 → +1/+2/+3/+5 级（引擎口径 = 官方 1.1/级曲线 ×级差, L17-21 表为跳级特例 [口径假设]）+ 回复 30% maxHP。测试 M8-03。


## 4. Valkyrie (Hero Valkyrie) — 觉醒+Hero 并存样本
- **普通版基准** (0x76c72641a5c0): HP 745, Damage 104, Range 1200, Speed 60。Fandom Hero L11: HP 1907 / AreaDmg 266 / AbilityDmg 97。
- **Hero 差异**: overlay 0x76c726418b80: `Base=CHARACTER.Valkyrie`, `Ability=ValkyrieHero_Ability`, DashDamage=0, DashLandingTime=100, DashToTargetRadius, LandingEffect/AttackStartEffect 复用 goldenknight dash vfx, `DisableMeleeAeoDamageEffect=true` (旋风期间关闭普攻 AOE 伤害)。
- **能力: Wild Whirlwind (3 费, 单次)**: 3.5s 旋风: 0.25s/击 (AttackChain), 半径 2.5, 自身移速提升, **减伤 15%** (ValkyrieHero_Damage_Reduction_Buff), 对皇冠塔伤害 -50%; 旋风结束位移冲刺 5.5 格 (Dash); 旋风后禁攻 (ValkyrieHero_Forbid_Attack_After_Whirlwind_Buff, AttackChain 计数)。
- **Wild slot 语义 (本卡为规格样本)**: 与觉醒 (EV1, Tornado) 同槽互斥二选一。觉醒装配 `Valkyrie_EV1` + tornado 组; Hero 装配 overlay + `ValkyrieHero_OnAbilityActivationGroup`。引擎侧 = deck slot 形态选择器, battle 开始时固化。
- **引擎实现**: 通道 buff (channelling): 期间攻击间隔 0.25s + 周期 AOE (EvoEffectZone 2.5 格, interval 0.25s) + damage_reduction buff + 移速 buff; 结束 → dash (复用 dash 管线: DashLandingTime=100ms, DashDamage=0) + Forbid_Attack buff。`ValkyrieHero_Charge`/`ValkyrieHero_Execute_Charge`/`ValkyrieHero_Charge_Target`/`ValkyrieHero_Set_Ability_Button_Pending` = 按钮 pending 状态机 (冠军同款)。
- **完备度**: medium (Fandom 页 stub, 能力正文缺; 属性表数字全)。
- **实现状态 (M8)**: 已实现（`HeroValkyrie`）。3.5s 旋风 0.25s/击 半径 2.5（AbilityDmg 97@L11, 塔伤 ×0.5）+ 减伤 15% + 移速 ×1.2 [假设]；结束冲刺 5.5 格（朝敌塔向, 可走性收缩兜底）；旋风后禁攻 1s [假设, 时长缺]；Wild slot 二选一经 hero_slots 声明实现。测试 M8-04。


## 5. Wizard (Hero Wizard)
- **普通版基准** (0x76c726414040): HP 295, Damage 295, Range 5500, Projectile=chr_wizardProjectile。Fandom Hero L11: HP 832 / Dmg 281 / TornadoDmg 43。
- **Hero 差异**: overlay 0x76c726494dc0/0x76c726496640 (中置信, 表混入符号条目): `Ability` 列含 WizardHero_*。buff: WizardHeroAbilityBuff, **WizardHero_MiniTornadoBuff**, HeroWizardNoMove。
- **能力: Fiery Flight (1~2 费, 单次; wiki 属性表 1 与信息框 2 矛盾, 双记)**: 1s 延迟后升空 5s: 移速 +50% (Medium→Fast), 火球命中处生成**半径 4 格、持续 2s 的火旋风** (独立伤害, 对皇冠塔减免), 对空对地。
- **引擎实现**: 飞行态 = is_flying 标记 (`Wizard_hero_set_is_flying_false` 结束组) + 移速 buff; 命中后 spawn tornado zone (半径 4, 2s, tick 伤害) — 复用 EvoEffectZone; `WizardHeroAbilityProjectile_spawn_tornado`/`_spawn_damage_aeo`/`_hit_target_action_group` 即命中分支; HeroWizardNoMove = 升空起跳/落地锁定帧。
- **数据缺口**: HeroSpell (WizardHeroSpell) 投掷物参数 missing — 火球参数沿用普通 Projectile + 旋风参数来自能力表。
- **完备度**: high。
- **实现状态 (M8)**: 已实现（`HeroWizard`）。1 费（双记取 1）；1s 延迟升空 5s（临时空中位面 + 移速 ×1.5）；攻击命中生成火旋风 EvoEffectZone（半径 4 / 2s / dps 43@L11, 对空对地）[近似: 旋风落点以攻击时刻目标位置代替弹道落点, 待 L4]。测试 M8-05。


## 6. Bowler (Hero Bowler)
- **普通版基准** (0x76c72640f6c0): HP 813, Range 4000, Projectile=BowlerProjectile (Damage 未解码; Fandom L11: HP 2081 / AreaDmg 289)。
- **Hero 差异**: 无独立 overlay 表 (配置在大型合并表); 动作组 19 个: `BowlerHero_activate_ability` / `_deactivate_ability` / `_select_siege_attack` / `_select_tower_in_range` / `_set_tower_in_range` / `_no_attack_while_casting` / `_spawn_ability_buff` / `_play_exclamation*` / `_play_rock_glow` / `BowlerHeroDeactivateGroup`。
- **能力: Stone Swish (2 费, 单次)**: 按下后 2.5s 蓄力 (播放感叹号+石头发光, 期间禁止攻击), 然后 7.3s 内切换为**远程迫击炮模式**: 射程 11.5 格, 共 3 发 (Hit Speed 1.9s), 弹丸射程 7 格 (2026-08-04 由 7.5 削)。能力伤害列 AbilityDmg (L11=508, 塔伤 254)。
- **引擎实现**: 模式切换 buff (BowlerHero_ability_buff, 7.3s): 攻击组件 swap (range 4→11.5, projectile→siege 版, 伤害→AbilityDmg); 蓄力 = disable attack + cast 动画; 结束自动 `deactivate_ability` 还原。塔选取逻辑 `_select_tower_in_range` = 蓄力前检测 11.5 格内有无法师塔。
- **数据缺口**: HeroSpell (BowlerHeroSpell) 参数 missing (即 siege 弹丸, 用 AbilityDmg 数值即可)。
- **完备度**: high。
- **实现状态 (M8)**: 已实现（`HeroBowler`）。2.5s 蓄力禁攻 → 7.3s 攻城模式（射程 11.5 / 3 发 / 攻速 1.9s / 伤害 508@L11 / 塔伤 ×0.5）, 弹尽或到期自动还原; 主伤害 289@L11 覆写弹丸槽; 弹丸 7 格飞行段简化为直接命中 [口径注]。测试 M8-06。


## 7. Giant (Hero Giant)
- **普通版基准** (0x76c726406ac0): HP 1550, Damage 99, Range 1200, Mass 18 (只攻建筑)。Fandom Hero L11: HP 3968 / Dmg 253 / ImpactDmg 135。
- **Hero 差异**: overlay 0x76c726487f40: `Base=CHARACTER.Giant`, `Ability=GiantHero_Ability`。动作组 17 个: `GiantHero_OnAbilityActivationGroup`, `GiantHero_Play_Slap_Animation_Left/Right`, `GiantHero_Slap_Apply_Stun`, `GiantHero_Slap_Active_Effect`, `GiantHero_Reenable_Ability`, `GiantHero_Set_Ability_Button_Pending`。buff: **GiantHero_Slap_Stun**。
- **能力: Heroic Hurl (2 费, 单次)**: 1s 延迟后抓取 2 格内**最高 HP 敌军部队** (1 个, 对空对地) 水平扔出 9 格, 落地造成 ImpactDmg 并**眩晕 2s**; 飞行中单位不可被地面单位/地震选中。
- **引擎实现**: 目标选择 = 半径 2 内 maxHP 部队; 效果 = 位移 (抛物线弹道, 9 格) + 落点 AOE (ImpactDmg) + stun buff 2s; 飞行中 = targetable-by-ground=false 标记。复用 knockback/位移管线 + stun。
- **完备度**: high。
- **实现状态 (M8)**: 已实现（`HeroGiant`）。抓 2 格内最高 HP 部队（对空对地）扔出 9 格（朝本方进攻方向, 落点不可走折半兜底）, 落地 ImpactDmg 135@L11（半径 1.5 [假设]）+ 眩晕 2s; 无目标不扣费返还; 飞行中段不可被地面选取未建模 [简化]。测试 M8-07。


## 8. Goblins (Hero Goblins)
- **普通版基准** (0x76c726489440, 行名 `Goblin`): HP 79, Damage 49, Range 500, x4。
- **Hero 差异**: 无独立 overlay 表; 动作组 29 个 (`GoblinHero_*`): Start_Group, Flag_Self_Destruct(_Conditional), Last_To_Fall_Group, Check_Last_Goblin_To_Fall, Spawn_Second_Wave_0/2, Flag_Limited_Time_Ability_Button, Reset_Ability_Charges, Enable_Champion_HP_Bar, Show_Disabled_Button 等。buff: GoblinHero_Last_To_Fall_Override_Effect_Buff。Hero 形态有 Champion 式 HP 条 (`Enable_Champion_HP_Bar`) 与部署监听 (`Listen_To_New_Deploy`)。
- **能力: Banner Brigade (1 费, 单次, 条件窗)**: 部署后按钮禁用; **最后一只哥布林死亡时**落旗 (持续 5s, Override_Effect), 窗口内按按钮 → 在后方召出 **2 只 Brigade Goblins** (属性与本体相同; 2026-08-04 由 x3 削为 x2)。Fandom Hero 主体 L1: HP 78 / Dmg 48 (x4, 与普通版几乎一致 — Hero 数值≈普通)。
- **引擎实现**: ① 组死亡监听 (count==1→last_to_fall); ② spawn 旗帜 dummy (5s 寿命, vfx) + 开窗 enable button; ③ 按下 → spawn 2 goblins (复用 SpawnCharacter 管线, GoblinHero_Spawn_Second_Wave_*); 未按键窗口结束 → 旗消失。GoblinHero_Flag_Self_Destruct_Conditional 提示旗帜/增援有自毁条件 (超时)。
- **完备度**: high。
- **实现状态 (M8)**: 已实现（`HeroGoblins` + `BattleState.hero_windows` 条件窗）。x4 同组监听（2s 内同批部署归组）→ 全灭落旗开 5s 窗 → 窗内按按钮（1 费）出 2 只 Brigade Goblins（本体同数值）; 增援 x2（2026-08-04）; 分批部署（0.2s 间隔）未落地成员不计入「最后一只阵亡」; 单次使用, 过期窗自动关闭。测试 M8-08。


## 9. MegaMinion (Hero Mega Minion)
- **普通版基准** (0x76c72640ec40): HP 327, Range 1600, Speed 45 (Damage 未解码; Fandom Hero L11: HP 837 / Dmg 312 / WarpDmg 399)。注意 Fandom Hero 主表 Range 写 Melee:Long(1.6) — 与普通版远程 Spit 不同, **Hero 版近战化**。
- **Hero 差异**: overlay 0x76c72655c780 (高置信): `Ability=MegaMinion_Teleport_Ability`, `Projectile=MegaMinionHeroProjectile`, `Damage=330` (内存 lv 基准; 与 Fandom WarpDmg 399@L11 不同量级, 双记), `ClonedVersion` 列。动作组 13 个: `_hero_ability_action`, `_hero_on_teleport_group`, `_hero_spawn_buff_for_bots`, `_hero_hide/disable/enable_ability_button`, `_hero_set_override_button`, `_hero_CrownTower_Buff` 相关。buff 3 个: MegaMinion_hero_Damage_Buff, MegaMinion_hero_CrownTower_Buff, MegaMinionHeroBuffForBots。
- **能力: Wounding Warp (2 费, 单次)**: 部署时被动标记**最低 HP 敌人** (目标死亡标记转移); 按钮瞬移到标记处造成 WarpDmg; 之后**永久**对皇冠塔伤害 -75% (2026-08-04: x0.25 且永久)。
- **引擎实现**: ① 部署 hook: 选敌方最低 HP 部队 → 挂标记 (视觉); ② 按钮: teleport + 落点伤害; ③ 落地后挂 CrownTower_Buff (multiplier 0.25, 永久)。
- **完备度**: high。
- **实现状态 (M8)**: 已实现（`HeroMegaMinion`）。部署被动标记最低 HP 敌军（目标死亡经 on_tick 转移）→ 按钮（2 费）瞬移标记处 + WarpDmg 399@L11（半径 1.5 [假设]）+ 永久皇冠塔伤害 ×0.25; Hero 近战化 → 伤害覆写走直接槽。测试 M8-09。


## 10. IceWizard (Hero Ice Wizard) — **数据缺失卡**
- **普通版基准** (0x76c726488800): HP 269, Range 5500, Projectile=ice_wizardProjectile。
- **Hero 侧内存证据 (结构完整, 数值缺)**: 动作组 54 个 (`IceWizardHero_*`, 含拼写变体 `IceWizarHero_set_ability_from_kill_true/false`): `ability_activation_actions`, `spawn_cube_character`, `spawn_cube_run_on_instigator_true/false`, `cube_lifetime_timer`, `cube_wait_to_kill`, `ability_kill_on_self`, `ability_from_kill_actions`, `spawn_freeze`, `FreezeAeo_starting_actions`, `FreezeAeo_inform_context_should_kill`, `attached_*` 系列 (attached_reapper/attached_play_animation/attached_wait_to_reapper = 冰封后复现), `IceWizardAOE`。buff 4 个: IceWizardHeroSlow, IceWizardHero_FreezeBuff, IceWizardHero_FreezeBuff_dummy, IceWizardHero_SlowBuff。prefab: **hero_prefab_icewizard_snowman_battle** (雪人形态)。
- **结构推断 (来自动作组名, 需实测校验)**: 击杀敌人时自身变为冰块 (cube) 进入 attached 隐藏态, 计时后或冰块被摧毁时复现 (reapper), 冰块破碎触发冻结 AOE; 能力含 freeze/slow 双轨。
- **Fandom**: 页面仅 "Coming soon..." — 无数值。**禁止编造**: 数值全部 missing。
- **完备度**: missing (机制结构 high, 数值 missing)。
- **实现状态 (M8)**: 机制先行已实现（`HeroIceWizard`）。冰封自身（不可选取+无敌+移速归零+禁攻, 不用 freeze_timer 以免 on_tick 饥饿）3s [假设-待实测] → 破碎冻结 AOE：冻结 2s [暂借 Freeze 语义] / 半径 3 [暂借] / 减速 30% 2s [暂借 Ice Golem/Hero]; 费用 2 [假设]; kill/from_kill 联动分支未建模（参数全缺, 待实测）。数值【暂借/待实测】。测试 M8-10。


## 11. Tombstone (Hero Tombstone / Tomb Queen)
- **普通版基准** (0x76c726488d40): 建筑, Spawn Speed 4s, Lifetime 30s, 产骷髅 (HP 207 为骷髅值)。
- **Hero 侧内存证据**: **TID_HERO_TOMBSTONE_MONSTER** 表 0x76c72648e140 (高置信): Hitpoints 1650, Damage 165, Range 1600, Speed 60 — 即 **Tomb Queen** 本体; `TombstoneHeroSkeleton` (SpawnData, 0x76c72640c080); overlay/dummy 表 0x76c72648f640 (Ability=Tombstone_hero_Ability), 0x76c726492040, 0x76c72655c5c0 (OnDeathAction/OnStartingAction, DeathSpawnCharacter×2); prefab: hero_prefab_tombstone_battle_dummy + **hero_prefab_tombstone_skeletonqueen_battle**。动作组 63 个: `Tombstone_hero_Ability_Timer`, `_Ability_Timer_Ended(_group)`, `_OnAbilityActivationGroup`, `_Monster_*` (SpawnMonster/HideMonster/ShowChampionBadge/StandStillBuff/NoAttacksBuff), `_switch_dummy_to_broken_tomb`, `_play_skeletonqueen_deploy_animation`, `_tombstone_check_death`, `_death_reset`, `_kill_group`。buff: Tombstone_hero_Monster_StandStillBuff / NoAttacksBuff。Fandom Hero L11: QueenHP 4224 / QueenDmg 422 (2026-08-04: Damage +32%, sight 5.5→7, HP +4%; 2026-07-06: HP +20%, 移除持续产骷髅)。
- **能力: Regal Revive (5 费, 单次)**: 墓碑破碎 → **Tomb Queen** 从墓中升起 (SkeletonQueen 形态), 只攻建筑 (sight 7 格), Queen 有计时器 (`Ability_Timer`), 计时结束/死亡 → 墓碑死亡复位 (`death_reset`)。
- **引擎实现**: 能力 = 建筑死亡/激活时 spawn Queen 单位 (Target=Buildings, HP/Damage 按等级表, Range 1.6 格/sight 7); Monster_* buff 控制女王站定/禁攻阶段; dummy→broken tomb 形态切换 = 渲染层。
- **数据缺口**: 能力正文 missing (Fandom stub); Queen 数值双源一致量级 (1650@内存 lv 基准 vs 4224@L11)。
- **完备度**: medium。
- **实现状态 (M8)**: 已实现（`HeroTombstone`）。5 费预付武装 → 墓碑破碎 spawn TombQueen（派生角色: HP4224/Dmg422@L11 [Fandom, 内存 1650 双记], 只攻建筑 sight 7 [2026-08-04], 攻速 1.5s [假设]）; Hero 形态移除持续产骷髅（2026-07-06）; Queen 计时器 15s [假设-待实测]（复用 `_evo_temp_lifetime` 临时寿命管线）。测试 M8-11。


## 12. Berserker (Hero Berserker)
- **普通版基准** (0x76c7264890c0, Name 列未解出, 靠 TID_SPELL_BERSERKER+PrestigeSWF 定位): HP 350, Range 800, Speed 90 (Damage 未解码; Fandom L1: 345/39)。
- **Hero 侧内存证据**: overlay 0x76c726573e00/0x76c726574c00: `Base=CHARACTER.Berserker`。动作组 7 个: `BerserkerHero_OnStartingAction`, `_BearForm_OnStartingAction`, `_change_to_bearform`, `_change_back_from_bearform`, `_ability_group`, `_effect`, `_spawn_buff`。buff: BerserkerHero_buff。另有普通觉醒动作 `Berserker_crazy_2_change_to_bearform` (熊形态概念与觉醒共享)。
- **能力: Savage Survival (3 费, 单次)**: 熊灵附体 4s: 攻速 0.2s (Hit Speed), 移速 Ultra Fast (135), **HP 不低于 1** (minimum hitpoints), 对皇冠塔伤害 -75%, BearDmg (L11=167)。持续结束 → change_back_from_bearform 还原。
- **引擎实现**: 变形 buff: attack_interval override + speed override + crown-tower multiplier + **HP floor=1** (伤害结算 clamp, 不死锁); 计时 4s 后还原。复用 crazy_2 bearform 变形组 (同引擎机制)。
- **完备度**: medium (正文缺, 属性表全)。
- **实现状态 (M8)**: 已实现（`HeroBerserker`）。熊灵 4s: 攻速 0.2s / 移速 UltraFast（135→2.7 格/s）/ 塔伤 ×0.25 / BearDmg 167@L11; HP 下限 = 1 经 `on_take_damage` 受击钩子致死钳制（含法术/AOE, source=None 也触发）; 结束 change_back 还原。测试 M8-12。


## 13. DarkPrince (Hero Dark Prince)
- **普通版基准** (0x76c726388040, 高噪声: 与爆炸物行合并 — HP 469/Mass 6/Range 1200 可信, Damage 25/Speed 550 为污染值, 弃用)。
- **Hero 侧内存证据**: overlay 0x76c72640a800 (59 列) + 0x76c726594cc0: `Ability=DarkPrinceHero_Ability`, `ChargeEffect`, `DamageEffectSpecial`, PrefabAsset=**hero_prefab_darkprince_mounted_battle** (骑犀牛)。动作组 17 个: `DarkPrinceHero_Ability_Activation_Group`, `_Ability_Spawn_Mount`, `_Ability_WarpBack(_Group)`, `_Ability_IncrementWarpTime`, `_Ability_Change_To_Walking`, `_Ability_TransformVFX`, `_Mount_Spawn_Projectile_AOE`, `_Mount_NoMove`, `_DisableAbility`, `_Reset_Ability_Button`, `_Mount_ImmediateCharge`, `_Ability_Activation_Group`。
- **能力: Destructive Dismount (3 费, 单次)**: 按钮下马: 本体跳跃落地造成 spawn 伤害 (溅射), 犀牛 (Rhino, 独立单位: L11 HP 1356 / Dmg 179 / ChargeDmg 358) 冲向建筑; 本体徒步溅射攻击但失去冲锋。另有 WarpBack 组 (早期版本传送回坐骑? 现行为分离)。
- **引擎实现**: 能力 = 分裂 spawn: ① 本体切换 walking 形态 (Change_To_Walking, TransformVFX) + 落地 AOE; ② spawn Mount 单位 (Target=Buildings, charge 行为复用 Prince charge); 两者独立存活。`Mount_NoMove`/`IncrementWarpTime` 为旧机制残留。
- **数据缺口**: 内存 Damage/Speed 污染 — 全数值以 Fandom 为准 (L6-16 全表已抓)。
- **完备度**: high。
- **实现状态 (M8)**: 已实现（`HeroDarkPrince` + `DarkPrinceHeroRhino` 复用 Prince 冲锋管线）。下马: 本体落地溅射（伤害=普攻 [假设, 数值缺], 半径 1.2）+ 失去冲锋（charge_range=0）+ 徒步溅射普攻; 犀牛独立冲塔（HP1356/Dmg179/ChargeDmg358@L11 [Fandom], chargeRange 3.5 [假设同 Prince], 攻速 1.4s [假设]）。测试 M8-13。


## 14. Balloon (Hero Balloon)
- **普通版基准** (0x76c72641ae80): HP 655, Damage 250, Range 100 (只攻建筑), 空中。
- **Hero 侧内存证据**: 配置在合并表 0x76c72653cc00 / 0x76c72657a8c0: `Ability=BalloonHero_Ability`, `DeathSpawnCharacter=BalloonHero_Bomb`。prefab: hero_prefab_balloon_battle + **hero_prefab_balloon_skydiver_battle** (0x76c72648c700, 伞兵形态)。动作组 13 个: `BalloonHero_Ability_Activation_Group`, `_Ability_Target_Finder/_Target_Seeker/_Target_Show_Effect`, `_Ability_Play_Spawn_Skeletrooper_Animation`, `_Ability_Spawn_Skeletrooper_Effect`, `_Skeletrooper_Remove_Target_Indicator`, `_Skeletrooper_Speed_Up_Group`, `_Failsafe_Spawn_Delayer`。
- **能力: Coffin Cadets (2 费, 单次)**: 召出**骷髅伞兵 (Skeletrooper)** 飞向 6 格内最近地面敌人, 造成落地伤害 (LandingDmg, L11=263, 对塔 -90%) 并驻场攻击 (L11: HP 473 / Dmg 204, Hit Speed 1.1s, Very Fast 120, Range 6.5, 对地); Balloon 本体死亡时 DeathSpawn=BalloonHero_Bomb (3 格死亡伤害, 3s 部署, 对空对地)。
- **引擎实现**: 能力 = 目标搜索 (最近地面, 6 格) → spawn 投放单位 (skydiver 弹道 → 落地 AOE → 转常规攻击); `_Skeletrooper_Speed_Up_Group` 与 `_Failsafe_Spawn_Delayer` 提示有加速分支与兜底延迟生成。
- **完备度**: high。
- **实现状态 (M8)**: 已实现（`HeroBalloon` + 派生角色 `Skeletrooper`）。伞兵飞向 6 格内最近地面敌人（伞降段不可选取+无敌; 落地阈值 = 碰撞半径和 + 0.3; 3s 兜底原地落地, 对应官方 `_Failsafe_Spawn_Delayer` 分支）→ 落地 AOE 263@L11（对塔 ×0.1, 半径 1.5 [假设]）→ 驻场攻击（HP473/Dmg204@L11/攻速 1.1s/对地）; 本体死亡炸弹继承基础 Balloon 亡语。测试 M8-14。


## 15. BarbarianBarrel (Hero Barbarian Barrel) — 唯一 spell Hero
- **普通版基准** (0x76c72648ff00): 卡本体 = 滚桶投射物角色 `BarbLogProjectileRolling` (无 HP/Damage 列, Speed 200), 死亡 spawn Barbarian。
- **Hero 侧内存证据**: 动作组 15 个 (`BarbLogHero_*` / `BarbLog_hero_*`): `_spawn_reroll`, `_on_rerroll_start_actions`, `_on_reroll_end_actions`, `heal`, `_override_ability_button_when_rolling`, `_reset_path`, `_reset_target`, `_select_modifier`, `_set_origin_modifier`, `_set_pro_modifier`, `_set_another_deploy_true`, `_interval_check_ability_played`, `_starting_actions`, `_Listen_To_New_Deploy`。特效: `ps_barbarian_barrel_hero_rolling_smokes`, `ps_barbarian_barrel_hero_explosion_02`, `barblog_hero_set_re_deploy_animation`。
- **能力: Rowdy Reroll (1 费, 单次)**: 桶**再滚一次** (重置路径 reset_path → 第二次全车道滚动, 伤害重新结算), 并**治疗野蛮人 = 桶伤害的 50%** (heal 组); Fandom: Range 3 (2026-05-04 由 4 削), Width 2.6。
- **引擎实现**: 能力 = 投射物重走 (reset path + 重置已命中列表) + 命中终点 heal (0.5×damage) 于 spawned Barbarian; 按钮状态机在 rolling 期间 override。
- **完备度**: high。
- **实现状态 (M8)**: 已实现（条件窗, 无实体 holder）。部署（法术弹滚出）即开 10s 按钮窗 [假设] → 窗内按按钮（1 费单次）: 二次全车道滚（重新生成 `BarbLogProjectileRolling`, 伤害重新结算 = reset path 语义）+ 治疗落点 6 格内野蛮人 = 桶伤害 50%（lv11 ≈75.5/s × 1s）[治疗范围 6 格为假设]。测试 M8-15。


## 16. EliteArcher (Hero Elite Archer) — **wiki 缺失卡**
- **普通版基准** (0x76c7263f52c0): HP 207, Range 3500/7000 双值 (噪声), Speed 60, Projectile=EliteArcherArrow。
- **Hero 侧内存证据**: **TID_CHARACTER_HERO_ELITE_ARCHER_DUMMY** 表 0x76c72648ce00: HP 104, Range 5500, Speed 40, DeathSpawn=Event_VoodooHog (列混排, 中置信); 表 0x76c7264381c0 (`EliteArcherHero`, `EliteArcherHero_Dummy`, `OnStartingAction=EliteArcherHero_Dummy_Start_Group`)。动作组 17 个: `EliteArcherHero_Ability_Activation_Group`, `_Ability_Action`, `_Ability_Warp`, `_Ability_Effect_Warp`, `_Ability_Warp_Done_Group`, `_Ability_Spawn_Buff`, `_Ability_Spawn_Dummy`, `_Triple_Shot_Action`, `_Set_Attack_Sequence_1`, `_Dummy_Start_Group`, `_Dummy_Hit_Animation`。buff: EliteArcherHero_Ability_Buff。
- **结构推断 (动作组名)**: 能力 = 瞬移 (Warp) + 三连射 (Triple Shot) + buff + dummy 分身; 与 SuperEliteArcher (K2 超级卡) 的 Triple Shot 同源概念。
- **Fandom**: 无页面 (基础页都不存在) — **数值全 missing, 禁止编造**。
- **完备度**: missing (机制结构 medium, 数值 missing)。
- **实现状态 (M8)**: 机制先行已实现（`HeroEliteArcher` + 派生角色 `EliteArcherHeroDummy`）。Warp 向最近敌人瞬移 3 格 [假设-暂借] + 三连射（攻速 0.3s [暂借-待实测] × 3 发 → 打完还原）+ 假人分身（HP104 [内存 Dummy 表·中置信] → lv11, 寿命 5s [假设]）; 费用 2 [假设]。数值【暂借/待实测】。测试 M8-16。


## 17. IceGolemite (Hero Ice Golemite) — **wiki 缺失卡 (Ice Golem/Hero 为最可能对应)**
- **普通版基准** (0x76c7263f5640): HP 514, Damage 33, Range 750 (只攻建筑), Speed 45。
- **Hero 侧内存证据**: buff 10 个, 命名揭示完整机制: **IceGolemiteHero_Freeze_Buff_{Base,Small,Medium,Large,Tower}** + **IceGolemiteHero_Slow_Buff_{Base,Small,Medium,Large,Tower}** — 按**目标体型分档** (Base/小/中/大/塔) 的冰冻+减速双轨。动作组 13 个: `IceGolemiteHero_OnAbilityActivationGroup`, `_Spawn_Damage_AEO`, `_Select_Freeze_Buff_SubActions0/1/2`, `_Select_Slow_Buff_SubActions0/1/2`。特效: `ps_icegolem_hero_ability_buff_snow_tower_behind`, `IceGolem_hero_slow_snow/_slow_puff`, `ps_hero_deploy_ice_aura`。
- **结构推断**: 能力 = 冰雪光环 AOE (Deploy_ice_aura) → 按目标档位施加 Freeze 或 Slow buff (SubActions0/1/2 = 档位分支)。Ice Golem/Hero (wiki): Snowstorm — 半径 4 格光环, 3 次脉冲, 减速 30% 2s, 脉冲伤害 (L11=69×3); 第 3 脉冲原 freeze 已改 slowdown。
- **引擎实现**: EvoEffectZone (半径 4, pulses=3, interval≈1s) + 命中时按目标 mass/体型选 buff 档位; freeze 轨仅在合规档位生效。
- **数据缺口**: Ice_Golemite 卡页与 Hero 页均无 — 若官方 Ice Golemite Hero 与 Ice Golem Hero 参数不同, 数值 missing。
- **完备度**: missing (机制结构 high, 数值借用 Ice Golem/Hero 需实测校验)。
- **实现状态 (M8)**: 机制先行已实现（`HeroIceGolemite` + `battle.IceGolemiteSnowZone`）。冰雪光环: 半径 4 / 3 次脉冲 / 间隔 1s / 脉冲伤害 69@L11 / 减速 30% 2s [全部暂借 Ice Golem/Hero]; 分档规则 [假设]: 部队 collisionRadius ≤ 0.5 → 冻结 1s, 其余部队 → 减速, 建筑/塔 → 仅减速（内存 buff 命名 Base/Small/Medium/Large/Tower 参数未解出）; 费用 2 [假设]。数值【暂借/待实测】。测试 M8-17。


---


## 附: 来源统计与汇总

- **内存来源** (17/17): 动作组符号注册表 0x76c726499ac0 (351 个 Hero 组名), buff 注册表 0x76c726404600 (34 个 Hero buff 键), hero overlay 表 (Knight/Musketeer/Valkyrie/Giant/MiniPekka/DarkPrince/MusketeerTurret/Berserker/MegaMinion 9 张高置信), Hero 专属单位表 (TombQueen/EliteArcherDummy/TombstoneHeroSkeleton), pool hero 块 (5 个 HeroSpell 名), globals HERO_FORM_POOLS/HEROES_ENABLED, meta 字符串 (hero_coin/hero_form/K2EliteCardTrial)。
- **Fandom 来源** (14/17 有效页): `_elite_<卡名>.txt` ×17 + `_elite_heroes_category.txt` + `_elite_card_evolution.txt`; 报告 `_elite_fandom_report.md`。
- **完备度**: high ×11 (Knight, Musketeer, MiniPekka, Wizard, Bowler, Giant, Goblins, MegaMinion, DarkPrince, Balloon, BarbarianBarrel) · medium ×3 (Valkyrie, Tombstone, Berserker) · missing 数值 ×3 (IceWizard, EliteArcher, IceGolemite — 三者内存侧机制结构均可用于先行实现, 数值待补)。
- **未解之谜 (后续攻关)**: ① 动作序列化流全解码 (记录头 `03 01 03` + 池字符串指针三元组, 可还原全部动作参数 → 数值不再依赖 Fandom); ② HitSpeed/部分 Damage 列的替代链路; ③ HeroSpell 投掷物参数。
