# 勘误批 2-13 实施汇总（2026-09-04）

> 流程：12 个采集子代理走 CDP 抓 Fandom（报告 re/fandom_check/batch02..13.md）→ 用户裁决三条原则：
> ①数值以游戏快照为权威，Fandom 只作机制参考；②可复用机制做通用组件，独有机制单做；③全量修复。
> 回归红线：test_m2 72 / m3 62 / m4 51 / m5 40 / m6 84 + batch_smoke 212 全绿。

## 一、通用组件（一次覆盖多卡）

| 组件 | 覆盖卡 |
|---|---|
| 通用亡语 deathSpawnCharacterData（Entity._generic_death_spawn，含炸弹型→TimedExplosive、单位型×deathSpawnCount、跳过专属 on_death/觉醒同名） | GoblinCage 基础斗士、BombTower 亡语炸弹、DarkWitch 亡语蝠、BarbarianHut/GoblinHut/GoblinPartyHut、SpearGoblinGiant→SpearGoblin、SkeletonBalloon（容器 0.6s→7 骷髅） |
| ElixirGolem 三级分裂链 + 死亡圣水馈赠（1/0.5/0.25，Fandom 机制【待对拍】） | ElixirGolem/ElixirGolem2/ElixirGolem4（快照 scd damage=None 走投射物/回退） |
| 双部队部署 summonCharacterSecondData（第二波 +0.2s【假设】） | GoblinGang 3 枪哥布林、Rascals 2 女兵、Goblinstein 医生 |
| 部队通用周期出兵（首波 1s【官方 Night Witch】，后续 spawnPauseTime） | Night Witch 2 蝠/5s |
| 法术 impact 击退（pushback 全字段消费） | Fireball、Rocket（此前仅 Log rolling） |
| 派生角色 damage/speed 回退同族基础卡 | 觉醒斗士 0 伤 0 速修复 |
| 塔兵=卡组第 9 张（PlayerState.set_tower_troop → 塔构建替换） | King_CannonTowers / King_KnifeTowers / King_ChefTowers 数值随塔兵装配【Duchess 蓄能节奏、Chef 做饼未建模】 |

## 二、法术层修复

| 法术 | 修复 |
|---|---|
| Lightning | 整卡从无行为隐形实体 → 半径 3.5 内最高 HP 至多 3 目标、伤害+0.5s 眩晕+重索敌、对塔 ×0.65【Fandom】 |
| Freeze | 伤害改施法单次（原每 0.5s 重复 ~8 次超发）；裁决包确认 |
| Earthquake | building_damage_percent 350（×4.5）+ 减速 -50% 消费 + 对塔降伤（bd 回退） |
| Poison | 对塔 ×0.3（bd -70）+ 减速 -15% |
| Rage | **修复目标反转**（原对己方每 0.3s 脉冲 179！）→ 光环纯增益 + 施法对敌一次性 179@lv11【Fandom, 待对拍】 |
| AreaEffect 对塔降伤 | 覆盖公主塔（原只对 'King' 名生效——Fireball/Freeze 等全部受益） |
| Mirror | 镜像卡等级 +1（官方现行；出兵窗口统一提级） |

## 三、专项机制类

| 卡 | 实现 |
|---|---|
| ElectroGiant | Zap Pack 反射：被 2 格内敌部队伤害 → 反射 75+0.5s 眩晕；冰冻不反射【一击多段计 1 次简化】 |
| ElectroWizard | 部署 Zap（3.0/75@基准/0.5s）+ 攻击眩晕（buffOnDamageData 首次消费） |
| MiniSparkys (Zappies) | 攻击 0.5s 眩晕 |
| ElectroSpirit | 链电（投射物行 39@lv1 基准缩放）+ 命中后自毁 |
| FireSpirits | 自毁走核心 kamikaze 管线（验证通过） |
| Phoenix | 死亡 → 亡语火球（64/2.5 格/击退）+ 产蛋（4.3s【Fandom 现行】）→ 满血再生体（2025/11 取消 80%）；再生体不再产蛋 |
| MovingCannon | HP≤50% 变身 BrokenCannon（定身建筑态、保留当前 HP、寿命 15s【Fandom】） |
| RamRider | 冲锋继承 Prince 管线 + 缠网 -70%/2s【Fandom；骑手独立攻击未建模】 |
| ThreeMusketeers | 近战形态动态切换（敌 3 格内→1.2 射程，远离→6.0；_range_override 管线）【仅地面过滤简化】 |
| GoblinGiant | 背载枪哥布林独立投掷（SpearGoblinProjectile 32@lv1 缩放；快照 scd damage=None）+ 死亡落地 SpearGoblin |
| Fisherman | 钩建筑把自己拉过去（原显式跳过） |
| Prince | 冲锋伤害随级缩放（原恒 lv1 基准：306→782@lv11） |
| InfernoTower/ Dragon | 眩晕重置蓄能 + 破盾重置蓄能（官方 2016 确认） |
| SkeletonKing | 克隆体/亡语衍生/召唤物不计魂（_soul_excluded 标记） |
| VoodooHog/WitchMother | 诅咒 5s 过期【Fandom】 |
| Miner | 全场部署（含敌半场）+ 对塔 25%（tower_damage_mult 覆盖公主塔） |
| Elixir Collector | 产水（1/12s，冻结期暂停【官方 2016】）+ 死亡返还 1 |

## 四、查证结论（裁决包/深挖包）

- Fisherman 钩中减速：**已移除**（2026/4/6 平衡）——不实现正确
- Freeze 伤害单次；冻结停 Collector 产水 + 重置 Inferno 蓄能（均已实现）
- ZapMachine = **Sparky**（6 费，id 26000033；自身攻击无 stun）
- DartBarrell = **Flying Machine**（非 Dart Goblin；Dart Goblin = BlowdartGoblin，已有觉醒钩子）
- DarkWitch = **Night Witch**；WitchMother = Mother Witch
- Skeleton Dragons 快照 1.9s/0.8 半径为旧值（官方现行 2.0s/1.5）——按用户原则保留快照
- Graveyard 快照驱动 14 vs 官方现行 12——保留快照【对拍时仲裁】
- PhoenixEgg/PhoenixFireball/BrokenCannon/ElixirGolem2/4/RascalGirl/Goblinstein_doctor/Container 已注册派生卡

## 四½、塔兵机制（第二批用户裁决落实，2026-09-04）

| 塔兵 | 实现 |
|---|---|
| Dagger Duchess | 8 飞刀；有刀 0.45s 连发（快照攻速）每次消耗 1 支；回充从消耗第一支起计时每 0.9s +1（独立计时器）；空刀期停攻 |
| Royal Chef | 烹饪拟合：基础 23s（首饼 7s），烹饪期每次攻击延长 Δ=(38-23)×攻速/38（持续攻击 38s 恰好拉满）；+1 级=治疗 10% max（可叠/可超上限【简化】）+伤害 ×1.1（_damage_mult 载体，core.on_attack 消费）；只喂 >33% 血友军部队，优先未喂过的最高血量；双塔失停 Cook、单塔失 ×2 慢【假设系数】 |
| RamRider / Tornado | 维持简化（用户认可） |

## 五、遗留（标注简化/待 L4）

1. Duchess 8 发蓄能节奏、Chef 做饼喂兵——塔兵机制深度建模
2. ~~RamRider 骑手独立攻击~~（用户认可简化）
3. Tornado 质量抵抗（现收敛模型）
4. Mirror 克隆体不可再克隆标记
5. 快照 vs 官方现行数值分歧清单（SkeletonDragons 攻速、Graveyard 数量、DartGoblin 费用等）→ 留待 Nulls 回放对拍仲裁
