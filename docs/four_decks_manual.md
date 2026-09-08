# 四种卡组使用手册（2026-09-07 定稿）

> 用途：为 solo 训练提供四个互补 archetype 的固定卡组对手池，并作为后续脚本对手
> （SelfDefenderPolicy / 专项奖励）与取证的"卡牌语义参照"。
>
> 数值口径：**本仓库引擎 lv11 数值**（`cards_stats_characters/projectile/spell/building.json`，
> 即真实游戏快照），与 FirstLight CR `card_specs.json.gz`（native 提取，25 级基准）**趋势一致、
> 绝对值不同**——引用本手册数值时一律以本表为准。
>
> 选型依据：对齐 FirstLight CR 训练历程（`E:/FirstLight_CR/docs/training_history.md`）的
> archetype 覆盖——它用 2.6 速猪专精打到名人堂；我们取四个互补原型：速攻循环 / 推进 /
> 自闭攻城 / 双线快攻，全部卡牌在引擎 batch smoke 通过（`docs/card_coverage.md` 122 implemented）。

---

## 卡组总览

| # | 卡组 | archetype | 8 卡 | 均费 | 一句话打法 |
|---|------|-----------|------|------|-----------|
| 1 | 速猪 2.6 | 速攻循环 | HogRider, IceGolemite, IceSpirits, Musketeer, Cannon, Skeletons, Fireball, Log | 2.62 | 野猪快攻+建筑拉扯，4 张 1-2 费保循环速度 |
| 2 | 石头人 | 重坦克推进 | Golem, Witch, BabyDragon, MegaMinion, Bats, Tombstone, Zap, Snowball | 3.62 | 石头人吸收伤害，Witch/龙群输出，小费法术护航 |
| 3 | X 弩 | 自闭攻城 | Xbow, Tesla, Skeletons, IceWizard, Archer, Knight, Log, Fireball | 3.25 | X 弩桥头压制，建筑+群杂保弩 |
| 4 | 巨骷髅攻城槌 | 攻城组合 | BarbLog, BattleRam, GiantSkeleton, WitchMother, Ghost, Vines, Wizard, MiniSparkys | 3.88 | 攻城槌开路，巨骷髅亡语炸弹清防守 |

> 2026-09-08 用户换血：双线快攻 → 巨骷髅攻城槌（用户指定卡单，id 对齐后落地）；
> 皇家巨人 → 石头人（引擎无 NightWitch/Lumberjack，用 Witch 对位配平）。

四个卡组两两镜像可打：速 vs 推、推 vs 自闭、自闭 vs 速…… 每一副都含
**1 张坦克/核心 + 1-2 张远程 + 1 张小法术 + 1 张大法术或第二核心**，保证闸门
（不裸下硬闸门、法术空砸硬闸门、坦克后屯兵几何门）都有合法行为面。

---

## 1️⃣ 速猪 2.6（Hog Cycle）—— 均费 2.62

**打法核心**：野猪骑直奔塔，防守靠 Cannon 拉扯 + 群杂围杀，循环极快（1 费×2 + 2 费×1），
永远有牌可出。FirstLight 的 Hog 2.6 specialist 就是这套结构（Evolved Cannon/Skeletons、
Musketeer、Hog、Ice Golem、Ice Spirit、Fireball、Log），是我们**最能对表专精路线的卡组**。

| 卡 | 费 | 类型 | 数值（引擎） | 战术角色 |
|----|----|------|--------------|----------|
| **HogRider** | 4 | 地面单体 | hp 1696, dmg 150/1.6s, 速度 2.4, 跳河 | **核心输出**。只打建筑；桥头下、贴塔输出。被建筑拉扯是最大克星，会绕 Cannon 跑半图 |
| **IceGolemite** | 2 | 地面坦克 | hp 1197, dmg 40/2.5s, 速度 0.45 | **坦克+减速**。死亡冰冻；顶在野猪前当肉盾（先猪后人会吃满塔伤）。≈FirstLight 速猪里的 IceGolem 位（本引擎无 IceGolem，IceGolemite 同位替代） |
| **IceSpirits** | 1 | 空地单体 | hp 230, 冻结 dps, 速度 2.4 | **循环+控场**。1 费冻结敌人 0.5s，给野猪多 1-2 次出手窗口；也可防守时冻近战 |
| **Musketeer** | 4 | 空地远程 | hp 721, dmg 103/1.0s, 射程 6.0 | **主力输出/防空**。放塔后，掩护野猪推进时清后排+解空中 |
| **Cannon** | 3 | 建筑 | hp 824, dmg 83/0.9s, 射程 5.5 | **拉扯核心**。放中场（河后 3-4 格）把野猪/巨人从塔边拉开；对地不能打空 |
| **Skeletons** | 1 | 地面×3 | hp 81×3, dmg 81/s/个 | **围杀+拉扯**。围死 MiniPekka/王子；1 费骗法术、刷循环 |
| **Fireball** | 4 | 法术 | dmg 325 半径 2.5, 塔伤 70% | **中法术**。解女巫/法师/火枪手中坚；斩杀残血塔（325×0.7≈227） |
| **Log** | 2 | 法术 | dmg 240 滚动, 塔伤 80%, 击退 | **小法术**。滚地清群杂（骷髅/哥布林/蝙蝠全灭），击退推迟敌推进 1-2s |

**克制关系**：怕空中流（Cannon/HogRider/IceGolemite 全不能打空，只剩 Muskie/Fireball 对空）；
不怕地面大费推进（Cannon 拉扯 + Skeletons 围杀换费赚）。

---

## 2️⃣ 石头人（Golem Beatdown）—— 均费 3.62

**打法核心**：石头人顶在前面吸收伤害，Witch/飞龙在后面持续输出，蝙蝠和骷髅做低成本
循环。石头人死后的 Golemite 分裂延续压力。经典的"前面扛、后面打"重推进结构。

| 卡 | 费 | 类型 | 数值（引擎） | 战术角色 |
|----|----|------|--------------|----------|
| **Golem** | 8 | 地面坦克 | hp 5120, dmg 312/1.5s | **核心**。死亡分裂 Golemite（双倍期更肉）；沉底攒费后的推进核心 |
| **Witch** | 5 | 空地对空 | hp 838, dmg 133/1.4s, 射程 5.5 | **后排输出+清杂**。持续召骷髅当肉盾，掩护 Golem 推进 |
| **BabyDragon** | 4 | 空中对空 | hp 1152, dmg 100/1.5s, 射程 3.5 | **空中护航**。跟在 Golem 后面输出，溅射清小单位 |
| **MegaMinion** | 3 | 空中对空 | hp 837, dmg 147/1.5s | 空中输出（同皇家巨人位） |
| **Bats** | 2 | 空中对空 | hp 32×5, dmg 32/1.3s | 群体骚扰+低成本对空 |
| **Tombstone** | 3 | 建筑 | hp 530, 持续产骷髅 | **拉扯+产兵**。防守时拉扯敌军，进攻前预置骷髅线 |
| **Zap** | 2 | 法术 | dmg 75 半径 2.5, 塔伤 70%, 0.5s 眩晕 | 重置地狱塔/清骷髅海，保护 Golem 推进 |
| **Snowball** | 2 | 法术 | dmg 75 半径 2.5, 击退 | 击退+减速，推开围杀 Golem 的近战群 |

**克制关系**：怕地狱塔/地狱龙+大电连招（Golem 被点燃就断档）；不怕单核爆发
（5100 血要吃满好几轮输出，身后 Witch 持续召骷髅消耗）。

---

## 3️⃣ X 弩（X-Bow Siege）—— 均费 3.25

**打法核心**：X 弩桥头/中线下，射程 11.5 覆盖敌方半场，若 3s 内没被解就烧穿塔。
防守靠 Tesla+Knight+群杂层层阻拦，冰法减速争取时间。

| 卡 | 费 | 类型 | 数值（引擎） | 战术角色 |
|----|----|------|--------------|----------|
| **Xbow** | 6 | 建筑 | hp 1600, dmg 26/0.3s(≈87 dps), 射程 11.5 | **核心**。桥头下压敌塔；被打断（充能重置）就亏费——需要配合保弩 |
| **Tesla** | 4 | 建筑 | hp 1152, dmg 90/1.1s, 射程 5.5, 对空 | **防守中枢**。遁地不被法术溅射；对空对地都能守 |
| **IceWizard** | 3 | 空地对空 | hp 688, dmg 75/1.7s+减速, 射程 5.5 | **减速控制**。群伤减速 35%，给 X 弩多 2-3s 输出窗口 |
| **Archer** | 3 | 空地对空 | hp 304×1, dmg 42/0.9s, 射程 5.0 | （本引擎 Archer 是 1 单位 3 费）后排输出+防空 |
| **Knight** | 3 | 地面单体 | hp 1766, dmg 79/1.2s | **肉盾**。顶在 X 弩前挡冲锋单位 |
| **Skeletons** | 1 | 地面×3 | hp 81×3 | 围杀/拉扯/刷循环 |
| **Log** | 2 | 法术 | dmg 240, 塔伤 80% | 清压弩的近战群杂 |
| **Fireball** | 4 | 法术 | dmg 325, 塔伤 70% | 解女巫/法师等中坚防守；终局磨塔 |

**克制关系**：怕大法术连招（Fireball+小电/毒药 X 弩直接残）；不怕慢速推进
（X 弩在对面组织推进前已烧掉半座塔）。

---

## 4️⃣ 巨骷髅攻城槌（Giant Skeleton Ram）—— 均费 3.88

**打法核心**：BattleRam 冲塔开路，GiantSkeleton 在旁吸收伤害——死亡炸弹（引擎实测：
209 伤、3s 引信、3 格半径、对塔 200%）清掉防守单位，为下一次冲锋清场。WitchMother
诅咒把防守单位变成我方骷髅（滚雪球），Ghost 不可被锁定特性保证稳定输出。

| 卡 | 费 | 类型 | 数值（引擎） | 战术角色 |
|----|----|------|--------------|----------|
| **BarbLog** | 2 | 法术 | 滚动伤害+击退, 召唤野蛮人 | **小法术+铺场**。清群杂并留下野蛮人反打 |
| **BattleRam** | 4 | 地面冲车 | hp 966, dmg 135/0.4s | **开路核心**。冲刺双倍伤害直奔塔，被拆后出两个野蛮人 |
| **GiantSkeleton** | 6 | 地面坦克 | hp 3424, 亡语炸弹 209/3s/3格 | **肉盾+清场**。吸收防守火力，死后炸弹一锅端守军（对塔 200%=418） |
| **WitchMother** | 4 | 空地对空 | hp 532, dmg 诅咒 | **滚雪球**。被诅咒杀死的敌军变成我方骷髅，滚起来防不住 |
| **Ghost** | 3 | 地面单体 | hp 1210, dmg, 相位不可锁定 | **稳定输出**。只能被溅射命中，站在攻城槌后面安心打 |
| **Vines** | 3 | 法术 | 束缚+持续伤害 | **控制**。把防守单位定在原地，给 BattleRam 冲锋开路 |
| **Wizard** | 5 | 空地对空 | hp 720, dmg 133/1.4s 溅射, 射程 5.5 | **后排爆发**。溅射清防守群杂，掩护攻城槌 |
| **MiniSparkys** | 4 | 空中对空×3 | hp 250×3, dmg 55/2.1s, 射程 4.5 | **对空+电击控制**（官方 Zappies，3 只对空对地）。电击打断充能/重置仇恨 |

**克制关系**：怕大法术连招（WitchMother/MiniSparkys 血脆，Fireball 一锅端）；
怕地狱塔（BattleRam 冲不上、GiantSkeleton 被烧穿）；实战核心是
"炸弹清场 → 第二波冲锋"的两段节奏。

**引擎实测注记**（2026-09-08）：GiantSkeleton 亡语炸弹走 TimedExplosive 链路
（death_spawn_data 带 deathDamage 无 hitpoints），击杀后 3.02s 引信精确爆炸，
溅射半径 3.0 格 + 目标碰撞半径，对塔按 crownTowerDamagePercent=200% 结算。
BarbLog 只能部署己方半场（法术部署区限制）。MiniSparkys 即官方 Zappies
（gamedata id 26000052，TID_SPELL_MINI_ZAPMACHINE，4 费 3 只对空对地）。

---

## 附：引擎缺口与替代位（对 FirstLight 口径）

| FirstLight 速猪卡 | 本引擎对应 | 差异说明 |
|-------------------|-----------|----------|
| IceGolem（2 费冰人） | **IceGolemite（2 费）** | 本引擎 gamedata 无 IceGolem；IceGolemite 同价位坦克，hp 更厚（1197 vs IceGolem≈1400@25级），无死亡冰冻（引擎口径），保留"先手坦克"战术位 |
| Evolved Cannon / Evolved Skeletons | Cannon / Skeletons | 未启用进化形态（引擎进化系统在 `evolutions.py`，仅部分卡启用） |
| Hero Musketeer | Musketeer | 同上 |

**BombTower / InfernoTower 修复记录**：`rl/opponents.py::build_card_pool` 原有
`"Tower" in n` 过滤是为了剔除 King_ 塔卡，但误伤这两张合法建筑卡（registry 均
implemented）。已把过滤改为 `n.startswith("King_") or n.endswith("Tower")` 精确匹配
King 塔 + 字面塔名（CannonTower 类），BombTower/InfernoTower 回到随机卡池。

---

## 训练接入（已实现）

- **配置**：`rl/config.py` 新增 `FOUR_DECKS`，`train_solo.py` 通过 `--deck-set four`
  或 config 字段选用；默认镜像卡组保持 `DEFAULT_SOLO_DECK` 不变（向后兼容）。
- **对手池**：`rl/opponents.py` 新增 `FOUR_DECK_SET`；`deck_pool` 型对手（hist/defend）
  会从四卡组抽牌，与 `docs/leaderboard_decks_classified.json` 三分类池叠加。
- **手册的用法**：写脚本对手/专项奖励时按本表查"这张卡该出现在哪、它的威胁是什么"
  （如：Cannon 应在河后 3-4 格中场 → 拉扯；RoyalGiant 桥头即开塔 → punish 窗口是它过河前）。
