# 卡牌覆盖矩阵（P0-1 基本信息录入）

- 元数据来源：仓库 cards.json + cr-api-data 补充(11)；per-level 数值来源：{'characters': 'repo(120)', 'building': 'repo(89)', 'spell': 'repo(71)', 'projectile': 'repo(92)'}
- gamedata 快照条目：146（塔 4）
- 批量冒烟（scripts/batch_smoke.py）：全卡 构造→部署→30s 战斗有行为。冒烟通过 146 张，其中 60 张 data_ready 升级为 implemented。

| 状态 | 数量 | 说明 |
|---|---|---|
| implemented | 107 | 已实现（冒烟接入 + 数值验证） |
| data_ready | 0 | 官方数值就绪，待机制接入 |
| needs_review | 15 | 拿不准，待人工决策 |
| temporary_event | 14 | 活动/超级临时卡（默认排除） |
| variant | 6 | 变体/形态条目（排除，基础形态保留） |
| tower | 4 | 王塔/公主塔 |
| missing_from_snapshot | 2 | 官方新卡，快照缺失 |

## 需人工评审（模型拿不准）

- **Little Prince**（内部名 `LittlePrince`，费=3）：STATS_UNRESOLVED: summonCharacter LittlePrince 不在 per-level 数值表（需 L1 升级数值表或人工录入）
- **Goblin Demolisher**（内部名 `GoblinDemolisher`，费=4）：STATS_UNRESOLVED: summonCharacter GoblinDemolisher 不在 per-level 数值表（需 L1 升级数值表或人工录入）
- **Goblin Machine**（内部名 `GoblinMachine`，费=5）：STATS_UNRESOLVED: summonCharacter GoblinMachine 不在 per-level 数值表（需 L1 升级数值表或人工录入）
- **Suspicious Bush**（内部名 `SuspiciousBush`，费=2）：STATS_UNRESOLVED: summonCharacter SuspiciousBush 不在 per-level 数值表（需 L1 升级数值表或人工录入）
- **Goblinstein**（内部名 `Goblinstein`，费=5）：STATS_UNRESOLVED: summonCharacter Goblinstein 不在 per-level 数值表（需 L1 升级数值表或人工录入）
- **Rune Giant**（内部名 `GiantBuffer`，费=4）：STATS_UNRESOLVED: summonCharacter GiantBuffer 不在 per-level 数值表（需 L1 升级数值表或人工录入）
- **Berserker**（内部名 `Berserker`，费=2）：STATS_UNRESOLVED: summonCharacter Berserker 不在 per-level 数值表（需 L1 升级数值表或人工录入）
- **Boss Bandit**（内部名 `BossBandit`，费=6）：STATS_UNRESOLVED: summonCharacter BossBandit 不在 per-level 数值表（需 L1 升级数值表或人工录入）
- **Goblin Hut**（内部名 `GoblinHut`，费=4）：STATS_UNRESOLVED: summonCharacter GoblinHut_Rework 不在 per-level 数值表（需 L1 升级数值表或人工录入）
- **Mirror**（内部名 `Mirror`，费=1）：STATS_UNRESOLVED: spell Mirror 不在 spell/projectile 表（需 L1 升级数值表或人工录入）
- **Royal Delivery**（内部名 `RoyalDelivery`，费=3）：STATS_UNRESOLVED: spell RoyalDelivery 不在 spell/projectile 表（需 L1 升级数值表或人工录入）
- **Lightning**（内部名 `GlobalLightning`，费=6）：ELIXIR_MISMATCH: 官方=6 vs 快照=1（数值按官方）；STATS_UNRESOLVED: spell GlobalLightning 不在 spell/projectile 表（需 L1 升级数值表或人工录入）
- **Void**（内部名 `DarkMagic`，费=3）：STATS_UNRESOLVED: spell DarkMagic 不在 spell/projectile 表（需 L1 升级数值表或人工录入）
- **Goblin Curse**（内部名 `GoblinCurse`，费=2）：STATS_UNRESOLVED: spell GoblinCurse 不在 spell/projectile 表（需 L1 升级数值表或人工录入）
- **Spirit Empress**（内部名 `MergeMaiden`，费=6）：STATS_UNRESOLVED: spell MergeMaiden 不在 spell/projectile 表（需 L1 升级数值表或人工录入）

## 快照缺失的官方新卡

- **Ronin** id=26000106 费=5 稀有度=legendary
- **Vines** id=28000026 费=3 稀有度=epic

## 临时/活动卡（默认排除，可人工恢复）

- SuperWitch（Super Witch）
- SuperLavaHound（Super Lava Hound）
- SuperHogRider（Santa Hog Rider）
- SuperIceGolemite（Super Ice Golem）
- SuperArcher（Super Archers）
- SuperMiniPekka（Super Mini P.E.K.K.A）
- PrinceBuff（Raging Prince）
- TriWizards（Wizard Trio）
- SuperKnight（Super Knight）
- BarbarianLauncher（Barbarian Launcher）
- GoblinPartyHut（Party Hut）
- GoblinRocketSilo（Rocket Silo）
- GoblinPartyRocket（Party Rocket）
- WarmSpell（Warmth）
- SuperEliteArcher（Super Magic Archer）
- RoyalRecruits_Chess（Royal Recruits）
- SuperHogRiderTerry（Terry）
- SkeletonWarriors_SpookyChess（Royal Recruits）
- MergeMaiden_Normal（Spirit Empress (Ground)）
- MergeMaiden_Mounted（Spirit Empress）

## 官方元数据未对照（健康检查，按 id）

（无）
