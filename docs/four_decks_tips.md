# 四卡组使用技巧（2026-09-09，供用户补充修订）

> 配套 `docs/four_decks_manual.md`（数值与角色表）。本文只写**怎么用**：
> 时机 / 落位 / 连招 / 引擎差异。数值口径：引擎 lv11 为主；**卡牌机制差异处以
> FirstLight CR 引擎数据为准**（2026-09-09 用户定稿，已核对 card_specs/card_logic.gz，
> 见文末"引擎差异清单"）。

---

## 1️⃣ 速猪 2.6（HogRider / IceGolemite / IceSpirits / Musketeer / Cannon / Skeletons / Fireball / Log）

**节奏**：均费 2.62，循环快，核心是"防守赚费 → 立刻猪反打"，不要裸猪开第一波。

- **HogRider**：跳河（jump 160），只打建筑。进攻落位**桥头贴桥柱**（x=3.5 或 14.5），
  直奔最近公主塔；对手圣水真空期（刚交完大费）下猪最赚。双倍期 2 猪连发顶塔。
- **IceGolemite**：先手 2 费顶在猪前 1-2 格吃塔仇恨，猪贴塔输出；防守时放中场把
  冲锋单位引到自己塔前。**死亡减速圈**（半径 2 格空地双打，移速/攻速 −30% 2s，
  FirstLight FreezeIceGolemite 口径）且亡语 33 伤**秒杀同级小骷髅和蝙蝠（32 血）**
  ——死在敌方群杂堆里=白赚一轮清场+减速，当 2 费肉盾够本，死对了位置是赚。
- **IceSpirits**：1 费冻结，进攻跟猪冻塔前守军 0.5s（多敲一下塔），防守冻近战
  给 Musketeer 拉输出窗口。被法术空砸闸门限制：它不是法术，可自由落点。
- **Musketeer**：射程 6，防守站塔后斜角打空中（本卡组唯一稳定对空+核心输出）；
  进攻跟在猪后 2 格，先清守军再让猪拆塔。
- **Cannon**：**拉扯核心**。中场河道后 3-4 格、中线偏猪进攻路线反侧——Hog/Giant/
  BattleRam 会被拉去绕半图。对空无效，对面空袭时换 Musketeer+Fireball 接。
- **Skeletons**：1 费三连，围杀单体高伤（MiniPekka/王子），挡 1 下塔伤换 1 次
  循环；手牌卡死时丢桥头过牌。
- **Fireball**：只解"中坚"（Witch/Wizard/Musketeer 及抱团 3+），对塔斩杀线
  325×70%≈227（残血塔直接收）。⚠ 9h 闸门：双倍期前纯砸塔落点非法，
  前段别留它磨塔，留它换血。
- **Log**：贴我方地面滚出去，清骷髅海/蝙蝠/哥布林+击退 1-2s；对塔 80%（192）。
  对手防守猪用群杂时，Log 跟进滚开守军=白赚一塔血。

---

## 2️⃣ 石头人（Golem / Witch / BabyDragon / MegaMinion / Bats / Tombstone / Zap / Snowball）

**节奏**：均费 3.62，"沉底攒费→一波推"。开局别急着交 8 费，先防守看清对面卡组。

- **Golem**：**沉底**王塔侧后（不进王塔后 1 格禁建带没关系，它是部队可走位），
  攒费等双倍期或费用领先再整波下；死后分裂 Golemite 继续顶。
- **Witch**：跟在 Golem 正后方 2-3 格，持续召骷髅当肉盾+清杂；对空、别单放
  （5 贞脆皮，Fireball 直接蒸发）。
- **BabyDragon**：空中护航，Golem 头顶斜后方，溅射清骷髅海/哥布林；
  也可防守解地面群杂（对地溅射）。
- **MegaMinion**：3 费高伤对空，解气球/飞龙；进攻补在空中护航第二梯队。
- **Bats**：2 费 5 只，围杀+防空骚扰；防守贴身围近战坦克，进攻跟 Golem 补输出。
- **Zap / Snowball**：**不着急都打**——两张小法术是 Golem 推进的两道保险，Zap 留给
  重置地狱塔/王子冲锋/围杀骷髅，Snowball 留给推开围杀 Golem 的近战圈。推进时
  手里捏着至少一张，对手会忌惮交围杀；两张都丢光后 Golem 就是在裸奔。
- **Tombstone**：产骷髅节奏 500ms×2 只（与官方一致），放中场把 Hog/Ram 引进
  骷髅海里；进攻前预置一波骷髅线。

---

## 3️⃣ X 弩（Xbow / Tesla / Skeletons / IceWizard / Archer / Knight / Log / Fireball）

**节奏**：均费 3.25，自闭攻城。核心循环：**保弩到锁塔 > 一切**，弩被拆不亏费别硬保。

- **Xbow**：桥头/中线偏己方 1 格下（射程 11.5 从己方半场即罩到敌方公主塔，
  罩不到王塔），下弩瞬间就烧塔；被近战贴身=充能重置血亏，所以**先手要有
  Knight 顶在前**或对手刚交完防守费再下。
- **Knight**：肉盾。下弩同帧顶弩前 1 格；防守时顶在桥后拉住冲锋单位喂 Tesla。
- **Tesla**：**防守中枢**。遁地躲法术溅射+对空，放弩后 1-2 格中路；对面
  空军流时 Tesla+Archer 双接。
- **IceWizard**：群伤减速 35%，贴弩放置——减速所有试图贴弩的单位，
  相当于给弩+2-3s 输出窗口；防守减速坦克喂塔。
- **Archer**：3 费单只（引擎口径），站弩后斜角补对空+点伤；被 Fireball
  一锅是常态，落位离 Tesla 错开半格防连坐。
- **Skeletons**：围杀贴弩的近战、骗 Log/Fireball；弩被贴时丢弩脚下 3 只围死。
- **Log**：滚开压弩的近战群杂+击退；对手下弩对射时，Log 滚对方弩磨它血。
- **Fireball**：解对方弩/炮台后面的中坚防守（Witch/法师）；终局残血塔
  斩杀 227。⚠ 前段纯砸塔非法（9h），磨塔账只在双倍期算得平。

---

## 4️⃣ 巨骷髅攻城槌（BarbLog / BattleRam / GiantSkeleton / WitchMother / Ghost / Vines / Wizard / MiniSparkys）

**节奏**：均费 3.88，两段式攻城：**攻城槌顶着骷髅巨人推进 → 第一波换掉防守 →
亡语/诅咒清场 → 第二波收割**。核心原则：**骷髅巨人绝不单下白送**——它必须有
BattleRam 顶在前吸收第一轮仇恨，或配合 Wizard/Ghost 走一路压制。

- **双路 vs 单路选择**：对面防守厚（双建筑/多群杂）→ **双路**：一路 BattleRam+
  野蛮人骚扰拉仇恨，另一路 GiantSkeleton 顶锤推进，让对手费用拆散两头救；
  对面防守薄或残局 → **单路**：BattleRam 顶前开路，GiantSkeleton 跟后，
  Wizard/Ghost 排后排，一路碾过去。
- **BattleRam**：开路核心。冲锋 3 格内双倍速+双倍伤，落位**离塔 ≥4 格**（预留
  冲锋距离），直奔塔；被拆后掉 2 野蛮人继续搅。单路推进时 Ram 永远在
  GiantSkeleton 前面 1-2 格——让它先吃塔伤/先撞防守，骷髅巨人留着血走进亡语圈。
- **GiantSkeleton**：6 费肉盾+清场，**亡语炸弹 209 伤/3s 引信/3 格半径、对塔
  200%**。它的价值在死的那一刻：跟着 Ram 打单路时贴塔走位，让炸弹同时罩到
  塔+守军；双路时作为第二路的核心自己扛。**别在对面满费且无 Ram 掩护时单独
  过河**——被拉扯+围杀，炸弹炸空气，6 费白送。
- **WitchMother**：4 费诅咒（Voodoo 弹 110 伤+诅咒 5s）。被诅咒敌军死亡生成
  **我方 VoodooHog×1**（FirstLight VoodooCurse 同口径：DeathSpawn=VoodooHog×1，
  死亡延迟部署）——诅咒贴着 Wizard 溅射走，"诅咒+溅射收人头"滚雪球。
- **Ghost**：部署即隐身（不可被锁定，但吃溅射/法术），出手才显形、脱战 1.8s
  再隐身——**绕开对方肉盾直接摸塔/摸后排**；对方单体防守全部落空。
- **Vines**：束缚 2s（不可移动+攻击）+ 60DPS×2s=**总伤 120**（FirstLight 口径：
  同级烟花 119/公主 102/吹箭手 52 全在毒杀线内），可放敌方半场、拽落空中单位。
  防守定住冲塔核心（Hog/Ram/王子）喂塔；进攻套住对方防守中坚给 Ram 冲锋开路；
  残局可以直接套对面残血公主/烟花**单人收掉一个防守单位**。
- **Wizard**：5 费溅射后排，跟 GiantSkeleton 后清防守群杂+配合诅咒收人头；防空主力。
- **MiniSparkys**（=官方 Zappies，3 只）：hp250×3，55 伤/2.1s/射程 4.5，
  对空对地——电击连发能打断充能/重置仇恨，分开放防一锅端。
- **BarbLog**：2 费滚动法术+落点召野蛮人，清杂后**野蛮人原地反打**，攻防一体；
  部署区限制：只能己方半场释放（法术闸门口径）。

---

## 跨卡组通用（引擎机制提醒）

- **塔是矩形**（公主 3×3 / 国王 4×4）：近战贴塔沿边站，围杀单位会自然摊开；
  建筑禁放王塔身后 1 格带（Tombstone/Tesla 别想藏王后）。
- **法术闸门**：双倍期前伤害法术必须罩到存活部队/建筑，纯罩塔非法（9h）；
  法术也别砸已毁塔本体（浪费）。
- **不裸下**：无圣水优势别单下高费核心（Golem/GiantSkeleton/Xbow 裸下=白送），
  攒费或 2 卡协同。
- 对空余量自查：速猪 2 张（Musketeer/Fireball）、石头人 4 张（全靠后排）、
  X弩 3 张（Tesla/IceWizard/Archer）、巨骷髅 3 张（WitchMother/Wizard/MiniSparkys）。

---

## 附：引擎差异清单（FirstLight CR 数据定稿，2026-09-09）

用户口径：**卡牌机制不同的地方，先以 FirstLight CR 引擎数据为准**。以下条目
已逐条核对 `E:/FirstLight_CR/native_runner/data/competitive/card_specs.json.gz`
与 `card_logic.json.gz`（25 级基准数值；本引擎 lv11 数值趋势一致）：

| 卡 | 定稿口径 | FirstLight 数据出处 |
|---|---------|--------------------|
| **IceGolemite 亡语** | 死亡减速圈：半径 2 格、空地双打，移速/攻速 **−30% 2s**；亡语伤害 33，**可秒同级小骷髅(32血)/蝙蝠(32血)** | `AEO.FreezeIceGolemite`（Buff=IceWizardSlowDown, BuffTime 2000, Radius 2000, HitsAir+Ground）+ `CHARACTER.IceGolemite` DeathDamage 33 |
| **Vines** | 束缚 **2s**（不可移动/攻击）+ DOT **60/s × 2s = 总伤 120**——同级毒杀烟花(119)/公主(102)/吹箭手(52)；**可放敌方半场**（CanDeployOnEnemySide）+ 拽落空中单位（Air_To_Ground） | `BUFF.Vines_Trap_Snare_Base`（DamagePerSecond 60）+ `AEO.Vines_AeO`（LifeDuration 2000, Radius 2500）+ `SPELL_OTHER.Vines` + `ACTION.Vines_Air_To_Ground` |
| **Tombstone** | 产兵 **500ms × 2 只**（与官方一致，此前"远超官方"为误判）；死亡额外爆 4 骷髅（DeathSpawnCount 4） | `BUILDING.Tombstone`（SpawnNumber 2, SpawnInterval 500, DeathSpawnCharacter Skeleton ×4） |
| **WitchMother 诅咒** | 被诅咒敌军死亡生成 **VoodooHog×1**（属施法方，延迟部署）——非"变成骷髅" | `BUFF.VoodooCurse`（DeathSpawn VoodooHog, Count 1, DeathSpawnIsEnemy true） |

> 本仓库引擎若与上表有出入（如 IceGolemite 死亡减速未接 buff、Vines DOT 缺失），
> 以补齐本引擎实现为后续工作项，文档口径以上表为准。
