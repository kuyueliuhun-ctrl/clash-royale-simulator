# 精英卡 / Hero 卡 — Fandom 数据抓取报告

抓取时间: 2026 会话 · 工具: `node scripts/cdp_extract.js` (CDP @ 172.28.144.1:9222)
原始落盘文件: `docs/_elite_<卡名>.txt` (每卡一个), 外加 `_elite_hero_overview.txt`、`_elite_heroes_category.txt`、`_elite_card_evolution.txt`。

## 命名规则 (Naming Rule)

- `https://clashroyale.fandom.com/wiki/Hero` **不存在** ("There is currently no text in this page")。
- Hero 总览页为复数: `wiki/Heroes` → 标题 "Heroes" (in: Basics, Cards), 列出全部 16 个 Hero 变体。
- **Hero 单页命名规则: `<原卡名>/Hero`**, 例如 `wiki/Knight/Hero` (页面标题 "Knight/Hero", 内文称 "Hero Knight" / "Heroic Knight")。候选命名 `Knight_(Hero)`、`Hero_Knight` 均为空页 (占位 "Create the page")。
- 例外: **Elite Archer** 与 **Ice Golemite** 在该 wiki 上连基础页都不存在 (基础页 + `/Hero` 子页 + 搜索均无)。`Ice Golem/Hero` 存在,已抓取作为 IceGolemite 的最可能对应。

## Wild slot 语义 (原文摘录)

来源 `wiki/Card_Evolution`:

> "Card Evolution is a feature that allows specific Cards placed in the Evolution slots to become more powerful. Card Evolution and the Evolution slot are both unlocked at Arena 3, **with the Wild slot, that lets you choose if you want to activate an Evolution, or a Hero version of a Card at Arena 10**."

> "Every card with an Evolution takes 6 Evolution Shards or **Wild Shards** to be used in battle."

> "Wild Shards can be obtained from Shop Offers, Special Challenges, Pass Royale, and 4-Star Lucky Chests. … **Unlike Evolution Shards, Wild Shards are capable of being used on any card with an Evolution, and go beyond the player's limit.**"

> "On 16/3/2026 the 2026 Mid-March Update **changed the format of Evolution and Hero slots into one Evolution, one Hero and one Wild (from 2 evo and 2 hero)**."

来源 `wiki/Heroes`:

> "Heroes are a feature rarity that allow specific Cards placed in the Champion slots to gain an ability. **Heroes and the first Champion slot are both unlocked at Arena 5.** Every card with a Hero variation requires 200 [hero fragments] to be unlocked."

> "**Only two Heroes can be in a deck at a time, and only in the Hero and Wild slots.** Those slots are also shared with Champion card, which means that the player can have 1 Hero and 1 Champion at the same time."

> "When a Hero activates their ability, but did not start casting it before they are defeated, **the Elixir spent on the ability will be refunded** once they are defeated."

> "Heroes can be upgraded by the base card's Wild Cards of respective rarity."

> "On 4/8/2026 the Balance Update made **every ability single-use**." (与各卡 History 中 "made the ability single use (from Nsec cooldown)" 一致)

语义小结: 槽位结构 = 1 Evolution 槽 + 1 Hero 槽 + 1 Wild 槽 (2026-03-16 起); Wild 槽可在对局中把卡临时"激活"为 Evolution 或 Hero 形态; 解锁: Evo 槽 Arena 3、Hero/Champion 槽 Arena 5、Wild 槽 Arena 10 (两页口径一致: Evolution 页说 Wild slot "at Arena 10", Heroes 页说 Wild slot "unlocked at Arena 10")。

## 逐卡数据 (17 卡)

每节: ① URL 与确切标题 ② Statistics ③ 能力描述原文 ④ 完备度。等级表均为 wiki 原表 (wiki 注明 "There will be rounding mistakes, as we have stats from level 11")。

---

### 1. Knight
- **URL/标题**: https://clashroyale.fandom.com/wiki/Knight/Hero — 标题 "Knight/Hero" (in: 3-Elixir Cards, Heroes with 2-Elixir Abilities, Common Cards)
- **基础信息**: Elixir 3, Ability 2, Common, Troop, Arena Training Camp, Release 2025-12-01
- **Ability: Triumphant Taunt** — "Gains a shield and taunts nearby enemies, forcing them to attack him."
  - 原文: "After a 1-second delay, the Hero Knight activates his 'Triumphant Taunt', taunting every enemy troop and building in a 7.5-tiles range, causing them to attack him. He also gains a shield. Both the shield and the taunt effect last for 5 seconds… The Triumphant Taunt ability costs 2 Elixir to activate."
  - 属性表: Cost 2 · Taunt Range 7.5 tiles (2026-03-02 削为 6.5) · Cast Time 0.933s · Ability Duration 5s
- **Statistics (主体)**: Hit Speed 1.2s · First Hit 0.5s · Speed Medium(60) · Deploy 1s · Range Melee: Medium(1.2) · Target Ground · x1
- **等级表** (HP / Damage / DPS / Shield HP):
  1: 681/94/78/197 · 2: 749/104/86/217 · 3: 824/94/78/239 · 4: 906/104/86/263 · 5: 997/114/95/289 · 6: 1097/125/104/318 · 7: 1206/138/115/350 · 8: 1327/152/126/385 · 9: 1460/167/139/423 · 10: 1605/184/153/465 · 11: 1766/202/168/512 · 12: 1943/222/185/563 · 13: 2137/244/203/620 · 14: 2351/269/224/681 · 15: 2586/296/246/750 · 16: 2844/325/270/825
- **完备度**: 完整 (统计✓ 能力✓ 平衡史✓)。Wild slot: 无单页引用 (见全局语义)。

### 2. Musketeer
- **URL/标题**: https://clashroyale.fandom.com/wiki/Musketeer/Hero — 标题 "Heroic Musketeer"
- **基础信息**: Elixir 4, Ability 3, Rare, Troop, Training Camp, 2025-12-01
- **Ability: Trusty Turret** — "Spawns a short-range, rapid-firing auto-turret that deals landing damage in front of the Musketeer." (单页内引言为 "Spawns a short-range auto-turret in front of the Musketeer.")
  - 原文: "After a 1-second delay, the Hero Musketeer activates her 'Trusty Turret', placing a turret 3 tiles forward of the Musketeer… This turret has a lifetime of 10 seconds… The Trusty Turret ability costs 3 [elixir] to activate."
  - 属性表: Cost 3 · Turret Range 4.0 · Target Air & Ground · Turret Lifetime 10 seconds · Deploy 1 second
- **Statistics (主体)**: Hit Speed 1s · First Hit 0.5s · Speed Medium(60) · Deploy 1s · Range 6 · Target Air & Ground · x1
- **等级表** (L3-16, HP/Dmg/DPS/TurretHP/Turret decay/s/TurretDmg/TurretDPS/TurretSpawnDmg):
  3: 336/101/101/717/71.7/65/130/95 · 4: 370/111/111/788/78.8/72/144/105 · 5: 407/122/122/867/86.7/79/158/115 · 6: 448/135/135/954/95.4/87/174/127 · 7: 492/148/148/1049/104.9/96/192/139 · 8: 542/163/163/1154/115.4/105/210/153 · 9: 596/179/179/1269/126.9/116/232/169 · 10: 655/197/197/1396/139.6/127/254/185 · 11: 721/217/217/721(表内异常,疑应为1556)/153.6/140/280/204 · 12: 793/239/239/1690/169/154/308/224 · 13: 872/263/263/1859/185.9/169/338/247 · 14: 960/289/289/2044/204.4/186/372/272 · 15: 1056/318/318/2249/224.9/205/410/299 · 16: 1161/349/349/2474/247.4/225/450/329
- **完备度**: 完整 (含 L11 TurretHP=721 的 wiki 表内疑似笔误,已如实记录)。

### 3. MiniPekka (Mini P.E.K.K.A.)
- **URL/标题**: https://clashroyale.fandom.com/wiki/Mini_P.E.K.K.A./Hero — 标题 "Heroic Mini P.E.K.K.A."
- **基础信息**: Elixir 4, Ability 1, Rare, Troop, Training Camp, 2025-12-01
- **Ability: Breakfast Boost** — "Eats pancakes to level up. The longer you wait, the more pancakes he cooks!"
  - 原文: "…immediately eats all the pancakes he cooked, granting him extra Levels and healing himself for 30%. … He will need 22 seconds to get one pancake meter filled, but he can also get 10 seconds of progress with every attack. Every filled pancake meter increases the Levels gained… With no meters… 1 Level, but with a maximum of 3 meters, he can gain 5 Levels."
  - 属性表: Cost 1 · Range Himself · Cast Time 0.933s · Heal 30% · Level Increase: 0 bars=+1 / 1 bar=+1 (累计+2) / 2 bars=+1 (累计+3) / 3 bars=+2 (累计+5)
- **Statistics (主体)**: Hit Speed 1.6s · First Hit 0.5s · Speed Fast(90) · Deploy 1s · Range Melee: Short(0.8) · Ground · x1
- **等级表** (L3-21, HP/Dmg/DPS; 17-21 为 L16 使用能力后的加级):
  3: 669/352/220 · 4: 735/387/241 · 5: 809/426/266 · 6: 890/469/293 · 7: 979/516/322 · 8: 1077/567/354 · 9: 1184/624/390 · 10: 1303/686/428 · 11: 1433/755/471 · 12: 1576/831/519 · 13: 1734/914/571 · 14: 1907/1005/628 · 15: 2098/1105/690 · 16: 2308/1216/760 · 17: 2539/1338/836 · 18: 2793/1471/919 · 19: 3072/1618/1011 · 20: 3379/1780/1112 · 21: 3717/1958/1223
- **完备度**: 完整。

### 4. Valkyrie
- **URL/标题**: https://clashroyale.fandom.com/wiki/Valkyrie/Hero — 标题 "Valkyrie/Hero" (2026-08-03 新增,页面仍偏 stub)
- **基础信息**: Elixir 4, Ability 3, Rare, Troop, Bone Pit
- **Ability: Wild Whirlwind** (Heroes 页描述) — "Spins rapidly, dealing damage and increasing her movement speed while taking less damage."
  - 属性表: Cost 3 · Hit Speed 0.25s · Crown Tower Damage −50% · Speed Medium(60) · Duration 3.5s · Radius 2.5 · Damage reduction 15% · Dash Distance 5.5
- **Statistics (主体)**: Hit Speed 1.5s · First Hit 0.1s · Speed Medium(60) · Deploy 1s · Range Melee: Medium(1.2) · Splash Radius 2 · Ground · x1
- **等级表** (L3-16, HP/AreaDmg/DPS/AbilityDmg):
  3: 890/124/82/45 · 4: 979/137/91/50 · 5: 1076/150/100/55 · 6: 1184/165/110/60 · 7: 1303/182/121/66 · 8: 1433/200/133/73 · 9: 1576/220/146/80 · 10: 1734/242/161/88 · 11: 1907/266/177/97 · 12: 2098/293/195/107 · 13: 2307/322/214/117 · 14: 2538/354/236/129 · 15: 2792/389/259/142 · 16: 3071/428/285/156
- **完备度**: 部分 — Statistics 与能力数字✓; 能力细节正文缺失 (页面正文在 "unlocked with 200 hero fragments." 后截断), History 空。

### 5. Wizard
- **URL/标题**: https://clashroyale.fandom.com/wiki/Wizard/Hero — 标题 "Wizard/Hero"
- **基础信息**: Elixir 5, Ability 2 (属性表记 1,两者矛盾,如实记录), Rare, Troop, Spell Valley, 2026-01-05
- **Ability: Fiery Flight** — "Launches into the air and throws fearsome fire tornadoes."
  - 原文: "After a 1-second delay, the Hero Wizard activates his fiery flight, which makes him take flight for 5 seconds. While he is flying, not only he will get a 50% movement speed increase (now classified as fast), his fireballs also create 3 tile radius tornadoes, which does its own damage (reduced against crown towers)…"
  - 属性表: Cost 1 · Radius 4 · Flight Speed Increase +50% · Tornado Duration 2s · Target Air & Ground · Transport Air
- **Statistics (主体)**: Hit Speed 1.4s · First Hit 0.4s · Speed Medium(60) · Deploy 1s · Range 5.5 · Splash Radius 1.5 · Projectile Speed 600 · Air & Ground · x1
- **等级表** (L3-16, HP/Dmg/DPS/TornadoDmg):
  3: 388/131/93/20 · 4: 427/144/102/22 · 5: 470/159/113/24 · 6: 517/174/124/27 · 7: 568/192/137/29 · 8: 625/211/150/32 · 9: 688/232/165/36 · 10: 756/255/182/39 · 11: 832/281/200/43 · 12: 915/309/220/47 · 13: 1007/340/242/52 · 14: 1107/374/267/57 · 15: 1218/411/293/63 · 16: 1340/453/323/69
- **完备度**: 完整 (能力费用 2 vs 属性表 1 矛盾已注明)。

### 6. Bowler
- **URL/标题**: https://clashroyale.fandom.com/wiki/Bowler/Hero — 标题 "Bowler/Hero"
- **基础信息**: Elixir 5, Ability 2, Epic, Troop, Rascal's Hideout, 2026-05-04
- **Ability: Stone Swish** — "Plants his feet and throws boulders with increased range."
  - 原文: "After 2.5sec of the ability pressed, the bowler will change the attack to a long-ranged mortar attack for 7.3sec. Getting a total of 3 shots during that duration."
  - 属性表: Cost 2 · Hit Speed 1.9s · Cast Time 2.5s · Duration 7.3s · Range 11.5 (2026-06-01 duration 9s→7.3s; 2026-08-04 projectile range 7.5→7)
- **Statistics (主体)**: Hit Speed 2.5s · First Hit 0.5s · Speed Slow(45) · Deploy 1s · Range 4 · Projectile Range 7 · Width 3.6 · Projectile Speed 170 · Ground · 1x
- **等级表** (L6-16, HP/AreaDmg/DPS/AbilityDmg/AbilityCTD):
  6: 1292/179/71/315/158 · 7: 1421/197/78/347/173 · 8: 1563/217/86/382/191 · 9: 1720/239/95/420/210 · 10: 1892/263/105/462/231 · 11: 2081/289/115/508/254 · 12: 2289/318/127/559/279 · 13: 2518/350/140/615/307 · 14: 2770/385/154/676/338 · 15: 3047/423/169/744/372 · 16: 3351/465/186/818/409
- **完备度**: 完整 (等级表自 L6 起)。

### 7. Giant
- **URL/标题**: https://clashroyale.fandom.com/wiki/Giant/Hero — 标题 "Giant/Hero"
- **基础信息**: Elixir 5, Ability 2, Rare, Troop, 2025-12-01
- **Ability: Heroic Hurl** — "Throws the highest-hitpoint enemy troop in reach across the Arena."
  - 原文: "After a 1-second delay and when a unit gets in range, the Hero Giant activates his 'Heroic Hurl', grabbing the highest HP enemy troop within 2 tiles around him and throws them horizontally, also dealing damage to them when they land. While on the air, these troops are untargetable by ground-targeting troops and the Earthquake… The Heroic Hurl ability costs 2 Elixir to activate."
  - 属性表: Cost 2 · Throwback Range 9 · Cast Time 0.933s · Units Affected 1 · Target Air & Ground · Unit Stun Duration 2s
- **Statistics (主体)**: Hit Speed 1.5s · First Hit 0.5s · Speed Slow(45) · Deploy 1s · Range Melee: Medium(1.2) · Target Buildings · x1
- **等级表** (L3-16, HP/Dmg/DPS/ImpactDmg):
  3: 1851/118/78/63 · 4: 2036/130/86/69 · 5: 2240/143/95/76 · 6: 2464/157/104/84 · 7: 2710/173/115/92 · 8: 2981/190/126/101 · 9: 3279/209/139/112 · 10: 3607/230/153/123 · 11: 3968/253/168/135 · 12: 4365/278/185/149 · 13: 4801/306/204/163 · 14: 5281/337/224/180 · 15: 5810/370/246/198 · 16: 6391/407/271/217
- **完备度**: 完整。

### 8. Goblins
- **URL/标题**: https://clashroyale.fandom.com/wiki/Goblins/Hero — 标题 "Goblins/Hero"
- **基础信息**: Elixir 2, Ability 1, Common, Troop, 2026-02-02
- **Ability: Banner Brigade** — "The last Goblin standing drops a banner that calls in reinforcements."
  - 原文: "The ability will be disabeld until the last goblin is killed. When the last goblin dies, a banner will be deployed that last 5sec. When the ability is pressed during that time, 2 Brigade Goblins will spawn a little behind. Brigade Goblins has the exact stats as the original goblins."
  - 属性表: Cost 1 · Hit Speed 1.1s · First Hit 0.6s · Speed Very Fast(120) · Deploy 1s · Range Melee: Short(0.5) · Ground · Brigade Count x2 · Banner Lifetime 5s (2026-08-04 由 x3 削为 x2)
- **Statistics (主体)**: Hit Speed 1.1s · First Hit 0.6s · Speed Very Fast(120) · Deploy 1s · Range Melee: Short(0.5) · Ground · x4
- **等级表** (L1-16, HP/Dmg/DPS/BrigadeHP/BrigadeDmg/BrigadeDPS — Brigade 三项与本体相同):
  1: 78/48/43 · 2: 86/53/48 · 3: 94/58/52 · 4: 104/64/58 · 5: 114/71/64 · 6: 125/78/70 · 7: 138/85/77 · 8: 152/94/85 · 9: 167/103/93 · 10: 184/114/103 · 11: 202/125/113 · 12: 222/138/125 · 13: 244/151/137 · 14: 269/166/150 · 15: 296/183/166 · 16: 325/201/182
- **完备度**: 完整。

### 9. MegaMinion (Mega Minion)
- **URL/标题**: https://clashroyale.fandom.com/wiki/Mega_Minion/Hero — 标题 "Mega Minion/Hero"
- **基础信息**: Elixir 3, Ability 2, Rare, Troop, 2026-02-02
- **Ability: Wounding Warp** — "Warps to the lowest-hitpoint enemy, dealing damage on arrival."
  - 原文: "When deployed, a marker will be deployed to the lowest hitpoint target. When that unit dies, the mark will move to the next unit. When the ability is preased, Mega Minion will teleport to that tile, dealing damage. Afterwards the Mega Minion will behave normally, but with -75% on the tower."
  - 属性表: Cost 2 · Crown Tower Damage 25% · Teleport Range Infinite (2026-08-04 后 tower multiplier x0.25 且永久)
- **Statistics (主体)**: Hit Speed 1.5s · First Hit 0.4s · Speed Medium(60) · Deploy 1s · Range Melee: Long(1.6) · Projectile Speed 1000 · Air & Ground · x1 · Air
- **等级表** (L3-16, HP/Dmg/DPS/WarpDmg):
  3: 390/146/97/186 · 4: 430/160/106/205 · 5: 472/176/117/225 · 6: 520/194/129/248 · 7: 572/213/142/273 · 8: 629/234/156/300 · 9: 692/258/172/330 · 10: 761/284/189/363 · 11: 837/312/208/399 · 12: 921/343/228/439 · 13: 1013/378/252/483 · 14: 1114/415/276/531 · 15: 1225/457/304/584 · 16: 1348/502/334/643
- **完备度**: 完整。

### 10. IceWizard (Ice Wizard) — **missing**
- **URL/标题**: https://clashroyale.fandom.com/wiki/Ice_Wizard/Hero — 标题 "Ice Wizard/Hero",但页面正文只有 "**Coming soon...**"
- 无 Statistics、无能力描述、无等级表。Heroes 页与 Knight/Hero 页的导航均不含 Ice Wizard 的 Hero 标注 (Legenday 列仅 "Ice Wizard")。
- **完备度**: missing (页面存在但为占位)。

### 11. Tombstone
- **URL/标题**: https://clashroyale.fandom.com/wiki/Tombstone/Hero — 标题 "Tombstone/Hero" (Building)
- **基础信息**: Elixir 3, Ability 5, Rare, Building, Bone Pit, 2026-06-01
- **Ability: Regal Revive** (Heroes 页描述) — "Tomb Queen rises from the earth, targeting buildings."
  - 单页无能力正文与 Tomb Queen 属性表 (页面较 stub); 平衡史: 2026-07-06 Queen HP +20%、ability cost 6→5、移除持续产骷髅; 2026-08-04 Queen Damage +32%、sight 5.5→7 tiles、HP +4%。
- **Statistics**: Tombstone Attributes: Cost 3 · Spawn Speed 4s · Deploy 1s · Lifetime 30s · Building · Rare。Skeleton Attributes: Hit Speed 1.1s · First Hit 0.5s · Speed Fast(90) · Range Melee: Short(0.5) · Ground。
- **等级表** (L3-16, TombHP/Decay/s/SkelHP/SkelDmg/SkelDPS/QueenHP/QueenDmg):
  3: 247/8.2/38/38/34/1971/197 · 4: 271/9/42/42/38/2168/217 · 5: 299/10/46/46/41/2384/238 · 6: 328/10.9/50/50/45/2623/262 · 7: 361/12/55/55/49/2885/288 · 8: 397/13.2/61/61/55/3174/317 · 9: 437/14.6/67/67/60/3491/349 · 10: 481/16/74/74/67/3840/384 · 11: 529/17.6/81/81/73/4224/422 · 12: 582/19.4/89/89/80/4646/464 · 13: 640/21.3/98/98/89/5111/511 · 14: 704/23.5/108/108/98/5622/562 · 15: 775/25.8/119/119/108/6184/618 · 16: 852/28.4/130/130/118/6803/680
- **完备度**: 部分 — Statistics✓ (Queen 数据仅等级表列), 能力细节正文 missing。

### 12. Berserker
- **URL/标题**: https://clashroyale.fandom.com/wiki/Berserker/Hero — 标题 "Berserker/Hero"
- **基础信息**: Elixir 2, Ability 3, Common, Troop, Jungle Arena, 2026-08-04
- **Ability: Savage Survival** (Heroes 页描述) — "A Bear spirit emerges through her, making her attacks go rapid and preventing her health from going below 1 HP while dealing reduced damage to Crown Towers."
  - 属性表: Cost 3 · Hit Speed 0.2s · First Hit 0.2s · Speed Ultra Fast(135) · Duration 4s · Crown Tower Damage −75% · Minimum Hitpoints 1
- **Statistics (主体)**: Hit Speed 0.6s · First Hit 0.2s · Speed Fast(90) · Deploy 1s · Range Melee: Short(0.8) · Ground · x1
- **等级表** (L1-16, HP/Dmg/DPS/BearDmg):
  1: 345/39/65/64 · 2: 380/43/71/71 · 3: 418/48/80/78 · 4: 460/52/86/86 · 5: 506/58/96/94 · 6: 556/63/105/104 · 7: 612/70/116/114 · 8: 673/77/128/125 · 9: 740/84/140/138 · 10: 815/93/155/152 · 11: 896/102/170/167 · 12: 986/112/186/184 · 13: 1084/123/205/202 · 14: 1193/136/226/222 · 15: 1312/149/248/245 · 16: 1443/164/273/269
- **完备度**: 部分 — Statistics 与能力属性表✓; 能力细节正文 missing (页面正文仅 "The Berserker goes berserk")。

### 13. DarkPrince (Dark Prince)
- **URL/标题**: https://clashroyale.fandom.com/wiki/Dark_Prince/Hero — 标题 "Dark Prince/Hero"
- **基础信息**: Elixir 4, Ability 3, Epic, Troop, 2026-05-04
- **Ability: Destructive Dismount** — "Dismount doing damage, attacking on foot while Rhino charges buildings."
  - 原文: "After pressing the ability, the Dark Prince will jump off the Rhino, dealing spawn damage on impact. The Rhino charges towards buildings, while the Dark Prince deals splash damage to troops but loses his charge in the process."
  - Rhino 属性表: Cost 3 · Hit Speed 1.6s · Range Melee: Short(0.8) · Speed Medium(60) · Charge Speed Very Fast(120) · Charge Range 2.5 · Target Buildings · Ground
- **Statistics (主体)**: Hit Speed 1.4s · First Hit 0.6s · Speed Medium(60) · Deploy 1s · Range Melee: Medium(1.2) · Splash Radius 1.1 · Ground · x1; Charge: Very Fast(120) · Charge Range 3
- **等级表** (L6-16, HP/ShieldHP/AreaDmg/ChargeDmg/DPS/RhinoHP/RhinoDmg/RhinoSpawnDmg/RhinoChargeDmg):
  6: 745/159/165/330/126/842/111/191/222 · 7: 820/175/182/363/140/926/122/210/245 · 8: 902/192/200/400/153/1019/134/231/269 · 9: 992/212/220/440/169/1121/148/254/296 · 10: 1091/233/242/484/186/1233/163/279/325 · 11: 1200/256/266/532/204/1356/179/307/358 · 12: 1320/282/293/585/225/1492/197/338/394 · 13: 1452/310/322/644/247/1641/217/371/433 · 14: 1597/341/354/708/272/1805/238/409/476 · 15: 1757/375/389/779/299/1985/262/449/524 · 16: 1933/412/428/857/329/2184/288/494/577
- **完备度**: 完整 (等级表自 L6 起)。

### 14. Balloon
- **URL/标题**: https://clashroyale.fandom.com/wiki/Balloon/Hero — 标题 "Balloon/Hero"
- **基础信息**: Elixir 5, Ability 2, Epic, Troop, 2026-04-06
- **Ability: Coffin Cadets** — "A Skeletrooper soars to the nearest enemy on the ground, dealing landing damage."
  - 原文: "Once pressed, a skeletropper will fly towards the closest ground unit within 6 tiles, deling spawn damage. The skeletropper will the attack, dealing less damage to the crown tower."
  - Skeletrooper 属性表: Cost 2 · Hit Speed 1.1s · Speed Very Fast(120) · Range 6.5 · Skeletrooper Range Melee: Short · Crown Tower Damage −90% · Target Ground · x1; Bomb: Death Damage Splash Radius 3 · Deploy 3s · Target Air & Ground
- **Statistics (主体)**: Hit Speed 2s · First Hit 0.2s · Speed Medium(60) · Deploy 1s · Range Melee: Short(0.1) · Target Buildings · x1 · Air
- **等级表** (L6-16, HP/Dmg/DPS/DeathDmg/SkelHP/SkelDmg/SkelDPS/LandingDmg/LandingCTD):
  6: 1043/397/198/149/294/127/115/163/17 · 7: 1147/437/218/164/323/139/126/180/18 · 8: 1261/481/240/180/355/153/139/198/20 · 9: 1388/529/264/198/391/169/153/217/22 · 10: 1526/582/291/218/430/185/168/239/25 · 11: 1679/640/320/240/473/204/185/263/27 · 12: 1847/704/352/264/520/224/203/289/30 · 13: 2032/774/387/290/572/247/224/318/33 · 14: 2235/852/426/319/630/272/247/350/36 · 15: 2458/937/468/351/693/299/271/385/40 · 16: 2704/1031/515/387/762/329/299/424/43
- **完备度**: 完整 (等级表自 L6 起)。

### 15. BarbarianBarrel (Barbarian Barrel) — 唯一的 spell Hero
- **URL/标题**: https://clashroyale.fandom.com/wiki/Barbarian_Barrel/Hero — 标题 "Barbarian Barrel/Hero" (Type: Spell, Epic; wiki trivia: "This is the only hero, that is a spell")
- **基础信息**: Elixir 2, Ability 1, 2026-03-02
- **Ability: Rowdy Reroll** — "Barrels down the lane a second time for maximum impact."
  - 原文: "When activatet, the Barbarian Barrel will roll for a second time, while healling the barbarian for 50% of the damage."
  - 属性表: Cost 1 · Range 3 (2026-05-04 由 4 削为 3) · Width 2.6 · Damage Healed 50% · Target Ground
- **Statistics**: Barrel: Cost 2 · Range 4.5 · Width 2.6 · Ground · Spell · Epic。Barbarian: Hit Speed 1.3s · First Hit 0.4s · Speed Medium(60) · Deploy 0.5s · Range Melee: Short(0.5) · x1
- **等级表** (L6-16, BarrelAreaDmg/BarbHP/BarbDmg/DPS/CTD):
  6: 144/445/119/85/72 · 7: 158/489/131/93/79 · 8: 174/538/144/102/87 · 9: 192/592/159/113/96 · 10: 211/651/175/125/105 · 11: 232/716/192/137/116 · 12: 255/788/211/150/128 · 13: 281/866/232/165/140 · 14: 309/953/256/182/154 · 15: 340/1048/281/200/170 · 16: 374/1153/309/220/187
- **完备度**: 完整 (等级表自 L6 起)。

### 16. EliteArcher (Elite Archer) — **missing**
- 尝试: `wiki/Elite_Archer`、`wiki/Elite_Archer/Hero`、`wiki/Elite_Archers/Hero` 均为占位空页 ("Create the page"); `Special:Search` 亦无该卡任何页面。
- 结论: 该 wiki 上 Elite Archer (含 Hero 变体) 不存在页面。**missing** (已留 `docs/_elite_EliteArcher.404.txt` 记录取证)。

### 17. IceGolemite (Ice Golemite) — **missing**(最可能对应 Ice Golem/Hero,已抓)
- 尝试: `wiki/Ice_Golemite`、`wiki/Ice_Golemite/Hero` 均为占位空页; 搜索 "Golemite" 只命中 Golem/Elixir Golem 相关讨论页, 无 "Ice Golemite" 卡页。
- 可行对应: **https://clashroyale.fandom.com/wiki/Ice_Golem/Hero** — 标题 "Ice Golem/Hero" (Heroes 页列出的 "Ice Golem" Hero 变体), 已抓取存为 `docs/_elite_IceGolem.txt`:
  - Elixir 2, Ability 2, Rare, Troop, 2026-01-05
  - **Ability: Snowstorm** — "Conjures a blizzard that damages and slows enemies." 原文: "After a 1-second delay, the Hero Ice Golem activates his 'Blizzard' ability, which generates a 4-tile aura radius that has 3 pulses, with each one slowing down enemy troops and dealing damage."
    - 属性表: Cost 2 · Radius 4 · Slowdown 30% · Blasts 3 · Slowdown Duration 2s · Target Air & Ground
  - Statistics: Hit Speed 2.5s · First Hit 1.5s · Speed Slow(45) · Deploy 1s · Range Melee: Medium(1.2) · Target Buildings · x1
  - 等级表 (L3-16, HP/Dmg/DPS/DeathDmg/BlizzardPulseDmg): 3: 613/39/15/39/32×3(96) · 4: 675/43/17/43/35×3(105) · 5: 742/47/18/47/39×3(117) · 6: 817/52/20/52/43×3(129) · 7: 898/57/22/57/47×3(141) · 8: 988/63/25/63/52×3(156) · 9: 1087/69/27/69/57×3(171) · 10: 1195/76/30/76/63×3(189) · 11: 1315/84/33/84/69×3(207) · 12: 1447/92/36/92/76×3(228) · 13: 1591/102/40/102/83×3(249) · 14: 1750/112/44/112/92×3(276) · 15: 1925/123/49/123/101×3(303) · 16: 2118/135/54/135/111×3(333)
  - 2026-08-04 平衡: 第 3 次脉冲 freeze→slowdown、blast 伤害 +82%、ability 改单次使用。
- 若 "Ice Golemite" 是wiki 上尚未收录的另一张卡, 则维持 **missing**; 已取证 `docs/_elite_IceGolemite.404.txt`。

---

## 数据完备度总览

| # | 卡 | 文件 | 页面标题 | Statistics | 能力描述原文 | 状态 |
|---|---|---|---|---|---|---|
| 1 | Knight | `_elite_Knight.txt` | Knight/Hero | ✓ 全 | ✓ | 完整 |
| 2 | Musketeer | `_elite_Musketeer.txt` | Heroic Musketeer | ✓ 全 | ✓ | 完整 |
| 3 | MiniPekka | `_elite_MiniPekka.txt` | Heroic Mini P.E.K.K.A. | ✓ 全(至L21) | ✓ | 完整 |
| 4 | Valkyrie | `_elite_Valkyrie.txt` | Valkyrie/Hero | ✓ 全 | 属性表✓/正文✗ | 部分 (stub 页) |
| 5 | Wizard | `_elite_Wizard.txt` | Wizard/Hero | ✓ 全 | ✓ (费用 2/1 矛盾已注) | 完整 |
| 6 | Bowler | `_elite_Bowler.txt` | Bowler/Hero | ✓ (L6-16) | ✓ | 完整 |
| 7 | Giant | `_elite_Giant.txt` | Giant/Hero | ✓ 全 | ✓ | 完整 |
| 8 | Goblins | `_elite_Goblins.txt` | Goblins/Hero | ✓ 全 | ✓ | 完整 |
| 9 | MegaMinion | `_elite_MegaMinion.txt` | Mega Minion/Hero | ✓ 全 | ✓ | 完整 |
| 10 | IceWizard | `_elite_IceWizard.txt` | Ice Wizard/Hero | ✗ | ✗ | **missing** ("Coming soon...") |
| 11 | Tombstone | `_elite_Tombstone.txt` | Tombstone/Hero | ✓ (Queen 仅等级表) | ✗ 正文 | 部分 |
| 12 | Berserker | `_elite_Berserker.txt` | Berserker/Hero | ✓ 全 | 属性表✓/正文✗ | 部分 |
| 13 | DarkPrince | `_elite_DarkPrince.txt` | Dark Prince/Hero | ✓ (L6-16) | ✓ | 完整 |
| 14 | Balloon | `_elite_Balloon.txt` | Balloon/Hero | ✓ (L6-16) | ✓ | 完整 |
| 15 | BarbarianBarrel | `_elite_BarbarianBarrel.txt` | Barbarian Barrel/Hero | ✓ (L6-16) | ✓ | 完整 |
| 16 | EliteArcher | `_elite_EliteArcher.404.txt` | — | ✗ | ✗ | **missing** (wiki 无页面) |
| 17 | IceGolemite | `_elite_IceGolemite.404.txt` + `_elite_IceGolem.txt` | Ice Golem/Hero | ✓ 全 | ✓ | missing 卡名本体; Ice Golem/Hero 完整 |

Hero 发布史 (Heroes 页): 2025-12-01 首批 Knight/Giant/Mini P.E.K.K.A./Musketeer → 2026-01-05 Wizard/Ice Golem → 02-02 Goblins/Mega Minion → 03-02 Barbarian Barrel/Magic Archer → 04-06 Balloon → 05-04 Dark Prince/Bowler → 06-01 Tombstone → 08-03 Berserker/Valkyrie。(Magic Archer 亦为 Hero, 但不在本次 17 卡清单内, 未抓单页。)

## 约束遵守说明

- 全部新增文件均在 `docs/` 下, 前缀 `_elite_`; 未覆盖/移动/删除任何既有 `_fp_*`、`_hero_1..7`、`_evo_*`、`_spell_*`、`*_evo` 文件。
- 两次探测用的临时空页 `_page_N.txt` 均已即时重命名或删除, 未留 `_page_*` 残留 (验证: `ls docs/_page_*` 为空)。
