# 2025 新觉醒 7 张 — 实现规格（Fandom CDP 采集 2026-09-03）

> 周期表已更新（evolutions.py，42 张闭环）；本 7 张快照无 evolvedSpellsData，需要数据层新建 + 引擎钩子。
> 来源：docs/_page_1..7.txt（原始抓取，**已于 2026-09-14 清理**）、re/fandom_stats/evo_42.json、evo7_mechanics.json
>
> **M6 实现状态（2026-09-03）**：✅ 7 张全部落地。
> - 数据层：`src/clasher_new/evo_2025_data.py`（evolvedSpellsData 等价数据 + evo2025Hooks 钩子参数包 +
>   Souldier/GeneralGerry/Shadow 派生角色注册；数值来源逐条标注 [Fandom]/[内存]/[假设]）
> - 注入点：`card_utils.py` 末尾 `evo_2025_data.apply(...)`；透传：`evolutions.M5_EVO_PASSTHROUGH += evo2025Hooks`
> - 引擎钩子：`battle.py`（首击面纱/起飞落地/减速箭/亡影/气流/热生成 + Gerry 阵亡清场 +
>   AreaEffect 法术穿透亡影）+ `card_mechanics.py Ghost`（隐身/显形/召唤）
> - 验收：`scripts/test_m4_evo7.py` 51 断言全绿；回归 test_m2 72/72、test_m3_evo 62/62、batch_smoke 195/195

## 1. Evolved Furnace（觉醒熔炉）✅
- 周期 2，4 费，Rare，2025-08-04 上线（gamedata 卡名 = **FirespiritHut**，周期表已加别名键）
- 数值：与原版完全相同（identical stats）[Fandom _page_1]
- 机制：攻击期间生成速度提升至 **2.4s**（Hot Spawn Speed 2.4s [Fandom Evolution Attributes]），
  火灵**从侧面生成**（首个左侧 → 左右交替 [Fandom 策略节]）
- 实现：`Building._evo_building_tick` 热生成循环（基础建筑出兵循环引擎本未建模, 觉醒钩子自带）；
  侧向/前向偏移 0.8 格 [假设]

## 2. Evolved Baby Dragon（觉醒幼龙）✅
- 周期 2，4 费，Epic，2025-09-01 上线
- 数值：与原版相同；伤害 ×1.04（6/7/2026 平衡 [Fandom History]；基础伤害载体=弹道）
- 机制：攻击时产生**气流**：8×9 格区域友军 +30% / 敌军 -30%（1/12/2025 平衡后 30%）；
  死后残留 2s [Fandom _page_2]
- 实现：`Troop._evo2025_gust_tick`（攻击中每 0.25s 脉冲 [假设节拍]）；死亡 → EvoEffectZone 残留；
  8×9 矩形 → 圆形半径 4.0 [假设]

## 3. Evolved Skeleton Army（觉醒骷髅军团）✅
- 周期 2，3 费，Epic，2025-10-06 上线
- 数值：15 只骷髅散布 + **General Gerry** 后排；Gerry HP/护盾/伤害 = **32（等级无关）**
  [内存 0x76c726384680：Hitpoints/ShieldHitpoints/Damage 三键均 32、10 行同值、Range=1600、Speed=90、
  TID_CHARACTER_SKELETON_ARMY_EV1_GENERAL 实证]；Gerry Hit Speed 1.0s / First Hit 0.5s / 射程 1.6 [Fandom]
- 机制：Gerry 存活时骷髅死亡 → **亡影**（不可被部队/建筑/塔选取 + 无限 HP，**法术可伤害**）；
  Gerry 死亡 → 亡影全灭 + 存活骷髅不再转化 [Fandom _page_3 策略节]
- 实现：亡影 = SkeletonArmy_EV1_Shadow（targetable=False + invincible=True；
  `AreaEffect._pulse` 对亡影 `pierce_invincible=True`；法术可伤血池 999 [假设]）；
  Gerry 阵亡 → `BattleState.on_death` 清场；亡影伤害拷贝官方 Skeleton 行（lv11=81）

## 4. Evolved Royal Ghost（觉醒皇家幽灵）✅
- 周期 2，3 费，Legendary，2025-10-17 上线
- 机制：显形时召唤 **2 名 Souldier**（召唤伤害 81@lv11 [Fandom Souldiers 表, Legendary 轴 67..130]）；
  幽灵本体数值与基础一致（[Fandom 表] HP 1000@lv9 = 官方 characters 行 ✓）
- ⚠️ 存活时长口径冲突：卡面引文「会消失」vs Fandom 策略节「do not despawn naturally」——
  **取策略节 = 不自然消失**（冲突记录于此, 待 L4）
- 实现：`card_mechanics.Ghost`（补全基础隐身：部署即隐身/攻击显形/脱战再隐身, 觉醒延迟 2.0s
  [Fandom History 2/3/2026]、基础 1.8s）；召唤伤害半径 1.0 [假设]

## 5. Evolved Royal Hogs（觉醒皇家野猪）✅
- 周期 2，5 费，Rare，2025-11-03 上线
- 数值：本体同原版；落地伤害 [Fandom 表 115@lv11] × 两次平衡（2/3/2026 -27%、4/5/2026 -49%）
  → **lv11 ≈ 43**（per-level 数组见 evo_2025_data.py）
  ⚠️ 口径说明：该页为 stub, 表值可能未随平衡更新（若表值已含削弱则为重复削弱, 备查记录在数据层）
- 机制：对建筑**起飞**（地面部队无法选取）, 受击或攻击时**落地 AoE**
- 实现：`_apply_evolution` 置 `data.is_air_unit=True`（跳河能力关闭防状态污染）；
  `take_damage` 落地钩子 + `_evo_on_attack` 攻击落地；落地半径 1.5 [假设]

## 6. Evolved Minion Horde（觉醒亡灵大军）✅
- 周期 **1**，5 费，Common，2026-04-06 上线
- 机制：每个成员**受到的第一次攻击使其短暂无敌**（Dark Elixir 面纱）
- 数值：面纱时长无直接字段 [假设 0.67s —— Fandom History「invisible hit speed multiplier
  x0.5→x0.67」解读为 面纱时长 = hitSpeed(1.0s) × 倍率；首击本身被闪避]
- 实现：`Entity.take_damage` 首击面纱（每成员一次, `_evo_veil_*` 状态）+ `Entity.update` 计时

## 7. Evolved Princess（觉醒公主）✅
- 周期 2，3 费，Legendary，2026-06 上线
- 数值：per-level 表 Fandom 未提供 → 沿用基础卡曲线（快照 Princess 本体已是 AoE+对空：
  projectileData radius 2000 + tidTarget AIR&GROUND, 规格书「单体对地→AoE+对空」为表述误差）
- 机制 [Fandom _page_7]：**减速箭**——首发减速（3 格 / 30% / Ice Wizard 同款）, 之后每 2 发
  （4/8/2026 平衡：every 2 hits from every 3）, 时长 5.5s（平衡前 7s）；死亡留下 3 格减速领域
- 实现：`_evo_on_attack` 减速箭（计数器 cadence）+ `_evo_on_death` 减速领域；
  领域时长与攻击减速同口径取 5.5s [假设：平衡同时覆盖领域]

## 数据缺口清单（M6 后结算）
- [x] 7 张 per-level 数值：Furnace/BabyDragon/SkeletonArmy/Ghost/RoyalHogs = Fandom 表全量落库；
  Princess 无表 → 基础卡曲线（机制数值 Fandom 全量）；MinionHorde = 同原版
- [x] Souldier HP/伤害/召唤伤害：Fandom 表 lv9-16 全量（67/74/81/89/98/108/119/130, Legendary 轴）
- [x] RoyalHogs 落地伤害：Fandom 表 × 平衡系数（lv11≈43）；飞行触发=部署即飞（Fandom 明示）
- [x] BabyDragon 气流：幅度 ±30%、残留 2s [Fandom]；半径 8×9→4.0 [假设]
- [x] MinionHorde 无敌时长：0.67s [假设, 由平衡注记推得]
- [ ] 待 L4 对拍：面纱时长、气流半径/节拍、落地/召唤/侧向偏移半径、Princess 领域时长、
  亡影法术血池、Souldier 存活时长口径冲突

## 优先级建议
按上线时间倒序实现（Princess/MinionHorde 最新=当前环境最常见）：Princess → MinionHorde → RoyalHogs → Ghost → SkeletonArmy → BabyDragon → Furnace
（M6 已按此完成全部 7 张）
