# 人工评审队列（P0-1 基本信息录入 v4 — L1 结论与剩余未知）

> **M1 进展（2026-09-02）**：族 1（递增伤害）+ 族 2（弹道生成链）已实现并通过 19/19 测试
> （`scripts/test_m1.py`），递增数值对照 Fandom 实测验证。详见 `P0-mechanics-plan.md` 进度标注。
> M1 过程中顺带修复：① `fastcore` 非必要依赖移除（此前阻塞全部导入）；② `resolve_collisions`
> 双建筑重叠除零保护；③ Firecracker 的 Card 构造崩溃（弹丸行 damage_per_level=None）。
> 仍开放：④ Building 构造 `persistent` 参数错位（部署建筑寿命衰减失效）；⑤ 建筑卡未走
> per-level 缩放（InfernoTower 伤害 17 vs 官方同级 51，约 3 倍偏弱）——建议并入 M2。

> 政策：特殊机制以 Null 服为基准；数值按官方；拿不准交人类。
> 配套：`docs/card_registry.json`（150 条主册）、`docs/card_coverage.md`、`scripts/coverage.py`（仓库为主 + data_official 增量合并，可重复执行）。

## 当前状态（v4）

| 状态 | 数量 | 说明 |
|---|---|---|
| implemented | 47 | 已实现 |
| data_ready | 60 | 官方数值就绪 |
| needs_review | 15 | **全部为 STATS_UNRESOLVED**（见 B，已确认任何公开源都没有） |
| temporary_event | 14 | 含 5 张人工确认限时卡 |
| variant | 6 | 变体/形态条目 |
| tower | 4 | 塔 |
| missing_from_snapshot | 4 | Ronin、Vines、Elite Berserker、Elite Valkyrie |

主册健康度：103 张带官方 lv11 数值；141 张带机制字段；34 觉醒 + 11 技能卡标记。

## A. ✅ 已关闭：限时卡确认（5 张 → temporary_event）

## B. L1 结论（重要更正）：cr-api-data 已过时，对本项目无增量

实测（2026-xx 拉取 `RoyaleAPI/cr-api-data@master/docs/json/`）：

| 文件 | cr-api-data | 仓库现有 | 结论 |
|---|---|---|---|
| cards_stats_characters.json | 120 条 | **125 条** | 仓库更新，缺的 9 张新角色两边都没有 |
| cards_stats_spell.json | 71 条 | 71 条 | 完全相同 |
| cards.json（元数据） | 120 条 | **123 条** | 仓库更全 |

**处理**：加载器已改为「仓库为主 + data_official 增量并集」（`coverage.py` 自动检测），cr-api-data 数据保留在 `data_official/` 但无增量贡献；`cards_evo.json`（8 个首批觉醒，`evolution_level` 全=1，非周期字段）与 `cards_i18n.json`（多语言卡名）留作后续参考。

**15 张 STATS_UNRESOLVED 的解决路径**（按优先级）：
1. **L2**：提取新版 Null 服 gamedata（含 2025-2026 卡的 summonCharacterData 与数值表）——与既有管线一致，一并解决 4 张快照缺失卡
2. **人工录入**：Fandom wiki / RoyaleAPI 卡牌页有全部现役卡的 11 级数值，可人工填表
3. **近似推导**（兜底）：用同稀有度已知卡的每级乘数曲线标定，主册标注置信度

## C. 待拍板决策项

1. **15 张卡的数值来源**：选 L2 / 人工录入 / 近似推导？（推荐：有 Null 服更新包就走 ①，否则 ② 只录卡组会用到的卡）
2. ✅ **三张 2026 卡基本信息录入完成**（v6 主册，`confirmed_live` 状态）：
   - **Hero Berserker（精英狂战士）**：2 费 Common/Ground，技能=Berserk Mode（提高攻速+保护窗口内血量不低于 1）；属性继承基础卡 Berserker
   - **Hero Valkyrie（精英女武神）**：4 费 Rare/Ground，技能=旋转机动（在近身敌军间旋转移动造成范围伤害，期间可被击杀）；属性继承基础卡 Valkyrie
   - **觉醒野蛮人精锐**：Elite Barbarians 觉醒形态，技能=Rage Spears（近战接敌前投矛：伤害+落点狂暴轨迹增幅友军）；**Evolution Cycle=1**（首个确认的周期数据点）；基础数值不变（6 费 2 单位）
   - 数据源：RoyaleZone + gamer.org（2026-09-02 经 CDP 浏览链抓取，来源 URL 已入主册 SOURCE 标记）
   - 仍待 L2：两张 Hero 的技能参数（窗口时长/倍率）、觉醒的 evolvedSpellsData
   - **类别确认**：2026 Hero（英雄/精英）卡 = 基础卡+主动技能，费用/属性继承基础卡——机制族 6 的实现模型可直接套用
3. **GlobalLightning**：并入排除名单？（快照 1 费 vs 官方 6 费，主册已按官方标值）
4. **KingTower 2100 硬编码**：改查 per-level 表（推荐），需确认无下游依赖
5. **觉醒周期表**：34 张 1/2 回归属地（cards_evo.json 已证无此字段；人工提供或 L4 实证）
6. **递增伤害语义**：先按「毫秒+百分比增量」实现并标注置信度，M1 后 L4 对拍定案
