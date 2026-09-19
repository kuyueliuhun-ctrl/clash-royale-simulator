[reconcile] 掩码 vs 引擎 · 逐格穷举 18x32 x 双方；card_level=11
[reconcile] 卡牌 24 张：Knight, Giant, Miner, Musketeer, Minions, Archer, MiniPekka, Cannon, Mortar, XBow, Arrows, Fireball, Zap, Poison, Tornado, Earthquake, Freeze, Rage, Heal, Lightning, Log, BarbLog, GoblinBarrel, Mirror

| 卡牌 | pid | 掩码可放格数 | 掩码否/引擎可 (过严) | 掩码可/引擎否 (缺口) | 异常 |
|---|---|---|---|---|---|
| Knight | 0 | 224 | 0 | 0 | 0 |
| Knight | 1 | 224 | 0 | 0 | 0 |
| Giant | 0 | 224 | 0 | 0 | 0 |
| Giant | 1 | 224 | 0 | 0 | 0 |
| Miner | 0 | 224 | 284 | 0 | 0 |
| Miner | 1 | 224 | 284 | 0 | 0 |
| Musketeer | 0 | 224 | 0 | 0 | 0 |
| Musketeer | 1 | 224 | 0 | 0 | 0 |
| Minions | 0 | 224 | 0 | 0 | 0 |
| Minions | 1 | 224 | 0 | 0 | 0 |
| Archer | 0 | 224 | 0 | 0 | 0 |
| Archer | 1 | 224 | 0 | 0 | 0 |
| MiniPekka | 0 | 224 | 0 | 0 | 0 |
| MiniPekka | 1 | 224 | 0 | 0 | 0 |
| Cannon | 0 | 220 | 0 | 0 | 0 |
| Cannon | 1 | 220 | 0 | 0 | 0 |
| Mortar | 0 | 220 | 0 | 0 | 0 |
| Mortar | 1 | 220 | 0 | 0 | 0 |
| XBow | - | - | - | - | 前置失败 KeyError |
| Arrows | 0 | 36 | 540 | 0 | 0 |
| Arrows | 1 | 36 | 540 | 0 | 0 |
| Fireball | 0 | 44 | 532 | 0 | 0 |
| Fireball | 1 | 44 | 532 | 0 | 0 |
| Zap | 0 | 576 | 0 | 0 | 0 |
| Zap | 1 | 576 | 0 | 0 | 0 |
| Poison | 0 | 576 | 0 | 0 | 0 |
| Poison | 1 | 576 | 0 | 0 | 0 |
| Tornado | 0 | 576 | 0 | 0 | 0 |
| Tornado | 1 | 576 | 0 | 0 | 0 |
| Earthquake | 0 | 576 | 0 | 0 | 0 |
| Earthquake | 1 | 576 | 0 | 0 | 0 |
| Freeze | 0 | 576 | 0 | 0 | 0 |
| Freeze | 1 | 576 | 0 | 0 | 0 |
| Rage | 0 | 576 | 0 | 0 | 0 |
| Rage | 1 | 576 | 0 | 0 | 0 |
| Heal | 0 | 576 | 0 | 0 | 0 |
| Heal | 1 | 576 | 0 | 0 | 0 |
| Lightning | 0 | 576 | 0 | 0 | 0 |
| Lightning | 1 | 576 | 0 | 0 | 0 |
| Log | 0 | 576 | 0 | 0 | 0 |
| Log | 1 | 576 | 0 | 0 | 0 |
| BarbLog | 0 | 576 | 0 | 352 | 0 |
| BarbLog | 1 | 576 | 0 | 352 | 0 |
| GoblinBarrel | 0 | 576 | 0 | 0 | 0 |
| GoblinBarrel | 1 | 576 | 0 | 0 | 0 |
| Mirror | 0 | 576 | 0 | 352 | 0 |
| Mirror | 1 | 576 | 0 | 352 | 0 |

## 有差异的条目（逐格样本）

- **Miner** pid=0: 过严 284 格 / 缺口 0 格 / 异常 0
- **Miner** pid=1: 过严 284 格 / 缺口 0 格 / 异常 0
- **Arrows** pid=0: 过严 540 格 / 缺口 0 格 / 异常 0
- **Arrows** pid=1: 过严 540 格 / 缺口 0 格 / 异常 0
- **Fireball** pid=0: 过严 532 格 / 缺口 0 格 / 异常 0
- **Fireball** pid=1: 过严 532 格 / 缺口 0 格 / 异常 0
- **BarbLog** pid=0: 过严 0 格 / 缺口 352 格 / 异常 0
- **BarbLog** pid=1: 过严 0 格 / 缺口 352 格 / 异常 0
- **Mirror** pid=0: 过严 0 格 / 缺口 352 格 / 异常 0
- **Mirror** pid=1: 过严 0 格 / 缺口 352 格 / 异常 0

[reconcile] 合计：过严 2712 格，缺口 1408 格，异常 0
