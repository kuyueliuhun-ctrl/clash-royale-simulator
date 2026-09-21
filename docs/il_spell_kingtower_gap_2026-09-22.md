# 「法术砸敌方国王塔」取证：**不是缺输入，是掩码 EV 闸门漏了「只罩王塔」这一类**（2026-09-22）

> 触发：用户看 8702（臂 R=0.25）第一局的 dashboard，发现「火球砸国王塔，人类对局不会这么打」，
> 问是不是网络少输入了什么。结论：**观测里塔是在的**（而且落点头部对塔格有强先验），
> 真正的开口在**动作合法性**：`_spell_tower_ev_illegal` 只对「罩到敌方**公主塔**」的落点做判定。
>
> 仪器：`scripts/il_probe_kingtower_cast.py`（新，只读旁挂，不改生产代码）；
> 录像取证：两个并行子智能体（人类侧 / 各臂侧），本文件只采信**我复核过的**读数。

## §1 现象（录像取证，可复算）

自对弈 10 局/臂（`runs/il_readout_mix*/replays/`，schema 5），p0 = 被测 IL 策略、
p1 = 同一 ckpt 的冻结副本（`--opponent self`；录像里的 `fl@0` 只是**记录器标签**，
不是 FirstLight 对手 —— 这点要跟子智能体的初版描述区分）。

| 臂 | 局 | 帧 | t(s) | 卡 | 本地落点 | 世界落点 | 到王塔中心 | 王塔掉血 |
|---|---|---|---|---|---|---|---|---|
| mixR00 | 2 | 0 | 0.5 | Fireball | (9,25) | (9.5,25.5) | 3.54 | **206** |
| mixR00 | 7 | 0 | 0.5 | Fireball | (9,25) | (9.5,25.5) | 3.54 | **206** |
| mixR025 | 2 | 1 | 1.0 | Fireball | **(9,29)** | (9.5,29.5) | **0.71** | **206** |
| mixR025 | 7 | 0 | 0.5 | Fireball | (9,29) | (9.5,29.5) | 0.71 | **206** |
| mixR025 | 9 | 1 | 1.0 | Fireball | (11,27) | (11.5,27.5) | 2.92 | **206** |
| mixR10 | 6 | 4 | 2.5 | Fireball | (9,25) | (9.5,25.5) | 3.54 | **206** |
| mixR10 | 7 | 2 | 1.5 | Fireball | (9,25) | (9.5,25.5) | 3.54 | **206** |

* **7/7 每次恰好 206**（= Fireball lv11 `damage 688 × crown_tower_percent 0.3`），
  命中边界 3.9 = `radius 2.5 + KingTower.collision_radius 1.4`（引擎实测，非我推的）；
* **7/7 都在开局**：5 次就是该局**第一个动作**，全部 `t ≤ 2.5 s`；
* 只出现在 `mixR00/R025/R10` 三臂（其余 9 臂 **0/90**；其中「开局手牌含 Fireball」的 12 局里占 **5 局 = 42%**，
  而另外 7 个同样含 Fireball 的臂是 **0/28**）⇒ 是**特定臂稳定的行为偏置**，不是随机偶发；
* 塌点臂 `mixR337` ≡ `save_m0`（同一录像，md5 相同）0 次 —— 但它 10 局总共只出 3 张牌，**没有分母，不能反证**。

## §2 根因：闸门的**提前 return**（`src/clasher_new/rl/action_mask.py:158-217`）

```python
def _spell_tower_ev_illegal(battle, player_id, card_name, pos, card_info=None):
    ...
    hits_princess = False
    for tid in ((1, 2) if opp == 1 else (3, 4)):     # ← 只看**公主塔**
        ... if pos.distance_to(tw.position) <= radius + col: hits_princess = True
    if not hits_princess:
        return False                                  # ★ 落点罩不到公主塔 ⇒ 一律放行
    ...
```

docstring 的依据是「王塔在公主塔后面，砸到公主塔必含王塔误差，不重复判王塔」——
这句话对**正面贴公主塔**的落点成立，但对**两座公主塔之间的中路口**（x≈9，y≈25~29）**不成立**：
那里到两座公主塔 5.0~6.5（> 2.5+1.4），到王塔却 ≤ 3.9 ⇒ **只罩王塔、罩不到任何公主塔** ⇒ `not hits_princess`
⇒ **直接放行**（连后面的「半径内有无非塔目标」「折费 EV」两条都没机会跑）。

`legal_cells` 对伤害型法术还有一条 `_spell_has_enemy_target`（溅射内必须有存活敌方目标）——
**塔也算目标**（`action_mask.py:97-109`），所以"只罩王塔"的格子是**合法**的。

## §3 探针实证（`scripts/il_probe_kingtower_cast.py`，逐帧、同 seed、零漂移）

探针把 `il_readout_games.py` 当模块跑同一套 setup，只在 `FollowerPolicy.act` 外挂只读钩子：

**(a) `mixR00` 局2 **第一次决策**（t=0.0，5.0 圣水，对方还没落任何单位）**

| 项 | 读数 |
|---|---|
| 手牌 | `[IceWizard, Knight, Tesla, Fireball, Log]` |
| 槽位 argmax | **Fireball（slot 4）p=0.4249**（STOP≈0.0） |
| Fireball 合法格 | **44** |
| 合法集分类 | **只罩王塔 40** / 只罩公主塔 4 / 罩到部队 0 |
| 被选格 | **(9,25)**，`splash_princess=false`、`splash_king=true`、`splash_unit=false`、`has_enemy_target=true`、**`ev_gate_illegal=false`** ⇒ 合法 |
| `cell_head` 原始 top-1 | (9,0) = **自己**王塔格（非法）——落点头部对塔格有强先验 |
| plan 软偏置 | `focus_region=own_center` 中心 (9,20)，被选格 `cell_bias = 0.0`（**与 plan 无关**；关掉偏置 argmax 相同） |

⇒ **"纯砸王塔"（半径内没有任何敌方部队/建筑）** 被放行了，而这正是 9h 闸门要禁的行为。

**(b) `mixR025` 局2 第二次决策（t=0.5 —— 就是用户看到的那次）**

| 项 | 读数 |
|---|---|
| 手牌 / 场上 | 同上；**对方把 Knight 放在自己王塔后面 (9.5,31.5)** |
| 槽位 argmax | **Fireball p=0.3567**（STOP 0.1414） |
| Fireball 合法格 | **45**（只罩王塔 23 / 只罩公主塔 4 / 罩到部队 18） |
| `cell_head` 原始 top-1 | (14,25) = **敌方公主塔格**，logit **+1.02**，`ev_gate_illegal=**true**` ⇒ **被闸门禁掉** |
| 被选格 | **(9,29) = 敌方王塔格**，logit −4.24，`splash_unit=true`（罩到那个 Knight）+ `splash_king=true` ⇒ 合法 |
| plan | 同上，`cell_bias=0` ⇒ 无关 |

⇒ 这次是「闸门把公主塔格禁掉后，**剩下的最高分格正好是王塔格**」。
两种情形合起来：**闸门禁掉公主塔 → 模型被迫（且乐于）转向王塔**。

## §4 人类侧对照（子智能体取证，我复核口径）

`runs/_fl_il_bc/`（47,715 条标签 = train 38,838 + holdout 8,877；2200 局）：

| 口径 | Fireball 砸王塔 | 全部法术 |
|---|---|---|
| 落点距王塔中心 ≤ `_spell_radius_m`（掩码口径） | **4 / 361 = 1.11%** | 66 / 10,565 = **0.62%** |
| 引擎真伤口径 `≤ radius + 1.4`（只有 Fireball/Arrows 能扣王塔血） | **11 / 361 = 3.05%** | — |
| **原始回放 native 坐标**独立复核（绕过 `int()` 量化） | **18 / 727 = 2.48%**（≤3.9） | — |
| `Log` / `BarbLog` | **0 / 1590、0 / 1909** | — |
| **`t ≤ 5 s` 的法术** | **0 / 361** | **0 / 10,565** |
| **`t ≤ 5 s` 的任意出牌** | — | **1 / 47,715**（唯一的 t=5.0 s 是 Goblins） |

原始回放独立复核：2200 局里**最早**的一次 team 出牌是 **5.40 s**，0 局在 5 s 前出牌。

⇒ 两点结论：
1. **「人类不可能砸王塔」不成立**——人类确实有（~0.6%~3%，甚至有一条是满血王塔 + 双公主塔存活时打的），
   所以这**不是**「模型独有的非法动作」；
2. 但 **「开局第一帧就砸王塔」在人类数据里是 0 样本**（t≤5 s 出牌仅 1/47,715，且不是法术）——
   模型这 7 次全在 `t ≤ 2.5 s`，属于**分布外行为**，其合法性完全由 §2 的闸门漏洞提供。

## §5 判决与修法（**未改代码**；须先预注册 + R13 位图对账）

* **判定：不是"网络少输入"。** 观测里塔以 `entity_id=KingTower` + `is_opponent=1` + HP 百分比在 `grid` 里；
  掩码也知道它在哪（`king_cell_legal=true`）；连模型的 `cell_head` 都**偏好**塔格。
  缺的不是"塔这个输入"，而是**动作合法性里少了一条**：只罩王塔的落点没有被 EV 判定。
* **修法（单变量、零参数）**：把 `_spell_tower_ev_illegal` 的 `hits_princess` 扩成 `hits_any_tower`
  （王塔同用 `radius + collision_radius` 判定），并让残血加权把被罩到的王塔也纳入
  `tower_value_mult(king=True, princesses_alive=…)`。**不**动 `_spell_has_enemy_target`
  ⇒ 「打躲在王塔后面的部队」仍然合法（`mixR025` 局2 那种，condition 4 本来就是放行的）。
* **跑前写死的判据**（R3/R13/R15/R16）：
  1. **位图对账**：`scripts/_mask_diff_snapshot.py --compare` 逐位全等只允许出现在"只罩王塔"格上，
     且 `--selftest` 通过（R13）；
  2. **行为**：开局首动作落点不再出现 `splash_unit=false & splash_king=true`（探针逐帧复算）；
  3. **不回归**：`n_legal_cells ≥ 1` 恒成立（掩码不低于 STOP），10 局双侧行为带 `[18,28.5]` 出牌/局 不变；
  4. **负对照**：把闸门整个关掉（改前行为）必须复现 7 次砸王塔（同一 seed）。
* **未定（照实记）**：① `Log` 滚动弹的 `proj.radius = 0.0` 是设计还是漏读 `projectile_radius`——未定；
  ② 10ep 臂那次 t=37 s 的 Log 单次伤害归因未定（并发部队伤害污染）；
  ③ 模型"想砸塔"的内部意图无法从录像判定，本文件只给**机械成因**；
  ④ R00/R025/R10 三臂为何独有该偏置（相空间/训练池差异）未定。
