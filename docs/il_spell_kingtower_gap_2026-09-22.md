# 「法术砸敌方国王塔」取证：**不是缺输入，是掩码 EV 闸门漏了「只罩王塔」这一类**（2026-09-22）

> **状态：已修复（2026-09-22，见 §6；F1 = 王塔纳入 EV 闸门，F2 = 效果载体不算目标）**。
>
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
| 引擎真伤口径 `≤ radius + 1.4` | **11 / 361 = 3.05%** | — |

> ⚠️ **更正（2026-09-22，我自己复核后改口）**：本节初版写的「本仓引擎只有 Fireball/Arrows 能扣王塔血」**是错的**（来自一个子智能体的引擎扫描，我误采信）。我直接复算（`deploy_card` 到敌方王塔格 + `step` 8 s，读王塔 hp 差）：**Zap 57.6 / Snowball 58 / Poison 343.7 / Lightning 686.4 / Log 290 / Earthquake 171 / Vines 76.5 / Rage 53.7 / Freeze 34.5 / Tornado 0.6，全都扣王塔血**；Fireball 206 / Arrows 75 / Rocket 371 亦然。成因（子智能体另一路取证）：`battle.py` 只对王塔乘 `projectile_data.crown_tower_percent`，而 `card_utils.py` 在字段缺失时**回落 1.0**（真值只有 Fireball 0.3 / Arrows 0.2049 / Snowball 0.3 / Rocket 0.25），滚动弹分支根本不走这个乘子 ⇒ **本仓大部分法术对王塔打的是「满伤」**。这条只影响上表「引擎口径」那一行的解释，**§1–§3 的掩码/机制结论与 §6 的修复都不依赖它**（修复判据用的是掩码自己的 `_spell_tower_damage` 与几何）。
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

## §6 修复实施（2026-09-22，**已落代码**）与门禁

### 6.1 改动（两处，各自单一目的）

| # | 文件:位置 | 改法 | 证据 |
|---|---|---|---|
| **F1** | `src/clasher_new/rl/action_mask.py::_spell_tower_ev_illegal` | `hits_princess` → `hits_princess or hits_king`；王塔按名字查找（不写死 id）；只罩王塔时用 `tower_value_mult(ratio, king=True, princesses_alive=…)` 取倍数 | §2/§3 的机制取证 |
| **F2** | `src/clasher_new/rl/action_mask.py::_spell_has_enemy_target` / `_spell_covers_non_tower` | 新增 `_is_effect_body()`（`data.type ∈ {projectile, area_effect, bomb}`）并**跳过效果载体** | `mixR025` 局9：旧谓词把**对方 Log 的滚动弹**当成"敌方目标"⇒ 王塔空砸合法 |
| **F3（不变式）** | `src/clasher_new/rl/env_wrapper.py::get_action_mask_for` | 新增「合法槽必须至少有一个合法落点」；否则禁该槽 | 否则 F1/F2 之后 `act()` 会把 576 格全填 −1e9、argmax 落到格 0（角落）⇒ 整包被 `validate_bundle` 拒收 |

### 6.2 门禁结果（【R13】位图对账 + 【R8】回归测试）

* 快照：`docs/mask_snapshots/kingtower_before.npz`（改前）、`..._after.npz`（F1 后）、`..._after2.npz`（F1+F2 后）；
  各 **128 张**位图、`__meta_errors__ = 0`。`--selftest` **5/5 PASS**（判别力有负对照）。
* `--compare before after`：**24/128 张不同，逐格分类后 960 格全部落在唯一允许类别**（「罩到王塔 ∧ 罩不到公主塔 ∧
  无任何非塔目标」），**方向全为 True→False（只收紧）**、**0 条越界**；差异只出现在 **Fireball / Arrows**
  两张卡（掩码里唯一被 9h 闸门覆盖的两张）。`before → after2` 同样 960 格、0 越界。
* `--compare after after2`（F2 单独看）：**128/128 逐位全等** —— ⚠️ 这说明 **R13 的 128 状态语料里没有「效果载体」**
  （Log 滚动弹这类的覆盖率为 0）⇒ **该语料对 F2 是盲的**；F2 的门禁只能靠回归测试 + 真实录像，这是一条
  **已记录的语料覆盖缺口**（见 §7）。
* 回归测试：`scripts/selftest_spell_kingtower.py` **16 条断言全 PASS**，含
  ①空场「只罩王塔」格全部非法（修前全部合法）；②部队在半径内仍合法（不误伤"打躲在王塔后的部队"）；
  ③敌方公主塔格仍非法（原闸门不变）；④t≥120 放行不变；⑤开局 Fireball 槽被禁且 `any_legal` 仍真；
  ⑤c **真实 `LogProjectileRolling` 在场 ⇒ 王塔格非法**（前提断言确认真有 projectile 实体，防"假绿"）；
  ⑥源码白盒（`hits_king` / `_is_effect_body` / 提前 return 已改）。

### 6.3 行为结果（同 seed、同约定、10 局/臂；普查器 `scripts/il_kingtower_cast_census.py`）

事件口径 = 敌方王塔血量**恰好 −206**（Fireball lv11 对王塔精确伤害）+ 前 4 帧内有 p0 的 `FireballSpell` 在飞；
落点用「最后可见位置」与「外推点」两者到王塔距离的**较小值**判命中（只取任一都会漏）。

| 臂 | 改前 命中 | 改后（F1+F2） | 出牌/局（改前→改后） | 帧/局 | Fireball 出牌数 |
|---|---|---|---|---|---|
| `mixR00` | **2**（均 king_only） | **0** | 38.2 → 37.1 | 368.5 → 359.4 | 4 → 1 |
| `mixR025` | **3**（2 king_only + 1 unit） | **1**（该 1 次为 **unit**：对方 Knight 藏在王塔后，Fireball 是冲它去的） | 32.3 → 32.8 | 325.3 → 330.1 | 5 → 2 |
| `mixR10` | **2**（均 king_only） | **0** | 30.7 → 30.5 | 313.5 → 312.5 | 3 → 1 |

* **纯砸王塔（`king_only`）3 臂合计 6 → 0** ✅；仅剩的 1 次是「**打躲在王塔后的部队**」——
  这是 9h 闸门**特意**放行的类别（条件 4：有非塔目标即放行），要再收紧属于**另一条判据**（"打这个单位值不值"），
  不在本次修复范围内（见 §7）。
* **无塌陷**：三臂出牌/局仍在 30.5–37.1（人类带 [18,28.5] 之上，与改前同侧），`Xbow` 仍 **0**（C14 不变）。
* 逐帧旁证（探针）：`mixR00` 局2 **第一次决策**由「Fireball→王塔格」变为「Fireball 槽被禁 ⇒ 出 IceWizard 到自家后场」；
  `mixR025` 局9（那次唯一"理由"是对方 Log 滚动弹）同样变为「Fireball 槽被禁 ⇒ 出 IceWizard」。

## §7 本轮**新建**的开放项（都没动代码；各有判据前置）

1. **F3：`_spell_deals_damage` 的覆盖只有 4 张卡**（Fireball/Rocket/Arrows/Snowball）⇒ 8h 空砸闸门与 9h 砸塔 EV 闸门
   对 **Zap/Log/BarbLog/Poison/Lightning/Earthquake/Tornado/Vines/Freeze 全部不生效**（掩码里另有 `spell_module._deals_damage`
   是第二套口径，两者不一致）。而我实测这些卡**确实扣王塔血**（Zap 57.6 … Lightning 686.4）。
   ⇒ 「谁该受闸门约束」当前**取决于掩码的漏读**，而不是卡的实际伤害。**要不要统一**须单独预注册
   （注意：用户提的「小法术过牌」正发生在这些**不受约束**的卡上 ⇒ 一刀切收紧会**误伤人类真实行为**）。
2. **F4：`_spell_tower_damage` 对 Log / Lightning 读 0**（实测 290 / 686.4）⇒ 即便闸门覆盖到，也会走
   `if dmg <= 0: return False`（"无标定 ⇒ 放行"）**后门**。先修仪器再谈收紧。
3. **Log / BarbLog 的半径口径**：`_spell_radius_m('Log') = 1.95` 只是**半宽**，伤害区是从落点沿 +y **10.1 / 4.5 格**的走廊
   （`battle.py` 滚动弹分支）⇒ 用「落点距塔中心 ≤ 半径」判它们是否会打到塔，**几何上是错的**（本次修复未依赖该口径）。
4. **觉醒/进化（用户提的"过觉醒轮次"）**：引擎**有**（`evolutions.py` 42 张周期表 + `battle.py` 触发），
   但 **`rl/` 全目录对 `evo_slots/evo_plays/hero_slots/evolution_state` 的引用数 = 0** ⇒ **本仓 RL 对局里觉醒永不发生**；
   且 `rl/observation.py::observe()` **只有 `grid/hand/elixir/next_card/time` 五个键，没有任何觉醒字段**；
   回放牌组键虽带 `-ev1/-hero`，事件流**不标注哪一次是觉醒**（`form_at_play` 全 "unknown"）。
   ⇒ 若将来接上觉醒，**观测确实会缺输入**（"这张是不是觉醒形态"/"本手是否觉醒"），这是本轮唯一被证实的
   "网络少输入"候选，但**它现在不构成任何已观测行为的成因**（当前对局里觉醒不可能触发）。
5. **R13 语料缺口**：128 张位图里没有任何「效果载体」状态 ⇒ F2 这类改动**过不了位图门禁的判别力**。
   建议（未做）给 `_mask_diff_snapshot.py` 的 `build_states()` 加一个「敌方 Log 滚动弹在场」的状态。
6. `_mask_diff_snapshot.py::build_states()` 的 **⑥ `late_low_tower` 是死代码**：那句 `s = make(); ...princess hp=150`
   构造的 `s` 被丢弃，实际 append 的是 `make(time=125.0)`（公主塔满血）⇒ 该状态**测不到"低血公主塔"口径**。
