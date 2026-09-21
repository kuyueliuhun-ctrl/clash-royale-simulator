# 现状与改动点 综合报告 —— 法术闸门 / 观测注入 / 对手信息 / 开局试探

> **合成自四份并行只读取证**（2026-09-22）。**纪律**：① 保留全部 `file:line` 出处；② **冲突处并列不调和**（见 §5）；③ 「未定」项集中 §6。
> **四份来源**：
> ① 法术掩码闸门 + 卡费来源（实读 `src/clasher_new/rl/action_mask.py` 全文 683 行，表格数值均为实跑本仓 `Card`/`action_mask`/`spell_module`）；
> ② 观测注入路径与维度代价（只读）；
> ③ 对方手牌 / 圣水 / 卡组信息可得性（标注【实测】= 亲自运行验证、【读码】= 仅读代码推定）；
> ④ 人类回放「开局试探」经验标定（FirstLight `IL_Replay` 本地唯一分片，脚本 `/tmp/il_opening_probe.py`，两跑 `md5 = 7b0b01f57c63dab4b4f40ada645ea404`）。
> **本报告只做合成与索引，未重新取数**。凡本报告作者亲自复核过的点加 `▸复核` 标注（见 §8）。行号 = 当前工作区实际行号（HEAD `37d6d83`）。
> **落盘状态**：本文件为新建；**尚未登记进 `docs/README.md` 索引与 `docs/agents/session_ledger.md`**（见 §8 末）。

---

## 0. 一页速查

| 主题 | 一句话现状 | 关键 `file:line` |
|---|---|---|
| **法术闸门** | 7 个具名谓词 + 2 个 frozenset；**两条并行真源**各内联一遍同样三条判断，改一处必须同改另一处 | `action_mask.py:507-514`、`:567-579` |
| **闸门总开关** | `_spell_requires_placement_target` 为 False ⇒ 8h 空砸 + 9h 砸塔 EV **两条闸门都不跑** | `action_mask.py:144-150` |
| **费用分档** | 掩码/闸门链路**零先例**；唯一费用用法是**连续式** `dmg/500 < 0.5×cost`，不是 `cost<=X` | `action_mask.py:328-329` |
| **卡费真值** | `gamedata.json → items.spells[].manaCost` → **标量** `Card.elixir`；掩码侧有 **3 份等价取费实现** | `card_utils.py:257`、`action_mask.py:30-36`、`:299`、`env_wrapper.py:323-328` |
| **若 X=2 豁免** | 真正受益者**只有 Zap(2) + Snowball(2)**；Log/BarbLog 已被滚动类豁免 | `action_mask.py:107`、`:150` |
| **观测键** | `observe()` 只 5 键（`intent_save` 开 +2）；`plan`/`belief` **不在 obs 字典** | `observation.py:138-144`、`env_wrapper.py:242-249` |
| **fused 维度** | `2560(grid) ‖ 40(hand) ‖ 3(scalar) ‖ 64(plan) ‖ 64(belief)` = **2731** → `enc` **128** | `follower.py:235-258,316-350` |
| **加标量代价** | `enc_fc` 形状变、**无兼容分支 ⇒ 静默重置 ⇒ 必须 `--fresh`** | `follower.py:129-152`、`:172-183` |
| **加 plan 维代价** | `plan_mlp` 有「前列拷贝+尾零」兼容 ⇒ 旧 ckpt 可加载；**但必须同修 `follower.py:424-427`** | `follower.py:135-139`、`:424-427` |
| **belief 563** | 规则贝叶斯 + 计数统计 + 事件通道；**非学习**，神经段全仓零调用 | `belief.py:36-38,344-362`、`:261` |
| **对手手牌** | 同进程**真值可直接读**；跟随者观测**不可见**；人类回放里**不可得（只是重建）** | `player.py:8-15`、`observation.py:148`、`fl_il_to_bc.py:338-343` |
| **对手圣水** | 唯一估计器 `_elixir_est`，**三处已确证偏差源**；录像 `elixir1` 逐帧有真值 | `belief.py:288-298,313-315`、`replay.py:181-182` |
| **开局试探** | T=30s 出 4 张牌 / 4 张不同卡；**90.0% 出牌在己方侧**；首牌圣水中位 **[6.33, 8.95]** | 见 §4 |
| **探试期下界** | 首次 ≥4 费牌 median **18.20 s**；首次进敌半场 **61.55 s** | 见 §4.4 R5 |

---

## 1. 卡费 / 法术闸门现状【侦察①】

### 1.1 七个谓词 + 两个 frozenset，及全部调用点

**结论**：共 7 个具名谓词 + 2 个 frozenset。调用链是**两条并行的真源**：`_position_legal`（提交路径，`action_mask.py:507-514`）与 `legal_cells`（掩码热路径，`:567-579`）各内联了一遍同样的三条判断。这 7 个谓词**没有任何一条包含费用分档**——唯一的费用出现是 `_spell_tower_ev_illegal` 内部的连续式 `edw × cost`（`:329`），不是 `cost <= X` 的豁免/分档。

#### 1.1.1 `_spell_deals_damage(card_name, card_info=None) -> bool`
定义 `action_mask.py:121-141`。逐字：
```python
def _spell_deals_damage(card_name: str, card_info: "Card" = None) -> bool:
    """是否输出**敌方**伤害的法术（引擎同源口径，见上方长注释）。"""
    if card_name in _NON_ENEMY_DAMAGE_SPELLS:
        return False
    info = card_info if card_info is not None else Card(card_name)
    pd = getattr(info, "projectile_data", None)
    if pd is not None and _num(getattr(pd, "damage", 0)) > 0.0:
        return True
    data = getattr(info, "data", None) or {}
    if _num(data.get("damage")) > 0.0:
        return True
    if card_name in _ROLLING_SPELLS:
        #: 滚动/释放单位类：伤害在飞行/落地后结算（引擎实测 Log 290、BarbLog 由野蛮人打）
        return True
    from card_utils import spells
    row = spells.get(card_name) or {}
    if _num(row.get("damage")) > 0.0:
        return True
    if _num((row.get("buff_data") or {}).get("damage_per_second")) > 0.0:
        return True
    return False
```
调用点（全仓 grep）：
- `action_mask.py:150` — 由 `_spell_requires_placement_target` 调用；
- `action_mask.py:262` — 由 `_spell_tower_ev_illegal` 调用；
- `scripts/selftest_spell_kingtower.py:198,200,202`（回归测试）；
- `docs/_archive/illegal_action_layers_2026-09-18.md:210`（文档里的**旧版**代码片段，非在场代码）。
- **注意**：`_position_legal` / `legal_cells` 都**不直接**调它，只经 `_spell_requires_placement_target`（`:507`、`:567`）。

`_num()` 辅助函数 `:110-118`（处理「按等级列表」字段，取 `max`）：
```python
def _num(x) -> float:
    """把引擎字段转成 float（`spells` 行里 damage 可能是标量或按等级列表）。"""
    if isinstance(x, (list, tuple)):
        vals = [v for v in (_num(v) for v in x) if v is not None]
        return max(vals) if vals else 0.0
```

#### 1.1.2 `_spell_requires_placement_target(card_name, card_info=None) -> bool`
定义 `action_mask.py:144-150`。逐字：
```python
def _spell_requires_placement_target(card_name: str, card_info: "Card" = None) -> bool:
    """落点几何闸门（8h 空砸 + 9h 砸塔 EV）是否适用于该卡。
    ...
    """
    return _spell_deals_damage(card_name, card_info) and card_name not in _ROLLING_SPELLS
```
**这是「8h 空砸闸门 + 9h 砸塔 EV 闸门」的唯一总开关**。调用点：`action_mask.py:507`、`:567`；`scripts/selftest_spell_kingtower.py:195` import、`:203-204` 断言。
**含义**：该谓词为 False 时**两条闸门都不跑** ⇒ 落点几何完全放开（只保留 `_hits_dead_enemy_tower`）。所以「非伤害型」和「滚动类」在当前实现下**已经是无落点限制的**。

#### 1.1.3 `_spell_has_enemy_target(battle, player_id, pos, radius) -> bool`（8h 空砸闸门本体）
定义 `:165-183`。逐字：
```python
def _spell_has_enemy_target(battle, player_id: int, pos: Position, radius: float) -> bool:
    """溅射半径内是否有存活敌方目标（塔/建筑/部队）。命中口径与引擎溅射一致：
    距离 ≤ 半径 + 目标碰撞半径。"""
    opp = 1 - player_id
    for e in battle.entities.values():
        if not getattr(e, "is_alive", True):
            continue
        if getattr(e, "player", None) != opp:
            continue
        if _is_effect_body(e):
            continue
        col = getattr(getattr(e, "data", None), "collision_radius", 0.0) or 0.0
        if pos.distance_to(e.position) <= radius + col + 1e-9:
            return True
    return False
```
`_is_effect_body`（`:159-162`，集合 `_EFFECT_BODY_TYPES = {"projectile","area_effect","bomb"}` 定义 `:156`）跳过效果载体。
调用点（全仓）：`action_mask.py:509`（`radius = _spell_radius_m(card_name, card_info)`，`:508`）、`:575`（`radius` 来自 `:568`）、`scripts/il_probe_kingtower_cast.py:230,263`。

#### 1.1.4 `_spell_covers_non_tower(battle, player_id, pos, radius) -> bool`（9h 的「有正事可干 ⇒ 放行」分支）
定义 `:215-231`。逐字：
```python
def _spell_covers_non_tower(battle, player_id: int, pos: Position, radius: float) -> bool:
    """溅射半径内是否有对手的**非塔**目标（部队/建筑）。有 → 法术有正事可干，放行。"""
    opp = 1 - player_id
    tower_ids = {1, 2, 3, 4, 5, 6}
    for e in battle.entities.values():
        if not getattr(e, "is_alive", True):
            continue
        if getattr(e, "player", None) != opp:
            continue
        if e.id in tower_ids:
            continue
        if _is_effect_body(e):
            continue          # ★ 2026-09-22：效果载体不是"有正事可干"的目标（同上）
        col = getattr(getattr(e, "data", None), "collision_radius", 0.0) or 0.0
        if pos.distance_to(e.position) <= radius + col + 1e-9:
            return True
    return False
```
调用点：**仅** `action_mask.py:297`（`_spell_tower_ev_illegal` 内）；`radius` 来自 `:264`。

#### 1.1.5 `_spell_tower_ev_illegal(battle, player_id, card_name, pos, card_info=None) -> bool`
定义 `:234-329`。全条件（全部满足才返回 True = 非法）：

| # | 条件 | 行 |
|---|---|---|
| 1 | `battle.time >= 120.0` → False（双倍期放行） | `:257-258` |
| 2 | `card_name in _ROLLING_SPELLS` → False | `:259-261` |
| 3 | `not _spell_deals_damage(...)` → False | `:262-263` |
| 4 | `radius = _spell_radius_m(...)`；`radius <= 0.0` → False | `:264-266` |
| 5 | `dmg = _spell_tower_damage(card_name)`；`dmg <= 0.0` → False | `:267-269` |
| 6 | 罩不到任何对手存活塔（公主 `:275-282` **或** 王塔 `:283-296`）→ False | `:295-296` |
| 7 | `_spell_covers_non_tower(...)` 为真 → False | `:297-298` |
| 8 | `cost = card_info.elixir if ... else Card(card_name).elixir`；`cost <= 0` → False | `:299-301` |
| 9 | 终判：`dmg_eff / TOWER_HP_PER_ELIXIR_EARLY < SPELL_EV_EDW * cost - 1e-9` | `:328-329` |

终判 + 费用出现的**唯一两处**逐字：
```python
    cost = card_info.elixir if card_info is not None else Card(card_name).elixir
    if cost <= 0:
        return False
```
```python
    dmg_eff = dmg * best_mult
    return dmg_eff / TOWER_HP_PER_ELIXIR_EARLY < SPELL_EV_EDW * cost - 1e-9
```
常量 `:193-194`：
```python
TOWER_HP_PER_ELIXIR_EARLY = 500.0   # 与 spell_module.TOWER_HP_PER_ELIXIR 一致
SPELL_EV_EDW = 0.5                  # 奖励经济前段费差权重（economy 预设）
```
调用点：`action_mask.py:512-513`（`_position_legal`，实参 `battle, player_id, card_name, pos, card_info`）、`:577-578`（`legal_cells`，实参 `battle, player_id, eff, pos, eff_info`）、`scripts/il_probe_kingtower_cast.py:231,264`。

#### 1.1.6 `_ROLLING_SPELLS` / `_NON_ENEMY_DAMAGE_SPELLS`
```python
_ROLLING_SPELLS = frozenset({"Log", "BarbLog"})                                  # :107
_NON_ENEMY_DAMAGE_SPELLS = frozenset({"Heal", "Rage", "Clone", "GlobalClone", "Mirror"})  # :106
```
- `_ROLLING_SPELLS` 消费 3 处：`:132`（`_spell_deals_damage` 内返回 True——注意它让滚动类「算伤害」）、`:150`（`and card_name not in _ROLLING_SPELLS`）、`:259`（`_spell_tower_ev_illegal` 提前 return False）。
- `_NON_ENEMY_DAMAGE_SPELLS` 消费 1 处：`:123-124`（`_spell_deals_damage` 首行 return False）。

#### 1.1.7 附：`_spell_radius_m` / `_spell_tower_damage`
- `_spell_radius_m(card_name, card_info=None)` 定义 `:79-86`：先读 `card.data["radius"]`，缺/0 时回退 `data["projectileData"]["radius"]`，`/1000.0`；无数据 → `0.0`（=闸门自动失效）。调用点 `:264, :508, :568`；**跨模块**：`rl/mcts.py:261` 直接 import 该私有函数（`:261-265`）。
- `_spell_tower_damage(card_name)` 定义 `:199-212`：经 `spell_module.get_spell_profile(card_name)`（默认 `Card.default_level`），标定失败/异常 → `0.0`。缓存 `_spell_tower_dmg_cache`（`:196`）。调用点 `:267`。

### 1.2 两条调用路径（影响最小侵入落点）

`_position_legal` 的法术分支（`:504-515`）逐字：
```python
    if card_info.type == "spell":
        if _hits_dead_enemy_tower(battle, player_id, pos):
            return False
        if _spell_requires_placement_target(card_name, card_info):
            radius = _spell_radius_m(card_name, card_info)
            if radius > 0.0 and not _spell_has_enemy_target(battle, player_id, pos, radius):
                return False
            # 9h 前段纯砸塔 EV 闸门：无部队/建筑可溅、账面亏费 → 非法
            if radius > 0.0 and _spell_tower_ev_illegal(battle, player_id, card_name, pos,
                                                        card_info):
                return False
        return True
```
`legal_cells` 的法术分支（`:566-580`）逐字：
```python
    if is_spell:
        deals_dmg = _spell_requires_placement_target(eff, eff_info)
        radius = _spell_radius_m(eff, eff_info) if deals_dmg else 0.0
        ev_gate = radius > 0.0 and deals_dmg
        for y in range(GRID_H):
            for x in range(GRID_W):
                pos = sub_position(player_id, x, y)
                if _hits_dead_enemy_tower(battle, player_id, pos):
                    cells[y, x] = False
                elif radius > 0.0 and not _spell_has_enemy_target(battle, player_id, pos, radius):
                    cells[y, x] = False
                elif ev_gate and _spell_tower_ev_illegal(battle, player_id, eff, pos,
                                                         eff_info):
                    cells[y, x] = False
        return cells
```
⇒ 两条路径**逐条重复**同样的三判断（`_hits_dead_enemy_tower` / `_spell_has_enemy_target` / `_spell_tower_ev_illegal`），**改一处必须同改另一处**。此外 `legal_cells` 用 `elif`：**空砸闸门先于 EV 闸门**，两者都在 `_hits_dead_enemy_tower` 之后。

### 1.3 卡费从哪里读？

**结论**：唯一真值来自 `gamedata.json → items.spells[].manaCost`，经 `card_utils.Card.__init__` 暴露为 **标量属性 `Card.elixir`**（**不是按等级列表**）。掩码侧统一走 `action_mask._card_cost(player, card_name)`（Mirror 特化）与 `Card(name).elixir`。

- 数据行：`card_utils.py:50-51`
  ```python
  data = data['items']['spells']
  card_data = {each['name']: each for each in data}
  ```
- 费用读取 `card_utils.py:257`（逐字）：
  ```python
          self.elixir = self.data.get('manaCost', 0) # princess towers don't have elixir cost
  ```
- 实测：27 张法术的 `manaCost` **全部是 `int` 标量**。`cards_stats_spell.json` 里另有 `elixir` / `mana_cost` / `damage_per_level` / `dps_per_level` 这些**按等级列表或行级冗余字段**，但 `Card.elixir` **不用它们**（例：`RoyalDelivery` 行内 `elixir=0`、`GlobalLightning` 行内 `elixir=0`，而 `Card(...).elixir` 分别是 3 / 1）。

**掩码侧的两个取费入口 + 一处重复**：
- `action_mask.py:30-36`（逐字）——**掩码/校验链路唯一该用的取费函数**：
  ```python
  def _card_cost(player, card_name: str) -> Optional[float]:
      """实际出牌费用（Mirror 按引擎语义 = 上一张牌费用 + 1；无上一张牌 → None）。"""
      if card_name == "Mirror":
          if not getattr(player, "last_card", None):
              return None
          return Card(player.last_card).elixir + 1
      return Card(card_name).elixir
  ```
  消费点：`action_mask.py:35-36`、`:51`（`_slot_playable`）、`:346`（`_opp_min_hand_cost`）、`:367`（env_wrapper `get_action_mask_for` partial 模拟）、`:402`（`_intent_holding`）、`:436,478`（`_apply_intent`）、`:675`（`validate_bundle` 扣费）。
- `action_mask.py:299` —— `_spell_tower_ev_illegal` **自己另写了一份**：`cost = card_info.elixir if card_info is not None else Card(card_name).elixir`（**绕过 `_card_cost` 的 Mirror 特化**；但 Mirror 在 `_NON_ENEMY_DAMAGE_SPELLS` 里 → `:262` 已提前 return False，实际不外泄）。
- `env_wrapper.py:323-328` —— partial bundle 扣费模拟**又写了一份**：
  ```python
                  from card_utils import Card
                  card = p.cycle[sa.slot - 1]
                  cost = Card(card).elixir
                  if card == "Mirror" and getattr(p, "last_card", None):
                      cost = Card(p.last_card).elixir + 1
  ```
  ⇒ 「取费」在本仓有 **3 份等价实现**（`_card_cost` / `:299` / `env_wrapper:325-327`），底层都是 `Card.elixir ← gamedata.manaCost`。

**「按等级列表」还是标量**：`Card.elixir` = **标量**。仓内**按等级列表**字段是 `damage_per_level`（`cards_stats_spell.json`，27 张里 6 张有：`Freeze/Zap/RoyalDelivery/GlobalLightning/DarkMagic/GoblinCurse/Vines`），由 `_num()`（`action_mask.py:110-118`）取 `max` 消费，**与费用无关**。

### 1.4 全部 27 张法术（卡名 / 费用 / 伤害 / 滚动类 / 友方语义）

数据来源：`gamedata.json` 中 `tidType == "TID_CARD_TYPE_SPELL"` 的 **27 行**；「落点伤害」三列 = `Card.projectile_data.damage`(pd) / `Card.data.damage`(data) / `card_utils.spells[卡].damage`(row)；`DOT` = `spells[卡].buff_data.damage_per_second`；`tower_dmg` = `_spell_tower_damage(卡)`（引擎标定 lv11）；末两列 = `_spell_deals_damage` / `_spell_requires_placement_target`。

| 卡名 | 费用 | pd.damage | data.damage | spells行 damage | DOT(dps) | 滚动类 | 友方语义 | radius_m | tower_dmg | deals_dmg | req_target |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Fireball | 4 | 688 | — | — | — | 否 | 否 | 2.5 | 206.0 | ✓ | ✓ |
| Arrows | 3 | 122 | — | — | — | 否 | 否 | 3.5 | 75.0 | ✓ | ✓ |
| Rage | 2 | 0 | — | 0 | 0 | 否 | **是** | 3.0 | 0.0 | ✗ | ✗ |
| Rocket | 6 | 1484 | — | — | — | 否 | 否 | 2.0 | 371.0 | ✓ | ✓ |
| GoblinBarrel | 3 | 0 | — | — | — | 否 | 否 | 1.5 | 0.0 | ✗ | ✗ |
| Freeze | 4 | 0 | — | 72 | 0 | 否 | 否 | 0.0 | 34.5 | ✓ | ✓ |
| Mirror | 1 | 0 | — | — | — | 否 | **是** | 0.0 | 0.0 | ✗ | ✗ |
| Lightning | 6 | 0 | — | 0 | — | 否 | 否 | 3.5 | 0.0 | **✗** | ✗ |
| Zap | 2 | 0 | — | 75 | 0 | 否 | 否 | 2.5 | 57.6 | ✓ | ✓ |
| Poison | 4 | 0 | — | 0 | **57** | 否 | 否 | 3.5 | 343.7 | ✓ | ✓ |
| Graveyard | 5 | 0 | — | 0 | — | 否 | 否 | 4.0 | 0.0 | ✗ | ✗ |
| Log | 2 | 0 | — | — | — | **是** | 否 | 1.95 | 0.0 | ✓ | ✗ |
| Tornado | 3 | 0 | — | 0 | **106** | 否 | 否 | 5.5 | 1.2 | ✓ | ✓ |
| Clone | 3 | 0 | — | 0 | 0 | 否 | **是** | 3.0 | 0.0 | ✗ | ✗ |
| Earthquake | 3 | 0 | — | 0 | **39** | 否 | 否 | 3.5 | 171.0 | ✓ | ✓ |
| BarbLog | 2 | 0 | — | — | — | **是** | 否 | 1.3 | 0.0 | ✓ | ✗ |
| Heal | 1 | **110** | — | 0 | 0 | 否 | **是** | 0.0 | 0.0 | ✗ | ✗ |
| Snowball | 2 | 192 | — | — | — | 否 | 否 | 2.5 | 58.0 | ✓ | ✓ |
| RoyalDelivery | 3 | 0 | — | [51…214] | — | 否 | 否 | 3.0 | 0.0 | ✓ | ✓ |
| GlobalClone | 3 | 0 | — | 0 | 0 | 否 | **是** | 30.0 | 0.0 | ✗ | ✗ |
| GoblinPartyRocket | 5 | 0 | — | — | — | 否 | 否 | 0.0 | 0.0 | ✗ | ✗ |
| WarmSpell | 1 | 0 | — | — | — | 否 | 否 | 4.0 | 0.0 | ✗ | ✗ |
| GlobalLightning | 1 | 0 | — | 0 | — | 否 | 否 | 0.0 | 0.0 | ✗ | ✗ |
| DarkMagic | 3 | 0 | — | 0 | — | 否 | 否 | 2.5 | 0.0 | ✗ | ✗ |
| GoblinCurse | 2 | 0 | — | 0 | — | 否 | 否 | 3.0 | 0.0 | ✗ | ✗ |
| MergeMaiden | 6 | 0 | — | — | — | 否 | 否 | 0.0 | 0.0 | ✗ | ✗ |
| Vines | 3 | **153** | — | 0 | — | 否 | 否 | 2.5 | 76.5 | ✓ | ✓ |

**必须标注的实测事实（不是推测）**：
1. **`Lightning` 的 `deals_dmg = ✗`**：伤害写在 `cards_stats_projectile.json`（行内 `projectile = 'LighningSpell'`），`_spell_deals_damage` 的四路取或**都不覆盖这张表** ⇒ 8h/9h 两条闸门对 Lightning **根本没开**（`req_target = ✗`）。这与 `docs/il_spell_kingtower_gap_2026-09-22.md:182` 记的「F4：`_spell_tower_damage` 对 Log / Lightning 读 0」是**同一条遗留缺口**（实跑仍为 0）。
2. **`_spell_tower_damage` 只对 `deals_damage=True` 的卡标定**（`spell_module.py:181` 起 `_calibrate` 前置 `if not prof["deals_damage"]: return prof`）⇒ Log / Lightning / BarbLog 读 0.0。
3. `Heal` 的 `pd.damage = 110` 是**治疗量**（这正是 `_NON_ENEMY_DAMAGE_SPELLS` 存在的原因，`action_mask.py:99-100` 注释）；`Rage / Clone / Mirror` 的 `damage` 字段是增益语义。
4. `RoyalDelivery` 的 `spells` 行 `radius=0`，`_spell_radius_m` 回退到 `projectileData.radius` 得 3.0（`:83-86` 的回退分支实证）。
5. 27 张法术**全部**在 `build_card_pool()` 的 144 张可部署池内（实跑 `opponents.py:49-68`，缺失 0 张）。

### 1.5 按费用阈值分组（用上表真实费用）

- **cost ≤ 2**：`Heal(1)`、`Mirror(1)`、`WarmSpell(1)`、`GlobalLightning(1)`、`Rage(2)`、`Zap(2)`、`Log(2)`、`BarbLog(2)`、`Snowball(2)`、`GoblinCurse(2)` — **实列 10 张**。
  > ⚠️ **报告①内部数字冲突（原样保留）**：标题写「**9 张**」，但紧随的清单为 **10 张**，且 §4 末的校验式 `10 + 9 + 3 + 5 = 27 ✓` 用的是 **10**。⇒ 以**清单/校验式（10 张）**为准；「9 张」标签为笔误（未调和）。
- **cost ≤ 3（新增 9 张，累计 19 张）**：`Arrows(3)`、`GoblinBarrel(3)`、`Tornado(3)`、`Clone(3)`、`Earthquake(3)`、`RoyalDelivery(3)`、`GlobalClone(3)`、`DarkMagic(3)`、`Vines(3)`。
- **cost == 4（3 张）**：`Fireball`、`Freeze`、`Poison`。
  > ⚠️ **报告①内部自我更正（原样保留）**：原文先写「5 张」并误列 `SkeletonKing`，随后自行更正为「以实跑为准：cost==4 仅 3 张」。⇒ 取 **3 张**（未调和）。
- **cost ≥ 5（5 张）**：`Graveyard(5)`、`GoblinPartyRocket(5)`、`Rocket(6)`、`Lightning(6)`、`MergeMaiden(6)`。
- **校验**：10 + 9 + 3 + 5 = 27 ✓（与 `tidType == TID_CARD_TYPE_SPELL` 行数一致）。

**对「8h 空砸闸门」的实际意义**（`req_target == ✓` 且 `radius > 0` 的只有这些）：
- cost ≤ 2 且 `req_target=✓`：**只有 `Zap(2)`、`Snowball(2)`**（`Log/BarbLog` 已被 `_ROLLING_SPELLS` 豁免；`Heal/Rage/Clone/Mirror/WarmSpell/GlobalLightning/GoblinCurse` 本就 `deals_dmg=✗`）。
- cost ≤ 3 新增：`Arrows`、`Tornado`、`Earthquake`、`RoyalDelivery`、`Vines`（`Freeze` 的 radius=0，闸门自动失效）。
- ⇒ **若 X=2，豁免的实际受益者只有 Zap + Snowball 两张**；若 X=3，再加 5 张。

### 1.6 「低费法术豁免 8h 空砸闸门（cost≤X）」的最小侵入落点

**结论**：落点在 **`action_mask.py:507`（提交路径）与 `action_mask.py:567`（掩码热路径）两处**——它们各自是两条闸门的**总开关**。最小侵入写法 = 改这两处的「开关布尔量」，**不动** `_spell_has_enemy_target` / `_spell_tower_ev_illegal` 的函数体。

**落点 A — `action_mask.py:144-150`（新增单一谓词，取费走 `Card.elixir`）**：现有函数见 §1.1.2。
**落点 B — `action_mask.py:507`（提交路径）**：现有 `if _spell_requires_placement_target(card_name, card_info):`。
**落点 C — `action_mask.py:567`（掩码热路径）**：现有 `deals_dmg = _spell_requires_placement_target(eff, eff_info)`。

因为 `legal_cells` 把 `deals_dmg` 同时用于 `radius` 与 `ev_gate`（`:568-569`），**只改 B/C 两处的开关表达式**即可同时豁免「空砸」与「砸塔 EV」两条闸门；`--mask-diff` 的位图变化面也就锁在这两条路径内。

**若只想豁免 8h 空砸、保留 9h EV 闸门**，则落点是**另一组行**（不能改 `:507/:567`）：
- `action_mask.py:509`：`if radius > 0.0 and not _spell_has_enemy_target(battle, player_id, pos, radius):`
- `action_mask.py:575`：`elif radius > 0.0 and not _spell_has_enemy_target(battle, player_id, pos, radius):`

这两处各加一个费用条件（如 `and _spell_cost(card_name) > X`）即可，`ev_gate` 保持不变。**⚠️ 两种落点语义不同，必须先拍板「豁免哪条闸门」**——任务描述只说了「空砸」，故侦察①**推荐落点是 `:509` + `:575`**；但 `:507/:567` 是更省事的「整块豁免」。

**建议新增的单一取费谓词**（放 `:150` 之后或 `:107` 附近）：
```python
#: 低费法术豁免闸门的费用阈值（0 = 关闭豁免，行为逐位不变）。
_SPELL_GATE_FREE_COST = 0.0

def _spell_gate_exempt(card_name: str, card_info: "Card" = None) -> bool:
    """低费法术豁免（cost <= X）：与卡费同源（Card.elixir ← gamedata.manaCost）。"""
    if _SPELL_GATE_FREE_COST <= 0.0:
        return False
    cost = card_info.elixir if card_info is not None else Card(card_name).elixir
    return float(cost or 0.0) <= _SPELL_GATE_FREE_COST
```

**新增豁免要遵守的既有约束**：
- **R13**：掩码/合法格改动先跑 `scripts/_mask_diff_snapshot.py`（128 张**逐位全等**；`--compare A B` + `--selftest`，见 `scripts/_mask_diff_snapshot.py:1-20`）。**但**「低费豁免」本质就是**逐位改变**，与 R13 字面「全等」冲突 ⇒ 需按 C20/C21 先例改走「意图位图 sweep（只允许收紧/按预期放宽）+ 回归断言 + 预注册判据」（`docs/agents/ledger.md` C21 已记：法术全卡位图 sweep **24 张 × 8 状态 × 2 方 = 384 张**，`96/384 变化、全部只收紧`）。
- **R2**：新增开关必须**默认关 / 默认行为逐位不变**（`config.py:295-297` 的 `value_bypass` 先例：架构变更须 `--fresh`）。费用豁免会**放宽**合法集，建议默认 `X = 0`（关闭），经 `TrainConfig` / CLI 显式开。

### 1.7 尺度常量单一来源（红线 R7）——**存在，且当前已有一处既成偏离**

- **R7 原文要点**（`AGENTS.md` A 区）：`汇率 / 值函数 / 闸门 edw×卡费 / MCTS 同源同步改`；改尺度时**同源同步**。
- **现状盘点（实测）**：
  - `action_mask.py:193` `TOWER_HP_PER_ELIXIR_EARLY = 500.0` 注释自称「与 spell_module.TOWER_HP_PER_ELIXIR 一致」；`spell_module.py:302` `TOWER_HP_PER_ELIXIR = 500.0`；`belief_planner.py:107` **又一份** `TOWER_HP_PER_ELIXIR_EARLY = 500.0`；`belief_planner.py:643` `TOWER_HP_PER_ELIXIR_LATE = 50.0`。⇒ **500.0 在仓内至少 3 处独立字面量**（`action_mask:193` / `spell_module:302` / `belief_planner:107`）。
  - `action_mask.py:194` `SPELL_EV_EDW = 0.5` 是**硬编码**，而奖励侧真值在 `rl/config.py:61` `"elixir_diff_weight": 0.5`（各预设可覆盖：`:77-79` 0.7/0.3/0.05；`flow_league.py:313` 会改 `rw_b2["elixir_diff_weight"]`）。`action_mask.py:192` 注释承认「消费者注入，缺省 0.5」，但**当前没有任何消费者注入**（全仓 grep：`SPELL_EV_EDW` 仅 `action_mask.py:194,329` 两处，无外部赋值）。
  - ⇒ **若新增的 X 阈值要落进 R7 语义**，正确做法是把 X 与 `SPELL_EV_EDW` / `TOWER_HP_PER_ELIXIR_EARLY` 一起收进**同一个常量源**（例如都从 `config.DEFAULT_REWARD` 注入到 `RLEnv`，再由 `get_action_mask_for` 传给掩码），而不是在 `action_mask.py` 里再写一个字面量 `X = 2`。**本仓当前 `SPELL_EV_EDW` 已经违反该原则**（三处 500 + 一处硬编码 0.5），新增 X 若也硬编码，会把偏离从「已存在」变成「+1 处」。

### 1.8 全仓是否已有「按费用豁免/分档」先例？

**结论**：**在掩码/闸门链路里没有任何先例**（0 处 `cost <= X` 型豁免）；但在**规划器/启发式链路**里有多处「按费用分档」的成熟写法，可直接借鉴其风格（同一份 `Card.elixir` 口径 + 模块级命名常量 + 邻近注释说明量纲）。

全仓 grep（`--include=*.py`）的 `cost <= / < / >= / >` 命中，按文件分类：

| 位置 | 逐字代码 | 性质 |
|---|---|---|
| `action_mask.py:300` | `if cost <= 0:` | 仅**哨兵**（0 费 = 无数据），不是分档 |
| `env_wrapper.py:578` | `if cost <= 1e-9:` | 哨兵（记账跳过） |
| `opponents.py:63` | `if cost is None or cost <= 0:` | 哨兵（卡池过滤） |
| `belief_planner.py:82` + `:520` | `PULL_CHEAP_COST = 3.0` … `if p.elixir < c.elixir or c.elixir > PULL_CHEAP_COST:` | **★ 最接近的先例**：命名常量「廉价阈值」+ 单一来源（`:82` 定义，`:520` 消费） |
| `belief_planner.py:480-481` | `if c.type in ("character","building") and 1.0 <= c.elixir <= 4.0 \` | 费用**区间**分档 |
| `belief_planner.py:567` | `if (card in TANK_CARDS or (c.type == "character" and c.elixir >= 5.0)) \` | 费用**下限**分档（坦克） |
| `belief_planner.py:601-602` | `return c.type == "character" and 3.0 <= c.elixir <= 6.0 \` | 费用区间分档 |
| `belief_planner.py:731-732` | `if c.type in ("character","building") and 1.0 <= c.elixir <= 3.0 \` | 费用区间分档 |
| `belief_planner.py:804` | `if c.elixir <= 2.0 and p.elixir >= c.elixir + 3.0 and card != "Mirror":` | **★「低费」判据先例**（`<= 2.0`，硬编码字面量） |
| `belief_planner.py:721` | `and c.elixir >= 5.0:` | 高费分档 |
| `belief_planner.py:786-787` | `if c.type == "character" and 3.0 <= c.elixir <= 6.0 \` | 费用区间分档 |
| `prophet.py:232,244,267,293,315,373,383,401,437,453` | 同上各条的 **ProphetPlanner 镜像副本**（`Card(c).elixir <= 4.0`、`1.0 <= c.elixir <= 3.0`、`c.elixir <= 2.0 and ...`） | 同上，**两处各写一份**（与 §1.2「两条真源」同型风险） |
| `belief_planner.py:424` / `prophet.py:193` | `if cost < 3.0:` | 费用分档 |
| `selftests/part5.py:1686` | `assert bool(m["slots"][i]) == (cost <= expect + 1e-9), (` | **测试侧**按费用对账槽掩码 |
| `selftests/part5.py:1888,1894` | `ge = [c for c in solo if Card(c).elixir >= j1_min]` | **测试侧**按费用阈值筛 ≥6 费牌（J1 判据） |
| `scripts/s1_plan_gate.py:387` | `if cost > float(plan.elixir_budget) * 10.0 + 1e-9:` | 计划预算比例 |
| `scripts/offline_engagement_trade.py:449` | `if not cands or cost <= 1e-9:` | 哨兵 |
| `scripts/pass_streak_audit.py:104` | `over = cost > pre[i] + 1e-6` | 审计 |

**可直接借鉴的写法（两条）**：
1. `belief_planner.py:82` 的「命名阈值常量 + 单一消费点」⇒ 掩码侧应写成 `_SPELL_GATE_FREE_COST = X`（定义一次），消费点只写 `cost <= _SPELL_GATE_FREE_COST`，**不要**在两条路径里各写一遍 `<= 2`。
2. `selftests/part5.py:1888` 的「按费用筛卡再断言」回归写法 ⇒ 新增豁免的回归测试应同风格地**按费用枚举**（而不是写死卡名列表），才能覆盖「X 改动时清单自动跟随」。

**没有先例的部分（明确写「无」）**：
- 掩码/闸门（`action_mask.py` 全 683 行）内**没有任何** `cost <= X` / `cost >= X` 的分档或豁免；`_spell_tower_ev_illegal` 的费用用法是**连续式** `dmg/500 < edw*cost`（`:329`）。
- **没有** `低费法术 / cheap spell / 豁免 / whitelist` 类的现存常量或开关（grep `豁免|白名单|EXEMPT|exempt|cheap_spell|low_cost|cost_threshold|cheap_tier` 在 `src/clasher_new/rl/` 下唯一实质命中是 `belief_planner.PULL_CHEAP_COST`，语义是「拉扯目标费用上限」，与法术无关）。
- **没有**把 `SPELL_EV_EDW` 做成可注入参数的现存管线（`env_wrapper.py:691` 只把 `elixir_diff_weight` 注入奖励 `rw`，不注入掩码常量）⇒ 若要做「单一来源」，这是一条**尚不存在、需要新建**的注入通道。

---

## 2. 观测注入路径与维度代价【侦察②】

### 2.1 `observe()` 的完整键列表 + 形状/dtype + fused 拼接

**结论**：`RLEnv.observe()` 只返回 **5 个键**（`intent_save` 开启时 +2）；`plan` / `belief` **不是** `observe()` 的键，它们是主训练循环里另外算出的两条 token，最后在模型里与 obs 一起拼成 `fused`（**2731** 维）→ `enc`（**128**）。

**生产者** `rl/observation.py:138-144`：
```python
    return {
        "grid": obs,                                    # (32,18,15) float32
        "hand": hand,                                   # (5,)  int32
        "elixir": np.array([p.elixir], dtype=np.float32),      # (1,) float32
        "next_card": np.array([next_card], dtype=np.int32),    # (1,) int32
        "time": np.array([battle.time], dtype=np.float32),     # (1,) float32
    }
```

| 键 | 形状 | dtype | 生成处 | 进 fused 的方式 |
|---|---|---|---|---|
| `grid` | `(32, 18, 15)` = 576 格 × 15 通道 | float32 | `observation.py:89-130` | `entity_emb`+one_hot → CNN → `grid_ln` → **2560** |
| `hand` | `(5,)`（`p.cycle[:5]`） | int32 | `observation.py:133-136` | `entity_emb` → reshape → **40**（5×8） |
| `elixir` | `(1,)` | float32 | `observation.py:141` | scalar 块第 1 维（**不归一化**） |
| `next_card` | `(1,)` | int32 | `observation.py:137,142` | scalar 块第 3 维（**÷12**，`follower.py:323`） |
| `time` | `(1,)` | float32 | `observation.py:143` | scalar 块第 2 维（**不归一化**） |
| `intent_slot` | `(1,)` | int32 | 仅 `intent_save=True` 且 `player_id==0`：`env_wrapper.py:244-248` | 经 `_intent_obs_vec` → scalar 块 +6 |
| `intent_age` | `(1,)` | float32 | 同上 | 同上 |

`observation_space` 登记处（`env_wrapper.py:129-141`）——**只在开启时**登记意图两键：
```python
        self.observation_space = gym.spaces.Dict({
            "grid": ... shape=(GRID_H, GRID_W, GRID_C) ...,
            "hand": ... shape=(5,) ...,
            "elixir": ... shape=(1,) ...,
            "next_card": ... shape=(1,) ...,
            "time": ... shape=(1,) ...,
        })
        if self.intent_save:
            self.observation_space.spaces["intent_slot"] = ...
            self.observation_space.spaces["intent_age"] = ...
```
`RLEnv.observe()` 本体只是薄包装（`env_wrapper.py:242-249`）：
```python
    def observe(self, player_id: int = 0) -> dict:
        d = observe(self.battle, player_id)
        if self.intent_save and player_id == 0:
            d["intent_slot"] = np.array([self._intent_slot], dtype=np.int32)
            d["intent_age"] = np.array([self._intent_age], dtype=np.float32)
        return d
```

**`plan` / `belief` 两条 token（不在 obs 字典里）**：
- `plan`：`PlanToken().to_vector()` → **`(58,)` float32**，常量源 `plan_space.py:187`。
- `belief`：`BeliefInference.encode()` → **`(563,)` float32**（默认 8 卡卡组）。组成（`belief.py:344-362`）：`hand_probs(8) + next_probs(8) + [elixir_mean,uncertainty](2) + tendency_probs(5) + opp_event_token(3×180=540)` = **563**。维度公式（`belief.py:36-38`）：`2*len(deck) + 2 + len(TENDENCIES) + OPP_EVENT_K*OPP_EVENT_DIM`，其中 `TENDENCIES` 5 项（`belief.py:32`）、`OPP_EVENT_K=3`、`OPP_EVENT_DIM=len(ENTITY_NAMES)+3=180`（`belief.py:47-49`）。
- 组装点（`rl/workers.py:45-51`）：`return obs, belief.encode(obs, None), plan.to_vector()`。

**fused 拼接顺序与维度**（唯一实现 `follower.py:316-350`）：
```python
        fused = torch.cat([grid_feat, hand_feat, scalar, plan_f, belief_f], dim=1)
        enc = self.enc_ln(torch.relu(self.enc_fc(fused)))            # (1,hidden)
```

| 段 | 来源 | 维度 |
|---|---|---|
| `grid_feat` | CNN(26ch) → Flatten → `grid_ln` | **2560** |
| `hand_feat` | `entity_emb(hand).reshape(1,-1)` | **40** |
| `scalar` | `cat([elixir, time, next_card/12] (+intent 6))` | **3**（intent 开启 **9**） |
| `plan_f` | `plan_mlp: Linear(plan_dim,64)+ReLU` | **64** |
| `belief_f` | `belief_mlp: Linear(belief_dim,64)+ReLU` | **64** |
| **`fused`** | — | **2731**（intent 开启 **2737**） |
| `enc` | `enc_ln(relu(enc_fc(fused)))`，`hidden=128` | **128** |

`cnn_out=2560` 由卷积栈推得并在 `docs/model_vector_inventory_2026-09-20.md:52,110` 有实证读数（`fused (1, 2731)`、`cnn_out 2560`）；`enc_dim=2560+40+3+64+64=2731`（`follower.py:257`）。批量路径同序：`follower.py:529`。

### 2.2 `plan` 向量 58 维逐段含义

**结论**：`plan_space.py` 的注释（`:11` `placement_hint(7)`、`:12` `PLAN_DIM = 57`、`:186` `= 57`）**是旧值**；实测 `PLAN_DIM=58`，根因是 `PLACEMENT_HINTS` 增了第 8 项 `intercept_mid` 后未同步注释。**唯一带逐段偏移清单的文档**是 `docs/model_vector_inventory_2026-09-20.md:43`（`逐段偏移 0:8/8:16/16:21/21:34/34:39/39:47/47:53/53:54/54:58`）；设计文档 `docs/_archive/rl_plan_design_v1.md:225` 只有散文式描述且文头写「57 维」已过期（`docs/plan_master.md:137,258` 记录该冲突）。**没有**逐维编号的正式清单文档。
> `▸复核`：本报告作者 grep `plan_space.py` 的 `PLAN_DIM` 仅命中 `:3`、`:12`、`:187`、`:188`。其中 `:187` 是定义行 `PLAN_DIM = int(len(PlanToken().to_vector()))`，`:12` 是含「PLAN_DIM = 57」的陈旧注释。**侦察②给出的 `:186` 出处未能复核**（`:186` 附近无 `PLAN_DIM` 字样）⇒ 视为「出处待核」。

逐段（`plan_space.py:124-160` 的 `to_vector()`，逐字对应）：

| 偏移 | 段 | 维度 | 含义 | 代码 |
|---|---|---|---|---|
| `0:8` | `intent_old` | 8 | 旧宏观意图 one-hot（`defend_left/right/king`、`push_left/right`、`counterpush`、`spell_value`、`cycle_and_wait`） | `plan_space.py:31-35,133-136` |
| `8:16` | `region` | 8 | `FOCUS_REGIONS` one-hot | `:53-57,141-142` |
| `16:21` | `old_scalars` | 5 | ①`suggested_card/4` ②`bundle_size_hint` ③`combo_hint` ④`risk_profile` ⑤`tanh(value_estimate)` | `:143-149` |
| `21:34` | `intent_new` | 13 | v1 新意图 one-hot（`soft_control`…`save_ace`） | `:36-50,134,137-138` |
| `34:39` | `target` | 5 | `TARGET_KINDS`：`none/unit/building/tower/my_backline` | `:60-62,150` |
| `39:47` | `hint` | 8 | `PLACEMENT_HINTS`：`none/pull_across/pull_aggro/support_zone/anti_spell_zone/bridge_front/king_front/intercept_mid` | `:65-74,151` |
| `47:53` | `threat` | 6 | `OPP_SPELL_THREATS`：`none/fireball/poison/lightning/freeze/big_unknown` | `:77-79,152` |
| `53:54` | `elixir_budget` | 1 | 本帧允许投入圣水比例 `clip(...,0,1)` | `:111,158` |
| `54:58` | `hold` | 4 | `hold_mask` 4 bit（`bit(slot-1)=1` → 本帧别出该槽，save_ace） | `:112,153-154,159` |

`to_vector()` 返回原文（`plan_space.py:155-160`）：
```python
        return np.concatenate([
            intent_old, region, old_scalars,          # 旧 21 维（兼容锚）
            intent_new, target, hint, threat,          # v1 意图组 + 目标 + 位置 + 法术威胁
            np.array([float(np.clip(self.elixir_budget, 0.0, 1.0))], dtype=np.float32),
            hold,
        ])
```

**`_plan_biases` 是「消费者」而非写入者**：它把 plan 的 3 个位置再翻译成 logit 软偏置（`follower.py:396-440`）：
- `v[16]`（suggested_card/4）→ `slot_bias[sug-1] += PLAN_CARD_BIAS(0.8)`（`:421-423`）
- **`v[PLAN_DIM-4+i]`（即最后 4 维）→ hold_mask 命中槽 −2.5**（`:424-427`）：
  ```python
        if v.shape[0] >= PLAN_DIM:
            for i in range(min(4, K_MAX)):
                if v[PLAN_DIM - 4 + i] > 0.5:
                    slot_bias[i] -= PLAN_HOLD_BIAS
  ```
- `v[8:16]`（region）→ 以 `_REGION_CENTERS` 为中心、曼哈顿半径 2 的格 +0.8（`:428-439`）

> ⚠️ **这是加 plan 维度的最大隐性坑**：`hold` 是**按 `PLAN_DIM` 反向偏移**读的。若在**尾部**追加新维度，`PLAN_DIM-4` 会指向错误的段，hold 偏置静默失效/错位——除非新维度插在 hold 之前或把偏移改成显式常量。

### 2.3 新增「标量特征」与「plan 维度」的改动点清单

**结论**：两条路径代价**不对称**。
- **新增 plan 维度**：`plan_mlp.0.weight` 有「前列拷贝+尾零」兼容分支 ⇒ 旧 ckpt **可加载**，新维从零学（`follower.py:135-139`）。
- **新增标量**：scalar 块在 `fused` **中段**（后面还跟 plan_f/belief_f）⇒ `enc_fc.weight` 形状变化**没有兼容分支**，会被**整体重置为随机初始化且不报错**（`follower.py:129-152` + 注释 `:172-183`）⇒ **必须 `--fresh`**。

| # | 改动点 | 加**标量**特征 | 加**plan**维度 | 证据 |
|---|---|---|---|---|
| 1 | 观测生产者 | 改 `observation.py:138-144`（或 env 派生则加在 `env_wrapper.observe` `:242-249`） | 不需要（plan 不是 obs） | 见 §2.1 |
| 2 | `observation_space` | 改 `env_wrapper.py:129-141` | 不需要 | 同上 |
| 3 | 维度常量 | `follower.py:248` `scalar_dim = 3` → `4`；`:253` `self.scalar_dim` 自动跟随 | `plan_space.py:98-112` 加 dataclass 字段；`:155-160` 加进 `to_vector`；`:187` `PLAN_DIM` **自动**重算 | `follower.py:247-258` |
| 4 | 单条编码 | `follower.py:336-340` 的 `_scalar_parts` | `follower.py` 无需改（plan 由 `plan_v` 整体进 MLP） | `follower.py:336-340` |
| 5 | 批量编码 | `follower.py:518-522` 的 `_scalar_parts` | 无需改 | `follower.py:518-522` |
| 6 | 融合/编码层形状 | `enc_dim` 自动 +1（`:257`），但 `enc_fc.weight`/`value_enc_fc.weight` 形状变 | `plan_mlp.0.weight` 形状变（`:255`） | `follower.py:255-258,293` |
| 7 | **ckpt 兼容** | **无兼容分支 ⇒ 静默重置 ⇒ `--fresh`** | **有兼容分支**（`plan_mlp.0.weight` 尾零） | `follower.py:129-152`（无分支即落 `:152` 注释「保持新初始化，不静默崩」）；`:135-139` |
| 8 | 偏置回读 | 不需要 | ⚠️ **必须改** `follower.py:424-427` 的 `PLAN_DIM-4+i`（若新维追加在 hold 之后） | `follower.py:424-427` |
| 9 | plan 生产者 | 不需要 | `belief_planner.py`（15 处 `PlanToken(`：`:400,433,482,545,569,607,616,652,695,702,733,770,788,805,881`）+ `prophet.py`（15 处：`:175…507`）填新字段 | grep `PlanToken(` |
| 10 | PPO 缓冲 | **无需改**（`t["obs"]` 整字典透传） | 无需改 | `ppo.py:194-196,328,371` |
| 11 | worker 协议 | **无需改**（obs 字典透传） | 无需改 | `workers.py:45-51` |
| 12 | BC/IL 采集 | `train_bc.py:65-77` 从 env 取 obs ⇒ 自动；但**手工造 obs 的离线脚本必须补键**，否则 `obs["新键"]` KeyError | 各 IL 脚本自动（`to_vector` 统一） | `train_bc.py:67,77`；脚本见下 |
| 13 | 跨进程对手池 | 形状变 ⇒ `_spec_to_policy` 的 `pol.load_state_dict(spec["state"])`（strict）会 **RuntimeError size mismatch** | 同左（但旧 ckpt 文件路径的 `load_checkpoint` 有兼容分支） | `run_league.py:461-465`；`train_solo.py:173` |
| 14 | 文档（R18） | 更新 `docs/model_vector_inventory_2026-09-20.md`（§A2/§c 表）、`AGENTS.md` R6 行 | 同步 `plan_space.py` 陈旧注释（`:11,:12`） | 红线 R18 |

> 第 12 项的手工造 obs 脚本（`next_card` 引用者）：`scripts/fl_il_to_bc.py`、`scripts/il_act_upper_bound.py`、`scripts/pomdp_ceiling_probe.py`、`scripts/il_probe_kingtower_cast.py`、`scripts/probe_credit_baseline.py`、`scripts/probe_encoding.py`、`scripts/probe_value_ln.py`、`scripts/diag_encoder_scale.py`、`src/clasher_new/rl/train_baseline.py`、`train_prophet.py`。**是否全部手工构造 obs：未定**（仅按 `next_card` 引用定位，未逐个读）。

**建议落点（供「手牌打分特征 + 前期卡组信息分」）**：
- **手牌打分（每帧可算）**：属「标量」→ 走 `scalar` 尾部/`_intent_obs_vec` 同款模式 ⇒ **必须 `--fresh`**。
- **前期卡组信息分（近似局级常量）**：若走 plan 尾部（如新 `PLACEMENT_HINTS` 之外的段）⇒ 旧 ckpt 可热启动（尾零），但**必须同时修 `follower.py:424-427`**；否则 hold 偏置错位（静默）。

### 2.4 R6「架构变更必须 --fresh」在代码里的体现

**结论**：R6 是**纪律 + 告警**，不是硬异常。真正的形状错位分成三种命运：①**显式 ValueError**（只对「不给维度」）；②**打印告警但继续**（缺 `enc_ln`/`grid_ln`、value 标志、intent_options 不一致）；③**完全静默**（其余形状不匹配，含 `enc_fc`）。`--fresh` 本身只是「不做 resume」。

**`--fresh` 的定义与语义**——`run_league.py:1510-1512`：
```python
    # 显式 --fresh 才从头（旧实验/想重跑时用）。
    ap.add_argument("--fresh", action="store_true",
                    help="忽略断点强制从头训练（默认：有断点就自动续训）")
```
`run_league.py:1629-1630`：
```python
    # 自动续训为默认：无 --fresh 时 resume=True（断点缺失/不存在时各入口会自行从头并提示）
    resume = not args.fresh
```
R6 原文（`docs/agents/redlines.md:20`）：
> `--fresh` **挡不住 `--main-init`**（须显式不传）。…**热启动 `--main-init` 必须显式传 `plan_dim`/`belief_dim`**（旧 ckpt 元数据 57/23 会把 main 建成旧维度 ⇒ `_sync_frozen_copy` shape 失配崩溃；当前 `plan_dim=58`、`belief_dim=563`）。

**热启动如何显式传 plan_dim/belief_dim**：
- **solo 主路径（正确做法）** `train_solo.py:1090-1102`：
  ```python
        # 显式 plan_dim/belief_dim=当前网络维度 → load_checkpoint 走"前列拷贝+尾零"
        # 兼容分支（旧 ckpt 的 plan_dim=57/belief_dim=23 元数据否则把 main 建成旧维度，
        # 之后 _sync_frozen_copy 拷进 PLAN_DIM 网络即 shape 失配崩溃）
        main = load_checkpoint(cfg.main_init, hidden_dim=cfg.hidden_dim,
                               plan_dim=PLAN_DIM, belief_dim=belief_dim, ...)
  ```
  `belief_dim` 来源：`train_solo.py:1075-1076` `belief_dim = len(BeliefInference(opp_deck=env.deck1, n_particles=128, seed=0).encode(None, None))`。
- **run 模式 main-init（R6 指的坑）** `run_league.py:980-983` **没有**传 `plan_dim`/`belief_dim`：
  ```python
    main = (load_checkpoint(cfg.main_init, hidden_dim=cfg.hidden_dim,
                            value_bypass=cfg.value_bypass,
                            value_independent=cfg.value_independent,
                            intent_options=bool(cfg.intent_save)) if cfg.main_init
            else FollowerPolicy(hidden=cfg.hidden_dim, plan_dim=PLAN_DIM, ...
  ```
  ⇒ 维度来自旧 ckpt 元数据（`follower.py:109-110` `pd = int(plan_dim or md.get("plan_dim") or PLAN_DIM)`），与 `PLAN_DIM=58` 不一致时无兼容（`belief_mlp`/`plan_mlp` 有分支，但 `_sync_frozen_copy` 的 strict `load_state_dict` 会崩）。
- **联赛跨进程 spec** 显式携带维度：`run_league.py:447-448`；逆操作 `_spec_to_policy` `:461-465`。

**`load_checkpoint` 的三类分支**（`follower.py:86-184`）：

| 情形 | 行为 | 位置 |
|---|---|---|
| 未显式给维度 | **硬报错**：`raise ValueError("FollowerPolicy 需要显式 plan_dim/belief_dim（禁止魔法默认值，P0-5）")` | `follower.py:225-226` |
| `plan_mlp.0.weight` / `belief_mlp.0.weight` / `entity_emb.weight` 形状小 | **兼容**：`tv.zero_()` 后拷贝前列（新维从零学） | `:135-139`、`:140-143`、`:144-149` |
| **其余任意形状不匹配（含 `enc_fc.weight`）** | **无分支 ⇒ 保留新网络随机初始化（静默）** | `:129-152`，注释 `:152` |
| 缺 `enc_ln.*` / `grid_ln.*` | **打印告警**（旧 LN 用默认仿射 1/0，trunk 是旧量级学的 ⇒ 不可续训） | `:160-171` |
| `value_bypass` / `value_independent` 显式值与元数据不一致 | **打印告警** | `:113-116`、`:119-122` |
| `intent_options` 不一致 | **打印告警**，并明说 `enc_fc` 被整体重置（**静默**） | `:172-183` |
| 启动前结构检查 | `check_policy_architecture`：缺/被换成 `nn.Identity` 的 `enc_ln`/`grid_ln` → 告警列表 | `diagnostics.py:143-161`；调用 `train_solo.py:1131-1133` |

告警文案逐字：
- `:169-171` `"[follower] ⚠️ {path} 缺 {'/'.join(_missing_ln)}.*（旧架构）：LayerNorm 为默认新初始化，该 ckpt 不可续训（要求 --fresh），只能当对照基线/对手池"`
- `:180-183` `"[follower] ⚠️ {path} 的 intent_options={...}，而目标网络 intent_options={...}：slot_head/sub_emb/enc_fc 形状不一致 ⇒ enc_fc 被整体重置（**静默**）⇒ 该 ckpt 不可续训，要求 --fresh；只能当对照基线/对手池"`

**真正的硬失败**只在 strict 加载处：`train_solo.py:171-173` `opp.load_state_dict(main.state_dict())`（`_sync_frozen_copy`）与 `run_league.py:465` `pol.load_state_dict(spec["state"])` —— 维度不一致会抛 `RuntimeError: size mismatch`（注释见 `train_solo.py:1097`「即 shape 失配崩溃」）。

### 2.5 加 plan/标量会不会让旧 ckpt 完全不可用？

- **加 plan 维度**：**不会**。`plan_mlp` 输出恒为 64，`enc_fc` 输入不变 ⇒ 只有 `plan_mlp.0.weight` 变形状，且有「前列拷贝+尾零」分支 ⇒ 旧 ckpt 可加载（新列从零学）。**这是设计上的「结构先行」兼容路径**（`plan_space.py:7-13`）。
- **加标量**：**会**。`enc_fc`（以及 `value_independent=True` 时的 `value_enc_fc`）输入形状变化，**没有兼容分支**，被整体重置为新随机初始化（静默）⇒ 旧 ckpt 名义上「能 load」但行为与训练时不同 ⇒ 必须 `--fresh`。

| 层名 | 加 plan 维度（58→59） | 加 1 个标量（3→4） |
|---|---|---|
| `plan_mlp.0.weight` | `(64, 58)` → **`(64, 59)`** ✅有尾零兼容（`follower.py:135-139`） | 不变 `(64,58)` |
| `plan_mlp.0.bias` | 不变 `(64,)` | 不变 |
| `enc_fc.weight` | **不变** `(128, 2731)` | `(128, 2731)` → **`(128, 2732)`** ❌无兼容 ⇒ 重置 |
| `enc_fc.bias` | 不变 `(128,)` | 不变 `(128,)` |
| `value_enc_fc.weight` | **不变** `(128, 2731)` | `(128, 2731)` → **`(128, 2732)`** ❌无兼容 ⇒ 重置 |
| `value_enc_ln.{weight,bias}` | 不变 | 不变 |
| `entity_emb.weight` | 不变 `(177, 8)` | 不变 |
| ckpt 元数据 | `plan_dim: 58` → `59`（`save_checkpoint` `follower.py:75`） | `enc_fc` 形状变、无元数据字段可记录 |

**形状证据**：`follower.py:255-258`（`Linear(plan_dim,64)` / `enc_dim=cnn_out+hand_dim+scalar_dim+64+64` / `Linear(enc_dim,hidden)`）、`:293`（`value_enc_fc = nn.Linear(enc_dim, hidden)`）；实测 ckpt 形状见 `docs/model_vector_inventory_2026-09-20.md:44,52,64,65,66`。

**其他会随架构变形的层**：

| 架构开关 | 变形层 | 位置 |
|---|---|---|
| `intent_options=True` | `slot_head` 出维 `6→11`、`sub_emb` 入维 `8→13`、`enc_fc` 入维 `+6` | `follower.py:48-55,251-253,285,298` |
| `num_entity` 扩容（词表 v2） | `entity_emb.weight` 行数（有兼容分支） | `:144-149` |
| `belief_dim` 变化 | `belief_mlp.0.weight`（有兼容分支） | `:140-143` |
| `value_independent` | 新增 `value_enc_fc/value_enc_ln/value_head_mlp`（8 个张量） | `:292-297` |

### 2.6 既有 feature-injection 先例（R12「能算的不许让网络猜」）

**结论**：仓库里**已有两条**「把确定性可算量直接注入观测」的成熟先例，**都遵循同一模式**：`env/侧 → obs 字典新键（或已有键的派生）→ 模型侧 `*_obs_vec`/MLP 转成定长块 → 追加到某一段`。另有若干「确定可算量」目前只存在于离线 dump、**尚未注入**。

**先例 1（最贴近本任务）：intent-save 的 `intent_slot` / `intent_age` 注入**
- 侧算好：`env_wrapper.py:122-124` 维护 `_intent_slot`/`_intent_age`（env 状态，确定性）。
- 注入 obs（`env_wrapper.py:244-248`）：
  ```python
        if self.intent_save and player_id == 0:
            # 攒费意图观测（**只在 agent 侧 p0 注入**；p1 由对手策略驱动，无意图状态）。
            # 默认关时不注入 ⇒ 观测 dict 与旧行为逐位一致。
            d["intent_slot"] = np.array([self._intent_slot], dtype=np.int32)
            d["intent_age"] = np.array([float(self._intent_age)], dtype=np.float32)
  ```
- 模型侧转定长块并**尾部追加到 scalar**（`follower.py:458-478` + `:336-340`）：
  ```python
    def _intent_obs_vec(self, obs):
        v = torch.zeros(1, INTENT_OBS_DIM, device=self.device)
        ...
        v[0, slot - 1 if 1 <= slot <= K_MAX else K_MAX] = 1.0   # index K_MAX = "无意图"档
        v[0, K_MAX + 1] = min(max(age, 0.0), 64.0) / 64.0
        return v
  ```
  ```python
        _scalar_parts = [elixir, time, next_card]
        if self.intent_options:
            _scalar_parts.append(self._intent_obs_vec(obs))
  ```
- 维度常量：`INTENT_OBS_DIM = K_MAX + 2 = 6`（`follower.py:53-55`），`scalar_dim` 随之 +6（`:253`），ckpt 元数据 `intent_options`（`save_checkpoint :82`）。
- 代价被显式记录（可当本次的「样板注释」）：`follower.py:172-183`「`enc_fc` 没有"尾部追加"兼容分支（标量在 fused 中段，不是尾部）⇒ 形状不一致会被整体重置（静默）⇒ 必须 `--fresh`」。

**先例 2：belief 的对手出牌事件通道**（不让 CNN 重新发现）
- 动机逐字（`belief.py:41-49`）：
  ```
  # 原来只有"概率分布"…，没有"事件"：对手 0.5s 前在
  # 桥头下了 Giant 这类即时信号要靠 CNN 从 grid 里重新发现（有效感受野 ~9 格，
  # 常与"我该在哪布防"落在不同感受野）。事件 token 把最近 k 次出牌直接喂给策略
  OPP_EVENT_K = 3
  OPP_EVENT_DIM = len(ENTITY_NAMES) + 3
  ```
- 写法（`belief.py:52-60` `_event_row`：卡 one-hot + `x/17` + `y/31` + `Δt/10`；`:358` `parts.append(opp_event_token(...))`）：确定性编码后拼进 belief token（540 维）。
- 注意：这只是「观测 token 注入」，**不改任何网络层形状**（belief token 维度公式 `:36-38` 自动扩展），因此不需要 `--fresh`（但旧 ckpt 的 `belief_mlp` 有尾零兼容分支，`follower.py:140-143`）。

**反例 / 缺口：确定可算量已算好但只在离线 dump，尚未注入**
- `scripts/fl_il_to_bc.py:295-307` 的 `_frames_to_next_elixir()` docstring 逐字：
  ```python
      """「距下 1 点圣水还有几帧」——**确定可算**（`battle.py:2881` 的三段回费）。…"""
  ```
- 它只写进 `dump`（`:522-541`，含 `next_card_cost`、`f2n_my`、`f2n_opp`、`b_hand`/`b_next` 等），**没有**进 `env.observe()`（`:551-556` 采集样本时用的是 `obs = env.observe(0)`）。这正是「能算但没注入」的现状 ⇒ 若要落「手牌打分 / 前期卡组信息分」，模板 = **先例 1 的写法**（新键 → `*_obs_vec` → 追加 scalar），代价 = **§2.3/§2.5 的 `--fresh`**。

**未定**：是否存在第三条「把派生量直接写进 `grid` 通道」的先例——本次只确认 `grid` 由 `observation.py:87-130` 纯从实体原始字段构建，未见派生打分通道。

### 2.7 训练/评估入口 + 标准 20k 协议 + `--fresh`

- 唯一训练/评估入口是 `src/clasher_new/rl/run_league.py`（`--mode solo|run|flow|eval`，`run_league.py:1490`）。标准 20k 协议在 `docs/agents/env.md` §2.2。命令必须在 `src/clasher_new` 目录下跑（`runs/...` 是 cwd 相对）。
- 相关脚本：`src/clasher_new/rl/train_bc.py`（BC）、`scripts/fl_il_to_bc.py`（IL→BC）、`scripts/random_eval_100.py`、`scripts/run_selftests.py`。评估：`--mode eval`（`run_league.py:1635-1641` → `evaluate_league`）或 `rl/evaluate.py::load_policy`（`:473-474`，维度从 ckpt 元数据读，不硬编码）。
- 标准 20k 协议（`docs/agents/env.md:42-54`，逐字）：
  ```bash
  python rl/run_league.py --mode solo --config economy --config-name <name> --fresh \
    --total-steps 20000 --steps-per-eval 2500 --n-eval-games 40 --eval-workers 12 --device cuda \
    --value-norm running --adv-norm scale --diagnose-every 10 \
    --hist-seed-dir runs/economy_9k_ft --hist-seed-dir runs/economy_9j \
    --ppo-epochs 4 --ppo-minibatch 32 --ppo-shuffle
  ```
  配套口径：`--eval-workers 12` **必须显式传**（`economy` 预设继承 dataclass 默认 `min(16, cpu_count())`，16 核 ⇒ 16，属 R1 风险档）；`runs/...` 真实位置是 `src/clasher_new/runs/`。

| 场景 | `--fresh` 用法 | 依据 |
|---|---|---|
| 全新架构（加标量/改 `intent_options`/`value_*`） | `--fresh`，且**不要**传 `--main-init`（`--fresh` 挡不住它） | `redlines.md:20` |
| 同架构续训 | **不加** `--fresh`（默认自动续训）；断点缺失会自动从头并提示 | `run_league.py:1510-1512,1629-1630` |
| 热启动（`--main-init`/`--mode run`） | 必须**显式**传 `plan_dim`/`belief_dim`——注意 **CLI 没有这两个 flag**，只能靠调用点代码（`train_solo.py:1098-1099` 已做；`run_league.py:980-983` **未做**） | `redlines.md:20`；`follower.py:109-110` |
| 与 `intent_save` 互斥 | `--intent-save` 帮助逐字：「架构变更 ⇒ 必须 --fresh」 | `run_league.py:1589-1591` |

> ⚠️ **结论性提醒**：**「加标量特征」在代码里没有任何护栏会报错**——`--fresh` 是唯一防线；若误续训，`enc_fc` 会被静默重置（`follower.py:152` + `:172-183` 的告警只在 `intent_options` 不一致时触发，标量变化**不触发任何告警**）。「加 plan 维度」则相反：旧 ckpt 可加载，但**必须同步修 `follower.py:424-427`**，否则 hold 偏置静默错位。

---

## 3. 对方手牌 / 圣水 / 卡组信息可得性【侦察③】

> 标注口径：**【实测】** = 亲自运行代码/加载产物验证；**【读码】** = 仅读代码推定；证据不足进「未定」。

### 3.1 `belief` 向量 563 维的逐段含义 / 谁生成 / 是否学习

**结论（已确证）**：563 = `2×8 + 2 + 5 + 3×180`，由 `BeliefInference.encode()` **生成**，三段来源均为**非学习**：①规则贝叶斯粒子滤波（`CycleBayesFilter`）、②计数统计（`StatisticalBelief`）、③事件通道（纯 one-hot + 归一化坐标/时间）。**神经段全仓零调用**（复核成立），且**本仓无该 checkpoint 产物**。

**【实测】维度分解**：

| 段 | 偏移 | 维 | 含义 |
|---|---|---|---|
| 对手手牌后验 `hand_probs` | `[0:8]` | 8 | 每张卡在**对手手牌**（`cycle[:4]`）的概率，**按 `belief.deck` 顺序** |
| 对手下一张 `next_probs` | `[8:16]` | 8 | 每张卡是对手**下一张进手**（`cycle[4]`）的概率，同序 |
| `[elixir_mean, uncertainty]` | `[16:18]` | 2 | 对手圣水估计 + 规则信念熵归一（`entropy()/log(8)`） |
| 风格倾向 `tendency_probs` | `[18:23]` | 5 | `["aggressive","defensive","cycle","spell_heavy","balanced"]` |
| 出牌事件通道 | `[23:563]` | 540 | = `OPP_EVENT_K(3) × OPP_EVENT_DIM(180)`；每条 180 = **卡 one-hot(177) + x/17 + y/31 + Δt/10** |

实测输出：`len(ENTITY_NAMES)=177`、`OPP_EVENT_DIM=180`、`BELIEF_DIM=563`、合计 563 ✔

**证据（逐字）**：
- `src/clasher_new/rl/belief.py:36-38`
  ```python
  def belief_token_dim(deck) -> int:
      return 2 * len(deck) + 2 + len(TENDENCIES) \
          + OPP_EVENT_K * OPP_EVENT_DIM
  ```
- `src/clasher_new/rl/belief.py:47-49`
  ```python
  OPP_EVENT_K = 3
  #: 每条事件维度：card_onehot(len(ENTITY_NAMES)) + x/17 + y/31 + Δt/10
  OPP_EVENT_DIM = len(ENTITY_NAMES) + 3
  ```
- `src/clasher_new/rl/belief.py:344-362`（生成处，逐段顺序即向量布局）
  ```python
      def encode(self, obs=None, opp_played=None) -> np.ndarray:
          """belief_token：规则向量 + 统计向量 + 事件通道 + 神经 token。"""
          parts = []
          st = self.state()
          parts.append(st.hand_probs.astype(np.float32))
          parts.append(st.next_probs.astype(np.float32))
          parts.append(np.array([st.elixir_mean, st.uncertainty], dtype=np.float32))
          parts.append(st.tendency_probs.astype(np.float32))
          ...
          parts.append(opp_event_token(self.event_history, now=self._now(obs)))
          if self.neural is not None and obs is not None:
              feat = build_feature(obs, opp_played)
              parts.append(self.neural.encode(feat).astype(np.float32))
          return np.concatenate(parts).astype(np.float32)
  ```

**是否学习 —— 复核结论：非学习（已确证）**
- 全仓 `neural=` 仅在 `belief.py:261` 出现（形参默认 `None`），**无任何调用方传入**。
  `src/clasher_new/rl/belief.py:261`
  ```python
  def __init__(self, opp_deck, use_rule=True, use_stat=True, neural=None, n_particles=128, seed=0):
  ```
- 官方注释逐字（`scripts/probe_encoding.py:75-76`）：
  ```
  ⚠️ 训练环 neural=None（train_solo.py:1171 只建 BeliefInference 规则+统计）⇒ **神经 belief token 未接入**
  ```
- **【实测】若接入神经段，维度会变成 627 而非 563**（`NeuralBeliefEncoder.belief_proj` 输出 `hidden=64`，`belief.py:186`）⇒ 563 这个数字本身即证明神经段关闭。
- **【实测】无训练产物**：`find . -name "belief_encoder*.pt"` → 0 命中。`rl/train_belief.py` 存在且能训练 `NeuralBeliefEncoder`（`train_belief.py:171`），但**从未接线到 `BeliefInference`**。
- **重要区分**：**token 生成器**非学习；**token 的消费端**是学习的 —— `belief_mlp = nn.Sequential(nn.Linear(belief_dim, 64), nn.ReLU())`（`follower.py:256`），并进 `fused`（`follower.py:347`）。网络只在 563 维上**学一层线性投影**，不产生信念。

**规则段的质量前提（已确证，关键）**：`hand_probs/next_probs` 是**「卡组已知」下的精确队列推断**，不是「卡组未知」的学习推断。
`src/clasher_new/rl/bayes_filter.py:9-14`（逐字）：
```
数学事实（暴力验证 3000 随机对局 × 59 步零反例）：
    对任意合法出牌流，从第 4 张起当前手牌集合 = 卡组 − 最近 4 张（互异），
    下一张进手 = 第 k−3 张打出的牌 —— 与开局洗牌顺序（40320 排列）无关。
    因此只要 8 张卡内容已知 + 出牌按序全观测，第 4 张起信念即精确 0/1，
    O(1) 滑动窗口即可；开局排列不可唯一复原（6144 兼容排列）不影响任何
    可观测量（手牌集合 / 下一张），不需要排列级粒子/全量重建。
```
`bayes_filter.py:178-184`：锁定流 `entropy()` 返回 **0.0**（"明牌"）。

**【实测】索引语义**：`hand_probs` 按 **`belief.deck` 顺序**返回，不是 `ENTITY_NAMES` 顺序 —— `bayes_filter.py:165` `return np.array([probs[c] for c in self.deck], ...)`；这也是 IL 管线必须额外存 `opp_deck` 的原因（见 §3.6）。

### 3.2 对手手牌是否可观测？`hidden_labels` 有哪些字段？哪些明确不进跟随者观测？

**结论（已确证）**：**跟随者观测里不可见**；引擎**同进程内真值可直接读**（§3.5）；env 仅在 `record_hidden=True` 时经 `info["hidden"]` 暴露，且该字段被文档逐字限定为「只允许训练期使用，绝不进跟随者观测」。

**`hidden_labels` 字段全集**（`observation.py:147-172`）逐字：
```python
def hidden_labels(battle, player_id: int = 0) -> dict:
    """特权隐藏状态标签（只允许训练期使用，绝不进跟随者观测）。

    用于信念模块监督：对手真实手牌 / 牌序 / 圣水 / 意图 / 风格。
    """
    opp_id = 1 - player_id
    ...
    # 对手当前手牌 = 循环前 4 张（可出牌）
    opp_hand = opp.cycle[:4]
    return {
        "opp_hand": _ids(opp_hand),            # (4,)
        "opp_next": _ids([opp.cycle[4]])[0],   # 下一张牌
        "opp_cycle": _ids(opp.cycle),          # 完整循环 (8,)
        "opp_elixir": np.array([opp.elixir], dtype=np.float32),
        "opp_crown": np.array([opp.get_crown_count()], dtype=np.int32),
        "opp_towers": np.array([opp.king_tower_hp, opp.left_tower_hp, opp.right_tower_hp],
                               dtype=np.float32),
        "my_elixir": np.array([me.elixir], dtype=np.float32),
        "my_hand": _ids(me.cycle[:4]),
        "time": np.array([battle.time], dtype=np.float32),
    }
```
共 **9 键**：`opp_hand(4) / opp_next(1) / opp_cycle(8) / opp_elixir(1) / opp_crown(1) / opp_towers(3) / my_elixir(1) / my_hand(4) / time(1)`（与 `scripts/probe_encoding.py:94-96` 清单一致）。

**明确不进跟随者观测的字段**：
- **红线注释①**（`observation.py:148` 逐字）：`"""特权隐藏状态标签（只允许训练期使用，绝不进跟随者观测）。`
- **红线注释②**（`env_wrapper.py:5` 逐字）：`- 只暴露玩家视角观测；特权状态通过 get_hidden_state() / get_prophet_state() 单独提供。`
- **`observation_space` 白名单**（`env_wrapper.py:129-135`）：只有 `grid / hand / elixir / next_card / time`（`intent_save` 开启时另加 `intent_slot / intent_age`）。**没有**任何 opp_* 键。
- **网络实际吃进的 obs 键**（`follower.py:319-340`）只有 `obs["grid"] / ["hand"] / ["elixir"] / ["time"] / ["next_card"]` ⇒ 即便 `obs` 里塞了 `opp_hand` 也不会被消费；而 `observe()` 压根不产出它。
- **`observe()` 只读自己的 player**（`observation.py:132-137`）：`p = battle.players[player_id]`，`hand = p.cycle[:5]` ⇒ **obs["hand"] 是我方手牌**。
- `hidden` 的唯一用途是**信念监督**（`train_belief.py:78-84` 用 `hidden["opp_hand"]/["opp_next"]` 造标签）与**诊断探针**（`evaluate.py:220-241` 只用来算 next-acc/Brier/手牌命中率，**不回灌策略**）；`info["hidden"]` 的产生受 `record_hidden` 开关控制（`env_wrapper.py:774-775`）。

**「不进网络」的精确口径（★ 局部冲突，原样并列）**：`probe_encoding.py:94` 原文是「**不进网络**，仅供信念模块监督」——这句针对的是**跟随者策略网络**（成立）。但**特权通道并非完全无人消费**：`ProphetPlanner` 直读 `fs["opp_cycle"]`（`prophet.py:234, 396, 504`）与 `fs["opp_elixir"]`（`prophet.py:281`），并用 `get_prophet_state()`（`env_wrapper.py:254-271`）取真值；它只经 **plan token（58 维）** 蒸馏进策略（`train_follower.py:119` 的 `plan_prophet_prob=0.3`），且 plan 只是**软偏置**。

### 3.3 对手圣水：能否观测/估计？现有是否有 estimator？

**结论（已确证）**：
1. **不进观测**（同 §3.2 白名单）。
2. **有唯一一个规则估计器**：`BeliefInference._elixir_est`（`belief.py:288-298, 314-315`），输出到 belief token 的**第 16 维**（`belief.py:350`）。
3. **它有三处已确证的偏差源**（读码，逐条给行号）：固定 2.8 s/点回费、忽略死亡圣水馈赠、初始化 5.0 且无观测校正。**是否造成可测偏差：未验证**。
4. **特权真值两条独立通道**：`hidden_labels["opp_elixir"]`（`observation.py:165`）与 `get_prophet_state()["opp_elixir"]`（`env_wrapper.py:262`）。
5. **【实测】录像里逐帧存了对手圣水真值**：`frame["elixir1"]`（`replay.py:182`）—— schema 3 与 schema 5 文件均实测存在。

**证据（逐字）**：
- `src/clasher_new/rl/belief.py:288-298`
  ```python
      def _tick_elixir(self, obs):
          """按观测时间推进圣水估计（决策间隔回复率近似 2.8s/点）。"""
          ...
          self._elixir_est = min(10.0, self._elixir_est + dt / 2.8)
  ```
- `belief.py:313-315`
  ```python
              from card_utils import Card
              self._elixir_est -= Card(card).elixir
          self._elixir_est = float(np.clip(self._elixir_est, 0.0, 10.0))
  ```
- `belief.py:326-327`（`elixir_std` 是**启发式伪标准差**，非后验方差）
  ```python
          st.elixir_mean = self._elixir_est
          st.elixir_std = max(0.5, abs(self._elixir_est - 5.0) / 5.0)
  ```

**偏差源 1（回费率与引擎不一致，已确证）**：估计器硬编码 2.8 s/点；引擎是**分阶段**的：
`src/clasher_new/battle.py:2880-2881`
```python
        for each in self.players:
            each.regenerate_elixir(dt, 2.8 if self.time < 120 else 1.4 if self.time < 240 else 2.8/3)
```
⇒ 120 s 后引擎回费快 **2×**、240 s 后 **3×**，而 `_elixir_est` 仍按 2.8 累加 ⇒ **加时段估计系统性偏低**。

**偏差源 2（忽略死亡圣水馈赠，已确证）**：`battle.py:377-388` `_death_elixir_gift`（`manaOnDeath`，含给对手）会计入真实 `p.elixir`，估计器只减卡费、不感知 ⇒ 漏加。

**偏差源 3（死参数，已确证）**：`update(..., elixir_est=None)` 这个外部校正入口**全仓无调用方**（grep `elixir_est=` 仅命中定义行 `belief.py:300`）⇒ 存在但形同虚设。

**消费方**：`belief_planner.py:10-12`（逐字）
```
- 圣水/手牌按"记忆即明牌"处理：punish 读 belief.elixir_mean，anti_spell/save_ace 读
  belief.hand_probs（粒子后验/后期确定性；信息不足时用概率阈值保守化），
```
实际读点：`belief_planner.py:554`（`belief.elixir_mean > 2.5`）、`672-674`（`elixir_mean > p.elixir + SAVE_BEHIND_ELIXIR`）。

**「未定」**：`_elixir_est` 的**实际误差分布从未被本仓标定**（无 `probe_belief_elixir` 类仪器）；未找到任何用它对齐 `hidden["opp_elixir"]` 的读数文档。

### 3.4 「对手已打出的牌」是否被记录？录像 schema 里每帧/每局存了什么？

**结论（已确证，且是本次最有利的一条）**：
- **逐帧记录**：`opp_played`（结构化 `[{card,x,y}]`）在**两套** replay 里都有。
- **逐局记录双方完整 8 卡卡组**：`meta["decks"] = [deck0, deck1]`（schema ≥ 5【实测】有；schema 3 **没有**）。
- **逐帧记录对手圣水真值** `elixir1`（schema 3 起就有）。
- **不记录**：任何帧的对手手牌 / 对手 cycle。
- ⇒ **「对手卡组信息分」在本仓是「确定性可算」的**（卡组已知 + 已出牌全记录 ⇒ 剩余牌/手牌集合可精确算），完全满足 **R12**「能算的不许让网络猜」。

**`battle_snapshot` 帧字段（联赛/仪表盘录像）**：`src/clasher_new/rl/replay.py:146-210` docstring 逐字：
```
        每帧含：时间、动作 bundle、奖励、对手出牌、双方塔血/圣水/皇冠、存活实体列表。
```
字段清单（`replay.py:174-193`）：`t / bundle / reward / opp_played / towers0 / towers1 / elixir0 / elixir1 / et / et_src / crown0 / crown1 / v0 / v1`；`entities` 每条 15 列（schema 5，`replay.py:194-209`）。关键两行：
```python
        "opp_played": info.get("opp_played"),
        ...
        "elixir1": float(p1.elixir),
```
另有可选帧键 `cards`（**我方**本步实际打出，`run_league.py:148-151`）。

**局级 meta** —— `src/clasher_new/rl/run_league.py:126-128`：
```python
    def set_decks(self, deck0, deck1):
        """记录本局双方实际卡组（play_pair 复用 env 时在 reset 之后才可知）。"""
        self.meta["decks"] = [list(deck0), list(deck1)]
```
（`run_league.py:119-120` 注释逐字：`# decks = (deck0, deck1)：双方本局实际卡组（卡名列表）。dashboard 的卡牌使用 / # 统计据此还原"这一局双方各带了什么"，也是卡组构成统计的数据源。`）

**【实测】三类真实产物对照**：

| 产物 | schema | `meta["decks"]` | 帧内 `opp_played` | 帧内 `elixir1` | 帧内对手手牌/cycle |
|---|---|---|---|---|---|
| `runs/archive/economy_100k_v1_ungated/replays/league_0.pkl` | 3 | **无** | ✅ `[{'card':'Minions','x':8.5,'y':26.5}]` | ✅ 2.178… | ❌ |
| `runs/il_readout_mixR337/replays/league_3000.pkl` | 5 | ✅ 双方 8 卡 | ✅ | ✅ 5.178… | ❌ |
| `runs/_il_probe_kt/replays/league_986.pkl` | 5 | ✅ + `tower_geom` | ✅ + `cards` | ✅ 4.178… | ❌ |

帧键实测（schema 5）：`['bundle','crown0','crown1','elixir0','elixir1','entities','et','et_src','opp_played','reward','t','towers0','towers1','v0','v1']` + 可选 `cards`。

**单局信念 replay（另一套，非联赛）**：`src/clasher_new/rl/replay.py:36-45`（`EpisodeReplay`，`SCHEMA_VERSION = 2`）每步存 `obs / bundle / reward / opp_played / time` + 可选 `hidden`（含真实 `opp_cycle`/`opp_hand`/`opp_elixir`）。**这是唯一逐帧存「对手手牌真值」的格式**，且它是**训练用特权数据**，不是联赛录像。

### 3.5 引擎侧属性：一手牌 / 8 张卡组 / 已用牌循环 / 下一张

**结论（已确证）**：
- **`PlayerState.cycle` 一个列表同时承担三件事**：`cycle[:4]` = 当前手牌（可出），`cycle[4]` = 下一张，整体 = 完整 8 张卡组（**牌集合恒定，只是循环位移**）。**不存在**独立的 `deck` / `hand` / `next_card` / 已用牌集合字段。
- **我方视角（同一进程内）完全公开可读**：都是普通实例属性（无 property 守卫、无权限检查）；`observe()` 自己就是这么读的。**跨进程/经 API 则必须显式特权通道**（§3.2）。

| 量 | 属性 | file:line |
|---|---|---|
| 8 张卡组（牌集合） | `PlayerState.cycle`（`__init__` 里 `self.cycle = cycle_queue[:]`，长度恒 8） | `player.py:8` |
| 当前手牌（4 张可出） | `p.cycle[:4]` | `player.py:37`（`can_play_card`）、`observation.py:160` |
| 下一张进手牌 | `p.cycle[4]`（有 `get_next_card()` 取值器） | `player.py:50-52`、`observation.py:137` |
| 圣水 | `PlayerState.elixir` | `player.py:9` |
| 上一张打出的牌（Mirror 用） | `PlayerState.last_card` | `player.py:11`、`player.py:47` |
| 出牌循环推进 | `cycle.remove(card); cycle.append(card)` | `player.py:45-46`；引擎路径 `battle.py:2938-2940` |
| 全局实例 | `battle.players[0]` / `[1]`；`env.deck0` / `env.deck1` | `env_wrapper.py:178-182`、`103-104` |

逐字证据：
```python
class PlayerState:
    def __init__(self, player_id, cycle_queue, elixir, tower_hps=(4824, 3052, 3052)):
        self.player_id = player_id
        self.cycle = cycle_queue[:]
        self.elixir = elixir
        ...
        self.last_card = None  # M1: 镜像法术需要记录上一张使用的卡
```
（`src/clasher_new/player.py:5-15`）
```python
    def can_play_card(self, card_name):
        return (card_name in self.cycle[:4] and
                self.elixir >= Card(card_name).elixir and
                self.king_tower_hp > 0)

    def play_card(self, card_name):
        """Update the player's deck when playing a card."""
        if not self.can_play_card(card_name): return False
        self.elixir -= Card(card_name).elixir
        self.cycle.remove(card_name)
        self.cycle.append(card_name)
        if card_name != 'Mirror': self.last_card = card_name
        return True

    def get_next_card(self):
        """Return the next card in cycle, if known."""
        return self.cycle[4]
```
（`src/clasher_new/player.py:36-52`）
```python
            if card_name in p.cycle:
                p.cycle.remove(card_name)
                p.cycle.append(card_name)
            ...
            p.elixir -= actual_cost
            if _hand in p.cycle:
                p.cycle.remove(_hand)
                p.cycle.append(_hand)
```
（`src/clasher_new/battle.py:2936-2946`）

**公开可读性的精确口径（已确证）**：
- `observation.py:132-137`：`p = battle.players[player_id]` … `p.cycle[:5]` / `p.cycle[4]` —— 只读**自己那一侧**。
- `observation.py:152-154`：`opp_id = 1 - player_id; opp = battle.players[opp_id]` —— 特权读对面，仅在 `hidden_labels()` 内。
- 因此：**同一进程内无任何机制阻止读对手 `cycle`/`elixir`**（普通属性）；「不可观测」是**调用约定/文档纪律**，不是代码强制。

### 3.6 IL 数据管线 `scripts/fl_il_to_bc.py` 的样本里有什么

**结论（已确证 + 【实测】）**：
- **BC 样本 pkl 里没有「双方手牌」**：5 元组 `(obs, belief_tok, plan_vec, bundle, masks)`；`obs` **只有我方手牌**。
- **旁挂 npz（`--dump-frames`）里存了对手的** `opp_hand / opp_cycle / opp_elixir / opp_deck`，**但没有我方完整手牌**（只有 `next_card / my_elixir`）。
- **对手卡组**同时进两处：npz 的 `opp_deck`（8 维，序 = `env.deck1`）与引擎 `env.deck1`（源自回放自带双方卡表）。
- **重要保真缺口**：npz 的 `opp_hand/opp_cycle` 是**我方引擎重建值**，不是人类原始手牌 —— 出牌由 `_force_hand` **对换**注入；作者自己逐字写明「人类手牌顺序不可观测 ⇒ 这是**假设**，不是重建」。

逐字证据：
- `scripts/fl_il_to_bc.py:519-538`（dump 键）
  ```python
              dump.append(dict(
                  frame=k, act=int(has_team_play), label_ok=int(label_bundle is not None),
                  opp_play_count=int(st["opp_plays"]),
                  ...
                  b_hand=np.asarray(bst.hand_probs, dtype=np.float32),
                  b_next=np.asarray(bst.next_probs, dtype=np.float32),
                  b_elixir=float(bst.elixir_mean), b_unc=float(bst.uncertainty),
                  opp_elixir=float(p1.elixir), f2n_opp=float(_frames_to_next_elixir(p1.elixir, env.battle.time)),
                  opp_hand=np.asarray([_card_idx(c) for c in p1.cycle[:4]], dtype=np.int32),
                  opp_cycle=np.asarray([_card_idx(c) for c in p1.cycle], dtype=np.int32),
  ```
- `scripts/fl_il_to_bc.py:598-602`
  ```python
          #: 对手牌组**顺序**（`belief.next_probs` 的下标语义 = `env.deck1` 的位置）
          #: ⇒ 没有它就无法把信念的 argmax 映射回卡名（「预判下一手」的必需字段）
          arr["opp_deck"] = np.asarray([_card_idx(c) for c in env.deck1], dtype=np.int32)
          ...
          np.savez_compressed(os.path.join(dump_dir, "feat_%04d.npz" % idx), **arr)
  ```
- `scripts/fl_il_to_bc.py:338-343`（保真缺口，逐字）
  ```python
  def _force_hand(ps, card, hand_slot=None):
      """把手牌强制成含 `card`（人类手牌顺序不可观测 ⇒ 这是**假设**，不是重建）。
      ...
      只在 `cycle` 内部做**对换** ⇒ 长度恒 8、牌集合不变；但 `next_card`（`cycle[4]`）会变
  ```
- `scripts/fl_il_to_bc.py:418-429`（对手出牌确实被注入引擎 ⇒ cycle 随之推进）
  ```python
              for e in bucket["o"]:                      # 对手侧：只注入，不产标签
                  ...
                  _force_hand(p1, card)
                  c = _card_cost(p1, card)
                  if c is not None:
                      _force_elixir(p1, c)
                  env.battle.deploy_card(1, card, ...)
                  injected.append(card)
  ```
- `scripts/fl_il_to_bc.py:365-366`（卡组来源 = 回放自带双方卡表）
  ```python
      deck0 = list(rep["_decks"]["team"])
      deck1 = list(rep["_decks"]["opponent"])
  ```

**【实测】npz 实际键（`runs/_fl_il_frames/feat_0000.npz`，25 键）**：
```
act, b_elixir, b_hand, b_next, b_unc, ev_c, ev_dt, ev_x, ev_y, f2n_my, f2n_opp,
frame, holdout, label_ok, my_elixir, next_card, next_card_cost,
opp_cycle, opp_deck, opp_elixir, opp_hand, opp_play_count, opt_cell, opt_slot, tag, time
```
形状：`opp_hand (298,4) int32`、`opp_cycle (298,8) int32`、`opp_deck (8,) int32`、`opp_elixir (298,) float32`、`b_hand (298,8) float32`。**无 `my_hand`**。

**【实测】BC pkl 实际结构（`runs/_fl_il_bc/holdout/bc_fl_0001.pkl`）**：list，每样本 5 元组：
```
0: dict keys=['elixir','grid','hand','next_card','time']   ← hand (5,) = 我方手牌
1: ndarray (563,)      ← belief_tok（含对手手牌后验 8 维，但非真值）
2: ndarray (58,)       ← plan_vec
3: ActionBundle
4: list len 2          ← masks
```

**时序注意（已确证，来自读码）**：dump 在**本帧对手出牌注入之后、`belief.update` 之前**（注入 `417-429` → dump `499-538` → `belief.update` 在 `567`）。`scripts/il_opp_prediction.py:17` 也逐字标注：
```
⚠️ 对手出牌时刻由 `opp_play_count` 的**增量**给出（dump 在本帧 `belief.update` **之前**）
```
⇒ 用 `opp_hand` 做「手牌 gold label」时，必须明确它与 `b_hand` 相差一个 update 步。**该语义是否已被下游正确处理：未逐一验证（未定）**。

### 3.7 三分类总表：可观测 / 可算 / 不可得

| 量 | 分类 | 依据（file:line） | 备注 |
|---|---|---|---|
| 我方手牌 4 张 + 下一张 | **可观测**（进 obs） | `observation.py:132-137`、`env_wrapper.py:131-133` | `obs["hand"]=cycle[:5]` |
| 我方圣水 | **可观测** | `observation.py:141` | |
| 场上所有单位（含敌方）+ 15 通道属性 | **可观测** | `observation.py:90-130` | `is_opponent` 通道区分敌我 |
| 对手**本步**出牌（卡名+世界坐标） | **可观测**（信息在 `info`，进 belief 事件通道；**不在 obs dict**） | `env_wrapper.py:497-529, 748`；`belief.py:306-307` | 物理上=单位落地可见 |
| 双方塔血 / 皇冠 / 时间 | **可观测** | `observation.py:138-144`、`replay.py:179-190` | |
| **对手 8 卡卡组** | **构造已知**（非观测） | `env_wrapper.py:103-104, 178-182`；`run_league.py:126-128`；`fl_il_to_bc.py:365-366` | 全仓**无卡组推断模块**；belief 以 `opp_deck=` 显式入参（`belief.py:261`） |
| 对手**已打出牌集合**（累计） | **可算**（确定性） | `info["opp_played"]` 逐帧 `env_wrapper.py:748`；录像 `replay.py:178`（【实测】）；IL `opp_play_count` 增量 `il_opp_prediction.py:78-79` | 满足 R12 ⇒ 应特征注入 |
| 对手**手牌集合**（≥4 张出牌后） | **可算**（确定性、与洗牌无关） | `bayes_filter.py:9-14`（逐字数学事实）；`bayes_filter.py:151-165` | 锁定流 = 精确 0/1 |
| 对手**下一张**（≥4 张后） | **可算** | `bayes_filter.py:167-176`；`bayes_filter.py:143-147` | = 第 k−3 张打出的牌 |
| 对手剩余未出牌 = 卡组 − 已出牌 | **可算** | 由上面两条合成 | 「卡组信息分」的确定性基础 |
| 对手圣水**估计** | **可算/可估**（有估计器，**有偏**） | `belief.py:288-298, 313-315, 326` | 固定 2.8 s vs 引擎分阶段 `battle.py:2881`；忽略 `_death_elixir_gift` `battle.py:377-388`；`elixir_est=` 死参数 `belief.py:300` |
| 对手圣水**真值** | **不在观测；特权可得 + （我方录像）有真值** | `observation.py:165`、`env_wrapper.py:262`、`replay.py:181-182`（【实测】`elixir1`） | 离线研究可直接用 `elixir1` |
| 对手**手牌真值** | **不在观测；仅特权 `hidden`** | `observation.py:160-162`；`env_wrapper.py:774-775` | 联赛录像**不含** |
| 开局前 3 张的对手手牌 / 初始洗牌顺序 | **不可得** | `bayes_filter.py:13-14`（"开局排列不可唯一复原（6144 兼容排列）"） | 只有后验概率，无真值 |
| 对手手牌**在手中的槽位顺序** | **不可得** | `fl_il_to_bc.py:339`（"人类手牌顺序不可观测 ⇒ 这是**假设**，不是重建"） | 只影响 `next_card` 语义 |
| IL 数据里人类对手的**真实**手牌 | **不可得（只是重建）** | `fl_il_to_bc.py:338-350, 531` | npz `opp_hand` = 我方引擎 `p1.cycle[:4]` |
| BC 样本里的「我方+对手」手牌 | **不存在** | pkl 5 元组【实测】；npz 无 `my_hand`【实测】 | 需 frame 号跨两个产物拼 |

### 3.8 给后续实现的四点硬提示（均为上面已确证事实的直接推论）

1. **「卡组信息分」在本仓是「卡组已给定」前提下的确定量**（§3.1 末、§3.4）——不存在「从观测推断卡组」的现成通路；若要「前期卡组信息分」，需先明确它是**未知卡组**语义（本仓无此场景）还是**已知卡组的剩余牌/手牌推断**语义（本仓可直接算，符合 R12）。
2. **对手手牌自第 4 张出牌起是 0/1 精确量**（`bayes_filter.py:9-14`），前 3 张是粒子后验 ⇒ 打分函数必须**分段**，否则会拿「精确 0/1」和「均匀先验」当同一量纲。
3. **`hand_probs` 的下标是 `belief.deck` 顺序而非 `ENTITY_NAMES`**（`bayes_filter.py:165`、`fl_il_to_bc.py:598-600`）⇒ 任何跨来源拼接必须先对齐到 `opp_deck`。
4. **对手圣水唯一估计器有已确证的阶段偏差**（`battle.py:2881` vs `belief.py:298`）——若打分要含圣水项，先把回费率同源（R7 尺度单常量源精神），否则分数在 120 s 后会系统性漂移。

---

## 4. 人类开局试探经验标定【侦察④】

> 数据 = FirstLight `IL_Replay` 的本地解析产物（**唯一分片**）。所有数字由 `/tmp/il_opening_probe.py` 复算、`/tmp/il_opening_md.py` 渲染；两跑 `md5 = 7b0b01f57c63dab4b4f40ada645ea404`（确定性）。**未修改仓库任何文件、未做任何 git 操作。**

### 4.1 数据集：实际路径、格式、规模

| 项 | 值 | 备注 |
|---|---|---|
| **解析产物（分析输入）** | `/mnt/e/fl_il_data/replays_part000000.jsonl` | 20,441,906 B |
| 上游 parquet（回放） | `/mnt/e/fl_il_data/replays/part-000000.parquet` | 17,432,059 B |
| 上游 parquet（动作，冗余对账） | `/mnt/e/fl_il_data/actions/part-000000.parquet` | 22,697,255 B |
| 转换脚本 | `scripts/_fl_il_extract.py`（parquet→jsonl） | 只做格式转换 |
| 下游消费 | `scripts/fl_il_to_bc.py --jsonl <上面这个>`（`reconcile`/`samples`） | CLI 输入即此 jsonl |
| **局数** | **5,000** | 本机只有 `part-000000` 一个分片 |
| **分析单位（局×侧）** | **10,000** | 两侧都是人类 |
| 事件总数 | 384,275 | `play_card` 365,362 / `activate_ability` 18,913 |
| 时长 `dur` | median 218.55 s（p25 184.55 / p75 288.55） | `>300s` 1,009 局 |
| 卡键不可映射 | 1,354（0.37%） | 全部是 `void` |
| Mirror 事件 | 54 | 费用按 `last+1` |
| 无坐标事件 | 0 | 有 `native` + `grid_cell_floor` 双份坐标 |
| 模式 | Ranked 4451 / 1v1 Battle 325 / Grand Challenge 91 / Ladder 59 / Normal Battle 48 / 1v1 21 / Classic 4 / Showdown 1 | |

**每局字段（15）**：`tag, bt, gm, res, tc, oc, ec, wc, rpt, sv, dur, agg, decks, meta, ev`。
**事件元组** `ev[i] = [kind, side, card_key, tick_20hz, native_x, native_y, grid_floor_x, grid_floor_y]`；`kind 0=play_card / 1=activate_ability`；`side 0=team / 1=opponent`；`card_key` 为 RoyaleAPI kebab（可带 `-ev1/-ev2/-ev3/-hero`）；秒 = `tick/20`。`decks[side]=[[key,level]×8]`、`agg[side]={total:[张数,圣水],leaked,…}`、`meta[side]={crowns,avg,leak,cycle4,final,tower_card}`。

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 /tmp/il_opening_probe.py --out /tmp/il_opening_probe.json
/usr/bin/python3 -c "import json;d=json.load(open('/tmp/il_opening_probe.json'));print(json.dumps(d['dataset'],ensure_ascii=False,indent=1))"
```

### 4.2 口径定义（写死在脚本里）

| 口径 | 定义 |
|---|---|
| **开局窗口** | `t0 = 该局最早 play_card 的秒数（任一方）`；`W_T = {t0 ≤ t < t0+T}`，主 **T=30 s**，敏感性 **20/45 s** |
| 锚敏感性 | `game_first`（主）/ `side_first`（该侧首牌）/ `abs0`（t=0 起） |
| 几何 | 与 `src/clasher_new/arena.py:19-20` 同源：`RIVER_Y1=15, RIVER_Y2=16`；桥柱带 `x∈[2,5)∪[13,16)` |
| **三区（side 相对）** | `bridge = 14≤gy≤17`（距河<2格）；`own = gy≤13`(team)/`gy≥18`(opp)；`enemy` 反之 |
| 以河为界 | `own_side = gy≤15`(team)/`gy≥16`(opp) |
| 塔后 | `gy≤6`(team)/`gy≥25`(opp) |
| 卡费 | `card_aliases.resolve_card` + `card_utils.Card.elixir`；`Mirror`=上一张+1；`void` 剔除并计数 |

> ⚠️ **自查出的关键前提**：回放是**单一蓝方视角帧** —— team 侧 `gy≤5` 占 15.1%、opponent 侧仅 1.4% ⇒ 坐标**不随 side 翻转**，但「己方半场」**必须按 side 镜像**。第一版没镜像，对手侧位置读数全反（已修）。

| 秒（t=0 起） | n | mean | p25 | median | p75 | p90 |
|---|---|---|---|---|---|---|
| 局级首张牌 | 9,994 | 11.08 | 9.05 | **9.85** | 11.30 | 13.75 |
| 该侧首张牌 | 9,982 | 12.59 | 9.60 | **11.05** | 13.25 | 16.64 |

⇒ 主窗口 T=30 的实际时间范围中位 ≈ **[9.85 s, 39.85 s)**。

```bash
/usr/bin/python3 -c "import json;d=json.load(open('/tmp/il_opening_probe.json'));print(d['dataset']['t_game_first_card']);print(d['dataset']['t_side_first_card'])"
```

### 4.3 窗口内统计

#### 4.3.1 每局出牌数分布（2a）

| T | n | mean | median | p25 | p75 | p90 | 窗口内花费中位 | 不同卡中位 |
|---|---|---|---|---|---|---|---|---|
| 20 s | 9,994 | 2.98 | 3.00 | 2.00 | 4.00 | 4.00 | 9.00 | 3.00 |
| **30 s** | 9,994 | **4.21** | **4.00** | 3.00 | 5.00 | 6.00 | **13.00** | **4.00** |
| 45 s | 9,994 | 5.96 | 6.00 | 5.00 | 7.00 | 8.00 | 19.00 | 5.00 |

**T=30 s 直方图（单位 = 局×侧，n=9,994）**

| 出牌数 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8+ |
|---|---|---|---|---|---|---|---|---|---|
| n | 20 | 69 | 406 | 2,453 | 3,500 | 2,150 | 872 | 384 | 140 |
| % | 0.20 | 0.69 | 4.06 | 24.54 | **35.02** | 21.51 | 8.73 | 3.84 | 1.40 |

**每局（两侧合计）**：median 8、mean 8.41（p25 7 / p75 10 / p90 11）。

**锚敏感性（T=30 s）**

| 锚 | n | 出牌 mean | median | 花费中位 | own% | bridge% | enemy% |
|---|---|---|---|---|---|---|---|
| `game_first`（主） | 9,994 | 4.21 | 4.00 | 13.00 | 64.21 | 26.32 | 9.47 |
| `side_first` | 9,982 | 4.38 | 4.00 | 14.00 | 64.09 | 26.41 | 9.50 |
| `abs0` | 10,000 | 2.90 | 3.00 | 9.00 | 65.23 | 25.12 | 9.64 |

```bash
/usr/bin/python3 -c "
import json;d=json.load(open('/tmp/il_opening_probe.json'))
for k in ('game_first_T20','game_first_T30','game_first_T45'):
    w=d['windows'][k];print(k,w['plays_per_unit'],w['plays_hist'])
print('per_game',d['plays_per_game_T30_window']);print('anchors',{k:v['plays_per_unit'] for k,v in d['anchor_sensitivity_T30'].items()})"
```

#### 4.3.2 出牌卡费分布（2b）

**出牌层**（T=30 s，覆盖 99.7575%，未映射 102 次 = `void`）

| 费用档 | 1 | 2 | 3 | 4 | 5 | 6+ |
|---|---|---|---|---|---|---|
| n | 6,038 | 8,823 | 9,994 | 10,302 | 3,711 | 3,084 |
| % | 14.39 | 21.03 | 23.82 | **24.56** | 8.85 | 7.35 |

**单位首张牌费用档**（n=9,974）

| 费用档 | 1 | 2 | 3 | 4 | 5 | 6+ | ? |
|---|---|---|---|---|---|---|---|
| n | 1,483 | 2,256 | 2,446 | 2,454 | 725 | 608 | 2 |
| % | 14.87 | 22.62 | 24.52 | **24.60** | 7.27 | 6.10 | 0.02 |

⇒ 首张牌 **≤2 费 = 37.49%**、**≥4 费 = 37.97%**：首牌费用接近平坦，**不是「开局必出便宜牌」**。

```bash
/usr/bin/python3 -c "
import json;d=json.load(open('/tmp/il_opening_probe.json'));w=d['windows']['game_first_T30']
print(w['play_cost_hist']);print(w['play_cost_by_unit_first']);print(w['first_play_is_spell'])
fc=d['opening_first_card_T30'];n=fc['n'];ch={k:v['n'] for k,v in fc['cost_hist'].items()}
print('<=2 %.2f%%  >=4 %.2f%%'%(100*(ch.get('1',0)+ch.get('2',0))/n,100*(ch.get('4',0)+ch.get('5',0)+ch.get('6+',0))/n))"
```

#### 4.3.3 出牌位置分布（2c，分卡费档）

**T=30 s 总体**（41,854 次有坐标出牌）

| 区 | own | bridge（距河<2格） | enemy |
|---|---|---|---|
| n | 27,003 | 11,068 | 3,983 |
| % | **64.21** | **26.32** | **9.47** |

**以河为界**：己方侧 **89.98%**（37,841）vs 敌方侧 **10.02%**（4,213）。桥头桶拆分（全局限时）：桥柱带 61,976 次 / 只到河边不在桥柱带 47,607 次。

**分卡费档 × 区（% 为该费档内占比）**

| 费档 | own n | own% | bridge n | bridge% | enemy n | enemy% |
|---|---|---|---|---|---|---|
| 1 | 4,119 | **68.22** | 1,917 | 31.75 | **2** | **0.03** |
| 2 | 4,492 | 50.91 | 3,056 | 34.64 | 1,275 | 14.45 |
| 3 | 6,448 | 64.52 | 1,900 | 19.01 | 1,646 | 16.47 |
| 4 | 6,987 | 67.82 | 2,580 | 25.04 | 735 | 7.13 |
| 5 | 2,660 | 71.68 | 979 | 26.38 | 72 | 1.94 |
| 6+ | 2,290 | 74.25 | 627 | 20.33 | 167 | 5.42 |

**硬事实**：30 s 窗口内 **1 费牌 6,038 次里只有 2 次（0.03%）进敌方半场**。

```bash
/usr/bin/python3 -c "
import json;d=json.load(open('/tmp/il_opening_probe.json'));w=d['windows']['game_first_T30']
print(w['position_all']);print(w['position_own_side_vs_enemy_side']);print(json.dumps(w['position_by_cost_bucket'],indent=1))"
```

#### 4.3.4 首张牌的圣水余量 → **圣水不可观测**（2d）

回放**无逐帧圣水字段**（§4.5.1 字段取证）：`battle.<side>.players[0]` 只有整局 `average_elixir / elixir_leaked / four_card_cycle_elixir`，`aggregate_stats` 只有整局计数。既有结论同：**直接重放只走通 35–62%，主缺口 = 圣水不可观测**（`docs/agents/reward_surveys.md`）。

**可算的区间代理**（首牌前该侧累计花费 = 0）：`remaining = min(10, 5+t0/2.8) − 期间漏费`，漏费只有整局总量：

| 口径 | n | mean | p25 | median | p75 | p90 | min | max |
|---|---|---|---|---|---|---|---|---|
| 上界（不扣漏费） | 9,982 | 9.01 | 8.43 | **8.95** | 9.73 | 10.00 | 6.89 | 10.00 |
| 下界（扣整局漏费） | 9,982 | 5.40 | 3.59 | **6.33** | 7.74 | 8.34 | 0.00 | 8.67 |

⇒ 人类**首牌≈手握 6.3–9.0 圣水时打出**（中位，即「等到接近 10 才动」）。回费 2.8 s/费（`player.py:32`、`battle.py:2881`）；`meta.leak` median 2.68 / p90 10.29 / max 193.24。

```bash
/usr/bin/python3 -c "
import json;d=json.load(open('/tmp/il_opening_probe.json'))
print(json.dumps(d['first_card_elixir'],indent=1));print('leak',d['dataset']['leak_per_unit'])"
```

#### 4.3.5 循环 / 过牌迹象（2e，可量化代理）

| 代理指标 | T=20 s | **T=30 s** | T=45 s |
|---|---|---|---|
| 不同卡数/单位（中位） | 3.00 | **4.00** | 5.00 |
| 同卡最大重复（中位 / 全局最大） | — | **1.00 / 2** | — |
| 某张 ≤2 费卡在窗口内被打 ≥2 次的单位% | 0.57% | **6.10%**（n=610） | 25.97% |

T=30 s 补充读数：

| 指标 | 值 | 备注 |
|---|---|---|
| 窗口内法术次数 | 8,199 | 63.62% 单位至少 1 个法术 |
| **≤2 费法术落己方侧** | **4,076** | = 窗口法术 **49.71%**；覆盖 **40.78%** 单位 |
| …其中落**自家塔后**（gy≤6/≥25） | 742 | = 窗口法术 **9.05%** |
| 窗口内圣水花费/单位 | median 13.00（mean 13.32） | 30 s 回费 ≈10.7 ⇒ 动用了开局 5 费 |
| 首牌「过牌签名」≤2 费 ∧ 己方半场深处 | **2,152 / 9,974 = 21.58%** | |
| 首牌「过牌签名」≤2 费 ∧ 自家塔后 | **1,440 / 9,974 = 14.44%** | |
| 首张牌就是法术 | 20.13% | |

**判读**：30 s 窗口内**「同一张低费卡被反复打」几乎不存在**（同卡最大重复 = 2，仅 6.1% 单位出现 ≤2 费重复），要到 T=45 s 才升到 25.97% ⇒ 「人类靠反复打低费卡过牌」在开局窗口内**不成立**。另：「低费法术落己方侧」占窗口法术约一半，但**含正当防守**（Log/Zap/BarbLog 解场），**不是纯净过牌信号**。

```bash
/usr/bin/python3 -c "
import json;d=json.load(open('/tmp/il_opening_probe.json'))
for k in ('game_first_T20','game_first_T30','game_first_T45'):
    w=d['windows'][k];print(k,'distinct',w['distinct_cards_per_unit']['median'],'repeat>=2%',w['units_with_lowcost_repeat_ge2_pct'],'maxrepeat',w['max_same_card_repeat_per_unit']['max'])
w=d['windows']['game_first_T30'];print(w['lowcost_spell_in_own_side'],w['spells_in_window'],w['units_with_any_spell_pct'])
print(d['opening_first_card_T30']['signature_cycle_le2_own_deep'],d['opening_first_card_T30']['signature_cycle_le2_behind_own_towers'])"
```

### 4.4 首次时间分布

#### 4.4.1 「首次对敌方塔造成伤害的时间」= **不可观测**

`/tmp/il_payload_fields.py` 读原始 parquet 的 `payload_json` 取证：事件 `kind` **只有两种**（`play_card`/`activate_ability`）；payload 顶层 7 键、`replay` 3 键（`aggregate_stats/card_counts/duration`）；塔血**只有终局快照** `final_tower_hitpoints`，`aggregate_stats` 只有整局 `[张数,圣水]`。⇒ **缺的字段就是「带时间戳的塔血/伤害事件」**，无替代路径。

```bash
/usr/bin/python3 /tmp/il_payload_fields.py --n 3
```

#### 4.4.2 可算的替代读数（**代理**，与真实首伤不等价）

| 读数 | n | mean | p25 | **median** | p75 | p90 | 全程无此事件% |
|---|---|---|---|---|---|---|---|
| **首次打出 ≥4 费牌** | 9,914 | 23.04 | 12.55 | **18.20** | 26.39 | 37.74 | 0.86% |
| 首次进敌半场（side 相对，代理） | 8,954 | 82.80 | 26.45 | **61.55** | 134.35 | 171.43 | 10.46% |
| 首次过河（代理） | 9,115 | 79.27 | 26.10 | **57.20** | 130.18 | 165.20 | 8.85% |

⇒ **探试期经验下界**：中位玩家 **18.2 s** 才第一次交出 ≥4 费牌；**~57–62 s** 才第一次把牌送进敌方半场；30 s 内 90.0% 出牌落己方侧。

```bash
/usr/bin/python3 -c "import json;d=json.load(open('/tmp/il_opening_probe.json'));print(json.dumps(d['first_times'],indent=1))"
```

### 4.5 结论：6 条可直接当设计参数的读数（R1–R6）

> **R1 窗口时长**：T=15 s 覆盖 **92.94%** 局的局级首张牌 / **84.66%** 局×侧；T=20 s → **97.44% / 94.90%**；T=25 s → 98.58% / 97.60%。建议「前期」硬边界取 **T=20 s**（≈97% 对局已出过牌），要贴人类中位则 T=15 s。
> 复算：`/usr/bin/python3 -c "import json;print(json.load(open('/tmp/il_opening_probe.json'))['first_card_time_coverage_from_battle_start'])"`

> **R2 信息量上界**：到 T=30 s，一侧只出 **4 张牌**（median，mean 4.21，p90 6）、见到 **4 张不同卡**（中位，占 8 张卡组 50%）；T=20 s 为 3/3。⇒ 前期信息分的**分母用 4 而不是 8**。
> 复算：`/usr/bin/python3 -c "import json;d=json.load(open('/tmp/il_opening_probe.json'));print({k:(d['windows'][k]['plays_per_unit']['median'],d['windows'][k]['distinct_cards_per_unit']['median']) for k in ('game_first_T20','game_first_T30','game_first_T45')})"`

> **R3 首牌成本平坦（别把「便宜」当试探）**：1/2/3/4/5/6+ = 14.87/22.62/24.52/24.60/7.27/6.10% ⇒ **≤2 费仅 37.49%，≥4 费 37.97%**。若信息分只对低费过牌加权，最多覆盖 **37.5%** 的首牌；识别试探应按「落己方半场/塔后」而非「首牌便宜」。
> 复算：见 §4.3.2 命令（`opening_first_card_T30`）。

> **R4 位置先验强，可当硬门限**：T=30 s 窗口 **90.0% 出牌在己方侧**，own/bridge/enemy = 64.21/26.32/9.47%；**1 费牌进敌方半场仅 0.03%（2/6,038）**，2 费 14.45%、3 费 16.47%。⇒「落敌方半场」= 承诺/进攻信号；「≤2 费 ∧ 己方半场深处」= 试探信号（占首牌 21.58%）。
> 复算：见 §4.3.3 命令。

> **R5 探试期下界**：首次 ≥4 费牌 median **18.20 s**（IQR 12.55–26.39）；首次进敌半场 **61.55 s**、首次过河 **57.20 s**（10.46%/8.85% 全程不发生）。⇒ 若「前期信息分」要在进攻前把信息换成分数，窗口必须 **≤18 s**；T=30 s 已长过半数玩家的首个大费动作。
> 复算：`/usr/bin/python3 -c "import json;print(json.load(open('/tmp/il_opening_probe.json'))['first_times'])"`

> **补充 R6 过牌在开局窗口内罕见**：同卡最大重复 = 2（10,000 单位无一在 30 s 内打同卡 ≥3 次）；「≤2 费卡重复 ≥2 次」单位占比 0.57%/6.10%/25.97%（T=20/30/45 s）。复算见 §4.3.5。

### 4.6 与既有已知读数的对账（★ 冲突，原样并列）

| 既有读数 | 本脚本复算 | 判定 |
|---|---|---|
| `t<=5s 法术 0/10565` | t≤5 s 法术 = 0；更强：**t≤5 s 的 play_card 总数 = 0，最早一张牌在 5.30 s** | ✅ 结论一致（**10565 分母本机不可复现**，见 U1） |
| 首法术 2 费 **54.4%** | 首法术 2 费 = **58.37%**（n=9,846）；限 t≤30 s 内首法术 = **61.39%** | ⚠️ 差 4–7 pp（U3） |
| 小法术 **52.4%** | 窗口内法术 2 费 = **56.52%**、≤2 费 = **66.29%** | ⚠️ 口径未定（U3） |
| `t<=10s 无 >=4 费法术` | t≤10 s 法术 **706** 个，其中 ≥4 费 **9** 个（Fireball×5、Rocket×2、Poison、Freeze），最早 **5.8 s** | ❌ **不能复现**（U2） |

```bash
/usr/bin/python3 -c "
import json;d=json.load(open('/tmp/il_opening_probe.json'));s=d['sanity_vs_known']
print(s['first_spell_cost_hist']);print(s['first_spell_in_first30s_cost_hist']);print(s['spell_cast_time_census'])
print(s['t_le_10s_cost_ge4_spell_counterexamples'])"
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -c "
import json
n=tot=0;m=1e9
for line in open('/mnt/e/fl_il_data/replays_part000000.jsonl',encoding='utf-8'):
    for e in json.loads(line)['ev']:
        if e[0]==0 and isinstance(e[3],(int,float)):
            t=e[3]/20.0;tot+=1;n+= t<=5.0;m=min(m,t)
print('play_card',tot,'t<=5s',n,'min_t',m)"
```

---

## 5. 冲突并列表（**并列不调和**）

> 以下每条都**不做裁决**；两边（或多边）证据同时在场。凡本报告作者已复核的加 `▸复核`。

### C-1 「对手圣水是否可得」——**两套录像不同源**
- **③**：我方联赛录像**逐帧存对手圣水真值** `elixir1`（`src/clasher_new/rl/replay.py:181-182`，schema 3 起有）；`hidden_labels["opp_elixir"]`（`observation.py:165`）与 `get_prophet_state()["opp_elixir"]`（`env_wrapper.py:262`）另两条特权通道。
- **④**：FirstLight IL 回放 parquet **无逐帧圣水字段**（`/tmp/il_payload_fields.py` 取证；`docs/agents/reward_surveys.md` 记「主缺口 = 圣水不可观测」，直接重放只走通 35–62%）。
- **不调和**：两处「回放/录像」指**不同数据源**（我方 `replay.py` 联赛录像 vs FirstLight IL parquet）。**未定**：是否存在第三份口径把二者混用（本报告未发现）。
- `▸复核`：`replay.py:181-182` 逐字确认为 `"elixir0"/"elixir1": float(p0/p1.elixir)` ✔

### C-2 回费率「2.8 s/费」vs「分阶段 2.8 / 1.4 / 2.8÷3」
- **③**：`battle.py:2880-2881` 是**分阶段**的；belief 估计器硬编码 2.8（`belief.py:288-298`）⇒ **120 s 后估计系统性偏低**。
- **④**：回费 2.8 s/费（`player.py:32`、`battle.py:2881`）。
- **不调和（两说共存）**：`player.py:32` 是默认形参 `base_regen_time: float = 2.8`；`battle.py:2881` 显式传分阶段值 ⇒ **「2.8 s/费」仅在 `t < 120` 成立**。④的首牌读数（t≈9.85 s）落在该区间内，故 ④ 的区间代理不受影响；③指出的偏差只在加时段。
- `▸复核`：`player.py:32` 与 `battle.py:2881` 逐字确认 ✔

### C-3 对手手牌：「同进程可读」vs「观测不可见」vs「人为不可得」
- **③ §3.5**：同一进程内**无任何机制阻止**读对手 `cycle`/`elixir`（普通实例属性，`player.py:8-15`）；「不可观测」是**调用约定/文档纪律**，不是代码强制。
- **③ §3.2**：跟随者观测**不可见**（红线注释 `observation.py:148`；`observation_space` 白名单无 opp_*，`env_wrapper.py:129-135`；网络只吃 5 键，`follower.py:319-340`）。
- **④ / ③ §3.6**：人类回放里对手手牌**真值不可得**（`fl_il_to_bc.py:338-343` 的 `_force_hand` 明确写「这是**假设**，不是重建」；npz `opp_hand` = 我方引擎重建值）。
- **不调和**：三者描述的是**三种不同主体/通道**（引擎内 i 同进程 = 可读 / 跟随者观测 = 不可见 / 人类原始数据 = 不可得），本报告不合并为一句结论。

### C-4 「hidden_labels 不进网络」vs `ProphetPlanner` 消费特权
- **文档口径**：`probe_encoding.py:94`「**不进网络**，仅供信念模块监督」；`observation.py:148`「绝不进跟随者观测」。
- **③ §3.2 实测**：**特权通道并非完全无人消费** —— `ProphetPlanner` 直读 `fs["opp_cycle"]`（`prophet.py:234,396,504`）与 `fs["opp_elixir"]`（`prophet.py:281`），经 `get_prophet_state()`（`env_wrapper.py:254-271`），只经 **plan token（58 维）** 蒸馏进策略（`train_follower.py:119` `plan_prophet_prob=0.3`）。
- **不调和**：「不进网络」对**跟随者策略网络**成立；对 **prophet 特权通道**不成立（前者是软偏置消费者）。二者并列，不裁定哪句为准。

### C-5 `plan_space.py` 陈旧注释「57」vs 实测「58」
- `plan_space.py:12` 注释「`PLAN_DIM = 57`」等为**旧值**；实测 `PLAN_DIM=58`（根因：`PLACEMENT_HINTS` 增第 8 项 `intercept_mid`）。
- `docs/agents/redlines.md:20`（R6）已按 **58** 写；`docs/plan_master.md:137,258` 记录该冲突；`docs/model_vector_inventory_2026-09-20.md:43` 给逐段偏移 `…/54:58`。
- **不调和**：代码注释 vs 实跑维度 vs 既有文档，三处数字并存。
- `▸复核`：grep 显示 `plan_space.py` 的 `PLAN_DIM` 只在 `:3 / :12 / :187 / :188`；`:187` 是定义行。**侦察②给的 `:186` 出处未能复核**（见 §8）⇒ 该出处标「待核」。

### C-6 R13「128 张逐位全等」vs「低费豁免必然改变位图」
- **R13 字面**：掩码/合法格改动须 `scripts/_mask_diff_snapshot.py` 128 张**逐位全等**（`scripts/_mask_diff_snapshot.py:1-20`）。
- **① 分析**：低费豁免**本质就是逐位放宽** ⇒ 与「全等」字面冲突。
- **先例**：C20/C21 已改走「法术全卡位图 sweep（`docs/agents/ledger.md` C21：24 张 × 8 状态 × 2 方 = 384 张，`96/384 变化、全部只收紧`）+ 回归断言 + 预注册判据」。
- **不调和**：规则字面与操作本质冲突；本报告只并列，不裁定 R13 是否应改文。

### C-7 R7「尺度单一常量源」vs 现状 4 处独立字面量
- **R7**：`汇率 / 值函数 / 闸门 edw×卡费 / MCTS 同源同步改`。
- **现状**：`500.0` 至少 3 处独立字面量（`action_mask.py:193`、`spell_module.py:302`、`belief_planner.py:107`）；`SPELL_EV_EDW = 0.5` 硬编码（`action_mask.py:194,329`）而奖励侧真值在 `rl/config.py:61`（`"elixir_diff_weight": 0.5`，预设可覆盖 `:77-79`）。
- **不调和**：红线要求 vs 代码现状；且**不存在**把 `SPELL_EV_EDW` 注入掩码的现存管线（`env_wrapper.py:691` 只注入奖励 `rw`）。
- `▸复核`：`action_mask.py:193-194`、`spell_module.py:302` 逐字确认 ✔

### C-8 C20「F1–F4 已修」vs Lightning `deals_dmg=✗`（F4 遗留仍在）
- **AGENTS.md C20 行**：已修 F1（王塔纳入 EV 判）、F2（效果载体不算目标）、F3（空落点槽禁）、F4（法术谓词同源）；「纯砸王塔 6→0」。
- **① 实测**：`Lightning` 的 `deals_dmg = ✗`（伤害在 `cards_stats_projectile.json`，四路取或不覆盖）；`_spell_tower_damage` 对 Log / Lightning 读 **0.0**；与 `docs/il_spell_kingtower_gap_2026-09-22.md:182` 的 F4 记录**同一条缺口**。
- **不调和**：C20 修的是「只罩王塔」路径；Lightning 是「伤害表覆盖不到」的**另一处**遗留。两说并存，**不判定 C20 已完成/未完成**。
- 代码侧可核对的「已修」痕迹：`action_mask.py:124`（`_is_effect_body` 跳过）、`:283-296`（王塔纳入）、`_spell_covers_non_tower` `:124-125` 注释「★ 2026-09-22」。

### C-9 人类回放与既有读数的三处数字对不上
- **`t<=10s 无 >=4 费法术`**：本数据 t∈[5.8, 9.9] s 有 9 个 ≥4 费法术 ⇒ **不能复现**。
- **首法术 2 费**：既有 54.4% vs 复算 **58.37% / 61.39%**。
- **小法术**：既有 52.4% vs 复算 **56.52%（2 费）/ 66.29%（≤2 费）**；「小法术」无字面定义。
- **不调和**：数字并列；④ 给 U2/U3 作为候选解释（是否只统计首个法术 / 是否排除滚动类与 GoblinBarrel / `t` 起点 / 「小法术」定义），**未裁决**。

### C-10 侦察①内部自相矛盾（原样保留）
- **cost ≤ 2**：标题「**9 张**」vs 紧随清单 **10 张** vs 校验式 `10+9+3+5=27`（用 10）⇒ 以 10 为准，「9」为笔误。
- **cost == 4**：原文先写「5 张」并误列 `SkeletonKing`，随后自我更正为「以实跑为准：3 张」。
- **不调和**：保留原文两处自我更正过程，不做静默修正。

### C-11 「卡组信息分」的语义未定（②/③/④ 交叉）
- **③ §3.4/§3.7**：对手 8 卡卡组是**构造已知**（`env_wrapper.py:103-104`、`run_league.py:126-128`），全仓**无卡组推断模块**；对手手牌/剩余牌是**确定性可算**（`bayes_filter.py:9-14`）。
- **④ R2**：到 T=30 s 一侧只见 4 张不同卡（占 8 张卡组 50%）⇒ 「前期信息分的**分母用 4 而不是 8**」。
- **② §2.3**：把「前期卡组信息分」列为可注入的 plan 维候选。
- **不调和**：若卡组**构造已知**，则「前期卡组信息分」到底度量什么（未知卡组推断 vs 已知卡组的剩余牌/手牌推断）**未定**；④ 的「分母 4」预设的是「观察到的卡」而非「卡组全集」。

---

## 6. 未定项汇总（四份合并，去重）

### 6.1 来自侦察①（法术闸门/卡费）

| # | 未定项 | 为什么不能猜 |
|---|---|---|
| U1 | `X` 取值 | 任务只说「cost≤X」，未给数值；本报告只给事实：X=2 ⇒ 受益卡 = Zap/Snowball |
| U2 | 豁免哪条闸门（8h 空砸 `:509/:575` vs 整块 `:507/:567` 含 9h EV） | 两种落点语义、位图变化面、与 R7 汇率的耦合都不同 |
| U3 | 是否同改 `validate_bundle`（`_position_legal`） | 不同改会产生「掩码放行 / 整包拒收」白掉一帧（O9 同类病理） |
| U4 | 是否顺带修 `Lightning` 的 `deals_dmg=✗`（F4 遗留） | 与豁免无关，但 Lightning 是 6 费、不受低费豁免影响；单独拍板 |
| U5 | R13「128 张逐位全等」与「豁免必然改变位图」如何对账 | 需按 C20/C21 先例走，而不是照 R13 字面 |

### 6.2 来自侦察②（观测注入）

| # | 未定项 |
|---|---|
| U6 | 第 12 项列出的 10 个「手工造 obs」脚本**是否全部**手工构造 obs（仅按 `next_card` 引用定位，未逐个读） |
| U7 | 是否存在第三条「把派生量直接写进 `grid` 通道」的先例（已确认 `grid` 由 `observation.py:87-130` 纯从实体原始字段构建，未见派生打分通道） |
| U8 | 把「前期卡组信息分」放 scalar（须 `--fresh`）还是 plan 尾部（须同修 `follower.py:424-427`）——**② 给出两个落点但未拍板** |

### 6.3 来自侦察③（对手信息）

| # | 未定项 |
|---|---|
| U9 | `_elixir_est` 的**实际误差分布**从未被本仓标定（无 `probe_belief_elixir` 类仪器） |
| U10 | dump 在 `belief.update` **之前** ⇒ `opp_hand` 与 `b_hand` 相差一个 update 步；**该语义是否已被下游正确处理，未逐一验证** |
| U11 | 「前期卡组信息分」是**未知卡组**语义（本仓无此场景）还是**已知卡组剩余牌/手牌推断**语义（本仓可算）——**未定** |

### 6.4 来自侦察④（人类开局试探）

| # | 不确定项 | 卡在哪 |
|---|---|---|
| U12 | `…/10565` **分母不可复现** | 本机只有 1 个分片（5,000 局）；`find /mnt/e -name '*.parquet'` 只命中这 2 个文件。10565 可能来自更大提取或另一数据集 |
| U13 | **`t≤10s 无 ≥4 费法术` 不能复现** | 本数据 t∈[5.8, 9.9] s 有 9 个 ≥4 费法术。差异可能来自：是否只统计**首个**法术 / 是否排除滚动类与 GoblinBarrel / `t` 是否从首张牌而非战斗开始起算 |
| U14 | **54.4% / 52.4% 口径未定** | 给了 3 个候选（58.37% / 61.39% / 56.52%），无一等于 54.4%/52.4%；「小法术」无字面定义（≤2？≤3？排除 Mirror？） |
| U15 | **圣水逐帧不可观测** | 首牌圣水只能给区间 `[6.33, 8.95]`（中位）；区间宽度由 `meta.leak` 的**整局总量**决定，漏费时间分布未知 |
| U16 | **「首次塔伤」不可观测** | 用「首次过河/进敌半场」代理，与真实首伤**不等价**（火箭/火球远距离砸塔不会进敌半场） |
| U17 | **过牌代理不纯净** | 「≤2 费法术落己方侧」含正当防守（Log/Zap/BarbLog 解场）；更可疑的「落自家塔后」742 次（9.05%）也可能只是「留着法术防守」 |
| U18 | **同 tick 多事件顺序不可辨** | 只有 20 Hz tick；同 tick 多牌用 `ev` 数组序做 tie-break（稳定排序），影响「首张牌」判定 |
| U19 | **觉醒/英雄变体未建模** | `-ev1/-hero` 已剥到基础卡，但事件流**不标注哪次是觉醒**（`form_at_play` 全 `unknown`，见 `docs/il_spell_kingtower_gap_2026-09-22.md` §7.4 与 **C21**）⇒ 觉醒轮过牌被当普通出牌 |
| U20 | **1,354 次 `void` 不可映射（0.37%）** | 窗口内 102 次出牌缺费用档；`void` 落敌方半场比例异常（86 次），语义未查 |
| U21 | **`meta.leak` 长尾可疑** | p90 10.29、**max 193.24**（超过该局最大圣水收入）⇒ 疑似逐帧累计；若如此 U15 下界**偏严**（真值更靠上界） |
| U22 | **分片级外部效度** | 样本以 `pathOfLegend` 为主（Ranked 4451/5000）；未做模式/杯段/卡组分层 |

### 6.5 本报告合成时新增的未定

| # | 未定项 |
|---|---|
| U23 | 侦察②给出的 `plan_space.py:186`（「= 57」）出处未能复核（grep 只见 `:3/:12/:187/:188`）⇒ **出处待核**（见 §8） |
| U24 | 侦察① §1.5 的「9 张 / 10 张」内部矛盾，本报告取「10 张」，但**未回读源码复算**该分组计数 |

---

## 7. 改动点总表（跨四主题）

> 状态：🟢 = 现状已确证、可直接动手（但受前置约束）；🟡 = 需先拍板；🔴 = 与红线冲突、需先定义对账方式。

| # | 主题 | 改动点 | `file:line` | 代价 / 前置 | 状态 |
|---|---|---|---|---|---|
| A1 | 法术闸门 | 「低费豁免 8h 空砸」只改空砸闸门 | `action_mask.py:509` + `:575`（`_spell_has_enemy_target` 两调用点） | 位图变化面 = 空砸闸门；9h EV 不变 | 🟡 需先定 `X`（U1）与是否同改 `validate_bundle`（U3） |
| A2 | 法术闸门 | 「整块豁免」（含 9h EV）只改总开关 | `action_mask.py:507` + `:567`（`_spell_requires_placement_target` 两调用点） | 同时豁免两条闸门 | 🟡 语义与 A1 不同（U2） |
| A3 | 法术闸门 | 新增单一取费谓词 + 单一阈值常量 | 建议放 `action_mask.py:150` 之后或 `:107` 附近 | 与 `Card.elixir` 同源 | 🟢 但 X 若硬编码会 +1 处违 R7（见 A4） |
| A4 | 法术闸门 | 把 `SPELL_EV_EDW` / `TOWER_HP_PER_ELIXIR_EARLY` / X 收进**同一常量源** | `action_mask.py:193-194`、`spell_module.py:302`、`belief_planner.py:107`、`rl/config.py:61`、`env_wrapper.py:691` | **需新建注入通道**（现不存在） | 🔴 R7 要求 vs 现状 3+1 处字面量 |
| A5 | 法术闸门 | 豁免的回归测试按**费用枚举**（不写死卡名） | 仿 `selftests/part5.py:1888,1894` | — | 🟢 |
| A6 | 法术闸门 | 掩码对账改走 C20/C21 先例（sweep + 只允许预期方向） | `scripts/_mask_diff_snapshot.py:1-20`；`docs/agents/ledger.md` C21 | 与 R13 字面冲突 | 🔴 需先定义对账方式（U5） |
| A7 | 法术闸门 | 默认关（`X=0`），经 `TrainConfig`/CLI 显式开 | 仿 `config.py:295-297` `value_bypass` 先例 | — | 🟢 R2 |
| B1 | 观测注入 | 加**标量**特征：改 obs 生产者 | `observation.py:138-144`（或 `env_wrapper.py:242-249`） | 追加 scalar 尾部 | 🟢 |
| B2 | 观测注入 | 加标量：改 `observation_space` | `env_wrapper.py:129-141` | — | 🟢 |
| B3 | 观测注入 | 加标量：`scalar_dim 3→4`（`self.scalar_dim` 自动跟随） | `follower.py:248`、`:253` | — | 🟢 |
| B4 | 观测注入 | 加标量：两处 `_scalar_parts` | `follower.py:336-340`（单条）、`:518-522`（批量） | — | 🟢 |
| B5 | 观测注入 | **`--fresh`**（`enc_fc`/`value_enc_fc` 无兼容分支，静默重置） | `follower.py:129-152`、`:172-183`、`:293` | **必须**；标量变化**不触发任何告警** | 🔴 无护栏 |
| B6 | 观测注入 | 手工造 obs 的离线脚本补键 | `train_bc.py:67,77` + §2.3 列出的 10 个脚本 | 否则 `KeyError` | 🟡 未确认清单（U6） |
| B7 | 观测注入 | 跨进程对手池 strict load | `run_league.py:461-465`、`train_solo.py:173` | 形状变 ⇒ `RuntimeError` | 🟢 |
| C1 | 观测注入 | 加 **plan** 维：dataclass 字段 + `to_vector`（`PLAN_DIM` 自动重算） | `plan_space.py:98-112`、`:155-160`、`:187` | — | 🟢 |
| C2 | 观测注入 | 加 plan 维后**必须同修 hold 偏置回读** | `follower.py:424-427`（`PLAN_DIM-4+i`） | 否则 hold 偏置静默错位 | 🔴 隐性坑 |
| C3 | 观测注入 | plan 生产者填新字段 | `belief_planner.py` 15 处 `PlanToken(`、`prophet.py` 15 处 | — | 🟢 |
| C4 | 观测注入 | 方案选择：卡组信息分放 scalar（须 `--fresh`）还是 plan 尾（须 C2） | 见 §2.3 建议落点 | 两路代价不同 | 🟡 U8 / U11 |
| D1 | 对手信息 | 对手圣水估计器回费率与引擎**同源** | `belief.py:288-298` vs `battle.py:2880-2881` | 修 `_elixir_est` 的阶段偏差 | 🟢 但需先标定误差（U9） |
| D2 | 对手信息 | 死亡圣水馈赠计入估计器 | `battle.py:377-388` 未被 `belief.py` 感知 | — | 🟢 |
| D3 | 对手信息 | 复活 `elixir_est=` 外部校正入口（或删除死参数） | `belief.py:300`（全仓无调用方） | — | 🟡 |
| D4 | 对手信息 | 「手牌/剩余牌」打分函数**分段**（前 3 张后验 / 第 4 张起 0-1） | `bayes_filter.py:9-14,151-165,167-176` | 否则量纲混用 | 🟢 |
| D5 | 对手信息 | 跨来源拼接先对齐 `opp_deck` 顺序 | `bayes_filter.py:165`、`fl_il_to_bc.py:598-602` | — | 🟢 |
| D6 | 对手信息 | 明确 `opp_hand` gold label 与 `b_hand` 的 update 步差 | `fl_il_to_bc.py:499-538,567`、`il_opp_prediction.py:17` | — | 🟡 U10 |
| E1 | 开局试探 | 「前期」硬边界取 **T=20 s**（贴人类中位则 15 s） | ④ §4.5 R1 | — | 🟢 设计参数 |
| E2 | 开局试探 | 信息分**分母用 4 而不是 8** | ④ §4.5 R2 | 依赖 U11 语义 | 🟡 |
| E3 | 开局试探 | 识别试探按「落己方半场/塔后」而非「首牌便宜」 | ④ §4.5 R3/R4 | — | 🟢 |
| E4 | 开局试探 | 若信息分要在进攻前兑现 ⇒ 窗口 **≤18 s** | ④ §4.5 R5（首次 ≥4 费牌 median 18.20 s） | — | 🟢 |
| E5 | 开局试探 | 觉醒/英雄变体在事件流中不可分辨（影响「过牌」计数） | ④ U19；`docs/il_spell_kingtower_gap_2026-09-22.md` §7.4；C21 | — | 🟡 需先建模 |

---

## 8. 附：来源、复核注与落盘状态

### 8.1 本报告作者亲自复核过的点（`▸复核`）

| 复核项 | 结果 |
|---|---|
| `rl/replay.py:181-182` 逐帧 `elixir0/elixir1` | ✔ 逐字确认（C-1 的「我方录像有圣水真值」成立） |
| `battle.py:2881` 分阶段回费 + `player.py:32` 默认形参 2.8 | ✔ 两者并存 ⇒ C-2 的「仅在 t<120 成立」成立 |
| `action_mask.py:106-107`（两 frozenset）、`:193-194`（两常量）、`:299-301`（Mirror 绕过的取费）、`:328-329`（连续式终判） | ✔ 全部逐字确认 |
| `spell_module.py:302` `TOWER_HP_PER_ELIXIR = 500.0` | ✔ |
| `plan_space.py` grep `PLAN_DIM` | 仅 `:3 / :12 / :187 / :188`；`:12` 是含「PLAN_DIM = 57」的陈旧注释，`:187` 是定义行 ⇒ **侦察②的 `:186` 出处未能复核**（U23） |

### 8.2 四份来源的可追溯路径

| 侦察 | 完整输出（会话转录） | 关键产物 |
|---|---|---|
| ① 法术掩码 + 卡费 | 子会话 `580d1a9a-c955-47a2-aabc-5ea6fd6803ac` | 实跑本仓 `Card`/`action_mask`/`spell_module` |
| ② 观测注入 | 子会话 `fb15fe1a-6bd5-4942-a85c-6f20476fea96` | — |
| ③ 对手信息 | 子会话 `03cf4494-b16b-4188-9560-13925fdbe5db` | `runs/_fl_il_frames/feat_0000.npz`、`runs/_fl_il_bc/holdout/bc_fl_0001.pkl`（【实测】） |
| ④ 人类开局试探 | 子会话 `37fd93a6-0dda-470f-926a-8b977d0401b5` | `/tmp/il_opening_report.md`（23,293 B）、`/tmp/il_opening_probe.json`（`md5=7b0b01f57c63dab4b4f40ada645ea404`）、`/tmp/il_opening_probe.py`、`/tmp/il_opening_md.py`、`/tmp/il_payload_fields.py` |

> 注：四份输出的**完整逐字原文**已从 DSH 会话转录（`~/.dsh-016/sessions/--mnt-e-clash-royale-simulator-main--/<id>/session.v3.jsonl.zstd`）中恢复，并在 `/tmp/sessdump/REPORT_{mask_cost,obs,belief,human}.md` 留档；本报告 §1–§4 与其一致（仅做结构重排与编号统一），**未改写任何 file:line 出处**。

### 8.3 落盘状态 / 待办

- 本文件为**新建**（未改任何既有文件、未执行任何 git 操作）。
- **尚未做**：① 登记进 `docs/README.md` 文档地图；② 登记进 `docs/agents/session_ledger.md` 文件占用表。
  ⇒ 依 AGENTS.md「多会话协作台账」纪律，**动手前先登记**；本报告为只读合成的产物，**建议由主会话决定是否登记/入库**，以免与「线 2 · 文档形态」会话冲突（见 `docs/agents/session_ledger.md` §6）。
- 本报告**不含**任何未在四份来源中出现的新事实；U23/U24 是本报告合成时的两处**出处/数字存疑**标注。

---

**（完）**