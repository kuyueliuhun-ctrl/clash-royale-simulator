# 非法动作定义 · 分层报告（L1 掩码 / L2 校验 / L3 惩罚统计 / L4 引擎物理）

> **本文件是合并稿**：把 5 份子任务的输出合并为一份按层组织、每条带 `文件:行号` + **原样**代码/理由字符串的报告。
> **不调和冲突**：凡两层/两处口径不一致，一律**并列**（第 6 节），不试图给出"统一口径"。
> **不确定处写「未定」**，不补全。
>
> 取证基线：工作树 HEAD `267455b`（`git log --oneline -1`）。所有代码块均从工作树**逐字**复制。
> 子任务稿中两处表格被**中途截断**（`action_mask.py` 格子层表尾、引擎↔掩码对照表第 7 行后），
> 合并时对截断处**重新读码核对**；仍无法核对的格子标「未定」。

---

## 0. 涉及文件与规模（`wc -l` 实测）

| 文件 | 行数 | 在本报告中的层 |
|---|---|---|
| `src/clasher_new/rl/action_mask.py` | 573 | L1（格子/槽位规则）、L2（`validate_bundle`） |
| `src/clasher_new/rl/env_wrapper.py` | 811 | L1（掩码字典）、L2（提交路径）、L3（扣分/计数） |
| `src/clasher_new/rl/follower.py` | 806 | L1（采样器消费掩码）、L1 守卫（`RuntimeError`） |
| `src/clasher_new/rl/action_bundle.py` | 110 | L1/L2 边界（`K_MAX`、`sub_position`） |
| `src/clasher_new/rl/evaluate.py` | 557 | L3（统计口径） |
| `src/clasher_new/rl/flow_league.py` | — | L3（`invalid_count=0`） |
| `src/clasher_new/rl/train_solo.py` | 1748 | L3（`ghost_rate` 门禁金丝雀） |
| `src/clasher_new/rl/config.py` | 520 | L3（`invalid_penalty` 常量） |
| `scripts/pass_streak_audit.py` | 279 | L3（B 类判别式） |
| `src/clasher_new/battle.py` | 3457 | L4（`deploy_card` 自身拒收） |
| `src/clasher_new/player.py` | 66 | L4（`can_play_card`） |

⚠️ 本仓有**两个** `evaluate.py`：`src/clasher_new/evaluate.py`（64 行）与 `src/clasher_new/rl/evaluate.py`（557 行）。
本报告 L3 引用的是 **`rl/evaluate.py`**（子任务稿中有一处只写"evaluate.py 557 行"，会指向错文件）。

---

## 1. 分层总览

| 层 | 名称 | 实现位置 | 输出形态 | 自身是否产生 reason 字符串 |
|---|---|---|---|---|
| **L1** | **掩码层（硬合法）** | `rl/action_mask.py` 全部规则 + `rl/env_wrapper.py:473-536` + `rl/follower.py:316-330,470-505` | `bool` / `(32,18)` / `(4,)` 位图；采样器 `masked_fill(-1e9)` | **否**（只有 `bool`；reason 由 L2 拼装） |
| **L2** | **引擎校验层（`validate_bundle`）** | `rl/action_mask.py:521-573`；消费 `rl/env_wrapper.py:667-692` | `(ok, reason, resolved_actions)` | **是**（9 条中文 reason 原文） |
| **L3** | **惩罚/统计层** | `rl/env_wrapper.py:262-263,667-692`；`rl/config.py:58`；`rl/evaluate.py:179,217-218,274`；`rl/flow_league.py:323`；`scripts/pass_streak_audit.py:95-107` | 标量奖励 / 计数器 / 分类标签 | 否（只有口径标签：`bundle_ok`、`invalid_count`、`B` 类） |
| **L4** | **引擎物理层（`deploy_card` 自身拒收）** | `battle.py:2702`（包装）→ `battle.py:2996-3077`；`player.py:36-39`；`battle.py:3398` `use_ability` | `bool`（**失败一律静默 `return False`**） | **否**（引擎无理由字符串） |

**层间唯一的事实通道**：L1 掩码 →（采样器）→ L2 `validate_bundle` →（通过则）→ L4 `deploy_card` → L3 记账。
L1 的规则表**声称**复刻 L4（`action_mask.py:2-4` 模块 docstring），但两者按第 6 节**并列存在差异**。

### L2/L4 边界歧义（原样保留，不调和）

用户对 L2 的描述是「**`validate_bundle` 与 `deploy` 的拒收**」，对 L4 的描述是「**`deploy_card` 自身的拒收**」。
本报告中：`validate_bundle` 归 **L2**，`battle._deploy_card_impl` / `player.can_play_card` 归 **L4**。
「L2 里的 deploy 的拒收」究竟指哪一层 —— **未定**（见 §8-U2）。

---

## 2. L1 掩码层（硬合法）

### 2.1 模块自述契约（`action_mask.py:1-10`，原样）

```python
"""动作合法性掩码 + 整包校验（规划文档 3.4 / 3.4.1）。

掩码规则尽量与 battle.deploy_card 的真实校验保持一致；即使掩码误判，
执行时仍以 deploy_card 的返回值作为最终依据。

坐标契约（docs/rl_review_fix_plan.md §5）：
- ``SubAction(x, y)`` 一律是**玩家本地坐标**；
- 掩码层与提交层共用 :func:`rl.action_bundle.sub_position` 做唯一换算，
  消除“掩码世界坐标 vs 提交镜像坐标”的分裂（P0-4）。
"""
```

> 注意末句：「即使掩码误判，执行时仍以 `deploy_card` 的返回值作为最终依据」——**这一句与 L3 的
> 「掩码外的包在 `validate_bundle` 就被整包拒收」并列**（见冲突 C7）。

### 2.2 掩码字典：6 个键（`env_wrapper.py:526-536`，原样）

```python
        return {
            "slots": slots,
            "cells": cells,
            "ability_legal": bool(ability_legal(self.battle, player_id,
                                                elixir_override=elixir, already_used=has_ability)),
            "used_slots": np.array(sorted(used), dtype=np.int32),
            "at_cap": len(partial_bundle.sub_actions) >= K_MAX,
            "any_legal": bool(slots.any()) or bool(ability_legal(self.battle, player_id,
                                                                 elixir_override=elixir,
                                                                 already_used=has_ability)),
        }
```

| 键 | 形状/类型 | 语义 |
|---|---|---|
| `slots` | `(4,)` bool | 槽位在当前模拟圣水下可出且未用 |
| `cells` | `(4,32,18)` bool | **每槽位**本地网格可部署格（索引 `cells[slot][y][x]`，`GRID_H=32` 在前） |
| `ability_legal` | bool | 是否还能追加英雄技能 |
| `used_slots` | `int32` 数组（升序，**0-based**） | partial bundle 已消耗槽位；`follower` **不消费**它（只用于报错信息） |
| `at_cap` | bool | `len(sub_actions) >= K_MAX(=4)` |
| `any_legal` | bool | 本帧是否还有任何可选项 |

### 2.3 槽位规则（`action_mask.py:46-68`，原样）

```python
def _slot_playable(player, card_name: str, elixir: float) -> bool:
    if player.king_tower_hp <= 0:
        return False
    if card_name not in player.cycle[:4]:
        return False
    cost = _card_cost(player, card_name)
    if cost is None:
        return False
    if elixir < cost:
        return False
    return True


def slot_mask(player, elixir_override: float = None, used_slots=None) -> np.ndarray:
    """返回 (K_MAX,) bool 掩码：哪些手牌槽在当前圣水下可出（且未在 bundle 中使用）。"""
    elixir = player.elixir if elixir_override is None else elixir_override
    used = set(used_slots or [])
    mask = np.zeros(K_MAX, dtype=bool)
    for i in range(K_MAX):
        if i in used:
            continue
        mask[i] = _slot_playable(player, player.cycle[i], elixir)
    return mask
```

费用口径 `_card_cost`（`action_mask.py:30-36`，原样）：

```python
def _card_cost(player, card_name: str) -> Optional[float]:
    """实际出牌费用（Mirror 按引擎语义 = 上一张牌费用 + 1；无上一张牌 → None）。"""
    if card_name == "Mirror":
        if not getattr(player, "last_card", None):
            return None
        return Card(player.last_card).elixir + 1
    return Card(card_name).elixir
```

⚠️ **本函数没有 `MergeMaiden` 分支** ⇒ 动态费在掩码层按卡面静态费算（见冲突 **C1**）。

### 2.4 圣水/已用槽模拟 + 8h「不裸下」（首卡）（`env_wrapper.py:498-525`，原样）

```python
        used = set()
        elixir = p.elixir
        has_ability = False
        for sa in partial_bundle.sub_actions:
            if sa.kind == "ability":
                cost = ability_mana(self.battle, player_id)
                elixir -= cost if cost is not None else 0.0
                has_ability = True
            elif sa.slot >= 1 and sa.slot <= K_MAX and (sa.slot - 1) not in used:
                from card_utils import Card
                card = p.cycle[sa.slot - 1]
                cost = Card(card).elixir
                if card == "Mirror" and getattr(p, "last_card", None):
                    cost = Card(p.last_card).elixir + 1
                elixir -= cost
                used.add(sa.slot - 1)
        elixir = max(0.0, elixir)
        slots = slot_mask(p, elixir_override=elixir, used_slots=used)
        for i in used:
            cells[i] = False
        # 8h 不裸下（mask 层）：bundle 还没放牌（首卡决策）时，若禁裸条件成立则整槽禁掉
        # 高承诺单位 → 模型只能 STOP 攒费或先放别的卡凑同刻多卡协同。
        # 防守压境/对手无法出手/己方圣水领先≥3 时会由 solo_commit_blocked 放行。
        if len(partial_bundle.sub_actions) == 0:
            for i in range(K_MAX):
                if slots[i] and solo_commit_blocked(self.battle, player_id, p.cycle[i], elixir):
                    slots[i] = False
                    cells[i] = False
```

⚠️ 注意 `env_wrapper.py:509` **没有** MergeMaiden 动态费分支（同一常量 `Card(card).elixir`），与 `action_mask._card_cost` 同口径。

`solo_commit_blocked`（`action_mask.py:255-276`）+ 常量（`action_mask.py:226-228`，原样）：

```python
SOLO_LEAD_ELIXIR = 3.0
#: 裸下受限的高承诺单位（用户例子 + 坦克裸下送费）；其余便宜卡/后排可单放。
SOLO_COMMIT_CARDS = frozenset({"MiniPekka", "Giant"})
```

```python
    if card_name not in SOLO_COMMIT_CARDS:
        return False
    min_opp = _opp_min_hand_cost(battle, player_id)
    if min_opp is None:
        return False
    opp = battle.players[1 - player_id]
    if opp.elixir < min_opp - 1e-9:
        return False
    if _enemy_in_my_half(battle, player_id):
        return False
    if own_elixir - opp.elixir >= SOLO_LEAD_ELIXIR - 1e-9:
        return False
    return True
```

`_enemy_in_my_half`（`action_mask.py:241-252`）压境判据：P0 `y <= 16.0` / P1 `y >= 16.0`。

### 2.5 格子层总判定 `_position_legal`（`action_mask.py:382-438`，原样）

```python
    if card_info is None:
        card_info = Card(card_name)
    if card_info.type == "spell":
        if _hits_dead_enemy_tower(battle, player_id, pos):
            return False
        if _spell_deals_damage(card_name, card_info):
            radius = _spell_radius_m(card_name, card_info)
            if radius > 0.0 and not _spell_has_enemy_target(battle, player_id, pos, radius):
                return False
            # 9h 前段纯砸塔 EV 闸门：无部队/建筑可溅、账面亏费 → 非法
            if radius > 0.0 and _spell_tower_ev_illegal(battle, player_id, card_name, pos,
                                                        card_info):
                return False
        return True
    if battle.is_position_occupied_by_building(pos, 0.0):
        return False
    # 王塔身后 1 格宽禁建筑（与 battle.deploy_card 同源，塔矩形几何 2026-09-09）
    if card_info.type == "building" and battle.arena.is_behind_king(pos, player_id):
        return False
    if player_id == 0:
        if pos.y <= 1.0 and (pos.x <= 6.0 or pos.x > 12.0):
            return False
        if pos.y >= 21.0:
            return False
        if pos.y >= 15.0:
            if pos.x <= 9:
                if battle.players[1].left_tower_hp > 0:
                    return False
            else:
                if battle.players[1].right_tower_hp > 0:
                    return False
    else:
        if pos.y > 31.0 and (pos.x <= 6.0 or pos.x > 12.0):
            return False
        if pos.y <= 10:
            return False
        if pos.y <= 17.0:
            if pos.x <= 9:
                if battle.players[0].left_tower_hp > 0:
                    return False
            else:
                if battle.players[0].right_tower_hp > 0:
                    return False
    # 8h 坦克后屯兵：后排落点必须待在推进坦克后面并留出攻击距离间距
    if _backline_placement_illegal(battle, player_id, card_name, pos, card_info):
        return False
    return True
```

> **法术提前 `return True`（`action_mask.py:405`）** ⇒ 其后的「建筑占位 / 王塔身后 / 六条区域规则 /
> 8h 坦克后屯兵」对法术**全部不生效**。`legal_cells` 的法术分支（`action_mask.py:456-470`）也不调用 `_position_legal`。

### 2.6 L1 格子层规则清单（逐条 `文件:行号`）

> **reason 原文**：本层所有函数**只返回 `bool`，自身不产生 reason**。用户可见的 reason 一律由
> **L2** `validate_bundle` 拼装：格子类 → `f"{card} 部署位置非法 ({sa.x},{sa.y})"`
> （`action_mask.py:564`）；槽位/费用类 → `f"{card} 不可出（圣水/手牌/塔状态）"`
> （`action_mask.py:562`）。下表不再重复。

| # | 规则 | 条件（常量/取值） | 判定位置 | 门的位置 |
|---|---|---|---|---|
| L1-1 | **法术不得砸已毁敌方塔本体格**（7h） | `is_alive=False` 且（名含 `"Tower"` 或 `id<=6`）且 `distance <= DEAD_TOWER_BODY_R`（**0.9**，`action_mask.py:73`） | `_hits_dead_enemy_tower` `action_mask.py:367-379`（判据 `:377`） | `:395-396`（`_position_legal`）、`:463`（`legal_cells`） |
| L1-2 | **伤害型法术不得空砸**（8h） | `_spell_deals_damage` 且 `radius>0` 且 `_spell_has_enemy_target==False`（命中口径 `distance <= radius + collision_radius + 1e-9`，`:107`） | 辅助 `:79-86`（半径，千分位 /1000）、`:89-94`（是否伤害）、`:97-109`（有无目标） | `:397-400`、`:465` |
| L1-3 | **前段纯砸塔 EV 闸门**（9h） | 五条件**全部满足**才拒：① `battle.time < 120.0`；② 伤害型 + 有半径；③ 罩得到对手存活公主塔；④ 半径内**无**对手非塔目标；⑤ `对塔伤折费 < edw×费用`（残血加权，`mult = tower_value_mult(...)`） | `_spell_tower_ev_illegal` `:158-219`；常量 `TOWER_HP_PER_ELIXIR_EARLY = 500.0`（`:119`）、`SPELL_EV_EDW = 0.5`（`:120`）；判据式 `:219` | `:402-404`、`:467` |
| L1-4 | **非法术不得落在建筑/塔占位上** | `is_position_occupied_by_building(pos, 0.0)`（掩码 **mover_radius=0.0**） | `:406-407` | 同上 |
| L1-5 | **建筑不得放王塔身后 1 格宽** | `type=="building"` 且 `arena.is_behind_king` | `:409-410` | 同上 |
| L1-6 | **P0 底角禁区** | `pos.y <= 1.0` 且（`pos.x <= 6.0` 或 `pos.x > 12.0`） | `:412-413` | 同上 |
| L1-7 | **P0 不得进敌方纵深** | `pos.y >= 21.0` | `:414-415` | 同上 |
| L1-8 | **P0 过河须先破对应侧公主塔** | `pos.y >= 15.0` 且（`pos.x<=9` → 对手 `left_tower_hp>0`；否则 `right_tower_hp>0`） | `:416-422` | 同上 |
| L1-9 | **P1 顶角禁区** | `pos.y > 31.0` 且（`pos.x <= 6.0` 或 `pos.x > 12.0`） | `:424-425` | 同上 |
| L1-10 | **P1 不得进敌方纵深** | `pos.y <= 10` | `:426-427` | 同上 |
| L1-11 | **P1 过河须先破对应侧公主塔** | `pos.y <= 17.0` 且对应侧对手 P0 塔存活 | `:428-434` | 同上 |
| L1-12 | **8h 坦克后屯兵几何门** | 卡在 `BACKLINE_CARDS`；附近 `<=TANK_DEFENSE_R(4.0)` 有敌 → 放行；同路有 `_active_push_tanks` 坦克；须 `gap_along >= need-0.5` | `_backline_placement_illegal` `:338-364`；`_backline_min_gap_m` `:326-335`；常量 `:282-289`（`TANK_CARDS`/`BACKLINE_CARDS`/`TANK_LANE_HALF_W=5.0`/`BACKLINE_SPEED_SCALE=2.0`/`TANK_DEFENSE_R=4.0`/`_TANK_ACTIVE_BAND`） | `:436-437`、`:471-475` |

`legal_cells` 全网格（`action_mask.py:441-476`）——`(32,18)` bool，法术分支 `:456-470`，非法术分支 `:471-474`：

```python
    for y in range(GRID_H):
        for x in range(GRID_W):
            if not _position_legal(battle, player_id, eff, sub_position(player_id, x, y),
                                   eff_info):
                cells[y, x] = False
    return cells
```

坐标换算唯一入口 `action_bundle.py:31-39`（原样）：P0 `(x+0.5, y+0.5)`；P1 `(17.5-x, 31.5-y)`。

### 2.7 技能层（`action_mask.py:479-518`，原样）

```python
def _ready_ability_cost(battle, player_id: int) -> Optional[float]:
    """返回场上首个就绪英雄技能的耗蓝；无就绪英雄返回 None。

    与引擎 battle.use_ability 的就绪判定保持一致（必要非充分：引擎还会先调
    holder.use_ability()，掩码层只做静态预判）。
    """
    p = battle.players[player_id]
    if p.king_tower_hp <= 0:
        return None
    for e in battle.entities.values():
        if not e.is_alive or e.player != player_id:
            continue
        ability = getattr(e.data, "ability", None)
        if not ability or getattr(e, "ability_cd", 0) > 0:
            continue
        if not hasattr(getattr(e, "entity_holder", None), "use_ability"):
            continue
        return float(ability.get("manaCost", 0))
    return None
```

```python
def ability_legal(battle, player_id: int, elixir_override: float = None,
                  already_used: bool = False) -> bool:
    if already_used:
        return False
    cost = _ready_ability_cost(battle, player_id)
    if cost is None:
        return False
    elixir = battle.players[player_id].elixir if elixir_override is None else elixir_override
    return elixir >= cost


def ability_mana(battle, player_id: int) -> Optional[float]:
    """返回就绪英雄技能的耗蓝；无就绪英雄返回 None（不再用 0 作哨兵，P2）。"""
    return _ready_ability_cost(battle, player_id)
```

> 掩码自述「**必要非充分**」—— 与 L4 `use_ability`（`battle.py:3398`）并列，见冲突 **C-column**。

### 2.8 采样器如何把 L1 变「硬」（`follower.py`）

决策空间常量（`follower.py:36-39`）：`ABILITY_IDX = K_MAX`、`STOP_IDX = K_MAX + 1`、`NUM_SLOT_OPTIONS = K_MAX + 2`。

掩码张量（`follower.py:322-330`，原样）：

```python
        sm = torch.ones(NUM_SLOT_OPTIONS, device=self.device)
        sm[:K_MAX] = torch.as_tensor(mask["slots"], dtype=torch.float32, device=self.device)
        sm[ABILITY_IDX] = 1.0 if mask.get("ability_legal") else 0.0
        sm[STOP_IDX] = 1.0
        if mask.get("at_cap"):
            sm[:K_MAX] = 0.0
            sm[ABILITY_IDX] = 0.0
            sm[STOP_IDX] = 1.0
        return sm
```

施加（单条路径 `follower.py:469-477`，原样）：

```python
                slot_mask = self._slot_mask_tensor(mask)
                slot_logits = self.slot_head(h) + slot_bias
                slot_logits = slot_logits.masked_fill(slot_mask == 0, -1e9)
                slot_dist = torch.distributions.Categorical(
                    logits=F.log_softmax(slot_logits, dim=-1))
                if deterministic:
                    option = int(torch.argmax(slot_logits, dim=-1).item())
                else:
                    option = int(slot_dist.sample().item())
```

落点（单条路径 `follower.py:496-505`，原样）：

```python
                cells = torch.as_tensor(mask["cells"][option], dtype=torch.float32, device=self.device)
                cell_logits = self.cell_head(h).view(1, GRID_H, GRID_W) + cell_bias
                cell_logits = cell_logits.masked_fill(cells == 0, -1e9)
                flat = cell_logits.reshape(1, -1)
                cell_dist = torch.distributions.Categorical(logits=F.log_softmax(flat, dim=-1))
                if deterministic:
                    cell = int(torch.argmax(flat, dim=-1).item())
                else:
                    cell = int(cell_dist.sample().item())
                logprob += float(cell_dist.log_prob(torch.tensor([cell], device=self.device)).item())
                x, y = int(cell % GRID_W), int(cell // GRID_W)
                bundle.add(option + 1, x, y)
```

同一序列在另外三条路径重复：`act_parallel` `follower.py:579-583`（`masked_fill`）/`588-594`（verify）、
`evaluate_batch` `:682-686`、`evaluate` `:771-774` 与 `:800-803`（STOP 步用 `masks[len(sub_actions)]`，
缺则**全合法回退** `:796-799`）。

### 2.9 L1 的软点（全部原样列出，不调和）

| # | 事实 | 位置 | 后果 |
|---|---|---|---|
| S1 | 掩码缺失时**返回全合法掩码**（`slots` 全 True / `cells` 全 True / `ability_legal=False`） | `follower.py:378-385` | 「硬合法」在批量路径上**不硬**（见冲突 C7） |
| S2 | 掩码用 `-1e9` 而非 `-inf` | `follower.py:471/498` 等 | 被掩项采样概率为 0；但某行**全 False** 时 `log_softmax` 退化为**均匀分布** |
| S3 | 落点采样**没有**采样后合法性校验 | `follower.py:496-505` 之后无任何 verify | 全 False 行会在全网格采样；slot 侧有不变式·2，cell 侧没有 |
| S4 | `STOP` 恒合法（`sm[STOP_IDX] = 1.0`） | `follower.py:325,329` | slot head **构造上不可能全掩**；无「全 0 检测」代码 |
| S5 | `stop_logit_bias=-1.0` 只在 `__init__` 加到 bias 一次 | `follower.py:156,252-255` | 只改先验，不改合法性 |
| S6 | MCTS 显式跳过空格行 | `mcts.py:166-167`（`if not cells.any(): continue`） | 与 S3 并列：**MCTS 有防御、Follower 没有** |
| S7 | 「cell 行是否真会全 False」 | — | **未定**（本次未做穷举取证） |

### 2.10 L1 的第三条路：`RuntimeError`（既非挡住也非吃罚）

采样器在 `act()` 内加了两条**零成本不变式**（`follower.py:453-468` 与 `478-486`），
批路径同口径（`:569-576` 与 `:588-594`）。触发时抛 `RuntimeError`，**不进入 L3 扣分**。

不变式·2 原文（`follower.py:483-486`）：

```python
                    raise RuntimeError(
                        f"[mask 不变式] 采样器选中了被掩掉的槽位 slot={option + 1}"
                        f"（mask['slots']={np.asarray(mask['slots']).astype(int).tolist()}，"
                        f"已用={np.asarray(mask.get('used_slots', [])).tolist()}）"
                        "——掩码与采样不一致，按引擎级问题处理")
```

---

## 3. L2 引擎校验层（`validate_bundle`）

### 3.1 `validate_bundle` 全部拒收理由（`action_mask.py:521-573`，原样）

| 行号 | 原样 reason 字符串 | 触发条件 |
|---|---|---|
| `:541` | `"bundle 内重复技能"` | `has_ability` 已 True 又遇 `kind=="ability"` |
| `:544` | `"无就绪英雄技能"` | `ability_mana(...) is None` |
| `:546` | `"技能圣水不足"` | `elixir < cost` |
| `:554` | `f"slot {sa.slot} 越界"` | `sa.slot < 1 or sa.slot > K_MAX` |
| `:556` | `f"slot {sa.slot} 重复"` | `sa.slot in used`（**1-based**，`:566` `used.add(sa.slot)`） |
| `:558` | `f"坐标越界 ({sa.x},{sa.y})"` | `sa.x < 0 or sa.x >= GRID_W or sa.y < 0 or sa.y >= GRID_H` |
| `:562` | `f"{card} 不可出（圣水/手牌/塔状态）"` | `not _slot_playable(p, card, elixir)` |
| `:564` | `f"{card} 部署位置非法 ({sa.x},{sa.y})"` | `not _position_legal(battle, player_id, eff, sa.to_position(player_id))` |
| `:572` | `"不裸下: 无圣水优势时禁止单独放高承诺卡"` | `len(deploys)==1 and solo_commit_blocked(...)` |
| `:573` | `"ok"` | 全部通过（**成功字符串**） |

关键片断（`action_mask.py:551-573`，原样）：

```python
        if sa.slot == 0:
            continue
        if sa.slot < 1 or sa.slot > K_MAX:
            return False, f"slot {sa.slot} 越界", resolved
        if sa.slot in used:
            return False, f"slot {sa.slot} 重复", resolved
        if sa.x < 0 or sa.x >= GRID_W or sa.y < 0 or sa.y >= GRID_H:
            return False, f"坐标越界 ({sa.x},{sa.y})", resolved
        card = p.cycle[sa.slot - 1]
        eff = _effective_card(p, card)
        if not _slot_playable(p, card, elixir):
            return False, f"{card} 不可出（圣水/手牌/塔状态）", resolved
        if not _position_legal(battle, player_id, eff, sa.to_position(player_id)):
            return False, f"{card} 部署位置非法 ({sa.x},{sa.y})", resolved
        elixir -= _card_cost(p, card) or 0.0
        used.add(sa.slot)
        resolved.append((card, sa))
    # 8h 不裸下兜底（mask 只管“空 bundle 首卡”，BC/旧回放里的单卡裸下在这里拒绝）：
    # 整包只放 1 张高承诺单位且无圣水优势 → 拒绝
    deploys = [c for c, _ in resolved if c != "__ability__"]
    if len(deploys) == 1 and solo_commit_blocked(battle, player_id, deploys[0], p.elixir):
        return False, "不裸下: 无圣水优势时禁止单独放高承诺卡", resolved
    return True, "ok", resolved
```

### 3.2 L2 的结构事实

| 事实 | 证据 |
|---|---|
| **不校验长度**（`K_MAX`） | `validate_bundle` 内无 `len(...)` 检查；唯一长度检查在 `action_bundle.py:79-81` 的 `__post_init__`：`if len(self.sub_actions) > K_MAX: raise ValueError(...)`（**仅构造期**） |
| `sa.slot == 0` 被**静默跳过** | `:551-552` |
| `used` 集合是 **1-based** | `:566` `used.add(sa.slot)`（与 L1 `env_wrapper.py:506,513` 的 **0-based** 集合并列存在，各自自洽） |
| `elixir` 按 `_card_cost` 扣（含 Mirror+1） | `:565` |
| 位置校验用 `sa.to_position(player_id)`（世界坐标） | `:563`；换算同源于 `action_bundle.py:31-39` |

### 3.3 L2 的消费方（`env_wrapper.py:667-692`，原样）

```python
        # 1) 整包校验（不修改状态）→ 返回按决策时刻手牌解析好的 (card, sub_action)
        ok, reason, resolved = validate_bundle(self.battle, 0, action_bundle)
        invalid_count = 0
        if not ok:
            invalid_count = 1
        else:
            # 2) 整包提交：同一 tick 内依次执行（技能用 use_ability、出牌用 deploy_card，
            #    均按决策时刻解析，避免循环前移错位），期间不推进 battle.step
            for card, sa in resolved:
                if card == "__ability__":
                    if not self.battle.use_ability(0):
                        invalid_count += 1
                    continue
                elixir_before_card = p0.elixir
                succeed = self.battle.deploy_card(0, card, sa.to_position(player_id=0))
                if succeed:
                    # 记账：本卡实际花费（elixir 差自动覆盖 Mirror+1/MergeMaiden 形态费）
                    self._deploy_ledger(0, elixir_before_card, card)
                    self._seen_max_id = max(self._seen_max_id,
                                            self.battle.next_entity_id - 1)
                else:
                    # 引擎级拒绝（掩码误判、同 tick 建筑占位快照滞后等），计为非法
                    # （整包已部分提交，无法回滚，记录之）。P1-20：报警以便发现掩码缺口。
                    invalid_count += 1
                    warnings.warn(
                        f"P1-20: validate 通过但引擎拒绝 {card}@{sa.x},{sa.y} —— 掩码缺口",
                        RuntimeWarning)
```

`info` 出口（`env_wrapper.py:781-785`，原样）：

```python
        info = {
            "bundle_ok": ok,
            "bundle_reason": reason if not ok else "ok",
            "invalid_count": invalid_count,
```

其它 L2 消费方：`mcts.py:185-189` 在枚举候选时调 `validate_bundle` 且 `if ok:` 才入候选。

---

## 4. L3 惩罚/统计层

### 4.1 `invalid_penalty` 常量（两处默认 + 预设）

```python
# env_wrapper.py:50
    "invalid_penalty": 0.05,
```

```python
# config.py:58
    "invalid_penalty": 0.05,    # 每次非法动作惩罚
```

预设覆盖：`config.py:390/402/408/414/441/462` = `0.05`；
**`config.py:396` = `0.1`**，属预设 `"defensive"`（`config.py:392-398`，`description="防守反击：非法动作惩罚更重，费差减码（1圣水≈300血）"`）。

### 4.2 扣分公式（`env_wrapper.py:262-263`，原样）

```python
    if invalid_count:
        reward -= rw["invalid_penalty"] * invalid_count
```

> `win_bonus = 10.0`（`config.py:55` / `env_wrapper.py:48`）⇒ 单次 = 0.5% = **1/200**。

### 4.3 `invalid_count` 的两个来源（原样）

| 来源 | 代码 | 计数值 |
|---|---|---|
| **L2 整包拒绝** | `env_wrapper.py:669-670` `if not ok: invalid_count = 1` | **恒 1**（整包只算一次，且整包**什么都不执行**） |
| **L4 引擎拒绝 / 技能失败** | `env_wrapper.py:676-677`（技能）、`:689`（deploy）`invalid_count += 1` | **逐卡累加**（包内每张被引擎拒的卡各 +1；整包**已部分提交，无法回滚**） |

### 4.4 `flow_league.py` 的硬编码 0（原样）

```python
# flow_league.py:323
            invalid_count=0,
```

自述理由（`flow_league.py:14-16`，原样）：

```python
- player-1 侧轨迹由 FollowerOpponent.take_last_step() 收集；player-1 的 reward 用
  compute_reward 交换 blue/red 视角镜像计算（invalid_count 视为 0，FollowerPolicy
  从掩码采样一般合法）；
```

### 4.5 `evaluate.py` 的 `bundle_illegal`（原样）

```python
# rl/evaluate.py:179
             "rew": 0.0, "bundle_sizes": [], "bundle_illegal": 0, "bundle_total": 0,
```

```python
# rl/evaluate.py:217-218
            if not info.get("bundle_ok"):
                stats["bundle_illegal"] += 1
```

```python
# rl/evaluate.py:274
    print(f"Bundle 合法率: {1 - stats['bundle_illegal']/max(1, stats['bundle_total']):.3f} "
```

⚠️ 统计的是 **`bundle_ok`（= L2 `validate_bundle` 结果）**；**L4 引擎拒绝不计入**
（此时 `bundle_ok` 仍为 `True`，只有 `invalid_count` 涨、并发 `RuntimeWarning`）。
与 `train_solo.py` 的 `P1-20` 计数并列，见冲突 **C5**。

### 4.6 B 类（「整包被拒」）的判别式（`scripts/pass_streak_audit.py`）

自述口径（`:18-25`，原样）：

```python
    `elixir0[i] >= pre[i] - 1e-6`  ⟺  **这一帧一个子动作都没真的执行**（没花任何费）

| 类别 | 定义 | 含义 |
|---|---|---|
| **A 主动不出牌** | `bundle == []`（且圣水没降） | 合法地选择"本帧不出" |
| **B 整包被拒** | `bundle != []` 且**圣水没降** | **非法动作、白掉一帧**（此前所有行为统计都看不见） |
| **C 实际出牌** | `bundle != []` 且圣水下降 | 真的部署了 |
```

判别式代码（`:95-107`，原样）：

```python
    for i, f in enumerate(frames):
        b = f.get("bundle") or []
        no_spend = not (post[i] < pre[i] - 1e-6)
        if len(b) == 0:
            cls.append("A")
        else:
            cls.append("B" if no_spend else "C")
        slots = tuple(int(s[1]) for s in b if s and s[0] == "deploy")
        cost = float(np.nansum([_cost(c) for c in (f.get("cards") or [])])) \
            if f.get("cards") else 0.0
        dup = len(set(slots)) != len(slots)
        over = cost > pre[i] + 1e-6
        sig.append({"slots": slots, "cost": round(cost, 2), "dup": dup, "over": over})
```

B 的子分类（`:207-214`，原样）：

```python
                for i, s in enumerate(rec["sig"]):
                    if rec["cls"][i] == "B":
                        if s["dup"]:
                            st["b_dup"] += 1
                        elif s["over"]:
                            st["b_over"] += 1
                        else:
                            st["b_unknown"] += 1
```

自述非法性自证（`:27-30`，原样）：

```python
**B 的非法性由脚本自己独立复算**（不靠引擎的 `invalid_count`）：
① **重复槽位**（`len(set(slots)) != len(slots)`）⇒ `validate_bundle` 必拒；
② **总费 > 决策圣水** ⇒ 必拒；③ 以上都不满足的 B 记作 `B_unknown`（掩码缺口 / 引擎级拒绝类，P1-20）。
```

阈值常量 `ALL_AFFORD = 6.0`（`:72`）；A 类自检 `anomaly`（`:144`）= 「A 类却花掉圣水」帧数。

### 4.7 `ghost_rate` 门禁金丝雀（`train_solo.py`，部分原样）

```python
# train_solo.py:566
      ghost_rate            幽灵动作率（落点 y>=20 的 deploy 占比；P0-2 门禁金丝雀）
```

```python
# train_solo.py:632
                ghost_deploys += sum(1 for b in my_deploys if float(b[3]) >= 20)
```

```python
# config.py:286-289
    gates: dict = field(default_factory=lambda: {
        "engagement_rate": {"rel": ">=", "frac": 0.5},
        "ghost_rate": {"rel": "<=", "frac": 2.0},
    })
```

`train_solo.py` 中该门禁为**只报警不阻断**（子任务稿 §0-J；本次未逐行复核报警分支 → 该点为**引用**，
非本报告独立取证）。

---

## 5. L4 引擎物理层（`deploy_card` 自身拒收）

### 5.1 入口与返回契约

```python
# battle.py:2702
    def deploy_card(self, player_id, card_name, position, _from_mirror=False):
```

包装体 docstring（`battle.py:2703-2706`，原样）：

```python
        """S2 包装（2026-09-18）：给 `_deploy_card_impl` 套一层「出牌事件」上下文。

        包装体**不改变任何行为**：只分配 `cast_id`、设置/恢复两个记录用上下文、
        成功时写 `cast_log`。镜像会递归进 `deploy_card` ⇒ 每张镜像卡算**独立**一次出牌事件。
        """
```

真实逻辑在 `BattleState._deploy_card_impl`（`battle.py:2996`）。**失败一律 `return False`：静默、无理由字符串、无日志、不抛异常。**

| 项 | 值 |
|---|---|
| 返回值 | `bool`（**引擎侧不产生任何 reason 字符串**） |
| 抛异常的唯一情形 | 未知卡名 → `Card(card_name)` → `card_data[card_name]` `KeyError`（`card_utils.py:224`） |
| 底层原语 | `PlayerState.can_play_card`（`player.py:36-39`）、`PlayerState.play_card`（`player.py:41-48`） |
| 调用方写的唯一「理由」 | `env_wrapper.py:690-692` 的 `RuntimeWarning`（**RL 层**，非引擎） |

### 5.2 逐条拒收条件

**L4-1 Mirror 专属（`battle.py:2998-3009`，原样）**

```python
        if card_name == 'Mirror':
            p = self.players[player_id]
            if not p.can_play_card('Mirror'): return False
            last = p.last_card
            if not last: return False
            # —— M7：Spirit Empress 双费用形态——镜像复制上一形态并按其「实际费用+1」
            # （官方：镜像沿用上一形态, 费用不再随圣水切档 [Fandom _fp_SpiritEmpress 策略节]）——
            if last == 'MergeMaiden':
                _mirror_base = getattr(p, 'last_card_cost', None) or 3
            else:
                _mirror_base = Card(last).elixir
            if p.elixir < _mirror_base + 1: return False
```

**L4-2 MergeMaiden 动态费（`battle.py:3021-3038`，原样；`_merge_cost = None` 在 `:3021`，分支入口 `:3023`）**

```python
        _merge_cost = None
        _hand_card = None
        if card_name == 'MergeMaiden':
            p = self.players[player_id]
            if _from_mirror:
                _form = getattr(p, 'last_card_form', None) or 'MergeMaiden_Mounted'
                _merge_cost = 6 if _form == 'MergeMaiden_Mounted' else 3
            else:
                _mounted = p.elixir >= 6
                _merge_cost = 6 if _mounted else 3
                # 手牌校验用实际费用（非卡面 6 费）：在手牌前 4 位 + 圣水足够 + 王塔存活
                if not (card_name in p.cycle[:4] and p.elixir >= _merge_cost
                        and p.king_tower_hp > 0):
                    return False
                _form = 'MergeMaiden_Mounted' if _mounted else 'MergeMaiden_Normal'
            p.last_card_cost = _merge_cost      # 镜像复制形态/费用用
            p.last_card_form = _form
            _hand_card = card_name              # 手牌循环按 MergeMaiden 推进
            card_name = _form
```

⚠️ 该分支**绕过** `can_play_card`（走 `elif`），故 MergeMaiden 的圣水门槛是 **3 或 6**，**不是卡面 6 费**。

**L4-3 圣水/手牌/王塔三合一（`battle.py:3040-3041` + `player.py:36-39`，原样）**

```python
        elif not _from_mirror and not self.players[player_id].can_play_card(card_name):
            return False
```

```python
    def can_play_card(self, card_name):
        return (card_name in self.cycle[:4] and
                self.elixir >= Card(card_name).elixir and
                self.king_tower_hp > 0)
```

**L4-4 BarbLog 专属部署区（`battle.py:3042-3046`，原样）**

```python
        # —— 勘误批1：BarbLog 仅可在与部队相同的部署区域施放（用户口径 2026-09-04,
        # 与其他法术「全场任意」不同）——
        if card_name == 'BarbLog' and not _from_mirror:
            if not self.arena.can_deploy_at(position, player_id, battle_state=self, is_spell=False):
                return False
```

**L4-5 建筑/塔占位格（仅非法术）（`battle.py:3055-3057`，原样）**

```python
        if card_info.type != 'spell':
            # Check the deployment area is legit
            if self.is_position_occupied_by_building(position, 0): return False
```

**L4-6 王塔身后 1 格禁建筑（`battle.py:3058-3060`，原样）**

```python
            # 王塔身后 1 格宽禁建筑（部队不限制；arena.behind_king 与掩码层同源）
            if card_info.type == 'building' and self.arena.is_behind_king(position, player_id):
                return False
```

**L4-7 部署区六条（`battle.py:3061-3077`，原样）**

```python
            # —— 勘误批8：Miner 官方可部署全场（含敌半场）——
            if card_name != 'Miner' and player_id == 0:
                if position.y <= 1.0 and (position.x <= 6.0 or position.x > 12.0): return False
                if position.y >= 21.0: return False
                elif position.y >= 15.0:
                    if position.x <= 9:
                        if self.players[1].left_tower_hp > 0: return False
                    else:
                        if self.players[1].right_tower_hp > 0: return False
            elif card_name != 'Miner' and player_id == 1:
                if position.y > 31.0 and (position.x <= 6.0 or position.x > 12.0): return False
                if position.y <= 10: return False
                elif position.y <= 17.0:
                    if position.x <= 9:
                        if self.players[0].left_tower_hp > 0: return False
                    else:
                        if self.players[0].right_tower_hp > 0: return False
```

⚠️ **Miner 全区豁免**（`card_name != 'Miner'`）；`rl/action_mask.py` 中 **Miner 无任何豁免**（照走 `:414-422`）——见冲突 **C6**。

### 5.3 L4 明确**不**拒收的项（原样证据 + 实测来源标注）

| 项 | 事实 | 证据 |
|---|---|---|
| **法术位置** | `card_info.type == 'spell'` 分支（`battle.py:3079` 起）**不经过** 5.2 的 L4-5/6/7；**BarbLog 是唯一例外** | `battle.py:3055` 的 `if card_info.type != 'spell':` 守卫；L4-4 |
| **坐标越界** | 引擎**无拒绝条件**；`is_valid_position`/`is_blocked_tile` 全仓只被 `arena.is_walkable:111` 与 `arena.can_deploy_at:176` 调用（后者只服务 BarbLog） | 子任务稿实测：`Knight @ (50.0, 5.0) -> engine=True`；`Arrows @ (-9.0, -9.0) -> engine=True`（**本报告未复跑**，标「引用」） |
| **越界夹取** | 越界由 `ensure_walkability` **静默夹取**（非拒绝） | `battle.py:2681-2692`（子任务稿逐字引用，本报告未读该段自证 → 「引用」） |
| **场上数量上限** | **不存在** | `grep -rn "MAX_UNITS\|unit_cap\|deploy_limit\|spawn_limit\|too many"` → **0 命中**（子任务稿）；实测 60 次同点 Knight 全 True（「引用」） |
| **单位重叠** | 不是拒绝条件；由 `resolve_collisions`（`battle.py:3274-3303`）事后推挤 | 「引用」 |
| **演化形态/英雄解锁** | 形态不满足只按普通形态出兵，**不 return False** | `battle.py:3050-3053`；`rl/action_mask.py` 中 `grep evo` → 0 命中 |

### 5.4 技能（`battle.py:3398` `use_ability`）

`battle.py:3398-3438` 要点（子任务稿摘录，**本报告未逐行自证**）：

```python
win = self.hero_windows.get(player_id)
if win and not win.get('used'):
    if self.time > win['until']: win['used'] = True      # 窗口过期
    else:
        cost = ab.get('manaCost', 0)
        if self.players[player_id].elixir < cost: return False
```

> 与 L1 `_ready_ability_cost` 自述「必要非充分」（`action_mask.py:481-482`）并列；`return False` 同样是静默的。

---

## 6. 冲突清单（**并列，不调和**）

### C1 · MergeMaiden 动态费：掩码 6 vs 引擎 3/6

- **L1（掩码）**：`_card_cost`（`action_mask.py:30-36`）**无 MergeMaiden 分支** → 返回 `Card("MergeMaiden").elixir`（卡面 6）；`env_wrapper.py:509` 同口径。
- **L4（引擎）**：`battle.py:3029-3030` `_mounted = p.elixir >= 6` / `_merge_cost = 6 if _mounted else 3`；且 `battle.py:3032-3034` 用 `_merge_cost` 做手牌校验（`if not (card_name in p.cycle[:4] and p.elixir >= _merge_cost and p.king_tower_hp > 0): return False`）。
- **并列结论**：`3 ≤ 圣水 < 6` 时 **L1 判「买不起」/ L4 可出**（L1 过严）。
  子任务稿实测：`elixir=3/4/5.9 → mask_slot_playable=False, engine_deploy=True`（**本报告未复跑 → 引用**）。

### C2 · BarbLog 部署区：掩码无 vs 引擎有

- **L1**：`_position_legal` 对 `type=="spell"` **提前 `return True`**（`action_mask.py:405`），**没有 BarbLog 分支**。
- **L4**：`battle.py:3044-3046` BarbLog 走 `arena.can_deploy_at(..., is_spell=False)`，**可拒**。
- **并列结论**：L1 **缺口**（掩码放行→L2 放行→L4 拒 → 计入 L3 `invalid_count` + `RuntimeWarning`）。
  子任务稿对账：**352 格/方**缺口（**本报告未复跑 → 引用**）。

### C3 · 8h「不裸下」的作用域差：掩码 vs `validate_bundle`

| 处 | 代码 | 作用域 |
|---|---|---|
| L1 | `env_wrapper.py:521` `if len(partial_bundle.sub_actions) == 0:` | **仅空 bundle 的首卡决策** |
| L2 | `action_mask.py:570-572` `if len(deploys) == 1 and solo_commit_blocked(...)` | **仅整包恰好 1 张 deploy** 的兜底 |
| 掩码注释自述 | `env_wrapper.py:518`「bundle 还没放牌（首卡决策）时」 | — |
| L2 注释自述 | `action_mask.py:568`「mask 只管"空 bundle 首卡"，BC/旧回放里的单卡裸下在这里拒绝」 | — |

- **并列结论**：两处被**注释声称互补**，但作用域并不相同 —— 一个 2 卡包里的首卡裸下，
  **L1 不拦**（`len != 0` 时不启用）、**L2 也不拦**（`len(deploys) == 2`）。该空隙**未被任何一层覆盖**
  （子任务稿实测「照掩码行事仍被 `validate_bundle` 拒绝」为另一条路径；此处**未定**是否可达，见 §8-U4）。

### C4 · 法术位置规则：L1 有三道闸门 vs L4 完全不校验

- **L1**：7h（已毁塔本体）、8h（空砸）、9h（纯砸塔 EV）三道闸门（`action_mask.py:395-404,456-470`）+ L2 同源（`:563`）。
- **L4**：法术**无任何位置/区域校验**（`battle.py:3055` 的 `if card_info.type != 'spell':` 守卫；BarbLog 例外）。
- **并列结论**：L1/L2 的这三条是**纯 RL 层额外约束**（引擎不会拒），不是「复刻引擎」。
  `action_mask.py:2-3` 的「掩码规则尽量与 `battle.deploy_card` 的真实校验保持一致」在此**不成立**。

### C5 · `evaluate.bundle_illegal` vs 引擎拒绝

- **L3 统计**：`rl/evaluate.py:217-218` 只在 `not info.get("bundle_ok")` 时 +1。
- **L2/L4 实况**：引擎拒绝时 `ok` 仍为 `True`（`env_wrapper.py:667-692`）⇒ `bundle_ok=True`、`invalid_count>0`。
- **并列结论**：**「Bundle 合法率」（`rl/evaluate.py:274`）看不见引擎级拒绝**；
  同一份非法动作在 `evaluate` 里**不计入**、在 `compute_reward` 里**计 −0.05/次**。

### C6 · Miner 全区豁免：L4 有 vs L1 无

- **L4**：`battle.py:3062` `if card_name != 'Miner' and player_id == 0:`（P1 同 `:3070`）⇒ Miner 全区。
- **L1**：`_position_legal`（`action_mask.py:411-434`）**无 Miner 豁免** ⇒ Miner 照走区域规则。
- **并列结论**：L1 **过严**（掩码禁掉的 Miner 落点引擎其实可出）。

### C7 · 「硬合法」vs 全合法回退 vs 整包拒收

- **L1 契约**：`action_mask.py:3`「执行时仍以 `deploy_card` 的返回值作为最终依据」。
- **L1 回退**：`follower.py:378-385` 掩码缺失时返回 **`cells` 全 True / `slots` 全 True**（含 `ability_legal=False`）。
- **L2**：非法包被**整包拒收**（`validate_bundle` → `invalid_count = 1`，`env_wrapper.py:669-670`）。
- **L4**：`deploy_card` 静默 `False`。
- **并列结论**：三条口径并存 —— ①「硬合法」（掩码→`-1e9`）；②「全合法回退」（掩码缺失时）；
  ③「最终以 `deploy_card` 为准」（`action_mask.py:3`）。**三者不互相推导**。

### C8 · `-1e9` vs `-inf`（全掩行 → 均匀采样）

- `follower.py:471` / `:498`（及批量同源）用 `masked_fill(..., -1e9)`。
- slot head 有 `STOP` 恒 1.0 兜底（`:325,329`）⇒ 构造上不会全掩；
  **cell head 无兜底、无采样后 verify**（`:496-505`）。
- **并列结论**：slot 侧「硬」由构造保证；cell 侧「硬」仅由 `-1e9` 的数值下溢保证，
  全 False 行会退化为**全网格均匀采样**（MCTS 侧有防御 `mcts.py:166-167`，Follower 侧没有）。

### C9 · `used` 集合的基：L1 0-based vs L2 1-based

- **L1**：`env_wrapper.py:506,513` 集合为 **0-based**（`sa.slot - 1`）；`used_slots` 对外也 0-based（`:531`）。
- **L2**：`action_mask.py:556,566` 集合为 **1-based**（`sa.slot`）。
- **并列结论**：两侧**各自自洽**（已被回归测试覆盖）；但**同名概念两种基**并存，
  是 off-by-one 类缺陷的复发土壤（见第 7 节）。

### C10 · 无坐标越界的引擎 vs 有坐标越界的 L2

- **L2**：`action_mask.py:557-558` `f"坐标越界 ({sa.x},{sa.y})"` —— 本地网格越界即拒。
- **L4**：引擎**无越界拒收**（第 5.3 节，引用）。
- **并列结论**：L2 的「坐标越界」是 **RL 层专有**概念（本地网格 32×18 的域约束），引擎不知道它。

---

## 7. 已知漏洞（**单列**）：掩码 `used` 集合 0/1-based off-by-one

> **当前状态：已修复**（提交 `4fe3933`，工作树 HEAD `267455b` 之上）。
> 本节把**修复前原样代码**与**修复后原样代码**并列留档。

### 7.1 修复前（提交 `792736a`，`git show 792736a:src/clasher_new/rl/env_wrapper.py` 第 498 行，原样）

```python
            elif sa.slot >= 1 and sa.slot <= K_MAX and sa.slot not in used:
                from card_utils import Card
                card = p.cycle[sa.slot - 1]
                cost = Card(card).elixir
                if card == "Mirror" and getattr(p, "last_card", None):
                    cost = Card(p.last_card).elixir + 1
                elixir -= cost
                used.add(sa.slot - 1)
```

（`used.add(sa.slot - 1)` 在旧文件第 **505** 行 —— 存 **0-based**，测 **1-based**。）

### 7.2 修复后（工作树 `env_wrapper.py:506,513`，原样）

```python
            elif sa.slot >= 1 and sa.slot <= K_MAX and (sa.slot - 1) not in used:
```

```python
                used.add(sa.slot - 1)
```

根因注释（`env_wrapper.py:490-497`，原样节选）：

```python
        # ⚠️ off-by-one 修复（2026-09-18）：`used` 里存的是 **0-based 槽位下标**（`sa.slot - 1`），
        # 成员测试也必须 0-based。原写法 `sa.slot not in used` 拿 **1-based** 的 `sa.slot` 去测
        # 0-based 集合：partial bundle 里一旦出现「(s, s−1)」这种**相邻降序对**，低槽位 `s−1`
        # 会因 `s−1 ∈ used`（那是高槽位存进去的下标）被**整段跳过** ⇒ 既漏记 `used`
        # （掩码放行**重复槽位** ⇒ `validate_bundle` **整包拒绝** ⇒ 白掉一帧 + 吃 `invalid_penalty`），
        # 又漏扣该卡费用（圣水模拟与校验口径不一致）。
```

### 7.3 漏洞链（跨四层，逐层原样）

| 层 | 环节 | 代码位置 |
|---|---|---|
| L1 | 掩码放行**已用槽位**（`slots[i]=True`、`cells[i]` 未清零） | `env_wrapper.py:506`（修复前） |
| L1 | 采样器照掩码行事 → 提交**重复槽位**包 | `follower.py:470-473,496-505` |
| L2 | `validate_bundle` 判 `f"slot {sa.slot} 重复"` → **整包拒绝** | `action_mask.py:556` |
| L3 | `invalid_count = 1` → `reward -= 0.05`；该帧**什么都没执行**（B 类） | `env_wrapper.py:669-670,262-263`；`pass_streak_audit.py:101` |

### 7.4 修复配套（同提交，工作树原样存在）

| # | 内容 | 位置 |
|---|---|---|
| ① | 一行修复 `(sa.slot - 1) not in used` | `env_wrapper.py:506` |
| ② | 掩码不变式·1（partial 已用槽位必须已非法） | `follower.py:453-468`（`act`）、`569-576`（`act_parallel`） |
| ③ | 掩码不变式·2（被掩槽位不得被选中） | `follower.py:478-486`（`act`）、`588-594`（`act_parallel`） |
| ④ | 回归测试 `test_mask_partial_bundle_invariants` | `rl/selftest.py`（子任务稿引用；**本报告未自证该测试体** → 「引用」） |

### 7.5 文档留证（本报告未重读，均为子任务稿引用）

- `docs/mask_used_slot_offbyone_fix_2026-09-18.md`（§0 总述 / §1 触发条件 / §2.3 症状 / §3 修复表 / §4 回归测试 / §6 影响 / §7 留证）
- 留证目录 `docs/mask_used_slot_offbyone_fix_2026-09-18/`（7 文件）
- 台账 `docs/agents/ledger.md` §6 · O9（「✅ 2026-09-18 已定位并修复」）
- 实测读数（引用）：A_et **497 / 75,891 帧（0.7%）**、**39 / 280 局**、最长**连续 225 帧 ≈ 112.5 s**、
  B 段起始圣水**全 = 5.0**、每帧 `invalid_penalty = 0.05` ⇒ 225 帧累计 **≈ −11.25**；
  修复后同种子对照：掩码层 **44/256 → 0/256**、rollout 拒绝帧 **474/1707 → 0/1609**、端到端 **B 类 = 0/3413 帧**。

### 7.6 与其它层的关系（并列、不调和）

- 该漏洞已修复，但 `evaluate.py` 的 `bundle_illegal`（L3）**本来就能**看见它（因为它是 L2 拒绝）；
  它看不见的是 **L4 引擎拒绝**（冲突 C5）。
- `flow_league.py:323` 的 `invalid_count=0`（L3）**在任何版本都看不见它**。

---

## 8. 未定清单（**不补全**）

| # | 未定项 | 为什么未定 |
|---|---|---|
| **U1** | 「cell 行是否真会全 `False`」 | 本次只读 `legal_cells`/`_position_legal` 未做穷举取证；MCTS 侧有防御（`mcts.py:166`），Follower 侧无 |
| **U2** | L2 描述里「`validate_bundle` 与 **deploy 的拒收**」中「deploy 的拒收」指 L2 的提交路径还是 L4 的 `deploy_card` | 用户分层描述自身重叠，本报告按 `validate_bundle`→L2 / `deploy_card`→L4 归位 |
| **U3** | 掩码层是否还有**其它**同类的 0/1-based 错位（除已修的 `used`） | 未做全文件的两侧基一致性审计 |
| **U4** | 冲突 C3 的空隙（2 卡包里首卡裸下）是否**实际可达** | 未构造 partial bundle 实测 |
| **U5** | 引擎级拒绝的**逐卡**计数上界 | `invalid_count += 1` 无上限；包内卡数受 `K_MAX` 约束、但 `validate_bundle` **不校验长度**，故实际上界未定 |
| **U6** | `flow_league.py:323` `invalid_count=0` 对 player-1 侧策略的**具体影响量** | 只有自述理由（「一般合法」），未实测 |
| **U7** | `train_solo.py` `ghost_rate` 门禁的报警分支（是否阻断） | 子任务稿称「只报警不阻断」，本次未逐行复核 |
| **U8** | L4 坐标越界「静默夹取」的完整分支 | `battle.py:2681-2692` 为子任务稿引用，本次未读该段自证 |
| **U9** | `battle.py:3398-3438` `use_ability` 全部 `return False` 分支 | 本次仅取子任务稿摘录，未逐行自证 |
| **U10** | `scripts/_mask_vs_engine_reconcile.py` 的 2712 格过严 / 1408 格缺口明细 | 未复跑脚本；本报告只保留冲突条目中对得上代码的部分 |

---

## 9. 一句话合并结论

**L1 说「硬合法」，L2 说「整包要么全过要么全拒」，L3 说「拒了扣 0.05、B 类用会计恒等式自证」，
L4 说「我只回 `bool`，没有理由，也不认越界/数量/法术位置」。**
四层**不是同一条规则的四个视角**，而是四套**并列且互不推导**的判定：
按 §6 的 C1–C10，掩码过严（C1/C6）、掩码缺口（C2）、作用域空隙（C3）、
RL 层专有约束（C4/C10）、统计错位（C5）、口径三并存（C7）、数值软化（C8）、
同名两种基（C9）——**冲突处一律并列，不调和**。

---

## 10. 编排者独立复核（**本会话亲自读源码自证**，非子智能体转述）

> 本节由编排者对**承重结论**逐条重读工作树自证（HEAD `267455b`），并补做暴露面/真实日志取证。
> 与 §2–§9 冲突时，本节为**自证读数**；§2–§9 中标「引用」的条目**仍未复跑**（见 §10.5）。

### 10.1 我亲自读源码自证的条目

| 结论 | 自证动作 | 文件:行号 |
|---|---|---|
| `validate_bundle` 9 条 reason 与短路顺序 | 逐行读完全函数 | `action_mask.py:521-573` |
| 8h「不裸下」两处**作用域不同** | 两处原样代码对读 | L1 `env_wrapper.py:521-525` vs L2 `action_mask.py:568-572` |
| 法术三道闸门 + **提前 `return True`** | 读全函数 | `action_mask.py:394-405` |
| **Miner 在掩码层无豁免** | `grep -n "Miner" rl/action_mask.py` ⇒ **0 命中**（全文件无此字样） | `action_mask.py:411-434` 照走区域规则 |
| `_card_cost` **无 MergeMaiden 分支** | 读全函数 | `action_mask.py:30-36` |
| `invalid_penalty` 默认值 + 扣分式 | 读常量与 compute_reward 尾段 | `env_wrapper.py:50`、`:262-263` |
| `invalid_count` 两来源（整包拒 =1 / 引擎拒 逐卡 +1） | 读提交路径 | `env_wrapper.py:667-692` |
| MergeMaiden 引擎侧**动态费 3/6** | 读引擎实现 | `battle.py:3023-3045` |

### 10.2 ★ 新增发现 F1：MergeMaiden 在掩码层**同时**有「费用过严」与「位置过松」两个**方向相反**的分裂

§6-C1 只报了费用那一半（掩码 6 vs 引擎 3/6）。**位置那一半是反方向的**，且根因是**卡面 type 与引擎实际形态 type 不同**：

| 面 | 用哪个卡名去查 `Card.type` | 实测 type | 后果 |
|---|---|---|---|
| **L1 掩码** | `Card("MergeMaiden")`（`action_mask.py:393` 自建） | **`spell`** | `_position_legal` 在 `action_mask.py:394-405` **提前 `return True`** ⇒ 占位格 / 王塔身后 / 六条区域规则 / 后排几何门**全部跳过** ⇒ 位置**过松** |
| **L4 引擎** | `_form` = `MergeMaiden_Mounted` / `MergeMaiden_Normal`（`battle.py:3031/3035/3040` **重绑 `card_name`**） | **`character`** | `battle.py:3055-3077` 的**非法术分支全部适用** ⇒ 引擎会拒 |

实测（本会话，`./.venv/Scripts/python.exe`，cwd `src/clasher_new`）：

```
MergeMaiden            cost=6    type=spell        ← 掩码层看到的
MergeMaiden_Mounted    cost=6    type=character    ← 引擎实际部署的（圣水≥6）
MergeMaiden_Normal     cost=3    type=character    ← 引擎实际部署的（圣水<6）
```

**并列结论（不调和）**：同一张卡，**费用方向掩码偏严**（6 > 引擎 3）、**位置方向掩码偏松**（无约束 < 引擎六条区域）。
两个方向**相反**，**不能只报一半**。缺口位置 = `_effective_card`（`action_mask.py:39-43`）**只处理 Mirror、不处理 MergeMaiden**。

### 10.3 ★ 新增发现 F2：三个冲突卡的**真实暴露面不同**（§2–§9 未做此切分）

| 卡 | 冲突 | `DEFAULT_SOLO_DECK` | `FOUR_DECK_SET` | `build_card_pool()`（实测 139 卡） |
|---|---|---|---|---|
| **BarbLog** | C2（掩码缺口） | ✗ | **✓（第 4 副「巨骷髅攻城槌」，`opponents.py:44`）** | ✓ |
| **Miner** | C6（掩码过严） | ✗ | ✗ | ✓ |
| **MergeMaiden** | C1 + **F1** | ✗ | ✗ | ✓ |

- **`solo` 模式**：`train_solo.py:115-118` 的 `deck_set="four"` ⇒ p1 从 `FOUR_DECK_SET` 抽一副 ⇒ **BarbLog 缺口在 solo 里实时暴露**；Miner / MergeMaiden 在 solo 的两套卡组集合里**都不出现** ⇒ **C1/C6/F1 在当前 solo 卡组下不可达（潜伏）**。
- **`run` 模式**：`run_league.py:795-812` 注册「全随机」agent（`pool = build_card_pool()`，`opponents.sample_deck` 每局抽 8 张，`opponents.py:71-73`）⇒ **Miner / MergeMaiden / BarbLog 全部可达**（p1 侧）。
- ⚠️ **但 `run` 模式的 p1 侧非法动作不被 L3 计数**（`flow_league.py:323` 硬编码 `invalid_count=0`）⇒ 这些冲突在 `run` 里**静默**。

### 10.4 ★ 真实日志读数：P1-20（`validate 通过但引擎拒绝`）的卡名分布

自证命令（可复跑，`docs/*.log`）：

```
grep -ah "P1-20" docs/*.log | wc -l                    → 1644
grep -ah "P1-20" docs/*.log | grep -ac "BarbLog"       → 1644
grep -ah "P1-20" docs/*.log | grep -av "BarbLog"       → 0
```

| 日志文件 | P1-20 行 | 其中 BarbLog |
|---|---|---|
| `train_long1m.log` | 561 | 561 |
| `train_run100k.log` | 487 | 487 |
| `train_run_smoke_2026-09-17.log` | 274 | 274 |
| `train_et_measure.log` | 165 | 165 |
| `train_et_measure2.log` | 157 | 157 |
| **合计** | **1,644** | **1,644（100.0%）** |

⇒ 与 §6-C2 的预测**逐项一致**：掩码缺口**只**发生在 BarbLog 上 —— 它正是**唯一**「引擎有区域限制而掩码没有」的卡，且在真实训练里**发生了 1,644 次**（每次 `invalid_count += 1` ⇒ 每次 −0.05）。

⚠️ **本读数的限定（不得越读）**：
- 1,644 是**日志行数，不是独立事件数** —— `train_et_measure*.log` 之间存在**重复 run**（discarded 重跑）的可能，未逐 run 去重；
- 累计扣分若按行数算为 **1644 × 0.05 = 82.2**，这是**跨 5 份日志的全部 run 之和**，**不得**按单局/单臂读（【R9】尺度纪律）；
- 本读数为**描述性**，**不是判据**（【R3】）；未做逐局/逐臂归因。

### 10.5 我本轮**仍未**自证（继承 §8，并新增 U11）

- **U8**（`battle.py:2681-2692` 越界静默夹取）、**U9**（`battle.py:3398-3438` `use_ability` 全部 `return False` 分支）、**U10**（`scripts/_mask_vs_engine_reconcile.py` 的 2712 格过严 / 1408 格缺口明细）⇒ **我本轮未读/未复跑**，仍为子任务稿**引用**。
- **U11（新增）**：**F1 的「位置过松」是否实际可达** —— 需要 MergeMaiden 出现在某个真实卡组里（当前只在 139 卡随机池内）⇒ **未构造对局实测**，**不补全**。
- **U12（新增）**：`Card("MergeMaiden").type == "spell"` 是**引擎数据侧的有意设计还是数据缺陷** ⇒ **未定**（我只量到「掩码据此跳过了位置规则」这一**代码后果**，不判数据对错）。

---

## 11. 附：`invalid_penalty` 的**量纲**（2026-09-19 追加；**推导，非实测**）

> 触发：用户提问「对于决策帧内非法出牌（圣水不够、位置不对），我们的惩罚是？」
> ⇒ 本节只做一件事：把 §4 的 `0.05` 换算到同一把尺子上，并指出它**与哪一项相等**。
> ⚠️ 本节全部数字来自**常数字面量 + 一行乘除**（【R3】不构成判据；【R10】无新实测）。

### 11.1 第一层：这两类**不可选**，惩罚恒 0（设计意图，不是"罚得轻"）

| 非法类型 | L1 掩码（不可选） | L2 校验（**同一谓词**） |
|---|---|---|
| **圣水不够** | `slot_mask` → `_slot_playable`（`elixir < cost` ⇒ False）：`action_mask.py:46-57`、`:63-71` | `validate_bundle` 调**同一函数**：`action_mask.py:561-562` |
| **位置不对** | `legal_cells` 576 格逐格 → `_position_legal`（`:447` 注释自陈「与提交路径完全同源，P0-4」）：`action_mask.py:438-478` | `validate_bundle` 调**同一函数**：`action_mask.py:563-564` |

⇒ `masked_fill(-1e9)`（`follower.py:471`/`:498`；`act_parallel:581`；`evaluate:773/787/802` 同口径）⇒ **采样概率恒 0**；且有运行时不变量断言（`follower.py:481-485`：采到被掩槽位直接抛错）。
**决策帧内**第 2 张及以后的牌同样被掩——掩码用**模拟扣费后的圣水**（`env_wrapper.py:284-301`：`elixir -= cost` → `slot_mask(..., elixir_override=elixir, used_slots=used)`）。

### 11.2 第二层：漏网时 `0.05 × invalid_count`，**每决策帧**结算一次

`compute_reward` 尾段 `rl/reward.py:258-259`；常量两处同值（`rl/config.py:58` = `rl/reward.py:46` = `0.05`）；**预设覆盖** `defensive` = **0.1**（`config.py:394-396`），其余 5 个预设 = 0.05。
两个来源（§4.3）：**L2 整包拒 = 恒 1**（`env_wrapper.py:454-456`：包内几张卡都只算 1，且**整包什么都不执行**）；**L4 引擎逐卡拒 = 逐卡 +1**（`:463`/`:475`：包内已部分提交**不可回滚**，并发 `RuntimeWarning P1-20`）。

### 11.3 ★ 换算：`0.05` 是多少

一决策帧 = `decision_frames=30` × `dt=1/60` = **0.5 s**（`env_wrapper.py:88,90`）；回费 `1/2.8` 圣水/s（`battle.py:2881`、`player.py:32-34`）⇒ **0.1786 圣水/决策帧**。

| 换算 | 值 |
|---|---|
| `0.05 ÷ elixir_diff_weight(0.5)` | **0.1 圣水** |
| `0.05 ÷ tower_dmg_self(0.0012)` | **41.7 点塔血**（≈公主塔 3052 的 **1.37%**） |
| `0.05 ÷ win_bonus(10)` | **1/200 个胜局** |
| 对照：**225 帧连续整包拒**（§7 实测最长） | **−11.25**（与 `lose_penalty=10` 同级）⇒ 比"输一局"还贵 |

### 11.4 ★ 更该注意：`edw` 对「这一帧到底有没有出牌」**不可分辨**

`config.py:31-33` 自陈资源账口径：「部署帧 `E−c` / `V+c` 抵消 → **下牌不罚**」。
⇒ **出牌帧**与**整包被拒的白掉帧**在 `edw` 项上取值**完全相同**（都是"Φ 只被回费推动"）⇒ 两者**唯一的差**就是 `−0.05`。
⇒ 后果一：**孤立一次白掉帧** = `−0.05` 加上回费自身的 edw 收益（`0.1786 × 0.5 = +0.089`，**仅在对手 Φ 不变的那一帧**）⇒ **净值可为正**，即 `0.05` **落在背景项之下**。
⇒ 后果二：奖励里**没有时间/节奏项** ⇒ 白掉一帧丢掉的**场面机会成本**（§2 L1、§5 L4）**不在任何一项里**。
⇒ 结论：该惩罚的真实效力来自**「别卡死」**（连续拒），**不是「别试探一次」**。

### 11.5 口径缺口：`evaluate` 只看得见 L2

`evaluate.py` 只有 `bundle_illegal`（`bundle_ok=False` ⇒ `:217-218`；打印 "Bundle 合法率" `:274`），**从不读 `info["invalid_count"]`** ⇒
**L4 逐卡拒（`bundle_ok=True` + `invalid_count>0`，§6-C2）在「合法率」里不可见**；但 `Mean Reward`（`:243` 累加 `r`）里的 `−0.05` **是在里面的**（来自 `compute_reward`）。
⇒ **「合法率 = 1.000」≠「一次罚都没吃」**，两处必须分开读。
另：`run`/flow 模式 p1 侧硬编码 `invalid_count=0`（`flow_league.py:323`）⇒ 对手非法**不计数、不罚**（§6）。
「不裸下」`solo_commit_blocked` 也是 **L2 拒收集**的一员（校验侧 `action_mask.py:568-572`；掩码侧 `env_wrapper.py:306-311`）⇒ 同样只吃 `−0.05 + 白掉一帧`。

### 11.6 边界（不越读）

- **全部为推导**：未新跑任何对局；`+0.089 / 净 +0.039` 只在「对手 Φ 不变」时成立，是**说明性算术**，**不是**对真实 run 的测量（【R10】/【R9】）。
- 未定项照 §8：**U5**（逐卡计数上界未定）、**U6**（flow `invalid_count=0` 的影响量未实测）、**U11**（F1 MergeMaiden「位置过松」是否实际可达）。
- 本节**不改任何代码、不改任何常量**（【R11】/【R3】）。

---

## 12. 附：**能否改造代码"惩罚模型非法放牌"** —— 四个杠杆 + 逐格对账复核（2026-09-19 追加）

> 触发：用户追问「所以我们能否改造代码从而惩罚模型非法放牌？」
> 结论：**能，但对本问题是最差的工具**——§11.1 的两类（圣水不够 / 位置不对）**已不在动作 support 里**，
> 加系数是给恒 0 事件调参；而真在发生的缺口，罚的是**掩码的谎**、不是模型的犯规。正确改造 = **补掩码**（概率归 0），
> 把 `invalid_penalty` 降级为**报警器**。本节含 **2712/1408 的独立复跑**（§8-U10 首次复现）+ **两处新缺口**。

### 12.1 四层杠杆与各自的代价

| 杠杆 | 改什么 | 对「圣水不够/位置不对」 | 对**真实**缺口 | 代价 / 门禁 |
|---|---|---|---|---|
| **A 抬系数** | `invalid_penalty` 0.05 → 更大（`config.py:58` = `reward.py:46`，预设 `config.py:394-396`） | ❌ **恒 0 事件**（已被掩） | ⚠️ 有效但**错位**（见 §12.2） | 奖励改动 ⇒ **【R11】** A′ 前置 + **【R3】** 预注册；且污染 critic |
| **B 计数口径** | L2「整包恒 1」→ 逐子动作（`env_wrapper.py:454-456`） | ❌ | ⚠️ 极小（事件量已近 0） | 低风险低价值 |
| **C 时间/节奏项** | 新增"白掉一帧"的机会成本 | ❌（与非法无关） | ✅ 治"卡死"真痛点 | 奖励改动 ⇒ 【R11】+ 预注册 + 【R7】汇率同源 |
| **D 补掩码（推荐）** | `action_mask` 的 3~5 处例外（§12.3） | ✅ **概率归 0**（强于任何罚） | ✅ **根因** | **【R13】** 位图对账 128 张 + 负对照；**非**奖励改动 ⇒ **不受 R11 约束** |

### 12.2 为什么「加罚」对本问题是错的工具（六条，按效力排序）

1. **这两类已不在 support 里**：`slot_mask`（圣水/手牌/塔状态）与 `legal_cells`（576 格）都掩，`masked_fill(-1e9)` + 运行时断言（`follower.py:481-485`）⇒ 采样概率恒 0。**加罚 = 给不会触发的事件调参。**
2. **非法动作在掩码的 support 里** ⇒ 罚的是**掩码撒谎**，不是模型犯规。梯度只能到达"这张卡在这族落点上少采样"，**到不了"越区"这条规则**。
3. **【R12 / EBK】规则是确定性可算的** ⇒ 「能算的不许让网络猜」：该把规则补进掩码，而不是让网络用 `0.05` 的噪声梯度去反推一条 `if`。
4. **污染 critic**：该帧的 `−0.05` 由**掩码 bug** 决定而非状态决定 ⇒ 价值头被迫拟合一个**不可行动**的量（与已确诊的"常数 critic / 负 EV"同族失效）。
5. **孤立帧内无指向性**：若某帧**只有** bug 卡合法可出，全部动作同吃 `−0.05` ⇒ 帧内策略梯度为 0（纯常数偏移）。
6. **BarbLog 缺口是"整张卡全场过松"** ⇒ 加罚学到的是「别用 BarbLog」（**错**），不是「BarbLog 只能放己方半场」（**对**）。

### 12.3 ★ 真正该改的：掩码的 3~5 处例外（逐格对账已复核）

**复跑**：`cd src/clasher_new && PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe ../../scripts/_mask_vs_engine_reconcile.py`
⇒ 原始输出留证 [`docs/mask_reconcile_2026-09-19.md`](mask_reconcile_2026-09-19.md)。**合计：过严 2712 格 / 缺口 1408 格 / 异常 0** —— 与 §8-**U10** 引用值（子任务稿）**逐值相同**，首度复现。

| # | 卡 | 掩码依据 | 引擎依据 | 方向 | 实测格数 | 后果 | 判定 |
|---|---|---|---|---|---|---|---|
| 1 | **BarbLog** | `Card('BarbLog').type == 'spell'` ⇒ 走**法术分支**（全场 + 法术闸门） | `battle.py:3061-3062` 强制部队区（`can_deploy_at(is_spell=False)`） | 掩码**过松** | **352 格/方**（合计 704） | L4 逐卡拒 ⇒ `invalid_count += 1` ⇒ `−0.05` + 该卡白掉（包内其余卡**已提交、不可回滚**） | **真缺口**（§10.4 已 1,644 次实测） |
| 2 | **MergeMaiden** | `Card('MergeMaiden').type == 'spell'`（掩码只处理 Mirror）；费用 = `elixir 6`（`_card_cost:36`） | 形态改写 `card_name = _form('_Normal'/'_Mounted')` ⇒ `type == 'character'`；`_merge_cost` = **3（<6 费）/ 6**（`battle.py:3040-3056`） | 位置**过松** + 费用**过严** | 未测（不在 reconcile 的 `CARDS`） | 位置 ⇒ L4 逐卡拒（含 `−0.05`）；费用 ⇒ **5 费时本可出的 3 费形态被掩**（丢合法动作） | 代码级成立；**实际可达性 = §8-U11，仍未测** |
| 3 | **Miner（★ 新发现）** | `_position_legal` **无 Miner 分支** ⇒ 按 `character` 己半场/河规则 | `battle.py:3078-3087` 勘误批8：**全场可部署**（含敌半场，P0/P1 都有） | 掩码**过严** | **284 格/方**（合计 568）；掩码 224 / 引擎 508 | **只是丢掉合法动作**（**不吃罚**） | **真缺口 · 放错方向的镜像** |
| 4 | Arrows/Fireball 等伤害法术 | 7h/8h/9h 闸门（已毁塔本体 / 空砸 / 前段塔 EV） | 引擎**不认**这些闸门 | 掩码**过严** | 540 + 532 格/方（合计 2144） | 无 | **设计如此**（策略侧硬约束），**不是 bug** |
| 5 | **Mirror** | `legal_cells` 内部**也**调 `_effective_card` ⇒ 与提交同源 | `_from_mirror=True` 递归 ⇒ **跳过** BarbLog 区域检查与手牌/形态检查；费用 `last_card_cost + 1` | 大体同源 | reconcile 显示 352×2 | ⇒ **⚠️ 该行是脚手架口径缺陷**（见下）；残余真差：Mirror 重放 **3 费** MergeMaiden 时掩码要 **7 费**、引擎只要 **4 费**（过严） | 显示值伪 / 余项真 |

**★ 对账脚手架自身的口径缺陷（【R17】违规，必须记下）**：`scripts/_mask_vs_engine_reconcile.py:60-72` 的
`engine_accepts` 在 **deepcopy 之后**改写 `cycle` 与 `last_card='Knight'`，而掩码一侧 `legal_cells(bs, ...)` 用的是
**未改写**的 `bs`（`last_card = None`，已实测默认值）。⇒ **两侧不同源**，`Mirror` 行（352×2）**不可判**：
掩码侧按 `Card('Mirror').type=='spell'` 放行全场、引擎侧按 `Knight` 要求部队区。**真实代码里 Mirror 是同源的**
（`legal_cells:452` 有 `eff = _effective_card(p, card_name)`）⇒ 修脚手架（两侧同源）后该行应归零。
**逐卡明细的统计口径因此只对 1/3/4 号可信；2/5 号须补测。**

### 12.4 推荐的执行顺序（D 先于 A，A 内部按"零误拒"排序）

1. **D（零风险，先做）**：把**非法率 + 分类**做成训练内建指标（`L2/L4 × 卡名 × reason`，落 `gates.json` **相对基线**）——
   现状只有 `RuntimeWarning` 进日志，且 `evaluate` **看不见 L4**（§11.5）⇒ 掩码缺口不会自曝。
2. **A4（Miner）**：`_position_legal` 补 Miner 全场分支（`battle.py:3078-3087`）⇒ **多回 568 格合法动作**，**纯赚、零误拒**。
3. **A1（BarbLog）**：`_position_legal` 加 `BarbLog` 走部队分支。★ **一处真实分叉**：掩码不知道 `_from_mirror`
   （引擎在 `_from_mirror=True` 时**跳过**该检查）⇒ ①**从严**（Mirror 重放 BarbLog 也按部队区，安全但少掉合法格）
   或 ②把 `_from_mirror` **透传进掩码**（改签名，波及 `validate_bundle` / `legal_cells` / `mcts`）。
   **建议先①从严**，并把"少掉的格数"量出来再定。
4. **A2（MergeMaiden）**：给 `_effective_card` / `_card_cost` 加形态解析（≥6 费 ⇒ `_Mounted`/6，否则 `_Normal`/3），
   一并收敛位置（过松）与费用（过严）。⚠️ 这条**放宽**合法集 ⇒ **【R13】位图对账是强制的**。
5. **A3（同 tick 建筑占位滞后）**：bundle 内第 2 张建筑落在第 1 张的格子上不会被掩 ⇒ 需在
   `get_action_mask_for` 的模拟里把已提交建筑叠加进 `is_position_occupied_by_building`（比逐子动作重算更贴"同源"）。
6. **B/C 暂不动**：B 价值极小；C 属奖励改动，**【R11】须先 A′ 类取证 + 单独预注册**，且 C 治的是"白掉一帧的时间成本"、
   **与"非法"是两件事**（§11.4 已证 `edw` 对出牌帧与白掉帧**不可分辨**）。

### 12.5 边界

- 本节**未改任何代码/常量**（【R11】【R3】）；`2712/1408/0` 是**只读复跑**（§12.3 命令原样可复现），`Miner 284` 与
  `MergeMaiden` 两行是**读源码 + 实测 `Card().type`** 所得，**非**新对局实测。
- 未定照 §8：**U11**（MergeMaiden 真实可达性）、**U5/U6** 不变；**U10 已复现**（2712/1408），可移出未定清单。
- **新增 U13**：`_mask_vs_engine_reconcile.py` 两侧状态不同源（§12.3），修好后须重跑才能判 `Mirror` 行。
