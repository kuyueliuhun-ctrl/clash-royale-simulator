# 引擎侧「出牌溯源」纯记录通道（S2 第 5 项）· 实施记录（2026-09-18）

> **状态：已在 sandbox 完整实现并验证行为中立，但【尚未落到真树】。**
> 原因：`run100k` 训练进行中（`spawn` worker 会**重新 import 源码**）⇒ 此刻改真树会污染在跑的实验（【R1】）。
> sandbox 与真树的**逐字 diff**：[`root_cast_channel_2026-09-18.diff`](root_cast_channel_2026-09-18.diff)（412 行 / +134 −40）。
>
> 规格：[`engagement_trade_prereg_2026-09-18.md`](engagement_trade_prereg_2026-09-18.md) §3（溯源路由）、§6（改动清单）

---

## §1 为什么需要它（与 S2 判读的关系）

S2 的**全部读数**（[`s2_instrument_2026-09-18.md`](s2_instrument_2026-09-18.md)）建立在 **schema-4** 录像上，
而 schema 4 缺两个字段 ⇒ 两处**退化**：

| 退化 | 后果 | 本通道解决什么 |
|---|---|---|
| 没有 `is_product` | 规则①（交战对至少一侧非产物）**退化为恒真** ⇒ "纯产物互殴"也被算作战交对 | `is_product` 直接写出 |
| 残值 `V` 只能**离线重建** | 帧内延迟出兵/建筑产物可能污染份额 | 帧内 `v0`/`v1` 与每实体 `share` 写出后**零误差**，并可**对账重建法** |

⇒ 本通道不是"为奖励项服务"的第一步，而是**把 S2 仪器从近似变成精确**的第一步。
（奖励项的接线属 §6 第 5 项，**仍未做**，且按 S1/S2 的门禁判定**暂不应做**，见 §5。）

---

## §2 改了什么（`battle.py`，纯新增 + 两处改名）

| # | 改动 | 位置 | 中立性依据 |
|---|---|---|---|
| 1 | 实体新增 `root_cast` / `is_product`（默认 `None`） | `Entity.__init__` | 纯新增字段，不参与任何分支 |
| 2 | 新增 `next_cast_id` / `cast_log` / `_cast_ctx` / `_cast_is_product` | `BattleState.__init__` | 纯新增字段 |
| 3 | `deploy_card` → `_deploy_card_impl` + **薄包装** `deploy_card` | `BattleState` | 包装体只分配 cast id、设置/恢复上下文、成功时写日志 |
| 4 | `_spawn_entity(self, entity, spawner=None, is_product=None)` | `BattleState` | 只填两个记录字段后 `return entity` |
| 5 | `delayed_spawn(..., is_product=None, spawner=None)`；`schedule` 第三格带 `(cast_id, is_product)` | `BattleState` | `schedule` 的消费者只读 `item[0]`/`item[1]`（`card_mechanics.py:1228`、`spell_module.py:126`）⇒ 3 元组兼容 |
| 6 | **全部 26 个出生点**按 AST 插入 `spawner=` / `is_product=` | 见 §3 审计 | 插入的实参不被任何分支读取 |
| 7 | 复用队列（觉醒临时复活）把宿主 `root_cast` 记进队列 | `Entity._evo_on_death` / `BattleState.step` | 队列元素多一格，消费侧按 `item[6]` 可选读 |
| 8 | `spawn_arrival_troops` / `_spawn_action_character` 增加 `host=None` | 模块级 | 只在 `delayed_spawn`/`_spawn_entity` 里用 |
| 9 | `Entity.create_projectile` 的**绕过点**就地补记 | `battle.py:800` | 该路径不经 `_spawn_entity`（否则 `is_product` 永远是 `None`） |

**溯源指针写在产物体自己身上**（规格 §3.4）：`root_cast` 链式继承到根
（`Tombstone → Skeleton` 的 `root_cast` 就是**那次出牌事件 id**，不是 Tombstone 的实体 id），
所以宿主死亡不影响归属，**也不需要"最近一次/最近两次"的批次计数**。

---

## §3 出生点覆盖审计（AST，不靠 grep 猜）

```
python3 - <<'PY'   # 见 §6 复算命令
未带 spawner=/is_product= 的调用点：
   L2889  step                 （预期：schedule 循环，字段已在排程时预置）
   L3091/3104/3115/3119/3152/3190/3206  _deploy_card_impl
                               （预期：出牌上下文 `_cast_ctx`/`_cast_is_product` 负责）
   L2786  delayed_spawn        （预期：定义体自身）
```

`_deploy_card_impl` 里 7 处**故意不插**：它们生成的正是**这次出牌直接产生的东西**
（法术的 `AreaEffect`/`EvoZapZone`/克隆体/墓园骷髅），由 `_cast_is_product` 统一判定；
塔的 6 处显式 `is_product=False`；其余 20 处 `spawner=self/src/projectile/host`。

---

## §4 ★ 行为中立性实测（开/关逐位对账）

工具：[`scripts/s2_neutrality_probe.py`](../scripts/s2_neutrality_probe.py)（同一段确定性推演跑**两棵树**，
对每 tick 的**全部实体**（id/存活/hp/位置/索敌/归属）做 SHA-256）：

```
PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe ../../scripts/s2_neutrality_probe.py <tree_dir>
```

| 项 | 原树（未打补丁） | sandbox（已打补丁） |
|---|---|---|
| **DIGEST（1200 tick × 全实体逐位）** | `9adc2aafee52aa55f964fb52af7ac86fc240d105058d880bb50772b95bcd5b55` | **`9adc…5b55`（逐字相同）** |
| TIME / ELIXIR / ENTITIES | 40.000000 / 10.0 10.0 / 10 | 同 |
| `is_product` | 全 `None`（0/0/10） | **True=4 / False=6 / None=0** |
| `root_cast` 已填 | 0/10 | **3/10**（= 3 个真产物） |
| `cast_log` | 不存在 | **5** 次成功出牌（`next_cast_id=85`，含 80 次被引擎拒绝的尝试） |

推演脚本刻意覆盖：单位（Giant/Knight/Skeletons）、建筑（Tombstone）、
法术（Fireball）、**建筑召唤物**（Tombstone→Skeleton）、亡语、塔的弹道（`create_projectile` 绕过点）。
⇒ **在同一 seed 的 1200 tick、5 次成功出牌、4 个在场产物上，两棵树的状态摘要逐位相同。**
（这是纯记录通道能做到的最强中立性证据；**不是**完整 9j 式指纹，跨局边界指纹要在真树落地后补。）

---

## §5 与门禁判定的关系（为什么"做了但不接线"）

同日两条**独立**门禁都指向"暂停结算改动"（【R11】）：

| 门禁 | 结论 | 出处 |
|---|---|---|
| **S1**（零训练成本，手写专家） | **未通过** ⇒ **约束主要在探索/机制侧**（Xbow 不在 `ACE_CARDS`/`SINK_TANK_CARDS` ⇒ `save_ace`/`setup_wait` 在 Xbow 卡组上 0 帧触发；`_pick_suggested_card` 只推荐付得起的牌 ⇒ Xbow 要 6 费而自锁） | [`s1_gate_2026-09-18.md`](s1_gate_2026-09-18.md) |
| **S2**（离线仪器 + 相关性） | **有候选、无判决**（P3 被推翻 / P1·P2 否决 / P4b 方向 5/5 稳但幅度未超批间散布） | [`s2_instrument_2026-09-18.md`](s2_instrument_2026-09-18.md) |

⇒ **`engagement_trade` 奖励项不接线、S3 的 A/B 不开**。
本通道保留为**已完成的、可即刻落地的**基础设施：它唯一的收益是把 S2 仪器从"近似"变成"精确"
（`is_product` + 帧内 `v0/v1` + 每实体 `share`），而**不依赖任何奖励项**。

---

## §6 落地步骤（`run100k` 结束后执行，一步一验）

```bash
cd /mnt/e/clash-royale-simulator-main

# 1) 备份 + 应用（sandbox diff 是逐字证据）
cp src/clasher_new/battle.py /tmp/battle.py.bak
patch -p0 src/clasher_new/battle.py < docs/root_cast_channel_2026-09-18.diff
#   （或用生成器：python3 scripts/_patch_battle_root_cast.py --file src/clasher_new/battle.py --apply）

# 2) 中立性逐位对账（必须与原树的 DIGEST 行完全一致）
cd src/clasher_new && PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe \
    ../../scripts/s2_neutrality_probe.py .

# 3) 非战斗实体契约 + 跨局（【R19】子集；本通道给弹道/领域实体写了两个新字段）
PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe ../../scripts/run_selftests.py \
    test_noncombat_entity_contract test_replay_roundtrip test_league_replays

# 4) 再升 schema（4→5）并让仪器吃精确值：实体追加 root_cast/is_product/share，
#    帧追加 v0/v1（清单见 s2_instrument §9）——**这一步还没做**。
```

**回滚**：`cp /tmp/battle.py.bak src/clasher_new/battle.py`（或 `git checkout --`）。

---

## §7 局限（【R10】）

1. **未在真树落地、未跑真仓 selftest 子集**、**未做跨局边界指纹** ⇒ 中立性证据局限于单局 1200 tick。
2. **`cast_log` 只记成功出牌**，失败留空洞（`next_cast_id` 不连续）——这是**有意**的，
   但任何按 cast id 连续性做的推断都会错。
3. **`is_product` 的语义是"生出它的不是出牌"**：法术直接生成的东西（墓园骷髅/克隆体/领域）记
   `True`；被出牌直接放下的单位/建筑记 `False`。这是一条**约定**，写在这里以免下游误读。
4. `create_projectile` 那条绕过路径已补记，但**同类绕过点没有做全仓静态扫描**
   （只用 AST 扫了 `_spawn_entity` 调用点 + 跑了一次探针看 `None` 计数）。
5. **本通道不产生任何奖励**；把它当作"奖励已经修好了"是错的（§5）。

---

## §8 schema 5（帧内真值）+ 受控落地（2026-09-18 第二轮补记）

### 8.1 schema 5 改了什么（sandbox 已端到端验证）
| 位置 | 内容 | 文件 |
|---|---|---|
| 实体元组 12 → **15** | 末尾追加 `root_cast`、`is_product`、`share` | `rl/replay.py` |
| 帧 | 追加 **`v0` / `v1`**（在线 `RLEnv._active_v`） | `rl/replay.py::battle_snapshot(battle, bundle, reward, info, v=None, shares=None)` |
| 调用方 | `LeagueGameRecorder.record` 从 `env._active_v` / `env._v_share` 取值传入 | `rl/run_league.py` |
| 常量 | `LEAGUE_REPLAY_SCHEMA` **4 → 5** | `rl/replay.py` |

**`share` 的口径**：不在台账里的实体（塔、产物体）记 **0.0**（不是 `None`）——
否则离线读方会回落到卡表，而 `IceGolemite` 在卡表里是 **2 费** ⇒ 会把产物估成 2 费（**违反 §4 的货币泵否决**）。
`v`/`shares` 缺省（如 `rl/human_play.py`）⇒ 这些字段为 `None`，读方须容忍。

**sandbox 探针**（`scripts/_schema5_probe.py`，合成局 + 影子 env 台账）实测：
`LEAGUE_REPLAY_SCHEMA = 5`、帧含 `v0/v1`、实体元组**全部 15 长**、
`is_product=True` 命中 16 条（如 `WitchProjectile`）、`share` 非零 6 条、
`phi_series` 报 **`exact=True` / `v_source=frame`**，并自动跑**重建 vs 帧内真值**对账。
⚠️ 该探针的"影子台账"是对 `RLEnv._deploy_ledger` 的**粗略仿写**（不处理 `_seen_max_id` 水位与延迟出兵），
所以它报出的重建误差（8.0）**不代表真 env**；真误差要等真树 smoke 产出 schema-5 录像后才能量。

### 8.2 受控落地脚本：`scripts/_apply_s2_channel_when_idle.sh`
**为什么需要它**：`run_league.py` 用 Windows `spawn` 起 worker，worker 会**重新 import 源码** ⇒
训练期间改源码会污染在跑的实验（【R1】）。脚本流程：

1. **等** `runs/run100k/replays/league_100000.pkl` 落盘 **且** `run_state.json` 的 `step == 100000`，
   再等 **120 s** 让 worker 退出（最多等 2 h，超时则**不做任何改动**）；
2. 备份 `battle.py` / `rl/replay.py` / `rl/run_league.py` 到 `/tmp/s2_apply_bak/`；
3. 三个补丁**先 `--dry-run` 预检**、再应用（任一失败 ⇒ 退出码非零）；
4. **验证 1/3 行为中立性**：`scripts/s2_neutrality_probe.py .` 的 DIGEST 必须等于
   `9adc2aafee52aa55f964fb52af7ac86fc240d105058d880bb50772b95bcd5b55`（= 未打补丁的基线），
   不一致 ⇒ **自动回滚**；
5. **验证 2/3**：`run_selftests.py test_noncombat_entity_contract test_replay_roundtrip test_league_replays test_dashboard_replays`（【R19】子集），失败 ⇒ 回滚；
6. **验证 3/3**：短 smoke（`--config-name run_schema5 --total-steps 600 --steps-per-eval 300`）
   产出真 schema-5 录像，打印元组长度与 `v0` 帧数，并用仪器复算（应报 `exact=True`）。

**补丁（三个，均可 `patch -p0 <file> < target` 独立应用）**：
`docs/root_cast_channel_2026-09-18.diff`（battle.py，+134 −40）、
`docs/schema5_replay_2026-09-18.diff`（rl/replay.py）、
`docs/schema5_run_league_2026-09-18.diff`（rl/run_league.py）。
三个都已在本机 `--dry-run` 通过。

**手动回滚**：`cp /tmp/s2_apply_bak/battle.py src/clasher_new/ && cp /tmp/s2_apply_bak/rl/*.py src/clasher_new/rl/`
（或 `git checkout -- src/clasher_new`）。
