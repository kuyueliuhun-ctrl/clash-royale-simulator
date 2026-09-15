# 预注册：用引擎精确塔伤取缔 `belief_planner` 的近似威胁估计（P3-1 测量阶段）

日期：2026-09-14
分支：`threat-precise-p3`
上游：`docs/channel_gradient_verdict_2026-09-14.md`（P1 判读）
用户目标：**让精确计算血量取缔近似估计**。

---

## 0. 被取缔的对象（脚本从源码取，不手抄）

`rl/belief_planner.py:159-173`：

```python
def _enemy_pressure(battle):
    """敌我双方实体在各自半场的推进压力（粗略威胁估计，排除静态塔）。"""
    threat = 0.0; my_pressure = 0.0
    for e in battle.entities.values():
        ...
        if e.player == 1:
            threat += 1.0 + max(0.0, (16 - e.position.y) / 16.0)   # ← 只数单位 + 距离
        else:
            my_pressure += 1.0 + max(0.0, (e.position.y - 16) / 16.0)
    return threat, my_pressure
```

**它错在哪（定性，跑之前就说清）**：它既不看**单位能不能打到塔**（卡组/索敌/射程/移速），
也不看**塔兵反击/国王塔激活**，也不看**单位会不会先死**，更**不区分一只 Giant 与一只
Musketeer**（各算 1.0）。它把"有多少个单位在场上"当成了"会掉多少塔血"。

**它的真实消费面（全仓 grep，`rl/belief_planner.py`）**：

| 位置 | 用法 | 尺度敏感性 |
|---|---|---|
| `:624` `_setup_wait` | `threat >= PRESSURE_THRESHOLD`（=2.0） | **阈值** |
| `:755` `_cycle_small` | 同上 | **阈值** |
| `:799` `plan()` | `threat >= 2.0 and threat >= my_pressure * 0.8` → `defend_*` | **阈值 + 比值** |
| `:801` | `my_pressure >= 2.0` → `push_*` | **阈值** |
| `:803` | `threat > 0.0` | **符号** |
| `:814` | `bundle_hint = 2 if (… threat >= 2.0 …)` | **阈值** |
| `:848` | **`value_estimate = float(my_pressure - threat)`** | **进 PlanToken → 进网络** |

⇒ `value_estimate` 经 `PlanToken.to_vector()` 的 `np.tanh` 成为 **plan 58 维里的 1 维**
（`rl/plan_space.py:148`）。**这是"近似值直接进网络"的确证**：tanh 在原始 HP 量纲下会恒饱和
⇒ 折算常量不是可选项，是必需品。

替代物 = 已落地且有引擎对账 selftest 的外置工具① `threat_calc.estimate_tower_threat`
（`deepcopy` + 引擎自身确定性推演，"双方不再部署"语义）。
当前**零生产调用**（`docs/tool_usage_audit_2026-09-14.md` §1）。

---

## 1. 本阶段（P3-1）只做测量，**不改任何生产代码**

仪器：`scripts/probe_threat_approx.py`（**只读**）。
在真 ckpt 的真 rollout 上，逐帧同时记录：

- 近似：`threat, my_pressure = belief_planner._enemy_pressure(battle)`（**调生产函数本体**）；
- 精确：`E = estimate_tower_threat(battle, 0, H)["total"]`（敌方现存部队将对我方塔造成的伤害）
  与 `E_my = estimate_tower_threat(battle, 1, H)["total"]`，`H ∈ {5, 8, 20}` 秒。

### 1.1 被测量

| 编号 | 量 | 口径 |
|---|---|---|
| **M1** | 秩一致 | Spearman ρ(`threat`, `E`)、ρ(`my_pressure`, `E_my`)、ρ(`my_pressure−threat`, `E_my−E`) |
| **M2** | 二元闸门混淆矩阵 | 近似闸门用**生产阈值** `threat ≥ 2.0`（不重标定）；精确闸门阈值用**等阳性率分位点**（本实验自身数据标定，避免跨量纲照抄，见【R15】） |
| **M3** | 定性反转清单 | 前若干例「近似说没威胁/精确说有大威胁」与反向，附盘面摘要 |
| **M4** | 成本 | 单次调用 `median / p90` 毫秒（`H ∈ {5,8,20}`）+ `_enemy_pressure` 自身耗时 |

### 1.2 判据（**跑之前写死**）

**判据 P3-1（相对判据——不设绝对阈值，满足【R16】）**

拿近似闸门与两个**平凡基线**比 balanced accuracy（BA = (TPR+TNR)/2）：

- **B0**：恒判"无威胁"（阳性率 0）
- **B1**：恒判"有威胁"当且仅当场上存在敌方非塔存活实体（`_hostiles_present` 等价条件）

| 形式判决 | 条件 |
|---|---|
| **`APPROX_USELESS`（必须取缔）** | `BA(近似) ≤ max(BA(B0), BA(B1))` —— 近似闸门**不比"什么都不看"更有判别力** |
| **`APPROX_WEAK`（应当取缔）** | `BA(近似) > max(BA(B0), BA(B1))` 但 `BA(近似) < BA(B1) + 0.5·(1 − BA(B1))`（判别力**低于"平凡基线到完美"的一半路程**） |
| **`APPROX_OK`（只报告，不接线）** | 上式均不成立 |

**判据 P3-2（成本预算，预算判据不是统计判据）**

依据：纯训练实测 **26 步/s = 38.5 ms/帧**（≥2 处独立互证）。
**接入后单帧墙钟增幅必须 ≤ 10%（≈3.85 ms）**，否则必须廉价化
（视界自适应 / 威胁守卫 / 每 N 帧缓存），且廉价化后仍须满足本判据。
本阶段只测量不实现；若 `median` 已超预算，则廉价化方案是 P3-2 的**前置任务**。

**判据 P3-3（口径守恒，【R7】；本阶段只声明，P3-2 执行）**

接线必须**保持 `PRESSURE_THRESHOLD=2.0` 与 `threat ≥ 0.8·my_pressure` 的语义不变**
⇒ 需一个**单常量源**把 HP 折算成 pressure，并用 selftest 在同一批参考盘面上对账两套刻度的
阳性率（要求差 ≤ 10%）。**禁止**在接线时改动 `PRESSURE_THRESHOLD` 或那个 `0.8`。

### 1.3 判决分支（写死）

| P3-1 | P3-2 | 动作 |
|---|---|---|
| `APPROX_USELESS` / `APPROX_WEAK` | 超预算 | P3-2 = **先廉价化**（预注册新的成本判据）→ 再接线替换 `_enemy_pressure` |
| `APPROX_USELESS` / `APPROX_WEAK` | 在预算内 | P3-2 = 直接接线替换，`plan`/`belief_planner` 决策规则随之改用精确值 |
| `APPROX_OK` | 任意 | **只写报告，不接线**（并说明"取缔"买不到判别力增益，只买可解释性） |

---

## 2. 明确不做（【R11】）

- 不改 `rl/` 下任何文件（本阶段）；
- 不改 `PRESSURE_THRESHOLD` / `0.8` 比率 / `PLAN_DIM`（保持 58，避免【R6】的 `--fresh`）；
- 不因本测量就上训练时 MCTS / 扩参 / 改奖励；
- 不用单帧或单次观测标定任何阈值（M2 用等阳性率分位点，来自本实验的整批帧）。
