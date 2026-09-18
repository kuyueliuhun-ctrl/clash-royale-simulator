# 修复：掩码 `used` 集合的 0/1-based off-by-one（B 类「非法卡包」根因）

日期：2026-09-18 ｜ 阶段：**根因定位 + 修复 + 守卫 + 回归 + 端到端验证**（全部已完成）
触发：`docs/elixir_saving_audit_2026-09-18.md` §4 查出的 **B 类病理**（模型提交非法卡包 → 整包拒绝 → 白掉一帧）
台账：**O9**（本条把它从「成因未定」升级为「已定位已修」）

---

## 0. 一句话

`RLEnv.get_action_mask_for` 里 `used` 存的是 **0-based 槽位下标**（`sa.slot - 1`），
而成员测试写成了 **1-based** 的 `sa.slot not in used` ⇒ partial bundle 一旦出现「**(s, s−1) 相邻降序对**」，
低槽位 `s−1` 会被**整段跳过**：既漏记 `used`（掩码放行**已用槽位** ⇒ 采样器照掩码行事 ⇒ 提交**重复槽位**的包
⇒ `validate_bundle` **整包拒绝** ⇒ 白掉一帧 + 吃 `invalid_penalty`），又漏扣那张卡的费用。
**一行修复** + **两条零成本不变式** + **一个回归测试**，并已端到端验证 **B 类 = 0 帧**。

## 1. 根因（源码级，可复算）

修复前 `src/clasher_new/rl/env_wrapper.py:498`：

```python
used = set()
for sa in partial_bundle.sub_actions:
    ...
    elif sa.slot >= 1 and sa.slot <= K_MAX and sa.slot not in used:   # ← sa.slot 是 1-based
        ...
        used.add(sa.slot - 1)                                        # ← 存进去的是 0-based
```

`used` 随后被两处按 **0-based** 消费（`slot_mask(..., used_slots=used)` 与 `for i in used: cells[i] = False`），
**只有这一处成员测试是 1-based**。后果分两支：

| 输入 partial | `used`（修复前） | 后果 |
|---|---|---|
| 槽位 `(2, 1)` | `{1}`（缺 `0`） | 槽位 1 **仍被掩码判合法** ⇒ 可被再选一次 ⇒ **重复槽位** |
| 槽位 `(3, 2)`, `(4, 3)`, `(3, 2, 1)` … | 缺低槽位 | 同上 |
| 任意含降序对的组合 | 圣水**少扣**该卡费用 | 掩码的"买得起"判断与 `validate_bundle` 口径不一致（可松可紧） |

**信号指纹**：输入 `(2,1,1,1)` 的 4 张包正是录像里那条连续 225 帧的形态。

## 2. 证据链（三组，同种子对照）

### 2.1 掩码层（**强制 `elixir=10`** 以排除"买不起"的掩盖）

| 实现 | 有序组合 | 「已用槽位仍被判合法」 |
|---|---|---|
| 修复前（提交 `792736a`） | 256 组 | **44 组（17.2%）** |
| 修复后（工作树） | 256 组 | **0 组（0.0%）** |

### 2.2 rollout 层（确定性采样，与评估路径同；10 局 × ≤200 帧）

| 实现 | 帧数 | `validate_bundle` 拒绝帧 | 掩码不变式命中 |
|---|---|---|---|
| 修复前（**先只加守卫·2**时直接量拒绝帧） | 1707 | **474（27.8%）** | 0（守卫·2 抓不到这类） |
| 修复前（**加上守卫·1**后） | 1508 | —（在提交前就被拦住） | **344（22.8%）** |
| 修复后 | 1609 | **0** | **0** |

> ⚠️ **缺陷率为何比录像里的 0.7% 高**：本 A/B 只有 10 局，而该病理是**按局聚集**的
> （真实 280 局里 39 局命中；局内一旦开始就因"花不掉钱 ⇒ 圣水涨 ⇒ 更多降序对"形成正反馈而持续）。
> 两个数不矛盾：**0.7% 是 280 局混合后的行为率**，**17%~28% 是 10 局小样本里的掩码/不变式命中率**。

### 2.3 真实 100k 录像里的**后果**（修复前跑的，作为"病例"）

`docs/elixir_saving_audit_2026-09-18.md`：A_et **497 / 75,891 帧（0.7%）**、**39 / 280 局**、
最长**连续 225 帧 ≈ 112.5 s**（单局 180 s）；非法性 **100% 可自证**（重复槽位 490 + 总费>决策圣水 7）。

### 2.4 端到端闭环（**用当初发现该 bug 的同一台仪器**）

用修复后的代码按评估路径跑 20 局（`seed_base=8000`，CPU，确定性）并落**真录像**，
再喂 `scripts/pass_streak_audit.py`：

```
step    帧数   A主动不出%   B整包被拒%   C实际出牌%   A∩≥6  B∩≥6  B段最长
8000    3413      92.8%       0.0%        7.2%        0     0      0
[FIXED] 帧 3413 ｜ A 3167 (92.8%) ｜ **B 整包被拒 0 (0.0%)** ｜ C 246 (7.2%)
[FIXED] validate 拒绝帧 = 0（仪器自检：A 类却花掉圣水 = 0）
```

原始输出：`audit_fixed_replays.log`；产录像脚本 `scripts/_verify_fix_replays.py`（约 2.5 min 可复跑）。
⚠️ 该 20 局与原始录像**不是同一批局**（并行评估 worker 的初始牌序取决于父进程 env 的当前牌序，不可知）
⇒ 这条是"**修复后不再发生**"的验证；**同种子的前后对照**由 §2.1/§2.2 提供。

## 3. 修复内容

| # | 文件 | 改动 |
|---|---|---|
| ① | `src/clasher_new/rl/env_wrapper.py:498` | `sa.slot not in used` → **`(sa.slot - 1) not in used`**（+ 30 行根因注释） |
| ② | `src/clasher_new/rl/follower.py::act` | **掩码不变式·1**：partial 里已用掉的槽位必须已被掩码置非法 ⇒ 否则 `RuntimeError`（零成本，掩码本来就要取）。**这类错误只能由它或整包校验抓到** |
| ③ | `src/clasher_new/rl/follower.py::act` | **掩码不变式·2**：被置 `-1e9` 的槽位不得被选中（抓"采样器违规"，补不上①那一类） |
| ④ | `src/clasher_new/rl/follower.py::act_parallel` | **·1 的批路径版本 + ·2 的向量化版本**（两条都加，`test_parallel_batch_equivalence` 已过 ⇒ 批路径未被改坏） |
| ⑤ | `src/clasher_new/rl/selftest.py` | 新增 **`test_mask_partial_bundle_invariants`** 并注册进全量 `main()` |

**不变式·1 的实测效力**：修复前实现上它以 `RuntimeError` 当场拦住
（例：`mask['slots']=[1,0,0,0]`, `used_slots=[1]`, partial 含 slot 1），修复后 **0 次命中**（无假阳性）。

## 4. 回归测试（`test_mask_partial_bundle_invariants`）

四条不变式（**跨局边界**，【R8】）：

1. 枚举全部 1/2/3 元**有序**槽位组合（含 `(2,1)`/`(3,2)`/`(4,3)` 等降序对）：
   出现过的槽位在 `mask["slots"]` 必须为 **False**；
2. `mask["used_slots"]` 与 partial 的槽位集合**逐位一致**（0-based）；
3. 掩码里仍合法的槽位 ⇔ 卡费 ≤ 剩余圣水（与 `validate_bundle` **同口径**，不重复扣费）；
4. 采样器产出的**每个** bundle 都通过 `validate_bundle`，且每一步选中的槽位在该步掩码里合法
   （跨 2 局 × sample/argmax 两分支）。

**判别力已证明**：把 `RLEnv` 换成修复前副本运行该测试 ⇒ **失败**并报
`局0 partial=(2, 1): used_slots=[1] != [0, 1]`；用当前工作树 ⇒ **通过**（6/6 子集全绿）。

## 5. 红线合规

| 红线 | 本条怎么做 |
|---|---|
| **R2 训练语义兼容** | 不改任何默认开关、不改架构、不改奖励；**修复的是掩码自身的错误**（原本就与 `validate_bundle` 矛盾）。旧 ckpt 可直接加载（无架构变更、无需 `--fresh`） |
| **R8 回归 + 跨局边界** | 新增测试覆盖 §4，**跨 2 局**；旧的 **录像/PPO 重放**不受影响（PPO 重放用 rollout 时记录的 `masks`，不重算；`evaluate()` 语义未动） |
| **R13 改判定逻辑必须位图对账** | ① 规范工具 `scripts/_mask_diff_snapshot.py` 快照的是 **`legal_cells`**，本修复**不触碰**该函数（只改 partial bundle 的记账）⇒ 其输出不受影响；② 被改动的**那一面**用**更强的逐位对账**代替：`scripts/_mask_ab_prefix.py` 在**同种子**下枚举 256 组有序组合 × 2 实现，逐组比 `slots`/`used_slots`（差异只出现在含降序对的组合，17.2% → 0%） |
| **R19 子集 selftest** | 子集 1（6/6）：`test_mask_partial_bundle_invariants`、`test_mask_validate_invariant_both_sides`、`test_action_bundle_same_tick`、`test_action_bundle_ability`、`test_bundle_cap_no_crash`、`test_no_solo_commit_without_lead`；子集 2（3/3）：`test_eval_solo_parallel`、`test_league_replays`、`test_eval_round_robin_parallel_equivalence`；**子集 3（5/5，热路径安全）**：`test_mask_partial_bundle_invariants`、`test_parallel_batch_equivalence`、`test_solo_mode_smoke`、`test_solo_resume`、`test_mp_training_loop`——后四条证明**新增的两条不变式在真实 solo/mp 训练循环与批路径里不会误报** |
| **R18 文档同步** | 本文 + `AGENTS.md` §B 一行 + `docs/README.md` + `docs/agents/ledger.md` **O9** 更新 + `docs/elixir_saving_audit_2026-09-18.md` §6 与判读文档补记各追加一行 |

## 6. 影响（修复前 → 修复后）

- **不再白掉帧**：A_et 每帧白掉率从 **0.7% → 0**（且这 0.7% 每帧还吃 `invalid_penalty = 0.05`，
  最长那条连续 225 帧累计 **≈ −11.25**，比胜利奖励 +10 还大）。
- **统计口径被净化**：`docs/readout_et_solo100k.md` 层 2 的「全帧圣水≥6 = 0.52%」有 **96% 来自 B 类**
  （其圣水是"花不掉"被动涨上去的），且**5 次 Xbow 全发生在 B 类解锁帧** ⇒ 这两列在**旧 run 里仍是那样读**
  （历史不改），但**新 run 不会再被这个病理污染**。
- **★ 但"攒费"问题没解决**：本修复与攒费**无关**。修复后同一台仪器在新录像上仍读出
  **`A ∩ 圣水≥6 = 0 帧`**、A 段峰值天花板仍是 **4.00**（= 手牌最便宜那张的费）
  ⇒ 【确证 C14】「**放弃一张买得起的牌**发生 0 次」**不受本修复影响**（它本来就是另一个病灶）。
- **遗留（未修，另案）**：`mask["slots"][i]` 合法但 `cells` 整行为空时，`act()` 的 `argmax` 会落在
  全 `-1e9` 的行上并返回 cell 0（位置非法 ⇒ 整包拒）。本次 80 帧回归 + 3413 帧验证里**均未出现**
  （回归测试会在出现时打印 `cell_gap` 计数，不掩盖）⇒ 记作**观察项**，需要时再单独立项。

## 7. 复现

```bash
# ① 同种子前后对照（约 5 min；脚本按 git show 792736a:src/clasher_new/rl/env_wrapper.py 现取修复前源码）
cd src/clasher_new && PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe \
  ../../scripts/_mask_ab_prefix.py --games 10 --max-steps 200

# ② 回归测试（【R19】子集）
cd src/clasher_new && PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe \
  ../../scripts/run_selftests.py test_mask_partial_bundle_invariants \
  test_mask_validate_invariant_both_sides test_bundle_cap_no_crash

# ③ 端到端：修复后产录像 → 同一台仪器审计（期望 B = 0）
cd src/clasher_new && PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe \
  ../../scripts/_verify_fix_replays.py --ckpt runs/et_solo100k/solo_main_8000.pt \
  --n-total 20 --out docs/_fix_verify_replays
cd src/clasher_new && PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe \
  ../../scripts/pass_streak_audit.py --replays FIXED=../../docs/_fix_verify_replays
```

留证目录：`docs/mask_used_slot_offbyone_fix_2026-09-18/`
（`ab_1_invalid_bundles.log`、`ab_2_with_invariant.log`、`verify_fixed_replays.log`、`audit_fixed_replays.log`、
`audit_fixed_replays.json`、`selftest_subset1.log`、`selftest_subset3.log`）。
⚠️ **不在树里放「修复前源码副本」**：`scripts/_mask_ab_prefix.py` 按**提交哈希**现取
（`git show 792736a:src/clasher_new/rl/env_wrapper.py`）⇒ 既可复算、又没有第二份真源
（若该提交不可达，用 `--prefix-py <副本>` 显式给）。
⚠️ ③ 产出的 `docs/_fix_verify_replays/league_8000.pkl`（4.2 MB）**未入库**（`.gitignore` 排 `*.pkl`，且可 2.5 min 复跑）；
其**读数**已留在 `audit_fixed_replays.log` / `.json`。
