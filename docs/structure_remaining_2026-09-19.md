# 结构优化 · 剩余项逐条判定（2026-09-19）

> **为什么单独一份**：Tier 0/1/2（除下列项）与 Tier 3 的 T3-2 Stage A 都已落地并验证。
> 剩下的 6 项**全部被「方案自己的 §7」或「既有契约」挡住** ⇒ 逐条给证据与判定，**不含糊**。
> **要强行做其中任何一项，必须先推翻下面对应的证据**（不是"再想想"）。

## 0. 已完成的（索引）

| 层 | 项 | 分支 |
|---|---|---|
| Tier 0 | 死件清理 31 / 只读核对器 / `scripts_inventory.md` / `card_utils` 报错 | `structure-tier0` |
| Tier 1 | UTF-8 兜底收敛（含 T1-1b 收尾，手写块 **0 残留**）/ 绝对路径 9→0 / selftest 计数 / tempfile / 假绿显式化 | `structure-tier1`、`structure-t1-1b-complete` |
| Tier 2 | T2-1/T2-2 dashboard 拆包（3,171→566）/ T2-3 判定语义半簇 / T2-5 `card_mechanics` M8 / **T2-8 `selftest.py` 6,023→172（全量 100 通过）** / T2-6 判定不做 + 自动索引 | `structure-tier2-dashboard`、`structure-tier2-batch2`、`structure-t2-8`、`structure-t2-6-scripts-index` |
| Tier 3 | T3-2 Stage A **取消 cwd 契约**（Stage B 判定不做） | `structure-tier3-cwd` |

---

## 1. T2-3 评估并行半簇 —— **被既有测试契约挡住**

**方案原写**：`run_league.py` L581-759（`_eval_pair_worker_main` / `_run_eval_pairs_parallel` / `_round_estimates` / `eval_round_robin`）→ `rl/league_eval.py`。

**挡住它的是一条 test**（拆分后位置）：

```
src/clasher_new/rl/selftests/part5.py:1123:
    assert run_league._eval_pair_worker_main.__module__ == "rl.run_league", ...
```

即本仓**显式把「该 worker 必须留在 `rl.run_league`」写成了契约**。原因也是硬的：它是 `mp.get_context("spawn")` 的 `Process(target=...)`
入口，**spawn 的 pickle 按 `__module__` + `__qualname__` 在子进程里重新导入** ⇒ 换模块就是换了一条被验证过的启动路径。

**判定**：**不做**。要做得先**改这条断言**（削弱一个既有契约）来换一个纯搬家收益 —— 不对称。
> 附带证据（独立支持同一结论）：`_run_eval_pairs_parallel` 读 `_ET_MEASURE`，而方案 §7.1 要求 `_ET_MEASURE`/`_set_et_measure` **留在原地**（【R7】接线点）⇒ 搬它要**改签名**，那就不是"纯搬运"了。

---

## 2. T2-4 `train_solo` 抽函数 —— **【R2】碰不得 + 无对账手段**

方案 §7.1 把 `train_solo.py` 的这些列为**碰不得**：
`DEFAULT_SOLO_DECK`(L56-57) / 锚点 / `_SOLO_PROPHET_PROB`(L67-69) / `_OPP_MIX` / `_HIST_POOL_MAX` / `_PFSP_*`，
以及「**`run_solo` 的调用顺序与 RNG 调用次序**」。

**判定**：**不做**。三条理由：
1. 方案 §7.1【R2】直列碰不得（新能力只经 `TrainConfig` 显式开、默认路径逐位回旧）；
2. **没有逐位对账手段**：本仓实测**同 seed 同配置重跑在 CUDA 上单点差 0.60**（预注册 §11.13.11）
   ⇒ 「搬完行为逐位相同」**在本环境无法证明**，只能靠"导入等价 + 子集 selftest"；
3. 收益（1,748 行文件的函数化）远小于一个**训练语义**回归的风险。

---

## 3. T2-7 抽奖励簇 → `rl/reward.py` —— **【R2】【R7】+ 并发期禁止**

**依赖面实测**：`compute_reward` / `DEFAULT_REWARD` 被 **8 个文件**引用 ——
`action_mask.py`、`config.py`、`env_wrapper.py`、`flow_league.py`、`mcts.py`、`phi_offline_check.py`、
`probe_reward_composition.py`（+ `_survey_merge.py` 的散文）。

**挡住它的**：方案 §7.2 **#8「并发期间禁止动 `rl/config.py` / `rl/engagement.py` / `battle.py` …」**
（并发会话在做奖励机制对比），叠加 §7.1【R2】（`DEFAULT_REWARD` 默认值）与【R7】（单常量源）。

**并发状态实测**：tracked 改动 **0**；对方最后写入 **01:02**，现在 **02:59**（≈2 h 无写入）。
⇒ **"看起来停了"但无法证明"已收尾"**；而这一项对奖励路径，代价不对称。

**判定**：**不做**（保持"并发期禁止"；若确认对方收尾，可作为独立一项另行预注册后做）。

---

## 4. T3-1 拆 `battle.py` —— **方案 §7.2 #3 明确「拒绝做」**

方案原文理由（逐字）：**「【R13】语义面太宽：`_mask_diff_snapshot.py` 只覆盖 128 张代表性快照，
拆 `BattleState` 无法用位图对账证明全局等价。**收益（可读性）远小于风险（掩码/部署语义漂移）。**」**

**判定**：**不做**。且方案在 §6 的 T3-1 里也写了「**并发期间明确禁止**」（对方在 `battle.py` 上有 `root_cast` 通道）。

---

## 5. T3-3 迁 `tests/` + pytest —— **方案 §7.2 #2 明确「拒绝做，默认不做」**

方案原文理由（逐字）：**「触及【R19】+ 2 个外部硬编码调用方（`run_selftests.py:36-42`、
`_apply_s2_channel_when_idle.sh:70-71`）；「`main()` 手工清单与全量路径逐位不变」的保证会丢。
⇒ Tier 3，需用户拍板；**默认不做**。」**

**判定**：**不做**（方案默认立场）。
> ⚠️ 而且 **T2-8 已经把它的主要动机解决了**：`rl/selftest.py` 从 **6,023 行**降到 **172 行**，
> 100 个测试按定义序切成 5 片（全量套件仍 100 通过）⇒ 「一个巨型测试文件」的问题**已消失**，
> 再引入 pytest 只剩"换框架"的成本（丢【R19】契约 + 改 2 个调用方）。

---

## 6. T3-4 统一 `sys.path` 引导 —— **未拒绝，但价值已被 T3-2 Stage A 消掉大半**

**面实测**：**124 个文件**自带 `sys.path` 引导。

**★ 关键：「重复」是假的 —— 那 35 处 `rl/*.py` 的 `_PARENT` 引导服务于另一个用途**：
`rl/__init__.py` **已经**做了 `_PARENT` 插入（供 `import rl.x` 用），但**当 `python rl/dashboard.py` 这样直接跑时，
`rl/__init__.py` 根本不会被执行** ⇒ 每个可当脚本运行的模块**必须自带**引导。
⇒ 这不是「同一件事抄了 35 遍」，而是**「包导入路径」与「脚本入口路径」两条不同的路**；
把它们"统一"掉会**打破脚本入口**（这正是 T2-8 里我实测撞到的那条：聚合文件缺引导 ⇒ 全量套件起不来）。

**判定**：**不做**。剩余可做的只有「把 3 种文本变体（`sys.path.insert(0, _PARENT)` 的空格差异）统一成一种写法」，
那是**纯排版**，收益接近 0，而改动面 124 个文件。
> T3-2 Stage A 已经取消了**真正咬人**的那部分 cwd 约束（数据文件不再依赖 cwd）。

---

## 7. 一句话总表

| 项 | 判定 | 挡它的是什么 |
|---|---|---|
| T2-3 评估并行半簇 | ❌ 不做 | **既有 test 契约**（worker `__module__` 断言）+ `_ET_MEASURE` 留原地 |
| T2-4 `train_solo` 抽函数 | ❌ 不做 | §7.1【R2】碰不得 + **无逐位对账**（CUDA 同配置单点差 0.60） |
| T2-7 抽奖励簇 | ❌ 不做 | §7.1【R2】【R7】+ §7.2 #8 **并发期禁止**（8 文件依赖面） |
| T3-1 拆 `battle.py` | ❌ 不做 | §7.2 **#3 拒绝做**（【R13】位图对账证不了全局） |
| T3-3 迁 `tests/`+pytest | ❌ 不做 | §7.2 **#2 拒绝做**（【R19】+ 2 个硬编码调用方）；**T2-8 已解决其动机** |
| T3-4 统一 `sys.path` | ❌ 不做 | 124 文件；且「重复」实为**两个不同用途**，统一会打破脚本入口 |
