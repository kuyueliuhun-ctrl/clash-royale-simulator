# -*- coding: utf-8 -*-
"""`rl/selftest.py` 的**共用底座**（Tier 2 · T2-8 拆分出的第一块）。

**为什么拆**：`rl/selftest.py` 原为 **6,023 行 / 111 个顶层定义**（100 个 `test_*`），是全仓最大的单文件。
拆分后：本文件 = **入口契约 + 共用导入 + 10 个 helper**；100 个测试按**定义序**切成 5 份到
`rl/selftests/part{1..5}.py`；`rl/selftest.py` 只剩 **聚合 + `main()`**。

**★★ 四条硬约束一个都没破（拆分前先审计过）**：
1. `scripts/run_selftests.py:36-42` 用 `dir()` 反射按名调用 `test_*` ⇒ 只要 100 个名字能作为
   `rl.selftest` 的属性被看到即可（本文件 `*` 导出 + 聚合模块显式导入 ⇒ 满足）；
2. `scripts/rl/selftest.py:5-7` 用 `runpy.run_path(<SRC>/rl/selftest.py, run_name="__main__")`
   ⇒ `main()` 与末尾 `if __name__` 必须留在 `rl/selftest.py`（满足）；
3. `main()` 那 **100 行手工调用清单**必须逐字不变（已 AST 对账）；
4. `scripts/_apply_s2_channel_when_idle.sh:70-71` 外部硬编码 4 个测试名 ⇒ 按名可达（满足）。

**为什么 `_PARENT` 必须留在这里**：它 = `os.path.dirname(os.path.dirname(abspath(__file__)))`，
而 `__file__` 是本文件（`rl/`）⇒ 值仍是 `src/clasher_new`。**测试搬到 `rl/selftests/` 后
自己算就会得到 `rl/`**（少一层）⇒ 两个原本自算 root 的测试（原 L5615/L5676）已改为直接用 `_PARENT`。
"""
import functools
import os
import shutil
import sys
import time
import random

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import numpy as np

from card_utils import Card



_INSTRUMENTED = False
#: T2-8：**聚合模块**（`rl/selftest.py`）把自己的 `globals()` 注册到这里 ——
#: 否则 `_instrument_tests()` 默认只看本模块（`selftest_common`）的 globals，
#: 那里**一个 `test_*` 都没有** ⇒ `discover_tests()` 会静默返回空。
_TEST_NS = None


def register_namespace(ns) -> None:
    """由聚合模块在导入期调用：把它的 `globals()` 交给本底座（见 `_TEST_NS` 注释）。"""
    global _TEST_NS
    _TEST_NS = ns




# ======================================================================================
# T1-3 / T1-5：测试运行的**可见性**（x/y 计数、逐测试耗时、显式 SKIP）
# --------------------------------------------------------------------------------------
# **硬约束（【R19】+ 并发会话）**：`main()` 里那 **100 行手工调用**的**顺序与内容不得改动** ——
# 它是全量路径"逐位不变"的保证，也是 `scripts/_apply_s2_channel_when_idle.sh:70-71`
# 等外部调用方所依赖的稳定接口。
# **做法**：在 `main()` **入口**把 `globals()` 里的每个 `test_*` 换成「计时 + 计数」包装。
# `main()` 的函数体在**调用时**才按名字从 globals 解析 ⇒ 那 100 行**逐字不用动**。
#
# 收敛到这里的三个既有问题（均由 T0 盘点实测）：
#   ① `main()` 无 x/y 计数、无逐测试耗时 ⇒ 长跑只能看"有没有最后那句 PASSED"；
#   ② 首个失败直接抛栈退出（本包装**不**改这一点，仍是 fail-fast）；
#   ③ 两个测试在缺 ckpt 时 `print("[SKIP] ...")` 或**静默整段跳过** ⇒ 换机/清 `runs/` 后
#      **照样全绿**（"假绿"）。现在一律走 `_mark_skip()` ⇒ 醒目标签 + 进汇总计数。
# ======================================================================================

_RUN_STATS: list = []   # [(name, elapsed_s, ok)]
_SKIPS: list = []       # [(name, reason)]
_ORIG_FUNCS: dict = {}  # name -> (原函数, 定义行号)


def _mark_skip(reason: str) -> None:
    """登记一次「跳过」（T1-5）。调用方必须是 `test_*`。

    为什么必须显式登记：这些分支原本只打一行 `[SKIP]` 或干脆静默 `if` 掉整段 ⇒
    在**缺 ckpt 的机器上测试照样全绿**。这条把"没测"与"测过且通过"分开。
    """
    import inspect
    name = "unknown"
    for fr in inspect.stack()[1:]:
        if fr.function.startswith("test_"):
            name = fr.function
            break
    _SKIPS.append((name, reason))
    print(f"[SKIP-NO-CKPT] {name}：{reason}")


def _instrument_tests(namespace=None) -> None:
    """把 `namespace`（缺省 = 本模块 `globals()`）里的 `test_*` 换成计时/计数包装（幂等）。

    ⚠️ **T2-8 拆分后必须显式传 namespace**：测试已搬到 `rl/selftests/part*.py`，
    而 `globals()` 在哪个模块里调用就指向哪个模块 ⇒ `rl/selftest.py::main()` 必须传
    它自己的 `globals()`，否则一个测试都包装不到（且**静默**）。
    """
    if namespace is None:
        namespace = globals()
    global _INSTRUMENTED
    if _INSTRUMENTED:
        return
    for name, fn in list(namespace.items()):
        if not (name.startswith("test_") and callable(fn)):
            continue
        _ORIG_FUNCS.setdefault(name, (fn, getattr(fn, "__code__", None) and fn.__code__.co_firstlineno or 0))
        namespace[name] = _make_wrapper(name, fn)
    _INSTRUMENTED = True


def _make_wrapper(name: str, fn):
    @functools.wraps(fn)          # 保留 __module__/__qualname__/__doc__：否则包装后所有测试都
    def _wrapped(*a, **kw):       # 显示成 `rl.selftest_common`，`discover_tests()` 的分片排序也就失效了

        t0 = time.perf_counter()
        try:
            r = fn(*a, **kw)
        except BaseException:
            _RUN_STATS.append((name, time.perf_counter() - t0, False))
            raise
        _RUN_STATS.append((name, time.perf_counter() - t0, True))
        return r
    _wrapped.__name__ = name
    _wrapped.__doc__ = getattr(fn, "__doc__", None)
    _wrapped._selftest_original = True  # type: ignore[attr-defined]
    return _wrapped


def discover_tests() -> list:
    """全量 `test_*`，按**定义序**（`co_firstlineno`）排列 —— 与 `main()` 的**手工序**无关。

    为什么要这个：`scripts/run_selftests.py --list` 是按 `dir()` 的**字母序**列出的，
    而 `main()` 是**手工序**，两者不同 ⇒ 子集跑的先后与全量不一致。本函数给出定义序，
    供 `--order-check` 做差集报告（`scripts/run_selftests.py`）。
    """
    _instrument_tests(_TEST_NS)
    # T2-8：测试已按定义序切到 `rl/selftests/part{1..5}.py` ⇒ **只按局部行号排会跨分片交错**
    # （part1 的第 30 行 vs part2 的第 30 行谁先？）。故按 `(模块名, 行号)` 排：
    # 模块名 `rl.selftests.part1` … `part5` 的字符串序恰好就是分片顺序。
    return [n for n, (fn, ln) in sorted(
        _ORIG_FUNCS.items(), key=lambda kv: (getattr(kv[1][0], "__module__", ""), kv[1][1]))]


def _report_tests() -> None:
    """汇总：x/y、跳过清单、最慢的若干项（T1-3/T1-5）。"""
    n = len(_RUN_STATS)
    ok = sum(1 for _, _, o in _RUN_STATS if o)
    print(f"\n[selftest] 共 {n} 个测试：{ok} 通过 / {n - ok} 失败；跳过 {len(_SKIPS)} 个")
    if _SKIPS:
        # 跳过必须**重复打一遍**：混在几千行输出里的一行 [SKIP-NO-CKPT] 是看不见的
        print(f"[selftest] ⚠️ 跳过清单（这些**没有**被验证，别当绿）:")
        for name, reason in _SKIPS:
            print(f"    - {name}：{reason}")
    slow = sorted(_RUN_STATS, key=lambda x: -x[1])[:5]
    if slow:
        print("[selftest] 最慢 5 项：" + ", ".join(f"{nm} {s:.1f}s" for nm, s, _ in slow))
    tot = sum(s for _, s, _ in _RUN_STATS)
    print(f"[selftest] 全部测试累计 {tot:.1f}s")


def _make_policy_and_tokens(env, seed=0):
    from rl.belief import BeliefInference
    from rl.plan_space import PlanToken, PLAN_DIM
    belief = BeliefInference(opp_deck=env.deck1, n_particles=128, seed=seed)
    tok = belief.encode(None, None)
    return belief, tok, PlanToken.zeros().to_vector(), PLAN_DIM


def _mk_env():
    from rl.env_wrapper import RLEnv
    return RLEnv(opponent=None, seed=0)


class _FakeCfg:
    """给 `_EvalScheduler` 用的最小 cfg 壳（只带它读的字段）。"""

    def __init__(self, d):
        self.__dict__.update(d)


def _intents():
    from rl.plan_space import MACRO_INTENTS
    return MACRO_INTENTS


def _tiny_rollout_transitions(pol, env, belief, tok, plan, n=4):
    """构造 n 条最小 transition（adv 正负交替、回报带 3.0 的离散度）。"""
    trans, hidden, obs = [], None, None
    obs, _ = env.reset()
    belief.reset(env.deck1)
    for i in range(n):
        ih = hidden
        bundle, lp, val, hidden, masks = pol.act(
            obs, tok, plan, env.get_action_mask, hidden=hidden, deterministic=False)
        obs2, r, term, trunc, info = env.step(bundle)
        sgn = 1.0 if i % 2 else -1.0
        trans.append({"obs": obs, "belief": tok, "plan": plan, "bundle": bundle,
                      "old_logprob": lp, "adv": sgn,
                      "returns": float(val) + 3.0 * sgn,
                      "masks": masks, "init_hidden": ih})
        belief.update(obs2, info.get("opp_played"))
        obs = obs2
    return trans
