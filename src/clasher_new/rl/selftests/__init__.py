# -*- coding: utf-8 -*-
"""`rl/selftest.py` 的 100 个测试按**定义序**切成的 5 部分（Tier 2 · T2-8）。

它们**不是**独立可跑的模块：共用底座在 `rl.selftest_common`，聚合入口仍是 `rl.selftest`：
`python rl/selftest.py`（全量）/ `scripts/run_selftests.py test_<名>`（子集，【R19】）。
⚠️ 每个 part 顶部**显式**导入它用到的私有 helper（`from ... import *` 不导出下划线名）。
"""
