# -*- coding: utf-8 -*-
"""stdout / stderr 的 UTF-8 兜底 —— **单一实现**（Tier 1 · T1-1）。

**为什么需要这个模块**
----------------------
本项目 Windows 侧实测 `sys.stdout.encoding` 为 **gbk**，且 **`PYTHONIOENCODING=utf-8` 在本环境不生效**
（stdlib 自报 `stdout.encoding=gbk`，见 `docs/train_health_metrics_2026-09-18.md` §8）。
训练/工具的输出含中文与 emoji（如 GRU 活力告警的 ⚠️）⇒ **管道或重定向**时 `print` 抛
`UnicodeEncodeError`；更坏的是它常发生在 `except Exception` 处理器里（`{e!r}` 内嵌不可编码字符），
于是**兜底本身把进程崩掉**。两起实测事故：

- **2026-09-11** `test_solo_resume` 复现 ⇒ 崩训练（见 `rl/run_league.py` 该函数的 docstring）；
- **2026-09-18** `dashboard.py --solo <目录>` 在目录里还没有 `league_state.json` 时打告警 ⇒ **exit 1，服务起不来**。

**收敛前的事实（`scripts/_structure_check.py` ② 实测，不是转述）**
---------------------------------------------------------------
3 处 `def _force_utf8_stdout`：

| 位置 | 函数体 | 结论 |
|---|---|---|
| `rl/run_league.py:1538` | 两路 `(sys.stdout, sys.stderr)` + `errors="replace"` | **强版** |
| `rl/dashboard.py:3033` | **与上一行逐字相同**（仅 docstring 不同） | **强版** |
| `scripts/probe_value_ln.py:116` | 只 `sys.stdout`、**无** `errors="replace"` | **弱版** |

⇒ 核对器 ② 给出「**函数体 2 种**（含 docstring 3 种）」，并单独点名弱化实现。
本模块 = 那个**强版**的规范化落点。

**注意（口径）**：把 docstring 也算进指纹会得到「3 种互异」这个**误导性**结论 ——
这正是核对器 ② 必须**把 `body_digest` 与 `full_digest` 分开**的原因（T1-1 实操教训，已写成回归断言）。

**用法**：在**任何 `print` 之前**调用一次即可。**幂等**，重复调用无害。
"""
from __future__ import annotations

import sys
from typing import Iterable, Optional

__all__ = ["force_utf8_stdout"]

#: 默认处理的两路（顺序稳定，便于测试与日志）
DEFAULT_STREAMS_ATTR = ("stdout", "stderr")


def force_utf8_stdout(streams: Optional[Iterable] = None) -> int:
    """把 `stdout`/`stderr` 切到 UTF-8 + `errors="replace"`。

    参数
    ----
    streams : 可选。显式传入要处理的流（供测试注入假流）。缺省 = `(sys.stdout, sys.stderr)`。

    返回
    ----
    `int`：**成功 reconfigure 的流数**（0…2）。注意 —— 返回值**不是**失败信号：
    该函数是兜底路径，**任何异常都被吞掉**（不能因为兜底本身而崩）。

    设计约束（与收敛前的强版逐字等价）
    ----------------------------------
    - 两路都处理（弱版只处理 stdout ⇒ `stderr` 上的中文仍会崩）；
    - `errors="replace"`（弱版没设 ⇒ 遇到不可编码字符仍抛 `UnicodeEncodeError`）；
    - `try/except Exception: pass` 包住每一次调用（`reconfigure` 在非 `TextIOWrapper`
      上不存在，例如 pytest 的 capture 对象）。
    """
    ok = 0
    targets = (sys.stdout, sys.stderr) if streams is None else tuple(streams)
    for stream in targets:
        if stream is None:
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
            ok += 1
        except Exception:
            pass
    return ok
