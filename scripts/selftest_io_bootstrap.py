#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""`rl/io_bootstrap` 的回归测试（Tier 1 · T1-1，【R8】）。

**测什么**
1. **行为**：`force_utf8_stdout()` 必须
   - 同时处理 **stdout 与 stderr**（收敛前的弱化版只处理 stdout）；
   - 带上 **`errors="replace"`**（弱化版没有 ⇒ 遇不可编码字符仍抛 `UnicodeEncodeError`）；
   - **吞掉异常**（`reconfigure` 不存在的流、`None` 都不许把兜底本身搞崩）；
   - **幂等**（重复调用无害）；
   - 返回**成功 reconfigure 的流数**。
2. **单一来源**（本文件的第二半，也是真正防回归的那半）：全仓**不得**再出现第二份 ——
   甚至是**弱化版** —— UTF-8 兜底实现。判据直接复用 `scripts/_structure_check.py` 的 `check_utf8`，
   **不另写一套口径**（【R17】：同一事实不许两套实现）。

**为什么值得一个独立测试**：这条兜底一旦退化，症状是「**进程直接崩**」而不是「输出难看」——
两起实测事故：2026-09-11 `test_solo_resume` 崩训练、2026-09-18 `dashboard` 在缺 `league_state.json`
时 exit 1 起不来。且退化**不影响任何既有测试**（它们都在有编码的终端里跑），
靠人工评审是抓不到 `errors="replace"` 被删掉这种事的。

用法：
    python scripts/selftest_io_bootstrap.py     # 独立跑（与 selftest_offline_engagement_trade.py 同约定）
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "src" / "clasher_new"))
sys.path.insert(0, str(_HERE))  # 为了 import _structure_check

_FAILS: list = []


def ck(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail and not cond else ""))
    if not cond:
        _FAILS.append(name)


class _FakeStream:
    """记录 `reconfigure` 调用的假流。"""

    def __init__(self) -> None:
        self.calls: list = []

    def reconfigure(self, **kw):
        self.calls.append(kw)


class _BadStream:
    """`reconfigure` 直接抛（模拟 pytest capture 之类不支持 reconfigure 的对象）。"""

    def reconfigure(self, **kw):
        raise OSError("this stream has no reconfigure")


def main() -> int:
    from rl.io_bootstrap import force_utf8_stdout

    print("① 行为")

    f1, f2 = _FakeStream(), _FakeStream()
    n = force_utf8_stdout((f1, f2))
    expect = [{"encoding": "utf-8", "errors": "replace"}]
    ck("stdout 与 stderr 两路都被处理", len(f1.calls) == 1 and len(f2.calls) == 1)
    ck("参数恰为 encoding='utf-8' + errors='replace'（弱化版会缺 replace）",
       f1.calls == expect and f2.calls == expect, f"实得 {f1.calls}")
    ck("返回值为成功处理的流数 = 2", n == 2, f"实得 {n}")

    ck("不支持 reconfigure 的流 + None ⇒ 不抛且返回 0",
       force_utf8_stdout((_BadStream(), None)) == 0)

    f3 = _FakeStream()
    force_utf8_stdout((f3,))
    force_utf8_stdout((f3,))
    ck("幂等：重复调用不抛、每次都被处理", len(f3.calls) == 2, f"实得 {len(f3.calls)}")

    # 真实流：调用后 sys.stdout/stderr 必须自报 utf-8（本机默认 gbk ⇒ 这条有判别力）
    real = force_utf8_stdout()
    ck("真实调用返回 ≥1（本机 stdout 默认 gbk，应被切到 utf-8）", real >= 1, f"实得 {real}")
    ck("调用后 sys.stdout.encoding 为 utf-8",
       (getattr(sys.stdout, "encoding", "") or "").lower().replace("-", "") == "utf8",
       f"实得 {getattr(sys.stdout, 'encoding', None)}")
    # 反向验证：不可编码字符必须**打得出来**（这正是这个模块存在的理由）
    try:
        print("     反向验证：⚠️ 中文 + emoji 打印不抛 = True")
        ck("不可编码字符打印不抛", True)
    except UnicodeEncodeError as e:  # pragma: no cover
        ck("不可编码字符打印不抛", False, f"{e!r}")

    print("\n② 单一来源（防「第二份/弱化版」重新长出来）")
    import _structure_check as sc

    res = sc.check_utf8(_ROOT)
    ck("全仓恰有 1 处强实现（rl/io_bootstrap）", res["single_impl"] is True,
       f"defs={res['defs']} weak={res['weak']}")
    ck("无弱化实现（未管 stderr 或缺 errors='replace'）", res["weak"] == [], f"{res['weak']}")
    ck("别名已登记（dashboard / run_league / probe_value_ln ≥3 处）", res["aliases"] >= 3,
       f"实得 {res['aliases']}")

    print(f"\n结果：{'ALL PASS' if not _FAILS else 'FAILED: ' + ', '.join(_FAILS)}")
    return 0 if not _FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
