# -*- coding: utf-8 -*-
"""按名运行 `rl/selftest.py` 里的**单个/指定**测试（配合【红线 R19】：不跑全量）。

背景：`rl/selftest.py::main()` 是无参全量跑，没有子集入口。本脚本从外部按名调用，
**不修改 selftest.py 一个字符**（于是全量路径的行为逐位不变）。

用法（必须在 src/clasher_new 下运行）：
    cd src/clasher_new && PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe \
        ../../scripts/run_selftests.py --list                 # 列出全部测试名
    cd src/clasher_new && PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe \
        ../../scripts/run_selftests.py test_precise_threat    # 只跑这一个（可给多个）

⚠️ 口径（别误读子集结果）：
- 子集运行**只证明被选测试自身通过**，不构成全量绿；
- 测试之间可能有**文件系统依赖**（例如某些测试会生成 ckpt / run 目录供后续测试读）⇒
  若被选测试依赖别的测试的产物，请把那个测试一起选上；
- 全量顺序保证仍只有 `python rl/selftest.py` 能给。
"""
from __future__ import annotations

import os
import sys
import traceback

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src", "clasher_new")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _names(mod):
    return sorted(n for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n)))


def main():
    from rl.run_league import _force_utf8_stdout   # 与全量跑同一兜底（GBK 管道下不崩）
    _force_utf8_stdout()
    import rl.selftest as st

    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if args else 2
    if args[0] == "--list":
        for n in _names(st):
            print(n)
        return 0

    want = []
    for a in args:
        n = a if a.startswith("test_") else f"test_{a}"
        if not hasattr(st, n):
            print(f"[FAIL] 没有这个测试: {a}（用 --list 看可用名）")
            return 2
        want.append(n)

    n_pass = 0
    for n in want:
        try:
            getattr(st, n)()
        except Exception:
            print(f"[FAIL] {n}")
            traceback.print_exc()
            print(f"\n子集运行：{n_pass}/{len(want)} 通过，**在 {n} 处失败**")
            return 1
        n_pass += 1
    print(f"\n子集运行：{n_pass}/{len(want)} 全部通过（⚠️ 只代表被选测试，不等于全量绿）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
