# -*- coding: utf-8 -*-
"""按名运行 `rl/selftest.py` 里的**单个/指定**测试（配合【红线 R19】：不跑全量）。

背景：`rl/selftest.py::main()` 是无参全量跑，没有子集入口。本脚本从外部按名调用，
**不修改 selftest.py 一个字符**（于是全量路径的行为逐位不变）。

用法（必须在 src/clasher_new 下运行）：
    cd src/clasher_new && PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe \
        ../../scripts/run_selftests.py --list                 # 列出全部测试名
    cd src/clasher_new && PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe \
        ../../scripts/run_selftests.py test_precise_threat    # 只跑这一个（可给多个）
    cd src/clasher_new && PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe \
        ../../scripts/run_selftests.py --order-check          # 定义序 / dir() 序 / main() 序 差集

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
from rl.io_bootstrap import force_utf8_stdout  # noqa: E402  (T1-1: UTF-8 兜底单一来源)
force_utf8_stdout()


def _names(mod):
    return sorted(n for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n)))


def _order_report(st) -> int:
    """T1-3：报告三种顺序的差集（**定义序 / `dir()` 字母序 / `main()` 手工序**）。

    为什么需要：三者在收敛前**各不相同**，而它们各自被不同的东西依赖 ——
      - **`main()` 手工序** = 全量路径的契约（`scripts/_apply_s2_channel_when_idle.sh:70-71`
        等外部调用方按名调用；本仓惯例是"全量路径逐位不变"）；
      - **`dir()` 字母序** = `--list` 的输出顺序 ⇒ 用 `--list` 挑子集时，跑的先后与全量**不一致**；
      - **定义序** = 人读源码时的顺序。
    三者不等**不是 bug**（本就没要求一致），但**必须可见**：否则"我只跑了子集"与
    "我按全量顺序跑了子集"会被混为一谈。本报告只陈述事实，不裁决哪种更好。
    """
    import ast
    src_path = getattr(st, "__file__", None)
    if not src_path or not os.path.isfile(src_path):
        print("[order] 拿不到 selftest.py 路径")
        return 2
    src = open(src_path, encoding="utf-8", errors="replace").read()
    tree = ast.parse(src)
    defined = [n.name for n in tree.body
               if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]
    defined.sort(key=lambda n: next(x.lineno for x in tree.body
                                    if isinstance(x, ast.FunctionDef) and x.name == n))
    called = []
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name == "main":
            for sub in ast.walk(n):
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) \
                        and sub.func.id.startswith("test_"):
                    called.append(sub.func.id)
    listed = _names(st)

    def _diff(a, b, an, bn):
        only_a = [x for x in a if x not in set(b)]
        only_b = [x for x in b if x not in set(a)]
        print(f"[order] {an} vs {bn}：顺序相同 = {a == b}；"
              f"只在 {an} = {len(only_a)}，只在 {bn} = {len(only_b)}")
        if only_a:
            print(f"          只在 {an}：{only_a}")
        if only_b:
            print(f"          只在 {bn}：{only_b}")
        return only_a, only_b

    print(f"[order] 三个集合的大小：定义 {len(defined)} / main() 调用 {len(called)} / "
          f"dir() 列出 {len(listed)}")
    d1 = _diff(defined, called, "定义序", "main() 序")
    _diff(called, listed, "main() 序", "dir() 字母序")
    ok = not d1[0] and not d1[1]
    print(f"[order] ① 定义集合 == main() 调用集合 ⇒ {'OK' if ok else '**FAIL**'}"
          "（这一项必须成立：有测试没被登记 = 它从不跑）")
    print(f"[order] ② main() 顺序 == 定义序 ⇒ {called == defined}"
          "（本项**不要求**成立，本仓 `main()` 是手工排序）")
    print(f"[order] ③ dir() 字母序 == main() 顺序 ⇒ {listed == called}"
          "（本项**不要求**成立；但它不同意味着 `--list` 挑出来的子集跑的先后 ≠ 全量）")
    return 0 if ok else 1


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
    if args[0] == "--order-check":
        # T1-3：单独的开关，**不动 `--list` 的输出契约**（有外部脚本按行数/内容解析它）。
        return _order_report(st)

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
