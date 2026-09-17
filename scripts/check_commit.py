# -*- coding: utf-8 -*-
"""长跑开跑前的宿主**提交内存**检查（【红线 R1】）。

背景：R1 原文要求跑前量 `wmic OS get FreeVirtualMemory`，但本机 `wmic` **已不可用**
（Windows 11 起被移除，实测返回空）⇒ 改用 `GlobalMemoryStatusEx` 的
`ullAvailPageFile`（= 还能提交多少）——正是 R1 关心的量（WinError 1455 就是提交耗尽）。

用法（Windows 侧解释器，任意 cwd）：
    .venv/Scripts/python.exe scripts/check_commit.py

判读：历史标定 = 可用提交 **20.6 GB** 时 `--eval-workers 12` 安全；**10.68 GB** 时同档位
出过 OOM/CUDA unknown ⇒ **< 12 GB 请降档或先清孤儿 spawn worker**。
"""
from __future__ import annotations

import ctypes
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")     # 仓库规矩：新脚本自带 UTF-8 stdout
except Exception:
    pass


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),   # 提交上限
                ("ullAvailPageFile", ctypes.c_ulonglong),   # 还能提交多少
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def main():
    m = MEMORYSTATUSEX()
    m.dwLength = ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
        print("[FAIL] GlobalMemoryStatusEx 调用失败")
        return 1
    gb = 1024 ** 3
    commit_limit = m.ullTotalPageFile / gb
    commit_avail = m.ullAvailPageFile / gb
    commit_used = commit_limit - commit_avail
    print(f"[commit] 上限 {commit_limit:6.2f} GB | 已用 {commit_used:6.2f} GB | "
          f"**可用 {commit_avail:6.2f} GB**")
    print(f"[物理]   总 {m.ullTotalPhys / gb:.2f} GB | 可用 {m.ullAvailPhys / gb:.2f} GB "
          f"| 负载 {m.dwMemoryLoad}%")
    if commit_avail < 12.0:
        print("[WARN] 可用提交 < 12 GB ⇒ 按【R1】降档（--eval-workers 8/4）或先清孤儿 spawn worker")
    else:
        print("[OK] >= 12 GB ⇒ --eval-workers 12 档位可用（R1 经验档）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
