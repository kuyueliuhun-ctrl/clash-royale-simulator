#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""打印某一组的任务简报（供子代理读取：文件清单 + 必须覆盖的符号清单）。

用法：
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/_survey_brief.py --group G002
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/_survey_brief.py --list
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from typing import Any, Dict, List

# T1-1b 补齐（2026-09-19）：原先这里是**手写**的 UTF-8 兜底块（只处理 stdout、且不处理 stderr
# ⇒ traceback 在 GBK 下仍是乱码）。现收敛到 T1-1 的**单一实现**（两路 + errors='replace'）。
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                                 "src", "clasher_new"))
from rl.io_bootstrap import force_utf8_stdout  # noqa: E402
force_utf8_stdout()

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SURVEY = os.path.join(ROOT, "docs", "_survey")


def load(name: str) -> Dict[str, Any]:
    with io.open(os.path.join(SURVEY, name), encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    groups = load("groups.json")
    inv = load("inventory.json")
    bypath = {f["path"]: f for f in inv["files"]}

    if args.list or not args.group:
        for g in groups["groups"]:
            print(f"{g['id']}\t{g['lines']}L\t{g['symbols']}sym\t{len(g['files'])}f\t{g['slug']}")
        return 0

    g = next((x for x in groups["groups"] if x["id"] == args.group), None)
    if g is None:
        print(f"ERROR: no such group {args.group}")
        return 2

    out = f"docs/_survey/parts/{g['id']}.md"
    print("=" * 78)
    print(f"组号: {g['id']}    输出文件: {out}")
    print(f"合计: {g['lines']} 行 / {g['symbols']} 个符号 / {len(g['files'])} 个文件")
    print("=" * 78)
    print()
    print("## 必须分析的文件（一个都不能跳过）")
    for p in g["files"]:
        f = bypath[p]
        doc = f.get("module_doc") or "无"
        print(f"- {p}  |  {f['lines']} 行 | {len(f['symbols'])} 符号 | 模块 docstring: {doc}")
    print()
    print("## 必须覆盖的符号清单（AST 抽取；行号 = 定义区间）")
    for p in g["files"]:
        f = bypath[p]
        print(f"\n### {p}")
        if not f["symbols"]:
            print("(无函数/类定义——只有模块级语句；请照实说明该文件内容)")
            continue
        for s in f["symbols"]:
            lo, hi = s["lineno"], s.get("end_lineno") or "?"
            sig = s.get("sig", {}).get("display") if s.get("sig") else None
            kind = s["kind"]
            extra = f" | {sig}" if sig else (f" | bases={s.get('bases')}" if s.get("bases") else "")
            print(f"- [{kind}] {s['qualname']} [L{lo}-{hi}]{extra}")
    print()
    print("## 覆盖对账")
    print(
        f"完成后 `docs/_survey/parts/{g['id']}.md` 里必须出现上述 **全部 {g['symbols']} 个符号**，"
        "且末尾有 `COVERAGE_OK <组号> files=<n> symbols=<n>` 行。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
