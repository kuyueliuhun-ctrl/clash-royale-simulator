#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 inventory.json 的文件切成「子代理任务组」，保证**全量覆盖、无文件遗漏**。

规则：
  * 单文件 >= BIG_LINES 行 ⇒ 独立成组（大文件信息密度高，必须独占一个子代理）；
  * 其余按**同目录**聚合，贪心装箱到 ~CAP_LINES 行 / 最多 MAX_FILES 个文件；
  * 输出 groups.json（含每组的文件清单 + 行数 + 符号数），供 orchestrate 任务与
    「覆盖对账」脚本共用。

用法：
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/_survey_groups.py \
        --inventory docs/_survey/inventory.json --out docs/_survey/groups.json
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from collections import OrderedDict
from typing import Any, Dict, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BIG_LINES = 400
CAP_LINES = 1000
MAX_FILES = 12


def slug(path: str) -> str:
    s = path.replace("src/clasher_new/", "").replace("scripts/", "scripts-")
    s = s[:-3] if s.endswith(".py") else s
    s = s.replace("/", "-").replace("_", "-").replace(".", "-")
    return s.strip("-").lower()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with io.open(args.inventory, encoding="utf-8") as fh:
        inv = json.load(fh)

    files: List[Dict[str, Any]] = inv["files"]
    groups: List[Dict[str, Any]] = []

    big = [f for f in files if f["lines"] >= BIG_LINES]
    small = [f for f in files if f["lines"] < BIG_LINES]

    for f in sorted(big, key=lambda r: -r["lines"]):
        groups.append(
            {
                "id": f"G{len(groups) + 1:03d}",
                "kind": "single",
                "files": [f["path"]],
                "lines": f["lines"],
                "symbols": len(f["symbols"]),
                "slug": slug(f["path"]),
            }
        )

    # 按目录聚合小文件
    bydir: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    for f in sorted(small, key=lambda r: r["path"]):
        bydir.setdefault(os.path.dirname(f["path"]), []).append(f)

    for d, fs in bydir.items():
        fs = sorted(fs, key=lambda r: -r["lines"])
        cur: List[Dict[str, Any]] = []
        cur_lines = 0
        for f in fs:
            if cur and (cur_lines + f["lines"] > CAP_LINES or len(cur) >= MAX_FILES):
                groups.append(
                    {
                        "id": f"G{len(groups) + 1:03d}",
                        "kind": "group",
                        "dir": d,
                        "files": [x["path"] for x in cur],
                        "lines": cur_lines,
                        "symbols": sum(len(x["symbols"]) for x in cur),
                        "slug": slug(cur[0]["path"]) + f"-x{len(cur)}",
                    }
                )
                cur, cur_lines = [], 0
            cur.append(f)
            cur_lines += f["lines"]
        if cur:
            groups.append(
                {
                    "id": f"G{len(groups) + 1:03d}",
                    "kind": "group",
                    "dir": d,
                    "files": [x["path"] for x in cur],
                    "lines": cur_lines,
                    "symbols": sum(len(x["symbols"]) for x in cur),
                    "slug": slug(cur[0]["path"]) + f"-x{len(cur)}",
                }
            )

    covered = sorted(p for g in groups for p in g["files"])
    allfiles = sorted(f["path"] for f in files)
    assert covered == allfiles, (
        f"COVERAGE BUG: missing={set(allfiles) - set(covered)} extra={set(covered) - set(allfiles)}"
    )

    payload = {
        "big_lines": BIG_LINES,
        "cap_lines": CAP_LINES,
        "max_files": MAX_FILES,
        "n_files": len(allfiles),
        "n_groups": len(groups),
        "groups": groups,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with io.open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    print(f"[groups] {len(allfiles)} files -> {len(groups)} groups; coverage exact = True")
    for g in groups:
        print(f"  {g['id']:>5} {g['lines']:>5}L {g['symbols']:>4}sym  {len(g['files'])}f  {g['slug']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
