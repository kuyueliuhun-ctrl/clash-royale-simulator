#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""签名/行号一致性抽检（防编造的第二道闸）。

覆盖对账（`_survey_verify.py`）只能证明「符号被提到了」，不能证明「描述是对的」。
本脚本把素材里每个符号条目的
  * 定义行号区间 `[L起始-结束]`
  * 类型（function / class / method / async_function）
  * 参数列表（签名里出现的参数名）
与 `docs/_survey/inventory.json` 的 AST 事实逐条对账，报出不一致。

用法：
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/_survey_audit.py \
        --out docs/_survey/audit_report.json
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
from typing import Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SURVEY = os.path.join(ROOT, "docs", "_survey")
PARTS = os.path.join(SURVEY, "parts")

F_RE = re.compile(r"^##\s+F:(.+?)\s*$")
S_RE = re.compile(r"^(#{3,5})\s+(?:S|M):([^\s\[]+)\s*(?:\[L(\d+)-(\d+)\])?\s*$")
SIG_RE = re.compile(r"^\s*-\s*签名:\s*`(.+?)`\s*$")
TYPE_RE = re.compile(r"^\s*-\s*类型:\s*(.+?)\s*$")


def norm_sig(s: str) -> str:
    s = re.sub(r"^\s*(async\s+)?def\s+", "", s.strip())
    s = s.strip("`").strip()
    s = re.sub(r"\s+", "", s)
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(SURVEY, "audit_report.json"))
    args = ap.parse_args()

    inv = json.load(io.open(os.path.join(SURVEY, "inventory.json"), encoding="utf-8"))
    ast_map: Dict[str, Dict[str, Dict]] = {}
    for f in inv["files"]:
        m = {s["qualname"]: s for s in f["symbols"]}
        for s in f["symbols"]:
            m.setdefault(s["qualname"].split(".")[-1], s)  # 短名不覆盖限定名（避免同名碰撞）
        ast_map[f["path"]] = m

    checked = 0
    not_found: List[str] = []
    range_bad: List[Tuple[str, str, str, str]] = []
    type_bad: List[Tuple[str, str, str]] = []
    param_bad: List[Tuple[str, str, str]] = []

    for name in sorted(os.listdir(PARTS)):
        if not name.endswith(".md"):
            continue
        path = os.path.join(PARTS, name)
        txt = io.open(path, encoding="utf-8", errors="replace").read()
        cur_file: str | None = None
        cur: Dict | None = None

        def flush(c: Dict | None):
            nonlocal checked
            if c is None or cur_file is None:
                return
            table = ast_map.get(cur_file, {})
            sym = table.get(c["name"]) or table.get(re.split(r"[.:/]+", c["name"])[-1])
            if sym is None:
                not_found.append(f"{name}:{cur_file}:{c['name']}")
                return
            checked += 1
            lo, hi = sym["lineno"], sym.get("end_lineno")
            if c["lo"] is not None and (int(c["lo"]) != lo or (hi and int(c["hi"]) != hi)):
                range_bad.append((f"{name}:{cur_file}:{c['name']}", f"L{c['lo']}-{c['hi']}", f"L{lo}-{hi}", ""))
            # 类型
            k = sym["kind"]
            is_method = "." in sym["qualname"]
            want = {
                "function": ("method", "staticmethod", "classmethod", "property", "function") if is_method else ("function",),
                "async_function": ("async_function", "async function", "coroutine", "method"),
                "class": ("class",),
                "method": ("method", "staticmethod", "classmethod", "property", "function"),
            }[k]
            if c.get("type") and not any(w in c["type"].lower() for w in want):
                type_bad.append((f"{name}:{cur_file}:{c['name']}", c["type"], k))
            # 参数名
            if k in ("function", "async_function", "method") and c.get("sig"):
                raw_sig = c["sig"]
                sig = norm_sig(raw_sig)
                params = [a["name"] for a in sym["sig"]["args"]]
                params += [a["name"] for a in sym["sig"]["kwonlyargs"]]
                if sym["sig"].get("vararg"):
                    params.append(sym["sig"]["vararg"])
                if sym["sig"].get("kwarg"):
                    params.append(sym["sig"]["kwarg"])
                missing = [p for p in params
                           if p != "self" and not re.search(r"(?<![\w.])" + re.escape(p) + r"\b", raw_sig)]
                if missing and not sig.startswith("<"):
                    param_bad.append((f"{name}:{cur_file}:{c['name']}", ",".join(missing), raw_sig[:70]))

        for ln in txt.splitlines():
            m = F_RE.match(ln)
            if m:
                flush(cur)
                cur = None
                cur_file = m.group(1).strip()
                continue
            m = S_RE.match(ln)
            if m:
                flush(cur)
                cur = {
                    "name": m.group(2).strip(),
                    "lo": m.group(3),
                    "hi": m.group(4),
                    "sig": None,
                    "type": None,
                }
                continue
            if cur is not None:
                ms = SIG_RE.match(ln)
                if ms and cur["sig"] is None:
                    cur["sig"] = ms.group(1)
                    continue
                mt = TYPE_RE.match(ln)
                if mt and cur["type"] is None:
                    cur["type"] = mt.group(1)
        flush(cur)

    rep = {
        "checked": checked,
        "not_found_in_ast": not_found,
        "range_mismatch": [list(x[:3]) for x in range_bad],
        "type_mismatch": [list(x) for x in type_bad],
        "param_name_missing": [list(x) for x in param_bad],
    }
    json.dump(rep, io.open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[audit] 抽检符号 = {checked}")
    print(f"[audit] 行号不符   = {len(range_bad)}")
    print(f"[audit] 类型不符   = {len(type_bad)}")
    print(f"[audit] 参数名缺失 = {len(param_bad)}")
    print(f"[audit] AST 中查无此符号 = {len(not_found)}")
    for label, rows in (("行号", range_bad), ("类型", type_bad), ("参数", param_bad)):
        for r in rows[:8]:
            print(f"   [{label}] {r[0]}  素材={r[1]}  AST={r[2]}")
    if not_found:
        for x in not_found[:10]:
            print(f"   [查无] {x}")
    print(f"[audit] wrote {os.path.relpath(args.out, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
