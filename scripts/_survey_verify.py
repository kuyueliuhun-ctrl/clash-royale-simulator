#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""覆盖对账（防漏文件 / 防漏符号 / 防凭空造符号）。

对每个组：
  1. `## F:<path>` 段落必须与 groups.json 的文件清单**逐一对应**；
  2. inventory.json 里该组的每个符号，必须在 part 文件里以
     `### S:<qualname>` 或 `#### M:<qualname>`（或 `#####`）出现；
  3. part 里出现的 `S:`/`M:` 标题若不在 inventory 中，记录为「未在 AST 中的符号」
     （用于发现子代理编造符号名）。

用法：
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/_survey_verify.py \
        --out docs/_survey/coverage_report.json
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
from typing import Dict, List, Set

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

F_RE = re.compile(r"^##\s+F:(.+?)\s*$", re.M)
S_RE = re.compile(r"^#{3,5}\s+(?:S|M):([^\s\[]+)", re.M)


def load(name: str) -> Dict:
    with io.open(os.path.join(SURVEY, name), encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(SURVEY, "coverage_report.json"))
    args = ap.parse_args()

    groups = load("groups.json")
    inv = load("inventory.json")
    symbols_by_file: Dict[str, Set[str]] = {
        f["path"]: {s["qualname"] for s in f["symbols"]} for f in inv["files"]
    }

    report = {"groups": [], "ok": True}
    n_file_ok = n_file_missing = 0
    n_sym_ok = n_sym_missing = 0
    all_extra_syms: List[str] = []

    for g in groups["groups"]:
        part = os.path.join(SURVEY, "parts", f"{g['id']}.md")
        rec = {"id": g["id"], "part": os.path.relpath(part, ROOT), "exists": os.path.exists(part)}
        if not rec["exists"]:
            rec["ok"] = False
            report["ok"] = False
            report["groups"].append(rec)
            n_file_missing += len(g["files"])
            continue
        with io.open(part, encoding="utf-8", errors="replace") as fh:
            txt = fh.read()

        found_files = set(m.strip() for m in F_RE.findall(txt))
        want_files = set(g["files"])
        missing_files = sorted(want_files - found_files)
        unknown_files = sorted(found_files - want_files)

        found_syms = set(S_RE.findall(txt))
        # 子代理可能写成 `模块.类.方法` / `文件::符号` / `a/b.py::main`；统一取末段短名兜底
        found_short = {re.split(r"[.:/]+", x)[-1] for x in found_syms}
        want_syms: Set[str] = set()
        for p in g["files"]:
            want_syms |= symbols_by_file.get(p, set())
        # 子代理可能写成 `模块.类.方法` 或 `文件::符号`；用短名兜底匹配
        missing_syms = []
        for s in sorted(want_syms):
            short = re.split(r"[.:/]+", s)[-1]
            if s in found_syms or short in found_syms or short in found_short:
                continue
            missing_syms.append(s)
        extra_syms = sorted(
            x
            for x in found_syms
            if x not in want_syms
            and re.split(r"[.:/]+", x)[-1] not in {re.split(r"[.:/]+", s)[-1] for s in want_syms}
            and x not in ("（无）",)
        )

        rec.update(
            {
                "files_want": len(want_files),
                "files_found": len(found_files & want_files),
                "missing_files": missing_files,
                "unknown_files": unknown_files,
                "symbols_want": len(want_syms),
                "symbols_missing": missing_syms,
                "symbols_extra": extra_syms,
                "ok": not missing_files and not missing_syms,
            }
        )
        n_file_ok += len(want_files) - len(missing_files)
        n_file_missing += len(missing_files)
        n_sym_ok += len(want_syms) - len(missing_syms)
        n_sym_missing += len(missing_syms)
        all_extra_syms += [f"{g['id']}:{x}" for x in extra_syms]
        if not rec["ok"]:
            report["ok"] = False
        report["groups"].append(rec)

    report["summary"] = {
        "groups": len(report["groups"]),
        "files_ok": n_file_ok,
        "files_missing": n_file_missing,
        "symbols_ok": n_sym_ok,
        "symbols_missing": n_sym_missing,
        "symbols_extra_not_in_ast": len(all_extra_syms),
    }
    with io.open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)

    print(f"[verify] groups={len(report['groups'])}  OVERALL={'PASS' if report['ok'] else 'FAIL'}")
    print(f"[verify] files   ok={n_file_ok} missing={n_file_missing}")
    print(f"[verify] symbols ok={n_sym_ok} missing={n_sym_missing}")
    if all_extra_syms:
        print(f"[verify] 未在 AST 中的符号（{len(all_extra_syms)}，可能为子代理编造或合法补充）:")
        for x in all_extra_syms[:40]:
            print("   -", x)
    for rec in report["groups"]:
        if not rec.get("ok", False):
            print(f"  FAIL {rec['id']}: missing_files={rec.get('missing_files')}")
            if rec.get("symbols_missing"):
                print(f"          missing_symbols({len(rec['symbols_missing'])}): {rec['symbols_missing'][:12]}")
    print(f"[verify] wrote {os.path.relpath(args.out, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
