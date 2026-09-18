#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""AST 静态清单生成器（文档摸底用，只读）。

用途：为「项目内容全解」文档提供**脚本可复算**的骨架——逐个 .py 文件列出
模块 docstring、导入、顶层函数/类/方法（含签名、默认值、注解、行号区间、
docstring 首行）。子代理只负责补"作用/实现说明"，符号与行号一律以此清单
为准，从机制上防止漏文件与编造符号。

用法（在仓库根目录）：
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/_survey_inventory.py \
        --out docs/_survey/inventory.json
"""
from __future__ import annotations

import argparse
import ast
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

SKIP_DIR_PARTS = {"__pycache__", ".venv", "node_modules", ".git", ".idea"}


def _fmt_default(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover - 极旧语法
        return "<unparseable>"


def _fmt_arg(a: ast.arg, default: str | None) -> Dict[str, Any]:
    ann = None
    if a.annotation is not None:
        try:
            ann = ast.unparse(a.annotation)
        except Exception:
            ann = "<unparseable>"
    return {"name": a.arg, "annotation": ann, "default": default}


def _signature(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> Dict[str, Any]:
    a = fn.args
    pos = list(a.posonlyargs) + list(a.args)
    defaults: List[str | None] = [None] * (len(pos) - len(a.defaults)) + [
        _fmt_default(d) for d in a.defaults
    ]
    args = [_fmt_arg(x, d) for x, d in zip(pos, defaults)]
    posonly = [x for x in args[: len(a.posonlyargs)]]
    args = args[len(a.posonlyargs):]
    kw_defaults: List[str | None] = [_fmt_default(d) for d in a.kw_defaults]
    kwonly = [_fmt_arg(x, d) for x, d in zip(a.kwonlyargs, kw_defaults)]
    vararg = a.vararg.arg if a.vararg else None
    kwarg = a.kwarg.arg if a.kwarg else None
    parts: List[str] = []
    if posonly:
        parts.append(
            ", ".join(
                x["name"] if x["default"] is None else f"{x['name']}={x['default']}"
                for x in posonly
            )
        )
        parts.append("/")
    parts += [x["name"] if x["default"] is None else f"{x['name']}={x['default']}" for x in args]
    if vararg:
        parts.append("*" + vararg)
    elif a.kwonlyargs:
        parts.append("*")
    parts += [x["name"] if x["default"] is None else f"{x['name']}={x['default']}" for x in kwonly]
    if kwarg:
        parts.append("**" + kwarg)
    ret = None
    if fn.returns is not None:
        try:
            ret = ast.unparse(fn.returns)
        except Exception:
            ret = "<unparseable>"
    return {
        "args": args,
        "kwonlyargs": kwonly,
        "vararg": vararg,
        "kwarg": kwarg,
        "returns": ret,
        "decorators": [ast.unparse(d) for d in fn.decorator_list],
        "display": f"{fn.name}({', '.join(parts)})",
    }


def _doc_first_line(node: ast.AST) -> str | None:
    d = ast.get_docstring(node, clean=True)
    if not d:
        return None
    return d.strip().splitlines()[0].strip()


def _sym(node: ast.AST, qual: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "qualname": qual,
        "name": getattr(node, "name", "?"),
        "lineno": node.lineno,
        "end_lineno": getattr(node, "end_lineno", None),
        "doc": _doc_first_line(node),
    }
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        out["kind"] = "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function"
        out["sig"] = _signature(node)
    else:
        out["kind"] = "class"
        out["bases"] = [ast.unparse(b) for b in getattr(node, "bases", [])]
    return out


def _walk(body: List[ast.stmt], prefix: str, out: List[Dict[str, Any]]) -> None:
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            qual = f"{prefix}{node.name}"
            out.append(_sym(node, qual))
            if isinstance(node, ast.ClassDef):
                _walk(node.body, qual + ".", out)


def scan_file(path: str, rel: str) -> Dict[str, Any]:
    with io.open(path, "r", encoding="utf-8", errors="replace") as fh:
        src = fh.read()
    rec: Dict[str, Any] = {
        "path": rel,
        "bytes": os.path.getsize(path),
        "lines": len(src.splitlines()),
        "module_doc": None,
        "imports": [],
        "symbols": [],
        "parse_error": None,
    }
    try:
        tree = ast.parse(src, filename=rel)
    except SyntaxError as e:
        rec["parse_error"] = f"SyntaxError: {e}"
        return rec
    rec["module_doc"] = _doc_first_line(tree)
    imps: List[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imps += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            imps.append(f"from {mod}")
    rec["imports"] = sorted(set(imps))
    _walk(tree.body, "", rec["symbols"])
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dirs", nargs="+", default=["src", "scripts"])
    args = ap.parse_args()

    files: List[Dict[str, Any]] = []
    for d in args.dirs:
        base = os.path.join(args.root, d)
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [x for x in dirnames if x not in SKIP_DIR_PARTS]
            for fn in sorted(filenames):
                if not fn.endswith(".py"):
                    continue
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, args.root).replace(os.sep, "/")
                files.append(scan_file(full, rel))
    files.sort(key=lambda r: r["path"])

    total_syms = sum(len(f["symbols"]) for f in files)
    payload = {
        "root": os.path.abspath(args.root),
        "dirs": args.dirs,
        "n_files": len(files),
        "n_symbols": total_syms,
        "total_lines": sum(f["lines"] for f in files),
        "files": files,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with io.open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    print(f"[inventory] files={len(files)} symbols={total_syms} lines={payload['total_lines']}")
    print(f"[inventory] wrote {args.out}")
    bad = [f["path"] for f in files if f["parse_error"]]
    if bad:
        print(f"[inventory] WARNING parse errors in {len(bad)}: {bad}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
