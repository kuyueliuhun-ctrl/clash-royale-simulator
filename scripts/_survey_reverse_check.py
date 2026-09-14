#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""抽样反查 / 防编造体检（**只读**，可复算）。

三件事：
  A. **符号条目抽检**（`docs/full_code_reference.md`）：按固定步长（每 60 条取 1 条，确定性、
     非人工挑选）取样本，对每条校验
       1) 标题里的 `[L起-止]` 区间内**确实**有该名字的 `def`/`class`；
       2) 条目写的**签名参数名集合**与源码 `def` 的形参集合一致；
       3) 该条目里出现的每一个 `文件.py:行号` 引用都**真实存在**且在文件行数范围内。
  B/C. **全量引用体检**（`docs/training_method.md`、`docs/game_engine.md`）：把正文里
       所有 `xxx.py:NNN`（含 `NNN-MMM` 区间）引用逐个回源，统计解析失败 / 越界的条数。

用法（仓库根目录）：
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/_survey_reverse_check.py \
        --out docs/_survey/reverse_check_report.json
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

ENTRY_RE = re.compile(r"^(#{4,5})\s+(\d+(?:\.\d+)+)\s+([^\s\[]+)\s*(?:\[L(\d+)-(\d+)\])?\s*$", re.M)
SEC_RE = re.compile(r"^###\s+(\d+\.\d+)\s+`([^`]+)`\s*$", re.M)
SIG_RE = re.compile(r"^-\s*签名:\s*`(.+?)`\s*$", re.M)
CITE_RE = re.compile(r"([A-Za-z0-9_][A-Za-z0-9_./\\-]*\.py):(\d+)(?:\s*-\s*(\d+))?")

# 允许的"模块名简写"（文档里常写 `config.py:127` 而真实路径是 src/clasher_new/rl/config.py）
SKIP_DIRS = {"__pycache__", ".git", ".venv", "clash-royale-simulator-main.venv", "node_modules", ".idea"}


def index_sources() -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """返回 (规范相对路径 -> 绝对路径, basename -> [相对路径...])"""
    canon: Dict[str, str] = {}
    bybase: Dict[str, List[str]] = {}
    for base in ("src", "scripts", ".", "docs", "ideas", "runs"):
        root = os.path.join(ROOT, base)
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in filenames:
                if not fn.endswith(".py"):
                    continue
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, ROOT).replace(os.sep, "/")
                canon[rel] = full
                bybase.setdefault(fn, []).append(rel)
    return canon, bybase


def resolve(cite: str, canon: Dict[str, str], bybase: Dict[str, List[str]],
            need_line: int = 0) -> Optional[str]:
    """把文档里的 `xxx.py` 引用解析到真实文件。

    同名文件（如 `evaluate.py` 同时存在于 `src/clasher_new/` 与 `src/clasher_new/rl/`）
    必须选**行数足够容纳被引用行号**、且路径后缀与原引用最匹配的那个。
    """
    c = cite.replace("\\", "/").lstrip("./")
    if c in canon:
        return canon[c]
    base = os.path.basename(c)
    cands = bybase.get(base, [])
    if not cands:
        return None
    if len(cands) == 1:
        return canon[cands[0]]

    def score(rel: str) -> Tuple[int, int, int]:
        n = len(file_lines(canon[rel]))
        fits = 1 if n >= need_line else 0
        common = len(os.path.commonprefix([rel[::-1], c[::-1]]))
        return (fits, common, -n)

    return canon[sorted(cands, key=score, reverse=True)[0]]


def file_lines(path: str) -> List[str]:
    with io.open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read().splitlines()


def parse_def(src: List[str], lo: int, hi: int, name: str) -> Optional[Tuple[str, str]]:
    """在 [lo,hi] 内找 `def name(...)` / `class name`，返回 (完整签名文本, 参数名集合文本)"""
    pat = re.compile(r"^\s*(?:async\s+)?(?:def|class)\s+" + re.escape(name) + r"\b")
    for i in range(max(0, lo - 1), min(len(src), hi)):
        if pat.match(src[i]):
            # 拼接直到括号配平
            buf = []
            depth = 0
            started = False
            for j in range(i, min(len(src), i + 40)):
                line = src[j]
                buf.append(line.strip())
                for ch in line:
                    if ch == "(":
                        depth += 1
                        started = True
                    elif ch == ")":
                        depth -= 1
                if started and depth <= 0:
                    break
                if not started and ":" in line and "class" in line:
                    break
            return " ".join(buf), " ".join(buf)
    return None


def params_of(sig_text: str) -> List[str]:
    m = re.search(r"\(([^)]*)", sig_text)
    if not m:
        return []
    out = []
    for tok in re.split(r",", m.group(1)):
        tok = tok.strip()
        if not tok or tok in ("/", "*"):
            continue
        tok = tok.split(":")[0].split("=")[0].strip().lstrip("*")
        if tok:
            out.append(tok)
    return out


def check_entries(doc: str, canon, bybase, step: int) -> Dict:
    entries = ENTRY_RE.findall(doc)
    secs = SEC_RE.findall(doc)
    # 章节号 -> 文件路径
    secmap = {num: path for num, path in secs}
    samples = entries[::step]
    rows = []
    for hashes, num, name, lo, hi in samples:
        parts = num.split(".")
        path, chapter = None, None
        for k in range(len(parts) - 1, 0, -1):   # 先试最长的前缀
            cand = ".".join(parts[:k])
            if cand in secmap:
                path, chapter = secmap[cand], cand
                break
        rec = {"chapter": num, "name": name, "range": f"L{lo}-{hi}", "file": path,
               "def_found": False, "params_ok": None, "missing_params": [], "extra_params": [],
               "citations_checked": 0, "citations_bad": [], "verdict": "FAIL"}
        if not path or path not in canon:
            rec["verdict"] = "无法判定(章节未映射到文件)"
            rows.append(rec)
            continue
        src = file_lines(canon[path])
        short = name.split(".")[-1]
        got = parse_def(src, int(lo), int(hi), short)
        if not got:
            rows.append(rec)
            continue
        rec["def_found"] = True
        # 找条目正文（到下一个 #### 或 #####）
        m = re.search(
            r"^#{4,5}\s+" + re.escape(num) + r"\s+.*?$(.*?)(?=^#{4,5}\s+\d|\Z)",
            doc, flags=re.M | re.S)
        body = m.group(1) if m else ""
        msig = SIG_RE.search(body)
        if msig:
            want = set(params_of(msig.group(1)))
            have = set(params_of(got[0]))
            rec["missing_params"] = sorted(p for p in have if p not in want and p != "self")
            rec["extra_params"] = sorted(p for p in want if p not in have)
            rec["params_ok"] = not rec["missing_params"] and not rec["extra_params"]
        for cite, a, b in CITE_RE.findall(body):
            p = resolve(cite, canon, bybase, int(b) if b else int(a))
            rec["citations_checked"] += 1
            if p is None:
                rec["citations_bad"].append(f"{cite}:{a} (文件未找到)")
                continue
            n = len(file_lines(p))
            if int(a) > n or (b and int(b) > n):
                rec["citations_bad"].append(f"{cite}:{a}{'-' + b if b else ''} (越界, 共 {n} 行)")
        rec["verdict"] = "PASS" if (rec["def_found"] and rec["params_ok"] is not False
                                   and not rec["citations_bad"]) else "FAIL"
        rows.append(rec)
    return {
        "entries_total": len(entries),
        "step": step,
        "sampled": len(samples),
        "pass": sum(1 for r in rows if r["verdict"] == "PASS"),
        "fail": sum(1 for r in rows if r["verdict"] == "FAIL"),
        "undecidable": sum(1 for r in rows if str(r["verdict"]).startswith("无法判定")),
        "rows": rows,
    }


def check_citations(doc_path: str, canon, bybase) -> Dict:
    with io.open(doc_path, encoding="utf-8", errors="replace") as fh:
        doc = fh.read()
    total = bad_file = bad_line = 0
    samples = []
    for cite, a, b in CITE_RE.findall(doc):
        if os.path.basename(cite) in ("xxx.py",):   # 文档里的格式占位符，不是真引用
            continue
        total += 1
        p = resolve(cite, canon, bybase, int(b) if b else int(a))
        if p is None:
            bad_file += 1
            if len(samples) < 20:
                samples.append(f"{cite}:{a} 文件未找到")
            continue
        n = len(file_lines(p))
        if int(a) > n or (b and int(b) > n):
            bad_line += 1
            if len(samples) < 20:
                samples.append(f"{cite}:{a}{'-' + b if b else ''} 越界(共 {n} 行)")
    return {"doc": os.path.relpath(doc_path, ROOT), "citations": total,
            "file_not_found": bad_file, "line_out_of_range": bad_line, "samples": samples}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", default=os.path.join(ROOT, "docs", "full_code_reference.md"))
    ap.add_argument("--step", type=int, default=60)
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "_survey", "reverse_check_report.json"))
    args = ap.parse_args()

    canon, bybase = index_sources()
    print(f"[reverse] 源码索引：{len(canon)} 个 .py")

    with io.open(args.doc, encoding="utf-8", errors="replace") as fh:
        doc = fh.read()
    A = check_entries(doc, canon, bybase, args.step)
    print(f"[reverse] A 符号条目抽检：总 {A['entries_total']} 条，按每 {A['step']} 条取 1 ⇒ "
          f"抽 {A['sampled']} 条；PASS {A['pass']} / FAIL {A['fail']} / 无法判定 {A['undecidable']}")
    for r in A["rows"]:
        if r["verdict"] != "PASS":
            print(f"   [{r['verdict']}] {r['chapter']} {r['name']} {r['range']} {r['file']} "
                  f"def_found={r['def_found']} missing={r['missing_params']} bad_cites={r['citations_bad'][:2]}")

    out: Dict[str, object] = {"A_entries": A, "docs": []}
    for rel in ("docs/full_code_reference.md", "docs/training_method.md", "docs/game_engine.md"):
        p = os.path.join(ROOT, rel)
        if os.path.exists(p):
            r = check_citations(p, canon, bybase)
            out["docs"].append(r)
            print(f"[reverse] B/C {rel}: 引用 {r['citations']} 条；文件未找到 {r['file_not_found']}；"
                  f"行号越界 {r['line_out_of_range']}")
            for s in r["samples"][:8]:
                print("    -", s)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with io.open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"[reverse] wrote {os.path.relpath(args.out, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
