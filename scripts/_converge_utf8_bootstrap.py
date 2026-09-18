#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""一次性 codemod：把内联的 UTF-8 兜底块收敛到 `rl/io_bootstrap.force_utf8_stdout`（Tier 1 · T1-1）。

**替换什么**（只匹配这一种形态，见 `_BLOCK`）：

    try:
        sys.stdout.reconfigure(encoding="utf-8")            # 或带 errors="replace"
    except Exception:
        pass

→

    from rl.io_bootstrap import force_utf8_stdout  # noqa: E402  (T1-1: UTF-8 兜底单一来源)
    force_utf8_stdout()

**只动「安全子集」**：目标文件必须在**该块之前**已有 `sys.path.insert(`（⇒ `rl.*` 可导入），
否则**跳过并报出**。理由：对没有 path guard 的文件，收敛就必须**新增一条 `sys.path.insert`**，
那会改变这些脚本的 import 解析顺序（它们可能刻意依赖 cwd 相对导入）—— 收益只是 DRY，
风险不对称 ⇒ **不做**，留档为未收敛清单。

用法：
    python scripts/_converge_utf8_bootstrap.py --dry-run     # 只报要改哪些
    python scripts/_converge_utf8_bootstrap.py               # 实际写入
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_ROOT = Path(__file__).resolve().parent.parent

#: 内联块（两种变体：带 / 不带 errors="replace"）
_BLOCK = re.compile(
    r"^([ \t]*)try:\n"
    r"\1[ \t]*sys\.stdout\.reconfigure\(encoding=[\"']utf-8[\"'](?:,\s*errors=[\"'][^\"']*[\"'])?\)\n"
    r"\1except Exception:\n"
    r"\1[ \t]*pass\n",
    re.M,
)

_REPL = ("{ind}from rl.io_bootstrap import force_utf8_stdout  # noqa: E402  "
         "(T1-1: UTF-8 兜底单一来源)\n"
         "{ind}force_utf8_stdout()\n")


def targets():
    return sorted(list((_ROOT / "scripts").glob("*.py"))
                  + list((_ROOT / "src" / "clasher_new" / "rl").glob("*.py"))
                  + list((_ROOT / "src" / "clasher_new").glob("*.py")))


def main() -> int:
    ap = argparse.ArgumentParser(description="收敛内联 UTF-8 兜底（T1-1）")
    ap.add_argument("--dry-run", action="store_true", help="只报告，不写文件")
    args = ap.parse_args()

    changed, skipped_no_guard, no_match = [], [], []
    for f in targets():
        if f.name in ("io_bootstrap.py", "_converge_utf8_bootstrap.py"):
            continue
        txt = f.read_text(encoding="utf-8", errors="replace")
        m = _BLOCK.search(txt)
        if not m:
            if "stdout.reconfigure(" in txt:
                no_match.append(f.relative_to(_ROOT).as_posix())
            continue
        # 安全子集判据：块之前必须已有 sys.path.insert(
        guards = [mm.start() for mm in re.finditer(r"sys\.path\.insert\(", txt)]
        if not any(g < m.start() for g in guards):
            skipped_no_guard.append(f.relative_to(_ROOT).as_posix())
            continue
        new = txt[:m.start()] + _REPL.format(ind=m.group(1)) + txt[m.end():]
        if new == txt:
            continue
        changed.append(f.relative_to(_ROOT).as_posix())
        if not args.dry_run:
            f.write_text(new, encoding="utf-8")

    print(f"{'[dry-run] ' if args.dry_run else ''}已收敛 = {len(changed)} 个")
    for c in changed:
        print("   -", c)
    print(f"\n⚠️ 跳过（reconfigure 之前没有 sys.path.insert ⇒ 收敛需新增 sys.path 项，"
          f"改 import 解析顺序，**本步不做**）= {len(skipped_no_guard)} 个")
    for c in skipped_no_guard:
        print("   !", c)
    if no_match:
        print(f"\n形态不匹配（可能是 `if hasattr(...)` 变体或缩进不同）= {len(no_match)} 个")
        for c in no_match:
            print("   ?", c)
    return 0


if __name__ == "__main__":
    sys.exit(main())
