#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""只读结构核对器（Tier 0 · T0-2）。

**用途**：把 [`docs/structure_optimization_plan_2026-09-19.md`](../docs/structure_optimization_plan_2026-09-19.md)
§1/§3/§4/§5 里的**结构性事实**变成**一条可复跑的命令**，供每轮结构改动前后对账。

**契约（重要）**：
- **纯只读**：不写任何文件、不改任何状态；`--json` 才写一份 JSON。
- **不 import 产品代码**：只用 `ast` 解析源码文本 ⇒ 不会因 `card_utils` 的 cwd 契约而在异地失败。
- **失败不抛异常**：单项检查失败记 `error`，其余照跑 ⇒ 可当长跑前的门禁用。
- 本项目 Windows python stdout 为 gbk ⇒ 一律 `reconfigure(encoding="utf-8", errors="replace")`。

检查项：
  ① `areas`     分区文件数 / LOC
  ② `utf8`      `_force_utf8_stdout` 各实现的字节级一致性（收敛目标）
  ③ `layering`  引擎 → `rl/` 的**反向 import 边**（应为 0）
  ④ `deadfiles` 已清理死件是否**复活**（应为 0 存在）
  ⑤ `abspaths`  硬编码绝对路径（`E:/...` 之类）清单
  ⑥ `lazyimp`   `rl/` 内**函数体内 import** 的条数与占比（结构信号）
  ⑦ `selftest`  `rl/selftest.py` 的 `test_*` 定义数 vs `main()` 调用数（应相等）

用法：
    python scripts/_structure_check.py                 # 人类可读
    python scripts/_structure_check.py --json out.json # 另写 JSON
    python scripts/_structure_check.py --selftest      # 自检（合成目录，零仓库依赖）
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

try:  # GBK 陷阱：本项目 Windows 侧 stdout 默认 gbk
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent

#: 已清理死件（T0-1，2026-09-19）。**这些路径不应再存在**。
#: 与 `docs/structure_cleanup_2026-09-19.md` 的清单逐条对应。
DEAD_FILES = [
    "re_lib.py",
    "src/clasher_new/pathfinding.py",
    "scripts/stop_solo_training_2200.py",
    "scripts/stop_solo_2200.log",
    "1",
    ".p6.png",
    "0631.pdf",
    "libg_strings.txt",
    ":memory:.ses",
    "src/clasher_new/tmp_formation_test.py",
    "src/clasher_new/tmp_path_debug.py",
    "src/clasher_new/tmp_spell_forensics.py",
    "src/clasher_new/tmp_target_verify.py",
    "src/clasher_new/tmp_tower_chip_test.py",
    "src/clasher_new/gamedata.json.bak_ronin_vines",
    # runs/_tmp_* 用前缀匹配（见 _dead_prefixes）
]
#: 前缀匹配的死件（glob）
DEAD_PREFIXES = ["src/clasher_new/runs/_tmp_"]

#: 硬编码绝对路径：Windows 盘符 / UNC，且**不是**注释
_ABS_RE = re.compile(r"""["']((?:[A-Za-z]:[\\/])[^"']{2,})["']""")

#: ⑤ 扫描时排除的文件（**自指**：本文件的 `--selftest` 里必须造一条假绝对路径才测得出判别力，
#: 那不是硬编码缺陷）。排除后 `--selftest` 的合成树仍会扫到 `scripts/hard.py`（排除项只匹配本文件）。
_ABSPATH_SKIP = {Path(__file__).name}

#: 排除目录（产物 / 依赖 / VCS）
_SKIP_DIRS = {".git", ".venv", "clash-royale-simulator-main.venv", "site-packages",
              "node_modules", "__pycache__", "runs", ".tmp", "data_official",
              "client_side", "replays", "dist", "build", ".pytest_cache"}


# ---------------------------------------------------------------- 基础设施
def iter_py(root: Path, subdir: str, recursive: bool = False):
    """列出 `root/subdir` 下的 .py（recursive 时递归），跳过 `_SKIP_DIRS`。"""
    base = root / subdir
    if not base.is_dir():
        return []
    out = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fn in filenames:
            if fn.endswith(".py"):
                out.append(Path(dirpath) / fn)
        if not recursive:
            break
    return sorted(out)


def read_text(p: Path) -> str:
    """读文本，容错（结构核对不需要正确解码，只要稳定）。"""
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def parse(p: Path):
    """AST 解析；失败返回 None（不抛）。"""
    try:
        return ast.parse(read_text(p))
    except Exception:
        return None


def loc(p: Path) -> int:
    """物理行数（与 `wc -l` 同口径：按行尾计数，末行无换行也算）。"""
    return len(read_text(p).splitlines())


def rel(root: Path, p: Path) -> str:
    try:
        return p.relative_to(root).as_posix()
    except ValueError:
        return p.as_posix()


# ---------------------------------------------------------------- ① areas
def check_areas(root: Path) -> dict:
    areas = {
        "engine_top": iter_py(root, "src/clasher_new", recursive=False),
        "rl": iter_py(root, "src/clasher_new/rl", recursive=False),
        "scripts": iter_py(root, "scripts", recursive=True),
    }
    out = {}
    total_n = total_loc = 0
    for name, files in areas.items():
        n = len(files)
        lc = sum(loc(f) for f in files)
        out[name] = {"files": n, "loc": lc}
        total_n += n
        total_loc += lc
    out["_total"] = {"files": total_n, "loc": total_loc}
    return out


# ---------------------------------------------------------------- ② utf8
def check_utf8(root: Path) -> dict:
    """`_force_utf8_stdout` 各实现的函数体是否逐字相同（去掉空白差异后比 sha256）。"""
    impls = {}
    for f in iter_py(root, "src/clasher_new", recursive=True) + iter_py(root, "scripts", recursive=True):
        tree = parse(f)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_force_utf8_stdout":
                seg = ast.get_source_segment(read_text(f), node) or ""
                norm = re.sub(r"\s+", "", seg)
                digest = hashlib.sha256(norm.encode("utf-8")).hexdigest()[:12]
                impls[rel(root, f)] = {"lineno": node.lineno, "digest": digest}
    digests = {v["digest"] for v in impls.values()}
    return {
        "count": len(impls),
        "distinct_bodies": len(digests),
        "identical": len(digests) <= 1,
        "impls": impls,
    }


# ---------------------------------------------------------------- ③ layering
def _imported_tops(tree) -> set:
    """一个 AST 里 import 到的**顶层模块名**集合（含函数体内的）。"""
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                mods.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # 相对 import 不算
                continue
            if node.module:
                mods.add(node.module.split(".")[0])
    return mods


def check_layering(root: Path) -> dict:
    """引擎 → rl/ 的反向边（应为 0）。"""
    reverse = []
    engine_files = iter_py(root, "src/clasher_new", recursive=False)
    for f in engine_files:
        tree = parse(f)
        if tree is None:
            continue
        mods = _imported_tops(tree)
        if "rl" in mods:
            reverse.append(rel(root, f))
    return {
        "engine_files": len(engine_files),
        "reverse_edges": len(reverse),
        "reverse_files": sorted(reverse),
        "ok": len(reverse) == 0,
    }


# ---------------------------------------------------------------- ④ deadfiles
def check_deadfiles(root: Path) -> dict:
    present = []
    for d in DEAD_FILES:
        if (root / d).exists():
            present.append(d)
    for pref in DEAD_PREFIXES:
        base = root / pref
        parent = base.parent
        if parent.is_dir():
            for p in sorted(parent.glob(base.name + "*")):
                present.append(rel(root, p))
    return {"checked": len(DEAD_FILES), "revived": len(present), "revived_paths": present,
            "ok": len(present) == 0}


# ---------------------------------------------------------------- ⑤ abspaths
def check_abspaths(root: Path) -> dict:
    hits = []
    for f in iter_py(root, "src/clasher_new", recursive=True) + iter_py(root, "scripts", recursive=True):
        if f.name in _ABSPATH_SKIP:
            continue
        text = read_text(f)
        if sys.version_info >= (3, 8):
            tree = parse(f)
            docstrings = set()
            if tree is not None:
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        ds = ast.get_docstring(node, clean=False)
                        if ds:
                            docstrings.add(ds)
            # 去掉 docstring 后扫（docstring 里出现盘符路径 = 文档，不是硬编码）
            body = text
            for ds in docstrings:
                body = body.replace(ds, "")
        else:  # pragma: no cover
            body = text
        # 去注释行（# 之后），避免把说明文字算成硬编码
        body = "\n".join(ln.split("#", 1)[0] for ln in body.splitlines())
        for ln_no, ln in enumerate(body.splitlines(), 1):
            for m in _ABS_RE.finditer(ln):
                hits.append({"file": rel(root, f), "lineno": ln_no, "path": m.group(1)})
    files = sorted({h["file"] for h in hits})
    return {"hits": len(hits), "files": len(files), "detail": hits, "ok": len(hits) == 0}


# ---------------------------------------------------------------- ⑥ lazyimp
def check_lazy_imports(root: Path) -> dict:
    """`rl/` 内写在函数体内的 import（顶层 vs 函数体内）。

    ⚠️ **两套口径必须并列**（【R17】）：`rl/selftest.py` 一个文件就占 ~471 条函数体内 import
    （测试层刻意不在模块顶层拖 torch/引擎），把它算进来与排除它**是两个不同的数**。
    本函数同时给出 `all`（含 selftest）与 `excl_selftest` 两套，**不得只引其中一个**。
    """
    per_file = {}
    agg = {"all": {"top": 0, "lazy": 0}, "excl_selftest": {"top": 0, "lazy": 0}}
    for f in iter_py(root, "src/clasher_new/rl", recursive=False):
        tree = parse(f)
        if tree is None:
            continue
        t = sum(1 for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom)))
        funcs = [(fn.lineno, fn.end_lineno or fn.lineno)
                 for fn in ast.walk(tree)
                 if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))]
        l = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)) and \
                    any(a <= node.lineno <= b for a, b in funcs):
                l += 1
        if t or l:
            per_file[rel(root, f)] = {"top": t, "lazy": l}
        for key in ("all",) + (() if f.name == "selftest.py" else ("excl_selftest",)):
            agg[key]["top"] += t
            agg[key]["lazy"] += l

    out = {"per_file": per_file}
    for key, v in agg.items():
        total = v["top"] + v["lazy"]
        out[key] = {"top": v["top"], "lazy": v["lazy"], "total": total,
                    "lazy_frac": round(v["lazy"] / total, 4) if total else 0.0}
    # 兼容旧键（= all 口径），便于既有调用方
    out.update({"top": agg["all"]["top"], "lazy": agg["all"]["lazy"],
                "total": out["all"]["total"], "lazy_frac": out["all"]["lazy_frac"]})
    return out


# ---------------------------------------------------------------- ⑦ selftest
def check_selftest(root: Path) -> dict:
    p = root / "src/clasher_new/rl/selftest.py"
    if not p.is_file():
        return {"error": "rl/selftest.py 不存在"}
    tree = parse(p)
    if tree is None:
        return {"error": "rl/selftest.py 解析失败"}
    defined = {n.name for n in tree.body
               if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")}
    called = set()
    has_main = False
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name == "main":
            has_main = True
            for sub in ast.walk(n):
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) \
                        and sub.func.id.startswith("test_"):
                    called.add(sub.func.id)
    return {
        "defined": len(defined),
        "called_in_main": len(called),
        "defined_not_called": sorted(defined - called),
        "called_not_defined": sorted(called - defined),
        "has_main": has_main,
        "loc": loc(p),
        "ok": defined == called and has_main,
    }


CHECKS = {
    "areas": check_areas,
    "utf8": check_utf8,
    "layering": check_layering,
    "deadfiles": check_deadfiles,
    "abspaths": check_abspaths,
    "lazyimp": check_lazy_imports,
    "selftest": check_selftest,
}


def run_all(root: Path) -> dict:
    out = {}
    for name, fn in CHECKS.items():
        try:
            out[name] = fn(root)
        except Exception as e:  # 单项失败不影响其余
            out[name] = {"error": f"{type(e).__name__}: {e}"}
    return out


# ---------------------------------------------------------------- 报告
def render(res: dict) -> str:
    L = []
    a = res.get("areas", {})
    L.append("== ① 规模 ==")
    for k in ("engine_top", "rl", "scripts", "_total"):
        if k in a:
            L.append(f"  {k:12s} files={a[k]['files']:4d}  loc={a[k]['loc']:6d}")
    u = res.get("utf8", {})
    _impls = u.get("impls", {}) or {}
    _loc = ", ".join("{}:{}".format(k, v["lineno"]) for k, v in _impls.items())
    L.append("== ② _force_utf8_stdout：{} 处，互异实现 {} 种 ⇒ {}".format(
        u.get("count"), u.get("distinct_bodies"),
        "一致" if u.get("identical") else "**不一致**") + ("（{}）".format(_loc) if _loc else ""))
    lay = res.get("layering", {})
    L.append(f"== ③ 引擎→rl 反向边 = {lay.get('reverse_edges')}"
             f"（应为 0）{' OK' if lay.get('ok') else ' **FAIL** ' + str(lay.get('reverse_files'))}")
    d = res.get("deadfiles", {})
    L.append(f"== ④ 死件复活 = {d.get('revived')}（应为 0）"
             f"{' OK' if d.get('ok') else ' **FAIL** ' + str(d.get('revived_paths'))}")
    ab = res.get("abspaths", {})
    L.append(f"== ⑤ 硬编码绝对路径 = {ab.get('hits')} 处 / {ab.get('files')} 文件"
             f"{' OK' if ab.get('ok') else ''}")
    for h in ab.get("detail", [])[:12]:
        L.append(f"     {h['file']}:{h['lineno']}  {h['path']}")
    li = res.get("lazyimp", {})
    _a, _e = li.get("all", {}), li.get("excl_selftest", {})
    L.append(f"== ⑥ rl/ 函数体内 import（**两套口径并列**，【R17】）")
    L.append(f"     含 selftest.py：{_a.get('lazy')}/{_a.get('total')}"
             f"（{100 * _a.get('lazy_frac', 0):.1f}%）")
    L.append(f"     排除 selftest.py：{_e.get('lazy')}/{_e.get('total')}"
             f"（{100 * _e.get('lazy_frac', 0):.1f}%）")
    st = res.get("selftest", {})
    L.append(f"== ⑦ selftest：定义 {st.get('defined')} / main() 调用 {st.get('called_in_main')}"
             f"（loc={st.get('loc')}）{' OK' if st.get('ok') else ' **FAIL**'}"
             + (f" 未登记={st.get('defined_not_called')}" if st.get("defined_not_called") else "")
             + (f" 未定义={st.get('called_not_defined')}" if st.get("called_not_defined") else ""))
    return "\n".join(L)


# ---------------------------------------------------------------- selftest
def _selftest() -> int:
    """自检：在临时目录造一棵**已知性质**的小树，断言每项检查读得对。

    零仓库依赖 ⇒ 换机也能跑（本项目 Windows 侧探针常见的失效模式就是依赖 cwd/绝对路径）。
    """
    fails = []

    def ck(name, cond):
        print(f"  [{'ok' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "src/clasher_new/rl").mkdir(parents=True)
        (root / "scripts").mkdir(parents=True)

        # 引擎模块：其中一个**反向 import rl**（用来验 ③ 能报出来）
        (root / "src/clasher_new/battle.py").write_text("import core\n", encoding="utf-8")
        (root / "src/clasher_new/bad.py").write_text("import rl.env_wrapper\n", encoding="utf-8")
        # rl 模块：一个顶层 import，一个函数体内 import（验 ⑥）
        (root / "src/clasher_new/rl/follower.py").write_text(
            "import torch\n\n\ndef act():\n    import battle\n    return 1\n", encoding="utf-8")
        # 两处 _force_utf8_stdout，函数体逐字相同（验 ② 判「一致」）
        for name in ("a.py", "b.py"):
            (root / "scripts" / name).write_text(
                "import sys\n\n\ndef _force_utf8_stdout():\n    try:\n"
                "        sys.stdout.reconfigure(encoding='utf-8')\n    except Exception:\n        pass\n",
                encoding="utf-8")
        # 硬编码绝对路径（验 ⑤）
        (root / "scripts/hard.py").write_text(
            "import sys\nsys.path.insert(0, r'E:/x/y/src')\n", encoding="utf-8")
        # 死件复活（验 ④）
        (root / "src/clasher_new/pathfinding.py").write_text("x = 1\n", encoding="utf-8")
        # selftest：定义 2 个，main() 只调 1 个（验 ⑦ 报「未登记」）
        (root / "src/clasher_new/rl/selftest.py").write_text(
            "def test_a():\n    pass\n\n\ndef test_b():\n    pass\n\n\n"
            "def main():\n    test_a()\n", encoding="utf-8")

        res = run_all(root)
        ck("① areas 数到 3 个引擎文件", res["areas"]["engine_top"]["files"] == 3)
        ck("① rl 数到 2 个文件", res["areas"]["rl"]["files"] == 2)
        ck("② 两处实现判为一致", res["utf8"]["count"] == 2 and res["utf8"]["identical"])
        ck("③ 反向边 = 1 且能指出文件", res["layering"]["reverse_edges"] == 1
           and res["layering"]["reverse_files"] == ["src/clasher_new/bad.py"])
        ck("④ 死件复活 = 1（pathfinding.py）", res["deadfiles"]["revived"] == 1)
        ck("⑤ 抓到 1 处硬编码绝对路径", res["abspaths"]["hits"] == 1
           and "E:/x/y/src" in res["abspaths"]["detail"][0]["path"])
        ck("⑥ 懒 import 记到 1 条 / 顶层 1 条", res["lazyimp"]["lazy"] == 1
           and res["lazyimp"]["top"] == 1)
        ck("⑥ 两套口径键并列存在（【R17】）",
           "all" in res["lazyimp"] and "excl_selftest" in res["lazyimp"])
        ck("⑦ 报出 test_b 未登记", res["selftest"]["defined_not_called"] == ["test_b"]
           and res["selftest"]["ok"] is False)

        # 反向验证：把 bad.py 改成干净，③ 必须变 0（证明不是恒真）
        (root / "src/clasher_new/bad.py").write_text("import core\n", encoding="utf-8")
        res2 = run_all(root)
        ck("③ 反向验证：修掉后反向边变 0", res2["layering"]["reverse_edges"] == 0)

    print(f"\n自检：{'ALL PASS' if not fails else 'FAILED: ' + ', '.join(fails)}")
    return 0 if not fails else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="只读结构核对器（T0-2）")
    ap.add_argument("--json", default=None, help="另写一份 JSON 到该路径")
    ap.add_argument("--selftest", action="store_true", help="自检（合成目录，零仓库依赖）")
    ap.add_argument("--root", default=None, help="仓库根（缺省从 __file__ 推）")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()

    root = Path(args.root).resolve() if args.root else _ROOT
    res = run_all(root)
    print(f"仓库根：{root}")
    print(render(res))
    if args.json:
        outp = Path(args.json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nJSON → {outp}")

    # 退出码：三项硬门禁（反向边 / 死件复活 / selftest 登记）任一失败 ⇒ 1
    hard = [res.get("layering", {}).get("ok", False),
            res.get("deadfiles", {}).get("ok", False),
            res.get("selftest", {}).get("ok", False)]
    return 0 if all(hard) else 1


if __name__ == "__main__":
    sys.exit(main())
