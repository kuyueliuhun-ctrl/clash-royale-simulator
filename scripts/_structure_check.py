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
    # ⚠️ **`rl` 必须递归**（T2-8 教训）：原先是 `recursive=False` ⇒ `rl/selftests/part*.py`
    # （6,164 行）被**静默漏掉**，`rl` 的读数从 24,145 掉到 18,495 —— **看起来像"代码少了 5,650 行"**，
    # 实际只是新子包没被数到。这类"仪器没跟上结构"的假读数最危险：没人会去质疑"行数变少了"。
    areas = {
        "engine_top": iter_py(root, "src/clasher_new", recursive=False),
        "rl": iter_py(root, "src/clasher_new/rl", recursive=True),
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
UTF8_NAMES = ("_force_utf8_stdout", "force_utf8_stdout")


def check_utf8(root: Path) -> dict:
    """UTF-8 兜底是否收敛到**单一实现**（T1-1）。

    ⚠️ **三个阶段必须都能读**（否则这条检查会在收敛后变成空检查）：

    1. **收敛前**：多份 `def _force_utf8_stdout` ⇒ 比**函数体**指纹（`body_distinct`）。
       把 docstring 也算进去会得到「3 种互异」这个**误导性**结论 —— 实测
       `rl/run_league.py` 与 `rl/dashboard.py` 的**函数体逐字相同**（仅 docstring 不同），
       真正不同的是 `scripts/probe_value_ln.py`（**弱化版**）。
    2. **收敛中**：`def` 与**别名赋值**（`_force_utf8_stdout = force_utf8_stdout`）并存 ⇒ 两边都数。
    3. **收敛后**：只剩 `rl/io_bootstrap.py` 一处 `def force_utf8_stdout` + N 处别名
       ⇒ 判据变成「**恰 1 处强定义 + 0 处弱定义 + ≥2 处别名**」。

    弱化判据（实现层的两处硬要求）：必须同时**碰 stderr** 且**带 `errors="replace"`**。
    """
    defs, aliases = {}, []
    for f in iter_py(root, "src/clasher_new", recursive=True) + iter_py(root, "scripts", recursive=True):
        tree = parse(f)
        if tree is None:
            continue
        src = read_text(f)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in UTF8_NAMES:
                seg = ast.get_source_segment(src, node) or ""
                body = [st for st in node.body
                        if not (isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant)
                                and isinstance(st.value.value, str))]
                try:
                    body_src = "\n".join(ast.unparse(st) for st in body)
                except Exception:
                    body_src = seg
                _n = lambda s: re.sub(r"\s+", "", s)
                defs[rel(root, f)] = {
                    "lineno": node.lineno, "name": node.name,
                    "body_digest": hashlib.sha256(_n(body_src).encode("utf-8")).hexdigest()[:12],
                    "full_digest": hashlib.sha256(_n(seg).encode("utf-8")).hexdigest()[:12],
                    "touches_stderr": "stderr" in body_src,
                    "has_errors_replace": "replace" in body_src,
                }
        # 别名两种形态都要认：
        #   ① 赋值      `_force_utf8_stdout = force_utf8_stdout`
        #   ② 导入改名  `from rl.io_bootstrap import force_utf8_stdout as _force_utf8_stdout`
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                    and isinstance(node.targets[0], ast.Name) \
                    and node.targets[0].id in UTF8_NAMES:
                v = node.value
                vname = v.id if isinstance(v, ast.Name) else (
                    v.attr if isinstance(v, ast.Attribute) else "")
                if vname in UTF8_NAMES:
                    aliases.append({"file": rel(root, f), "lineno": node.lineno,
                                    "target": node.targets[0].id, "source": vname,
                                    "form": "assign"})
            elif isinstance(node, ast.ImportFrom):
                for al in node.names:
                    if al.asname in UTF8_NAMES and al.name in UTF8_NAMES:
                        aliases.append({"file": rel(root, f), "lineno": node.lineno,
                                        "target": al.asname, "source": al.name,
                                        "form": "import-as"})
    body_d = {v["body_digest"] for v in defs.values()}
    full_d = {v["full_digest"] for v in defs.values()}
    weak = sorted(k for k, v in defs.items()
                  if not (v["touches_stderr"] and v["has_errors_replace"]))
    strong_defs = [k for k in defs if k not in weak]
    return {
        "defs": len(defs), "aliases": len(aliases), "alias_detail": aliases,
        "body_distinct": len(body_d), "full_distinct": len(full_d),
        "docstrings_differ": len(full_d) > len(body_d),
        "weak": weak,
        "single_impl": len(strong_defs) == 1 and not weak,
        "identical": len(body_d) <= 1 and not weak,   # 兼容旧键：函数体层面一致且无弱版
        "impls": defs,
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
    """`rl/selftest.py` 的登记对账。

    ⚠️ **T2-8 后必须「分片感知」**：`rl/selftest.py` 已拆成「聚合 + `main()`」，100 个 `test_*`
    搬到了 `rl/selftests/part{1..5}.py` ⇒ 只解析聚合文件会得到「定义 **0** 个」这个**假失败**
    （我第一版就撞上了）。故定义数从 **聚合文件 + 全部分片** 一起数，`main()` 调用清单仍只看聚合文件。
    """
    p = root / "src/clasher_new/rl/selftest.py"
    if not p.is_file():
        return {"error": "rl/selftest.py 不存在"}
    tree = parse(p)
    if tree is None:
        return {"error": "rl/selftest.py 解析失败"}
    defined = {n.name for n in tree.body
               if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")}
    parts = sorted((root / "src/clasher_new/rl/selftests").glob("part*.py")) \
        if (root / "src/clasher_new/rl/selftests").is_dir() else []
    for pf in parts:
        ptree = parse(pf)
        if ptree is None:
            continue
        defined |= {n.name for n in ptree.body
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
        "parts": [rel(root, x) for x in parts],
        "ok": defined == called and has_main,
    }


# ---------------------------------------------------------------- ⑧ bare-reconfigure
#: 检查 ⑧ 的排除项：这两个文件的名字里含 `sys.stdout.reconfigure(` 是**工具自身**的字符串
#: （`io_bootstrap` 只调用 `stream.reconfigure`，不碰 `sys.stdout`；codemod/核对器里是模式串）。
_BARE_SKIP = {"io_bootstrap.py", "_converge_utf8_bootstrap.py", "_structure_check.py"}


def check_bare_reconfigure(root: Path) -> dict:
    """**手写 `sys.stdout.reconfigure(...)` 块是否复活**（T1-1b 的防回归闸）。

    为什么需要：T1-1 把 UTF-8 兜底收敛成**单一实现**（`rl/io_bootstrap.force_utf8_stdout`，
    两路 + `errors="replace"`），但收敛是**逐个文件**做的 —— 手写块只要有人复制回去就会**悄悄**
    重新长出来。手写块的已知缺陷是**只处理 stdout**（⇒ traceback 走 stderr 仍是 GBK 乱码，
    实测过 `duel_search.py`）。判据：**0 个**。
    """
    hits = []
    for f in iter_py(root, "src/clasher_new", recursive=True) + iter_py(root, "scripts", recursive=True):
        if f.name in _BARE_SKIP:
            continue
        txt = read_text(f)
        if "sys.stdout.reconfigure(" in txt:
            ln = next(i for i, l in enumerate(txt.split("\n"), 1) if "sys.stdout.reconfigure(" in l)
            hits.append({"file": rel(root, f), "lineno": ln})
    return {"files": len(hits), "detail": hits, "ok": len(hits) == 0}


# ---------------------------------------------------------------- ⑨ bootstrap 顺序
def check_bootstrap_order(root: Path) -> dict:
    """`from rl.io_bootstrap import ...` 是否排在**一条 `__file__` 依据的 path 引导之后**。

    ⚠️ **只报告、不设门禁**（故不进 `hard`）：判据是"import 之前出现过 `__file__`"这一**启发式**，
    对间接写法（如 `_ROOT = ...__file__...` 另起一行）是准的，但**不排除**更绕的写法 ⇒ 当门禁会假阳性。
    真实事故（2026-09-19）：`_schema5_probe.py` 的 io_bootstrap import 被插在 `os.chdir` 与
    `sys.path.insert` **之间** ⇒ 运行时 `ModuleNotFoundError: No module named 'rl.io_bootstrap'`。
    `rl/` 包内模块**跳过**（它们以 `rl.x` 被导入 ⇒ `rl.io_bootstrap` 天然可导入）。
    """
    warns = []
    # T2-8 教训：同类失败模式在 `rl/selftest.py` 上又发生了一次（`from rl.selftest_common import`
    # 排在 `sys.path.insert` **之前** ⇒ `python rl/selftest.py` 直接 ModuleNotFoundError）⇒ 覆盖面
    # 从 `scripts/` 扩到 `rl/` 与引擎顶层（`__init__.py` 除外：它们本来就在包里）。
    cands = iter_py(root, "scripts", recursive=True) \
        + [x for x in iter_py(root, "src/clasher_new/rl", recursive=False) if x.name != "__init__.py"] \
        + [x for x in iter_py(root, "src/clasher_new", recursive=False) if x.name != "__init__.py"]
    for f in cands:
        txt = read_text(f)
        tree = parse(f)
        if tree is None:
            continue
        imp = None
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module and \
                    "rl.io_bootstrap" in node.module:
                imp = node.lineno
                break
            if isinstance(node, ast.Import):
                for al in node.names:
                    if "rl.io_bootstrap" in al.name:
                        imp = node.lineno
        if imp is None:
            continue
        before = "\n".join(txt.split("\n")[:imp - 1])
        if "__file__" not in before:
            warns.append(rel(root, f))
    return {"files": len(warns), "detail": sorted(warns)}


# ---------------------------------------------------------------- ⑩ scripts/README.md 漂移
#: `scripts/` 的分类规则（**按文件名前缀**，确定性、可复现）。判据 = 生成的 README 必须与磁盘上的
#: 实际文件集一致 ⇒ 新增/删除脚本而忘了更新索引时会 **FAIL**。
#: 分类只影响"读到哪一行"，**不影响任何运行行为**（不是 import 路径、不是包结构）。
SCRIPTS_CATEGORIES = [
    ("① 测试（引擎验收）", ("test_m",),
     "引擎机制验收脚本（`test_m2/m3_evo/m4_evo7/m5_data/m6_elite/m1`）。**不属于** `run_selftests.py` 那套。"),
    ("② 一次性取证 / 归档工具（`_` 前缀）", ("_",),
     "按仓内既有约定：`_` = 不进正式仪器表；多为**一次性**取证/迁移工具，**保留作证据**。"),
    ("③ 探针 / 诊断（可复用仪器）", ("probe_", "diag_", "forensics_", "analyze_", "coverage",
                                    "bench_", "judge_", "phi_", "pomdp_", "value_", "duel_",
                                    "batch_", "summarize_", "health_", "offline_", "random_",
                                    "pass_", "s1_", "s2_", "check_", "finalize_", "kill_",
                                    "run_selftests", "selftest_"),
     "可复跑的判读/诊断仪器（多数已在 `docs/agents/env.md` §2.4 登记）。"),
    ("④ 其它（脚本 / 引擎侧辅助 / 待归类）", (),
     "未落入上面三类者：领域脚本、引擎侧辅助、以及**待归类**项。"),
]


def classify_script(name: str) -> str:
    for label, prefixes, _ in SCRIPTS_CATEGORIES:
        if not prefixes:            # 兜底类放最后
            continue
        if any(name.startswith(x) for x in prefixes):
            return label
    return SCRIPTS_CATEGORIES[-1][0]


def render_scripts_readme(root: Path) -> str:
    """生成 `scripts/README.md`（确定性；由 `--write-scripts-readme` 写盘，由 ⑩ 校验）。"""
    names = sorted(x.name for x in (root / "scripts").glob("*.py"))
    buckets = {label: [] for label, _, _ in SCRIPTS_CATEGORIES}
    for n in names:
        buckets[classify_script(n)].append(n)
    L = ["# `scripts/` 分类索引（**自动生成，勿手改**）", "",
         "> 生成器 / 校验器：`scripts/_structure_check.py`（`--write-scripts-readme` 重写；检查 **⑩** 防漂移）。",
         "> 逐文件的 docstring / `--selftest` / assert 数 / 是否已被 `env.md` 登记 ⇒ 见",
         "> [`../docs/agents/scripts_inventory.md`](../docs/agents/scripts_inventory.md)。", "",
         "## ★ 为什么**没有**把 `scripts/` 拆成子目录（Tier 2 · T2-6 的决定）", "",
         "方案原写「`scripts/` 分目录（tools / probes / tests）」。**实测代价后判定不做**，理由是数字：", "",
         "| 家族 | 文件数 | 文档提及 | 其中活跃索引 | 涉及**历史留证**文档数 |",
         "|---|---:|---:|---:|---:|",
         "| `test_m*.py` | 6 | 193 | 4 | 13 |",
         "| `_survey_*.py` | 9 | 128 | 0 | 8 |",
         "| 其它 `_*.py` | 13 | 138 | 15 | 36 |",
         "| `probe_/diag_/forensics_*` | 30 | 264 | 30 | 59 |",
         "| 其它 | 35 | 552 | 75 | 72 |",
         "",
         "- **全仓 `scripts/<名>.py` 被提及 1,293 次 / 113 份文档**，其中 **~1,167 次落在 `docs/*.md` 的历史留证文档里**；",
         "- 本仓纪律是「新决策先写 docs（可改）」+「**只增不改历史结论**」⇒ 搬迁必然要么**漏改**（文档说谎），",
         "  要么**改写历史**（违反纪律）。**即使只搬最小的家族（`_survey_*` 9 个）也要碰 8 份历史文档**；",
         "- **功能面其实很小**：`.sh` 5 处（`finalize_et_solo100k.sh` / `run_probe_v3.sh` / `_apply_s2_channel_when_idle.sh`）、",
         "  `.bat` 只引用 `scripts/rl/*`（方案本就要求**保持原位**）、`.ps1` 1 处且是散文 ⇒ **风险不在功能面，全在文档面**；",
         "- 而 T2-6 想要的「可寻性」已经由 **T0-3 的 `docs/agents/scripts_inventory.md`**（91 个脚本 + docstring + 4 分类）",
         "  与本文件（分类 + 漂移检查）**给到**，无需搬文件。", "",
         "> 若将来确实要搬：**必须同批**改 `env.md §2.4` / `AGENTS.md` / `docs/README.md` / 3 个 `.sh` /",
         "> `docs/agents/scripts_inventory.md`，并在 98 份历史文档**顶部加一行「路径已迁移」备注**（而不是改写正文）。", "",
         "## 分类", ""]
    for label, prefixes, desc in SCRIPTS_CATEGORIES:
        items = buckets[label]
        L += [f"### {label}（{len(items)} 个）", "", desc, ""]
        L += ["`" + "`, `".join(items) + "`" if items else "（空）", ""]
    L += [f"**合计 {len(names)} 个 `*.py`**（另有 `scripts/rl/*.py` **11 个**：`start_rl.bat` 依赖其位置的入口包装脚本，**不参与**本分类）。", ""]
    return "\n".join(L)


def check_scripts_readme(root: Path) -> dict:
    """检查 ⑩：`scripts/README.md` 是否与磁盘上的脚本集**一致**（防漂移）。"""
    p = root / "scripts" / "README.md"
    want = render_scripts_readme(root)
    if not p.is_file():
        return {"ok": False, "reason": "scripts/README.md 不存在",
                "hint": "python scripts/_structure_check.py --write-scripts-readme"}
    got = p.read_text(encoding="utf-8").replace("\r\n", "\n")
    # 只比"分类清单"部分（正文里的说明段落允许人工润色）
    def lists_of(txt):
        return [ln for ln in txt.split("\n") if ln.startswith("`") and ln.endswith("`")]
    return {"ok": lists_of(want) == lists_of(got), "n_scripts": len(list(x.name for x in (root / "scripts").glob("*.py"))),
            "hint": "python scripts/_structure_check.py --write-scripts-readme",
            "diff": [x for x in lists_of(want) if x not in lists_of(got)][:2]}


CHECKS = {
    "areas": check_areas,
    "utf8": check_utf8,
    "layering": check_layering,
    "deadfiles": check_deadfiles,
    "abspaths": check_abspaths,
    "lazyimp": check_lazy_imports,
    "selftest": check_selftest,
    "bare_reconf": check_bare_reconfigure,
    "boot_order": check_bootstrap_order,
    "scripts_readme": check_scripts_readme,
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
    L.append("== ② UTF-8 兜底收敛：定义 {} 处 / 别名 {} 处 ⇒ 函数体 {} 种（含 docstring {} 种）{}".format(
        u.get("defs"), u.get("aliases"), u.get("body_distinct"), u.get("full_distinct"),
        "**已收敛为单一实现**" if u.get("single_impl") else "**未收敛**")
        + ("（定义：{}）".format(_loc) if _loc else "（已无 def）"))
    if u.get("weak"):
        L.append("     ⚠️ 弱化实现（未管 stderr 或缺 errors='replace'）：{}".format(", ".join(u["weak"])))
    for a in (u.get("alias_detail") or [])[:8]:
        L.append("     别名 {}:{}  {} = {}（{}）".format(
            a["file"], a["lineno"], a["target"], a["source"], a.get("form", "?")))
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
    sr = res.get("scripts_readme", {})
    L.append("== ⑩ scripts/README.md 与脚本集一致 = {}（{} 个 *.py）".format(
        "OK" if sr.get("ok") else "**FAIL/DRIFT**", sr.get("n_scripts")))
    if not sr.get("ok"):
        L.append("     修法：{}".format(sr.get("hint")))
        for d in (sr.get("diff") or []):
            L.append("     期待: " + d[:120])
    bo = res.get("boot_order", {})
    L.append("== ⑨ io_bootstrap import 早于 path 引导（**只报告、不设门禁**）= {} 个".format(bo.get("files")))
    for f in (bo.get("detail") or [])[:8]:
        L.append("     ! " + f)
    br = res.get("bare_reconf", {})
    L.append("== ⑧ 手写 `sys.stdout.reconfigure` 块（应为 0，UTF-8 兜底须走单一实现）"
             + ("  OK" if br.get("ok") else "  **FAIL** {} 个".format(br.get("files"))))
    for h in (br.get("detail") or [])[:10]:
        L.append("     {}:{}".format(h["file"], h["lineno"]))
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
        # ② 三处 _force_utf8_stdout：a/b **函数体逐字相同但 docstring 不同**，c 是**弱化版**
        #    ⇒ 判据 body_distinct 应为 2（强/弱），full_distinct 应为 3（含 docstring 差），
        #    weak 必须点出 c.py。这一条正是 T1-1 实操中「把 docstring 算进指纹会得到误导结论」的回归。
        _strong = ("import sys\n\n\ndef _force_utf8_stdout():\n"
                   "%s    for _s in (sys.stdout, sys.stderr):\n"
                   "        try:\n"
                   "            _s.reconfigure(encoding='utf-8', errors='replace')\n"
                   "        except Exception:\n            pass\n")
        (root / "scripts/a.py").write_text(_strong % '    """甲：完整说明。"""\n', encoding="utf-8")
        (root / "scripts/b.py").write_text(_strong % '    """乙：另一份说明（体相同）。"""\n',
                                           encoding="utf-8")
        (root / "scripts/c.py").write_text(
            "import sys\n\n\ndef _force_utf8_stdout():\n    try:\n"
            "        sys.stdout.reconfigure(encoding='utf-8')\n    except Exception:\n        pass\n",
            encoding="utf-8")
        # 硬编码绝对路径（验 ⑤）
        (root / "scripts/hard.py").write_text(
            "import sys\nsys.path.insert(0, r'E:/x/y/src')\n", encoding="utf-8")
        # ⑧：合成树里放一个**手写 reconfigure 块** ⇒ 必须报 1；随后删掉必须归 0
        #    （否则「0 个」可能只是**空检查** —— 模式根本没匹配过）
        (root / "scripts/bare.py").write_text(
            "import sys\n\ntry:\n    sys.stdout.reconfigure(encoding='utf-8')\n"
            "except Exception:\n    pass\n", encoding="utf-8")
        # 死件复活（验 ④）
        (root / "src/clasher_new/pathfinding.py").write_text("x = 1\n", encoding="utf-8")
        # selftest：定义 2 个，main() 只调 1 个（验 ⑦ 报「未登记」）
        (root / "src/clasher_new/rl/selftest.py").write_text(
            "def test_a():\n    pass\n\n\ndef test_b():\n    pass\n\n\n"
            "def main():\n    test_a()\n", encoding="utf-8")

        res = run_all(root)
        ck("① areas 数到 3 个引擎文件", res["areas"]["engine_top"]["files"] == 3)
        ck("① rl 数到 2 个文件", res["areas"]["rl"]["files"] == 2)
        ck("② a/b 函数体判为同一实现、含 docstring 才分得开",
           res["utf8"]["defs"] == 3 and res["utf8"]["aliases"] == 0
           and res["utf8"]["body_distinct"] == 2
           and res["utf8"]["full_distinct"] == 3 and res["utf8"]["docstrings_differ"])
        ck("② 弱化实现被点名（只碰 stdout 且无 errors=replace）",
           res["utf8"]["weak"] == ["scripts/c.py"])
        ck("② 未收敛时 single_impl=False（不能恒真）", res["utf8"]["single_impl"] is False)

        # ② 第三阶段：模拟「收敛后」—— 只留 1 处强定义 + 2 处别名，判据必须翻成 True
        (root / "scripts/a.py").write_text(
            "import sys\nfrom rl.io_bootstrap import force_utf8_stdout\n"
            "_force_utf8_stdout = force_utf8_stdout\n", encoding="utf-8")
        (root / "scripts/b.py").write_text(
            "import sys\nfrom rl.io_bootstrap import force_utf8_stdout\n"
            "_force_utf8_stdout = force_utf8_stdout\n", encoding="utf-8")
        (root / "scripts/c.py").write_text(
            "import sys\nfrom rl.io_bootstrap import force_utf8_stdout\n"
            "_force_utf8_stdout = force_utf8_stdout\n", encoding="utf-8")
        (root / "scripts/shared.py").write_text(_strong % '    """唯一实现。"""\n', encoding="utf-8")
        res_c = run_all(root)
        ck("② 收敛后：1 处强定义 + 3 处别名 ⇒ single_impl=True",
           res_c["utf8"]["defs"] == 1 and res_c["utf8"]["aliases"] == 3
           and res_c["utf8"]["single_impl"] is True and res_c["utf8"]["weak"] == [])
        ck("③ 反向边 = 1 且能指出文件", res["layering"]["reverse_edges"] == 1
           and res["layering"]["reverse_files"] == ["src/clasher_new/bad.py"])
        # ⑧：合成树里**恰好 2 个**含手写块 —— `bare.py`（本行新建）与 `c.py`（② 的弱化版样板）。
        #    ⚠️ 我第一版断言写成 `files == 1` ⇒ 失败；**核对器是对的**（它确实扫到 2 个）。
        #    教训：断言要按**实际合成树**算，不能按"我只新建了 1 个"想当然。
        _files8 = [d["file"] for d in res["bare_reconf"]["detail"]]
        ck("⑧ 抓到 2 个手写块且含 bare.py（判别力）",
           res["bare_reconf"]["files"] == 2 and "scripts/bare.py" in _files8
           and "scripts/c.py" in _files8)
        (root / "scripts/bare.py").write_text("import sys\n", encoding="utf-8")
        _r8 = run_all(root)
        # ⚠️ 这里必须是 **0** 而不是 1：本自检**前面**的 ② 第三阶段已经把 `c.py` 覆写成了
        #    「别名」形态（不再含手写块）⇒ 删掉 bare.py 后合成树里**一个都不剩**。
        #    （我第一版写成 1 ⇒ 失败。**教训：自检会改写自己的合成树，后面的断言必须按改写后的树算。**）
        ck("⑧ 删掉 bare.py 后归 0（反向验证：不是恒不匹配）",
           _r8["bare_reconf"]["ok"] is True and _r8["bare_reconf"]["files"] == 0)
        # ⑩：合成树里先写 README ⇒ 必须 OK；再加一个脚本 ⇒ **必须报漂移**；删掉 ⇒ 必须回 OK。
        #    ⚠️ 没有这条，⑩ 的「OK」可能只是**空检查**（README 根本不存在 vs 内容真的对得上）。
        (root / "scripts" / "README.md").write_text(
            render_scripts_readme(root), encoding="utf-8")
        _r10a = run_all(root)
        ck("⑩ 生成 README 后判 OK", _r10a["scripts_readme"]["ok"] is True)
        (root / "scripts" / "brand_new_tool.py").write_text("x = 1\n", encoding="utf-8")
        _r10b = run_all(root)
        ck("⑩ 新增脚本后**报漂移**（判别力）", _r10b["scripts_readme"]["ok"] is False)
        (root / "scripts" / "brand_new_tool.py").unlink()
        ck("⑩ 删掉后回 OK（反向验证）", run_all(root)["scripts_readme"]["ok"] is True)
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
    ap.add_argument("--write-scripts-readme", action="store_true",
                    help="重新生成 scripts/README.md（分类索引；检查 ⑩ 会校验它）")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()

    root = Path(args.root).resolve() if args.root else _ROOT
    if args.write_scripts_readme:
        out = root / "scripts" / "README.md"
        out.write_text(render_scripts_readme(root), encoding="utf-8")
        print(f"已重写 {out}")
        return 0
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
            res.get("selftest", {}).get("ok", False),
            res.get("bare_reconf", {}).get("ok", False),
            res.get("scripts_readme", {}).get("ok", False)]
    return 0 if all(hard) else 1


if __name__ == "__main__":
    sys.exit(main())
