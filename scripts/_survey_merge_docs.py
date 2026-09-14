#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 A/B 两部分草稿合并成《训练方法文档》/《游戏引擎文档》（Markdown）。

不做二次创作：正文逐字保留，只做
  * 标题层级下移一级（草稿的 `#` → 文档的 `##`，以便本文档只有一个 H1）；
  * 生成静态目录（H2/H3）；
  * 追加「文档说明与方法」「待确认事项汇总」「生成方式」三个统一样式章节。

用法：
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/_survey_merge_docs.py \
        --kind training --out docs/training_method.md
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/_survey_merge_docs.py \
        --kind engine   --out docs/game_engine.md
"""
from __future__ import annotations

import argparse
import io
import os
import re
import sys
from typing import Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DRAFTS = os.path.join(ROOT, "docs", "_survey", "drafts")

H_RE = re.compile(r"^(#{1,6})\s+(.*)$")

CONF: Dict[str, Dict] = {
    "training": {
        "title": "训练方法文档",
        "drafts": ["training_A.md", "training_B.md"],
        "intro": [
            "本文档说明**这个项目里的模型是怎么被训练出来的**：训练入口、每一个训练环节调用了哪个函数、",
            "这些函数的参数是什么、超参数在哪里定义、评估与诊断怎么算、以及全部相关函数的索引。",
            "",
            "**资料来源**：`src/clasher_new/rl/` 下的源码 + 逐文件函数级分析素材（`docs/_survey/parts/*.md`）。",
            "**不引用**任何设计文档、计划文件或外部资料。",
        ],
        "scope_table": [
            ("训练相关源码", "`src/clasher_new/rl/` 全部 33 个 `.py` 文件（PPO、环境封装、策略网络、对手池、评估、诊断、辅助训练脚本）"),
            ("引擎接口", "`src/clasher_new/rl/env_wrapper.py` 调用的 `battle.py` / `action_mask.py` 接口（仅涉及训练接口部分）"),
            ("函数索引", "§B.8（本文件内，含「函数名 → 文件:行号」字母序快速索引）"),
            ("配套文档", "《项目内容全解文档》`docs/project_full_reference.md`（逐文件全量函数说明）、《游戏引擎文档》`docs/game_engine.md`"),
        ],
        "howto_index": "按 `Ctrl+F` 搜函数名即可；先用 §B.8 的字母序快速索引定位 `文件:行号`。",
    },
    "engine": {
        "title": "游戏引擎文档",
        "drafts": ["engine_A.md", "engine_B.md"],
        "intro": [
            "本文档说明**这个项目的游戏引擎是怎么实现的**：地图与坐标、实体与战斗主循环、皇家塔与胜负判定、",
            "寻路、索敌、卡牌机制与法术、以及部署合法性判定。",
            "",
            "**资料来源**：`src/clasher_new/` 下的引擎源码（`battle.py`、`arena.py`、`core.py`、`player.py`、",
            "`pathfinding*.py`、`card_mechanics.py`、`spell_module.py`、`threat_calc.py`、`card_utils.py`、",
            "`rl/action_mask.py` 等）+ 逐文件函数级分析素材。**不引用**任何设计文档、计划文件或外部资料。",
        ],
        "scope_table": [
            ("地图", "§A.2（场地尺寸、坐标系统、桥/河/塔位、部署区）"),
            ("寻路", "§B.1–B.4（两套寻路实现、通行性网格、A* 代价与启发式）"),
            ("索敌", "§B.5（视野/射程、目标选择、锁定与重选、威胁评估）"),
            ("战斗与实体", "§A.3–A.5（实体体系、`step(dt)` 逐阶段、伤害与命中）"),
            ("胜负判定", "§A.6（塔血量、摧毁效果、`winner`/`game_over` 全部置位点）"),
            ("卡牌与法术", "§B.6–B.7（机制类组织、法术落点与命中公式）"),
            ("部署合法性", "§B.8（`legal_cells`/`_position_legal` 与引擎侧校验的一致性）"),
            ("配套文档", "《项目内容全解文档》`docs/project_full_reference.md`、《训练方法文档》`docs/training_method.md`"),
            ("一致性取证", "`docs/mask_vs_engine_reconcile_2026-09-14.log` —— 掩码 `legal_cells` 与引擎 `deploy_card` 的**逐格双向对账**（18×32×2 方 × 24 张卡），脚本 `scripts/_mask_vs_engine_reconcile.py`（只读）"),
        ],
        "howto_index": "按 `Ctrl+F` 搜函数名或常量名；引擎常量（格子尺寸、塔血量、代价表）都在 §A.2/§B.3/§B.5 的表格里。",
    },
}


def read_draft(name: str) -> str:
    p = os.path.join(DRAFTS, name)
    with io.open(p, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def shift_headings(txt: str) -> Tuple[str, List[Tuple[int, str]]]:
    """标题整体下移一级，返回 (新文本, [(级别, 标题)])"""
    out: List[str] = []
    heads: List[Tuple[int, str]] = []
    for ln in txt.splitlines():
        m = H_RE.match(ln)
        if m:
            lvl = min(6, len(m.group(1)) + 1)
            title = m.group(2).strip()
            heads.append((lvl, title))
            out.append("#" * lvl + " " + title)
        else:
            out.append(ln)
    return "\n".join(out), heads


def extract_pending(txt: str) -> List[Tuple[str, List[str]]]:
    """抽出所有「待确认清单」小节里的表格行，按 (所属标题, 表格行列表) 返回。"""
    res: List[Tuple[str, List[str]]] = []
    cur_title: str | None = None
    cur_rows: List[str] = []
    in_tbl = False
    for ln in txt.splitlines():
        m = H_RE.match(ln)
        if m:
            if cur_title and cur_rows:
                res.append((cur_title, cur_rows))
            cur_title = m.group(2).strip() if "待确认" in m.group(2) else None
            cur_rows, in_tbl = [], False
            continue
        if cur_title is None:
            continue
        if ln.strip().startswith("|"):
            if re.match(r"^\|[\s:|-]+\|$", ln.strip()):
                in_tbl = True
                continue
            if in_tbl:
                cur_rows.append(ln.strip())
    if cur_title and cur_rows:
        res.append((cur_title, cur_rows))
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=sorted(CONF))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = CONF[args.kind]

    bodies: List[str] = []
    heads_all: List[Tuple[int, str]] = []
    pending: List[Tuple[str, List[str]]] = []
    for name in cfg["drafts"]:
        txt = read_draft(name)
        txt = re.sub(r"^\s*(PART_[AB]_OK|TRAIN_[AB]_OK).*$", "", txt, flags=re.M).strip()
        shifted, heads = shift_headings(txt)
        bodies.append(shifted)
        heads_all += [h for h in heads if h[0] <= 3]
        pending += extract_pending(txt)

    L: List[str] = []
    W = L.append
    W(f"# {cfg['title']}")
    W("")
    for para in cfg["intro"]:
        W(para)
    W("")
    W("## 目录")
    W("")
    W("- [0. 文档说明与方法](#0-文档说明与方法)")
    for lvl, title in heads_all:
        anchor = re.sub(r"[^\w\u4e00-\u9fff -]", "", title).strip().lower().replace(" ", "-")
        if lvl == 2:
            W(f"- [{title}](#{anchor})")
        elif lvl == 3:
            W(f"  - [{title}](#{anchor})")
    W("- [附录 A：待确认事项汇总](#附录-a待确认事项汇总)")
    W("- [附录 B：生成方式与可复现性](#附录-b生成方式与可复现性)")
    W("")
    W("---")
    W("")
    W("## 0. 文档说明与方法")
    W("")
    W("### 0.1 覆盖范围")
    W("")
    W("| 项 | 内容 |")
    W("|---|---|")
    for k, v in cfg["scope_table"]:
        W(f"| {k} | {v} |")
    W("")
    W("### 0.2 资料来源与「不编造」的保证")
    W("")
    W("1. 全部内容来自**源码本身**：分析过程由若干子代理执行，每个子代理只读它被指派的源码文件，")
    W("   **禁止**阅读设计文档 / 计划文件 / README / `AGENTS.md`，**禁止**上网。")
    W("2. 每条机制描述与数值都带**行内来源标注**，格式 `（文件名:行号）`，可直接回源核对。")
    W("3. 读不懂、依赖模块未读到、行为只能推测的地方，一律显式写 "
      "**「待确认：…（原因：…）」**，并汇总到附录 A——**宁可留白，不许猜测**。")
    W("4. 函数清单是先用 Python `ast` 机械抽取（文件/函数名/签名/行号），再逐个补写作用与实现，")
    W("   因此**不会漏函数**；覆盖对账见《项目内容全解文档》§4。")
    W("")
    W("### 0.3 阅读约定与索引方式")
    W("")
    W("| 标记 | 含义 |")
    W("|---|---|")
    W("| `（xxx.py:123）` | 该结论的源码出处：文件与行号 |")
    W("| **待确认** | 未能从源码确认，不得当作事实引用 |")
    W("| `A.x` / `B.x` | 章节号；A 与 B 是本文档的两个部分（编号保留 A/B 前缀，以保证文中相互引用不会错位）|")
    W("")
    W(f"**怎么找函数**：{cfg['howto_index']}")
    W("")
    W("---")
    W("")
    for b in bodies:
        W(b)
        W("")
        W("---")
        W("")
    W("## 附录 A：待确认事项汇总")
    W("")
    W("下列条目是各部分**显式标注为无法从源码确认**的内容，集中列在这里以防被误当作事实。")
    W("")
    if not pending:
        W("（无——各部分未产生待确认项。）")
    for title, rows in pending:
        W(f"### {title}")
        W("")
        W("| # | 条目 | 无法确认的原因 | 要确认需要什么 |")
        W("|---|---|---|---|")
        for i, r in enumerate(rows, start=1):
            cells = [c.strip() for c in r.strip().strip("|").split("|")]
            # 跳过表头行（首列是 "#"）
            if i == 1 and cells and cells[0] == "#":
                continue
            if len(cells) >= 4:
                body = " | ".join(cells[1:4])
                if not body.strip(" |"):
                    body = " | ".join(cells[:3])
            else:
                body = " | ".join(cells)
            W(f"| {i} | {body} |")
        W("")
    W("---")
    W("")
    W("## 附录 B：生成方式与可复现性")
    W("")
    W("```bash")
    W("cd <仓库根目录>")
    W("PY=.venv/Scripts/python.exe")
    W("# 1) 逐文件摸底（AST 清单 + 分组 + 子代理分析 + 覆盖对账）")
    W("PYTHONIOENCODING=utf-8 $PY scripts/_survey_inventory.py --out docs/_survey/inventory.json")
    W("PYTHONIOENCODING=utf-8 $PY scripts/_survey_groups.py --inventory docs/_survey/inventory.json --out docs/_survey/groups.json")
    W("PYTHONIOENCODING=utf-8 $PY scripts/_survey_verify.py")
    W("# 2) 合并三份交付物")
    W("PYTHONIOENCODING=utf-8 $PY scripts/_survey_merge.py         # -> docs/project_full_reference.md")
    W("PYTHONIOENCODING=utf-8 $PY scripts/_survey_merge_docs.py --kind training --out docs/training_method.md")
    W("PYTHONIOENCODING=utf-8 $PY scripts/_survey_merge_docs.py --kind engine   --out docs/game_engine.md")
    W("# 3) 生成 DOCX")
    W("PYTHONIOENCODING=utf-8 $PY scripts/_survey_md_to_docx.py --md docs/training_method.md --docx docs/training_method.docx --title 训练方法文档")
    W("PYTHONIOENCODING=utf-8 $PY scripts/_survey_md_to_docx.py --md docs/game_engine.md     --docx docs/game_engine.docx     --title 游戏引擎文档")
    W("```")
    W("")
    W("| 源材料 | 说明 |")
    W("|---|---|")
    W("| `docs/_survey/inventory.json` | AST 先验清单（每个 `.py` 的全部符号/签名/行号）|")
    W("| `docs/_survey/parts/*.md` | 48 份逐文件分析素材（本文件正文的来源）|")
    W("| `docs/_survey/drafts/` | 本文档 A/B 两部分草稿 |")
    W("| `docs/_survey/coverage_report.json` | 覆盖对账结果 |")
    W("")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with io.open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[merge_docs:{args.kind}] wrote {os.path.relpath(args.out, ROOT)} "
          f"({os.path.getsize(args.out)/1024:.0f} KB, {len(L)} lines, pending_sections={len(pending)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
