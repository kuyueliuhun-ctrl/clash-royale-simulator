#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""AGENTS.md 拆分 + 不丢内容校验（一次性迁移工具，保留作证据）。

背景：`AGENTS.md` 涨到 65,196 B，而已知注入上限 65,244 B（实测）⇒ **余量 48 B**
⇒ 任何一次编辑都可能让末尾被截断（2026-09-13 那次失效模式的复发）。

拆分原则：
  1. **逐字不丢**：原文件每一行非空内容必须**逐字**出现在新 `AGENTS.md` 或某个分册里；
  2. **节号不变**：分册保留原 `## N.` / `### N.M` 编号 ⇒ 历史文档的「`AGENTS.md` §2.1」类引用一跳可达；
  3. **注入文件仍自足于"红线"**：`AGENTS.md` 保留 19 条红线**一行式摘要**（长名逐字），
     全文与事故证据移入 `docs/agents/redlines.md`；
  4. 新增内容（索引表 / 当前活跃工作）单独登记，不计入"原文"。

用法：
    python scripts/_agents_split.py --dry-run     # 打印计划，不写
    python scripts/_agents_split.py --write       # 写分册与新 AGENTS.md
    python scripts/_agents_split.py --check       # 只校验（不丢内容 + 链接可解析 + 尺寸）
"""
from __future__ import annotations

import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENTS = os.path.join(ROOT, "AGENTS.md")
OUTDIR = os.path.join(ROOT, "docs", "agents")

# ---------------------------------------------------------------- 目标分册定义
# 每册 = (文件名, 标题, 原 §编号列表, 一句话说明)
BOOKS = [
    ("redlines.md", "红线全文与事故证据", [1], "19 条红线（违反 = 结论作废或事故）"),
    ("env.md", "环境与命令", [2], "解释器 / venv / 三种模式 / 协议 / 工具 / 仪表盘 / 自检"),
    ("metrics.md", "训练口径与关键常量", [3], "step 语义 / 评估口径 / 噪声地板 / 判读禁则 / 成本 / 常量表"),
    ("ledger.md", "结论台账（已确证 / 已否证 / 未决）", [4, 5, 6], "C1–C13 / X1–X18 / O1–O6"),
    ("plans_runs_docs.md", "计划 · Run · 文档地图", [7, 8, 9], "计划台账 / run 台账 / 文档地图"),
]

HEAD_BACK = """> **来源**：原 `AGENTS.md` {secs}（2026-09-18 拆分，**逐字未改**）。
> **节号保持不变** ⇒ 历史文档里的「`AGENTS.md` §N.M」引用经索引一跳可达。
> 返回索引 → [`../../AGENTS.md`](../../AGENTS.md)
"""


def split_sections(text: str):
    """按 `## N.` 切分，返回 {N: (heading_line, body_text)} 与头部块。"""
    lines = text.split("\n")
    marks = [(i, l) for i, l in enumerate(lines) if re.match(r"^## (\d+)\.", l)]
    head_end = marks[0][0] if marks else len(lines)
    head = "\n".join(lines[:head_end]).rstrip("\n")
    secs = {}
    for j, (idx, line) in enumerate(marks):
        num = int(re.match(r"^## (\d+)\.", line).group(1))
        end = marks[j + 1][0] if j + 1 < len(marks) else len(lines)
        body = "\n".join(lines[idx:end]).rstrip("\n")
        secs[num] = body
    return head, secs


# ---------------------------------------------------------------- 新 AGENTS.md
def new_agents_md(secs) -> str:
    red = secs[1]
    # 从红线表里抽出 `| **R1** | **摘要** |` 两列，逐字复用
    rows = []
    for line in red.split("\n"):
        m = re.match(r"^\|\s*\*\*(R\d+)\*\*\s*\|\s*(.+?)\s*\|", line)
        if m:
            rows.append((m.group(1), m.group(2)))
    assert len(rows) == 19, f"红线条数异常：{len(rows)}"

    # 逐条要点（均为原 detail 里的短片段照抄，非新数字）
    hints = {
        "R1": "症状先归因外部、**不自动降级**；长跑前 `scripts/check_commit.py`；`--eval-workers 12`",
        "R2": "`ppo.py` 默认参数 = 旧行为；新能力只经 `TrainConfig` 显式开",
        "R3": "**判据跑前写死**（含失败分支）",
        "R4": "基线用 `scripts/judge_anchor_blocks.py --groups` **复算**，禁手抄",
        "R5": "20k 单跑只能看**大效应**；跨 run 数字不可直接比",
        "R6": "热启动须显式 `plan_dim=58` / `belief_dim=563`；缺 `enc_ln`/`grid_ln` 会静默错",
        "R7": "汇率 / 值函数 / 闸门 `edw×卡费` / MCTS **同源同步改**",
        "R8": "回归测试必须**跨局边界**；9j 指纹看日志头 30 行",
        "R9": "`step`=决策帧；EV 用**更新前池化**；探针**按局分组留出**",
        "R10": "写「成因未定」，不写自信的错答案",
        "R11": "**不在 A′ 类取证之前改奖励**；不上训练时 MCTS；不扩参",
        "R12": "确定性可算量 → **特征注入**，不让网络猜",
        "R13": "掩码/合法格改动跑 `scripts/_mask_diff_snapshot.py`（128 张**逐位全等**）",
        "R14": "逐窗 `‖ΔW‖/‖W‖`；解读前先确认**前向路径**",
        "R15": "阈值在**本实验自己的量纲**上标定",
        "R16": "阈值余量 **≫ run 间散布**；否则改用**比值/配对**判据",
        "R17": "分子/分母**同超参**；包含关系**逐位验证**",
        "R18": "改代码**同一步**改文档；收尾报告列出改了哪些",
        "R19": "默认只跑相关子集：`run_selftests.py test_<名>`",
    }

    lines = []
    lines.append("# AGENTS — 红线速查 + 决策索引（跨会话必读）\n")
    lines.append("> **本文件 = 红线（每条一行）+ 索引。** 全文/证据/口径/台账在 [`docs/agents/`](docs/agents/)，**按需打开**。")
    lines.append("> **2026-09-18 拆分**：单文件曾涨到 **65,196 B**，而实测注入上限 **65,244 B** ⇒ **余量仅 48 B**，")
    lines.append("> 任何一次编辑都可能让**末尾被截断**（= 2026-09-13 那次失效模式的复发：旧文件末尾 35 KB 对会话不可见）。")
    lines.append("> 现拆为 6 个文件，`AGENTS.md` 只留**红线一行式摘要 + 索引 + 当前活跃工作**。")
    lines.append(">")
    lines.append("> **⚠️ 节号保持不变**：分册里保留原 `## N.` / `### N.M` 编号 ⇒ 历史文档里的「`AGENTS.md` §2.1」")
    lines.append("> 这类引用**经本文索引一跳可达**，不必逐份改写。")
    lines.append("> **纪律不变**：① 新决策**先写 `docs/`** → 再在本文件加**一行指针**；② **只增不改历史结论**")
    lines.append("> （被推翻时保留原条目并标注「已被 X 推翻」）；③ 条目编号可引用：`【红线 R7】` `【确证 C5】` `【否证 X2】` `【未决 O3】`。\n")
    lines.append("---\n")

    lines.append("## A. 红线速查（19 条；**全文 + 事故证据** → [`docs/agents/redlines.md`](docs/agents/redlines.md)）\n")
    lines.append("| # | 红线 | 必记要点 |")
    lines.append("|---|---|---|")
    for rid, name in rows:
        lines.append(f"| **{rid}** | {name} | {hints[rid]} |")
    lines.append("")

    lines.append("---\n")
    lines.append("## B. 决策索引（我要找什么 → 读哪个文件）\n")
    lines.append("| 我要找什么 | 去哪 |")
    lines.append("|---|---|")
    idx = [
        ("**红线全文 + 事故证据（R1–R19）**", "[`docs/agents/redlines.md`](docs/agents/redlines.md) §1"),
        ("解释器 / venv / 统一运行姿势 / GBK 陷阱", "[`docs/agents/env.md`](docs/agents/env.md) §2"),
        ("三种训练模式（`solo` / `run` / `flow` 语义与选择）", "env.md §2.1"),
        ("**标准 20k 协议**（与历史 run 可比）", "env.md §2.2"),
        ("100k 长跑 + 评估节奏 C（`--anchor-every`）", "env.md §2.3"),
        ("判读 / 诊断工具表（含 forensics / probe / 子集 selftest）", "env.md §2.4"),
        ("仪表盘（**端口 8700**；8090 已被系统保留，废）", "env.md §2.5"),
        ("自检命令（【R19】只跑相关子集）", "env.md §2.6"),
        ("训练口径：`step` 语义 / 评估口径 / 噪声地板 / **判读禁则** / 平局与早停 / run 成本", "[`docs/agents/metrics.md`](docs/agents/metrics.md) §3"),
        ("**关键常量表**（改代码前先对照）", "metrics.md §3.1"),
        ("**已确证 C1–C13**（可作前提引用）", "[`docs/agents/ledger.md`](docs/agents/ledger.md) §4"),
        ("**已否证 / 已关闭 X1–X18**（**勿重走**）", "ledger.md §5"),
        ("**未决 O1–O6 + 候选下一步**", "ledger.md §6"),
        ("计划台账（哪条路走通 / 被否 / 未决）", "[`docs/agents/plans_runs_docs.md`](docs/agents/plans_runs_docs.md) §7"),
        ("**Run 台账**（每个 run 一句话结论 + 证据路径）", "plans_runs_docs.md §8"),
        ("**文档地图**（想做什么 → 读哪份 `docs/`）", "plans_runs_docs.md §9"),
        ("历史决策全文（旧 `AGENTS.md` 逐字冻结）", "[`docs/agents_archive_2026-09.md`](docs/agents_archive_2026-09.md)"),
        ("**`long1m` 提前终止的账**（12.8% 的实际状态）", "[`docs/long1m_stopped_2026-09-18.md`](docs/long1m_stopped_2026-09-18.md)"),
        ("**当前新方案**：按局面结算的圣水交换信用分配（§7–§9 规格）", "[`docs/frame_credit_proposal_review_2026-09-18.md`](docs/frame_credit_proposal_review_2026-09-18.md)"),
        ("长跑工程（两级评估 + 评估并行分片）", "[`docs/long1m_prereg_2026-09-18.md`](docs/long1m_prereg_2026-09-18.md)"),
        ("仪表盘对接长跑（进度条 / 大点 / 逐对手胜率）", "[`docs/dashboard_long1m_2026-09-18.md`](docs/dashboard_long1m_2026-09-18.md)"),
    ]
    for a, b in idx:
        lines.append(f"| {a} | {b} |")
    lines.append("")

    lines.append("---\n")
    lines.append("## C. 当前活跃工作（2026-09-18）\n")
    lines.append("1. **主线变更**：`long1m`（1M 步 `run` 模式）**已按用户指令提前终止**在 **128,000 步（12.8%）**——")
    lines.append("   18 个评估点（含插入的大点 100k）、**0 次降级**、**无判读产出且 J1 按定义不成立**。")
    lines.append("   账见 [`docs/long1m_stopped_2026-09-18.md`](docs/long1m_stopped_2026-09-18.md)。")
    lines.append("   ⚠️ `runs/long1m/`（393 MB，18 ckpt + 18 录像）**保留** —— 它是新方案的现成回放素材。")
    lines.append("2. **新方案（实现中）**：按**局面**结算的**圣水交换**信用分配。")
    lines.append("   - 局面 = 索敌关系连通分量；活跃 = 分量内有**交战对**；结算 = 连续 K tick 无交战对（去抖，K=30 tick）")
    lines.append("   - 净交换 ≈ 窗口内 `−ΔΦ` + **塔血保全**（后者是唯一真·新信息）；阈值用**平滑 hinge**")
    lines.append("   - **免费产物只走溯源路由（`root_cast`）、绝不估值**（估值 = 货币泵，见评审 §9.1）")
    lines.append("   - ⚠️ 三条红线级约束：① 绝不估值；② **实验必须挂 `solo`**（`run` 对手是 mask-随机，无拉扯样本）；")
    lines.append("     ③ 判据须带**优势兑现率**门禁（圣水上限 10 ⇒ 只守不推可攒交易打平）")
    lines.append("   - 规格与判据：[`docs/frame_credit_proposal_review_2026-09-18.md`](docs/frame_credit_proposal_review_2026-09-18.md) §7–§9")
    lines.append("")
    return "\n".join(lines)


def do_write():
    text = open(AGENTS, encoding="utf-8").read()
    head, secs = split_sections(text)
    os.makedirs(OUTDIR, exist_ok=True)
    book_of = {}
    for fname, title, nums, desc in BOOKS:
        parts = []
        for n in nums:
            assert n in secs, f"缺 §{n}"
            parts.append(secs[n])
            book_of[n] = fname
        secs_txt = [f"§{n}" for n in nums]
        body = "\n\n".join(parts)
        content = (
            f"# AGENTS 分册 · {title}\n\n"
            + HEAD_BACK.format(secs=" / ".join(secs_txt))
            + f"\n> 本册内容：{desc}。\n\n---\n\n"
            + body
            + "\n"
        )
        with open(os.path.join(OUTDIR, fname), "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  写入 docs/agents/{fname:22s} {len(content.encode()):6d} B  （原 {secs_txt}）")
    newmd = new_agents_md(secs)
    with open(AGENTS, "w", encoding="utf-8") as f:
        f.write(newmd)
    print(f"  写入 AGENTS.md（新）          {len(newmd.encode()):6d} B")
    return text, secs


def do_check(orig_text=None):
    ok = True
    # 1) 尺寸
    size = len(open(AGENTS, "rb").read())
    limit = 65244
    print(f"[1] AGENTS.md = {size} B（上限 {limit} B，余量 {limit - size} B）")
    if size > limit:
        print("    ✗ 超上限"); ok = False

    # 2) 不丢内容：原文每个非空行（除头部块外）必须逐字出现在某处
    if orig_text is None:
        # 从分册 + 新 AGENTS 反推；用 git 里的旧版本
        import subprocess
        try:
            orig_text = subprocess.check_output(
                ["git", "-C", ROOT, "show", "HEAD:AGENTS.md"], text=True, encoding="utf-8")
            print("    （原文取自 git HEAD:AGENTS.md）")
        except Exception as e:  # pragma: no cover
            print(f"    ! 无法取原文（{e}），跳过不丢内容校验"); orig_text = None
    if orig_text:
        head, secs = split_sections(orig_text)
        pool = new_agents_md(secs) if False else open(AGENTS, encoding="utf-8").read()
        # 比对池 = AGENTS.md + **docs/agents/ 下的全部 .md**（不只 BOOKS 里那 5 册）。
        # 2026-09-19 修：原实现只池化 `BOOKS` 的固定 5 个文件名 ⇒ **新增一个分册
        # （如 `structure.md`）会被判「丢内容」**，而本文件的语义明明是
        # 「逐字出现在新 AGENTS.md **或某个分册**里」⇒ 固定清单是实现与语义不符。
        import glob as _glob
        for path in sorted(_glob.glob(os.path.join(OUTDIR, "*.md"))):
            try:
                pool += "\n" + open(path, encoding="utf-8").read()
            except Exception:
                pass
        missing = []
        for n, body in sorted(secs.items()):
            for line in body.split("\n"):
                if line.strip() and line not in pool:
                    missing.append((n, line[:90]))
        print(f"[2] 不丢内容：缺 {len(missing)} 行")
        for n, l in missing[:12]:
            print(f"    ✗ §{n}: {l}")
        if missing:
            ok = False

    # 3) 链接可解析
    md = open(AGENTS, encoding="utf-8").read()
    bad = []
    for m in re.finditer(r"\]\(([^)#]+?)\)", md):
        t = m.group(1).strip()
        if t.startswith("http"):
            continue
        p = os.path.normpath(os.path.join(ROOT, t))
        if not os.path.exists(p):
            bad.append(t)
    print(f"[3] AGENTS.md 内相对链接：坏链 {len(bad)}")
    for b in sorted(set(bad)):
        print(f"    ✗ {b}")
    if bad:
        ok = False

    # 4) 节号齐全
    _, secs_now = split_sections(open(AGENTS, encoding="utf-8").read())
    print(f"[4] 新 AGENTS.md 自带的 `## N.` 节：{sorted(secs_now)}（应为空：内容已全部外移）")
    allsec = set()
    for fname, _, nums, _ in BOOKS:
        body = open(os.path.join(OUTDIR, fname), encoding="utf-8").read()
        allsec |= {int(x) for x in re.findall(r"^## (\d+)\.", body, re.M)}
    print(f"[5] 分册覆盖的原节号：{sorted(allsec)}（应为 1..9）")
    if allsec != set(range(1, 10)):
        print("    ✗ 节号不全"); ok = False

    print("\n" + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.dry_run:
        text = open(AGENTS, encoding="utf-8").read()
        head, secs = split_sections(text)
        print(f"原文 {len(text.encode())} B；头部块 {len(head.encode())} B")
        book_of = {}
        for n in sorted(secs):
            print(f"  §{n}: {len(secs[n].encode()):6d} B")
        for fname, title, nums, desc in BOOKS:
            tot = sum(len(secs[n].encode()) for n in nums)
            print(f"  → docs/agents/{fname:22s} 收 §{nums} 共 {tot} B")
        newmd = new_agents_md(secs)
        print(f"  → 新 AGENTS.md 预估 {len(newmd.encode())} B")
        nred = len(re.findall(r'^\| \*\*R\d+\*\*', secs[1], re.M))
        print(f"  红线条数 = {nred}")
        return 0
    if a.write:
        do_write()
        print("--- 校验 ---")
        return do_check()
    if a.check:
        return do_check()
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
