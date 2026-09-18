#!/usr/bin/env python3
"""训练健康曲线读数：**价值损失**与**策略熵**（稠密，逐 update）+ 平台（步数瓶颈）描述性读数。

    python scripts/health_curve.py --log docs/train_et_solo100k.log
    python scripts/health_curve.py --log docs/train_et_solo100k.log --log-b docs/train_ctrl100k.log
    python scripts/health_curve.py --log <log> --json out.json      # dashboard /api/health 同源
    python scripts/health_curve.py --selftest                       # 【R19】子集自检

## 这两个指标不是"新增"的

计算在 ``rl/ppo.py::_loss_pass``（早就有），落盘在**每个 update 一行**：solo = ``[solo step N]``
（`train_solo.py:1704`）、run/flow = ``[step N]``（`run_league.py:1158`）=> 100k 步约 **781 点**。
``solo_state.json`` 的 ``history[]``（逐评估点 14 点、32 键）**没有**这两项 => 只在 UI 上一直不可见。
本脚本 = 把它们**取出来**并给出可审计的读数；口径见 ``rl/train_health.py`` 模块 docstring。

## 三层读数，别混用（【R9】【R17】）

| 读数 | 窗口 | 是不是预注册判据 |
|---|---|---|
| **局部平台**（本脚本 ``--param`` 主输出） | 末窗 vs **紧邻前一窗**，阈值 k×前窗 MAD | **不是**（本模块自定；回答"最近还动不动"） |
| **within-run**（§11.13.2 口径） | 本 run **前 100 点**中位 ± k×MAD，看其后点数 | 是，但**只**预注册给了 6 个 advinert 指标 |
| **跨臂配对**（§11.13.4 注算子） | `diff=|median_later(A)-median_later(B)|` vs `spread=max(MAD_A,MAD_B)` | 算子是预注册的，但**指标不在**那六项里 |

[!] **不得**把本脚本的输出并入 ``engagement_trade_prereg §11.13`` 的判据集：那是在开跑前写死的，
事后把新指标塞进判据集就是【R3】违规。此处所有输出一律标注"**描述性、非判据**"。
"""

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SRC = os.path.join(_ROOT, "src", "clasher_new")
sys.path.insert(0, _HERE)   # import judge_critic_inertia（复用 §11.13.2 within-run 口径）
sys.path.insert(0, _SRC)    # import rl.train_health

# GBK 陷阱（本仓已知，env.md §2；与 judge_critic_inertia.py:19 / et_solo100k_readout.py:32
# **同一个修法**）：Windows python 的 stdout/stderr 默认 = **gbk**，且**实测** `PYTHONIOENCODING=utf-8`
# 在本环境**不生效**（加了它 stdlib 仍自报 stdout.encoding=gbk）。后果不只是乱码：任何非 GBK 字符
# （⇒ / ⚠ / 上标 2 / U+2212 减号）都会 UnicodeEncodeError **直接打断读数**。
# 实测证据（2026-09-18）：不 reconfigure 时本脚本前 3 行输出是 **GBK 字节**、而
# `import judge_critic_inertia`（它在模块顶层 reconfigure 成 utf-8）**之后**的行变成 **UTF-8 字节**
# —— 同一个流里两种编码，就是这个坑的指纹。故此处显式 encoding="utf-8"（与另两个脚本一致）。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from rl.train_health import (HEALTH_FIELDS, blocks, health_summary, mad,  # noqa: E402
                            parse_line, parse_log, plateau_read)

_NAME = {"entropy": "策略熵(内积 nat)", "value_loss_raw": "价值损失(原始 MSE)",
         "value_loss": "value=（÷v_scale^2，跨时间不可比）", "policy_loss": "policy loss"}


def _f(x, nd=4):
    return "n/a" if x is None else f"{x:.{nd}f}"


def _print_series_table(label, bl, nd=4):
    print(f"\n  [{label}] 分段（按点数等分；step 范围 / 中位 / 原始MAD）")
    print(f"    {'step 区间':<24}{'中位':>10}{'MAD':>10}{'n':>6}")
    for b in bl:
        print(f"    {str(b['step_lo']) + '-' + str(b['step_hi']):<24}"
              f"{_f(b['median'], nd):>10}{_f(b['mad'], nd):>10}{b['n']:>6}")


def _print_read(label, r, nd=4):
    v = r["verdict"]
    zh = {"plateau": "平台（末窗中位落在前窗 k×MAD 带内）",
          "moving_down": "仍在下降（越出带）", "moving_up": "仍在上升（越出带）",
          "unresolvable": "不可判（退化/点数不足）"}[v]
    t, b = r["tail"], r["base"]
    print(f"\n  [{label}] 局部平台读数：**{zh}**")
    if t and b:
        print(f"    末窗 step {t['step_lo']}-{t['step_hi']} (n={t['n']}) 中位={_f(t['median'], nd)}"
              f" | 前窗 step {b['step_lo']}-{b['step_hi']} (n={b['n']}) 中位={_f(b['median'], nd)}"
              f" MAD={_f(b['mad'], nd)}")
        print(f"    diff={_f(r['diff'], nd)}  阈值 k×MAD={_f((r['k'] * (r['spread'] or 0)), nd)}"
              f"  snr=|diff|/MAD={_f(r['snr'], 2)}  后半斜率={_f(r['slope_per_10k'], nd)}/10k步")
    if r.get("why"):
        print(f"    理由：{r['why']}")


def _print_within(rows, key, k, n_base=100):
    """§11.13.2 within-run 带（复用 judge_critic_inertia 的公式，不另写一份实现）。"""
    try:
        import judge_critic_inertia as jci
    except Exception as e:  # pragma: no cover
        print(f"    （within-run 跳过：无法 import judge_critic_inertia: {e}）")
        return None
    w = jci.within_run_baseline(rows, n_base=n_base, k=k, keys=(key,))
    m = w["metrics"][key]
    nb = w["n_base"]
    print(f"    within-run（§11.13.2 口径；【R4】脚本复算）：前 {nb} 点 中位={_f(m['base_median'])}"
          f" MAD={_f(m['base_mad'])} => 带 [{_f(m['lo'])}, {_f(m['hi'])}]；"
          f"后续 {m['later_n']} 点中位={_f(m['later_median'])}，带内 {m['later_inside']}/{m['later_n']}"
          + ("  [!]退化(MAD=0)" if m["degenerate"] else ""))
    return w


def _paired(rows_a, rows_b, key, k, n_base=100):
    """§11.13.4 注的**算子**（diff vs spread=max(MAD_A,MAD_B)，MAD 取前 n_base 点，原始 MAD）。

    [!] 算子照抄预注册，但 **entropy/value_loss 不在** 预注册那 6 个指标里 => 本表**不是**
    §11.13.4-3 的判决，只是把同一把尺子借来量这两个新量（描述性）。
    """
    import judge_critic_inertia as jci
    wa = jci.within_run_baseline(rows_a, n_base=n_base, k=k, keys=(key,))["metrics"][key]
    wb = jci.within_run_baseline(rows_b, n_base=n_base, k=k, keys=(key,))["metrics"][key]
    la, lb = wa["later_median"], wb["later_median"]
    ma, mb = wa["base_mad"], wb["base_mad"]
    if la is None or lb is None or ma is None or mb is None:
        return None
    deg = (ma == 0.0) or (mb == 0.0)
    diff = abs(la - lb)
    spread = max(ma, mb)
    return {"key": key, "later_median_A": la, "later_median_B": lb, "diff": diff,
            "spread": spread, "degenerate": deg,
            "indistinguishable": (None if deg else diff < spread)}


def _selftest():
    """合成日志自检（不依赖引擎）：三条分支 + 口径对账。"""
    ok = True
    hdr = ("[solo step {st}] policy=0.0100 value=0.0010 vraw=1.00 EVb=-2.000 EVin=+0.010 "
           "entropy={e:.4f} | deploy=9.0% bundle=0.09 ratio=0.980 clip=5.0% gs=16 "
           "adv=-1.000±2.000 gnorm=1.00 | p_gnorm=1.0 v_gnorm=0.1 v/p=0.10 scale=2.0 n=128")

    def rows_of(es):
        return [parse_line(hdr.format(st=(i + 1) * 128, e=e)) for i, e in enumerate(es)]

    # 1) 平台：常数 + 微噪声
    es = [0.5 + (0.001 if i % 2 else -0.001) for i in range(100)]
    r = plateau_read(rows_of(es), "entropy", k=3.0)
    t1 = r["verdict"] == "plateau"
    print(f"[{'PASS' if t1 else 'FAIL'}] 平台分支：常数+微噪声 => verdict={r['verdict']}（期望 plateau）")
    ok &= t1

    # 2) 仍在下降：前 92 点 0.5，末 8 点 0.2（带噪声保证 MAD>0）
    es = [0.5 + (0.001 if i % 2 else -0.001) for i in range(92)] + \
         [0.2 + (0.001 if i % 2 else -0.001) for i in range(8)]
    r = plateau_read(rows_of(es), "entropy", k=3.0)
    t2 = r["verdict"] == "moving_down"
    print(f"[{'PASS' if t2 else 'FAIL'}] 下降分支：末 8 点阶跃下移 => verdict={r['verdict']}"
          f"（期望 moving_down）diff={_f(r['diff'])} 3×MAD={_f(3 * (r['spread'] or 0))}")
    ok &= t2

    # 3) 退化：恒定 => MAD=0 => 不可判
    r = plateau_read(rows_of([0.42] * 100), "entropy", k=3.0)
    t3 = r["verdict"] == "unresolvable"
    print(f"[{'PASS' if t3 else 'FAIL'}] 退化分支：恒定序列 => verdict={r['verdict']}（期望 unresolvable）")
    ok &= t3

    # 4) mad() 与 jci.mad() 逐值对账（同一公式的两份实现必须一致）
    try:
        import judge_critic_inertia as jci
        import random
        rnd = random.Random(0)
        same = True
        for _ in range(20):
            xs = [rnd.gauss(0, 1) for _ in range(37)]
            if mad(xs) != jci.mad(xs):
                same = False
        t4 = same
    except Exception as e:
        t4 = False
        print(f"      （对账异常：{e}）")
    print(f"[{'PASS' if t4 else 'FAIL'}] mad() 与 judge_critic_inertia.mad() 逐值相同（20 组随机）")
    ok &= t4

    # 5) 两种日志格式都能解析；run 模式无 vraw
    rs = parse_line(hdr.format(st=128, e=0.5))
    rr = parse_line("[step 128] policy=0.0100 value=0.0010 entropy=0.5000")
    t5 = (rs["mode"] == "solo" and rs["value_loss_raw"] == 1.00 and rs["value_scale"] == 2.0
          and rr["mode"] == "run" and rr["entropy"] == 0.5 and rr["value_loss_raw"] is None)
    print(f"[{'PASS' if t5 else 'FAIL'}] 解析两种格式；run 模式 value_loss_raw=None（无 vraw 可还原）")
    ok &= t5
    print(f"\n{'ALL PASS' if ok else 'FAILED'} (5/5 期望)")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="价值损失 / 策略熵稠密曲线 + 平台（步数瓶颈）描述性读数")
    ap.add_argument("--log", type=str, default=None, help="训练日志（solo 或 run 模式皆可）")
    ap.add_argument("--log-b", type=str, default=None, help="对照臂日志（给则出跨臂配对表）")
    ap.add_argument("--param", type=str, default=None,
                    help="指标，逗号分隔；缺省 = entropy,value_loss_raw[,value_loss]")
    ap.add_argument("--k", type=float, default=3.0, help="阈值倍数（默认 3；【R16】阈值须另行标定）")
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--n-base", type=int, default=100, help="within-run/配对的基线点数（§11.13.2 = 100）")
    ap.add_argument("--tail-bytes", type=int, default=None, help="只读日志末尾 N 字节（实时 UI 用）")
    ap.add_argument("--json", type=str, default=None, help="把 health_summary 落盘（dashboard 同源）")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return _selftest()
    if not a.log:
        ap.error("需要 --log（或 --selftest）")

    rows, info = parse_log(a.log, tail_bytes=a.tail_bytes)
    if not rows:
        print(f"[health] 没解析到任何训练步：{info['path']}"
              f"（step 行 {info['n_step_lines']}，跳过 {info['n_skipped']}）")
        return 2
    params = ([p.strip() for p in a.param.split(",") if p.strip()] if a.param
              else [f for f in HEALTH_FIELDS if any(r[f] is not None for r in rows)]
              + (["value_loss"] if rows[0]["mode"] == "run" else []))
    dens = (info["step_last"] - info["step_first"]) / max(1, len(rows) - 1)

    print("=" * 100)
    print(f"[health] {info['path']}")
    print(f"  模式={info['mode']}  点数={info['n_rows']}  step {info['step_first']}→{info['step_last']}"
          f"  平均每 {dens:.0f} 步一个点（≈ 1 update/行）  vraw 可用={info['has_value_loss_raw']}")
    print(f"  指标={params}  k={a.k}  （[!] 描述性读数，**非**预注册判据；见文件头三层读数表）")
    if info["mode"] == "run":
        print("  [!] run 模式日志没有 vraw/scale => 只有 ÷v_scale^2 后的 value=，**跨时间不可比**")
    for p in params:
        if not any(r[p] is not None for r in rows):
            print(f"\n  [{p}] 该日志无此字段，跳过")
            continue
        print(f"\n{'─' * 100}\n  指标：{_NAME.get(p, p)}")
        _print_series_table(p, blocks(rows, p, a.blocks),
                            nd=2 if p == "value_loss_raw" else 4)
        _print_read(p, plateau_read(rows, p, k=a.k))
        _print_within(rows, p, a.k, n_base=a.n_base)

    if a.log_b:
        rb, ib = parse_log(a.log_b, tail_bytes=a.tail_bytes)
        if not rb:
            print(f"\n[health] 对照日志没解析到训练点：{ib['path']}")
        else:
            print(f"\n{'=' * 100}\n[health] 跨臂配对（A={os.path.basename(info['path'])} "
                  f"B={os.path.basename(ib['path'])}）")
            print("  [!] 算子抄自 §11.13.4 注（diff vs spread=max(MAD_A,MAD_B)，MAD 取前 "
                  f"{a.n_base} 点）——但**这两个指标不在**预注册六项内 => 此表**不是**失败分支 3 的判决")
            print(f"  {'指标':<26}{'A 后续中位':>13}{'B 后续中位':>13}{'diff':>10}{'spread':>10}  判定")
            for p in params:
                if not (any(r[p] is not None for r in rows) and any(r[p] is not None for r in rb)):
                    continue
                c = _paired(rows, rb, p, a.k, n_base=a.n_base)
                if c is None:
                    print(f"  {_NAME.get(p, p):<26}{'n/a':>13}{'n/a':>13}{'n/a':>10}{'n/a':>10}  数据不足")
                    continue
                vd = ("[!]退化(不判)" if c["degenerate"]
                      else ("不可分辨" if c["indistinguishable"] else "可分辨"))
                print(f"  {_NAME.get(p, p):<26}{c['later_median_A']:>13.4f}{c['later_median_B']:>13.4f}"
                      f"{c['diff']:>10.4f}{c['spread']:>10.4f}  {vd}")

    if a.json:
        s = health_summary(a.log, k=a.k, n_blocks=a.blocks, tail_bytes=a.tail_bytes)
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(s, f, ensure_ascii=False, indent=1)
        print(f"\n[health] JSON（dashboard 同源）→ {os.path.abspath(a.json)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
