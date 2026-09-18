"""训练健康曲线：从训练日志取「价值损失」与「策略熵」两条**稠密**曲线 + 平台读数。

## 为什么是"从日志取"而不是"新加指标"

这两个量**早就在算**，也**早就在打**：
- 计算：``rl/ppo.py::PPOTrainer._loss_pass`` 返回 ``stats["policy_loss" | "value_loss" |
  "value_loss_raw" | "entropy"]``（`reduction="mean"` 分支逐小批 ``ent.mean()``，再按
  小批样本数加权平均跨全部轮次，见 ``_update_epochs`` 的统计口径 docstring）；
- 落盘：**每个 update 一行** —— solo 走 ``[solo step N]``（`train_solo.py:1704`），
  run/flow 走 ``[step N]``（`run_league.py:1158/1298/1493`）。
  1 update = ``update_interval`` 128 决策帧 => 100k 步约 **781 点**。

而 dashboard 读的 ``solo_state.json`` 里 ``history[]`` 是**逐评估点**（14 点）且 32 键中
**没有** entropy / value_loss（2026-09-18 实测）=> 这两个量在 UI 上一直不可见。

=> 缺的**不是计算**，是「取出来看 + 怎么判」。本模块负责"取出来"，判据设计见
``docs/train_health_metrics_2026-09-18.md``。

## 口径（判读前必读，【R9】）

``entropy`` = **一次决策内所有 decoder 步的（掩码后）类别分布熵之和**，单位 nat：

    H = Σ_j H(π(slot_j)) + Σ_{j: deploy} H(π(cell_j))

（``follower.py::evaluate_batch`` L647 ``slot_dist.entropy()`` / L677 ``cell_dist.entropy()``；
logits 先 ``masked_fill(mask==0, -1e9)`` => 只在**合法**动作集上求熵，**不含**"学会别选非法动作"
这种平凡下降。）三条后果：

1. 它是**求和**，随 bundle 长度（decoder 步数）变化 => 与"锐度"有非零混杂；
2. 合法集大小随局面变化，而**局面分布本身由策略决定**（自对弈）=> 跨时间的熵变化同时含
   「策略变尖」与「走到了不同局面」两种成分，**不是**纯策略锐度；
3. 它是在 **plan 软偏置之后**（``self._plan_biases``）的分布上算的。

``value_loss`` 有**两个**，日志里都打，别混用：

- ``value=`` = 原始 MSE ÷ ``v_scale^2``（``value_norm=running`` 时 ``v_scale`` 随训练增长，
  2026-09-18 的 A_et 从 1.75 长到 24.87）=> **不是跨时间可比的量**，会"假性下降"；
- ``vraw=`` = 原始 MSE（未缩放）=> **趋势判读一律用 vraw**；只有 solo 模式打 vraw，
  run 模式（``[step N]``）只有缩放后的 ``value=``，**无法**还原 vraw。

## 不构成判决

本模块的 ``plateau_read`` 是**描述性**读数，**不是**预注册判据：它没有阳性对照、不在
任何 run 的预注册判据集里（尤其**不得**并入 ``engagement_trade_prereg`` §11.13 的判据集——
那是在开跑前写死的，事后加指标进去就是【R3】违规）。阈值/窗口**必须在**下一次 run **开跑前**
写死并带多 seed 标定（【R15】【R16】）。
"""

import math
import os
import re

__all__ = [
    "parse_log", "parse_line", "blocks", "mad", "plateau_read",
    "health_summary", "derive_train_log", "HEALTH_FIELDS",
]

# 只取「键=数值」对；键前不吃 `/`（否则 `v/p=12.06` 会被误读成键 `p`）
_KV = re.compile(r"(?<![A-Za-z0-9_/])([A-Za-z_][A-Za-z0-9_]*)=([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)")
_SOLO_HEAD = re.compile(r"^\[solo step (\d+)\]")
_RUN_HEAD = re.compile(r"^\[step (\d+)\]")

# 可以安全跨时间比较、用于判读平台的两条（vraw 而非 value；见模块 docstring）
HEALTH_FIELDS = ("entropy", "value_loss_raw")


def _num(d, k):
    v = d.get(k)
    return None if v is None else float(v)


def parse_line(line):
    """一行 → dict；不是训练步日志则返回 None。

    键名统一（solo / run 两种格式共用）：

    ``step`` ``mode`` ``policy_loss`` ``value_loss``(=日志 value=，缩放后) ``value_loss_raw``
    (仅 solo 有) ``value_scale``(仅 solo) ``entropy`` ``deploy_pct`` ``bundle_mean``
    ``clip_pct`` ``ratio_mean`` ``gnorm`` ``evb`` ``evin`` ``adv_mean`` ``adv_std`` ``n``
    """
    s = line.strip()
    m = _SOLO_HEAD.match(s)
    mode = "solo"
    if m is None:
        m = _RUN_HEAD.match(s)
        mode = "run"
    if m is None:
        return None
    d = {k: float(v) for k, v in _KV.findall(s)}
    return {
        "step": int(m.group(1)),
        "mode": mode,
        "policy_loss": _num(d, "policy"),
        "value_loss": _num(d, "value"),
        "value_loss_raw": _num(d, "vraw"),          # run 模式没有 => None
        "value_scale": _num(d, "scale"),            # run 模式没有 => None
        "entropy": _num(d, "entropy"),
        "deploy_pct": _num(d, "deploy"),
        "bundle_mean": _num(d, "bundle"),
        "clip_pct": _num(d, "clip"),
        "ratio_mean": _num(d, "ratio"),
        "gnorm": _num(d, "gnorm"),
        "evb": _num(d, "EVb"),
        "evin": _num(d, "EVin"),
        "adv_mean": _num(d, "adv"),
        "adv_std": _num(d, "advstd"),
        "n": _num(d, "n"),
    }


def parse_log(path, tail_bytes=None):
    """解析训练日志 → (rows, info)。``info`` 含解析计数（缺失时**大声**报告，不静默返回空）。

    ``tail_bytes``：只读文件末尾这么多字节（长跑实时刷 UI 时避免每次重读整个日志）。
    行可能被截断 => 末尾不完整的第一行会被 ``parse_line`` 自然丢掉。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"训练日志不存在：{path}")
    rows = []
    n_lines = n_skip = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        if tail_bytes:
            try:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - int(tail_bytes)))
                f.readline()          # 丢掉可能被截断的首行
            except OSError:
                f.seek(0)
        for line in f:
            s = line.strip()
            if not (s.startswith("[solo step ") or s.startswith("[step ")):
                continue
            n_lines += 1
            r = parse_line(s)
            if r is None or (r["entropy"] is None and r["value_loss"] is None):
                n_skip += 1
                continue
            rows.append(r)
    rows.sort(key=lambda r: r["step"])
    info = {"path": os.path.abspath(path), "n_rows": len(rows), "n_step_lines": n_lines,
            "n_skipped": n_skip,
            "mode": (rows[0]["mode"] if rows else None),
            "step_first": (rows[0]["step"] if rows else None),
            "step_last": (rows[-1]["step"] if rows else None),
            "has_value_loss_raw": any(r["value_loss_raw"] is not None for r in rows)}
    return rows, info


def mad(xs):
    """原始 MAD（中位绝对偏差）——**不乘** 1.4826（与 ``judge_critic_inertia.py`` 同口径）。"""
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    m = _median(xs)
    return _median([abs(x - m) for x in xs])


def _median(xs):
    ys = sorted(xs)
    n = len(ys)
    if n == 0:
        return None
    if n % 2:
        return ys[n // 2]
    return 0.5 * (ys[n // 2 - 1] + ys[n // 2])


def _vals(rows, key):
    return [r[key] for r in rows if r.get(key) is not None]


def blocks(rows, key, n_blocks=10):
    """按点数等分 n_blocks 段，返回每段的 ``{step_lo, step_hi, n, median, mad, mean}``。"""
    vs = _vals(rows, key)
    if not vs:
        return []
    n = len(rows)
    out = []
    for b in range(n_blocks):
        seg = rows[b * n // n_blocks:(b + 1) * n // n_blocks]
        if not seg:
            continue
        v = _vals(seg, key)
        if not v:
            continue
        out.append({"step_lo": seg[0]["step"], "step_hi": seg[-1]["step"], "n": len(v),
                    "median": _median(v), "mad": mad(v),
                    "mean": sum(v) / len(v)})
    return out


def plateau_read(rows, key, k=3.0, win_frac=0.05, min_win=8, tail_bytes=None):
    """**描述性**平台读数：末窗中位 vs 紧邻前一窗，阈值 = ``k × 前窗 MAD``。

    为什么用"末窗 vs 前一窗"（局部）而不是"末窗 vs 全程"（全局）：问「是否已到步数瓶颈」
    等价于问「**最近**还动不动」，而全程带会把训练早期的陡降算进去、把带宽撑爆 => 恒判平台。

    返回 dict：
    - ``verdict``：``plateau`` / ``moving_down`` / ``moving_up`` / ``unresolvable``
    - ``key`` ``k`` ``n`` ``win`` ``tail``/``base``（``{step_lo,step_hi,n,median,mad}``）
    - ``diff``（末窗中位 - 前窗中位）、``spread``（前窗 MAD）、``snr`` = |diff|/spread
    - ``slope_per_10k``：**后一半**点的最小二乘斜率（辅助读数）
    """
    vs_rows = [r for r in rows if r.get(key) is not None]
    n = len(vs_rows)
    win = max(int(min_win), int(n * win_frac))
    out = {"key": key, "k": float(k), "n": n, "win": win, "verdict": "unresolvable",
           "diff": None, "spread": None, "snr": None, "slope_per_10k": None,
           "tail": None, "base": None, "why": ""}
    if n < 2 * max(min_win, 4):
        out["why"] = f"点数不足（n={n} < 2×{max(min_win, 4)}）"
        return out
    win = min(win, n // 2)
    tail_rows, base_rows = vs_rows[-win:], vs_rows[-2 * win:-win]
    tv, bv = _vals(tail_rows, key), _vals(base_rows, key)
    if not tv or not bv:
        out["why"] = "窗口内无有效值"
        return out
    t_med, b_med = _median(tv), _median(bv)
    b_mad = mad(bv)
    out["tail"] = {"step_lo": tail_rows[0]["step"], "step_hi": tail_rows[-1]["step"],
                   "n": len(tv), "median": t_med, "mad": mad(tv)}
    out["base"] = {"step_lo": base_rows[0]["step"], "step_hi": base_rows[-1]["step"],
                   "n": len(bv), "median": b_med, "mad": b_mad}
    out["diff"] = t_med - b_med
    out["spread"] = b_mad
    if b_mad is None or b_mad <= 0:
        out["why"] = "前窗 MAD=0（退化）=> 按 §11.13.4-3 退出该指标的检验"
        return out
    out["snr"] = abs(out["diff"]) / b_mad
    if abs(out["diff"]) <= k * b_mad:
        out["verdict"] = "plateau"
    else:
        out["verdict"] = "moving_down" if out["diff"] < 0 else "moving_up"
    # 后一半的线性斜率（辅助；与平台判定解耦）
    half = vs_rows[n // 2:]
    if len(half) >= 3:
        xs = [r["step"] / 10000.0 for r in half]
        ys = [r[key] for r in half]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        den = sum((x - mx) ** 2 for x in xs)
        if den > 0:
            out["slope_per_10k"] = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    return out


def health_summary(path, k=3.0, n_blocks=10, tail_bytes=None, keys=HEALTH_FIELDS):
    """一次取全：稠密序列（下采样后）+ 分段表 + 每指标平台读数 + 诊断。

    返回可直接 ``json.dumps`` 的 dict（dashboard ``/api/health`` 与 CLI 共用同一份读数
    => 不会出现"脚本这样读、UI 那样读"的两套口径）。
    """
    rows, info = parse_log(path, tail_bytes=tail_bytes)
    if not rows:
        return {"ok": False, "error": "日志里没有解析到任何训练步（口径/路径不对？）", "log": info}
    # 下采样：UI 不需要 781 点，保留形状（均匀抽样 + 必含末点）
    max_pts = 400
    step = max(1, len(rows) // max_pts)
    pts = rows[::step]
    if pts[-1] is not rows[-1]:
        pts = pts + [rows[-1]]
    fields = [f for f in keys if _vals(rows, f)]
    return {
        "ok": True,
        "log": info,
        "k": float(k),
        "fields": fields,
        "points": [{kk: r.get(kk) for kk in ("step", "entropy", "value_loss", "value_loss_raw",
                                            "value_scale", "policy_loss", "deploy_pct",
                                            "bundle_mean", "gnorm")} for r in pts],
        "blocks": {f: blocks(rows, f, n_blocks) for f in fields},
        "read": {f: plateau_read(rows, f, k=k) for f in fields},
        "caveat": ("描述性读数，非预注册判据（无人认领的阈值不得当判决用，见模块 docstring）；"
                   "entropy = 掩码后一次决策内各 decoder 步熵之和；value_loss_raw = 原始 MSE"
                   "（value= 是 ÷v_scale^2 的缩放量，跨时间不可比）"),
    }


def derive_train_log(solo_dir=None, repo_root=None, explicit=None):
    """猜训练日志路径：显式优先，其次 ``<repo>/docs/train_<run_basename>.log``。

    返回第一个存在的候选；都不存在则返回 None（**不猜一个不存在的路径**给上层用）。
    """
    cands = []
    if explicit:
        cands.append(explicit)
    if solo_dir:
        name = os.path.basename(os.path.abspath(solo_dir).rstrip("/\\"))
        if repo_root:
            cands.append(os.path.join(repo_root, "docs", f"train_{name}.log"))
        # 也接受放在 run 目录里的同名日志
        cands.append(os.path.join(solo_dir, f"train_{name}.log"))
    for c in cands:
        if c and os.path.exists(c):
            return os.path.abspath(c)
    return None
