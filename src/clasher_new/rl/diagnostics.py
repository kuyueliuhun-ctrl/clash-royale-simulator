"""训练期诊断探针（v3 P0-B / P0-C，2026-09-11）。

背景：`enc` 无归一化时 L2 范数 ≈533（CNN 输出的常数分量 ≈468），把 GRUCell 的
tanh 候选压饱和（|n|≈0.994）→ 隐状态 h 跨帧 std ≈2.6e-5（数值恒定）→
`value_head(h)` 恒输出常数 → EV 恒负；`slot_head/cell_head` 同样吃常数 h，策略
网络基本开环。完整取证见 `docs/value_channel_saturation_diagnosis_2026-09-11.md`。

本模块把这些**取证时手写的探针**固化为训练内可调用函数，用于：
1. 评估时落盘进 history（dashboard 可见，P0-B）；
2. selfcheck / 启动前检查（不达标报警，P0-C）。

指标口径与 `scripts/diag_value_head.py` 一致，可交叉核对：

- `h_std`    : 隐状态跨帧 std（逐维 std 后取均值）。饱和 ~1e-5；健康 >0.05。
- `n_abs`    : GRU 候选 `tanh(...)` 的 |·| 均值。饱和 →1.0；健康 <0.9。
- `value_std`: value_head 输出跨帧 std。
- `r_std`    : （调用方另测，2026-09-11 补齐）同批回报 std，用来看 value 波动是否
  与回报同量级。

**v3 §2 验收表第 3 行（`value_head 输出 std / 批内 R std > 0.3`）的分母口径**：
`r_std` 由调用方 `rl/train_solo.py::eval_and_write` 从**评估窗口累积的逐帧 return**
（`PPOTrainer.last_ev_pairs`，与池化 EV 同一批数据、**未缩放**量纲，与 value_head
输出同尺度）算出两版：`r_std`（窗口池化）与 `r_std_batch`（每 128 帧一批取批内 std
再平均，对应计划原文的"批内"）。比值用 `value_std / r_std_batch` 落盘为
`value_std_ratio`。注意分子来自"最近 96 帧探针"、分母来自"整窗口"，两者帧集不同，
是量级对照而非严格同帧配对。

**注意**：探针只做前向、不推进 env，传入的帧可来自真实 rollout 缓冲，无副作用。
"""

import sys

import numpy as np
import torch


def print_safe(msg):
    """print 兜底：编码不支持的字符降级为 '?'，而不是抛 UnicodeEncodeError。

    训练日志含中文/emoji；Windows 控制台/重定向管道默认 cp936，`print` 会抛
    UnicodeEncodeError。该异常若发生在 `eval_and_write` 或 `load_checkpoint` 的
    try 内，会被 `except Exception` 捕获，而 except 处理器里的 `{e!r}` 又内嵌
    同一个不可编码字符 → 二次抛错、**直接崩训练/崩对手池加载**
    （2026-09-11 由 `test_solo_resume` 与 `test_opponent_pool_mix` 实际复现）。
    """
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(msg.encode(enc, "replace").decode(enc, "replace"), flush=True)


#: 验收门槛（v3 §2）。低于/高于即视为 GRU 未解冻，必须报警。
#: `h_std` 的 0.05 是 v3 §2 验收表的值（§1 P0-C 曾误写 0.02，2026-09-11 已统一为 0.05；
#: dashboard 把 0.02 用作黄色预警带）。
THRESHOLDS = {"h_std": 0.05, "n_abs": 0.9, "value_std_ratio": 0.3}


def _gru_gate_stats(policy, enc, h_prev):
    """手动复现 GRUCell 内部，返回候选 n 的 |·| 均值（饱和判据）。

    PyTorch GRUCell 前向（weight_ih 布局 = [W_ir | W_iz | W_in]）::

        gi = x @ W_ih.T + b_ih ; gh = h @ W_hh.T + b_hh
        r = σ(gi_r + gh_r) ; z = σ(gi_z + gh_z)
        n = tanh(gi_n + r * gh_n) ; h' = (1 − z) * n + z * h

    |n| 恒接近 1 即 tanh 饱和 —— 这是 h 被冻成常数的直接机理。
    """
    cell = policy.gru_cell
    h = policy.hidden_dim
    gi = enc @ cell.weight_ih.t() + cell.bias_ih
    gh = h_prev @ cell.weight_hh.t() + cell.bias_hh
    r = torch.sigmoid(gi[:, :h] + gh[:, :h])
    n = torch.tanh(gi[:, 2 * h:] + r * gh[:, 2 * h:])
    return float(n.abs().mean().item())


@torch.no_grad()
def gru_vitality(policy, frames, max_frames=96):
    """测 GRU / value_head 的"活力"。

    frames: list of `(obs, belief_token, plan_token)` —— 用真实 rollout 帧即可
    （纯前向，不推进 env、不改策略状态）。返回 dict：

    - `h_std`   隐状态跨帧 std（逐维后取均值）
    - `n_abs`   GRU 候选 |·| 均值
    - `value_std` value_head 输出跨帧 std（**走策略真实的 value 通路**：
      `value_independent` → 独立编码器+MLP 头；`value_bypass` → `value_head(enc)`；
      否则 `value_head(h)`。2026-09-12 修复——原先硬编码 `value_head(h)`，
      对 bypass/independent ckpt 测的不是被训练的量）
    - `n_frames` 实际用到的帧数
    """
    frames = [f for f in frames if f is not None]
    if not frames:
        return {}
    if len(frames) > max_frames:
        frames = frames[-max_frames:]
    was_training = policy.training
    policy.eval()
    dev = policy.device
    hidden = None
    hs, vs, ns = [], [], []
    for (o, b, p) in frames:
        fused, enc = policy._encode_parts(o, b, p)
        if hidden is None:
            hidden = torch.zeros(1, policy.hidden_dim, device=dev)
        ns.append(_gru_gate_stats(policy, enc, hidden))
        hidden = policy.gru_cell(enc, hidden)
        hs.append(hidden.detach().cpu().numpy().ravel())
        vs.append(float(policy._value_from(enc, hidden, fused).item()))
    if was_training:
        policy.train()
    H = np.asarray(hs, dtype=np.float64)
    V = np.asarray(vs, dtype=np.float64)
    return {
        "h_std": float(H.std(axis=0).mean()),
        "n_abs": float(np.mean(ns)),
        "value_std": float(V.std()),
        "n_frames": int(H.shape[0]),
    }


def check_vitality(vit):
    """按 THRESHOLDS 返回告警列表（空 = 通过）。P0-C 用。"""
    warns = []
    if not vit:
        return ["未取得 GRU 活力指标（帧缓冲为空）"]
    if "h_std" in vit and vit["h_std"] <= THRESHOLDS["h_std"]:
        warns.append(f"h 跨帧 std={vit['h_std']:.2e} ≤ {THRESHOLDS['h_std']}"
                     "（隐状态被冻住，GRU 未解冻）")
    if "n_abs" in vit and vit["n_abs"] >= THRESHOLDS["n_abs"]:
        warns.append(f"GRU n(|·|)={vit['n_abs']:.4f} ≥ {THRESHOLDS['n_abs']}"
                     "（tanh 候选饱和）")
    if "value_std_ratio" in vit and vit["value_std_ratio"] is not None \
            and vit["value_std_ratio"] <= THRESHOLDS["value_std_ratio"]:
        warns.append(f"value/R std={vit['value_std_ratio']:.3f} ≤ "
                     f"{THRESHOLDS['value_std_ratio']}（critic 输出波动远小于回报波动，"
                     "价值头近似常数）")
    return warns


def check_policy_architecture(policy):
    """**启动前检查**（v3 P0-C）：静态可判定的饱和病因护栏，返回告警列表。

    与 `check_vitality` 的分工（诚实口径）：GRU 活力必须用真实 rollout 帧才能测，
    启动时一帧都没有 ⇒ 启动前**测不了** h_std/n_abs。启动前能查的只有"结构性病因"：

    - 策略没有 `enc_ln` / `grid_ln`（旧架构代码）→ 归一化缺失，GRU 必饱和或融合尺度失衡；
    - 某个归一化模块被换回 `nn.Identity`（selftest 负对照残留 / 手工调试）→ 同上。

    真实活力仍由评估点 `check_vitality` 报警，并落盘到 history 的 `vitality_warns`。
    """
    warns = []
    for _name, _why in (("enc_ln", "enc 未归一化 → GRU 必饱和"),
                        ("grid_ln", "CNN 输出未归一化 → 融合层尺度失衡"
                                    "（grid_feat 占 fused 范数 98%，见 v3 §3.7）")):
        _m = getattr(policy, _name, None)
        if _m is None:
            warns.append(f"策略缺少 {_name}（v3 架构未落地）：{_why}")
        elif isinstance(_m, torch.nn.Identity):
            warns.append(f"{_name} 是 nn.Identity（负对照/调试残留）：{_why}")
    return warns
