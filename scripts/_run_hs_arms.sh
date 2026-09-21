#!/usr/bin/env bash
# W2/W3 三臂 × 三 seed 训练（预注册 docs/il_whiff_handscore_prereg_2026-09-22.md §6.4/§6.5）
#
# 三臂（**同语料、同形状、同初值**；唯一变量 = plan 尾 17 列有没有携带信息）：
#   HS17 → --plan-extras-zero 17（对照组，尾列全抹）
#   HS04 → --plan-extras-zero 4 （只有 W2 手牌打分 13 维活着）
#   HS00 → --plan-extras-zero 0 （W2 + W3 全活）
#
# 三条泳道（一臂一泳道）× 三条 seed，`--threads 4` ⇒ 峰值 12 线程（本机 16 logical），
# 避免与语料生成/其它读数抢核（实测：16 线程 × 多进程会互相饿死，见会话记录）。
#
# 用法（仓库根）：bash scripts/_run_hs_arms.sh > docs/fl_il_2026-09-21/hs_arms.log 2>&1
set -u
cd "$(dirname "$0")/.." || exit 1

# ⚠️ **必须**（2026-09-22 实测事故）：三条泳道并发时**全部**在第一处 `opt.step()` 崩：
#     torch.AcceleratorError: CUDA error: out of memory
#       ← `torch.optim.Optimizer.step` 的 `_accelerator_graph_capture_health_check()`
#         会调 `torch.accelerator.current_stream()`，**即使模型在 CPU 也会去建 CUDA 上下文**
#       ← 3 个进程 × 各建一个 CUDA 上下文 ⇒ 小显存卡直接 OOM（本机 875 MiB 已占用）
# 本训练**本来就是 CPU**（`FollowerPolicy.device == "cpu"`，`il_bc_sweep` 从不 `to_device`）
# ⇒ 屏蔽 CUDA 不改变任何数值，只是不碰驱动。缺这一行的症状 = 泳道**静默没人跑**。
export CUDA_VISIBLE_DEVICES=""

DATA=runs/_fl_il_bc_hs/train
ROOT=runs/_fl_il_bc_hs

lane() {
  local arm="$1" z="$2"
  for s in 0 1 2; do
    local out="$ROOT/sweep_mix$arm/s$s"
    echo "=== [$(date +%H:%M:%S)] $arm seed=$s zero=$z → $out ==="
    PYTHONDONTWRITEBYTECODE=1 ./.venv/Scripts/python.exe -u scripts/il_bc_sweep.py \
      --data-dir "$DATA" --out-dir "$out" \
      --configs 3:1e-3 --seed "$s" --mix-ratio 1.0 \
      --plan-extras-zero "$z" --threads 4
  done
  echo "=== [$(date +%H:%M:%S)] lane $arm 完成 ==="
}

lane HS17 17 &
lane HS04 4 &
lane HS00 0 &
wait
echo "=== [$(date +%H:%M:%S)] 全部 9 跑完成 ==="
