#!/usr/bin/env bash
# ============================================================================
# intent-save **A/B 各 100k 步对比训练**（用户 2026-09-19 发令「开始测试」）
#
# 预注册：docs/intent_save_prereg_2026-09-19.md（判据 J1 跑前写死、失败分支 F1/F2/F3）
#   A = 意图关（--config economy，状态 quo，**旧架构** 6 option）
#   B = 意图开（--config economy **+ --intent-save**，11 option / 观测 +6）
#   ★ 唯一变量 = --intent-save；**两臂都 --fresh**（【R6】intent 改 slot_head/sub_emb/enc_fc
#     形状，`enc_fc` 加宽无「尾部追加」兼容 —— 标量在 fused 中段，续训会静默重置）
#
# 为什么**顺序跑**而不是并行：并行会让两臂互相争抢 commit 内存 / GPU / CPU，
# 那就不是单变量了（【R3】【R17】）。本仓自己的 100k 两臂前例（et_solo100k）也是顺序跑。
#
# 阳性完成门禁（照 docs/et_solo100k 收尾事故的教训）：每臂日志必须同时出现
#   `[done] solo` 与 `eval@100000`；**不得**用「已无 run_league 进程」当完成证据
#   —— 顺序跑的交接空窗里判据会误触发（用缺席当完成证据 = 同类错误第二次）。
#
# 【R1】开跑前实测（2026-09-19，scripts/check_commit.py）：
#   提交上限 47.31 GB / 已用 36.80 GB / **可用 10.51 GB** ⇒ 低于自身文档的 12 GB 门 ⇒
#   按 check_commit.py 自身建议**降档 `--eval-workers 8`**（历史标定：20.6 GB→12 安全；
#   10.68 GB→12 出过 OOM/CUDA unknown）。**两臂同档** ⇒ A/B 单变量不受影响。
#   ⚠️ 这是**开跑前**的显式配置选择、且已写进启动记录，**不是**训练期静默降级。
# ============================================================================
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/src/clasher_new" || exit 1
PY="../../.venv/Scripts/python.exe"

COMMON="--mode solo --config economy --fresh --total-steps 100000 \
--steps-per-eval 8000 --big-eval-every 100000 --n-eval-games 20 --n-eval-games-big 40 \
--eval-workers 8 --device cuda --value-norm running --adv-norm scale --diagnose-every 10 \
--hist-seed-dir runs/economy_9k_ft --hist-seed-dir runs/economy_9j \
--ppo-epochs 4 --ppo-minibatch 32 --ppo-shuffle"

gate () {   # $1 = log 路径, $2 = 臂名
  if grep -q "\[done\] solo" "$1" && grep -q "eval@100000" "$1"; then
    echo "[gate] $2 OK（[done] solo + eval@100000 均在场）"
    return 0
  fi
  echo "[gate] $2 ABORT：缺完成行（[done] solo / eval@100000）—— 拒绝继续，见 $1"
  return 9
}

echo "=== 启动记录 $(date) ==="
echo "COMMON = $COMMON"
echo "A: --config-name intent_off_100k"
echo "B: --config-name intent_on_100k --intent-save"
echo "git HEAD = $(git -C "$ROOT" log --oneline -1)"

echo "=== ARM A (intent OFF) $(date) ==="
PYTHONIOENCODING=utf-8 $PY -u rl/run_league.py $COMMON --config-name intent_off_100k \
  > "$ROOT/docs/train_intent_off_100k.log" 2>&1
echo "EXIT_A=$?"
gate "$ROOT/docs/train_intent_off_100k.log" A || { echo "ABORT_ALL"; exit 9; }

echo "=== ARM B (intent ON) $(date) ==="
PYTHONIOENCODING=utf-8 $PY -u rl/run_league.py $COMMON --intent-save --config-name intent_on_100k \
  > "$ROOT/docs/train_intent_on_100k.log" 2>&1
echo "EXIT_B=$?"
gate "$ROOT/docs/train_intent_on_100k.log" B || { echo "ABORT_ALL"; exit 9; }

echo "ALLDONE $(date)"
