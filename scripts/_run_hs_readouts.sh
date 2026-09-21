#!/usr/bin/env bash
# W2/W3 三臂 × 三 seed 的**读数链**（留出 + 部署 + 使用率 + 台账 + 归因）。
# 串行执行：本机并发上限实测 = 3 个 torch 进程（第 4 个撞 WinError 1455 页面文件太小）
# ⇒ 训练泳道跑完后才能开始，且本链内部**不并发**。
#
# **可续跑**：每一步都 `[ -f 产物 ] ||` 跳过已完成的 ⇒ 中断/报错后直接重跑即可
#（`FORCE=1 bash ...` 强制全部重做）。
#
# ⚠️ 2026-09-22 事故（已修）：首版把文件名写成 `"$DOCS/readout_hs$spec""_s$s_stats.json"`，
#    bash 把 `$s_stats` 当成**一个**变量名 ⇒ 配 `set -u` 直接 `unbound variable` 退出
#    ⇒ 整条链只跑完第一个留出读数就死。**教训：`$var` 后紧跟 `_` 必须写 `${var}`。**
#
# 用法（仓库根）：bash scripts/_run_hs_readouts.sh > docs/fl_il_2026-09-21/hs_readouts.log 2>&1
set -u
cd "$(dirname "$0")/.." || exit 1
export CUDA_VISIBLE_DEVICES=""          # 见 _run_hs_arms.sh 的事故说明（CPU 训练不许碰 CUDA）
PY=./.venv/Scripts/python.exe
DOCS=docs/fl_il_2026-09-21
ROOT=runs/_fl_il_bc_hs
HOLD=$ROOT/holdout
FORCE="${FORCE:-0}"

run_step() {   # run_step <产物路径> <描述> <命令...>
  local out="$1"; shift
  local desc="$1"; shift
  if [ "$FORCE" != "1" ] && [ -f "$out" ]; then
    echo "--- [$(date +%H:%M:%S)] 跳过（已有 $out）：$desc"
    return 0
  fi
  echo "=== [$(date +%H:%M:%S)] $desc"
  PYTHONDONTWRITEBYTECODE=1 "$@" || echo "!!! [$(date +%H:%M:%S)] 该步失败：$desc"
}

for spec in 17 04 00; do
  for s in 0 1 2; do
    CK="$ROOT/sweep_mixHS${spec}/s${s}/bc_fl_e3_lr0.001.pt"
    if [ ! -f "$CK" ]; then echo "!!! 缺 ckpt $CK ⇒ 跳过"; continue; fi
    run_step "$DOCS/holdout_hs${spec}_s${s}.json" "HS${spec} s${s} 留出读数" \
      $PY -u scripts/il_eval_holdout.py --ckpt "$CK" --data-dir "$HOLD" \
      --out "$DOCS/holdout_hs${spec}_s${s}.json"
    run_step "$DOCS/readout_hs${spec}_s${s}_stats.json" "HS${spec} s${s} 部署读数（10 局）" \
      $PY -u scripts/il_readout_games.py --ckpt "$CK" --games 10 --decks solo \
      --hidden none --opp-hidden same --out "runs/il_readout_hs${spec}_s${s}" \
      --json "$DOCS/readout_hs${spec}_s${s}_stats.json"
    run_step "$DOCS/usage_hs${spec}_s${s}.json" "HS${spec} s${s} 卡牌使用率（用户口径）" \
      $PY -u scripts/il_card_usage.py --ckpt "$CK" --data-dir "$HOLD" --limit 12000 \
      --out "$DOCS/usage_hs${spec}_s${s}.json"
  done
done

run_step "$DOCS/hs_feature_use_HS00_s0.json" "归因仪器（全活臂 HS00 seed 0）" \
  $PY -u scripts/il_hs_feature_use.py \
  --ckpt "$ROOT/sweep_mixHS00/s0/bc_fl_e3_lr0.001.pt" --data-dir "$HOLD" --limit 4000 \
  --out "$DOCS/hs_feature_use_HS00_s0.json"

run_step "$DOCS/hs_report.json" "台账（脚本复算）" \
  $PY -u scripts/il_hs_report.py --root "$ROOT" --docs "$DOCS" --out "$DOCS/hs_report.json"

echo "=== [$(date +%H:%M:%S)] 读数链完成 ==="
