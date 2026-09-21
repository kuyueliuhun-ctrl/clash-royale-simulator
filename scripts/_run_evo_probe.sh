#!/usr/bin/env bash
# 觉醒 S1 探针（预注册 docs/il_evo_prereg_2026-09-22.md §2 S1 判据①②③）
#
# 同一 200 局、同一配置，**唯一变量 = `--evo-slots`**（关 / 开）：
#   off → 旧路径（不声明觉醒位）；on → 从牌组后缀 `-ev*` 声明觉醒位
# 判据：① 覆盖率 ≥ 0.95；② `labels` 漂移 ≤ 5%、`errors = 0`；③ 触发次数 > 0。
#
# 用法（仓库根）：bash scripts/_run_evo_probe.sh > docs/fl_il_2026-09-21/evo_probe.log 2>&1
# 可续跑：产物在就跳过（FORCE=1 强制重跑）。
set -u
cd "$(dirname "$0")/.." || exit 1

# ⚠️ 沿用 _run_hs_arms.sh 的事故记录：CPU 训练也会在 `Optimizer.step` 里建 CUDA 上下文。
# 本探针**只跑引擎**（不训练），但仍显式屏蔽，避免与其它泳道抢显存。
export CUDA_VISIBLE_DEVICES=""

PY=./.venv/Scripts/python.exe
DOCS=docs/fl_il_2026-09-21
ROOT=runs/_fl_il_bc_evo
COMMON="--mode samples --jsonl ../fl_il_data/replays_part000000.jsonl --games 200 \
--workers 10 --level 11 --coord raw --stop-mode save --stop-stride 4 --plan-extras"

run_step() {   # run_step <产物> <描述> <cmd...>
  local out="$1" desc="$2"; shift 2
  if [ -s "$out" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "=== [$(date +%H:%M:%S)] SKIP（已有产物）$out：$desc ==="
    return 0
  fi
  echo "=== [$(date +%H:%M:%S)] $desc → $out ==="
  "$@" || { echo "!!! 失败：$desc（exit $?）"; return 1; }
}

run_step "$DOCS/evo200_off.json" "S1 基线：--evo-slots 关（旧路径）" \
  $PY -u scripts/fl_il_to_bc.py $COMMON --out-dir "$ROOT/off" --out "$DOCS/evo200_off.json" || exit 1

run_step "$DOCS/evo200_on.json" "S1 处置：--evo-slots 开（声明觉醒位）" \
  $PY -u scripts/fl_il_to_bc.py $COMMON --evo-slots \
  --out-dir "$ROOT/on" --out "$DOCS/evo200_on.json" || exit 1

echo "=== [$(date +%H:%M:%S)] 探针完成 ==="
