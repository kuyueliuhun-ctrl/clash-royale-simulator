#!/usr/bin/env bash
# v3 探针两阶段执行（预注册 docs/value_ln_probe3_prereg_2026-09-14.md）
#   阶段 1：rollout（并行）→ npz
#   阶段 2：逐条离线拟合 → json + log
#
# ⚠️ 路径纪律：`python.exe` 是 **Windows 进程**，`/mnt/e/...` 形式的**参数**它解析不了
#    （会变成 `E:\mnt\e\...`）。因此传给它的每一个路径都必须是**相对 `src/clasher_new` 的路径**；
#    只有 bash 自己用的路径（重定向、mkdir）才可以用 /mnt/e 绝对路径。
#
# 用法：bash scripts/run_probe_v3.sh [rollout|fit|all] [seed...]
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src/clasher_new"
PY="$ROOT/.venv/Scripts/python.exe"
PROBE="../../scripts/probe_value_ln.py"          # 相对 src/clasher_new
LOGABS="$ROOT/runs/_probe_v3"
RELOUT="../../runs/_probe_v3"                     # 相对 src/clasher_new
CKPT="runs/critic_inert_probe_20k/solo_main_20000.pt"
mkdir -p "$LOGABS"

phase="${1:-all}"
shift || true
seeds=("$@")
if [ "${#seeds[@]}" -eq 0 ]; then seeds=(7 11 13 17); fi

rollout() {
  local s="$1"; local extra="${2:-}"
  cd "$SRC" || exit 1
  PYTHONIOENCODING=utf-8 "$PY" "$PROBE" \
    --ckpt "$CKPT" --frames 30000 --seed "$s" --device cpu \
    --tag "v3_seed$s" --ladder v3 --raw-obs $extra \
    --save-npz "$RELOUT/seed$s.npz" --rollout-only \
    > "$LOGABS/rollout_seed$s.log" 2>&1
  echo "[rollout] seed=$s exit=$? log=$LOGABS/rollout_seed$s.log"
}

fit() {
  local s="$1"
  cd "$SRC" || exit 1
  PYTHONIOENCODING=utf-8 "$PY" "$PROBE" \
    --ckpt "$CKPT" --seed "$s" --device cpu \
    --tag "v3_seed$s" --ladder v3 --raw-obs \
    --npz "$RELOUT/seed$s.npz" --exclude grid_x --out "$RELOUT/seed$s.json" \
    > "$LOGABS/fit_seed$s.log" 2>&1
  echo "[fit] seed=$s exit=$? json=$LOGABS/seed$s.json"
}

case "$phase" in
  rollout)
    for s in "${seeds[@]}"; do
      extra=""; [ "$s" = "7" ] && extra="--cnn-input"
      rollout "$s" "$extra" &
    done
    wait
    ;;
  fit)
    for s in "${seeds[@]}"; do fit "$s"; done
    ;;
  all)
    for s in "${seeds[@]}"; do
      extra=""; [ "$s" = "7" ] && extra="--cnn-input"
      rollout "$s" "$extra" &
    done
    wait
    for s in "${seeds[@]}"; do fit "$s"; done
    ;;
  *) echo "usage: $0 [rollout|fit|all] [seeds...]"; exit 2;;
esac
echo "[done] phase=$phase seeds=${seeds[*]}"
