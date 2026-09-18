#!/usr/bin/env bash
# 收尾脚本：等 et_solo100k 两臂**都真正跑完** → 出 §11.13.2 四层读数 + §11.13.8 判读骨架
# → 新建分支推送。幂等，可重复运行（已存在同分支则复用）。
#
# ★ 门禁设计（2026-09-18 修正一个真实竞态）：
#   旧版判据 = 「无 run_league 进程」。但两臂是**顺序**跑的，A_et 退出到 B_ctrl 启动之间
#   存在一个（数秒级）无进程窗口 ⇒ 60 s 轮询有概率落在窗口里 ⇒ 在 B_ctrl 还没跑的情况下
#   就出读数 + 判读文档（`--log-b` 指向不存在的日志）。这是"用缺席当完成证据"的经典错误。
#   现改为**阳性门禁**：两臂各自的日志都必须出现 `[done] solo` 完成行 + 末点回放落盘，
#   再叠加"确实没有 run_league 进程"作二次确认。
set -u
ROOT=/mnt/e/clash-royale-simulator-main
PS='E:\clash-royale-simulator-main\scripts\kill_by_cmdline.ps1'
LOG_A="$ROOT/docs/train_et_solo100k.log"
LOG_B="$ROOT/docs/train_ctrl100k.log"
DIR_A="$ROOT/src/clasher_new/runs/et_solo100k"
DIR_B="$ROOT/src/clasher_new/runs/et_ctrl100k"
NEED_REPLAYS=14          # 0,8000..96000（13）+ 100000 终局点 = 14
MAX_WAIT_MIN=480         # 8 h

cd "$ROOT" || exit 1
echo "[finalizer] armed $(date)"

ps_marker () {  # $1 = marker; 打印 NONE 或 FOUND 行
    cmd.exe /c "powershell -NoProfile -ExecutionPolicy Bypass -File $PS -Marker $1 -DryRun" 2>&1 | tr -d '\r'
}

# 「该臂已完成」的阳性判据：日志有完成行 + 终局评估点 + 回放数够
arm_done () {  # $1 = log path, $2 = run dir, $3 = label
    [ -f "$1" ] || { echo "[finalizer]   $3: 日志不存在"; return 1; }
    tr -d '\r' < "$1" | grep -q "\[done\] solo" || { echo "[finalizer]   $3: 缺完成行 [done] solo"; return 1; }
    tr -d '\r' < "$1" | grep -q "eval@100000" || { echo "[finalizer]   $3: 缺终局评估点 eval@100000"; return 1; }
    local n=0
    [ -d "$2/replays" ] && n=$(ls "$2/replays" 2>/dev/null | wc -l)
    [ "$n" -ge "$NEED_REPLAYS" ] || { echo "[finalizer]   $3: 回放只有 $n < $NEED_REPLAYS"; return 1; }
    echo "[finalizer]   $3: DONE（完成行 + eval@100000 + $n 个回放）"
    return 0
}

for i in $(seq 1 "$MAX_WAIT_MIN"); do
    a_ok=0; b_ok=0
    arm_done "$LOG_A" "$DIR_A" "A_et"   && a_ok=1
    arm_done "$LOG_B" "$DIR_B" "B_ctrl" && b_ok=1
    if [ "$a_ok" = "1" ] && [ "$b_ok" = "1" ]; then
        out=$(ps_marker run_league)
        if [ "$out" = "NONE" ]; then
            echo "[finalizer] 两臂阳性命中 + 无残留进程 @$(date)"; break
        fi
        echo "[finalizer] 两臂日志已完但仍有 run_league 进程（等它退干净）: $out"
    fi
    [ $((i % 5)) -eq 0 ] && echo "[finalizer] waiting… $i/$MAX_WAIT_MIN min $(date)"
    sleep 60
done

# 最终硬门禁：任一臂未完成则**拒绝出产物**（宁可不出，也不出半份判读）
arm_done "$LOG_A" "$DIR_A" "A_et"   || { echo "[finalizer] ABORT: A_et 未完成，拒绝出产物"; exit 9; }
arm_done "$LOG_B" "$DIR_B" "B_ctrl" || { echo "[finalizer] ABORT: B_ctrl 未完成，拒绝出产物"; exit 9; }
sleep 20

echo "[finalizer] A_et evals=$(tr -d '\r' < "$LOG_A" | grep -cE 'eval@[0-9]+:') ctrl_evals=$(tr -d '\r' < "$LOG_B" | grep -cE 'eval@[0-9]+:')"
echo "[finalizer] replays: A=$(ls "$DIR_A/replays" | wc -l) B=$(ls "$DIR_B/replays" | wc -l)"

# 出四层读数 + §11.13.8 判读骨架
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/et_solo100k_readout.py \
  --log-a docs/train_et_solo100k.log --log-b docs/train_ctrl100k.log \
  --run-a src/clasher_new/runs/et_solo100k --run-b src/clasher_new/runs/et_ctrl100k \
  --out-dir docs/readout_et_solo100k \
  --markdown docs/readout_et_solo100k.md \
  --judgment docs/et_solo100k_judgment_2026-09-18.md > /tmp/finalizer_readout.log 2>&1
rc=$?
echo "[finalizer] readout rc=$rc"
tail -5 /tmp/finalizer_readout.log
if [ "$rc" != "0" ]; then
    echo "[finalizer] ABORT: 读数脚本失败（rc=$rc），不提交半成品。全文：/tmp/finalizer_readout.log"
    exit 8
fi

# 新建分支并推送（只 add 本次产物）
git checkout -b et-solo100k-readout 2>/dev/null || git checkout et-solo100k-readout
git add docs/readout_et_solo100k.md docs/et_solo100k_judgment_2026-09-18.md docs/readout_et_solo100k 2>/dev/null
git commit -q -m "两臂跑完：出 §11.13.2 四层读数（脚本复算）+ §11.13.8 结构判读骨架（自动生成，待人工判读分支 1/2/4）" 2>&1 | tail -2
git push -q -u origin et-solo100k-readout 2>&1 | tail -2
echo "[finalizer] DONE $(date)"; git log --oneline -1
