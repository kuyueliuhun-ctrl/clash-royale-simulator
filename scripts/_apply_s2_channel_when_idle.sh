#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# S2 引擎侧改动 **受控落地**：等 run100k 收尾 → 打三个已验证的补丁 → 中立性对账 → 跑相关
# selftest 子集 → 用短 smoke 产出一份 schema-5 录像并复算。
#
# 为什么必须等：`rl/run_league.py` 用 Windows `spawn` 起 worker，worker 会**重新 import 源码**
# ⇒ 训练期间改源码会污染在跑的实验（红线 R1）。本脚本只在**终局评估点落盘 + 进程退出后**才动手。
#
# 幂等：三个补丁都用 `patch --dry-run` 预检；任一失败 ⇒ 从备份整体回滚并以非零码退出。
# 手动回滚：cp /tmp/s2_apply_bak/* src/clasher_new/  （保留原路径结构见下）
set -u
REPO=/mnt/e/clash-royale-simulator-main
SRC=$REPO/src/clasher_new
PY=$REPO/.venv/Scripts/python.exe
RT=$SRC/runs/run100k
BAK=/tmp/s2_apply_bak
EXPECT_DIGEST=9adc2aafee52aa55f964fb52af7ac86fc240d105058d880bb50772b95bcd5b55

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "=== 等 run100k 收尾（终局评估点 + run_state.step=100000）==="
ok=0
for i in $(seq 1 240); do          # 最多 2 小时
  if [ -f "$RT/replays/league_100000.pkl" ] && grep -q '"step": 100000' "$RT/run_state.json" 2>/dev/null; then
    ok=1; break
  fi
  sleep 30
done
if [ "$ok" != "1" ]; then log "!! 等待超时，未做任何改动"; exit 3; fi
log "终局评估点已落盘；再等 120 s 让 worker 退出"
sleep 120

log "=== 备份 ==="
mkdir -p "$BAK/rl"
cp "$SRC/battle.py"        "$BAK/battle.py"
cp "$SRC/rl/replay.py"     "$BAK/rl/replay.py"
cp "$SRC/rl/run_league.py" "$BAK/rl/run_league.py"
cd "$REPO" || exit 4

log "=== 预检三个补丁（dry-run）==="
patch --dry-run -p0 "$SRC/battle.py"        < docs/root_cast_channel_2026-09-18.diff || { log "!! 预检失败 battle"; exit 5; }
patch --dry-run -p0 "$SRC/rl/replay.py"     < docs/schema5_replay_2026-09-18.diff   || { log "!! 预检失败 replay"; exit 5; }
patch --dry-run -p0 "$SRC/rl/run_league.py" < docs/schema5_run_league_2026-09-18.diff || { log "!! 预检失败 run_league"; exit 5; }

log "=== 应用 ==="
patch -p0 "$SRC/battle.py"        < docs/root_cast_channel_2026-09-18.diff || { log "!! 失败 battle"; exit 6; }
patch -p0 "$SRC/rl/replay.py"     < docs/schema5_replay_2026-09-18.diff   || { log "!! 失败 replay"; exit 6; }
patch -p0 "$SRC/rl/run_league.py" < docs/schema5_run_league_2026-09-18.diff || { log "!! 失败 run_league"; exit 6; }
grep -c "root_cast" "$SRC/battle.py" "$SRC/rl/replay.py" || true

rollback() {
  log "!! 回滚"
  cp "$BAK/battle.py" "$SRC/battle.py"
  cp "$BAK/rl/replay.py" "$SRC/rl/replay.py"
  cp "$BAK/rl/run_league.py" "$SRC/rl/run_league.py"
}

log "=== 验证 1/3：行为中立性逐位对账（DIGEST 必须 = $EXPECT_DIGEST）==="
cd "$SRC" || exit 7
# ⚠️ Windows 版 python 的 stdout 在重定向下是 CRLF ⇒ awk 取出的字段**带尾随 \r**
# ⇒ 与 EXPECT_DIGEST 逐字节比较会**假阴性**（2026-09-18 实测把一次成功的补丁回滚了）。
D=$(PYTHONIOENCODING=utf-8 "$PY" ../../scripts/s2_neutrality_probe.py . 2>&1 | tee /tmp/s2_neutral_patched.txt | awk '/^DIGEST/{print $2}' | tr -d '\r')
if [ "$D" != "$EXPECT_DIGEST" ]; then
  log "!! DIGEST 不一致：$D"; cat /tmp/s2_neutral_patched.txt; rollback; exit 8
fi
log "DIGEST 一致 ✅"; tail -3 /tmp/s2_neutral_patched.txt

log "=== 验证 2/3：相关 selftest 子集（R19）==="
PYTHONIOENCODING=utf-8 "$PY" ../../scripts/run_selftests.py \
    test_noncombat_entity_contract test_replay_roundtrip test_league_replays \
    test_dashboard_replays > /tmp/s2_selftests.txt 2>&1
RC=$?
tail -20 /tmp/s2_selftests.txt
if [ "$RC" != "0" ]; then log "!! selftest 子集失败（RC=$RC）⇒ 回滚"; rollback; exit 9; fi
log "selftest 子集通过 ✅"

log "=== 验证 3/3：短 smoke 产出 schema-5 录像 ==="
PYTHONIOENCODING=utf-8 "$PY" -u rl/run_league.py --mode run --config-name run_schema5 \
    --fresh --total-steps 600 --steps-per-eval 300 --n-eval-games 4 --eval-workers 2 \
    --device cuda > /tmp/s2_schema5_smoke.log 2>&1
log "smoke 退出码 $?"
grep -E "eval-plan|评估点|schema" /tmp/s2_schema5_smoke.log | head -5
PYTHONIOENCODING=utf-8 "$PY" -c "
import sys; sys.path.insert(0,'.')
from rl.replay import load_league_replays, LEAGUE_REPLAY_SCHEMA
import collections
gs = load_league_replays('runs/run_schema5/replays/league_0.pkl')
lens = collections.Counter(len(e) for g in gs for f in g['frames'] for e in f['entities'])
has_v = sum(1 for g in gs for f in g['frames'] if f.get('v0') is not None)
print('LEAGUE_REPLAY_SCHEMA =', LEAGUE_REPLAY_SCHEMA)
print('games =', len(gs), 'entity arity =', dict(lens), 'frames with v0 =', has_v)
"
log "=== 复算（schema 5 应报 exact=True）==="
PYTHONIOENCODING=utf-8 "$PY" ../../scripts/offline_engagement_trade.py \
    --replays runs/run_schema5/replays --tower-mode none 2>&1 | sed -n '1,18p'
log "=== 完成。回滚命令：cp $BAK/battle.py $SRC/battle.py && cp $BAK/rl/*.py $SRC/rl/ ==="
