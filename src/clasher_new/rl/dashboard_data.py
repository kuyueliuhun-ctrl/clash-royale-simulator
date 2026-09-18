# -*- coding: utf-8 -*-
"""dashboard 的数据面（Tier 2 · T2-2，从 `rl/dashboard.py` 整体搬出）。

**搬了什么**：颜色/标签常量、4 个模块级缓存、以及 **24 个纯函数**
（`load_state` / `resolve_state_path` / `build_*_payload` / `scan_*` / `_deck_*` / `_stat_file_cards` …）。
**没搬什么**：`Handler` / `main()` / `make_demo_*` / 内嵌页面（后者在 `rl/dashboard_html.py`）。

**为什么能整体搬（实测，不是推测）**：
- `dashboard.py` 里 **L811 之后的代码没有任何一处直接读这些缓存**（`Handler` 只调 `build_*` 函数，
  `main` 只调 `resolve_state_path` / `make_demo_*`）⇒ 缓存搬走后不会有"读到旧对象"的静默问题；
- `_deck_index()` 里有 `global _DECK_INDEX`（**会在本模块内重绑定**）⇒ 它必须与 `_DECK_INDEX`
  **同模块**，本文件正是如此；
- 外部调用方（`rl/selftest.py`、`scripts/check_dashboard_js.py`）都按 `dash.<名>` 使用 ⇒
  `dashboard.py` 侧做**显式 re-export**即可，调用点**一行未动**。

**逐字保证**：本文件正文由 `dashboard.py` 的原行**整段切片**生成（见 `docs/structure_tier2_2026-09-19.md`），
re-export 前后 `build_replays_payload` / `build_card_stats_payload` 的 JSON 指纹**逐字相同**（已 A/B 对账）。
"""
from __future__ import annotations

import os
import sys
import re
import json
import time
import pickle
import datetime

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

# 仓库根。⚠️ `_PARENT` = `src/clasher_new`（= `rl` 包的父目录，供 `import rl.*`），**不是** `src`。
_REPO_ROOT = os.path.dirname(os.path.dirname(_PARENT))

from rl.config import eval_schedule                      # noqa: E402
from rl.train_health import derive_train_log, health_summary  # noqa: E402

MODEL_COLORS = {
    "main": "#2563eb",
    "push_flow": "#ef4444",
    "counter_flow": "#22c55e",
    "lockdown_flow": "#a855f7",
    "all_decks": "#f59e0b",
    "random_deck": "#ea580c",
    "heuristic": "#16a34a",
    "random": "#64748b",
    "exploiter": "#dc2626",
}
MODEL_LABELS = {
    "main": "main（跟随者）",
    "push_flow": "推进流 (60)",
    "counter_flow": "防守反击流 (120)",
    "lockdown_flow": "自闭流 (20)",
    "all_decks": "全 200 卡组",
    "random_deck": "全随机",
    "heuristic": "启发式",
    "random": "随机",
    "exploiter": "exploiter",
}
# 任意 *_ckpt 快照用紫色
CKPT_COLOR = "#7c3aed"
FALLBACK_COLORS = ["#0891b2", "#d946ef", "#65a30d", "#a16207"]

#: 回放文件元数据缓存：{path: ((mtime_ts, size), n_games)}，避免每 5s 重复反序列化大 pickle
_REPLAY_META_CACHE = {}


def load_state(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_state_path(path):
    """`--state` 既接受 JSON 文件，也接受 `runs/<name>` 目录（自动找 league_state.json）。

    2026-09-18 起 `--solo`/`--play` 都支持目录，联赛面板是唯一还需要写全文件名的入口——
    长跑（1M 步）期间每 3 秒轮询要盯着 `runs/long1m/`，记文件名是纯摩擦。
    """
    if not path:
        return None
    p = os.path.abspath(path)
    if os.path.isdir(p):
        cand = os.path.join(p, "league_state.json")
        return cand if os.path.exists(cand) else None
    return p


def _league_run_meta(state_path, state):
    """长跑实时进度（run_state.json 的 step + config.json 的两级评估超参）。

    为什么需要：`league_state.total_steps` 只是**最近一个评估点**的步数，1M 步长跑里
    它每 5 分钟才跳一次、且和"总共要跑到哪"无关 ⇒ 没有 run_state.json 就只能显示
    "总训练步数：16000"，看起来像跑完了。这里给出 step/total/百分比/评估点进度/ETA。

    ETA 用**观测节奏外推**（`step ÷ (now − main_ckpt_0.pt 的 mtime)`）——包含评估占用，
    正是墙钟；标注为"粗估"。`state_age_s` 用于前端判断训练是不是卡死/崩了
    （小点节奏 ≈5.5 分钟，>15 分钟没写入就变红）。
    """
    d = os.path.dirname(os.path.abspath(state_path))
    rs = load_state(os.path.join(d, "run_state.json")) or {}
    cf = load_state(os.path.join(d, "config.json")) or {}
    total = int(rs.get("total_steps") or cf.get("total_steps") or 0)
    cur = int(rs.get("step") or state.get("total_steps") or 0)
    plan = eval_schedule(cf.get("steps_per_eval"), cf.get("big_eval_every"),
                         cf.get("n_eval_games"), cf.get("n_eval_games_big"),
                         total, big_at_start=bool(cf.get("eval_big_at_start", True)))
    n_big = sum(1 for _s, k, _n in plan if k == "big")
    done = sum(1 for _s, _k, _n in plan if _s <= cur and cur > 0)
    try:
        age = max(0.0, time.time() - os.path.getmtime(state_path))
    except OSError:
        age = None
    started_m = None
    ck0 = os.path.join(d, "main_ckpt_0.pt")
    if os.path.exists(ck0) and cur > 0:
        try:
            started_m = max(0.0, (time.time() - os.path.getmtime(ck0)) / 60.0)
        except OSError:
            started_m = None
    eta_m = None
    if started_m and started_m > 0.5 and total > cur:
        eta_m = (total - cur) / (cur / started_m)
    return {
        "cur_step": cur, "total_steps": total,
        "pct": round(100.0 * cur / total, 2) if total else None,
        "plan_points": len(plan), "plan_big": n_big, "plan_small": len(plan) - n_big,
        "points_done": len(state.get("round_stats") or []), "plan_done": done,
        "steps_per_eval": cf.get("steps_per_eval"),
        "big_eval_every": cf.get("big_eval_every"),
        "n_eval_games": cf.get("n_eval_games"), "n_eval_games_big": cf.get("n_eval_games_big"),
        "only_vs_main": cf.get("only_vs_main"), "eval_workers": cf.get("eval_workers"),
        "n_envs": cf.get("n_envs"), "device": cf.get("device"),
        "state_age_s": round(age, 1) if age is not None else None,
        "elapsed_min": round(started_m, 1) if started_m is not None else None,
        "eta_min": round(eta_m, 1) if eta_m is not None else None,
        # 前端配色阈值：小点节奏 ≈5.5 min；>15 min 未写入 ⇒ 疑似卡死/退出
        "stale": bool(age is not None and age > 900),
        "schedule": [[int(s), k, int(n)] for s, k, n in plan],
    }


def _winrate_curves(state):
    """从 `history` + `round_stats` 派生**逐点对手胜率曲线**（`league_state` 只存 EMA 标量）。

    `history` 是逐局 `[a, b, score_a]` 的追加流（**不含 step**），但 `round_stats[i].games[aid]`
    给出该评估点每方的局数 ⇒ `sum(games)/2` 就是该点的总对局数，按此切分即可复原
    "第 i 个评估点、a 对 b 的胜率"。返回 `(curves, counts)`：
    `curves["a|b"] = [[step, winrate], ...]`、`counts["a|b"] = [该点局数, ...]`（前端算 ±SE）。
    局数不足的最后一个点（正在写）直接丢弃，避免曲线上出现半截读数的假跳变。
    """
    hist = state.get("history") or []
    curves, counts = {}, {}
    cur = 0
    for rt in state.get("round_stats") or []:
        g = rt.get("games") or {}
        try:
            n = int(sum(int(v) for v in g.values()) // 2)
        except (TypeError, ValueError):
            n = 0
        if n <= 0:
            continue
        seg = hist[cur:cur + n]
        cur += n
        if len(seg) < n:
            break
        agg = {}
        for row in seg:
            try:
                a, b, sc = row[0], row[1], float(row[2])
            except (TypeError, ValueError, IndexError):
                continue
            w, d, m = agg.get(f"{a}|{b}", (0.0, 0.0, 0))
            agg[f"{a}|{b}"] = (w + (1.0 if sc == 1.0 else 0.0),
                               d + (1.0 if sc == 0.5 else 0.0), m + 1)
        stp = int(rt.get("step", 0))
        for key, (w, d, m) in agg.items():
            if m <= 0:
                continue
            curves.setdefault(key, []).append([stp, round((w + 0.5 * d) / m, 4)])
            counts.setdefault(key, []).append(m)
    return curves, counts


def build_payload(path):
    # 错误时也返回完整结构（空 agents/elo_history 等），前端渲染链不依赖 ok 分支
    empty = {"agents": [], "elo_history": {}, "round_stats": [], "total_steps": 0,
             "winrate_curves": {}, "winrate_counts": {}, "winrates": {}, "run_meta": None}
    if path is None:
        return {"ok": False, "error": "未指定 --state（联赛面板关闭；可看 --solo/--sweep/--play 面板）",
                "state_path": None, **empty}
    st = load_state(path)
    if st is None:
        return {"ok": False, "error": f"状态文件不存在: {path}", "state_path": path, **empty}
    agents = []
    for a in st.get("agents", []):
        aid = a["agent_id"]
        agents.append({
            "id": aid,
            "label": MODEL_LABELS.get(aid, aid),
            "kind": a.get("kind", "baseline"),
            "path": a.get("path"),
            "elo": round(st.get("ratings", {}).get(aid, 1500.0), 1),
        })
    agents.sort(key=lambda x: -x["elo"])
    elo_history = {
        k: [[float(x), float(y)] for x, y in v]
        for k, v in st.get("elo_history", {}).items()
    }
    run_meta = _league_run_meta(path, st)
    # 给每个评估点标注大/小（来自同一份 eval_schedule ⇒ 与训练侧不可能漂移）
    kind_by_step = {s: k for s, k, _n in run_meta.get("schedule") or []}
    games_by_step = {s: n for s, _k, n in run_meta.get("schedule") or []}
    round_stats = []
    for rt in st.get("round_stats", []):
        rt = dict(rt)
        stp = int(rt.get("step", 0))
        rt["kind"] = kind_by_step.get(stp)
        rt["games_per_pair"] = games_by_step.get(stp)
        round_stats.append(rt)
    wr_curves, wr_counts = _winrate_curves(st)
    return {
        "ok": True,
        "agents": agents,
        "elo_history": elo_history,
        "round_stats": round_stats,   # [{step, est:{aid:[R,SE]}, games:{aid:n}, kind, games_per_pair}]
        "total_steps": int(st.get("total_steps", 0)),
        "winrates": {k: round(float(v), 4) for k, v in (st.get("winrates") or {}).items()},
        "winrate_curves": wr_curves,      # {"main|push_flow": [[step, wr], ...]}
        "winrate_counts": wr_counts,      # {"main|push_flow": [n, ...]}（±SE 用）
        "run_meta": run_meta,             # 长跑进度/ETA/评估点计划（dashboard 进度条）
        "demo": bool(st.get("demo", False)),   # --demo 生成的合成数据标记
        "state_path": path,
        "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }


# ---------------------------------------------------------------------------
# flow-sweep 数据效率 A/B（flow_league.run_flow_sweep 逐轮增量写 summary.json）
# ---------------------------------------------------------------------------

#: 策略展示顺序（stream 先，games5 后；未知策略按字母序排后面）
_SWEEP_PRIORITY = ["stream", "games5"]


def scan_sweep_dirs(root):
    """扫描 sweep 根目录 → 策略目录列表。

    root 可以是：配置目录（runs/<name>/，含 flow_sweep_* 子目录），
    或单个策略目录（自身含 summary.json）。找不到返回 []。
    """
    if not root or not os.path.isdir(root):
        return []
    if os.path.isfile(os.path.join(root, "summary.json")):
        return [root]
    found = []
    for name in os.listdir(root):
        if name.startswith("flow_sweep_") and \
                os.path.isfile(os.path.join(root, name, "summary.json")):
            found.append(os.path.join(root, name))
    found.sort(key=lambda d: (_SWEEP_PRIORITY.index(os.path.basename(d).replace("flow_sweep_", ""))
                              if os.path.basename(d).replace("flow_sweep_", "") in _SWEEP_PRIORITY
                              else 99, os.path.basename(d)))
    return found


def build_sweep_payload(sweep_root):
    """读取 flow_sweep_<strategy>/summary.json（逐轮增量写）→ 前端进度+曲线数据。"""
    if not sweep_root:
        return {"ok": False, "error": "未指定 --sweep 目录", "sweep_root": None}
    dirs = scan_sweep_dirs(sweep_root)
    if not dirs:
        return {"ok": False,
                "error": f"未找到 flow_sweep_* 策略目录（或 summary.json）: {sweep_root}",
                "sweep_root": sweep_root}
    strategies = []
    for d in dirs:
        s = load_state(os.path.join(d, "summary.json"))
        if not s:
            continue
        rows = s.get("rows") or []
        strategies.append({
            "dir": d,
            "strategy": s.get("strategy", os.path.basename(d).replace("flow_sweep_", "")),
            "desc": s.get("desc", ""),
            "status": s.get("status", "done"),
            "run_current": int(s.get("run_current", len(rows))),
            "n_runs": int(s.get("n_runs", 0)),
            "per_run_games": int(s.get("per_run_games", 0)),
            "total_games": int(s.get("total_games", 0)),
            "pool_sizes": s.get("pool_sizes", {}),
            "eval_games": int(s.get("eval_games_per_pair", 0)),
            "elapsed_s": float(s.get("elapsed_s", 0) or 0),
            "eta_s": float(s.get("eta_s", 0) or 0),
            "rows": rows,
            "trend": s.get("trend", {}),
            "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        })
    return {"ok": True, "sweep_root": sweep_root, "strategies": strategies}


# ---------------------------------------------------------------------------
# 训练健康（价值损失 / 策略熵）：数据源 = **训练日志**，不是 solo_state.json
# ---------------------------------------------------------------------------

_HEALTH_CACHE = {}      # {path: (mtime, size, payload)}：3 s 轮询不必每轮重解析 800 行


def build_health_payload(log_path):
    """训练日志 → 熵 / 价值损失稠密曲线 + 平台读数（`/api/health`）。

    ⚠️ 口径（与 `rl/train_health.py` 模块 docstring 一致，别在 UI 上另立一套）：
    - `entropy` = **掩码后**、一次决策内各 decoder 步熵**之和**（nat）⇒ 随 bundle 长度有混杂；
    - `value_loss_raw` = 原始 MSE ⇒ **判读用这个**；`value_loss`（日志 `value=`）是 ÷ v_scale² 的
      缩放量，`value_norm=running` 时 v_scale 会长（A_et 实测 1.75 -> 24.87）⇒ **跨时间不可比**，
      单独看它会得到"损失在降"的**假读数**。UI 所以两条都画、并在注里写明。
    - 读数一律**描述性**，非预注册判据（不得并入 §11.13 的判据集，【R3】）。
    """
    if not log_path:
        return {"ok": False, "error": "未找到训练日志（--train-log 未指定且未能从 --solo 目录推出）",
                "log": None}
    try:
        mtime = os.path.getmtime(log_path)
        size = os.path.getsize(log_path)
    except OSError as e:
        return {"ok": False, "error": f"训练日志不可读: {e}", "log": log_path}
    hit = _HEALTH_CACHE.get(log_path)
    if hit and hit[0] == mtime and hit[1] == size:
        return hit[2]
    try:
        payload = health_summary(log_path)
    except Exception as e:
        return {"ok": False, "error": f"解析失败: {type(e).__name__}: {e}", "log": log_path}
    _HEALTH_CACHE[log_path] = (mtime, size, payload)
    return payload


# ---------------------------------------------------------------------------
# solo 自对弈（train_solo.run_solo 增量写 solo_state.json；无联赛机制）
# ---------------------------------------------------------------------------

def build_solo_payload(path):
    """读取 --mode solo 写出的 solo_state.json → 指标曲线/进度/卡组信息。

    2026-09-17：**不再以胜率为唯一曲线**——除 `history`（含行为/critic/活力共 32 键）外，
    额外透传 `_controls_history`（三路外生对照曲线）与同目录 `gates.json`（行为门禁快照）。
    """
    if not path:
        return {"ok": False, "error": "未指定 --solo 状态文件", "solo_path": None}
    st = load_state(path)
    if st is None:
        return {"ok": False, "error": f"solo 状态文件不存在: {path}", "solo_path": path}
    history = st.get("history") or []
    gates = None
    try:
        gp = os.path.join(os.path.dirname(os.path.abspath(path)), "gates.json")
        if os.path.isfile(gp):
            with open(gp, "r", encoding="utf-8") as f:
                gates = json.load(f)
    except Exception:
        gates = None
    return {
        "ok": True,
        "agents": [{"id": "main", "label": "main（自对弈）", "kind": "main"}],
        "mode": st.get("mode", "solo"),
        "history": history,   # [{step,wins,...,winrate,winrate_se,mean_reward,engagement_rate,...}]
        "controls_history": st.get("_controls_history") or [],   # [{step,vs,winrate,...}]
        "gates": gates,       # {step,ok,baseline,checks[]}
        "total_steps": int(st.get("total_steps", 0)),
        "target_steps": int(st.get("target_steps", 0) or 0),
        "deck": st.get("deck", []),
        "opponent": st.get("opponent", "self-play-frozen-copy"),
        "copy_every": int(st.get("copy_every", 0) or 0),
        "status": st.get("status", "done"),
        "demo": bool(st.get("demo", False)),
        "solo_path": path,
        "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }


# ---------------------------------------------------------------------------
# 回放：扫描 / 列表 / 加载（联赛录像 league_<step>.pkl，见 rl/replay.py schema 3）
# ---------------------------------------------------------------------------

def _parse_replay_step(fn):
    m = re.match(r"league_(\d+)\.pkl$", fn)
    return int(m.group(1)) if m else None


def _replay_n_games(path, mtime_ts, size):
    """返回回放文件对局数；带 (mtime,size) 缓存避免反复加载大文件。"""
    key = (mtime_ts, size)
    hit = _REPLAY_META_CACHE.get(path)
    if hit and hit[0] == key:
        return hit[1]
    n = None
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        games = data["games"] if isinstance(data, dict) else data
        n = len(games) if isinstance(games, list) else None
    except Exception:
        n = None
    _REPLAY_META_CACHE[path] = (key, n)
    return n


def scan_replays(replays_dir, limit=30):
    """扫描回放目录，返回最近 limit 个 league_*.pkl 的元数据（按修改时间倒序）。"""
    if not replays_dir or not os.path.isdir(replays_dir):
        return []
    items = []
    for fn in os.listdir(replays_dir):
        if not fn.startswith("league_") or not fn.endswith(".pkl"):
            continue
        p = os.path.join(replays_dir, fn)
        try:
            st = os.stat(p)
        except OSError:
            continue
        items.append({
            "file": fn,
            "step": _parse_replay_step(fn),
            "size": st.st_size,
            "mtime": datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
            "mtime_ts": st.st_mtime,
            "n_games": _replay_n_games(p, st.st_mtime, st.st_size),
        })
    items.sort(key=lambda x: x["mtime_ts"], reverse=True)
    return items[:limit]


def build_replays_payload(replays_dir):
    if not replays_dir or not os.path.isdir(replays_dir):
        return {"ok": False, "error": "未找到回放目录", "replays_dir": replays_dir or ""}
    replays = scan_replays(replays_dir)
    return {
        "ok": True,
        "replays_dir": replays_dir,
        "replays": replays,
        "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }


def load_replay_payload(replays_dir, filename, game_idx=None):
    """加载单个回放文件。

    - game_idx=None：返回对局列表元数据（不含帧，轻量）；
    - game_idx 指定：返回该局完整帧（供播放器）。
    """
    if not filename or "/" in filename or "\\" in filename or ".." in filename:
        return {"ok": False, "error": "非法回放文件名"}
    p = os.path.join(replays_dir, filename)
    if not os.path.isfile(p):
        return {"ok": False, "error": f"回放文件不存在: {filename}"}
    try:
        with open(p, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        return {"ok": False, "error": f"读取回放失败: {e}"}
    games = data["games"] if isinstance(data, dict) else data
    if not isinstance(games, list):
        return {"ok": False, "error": "回放格式无法识别"}
    if game_idx is None:
        out = []
        for gi, g in enumerate(games):
            frames = g.get("frames") or []
            out.append({
                "index": gi,
                "pair": list((g.get("meta") or {}).get("pair", [])),
                "steps": list((g.get("meta") or {}).get("steps", [])),
                "side0": (g.get("meta") or {}).get("side0"),
                "winner": g.get("winner"),
                "n_frames": len(frames),
                "duration": round(frames[-1]["t"], 2) if frames else 0.0,
            })
        return {"ok": True, "file": filename, "games": out}
    if not 0 <= game_idx < len(games):
        return {"ok": False, "error": f"对局索引越界: {game_idx}"}
    g = games[game_idx]
    return {
        "ok": True,
        "file": filename,
        "index": game_idx,
        "meta": g.get("meta", {}),
        "winner": g.get("winner"),
        "frames": g.get("frames", []),
    }


# ---------------------------------------------------------------------------
# 卡牌使用统计（各卡组/模型的实际出牌）
# ---------------------------------------------------------------------------
# 数据口径：
# - 对手侧（player-1）：帧里的 ``opp_played`` 已带卡名，**所有录像**都能统计；
# - 我方侧（player-0）：新录像帧的 ``cards``（record 时解析的手牌卡名），旧录像没有；
# - 卡组构成：新录像 ``meta["decks"]``（双方本局实际卡组）。
# 联赛双方轮流当先手（side0），因此每个模型都会有一半对局以 player-1 身份出现 ——
# 即便对旧录像，"每个模型打了哪些牌"依然可统计（只是每局只覆盖一侧）。

#: 单文件卡牌统计缓存：{path: ((mtime_ts, size), stats)}，切换统计范围时免重复反序列化
_CARD_STATS_CACHE = {}


def _agent_label(mid):
    """模型 id → 友好名（对齐 Elo 面板；frozen_copy 视作 main 的冻结副本）。"""
    if mid == "frozen_copy":
        return "main（冻结副本）"
    return MODEL_LABELS.get(mid, mid)


def _other_side_id(meta):
    """由 meta.pair / meta.side0 推出 player-1 一侧的模型 id（opp_played 的归属）。"""
    pair = [str(x) for x in (meta.get("pair") or [])]
    if len(pair) < 2:
        return None
    side0 = meta.get("side0")
    if side0 == pair[0]:
        return pair[1]
    if side0 == pair[1]:
        return pair[0]
    return pair[1]


def _new_agent(mid):
    return {"model": mid, "label": _agent_label(mid), "games": 0, "plays": 0,
            "cards": {}, "deck_cards": {}, "n_decks": 0}


# --- 按「卡组」维度的统计（多卡组对战：行=卡牌、列=卡组） -------------------
# 2026-09-17 新增：solo 是双方同一副牌，`--mode run` / `flow` 是真正的多卡组
# （run_league.py:559-565 的 5 个脚本 agent 各带卡组池）⇒ 需要按**卡组**而不是按**模型**
# 汇总出牌，才能回答"哪个卡组爱打哪张卡"。

_DECK_INDEX = None      # deck_key -> archetype（来自 docs/leaderboard_decks_classified.json）


def _deck_key(deck):
    """卡组指纹：卡名排序后拼接（与顺序无关，双方同副牌的不同排列视为同一卡组）。"""
    cards = [str(c) for c in (deck or []) if c]
    if not cards:
        return None
    return "|".join(sorted(cards))


def _deck_index():
    """懒加载「卡组 → archetype」索引（200 副天梯卡组，rl/decks.py）。失败即空表。"""
    global _DECK_INDEX
    if _DECK_INDEX is None:
        idx = {}
        try:
            from rl.decks import load_classified_decks
            for d in load_classified_decks():
                k = _deck_key(d.get("cards") or [])
                if k and k not in idx:
                    idx[k] = d.get("archetype") or "未知"
        except Exception:
            idx = {}
        _DECK_INDEX = idx
    return _DECK_INDEX


def _new_deck(key, deck):
    arch = _deck_index().get(key)
    cards = [str(c) for c in (deck or [])]
    # 未知卡组给一个短指纹（列多时可区分），完整 8 张在列标题的 title 里
    short = "·".join(sorted(cards)[:3]) + ("…" if len(cards) > 3 else "")
    return {"key": key, "deck": cards,
            "label": arch or ("自定义 " + short), "archetype": arch,
            "known": arch is not None,
            "side_games": 0, "plays": 0, "cards": {}}


def _deck_bucket(decks, deck):
    key = _deck_key(deck)
    if not key:
        return None
    return decks.setdefault(key, _new_deck(key, deck))


def _stat_file_cards(path):
    """统计单个回放文件里"每个模型 / 每个卡组打了哪些牌"。带 (mtime,size) 缓存。"""
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (st.st_mtime, st.st_size)
    hit = _CARD_STATS_CACHE.get(path)
    if hit and hit[0] == key:
        return hit[1]
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except Exception:
        return None
    games = data["games"] if isinstance(data, dict) else data
    if not isinstance(games, list):
        return None

    agents = {}
    deck_stats = {}
    n_games = side0_games = side1_games = games_with_decks = 0

    def _agent(mid):
        return agents.setdefault(mid, _new_agent(mid))

    for g in games:
        meta = g.get("meta") or {}
        pair = list(meta.get("pair") or [])
        side0_id = meta.get("side0") or (pair[0] if pair else None)
        side1_id = _other_side_id(meta)
        meta_decks = meta.get("decks")
        deck0 = deck1 = None
        if isinstance(meta_decks, list) and len(meta_decks) == 2:
            deck0, deck1 = meta_decks[0], meta_decks[1]
            games_with_decks += 1
        b0 = _deck_bucket(deck_stats, deck0)
        b1 = _deck_bucket(deck_stats, deck1)
        n_games += 1
        seen0 = seen1 = False
        for fr in (g.get("frames") or []):
            # 对手侧：opp_played（结构化 [{card,x,y}]，含卡名）
            if side1_id:
                for p in (fr.get("opp_played") or []):
                    c = p.get("card") if isinstance(p, dict) else None
                    if not c or c == "__ability__":
                        continue
                    a = _agent(side1_id)
                    a["cards"][c] = a["cards"].get(c, 0) + 1
                    a["plays"] += 1
                    if b1 is not None:
                        b1["cards"][c] = b1["cards"].get(c, 0) + 1
                        b1["plays"] += 1
                    seen1 = True
            # 我方侧：cards（新录像才有）
            if side0_id:
                for c in (fr.get("cards") or []):
                    if not c:
                        continue
                    a = _agent(side0_id)
                    a["cards"][c] = a["cards"].get(c, 0) + 1
                    a["plays"] += 1
                    if b0 is not None:
                        b0["cards"][c] = b0["cards"].get(c, 0) + 1
                        b0["plays"] += 1
                    seen0 = True
        if side0_id:
            _agent(side0_id)["games"] += 1
        if side1_id:
            _agent(side1_id)["games"] += 1
        side0_games += 1 if seen0 else 0
        side1_games += 1 if seen1 else 0
        # 卡组构成（meta.decks，新录像才有）+ 每卡组的出战边数
        for mid, deck, b in ((side0_id, deck0, b0), (side1_id, deck1, b1)):
            if not mid:
                continue
            if deck:
                a = _agent(mid)
                a["n_decks"] += 1
                for c in (deck or []):
                    a["deck_cards"][c] = a["deck_cards"].get(c, 0) + 1
            if b is not None:
                b["side_games"] += 1

    stats = {"n_games": n_games, "side0_games": side0_games,
             "side1_games": side1_games, "games_with_decks": games_with_decks,
             "agents": agents, "decks": deck_stats}
    _CARD_STATS_CACHE[path] = (key, stats)
    return stats


def build_card_stats_payload(replays_dir, filename=None, n_files=3):
    """汇总回放的卡牌使用统计。

    filename 给定 → 只统计该文件；否则汇总最近 n_files 个（n_files<=0 = 全部）。
    """
    if not replays_dir or not os.path.isdir(replays_dir):
        return {"ok": False, "error": "未找到回放目录", "replays_dir": replays_dir or ""}
    if filename:
        if "/" in filename or "\\" in filename or ".." in filename:
            return {"ok": False, "error": "非法回放文件名"}
        if not os.path.isfile(os.path.join(replays_dir, filename)):
            return {"ok": False, "error": f"回放文件不存在: {filename}"}
        files = [filename]
    else:
        limit = n_files if (n_files and n_files > 0) else 10 ** 9
        files = [r["file"] for r in scan_replays(replays_dir, limit=limit)]

    agents = {}
    decks = {}
    n_games = side0_games = side1_games = games_with_decks = 0
    used = []
    for fn in files:
        st = _stat_file_cards(os.path.join(replays_dir, fn))
        if not st:
            continue
        used.append(fn)
        n_games += st["n_games"]
        side0_games += st["side0_games"]
        side1_games += st["side1_games"]
        games_with_decks += st.get("games_with_decks", 0)
        for mid, a in st["agents"].items():
            tgt = agents.setdefault(mid, _new_agent(mid))
            tgt["games"] += a["games"]
            tgt["plays"] += a["plays"]
            tgt["n_decks"] += a["n_decks"]
            for c, v in a["cards"].items():
                tgt["cards"][c] = tgt["cards"].get(c, 0) + v
            for c, v in a["deck_cards"].items():
                tgt["deck_cards"][c] = tgt["deck_cards"].get(c, 0) + v
        for key, d in (st.get("decks") or {}).items():
            tgt = decks.get(key)
            if tgt is None:
                tgt = decks[key] = {"key": key, "deck": list(d["deck"]), "label": d["label"],
                                    "archetype": d["archetype"], "known": d["known"],
                                    "side_games": 0, "plays": 0, "cards": {}}
            tgt["side_games"] += d["side_games"]
            tgt["plays"] += d["plays"]
            for c, v in d["cards"].items():
                tgt["cards"][c] = tgt["cards"].get(c, 0) + v

    out = []
    for mid, a in agents.items():
        total = a["plays"]
        top = sorted(a["cards"].items(), key=lambda kv: (-kv[1], kv[0]))
        out.append({
            "model": mid,
            "label": a["label"],
            "games": a["games"],
            "plays": total,
            "n_distinct": len(a["cards"]),
            "per_game": round(total / a["games"], 2) if a["games"] else 0.0,
            "cards": a["cards"],
            "top": [{"card": c, "n": v,
                     "share": round(v / total, 4) if total else 0.0} for c, v in top],
            "deck_cards": a["deck_cards"],
            "n_decks": a["n_decks"],
        })
    out.sort(key=lambda x: -x["plays"])

    decks_out = []
    for key, d in decks.items():
        total = d["plays"]
        top = sorted(d["cards"].items(), key=lambda kv: (-kv[1], kv[0]))
        decks_out.append({
            "key": key,
            "deck": d["deck"],
            "label": d["label"],
            "archetype": d["archetype"],
            "known": d["known"],
            "side_games": d["side_games"],
            "plays": total,
            "n_distinct": len(d["cards"]),
            "per_side_game": round(total / d["side_games"], 2) if d["side_games"] else 0.0,
            "cards": d["cards"],
            "top": [{"card": c, "n": v,
                     "share": round(v / total, 4) if total else 0.0} for c, v in top],
        })
    decks_out.sort(key=lambda x: -x["plays"])

    return {
        "ok": True,
        "files": used,
        "n_games": n_games,
        "agents": out,
        "decks": decks_out,
        "n_decks": len(decks_out),
        "coverage": {
            "side0_games": side0_games,
            "side1_games": side1_games,
            "n_games": n_games,
            "games_with_decks": games_with_decks,
            # 旧录像只有对手侧可统计 → 前端据此提示"部分覆盖"
            "partial": side0_games < n_games,
            # 旧录像没有 meta.decks ⇒ 无法按卡组切分
            "deck_meta_partial": games_with_decks < n_games,
        },
        "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
