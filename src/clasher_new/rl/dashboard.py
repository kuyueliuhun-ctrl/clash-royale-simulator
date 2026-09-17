"""训练网页 UI：各模型 Elo-训练次数 曲线仪表盘 + flow-sweep 进度/曲线 + solo 自对弈 + 人机对战 + 回放。

- 读取 ``run_league --mode run`` 写出的联赛状态 JSON（含 elo_history）；
- ``--sweep`` 指向 flow-sweep 产物（runs/<name>/ 或单个 flow_sweep_<strategy> 目录），
  每 3 秒读取逐轮增量写的 ``summary.json``（status/run_current/eta_s）→ 进度条 +
  main 轮内估计 ±1σ 曲线（flow_sweep_stream / flow_sweep_games5 两个策略并排显示）；
- ``--solo`` 指向 solo 自对弈状态（solo_state.json）→ **指标多选器** + 训练进度。
  ⚠️ **2026-09-17 起不再以胜率为曲线指标**：AGENTS §3 判读禁则明确「不判 main 曲线与单点胜率」
  （main vs 刚同步的冻结副本 = 自引用读数，结构性 ≈0.5）⇒ 默认画**行为指标**
  （接敌率 / 单边堆牌 / 平均圣水），下拉可切防守链 / 部署量 / 回报 / critic / GRU 活力 /
  外生对照（`_controls_history` 的 baseline0·prev·rand）/ 以及「⚠ 自引用（禁读）」分组的胜率。
  数据源 = ``history[]``（逐评估点 32 键）与 ``_controls_history``；指标定义见 ``SOLO_METRICS``；
- ``--play`` 指向 FollowerPolicy checkpoint 或 runs/<name> 目录 → **人机对战页面**：
  人在浏览器点手牌 + 点格子出牌（player-0），对手 = 训练模型（player-1）；每步记录
  EpisodeReplay（含 hidden）与 BC 样本，落盘 ``--play-out`` 供 train_belief / BC 训练；
- 扫描 ``<state 同目录>/replays/league_<step>.pkl`` 联赛录像，列出最近回放；
- 浏览器内 Canvas 播放器回放单局（纯前端自绘，无外部 CDN 依赖，离线可用）；
- **卡牌使用统计**（``/api/cardstats``）：两种切分 —— ①**按模型**（行=卡牌、列=模型）；
  ②**按卡组**（行=卡牌、列=卡组，2026-09-17 新增，面向 ``--mode run``/``flow`` 的**多卡组对战**：
  卡组指纹 = 卡名排序（与顺序无关），列标题悬停可见该卡组 8 张卡，不在 200 副天梯池里的
  未知卡组给短指纹）。单元格 = 出牌次数（热力配色）+ 占该列总出牌的比例。数据口径：
  对手侧取帧的 ``opp_played``（所有录像都有），我方侧取帧的 ``cards``（新录像），
  卡组归属取局级 ``meta.decks``（新录像才有 ⇒ 旧录像「按卡组」表为空并提示）；
  统计范围可选当前打开的回放 / 最近 N 个 / 全部；
- **回归**：``scripts/check_dashboard_js.py``（node + DOM 桩，用真实 payload 跑本文件内嵌
  JS 的渲染冒烟：语法 + 全指标绘制 + 空表/旧录像分支）；
- ``/api/state`` 每 3 秒轮询刷新，``/api/sweep`` / ``/api/solo`` 同频，``/api/replays`` 每 5 秒。

用法：
    python rl/dashboard.py --state league_state.json --sweep runs/economy --port 8700
    python rl/dashboard.py --solo runs/nostall20k --replays runs/nostall20k/replays --port 8700
    python rl/dashboard.py --play runs/solo --port 8700      # 人机对战 + 数据采集
打开 http://127.0.0.1:<port> 查看（⚠️ 8090 在本机被系统保留，见 AGENTS §2.5）。仓库根目录另有 scripts/rl/dashboard.py 包装。
"""

import os
import re
import sys
import json
import time
import pickle
import argparse
import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from rl.replay import save_league_replays
from rl.config import eval_schedule

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


def make_demo_replays(replays_dir, n_games=2, n_frames=40):
    """生成演示用联赛录像（合成帧），便于 --demo 直接预览回放播放器。"""
    import random
    os.makedirs(replays_dir, exist_ok=True)
    out = os.path.join(replays_dir, "league_demo.pkl")
    if os.path.exists(out):
        return
    rng = random.Random(1)
    games = []
    for g in range(n_games):
        frames = []
        t = 0.0
        towers0 = [4824.0, 3052.0, 3052.0]
        towers1 = [4824.0, 3052.0, 3052.0]
        elix0, elix1 = 5.0, 5.0
        for i in range(n_frames):
            t += 0.7
            if i > 10:
                towers1[1] = max(0.0, towers1[1] - 42.0)
            if i > 25:
                towers0[2] = max(0.0, towers0[2] - 34.0)
            elix0 = min(10.0, elix0 + 0.14)
            elix1 = min(10.0, elix1 + 0.14)
            entities = [
                ["KingTower", 9.0, 3.0, towers0[0], 0],
                ["KingTower", 9.0, 29.0, towers1[0], 1],
                ["King_PrincessTowers", 3.5, 6.5, towers0[1], 0],
                ["King_PrincessTowers", 14.5, 6.5, towers0[2], 0],
                ["King_PrincessTowers", 3.5, 25.5, towers1[1], 1],
                ["King_PrincessTowers", 14.5, 25.5, towers1[2], 1],
                ["Knight", 4.0 + i * 0.10, 12.0 + rng.uniform(-0.3, 0.3), 700.0, 0],
                ["Archers", 12.0 - i * 0.12, 20.0 + rng.uniform(-0.3, 0.3), 250.0, 1],
            ]
            frames.append({
                "t": round(t, 2),
                "bundle": [["deploy", 2, 4.0, 12.0]] if i % 6 == 0 else [],
                "reward": round(rng.uniform(-0.2, 0.3), 4),
                "opp_played": [{"card": "Archers", "x": 12.0, "y": 20.0}] if i % 7 == 0 else [],
                "towers0": towers0[:], "towers1": towers1[:],
                "elixir0": round(elix0, 2), "elixir1": round(elix1, 2),
                "crown0": 1 if i > 25 else 0, "crown1": 0,
                "entities": entities,
            })
        games.append({
            "meta": {"pair": ["main", "push_flow"], "side0": "main", "max_steps": 600},
            "winner": 0 if g == 0 else 1,
            "frames": frames,
        })
    save_league_replays(games, out)
    print(f"[demo] 已生成演示回放 -> {out}（{n_games} 局 × {n_frames} 帧）")


# ---------------------------------------------------------------------------
# 页面（Elo 仪表盘 + 最近回放列表 + Canvas 播放器）
# ---------------------------------------------------------------------------

_HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RL 联赛训练仪表盘 · Elo / 回放</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
         background:#0f172a; color:#e2e8f0; }
  header { padding:18px 24px; border-bottom:1px solid #1e293b;
           display:flex; align-items:center; gap:16px; flex-wrap:wrap; }
  h1 { font-size:20px; margin:0; font-weight:600; }
  .sub { color:#94a3b8; font-size:13px; }
  #status { margin-left:auto; font-size:13px; color:#94a3b8; }
  main { padding:20px 24px; display:grid; grid-template-columns: 2fr 1fr; gap:20px; }
  @media (max-width: 900px) { main { grid-template-columns: 1fr; } }
  .card { background:#1e293b; border:1px solid #334155; border-radius:10px; padding:16px;
          margin:0 24px 20px; }
  main .card { margin:0; }
  .card h2 { font-size:15px; margin:0 0 12px; color:#cbd5e1; display:flex;
             align-items:center; gap:12px; flex-wrap:wrap; }
  canvas { width:100%; height:420px; display:block; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th, td { text-align:left; padding:7px 8px; border-bottom:1px solid #334155; }
  th { color:#94a3b8; font-weight:500; }
  .dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; }
  .legend { display:flex; flex-wrap:wrap; gap:10px 18px; margin:10px 0 4px; font-size:13px; }
  .legend span { display:inline-flex; align-items:center; gap:6px; }
  #tooltip { position:absolute; pointer-events:none; background:#0f172a; border:1px solid #475569;
             border-radius:6px; padding:6px 9px; font-size:12px; display:none; z-index:10; }
  .kind { font-size:11px; padding:1px 7px; border-radius:999px; background:#334155; }
  .kind.main { background:#1d4ed8; } .kind.baseline { background:#334155; }
  .kind.historical { background:#6d28d9; } .kind.exploiter { background:#991b1b; }
  /* —— flow-sweep 进度 / 曲线 —— */
  .strategy-card { border:1px solid #334155; border-radius:10px; padding:14px; margin-bottom:14px;
                   background:#16213a; }
  .strategy-head { display:flex; align-items:center; gap:12px; flex-wrap:wrap; margin-bottom:10px; }
  .strategy-head h3 { margin:0; font-size:14px; color:#cbd5e1; }
  .chip { font-size:11px; padding:1px 9px; border-radius:999px; white-space:nowrap; }
  .chip.running { background:#1d4ed8; color:#dbeafe; }
  .chip.done { background:#065f46; color:#6ee7b7; }
  .chip.up { background:#065f46; color:#6ee7b7; }
  .chip.flat { background:#334155; color:#cbd5e1; }
  .chip.down { background:#7f1d1d; color:#fca5a5; }
  .progress { display:flex; align-items:center; gap:10px; flex:1; min-width:220px; }
  .progress .bar { flex:1; height:10px; background:#0f172a; border:1px solid #334155;
                   border-radius:999px; overflow:hidden; }
  .progress .bar i { display:block; height:100%; background:#2563eb; border-radius:999px;
                     transition:width .5s; }
  .progress .pct { font-size:12px; color:#94a3b8; white-space:nowrap; }
  .sweep-layout { display:grid; grid-template-columns: 1.6fr 1fr; gap:14px; }
  @media (max-width: 900px) { .sweep-layout { grid-template-columns: 1fr; } }
  .sweep-layout canvas { height:260px; }
  .sweep-layout .sub { font-size:12px; }
  /* —— solo 自对弈（无联赛）—— */
  .solo-head { display:flex; align-items:center; gap:12px; flex-wrap:wrap; margin-bottom:10px; }
  .solo-metricbar { display:flex; align-items:center; gap:12px; flex-wrap:wrap; margin-bottom:6px; }
  .solo-metricbar .btn { width:auto; }
  .solo-layout { display:grid; grid-template-columns: 1.6fr 1fr; gap:14px; }
  @media (max-width: 900px) { .solo-layout { grid-template-columns: 1fr; } }
  .solo-layout canvas { height:300px; }
  .solo-layout .sub { font-size:12px; }
  #soloDeck { font-family:ui-monospace,Consolas,monospace; color:#cbd5e1; }
  /* —— 人机对战 —— */
  .hand-row { display:flex; gap:8px; margin:12px 0; flex-wrap:wrap; }
  .hand-card { padding:10px 14px; background:#0f172a; border:1px solid #334155;
               border-radius:8px; cursor:pointer; font-size:12px; color:#e2e8f0;
               user-select:none; }
  .hand-card:hover { border-color:#64748b; }
  .hand-card.sel { border-color:#60a5fa; background:#1e3a8a; }
  #playArena { cursor:crosshair; }
  /* —— 回放列表 / 播放器 —— */
  .rep-list { display:flex; flex-direction:column; gap:6px; }
  .rep-row { display:flex; justify-content:space-between; align-items:center; gap:12px;
             padding:9px 12px; background:#0f172a; border:1px solid #334155;
             border-radius:8px; cursor:pointer; }
  .rep-row:hover { border-color:#64748b; }
  .rep-name { font-weight:600; color:#e2e8f0; }
  .rep-meta { color:#94a3b8; font-size:12px; }
  .btn { background:#334155; color:#e2e8f0; border:1px solid #475569; border-radius:6px;
         padding:5px 12px; cursor:pointer; font-size:12px; font-family:inherit; }
  .btn:hover { background:#475569; }
  .btn.sm { padding:3px 10px; }
  .badge { font-size:11px; padding:1px 8px; border-radius:999px; }
  .badge.w0 { background:#1d4ed8; } .badge.w1 { background:#991b1b; } .badge.wd { background:#334155; }
  /* —— 卡牌使用统计 —— */
  .stats-agents { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:12px; }
  .stats-agent { background:#0f172a; border:1px solid #334155; border-radius:8px;
                 padding:8px 11px; font-size:12px; }
  .stats-agent b { color:#e2e8f0; }
  .stats-agent .m { color:#94a3b8; }
  .stats-scroll { max-height:460px; overflow:auto; border:1px solid #334155; border-radius:8px; }
  .stats-h { font-size:13px; color:#cbd5e1; margin:14px 0 6px; display:flex; align-items:baseline; gap:8px; }
  .stats-h:first-of-type { margin-top:10px; }
  table.cards { font-size:12px; }
  table.cards th { position:sticky; top:0; background:#1e293b; z-index:1; }
  table.cards td, table.cards th { padding:5px 8px; white-space:nowrap;
                                   border-bottom:1px solid #26334a; }
  table.cards td.card-name { font-family:ui-monospace,Consolas,monospace; color:#cbd5e1; }
  table.cards td.num, table.cards th.num { text-align:right; font-variant-numeric:tabular-nums; }
  .cellbar { display:inline-block; height:10px; border-radius:3px; background:#60a5fa;
             vertical-align:middle; margin-right:6px; }
  .player-grid { display:grid; grid-template-columns: minmax(280px, 420px) 1fr; gap:18px; }
  @media (max-width: 900px) { .player-grid { grid-template-columns: 1fr; } }
  .arena-wrap canvas { width:100%; height:auto; max-height:74vh; background:#0b3d2e;
                       border-radius:8px; }
  .info { font-size:13px; line-height:1.9; }
  .info .kv { display:flex; gap:10px; }
  .info .kv span { color:#94a3b8; min-width:64px; flex-shrink:0; }
  .controls { display:flex; align-items:center; gap:8px; margin-top:14px; flex-wrap:wrap; }
  #scrub { flex:1; min-width:120px; }
  .legend-row { margin-top:12px; font-size:12px; color:#94a3b8; display:flex; gap:16px; }
</style>
</head>
<body>
<header>
  <h1>RL 联赛训练仪表盘</h1>
  <span class="sub">Elo vs 训练次数 · 5 模型同时维护 · 最近回放</span>
  <span id="datasrc" style="font-size:12px;color:#94a3b8"></span>
  <span id="status">加载中…</span>
</header>
<main>
  <div class="card">
    <h2>Elo 曲线（训练步数）
      <span class="sub" id="leagueSub"></span>
      <select id="leagueMetric" class="btn" title="曲线指标" onchange="drawChart()">
        <option value="elo">指标：轮内估计 Elo</option>
        <option value="winrate">指标：对每个对手的胜率</option>
      </select>
    </h2>
    <div class="progress" id="leagueProgress" style="margin-bottom:10px"></div>
    <div class="legend" id="legend"></div>
    <div style="position:relative">
      <canvas id="chart"></canvas>
      <div id="tooltip"></div>
    </div>
  </div>
  <div class="card">
    <h2>当前排名</h2>
    <table>
      <thead><tr><th>模型</th><th>类型</th><th>Elo</th><th>Δ上轮</th><th>σ 信号/噪声</th><th>vs 对手胜率(EMA)</th><th>checkpoint</th></tr></thead>
      <tbody id="tbody"></tbody>
    </table>
    <p id="meta" class="sub" style="margin-top:12px"></p>
  </div>
</main>

<section class="card" id="sweepCard" style="display:none">
  <h2>flow-sweep 数据效率 A/B（main 轮内估计 ±1σ 噪声棒）
    <span class="sub" id="sweepSub"></span>
  </h2>
  <div id="sweepBody"></div>
</section>

<section class="card" id="soloCard" style="display:none">
  <h2>solo 自对弈（固定卡组镜像 · 无联赛）
    <span class="sub" id="soloSub"></span>
  </h2>
  <div class="solo-head">
    <span class="chip" id="soloStatus"></span>
    <div class="progress" id="soloProgress"></div>
    <span class="sub" id="soloMeta"></span>
  </div>
  <div class="solo-metricbar">
    <label class="sub" for="soloMetric" style="margin:0">曲线指标</label>
    <select id="soloMetric" class="btn">
      <optgroup label="行为（默认 · AGENTS §3：行为是相位型，只读相对退化）">
        <option value="behavior" selected>接敌率 · 单边堆牌 · 平均圣水</option>
        <option value="defense">防守投入 · 响应延迟 · 拦截率</option>
        <option value="deploy">deploy/局 · 多卡同帧 · 塔血差</option>
      </optgroup>
      <optgroup label="回报">
        <option value="reward">mean_reward（逐点）</option>
      </optgroup>
      <optgroup label="critic / 活力">
        <option value="critic">critic EV · value÷R std</option>
        <option value="vitality">GRU h 跨帧 std · n(abs)</option>
      </optgroup>
      <optgroup label="外生对照（唯一可测绝对强度的仪器）">
        <option value="controls">baseline0 / prev / rand 胜率</option>
      </optgroup>
      <optgroup label="⚠ 自引用（AGENTS §3 判读禁则：不得当变强证据）">
        <option value="winrate">main vs 冻结副本 胜率 ±1σ</option>
      </optgroup>
    </select>
    <span class="legend" id="soloMetricLegend" style="margin:0"></span>
  </div>
  <p class="sub" id="soloMetricNote" style="margin:0 0 8px"></p>
  <div class="solo-layout">
    <div style="position:relative"><canvas id="soloChart"></canvas></div>
    <div>
      <p class="sub" style="margin:0 0 6px">卡组（双方同一副）：</p>
      <p id="soloDeck" class="sub"></p>
      <p class="sub" style="margin:10px 0 6px">评估（最新在上）：</p>
      <table>
        <thead><tr><th>步数</th><th>胜/负/平</th><th>胜率 ±SE</th><th>mean_reward</th>
        <th>接敌%</th><th>单边堆牌%</th><th>防守投入%</th><th>响应延迟</th><th>deploy/局</th><th>平均圣水</th>
        <th title="critic 对回报的解释方差 EV（池化口径）：≥0.2 才算价值头在有效学习；≈0 等于只会预测均值；持续为负说明比常数预测还差">critic EV</th>
        <th title="GRU 隐状态跨帧 std（逐维后取均值）：健康 >0.05；≤0.02 说明隐状态被冻成常数（v3 根因：enc 未归一化导致 tanh 候选饱和）">h 跨帧 std</th>
        <th title="GRU 候选 tanh 的 |·| 均值：饱和时 →1.0；健康 <0.9。与 h 跨帧 std 一起判 GRU 是否解冻">GRU n(abs)</th>
        <th title="value_head 输出 std ÷ 批内回报 std（v3 §2 验收第 3 行）：>0.3 说明 critic 输出波动与回报同量级；≈0 说明价值头输出近似常数">value/R std</th></tr></thead>
        <tbody id="soloTable"></tbody>
      </table>
    </div>
  </div>
</section>

<section class="card" id="playCard" style="display:none">
  <h2>人机对战（你 = 蓝方 p0，对手 = 训练模型）
    <span class="sub" id="playSub"></span>
  </h2>
  <div class="player-grid">
    <div class="arena-wrap">
      <canvas id="playArena" width="450" height="800"></canvas>
    </div>
    <div>
      <div id="playInfo" class="info"></div>
      <div id="playHand" class="hand-row"></div>
      <div id="playLegal" class="sub" style="margin:4px 0 10px"></div>
      <div class="controls">
        <button class="btn" id="btnPlayNew">⏹ 结束本局并保存</button>
        <span id="playCounter" class="sub"></span>
      </div>
      <p class="sub" style="margin-top:14px">每出一个动作推进 0.5s 战斗时间；对局数据（含 hidden 特权标签）实时落盘为 EpisodeReplay + BC 样本，可喂 train_belief / train_bc_from_human。</p>
    </div>
  </div>
</section>

<section class="card">
  <h2>最近训练回放
    <span class="sub" id="replaysSub"></span>
    <button class="btn" id="reloadReplays" style="margin-left:auto">刷新</button>
  </h2>
  <div id="replaysList" class="rep-list"></div>
  <div id="gamesPanel" style="display:none">
    <h3 style="margin:16px 0 8px;font-size:14px;color:#cbd5e1">
      <span id="gamesTitle"></span>
      <span class="sub" id="gamesMeta"></span>
    </h3>
    <table>
      <thead><tr><th>局</th><th>对阵</th><th>结果</th><th>帧数</th><th>时长</th><th></th></tr></thead>
      <tbody id="gamesBody"></tbody>
    </table>
    <p id="gamesEmpty" class="sub"></p>
  </div>
</section>

<section class="card" id="statsCard" style="display:none">
  <h2>卡牌使用统计（各卡组 / 模型的实际出牌）
    <span class="sub" id="statsSub"></span>
    <span style="margin-left:auto;display:flex;gap:8px;align-items:center">
      <select id="statsScope" class="btn">
        <option value="3" selected>最近 3 个回放</option>
        <option value="10">最近 10 个回放</option>
        <option value="0">全部回放</option>
        <option value="file">当前打开的回放</option>
      </select>
      <button class="btn" id="statsReload">刷新</button>
    </span>
  </h2>
  <div class="stats-agents" id="statsAgents"></div>
  <h3 class="stats-h">按模型 <span class="sub">行 = 卡牌，列 = 模型</span></h3>
  <div class="stats-scroll"><table class="cards" id="statsTable"></table></div>
  <h3 class="stats-h">按卡组（多卡组对战）
    <span class="sub">行 = 卡牌，列 = 卡组（悬停列标题可见该卡组 8 张卡）</span>
    <span class="sub" id="statsDeckSub"></span></h3>
  <div class="stats-scroll"><table class="cards" id="statsDeckTable"></table></div>
  <p class="sub" id="statsNote" style="margin-top:10px"></p>
</section>

<section class="card" id="playerCard" style="display:none">
  <h2>回放播放器
    <span class="sub" id="playerTitle"></span>
    <button class="btn" id="closePlayer" style="margin-left:auto">× 关闭</button>
  </h2>
  <div class="player-grid">
    <div class="arena-wrap">
      <canvas id="arena" width="450" height="800"></canvas>
    </div>
    <div>
      <div id="playerInfo" class="info"></div>
      <div id="playerControls" class="controls">
        <button id="btnPlay" class="btn">⏵ 播放</button>
        <button id="btnPrev" class="btn" title="上一帧">⏮</button>
        <button id="btnNext" class="btn" title="下一帧">⏭</button>
        <input id="scrub" type="range" min="0" max="0" value="0" step="1">
        <span id="frameLabel" class="sub"></span>
        <select id="speedSel" class="btn">
          <option value="0.5">0.5×</option>
          <option value="1" selected>1×</option>
          <option value="2">2×</option>
          <option value="4">4×</option>
          <option value="8">8×</option>
        </select>
      </div>
      <div class="legend-row">
        <span><span class="dot" style="background:#60a5fa"></span>我方 (p0)</span>
        <span><span class="dot" style="background:#f87171"></span>对手 (p1)</span>
        <span><span class="dot" style="background:#3b82f6"></span>我方塔</span>
        <span><span class="dot" style="background:#ef4444"></span>对手塔</span>
      </div>
    </div>
  </div>
</section>

<script>
const COLORS = {
  main:"#2563eb", push_flow:"#ef4444", counter_flow:"#22c55e", lockdown_flow:"#a855f7",
  all_decks:"#f59e0b", random_deck:"#ea580c", heuristic:"#16a34a", random:"#64748b",
  exploiter:"#dc2626", ckpt:"#7c3aed",
  fallback:["#0891b2","#d946ef","#65a30d","#a16207"]
};
const LABELS = {
  main:"main（跟随者）", push_flow:"推进流 (60)", counter_flow:"防守反击流 (120)",
  lockdown_flow:"自闭流 (20)", all_decks:"全 200 卡组", random_deck:"全随机",
  heuristic:"启发式", random:"随机", exploiter:"exploiter"
};
function colorOf(id){
  if (COLORS[id] !== undefined) return COLORS[id];
  if (id.endsWith("_ckpt")) return COLORS.ckpt;
  let h=0; for (const c of id) h=(h*31+c.charCodeAt(0))>>>0;
  return COLORS.fallback[h % COLORS.fallback.length];
}
let payload = {ok:false, agents:[], elo_history:{}, round_stats:[], total_steps:0,
               winrate_curves:{}, winrate_counts:{}, winrates:{}, run_meta:null};
let sweep = {ok:false, strategies:[]};
let solo = {ok:false, history:[]};

async function refresh(){
  try{
    // 时间戳 query 防任何中间层/浏览器缓存，保证 3s 轮询拿到最新状态
    const [rs, sw, so] = await Promise.all([
      fetch("/api/state?_t=" + Date.now(), {cache: "no-store"}).then(r=>r.json()),
      fetch("/api/sweep?_t=" + Date.now(), {cache: "no-store"}).then(r=>r.json()),
      fetch("/api/solo?_t=" + Date.now(), {cache: "no-store"}).then(r=>r.json())
    ]);
    payload = rs; sweep = sw; solo = so;
    await refreshPlay();   // 先拿到 play 状态再决定状态栏显示
    const src = document.getElementById("datasrc");
    const hasSweep = sweep.ok && sweep.strategies.length;
    const hasSolo = solo.ok;
    const hasPlay = play.ok;
    if (!payload.ok && (hasSweep || hasSolo || hasPlay)){
      // 非联赛模式（flow-sweep / solo / 人机对战）：不显示联赛错误，只显示对应面板
      const s0 = hasSweep ? sweep.strategies[0] : null;
      const up = (hasSweep && s0 && s0.updated_at) || (hasSolo && solo.updated_at) || "";
      const modes = [];
      if (hasSweep) modes.push("flow-sweep");
      if (hasSolo) modes.push("solo 自对弈");
      if (hasPlay) modes.push("人机对战");
      document.getElementById("status").textContent = modes.join(" + ") + " 模式 · 更新于 " + up;
      src.textContent = (hasSweep ? "sweep 根目录: " + (sweep.sweep_root || "") : "") +
        (hasSweep && (hasSolo || hasPlay) ? " · " : "") +
        (hasSolo ? "solo 状态: " + (solo.solo_path || "") : "") +
        ((hasSolo && hasPlay) ? " · " : "") +
        (hasPlay ? "人机对战已就绪" : "");
      src.style.color = "#22c55e";
    } else if (!payload.ok){
      document.getElementById("status").textContent = "⚠ " + payload.error;
      src.textContent = "";
    } else {
      document.getElementById("status").textContent =
        "更新于 " + payload.updated_at;
      const hasHist = Object.values(payload.elo_history || {}).some(h => h.length);
      const rsN = (payload.round_stats || []).length;
      src.textContent = (payload.demo ? "⚠ DEMO 合成数据（非真实训练） · " : "")
        + "状态文件: " + (payload.state_path || "");
      if (payload.demo) src.style.color = "#f59e0b";
      else src.style.color = "#22c55e";
      if (!hasHist) src.textContent += " · 暂无评估数据（等待首次评估…）";
      else if (rsN) src.textContent += " · 竖线误差棒 = 评估噪声 1σ（轮内聚合 SE≈347.5/√N）";
      renderRunMeta();
    }
  }catch(e){
    document.getElementById("status").textContent = "连接失败：" + e;
  }
  render();
  renderSweep();
  renderSolo();
  refreshPlay();
}

// 长跑进度条：来自 run_state.json（当前 step）+ config.json（两级评估计划）。
// 关键点：league_state.total_steps 只是"最近评估点"，与"总共要跑到哪"无关 ⇒ 必须显示
// cur/total，否则 1M 步跑到 16k 时看起来像已经跑完。
function renderRunMeta(){
  const el = document.getElementById("leagueProgress");
  const sub = document.getElementById("leagueSub");
  if (!el) return;
  const rm = payload.run_meta;
  if (!payload.ok || !rm || !rm.total_steps){
    el.innerHTML = ""; if (sub) sub.textContent = ""; return;
  }
  const pct = rm.pct == null ? 0 : rm.pct;
  const h = (m) => m == null ? "—" : (m >= 60 ? (m/60).toFixed(1) + " h" : m.toFixed(1) + " min");
  const stale = rm.stale;
  const ageTxt = rm.state_age_s == null ? "" :
    (rm.state_age_s < 90 ? `${rm.state_age_s.toFixed(0)} s 前写入`
                         : `${(rm.state_age_s/60).toFixed(1)} min 前写入`);
  const barColor = stale ? "#ef4444" : (pct >= 99.5 ? "#22c55e" : "#3b82f6");
  el.innerHTML =
    `<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;font-size:12px">` +
    `<b>${rm.cur_step.toLocaleString()}</b> / ${rm.total_steps.toLocaleString()} 步` +
    `<span style="color:#94a3b8">(${pct}%)</span>` +
    `<span style="color:#94a3b8">· 评估点 ${rm.points_done}/${rm.plan_points}` +
    `（大 ${rm.plan_big} / 小 ${rm.plan_small}）</span>` +
    `<span style="color:#94a3b8">· 已跑 ${h(rm.elapsed_min)} · 粗估剩余 ${h(rm.eta_min)}</span>` +
    `<span style="color:${stale ? "#ef4444" : "#94a3b8"}">· ${ageTxt}` +
    `${stale ? " ⚠ 超过 15 min 未写入（疑似卡死/退出）" : ""}</span>` +
    `</div>` +
    `<div style="height:6px;background:#1e293b;border-radius:4px;overflow:hidden;margin-top:6px">` +
    `<div style="height:100%;width:${Math.min(100,pct)}%;background:${barColor}"></div></div>`;
  if (sub) sub.textContent =
    `两级评估：小点每 ${rm.steps_per_eval} 步 × ${rm.n_eval_games} 局/对` +
    (rm.big_eval_every ? `，大点每 ${rm.big_eval_every} 步 × ${rm.n_eval_games_big} 局/对`
                       : "（未开大评估）");
}

function renderLegend(){
  const el = document.getElementById("legend");
  el.innerHTML = (payload.agents || []).map(a =>
    `<span><span class="dot" style="background:${colorOf(a.id)}"></span>${a.label}</span>`
  ).join("");
}

function renderTable(){
  const tbody = document.getElementById("tbody");
  const rs = payload.round_stats || [];
  const rPrev = rs.length >= 2 ? rs[rs.length-2] : null;
  const rCur = rs.length >= 1 ? rs[rs.length-1] : null;
  const seOf = (rt, aid) => rt && rt.est && rt.est[aid] ? rt.est[aid][1] : null;
  const rows = (payload.agents || []).map(a => {
    const hist = (payload.elo_history[a.id] || []);
    let delta = "—", sigma = "—";
    if (hist.length >= 2) {
      const d = hist[hist.length-1][1] - hist[hist.length-2][1];
      delta = (d >= 0 ? "+" : "") + d.toFixed(1);
      const se0 = seOf(rPrev, a.id), se1 = seOf(rCur, a.id);
      if (se0 && se1){
        const comb = Math.hypot(se0, se1);
        const z = Math.abs(d) / comb;
        sigma = `<span style="color:${z >= 2 ? "#22c55e" : (z >= 1 ? "#f59e0b" : "#ef4444")}">${z.toFixed(1)}σ${z < 2 ? " 噪声" : ""}</span>`;
      }
    }
    const kindCls = a.kind === "main" ? "main" : (a.kind === "historical" ? "historical"
                   : (a.kind === "exploiter" ? "exploiter" : "baseline"));
    // vs 对手胜率（EMA）：league_state.winrates 的键是 "a|b" = a 对 b 的胜率。
    // main 行给"对手均值 + 逐对手明细"，其余行给自己的胜率（= 1 − main 对自己的胜率）。
    const wr = payload.winrates || {};
    let wrCell = "—";
    if (a.id === "main"){
      const vals = (payload.agents || []).filter(x => x.id !== "main")
        .map(x => wr["main|" + x.id]).filter(v => typeof v === "number");
      if (vals.length){
        const mean = vals.reduce((s,v)=>s+v,0) / vals.length;
        const detail = (payload.agents || []).filter(x => x.id !== "main")
          .map(x => `${x.id}=${typeof wr["main|"+x.id] === "number" ? wr["main|"+x.id].toFixed(2) : "—"}`)
          .join(" ");
        wrCell = `<span title="${detail}">${(100*mean).toFixed(1)}%</span>`;
      }
    } else {
      const v = wr["main|" + a.id];
      if (typeof v === "number") wrCell = `${(100*(1-v)).toFixed(1)}%`;
    }
    return `<tr>
      <td><span class="dot" style="background:${colorOf(a.id)}"></span>${a.label}</td>
      <td><span class="kind ${kindCls}">${a.kind}</span></td>
      <td><b>${a.elo.toFixed(1)}</b></td>
      <td>${delta}</td>
      <td>${sigma}</td>
      <td>${wrCell}</td>
      <td style="color:#94a3b8;font-size:11px">${a.path ? a.path.split(/[\\/]/).pop() : "—"}</td>
    </tr>`;
  }).join("");
  tbody.innerHTML = rows || `<tr><td colspan="7">暂无模型（等待训练写入状态文件…）</td></tr>`;
  document.getElementById("meta").textContent =
    `总训练步数：${payload.total_steps} · 模型数：${payload.agents.length}` +
    (rCur ? ` · 最近评估 ${rCur.step} 步` + (rCur.kind ? `（${rCur.kind === "big" ? "大" : "小"}评估` +
      (rCur.games_per_pair ? ` ${rCur.games_per_pair} 局/对` : "") + `）` : "") : "") +
    (payload.run_meta && payload.run_meta.total_steps ?
      ` · 计划跑到 ${payload.run_meta.total_steps}` : "");
}

function drawChart(){
  const sel = document.getElementById("leagueMetric");
  const mode = sel ? sel.value : "elo";
  if (mode === "winrate") return drawWinrateChart();
  return drawEloChart();
}

// 大评估点（两级评估）的竖虚线：横轴整个画布只画一次，不随曲线条数重复。
function drawBigPointMarkers(ctx, bigSteps, X, padT, H, padB){
  if (!bigSteps.length) return;
  ctx.save();
  ctx.strokeStyle = "#64748b"; ctx.globalAlpha = 0.5;
  ctx.setLineDash([2,4]); ctx.lineWidth = 1;
  bigSteps.forEach(s => { ctx.beginPath(); ctx.moveTo(X(s), padT); ctx.lineTo(X(s), H-padB); ctx.stroke(); });
  ctx.restore();
}

function drawEloChart(){
  const canvas = document.getElementById("chart");
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth, H = canvas.clientHeight;
  canvas.width = W*dpr; canvas.height = H*dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0,0,W,H);

  const padL=52, padR=16, padT=16, padB=40;
  const plotW=W-padL-padR, plotH=H-padT-padB;
  const hist = Object.values(payload.elo_history);
  let allPts = [];
  hist.forEach(h => allPts = allPts.concat(h));
  const xs = allPts.map(p=>p[0]);
  let ys = allPts.map(p=>p[1]);
  if (payload.agents.length) ys = ys.concat(payload.agents.map(a=>a.elo));
  // 横轴优先用 run_state 的**总计划步数**（1M 长跑）：否则曲线会被最近几个评估点挤满，
  // 看不见"才跑了 1.6%"。没有 run_meta 时退回旧行为（最近评估点 / 当前值）。
  const rm = payload.run_meta || {};
  const maxStep = Math.max(100, ...xs, payload.total_steps, rm.total_steps || 0);
  let minElo = Math.min(1400, ...ys), maxElo = Math.max(1600, ...ys);
  const span = Math.max(50, maxElo - minElo);
  minElo = minElo - span*0.08; maxElo = maxElo + span*0.08;

  const X = s => padL + (s / maxStep) * plotW;
  const Y = e => padT + (1 - (e - minElo) / (maxElo - minElo)) * plotH;

  // grid + axes
  ctx.strokeStyle="#334155"; ctx.fillStyle="#94a3b8";
  ctx.font="11px sans-serif"; ctx.lineWidth=1;
  ctx.beginPath();
  for (let i=0;i<=4;i++){
    const v = minElo + (maxElo-minElo)*i/4, y = Y(v);
    ctx.moveTo(padL, y); ctx.lineTo(W-padR, y);
    ctx.fillText(v.toFixed(0), 4, y+4);
  }
  ctx.stroke();
  ctx.beginPath();
  for (let i=0;i<=5;i++){
    const v = Math.round(maxStep*i/5), x = X(v);
    ctx.moveTo(x, padT); ctx.lineTo(x, H-padB);
    ctx.fillText(v >= 10000 ? (v/1000).toFixed(0)+"k" : String(v), x-12, H-padB+16);
  }
  ctx.stroke();
  // 1500 基准线
  ctx.strokeStyle="#475569"; ctx.setLineDash([4,4]); ctx.beginPath();
  ctx.moveTo(padL, Y(1500)); ctx.lineTo(W-padR, Y(1500));
  ctx.stroke(); ctx.setLineDash([]);
  ctx.fillText("1500", W-padR-30, Y(1500)-4);
  // 大评估点标记
  const bigSteps = (payload.round_stats || []).filter(rt => rt.kind === "big").map(rt => rt.step);
  drawBigPointMarkers(ctx, bigSteps, X, padT, H, padB);

  // 每条模型曲线
  const series = payload.agents.map(a => ({
    id: a.id, color: colorOf(a.id), pts: payload.elo_history[a.id] || [],
    cur: a.elo
  }));
  series.forEach(s => {
    ctx.strokeStyle = s.color; ctx.lineWidth = 2; ctx.beginPath();
    let first = true;
    s.pts.forEach(p => { const x=X(p[0]), y=Y(p[1]); first ? ctx.moveTo(x,y) : ctx.lineTo(x,y); first=false; });
    if (s.pts.length === 1) { const x=X(s.pts[0][0]), y=Y(s.pts[0][1]); ctx.moveTo(x,y); ctx.lineTo(x+0.01,y); }
    ctx.stroke();
    // 端点（最新）
    if (s.pts.length){
      const p = s.pts[s.pts.length-1];
      ctx.fillStyle=s.color; ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 3.5, 0, 7); ctx.fill();
    } else {
      // 只有一个当前值：画在 maxStep
      ctx.fillStyle=s.color; ctx.beginPath(); ctx.arc(X(maxStep), Y(s.cur), 3.5, 0, 7); ctx.fill();
    }
    // 轮内聚合估计：竖线误差棒 = 评估噪声 1σ（SE≈347.5/√N，随局数收窄）
    (payload.round_stats || []).forEach(rt => {
      const est = rt.est && rt.est[s.id];
      if (!est) return;
      const x = X(rt.step), y = Y(est[0]), se = est[1];
      if (!(se > 0)) return;
      ctx.globalAlpha = 0.9;
      ctx.strokeStyle = s.color; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(x, Y(est[0]+se)); ctx.lineTo(x, Y(est[0]-se)); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x-4, Y(est[0]+se)); ctx.lineTo(x+4, Y(est[0]+se)); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x-4, Y(est[0]-se)); ctx.lineTo(x+4, Y(est[0]-se)); ctx.stroke();
      ctx.fillStyle = s.color; ctx.beginPath(); ctx.arc(x, y, 2.5, 0, 7); ctx.fill();
      ctx.globalAlpha = 1;
    });
  });

  // 悬停 tooltip
  canvas.onmousemove = e => {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX-rect.left, my = e.clientY-rect.top;
    let best=null, bestD=1e9;
    series.forEach(s=>{
      s.pts.forEach(p=>{
        const d=(X(p[0])-mx)**2 + (Y(p[1])-my)**2;
        if (d<bestD){ bestD=d; best={id:s.id, step:p[0], elo:p[1]}; }
      });
    });
    const tip=document.getElementById("tooltip");
    if (best && bestD < 900){
      tip.style.display="block";
      tip.style.left=(mx+12)+"px"; tip.style.top=(my+10)+"px";
      tip.innerHTML=`<b style="color:${colorOf(best.id)}">${LABELS[best.id] || best.id}</b><br>步数 ${best.step}<br>Elo ${best.elo.toFixed(1)}`;
    } else tip.style.display="none";
  };
  canvas.onmouseleave=()=>{ document.getElementById("tooltip").style.display="none"; };
}

// 对每个对手的胜率曲线（run 模式专用）：对手是**固定脚本**、不学习 ⇒ 这是非自引用读数，
// 与 solo 的"打冻结副本"完全不同（后者结构性≈0.5，禁读）。数据来自
// payload.winrate_curves（父进程按累计局数从 league_state.history 切分复原的逐点胜率）。
function drawWinrateChart(){
  const canvas = document.getElementById("chart");
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth, H = canvas.clientHeight;
  canvas.width = W*dpr; canvas.height = H*dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0,0,W,H);

  const padL=52, padR=16, padT=16, padB=40;
  const plotW=W-padL-padR, plotH=H-padT-padB;
  const rm = payload.run_meta || {};
  const curves = payload.winrate_curves || {}, counts = payload.winrate_counts || {};
  // 只画"learner vs 对手"的 pair：优先 main|*，否则用 agents 里第一个 id 当 learner
  let keys = Object.keys(curves).filter(k => k.startsWith("main|"));
  if (!keys.length && payload.agents.length){
    const lid = payload.agents[0].id;
    keys = Object.keys(curves).filter(k => k.startsWith(lid + "|"));
  }
  keys.sort();
  let xs = [];
  keys.forEach(k => (curves[k] || []).forEach(p => xs.push(p[0])));
  const maxStep = Math.max(100, ...xs, rm.total_steps || 0, payload.total_steps);
  const X = s => padL + (s / maxStep) * plotW;
  const Y = v => padT + (1 - v) * plotH;

  ctx.strokeStyle="#334155"; ctx.fillStyle="#94a3b8";
  ctx.font="11px sans-serif"; ctx.lineWidth=1;
  ctx.beginPath();
  [0,0.25,0.5,0.75,1].forEach(v => {
    const y = Y(v); ctx.moveTo(padL, y); ctx.lineTo(W-padR, y);
    ctx.fillText(v.toFixed(2), 4, y+4);
  });
  ctx.stroke();
  ctx.beginPath();
  for (let i=0;i<=5;i++){
    const v = Math.round(maxStep*i/5), x = X(v);
    ctx.moveTo(x, padT); ctx.lineTo(x, H-padB);
    ctx.fillText(v >= 10000 ? (v/1000).toFixed(0)+"k" : String(v), x-12, H-padB+16);
  }
  ctx.stroke();
  ctx.strokeStyle="#475569"; ctx.setLineDash([4,4]); ctx.beginPath();
  ctx.moveTo(padL, Y(0.5)); ctx.lineTo(W-padR, Y(0.5));
  ctx.stroke(); ctx.setLineDash([]);
  ctx.fillText("0.50", W-padR-34, Y(0.5)-4);
  const bigSteps = (payload.round_stats || []).filter(rt => rt.kind === "big").map(rt => rt.step);
  drawBigPointMarkers(ctx, bigSteps, X, padT, H, padB);

  const series = keys.map(k => {
    const opp = k.split("|")[1];
    return {id: k, opp: opp, color: colorOf(opp),
            pts: (curves[k] || []), ns: (counts[k] || [])};
  });
  series.forEach(s => {
    ctx.strokeStyle = s.color; ctx.lineWidth = 2; ctx.beginPath();
    let first = true;
    s.pts.forEach(p => { const x=X(p[0]), y=Y(p[1]); first ? ctx.moveTo(x,y) : ctx.lineTo(x,y); first=false; });
    ctx.stroke();
    s.pts.forEach((p,i) => {
      const n = s.ns[i] || 0, se = n > 1 ? Math.sqrt(Math.max(0,p[1]*(1-p[1]))/n) : 0;
      const x = X(p[0]);
      if (se > 0){
        ctx.globalAlpha = 0.85; ctx.strokeStyle = s.color; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(x, Y(Math.min(1,p[1]+se))); ctx.lineTo(x, Y(Math.max(0,p[1]-se))); ctx.stroke();
        ctx.globalAlpha = 1;
      }
      ctx.fillStyle = s.color; ctx.beginPath(); ctx.arc(x, Y(p[1]), 2.5, 0, 7); ctx.fill();
    });
  });

  canvas.onmousemove = e => {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX-rect.left, my = e.clientY-rect.top;
    let best=null, bestD=1e9;
    series.forEach(s => s.pts.forEach((p,i) => {
      const d=(X(p[0])-mx)**2 + (Y(p[1])-my)**2;
      if (d<bestD){ bestD=d; best={s:s, step:p[0], wr:p[1], n:s.ns[i]||0}; }
    }));
    const tip=document.getElementById("tooltip");
    if (best && bestD < 900){
      const n = best.n, se = n > 1 ? Math.sqrt(Math.max(0,best.wr*(1-best.wr))/n) : 0;
      tip.style.display="block";
      tip.style.left=(mx+12)+"px"; tip.style.top=(my+10)+"px";
      tip.innerHTML=`<b style="color:${best.s.color}">main vs ${LABELS[best.s.opp] || best.s.opp}</b>` +
        `<br>步数 ${best.step}<br>胜率 ${(100*best.wr).toFixed(1)}% ± ${(100*se).toFixed(1)}%` +
        `<br>${n} 局`;
    } else tip.style.display="none";
  };
  canvas.onmouseleave=()=>{ document.getElementById("tooltip").style.display="none"; };
}

function render(){
  if (!payload.ok){ renderLegend(); renderTable(); return; }
  renderLegend(); renderTable(); drawChart();
}

/* ================= flow-sweep：训练进度 + main 曲线 ================= */

function renderSweep(){
  const card = document.getElementById("sweepCard");
  const body = document.getElementById("sweepBody");
  if (!sweep.ok || !(sweep.strategies || []).length){
    card.style.display = "none";
    return;
  }
  card.style.display = "";
  document.getElementById("sweepSub").textContent =
    "根目录 " + sweep.sweep_root + " · 每 3s 刷新";
  body.innerHTML = sweep.strategies.map(s => {
    const done = s.status === "done";
    const pct = done ? 100 : Math.round(100 * s.run_current / Math.max(1, s.n_runs));
    const t = s.trend || {};
    const tv = t.verdict || "";
    const vCls = tv.indexOf("上涨") === 0 ? "up" : (tv.indexOf("下跌") === 0 ? "down" : "flat");
    const rows = s.rows || [];
    const last = rows[rows.length - 1];
    const sname = s.strategy === "stream" ? "stream · 每对 1 局 × 20 次"
               : (s.strategy === "games5" ? "games5 · 每对 5 局 × 4 次"
               : (s.strategy || "?"));
    const perModel = last ? Object.keys(last.est || {}).map(mid => {
      const e = last.est[mid];
      return `<span><span class="dot" style="background:${colorOf(mid)}"></span>${LABELS[mid] || mid} ${e[0].toFixed(0)}±${e[1].toFixed(0)}</span>`;
    }).join("") : "";
    const tableRows = rows.slice(-14).reverse().map(r =>
      `<tr><td>#${r.run + 1}</td><td>${r.games.toLocaleString()}</td>
       <td><b>${r.main_est[0].toFixed(0)}</b> ±${r.main_est[1].toFixed(0)}</td></tr>`).join("");
    const cid = "sweepChart_" + (s.strategy || "s");
    const eta = (s.status === "running" && s.eta_s > 0)
      ? " · 预计剩余 " + (s.eta_s >= 60 ? (s.eta_s / 60).toFixed(1) + " 分钟" : s.eta_s.toFixed(0) + " 秒") : "";
    return `<div class="strategy-card">
      <div class="strategy-head">
        <h3>${sname}</h3>
        <span class="chip ${done ? "done" : "running"}">${done ? "已完成" : "训练中"}</span>
        <span class="chip ${vCls}">${tv || "—"}</span>
        <div class="progress">
          <div class="bar"><i style="width:${pct}%"></i></div>
          <span class="pct">${s.run_current}/${s.n_runs} 轮 · ${pct}%</span>
        </div>
        <span class="sub">${s.desc || ""}${eta}</span>
      </div>
      <div class="sweep-layout">
        <div style="position:relative"><canvas id="${cid}"></canvas></div>
        <div>
          <p class="sub" style="margin:0 0 8px">最新一轮（#${s.run_current}）各模型轮内估计 R±SE：</p>
          <div class="legend" style="margin:0 0 10px">${perModel || "—"}</div>
          <p class="sub" style="margin:0 0 6px">main 逐轮估计（最近 ${Math.min(14, rows.length)} 轮，最新在上）：</p>
          <table>
            <thead><tr><th>轮次</th><th>对局数</th><th>main R±SE</th></tr></thead>
            <tbody>${tableRows || '<tr><td colspan="3">等待第一轮完成…</td></tr>'}</tbody>
          </table>
          <p class="sub" style="margin:8px 0 0">总预算 ${(s.total_games || 0).toLocaleString()} 局 · 每轮 ${(s.per_run_games || 0).toLocaleString()} 局 · 评估 ${s.eval_games} 局/对 · 误差棒 = 评估噪声 1σ（SE≈347.5/√N）</p>
        </div>
      </div>
    </div>`;
  }).join("");
  sweep.strategies.forEach(s => drawSweepChart(s, "sweepChart_" + (s.strategy || "s")));
}

function drawSweepChart(s, canvasId){
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth || 600, H = canvas.clientHeight || 260;
  canvas.width = W * dpr; canvas.height = H * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, W, H);
  const padL = 50, padR = 16, padT = 18, padB = 34;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const rows = s.rows || [];
  ctx.fillStyle = "#94a3b8"; ctx.font = "12px sans-serif";
  if (!rows.length){ ctx.fillText("等待第一轮完成…", padL, padT + 20); return; }
  const maxRun = Math.max(1, s.n_runs - 1);
  let lo = 1e9, hi = -1e9;
  rows.forEach(r => { const v = r.main_est[0], se = r.main_est[1] || 0; lo = Math.min(lo, v - se); hi = Math.max(hi, v + se); });
  let span = Math.max(60, hi - lo); lo -= span * 0.15; hi += span * 0.15;
  const X = i => padL + (i / maxRun) * plotW;
  const Y = v => padT + (1 - (v - lo) / (hi - lo)) * plotH;
  // 网格
  ctx.strokeStyle = "#334155"; ctx.fillStyle = "#94a3b8"; ctx.font = "11px sans-serif"; ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i <= 4; i++){ const v = lo + (hi - lo) * i / 4, y = Y(v); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.fillText(v.toFixed(0), 4, y + 4); }
  ctx.stroke();
  ctx.beginPath();
  for (let i = 0; i <= 4; i++){ const r = Math.round(maxRun * i / 4), x = X(r); ctx.moveTo(x, padT); ctx.lineTo(x, H - padB); ctx.fillText(String(r + 1), x - 10, H - padB + 16); }
  ctx.stroke();
  // 1500 基准虚线
  if (lo < 1500 && 1500 < hi){
    ctx.strokeStyle = "#475569"; ctx.setLineDash([4, 4]); ctx.beginPath();
    ctx.moveTo(padL, Y(1500)); ctx.lineTo(W - padR, Y(1500)); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillText("1500", W - padR - 30, Y(1500) - 4);
  }
  // main 估计曲线 + 竖线误差棒（1σ 噪声）
  const color = "#2563eb";
  ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
  rows.forEach((r, i) => { const x = X(r.run), y = Y(r.main_est[0]); i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); });
  ctx.stroke();
  rows.forEach(r => {
    const x = X(r.run), y = Y(r.main_est[0]), se = r.main_est[1] || 0;
    ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.globalAlpha = 0.9;
    ctx.beginPath(); ctx.moveTo(x, Y(r.main_est[0] + se)); ctx.lineTo(x, Y(r.main_est[0] - se)); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x - 4, Y(r.main_est[0] + se)); ctx.lineTo(x + 4, Y(r.main_est[0] + se)); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x - 4, Y(r.main_est[0] - se)); ctx.lineTo(x + 4, Y(r.main_est[0] - se)); ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x, y, 3, 0, 7); ctx.fill();
  });
  ctx.fillStyle = "#e2e8f0"; ctx.font = "12px sans-serif";
  ctx.fillText("main 轮内估计 R±1σ（x 轴 = 训练轮次）", padL + 6, 12);
}

/* ================= solo 自对弈：胜率曲线 + 训练进度 ================= */

function renderSolo(){
  const card = document.getElementById("soloCard");
  if (!solo.ok){
    card.style.display = "none";
    return;
  }
  card.style.display = "";
  const done = solo.status === "done";
  document.getElementById("soloSub").textContent =
    "状态 " + solo.solo_path + " · 每 3s 刷新";
  document.getElementById("soloStatus").textContent = done ? "已完成" : "训练中";
  document.getElementById("soloStatus").className = "chip " + (done ? "done" : "running");
  const pct = solo.target_steps > 0 ? Math.round(100 * solo.total_steps / solo.target_steps) : 0;
  document.getElementById("soloProgress").innerHTML =
    `<div class="bar"><i style="width:${pct}%"></i></div>` +
    `<span class="pct">${solo.total_steps.toLocaleString()} / ${solo.target_steps.toLocaleString()} 步 · ${pct}%</span>`;
  document.getElementById("soloMeta").textContent =
    "对手=冻结副本（每 " + solo.copy_every + " 步同步）· " + solo.opponent + " · 评估 " +
    (solo.history.length ? solo.history[solo.history.length - 1].games : "?") + " 局/次";
  document.getElementById("soloDeck").textContent = (solo.deck || []).join(" · ");
  const rows = (solo.history || []).slice(-14).reverse();
  document.getElementById("soloTable").innerHTML = rows.map(h =>
    `<tr><td>${h.step.toLocaleString()}</td><td>${h.wins}W / ${h.losses}L / ${h.draws}D</td>` +
    `<td><b>${(h.winrate * 100).toFixed(1)}%</b> ±${(h.winrate_se * 100).toFixed(1)}</td>` +
    `<td>${h.mean_reward.toFixed(3)}</td>` +
    `<td>${h.engagement_rate === undefined ? "–" : h.engagement_rate.toFixed(0) + "%"}</td>` +
    `<td>${h.unilateral_rate === undefined ? "–" : h.unilateral_rate.toFixed(0) + "%"}</td>` +
    `<td>${h.defense_invest_rate === undefined ? "–" : h.defense_invest_rate.toFixed(0) + "%"}</td>` +
    `<td>${h.response_latency_med === undefined || h.response_latency_med === null ? "–" : h.response_latency_med.toFixed(1) + "s"}</td>` +
    `<td>${h.deploy_per_game === undefined ? "–" : h.deploy_per_game.toFixed(1)}</td>` +
    `<td>${h.elixir_avg === undefined ? "–" : h.elixir_avg.toFixed(1)}</td>` +
    (() => {
      // EV：≥0.2 绿 / 0~0.2 黄 / <0 红
      const ev = h.explained_variance;
      const evTd = (ev === undefined || ev === null) ? "<td>–</td>"
        : `<td style="color:${ev >= 0.2 ? "#4ade80" : (ev >= 0 ? "#fbbf24" : "#f87171")};font-weight:600">${ev.toFixed(3)}</td>`;
      // GRU 活力（v3）：h 跨帧 std 健康 >0.05、GRU n(abs) 健康 <0.9
      const hs = h.h_std;
      const hsTd = (hs === undefined || hs === null) ? "<td>–</td>"
        : `<td style="color:${hs > 0.05 ? "#4ade80" : (hs > 0.02 ? "#fbbf24" : "#f87171")};font-weight:600">${hs.toExponential(1)}</td>`;
      const na = h.gru_n_abs;
      const naTd = (na === undefined || na === null) ? "<td>–</td>"
        : `<td style="color:${na < 0.9 ? "#4ade80" : (na < 0.97 ? "#fbbf24" : "#f87171")};font-weight:600">${na.toFixed(3)}</td>`;
      // v3 §2 第 3 行：value_head 输出 std / 批内 R std（>0.3 绿 / >0.1 黄 / 否则红）
      const vr = h.value_std_ratio;
      const vrTd = (vr === undefined || vr === null) ? "<td>–</td>"
        : `<td style="color:${vr > 0.3 ? "#4ade80" : (vr > 0.1 ? "#fbbf24" : "#f87171")};font-weight:600">${vr.toFixed(3)}</td>`;
      return evTd + hsTd + naTd + vrTd + "</tr>";
    })()).join("") ||
    '<tr><td colspan="14">等待首次评估…</td></tr>';
  drawSoloMetricChart();
}

/* --- 指标注册表：2026-09-17 起 **不再以胜率为曲线指标** ---
   AGENTS §3 判读禁则：「不判 main 曲线与单点胜率」——main vs 刚同步的冻结副本是自引用读数
   （结构性 ≈0.5）⇒ 默认画**行为指标**，胜率降级进"⚠ 自引用（禁读）"分组。
   数据源：solo_state.json 的 history[]（逐评估点 32 键）与 _controls_history（三路外生对照）。 */
const SOLO_METRICS = {
  behavior: {
    note: "行为指标（相位型，只读相对退化）：接敌率 = 防守部署 8s 内 5 格内敌我同框；单边堆牌 = 只攻不防；平均圣水低 = 不会攒费（O3 病理）。",
    series: [
      {key: "engagement_rate", label: "接敌率", color: "#22c55e", pct: true},
      {key: "unilateral_rate", label: "单边堆牌", color: "#ef4444", pct: true},
      {key: "elixir_avg", label: "平均圣水", color: "#38bdf8", pct: false},
    ],
  },
  defense: {
    note: "防守链：防守投入 = 敌过河帧中我方有部署的比例；响应延迟 = 威胁→首次响应中位秒数（越低越主动）；拦截率 = 落点在敌→我塔直线 4 格内。",
    series: [
      {key: "defense_invest_rate", label: "防守投入", color: "#22c55e", pct: true},
      {key: "intercept_rate", label: "拦截率", color: "#a855f7", pct: true},
      {key: "response_latency_med", label: "响应延迟(s)", color: "#fbbf24", pct: false},
    ],
  },
  deploy: {
    note: "部署量与场面：deploy/局 = 每局平均部署次数（⚠️ 早停关闭后局变长，绝对数不跨 run 比）；多卡同帧 = 同帧 ≥2 张的比例；塔血差 = 平均塔血差（正 = 我方领先）。",
    series: [
      {key: "deploy_per_game", label: "deploy/局", color: "#60a5fa", pct: false},
      {key: "bundle_multi_rate", label: "多卡同帧", color: "#f472b6", pct: true},
      {key: "tower_diff_avg", label: "塔血差", color: "#4ade80", pct: false},
    ],
  },
  reward: {
    note: "平均回报（逐点）。⚠️ 早停关闭后训练侧均局 ≈345 帧 ⇒ 终局 ±10 的可见度 (γλ)^k ≈ 0.011（旧早停时代 ~100 帧 ≈ 0.27），见 docs/nostall20k_verdict_2026-09-17.md §4。",
    series: [{key: "mean_reward", label: "mean_reward", color: "#f59e0b", pct: false}],
  },
  critic: {
    note: "critic 健康度：EV ≥0.2 才算价值头在有效学习（≈0 = 只会预测均值，<0 = 比常数还差）；value÷R std >0.3 才算输出波动与回报同量级（C3/O2）。",
    series: [
      {key: "explained_variance", label: "critic EV", color: "#22c55e", pct: false},
      {key: "value_std_ratio", label: "value/R std", color: "#38bdf8", pct: false},
    ],
  },
  vitality: {
    note: "GRU 活力：h 跨帧 std 健康 >0.05（≤0.02 = 隐状态冻成常数）；n(abs) 健康 <0.9（→1.0 = tanh 候选饱和）。",
    series: [
      {key: "h_std", label: "h 跨帧 std", color: "#22c55e", pct: false},
      {key: "gru_n_abs", label: "GRU n(abs)", color: "#f87171", pct: false},
    ],
  },
  controls: {
    note: "外生对照（**目前唯一能测绝对强度的仪器**）：baseline0 = 训练起点、baseline_prev = 上一评估点、baseline_rand = 固定种子的随机策略。三路各 40 局且每点重抽 ⇒ 1σ≈0.078。",
    fromControls: true,
    series: [
      {key: "baseline0", label: "vs 起点", color: "#64748b", pct: true},
      {key: "baseline_prev", label: "vs 上一评估点", color: "#a855f7", pct: true},
      {key: "baseline_rand", label: "vs 固定随机锚点", color: "#f59e0b", pct: true},
    ],
  },
  winrate: {
    note: "⚠️ 自引用读数：main vs **刚同步的冻结副本**（结构性 ≈0.5）。AGENTS §3 判读禁则明确：这类的 0.85 **不得当作「变强」证据**，仅用于排查。",
    series: [{key: "winrate", label: "胜率（自引用）", color: "#2563eb", pct: true, se: "winrate_se"}],
  },
};
let curSoloMetric = "behavior";

function _fmtVal(v){
  const a = Math.abs(v);
  if (a >= 10000) return (v / 1000).toFixed(1) + "k";
  if (a >= 100) return v.toFixed(0);
  if (a >= 1) return v.toFixed(2);
  if (a === 0) return "0";
  return v.toFixed(3);
}

function _allPct(lines){ return lines.length > 0 && lines.every(l => l.pct); }

function drawSoloMetricChart(){
  const canvas = document.getElementById("soloChart");
  if (!canvas) return;
  const spec = SOLO_METRICS[curSoloMetric] || SOLO_METRICS.behavior;
  const noteEl = document.getElementById("soloMetricNote");
  if (noteEl) noteEl.textContent = spec.note || "";
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth || 600, H = canvas.clientHeight || 300;
  canvas.width = W * dpr; canvas.height = H * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, W, H);
  const padL = 54, padR = 16, padT = 30, padB = 34;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const hist = solo.history || [];
  const ctrl = solo.controls_history || [];
  const legendEl = document.getElementById("soloMetricLegend");
  ctx.fillStyle = "#94a3b8"; ctx.font = "12px sans-serif";
  if (!(spec.fromControls ? ctrl.length : hist.length)){
    ctx.fillText("等待首次评估…", padL, padT + 20);
    return;
  }
  const lines = spec.series.map(s => {
    let pts;
    if (spec.fromControls){
      pts = ctrl.filter(r => r.vs === s.key && r.winrate !== undefined && r.winrate !== null)
                .map(r => [r.step, r.winrate, r.winrate_se || 0]);
    } else {
      pts = hist.filter(r => r[s.key] !== undefined && r[s.key] !== null)
                .map(r => [r.step, r[s.key], s.se ? (r[s.se] || 0) : null]);
    }
    return {label: s.label, color: s.color, pct: !!s.pct, pts};
  }).filter(l => l.pts.length);
  if (!lines.length){
    ctx.fillText("该指标尚无数据（需要新的评估点）", padL, padT + 20);
    if (legendEl) legendEl.innerHTML = "";
    return;
  }
  const allV = [];
  lines.forEach(l => l.pts.forEach(p => {
    allV.push(p[1]);
    if (p[2]){ allV.push(p[1] + p[2]); allV.push(p[1] - p[2]); }
  }));
  let lo = Math.min.apply(null, allV), hi = Math.max.apply(null, allV);
  if (hi - lo < 1e-9){ const m = Math.max(1e-6, Math.abs(hi)); lo -= m * 0.5; hi += m * 0.5; }
  const span = hi - lo; lo -= span * 0.15; hi += span * 0.15;
  let maxStep = 1;
  lines.forEach(l => l.pts.forEach(p => { maxStep = Math.max(maxStep, p[0]); }));
  if (solo.target_steps) maxStep = Math.max(maxStep, solo.target_steps);
  const X = s => padL + (s / maxStep) * plotW;
  const Y = v => padT + (1 - (v - lo) / (hi - lo)) * plotH;
  const pctAxis = _allPct(lines);
  ctx.strokeStyle = "#334155"; ctx.fillStyle = "#94a3b8"; ctx.font = "11px sans-serif"; ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i <= 4; i++){
    const v = lo + (hi - lo) * i / 4, y = Y(v);
    ctx.moveTo(padL, y); ctx.lineTo(W - padR, y);
    ctx.fillText(_fmtVal(v) + (pctAxis ? "%" : ""), 4, y + 4);
  }
  ctx.stroke();
  ctx.beginPath();
  for (let i = 0; i <= 4; i++){
    const s = Math.round(maxStep * i / 4), x = X(s);
    ctx.moveTo(x, padT); ctx.lineTo(x, H - padB);
    ctx.fillText(s.toLocaleString(), x - 12, H - padB + 16);
  }
  ctx.stroke();
  if (pctAxis && lo < 0.5 && 0.5 < hi){
    ctx.strokeStyle = "#475569"; ctx.setLineDash([4, 4]); ctx.beginPath();
    ctx.moveTo(padL, Y(0.5)); ctx.lineTo(W - padR, Y(0.5)); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillText("50%", W - padR - 34, Y(0.5) - 4);
  }
  lines.forEach(l => {
    ctx.strokeStyle = l.color; ctx.lineWidth = 2; ctx.beginPath();
    l.pts.forEach((p, i) => { const x = X(p[0]), y = Y(p[1]); i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); });
    ctx.stroke();
    l.pts.forEach(p => {
      const x = X(p[0]), y = Y(p[1]);
      if (p[2]){
        ctx.strokeStyle = l.color; ctx.lineWidth = 1.5; ctx.globalAlpha = 0.9;
        ctx.beginPath(); ctx.moveTo(x, Y(p[1] + p[2])); ctx.lineTo(x, Y(p[1] - p[2])); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(x - 4, Y(p[1] + p[2])); ctx.lineTo(x + 4, Y(p[1] + p[2])); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(x - 4, Y(p[1] - p[2])); ctx.lineTo(x + 4, Y(p[1] - p[2])); ctx.stroke();
        ctx.globalAlpha = 1;
      }
      ctx.fillStyle = l.color; ctx.beginPath(); ctx.arc(x, y, 3, 0, 7); ctx.fill();
    });
  });
  if (legendEl) legendEl.innerHTML = lines.map(l =>
    `<span><span class="dot" style="background:${l.color}"></span>${escHtml(l.label)}${l.pct ? "（%）" : ""}</span>`
  ).join("");
  const sel = document.getElementById("soloMetric");
  const titleTxt = (sel && sel.selectedOptions && sel.selectedOptions[0]) ? sel.selectedOptions[0].textContent : "";
  ctx.fillStyle = "#e2e8f0"; ctx.font = "12px sans-serif";
  ctx.fillText("solo · " + titleTxt, padL + 4, 15);
}

/* 指标下拉：切换即重画（不必等下一次 3s 轮询） */
document.getElementById("soloMetric").onchange = (e) => {
  curSoloMetric = e.target.value;
  drawSoloMetricChart();
};

/* ================= 人机对战：实时对局 + 数据采集 ================= */

let play = {ok:false, frame:null, hand:[], legal:{}, game_over:false, winner:null,
            reward_total:0, steps:0, max_steps:600, elixir:0};
let playSlot = null;
let playBusy = false;

async function refreshPlay(){
  try{
    const r = await fetch("/api/play/state?_t=" + Date.now(), {cache:"no-store"});
    const d = await r.json();
    if (d.ok){ play = d; renderPlay(); }
    else {
      play.ok = false;
      document.getElementById("playCard").style.display = "none";
    }
  }catch(e){ play.ok = false; }
}

function renderPlay(){
  const card = document.getElementById("playCard");
  if (!play.ok){ card.style.display = "none"; return; }
  card.style.display = "";
  drawInterpOn(document.getElementById("playArena"), play.frame, play.frame, 1);
  const w = play.game_over
    ? (play.winner === 0 ? "你赢了 🎉" : play.winner === 1 ? "你输了 😔" : "平局")
    : "对局中…";
  document.getElementById("playInfo").innerHTML =
    `<div><b>${w}</b> · 步数 ${play.steps}/${play.max_steps} · 累计奖励 ` +
    `<b style="color:${play.reward_total >= 0 ? "#22c55e" : "#ef4444"}">` +
    `${play.reward_total >= 0 ? "+" : ""}${play.reward_total}</b></div>` +
    `<div class="kv"><span>我方圣水</span><b>${Number(play.elixir).toFixed(1)}</b></div>` +
    `<div class="kv"><span>规则</span><b>先点手牌选卡，再点场地格子出牌；每步推进 0.5s 战斗</b></div>`;
  document.getElementById("playHand").innerHTML = play.hand.map((c, i) => {
    const legal = play.legal[i + 1];
    const disabled = !legal || play.game_over;
    return `<div class="hand-card ${playSlot === i + 1 ? "sel" : ""}" ` +
      `style="${disabled ? "opacity:.4;cursor:default" : ""}" ` +
      `onclick="${disabled ? "" : "selectPlaySlot(" + (i + 1) + ")"}">${c}</div>`;
  }).join("") || '<span class="sub">（无手牌）</span>';
  document.getElementById("playLegal").textContent = play.game_over
    ? "本局结束，点「结束本局并保存」后开新局"
    : (playSlot ? "已选 " + (play.hand[playSlot - 1] || "") + " · 点场地格子出牌"
                : "点击手牌选择要出的卡");
  document.getElementById("playSub").textContent = "实时采集（EpisodeReplay + BC 样本）";
}

function selectPlaySlot(s){
  if (play.game_over) return;
  playSlot = (playSlot === s) ? null : s;
  renderPlay();
}

document.getElementById("playArena").addEventListener("click", async (e) => {
  if (!play.ok || play.game_over || playBusy || !playSlot) return;
  const cv = e.currentTarget;
  const rect = cv.getBoundingClientRect();
  const scale = Math.min(cv.width / 18, cv.height / 32);
  const ox = (cv.width - 18 * scale) / 2, oy = (cv.height - 32 * scale) / 2;
  const mx = (e.clientX - rect.left) * (cv.width / rect.width);
  const my = (e.clientY - rect.top) * (cv.height / rect.height);
  const x = Math.floor((mx - ox) / scale), y = Math.floor((my - oy) / scale);
  if (x < 0 || x >= 18 || y < 0 || y >= 32) return;
  playBusy = true;
  try{
    const r = await fetch("/api/play/action?_t=" + Date.now(), {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({slot: playSlot, x: x, y: y})
    });
    const d = await r.json();
    if (!d.ok){ alert(d.error || "非法动作"); }
    else { play = d; playSlot = null; renderPlay(); }
  }catch(err){ alert("提交失败：" + err); }
  playBusy = false;
});

document.getElementById("btnPlayNew").onclick = async () => {
  const r = await fetch("/api/play/new?_t=" + Date.now(), {method: "POST"});
  const d = await r.json();
  if (d.ok){ play = d; playSlot = null; renderPlay(); }
  else alert(d.error || "开新局失败");
};

/* ================= 最近回放：列表 / 对局 / 播放器 ================= */

let replays = [];
let curReplay = null;   // /api/replay?file=.. 的 {file, games:[...]}
let curGame = null;     // /api/replay?file=..&game=N 的 {frames, meta, winner}
let curFrame = 0;
let playing = false;
let speed = 1;
let rafId = null;
let towerMax = {k0:4824,l0:3052,r0:3052,k1:4824,l1:3052,r1:3052};
let towerPos = {
  p0k:{x:9,y:3}, p0l:{x:3.5,y:6.5}, p0r:{x:14.5,y:6.5},
  p1k:{x:9,y:29}, p1l:{x:3.5,y:25.5}, p1r:{x:14.5,y:25.5},
};

function fmtSize(n){
  if (n >= 1048576) return (n/1048576).toFixed(1) + " MB";
  if (n >= 1024) return (n/1024).toFixed(1) + " KB";
  return n + " B";
}

/* —— 对阵标签：明确写出双方模型各自的训练步数 ——
   冻结副本 = main 的早期权重快照，因此统一显示为 main@步数：
   - 新录像：meta.steps 由评估端写入精确步数（main 当前步 / 副本同步步）；
   - 旧录像回退：main 用录像文件步数；frozen_copy 用 solo.copy_every 的最近同步边界近似。
*/
function replayStepVal(v){
  return (v === null || v === undefined || v === "") ? null : Number(v);
}
function replayModelName(id){
  return id === "frozen_copy" ? "main" : id;
}
function replayFileStep(file){
  const m = /league_(\d+)\.pkl$/.exec(file || "");
  return m ? parseInt(m[1], 10) : null;
}
function replayOldFrozenStep(fileStep){
  const C = (solo && solo.ok && solo.copy_every) ? Number(solo.copy_every) : 0;
  if (!C || fileStep === null) return null;
  return fileStep - (fileStep % C);   // 最近一次冻结副本同步边界（含当前边界）
}
function matchupText(pair, steps, file){
  const ids = Array.isArray(pair) ? pair : [];
  if (!ids.length) return "?";
  const st = Array.isArray(steps) ? steps : [];
  const fileStep = replayFileStep(file);
  const parts = ids.map((id, i) => {
    const name = replayModelName(id);
    let v = replayStepVal(st[i]);
    if (v === null && id === "frozen_copy") v = replayOldFrozenStep(fileStep);
    if (v === null && id === "main" && fileStep !== null) v = fileStep;
    return v === null ? name : name + "@" + v.toLocaleString();
  });
  return parts.join(" vs ");
}

async function refreshReplays(){
  try{
    const r = await fetch("/api/replays?_t=" + Date.now(), {cache:"no-store"});
    const data = await r.json();
    replays = data.ok ? (data.replays || []) : [];
  }catch(e){
    replays = [];
  }
  renderReplaysList();
}

function renderReplaysList(){
  const el = document.getElementById("replaysList");
  const sub = document.getElementById("replaysSub");
  if (!replays.length){
    el.innerHTML = '<p class="sub">暂无回放。训练每个评估周期（默认 2000 步）会自动保存 replays/league_&lt;步数&gt;.pkl。</p>';
    sub.textContent = "";
    return;
  }
  sub.textContent = `${replays.length} 个文件 · 每 5s 自动刷新`;
  el.innerHTML = replays.map(r => {
    const n = r.n_games === null ? "?" : r.n_games;
    const name = r.step === null ? r.file : "步数 " + r.step;
    const esc = r.file.replace(/"/g, "&quot;");
    return `<div class="rep-row" onclick="openReplay('${esc}')" title="${r.file}">
      <span class="rep-name">${name}</span>
      <span class="rep-meta">${n} 局 · ${fmtSize(r.size)} · ${r.mtime}</span>
    </div>`;
  }).join("");
}

async function openReplay(file){
  stopPlay();
  document.getElementById("playerCard").style.display = "none";
  const r = await fetch("/api/replay?file=" + encodeURIComponent(file) + "&_t=" + Date.now(),
                        {cache:"no-store"});
  const data = await r.json();
  if (!data.ok){ alert(data.error); return; }
  curReplay = data;
  renderGamesPanel();
  document.getElementById("gamesPanel").style.display = "block";
  document.getElementById("gamesPanel").scrollIntoView({behavior:"smooth"});
  // 统计范围若为"当前打开的回放"，打开新文件后同步刷新
  if (document.getElementById("statsScope").value === "file") loadCardStats("file");
}

function renderGamesPanel(){
  document.getElementById("gamesTitle").textContent = "对局列表 · " + curReplay.file;
  document.getElementById("gamesMeta").textContent =
    `${curReplay.games.length} 局 · 单击“播放”回放`;
  const body = document.getElementById("gamesBody");
  body.innerHTML = curReplay.games.map(g => {
    const pair = matchupText(g.pair, g.steps, curReplay.file);
    const w = g.winner === 0 ? '<span class="badge w0">先手胜</span>'
            : g.winner === 1 ? '<span class="badge w1">后手胜</span>'
            : '<span class="badge wd">平局</span>';
    return `<tr>
      <td>#${g.index + 1}</td>
      <td>${pair}</td>
      <td>${w}</td>
      <td>${g.n_frames}</td>
      <td>${g.duration.toFixed(1)}s</td>
      <td><button class="btn sm" onclick="openGame(${g.index})">▶ 播放</button></td>
    </tr>`;
  }).join("");
  document.getElementById("gamesEmpty").textContent = curReplay.games.length ? "" : "该文件无对局";
}

async function openGame(gi){
  if (!curReplay) return;
  const r = await fetch("/api/replay?file=" + encodeURIComponent(curReplay.file) +
                        "&game=" + gi + "&_t=" + Date.now(), {cache:"no-store"});
  const data = await r.json();
  if (!data.ok){ alert(data.error); return; }
  curGame = data;
  curFrame = 0;
  stopPlay();
  if (!curGame.frames || !curGame.frames.length){
    alert("该局无帧数据");
    return;
  }
  computeTowerMax();
  drawFrame(0);
  renderPlayerInfo();
  document.getElementById("playerCard").style.display = "block";
  document.getElementById("playerCard").scrollIntoView({behavior:"smooth"});
}

function computeTowerMax(){
  const f0 = curGame.frames[0];
  if (!f0) return;
  const t0 = f0.towers0 || [4824, 3052, 3052];
  const t1 = f0.towers1 || [4824, 3052, 3052];
  towerMax = {k0:t0[0], l0:t0[1], r0:t0[2], k1:t1[0], l1:t1[1], r1:t1[2]};
  // 塔位从首帧实体推导（引擎换图也自适应），缺省回退标准位
  const pos = {
    p0k:null, p0l:null, p0r:null,
    p1k:null, p1l:null, p1r:null,
  };
  (f0.entities || []).forEach(e => {
    const name = e[0], x = e[1], pl = e[4];
    if (name !== "KingTower" && name !== "King_PrincessTowers") return;
    const slot = name === "KingTower" ? "k" : (x < 9 ? "l" : "r");
    const key = "p" + pl + slot;
    if (pos[key] === null) pos[key] = {x: e[1], y: e[2]};
  });
  Object.keys(towerPos).forEach(k => {
    if (pos[k] !== null) towerPos[k] = pos[k];
  });
}

function frameAt(i){
  if (!curGame || !curGame.frames || !curGame.frames.length) return null;
  return curGame.frames[Math.max(0, Math.min(curGame.frames.length - 1, i))];
}

/* ---- 播放控制 ---- */

function renderControls(){
  const btn = document.getElementById("btnPlay");
  if (btn) btn.textContent = playing ? "⏸ 暂停" : "⏵ 播放";
}

function startPlay(){
  if (!curGame || !curGame.frames || curGame.frames.length <= 1) return;
  if (curFrame >= curGame.frames.length - 1) curFrame = 0;
  playing = true;
  animFrom = frameAt(Math.max(0, curFrame - 1));
  animTo = frameAt(curFrame);
  animStart = performance.now();
  animDur = 600 / speed;
  renderControls();
  loop();
}

function stopPlay(){
  playing = false;
  if (rafId){ cancelAnimationFrame(rafId); rafId = null; }
  renderControls();
}

function togglePlay(){
  if (!curGame) return;
  if (playing){ stopPlay(); }
  else startPlay();
}

function stepFrame(d){
  if (!curGame) return;
  stopPlay();
  curFrame = Math.max(0, Math.min(curGame.frames.length - 1, curFrame + d));
  drawFrame(curFrame);
}

let animFrom = null, animTo = null, animStart = 0, animDur = 1;

function loop(){
  if (!playing) return;
  const now = performance.now();
  let p = (now - animStart) / animDur;
  if (p >= 1){
    if (curFrame >= curGame.frames.length - 1){
      playing = false;
      renderControls();
      drawInterp(frameAt(curFrame), frameAt(curFrame), 1);
      renderPlayerInfo();
      return;
    }
    curFrame++;
    animFrom = frameAt(curFrame - 1);
    animTo = frameAt(curFrame);
    animStart = now;
    animDur = 600 / speed;
    p = 0;
  }
  drawInterp(animFrom, animTo, p);
  renderPlayerInfo();
  rafId = requestAnimationFrame(loop);
}

function drawFrame(i){
  const fr = frameAt(i);
  if (!fr) return;
  drawInterp(fr, fr, 1);
  renderPlayerInfo();
}

function renderPlayerInfo(){
  const el = document.getElementById("playerInfo");
  if (!curGame || !curGame.frames || !curGame.frames.length){ el.innerHTML = ""; return; }
  const fr = frameAt(curFrame);
  const n = curGame.frames.length;
  const meta = curGame.meta || {};
  const pair = matchupText(meta.pair, meta.steps, curGame.file);
  const winner = curGame.winner;
  const w = winner === 0 ? "先手胜" : winner === 1 ? "后手胜" : "平局";
  const bundle = (fr.bundle || []).map(b => {
    if (b[0] === "deploy") return "部署 slot" + b[1] + " @(" + b[2] + "," + b[3] + ")";
    return b[0] + "(" + (b[1] !== undefined ? b[1] : "") + ")";
  }).join("；") || "—";
  const opp = (fr.opp_played || []).map(o =>
    o.card ? o.card + "(" + o.x + "," + o.y + ")" : JSON.stringify(o)
  ).join("；") || "—";
  el.innerHTML = `
    <div><b>${pair}</b> · ${w}</div>
    <div class="kv"><span>帧</span><b>${curFrame + 1} / ${n}</b></div>
    <div class="kv"><span>时间</span><b>${fr.t.toFixed(1)}s</b></div>
    <div class="kv"><span>本步奖励</span>
      <b style="color:${fr.reward >= 0 ? "#22c55e" : "#ef4444"}">
        ${fr.reward >= 0 ? "+" : ""}${Number(fr.reward).toFixed(3)}</b></div>
    <div class="kv"><span>我方动作</span><b>${bundle}</b></div>
    <div class="kv"><span>对手出牌</span><b>${opp}</b></div>`;
  const scr = document.getElementById("scrub");
  scr.max = n - 1;
  scr.value = curFrame;
  document.getElementById("frameLabel").textContent = (curFrame + 1) + "/" + n;
  document.getElementById("playerTitle").textContent =
    ` · ${curGame.file} · 第 ${curGame.index + 1} 局 · ${pair} · ${w}`;
}

/* ---- Canvas 战场渲染（18×32 网格，纯前端自绘） ---- */

function lerpA(a, b, p){
  if (!a || !b) return a || b || [0, 0, 0];
  return a.map((v, i) => (b[i] === undefined ? v : v + (b[i] - v) * p));
}

function interpEntities(fromEnts, toEnts, p){
  fromEnts = fromEnts || [];
  toEnts = toEnts || [];
  const used = new Set();
  const out = [];
  const pick = t => ({kind: t[5], max_hp: t[6], shield: t[7], shield_max: t[8], radius: t[9]});
  for (const t of toEnts){
    let best = -1, bestD = 1e9;
    for (let i = 0; i < fromEnts.length; i++){
      if (used.has(i)) continue;
      const f = fromEnts[i];
      if (f[0] === t[0] && f[4] === t[4]){
        const d = (f[1] - t[1]) ** 2 + (f[2] - t[2]) ** 2;
        if (d < bestD){ bestD = d; best = i; }
      }
    }
    if (best >= 0 && bestD < 4){
      used.add(best);
      const f = fromEnts[best];
      out.push({name: t[0], x: f[1] + (t[1] - f[1]) * p, y: f[2] + (t[2] - f[2]) * p,
                hp: f[3] + (t[3] - f[3]) * p, player: t[4], alpha: 1, scale: 1,
                ...pick(t)});
    } else {
      out.push({name: t[0], x: t[1], y: t[2], hp: t[3], player: t[4], alpha: p, scale: p,
                ...pick(t)});
    }
  }
  for (let i = 0; i < fromEnts.length; i++){
    if (!used.has(i)){
      const f = fromEnts[i];
      out.push({name: f[0], x: f[1], y: f[2], hp: f[3], player: f[4], alpha: 1 - p, scale: 1 - p,
                ...pick(f)});
    }
  }
  return out;
}

// 实体名显示：内部英文名 → 友好中文名（2026-09-10）。
// 滚木/滚筒引擎侧已改为"凭空出现在部署点"（无 LogProjectile 第一段），此处映射
// 兼容旧 replay 里残留的 LogProjectile 与新 replay 的 LogProjectileRolling。
function displayName(name, kind){
  const M = {
    "LogProjectileRolling": "滚木", "LogProjectile": "滚木",
    "BarbLogProjectileRolling": "滚筒", "BarbLogProjectile": "滚筒",
    "ArrowsSpell": "箭雨", "ArcherArrow": "箭",
    "TowerPrincessProjectile": "塔箭", "KingProjectile": "王塔箭",
    "FirecrackerExplosion": "爆裂",
    "ArrowsSpellDeco": "箭雨", "FireballSpell": "火球", "RocketSpell": "火箭",
    "GoblinBarrelSpell": "飞桶", "xbow_projectile": "连弩箭",
    "TowerCannonball": "炮塔弹", "MortarProjectile": "迫击炮弹",
    "MusketeerProjectile": "火枪弹", "BabyDragonProjectile": "龙焰",
    "WitchProjectile": "女巫弹", "PrincessProjectile": "公主弹",
    "SpearGoblinProjectile": "投矛", "RoyalGiantProjectile": "巨炮弹",
    "BombTowerProjectile": "炸弹塔弹", "MinionSpit": "小骷髅弹",
  };
  const n = M[name];
  if (n) return n;
  return name;
}

function drawInterp(from, to, p){
  drawInterpOn(document.getElementById("arena"), from, to, p);
}

function drawInterpOn(canvas, from, to, p){
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#0b3d2e";
  ctx.fillRect(0, 0, W, H);
  const scale = Math.min(W / 18, H / 32);
  const ox = (W - 18 * scale) / 2, oy = (H - 32 * scale) / 2;
  const X = x => ox + x * scale, Y = y => oy + y * scale;

  // 网格
  ctx.strokeStyle = "rgba(255,255,255,0.07)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 1; i < 18; i++){ ctx.moveTo(X(i), Y(0)); ctx.lineTo(X(i), Y(32)); }
  for (let j = 1; j < 32; j++){ ctx.moveTo(X(0), Y(j)); ctx.lineTo(X(18), Y(j)); }
  ctx.stroke();
  // 河道（y 14..18）
  ctx.fillStyle = "rgba(56,189,248,0.16)";
  ctx.fillRect(X(0), Y(14), 18 * scale, 4 * scale);
  ctx.fillStyle = "rgba(56,189,248,0.28)";
  ctx.fillRect(X(0), Y(15.5), 18 * scale, 1 * scale);
  // 桥（pygame：x∈[2,5] 与 [13,16] 跨河道 2 格）
  ctx.fillStyle = "rgba(148,163,184,0.30)";
  ctx.fillRect(X(2), Y(15), 3 * scale, 2 * scale);
  ctx.fillRect(X(13), Y(15), 3 * scale, 2 * scale);
  // 四角部署区（6×1，pygame 布局：上下两端左右角）
  ctx.fillStyle = "rgba(100,116,139,0.22)";
  ctx.fillRect(X(0), Y(0), 6 * scale, 1 * scale);
  ctx.fillRect(X(12), Y(0), 6 * scale, 1 * scale);
  ctx.fillRect(X(0), Y(31), 6 * scale, 1 * scale);
  ctx.fillRect(X(12), Y(31), 6 * scale, 1 * scale);

  if (!to) return;

  const t = (from && from.t !== undefined && to.t !== undefined)
    ? from.t + (to.t - from.t) * p : (to.t || 0);
  const t0 = lerpA(from && from.towers0, to.towers0, p);
  const t1 = lerpA(from && from.towers1, to.towers1, p);

  // 塔（player0 在下半场，player1 在上半场；位置取自首帧实体，缺省标准位）
  const towers = [
    {x: towerPos.p0l.x, y: towerPos.p0l.y, king: false, hp: t0[1], player: 0},
    {x: towerPos.p0r.x, y: towerPos.p0r.y, king: false, hp: t0[2], player: 0},
    {x: towerPos.p0k.x, y: towerPos.p0k.y, king: true,  hp: t0[0], player: 0},
    {x: towerPos.p1l.x, y: towerPos.p1l.y, king: false, hp: t1[1], player: 1},
    {x: towerPos.p1r.x, y: towerPos.p1r.y, king: false, hp: t1[2], player: 1},
    {x: towerPos.p1k.x, y: towerPos.p1k.y, king: true,  hp: t1[0], player: 1},
  ];
  towers.forEach(tw => {
    const mx = tw.king
      ? (tw.player === 0 ? towerMax.k0 : towerMax.k1)
      : (tw.player === 0 ? (tw.x < 9 ? towerMax.l0 : towerMax.r0)
                          : (tw.x < 9 ? towerMax.l1 : towerMax.r1));
    const frac = mx > 0 ? Math.max(0, Math.min(1, tw.hp / mx)) : 0;
    // 塔 = 方形（2026-09-10 用户口径：圆形换方形、不显示血量数字）；
    // 方形内按血量比例填充颜色（高血绿/中黄/低血红），塔破画灰×
    const s = (tw.king ? 0.62 : 0.50) * scale;
    const col = tw.player === 0 ? "#3b82f6" : "#ef4444";
    const bx = X(tw.x) - s, by = Y(tw.y) - s, bs = s * 2;
    ctx.fillStyle = "rgba(15,23,42,0.55)";
    ctx.fillRect(bx, by, bs, bs);
    if (frac > 0){
      ctx.fillStyle = frac > 0.5 ? "#22c55e" : (frac > 0.25 ? "#eab308" : "#ef4444");
      // 血条：方形内底部按比例填充（宽度 = 血量比例）
      ctx.fillRect(bx + 2, by + bs - 5, (bs - 4) * frac, 3);
    } else {
      ctx.strokeStyle = "#475569";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(bx, by); ctx.lineTo(bx + bs, by + bs);
      ctx.moveTo(bx + bs, by); ctx.lineTo(bx, by + bs);
      ctx.stroke();
    }
    ctx.strokeStyle = col;
    ctx.lineWidth = 2;
    ctx.strokeRect(bx, by, bs, bs);
  });

  // 实体（pygame 风格：部队实心圆+名字+血条 / Building 中央 HP / Projectile 空心 / 效果淡出）
  const ents = interpEntities(from && from.entities, to.entities, p)
    .filter(e => e.name !== "KingTower" && e.name !== "King_PrincessTowers");
  ents.forEach(e => {
    const rad = e.radius ? e.radius * scale : 0.34 * scale;
    const r = Math.max(3.5, rad);
    const col = e.player === 0 ? "#60a5fa" : "#f87171";
    ctx.globalAlpha = Math.max(0, Math.min(1, e.alpha || 1));
    ctx.beginPath();
    ctx.arc(X(e.x), Y(e.y), r * (e.scale || 1), 0, 7);
    if (e.kind === "projectile"){
      ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.stroke();   // 空心
    } else {
      ctx.fillStyle = col; ctx.fill();
      ctx.strokeStyle = "rgba(255,255,255,0.7)"; ctx.lineWidth = 1; ctx.stroke();
    }
    // HP 条（shield 优先）；Building 中央显示 HP 数字
    const mh = (e.max_hp && e.max_hp > 0) ? e.max_hp : 1000;
    const frac = (e.shield > 0 && e.shield_max > 0)
      ? Math.max(0, Math.min(1, e.shield / e.shield_max))
      : Math.max(0, Math.min(1, e.hp / mh));
    if (e.kind !== "projectile"){
      const bw = Math.max(r * 2, 16), bh = 3;
      ctx.fillStyle = "rgba(0,0,0,0.5)";
      ctx.fillRect(X(e.x) - bw / 2, Y(e.y) - r - 8, bw, bh + 1);
      ctx.fillStyle = e.shield > 0 ? "#a78bfa" : "#22c55e";
      ctx.fillRect(X(e.x) - bw / 2, Y(e.y) - r - 7, bw * frac, bh);
    }
    // 实体名：内部名→中文（displayName）；弹射物/特效用浅色小字淡出不喧宾夺主
    const isFx = (e.kind === "projectile" || e.kind === "effect");
    ctx.fillStyle = isFx ? "rgba(226,232,240,0.55)" : "#e2e8f0";
    ctx.font = isFx ? "8px sans-serif" : "9px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(displayName(e.name, e.kind), X(e.x), Y(e.y) - r - 9);
    if (e.kind === "building"){
      ctx.fillStyle = "#fff";
      ctx.fillText(Math.round(e.hp), X(e.x), Y(e.y) + 3);
    }
    ctx.globalAlpha = 1;
  });

  // 圣水条 + 时间 + 皇冠
  const e0 = from && to ? from.elixir0 + (to.elixir0 - from.elixir0) * p : (to.elixir0 || 0);
  const e1 = from && to ? from.elixir1 + (to.elixir1 - from.elixir1) * p : (to.elixir1 || 0);
  drawElixir(ctx, W, H, 0, e0, "#60a5fa", "我方");
  drawElixir(ctx, W, H, 1, e1, "#f87171", "对手");
  ctx.fillStyle = "#e2e8f0";
  ctx.font = "12px sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("⏱ " + t.toFixed(1) + "s", W / 2, 18);
  ctx.textAlign = "left";
  ctx.fillStyle = "#60a5fa";
  ctx.fillText("皇冠 " + (to.crown0 || 0), 10, 18);
  ctx.textAlign = "right";
  ctx.fillStyle = "#f87171";
  ctx.fillText("皇冠 " + (to.crown1 || 0), W - 10, 18);
}

function drawElixir(ctx, W, H, side, val, color, label){
  const bw = W * 0.32, bh = 7;
  const x = side === 0 ? 8 : W - bw - 8;
  const y = H - 14;
  ctx.fillStyle = "rgba(15,23,42,0.7)";
  ctx.fillRect(x, y, bw, bh);
  ctx.fillStyle = color;
  ctx.fillRect(x, y, bw * Math.max(0, Math.min(1, val / 10)), bh);
  ctx.fillStyle = "#e2e8f0";
  ctx.font = "10px sans-serif";
  ctx.textAlign = side === 0 ? "left" : "right";
  ctx.fillText(label + " " + val.toFixed(1), side === 0 ? x : W - 8, y - 3);
}

/* ---- 卡牌使用统计（各卡组/模型的实际出牌） ---- */

let cardStats = null;
let cardStatsLoaded = false;

function escHtml(s){
  return String(s === null || s === undefined ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function statColor(frac){
  const a = 0.10 + 0.72 * Math.max(0, Math.min(1, frac));
  return "rgba(37,99,235," + a.toFixed(3) + ")";
}

async function loadCardStats(scopeOverride){
  const sel = document.getElementById("statsScope");
  const scope = (scopeOverride !== undefined) ? scopeOverride : sel.value;
  let url = "/api/cardstats?_t=" + Date.now();
  if (scope === "file"){
    if (curReplay && curReplay.file) url += "&file=" + encodeURIComponent(curReplay.file);
    else url += "&files=3";   // 还没打开过回放 → 退回最近 3 个
  } else {
    url += "&files=" + encodeURIComponent(scope);
  }
  document.getElementById("statsSub").textContent = "加载中…";
  let data = null;
  try{
    const r = await fetch(url, {cache:"no-store"});
    data = await r.json();
  }catch(e){ data = {ok:false, error:String(e)}; }
  cardStats = data;
  cardStatsLoaded = true;
  renderCardStats();
}

function renderCardStats(){
  const d = cardStats;
  const card = document.getElementById("statsCard");
  const sub = document.getElementById("statsSub");
  const agentsEl = document.getElementById("statsAgents");
  const table = document.getElementById("statsTable");
  const note = document.getElementById("statsNote");
  if (!d || !d.ok){
    if (d && d.error && d.error.indexOf("回放目录") >= 0){ card.style.display = "none"; return; }
    card.style.display = "block";
    sub.textContent = "";
    agentsEl.innerHTML = "";
    table.innerHTML = "";
    note.textContent = (d && d.error) ? ("统计失败：" + d.error) : "暂无统计";
    return;
  }
  card.style.display = "block";
  sub.textContent = `${d.files.length} 个回放 · ${d.n_games} 局 · ${d.agents.length} 个模型`;

  if (!d.agents.length){
    agentsEl.innerHTML = "";
    table.innerHTML = "";
    note.textContent = "该范围内的回放里没有可统计的出牌记录。";
    return;
  }

  // 概览：每个模型（卡组）的出战局数 / 出牌数 / 卡种数
  agentsEl.innerHTML = d.agents.map(a =>
    `<div class="stats-agent"><b>${escHtml(a.label)}</b>
      <span class="m">· ${a.games} 局 · 出牌 ${a.plays}（${a.per_game}/局）· ${a.n_distinct} 种卡</span>
    </div>`).join("");

  // 矩阵：行 = 卡牌（按总出牌次数降序），列 = 模型；单元格 = 次数 + 占该模型的百分比
  const totals = {};
  d.agents.forEach(a => {
    for (const c in a.cards) totals[c] = (totals[c] || 0) + a.cards[c];
  });
  const cards = Object.keys(totals).sort((x, y) => totals[y] - totals[x] || x.localeCompare(y));
  const colMax = {};
  d.agents.forEach(a => {
    let mx = 1;
    for (const c in a.cards) if (a.cards[c] > mx) mx = a.cards[c];
    colMax[a.model] = mx;
  });

  let html = "<thead><tr><th>卡牌</th><th class='num'>总计</th>" +
    d.agents.map(a =>
      `<th class="num">${escHtml(a.label)}<br>` +
      `<span style="font-weight:400;color:#94a3b8">${a.games} 局</span></th>`).join("") +
    "</tr></thead><tbody>";
  for (const c of cards){
    html += `<tr><td class="card-name">${escHtml(c)}</td><td class="num">${totals[c]}</td>`;
    for (const a of d.agents){
      const n = a.cards[c] || 0;
      if (!n){ html += `<td class="num" style="color:#475569">·</td>`; continue; }
      const frac = n / (colMax[a.model] || 1);
      const pct = a.plays ? (n / a.plays * 100) : 0;
      html += `<td class="num" style="background:${statColor(frac)}">` +
              `<span class="cellbar" style="width:${Math.round(frac * 36)}px"></span>` +
              `${n} <span style="color:#cbd5e1">${pct.toFixed(1)}%</span></td>`;
    }
    html += "</tr>";
  }
  table.innerHTML = html + "</tbody>";
  renderDeckMatrix(d);

  const cov = d.coverage || {};
  if (cov.partial){
    note.textContent = `注：共 ${cov.n_games} 局，其中 ${cov.n_games - cov.side0_games} 局为旧录像` +
      `（未记录我方出牌），这些局只统计到对手侧；新录像起双方都会被记录。`;
  } else {
    note.textContent = "双侧完整统计：每次出牌都归属到对应卡组/模型。百分比 = 该卡占该模型总出牌数的比例。";
  }
  if (cov.deck_meta_partial){
    note.textContent += ` ⚠️ 其中仅 ${cov.games_with_decks}/${cov.n_games} 局带 meta.decks（旧录像无卡组元数据）` +
      `⇒「按卡组」表只覆盖有元数据的部分。`;
  }
}

/* 多卡组对战统计：行 = 卡牌，列 = **卡组**（不是模型）。
   数据源：回放 meta.decks（双方卡组清单）+ 逐帧出牌归因（我方 frames[].cards / 对手 frames[].opp_played）。
   后端 /api/cardstats 的 decks[] 已算好；旧录像没有 meta.decks ⇒ 该表为空并给出提示。 */
function renderDeckMatrix(d){
  const tbl = document.getElementById("statsDeckTable");
  const sub = document.getElementById("statsDeckSub");
  if (!tbl) return;
  const decks = d.decks || [];
  const cov = d.coverage || {};
  if (!decks.length){
    tbl.innerHTML = "";
    if (sub) sub.textContent = cov.deck_meta_partial ? "（旧录像无 meta.decks，无法按卡组切分）" : "（暂无卡组数据）";
    return;
  }
  const totals = {};
  decks.forEach(k => { for (const c in k.cards) totals[c] = (totals[c] || 0) + k.cards[c]; });
  const cards = Object.keys(totals).sort((x, y) => totals[y] - totals[x] || x.localeCompare(y));
  const colMax = {};
  decks.forEach(k => { let mx = 1; for (const c in k.cards) if (k.cards[c] > mx) mx = k.cards[c]; colMax[k.key] = mx; });
  const head = k => `${escHtml(k.known ? k.archetype : k.label)}<br>` +
    `<span style="font-weight:400;color:#94a3b8">${k.side_games} 边 · ${k.plays} 次` +
    (k.per_side_game !== undefined ? ` · ${k.per_side_game}/边` : "") + `</span>`;
  let html = "<thead><tr><th>卡牌</th><th class='num'>总计</th>" +
    decks.map(k => `<th class="num" title="${escHtml((k.deck || []).join(' · '))}">${head(k)}</th>`).join("") +
    "</tr></thead><tbody>";
  for (const c of cards){
    html += `<tr><td class="card-name">${escHtml(c)}</td><td class="num">${totals[c]}</td>`;
    for (const k of decks){
      const n = k.cards[c] || 0;
      if (!n){ html += `<td class="num" style="color:#475569">·</td>`; continue; }
      const frac = n / (colMax[k.key] || 1);
      const pct = k.plays ? (n / k.plays * 100) : 0;
      html += `<td class="num" style="background:${statColor(frac)}">` +
              `<span class="cellbar" style="width:${Math.round(frac * 36)}px"></span>` +
              `${n} <span style="color:#cbd5e1">${pct.toFixed(1)}%</span></td>`;
    }
    html += "</tr>";
  }
  tbl.innerHTML = html + "</tbody>";
  if (sub) sub.textContent = `${decks.length} 副卡组 · 「边」= 该卡组出战（局×边）次数` +
    (cov.deck_meta_partial ? ` · ⚠️ 仅 ${cov.games_with_decks}/${cov.n_games} 局有卡组元数据` : "");
}

/* ---- 事件绑定 + 启动 ---- */

document.getElementById("btnPlay").onclick = togglePlay;
document.getElementById("btnPrev").onclick = () => stepFrame(-1);
document.getElementById("btnNext").onclick = () => stepFrame(1);
document.getElementById("closePlayer").onclick = () => {
  stopPlay();
  document.getElementById("playerCard").style.display = "none";
};
document.getElementById("reloadReplays").onclick = refreshReplays;
document.getElementById("statsReload").onclick = () => loadCardStats();
document.getElementById("statsScope").onchange = () => loadCardStats();
document.getElementById("speedSel").onchange = e => { speed = parseFloat(e.target.value); };
document.getElementById("scrub").oninput = e => {
  stopPlay();
  curFrame = parseInt(e.target.value, 10) || 0;
  drawFrame(curFrame);
};

refresh();
setInterval(refresh, 3000);
refreshReplays();
setInterval(refreshReplays, 5000);
loadCardStats();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    state_path = "league_state.json"
    replays_dir = "replays"
    sweep_root = None   # flow-sweep 根目录（--sweep；可为 runs/<name>/ 或某策略目录）
    solo_path = None    # solo 状态文件（--solo；单人自对弈，无联赛）
    play_policy_path = None   # 人机对战：FollowerPolicy checkpoint（--play）
    play_out_dir = None       # 人对局数据落盘目录（--play-out）
    play_cfg = None
    play_seed = 0
    play_games_saved = 0
    play_session = None
    play_error = None

    def _ensure_play(self):
        """懒加载人机对战 session（首次 /api/play/* 时初始化）。"""
        if Handler.play_error:
            return None
        if Handler.play_session is None and Handler.play_policy_path:
            try:
                from rl.human_play import HumanPlaySession, load_policy, DEFAULT_PLAY_DECK
                from rl.config import TrainConfig
                cfg = Handler.play_cfg or TrainConfig.resolve("standard")
                pol = load_policy(Handler.play_policy_path, hidden_dim=cfg.hidden_dim)
                Handler.play_session = HumanPlaySession(
                    pol, cfg=cfg, deck=DEFAULT_PLAY_DECK, seed=Handler.play_seed,
                    max_steps=600, out_dir=Handler.play_out_dir)
            except Exception as e:   # noqa: BLE001 —— 初始化失败给出可见错误
                Handler.play_error = f"{type(e).__name__}: {e}"
                return None
        return Handler.play_session

    def _read_body(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return {}

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        route = parsed.path
        qs = parse_qs(parsed.query)
        if route == "/" or route == "/index.html":
            self._send(200, _HTML.encode("utf-8"), "text/html; charset=utf-8")
        elif route == "/api/state":
            self._send(200, json.dumps(build_payload(self.state_path)).encode("utf-8"),
                       "application/json; charset=utf-8")
        elif route == "/api/sweep":
            self._send(200, json.dumps(build_sweep_payload(self.sweep_root)).encode("utf-8"),
                       "application/json; charset=utf-8")
        elif route == "/api/solo":
            self._send(200, json.dumps(build_solo_payload(self.solo_path)).encode("utf-8"),
                       "application/json; charset=utf-8")
        elif route == "/api/play/state":
            sess = self._ensure_play()
            if Handler.play_error:
                self._send(200, json.dumps({"ok": False, "error": Handler.play_error}).encode("utf-8"),
                           "application/json; charset=utf-8")
            elif sess is None:
                self._send(200, json.dumps({"ok": False, "error": "未指定 --play 策略"}).encode("utf-8"),
                           "application/json; charset=utf-8")
            else:
                self._send(200, json.dumps(sess.state()).encode("utf-8"),
                           "application/json; charset=utf-8")
        elif route == "/api/replays":
            self._send(200, json.dumps(build_replays_payload(self.replays_dir)).encode("utf-8"),
                       "application/json; charset=utf-8")
        elif route == "/api/replay":
            file = (qs.get("file") or [""])[0]
            game = qs.get("game")
            gi = int(game[0]) if game and game[0].isdigit() else None
            payload = load_replay_payload(self.replays_dir, file, gi)
            self._send(200, json.dumps(payload).encode("utf-8"),
                       "application/json; charset=utf-8")
        elif route == "/api/cardstats":
            # 卡牌使用统计：?file=xxx 单文件；?files=N 最近 N 个（N<=0 = 全部）；缺省最近 3 个
            file = (qs.get("file") or [""])[0] or None
            nf = qs.get("files")
            n_files = int(nf[0]) if nf and nf[0].lstrip("-").isdigit() else 3
            payload = build_card_stats_payload(self.replays_dir, file, n_files)
            self._send(200, json.dumps(payload).encode("utf-8"),
                       "application/json; charset=utf-8")
        elif route == "/favicon.ico":
            # 空 204：页面全内联无外部资源，消除浏览器自动请求 favicon 的 404
            # （DevTools 会报 "No resource with given URL found"，纯噪音非白屏）
            self._send(204, b"", "image/x-icon")
        else:
            self._send(404, b"not found", "text/plain; charset=utf-8")

    def log_message(self, fmt, *args):
        sys.stderr.write("[dashboard] %s\n" % (fmt % args))

    def do_POST(self):
        parsed = urlparse(self.path)
        route = parsed.path
        if route == "/api/play/action":
            sess = self._ensure_play()
            if Handler.play_error:
                self._send(200, json.dumps({"ok": False, "error": Handler.play_error}).encode("utf-8"),
                           "application/json; charset=utf-8")
                return
            if sess is None:
                self._send(200, json.dumps({"ok": False, "error": "未指定 --play 策略"}).encode("utf-8"),
                           "application/json; charset=utf-8")
                return
            body = self._read_body()
            try:
                st = sess.act(int(body.get("slot")), int(body.get("x")), int(body.get("y")))
            except (TypeError, ValueError):
                st = {"ok": False, "error": "参数需为 slot,x,y 整数"}
            self._send(200, json.dumps(st).encode("utf-8"), "application/json; charset=utf-8")
        elif route == "/api/play/new":
            sess = self._ensure_play()
            if sess is None:
                self._send(200, json.dumps({"ok": False, "error": "未指定 --play 策略"}).encode("utf-8"),
                           "application/json; charset=utf-8")
                return
            from rl.human_play import HumanPlaySession
            sess.save(out_dir=Handler.play_out_dir)   # 结束并保存上一局数据
            Handler.play_games_saved += 1
            Handler.play_seed += 1
            Handler.play_session = HumanPlaySession(
                sess.policy, cfg=Handler.play_cfg, deck=sess.deck,
                seed=Handler.play_seed, max_steps=sess.max_steps,
                out_dir=Handler.play_out_dir)
            self._send(200, json.dumps(Handler.play_session.state()).encode("utf-8"),
                       "application/json; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain; charset=utf-8")


def make_demo_state(path, n_points=10, seed=0):
    """生成一份演示用联赛状态（5 卡组模型 + main + 合成 Elo 历史），便于直接预览 UI。"""
    import random
    rng = random.Random(seed)
    models = [
        ("main", "main", 1500.0, +4.0),
        ("push_flow", "baseline", 1500.0, +1.8),
        ("counter_flow", "baseline", 1500.0, +1.2),
        ("lockdown_flow", "baseline", 1500.0, +0.8),
        ("all_decks", "baseline", 1500.0, +0.4),
        ("random_deck", "baseline", 1500.0, -0.5),
    ]
    agents = []
    ratings = {}
    elo_history = {}
    round_stats = []
    total = n_points * 2000
    for aid, kind, init, trend in models:
        agents.append({"agent_id": aid, "kind": kind,
                       "path": f"{aid}_ckpt.pt" if kind == "historical" else None})
        elo = init
        hist = []
        for i in range(n_points + 1):
            step = i * (total // n_points)
            elo = min(1800.0, max(1400.0, elo + trend + rng.uniform(-8, 8)))
            hist.append([step, round(elo, 1)])
        ratings[aid] = round(hist[-1][1], 1)
        elo_history[aid] = hist
        # 演示误差棒：每轮聚合 SE≈347.5/√(5×40)≈25（50% 胜率最坏情形）
        for i in range(n_points + 1):
            if len(round_stats) <= i:
                round_stats.append({"step": i * (total // n_points), "est": {}, "games": {}})
            round_stats[i]["est"][aid] = [hist[i][1], 25.0]
            round_stats[i]["games"][aid] = 200
    state = {
        "ratings": ratings,
        "winrates": {},
        "agents": agents,
        "history": [],
        "exploiter_counter": 1,
        "ckpt_counter": {},
        "elo_history": elo_history,
        "round_stats": round_stats,
        "total_steps": total,
        "demo": True,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f)
    print(f"[demo] 已生成演示状态 {path}（{len(models)} 模型，{n_points + 1} 个评估点）")


def make_demo_sweep(root, seed=1):
    """生成两份演示 flow-sweep summary（stream 20 轮 / games5 4 轮），便于预览 UI。"""
    import random
    rng = random.Random(seed)
    os.makedirs(root, exist_ok=True)
    specs = [
        ("stream", 20, 1, "每对 1 局（忠实流式）× 20 次完整训练", 1488, 29760),
        ("games5", 4, 5, "每对 5 局 × 4 次完整训练", 7440, 29760),
    ]
    ids = ["push_flow", "counter_flow", "lockdown_flow",
           "all_decks", "random_deck", "main"]
    for strategy, n_runs, gpp, desc, per_run, total in specs:
        rows = []
        se = 347.5 / (5 * 10) ** 0.5   # main 每轮 5 对 × 10 局评估 → 1σ≈49
        for i in range(n_runs):
            base = 1500 + (i / max(1, n_runs - 1)) * 55 + rng.uniform(-12, 12)
            main_est = [round(base, 1), round(se, 1)]
            est = {mid: [round(base + rng.uniform(-25, 25), 1), round(se, 1)]
                   for mid in ids}
            rows.append({"run": i, "games": per_run,
                         "main_est": main_est, "est": est})
        summary = {
            "strategy": strategy, "desc": desc,
            "pool_scale": 0.1,
            "pool_sizes": {"push_flow": 6, "counter_flow": 12, "lockdown_flow": 2,
                           "all_decks": 20, "random_deck": 3, "main": 20},
            "n_runs": n_runs, "games_per_pair": gpp,
            "per_run_games": per_run, "total_games": total,
            "eval_games_per_pair": 10, "device": "cpu",
            "status": "done", "run_current": n_runs,
            "elapsed_s": 0.0, "eta_s": 0.0,
            "rows": rows,
            "trend": {"first_main_est": rows[0]["main_est"],
                      "last_main_est": rows[-1]["main_est"],
                      "delta": round(rows[-1]["main_est"][0] - rows[0]["main_est"][0], 1),
                      "delta_se": round(2 ** 0.5 * se, 1),
                      "z": round((rows[-1]["main_est"][0] - rows[0]["main_est"][0]) /
                                 (2 ** 0.5 * se), 2),
                      "verdict": "上涨（≥2σ）"},
        }
        d = os.path.join(root, f"flow_sweep_{strategy}")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[demo] 已生成演示 flow-sweep -> {d}/summary.json（{n_runs} 轮）")


def make_demo_solo(path, n_points=10, seed=3):
    """生成一份演示 solo_state.json（自对弈胜率上升曲线），便于预览 UI。"""
    import random
    rng = random.Random(seed)
    deck = ["Knight", "MiniPekka", "Arrows", "Minions",
            "Musketeer", "Fireball", "Giant", "Archer"]
    history = []
    total = n_points * 2000
    wr = 0.42
    for i in range(n_points + 1):
        step = i * (total // n_points)
        wr = min(0.72, wr + 0.018 + rng.uniform(-0.02, 0.02))
        n = 40
        wins = int(round(wr * n))
        se = (wr * (1 - wr) / n) ** 0.5
        history.append({"step": step, "wins": wins, "losses": n - wins, "draws": 0,
                        "games": n, "winrate": round(wr, 4), "winrate_se": round(se, 4),
                        "mean_reward": round(0.2 + i * 0.03 + rng.uniform(-0.05, 0.05), 3)})
    state = {
        "mode": "solo",
        "agents": [{"agent_id": "main", "kind": "main", "path": None}],
        "history": history, "total_steps": total, "target_steps": total,
        "deck": deck, "opponent": "self-play-frozen-copy", "copy_every": 2000,
        "status": "done", "demo": True,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    print(f"[demo] 已生成演示 solo 状态 -> {path}（{len(history)} 个评估点）")


def main():
    ap = argparse.ArgumentParser(description="RL 训练仪表盘（Elo / flow-sweep / solo 自对弈 + 回放）")
    ap.add_argument("--state", type=str, default=None,
                    help="run 模式训练目录或 league_state.json（`--state runs/long1m` 亦可；"
                         "目录里会再读 run_state.json + config.json 显示 1M 长跑进度/ETA/评估点计划）。"
                         "不传则联赛面板关闭；只显示 --solo/--sweep/--play 面板")
    ap.add_argument("--sweep", type=str, default=None,
                    help="flow-sweep 根目录（--mode flow-sweep-* 的产物 runs/<name>/，或某个 "
                         "flow_sweep_<strategy> 策略目录；实时显示训练进度/曲线）")
    ap.add_argument("--solo", type=str, default=None,
                    help="solo 自对弈状态文件（--mode solo 写出的 solo_state.json，或含它的目录；"
                         "无联赛机制，显示胜率曲线/进度）")
    ap.add_argument("--play", type=str, default=None,
                    help="人机对战：FollowerPolicy checkpoint（.pt）或 runs/<name> 目录"
                         "（自动找 solo_main.pt / main_final.pt / flow_main.pt）；人在浏览器打模型，"
                         "对局数据落盘 --play-out（EpisodeReplay + BC 样本）")
    ap.add_argument("--play-out", type=str, default=None,
                    help="人机对战数据落盘目录（缺省 = 策略所在目录/human_data）")
    ap.add_argument("--play-config", type=str, default=None,
                    help="人机对战奖励配置（缺省 standard）")
    ap.add_argument("--replays", type=str, default=None,
                    help="回放目录（缺省 = 状态文件同目录/replays）")
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--demo", action="store_true",
                    help="状态文件不存在时生成演示数据（Elo + flow-sweep + solo + 回放）")
    ap.add_argument("--demo-points", type=int, default=10)
    args = ap.parse_args()
    # --state 接受 JSON 文件或 runs/<name> 目录（目录里自动找 league_state.json）
    state_abs = resolve_state_path(args.state) if args.state else None
    if args.state and os.path.isdir(os.path.abspath(args.state)) and state_abs is None:
        print(f"[dashboard] ⚠ {args.state} 是目录但里面没有 league_state.json"
              f"（run 模式还没写出第一个评估点？）", flush=True)
    if args.demo and not (state_abs and os.path.exists(state_abs)):
        state_abs = state_abs or os.path.join(os.path.abspath("."), "league_state.json")
        make_demo_state(state_abs, n_points=args.demo_points)
    sweep_abs = args.sweep and os.path.abspath(args.sweep) or None
    if args.demo:
        if sweep_abs is None:
            sweep_abs = os.path.join(os.path.dirname(state_abs), "demo_sweep")
        if not os.path.isdir(sweep_abs):
            make_demo_sweep(sweep_abs)
    solo_abs = None
    if args.solo:
        solo_abs = os.path.abspath(args.solo)
        if os.path.isdir(solo_abs):
            solo_abs = os.path.join(solo_abs, "solo_state.json")
    elif args.demo:
        solo_abs = os.path.join(os.path.dirname(state_abs), "solo_state.json")
        if not os.path.exists(solo_abs):
            make_demo_solo(solo_abs, n_points=args.demo_points)
    # 回放目录：显式 --replays 优先；否则从状态文件所在 runs 目录派生。
    # 旧版只从 --state（联赛）派生——--solo 启动时落到 cwd/replays，扫不到录像。
    _replays_root = None
    if state_abs:
        _replays_root = os.path.dirname(state_abs)
    elif solo_abs:
        _replays_root = os.path.dirname(solo_abs)
    elif sweep_abs:
        _replays_root = sweep_abs if os.path.isdir(sweep_abs) else os.path.dirname(sweep_abs)
    replays_abs = args.replays and os.path.abspath(args.replays) \
        or (os.path.join(_replays_root, "replays") if _replays_root
            else os.path.join(os.path.abspath("."), "replays"))
    if args.demo:
        make_demo_replays(replays_abs)
    Handler.state_path = state_abs
    Handler.replays_dir = replays_abs
    Handler.sweep_root = sweep_abs
    Handler.solo_path = solo_abs
    # 人机对战：--play 指向 checkpoint 或 runs/<name> 目录（自动挑主模型）
    if args.play:
        p = os.path.abspath(args.play)
        if os.path.isdir(p):
            for cand in ("solo_main.pt", "main_final.pt", "flow_main.pt"):
                if os.path.isfile(os.path.join(p, cand)):
                    p = os.path.join(p, cand)
                    break
        Handler.play_policy_path = p
        Handler.play_out_dir = args.play_out and os.path.abspath(args.play_out) \
            or os.path.join(os.path.dirname(p) or ".", "human_data")
        if args.play_config:
            from rl.config import TrainConfig
            Handler.play_cfg = TrainConfig.resolve(args.play_config)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[dashboard] http://{args.host}:{args.port}  (state={Handler.state_path or '未指定，联赛面板关闭'})",
          flush=True)
    if Handler.sweep_root:
        print(f"[dashboard] flow-sweep 目录: {Handler.sweep_root}", flush=True)
    if Handler.solo_path:
        print(f"[dashboard] solo 状态: {Handler.solo_path}", flush=True)
    if Handler.play_policy_path:
        print(f"[dashboard] 人机对战策略: {Handler.play_policy_path}"
              f"（数据 -> {Handler.play_out_dir}）", flush=True)
    print(f"[dashboard] 回放目录: {Handler.replays_dir}  Ctrl+C 退出", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
