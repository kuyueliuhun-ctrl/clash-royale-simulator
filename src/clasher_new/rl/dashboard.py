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

# 仓库根。⚠️ `_PARENT` = `src/clasher_new`（= `rl` 包的父目录，供 `import rl.*`），**不是** `src`。
# 2026-09-18 我第一版写成 `os.path.dirname(_PARENT)` ⇒ 得到 `src` ⇒ `<repo>/docs/train_<run>.log`
# 一个也推不出来、面板永远显示"没有日志"。这个 bug 是 `scripts/check_dashboard_js.py` 新加的
# **正面路径回归**抓出来的（它打印 `log=None` 而不是让 SKIP 悄悄过去）。
_REPO_ROOT = os.path.dirname(os.path.dirname(_PARENT))

from rl.replay import save_league_replays
from rl.config import eval_schedule
# 训练健康曲线（价值损失 / 策略熵）：**从训练日志读**，不碰 solo_state.json。
# 为什么单独一条路：这两个量只在 `[solo step N]` / `[step N]` 行里（**逐 update**，100k 约 781 点），
# 而 solo_state.json 的 history[] 是逐评估点（14 点）且**不含**这两项（2026-09-18 实测）。
from rl.train_health import derive_train_log, health_summary
from rl.io_bootstrap import force_utf8_stdout
from rl.dashboard_html import HTML as _HTML

# T2-2：数据面（常量 + 缓存 + 24 个纯函数）已整体搬到 `rl/dashboard_data.py`。
# 这里**显式 re-export** ⇒ 本文件与外部调用方（`rl/selftest.py`、`scripts/check_dashboard_js.py`）
# 的 `dash.<名>` / `from rl.dashboard import <名>` **一行未动**。
from rl.dashboard_data import (  # noqa: E402,F401
    MODEL_COLORS, MODEL_LABELS, CKPT_COLOR, FALLBACK_COLORS,
    _REPLAY_META_CACHE, load_state, resolve_state_path, _league_run_meta,
    _winrate_curves, build_payload, _SWEEP_PRIORITY, scan_sweep_dirs,
    build_sweep_payload, _HEALTH_CACHE, build_health_payload, build_solo_payload,
    _parse_replay_step, _replay_n_games, scan_replays, build_replays_payload,
    load_replay_payload, _CARD_STATS_CACHE, _agent_label, _other_side_id,
    _new_agent, _DECK_INDEX, _deck_key, _deck_index,
    _new_deck, _deck_bucket, _stat_file_cards, build_card_stats_payload,
)


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

# T2-1：`_HTML`（1879 行内联页面，80,974 字符）已抽到 `rl/dashboard_html.py`；
# 顶部用 `HTML as _HTML` 别名导入 ⇒ 本文件所有引用点**一行未动**。



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
            # 目录入口（`--state runs/<name>`）**按需重解析**：长跑常常是"先起 dashboard 再起训练"，
            # 启动时 league_state.json 还不存在（要等第一个评估点写完）。若只在启动时解析一次，
            # 联赛面板会被**永久关闭**（2026-09-18 实测）。文件入口不受影响。
            _sp = self.state_path
            if not _sp and getattr(Handler, "state_dir_lazy", False):
                _sp = resolve_state_path(Handler.state_arg)
                if _sp:
                    Handler.state_path = _sp
            self._send(200, json.dumps(build_payload(_sp)).encode("utf-8"),
                       "application/json; charset=utf-8")
        elif route == "/api/sweep":
            self._send(200, json.dumps(build_sweep_payload(self.sweep_root)).encode("utf-8"),
                       "application/json; charset=utf-8")
        elif route == "/api/solo":
            self._send(200, json.dumps(build_solo_payload(self.solo_path)).encode("utf-8"),
                       "application/json; charset=utf-8")
        elif route == "/api/health":
            # 训练健康（价值损失 / 策略熵）：读训练日志，与 solo_state.json 无关
            self._send(200, json.dumps(build_health_payload(Handler.train_log)).encode("utf-8"),
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


# T1-1：实现已收敛到 `rl/io_bootstrap.force_utf8_stdout`（**单一来源**）。本名保留为**别名**
# ⇒ 调用点（本文件 `main()`）无需改动。收敛前此处的函数体与 `rl/run_league.py` 的那一份
# **逐字相同**（仅 docstring 不同）⇒ 行为不变；由 `scripts/_structure_check.py` ② 复核。
#
# 历史背景（保留自查）：本文件此前**没有**这个兜底 ⇒ 2026-09-18 实测：把 dashboard 的输出
# 重定向到管道时，`main()` 里那句 `print(f"[dashboard] ⚠ ...")`（目录里还没有 league_state.json
# 时的告警）在 GBK locale 下抛 `UnicodeEncodeError: 'gbk' codec can't encode character '\u26a0'`
# ⇒ **整个 dashboard 直接崩掉（exit 1），服务起不来**。
# 这正是 `docs/agents/env.md` §2 记的陷阱：「新脚本一律自带 UTF-8 stdout reconfigure」。
_force_utf8_stdout = force_utf8_stdout


def main():
    _force_utf8_stdout()
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
    ap.add_argument("--train-log", type=str, default=None,
                    help="训练日志（供「训练健康」面板读价值损失/策略熵；缺省 = 从 --solo 目录名推 "
                         "<repo>/docs/train_<run>.log）。这两个量只在逐 update 的日志行里，"
                         "solo_state.json 没有。")
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--demo", action="store_true",
                    help="状态文件不存在时生成演示数据（Elo + flow-sweep + solo + 回放）")
    ap.add_argument("--demo-points", type=int, default=10)
    args = ap.parse_args()
    # --state 接受 JSON 文件或 runs/<name> 目录（目录里自动找 league_state.json）
    state_abs = resolve_state_path(args.state) if args.state else None
    if args.state and os.path.isdir(os.path.abspath(args.state)) and state_abs is None:
        print(f"[dashboard] ⚠ {args.state} 是目录但里面还没有 league_state.json；"
              f"**会持续按需重查**（run 模式写完第一个评估点后自动生效）", flush=True)
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
    Handler.state_arg = args.state
    Handler.state_dir_lazy = bool(args.state) and os.path.isdir(os.path.abspath(args.state))
    Handler.replays_dir = replays_abs
    Handler.sweep_root = sweep_abs
    Handler.solo_path = solo_abs
    # 训练健康面板（价值损失 / 策略熵）的日志：--train-log 优先，否则从 --solo 的目录名推
    # <repo>/docs/train_<run>.log。推不到就**明确报"没有"**，绝不拿别的 run 的日志顶上
    # （否则会把 A 臂的熵画在 B 臂的面板上，是比"没图"坏得多的错）。
    _log_arg = os.path.abspath(args.train_log) if args.train_log else None
    Handler.train_log = derive_train_log(
        solo_dir=(os.path.dirname(solo_abs) if solo_abs else None),
        repo_root=_REPO_ROOT, explicit=_log_arg)
    if _log_arg and not os.path.exists(_log_arg):
        print(f"[dashboard] 警告：--train-log 不存在：{_log_arg}")
    if Handler.train_log:
        print(f"[dashboard] 训练健康面板日志: {Handler.train_log}")
    else:
        print("[dashboard] 训练健康面板：未找到训练日志（可用 --train-log 显式指定）")
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
    _state_note = Handler.state_path or (
        f"{args.state}（目录入口·等待第一个评估点）" if Handler.state_dir_lazy
        else "未指定，联赛面板关闭")
    print(f"[dashboard] http://{args.host}:{args.port}  (state={_state_note})",
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
