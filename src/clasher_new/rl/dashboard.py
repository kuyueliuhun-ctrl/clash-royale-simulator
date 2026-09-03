"""训练网页 UI：各模型 Elo-训练次数 曲线仪表盘 + 最近训练回放/播放器。

- 读取 ``run_league --mode run`` 写出的联赛状态 JSON（含 elo_history）；
- 扫描 ``<state 同目录>/replays/league_<step>.pkl`` 联赛录像，列出最近回放；
- 浏览器内 Canvas 播放器回放单局（纯前端自绘，无外部 CDN 依赖，离线可用）；
- ``/api/state`` 每 3 秒轮询刷新，``/api/replays`` 每 5 秒刷新回放列表。

用法：
    python rl/dashboard.py --state league_state.json --port 8090
打开 http://127.0.0.1:8090 查看。仓库根目录另有 scripts/rl/dashboard.py 包装。
"""

import os
import re
import sys
import json
import pickle
import argparse
import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from rl.replay import save_league_replays

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


def build_payload(path):
    st = load_state(path)
    if st is None:
        return {"ok": False, "error": f"状态文件不存在: {path}", "state_path": path}
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
    return {
        "ok": True,
        "agents": agents,
        "elo_history": elo_history,
        "round_stats": st.get("round_stats", []),   # [{step, est:{aid:[R,SE]}, games:{aid:n}}]
        "total_steps": int(st.get("total_steps", 0)),
        "demo": bool(st.get("demo", False)),   # --demo 生成的合成数据标记
        "state_path": path,
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
    <h2>Elo 曲线（训练步数）</h2>
    <div class="legend" id="legend"></div>
    <div style="position:relative">
      <canvas id="chart"></canvas>
      <div id="tooltip"></div>
    </div>
  </div>
  <div class="card">
    <h2>当前排名</h2>
    <table>
      <thead><tr><th>模型</th><th>类型</th><th>Elo</th><th>Δ上轮</th><th>σ 信号/噪声</th><th>checkpoint</th></tr></thead>
      <tbody id="tbody"></tbody>
    </table>
    <p id="meta" class="sub" style="margin-top:12px"></p>
  </div>
</main>

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
let payload = {ok:false, agents:[], elo_history:{}, round_stats:[], total_steps:0};

async function refresh(){
  try{
    // 时间戳 query 防任何中间层/浏览器缓存，保证 3s 轮询拿到最新状态
    const r = await fetch("/api/state?_t=" + Date.now(), {cache: "no-store"});
    payload = await r.json();
    const src = document.getElementById("datasrc");
    if (!payload.ok){
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
    }
  }catch(e){
    document.getElementById("status").textContent = "连接失败：" + e;
  }
  render();
}

function renderLegend(){
  const el = document.getElementById("legend");
  el.innerHTML = payload.agents.map(a =>
    `<span><span class="dot" style="background:${colorOf(a.id)}"></span>${a.label}</span>`
  ).join("");
}

function renderTable(){
  const tbody = document.getElementById("tbody");
  const rs = payload.round_stats || [];
  const rPrev = rs.length >= 2 ? rs[rs.length-2] : null;
  const rCur = rs.length >= 1 ? rs[rs.length-1] : null;
  const seOf = (rt, aid) => rt && rt.est && rt.est[aid] ? rt.est[aid][1] : null;
  const rows = payload.agents.map(a => {
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
    return `<tr>
      <td><span class="dot" style="background:${colorOf(a.id)}"></span>${a.label}</td>
      <td><span class="kind ${kindCls}">${a.kind}</span></td>
      <td><b>${a.elo.toFixed(1)}</b></td>
      <td>${delta}</td>
      <td>${sigma}</td>
      <td style="color:#94a3b8;font-size:11px">${a.path ? a.path.split(/[\\/]/).pop() : "—"}</td>
    </tr>`;
  }).join("");
  tbody.innerHTML = rows || `<tr><td colspan="6">暂无模型（等待训练写入状态文件…）</td></tr>`;
  document.getElementById("meta").textContent =
    `总训练步数：${payload.total_steps} · 模型数：${payload.agents.length}` +
    (rCur ? ` · 最近评估 ${rCur.step} 步` : "");
}

function drawChart(){
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
  const maxStep = Math.max(100, ...xs, payload.total_steps);
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
    ctx.fillText(String(v), x-12, H-padB+16);
  }
  ctx.stroke();
  // 1500 基准线
  ctx.strokeStyle="#475569"; ctx.setLineDash([4,4]); ctx.beginPath();
  ctx.moveTo(padL, Y(1500)); ctx.lineTo(W-padR, Y(1500));
  ctx.stroke(); ctx.setLineDash([]);
  ctx.fillText("1500", W-padR-30, Y(1500)-4);

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

function render(){
  if (!payload.ok){ renderLegend(); renderTable(); return; }
  renderLegend(); renderTable(); drawChart();
}

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
}

function renderGamesPanel(){
  document.getElementById("gamesTitle").textContent = "对局列表 · " + curReplay.file;
  document.getElementById("gamesMeta").textContent =
    `${curReplay.games.length} 局 · 单击“播放”回放`;
  const body = document.getElementById("gamesBody");
  body.innerHTML = curReplay.games.map(g => {
    const pair = (g.pair || []).join(" vs ") || "?";
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
  const pair = (meta.pair || []).join(" vs ") || "?";
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
                hp: f[3] + (t[3] - f[3]) * p, player: t[4], alpha: 1, scale: 1});
    } else {
      out.push({name: t[0], x: t[1], y: t[2], hp: t[3], player: t[4], alpha: p, scale: p});
    }
  }
  for (let i = 0; i < fromEnts.length; i++){
    if (!used.has(i)){
      const f = fromEnts[i];
      out.push({name: f[0], x: f[1], y: f[2], hp: f[3], player: f[4], alpha: 1 - p, scale: 1 - p});
    }
  }
  return out;
}

function drawInterp(from, to, p){
  const canvas = document.getElementById("arena");
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
    const r = (tw.king ? 0.62 : 0.50) * scale;
    const col = tw.player === 0 ? "#3b82f6" : "#ef4444";
    ctx.beginPath();
    ctx.arc(X(tw.x), Y(tw.y), r, 0, 7);
    ctx.fillStyle = "rgba(15,23,42,0.55)";
    ctx.fill();
    ctx.strokeStyle = col;
    ctx.lineWidth = 2;
    ctx.stroke();
    if (frac > 0){
      ctx.beginPath();
      ctx.arc(X(tw.x), Y(tw.y), Math.max(1, r - 3), -Math.PI / 2,
              -Math.PI / 2 + frac * 2 * Math.PI);
      ctx.strokeStyle = frac > 0.5 ? "#22c55e" : (frac > 0.25 ? "#eab308" : "#ef4444");
      ctx.lineWidth = 3;
      ctx.stroke();
    } else {
      ctx.strokeStyle = "#475569";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(X(tw.x) - r / 2, Y(tw.y) - r / 2);
      ctx.lineTo(X(tw.x) + r / 2, Y(tw.y) + r / 2);
      ctx.moveTo(X(tw.x) + r / 2, Y(tw.y) - r / 2);
      ctx.lineTo(X(tw.x) - r / 2, Y(tw.y) + r / 2);
      ctx.stroke();
    }
  });

  // 实体（插值位置/血量，新兵淡入、阵亡淡出；塔由上方独立绘制，跳过塔实体）
  const ents = interpEntities(from && from.entities, to.entities, p)
    .filter(e => e.name !== "KingTower" && e.name !== "King_PrincessTowers");
  ents.forEach(e => {
    const r = Math.max(3.5, 0.34 * scale);
    const col = e.player === 0 ? "#60a5fa" : "#f87171";
    ctx.globalAlpha = Math.max(0, Math.min(1, e.alpha || 1));
    ctx.beginPath();
    ctx.arc(X(e.x), Y(e.y), r * (e.scale || 1), 0, 7);
    ctx.fillStyle = col;
    ctx.fill();
    ctx.strokeStyle = "rgba(255,255,255,0.7)";
    ctx.lineWidth = 1;
    ctx.stroke();
    const bw = r * 2, bh = 2.5;
    ctx.fillStyle = "rgba(0,0,0,0.5)";
    ctx.fillRect(X(e.x) - bw / 2, Y(e.y) - r - 6, bw, bh);
    ctx.fillStyle = "#22c55e";
    ctx.fillRect(X(e.x) - bw / 2, Y(e.y) - r - 6, bw * Math.max(0, Math.min(1, e.hp / 1000)), bh);
    ctx.fillStyle = "#e2e8f0";
    ctx.font = "9px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(e.name, X(e.x), Y(e.y) - r - 9);
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

/* ---- 事件绑定 + 启动 ---- */

document.getElementById("btnPlay").onclick = togglePlay;
document.getElementById("btnPrev").onclick = () => stepFrame(-1);
document.getElementById("btnNext").onclick = () => stepFrame(1);
document.getElementById("closePlayer").onclick = () => {
  stopPlay();
  document.getElementById("playerCard").style.display = "none";
};
document.getElementById("reloadReplays").onclick = refreshReplays;
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
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    state_path = "league_state.json"
    replays_dir = "replays"

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
        else:
            self._send(404, b"not found", "text/plain; charset=utf-8")

    def log_message(self, fmt, *args):
        sys.stderr.write("[dashboard] %s\n" % (fmt % args))


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


def main():
    ap = argparse.ArgumentParser(description="RL 联赛训练仪表盘（Elo 曲线 + 最近回放/播放器）")
    ap.add_argument("--state", type=str, default="league_state.json",
                    help="run_league 写出的联赛状态 JSON")
    ap.add_argument("--replays", type=str, default=None,
                    help="回放目录（缺省 = 状态文件同目录/replays）")
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--demo", action="store_true",
                    help="状态文件不存在时生成演示数据（5 模型合成 Elo 曲线 + 演示回放）")
    ap.add_argument("--demo-points", type=int, default=10)
    args = ap.parse_args()
    state_abs = os.path.abspath(args.state)
    if args.demo and not os.path.exists(state_abs):
        make_demo_state(state_abs, n_points=args.demo_points)
    replays_abs = args.replays and os.path.abspath(args.replays) \
        or os.path.join(os.path.dirname(state_abs), "replays")
    if args.demo:
        make_demo_replays(replays_abs)
    Handler.state_path = state_abs
    Handler.replays_dir = replays_abs
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[dashboard] http://{args.host}:{args.port}  (state={Handler.state_path})")
    print(f"[dashboard] 回放目录: {Handler.replays_dir}  Ctrl+C 退出")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
