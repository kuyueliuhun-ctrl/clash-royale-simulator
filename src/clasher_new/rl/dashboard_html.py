# -*- coding: utf-8 -*-
"""dashboard 的内联页面（Tier 2 · T2-1，从 `rl/dashboard.py` 抽出）。

**为什么抽出**：`_HTML` 是一个 **1879 行 / 80,974 字符**的单行级常量，占 `dashboard.py`
（3,174 行）的一半以上，却与后端逻辑毫无耦合 ⇒ 抽走后 `dashboard.py` 只剩逻辑。

**逐字保证**：本文件由 `ast.get_source_segment` 取**精确源片段**生成（只把左侧名字
`_HTML` 改成 `HTML`），右侧字符串字面量**一个字符未改**；`dashboard.py` 侧用
`from rl.dashboard_html import HTML as _HTML` 别名 ⇒ **原先所有引用点一行未动**。
值指纹见 `docs/structure_tier2_2026-09-19.md`。
"""

HTML = r"""<!DOCTYPE html>
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
      <optgroup label="训练健康（读训练日志 · 逐 update；⚠ 描述性读数，非预注册判据）">
        <option value="health">策略熵 · 价值损失(原始MSE)</option>
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
/* 训练健康（价值损失 / 策略熵）：来自 /api/health，数据源是**训练日志**（逐 update），
   与 solo.history（逐评估点）无关。ok=false 通常 = 没给 --train-log 且目录名推不出日志。 */
let health = {ok:false, points:[], read:{}, log:null};

async function refresh(){
  try{
    // 时间戳 query 防任何中间层/浏览器缓存，保证 3s 轮询拿到最新状态
    const [rs, sw, so, he] = await Promise.all([
      fetch("/api/state?_t=" + Date.now(), {cache: "no-store"}).then(r=>r.json()),
      fetch("/api/sweep?_t=" + Date.now(), {cache: "no-store"}).then(r=>r.json()),
      fetch("/api/solo?_t=" + Date.now(), {cache: "no-store"}).then(r=>r.json()),
      fetch("/api/health?_t=" + Date.now(), {cache: "no-store"}).then(r=>r.json())
    ]);
    payload = rs; sweep = sw; solo = so; health = he;
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
  health: {
    fromHealth: true,
    note: "训练健康：数据源 = 训练日志的逐 update 行（100k 约 781 点），不是评估点。"
        + "「策略熵」= 掩码后一次决策内各 decoder 步分布熵之**和**（nat）⇒ 与 bundle 长度有非零混杂；"
        + "平台 = 策略锐度不再变化，但⚠ ent_coef 固定时熵**本就**趋于平衡点 ⇒ 熵平台≠性能平台。"
        + "「价值损失」画**原始 MSE**（日志 vraw）：日志里的 value= 是 ÷v_scale² 的缩放量，"
        + "value_norm=running 时 scale 会长（A_et 实测 1.75→24.87）⇒ 只看 value= 会读到「损失在降」的假象。"
        + "⚠ 全部为描述性读数，非预注册判据，不得并入 §11.13 判据集。",
    series: [
      {key: "entropy", label: "策略熵（nat）", color: "#22c55e", pct: false},
      {key: "value_loss_raw", label: "价值损失(原始MSE)", color: "#38bdf8", pct: false},
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
  const hist = solo.history || [];
  const ctrl = solo.controls_history || [];
  const hpts = (health && health.points) || [];
  const _READ_ZH = {plateau: "平台", moving_down: "仍在下降", moving_up: "仍在上升",
                    unresolvable: "不可判"};
  if (noteEl){
    let t = spec.note || "";
    if (spec.fromHealth){
      if (health && health.log) t += " ｜ 日志：" + String(health.log).split(/[\\/]/).pop();
      const parts = spec.series.map(s => {
        const r = health && health.read && health.read[s.key];
        if (!r) return null;
        const zh = _READ_ZH[r.verdict] || r.verdict;
        const win = r.tail ? `末窗 step ${r.tail.step_lo}-${r.tail.step_hi}` : "末窗";
        const base = r.base ? `前窗中位 ${_fmtVal(r.base.median)}` : "";
        const thr = _fmtVal(3 * (r.spread || 0));
        const snr = (r.snr === null || r.snr === undefined) ? "n/a" : r.snr.toFixed(2);
        return `${s.label}：${zh}（${win} 中位 ${_fmtVal(r.tail ? r.tail.median : null)} vs ${base}，`
             + `diff=${_fmtVal(r.diff)}，阈值 3×MAD=${thr}，snr=${snr}）`;
      }).filter(Boolean);
      if (parts.length) t += " ｜ 局部平台读数（描述性）：" + parts.join("；");
      else t += " ｜ 尚无读数";
    }
    noteEl.textContent = t;
  }
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth || 600, H = canvas.clientHeight || 300;
  canvas.width = W * dpr; canvas.height = H * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, W, H);
  const padL = 54, padR = 16, padT = 30, padB = 34;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const legendEl = document.getElementById("soloMetricLegend");
  ctx.fillStyle = "#94a3b8"; ctx.font = "12px sans-serif";
  const _has = spec.fromHealth ? hpts.length : (spec.fromControls ? ctrl.length : hist.length);
  if (!_has){
    if (spec.fromHealth){
      // 三种"没数据"要分开报，否则用户会把"没接上日志"误读成"这一项没测"
      ctx.fillText(health && health.ok === false
        ? ("训练健康不可用：" + (health.error || "无日志"))
        : "等待训练日志出现可解析的 step 行…", padL, padT + 20);
    } else {
      ctx.fillText("等待首次评估…", padL, padT + 20);
    }
    if (legendEl) legendEl.innerHTML = "";
    return;
  }
  const lines = spec.series.map(s => {
    let pts;
    if (spec.fromHealth){
      pts = hpts.filter(r => r[s.key] !== undefined && r[s.key] !== null)
                .map(r => [r.step, r[s.key], null]);
    } else if (spec.fromControls){
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
