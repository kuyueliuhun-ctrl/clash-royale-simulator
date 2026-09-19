# -*- coding: utf-8 -*-
"""前端冒烟回归：dashboard.py 内嵌 JS 的**真实运行级**检查（不只是语法）。

背景：`rl/dashboard.py` 把整个 UI 写在一个 `<script>` 大字符串里（约 55 KB JS），
仓库里没有任何测试覆盖它 ⇒ 改了前端只能靠"打开浏览器看"。本脚本补上这条：
用 node + 极简 DOM 桩，把**真实后端 payload** 喂给渲染函数，捕获运行时异常。

三层检查：
  1. `node --check`：语法（抓括号/引号类错误）；
  2. 渲染冒烟：以真实 payload 调 `renderSolo()` / `renderCardStats()`，
     并遍历 `SOLO_METRICS` 的**每一个指标**调 `drawSoloMetricChart()`；
  3. 分支覆盖：旧录像（无 `meta.decks`）走"按卡组"空表分支。

**2026-09-19 新增第 4 层 · 塔几何**（用户：「引擎已经正常实现塔了，但 dashboard 回放的塔
还是 1 格」）：前端画的塔必须与引擎 `arena.TileGrid.towers` 同尺寸（公主塔 3×3 / 国王塔 4×4）。
  - Python 侧**三方对账**：前端写死的 `TOWER_GEOM` / `towerPos` / 竞技场 18×32
    ↔ 引擎 `arena.TileGrid` ↔ `replay.tower_geometry()`；
  - node 侧**真绘制路径**：`computeTowerMax()` → `drawInterpOn()`，记录 `strokeRect` 的
    **实参像素**，覆盖 老录像回落 / 本仓几何 / 上游 3 元组几何 / 陈旧几何 / 残缺项 / 塔心不漂。
  只跑这层：`--tower-only`（不需要 run 目录）。

用法：
    # 用真实 run 的落盘数据（推荐）
    cd src/clasher_new && PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe \
        ../../scripts/check_dashboard_js.py --run runs/nostall20k
    # 只做语法检查（不需要 node 之外的东西）
    ... ../../scripts/check_dashboard_js.py --syntax-only

退出码：0 = 全过；1 = 有失败项；2 = 只跑了语法（找不到 node）。
依赖 `node`（本机实测 v24.9.0）；**若 Windows python 的 PATH 里没有 node**，自动回退 `wsl.exe node`
（本机 WSL 有 Linux 版 node），脚本经 **stdin 管道**喂给 node ⇒ 不涉及 Windows↔WSL 路径转换。
**只读**：不改 `dashboard.py`；证据 JSON 写在 `src/clasher_new/runs/_tmp_dashjs/`（runs/ 已在 .gitignore）。
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src", "clasher_new")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from rl.io_bootstrap import force_utf8_stdout  # noqa: E402  (T1-1: UTF-8 兜底单一来源)
force_utf8_stdout()

_DASH = os.path.join(_SRC, "rl", "dashboard.py")

_STUB = r"""
const ctxStub = new Proxy({}, {
  get(t, k){ return (k in t) ? t[k] : function(){ return {width: 8}; }; },
  set(t, k, v){ t[k] = v; return true; },
});
const _els = {};
function mkEl(id){
  return { id, style:{}, innerHTML:"", textContent:"", value:"", className:"",
    selectedOptions: [{textContent: "stub-option"}], clientWidth:600, clientHeight:300,
    width:0, height:0, scrollTop:0, scrollHeight:0, dataset:{},
    classList:{add(){},remove(){},toggle(){}},
    addEventListener(){}, appendChild(){}, removeChild(){}, focus(){}, click(){},
    getContext(){ return ctxStub; },
    getBoundingClientRect(){ return {left:0,top:0,width:600,height:300}; } };
}
global.document = { getElementById(id){ return _els[id] || (_els[id] = mkEl(id)); },
                    querySelector(){ return mkEl("q"); }, createElement(){ return mkEl("c"); },
                    addEventListener(){} };
global.window = { devicePixelRatio: 1, addEventListener(){} };
global.fetch = function(){ return new Promise(function(){}); };   // 挂住轮询，避免覆盖我们注入的 payload
global.setInterval = function(){ return 0; };
global.clearInterval = function(){};
global.requestAnimationFrame = function(){ return 0; };
global.cancelAnimationFrame = function(){};
"""

_TAIL = r"""
const fs = require("fs");
const B = %(base)r;
const soloPayload = JSON.parse(fs.readFileSync(B + "_solo.json", "utf8"));
const csPayload = JSON.parse(fs.readFileSync(B + "_cs.json", "utf8"));
const csOld = JSON.parse(fs.readFileSync(B + "_cs_old.json", "utf8"));
const out = []; let fail = 0;
function t(name, fn){ try { fn(); out.push("PASS " + name); } catch(e){ fail++; out.push("FAIL " + name + " :: " + e.message); } }

t("renderSolo(真实 payload)", function(){ solo = soloPayload; renderSolo(); });
t("solo 表格已渲染", function(){
  const h = document.getElementById("soloTable").innerHTML;
  if (!h || h.indexOf("W /") < 0) throw new Error("表格为空: " + h.slice(0,80));
});
t("默认指标 = behavior（不再以胜率为默认曲线）", function(){
  if (curSoloMetric !== "behavior") throw new Error(curSoloMetric);
});
t("全部指标可绘制（逐项遍历 SOLO_METRICS）", function(){
  const bad = [];
  Object.keys(SOLO_METRICS).forEach(function(k){
    curSoloMetric = k;
    try { drawSoloMetricChart(); } catch(e){ bad.push(k + ":" + e.message); }
  });
  curSoloMetric = "behavior"; drawSoloMetricChart();
  if (bad.length) throw new Error(bad.join(" | "));
});
t("指标说明文字已写入", function(){
  const n = document.getElementById("soloMetricNote").textContent;
  if (!n || n.length < 10) throw new Error("说明为空");
});
t("图例已渲染", function(){
  const l = document.getElementById("soloMetricLegend").innerHTML;
  if (l.indexOf("dot") < 0) throw new Error("图例为空: " + l.slice(0,80));
});
t("controls 曲线 = 3 条外生对照", function(){
  curSoloMetric = "controls"; drawSoloMetricChart();
  const l = document.getElementById("soloMetricLegend").innerHTML;
  const n = (l.match(/<span><span class="dot"/g) || []).length;
  if (n !== %(n_controls)d) throw new Error("图例条数=" + n + "（期望 %(n_controls)d）");
  curSoloMetric = "behavior"; drawSoloMetricChart();
});
t("winrate 曲线（含 SE 竖线路径）可绘制", function(){
  curSoloMetric = "winrate"; drawSoloMetricChart(); curSoloMetric = "behavior"; drawSoloMetricChart();
});
/* 训练健康面板（策略熵 / 价值损失）：数据源是训练日志，不是 solo_state.json。
   2026-09-18 新增 —— 【R8】新能力必须带**正面路径**回归，不能只测"没数据不抛"。 */
if (_HEALTH_PAYLOAD && _HEALTH_PAYLOAD.ok){
  t("health 正面路径：真实日志 payload 可绘制 + 读数写进 note", function(){
    health = _HEALTH_PAYLOAD;
    curSoloMetric = "health"; drawSoloMetricChart();
    const leg = document.getElementById("soloMetricLegend").innerHTML;
    if (leg.indexOf("策略熵") < 0) throw new Error("图例缺策略熵: " + leg.slice(0,120));
    const note = document.getElementById("soloMetricNote").textContent;
    if (note.indexOf("平台读数") < 0) throw new Error("note 里没有平台读数: " + note.slice(0,200));
    if (note.indexOf("非预注册判据") < 0) throw new Error("note 缺『非预注册判据』声明");
    curSoloMetric = "behavior"; drawSoloMetricChart();
  });
} else {
  out.push("SKIP health 正面路径（该 run 未提供训练日志）");
}
t("health 无日志 ⇒ 走『不可用』分支（不抛）", function(){
  const keep = health;
  health = {ok: false, error: "未找到训练日志", points: [], read: {}};
  curSoloMetric = "health"; drawSoloMetricChart();
  curSoloMetric = "behavior"; drawSoloMetricChart();
  health = keep;
});
t("renderCardStats（按模型矩阵）", function(){ cardStats = csPayload; renderCardStats(); });
t("按卡组矩阵已渲染", function(){
  const h = document.getElementById("statsDeckTable").innerHTML;
  if (!h || h.indexOf("<thead>") < 0) throw new Error("卡组表为空");
  if (h.indexOf("边 ·") < 0) throw new Error("缺 side_games 表头: " + h.slice(0,160));
});
t("卡组表含卡牌行", function(){
  const h = document.getElementById("statsDeckTable").innerHTML;
  if (h.indexOf("card-name") < 0) throw new Error("无卡牌行");
});
t("旧录像(无 meta.decks)走空表分支", function(){
  cardStats = csOld; renderCardStats();
  const h = document.getElementById("statsDeckTable").innerHTML;
  if (h !== "") throw new Error("应为空表，实际: " + h.slice(0,80));
  const sub = document.getElementById("statsDeckSub").textContent;
  if (sub.indexOf("旧录像") < 0) throw new Error("缺旧录像提示: " + sub);
});
t("卡组元数据覆盖度写入 note", function(){
  cardStats = csOld; renderCardStats();
  const n = document.getElementById("statsNote").textContent;
  if (n.indexOf("meta.decks") < 0) throw new Error("note=" + n);
});
console.log(out.join("\n"));
console.log("FAILURES=" + fail);
process.exit(fail ? 1 : 0);
"""

#: 联赛/长跑面板（run 模式）：进度条 + 两级评估计划 + 逐点对手胜率 + 大点竖虚线。
#: 2026-09-18 起 `--state` 传 `runs/<name>` 目录即可，供 1M 步长跑 3 s 轮询。
_LEAGUE_TAIL = r"""
const out2 = []; let fail2 = 0;
function t2(name, fn){ try { fn(); out2.push("PASS " + name); } catch(e){
  fail2++; out2.push("FAIL " + name + " :: " + (e && e.stack
    ? String(e.stack).split("\n").slice(0, 4).join(" | ") : e.message)); } }
const LP = %(league)s;
t2("renderTable(联赛 payload)：主体 + 排名表", function(){
  payload = LP; renderTable(); renderLegend();
  const h = document.getElementById("tbody").innerHTML;
  if (h.indexOf("<tr>") < 0) throw new Error("无数据行: " + h.slice(0,120));
  const nRows = (h.match(/<tr>/g) || []).length;
  const nTds = (h.match(/<td/g) || []).length;
  if (nTds !== nRows * 7) throw new Error("列数错: tds=" + nTds + " rows=" + nRows);
  if (document.getElementById("legend").innerHTML.indexOf("dot") < 0) throw new Error("图例为空");
});
t2("进度条：显示 当前/总计划步数与评估点进度", function(){
  renderRunMeta();
  const h = document.getElementById("leagueProgress").innerHTML;
  const rm = LP.run_meta || {};
  if (!h) throw new Error("进度条为空");
  if (rm.total_steps && h.indexOf("评估点") < 0) throw new Error("缺评估点进度");
  const hNum = h.replace(/,/g, "");   // 进度条用 toLocaleString（千分位）
  if (rm.total_steps && hNum.indexOf("评估点") < 0) throw new Error("缺评估点进度");
  if (rm.total_steps && hNum.indexOf(String(rm.cur_step)) < 0) throw new Error("缺当前步数");
  if (rm.total_steps && hNum.indexOf(String(rm.total_steps)) < 0) throw new Error("缺总计划步数");
  const sub = document.getElementById("leagueSub").textContent;
  if (sub.indexOf("两级评估") < 0) throw new Error("缺两级评估说明: " + sub);
});
t2("缺 run_state 的旧 run：进度条留空、不崩", function(){
  const keep = payload; payload = {ok:true, agents:[], elo_history:{}, round_stats:[],
                                   run_meta:{total_steps:0}};
  renderRunMeta(); renderTable();
  if (document.getElementById("leagueProgress").innerHTML !== "") throw new Error("应留空");
  payload = keep;
});
t2("Elo 曲线（含大评估点竖虚线路径）", function(){
  payload = LP; document.getElementById("leagueMetric").value = "elo"; drawChart();
});
t2("胜率曲线：退出/恢复切换 + 有对手曲线", function(){
  const sel = document.getElementById("leagueMetric");
  sel.value = "winrate"; drawChart();
  const keys = Object.keys(LP.winrate_curves || {}).filter(k => k.indexOf("main|") === 0);
  if (!keys.length) throw new Error("payload 里没有 main|* 曲线");
  sel.value = "elo"; drawChart();
});
t2("winrate 空 payload 走空分支（不抛）", function(){
  const keep = payload;
  payload = {ok:true, agents:[], elo_history:{}, round_stats:[], winrate_curves:{},
             winrate_counts:{}, winrates:{}, run_meta:{total_steps:0}};
  document.getElementById("leagueMetric").value = "winrate";
  drawChart();
  document.getElementById("leagueMetric").value = "elo";
  drawChart();
  payload = keep;
});
console.log(out2.join("\n"));
console.log("FAILURES=" + fail2);
process.exit(fail2 ? 1 : 0);
"""


#: 塔几何（2026-09-19，用户：「引擎已经正常实现塔了，但 dashboard 回放的塔还是 1 格」）：
#: 前端画的塔**必须**与引擎 `arena.TileGrid.towers` 同尺寸（公主塔 3×3 / 国王塔 4×4）。
#: 度量方式 = 走**真绘制路径** `drawInterpOn`，记录 `strokeRect` 的**实参像素**（不是读常量）
#: ⇒ 常量改对了但绘制仍用旧变量这类错误也会被抓到。
#: 三条覆盖：① 老录像无 `meta.tower_geom` ⇒ 回落常量；② 本仓 4 元组几何；③ 上游 3 元组几何
#: （公主塔 r=1.0 / 王塔 r=1.4 ⇒ 2 格 / 2.8 格，**按上游自己的尺寸画**，不冒充本仓几何）。
_TOWER_TAIL = r"""
const outT = []; let failT = 0;
function tT(name, fn){ try { fn(); outT.push("PASS " + name); } catch(e){
  failT++; outT.push("FAIL " + name + " :: " + (e && e.message)); } }

const RECTS = [];
ctxStub.strokeRect = function(x, y, w, h){ RECTS.push([x, y, w, h]); };
const CANVAS_W = 450, CANVAS_H = 800;          // 与页面里 <canvas id="arena"> 同尺寸
const SCALE = Math.min(CANVAS_W / 18, CANVAS_H / 32);   // = 25 px/格
function pxOf(half){ return Math.round(half * 2 * SCALE); }

function mkTowerGame(towerGeom){
  const g = { file:"t.pkl", index:0, winner:0, meta:{}, frames:[
    { t:0, bundle:[], reward:0, opp_played:[], towers0:[4824,3052,3052],
      towers1:[4824,3052,3052], elixir0:5, elixir1:5, crown0:0, crown1:0, entities:[] }] };
  if (towerGeom) g.meta.tower_geom = towerGeom;
  return g;
}
/* 走真路径画一帧，返回 6 个塔的 strokeRect 实参（按半宽排序便于比对） */
function drawnRects(game){
  curGame = game; computeTowerMax();
  const cv = document.getElementById("arena");
  cv.width = CANVAS_W; cv.height = CANVAS_H;
  RECTS.length = 0;
  drawInterpOn(cv, game.frames[0], game.frames[0], 1);
  return RECTS.map(r => ({x:r[0], y:r[1], w:r[2], h:r[3]})).sort((a,b) => a.w - b.w || a.y - b.y);
}
function sideCounts(rects){
  const c = {}; rects.forEach(r => { const k = Math.round(r.w) + "x" + Math.round(r.h); c[k] = (c[k]||0)+1; });
  return c;
}
const OUR_GEOM = [[3.5,6.5,1.5,1.5,0],[14.5,6.5,1.5,1.5,0],[9,3,2,2,0],
                  [3.5,25.5,1.5,1.5,1],[14.5,25.5,1.5,1.5,1],[9,29,2,2,1]];
const UP_GEOM  = [[3.5,6.5,1,1,0],[14.5,6.5,1,1,0],[9,3,1.4,1.4,0],
                  [3.5,25.5,1,1,1],[14.5,25.5,1,1,1],[9,29,1.4,1.4,1]];

tT("① 老录像（无 meta.tower_geom）⇒ 回落引擎几何：公主塔 3 格 / 王塔 4 格", function(){
  const c = sideCounts(drawnRects(mkTowerGame(null)));
  const p = pxOf(1.5), k = pxOf(2.0);
  if (c[p+"x"+p] !== 4) throw new Error("公主塔应有 4 座 " + p + "px，实测 " + JSON.stringify(c));
  if (c[k+"x"+k] !== 2) throw new Error("王塔应有 2 座 " + k + "px，实测 " + JSON.stringify(c));
});
tT("② meta.tower_geom（本仓 4 元组）⇒ 逐塔按引擎几何", function(){
  const c = sideCounts(drawnRects(mkTowerGame(OUR_GEOM)));
  const p = pxOf(1.5), k = pxOf(2.0);
  if (c[p+"x"+p] !== 4 || c[k+"x"+k] !== 2) throw new Error(JSON.stringify(c));
});
tT("③ meta.tower_geom（上游 3 元组 r=1.0/1.4）⇒ 按上游尺寸 2 格 / 2.8 格", function(){
  const c = sideCounts(drawnRects(mkTowerGame(UP_GEOM)));
  const p = pxOf(1.0), k = pxOf(1.4);
  if (c[p+"x"+p] !== 4) throw new Error("上游公主塔应有 4 座 " + p + "px，实测 " + JSON.stringify(c));
  if (c[k+"x"+k] !== 2) throw new Error("上游王塔应有 2 座 " + k + "px，实测 " + JSON.stringify(c));
});
tT("④ 几何按**位置**命中（顺序打乱无影响）", function(){
  const shuffled = OUR_GEOM.slice().reverse();
  const c = sideCounts(drawnRects(mkTowerGame(shuffled)));
  const p = pxOf(1.5), k = pxOf(2.0);
  if (c[p+"x"+p] !== 4 || c[k+"x"+k] !== 2) throw new Error(JSON.stringify(c));
});
tT("⑤ 几何与本局塔位不匹配（陈旧/异图 meta）⇒ 该塔回落常量，不崩", function(){
  const stale = [[3.5,11.5,1.5,1.5,0],[14.5,11.5,1.5,1.5,0],[9,8,2,2,0],
                 [3.5,20.5,1.5,1.5,1],[14.5,20.5,1.5,1.5,1],[9,23,2,2,1]];
  const c = sideCounts(drawnRects(mkTowerGame(stale)));
  const p = pxOf(1.5), k = pxOf(2.0);
  if (c[p+"x"+p] !== 4 || c[k+"x"+k] !== 2) throw new Error("应全部回落常量: " + JSON.stringify(c));
});
tT("⑥ 残缺几何项（长度 < 4）被忽略而不是画错", function(){
  const mixed = [[3.5,6.5,1.5],[14.5,6.5,1.5,1.5,0],[9,3,2,2,0],
                 [3.5,25.5,1.5,1.5,1],[14.5,25.5,1.5,1.5,1],[9,29,2,2,1]];
  const c = sideCounts(drawnRects(mkTowerGame(mixed)));
  const p = pxOf(1.5), k = pxOf(2.0);
  if (c[p+"x"+p] !== 4 || c[k+"x"+k] !== 2) throw new Error(JSON.stringify(c));
});
tT("⑦ 塔心不动（放大后仍以引擎塔位为中心）", function(){
  const rects = drawnRects(mkTowerGame(null));
  const king = rects.filter(r => Math.round(r.w) === pxOf(2.0));
  const ox = (CANVAS_W - 18 * SCALE) / 2, oy = (CANVAS_H - 32 * SCALE) / 2;
  const want = [[9, 3], [9, 29]].map(p => (ox + p[0] * SCALE) + "," + (oy + p[1] * SCALE)).sort();
  const got = king.map(r => (r.x + r.w / 2) + "," + (r.y + r.h / 2)).sort();
  if (JSON.stringify(got) !== JSON.stringify(want)) throw new Error("塔心漂移: " + got + " != " + want);
});
console.log(outT.join("\n"));
console.log("FAILURES=" + failT);
process.exit(failT ? 1 : 0);
"""


def extract_js():
    """抽出内嵌页面的 `<script>` 块内容。

    ⚠️ **2026-09-19（Tier 2 · T2-1）**：页面常量 `_HTML` 已从 `rl/dashboard.py` 抽到
    `rl/dashboard_html.py` ⇒ 本函数原先"只读 `dashboard.py` 源码文本"的做法会**直接失效**
    （实测报 `[FAIL] dashboard.py 里找不到 <script> 块`）。现按**优先级**取：

      1. `rl/dashboard_html.py`（当前位置）；
      2. `rl/dashboard.py`（抽走前的历史位置 —— 保留，向后兼容）；
      3. 都找不到时，**退化为 import** `rl.dashboard._HTML`（最权威，但要拉起 torch）。

    抽走前后本函数返回的字符串**逐字相同**（T2-1 的 sha256 对账已证 `_HTML` 值不变）。
    """
    for path in (os.path.join(_SRC, "rl", "dashboard_html.py"), _DASH):
        if not os.path.isfile(path):
            continue
        with io.open(path, encoding="utf-8") as f:
            src = f.read()
        m = re.search(r"<script>(.*?)</script>", src, re.S)
        if m:
            return m.group(1)
    # 退化路径：直接取常量（权威但重）
    try:
        sys.path.insert(0, _SRC)
        from rl.dashboard import _HTML as _html  # noqa
        m = re.search(r"<script>(.*?)</script>", _html, re.S)
        if m:
            return m.group(1)
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"[FAIL] 既读不到 dashboard_html.py / dashboard.py 的 <script>，"
                         f"也 import 不到 rl.dashboard._HTML：{e!r}")
    raise SystemExit("[FAIL] dashboard_html.py / dashboard.py / rl.dashboard._HTML 里都找不到 <script> 块")


def _extract_js_consts(js):
    """从 JS 文本里取前端**写死**的三组引擎几何常量（取不到 ⇒ 抛，绝不静默跳过）。

    返回 `(TOWER_GEOM, (arena_w, arena_h), [塔中心...])`。
    """
    m = re.search(r"const TOWER_GEOM = \{(.*?)\n\};", js, re.S)
    if not m:
        raise AssertionError("JS 里找不到 `const TOWER_GEOM = {...};`（前端塔几何常量）")
    body = m.group(1)
    geom = {}
    for kind in ("princess", "king"):
        mm = re.search(kind + r"\s*:\s*\{\s*hw:\s*([0-9.]+)\s*,\s*hh:\s*([0-9.]+)\s*\}", body)
        if not mm:
            raise AssertionError("TOWER_GEOM 里找不到 %s 的 hw/hh" % kind)
        geom[kind] = (float(mm.group(1)), float(mm.group(2)))
    m2 = re.search(r"Math\.min\(W\s*/\s*([0-9.]+)\s*,\s*H\s*/\s*([0-9.]+)\)", js)
    if not m2:
        raise AssertionError("JS 里找不到竞技场尺寸 `Math.min(W / <宽>, H / <高>)`")
    m3 = re.findall(r"\b(p[01][klr]):\{x:([0-9.]+),y:([0-9.]+)\}", js)
    if len(m3) != 6:
        raise AssertionError("JS 里 towerPos 默认位应 6 条，实得 %d 条" % len(m3))
    return (geom, (int(float(m2.group(1))), int(float(m2.group(2)))),
            sorted((float(x), float(y)) for _k, x, y in m3))


def _tower_geom_checks(js):
    """**三方对账**：前端写死的塔几何 / 塔位 / 场尺寸 ↔ 引擎 `arena.TileGrid` ↔ `tower_geometry()`。

    动机 = 本次 bug 的根因：同一份几何前后端各写一份，而前端那份**不随引擎更新**
    （引擎 2026-09-09 把塔从圆形改成 3×3/4×4 矩形，前端一直画 1 格见方）。
    本函数把那条边做成**可执行的**：引擎塔尺寸/塔位一变而前端没同步 ⇒ FAIL。
    """
    from arena import TileGrid as TG
    from rl.replay import tower_geometry

    fails = []

    def chk(name, ok, detail=""):
        print(("[dashboard-js] PASS " if ok else "[dashboard-js] FAIL ") + name
              + ("" if ok else " :: %s" % (detail,)))
        if not ok:
            fails.append(name)

    princess, king, centers = set(), set(), []
    for pos, hw, hh, _pid in TG.towers:
        is_king = (pos.x, pos.y) in ((TG.BLUE_KING_TOWER.x, TG.BLUE_KING_TOWER.y),
                                     (TG.RED_KING_TOWER.x, TG.RED_KING_TOWER.y))
        (king if is_king else princess).add((hw, hh))
        centers.append((pos.x, pos.y))
    chk("引擎塔几何 = 公主塔 3×3 / 国王塔 4×4",
        princess == {(1.5, 1.5)} and king == {(2.0, 2.0)},
        "princess=%s king=%s" % (sorted(princess), sorted(king)))

    try:
        js_geom, js_arena, js_centers = _extract_js_consts(js)
    except AssertionError as e:
        chk("前端几何常量可提取", False, e)
        return 1
    chk("前端 TOWER_GEOM == 引擎 arena.TileGrid.towers",
        js_geom == {"princess": (1.5, 1.5), "king": (2.0, 2.0)},
        "js=%s 引擎 princess=%s king=%s" % (js_geom, sorted(princess), sorted(king)))
    chk("前端竞技场尺寸 == 引擎 TileGrid.width/height",
        js_arena == (TG.width, TG.height), "js=%s 引擎=%s" % (js_arena, (TG.width, TG.height)))
    chk("前端塔位缺省 == 引擎塔中心", js_centers == sorted(centers),
        "js=%s 引擎=%s" % (js_centers, sorted(centers)))

    class _Arena:
        def __init__(self, towers):
            self.towers = towers

    class _Battle:
        def __init__(self, towers):
            self.arena = _Arena(towers)

    fb = _Battle(None)
    fb.arena = TG()                     # 真 TileGrid ⇒ 对账的是**真几何**，不是手抄常量
    got = tower_geometry(fb)
    chk("tower_geometry(真 TileGrid) = 6 座且半宽同引擎",
        got is not None and len(got) == 6
        and {(round(g[2], 3), round(g[3], 3)) for g in got} == {(1.5, 1.5), (2.0, 2.0)},
        got)

    from core import Position
    up = tower_geometry(_Battle([(Position(3.5, 25.5), 1.0, 1), (Position(9.0, 29.0), 1.4, 1)]))
    chk("tower_geometry 认上游 3 元组（圆形 r ⇒ hw=hh=r）",
        up == [[3.5, 25.5, 1.0, 1.0, 1], [9.0, 29.0, 1.4, 1.4, 1]], up)
    chk("tower_geometry 形状不认识/空 ⇒ None（不抛）",
        tower_geometry(_Battle([(Position(0.0, 0.0), 1, 2, 3, 4)])) is None
        and tower_geometry(_Battle([])) is None and tower_geometry(None) is None, "")
    return 1 if fails else 0


def _node_runner():
    """返回 (runner_argv, 说明)。Windows python 的 PATH 里常没有 node ⇒ 回退 wsl.exe node。"""
    exe = shutil.which("node")
    if exe:
        return [exe, "-"], f"node - ({exe})"
    try:
        r = subprocess.run(["wsl.exe", "node", "--version"], capture_output=True,
                           text=True, timeout=30)
        if r.returncode == 0:
            return ["wsl.exe", "node", "-"], f"wsl.exe node - ({r.stdout.strip()})"
    except Exception:
        pass
    return None, "无"


def _run_node(runner, code, timeout=180):
    """把脚本经 stdin 管道喂给 node（避开 Windows↔WSL 路径转换）。"""
    r = subprocess.run(runner, input=code, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=timeout)
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/nostall20k",
                    help="run 目录（相对 src/clasher_new）；取其 solo_state.json 与 replays/")
    ap.add_argument("--old-run", default="runs/economy_9j",
                    help="一个**旧录像** run（无 meta.decks），用于测空表分支")
    ap.add_argument("--league-run", default=None,
                    help="run 模式训练目录（含 league_state.json，可给 runs/<name>）："
                         "额外跑联赛/长跑面板渲染冒烟（进度条 + 两级评估点 + 对手胜率曲线）")
    ap.add_argument("--syntax-only", action="store_true")
    ap.add_argument("--tower-only", action="store_true",
                    help="只跑塔几何检查（三方对账 + 真绘制路径实宽）；不需要 run 目录/payload")
    ap.add_argument("--keep", action="store_true", help="保留证据 JSON（排查用）")
    a = ap.parse_args()

    js = extract_js()
    runner, how = _node_runner()
    print(f"[dashboard-js] JS {len(js)} 字符；node 运行器 = {how}")
    if runner is None:
        print("[SKIP] 找不到 node（也不在 WSL 里）⇒ 只做提取，不做运行级检查")
        return 2
    if a.syntax_only:
        r = _run_node(runner, _STUB + js + '\nconsole.log("SYNTAX_OK");\n')
        ok = (r.returncode == 0) and ("SYNTAX_OK" in r.stdout)
        print("[dashboard-js] 语法 " + ("OK" if ok else "FAIL"))
        if not ok:
            print((r.stdout or "")[-800:]); print((r.stderr or "")[-800:])
        return 0 if ok else 1

    # —— 塔几何（2026-09-19，见 `_TOWER_TAIL` / `_tower_geom_checks` 的动机注释）——
    #    不依赖任何 run 目录：Python 侧做引擎↔前端常量三方对账，node 侧量真绘制路径的像素。
    rc_tower = _tower_geom_checks(js)
    print("[dashboard-js] 塔几何绘制回归（node + DOM 桩 + 记录 strokeRect 实参）…")
    rt = _run_node(runner, _STUB + js + _TOWER_TAIL)
    outt = (rt.stdout or "").strip()
    print(outt if outt else "(塔几何无 stdout)")
    if rt.returncode != 0:
        rc_tower = 1
        if (rt.stderr or "").strip():
            print("--- stderr (tower) ---")
            print((rt.stderr or "")[-1500:])
    if a.tower_only:
        return rc_tower

    from rl import dashboard as D  # 真实后端 payload

    run_dir = a.run if os.path.isabs(a.run) else os.path.join(_SRC, a.run)
    old_dir = a.old_run if os.path.isabs(a.old_run) else os.path.join(_SRC, a.old_run)
    solo_path = os.path.join(run_dir, "solo_state.json")
    if not os.path.isfile(solo_path):
        print(f"[FAIL] 找不到 {solo_path}")
        return 1
    solo_pl = D.build_solo_payload(solo_path)
    cs_pl = D.build_card_stats_payload(os.path.join(run_dir, "replays"), None, 2)
    old_replays = os.path.join(old_dir, "replays")
    cs_old = (D.build_card_stats_payload(old_replays, None, 1)
              if os.path.isdir(old_replays) else {"ok": False, "error": "no old replays"})
    n_controls = len({r.get("vs") for r in (solo_pl.get("controls_history") or [])})
    # 训练健康（策略熵/价值损失）：从**训练日志**取真实 payload。【R8】给新面板配正面路径回归：
    # 用后端真正的 build_health_payload（同一实现），而不是前端造一份假数据。
    h_log = D.derive_train_log(solo_dir=run_dir, repo_root=D._REPO_ROOT)
    health_pl = (D.build_health_payload(h_log) if h_log
                 else {"ok": False, "error": "未找到训练日志", "log": None})
    print(f"[dashboard-js] payload：history={len(solo_pl.get('history') or [])} "
          f"controls={len(solo_pl.get('controls_history') or [])}（{n_controls} 路） "
          f"decks={len(cs_pl.get('decks') or [])} old_decks={len(cs_old.get('decks') or [])} "
          f"health_ok={health_pl.get('ok')} health_points={len(health_pl.get('points') or [])} "
          f"log={os.path.basename(h_log) if h_log else None}")
    if not health_pl.get("ok"):
        print(f"[WARN] health 面板只有空分支可测：{h_log or '（该 run 推不出训练日志）'}")
    if n_controls < 1:
        print("[WARN] 该 run 没有 _controls_history ⇒ controls 曲线条数判据会用 0，跳过该断言")
        n_controls = 0

    tmp = os.path.join(_SRC, "runs", "_tmp_dashjs")
    os.makedirs(tmp, exist_ok=True)
    for name, obj in (("_solo.json", solo_pl), ("_cs.json", cs_pl), ("_cs_old.json", cs_old),
                      ("_health.json", health_pl)):
        with io.open(os.path.join(tmp, name), "w", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False, indent=1))
    # payload 直接**内嵌**进 JS（不读文件）⇒ 与路径无关
    prelude = ("const _SOLO_PAYLOAD = %s;\nconst _CS_PAYLOAD = %s;\n"
               "const _CS_OLD_PAYLOAD = %s;\nconst _HEALTH_PAYLOAD = %s;\n"
               % (json.dumps(solo_pl, ensure_ascii=False),
                  json.dumps(cs_pl, ensure_ascii=False),
                  json.dumps(cs_old, ensure_ascii=False),
                  json.dumps(health_pl, ensure_ascii=False)))
    tail = _TAIL % {"base": "unused/", "n_controls": n_controls}
    tail = (tail.replace('const soloPayload = JSON.parse(fs.readFileSync(B + "_solo.json", "utf8"));',
                         'const soloPayload = _SOLO_PAYLOAD;')
                .replace('const csPayload = JSON.parse(fs.readFileSync(B + "_cs.json", "utf8"));',
                         'const csPayload = _CS_PAYLOAD;')
                .replace('const csOld = JSON.parse(fs.readFileSync(B + "_cs_old.json", "utf8"));',
                         'const csOld = _CS_OLD_PAYLOAD;'))
    print("[dashboard-js] 运行渲染冒烟（node + DOM 桩 + 真实 payload，经 stdin）…")
    r = _run_node(runner, _STUB + prelude + js + tail)
    out = (r.stdout or "").strip()
    print(out if out else "(无 stdout)")
    if r.returncode != 0 and (r.stderr or "").strip():
        print("--- stderr ---")
        print((r.stderr or "")[-1500:])

    rc = 0 if r.returncode == 0 else 1
    if rc_tower:
        rc = 1

    # —— 联赛/长跑面板（run 模式）——
    if a.league_run:
        lg_dir = a.league_run if os.path.isabs(a.league_run) \
            else os.path.join(_SRC, a.league_run)
        sp = D.resolve_state_path(lg_dir)
        if sp is None:
            print(f"[FAIL] {lg_dir} 里找不到 league_state.json"
                  "（run 模式还没写出第一个评估点？）")
            return 1
        lg_pl = D.build_payload(sp)
        if not lg_pl.get("ok"):
            print(f"[FAIL] build_payload 失败：{lg_pl.get('error')}")
            return 1
        rm = lg_pl.get("run_meta") or {}
        wr_keys = [k for k in (lg_pl.get("winrate_curves") or {}) if k.startswith("main|")]
        n_big = sum(1 for rt in lg_pl.get("round_stats") or [] if rt.get("kind") == "big")
        print(f"[dashboard-js] 联赛 payload：state={os.path.basename(sp)} "
              f"曲线点={len(lg_pl.get('round_stats') or [])} "
              f"main|* 胜率曲线={len(wr_keys)} 大点={n_big} "
              f"进度={rm.get('cur_step')}/{rm.get('total_steps')}"
              f"（计划 {rm.get('plan_points')} 点 = 大 {rm.get('plan_big')} / 小 {rm.get('plan_small')}）")
        # 静默失效防线：已评估点里的"大点"个数，必须等于**计划中 step 已到达的那些大点**
        # （漏读 config 字段 ⇒ schedule 为空 ⇒ 标注 0 个；与"计划总数"比是错的量）
        sched = rm.get("schedule") or []
        last_step = max([rt.get("step", 0) for rt in
                         (lg_pl.get("round_stats") or [])] or [0])
        expect_big_done = sum(1 for s, k, _n in sched if k == "big" and int(s) <= last_step)
        expect_done = sum(1 for s, _k, _n in sched if sched and int(s) <= last_step)
        if sched and n_big != expect_big_done:
            print(f"[FAIL] 已评估点里的大点标注数 {n_big} != 计划中已到达的大点数 "
                  f"{expect_big_done}（last_step={last_step}）")
            return 1
        if sched and len(lg_pl.get("round_stats") or []) != expect_done:
            print(f"[FAIL] 曲线点数 {len(lg_pl.get('round_stats') or [])} != "
                  f"计划中已到达的点数 {expect_done}")
            return 1
        prelude2 = "const _LEAGUE_PAYLOAD = %s;\n" % json.dumps(lg_pl, ensure_ascii=False)
        tail2 = _LEAGUE_TAIL % {"league": "_LEAGUE_PAYLOAD"}
        r2 = _run_node(runner, _STUB + prelude2 + js + tail2)
        out2 = (r2.stdout or "").strip()
        print(out2 if out2 else "(联赛面板无 stdout)")
        if r2.returncode != 0 and (r2.stderr or "").strip():
            print("--- stderr (league) ---")
            print((r2.stderr or "")[-1500:])
        if r2.returncode != 0:
            rc = 1

    if a.keep:
        print(f"[dashboard-js] 证据 JSON 保留：{tmp}")
    else:
        shutil.rmtree(tmp, ignore_errors=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
