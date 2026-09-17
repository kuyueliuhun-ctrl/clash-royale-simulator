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
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

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


def extract_js():
    with io.open(_DASH, encoding="utf-8") as f:
        src = f.read()
    m = re.search(r"<script>(.*?)</script>", src, re.S)
    if not m:
        raise SystemExit("[FAIL] dashboard.py 里找不到 <script> 块")
    return m.group(1)


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
    ap.add_argument("--syntax-only", action="store_true")
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
    print(f"[dashboard-js] payload：history={len(solo_pl.get('history') or [])} "
          f"controls={len(solo_pl.get('controls_history') or [])}（{n_controls} 路） "
          f"decks={len(cs_pl.get('decks') or [])} old_decks={len(cs_old.get('decks') or [])}")
    if n_controls < 1:
        print("[WARN] 该 run 没有 _controls_history ⇒ controls 曲线条数判据会用 0，跳过该断言")
        n_controls = 0

    tmp = os.path.join(_SRC, "runs", "_tmp_dashjs")
    os.makedirs(tmp, exist_ok=True)
    for name, obj in (("_solo.json", solo_pl), ("_cs.json", cs_pl), ("_cs_old.json", cs_old)):
        with io.open(os.path.join(tmp, name), "w", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False, indent=1))
    # payload 直接**内嵌**进 JS（不读文件）⇒ 与路径无关
    prelude = ("const _SOLO_PAYLOAD = %s;\nconst _CS_PAYLOAD = %s;\nconst _CS_OLD_PAYLOAD = %s;\n"
               % (json.dumps(solo_pl, ensure_ascii=False),
                  json.dumps(cs_pl, ensure_ascii=False),
                  json.dumps(cs_old, ensure_ascii=False)))
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
    if a.keep:
        print(f"[dashboard-js] 证据 JSON 保留：{tmp}")
    else:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0 if r.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
