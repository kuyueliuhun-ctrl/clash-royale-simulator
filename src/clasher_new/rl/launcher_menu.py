#!/usr/bin/env python3
"""CR-RL 交互式启动向导（start_rl.bat 无参数 / --menu 时进入）。

设计（用户口径 7i）：打开启动脚本后分层问答，全部带预设值，直接回车即用预设——
  第一层：训练类型（solo / 联赛 run / 分流联赛 flow）
  第二层：配置预设（standard / aggressive / defensive / lockdown / elixir /
          economy / fast / 自定义 JSON）
  第三层：常用参数逐项（显示预设值，回车不改）
  第四层：确认并启动（可选一并打开 dashboard + 浏览器）

不引入任何训练/引擎依赖（纯 stdlib），在 .venv 探测完成后由 bat 调用；
训练与 dashboard 用同一 Python 在新控制台窗口启动。
"""

import os
import subprocess
import sys
import time
import webbrowser

#: 仓库根目录：<root>/src/clasher_new/rl/launcher_menu.py → 上溯 3 级
ROOT = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "..", "..", ".."))
RUN_LEAGUE_WRAP = os.path.join(ROOT, "scripts", "rl", "run_league.py")
DASH_WRAP = os.path.join(ROOT, "scripts", "rl", "dashboard.py")
DEFAULT_PORT = 8090

MODES = [
    ("solo", "单人自对弈（固定卡组镜像 + 冻结副本，写 solo_state；最常见）"),
    ("run",  "联赛（main PPO + 5 模型槽 + PFSP 对手采样）"),
    ("flow", "分流联赛（全配对 6 模型 pairwise；量大）"),
]
CONFIGS = [
    "standard", "aggressive", "defensive", "lockdown", "elixir", "economy", "fast",
]

#: 各模式的预设默认值（与 start_rl.bat / economy 预设保持一致）
DEFAULTS = {
    "out_dir": "runs",
    "total_steps": 20000,
    "steps_per_eval": 2000,
    "n_eval_games": 16,
    "max_ep_steps": 360,
    "eval_workers": 16,
    "solo_copy_every": 2000,
    "n_envs": 1,
    "gae_lambda": 0.99,
    "ent_coef": 0.01,
    "device": "auto",
    "config_name": "",      # 空 = 用配置名
    "load_config": "",      # 空 = 命名预设
    "fresh": False,         # False = 默认自动续训
    "keep_snapshot": False,
    "only_vs_main": False,
    "with_dashboard": True,
    "port": DEFAULT_PORT,
}


def ask(prompt, default, cast=str, allow_back=True):
    """逐项问答：回车=默认值；输入 b/B=返回上一步；q/Q=退出。"""
    while True:
        try:
            raw = input(f"{prompt} [默认 {default}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[向导] 已退出。")
            sys.exit(0)
        if raw.lower() == "q":
            print("[向导] 已退出。")
            sys.exit(0)
        if allow_back and raw.lower() == "b":
            return ".."
        if raw == "":
            return default
        try:
            return cast(raw)
        except ValueError:
            print(f"[向导] “{raw}” 不是有效的 {cast.__name__}，请重试。")


def ask_yesno(prompt, default=True):
    d = "Y/n" if default else "y/N"
    while True:
        v = ask(f"{prompt}（{d}）", "y" if default else "n", str).lower()
        if v == "..":
            return ".."
        if v in ("y", "n"):
            return v == "y"
        print("[向导] 请输入 y 或 n。")


def pick(label, options, default_idx=0, extra=None):
    """打印选项列表并让用户选序号；回车选 default_idx。"""
    print()
    print(f"—— {label} ——")
    for i, (key, desc) in enumerate(options):
        mark = "  [预设]" if i == default_idx else ""
        print(f"  {i + 1}. {key:<8} {desc}{mark}")
    if extra:
        print(f"  {len(options) + 1}. {extra}")
    while True:
        raw = input(f"请选择 (1-{len(options) + 1 if extra else len(options)}，回车={default_idx + 1}) > ").strip()
        if raw == "":
            return options[default_idx][0]
        if raw.lower() == "q":
            print("[向导] 已退出。")
            sys.exit(0)
        try:
            n = int(raw)
        except ValueError:
            print(f"[向导] “{raw}” 不是序号，请重试。")
            continue
        if 1 <= n <= len(options):
            return options[n - 1][0]
        if extra and n == len(options) + 1:
            return None  # 自定义
        print(f"[向导] 序号超出范围 (1-{len(options) + 1 if extra else len(options)})，请重试。")


def pick_config():
    options = [(c, "") for c in CONFIGS]
    sel = pick("第二层 · 配置预设", options, default_idx=5, extra="自定义 JSON (--load-config)")
    if sel is not None:
        return sel, ""
    while True:
        path = input("输入自定义 config JSON 路径 > ").strip()
        if path.lower() == "q":
            sys.exit(0)
        if os.path.exists(path):
            return "", path
        print(f"[向导] 文件不存在: {path}，请重试（q 退出）")


def collect_params(mode):
    """第三层：逐项参数（显示预设值，回车不改）。返回参数 dict + 标志（..=返回）。"""
    cfg_name = ask("输出配置名（文件夹名；回车=用配置名）", DEFAULTS["config_name"], str)
    if cfg_name == "..":
        return None, ".."
    out_dir = ask("输出根目录 out_dir", DEFAULTS["out_dir"], str)
    if out_dir == "..":
        return None, ".."
    total = ask("总训练步数 total_steps", DEFAULTS["total_steps"], int)
    if total == "..":
        return None, ".."
    spe = ask("每隔多少步评估一次 steps_per_eval", DEFAULTS["steps_per_eval"], int)
    if spe == "..":
        return None, ".."
    neg = ask("每次评估对局数 n_eval_games", DEFAULTS["n_eval_games"], int)
    if neg == "..":
        return None, ".."
    mes = ask("每局最大决策步 max_ep_steps", DEFAULTS["max_ep_steps"], int)
    if mes == "..":
        return None, ".."
    if mode == "solo":
        ew = ask("评估并行进程数 eval_workers（0=串行）", DEFAULTS["eval_workers"], int)
        if ew == "..":
            return None, ".."
        sce = ask("冻结副本同步间隔 solo_copy_every", DEFAULTS["solo_copy_every"], int)
        if sce == "..":
            return None, ".."
    else:
        ew, sce = DEFAULTS["eval_workers"], DEFAULTS["solo_copy_every"]
    ne = ask("并行环境数 n_envs（1=单环境；>1 批量推理）", DEFAULTS["n_envs"], int)
    if ne == "..":
        return None, ".."
    gl = ask("GAE lambda（economy 预设=0.99；0.95=旧视野）", DEFAULTS["gae_lambda"], float)
    if gl == "..":
        return None, ".."
    ec = ask("ent_coef（熵系数）", DEFAULTS["ent_coef"], float)
    if ec == "..":
        return None, ".."
    dev = ask("device", DEFAULTS["device"], str)
    if dev == "..":
        return None, ".."
    # 训练入口默认自动续训（c857c9f 起）；fresh 才从头
    fresh = ask_yesno("从头训练（--fresh）？（默认自动续训）", False)
    if fresh == "..":
        return None, ".."
    keep = False
    if mode == "run":
        keep = ask_yesno("联赛：保留 main_ckpt 快照槽（--keep-snapshot）？", False)
        if keep == "..":
            return None, ".."
    ovm = True
    if mode == "run":
        ovm = ask_yesno("联赛：评估只测 main vs 其他（--only-vs-main）？", True)
        if ovm == "..":
            return None, ".."
    params = dict(DEFAULTS)
    params.update({
        "config_name": cfg_name, "out_dir": out_dir,
        "total_steps": total, "steps_per_eval": spe, "n_eval_games": neg,
        "max_ep_steps": mes, "eval_workers": ew, "solo_copy_every": sce,
        "n_envs": ne, "gae_lambda": gl, "ent_coef": ec, "device": dev,
        "fresh": fresh, "keep_snapshot": keep, "only_vs_main": ovm,
    })
    return params, None


def build_cmd(mode, config, load_config, p):
    cmd = [sys.executable, RUN_LEAGUE_WRAP,
           "--mode", mode, "--out-dir", p["out_dir"],
           "--device", p["device"],
           "--n-envs", str(p["n_envs"]),
           "--total-steps", str(p["total_steps"]),
           "--steps-per-eval", str(p["steps_per_eval"]),
           "--n-eval-games", str(p["n_eval_games"]),
           "--max-ep-steps", str(p["max_ep_steps"]),
           "--gae-lambda", str(p["gae_lambda"]),
           "--ent-coef", str(p["ent_coef"])]
    if load_config:
        cmd += ["--load-config", load_config]
    else:
        cmd += ["--config", config]
    if p["config_name"]:
        cmd += ["--config-name", p["config_name"]]
    if mode == "solo":
        cmd += ["--eval-workers", str(p["eval_workers"]),
                "--solo-copy-every", str(p["solo_copy_every"])]
    if p["fresh"]:
        cmd += ["--fresh"]
    if p["keep_snapshot"]:
        cmd += ["--keep-snapshot"]
    if p["only_vs_main"]:
        cmd += ["--only-vs-main"]
    return cmd


def state_arg(mode, p):
    folder = p["config_name"] if p["config_name"] else None
    base = p["out_dir"]
    sub = None
    # 配置名是第三层才选的输出名，state 目录 = out_dir/<config_name 或 config>
    # 由于第三层 config_name 默认空，这里展示的是"最终目录"提示，真正路径由
    # run_league 侧保持一致（out_dir/<config-name>）。
    return base, folder


def main():
    print("=" * 62)
    print("  CR-RL 交互式启动向导（回车=预设值；b=返回；q=退出）")
    print("=" * 62)
    mode = pick("第一层 · 训练类型", MODES, default_idx=0)
    config, load_config = pick_config()
    if mode == "flow":
        print()
        print("[flow] 分流联赛规模大，本向导不逐项展开参数，直接采用预设并启动。")
        print("[flow] 若需详细参数，请退出后用命令行: start_rl.bat --mode flow ...")
        p = dict(DEFAULTS)
        p["total_steps"] = 20000
        p["only_vs_main"] = True
        cmd = build_cmd(mode, config, load_config, p)
        print()
        print("启动命令：")
        print("  " + subprocess.list2cmdline(cmd))
        if not ask_yesno("确认启动？", True):
            print("[向导] 已取消。")
            return
        launch(cmd)
        return
    while True:
        p, back = collect_params(mode)
        if back == "..":
            print("[向导] 返回第一层……")
            mode = pick("第一层 · 训练类型", MODES, default_idx=0)
            continue
        print()
        print("=" * 62)
        print("  第四层 · 确认")
        print("-" * 62)
        print(f"  训练类型    : {mode}")
        print(f"  配置        : {config if config else load_config}")
        print(f"  输出目录    : {p['out_dir']}" +
              (f"\\{p['config_name']}" if p["config_name"] else ""))
        print(f"  total_steps : {p['total_steps']}   steps_per_eval: {p['steps_per_eval']}")
        print(f"  n_eval_games: {p['n_eval_games']}   max_ep_steps  : {p['max_ep_steps']}")
        if mode == "solo":
            print(f"  eval_workers: {p['eval_workers']}   solo_copy_every: {p['solo_copy_every']}")
        print(f"  n_envs      : {p['n_envs']}   gae_lambda: {p['gae_lambda']}   "
              f"ent_coef: {p['ent_coef']}")
        print(f"  device      : {p['device']}   fresh(从头): {p['fresh']}")
        print("-" * 62)
        cmd = build_cmd(mode, config, load_config, p)
        print("  启动命令：")
        print("  " + subprocess.list2cmdline(cmd))
        act = ask("输入 1=启动  2=改参数  3=重新选训练类型（默认 1）", "1", str).strip().lower()
        if act in ("2", "s"):
            continue
        if act in ("3", "m"):
            mode = pick("第一层 · 训练类型", MODES, default_idx=0)
            continue
        if act in ("q", "0"):
            return
        if not ask_yesno("确认启动训练？", True):
            return
        launch(cmd)
        # dashboard（可选）
        if mode in ("solo", "run", "flow") and p["with_dashboard"]:
            folder = p["config_name"] if p["config_name"] else config
            if mode == "solo":
                dargs = ["--solo", os.path.join(p["out_dir"], folder)]
            elif mode == "run":
                dargs = ["--state", os.path.join(p["out_dir"], folder, "league_state.json")]
            else:
                dargs = ["--sweep", os.path.join(p["out_dir"], folder)]
            dcmd = [sys.executable, DASH_WRAP] + dargs + ["--port", str(p["port"])]
            print("[向导] 启动 dashboard: " + subprocess.list2cmdline(dcmd))
            launch(dcmd)
            time.sleep(3.0)
            webbrowser.open(f"http://127.0.0.1:{p['port']}/")
        print("[向导] 训练已在独立窗口启动（关闭该窗口=停止）。")
        return


def launch(cmd):
    """新开独立控制台窗口（Windows），其它平台前台子进程。"""
    if os.name == "nt":
        return subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
    return subprocess.Popen(cmd)


if __name__ == "__main__":
    main()
