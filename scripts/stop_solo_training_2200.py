# -*- coding: utf-8 -*-
"""22:00 定时停止 RL 训练（保留 dashboard）。

只结束 run_league.py --mode=solo 训练进程树，绝不碰 dashboard 或其它进程。
训练每 4000 步自动落盘（solo_main_<step>.pt + run_state.json，原子写入），
本脚本在 22:00 触发后：
  1. 定位训练进程 PID 集合（按命令行特征，不依赖 PID 硬编码）；
  2. 等最多 WAIT_MAX 秒，直到最近 checkpoint 步数到达下一个 4000 倍数
     （此时 checkpoint 已完整落盘，resume 干净），期间若训练已自然结束则直接返回；
  3. 到点后 taskkill /T /F 结束训练进程树。
用法：python stop_solo_training_2200.py
"""
import os
import re
import subprocess
import sys
import time

RUN_DIR = r"E:\clash-royale-simulator-main\src\clasher_new\runs\economy_towerprem"
RUN_STATE = os.path.join(RUN_DIR, "run_state.json")
LOG = r"E:\clash-royale-simulator-main\scripts\stop_solo_2200.log"
WAIT_MAX = 240  # 秒：最多等到下一个 4000 步边界（约 194s @20.6步/s）


def log(msg):
    line = "[%s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def find_training_pids():
    """返回命令行含 run_league.py --mode=solo 的所有 python 进程 PID 列表。"""
    try:
        out = subprocess.run(
            ["wmic", "process", "where",
             "name='python.exe' and CommandLine like '%run_league.py --mode=solo%'",
             "get", "ProcessId", "/value"],
            capture_output=True, text=True, timeout=30).stdout
    except Exception as e:
        log("wmic 查询失败: %r" % (e,))
        return []
    pids = []
    for m in re.finditer(r"ProcessId=(\d+)", out):
        pids.append(int(m.group(1)))
    return sorted(set(pids))


def read_step():
    """从 run_state.json 读最近已落盘步数；失败返回 None。"""
    try:
        with open(RUN_STATE, "r", encoding="utf-8") as f:
            return json.load(f).get("step")
    except Exception:
        return None


def kill_tree(pids, dry_run=False):
    for pid in pids:
        if dry_run:
            log("[dry-run] 将 taskkill /PID %d /T /F" % pid)
            continue
        try:
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                           capture_output=True, text=True, timeout=30)
            log("taskkill /PID %d /T /F 已发送" % pid)
        except Exception as e:
            log("taskkill %d 失败: %r" % (pid, e))


def main():
    dry_run = "--dry-run" in sys.argv
    log("== 22:00 定时停止 RL 训练 触发 (%s) ==" % ("DRY-RUN" if dry_run else "实际执行"))
    pids = find_training_pids()
    if not pids:
        log("未找到训练进程（已自然结束或未启动），退出")
        return

    # 等到下一个 4000 步边界，确保 checkpoint 完整落盘
    if os.path.exists(RUN_STATE) and not dry_run:
        t0 = time.time()
        last_log = 0
        while time.time() - t0 < WAIT_MAX:
            step = read_step()
            if step is not None and step % 4000 == 0 and step > 0:
                log("检测到 checkpoint @step %d 已完整落盘" % step)
                break
            now = time.time()
            if now - last_log > 30:
                log("等待 checkpoint 落盘… (step=%s, 已等 %.0fs)" % (step, now - t0))
                last_log = now
            time.sleep(2)
        else:
            log("等待超时 %ds，按当前状态强杀" % WAIT_MAX)
    elif dry_run:
        step = read_step()
        log("[dry-run] 当前 checkpoint step=%s；实际执行时会等下一个 4000 边界落盘后再终止"
            % (step,))
    else:
        log("run_state.json 不存在，直接按当前状态终止")

    # 二次确认进程仍在，避免误杀已重启的新训练
    pids = find_training_pids()
    if not pids:
        log("训练进程已自然结束，无需终止")
        return
    kill_tree(pids, dry_run=dry_run)
    log("完成：训练已停止，dashboard 保留运行")


if __name__ == "__main__":
    import json  # noqa: E402
    main()
