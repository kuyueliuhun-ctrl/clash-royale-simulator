#!/usr/bin/env python3
"""run_league 入口（wrapper）。

注意：必须用 if __name__ == "__main__" 保护，否则跨进程并行（spawn）的子进程会
重新导入本文件并再次 runpy 执行 run_league，造成无限递归。
"""
import os, sys, runpy

def main():
    _SRC = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "../../src/clasher_new"))
    os.chdir(_SRC)          # card_utils 等以相对路径读 gamedata.json
    sys.path.insert(0, _SRC)
    runpy.run_path(os.path.join(_SRC, "rl", "run_league.py"), run_name="__main__")

if __name__ == "__main__":
    main()
