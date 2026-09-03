#!/usr/bin/env python3
"""selftest 入口（wrapper）。"""
import os, sys, runpy
_SRC = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../src/clasher_new"))
os.chdir(_SRC)
sys.path.insert(0, _SRC)
runpy.run_path(os.path.join(_SRC, "rl", "selftest.py"), run_name="__main__")
