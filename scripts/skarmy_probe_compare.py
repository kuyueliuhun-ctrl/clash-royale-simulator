# -*- coding: utf-8 -*-
"""把**两台引擎**的骷髅军团探针录像并列成一张表（只读）。

输入：`runs/skarmy_probe/replays/league_*.pkl`（我方）与
      `runs/skarmy_probe_upstream/replays/league_*.pkl`（上游，已由
      `scripts/convert_upstream_frames.py` 转成 schema 5）。

输出：每个投放点一张表，每 1 s 一行：存活骷髅数 / 平均 y / x 区间 / 平均 |x−桥心|。
（表只描述录像里已有的事实；**不作判据**，判据要先预注册。）
"""
import argparse
import glob
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src", "clasher_new")
sys.path.insert(0, SRC)
from rl.io_bootstrap import force_utf8_stdout  # noqa: E402  (T1-1: UTF-8 兜底单一来源)
force_utf8_stdout()

#: 桥心 x：左桥 3.5、右桥 14.5（王塔后那一局两桥都会走，故列「距最近桥心」与「距中轴」两个量）
BRIDGE_X = {"bridge_left": 3.5, "bridge_right": 14.5}


def load(path):
    from rl.replay import load_league_replays
    games = load_league_replays(path)
    out = {}
    for g in games:
        side0 = g["meta"]["side0"]                      # skarmy@<placement>
        name = side0.split("@", 1)[1] if "@" in side0 else side0
        out[name] = g
    return out


def stats(game, every=10):
    rows = []
    for i, f in enumerate(game["frames"]):
        if i % every:
            continue
        sk = [e for e in f["entities"] if str(e[0]).startswith("SkeletonArmy")]
        xs = [e[1] for e in sk]
        ys = [e[2] for e in sk]
        rows.append((f["t"], len(sk),
                     (sum(ys) / len(ys)) if ys else None,
                     (min(xs), max(xs)) if xs else None))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default=os.path.join(SRC, "runs", "skarmy_probe", "replays", "league_*.pkl"))
    ap.add_argument("--upstream", default=os.path.join(SRC, "runs", "skarmy_probe_upstream", "replays", "league_*.pkl"))
    ap.add_argument("--every", type=int, default=10, help="每 N 帧取一行（10 帧 = 1 s）")
    args = ap.parse_args(argv)

    def pick(pat):
        hits = sorted(glob.glob(pat))
        if not hits:
            raise SystemExit("找不到录像：%s" % pat)
        return hits[-1]

    a, b = load(pick(args.ours)), load(pick(args.upstream))
    print("我方 %s\n上游 %s\n" % (pick(args.ours), pick(args.upstream)))
    for name in a:
        if name not in b:
            print("!! 上游缺投放点 %s" % name)
            continue
        print("=" * 78)
        print("投放点 %s" % name)
        print("%-6s | %-26s | %-26s" % ("t(s)", "我方（alive / meanY / x-range）",
                                        "上游（alive / meanY / x-range）"))
        ra, rb = stats(a[name], args.every), stats(b[name], args.every)
        # 双方都阵亡后不再逐行打印（尾巴全是 0，占版面）
        last_alive = 0
        for _i, r in enumerate(ra):
            if r[1] or rb[_i][1]:
                last_alive = _i
        for i, ((ta, na, ya, xa), (tb, nb, yb, xb)) in enumerate(zip(ra, rb)):
            if i > last_alive:
                break
            fa = "%2d / %5.1f / %s" % (na, ya if ya is not None else -1,
                                       "%.1f~%.1f" % xa if xa else "-")
            fb = "%2d / %5.1f / %s" % (nb, yb if yb is not None else -1,
                                       "%.1f~%.1f" % xb if xb else "-")
            mark = ""
            if xa and xb:
                d = abs((xa[0] + xa[1]) - (xb[0] + xb[1])) / 2.0
                if d >= 1.0:
                    mark = "   <= x 中心差 %.1f 格" % d
            print("%-6.1f | %-26s | %-26s%s" % (ta, fa, fb, mark))
        print()


if __name__ == "__main__":
    main()
