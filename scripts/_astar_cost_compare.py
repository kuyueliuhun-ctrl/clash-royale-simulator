# -*- coding: utf-8 -*-
"""O3 离线判定：同一张 tilemap 上，**两套 A* 代价表**会不会产生不同的路径形状。

背景（我方 `pathfinding_heap.py:132-142` vs 上游 `pathfinding_heap.py:93-102`）：

| 格字符 | 我方代价 | 上游代价 |
|---|---|---|
| `.`（车道外地面 / 桥面） | **5** | **8** |
| `1` / `2`（**车道走廊**，从王塔区直通桥口） | 5 | 5 |
| `W`（河面等） | 800（地面） | 50（地面）／7（空中或 jump） |

我方 2026-09-10 把 `.` 从 8 改成 5，注释自述「代价相同，单位保持部署时的横向车道」。
本脚本**不改引擎**，只把两套代价表套在同一个起点/终点上跑同一份 A*，看路径形状差异。

判定：若「我方代价 ⇒ 末段才横移（dogleg）」而「上游代价 ⇒ 全程逐步收拢」，
则 O3 的成因 = **代价表抹掉了 `1`/`2` 走廊相对 `.` 的价差**（不是跟随路径的方式）。

用法（纯 stdlib，WSL python3 即可）：
    python3 scripts/_astar_cost_compare.py
"""
import heapq
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "src", "clasher_new"))
from rl.io_bootstrap import force_utf8_stdout  # noqa: E402
force_utf8_stdout()

TILEMAP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "src", "clasher_new", "tilemap_lane_grid.txt")
with open(TILEMAP, encoding="utf-8") as fh:
    CONTENTS = [list(line) for line in fh.read().splitlines()]

#: 我方 arena.BLOCKED_TILES 里与河道有关的部分（角旗 + 河面中段）
OURS_RIVER_BLOCK = {(i, j) for i in range(5, 13) for j in range(15, 17)} | {
    (0, 15), (0, 16), (1, 15), (1, 16), (16, 15), (16, 16), (17, 15), (17, 16)}
CORNERS = ([(i, 0) for i in range(0, 6)] + [(i, 0) for i in range(12, 18)]
           + [(i, 31) for i in range(0, 6)] + [(i, 31) for i in range(12, 18)])


def char_at(cell):
    x, y = cell
    if x < 0 or y < 0 or x >= 36 or y >= 64:
        return None
    return CONTENTS[63 - y][x]


def walkable_ours(cell):
    x, y = cell
    if x < 0 or y < 0 or x >= 36 or y >= 64:
        return False
    if (x, y) in CORNERS or (x, y) in OURS_RIVER_BLOCK:
        return False
    px, py = (x + 0.5) / 2, (y + 0.5) / 2
    if 15.0 <= py <= 16.0:                       # 我方：河带只放行两条桥
        return (2.0 <= px < 5.0) or (13.0 <= px < 16.0)
    return True


def walkable_upstream(cell):
    x, y = cell
    if x < 0 or y < 0 or x >= 36 or y >= 64:
        return False
    return (x, y) not in CORNERS               # 上游：河面可走（只挡角旗）


def cost_ours(cell):
    ch = char_at(cell)
    if ch == 'W':
        return 800
    return 5                                    # '.' 与 '1'/'2' 同价（2026-09-10 改动）


def cost_upstream(cell):
    ch = char_at(cell)
    if ch == 'W':
        return 50
    if ch == '.':
        return 8
    return 5                                    # '1'/'2'


def astar(start, goal, walkable, cost):
    def h(c):
        return 10 * max(abs(c[0] - goal[0]), abs(c[1] - goal[1]))
    open_heap = [(h(start), start)]
    g = {start: 0}
    parent = {}
    closed = set()
    while open_heap:
        _, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        closed.add(cur)
        if cur == goal:
            path = [cur]
            while path[-1] in parent:
                path.append(parent[path[-1]])
            return list(reversed(path)), g[goal]
        cx, cy = cur
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nb = (cx + dx, cy + dy)
                if nb in closed or not walkable(nb):
                    continue
                tile = cost(nb)
                geo = 14 if (dx and dy) else 10
                tentative = g[cur] + tile * geo
                if nb not in g or tentative < g[nb]:
                    g[nb] = tentative
                    parent[nb] = cur
                    heapq.heappush(open_heap, (tentative + h(nb), nb))
    return None, None


def profile(path):
    """每个 y 行取路径 x（用于看「何时横移」）。"""
    out = {}
    for (x, y) in path:
        out.setdefault(y, x)
    return out


def report(tag, start, goal, walkable, cost):
    path, total = astar(start, goal, walkable, cost)
    print("=" * 72)
    print(f"### {tag}")
    if path is None:
        print("  无解")
        return None
    print(f"  代价合计 = {total}   路径点数 = {len(path)}")
    # 横向位移分布：每 5 行取一次 x
    prof = profile(path)
    rows = sorted(prof)
    samp = rows[::max(1, len(rows) // 10)]
    print("  y行 → x列（每 ~10% 抽样）: " + "  ".join(f"{y}:{prof[y]}" for y in samp))
    # 关键形状指标：路径在「只走前一半 y」时横向走了多少
    xs = [prof[y] for y in rows]
    total_lat = abs(xs[-1] - xs[0]) or 1
    half = rows[len(rows) // 2]
    early = abs(prof[half] - xs[0])
    print(f"  首行 y={rows[0]} x={xs[0]}  中段 y={half} x={prof[half]}  末行 y={rows[-1]} x={xs[-1]}")
    print(f"  ⇒ **前半程完成的横向位移占比 = {early / total_lat:.1%}**"
          f"（越低 = 越像 dogleg「末段才横移」）")
    return early / total_lat


def main():
    # 与探针实测同一场景：左道骷髅，起点 ≈ (5.75,0.75)，终点 ≈ (4.75,23.25)
    start, goal = (11, 1), (9, 46)
    print(f"起点格 {start} (世界 {((start[0])+0.5)/2:.2f},{((start[1])+0.5)/2:.2f})  "
          f"终点格 {goal} (世界 {((goal[0])+0.5)/2:.2f},{((goal[1])+0.5)/2:.2f})")
    a = report("我方：'.'=5, 'W'=800, 河带只放行两条桥", start, goal,
               walkable_ours, cost_ours)
    b = report("上游：'.'=8, 'W'=50, 河面可走", start, goal,
               walkable_upstream, cost_upstream)
    c = report("只改代价、不改可走性：'.'=8, 'W'=800, 河带只放行两条桥", start, goal,
               walkable_ours, cost_upstream)
    print("=" * 72)
    print("解读：A 若显著低于 B/C ⇒ O3 的成因是**代价表**（不是可走性），"
          "最小改动 = 把 '.' 从 5 改回 8。")


if __name__ == "__main__":
    main()
