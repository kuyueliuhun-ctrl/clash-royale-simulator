# -*- coding: utf-8 -*-
"""骷髅军团轨迹**逐单位**分析：验证用户四条观测（2026-09-19）。

用户观测（原文）：
  O1 我方"能够做到桥上三只小骷髅同时通过，但通过后出现了寻路混乱"
  O2 沉底后我方"分到少骷髅的那一侧，出动比另一面更慢"
  O3 我方"己方半场单位寻路不会主动靠近桥中间的 y 轴，而是走到河道才会向桥上走"
  O4 "从空地分应该是少的一面6只小骷髅，王塔后分应该是少的一面5只，这点两个引擎都有问题"

本脚本**只读录像**（schema 5），除 stdout / `--json` 外不写任何文件、不碰训练与权重。

读入两份录像（同一套投放点、同节奏）：
  我方   src/clasher_new/runs/skarmy_probe/replays/league_1800.pkl
  上游   src/clasher_new/runs/skarmy_probe_upstream/replays/league_1800.pkl
  3 局 = 3 个投放点，每局 600 帧 @ 0.1 s = 60 s。

⚠️ 口径声明：本脚本产出**描述性读数**，不是预注册判据（【R3】）；不写"谁的引擎更好"。
"""
import argparse
import json
import math
import os
import pickle
import sys

# path 引导必须先于产品 import（核对器 ⑪）
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "src", "clasher_new")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from rl.io_bootstrap import force_utf8_stdout  # noqa: E402
force_utf8_stdout()

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

SOURCES = [
    ("ours", os.path.join(ROOT, "src", "clasher_new", "runs",
                          "skarmy_probe", "replays", "league_1800.pkl")),
    ("upstream", os.path.join(ROOT, "src", "clasher_new", "runs",
                              "skarmy_probe_upstream", "replays", "league_1800.pkl")),
]

PLACEMENTS = ["behind_king", "bridge_left", "bridge_right"]
BRIDGE_CENTER_X = {"left": 3.5, "right": 14.5}
BRIDGE_CORRIDOR = {"left": (2.0, 5.0), "right": (13.0, 16.0)}
MID_X = 9.0
RIVER_LO, RIVER_HI = 15.0, 17.0   # 河带（世界 y）

#: schema 5 实体元组（15 长）下标
E_NAME, E_X, E_Y, E_HP, E_PLAYER, E_KIND = 0, 1, 2, 3, 4, 5
E_ID, E_TARGET = 10, 11


# ---------------------------------------------------------------- 载入
def load_games(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    if isinstance(d, list):
        games = d
    elif isinstance(d, dict):
        for key in ("games", "replays", "league"):
            if isinstance(d.get(key), list):
                games = d[key]
                break
        else:
            games = [d] if "frames" in d else []
    else:
        games = []
    return games


def frames_ents(game):
    """返回 (t_list, [entities 列表...])，兼容 dict 帧与位置帧。"""
    frames = game["frames"]
    ts, ents = [], []
    for fr in frames:
        if isinstance(fr, dict):
            ts.append(fr.get("t"))
            ents.append(fr.get("entities") or [])
        else:
            ts.append(fr[0])
            ents.append(fr[-1] or [])
    return ts, ents


def trajectories(ts, ents, name_prefix, player):
    """{id: {'name':.., 'samples':[(t,x,y,hp)]}}，只留指定玩家 + 名字前缀。"""
    traj = {}
    for t, elist in zip(ts, ents):
        for e in elist:
            if len(e) < 13:
                continue
            if e[E_PLAYER] != player:
                continue
            if not str(e[E_NAME]).startswith(name_prefix):
                continue
            traj.setdefault(e[E_ID], {"name": e[E_NAME], "samples": []})
            traj[e[E_ID]]["samples"].append((t, e[E_X], e[E_Y], e[E_HP]))
    return traj


# ---------------------------------------------------------------- 几何工具
def x_at_y(samples, y_target):
    """线性插值出 y=target 时的 x；不在覆盖范围内返回 None。"""
    for (t0, x0, y0, _), (t1, x1, y1, _) in zip(samples, samples[1:]):
        if (y0 - y_target) * (y1 - y_target) <= 0 and y1 != y0:
            f = (y_target - y0) / (y1 - y0)
            return x0 + f * (x1 - x0)
    return None


def first_cross(samples, y_thresh):
    for t, x, y, _ in samples:
        if y >= y_thresh:
            return t
    return None


def lane_of(samples):
    """用河带内的 x 判定该单位走的是哪座桥。"""
    xs = [x for _, x, y, _ in samples if RIVER_LO <= y <= RIVER_HI]
    if not xs:
        return None, None
    mx = sum(xs) / len(xs)
    return ("left" if mx < MID_X else "right"), mx


def longest_stall(samples, eps=0.05):
    """最长「连续不动」段（帧数；1 帧 = 0.1 s）。只统计存活期间的样本。"""
    alive = [s for s in samples if s[3] > 0]
    best = cur = 0
    for (_, x0, y0, _), (_, x1, y1, _) in zip(alive, alive[1:]):
        if math.hypot(x1 - x0, y1 - y0) < eps:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def sign_changes(vals, eps=0.02):
    """方向反转次数（忽略 |d|<eps 的抖动）。"""
    n, prev = 0, 0
    for a, b in zip(vals, vals[1:]):
        d = b - a
        if abs(d) < eps:
            continue
        s = 1 if d > 0 else -1
        if prev and s != prev:
            n += 1
        prev = s
    return n


# ---------------------------------------------------------------- 分析
def analyze_game(label, engine, ts, ents):
    traj = trajectories(ts, ents, "Skeleton", 0)
    if not traj:
        return None, {"error": "no Skeleton* entities for player 0"}

    units = {}
    for eid, rec in traj.items():
        s = rec["samples"]
        lane, mean_river_x = lane_of(s)
        units[eid] = {
            "name": rec["name"],
            "n_samples": len(s),
            "spawn": (s[0][1], s[0][2]),
            "last": (s[-1][1], s[-1][2]),
            "last_hp": s[-1][3],
            "lane": lane,
            "river_mean_x": mean_river_x,
            "t_cross_17": first_cross(s, 17.5),
            "t_river": first_cross(s, RIVER_HI),
            "x_at": {y: x_at_y(s, float(y)) for y in (2, 4, 6, 8, 10, 12, 14, 16)},
            "x_span": (max(x for _, x, _, _ in s) - min(x for _, x, _, _ in s)),
            "samples": s,
        }

    out = {
        "engine": engine, "label": label,
        "n_units": len(units),
        "names": sorted({u["name"] for u in units.values()}),
        "lane_counts": {k: sum(1 for u in units.values() if u["lane"] == k)
                        for k in ("left", "right", None)},
        "units": {str(k): {kk: vv for kk, vv in v.items() if kk != "samples"}
                  for k, v in units.items()},
    }

    # ---- 首帧坐标（判定「出兵环」的真实半径 / 是否被夹取 / 是否被碰撞炸开） ----
    out["first_frame_positions"] = {
        str(k): [round(u["spawn"][0], 3), round(u["spawn"][1], 3), u["lane"]]
        for k, u in units.items()}
    out["first_frame_x_range"] = [round(min(u["spawn"][0] for u in units.values()), 3),
                                  round(max(u["spawn"][0] for u in units.values()), 3)]
    out["first_frame_y_range"] = [round(min(u["spawn"][1] for u in units.values()), 3),
                                  round(max(u["spawn"][1] for u in units.values()), 3)]
    cx = sum(u["spawn"][0] for u in units.values()) / len(units)
    cy = sum(u["spawn"][1] for u in units.values()) / len(units)
    out["first_frame_centroid"] = [round(cx, 3), round(cy, 3)]
    rr = sorted(math.hypot(u["spawn"][0] - cx, u["spawn"][1] - cy) for u in units.values())
    out["first_frame_radius_median"] = round(rr[len(rr) // 2], 3)
    out["first_frame_radius_max"] = round(rr[-1], 3)

    # ---- 「从空地分」：按 y 带统计左/右（每个单位在该带内取平均 x 判侧） ----
    bands = {}
    for lo in range(0, 34, 2):
        L = R = 0
        for rec in traj.values():
            xs = [x for _, x, y, _ in rec["samples"] if lo <= y < lo + 2]
            if not xs:
                continue
            if sum(xs) / len(xs) < MID_X:
                L += 1
            else:
                R += 1
        if L + R:
            bands[f"y{lo}-{lo+2}"] = {"left": L, "right": R}
    out["split_by_y_band"] = bands

    # ---- 卡死/拥堵：最长「连续不动」段（帧 = 0.1 s），只算存活期 ----
    stalls = {str(k): longest_stall(u["samples"]) for k, u in units.items()}
    sv = sorted(stalls.values())
    out["stall_frames"] = stalls
    out["stall_median"] = sv[len(sv) // 2]
    out["stall_max"] = sv[-1]

    # ---- O3：己方半场是否主动向桥中心收敛 ----
    conv = {}
    for y in (2, 4, 6, 8, 10, 12, 14):
        vals = [u["x_at"][y] for u in units.values() if u["x_at"].get(y) is not None]
        if vals:
            conv[y] = {"n": len(vals), "mean_x": round(sum(vals) / len(vals), 3),
                       "mean_off_bridge": None}
    # 以「该单位最终走的那座桥的中心」为基准算平均横向偏离
    for y, row in conv.items():
        offs = []
        for u in units.values():
            xv = u["x_at"].get(y)
            if xv is None or u["lane"] is None:
                continue
            offs.append(abs(xv - BRIDGE_CENTER_X[u["lane"]]))
        row["mean_off_bridge"] = round(sum(offs) / len(offs), 3) if offs else None
    out["own_half_convergence"] = conv

    # 逐单位「横向行程完成度」：从出兵 x 到桥心 x 的进度，在 y=6 / 10 / 14 时完成了多少
    prog = {}
    for y in (6, 10, 14):
        fr = []
        for u in units.values():
            if u["lane"] is None:
                continue
            x0, xb = u["spawn"][0], BRIDGE_CENTER_X[u["lane"]]
            xv = u["x_at"].get(y)
            if xv is None or abs(xb - x0) < 0.3:
                continue  # 出兵点已经在桥心附近 ⇒ 无横向行程可测
            fr.append((xv - x0) / (xb - x0))
        prog[y] = {"n": len(fr),
                   "median": round(sorted(fr)[len(fr) // 2], 3) if fr else None,
                   "mean": round(sum(fr) / len(fr), 3) if fr else None}
    out["lateral_progress_to_bridge"] = prog

    # ---- O2：分道后各道"出动"速度 ----
    per_lane = {}
    for lane in ("left", "right"):
        grp = [u for u in units.values() if u["lane"] == lane]
        if not grp:
            per_lane[lane] = {"n": 0}
            continue
        crosses = sorted(u["t_cross_17"] for u in grp if u["t_cross_17"] is not None)
        arrived_river = sorted(u["t_river"] for u in grp if u["t_river"] is not None)
        per_lane[lane] = {
            "n": len(grp),
            "n_reached_river": len(arrived_river),
            "n_crossed": len(crosses),
            "t_first_river": arrived_river[0] if arrived_river else None,
            "t_first_cross": crosses[0] if crosses else None,
            "t_median_cross": crosses[len(crosses) // 2] if crosses else None,
            "t_last_cross": crosses[-1] if crosses else None,
            "spawn_x_range": [round(min(u["spawn"][0] for u in grp), 2),
                              round(max(u["spawn"][0] for u in grp), 2)],
        }
    out["per_lane"] = per_lane

    # ---- O1：过桥之后的行为（混乱度指标） ----
    post = {}
    for lane in ("left", "right"):
        grp = [u for u in units.values() if u["lane"] == lane and
               u["t_cross_17"] is not None]
        if not grp:
            post[lane] = {"n": 0}
            continue
        rev_x, rev_y, xs, ys, n_tr = [], [], [], [], 0
        for u in grp:
            tail = [s for s in u["samples"] if u["t_cross_17"] is None or
                    s[0] >= u["t_cross_17"]]
            if len(tail) < 3:
                continue
            n_tr += 1
            rev_x.append(sign_changes([x for _, x, _, _ in tail]))
            rev_y.append(sign_changes([y for _, _, y, _ in tail]))
            xs.append(max(x for _, x, _, _ in tail) - min(x for _, x, _, _ in tail))
            ys.append(max(y for _, _, y, _ in tail) - min(y for _, _, y, _ in tail))
        if not n_tr:
            post[lane] = {"n": 0}
            continue
        med = lambda a: round(sorted(a)[len(a) // 2], 3) if a else None
        post[lane] = {
            "n": n_tr,
            "median_x_dir_reversals": med(rev_x),
            "mean_x_dir_reversals": round(sum(rev_x) / len(rev_x), 2),
            "median_y_dir_reversals": med(rev_y),
            "median_x_span_after_cross": med(xs),
            "median_y_gain_after_cross": med(ys),
        }
    out["post_cross"] = post

    # 桥廊内最大同时在员（"同时过桥几只"）
    occ = {"left": 0, "right": 0}
    for elist in ents:
        cur = {"left": 0, "right": 0}
        for e in elist:
            if len(e) < 13 or e[E_PLAYER] != 0:
                continue
            if not str(e[E_NAME]).startswith("Skeleton"):
                continue
            if not (RIVER_LO <= e[E_Y] <= RIVER_HI):
                continue
            lo, hi = BRIDGE_CORRIDOR["left"]
            if lo <= e[E_X] <= hi:
                cur["left"] += 1
            lo, hi = BRIDGE_CORRIDOR["right"]
            if lo <= e[E_X] <= hi:
                cur["right"] += 1
        for k in occ:
            occ[k] = max(occ[k], cur[k])
    out["max_simultaneous_in_corridor"] = occ

    # 出桥廊的帧数（河带内 x 落在桥廊之外 = 几何上在河面上）
    off = {"left": 0, "left_tot": 0, "right": 0, "right_tot": 0}
    for elist in ents:
        for e in elist:
            if len(e) < 13 or e[E_PLAYER] != 0:
                continue
            if not str(e[E_NAME]).startswith("Skeleton"):
                continue
            if not (RIVER_LO <= e[E_Y] <= RIVER_HI):
                continue
            for lane in ("left", "right"):
                lo, hi = BRIDGE_CORRIDOR[lane]
                if abs(e[E_X] - BRIDGE_CENTER_X[lane]) < 3.0:   # 归属该道
                    off[lane + "_tot"] += 1
                    if not (lo <= e[E_X] <= hi):
                        off[lane] += 1
    out["river_frames_outside_corridor"] = off
    return out, units


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None, help="把结构化结果写到该文件")
    ap.add_argument("--dump-frame", action="store_true", help="打印第一帧结构后退出")
    ap.add_argument("--add", action="append", default=[], metavar="NAME=PATH",
                    help="追加一个录像源（可重复；如 nolane=.../league_1800.pkl）")
    args = ap.parse_args(argv)
    for _s in args.add:
        _n, _p = _s.split("=", 1)
        SOURCES.append((_n, _p))

    results = {}
    for engine, path in SOURCES:
        if not os.path.exists(path):
            print(f"[skip] {engine}: 录像不存在 {path}")
            continue
        games = load_games(path)
        print("=" * 78)
        print(f"### {engine}  {path}")
        print(f"    局数 = {len(games)}")
        per_game = []
        for gi, game in enumerate(games):
            label = PLACEMENTS[gi] if gi < len(PLACEMENTS) else f"game{gi}"
            meta = game.get("meta") if isinstance(game, dict) else None
            ts, ents = frames_ents(game)
            if args.dump_frame:
                print("frame0 type:", type(game["frames"][0]).__name__)
                print("frame0:", json.dumps(game["frames"][0], ensure_ascii=False)[:600])
                return 0
            res, units = analyze_game(label, engine, ts, ents)
            if res is None:
                print(f"  [{label}] 解析失败: {units}")
                continue
            if meta:
                res["meta"] = {k: meta.get(k) for k in ("a_id", "b_id", "winner") if k in meta}
            res["winner"] = game.get("winner") if isinstance(game, dict) else None
            per_game.append(res)
            print(f"\n  --- 局 {gi} = {label}   n_units={res['n_units']} "
                  f"names={res['names']}  胜者={res['winner']}")
            print(f"      分道数量: {res['lane_counts']}")
            print(f"      桥廊最大同时在员: {res['max_simultaneous_in_corridor']}")
            print(f"      河带内越出桥廊帧数: {res['river_frames_outside_corridor']}")
            print("      各道：(n=出兵数 / n_crossed=过河数 / t_first_cross / t_median / t_last，秒)")
            for lane in ("left", "right"):
                row = res["per_lane"][lane]
                if not row.get("n"):
                    print(f"        {lane:5s}: n=0")
                    continue
                f = lambda v: "None" if v is None else f"{v:.1f}"
                print(f"        {lane:5s}: n={row['n']:2d} toward_river={row['n_reached_river']:2d} "
                      f"crossed={row['n_crossed']:2d} first={f(row['t_first_cross'])} "
                      f"median={f(row['t_median_cross'])} last={f(row['t_last_cross'])} "
                      f"spawn_x={row['spawn_x_range']}")
            print("      己方半场：按『最终走的那座桥的中心』算平均横向偏离（格）")
            for y in (2, 4, 6, 8, 10, 12, 14):
                row = res["own_half_convergence"].get(y)
                if row and row["mean_off_bridge"] is not None:
                    print(f"        y={y:2d}  n={row['n']:2d}  mean_x={row['mean_x']:6.2f}  "
                          f"mean|Δ桥心|={row['mean_off_bridge']:5.2f}")
            print("      横向行程完成度（0=还在出兵 x，1=已到桥心）")
            for y in (6, 10, 14):
                row = res["lateral_progress_to_bridge"][y]
                print(f"        y={y:2d}  n={row['n']:2d}  median={row['median']}  mean={row['mean']}")
            print("      过桥后（混乱度）：x 方向反转次数 / y 方向反转次数 / x 跨度 / y 推进")
            for lane in ("left", "right"):
                row = res["post_cross"][lane]
                if not row.get("n"):
                    print(f"        {lane:5s}: n=0")
                    continue
                print(f"        {lane:5s}: n={row['n']:2d} median_x_rev={row['median_x_dir_reversals']} "
                      f"mean_x_rev={row['mean_x_dir_reversals']} median_y_rev={row['median_y_dir_reversals']} "
                      f"median_x_span={row['median_x_span_after_cross']} "
                      f"median_y_gain={row['median_y_gain_after_cross']}")
        results[engine] = per_game

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=1)
        print(f"\n[json] {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
