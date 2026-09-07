# -*- coding: utf-8 -*-
"""取证 v2：单边堆牌 vs 对牌响应率（2026-09-08，坐标系已修正）。

引擎世界坐标口径（league_100000.pkl 实测 + belief_planner.py:53 注释互证）：
- P0 塔 y≈3-6.5（下半场 y 小），P1 塔 y≈25.5-29（上半场 y 大）；河 y≈16；
- P1 部队进攻方向 y 小 → "敌军过河威胁我方" = P1 troop y < 16；
- bundle（我方 P0 动作）= P0 本地网格 = 世界网格，世界 = (x+0.5, y+0.5)；
- opp_played 已是世界坐标。

分类口径（对每个我方 deploy 动作）：
- response_defense：场上有 P1 troop y<16（已过河进我方半场）；
- response_session：无过河敌军，但过去 5s 内对手有出牌；
- unilateral：以上都不满足。

对照：对手 deploy 的响应分类 = P0 troop y>16（我方过河威胁对手）。
"""
import pickle, sys, io, glob
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

RIVER = 16.0
RESP_WINDOW = 5.0


def load_games(pattern):
    games = []
    for p in sorted(glob.glob(pattern)):
        with open(p, "rb") as f:
            rep = pickle.load(f)
        games.extend(rep["games"])
    return games


def troop_list(fr):
    return [(float(e[1]), float(e[2])) for e in fr["entities"] if e[5] == "troop"]


def classify_games(games, label):
    my_cls = Counter()
    opp_cls = Counter()
    my_deploy_total = 0
    opp_deploy_total = 0
    def_frames_my = 0        # 敌军过河帧
    def_deploy_my = 0        # 其中有我方 deploy 的帧
    latencies = []           # 威胁开始 → 我方首次 deploy 延迟
    dist_near = []           # 威胁场景下落点到最近过河敌军距离
    for g in games:
        frames = g["frames"]
        last_opp_play_t = -999.0
        in_threat = False
        threat_start = None
        for fr in frames:
            t = fr["t"]
            tp = troop_list(fr)
            foes_crossed = any(y < RIVER for _, y in tp if True) if False else \
                any(y < RIVER for x, y in [(x, y) for x, y in tp]) if False else False
            # P1 troops: player==1
            p1_troops = [(x, y) for e in fr["entities"] if e[5] == "troop" and int(e[4]) == 1
                         for x, y in [(float(e[1]), float(e[2]))]]
            p0_troops = [(x, y) for e in fr["entities"] if e[5] == "troop" and int(e[4]) == 0
                         for x, y in [(float(e[1]), float(e[2]))]]
            foes_crossed = any(y < RIVER for _, y in p1_troops)
            my_crossed = any(y > RIVER for _, y in p0_troops)
            if fr["opp_played"]:
                last_opp_play_t = t
                for _ in fr["opp_played"]:
                    opp_deploy_total += 1
                    if my_crossed:
                        opp_cls["response_defense"] += 1
                    elif t - last_opp_play_t <= RESP_WINDOW:
                        opp_cls["response_session"] += 1
                    else:
                        opp_cls["unilateral"] += 1
            bundle = fr.get("bundle") or []
            my_deploys = [b for b in bundle if b[0] == "deploy"]
            if my_deploys:
                my_deploy_total += len(my_deploys)
                if foes_crossed:
                    my_cls["response_defense"] += len(my_deploys)
                    for b in my_deploys:
                        wx, wy = b[2] + 0.5, b[3] + 0.5
                        if p1_troops:
                            dist_near.append(min(abs(wx - fx) + abs(wy - fy)
                                                 for fx, fy in p1_troops))
                elif t - last_opp_play_t <= RESP_WINDOW:
                    my_cls["response_session"] += len(my_deploys)
                else:
                    my_cls["unilateral"] += len(my_deploys)
            if foes_crossed:
                def_frames_my += 1
                if my_deploys:
                    def_deploy_my += 1
            if foes_crossed and not in_threat:
                in_threat, threat_start = True, fr["t"]
            elif not foes_crossed and in_threat:
                in_threat = False
        # 延迟：逐威胁区间（简化：逐帧扫，敌军过河 t0 → 下一个我方 deploy 帧 t）
        in_threat = False
        threat_start = None
        for fr in frames:
            p1_troops = [(float(e[1]), float(e[2])) for e in fr["entities"]
                         if e[5] == "troop" and int(e[4]) == 1]
            foes_crossed = any(y < RIVER for _, y in p1_troops)
            if foes_crossed and not in_threat:
                in_threat, threat_start = True, fr["t"]
            elif not foes_crossed and in_threat:
                in_threat = False
            if in_threat and fr.get("bundle"):
                if any(b[0] == "deploy" for b in fr["bundle"]):
                    latencies.append(fr["t"] - threat_start)
                    in_threat = False   # 只记区间首个响应

    total_my = sum(my_cls.values()) or 1
    print(f"\n===== {label} ({len(games)} games) =====")
    print(f"我方 deploy {my_deploy_total} | 对手 deploy {opp_deploy_total}")
    for k in ("response_defense", "response_session", "unilateral"):
        print(f"  我方 {k:18s} {my_cls[k]:6d}  {100.0*my_cls[k]/total_my:5.1f}%")
    to = sum(opp_cls.values()) or 1
    for k in ("response_defense", "response_session", "unilateral"):
        print(f"  对手 {k:18s} {opp_cls[k]:6d}  {100.0*opp_cls[k]/to:5.1f}%")
    print(f"敌军过河帧 {def_frames_my}，其中有我方 deploy {def_deploy_my}"
          f"（防守投入率 {100.0*def_deploy_my/max(1,def_frames_my):.1f}%）")
    if latencies:
        latencies.sort()
        n = len(latencies)
        print(f"威胁→首次响应延迟: n={n} median={latencies[n//2]:.1f}s "
              f"p25={latencies[n//4]:.1f}s p75={latencies[3*n//4]:.1f}s")
    if dist_near:
        dist_near.sort()
        n = len(dist_near)
        print(f"威胁场景落点到最近过河敌军: n={n} median={dist_near[n//2]:.1f} "
              f"p25={dist_near[n//4]:.1f} p75={dist_near[3*n//4]:.1f} | "
              f">8格 {100.0*sum(1 for d in dist_near if d > 8)/n:.1f}%")


if __name__ == "__main__":
    new_games = load_games("src/clasher_new/runs/economy/replays/league_*.pkl")
    classify_games(new_games, "带闸门 100k 复训（v2 修正坐标）")
    old_games = load_games("runs/archive/economy_100k_v1_ungated/replays/league_*.pkl")
    classify_games(old_games, "上次无闸门 100k（v2 修正坐标）")
