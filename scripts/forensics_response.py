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
    intercepts = []          # 拦截口径：落点是否在 敌军→我方塔 路径 4 格内
    engagements = []         # 接敌口径（9k 正式指标）：部署后 8s 内 5 格内敌我同框
    for g in games:
        frames = g["frames"]
        last_opp_play_t = -999.0
        in_threat = False
        threat_start = None
        for idx, fr in enumerate(frames):
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
                        if b[3] >= 20:
                            continue   # 幽灵动作：非法尝试，引擎未执行，不进空间统计
                        if p1_troops:
                            dist_near.append(min(abs(wx - fx) + abs(wy - fy)
                                                 for fx, fy in p1_troops))
                            # 接敌口径（9k 正式指标，用户防守语义）：部署后 8s 内
                            # 部署点 5 格内是否出现敌我 troop 同框（真交战才叫防守）
                            t0, eng = t, False
                            for j in range(idx + 1, len(frames)):
                                fj = frames[j]
                                if fj["t"] > t0 + 8.0:
                                    break
                                near = [e for e in fj.get("entities") or []
                                        if e[5] == "troop"
                                        and abs(float(e[1]) - wx) + abs(float(e[2]) - wy) < 5.0]
                                if any(int(e[4]) == 0 for e in near) and \
                                        any(int(e[4]) == 1 for e in near):
                                    eng = True
                                    break
                            engagements.append(1 if eng else 0)
                            # 拦截口径：落点是否挡在"最近过河敌军 → 我方最前塔"的
                            # 直线路径近旁（曼氏距离到线段 ≤4 格）。防守方落塔前
                            # 迎击（距敌远但在线上）是正确行为，"距敌 8 格"会误伤它。
                            fy_near = min(p1_troops,
                                          key=lambda p: abs(wx - p[0]) + abs(wy - p[1]))
                            tx, ty = 8.5, 6.0   # 我方公主塔前缘近似（y≈3-6.5）
                            vx, vy = tx - fy_near[0], ty - fy_near[1]
                            L2 = vx * vx + vy * vy
                            if L2 > 1e-9:
                                s = max(0.0, min(1.0, ((wx - fy_near[0]) * vx
                                                       + (wy - fy_near[1]) * vy) / L2))
                                px, py = fy_near[0] + s * vx, fy_near[1] + s * vy
                                if abs(wx - px) + abs(wy - py) <= 4.0:
                                    intercepts.append(1)
                                else:
                                    intercepts.append(0)
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
    if intercepts:
        ni = len(intercepts)
        print(f"拦截口径（落点在 敌→塔 路径4格内）: {sum(intercepts)}/{ni} = "
              f"{100.0*sum(intercepts)/ni:.1f}%")
    if engagements:
        ne = len(engagements)
        print(f"接敌口径（9k 正式指标：部署后8s内部署点5格内出现敌我交战）: "
              f"{sum(engagements)}/{ne} = {100.0*sum(engagements)/ne:.1f}%")


if __name__ == "__main__":
    import sys as _sys
    pat = _sys.argv[1] if len(_sys.argv) > 1 else \
        "src/clasher_new/runs/economy/replays/league_*.pkl"
    classify_games(load_games(pat),
                   f"9j 三层复训 economy_9j（v2 修正坐标）[{pat}]")
    old_games = load_games("runs/archive/economy_100k_v1_ungated/replays/league_*.pkl")
    classify_games(old_games, "上次无闸门 100k（v2 修正坐标）")
