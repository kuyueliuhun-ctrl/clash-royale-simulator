# -*- coding: utf-8 -*-
"""泛化对抗搜索器：敌方行进 0.1 格评估 × 我方整格部署，求最小损失。

场景（用户口径 2026-09-10）：
  - 敌方单位从桥头沿引擎寻路路径行进，路径按 0.1 格采样成检查点；
  - 我方单位在整格（格子中心）部署，部署时机 = 敌方到达某检查点位置时；
  - 敌我同步对抗：每个「时机 × 我方部署位」跑完整推演，度量损失；
  - 目标是最小损失（非完美解）：损失 = 我方塔损×w_tower + 我方单位价值损失
    + 敌塔损×w_enemy_tower + 圣水交换×w_elixir。

支持场景：
  -v 1v1   敌1张 vs 我1张（全扫：检查点时机 × 整格部署位）
  -v 2v1   敌2张(异步) vs 我1张（搜索规模大，采样）
  -v 2v2   敌2张 vs 我2张（更大，启发式/并行）

输出：
  1) 配置清单：按损失升序，最优解在前
  2) 部署点热图：每个整格部署位的期望损失（模型费用不足时选次优）

用法：
  python duel_search.py --enemy MegaKnight --mine Assassin --mode 1v1
  python duel_search.py --enemy Giant --mine MiniPekka --mode 2v1 --enemy2 Musketeer
"""
import io
import sys
import os
import time
import argparse
import itertools
import json
from collections import defaultdict

# 注意：stdout 重包装放在 main 内（__main__），避免 import 时污染调用方 stdout
# （spawn 子进程/外部 import 会被关掉的 TextIOWrapper 破坏）
sys.path.insert(0, r"E:/clash-royale-simulator-main/src/clasher_new")

import battle as battle_mod
import player as player_mod
from core import Position
from arena import TileGrid

DECK = ['Knight', 'Arrows', 'Fireball', 'Musketeer', 'Giant',
        'Minions', 'MiniPekka', 'Skeletons']

# 我方整格部署位（P0 半场格子中心）
def _own_deploy_cells():
    tg = TileGrid()
    cells = []
    for y in [i + 0.5 for i in range(15)]:
        for x in [i + 0.5 for i in range(18)]:
            if tg.can_deploy_at(Position(x, y), 0):
                cells.append((x, y))
    return cells

OWN_CELLS = _own_deploy_cells()
ENEMY_START = (3.5, 18.0)      # 敌左桥头（P1）
HORIZON = 35.0


# ============ 敌方时间轴检查点（时间驱动） ============
def enemy_timeline(enemy_card, start=None, horizon=HORIZON, dt=0.1):
    """预演敌方单卡，按固定时间步 dt 记录时刻表。

    返回 [(t, x, y, phase), ...]：
      - t: 时刻
      - x, y: 敌方位置
      - phase: 敌方当前状态（walk/移动、jump_prep/起跳预备、jump_air/空中、landed/落地、idle/不动）

    时间驱动而非位置采样：超骑起跳预备阶段位置不动，但时间在走；
    按时间采样才能覆盖 prep/air 这些「位置不变但阶段变化」的关键窗口
    （用户 2026-09-10 指出：0.1 格位置采样会漏掉起跳不动的阶段，
    小骷髅开主塔这类卡超骑起跳窗口的操作就搜不出来）。"""
    bs = battle_mod.BattleState(
        player_mod.PlayerState(0, list(DECK), 10.0),
        player_mod.PlayerState(1, list(DECK), 10.0), card_level=11)
    bs.step(0.033)
    e = battle_mod.Troop(bs.next_entity_id, Position(*(start or ENEMY_START)),
                         1, enemy_card, bs)
    bs._spawn_entity(e)
    holder = e.entity_holder
    timeline = []
    steps = int(horizon / dt)
    for i in range(steps):
        t = i * dt
        # 记录当前阶段
        j = getattr(holder, '_mk_jump', None)
        if j is not None:
            phase = f"jump_{j['phase']}"
        elif e.jumping_across_river:
            phase = "river_jump"
        elif not e.is_alive:
            phase = "dead"
        else:
            phase = "walk"
        timeline.append((round(t, 3), round(e.position.x, 3),
                         round(e.position.y, 3), phase))
        # 推进到下一时刻
        remaining = dt
        while remaining > 1e-9 and e.is_alive:
            step = min(remaining, 1 / 60)
            bs.step(step)
            remaining -= step
        if not e.is_alive:
            # 补录死亡后时间点（位置不变）
            for k in range(i + 1, steps):
                timeline.append((round(k * dt, 3), round(e.position.x, 3),
                                 round(e.position.y, 3), "dead"))
            break
        if e.position.y < 5.0:
            # 已推到塔后，补录到 horizon
            for k in range(i + 1, steps):
                timeline.append((round(k * dt, 3), round(e.position.x, 3),
                                 round(e.position.y, 3), phase))
            break
    return timeline


def enemy_path_checkpoints(enemy_card, start=None, horizon=HORIZON, step=0.1):
    """兼容旧接口：位置检查点（每 step 格一个 y），内部转时间轴。"""
    tl = enemy_timeline(enemy_card, start, horizon)
    cps = {}
    for t, x, y, phase in tl:
        yt = round(y * 10) / 10
        if yt not in cps:
            cps[yt] = t
    return sorted(cps.items())


def _loss_metrics(bs, my_card, enemy_card, towers0, my_ent, enemy_ent):
    """度量一次对抗的损失。"""
    towers = (bs.entities[3], bs.entities[4], bs.entities[6])
    my_tower_loss = sum(towers0[t.id] - (t.hp if t.is_alive else 0)
                        for t in towers)
    # 敌方塔损（进攻价值）
    etowers = (bs.entities[1], bs.entities[2], bs.entities[5])
    et0 = {t.id: t.hp for t in etowers}
    # 注意：etowers 的 hp 在克隆时已变，需传初始
    enemy_tower_loss = 0.0
    for t in etowers:
        base = et0.get(t.id, t.hp)
        enemy_tower_loss += base - (t.hp if t.is_alive else 0)
    my_alive = my_ent is not None and my_ent.is_alive
    enemy_alive = enemy_ent is not None and enemy_ent.is_alive
    return {
        "my_tower_loss": my_tower_loss,
        "enemy_tower_loss": enemy_tower_loss,
        "my_unit_alive": my_alive,
        "enemy_unit_alive": enemy_alive,
    }


def run_duel(enemy_card, my_card, my_pos, deploy_at, horizon=HORIZON):
    """敌单卡从桥头走，deploy_at 时刻在 my_pos 部署我方单位，推演到 horizon。"""
    import copy
    bs = battle_mod.BattleState(
        player_mod.PlayerState(0, list(DECK), 10.0),
        player_mod.PlayerState(1, list(DECK), 10.0), card_level=11)
    bs.step(0.033)
    e = battle_mod.Troop(bs.next_entity_id, Position(*ENEMY_START),
                         1, enemy_card, bs)
    bs._spawn_entity(e)
    towers0 = {t.id: t.hp for t in (bs.entities[3], bs.entities[4], bs.entities[6])}
    et0 = {t.id: t.hp for t in (bs.entities[1], bs.entities[2], bs.entities[5])}
    m = None
    for i in range(int(horizon * 60)):
        t = i / 60
        if m is None and t >= deploy_at and e.is_alive:
            m = battle_mod.Troop(bs.next_entity_id, Position(*my_pos),
                                 0, my_card, bs)
            bs._spawn_entity(m)
        bs.step(1 / 60)
        if not e.is_alive and (m is None or not m.is_alive):
            break
    towers = (bs.entities[3], bs.entities[4], bs.entities[6])
    my_tower_loss = sum(towers0[t.id] - (t.hp if t.is_alive else 0)
                        for t in towers)
    etowers = (bs.entities[1], bs.entities[2], bs.entities[5])
    enemy_tower_loss = sum(et0[t.id] - (t.hp if t.is_alive else 0)
                           for t in etowers)
    return {
        "my_tower_loss": my_tower_loss,
        "enemy_tower_loss": enemy_tower_loss,
        "my_alive": m is not None and m.is_alive,
        "enemy_alive": e.is_alive,
    }


def run_duel_2v1(enemy1, enemy2, enemy2_pos, enemy2_at, my_card, my_pos,
                 my_at, horizon=HORIZON):
    """敌双卡异步 vs 我单卡。

    enemy1 从桥头出发，enemy2 在 enemy2_at 时刻于 enemy2_pos 部署，
    我方在 my_at 时刻于 my_pos 部署。推演到 horizon。
    """
    bs = battle_mod.BattleState(
        player_mod.PlayerState(0, list(DECK), 10.0),
        player_mod.PlayerState(1, list(DECK), 10.0), card_level=11)
    bs.step(0.033)
    e1 = battle_mod.Troop(bs.next_entity_id, Position(*ENEMY_START),
                          1, enemy1, bs)
    bs._spawn_entity(e1)
    e2 = None
    m = None
    towers0 = {t.id: t.hp for t in (bs.entities[3], bs.entities[4], bs.entities[6])}
    et0 = {t.id: t.hp for t in (bs.entities[1], bs.entities[2], bs.entities[5])}
    for i in range(int(horizon * 60)):
        t = i / 60
        if e2 is None and t >= enemy2_at and e1.is_alive:
            e2 = battle_mod.Troop(bs.next_entity_id, Position(*enemy2_pos),
                                  1, enemy2, bs)
            bs._spawn_entity(e2)
        if m is None and t >= my_at and (e1.is_alive or (e2 is not None and e2.is_alive)):
            m = battle_mod.Troop(bs.next_entity_id, Position(*my_pos),
                                 0, my_card, bs)
            bs._spawn_entity(m)
        bs.step(1 / 60)
        # 结束条件：双方都死了
        e2_alive = e2 is not None and e2.is_alive
        if not e1.is_alive and not e2_alive and (m is None or not m.is_alive):
            break
    towers = (bs.entities[3], bs.entities[4], bs.entities[6])
    my_tower_loss = sum(towers0[t.id] - (t.hp if t.is_alive else 0)
                        for t in towers)
    etowers = (bs.entities[1], bs.entities[2], bs.entities[5])
    enemy_tower_loss = sum(et0[t.id] - (t.hp if t.is_alive else 0)
                           for t in etowers)
    return {
        "my_tower_loss": my_tower_loss,
        "enemy_tower_loss": enemy_tower_loss,
        "my_alive": m is not None and m.is_alive,
        "enemy_alive": e1.is_alive or (e2 is not None and e2.is_alive),
    }


def _run_2v1_worker(args):
    enemy1, enemy2, e2_pos, e2_at, my_card, my_pos, my_at = args
    r = run_duel_2v1(enemy1, enemy2, e2_pos, e2_at, my_card, my_pos, my_at)
    return (my_pos, my_at, e2_pos, e2_at, r)


def run_duel_2v2(enemy1, enemy2, e2_pos, e2_at, my1, my2,
                 my1_pos, my1_at, my2_pos, my2_at, horizon=HORIZON):
    """敌双卡异步 vs 我双卡异步（我方二张不一定同时释放）。"""
    bs = battle_mod.BattleState(
        player_mod.PlayerState(0, list(DECK), 10.0),
        player_mod.PlayerState(1, list(DECK), 10.0), card_level=11)
    bs.step(0.033)
    e1 = battle_mod.Troop(bs.next_entity_id, Position(*ENEMY_START),
                          1, enemy1, bs)
    bs._spawn_entity(e1)
    e2 = None
    m1 = None
    m2 = None
    towers0 = {t.id: t.hp for t in (bs.entities[3], bs.entities[4], bs.entities[6])}
    et0 = {t.id: t.hp for t in (bs.entities[1], bs.entities[2], bs.entities[5])}
    for i in range(int(horizon * 60)):
        t = i / 60
        if e2 is None and t >= e2_at and e1.is_alive:
            e2 = battle_mod.Troop(bs.next_entity_id, Position(*e2_pos),
                                  1, enemy2, bs)
            bs._spawn_entity(e2)
        if m1 is None and t >= my1_at:
            m1 = battle_mod.Troop(bs.next_entity_id, Position(*my1_pos),
                                  0, my1, bs)
            bs._spawn_entity(m1)
        if m2 is None and t >= my2_at:
            m2 = battle_mod.Troop(bs.next_entity_id, Position(*my2_pos),
                                  0, my2, bs)
            bs._spawn_entity(m2)
        bs.step(1 / 60)
        e2_alive = e2 is not None and e2.is_alive
        m1_alive = m1 is not None and m1.is_alive
        m2_alive = m2 is not None and m2.is_alive
        if not e1.is_alive and not e2_alive and not m1_alive and not m2_alive:
            break
    towers = (bs.entities[3], bs.entities[4], bs.entities[6])
    my_tower_loss = sum(towers0[t.id] - (t.hp if t.is_alive else 0)
                        for t in towers)
    etowers = (bs.entities[1], bs.entities[2], bs.entities[5])
    enemy_tower_loss = sum(et0[t.id] - (t.hp if t.is_alive else 0)
                           for t in etowers)
    my_alive = (m1 is not None and m1.is_alive) or (m2 is not None and m2.is_alive)
    return {
        "my_tower_loss": my_tower_loss,
        "enemy_tower_loss": enemy_tower_loss,
        "my_alive": my_alive,
        "enemy_alive": e1.is_alive or (e2 is not None and e2.is_alive),
    }


def _run_2v2_worker(args):
    (e1, e2, e2p, e2at, m1, m2, m1p, m1at, m2p, m2at) = args
    r = run_duel_2v2(e1, e2, e2p, e2at, m1, m2, m1p, m1at, m2p, m2at)
    return (m1p, m1at, m2p, m2at, e2p, e2at, r)


def search_2v2(enemy1, enemy2, my1, my2, workers=None,
               max_e2=None, max_my=None, guide_topk=None):
    """敌2我2：双卡对抗，热图引导。

    我双卡部署位从 敌1我1 热图 TopK 取 2 组合，时机错开（二张不一定同时）。
    """
    if guide_topk is None:
        guide_topk = 16
    _hm = None
    if guide_topk:
        try:
            r1, n1, _ = search_1v1(enemy1, my1, max_timings=6, workers=workers)
            _hm = heatmap(r1)
            top_pos = [p for p, _ in sorted(_hm.items(), key=lambda kv: kv[1])
                       [:guide_topk]]
        except Exception as e:
            print(f"[warn] 热图引导失败，回退全位: {e}")
            top_pos = OWN_CELLS
    else:
        top_pos = OWN_CELLS
    tl1 = enemy_timeline(enemy1)
    # 敌方2 时机（早期异步）
    if max_e2 is None:
        max_e2 = 4
    e2_times = [t for t, x, y, phase in tl1
                if phase != "dead" and t <= 8.0]
    if len(e2_times) > max_e2:
        step = len(e2_times) / max_e2
        e2_times = [e2_times[int(i * step)] for i in range(max_e2)]
    e2_positions = [(3.5, 17.5), (8.5, 17.5), (14.5, 17.5),
                    (3.5, 18.5), (8.5, 18.5), (14.5, 18.5)]
    # 我方双卡部署位组合（TopK 取 2）+ 时机错开
    if max_my is None:
        max_my = 4
    my_times = [t for t, x, y, phase in tl1
                if phase != "dead" and t <= 8.0]
    if len(my_times) > max_my:
        step = len(my_times) / max_my
        my_times = [my_times[int(i * step)] for i in range(max_my)]
    # 我方两卡：位置从 top_pos 取 2（允许同点不同时），时机从 my_times 取 2
    import random
    my_combos = []
    for p1 in top_pos:
        for p2 in top_pos:
            for t1 in my_times:
                for t2 in my_times:
                    my_combos.append((p1, t1, p2, t2))
    # 任务
    tasks = []
    for e2p in e2_positions:
        for e2at in e2_times:
            for (p1, t1, p2, t2) in my_combos:
                tasks.append((enemy1, enemy2, e2p, e2at,
                              my1, my2, p1, t1, p2, t2))
    t0 = time.time()
    if workers and workers > 1:
        outs = _pmap(_run_2v2_worker, tasks, workers)
    else:
        outs = [_run_2v2_worker(t) for t in tasks]
    dt_elapsed = time.time() - t0
    results = []
    for (m1p, m1at, m2p, m2at, e2p, e2at, r) in outs:
        results.append({
            "my1_pos": m1p, "my1_at": m1at, "my2_pos": m2p, "my2_at": m2at,
            "e2_pos": e2p, "e2_at": e2at,
            "my_tower_loss": r["my_tower_loss"],
            "enemy_tower_loss": r["enemy_tower_loss"],
            "my_alive": r["my_alive"], "enemy_alive": r["enemy_alive"],
            "loss": total_loss(r),
        })
    return results, len(results), dt_elapsed


def search_2v1(enemy1, enemy2, my_card, my_at_grid=None,
               e2_timings=None, workers=None, max_e2=None, max_my=None,
               guide_topk=None):
    """敌2我1：敌双卡异步 vs 我单卡（热图引导，避免全扫不可行）。

    enemy1 从桥头走；enemy2 在 e2_timings 异步部署；我方在 (t,pos) 部署。
    搜索规模控制：
      - 先跑 敌1我1 全扫拿热图（guide_topk 部署位）
      - 2v1 只扫热图 Top-K 部署位，避免 224 全位 × 敌2组合爆炸
    Windows spawn 实测：>512 任务批量并行会 WinError，分小批（_pmap ≤256）。
    """
    # 敌1我1 热图引导（默认取 TopK 部署位）
    if guide_topk is None:
        guide_topk = 32
    _hm = None
    if guide_topk:
        try:
            r1, n1, _ = search_1v1(enemy1, my_card, max_timings=6, workers=workers)
            _hm = heatmap(r1)
            top_pos = [p for p, _ in sorted(_hm.items(), key=lambda kv: kv[1])
                       [:guide_topk]]
        except Exception as e:
            print(f"[warn] 热图引导失败，回退全位: {e}")
            top_pos = OWN_CELLS
    else:
        top_pos = OWN_CELLS
    # 默认 enemy2 时机 = enemy1 时间轴早期（异步跟牌）
    tl1 = enemy_timeline(enemy1)
    if e2_timings is None:
        e2_cands = [(t, (x, y)) for t, x, y, phase in tl1
                    if phase != "dead" and t <= 8.0]
        if not e2_cands:
            e2_cands = [(t, (x, y)) for t, x, y, phase in tl1 if phase != "dead"]
        if max_e2 and len(e2_cands) > max_e2:
            step = len(e2_cands) / max_e2
            e2_cands = [e2_cands[int(i * step)] for i in range(max_e2)]
    else:
        e2_cands = e2_timings
    # enemy2 部署位置：桥头附近几个整格
    e2_positions = [(3.5, 17.5), (8.5, 17.5), (14.5, 17.5),
                    (3.5, 18.5), (8.5, 18.5), (14.5, 18.5)]
    # 我方时机：从 enemy1 时间轴早期取，部署位 = 热图 TopK 整格
    if my_at_grid is None:
        my_times = [t for t, x, y, phase in tl1
                    if phase != "dead" and t <= 8.0]
        if not my_times:
            my_times = [t for t, x, y, phase in tl1 if phase != "dead"]
        if max_my and len(my_times) > max_my:
            step = len(my_times) / max_my
            my_times = [my_times[int(i * step)] for i in range(max_my)]
        my_cands = [(t, pos) for t in my_times for pos in top_pos]
    else:
        my_cands = my_at_grid
    # 展开任务
    tasks = []
    for e2_at, e2_pos in e2_cands:
        for e2p in e2_positions:
            for my_at, myp in my_cands:
                tasks.append((enemy1, enemy2, e2p, e2_at, my_card, myp, my_at))
    t0 = time.time()
    if workers and workers > 1:
        outs = _pmap(_run_2v1_worker, tasks, workers)
    else:
        outs = [_run_2v1_worker(t) for t in tasks]
    dt_elapsed = time.time() - t0
    results = []
    for (myp, my_at, e2p, e2_at, r) in outs:
        results.append({
            "my_pos": myp, "my_at": my_at,
            "e2_pos": e2p, "e2_at": e2_at,
            "my_tower_loss": r["my_tower_loss"],
            "enemy_tower_loss": r["enemy_tower_loss"],
            "my_alive": r["my_alive"], "enemy_alive": r["enemy_alive"],
            "loss": total_loss(r),
        })
    return results, len(results), dt_elapsed


def total_loss(r, w_tower=1.0, w_unit=0.5, w_enemy_tower=1.5, w_elixir=0.0):
    """加权总损失（最小化）。"""
    my_unit_pen = 0.0 if r["my_alive"] else 0.5
    return (w_tower * r["my_tower_loss"]
            + w_unit * my_unit_pen
            - w_enemy_tower * r["enemy_tower_loss"])


def _run_1v1_worker(args):
    """多进程 worker：单次 (enemy, mine, pos, dt) 推演。"""
    enemy_card, my_card, pos, dt = args
    r = run_duel(enemy_card, my_card, pos, dt)
    return (pos, dt, r)


def _pmap(fn, tasks, workers):
    """流式并行：分批 imap，避免 Windows spawn 一次性传大量任务触发句柄/管道限制。

    Windows spawn 实测：单批 5376 任务 BrokenPipe/PermissionError，
    分小批（≤512）流式传输稳定。"""
    import multiprocessing as mp
    out = []
    BATCH = 256
    with mp.Pool(workers) as pool:
        for i in range(0, len(tasks), BATCH):
            batch = tasks[i:i + BATCH]
            out.extend(pool.map(fn, batch, chunksize=max(1, len(batch) // workers // 2)))
    return out


def search_1v1(enemy_card, my_card, time_every=0.1, phases=None,
               max_timings=None, workers=None):
    """敌1我1全扫：时间轴检查点 × 我方整格部署位。

    - time_every: 时间检查点间隔（秒），默认 0.1s
    - phases: 仅在这些阶段部署（如 ['jump_prep']）；None=全部
    - max_timings: 时间点数量上限（None=全部，用于性能控制）
    - workers: 并行进程数（None=CPU核数-2）
    """
    tl = enemy_timeline(enemy_card, dt=time_every)
    timings = []
    for t, x, y, phase in tl:
        if phases and not any(phase.startswith(p) for p in phases):
            continue
        if phase == "dead":
            continue
        timings.append((t, x, y, phase))
    if max_timings and len(timings) > max_timings:
        step = len(timings) / max_timings
        timings = [timings[int(i * step)] for i in range(max_timings)]
    # 展开 (时机 × 部署位) 任务
    tasks = [(enemy_card, my_card, pos, dt)
             for (dt, ex, ey, phase), pos in itertools.product(timings, OWN_CELLS)]
    t0 = time.time()
    if workers and workers > 1:
        outs = _pmap(_run_1v1_worker, tasks, workers)
    else:
        outs = [_run_1v1_worker(t) for t in tasks]
    dt_elapsed = time.time() - t0
    # 重映射时机元数据（timing→enemy_pos/phase）
    timing_meta = {dt: (ex, ey, phase) for dt, ex, ey, phase in timings}
    results = []
    for (pos, dt, r) in outs:
        ex, ey, phase = timing_meta[dt]
        results.append({
            "deploy_pos": pos, "deploy_at": dt, "enemy_pos": (ex, ey),
            "enemy_phase": phase,
            "my_tower_loss": r["my_tower_loss"],
            "enemy_tower_loss": r["enemy_tower_loss"],
            "my_alive": r["my_alive"], "enemy_alive": r["enemy_alive"],
            "loss": total_loss(r),
        })
    return results, len(results), dt_elapsed


def heatmap(results):
    """部署点热图：每个整格位的期望损失（所有时机平均）。"""
    hm = defaultdict(list)
    for r in results:
        hm[r["deploy_pos"]].append(r["loss"])
    out = {pos: sum(v) / len(v) for pos, v in hm.items()}
    return out


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--enemy", default="MegaKnight", help="敌方卡")
    ap.add_argument("--mine", default="Assassin", help="我方卡")
    ap.add_argument("--mode", default="1v1", choices=["1v1", "2v1", "2v2"])
    ap.add_argument("--enemy2", default=None, help="敌方第二张卡")
    ap.add_argument("--mine2", default=None, help="我方第二张卡")
    ap.add_argument("--time-every", type=float, default=0.1, help="时间检查点间隔(秒)")
    ap.add_argument("--max-timings", type=int, default=30, help="时间点数量上限(None=全部)")
    ap.add_argument("--phases", default=None, help="阶段过滤，逗号分隔(如 jump_prep,jump_air)")
    ap.add_argument("--out", default=None, help="输出 JSON 路径")
    ap.add_argument("--workers", type=int, default=None, help="并行进程数(默认CPU-2)")
    args = ap.parse_args()

    print(f"—— 泛化对抗搜索：敌{args.enemy} vs 我{args.mine} [{args.mode}] ——")
    print(f"我方整格部署位: {len(OWN_CELLS)} 个，敌方时间检查点 {args.time_every}s"
          f"{'，阶段过滤 '+args.phases if args.phases else ''}"
          f"{'，上限 '+str(args.max_timings)+' 时间点' if args.max_timings else ''}\n")

    if args.mode == "1v1":
        phases = args.phases.split(",") if args.phases else None
        workers = args.workers or (os.cpu_count() - 2 if os.cpu_count() else 4)
        results, n, el = search_1v1(args.enemy, args.mine,
                                    time_every=args.time_every,
                                    phases=phases,
                                    max_timings=args.max_timings,
                                    workers=workers)
        print(f"扫 {n} 配置，耗时 {el:.1f}s（{n/el:.0f} 局/s）\n")
        results.sort(key=lambda r: r["loss"])
        print("== 最优解（损失升序前 10）==")
        for r in results[:10]:
            print(f"  我放{r['deploy_pos']} @t={r['deploy_at']:.2f}s"
                  f"[{r['enemy_phase']}] 敌位{r['enemy_pos']}: "
                  f"loss={r['loss']:.1f} 我塔损={r['my_tower_loss']:.0f} "
                  f"敌塔损={r['enemy_tower_loss']:.0f} "
                  f"我单位{'活' if r['my_alive'] else '亡'} "
                  f"敌单位{'活' if r['enemy_alive'] else '亡'}")
        # 热图
        hm = heatmap(results)
        print("\n== 部署点热图（期望损失最低 10 位）==")
        for pos, loss in sorted(hm.items(), key=lambda kv: kv[1])[:10]:
            print(f"  {pos}: 期望loss={loss:.1f}")
        # 最差 5 位
        print("\n== 部署点热图（最差 5 位）==")
        for pos, loss in sorted(hm.items(), key=lambda kv: -kv[1])[:5]:
            print(f"  {pos}: 期望loss={loss:.1f}")
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump({"results": results, "heatmap": {f"{k[0]},{k[1]}": v
                          for k, v in hm.items()}}, f, ensure_ascii=False, indent=1)
            print(f"\n[已写 {args.out}]")
    elif args.mode == "2v1":
        if not args.enemy2:
            print("2v1 需要 --enemy2 指定敌方第二张卡")
            return
        workers = args.workers or (os.cpu_count() - 2 if os.cpu_count() else 4)
        results, n, el = search_2v1(args.enemy, args.enemy2, args.mine,
                                    workers=workers,
                                    max_e2=args.max_timings,
                                    max_my=args.max_timings)
        print(f"扫 {n} 配置，耗时 {el:.1f}s（{n/el:.0f} 局/s）\n")
        results.sort(key=lambda r: r["loss"])
        print(f"== 敌{args.enemy}+{args.enemy2} vs 我{args.mine} 最优解（损失升序前 10）==")
        for r in results[:10]:
            print(f"  我放{r['my_pos']} @t={r['my_at']:.2f}s | "
                  f"敌2={args.enemy2}@{r['e2_pos']} @t={r['e2_at']:.2f}s: "
                  f"loss={r['loss']:.1f} 我塔损={r['my_tower_loss']:.0f} "
                  f"我单位{'活' if r['my_alive'] else '亡'} "
                  f"敌单位{'活' if r['enemy_alive'] else '亡'}")
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump({"results": results}, f, ensure_ascii=False, indent=1)
            print(f"\n[已写 {args.out}]")
    elif args.mode == "2v2":
        if not args.enemy2 or not args.mine2:
            print("2v2 需要 --enemy2 和 --mine2")
            return
        workers = args.workers or (os.cpu_count() - 2 if os.cpu_count() else 4)
        # 2v2 空间大：热图 TopK 小 + 时机少，避免任务爆炸
        results, n, el = search_2v2(args.enemy, args.enemy2,
                                    args.mine, args.mine2,
                                    workers=workers,
                                    max_e2=min(args.max_timings or 3, 3),
                                    max_my=min(args.max_timings or 2, 2),
                                    guide_topk=4)
        print(f"扫 {n} 配置，耗时 {el:.1f}s（{n/el:.0f} 局/s）\n")
        results.sort(key=lambda r: r["loss"])
        print(f"== 敌{args.enemy}+{args.enemy2} vs 我{args.mine}+{args.mine2}"
              f" 最优解（损失升序前 10）==")
        for r in results[:10]:
            print(f"  我1={args.mine}@{r['my1_pos']} t={r['my1_at']:.1f}s "
                  f"我2={args.mine2}@{r['my2_pos']} t={r['my2_at']:.1f}s | "
                  f"敌2={args.enemy2}@{r['e2_pos']} t={r['e2_at']:.1f}s: "
                  f"loss={r['loss']:.1f} 我塔损={r['my_tower_loss']:.0f} "
                  f"我单位{'活' if r['my_alive'] else '亡'} "
                  f"敌单位{'活' if r['enemy_alive'] else '亡'}")
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump({"results": results}, f, ensure_ascii=False, indent=1)
            print(f"\n[已写 {args.out}]")


if __name__ == "__main__":
    main()
