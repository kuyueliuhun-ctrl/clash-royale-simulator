# -*- coding: utf-8 -*-
"""「刺客完美解满蓄力电磁炮」搜索脚本 v2（2026-09-10，含防御塔参与）。

场景（用户勘误：真实技巧里防御塔是击杀主力）：
  P1 红桥头部署 Sparky（引擎卡名 ZapMachine，englishName=Sparky，6 费电磁炮），
  它过桥走进 P0 蓝左塔射程（t≈4.4），蓝塔开始输出。
  刺客在 Sparky「满蓄力将发」（attack_cooldown 接近 0）时部署，
  利用冲刺无敌（0.8s）躲掉这一炮，贴身输出；蓝塔持续输出完成击杀。

判据（用户口径）：
  完美解 = 蓝方三塔 0 掉血 + 刺客存活（甚至自身存活）。
搜索参数：刺客部署 x/y（中轴偏左列附近）+ 部署时机（相对 Sparky 开炮）。

用法：
  python assassin_vs_sparky.py             # 扫部署点+固定时机
  python assassin_vs_sparky.py --timing    # 附带扫部署时机
  python assassin_vs_sparky.py --trace     # 打印一个案例的逐帧
"""
import io
import os
import sys
import argparse
import itertools

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
# T1-2：不再硬编码 `E:/clash-royale-simulator-main/...`（换机器/换目录即失效）。
# 本文件在 `scripts/` 下 ⇒ 仓库根 = 上两级目录 ⇒ 引擎源码在 `<root>/src/clasher_new`。
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "clasher_new"))

# T1-1 追加（2026-09-19）：UTF-8 兜底。本文件含 **GBK 编不出**的字符 ⇒ 无兜底时
# `print` 抛 UnicodeEncodeError（实测：test_m3_evo 因此产生 **10 个假失败**）。
# 用 T1-1 的**单一实现**；只用在入口脚本上（`rl/` 库模块不加 —— 库不该改宿主 stdout）。
from rl.io_bootstrap import force_utf8_stdout  # noqa: E402
force_utf8_stdout()

import battle as battle_mod
import player as player_mod
from core import Position

DECK = ['Knight', 'Arrows', 'Fireball', 'Musketeer', 'Giant',
        'Minions', 'MiniPekka', 'Skeletons']

SPARKY_POS = (3.5, 18.0)      # P1 左桥头
HORIZON = 35.0


def run_case(label, assassin_pos, deploy_at, horizon=HORIZON, trace=False):
    """Sparky 过桥 + 刺客 deploy_at 秒部署；返回 (塔损, 刺客存活, 刺客hp, Sparky存活)。"""
    bs = battle_mod.BattleState(
        player_mod.PlayerState(0, list(DECK), 10.0),
        player_mod.PlayerState(1, list(DECK), 10.0), card_level=11)
    bs.step(0.033)
    bs.players[1].cycle = ['ZapMachine'] + [c for c in bs.players[1].cycle
                                            if c != 'ZapMachine']
    assert bs.deploy_card(1, 'ZapMachine', Position(*SPARKY_POS)), "Sparky 部署失败"
    sparky = next(e for e in bs.entities.values() if e.name == 'ZapMachine')
    towers0 = (bs.entities[3], bs.entities[4], bs.entities[6])
    init_hp = {t.id: t.hp for t in towers0}

    assassin = None
    a_dash_t = None
    sparky_fires = []
    prev_cd = None
    tower_damage_to_sparky = 0.0
    last_tower_tgt = None

    for i in range(int(horizon * 60)):
        t = i / 60
        if assassin is None and t >= deploy_at and sparky.is_alive:
            a = battle_mod.Troop(bs.next_entity_id, Position(*assassin_pos), 0,
                                 'Assassin', bs)
            bs._spawn_entity(a)
            assassin = a
            if trace:
                print(f"[trace] t={t:.2f} 部署刺客 at {assassin_pos}")
        bs.step(1 / 60)
        cd = sparky.attack_cooldown
        if prev_cd is not None and cd > 3.5 and prev_cd < 0.2:
            sparky_fires.append(t)
            if trace:
                print(f"[trace] t={t:.2f} Sparky 开炮！刺客距={assassin.position.distance_to(sparky.position):.2f}"
                      f" 刺客无敌={assassin.invincible if assassin else '-'}")
        prev_cd = cd
        if assassin is not None and a_dash_t is None and assassin._dash_active:
            a_dash_t = t
            if trace:
                print(f"[trace] t={t:.2f} 刺客突进开始")
        if assassin is not None and not assassin.is_alive and trace:
            print(f"[trace] t={t:.2f} 刺客死亡（Sparky hp={sparky.hp:.0f}）")
        if not sparky.is_alive:
            if trace:
                print(f"[trace] t={t:.2f} Sparky 被击杀（刺客存活={assassin and assassin.is_alive}）")
            break

    tower_loss = sum(init_hp[t.id] - (t.hp if t.is_alive else 0)
                     for t in towers0)
    s_alive = assassin is not None and assassin.is_alive
    s_hp = assassin.hp if s_alive else 0
    sp_alive = sparky.is_alive
    return {"tower_loss": tower_loss, "assassin_alive": s_alive,
            "assassin_hp": s_hp, "sparky_alive": sp_alive,
            "a_dash_t": a_dash_t, "fires": sparky_fires}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--trace", action="store_true")
    args = ap.parse_args()

    print("—— 刺客完美解满蓄力电磁炮 v2（Sparky/ZapMachine, lv11, 蓝塔参与）——")
    print(f"Sparky @ {SPARKY_POS}（P1左桥头），过桥进蓝左塔射程后蓝塔输出\n")

    # 先定位 Sparky 过桥后的满蓄力窗口（开炮时刻）
    bs = battle_mod.BattleState(player_mod.PlayerState(0, list(DECK), 10.0),
                                player_mod.PlayerState(1, list(DECK), 10.0),
                                card_level=11)
    bs.step(0.033)
    bs.players[1].cycle = ['ZapMachine'] + [c for c in bs.players[1].cycle if c != 'ZapMachine']
    bs.deploy_card(1, 'ZapMachine', Position(*SPARKY_POS))
    sp = next(e for e in bs.entities.values() if e.name == 'ZapMachine')
    fires = []
    prev_cd = None
    for i in range(int(18 * 60)):
        bs.step(1 / 60)
        t = i / 60
        cd = sp.attack_cooldown
        if prev_cd is not None and cd > 3.5 and prev_cd < 0.2:
            fires.append(t)
        prev_cd = cd
        if not sp.is_alive:
            break
    print(f"Sparky 开炮时刻: {[round(x,2) for x in fires]}")
    print("（刺客应在每次开炮前 ~0.8s 内突进，用 0.8s 无敌躲炮）\n")

    # 扫描：刺客部署点 + 时机（相对第一炮前后）
    xs = [7.5, 8.5, 9.5, 10.5]
    ys = [9.5, 10.5, 11.5, 12.5, 13.5, 14.5]
    f0 = fires[0] if fires else 7.1
    if args.timing:
        deploy_times = [f0 - 2.2, f0 - 2.0, f0 - 1.7, f0 - 1.4, f0 - 1.1,
                        f0 - 0.9, f0 - 0.7, f0 - 0.5, f0 - 0.3, f0,
                        f0 + 0.3, f0 + 0.6, f0 + 1.0]
    else:
        deploy_times = [f0 - 2.0, f0 - 1.5, f0 - 1.0, f0 - 0.6, f0 - 0.3]

    results = []
    for (x, y), dt in itertools.product([(a, b) for a in xs for b in ys],
                                        deploy_times):
        r = run_case(f"({x},{y})@{dt:.2f}", (x, y), dt)
        results.append(((x, y), dt, r))

    perfect = [r for r in results if r[2]["tower_loss"] <= 0
               and r[2]["assassin_alive"]]
    alive_only = [r for r in results if r[2]["assassin_alive"]]
    zero_only = [r for r in results if r[2]["tower_loss"] <= 0]

    print(f"扫描 {len(results)} 配置：完美解 {len(perfect)} / 刺客存活 {len(alive_only)}"
          f" / 0塔损 {len(zero_only)}\n")

    if perfect:
        print("== 完美解（0 塔损 + 刺客存活）==")
        for (x, y), dt, r in perfect[:15]:
            print(f"  刺客({x},{y}) @ t={dt:.2f}: 塔损={r['tower_loss']:.0f} "
                  f"刺客存活 hp={r['assassin_hp']:.0f} "
                  f"Sparky={'存活' if r['sparky_alive'] else '被击杀'} "
                  f"突进={round(r['a_dash_t'],2) if r['a_dash_t'] else '-'}s "
                  f"开炮={[round(z,2) for z in r['fires']]}")
    elif zero_only:
        print("== 0 塔损但刺客阵亡 ==")
        for (x, y), dt, r in zero_only[:10]:
            print(f"  刺客({x},{y}) @ t={dt:.2f}: 塔损={r['tower_loss']:.0f} "
                  f"Sparky={'存活' if r['sparky_alive'] else '被击杀'}")
    else:
        print("== 无 0 塔损配置，最近 5 个 ==")
        for (x, y), dt, r in sorted(results, key=lambda z: z[2]["tower_loss"])[:5]:
            print(f"  刺客({x},{y}) @ t={dt:.2f}: 塔损={r['tower_loss']:.0f} "
                  f"刺客存活={r['assassin_alive']} "
                  f"Sparky={'存活' if r['sparky_alive'] else '被击杀'}")

    if args.trace and results:
        # 打印第一个完美解/最近配置的逐帧
        (x, y), dt, r = perfect[0] if perfect else sorted(
            results, key=lambda z: z[2]["tower_loss"])[0]
        print(f"\n—— 逐帧 trace：刺客({x},{y}) @ t={dt:.2f} ——")
        run_case("trace", (x, y), dt, trace=True)


if __name__ == "__main__":
    main()
