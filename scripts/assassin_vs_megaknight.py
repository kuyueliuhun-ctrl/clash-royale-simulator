# -*- coding: utf-8 -*-
"""「刺客解超级骑士」对卡实验（2026-09-09，用户技巧口径）。

官方技巧三要素：
  A. 用突进规避 MK 落地溅射（MegaKnightAppear：908@lv11、半径 2.2、击退 1.0）；
  B. MK 只打地面/移速慢（60）→ 3 费刺客（移速 90+突进 8.3 格/s）可以贴身且不被甩开；
  C. 把 MK 拉到两座公主塔射程交集（中线 x=9 上 y≤11.6 双塔同打）借塔输出。

实验设计（同种子对照）：
  E0 基线：MK 无防守推到塔；
  E1 站桩：MK 落地前在落点旁放刺客（会吃落地弹——反面教材）；
  E2 时序规避：MK 落地弹结算后才放刺客（躲过 908 溅射）；
  E3 拉扯：E2 基础上，刺客入场位在双塔交集一侧，把 MK 引到双塔同打区。
对比口径：MK 剩余血量（或击杀）、我方塔损、刺客存活、圣水交换。
"""
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
# T1-2：不再硬编码 `E:/clash-royale-simulator-main/...`（换机器/换目录即失效）。
# 本文件在 `scripts/` 下 ⇒ 仓库根 = 上两级目录 ⇒ 引擎源码在 `<root>/src/clasher_new`。
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "clasher_new"))

import battle as battle_mod
import player as player_mod
from core import Position

DECK = ['Knight', 'Arrows', 'Fireball', 'Musketeer', 'Giant',
        'Minions', 'MiniPekka', 'Skeletons']

# 双塔射程交集（P0）：塔3(3.5,6.5)/塔4(14.5,6.5) 射程 7.5+碰撞 → x=9 线 y≤12.6
DUAL_TOWER_Y = 11.5


def run_case(label, assassin_delay, assassin_pos, horizon=30.0):
    """MK P1 (14.5,18) 过桥攻 P0 左塔？→ 不，攻右路 (14.5,18) 向下；
    刺客在 assassin_delay 秒后于 assassin_pos 部署。"""
    bs = battle_mod.BattleState(
        player_mod.PlayerState(0, list(DECK), 10.0),
        player_mod.PlayerState(1, list(DECK), 10.0), card_level=11)
    bs.step(0.033)
    # P0 左塔半血模拟战斗损耗背景（统一）
    bs.players[1].cycle = ['MegaKnight'] + [c for c in bs.players[1].cycle
                                            if c != 'MegaKnight']
    assert bs.deploy_card(1, 'MegaKnight', Position(14.5, 18.0)), "MK 部署失败"
    mk = next(e for e in bs.entities.values() if e.name == 'MegaKnight')
    # P1 用 Zap 假设不存在；只跑单位交互。刺客部署
    deployed = False
    dash_start = None
    splash_eaten = 0.0
    for i in range(int(horizon * 60)):
        t = i / 60
        if not deployed and t >= assassin_delay:
            bs.players[0].cycle = ['MiniPekka'] + [c for c in bs.players[0].cycle
                                                   if c != 'MiniPekka']
            # 刺客（Assassin）：手牌首位化再部署
            bs.players[0].cycle = ['Assassin' if c == 'MiniPekka' else c
                                   for c in bs.players[0].cycle]
            # 直接构造（避免手牌 8 卡里没有 Assassin 的枚举问题）：
            a = battle_mod.Troop(bs.next_entity_id, Position(*assassin_pos), 0,
                                 'Assassin', bs)
            bs._spawn_entity(a)
            deployed = True
            assassin = a
        bs.step(1 / 60)
        if deployed and dash_start is None and assassin._dash_active:
            dash_start = t
    # 结算统计
    towers0 = (bs.entities[3], bs.entities[4], bs.entities[6])
    tower_loss = sum(3052 - (t.hp if t.is_alive else 0) for t in towers0)
    mk_alive = mk.is_alive
    mk_hp = mk.hp if mk_alive else 0
    a_alive = deployed and assassin.is_alive
    # MK 最终位置（是否进双塔区）
    in_dual = (mk.position.y <= DUAL_TOWER_Y + 1.0 and
               abs(mk.position.x - 9.0) < 7.0) if mk_alive else None
    print(f"{label}: MK={'存活 hp ' + format(int(mk_hp), ',') if mk_alive else '被击杀'}"
          f" | 塔损={tower_loss:.0f} | 刺客={'存活 hp ' + format(int(assassin.hp), ',') if a_alive else '阵亡'}"
          f" | MK末位=({mk.position.x:.1f},{mk.position.y:.1f}){' 双塔区✓' if in_dual else ''}"
          f" | 突进启动={f'{dash_start:.1f}s' if dash_start else '未'}")
    return {"mk_hp": mk_hp, "tower_loss": tower_loss,
            "assassin_alive": a_alive, "in_dual": in_dual}


if __name__ == "__main__":
    print("—— 「刺客解超级骑士」对照实验（lv11，30s 视界）——")
    print("（突进触发窗口 [3.5, 6.0] 格：部署点须离 MK 3.5+ 格才会突进贴脸）")
    r0 = run_case("E0 无防守基线        ", 999, (0, 0))
    r1 = run_case("E1 站桩(吃落地弹)    ", 0.2, (15.0, 16.5))
    r2 = run_case("E2 规避+突进(远距部署)", 2.5, (14.5, 11.5))
    r3 = run_case("E3 规避+拉扯双塔区   ", 2.5, (10.0, 12.0))
    print("\n—— 结论指标 ——")
    print(f"规避收益（E1 vs E2 刺客存活/输出差）: "
          f"E1 存活={r1['assassin_alive']} E2 存活={r2['assassin_alive']}")
    print(f"塔损对比: E0={max(r0['tower_loss'],0):.0f} E1={max(r1['tower_loss'],0):.0f} "
          f"E2={max(r2['tower_loss'],0):.0f} E3={max(r3['tower_loss'],0):.0f}（负=含塔反打回血口径）")
