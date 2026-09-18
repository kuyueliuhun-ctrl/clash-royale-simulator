# -*- coding: utf-8 -*-
"""用户指定测试（2026-09-09 修订版）：完美解 = 蓝方塔不掉血。

用户机制（2026-09-09）：
  - 普通超骑有冲刺跳（起跳距离 3.5-5.0 格，dashMinRange/dashMaxRange）；
  - 超骑起跳有滞空（空中不无敌、仍可被攻击），落地才结算溅射；
  - 刺客放置瞬间超骑开始起跳；超骑半空中刺客部署前摇结束、进入冲刺（无敌）；
  - 超骑落地时刺客冲刺正好结束或稍晚 → 规避落地溅射。
  - 超骑吃到的伤害：进单塔射程的塔伤 + 刺客冲刺伤 + 刺客后续伤害 + 双塔伤害。

配置：
  - MK（红 P1）左路桥头部署 (3.5, 18.0)，下桥走 2.5 格（y≈13.5）时蓝方 P0 放刺客；
  - 刺客：中轴偏左列 x=8.5，自河边缘（y=15）往下第 5 格 (8.5,10.5) / 第 6 格 (8.5,9.5)。
判据：蓝方三塔损失 = 0 为完美解。
"""
import io
import os
import sys

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

MK_DEPLOY = (3.5, 18.0)
TRIGGER_Y = 13.5


def run_case(label, assassin_pos, horizon=45.0):
    bs = battle_mod.BattleState(
        player_mod.PlayerState(0, list(DECK), 10.0),
        player_mod.PlayerState(1, list(DECK), 10.0), card_level=11)
    bs.step(0.033)
    bs.players[1].cycle = ['MegaKnight'] + [c for c in bs.players[1].cycle
                                            if c != 'MegaKnight']
    assert bs.deploy_card(1, 'MegaKnight', Position(*MK_DEPLOY)), "MK 部署失败"
    mk = next(e for e in bs.entities.values() if e.name == 'MegaKnight')
    mkh = mk.entity_holder
    towers0 = (bs.entities[3], bs.entities[4], bs.entities[6])
    init_hp = {t.id: t.hp for t in towers0}
    assassin = None
    a_deploy_t = None
    a_dash_t = None
    mk_jump_t = None
    mk_jump_tgt = None
    mk_land_t = None
    mk_splash_hit_a = False
    for i in range(int(horizon * 60)):
        t = i / 60
        if assassin is None and mk.is_alive and t > 2.0 and mk.position.y <= TRIGGER_Y:
            a = battle_mod.Troop(bs.next_entity_id, Position(*assassin_pos), 0,
                                 'Assassin', bs)
            bs._spawn_entity(a)
            assassin = a
            a_deploy_t = t
        bs.step(1 / 60)
        if mk_jump_t is None and mkh._mk_jump is not None:
            mk_jump_t = t
            j = mkh._mk_jump
            mk_jump_tgt = (j['x1'], j['y1'])
        if mk_jump_t is not None and mk_land_t is None and mkh._mk_jump is None:
            mk_land_t = t
        if assassin is not None and a_dash_t is None and assassin._dash_active:
            a_dash_t = t
        if assassin is not None and a_dash_t is not None and assassin.hp < 907 and not mk_splash_hit_a:
            mk_splash_hit_a = True
    tower_loss = sum(init_hp[t.id] - (t.hp if t.is_alive else 0)
                     for t in towers0)
    mk_alive = mk.is_alive
    a_alive = assassin.is_alive
    jmp = f"起跳t={mk_jump_t:.1f} 落点={mk_jump_tgt} 落地t={mk_land_t:.1f}" if mk_jump_t else "无跳跃"
    dash = f"冲刺t={a_dash_t:.1f}" if a_dash_t else "未冲刺"
    print(f"{label}: 塔损={tower_loss:>5.0f} | "
          f"MK={'存活' if mk_alive else '被击杀'} | "
          f"刺客={'存活' if a_alive else '阵亡'} | 放刺t={a_deploy_t:.1f} | "
          f"{jmp} | {dash} | 落地溅射打到刺客={mk_splash_hit_a}")
    return {"tower_loss": tower_loss, "mk_alive": mk_alive, "a_alive": a_alive,
            "mk_jump_t": mk_jump_t, "a_dash_t": a_dash_t, "splash_hit_a": mk_splash_hit_a}


if __name__ == "__main__":
    print("—— 用户机制版：左路MK走2.5格 + 中轴偏左列刺客（lv11，45s 视界）——")
    print("（MK 起跳 3.5-5.0 格 / 滞空不无敌 / 刺客冲刺无敌规避落地溅射）")
    r5 = run_case("刺客第5格(8.5,10.5)", (8.5, 10.5))
    r6 = run_case("刺客第6格(8.5,9.5)", (8.5, 9.5))
    print()
    for lbl, r in (("第5格(8.5,10.5)", r5), ("第6格(8.5,9.5)", r6)):
        verdict = "完美解 ✓" if r["tower_loss"] <= 0 else "塔掉血 ✗"
        print(f"{lbl}: 塔损={r['tower_loss']:.0f} -> {verdict}")
