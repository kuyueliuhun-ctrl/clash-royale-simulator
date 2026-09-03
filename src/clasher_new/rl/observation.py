"""玩家视角观测构建 + 特权隐藏状态标签（信念模块监督用）。

观测约定与现有 CREnv 保持一致：grid (32,18,15)、hand (5,)、elixir (1,)。
额外提供 next_card / time，供 follower 与信念模块使用。
"""

import os
import sys

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import numpy as np

# 与 environment.py 保持一致
ENTITY_NAMES = [
    "None", "Knight", "MiniPekka", "Arrows", "Minions", "Archer",
    "Musketeer", "Fireball", "Giant", "King_PrincessTowers",
    "KingTower", "ArrowsSpell", "FireballSpell",
]
CARD_TYPES = ["troop", "character", "spell", "building"]

GRID_H, GRID_W = 32, 18
GRID_C = 15


def observe(battle, player_id: int = 0) -> dict:
    """返回玩家 player_id 的可见观测字典。"""
    obs = np.zeros((GRID_H, GRID_W, GRID_C), dtype=np.float32)
    for each in battle.entities.values():
        if not each.is_alive:
            continue
        if each.name not in ENTITY_NAMES:
            continue
        entity_id = ENTITY_NAMES.index(each.name)
        card_type = CARD_TYPES.index(each.data.type)
        is_opponent = each.player != player_id  # 己方单位统一标为 0
        elixir = each.data.elixir
        is_air = int(each.data.is_air_unit)
        attacks_ground, attacks_air = int(each.data.attack_ground), int(each.data.attack_air)
        speed = each.data.speed
        hp_left = np.log(each.hp) / 10 if each.hp != 0 else 0
        hp_percentage = each.hp / each.data.hp if each.data.hp != 0 else 0
        hit_speed = each.data.hit_speed
        attack_range = each.data.range / 3
        sight_range = each.data.sight_range / 3
        damage = each.data.damage / 200
        projectile_damage = each.data.projectile_data.damage / 200

        x, y = int(each.position.x), int(each.position.y)
        if player_id == 1:
            x = 17 - x
            y = 31 - y
        if 0 <= x < GRID_W and 0 <= y < GRID_H:
            obs_arr = np.array([
                entity_id, is_opponent, elixir, card_type, speed, is_air,
                attacks_ground, attacks_air, hp_left, hp_percentage, hit_speed,
                attack_range, sight_range, damage, projectile_damage,
            ], dtype=np.float32)
            obs[y][x] = obs_arr

    p = battle.players[player_id]
    hand = np.array(
        [ENTITY_NAMES.index(each) if each in ENTITY_NAMES else 0 for each in p.cycle[:5]],
        dtype=np.int32,
    )
    next_card = ENTITY_NAMES.index(p.cycle[4]) if p.cycle[4] in ENTITY_NAMES else 0
    return {
        "grid": obs,
        "hand": hand,
        "elixir": np.array([p.elixir], dtype=np.float32),
        "next_card": np.array([next_card], dtype=np.int32),
        "time": np.array([battle.time], dtype=np.float32),
    }


def hidden_labels(battle, player_id: int = 0) -> dict:
    """特权隐藏状态标签（只允许训练期使用，绝不进跟随者观测）。

    用于信念模块监督：对手真实手牌 / 牌序 / 圣水 / 意图 / 风格。
    """
    opp_id = 1 - player_id
    me = battle.players[player_id]
    opp = battle.players[opp_id]

    def _ids(cards):
        return np.array([ENTITY_NAMES.index(c) if c in ENTITY_NAMES else 0 for c in cards], dtype=np.int32)

    # 对手当前手牌 = 循环前 4 张（可出牌）
    opp_hand = opp.cycle[:4]
    return {
        "opp_hand": _ids(opp_hand),            # (4,)
        "opp_next": _ids([opp.cycle[4]])[0],   # 下一张牌
        "opp_cycle": _ids(opp.cycle),          # 完整循环 (8,)
        "opp_elixir": np.array([opp.elixir], dtype=np.float32),
        "opp_crown": np.array([opp.get_crown_count()], dtype=np.int32),
        "opp_towers": np.array([opp.king_tower_hp, opp.left_tower_hp, opp.right_tower_hp],
                               dtype=np.float32),
        "my_elixir": np.array([me.elixir], dtype=np.float32),
        "my_hand": _ids(me.cycle[:4]),
        "time": np.array([battle.time], dtype=np.float32),
    }
