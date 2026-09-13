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

#: G'-fix：塔血标量的归一化兜底锚（lv11）。正常情况下用 `battle.tower_max_hp`
#: （reset 时由 `rl/env_wrapper` 按本局等级写入），缺失时才退回这里。
_KING_HP_LV11 = 4824.0
_PRINCESS_HP_LV11 = 3052.0


def _clip01(v):
    return float(min(1.0, max(0.0, float(v))))

# 词表 v2（2026-09-09，连弩镜像卡组切换）：原 13 名只覆盖原版 8 卡的实体/包装器，
# 换镜像卡组（Xbow/Tesla/Log）或四卡组对手（HogRider/Golem...）时全部观测隐形
# （grid 里直接丢弃、手牌编码 0）。扩容为全卡池实体名 dump（部署+死亡刷出+法术
# 包装器+弹射物，见 scripts/_vocab_dump 记录）∪ 全卡池卡名（手牌/hand 编码用卡名，
# 与实体名是两套：手牌 "Snowball" ↔ 场上 "SnowballSpell"）。
# 【旧位序冻结】前 13 位与 v1 完全一致——旧 checkpoint 的 entity_emb 前 13 行
# 语义保留；行数变化由 follower.load_checkpoint 的 entity_emb 行兼容分支处理
# （旧行拷贝、新行从零学）。 belief token 事件 one-hot 用 len(ENTITY_NAMES)，
# 71→大维度走 belief_mlp.0.weight 尾零兼容（前 71 列语义不变）。
ENTITY_NAMES = [
    # —— v1 原 13 位（位序冻结，勿动）——
    "None", "Knight", "MiniPekka", "Arrows", "Minions", "Archer",
    "Musketeer", "Fireball", "Giant", "King_PrincessTowers",
    "KingTower", "ArrowsSpell", "FireballSpell",
    # —— v2 尾部追加：全卡池实体名（部署/死亡刷出/法术包装器/弹射物）——
    "AngryBarbarians", "ArcherQueen", "Assassin", "AxeMan", "BabyDragon",
    "Balloon", "BalloonBomb", "BarbLogProjectile", "BarbLogProjectileRolling",
    "Barbarian", "BarbarianHut", "BarbarianLauncher", "Barbarians", "Bats",
    "BattleHealer", "BattleRam", "Berserker", "BlowdartGoblin", "BombTower",
    "BombTowerBomb", "Bomber", "BossBandit", "Bowler", "Cannon", "DarkMagic",
    "DarkPrince", "DarkWitch", "DartBarrell", "Earthquake", "ElectroDragon",
    "ElectroGiant", "ElectroSpirit", "ElectroWizard", "EliteArcher",
    "Elixir Collector", "ElixirGolem", "ElixirGolem2", "FireSpirits",
    "Firecracker", "FirespiritHut", "Fisherman", "Freeze", "Ghost",
    "GiantBuffer", "GiantSkeleton", "GiantSkeletonBomb", "GlobalLightning",
    "GoblinBarrelSpell", "GoblinBrawler", "GoblinCage", "GoblinCurse",
    "GoblinDemolisher", "GoblinDrill", "GoblinGang", "GoblinGiant",
    "GoblinHut", "GoblinMachine", "GoblinMorphProjectile", "GoblinPartyHut",
    "GoblinRocketSilo", "Goblins", "Goblinstein", "Goblinstein_doctor",
    "GoldenKnight", "Golem", "Golemite", "Heal", "HealAura", "HogRider",
    "Hunter", "IceGolemite", "IceSpirits", "IceWizard", "InfernoDragon",
    "InfernoTower", "LavaHound", "LavaPups", "LittlePrince",
    "LogProjectile", "LogProjectileRolling", "MegaKnight", "MegaMinion",
    "MergeMaiden_Mounted", "MergeMaiden_Normal", "MightyMiner", "Miner",
    "MiniSparkys", "MinionHorde", "Monk", "Mortar", "MovingCannon", "Pekka",
    "Phoenix", "PhoenixEgg", "PhoenixNoRespawn", "Poison", "Prince",
    "PrinceBuff", "Princess", "Rage", "RageBarbarian", "RamRider",
    "RascalGirl", "Rascals", "RocketSpell", "Ronin", "RoyalDelivery",
    "RoyalGiant", "RoyalHogs", "RoyalRecruits", "RoyalRecruits_Chess",
    "SkeletonArmy", "SkeletonBalloon", "SkeletonDragons", "SkeletonKing",
    "SkeletonWarriors", "SkeletonWarriors_SpookyChess", "Skeletons",
    "SnowballSpell", "SpearGoblin", "SpearGoblinParty",
    "SpearGoblinProjectile", "SpearGoblins", "SuperArcher",
    "SuperEliteArcher", "SuperHogRider", "SuperHogRiderTerry",
    "SuperIceGolemite", "SuperKnight", "SuperLavaHound", "SuperLavaHound2",
    "SuperMiniPekka", "SuperWitch", "SuspiciousBush", "Tesla",
    "ThreeMusketeers", "Tombstone", "Tornado", "TowerPrincessProjectile",
    "TriWizards", "Valkyrie", "VinesProjectile", "Vines_AeO",
    "Wallbreakers", "WarmSpell", "Witch", "WitchMother", "Wizard", "Xbow",
    "Zap", "ZapMachine",
    # —— v2 尾部追加：卡名（手牌/hand 编码用；不以实体名出现的法术/特殊卡）——
    # 手牌编码 p.cycle[:5] 是卡名，与场上实体名两套（"Snowball" ↔ "SnowballSpell"）。
    # 集合 = build_card_pool() 全卡池 ∪ FOUR_DECK_SET ∪ DEFAULT_SOLO_DECK 与实体名的
    # 实测差集（Clone/Graveyard/Lightning 部署瞬发无包装器实体；Mirror 依赖 last_card）。
    "BarbLog", "Clone", "GlobalClone", "GoblinBarrel", "GoblinPartyRocket",
    "Graveyard", "Lightning", "Log", "MergeMaiden", "Mirror", "Rocket",
    "Snowball", "Vines",
]
ENTITY_SET = frozenset(ENTITY_NAMES)
CARD_TYPES = ["troop", "character", "spell", "building"]   # 位序冻结（grid 通道3 + one_hot num_classes=4）
# 引擎还有两个附加 data.type（词表 v2 后新实体才可见，全卡池实测枚举）：
# area_effect（Earthquake/Poison/HealAura 等区域效果）、projectile（Log/BarbLog
# 滚动弹射物）。两者都是法术衍生的瞬时效果体、身份已由 entity_id 通道区分，
# 折叠进 "spell" 类——不扩 one_hot 类别数（4→6 会改 CNN 输入通道 26→28，
# 破坏全部旧卷积权重）。
_TYPE_ALIAS = {"area_effect": "spell", "projectile": "spell", "bomb": "spell"}

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
        _t = _TYPE_ALIAS.get(each.data.type, each.data.type)
        # 兜底：数据行缺 tidType 时 type 为空串（新合成卡漏 tidType 不再崩观测），
        # 折叠进 character（与"未知类型当普通部队"同语义）。
        card_type = CARD_TYPES.index(_t) if _t in CARD_TYPES else CARD_TYPES.index("character")
        is_opponent = each.player != player_id  # 己方单位统一标为 0
        d = each.data
        # 垫片实体（AreaEffect/SpawnProjectile/GenericBomb 的 data 垫片）只提供
        # "下游最小属性集"，没有攻击/移动物理字段 → 统一 getattr 补零读取，
        # 而不是要求每个垫片补齐 13 个字段（新垫片漏补即崩，真实对局路径）。
        elixir = getattr(d, "elixir", 0) or 0
        is_air = int(bool(getattr(d, "is_air_unit", False)))
        attacks_ground = int(bool(getattr(d, "attack_ground", False)))
        attacks_air = int(bool(getattr(d, "attack_air", False)))
        speed = getattr(d, "speed", 0) or 0
        hp_left = np.log(each.hp) / 10 if each.hp != 0 else 0
        max_hp = getattr(d, "hp", 0) or 0
        hp_percentage = each.hp / max_hp if max_hp != 0 else 0
        hit_speed = getattr(d, "hit_speed", 0) or 0
        attack_range = (getattr(d, "range", 0) or 0) / 3
        sight_range = (getattr(d, "sight_range", 0) or 0) / 3
        damage = (getattr(d, "damage", 0) or 0) / 200
        pd_ = getattr(d, "projectile_data", None)
        projectile_damage = ((getattr(pd_, "damage", 0) or 0) if pd_ is not None else 0) / 200

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
    # —— G'-fix（2026-09-12）：双方塔血/皇冠/圣水差 做成**显式标量** ——
    # 背景（G′ 三层对照，docs/rl_training_fix_plan_v3.md §3.13）：塔血此前只以
    # grid 每格通道（log hp / hp%）存在，经 CNN→grid_ln→enc_fc→relu→enc_ln→GRU
    # 之后**线性不可解码**（`enc → 塔血差` 按局分组 R² = −0.03，而同一 enc →
    # time = +0.9999），且所有策略头（slot_head/cell_head）只吃 h
    # ⇒ 决策通路没有可靠的塔血输入。归一化：塔血用**本局满血**
    # （`battle.tower_max_hp`，reset 时由 env_wrapper 写入；缺失退回 lv11 锚），
    # 皇冠 /3，圣水差 /10 —— 全部落在 [0,1] 附近。
    _other = battle.players[1 - player_id]
    _tmax = getattr(battle, "tower_max_hp", None)
    _mine_max = _tmax.get(player_id) if isinstance(_tmax, dict) else None
    _opp_max = _tmax.get(1 - player_id) if isinstance(_tmax, dict) else None
    if not _mine_max:
        _mine_max = [_KING_HP_LV11, _PRINCESS_HP_LV11, _PRINCESS_HP_LV11]
    if not _opp_max:
        _opp_max = [_KING_HP_LV11, _PRINCESS_HP_LV11, _PRINCESS_HP_LV11]
    _mine = (p.king_tower_hp, p.left_tower_hp, p.right_tower_hp)
    _theirs = (_other.king_tower_hp, _other.left_tower_hp, _other.right_tower_hp)
    tower_state = np.array([
        _clip01(_mine[0] / max(1e-6, float(_mine_max[0]))),
        _clip01(_mine[1] / max(1e-6, float(_mine_max[1]))),
        _clip01(_mine[2] / max(1e-6, float(_mine_max[2]))),
        _clip01(_theirs[0] / max(1e-6, float(_opp_max[0]))),
        _clip01(_theirs[1] / max(1e-6, float(_opp_max[1]))),
        _clip01(_theirs[2] / max(1e-6, float(_opp_max[2]))),
        float(p.get_crown_count()) / 3.0,
        float(_other.get_crown_count()) / 3.0,
        (float(p.elixir) - float(_other.elixir)) / 10.0,
    ], dtype=np.float32)
    return {
        "grid": obs,
        "hand": hand,
        "elixir": np.array([p.elixir], dtype=np.float32),
        "next_card": np.array([next_card], dtype=np.int32),
        "time": np.array([battle.time], dtype=np.float32),
        #: G'-fix：9 维塔血/皇冠/圣水差标量（**tail 追加**，旧 ckpt 列序不变）
        "tower_state": tower_state,
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
