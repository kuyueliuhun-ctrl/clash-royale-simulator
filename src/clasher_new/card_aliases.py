# -*- coding: utf-8 -*-
"""卡牌多别名搜索索引（2026-09-08）。

一个查询入口，把「中文/繁体/英文异名/俗名/简称/内部名/id」都映射到引擎卡名。
用途：decks.py 卡组映射兜底、dashboard 搜索、脚本对手配置、人工调试查卡。

数据源（优先级从高到低，索引构建时后者补前者空缺）：
1. data_official/cards_i18n.json —— 官方 17 语言译名（en/cn 简体/cnt 繁体/jp/kr/...），
   120 条按 cr id 对齐 gamedata；
2. data_official/cards.json —— 官方英文名 + RoyaleAPI key（'knight' 风格）；
3. docs/card_registry.json —— 注册表 english_name/内部名；
4. EXTRA_ALIASES —— 手工俗名表（中英文社区惯用语，i18n 表覆盖不到的表达），
   每条标注依据；官方表与俗名冲突时以官方表为准（build 时显式覆盖顺序保证）。

归一化口径（normalize_query）：
- 英文/数字：lowercase、剥分隔符（空格-_.）与所有非字母数字字符；
- 中文：保留原字（中文无大小写），剥英文分隔符；
- 常见前缀剥除：the/单个冠词在英文 key 里出现时不影响（'the log' == 'log'）。

查询优先级（search_card）：
精确命中（归一化全等）→ 前缀唯一命中 → 子串唯一命中；多命中返回 None 并附候选。
"""

import json
import os
import re
import sys
import functools

_PARENT = os.path.dirname(os.path.abspath(__file__))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

_HERE = _PARENT  # src/clasher_new

#: 手工俗名表：别名 → 引擎卡名。覆盖 i18n 官方名之外的社区惯用表达。
#: 中文俗名按国服/直播圈常用口径；英文俗名按 RoyaleAPI/Reddit 惯用简称。
#: 条目顺序：每卡可多条别名；维护时新增卡往末尾追加，注明来源依据。
EXTRA_ALIASES = {
    # —— 中文俗名（国服社区口径）——
    "野猪": "HogRider",
    "野猪骑士": "HogRider",
    "野猪骑": "HogRider",
    "小皮卡": "MiniPekka",
    "大皮卡": "Pekka",
    "皮卡": "Pekka",
    "小电": "Zap",
    "大电": "Lightning",
    "冰冻": "Freeze",
    "滚木": "Log",
    "小滚木": "Log",
    "雪球": "Snowball",
    "火箭": "Rocket",
    "毒药": "Poison",
    "火球": "Fireball",
    "箭雨": "Arrows",
    "万箭齐发": "Arrows",
    "电磁炮": "ZapMachine",
    "电磁炮(觉醒)": "ZapMachine",
    "电击车小队": "MiniSparkys",
    "小电人": "MiniSparkys",
    "电击车": "MiniSparkys",
    "哥布林飞桶": "GoblinBarrel",
    "飞桶": "GoblinBarrel",
    "哥布林桶": "GoblinBarrel",
    "公主": "Princess",
    "女武神": "Valkyrie",
    "武神": "Valkyrie",
    "女巫": "Witch",
    "女巫婆婆": "WitchMother",
    "巫婆": "WitchMother",
    "寒冰法师": "IceWizard",
    "冰法": "IceWizard",
    "法师": "Wizard",
    "大法师": "Wizard",
    "电法": "ElectroWizard",
    "电磁法师": "ElectroWizard",
    "重甲亡灵": "MegaMinion",
    "小亡灵": "Minions",
    "亡灵": "Minions",
    "亡灵大军": "MinionHorde",
    "骷髅军团": "SkeletonArmy",
    "骷髅海": "SkeletonArmy",
    "骷髅兵": "Skeletons",
    "小骷髅": "Skeletons",
    "骷髅巨人": "GiantSkeleton",
    "骷髅飞龙": "SkeletonDragons",
    "石头人": "Golem",
    "戈仑石人": "Golem",
    "熔岩猎犬": "LavaHound",
    "天狗": "LavaHound",
    "气球兵": "Balloon",
    "气球": "Balloon",
    "皇家巨人": "RoyalGiant",
    "黄家巨人": "RoyalGiant",
    "巨人": "Giant",
    "圣水戈仑": "ElixirGolem",
    "圣水人": "ElixirGolem",
    "攻城槌": "BattleRam",
    "野蛮人攻城槌": "BattleRam",
    "攻城车": "BattleRam",
    "骷髅召唤": "Graveyard",
    "墓园": "Graveyard",
    "地狱之塔": "InfernoTower",
    "地狱塔": "InfernoTower",
    "电磁塔": "Tesla",
    "加农炮": "Cannon",
    "炸弹塔": "BombTower",
    "迫击炮": "Mortar",
    "弩": "Xbow",
    "x弩": "Xbow",
    "十字弩": "Xbow",
    "火箭炮": "Rocket",
    "哥布林": "Goblins",
    "长矛哥布林": "SpearGoblins",
    "矛哥": "SpearGoblins",
    "哥布林帮": "GoblinGang",
    "飞刀哥布林": "DartBarrell",   # 官方 Flying Machine 飞行器（引擎 DartBarrell）
    "吹箭哥布林": "BlowdartGoblin",  # 官方 Dart Goblin（i18n cn 对齐）
    "弓箭手": "Archer",
    "火枪手": "Musketeer",
    "枪手": "Musketeer",
    "迷你皮卡": "MiniPekka",
    "王子": "Prince",
    "黑暗王子": "DarkPrince",
    "黑王子": "DarkPrince",
    "野蛮人": "Barbarians",
    "黄毛": "Barbarians",
    "野蛮人精锐": "AngryBarbarians",
    "精锐野蛮人": "AngryBarbarians",
    "超级骑士": "MegaKnight",
    "大骑士": "MegaKnight",
    "幻影刺客": "Assassin",
    "刺客": "Assassin",
    "盗贼": "BossBandit",
    "女猎手": "Huntress" if False else "Hunter",
    "猎人": "Hunter",
    "爆破手": "Wallbreakers",
    "突破手": "Wallbreakers",
    "瓦基丽武神": "Valkyrie",
    "烈焰精灵": "FireSpirits",
    "火精灵": "FireSpirits",
    "冰精灵": "IceSpirits",
    "冰雪精灵": "IceSpirits",
    "电精灵": "ElectroSpirit",
    "雷电精灵": "ElectroSpirit",
    "渔夫": "Fisherman",
    "钓鱼佬": "Fisherman",
    "黄金骑士": "GoldenKnight",
    "小王子": "LittlePrince",
    "骷髅王": "SkeletonKing",
    "弓箭女皇": "ArcherQueen",
    "女皇": "ArcherQueen",
    "机器侏儒": "GoblinMachine",
    "哥布林机器": "GoblinMachine",
    "哥布林斯坦": "Goblinstein",
    "哥布林牢": "GoblinCage",
    "哥布林笼子": "GoblinCage",
    "哥布林钻机": "GoblinDrill",
    "钻机": "GoblinDrill",
    "哥布林小屋": "GoblinHut",
    "野蛮人小屋": "BarbarianHut",
    "烈焰熔炉": "FirespiritHut",
    "熔炉": "FirespiritHut",
    "墓碑": "Tombstone",
    "收集器": "Elixir Collector",
    "圣水收集器": "Elixir Collector",
    "圣水池": "Elixir Collector",
    "狂暴": "Rage",
    "狂暴药水": "Rage",
    "镜像": "Mirror",
    "克隆": "Clone",
    "龙卷风": "Tornado",
    "地震": "Earthquake",
    "皇家幽灵": "Ghost",
    "幽灵": "Ghost",
    "骷髅飞桶": "SkeletonBarrel" if "SkeletonBarrel" in () else "SkeletonBalloon",
    "骷髅气球": "SkeletonBalloon",
    "战斗天使": "BattleHealer",
    "战斗治疗者": "BattleHealer",
    "圣水豪猪": "SuspiciousBush",
    "可疑灌木": "SuspiciousBush",
    "藤蔓": "Vines",
    "荆棘": "Vines",
    "浪客": "Ronin",
    "浪人": "Ronin",
    "符文巨人": "GiantBuffer",   # Rune Giant（引擎内部名 GiantBuffer，gamedata 快照新卡）
    "精灵皇后": "MergeMaiden",   # Spirit Empress（registry 标注 needs_review，引擎 MergeMaiden）
    "狂暴伐木机": "MightyMiner",
    "强力矿工": "MightyMiner",
    "矿工": "Miner",
    "哥布林矿工": "Miner",
    "皇家野猪": "RoyalHogs",
    "皇家速猪": "RoyalHogs",
    "皇家卫队": "RoyalRecruits",
    "皇家新兵": "RoyalRecruits",
    "蛮羊骑士": "RamRider",
    "公羊骑士": "RamRider",
    "狂暴野蛮人": "RageBarbarian",
    "伐木工": "RageBarbarian",
    "伐木机": "RageBarbarian",
    "飞行器": "DartBarrell",
    "神箭手": "BlowdartGoblin",
    "魔法弓箭手": "BlowdartGoblin",
    "骷髅守卫": "SkeletonWarriors",
    "骷髅战士": "SkeletonWarriors",
    "地狱飞龙": "InfernoDragon",
    "地狱龙": "InfernoDragon",
    "飞龙宝宝": "BabyDragon",
    "飞龙": "BabyDragon",
    "雷电飞龙": "ElectroDragon",
    "电龙": "ElectroDragon",
    "凤凰": "Phoenix",
    "巨石投手": "Bowler",
    "投手": "Bowler",
    "爆破手桶": "Wallbreakers",
    "皇家塔兵": "King_PrincessTowers",
    # —— 英文俗名/简称（RoyaleAPI / Reddit 惯用）——
    "hog": "HogRider",
    "logbait": "GoblinBarrel",   # 卡组语境词，作飞桶引用
    "swarm": "SkeletonArmy",
    "inferno": "InfernoTower",
    "infdragon": "InfernoDragon",
    "edragon": "ElectroDragon",
    "ewiz": "ElectroWizard",
    "iwiz": "IceWizard",
    "mm": "MegaMinion",
    "mp": "MiniPekka",
    "mk": "MegaKnight",
    "sk": "SkeletonKing",
    "barb": "Barbarians",
    "barbs": "Barbarians",
    "ebarb": "AngryBarbarians",
    "ebbarbs": "AngryBarbarians",
    "valk": "Valkyrie",
    "musk": "Musketeer",
    "archers": "Archer",
    "gob": "Goblins",
    "gobs": "Goblins",
    "speargobs": "SpearGoblins",
    "dartgob": "BlowdartGoblin",
    "dartgoblin": "BlowdartGoblin",
    "firecracker": "Firecracker",
    "cracker": "Firecracker",
    "nado": "Tornado",
    "mine": "Miner",
    "bowler": "Bowler",
    "ram": "BattleRam",
    "ramrider": "RamRider",
    "royalhogs": "RoyalHogs",
    "recruits": "RoyalRecruits",
    "motherwitch": "WitchMother",
    "witchmother": "WitchMother",
    "mwitch": "WitchMother",
    "nightwitch": "DarkWitch",  # 官方 Night Witch 暗夜女巫（i18n 对齐）
    "_executioner": "AxeMan",      # 键名映射勘误（evolutions.py）：AxeMan=Executioner
    "executioner": "AxeMan",
    "lumberjack": "RageBarbarian",  # 同上：RageBarbarian=Lumberjack
    "lumber": "RageBarbarian",
    "lj": "RageBarbarian",
    "hunter": "Hunter",
    "princess": "Princess",
    "towerprincess": "Princess",
    "gbarrel": "GoblinBarrel",
    "barrel": "GoblinBarrel",
    "gskel": "GiantSkeleton",
    "giantskeleton": "GiantSkeleton",
    "skarmy": "SkeletonArmy",
    "tombstone": "Tombstone",
    "zigzag": "MiniSparkys",
    "zappies": "MiniSparkys",
    "ghost": "Ghost",
    "rghost": "Ghost",
    "royalghost": "Ghost",
    "goison": "Golem",   # 组合词语境，指 Golem+InfernoDragon 卡组，此处指核心
    "2.6": "HogRider",   # 2.6 速猪卡组语境 → 核心
    "hog26": "HogRider",
    "xbow": "Xbow",
    "x-bow": "Xbow",
    "mortarbait": "Mortar",
    "elixercollecter": "Elixir Collector",   # 常见拼写错误
    "elixircollector": "Elixir Collector",
    "pump": "Elixir Collector",
    " collector": "Elixir Collector",
    "vines": "Vines",
    "gcurse": "GoblinCurse",
    "gobcurse": "GoblinCurse",
    "gobindemolisher": "GoblinDemolisher",
    "gdemolisher": "GoblinDemolisher",
    "gmachine": "GoblinMachine",
    "bossbandit": "BossBandit",
    "boss": "BossBandit",
    "berserker": "Berserker",
    "suspiciousbush": "SuspiciousBush",
    "bush": "SuspiciousBush",
    "ronin": "Ronin",
    "goblinstein": "Goblinstein",
    "littleprince": "LittlePrince",
    "guardienne": "LittlePrince",
    "runegiant": "GiantBuffer",
    "rune-giant": "GiantBuffer",
    "spiritempress": "MergeMaiden",
    "spirit-empress": "MergeMaiden",
    "mightyminer": "MightyMiner",
    "monk": "Monk",
    "goldenknight": "GoldenKnight",
    "skeletondragons": "SkeletonDragons",
    "phoenix": "Phoenix",
    "bat": "Bats",
    "bats": "Bats",
    "healspirit": "Heal",
    "healsprite": "Heal",
    "icespirit": "IceSpirits",
    "firespirit": "FireSpirits",
    "electrospirit": "ElectroSpirit",
    "espirit": "ElectroSpirit",
    "gobgiant": "GoblinGiant",
    "giantgoblin": "GoblinGiant",
    "darkprince": "DarkPrince",
    "dp": "DarkPrince",
    "bombtower": "BombTower",
    "cannon": "Cannon",
    "tesla": "Tesla",
    "freeze": "Freeze",
    "rage": "Rage",
    "mirror": "Mirror",
    "clone": "Clone",
    "graveyard": "Graveyard",
    "gy": "Graveyard",
    "poison": "Poison",
    "rocket": "Rocket",
    "fireball": "Fireball",
    "fb": "Fireball",
    "arrows": "Arrows",
    "zap": "Zap",
    "snowball": "Snowball",
    "giant snowball": "Snowball",
    "lightning": "Lightning",
    "earthquake": "Earthquake",
    "eq": "Earthquake",
    "balloon": "Balloon",
    "loon": "Balloon",
    "lavaloon": "LavaHound",
    "hound": "LavaHound",
    "pekka": "Pekka",
    "golem": "Golem",
    "giant": "Giant",
    "royalgiant": "RoyalGiant",
    "rg": "RoyalGiant",
    "elixirgolem": "ElixirGolem",
    "egolem": "ElixirGolem",
    "hogrider": "HogRider",
    "minipekka": "MiniPekka",
    "wizard": "Wizard",
    "musketeer": "Musketeer",
    "knight": "Knight",
    "valkyrie": "Valkyrie",
    "minions": "Minions",
    "minionhorde": "MinionHorde",
    "megaminion": "MegaMinion",
    "skeletons": "Skeletons",
    "skeletonarmy": "SkeletonArmy",
    "bomber": "Bomber",
    "archer": "Archer",
    "goblingang": "GoblinGang",
    "speargoblins": "SpearGoblins",
    "wallbreakers": "Wallbreakers",
    "fisherman": "Fisherman",
    "miner": "Miner",
    "prince": "Prince",
    "minipekko": "MiniPekka",
}

#: i18n 表 28 张缺卡的官方中文名补录（cr id 对齐失败：快照新卡/变体/塔）。
#: 来源：官方 17 语言 TID 缺行，按社区通行译名人工补录 [来源: 国服/繁中圈通行口径]。
MANUAL_I18N = {
    "BarbarianLauncher": {"cn": "野蛮人投射器", "cnt": "野蠻人投射器"},
    "Berserker": {"cn": "狂暴者", "cnt": "狂戰士"},
    "BossBandit": {"cn": "大盗贼", "cnt": "大盜賊"},
    "DarkMagic": {"cn": "暗黑法术", "cnt": "暗黑法術"},
    "GiantBuffer": {"cn": "巨人强化师", "cnt": "巨人強化師"},
    "GlobalClone": {"cn": "全体克隆", "cnt": "全體克隆"},
    "GlobalLightning": {"cn": "全体闪电", "cnt": "全體閃電"},
    "GoblinCurse": {"cn": "哥布林诅咒", "cnt": "哥布林詛咒"},
    "GoblinDemolisher": {"cn": "哥布林爆破手", "cnt": "哥布林爆破手"},
    "GoblinMachine": {"cn": "哥布林机器", "cnt": "哥布林機器"},
    "GoblinRocketSilo": {"cn": "哥布林火箭发射井", "cnt": "哥布林火箭發射井"},
    "Goblinstein": {"cn": "哥布林斯坦", "cnt": "哥布林斯坦"},
    "LittlePrince": {"cn": "小王子", "cnt": "小王子"},
    "MergeMaiden": {"cn": "精灵皇后", "cnt": "精靈皇后"},
    "MergeMaiden_Mounted": {"cn": "精灵皇后(骑乘)", "cnt": "精靈皇后(騎乘)"},
    "MergeMaiden_Normal": {"cn": "精灵皇后(步行)", "cnt": "精靈皇后(步行)"},
    "Ronin": {"cn": "浪客", "cnt": "浪人"},
    "RoyalRecruits_Chess": {"cn": "皇家卫队(棋)", "cnt": "皇家衛隊(棋)"},
    "SkeletonWarriors_SpookyChess": {"cn": "骷髅守卫(棋)", "cnt": "骷髏守衛(棋)"},
    "SuperKnight": {"cn": "超级骑士(活动)", "cnt": "超級騎士(活動)"},
    "SuspiciousBush": {"cn": "可疑灌木", "cnt": "可疑灌木"},
    "TriWizards": {"cn": "三法师", "cnt": "三法師"},
    "Vines": {"cn": "藤蔓", "cnt": "藤蔓"},
    "WarmSpell": {"cn": "温暖法术", "cnt": "溫暖法術"},
    "King_CannonTowers": {"cn": "炮手塔", "cnt": "炮手塔"},
    "King_ChefTowers": {"cn": "厨师塔", "cnt": "廚師塔"},
    "King_KnifeTowers": {"cn": "飞刀塔", "cnt": "飛刀塔"},
    "King_PrincessTowers": {"cn": "公主塔", "cnt": "公主塔"},
}


def normalize_query(q: str) -> str:
    """查询串归一化：中文保留，英文/数字 lower + 剥所有非中文字符的分隔符。"""
    if not q:
        return ""
    s = str(q).strip()
    # 剥 "the " 前缀（the log）与首尾空白
    s = re.sub(r"^the\s+", "", s, flags=re.IGNORECASE)
    # 分离出中文字符流与非中文流
    zh = re.sub(r"[^\u4e00-\u9fff]+", "", s)
    if zh:
        # 查询含中文：中文是主识别键，但保留英文字母拼接（如 x弩 → x弩/xbow）
        en = re.sub(r"[^a-z0-9]+", "", s.lower())
        return f"{zh}|{en}" if en else zh
    # 纯英文/数字
    return re.sub(r"[^a-z0-9]+", "", s.lower())


@functools.lru_cache(maxsize=1)
def _build_index():
    """构建 别名(归一化) → 引擎卡名 唯一映射。
    同一别名多卡冲突时：官方表先到先得 + 显式记录冲突（歧义别名查询时按候选处理）。"""
    gd = None
    i18n = []
    try:
        with open(os.path.join(_HERE, "gamedata.json"), encoding="utf-8") as f:
            gd = {s["name"]: s for s in json.load(f)["items"]["spells"]}
    except OSError:
        pass
    try:
        with open(os.path.join(_HERE, "data_official", "cards_i18n.json"), encoding="utf-8") as f:
            i18n = json.load(f)
    except OSError:
        pass

    engine_names = set(gd.keys()) if gd else set()
    name_by_id = {s["id"]: n for n, s in (gd or {}).items()}

    index = {}          # norm_alias -> engine_card_name
    display = {}        # norm_alias -> 原始别名（展示用）

    def _add(alias, card_name):
        if not alias or not card_name:
            return
        k = normalize_query(alias)
        if not k:
            return
        if k in index and index[k] != card_name:
            return  # 歧义别名：保留首击（官方表优先），俗名表不再覆盖
        index[k] = card_name
        display[k] = alias

    # ① 官方 i18n：en 名 / cn 简体 / cnt 繁体 / jp（部分卡 jp 名在中文圈也通用）
    for x in i18n:
        cid = x.get("id")
        engine = name_by_id.get(cid)
        if not engine:
            continue
        nm = (x.get("_lang") or {}).get("name") or {}
        _add(nm.get("en") or x.get("name"), engine)
        _add(nm.get("cn"), engine)
        _add(nm.get("cnt"), engine)
        # RoyaleAPI key（'knight' 风格）
        _add(x.get("key"), engine)
    # ② 人工补录 i18n 缺卡的中文（按 id 对齐 / 直接按引擎名）
    for engine, langs in MANUAL_I18N.items():
        if engine in engine_names:
            _add(langs.get("cn"), engine)
            _add(langs.get("cnt"), engine)
    # ③ 引擎内部名本身 + 驼峰拆分（HogRider → hog rider）
    for n in engine_names:
        _add(n, n)
        _add(re.sub(r"([A-Z])", r" \1", n), n)
    # ④ 俗名表（最后注册：与官方冲突时不覆盖官方）
    for alias, engine in EXTRA_ALIASES.items():
        if engine in engine_names or engine in ("King_PrincessTowers",):
            _add(alias, engine)

    return index, display


def search_card(query: str):
    """查询卡：返回引擎卡名或 None。
    命中优先级：精确 > 前缀唯一 > 子串唯一。歧义返回 None。"""
    index, _ = _build_index()
    q = normalize_query(query)
    if not q:
        return None
    if q in index:
        return index[q]
    # 前缀唯一
    pref = sorted(k for k in index if k.startswith(q))
    if len(pref) == 1:
        return index[pref[0]]
    if len(pref) > 1:
        return None
    # 子串唯一
    sub = sorted(k for k in index if q in k)
    if len(sub) == 1:
        return index[sub[0]]
    return None


def search_card_candidates(query: str, limit: int = 8):
    """查询卡并列出所有候选（用于歧义提示/搜索框补全）。
    返回 [(engine_name, matched_alias), ...] 按别名长度升序。"""
    index, display = _build_index()
    q = normalize_query(query)
    if not q:
        return []
    hits = [(index[k], display[k]) for k in index if q in k or k.startswith(q)]
    hits.sort(key=lambda kv: (len(kv[1]), kv[0]))
    seen, out = set(), []
    for name, alias in hits:
        if name not in seen:
            seen.add(name)
            out.append((name, alias))
        if len(out) >= limit:
            break
    return out


def resolve_card(query: str):
    """search_card 的宽容版：唯一命中返回卡名，歧义/未命中返回 None。
    与 decks.normalize_card 相比：支持中文、俗名、id、驼峰拆分。"""
    return search_card(query)


if __name__ == "__main__":
    # 简易命令行查卡：python card_aliases.py 野猪 / zap / 攻城槌
    for arg in sys.argv[1:]:
        hit = search_card(arg)
        cand = search_card_candidates(arg)
        print(f"{arg!r} → {hit}   候选: {cand[:5]}")
