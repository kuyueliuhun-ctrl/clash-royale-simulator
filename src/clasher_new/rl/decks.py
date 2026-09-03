"""三分类卡组加载器（用户数据集 docs/leaderboard_decks_classified.json）。

数据集：200 副天梯卡组，按 archetype 分三类：
- 推进流（60 副）
- 防守反击流（120 副）
- 自闭流（20 副）

卡名映射：
- 数据集用 RoyaleAPI 风格小写短横线名（如 royal-giant-ev1 / barbarian-barrel-hero）；
- 先剥后缀（-ev1/-hero/-star*），再归一化（去分隔符、小写）与引擎 card_data 对表；
- 明确别名表覆盖火精灵/冰精灵/雪球/滚木等复数或异名；
- 仍对不上的卡槽用引擎可部署卡池随机补位（保持 8 卡完整）。
"""

import os
import sys
import json
import re
import random

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from card_utils import card_data
from rl.opponents import build_card_pool

ARCHETYPES = ["推进流", "防守反击流", "自闭流"]

#: 明确别名：数据集卡名 → 引擎卡名（复数/异名/官方名差异）
CARD_ALIASES = {
    "fire-spirit": "FireSpirits",
    "ice-spirit": "IceSpirits",
    "giant-snowball": "Snowball",
    "the-log": "Log",
}

_ENGINE_LOOKUP = None
_FILL_POOL = None


def _engine_lookup():
    global _ENGINE_LOOKUP
    if _ENGINE_LOOKUP is None:
        lookup = {}
        for e in card_data:
            n = re.sub(r"[^a-z0-9]", "", e.lower())
            lookup.setdefault(n, e)
        _ENGINE_LOOKUP = lookup
    return _ENGINE_LOOKUP


def normalize_card(name: str):
    """数据集卡名 → 引擎卡名；无法映射返回 None。"""
    n = re.sub(r"-(ev\d+|hero|star\d*|lvl\d+)$", "", name)
    n = re.sub(r"[^a-z0-9]", "", n.lower())
    hit = _engine_lookup().get(n)
    if hit:
        return hit
    return CARD_ALIASES.get(name)


def map_deck_cards(cards, seed=0):
    """映射一副 8 卡卡组；未命中的卡槽用引擎卡池补位（确定性）。"""
    global _FILL_POOL
    if _FILL_POOL is None:
        _FILL_POOL = build_card_pool()
    rng = random.Random(seed)
    out = []
    missing = 0
    for c in cards:
        hit = normalize_card(c)
        if hit:
            out.append(hit)
        else:
            out.append(rng.choice(_FILL_POOL))
            missing += 1
    return out, missing


def _default_paths():
    here = os.path.dirname(os.path.abspath(__file__))          # .../src/clasher_new/rl
    src = os.path.dirname(here)                                 # .../src/clasher_new
    repo = os.path.dirname(os.path.dirname(src))                # 仓库根
    return [
        os.path.join(repo, "docs", "leaderboard_decks_classified.json"),
        os.path.join(src, "..", "..", "docs", "leaderboard_decks_classified.json"),
        "docs/leaderboard_decks_classified.json",
        "leaderboard_decks_classified.json",
    ]


def load_classified_decks(path=None):
    """加载三分类卡组 → list[{"archetype", "cards"(8 张引擎卡), "missing"}]

    path 缺省时自动探测仓库 docs/leaderboard_decks_classified.json。
    """
    path = path or next((p for p in _default_paths() if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError("找不到 leaderboard_decks_classified.json，请用 --decks-path 指定")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for i, x in enumerate(data["decks"]):
        cards, missing = map_deck_cards(x["cards"], seed=i)
        out.append({
            "archetype": x.get("archetype", "未知"),
            "cards": cards,
            "missing": missing,
        })
    return out


def decks_by_archetype(decks):
    """按 archetype 分组：dict[str, list[deck]]。"""
    grouped = {a: [] for a in ARCHETYPES}
    for d in decks:
        grouped.setdefault(d["archetype"], []).append(d)
    return grouped


def classify_stats(decks):
    """统计各分类卡组数 / 平均补位卡数。"""
    from collections import Counter
    counts = Counter(d["archetype"] for d in decks)
    missing = {a: sum(d["missing"] for d in decks if d["archetype"] == a) / max(1, counts[a])
               for a in counts}
    return counts, missing
