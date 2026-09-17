"""对局 replay 记录（规划文档 8.1）：含特权隐藏状态标签，供信念模块监督训练。

修复（P1-21）：
- 统一容器格式，带 schema 版本号（save/load/end 同一结构，消除与 export_replay 的双格式）；
- record_hidden=True 但 hidden 缺失时显式 warn，不再静默丢数据；
- to_belief_dataset 返回 [(obs, opp_played, hidden), ...]。
"""

import os
import sys
import warnings
from dataclasses import dataclass, field

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

SCHEMA_VERSION = 2


@dataclass
class EpisodeReplay:
    record_hidden: bool = True
    steps: list = field(default_factory=list)
    schema: int = SCHEMA_VERSION
    _active: bool = False

    def start(self):
        self.steps = []
        self._active = True

    def record_step(self, obs, bundle, reward, info, hidden=None):
        hid = hidden if hidden is not None else info.get("hidden")
        if self.record_hidden and hid is None:
            warnings.warn("record_hidden=True 但本步无 hidden 标签，该步将丢失监督信号")
        step = {
            "obs": obs,
            "bundle": [(sa.kind, sa.slot, sa.x, sa.y) for sa in bundle.sub_actions],
            "reward": reward,
            "opp_played": info.get("opp_played"),
            "time": info.get("battle_time"),
        }
        if self.record_hidden:
            step["hidden"] = hid
        self.steps.append(step)

    def end(self) -> dict:
        self._active = False
        return {"schema": SCHEMA_VERSION, "steps": self.steps, "record_hidden": self.record_hidden}

    def save(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self.end(), f)

    @classmethod
    def load(cls, path) -> "EpisodeReplay":
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        # 兼容 v1 裸 {"steps": ...} 容器
        if isinstance(data, dict) and "steps" in data:
            ep = cls(record_hidden=data.get("record_hidden", True),
                     schema=data.get("schema", 1))
            ep.steps = data["steps"]
            return ep
        raise ValueError(f"无法识别的 replay 格式: {type(data)}")

    def to_belief_dataset(self):
        """把 replay 转成信念监督样本列表：[(obs, opp_played, hidden), ...]"""
        out = []
        for st in self.steps:
            hidden = st.get("hidden")
            if hidden is not None:
                out.append((st["obs"], st["opp_played"], hidden))
        return out


# ---------------------------------------------------------------------------
# 联赛录像（紧凑格式，供网页/回放观看，非信念训练数据）
# ---------------------------------------------------------------------------
LEAGUE_REPLAY_SCHEMA = 4
#: schema 4（2026-09-18）：实体条目**末尾追加** `id` 与 `target_id` 两个字段。
#: 动机：离线做「局面圣水交换」需要 **实体身份** 与 **索敌关系**，而 schema ≤3 只有
#: `[name,x,y,hp,player,...]` ⇒ 实测 `league_88000.pkl` 有 **8584/9079 帧存在重复 (name,player)**
#: （两只 Minions 不可区分）⇒ 精确归因不可能。`battle_state.next_entity_id` 本就是**每局单调唯一**的，
#: 故无需新增 `uid`，只需把**已有的** `entity.id` 写进快照。
#: **向后兼容**：新字段追加在**末尾**，所有按下标读 0..9 的消费者（`dashboard.py` 前端、
#: `scripts/forensics_card_usage.py`、`rl/train_solo.py::behavioral_metrics`）**不受影响**；
#: 读方须以 `schema >= 4` 判定字段是否存在。


def battle_snapshot(battle, bundle, reward, info):
    """把一个决策步压缩成轻量帧（不含 32×18 观测网格，体积可控）。

    每帧含：时间、动作 bundle、奖励、对手出牌、双方塔血/圣水/皇冠、存活实体列表。
    实体条目（schema 4）：
        [name, x, y, hp, player, kind, max_hp, shield, shield_max, radius, id, target_id]
    kind ∈ troop/building/projectile/effect（供前端按 pygame 风格渲染）；
    `id` = 实体在本局的唯一 id（= `BattleState.next_entity_id` 分配值）；
    `target_id` = 当前索敌目标的 id（无目标为 `None`）⇒ 离线可**直接重建索敌关系图**。
    """
    from battle import Building, Projectile, SpawnProjectile, AreaEffect, TimedExplosive

    def _kind(e):
        if isinstance(e, Building):
            return "building"
        if isinstance(e, (SpawnProjectile, AreaEffect, TimedExplosive)):
            return "effect"
        if isinstance(e, Projectile):
            return "projectile"
        return "troop"

    p0, p1 = battle.players
    return {
        "t": round(float(battle.time), 2),
        "bundle": [(sa.kind, sa.slot, sa.x, sa.y) for sa in bundle.sub_actions],
        "reward": float(reward),
        "opp_played": info.get("opp_played"),
        "towers0": [float(p0.king_tower_hp), float(p0.left_tower_hp), float(p0.right_tower_hp)],
        "towers1": [float(p1.king_tower_hp), float(p1.left_tower_hp), float(p1.right_tower_hp)],
        "elixir0": float(p0.elixir),
        "elixir1": float(p1.elixir),
        "crown0": int(p0.get_crown_count()),
        "crown1": int(p1.get_crown_count()),
        "entities": [
            [e.name, round(float(e.position.x), 1), round(float(e.position.y), 1),
             round(float(e.hp), 1), int(e.player),
             _kind(e),
             float(getattr(e.data, "hp", e.hp) or e.hp),
             float(getattr(e, "shield_health", 0.0) or 0.0),
             float(getattr(e.data, "shield_health", 0.0) or 0.0),
             float(getattr(e.data, "collision_radius", 0.5) or 0.5),
             int(e.id),                                    # schema 4：实体身份
             (int(e.target_id) if getattr(e, "target_id", None) is not None else None)]  # schema 4：索敌关系
            for e in battle.entities.values() if e.is_alive
        ],
    }


def save_league_replays(games, path):
    """保存联赛录像集合：games = [{meta, winner, frames}, ...]。"""
    import pickle
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"schema": LEAGUE_REPLAY_SCHEMA, "games": games}, f)


def load_league_replays(path):
    """读取联赛录像集合；兼容旧容器。"""
    import pickle
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "games" in data:
        return data["games"]
    return data
