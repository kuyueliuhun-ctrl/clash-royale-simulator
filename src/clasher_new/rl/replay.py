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
LEAGUE_REPLAY_SCHEMA = 5
#: schema 5（2026-09-18，S2 第二轮）：实体条目**末尾再追加** `root_cast` / `is_product` / `share`，
#: 帧**追加** `v0` / `v1`（= 在线 `RLEnv._active_v` 真值）。
#: 动机（`docs/s2_instrument_2026-09-18.md` §9）：schema 4 有两处**退化** ——
#: ①没有 `is_product` ⇒ 规则①(交战对至少一侧非产物)**退化为恒真**；
#: ②残值 `V` 只能**离线重建** ⇒ 帧内延迟出兵可能污染份额。
#: 带上帧内真值后：`V` 零误差、可直接**对账重建法**、`cost(E)` 用在线 `share` 而不是卡表回退。
#: **向后兼容**：仍全部追加在**末尾**，按下标读 0..9 的消费者不受影响；读方以 `schema >= 5` 判定。
#:
#: schema 4（2026-09-18）：实体条目**末尾追加** `id` 与 `target_id` 两个字段。
#: 动机：离线做「局面圣水交换」需要 **实体身份** 与 **索敌关系**，而 schema ≤3 只有
#: `[name,x,y,hp,player,...]` ⇒ 实测 `league_88000.pkl` 有 **8584/9079 帧存在重复 (name,player)**
#: （两只 Minions 不可区分）⇒ 精确归因不可能。`battle_state.next_entity_id` 本就是**每局单调唯一**的，
#: 故无需新增 `uid`，只需把**已有的** `entity.id` 写进快照。
#: **向后兼容**：新字段追加在**末尾**，所有按下标读 0..9 的消费者（`dashboard.py` 前端、
#: `scripts/forensics_card_usage.py`、`rl/train_solo.py::behavioral_metrics`）**不受影响**；
#: 读方须以 `schema >= 4` 判定字段是否存在。
#:
#: **schema 5.x（2026-09-19）· 可选键 `meta["tower_geom"]`**：本局引擎的塔几何
#: `[[cx, cy, hw, hh, player], ...]`（来源 = `tower_geometry(battle)`）。**不升 schema**：
#: 它是**局级可选键**（与帧级可选键 `et`/`cards` 同级做法），老录像没有它，读方必须容忍
#: （dashboard 前端回落常量）。动机与体积实测见 `tower_geometry` 的 docstring。


def tower_geometry(battle):
    """引擎的**塔几何**（前端按真实足迹画塔用）：`[[cx, cy, hw, hh, player], ...]`。

    为什么要有这个（2026-09-19，用户：「我们引擎已经正常实现塔了，但 dashboard 回放的塔
    还是 1 格，把它恢复到正常尺寸」）：dashboard 的塔尺寸原先**写死在前端 JS 里**
    （`half = 0.5 格` ⇒ 公主塔画成 1 格见方），而引擎 2026-09-09 起塔是**矩形**
    （公主塔 3×3 / 国王塔 4×4，`arena.TileGrid.towers`）⇒ 两套几何，且前端那份
    **不会随引擎更新而更新**（这正是本次的根因）。
    现在由引擎侧把几何写进录像的 `meta["tower_geom"]`（**不是**逐帧：实测逐帧会让
    录像体积 +24%~28%，见 `docs/dashboard_tower_geom_2026-09-19.md` §2），前端优先用它；
    老录像（无该键）回落到与 `arena.TileGrid.towers` 同值的 JS 常量，由
    `scripts/check_dashboard_js.py` 的对账测试守着（改了引擎塔尺寸而不同步即 FAIL）。

    **只读**：只读 `battle.arena.towers`，不改任何状态。两种表形状都认：
      - **4 元组** `(pos, hw, hh, player)` —— 本仓（矩形口径，hw/hh 是半宽/半高）；
      - **3 元组** `(pos, r, player)` —— 上游（圆形口径，`hw = hh = r`）。实测上游
        王塔 `r=1.4` = `card_utils` 的 `collisionRadius` 1400/1000、公主塔 `1.0` =
        1000/1000 ⇒ 与本仓录进录像的实体 `radius` 同值，可交叉对账。

    形状不认识 / 拿不到 ⇒ 返回 `None`（前端回落常量，绝不抛）。
    """
    towers = getattr(getattr(battle, "arena", None), "towers", None)
    if not towers:
        return None
    out = []
    try:
        for t in towers:
            pos, pid, rest = t[0], t[-1], t[1:-1]
            if len(rest) == 2:
                hw, hh = float(rest[0]), float(rest[1])
            elif len(rest) == 1:
                hw = hh = float(rest[0])
            else:
                return None
            out.append([float(pos.x), float(pos.y), hw, hh, int(pid)])
    except Exception:
        return None
    return out or None


def battle_snapshot(battle, bundle, reward, info, v=None, shares=None):
    """把一个决策步压缩成轻量帧（不含 32×18 观测网格，体积可控）。

    每帧含：时间、动作 bundle、奖励、对手出牌、双方塔血/圣水/皇冠、存活实体列表。
    实体条目（schema 4）：
        [name, x, y, hp, player, kind, max_hp, shield, shield_max, radius, id, target_id]
    kind ∈ troop/building/projectile/effect（供前端按 pygame 风格渲染）；
    `id` = 实体在本局的唯一 id（= `BattleState.next_entity_id` 分配值）；
    `target_id` = 当前索敌目标的 id（无目标为 `None`）⇒ 离线可**直接重建索敌关系图**。

    schema 5 追加：`root_cast`（产出该实体的**出牌事件 id**；塔/出牌直接产物为 null）、
    `is_product`（是否由建筑/单位/法术**生成**）、`share`（该实体当前的**部署份额**，
    不在台账里的实体（塔/产物）记 **0.0**，与 `rl_reward_plan_v2:19` 同口径）；
    帧级 `v0`/`v1` = 在线 `RLEnv._active_v`。
    `v`/`shares` 缺省（如 `human_play`）⇒ 这些字段为 `None`（读方须容忍）。
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
        # schema 5.x（2026-09-18 第四轮，**可选键**）：在线局面交换的逐帧明细
        # `info["engagement_trade_detail"] = [phi_part, tau, score, n_windows]`。
        # 只有 measure-only（或开了该项）跑才有；缺省不写键 ⇒ 完全向后兼容。
        "et": (list(info["engagement_trade_detail"])
               if info.get("engagement_trade_detail") is not None else None),
        "et_src": ("online" if info.get("engagement_trade_detail") is not None else None),
        "crown0": int(p0.get_crown_count()),
        "crown1": int(p1.get_crown_count()),
        # schema 5：在线残值真值（`RLEnv._active_v`）；缺省 None ⇒ 读方回落到离线重建
        "v0": (float(v[0]) if v is not None else None),
        "v1": (float(v[1]) if v is not None else None),
        "entities": [
            [e.name, round(float(e.position.x), 1), round(float(e.position.y), 1),
             round(float(e.hp), 1), int(e.player),
             _kind(e),
             float(getattr(e.data, "hp", e.hp) or e.hp),
             float(getattr(e, "shield_health", 0.0) or 0.0),
             float(getattr(e.data, "shield_health", 0.0) or 0.0),
             float(getattr(e.data, "collision_radius", 0.5) or 0.5),
             int(e.id),                                    # schema 4：实体身份
             (int(e.target_id) if getattr(e, "target_id", None) is not None else None),
             # schema 5：出牌溯源 + 产物标记 + 部署份额（只读记录，不参与任何行为）
             (int(e.root_cast) if getattr(e, "root_cast", None) is not None else None),
             (bool(e.is_product) if getattr(e, "is_product", None) is not None else None),
             (float(shares.get(e.id, 0.0)) if shares is not None else None)]
            for e in battle.entities.values() if e.is_alive
        ],
    }


def save_league_replays(games, path):
    """保存联赛录像集合：games = [{meta, winner, frames}, ...]。

    可选局级键 `meta["tower_geom"]`（引擎塔几何）由**生产方**写入（`LeagueGameRecorder`
    首帧写一次、上游探针写一次），本函数**不改** `games` 一个字符，只负责落盘。
    """
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
