"""推理时浅 MCTS（RL-MCTS v1）——方案见 docs/mcts_design.md。

定位：零训练风险的推理侧增强，直接回答「搜索能赚多少 Elo」。

- 节点 = (BattleState, to_act)：deepcopy 战场 + 待行动方；
- 转移 = 我方 bundle 部署 + 对手 bundle 部署（回调注入，None=对手不再部署）
  + battle.step × decision_frames（0.5s 决策帧，与 RLEnv 一致）；
- 叶估值 = 叶状态再确定性推演 leaf_horizon_s，按 economy 口径计塔血差 + 资源账
  （列式权重与训练奖励同源：normalize_tower_dmg + elixir_diff_weight 两段相位）；
- 候选动作经 validate_bundle 全量校验（空砸门/不裸下门/EV 闸门与提交路径同源），
  先验来自 FollowerPolicy 的 (slot, cell) logits（可选，缺省均匀）；
- 引擎确定性 + 对手确定性 → 树统计量可复用（标准 UCT）。

用法：
    mcts = RLMCTS(policy=main, opponent_fn=None)          # 或 opponent_fn=battle->bundle
    bundle, info = mcts.search(battle, player_id=0)
    # bundle 可直接交 RLEnv.step()；info 含 elapsed_s / n_sims / root_value

预算参考（18.9k battle-steps/s 实测）：默认 24 次模拟 ≈ 0.9s/决策帧。
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from core import Position
from card_utils import Card
from player import PlayerState
from rl.action_bundle import ActionBundle, SubAction, sub_position
from rl.action_mask import GRID_H, GRID_W, K_MAX, legal_cells, validate_bundle
from rl.env_wrapper import _phase_weights


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

@dataclass
class MCTSConfig:
    n_simulations: int = 24       # 每决策帧模拟次数
    max_depth: int = 3            # 决策帧深度（0.5s/帧 → 一次交锋 1.5s）
    leaf_horizon_s: float = 8.0   # 叶确定性推演秒数
    prior_top_k: int = 8          # 每帧策略先验保留的候选 bundle 数
    c_uct: float = 1.4            # UCT 探索系数
    decision_frames: int = 30     # 每决策帧 battle.step 次数（0.5s @ dt=1/60）
    dt: float = 1 / 60
    max_bundles_per_node: int = 64  # 候选 bundle 枚举上限（先验/top-k 截断前）
    reward: dict = field(default_factory=lambda: {
        "crown_weight": 8.0, "crown_lose_weight": 10.0,
        "tower_dmg_opp": 0.001, "tower_dmg_self": 0.0012,
        "tower_dmg_late": 0.002, "tower_dmg_self_late": 0.0022,
        "elixir_diff_weight": 0.5, "elixir_diff_late": 0.1,
    })


# ---------------------------------------------------------------------------
# 值函数（与训练奖励 economy 口径同构）
# ---------------------------------------------------------------------------

#: 每塔 lv11 满血锚（与 env_wrapper._per_tower_norm_dmg 同源：King 4824 / Princess 3052）
_TOWER_HP_ANCHORS = [4824.0, 3052.0, 3052.0]


def _tower_premium_loss(player: PlayerState) -> float:
    """逐塔累计塔损（差异化定价，存量视角）：Σ ∫₀^dmg mult(x) dx × (锚/max_i)。

    与 compute_reward 的 per-tower 归一化同源：低血塔单位血价值凹形溢价、
    王塔两公主塔存活×king_gate（≈0）。存量视角没有"伤害发生血线"历史，按
    "从满血累计掉 D 比例"积分：mult(残血比例 x)=1+k(1−x)² 的解析积分 =
    D + k(D − D² + D³/3)——**把塔打爆（D=1）的累计价值 ≥ 停在残血**，
    与"斩杀最值钱"一致（否则已破塔 mult(0)=1 会被低估，test_mcts_basic 回归抓出）。
    返回正 = 我方累计塔损（node_value 里 me 侧取负、op 侧取正）。
    """
    hps = [player.king_tower_hp, player.left_tower_hp, player.right_tower_hp]
    princesses_alive = int(hps[1] > 0.0) + int(hps[2] > 0.0)
    from rl.env_wrapper import DEFAULT_TOWER_PREMIUM_K, DEFAULT_KING_GATE
    k = DEFAULT_TOWER_PREMIUM_K
    total = 0.0
    for i in range(3):
        m_i = _TOWER_HP_ANCHORS[i]
        dmg_i = m_i - hps[i]
        if dmg_i <= 0.0:
            continue
        D = dmg_i / m_i   # 累计损比例 0..1
        # 王塔贬值：两公主塔存活时全损价值 ×king_gate（不参与积分缩放，直接乘总账）
        gate = 1.0
        if i == 0 and princesses_alive >= 2:
            gate = DEFAULT_KING_GATE
        # 积分 ∫₀^D (1 + k(1−x)²) dx = D + k(D − D² + D³/3)
        premium_loss = D + k * (D - D * D + D * D * D / 3.0)
        total += dmg_i * premium_loss * gate
    return total


def node_value(battle, player_id: int, cfg: MCTSConfig) -> float:
    """从 player_id 视角评估当前战场（塔血差 + 资源账，双倍期权重切换）。

    值域约定：越大对我方越好（敌方塔掉血为正、我方塔掉血为负）。
    叶推演/树回传统一用这个函数，保证量纲一致。

    塔血存量差相对开局锚（≈满血）计：v = 我方累计损；敌损为正贡献。
    权重与训练奖励同源（tower_dmg_opp/self 两段相位，self 侧 1.2× 不对称），
    塔损先按**每塔**归一化到各自 lv11 锚再乘权重（×1000 抬回数值尺度，仅影响 UCT 分母尺度）；
    塔血差异化定价（tower_value_mult）：低血塔单位血价值凹形溢价、王塔两公主塔存活≈0，
    与训练奖励 compute_reward 的 per-tower 分支同源。
    皇冠差直接按 crown_weight 计（破塔 = 大额里程碑，与训练奖励一致）。
    资源账：手牌圣水差 × edw（叶推演双方不再部署，Φ 的部署份额项为 0）。
    """
    rw = cfg.reward
    tw_opp, tw_self, edw = _phase_weights(rw, battle.time)
    me, op = battle.players[player_id], battle.players[1 - player_id]
    v_me = _tower_premium_loss(me)   # 我方累计塔损（正=损）
    v_op = _tower_premium_loss(op)   # 敌方累计塔损
    # 权重直接用训练奖励原值（0.001/塔血 vs 0.5/费——量纲自洽：前段 1费=500塔血），
    # 不得额外缩放（曾用 ×1000 抬尺度导致塔伤/费差量纲失配，空砸被误判正 EV）。
    val = tw_opp * v_op - tw_self * v_me
    # get_crown_count() 语义 = 自己【丢掉的】塔数（见 player.get_crown_count），
    # 对手丢塔多 = 我方皇冠领先 → 取 op - me。
    val += rw["crown_weight"] * (op.get_crown_count() - me.get_crown_count())
    val += edw * (me.elixir - op.elixir)
    return val


def leaf_value(battle, player_id: int, cfg: MCTSConfig) -> float:
    """叶估值：确定性推演 leaf_horizon_s 后计 node_value（双方不再部署）。"""
    sim = copy.deepcopy(battle)
    steps = int(cfg.leaf_horizon_s / cfg.dt)
    for _ in range(steps):
        if sim.game_over:
            break
        sim.step(cfg.dt)
    return node_value(sim, player_id, cfg)


# ---------------------------------------------------------------------------
# 候选动作枚举（掩码同源）
# ---------------------------------------------------------------------------

def enumerate_bundles(battle, player_id: int, cfg: MCTSConfig,
                      priors: Optional[dict] = None) -> list:
    """枚举决策帧的可执行单卡 bundle（逐槽位×合法格），validate_bundle 全量校验。

    priors: {"slot": np.ndarray(K_MAX+2), "cell": np.ndarray(GRID_H*GRID_W)}
            可选策略 logits，用于候选排序（top-k 截断）；None 时按槽位/格心序。
    返回 [(bundle, slot_idx, cell_flat), ...]（不含空 bundle——"等待"由 STOP 动作表达）。
    """
    p = battle.players[player_id]
    slots = validate_slots(p)
    if not slots.any():
        return []
    slot_order = np.argsort(-priors["slot"][:K_MAX]) if priors is not None else np.arange(K_MAX)
    candidates = []
    # 先按槽位均匀收集（每槽 max_per_slot 个），避免槽 0 的格子淹没其他槽；
    # 有先验时最后统一按先验截断到 prior_top_k。
    max_per_slot = cfg.max_bundles_per_node
    for si in slot_order:
        si = int(si)
        if not slots[si]:
            continue
        card = p.cycle[si]
        cells = legal_cells(battle, player_id, card)
        if not cells.any():
            continue
        if priors is not None:
            flat_order = np.argsort(-priors["cell"])
        elif _is_spell_card(card):
            # 法术：按「范围内敌方实体 HP 总量」降序（罩得多排前）——
            # 距离排序会把满血塔/残血塔同等对待，法术关心的是伤害覆盖量
            flat_order = _cells_by_spell_value(battle, player_id, card)
        else:
            # 部队：距最近敌方部队近的格在前（防守相关格优先），
            # 低血量敌塔加权（斩杀线附近的格序压倒性优先）。
            flat_order = _cells_by_threat(battle, player_id)
        n = 0
        for ci in flat_order:
            ci = int(ci)
            y, x = divmod(ci, GRID_W)
            if not cells[y, x]:
                continue
            b = ActionBundle(sub_actions=[SubAction(kind="deploy", slot=si + 1, x=x, y=y)])
            ok, _reason, _res = validate_bundle(battle, player_id, b)
            if ok:
                candidates.append((b, si, ci))
                n += 1
            if n >= max_per_slot:
                break
    if priors is not None and len(candidates) > cfg.prior_top_k:
        # 按先验分排序截断
        def score(item):
            _b, si, ci = item
            return float(priors["slot"][si]) + float(priors["cell"][ci]) * 1e-3
        candidates.sort(key=score, reverse=True)
        candidates = candidates[:cfg.prior_top_k]
    return candidates


def validate_slots(p: PlayerState) -> np.ndarray:
    """手牌槽圣水掩码（简化版 slot_mask，避免额外依赖）。"""
    from rl.action_mask import slot_mask
    return slot_mask(p)


def _cells_by_threat(battle, player_id: int) -> np.ndarray:
    """本地网格 flat index 的缺省排序：距最近敌方部队近的格在前。

    P1 视角同样按本地格算（sub_position 换算世界坐标后比距离）。
    确定性（距离和格 index 双键排序）。

    塔的权重：距低血量敌塔更近的格排更前（斩杀/施压优先）——
    以 (dist - hp_bonus) 排序，hp_bonus = 满血比例越低的塔权重越大。"""
    foes = [(e.position.x, e.position.y, _foe_priority(e))
            for e in battle.entities.values()
            if e.player != player_id and getattr(e, "is_alive", False)
            and e.position is not None]
    ys, xs = np.mgrid[0:GRID_H, 0:GRID_W]
    xf = xs.reshape(-1).astype(float) + 0.5
    yf = ys.reshape(-1).astype(float) + 0.5
    if player_id == 1:
        xf, yf = 17.5 - xf, 31.5 - yf
    if foes:
        fx = np.array([f[0] for f in foes])
        fy = np.array([f[1] for f in foes])
        fp = np.array([f[2] for f in foes])
        d = np.sqrt((xf[:, None] - fx[None, :]) ** 2 + (yf[:, None] - fy[None, :]) ** 2)
        # 加权距离：优先级越高的敌军（低血塔），等效距离越短
        d = (d / np.maximum(fp[None, :], 1e-6)).min(axis=1)
    else:
        # 无敌军：靠近敌方中线（本地 y 朝对方半场）优先，P0 中线 y≈16，P1 本地 y≈15
        d = np.abs(yf - (16.0 if player_id == 0 else 15.0))
    # 双键：距离升序，其次格 index 升序（确定性）
    return np.lexsort((np.arange(GRID_H * GRID_W), d))


def _foe_priority(e) -> float:
    """敌军目标优先级（≥1）：低血量塔权重高（残血塔=斩杀目标）。"""
    max_hp = getattr(e.data, "hp", None)
    if max_hp and max_hp > 0:
        frac = float(e.hp) / float(max_hp)
        # 满血塔 frac=1 → 1.0；残血 70/3052 → ≈45（等效距离 ÷45，格序压倒性优先）
        return 1.0 + (1.0 - min(1.0, frac)) * 50.0
    return 1.0


def _is_spell_card(card_name: str) -> bool:
    from card_utils import Card
    return Card(card_name).type == "spell"


def _cells_by_spell_value(battle, player_id: int, card_name: str) -> np.ndarray:
    """法术格序：范围内敌方实体「价值」降序（罩得多/罩到可击杀目标排前）。

    目标价值 = min(hp, 预估法术伤)；可击杀（hp ≤ 法术伤）再乘击杀加成
    （塔被击杀 ≈ 皇冠，价值远超剩余血量）。距离排序会把满血塔/残血塔
    同等对待，法术真正关心的是"这次施放能兑换多少价值"。
    radius 取卡牌溅射半径（spell 口径），无半径法术退化为 threat 序。
    确定性（价值降序 + 格 index 升序）。"""
    from rl.action_mask import _spell_radius_m
    from spell_module import get_spell_profile
    profile = get_spell_profile(card_name, Card.default_level)
    dmg = float(profile.get("troop_damage") or 0.0)
    radius = _spell_radius_m(card_name)
    foes = [(e.position.x, e.position.y, _spell_target_value(e, dmg))
            for e in battle.entities.values()
            if e.player != player_id and getattr(e, "is_alive", False)
            and e.position is not None]
    ys, xs = np.mgrid[0:GRID_H, 0:GRID_W]
    xf = xs.reshape(-1).astype(float) + 0.5
    yf = ys.reshape(-1).astype(float) + 0.5
    if player_id == 1:
        xf, yf = 17.5 - xf, 31.5 - yf
    if not foes or radius <= 0.0:
        return _cells_by_threat(battle, player_id)
    fx = np.array([f[0] for f in foes])
    fy = np.array([f[1] for f in foes])
    fv = np.array([f[2] for f in foes])
    d = np.sqrt((xf[:, None] - fx[None, :]) ** 2 + (yf[:, None] - fy[None, :]) ** 2)
    cover = ((d <= radius) * fv[None, :]).sum(axis=1)   # 每格罩到的敌方总价值
    # 主键覆盖量降序，次键格 index 升序 → lexsort 主键在最后
    return np.lexsort((np.arange(GRID_H * GRID_W), -cover))


def _spell_target_value(e, spell_damage: float) -> float:
    """法术视角的目标价值：min(hp, 法术伤)；可击杀目标乘击杀加成。

    塔：可击杀（斩杀）价值 = 满血量级 ×3（≈皇冠+终局意义）；
    部队：可击杀价值 = 其圣水费的量级（这里用 HP 比例近似，够排序用）。"""
    hp = float(e.hp)
    if spell_damage > 0 and hp <= spell_damage:
        # 可击杀：塔 → 高价值（皇冠）；部队 → 中价值
        is_tower = getattr(e.data, "hp", None) and e.hp > 0 and "Tower" in type(e).__name__
        if "Tower" in type(e).__name__ or "tower" in getattr(e, "name", "").lower():
            return hp + 9000.0
        return hp + 1500.0
    return hp


# ---------------------------------------------------------------------------
# 树节点
# ---------------------------------------------------------------------------

class _Node:
    __slots__ = ("battle", "to_act", "parent", "action", "children", "untried",
                 "n_visits", "sum_value")

    def __init__(self, battle, to_act, parent=None, action=None):
        self.battle = battle
        self.to_act = to_act          # 本节点行动方（轮到我方/对手交替）
        self.parent = parent
        self.action = action          # 到达本节点的 (bundle, applier) 供调试
        self.children = []
        self.untried = None           # 惰性枚举
        self.n_visits = 0
        self.sum_value = 0.0

    def q(self) -> float:
        return self.sum_value / self.n_visits if self.n_visits else 0.0


# ---------------------------------------------------------------------------
# 主搜索器
# ---------------------------------------------------------------------------

OpponentFn = Callable[[object, int], Optional[ActionBundle]]


class RLMCTS:
    """推理时浅 MCTS。search() 一次决策帧调用一次。"""

    def __init__(self, policy=None, opponent_fn: Optional[OpponentFn] = None,
                 cfg: Optional[MCTSConfig] = None):
        self.policy = policy          # FollowerPolicy（可选，先验来源）
        self.opponent_fn = opponent_fn  # battle -> ActionBundle | None
        self.cfg = cfg or MCTSConfig()

    # ---- 公开入口 ----

    def search(self, battle, player_id: int, obs=None) -> tuple:
        """返回 (ActionBundle, info)。空 bundle = 本帧等待（合法且可能最优）。

        原 battle 不被修改（全部在 deepcopy 上推演）。
        """
        t0 = time.monotonic()
        cfg = self.cfg
        root = _Node(copy.deepcopy(battle), player_id)
        root.untried = self._expand_actions(root)
        # 空动作（等待）总在候选里
        for _ in range(cfg.n_simulations):
            node = root
            # 1) 选择：沿树走 UCT，直到有未扩展动作或叶
            depth = 0
            while node.untried is not None and not node.untried and node.children and depth < cfg.max_depth:
                node = self._uct_select(node)
                depth += 1
            # 2) 扩展
            if node.untried is None:
                node.untried = self._expand_actions(node)
            if node.untried:
                action = node.untried.pop(0)
                child = self._apply(node, action)
                node.children.append(child)
                node = child
            # 3) 推演估值（叶确定性推演）
            value = leaf_value(node.battle, player_id, cfg)
            # 4) 回传：v1 决策帧语义 = 我方出手 + 对手同帧响应 → 树中所有节点
            #    都是我方决策（child.to_act 恒为 root 视角），值全从根视角评估，
            #    不做符号翻转（sign 翻转仅适用于真正的对手交替节点树）。
            cur = node
            while cur is not None:
                cur.n_visits += 1
                cur.sum_value += value
                cur = cur.parent
        # 选根最优子节点（访问数优先，平手取均值）
        if not root.children:
            return ActionBundle(), {"n_sims": cfg.n_simulations,
                                    "elapsed_s": time.monotonic() - t0,
                                    "root_value": 0.0, "wait": True}
        best = max(root.children, key=lambda c: (c.n_visits, c.q()))
        info = {"n_sims": cfg.n_simulations,
                "elapsed_s": time.monotonic() - t0,
                "root_value": best.q(),
                "visits": [c.n_visits for c in root.children],
                "wait": best.action is None or len(best.action[0].sub_actions) == 0}
        return (best.action[0] if best.action else ActionBundle()), info

    # ---- 内部 ----

    def _expand_actions(self, node: _Node) -> list:
        """枚举本节点行动方的候选：(等待) + 单卡 bundle 列表。"""
        acts = [None]  # None = 等待（不部署，纯推进）
        priors = self._priors(node) if self.policy is not None else None
        if node.to_act == self._root_player():
            bundles = enumerate_bundles(node.battle, node.to_act, self.cfg, priors)
        else:
            # 对手节点：回调给一个 bundle（或不部署）
            ob = self._opponent_bundle(node.battle, node.to_act)
            bundles = [(ob, -1, -1)] if ob is not None and len(ob.sub_actions) else []
            if not bundles:
                acts = [None]  # 对手也不动
                return acts
            acts = [bundles[0]]
            return acts
        return acts + [(b, si, ci) for b, si, ci in bundles]

    def _root_player(self) -> int:
        return getattr(self, "_player_id", 0)

    def _priors(self, node: _Node) -> Optional[dict]:
        """从策略网络提取 (slot, cell) logits 先验。策略不可用时返回 None。"""
        try:
            from rl.observation import observe
            from rl.belief import BeliefInference  # noqa: F401 占位，先验不含信念
            obs = observe(node.battle, node.to_act)
            import torch
            with torch.no_grad():
                enc = self.policy._encode(obs, None, None)
                slot_logits = self.policy.slot_head(enc)
                cell_logits = self.policy.cell_head(enc).view(-1)
            return {"slot": slot_logits.cpu().numpy(),
                    "cell": cell_logits.cpu().numpy()}
        except Exception:
            return None

    def _opponent_bundle(self, battle, opp_id: int) -> Optional[ActionBundle]:
        if self.opponent_fn is None:
            return None
        try:
            return self.opponent_fn(battle, opp_id)
        except Exception:
            return None

    def _apply(self, node: _Node, action) -> _Node:
        """把动作应用到节点战场的副本，返回子节点。action=None = 等待。"""
        sim = copy.deepcopy(node.battle)
        to_act = 1 - node.to_act if action is not None and action[1] == -1 else node.to_act
        # 轮转：本节点先行动作，然后推演一帧，下一节点行动方切换
        if action is not None:
            bundle = action[0]
            ok, _r, resolved = validate_bundle(sim, node.to_act, bundle)
            if ok:
                for card, sa in resolved:
                    if card == "__ability__":
                        sim.use_ability(node.to_act)
                    else:
                        sim.deploy_card(node.to_act, card, sa.to_position(node.to_act))
        # 对手响应（我方节点才需要；对手节点已在上层注入）
        if node.to_act == self._root_player():
            ob = self._opponent_bundle(sim, 1 - node.to_act)
            if ob is not None and len(ob.sub_actions):
                ok, _r, resolved = validate_bundle(sim, 1 - node.to_act, ob)
                if ok:
                    for card, sa in resolved:
                        if card == "__ability__":
                            sim.use_ability(1 - node.to_act)
                        else:
                            sim.deploy_card(1 - node.to_act, card,
                                            sa.to_position(1 - node.to_act))
        # 推进一决策帧
        for _ in range(self.cfg.decision_frames):
            if sim.game_over:
                break
            sim.step(self.cfg.dt)
        child = _Node(sim, 1 - node.to_act if False else node.to_act, parent=node, action=action)
        # v1 简化：决策帧语义 = 我方出手 + 对手同帧响应 → 下一节点仍轮到我方决策
        child.to_act = self._root_player()
        return child

    def _uct_select(self, node: _Node) -> _Node:
        log_n = np.log(max(1, node.n_visits))
        best, best_v = None, -1e18
        for c in node.children:
            exploit = c.q()
            explore = self.cfg.c_uct * np.sqrt(log_n / max(1, c.n_visits))
            v = exploit + explore
            if v > best_v:
                best, best_v = c, v
        return best
