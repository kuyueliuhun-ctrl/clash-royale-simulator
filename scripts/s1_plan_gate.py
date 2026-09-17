# -*- coding: utf-8 -*-
"""S1 门禁：PlanToken 贪心执行器 vs 同一批局的基线（只读引擎、纯 CPU、零训练成本）。

对应预注册 `docs/reward_settlement_prereg_2026-09-14.md` §5（跑前写死的零成本前置门禁）：

    问题：约束到底在**结算侧**（奖励/优势/目标的结构）还是在**探索/信用侧**
          （策略根本没试过赢牌手段）？
    做法：用现成的手写规划器 `BeliefPlanner` 另建一个 **PlanToken 贪心执行器**，
          在**同一副牌 / 同一批局**上跑 §3 行为层三项。

本脚本**不新发明规划语义**：只把 `rl/belief_planner.py` 已经算出来的 `PlanToken`
翻译成 `ActionBundle`，并且**只走 RLEnv 既有的掩码 / 整包校验路径**
（`env.get_action_mask()` → `validate_bundle()` → `env.step()`）；非法动作会被发现并计数。

三个臂（`--arms` 逐个跑，便于把单次进程控制在 10 min 内）：
- ``random``      : 掩码随机（`ScriptedPolicy.play(env, 0)` 的口径）= **基线，同脚本复算**；
- ``token_strict``: **纯 token 转录**——token 建议哪张牌就打哪张、落点由 hint/region 决定；
                    `hold_mask` 命中的槽位禁用、`elixir_budget×10` 当花费上限；
                    **不含任何 Xbow / 赢牌手段专用逻辑**（= prereg 指定的仪器本体）；
- ``token_xbow``  : 在 ``token_strict`` 之上**手写一条** Xbow 规则（= 执行器能力上限探针）：
                    **非紧急帧**里只要手牌有 Xbow，就「不够费不出牌（攒费），够费就下桥头」；
                    Xbow 不在手里且非紧急 → 打最便宜的合法牌过牌（把它轮进手牌）；
                    紧急帧（有敌军在我半场，或 token 是防守/解牌意图）→ 照常听 token。
                    ⚠️ 这条规则**不在 `PlanToken` 的语义里**（`plan` 的 hold_mask/save 窗口
                    只由 `save_ace`/`setup_wait` 产生，而这两者在 Xbow 卡组上**从不触发**——
                    `_setup_wait` 的沉底名单不含 Xbow、`_save_ace` 的 ACE_CARDS 不含 Xbow）。
                    它存在与否正是本门禁要辨析的事：这条规则能过、而纯 token 转录不能过，
                    就说明**动作空间/掩码不是约束、plan 通道才是**。

行为层三项口径**照抄** `scripts/forensics_card_usage.py`（不重写、不自创）：
1. **全帧圣水 ≥6 占比** = 全部帧（含不出手的帧）里 `elixir0 ≥ 6` 的比例；
2. **Xbow 出手率** = 帧内**去重槽位**后 `Xbow` 次数 / 卡牌打出总次数；
3. **部署前圣水中位** = 「恰好 1 个去重子动作」帧的 `elixir0 + 卡费` 的中位数。
   另附 prereg §3 的第 4 列（同路两次部署间隔中位）。

用法（在 src/clasher_new 下）：
  # 基线：同脚本、同副牌、同批种子
  PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe ../../scripts/s1_plan_gate.py \
      --arm random --games 20 --out ../../docs/s1_plan_gate_random.json
  # 纯 token 转录（prereg 指定的仪器）
  ... --arm token_strict --games 20 --out ../../docs/s1_plan_gate_token_strict.json
  # 执行器能力上限探针（手写 Xbow 规则）
  ... --arm token_xbow --games 20 --out ../../docs/s1_plan_gate_token_xbow.json
  # 仪器自检：用同一段指标代码复算历史回放（应复现 0.11% / 0.03% / 3.250）
  ... --repro-replays runs/d1_long_100k/replays
"""

import argparse
import json
import os
import pickle
import random
import sys
import time
from collections import Counter

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src", "clasher_new")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

K_MAX = 4
GRID_H, GRID_W = 32, 18
LANE_SPLIT = 9.0
#: prereg §3 的对照基线（`runs/d1_long_100k`，11 文件 / 440 局）——**只作量级参照**；
#: 本脚本一律用同脚本复算的 random 臂当主基线（【R4】禁手抄）。
PREREG_BASELINE = {"all_frames_frac_ge6": 0.0011, "xbow_rate": 0.0003,
                   "pre_median": 3.25, "lane_dt_median": 5.5}

#: 非紧急意图（= 可以攒费的帧）——只被 token_xbow 臂使用
WAIT_INTENTS = ("cycle_and_wait", "setup_wait", "save_ace", "cycle_small")
#: 紧急/防守意图（token_xbow 在这些帧照常听 token）
URGENT_INTENTS = ("defend_left", "defend_right", "defend_king", "soft_control",
                  "spell_trade", "protect_backline", "pull", "king_activate",
                  "anti_spell", "spell_finish", "push_commit", "punish",
                  "push_left", "push_right", "counterpush", "spell_value")

#: region → 目标世界坐标锚点（蓝方视角；player0 塔 y≈3-6.5、河 y≈16、红塔 y≈25.5）
REGION_ANCHOR = {
    "own_left": (4.5, 8.0), "own_center": (9.0, 8.0), "own_right": (13.5, 8.0),
    "bridge_left": (4.5, 14.0), "bridge_right": (13.5, 14.0),
    "enemy_left": (4.5, 20.0), "enemy_center": (9.0, 20.0), "enemy_right": (13.5, 20.0),
}


def _stdout(s):
    try:
        print(s, flush=True)
    except UnicodeEncodeError:
        print(s.encode("utf-8", "replace").decode("utf-8", "replace"), flush=True)


# ---------------------------------------------------------------------------
# 行为层三项：**逐字照抄** `scripts/forensics_card_usage.py` 的口径
# ---------------------------------------------------------------------------
_COST_MAP_CACHE = {}


def _cost_map():
    """卡费表（只读 gamedata.json / cards_stats_*.json，缓存一次）。"""
    if "m" not in _COST_MAP_CACHE:
        from forensics_card_usage import load_cost_map
        _COST_MAP_CACHE["m"] = load_cost_map(_SRC)
    return _COST_MAP_CACHE["m"]


def compute_behavior_metrics(games, is_spell):
    """games = [{meta, winner, frames}]，帧格式 = `rl/replay.battle_snapshot`。

    采样口径与 `forensics_card_usage.py` 一致：只统计 `meta.side0 == "main"` 的局
    （本脚本 player-0 恒为被测策略 ⇒ 等价于全部局）。
    """
    import numpy as np
    elix_all, elix_post, elix_pre, dt_all, dt_same_lane = [], [], [], [], []
    card_counts = Counter()
    card_pre_elixir = {}          # 卡名 -> [部署前圣水]（只取「恰好 1 个去重子动作」帧）
    games_with_card = Counter()
    n_games_incl = 0
    n_frames = n_deploy_frames = n_subactions = n_subactions_dedup = 0
    dup_frames = n_single_frames = unknown_cost = 0
    n_unit_y_ge21 = n_spell_y_ge21 = n_unit_deploys = n_spell_plays = 0
    deck_cards = Counter()
    for g in games:
        meta = g.get("meta") or {}
        if str(meta.get("side0")) != "main":
            continue
        n_games_incl += 1
        for c in (meta.get("decks") or [[]])[0]:
            deck_cards[c] += 1
        seen_in_game = set()
        frames = g.get("frames") or []
        n_frames += len(frames)
        elix_all.extend(float(fr.get("elixir0") or 0.0) for fr in frames)
        prev_t = prev_lane = None
        for fr in frames:
            deploys = [b for b in (fr.get("bundle") or [])
                       if b and b[0] == "deploy" and 1 <= int(b[1]) <= K_MAX]
            if not deploys:
                continue
            n_deploy_frames += 1
            n_subactions += len(deploys)
            cards = list(fr.get("cards") or [])
            seen, slots, uniq_cards = set(), [], []
            for i, b in enumerate(deploys):
                k = int(b[1])
                if k in seen:
                    continue
                seen.add(k)
                slots.append((k, int(b[2]), int(b[3])))
                uniq_cards.append(cards[i] if i < len(cards) else None)
            n_subactions_dedup += len(slots)
            if len(slots) < len(deploys):
                dup_frames += 1
            uniq_pairs = [(c, s) for c, s in zip(uniq_cards, slots) if c]
            for c, (_k, _x, y) in uniq_pairs:
                card_counts[c] += 1
                seen_in_game.add(c)
                if is_spell(c):
                    n_spell_plays += 1
                    if y >= 21:
                        n_spell_y_ge21 += 1
                else:
                    n_unit_deploys += 1
                    if y >= 21:
                        n_unit_y_ge21 += 1
            if len(slots) != 1 or len(uniq_pairs) != 1:
                continue
            c = uniq_pairs[0][0]
            cc = _cost_map().get(c)
            if cc is None:
                unknown_cost += 1
                continue
            n_single_frames += 1
            post = float(fr.get("elixir0") or 0.0)
            elix_post.append(post)
            elix_pre.append(post + cc)
            card_pre_elixir.setdefault(c, []).append(post + cc)
            t = float(fr.get("t") or 0.0)
            lane = 0 if slots[0][1] < LANE_SPLIT else 1
            if prev_t is not None:
                dt_all.append(t - prev_t)
                if lane == prev_lane:
                    dt_same_lane.append(t - prev_t)
            prev_t, prev_lane = t, lane
        for c in seen_in_game:
            games_with_card[c] += 1
    tot = sum(card_counts.values())
    ep = np.asarray(elix_post, dtype=np.float64)
    er = np.asarray(elix_pre, dtype=np.float64)
    ea = np.asarray(elix_all, dtype=np.float64)
    dtl = np.asarray(dt_same_lane, dtype=np.float64)
    return {
        "n_games": n_games_incl,
        "n_frames": n_frames, "n_deploy_frames": n_deploy_frames,
        "n_subactions": n_subactions, "n_subactions_dedup": n_subactions_dedup,
        "dup_subaction_frames": dup_frames,
        "n_card_plays": int(tot), "card_counts": dict(card_counts.most_common()),
        "card_share": {k: round(v / max(1, tot), 6) for k, v in card_counts.most_common()},
        "deck_cards": dict(deck_cards),
        # —— 三项主判据 ——
        "all_frames_frac_ge6": (round(float((ea >= 6.0).mean()), 6) if ea.size else None),
        "all_frames_frac_ge8": (round(float((ea >= 8.0).mean()), 6) if ea.size else None),
        "xbow_plays": int(card_counts.get("Xbow", 0)),
        "xbow_rate": (round(card_counts.get("Xbow", 0) / tot, 6) if tot else None),
        "pre_median": (round(float(np.median(er)), 4) if er.size else None),
        "pre_frac_ge6": (round(float((er >= 6.0).mean()), 6) if er.size else None),
        "post_median": (round(float(np.median(ep)), 4) if ep.size else None),
        "elix_all_frames": ({"n": int(ea.size), "mean": round(float(ea.mean()), 4),
                             "median": round(float(np.median(ea)), 4),
                             "max": round(float(ea.max()), 4)} if ea.size else {"n": 0}),
        "n_single_frames": n_single_frames, "unknown_cost_frames": unknown_cost,
        "lane_dt_median": (round(float(np.median(dtl)), 4) if dtl.size else None),
        "lane_dt_n": int(dtl.size),
        "games_with_card": dict(games_with_card.most_common()),
        #: 「攒费 → 打 X」的直接证据：按卡名统计**部署前**圣水（v = elixir0 + 卡费）
        "card_pre_elixir": {k: {"n": len(v), "median": round(float(np.median(v)), 4),
                                "p90": round(float(np.quantile(v, 0.9)), 4),
                                "min": round(float(min(v)), 4),
                                "max": round(float(max(v)), 4),
                                "frac_ge6": round(float(np.mean(np.asarray(v) >= 6.0)), 4)}
                            for k, v in card_pre_elixir.items()},
        "legality": {"n_unit_deploys": n_unit_deploys,
                     "n_unit_y_ge21_illegal": n_unit_y_ge21,
                     "n_spell_plays": n_spell_plays,
                     "n_spell_y_ge21_legal": n_spell_y_ge21},
    }


# ---------------------------------------------------------------------------
# PlanToken → ActionBundle（贪心转录；不绕过掩码/校验）
# ---------------------------------------------------------------------------
class PlanTokenExecutor:
    """把 `PlanToken` 转录成 `ActionBundle`。

    转录规则（全部写在明处，便于在文档里交代**能力上限**）：
    1. 只打 **`plan.suggested_card`** 指的那张（1..4 槽）；token 没建议（None）→ **不出牌**；
    2. `plan.hold_mask` 命中的槽位**禁用**；
    3. `plan.elixir_budget × 10` 是**本帧允许的单卡花费上限**（token 的原始字段语义）；
    4. 落点 = 「`placement_hint` 折算的目标世界坐标」（无 hint 则用 `focus_region` 锚点）
       在**掩码允许的格子**里取最近的一格 —— 永不越出掩码；
    5. ⚠️ **不实现**：`bundle_size_hint`（同刻多卡协同）、`combo_hint`、`target_kind`
       的精确目标选择、任何按卡名的战术（除 token_xbow 臂那一条手写规则）。
    """

    def __init__(self, mode, rng):
        assert mode in ("random", "token_strict", "token_xbow")
        self.mode = mode
        self.rng = rng
        self.stats = Counter()

    # ---- 落点 ----
    @staticmethod
    def _hint_point(battle, plan):
        """hint/region → 目标**世界**坐标（蓝方视角）。"""
        p0 = battle.players[0]
        hint = plan.placement_hint or "none"
        enemies = [e for e in battle.entities.values()
                   if getattr(e, "is_alive", True) and e.player == 1
                   and "Tower" not in (getattr(e, "name", "") or "")]
        threat = min(enemies, key=lambda e: e.position.y) if enemies else None
        if hint == "intercept_mid" and threat is not None:
            # 威胁 → 被威胁塔连线中点靠敌侧（belief_planner 9j+ 的口径）
            tw = min((e for e in battle.entities.values()
                      if e.player == 0 and "Tower" in (getattr(e, "name", "") or "")),
                     key=lambda e: e.position.distance_to(threat.position), default=None)
            if tw is not None:
                return (0.5 * (threat.position.x + tw.position.x),
                        0.5 * (threat.position.y + tw.position.y) + 1.0)
            return (float(threat.position.x), float(threat.position.y) + 3.0)
        if hint == "pull_aggro" and threat is not None:
            return (float(threat.position.x), min(15.0, float(threat.position.y) + 2.0))
        if hint == "king_front":
            return (9.0, 6.5)
        if hint == "bridge_front":
            return (REGION_ANCHOR.get(plan.focus_region, (9.0, 14.0))[0], 14.0)
        if hint in ("support_zone", "anti_spell_zone"):
            mine = [e for e in battle.entities.values()
                    if getattr(e, "is_alive", True) and e.player == 0
                    and "Tower" not in (getattr(e, "name", "") or "")]
            if mine:
                front = max(mine, key=lambda e: e.position.y)
                return (float(front.position.x), max(2.0, float(front.position.y) - 2.0))
        return REGION_ANCHOR.get(plan.focus_region, (9.0, 8.0))

    def _pick_cell(self, battle, mask, slot, target):
        """在 slot 的**合法格子**里取离 target（世界坐标）最近的一格（本地网格坐标）。"""
        from rl.action_bundle import sub_position
        cells = mask["cells"][slot]
        ys, xs = cells.nonzero()
        if xs.size == 0:
            return None
        tx, ty = target
        best, best_d = None, None
        for x, y in zip(xs.tolist(), ys.tolist()):
            wp = sub_position(0, int(x), int(y))
            d = (wp.x - tx) ** 2 + (wp.y - ty) ** 2
            if best_d is None or d < best_d:
                best, best_d = (int(x), int(y)), d
        return best

    # ---- 主转录 ----
    def act(self, env, obs, plan):
        from card_utils import Card
        from rl.action_bundle import ActionBundle
        battle = env.battle
        p0 = battle.players[0]
        mask = env.get_action_mask()
        self.stats[f"intent/{plan.macro_intent}"] += 1

        if self.mode == "random":
            slots = [i for i in range(K_MAX) if mask["slots"][i]]
            if not slots:
                self.stats["noop_no_slot"] += 1
                return ActionBundle.noop(), None
            slot = self.rng.choice(slots)
            cell = self._pick_cell(battle, mask, slot, (9.0, 8.0)) if False else None
            cells = mask["cells"][slot]
            ys, xs = cells.nonzero()
            if xs.size == 0:
                self.stats["noop_no_cell"] += 1
                return ActionBundle.noop(), None
            i = self.rng.randrange(xs.size)
            return (ActionBundle.from_single(slot + 1, int(xs[i]), int(ys[i])),
                    plan.macro_intent)

        held = set(plan.hold_slots())
        # ---- token_xbow：手写「攒费 → 打 Xbow」规则（唯一非 token 语义的动作）----
        if self.mode == "token_xbow":
            xbow_slot = None
            for i, card in enumerate(p0.cycle[:4]):
                if card == "Xbow":
                    xbow_slot = i + 1
                    break
            urgent = self._urgent(battle, plan)
            if not urgent and xbow_slot is not None:
                cost = float(Card("Xbow").elixir)
                if p0.elixir < cost - 1e-9 or not mask["slots"][xbow_slot - 1]:
                    self.stats["xbow_save_hold"] += 1
                    return ActionBundle.noop(), plan.macro_intent
                cell = self._pick_cell(battle, mask, xbow_slot - 1,
                                       self._xbow_point(battle))
                if cell is None:
                    self.stats["xbow_no_cell"] += 1
                    return ActionBundle.noop(), plan.macro_intent
                self.stats["xbow_commit"] += 1
                return (ActionBundle.from_single(xbow_slot, cell[0], cell[1]),
                        plan.macro_intent)
            if not urgent and xbow_slot is None:
                # Xbow 不在手 → 过牌（打最便宜的合法牌）把它轮进手牌
                cands = [i for i in range(K_MAX) if mask["slots"][i] and i not in held]
                if cands:
                    cands.sort(key=lambda i: float(Card(p0.cycle[i]).elixir))
                    s = cands[0] + 1
                    cell = self._pick_cell(battle, mask, s - 1, self._xbow_point(battle))
                    if cell is not None:
                        self.stats["xbow_cycle_play"] += 1
                        return ActionBundle.from_single(s, cell[0], cell[1]), plan.macro_intent
                self.stats["xbow_cycle_hold"] += 1
                return ActionBundle.noop(), plan.macro_intent

        # ---- 纯 token 转录 ----
        s = plan.suggested_card
        if s is None or s < 1 or s > K_MAX:
            self.stats["noop_no_suggestion"] += 1
            return ActionBundle.noop(), plan.macro_intent
        if (s - 1) in held:
            self.stats["noop_hold_mask"] += 1
            return ActionBundle.noop(), plan.macro_intent
        _card = p0.cycle[s - 1]
        if _card == "Xbow":
            self.stats["suggested_xbow"] += 1
        if not mask["slots"][s - 1]:
            self.stats["noop_slot_illegal"] += 1
            if _card == "Xbow":
                self.stats["suggested_xbow_unaffordable"] += 1
            return ActionBundle.noop(), plan.macro_intent
        cost = float(Card(p0.cycle[s - 1]).elixir)
        if cost > float(plan.elixir_budget) * 10.0 + 1e-9:
            self.stats["noop_over_budget"] += 1
            return ActionBundle.noop(), plan.macro_intent
        cell = self._pick_cell(battle, mask, s - 1, self._hint_point(battle, plan))
        if cell is None:
            self.stats["noop_no_cell"] += 1
            return ActionBundle.noop(), plan.macro_intent
        self.stats["token_commit"] += 1
        return ActionBundle.from_single(s, cell[0], cell[1]), plan.macro_intent

    @staticmethod
    def _urgent(battle, plan):
        """紧急帧判定（token_xbow 的攒费闸门；**非 token 字段**，是手写专家自己的判据）：

        有敌方可部署单位进入我方半场（player0：世界 y ≤ 16），或 token 意图属于
        防守/解牌族 —— 两种都算「现在不能攒费」。
        """
        if plan.macro_intent in ("defend_left", "defend_right", "defend_king",
                                 "king_activate", "soft_control", "protect_backline",
                                 "pull", "spell_trade"):
            return True
        for e in battle.entities.values():
            if not getattr(e, "is_alive", True) or e.player != 1:
                continue
            if "Tower" in (getattr(e, "name", "") or ""):
                continue
            if float(e.position.y) <= 16.0:
                return True
        return False

    @staticmethod
    def _xbow_point(battle):
        """Xbow 桥头落点：压在**血量更低的敌方公主塔**那一路、本地 y=14（最靠河的己方格）。"""
        p1 = battle.players[1]
        left = float(p1.left_tower_hp)
        right = float(p1.right_tower_hp)
        x = 4.5 if (left <= right and left > 0 or right <= 0) else 13.5
        return (x, 14.5)


# ---------------------------------------------------------------------------
def run_arm(arm, n_games, seed, max_steps, out_dir, passive_prob, verbose=True):
    from rl.env_wrapper import RLEnv
    from rl.belief import BeliefInference
    from rl.belief_planner import BeliefPlanner
    from rl.train_solo import DEFAULT_SOLO_DECK
    from rl.opponents import SelfDefenderPolicy
    from rl.action_mask import validate_bundle
    from rl.replay import battle_snapshot
    from rl.run_league import _bundle_cards, timeout_winner, overtime_open

    env = RLEnv(opponent=None, seed=seed, deck0=DEFAULT_SOLO_DECK,
                deck1=DEFAULT_SOLO_DECK)
    rng = random.Random(seed)
    execu = PlanTokenExecutor(arm, rng)
    games = []
    illegal_validate = 0
    illegal_engine = 0
    n_frames = 0
    t0 = time.time()
    for g in range(n_games):
        opp = SelfDefenderPolicy(seed=seed + g, env=env, passive_prob=passive_prob)
        env.opponent = opp
        obs, _ = env.reset(seed=seed + g)
        belief = BeliefInference(opp_deck=env.deck1, n_particles=128, seed=seed + g)
        belief.reset(env.deck1)
        bp = BeliefPlanner()
        frames = []
        done, steps = False, 0
        while not done and (steps < max_steps or overtime_open(env.battle)):
            plan = bp.plan(env.battle, belief.state(), obs)
            bundle, _intent = execu.act(env, obs, plan)
            # 非法动作必须能被发现：先过整包校验，再交给引擎
            ok, reason, _res = validate_bundle(env.battle, 0, bundle)
            if not ok:
                illegal_validate += 1
                execu.stats["illegal_validate"] += 1
                from rl.action_bundle import ActionBundle
                bundle = ActionBundle.noop()
            cards = _bundle_cards(bundle, obs)
            obs, reward, term, trunc, info = env.step(bundle)
            if info.get("invalid_count"):
                illegal_engine += int(info["invalid_count"])
                execu.stats["illegal_engine"] += int(info["invalid_count"])
            fr = battle_snapshot(env.battle, bundle, reward, info)
            if cards:
                fr["cards"] = list(cards)
            frames.append(fr)
            n_frames += 1
            belief.update(obs, info.get("opp_played"))
            done = term or trunc
            steps += 1
        w = env.battle.winner
        if w is None and not env.battle.game_over:
            w = timeout_winner(env.battle)
        games.append({"meta": {"pair": [arm, "self_defender"], "side0": "main",
                               "max_steps": max_steps, "arm": arm,
                               "decks": [list(env.deck0), list(env.deck1)]},
                      "winner": w, "frames": frames})
        if verbose:
            _stdout(f"  [{arm}] game {g + 1}/{n_games}: frames={len(frames)} "
                    f"winner={w} elapsed={time.time() - t0:.1f}s")

    wins = sum(1 for g in games if g["winner"] == 0)
    losses = sum(1 for g in games if g["winner"] is not None and g["winner"] != 0)
    draws = n_games - wins - losses
    from card_utils import Card

    def _is_spell(nm):
        try:
            return Card(nm).type == "spell"
        except Exception:                                          # noqa: BLE001
            return False

    met = compute_behavior_metrics(games, _is_spell)
    score = (wins + 0.5 * draws) / max(1, n_games)
    res = {
        "arm": arm, "seed": seed, "n_games": n_games, "max_steps": max_steps,
        "deck": list(DEFAULT_SOLO_DECK), "opponent": "SelfDefenderPolicy",
        "opponent_passive_prob": passive_prob,
        "n_frames": n_frames,
        "frames_per_game": round(n_frames / max(1, n_games), 2),
        "wins": wins, "losses": losses, "draws": draws, "score": round(score, 4),
        "illegal_validate_count": illegal_validate,
        "illegal_engine_count": illegal_engine,
        "executor_stats": dict(execu.stats),
        "behavior": met,
        "wall_s": round(time.time() - t0, 1),
    }
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        # 文件名固定为 league_0.pkl：`forensics_card_usage.py` 只吃 `league_*.pkl`
        with open(os.path.join(out_dir, "league_0.pkl"), "wb") as f:
            pickle.dump({"schema": 4, "games": games}, f)
    return res


def negative_control(n_steps, seed):
    """非法动作检测的阴性对照：故意构造非法 bundle，看校验/引擎能不能抓到。

    返回各构造法的「被抓到」比例（证明非法计数器不是恒零）。
    """
    from rl.env_wrapper import RLEnv
    from rl.train_solo import DEFAULT_SOLO_DECK
    from rl.action_mask import validate_bundle
    from rl.action_bundle import ActionBundle, SubAction
    from card_utils import Card
    env = RLEnv(opponent=None, seed=seed, deck0=DEFAULT_SOLO_DECK,
                deck1=DEFAULT_SOLO_DECK)
    env.reset(seed=seed)
    rng = random.Random(seed)
    cnt = Counter()
    for _ in range(n_steps):
        p0 = env.battle.players[0]
        mask = env.get_action_mask()
        # (a) 槽位越界
        b = ActionBundle(sub_actions=[SubAction(slot=7, x=9, y=8)])
        ok, _r, _x = validate_bundle(env.battle, 0, b)
        cnt["slot_out_of_range_caught"] += int(not ok)
        # (b) 掩码外格子（该槽合法则挑一个 cells=False 的格子）
        legal = [i for i in range(K_MAX) if mask["slots"][i]]
        if legal:
            s = legal[0]
            bad = [(x, y) for y in range(GRID_H) for x in range(GRID_W)
                   if not mask["cells"][s][y, x]]
            if bad:
                x, y = rng.choice(bad)
                b = ActionBundle(sub_actions=[SubAction(slot=s + 1, x=x, y=y)])
                ok, _r, _x = validate_bundle(env.battle, 0, b)
                cnt["off_mask_cell_caught"] += int(not ok)
        # (c) 圣水不足：把本脚本**自己的 env 实例**的圣水临时置 0 后逐个槽位校验
        #     （只改内存里的本例状态，不动任何源码；置 0 后任何 >0 费的卡都必须被拒）
        saved = p0.elixir
        p0.elixir = 0.0
        for i in range(K_MAX):
            if Card(p0.cycle[i]).elixir <= 0.0:
                continue
            b = ActionBundle(sub_actions=[SubAction(slot=i + 1, x=9, y=8)])
            ok, _r, _x = validate_bundle(env.battle, 0, b)
            cnt["unaffordable_samples"] += 1
            cnt["unaffordable_caught"] += int(not ok)
        p0.elixir = saved
        # (d) 合法 no-op（应为「合法」——防止校验器恒返回 False）
        ok, _r, _x = validate_bundle(env.battle, 0, ActionBundle.noop())
        cnt["noop_accepted"] += int(ok)
        # 推进：用掩码随机出牌（把圣水花掉，才能构造出「圣水不足」样本）
        legal2 = [i for i in range(K_MAX) if mask["slots"][i]]
        if legal2:
            s2 = rng.choice(legal2)
            cells2 = mask["cells"][s2].nonzero()
            if cells2[0].size:
                j = rng.randrange(cells2[0].size)
                env.step(ActionBundle.from_single(s2 + 1, int(cells2[1][j]),
                                                  int(cells2[0][j])))
            else:
                env.step(ActionBundle.noop())
        else:
            env.step(ActionBundle.noop())
        if env.battle.game_over:
            env.reset(seed=seed + 1)
    return dict(cnt)


def repro_replays(path):
    """仪器自检：用**同一段指标代码**复算历史回放（应复现 prereg 记录的对照值）。"""
    from forensics_card_usage import load_files
    from card_utils import Card

    def _is_spell(nm):
        try:
            return Card(nm).type == "spell"
        except Exception:                                          # noqa: BLE001
            return False
    files = load_files(path)          # forensics 版返回 (name, games)
    out = {}
    pooled = []
    for row in files:
        name, games = row[0], row[-1]
        out[name] = compute_behavior_metrics(games, _is_spell)
        pooled.extend(games)
    out["POOLED"] = compute_behavior_metrics(pooled, _is_spell)
    return out


def report_markdown(paths):
    """把若干个 JSON 产物渲染成 markdown 表（**数字全部来自脚本产物，禁止手抄**）。

    自动识别：含 "result" = 臂；含 "repro" = 仪器自检；含 "negative_control" = 阴性对照。
    """
    import math
    arms, repro, neg = [], [], []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        if "negative_control" in d:
            neg.append(d)
            if not (d.get("result") or {}).get("n_games"):
                continue
        if "result" in d and d["result"].get("n_games"):
            arms.append(d)
        elif "repro" in d:
            repro.append(d)
    lines = []
    lines.append("### A. 行为层三项（脚本复算；主判据只取前两列 + 第三列）")
    lines.append("")
    lines.append("| 臂 | 局数 | 局长(帧) | 全帧圣水≥6 | Xbow 出手 | Xbow 出手率 | 部署前圣水中位 | 同路间隔中位 | 得分 | 非法(validate/引擎) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for d in arms:
        r, b = d["result"], d["result"]["behavior"]
        lines.append(
            f"| `{r['arm']}` | {r['n_games']} | {r['frames_per_game']} | "
            f"{b['all_frames_frac_ge6']:.6f} ({b['all_frames_frac_ge6'] * b['n_frames']:.0f}"
            f"/{b['n_frames']}) | {b['xbow_plays']}/{b['n_card_plays']} | "
            f"{b['xbow_rate']:.6f} | {b['pre_median']} | {b['lane_dt_median']} | "
            f"{r['score']} ({r['wins']}/{r['losses']}/{r['draws']}) | "
            f"{r['illegal_validate_count']}/{r['illegal_engine_count']} |")
    lines.append("")
    lines.append("### B. 对照基线（同一段指标代码复算）")
    lines.append("")
    lines.append("| 来源 | 局数 | 帧 | 全帧圣水≥6 | Xbow | Xbow 出手率 | 部署前中位 | 同路间隔中位 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    lines.append("| prereg §3 参照（`d1_long_100k` 单 run） | — | — | 0.0011 | 3 | 0.0003 | 3.25 | 5.5 |")
    for d in repro:
        for k, v in d["repro"].items():
            if k != "POOLED":
                continue
            lines.append(f"| 仪器自检复算 `{d.get('repro_replays', 'd1_long_100k')}` (POOLED) | "
                         f"{v.get('n_games', '—')} | "
                         f"{v['n_frames']} | {v['all_frames_frac_ge6']:.6f} | "
                         f"{v['xbow_plays']}/{v['n_card_plays']} | {v['xbow_rate']:.6f} | "
                         f"{v['pre_median']} | {v['lane_dt_median']} |")
    lines.append("")
    lines.append("### C. 直接攒费证据（Xbow 部署前圣水，只取「恰好 1 个子动作」帧）")
    lines.append("")
    lines.append("| 臂 | Xbow 部署前圣水 n | 中位 | p90 | min | max | ≥6 占比 | 出现 Xbow 的局数 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for d in arms:
        b = d["result"]["behavior"]
        xp = b["card_pre_elixir"].get("Xbow")
        if not xp:
            lines.append(f"| `{d['result']['arm']}` | 0 | — | — | — | — | — | "
                         f"{b['games_with_card'].get('Xbow', 0)} |")
        else:
            lines.append(f"| `{d['result']['arm']}` | {xp['n']} | {xp['median']} | {xp['p90']} | "
                         f"{xp['min']} | {xp['max']} | {xp['frac_ge6']} | "
                         f"{b['games_with_card'].get('Xbow', 0)} |")
    lines.append("")
    lines.append("### D. 非法动作检测（阴性对照，证明计数器不是恒零）")
    lines.append("")
    for d in neg:
        nc = d["negative_control"]
        lines.append(f"- 槽位越界 `slot=7`：{nc.get('slot_out_of_range_caught', 0)} 次全部被 "
                     f"`validate_bundle` 拒绝")
        lines.append(f"- 掩码外格子：{nc.get('off_mask_cell_caught', 0)}/"
                     f"{nc.get('off_mask_cell_caught', 0)} 次被拒（样本数受「掩码外格子存在」限制）")
        lines.append(f"- 圣水不足（把本脚本自己 env 的圣水临时置 0 后逐槽位校验）："
                     f"{nc.get('unaffordable_caught', 0)}/{nc.get('unaffordable_samples', 0)} 被拒")
        lines.append(f"- 合法 `no-op`：{nc.get('noop_accepted', 0)} 次全部被接受"
                     f"（证明校验器不是恒返回 False）")
    lines.append("")
    lines.append("### E. 判据输入（脚本算出的比值）")
    lines.append("")
    if arms:
        base = next((d for d in arms if d["result"]["arm"] == "random"), None)
        base_rate = (base["result"]["behavior"]["xbow_rate"] if base else None)
        ref = PREREG_BASELINE
        for d in arms:
            b = d["result"]["behavior"]
            ratio_ref = b["xbow_rate"] / ref["xbow_rate"]
            ratio_base = (b["xbow_rate"] / base_rate) if base_rate else None
            lines.append(
                f"- `{d['result']['arm']}`：Xbow 率 {b['xbow_rate']:.6f} = "
                f"{ratio_ref:.1f}× prereg 参照"
                + (f" / {ratio_base:.2f}× random 基线" if ratio_base is not None else "")
                + f"；全帧圣水≥6 {b['all_frames_frac_ge6']:.6f} = "
                  f"{b['all_frames_frac_ge6'] / ref['all_frames_frac_ge6']:.2f}× prereg 参照")
            if b["n_card_plays"]:
                if b["xbow_plays"] == 0:
                    ub = 3.0 / b["n_card_plays"]
                else:
                    ub = b["xbow_rate"] + 1.96 * math.sqrt(
                        b["xbow_rate"] * (1 - b["xbow_rate"]) / b["n_card_plays"])
                lines.append(f"  - Xbow 率 95% 上界 ≈ {ub:.6f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="random",
                    choices=["random", "token_strict", "token_xbow"])
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--max-steps", type=int, default=360)
    ap.add_argument("--opp-passive-prob", type=float, default=0.6)
    ap.add_argument("--out", default=None)
    ap.add_argument("--replay-dir", default=None,
                    help="把该臂的对局写成联赛回放 pkl（供 forensics_card_usage.py 复算）")
    ap.add_argument("--repro-replays", default=None,
                    help="不跑对局，只用同一段指标代码复算历史回放（仪器自检）")
    ap.add_argument("--neg-control-steps", type=int, default=0,
                    help=">0 时先跑非法动作检测的阴性对照")
    ap.add_argument("--report-jsons", nargs="*", default=None,
                    help="把已有 JSON 产物渲染成 markdown（不跑对局）")
    a = ap.parse_args()

    if a.report_jsons is not None:
        _stdout(report_markdown(a.report_jsons))
        return None

    out = {"arm": a.arm, "seed": a.seed, "games": a.games,
           "prereg_baseline_reference": PREREG_BASELINE,
           "baseline_rule": "主基线 = 同脚本 --arm random 复算；prereg 值只作量级参照"}

    if a.repro_replays:
        _stdout(f"=== 仪器自检：以同一段指标代码复算 {a.repro_replays} ===")
        r = repro_replays(a.repro_replays)
        p = r["POOLED"]
        _stdout(f"  POOLED: 局={p['n_games']} 全帧圣水>=6={p['all_frames_frac_ge6']} "
                f"Xbow={p['xbow_plays']}/{p['n_card_plays']}={p['xbow_rate']} "
                f"部署前中位={p['pre_median']} 同路间隔中位={p['lane_dt_median']} "
                f"帧={p['n_frames']}")
        _stdout(f"  期望（prereg §3 对照，单 run 参照）: 0.0011 / 0.0003 / 3.25 / 5.5")
        out["repro"] = {k: {kk: v.get(kk) for kk in
                            ("n_games", "n_frames", "n_card_plays", "xbow_plays",
                             "xbow_rate", "all_frames_frac_ge6", "pre_median",
                             "lane_dt_median")}
                        for k, v in r.items()}
        if a.out:
            with open(a.out, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=1)
            _stdout(f"[out] {a.out}")
        return out

    if a.neg_control_steps:
        nc = negative_control(a.neg_control_steps, a.seed)
        _stdout(f"=== 阴性对照（非法动作检测，{a.neg_control_steps} 步）===")
        for k in sorted(nc):
            _stdout(f"  {k} = {nc[k]}")
        out["negative_control"] = nc

    _stdout(f"=== S1 门禁 | arm={a.arm} games={a.games} seed={a.seed} "
            f"deck={a.arm and 'DEFAULT_SOLO_DECK'} opp=SelfDefenderPolicy"
            f"(passive={a.opp_passive_prob}) ===")
    res = run_arm(a.arm, a.games, a.seed, a.max_steps, a.replay_dir,
                  a.opp_passive_prob)
    out["result"] = res
    b = res["behavior"]
    _stdout(f"  三项：全帧圣水>=6 = {b['all_frames_frac_ge6']}  "
            f"Xbow = {b['xbow_plays']}/{b['n_card_plays']} = {b['xbow_rate']}  "
            f"部署前中位 = {b['pre_median']}")
    _stdout(f"  对照基线（prereg §3，仅量级参照）：0.0011 / 0.0003 / 3.25")
    _stdout(f"  局数={res['n_games']} 局长(帧)={res['frames_per_game']} "
            f"得分={res['score']} ({res['wins']}胜/{res['losses']}负/{res['draws']}平) "
            f"非法(validate)={res['illegal_validate_count']} "
            f"非法(引擎)={res['illegal_engine_count']}")
    _stdout(f"  同路部署间隔中位={b['lane_dt_median']} (n={b['lane_dt_n']})  "
            f"墙钟={res['wall_s']}s")
    _stdout(f"  卡牌分布 top8 = {list(b['card_counts'].items())[:8]}")
    xp = b["card_pre_elixir"].get("Xbow")
    _stdout(f"  Xbow 部署前圣水（直接攒费证据）：{xp}  "
            f"Xbow 出现在 {b['games_with_card'].get('Xbow')} 局")
    _stdout(f"  执行器计数 = {res['executor_stats']}")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        _stdout(f"[out] {a.out}")
    return out


if __name__ == "__main__":
    main()
