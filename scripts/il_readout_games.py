# -*- coding: utf-8 -*-
"""**IL（行为克隆）策略的对局观测 runner** —— 跑几局、落 schema 5 录像，供 dashboard `--solo` 看/播。

用途（用户 2026-09-20：「给我 dashboard，我看几场 IL 后的对局」）：把一个 BC ckpt
（例如 `runs/_fl_il_bc/bc_fl.pt`，用 FirstLight 的 IL_Replay 训出来的）当成**生产策略**跑真引擎，
逐块落盘**生产同格式录像** + `solo_state.json`，dashboard 边看指标边播回放。

* **p0（被测方）= IL 策略**：`FollowerPolicy.act(...)`，默认 `deterministic=True`（argmax，
  与生产评估的脚本对手同约定）。信念/plan 链路与 `rl/human_play.py:142-154` **逐句同序**
  （`BeliefInference.encode` → `BeliefPlanner.plan` → `act`）。
* **p1（对手）= 同一 ckpt 的独立冻结副本**（自对弈镜像），或 `--opponent main --main-ckpt <我方 ckpt>`。
* **卡组**：`--decks solo` = `rl.train_solo.DEFAULT_SOLO_DECK`（与既有评估可比）；
  `--decks fl` = 从 FL 的 IL_Replay JSONL 里抽**真实人类卡组对**（过滤口径与
  `scripts/fl_il_to_bc.py` 的 `--mode samples` **逐条一致**：模式白名单 + 时长 ≤300 s + 两侧 8 张全部可解析）。
* `--hidden {carry,none}`：**carry**（默认）= 生产约定，跨帧带 GRU 隐状态；
  **none** = 每条帧都当首帧（= BC 训练时的口径，`human_play.py:224`）。

⚠️ 描述性读数、非判据（【R3】【R5】：n=1 ckpt、单 seed）。本 runner **不改任何训练/奖励/判定代码**。

用法（必须在 `src/clasher_new` 下运行，或从仓库根用绝对路径）:
    cd src/clasher_new && ../../.venv/Scripts/python.exe ../../scripts/il_readout_games.py \
        --ckpt ../../runs/_fl_il_bc/bc_fl.pt --decks solo --games 10 --block 5 \
        --out ../../runs/il_readout_solo --seed 0
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src", "clasher_new")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from rl.io_bootstrap import force_utf8_stdout  # noqa: E402

force_utf8_stdout()

import numpy as np  # noqa: E402

#: 与 `scripts/fl_il_to_bc.py` 同口径（改这里必须同步那两处）
ALLOWED_MODES = ("Ranked", "Ladder", "1v1 Battle", "1v1",
                 "Grand Challenge", "Classic Challenge")
VARIANTS = ("-ev1", "-ev2", "-ev3", "-hero")
TIME_CAP = 300.0


def _map_key(key):
    from card_aliases import resolve_card
    k = key or ""
    for v in VARIANTS:
        if k.endswith(v):
            k = k[: -len(v)]
    try:
        return resolve_card(k) if k else None
    except Exception:  # noqa: BLE001
        return None


def load_fl_deck_pairs(jsonl, limit=4000):
    """从 `_fl_il_extract.py` 产出的 JSONL 里取**可用的人类卡组对**（过滤口径同 fl_il_to_bc）。"""
    pairs, seen = [], set()
    with open(jsonl, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            if not line.strip():
                continue
            r = json.loads(line)
            if r["gm"] not in ALLOWED_MODES:
                continue
            if not isinstance(r["dur"], (int, float)) or r["dur"] > TIME_CAP:
                continue
            decks, bad = {}, False
            for side in ("team", "opponent"):
                names = [_map_key(k) for k, _lv in r["decks"][side]]
                if any(n is None for n in names) or len(names) != 8:
                    bad = True
                    break
                decks[side] = names
            if bad:
                continue
            key = tuple(decks["team"]) + tuple(decks["opponent"])
            if key in seen:
                continue
            seen.add(key)
            pairs.append((decks["team"], decks["opponent"], r["tag"]))
    return pairs


def build_plan_vec(bp, env, belief, obs, known_opp, plan_dim):
    """按 ckpt 的 `plan_dim` 产 plan 向量（W2/W3 的**推理侧唯一实现**，与 `fl_il_to_bc.py` 同口径）。

    `plan_dim == PLAN_BASE_DIM(58)` ⇒ 旧口径（逐位同旧行为）；
    `plan_dim == 58 + PLAN_EXTRA_DIM(17)` ⇒ 尾部追加手牌打分/信息分；
    其它值 ⇒ **显式报错**（不静默产短向量：那会让 `plan_mlp` 直接 shape 崩或读错列）。
    """
    from rl.plan_space import PLAN_BASE_DIM
    from rl.hand_score import PLAN_EXTRA_DIM, plan_extras
    pv = bp.plan(env.battle, belief.state(), obs).to_vector()
    pd = int(plan_dim)
    if pd == int(pv.shape[0]):
        return pv
    if pd != int(PLAN_BASE_DIM) + int(PLAN_EXTRA_DIM) or int(pv.shape[0]) != int(PLAN_BASE_DIM):
        raise ValueError(f"plan_dim={pd} 与基础布局 {PLAN_BASE_DIM}+{PLAN_EXTRA_DIM} 不符"
                         f"（bp 产出 {pv.shape[0]}）⇒ 口径不明，拒绝静默继续")
    _st = belief.state()
    ext = plan_extras(own_cards=list(env.battle.players[0].cycle[:4]),
                      own_elixir=float(env.battle.players[0].elixir),
                      opp_cards=list(env.deck1), opp_hand_probs=_st.hand_probs,
                      opp_elixir_est=float(_st.elixir_mean),
                      time_s=float(env.battle.time), known_opp=int(known_opp))
    return np.concatenate([pv, ext]).astype(np.float32)


def _opp_play_name(nm):
    """`opp_played` 的元素实测是 dict ⇒ 统一取卡名（读成 str 的旧路径也兼容）。"""
    if isinstance(nm, dict):
        nm = nm.get("card") or nm.get("card_name") or nm.get("name")
    return nm if isinstance(nm, str) else None


#: ★ W3 前期窗口（秒）——与 `rl/hand_score.py::T_INFO` **同一约定**（改一处必须同步）
INFO_T_WINDOW = 30.0


def _median(xs):
    return float(np.median(xs)) if len(xs) else None


def build_policy(ckpt, device, torch, label):
    from rl.follower import FollowerPolicy
    d = torch.load(ckpt, map_location="cpu")
    sd = d["state_dict"] if isinstance(d, dict) and "state_dict" in d else d
    meta = {k: v for k, v in d.items() if k != "state_dict"} if isinstance(d, dict) else {}
    pol = FollowerPolicy(hidden=int(meta.get("hidden_dim", 128)),
                         plan_dim=int(meta["plan_dim"]), belief_dim=int(meta["belief_dim"]),
                         value_bypass=bool(meta.get("value_bypass", False)),
                         value_independent=bool(meta.get("value_independent", False)),
                         intent_options=bool(meta.get("intent_options", False)),
                         #: ★ 2026-09-22 独立 act 头（预注册 §7）：**必须读**，
                         #: 否则给 5 维 `slot_head` 的 ckpt 建 6 维头 ⇒ 形状不匹配 ⇒
                         #: `load_state_dict(strict=False)` **静默随机**（实测过的坑，见 §7.7）
                         decoupled_act=bool(meta.get("decoupled_act", False)),
                         #: ★ 2026-09-22 W2/W3：**必须读**（同 §7.7 纪律）。它不改形状，
                         #: 但决定 plan 尾列**是否被抹零** ⇒ 读漏 = 形状匹配、语义相反（静默）。
                         plan_extras_zero=int(meta.get("plan_extras_zero", 0)))
    miss, unexp = pol.load_state_dict(sd, strict=False)
    if miss or unexp:
        print(f"[warn] {label} 载入 missing={len(miss)} unexpected={len(unexp)}", flush=True)
    pol.eval()
    pol.to_device(device)
    return pol, meta


def _gate_p_stop(policy, obs, tok, plan, masks):
    """复刻 `FollowerPolicy.act()` 第一步 6 类分布里的 `p(STOP)`（`hidden=None` 口径）。

    ⚠️ **必须与 `scripts/il_eval_holdout.py::first_option_probs` 保持同步**（同一复刻，两处实现）：
    那里是**留出侧**读数（预注册 §6 C6-4 指定口径），这里是**部署侧**门。**改一处必须改另一处。**
    做法 = `_encode_parts → gru_cell(zeros) → slot_head + plan_bias → 掩码 → softmax[STOP]`，
    **不改 `follower.py`**（与 `il_nll_decompose.py` 同款）。
    """
    import torch
    from rl.follower import STOP_IDX
    with torch.no_grad():
        _fused, enc = policy._encode_parts(obs, tok, plan)
        h = policy.gru_cell(enc, torch.zeros(1, policy.hidden_dim, device=policy.device))
        slot_bias, _cb = policy._plan_biases(plan)
        logits = policy.slot_head(h) + slot_bias
        sm = policy._slot_mask_tensor(masks[0])
        logits = logits.masked_fill(sm == 0, -1e9)
        p = torch.softmax(logits, dim=-1)
    return p[0, STOP_IDX]


def main():
    ap = argparse.ArgumentParser(description="IL 策略对局观测 runner（落 schema 5 录像供 dashboard）")
    ap.add_argument("--ckpt", required=True, help="IL（BC）ckpt")
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--block", type=int, default=5, help="每多少局落一次盘")
    ap.add_argument("--out", required=True, help="run 目录（写 replays/ + solo_state.json）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--decks", choices=("solo", "fl"), default="solo")
    ap.add_argument("--fl-jsonl", default=None,
                    help="--decks fl 时的回放 JSONL（Windows 路径，如 E:\\fl_il_data\\replays_part000000.jsonl）")
    ap.add_argument("--opponent", choices=("self", "main"), default="self")
    ap.add_argument("--main-ckpt", default=None, help="--opponent main 时我方 ckpt")
    ap.add_argument("--hidden", choices=("carry", "none"), default="carry",
                    help="p0 的隐状态口径；none = 逐帧（= BC 训练口径），carry = 生产约定")
    ap.add_argument("--opp-hidden", choices=("same", "carry", "none"), default="same",
                    help="p1（FollowerOpponent）的隐状态口径；same = 跟随 --hidden。默认 same："
                         "FollowerOpponent 内部恒 carry，而逐帧 BC 的 ckpt 在 carry 下会塌成 STOP"
                         "（判读 §16）⇒ 此前 p1 每局只出 1-2 张、胜率虚高（口径不对称 bug）")
    ap.add_argument("--sample", action="store_true", help="随机采样而非 argmax（默认 argmax）")
    #: ★ 2026-09-22 预注册 §6「解耦组合臂」：门（只看 STOP logit）用另一个 ckpt，动作走 --ckpt。
    #: τ 默认 **0.6610** = §6 C6-5 写死的「留出侧基率匹配」档（留出基率 0.2242）。
    #: ⚠️ §6 纪律：**不许在部署侧扫 τ 拟合**。
    ap.add_argument("--gate-ckpt", default=None,
                    help="解耦组合臂的 act 门 ckpt（只读它的 p(STOP)）；不给 = 单 ckpt 原行为")
    ap.add_argument("--gate-threshold", type=float, default=0.6610,
                    help="门的 STOP 阈值 τ（默认 0.6610，预注册 §6 C6-5 写死档）")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    import torch
    from rl.env_wrapper import RLEnv
    from rl.belief import BeliefInference
    from rl.belief_planner import BeliefPlanner
    from rl import train_solo as ts
    from rl.train_follower import FollowerOpponent
    from rl.action_bundle import ActionBundle

    class _FrameWiseOpponent(FollowerOpponent):
        """**逐帧口径**的对手包装：`FollowerOpponent` 内部恒把 `self.hidden` 带下去（生产约定），
        而逐帧 BC 的 ckpt 在该口径下会塌成 STOP（判读 §16）⇒ 每次决策前清空隐状态。
        ⚠️ 必须在**子类**里覆盖 `__call__`：实例上挂 `__call__` 无效（特殊方法按**类型**查找）。
        """

        def __call__(self, obs):
            self.hidden = None
            return super().__call__(obs)
    from rl.run_league import LeagueGameRecorder, _bundle_cards, timeout_winner
    from rl.replay import save_league_replays
    from rl.overtime import overtime_open

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = a.out
    os.makedirs(os.path.join(out, "replays"), exist_ok=True)

    pol, meta = build_policy(a.ckpt, dev, torch, "IL 策略")
    if a.opponent == "main":
        if not a.main_ckpt:
            raise SystemExit("--opponent main 需要 --main-ckpt")
        opp_pol, _ = build_policy(a.main_ckpt, dev, torch, "我方主策略")
    else:
        opp_pol, _ = build_policy(a.ckpt, dev, torch, "对手(同 ckpt 冻结副本)")
    gate_pol, gate_meta = (build_policy(a.gate_ckpt, dev, torch, "act 门")
                           if a.gate_ckpt else (None, {}))
    if gate_pol is not None:
        print(f"[runner] ★ 解耦组合臂（预注册 §6）：门 ckpt={a.gate_ckpt} τ={a.gate_threshold}"
              f"（门只看 STOP logit、hidden=None 逐帧；动作走 --ckpt={a.ckpt}）", flush=True)
    bp = BeliefPlanner()

    deck_pairs = None
    if a.decks == "fl":
        if not a.fl_jsonl or not os.path.isfile(a.fl_jsonl):
            raise SystemExit(f"--decks fl 需要 --fl-jsonl（给的：{a.fl_jsonl}）")
        deck_pairs = load_fl_deck_pairs(a.fl_jsonl)
        if not deck_pairs:
            raise SystemExit("FL JSONL 里没有可用的人类卡组对")
        print(f"[runner] FL 卡组对：{len(deck_pairs)} 套（过滤口径同 fl_il_to_bc）", flush=True)

    #: p1 的有效口径：`same` 跟随 p0（默认）——修掉「p0 逐帧 / p1 恒 carry」的口径不对称
    opp_hidden = a.hidden if a.opp_hidden == "same" else a.opp_hidden
    print(f"[runner] ckpt={a.ckpt} device={dev} 局数={a.games} 每块={a.block} "
          f"decks={a.decks} opponent={a.opponent} hidden={a.hidden} opp_hidden={opp_hidden} "
          f"deterministic={not a.sample}", flush=True)

    games, hist = [], []
    cw = cl = cd = frames_total = plays_total = empty_total = 0
    rew_total = 0.0
    t0 = time.time()
    bf = bg = bw = bl = bd = 0
    brew = 0.0
    card_hist = collections.Counter()
    playable = forced_stop = stop_when_playable = 0
    gate_eval = gate_stop = 0     #: ★ §6：门被求值帧数 / 门判 STOP 帧数
    opp_card_hist = collections.Counter()
    #: ★ W3 前期窗口（≤ INFO_T_WINDOW s）聚合（J-W3.2 方向判据的读数）
    win_frames, win_known, win_plays, win_distinct = 0, [], 0.0, 0.0
    win_first_t = []

    for g in range(a.games):
        if deck_pairs is not None:
            d0, d1, tag = deck_pairs[(a.seed + g) % len(deck_pairs)]
            d0, d1 = list(d0), list(d1)
        else:
            d0 = d1 = list(ts.DEFAULT_SOLO_DECK)
            tag = None
        env = RLEnv(opponent=None, seed=a.seed + 31 * g, card_level=11, deck0=d0, deck1=d1)
        _cls = FollowerOpponent if opp_hidden == "carry" else _FrameWiseOpponent
        env.opponent = _cls(
            opp_pol, env,
            belief=BeliefInference(opp_deck=list(env.deck0), n_particles=128, seed=a.seed + g),
            deterministic=True)

        obs, _ = env.reset(seed=a.seed + 2000 + g)
        belief0 = BeliefInference(opp_deck=list(env.deck1), n_particles=128, seed=a.seed + g)
        belief0.reset(env.deck1)
        rec = LeagueGameRecorder("il", "fl@0" if tag is None else "fl@1", "il", a.max_steps)
        rec.set_decks(env.deck0, env.deck1)
        done = False
        steps = 0
        hidden = None
        _known_opp = set()          #: ★ W2/W3：本局对手已打出过的**不同**卡数（信息分原料）
        _win_play_n, _win_cards, _first_t = 0, set(), None   #: ★ W3 前期窗口（≤30 s）统计
        while not done and (steps < a.max_steps or overtime_open(env.battle)):
            tok = belief0.encode(obs, None)
            plan = build_plan_vec(bp, env, belief0, obs, len(_known_opp), meta["plan_dim"])
            h_in = hidden if a.hidden == "carry" else None
            _gate_stop = False
            if gate_pol is not None:
                #: ★ §6 C6-4：门用 `hidden=None` 逐帧口径（与四臂留出读数同口径）
                #: ⚠️ `_gate_p_stop` 返回 **0-dim tensor**（与 `first_option_probs` 同），不用再 `[0]`
                #: ★ W2/W3：门的 `plan_dim` 可能与主策略不同（如门是 58 维旧臂）⇒ 各建各的向量
                _gplan = (plan if int(gate_meta.get("plan_dim", 0)) == int(meta["plan_dim"])
                          else build_plan_vec(bp, env, belief0, obs, len(_known_opp),
                                              gate_meta["plan_dim"]))
                _p_stop = float(_gate_p_stop(gate_pol, obs, tok, _gplan,
                                             [env.get_action_mask()]))
                gate_eval += 1
                if _p_stop > a.gate_threshold:
                    _gate_stop = True
                    gate_stop += 1
            if _gate_stop:
                bundle, h_out = ActionBundle(), hidden
            else:
                bundle, _lp, _v, h_out, _mk = pol.act(obs, tok, plan, env.get_action_mask,
                                                      hidden=h_in, deterministic=not a.sample)
            hidden = h_out if a.hidden == "carry" else None
            #: ★ 行为画像的关键分层（与 `scripts/probe_pass_prob.py` 同一口径）：
            #: 「本帧买得起吗」用**掩码**判（`slots` 任一为真 或 `ability_legal`），
            #: 分三类：①被迫不出（唯一合法项 = STOP）②买得起却选择不出 ③真出牌
            _m0 = env.get_action_mask()
            _can = bool(np.any(_m0["slots"])) or bool(_m0.get("ability_legal"))
            playable += int(_can)
            forced_stop += int(not _can)
            cards = _bundle_cards(bundle, obs)
            #: ★ W3 前期窗口（J-W3.2 仪器）：**决策帧当时**的 t 与对手已知卡数
            _t_now = float(env.battle.time)
            if _t_now <= INFO_T_WINDOW:
                win_frames += 1
                win_known.append(len(_known_opp))
                if cards:
                    _win_play_n += len(cards)
                    _win_cards.update(cards)
                    if _first_t is None:
                        _first_t = _t_now
            if _can and bundle.size == 0:
                stop_when_playable += 1
            obs, reward, term, trunc, info = env.step(bundle)
            rec.record(env, bundle, reward, info, cards=cards)
            #: 出牌统计（p0 用 `_bundle_cards`；对手用 env 交回的 `opp_played`）
            for nm in cards:
                card_hist[nm] += 1
            for nm in (info.get("opp_played") or []):
                #: 实测 `opp_played` 的元素是 **dict**（不是卡名字符串）⇒ 取名字段再计数
                nm = _opp_play_name(nm)
                if isinstance(nm, str):
                    opp_card_hist[nm] += 1
                    _known_opp.add(nm)      #: ★ W2/W3 信息分：不同卡数（集合自动去重）
            belief0.update(obs, info.get("opp_played"))
            frames_total += 1
            brew += float(reward)
            rew_total += float(reward)
            bf += 1
            if bundle.size == 0:
                empty_total += 1
            plays_total += len(cards)
            done = term or trunc
            steps += 1
        w = env.battle.winner
        if w is None and not env.battle.game_over:
            w = timeout_winner(env.battle)
        games.append(rec.done(w))
        win_plays += float(_win_play_n)
        win_distinct += float(len(_win_cards))
        if _first_t is not None:
            win_first_t.append(float(_first_t))
        cw += int(w == 0)
        cl += int(w == 1)
        cd += int(w is None)
        bw += int(w == 0)
        bl += int(w == 1)
        bd += int(w is None)
        bg += 1

        if bg >= a.block or g == a.games - 1:
            n = cw + cl + cd
            hist.append({
                "step": int(frames_total),
                "wins": bw, "losses": bl, "draws": bd, "games": bg,
                "cum_wins": cw, "cum_losses": cl, "cum_draws": cd, "cum_games": n,
                "winrate": cw / max(1, n),
                "winrate_se": float(np.sqrt(max(1e-9, (cw / max(1, n)) * (1 - cw / max(1, n)))
                                             / max(1, n))),
                "mean_reward": rew_total / max(1, frames_total),
                "block_mean_reward": brew / max(1, bf),
                "empty_bundle_rate": empty_total / max(1, frames_total),
                "plays_per_frame": plays_total / max(1, frames_total),
                "playable_frame_rate": playable / max(1, frames_total),
                "stop_when_playable_rate": stop_when_playable / max(1, playable),
            })
            save_league_replays(games, os.path.join(out, "replays",
                                                    f"league_{int(frames_total)}.pkl"))
            with open(os.path.join(out, "solo_state.json"), "w", encoding="utf-8") as fh:
                json.dump({
                    "mode": "solo",
                    "agents": [{"agent_id": "il", "kind": "main", "path": a.ckpt}],
                    "history": hist,
                    "total_steps": int(frames_total),
                    "target_steps": int(frames_total),
                    "deck": list(d0),
                    "opponent": ("self-frozen-copy(il)" if a.opponent == "self"
                                 else f"main:{a.main_ckpt}"),
                    "copy_every": 0,
                    "status": "running" if g < a.games - 1 else "done",
                    "demo": False,
                    "_runner": {"kind": "il_readout", "ckpt": a.ckpt, "decks": a.decks,
                                "opponent": a.opponent, "hidden": a.hidden,
                                "games_planned": a.games, "block": a.block, "seed": a.seed},
                }, fh, ensure_ascii=False, indent=2)
            el = time.time() - t0
            print(f"  [块 {len(hist)}] {n}/{a.games} 局、{frames_total} 帧 | "
                  f"胜 {cw} 负 {cl} 平 {cd} = {100*cw/max(1,n):.1f}% | "
                  f"局均帧 {frames_total/max(1,n):.1f} | 空 bundle "
                  f"{100*empty_total/max(1,frames_total):.1f}% | 买得起帧 "
                  f"{playable}/{frames_total}、其中选择不出 "
                  f"{100*stop_when_playable/max(1,playable):.1f}% | {el:.0f}s", flush=True)
            bf = bg = bw = bl = bd = 0
            brew = 0.0

    n = cw + cl + cd
    summary = {
        "ckpt": a.ckpt, "decks": a.decks, "opponent": a.opponent, "hidden": a.hidden,
        "opp_hidden": opp_hidden,
        "games": n, "frames": frames_total, "wins": cw, "losses": cl, "draws": cd,
        "winrate": cw / max(1, n),
        "winrate_se": float(np.sqrt(max(1e-9, (cw / max(1, n)) * (1 - cw / max(1, n)))
                                    / max(1, n))),
        "mean_reward_per_frame": rew_total / max(1, frames_total),
        "frames_per_game": frames_total / max(1, n),
        "empty_bundle_rate": empty_total / max(1, frames_total),
        "plays_per_frame": plays_total / max(1, frames_total),
        "playable_frame_rate": playable / max(1, frames_total),
        "playable_frames": playable,
        "forced_stop_frames": forced_stop,
        "stop_when_playable": stop_when_playable,
        "stop_when_playable_rate": stop_when_playable / max(1, playable),
        #: ★ §6 解耦组合臂（不给 --gate-ckpt 时全为 None/0，原行为不变）
        "gate_ckpt": a.gate_ckpt,
        "gate_threshold": (a.gate_threshold if gate_pol is not None else None),
        "gate_eval_frames": gate_eval,
        "gate_stop_frames": gate_stop,
        "gate_stop_rate": (gate_stop / gate_eval) if gate_eval else None,
        "p0_plays": plays_total,
        "p0_top_cards": card_hist.most_common(20),
        "p1_top_cards": opp_card_hist.most_common(20),
        #: ★ W3（预注册 `docs/il_whiff_handscore_prereg_2026-09-22.md` §3.3 J-W3.2）：
        #: 前期窗口 T=30 s 内的**对手已知卡数**与出牌节奏 —— 「试探」的**方向判据**仪器。
        #: 只做描述性读数（R15：不设绝对阈值，判定用**臂间配对方向**）。
        "info_window": {
            "t_info": INFO_T_WINDOW,
            "frames_in_window": win_frames,
            "known_opp_median_in_window": _median(win_known),
            "known_opp_mean_in_window": (float(np.mean(win_known)) if win_known else None),
            "known_opp_max_in_window": (int(max(win_known)) if win_known else None),
            "plays_in_window_per_game": win_plays / max(1, n),
            "distinct_cards_in_window_per_game": win_distinct / max(1, n),
            "first_play_t_median": _median(win_first_t),
        },
        "seconds": round(time.time() - t0, 1),
        "history": hist,
    }
    jp = a.json or os.path.join(out, "stats.json")
    with open(jp, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(f"[runner] 完成 {n} 局 / {frames_total} 帧 / {summary['seconds']}s | "
          f"胜率 {100*summary['winrate']:.1f}%±{100*summary['winrate_se']:.1f} | "
          f"空 bundle {100*summary['empty_bundle_rate']:.1f}% | "
          f"每帧出牌 {summary['plays_per_frame']:.2f}", flush=True)
    print(f"[runner] p0 出牌 {plays_total} 次 / 买得起 {playable} 帧（被迫不出 {forced_stop} 帧）"
          f"｜买得起却选择不出 {stop_when_playable} = "
          f"{100*stop_when_playable/max(1,playable):.1f}%", flush=True)
    print(f"[runner] p0 出牌 top: {card_hist.most_common(6)}", flush=True)
    print(f"[runner] 落盘：{out}/replays/league_*.pkl、{out}/solo_state.json、{jp}", flush=True)


if __name__ == "__main__":
    main()
