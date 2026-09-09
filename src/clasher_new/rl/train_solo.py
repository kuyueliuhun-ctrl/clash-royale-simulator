"""单人自对弈训练（原版 ``train.py`` 思路的现代版；**无联赛机制**）。

- 单模型 main（``FollowerPolicy`` + ``PPOTrainer``），双方使用**同一副固定卡组**
  （默认 8 卡，镜像对局；``DEFAULT_SOLO_DECK``）；
- 对手 = main 的**周期冻结副本**：每 ``cfg.solo_copy_every`` 步把 main 权重拷给
  opponent（原版 ``train.py`` ``WeightsCopyingCallback`` 的现代版），对手 deterministic；
- 无联赛：不建 ``League``、不写 Elo/PFSP/``league_state.json``；
- 周期评估（``cfg.steps_per_eval``）写 ``solo_state.json``
  （winrate±SE / mean_reward / 进度字段），供 dashboard ``--solo`` 实时显示；
- 评估回放存 ``replays/league_<step>.pkl``（复用 dashboard 回放列表/播放器）。

用法（run_league 入口）：
    python rl/run_league.py --mode solo --config economy --device cuda
"""

import os
import sys
import json
import math
import time
import random

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import numpy as np
import torch

from rl.env_wrapper import RLEnv
from card_utils import Card
from rl.belief import BeliefInference
from rl.belief_planner import BeliefPlanner
from rl.prophet import ProphetPlanner
from rl.plan_space import PLAN_DIM
from rl.follower import FollowerPolicy, save_checkpoint, load_checkpoint
from rl.ppo import PPOTrainer
from rl.config import reward_to_env
from rl.train_follower import FollowerOpponent
from rl.pfsp import PFSP as _PFSP
from rl.run_league import (resolve_device, _bundle_cards, LeagueGameRecorder,
                           _stall_probe, STALL_WINDOW, _load_run_state,
                           timeout_winner, overtime_open)
from rl.replay import save_league_replays

#: 固定卡组（连弩 2.9，2026-09-09 用户切换）：双方镜像使用同一副。
#: 原版 8 卡（Knight/MiniPekka/...）"几乎没有实战价值"——Xbow 核心的自闭阵地
#: archetype 才有真实战术结构（阵地/法术解场/循环防守）。与 FOUR_DECK_SET 的
#: X弩同族（IceWizard 位置相同），docs/four_decks_manual.md。
DEFAULT_SOLO_DECK = ["Xbow", "Tesla", "Skeletons", "IceWizard",
                     "Archer", "Knight", "Log", "Fireball"]


def resolve_deck_set(deck_set: str):
    """cfg.deck_set → (mirror_deck, deck_pool_for_defender)。

    - "default"：原版 8 卡镜像，defend 对手无 deck_pool（旧行为）；
    - "four"：原版 8 卡镜像不变，defend 对手每局从 FOUR_DECK_SET 抽一副
      （docs/four_decks_manual.md：速猪/皇家巨人/X弩/双线快攻）；
    - "list:Card1,..."：显式 8 卡镜像（逗号分隔引擎卡名），defend 无 deck_pool。
    """
    from rl.opponents import FOUR_DECK_SET
    s = (deck_set or "default").strip()
    if s == "four":
        return list(DEFAULT_SOLO_DECK), FOUR_DECK_SET
    if s.startswith("list:"):
        cards = [c.strip() for c in s[5:].split(",") if c.strip()]
        if len(cards) != 8:
            raise ValueError(f"--deck-set list: 需要 8 张卡，收到 {len(cards)}: {cards}")
        for c in cards:
            Card(c)   # 不可部署卡直接抛错（快速失败）
        return cards, None
    return list(DEFAULT_SOLO_DECK), None

#: 训练中先知规划注入概率（与 run_league 主训练一致）
_SOLO_PROPHET_PROB = 0.3

# —— A 层：训练对手池（2026-09-08，"单边堆牌"根因修复）——
# 取证（scripts/forensics_response.py v2）：对手是 frozen_copy 自我对冲 → 两边都不
# 防守时"换家"是合法策略，模型学不到对牌。对手池三类混合：
#   frozen（主力，1-p_pool 权重内默认）  main 周期冻结副本（原行为）；
#   hist    历史 checkpoint PFSP 采样（自博弈多样性，会惩罚过时策略的漏洞）；
#   defend  真防守脚本（SelfDefenderPolicy：script_defender 反制 + 低频缓出）——
#           单边推进在它面前讨不到便宜 → 单边策略直接亏塔损奖励。
#: 训练局对手构成：frozen 0.7 / hist 0.2 / defend 0.1（frozen 仍是主力避免 curriculum 断裂）
_OPP_MIX = {"frozen": 0.7, "hist": 0.2, "defend": 0.1}
#: hist 采样池：从磁盘 checkpoint 目录收集 solo_main_<step>.pt（最多保留 12 个，
#: 按步数均匀抽样——几百个文件全加载内存吃不消）
_HIST_POOL_MAX = 12


def solo_env(cfg, seed, deck0=None, deck1=None):
    """双方固定卡组的镜像 RLEnv（deck_set=list:/default 时同副镜像）。"""
    return RLEnv(opponent=None, seed=seed, reward_weights=reward_to_env(cfg),
                 card_level=cfg.card_level,
                 deck0=deck0 or DEFAULT_SOLO_DECK, deck1=deck1 or DEFAULT_SOLO_DECK)


def _draw_penalty(cfg) -> float:
    """平局惩罚（= 失败：平局不再免费）。缺省与 lose_penalty 相同。"""
    rw = reward_to_env(cfg)
    return float(rw.get("draw_penalty", rw.get("lose_penalty", 10.0)))


def _sync_frozen_copy(main, opp):
    """把 main 当前权重同步给冻结副本（周期执行）。"""
    opp.load_state_dict(main.state_dict())


def _collect_hist_ckpts(folder, max_n=_HIST_POOL_MAX):
    """收集 solo 输出目录的历史 checkpoint（solo_main_<step>.pt）。

    按 step 升序均匀抽 max_n 个（含最旧不含当前正在写的 solo_main.pt）。
    folder 不存在/无文件 → 空列表（对手池退化为 frozen+defend 两类）。"""
    if not folder or not os.path.isdir(folder):
        return []
    steps = []
    for fn in os.listdir(folder):
        if fn.startswith("solo_main_") and fn.endswith(".pt"):
            try:
                steps.append(int(fn[len("solo_main_"):-3]))
            except ValueError:
                continue
    steps.sort()
    if len(steps) <= max_n:
        return [os.path.join(folder, f"solo_main_{s}.pt") for s in steps]
    idx = np.linspace(0, len(steps) - 1, max_n).astype(int)
    return [os.path.join(folder, f"solo_main_{steps[i]}.pt") for i in idx]


class _OpponentPool:
    """训练对手选择器（9j）：frozen / hist / defend 三类按 _OPP_MIX 概率采样。

    - frozen：返回主 frozen_copy（FollowerOpponent，权重周期同步）——原行为；
    - hist：从历史 checkpoint 池 PFSP 采样一个，载入专用 hist 策略 → 包装
      FollowerOpponent（belief/planner 完整链路）；PFSP 权重按 main 对各 hist
      ckpt 的近期胜率（低胜率高权重，pfsp.PFSP 语义）；
    - defend：SelfDefenderPolicy（真防守脚本）。
    每局开始由外部调 sample()，返回的对手直接赋给 env.opponent。
    """

    def __init__(self, cfg, env, frozen_side, rng, device, defender_deck_pool=None):
        from rl.opponents import SelfDefenderPolicy
        self.cfg = cfg
        self.env = env
        self.frozen_side = frozen_side       # FollowerOpponent（冻结副本）
        self.rng = rng
        self.device = device
        # defend 对手卡组：deck_set=four 时从四卡组抽一副（docs/four_decks_manual.md）
        #（script_defender 的反制逻辑按卡牌语义工作，四套 archetype 各逼出不同防守模式）。
        self.defender = SelfDefenderPolicy(seed=cfg.seed + 7, env=env,
                                           deck_pool=defender_deck_pool)
        self.hist_paths = _collect_hist_ckpts(cfg.folder())
        self._pfsp = _PFSP(beta=1.0, seed=cfg.seed + 11)
        self._hist_id = {p: f"hist_{i}" for i, p in enumerate(self.hist_paths)}
        self._hist_policy = None             # 惰性建（需要 belief_dim/hidden_dim）
        self._hist_side = None
        self._loaded_path = None
        self._last_kind = None
        self._last_hist_id = None
        if self.hist_paths:
            print(f"[solo] 对手池: hist ckpts={len(self.hist_paths)} "
                  f"(mix frozen={_OPP_MIX['frozen']}/hist={_OPP_MIX['hist']}/"
                  f"defend={_OPP_MIX['defend']})", flush=True)
        else:
            print(f"[solo] 对手池: 无历史 ckpt（本目录首轮训练），退化为 "
                  f"frozen={_OPP_MIX['frozen']/(1-_OPP_MIX['hist'])} "
                  f"/ defend={_OPP_MIX['defend']/(1-_OPP_MIX['hist'])}", flush=True)

    def sample(self):
        """为本局选对手：返回 (kind, opponent, hist_id_or_None)。"""
        r = self.rng.random()
        if self.hist_paths and r < _OPP_MIX["hist"]:
            opp_id = self._pfsp.sample("main", list(self.hist_paths))
            self._ensure_hist(opp_id)
            self._last_kind, self._last_hist_id = "hist", self._hist_id[opp_id]
            return "hist", self._hist_side, self._last_hist_id
        if r < _OPP_MIX["hist"] + _OPP_MIX["defend"] or \
                (not self.hist_paths and r >= _OPP_MIX["frozen"] / (1 - _OPP_MIX["hist"])):
            self._last_kind, self._last_hist_id = "defend", None
            return "defend", self.defender, None
        self._last_kind, self._last_hist_id = "frozen", None
        return "frozen", self.frozen_side, None

    def record(self, winner):
        """上一局结束回填 PFSP 胜率（frozen/defend 局无操作）。

        winner: 0=main 胜（score 1.0）/ 1=main 负（0.0）/ None=平局（0.5）。"""
        if self._last_kind == "hist" and self._last_hist_id is not None:
            score = {0: 1.0, 1: 0.0}.get(winner, 0.5)
            self._pfsp.update_winrate("main", self._last_hist_id, score)

    def _ensure_hist(self, path):
        """载入 hist ckpt（换目标才重载；belief_dim 尾部零拷贝兼容旧 23 维）。"""
        if self._hist_policy is None:
            self._hist_policy = FollowerPolicy(
                hidden=self.cfg.hidden_dim, plan_dim=PLAN_DIM,
                belief_dim=len(BeliefInference(opp_deck=self.env.deck1,
                                               n_particles=128, seed=0).encode(None, None)))
            self._hist_policy.to_device(self.device)
        if self._loaded_path != path:
            ck = load_checkpoint(path, plan_dim=PLAN_DIM,
                                 belief_dim=self._hist_policy.belief_dim)
            self._hist_policy.load_state_dict(ck.state_dict())
            self._loaded_path = path
        if self._hist_side is None:
            self._hist_side = FollowerOpponent(
                self._hist_policy, self.env,
                belief=BeliefInference(opp_deck=self.env.deck1, n_particles=128,
                                       seed=self.cfg.seed + 3),
                deterministic=True)
        else:
            self._hist_side.policy = self._hist_policy
        self._hist_side.reset()


def write_solo_state(path, cfg, history, step, status="running",
                     deck=None, copy_every=None, target_steps=None, controls=None):
    """把 solo 训练状态落盘（增量写；dashboard --solo 实时读取）。

    controls: 本周期对照组结果 [{"step","vs","wins",...}]（vs 初始模型/上一评估点模型，
    只做对比不进迭代）；按周期累积在 state["controls_history"]。
    """
    state = {
        "mode": "solo",
        "agents": [{"agent_id": "main", "kind": "main", "path": None}],
        "history": history,             # [{step,wins,losses,draws,games,winrate,winrate_se,mean_reward}]
        "total_steps": int(step),
        "target_steps": int(target_steps if target_steps is not None else cfg.total_steps),
        "deck": deck if deck is not None else list(DEFAULT_SOLO_DECK),
        "opponent": "self-play-frozen-copy",
        "copy_every": int(copy_every if copy_every is not None else cfg.solo_copy_every),
        "status": status,
        "demo": False,
    }
    if controls is not None:
        ch = state.setdefault("_controls_history", [])
        prev = {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                prev = json.load(f) or {}
        except (OSError, ValueError):
            pass
        ch = [c for c in (prev.get("_controls_history") or [])
              if c.get("step") != int(step)]
        ch.extend(controls)
        state["_controls_history"] = ch
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    return state


def eval_solo(env, main, opp, n_games, max_steps, seed, cfg,
              record_replays=False, replays_dir=None, step=None, frozen_step=None,
              save_replays=True):
    """main（deterministic）vs 冻结副本（deterministic）打 n_games。

    返回 (stats, replays)。replays 非空且 save_replays 时以 league_<step>.pkl 落盘
    （复用 dashboard 回放；对照组评估传 save_replays=False 只留数字不落盘）。
    step: main 当前训练步（录像文件步数）；frozen_step: 冻结副本最后同步时的训练步，
    两者一并写入每局 meta.steps，dashboard 对阵列显示 "main@<step> vs main@<frozen_step>"。
    """
    bp = BeliefPlanner()
    wins = losses = draws = 0
    rew_sum = 0.0
    replays = []
    for g in range(n_games):
        opp_side = FollowerOpponent(
            opp, env,
            belief=BeliefInference(opp_deck=env.deck1, n_particles=128, seed=seed + g),
            deterministic=True)
        env.opponent = opp_side
        belief = BeliefInference(opp_deck=env.deck1, n_particles=128, seed=seed + 1000 + g)
        obs, _ = env.reset(seed=seed + 2000 + g)
        belief.reset(env.deck1)
        hidden = None
        rec = LeagueGameRecorder("main", "frozen_copy", "main", max_steps,
                                 steps=(step, frozen_step)) if record_replays else None
        done = False
        steps = 0
        ep_rew = 0.0
        stall_count = 0
        last_hp = None
        while not done and (steps < max_steps or overtime_open(env.battle)):
            if steps % STALL_WINDOW == 0:
                early, last_hp, stall_count = _stall_probe(env, last_hp, stall_count)
                if early:
                    break   # 僵局判平，提前结束
            plan = bp.plan(env.battle, belief.state(), obs)
            tok = belief.encode(obs, None)
            bundle, _, _, hidden, _ = main.act(
                obs, tok, plan.to_vector(), env.get_action_mask,
                hidden=hidden, deterministic=True)
            played = _bundle_cards(bundle, obs)
            obs, reward, term, trunc, info = env.step(bundle)
            ep_rew += float(reward)
            if rec is not None:
                rec.record(env, bundle, reward, info)
            opp_side.observe_opponent_played(played)
            belief.update(obs, info.get("opp_played"))
            done = term or trunc
            steps += 1
        w = env.battle.winner
        if w is None and not env.battle.game_over:
            # 僵局早停/截断早于引擎结算 → 皇冠差已定胜负；皇冠相同（加时未破塔）→ 平局=失败
            virt = timeout_winner(env.battle)
            if virt is None:
                ep_rew -= _draw_penalty(cfg)
            else:
                w = virt
                rw = reward_to_env(cfg)
                ep_rew += float(rw["win_bonus"] if w == 0 else -rw["lose_penalty"])
        if w == 0:
            wins += 1
        elif w == 1:
            losses += 1
        else:
            draws += 1
        rew_sum += ep_rew
        if rec is not None:
            replays.append(rec.done(w))
    n = max(1, n_games)
    winrate = (wins + 0.5 * draws) / n
    se = math.sqrt(max(0.0, winrate * (1.0 - winrate)) / n) if n > 1 else 0.5
    stats = {"step": step, "wins": wins, "losses": losses, "draws": draws,
             "games": n, "winrate": round(winrate, 4),
             "winrate_se": round(se, 4), "mean_reward": round(rew_sum / n, 4)}
    if replays and replays_dir and step is not None and save_replays:
        os.makedirs(replays_dir, exist_ok=True)
        save_league_replays(replays, os.path.join(replays_dir, f"league_{step}.pkl"))
    return stats, replays


def _eval_worker_main(worker_id, main_sd, opp_sd, games, env_kwargs,
                      seed_base, max_steps, n_particles, record, out_q):
    """并行评估 worker：独立进程打 games（全局游戏索引列表）里每局。

    战斗模拟是纯 Python（GIL），跨进程才能真正吃满多核。每 worker 自建 env+信念+策略
    （从主进程收 state_dict），逐局回传 (game_idx, winner, ep_rew, replay_or_None)。
    与串行 eval_solo 同种子等价：串行复用单个 env，reset() 会基于上一局牌序继续
    shuffle（belief 先验 = 上一局结束后的牌序），所以 worker 必须把 0..n_games-1 的
    reset 链全部走一遍（reset 本身极便宜），只打分配到的局，才能还原同一信念先验。
    """
    try:
        from rl.env_wrapper import RLEnv
        from rl.belief import BeliefInference
        from rl.belief_planner import BeliefPlanner
        from rl.follower import FollowerPolicy
        from rl.plan_space import PLAN_DIM
        from rl.train_follower import FollowerOpponent
        from rl.run_league import LeagueGameRecorder, _stall_probe, STALL_WINDOW, _bundle_cards

        # 16 进程 × 默认 16 线程 = 256 线程在 16 核上互相争抢（过订阅），
        # 战斗模拟是纯 Python 单线程、推理 batch 极小 → 每 worker 1 线程即可
        import torch as _torch
        _torch.set_num_threads(1)
        try:
            import numpy as _np
            _np.set_num_threads(1)
        except Exception:
            pass

        env = RLEnv(opponent=None, seed=worker_id + 777,
                    reward_weights=dict(env_kwargs.get("reward_weights") or {}),
                    card_level=env_kwargs.get("card_level"),
                    deck0=list(env_kwargs["deck0"]), deck1=list(env_kwargs["deck1"]))
        belief_dim = len(BeliefInference(opp_deck=env.deck1, n_particles=n_particles,
                                         seed=0).encode(None, None))
        main = FollowerPolicy(hidden=env_kwargs["hidden_dim"], plan_dim=PLAN_DIM,
                              belief_dim=belief_dim)
        opp = FollowerPolicy(hidden=env_kwargs["hidden_dim"], plan_dim=PLAN_DIM,
                             belief_dim=belief_dim)
        main.load_state_dict(main_sd)
        opp.load_state_dict(opp_sd)
        main.to_device("cpu")
        opp.to_device("cpu")
        bp = BeliefPlanner()
        n_total = int(env_kwargs["n_total"])
        do = set(int(g) for g in games)
        results = []
        for g in range(n_total):
            # 与串行 eval_solo 同序：先按当前 env.deck1（上一局牌序）构造信念，再 reset 换牌序
            opp_side = FollowerOpponent(
                opp, env,
                belief=BeliefInference(opp_deck=env.deck1, n_particles=n_particles,
                                       seed=seed_base + g),
                deterministic=True)
            belief = BeliefInference(opp_deck=env.deck1, n_particles=n_particles,
                                     seed=seed_base + 1000 + g)
            obs, _ = env.reset(seed=seed_base + 2000 + g)
            belief.reset(env.deck1)
            if g not in do:
                continue
            env.opponent = opp_side
            hidden = None
            rec = LeagueGameRecorder("main", "frozen_copy", "main", max_steps,
                                     steps=(env_kwargs.get("eval_step"),
                                            env_kwargs.get("frozen_step"))) if record else None
            done = False
            steps = 0
            ep_rew = 0.0
            stall_count = 0
            last_hp = None
            while not done and (steps < max_steps or overtime_open(env.battle)):
                if steps % STALL_WINDOW == 0:
                    early, last_hp, stall_count = _stall_probe(env, last_hp, stall_count)
                    if early:
                        break
                plan = bp.plan(env.battle, belief.state(), obs)
                tok = belief.encode(obs, None)
                bundle, _, _, hidden, _ = main.act(
                    obs, tok, plan.to_vector(), env.get_action_mask,
                    hidden=hidden, deterministic=True)
                played = _bundle_cards(bundle, obs)
                obs, reward, term, trunc, info = env.step(bundle)
                ep_rew += float(reward)
                if rec is not None:
                    rec.record(env, bundle, reward, info)
                opp_side.observe_opponent_played(played)
                belief.update(obs, info.get("opp_played"))
                done = term or trunc
                steps += 1
            w = env.battle.winner
            if w is None and not env.battle.game_over:
                # 与串行 eval_solo 一致：早停/截断补结算（皇冠差→胜负，皇冠相同=加时未破塔→平局=失败）
                virt = timeout_winner(env.battle)
                rw = env_kwargs.get("reward_weights") or {}
                if virt is None:
                    ep_rew -= float(rw.get("draw_penalty", rw.get("lose_penalty", 10.0)))
                else:
                    w = virt
                    ep_rew += float(rw["win_bonus"] if w == 0 else -rw["lose_penalty"])
            results.append((g, w, ep_rew, rec.done(w) if rec is not None else None))
        out_q.put(("result", results))
    except Exception as e:
        import traceback
        try:
            out_q.put(("error", "%r\n%s" % (e, traceback.format_exc())))
        except Exception:
            pass


def _collect_worker_results(procs, out_q, expected):
    """收齐 expected 份 worker 消息，返回 (results, failure_or_None)。

    worker 静默死亡（启动即崩，如 WinError 1455 页面文件不足导致 import torch
    失败）不会往队列里放任何消息——原实现 out_q.get() 会永久阻塞；这里用
    超时 + 存活探针兜底。
    """
    import queue as _queue
    results = []
    got = 0
    while got < expected:
        try:
            msg = out_q.get(timeout=20)
        except _queue.Empty:
            if not any(p.is_alive() for p in procs):
                return results, (f"{expected - got} 个 worker 退出但未回传结果"
                                 "（启动即崩溃，典型原因：页面文件不足 WinError 1455）")
            continue
        if msg[0] == "error":
            return results, msg[1]
        results.extend(msg[1])
        got += 1
    return results, None


def eval_solo_parallel(env, main, opp, n_games, max_steps, seed, cfg,
                       n_workers=8, record_replays=False, replays_dir=None, step=None,
                       frozen_step=None, save_replays=True):
    """eval_solo 的进程池并行版：n_games 局均分到 n_workers 个 spawn 进程打。

    串行 16 局≈126s（主进程单核跑纯 Python 模拟）；16 进程理论 ≈ 126/16 + spawn/import
    开销 ≈ 15-25s。统计与串行同公式（wins/losses/draws + mean_reward + SE）。

    Windows 上每个 spawn worker import torch 都要加载 CUDA DLL（约 2GB 提交内存）：
    任一 worker 启动失败/崩溃时整体降级串行 eval_solo（同种子结果等价），
    评估乃至整个训练不再因此崩掉。
    """
    import multiprocessing as mp
    n_games = int(n_games)
    n_workers = min(int(n_workers), n_games)
    if n_workers < 2:
        print("[eval] 并行评估不可用，降级串行（同种子结果等价，只是慢）", flush=True)
        return eval_solo(env, main, opp, n_games, max_steps, seed, cfg,
                         record_replays=record_replays, replays_dir=replays_dir,
                         step=step, frozen_step=frozen_step, save_replays=save_replays)
    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    main_sd = {k: v.detach().cpu() for k, v in main.state_dict().items()}
    opp_sd = {k: v.detach().cpu() for k, v in opp.state_dict().items()}
    games = list(range(n_games))
    chunks = [games[i::n_workers] for i in range(n_workers)]
    env_kwargs = {"reward_weights": reward_to_env(cfg), "card_level": cfg.card_level,
                  "deck0": list(env.deck0), "deck1": list(env.deck1),
                  "hidden_dim": int(cfg.hidden_dim), "n_total": n_games,
                  "eval_step": step, "frozen_step": frozen_step}
    procs = []
    # worker 是纯 CPU 推理：启动前屏蔽 CUDA 省掉子进程的 CUDA 初始化。父进程不受
    # 影响（torch 已初始化），环境变量在全部 worker 结束后才恢复。
    _prev_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        for wid, chunk in enumerate(chunks):
            if not chunk:
                continue
            p = ctx.Process(target=_eval_worker_main,
                            args=(wid, main_sd, opp_sd, chunk, env_kwargs,
                                  int(seed), int(max_steps), 128, bool(record_replays), out_q))
            try:
                p.start()
            except OSError as e:
                failure = f"启动 worker{wid} 失败: {e!r}"
                break
            procs.append(p)
        else:
            results, failure = _collect_worker_results(procs, out_q, len(procs))
        if failure is not None:
            for p in procs:
                if p.is_alive():
                    p.terminate()
            print(f"[eval] 并行评估 worker 失败，降级串行: {failure}", flush=True)
            return eval_solo(env, main, opp, n_games, max_steps, seed, cfg,
                             record_replays=record_replays, replays_dir=replays_dir,
                             step=step, frozen_step=frozen_step, save_replays=save_replays)
        for p in procs:
            p.join(timeout=10)
        results.sort(key=lambda r: r[0])   # 按游戏索引还原顺序
    finally:
        if _prev_cvd is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = _prev_cvd
    wins = losses = draws = 0
    rew_sum = 0.0
    replays = []
    for _g, w, ep_rew, rec in results:
        if w == 0:
            wins += 1
        elif w == 1:
            losses += 1
        else:
            draws += 1
        rew_sum += ep_rew
        if rec is not None:
            replays.append(rec)
    n = max(1, int(n_games))
    winrate = (wins + 0.5 * draws) / n
    se = math.sqrt(max(0.0, winrate * (1.0 - winrate)) / n) if n > 1 else 0.5
    stats = {"step": step, "wins": wins, "losses": losses, "draws": draws,
             "games": n, "winrate": round(winrate, 4),
             "winrate_se": round(se, 4), "mean_reward": round(rew_sum / n, 4)}
    if replays and replays_dir and step is not None and save_replays:
        os.makedirs(replays_dir, exist_ok=True)
        save_league_replays(replays, os.path.join(replays_dir, f"league_{step}.pkl"))
    return stats, replays


def run_solo(cfg, resume=False, record_replays=True):
    """单人自对弈主循环（无联赛；写 solo_state.json + solo_main.pt）。

    流程：镜像固定卡组 → main 训练 vs 冻结副本（每 solo_copy_every 步同步）→
    每 steps_per_eval 评估并写 solo_state.json（dashboard 实时显示）。
    """
    _t_start = time.monotonic()
    device = resolve_device(cfg.device)   # 首次调用即触发 torch CUDA 上下文初始化
    _t_cuda = time.monotonic()
    cfg.ensure_dirs()
    cfg.save()
    print(f"[solo] 单人自对弈 配置 '{cfg.name}' -> {cfg.folder()} "
          f"(device={device}, seed={cfg.seed}, 固定卡组 {len(DEFAULT_SOLO_DECK)} 卡镜像, "
          f"冻结副本同步间隔={cfg.solo_copy_every})", flush=True)
    _t_cfg = time.monotonic()
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # deck_set 解析：镜像卡组 + defend 对手卡组池（four=四卡组对手池）
    mirror_deck, defender_deck_pool = resolve_deck_set(getattr(cfg, "deck_set", None)
                                                       or "default")
    env = solo_env(cfg, cfg.seed, deck0=mirror_deck, deck1=mirror_deck)
    _t_env = time.monotonic()
    belief_dim = len(BeliefInference(opp_deck=env.deck1, n_particles=128,
                                     seed=0).encode(None, None))
    # —— 断点续练（--resume）：恢复 step / main 权重 / 优化器 / 历史曲线 ——
    start_step = 0
    history = []
    rs = None
    if resume:
        rs = _load_run_state(cfg)
        if rs and rs.get("solo_ckpt") and os.path.exists(rs["solo_ckpt"]):
            start_step = int(rs.get("step", 0))
            if os.path.exists(cfg.solo_state_path()):
                try:
                    with open(cfg.solo_state_path(), "r", encoding="utf-8") as f:
                        history = (json.load(f) or {}).get("history", [])
                except (OSError, ValueError):
                    history = []
        else:
            print("[solo] resume 检查点缺失，从头开始", flush=True)

    if cfg.main_init and not (rs and rs.get("solo_ckpt")
                              and os.path.exists(rs["solo_ckpt"])):
        # 显式 plan_dim/belief_dim=当前网络维度 → load_checkpoint 走"前列拷贝+尾零"
        # 兼容分支（旧 ckpt 的 plan_dim=57/belief_dim=23 元数据否则把 main 建成旧维度，
        # 之后 _sync_frozen_copy 拷进 PLAN_DIM 网络即 shape 失配崩溃）
        main = load_checkpoint(cfg.main_init, hidden_dim=cfg.hidden_dim,
                               plan_dim=PLAN_DIM, belief_dim=belief_dim)
    else:
        main = FollowerPolicy(hidden=cfg.hidden_dim, plan_dim=PLAN_DIM,
                              belief_dim=belief_dim)
    main.to_device(device)
    if rs and rs.get("solo_ckpt") and os.path.exists(rs["solo_ckpt"]):
        # resume：断点权重为准（覆盖 main_init）
        main = load_checkpoint(rs["solo_ckpt"], hidden_dim=cfg.hidden_dim)
        main.to_device(device)
        print(f"[solo] resume 从 step {start_step} 续训（继续到 {cfg.total_steps}）", flush=True)
    opp = FollowerPolicy(hidden=cfg.hidden_dim, plan_dim=PLAN_DIM, belief_dim=belief_dim)
    opp.to_device(device)
    _sync_frozen_copy(main, opp)   # 开局副本 = main（resume 后即断点权重）
    frozen_step = start_step       # 冻结副本当前所在训练步（录像 meta.steps 用）
    ppo = PPOTrainer(main, lr=cfg.lr, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda,
                     clip=cfg.clip, vf_coef=cfg.vf_coef, ent_coef=cfg.ent_coef,
                     max_grad_norm=cfg.max_grad_norm, adv_norm=cfg.adv_norm)
    if rs and rs.get("solo_opt") and os.path.exists(rs["solo_opt"]):
        try:
            ppo.opt.load_state_dict(torch.load(rs["solo_opt"], map_location=device))
        except Exception:
            print("[solo] resume 优化器状态不匹配，Adam 从头（模型仍续训）", flush=True)
    bp = BeliefPlanner()
    prophet = ProphetPlanner()
    rng = random.Random(cfg.seed)

    opp_side = FollowerOpponent(opp, env,
                                belief=BeliefInference(opp_deck=env.deck1,
                                                       n_particles=128, seed=cfg.seed),
                                deterministic=True)
    env.opponent = opp_side
    belief = BeliefInference(opp_deck=env.deck1, n_particles=128, seed=cfg.seed)
    # A 层：训练对手池（frozen 主力 + hist PFSP + defend 脚本；sample() 按局选）
    opp_pool = _OpponentPool(cfg, env, opp_side, rng, device,
                             defender_deck_pool=defender_deck_pool)
    _t_policy = time.monotonic()

    # —— 评估对照组（2026-09-07）：对手每 copy_every 步同步变强，solo 曲线自我对冲
    # 没有区分度。补两组固定参照对手，只做对比、绝不进训练/迭代：
    #   baseline0 = 训练起点模型（fresh=随机初始化；resume=断点起点权重）——
    #               回答"比开始时强了多少"；
    #   baseline_prev = 上一个评估点权重（每周期末刷新）——回答"这一段有没有真涨"。
    # 对照结果写 solo_state.json 的 controls 数组，dashboard/分析按对手分别画线。
    baseline0 = FollowerPolicy(hidden=cfg.hidden_dim, plan_dim=PLAN_DIM,
                               belief_dim=belief_dim)
    baseline0.to_device(device)
    baseline_prev = FollowerPolicy(hidden=cfg.hidden_dim, plan_dim=PLAN_DIM,
                                   belief_dim=belief_dim)
    baseline_prev.to_device(device)
    _controls_ready = {"synced": False}   # main 权重定稿后一次性同步 baseline0

    def _sync_controls_once():
        if not _controls_ready["synced"]:
            _sync_frozen_copy(main, baseline0)
            _sync_frozen_copy(main, baseline_prev)
            _controls_ready["synced"] = True

    def eval_control(step, label, opp_model, seed):
        """main vs 对照对手打 n_eval_games 局（确定性），返回 stats dict（不落盘、不迭代）。"""
        if int(cfg.eval_workers) > 1:
            stats, _ = eval_solo_parallel(env, main, opp_model, int(cfg.n_eval_games),
                                          int(cfg.max_ep_steps), seed, cfg,
                                          n_workers=int(cfg.eval_workers),
                                          record_replays=False, step=step,
                                          frozen_step=None, save_replays=False)
        else:
            stats, _ = eval_solo(env, main, opp_model, int(cfg.n_eval_games),
                                 int(cfg.max_ep_steps), seed, cfg,
                                 record_replays=False, step=step,
                                 frozen_step=None, save_replays=False)
        stats = dict(stats)
        stats["vs"] = label
        return stats

    def eval_and_write(step):
        _sync_controls_once()
        if int(cfg.eval_workers) > 1:
            stats, _ = eval_solo_parallel(env, main, opp, int(cfg.n_eval_games),
                                          int(cfg.max_ep_steps), cfg.seed + step, cfg,
                                          n_workers=int(cfg.eval_workers),
                                          record_replays=record_replays,
                                          replays_dir=cfg.replays_dir(), step=step,
                                          frozen_step=frozen_step)
        else:
            stats, _ = eval_solo(env, main, opp, int(cfg.n_eval_games),
                                 int(cfg.max_ep_steps), cfg.seed + step, cfg,
                                 record_replays=record_replays,
                                 replays_dir=cfg.replays_dir(), step=step,
                                 frozen_step=frozen_step)
        history.append(stats)
        # 对照组：种子错开 50000/60000，与主评估、彼此互不重叠
        controls = []
        try:
            controls.append(eval_control(step, "baseline0", baseline0,
                                         cfg.seed + 50000 + step))
            controls.append(eval_control(step, "baseline_prev", baseline_prev,
                                         cfg.seed + 60000 + step))
        except OSError as e:
            print(f"[solo] 对照评估失败（不影响主评估/训练）: {e!r}", flush=True)
        write_solo_state(cfg.solo_state_path(), cfg, history, step,
                         status="done" if step >= cfg.total_steps else "running",
                         deck=list(mirror_deck), controls=controls)
        # 对照结束、主 checkpoint 落盘后，把"上一评估点"推进到当前权重
        _sync_frozen_copy(main, baseline_prev)
        print(f"[solo] eval@{step}: 胜率 {stats['winrate']:.3f}±{stats['winrate_se']:.3f} "
              f"({stats['wins']}W/{stats['losses']}L/{stats['draws']}D, "
              f"{stats['games']}局) mean_reward={stats['mean_reward']:.3f}", flush=True)
        for c in controls:
            print(f"[solo]   vs {c['vs']}: 胜率 {c['winrate']:.3f}±{c['winrate_se']:.3f} "
                  f"({c['wins']}W/{c['losses']}L/{c['draws']}D)", flush=True)
        save_checkpoint(main, cfg.solo_main_path())
        save_checkpoint(main, cfg.solo_ckpt_path(step))   # 历史版本保留（solo_main_<step>.pt）
        torch.save(ppo.opt.state_dict(), cfg.solo_opt_path())   # 断点续练恢复 Adam
        with open(cfg.run_state_path(), "w", encoding="utf-8") as f:
            json.dump({"step": int(step), "solo_ckpt": cfg.solo_main_path(),
                       "solo_opt": cfg.solo_opt_path(),
                       "config": cfg.name, "device": device}, f)

    # 训练开始先跑一次评估（WebUI 立即有真实数据）；resume 时不重跑起始评估
    if cfg.eval_at_start and start_step == 0:
        _t_eval0 = time.monotonic()
        eval_and_write(0)
        _t_eval1 = time.monotonic()
        print(f"[solo] 启动耗时分解: torch+CUDA init={_t_cuda-_t_start:.1f}s | "
              f"cfg/dirs={_t_cfg-_t_cuda:.1f}s | 环境+卡牌(BattleState/arena/卡池)="
              f"{_t_env-_t_cfg:.1f}s | 策略/信念/PPO={_t_policy-_t_env:.1f}s | "
              f"eval@0 {cfg.n_eval_games}局={_t_eval1-_t_eval0:.1f}s "
              f"(并行worker={cfg.eval_workers}) | "
              f"合计(A→B)={_t_eval1-_t_cfg:.1f}s", flush=True)

    obs, _ = env.reset()
    belief.reset(env.deck1)
    hidden = None
    last_eval_step = start_step
    ep_obs, ep_belief, ep_plan, ep_bundle, ep_lp, ep_val, ep_rew = [], [], [], [], [], [], []
    ep_term, ep_trunc, ep_masks, ep_init = [], [], [], []
    transitions = []
    stall_count = 0                 # 训练环僵局探针（与 eval 同语义）
    last_hp = None
    _t0 = time.monotonic()

    def _new_episode_reset(winner):
        """局间重置 + A 层对手池采样。

        winner：上一局胜负（0/1/None），先于 env.reset() 由调用方捕获 → 回填 PFSP
        （首局前的 initial reset 不经本函数，无 winner 可回填）；随后采样本局对手
        （frozen/hist/defend）并替换 env.opponent。"""
        # 缓冲区必须一并 nonlocal：漏了的话这里只是给嵌套函数自己的局部名字
        # 绑新列表，外层循环的 ep_* 从未被清空 → done 后每迭代重复 flush/结算，
        # 平局惩罚在泄漏缓冲上逐帧堆积（adv=-671、value loss 43 万的事故形态）。
        nonlocal obs, hidden, last_hp, stall_count, \
            ep_obs, ep_belief, ep_plan, ep_bundle, ep_lp, ep_val, ep_rew, \
            ep_term, ep_trunc, ep_masks, ep_init
        opp_pool.record(winner)
        kind, side, hist_id = opp_pool.sample()
        env.opponent = side
        # 对手卡组工厂：defend 脚本有 deck_pool（四卡组）时每局抽一副；
        # frozen/hist（FollowerOpponent）或无 deck_pool → None = 保持镜像固定卡组。
        env.deck1_factory = (side.deck if kind == "defend"
                             and getattr(side, "deck_pool", None) else None)
        obs, _ = env.reset()
        belief.reset(env.deck1)
        hidden = None
        last_hp = None
        stall_count = 0
        ep_obs, ep_belief, ep_plan, ep_bundle, ep_lp, ep_val, ep_rew = [], [], [], [], [], [], []
        ep_term, ep_trunc, ep_masks, ep_init = [], [], [], []

    for step in range(start_step + 1, cfg.total_steps + 1):
        # —— 训练环僵局早停（纯 RL 修复）：连续 100 步双方塔血零变化 → 判平结束本局。
        # 否则躺平要拖满 max_ep_steps 才在 360 帧末罚一次 −10，(γλ)^k 视野内不可见，
        # “平局=失败”只修了终局、没修信用视野。与 eval 用同一个探针/判罚语义。
        if cfg.train_stall_stop and len(ep_rew) and len(ep_rew) % STALL_WINDOW == 0:
            early, last_hp, stall_count = _stall_probe(env, last_hp, stall_count)
            if early:
                if env.battle.winner is None and not env.battle.game_over and ep_rew:
                    # 早停补结算：皇冠差已分胜负 → 终端胜负；皇冠相同（加时未破塔）→ 平局=失败
                    virt = timeout_winner(env.battle)
                    rw = reward_to_env(cfg)
                    if virt == 0:
                        ep_rew[-1] += float(rw["win_bonus"])
                    elif virt == 1:
                        ep_rew[-1] -= float(rw["lose_penalty"])
                    else:
                        ep_rew[-1] -= _draw_penalty(cfg)
                adv, ret = PPOTrainer.compute_gae(
                    ep_rew, ep_val, ep_term, cfg.gamma, cfg.gae_lambda,
                    truncated=ep_trunc, last_value=0.0)
                for i in range(len(ep_rew)):
                    transitions.append({"obs": ep_obs[i], "belief": ep_belief[i],
                                        "plan": ep_plan[i], "bundle": ep_bundle[i],
                                        "old_logprob": ep_lp[i], "adv": float(adv[i]),
                                        "returns": float(ret[i]), "masks": ep_masks[i],
                                        "init_hidden": ep_init[i]})
                _last_winner = env.battle.winner
                _new_episode_reset(_last_winner)
                if isinstance(env.opponent, FollowerOpponent):
                    env.opponent.reset()
                continue
        use_prophet = rng.random() < _SOLO_PROPHET_PROB
        plan = prophet.plan(env.get_prophet_state()) if use_prophet \
            else bp.plan(env.battle, belief.state(), obs)
        plan_vec = plan.to_vector()
        belief_tok = belief.encode(obs, None)
        init_hidden = hidden
        bundle, lp, val, hidden, masks = main.act(
            obs, belief_tok, plan_vec, env.get_action_mask,
            hidden=hidden, deterministic=False)
        obs2, reward, term, trunc, info = env.step(bundle)
        done = term or trunc
        ep_obs.append(obs); ep_belief.append(belief_tok); ep_plan.append(plan_vec)
        ep_bundle.append(bundle); ep_lp.append(lp); ep_val.append(val); ep_rew.append(reward)
        ep_term.append(term); ep_trunc.append(trunc); ep_masks.append(masks); ep_init.append(init_hidden)
        belief.update(obs2, info.get("opp_played"))
        obs = obs2

        if done or (len(ep_rew) >= cfg.max_ep_steps and not overtime_open(env.battle)):
            truncated = ((not term) and (len(ep_rew) >= cfg.max_ep_steps
                                         and not overtime_open(env.battle)))
            # 平局=失败：对局以无胜者结束（僵局/截断）→ 先补到期结算（皇冠差
            # 已分胜负就给终端胜负奖励），真平才按平局=失败惩罚
            virt = None
            if env.battle.winner is None and not env.battle.game_over and ep_rew:
                virt = timeout_winner(env.battle)
                rw = reward_to_env(cfg)
                if virt == 0:
                    ep_rew[-1] += float(rw["win_bonus"])
                elif virt == 1:
                    ep_rew[-1] -= float(rw["lose_penalty"])
                else:
                    ep_rew[-1] -= _draw_penalty(cfg)
            # P1-7 修复：env 恒返回 trunc=False，若不显式标记，截断局会被 compute_gae
            # 当成普通终止（next_val=0），last_value bootstrap 从未生效——躺平拖满 360
            # 帧时最后一步 δ 巨大而前面 ~300 帧毫无信号。达步数上限、非终局且未虚拟
            # 判出胜负（真平需继续）→ 标记末步用 last_value bootstrap。
            if truncated and virt is None:
                ep_trunc[-1] = True
                last_val = main.value(obs, belief_tok, plan_vec, hidden)
            else:
                last_val = 0.0
            adv, ret = PPOTrainer.compute_gae(ep_rew, ep_val, ep_term, cfg.gamma,
                                              cfg.gae_lambda, truncated=ep_trunc,
                                              last_value=last_val)
            for i in range(len(ep_rew)):
                transitions.append({"obs": ep_obs[i], "belief": ep_belief[i],
                                    "plan": ep_plan[i], "bundle": ep_bundle[i],
                                    "old_logprob": ep_lp[i], "adv": float(adv[i]),
                                    "returns": float(ret[i]), "masks": ep_masks[i],
                                    "init_hidden": ep_init[i]})
            _last_winner = env.battle.winner
            _new_episode_reset(_last_winner)
            if isinstance(env.opponent, FollowerOpponent):
                env.opponent.reset()

        if len(transitions) >= cfg.update_interval:
            batch = (transitions[:cfg.batch_size] if len(transitions) > cfg.batch_size
                     else transitions)
            n_play = sum(1 for t in batch
                         if any(sa.kind == "deploy" for sa in t["bundle"].sub_actions))
            avg_size = sum(len(t["bundle"].sub_actions) for t in batch) / max(1, len(batch))
            stats = ppo.update(batch)
            transitions = transitions[cfg.batch_size:] if len(transitions) > cfg.batch_size else []
            print(f"[solo step {step}] policy={stats['policy_loss']:.4f} "
                  f"value={stats['value_loss']:.4f} entropy={stats['entropy']:.4f} "
                  f"| deploy={100.0 * n_play / len(batch):.1f}% bundle={avg_size:.2f} "
                  f"ratio={stats['ratio_mean']:.3f} clip={100.0 * stats['clip_frac']:.1f}% "
                  f"adv={stats['adv_mean']:+.3f}±{stats['adv_std']:.3f} "
                  f"gnorm={stats['grad_norm']:.2f} n={len(batch)}", flush=True)

        # 周期同步冻结副本（原版 WeightsCopyingCallback 思路）
        if cfg.solo_copy_every and step % cfg.solo_copy_every == 0:
            _sync_frozen_copy(main, opp)
            frozen_step = step
            print(f"[solo] 冻结副本已同步 @step {step}", flush=True)

        if cfg.steps_per_eval and step % cfg.steps_per_eval == 0:
            eval_and_write(step)
            last_eval_step = step

    print(f"[solo] 训练循环耗时 {time.monotonic() - _t0:.1f}s", flush=True)
    save_checkpoint(main, cfg.solo_main_path())
    if last_eval_step != cfg.total_steps:
        eval_and_write(cfg.total_steps)
    print(f"[done] solo '{cfg.name}' 完成，产物在 {cfg.folder()} "
          f"（solo_state.json 供 dashboard --solo 读取）")


if __name__ == "__main__":
    from rl.config import TrainConfig
    run_solo(TrainConfig.resolve("standard"))
