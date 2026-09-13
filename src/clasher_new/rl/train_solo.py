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
from rl.ppo import PPOTrainer, ReturnScaler
from rl import diagnostics as _diag
from rl.config import reward_to_env, DEFAULT_OPP_MIX
from rl.train_follower import FollowerOpponent
from rl.pfsp import PFSP as _PFSP
from rl.run_league import (resolve_device, _bundle_cards, LeagueGameRecorder,
                           _stall_probe, STALL_WINDOW, _load_run_state,
                           timeout_winner, overtime_open, settle_stall)
from rl.replay import save_league_replays


# print 兜底（编码不支持的字符降级为 '?'，避免 cp936 控制台崩训练）——实现见
# rl/diagnostics.print_safe；此别名保持本模块既有调用点不变。
_print_safe = _diag.print_safe

#: 固定卡组（连弩 2.9，2026-09-09 用户切换）：双方镜像使用同一副。
#: 原版 8 卡（Knight/MiniPekka/...）"几乎没有实战价值"——Xbow 核心的自闭阵地
#: archetype 才有真实战术结构（阵地/法术解场/循环防守）。与 FOUR_DECK_SET 的
#: X弩同族（IceWizard 位置相同），docs/four_decks_manual.md。
DEFAULT_SOLO_DECK = ["Xbow", "Tesla", "Skeletons", "IceWizard",
                     "Archer", "Knight", "Log", "Fireball"]

#: E1 固定随机锚点（2026-09-12，v3 §3.8.4 取证后）：自对弈无外部锚点会在策略循环里
#: 打转（取证：main@20000 打冻结副本 0.85 / 打起点随机 0.13 / 打全新随机 0.505 =
#: RPS 三角；`scripts/_forensics_cycling.py`）。每个评估点加一组 "main vs 固定种子
#: 随机策略" 对照来量化**绝对强度**；固定种子保证跨评估点/跨 run 可复现。
#: RAND_ANCHOR_SEED 是锚点权重的种子；RAND_ANCHOR_EVAL_SEED 是锚点对局的固定评估种子
#: （同 40 局每评估点重打，曲线逐点可比）。
#: RAND_ANCHOR_WARN_FLOOR 未标定——40 局 1σ≈0.078，0.35 ≈ 起点 0.5 下方 ~2σ，仅作
#: 报警线（不阻断训练）。
RAND_ANCHOR_SEED = 99999
RAND_ANCHOR_EVAL_SEED = 90000
RAND_ANCHOR_WARN_FLOOR = 0.35


def _rand_anchor_warns(winrate, floor=RAND_ANCHOR_WARN_FLOOR):
    """绝对强度警报（E1）：低于阈值返回告警列表（空=通过）。阈值未标定，先只报警。"""
    if winrate is None or winrate >= floor:
        return []
    return [f"vs 固定随机锚点 胜率={winrate:.3f} < {floor}"
            "（自对弈可能在循环里打转，v3 §3.8.4 取证）"]


def _make_rand_anchor(cfg, belief_dim, device=None):
    """构造固定随机锚点策略（E1/E2 共用，权重种子 RAND_ANCHOR_SEED）。

    - 权重只由 RAND_ANCHOR_SEED + 网络结构决定 ⇒ 训练侧锚点（E2 对手池第 4 槽）与
      评估侧锚点（E1 的 baseline_rand 对照）**逐位一致**，绝对强度测量与训练目标对齐；
    - 构造前后保存/恢复 torch CPU RNG（FollowerPolicy 初始化只消费 CPU RNG），
      不扰动调用方的随机序列；
    - 锚点永不参与训练 / 永不同步 / 不进 PFSP。
    - **架构标志必须随 cfg**（B'/E'）：锚点会作为 control 的 `opp_model` 传进
      `eval_solo_parallel`，worker 按 `env_kwargs`（= cfg 架构）构造网络再
      `load_state_dict(opp_sd)`——键集不匹配会**每周期 worker 启动失败 → 静默降级串行**
      （2026-09-12 在 E' 冒烟里实际踩到：worker 报 Missing key(s) value_enc_fc/value_head_mlp）。
    """
    from rl.follower import FollowerPolicy
    from rl.plan_space import PLAN_DIM
    _rng_save = torch.get_rng_state()
    torch.manual_seed(RAND_ANCHOR_SEED)
    pol = FollowerPolicy(hidden=cfg.hidden_dim, plan_dim=PLAN_DIM,
                         belief_dim=belief_dim,
                         value_bypass=bool(getattr(cfg, "value_bypass", False)),
                         value_independent=bool(getattr(cfg, "value_independent", False)))
    torch.set_rng_state(_rng_save)
    if device is not None:
        pol.to_device(device)
    return pol


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
#: 训练局对手构成：frozen 0.1 / hist 0.6 / defend 0.2 / rand_anchor 0.1
#: （E2，2026-09-12：rand_anchor = 固定随机锚点进训练分布，打破自对弈 RPS 循环）
#: （D1，2026-09-13，`docs/cycling_league_plan_2026-09-13.md`：frozen 0.4→0.1、
#:  hist 0.3→0.6。理由：`frozen` 是唯一与当前策略**同步演化**的对手（每 2000 步同步）
#:  = RPS 锁步的载体；hist 是慢变的历史分布，追逐它拿不到"只赢近亲"的快钱。
#:  rand_anchor 剂量**刻意不变**（E2 已测 0.1 无效 ⇒ 留给 D2 做剂量-反应）。
#:  注意：配置真源是 `config.DEFAULT_OPP_MIX`，本常量只是 cfg 缺失时的兜底，两处须同步。）
_OPP_MIX = {"frozen": 0.1, "hist": 0.6, "defend": 0.2, "rand_anchor": 0.1}
#: hist 采样池：从磁盘 checkpoint 目录收集 solo_main_<step>.pt（最多保留 12 个，
#: 按步数均匀抽样——几百个文件全加载内存吃不消）
_HIST_POOL_MAX = 12
#: D1：PFSP 门禁参数（训练侧显式传入；`rl/pfsp.py` 的默认值仍是旧行为）。
#: alpha 0.05→0.20：20k run ≈64 局、hist 12 ckpt ⇒ 每 ckpt ≈1.5 局，旧 EMA 几乎不动。
#: gate：EMA 胜率 > 0.85 的 ckpt（"已能碾压的旧自己"）权重 ×0.2，把预算让给有信息量的对手。
_PFSP_ALPHA = 0.20
_PFSP_GATE_HI = 0.85
_PFSP_GATE_PENALTY = 0.2


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


def _collect_hist_ckpts(folder, max_n=_HIST_POOL_MAX, extra_dirs=None):
    """收集 solo 输出目录的历史 checkpoint（solo_main_<step>.pt）。

    - ``folder``：本 run 目录（正在写的 ``solo_main.pt`` 不计入）；
    - ``extra_dirs``：热启动补种目录（P1-1，2026-09-11）。**本目录空/不足时**从这些
      目录补齐——修的是"热启动 run 本目录首轮训练 → hist 池为空 → 对手池退化为
      frozen+defend"（实测 9k_ft 日志 frozen=0.875/defend=0.125），而热启动恰恰是
      当前推荐做法（软偏置 hint 需热启动才能吸收）。
      本目录 ckpt 优先（更贴近当前策略分布），extra 目录按传入顺序补。
    每个目录内按 step 升序均匀抽，整体上限 max_n。

    folder 与 extra_dirs 都不可用 → 空列表（对手池退化为 frozen+defend）。"""
    def _scan(d, budget):
        if not d or not os.path.isdir(d):
            return []
        steps = []
        for fn in os.listdir(d):
            if fn.startswith("solo_main_") and fn.endswith(".pt"):
                try:
                    steps.append(int(fn[len("solo_main_"):-3]))
                except ValueError:
                    continue
        steps.sort()
        if not steps:
            return []
        if len(steps) <= budget:
            return [os.path.join(d, f"solo_main_{s}.pt") for s in steps]
        idx = np.linspace(0, len(steps) - 1, budget).astype(int)
        return [os.path.join(d, f"solo_main_{steps[i]}.pt") for i in idx]

    primary = _scan(folder, max_n)
    if len(primary) >= max_n or not extra_dirs:
        return primary
    out, seen = list(primary), {os.path.abspath(p) for p in primary}
    for d in extra_dirs:
        if len(out) >= max_n:
            break
        if folder and os.path.abspath(d) == os.path.abspath(folder):
            continue
        for p in _scan(d, max_n - len(out)):
            if len(out) >= max_n:
                break
            ap = os.path.abspath(p)
            if ap not in seen:
                seen.add(ap)
                out.append(p)
    return out


def _dedup_history(history, step):
    """按 step 去重（同 step 只留一条）。

    10e 事故形态：同一目录多次启动 + ``eval_at_start`` 在 ``start_step==0`` 时重跑
    → history 出现两条 ``step:0``（数值完全相同），dashboard 在 x=0 画两个点。
    评估落盘前统一去重。返回过滤后的新列表（调用方 ``history[:] = ...``）。"""
    st = int(step)
    return [h for h in history if int(h.get("step", -1)) != st]


def _check_gates(stats, cfg, path=None):
    """行为指标门禁（P0-2b，2026-09-11 判读整改）：**先只报警不阻断**。

    设计原则：阈值误杀代价高 → 训练在阈值上绝不中断，只把"行为退化"变成可见信号。

    阈值两种写法（见 ``TrainConfig.gates``）：
    - **数字** = 绝对阈值（旧行为；op 推断：`*_max` / `ghost_rate` 用 `<=`，其余 `>=`）；
    - **``{"rel": ">=", "frac": 0.5}``** = **相对本 run 首个评估点**的比值门禁。

    为什么改相对门禁：20k 验证跑证实绝对阈值标定错了口径——阈值 9.5 来自一次性
    取证脚本（9k_ft=10.5%），而训练内建指标对同一批 9k_ft 权重实测 28.3~38.1，
    差约 3 倍 → PASS/FAIL 语义是假的。相对自身起点则口径漂移自免疫。

    第一个评估点（eval@0）只**建立基线**、不判定；基线写进 ``gates.json`` 的
    ``baseline`` 字段，续训/后续评估沿用（不在每个点上漂移）。"""
    gates = getattr(cfg, "gates", None) or {}
    step = int(stats.get("step") or 0)
    if not gates:
        return {"step": step, "ok": True, "baseline": {}, "checks": []}
    prev = {}
    if path:
        try:
            with open(path, "r", encoding="utf-8") as f:
                prev = json.load(f) or {}
        except (OSError, ValueError):
            prev = {}
    baseline = {k: float(v) for k, v in (prev.get("baseline") or {}).items()
                if isinstance(v, (int, float))}
    # 首个评估点：只建基线（用当前这一批原始值），本点不判定
    if not baseline:
        for name in gates:
            v = stats.get(name)
            if isinstance(v, (int, float)):
                baseline[name] = float(v)
        report = {"step": step, "ok": True, "baseline": baseline,
                  "checks": [], "note": "基线点（建立 baseline，不判定）"}
        _write_gate_report(path, report)
        return report

    checks = []
    for name, spec in gates.items():
        val = stats.get(name)
        if val is None or not isinstance(val, (int, float)):
            continue
        fval = float(val)
        if isinstance(spec, dict):
            base = baseline.get(name)
            if base is None:
                continue
            op = str(spec.get("rel", ">="))
            frac = float(spec.get("frac", 1.0))
            thr = frac * float(base)
            tag = f"{frac:.0%}×起点({base:.3g})"
        else:
            try:
                thr = float(spec)
            except (TypeError, ValueError):
                continue
            op = "<=" if (name.endswith("_max") or name == "ghost_rate") else ">="
            tag = f"绝对阈值"
        ok = (fval <= thr) if op == "<=" else (fval >= thr)
        checks.append({"name": name, "value": fval, "op": op,
                       "threshold": round(thr, 4), "baseline": baseline.get(name),
                       "basis": tag, "ok": bool(ok)})
    report = {"step": step, "ok": all(c["ok"] for c in checks),
              "baseline": baseline, "checks": checks}
    _write_gate_report(path, report)
    for c in checks:
        if not c["ok"]:
            print(f"[gate] WARN @step {step}: {c['name']}={c['value']:.3f} "
                  f"不满足 {c['op']} {c['threshold']:.3f}"
                  f"（{c['basis']}；只报警，不中断训练）", flush=True)
    return report


def _write_gate_report(path, report):
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"[gate] 写 {path} 失败（忽略，不影响训练）: {e!r}", flush=True)



class _OpponentPool:
    """训练对手选择器（9j + E2）：frozen / hist / defend / rand_anchor 四类按 mix 采样。

    - frozen：返回主 frozen_copy（FollowerOpponent，权重周期同步）——原行为；
    - hist：从历史 checkpoint 池 PFSP 采样一个，载入专用 hist 策略 → 包装
      FollowerOpponent（belief/planner 完整链路）；PFSP 权重按 main 对各 hist
      ckpt 的近期胜率（低胜率高权重，pfsp.PFSP 语义）；
    - defend：SelfDefenderPolicy（真防守脚本）。
    - rand_anchor（E2，2026-09-12）：固定随机锚点（权重种子 RAND_ANCHOR_SEED，
      deterministic 决策）——把"输给固定外部基准"放进训练分布，打破自对弈 RPS
      循环（v3 §3.8.4 取证）。与 E1 评估侧 baseline_rand 同权重；永不参与训练/
      同步/PFSP。
    每局开始由外部调 sample()，返回的对手直接赋给 env.opponent。
    """

    def __init__(self, cfg, env, frozen_side, rng, device, defender_deck_pool=None,
                 hist_seed_dirs=None):
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
        # P1-1：本目录空时从 hist_seed_dirs（旧 run 目录）补种，恢复 70/20/10。
        seed_dirs = list(hist_seed_dirs if hist_seed_dirs is not None
                         else (getattr(cfg, "hist_seed_dirs", None) or []))
        # P1-1b：配比可被 cfg.opp_mix 覆盖（缺省 DEFAULT_OPP_MIX = 0.4/0.3/0.2/0.1）
        self.mix = dict(getattr(cfg, "opp_mix", None) or _OPP_MIX)
        self.hist_seed_dirs = seed_dirs
        self.hist_paths = _collect_hist_ckpts(cfg.folder(), extra_dirs=seed_dirs)
        # D1：PFSP 门禁参数（缺省参数 = 旧行为；训练侧显式开启，见 _PFSP_* 常量）
        self.pfsp_alpha = float(getattr(cfg, "pfsp_alpha", None) or _PFSP_ALPHA)
        self.pfsp_gate_hi = float(getattr(cfg, "pfsp_gate_hi", None) or _PFSP_GATE_HI)
        self.pfsp_gate_penalty = float(getattr(cfg, "pfsp_gate_penalty", None)
                                       or _PFSP_GATE_PENALTY)
        self._pfsp = _PFSP(beta=1.0, seed=cfg.seed + 11, alpha=self.pfsp_alpha,
                           gate_hi=self.pfsp_gate_hi,
                           gate_penalty=self.pfsp_gate_penalty)
        self._reindex_hist()
        self._hist_policy = None             # 惰性建（需要 belief_dim/hidden_dim）
        self._hist_side = None
        self._loaded_path = None
        self._last_kind = None
        self._last_hist_id = None
        #: D1：本 run 累计各对手类型的局数（判读用的"实际配比"指纹）。
        self.kind_counts = {}
        # E2：训练侧固定随机锚点（权重与 E1 baseline_rand 逐位一致）。
        # belief_dim 口径与 _ensure_hist 相同（opp_deck=env.deck1, n_particles=128）。
        self.rand_anchor_side = None
        if float(self.mix.get("rand_anchor", 0) or 0) > 0:
            _ab_dim = len(BeliefInference(opp_deck=self.env.deck1, n_particles=128,
                                          seed=0).encode(None, None))
            self._rand_anchor_pol = _make_rand_anchor(cfg, _ab_dim, device=device)
            self.rand_anchor_side = FollowerOpponent(
                self._rand_anchor_pol, self.env,
                belief=BeliefInference(opp_deck=self.env.deck1, n_particles=128,
                                       seed=cfg.seed + 13),
                deterministic=True)
        _ra = float(self.mix.get("rand_anchor", 0) or 0)
        if self.hist_paths:
            n_seed = sum(1 for p in self.hist_paths
                         if seed_dirs and any(os.path.abspath(p).startswith(
                             os.path.abspath(d) + os.sep) for d in seed_dirs))
            print(f"[solo] 对手池: hist ckpts={len(self.hist_paths)}"
                  + (f"（其中 {n_seed} 来自补种目录 {seed_dirs}）" if n_seed else "")
                  + f" mix frozen={self.mix['frozen']}/hist={self.mix['hist']}/"
                    f"defend={self.mix['defend']}/rand_anchor={_ra}", flush=True)
        else:
            _den = 1.0 - self.mix["hist"]
            print(f"[solo] 对手池: 无历史 ckpt（本目录首轮训练，且无 --hist-seed-dir），"
                  f"退化为 frozen={self.mix['frozen']/_den:.3f} "
                  f"/ defend={self.mix['defend']/_den:.3f} "
                  f"/ rand_anchor={_ra/_den:.3f}", flush=True)
        print(f"[solo] PFSP 门禁（D1）: alpha={self.pfsp_alpha} "
              f"gate_hi={self.pfsp_gate_hi} gate_penalty={self.pfsp_gate_penalty}"
              f"（hist 池每 {getattr(cfg, 'solo_copy_every', 0)} 步重扫本 run 目录）",
              flush=True)

    def _reindex_hist(self):
        """D1：把 hist 池映射到**稳定 id**（父目录名 + 文件名），供 PFSP 统计跨刷新保留。

        旧实现用 `hist_<下标>` —— 池一旦动态增长（新快照进池），下标与 ckpt 的对应关系
        就会错位，PFSP 胜率会被张冠李戴。用文件名做 id 则在刷新前后指向同一个 ckpt。
        """
        self._hist_id = {
            p: "hist_" + os.path.basename(os.path.dirname(p)) + "_"
               + os.path.basename(p)[len("solo_main_"):-len(".pt")]
            for p in self.hist_paths}

    def refresh_hist(self, step=None):
        """D1：重扫磁盘，把本 run 自己的新快照纳入 hist 池（原来只在 __init__ 扫一次，
        `--fresh` 时本目录为空 ⇒ 整个 run 的 hist 槽全是外部旧 ckpt）。

        返回新增 ckpt 数。父目录优先、上限 _HIST_POOL_MAX 的选择逻辑在
        `_collect_hist_ckpts` 内（本目录 ckpt 排前）。
        """
        old = set(self.hist_paths)
        paths = _collect_hist_ckpts(self.cfg.folder(), extra_dirs=self.hist_seed_dirs)
        added = [p for p in paths if p not in old]
        if not added and len(paths) == len(self.hist_paths):
            return 0
        self.hist_paths = paths
        self._reindex_hist()
        own = sum(1 for p in paths
                  if os.path.dirname(p) == os.path.abspath(self.cfg.folder())
                  or os.path.dirname(p) == self.cfg.folder())
        _print_safe(f"[solo] 对手池刷新 @step {step}: hist ckpts={len(paths)}"
                    f"（本目录 {own}，新增 {len(added)}）"
                    f" | 累计对手局 {dict(sorted(self.kind_counts.items()))}")
        return len(added)

    def sample(self):
        """为本局选对手：返回 (kind, opponent, hist_id_or_None)。

        E2（2026-09-12）：新增第 4 槽 rand_anchor。无 hist ckpt 时从剩余概率归一化
        （hist 的概率空间按比例重分给 defend/rand_anchor/frozen）——同时修复旧实现
        缺陷：无 hist 时 `r < hist+defend` 未受 hist 保护，把 hist 空间误分给 defend
        （实测空目录 defend 0.78 / frozen 0.22，而打印宣称 0.286/0.714）。
        """
        kind, side, hid = self._sample_kind()
        self.kind_counts[kind] = self.kind_counts.get(kind, 0) + 1
        return kind, side, hid

    def _sample_kind(self):
        r = self.rng.random()
        mix = self.mix
        r_anchor = float(mix.get("rand_anchor", 0) or 0)
        if self.hist_paths:
            if r < mix["hist"]:
                opp_id = self._pfsp.sample("main", list(self.hist_paths))
                self._ensure_hist(opp_id)
                self._last_kind, self._last_hist_id = "hist", self._hist_id[opp_id]
                return "hist", self._hist_side, self._last_hist_id
            r -= mix["hist"]
            if r < mix["defend"]:
                self._last_kind, self._last_hist_id = "defend", None
                return "defend", self.defender, None
            if r_anchor > 0 and self.rand_anchor_side is not None \
                    and r < mix["defend"] + r_anchor:
                self._last_kind, self._last_hist_id = "rand_anchor", None
                return "rand_anchor", self.rand_anchor_side, None
            self._last_kind, self._last_hist_id = "frozen", None
            return "frozen", self.frozen_side, None
        denom = 1.0 - mix["hist"]
        if r < mix["defend"] / denom:
            self._last_kind, self._last_hist_id = "defend", None
            return "defend", self.defender, None
        if r_anchor > 0 and self.rand_anchor_side is not None \
                and r < (mix["defend"] + r_anchor) / denom:
            self._last_kind, self._last_hist_id = "rand_anchor", None
            return "rand_anchor", self.rand_anchor_side, None
        self._last_kind, self._last_hist_id = "frozen", None
        return "frozen", self.frozen_side, None

    def record(self, winner):
        """上一局结束回填 PFSP 胜率（frozen/defend 局无操作）。

        winner: 0=main 胜（score 1.0）/ 1=main 负（0.0）/ None=平局（0.5）。"""
        if self._last_kind == "hist" and self._last_hist_id is not None:
            score = {0: 1.0, 1: 0.0}.get(winner, 0.5)
            self._pfsp.update_winrate("main", self._last_hist_id, score)

    def _ensure_hist(self, path):
        """载入 hist ckpt（换目标才重载；belief_dim 尾部零拷贝兼容旧 23 维）。

        ⚠️ B'/E'（2026-09-12）：**直接用 load_checkpoint 返回的策略**（它按 ckpt 元数据
        构造正确架构），不要"默认架构建网再 load_state_dict"——hist 目录可能指向
        E'/bypass 架构的 run，键集不匹配会崩（同类 bug 第三次）。
        """
        if self._loaded_path != path:
            bd = (self._hist_policy.belief_dim if self._hist_policy is not None
                  else len(BeliefInference(opp_deck=self.env.deck1, n_particles=128,
                                           seed=0).encode(None, None)))
            self._hist_policy = load_checkpoint(path, plan_dim=PLAN_DIM, belief_dim=bd)
            self._hist_policy.to_device(self.device)
            self._loaded_path = path
            self._hist_side = None
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


def behavioral_metrics(games):
    """从回放 games 算行为指标（2026-09-10 用户：胜率在镜像自对弈下自我对冲≈0.5 恒定，
    无法反映真实爬坡——用行为质量指标替代）。复用 forensics_response 的口径，输出三组：

    防守质量：
      defense_invest_rate   敌过河帧中我方有 deploy 的比例（防守投入率）
      engagement_rate       防守部署的接敌率（部署后 8s 内 5 格内敌我 troop 同框）
      intercept_rate        防守部署的拦截率（落点在 敌→我方塔 直线路径 4 格内）
      response_latency_med  威胁开始→首次响应延迟中位数（秒）
    进攻效率：
      unilateral_rate       单边堆牌率（非防守/非响应窗口的 deploy 占比；只攻不防）
      tower_diff_avg        平均塔血差（我方总塔血−对手总塔血，正=我方领先）
    资源/组织：
      bundle_multi_rate     多卡 bundle 率（同帧 ≥2 张 deploy 的比例；组波进攻）
      deploy_per_game       每局平均 deploy 次数
      elixir_avg            整局平均圣水（低=一够费就花，AGENTS.md「不会攒费」病理）
      ghost_rate            幽灵动作率（落点 y>=20 的 deploy 占比；P0-2 门禁金丝雀）

    输入 games = [{meta, winner, frames:[{t,bundle,entities,towers0,towers1,elixir0,...}]}]。
    我方 = player 0（side0=main，下半场，向 +y 推进）；敌过河 = P1 troop y<16（RIVER）。
    """
    if not games:
        return {}
    RIVER = 16.0        # P1 troop y<16 算过河（与 forensics_response/AGENTS.md 一致）
    RESP_WINDOW = 5.0   # 对手出牌后 5s 内我方 deploy = 响应（与 forensics 一致）
    # 累计器
    def_frames = 0          # 敌过河帧
    def_deploy_frames = 0   # 敌过河帧中有我方 deploy
    engagements = []        # 防守部署接敌 (0/1)
    intercepts = []         # 防守部署拦截 (0/1)
    latencies = []          # 威胁区间首次响应延迟
    unilateral = 0          # 单边堆牌 deploy 数
    response = 0            # 响应窗口 deploy 数
    multi_bundles = 0       # 多卡 bundle 帧数
    deploy_frames = 0       # 有 deploy 的帧数
    deploy_total = 0        # 总 deploy 数
    ghost_deploys = 0       # 幽灵动作（落点 y>=20 越界/脏回放）数，见下方注释
    elixir_sum = 0.0
    elixir_frames = 0
    tower_diff_sum = 0.0
    tower_diff_frames = 0
    n_games = len(games)

    for g in games:
        frames = g.get("frames") or []
        nf = len(frames)
        in_threat = False
        threat_start = None
        last_opp_t = -999.0   # 最近对手出牌时间（跨帧，供响应窗口判定）
        for idx, fr in enumerate(frames):
            ents = fr.get("entities") or []
            # P1 部队过河判定（我方视角：P0=我，P1=敌；RIVER=15，P1 从 y>15 推向下）
            p1_troops = [(float(e[1]), float(e[2])) for e in ents
                         if len(e) > 5 and e[5] == "troop" and int(e[4]) == 1]
            p0_troops = [(float(e[1]), float(e[2])) for e in ents
                         if len(e) > 5 and e[5] == "troop" and int(e[4]) == 0]
            foes_crossed = any(y < RIVER for _, y in p1_troops)
            bundle = fr.get("bundle") or []
            # 去重：同帧内相同的 (slot,x,y) deploy 只算一次（防历史脏回放把同一部署
            # 重复记录 inflate 计数——实测 league_40000 有每帧 4× 重复 deploy 的脏帧）
            my_deploys = []
            _seen = set()
            for b in bundle:
                if b[0] == "deploy":
                    key = (b[1], b[2], b[3])
                    if key not in _seen:
                        _seen.add(key)
                        my_deploys.append(b)
            elixir_sum += float(fr.get("elixir0", 0) or 0)
            elixir_frames += 1
            t0 = fr.get("t") or 0.0
            t1 = fr.get("towers1") or [4824, 3052, 3052]
            t0s = fr.get("towers0") or [4824, 3052, 3052]
            tower_diff_sum += sum(t0s) - sum(t1)
            tower_diff_frames += 1
            if fr.get("opp_played"):
                last_opp_t = t0   # 本帧对手出牌 → 刷新最近对手出牌时间
            if my_deploys:
                deploy_frames += 1
                deploy_total += len(my_deploys)
                # 幽灵动作率（P0-2 门禁）：y>=20 的落点在提交路径应被 validate_bundle
                # 拒绝，出现即说明"脏回放/越界落点"，同时是历史脏数据的金丝雀。
                ghost_deploys += sum(1 for b in my_deploys if float(b[3]) >= 20)
                if len(my_deploys) >= 2:
                    multi_bundles += 1
                if foes_crossed:
                    def_deploy_frames += 1
                    for b in my_deploys:
                        wx, wy = b[2] + 0.5, b[3] + 0.5
                        if b[3] >= 20:
                            continue   # 幽灵动作：非法尝试不参与空间统计
                        if p1_troops:
                            # 接敌：部署后 8s 内 5 格内敌我 troop 同框
                            eng = False
                            for j in range(idx + 1, min(nf, idx + 1 + int(8.0 / 0.5))):
                                fj = frames[j]
                                if (fj.get("t") or 0.0) > t0 + 8.0:
                                    break
                                near = [e for e in (fj.get("entities") or [])
                                        if len(e) > 5 and e[5] == "troop"
                                        and abs(float(e[1]) - wx) + abs(float(e[2]) - wy) < 5.0]
                                if any(int(e[4]) == 0 for e in near) and \
                                        any(int(e[4]) == 1 for e in near):
                                    eng = True
                                    break
                            engagements.append(1 if eng else 0)
                            # 拦截：落点在 敌→我方塔 直线路径 4 格内
                            fy = min(p1_troops, key=lambda p: abs(wx - p[0]) + abs(wy - p[1]))
                            tx, ty = 8.5, 6.0
                            vx, vy = tx - fy[0], ty - fy[1]
                            L2 = vx * vx + vy * vy
                            if L2 > 1e-9:
                                s = max(0.0, min(1.0, ((wx - fy[0]) * vx
                                                       + (wy - fy[1]) * vy) / L2))
                                px, py = fy[0] + s * vx, fy[1] + s * vy
                                intercepts.append(1 if abs(wx - px) + abs(wy - py) <= 4.0 else 0)
                            else:
                                intercepts.append(0)
                elif t0 - last_opp_t <= RESP_WINDOW:
                    response += len(my_deploys)
                else:
                    unilateral += len(my_deploys)
            if foes_crossed:
                def_frames += 1
                if not in_threat:
                    in_threat, threat_start = True, t0
            elif in_threat:
                in_threat = False
        # 威胁→首次响应延迟（逐威胁区间）
        in_threat = False
        threat_start = None
        for fr in frames:
            p1_troops = [(float(e[1]), float(e[2])) for e in (fr.get("entities") or [])
                         if len(e) > 5 and e[5] == "troop" and int(e[4]) == 1]
            foes_crossed = any(y < RIVER for _, y in p1_troops)
            t0 = fr.get("t") or 0.0
            if foes_crossed and not in_threat:
                in_threat, threat_start = True, t0
            elif not foes_crossed and in_threat:
                in_threat = False
            if in_threat and fr.get("bundle"):
                if any(b[0] == "deploy" for b in fr["bundle"]):
                    latencies.append(t0 - threat_start)
                    in_threat = False

    def _pct(a, b):
        return round(100.0 * a / b, 1) if b else 0.0

    stats = {
        "defense_invest_rate": _pct(def_deploy_frames, def_frames),
        "engagement_rate": _pct(sum(engagements), len(engagements)),
        "intercept_rate": _pct(sum(intercepts), len(intercepts)),
        "response_latency_med": round(sorted(latencies)[len(latencies) // 2], 2) if latencies else None,
        "unilateral_rate": _pct(unilateral, max(1, unilateral + response)),
        "tower_diff_avg": round(tower_diff_sum / max(1, tower_diff_frames), 1),
        "bundle_multi_rate": _pct(multi_bundles, deploy_frames),
        "deploy_per_game": round(deploy_total / max(1, n_games), 2),
        "elixir_avg": round(elixir_sum / max(1, elixir_frames), 2),
        "ghost_rate": _pct(ghost_deploys, deploy_total),
    }
    return stats


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
                                 steps=(step, frozen_step),
                                 decks=(env.deck0, env.deck1)) if record_replays else None
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
                rec.record(env, bundle, reward, info, cards=played)
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
    if replays:
        # 行为指标（2026-09-10）：镜像自对弈胜率自我对冲≈0.5，行为质量才是真实爬坡。
        # 对照组评估 record_replays=False → replays 空 → 无行为指标（只给主对手曲线算）。
        try:
            stats.update(behavioral_metrics(replays))
        except Exception as e:
            print(f"[eval] 行为指标计算失败（忽略，不影响评估）: {e!r}", flush=True)
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
                              belief_dim=belief_dim,
                              value_bypass=bool(env_kwargs.get("value_bypass", False)),
                              value_independent=bool(env_kwargs.get("value_independent", False)))
        opp = FollowerPolicy(hidden=env_kwargs["hidden_dim"], plan_dim=PLAN_DIM,
                             belief_dim=belief_dim,
                             value_bypass=bool(env_kwargs.get("value_bypass", False)),
                             value_independent=bool(env_kwargs.get("value_independent", False)))
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
                                            env_kwargs.get("frozen_step")),
                                     decks=(env.deck0, env.deck1)) if record else None
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
                    rec.record(env, bundle, reward, info, cards=played)
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
                  "eval_step": step, "frozen_step": frozen_step,
                  "value_bypass": bool(getattr(cfg, "value_bypass", False)),
                  "value_independent": bool(getattr(cfg, "value_independent", False))}
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
    if replays:
        try:
            stats.update(behavioral_metrics(replays))
        except Exception as e:
            print(f"[eval] 行为指标计算失败（忽略，不影响评估）: {e!r}", flush=True)
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
          f"冻结副本同步间隔={cfg.solo_copy_every}, value_norm={cfg.value_norm}, "
          f"adv_norm={cfg.adv_norm}, diagnose_every={cfg.diagnose_every}, "
          f"n_eval_games={cfg.n_eval_games}, eval_workers={cfg.eval_workers}, "
          f"opp_mix={getattr(cfg, 'opp_mix', None) or _OPP_MIX})", flush=True)
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
                               plan_dim=PLAN_DIM, belief_dim=belief_dim,
                               value_bypass=cfg.value_bypass,
                               value_independent=cfg.value_independent)
    else:
        main = FollowerPolicy(hidden=cfg.hidden_dim, plan_dim=PLAN_DIM,
                              belief_dim=belief_dim, value_bypass=cfg.value_bypass,
                              value_independent=cfg.value_independent)
    main.to_device(device)
    if rs and rs.get("solo_ckpt") and os.path.exists(rs["solo_ckpt"]):
        # resume：断点权重为准（覆盖 main_init）
        main = load_checkpoint(rs["solo_ckpt"], hidden_dim=cfg.hidden_dim,
                               value_bypass=cfg.value_bypass,
                               value_independent=cfg.value_independent)
        main.to_device(device)
        print(f"[solo] resume 从 step {start_step} 续训（继续到 {cfg.total_steps}）", flush=True)
    # B'/E'（2026-09-12）：value 架构一致性检查（load_checkpoint 只按元数据/显式值
    # 构造，这里兜底确认与配置一致；不一致 = value 通路语义错位，续训须 --fresh）。
    for _flag in ("value_bypass", "value_independent"):
        if bool(getattr(main, _flag, False)) != bool(getattr(cfg, _flag, False)):
            _print_safe(f"[solo] ⚠️ main.{_flag}={bool(getattr(main, _flag, False))} "
                        f"与 cfg.{_flag}={bool(getattr(cfg, _flag, False))} 不一致："
                        f"value 通路语义错位，必须 --fresh 重训")
    # v3 P0-C 启动前检查：静态可判定的饱和病因（enc_ln 缺失 / 被 Identity 替换）。
    # 真实 GRU 活力需要 rollout 帧，只能在评估点测（check_vitality + vitality_warns 落盘）。
    for _w in _diag.check_policy_architecture(main):
        _print_safe(f"[solo] ⚠️ 启动前检查未通过: {_w}")
    # F'（2026-09-12）resume 脚枪：config.json 不参与 resume 解析，续训必须重传
    # --ppo-epochs/--ppo-minibatch/--ppo-shuffle，否则静默退回"1 次梯度步/更新"。
    if rs and rs.get("ppo_epochs") is not None:
        _old_budget = (int(rs.get("ppo_epochs", 1)),
                       int(rs.get("ppo_minibatch", 0)),
                       bool(rs.get("ppo_shuffle", False)))
        _new_budget = (int(cfg.ppo_epochs), int(cfg.ppo_minibatch),
                       bool(cfg.ppo_shuffle))
        if _old_budget != _new_budget:
            _print_safe(
                f"[solo] ⚠️ PPO 更新预算与断点记录不一致：断点 "
                f"epochs={_old_budget[0]} minibatch={_old_budget[1]} "
                f"shuffle={_old_budget[2]} / 本次 epochs={_new_budget[0]} "
                f"minibatch={_new_budget[1]} shuffle={_new_budget[2]} —— "
                f"续训请重传同样的 --ppo-* 参数（不同预算混跑 = 两个实验拼在一起）")
    opp = FollowerPolicy(hidden=cfg.hidden_dim, plan_dim=PLAN_DIM, belief_dim=belief_dim,
                         value_bypass=cfg.value_bypass,
                         value_independent=cfg.value_independent)
    opp.to_device(device)
    _sync_frozen_copy(main, opp)   # 开局副本 = main（resume 后即断点权重）
    frozen_step = start_step       # 冻结副本当前所在训练步（录像 meta.steps 用）
    ppo = PPOTrainer(main, lr=cfg.lr, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda,
                     clip=cfg.clip, vf_coef=cfg.vf_coef, ent_coef=cfg.ent_coef,
                     max_grad_norm=cfg.max_grad_norm, adv_norm=cfg.adv_norm,
                     value_norm=cfg.value_norm, diagnose_every=cfg.diagnose_every,
                     n_epochs=cfg.ppo_epochs, minibatch_size=cfg.ppo_minibatch,
                     shuffle=cfg.ppo_shuffle, seed=cfg.seed)
    _n_mb = (cfg.ppo_minibatch if cfg.ppo_minibatch > 0
             else max(1, cfg.update_interval))
    print(f"[solo] PPO 更新预算: epochs={cfg.ppo_epochs} "
          f"minibatch={_n_mb} shuffle={bool(cfg.ppo_shuffle)} "
          f"⇒ 每 {cfg.update_interval} 帧 "
          f"{cfg.ppo_epochs * max(1, -(-cfg.update_interval // _n_mb))} 次梯度步"
          f"{'（旧行为：1 次）' if cfg.ppo_epochs <= 1 and not cfg.ppo_shuffle and cfg.ppo_minibatch <= 0 else ''}",
          flush=True)
    if rs and rs.get("ret_scaler"):
        # 回报尺度统计随 run_state 落盘/恢复：续训不重置（否则缩放因子从头爬）。
        ppo.ret_scaler = ReturnScaler.from_dict(rs.get("ret_scaler"))
        print(f"[solo] 回报尺度统计已恢复: mean={ppo.ret_scaler.mean:.3f} "
              f"std={ppo.ret_scaler.std():.3f} (n={ppo.ret_scaler.count})", flush=True)
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
    # P1-1：--hist-seed-dir 指定旧 run 目录 → 热启动时 hist 槽不再为空（70/20/10）。
    opp_pool = _OpponentPool(cfg, env, opp_side, rng, device,
                             defender_deck_pool=defender_deck_pool,
                             hist_seed_dirs=getattr(cfg, "hist_seed_dirs", None))
    _t_policy = time.monotonic()

    # —— 评估对照组（2026-09-07）：对手每 copy_every 步同步变强，solo 曲线自我对冲
    # 没有区分度。补两组固定参照对手，只做对比、绝不进训练/迭代：
    #   baseline0 = 训练起点模型（fresh=随机初始化；resume=断点起点权重）——
    #               回答"比开始时强了多少"；
    #   baseline_prev = 上一个评估点权重（每周期末刷新）——回答"这一段有没有真涨"。
    # 对照结果写 solo_state.json 的 controls 数组，dashboard/分析按对手分别画线。
    baseline0 = FollowerPolicy(hidden=cfg.hidden_dim, plan_dim=PLAN_DIM,
                               belief_dim=belief_dim, value_bypass=cfg.value_bypass,
                               value_independent=cfg.value_independent)
    baseline0.to_device(device)
    baseline_prev = FollowerPolicy(hidden=cfg.hidden_dim, plan_dim=PLAN_DIM,
                                   belief_dim=belief_dim, value_bypass=cfg.value_bypass,
                                   value_independent=cfg.value_independent)
    baseline_prev.to_device(device)
    # E1（2026-09-12）：固定随机锚点（绝对强度参照）。权重只由 RAND_ANCHOR_SEED 决定，
    # 与 E2 训练侧锚点（_OpponentPool.rand_anchor_side）同源逐位一致；helper 内部
    # 保存/恢复 torch CPU RNG，不扰动训练序列的随机性。
    baseline_rand = _make_rand_anchor(cfg, belief_dim, device=device)
    _controls_ready = {"synced": False}   # main 权重定稿后一次性同步 baseline0
    # 训练期滚动探针（P0-1c / 步数口径，2026-09-11；v3 P0-B 扩展）：
    #   games    累计完成局数（步数口径改"局数"用；20000 步 ≈ 55~80 局）
    #   ev_pairs 本评估窗口内各次 update 的逐帧 (value, return)，评估时池化算一次 EV
    #            （旧口径"批 EV 求均值"被 128 连续帧的批内低方差放大 ~3 倍）
    #   frames   最近 96 帧 (obs, belief, plan)，供 GRU 活力探针（不推进 env）
    _probe = {"games": 0, "ev_pairs": [], "frames": [],
              "stall_games": 0, "stall_close_draws": 0}

    def _sync_controls_once():
        if not _controls_ready["synced"]:
            _sync_frozen_copy(main, baseline0)
            _sync_frozen_copy(main, baseline_prev)
            _controls_ready["synced"] = True

    def eval_control(step, label, opp_model, seed):
        """main vs 对照对手打 n_eval_games 局（确定性），返回 stats dict（不落盘、不迭代）。

        P0-2（2026-09-11）：对照组也开 record_replays（只算不落盘）→ 对照组同样带
        行为指标，可与主曲线逐项对比；原先 record_replays=False 使对照组只有胜率，
        而胜率在本分辨率下恰是最没信息量的那一项。
        """
        if int(cfg.eval_workers) > 1:
            stats, _ = eval_solo_parallel(env, main, opp_model, int(cfg.n_eval_games),
                                          int(cfg.max_ep_steps), seed, cfg,
                                          n_workers=int(cfg.eval_workers),
                                          record_replays=True, step=step,
                                          frozen_step=None, save_replays=False)
        else:
            stats, _ = eval_solo(env, main, opp_model, int(cfg.n_eval_games),
                                 int(cfg.max_ep_steps), seed, cfg,
                                 record_replays=True, step=step,
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
        # P0-2：同 step 去重（10e 出现两条 step:0 的落盘去重缺失 → dashboard x=0 双点）
        history[:] = _dedup_history(history, step)
        # v3 P0-B：EV 改**池化口径** —— 把本评估窗口累积的逐帧 (v, R) 合并后算一次 EV。
        # 旧口径（各更新批 EV 求均值）的批 = 128 连续帧，相邻帧 corr(R)≈0.99
        # → 批内 Var(R) 仅为全局 0.32 倍，逐批平均把 EV 放大约 3 倍（−0.58 vs −1.74）。
        # 这里另给一个"随机打乱后按 batch_size 分批再平均"的对照值，量化批切片剩余影响。
        _vp = _probe["ev_pairs"]
        if _vp:
            vs_all = np.concatenate([p[0] for p in _vp])
            rs_all = np.concatenate([p[1] for p in _vp])
            stats["explained_variance"] = round(
                float(PPOTrainer.explained_variance(vs_all, rs_all)), 4)
            _perm = np.random.default_rng(cfg.seed).permutation(len(rs_all))
            _bs = max(2, int(cfg.batch_size))
            _splits = np.array_split(_perm, max(1, len(rs_all) // _bs))
            _chunks = [PPOTrainer.explained_variance(vs_all[idx], rs_all[idx])
                       for idx in _splits]
            _chunks = [c for c in _chunks if c is not None]
            stats["explained_variance_batched"] = (
                round(float(np.mean(_chunks)), 4) if _chunks else None)
            # v3 §2 验收表第 3 行「value_head 输出 std / 批内 R std > 0.3」的分母：
            # 旧实现全仓没有算过 R std（diagnostics.py 里注明"调用方另测"而调用方没测）
            # → 该门槛一直不可判读。分母与分子同量纲：last_ev_pairs 存的是**未缩放**的
            # (value, return)，与 value_head 原始输出同尺度。
            stats["r_std"] = round(float(rs_all.std()), 6)
            _r_chunks = [float(np.asarray(rs_all[idx]).std()) for idx in _splits]
            stats["r_std_batch"] = (
                round(float(np.mean(_r_chunks)), 6) if _r_chunks else None)
            # 2026-09-12 口径修正（F' 首跑发现）：门槛「value_std / R std > 0.3」的
            # **分子必须与分母同窗口**。旧实现分子 value_std 来自 GRU 探针的**最近 96
            # 帧**、分母 r_std_batch 来自 **整个评估窗口** 的回报 ⇒ 两个样本集，
            # 比值不可判读（fprime_20k @2500 实测 value_std_ratio=0.0006 却 EV=+0.10，
            # 而 EV>0 数学上要求 critic 不是常数 —— 两个数不可能同时为真）。
            # 正确口径：直接用 last_ev_pairs（未缩放 value_head 输出）自身的 std。
            # 探针口径的 value_std 保留（它是 GRU 活力的指标，另一个用途），
            # 比值另存 value_std_ratio_probe 以免旧口径被误用。
            stats["value_std_ev"] = round(float(vs_all.std()), 6)
            stats["value_std_ratio"] = round(
                float(vs_all.std() / (rs_all.std() + 1e-12)), 4)
        else:
            stats["explained_variance"] = None
            stats["explained_variance_batched"] = None
            stats["r_std"] = None
            stats["r_std_batch"] = None
            # 2026-09-12 修（从 no-human-watch-B 移植）：这一行原先在 if/else **之外**
            # 无条件执行，会把上面刚算好的同窗口 `value_std_ratio`/`value_std_ev`
            # 立刻清成 None ⇒ 评估行 vstd/rstd 恒 None（gfix_20k 前两点就是这么丢的）。
            stats["value_std_ratio"] = None
            stats["value_std_ev"] = None
        _probe["ev_pairs"] = []
        # v3 P0-B：GRU 活力（h 跨帧 std / 候选饱和）—— 负 EV 的直接机理指标。
        # 门槛见 rl/diagnostics.THRESHOLDS（h_std>0.05 且 n_abs<0.9）；不达标只报警。
        try:
            _vit = _diag.gru_vitality(main, _probe["frames"])
            # 落盘键名与 dashboard/日志口径一致（gru_n_abs 而非 n_abs）
            for _src, _dst in (("h_std", "h_std"), ("n_abs", "gru_n_abs"),
                               ("value_std", "value_std")):
                _val = _vit.get(_src)
                stats[_dst] = round(float(_val), 6) if _val is not None else None
            # 探针口径的比值另存（**跨窗口、不可判读**，仅留档对照——见上面 2026-09-12
            # 口径修正：分母来自评估窗口、分子来自最近 96 帧，两个样本集）。
            _vs, _rsb = stats.get("value_std"), stats.get("r_std_batch")
            if _vs is not None and _rsb:
                stats["value_std_ratio_probe"] = round(float(_vs) / float(_rsb), 4)
            # 空缓冲（eval@0，尚未跑过训练步）不算不达标，只静默跳过。
            # 门槛用的 value_std_ratio 是**修正后的同窗口口径**（ev_pairs），
            # 不是上面那个跨窗口的 _probe 比值。
            if stats.get("value_std_ratio") is not None:
                _vit["value_std_ratio"] = stats["value_std_ratio"]
            _warns = _diag.check_vitality(_vit) if _vit else []
            # P0-C：告警落盘（旧实现只 print → 事后无法在 state/dashboard 追溯）
            stats["vitality_warns"] = _warns
            if _warns:
                _print_safe(f"[solo] ⚠️ GRU 活力不达标 @step {step}: "
                            + " | ".join(_warns))
        except Exception as e:      # 探针失败不得影响训练主流程
            # 注意：不要直接 f"{e!r}" —— repr 可能内嵌不可编码字符（见 _print_safe）
            _emsg = repr(e).encode("ascii", "replace").decode("ascii")
            stats["vitality_warns"] = [f"探针异常: {_emsg}"]
            _print_safe(f"[solo] GRU 活力探针失败（不影响训练）: {_emsg}")
        stats["cum_games"] = int(_probe["games"])
        # C'（2026-09-12）：早停低置信裁定降噪的测量 —— 早停局数 / 被降级为平局的
        # 低置信局数（皇冠相同且塔血%细差 < stall_draw_margin）。
        stats["stall_games"] = int(_probe["stall_games"])
        stats["stall_close_draws"] = int(_probe["stall_close_draws"])
        history.append(stats)
        # 对照组：种子错开 50000/60000，与主评估、彼此互不重叠
        controls = []
        try:
            controls.append(eval_control(step, "baseline0", baseline0,
                                         cfg.seed + 50000 + step))
            controls.append(eval_control(step, "baseline_prev", baseline_prev,
                                         cfg.seed + 60000 + step))
            # E1（2026-09-12）：固定随机锚点 —— 固定评估种子 ⇒ 每评估点重打同一 40 局，
            # 量化绝对强度（防自引用指标在自对弈循环里自欺，v3 §3.8.4）。
            controls.append(eval_control(step, "baseline_rand", baseline_rand,
                                         RAND_ANCHOR_EVAL_SEED))
        except OSError as e:
            print(f"[solo] 对照评估失败（不影响主评估/训练）: {e!r}", flush=True)
        write_solo_state(cfg.solo_state_path(), cfg, history, step,
                         status="done" if step >= cfg.total_steps else "running",
                         deck=list(mirror_deck), controls=controls)
        # P0-2 行为指标门禁（先只报警不阻断；对照组不参与门禁，只报主曲线）
        _check_gates(stats, cfg, cfg.gates_path())
        # 对照结束、主 checkpoint 落盘后，把"上一评估点"推进到当前权重
        _sync_frozen_copy(main, baseline_prev)
        _hs = stats.get("h_std")
        _hs_s = "n/a" if _hs is None else f"{_hs:.2e}"
        print(f"[solo] eval@{step}: 胜率 {stats['winrate']:.3f}±{stats['winrate_se']:.3f} "
              f"({stats['wins']}W/{stats['losses']}L/{stats['draws']}D, "
              f"{stats['games']}局) mean_reward={stats['mean_reward']:.3f} "
              f"EV={stats['explained_variance']}（池化）"
              f" EVb={stats.get('explained_variance_batched')} "
              f"h_std={_hs_s} n_abs={stats.get('gru_n_abs')} "
              f"vstd/rstd={stats.get('value_std_ratio')} "
              f"累计局数={stats['cum_games']}", flush=True)
        for c in controls:
            print(f"[solo]   vs {c['vs']}: 胜率 {c['winrate']:.3f}±{c['winrate_se']:.3f} "
                  f"({c['wins']}W/{c['losses']}L/{c['draws']}D)", flush=True)
        # E1：绝对强度警报（先只报警不阻断；阈值未标定，见 RAND_ANCHOR_WARN_FLOOR）
        for _c in controls:
            if _c.get("vs") == "baseline_rand":
                for _w in _rand_anchor_warns(_c.get("winrate")):
                    _print_safe(f"[solo] ⚠️ 绝对强度警报 @step {step}: {_w}")
                break
        _persist(step)

    def _persist(step):
        """落盘训练态：主快照 + 步进快照（D1 自身联赛成员）+ Adam + run_state。

        从 eval_and_write 尾部抽出（C 方案 2026-09-13），供**轻量锚点评估点**复用：
        轻点不跑 main/对照三块、只跑锚点，但保留同样的落盘语义 ⇒ 崩溃续训的粒度
        仍是 anchor_every（2500）而不是 steps_per_eval（10000）。
        """
        save_checkpoint(main, cfg.solo_main_path())
        save_checkpoint(main, cfg.solo_ckpt_path(step))   # 历史版本保留（solo_main_<step>.pt）
        torch.save(ppo.opt.state_dict(), cfg.solo_opt_path())   # 断点续练恢复 Adam
        with open(cfg.run_state_path(), "w", encoding="utf-8") as f:
            json.dump({"step": int(step), "solo_ckpt": cfg.solo_main_path(),
                       "solo_opt": cfg.solo_opt_path(),
                       "config": cfg.name, "device": device,
                       "value_norm": cfg.value_norm, "adv_norm": cfg.adv_norm,
                       # F'（2026-09-12）：PPO 更新预算写进 run_state —— config.json 不
                       # 参与 resume 解析，续训时忘记重传 --ppo-epochs/--ppo-minibatch
                       # 会静默退回旧行为（1 次梯度步/更新），这是本轮最贵的脚枪。
                       "ppo_epochs": int(cfg.ppo_epochs),
                       "ppo_minibatch": int(cfg.ppo_minibatch),
                       "ppo_shuffle": bool(cfg.ppo_shuffle),
                       "ret_scaler": ppo.ret_scaler.to_dict()}, f)

    def anchor_point(step):
        """C 方案（2026-09-13）：轻量评估点 = 只跑固定随机锚点（C1 判据原料）+ 落盘。

        不跑 main vs 冻结副本 / baseline0 / baseline_prev 三块（省掉一个评估点 3/4 的
        成本 ≈155s），因此锚点序列能维持 2500 分辨率、其余块稀疏到 steps_per_eval。

        为什么必须这么分（实测，非直觉）：D1 三跑锚点序列的谷底**只有 1 个点宽**
        （相邻点 |Δ|≈0.225~0.240 ≈ 3σ），粗采样回放显示 5000 就开始漏真谷底、10000 把
        "最差点"从 0.192 系统性抬到 0.462 —— 病态组的塌陷是持续性的（多个点 0.000，
        任何粗采样都留得住），而 D1 的低点是周期性瞬态（粗采样直接丢）。粗采样只会让
        D1 显得更好，恰是"长 run 会不会真塌"最不该失真的地方。
        """
        _sync_controls_once()   # 保证 baseline0/baseline_prev = 训练起点权重（与全点同语义）
        try:
            c = eval_control(step, "baseline_rand", baseline_rand, RAND_ANCHOR_EVAL_SEED)
        except OSError as e:
            print(f"[solo] 锚点评估失败（不影响训练）: {e!r}", flush=True)
            return
        for _w in _rand_anchor_warns(c.get("winrate")):
            _print_safe(f"[solo] ⚠️ 绝对强度警报 @step {step}: {_w}")
        _persist(step)
        write_solo_state(cfg.solo_state_path(), cfg, history, step,
                         status="running", deck=list(mirror_deck), controls=[c])
        print(f"[solo] anchor@{step}: vs baseline_rand 胜率 {c['winrate']:.3f}"
              f"±{c['winrate_se']:.3f} ({c['wins']}W/{c['losses']}L/{c['draws']}D, "
              f"{c['games']}局)", flush=True)

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

    # C 方案：评估节奏自述（事后凭日志即可复原协议，不必翻命令行）
    if cfg.anchor_every and cfg.steps_per_eval:
        _ts, _spe, _ae = int(cfg.total_steps), int(cfg.steps_per_eval), int(cfg.anchor_every)
        _n_full = 1 + _ts // _spe
        _n_light = sum(1 for s in range(_ae, _ts + 1, _ae) if s % _spe)
        print(f"[solo] 评估节奏: 全点每 {_spe} 步（{_n_full} 个，含 eval@0）/ "
              f"轻量锚点每 {_ae} 步（{_n_light} 个，只跑 baseline_rand + 落盘）",
              flush=True)

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
        _probe["games"] += 1
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
                    # 早停补结算：C'（2026-09-12）低置信裁定降噪 —— 皇冠相同的
                    # 塔血%细差 < stall_draw_margin 视为掷硬币，不再判胜负（记平局=
                    # 失败）；皇冠不同/细差决定性照常 ±胜负。皇冠相同但细差小 → 平局。
                    virt = settle_stall(env.battle, getattr(cfg, "stall_draw_margin", 0.05))
                    rw = reward_to_env(cfg)
                    if virt == 0:
                        ep_rew[-1] += float(rw["win_bonus"])
                    elif virt == 1:
                        ep_rew[-1] -= float(rw["lose_penalty"])
                    else:
                        ep_rew[-1] -= _draw_penalty(cfg)
                    _probe["stall_games"] += 1
                    if virt is None:
                        _probe["stall_close_draws"] += 1
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
        # v3 P0-B：维护最近 96 帧，供评估时 GRU 活力探针用（纯前向，不推进 env）
        _probe["frames"].append((obs, belief_tok, plan_vec))
        if len(_probe["frames"]) > 96:
            _probe["frames"].pop(0)
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
            # v3 P0-B：累积逐帧 (value, return)，评估窗口结束时池化算一次 EV
            if getattr(ppo, "last_ev_pairs", None) is not None:
                _probe["ev_pairs"].append(ppo.last_ev_pairs)
                # 内存上界（P0-B 审计缺口）：steps_per_eval=0 时该列表在 eval_and_write
                # 之外永不清理（frames 有 96 上限，ev_pairs 原先没有）→ 长跑无界增长。
                # 超过 20 万帧就丢掉最老的更新批（仍是池化口径，只是窗口被截断）。
                _n_ev = sum(len(p[0]) for p in _probe["ev_pairs"])
                while len(_probe["ev_pairs"]) > 1 and _n_ev > 200000:
                    _n_ev -= len(_probe["ev_pairs"].pop(0)[0])
            # vraw = 未缩放的原始 MSE（与旧日志的 value= 同口径，便于跨版本对比）；
            # value= 在 value_norm=running 时是缩放后的量纲（目标 <10，见 G1）。
            diag = ""
            if stats.get("p_gnorm") is not None:
                _pv = stats["v_gnorm"] / max(1e-12, stats["p_gnorm"])
                diag = (f" | p_gnorm={stats['p_gnorm']:.4g} v_gnorm={stats['v_gnorm']:.4g}"
                        f" v/p={_pv:.2f} scale={stats['value_scale']:.4g}")
            # 注意：单步日志里的 EVb 是**批内口径**（128 连续帧，因批内低方差被放大
            # ~3 倍），只用于实时观察趋势；判读用评估行的池化 EV（v3 P0-B）。
            # F'（2026-09-12）：EVb 现在是**更新前**口径（与历史可比）；
            # EVin 是末轮 in-sample 口径（在那批上训过 12 步后的读数），
            # 两者的差就是"过拟合到本批"的量。
            _ev_in = stats.get("explained_variance_insample")
            _ev_in_s = f" EVin={_ev_in:+.3f}" if _ev_in is not None else ""
            print(f"[solo step {step}] policy={stats['policy_loss']:.4f} "
                  f"value={stats['value_loss']:.4f} vraw={stats['value_loss_raw']:.2f} "
                  f"EVb={stats['explained_variance']:+.3f}{_ev_in_s} "
                  f"entropy={stats['entropy']:.4f} "
                  f"| deploy={100.0 * n_play / len(batch):.1f}% bundle={avg_size:.2f} "
                  f"ratio={stats['ratio_mean']:.3f} clip={100.0 * stats['clip_frac']:.1f}% "
                  f"gs={stats.get('grad_steps', 0)} "
                  f"adv={stats['adv_mean']:+.3f}±{stats['adv_std']:.3f} "
                  f"gnorm={stats['grad_norm']:.2f}{diag} n={len(batch)}", flush=True)

        # 周期评估 **先于** 冻结副本同步（P0-2c，2026-09-11 判读整改）：
        # 旧顺序是"同步 → 评估"，两者步长整除时（copy_every=2000 / steps_per_eval=8000）
        # 评估对手恒为**刚同步的 main 自己** → main 曲线是镜像局、期望恒 0.5、
        # 结构性无信息量（20k 判读 §4 "main 曲线结构性无意义"）。
        # 改成先评估后同步：评估对手 = 上一个同步点的副本（step-N），是非镜像局，
        # 曲线才反映"相对 N 步前自己的进步"。录像 meta.steps 的 frozen_step 同步修正。
        if cfg.steps_per_eval and step % cfg.steps_per_eval == 0:
            eval_and_write(step)
            last_eval_step = step
        elif (cfg.anchor_every and step % cfg.anchor_every == 0
              and step != cfg.total_steps):
            # C 方案轻量点：只跑锚点（全点已含锚点，故用 elif；末步交给收尾全点）
            anchor_point(step)

        # 周期同步冻结副本（原版 WeightsCopyingCallback 思路）
        if cfg.solo_copy_every and step % cfg.solo_copy_every == 0:
            _sync_frozen_copy(main, opp)
            frozen_step = step
            print(f"[solo] 冻结副本已同步 @step {step}", flush=True)
            # D1：同步冻结副本的同时重扫本 run 目录 —— 让"历史自身"进 hist 联赛池
            # （原来只在 __init__ 扫一次，--fresh 时本目录为空 ⇒ 全程只用外部旧 ckpt）。
            if opp_pool is not None:
                opp_pool.refresh_hist(step)

    print(f"[solo] 训练循环耗时 {time.monotonic() - _t0:.1f}s", flush=True)
    save_checkpoint(main, cfg.solo_main_path())
    if last_eval_step != cfg.total_steps:
        eval_and_write(cfg.total_steps)
    print(f"[done] solo '{cfg.name}' 完成，产物在 {cfg.folder()} "
          f"（solo_state.json 供 dashboard --solo 读取）")


if __name__ == "__main__":
    from rl.config import TrainConfig
    run_solo(TrainConfig.resolve("standard"))
