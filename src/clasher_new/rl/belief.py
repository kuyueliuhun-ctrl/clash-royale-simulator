"""信念推断：规则(贝叶斯粒子滤波) + 统计倾向 + 神经编码（规划文档 3.6 / 10.3）。

- RuleBelief：对手循环牌序的粒子滤波；
- StatisticalBelief：对手风格/路线倾向的计数统计；
- NeuralBeliefEncoder：GRU 压缩历史 → belief_token（torch 惰性导入，避免强依赖）。
- BeliefInference：三者组合，对外提供 BeliefState 与 belief_token。

对 ``opp_played`` 的输入契约（docs/rl_review_fix_plan.md §5）：
- 可以是单张卡名 str；
- 也可以是结构化列表 ``[{"card": name, "x": x, "y": y}, ...]``（env info["opp_played"]）；
- 非卡名哨兵（如 ``"__ability__"``、None）在入口统一过滤，不进粒子滤波/统计，
  避免 KeyError 与信念重置（P0-6）。
"""

import os
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import numpy as np

from rl.bayes_filter import CycleBayesFilter
from rl.plan_space import MACRO_INTENTS
from rl.observation import ENTITY_NAMES

NUM_CARDS = len(ENTITY_NAMES)
TENDENCIES = ["aggressive", "defensive", "cycle", "spell_heavy", "balanced"]

#: 无神经编码时的信念 token 维度（rule hand + rule next + elixir2 + tendency
#: + 事件通道 OPP_EVENT_K×OPP_EVENT_DIM）
def belief_token_dim(deck) -> int:
    return 2 * len(deck) + 2 + len(TENDENCIES) \
        + OPP_EVENT_K * OPP_EVENT_DIM


# —— B 层：对手出牌事件通道（2026-09-08）——
# belief_token 原来只有"概率分布"（手牌后验/风格计数），没有"事件"：对手 0.5s 前在
# 桥头下了 Giant 这类即时信号要靠 CNN 从 grid 里重新发现（有效感受野 ~9 格，
# 常与"我该在哪布防"落在不同感受野）。事件 token 把最近 k 次出牌直接喂给策略
# （FirstLight relation-attention 的最低成本替身：先给"谁/在哪/多久前"）。
#: 事件窗口长度（最近 k 条对手出牌）
OPP_EVENT_K = 3
#: 每条事件维度：card_onehot(len(ENTITY_NAMES)) + x/17 + y/31 + Δt/10
OPP_EVENT_DIM = len(ENTITY_NAMES) + 3


def _event_row(card, x, y, dt) -> np.ndarray:
    """单条事件 → 16 维行向量。"""
    row = np.zeros(OPP_EVENT_DIM, dtype=np.float32)
    if card in ENTITY_NAMES:
        row[ENTITY_NAMES.index(card)] = 1.0
    row[len(ENTITY_NAMES)] = (x / 17.0) if x is not None else 0.0
    row[len(ENTITY_NAMES) + 1] = (y / 31.0) if y is not None else 0.0
    row[len(ENTITY_NAMES) + 2] = min(max(dt, 0.0), 10.0) / 10.0
    return row


def opp_event_token(history, now=None, k: int = OPP_EVENT_K) -> np.ndarray:
    """事件历史 [(card, x, y, t), ...]（旧→新）→ 展平事件向量。

    Δt 以当前决策时刻 ``now`` 为基准（无新事件时旧事件 Δt 增大 → "事件陈旧度"
    可学习）；now 缺省取最后一条事件时刻（维度探测场景）。不足 k 条前补全零行
    （零行 = "无事件"）。"""
    out = np.zeros(k * OPP_EVENT_DIM, dtype=np.float32)
    recent = list(history)[-k:]
    if not recent:
        return out
    if now is None:
        now = float(recent[-1][3])
    pad = k - len(recent)
    for i, (card, x, y, t) in enumerate(recent):
        out[(pad + i) * OPP_EVENT_DIM:(pad + i + 1) * OPP_EVENT_DIM] = \
            _event_row(card, x, y, now - float(t))
    return out


def normalize_played(opp_played):
    """把 opp_played 规范化为 [(card_name, x, y), ...]（哨兵/非卡名已过滤）。

    - None → []
    - str → 单卡
    - list[dict | str] → 逐条

    ``"None"``（观测层未知卡的占位 id 0）与 ``"__ability__"`` 一样被过滤，
    避免 ``Card("None")`` KeyError（随机卡组模型会打出观测层未知的卡）。
    """
    out = []
    if opp_played is None:
        return out
    items = opp_played if isinstance(opp_played, (list, tuple)) else [opp_played]
    for it in items:
        if isinstance(it, dict):
            card, x, y = it.get("card"), it.get("x"), it.get("y")
        else:
            card, x, y = it, None, None
        if card is None or card == "None" or card not in ENTITY_NAMES:
            continue
        out.append((card, x, y))
    return out


@dataclass
class BeliefState:
    deck: list = field(default_factory=list)
    hand_probs: np.ndarray = None          # (len(deck),) 每张卡在对手手牌的概率
    next_probs: np.ndarray = None          # (len(deck),) 每张卡是对手下一张的概率
    elixir_mean: float = 5.0
    elixir_std: float = 1.0
    intent_probs: np.ndarray = None        # (len(MACRO_INTENTS),)
    tendency_probs: np.ndarray = None      # (len(TENDENCIES),)
    uncertainty: float = 1.0               # 规则信念熵的归一化值

    def normalize(self):
        if self.hand_probs is None or len(self.hand_probs) == 0:
            self.hand_probs = np.ones(len(self.deck), dtype=np.float32) / max(1, len(self.deck))
        if self.next_probs is None or len(self.next_probs) == 0:
            self.next_probs = np.ones(len(self.deck), dtype=np.float32) / max(1, len(self.deck))
        if self.intent_probs is None:
            self.intent_probs = np.ones(len(MACRO_INTENTS), dtype=np.float32) / len(MACRO_INTENTS)
        if self.tendency_probs is None:
            self.tendency_probs = np.ones(len(TENDENCIES), dtype=np.float32) / len(TENDENCIES)
        return self


class StatisticalBelief:
    """对手风格/路线倾向的轻量统计信念（P1-2：真正记录 side_counts / tendency_counts）。"""

    def __init__(self):
        self.tendency_counts = np.zeros(len(TENDENCIES), dtype=np.float32)
        self.side_counts = np.zeros(2, dtype=np.float32)  # 左/右
        self.push_back_count = 0
        self.spell_count = 0
        self.total = 0

    def update(self, card: str, x=None, y=None):
        """card 已由 normalize_played 过滤（在 ENTITY_NAMES 内）。"""
        from card_utils import Card
        self.total += 1
        ctype = Card(card).type if card else None
        if ctype == "spell":
            self.spell_count += 1
            self.tendency_counts[TENDENCIES.index("spell_heavy")] += 1
        elif ctype == "building":
            self.tendency_counts[TENDENCIES.index("defensive")] += 1
        if x is not None:
            self.side_counts[0 if x < 9 else 1] += 1
        # 落点半场：世界坐标 y 大的一侧是 player1 的本方半场
        if y is not None and ctype in ("troop", "character"):
            if y >= 16.0:
                self.tendency_counts[TENDENCIES.index("defensive")] += 1
                self.push_back_count += 1
            else:
                self.tendency_counts[TENDENCIES.index("aggressive")] += 1

    def probs(self) -> np.ndarray:
        if self.total == 0:
            return np.ones(len(TENDENCIES), dtype=np.float32) / len(TENDENCIES)
        t = self.tendency_counts + 1.0
        return (t / t.sum()).astype(np.float32)


class NeuralBeliefEncoder:
    """神经信念编码器：GRU 压缩历史观测 → belief_token + 下一张牌预测头。

    torch 惰性导入：模块未安装 torch 时也可 import 本包。
    """

    def __init__(self, in_dim: int, hidden: int = 64, num_classes: int = NUM_CARDS,
                 hand_dim: int = 8, max_len: int = 32):
        import torch
        import torch.nn as nn

        self.hidden_dim = hidden
        self.num_classes = num_classes
        self.hand_dim = hand_dim
        self.max_len = max_len
        self._torch = torch
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.next_head = nn.Linear(hidden, num_classes)
        self.hand_head = nn.Linear(hidden, hand_dim)
        self.belief_proj = nn.Linear(hidden, hidden)
        self._hist = deque(maxlen=max_len)

    def reset_history(self):
        self._hist.clear()

    def push_frame(self, feat: np.ndarray):
        self._hist.append(np.asarray(feat, dtype=np.float32))

    def _history_tensor(self):
        torch = self._torch
        if not self._hist:
            seq = torch.zeros(1, 1, self.gru.input_size)
        else:
            arr = np.stack(list(self._hist), axis=0)[None, ...]  # (1, T, D)
            seq = torch.from_numpy(arr)
        return seq

    def encode(self, feat: np.ndarray) -> np.ndarray:
        """输入当前帧特征 → belief_token (hidden,)"""
        torch = self._torch
        self.push_frame(feat)
        seq = self._history_tensor()
        with torch.no_grad():
            self.gru.eval()
            _, h = self.gru(seq)
            token = self.belief_proj(h[-1])  # (1, hidden)
        return token.squeeze(0).numpy().astype(np.float32)

    @classmethod
    def load(cls, path, in_dim=None):
        """从 train_belief 保存的 checkpoint 加载编码器（P1-1 训练产物消费）。"""
        import torch
        data = torch.load(path, map_location="cpu")
        enc = cls(in_dim=int(in_dim or data["in_dim"]),
                  hidden=int(data.get("hidden", 64)),
                  num_classes=int(data.get("num_classes", NUM_CARDS)),
                  hand_dim=int(data.get("hand_dim", 8)))
        enc.gru.load_state_dict(data["gru"])
        enc.next_head.load_state_dict(data["next_head"])
        enc.hand_head.load_state_dict(data["hand_head"])
        enc.belief_proj.load_state_dict(data["belief_proj"])
        enc.gru.eval()
        return enc


def build_feature(obs: dict, opp_played) -> np.ndarray:
    """把一帧观测压缩成神经编码器输入向量。"""
    hand = obs["hand"]  # (5,)
    hand_oh = np.zeros(NUM_CARDS * 5, dtype=np.float32)
    for i, c in enumerate(hand):
        if 0 <= c < NUM_CARDS:
            hand_oh[i * NUM_CARDS + c] = 1.0
    next_oh = np.zeros(NUM_CARDS, dtype=np.float32)
    nc = int(obs["next_card"][0])
    if 0 <= nc < NUM_CARDS:
        next_oh[nc] = 1.0
    played_oh = np.zeros(NUM_CARDS, dtype=np.float32)
    played = normalize_played(opp_played)
    if played:
        for card, _, _ in played:
            played_oh[ENTITY_NAMES.index(card)] = 1.0
    time = float(obs["time"][0]) if not np.isscalar(obs["time"]) else float(obs["time"])
    return np.concatenate([
        hand_oh,
        np.asarray(obs["elixir"], dtype=np.float32),
        np.asarray([time / 180.0], dtype=np.float32),
        next_oh,
        played_oh,
    ]).astype(np.float32)


class BeliefInference:
    """组合信念模块：规则 + 统计 + 神经。"""

    def __init__(self, opp_deck, use_rule=True, use_stat=True, neural=None, n_particles=128, seed=0):
        self.deck = list(opp_deck)
        self.use_rule = use_rule
        self.use_stat = use_stat
        self.rule = CycleBayesFilter(self.deck, n_particles=n_particles, seed=seed) if use_rule else None
        self.stat = StatisticalBelief() if use_stat else None
        self.neural = neural
        self._elixir_est = 5.0
        self._last_time = None
        self._now_cache = 0.0
        # B 层：对手出牌事件历史 [(card, x, y, t), ...]（旧→新），encode 时取尾部
        self.event_history = deque(maxlen=16)

    def reset(self, opp_deck=None):
        if opp_deck is not None:
            self.deck = list(opp_deck)
        if self.rule is not None:
            self.rule.reset(self.deck)
        if self.stat is not None:
            self.stat = StatisticalBelief()
        if self.neural is not None:
            self.neural.reset_history()
        self._elixir_est = 5.0
        self._last_time = None
        self._now_cache = 0.0
        self.event_history.clear()

    def _tick_elixir(self, obs):
        """按观测时间推进圣水估计（决策间隔回复率近似 2.8s/点）。"""
        if obs is None:
            return
        t = float(obs["time"][0]) if not np.isscalar(obs["time"]) else float(obs["time"])
        if self._last_time is None:
            self._last_time = t
            return
        dt = max(0.0, t - self._last_time)
        self._last_time = t
        self._elixir_est = min(10.0, self._elixir_est + dt / 2.8)

    def update(self, obs, opp_played, opp_x=None, opp_card_type=None, elixir_est=None):
        """用本 tick 已观测的对手出牌更新信念（支持结构化多卡列表，P1-5）。"""
        self._tick_elixir(obs)
        played = normalize_played(opp_played)
        # B 层：事件入历史（带观测时刻；obs 不可用时 _now 尾推近似，Δt 仍有效）
        t_now = self._now(obs)
        for card, x, y in played:
            self.event_history.append((card, x, y, t_now))
        for card, x, y in played:
            if self.rule is not None:
                self.rule.update(card)
            if self.stat is not None:
                self.stat.update(card, x=x if opp_x is None else opp_x, y=y)
            from card_utils import Card
            self._elixir_est -= Card(card).elixir
        self._elixir_est = float(np.clip(self._elixir_est, 0.0, 10.0))
        return self.state()

    def state(self) -> BeliefState:
        st = BeliefState(deck=self.deck)
        if self.rule is not None:
            st.hand_probs = self.rule.hand_probs()
            st.next_probs = self.rule.next_probs()
            st.uncertainty = float(np.clip(self.rule.entropy() / np.log(8), 0.0, 1.0))
        if self.stat is not None:
            st.tendency_probs = self.stat.probs()
        st.elixir_mean = self._elixir_est
        st.elixir_std = max(0.5, abs(self._elixir_est - 5.0) / 5.0)
        return st.normalize()

    def _now(self, obs):
        """当前决策时刻（obs.time 优先；obs 缺失时用"上次 _now 的值"）。

        fallback 语义：事件 t 取自上次推算的 now（首个事件前 now=0）——同一帧的
        update/encode 得到同一 t；跨帧调用 _now 不再无中生有 +0.5，Δt 只在
        obs.time 缺失的降级场景里保持 0（事件陈旧度信号退化，可用性优先）。"""
        if obs is not None and not np.isscalar(obs) and isinstance(obs, dict) \
                and obs.get("time") is not None and hasattr(obs["time"], "__len__") \
                and len(obs["time"]):
            self._now_cache = float(obs["time"][0])
        elif isinstance(obs, (int, float)):
            self._now_cache = float(obs)
        return getattr(self, "_now_cache", 0.0)

    def encode(self, obs=None, opp_played=None) -> np.ndarray:
        """belief_token：规则向量 + 统计向量 + 事件通道 + 神经 token。"""
        parts = []
        st = self.state()
        parts.append(st.hand_probs.astype(np.float32))
        parts.append(st.next_probs.astype(np.float32))
        parts.append(np.array([st.elixir_mean, st.uncertainty], dtype=np.float32))
        parts.append(st.tendency_probs.astype(np.float32))
        # B 层：事件通道（update 已入历史；这里若显式传入 opp_played 先补录——
        # 兼容"先 encode 后 update"的调用序，避免本帧事件丢窗口）
        if opp_played:
            played = normalize_played(opp_played)
            for card, x, y in played:
                self.event_history.append((card, x, y, self._now(obs)))
        parts.append(opp_event_token(self.event_history, now=self._now(obs)))
        if self.neural is not None and obs is not None:
            feat = build_feature(obs, opp_played)
            parts.append(self.neural.encode(feat).astype(np.float32))
        return np.concatenate(parts).astype(np.float32)
