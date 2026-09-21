"""手牌打分（W2）+ 前期卡组信息分（W3）——注入 `plan` 尾部的确定性特征（【R12】EBK）。

预注册：`docs/il_whiff_handscore_prereg_2026-09-22.md` §2 / §3。
本模块**全是纯函数**（无副作用、无全局状态），`inputs → np.ndarray`，可单测。

为什么走 `plan` 尾部而不是 `scalar` 通道（预注册 §2.3）：
  `scalar` 在 `fused` 的**中段** ⇒ 追加会使 `enc_fc` 入维变化而它**没有尾部拷贝兼容分支**
  ⇒ 旧 ckpt 的 `enc_fc` 会被整体重置（静默），且必须 `--fresh`；
  `plan_mlp.0.weight` **有**"前列拷贝 + 尾零"兼容分支（`rl/follower.py:162-166`）。
本模块因此**只负责产向量**，不改任何布局常量；布局常量在 `rl/plan_space.py`。

口际（**全部 [0,1] 或有界**，逐条可复算）：

| 段 | 维 | 定义 | 出处 |
|---|---|---|---|
| own | 4 | `n_playable/4`、`cheapest/10`、`total_cost/40`、`mean_cost/10` | §2.2 `S_own` |
| opp | 4 | 同上四式，但**用后验期望**代替精确集合 | §2.2 `S_opp` |
| diff | 1 | `n_playable_own/4 − E[n_playable_opp]/4` | §2.2 `S_diff` |
| elixir | 3 | `elixir/10`、`ê_opp/10`、`(elixir − ê_opp)/10` | §2.2 `E_*` |
| t_afford | 1 | `max(0, max_cost_own − elixir)/2.8/10`（秒/10） | §2.2 `t_afford` |
| info | 4 | `1−k/8`、`w(t)`、`probe`、`info_unknown·w(t)` | §3.1 |

⇒ `PLAN_EXTRA_DIM = 4+4+1+3+1+4 = 17`。

**两处预注册修正**（跑前写死，见预注册 §2.2/§3.1 的「§2.6 跑前修正」）：
  ① `S_own` 第 4 维原文写 `afford_frac = n_playable/4`，与第 1 维**逐值恒等**（线性重复）
     ⇒ 改为 `mean_cost/10`（不同信息，仍有界于 [0.1,1]）；
  ② 信息分第 4 维原文写 `known_opp/8`，与第 1 维 `1 − known_opp/8` **仿射等价**
     ⇒ 改为 `info_unknown · w(t)`（**时间门控后的剩余未知度**，与另三维不共线）。

⚠️ 口径诚实标注（【R10】）：`S_opp` 的输入 `hand_probs` **本来就是 belief token 的 [0:8]**
（`rl/belief.py:348`）⇒ 这 4 维是 belief token 的**确定性再编码**（EBK 免去网络自己学阈值），
**不是**新增信息源；真正不在任何既有输入里的是「**cost × elixir 的合取**」
（手牌成本要从 `entity_emb` 里间接学，圣水门槛要从标量里学）。同理 `own` 段的原始量
（手牌 id + 圣水）都在观测里，新增的是**合取后的可用牌数**。
"""

import functools

import numpy as np

#: 本批 plan 尾部追加的总维数（W2 13 + W3 4）。
PLAN_EXTRA_DIM = 17
HAND_SCORE_DIM = 13
INFO_SCORE_DIM = 4

#: 归一化常量（【R7】单常量源；改这里即改所有特征，禁止在调用点写魔法数）
MAX_ELIXIR = 10.0
COST_NORM = 10.0
TOTAL_COST_NORM = 40.0          # 4 张 × 单卡最高 10
ELIXIR_REGEN_S = 2.8            # 与 `rl/belief.py:298` 同一约定（引擎分期回费 ≠ 此值，见预注册 O-d）
BIG_CARD_COST = 5.0             # 「大牌」阈值（Xbow/Poison/Rocket 档）
#: 前期卡组信息分的窗口（秒）——人类开局标定 T=30 s（预注册 §3.1/§3.2）
T_INFO = 30.0
MAX_DECK = 8                    # 卡组张数（known_opp 归一化的分母）


@functools.lru_cache(maxsize=512)
def card_cost(card_name) -> float:
    """卡名 → 圣水费（缓存；`Card` 构造昂贵，逐帧调用不可接受）。未知卡名返回 0.0 并计数。"""
    from card_utils import Card
    try:
        return float(Card(card_name).elixir)
    except Exception:
        _UNKNOWN.add(str(card_name))
        return 0.0


#: 未知卡名（诊断用；正常对局应恒空）
_UNKNOWN = set()


def unknown_cards():
    """返回出现过的未知卡名集合（只读诊断）。"""
    return frozenset(_UNKNOWN)


def card_costs(cards) -> np.ndarray:
    """卡名序列 → float64 费用数组。"""
    return np.array([card_cost(c) for c in cards], dtype=np.float64)


def _own_block(costs: np.ndarray, elixir: float) -> np.ndarray:
    """己方手牌块（4 维，精确可算——手牌完全可观测）。"""
    if costs.size == 0:
        return np.zeros(4, dtype=np.float64)
    n_playable = float((costs <= elixir).sum())
    cheapest = float(costs.min())
    total = float(costs.sum())
    mean = total / float(costs.size)
    return np.array([n_playable / 4.0,
                     cheapest / COST_NORM,
                     total / TOTAL_COST_NORM,
                     mean / COST_NORM], dtype=np.float64)


def _opp_block(costs: np.ndarray, hand_probs, opp_elixir_est: float) -> np.ndarray:
    """对方手牌块（4 维，**后验期望**——对手手牌不可直接观测）。

    `hand_probs[i]` = 第 i 张在对手手牌里的概率（`rl/bayes_filter.py:151`，锁定流精确 0/1）。
    `cheapest` 取「支持集（p>0）里的最小费用」，恒有定义（合法后验支持非空）；
    支持集为空（理论上不发生）时回退 `COST_NORM` = 1.0（最保守）。
    """
    p = np.asarray(hand_probs, dtype=np.float64).reshape(-1)
    if costs.size == 0 or p.size != costs.size:
        return np.zeros(4, dtype=np.float64)
    afford = (costs <= float(opp_elixir_est)).astype(np.float64)
    exp_n_playable = float((p * afford).sum())
    sup = costs[p > 0.0]
    cheapest = float(sup.min()) if sup.size else COST_NORM
    exp_cost = float((p * costs).sum())
    p_big = float((p * (costs >= BIG_CARD_COST)).sum())
    return np.array([exp_n_playable / 4.0,
                     cheapest / COST_NORM,
                     exp_cost / COST_NORM,          # 与 own 的 total/40 同值域 [0.1,1]
                     p_big], dtype=np.float64)


def _elixir_block(elixir: float, opp_elixir_est: float) -> np.ndarray:
    """圣水块（3 维）。对方一项为**规则估计**（非真值，见预注册 O-d）。"""
    return np.array([elixir / MAX_ELIXIR,
                     opp_elixir_est / MAX_ELIXIR,
                     (elixir - opp_elixir_est) / MAX_ELIXIR], dtype=np.float64)


def _t_afford_block(costs: np.ndarray, elixir: float) -> np.ndarray:
    """「还要几拍才买得起最贵手牌」= `max(0, max_cost − elixir)/2.8` 秒，再 /10。"""
    if costs.size == 0:
        return np.zeros(1, dtype=np.float64)
    wait_s = max(0.0, float(costs.max()) - float(elixir)) / ELIXIR_REGEN_S
    return np.array([wait_s / 10.0], dtype=np.float64)


def info_block(known_opp, time_s: float, n_playable_own: float) -> np.ndarray:
    """前期卡组信息分（W3，4 维）。

    `known_opp` = 对手**已打出过的不同卡数**（确定性可读，【R12】）。
    `w(t) = clip(1 − t/T_info, 0, 1)`：窗口内 >0、窗口外**恒 0**（J-W3.1 逐帧可验）。
    `probe = (n_playable_own/4) · info_unknown`：这手牌还能逼出多少未知的代理量。
    """
    k = int(min(MAX_DECK, max(0, int(known_opp))))
    unknown = 1.0 - k / float(MAX_DECK)
    w = float(np.clip(1.0 - float(time_s) / T_INFO, 0.0, 1.0))
    probe = (float(n_playable_own) / 4.0) * unknown
    return np.array([unknown, w, probe, unknown * w], dtype=np.float64)


def plan_extras(own_cards, own_elixir, opp_cards, opp_hand_probs, opp_elixir_est,
                time_s, known_opp) -> np.ndarray:
    """产 `PLAN_EXTRA_DIM` 维特征向量（float32，全部有界）。

    参数
    ----
    own_cards      : 己方可出牌卡名序列（`p.cycle[:4]`）
    own_elixir     : 己方圣水真值（可观测）
    opp_cards      : 对手卡组卡名序列（**顺序**必须与 `opp_hand_probs` 一致，即 `env.deck1`）
    opp_hand_probs : 对手手牌后验（`belief.state().hand_probs`，长度 = len(opp_cards)）
    opp_elixir_est : 对手圣水**估计**（`belief.state().elixir_mean`，非真值）
    time_s         : 对局时间（秒）
    known_opp      : 对手已打出过的不同卡数
    """
    own_costs = card_costs(own_cards)
    opp_costs = card_costs(opp_cards)
    own = _own_block(own_costs, float(own_elixir))
    opp = _opp_block(opp_costs, opp_hand_probs, float(opp_elixir_est))
    diff = np.array([own[0] - opp[0]], dtype=np.float64)
    elx = _elixir_block(float(own_elixir), float(opp_elixir_est))
    taf = _t_afford_block(own_costs, float(own_elixir))
    n_playable_own = float((own_costs <= float(own_elixir)).sum())
    info = info_block(known_opp, float(time_s), n_playable_own)
    vec = np.concatenate([own, opp, diff, elx, taf, info])
    assert vec.shape[0] == PLAN_EXTRA_DIM, (vec.shape, PLAN_EXTRA_DIM)
    return vec.astype(np.float32)


def plan_extras_from_battle(battle, player_id, belief_state, known_opp,
                            opp_deck=None) -> np.ndarray:
    """便捷入口：直接从 `battle` + `BeliefState` 取原料（生产/回放路径共用**唯一实现**）。

    `opp_deck` 缺省用 `belief_state` 没有的卡组 ⇒ 必须显式传（对手卡组顺序 = 后验下标语义）。
    """
    if opp_deck is None:
        raise ValueError("plan_extras_from_battle 需要显式 opp_deck（后验下标 = 卡组位置）")
    me = battle.players[player_id]
    return plan_extras(
        own_cards=list(me.cycle[:4]),
        own_elixir=float(me.elixir),
        opp_cards=list(opp_deck),
        opp_hand_probs=belief_state.hand_probs,
        opp_elixir_est=float(belief_state.elixir_mean),
        time_s=float(battle.time),
        known_opp=int(known_opp),
    )
