# -*- coding: utf-8 -*-
"""奖励数学（Tier 2 · T2-7 从 `rl/env_wrapper.py` **纯搬运**而来，逐字未改）。

**为什么拆**：`env_wrapper.py` 原是 **812 行**且同时承担三件事 —— ① **奖励数学**（本文件）、
② 动作空间 `ActionBundleSpace`、③ `RLEnv` 本体。奖励数学是**纯函数**（零 import、零引擎依赖），
与 `RLEnv` 的状态机耦合为零，是最干净的一刀。

**★ 这一刀的两条纪律（【R2】【R7】）**：
1. **纯搬运、逐字未改**：块内每个字符都与 HEAD 的 `env_wrapper.py:38-264` 逐字节相同
   （已用 `git show HEAD:` 对账）；**没有**动任何权重数值，也**没有**把 `_DEFAULT_REWARD`
   改成从 `rl/config.DEFAULT_REWARD` 派生（那会引入"两处口径何时相等"的新风险 —— 见
   `_DEFAULT_REWARD` 上方那行「与 rl/config.DEFAULT_REWARD 保持一致；勿单独改一处」）。
2. **`env_wrapper` 显式重导出** ⇒ 既有调用方（`rl/flow_league.py` / `rl/mcts.py` /
   `rl/action_mask.py` / `scripts/probe_reward_composition.py` / `scripts/phi_offline_check.py` …）
   继续 `from rl.env_wrapper import compute_reward, _DEFAULT_REWARD, …`，**一行都不用改**。

**验证（不是"我觉得一样"）**：`compute_reward` 是纯函数 ⇒ 可以做**数值逐位 A/B**——
把 HEAD 的同一块独立加载，用固定 seed 生成覆盖各分支的输入网格（per-tower 齐/缺、
`normalize_tower_dmg` 开/关、`elixir_diff_weight` 0/0.5、`winner` None/0/1/`game_over`、
`invalid_count`、自定义 `draw_penalty`），逐例比较返回值**要求精确相等**；
再对 5 个 helper 各自做同样的事。见 `docs/structure_tier2_2026-09-19.md` §11。
"""


#: 默认奖励权重（与 rl/config.DEFAULT_REWARD 保持一致；勿单独改一处）
#: reward v2（2026 重构）：费差=资源账（手牌圣水+场上部署份额）；价格两段离散
#: （120s 切双倍：tower_dmg_late 塔血贵、elixir_diff_late 费贱）；unit_dmg_k 单位受伤 shaping。
_DEFAULT_REWARD = {
    "crown_weight": 8.0,
    "crown_lose_weight": 10.0,   # 被破塔惩罚（> crown_weight：丢塔比破塔更痛，教防守价值）
    "tower_dmg_opp": 0.001,
    "tower_dmg_self": 0.0012,    # 挨打 > 打人（塔伤奖励 0.001 末位加"2"→ 0.0012，防守不对称）
    "tower_dmg_late": 0.002,     # v2 双倍期塔血系数（t≥120；斩杀/法术砸塔自动变正 EV）
    "tower_dmg_self_late": 0.0022,  # 双倍期我方塔损（同上不对称：0.002 末位加"2"）
    "win_bonus": 10.0,
    "lose_penalty": 10.0,
    "invalid_penalty": 0.05,
    "elixir_bonus": 0.0,
    "normalize_tower_dmg": True,   # 塔损按塔血%归一化（默认打开=跨等级一致）
    "elixir_diff_weight": 0.5,     # v2 资源账 edw 前段（t<120：费贵，教珍惜圣水）
    "elixir_diff_late": 0.1,       # v2 双倍期 edw（t≥120：费贱，亏费换塔血可接受）
    "unit_dmg_k": 0.0005,          # v2 单位受伤 shaping（客观伤害：敌方单位每掉 1 血 → 我方 +k）
    "tower_premium_k": 2.0,        # 塔血差异化定价：凹形溢价强度（残血塔单位血价值
                                   # ×(1+k×(1−ratio)²)），双向对称（与 config 同步）
    "king_gate": 0.05,             # 两公主塔都存活时王塔单位血价值系数（≈0，王塔贬值）
}

#: 塔血参考（真实游戏，lv11）：**国王塔对所有人恒定 4824**；四种公主塔血量各不相同。
#: （lv1 基础值来自 gamedata.json items.spells：Tower Princess 1400 / Cannoneer 1200 /
#:  Dagger Duchess 1270 / Royal Chef 1340；lv11 参考值为用户提供的真实游戏数据。）
#: 注意：引擎目前只模拟标准 Tower Princess（3052/4824，即 PlayerState 硬编码值），
#: 其余塔型仅作奖励归一化的参考表；一旦引擎支持塔型，分母自动按本局真实塔血适配。
TOWER_TROOP_HP_LV11 = {
    "PrincessTower": 3052.0,   # 皇家塔公主
    "DaggerDuchess": 2768.0,   # 飞刀女爵
    "RoyalChef": 2703.0,       # 皇家大厨
    "Cannoneer": 2616.0,       # 炮兵
}
KING_TOWER_HP_LV11 = 4824.0


def tower_total_hp(troop_hp: float, king_hp: float) -> float:
    """一方的总塔血 = 国王塔 + 两座公主塔（塔血归一化的分母；lv11 标准塔 = 10928）。"""
    return 2.0 * troop_hp + king_hp


#: 塔血归一化锚点 = 标准 Tower Princess 的 lv11 总塔血 2×3052 + 4824 = 10928。
#: normalize_tower_dmg 用 ``raw_delta * 锚点 / 本局初始总塔血``，分母取本局**真实**塔血
#: （自动适应不同公主塔型与等级）→ 同一"塔血百分比事件"在任何塔型、任何等级给同一奖励，
#: 且 lv11 标准塔下与旧公式逐位一致（见 selftest.test_reward_economy_level_invariance /
#: test_tower_troop_hp_reference）。
_TOWER_HP_ANCHOR = tower_total_hp(TOWER_TROOP_HP_LV11["PrincessTower"], KING_TOWER_HP_LV11)

#: reward v2 价格分段切换点：引擎圣水回复 120s 切双倍（<120 正常 2.8s/点；>=120 双倍 1.4s/点）。
#: 价格只做**两段离散值**（不线性）：前段费贵塔血便宜（教珍惜圣水），后段费贱塔血贵
#: （双倍期亏费换塔血/法术砸塔自动变正 EV，行为随价格涌现，不手工规定"何时该砸塔"）。
PHASE_SWITCH_S = 120.0


#: —— 塔血差异化定价（low-tower premium，2026-09-10 用户定稿）——
#: 塔的单位血价值随残血上升：凹形溢价 w×(1 + k×(1−ratio)²)，双向对称（打敌方低血塔
#: 奖励涨、我方低血塔挨打惩罚同曲线放大）；国王塔在两座公主塔都存活时价值≈0（×king_gate）。
#: 同一机制必须与 MCTS 值函数（mcts.node_value）和 belief_planner 法术估值同源——
#: 搜索/规划器与训练奖励量纲失配会重蹈 AGENTS.md 9h/MCTS 教训。
DEFAULT_TOWER_PREMIUM_K = 2.0     # 凹形溢价强度（残血塔最多 ×(1+k)=3 倍单位血价值）
DEFAULT_KING_GATE = 0.05          # 两公主塔存活时王塔单位血价值系数（≈0）


def tower_value_mult(hp_ratio: float, *, king: bool, princesses_alive: int,
                     k: float = DEFAULT_TOWER_PREMIUM_K,
                     king_gate: float = DEFAULT_KING_GATE) -> float:
    """单位血价值倍数：凹形溢价 + 王塔贬值闸门。

    hp_ratio = 当前血/满血（0..1，塔破后按 0 计）；princesses_alive = 存活公主塔数
    （调用方用 PlayerState.left/right_tower_hp>0 数出，或实体 is_alive）。
    返回 0..(1+k)；全满血 + 非王塔 = 1.0（旧行为逐位不变）。
    凹形（二次）：中高血量几乎不涨价、残血塔暴涨——模型学"斩杀残血塔"而非"乱放塔血买卖"。
    王塔闸门：两公主塔都存活 → 王塔单位血价值×king_gate（未激活王塔不反击≈无价值）；
    任一座公主塔被破后恢复全价。塔已破（ratio<=0）→ 返回 1.0（无边际伤害，倍数无意义）。
    """
    if hp_ratio <= 0.0:
        return 1.0
    base = 1.0 + k * (1.0 - hp_ratio) ** 2
    if king and princesses_alive >= 2:
        base *= king_gate
    return base


def tower_premium_k(rw) -> float:
    """从奖励字典取溢价强度（缺省 2.0；经 reward_to_env 透传可调）。"""
    return float(rw.get("tower_premium_k", DEFAULT_TOWER_PREMIUM_K))


def _princesses_alive(hp_ratio_king, hp_ratio_left, hp_ratio_right) -> int:
    """存活公主塔数（血比例 >0 = 未破）。"""
    return int(hp_ratio_left > 0.0) + int(hp_ratio_right > 0.0)


def _per_tower_norm_dmg(dmg_list, old_list, new_list, max_list, *, king, k, king_gate):
    """per-tower 塔血差 → 差异化后按 lv11 锚归一化的加权伤害（正=塔掉血）。

    dmg_list: 三塔 [king, left, right] 的 旧血−新血 差（正=掉血）；
    old_list/new_list: 三塔事件前后残血；max_list: 三塔满血基准。
    每塔：delta × mult(旧血线/满血) × (该塔 lv11 锚 / max_i)——"同一比例事件跨等级
    同奖励"在**每塔各自**口径下成立（King 锚 4824、Princess 锚 3052）。
    残血比例取**伤害发生时的旧血线** old/max：斩杀残血塔（old 低 → mult 高）最值钱；
    若取 new/max（事件后），破塔帧 new=0 → mult=1.0 会把"斩杀"这个最该值钱的动作
    低估成满血价（test_mcts_basic 4b 实测 kill<wait 回归抓出）。返回三塔加权伤害和。
    """
    anchors = [KING_TOWER_HP_LV11, TOWER_TROOP_HP_LV11["PrincessTower"],
               TOWER_TROOP_HP_LV11["PrincessTower"]]
    total = 0.0
    princesses_alive = _princesses_alive(
        new_list[0] / max_list[0] if max_list[0] > 0 else 0.0,
        new_list[1] / max_list[1] if max_list[1] > 0 else 0.0,
        new_list[2] / max_list[2] if max_list[2] > 0 else 0.0)
    for i in range(3):
        dmg_i = dmg_list[i]
        if dmg_i <= 0.0:
            continue
        m_i = max_list[i]
        if m_i <= 0.0:
            continue
        hp_ratio = old_list[i] / m_i   # 伤害发生时的旧血线（斩杀残血塔最值钱）
        mult = tower_value_mult(hp_ratio, king=(i == 0),
                                princesses_alive=princesses_alive,
                                k=k, king_gate=king_gate)
        total += dmg_i * mult * (anchors[i] / m_i)
    return total


def _phase_weights(rw, battle_time):
    """返回 (tower_opp, tower_self, edw_coef)：120s 切换的两段离散权重。

    tower_opp 作用于打击敌方塔血；tower_self 作用于我方塔损（不对称：挨打 > 打人）；
    edw_coef 作用于资源账 Φ。rw 缺 late 键时回退 base（兼容旧 reward 字典）。
    """
    rw = dict(_DEFAULT_REWARD, **(rw or {}))
    late = float(battle_time) >= PHASE_SWITCH_S
    tower_opp = (float(rw.get("tower_dmg_late", rw["tower_dmg_opp"])) if late
                 else float(rw["tower_dmg_opp"]))
    tower_self = (float(rw.get("tower_dmg_self_late", rw["tower_dmg_late"])) if late
                  else float(rw["tower_dmg_self"]))
    edw = (float(rw.get("elixir_diff_late", rw.get("elixir_diff_weight", 0.0))) if late
           else float(rw.get("elixir_diff_weight", 0.0)))
    return tower_opp, tower_self, edw


def compute_reward(rw, *, blue_hps_old, red_hps_old, blue_hps_new, red_hps_new,
                   blue_left_old, red_left_old, blue_left_new, red_left_new,
                   my_elixir_before, opp_elixir_before, my_elixir_after, opp_elixir_after,
                   my_v_before=0.0, opp_v_before=0.0, my_v_after=0.0, opp_v_after=0.0,
                   winner, invalid_count, blue_hps_max=None, red_hps_max=None,
                   game_over=False, draw_penalty=None,
                   blue_towers_old=None, red_towers_old=None,
                   blue_towers_new=None, red_towers_new=None,
                   blue_towers_max=None, red_towers_max=None):
    """逐决策帧奖励（RLEnv.step 与 selftest 共用）。

    全部新开关关闭（normalize_tower_dmg=False、elixir_diff_weight=0）时与旧公式逐位一致。
    - ``normalize_tower_dmg``：塔损按 本局初始总塔血 归一化到 lv11 锚 → 奖励等级不变，
      修复"等级越高磨血/挨打越值钱、皇冠/胜负不变"的漂移（费差↔塔血校准）；
    - ``elixir_diff_weight``：potential-style shaping，Δ(手牌圣水 + 场上部署费)差 即"资源账"，
      显式给圣水定价，让模型学会"让塔挨打换圣水/费差"这类真实游戏 trade。
      v2 记账式：场上部署费 = 存活部署实体的部署份额（RLEnv 维护，创建即固定、死亡注销，
      不做 HP 折价）。部署帧 E−c 与 V+c 同帧抵消 → 下牌不罚；死亡才注销；法术施放挂账、
      击杀返还对方份额 → 空砸/纯砸塔净亏、解牌赚费差（防双算：spell 产物份额记 0）。
    winner: None=未终局/平局；0=我方胜；其它=负。invalid_count: 非法动作次数。
    game_over: 对局是否已结束（平局判负需要它，避免把进行中的普通步当失败罚）。
    draw_penalty: 平局惩罚（缺省取 rw["draw_penalty"]，再缺省与 lose_penalty 相同）。

    **塔血差异化定价（2026-09-10）**：传 per-tower 参数（blue/red_towers_old/new/max，
    每元素 [king, left, right]）时启用——低血塔单位血价值按 tower_value_mult 凹形溢价
    （双向对称：打敌方低血塔奖励涨、我方低血塔挨打惩罚同曲线放大）、王塔两公主塔存活时
    价值≈0。per-tower 参数**全缺省（None）时完全走旧聚合口径，逐位不变**（旧调用点兼容）。
    """
    rw = dict(_DEFAULT_REWARD, **(rw or {}))
    if draw_penalty is None:
        draw_penalty = rw.get("draw_penalty", rw["lose_penalty"])
    blue_dmg = blue_hps_old - blue_hps_new
    red_dmg = red_hps_old - red_hps_new
    # 塔血差异化定价分支：**逐侧**独立判定——该侧 per-tower 参数齐了才用 per-tower
    # （低血塔溢价 + 王塔贬值），否则回落该侧的聚合归一化（缺省逐位不变，兼容旧调用点）。
    k_prem = tower_premium_k(rw)
    if rw.get("normalize_tower_dmg"):
        if (red_towers_old is not None and red_towers_new is not None
                and red_towers_max is not None):
            red_dmg = _per_tower_norm_dmg(
                [red_towers_old[0] - red_towers_new[0],
                 red_towers_old[1] - red_towers_new[1],
                 red_towers_old[2] - red_towers_new[2]],
                red_towers_old, red_towers_new, red_towers_max,
                king=True, k=k_prem, king_gate=rw.get("king_gate", DEFAULT_KING_GATE))
        elif red_hps_max:
            red_dmg = red_dmg * (_TOWER_HP_ANCHOR / red_hps_max)
        if (blue_towers_old is not None and blue_towers_new is not None
                and blue_towers_max is not None):
            blue_dmg = _per_tower_norm_dmg(
                [blue_towers_old[0] - blue_towers_new[0],
                 blue_towers_old[1] - blue_towers_new[1],
                 blue_towers_old[2] - blue_towers_new[2]],
                blue_towers_old, blue_towers_new, blue_towers_max,
                king=True, k=k_prem, king_gate=rw.get("king_gate", DEFAULT_KING_GATE))
        elif blue_hps_max:
            blue_dmg = blue_dmg * (_TOWER_HP_ANCHOR / blue_hps_max)
    reward = (
        rw["crown_weight"] * (red_left_old - red_left_new)
        - rw.get("crown_lose_weight", rw["crown_weight"]) * (blue_left_old - blue_left_new)
        + rw["tower_dmg_opp"] * red_dmg
        - rw["tower_dmg_self"] * blue_dmg
        + rw["elixir_bonus"] * my_elixir_after
    )
    edw = rw.get("elixir_diff_weight") or 0.0
    if edw:
        # 资源账（v2 记账式）：potential Φ = (手牌圣水 + 场上部署费) 差；
        # 未传 v_*（纯函数/旧测试）时退化为纯手牌圣水差（V≡0），行为不变。
        me_b = my_elixir_before + float(my_v_before)
        op_b = opp_elixir_before + float(opp_v_before)
        me_a = my_elixir_after + float(my_v_after)
        op_a = opp_elixir_after + float(opp_v_after)
        reward += edw * ((me_a - op_a) - (me_b - op_b))
    if winner == 0:
        reward += rw["win_bonus"]
    elif winner is not None:
        reward -= rw["lose_penalty"]
    elif game_over:
        # 对局结束且无胜者 = 平局 → 按失败惩罚（平局不再免费，"躺平即最优"被消除）
        reward -= float(draw_penalty)
    if invalid_count:
        reward -= rw["invalid_penalty"] * invalid_count
    return reward
