# -*- coding: utf-8 -*-
"""**回归测试**（【R8】）：W2 手牌打分 + W3 前期卡组信息分（预注册
`docs/il_whiff_handscore_prereg_2026-09-22.md` §2.3/§2.4/§3.3）。

被保护的改动（2026-09-22）共四处：
  ① 新增 `rl/hand_score.py`：plan 尾部 17 维确定性特征（纯函数）；
  ② `rl/plan_space.py`：新增 `PLAN_BASE_DIM` / `PLAN_HOLD_OFFSET`（**不变量修复**）；
  ③ `rl/follower.py::_plan_biases`：hold 下标由 `PLAN_DIM-4` 改为 `PLAN_HOLD_OFFSET`；
  ④ `FollowerPolicy(plan_extras_zero=K)`：前向里把 plan 最后 K 列置零（对照臂）。

**两条断言是核心**：
  * **T2**（J-W2.1 不变量）：对 58 维 plan 向量，新旧 hold 读法**逐位同值**；且对 75 维向量
    （58 + 17 追加），偏置必须**只**由前 58 维决定 —— 这正是"尾追加会让 `PLAN_DIM-4` 静默
    指到新特征"这个坑的门禁。修之前 75 维向量的 hold 会读到 extras 的第 10..13 维。
  * **T6**（对照臂有效性）：`plan_extras_zero=17` 时，改动尾 17 列**不得**改变 `enc`；
    但 `plan_extras_zero=4` 时改尾 4 列不得变、改倒数第 5..17 列（W2 段）**必须**变
    ⇒ 证明消融粒度与预定口径一致（否则"对照组"和"实验组"实际是同一个网络）。

用法（仓库根）：
    PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 scripts/selftest_hand_score.py   # PASS -> rc=0
    ./.venv/Scripts/python.exe scripts/selftest_hand_score.py                   # 同上（需 torch）
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src", "clasher_new")
sys.path.insert(0, SRC)
os.chdir(SRC)

from rl.io_bootstrap import force_utf8_stdout  # noqa: E402

force_utf8_stdout()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rl.follower import FollowerPolicy, load_checkpoint, save_checkpoint  # noqa: E402
from rl.plan_space import (PLAN_BASE_DIM, PLAN_DIM, PLAN_HOLD_OFFSET,  # noqa: E402
                           PlanToken)
from rl import hand_score as hs  # noqa: E402

FAILS = []
N_ASSERT = [0]


def check(name, cond, detail=""):
    N_ASSERT[0] += 1
    print(f"[{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


# ---------------------------------------------------------------- 工具

def fake_obs(seed=0):
    """合成一条与真实样本同形的 obs（不需要跑引擎 ⇒ 测试零成本）。

    ⚠️ 必须只填**合法**的通道值：`grid[...,0]` 是实体 id（`entity_emb` 行）、
    `grid[...,3]`（= `rest[...,2]`）是 `card_type`（one_hot num_classes=4）
    —— 随机填满 15 通道会让 `F.one_hot` 直接 `RuntimeError`（本测试第一版踩过）。
    """
    rng = np.random.RandomState(seed)
    g = np.zeros((32, 18, 15), dtype=np.float32)
    g[..., 0] = rng.randint(0, 177, size=(32, 18))       # 实体 id
    g[..., 3] = rng.randint(0, 4, size=(32, 18))         # card_type
    return {
        "grid": g,
        "hand": rng.randint(0, 177, size=(5,)).astype(np.int64),
        "elixir": np.array([5.0], dtype=np.float32),
        "next_card": np.array([3], dtype=np.int32),
        "time": np.array([12.0], dtype=np.float32),
    }


def make_pol(plan_dim, **kw):
    pol = FollowerPolicy(hidden=32, plan_dim=plan_dim, belief_dim=563, **kw)
    pol.eval()
    return pol


def old_slot_bias(pol, v, plan_dim_for_hold):
    """**改动前**的 `_plan_biases` 槽位段**逐字复刻**（hold 用 `plan_dim_for_hold - 4`）。

    `plan_dim_for_hold=58` ⇒ 改动前的真实行为（那是它在 58 维下的读法）；
    `plan_dim_for_hold=75` ⇒ 把「旧代码 `PLAN_DIM-4` 恒为 58」这个事实换成 75
    ⇒ 用来演示**若照旧写 `PLAN_DIM-4` 会在 75 维向量上读到什么**。
    """
    slot_bias = torch.zeros(pol.num_slot_options)
    sug = int(round(float(v[16]) * 4.0)) if v[16] > 0.0 else None
    if sug is not None and 1 <= sug <= 4:
        slot_bias[sug - 1] += 0.8                     # PLAN_CARD_BIAS
    if v.shape[0] >= plan_dim_for_hold:
        for i in range(4):
            if v[plan_dim_for_hold - 4 + i] > 0.5:
                slot_bias[i] -= 2.5                   # PLAN_HOLD_BIAS
    return slot_bias


# ---------------------------------------------------------------- T1 常量

def t1_constants():
    check("T1.1 PLAN_BASE_DIM == 58", PLAN_BASE_DIM == 58, f"实测 {PLAN_BASE_DIM}")
    check("T1.2 PLAN_DIM == PLAN_BASE_DIM（基础布局不被尾部追加改动）",
          PLAN_DIM == PLAN_BASE_DIM, f"PLAN_DIM={PLAN_DIM}")
    check("T1.3 PLAN_HOLD_OFFSET == 54（= 基础布局最后 4 维起点）",
          PLAN_HOLD_OFFSET == 54, f"实测 {PLAN_HOLD_OFFSET}")
    check("T1.4 PLAN_EXTRA_DIM == 17 == 13(W2) + 4(W3)",
          hs.PLAN_EXTRA_DIM == 17 and hs.HAND_SCORE_DIM == 13 and hs.INFO_SCORE_DIM == 4,
          f"{hs.HAND_SCORE_DIM}+{hs.INFO_SCORE_DIM}={hs.PLAN_EXTRA_DIM}")


# ---------------------------------------------------------------- T2 不变量（J-W2.1）

def t2_hold_offset_invariant():
    # 冻结布局 fixture（含 hold_mask=0b1010 ⇒ 槽 2/4）
    tok = PlanToken(macro_intent="save_ace", focus_region="bridge_left", suggested_card=3,
                    bundle_size_hint=2, combo_hint=1, risk_profile=0.25, value_estimate=1.5,
                    target_kind="tower", placement_hint="intercept_mid",
                    opp_spell_threat="fireball", elixir_budget=0.7, hold_mask=0b1010)
    v58 = tok.to_vector()
    check("T2.0 布局 fixture：58 维、hold 位在 54..57 = [0,1,0,1]",
          v58.shape == (58,) and np.array_equal(v58[54:], np.array([0.0, 1.0, 0.0, 1.0],
                                                                   dtype=np.float32)),
          f"tail={v58[54:].tolist()}")

    pol58 = make_pol(58)
    # (a) 新旧读法对 58 维向量**逐位同值**（含 suggested_card 的 +0.8，故比全 6 项）
    new_b, _ = pol58._plan_biases(v58)
    old_b = old_slot_bias(pol58, v58, 58)
    check("T2.1 【核心】58 维向量：新读法 == 旧读法（逐位 torch.equal）",
          torch.equal(new_b, old_b),
          f"new={[round(float(x),4) for x in new_b]} old={[round(float(x),4) for x in old_b]}")
    # T2.2 精确不变量：hold 位的**增量**恰为 −2.5，落在 {2,4}（hold_mask=0b1010）
    v_no_hold = v58.copy()
    v_no_hold[54:58] = 0.0
    delta = pol58._plan_biases(v58)[0] - pol58._plan_biases(v_no_hold)[0]
    check("T2.2 hold 命中槽 = {2,4}，扣减恰为 −2.5，其余槽 Δ=0",
          abs(float(delta[1]) + 2.5) < 1e-9 and abs(float(delta[3]) + 2.5) < 1e-9
          and abs(float(delta[0])) < 1e-9 and abs(float(delta[2])) < 1e-9
          and abs(float(delta[4])) < 1e-9 and abs(float(delta[5])) < 1e-9,
          f"Δslot_bias={[round(float(x),4) for x in delta]}")

    # (b) 75 维（58 + 17）：偏置只由前 58 维决定，**与 extras 无关**
    pol75 = make_pol(75)
    rng = np.random.RandomState(7)
    v75 = np.concatenate([v58, rng.rand(17).astype(np.float32)])
    new_b75, cell75 = pol75._plan_biases(v75)
    check("T2.3 【核心】75 维向量：slot_bias 与 58 维臂**逐位相同**（extras 不泄漏进偏置）",
          torch.equal(new_b75, new_b),
          f"75={[round(float(x),4) for x in new_b75]}")
    # (c) 若按**向量长度**取尾 4 维（75−4=71..74）⇒ 读到 extras、hold 位丢失
    old75 = old_slot_bias(pol75, v75, 75)
    check("T2.4 若按**向量长度**取尾 4 维（71..74）⇒ 读到 extras、hold 位丢失（≠ 正确值）",
          not torch.equal(old75, new_b75),
          f"错读={[round(float(x),4) for x in old75]} vs 正确={[round(float(x),4) for x in new_b75]}")
    # (d) 改 extras 不改任何偏置；改 hold 位才改
    v75b = v75.copy()
    v75b[58:] = 0.0
    b75b, c75b = pol75._plan_biases(v75b)
    check("T2.5 extras 全零 ⇒ slot_bias/cell_bias 与含 extras 时**逐位全等**",
          torch.equal(b75b, new_b75) and torch.equal(c75b, cell75))
    v75c = v75.copy()
    v75c[54:58] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)   # 改 hold 位
    b75c, _ = pol75._plan_biases(v75c)
    check("T2.6 改 hold 位（54..57）⇒ slot 1 被扣 2.5、slot 2/4 归零",
          abs(float(b75c[0]) + 2.5) < 1e-9 and abs(float(b75c[1])) < 1e-9
          and abs(float(b75c[3])) < 1e-9,
          f"{[round(float(x),4) for x in b75c[:4]]}")


# ---------------------------------------------------------------- T3 特征数学

def t3_feature_math():
    # 手牌：Knight(3) / Arrows(3) / Giant(5) / Zap(2)；圣水 4.0 ⇒ 3 张买得起（Giant 差 1 费）
    own = ["Knight", "Arrows", "Giant", "Zap"]
    costs = hs.card_costs(own)
    check("T3.0 费用查表（Card 路径）", costs.tolist() == [3.0, 3.0, 5.0, 2.0],
          f"{costs.tolist()}")
    blk = hs._own_block(costs, 4.0)
    check("T3.1 own 块（圣水 4.0）：n/4=0.75、cheapest/10=0.2、total/40=0.325、mean/10=0.325",
          np.allclose(blk, [0.75, 0.2, 0.325, 0.325]),
          f"{np.round(blk, 6).tolist()}")
    # 对手：同一费用表，后验 p=[0.5,0,0.5,0]（Knight/Giant 各半），圣水估计 4.0
    ob = hs._opp_block(costs, np.array([0.5, 0.0, 0.5, 0.0]), 4.0)
    # E[n_playable] = 0.5·1[3≤4] + 0.5·1[5≤4] = 0.5 ⇒ /4 = 0.125
    # 支持集费用 {3,5} ⇒ cheapest 3 → 0.3 ; E[cost] = 1.5+2.5 = 4 → 0.4 ; P(cost≥5) = 0.5
    check("T3.2 opp 块：E[n]/4=0.125、cheapest/10=0.3、E[cost]/10=0.4、P(big)=0.5",
          np.allclose(ob, [0.125, 0.3, 0.4, 0.5]), f"{np.round(ob, 6).tolist()}")
    # 锁定流（p 是精确 0/1）也要对：只 Knight 在手、圣水 4.0
    ob2 = hs._opp_block(costs, np.array([1.0, 0.0, 0.0, 0.0]), 4.0)
    check("T3.3 opp 块（锁定流 p=one-hot）：E[n]/4=0.25、P(big)=0",
          np.allclose(ob2, [0.25, 0.3, 0.3, 0.0]), f"{np.round(ob2, 6).tolist()}")
    ex = hs._elixir_block(6.0, 4.0)
    check("T3.4 圣水块 3 维", np.allclose(ex, [0.6, 0.4, 0.2]), f"{np.round(ex, 6).tolist()}")
    taf = hs._t_afford_block(costs, 6.0)
    check("T3.5 t_afford：买得起最贵牌(Giant 5) ⇒ 0",
          np.allclose(taf, [0.0]), f"{taf.tolist()}")
    taf2 = hs._t_afford_block(costs, 3.0)
    check("T3.6 t_afford：elixir=3、最贵 5 ⇒ (5−3)/2.8/10 = 0.0714286",
          abs(float(taf2[0]) - 2.0 / hs.ELIXIR_REGEN_S / 10.0) < 1e-12, f"{float(taf2[0]):.7f}")
    info = hs.info_block(known_opp=2, time_s=10.0, n_playable_own=3.0)
    check("T3.7 info 块：unknown=0.75、w=2/3、probe=0.75*(3/4)、unknown*w=0.5",
          np.allclose(info, [0.75, 2.0 / 3.0, 0.75 * 0.75, 0.75 * (2.0 / 3.0)]),
          f"{np.round(info, 6).tolist()}")


# ---------------------------------------------------------------- T4 形状/值域/边界

def t4_shape_range():
    v = hs.plan_extras(["Knight"], 5.0, ["Giant"], np.array([1.0]), 5.0, 3.0, 0)
    check("T4.1 维度 == PLAN_EXTRA_DIM 且 dtype=float32",
          v.shape == (hs.PLAN_EXTRA_DIM,) and v.dtype == np.float32, f"{v.shape} {v.dtype}")
    rng = np.random.RandomState(3)
    deck = ["Knight", "Arrows", "Giant", "Zap", "Fireball", "Musketeer", "Minions", "HogRider"]
    #: 逐维值域（口径见 `rl/hand_score.py` 模块 docstring）：
    #: 索引 8 = S_diff ∈ [−1,1]；索引 11 = (elixir − ê_opp)/10 ∈ [−1,1]；其余 ∈ [0,1]。
    lo = np.zeros(hs.PLAN_EXTRA_DIM)
    hi = np.ones(hs.PLAN_EXTRA_DIM)
    lo[8] = lo[11] = -1.0
    bad = None
    for _ in range(200):
        own = [deck[i] for i in rng.randint(0, 8, size=4)]
        p = rng.rand(8).astype(np.float32)
        p = p / p.sum()
        v = hs.plan_extras(own, float(rng.uniform(0, 10)), deck, p,
                           float(rng.uniform(0, 10)), float(rng.uniform(0, 300)),
                           int(rng.randint(0, 9)))
        if not np.all(np.isfinite(v)):
            bad = "non-finite"
            break
        if np.any(v < lo - 1e-6) or np.any(v > hi + 1e-6):
            bad = f"越界 [{v.min():.4f},{v.max():.4f}]（逐维口径 0/1 或 ±1）"
            break
        if not np.all(v == v.astype(np.float32)):
            bad = "dtype 不一致"
            break
    check("T4.2 200 组随机输入：全有限、逐维值域合法（2 维 ±1、其余 [0,1]）",
          bad is None, bad or "OK")
    check("T4.3 未知卡名不抛异常（保守 0 费）、且进诊断集",
          hs.card_cost("__not_a_card__") == 0.0 and "__not_a_card__" in hs.unknown_cards())
    check("T4.4 空手牌不崩（返回全零块）",
          np.allclose(hs._own_block(np.zeros(0), 5.0), 0.0)
          and np.allclose(hs._t_afford_block(np.zeros(0), 5.0), 0.0))


# ---------------------------------------------------------------- T5 门控（J-W3.1）

def t5_gate():
    def w(t):
        return float(hs.info_block(0, t, 4.0)[1])
    check("T5.1 窗口内 t<30 ⇒ w(t)>0 且随 t 单调降",
          w(0.0) == 1.0 and w(10.0) > 0 and w(20.0) > 0 and w(0.0) > w(10.0) > w(20.0),
          f"w(0)={w(0.0)} w(10)={w(10.0):.4f} w(20)={w(20.0):.4f}")
    check("T5.2 【J-W3.1】窗口外 t>=30 ⇒ w(t) **恒 0**",
          w(30.0) == 0.0 and w(31.0) == 0.0 and w(300.0) == 0.0,
          f"w(30)={w(30.0)} w(300)={w(300.0)}")
    check("T5.3 known_opp 夹到 [0,8]（越界不产生负 unknown）",
          hs.info_block(-5, 0.0, 4.0)[0] == 1.0 and hs.info_block(99, 0.0, 4.0)[0] == 0.0)


# ---------------------------------------------------------------- T6 消融臂有效性

def t6_ablation():
    obs = fake_obs(1)
    tok = np.zeros(563, dtype=np.float32)
    rng = np.random.RandomState(11)
    base = rng.rand(58).astype(np.float32)
    ext = rng.rand(17).astype(np.float32)
    v75 = np.concatenate([base, ext])
    v75_ext2 = np.concatenate([base, rng.rand(17).astype(np.float32)])

    pol_z17 = make_pol(75, plan_extras_zero=17)
    _, e_a = pol_z17._encode_parts(obs, tok, v75)
    _, e_b = pol_z17._encode_parts(obs, tok, v75_ext2)
    check("T6.1 【核心】plan_extras_zero=17 ⇒ 改尾 17 列 **enc 逐位不变**（严格对照臂）",
          torch.equal(e_a, e_b))
    v75_base0 = np.concatenate([base, np.zeros(17, dtype=np.float32)])
    _, e_c = pol_z17._encode_parts(obs, tok, v75_base0)
    check("T6.2 plan_extras_zero=17 时 enc == 尾列置零的 enc（即对照组≡零信息输入）",
          torch.equal(e_a, e_c))

    pol_z4 = make_pol(75, plan_extras_zero=4)
    _, e1 = pol_z4._encode_parts(obs, tok, v75)
    v_tail4 = v75.copy()
    v_tail4[71:75] = rng.rand(4).astype(np.float32)      # 只改最后 4 列（W3 段）
    _, e2 = pol_z4._encode_parts(obs, tok, v_tail4)
    check("T6.3 plan_extras_zero=4 ⇒ 改尾 4 列 enc 不变（W3 被抹）", torch.equal(e1, e2))
    v_mid = v75.copy()
    v_mid[58:71] = rng.rand(13).astype(np.float32)       # 改 W2 段（13 列）
    _, e3 = pol_z4._encode_parts(obs, tok, v_mid)
    check("T6.4 plan_extras_zero=4 ⇒ 改 W2 段（58..70）enc **必须变**（W2 仍在用）",
          not torch.equal(e1, e3))

    pol_z0 = make_pol(75, plan_extras_zero=0)
    _, e4 = pol_z0._encode_parts(obs, tok, v75)
    _, e5 = pol_z0._encode_parts(obs, tok, v_mid)
    check("T6.5 plan_extras_zero=0 ⇒ 改 W2 段 enc 变（完整臂）", not torch.equal(e4, e5))

    check("T6.6 越界参数显式报错（不静默）",
          _raises(ValueError, lambda: make_pol(75, plan_extras_zero=76)))


def _raises(exc, fn):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


# ---------------------------------------------------------------- T7 ckpt 兼容（J-W2.2）

def t7_ckpt_compat():
    import tempfile
    pol58 = make_pol(58)
    with torch.no_grad():
        for p in pol58.parameters():
            p.add_(torch.randn_like(p) * 0.01)
    tmp = os.path.join(tempfile.gettempdir(), "_selftest_handscore_58.pt")
    save_checkpoint(pol58, tmp)
    md = torch.load(tmp, map_location="cpu")
    check("T7.1 ckpt 元数据带 plan_extras_zero（旧 ckpt 无该键 => 0）",
          int(md.get("plan_extras_zero", -1)) == 0, f"{md.get('plan_extras_zero')}")

    pol75 = load_checkpoint(tmp, plan_dim=75)
    check("T7.2 58 维 ckpt 载入 75 维网络**不报错**且 plan_dim=75", pol75.plan_dim == 75)
    w_new = pol75.plan_mlp[0].weight
    w_old = pol58.plan_mlp[0].weight
    check("T7.3 【J-W2.2】新增 17 列**严格 0**", bool((w_new[:, 58:] == 0).all()),
          f"max|new cols|={float(w_new[:, 58:].abs().max()):.3e}")
    check("T7.4 前 58 列**逐位拷贝**",
          torch.equal(w_new[:, :58], w_old))
    check("T7.5 `plan_extras_zero` 不一致时告警（口径错配防线）",
          _warns_on_mismatch(tmp))

    pol_z = load_checkpoint(tmp, plan_dim=75, plan_extras_zero=17)
    check("T7.6 显式覆盖 plan_extras_zero=17 生效", pol_z.plan_extras_zero == 17)
    os.remove(tmp)


def _warns_on_mismatch(path):
    """把 ckpt 元数据改成 17，再按 0 载入 ⇒ 必须打印告警（捕获 stdout）。"""
    import contextlib
    import io
    md = torch.load(path, map_location="cpu")
    md["plan_extras_zero"] = 17
    p2 = path + ".z17"
    torch.save(md, p2)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        load_checkpoint(p2, plan_dim=75, plan_extras_zero=0)
    os.remove(p2)
    return "plan_extras_zero" in buf.getvalue() and "⚠️" in buf.getvalue()


# ---------------------------------------------------------------- T8 冻结布局（R2）

def t8_frozen_layout():
    tok = PlanToken(macro_intent="save_ace", focus_region="bridge_left", suggested_card=3,
                    bundle_size_hint=2, combo_hint=1, risk_profile=0.25, value_estimate=1.5,
                    target_kind="tower", placement_hint="intercept_mid",
                    opp_spell_threat="fireball", elixir_budget=0.7, hold_mask=0b1010)
    v = tok.to_vector()
    check("T8.1 布局冻结：shape=58、sum=12.6051483154、非零位 [11,16..20,33,37,46,48,53,55,57]",
          v.shape == (58,) and abs(float(v.sum()) - 12.6051483154) < 1e-6
          and np.nonzero(v)[0].tolist() == [11, 16, 17, 18, 19, 20, 33, 37, 46, 48, 53, 55, 57],
          f"sum={float(v.sum()):.10f} nz={np.nonzero(v)[0].tolist()}")
    check("T8.2 old_scalars 逐值冻结",
          np.allclose(v[16:21], [0.75, 2.0, 1.0, 0.25, 0.9051482677], atol=1e-9),
          f"{np.round(v[16:21], 10).tolist()}")


# ---------------------------------------------------------------- T9 便捷入口

def t9_from_battle_guard():
    check("T9.1 缺 opp_deck ⇒ 显式 ValueError（不猜卡组）",
          _raises(ValueError, lambda: hs.plan_extras_from_battle(None, 0, None, 0)))


# ---------------------------------------------------------------- T10 推理侧守卫

def t10_readout_guard():
    """`il_readout_games.build_plan_vec` 的 plan_dim 守卫（推理侧 58/75 错配必须显式报错）。"""
    import importlib.util
    path = os.path.join(ROOT, "scripts", "il_readout_games.py")
    spec = importlib.util.spec_from_file_location("_il_readout_guard", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:  # noqa: BLE001
        check("T10.0 能导入 il_readout_games（守卫可测）", False,
              f"{type(e).__name__}: {e}")
        return

    class _Env:
        deck1 = ["Knight", "Arrows", "Giant", "Zap", "Fireball", "Musketeer",
                 "Minions", "HogRider"]

        class battle:
            class players:
                pass

        def __init__(self):
            self.battle = type("B", (), {})()
            self.battle.time = 5.0
            self.battle.players = [type("P", (), {"cycle": ["Knight"] * 8, "elixir": 5.0})()]

    class _BP:
        def plan(self, battle, st, obs):
            class T:
                @staticmethod
                def to_vector():
                    return np.zeros(58, dtype=np.float32)
            return T()

    class _Bel:
        def state(self):
            return type("S", (), {"hand_probs": np.ones(8, dtype=np.float32) / 8,
                                  "elixir_mean": 5.0})()

    v58 = mod.build_plan_vec(_BP(), _Env(), _Bel(), {}, 0, 58)
    check("T10.1 plan_dim=58 ⇒ 原样返回 58 维（旧路径逐位同旧行为）",
          v58.shape == (58,))
    v75 = mod.build_plan_vec(_BP(), _Env(), _Bel(), {}, 0, 75)
    check("T10.2 plan_dim=75 ⇒ 返回 58+17=75 维且尾 17 维非全零（真的算了特征）",
          v75.shape == (75,) and float(np.abs(v75[58:]).sum()) > 0, f"{v75.shape}")
    check("T10.3 plan_dim 既非 58 也非 75（如 66）⇒ 显式 ValueError，拒绝静默继续",
          _raises(ValueError, lambda: mod.build_plan_vec(_BP(), _Env(), _Bel(), {}, 0, 66)))


def main():
    print("=== W2/W3 手牌打分 + 前期卡组信息分 回归测试（预注册 §2.4/§3.3）===")
    t1_constants()
    t2_hold_offset_invariant()
    t3_feature_math()
    t4_shape_range()
    t5_gate()
    t6_ablation()
    t7_ckpt_compat()
    t8_frozen_layout()
    t9_from_battle_guard()
    t10_readout_guard()
    print()
    print(f"=== {N_ASSERT[0] - len(FAILS)}/{N_ASSERT[0]} PASS ===")
    if FAILS:
        print("FAIL 清单：" + ", ".join(FAILS))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
