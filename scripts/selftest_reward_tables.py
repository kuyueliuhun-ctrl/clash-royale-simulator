# -*- coding: utf-8 -*-
"""reward 常量表一致性回归（【R7】：尺度改动必须单常量源 + 对账 selftest）。

背景（2026-09-19 复核）：
- `rl/config.py::DEFAULT_REWARD` = **22 键**（`config.py:37-68`）——训练/评估路径经
  `reward_to_env()`（`config.py:509`）把整表传进 `RLEnv`；
- `rl/reward.py::_DEFAULT_REWARD` = **16 键**（`reward.py:28-46`）——`compute_reward()` 的基表
  （`reward.py:197` `rw = dict(_DEFAULT_REWARD, **(rw or {}))`）；
- 差集**恰好 6** = {`engagement_trade`, `engagement_trade_theta`, `engagement_trade_t_ref`,
  `engagement_trade_gate`, `engagement_trade_measure_only`（这 5 个是 config-only，设计如此）,
  **`draw_penalty`**（唯一"真实缺口"：`reward.py:199` 用 `.get("draw_penalty", rw["lose_penalty"])` 回退）}；
- ⚠️ 但 `rl/reward.py:25` 的注释写「与 rl/config.DEFAULT_REWARD 保持一致；勿单独改一处」——
  **注释与代码不符**，而【R7】要的是"单常量源 + 对账 selftest"，此前**没有可执行的核对**。

本脚本把这条关系变成**可执行断言**，并钉住那个**看起来等价、其实靠回退**的边界：
把 `draw_penalty` 补进基表**会让"自定义 lose_penalty 而不传 draw_penalty"的调用方静默改行为**
（用例 E 就是这件事的 A/B 证据）。

断言：
  A 16 ⊂ 22（逐键包含）
  B 差集 == 白名单（**精确集合相等**，不许悄悄多/少一键）
  C 共有 16 键的值逐键 `==`（字面常量，不用 isclose）
  D A/B：`rw=None`（裸 16 键）与 `rw=config.DEFAULT_REWARD`（22 键）在 win/lose/draw 三种终局下
    `compute_reward` 返回值**完全相同**（= 今天"数值巧合相同"这句话的机器证明）
  E A/B 反例：自定义 `lose_penalty=5.0` 时两者**必须不同**（裸 16 键平局罚 5.0 / 22 键平局罚 10.0）
    ⇒ 证明 D 不是恒等式，而是"恰好相等"；也证明本脚本能看见差异
  F 判别力：在**副本**上删掉 22 键里的一键 ⇒ B 断言必须 FAIL（在副本上做，不改真表）

退出码：0 = 全部成立；1 = 有不成立（= 表被改过，须按【R7】同步两处并更新白名单）。

⚠️ 本脚本**只读**：不改任何表、不构造环境、不训练。
stdout 保持 ASCII-only（本环境 `PYTHONIOENCODING=utf-8` 实测不生效，stdout=gbk）。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src", "clasher_new")
sys.path.insert(0, SRC)

from rl import config as cfgmod  # noqa: E402
from rl import reward as rwmod  # noqa: E402

#: 预期的差集（22 − 16）。**这是一份"陈述现状"的白名单**，不是"应该如此"的规范：
#: 前 5 个是 config-only 的设计选择；`draw_penalty` 是缺口，靠 reward.py:199 的 `.get` 回退掩盖。
EXPECTED_CONFIG_ONLY = {
    "engagement_trade",
    "engagement_trade_theta",
    "engagement_trade_t_ref",
    "engagement_trade_gate",
    "engagement_trade_measure_only",
    "draw_penalty",
}

#: compute_reward 的终局 case：(名称, winner, game_over) —— winner=None 且 game_over=True = 平局
TERMINALS = [("win", 0, True), ("lose", 1, True), ("draw", None, True)]


def _call(rw, winner, game_over):
    """最小合法调用：无塔伤、无费差、无非法动作 ⇒ 只剩终局项（与 draw 回退项）。"""
    return rwmod.compute_reward(
        rw,
        blue_hps_old=1000.0, red_hps_old=1000.0,
        blue_hps_new=1000.0, red_hps_new=1000.0,
        blue_left_old=1, red_left_old=1, blue_left_new=1, red_left_new=1,
        my_elixir_before=5.0, opp_elixir_before=5.0,
        my_elixir_after=5.0, opp_elixir_after=5.0,
        winner=winner, invalid_count=0, game_over=game_over,
    )


def main(argv):
    checks = []
    twenty_two = cfgmod.DEFAULT_REWARD
    sixteen = rwmod._DEFAULT_REWARD

    # A 包含关系
    missing = sorted(set(sixteen) - set(twenty_two))
    checks.append(("A 16 subset of 22", not missing, f"keys_in_16_not_in_22={missing}"))

    # B 差集精确等于白名单
    diff = set(twenty_two) - set(sixteen)
    checks.append(("B diff == whitelist", diff == EXPECTED_CONFIG_ONLY,
                   f"diff={sorted(diff)} expected={sorted(EXPECTED_CONFIG_ONLY)}"
                   f" extra={sorted(diff - EXPECTED_CONFIG_ONLY)}"
                   f" missing={sorted(EXPECTED_CONFIG_ONLY - diff)}"))

    # C 共有键取值逐键相同
    mism = {k: (sixteen[k], twenty_two[k]) for k in sixteen if sixteen[k] != twenty_two[k]}
    checks.append(("C shared 16 keys have equal values", not mism, f"mismatch={mism}"))

    # D 终局 A/B：裸 16 键 vs config 22 键 必须完全相同
    d_bad = []
    for name, w, over in TERMINALS:
        bare = _call(None, w, over)               # rw=None -> _DEFAULT_REWARD 16 键
        full = _call(twenty_two, w, over)         # 22 键（含显式 draw_penalty=10.0）
        if bare != full:
            d_bad.append((name, bare, full))
    checks.append(("D terminal A/B identical (16-bare vs 22-full)", not d_bad,
                   f"diffs={d_bad}"))

    # E 反例：自定义 lose_penalty=5.0 -> 两者必须不同，且差值 = lose_penalty - draw_penalty
    bare5 = dict(sixteen); bare5["lose_penalty"] = 5.0
    full5 = dict(twenty_two); full5["lose_penalty"] = 5.0
    r_bare = _call(bare5, None, True)   # 平局 -> 回退 lose_penalty = 5.0
    r_full = _call(full5, None, True)   # 平局 -> draw_penalty = 10.0
    # ⚠️ 奖励是**惩罚项**（负号），所以 full − bare = −(draw − lose)：
    # 裸 16 键用回退的 lose_penalty=5.0 ⇒ −5.0；22 键用显式 draw_penalty=10.0 ⇒ −10.0。
    expect_gap = -(float(full5["draw_penalty"]) - float(bare5["lose_penalty"]))
    checks.append(("E custom lose_penalty makes A/B differ (not an identity)",
                   abs((r_full - r_bare) - expect_gap) < 1e-9,
                   f"bare={r_bare} full={r_full} gap={r_full - r_bare} expected={expect_gap}"))

    # F 判别力：副本上删一键 -> B 的判据必须翻为 False
    tampered = dict(twenty_two)
    tampered.pop(sorted(EXPECTED_CONFIG_ONLY - {"draw_penalty"})[0])
    tampered_diff = set(tampered) - set(sixteen)
    flipped = (tampered_diff == EXPECTED_CONFIG_ONLY)
    checks.append(("F tampering flips check B (discriminating power)", not flipped,
                   f"tampered_diff={sorted(tampered_diff)}"))

    ok = 0
    for name, passed, detail in checks:
        print(("[PASS] " if passed else "[FAIL] ") + name + "  " + detail)
        ok += bool(passed)
    print(f"[reward-tables] {ok}/{len(checks)} assertions PASS"
          f"  (16 keys={len(sixteen)} / 22 keys={len(twenty_two)} / diff={len(diff)})")
    return 0 if ok == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
