# -*- coding: utf-8 -*-
"""预注册 §7-7 `test_engagement_trade_default_off`（**引擎侧**）：默认关 ⇒ 逐位回旧。

预注册原文：*"默认关逐位回旧"*。本文件把它做成**可复跑的回归测试**，三段：

1. **off ⇒ 逐位回旧**：用 `DEFAULT_REWARD`（不含 `engagement_trade` 或为 0）跑一段**确定性脚本局**，
   逐决策帧奖励 + 全实体状态摘要的 SHA-256 必须等于**打补丁之前**记录的黄金值
   （`GOLDEN_OFF_DIGEST`，2026-09-18 由**未打补丁的真树**实测得到；这就是 9j 式指纹的用法）。
2. **off ⇒ 零开销**：`env._et is None`（不构造监视器、不跑 tick）。
3. **on ⇒ 项确实进奖励**：把 `theta` 压到 0（保证 `score > 0`），digest **必须改变**
   —— 否则说明这条链根本没接上（"关时相同"会是**假通过**）。

跑法（必须 cwd = ``src/clasher_new``）：
    cd src/clasher_new && PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe \
        ../../scripts/selftest_engagement_trade_online.py
"""
from __future__ import annotations

import hashlib
import os
import sys
import traceback

#: 脚本目录不是包目录；`rl.*` 要从**当前工作目录**（= src/clasher_new）解析
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

#: 打补丁**之前**的真树实测值（2026-09-18；工具 `scripts/s2_trade_probe.py` 的同一段脚本局）
GOLDEN_OFF_DIGEST = "4d8b27b839af7bbddd7af754ee383aa7540bf0b96d3188b1f475bf5b448eca0c"

DECK = ['Tombstone', 'Witch', 'Skeletons', 'Giant', 'Minions', 'Fireball',
        'Knight', 'Archer']


def _run_scripted(reward_weights, *, steps=60):
    """跑一段确定性脚本局，返回 (digest, reward_sum, n_settled)。"""
    from rl.env_wrapper import RLEnv
    from rl.action_bundle import ActionBundle, SubAction
    from rl.observation import ENTITY_NAMES

    env = RLEnv(deck0=list(DECK), deck1=list(DECK), seed=12345, speed=1.0,
                decision_frames=30, dt=1 / 60, record_hidden=False,
                reward_weights=reward_weights)
    obs, _ = env.reset(seed=12345)

    def slot_of(card):
        for i in range(5):
            if ENTITY_NAMES[int(obs["hand"][i])] == card:
                return i + 1
        return 0

    plan = []
    for k in range(10):
        plan += [(3 * k + 0, 'Giant', (9.0, 12.0 + 0.2 * k)),
                 (3 * k + 1, 'Tombstone', (4.0, 19.0 - 0.2 * k)),
                 (3 * k + 2, 'Skeletons', (10.0, 13.5))]
    plan.sort()

    h = hashlib.sha256()
    total = 0.0
    n_detail = 0
    fired = 0
    for step in range(steps):
        sa = []
        while fired < len(plan) and plan[fired][0] <= step:
            _, card, pos = plan[fired]
            s = slot_of(card)
            if s:
                sa.append(SubAction(kind="deploy", slot=s, x=pos[0], y=pos[1]))
            fired += 1
        bundle = ActionBundle(sa) if sa else ActionBundle.noop()
        obs, reward, term, trunc, _info = env.step(bundle)
        _d = _info.get("engagement_trade_detail")
        if _d:
            n_detail += int(_d[3])
        total += float(reward)
        h.update(("%.9f;" % reward).encode())
        for eid in sorted(env.battle.entities):
            e = env.battle.entities[eid]
            h.update(("%d|%d|%.6f|%.6f|%.6f|%d;"
                      % (eid, int(e.is_alive), e.hp, e.position.x, e.position.y,
                         int(getattr(e, "target_id", -1) or -1))).encode())
        if term or trunc:
            break
    et = getattr(env, "_et", None)
    if et is not None and not env.battle.game_over:
        # 复现"局末 flush"（真路径在 `battle.game_over` 时自动做；见文档 §1 的**已知缺口**：
        # 按 max_ep_steps 截断时 RLEnv 看不到，尾部窗口会丢 —— 本条测试显式补上，
        # 以免把"没 flush"误读成"没接上"）。
        et.flush(env.battle, env._active_v)
    return h.hexdigest(), total, (et.n_settled if et is not None else 0), n_detail


def test_engagement_trade_default_off():
    from rl.config import DEFAULT_REWARD

    # —— 1/2：默认关 ⇒ 逐位回旧 + 零开销 ——
    assert float(DEFAULT_REWARD.get("engagement_trade", 0.0)) == 0.0, \
        "DEFAULT_REWARD 里 engagement_trade 必须是 0.0（默认关）"
    d_off, sum_off, n_off, _nd = _run_scripted(None)     # 缺省 reward ⇒ 旧公式
    assert n_off == 0, "关时不该构造监视器（n_settled=%d）" % n_off
    assert d_off == GOLDEN_OFF_DIGEST, (
        "默认关时没有逐位回旧！\n  期望 %s\n  实得 %s\n"
        "（若这是**有意**的行为改动，必须同时改 GOLDEN 并在文档里登记）"
        % (GOLDEN_OFF_DIGEST, d_off))

    # 显式传 0.0 与"完全不传该键"必须**同样**是旧行为
    d_off2, sum_off2, n_off2, _nd2 = _run_scripted({
        "engagement_trade": 0.0, "engagement_trade_theta": 1.0,
        "engagement_trade_t_ref": 2.0, "engagement_trade_gate": 1})
    assert (d_off2, sum_off2, n_off2) == (d_off, sum_off, n_off), \
        "显式 0.0 与缺省必须逐位相同"

    # —— 3：开关打开且 θ=0 ⇒ 项**必须**改变奖励（否则是假通过）——
    #  ⚠️ 必须长到**局内**就有窗口结算：score 是在**结算的那一决策帧**加进奖励的，
    #  只在局末 flush 的话分数永远进不了任何一帧（这正是文档 §1 的"尾部窗口"已知缺口）。
    d_on, sum_on, n_on, _nd3 = _run_scripted({
        "engagement_trade": 1.0, "engagement_trade_theta": 0.0,
        "engagement_trade_t_ref": 2.0, "engagement_trade_gate": 1}, steps=200)
    assert n_on > 0, ("打开开关后应至少结算一个局面（实得 %d）—— 注意本测试已显式补 flush"
                      % n_on)
    assert d_on != d_off, (
        "打开开关且 θ=0 后奖励/状态摘要**没有变化** ⇒ 这条链没接上"
        "（关时的'相同'是假通过）。n_settled=%d" % n_on)
    return ("off ⇒ digest == 黄金值 %s…，_et is None；显式 0.0 亦同；"
            "on(θ=0) ⇒ digest 改变且 n_settled=%d（和 %.3f vs %.3f）"
            % (GOLDEN_OFF_DIGEST[:12], n_on, sum_off, sum_on))


def test_measure_only_is_behavior_neutral():
    """★ measure-only：跑监视器 + 把在线窗口写进 `info`，**但一分奖励都不加**
    ⇒ 行为必须与"完全不接线"**逐位相同**（否则后续用它量的口径就不能代表真实训练）。"""
    d_off, sum_off, _n_off, _nd = _run_scripted(None, steps=200)   # ⚠️ 必须与下一行同长度
    d_m, sum_m, n_m, nd_m = _run_scripted({
        "engagement_trade": 0.0, "engagement_trade_measure_only": 1,
        "engagement_trade_theta": 1.0, "engagement_trade_t_ref": 2.0,
        "engagement_trade_gate": 1}, steps=200)
    assert d_m == d_off and sum_m == sum_off, (
        "measure-only 改变了行为！\n  off  %s (%.6f)\n  meas %s (%.6f)"
        % (d_off, sum_off, d_m, sum_m))
    assert n_m > 0 and nd_m > 0, (
        "measure-only 必须把在线窗口记下来（n_settled=%d, 明细窗口=%d）" % (n_m, nd_m))
    return ("measure-only ⇒ 行为与不接线逐位相同（%s…）；同时结算 %d 个窗口 / 落 %d 条明细"
            % (d_m[:12], n_m, nd_m))


_TESTS = [("test_engagement_trade_default_off", test_engagement_trade_default_off),
          ("test_measure_only_is_behavior_neutral", test_measure_only_is_behavior_neutral)]


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ok = 0
    for name, fn in _TESTS:
        try:
            print("[PASS] %-40s %s" % (name, fn() or ""))
            ok += 1
        except Exception as e:                       # noqa: BLE001
            print("[FAIL] %-40s %s: %s" % (name, type(e).__name__, e))
            traceback.print_exc()
    print("\n%d/%d PASS" % (ok, len(_TESTS)))
    return 0 if ok == len(_TESTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
