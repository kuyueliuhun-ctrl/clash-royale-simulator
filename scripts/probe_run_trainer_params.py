# -*- coding: utf-8 -*-
"""H1 定论探针：`run` 模式的 5 个 PPO 开关到底有没有生效（只读，不训练）。

背景（2026-09-19 复核三轴对照 doc 的 H1）：
`rl/run_league.py` 的 argparse **确实**收下 `--ppo-epochs / --ppo-minibatch / --ppo-shuffle /
--value-norm / --diagnose-every` 并写进 `cfg`（`:1589-1595` → `TrainConfig.resolve(..., **overrides)`），
但 `_make_trainer(main, cfg)` **只传 8 个参数**、不含这 5 个 ⇒ `PPOTrainer` 用**构造函数默认值**
（`n_epochs=1, minibatch_size=0, shuffle=False, value_norm="none", diagnose_every=0`）。

本脚本把"cfg 说 4、trainer 是 1"这件事变成**可执行断言**，并用**反向用例**证明它有判别力
（同一批 kwarg **显式传**进去时确实会落地 ⇒ 说明差异来自 `_make_trainer` 的遗漏，
而不是 `PPOTrainer` 忽略参数）。

用法（Windows venv python）：
    .venv/Scripts/python.exe scripts/probe_run_trainer_params.py            # 跑断言
    .venv/Scripts/python.exe scripts/probe_run_trainer_params.py --json     # 额外输出 JSON

退出码：0 = 全部断言成立（= H1 的代码事实成立）；1 = 有断言不成立（H1 被推翻，需重查）。

⚠️ **本脚本只测量**：不改任何训练参数、不构造环境、不读 ckpt。
stdout 保持 ASCII-only（本环境 `PYTHONIOENCODING=utf-8` 实测不生效，stdout=gbk）。
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src", "clasher_new")
sys.path.insert(0, SRC)

import torch  # noqa: E402

from rl.io_bootstrap import force_utf8_stdout  # noqa: E402

#: 走单一实现（不手写 reconfigure，见 _structure_check ⑧）；path 引导已在上面完成（⑨/⑪）。
force_utf8_stdout()

from rl.config import TrainConfig  # noqa: E402
from rl.ppo import PPOTrainer  # noqa: E402
from rl.run_league import _make_trainer  # noqa: E402

#: 故意全部设成**非默认**（正是文档 `--ppo-epochs 4` 那一组）
WANT = {
    "ppo_epochs": 4,
    "ppo_minibatch": 4,
    "ppo_shuffle": True,
    "value_norm": "running",
    "diagnose_every": 10,
}
#: PPOTrainer 的构造默认值（rl/ppo.py:88-90）
CTOR_DEFAULTS = {
    "ppo_epochs": 1,
    "ppo_minibatch": 0,
    "ppo_shuffle": False,
    "value_norm": "none",
    "diagnose_every": 0,
}


def _stub_policy():
    """PPOTrainer 构造只用到 `policy.parameters()`（`ppo.py:113`）⇒ 线性层足够。"""
    return torch.nn.Linear(2, 2)


def _read(ppo):
    return {
        "ppo_epochs": int(ppo.n_epochs),
        "ppo_minibatch": int(ppo.minibatch_size),
        "ppo_shuffle": bool(ppo.shuffle),
        "value_norm": str(ppo.value_norm),
        "diagnose_every": int(ppo.diagnose_every),
    }


def _grad_steps_per_update(ppo, batch):
    """按 `ppo.py:251-252` 的 legacy 判据与 `:472-475` 的切批规则算每 update 的 `opt.step()` 次数。

    这是**读代码得出的推导**（不是实测训练），用于把 H1 的量级说清楚。
    """
    legacy = (ppo.n_epochs <= 1 and not ppo.shuffle
              and (ppo.minibatch_size <= 0 or ppo.minibatch_size >= batch))
    if legacy:
        return 1
    mb = max(1, ppo.minibatch_size)
    n_mb = (batch + mb - 1) // mb
    return ppo.n_epochs * n_mb


def main(argv):
    batch = 128
    out = {"want": WANT, "ctor_defaults": CTOR_DEFAULTS}
    checks = []

    # preset 必须是一个已知名字（`TrainConfig.resolve` 对未知 preset 直接 raise）；
    # 这里选 `standard`，因为它与 run 模式的历史默认最接近，且我们只读被 override 的 5 个字段。
    cfg = TrainConfig.resolve("standard", **WANT)
    got_cfg = {k: getattr(cfg, k) for k in WANT}
    out["cfg"] = got_cfg

    # A. cfg 确实收下了这 5 个值（否则 H1 的诊断对象都不存在）
    checks.append(("A cfg carries the 5 CLI values", got_cfg == WANT, f"cfg={got_cfg}"))

    # B. run 模式构造出来的 trainer = 构造默认值（= 开关没到）
    ppo = _make_trainer(_stub_policy(), cfg)
    got_run = _read(ppo)
    out["run_trainer"] = got_run
    checks.append(("B run-mode trainer == ctor defaults", got_run == CTOR_DEFAULTS,
                   f"trainer={got_run}"))
    checks.append(("B2 run-mode trainer != requested",
                   got_run != got_cfg, f"{got_run} vs {got_cfg}"))

    # C. 判别力：同样的 kwarg **显式传**进去必须落地（证明 B 的读数是可测的差异）
    ppo2 = PPOTrainer(_stub_policy(),
                      n_epochs=cfg.ppo_epochs, minibatch_size=cfg.ppo_minibatch,
                      shuffle=cfg.ppo_shuffle, value_norm=cfg.value_norm,
                      diagnose_every=cfg.diagnose_every)
    got_explicit = _read(ppo2)
    out["explicit_trainer"] = got_explicit
    checks.append(("C explicit kwargs DO land (discriminating power)",
                   got_explicit == WANT, f"trainer={got_explicit}"))

    # D. 量级：每 update 的梯度步数（读代码推导）
    gs_run = _grad_steps_per_update(ppo, batch)
    gs_want = _grad_steps_per_update(ppo2, batch)
    out["grad_steps_per_update"] = {"run_actual": gs_run, "if_wired": gs_want, "batch": batch}
    checks.append(("D grad steps: run=%d vs if-wired=%d (hypothetical batch=%d, not solo's batch)"
                   % (gs_run, gs_want, batch),
                   gs_run == 1 and gs_want == WANT["ppo_epochs"] * (batch // WANT["ppo_minibatch"]),
                   f"run={gs_run} if_wired={gs_want}"))

    ok = 0
    for name, passed, detail in checks:
        print(("[PASS] " if passed else "[FAIL] ") + name + "  " + detail)
        ok += bool(passed)
    print(f"[probe] H1 code fact = {'CONFIRMED' if ok == len(checks) else 'NOT CONFIRMED'}"
          f"  ({ok}/{len(checks)} assertions)")
    if "--json" in argv:
        out["assertions"] = [{"name": n, "pass": bool(p), "detail": d} for n, p, d in checks]
        print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if ok == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
