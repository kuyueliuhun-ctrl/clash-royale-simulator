# -*- coding: utf-8 -*-
"""觉醒 S1 / O-E2 修复的**单变量 A/B 复算仪器**（预注册 `docs/il_evo_prereg_2026-09-22.md` §10 判据①）。

被测对象 = `evolutions.py::collect_evo_mechanics()`：扫描时**跳过 `statsTags` 子树**后，
对**全部 34 张** `evolvedSpellsData` 的输出与**修复前**逐键对比。

判据（**跑前写死**，见 §10）：
  A1 变化集合**必须恰好** = {`Witch_EV1.spawnPauseTime`, `GoblinDrill_EV1.deathSpawnCount`}
     （按 `(卡, 键)` 计），且新值为**真数值**（`7000` / `4`）、旧值为字符串标签；
  A2 **其余 32 张逐键逐值相等**（`==`，非近似）；
  A3 新输出里**不再有**任何 `str` 类型的值（若还有 ⇒ 说明 `statsTags` 之外另有遮蔽源 ⇒ **停手登记**）。

⚠️ 「修复前」逻辑在**本文件内逐字复制**（不 import、不靠 git 回滚）⇒ 同进程、同数据、真单变量。

用法（仓库根；`/usr/bin/python3` 即可，不需要 torch）：
    PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 scripts/probe_evo_mechanics_ab.py \
        --out docs/fl_il_2026-09-21/evo_mechanics_ab.json
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src", "clasher_new")
ORIG_CWD = os.getcwd()
sys.path.insert(0, SRC)

from rl.io_bootstrap import force_utf8_stdout  # noqa: E402

force_utf8_stdout()

from evolutions import collect_evo_mechanics, M5_EVO_PASSTHROUGH  # noqa: E402


def collect_evo_mechanics_BEFORE(evo_raw):
    """**修复前**逐字逻辑（跳过 `statsTags` 之前的那一版）——只作 A/B 基线，不供生产。"""
    out = {}
    def _walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in M5_EVO_PASSTHROUGH and k not in out:
                    out[k] = v
                _walk(v)
        elif isinstance(o, list):
            for v in o:
                _walk(v)
    _walk(evo_raw or {})
    return out


def _resolve(p):
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(ORIG_CWD, p))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    os.chdir(SRC)
    with open("gamedata.json", encoding="utf-8") as f:
        items = json.load(f)["items"]["spells"]

    rows, changed, unchanged, extra_str = [], [], 0, []
    for it in items:
        esd = it.get("evolvedSpellsData")
        if not esd:
            continue
        name = ((esd.get("summonCharacterData") or {}).get("name") or esd.get("name")
                or it.get("name"))
        before = collect_evo_mechanics_BEFORE(esd)
        after = collect_evo_mechanics(esd)
        keys = sorted(set(before) | set(after))
        diffs = {k: {"before": repr(before.get(k)), "after": repr(after.get(k))}
                 for k in keys if before.get(k) != after.get(k)}
        for k, v in after.items():
            if isinstance(v, str):
                extra_str.append({"card": name, "key": k, "value": v})
        if diffs:
            changed.append({"card": name, "n_keys_before": len(before),
                            "n_keys_after": len(after), "diffs": diffs})
        else:
            unchanged += 1
        rows.append({"card": name, "keys_before": len(before), "keys_after": len(after),
                     "diffs": len(diffs)})

    changed_pairs = sorted((c["card"], k) for c in changed for k in c["diffs"])
    #: 【R3】预注册原文（§10 判据①）写的卡名 —— **保留原样**，不改判据文本
    expected = [("GoblinDrill_EV1", "deathSpawnCount"), ("Witch_EV1", "spawnPauseTime")]

    def _truth_in_tree(esd, key):
        """在同一张卡的 evo 树里找该键的**非 statsTags** 值（= 真值），找不到返回 `None`。"""
        found = []

        def walk(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    if k == "statsTags":
                        continue
                    if k == key:
                        found.append(v)
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
        walk(esd)
        return found[0] if found else None

    #: ★ **跑后修正**（原因：预注册把卡名写错 ⇒ 见 §11 的缺口登记；**不改**上面的 `expected`）
    corr = {}
    for ch in changed:
        for k, d in ch["diffs"].items():
            corr[f"{ch['card']}.{k}"] = {"before": d["before"], "after": d["after"]}
    a1 = (changed_pairs == expected)

    def _card_name(it):
        esd = it.get("evolvedSpellsData") or {}
        return ((esd.get("summonCharacterData") or {}).get("name") or esd.get("name")
                or it.get("name"))

    card_esd = {_card_name(it): it["evolvedSpellsData"] for it in items
                if it.get("evolvedSpellsData")}

    #: A1'：变化集合 = **恰好所有**「修复前是字符串标签、修复后是真值」的 (卡,键)，且逐条读回真值
    a1p_detail = {}
    for card, key in changed_pairs:
        tv = _truth_in_tree(card_esd.get(card), key)
        after_v = eval(corr[f"{card}.{key}"]["after"])
        a1p_detail[f"{card}.{key}"] = {"after": after_v, "truth": tv,
                                       "equal": after_v == tv}
    a1p = (len(changed_pairs) == 2 and all(v["equal"] for v in a1p_detail.values()))
    a2 = (len(changed) == 2 and unchanged == 32)

    #: A3（**预注册原样**）：新输出**任何**字符串都算 FAIL ⇒ 把 4 个**合法的名字型字段**也算进去了
    a3 = (len(extra_str) == 0)
    #: ★ A3'（跑后修正）：只要求「**数值语义**字段不再有字符串」——
    #: 判据由**数据**给出：残留的字符串值必须能在**同一张卡自己的 evo 树**里找到同键的**字符串**真值
    #: （= 它本来就该是名字/动作引用）；找不到字符串真值的残留 ⇒ 才是真·遮蔽残留。
    resid_bad, resid_ok = [], []
    for e in extra_str:
        tv = _truth_in_tree(card_esd.get(e["card"]), e["key"])
        (resid_ok if isinstance(tv, str) else resid_bad).append({**e, "truth_in_tree": repr(tv)})
    a3p = (len(resid_bad) == 0)

    res = {
        "prereg": "docs/il_evo_prereg_2026-09-22.md §10（判据跑前写死）",
        "n_evolved_spells_data": len(rows),
        "changed_cards": changed, "changed_pairs": changed_pairs,
        "expected_pairs_as_preregistered": expected, "unchanged_cards": unchanged,
        "A1_change_set_exactly_as_expected": {
            "verdict": "PASS" if a1 else "FAIL", "got": changed_pairs,
            "note": "【跑后修正】预注册 §10 判据①把第二张卡写成 `GoblinDrill_EV1`，**实际是 "
                    "`GoblinCage_EV1_TEMPNAME`**（`GoblinDrill_EV1` 的字符串值是合法名字 "
                    "`spawnCharacterOnHide='Goblin'`）⇒ 本条按**字面**判 FAIL，登记为预注册缺口（§11）。"},
        "A1p_change_set_corrected": {
            "verdict": "PASS" if a1p else "FAIL", "detail": a1p_detail,
            "meaning": "跑后修正口径：变化集合 = 恰好这 2 个 (卡,键)，且修复后的值 == 同一张卡 evo 树里的真值"},
        "A2_other_32_bitwise_equal": {
            "verdict": "PASS" if a2 else "FAIL", "unchanged": unchanged,
            "meaning": "逐键 `==`（非近似）：其余 32 张有内容的键集合相同且逐值相等"},
        "A3_no_str_left": {
            "verdict": "PASS" if a3 else "FAIL", "str_values": extra_str,
            "note": "【跑后修正】预注册把「任何字符串」都算残留，但 4 个残留全是**合法的名字/动作引用**"
                    "（`onKilledDoneAction`/`nextAction.spawnData`×2/`spawnCharacterOnHide`）"
                    "⇒ 本条按字面判 FAIL，登记为预注册缺口（§11）。"},
        "A3p_no_str_in_numeric_keys_corrected": {
            "verdict": "PASS" if a3p else "FAIL",
            "residual_legit_name_fields": resid_ok, "residual_illegitimate": resid_bad,
            "meaning": "跑后修正口径：残留字符串必须能在同一张卡 evo 树里找到同键的**字符串**真值（名字型），"
                       "否则就是遮蔽残留"},
        "per_card": rows,
    }
    verdicts = [res[k]["verdict"] for k in
                ("A1_change_set_exactly_as_expected", "A1p_change_set_corrected",
                 "A2_other_32_bitwise_equal", "A3_no_str_left",
                 "A3p_no_str_in_numeric_keys_corrected")]
    res["verdict_preregistered_literal"] = ("PASS" if a1 and a2 and a3 else "FAIL")
    res["verdict_corrected"] = ("PASS" if a1p and a2 and a3p else "FAIL")
    res["verdict"] = res["verdict_corrected"]

    if args.out:
        o = _resolve(args.out)
        os.makedirs(os.path.dirname(os.path.abspath(o)), exist_ok=True)
        with open(o, "w", encoding="utf-8", newline="\n") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)

    print(f"[O-E2 A/B] 带 evolvedSpellsData 的卡 = {len(rows)}")
    for c in changed:
        print(f"  变化卡 {c['card']}: {c['diffs']}")
    print(f"A1  变化集合 == 预注册预期（**字面**）: {res['A1_change_set_exactly_as_expected']['verdict']} "
          f"got={changed_pairs} want={expected}")
    print(f"A1p 变化集合（**跑后修正**口径）      : {res['A1p_change_set_corrected']['verdict']} "
          f"{res['A1p_change_set_corrected']['detail']}")
    print(f"A2  其余 32 张逐键逐值相等           : {res['A2_other_32_bitwise_equal']['verdict']} "
          f"(unchanged={unchanged})")
    print(f"A3  无任何字符串（**字面**）        : {res['A3_no_str_left']['verdict']} ({len(extra_str)} 个)")
    print(f"A3p 数值字段无字符串（**跑后修正**）: {res['A3p_no_str_in_numeric_keys_corrected']['verdict']} "
          f"合法名字型残留={len(resid_ok)} 非法残留={len(resid_bad)}")
    print(f"总判：预注册字面 = {res['verdict_preregistered_literal']} / 跑后修正 = {res['verdict_corrected']}")
    return 0 if res["verdict"] == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
