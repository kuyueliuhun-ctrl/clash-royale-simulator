# -*- coding: utf-8 -*-
"""把 S2 的「出牌溯源纯记录通道」打进 battle.py（sandbox 里先做，验证后再进真树）。

用法：
    python3 patch_battle_root_cast.py --file /tmp/rtpatch/battle.py            # dry-run
    python3 patch_battle_root_cast.py --file /tmp/rtpatch/battle.py --apply
"""
from __future__ import annotations

import argparse
import ast
import sys

# T1-1 追加（2026-09-19）：UTF-8 兜底。本文件含 **GBK 编不出**的字符（⇒ 无兜底时 `print` 抛
# UnicodeEncodeError）。本文件**不 import `rl`/引擎** ⇒ 这里补一段自足引导（与本仓其它入口脚本同形态），
# 再取 T1-1 的**单一实现**（不用第二份 reconfigure 手写块）。
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                                 "src", "clasher_new"))
from rl.io_bootstrap import force_utf8_stdout  # noqa: E402
force_utf8_stdout()

ENTITY_FIELDS_ANCHOR = ("        self.shield_health = self.data.shield_health\n"
                       "        self.target_id = None\n")
ENTITY_FIELDS_NEW = """        self.shield_health = self.data.shield_health
        self.target_id = None
        # —— S2 纯记录通道（2026-09-18）：出牌溯源。**只写不读** ——
        # `root_cast` = 产出本实体的那次**出牌事件 id**（不是实体 id；链式继承到根）；
        # `is_product` = 本实体是否由建筑/单位/法术**生成**（None = 未分类，由 _spawn_entity 填）。
        # 二者不参与任何行为/掩码/伤害判定 ⇒ 默认关奖励项时逐位回旧。
        # 规格：docs/engagement_trade_prereg_2026-09-18.md §3/§6
        self.root_cast = None
        self.is_product = None
"""

STATE_FIELDS_ANCHOR = "        self.next_entity_id = 1\n        self.regen = 2.8\n"
STATE_FIELDS_NEW = """        self.next_entity_id = 1
        self.regen = 2.8
        # —— S2 纯记录通道（2026-09-18）：出牌事件台账 ——
        self.next_cast_id = 1        # 出牌事件 id（单调；出牌失败也消耗，留空洞）
        self.cast_log = {}           # cast_id -> {"player","card","t","mirror"}
        self._cast_ctx = None        # 正在进行的出牌事件 id（deploy_card 期间）
        self._cast_is_product = False  # 当前出牌是法术 ⇒ 它直接生成的东西算产物
"""

DEPLOY_WRAPPER = '''
    def _finish_cast(self, cast_id, player_id, card_name, from_mirror, ok):
        """S2 纯记录：只把**成功**的出牌事件写进 `cast_log`（失败留空洞，无害）。"""
        if ok and cast_id is not None:
            self.cast_log[int(cast_id)] = {
                "player": int(player_id), "card": str(card_name),
                "t": float(self.time), "mirror": bool(from_mirror)}

    def deploy_card(self, player_id, card_name, position, _from_mirror=False):
        """S2 包装（2026-09-18）：给 `_deploy_card_impl` 套一层「出牌事件」上下文。

        包装体**不改变任何行为**：只分配 `cast_id`、设置/恢复两个记录用上下文、
        成功时写 `cast_log`。镜像会递归进 `deploy_card` ⇒ 每张镜像卡算**独立**一次出牌事件。
        """
        cast_id = self.next_cast_id
        self.next_cast_id += 1
        prev_ctx, prev_flag = self._cast_ctx, self._cast_is_product
        self._cast_ctx = cast_id
        ok = False
        try:
            ok = self._deploy_card_impl(player_id, card_name, position,
                                        _from_mirror=_from_mirror)
        finally:
            self._finish_cast(cast_id, player_id, card_name, _from_mirror, ok)
            self._cast_ctx, self._cast_is_product = prev_ctx, prev_flag
        return ok
'''

SPAWN_ANCHOR = """    def _spawn_entity(self, entity):
        self.ensure_walkability(entity)
        entity.battle_state = self
        entity.id = self.next_entity_id
        self.entities[self.next_entity_id] = entity
        self.next_entity_id += 1
"""
SPAWN_NEW = '''    def _spawn_entity(self, entity, spawner=None, is_product=None):
        """出生登记。**新增两个参数只服务 S2 纯记录通道，不改变任何行为。**

        - `spawner`：生成者实体。有 ⇒ 该实体是**产物体**，且 `root_cast` 从宿主
          **链式继承**（宿主可能已死 ⇒ 指针写在产物体自己身上，规格 §3.4）。
        - `is_product`：显式覆盖（塔 = False；法术直接生成 = True）。
        """
        self.ensure_walkability(entity)
        entity.battle_state = self
        entity.id = self.next_entity_id
        self.entities[self.next_entity_id] = entity
        self.next_entity_id += 1
        # —— S2 纯记录通道（只写不读）——
        if getattr(entity, "root_cast", None) is None:
            _rc = getattr(spawner, "root_cast", None) if spawner is not None else None
            entity.root_cast = _rc if _rc is not None else self._cast_ctx
        if getattr(entity, "is_product", None) is None:
            if is_product is not None:
                entity.is_product = bool(is_product)
            elif spawner is not None:
                entity.is_product = True
            else:
                entity.is_product = bool(self._cast_is_product)
        return entity
'''

DELAY_ANCHOR = """    def delayed_spawn(self, entity, delay):
        if delay:
            self.schedule.append((entity, self.time+delay))
        else:
            self._spawn_entity(self._wrap(entity))
"""
DELAY_NEW = '''    def delayed_spawn(self, entity, delay, is_product=None, spawner=None):
        """延迟出兵。`schedule` 第三格带上 `(cast_id, is_product)` 快照。

        ⚠️ 出牌上下文在**排程那一刻**捕获（真正出生要等若干 tick，届时上下文已恢复）。
        """
        _root = getattr(spawner, "root_cast", None) if spawner is not None else None
        if _root is None:
            _root = self._cast_ctx
        if is_product is None:
            _isprod = True if spawner is not None else self._cast_is_product
        else:
            _isprod = bool(is_product)
        _meta = (_root, _isprod)
        if delay:
            self.schedule.append((entity, self.time+delay, _meta))
        else:
            _ent = self._wrap(entity)
            if getattr(_ent, "root_cast", None) is None:
                _ent.root_cast = _meta[0]
            if getattr(_ent, "is_product", None) is None:
                _ent.is_product = bool(_meta[1])
            self._spawn_entity(_ent)
'''

SCHED_ANCHOR = """        for entity, spawn_time in self.schedule:
            if self.time >= spawn_time: self._spawn_entity(self._wrap(entity))
"""
SCHED_NEW = """        for _item in self.schedule:
            if self.time >= _item[1]:
                _ent = self._wrap(_item[0])
                if len(_item) > 2 and _item[2] is not None:
                    if getattr(_ent, "root_cast", None) is None:
                        _ent.root_cast = _item[2][0]
                    if getattr(_ent, "is_product", None) is None:
                        _ent.is_product = bool(_item[2][1])
                self._spawn_entity(_ent)
"""

TOWER_ANCHORS = []
for _i in range(1, 7):
    TOWER_ANCHORS.append(("        self._spawn_entity(Building(%d, " % _i))

HOST_HINTS = ("src", "source", "owner", "caster", "host", "projectile")


def build_parents(tree):
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def nearest_fn(node, parents):
    cur = parents.get(node)
    while cur is not None:
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return cur
        cur = parents.get(cur)
    return None


def nearest_class(fn, parents):
    cur = parents.get(fn) if fn is not None else None
    while cur is not None:
        if isinstance(cur, ast.ClassDef):
            return cur
        cur = parents.get(cur)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    src = open(a.file, encoding="utf-8").read()
    if "root_cast" in src:
        print("[SKIP] 已经打过补丁")
        return 0

    for anchor, new in ((ENTITY_FIELDS_ANCHOR, ENTITY_FIELDS_NEW),
                        (STATE_FIELDS_ANCHOR, STATE_FIELDS_NEW)):
        assert src.count(anchor) == 1, ("锚点不唯一", anchor[:40], src.count(anchor))
        src = src.replace(anchor, new, 1)

    d_old = ("    def deploy_card(self, player_id, card_name, position, "
             "_from_mirror=False):")
    assert src.count(d_old) == 1
    src = src.replace(d_old, d_old.replace("def deploy_card", "def _deploy_card_impl"), 1)

    assert src.count(SPAWN_ANCHOR) == 1
    src = src.replace(SPAWN_ANCHOR, DEPLOY_WRAPPER + "\n" + SPAWN_ANCHOR, 1)
    src = src.replace(SPAWN_ANCHOR, SPAWN_NEW, 1)
    assert src.count(DELAY_ANCHOR) == 1
    src = src.replace(DELAY_ANCHOR, DELAY_NEW, 1)
    assert src.count(SCHED_ANCHOR) == 1
    src = src.replace(SCHED_ANCHOR, SCHED_NEW, 1)

    tree = ast.parse(src)
    parents = build_parents(tree)
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr in ("_spawn_entity", "delayed_spawn")]
    lines = src.splitlines(keepends=True)
    edits, report = [], []
    for c in calls:
        fn = nearest_fn(c, parents)
        cls = nearest_class(fn, parents)
        fn_name = getattr(fn, "name", None)
        if fn_name in ("_spawn_entity", "delayed_spawn", "deploy_card",
                       "_finish_cast", "_deploy_card_impl"):
            continue
        # 已经带 spawner=/is_product= 的（手工替换过）不再插
        if any(k.arg in ("spawner", "is_product") for k in c.keywords):
            continue
        # 塔：__init__ 里的 6 个
        if fn_name == "__init__" and cls is not None and cls.name == "BattleState":
            edits.append((c, None, False, "§7 塔：显式 is_product=False"))
            continue
        # BattleState 自己的管理性调用（step/schedule/resurrect/spawn_arrival_troops）
        # 由下面的手工替换处理，不进 AST 批量插入
        if cls is not None and cls.name == "BattleState":
            continue
        if cls is not None:
            host, why = "self", "类 %s 的方法 %s(self)" % (cls.name, fn_name)
        else:
            args = []
            if fn is not None:
                aa = fn.args
                args = [x.arg for x in list(aa.posonlyargs) + list(aa.args)
                        + list(aa.kwonlyargs)]
            host = next((h for h in HOST_HINTS if h in args), None)
            why = ("模块级 %s(args=%s)" % (fn_name, args)) if fn is not None \
                else "无外层函数"
        edits.append((c, host, True, why))

    edits.sort(key=lambda e: (e[0].end_lineno, e[0].end_col_offset), reverse=True)
    for c, host, use_product, why in edits:
        end = c.end_lineno
        line = lines[end - 1]
        col = c.end_col_offset - 1
        assert line[col] == ")", (end, col, line)
        if c.func.attr == "delayed_spawn":
            ins = ""
        elif host:
            ins = ", spawner=%s" % host
        else:
            ins = ", is_product=%s" % ("False" if use_product is False else "True")
        lines[end - 1] = line[:col] + ins + line[col:]
        report.append("  L%-5d %-34s %s%s" % (c.lineno, c.func.attr,
                                              host or "(no host)", "  # " + why))
    src = "".join(lines)
    print("调用点 %d 个：" % len(report))
    print("\n".join(sorted(report)))
    try:
        ast.parse(src)
        print("\n[AST OK]")
    except SyntaxError as e:
        print("\n[AST FAIL] %s" % e)
        return 1
    if not a.apply:
        print("[DRY-RUN] 未写文件")
        return 0
    open(a.file, "w", encoding="utf-8").write(src)
    print("[APPLIED] %s" % a.file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
