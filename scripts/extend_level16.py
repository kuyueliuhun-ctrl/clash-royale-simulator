#!/usr/bin/env python3
"""数据文件 16 级支持：把 cards_stats_*.json 中所有 per_level 数值数组延伸到「稀有度规范长度」。
- 规范长度（实测主卡）：Common 19 / Rare 17 / Epic 14 / Legendary 11 / Champion 9（覆盖到 16 级及更远）
- 延伸曲线：取数组末 3 步比值均值（夹取 [1.05,1.15]，兜底 1.0985）延续，round 取整
- 只延伸不足规范长度的数组；为修改过的行加 "level16_extended": true 标记
- 先备份原文件到 <file>.bak
运行：cd src/clasher_new && python3 ../../scripts/extend_level16.py
"""
import json, os, shutil

SRC = os.path.join(os.path.dirname(__file__), '..', 'src', 'clasher_new')
FILES = ['cards_stats_characters.json', 'cards_stats_building.json',
         'cards_stats_spell.json', 'cards_stats_projectile.json']
CANON = {'Common': 19, 'Rare': 17, 'Epic': 14, 'Legendary': 11, 'Champion': 9}
FIELDS = ('hitpoints_per_level', 'damage_per_level', 'dps_per_level')


def extend_array(arr):
    """把数组延续到 CANON[rarity] 长度，返回 (新数组, 是否修改)。"""
    n = len(arr)
    if n == 0:
        return arr, False
    # 末 3 步比值均值（防跳变夹取）
    steps = []
    for i in range(max(1, n - 3), n):
        if arr[i - 1]:
            steps.append(arr[i] / arr[i - 1])
    step = (sum(steps) / len(steps)) if steps else 1.0985
    step = max(1.05, min(1.15, step))
    out = list(arr)
    for _ in range(CANON.get(rarity_of(arr), 19) - n):
        out.append(round(out[-1] * step))
    return out, len(out) != n


def rarity_of(arr):
    # 占位：由调用方注入（数组本身不带稀有度）
    return None


def main():
    modified_total = 0
    for fn in FILES:
        path = os.path.join(SRC, fn)
        with open(path) as f:
            rows = json.load(f)
        changed = 0
        for r in rows:
            rr = r.get('rarity') or 'Common'
            target = CANON.get(rr, 19)
            for col in FIELDS:
                arr = r.get(col) or []
                if arr and len(arr) < target:
                    steps = []
                    for i in range(max(1, len(arr) - 3), len(arr)):
                        if arr[i - 1]:
                            steps.append(arr[i] / arr[i - 1])
                    step = (sum(steps) / len(steps)) if steps else 1.0985
                    step = max(1.05, min(1.15, step))
                    out = list(arr)
                    for _ in range(target - len(arr)):
                        out.append(round(out[-1] * step))
                    r[col] = out
                    changed += 1
            if changed:
                r['level16_extended'] = True
        if changed:
            shutil.copyfile(path, path + '.bak')
            with open(path, 'w') as f:
                json.dump(rows, f, ensure_ascii=False, indent=1)
            modified_total += changed
            print(f'{fn}: 延伸 {changed} 个数组 → 稀有度规范长度')
        else:
            print(f'{fn}: 无需延伸')
    print(f'\n完成。共延伸 {modified_total} 个数组（原文件已备份为 .bak）')


if __name__ == '__main__':
    main()
