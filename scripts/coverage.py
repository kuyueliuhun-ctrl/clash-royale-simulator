#!/usr/bin/env python3
"""卡牌内容覆盖矩阵生成器（P0-1 基本信息录入）v2 —— 证据驱动分类

基准政策（已确认）：
  - 特殊机制语义：以 Null 服快照（gamedata.json）为基准
  - 数值：以官方数值为准（官方API元数据 + cards_stats_*.json 官方同源 per-level 表）
  - 拿不准的条目：进 review 队列交人类决策，不擅自定案

官方数据目录（可选）：src/clasher_new/data_official/
  存在则优先使用（L1 升级：cr-api-data 的 cards.json + cards_stats_*.json）。
  获取命令（需有网环境）：
    for f in cards.json cards_stats_characters.json cards_stats_building.json \
             cards_stats_spell.json cards_stats_projectile.json; do
      curl -sL "https://raw.githubusercontent.com/RoyaleAPI/cr-api-data/master/data/$f" \
        -o src/clasher_new/data_official/$f; done

输出：docs/card_registry.json（机器可读主册）、docs/card_coverage.md（人类可读报告）
"""
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'src', 'clasher_new')
OFFICIAL = os.path.join(SRC, 'data_official')
DOCS = os.path.join(ROOT, 'docs')

TOWER_TID = 'TID_TYPE_TOWER_TROOP'
# 证据确凿的临时/内部卡（活动模式专用、超级卡、变体），不进可部署池
CERTAIN_TEMPORARY = {'SkeletonWarriors_SpookyChess', 'SuperHogRiderTerry', 'SuperEliteArcher',
                     'MergeMaiden_Normal', 'MergeMaiden_Mounted'}
# 用户已确认的限时卡（人工拍板：不影响现行游戏，排除出可部署池）
USER_CONFIRMED_LIMITED = {'PrinceBuff', 'TriWizards', 'BarbarianLauncher',
                          'GoblinRocketSilo', 'WarmSpell'}
# 用户确认 + 公开数据源抓取完成的 2026 新卡（RoyaleZone/gamer.org，2026-09-02 经 CDP 抓取）
# 机制洞察：2026 Hero 卡 = 基础卡 + 主动技能（费用/属性继承基础卡），与 abilityData 英雄卡同体系
USER_CONFIRMED_2026 = [
    {'english_name': 'Hero Berserker', 'elixir': 2, 'rarity': 'Common', 'targets': 'Ground',
     'type': 'Troop', 'underlying': 'Berserker',
     'ability': 'Berserk Mode：激活后唤出灵兽化身，提高攻速，且保护窗口内血量不会降至 1 以下',
     'source': 'https://royalezone.com/cards/hero-berserker'},
    {'english_name': 'Hero Valkyrie', 'elixir': 4, 'rarity': 'Rare', 'targets': 'Ground',
     'type': 'Troop', 'underlying': 'Valkyrie',
     'ability': '旋转机动：激活后旋转向前、在近身敌军之间移动并造成范围伤害；技能期间仍可被击杀',
     'source': 'https://royalezone.com/cards/hero-valkyrie'},
]
# 觉醒野蛮人精锐：非独立卡，是 Elite Barbarians 的觉醒形态（内部名 AngryBarbarians）
EVO_CONFIRMED_2026 = {
    'name': 'Elite Barbarians Evolution（觉醒野蛮人精锐）',
    'evolution_cycle': 1,  # 首个经公开资料确认的周期数据点（gamer.org 指南表格）
    'ability': 'Rage Spears：近战接敌前投掷狂暴长矛——命中造成伤害，落点留下狂暴轨迹'
               '（提升经过友军的移速与攻速）；基础数值与基础卡完全一致',
    'stats': {'elixir': 6, 'units': 2, 'target': 'Ground', 'move_speed': 'Very Fast',
              'hit_speed': '1.4s'},
    'source': 'https://www.gamer.org/clash-royale-evo-elite-barbarians-guide/ + deckmelon.com',
}


def norm(s):
    return re.sub(r'[^a-z0-9]', '', (s or '').lower())


def jload(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def load_sources():
    gamedata = jload(os.path.join(SRC, 'gamedata.json'))['items']['spells']

    def load_meta_list(path):
        raw = jload(path)
        if isinstance(raw, dict) and 'items' in raw:
            raw = raw['items']
        return [c for c in raw if isinstance(c, dict)]

    # 元数据：仓库 cards.json（官方API导出）为主；data_official（cr-api-data，实测更旧）只做增量并集
    cards_meta = load_meta_list(os.path.join(SRC, 'cards.json'))
    extra_m = 0
    off_meta = os.path.join(OFFICIAL, 'cards.json')
    if os.path.exists(off_meta):
        seen = {norm(c.get('name')) for c in cards_meta} | {c.get('id') for c in cards_meta}
        for c in load_meta_list(off_meta):
            if norm(c.get('name')) not in seen and c.get('id') not in seen:
                cards_meta.append(c)
                extra_m += 1
    meta_src = f'仓库 cards.json + cr-api-data 补充({extra_m})' if extra_m else '仓库 cards.json'

    # 数值表：仓库为主（实测更新），data_official 只并入仓库没有的条目
    stats, stats_src = {}, {}
    for kind, fn in (('characters', 'cards_stats_characters.json'),
                     ('building', 'cards_stats_building.json'),
                     ('spell', 'cards_stats_spell.json'),
                     ('projectile', 'cards_stats_projectile.json')):
        rows = jload(os.path.join(SRC, fn))
        idx = {r['name']: r for r in rows if isinstance(r, dict) and 'name' in r}
        n_repo, extra_s = len(idx), 0
        off = os.path.join(OFFICIAL, fn)
        if os.path.exists(off):
            for r in jload(off):
                if isinstance(r, dict) and r.get('name') and r['name'] not in idx:
                    idx[r['name']] = r
                    extra_s += 1
        stats[kind] = idx
        stats_src[kind] = f'repo({n_repo})+official补充({extra_s})' if extra_s else f'repo({n_repo})'
    images = {f[:-4] for f in os.listdir(os.path.join(SRC, 'client_side', 'images'))
              if f.endswith('.png')}
    return gamedata, cards_meta, stats, images, meta_src, stats_src


def special_fields(entry):
    scd = entry.get('summonCharacterData', {}) or {}
    if not isinstance(scd, dict):
        scd = {}
    common = {'name', 'rarity', 'hitpoints', 'hitSpeed', 'loadTime', 'damage', 'range',
              'sightRange', 'speed', 'deployTime', 'collisionRadius', 'tidTarget',
              'tid', 'tidSpeed', 'mass', 'source', 'attacksGround', 'tidType'}
    skip = {'summonCharacterData', 'name', 'rarity', 'source', 'tid', 'tidInfo',
            'tidType', 'englishName', 'iconFile', 'highresImageFilename', 'id',
            'manaCost', 'unlockArena', 'tribe', 'evolvedSpellsData', 'attacksGround'}
    found = {k for k in list(entry.keys()) + list(scd.keys()) if k not in common}
    found -= skip
    return sorted(found)


def basic_stats_lv11(scd_name, stats, rarity):
    rarity_idx = {'Common': 10, 'Rare': 8, 'Epic': 5, 'Legendary': 2, 'Champion': 0}
    li = rarity_idx.get((rarity or '').lower().capitalize())
    out = {}
    if li is None:
        return out
    for kind in ('characters', 'building'):
        row = stats[kind].get(scd_name)
        if row:
            hpl, dpl = row.get('hitpoints_per_level'), row.get('damage_per_level')
            if hpl and li < len(hpl):
                out['hp_lv11'] = hpl[li]
            if dpl and li < len(dpl):
                out['damage_lv11'] = dpl[li]
            out['stats_source'] = kind
    return out


def classify(e, meta, has_meta):
    internal = e.get('name', '?')
    eng = e.get('englishName') or ''
    tid = e.get('tidType', '')
    if tid == TOWER_TID:
        return 'tower', []
    if internal in CERTAIN_TEMPORARY or internal.endswith('_Chess'):
        return 'variant', ['TEMPORARY_VARIANT: 活动变体/形态条目，排除出可部署池（基础形态保留）']
    if internal in USER_CONFIRMED_LIMITED:
        return 'temporary_event', ['USER_CONFIRMED_LIMITED: 人工确认为限时卡，不影响现行游戏，排除']
    if internal.startswith(('Super', 'Party')) or eng.startswith(('Super ', 'Party ', 'Santa ')):
        return 'temporary_event', ['TEMPORARY_EVENT: 活动/超级临时卡，默认排除（如需启用请人工确认）']
    if not has_meta:
        # 只记旗标不定死状态：若数值表可解析仍可成为 data_ready
        return None, ['META_MISSING: 官方元数据中无此卡（真伪/费用以快照为准，待人工核认）']
    return None, []  # 交由数据完备度决定


def build_registry():
    gamedata, cards_meta, stats, images, meta_src, stats_src = load_sources()
    meta_by_norm = {norm(c.get('name')): c for c in cards_meta}
    meta_by_id = {c.get('id'): c for c in cards_meta}
    registry = []

    for e in gamedata:
        internal, eng = e.get('name', '?'), e.get('englishName')
        flags = []
        meta = (meta_by_norm.get(norm(eng)) if eng else None) or meta_by_id.get(e.get('id'))
        has_meta = meta is not None

        status0, flags0 = classify(e, meta, has_meta)
        flags += flags0

        rarity_snap = e.get('rarity')
        elixir_snap = e.get('manaCost')
        # cr-api-data 元数据字段名可能与官方 API 不同，做宽容匹配
        rarity_official = (meta or {}).get('rarity')
        elixir_official = (meta or {}).get('elixirCost')
        if elixir_official is None and meta:
            elixir_official = meta.get('manaCost', meta.get('elixir'))
        if elixir_official is not None and elixir_snap is not None \
                and elixir_official != elixir_snap:
            flags.append(f'ELIXIR_MISMATCH: 官方={elixir_official} vs 快照={elixir_snap}（数值按官方）')
        if rarity_official and rarity_snap \
                and str(rarity_official).lower() != str(rarity_snap).lower():
            flags.append(f'RARITY_MISMATCH: 官方={rarity_official} vs 快照={rarity_snap}（按官方）')

        scd = e.get('summonCharacterData', {}) or {}
        scd_name = scd.get('name') if isinstance(scd, dict) else None
        tid = e.get('tidType', '')
        is_spell = tid == 'TID_CARD_TYPE_SPELL'
        # 法术的官方数值走 spell 表或 projectile 表，不依赖 summonCharacter
        resolved = (internal in stats['spell'] or bool(e.get('projectileData'))) if is_spell \
            else bool(scd_name and (scd_name in stats['characters'] or scd_name in stats['building']))
        if not resolved and status0 is None:
            reason = f'spell {internal} 不在 spell/projectile 表' if is_spell \
                else f'summonCharacter {scd_name} 不在 per-level 数值表'
            flags.append(f'STATS_UNRESOLVED: {reason}（需 L1 升级数值表或人工录入）')

        impl = bool(eng and eng in images)
        bs = basic_stats_lv11(scd_name, stats, rarity_official or rarity_snap) if scd_name else {}
        cat = {'TID_CARD_TYPE_CHARACTER': 'character', 'TID_CARD_TYPE_SPELL': 'spell',
               'TID_CARD_TYPE_BUILDING': 'building', TOWER_TID: 'tower'}.get(tid, tid or 'unknown')

        if status0:
            status = status0
        elif impl:
            status = 'implemented'
        elif resolved:
            status = 'data_ready'
        else:
            status = 'needs_review'

        registry.append({
            'internal_name': internal, 'english_name': eng,
            'cr_id': e.get('id') or (meta or {}).get('id'), 'category': cat,
            'rarity_snapshot': rarity_snap, 'rarity_official': rarity_official,
            'elixir_snapshot': elixir_snap, 'elixir_official': elixir_official,
            'elixir_canonical': elixir_official if elixir_official is not None else elixir_snap,
            'implemented': impl, 'summon_character': scd_name,
            'stats_resolved': resolved, 'basic_stats_lv11': bs,
            'special_fields': special_fields(e),
            'has_evolution': 'evolvedSpellsData' in e,
            'has_ability': 'abilityData' in (scd if isinstance(scd, dict) else {}) or 'abilityData' in e,
            'status': status, 'flags': flags,
        })

    # 2026 新卡（用户核实 + 公开源抓取，基本信息完整；机制参数待 L2 对齐 Null服格式）
    for p in USER_CONFIRMED_2026:
        registry.append({
            'internal_name': None, 'english_name': p['english_name'], 'cr_id': None,
            'category': 'character', 'rarity_snapshot': None, 'rarity_official': p['rarity'],
            'elixir_snapshot': None, 'elixir_official': p['elixir'],
            'elixir_canonical': p['elixir'],
            'implemented': False, 'summon_character': p.get('underlying'),
            'stats_resolved': False, 'basic_stats_lv11': {},
            'special_fields': ['hero_ability: ' + p['ability']],
            'has_evolution': False, 'has_ability': True, 'status': 'confirmed_live',
            'flags': ['USER_CONFIRMED_REAL: 用户已在 RoyaleAPI 核实（2026 Hero 类别）',
                      'BASIC_INFO_COMPLETE: 费用/稀有度/技能描述已从公开源录入',
                      f"UNDERLYING_CARD: 属性继承基础卡 {p['underlying']}（快照已有）",
                      'MECHANICS_PENDING_L2: 技能参数（窗口时长/倍率）待 L2 新快照 abilityData',
                      'SOURCE: ' + p['source']],
        })
    # 觉醒野蛮人精锐：Elite Barbarians 的觉醒（Season 86，用户核实 + gamer.org 数据）
    for r in registry:
        if r['internal_name'] == 'AngryBarbarians':
            r['has_evolution'] = True
            r['flags'].append('USER_CONFIRMED_REAL: 觉醒野蛮人精锐（'
                              + EVO_CONFIRMED_2026['name'] + '，Season 86）已经用户核实')
            r['flags'].append('EVOLUTION_CONFIRMED_2026: ' + EVO_CONFIRMED_2026['ability']
                              + f"｜Evolution Cycle={EVO_CONFIRMED_2026['evolution_cycle']}"
                              + '｜基础数值不变；evolvedSpellsData 待 L2｜SOURCE: '
                              + EVO_CONFIRMED_2026['source'])

    gnorms = {norm(e.get('englishName')) for e in gamedata} | {norm(e.get('name')) for e in gamedata}
    for c in cards_meta:
        if norm(c.get('name')) not in gnorms:
            registry.append({
                'internal_name': None, 'english_name': c.get('name'), 'cr_id': c.get('id'),
                'category': 'unknown', 'rarity_snapshot': None,
                'rarity_official': c.get('rarity'), 'elixir_snapshot': None,
                'elixir_official': c.get('elixirCost'),
                'elixir_canonical': c.get('elixirCost'), 'implemented': False,
                'summon_character': None, 'stats_resolved': False, 'basic_stats_lv11': {},
                'special_fields': [], 'has_evolution': False, 'has_ability': False,
                'status': 'missing_from_snapshot',
                'flags': ['MISSING_FROM_SNAPSHOT: Null服快照无此卡（官方新卡），'
                          '机制字段需 L2 提取；数值可先按官方录入'],
            })
    reg_ids = {r['cr_id'] for r in registry if r['cr_id']}
    unmatched_meta = [c.get('name') for c in cards_meta if c.get('id') not in reg_ids]
    return registry, unmatched_meta, meta_src, stats_src


def apply_smoke(registry):
    """合并 docs/batch_smoke_report.json（scripts/batch_smoke.py 产出）：
    - 给 registry 条目打 smoke_verified（构造→部署→30s 战斗有行为）
    - 冒烟通过的 data_ready 升级为 implemented（机制接入验证完成）
    - needs_review 不自动升级（数据置信度低，仍交人工）
    """
    report_path = os.path.join(DOCS, 'batch_smoke_report.json')
    if not os.path.exists(report_path):
        return registry, 0, 0
    with open(report_path) as f:
        report = json.load(f)
    ok = set(report.get('ok', []))
    marked = upgraded = 0
    for r in registry:
        internal = r.get('internal_name')
        if internal and internal in ok:
            r['smoke_verified'] = True
            marked += 1
            if r.get('status') == 'data_ready':
                r['status'] = 'implemented'
                r['implemented'] = True
                upgraded += 1
        else:
            r['smoke_verified'] = False
    return registry, marked, upgraded


def write_outputs(registry, unmatched_meta, meta_src, stats_src, smoke_note=''):
    os.makedirs(DOCS, exist_ok=True)
    policy = {
        'special_mechanics_baseline': 'nulls-royale-snapshot (gamedata.json, meta.fingerprint)',
        'numbers_baseline': 'official (meta: %s; per-level: %s)' % (meta_src, stats_src),
        'uncertainty_policy': 'flag-for-human-review, no silent guessing',
        'official_data_upgrade': 'curl cr-api-data -> src/clasher_new/data_official/（命令见脚本头）',
    }
    with open(os.path.join(DOCS, 'card_registry.json'), 'w', encoding='utf-8') as f:
        json.dump({'policy': policy, 'cards': registry}, f, ensure_ascii=False, indent=1)

    by = lambda s: [r for r in registry if r['status'] == s]
    statuses = ('implemented', 'data_ready', 'confirmed_live', 'needs_review', 'temporary_event',
                'variant', 'tower', 'missing_from_snapshot')
    counts = {s: len(by(s)) for s in statuses}
    lines = ['# 卡牌覆盖矩阵（P0-1 基本信息录入）', '',
             f'- 元数据来源：{meta_src}；per-level 数值来源：{stats_src}',
             f'- gamedata 快照条目：{len([r for r in registry if r["internal_name"]])}'
             f'（塔 {counts["tower"]}）',
             f'- 批量冒烟（scripts/batch_smoke.py）：全卡 构造→部署→30s 战斗有行为。{smoke_note}', '',
             '| 状态 | 数量 | 说明 |', '|---|---|---|',
             f'| implemented | {counts["implemented"]} | 已实现（冒烟接入 + 数值验证） |',
             f'| data_ready | {counts["data_ready"]} | 官方数值就绪，待机制接入 |',
             f'| needs_review | {counts["needs_review"]} | 拿不准，待人工决策 |',
             f'| temporary_event | {counts["temporary_event"]} | 活动/超级临时卡（默认排除） |',
             f'| variant | {counts["variant"]} | 变体/形态条目（排除，基础形态保留） |',
             f'| tower | {counts["tower"]} | 王塔/公主塔 |',
             f'| missing_from_snapshot | {counts["missing_from_snapshot"]} | 官方新卡，快照缺失 |',
             '', '## 需人工评审（模型拿不准）', '']
    for r in by('needs_review'):
        lines.append(f"- **{r['english_name'] or r['internal_name']}**"
                     f"（内部名 `{r['internal_name']}`，费={r['elixir_canonical']}）："
                     + '；'.join(r['flags']))
    lines += ['', '## 快照缺失的官方新卡', '']
    for r in by('missing_from_snapshot'):
        lines.append(f"- **{r['english_name']}** id={r['cr_id']} 费={r['elixir_official']}"
                     f" 稀有度={r['rarity_official']}")
    lines += ['', '## 临时/活动卡（默认排除，可人工恢复）', '']
    for r in by('temporary_event') + by('variant'):
        lines.append(f"- {r['internal_name']}（{r['english_name']}）")
    lines += ['', '## 官方元数据未对照（健康检查，按 id）', '']
    lines += [f'- {n}' for n in unmatched_meta] or ['（无）']
    with open(os.path.join(DOCS, 'card_coverage.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    return counts


if __name__ == '__main__':
    registry, unmatched, meta_src, stats_src = build_registry()
    registry, marked, upgraded = apply_smoke(registry)
    smoke_note = f'冒烟通过 {marked} 张，其中 {upgraded} 张 data_ready 升级为 implemented。' if marked else '未运行（缺 docs/batch_smoke_report.json）。'
    counts = write_outputs(registry, unmatched, meta_src, stats_src, smoke_note)
    print('元数据源:', meta_src)
    print('统计:', json.dumps(counts, ensure_ascii=False))
    print('冒烟:', smoke_note)
