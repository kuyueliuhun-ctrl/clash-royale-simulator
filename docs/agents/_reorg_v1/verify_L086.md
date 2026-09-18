row: L086
target_docs: docs/upstream_engine_delta_2026-09-19.md; docs/upstream_four_sections_merge_2026-09-19.md; docs/upstream_claim_audit_2026-09-19.md
counts: PRESENT=18 PARTIAL=0 MISSING=0 INDEX-ONLY=2
MISSING/PARTIAL 清单：
none
关键证据（≤5 条）：
- 分叉点 f20fa4d 两仓 rc=0 / 242 vs 24 → upstream_engine_delta_2026-09-19.md:44-48「rc=0」「24」「242」
- 退出码陷阱 rc=128 + fatal: Not a valid object name → 同上:57-62
- 24 提交中 6 动引擎 / 4 文件 / 净 +52 −33 → 同上:14
- P3 闩锁 y>17 / [15,17] / 贴脸 18 s 承伤 0 / 上游死代码 → 同上:34,378,421
- P2 16 半格 x=2.25/4.75/13.25/15.75 各 4 行 → 同上:311
其余逐条（D=upstream_engine_delta_2026-09-19.md，M=upstream_four_sections_merge_2026-09-19.md）：
- ① 372ed0e 已独立修好 card_utils.py:304-306 + battle.py:1357-1360 → D:164-165
- ② *0.5 未跟进 / battle.py:3299-3303 无 / 除零+塔矩形推出 / 先测 → D:21,140,144-145
- ③ 20 Hz BREAKING 未跟 / env_wrapper.py:88-89=0.5 s 等价 → D:23,198
- ④ 不照搬 / 上游 jump 死代码 / 我方活 1187-1190、1233-1239 → D:25-26,229,257-258
- ⑤ 1e9716b 仅注释 → D:24,209
- 引擎数据与基座零改动 → D:15
- 非引擎侧零影响 grep environment=0 → D:474,540
- P1 battle.py:3292 return 应 continue / 上游逐字相同未修 → D:32,300
- P2 W=800 而 arena.py:113-116 可走 / 3 格实 2 格 / 上游 W→50 / 根因未修 → D:311,335,341
- P4 walkable_cache 全局+int(x) 截断 / (0,y)、(6..11,0) / 跨局 / 上游同构未修 → D:433-446
- 本轮一行未改待拍板 → D:549
- 与 claim_audit 并列不调和 / 两说 / §1.2 → D:65-76；claim_audit:232
- 四稿 4 处截断已在 §0.2 标注 → M:26-33
- INDEX-ONLY: 行首两处文档指针（全文→、合并件→）为索引功能，无需在目标文档存在
