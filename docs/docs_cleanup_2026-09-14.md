# docs/ 中间文件清理（2026-09-14）

> 依据：用户指令「清除我们之前编写文档的中间文件」。
> 原则：**只删"可再生产、且不被任何文件引用"的中间产物**；判读文档的证据链（被引用的日志/json/npz）**一律保留**（【红线 R4】）。
> 保守引用检查：对每个候选文件，用其文件名**与去扩展名 stem** 在**全仓 739 个文本文件**（.md/.py/.json/.txt/.bat/.sh）里搜索，
> 命中即视为"被引用"⇒ 保留（该口径能挡住 `docs/train_{a,b}_20k.log` 这类花括号写法）。

| 组 | 文件数 | 释放 | 说明 |
|---|---|---|---|
| web 原始抓取素材（_page_/_elite_/_fandom_/_fp_/_hero_/_spell_/_evo_/_*_evo.txt + 页面级 .cdp*.js） | 190 | 6.88 MB | |
| 未被任何文件引用的日志（含 15 份过期全量 selftest 转储） | 39 | 5.71 MB | |
| 未被引用的 npz 中间产物 | 2 | 1.19 MB | |
| 代码摸底工具链的工作区（_survey/parts 51 份 + _survey/drafts 4 份） | 55 | 2.53 MB | |

**合计删除 286 个文件，释放 16.31 MB**（docs/ 由 25.6 MB → 约 9.3 MB）。

## 保留清单（为什么没删）

| 类别 | 数量/体积 | 理由 |
|---|---|---|
| 全部正式文档 `*.md` / `*.docx`（含 `full_code_reference.md` / `training_method.*` / `game_engine.*`） | 86 个 | 交付物本体 |
| **被引用的日志** `*.log` | 72 个 ≈1.8 MB | 判读文档的证据路径（【R4】禁手抄，必须可追溯到原始输出） |
| **被引用的 npz** `_diag_predict_fprime{,_ev}.npz` | 2 个 ≈1.17 MB | `fprime_rerun_20k_verdict` 的离线取证 |
| `_digest_*.md`（3）+ `_five_reports_merged_2026-09-13.md` | ≈64 KB | `plan_master.md` §6 的出处、`AGENTS.md` 引用 |
| `docs/_survey/*.json` + `_survey/verification_report_2026-09-14.md` | 5 个 ≈92 KB | `full_code_reference.md`「147 个 .py / 1479 符号 100% 覆盖」这条**结论的证据**（coverage/audit/reverse_check 报告） |
| `_elite_fandom_report.md` / `ab_valnorm_20k_verdict_*` 等 `_` 前缀**正文文档** | — | 是文档（含抓取过程记录），不是原料 |

## 如何重新生成被删掉的东西

```bash
# ① web 原始素材（CDP 抓取，脚本仍在 scripts/）
node scripts/cdp_extract.js            # 通用页面抽取（输出前缀见脚本内 L54 附近）
node scripts/cdp_evo.js                # 觉醒页抽取（输出 docs/_evo_<idx>.txt）
# ② 代码摸底工作区（工具链 scripts/_survey_*.py 完整保留）
python scripts/_survey_inventory.py && python scripts/_survey_groups.py \
  && python scripts/_survey_brief.py && python scripts/_survey_verify.py \
  && python scripts/_survey_audit.py && python scripts/_survey_merge.py
# ③ 过期全量 selftest 转储：按【红线 R19】不再产全量日志；
#    需要时只跑子集：scripts/run_selftests.py test_<名字>
```

## 副作用与对账

- 被删的 `docs/.cdp_extract_*.js` 曾在 `docs/full_code_reference.md` 的「非代码文件清单」里占 2 行（#5/#6）⇒ 已在该文档加清理注记。
- `docs/README.md` §3「原始采集数据」表已改为**已清理**状态（保留命名约定与引用方信息，便于重新抓取）。
- `AGENTS.md` §9 关于「素材 `docs/_survey/`」的表述已同步（json 报告保留，parts/drafts 已删）。
- `docs/_survey/parts/*` 是 `full_code_reference.md` 的**稿件碎片**，其内容已全文并入 `full_code_reference.md`（2.5 MB，未删）。

## ⚠️ 不可恢复项（诚实披露）

**删除集中 24 个文件此前未被 git 跟踪**（`docs/*.log` 在 `.gitignore` 内 ⇒ 无历史副本）：

```
_selftest_full_run.log  _selftest_full_run2.log  bench_pure.log  diag_ev_fprime_20k.log
diag_ev_fprime_probe.log  diag_predict_gfix.log  diag_vh_fprime_20k.log  e2_selftest.log
ft_bp_smoke.log  probe_norm.log  selftest_adv_inert.log  selftest_after_diagfix.log
selftest_eind.log  selftest_eind2.log  selftest_fprime.log  selftest_fprime2.log
selftest_fprime3.log  selftest_gfix.log  train_economy_10d.log  train_economy_9j.log
train_economy_9k.log  train_economy_9k_ft.log  train_gfix_20k.log  value_ln_probe_d1_alive.log
```

**这 24 个文件已永久消失**（无法从 git 取回）。删除前已用「文件名 + 去扩展名 stem 在 739 个文本文件里搜索」确认
**没有任何文件引用它们**，且其中 `selftest_*` / `_selftest_full_run*` 是过期全量自检转储（【红线 R19】后不再产出）。
其余被删文件都**在 git 里**：`git log --diff-filter=D --name-only -- docs/ | head` 可查，`git show <commit>^:<path>` 可取回。

## ⚠️ 清理顺带发现（2026-09-17 **已更正：原文结论是错的**）

> **更正留证**：下文 2026-09-14 写的「`runs/economy_9k_ft` 等目录已不存在」——**错**。
> 原因：我在**仓库根目录**跑 `find runs`，而所有命令实际都在 `src/clasher_new` 下执行
> ⇒ `runs/` 是 **cwd 相对路径**，真实位置是 **`src/clasher_new/runs/`**（4.7 GB，
> `economy_9k_ft` / `economy_9j` / `d1_long_100k` **都在**）。仓库根那个 `runs/`（2.1 GB，
> 含 `archive/economy_100k_v1_ungated` 的 53 个快照）是**另一份旧目录**。
> **实测反证**（2026-09-17 `demo20k` 试跑启动日志）：
> `[solo] 对手池: hist ckpts=12（其中 12 来自补种目录 ['runs/economy_9k_ft','runs/economy_9j']）`
> `mix frozen=0.1/hist=0.6/defend=0.2/rand_anchor=0.1` ⇒ **AGENTS §2.2/§2.3 的协议照抄即完整生效**。
> 原判据（保留原文，勿删）：

`runs/economy_9k_ft`、`runs/economy_9j`、`runs/economy_9k`、`runs/economy_10d`、`runs/gfix_20k`
**在磁盘上均不存在**（`find runs -maxdepth 3 -name "*economy_9*"` 为空；`runs/` 在 `.gitignore` 内 ⇒ 无 git 副本）。
⇒ `AGENTS.md` §2.2 / §2.3 的标准 20k / 100k 协议命令里的
`--hist-seed-dir runs/economy_9k_ft --hist-seed-dir runs/economy_9j` **照抄会因目录缺失而退化**
（对手池退化为「本 run 快照」，即文档配比 frozen/hist/defend/rand_anchor 失效）。
**未定**：这两个目录是被迁移/清理，还是从未入库。已在 `AGENTS.md` §2.2 加 ⚠️ 标注。

## 行尾规范化（**顺带发生，如实记录**）

本次编辑用 `newline='\n'` 重写了被改动的文档，于是两个**原本是 CRLF** 的文件被规范化成 LF：

| 文件 | 改动前 CR 行 | 改动后 | git 显示 |
|---|---|---|---|
| `docs/README.md` | 98 | 0（LF） | `104 98`（整文件重写） |
| `docs/full_code_reference.md` | 26419 | 0（LF） | `26421 26419`（整文件重写） |

**为什么保留 LF**：仓库 `docs/*.md` 里 **81/84 本来就是 LF**，这两个是少数派；`AGENTS.md` 也一直是 LF。
**已知副作用**：`scripts/_survey_merge.py` 第 555 行用默认文本模式写 `full_code_reference.md`
（Windows 下 = CRLF）⇒ **将来重新生成本文件会把行尾翻回 CRLF**，届时 diff 会再噪一次。
要根治需给 `_survey_merge.py` 加 `newline='\n'`（未做：属独立的代码改动，须按【R3】单变量走）。
