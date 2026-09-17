<#
.SYNOPSIS
  列出 / 清理【R1】的「孤儿 spawn worker」（父进程已死的 `--multiprocessing-fork` 进程）。

.DESCRIPTION
  为什么需要这个工具：`--eval-workers N` 走 `multiprocessing` spawn，worker 是**独立进程**。
  训练进程被强杀（Ctrl+C 不成、job_kill、shell 超时）时，**worker 不会跟着死** ⇒ 变成孤儿，
  每个占 ~0.55 GB 提交内存，且 `tasklist` 里看不出区别。
  2026-09-18 实测：**14 个孤儿（12 个来自 long1m 的 12 worker + 2 个来自一次 smoke）占 ~7.7 GB**，
  把可用提交从 ~23 GB 压到 **7.51 GB**，直接触发 `check_commit.py` 的 `[WARN] < 12 GB ⇒ 降档`；
  清掉后回到 **23.24 GB**、档位判回 `[OK] 12 档可用`。
  ⇒ **长跑开跑前先跑这个**（与 `scripts/check_commit.py` 配对使用；见【红线 R1】）。

  安全性：只杀**父进程已确认不存在**的 fork worker；父进程还活着的会打印 `SKIP` 并跳过。
  默认 dry-run（只列不杀）。

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\kill_orphan_workers.ps1
  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\kill_orphan_workers.ps1 -Kill
#>
param(
  [switch]$Kill
)

$procs = Get-CimInstance Win32_Process -Filter "name='python.exe'"
$forks = $procs | Where-Object { $_.CommandLine -match 'multiprocessing-fork' }

Write-Output ("python.exe 总数 = " + @($procs).Count + " | fork worker = " + @($forks).Count)
if (@($forks).Count -eq 0) { Write-Output "无 fork worker。"; exit 0 }

$orphans = @()
foreach ($w in $forks) {
  $m = [regex]::Match($w.CommandLine, 'parent_pid=(\d+)')
  $ppid = if ($m.Success) { [int]$m.Groups[1].Value } else { -1 }
  $alive = Get-Process -Id $ppid -ErrorAction SilentlyContinue
  $isOrphan = ($null -eq $alive)
  if ($isOrphan) { $orphans += $w.ProcessId }
  Write-Output ("  PID " + $w.ProcessId + "  parent=" + $ppid + "  parentAlive=" + (-not $isOrphan) + $(if ($isOrphan) {"  <== ORPHAN"} else {"  (skip)"}))
}

Write-Output ("孤儿 = " + @($orphans).Count)
if ($Kill) {
  foreach ($pid2 in $orphans) {
    Stop-Process -Id $pid2 -Force -ErrorAction SilentlyContinue
    Write-Output ("  killed " + $pid2)
  }
  Start-Sleep -Seconds 3
  $left = (Get-CimInstance Win32_Process -Filter "name='python.exe'" | Where-Object { $_.CommandLine -match 'multiprocessing-fork' })
  Write-Output ("剩余 fork worker = " + @($left).Count)
  Write-Output "下一步：.venv\Scripts\python.exe scripts\check_commit.py"
} else {
  Write-Output "（dry-run；加 -Kill 执行清理）"
}
