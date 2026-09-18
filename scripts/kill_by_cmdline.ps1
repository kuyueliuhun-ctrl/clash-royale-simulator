# Find (and optionally kill) python.exe processes whose command line contains a marker.
#
# Usage:
#   powershell -File kill_by_cmdline.ps1 -Marker et_ctrl100k            # kill matches
#   powershell -File kill_by_cmdline.ps1 -Marker et_ctrl100k -DryRun    # list only
#
# Exit codes:
#   0 = at least one match (killed, or listed with -DryRun)
#   1 = no match  ==> callers may use this as a LIVENESS PREDICATE
#
# WHY A FILE INSTEAD OF INLINE -Command:
#   Passing the filter inline from bash mangles `$_` into `\$`, so the filter silently
#   matches nothing and always reports zero processes -- i.e. a FALSE NEGATIVE for
#   "are there orphan workers?".  Hit for real on 2026-09-18; evidence in
#   docs/et_solo100k_2026-09-18.md section 5.  Calling with -File avoids it entirely.
#
# WHY ASCII ONLY:
#   Windows PowerShell 5.1 reads .ps1 as ANSI/GBK when there is no BOM.  Non-ASCII
#   comments then decode into bytes that break the parser (observed: ParserError at
#   `param(`).  Keep this file ASCII; put prose in the docs.
#
# WHY `cmd.exe /c "powershell ..."`:
#   WSL bash cannot resolve a bare `powershell` (no such filename); callers must go
#   through cmd.exe.  Verified working 3/3.
param(
    [Parameter(Mandatory=$true)][string]$Marker,
    [switch]$DryRun
)

$hits = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -and ($_.CommandLine -like "*$Marker*") })

if ($hits.Count -eq 0) { Write-Output "NONE"; exit 1 }

foreach ($h in $hits) {
    $cl = ($h.CommandLine -replace '\s+', ' ')
    if ($cl.Length -gt 120) { $cl = $cl.Substring(0, 120) }
    if ($DryRun) {
        Write-Output ("FOUND {0} {1}" -f $h.ProcessId, $cl)
    } else {
        $r = & taskkill /F /T /PID $h.ProcessId 2>&1
        Write-Output ("KILLED {0} {1} :: {2}" -f $h.ProcessId, $cl, ($r -join ' '))
    }
}
if (-not $DryRun) { Write-Output ("TOTAL_KILLED {0}" -f $hits.Count) }
exit 0
