# Kill only python.exe processes whose command line contains a given marker.
# Usage: powershell -File _kill_by_cmdline.ps1 -Marker et_ctrl100k
# Prints "KILLED <pid> <cmdline-head>" per kill, or "NONE" when nothing matched.
param([Parameter(Mandatory=$true)][string]$Marker)

$hits = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -and ($_.CommandLine -like "*$Marker*") }

if (-not $hits) { Write-Output "NONE"; exit 1 }

$killed = 0
foreach ($h in $hits) {
    $cl = ($h.CommandLine -replace '\s+', ' ')
    if ($cl.Length -gt 120) { $cl = $cl.Substring(0, 120) }
    $r = & taskkill /F /T /PID $h.ProcessId 2>&1
    Write-Output ("KILLED {0} {1} :: {2}" -f $h.ProcessId, $cl, ($r -join ' '))
    $killed++
}
Write-Output "TOTAL_KILLED $killed"
exit 0
