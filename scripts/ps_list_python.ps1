# Diagnostic: list python.exe processes with parent, CPU time and command line head.
# ASCII only (PowerShell 5.1 reads .ps1 as ANSI/GBK without a BOM -- see docs/et_solo100k_2026-09-18.md).
$now = Get-Date
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Sort-Object CreationDate |
    ForEach-Object {
        $cl = $_.CommandLine
        if ($cl -eq $null) { $cl = "" }
        $short = ($cl -replace '\s+', ' ')
        if ($short.Length -gt 90) { $short = $short.Substring(0, 90) }
        $cpu = [math]::Round(($_.KernelModeTime + $_.UserModeTime) / 10000000.0, 1)
        $age = [math]::Round(($now - $_.CreationDate).TotalSeconds, 0)
        "{0}|ppid={1}|cpu={2}s|age={3}s|{4}" -f $_.ProcessId, $_.ParentProcessId, $cpu, $age, $short
    }
