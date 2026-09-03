# ===== L1 官方数据升级 v3（路径已由诊断确认：docs/json/）=====
$repo = "E:\clash-royale-simulator-main"
$dst  = Join-Path $repo "src\clasher_new\data_official"
New-Item -ItemType Directory -Force -Path $dst | Out-Null
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$base = "https://raw.githubusercontent.com/RoyaleAPI/cr-api-data/master/docs/json"
# 前 5 个是 coverage.py 需要的；后 2 个是白捡的（觉醒数据、多语言卡名，后续用得上）
$files = "cards.json","cards_stats_characters.json","cards_stats_building.json",
         "cards_stats_spell.json","cards_stats_projectile.json",
         "cards_evo.json","cards_i18n.json"

$ok = $true
foreach ($f in $files) {
    try {
        Invoke-WebRequest -Uri "$base/$f" -OutFile (Join-Path $dst $f) -UseBasicParsing -TimeoutSec 120
        "{0,-32} {1,10:N0} bytes" -f $f, (Get-Item (Join-Path $dst $f)).Length
    } catch {
        "下载失败 $f ：$($_.Exception.Message)"
        $ok = $false
    }
}

if ($ok) {
    "全部下载成功，重跑覆盖矩阵..."
    Set-Location $repo
    python scripts\coverage.py
} else {
    "有文件下载失败，把上面输出发给我"
}
