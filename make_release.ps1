# make_release.ps1 - Build SokuAdvisor.exe and pack it with readme.txt into a zip for distribution
#
#   .\make_release.ps1 -Version 1.5
#
# Output: release\SokuAdvisor_v1.5.zip (SokuAdvisor.exe + readme.txt)

param(
    [Parameter(Mandatory = $true)][string]$Version,
    [switch]$SkipBuild
)

Set-Location $PSScriptRoot

# The version in readme.txt must match, so the zip never ships with a stale readme
$readme = Get-Content "readme.txt" -Encoding UTF8 -TotalCount 3 | Out-String
if ($readme -notmatch "v$([regex]::Escape($Version))\b") {
    Write-Error "readme.txt does not mention v$Version in its title. Update readme.txt first."
    exit 1
}

if (-not $SkipBuild) {
    if (Get-Process SokuAdvisor -ErrorAction SilentlyContinue) {
        Write-Error "SokuAdvisor is running. Close it before building."
        exit 1
    }
    & powershell -NoProfile -ExecutionPolicy Bypass -File ".\build_exe.ps1"
}

$exe = Join-Path $PSScriptRoot "dist\SokuAdvisor.exe"
if (-not (Test-Path $exe)) {
    Write-Error "dist\SokuAdvisor.exe not found"
    exit 1
}

$outDir = Join-Path $PSScriptRoot "release"
New-Item -ItemType Directory -Force $outDir | Out-Null
$zip = Join-Path $outDir "SokuAdvisor_v$Version.zip"
if (Test-Path $zip) { Remove-Item $zip }

# readme.txt goes out as UTF-8 with BOM so that Notepad on any Windows shows it correctly
$stage = Join-Path $outDir "_stage"
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force $stage | Out-Null
Copy-Item $exe $stage
$text = [IO.File]::ReadAllText((Join-Path $PSScriptRoot "readme.txt"), [Text.Encoding]::UTF8)
[IO.File]::WriteAllText((Join-Path $stage "readme.txt"), $text, (New-Object Text.UTF8Encoding($true)))

Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $zip
Remove-Item -Recurse -Force $stage

$item = Get-Item $zip
Write-Host "====================================="
Write-Host ("Release: {0} ({1:N1} MB)" -f $item.FullName, ($item.Length / 1MB))
Write-Host ("SHA256 : {0}" -f (Get-FileHash $zip -Algorithm SHA256).Hash)
Write-Host "====================================="
