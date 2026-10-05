# build_exe.ps1 - Build SokuAdvisor.exe with PyInstaller

Set-Location $PSScriptRoot

# Prefer tensoku_rep_movier venv (has cv2 + PyInstaller)
$toolsRoot = Split-Path $PSScriptRoot -Parent
$preferred = Join-Path $toolsRoot "tensoku_rep_movier\.venv\Scripts\python.exe"

if (Test-Path $preferred) {
    $python = $preferred
} else {
    # Fallback: find any venv python that has PyInstaller
    $python = Get-ChildItem $toolsRoot -Recurse -Filter "python.exe" -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -like "*.venv*Scripts*" } |
        Where-Object {
            $hasPI = & $_.FullName -m PyInstaller --version 2>$null
            $LASTEXITCODE -eq 0
        } |
        Select-Object -First 1 -ExpandProperty FullName
}

if (-not $python) {
    Write-Error "python.exe with PyInstaller not found"
    exit 1
}

Write-Host "Python: $python"
Write-Host "Building..."

if (Test-Path "dist")  { Remove-Item -Recurse -Force "dist" }
if (Test-Path "build") { Remove-Item -Recurse -Force "build" }

& $python -m PyInstaller `
    --onefile `
    --windowed `
    --name "SokuAdvisor" `
    --add-data "analyzer.py;." `
    --add-data "soku_live_reader.py;." `
    --add-data "char_advisor.py;." `
    --add-data "ai_advisor.py;." `
    --add-data "char_data.json;." `
    --add-data "chart.umd.min.js;." `
    --add-data "soku_advisor.ico;." `
    --manifest "app.manifest" `
    --icon "soku_advisor.ico" `
    --hidden-import cv2 `
    --hidden-import numpy `
    --hidden-import tkinter `
    --hidden-import player_history `
    soku_advisor_app.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "====================================="
    Write-Host "Build succeeded!"
    Write-Host "EXE: $PSScriptRoot\dist\SokuAdvisor.exe"
    Write-Host "====================================="
} else {
    Write-Host "Build failed (code $LASTEXITCODE)"
}
