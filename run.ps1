# Start pixelmeme on Windows.
#
#   .\run.ps1 --windowed --debug     recommended for a first look
#   .\run.ps1                        fullscreen (Esc quits)
#
# Uses .venv if setup.ps1 created one, otherwise whatever `python` is on PATH.
# Anything you pass is handed straight to the app, so every flag in
# `.\run.ps1 --help` works here.

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$VenvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (Test-Path $VenvPython) {
    $Python = $VenvPython
} else {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($null -eq $cmd) {
        Write-Host "No Python found. Install Python 3.9+ or run .\setup.ps1" -ForegroundColor Red
        exit 1
    }
    $Python = $cmd.Source
}

# Fail with something readable rather than a traceback if a dependency is
# missing. The probe reports on stdout and never touches stderr: redirecting
# a native command's stderr in PowerShell 5.1 raises NativeCommandError even
# on success, which under ErrorActionPreference=Stop would kill this script.
$probe = & $Python -c "import os, sys
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
try:
    import pygame, yaml
except ImportError as exc:
    sys.stdout.write(str(exc))
    sys.exit(1)"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Missing dependencies: $probe" -ForegroundColor Yellow
    Write-Host "Run .\setup.ps1  (or: pip install -r requirements.txt)" -ForegroundColor Yellow
    exit 1
}

& $Python -m pixelmeme.main @args
exit $LASTEXITCODE
