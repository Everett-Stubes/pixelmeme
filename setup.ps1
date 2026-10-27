# One-shot setup for testing pixelmeme on a Windows PC.
#
#   .\setup.ps1              create .venv and install everything
#   .\setup.ps1 -NoVenv      install into the current Python instead
#
# Safe to re-run. The Pi equivalent is install.sh.

[CmdletBinding()]
param(
    [switch]$NoVenv
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

Write-Host '==> python' -ForegroundColor Cyan
$cmd = Get-Command python -ErrorAction SilentlyContinue
if ($null -eq $cmd) {
    Write-Host 'No `python` on PATH. Install Python 3.9+ from python.org or the Microsoft Store.' -ForegroundColor Red
    exit 1
}
& $cmd.Source --version

if ($NoVenv) {
    $Python = $cmd.Source
    Write-Host "    installing into $Python (no venv)"
} else {
    $Python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (Test-Path $Python) {
        Write-Host '    .venv already exists'
    } else {
        Write-Host '    creating .venv'
        & $cmd.Source -m venv .venv
        if ($LASTEXITCODE -ne 0) { exit 1 }
    }
}

Write-Host '==> dependencies' -ForegroundColor Cyan
& $Python -m pip install --upgrade pip --quiet
& $Python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host 'pip failed. Check you are on 64-bit Python 3.9 or newer.' -ForegroundColor Red
    exit 1
}

# The writer. Windows has no C++ toolchain by default, so take the prebuilt
# CPU wheel rather than building; if neither works the app writes the show
# from the lexicon instead and says so once at startup.
$Index = 'https://abetlen.github.io/llama-cpp-python/whl/cpu'
Write-Host '==> on-device writer (llama-cpp-python)' -ForegroundColor Cyan
& $Python -m pip install 'llama-cpp-python>=0.3' --prefer-binary --only-binary=:all: `
    --extra-index-url $Index
$HaveWriter = ($LASTEXITCODE -eq 0)

# Installing is not the same as working. On the box this was built on, the
# newest wheel (0.3.36) installed cleanly and then died inside
# llama_backend_init with an access violation, while 0.3.2 ran fine. So probe
# it, and pin back to the known-good version if the probe fails.
if ($HaveWriter) {
    & $Python tools/probe_llm.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host '    that wheel does not run here; pinning to 0.3.2' -ForegroundColor Yellow
        & $Python -m pip install 'llama-cpp-python==0.3.2' --prefer-binary --only-binary=:all: `
            --extra-index-url $Index
        & $Python tools/probe_llm.py
        $HaveWriter = ($LASTEXITCODE -eq 0)
    }
}

if (-not $HaveWriter) {
    Write-Host '    no working wheel for this Python; the lexicon generator will write the show' -ForegroundColor Yellow
} else {
    Write-Host '==> the model (800MB, or 398MB on a small machine)' -ForegroundColor Cyan
    & $Python tools/get_model.py --auto
    if ($LASTEXITCODE -ne 0) {
        Write-Host '    download failed; run tools/get_model.py again when convenient' -ForegroundColor Yellow
    }
}

Write-Host ''
Write-Host 'Ready. Try:' -ForegroundColor Green
Write-Host '  .\run.ps1 --windowed --debug'
