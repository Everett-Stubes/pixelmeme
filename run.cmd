@echo off
rem Double-clickable launcher. Bypasses the PowerShell execution policy,
rem which blocks unsigned .ps1 files by default on Windows.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
