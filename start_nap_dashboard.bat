@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_nap_dashboard.ps1"
if errorlevel 1 pause
