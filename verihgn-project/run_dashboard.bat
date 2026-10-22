@echo off
rem Offline HEADPATH dashboard. Usage:  run_dashboard.bat "C:\path\to\headpath_dashboard_<timestamp>.zip"
rem With no argument the newest headpath_dashboard_*.zip in your Downloads folder is used.
rem Needs only Python 3.10+ (standard library). Paths with spaces are supported (keep the quotes).
setlocal
set "HERE=%~dp0"
set "PY=python"
if exist "%HERE%.venv\Scripts\python.exe" set "PY=%HERE%.venv\Scripts\python.exe"
"%PY%" "%HERE%tools\run_dashboard.py" %*
if errorlevel 1 pause
endlocal
