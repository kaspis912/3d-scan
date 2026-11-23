@echo off
rem Check the installation (double-click). Add --selftest for a complete test scan:
rem   scripts\doctor_windows.bat --selftest
setlocal
cd /d "%~dp0.."
set PYTHONUTF8=1
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
uv run --extra masking --extra silhouette scan3d-doctor %*
echo.
pause
