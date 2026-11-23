@echo off
rem Installs scan3d and everything it needs (double-click this file). Safe to run again.
setlocal
cd /d "%~dp0.."
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_windows.ps1" %*
set RC=%ERRORLEVEL%
echo.
if not "%RC%"=="0" (
  echo The installation did not finish. Read the message above, fix it and run this file again.
) else (
  echo Installation finished.
)
pause
exit /b %RC%
