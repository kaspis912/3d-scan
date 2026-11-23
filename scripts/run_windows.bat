@echo off
rem Start the scan3d web service at http://localhost:8000 (double-click this file).
rem Other port:  set PORT=8080  before starting.  Close this window to stop the service.
setlocal
cd /d "%~dp0.."
set PYTHONUTF8=1
if "%PORT%"=="" set PORT=8000
if "%HOST%"=="" set HOST=127.0.0.1
set "PATH=%USERPROFILE%\.local\bin;%PATH%"

where uv >nul 2>nul
if errorlevel 1 (
  echo uv is not installed yet. Run scripts\install_windows.bat first.
  pause
  exit /b 1
)

echo Starting scan3d at http://%HOST%:%PORT%
echo Keep this window open while you scan. Close it to stop the service.
rem Open the page once the server has had a few seconds to come up.
start "" /b cmd /c "timeout /t 6 /nobreak >nul & start http://%HOST%:%PORT%"
rem SCAN3D_NO_SYNC=1 keeps hand-installed packages (e.g. AMD's ROCm build of PyTorch) instead of
rem restoring the locked versions on every start.
set UVFLAGS=
if "%SCAN3D_NO_SYNC%"=="1" set UVFLAGS=--no-sync
uv run %UVFLAGS% --extra masking --extra silhouette uvicorn app.main:app --host %HOST% --port %PORT%
echo.
echo The service has stopped.
pause
