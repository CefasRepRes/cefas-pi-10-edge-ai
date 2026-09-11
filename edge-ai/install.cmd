@echo off
setlocal
cd /d "%~dp0"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1"
if errorlevel 1 (
  echo.
  echo INSTALL FAILED. See messages above.
  pause
  exit /b 1
)

echo.
echo Install complete.
pause