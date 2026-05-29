@echo off
setlocal enabledelayedexpansion
title Jalali Lab — Public URL
cd /d "%~dp0"

echo.
echo  +------------------------------------------+
echo  ^|  JALALI LAB  PUBLIC TUNNEL               ^|
echo  ^|  Flask + cloudflared  (internet URL)     ^|
echo  +------------------------------------------+
echo.

REM ── Find Python ──────────────────────────────────────────────────
set PY=
for %%p in (
  "C:\Users\mrjel\AppData\Local\Programs\Python\Python312\python.exe"
  "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
  "python"
) do (
  if not defined PY (
    %%p --version >nul 2>&1 && set PY=%%~p
  )
)
if not defined PY ( echo [ERROR] Python not found. & pause & exit /b 1 )

REM ── Find cloudflared ─────────────────────────────────────────────
set CF=
REM 1. Already on PATH
where cloudflared >nul 2>&1 && set CF=cloudflared
REM 2. Hardcoded install paths — avoid for-loop + parentheses bug
if not defined CF (
  set "_C1=C:\Program Files (x86)\cloudflared\cloudflared.exe"
  if exist "C:\Program Files (x86)\cloudflared\cloudflared.exe" set "CF=C:\Program Files (x86)\cloudflared\cloudflared.exe"
)
if not defined CF (
  if exist "C:\Program Files\cloudflared\cloudflared.exe" set "CF=C:\Program Files\cloudflared\cloudflared.exe"
)
if not defined CF (
  if exist "%~dp0cloudflared.exe" set "CF=%~dp0cloudflared.exe"
)

if not defined CF (
  echo  cloudflared not found.  Installing via winget...
  winget install --id Cloudflare.cloudflared --silent --accept-package-agreements --accept-source-agreements

  REM winget adds to PATH but this session has not refreshed — probe known locations
  for %%d in (
    "%ProgramFiles%\cloudflared\cloudflared.exe"
    "%ProgramFiles(x86)%\cloudflared\cloudflared.exe"
    "%LOCALAPPDATA%\Microsoft\WinGet\Packages\Cloudflare.cloudflared_Microsoft.Winget.Source_8wekyb3d8bbwe\cloudflared.exe"
    "%SystemDrive%\cloudflared\cloudflared.exe"
    "%~dp0cloudflared.exe"
  ) do (
    if not defined CF (
      if exist %%d set CF=%%d
    )
  )

  REM Also re-try where after refreshing PATH from registry
  if not defined CF (
    for /f "usebackq delims=" %%p in (
      `powershell -NoProfile -Command "[System.Environment]::GetEnvironmentVariable('PATH','Machine') -split ';' | ForEach-Object { $f = Join-Path $_ 'cloudflared.exe'; if (Test-Path $f) { $f } }" 2^>nul`
    ) do ( if not defined CF set CF=%%p )
  )
)

if not defined CF (
  echo.
  echo  [ERROR] cloudflared installed but not found in PATH yet.
  echo  Close this window, open a new terminal, and run:
  echo    .\start_public.bat
  echo  (winget PATH changes require a new session to take effect)
  echo.
  echo  OR download cloudflared.exe manually and place it here:
  echo    https://github.com/cloudflare/cloudflared/releases/latest
  echo.
  pause
  exit /b 1
)
echo  cloudflared: %CF%

REM ── Start Flask ───────────────────────────────────────────────────
if not exist "optical_dashboard\uploads\" mkdir "optical_dashboard\uploads"
echo  Starting Flask on http://localhost:5000...
start "Jalali-Flask" /min cmd /c "%PY% optical_dashboard\app.py"

:wait
timeout /t 1 /nobreak >nul
%PY% -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" >nul 2>&1
if errorlevel 1 goto wait

REM ── Start cloudflared — URL printed to this console ───────────────
echo.
echo  ============================================================
echo   PUBLIC URL will appear below in a few seconds.
echo   Share that  https://xxxx.trycloudflare.com  link.
echo   Works as long as THIS WINDOW stays open.
echo  ============================================================
echo.
%CF% tunnel --url http://localhost:5000

REM ── Cleanup on Ctrl-C ─────────────────────────────────────────────
echo.
echo  Tunnel closed.  Stopping Flask...
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":5000 "') do (
  taskkill /F /PID %%a >nul 2>&1
)
echo  Done.
