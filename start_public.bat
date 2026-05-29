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
where cloudflared >nul 2>&1 && set CF=cloudflared
if not defined CF (
  if exist "%LOCALAPPDATA%\Microsoft\WinGet\Packages\Cloudflare.cloudflared*\cloudflared.exe" (
    for /f "delims=" %%f in ('dir /s /b "%LOCALAPPDATA%\Microsoft\WinGet\Packages\Cloudflare.cloudflared*\cloudflared.exe" 2^>nul') do set CF=%%f
  )
)

if not defined CF (
  echo  cloudflared not found.  Installing via winget...
  winget install --id Cloudflare.cloudflared --silent --accept-package-agreements --accept-source-agreements
  where cloudflared >nul 2>&1 && set CF=cloudflared
)

if not defined CF (
  echo.
  echo  [ERROR] cloudflared still not found.
  echo  Download manually from:
  echo    https://github.com/cloudflare/cloudflared/releases/latest
  echo  Place cloudflared.exe in this folder and re-run.
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
