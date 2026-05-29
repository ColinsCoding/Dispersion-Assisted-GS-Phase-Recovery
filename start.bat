@echo off
setlocal enabledelayedexpansion
title Jalali Lab — Optical Dashboard
cd /d "%~dp0"

echo.
echo  +------------------------------------------+
echo  ^|  JALALI LAB  OPTICAL DASHBOARD           ^|
echo  ^|  Flask + TD-GS Phase Recovery            ^|
echo  +------------------------------------------+
echo.

REM ── Find Python ──────────────────────────────────────────────────
set PY=
for %%p in (
  "C:\Users\mrjel\AppData\Local\Programs\Python\Python312\python.exe"
  "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
  "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
  "python"
) do (
  if not defined PY (
    %%p --version >nul 2>&1 && set PY=%%~p
  )
)

if not defined PY (
  echo  [ERROR] Python not found.
  echo  Install from: https://www.python.org/downloads/
  pause
  exit /b 1
)
echo  Python : %PY%

REM ── Install requirements if needed ───────────────────────────────
if not exist "optical_dashboard\uploads\" mkdir "optical_dashboard\uploads"
%PY% -c "import flask,numpy,scipy,matplotlib" >nul 2>&1
if errorlevel 1 (
  echo  Installing requirements...
  %PY% -m pip install -q flask numpy scipy matplotlib
)

REM ── Start Flask in a minimised background window ─────────────────
echo  [1/2] Starting Flask server on http://localhost:5000
start "Jalali-Flask" /min cmd /c "%PY% optical_dashboard\app.py"

REM ── Wait for server to come up ────────────────────────────────────
echo  Waiting for server...
:wait
timeout /t 1 /nobreak >nul
%PY% -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" >nul 2>&1
if errorlevel 1 goto wait

REM ── Open browser ─────────────────────────────────────────────────
echo  [2/2] Opening http://localhost:5000
start http://localhost:5000

echo.
echo  Dashboard is running.
echo  Close this window to STOP the server.
echo.
echo  To share publicly:  run  start_public.bat
echo.
pause >nul

REM ── Stop Flask on exit ────────────────────────────────────────────
echo  Stopping server...
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":5000 "') do (
  taskkill /F /PID %%a >nul 2>&1
)
echo  Done.
