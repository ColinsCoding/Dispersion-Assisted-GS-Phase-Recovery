@echo off
REM ─────────────────────────────────────────────────────────────────────────
REM  start_tunnel.bat  —  double-click to start Flask + Cloudflare Tunnel
REM  Requirements: Python 3, cloudflared on PATH
REM    Install cloudflared: winget install --id Cloudflare.cloudflared
REM ─────────────────────────────────────────────────────────────────────────
cd /d "%~dp0"
C:\Users\mrjel\AppData\Local\Programs\Python\Python312\python.exe start_tunnel.py
pause
