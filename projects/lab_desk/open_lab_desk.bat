@echo off
rem Opens the lab-desk simulator. Double-click this file, or run it from a terminal.
cd /d "%~dp0"
py -3.12 lab_desk.py
if errorlevel 1 (
    echo.
    echo The desk did not start. Check that Python 3.12 has pygame and numpy:
    echo     py -3.12 -m pip install pygame numpy scipy nbformat
    pause
)
