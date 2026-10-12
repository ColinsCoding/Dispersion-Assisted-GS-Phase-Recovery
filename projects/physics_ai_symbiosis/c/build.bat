@echo off
rem Build all four ANSI C89 programs into ..\out\r
cd /d "%~dp0"
if not exist ..\out mkdir ..\out
for %%f in (kalman nlse_kernel pst discover) do gcc -std=c89 -pedantic -Wall -Wextra -O2 %%f.c -o ..\out\%%f.exe -lm || exit /b 1
echo Build OK.
