@echo off
rem Build both TDGSA versions. Run from this folder (or double-click).
setlocal
cd /d "%~dp0"

echo === ANSI C (gcc -std=c89) ===
gcc -std=c89 -pedantic -Wall -Wextra -O2 tdgsa.c -o tdgsa_c.exe -lm || exit /b 1
echo built tdgsa_c.exe

echo === CUDA (nvcc + cuFFT) ===
rem vcvars64 can return a nonzero code from harmless warnings (e.g. vswhere not on PATH),
rem so check for cl.exe instead of trusting its exit code.
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
where cl >nul 2>&1 || (echo cl.exe not found: install MSVC Build Tools & exit /b 1)
nvcc -O3 -arch=sm_89 tdgsa.cu -o tdgsa_cuda.exe -lcufft || exit /b 1
echo built tdgsa_cuda.exe

echo Build OK.
