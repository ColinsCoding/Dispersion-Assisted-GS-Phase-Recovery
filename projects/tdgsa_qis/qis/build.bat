@echo off
rem Build the quantum GS programs: C++/POSIX threads (g++) and CUDA (nvcc).
setlocal
cd /d "%~dp0"

echo === C++17 + POSIX threads (g++) ===
g++ -std=c++17 -O2 -Wall -Wextra -pthread qgs.cpp -o qgs_cpu.exe || exit /b 1
echo built qgs_cpu.exe

echo === CUDA (nvcc + cuFFT + cuRAND) ===
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
where cl >nul 2>&1 || (echo cl.exe not found: install MSVC Build Tools & exit /b 1)
nvcc -O3 -std=c++17 -arch=sm_89 -diag-suppress 20199 qgs.cu -o qgs_gpu.exe -lcufft -lcurand || exit /b 1
echo built qgs_gpu.exe

echo Build OK.
