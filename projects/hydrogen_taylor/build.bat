@echo off
rem Build and run the hydrogen Taylor-series programs: C (gcc, lint = warnings as errors) and CUDA (nvcc).
setlocal
cd /d "%~dp0"

echo === C11 + math.h (gcc lint: -Wall -Wextra -Wpedantic -Wconversion -Werror) ===
set "PATH=C:\msys64\mingw64\bin;%PATH%"
gcc -std=c11 -O2 -Wall -Wextra -Wpedantic -Wconversion -Werror taylor.c -o taylor_c.exe -lm || exit /b 1
echo built taylor_c.exe
"%~dp0taylor_c.exe" || exit /b 1

echo === CUDA (nvcc, sm_89) ===
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
where cl >nul 2>&1 || (echo cl.exe not found: install MSVC Build Tools & exit /b 2)
nvcc -O3 -arch=sm_89 taylor.cu -o taylor_gpu.exe || exit /b 3
echo built taylor_gpu.exe
"%~dp0taylor_gpu.exe" || exit /b 4

echo Build OK.
