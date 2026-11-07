@echo off
rem Build, test and run the droid (2D) and the 3D battle on Windows (nvcc + MSVC + cuFFT). Exit codes:
rem   1 = a build failed, 2 = tests failed, 4 = a game run failed.
setlocal
cd /d "%~dp0"
if not exist out mkdir out
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
where cl >nul 2>&1 || (echo cl.exe not found: install MSVC Build Tools & exit /b 1)
set NVCC=nvcc -O2 -std=c++17 -arch=sm_89 -DDROID_CUDA -Iinclude -diag-suppress 20199 -Xcompiler "/EHsc /W3 /utf-8"
set CORE=src\physics.cpp src\vision_cpu.cpp src\vision_gpu.cu
set G3=src\defs.cpp src\render3d.cu
%NVCC% %CORE% tests\test_droid.cpp -o out\test_droid.exe -lcufft || exit /b 1
%NVCC% %CORE% %G3% tests\test_scene3d.cpp -o out\test_scene3d.exe -lcufft || exit /b 1
%NVCC% %CORE% src\main.cpp -o out\droid.exe -lcufft || exit /b 1
%NVCC% %CORE% %G3% src\main3d.cpp -o out\droid3d.exe -lcufft || exit /b 1
%NVCC% %CORE% %G3% src\play3d.cpp -o out\droid_play.exe -lcufft -luser32 -lgdi32 -lwinmm || exit /b 1
"%~dp0out\test_droid.exe" || exit /b 2
"%~dp0out\test_scene3d.exe" || exit /b 2
"%~dp0out\droid3d.exe" --frames 900 %* || exit /b 4
echo Build OK.
