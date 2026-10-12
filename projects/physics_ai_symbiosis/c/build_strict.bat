@echo off
rem Strict build of every program: CPU with gcc (ANSI C89, warnings are errors),
rem GPU with nvcc (all device warnings are errors, MSVC host warnings at /W4 as errors).
setlocal
cd /d "%~dp0"
if not exist ..\out mkdir ..\out

echo === CPU: gcc -std=c89 -pedantic -Wall -Wextra -Werror ===
for %%f in (kalman nlse_kernel pst discover) do (
    gcc -std=c89 -pedantic -Wall -Wextra -Werror -O2 %%f.c -o ..\out\%%f.exe -lm || exit /b 1
    echo   built %%f.exe
)

echo === GPU: nvcc -Werror all-warnings ===
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
where cl >nul 2>&1 || (echo cl.exe not found: install MSVC Build Tools & exit /b 1)
for %%f in (nlse_cuda pst_cuda) do (
    nvcc -O3 -std=c++17 -arch=sm_89 -Werror all-warnings -Xcompiler "/W4 /WX" %%f.cu -o ..\out\%%f.exe -lcufft || exit /b 1
    echo   built %%f.exe
)
echo Build OK.
