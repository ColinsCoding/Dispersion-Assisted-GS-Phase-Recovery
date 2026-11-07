#!/usr/bin/env bash
# Build and run the hydrogen Taylor-series programs on Linux / WSL: C (gcc, lint = warnings as
# errors) and CUDA (nvcc). Linux twin of build.bat. Each step checks $? (the exit code of the
# last command: 0 = success) and stops with its own code on failure:
#   1 = gcc build, 2 = C run, 3 = nvcc missing, 4 = nvcc build, 5 = GPU run.
# GPU architecture: ARCH=sm_89 by default (RTX 4060); override, e.g.  ARCH=sm_75 ./build.sh
cd "$(dirname "$0")" || exit 1
ARCH="${ARCH:-sm_89}"

echo "=== C11 + math.h (gcc lint: -Wall -Wextra -Wpedantic -Wconversion -Werror) ==="
gcc -std=c11 -O2 -Wall -Wextra -Wpedantic -Wconversion -Werror taylor.c -o taylor_c -lm
status=$?
if [ $status -ne 0 ]; then echo "gcc build failed (\$? = $status)"; exit 1; fi
echo "built taylor_c"
./taylor_c
status=$?
if [ $status -ne 0 ]; then echo "taylor_c failed (\$? = $status)"; exit 2; fi

echo "=== CUDA (nvcc, $ARCH) ==="
command -v nvcc >/dev/null
if [ $? -ne 0 ]; then echo "nvcc not found: install the CUDA toolkit (e.g. nvidia-cuda-toolkit)"; exit 3; fi
nvcc -O3 -arch="$ARCH" taylor.cu -o taylor_gpu
status=$?
if [ $status -ne 0 ]; then echo "nvcc build failed (\$? = $status)"; exit 4; fi
echo "built taylor_gpu"
./taylor_gpu
status=$?
if [ $status -ne 0 ]; then echo "taylor_gpu failed (\$? = $status)"; exit 5; fi

echo "Build OK."
