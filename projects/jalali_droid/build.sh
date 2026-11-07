#!/usr/bin/env bash
# Build, test and run the droid (2D) and the 3D battle on Linux / WSL. Checks $? after each step:
#   1 = a build failed, 2 = tests failed, 4 = a game run failed.
# CUDA (nvcc + cuFFT) if available, else a CPU-only g++ build.  ARCH=sm_89 by default.
cd "$(dirname "$0")" || exit 1
ARCH="${ARCH:-sm_89}"
mkdir -p out
CORE="src/physics.cpp src/vision_cpu.cpp"
G3="src/defs.cpp"
if command -v nvcc >/dev/null; then
    echo "=== CUDA build (nvcc, $ARCH, cuFFT) ==="
    CXX="nvcc -O2 -std=c++17 -arch=$ARCH -DDROID_CUDA -Iinclude"
    GPU="src/vision_gpu.cu"; GPU3="src/render3d.cu"; LIBS="-lcufft"
else
    echo "=== CPU-only build (g++; nvcc not found) ==="
    CXX="g++ -O2 -std=c++17 -Wall -Wextra -Iinclude"
    GPU=""; GPU3=""; LIBS=""
fi
build() { $CXX "$@" $LIBS; s=$?; if [ $s -ne 0 ]; then echo "build failed (\$? = $s): $*"; exit 1; fi; }
build $CORE $GPU tests/test_droid.cpp -o out/test_droid
build $CORE $GPU $G3 $GPU3 tests/test_scene3d.cpp -o out/test_scene3d
build $CORE $GPU src/main.cpp -o out/droid
build $CORE $GPU $G3 $GPU3 src/main3d.cpp -o out/droid3d
for t in out/test_droid out/test_scene3d; do
    ./$t; s=$?; if [ $s -ne 0 ]; then echo "$t: $s check(s) failed (\$? = $s)"; exit 2; fi
done
./out/droid3d --frames 900 "$@"
s=$?; if [ $s -ne 0 ]; then echo "droid3d exited with \$? = $s"; exit 4; fi
echo "Build OK."
