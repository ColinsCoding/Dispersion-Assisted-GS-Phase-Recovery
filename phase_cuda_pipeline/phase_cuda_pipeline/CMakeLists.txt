cmake_minimum_required(VERSION 3.22)
project(phase_recovery_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES native)

find_package(CUDAToolkit REQUIRED)
include_directories(include)

add_executable(phase_recovery_cuda
    src/main.cu
    src/kernels.cu
)

target_link_libraries(phase_recovery_cuda CUDA::cufft CUDA::cudart)
