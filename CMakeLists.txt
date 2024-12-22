cmake_minimum_required(VERSION 3.11)

project(perf-gemm CUDA CXX)

find_package(CUDAToolkit)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES 86;89)

add_executable(perf-gemm-wmma-sm80 perf-gemm-wmma-sm80.cu)
add_executable(perf-gemm-wmma-multi-stage-sm80 perf-gemm-wmma-multi-stage-sm80.cu)
add_executable(perf-gemm-ldmatrix-sm80 perf-gemm-ldmatrix-sm80.cu)

target_link_libraries(perf-gemm-wmma-sm80 PRIVATE CUDA::cublas)
target_link_libraries(perf-gemm-wmma-multi-stage-sm80 PRIVATE CUDA::cublas)
target_link_libraries(perf-gemm-ldmatrix-sm80 PRIVATE CUDA::cublas)
