cmake_minimum_required(VERSION 3.11)

project(perf-gemm CUDA CXX)

find_package(CUDAToolkit)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES 86;89)

add_executable(${PROJECT_NAME} perf-gemm-sm80.cu)

target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cublas)
