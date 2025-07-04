cmake_minimum_required(VERSION 3.18)

project(perf-gemm CUDA CXX)

find_package(CUDAToolkit)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES 86;89)

add_executable(flash_attention_2_sm80 flash_attention_2_sm80.cu)

target_include_directories(flash_attention_2_sm80 PRIVATE "../cutlass/include")
target_link_libraries(flash_attention_2_sm80 PRIVATE CUDA::cublas)
