cmake_minimum_required(VERSION 3.18)

project(perf-gemm CUDA CXX)

find_package(CUDAToolkit)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES 86;89)

include(FetchContent)
FetchContent_Declare(cutlass
        GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
        GIT_TAG v3.8.0
        GIT_PROGRESS TRUE
        GIT_SHALLOW TRUE)
FetchContent_GetProperties(cutlass)
if(NOT cutlass_POPULATED)
    FetchContent_Populate(cutlass)
endif()

add_executable(flash_attention_2_sm80 flash_attention_2_sm80.cu)

target_include_directories(flash_attention_2_sm80 PRIVATE "${cutlass_SOURCE_DIR}/include")
target_link_libraries(flash_attention_2_sm80 PRIVATE CUDA::cublas)
