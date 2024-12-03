#pragma once

#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
#  define NEO_HOST_DEVICE __forceinline__ __host__ __device__
#  define NEO_DEVICE      __forceinline__          __device__
#  define NEOHOST        __forceinline__ __host__
#else
#  define NEO_HOST_DEVICE inline
#  define NEO_DEVICE      inline
#  define NEO_HOST        inline
#endif // NEO_HOST_DEVICE,NEO_DEVICE
