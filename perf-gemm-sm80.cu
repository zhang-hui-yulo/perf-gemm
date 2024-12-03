﻿#include <random>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "thrust/host_vector.h"
#include "thrust/device_vector.h"
#include "mma.h"
#include "cublas_v2.h"

#include "neo/layout.hpp"
#include "neo/tensor.hpp"
#include "neo/numeric/integral_constant.hpp"

struct NeoConfig {
    using T = half;

    static constexpr int TILE_M = 128, TILE_N = 128, TILE_K = 32;
    static constexpr int MMA_M = 16, MMA_N = 16, MMA_K = 16;
    static constexpr int CopyUnitSize = 16;
    static constexpr int Threads = 128;
    static constexpr int WrapSize = 32;
    static constexpr int WrapCount = Threads / WrapSize;
    static constexpr int CopyCount = CopyUnitSize / sizeof(T);
    static constexpr int CopyColAB = TILE_K / CopyCount;
    static constexpr int CopyRowAB = Threads / CopyColAB;
    static constexpr int CopyColC = TILE_N / CopyCount;
    static constexpr int CopyRowC = Threads / CopyColC;
    static constexpr int Stage = 3;

    using CopyTiledShapeA = decltype(neo::make_shape(neo::Int<CopyRowAB>{}, neo::Int<CopyColAB * CopyCount>{}));
    using CopyThrdShapeA = decltype(neo::make_shape(neo::Int<CopyRowAB>{}, neo::Int<CopyColAB>{}));
    using CopyTiledShapeB = CopyTiledShapeA;
    using CopyThrdShapeB = CopyThrdShapeA;
    using CopyTiledShapeC = decltype(neo::make_shape(neo::Int<CopyRowC>{}, neo::Int<CopyColC * CopyCount>{}));
    using CopyThrdShapeC = decltype(neo::make_shape(neo::Int<CopyRowC>{}, neo::Int<CopyColC>{}));

    static constexpr int PartitionA = 2;
    static constexpr int PartitionB = WrapCount / PartitionA;
    static constexpr int PartitionM = TILE_M / WrapCount * PartitionA;
    static constexpr int PartitionN = TILE_N / WrapCount * PartitionB;
    using TiledMmaShapeA = decltype(neo::make_shape(neo::Int<PartitionM>{}, neo::Int<MMA_K>{}));
    using TiledMmaShapeB = decltype(neo::make_shape(neo::Int<PartitionN>{}, neo::Int<MMA_K>{}));
    using TiledMmaShapeC = decltype(neo::make_shape(neo::Int<PartitionM>{}, neo::Int<PartitionN>{}));

    using MmaShapeA = decltype(neo::make_shape(neo::Int<MMA_M>{}, neo::Int<MMA_K>{}));
    using MmaShapeB = decltype(neo::make_shape(neo::Int<MMA_N>{}, neo::Int<MMA_K>{}));
    using MmaShapeC = decltype(neo::make_shape(neo::Int<MMA_N>{}, neo::Int<MMA_N>{}));
};

template <typename T>
__global__ void gpu_compare_kernel(const T* x, const T* y, int n,
    float threshold, int* count,
    float* max_error) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= n) {
        return;
    }

    float v0 = x[idx];
    float v1 = y[idx];

    float diff = fabs(v0 - v1);
    if (diff > threshold) {
        atomicAdd(count, 1);

        // for positive floating point, there int representation is in the same
        // order.
        int int_diff = *((int*)(&diff));
        atomicMax((int*)max_error, int_diff);
    }
}

template <typename T>
void gpu_compare(const T *x, const T *y, int n, float threshold = 1.E-1) {
  int *num_count;
  float *max_error;
  cudaMalloc(&num_count, sizeof(int));
  cudaMalloc(&max_error, sizeof(float));
  cudaMemset(num_count, 0, sizeof(int));
  cudaMemset(max_error, 0, sizeof(float));

  dim3 block(256);
  dim3 grid((n + block.x - 1) / block.x);
  gpu_compare_kernel<<<grid, block>>>(x, y, n, threshold, num_count, max_error);
  int num = 0;
  float error = 0;
  cudaMemcpy(&num, num_count, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&error, max_error, sizeof(int), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  if (num == 0) {
    std::cout << "check ok, max_error = " << error << std::endl;
  } else {
    std::cout << "===============================" << std::endl <<
    "check fail: diff " << num << " max_error = " << error << std::endl <<
    "===============================" << std::endl;
  }
}

template <typename Config>
__global__ /*__launch_bounds__(128, 1)*/
void mma_aligned_128(Config::T* __restrict__ c, const Config::T* __restrict__ a, const Config::T* __restrict__ b, const int m, const int n, const int k) {
    int idx = threadIdx.x;
    int ix = blockIdx.x;
    int iy = blockIdx.y;
    int wrapid = idx / Config::WrapSize;
    int tiledM = wrapid / Config::PartitionA;
    int tiledN = wrapid % Config::PartitionB;
    
    constexpr int PADDING_A = 8;
    constexpr int PADDING_B = 8;
    constexpr int PADDING_C = 0;

    using T = typename Config::T;
    constexpr int shm_a_size = Config::TILE_M * (Config::TILE_K + PADDING_A);
    constexpr int shm_b_size = Config::TILE_N * (Config::TILE_K + PADDING_B);
    constexpr int shm_c_size = Config::TILE_M * (Config::TILE_N + PADDING_C);
    constexpr int shm_size = shm_a_size + shm_b_size > shm_c_size ? shm_a_size + shm_b_size : shm_c_size;
    __shared__ T shmem[shm_size];
    auto shm_a = shmem;
    auto shm_b = shmem + shm_a_size;
    auto shm_c = shmem;

    using CopyTiledShapeA = typename Config::CopyTiledShapeA;
    using CopyThrdShapeA = typename Config::CopyThrdShapeA;
    auto copyTiledShapeA = CopyTiledShapeA{};
    auto copyThrdShapeA = CopyThrdShapeA{};

    auto A = neo::make_tensor(a, neo::make_shape(m, k), neo::make_stride(k, neo::Int<1>{}));
    auto gA = neo::local_tile(A, neo::make_shape(neo::Int<Config::TILE_M>{}, neo::Int<Config::TILE_K>{}), neo::make_coord(iy, neo::Int<0>{}));
    auto sA = neo::make_tensor(shm_a, neo::make_shape(neo::Int<Config::TILE_M>{}, neo::Int<Config::TILE_K>{}), neo::make_stride(neo::Int<Config::TILE_K + PADDING_A>{}, neo::Int<1>{}));
    auto gOuterShapeA = neo::inner_div(gA.shape(), copyTiledShapeA);
    auto thrCopyCoordA = neo::copy_partition(copyThrdShapeA, idx, neo::Int<Config::CopyCount>{});

    using CopyTiledShapeB = typename Config::CopyTiledShapeB;
    using CopyThrdShapeB = typename Config::CopyThrdShapeB;
    auto copyTiledShapeB = CopyTiledShapeB{};
    auto copyThrdShapeB = CopyThrdShapeB{};

    auto B = neo::make_tensor(b, neo::make_shape(n, k), neo::make_stride(k, neo::Int<1>{}));
    auto gB = neo::local_tile(B, neo::make_shape(neo::Int<Config::TILE_N>{}, neo::Int<Config::TILE_K>{}), neo::make_coord(ix, neo::Int<0>{}));
    auto sB = neo::make_tensor(shm_b, neo::make_shape(neo::Int<Config::TILE_N>{}, neo::Int<Config::TILE_K>{}), neo::make_stride(neo::Int<Config::TILE_K + PADDING_B>{}, neo::Int<1>{}));
    auto gOuterShapeB = neo::inner_div(gA.shape(), copyTiledShapeB);
    auto thrCopyCoordB = neo::copy_partition(copyThrdShapeB, idx, neo::Int<Config::CopyCount>{});

    using TiledMmaShapeA = typename Config::TiledMmaShapeA;
    auto tiledMmaShapeA = TiledMmaShapeA{};
    auto sOuterTiledMmaShapeA = neo::inner_div(sA.shape(), tiledMmaShapeA);
    auto tiledMmaA = neo::local_tile(sA, tiledMmaShapeA);

    using TiledMmaShapeB = typename Config::TiledMmaShapeB;
    auto tiledWrapMmaShapeB = TiledMmaShapeB{};
    auto sOuterTiledWrapMmaShapeB = neo::inner_div(sB.shape(), tiledWrapMmaShapeB);
    auto tiledMmaB = neo::local_tile(sB, tiledWrapMmaShapeB);

    using MmaShapeA = typename Config::MmaShapeA;
    auto mmaShapeA = MmaShapeA{};
    auto sOuterMmaShapeA = neo::inner_div(tiledMmaA.shape(), mmaShapeA);

    using MmaShapeB = typename Config::MmaShapeB;
    auto mmaShapeB = MmaShapeB{};
    auto sOuterMmaShapeB = neo::inner_div(tiledMmaB.shape(), mmaShapeB);

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, Config::MMA_M, Config::MMA_N, Config::MMA_K, half, nvcuda::wmma::row_major> a_frag[sOuterMmaShapeA.row_spacing][sOuterMmaShapeA.col_spacing];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, Config::MMA_M, Config::MMA_N, Config::MMA_K, half, nvcuda::wmma::col_major> b_frag[sOuterMmaShapeB.row_spacing][sOuterMmaShapeB.col_spacing];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, Config::MMA_M, Config::MMA_N, Config::MMA_K, half> c_frag[sOuterMmaShapeA.row_spacing][sOuterMmaShapeB.row_spacing];

#pragma unroll
    for (int im = 0; im < sOuterMmaShapeA.row_spacing; ++im) {
#pragma unroll
        for (int in = 0; in < sOuterMmaShapeB.row_spacing; ++in) {
            nvcuda::wmma::fill_fragment(c_frag[im][in], 0);
        }
    }

#pragma unroll
    for (int itile = 0, ntile = k / Config::TILE_K; itile < ntile; ++itile) {
        gA.jump(neo::make_coord(iy, itile));
        auto gAcopyTile = neo::local_tile(gA, copyTiledShapeA);
        auto sAcopyTile = neo::local_tile(sA, copyTiledShapeA);

        gB.jump(neo::make_coord(ix, itile));
        auto gBcopyTile = neo::local_tile(gB, copyTiledShapeB);
        auto sBcopyTile = neo::local_tile(sB, copyTiledShapeB);
        
        // copy global A to shared A
#pragma unroll
        for (int i = 0; i < gOuterShapeA.row_spacing; ++i) {
#pragma unroll
            for (int j = 0; j < gOuterShapeA.col_spacing; ++j) {
                auto coord = neo::make_coord(i, j);
                gAcopyTile.jump(coord);
                sAcopyTile.jump(coord);
                auto gAptr = gAcopyTile.move_at(thrCopyCoordA);
                auto sAptr = __cvta_generic_to_shared(sAcopyTile.move_at(thrCopyCoordA));
                asm("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" :
                : "l"(sAptr), "l"(gAptr), "n"(Config::CopyUnitSize));
            }
        }

        // copy global B to shared B
#pragma unroll
        for (int i = 0; i < gOuterShapeB.row_spacing; ++i) {
#pragma unroll
            for (int j = 0; j < gOuterShapeB.col_spacing; ++j) {
                auto coord = neo::make_coord(i, j);
                gBcopyTile.jump(coord);
                sBcopyTile.jump(coord);
                auto gBptr = gBcopyTile.move_at(thrCopyCoordB);
                auto sBptr = __cvta_generic_to_shared(sBcopyTile.move_at(thrCopyCoordB));
                asm("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" :
                : "l"(sBptr), "l"(gBptr), "n"(Config::CopyUnitSize));
            }
        }

        asm("cp.async.commit_group;\n" ::);
        asm("cp.async.wait_group 0;\n" ::);

        __syncthreads();

#pragma unroll
        for (int ik = 0, nk = sOuterTiledMmaShapeA.col_spacing; ik < nk; ++ik) {
            tiledMmaA.jump(neo::make_coord(tiledM, ik));
            tiledMmaB.jump(neo::make_coord(tiledN, ik));
            auto MmaA = neo::local_tile(tiledMmaA, mmaShapeA);
            auto MmaB = neo::local_tile(tiledMmaB, mmaShapeB);

#pragma unroll
            for (int mma_m = 0; mma_m < sOuterMmaShapeA.row_spacing; ++mma_m) {
#pragma unroll
                for (int mma_k = 0; mma_k < sOuterMmaShapeA.col_spacing; ++mma_k) {
                    auto afragPtr = MmaA.jump_at(neo::make_coord(mma_m, mma_k));
                    nvcuda::wmma::load_matrix_sync(a_frag[mma_m][mma_k], afragPtr, MmaA.stride().row_spacing);
                }
            }

#pragma unroll
            for (int mma_n = 0; mma_n < sOuterMmaShapeB.row_spacing; ++mma_n) {
#pragma unroll
                for (int mma_k = 0; mma_k < sOuterMmaShapeB.col_spacing; ++mma_k) {
                    auto bfragPtr = MmaB.jump_at(neo::make_coord(mma_n, mma_k));
                    nvcuda::wmma::load_matrix_sync(b_frag[mma_n][mma_k], bfragPtr, MmaB.stride().row_spacing);
                }
            }

#pragma unroll
            for (int mma_m = 0; mma_m < sOuterMmaShapeA.row_spacing; ++mma_m) {
#pragma unroll
                for (int mma_n = 0; mma_n < sOuterMmaShapeB.row_spacing; ++mma_n) {
#pragma unroll
                    for (int mma_k = 0; mma_k < sOuterMmaShapeB.col_spacing; ++mma_k) {
                        nvcuda::wmma::mma_sync(c_frag[mma_m][mma_n], a_frag[mma_m][mma_k], b_frag[mma_n][mma_k], c_frag[mma_m][mma_n]);
                    }
                }
            }
        }

        __syncthreads();
    }
    
    auto sC = neo::make_tensor(shm_c, neo::make_shape(neo::Int<Config::TILE_M>{}, neo::Int<Config::TILE_N>{}), neo::make_stride(neo::Int<Config::TILE_N + PADDING_C>{}, neo::Int<1>{}));
    using TiledMmaShapeC = typename Config::TiledMmaShapeC;
    auto tiledWrapMmaShapeC = TiledMmaShapeC{};
    auto tiledMmaC = neo::local_tile(sC, tiledWrapMmaShapeC, neo::make_coord(tiledM, tiledN));

    using MmaShapeC = typename Config::MmaShapeC;
    auto mmaShapeC = MmaShapeC{};
    auto sOuterMmaShapeC = neo::inner_div(tiledMmaC.shape(), mmaShapeC);
    auto MmaC = neo::local_tile(tiledMmaC, mmaShapeC);

    using CopyTiledShapeC = typename Config::CopyTiledShapeC;
    using CopyThrdShapeC = typename Config::CopyThrdShapeC;
    auto copyTiledShapeC = CopyTiledShapeC{};
    auto copyThrdShapeC = CopyThrdShapeC{};

    auto C = neo::make_tensor(c, neo::make_shape(m, n), neo::make_stride(n, neo::Int<1>{}));
    auto gC = neo::local_tile(C, neo::make_shape(neo::Int<Config::TILE_M>{}, neo::Int<Config::TILE_N>{}), neo::make_coord(iy, ix));
    auto gOuterShapeC = neo::inner_div(gC.shape(), copyTiledShapeC);
    auto thrCopyCoordC = neo::copy_partition(copyThrdShapeC, idx, neo::Int<Config::CopyCount>{});
    auto gCcopyTile = neo::local_tile(gC, copyTiledShapeC);
    auto sCcopyTile = neo::local_tile(sC, copyTiledShapeC);

    // reg C to shared C
#pragma unroll
    for (int im = 0; im < sOuterMmaShapeC.row_spacing; ++im) {
#pragma unroll
        for (int in = 0; in < sOuterMmaShapeC.col_spacing; ++in) {
            auto cPtr = MmaC.jump_at(neo::make_coord(im, in));
            nvcuda::wmma::store_matrix_sync(cPtr, c_frag[im][in], MmaC.stride().row_spacing, nvcuda::wmma::mem_row_major);
        }
    }

    __syncthreads();

    // copy shared C to global C
#pragma unroll
    for (int i = 0; i < gOuterShapeC.row_spacing; ++i) {
#pragma unroll
        for (int j = 0; j < gOuterShapeC.col_spacing; ++j) {
            auto coord = neo::make_coord(i, j);
            gCcopyTile.jump(coord);
            sCcopyTile.jump(coord);
            auto gCptr = reinterpret_cast<int4*>(gCcopyTile.move_at(thrCopyCoordC));
            auto sCptr = reinterpret_cast<int4*>(sCcopyTile.move_at(thrCopyCoordC));
            *gCptr = *sCptr;
        }
    }
}

int main() {
    int m = NeoConfig::TILE_M * 640, n = NeoConfig::TILE_N * 2, k = NeoConfig::TILE_K * 16;

    thrust::host_vector<NeoConfig::T> h_a(m * k);
    thrust::host_vector<NeoConfig::T> h_b(n * k);

    std::mt19937 gen(10086);
    std::uniform_int_distribution<> dis(0, 200);

    for (auto& i : h_a) {
        float n = (dis(gen) - 100.0f) * 0.01f;
        i = n;
    }

    for (auto& i : h_b) {
        float n = (dis(gen) - 100.0f) * 0.01f;
        i = n;
    }

    thrust::device_vector<NeoConfig::T> d_a = h_a;
    thrust::device_vector<NeoConfig::T> d_b = h_b;
    thrust::device_vector<NeoConfig::T> d_c1(m * n, 0);
    thrust::device_vector<NeoConfig::T> d_c2(m * n, 0);

    dim3 grid(n / NeoConfig::TILE_N, m / NeoConfig::TILE_M);
    dim3 block(NeoConfig::Threads);

    // cuda warmup
    constexpr int warmup = 100;
    constexpr int nt = 1000;

    for (int i = 0; i < warmup; ++i) {
        mma_aligned_128<NeoConfig> << <grid, block >> > (d_c1.data().get(), d_a.data().get(), d_b.data().get(), m, n, k);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time_mma = 0;

    for (int i = 0; i < nt; ++i) {
        cudaEventRecord(start);

        mma_aligned_128<NeoConfig> << <grid, block >> > (d_c1.data().get(), d_a.data().get(), d_b.data().get(), m, n, k);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float elapsed = 0;
        cudaEventElapsedTime(&elapsed, start, stop);
        time_mma += elapsed;

        auto err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cout << "mma_aligned_128 launch error: " << err << std::endl;
        }
    }

    cublasHandle_t handle;
    cublasCreate(&handle);

    NeoConfig::T alpha = NeoConfig::T(1.f);
    NeoConfig::T beta = NeoConfig::T(0.f);

    // cublas warmup
    for (int i = 0; i < warmup; ++i) {
        cublasStatus_t ret = cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
            n, m, k,
            &alpha,
            d_b.data().get(), k,
            d_a.data().get(), k,
            &beta,
            d_c2.data().get(), n);
    }

    float time_cublas = 0;

    for (int i = 0; i < nt; ++i) {
        cudaEventRecord(start);

        cublasStatus_t ret = cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
            n, m, k,
            &alpha,
            d_b.data().get(), k,
            d_a.data().get(), k,
            &beta,
            d_c2.data().get(), n);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float elapsed = 0;
        cudaEventElapsedTime(&elapsed, start, stop);
        time_cublas += elapsed;

        if (ret != cudaSuccess) {
            std::cout << "cublasHgemm error: " << ret << std::endl;
        }
    }


    gpu_compare(d_c1.data().get(), d_c2.data().get(), m * n);

    std::cout << "my mma: " << time_mma / nt << " cublas: " << time_cublas / nt << std::endl;

    return 0;
}