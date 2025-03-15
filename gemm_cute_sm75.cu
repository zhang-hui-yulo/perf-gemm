#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include <cublas_v2.h>

#include "cutlass/util/GPU_Clock.hpp"


template <
    class ProblemShape, class SplitShape, class CtaTiler,
    class TA, class AStride, class ASmemLayout, class TiledCopyA,
    class TB, class BStride, class BSmemLayout, class TiledCopyB,
    class TC, class CStride, class CSmemLayout, class TiledCopyC, class TiledCopyDestC,
    class TiledMma, class MmaCopyA, class MmaCopyB, class MmaCopyC,
    class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void gemm_device_twostage_aligned_splitk(
    ProblemShape shape_MNK, SplitShape shape_MNSK, CtaTiler cta_tiler,
    TA const* A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a,
    TB const* B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b,
    TC* C, CStride dC, CSmemLayout sC_layout, TiledCopyC copy_c, TiledCopyDestC copy_dest_c,
    TiledMma mma, MmaCopyA copy_mma_a, MmaCopyB copy_mma_b, MmaCopyC copy_mma_c,
    Alpha alpha, Beta beta) {
    using namespace cute;

    // Preconditions
    CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});                   // (M, N, K)
    CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});                   // (BLK_M, BLK_N, BLK_K)

    CUTE_STATIC_ASSERT_V(size(copy_a) == size(mma));                     // NumThreads
    CUTE_STATIC_ASSERT_V(size(copy_b) == size(mma));                     // NumThreads

    static_assert(is_static<ASmemLayout>::value);
    static_assert(is_static<BSmemLayout>::value);
    static_assert(is_static<CSmemLayout>::value);

    CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler));  // BLK_M
    CUTE_STATIC_ASSERT_V(size<0>(CSmemLayout{}) == size<0>(cta_tiler));  // BLK_M
    CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
    CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
    CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler));  // BLK_K
    CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler));  // BLK_K

    CUTE_STATIC_ASSERT_V(congruent(select<0, 2>(shape_MNK), dA));        // dA strides for shape MK
    CUTE_STATIC_ASSERT_V(congruent(select<1, 2>(shape_MNK), dB));        // dB strides for shape NK
    CUTE_STATIC_ASSERT_V(congruent(select<0, 1>(shape_MNK), dC));        // dC strides for shape MN

    //
    // Full and Tiled Tensors
    //

    // Represent the full tensors
    Tensor mA = make_tensor(make_gmem_ptr(A), select<0, 2>(shape_MNK), dA); // (M,K)
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1, 2>(shape_MNK), dB); // (N,K)
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0, 1>(shape_MNK), dC); // (M,N)

    // Represent the split tensors
    auto splitk_coord = make_coord(blockIdx.x, blockIdx.y, blockIdx.z);                    // (m,n,splitk)
    Tensor mSplitA = local_tile(mA, select<0, 2>(shape_MNSK), select<0, 2>(splitk_coord)); // (M, SK)
    Tensor mSplitB = local_tile(mB, select<1, 2>(shape_MNSK), select<1, 2>(splitk_coord)); // (N, SK)

    // Get the appropriate blocks for this thread block
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);                     // (m,n,k)
    Tensor gA = local_tile(mSplitA, cta_tiler, cta_coord, Step<_1, X, _1>{});   // (BLK_M,BLK_K,k)
    Tensor gB = local_tile(mSplitB, cta_tiler, cta_coord, Step< X, _1, _1>{});  // (BLK_N,BLK_K,k)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});        // (BLK_M,BLK_N)

    // Shared memory buffers
    __shared__ TA smemA[cosize_v<ASmemLayout>];
    __shared__ TB smemB[cosize_v<BSmemLayout>];
    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);            // (BLK_M,BLK_K)
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);            // (BLK_N,BLK_K)

    //
    // Partition the copying of A and B tiles across the threads
    //

    ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
    Tensor tAgA = thr_copy_a.partition_S(gA);                            // (CPY,CPY_M,CPY_K,k)
    Tensor tAsA = thr_copy_a.partition_D(sA);                            // (CPY,CPY_M,CPY_K)
    Tensor tArA = make_fragment_like(tAsA);                              // (CPY,CPY_M,CPY_K)

    ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
    Tensor tBgB = thr_copy_b.partition_S(gB);                            // (CPY,CPY_N,CPY_K,k)
    Tensor tBsB = thr_copy_b.partition_D(sB);                            // (CPY,CPY_N,CPY_K)
    Tensor tBrB = make_fragment_like(tBsB);                              // (CPY,CPY_N,CPY_K)

    CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA));                // CPY_M
    CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tArA));                // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tAsA));                // CPY_K
    CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tArA));                // CPY_K
    CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB));                // CPY_N
    CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBrB));                // CPY_N
    CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBsB));                // CPY_K
    CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBrB));                // CPY_K

    // Copy gmem to rmem for k_tile=0
    copy(copy_a, tAgA(_, _, _, 0), tArA);
    copy(copy_b, tBgB(_, _, _, 0), tBrB);
    //
    // Define A/B partitioning and C accumulators
    //

    ThrMMA thr_mma = mma.get_slice(threadIdx.x);
    Tensor tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));  // (MMA, MMA_M, MMA_K)
    Tensor tCrB = thr_mma.partition_fragment_B(gB(_, _, 0));  // (MMA, MMA_N, MMA_K)
    Tensor tCrC = thr_mma.partition_fragment_C(gC);           // (MMA, MMA_M, MMA_N)

    ThrCopy thr_mma_copy_a = copy_mma_a.get_slice(threadIdx.x);
    Tensor tCsA = thr_mma_copy_a.partition_S(sA);             // (CPY, CPY_M, CPY_K)
    Tensor tCrA_view = thr_mma_copy_a.retile_D(tCrA);         // (CPY, CPY_M, CPY_K)

    ThrCopy thr_mma_copy_b = copy_mma_b.get_slice(threadIdx.x);
    Tensor tCsB = thr_mma_copy_b.partition_S(sB);             // (CPY, CPY_M, CPY_K)
    Tensor tCrB_view = thr_mma_copy_b.retile_D(tCrB);         // (CPY, CPY_M, CPY_K)

    // fill zero for accumulator
    clear(tCrC);

#if 0
    if (thread0()) {
        print("  mA : "); print(mA); print("\n");
        print("  mSplitA : "); print(mSplitA); print("\n");
        print("  gA : "); print(gA); print("\n");
        print("  sA : "); print(sA); print("\n");
        print("tAgA : "); print(tAgA); print("\n");
        print("tAsA : "); print(tAsA); print("\n");
        print("tArA : "); print(tArA); print("\n");
    }
#endif

#if 0
    if (thread0()) {
        print("  mB : "); print(mB); print("\n");
        print("  mSplitB : "); print(mSplitB); print("\n");
        print("  gB : "); print(gB); print("\n");
        print("  sB : "); print(sB); print("\n");
        print("tBgB : "); print(tBgB); print("\n");
        print("tBsB : "); print(tBsB); print("\n");
        print("tArA : "); print(tArA); print("\n");
    }
#endif

#if 0
    if (thread0()) {
        print("  mC : "); print(mC); print("\n");
        print("  gC : "); print(gC); print("\n");
        print("tCsA : "); print(tCsA); print("\n");
        print("tCsB : "); print(tCsB); print("\n");
        print("tCgC : "); print(tCgC); print("\n");
        print("tCrC : "); print(tCrC); print("\n");
    }
#endif

#if 1

    // Copy rmem to smem
    copy(copy_a, tArA, tAsA);
    copy(copy_b, tBrB, tBsB);
    __syncthreads();

    //
    // PIPELINED MAIN LOOP

    // Load A, B shmem->regs for k_block=0
    copy(copy_mma_a, tCsA(_, _, 0), tCrA_view(_, _, 0));
    copy(copy_mma_b, tCsB(_, _, 0), tCrB_view(_, _, 0));

    auto K_TILE_MAX = size<3>(tAgA);
    auto K_BLOCK_MAX = size<2>(tCrA);

    CUTE_NO_UNROLL
    for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile) {
        // Pipeline the k-mode of the block registers
        CUTE_UNROLL
        for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
            if (k_block == K_BLOCK_MAX - 1) {
                // Copy rmem to smem
                __syncthreads();
                copy(copy_a, tArA, tAsA);
                copy(copy_b, tBrB, tBsB);
                __syncthreads();
            }

            // Copy smem to rmem for k_block+1
            int k_block_next = (k_block + 1) % K_BLOCK_MAX;
            copy(copy_mma_a, tCsA(_, _, k_block_next), tCrA_view(_, _, k_block_next));
            copy(copy_mma_b, tCsB(_, _, k_block_next), tCrB_view(_, _, k_block_next));

            if (k_block == 0) {
                // Copy gmem to rmem for k_tile+1
                int k_tile_next = (k_tile + 1 < K_TILE_MAX) ? k_tile + 1 : k_tile;
                copy(copy_a, tAgA(_, _, _, k_tile_next), tArA);
                copy(copy_b, tBgB(_, _, _, k_tile_next), tBrB);
            }

            // Thread-level register gemm for k_block
            gemm(mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrC);
        } // k_block
    } // k_tile

#if 0
    if (thread(0, 0)) {
        print_tensor(tCrC); printf("\n");
    }
#endif

    // Epilogue

    // Shared memory buffers
    __shared__ TC smemC[cosize_v<CSmemLayout>];
    Tensor sC = make_tensor(make_smem_ptr(smemC), sC_layout);            // (BLK_M,BLK_N)

    if (blockIdx.z == 0) {
        // Define copy for smem to gmem
        ThrCopy thr_copy_c = copy_c.get_slice(threadIdx.x);
        Tensor tDgC = thr_copy_c.partition_S(gC);                            // (CPY,CPY_N,CPY_K,k)
        Tensor tDsC = thr_copy_c.partition_D(sC);                            // (CPY,CPY_N,CPY_K)

        CUTE_STATIC_ASSERT_V(size<1>(tDgC) == size<1>(tDsC));                // CPY_M
        CUTE_STATIC_ASSERT_V(size<2>(tDgC) == size<2>(tDsC));                // CPY_N

#if 0
        if (thread0()) {
            print("tDgC : "); print(tDgC); print("\n");
            print("tDsC : "); print(tDsC); print("\n");
        }
#endif
        // Copy gmem to smem
        if (blockIdx.z == 0) {
            copy(copy_c, tDgC, tDsC);
        }

        __syncthreads();
    }

    // Define mma copy
    ThrCopy thr_mma_copy_c = copy_mma_c.get_slice(threadIdx.x);
    Tensor tCsC = thr_mma_copy_c.partition_S(sC);             // (CPY, CPY_M, CPY_N)

    if (blockIdx.z == 0) {
        Tensor tDrC = thr_mma.partition_fragment_C(gC);           // (MMA, MMA_M, MMA_N)
        Tensor tDrC_view = thr_mma_copy_c.retile_D(tDrC);         // (CPY, CPY_M, CPY_N)

        copy(copy_mma_c, tCsC, tDrC_view);
        axpby(alpha, tCrC, beta, tDrC);
        copy(copy_mma_c, tDrC_view, tCsC);
    } else {
        Tensor tDrC_view = thr_mma_copy_c.retile_D(tCrC);         // (CPY, CPY_M, CPY_N)

        axpby(alpha, tCrC, 0, tCrC);
        copy(copy_mma_c, tDrC_view, tCsC);
    }

    __syncthreads();

    // Define copy for smem to gmem
    ThrCopy thr_copy_dest_c = copy_dest_c.get_slice(threadIdx.x);
    Tensor tDgDestC = thr_copy_dest_c.partition_S(gC);                   // (CPY,CPY_N,CPY_K,k)
    Tensor tDsDestC = thr_copy_dest_c.partition_D(sC);                   // (CPY,CPY_N,CPY_K)

    CUTE_STATIC_ASSERT_V(size<1>(tDgDestC) == size<1>(tDsDestC));        // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tDgDestC) == size<2>(tDsDestC));        // CPY_N

    // Copy smem to gmem
    copy(copy_dest_c, tDsDestC, tDgDestC);

#endif
}

template <
    class ProblemShape, class CtaTiler,
    class TA, class AStride, class ASmemLayout, class TiledCopyA,
    class TB, class BStride, class BSmemLayout, class TiledCopyB,
    class TC, class CStride, class CSmemLayout, class TiledCopyC,
    class TiledMma, class MmaCopyA, class MmaCopyB, class MmaCopyC,
    class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void gemm_device_twostage_aligned(
    ProblemShape shape_MNK, CtaTiler cta_tiler,
    TA const* A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a,
    TB const* B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b,
    TC* C, CStride dC, CSmemLayout sC_layout, TiledCopyC copy_c,
    TiledMma mma, MmaCopyA copy_mma_a, MmaCopyB copy_mma_b, MmaCopyC copy_mma_c,
    Alpha alpha, Beta beta) {
    using namespace cute;

    // Preconditions
    CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});                   // (M, N, K)
    CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});                   // (BLK_M, BLK_N, BLK_K)

    CUTE_STATIC_ASSERT_V(size(copy_a) == size(mma));                     // NumThreads
    CUTE_STATIC_ASSERT_V(size(copy_b) == size(mma));                     // NumThreads

    static_assert(is_static<ASmemLayout>::value);
    static_assert(is_static<BSmemLayout>::value);
    static_assert(is_static<CSmemLayout>::value);

    CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler));  // BLK_M
    CUTE_STATIC_ASSERT_V(size<0>(CSmemLayout{}) == size<0>(cta_tiler));  // BLK_M
    CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
    CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
    CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler));  // BLK_K
    CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler));  // BLK_K

    CUTE_STATIC_ASSERT_V(congruent(select<0, 2>(shape_MNK), dA));        // dA strides for shape MK
    CUTE_STATIC_ASSERT_V(congruent(select<1, 2>(shape_MNK), dB));        // dB strides for shape NK
    CUTE_STATIC_ASSERT_V(congruent(select<0, 1>(shape_MNK), dC));        // dC strides for shape MN

    //
    // Full and Tiled Tensors
    //

    // Represent the full tensors
    Tensor mA = make_tensor(make_gmem_ptr(A), select<0, 2>(shape_MNK), dA); // (M,K)
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1, 2>(shape_MNK), dB); // (N,K)
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0, 1>(shape_MNK), dC); // (M,N)

    // Get the appropriate blocks for this thread block
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);                // (m,n,k)
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});   // (BLK_M,BLK_K,k)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X, _1, _1>{});  // (BLK_N,BLK_K,k)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});   // (BLK_M,BLK_N)

    // Shared memory buffers
    __shared__ TA smemA[cosize_v<ASmemLayout>];
    __shared__ TB smemB[cosize_v<BSmemLayout>];
    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);            // (BLK_M,BLK_K)
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);            // (BLK_N,BLK_K)

    //
    // Partition the copying of A and B tiles across the threads
    //

    ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
    Tensor tAgA = thr_copy_a.partition_S(gA);                            // (CPY,CPY_M,CPY_K,k)
    Tensor tAsA = thr_copy_a.partition_D(sA);                            // (CPY,CPY_M,CPY_K)
    Tensor tArA = make_fragment_like(tAsA);                              // (CPY,CPY_M,CPY_K)

    ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
    Tensor tBgB = thr_copy_b.partition_S(gB);                            // (CPY,CPY_N,CPY_K,k)
    Tensor tBsB = thr_copy_b.partition_D(sB);                            // (CPY,CPY_N,CPY_K)
    Tensor tBrB = make_fragment_like(tBsB);                              // (CPY,CPY_N,CPY_K)

    CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA));                // CPY_M
    CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tArA));                // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tAsA));                // CPY_K
    CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tArA));                // CPY_K
    CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB));                // CPY_N
    CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBrB));                // CPY_N
    CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBsB));                // CPY_K
    CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBrB));                // CPY_K

    // Copy gmem to rmem for k_tile=0
    copy(copy_a, tAgA(_, _, _, 0), tArA);
    copy(copy_b, tBgB(_, _, _, 0), tBrB);
    //
    // Define A/B partitioning and C accumulators
    //

    ThrMMA thr_mma = mma.get_slice(threadIdx.x);
    Tensor tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));  // (MMA, MMA_M, MMA_K)
    Tensor tCrB = thr_mma.partition_fragment_B(gB(_, _, 0));  // (MMA, MMA_N, MMA_K)
    Tensor tCrC = thr_mma.partition_fragment_C(gC);           // (MMA, MMA_M, MMA_N)

    ThrCopy thr_mma_copy_a = copy_mma_a.get_slice(threadIdx.x);
    Tensor tCsA = thr_mma_copy_a.partition_S(sA);             // (CPY, CPY_M, CPY_K)
    Tensor tCrA_view = thr_mma_copy_a.retile_D(tCrA);         // (CPY, CPY_M, CPY_K)

    ThrCopy thr_mma_copy_b = copy_mma_b.get_slice(threadIdx.x);
    Tensor tCsB = thr_mma_copy_b.partition_S(sB);             // (CPY, CPY_M, CPY_K)
    Tensor tCrB_view = thr_mma_copy_b.retile_D(tCrB);         // (CPY, CPY_M, CPY_K)

    // fill zero for accumulator
    clear(tCrC);

#if 0
    if (thread0()) {
        print("  mA : "); print(mA); print("\n");
        print("  gA : "); print(gA); print("\n");
        print("  sA : "); print(sA); print("\n");
        print("tAgA : "); print(tAgA); print("\n");
        print("tAsA : "); print(tAsA); print("\n");
        print("tArA : "); print(tArA); print("\n");
    }
#endif

#if 0
    if (thread0()) {
        print("  mB : "); print(mB); print("\n");
        print("  gB : "); print(gB); print("\n");
        print("  sB : "); print(sB); print("\n");
        print("tBgB : "); print(tBgB); print("\n");
        print("tBsB : "); print(tBsB); print("\n");
        print("tArA : "); print(tArA); print("\n");
    }
#endif

#if 0
    if (thread0()) {
        print("  mC : "); print(mC); print("\n");
        print("  gC : "); print(gC); print("\n");
        print("tCsA : "); print(tCsA); print("\n");
        print("tCsB : "); print(tCsB); print("\n");
        print("tCgC : "); print(tCgC); print("\n");
        print("tCrC : "); print(tCrC); print("\n");
    }
#endif

#if 1

    // Copy rmem to smem
    copy(copy_a, tArA, tAsA);
    copy(copy_b, tBrB, tBsB);
    __syncthreads();

    //
    // PIPELINED MAIN LOOP

    // Load A, B shmem->regs for k_block=0
    copy(copy_mma_a, tCsA(_, _, 0), tCrA_view(_, _, 0));
    copy(copy_mma_b, tCsB(_, _, 0), tCrB_view(_, _, 0));

    auto K_TILE_MAX = size<3>(tAgA);
    auto K_BLOCK_MAX = size<2>(tCrA);

    CUTE_NO_UNROLL
    for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile) {
        // Pipeline the k-mode of the block registers
        CUTE_UNROLL
        for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
            if (k_block == K_BLOCK_MAX - 1) {
                // Copy rmem to smem
                __syncthreads();
                copy(copy_a, tArA, tAsA);
                copy(copy_b, tBrB, tBsB);
                __syncthreads();
            }

            // Copy smem to rmem for k_block+1
            int k_block_next = (k_block + 1) % K_BLOCK_MAX;
            copy(copy_mma_a, tCsA(_, _, k_block_next), tCrA_view(_, _, k_block_next));
            copy(copy_mma_b, tCsB(_, _, k_block_next), tCrB_view(_, _, k_block_next));

            if (k_block == 0) {
                // Copy gmem to rmem for k_tile+1
                int k_tile_next = (k_tile + 1 < K_TILE_MAX) ? k_tile + 1 : k_tile;
                copy(copy_a, tAgA(_, _, _, k_tile_next), tArA);
                copy(copy_b, tBgB(_, _, _, k_tile_next), tBrB);
            }

            // Thread-level register gemm for k_block
            gemm(mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrC);
        } // k_block
    } // k_tile

#if 0
    if (thread(0)) {
        print_tensor(tCrC); printf("\n");
    }
#endif

    // Epilogue

    // Shared memory buffers
    __shared__ TC smemC[cosize_v<CSmemLayout>];
    Tensor sC = make_tensor(make_smem_ptr(smemC), sC_layout);            // (BLK_M,BLK_N)

    // Define copy for smem to gmem
    ThrCopy thr_copy_c = copy_c.get_slice(threadIdx.x);
    Tensor tDgC = thr_copy_c.partition_S(gC);                            // (CPY,CPY_N,CPY_K,k)
    Tensor tDsC = thr_copy_c.partition_D(sC);                            // (CPY,CPY_N,CPY_K)

    CUTE_STATIC_ASSERT_V(size<1>(tDgC) == size<1>(tDsC));                // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tDgC) == size<2>(tDsC));                // CPY_N

#if 0
    if (thread0()) {
        print("tDgC : "); print(tDgC); print("\n");
        print("tDsC : "); print(tDsC); print("\n");
    }
#endif

    // Copy gmem to smem
    copy(copy_c, tDgC, tDsC);

    __syncthreads();

    // Define mma copy
    Tensor tDrC = thr_mma.partition_fragment_C(gC);           // (MMA, MMA_M, MMA_N)

    ThrCopy thr_mma_copy_c = copy_mma_c.get_slice(threadIdx.x);
    Tensor tCsC = thr_mma_copy_c.partition_S(sC);             // (CPY, CPY_M, CPY_N)
    Tensor tDrC_view = thr_mma_copy_c.retile_D(tDrC);         // (CPY, CPY_M, CPY_N)

    copy(copy_mma_c, tCsC, tDrC_view);

    axpby(alpha, tCrC, beta, tDrC);

    copy(copy_mma_c, tDrC_view, tCsC);

    __syncthreads();

    // Copy smem to gmem
    copy(copy_c, tDsC, tDgC);

#endif
}


//
// Direct Copy for any specific types
//

namespace cute {

template <class S, class D = S, int value = 0>
struct UniversalCopyClearSrc {
    using SRegisters = S[1];
    using DRegisters = D[1];

    // Sanity
    static_assert(sizeof_bits_v<S> >= 8);
    static_assert(sizeof_bits_v<D> >= 8);
    
    CUTE_HOST_DEVICE static constexpr void
    copy(S & src,
         D & dst) {
        dst = src;
        src = S(value);
    }
};

template <class S, class D, int value>
struct Copy_Traits<UniversalCopyClearSrc<S, D, value>>
     : Copy_Traits<UniversalCopy<S, D>> {};

template <class S, class D = S>
struct UniversalCopyAtomicAdd {
    using SRegisters = S[1];
    using DRegisters = D[1];

    // Sanity
    static_assert(sizeof_bits_v<S> >= 8);
    static_assert(sizeof_bits_v<D> >= 8);
    
    CUTE_HOST_DEVICE static constexpr void
    copy(S const& src,
         D      & dst) {
        atomicAdd(&dst, src);
    }
};

template <class S, class D>
struct Copy_Traits<UniversalCopyAtomicAdd<S, D>>
     : Copy_Traits<UniversalCopy<S, D>> {};

}

// Setup params for a NT GEMM
template <
    class TA, class TB, class TC,
    class Alpha, class Beta>
void
gemm_nt(
    int m, int n, int k,
    Alpha alpha,
    TA const* A, int ldA,
    TB const* B, int ldB,
    Beta beta,
    TC* C, int ldC,
    cudaStream_t stream = 0) {
    using namespace cute;

    // Define shapes (dynamic)
    auto M = int(m);
    auto N = int(n);
    auto K = int(k);
    auto SpiltK = int(2);
    auto prob_shape = make_shape(M, N, K);                     // (M, N, K)
    auto split_shape = make_shape(M, N, K / SpiltK);           // (M, N, SpiltK)

    // Define NT strides (mixed)
    auto dA = make_stride(Int<1>{}, ldA);                      // (dM, dK)
    auto dB = make_stride(Int<1>{}, ldB);                      // (dN, dK)
    auto dC = make_stride(Int<1>{}, ldC);                      // (dM, dN)

    // Define CTA tile sizes (static)
    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int< 32>{};
    auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)

    // Define the smem layouts (static)
    auto sA = make_layout(make_shape(bM, bK));                 // (m,k) -> smem_idx; m-major
    auto sB = make_layout(make_shape(bN, bK));                 // (n,k) -> smem_idx; n-major
    auto sC = make_layout(make_shape(bM, bN));                 // (m,n) -> smem_idx; m-major

    // Define the thread layouts (static)
    TiledCopy copyA = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, TA>{},
        Layout<Shape<_32, _4>>{},  // Thr layout 32x8 m-major
        Layout<Shape< _8, _1>>{}); // Val layout  4x1 m-major
    TiledCopy copyB = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, TB>{},
        Layout<Shape<_32, _4>>{},  // Thr layout 32x8 n-major
        Layout<Shape< _8, _1>>{}); // Val layout  4x1 n-major

    using mma_op = SM80_16x8x16_F16F16F16F16_TN;

    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;

    static constexpr int kMmaEURepeatM = 2;
    static constexpr int kMmaEURepeatN = 2;
    static constexpr int kMmaEURepeatK = 1;

    using mma_atom_shape = mma_traits::Shape_MNK;
    static constexpr int kMmaPM = 1 * kMmaEURepeatM * get<0>(mma_atom_shape{});
    static constexpr int kMmaPN = 2 * kMmaEURepeatN * get<1>(mma_atom_shape{});
    static constexpr int kMmaPK = 1 * kMmaEURepeatK * get<2>(mma_atom_shape{});

    using MMA_EU_RepeatT = decltype(make_layout(make_shape(
        Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
    using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;

    TiledMMA mmaC = make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{});  // 16x16x1 TiledMMA

#if 0
    print(copyA);
    print(copyB);
    print(mmaC);
#endif

#if 0
    print_latex(copyA);
    print_latex(copyB);
    print_latex(mmaC);
#endif

    dim3 dimBlock(size(mmaC));
    dim3 dimGrid(size(ceil_div(M, bM)),
                 size(ceil_div(N, bN)));

    /*
    gemm_device_2stage_aligned <<<dimGrid, dimBlock, 0, stream >>>
        (prob_shape, cta_tiler,
            A, dA, sA, copyA,
            B, dB, sB, copyB,
            C, dC, sC, mmaC,
            alpha, beta);
    */
}


// Setup params for a TN GEMM
template <
    class TA, class TB, class TC,
    class Alpha, class Beta>
void
gemm_tn(
    int m, int n, int k,
    Alpha alpha,
    TA const* A, int ldA,
    TB const* B, int ldB,
    Beta beta,
    TC* C, int ldC,
    cudaStream_t stream = 0) {
    using namespace cute;

    // Define shapes (dynamic)
    auto M = int(m);
    auto N = int(n);
    auto K = int(k);
    auto SpiltK = int(2);
    auto prob_shape = make_shape(M, N, K);                     // (M, N, K)
    auto split_shape = make_shape(M, N, K / SpiltK);           // (M, N, SpiltK)

    // Define TN strides (mixed)
    auto dA = make_stride(ldA, Int<1>{});                      // (dM, dK)
    auto dB = make_stride(ldB, Int<1>{});                      // (dN, dK)
    auto dC = make_stride(Int<1>{}, ldC);                      // (dM, dN)

    // Define CTA tile sizes (static)
    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int< 32>{};
    auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)

    // Define the smem layouts (static)
    auto sA = make_layout(make_shape(bM, bK),
        make_stride(bK, Int<1>{}));        // (m,k) -> smem_idx; padded m-major
    auto sB = make_layout(make_shape(bN, bK),
        make_stride(bK, Int<1>{}));        // (n,k) -> smem_idx; padded n-major
    auto sC = make_layout(make_shape(bM, bN));                        // (m,n) -> smem_idx

    // Define the thread layouts (static)

    TiledCopy copyA = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, TA>{},
        Layout<Shape<_32, _4>, Stride<_4, _1>>{}, // Thr layout 32x8 k-major
        Layout<Shape< _1, _8>>{});                // Val layout  1x1
    TiledCopy copyB = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, TB>{},
        Layout<Shape<_32, _4>, Stride<_4, _1>>{}, // Thr layout 32x8 k-major
        Layout<Shape< _1, _8>>{});                // Val layout  1x1
    TiledCopy copyC = make_tiled_copy(Copy_Atom<UniversalCopyClearSrc<uint128_t>, TC>{},
        Layout<Shape<_16, _8>>{},  // Thr layout 32x8 k-major
        Layout<Shape< _8, _1>>{}); // Val layout  1x1
    TiledCopy copyDestC = make_tiled_copy(Copy_Atom<UniversalCopyAtomicAdd<half2>, TC>{},
        Layout<Shape<_64, _2>>{},  // Thr layout 32x8 k-major
        Layout<Shape< _2, _1>>{}); // Val layout  1x1

    // Define the mma
    using mma_traits = MMA_Traits<SM80_16x8x16_F16F16F16F16_TN>;
    using mma_atom = MMA_Atom<mma_traits>;

    static constexpr int kMmaEURepeatM = 2;
    static constexpr int kMmaEURepeatN = 2;
    static constexpr int kMmaEURepeatK = 1;

    using mma_atom_shape = mma_traits::Shape_MNK;
    static constexpr int kMmaPM = 1 * kMmaEURepeatM * get<0>(mma_atom_shape{});
    static constexpr int kMmaPN = 2 * kMmaEURepeatN * get<1>(mma_atom_shape{});
    static constexpr int kMmaPK = 1 * kMmaEURepeatK * get<2>(mma_atom_shape{});

    using MMA_EU_RepeatT = decltype(make_layout(make_shape(
        Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
    using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;

    TiledMMA mmaC = make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{});  // 16x16x1 TiledMMA

    // mma copy
    TiledCopy mmaCopyA = make_tiled_copy_A(Copy_Atom<Copy_Traits<SM75_U32x4_LDSM_N>, TA>{}, mmaC);
    TiledCopy mmaCopyB = make_tiled_copy_B(Copy_Atom<Copy_Traits<SM75_U32x4_LDSM_N>, TB>{}, mmaC);
    TiledCopy mmaCopyC = make_tiled_copy_C(Copy_Atom<Copy_Traits<UniversalCopy<uint16_t>>, TC>{}, mmaC);

#if 0
    print(copyA);
    print(copyB);
    print(mmaC);
#endif

#if 0
    print_latex(copyA);
    print_latex(copyB);
    print_latex(mmaC);
#endif

    dim3 dimBlock(size(mmaC));
    dim3 dimGrid(size(ceil_div(M, bM)),
                 size(ceil_div(N, bN)),
                 SpiltK);

    gemm_device_twostage_aligned_splitk <<<dimGrid, dimBlock, 0, stream>>> (
        prob_shape, split_shape, cta_tiler,
        A, dA, sA, copyA,
        B, dB, sB, copyB,
        C, dC, sC, copyC, copyDestC,
        mmaC, mmaCopyA, mmaCopyB, mmaCopyC,
        alpha, beta);
}

template <
    class TA, class TB, class TC,
    class Alpha, class Beta>
void
gemm(
    char transA, char transB, int m, int n, int k,
    Alpha alpha,
    TA const* A, int ldA,
    TB const* B, int ldB,
    Beta beta,
    TC* C, int ldC,
    cudaStream_t stream = 0) {
    if (transA == 'N' && transB == 'T') {
        return gemm_nt(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
    }
    else if (transA == 'T' && transB == 'N') {
        return gemm_tn(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
    }
    assert(false && "Not implemented");
}


int main(int argc, char** argv) {
    cudaDeviceProp props;
    cudaError_t error = cudaGetDeviceProperties(&props, 0);
    if (error != cudaSuccess) {
        std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    if (props.major < 7) {
        std::cout << "This example requires an Volta GPU or newer (CC >= 70)" << std::endl;
        // Return 0 so tests pass if run on unsupported architectures or CUDA Toolkits.
        return 0;
    }

    int m = 128;
    if (argc >= 2)
        sscanf(argv[1], "%d", &m);

    int n = 128;
    if (argc >= 3)
        sscanf(argv[2], "%d", &n);

    int k = 128;
    if (argc >= 4)
        sscanf(argv[3], "%d", &k);

    char transA = 'T';
    if (argc >= 5)
        sscanf(argv[4], "%c", &transA);

    char transB = 'N';
    if (argc >= 6)
        sscanf(argv[5], "%c", &transB);

    using TA = cute::half_t;
    using TB = cute::half_t;
    using TC = cute::half_t;
    using TI = cute::half_t;

    TI alpha(2.0f);
    TI beta(3.0f);

    std::cout << "M = " << m << std::endl;
    std::cout << "N = " << n << std::endl;
    std::cout << "K = " << k << std::endl;
    std::cout << "C = A^" << transA << " B^" << transB << std::endl;

    thrust::host_vector<TA> h_A(m * k);
    thrust::host_vector<TB> h_B(n * k);
    thrust::host_vector<TC> h_C(m * n);
    thrust::host_vector<TC> h_C1 = h_C;

    for (int j = 0; j < m * k; ++j) h_A[j] = j / 1000.0f;
    for (int j = 0; j < n * k; ++j) h_B[j] = (j + 1) / 1000.0f;
    for (int j = 0; j < m * n; ++j) h_C[j] = (j + 2) / 1000.0f;

    thrust::device_vector<TA> d_A = h_A;
    thrust::device_vector<TB> d_B = h_B;
    thrust::device_vector<TC> d_C = h_C;
    thrust::device_vector<TC> d_C1 = h_C;

    double gflops = (2.0 * m * n * k) * 1e-9;

    const int timing_iterations = 0;
    GPU_Clock timer;

    int ldA = 0, ldB = 0, ldC = m;

    if (transA == 'N') {
        ldA = m;
    }
    else if (transA == 'T') {
        ldA = k;
    }
    else {
        assert(false);
    }

    if (transB == 'N') {
        ldB = k;
    }
    else if (transB == 'T') {
        ldB = n;
    }
    else {
        assert(false);
    }

    // Run once
    d_C = h_C;
    gemm(transA, transB, m, n, k,
        alpha,
        d_A.data().get(), ldA,
        d_B.data().get(), ldB,
        beta,
        d_C.data().get(), ldC);
    CUTE_CHECK_LAST();
    h_C = d_C;

    // Timing iterations
    timer.start();
    for (int i = 0; i < timing_iterations; ++i) {
        gemm(transA, transB, m, n, k,
            alpha,
            d_A.data().get(), ldA,
            d_B.data().get(), ldB,
            beta,
            d_C.data().get(), ldC);
    }
    double cute_time = timer.seconds() / timing_iterations;
    CUTE_CHECK_LAST();
    printf("CUTE_GEMM:     [%6.1f]GFlop/s  (%6.4f)ms\n", gflops / cute_time, cute_time * 1000);

    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasStatus_t ret = cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        m, n, k,
        (half*)&alpha,
        (half*)d_A.data().get(), k,
        (half*)d_B.data().get(), k,
        (half*)&beta,
        (half*)d_C1.data().get(), m);

    cublasDestroy(handle);

    h_C1 = d_C1;

    auto tC_host = cute::make_tensor(h_C.data(), cute::make_shape(m, n), cute::make_stride(1, m));
    auto tC1_host = cute::make_tensor(h_C1.data(), cute::make_shape(m, n), cute::make_stride(1, m));
    auto tile = cute::make_tile(min(8, m), min(8, n));
    auto t32x32 = local_tile(tC_host, tile, cute::make_coord(0, 0));
    auto t32x32_cublas = local_tile(tC1_host, tile, cute::make_coord(0, 0));

    cute::print_tensor(t32x32); printf("\n");
    cute::print_tensor(t32x32_cublas); printf("\n");

    return 0;
}
