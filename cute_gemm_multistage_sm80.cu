#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cute/tensor.hpp>

#include <cublas_v2.h>


template <
    class Shape, class SplitkShape, class SplitkCoord,
    class CtaTiler, class CtaCoord, class CtaStep, class ThrCopy>
CUTE_HOST_DEVICE constexpr
auto
make_pred_splitk(
    Shape const& shape, SplitkShape const& splitk_shape, SplitkCoord const& splitk_coord,
    CtaTiler const& cta_tiler, CtaCoord const& cta_coord, CtaStep const& cta_step, ThrCopy const& thr_copy) {
    using namespace cute;

    Tensor m = make_identity_tensor(shape);
    Tensor mSplit = local_tile(m, splitk_shape, splitk_coord);
    Tensor cSplit = local_tile(mSplit, cta_tiler, cta_coord, cta_step);
    return thr_copy.partition_S(cSplit);
}

template <
    class Shape,
    class CtaTiler, class CtaCoord, class CtaStep,
    class ThrCopy>
CUTE_HOST_DEVICE constexpr
auto
make_pred(
    Shape const& shape,
    CtaTiler const& cta_tiler, CtaCoord const& cta_coord, CtaStep const& cta_step,
    ThrCopy const& thr_copy, bool src) {
    using namespace cute;

    Tensor m = make_identity_tensor(shape);
    Tensor c = local_tile(m, cta_tiler, cta_coord, cta_step);
    return src ? thr_copy.partition_S(c) : thr_copy.partition_D(c);
}


template <
    class ProblemShape, class CtaTiler,
    class TA, class AStride, class ASmemLayout, class TiledCopyA,
    class TB, class BStride, class BSmemLayout, class TiledCopyB,
    class TC, class CStride, class CSmemLayout, class TiledCopyC,
    class TiledMma, class MmaCopyA, class MmaCopyB, class MmaCopyC>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void gemm_device_twostage_slicek(
    ProblemShape shape_MNK, CtaTiler cta_tiler,
    TA const* A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a,
    TB const* B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b,
    TC* C, CStride dC, CSmemLayout sC_layout, TiledCopyC copy_c,
    TiledMma mma, MmaCopyA copy_mma_a, MmaCopyB copy_mma_b, MmaCopyC copy_mma_c) {
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
    CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
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
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);                 // (m,n,k)
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1,  X, _1>{});   // (BLK_M,BLK_K,k)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X, _1, _1>{});   // (BLK_N,BLK_K,k)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1,  X>{});   // (BLK_M,BLK_N)

    // Shared memory buffers
    constexpr int smem_size_a = cosize_v<ASmemLayout> * sizeof(TA);
    constexpr int smem_size_b = cosize_v<BSmemLayout> * sizeof(TB);
    constexpr int smem_size_c = cosize_v<CSmemLayout> * sizeof(TC);
    constexpr int smem_size = cute::max(smem_size_a + smem_size_b, smem_size_c);

    __shared__ uint8_t smem[smem_size];
    TA* smemA = (TA*)smem;
    TB* smemB = (TB*)(smem + smem_size_a);
    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);            // (BLK_M,BLK_K)
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);            // (BLK_N,BLK_K)

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
    auto tAcA = make_pred(
        shape(mA),
        cta_tiler, cta_coord, Step<_1, X, _1>{},
        thr_copy_a, true);

    auto tApA = [&](auto... coords) {
        return elem_less(tAcA(coords...), shape(mA));
    };

    clear(tArA);
    copy_if(copy_a, tApA, tAgA(_, _, _, 0), tArA);

    auto tBcB = make_pred(
        shape(mB),
        cta_tiler, cta_coord, Step< X, _1, _1>{},
        thr_copy_b, true);

    auto tBpB = [&](auto... coords) {
        return elem_less(tBcB(coords...), shape(mB));
    };

    clear(tBrB);
    copy_if(copy_b, tBpB, tBgB(_, _, _, 0), tBrB);

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

                clear(tArA);
                copy_if(copy_a, tApA, tAgA(_, _, _, k_tile_next), tArA);

                clear(tBrB);
                copy_if(copy_b, tBpB, tBgB(_, _, _, k_tile_next), tBrB);
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

#if 1
    // Epilogue

    // Shared memory buffers
    TC* smemC = (TC*)smem;
    Tensor sC = make_tensor(make_smem_ptr(smemC), sC_layout); // (BLK_M,BLK_N)

    // Define mma copy
    ThrCopy thr_mma_copy_c = copy_mma_c.get_slice(threadIdx.x);
    Tensor tCrC_view = thr_mma_copy_c.retile_S(tCrC);         // (CPY, CPY_M, CPY_N)
    Tensor tCsC_mma = thr_mma_copy_c.partition_D(sC);         // (CPY, _1, _1, pipe)
    Tensor tCrC_viewx = group_modes<1, 3>(tCrC_view);         // (CPY, CPY_MN)

    ThrCopy thr_copy_c = copy_c.get_thread_slice(threadIdx.x);
    Tensor tCsC = thr_copy_c.partition_S(sC);                 // (CPY, _1, _1, pipe)
    Tensor tCgC_copy = thr_copy_c.partition_D(gC);            // (CPY, CPY_M, CPY_N)
    Tensor tCgC_copy_x = group_modes<1, 3>(tCgC_copy);        // (CPY_, CPY_MN)

    CUTE_STATIC_ASSERT_V(size<1>(tCrC_viewx) == size<1>(tCgC_copy_x));        // CPY_MN
    CUTE_STATIC_ASSERT_V(size<3>(tCsC_mma) == size<3>(tCsC));                 // pipe
    CUTE_STATIC_ASSERT_V((size<1>(tCrC_viewx) % size<3>(tCsC_mma)) == _0{});  // CPY_MN % pipe == 0
    CUTE_STATIC_ASSERT_V((size<1>(tCgC_copy_x) % size<3>(tCsC_mma)) == _0{}); // CPY_MN % pipe == 0

    auto tCcC = make_pred(
        shape(mC),
        cta_tiler, cta_coord, Step<_1, _1,  X>{},
        thr_copy_c, false);
    auto tCcC_view = group_modes<1, 3>(tCcC);

    int step = size<3>(tCsC_mma);  // pipe

#if 0
    if (thread0()) {
        print("   tCsC_mma : "); print(tCsC_mma); printf("\n");
        print(" tCrC_viewx : "); print(tCrC_viewx); printf("\n");
        print("       tCsC : "); print(tCsC); printf("\n");
        print("         gC : "); print(gC); printf("\n");
        print("  tCgC_copy : "); print(tCgC_copy); printf("\n");
        print("tCgC_copy_x : "); print(tCgC_copy_x); printf("\n");
        print("     copy_c : "); print(copy_c); printf("\n");
    }
#endif

    CUTE_UNROLL
    for (int i = 0; i < size<1>(tCrC_viewx); i += step) {
        // reg -> shm
        CUTE_UNROLL
        for (int j = 0; j < step; ++j) {
            // we add a temp tensor to cope with accumulator and output data type
            // difference
            Tensor t = make_tensor_like<TC>(tCrC_viewx(_, i + j));
            copy(tCrC_viewx(_, i + j), t);

            copy(copy_mma_c, t, tCsC_mma(_, 0, 0, j));
        }
        __syncthreads();

        // shm -> global
        CUTE_UNROLL
        for (int j = 0; j < step; ++j) {
            auto tCpC = [&](auto... coords) {
                auto pred = tCcC_view(_, i + j);
                return elem_less(pred(_0{}, coords...), shape(mC));
            };

            copy_if(copy_c, tCpC, tCsC(_, 0, 0, j), tCgC_copy_x(_, i + j));
        }

        __syncthreads();

    }
#endif

#endif
}

template <
    class ProblemShape, class CtaTiler,
    class TA, class AStride, class ASmemLayout, class TiledCopyA,
    class TB, class BStride, class BSmemLayout, class TiledCopyB,
    class TC, class CStride, class CSmemLayout, class TiledCopyC,
    class TiledMma, class MmaCopyA, class MmaCopyB, class MmaCopyC>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void gemm_device_multistage_slicek(
    ProblemShape shape_MNK, CtaTiler cta_tiler,
    TA const* A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a,
    TB const* B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b,
    TC* C, CStride dC, CSmemLayout sC_layout, TiledCopyC copy_c,
    TiledMma mma, MmaCopyA copy_mma_a, MmaCopyB copy_mma_b, MmaCopyC copy_mma_c) {
    using namespace cute;

    // Preconditions
    CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});                   // (M, N, K)
    CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});                   // (BLK_M, BLK_N, BLK_K)

    CUTE_STATIC_ASSERT_V(size(copy_a) == size(mma));                     // NumThreads
    CUTE_STATIC_ASSERT_V(size(copy_b) == size(mma));                     // NumThreads

    static_assert(is_static<ASmemLayout>::value);
    static_assert(is_static<BSmemLayout>::value);
    static_assert(is_static<CSmemLayout>::value);

    CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler));      // BLK_M
    CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler));      // BLK_N
    CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler));      // BLK_K
    CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler));      // BLK_K
    CUTE_STATIC_ASSERT_V(size<2>(ASmemLayout{}) == size<2>(BSmemLayout{}));  // PIPE

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
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);                 // (m,n,k)
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});   // (BLK_M,BLK_K,k)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X, _1, _1>{});   // (BLK_N,BLK_K,k)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});   // (BLK_M,BLK_N)

    // global memory -> shared memory
    // Shared memory buffers
    constexpr int smem_size_a = cosize_v<ASmemLayout> *sizeof(TA);
    constexpr int smem_size_b = cosize_v<BSmemLayout> *sizeof(TB);
    constexpr int smem_size_c = cosize_v<CSmemLayout> *sizeof(TC);
    constexpr int smem_size = cute::max(smem_size_a + smem_size_b, smem_size_c);

    __shared__ uint8_t smem[smem_size];
    TA* smemA = (TA*)smem;
    TB* smemB = (TB*)(smem + smem_size_a);
    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);            // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);            // (BLK_N,BLK_K,PIPE)

    ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
    Tensor tAgA = thr_copy_a.partition_S(gA);                            // (CPY,CPY_M,CPY_K,k)
    Tensor tAsA = thr_copy_a.partition_D(sA);                            // (CPY,CPY_M,CPY_K,PIPE)

    ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
    Tensor tBgB = thr_copy_b.partition_S(gB);                            // (CPY,CPY_M,CPY_K,k)
    Tensor tBsB = thr_copy_b.partition_D(sB);                            // (CPY,CPY_N,CPY_K,PIPE)
    
    clear(tAsA);
    clear(tBsB);

    CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA));                // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tAsA));                // CPY_K
    CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB));                // CPY_N
    CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBsB));                // CPY_K

    // gmem -> smem
    int istage = 0;
    int itile_to_read = 0;
    int ismem_read = 0;
    int ismem_write = 0;
    auto kstage = max(size<3>(tAsA), _2{});

    // Copy gmem to smem for kstage-1
    auto tAcA = make_pred(
        shape(mA),
        cta_tiler, cta_coord, Step<_1, X, _1>{},
        thr_copy_a, true);

    auto tApA = [&](auto... coords) {
        return elem_less(tAcA(coords...), shape(mA));
    };

    auto tBcB = make_pred(
        shape(mB),
        cta_tiler, cta_coord, Step< X, _1, _1>{},
        thr_copy_b, true);

    auto tBpB = [&](auto... coords) {
        return elem_less(tBcB(coords...), shape(mB));
    };

    CUTE_UNROLL
    for (; istage < kstage - _1{}; ++istage) {
        copy_if(copy_a, tApA, tAgA(_, _, _, istage), tAsA(_, _, _, istage));
        copy_if(copy_b, tBpB, tBgB(_, _, _, istage), tBsB(_, _, _, istage));

        cp_async_fence();

        ++itile_to_read;
        ++ismem_write;
    }

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
    // Size of the register pipeline
    auto nk = size<2>(tCrA);

    // PREFETCH register pipeline
    if (nk > 1) {
        // Wait until our first prefetched tile is loaded in
        cp_async_wait<kstage - 2>();
        __syncthreads();

        // Prefetch the first rmem from the first k-tile
        copy(copy_mma_a, tCsA(_, _, _0{}, ismem_read), tCrA_view(_, _, _0{}));
        copy(copy_mma_b, tCsB(_, _, _0{}, ismem_read), tCrB_view(_, _, _0{}));
    }

    // loop over k: i. load tile, ii. mma
    auto ntile = size<3>(tAgA);
    
    CUTE_NO_UNROLL
    for (int itile = 0; itile < ntile; ++itile) {
        CUTE_UNROLL
        for (int ik = 0; ik < nk; ++ik) {
            int ik_next = (ik + 1) % nk;

            if (ik == nk - 1) {
                cp_async_wait<kstage - 2>();
                __syncthreads();

                ismem_read = (ismem_read + 1) % kstage;
            }

            // shm -> reg s[itile][ik + 1] -> r[ik + 1]
            copy(copy_mma_a, tCsA(_, _, ik_next, ismem_read), tCrA_view(_, _, ik_next));
            copy(copy_mma_b, tCsB(_, _, ik_next, ismem_read), tCrB_view(_, _, ik_next));

            if (ik == 0) {
                if (itile_to_read < ntile) {
                    copy_if(copy_a, tApA, tAgA(_, _, _, itile_to_read), tAsA(_, _, _, ismem_write));
                    copy_if(copy_b, tBpB, tBgB(_, _, _, itile_to_read), tBsB(_, _, _, ismem_write));

                    ++itile_to_read;
                    ismem_write = (ismem_write + 1) % kstage;
                }

                cp_async_fence();
            }

            gemm(mma, tCrC, tCrA(_, _, ik), tCrB(_, _, ik), tCrC);
        }  // for ik
    }      // itile

#if 0
    if (thread(0)) {
        print_tensor(tCrC); printf("\n");
    }
#endif

#if 1
    // Epilogue

    // Shared memory buffers
    TC* smemC = (TC*)smem;
    Tensor sC = make_tensor(make_smem_ptr(smemC), sC_layout); // (BLK_M,BLK_N)

    // Define mma copy
    ThrCopy thr_mma_copy_c = copy_mma_c.get_slice(threadIdx.x);
    Tensor tCrC_view = thr_mma_copy_c.retile_S(tCrC);         // (CPY, CPY_M, CPY_N)
    Tensor tCsC_mma = thr_mma_copy_c.partition_D(sC);         // (CPY, _1, _1, pipe)
    Tensor tCrC_viewx = group_modes<1, 3>(tCrC_view);         // (CPY, CPY_MN)

    ThrCopy thr_copy_c = copy_c.get_thread_slice(threadIdx.x);
    Tensor tCsC = thr_copy_c.partition_S(sC);                 // (CPY, _1, _1, pipe)
    Tensor tCgC_copy = thr_copy_c.partition_D(gC);            // (CPY, CPY_M, CPY_N)
    Tensor tCgC_copy_x = group_modes<1, 3>(tCgC_copy);        // (CPY_, CPY_MN)

    CUTE_STATIC_ASSERT_V(size<1>(tCrC_viewx) == size<1>(tCgC_copy_x));        // CPY_MN
    CUTE_STATIC_ASSERT_V(size<3>(tCsC_mma) == size<3>(tCsC));                 // pipe
    CUTE_STATIC_ASSERT_V((size<1>(tCrC_viewx) % size<3>(tCsC_mma)) == _0{});  // CPY_MN % pipe == 0
    CUTE_STATIC_ASSERT_V((size<1>(tCgC_copy_x) % size<3>(tCsC_mma)) == _0{}); // CPY_MN % pipe == 0

    auto tCcC = make_pred(
        shape(mC),
        cta_tiler, cta_coord, Step<_1, _1,  X>{},
        thr_copy_c, false);
    auto tCcC_view = group_modes<1, 3>(tCcC);

    int step = size<3>(tCsC_mma);  // pipe

#if 0
    if (thread0()) {
        print("   tCsC_mma : "); print(tCsC_mma); printf("\n");
        print(" tCrC_viewx : "); print(tCrC_viewx); printf("\n");
        print("       tCsC : "); print(tCsC); printf("\n");
        print("         gC : "); print(gC); printf("\n");
        print("  tCgC_copy : "); print(tCgC_copy); printf("\n");
        print("tCgC_copy_x : "); print(tCgC_copy_x); printf("\n");
        print("     copy_c : "); print(copy_c); printf("\n");
    }
#endif

    CUTE_UNROLL
    for (int i = 0; i < size<1>(tCrC_viewx); i += step) {
        // reg -> shm
        CUTE_UNROLL
        for (int j = 0; j < step; ++j) {
            // we add a temp tensor to cope with accumulator and output data type
            // difference
            Tensor t = make_tensor_like<TC>(tCrC_viewx(_, i + j));
            copy(tCrC_viewx(_, i + j), t);

            copy(copy_mma_c, t, tCsC_mma(_, 0, 0, j));
        }
        __syncthreads();

        // shm -> global
        CUTE_UNROLL
        for (int j = 0; j < step; ++j) {
            auto tCpC = [&](auto... coords) {
                auto pred = tCcC_view(_, i + j);
                return elem_less(pred(_0{}, coords...), shape(mC));
            };

            copy_if(copy_c, tCpC, tCsC(_, 0, 0, j), tCgC_copy_x(_, i + j));
        }

        __syncthreads();

    }
#endif

#endif
}

// Setup params for a TN GEMM
template <
    class TA, class TB, class TC>
void
gemm_tn_twostage(
    int m, int n, int k,
    TA const* A, int ldA,
    TB const* B, int ldB,
    TC* C, int ldC,
    cudaStream_t stream = 0) {
    using namespace cute;

    // Define shapes (dynamic)
    auto M = int(m);
    auto N = int(n);
    auto K = int(k);
    auto prob_shape = make_shape(M, N, K);                     // (M, N, K)

    // Define TN strides (mixed)
    auto dA = make_stride(ldA, Int<1>{});                      // (dM, dK)
    auto dB = make_stride(ldB, Int<1>{});                      // (dN, dK)
    auto dC = make_stride(Int<1>{}, ldC);                      // (dM, dN)

    // Define CTA tile sizes (static)
    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int< 64>{};
    auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)

    // Define the smem layouts (static)
    auto sA = composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(bM, bK),
                    make_stride(bK, Int<1>{})));        // (m,k) -> smem_idx; padded m-major
    auto sB = composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(bN, bK),
                    make_stride(bK, Int<1>{})));        // (n,k) -> smem_idx; padded n-major

    // Define the thread layouts (static)

    TiledCopy copyA = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, TA>{},
        Layout<Shape<_16, _8>, Stride<_8, _1>>{}, // Thr layout 32x8 k-major
        Layout<Shape< _1, _8>>{});                // Val layout  1x1
    TiledCopy copyB = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, TB>{},
        Layout<Shape<_16, _8>, Stride<_8, _1>>{}, // Thr layout 32x8 k-major
        Layout<Shape< _1, _8>>{});                // Val layout  1x1
    TiledCopy copyC = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, TC>{},
        Layout<Shape<_4, _32>>{},                 // Thr layout 32x8 k-major
        Layout<Shape<_8,  _1>>{});                // Val layout  1x1


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
    TiledCopy mmaCopyC = make_tiled_copy_C(Copy_Atom<Copy_Traits<UniversalCopy<TC>>, TC>{}, mmaC);

    // batched copy for C
    constexpr int kSmemLayoutCBatch = 2;
    auto SmemLayoutAtomC = composition(
        Swizzle<2, 3, 3>{},
        make_layout(make_shape(Int<kMmaPM>{}, Int<kMmaPN>{})));
    auto sC = tile_to_shape(SmemLayoutAtomC,
        make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}, Int<kSmemLayoutCBatch>{}));       // (m,n) -> smem_idx

#if 0
    print(copyA);
    print(copyB);
    print(copyC);
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

    gemm_device_twostage_slicek <<<dimGrid, dimBlock, 0, stream>>> (
        prob_shape, cta_tiler,
        A, dA, sA, copyA,
        B, dB, sB, copyB,
        C, dC, sC, copyC,
        mmaC, mmaCopyA, mmaCopyB, mmaCopyC);

}

// Setup params for a TN GEMM
template <
    class TA, class TB, class TC>
void
gemm_tn_multistage(
    int m, int n, int k,
    TA const* A, int ldA,
    TB const* B, int ldB,
    TC* C, int ldC,
    cudaStream_t stream = 0) {
    using namespace cute;

    // Define shapes (dynamic)
    auto M = int(m);
    auto N = int(n);
    auto K = int(k);
    auto prob_shape = make_shape(M, N, K);                     // (M, N, K)

    // Define TN strides (mixed)
    auto dA = make_stride(ldA, Int<1>{});                      // (dM, dK)
    auto dB = make_stride(ldB, Int<1>{});                      // (dN, dK)
    auto dC = make_stride(Int<1>{}, ldC);                      // (dM, dN)

    // Define CTA tile sizes (static)
    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int< 32>{};
    auto bP = Int<  3>{};
    auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)

    // Define the smem layouts (static)
    // Swizzles for LDSM and 128b k-major loads
    auto swizzle_atom_a = composition(Swizzle<3, 3, 3>{},
                                      make_layout(make_shape(Int<8>{}, Int<bK>{}),
                                                             GenRowMajor{}));
    auto swizzle_atom_b = composition(Swizzle<3, 3, 3>{},
                                      make_layout(make_shape(Int<8>{}, Int<bK>{}),
                                                             GenRowMajor{}));
    auto sA = tile_to_shape(swizzle_atom_a, make_shape(bM, bK, bP));
    auto sB = tile_to_shape(swizzle_atom_b, make_shape(bN, bK, bP));

    // Define the thread layouts (static)

    TiledCopy copyA = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, TA>{},
        Layout<Shape<_32, _4>, Stride<_4, _1>>{}, // Thr layout 32x8 k-major
        Layout<Shape< _1, _8>>{});                // Val layout  1x1
    TiledCopy copyB = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, TB>{},
        Layout<Shape<_32, _4>, Stride<_4, _1>>{}, // Thr layout 32x8 k-major
        Layout<Shape< _1, _8>>{});                // Val layout  1x1
    TiledCopy copyC = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, TC>{},
        Layout<Shape<_4, _32>>{},                 // Thr layout 32x8 k-major
        Layout<Shape<_8, _1>>{});                // Val layout  1x1

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
    TiledCopy mmaCopyC = make_tiled_copy_C(Copy_Atom<Copy_Traits<UniversalCopy<TC>>, TC>{}, mmaC);

    // batched copy for C
    constexpr int kSmemLayoutCBatch = 2;
    auto SmemLayoutAtomC = composition(
        Swizzle<2, 3, 3>{},
        make_layout(make_shape(Int<kMmaPM>{}, Int<kMmaPN>{})));
    auto sC = tile_to_shape(SmemLayoutAtomC,
        make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}, Int<kSmemLayoutCBatch>{}));       // (m,n) -> smem_idx

#if 0
    print(copyA);
    print(copyB);
    print(copyC);
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

    static constexpr int shm_size_AB =
        cute::cosize(sA) * sizeof(TA) + cute::cosize(sB) * sizeof(TB);
    static constexpr int shm_size_C = cute::cosize(sC) * sizeof(TC);

    static constexpr int shm_size =
        cute::max(shm_size_AB, shm_size_C);

    static bool checked = true;
    if (!checked) {
        cudaFuncSetAttribute(gemm_device_multistage_slicek<
            decltype(prob_shape), decltype(cta_tiler),
            TA, decltype(dA), decltype(sA), decltype(copyA),
            TB, decltype(dB), decltype(sB), decltype(copyB),
            TC, decltype(dC), decltype(sC), decltype(copyC),
            decltype(mmaC), decltype(mmaCopyA), decltype(mmaCopyB), decltype(mmaCopyC)>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
        checked = true;
    }
    gemm_device_multistage_slicek <<<dimGrid, dimBlock, 0, stream>>> (
        prob_shape, cta_tiler,
        A, dA, sA, copyA,
        B, dB, sB, copyB,
        C, dC, sC, copyC,
        mmaC, mmaCopyA, mmaCopyB, mmaCopyC);
}

template <
    class TA, class TB, class TC>
void
gemm(
    char transA, char transB, int m, int n, int k,
    TA const* A, int ldA,
    TB const* B, int ldB,
    TC* C, int ldC,
    bool multistage = false,
    cudaStream_t stream = 0) {
    if (multistage) {
        return gemm_tn_multistage(m, n, k, A, ldA, B, ldB, C, ldC, stream);
    } else {
        return gemm_tn_twostage(m, n, k, A, ldA, B, ldB, C, ldC, stream);
    }
}


template <typename T>
void gen_rand_data(T* data, int n) {
    for (int i = 0; i < n; ++i) {
        float v = (rand() % 200 - 100) * 0.01;
        data[i] = v;
    }
}

int main(int argc, char** argv) {
    using namespace cute;

    srand(10086);

    int m = 4096;
    if (argc >= 2)
        sscanf(argv[1], "%d", &m);

    int n = 338;
    if (argc >= 3)
        sscanf(argv[2], "%d", &n);

    int k = 4096;
    if (argc >= 4)
        sscanf(argv[3], "%d", &k);

    double flops = (2.0 * m * n * k) * 1e-9;

    char transA = 'T';
    char transB = 'N';

    using TA = cute::half_t;
    using TB = cute::half_t;
    using TC = cute::half_t;

    TC alpha(1.0f);
    TC beta(0.0f);

    std::cout << "M = " << m << std::endl;
    std::cout << "N = " << n << std::endl;
    std::cout << "K = " << k << std::endl;
    std::cout << "C = A^" << transA << " B^" << transB << std::endl;

    thrust::host_vector<TA> h_A(m * k);
    thrust::host_vector<TB> h_B(n * k);
    thrust::host_vector<TC> h_C(m * n, TC(0));
    thrust::host_vector<TC> h_C1 = h_C;

    //for (int j = 0; j < m * k; ++j) h_A[j] = (j / 100.0f);
    //for (int j = 0; j < n * k; ++j) h_B[j] = ((j + 1) / 100.0f);
    for (int j = 0; j < m * n; ++j) h_C[j] = ((j + 1) / 1000.0f);
    gen_rand_data(h_A.data(), h_A.size());
    gen_rand_data(h_B.data(), h_B.size());
    //gen_rand_data(h_C.data(), h_C.size());

    thrust::device_vector<TA> d_A = h_A;
    thrust::device_vector<TB> d_B = h_B;
    thrust::device_vector<TC> d_C = h_C;
    thrust::device_vector<TC> d_C1 = h_C1;

    const int warmup = 100;
    const int timing_iterations = 1000;

    int ldA = k, ldB = k, ldC = m;

    // Run once
    for (int i = 0; i < warmup; ++i) {
        gemm(transA, transB, m, n, k,
            d_A.data().get(), ldA,
            d_B.data().get(), ldB,
            d_C.data().get(), ldC,
            true, 0);
    }
    CUTE_CHECK_LAST();
    h_C = d_C;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Timing iterations
    float cute_time = 0;
    for (int i = 0; i < timing_iterations; ++i) {
        float elapsed = 0;
        cudaEventRecord(start);

        gemm(transA, transB, m, n, k,
            d_A.data().get(), ldA,
            d_B.data().get(), ldB,
            d_C.data().get(), ldC,
            true, 0);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        cute_time += elapsed;
    }
    cute_time /= timing_iterations;
    CUTE_CHECK_LAST();
    printf("CUTE_GEMM:     [%6.1f]TFlop/s  (%6.4f)ms\n", flops / cute_time, cute_time);

    cublasHandle_t handle;
    cublasCreate(&handle);

    for (int i = 0; i < warmup; ++i) {
        cublasStatus_t ret = cublasGemmEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k,
            &alpha, d_A.data().get(), CUDA_R_16F, ldA,
            d_B.data().get(), CUDA_R_16F, ldB, &beta,
            d_C1.data().get(), CUDA_R_16F, ldC,
            CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);

        if (ret != CUBLAS_STATUS_SUCCESS) {
            printf("cublas err = %d, str = %s\n", ret, cublasGetStatusString(ret));
        }
    }
    h_C1 = d_C1;

    // Timing iterations
    float blas_time = 0;
    for (int i = 0; i < timing_iterations; ++i) {
        float elapsed = 0;
        cudaEventRecord(start);

        cublasGemmEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k,
            &alpha, d_A.data().get(), CUDA_R_16F, ldA,
            d_B.data().get(), CUDA_R_16F, ldB, &beta,
            d_C1.data().get(), CUDA_R_16F, ldC,
            CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        blas_time += elapsed;
    }
    blas_time /= timing_iterations;
    printf("BLAS_GEMM:     [%6.1f]TFlop/s  (%6.4f)ms\n", flops / blas_time, blas_time);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);

    auto tC_host = make_tensor(h_C.data(), make_layout(make_shape(m, n)));
    auto tC_host_blas =
        make_tensor(h_C1.data(), make_layout(make_shape(m, n)));

    auto tile = make_tile(min(8, m), min(8, n));
    auto t32x32 = local_tile(tC_host, tile, make_coord(0, 0));
    auto t32x32_blas = local_tile(tC_host_blas, tile, make_coord(0, 0));

    printf("our-impl:\n");
    print_tensor(t32x32);

    printf("blas:\n");
    print_tensor(t32x32_blas);

    int error_count = 0;
    float threshold = 0.01;
    for (int i = 0; i < h_C.size(); ++i) {
        float v1 = h_C[i];
        float v2 = h_C1[i];
        if (std::abs(v1 - v2) > threshold) {
            printf("mma: %f, blas: %f, idx: %i\n", v1, v2, i);
            error_count++;
            break;
        }
    }

    printf("wrong result count: %i\n", error_count);

    return 0;
}
