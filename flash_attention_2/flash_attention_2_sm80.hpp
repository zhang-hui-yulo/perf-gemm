#pragma once

#include "flash.hpp"
#include "kernel_traits.hpp"


namespace flash {

// Shared Storage with Aligned addresses.
template <class ElementType, class SmemLayoutQ, class SmemLayoutK, class SmemLayoutV>
struct SharedStorage {
    cute::array_aligned<ElementType, cute::cosize_v<SmemLayoutQ>> smem_q;
    cute::array_aligned<ElementType, cute::cosize_v<SmemLayoutK>> smem_k;
    cute::array_aligned<ElementType, cute::cosize_v<SmemLayoutV>> smem_v;
};

template <typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
inline __device__ void copy(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const& S,
    Tensor<Engine1, Layout1>& D) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K

    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        // if (get<0>(identity_MN(0, m, 0)) < max_MN)
        #pragma unroll
        for (int k = 0; k < size<2>(S); ++k) {
            cute::copy(tiled_copy, S(_, m, k), D(_, m, k));
        }
    }
}

template <int N>
CUTE_HOST_DEVICE
void cp_async_wait() {
#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}

template <typename Kernel_traits, bool Is_causal = false, typename Params>
__global__ void flash_attention_v2_cute_kernel(const Params params) {
    using namespace cute;

    // num_m_block: seqlen_q group
    const int m_block = blockIdx.x;

    // bs * head
    const int base_id = blockIdx.y;
    // The thread index.
    const int tidx = threadIdx.x;
    const auto bs_head_q_offset = base_id * params.q_head_stride;
    const auto bs_head_k_offset = base_id * params.k_head_stride;
    const auto bs_head_v_offset = base_id * params.v_head_stride;


    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using TiledMMA = typename Kernel_traits::TiledMma;
    using index_t = typename Kernel_traits::index_t;
    using SmemLayoutQ = typename Kernel_traits::SmemLayoutQ;
    using SmemLayoutK = typename Kernel_traits::SmemLayoutKV;
    using SmemLayoutV = typename Kernel_traits::SmemLayoutKV;
    using SmemLayoutVt = typename Kernel_traits::SmemLayoutVtransposed;
    using SmemLayoutVtNoSwizzle = typename Kernel_traits::SmemLayoutVtransposedNoSwizzle;

    constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;

    // Shared memory.
    extern __shared__ char smem_[];
    using SharedStorage = SharedStorage<Element, SmemLayoutQ, SmemLayoutK, SmemLayoutV>;
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_);

    Tensor Q = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr) + bs_head_q_offset),
        make_shape(params.seqlen_q, params.dim),
        make_stride(params.dim, Int<1>{}));
    Tensor K = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr) + bs_head_k_offset),
        make_shape(params.seqlen_k, params.dim),
        make_stride(params.dim, Int<1>{}));
    Tensor V = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr) + bs_head_v_offset),
        make_shape(params.seqlen_k, params.dim),
        make_stride(params.dim, Int<1>{}));

    Tensor gQ = local_tile(Q, make_tile(Int<kBlockM>{}, Int<kHeadDim>{}), make_coord(m_block, _)); // (kBlockM, kHeadDim, num_tile_n)
    Tensor gK = local_tile(K, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(0, _));       // (kBlockN, kHeadDim, num_tile_n)
    Tensor gV = local_tile(V, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(0, _));       // (kBlockN, kHeadDim, num_tile_n)

    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tidx);

    // Construct SMEM tensors.
    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutV{});

    Tensor sVt = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutVt{});
    Tensor sVtNoSwizzle = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutVtNoSwizzle{});

    // gmem -> smem pattern
    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ(_, _, 0));
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK(_, _, 0));
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV(_, _, 0));
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    // smem -> rmem pattern
    Tensor tSrQ = thr_mma.partition_fragment_A(sQ); // (MMA,MMA_M,MMA_K)
    Tensor tSrK = thr_mma.partition_fragment_B(sK); // (MMA,MMA_N,MMA_K)
    Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle); // (MMA, MMA_K,MMA_N)

    //
    // Copy Atom retiling
    //

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    // gmem - > smem
    flash::copy(gmem_tiled_copy_QKV, tQgQ, tQsQ);
    flash::copy(gmem_tiled_copy_QKV, tKgK, tKsK);
    cute::cp_async_fence();

    flash::cp_async_wait<0>();
    __syncthreads();

    Tensor rAccOut = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});

    const int n_block_min = 0;
    // NOTE: 1. mask between N BLOCKs if is causal mode
    int seqlen_q_start = m_block * kBlockM;
    int seqlen_q_end = (m_block + 1) * kBlockM;
    int n_block_max = Is_causal ? cute::ceil_div(seqlen_q_end, kBlockN) : cute::ceil_div(params.seqlen_k, kBlockN);

    Tensor scores_max = make_tensor<ElementAccum>(Shape<Int<2 * size<1>(rAccOut)>>{});
    Tensor scores_sum = make_fragment_like(scores_max);

    clear(rAccOut);


    for (int nbi = n_block_min; nbi < n_block_max; nbi++) {
        auto rAccScore = partition_fragment_C(tiled_mma, make_shape(Int<kBlockM>{}, Int<kBlockN>{}));

        clear(rAccScore);

        // wait gQ -> sQ and gK - > sK
        flash::cp_async_wait<0>();
        __syncthreads();

        // gV - > sV
        gV = local_tile(V, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(nbi, _));
        tVgV = gmem_thr_copy_QKV.partition_S(gV(_, _, 0));
        flash::copy(gmem_tiled_copy_QKV, tVgV, tVsV);
        cute::cp_async_fence();
    }
}

}
