#pragma once

#include "flash.hpp"
#include "kernel_traits.hpp"
#include "utils.hpp"

#define M_LOG2E    1.44269504088896340736

namespace flash {

template <typename Fragment>
inline __device__ auto convert_type_f32_to_f16(Fragment const &acc_fp32) {
    Tensor acc_fp16 = make_tensor<cute::half_t>(shape(acc_fp32));
    {
        Tensor acc_fp32x2 = recast< float2>(acc_fp32);
        Tensor acc_fp16x2 = recast<__half2>(acc_fp16);
        for (int i = 0; i < size(acc_fp32x2); ++i) { acc_fp16x2(i) = __float22half2_rn(acc_fp32x2(i)); }
    }
    return acc_fp16;
}

template<typename MMA_traits, typename Layout>
inline __device__ auto convert_layout_rowcol_Aregs(Layout rowcol_layout) {
    using X = Underscore;
    static_assert(decltype(size<0, 0>(rowcol_layout))::value == 2);
    static_assert(decltype(size<1, 0>(rowcol_layout))::value == 2);
    constexpr int mma_shape_K = get<2>(typename MMA_traits::Shape_MNK{});
    static_assert(mma_shape_K == 8 || mma_shape_K == 16);
    constexpr int MMA_N_divisor = mma_shape_K == 8 ? 1 : 2;
    auto l = logical_divide(rowcol_layout, Shape<X, Shape<X, Int<MMA_N_divisor>>>{});  // ((2, MMA_M), (2, (2, MMA_N / 2)))
    // TD [2023-08-13]: Same error as above on Cutlass 3.2
    // return make_layout(make_layout(get<1, 0>(l), get<0, 0>(l), get<1, 1, 0>(l)),
    //                    get<0, 1>(l),
    //                    get<1, 1, 1>(l));
    return make_layout(make_layout(get<0>(get<1>(l)), get<0>(get<0>(l)), get<0>(get<1>(get<1>(l)))),
        get<1>(get<0>(l)),
        get<1>(get<1>(get<1>(l))));
};

template <int kBlockM, int kBlockN, int kNWarps,typename Engine, typename Layout>
inline __device__ void mask_within_nblock(Tensor<Engine, Layout> &tensor, const int m_block, const int nbi) {
    // tensor has shape (nrow=(2, MMA_M), ncol=(2, MMA_N))
    static_assert(Layout::rank == 2, "Only support 2D Tensor");
    const int lane_id = threadIdx.x % 32;
    const int col_idx_offset = kBlockN * nbi + (lane_id % 4) * 2;

    const int nrow_group = threadIdx.x / 32;
    const int row_idx_offset = kBlockM * m_block + lane_id / 4 + nrow_group * 16 /* 2*8 */;
    // (2, nrow), 2*8 for each
    const int group_stride = kNWarps * 16;

    #pragma unroll
    for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
        const int col_idx_base = col_idx_offset + nj * 8;
        #pragma unroll
        for (int j = 0; j < size<1, 0>(tensor); ++j) {
            const int col_idx = col_idx_base + j;

            // Without the "make_coord" we get wrong results
            // for nrow(2, MMA_M)
            #pragma unroll
            for (int mi = 0; mi < size<0, 0>(tensor); ++mi) {

                #pragma unroll
                for (int mj = 0; mj < size<0, 1>(tensor); ++mj) {
                    const int row_idx = row_idx_offset + mi * 8 + mj * group_stride;
                    if (col_idx > row_idx) {
                      tensor(make_coord(mi, mj), make_coord(j, nj)) = -INFINITY;
                    }
                }
            }

        }
    }
}

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

template<typename Layout>
inline __device__ auto convert_layout_acc_rowcol(Layout acc_layout) {
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = logical_divide(acc_layout, Shape<_2>{});  // ((2, 2), MMA_M, MMA_N)
    // TD [2023-08-13]: Idk why but get<0, 1>(l) doesn't work for Cutlass 3.2, I'm getting
    // "int_tuple.hpp(74): error: conversion to inaccessible base class"
    // return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<2>(l)));
    return make_layout(make_layout(get<1>(get<0>(l)), get<1>(l)), make_layout(get<0>(get<0>(l)), get<2>(l)));
}

// Apply the exp to all the elements.
template <bool Scale_max = true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
inline __device__ void scale_apply_exp2(Tensor<Engine0, Layout0>& tensor, Tensor<Engine1, Layout1> const& max, const float scale) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        // If max is -inf, then all elements must have been -inf (possibly due to masking).
        // We don't want (-inf - (-inf)) since that would give NaN.
        // If we don't have float around M_LOG2E the multiplication is done in fp64.
        const float max_scaled = max(mi) == -INFINITY ? 0.f : max(mi) * (Scale_max ? scale : float(M_LOG2E));
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni) {
            // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
            // max * log_2(e)) This allows the compiler to use the ffma
            // instruction instead of fadd and fmul separately.
            tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
        }
    }
}

template<bool Is_first, typename Tensor0, typename Tensor1, typename Tensor2>
inline __device__ void softmax_rescale_o(Tensor0 &scores, Tensor1 &scores_max, Tensor1 &scores_sum,
                                         Tensor2 &acc_o, float softmax_scale_log2) {
    if (Is_first) {
        reduce_max</*zero_init=*/true>(scores, scores_max);
        flash::scale_apply_exp2(scores, scores_max, softmax_scale_log2);
        reduce_sum(scores, scores_sum);
    } else {
        Tensor scores_max_prev = make_fragment_like(scores_max);
        cute::copy(scores_max, scores_max_prev);
        reduce_max</*zero_init=*/false>(scores, scores_max);
        Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
        #pragma unroll
        for (int mi = 0; mi < size(scores_max); ++mi) {
            float scores_max_cur = scores_max(mi);
            float scores_scale = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
            scores_sum(mi) *= scores_scale;
            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scores_scale; }
        }
        flash::scale_apply_exp2(scores, scores_max, softmax_scale_log2);

        Tensor scores_sum_cur = make_fragment_like(scores_sum);

        reduce_sum(scores, scores_sum_cur);
        #pragma unroll
        for (int mi = 0; mi < size(scores_sum); ++mi) { scores_sum(mi) += scores_sum_cur(mi); }
    }
}

template<typename Tensor0, typename Tensor1,
         typename Tensor2, typename Tensor3, typename Tensor4,
         typename TiledMma, typename TiledCopyA, typename TiledCopyB,
         typename ThrCopyA, typename ThrCopyB>
inline __device__ void gemm_smem(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsA,
                            Tensor4 const& tCsB, TiledMma tiled_mma,
                            TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
                            ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // M
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
    // NOTE: s -> reg
    cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{}));
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1));
            cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

template<typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
    typename TiledMma, typename TiledCopy, typename ThrCopy>
inline __device__ void gemm_A_in_regs(Tensor0& acc, Tensor1& tCrA, Tensor2& tCrB, Tensor3 const& tCsB,
    TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
    ThrCopy smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
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

        flash::gemm_smem(rAccScore, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );

        Tensor scores = make_tensor(rAccScore.data(), flash::convert_layout_acc_rowcol(rAccScore.layout())); // (MMA_M, MMA_N)

        // NOTE: 2. mask within N BLOCKs
        if (Is_causal == true && nbi * kBlockN >= seqlen_q_start) {
            flash::mask_within_nblock<kBlockM, kBlockN, kNWarps>(scores, m_block, nbi);
        }

        // wait gV -> sV
        flash::cp_async_wait<0>();
        __syncthreads();

        // advance K
        if (nbi != n_block_max - 1) {
            gK = local_tile(K, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(nbi + 1, _));
            tKgK = gmem_thr_copy_QKV.partition_S(gK(_, _, 0));
            flash::copy(gmem_tiled_copy_QKV, tKgK, tKsK);
            cute::cp_async_fence();
        }

        // softmax
        nbi == 0 ? flash::softmax_rescale_o</*Is_first=*/true>(scores, scores_max, scores_sum, rAccOut, params.softmax_scale) :
            flash::softmax_rescale_o</*Is_first=*/false>(scores, scores_max, scores_sum, rAccOut, params.softmax_scale);

        Tensor rP = flash::convert_type_f32_to_f16(rAccScore);
        Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_rowcol_Aregs<TiledMMA>(scores.layout()));

        flash::gemm_A_in_regs(rAccOut, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
    }

    // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2, MMA_K))
    Tensor acc_o_rowcol = make_tensor(rAccOut.data(), flash::convert_layout_acc_rowcol(rAccOut.layout()));

    // for row
    #pragma unroll
    for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
        float sum = scores_sum(mi);
        float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
        float scale = inv_sum;
        // for col
        #pragma unroll
        for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
            acc_o_rowcol(mi, ni) *= scale;
        }
    }

    // Convert acc_o from fp32 to fp16/bf16
    Tensor rO = flash::convert_type_f32_to_f16(rAccOut);
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});

    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)
}

}
