#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cudnn.h>

#include <cute/tensor.hpp>


template <
    class TY, class TX, class TW, class CtaTiler,
    class GEMMYGLayout, class GEMMXGLayout, class GEMMWGLayout,
    class YSLayout, class XSLayout, class WSLayout,
    class TiledCopyY, class TiledCopyX, class TiledCopyW,
    class TiledMmaCopyY, class TiledMmaCopyX, class TiledMmaCopyW,
    class TiledMma, class Im2ColPredictor
>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void cute_implicit_gemm_multistage(TY* y_ptr, const TX* x_ptr, const TW* w_ptr, CtaTiler cta_tiler,
    GEMMYGLayout GEMM_y_glayout, GEMMXGLayout GEMM_x_glayout, GEMMWGLayout GEMM_w_glayout,
    YSLayout y_slayout, XSLayout x_slayout, WSLayout w_slayout,
    TiledCopyY copy_y, TiledCopyX copy_x, TiledCopyW copy_w,
    TiledMmaCopyY copy_mma_y, TiledMmaCopyX copy_mma_x, TiledMmaCopyW copy_mma_w, TiledMma mma,
    int GEMM_M, int GEMM_N, int GEMM_K,
    Im2ColPredictor im2col_predictor) {
    using namespace cute;

    CUTE_STATIC_ASSERT_V(size(copy_x) == size(mma));
    CUTE_STATIC_ASSERT_V(size(copy_w) == size(mma));

    CUTE_STATIC_ASSERT_V(rank(GEMM_y_glayout) == Int<2>{}); // (N * P * Q, K)         <-> (GEMM_M, GEMM_N)
    CUTE_STATIC_ASSERT_V(rank(GEMM_x_glayout) == Int<2>{}); // (N * P * Q, R * S * C) <-> (GEMM_M, GEMM_K)
    CUTE_STATIC_ASSERT_V(rank(GEMM_w_glayout) == Int<2>{}); // (K        , R * S * C) <-> (GEMM_N, GEMM_K)
    CUTE_STATIC_ASSERT_V(rank(x_slayout) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(w_slayout) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<2>(x_slayout) == size<2>(w_slayout));

    CUTE_STATIC_ASSERT_V(size<0>(XSLayout{}) == size<0>(cta_tiler));  // BLK_M
    CUTE_STATIC_ASSERT_V(size<0>(WSLayout{}) == size<1>(cta_tiler));  // BLK_N
    CUTE_STATIC_ASSERT_V(size<1>(XSLayout{}) == size<2>(cta_tiler));  // BLK_K
    CUTE_STATIC_ASSERT_V(size<1>(WSLayout{}) == size<2>(cta_tiler));  // BLK_K

    Tensor GEMM_mX = make_tensor(make_gmem_ptr(x_ptr), GEMM_x_glayout);  // ((N, P, Q), (R, S, C))
    Tensor GEMM_mW = make_tensor(make_gmem_ptr(w_ptr), GEMM_w_glayout);  // (K        , (R, S, C))
    Tensor GEMM_mY = make_tensor(make_gmem_ptr(y_ptr), GEMM_y_glayout);  // ((N, P, Q), K)

    // Get the appropriate blocks for this thread block
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
    Tensor GEMM_gX = local_tile(GEMM_mX, cta_tiler, cta_coord, Step<_1,  X, _1>{});  // (BLK_M,BLK_K,k)
    Tensor GEMM_gW = local_tile(GEMM_mW, cta_tiler, cta_coord, Step< X, _1, _1>{});  // (BLK_N,BLK_K,k)
    Tensor GEMM_gY = local_tile(GEMM_mY, cta_tiler, cta_coord, Step<_1, _1,  X>{});  // (BLK_M,BLK_N)

    // global memory -> shared memory
    // Shared memory buffers
    __shared__ TX smemX[cosize_v<XSLayout>];
    __shared__ TW smemW[cosize_v<WSLayout>];
    Tensor sX = make_tensor(make_smem_ptr(smemX), x_slayout);            // (BLK_M,BLK_K,PIPE)
    Tensor sW = make_tensor(make_smem_ptr(smemW), w_slayout);            // (BLK_N,BLK_K,PIPE)

    ThrCopy thr_copy_x = copy_x.get_slice(threadIdx.x);
    Tensor GEMM_tAgX = thr_copy_x.partition_S(GEMM_gX);                  // (CPY,CPY_M,CPY_K,k)
    Tensor tAsX = thr_copy_x.partition_D(sX);                            // (CPY,CPY_M,CPY_K,PIPE)

    ThrCopy thr_copy_w = copy_w.get_slice(threadIdx.x);
    Tensor GEMM_tBgW = thr_copy_w.partition_S(GEMM_gW);                  // (CPY,CPY_M,CPY_K,k)
    Tensor tBsW = thr_copy_w.partition_D(sW);                            // (CPY,CPY_N,CPY_K,PIPE)
    
    clear(tAsX);
    clear(tBsW);

    // gmem -> im2col -> smem
    int istage = 0;
    int itile_to_read = 0;
    int ismem_read = 0;
    int ismem_write = 0;
    auto kstage = max(size<3>(tAsX), _2{});

    Tensor GEMM_mcX = make_identity_tensor(shape(GEMM_mX));
    Tensor GEMM_cX = local_tile(GEMM_mcX, cta_tiler, cta_coord, Step<_1, X, _1>{});
    Tensor GEMM_tAcX = thr_copy_x.partition_S(GEMM_cX);                  // (CPY,CPY_M,CPY_K,k)
    auto tApX = [&](auto... coords) {
        Tensor GEMM_tAcX_ = GEMM_tAcX(_0{}, _, _, istage);               // (CPY_M,CPY_K)
        auto gemm_mk = GEMM_tAcX_(coords...);                            // (gemm_m, gemm_k)
        return im2col_predictor.pred_x(gemm_mk, shape(GEMM_mX));
    };

    Tensor GEMM_mcW = make_identity_tensor(shape(GEMM_mW));
    Tensor GEMM_cW = local_tile(GEMM_mcW, cta_tiler, cta_coord, Step< X, _1, _1>{});
    Tensor GEMM_tBcW = thr_copy_w.partition_S(GEMM_cW);                  // (CPY,CPY_M,CPY_K,k)

    auto tBpW = [&](auto... coords) {
        Tensor GEMM_tBcW_ = GEMM_tBcW(_0{}, _, _, istage);               // (CPY_M,CPY_K)
        auto gemm_nk = GEMM_tBcW_(coords...);                            // (gemm_n, gemm_k)
        return im2col_predictor.pred_w(gemm_nk, shape(GEMM_mW));
    };

    CUTE_UNROLL
    for (; istage < kstage - _1{}; ++istage) {
        copy_if(copy_x, tApX, GEMM_tAgX(_, _, _, istage), tAsX(_, _, _, istage));
        copy_if(copy_w, tBpW, GEMM_tBgW(_, _, _, istage), tBsW(_, _, _, istage));

        cp_async_fence();

        ++itile_to_read;
        ++ismem_write;
    }

    //
    // Define A/B partitioning and C accumulators
    //

    ThrMMA thr_mma = mma.get_slice(threadIdx.x);
    Tensor tCrX = thr_mma.partition_fragment_A(GEMM_gX(_, _, 0));          // (MMA,MMA_M,MMA_K)
    Tensor tCrW = thr_mma.partition_fragment_B(GEMM_gW(_, _, 0));          // (MMA, MMA_N, MMA_K)
    Tensor tCrY = thr_mma.partition_fragment_C(GEMM_gY);                   // (MMA, MMA_M, MMA_N)

    ThrCopy thr_mma_copy_x = copy_mma_x.get_slice(threadIdx.x);
    Tensor tCsX = thr_mma_copy_x.partition_S(sX);             // (CPY, CPY_M, CPY_K, PIPE)
    Tensor tCrX_view = thr_mma_copy_x.retile_D(tCrX);         // (CPY, CPY_M, CPY_K)

    ThrCopy thr_mma_copy_w = copy_mma_w.get_slice(threadIdx.x);
    Tensor tCsW = thr_mma_copy_w.partition_S(sW);             // (CPY, CPY_M, CPY_K, PIPE)
    Tensor tCrW_view = thr_mma_copy_w.retile_D(tCrW);         // (CPY, CPY_M, CPY_K)

    // fill zero for accumulator
    clear(tCrY);

#if 0
    if (thread(0)) {
        print("  mX : "); print(mX); print("\n");
        print("  gX : "); print(gX); print("\n");
        print("  sX : "); print(sX); print("\n");
        print("  GEMM_mX : "); print(GEMM_mX); print("\n");
        print("  GEMM_gX : "); print(GEMM_gX); print("\n");

        print("  gX : "); print(gX(_, _, 0)); print("\n");
        print("  rank(gX) : "); print(rank(gX(_, _, 0))); print("\n");
        //print_latex(layout(gX(_,_,0)));
    }
#endif

#if 1
    // Size of the register pipeline
    auto nk = size<2>(tCrX);
    
    // PREFETCH register pipeline
    if (nk > 1) {
        // Wait until our first prefetched tile is loaded in
        cp_async_wait<kstage - 2>();
        __syncthreads();

        // Prefetch the first rmem from the first k-tile
        copy(copy_mma_x, tCsX(_, _, _0{}, ismem_read), tCrX_view(_, _, _0{}));
        copy(copy_mma_x, tCsW(_, _, _0{}, ismem_read), tCrW_view(_, _, _0{}));
    }
    
    // loop over k: i. load tile, ii. mma
    auto ntile = size<3>(GEMM_tAgX);

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
            copy(copy_mma_x, tCsX(_, _, ik_next, ismem_read), tCrX_view(_, _, ik_next));
            copy(copy_mma_x, tCsW(_, _, ik_next, ismem_read), tCrW_view(_, _, ik_next));

            if (ik == 0) {
                if (itile_to_read < ntile) {
                    copy_if(copy_x, tApX, GEMM_tAgX(_, _, _, itile_to_read), tAsX(_, _, _, ismem_write));
                    copy_if(copy_w, tBpW, GEMM_tBgW(_, _, _, itile_to_read), tBsW(_, _, _, ismem_write));

                    ++itile_to_read;
                    ismem_write = (ismem_write + 1) % kstage;
                }

                cp_async_fence();
            }

            gemm(mma, tCrY, tCrX(_, _, ik), tCrW(_, _, ik), tCrY);
        }  // for ik
    }      // itile

#if 0
    if (thread(0)) {
        print_tensor(tCrY); printf("\n");
    }
#endif

    // Epilogue

    // Shared memory buffers
    TY* smemY = (TY*)smemX;
    Tensor sY = make_tensor(make_smem_ptr(smemY), y_slayout); // (BLK_M,BLK_N)

    // Define mma copy
    ThrCopy thr_mma_copy_y = copy_mma_y.get_slice(threadIdx.x);
    Tensor tCrY_view = thr_mma_copy_y.retile_S(tCrY);         // (CPY, CPY_M, CPY_N)
    Tensor tCsY_mma = thr_mma_copy_y.partition_D(sY);         // (CPY, _1, _1, pipe)
    Tensor tCrY_viewx = group_modes<1, 3>(tCrY_view);         // (CPY, CPY_MN)

    ThrCopy thr_copy_y = copy_y.get_thread_slice(threadIdx.x);
    Tensor tCsY = thr_copy_y.partition_S(sY);                 // (CPY, _1, _1, pipe)
    Tensor tCgY_copy = thr_copy_y.partition_D(GEMM_gY);       // (CPY, CPY_M, CPY_N)
    Tensor tCgY_copy_x = group_modes<1, 3>(tCgY_copy);        // (CPY_, CPY_MN)

    CUTE_STATIC_ASSERT_V(size<1>(tCrY_viewx) == size<1>(tCgY_copy_x));        // CPY_MN
    CUTE_STATIC_ASSERT_V(size<3>(tCsY_mma) == size<3>(tCsY));                 // pipe
    CUTE_STATIC_ASSERT_V((size<1>(tCrY_viewx) % size<3>(tCsY_mma)) == _0{});  // CPY_MN % pipe == 0
    CUTE_STATIC_ASSERT_V((size<1>(tCgY_copy_x) % size<3>(tCsY_mma)) == _0{}); // CPY_MN % pipe == 0

    Tensor GEMM_mcY = make_identity_tensor(shape(GEMM_mY));
    Tensor GEMM_cY = local_tile(GEMM_mcY, cta_tiler, cta_coord, Step<_1, _1, X>{});
    Tensor GEMM_tCcY = thr_mma_copy_y.partition_D(GEMM_cY);   // (CPY, CPY_M, CPY_N)
    Tensor GEMM_tCcY_x = group_modes<1, 3>(GEMM_tCcY);     // (CPY_, CPY_MN)

    auto step = size<3>(tCsY_mma);  // pipe

#if 0
    if (thread0()) {
        print("   tCsY_mma : "); print(tCsY_mma); printf("\n");
        print(" tCrY_viewx : "); print(tCrY_viewx); printf("\n");
        print("       tCsY : "); print(tCsY); printf("\n");
        print("    GEMM_gY : "); print(GEMM_gY); printf("\n");
        print("  tCgY_copy : "); print(tCgY_copy); printf("\n");
        print("tCgY_copy_x : "); print(tCgY_copy_x); printf("\n");
        print("     copy_y : "); print(copy_y); printf("\n");
        print("GEMM_tCcY_x : "); print(GEMM_tCcY_x); printf("\n");
    }
#endif

    CUTE_UNROLL
    for (int i = 0; i < size<1>(tCrY_viewx); i += step) {
        // reg -> shm
        CUTE_UNROLL
        for (int j = 0; j < step; ++j) {
            // we add a temp tensor to cope with accumulator and output data type
            // difference
            Tensor t = make_tensor_like<TY>(tCrY_viewx(_, i + j));
            copy(tCrY_viewx(_, i + j), t);

            copy(copy_mma_y, t, tCsY_mma(_, 0, 0, j));
        }
        __syncthreads();

        // shm -> global
        CUTE_UNROLL
        for (int j = 0; j < step; ++j) {
            auto tCpY = [&](auto... coords) {
                auto pred = GEMM_tCcY_x(_, i + j);
                return elem_less(pred(_0{}, coords...), shape(GEMM_mY));
            };

            copy_if(copy_y, tCpY, tCsY(_, 0, 0, j), tCgY_copy_x(_, i + j));
        }

        __syncthreads();

    }

#endif
}

namespace cute {

struct Redirected_Rst {
    static constexpr char no_action = -1;
    static constexpr char zfill = 0;
    static constexpr char copy = 1;

    const void* src = nullptr;
    const char action = no_action;
};

template<
    class LayoutNPQ, class LayoutRSC,
    class TensorNHWC, class TensorKRSC,
    class T
>
struct Im2Col_NPQ_RSC_NHWC {
    LayoutNPQ layout_npq;

    LayoutRSC layout_rsc;

    TensorNHWC tensor_nhwc;

    TensorKRSC tensor_krsc;

    T stride_h, stride_w;
    T dila_h, dila_w;
    T pad_h, pad_w;

    /*
        int n = gemm_m / (P * Q);
        int npq_residual = gemm_m % (P * Q);
        int p = npq_residual / Q;
        int q = npq_residual % Q;

        int k = gemm_n;

        int r = gemm_k / (S * C);
        int crs_residual = gemm_k % (S * C);
        int s = crs_residual / C;
        int c = crs_residual % C;
        int h = p * stride_h + r * dila_h - pad_h;
        int w = q * stride_w + s * dila_w - pad_w;

        ElementA a = tensor_X.at({n, h, w, c});
        ElementB b = tensor_W.at({k, r, s, c});
    */
    template <class ShapeGemmMK, class ShapeMK>
    CUTE_HOST_DEVICE constexpr
    auto pred_x(const ShapeGemmMK& gemm_mk, const ShapeMK& global_mk) const {
        if (elem_less(gemm_mk, global_mk)) {
            auto x_npq = layout_npq.get_hier_coord(get<0>(gemm_mk));
            auto x_rsc = layout_rsc.get_hier_coord(get<1>(gemm_mk));
            int h = get<1>(x_npq) * stride_h + get<0>(x_rsc) * dila_h - pad_h;
            int w = get<2>(x_npq) * stride_w + get<1>(x_rsc) * dila_w - pad_w;
            if (h >= 0 && h < get<1>(tensor_nhwc.shape()) && w >= 0 && w < get<2>(tensor_nhwc.shape())) {
                return Redirected_Rst{
                    reinterpret_cast<const void*>(&tensor_nhwc(get<0>(x_npq), h, w, get<2>(x_rsc))),
                    Redirected_Rst::copy
                }; 
            } else {
                return Redirected_Rst{ 0, Redirected_Rst::zfill };
            }
        } else {
            return Redirected_Rst{ 0, Redirected_Rst::no_action };
        }
    }

    template <class ShapeGemmNK, class ShapeNK>
    CUTE_HOST_DEVICE constexpr
    auto pred_w(const ShapeGemmNK& gemm_nk, const ShapeNK& global_nk) const {
        if (elem_less(gemm_nk, global_nk)) {
            auto w_rsc = layout_rsc.get_hier_coord(get<1>(gemm_nk));
            return Redirected_Rst{
                reinterpret_cast<const void*>(&tensor_krsc(get<0>(gemm_nk), w_rsc)),
                Redirected_Rst::copy
            };
        } else {
            return Redirected_Rst{ 0, Redirected_Rst::no_action };
        }
    }
};

template<class LayoutNPQ, class LayoutRSC, class TensorNHWC, class TensorKRSC, class T>
CUTE_HOST_DEVICE constexpr
static auto make_npq_rsc_nhwc_pred(const LayoutNPQ&& npq, const LayoutRSC& rsc,
    const TensorNHWC& nhwc, const TensorKRSC& krsc,
    T stride_h, T stride_w, T dila_h, T dila_w, T pad_h, T pad_w) {
    return Im2Col_NPQ_RSC_NHWC<LayoutNPQ, LayoutRSC, TensorNHWC, TensorKRSC, T> {
        npq, rsc, nhwc, krsc,
        stride_h, stride_w,
        dila_h, dila_w,
        pad_h, pad_w
    };
}

template<class LayoutNPQ, class LayoutRSC, class TensorNHWC, class TensorKRSC, class T>
CUTE_HOST_DEVICE constexpr
void print(const Im2Col_NPQ_RSC_NHWC<LayoutNPQ, LayoutRSC, TensorNHWC, TensorKRSC, T>& pred) {
    print(pred.layout_npq); printf("\n");
    print(pred.layout_rsc); printf("\n");
    print(pred.tensor_nhwc); printf("\n");
    print(pred.tensor_krsc); printf("\n");
    print(pred.stride_h); printf("\n");
    print(pred.stride_w); printf("\n");
    print(pred.dila_h); printf("\n");
    print(pred.dila_w); printf("\n");
    print(pred.pad_h); printf("\n");
    print(pred.pad_w); printf("\n");
}

template <class TS, class TD = TS>
struct SM80_CP_ASYNC_CACHEGLOBAL_ZFILL_REDIRECT
     : SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<TS, TD> {};

template <class S, class D>
struct Copy_Traits<SM80_CP_ASYNC_CACHEGLOBAL_ZFILL_REDIRECT<S, D>> {
    // Logical thread id to thread idx (one-thread)
    using ThrID = Layout<_1>;

    // Map from (src-thr,src-val) to bit
    using SrcLayout = Layout<Shape<_1, Int<sizeof_bits<S>::value>>>;
    // Map from (dst-thr,dst-val) to bit
    using DstLayout = Layout<Shape<_1, Int<sizeof_bits<D>::value>>>;

    // Reference map from (thr,val) to bit
    using RefLayout = SrcLayout;

    // Predicate value: true = load, false = zfill
    Redirected_Rst rst;

    // Construct a zfill variant with a given predicate value
    CUTE_HOST_DEVICE constexpr
    Copy_Traits<SM80_CP_ASYNC_CACHEGLOBAL_ZFILL_REDIRECT<S, D>>
    with(bool pred) const {
        return { };
    }

    // Construct a zfill variant with a given predicate value
    CUTE_HOST_DEVICE constexpr
    Copy_Traits<SM80_CP_ASYNC_CACHEGLOBAL_ZFILL_REDIRECT<S, D>>
    with(Redirected_Rst rst) const {
        return { rst };
    }

    // Overload copy_unpack for zfill variant to pass the predicate into the op
    template <class TS, class SLayout,
              class TD, class DLayout>
    CUTE_HOST_DEVICE friend constexpr
    void
    copy_unpack(Copy_Traits         const& traits,
                Tensor<TS, SLayout> const& src,
                Tensor<TD, DLayout>&       dst) {
        static_assert(is_gmem<TS>::value, "Expected gmem source for cp.async.");
        static_assert(is_smem<TD>::value, "Expected smem destination for cp.async.");

        Tensor rS = recast<S>(src);
        Tensor rD = recast<D>(dst);

        CUTE_STATIC_ASSERT_V(size(rS) == Int<1>{},
            "In CopyAtom, src layout doesn't vectorize into registers. This src layout is incompatible with this tiled copy.");
        CUTE_STATIC_ASSERT_V(size(rD) == Int<1>{},
            "In CopyAtom, dst layout doesn't vectorize into registers. This dst layout is incompatible with this tiled copy.");

        if (traits.rst.action != traits.rst.no_action) {
            auto src = reinterpret_cast<const S*>(traits.rst.src);
            SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<S, D>::copy(*src, rD[0], traits.rst.action);
        }
    }
};

}

template <class TY, class TX, class TW>
void cute_implicit_gemm(TY* y, const TX* x, const TW* w, int N, int H, int W, int C,
    int K, int R, int S, int pad_h, int pad_w, int stride_h, int stride_w, int dilation_h, int dilation_w) {
    using namespace cute;

    int P = (H + 2 * pad_h - dilation_h * (R - 1) - 1) / stride_h + 1;
    int Q = (W + 2 * pad_w - dilation_w * (S - 1) - 1) / stride_w + 1;

    int GEMM_M = N * P * Q;
    int GEMM_N = K;
    int GEMM_K = R * S * C;

    auto GEMM_y_glayout = make_layout(make_shape(GEMM_M, GEMM_N), GenRowMajor{});
    auto GEMM_x_glayout = make_layout(make_shape(GEMM_M, GEMM_K), GenRowMajor{});
    auto GEMM_w_glayout = make_layout(make_shape(GEMM_N, GEMM_K), GenRowMajor{});

    auto bM = Int<128>{};
    auto bN = Int<16>{};
    auto bK = Int<32>{};
    auto bP = Int<1>{};

    auto cta_tiler = make_shape(bM, bN, bK);

    // Define the smem layouts (static)
    // Swizzles for LDSM and 128b k-major loads
    auto swizzle_atom_x = composition(Swizzle<3, 3, 3>{},
                                      make_layout(make_shape(Int<8>{}, Int<bK>{}),
                                                             GenRowMajor{}));
    auto swizzle_atom_w = composition(Swizzle<3, 3, 3>{},
                                      make_layout(make_shape(Int<8>{}, Int<bK>{}),
                                                             GenRowMajor{}));
    auto x_slayout = tile_to_shape(swizzle_atom_x, make_shape(bM, bK, bP));
    auto w_slayout = tile_to_shape(swizzle_atom_w, make_shape(bN, bK, bP));

    auto copy_x = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL_ZFILL_REDIRECT<uint128_t>, TX>{},
        Layout<Shape<_16, _4>, Stride<_4, _1>>{},  // Thr layout 16x8 k-major
        Layout<Shape< _1, _8>>{});                 // Val layout  1x8 k-major
    auto copy_w = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL_ZFILL_REDIRECT<uint128_t>, TW>{},
        Layout<Shape<_16, _4>, Stride<_4, _1>>{},  // Thr layout 16x8 k-major
        Layout<Shape< _1, _8>>{});                 // Val layout  1x8 n-major
    auto copy_y = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, TY>{},
        Layout<Shape<_32, _2>, Stride<_2, _1>>{},  // Thr layout 16x8 k-major
        Layout<Shape< _1, _8>>{});                 // Val layout  1x8 n-major

    // im2col projector
    auto im2col_x_layout = make_layout(make_shape(N, H, W, C), GenRowMajor{});
    auto im2col_w_layout = make_layout(make_shape(K, make_shape(R, S, C)),
        make_stride(C * R * S, make_stride(C * S, C, _1{})));
    auto im2col_predictor = make_npq_rsc_nhwc_pred(
        make_layout(make_shape(N, P, Q), GenRowMajor{}),
        make_layout(make_shape(R, S, C), GenRowMajor{}),
        make_tensor(make_gmem_ptr(x), im2col_x_layout),
        make_tensor(make_gmem_ptr(w), im2col_w_layout),
        stride_h, stride_w,
        dilation_h, dilation_w,
        pad_h, pad_w);

    // Define the mma
    using mma_traits = MMA_Traits<SM80_16x8x16_F16F16F16F16_TN>;
    using mma_atom = MMA_Atom<mma_traits>;

    static constexpr int kMmaEURepeatM = 2;
    static constexpr int kMmaEURepeatN = 1;
    static constexpr int kMmaEURepeatK = 1;

    using mma_atom_shape = mma_traits::Shape_MNK;
    static constexpr int kMmaPM = 1 * kMmaEURepeatM * get<0>(mma_atom_shape{});
    static constexpr int kMmaPN = 2 * kMmaEURepeatN * get<1>(mma_atom_shape{});
    static constexpr int kMmaPK = 1 * kMmaEURepeatK * get<2>(mma_atom_shape{});

    using MMA_EU_RepeatT = decltype(make_layout(make_shape(
        Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
    using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;

    auto mma = make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{});  // 16x16x1 TiledMMA

    // mma copy
    auto mma_copy_x = make_tiled_copy_A(Copy_Atom<Copy_Traits<SM75_U32x4_LDSM_N>, TX>{}, mma);
    auto mma_copy_w = make_tiled_copy_B(Copy_Atom<Copy_Traits<SM75_U32x4_LDSM_N>, TW>{}, mma);
    auto mma_copy_y = make_tiled_copy_C(Copy_Atom<Copy_Traits<UniversalCopy<uint16_t>>, TY>{}, mma);

    // batched copy for Y
    constexpr int kSmemLayoutCBatch = 1;
    auto SmemLayoutAtomY = composition(
        Swizzle<2, 3, 3>{}, make_layout(make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}),
                                        GenRowMajor{}));
    auto y_slayout = tile_to_shape(SmemLayoutAtomY,
        make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}, Int<kSmemLayoutCBatch>{}));       // (m,n) -> smem_idx


#if 0
    printf("y_glayout: "); print(y_glayout); printf("\n");
    printf("x_glayout: "); print(x_glayout); printf("\n");
    printf("w_glayout: "); print(w_glayout); printf("\n");
    printf("x_slayout: "); print(x_slayout); printf("\n");
    printf("w_slayout: "); print(w_slayout); printf("\n");
#endif

    dim3 dimBlock(size(mma));
    dim3 dimGrid(size(ceil_div(GEMM_M, bM)),
                 size(ceil_div(GEMM_N, bN)));

    cute_implicit_gemm_multistage<<<dimGrid, dimBlock>>>
                                (y, x, w,  
                                 cta_tiler, 
                                 GEMM_y_glayout, GEMM_x_glayout, GEMM_w_glayout,
                                 y_slayout, x_slayout, w_slayout,
                                 copy_y, copy_x, copy_w,
                                 mma_copy_y, mma_copy_x, mma_copy_w, mma,
                                 GEMM_M, GEMM_N, GEMM_K,
                                 im2col_predictor);

}

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
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

    using TX = half_t;
    using TW = half_t;
    using TB = half_t;
    using TY = half_t;

    constexpr int N = 1;
    constexpr int H = 3840;
    constexpr int W = 2160;
    constexpr int C = 8;

    constexpr int K = 16;
    constexpr int R = 2;
    constexpr int S = 2;

    constexpr int pad_h = 0;
    constexpr int pad_w = 0;
    constexpr int stride_h = 1;
    constexpr int stride_w = 1;
    constexpr int dilation_h = 1;
    constexpr int dilation_w = 1;

    int P = (H + 2 * pad_h - dilation_h * (R - 1) - 1) / stride_h + 1;
    int Q = (W + 2 * pad_w - dilation_w * (S - 1) - 1) / stride_w + 1;

    thrust::host_vector<TX> h_x(N * H * W * C);
    thrust::host_vector<TW> h_w(K * R * S * C);
    thrust::host_vector<TY> h_y(N * P * Q * K);
    thrust::host_vector<TY> h_y1 = h_y;

    gen_rand_data(h_x.data(), h_x.size());
    gen_rand_data(h_w.data(), h_w.size());

    thrust::device_vector<TX> d_x = h_x;
    thrust::device_vector<TW> d_w = h_w;
    thrust::device_vector<TW> d_y = h_y;
    thrust::device_vector<TW> d_y1 = h_y1;

    cute_implicit_gemm(d_y.data().get(), d_x.data().get(), d_w.data().get(),
        N, H, W, C, K, R, S, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w);

    h_y = d_y;

    // cudnn

    // create cudnn handle
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    // create input tensor descriptor
    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
        /*format=*/CUDNN_TENSOR_NHWC, // todo why this format
        /*dataType=*/CUDNN_DATA_HALF,
        /*batch_size=*/N,
        /*channels=*/C,
        /*image_height=*/H,
        /*image_width=*/W));

    // create filter descriptor
    cudnnFilterDescriptor_t kernel_descriptor;
    checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
        /*dataType=*/CUDNN_DATA_HALF,
        /*format=*/CUDNN_TENSOR_NHWC,
        /*out_channels=*/K,
        /*in_channels=*/C,
        /*kernel_height=*/R,
        /*kernel_width=*/S));

    // create conv descriptor
    cudnnConvolutionDescriptor_t convolution_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
        /*pad_height=*/pad_h,
        /*pad_width=*/pad_w,
        /*vertical_stride=*/stride_h,
        /*horizontal_stride=*/stride_w,
        /*dilation_height=*/dilation_h,
        /*dilation_width=*/dilation_w,
        /*mode=*/CUDNN_CROSS_CORRELATION, //  todo  how to compute
        /*computeType=*/CUDNN_DATA_HALF));
    checkCUDNN(cudnnSetConvolutionMathType(convolution_descriptor, CUDNN_TENSOR_OP_MATH));

    int batch_size{ 0 }, channels{ 0 }, height{ 0 }, width{ 0 };
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convolution_descriptor,
        input_descriptor,
        kernel_descriptor,
        &batch_size,
        &channels,
        &height,
        &width));

    // create output descriptor
    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
        /*format=*/CUDNN_TENSOR_NHWC,
        /*dataType=*/CUDNN_DATA_HALF,
        /*batch_size=*/batch_size,
        /*channels=*/channels,
        /*image_height=*/height,
        /*image_width=*/width));

    cudnnConvolutionFwdAlgo_t fwd_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

    std::size_t workspace_size = 0;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn,
        input_descriptor,
        kernel_descriptor,
        convolution_descriptor,
        output_descriptor,
        fwd_algo,
        &workspace_size
    ));

    thrust::device_vector<char> d_workspace(workspace_size);

    float alpha(1.f);
    float beta(0.f);

    checkCUDNN(cudnnConvolutionForward(
        cudnn,
        &alpha,
        input_descriptor,
        d_x.data().get(),
        kernel_descriptor,
        d_w.data().get(),
        convolution_descriptor,
        fwd_algo,
        d_workspace.data().get(),
        workspace_size,
        &beta,
        output_descriptor,
        d_y1.data().get()
    ));
    h_y1 = d_y1;


    float threshold = 0.1;
    for (int i = 0; i < h_y.size(); ++i) {
        float v1 = h_y[i];
        float v2 = h_y1[i];
        if (fabs(v2 - v1) > threshold) {
            printf("v1 = %f, v2 = %f, idx = %i\n", v1, v2, i);
            break;
        }
    }

    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyFilterDescriptor(kernel_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);

    cudnnDestroy(cudnn);


    return 0;
}