#pragma once

// hip passed

#include <cute/arch/mma_gfx11.hpp>
#include <cute/atom/mma_traits.hpp>
#include <cute/layout.hpp>
#include <cute/numeric/numeric_types.hpp>

namespace cute
{

namespace {

// (T32,V16) -> (M16,N16)
using GFX11_16x16_Row = Layout<Shape <Shape <_16,_2>,Shape <_16>>,
                               Stride<Stride< _1,_0>,Stride<_16>>>;

using GFX11_16x16_Col = Layout<Shape <Shape <_16,_2>,Shape <_16>>,
                               Stride<Stride<_16,_0>,Stride< _1>>>;

using GFX11_16x16_32b = Layout<Shape <Shape <_16, _2>, Shape <_8>>,
                               Stride<Stride<_16, _1>, Stride<_2>>>;

using GFX11_16x16_32bN = Layout<Shape <Shape <_32>, Shape < _8>>,
                                Stride<Stride< _1>, Stride<_32>>>;

template <typename T, bool opsel>
struct CUTE_ALIGNAS(4) GFX11FrgTypeAccum {
    T data[2];

    CUTE_HOST_DEVICE constexpr operator T() const noexcept {
        if constexpr (!opsel) {
            return data[0];
        } else {
            return data[1];
        }
    }
};

}

///////////////////////////////////////////////////////////////////////////////
//////////////////////// fp32 = fp16 * fp16 + fp32 ////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<GFX11_16x16x16_F32F16F16F32_TN>
{
    using ValTypeD = float;
    using ValTypeA = half_t;
    using ValTypeB = half_t;
    using ValTypeC = float;

    using Shape_MNK = Shape<_16, _16, _16>;
    using ThrID = Layout<_32>;
    using ALayout = GFX11_16x16_Row;
    using BLayout = GFX11_16x16_Row;
    using CLayout = GFX11_16x16_32b;
};

template <>
struct MMA_Traits<GFX11_16x16x16_F32F16F16F32_TN_N>
     : MMA_Traits<GFX11_16x16x16_F32F16F16F32_TN>
{
    using CLayout = GFX11_16x16_32bN;
};

///////////////////////////////////////////////////////////////////////////////
//////////////////////// fp32 = bf16 * bf16 + fp32 ////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<GFX11_16x16x16_F32BF16BF16F32_TN>
     : MMA_Traits<GFX11_16x16x16_F32F16F16F32_TN>
{
    using ValTypeD = float;
    using ValTypeA = bfloat16_t;
    using ValTypeB = bfloat16_t;
    using ValTypeC = float;
};


///////////////////////////////////////////////////////////////////////////////
//////////////////////// fp16 = fp16 * fp16 + fp16 ////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<GFX11_16x16x16_F16F16F16F16_TN<false>>
     : MMA_Traits<GFX11_16x16x16_F32F16F16F32_TN>
{
    using ValTypeD = half_t;
    using ValTypeA = half_t;
    using ValTypeB = half_t;
    using ValTypeC = half_t;

    using FrgTypeC = GFX11FrgTypeAccum<ValTypeC, false>;
};

template <>
struct MMA_Traits<GFX11_16x16x16_F16F16F16F16_TN_N<false>>
     : MMA_Traits<GFX11_16x16x16_F16F16F16F16_TN<false>>
{
    using CLayout = GFX11_16x16_32bN;
};

template <>
struct MMA_Traits<GFX11_16x16x16_F16F16F16F16_TN<true>>
     : MMA_Traits<GFX11_16x16x16_F16F16F16F16_TN<false>>
{
    using FrgTypeC = GFX11FrgTypeAccum<ValTypeC, true>;
};

template <>
struct MMA_Traits<GFX11_16x16x16_F16F16F16F16_TIED_TN<false>>
     : MMA_Traits<GFX11_16x16x16_F16F16F16F16_TN<false>> {};

template <>
struct MMA_Traits<GFX11_16x16x16_F16F16F16F16_TIED_TN<true>>
     : MMA_Traits<GFX11_16x16x16_F16F16F16F16_TN<true>> {};

///////////////////////////////////////////////////////////////////////////////
//////////////////////// bf16 = bf16 * bf16 + bf16 ////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<GFX11_16x16x16_BF16BF16BF16BF16_TN<false>>
     : MMA_Traits<GFX11_16x16x16_F32F16F16F32_TN>
{
    using ValTypeD = bfloat16_t;
    using ValTypeA = bfloat16_t;
    using ValTypeB = bfloat16_t;
    using ValTypeC = bfloat16_t;

    using FrgTypeC = GFX11FrgTypeAccum<ValTypeC, false>;
};

template <>
struct MMA_Traits<GFX11_16x16x16_BF16BF16BF16BF16_TN<true>>
     : MMA_Traits<GFX11_16x16x16_BF16BF16BF16BF16_TN<false>>
{
    using FrgTypeC = GFX11FrgTypeAccum<ValTypeC, true>;
};

template <>
struct MMA_Traits<GFX11_16x16x16_BF16BF16BF16BF16_TIED_TN<false>>
     : MMA_Traits<GFX11_16x16x16_BF16BF16BF16BF16_TN<false>> {};

template <>
struct MMA_Traits<GFX11_16x16x16_BF16BF16BF16BF16_TIED_TN<true>>
     : MMA_Traits<GFX11_16x16x16_BF16BF16BF16BF16_TN<true>> {};

///////////////////////////////////////////////////////////////////////////////
//////////////////////// i32 = iu8 * iu8 + iu32 ///////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template<>
struct MMA_Traits<GFX11_16x16x16_I32IU8IU8I32_TN<>>
     : MMA_Traits<GFX11_16x16x16_F32F16F16F32_TN>
{
    using ValTypeD = int32_t;
    using ValTypeA = int8_t;
    using ValTypeB = int8_t;
    using ValTypeC = int32_t;
};

template<>
struct MMA_Traits<GFX11_16x16x16_I32IU8IU8I32_TN<false, false, false>>
     : MMA_Traits<GFX11_16x16x16_F32F16F16F32_TN>
{
    using ValTypeD = uint32_t;
    using ValTypeA = uint8_t;
    using ValTypeB = uint8_t;
    using ValTypeC = uint32_t;
};

///////////////////////////////////////////////////////////////////////////////
//////////////////////// i32 = iu4 * iu4 + iu32 ///////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template<>
struct MMA_Traits<GFX11_16x16x16_I32IU4IU4I32_TN<>>
     : MMA_Traits<GFX11_16x16x16_F32F16F16F32_TN>
{
    using ValTypeD = int32_t;
    using ValTypeA = int4_t;
    using ValTypeB = int4_t;
    using ValTypeC = int32_t;
};

template<>
struct MMA_Traits<GFX11_16x16x16_I32IU4IU4I32_TN<false, false, false>>
     : MMA_Traits<GFX11_16x16x16_F32F16F16F32_TN>
{
    using ValTypeD = uint32_t;
    using ValTypeA = uint4_t;
    using ValTypeB = uint4_t;
    using ValTypeC = uint32_t;
};

}