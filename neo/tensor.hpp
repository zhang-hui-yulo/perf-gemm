#pragma once

#include "layout.hpp"

namespace neo {

template <typename T1, typename T2, typename T3>
class Tensor {
public:
    NEO_HOST_DEVICE Tensor(T1 ptr_base, T1 ptr, T2 shape, T3 stride)
    :m_ptr_base(ptr_base)
    ,m_ptr(ptr)
    ,m_shape(shape)
    ,m_stride(stride) {}

    NEO_HOST_DEVICE ~Tensor() {}

    NEO_HOST_DEVICE constexpr const auto& base() const {
        return m_ptr_base;
    }

    NEO_HOST_DEVICE constexpr auto& data() const {
        return m_ptr;
    }

    NEO_HOST_DEVICE constexpr const auto& shape() const {
        return m_shape;
    }

    NEO_HOST_DEVICE constexpr const auto& stride() const {
        return m_stride;
    }

    template <typename T>
    NEO_HOST_DEVICE constexpr auto& move(T &&coord) {
        m_ptr += dot(coord, stride());
        return *this;
    }

    template <typename T>
    NEO_HOST_DEVICE constexpr auto move_at(T&& coord) const {
        return m_ptr + dot(coord, stride());
    }

    template <typename T>
    NEO_HOST_DEVICE constexpr auto& jump(T&& coord) {
        auto outer_stride = inner_product(shape(), stride());
        auto offset = dot(coord, outer_stride);
        m_ptr = m_ptr_base + offset;
        return *this;
    }

    template <typename T>
    NEO_HOST_DEVICE constexpr auto jump_at(T&& coord) const {
        auto outer_stride = inner_product(shape(), stride());
        auto offset = dot(coord, outer_stride);
        return m_ptr_base + offset;
    }

private:
    const T1 m_ptr_base;
    T1 m_ptr;
    T2 m_shape;
    T3 m_stride;
};

template <typename T1, typename T2, typename T3>
NEO_HOST_DEVICE constexpr Tensor<T1, T2, T3> make_tensor(T1 ptr_base, T2 shape, T3 stride) {
    return { ptr_base, ptr_base, shape, stride };
}

template <typename T1, typename T2, typename T3>
NEO_HOST_DEVICE constexpr Tensor<T1, T2, T3> make_tensor(T1 ptr_base, T1 ptr, T2 shape, T3 stride) {
    return { ptr_base, ptr, shape, stride };
}

template <typename T1, typename T2>
NEO_HOST_DEVICE constexpr auto slice_tile(T1&& tensor, T2&& shape) {
    return make_tensor(tensor.data(), tensor.data(), inner_div(tensor.shape(), shape), inner_product(shape, tensor.stride()));
}

template <typename T1, typename T2, typename T3>
NEO_HOST_DEVICE constexpr auto slice_tile(T1 &&tensor, T2 &&shape, T3 &&coord) {
    auto outer_stride = inner_product(shape, tensor.stride());
    auto offset = dot(coord, outer_stride);
    return make_tensor(tensor.data(), tensor.data() + offset, inner_div(tensor.shape(), shape), outer_stride);
}

template <typename T1, typename T2>
NEO_HOST_DEVICE constexpr auto local_tile(T1&& tensor, T2&& shape) {
    return make_tensor(tensor.data(), tensor.data(), shape, tensor.stride());
}

template <typename T1, typename T2, typename T3>
NEO_HOST_DEVICE constexpr auto local_tile(T1 &&tensor, T2 &&shape, T3 &&coord) {
    auto ret = make_tensor(tensor.data(), shape, tensor.stride());
    ret.jump(coord);
    return ret;
}

}