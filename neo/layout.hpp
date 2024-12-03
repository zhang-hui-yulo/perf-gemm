#pragma once

#include "config.hpp"
#include "numeric/integral_constant.hpp"

namespace neo {

template <typename T1, typename T2, typename T3>
struct Shape {
    T1 row_spacing;
    T2 col_spacing;
    T3 depth_spacing;
};

template <typename T1, typename T2, typename T3>
struct Stride {
    T1 row_spacing;
    T2 col_spacing;
    T3 depth_spacing;
};

template <typename T1, typename T2, typename T3>
struct Coord {
    T1 row_spacing;
    T2 col_spacing;
    T3 depth_spacing;
};

template <typename T1, typename T2, typename T3>
NEO_HOST_DEVICE constexpr Shape<T1, T2, T3> make_shape(T1 rows, T2 cols, T3 depth) {
    return { rows, cols, depth };
}

template <typename T1, typename T2>
NEO_HOST_DEVICE constexpr Shape<T1, T2, neo::Int<1>> make_shape(T1 rows, T2 cols) {
    return { rows, cols, neo::Int<1>{} };
}

template <typename T1, typename T2, typename T3>
NEO_HOST_DEVICE constexpr Stride<T1, T2, T3> make_stride(T1 rows, T2 cols, T3 depth) {
    return { rows, cols, depth };
}

template <typename T1, typename T2>
NEO_HOST_DEVICE constexpr Stride <T1, T2, Int<0>> make_stride(T1 rows, T2 cols) {
    return { rows, cols, Int<0>{} };
}

template <typename T1, typename T2, typename T3>
NEO_HOST_DEVICE constexpr Coord<T1, T2, T3> make_coord(T1 rows, T2 cols, T3 depth) {
    return { rows, cols, depth };
}

template <typename T1, typename T2>
NEO_HOST_DEVICE constexpr Coord<T1, T2, Int<0>> make_coord(T1 rows, T2 cols) {
    return { rows, cols, Int<0>{} };
}

template <typename T1, typename T2>
NEO_HOST_DEVICE constexpr auto inner_product(T1&& vec1, T2&& vec2) {
    return make_stride(vec1.row_spacing * vec2.row_spacing, vec1.col_spacing * vec2.col_spacing, vec1.depth_spacing * vec2.depth_spacing);
}

template <typename T1, typename T2>
NEO_HOST_DEVICE constexpr auto inner_div(T1&& vec1, T2&& vec2) {
    return make_shape(vec1.row_spacing / vec2.row_spacing, vec1.col_spacing / vec2.col_spacing, vec1.depth_spacing / vec2.depth_spacing);
}

template <typename T1, typename T2>
NEO_HOST_DEVICE constexpr auto dot(T1&& vec1, T2&& vec2) {
    return (vec1.row_spacing * vec2.row_spacing) + ((vec1.col_spacing * vec2.col_spacing) + (vec1.depth_spacing * vec2.depth_spacing));
}

template <typename T1, typename T2, typename T3>
NEO_HOST_DEVICE constexpr auto copy_partition(T1 &&shape, T2 &&id, T3 &&count) {
    return make_coord(id / shape.col_spacing, (id % shape.col_spacing) * count, Int<0>{});
}


}
