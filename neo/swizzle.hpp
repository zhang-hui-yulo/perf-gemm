#pragma once

#include "config.hpp"

namespace neo {

template <int B, int M, int S = B>
struct Swizzle {
    static constexpr int b = B;
    static constexpr int m = M;
    static constexpr int s = S;

    template <class Offset>
    NEO_HOST_DEVICE constexpr static
    auto apply(Offset const& offset) {
        return offset ^ (((offset >> (m + s)) & ((1 << b) - 1)) << m);
    }

    template <class Offset>
    NEO_HOST_DEVICE constexpr
    auto
    operator()(Offset const& offset) const {
        return apply(offset);
    }
};
}
