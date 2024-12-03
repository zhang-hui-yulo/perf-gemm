#pragma once

#include "../config.hpp"

namespace neo {

template <auto v>
struct C {
  using type = C<v>;
  static constexpr auto value = v;
  using value_type = decltype(v);
  NEO_HOST_DEVICE constexpr operator   value_type() const noexcept { return value; }
  NEO_HOST_DEVICE constexpr value_type operator()() const noexcept { return value; }
};

// Deprecate
template <class T, T v>
using constant = C<v>;

template <bool b>
using bool_constant = C<b>;

using true_type = bool_constant<true>;
using false_type = bool_constant<false>;

// A more std:: conforming integral_constant that enforces type but interops with C<v>
template <class T, T v>
struct integral_constant : C<v> {
    using type = integral_constant<T, v>;
    static constexpr T value = v;
    using value_type = T;
    // Disambiguate C<v>::operator value_type()
    //CUTE_HOST_DEVICE constexpr operator   value_type() const noexcept { return value; }
    NEO_HOST_DEVICE constexpr value_type operator()() const noexcept { return value; }
};

template <int v>
using Int = C<v>;

template <auto t1, auto t2>
NEO_HOST_DEVICE constexpr C<t1 * t2> operator *(C<t1>, C<t2>) {
    return {};
}

template <auto t1, auto t2>
NEO_HOST_DEVICE constexpr C<t1 / t2> operator /(C<t1>, C<t2>) {
    return {};
}

template <auto t1, auto t2>
NEO_HOST_DEVICE constexpr C<t1 + t2> operator +(C<t1>, C<t2>) {
    return {};
}

template <auto t1, auto t2>
NEO_HOST_DEVICE constexpr C<t1 - t2> operator -(C<t1>, C<t2>) {
    return {};
}

template <auto t1, auto t2>
NEO_HOST_DEVICE constexpr C<t1 % t2> operator %(C<t1>, C<t2>) {
    return {};
}

}