#pragma once

#include <cmath>
#include "hice/util/traits.h"

namespace hice {

namespace arith_ops {

template<typename scalar_t>
inline scalar_t abs(scalar_t a) {
  // bool is_uint = std::is_same<scalar_t, uint8_t>::value || 
  //                 std::is_same<scalar_t, uint16_t>::value ||
  //                 std::is_same<scalar_t, uint32_t>::value || 
  //                 std::is_same<scalar_t, uint64_t>::value;
  // return std::abs<int64_t>(a, b) : std::abs(a, b);
  return std::abs(a);
}

inline int8_t abs(uint8_t a) {
  return std::abs<int8_t>(a);
}

inline int16_t abs(uint16_t a) {
  return std::abs<int16_t>(a);
}

inline int32_t abs(uint32_t a) {
  return std::abs<int32_t>(a);
}

inline int64_t abs(uint64_t a) {
  return std::abs<int64_t>(a);
}

} // namespace arith_utils

} // namespace hice