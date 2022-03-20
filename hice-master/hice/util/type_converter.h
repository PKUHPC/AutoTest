// This file is based on TypeConverter from PyTorch. 
// PyTorch is BSD-style licensed, as found in its LICENSE file.
// From https://github.com/pytorch/pytorch.
// And it's modified for HICE's usage. 

#pragma once

#include <limits>
#include <complex>

#include "hice/util/traits.h"

namespace hice {

template <typename To, typename From, typename Enable = void>
struct Converter {
  To operator()(From f) {
    return static_cast<To>(f);
  }
};

template <typename To, typename From>
To convert(From from) {
  return Converter<To, From>()(from);
}

template <typename To, typename FromV>
struct Converter<
    To,
    std::complex<FromV>,
    typename std::enable_if<ext::negation<ext::is_complex<To>>::value>::type> {
  To operator()(std::complex<FromV> f) {
    return static_cast<To>(f.real());
  }
};

// In some versions of MSVC, there will be a compiler error when building.
// C4146: unary minus operator applied to unsigned type, result still unsigned
// It can be addressed by disabling the following warning. 
#ifdef _MSC_VER
#pragma warning( push )
#pragma warning( disable : 4146 )
#endif

// skip isnan and isinf check for integral types
template <typename To, typename From>
typename std::enable_if<std::is_integral<From>::value, bool>::type overflows(
    From f) {
  using limit = std::numeric_limits<typename ext::scalar_value_type<To>::type>;
  if (!limit::is_signed && std::numeric_limits<From>::is_signed) {
    // allow for negative numbers to wrap using two's complement arithmetic.
    // For example, with uint8, this allows for `a - b` to be treated as
    // `a + 255 * b`.
    return f > limit::max() ||
        (f < 0 && -static_cast<uint64_t>(f) > limit::max());
  } else {
    return f < limit::lowest() || f > limit::max();
  }
}

#ifdef _MSC_VER
#pragma warning( pop )
#endif

template <typename To, typename From>
typename std::enable_if<std::is_floating_point<From>::value, bool>::type
overflows(From f) {
  using limit = std::numeric_limits<typename ext::scalar_value_type<To>::type>;
  if (limit::has_infinity && std::isinf(static_cast<double>(f))) {
    return false;
  }
  if (!limit::has_quiet_NaN && (f != f)) {
    return true;
  }
  return f < limit::lowest() || f > limit::max();
}

template <typename To, typename From>
typename std::enable_if<ext::is_complex<From>::value, bool>::type overflows(
    From f) {
  // casts from complex to real are considered to overflow if the
  // imaginary component is non-zero
  if (!ext::is_complex<To>::value && f.imag() != 0) {
    return true;
  }
  // Check for overflow componentwise
  // (Technically, the imag overflow check is guaranteed to be false
  // when !ext::is_complex<To>, but any optimizer worth its salt will be
  // able to figure it out.)
  return overflows<
             typename ext::scalar_value_type<To>::type,
             typename From::value_type>(f.real()) ||
      overflows<
             typename ext::scalar_value_type<To>::type,
             typename From::value_type>(f.imag());
}

template <typename To, typename From>
To checked_convert(From f, const char* name) {
  if (overflows<To, From>(f)) {
    std::ostringstream oss;
    oss << "value cannot be converted to type " << name
        << " without overflow: " << f;
    throw std::domain_error(oss.str());
  }
  return convert<To, From>(f);
}

} // namespace hice 