// This file is based on c10\core\Scalar.h from PyTorch. 
// PyTorch is BSD-style licensed, as found in its LICENSE file.
// From https://github.com/pytorch/pytorch.
// And it's modified for HICE's usage. 

// FIXME: For some reasons, unsigned types are storaged into signed variables,
// eg. uint32_t will be keeped by a int32_t variable inside. So it may cause
// problems about overflow when use unsigned type like uint8_t, uint16_t, 
// uint32_t and uint64_t.

#pragma once

#include <assert.h>
#include <stdint.h>
#include <stdexcept>
#include <string>

#include "hice/core/macros.h"
#include "hice/util/traits.h"
#include "hice/util/type_converter.h"
#include "hice/core/scalar_type.h"

// This codes are extracted from pytorch and slightly adjusted for hice's usage. 

namespace hice {

/**
 * Scalar represents a 0-dimensional tensor which contains a single element.
 * Unlike a tensor, numeric literals (in C++) are implicitly convertible to Scalar
 * (which is why, for example, we provide both add(Tensor) and add(Scalar) overloads
 * for many operations). It may also be used in circumstances where you statically
 * know a tensor is 0-dim and single size, but don't know it's type.
 */

class HICE_API Scalar {
 public:
  Scalar() : Scalar(int64_t(0)) {}

#define DEFINE_IMPLICIT_CTOR(type,name,member) \
  Scalar(type vv) \
  : tag(Tag::HAS_##member) { \
    v.member = convert<decltype(v.member),type>(vv); \
  }
  // We can't set v in the initializer list using the
  // syntax v{ .member = ... } because it doesn't work on MSVC

  HICE_FORALL_SCALAR_TYPES(DEFINE_IMPLICIT_CTOR)

#undef DEFINE_IMPLICIT_CTOR

  // Value* is both implicitly convertible to SymbolicVariable and bool which
  // causes ambiguity error. Specialized constructor for bool resolves this
  // problem.
  template <
      typename T,
      typename std::enable_if<std::is_same<T, bool>::value, bool>::type* =
          nullptr>
  Scalar(T vv) : tag(Tag::HAS_b) {
    v.b = convert<decltype(v.b), bool>(vv);
  }

#define DEFINE_IMPLICIT_COMPLEX_CTOR(type, name, member) \
  Scalar(type vv) : tag(Tag::HAS_##member) {             \
    v.member[0] = convert<double>(vv.real());       \
    v.member[1] = convert<double>(vv.imag());       \
  }

  DEFINE_IMPLICIT_COMPLEX_CTOR(std::complex<float>,ComplexFloat,zf)
  DEFINE_IMPLICIT_COMPLEX_CTOR(std::complex<double>,ComplexDouble,zd)

#undef DEFINE_IMPLICIT_COMPLEX_CTOR

#define DEFINE_ACCESSOR(type,name,member) \
  type to##name () const { \
    if (Tag::HAS_ui8 == tag) { \
      return checked_convert<type, int8_t>(v.i8, #type); \
    } else if (Tag::HAS_i8 == tag) { \
      return checked_convert<type, int8_t>(v.i8, #type); \
    } else if (Tag::HAS_ui16 == tag) { \
      return checked_convert<type, int16_t>(v.i16, #type); \
    } else if (Tag::HAS_i16 == tag) { \
      return checked_convert<type, int16_t>(v.i16, #type); \
    } else if (Tag::HAS_ui32 == tag) { \
      return checked_convert<type, int32_t>(v.i32, #type); \
    } else if (Tag::HAS_i32 == tag) { \
      return checked_convert<type, int32_t>(v.i32, #type); \
    } else if (Tag::HAS_ui64 == tag) { \
      return checked_convert<type, int64_t>(v.i64, #type); \
    } else if (Tag::HAS_i64 == tag) { \
      return checked_convert<type, int64_t>(v.i64, #type); \
    } else if (Tag::HAS_f == tag) { \
      return checked_convert<type, float>(v.f, #type); \
    } else if (Tag::HAS_d == tag) { \
      return checked_convert<type, double>(v.d, #type); \
    } else if (Tag::HAS_b == tag) { \
      return checked_convert<type, bool>(v.b, #type); \
    } else if (Tag::HAS_zf == tag) { \
      return checked_convert<type, std::complex<float>>({v.zf[0], v.zf[1]}, #type); \
    } else { \
      return checked_convert<type, std::complex<double>>({v.zd[0], v.zd[1]}, #type); \
    } \
  }

  // TODO: Support ComplexHalf accessor
  HICE_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_ACCESSOR)

  //also support scalar.to<int64_t>();
  template<typename T>
  T to();

#undef DEFINE_ACCESSOR
  // integer
  bool is_uint8() const {
    return Tag::HAS_ui8 == tag;
  }
  bool is_int8() const {
    return Tag::HAS_i8 == tag;
  }
  bool is_uint16() const {
    return Tag::HAS_ui16 == tag;
  }
  bool is_int16() const {
    return Tag::HAS_i16 == tag;
  }
  bool is_uint32() const {
    return Tag::HAS_ui32 == tag;
  }
  bool is_int32() const {
    return Tag::HAS_i32 == tag;
  }
  bool is_uint64() const {
    return Tag::HAS_ui64 == tag;
  }
  bool is_int64() const {
    return Tag::HAS_i64 == tag;
  }
  bool is_float() const {
    return Tag::HAS_f == tag;
  }
  bool is_double() const {
    return Tag::HAS_d == tag;
  }
  bool is_bool() const {
    return Tag::HAS_b == tag;
  }
  bool is_complex_float() const {
    return Tag::HAS_zf == tag;
  }
  bool is_complex_double() const {
    return Tag::HAS_zd == tag;
  }

  Scalar operator-() const;
private:
  enum class Tag { HAS_ui8, HAS_i8, HAS_ui16, HAS_i16, 
                   HAS_ui32, HAS_i32, HAS_ui64, HAS_i64,
                   HAS_f, HAS_d, HAS_b, HAS_zf, HAS_zd };
  Tag tag;
  union {
    int8_t i8;
    int16_t i16;
    int32_t i32;
    int64_t i64;
    float f;
    double d;
    bool b;
    // Can't do put std::complex in the union, because it triggers
    // an nvcc bug:
    //    error: designator may not specify a non-POD subobject
    float zf[2];
    double zd[2];
  } v;
};

// define the scalar.to<int64_t>() specializations
template<typename T>
inline T Scalar::to() {
  throw std::runtime_error("to() cast to unexpected type.");
}

#define DEFINE_TO(T,name,_) \
template<> \
inline T Scalar::to<T>() { \
  return to##name(); \
}
HICE_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_TO)
#undef DEFINE_TO

} // namesapce hice