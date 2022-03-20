// This file is based on c10\core\ScalarType.h from PyTorch. 
// PyTorch is BSD-style licensed, as found in its LICENSE file.
// From https://github.com/pytorch/pytorch.
// And it's slightly modified for HICE's usage. 

#pragma once

#include <cstdint>
#include <iostream>
#include <complex>

#include "hice/util/loguru.h"
#include "hice/util/type_id.h"

namespace hice {

#define HICE_FORALL_SCALAR_TYPES(_) \
  _(uint8_t, UInt8, i8)              \
  _(int8_t, Int8, i8)                \
  _(uint16_t, UInt16, i16)            \
  _(int16_t, Int16, i16)              \
  _(uint32_t, UInt32, i32)            \
  _(int32_t, Int32, i32)              \
  _(uint64_t, UInt64, i64)            \
  _(int64_t, Int64, i64)              \
  _(float, Float, f)                \
  _(double, Double, d)              \
  _(bool, Bool, b)

#define HICE_FORALL_SCALAR_TYPES_WITH_COMPLEX(_) \
  _(uint8_t, UInt8, i8)              \
  _(int8_t, Int8, i8)                \
  _(uint16_t, UInt16, i16)            \
  _(int16_t, Int16, i16)              \
  _(uint32_t, UInt32, i32)            \
  _(int32_t, Int32, i32)              \
  _(uint64_t, UInt64, i64)            \
  _(int64_t, Int64, i64)              \
  _(float, Float, f)                \
  _(double, Double, d)              \
  _(bool, Bool, b)              \
  _(std::complex<float>, ComplexFloat, zf)        \
  _(std::complex<double>, ComplexDouble, zd)

enum class ScalarType : int8_t {
#define DEFINE_ENUM(_1,n,_2) n,
  HICE_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_ENUM)
#undef DEFINE_ENUM
  Undefined,
  NumScalarTypes
};

#define DEFINE_CONSTANT(_,name,_2) \
constexpr ScalarType k##name = ScalarType::name;

HICE_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_CONSTANT)
#undef DEFINE_CONSTANT

constexpr int kNumScalarTypes = static_cast<int>(ScalarType::NumScalarTypes);  

inline DataType ScalarTypeToDataType(ScalarType scalar_type) {
#define DEFINE_CASE(ctype,name,_) \
  case ScalarType::name : return DataType::make<ctype>();

  switch(scalar_type) {
    HICE_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_CASE)
    case ScalarType::Undefined: return DataType();
    default: HICE_LOG(ERROR) << "Unrecognized ScalarType in HICE: " << scalar_type;
  }
#undef DEFINE_CASE
  return DataType(); // Used to surpress the warning of missing return
}

inline ScalarType DataTypeToScalarType(const DataType &dtype) {
#define DEFINE_IF(ctype, name, _)            \
  if (dtype == DataType::make<ctype>()) {    \
    return ScalarType::name;                 \
  }

  HICE_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_IF)
#undef DEFINE_IF

  if (dtype == DataType()) {
    return ScalarType::Undefined;
  }
  HICE_LOG(ERROR) << "Unsupported DataType in HICE: " << dtype; 
  return ScalarType::Undefined; // Used to surpress the warning of missing return 
}

// Convert c++ types to ScalarType enum members
template <typename T>
struct CTypeToScalarType {};

#define DEFINE_CTYPE_TO_SCALARTYPE(ct, st, _2)                    \
template <>                                                       \
struct CTypeToScalarType<ct> {                                    \
  static constexpr ScalarType value = ScalarType::st;             \
};

HICE_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_CTYPE_TO_SCALARTYPE)
#undef DEFINE_CTYPE_TO_SCALARTYPE

// Convert ScalarType enum members to c++ types
template <ScalarType sc_type>
struct ScalarTypeToCType {};

#define DEFINE_SCALARTYPE_TO_CTYPE(ct, st, _2)                     \
template <>                                                        \
struct ScalarTypeToCType<ScalarType::st> {                         \
  using type = ct;                                                 \
  static ct t;                                                     \
};

HICE_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_SCALARTYPE_TO_CTYPE)
#undef DEFINE_SCALARTYPE_TO_CTYPE

inline bool operator==(ScalarType s, DataType d) {
  return DataTypeToScalarType(d) == s;
}

inline bool operator==(DataType d, ScalarType s) {
  return s == d;
}

inline const char * to_string(ScalarType t) {
#define DEFINE_CASE(_,name,_2) \
  case ScalarType::name : return #name;

  switch(t) {
    HICE_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_CASE)
    default:
      return "Unknown ScalarType";
  }
#undef DEFINE_CASE
}

inline size_t ScalarTypeSize(ScalarType t) {
#define CASE_ELEMENTSIZE_CASE(ctype,name,_2) \
  case ScalarType::name : return sizeof(ctype);

  switch(t) {
    HICE_FORALL_SCALAR_TYPES_WITH_COMPLEX(CASE_ELEMENTSIZE_CASE)
    default:
      HICE_LOG(ERROR) << "Unknown ScalarType";
      return 0; // Used to surpress the warning of missing return 
  }
#undef CASE_ELEMENTSIZE_CASE
}

inline bool is_integral(ScalarType t) {
  return (t == ScalarType::UInt8 ||
          t == ScalarType::Int8 ||
          t == ScalarType::UInt16 ||
          t == ScalarType::Int16 ||
          t == ScalarType::UInt32 ||
          t == ScalarType::Int32 ||
          t == ScalarType::UInt64 ||
          t == ScalarType::Int64);
}

inline bool is_floating_point(ScalarType t) {
  return (t == ScalarType::Double ||
          t == ScalarType::Float);
}

inline bool is_complex(ScalarType t) {
  return (t == ScalarType::ComplexFloat ||
          t == ScalarType::ComplexDouble);
}

inline std::ostream& operator<<(std::ostream& stream, ScalarType scalar_type) {
  return stream << to_string(scalar_type);
}

} // namespace hice 
