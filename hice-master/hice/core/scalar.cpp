#include "hice/core/scalar.h"

namespace hice {

Scalar Scalar::operator-() const {
  if (is_int8()) {
    return Scalar(-v.i8);
  } else if (is_int16()) {
    return Scalar(-v.i16);
  } else if (is_int32()) {
    return Scalar(-v.i32);
  } else if (is_int64()) {
    return Scalar(-v.i64);
  } else if (is_float()) {
    return Scalar(-v.f);
  } else if (is_double()) {
    return Scalar(-v.d);
  } else if (is_bool()) {
    return Scalar(-v.b);
  } else if (is_complex_float()) {
    return Scalar(std::complex<float>(-v.zf[0], -v.zf[1]));
  } else {
    HICE_CHECK(is_complex_double()) << "Unsupported type of Scalar in operator -.";
    return Scalar(std::complex<double>(-v.zd[0], -v.zd[1]));
  }
}

}  // namespace hice
