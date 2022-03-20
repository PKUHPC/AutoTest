#pragma once

#include <vector>
#include "hice/util/types.h"
#include "hice/util/loguru.h"
//#include "hice/util/array_ref.h"
//#include "hice/util/small_vector.h"
#include <iostream>

namespace hice {

// Return product of all dimensions starting from k

inline int64_t size_of_dims(ConstIntArrayRef dims) {
  int64_t r = 1;
  for (int64_t i = 0; i < dims.size(); ++i) {
    r *= dims[i];
  }
  return r;
}

inline int64_t size_from_dim(int64_t k, ConstIntArrayRef dims) {
  int64_t r = 1;
  for (int64_t i = k; i < dims.size(); ++i) {
    r *= dims[i];
  }
  return r;
}

// Product of all dims up to k (not including dims[k])
inline int64_t size_to_dim(int64_t k, ConstIntArrayRef dims) {
  HICE_CHECK_LE(k, dims.size());
  int64_t r = 1;
  for (int i = 0; i < k; ++i) {
    r *= dims[i];
  }
  return r;
}

// Product of all dims between k and l (not including dims[k] and dims[l])
inline int64_t size_between_dim(int64_t k, int64_t l, ConstIntArrayRef dims) {
  HICE_CHECK((unsigned)l < dims.size());
  int64_t r = 1;
  if (k < l) {
    for (int i = k + 1; i < l; ++i) {
      r *= dims[i];
    }
  } else {
    for (int i = l + 1; i < k; ++i) {
      r *= dims[i];
    }
  }
  return r;
}

// Wrap around axis_index if it is negative, s.t., -1 is the last dim
inline int64_t get_true_axis(int64_t axis, int64_t ndims) {
  HICE_CHECK_GE(axis, -ndims);
  HICE_CHECK_LT(axis, ndims);
  if (axis < 0) {
    return axis + ndims;
  }
  return axis;
}

inline void check_nonnegative_dims(ConstIntArrayRef dims) {
  for (auto x : dims) {
    HICE_CHECK(x >= 0) << "Dimension has a negative dimension " << x;
    //HICE_CHECK(x >= 0) << "Dimension has a negative dimension " << x << ": "
    //                   << dims;
  }
}

inline int compare_dims(ConstIntArrayRef lhs, ConstIntArrayRef rhs) {
  int64_t ndim = lhs.size();
  HICE_CHECK_EQ(rhs.size(), ndim);
  for (int64_t dim = 0; dim < ndim; ++dim) {
    //std::cout << "(" << lhs[dim] << ", " << rhs[dim] << std::endl;
    if (lhs[dim] < rhs[dim]) {
      return -1;
    } else if (lhs[dim] > rhs[dim]) {
      return 1;
    }
  }
  return 0;
} 

// NOTE: It compares the every pair of elements of dims_in1 and dims_in2 in
// reverse order. Errors will be complained if any pair is not matchable.
// Suppose that k and m are the [ith]-last elements of dims_in1 and
// dims_in2. There will be 3 cases: case 1: k == m, [ith]-last of dim_out
// equals to k. case 2: k == 1 or m ==1, [ith]-last of dim_out equals to
// max(k, m). case 3: k != m and k != 1 and m !=1, error. For example: if
// dims_in1 = {3, 2, 3, 1, 5}, dims_in2 = {1, 3, 4, 5}, then dims_out = {3,
// 2, 3, 4, 5}
inline std::vector<int64_t> broadcast(ConstIntArrayRef dims_in1,
                                      ConstIntArrayRef dims_in2) {
  int64_t ndim_in1 = dims_in1.size();
  int64_t ndim_in2 = dims_in2.size();
  int64_t i = 0;
  int64_t ndim_out = std::max(ndim_in1, ndim_in2);
  std::vector<int64_t> dims_out(ndim_out, 0);
  while (i < ndim_in1 && i < ndim_in2) {
    auto d_in1 = dims_in1[ndim_in1 - i - 1];
    auto d_in2 = dims_in2[ndim_in2 - i - 1];
    bool match = d_in1 == 1 || 
                 d_in2 == 1 || 
                 d_in1 == d_in2;
    HICE_CHECK(match)
        << "Dims of inputs can not be broadcasted at the " 
        << i << "th inner-most dim";
    dims_out[ndim_out - i - 1] = std::max(d_in1, d_in2);
    i++;
  }
  while (i < ndim_in1) {
    dims_out[ndim_out - i - 1] = dims_in1[ndim_in1 - i - 1];
    i++;
  }
  while (i < ndim_in2) {
    dims_out[ndim_out - i - 1] = dims_in2[ndim_in2 - i - 1];
    i++;
  }
  return dims_out;
}

} // namespace hice
