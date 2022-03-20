#pragma once

#include "hice/core/dimension.h"
#include "hice/device/cuda/common_cuda.h"

namespace hice {

struct IndexHelper {
  static constexpr int MAX_DIMS_NUM = 25;

  IndexHelper(ConstIntArrayRef dims, ConstIntArrayRef strides)
      : IndexHelper(dims.size(), dims.data(), strides.data()) {}

  IndexHelper(int ndim, const int64_t *dims, const int64_t *strides)
      : ndim_(ndim) {
    HICE_CHECK_LE(ndim, MAX_DIMS_NUM);
    for (int i = 0; i < MAX_DIMS_NUM; i++) {
      dims_[i] = (i < ndim) ? dims[i] : 1;
      strides_[i] = (i < ndim) ? strides[i] : 0;
    }
  }

  __device__ int64_t multi_index_to_offset(int64_t *multi_index) const {
    int64_t offset = 0;
#pragma unroll
    for (int64_t i = 0; i < ndim_; i++) {
      offset += multi_index[i] * strides_[i];
    }
    return offset;
  }

  __device__ int64_t linear_index_to_offset(int64_t index) const {
    int64_t offset = 0;
    int64_t rem = index;
    int64_t prod = 1;
#pragma unroll
    for (int64_t i = 0; i < ndim_; i++) {
      int64_t mod_value = (rem / prod) % dims_[i];
      rem = index - mod_value * prod;
      prod = prod * dims_[i];
      offset += mod_value * strides_[i];
    }
    return offset;
  }

  __device__ void linear_index_to_multi_index(int64_t index,
                                              int64_t *multi_index) const {
    int64_t rem = index;
    int64_t prod = 1;
#pragma unroll
    for (int64_t i = 0; i < ndim_; i++) {
      int64_t mod_value = (rem / prod) % dims_[i];
      rem = index - mod_value * prod;
      prod = prod * dims_[i];
      multi_index[i] = mod_value;
    }
  }

  int64_t ndim_;
  int64_t dims_[MAX_DIMS_NUM];
  int64_t strides_[MAX_DIMS_NUM];
};

} // namespace hice