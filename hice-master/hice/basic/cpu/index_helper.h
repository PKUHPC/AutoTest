#pragma once

#include "hice/core/dimension.h"

namespace hice {

struct IndexHelper {
  IndexHelper(ConstIntArrayRef dims, ConstIntArrayRef strides)
      : dims_(dims.begin(), dims.end()),
        strides_(strides.begin(), strides.end()) {
    HICE_CHECK_EQ(dims_.size(), strides_.size());
  }

  int64_t multi_index_to_offset(ConstIntArrayRef multi_index) {
    HICE_CHECK_EQ(dims_.size(), multi_index.size());
    int64_t offset = 0;
    for (int64_t i = 0; i < dims_.size(); ++i) {
      offset += multi_index[i] * strides_[i];
    }
    return offset;
  }

  int64_t linear_index_to_offset(int64_t index) {
    auto ndim = dims_.size();
    int64_t offset = 0;
    int64_t rem = index;
    int64_t prod = 1;
#pragma unroll
    for (int64_t i = 0; i < ndim; i++) {
      int64_t mod_value = (rem / prod) % dims_[i];
      rem = index - mod_value * prod;
      prod = prod * dims_[i];
      offset += mod_value * strides_[i];
    }
    return offset;
  }

 private:
  std::vector<int64_t> dims_;
  std::vector<int64_t> strides_;
};

} // namespace hice