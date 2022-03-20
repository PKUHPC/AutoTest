#pragma once
#include "hice/math/cpu/openmp/parallel.h"
#include "hice/util/numeric_utils.h"

namespace hice {

template <typename scalar_t, typename index_t>
void launch_compare(Tensor& res, Tensor& res_indices, const Tensor& self,
                    int64_t reduce_dim, bool greater) {
  auto data_out = res.mutable_data<scalar_t>();
  auto data_indices = res_indices.mutable_data<index_t>();
  auto data_in = self.data<scalar_t>();
  auto numel = self.size();

  int64_t n = self.dim(reduce_dim);
  int64_t stride = self.stride(reduce_dim);

  if (n == 1) {
    stride = 1;
    for (int64_t i = self.ndim() - 1; i > reduce_dim; i--) {
      stride *= self.dim(i);
    }
  }
  int64_t batch = numel / (n * stride);
  parallel_for(0, batch * stride, 1, [=](int64_t begin, int64_t end) {
    for (int64_t bi = begin; bi < end; bi++) {
      int64_t b = bi / stride;
      int64_t i = bi % stride;
      const scalar_t* data = &data_in[b * n * stride + i];
      scalar_t result = data[0];
      index_t result_index = 0;
      for (int64_t k = 0; k < n; k++) {
        scalar_t value = data[k * stride];
        bool cmp = greater ? (result > value) : (result < value);
        result = cmp ? result : value;
        result_index = cmp ? result_index : k;
        if (_isnan<scalar_t>(result)) {
          break;
        }
      }
      data_out[b * stride + i] = result;
      data_indices[b * stride + i] = result_index;
    }
  });
  // if (stride == 1) {
  //   parallel_for(0, batch, 1, [=](int64_t begin, int64_t end) {
  //     for (int64_t b = begin; b < end; b++) {
  //       const scalar_t* data = &data_in[b * n];
  //       scalar_t result = data[0];
  //       index_t result_index = 0;
  //       for (int64_t k = 0; k < n; k++) {
  //         scalar_t value = data[k];
  //         bool cmp = greater ? (result > value) : (result < value);
  //         result = cmp ? result : value;
  //         result_index = cmp ? result_index : k;
  //         if (_isnan<scalar_t>(result)) {
  //           break;
  //         }
  //       }
  //       data_out[b] = result;
  //       data_indices[b] = result_index;
  //     }
  //   });
  // } else {
  //   parallel_for(0, batch * stride, 1, [=](int64_t begin, int64_t end) {
  //     for (int64_t bi = begin; bi < end; bi++) {
  //       int64_t b = bi / stride;
  //       int64_t i = bi % stride;
  //       const scalar_t* data = &data_in[b * n * stride + i];
  //       scalar_t result = data[0];
  //       index_t result_index = 0;
  //       for (int64_t k = 0; k < n; k++) {
  //         scalar_t value = data[k * stride];
  //         bool cmp = greater ? (result > value) : (result < value);
  //         result = cmp ? result : value;
  //         result_index = cmp ? result_index : k;
  //         if (_isnan<scalar_t>(result)) {
  //           break;
  //         }
  //       }
  //       data_out[b * stride + i] = result;
  //       data_indices[b * stride + i] = result_index;
  //     }
  //   });
  // }
}
}  // namespace hice