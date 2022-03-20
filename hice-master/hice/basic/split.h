#pragma once

#include "hice/core/tensor.h"

namespace hice {

// Split Tensor into smaller tensors along the given dimension.
// NOTE: If the given dim=0, result tensors SHARE storage with the origin
// tensor. Otherwise there will be data copying.

// The Tensor will be divided into {num_tensors} equal arrays along given
// dimension.
HICE_API std::vector<Tensor> split(const Tensor& self, int64_t axis,
                                   int64_t num_tensors);

// The given dimension of i-th result Tensor equals to sizes[i].
HICE_API std::vector<Tensor> split_with_sizes(const Tensor& self, int64_t axis,
                                              ConstIntArrayRef sizes);
}  // namespace hice