#pragma once

#include "hice/core/tensor.h"

namespace hice {

HICE_API Tensor reshape(const Tensor& input, ConstIntArrayRef new_dims);

HICE_API Tensor& reshape_(Tensor& input, ConstIntArrayRef new_dims);

HICE_API Tensor expand_dims(const Tensor& input, int64_t axis);

HICE_API Tensor& expand_dims_(Tensor& input, int64_t axis);

HICE_API Tensor squeeze(const Tensor& input, int64_t axis);

HICE_API Tensor& squeeze_(Tensor& input, int64_t axis);

HICE_API Tensor contiguous(const Tensor& input);

HICE_API Tensor& contiguous_(Tensor& input);

} // namespace hice