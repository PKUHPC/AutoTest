#pragma once

#include "hice/core/tensor.h"

namespace hice {

HICE_API Tensor& resize_(Tensor& input, ConstIntArrayRef new_dims);

} // namespace hice