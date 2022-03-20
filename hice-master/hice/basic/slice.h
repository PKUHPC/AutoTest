#pragma once

#include "hice/core/tensor.h"

namespace hice {

// NOTE: If the given axis=0, result tensor SHARE storage with the origin tensor.
// Otherwise there will be data copying.
HICE_API Tensor slice(const Tensor& self, int64_t axis, int64_t start, int64_t end);

} // namespace hice