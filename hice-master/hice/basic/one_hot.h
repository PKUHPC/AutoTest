#pragma once

#include "hice/core/tensor.h"
#include "hice/core/dispatch.h"


namespace hice {

using one_hot_kernel_fn_type = void (*)(const Tensor &input,
                                        int64_t num_classes, int64_t axis,
                                        Tensor &output);
HICE_DECLARE_DISPATCHER(one_hot_kernel_dispatcher, one_hot_kernel_fn_type);

HICE_API Tensor one_hot(const Tensor &input, int64_t num_classes, int64_t axis);

#if 0
HICE_API Tensor &one_hot(const Tensor &input, int64_t num_classes, int64_t axis,
                         Tensor &output);
#endif

} // namespace hice