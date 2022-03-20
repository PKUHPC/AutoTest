#pragma once

#include "hice/core/dispatch.h"
#include "hice/core/scalar.h"
#include "hice/core/tensor.h"

namespace hice {

// Dispatcher
using dropout_fwd_kernel_fn_type = void (*)(Tensor &input, double rate,
                                            Tensor &mask, Tensor &output);
HICE_DECLARE_DISPATCHER(dropout_fwd_dispatcher, dropout_fwd_kernel_fn_type);

using dropout_bwd_kernel_fn_type = void (*)(Tensor &input, double rate,
                                            Tensor &mask, Tensor &output);
HICE_DECLARE_DISPATCHER(dropout_bwd_dispatcher, dropout_bwd_kernel_fn_type);
// Operators
HICE_API Tensor dropout_fwd(Tensor &input, double rate, Tensor &mask);
HICE_API Tensor& dropout_fwd(Tensor &input, double rate, Tensor &mask, Tensor& output);
HICE_API Tensor dropout_bwd(Tensor &input, double rate, Tensor &mask);

} // namespace hice
