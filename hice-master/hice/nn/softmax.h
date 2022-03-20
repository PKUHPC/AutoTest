#pragma once 

#include "hice/core/tensor.h"
#include "hice/core/dispatch.h"

namespace hice {

// Forward dispatcher
using softmax_fwd_kernel_fn_type = void (*)(const Tensor &input, int64_t axis,
                                            Tensor &output);
HICE_DECLARE_DISPATCHER(softmax_fwd_dispatcher, softmax_fwd_kernel_fn_type);
// Forward operators
HICE_API Tensor softmax_fwd(const Tensor &input, int64_t axis);
HICE_API Tensor& softmax_fwd(const Tensor &input, int64_t axis, Tensor &output);

// Backward dispatcher
using softmax_bwd_kernel_fn_type = void (*)(const Tensor &output,
                                            const Tensor &grad_output,
                                            int64_t axis, Tensor &grad_input);
HICE_DECLARE_DISPATCHER(softmax_bwd_dispatcher, softmax_bwd_kernel_fn_type);
// Backward operators
HICE_API Tensor softmax_bwd(const Tensor &output, const Tensor &grad_output,
                            int64_t axis);
HICE_API Tensor& softmax_bwd(const Tensor &output, const Tensor &grad_output,
                             int64_t axis, Tensor &grad_input);

} // namespace hice
