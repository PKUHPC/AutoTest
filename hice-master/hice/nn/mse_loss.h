#pragma once

#include "hice/core/dispatch.h"
#include "hice/core/tensor.h"
#include "hice/nn/loss_reduction_type.h"

namespace hice {

// Forward dispatcher
using mse_loss_fwd_kernel_fn_type = void (*)(const Tensor &input,
                                             const Tensor &target,
                                             hice::optional<Tensor> weight,
                                             Reduction reduction, Tensor &loss);
HICE_DECLARE_DISPATCHER(mse_loss_fwd_dispatcher, mse_loss_fwd_kernel_fn_type);
// Forward operators
HICE_API Tensor mse_loss_fwd(const Tensor &input, const Tensor &target,
                             hice::optional<Tensor> weight,
                             Reduction reduction);
HICE_API Tensor &mse_loss_fwd(const Tensor &input, const Tensor &target,
                              hice::optional<Tensor> weight,
                              Reduction reduction, Tensor &loss);

// Backward dispatcher
using mse_loss_bwd_kernel_fn_type = void (*)(
    const Tensor &input, const Tensor &target, hice::optional<Tensor> weight,
    Reduction reduction, const Tensor &grad_loss, Tensor &grad_input);
HICE_DECLARE_DISPATCHER(mse_loss_bwd_dispatcher, mse_loss_bwd_kernel_fn_type);
// Backward operators
HICE_API Tensor mse_loss_bwd(const Tensor &input, const Tensor &target,
                             hice::optional<Tensor> weight, Reduction reduction,
                             const Tensor &grad_loss);
HICE_API Tensor &mse_loss_bwd(const Tensor &input, const Tensor &target,
                              hice::optional<Tensor> weight,
                              Reduction reduction, const Tensor &grad_loss,
                              Tensor &grad_input);

}  // namespace hice
