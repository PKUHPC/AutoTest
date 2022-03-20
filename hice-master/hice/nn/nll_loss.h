#pragma once

#include "hice/core/dispatch.h"
#include "hice/core/tensor.h"

namespace hice {

// Forward dispatcher
using nll_loss_fwd_kernel_fn_type = void (*)(const Tensor &input,
                                             const Tensor &target,
                                             hice::optional<Tensor> weight,
                                             Tensor &loss);
HICE_DECLARE_DISPATCHER(nll_loss_fwd_dispatcher, nll_loss_fwd_kernel_fn_type);
// Forward operators
HICE_API Tensor nll_loss_fwd(const Tensor &input, const Tensor &target,
                             hice::optional<Tensor> weight);
HICE_API Tensor &nll_loss_fwd(const Tensor &input, const Tensor &target,
                              hice::optional<Tensor> weight, Tensor &loss);

// Backward dispatcher
using nll_loss_bwd_kernel_fn_type = void (*)(const Tensor &input,
                                            const Tensor &target,
                                            hice::optional<Tensor> weight,
                                            const Tensor &grad_loss,
                                            Tensor &grad_input);
HICE_DECLARE_DISPATCHER(nll_loss_bwd_dispatcher, nll_loss_bwd_kernel_fn_type);
// Backward operators
HICE_API Tensor nll_loss_bwd(const Tensor &input, const Tensor &target,
                            hice::optional<Tensor> weight,
                            const Tensor &grad_loss);
HICE_API Tensor &nll_loss_bwd(const Tensor &input, const Tensor &target,
                             hice::optional<Tensor> weight,
                             const Tensor &grad_loss, Tensor &grad_input);

}  // namespace hice
