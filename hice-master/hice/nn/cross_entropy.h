#pragma once

#include "hice/core/dispatch.h"
#include "hice/core/tensor.h"

namespace hice {
// forward dispatcher
using cross_entropy_fwd_kernel_fn_type = void (*)(const Tensor &prob,
                                                  const Tensor &target,
                                                  hice::optional<Tensor> weight,
                                                  const int64_t axis,
                                                  Tensor &loss);
HICE_DECLARE_DISPATCHER(cross_entropy_fwd_dispatcher,
                        cross_entropy_fwd_kernel_fn_type);
// Forward operators
HICE_API Tensor cross_entropy_fwd(const Tensor &prob, const Tensor &target,
                                  hice::optional<Tensor> weight,
                                  const int64_t axis);
HICE_API Tensor &cross_entropy_fwd(const Tensor &prob, const Tensor &target,
                                   hice::optional<Tensor> weight,
                                   const int64_t axis, Tensor &loss);

// Backward dispatcher
using cross_entropy_bwd_kernel_fn_type = void (*)(
    const Tensor &prob, const Tensor &target, hice::optional<Tensor> weight,
    const Tensor &grad_loss, const int64_t axis, Tensor &grad_prob);
HICE_DECLARE_DISPATCHER(cross_entropy_bwd_dispatcher,
                        cross_entropy_bwd_kernel_fn_type);
// Backward operators
HICE_API Tensor cross_entropy_bwd(const Tensor &prob, const Tensor &target,
                                  hice::optional<Tensor> weight,
                                  const Tensor &grad_loss, const int64_t axis);
HICE_API Tensor &cross_entropy_bwd(const Tensor &prob, const Tensor &target,
                                   hice::optional<Tensor> weight,
                                   const Tensor &grad_loss, const int64_t axis,
                                   Tensor &grad_prob);

}  // namespace hice