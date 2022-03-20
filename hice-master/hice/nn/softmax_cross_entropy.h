#pragma once

#include <hice/util/types.h>
#include <tuple>
#include "hice/core/dispatch.h"
#include "hice/core/tensor.h"

namespace hice {

// forward dispatcher
using softmax_cross_entropy_fwd_kernel_fn_type = void (*)(
    const Tensor &logit, const Tensor &target, hice::optional<Tensor> weight,
    const int64_t axis, Tensor &prob, Tensor &loss);
HICE_DECLARE_DISPATCHER(softmax_cross_entropy_fwd_dispatcher,
                        softmax_cross_entropy_fwd_kernel_fn_type);
// Forward operators
HICE_API std::tuple<Tensor, Tensor> softmax_cross_entropy_fwd(
    const Tensor &logit, const Tensor &target, hice::optional<Tensor> weight,
    const int64_t axis);
HICE_API std::tuple<Tensor &, Tensor &> softmax_cross_entropy_fwd(
    const Tensor &logit, const Tensor &target, hice::optional<Tensor> weight,
    const int64_t axis, Tensor &prob, Tensor &loss);

// Backward dispatcher
using softmax_cross_entropy_bwd_kernel_fn_type = void (*)(
    const Tensor &prob, const Tensor &target, hice::optional<Tensor> weight,
    const Tensor &grad_loss, const int64_t axis, Tensor &grad_logit);
HICE_DECLARE_DISPATCHER(softmax_cross_entropy_bwd_dispatcher,
                        softmax_cross_entropy_bwd_kernel_fn_type);
// Backward operators
HICE_API Tensor softmax_cross_entropy_bwd(const Tensor &prob,
                                          const Tensor &target,
                                          hice::optional<Tensor> weight,
                                          const Tensor &grad_loss,
                                          const int64_t axis);
HICE_API Tensor &softmax_cross_entropy_bwd(
    const Tensor &prob, const Tensor &target, hice::optional<Tensor> weight,
    const Tensor &grad_loss, const int64_t axis, Tensor &grad_logit);

}  // namespace hice