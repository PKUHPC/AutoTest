#include "hice/basic/copy.h"
#include "hice/core/expression_util.h"
#include "hice/math/binary_expr.h"
#include "hice/math/cpu/eval_expr_dense.h"
#include "hice/math/reduce.h"
#include "hice/nn/activation.h"
#include "hice/nn/l1_loss.h"

namespace hice {

namespace {

void l1_loss_fwd_impl(const Tensor &input, const Tensor &target,
                      hice::optional<Tensor> weight, Reduction reduction,
                      Tensor &loss) {
  Tensor tmp = sub(input, target);
  abs_fwd(tmp, tmp);
  if (weight) {
    mul(tmp, weight.value(), tmp);
  }
  switch (reduction) {
    case Reduction::none: {
      hice::copy(tmp, loss);
      break;
    }
    case Reduction::mean: {
      auto all_range = get_all_dims_range(tmp.dims());
      reduce_sum(tmp, all_range, false, loss);
      auto size = input.size();
      div(loss, size, loss);
      break;
    }
    case Reduction::sum: {
      auto all_range = get_all_dims_range(tmp.dims());
      reduce_sum(tmp, all_range, false, loss);
      break;
    }
    default: {
      HICE_LOG(ERROR) << "Not supported reduction type";
      break;
    }
  }
}

void l1_loss_bwd_impl(const Tensor &input, const Tensor &target,
                      hice::optional<Tensor> weight, Reduction reduction,
                      const Tensor &grad_loss, Tensor &grad_input) {
  Tensor tmp = sub(input, target);
  abs_bwd(tmp, grad_loss, grad_input);
  if (weight) {
    mul(grad_input, weight.value(), grad_input);
  }
  if (reduction == Reduction::mean) {
    auto size = input.size();
    div(grad_input, size, grad_input);
  }
}

}  // namespace

// Forward
HICE_REGISTER_KERNEL(l1_loss_fwd_dispatcher, &l1_loss_fwd_impl,
                     {kCPU, kDense},  // input
                     {kCPU, kDense},  // target
                     {kCPU, kDense}   // loss
);

// Backward
HICE_REGISTER_KERNEL(l1_loss_bwd_dispatcher, &l1_loss_bwd_impl,
                     {kCPU, kDense},  // input
                     {kCPU, kDense},  // target
                     {kCPU, kDense},  // grad_loss
                     {kCPU, kDense}   // grad_input
);

}  // namespace hice
