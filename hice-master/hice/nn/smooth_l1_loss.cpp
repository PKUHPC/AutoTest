#include "hice/nn/smooth_l1_loss.h"

namespace hice {

HICE_DEFINE_DISPATCHER(smooth_l1_loss_fwd_dispatcher);

Tensor smooth_l1_loss_fwd(const Tensor &input, const Tensor &target,
                   hice::optional<Tensor> weight, Reduction reduction) {
  Tensor loss({}, input.options());
  if (reduction == Reduction::none) {
    loss.resize(input.dims());
  }
  smooth_l1_loss_fwd_dispatcher(input, target, weight, reduction, loss);
  return loss;
}

Tensor &smooth_l1_loss_fwd(const Tensor &input, const Tensor &target,
                    hice::optional<Tensor> weight, Reduction reduction, Tensor &loss) {
  smooth_l1_loss_fwd_dispatcher(input, target, weight, reduction, loss);
  return loss;
}

HICE_DEFINE_DISPATCHER(smooth_l1_loss_bwd_dispatcher);

// Backward operators
Tensor smooth_l1_loss_bwd(const Tensor &input, const Tensor &target,
                            hice::optional<Tensor> weight, Reduction reduction,
                            const Tensor &grad_loss) {
  Tensor grad_input(input.dims(), input.options());
  smooth_l1_loss_bwd_dispatcher(input, target, weight, reduction, grad_loss, grad_input);
  return grad_input;
}

Tensor &smooth_l1_loss_bwd(const Tensor &input, const Tensor &target,
                             hice::optional<Tensor> weight, Reduction reduction,
                             const Tensor &grad_loss, Tensor &grad_input) {
  smooth_l1_loss_bwd_dispatcher(input, target, weight, reduction, grad_loss, grad_input);
  return grad_input;
}

}  // namespace hice
