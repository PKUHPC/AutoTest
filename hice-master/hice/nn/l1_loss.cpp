#include "hice/nn/l1_loss.h"

namespace hice {

HICE_DEFINE_DISPATCHER(l1_loss_fwd_dispatcher);

Tensor l1_loss_fwd(const Tensor &input, const Tensor &target,
                   hice::optional<Tensor> weight, Reduction reduction) {
  Tensor loss({}, input.options());
  if (reduction == Reduction::none) {
    loss.resize(input.dims());
  }
  l1_loss_fwd_dispatcher(input, target, weight, reduction, loss);
  return loss;
}

Tensor &l1_loss_fwd(const Tensor &input, const Tensor &target,
                    hice::optional<Tensor> weight, Reduction reduction, Tensor &loss) {
  l1_loss_fwd_dispatcher(input, target, weight, reduction, loss);
  return loss;
}

HICE_DEFINE_DISPATCHER(l1_loss_bwd_dispatcher);

// Backward operators
Tensor l1_loss_bwd(const Tensor &input, const Tensor &target,
                            hice::optional<Tensor> weight, Reduction reduction,
                            const Tensor &grad_loss) {
  Tensor grad_input(input.dims(), input.options());
  l1_loss_bwd_dispatcher(input, target, weight, reduction, grad_loss, grad_input);
  return grad_input;
}

Tensor &l1_loss_bwd(const Tensor &input, const Tensor &target,
                             hice::optional<Tensor> weight, Reduction reduction,
                             const Tensor &grad_loss, Tensor &grad_input) {
  l1_loss_bwd_dispatcher(input, target, weight, reduction, grad_loss, grad_input);
  return grad_input;
}

}  // namespace hice
