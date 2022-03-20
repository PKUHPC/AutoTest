#include "hice/nn/mse_loss.h"

namespace hice {

HICE_DEFINE_DISPATCHER(mse_loss_fwd_dispatcher);

Tensor mse_loss_fwd(const Tensor &input, const Tensor &target,
                    hice::optional<Tensor> weight, Reduction reduction) {
  Tensor loss({}, input.options());
  if (reduction == Reduction::none) {
    loss.resize(input.dims());
  }
  mse_loss_fwd_dispatcher(input, target, weight, reduction, loss);
  return loss;
}

Tensor &mse_loss_fwd(const Tensor &input, const Tensor &target,
                     hice::optional<Tensor> weight, Reduction reduction,
                     Tensor &loss) {
  mse_loss_fwd_dispatcher(input, target, weight, reduction, loss);
  return loss;
}

HICE_DEFINE_DISPATCHER(mse_loss_bwd_dispatcher);

// Backward operators
Tensor mse_loss_bwd(const Tensor &input, const Tensor &target,
                             hice::optional<Tensor> weight, Reduction reduction,
                             const Tensor &grad_loss) {
  Tensor grad_input(input.dims(), input.options());
  mse_loss_bwd_dispatcher(input, target, weight, reduction, grad_loss, grad_input);
  return grad_input;
}

Tensor &mse_loss_bwd(const Tensor &input, const Tensor &target,
                              hice::optional<Tensor> weight,
                              Reduction reduction, const Tensor &grad_loss,
                              Tensor &grad_input) {
  mse_loss_bwd_dispatcher(input, target, weight, reduction, grad_loss, grad_input);
  return grad_input;
}

}  // namespace hice
