#include "hice/nn/nll_loss.h"

namespace hice {

HICE_DEFINE_DISPATCHER(nll_loss_fwd_dispatcher);

Tensor nll_loss_fwd(const Tensor &input, const Tensor &target,
                    hice::optional<Tensor> weight) {
  std::vector<int64_t> loss_dims(input.dims().begin(), input.dims().end());
  loss_dims.erase(loss_dims.begin() + 1);
  Tensor loss(loss_dims, input.options());
  nll_loss_fwd_dispatcher(input, target, weight, loss);
  return loss;
}

Tensor &nll_loss_fwd(const Tensor &input, const Tensor &target,
                     hice::optional<Tensor> weight, Tensor &loss) {
  nll_loss_fwd_dispatcher(input, target, weight, loss);
  return loss;
}

HICE_DEFINE_DISPATCHER(nll_loss_bwd_dispatcher);

// Backward operators
Tensor nll_loss_bwd(const Tensor &input, const Tensor &target,
                            hice::optional<Tensor> weight,
                            const Tensor &grad_loss) {
  Tensor grad_input(input.dims(), input.options());
  nll_loss_bwd_dispatcher(input, target, weight, grad_loss, grad_input);
  return grad_input;
}

Tensor &nll_loss_bwd(const Tensor &input, const Tensor &target,
                             hice::optional<Tensor> weight,
                             const Tensor &grad_loss, Tensor &grad_input) {
  nll_loss_bwd_dispatcher(input, target, weight, grad_loss, grad_input);
  return grad_input;
}

}  // namespace hice
