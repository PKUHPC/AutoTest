#include <functional>
#include "hice/nn/softmax_cross_entropy.h"

namespace hice {

HICE_DEFINE_DISPATCHER(softmax_cross_entropy_fwd_dispatcher);
// Forward operators
std::tuple<Tensor, Tensor> softmax_cross_entropy_fwd(
    const Tensor &logit, const Tensor &target, hice::optional<Tensor> weight,
    const int64_t axis) {
  Tensor prob({logit.dims()}, device(logit.device()).dtype(logit.data_type()));
  std::vector<int64_t> loss_dims(target.dims().begin(), target.dims().end());
  loss_dims.erase(loss_dims.begin() + target.get_true_axis(axis));
  Tensor loss({loss_dims}, device(target.device()).dtype(target.data_type()));
  softmax_cross_entropy_fwd_dispatcher(logit, target, weight, axis, prob, loss);
  return std::tuple<Tensor, Tensor>(prob, loss);
}

std::tuple<Tensor &, Tensor &> softmax_cross_entropy_fwd(
    const Tensor &logit, const Tensor &target, hice::optional<Tensor> weight,
    const int64_t axis, Tensor &prob, Tensor &loss) {
  if (!prob.is_defined())
    Tensor prob({logit.dims()},
                device(logit.device()).dtype(logit.data_type()));
  softmax_cross_entropy_fwd_dispatcher(logit, target, weight, axis, prob, loss);
  return std::tuple<Tensor &, Tensor &>(prob, loss);
}

HICE_DEFINE_DISPATCHER(softmax_cross_entropy_bwd_dispatcher);
// Backward operators
Tensor softmax_cross_entropy_bwd(const Tensor &prob, const Tensor &target,
                                 hice::optional<Tensor> weight,
                                 const Tensor &grad_loss, const int64_t axis) {
  Tensor grad_logit(prob.dims(), prob.options());
  softmax_cross_entropy_bwd_dispatcher(prob, target, weight, grad_loss, axis,
                                       grad_logit);
  return grad_logit;
}

Tensor &softmax_cross_entropy_bwd(const Tensor &prob, const Tensor &target,
                                  const Tensor &weight, const Tensor &grad_loss,
                                  const int64_t axis, Tensor &grad_logit) {
  softmax_cross_entropy_bwd_dispatcher(prob, target, weight, grad_loss, axis,
                                       grad_logit);
  return grad_logit;
}

}  // namespace hice