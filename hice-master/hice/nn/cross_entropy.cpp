#include "hice/nn/cross_entropy.h"
#include <vector>

namespace hice {

HICE_DEFINE_DISPATCHER(cross_entropy_fwd_dispatcher);
// Forward operators
Tensor cross_entropy_fwd(const Tensor &prob, const Tensor &target,
                         hice::optional<Tensor> weight, const int64_t axis) {
  std::vector<int64_t> loss_dims(target.dims().begin(), target.dims().end());
  loss_dims.erase(loss_dims.begin() + target.get_true_axis(axis));
  Tensor loss({loss_dims}, device(prob.device()).dtype(prob.data_type()));
  // if (!weight.is_defined())
  //  const_cast<Tensor &>(weight) = Tensor(prob.options());
  cross_entropy_fwd_dispatcher(prob, target, weight, axis, loss);
  return loss;
}

Tensor &cross_entropy_fwd(const Tensor &prob, const Tensor &target,
                          hice::optional<Tensor> weight, const int64_t axis,
                          Tensor &loss) {
  cross_entropy_fwd_dispatcher(prob, target, weight, axis, loss);
  return loss;
}

HICE_DEFINE_DISPATCHER(cross_entropy_bwd_dispatcher);
// Backward operators
Tensor cross_entropy_bwd(const Tensor &prob, const Tensor &target,
                         hice::optional<Tensor> weight, const Tensor &grad_loss,
                         const int64_t axis) {
  Tensor grad_prob(prob.dims(), prob.options());
  cross_entropy_bwd_dispatcher(prob, target, weight, grad_loss, axis,
                               grad_prob);
  return grad_prob;
}

Tensor &cross_entropy_bwd(const Tensor &prob, const Tensor &target,
                          hice::optional<Tensor> weight,
                          const Tensor &grad_loss, const int64_t axis,
                          Tensor &grad_prob) {
  cross_entropy_bwd_dispatcher(prob, target, weight, grad_loss, axis,
                               grad_prob);
  return grad_prob;
}

}  // namespace hice