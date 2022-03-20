#include "hice/nn/sparse_cross_entropy.h"

namespace hice {

HICE_DEFINE_DISPATCHER(sparse_cross_entropy_fwd_dispatcher);
// Forward operators
Tensor sparse_cross_entropy_fwd(const Tensor &prob, const Tensor &target,
                                hice::optional<Tensor> weight, const int64_t axis) {
  Tensor loss({target.dims()}, device(prob.device()).dtype(prob.data_type()));
  // if (!weight.defined()) const_cast<Tensor &>(weight) =
  // Tensor(prob.options());
  sparse_cross_entropy_fwd_dispatcher(prob, target, weight, axis, loss);
  return loss;
}

Tensor &sparse_cross_entropy_fwd(const Tensor &prob, const Tensor &target,
                                 hice::optional<Tensor> weight, const int64_t axis,
                                 Tensor &loss) {
  //if (!weight.defined()) const_cast<Tensor &>(weight) = Tensor(prob.options());
  sparse_cross_entropy_fwd_dispatcher(prob, target, weight, axis, loss);
  return loss;
}

HICE_DEFINE_DISPATCHER(sparse_cross_entropy_bwd_dispatcher);
// Backward operators
Tensor sparse_cross_entropy_bwd(const Tensor &prob, const Tensor &target,
                                hice::optional<Tensor> weight, const Tensor &grad_loss,
                                const int64_t axis) {
  Tensor grad_prob(prob.dims(), prob.options());
  //if (!weight.defined()) const_cast<Tensor &>(weight) = Tensor(prob.options());
  sparse_cross_entropy_bwd_dispatcher(prob, target, weight, grad_loss, axis,
                                      grad_prob);
  return grad_prob;
}

Tensor &sparse_cross_entropy_bwd(const Tensor &prob, const Tensor &target,
                                 hice::optional<Tensor> weight, const Tensor &grad_loss,
                                 const int64_t axis, Tensor &grad_prob) {
  //if (!weight.defined()) const_cast<Tensor &>(weight) = Tensor(prob.options());
  sparse_cross_entropy_bwd_dispatcher(prob, target, weight, grad_loss, axis,
                                      grad_prob);
  return grad_prob;
}

} // namespace hice