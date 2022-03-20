#include <functional>
#include "hice/nn/sparse_softmax_cross_entropy.h"

namespace hice {

HICE_DEFINE_DISPATCHER(sparse_softmax_cross_entropy_fwd_dispatcher);
// Forward operators
std::tuple<Tensor, Tensor> sparse_softmax_cross_entropy_fwd(
    const Tensor &logit, const Tensor &target, hice::optional<Tensor> weight,
    const int64_t axis) {
  Tensor prob({logit.dims()}, device(logit.device()).dtype(logit.data_type()));
  Tensor loss({target.dims()}, device(logit.device()).dtype(logit.data_type()));
  sparse_softmax_cross_entropy_fwd_dispatcher(logit, target, weight, axis, prob,
                                              loss);
  return std::tuple<Tensor, Tensor>(prob, loss);
}

std::tuple<Tensor&, Tensor&> sparse_softmax_cross_entropy_fwd(
    const Tensor &logit, const Tensor &target, hice::optional<Tensor> weight,
    const int64_t axis, Tensor &prob, Tensor &loss) {
  if (!prob.is_defined())
    Tensor prob({logit.dims()},
                device(logit.device()).dtype(logit.data_type()));
  sparse_softmax_cross_entropy_fwd_dispatcher(logit, target, weight, axis, prob,
                                              loss);
  return std::tuple<Tensor &, Tensor &>(prob, loss);
}

HICE_DEFINE_DISPATCHER(sparse_softmax_cross_entropy_bwd_dispatcher);
// Backward operators
Tensor sparse_softmax_cross_entropy_bwd(const Tensor &prob,
                                        const Tensor &target,
                                        hice::optional<Tensor> weight,
                                        const Tensor &grad_loss,
                                        const int64_t axis) {
  Tensor grad_logit(prob.dims(), prob.options());
  sparse_softmax_cross_entropy_bwd_dispatcher(prob, target, weight, grad_loss,
                                              axis, grad_logit);
  return grad_logit;
}

Tensor &sparse_softmax_cross_entropy_bwd(
    const Tensor &prob, const Tensor &target, hice::optional<Tensor> weight,
    const Tensor &grad_loss, const int64_t axis, Tensor &grad_logit) {
  sparse_softmax_cross_entropy_bwd_dispatcher(prob, target, weight, grad_loss,
                                              axis, grad_logit);
  return grad_logit;
}

}  // namespace hice