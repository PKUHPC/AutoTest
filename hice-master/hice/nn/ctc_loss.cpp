#include "hice/nn/ctc_loss.h"
#include "hice/math/reduce.h"

namespace hice {

HICE_DEFINE_DISPATCHER(ctc_loss_fwd_dispatcher);

std::tuple<Tensor, Tensor> ctc_loss_fwd(const Tensor &probs,
                                        const Tensor &target,
                                        const Tensor &probs_lengths,
                                        const Tensor &target_lengths,
                                        Reduction reduction) {
  auto max_time = probs.dim(0);
  auto batch_size = probs.dim(1);
  auto max_target_length = target.dim(1);
  Tensor loss({batch_size}, probs.options());
  Tensor log_alphas({batch_size, max_time, max_target_length * 2 + 1},
                    probs.options());
  ctc_loss_fwd_dispatcher(probs, target, probs_lengths, target_lengths, loss,
                          log_alphas);
  if (reduction == Reduction::none) {
    return std::make_tuple(loss, log_alphas);
  }
  Tensor loss_reduced({}, probs.options());
  auto all_range = get_all_dims_range(loss.dims());
  if (reduction == Reduction::mean) {
    reduce_mean(loss, all_range, false, loss_reduced);
  } else if (reduction == Reduction::sum) {
    reduce_sum(loss, all_range, false, loss_reduced);
  }
  return std::make_tuple(loss_reduced, log_alphas);
}

// void ctc_loss_fwd(const Tensor &input, const Tensor &target,
//                   const Tensor &input_lengths, const Tensor &target_lengths,
//                   Reduction reduction, Tensor &loss, Tensor &log_alpha) {
//   ctc_loss_fwd_dispatcher(input, target, input_lengths, target_lengths,
//                           reduction, loss, log_alpha);
//   return loss;
// }

HICE_DEFINE_DISPATCHER(ctc_loss_bwd_dispatcher);

// Backward operators
Tensor ctc_loss_bwd(const Tensor &probs, const Tensor &target,
                             const Tensor &probs_lengths,
                             const Tensor &target_lengths, Reduction reduction,
                             const Tensor &log_alphas,
                             const Tensor &grad_loss) {
  Tensor grad_probs(probs.dims(), probs.options());
  ctc_loss_bwd_dispatcher(probs, target, probs_lengths, target_lengths,
                          reduction, log_alphas, grad_loss, grad_probs);
  return grad_probs;
}

#if 0
Tensor &ctc_loss_bwd(const Tensor &probs,  
                              const Tensor &target, 
                              const Tensor &probs_lengths,
                              const Tensor &target_lengths, 
                              const Tensor &log_alphas,
                              const Tensor &grad_loss, 
                              Tensor &grad_input) {
  ctc_loss_bwd_dispatcher(input, target, weight, reduction, grad_loss, grad_input);
  return grad_input;
}
#endif

}  // namespace hice
