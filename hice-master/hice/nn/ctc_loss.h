#pragma once

#include "hice/basic/factories.h"
#include "hice/core/dispatch.h"
#include "hice/core/tensor.h"
#include "hice/nn/loss_reduction_type.h"

namespace hice {

// This is the implementation of ctc_loss_fwd
// To ensure the numerical stability, log_alpha is kept instead.
// -param
//      probs: float or double, [max_time, batch_size, num_classes]
//      target: int32 or int64, [batch_size, max_length]
//      probs_lengths: same type with target, number of time-steps of each
//      sample. target_lengths: same type with target, length of target of each
//      sample. loss: same type with probs, [batch_size] or scalar tensor
//      log_alpha: same type with probs, [batch_size, max_time, max_length * 2 +
//      1]
// -return
//      tuple<loss, log_alpha>
//
// NOTE: 1. The grad_probs gained from hice::ctc_backward is 1 lower than pytorch(I don't even know why).
//       2. cudnn_ctc_loss is disabled for the following two reasons:
//          1) The result of cudnn_ctc_loss are inconsistence with hice::ctc_loss_bwd.
//          2) cudnn_ctc_loss does not keep log_alphas for users.
//         
// Forward dispatcher
using ctc_loss_fwd_kernel_fn_type = void (*)(const Tensor &probs,
                                             const Tensor &target,
                                             const Tensor &probs_lengths,
                                             const Tensor &target_lengths,
                                             Tensor &loss, Tensor &log_alphas);
HICE_DECLARE_DISPATCHER(ctc_loss_fwd_dispatcher, ctc_loss_fwd_kernel_fn_type);
// Forward operators
HICE_API std::tuple<Tensor, Tensor> ctc_loss_fwd(const Tensor &probs,
                                                 const Tensor &target,
                                                 const Tensor &probs_lengths,
                                                 const Tensor &target_lengths,
                                                 Reduction reduction);

// Backward dispatcher
using ctc_loss_bwd_kernel_fn_type = void (*)(const Tensor &probs,  
                                            const Tensor &target, 
                                            const Tensor &probs_lengths,
                                            const Tensor &target_lengths,
                                            Reduction reduction, 
                                            const Tensor &log_alphas,
                                            const Tensor &grad_loss, 
                                            Tensor &grad_probs);
HICE_DECLARE_DISPATCHER(ctc_loss_bwd_dispatcher, ctc_loss_bwd_kernel_fn_type);
// Backward operators
HICE_API Tensor ctc_loss_bwd(const Tensor &probs, 
                             const Tensor &target,
                             const Tensor &probs_lengths,
                             const Tensor &target_lengths,
                             Reduction reduction,
                             const Tensor &log_alphas, 
                             const Tensor &grad_loss);

#if 0
HICE_API Tensor &ctc_loss_bwd(const Tensor &probs, 
                             const Tensor &target,
                             const Tensor &probs_lengths,
                             const Tensor &target_lengths,
                             Reduction reduction,
                             const Tensor &log_alphas, 
                             const Tensor &grad_loss,
                             Tensor &grad_probs);
#endif

}  // namespace hice
