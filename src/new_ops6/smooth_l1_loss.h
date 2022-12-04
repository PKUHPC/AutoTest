#ifndef SMOOTH_L1_LOSS
#define SMOOTH_L1_LOSS

#include "src/core/tensor.h"

/**
 * @brief This criterion computes the smooth l1 loss between input logits and target.
 *
 * @param input
 * @param target
 * @param weight a manual rescaling weight given to each loss.
 * @param output The output tensor pointer.
 */
AITISA_API_PUBLIC Status aitisa_smooth_l1_loss(const Tensor input,
                                               const Tensor target,
                                               const Tensor weight,
                                               Tensor* output);

#endif  // SMOOTH_L1_LOSS
