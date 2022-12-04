#ifndef MSE_LOSS
#define MSE_LOSS

#include "src/core/tensor.h"

/**
 * @brief This criterion computes the l1 loss between input logits and target.
 *
 * @param input
 * @param target
 * @param weight a manual rescaling weight given to each loss.
 * @param output The output tensor pointer.s
 */
AITISA_API_PUBLIC Status aitisa_mse_loss(const Tensor input, const Tensor target,
                                        const Tensor weight, Tensor* output);

#endif  // MSE_LOSS