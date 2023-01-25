#ifndef L1_LOSS
#define L1_LOSS

#include "src/core/tensor.h"

/**
 * @brief This criterion computes the l1 loss between input logits and target.
 *
 * @param input
 * @param target
 * @param weight a manual rescaling weight given to each loss.
 * @param output The output tensor pointer.
 */
AITISA_API_PUBLIC Status aitisa_l1_loss(const Tensor input, const Tensor target,
                                        const Tensor weight,
                                        const int reduction, Tensor* output);

#endif  // L1_LOSS
