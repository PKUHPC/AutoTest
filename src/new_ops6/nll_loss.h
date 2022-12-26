#ifndef NLL_LOSS
#define NLL_LOSS

#include "src/core/tensor.h"

/**
 * @brief This criterion computes the nll loss between input logits and target.
 *
 * @param prob float or double, [max_time, batch_size, num_classes]
 * @param target int32 or int64, [batch_size, max_length]
 * @param weight  same type with target, number of time-steps of each
 * @param reduction Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
 * @param output The output tensor pointer.
 */
AITISA_API_PUBLIC Status aitisa_nll_loss(const Tensor probs, const Tensor target,
                                         const Tensor weight,
                                         Tensor* output);

#endif  // NLL_LOSS
