#ifndef CTC_LOSS
#define CTC_LOSS

#include "src/core/tensor.h"

/**
 * @brief This criterion computes the cross entropy loss between input logits and target.
 *
 * @param prob float or double, [max_time, batch_size, num_classes]
 * @param target int32 or int64, [batch_size, max_length]
 * @param probs_lengths  same type with target, number of time-steps of each
 * @param target_lengths same type with target, length of target of each
 * @param reduction Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
 * @param output The output tensor pointer.
 */
AITISA_API_PUBLIC Status aitisa_ctc_loss(const Tensor probs, const Tensor target,
                                         const Tensor probs_lengths,
                                         const Tensor target_lengths,
                                         Tensor* output);

#endif  // CTC_LOSS
