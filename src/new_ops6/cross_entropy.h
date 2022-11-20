#ifndef CROSS_ENTROPY
#define CROSS_ENTROPY

#include "src/core/tensor.h"

/**
 * @brief This criterion computes the cross entropy loss between input logits and target.
 *
 * @param prob Predicted unnormalized logits.
 * @param target Ground truth class indices or class probabilities.
 * @param weight  a manual rescaling weight given to each class.
 * @param reduction Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
 * @param output The output tensor pointer.
 * @return Status The Status enum indicates whether the routine is OK.
 */
AITISA_API_PUBLIC Status aitisa_cross_entropy(const Tensor prob, const Tensor target, const Tensor weight, Tensor *output);

#endif // CROSS_ENTROPY
