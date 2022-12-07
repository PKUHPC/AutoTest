#ifndef EXP
#define EXP

#include "src/core/tensor.h"
/**
 * @brief Applies a exp over an input tensor.
 *
 * @param input The input tensor.
 * @param output The output tensor pointer.
 * @return Status The Status enum indicates whether the routine is OK.
 */
AITISA_API_PUBLIC Status aitisa_exp(const Tensor input, Tensor* output);

#endif  // EXP
