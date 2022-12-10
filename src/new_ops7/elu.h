#ifndef ELU
#define ELU

#include "src/core/tensor.h"
/**
 * @brief Applies a elu over an input tensor.
 *
 * @param input The input tensor.
 * @param output The output tensor pointer.
 * @return Status The Status enum indicates whether the routine is OK.
 */
AITISA_API_PUBLIC Status aitisa_elu(Tensor input, float alpha, Tensor* output);

#endif  // ELU
