#ifndef ABS
#define ABS

#include "src/core/tensor.h"
/**
 * @brief Applies a abs  over an input tensor.
 *
 * @param input The input tensor.
 * @param output The output tensor pointer.
 * @return Status The Status enum indicates whether the routine is OK.
 */
AITISA_API_PUBLIC Status aitisa_abs(const Tensor input, Tensor* output);

#endif  // ABS
