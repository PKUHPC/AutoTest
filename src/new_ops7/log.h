#ifndef AITISA_LOG
#define AITISA_LOG

#include "src/core/tensor.h"
/**
 * @brief Applies a log over an input tensor.
 *
 * @param input The input tensor.
 * @param output The output tensor pointer.
 * @return Status The Status enum indicates whether the routine is OK.
 */
AITISA_API_PUBLIC Status aitisa_log(const Tensor input, Tensor* output);

#endif  // AITISA_LOG