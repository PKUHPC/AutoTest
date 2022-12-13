#ifndef REDUCE_PROD
#define REDUCE_PROD

#include "src/core/tensor.h"
/**
 * @brief Applies a reduce_prod over an input tensor.
 *
 * @param input The input tensor.
 * @param output The output tensor pointer.
 * @param dims the dimension or dimensions to reduce
 * @param keepdim whether the output tensor has dim retained or not.
 * @return Status The Status enum indicates whether the routine is OK.
 */
AITISA_API_PUBLIC Status aitisa_reduce_prod(const Tensor input,
                                            const int64_t* dims,
                                            const int64_t dims_length,
                                            const int keepdim, Tensor* output);

#endif  // REDUCE_PROD
