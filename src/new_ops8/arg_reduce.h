#ifndef ARG_REDUCE
#define ARG_REDUCE

#include "src/core/tensor.h"
/**
 * @brief Returns the indices of the maximum values along an axis.
*
 * @param input The input tensor.
 * @param output The output tensor pointer.
 * @param dim the dimension or dimensions to reduce.
 * @param keepdim whether the output tensor has dim retained or not.
 * @return Status The Status enum indicates whether the routine is OK.
 */
AITISA_API_PUBLIC Status aitisa_argmax(const Tensor input, const int64_t dim,
                                       const int keepdim, Tensor* output);

/**
 * @brief Returns the indices of the maximum values along an axis.
 *
 * @param input The input tensor.
 * @param output The output tensor pointer.
 * @param dim the dimension or dimensions to reduce.
 * @param keepdim whether the output tensor has dim retained or not.
 * @return Status The Status enum indicates whether the routine is OK.
 */
AITISA_API_PUBLIC Status aitisa_argmin(const Tensor input, const int64_t dim,
                                       const int keepdim, Tensor* output);

/**
 * @brief Returns the values of the maximum values along an axis.
 *
 * @param input The input tensor.
 * @param output The output tensor pointer.
 * @param dim the dimension or dimensions to reduce.
 * @param keepdim whether the output tensor has dim retained or not.
 * @return Status The Status enum indicates whether the routine is OK.
 */
AITISA_API_PUBLIC Status aitisa_min(const Tensor input, const int64_t dim,
                                    const int keepdim, Tensor* output);

/**
 * @brief Returns the values of the maximum values along an axis.
 *
 * @param input The input tensor.
 * @param output The output tensor pointer.
 * @param dim the dimension or dimensions to reduce.
 * @param keepdim whether the output tensor has dim retained or not.
 * @return Status The Status enum indicates whether the routine is OK.
 */
AITISA_API_PUBLIC Status aitisa_max(const Tensor input, const int64_t dim,
                                    const int keepdim, Tensor* output);

#endif  // ARG_REDUCE
