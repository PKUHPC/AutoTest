#ifndef YIQ_TO_RGB_H
#define YIQ_TO_RGB_H

#include "src/core/tensor.h"

/**
 * @brief Convert yiq image to rgb image.
 *
 * @param input The input tensor.
 * @param output The output tensor pointer.
 * @return Status The Status enum indicates whether the routine is OK.
 */
AITISA_API_PUBLIC Status aitisa_yiq_to_rgb(const Tensor input, Tensor *output);

#endif // YIQ_TO_RGB_H
