#ifndef RGB_TO_YIQ_H
#define RGB_TO_YIQ_H

#include "src/core/tensor.h"

/**
 * @brief Convert rgb image to yiq image.
 *
 * @param input The input tensor.
 * @param output The output tensor pointer.
 * @return Status The Status enum indicates whether the routine is OK.
 */
AITISA_API_PUBLIC Status aitisa_rgb_to_yiq(const Tensor input, Tensor *output);

#endif // RGB_TO_YIQ_H
