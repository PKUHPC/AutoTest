#ifndef RGB_TO_GRAYSCALE_H
#define RGB_TO_GRAYSCALE_H

#include "src/core/tensor.h"

/**
 * @brief Convert rgb image to grayscale image.
 *
 * @param input The input tensor.
 * @param output The output tensor pointer.
 * @return Status The Status enum indicates whether the routine is OK.
 */
AITISA_API_PUBLIC Status aitisa_rgb_to_grayscale(const Tensor input, Tensor *output);

#endif // RGB_TO_GRAYSCALE_H
