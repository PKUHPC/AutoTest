#ifndef HSV_TO_RGB_H
#define HSV_TO_RGB_H

#include "src/core/tensor.h"

/**
 * @brief Convert hsv image to rgb image.
 *
 * @param input The input tensor.
 * @param output The output tensor pointer.
 * @return Status The Status enum indicates whether the routine is OK.
 */
AITISA_API_PUBLIC Status aitisa_hsv_to_rgb(const Tensor input, Tensor *output);

#endif // HSV_TO_RGB_H
