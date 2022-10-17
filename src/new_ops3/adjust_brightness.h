#ifndef ADJUST_BRIGHTNESS_H
#define ADJUST_BRIGHTNESS_H

#include "src/core/tensor.h"

/**
 * @brief Adjust the brightness of the input image.
 *
 * @param input The input tensor.
 * @param brightness_factor Parameter for adjusting the brightness of the input image.
 * @param output The output tensor pointer.
 * @return Status The Status enum indicates whether the routine is OK.
 */
AITISA_API_PUBLIC Status aitisa_adjust_brightness(const Tensor input, double brightness_factor, Tensor *output);

#endif // ADJUST_BRIGHTNESS_H
