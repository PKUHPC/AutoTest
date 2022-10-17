#ifndef ADJUST_HUE_H
#define ADJUST_HUE_H

#include "src/core/tensor.h"

/**
 * @brief Adjust the hue of the input image.
 *
 * @param input The input tensor.
 * @param hue_factor Parameter for adjusting the hue of the input image.
 * @param output The output tensor pointer.
 * @return Status The Status enum indicates whether the routine is OK.
 */
AITISA_API_PUBLIC Status aitisa_adjust_hue(const Tensor input, double hue_factor, Tensor *output);

#endif // ADJUST_HUE_H
