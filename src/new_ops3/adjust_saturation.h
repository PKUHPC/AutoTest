#ifndef ADJUST_SATURATION_H
#define ADJUST_SATURATION_H

#include "src/core/tensor.h"

/**
 * @brief Adjust the saturation of the input image.
 *
 * @param input The input tensor.
 * @param saturation_factor Parameter for adjusting the saturation of the input image.
 * @param output The output tensor pointer.
 * @return Status The Status enum indicates whether the routine is OK.
 */
AITISA_API_PUBLIC Status aitisa_adjust_saturation(const Tensor input, double saturation_factor, Tensor *output);

#endif // ADJUST_SATURATION_H
