#ifndef RANDOM_HUE_H
#define RANDOM_HUE_H

#include "src/core/tensor.h"

/**
 * @brief Randomly adjust the hue of the input image according to the given parameter range.
 *
 * @param input The input tensor.
 * @param lower The lower bound of the hue factor.
 * @param upper The upper bound of the hue factor.
 * @param output The output tensor pointer.
 * @return Status The Status enum indicates whether the routine is OK.
 */
AITISA_API_PUBLIC Status aitisa_random_hue(const Tensor input, double lower, double upper, Tensor *output);

#endif // RANDOM_HUE_H
