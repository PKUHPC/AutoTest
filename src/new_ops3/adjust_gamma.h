#ifndef ADJUST_GAMMA_H
#define ADJUST_GAMMA_H

#include "src/core/tensor.h"

/**
 * @brief Gamma transform the image according to the input parameters.
 *
 * @param input The input tensor.
 * @param gain Parameter gain of the transform.
 * @param gamma Parameter gamma of the transform.
 * @param output The output tensor pointer.
 * @return Status The Status enum indicates whether the routine is OK.
 */
AITISA_API_PUBLIC Status aitisa_adjust_gamma(const Tensor input, double gain, double gamma, Tensor *output);

#endif // ADJUST_GAMMA_H
