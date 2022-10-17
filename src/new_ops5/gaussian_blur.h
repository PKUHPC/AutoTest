#ifndef GAUSSIAN_BLUR_H
#define GAUSSIAN_BLUR_H

#include "src/core/tensor.h"

/**
 * @brief Performs a Gaussian blur of the image
 *
 * @param tensor1 Input image.
 * @param kernel_size The size of the convolutional kernel.
 * @param sigma The sigma parameter of the convolutional kernel.
 * @param output The output image pointer.
 * @return Status The Status enum indicates whether the routine is OK.
 */

AITISA_API_PUBLIC Status aitisa_gaussian_blur(const Tensor input,
                                              const int kernel_size,
                                              const double sigma,
                                              Tensor *output);

#endif