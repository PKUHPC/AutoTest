#ifndef IMAGE_GRADIENTS_H
#define IMAGE_GRADIENTS_H

#include "src/core/tensor.h"

/**
 * @brief Calculate the gradients of the input image in x and y directions.
 *
 * @param tensor1 Input image.
 * @param grad_x The gradient in the x direction.
 * @param grad_y The gradient in the y direction.
 * @return Status The Status enum indicates whether the routine is OK.
 */

AITISA_API_PUBLIC Status aitisa_image_gradients(const Tensor input,
                                                Tensor *grad_x,
                                                Tensor *grad_y);

#endif