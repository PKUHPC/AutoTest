#ifndef RANDOM_CROP_H
#define RANDOM_CROP_H

#include "src/core/tensor.h"

/**
 * @brief Randomly crop the image to a given size.
 *
 * @param input The input tensor.
 * @param target_h The height of the cropped image.
 * @param target_w The width of the cropped image.
 * @param output The output tensor pointer.
 * @return Status The Status enum indicates whether the routine is OK.
 */
AITISA_API_PUBLIC Status aitisa_random_crop(const Tensor input, int target_h, int target_w, Tensor *output);

#endif // RANDOM_CROP_H
