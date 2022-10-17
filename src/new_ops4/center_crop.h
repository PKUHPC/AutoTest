#ifndef CENTER_CROP_H
#define CENTER_CROP_H

#include "src/core/tensor.h"

/**
 * @brief Crop from the center of the input image according to the specified size.
 *
 * @param input The input tensor.
 * @param target_h The height of the cropped image.
 * @param target_w The width of the cropped image.
 * @param output The output tensor pointer.
 * @return Status The Status enum indicates whether the routine is OK.
 */
AITISA_API_PUBLIC Status aitisa_center_crop(const Tensor input, int target_h, int target_w, Tensor *output);

#endif // CENTER_CROP_H
