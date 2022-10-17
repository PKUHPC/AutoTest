#ifndef CROP_TO_BOUNDING_BOX_H
#define CROP_TO_BOUNDING_BOX_H

#include "src/core/tensor.h"

/**
 * @brief Crop the boundary of the input image to the given range.
 *
 * @param input The input tensor.
 * @param offset_h The starting position of the upper boundary.
 * @param offset_w The starting position of the left boundary.
 * @param target_h The height of the cropped image.
 * @param target_w The width of the cropped image.
 * @param output The output tensor pointer.
 * @return Status The Status enum indicates whether the routine is OK.
 */
AITISA_API_PUBLIC Status aitisa_crop_to_bounding_box(const Tensor input, int offset_h, int offset_w, int target_h, int target_w, Tensor *output);

#endif // CROP_TO_BOUNDING_BOX_H
