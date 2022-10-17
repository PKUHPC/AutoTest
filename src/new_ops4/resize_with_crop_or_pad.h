#ifndef RESIZE_WITH_CROP_OR_PAD_H
#define RESIZE_WITH_CROP_OR_PAD_H

#include "src/core/tensor.h"

/**
 * @brief Transforms the image to a specified size by cropping or padding.
 *
 * @param input The input tensor.
 * @param target_h The height of the cropped image.
 * @param target_w The width of the cropped image.
 * @param output The output tensor pointer.
 * @return Status The Status enum indicates whether the routine is OK.
 */
AITISA_API_PUBLIC Status aitisa_resize_with_crop_or_pad(const Tensor input, int target_h, int target_w, Tensor *output);

#endif // RESIZE_WITH_CROP_OR_PAD_H
