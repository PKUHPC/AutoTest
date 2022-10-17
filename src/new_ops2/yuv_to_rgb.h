#ifndef YUV_TO_RGB_H
#define YUV_TO_RGB_H

#include "src/core/tensor.h"

/**
 * @brief Convert yuv image to rgb image.
 *
 * @param input The input tensor.
 * @param output The output tensor pointer.
 * @return Status The Status enum indicates whether the routine is OK.
 */
AITISA_API_PUBLIC Status aitisa_yuv_to_rgb(const Tensor input, Tensor *output);

#endif // YUV_TO_RGB_H
