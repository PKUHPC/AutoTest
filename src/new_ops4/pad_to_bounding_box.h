#ifndef PAD_TO_BOUNDING_BOX_H
#define PAD_TO_BOUNDING_BOX_H

#include "src/core/tensor.h"

/**
 * @brief Pad the boundary of the input image to the given range.
 *
 * @param input The input tensor.
 * @param top The width of the top boundary.
 * @param bot The width of the bottom boundary.
 * @param left The width of the left boundary.
 * @param right The width of the right boundary.
 * @param output The output tensor pointer.
 * @return Status The Status enum indicates whether the routine is OK.
 */
AITISA_API_PUBLIC Status aitisa_pad_to_bounding_box(const Tensor input, int top, int bot, int left, int right, Tensor *output);

#endif // PAD_TO_BOUNDING_BOX_H
