#ifndef ROT90_H
#define ROT90_H

#include "src/core/tensor.h"

/**
 * @brief Rotate the input image counterclockwise several times.
 * @details Rotate the input image counterclockwise several times, the parameter k is the specified number of rotations.
 *
 * @param input Input image.
 * @param k The number of times the image is rotated.
 * @param output The output image pointer.
 * @return Status The Status enum indicates whether the routine is OK.
 */

AITISA_API_PUBLIC Status aitisa_rot90(const Tensor input, const int k, Tensor *output);

#endif