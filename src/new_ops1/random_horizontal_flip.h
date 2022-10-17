#ifndef RANDOM_HORIZONTAL_FLIP_H
#define RANDOM_HORIZONTAL_FLIP_H

#include "src/core/tensor.h"

/**
 * @brief Randomly flip the input image horizontally.
 * @details Randomly flip the input image horizontally with probability prob for each channel and batch.
 *
 * @param input Input image.
 * @param prob Probability of flipping.
 * @param seed Random seed.
 * @param output The output image pointer.
 * @return Status The Status enum indicates whether the routine is OK.
 */

AITISA_API_PUBLIC Status aitisa_random_horizontal_flip(const Tensor input,
                                                       const float prob,
                                                       const int seed,
                                                       Tensor *output);

#endif